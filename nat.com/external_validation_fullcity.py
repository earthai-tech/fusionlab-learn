#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
external_validation_fullcity.py

Build a full-city union input NPZ from Stage-1 split artifacts,
export a full-city physics payload from the trained Stage-2 model,
and compute external borehole / pumping validation metrics.

Designed for GeoPrior-v3 / NATCOM workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from geoprior.compat import load_inference_model
from geoprior.models import (
    GeoPriorSubsNet,
    PoroElasticSubsNet,
    make_weighted_pinball,
)
from geoprior.params import (
    FixedGammaW,
    FixedHRef,
    LearnableKappa,
    LearnableMV,
)
from geoprior.utils import make_tf_dataset


ArrayDict = Dict[str, np.ndarray]


# ---------------------------------------------------------------------
# Generic I/O helpers
# ---------------------------------------------------------------------
def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz_dict(path: str) -> ArrayDict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_npz_dict(path: str, data: ArrayDict) -> None:
    outdir = os.path.dirname(os.path.abspath(path))
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(path, **data)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_readable_file(path: Optional[str]) -> bool:
    if not isinstance(path, str) or not path:
        return False
    try:
        return os.path.isfile(path) and os.access(path, os.R_OK)
    except OSError:
        return False


# ---------------------------------------------------------------------
# Path rebasing for stale Stage-1 manifests
# ---------------------------------------------------------------------
def split_after_artifacts(path: str) -> Optional[str]:
    norm = path.replace("/", "\\")
    key = "\\artifacts\\"
    low = norm.lower()
    i = low.find(key)
    if i < 0:
        return None
    return norm[i + len(key):].replace("\\", os.sep)


def resolve_stage1_artifact(
    stage1_manifest_path: str,
    recorded_path: Optional[str],
    explicit: Optional[str] = None,
) -> Optional[str]:
    if explicit and is_readable_file(explicit):
        return explicit

    if is_readable_file(recorded_path):
        return recorded_path

    if not isinstance(recorded_path, str):
        return None

    stage1_dir = os.path.dirname(os.path.abspath(stage1_manifest_path))
    artifacts_dir = os.path.join(stage1_dir, "artifacts")

    cands: List[str] = []
    rel = split_after_artifacts(recorded_path)
    if rel:
        cands.append(os.path.join(artifacts_dir, rel))

    cands.append(os.path.join(stage1_dir, os.path.basename(recorded_path)))
    cands.append(os.path.join(artifacts_dir, os.path.basename(recorded_path)))

    for cand in cands:
        if is_readable_file(cand):
            return cand

    return None


def resolve_stage1_inputs_paths(
    stage1_manifest_path: str,
    train_inputs: Optional[str],
    val_inputs: Optional[str],
    test_inputs: Optional[str],
    coord_scaler: Optional[str],
) -> Tuple[str, str, Optional[str], Optional[str], dict]:
    manifest = read_json(stage1_manifest_path)
    npz = (manifest.get("artifacts") or {}).get("numpy") or {}
    enc = (manifest.get("artifacts") or {}).get("encoders") or {}

    train_path = resolve_stage1_artifact(
        stage1_manifest_path,
        npz.get("train_inputs_npz"),
        explicit=train_inputs,
    )
    val_path = resolve_stage1_artifact(
        stage1_manifest_path,
        npz.get("val_inputs_npz"),
        explicit=val_inputs,
    )
    test_path = resolve_stage1_artifact(
        stage1_manifest_path,
        npz.get("test_inputs_npz"),
        explicit=test_inputs,
    )
    coord_path = resolve_stage1_artifact(
        stage1_manifest_path,
        enc.get("coord_scaler"),
        explicit=coord_scaler,
    )

    if not train_path:
        raise FileNotFoundError(
            "Could not resolve Stage-1 train_inputs_npz. "
            "Pass --train-inputs explicitly."
        )
    if not val_path:
        raise FileNotFoundError(
            "Could not resolve Stage-1 val_inputs_npz. "
            "Pass --val-inputs explicitly."
        )

    return train_path, val_path, test_path, coord_path, manifest


# ---------------------------------------------------------------------
# Full-city union NPZ
# ---------------------------------------------------------------------
def make_full_inputs_npz(
    input_paths: Sequence[str],
    out_npz: str,
) -> str:
    parts = [load_npz_dict(p) for p in input_paths if p]
    if not parts:
        raise ValueError("No input NPZs were provided.")

    key_sets = [set(d.keys()) for d in parts]
    first_keys = key_sets[0]
    for ks in key_sets[1:]:
        if ks != first_keys:
            raise KeyError(
                "Split NPZ keys are not aligned: "
                f"{[sorted(ks) for ks in key_sets]}"
            )

    full: ArrayDict = {}
    for k in sorted(first_keys):
        full[k] = np.concatenate([d[k] for d in parts], axis=0)

    save_npz_dict(out_npz, full)
    return out_npz


# ---------------------------------------------------------------------
# Stage-2 model resolution / loading
# ---------------------------------------------------------------------
def resolve_run_dir(
    stage2_run_dir: Optional[str],
    stage2_manifest_path: Optional[str],
) -> Optional[str]:
    if stage2_run_dir and os.path.isdir(stage2_run_dir):
        return stage2_run_dir

    if stage2_manifest_path and is_readable_file(stage2_manifest_path):
        m = read_json(stage2_manifest_path)
        run_dir = ((m.get("paths") or {}).get("run_dir"))
        if isinstance(run_dir, str) and os.path.isdir(run_dir):
            return run_dir

        man_dir = os.path.dirname(os.path.abspath(stage2_manifest_path))
        if os.path.isdir(man_dir):
            return man_dir

    return None


def _search_run_dir_for_model_files(run_dir: str) -> Dict[str, Optional[str]]:
    out = {
        "final_keras": None,
        "best_keras": None,
        "weights_h5": None,
        "model_init_manifest": None,
    }
    if not os.path.isdir(run_dir):
        return out

    for name in os.listdir(run_dir):
        low = name.lower()
        path = os.path.join(run_dir, name)

        if low.endswith("_final.keras") or (
            low.endswith(".keras") and "final" in low
        ):
            out["final_keras"] = out["final_keras"] or path
        elif low.endswith("_best.keras") or (
            low.endswith(".keras") and "best" in low
        ):
            out["best_keras"] = out["best_keras"] or path
        elif low.endswith(".weights.h5"):
            out["weights_h5"] = out["weights_h5"] or path
        elif low == "model_init_manifest.json":
            out["model_init_manifest"] = (
                out["model_init_manifest"] or path
            )

    return out


def resolve_stage2_bundle(
    model_path: Optional[str],
    stage2_manifest_path: Optional[str],
    stage2_run_dir: Optional[str],
) -> Dict[str, Optional[str]]:
    out = {
        "model_path": None,
        "weights_path": None,
        "model_init_manifest": None,
        "run_dir": None,
    }

    if model_path and is_readable_file(model_path):
        out["model_path"] = model_path

    run_dir = resolve_run_dir(stage2_run_dir, stage2_manifest_path)
    out["run_dir"] = run_dir

    if stage2_manifest_path and is_readable_file(stage2_manifest_path):
        m = read_json(stage2_manifest_path)
        paths = m.get("paths") or {}
        man_dir = os.path.dirname(os.path.abspath(stage2_manifest_path))

        for key in (
            "final_keras",
            "best_keras",
            "best_weights",
            "weights_h5",
            "model_init_manifest",
        ):
            p = paths.get(key)
            if is_readable_file(p):
                if key in ("final_keras", "best_keras") and out["model_path"] is None:
                    out["model_path"] = p
                elif key in ("best_weights", "weights_h5"):
                    out["weights_path"] = p
                elif key == "model_init_manifest":
                    out["model_init_manifest"] = p
            elif isinstance(p, str):
                rebased = os.path.join(man_dir, os.path.basename(p))
                if is_readable_file(rebased):
                    if key in ("final_keras", "best_keras") and out["model_path"] is None:
                        out["model_path"] = rebased
                    elif key in ("best_weights", "weights_h5"):
                        out["weights_path"] = rebased
                    elif key == "model_init_manifest":
                        out["model_init_manifest"] = rebased

    if run_dir:
        found = _search_run_dir_for_model_files(run_dir)
        if out["model_path"] is None:
            out["model_path"] = found["final_keras"] or found["best_keras"]
        out["weights_path"] = out["weights_path"] or found["weights_h5"]
        out["model_init_manifest"] = (
            out["model_init_manifest"] or found["model_init_manifest"]
        )

    if out["model_path"] is None:
        raise FileNotFoundError(
            "Could not resolve a Stage-2 .keras model. "
            "Pass --model-path or --stage2-run-dir / --stage2-manifest."
        )

    return out


def build_custom_objects() -> Dict[str, object]:
    return {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "PoroElasticSubsNet": PoroElasticSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
    }


def _model_class_from_name(name: Optional[str]):
    name = (name or "").strip()
    if name == "PoroElasticSubsNet":
        return PoroElasticSubsNet
    return GeoPriorSubsNet


def _builder_from_init_manifest(model_init_manifest_path: str):
    init_m = read_json(model_init_manifest_path)
    dims = (init_m.get("dims") or {})
    cfg = (init_m.get("config") or {})
    gp = (cfg.get("geoprior") or {})
    model_cls = _model_class_from_name(init_m.get("model_class"))

    def builder(_manifest: dict):
        return model_cls(
            static_input_dim=int(dims["static_input_dim"]),
            dynamic_input_dim=int(dims["dynamic_input_dim"]),
            future_input_dim=int(dims["future_input_dim"]),
            output_subsidence_dim=int(dims["output_subsidence_dim"]),
            output_gwl_dim=int(dims["output_gwl_dim"]),
            forecast_horizon=int(dims["forecast_horizon"]),
            quantiles=cfg.get("quantiles"),
            pde_mode=cfg.get("pde_mode", "off"),
            mode=cfg.get("mode", "tft_like"),
            time_units=cfg.get("time_units", "year"),
            embed_dim=int(cfg.get("embed_dim", 32)),
            hidden_units=int(cfg.get("hidden_units", 64)),
            lstm_units=int(cfg.get("lstm_units", 64)),
            attention_units=int(cfg.get("attention_units", 64)),
            num_heads=int(cfg.get("num_heads", 2)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
            memory_size=int(cfg.get("memory_size", 50)),
            scales=list(cfg.get("scales") or [1, 2]),
            use_residuals=bool(cfg.get("use_residuals", True)),
            use_batch_norm=bool(cfg.get("use_batch_norm", False)),
            use_vsn=bool(cfg.get("use_vsn", True)),
            vsn_units=int(cfg.get("vsn_units", 32))
            if cfg.get("vsn_units") is not None
            else None,
            mv=LearnableMV(initial_value=float(gp.get("init_mv", 1e-7))),
            kappa=LearnableKappa(initial_value=float(gp.get("init_kappa", 1.0))),
            gamma_w=FixedGammaW(value=float(gp.get("gamma_w", 9810.0))),
            h_ref=FixedHRef(
                value=float(gp.get("h_ref_value", 0.0)),
                mode=gp.get("h_ref_mode"),
            ),
            kappa_mode=gp.get("kappa_mode", "bar"),
            use_effective_h=bool(gp.get("use_effective_h", True)),
            hd_factor=float(gp.get("hd_factor", 0.6)),
            offset_mode=gp.get("offset_mode", "mul"),
            scaling_kwargs=dict(cfg.get("scaling_kwargs") or {}),
            verbose=0,
        )

    return builder


def load_inference_bundle(
    bundle: Dict[str, Optional[str]],
    build_inputs: dict,
):
    custom_objects = build_custom_objects()
    model_path = bundle["model_path"]
    weights_path = bundle["weights_path"]
    model_init_manifest = bundle["model_init_manifest"]

    # Fast path
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
        print(f"[OK] Loaded .keras model -> {model_path}")
        return model
    except Exception as e:
        print(f"[Warn] Direct .keras load failed: {e}")

    # Bundle fallback
    if weights_path and model_init_manifest:
        builder = _builder_from_init_manifest(model_init_manifest)
        model = load_inference_model(
            keras_path=model_path,
            weights_path=weights_path,
            manifest_path=model_init_manifest,
            custom_objects=custom_objects,
            compile=False,
            builder=builder,
            build_inputs=build_inputs,
            prefer_full_model=True,
            log_fn=print,
            use_in_memory_model=False,
        )
        print("[OK] Loaded inference model via compat bundle loader.")
        return model

    raise RuntimeError(
        "Could not load inference bundle. "
        "Provide a readable .keras model, or a valid bundle "
        "(weights + model_init_manifest)."
    )


# ---------------------------------------------------------------------
# Config extraction for make_tf_dataset()
# ---------------------------------------------------------------------
def get_dataset_mode(stage1_manifest: dict) -> str:
    cfg = stage1_manifest.get("config") or {}
    nested_model = cfg.get("model") or {}
    return (
        cfg.get("MODE")
        or cfg.get("mode")
        or nested_model.get("mode")
        or "tft_like"
    )


def get_forecast_horizon(stage1_manifest: dict, fallback_h: int) -> int:
    cfg = stage1_manifest.get("config") or {}
    nested_model = cfg.get("model") or {}
    val = (
        cfg.get("FORECAST_HORIZON_YEARS")
        or cfg.get("forecast_horizon_years")
        or nested_model.get("forecast_horizon_years")
        or fallback_h
    )
    return int(val)


def get_feature_names(stage1_manifest: dict) -> Tuple[List[str], List[str]]:
    cfg = stage1_manifest.get("config") or {}
    features = cfg.get("features") or {}
    dyn = list(features.get("dynamic") or [])
    fut = list(features.get("future") or [])
    return dyn, fut


# ---------------------------------------------------------------------
# Full-city physics payload export
# ---------------------------------------------------------------------
def export_fullcity_payload(
    stage1_manifest: dict,
    full_inputs_npz: str,
    bundle: Dict[str, Optional[str]],
    out_payload: str,
    batch_size: int,
) -> str:
    X_full = load_npz_dict(full_inputs_npz)
    n = int(X_full["coords"].shape[0])
    H = int(X_full["coords"].shape[1])

    dyn_names, fut_names = get_feature_names(stage1_manifest)
    mode = get_dataset_mode(stage1_manifest)
    forecast_h = get_forecast_horizon(stage1_manifest, fallback_h=H)

    y_dummy = {
        "subs_pred": np.zeros((n, H, 1), dtype=np.float32),
        "gwl_pred": np.zeros((n, H, 1), dtype=np.float32),
    }

    ds_full = make_tf_dataset(
        X_full,
        y_dummy,
        batch_size=int(batch_size),
        shuffle=False,
        mode=mode,
        forecast_horizon=forecast_h,
        check_npz_finite=True,
        check_finite=True,
        dynamic_feature_names=dyn_names,
        future_feature_names=fut_names,
    )

    build_inputs = None
    for xb, _ in ds_full.take(1):
        build_inputs = xb
        break
    if build_inputs is None:
        raise RuntimeError("Could not get a build batch from ds_full.")

    model = load_inference_bundle(bundle, build_inputs)

    payload = model.export_physics_payload(
        ds_full,
        max_batches=None,
        save_path=out_payload,
        format="npz",
        overwrite=True,
        metadata={
            "split": "full_city_union",
            "source_inputs_npz": os.path.abspath(full_inputs_npz),
        },
    )

    if not os.path.isfile(out_payload):
        raise RuntimeError(
            "Payload export reported success but file was not created."
        )

    n_payload = len(np.asarray(payload["K"]).reshape(-1))
    print(
        f"[OK] Exported full-city payload -> {out_payload} "
        f"(rows={n_payload})"
    )
    return out_payload


# ---------------------------------------------------------------------
# Coordinate / matching helpers
# ---------------------------------------------------------------------
def inverse_txy(
    coords_bh3: np.ndarray,
    coord_scaler_path: Optional[str],
) -> np.ndarray:
    first = np.asarray(coords_bh3[:, 0, :], dtype=float)
    if not coord_scaler_path:
        return first
    scaler = joblib.load(coord_scaler_path)
    return scaler.inverse_transform(first)


def reduce_horizon(
    arr: np.ndarray,
    n_seq: int,
    horizon: int,
    how: str,
) -> np.ndarray:
    x = np.asarray(arr, dtype=float)

    if x.size == n_seq:
        x = x.reshape(n_seq, 1)
    elif x.size == n_seq * horizon:
        x = x.reshape(n_seq, horizon)
    else:
        raise ValueError(
            "Cannot align array with sequences. "
            f"size={x.size}, n_seq={n_seq}, horizon={horizon}"
        )

    how = str(how).strip().lower()
    if how == "first":
        return x[:, 0]
    if how == "mean":
        return np.mean(x, axis=1)
    if how == "median":
        return np.median(x, axis=1)

    raise ValueError(
        "horizon reducer must be one of {'first','mean','median'}"
    )


def build_pixel_table(
    inputs_npz: str,
    payload_npz: str,
    coord_scaler_path: Optional[str],
    horizon_reducer: str,
    site_reducer: str,
) -> pd.DataFrame:
    X = load_npz_dict(inputs_npz)
    payload = load_npz_dict(payload_npz)

    coords = np.asarray(X["coords"], dtype=float)
    n_seq, horizon, _ = coords.shape

    txy = inverse_txy(coords, coord_scaler_path)
    # coord order from Stage-1 is [t, x, y]
    x = txy[:, 1]
    y = txy[:, 2]

    h_eff = reduce_horizon(
        X["H_field"],
        n_seq=n_seq,
        horizon=horizon,
        how=horizon_reducer,
    )
    k_vals = reduce_horizon(
        payload["K"],
        n_seq=n_seq,
        horizon=horizon,
        how=horizon_reducer,
    )
    hd_vals = reduce_horizon(
        payload["Hd"],
        n_seq=n_seq,
        horizon=horizon,
        how=horizon_reducer,
    )

    h_payload = None
    if "H" in payload:
        h_payload = reduce_horizon(
            payload["H"],
            n_seq=n_seq,
            horizon=horizon,
            how=horizon_reducer,
        )

    df = pd.DataFrame(
        {
            "seq_idx": np.arange(n_seq, dtype=int),
            "x": x,
            "y": y,
            "H_eff_input_m": h_eff,
            "K_mps": k_vals,
            "Hd_m": hd_vals,
        }
    )
    if h_payload is not None:
        df["H_payload_m"] = h_payload

    num_cols = ["H_eff_input_m", "K_mps", "Hd_m"]
    if h_payload is not None:
        num_cols.append("H_payload_m")

    agg_fun = "mean" if str(site_reducer).lower() == "mean" else "median"
    pix = (
        df.groupby(["x", "y"], as_index=False)[num_cols]
        .agg(agg_fun)
        .reset_index(drop=True)
    )
    pix["pixel_idx"] = np.arange(len(pix), dtype=int)
    return pix


def nearest_match(
    pixels: pd.DataFrame,
    sx: float,
    sy: float,
    allow_swapped_xy: bool = True,
) -> Tuple[str, float, float, float, int]:
    px = pixels["x"].to_numpy(dtype=float)
    py = pixels["y"].to_numpy(dtype=float)

    d_dir = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
    i_dir = int(np.argmin(d_dir))

    if not allow_swapped_xy:
        row = pixels.iloc[i_dir]
        return (
            "direct_xy",
            float(row["x"]),
            float(row["y"]),
            float(d_dir[i_dir]),
            int(row["pixel_idx"]),
        )

    d_swp = np.sqrt((px - sy) ** 2 + (py - sx) ** 2)
    i_swp = int(np.argmin(d_swp))

    if float(d_swp[i_swp]) < float(d_dir[i_dir]):
        idx = i_swp
        mode = "swapped_xy"
        dist = float(d_swp[idx])
    else:
        idx = i_dir
        mode = "direct_xy"
        dist = float(d_dir[idx])

    row = pixels.iloc[idx]
    return (
        mode,
        float(row["x"]),
        float(row["y"]),
        dist,
        int(row["pixel_idx"]),
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size != y.size or x.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")

    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()

    xs = xr - xr.mean()
    ys = yr - yr.mean()
    den = math.sqrt(float(np.sum(xs * xs) * np.sum(ys * ys)))
    if den <= 0.0:
        return float("nan")
    return float(np.sum(xs * ys) / den)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return float(np.mean(np.abs(x[mask] - y[mask])))


def median_bias(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    return float(np.median(pred[mask] - obs[mask]))


def validate_site_matches(
    site_df: pd.DataFrame,
    max_distance_m: float,
    min_unique_pixels: int,
) -> None:
    n_unique = int(site_df["pixel_idx"].nunique())
    max_dist = float(site_df["match_distance_m"].max())

    if n_unique < int(min_unique_pixels):
        raise RuntimeError(
            "Too few unique matched pixels for site validation "
            f"(unique={n_unique}). This usually means CRS/order mismatch."
        )

    if max_dist > float(max_distance_m):
        raise RuntimeError(
            "Site-to-pixel matching distance is too large "
            f"(max={max_dist:.1f} m > {float(max_distance_m):.1f} m)."
        )


# ---------------------------------------------------------------------
# External validation computation
# ---------------------------------------------------------------------
def compute_external_metrics(
    validation_csv: str,
    full_inputs_npz: str,
    full_payload_npz: str,
    coord_scaler_path: Optional[str],
    outdir: str,
    x_col: str,
    y_col: str,
    productivity_col: str,
    horizon_reducer: str,
    site_reducer: str,
    max_match_distance_m: float,
    min_unique_pixels: int,
    allow_swapped_xy: bool,
):
    ensure_dir(outdir)

    pixels = build_pixel_table(
        full_inputs_npz,
        full_payload_npz,
        coord_scaler_path,
        horizon_reducer,
        site_reducer,
    )

    sites = pd.read_csv(validation_csv)
    if x_col not in sites.columns:
        raise KeyError(f"x column {x_col!r} not found in validation CSV.")
    if y_col not in sites.columns:
        raise KeyError(f"y column {y_col!r} not found in validation CSV.")
    if productivity_col not in sites.columns:
        raise KeyError(
            f"productivity column {productivity_col!r} "
            "not found in validation CSV."
        )

    rows = []
    for _, rec in sites.iterrows():
        mode, px, py, dist, pidx = nearest_match(
            pixels,
            sx=float(rec[x_col]),
            sy=float(rec[y_col]),
            allow_swapped_xy=allow_swapped_xy,
        )
        pix = pixels.loc[pixels["pixel_idx"] == pidx].iloc[0]

        out = dict(rec)
        out.update(
            {
                "match_mode": mode,
                "matched_pixel_x": px,
                "matched_pixel_y": py,
                "match_distance_m": dist,
                "pixel_idx": pidx,
                "model_H_eff_m": float(pix["H_eff_input_m"]),
                "model_K_mps": float(pix["K_mps"]),
                "model_Hd_m": float(pix["Hd_m"]),
            }
        )
        if "H_payload_m" in pix.index:
            out["payload_H_m"] = float(pix["H_payload_m"])
        rows.append(out)

    site_df = pd.DataFrame(rows)
    validate_site_matches(
        site_df,
        max_match_distance_m,
        min_unique_pixels,
    )

    obs_h = site_df["approx_compressible_thickness_m"].to_numpy(dtype=float)
    mod_h = site_df["model_H_eff_m"].to_numpy(dtype=float)
    prod = site_df[productivity_col].to_numpy(dtype=float)
    mod_k = site_df["model_K_mps"].to_numpy(dtype=float)

    metrics = {
        "n_sites": int(len(site_df)),
        "borehole_vs_H_eff": {
            "spearman_rho": spearman_rho(obs_h, mod_h),
            "mae_m": mae(obs_h, mod_h),
            "median_bias_m": median_bias(obs_h, mod_h),
        },
        "pumping_vs_K": {
            "productivity_column": productivity_col,
            "spearman_rho": spearman_rho(prod, mod_k),
        },
        "reducers": {
            "horizon_reducer": horizon_reducer,
            "site_reducer": site_reducer,
        },
        "match_summary": {
            "unique_pixels": int(site_df["pixel_idx"].nunique()),
            "min_distance_m": float(site_df["match_distance_m"].min()),
            "median_distance_m": float(site_df["match_distance_m"].median()),
            "max_distance_m": float(site_df["match_distance_m"].max()),
        },
        "files": {
            "validation_csv": os.path.abspath(validation_csv),
            "full_inputs_npz": os.path.abspath(full_inputs_npz),
            "full_payload_npz": os.path.abspath(full_payload_npz),
            "coord_scaler": (
                os.path.abspath(coord_scaler_path)
                if coord_scaler_path
                else None
            ),
        },
        "site_columns": {
            "x_col": x_col,
            "y_col": y_col,
        },
    }

    site_csv = os.path.join(
        outdir,
        "site_level_external_validation_fullcity.csv",
    )
    metrics_json = os.path.join(
        outdir,
        "external_validation_metrics_fullcity.json",
    )

    site_df.to_csv(site_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return site_df, metrics


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build full-city union inputs, export full-city "
            "physics payload, and compute external validation metrics."
        )
    )
    p.add_argument("--stage1-manifest", required=True)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--train-inputs", default=None)
    p.add_argument("--val-inputs", default=None)
    p.add_argument("--test-inputs", default=None)
    p.add_argument("--coord-scaler", default=None)

    p.add_argument("--model-path", default=None)
    p.add_argument("--stage2-manifest", default=None)
    p.add_argument("--stage2-run-dir", default=None)

    p.add_argument("--full-inputs-npz", default=None)
    p.add_argument("--full-payload-npz", default=None)
    p.add_argument("--batch-size", type=int, default=256)

    p.add_argument("--x-col", default="x")
    p.add_argument("--y-col", default="y")
    p.add_argument(
        "--productivity-col",
        default="step3_specific_capacity_Lps_per_m",
    )
    p.add_argument(
        "--horizon-reducer",
        choices=["first", "mean", "median"],
        default="mean",
    )
    p.add_argument(
        "--site-reducer",
        choices=["mean", "median"],
        default="median",
    )
    p.add_argument(
        "--max-match-distance-m",
        type=float,
        default=50000.0,
    )
    p.add_argument(
        "--min-unique-pixels",
        type=int,
        default=3,
    )
    p.add_argument(
        "--no-swapped-xy",
        action="store_true",
        help="Disable direct-vs-swapped x/y auto matching.",
    )
    p.add_argument("--skip-export", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.outdir)

    train_inputs, val_inputs, test_inputs, coord_scaler, stage1_manifest = (
        resolve_stage1_inputs_paths(
            args.stage1_manifest,
            train_inputs=args.train_inputs,
            val_inputs=args.val_inputs,
            test_inputs=args.test_inputs,
            coord_scaler=args.coord_scaler,
        )
    )

    full_inputs_npz = (
        args.full_inputs_npz
        or os.path.join(args.outdir, "full_inputs.npz")
    )
    if not is_readable_file(full_inputs_npz):
        parts = [train_inputs, val_inputs]
        if test_inputs:
            parts.append(test_inputs)
        full_inputs_npz = make_full_inputs_npz(parts, full_inputs_npz)
        print(f"[OK] Saved full-city inputs -> {full_inputs_npz}")
    else:
        print(f"[OK] Reusing full-city inputs -> {full_inputs_npz}")

    full_payload_npz = (
        args.full_payload_npz
        or os.path.join(args.outdir, "physics_payload_fullcity.npz")
    )
    if not args.skip_export:
        bundle = resolve_stage2_bundle(
            args.model_path,
            args.stage2_manifest,
            args.stage2_run_dir,
        )
        print(f"[OK] Using Stage-2 bundle -> {bundle}")
        export_fullcity_payload(
            stage1_manifest,
            full_inputs_npz,
            bundle,
            full_payload_npz,
            args.batch_size,
        )
    else:
        if not is_readable_file(full_payload_npz):
            raise FileNotFoundError(
                "--skip-export was set but full payload file does not exist."
            )
        print(f"[OK] Reusing full-city payload -> {full_payload_npz}")

    site_df, metrics = compute_external_metrics(
        validation_csv=args.validation_csv,
        full_inputs_npz=full_inputs_npz,
        full_payload_npz=full_payload_npz,
        coord_scaler_path=coord_scaler,
        outdir=args.outdir,
        x_col=args.x_col,
        y_col=args.y_col,
        productivity_col=args.productivity_col,
        horizon_reducer=args.horizon_reducer,
        site_reducer=args.site_reducer,
        max_match_distance_m=args.max_match_distance_m,
        min_unique_pixels=args.min_unique_pixels,
        allow_swapped_xy=(not args.no_swapped_xy),
    )

    print("\nPer-site summary\n")
    print(site_df.to_string(index=False))
    print("\nHeadline metrics\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()