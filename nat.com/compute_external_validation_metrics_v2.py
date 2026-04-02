#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


ArrayDict = Dict[str, np.ndarray]


@dataclass
class MatchResult:
    well_id: str
    match_mode: str
    site_x: float
    site_y: float
    pixel_x: float
    pixel_y: float
    distance_m: float
    pixel_idx: int


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz_dict(path: str) -> ArrayDict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_existing(*paths: Optional[str]) -> Optional[str]:
    for path in paths:
        if isinstance(path, str) and os.path.exists(path):
            return path
    return None


def resolve_inputs_npz(
    stage1_manifest: dict,
    split: str,
    override: Optional[str],
) -> str:
    if override and os.path.exists(override):
        return override

    npz_art = (
        (stage1_manifest.get("artifacts") or {})
        .get("numpy", {})
    )
    path = npz_art.get(f"{split}_inputs_npz")
    if isinstance(path, str) and os.path.exists(path):
        return path

    raise FileNotFoundError(
        f"Could not resolve {split}_inputs_npz. "
        "Pass --inputs-npz explicitly."
    )


def resolve_coord_scaler(
    stage1_manifest: dict,
    override: Optional[str],
):
    if override and os.path.exists(override):
        return joblib.load(override)

    enc = (
        (stage1_manifest.get("artifacts") or {})
        .get("encoders", {})
    )
    path = enc.get("coord_scaler")
    if isinstance(path, str) and os.path.exists(path):
        return joblib.load(path)
    return None


def resolve_payload_path(
    override: Optional[str],
    stage2_manifest: Optional[dict],
) -> str:
    if override and os.path.exists(override):
        return override

    if not stage2_manifest:
        raise FileNotFoundError(
            "No payload path was found. Pass --physics-payload."
        )

    paths = stage2_manifest.get("paths", {}) or {}
    run_dir = paths.get("run_dir")
    if isinstance(run_dir, str) and os.path.isdir(run_dir):
        cands = []
        for name in os.listdir(run_dir):
            low = name.lower()
            if low.endswith(".npz") and "phys_payload" in low:
                cands.append(os.path.join(run_dir, name))
        cands.sort()
        if cands:
            return cands[0]

    raise FileNotFoundError(
        "Could not resolve a physics payload. "
        "Pass --physics-payload explicitly."
    )


def inverse_txy(
    coords_bh3: np.ndarray,
    coord_scaler,
) -> np.ndarray:
    first = np.asarray(coords_bh3[:, 0, :], dtype=float)
    if coord_scaler is None:
        return first
    return coord_scaler.inverse_transform(first)


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
        "horizon reducer must be one of "
        "{'first','mean','median'}"
    )


def build_pixel_table(
    stage1_manifest: dict,
    inputs_npz: str,
    payload_npz: str,
    coord_scaler,
    horizon_reducer: str,
    site_reducer: str,
) -> pd.DataFrame:
    x_np = load_npz_dict(inputs_npz)
    payload = load_npz_dict(payload_npz)

    coords = np.asarray(x_np["coords"], dtype=float)
    n_seq, horizon, _ = coords.shape

    txy = inverse_txy(coords, coord_scaler)
    x = txy[:, 1]
    y = txy[:, 2]

    h_eff = reduce_horizon(
        x_np["H_field"],
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

    group_cols = ["x", "y"]
    num_cols = [
        "H_eff_input_m",
        "K_mps",
        "Hd_m",
    ]
    if h_payload is not None:
        num_cols.append("H_payload_m")

    site_reducer = str(site_reducer).strip().lower()
    if site_reducer not in {"mean", "median"}:
        raise ValueError(
            "site reducer must be one of {'mean','median'}"
        )

    agg_fun = np.mean if site_reducer == "mean" else np.median

    pix = (
        df.groupby(group_cols, as_index=False)[num_cols]
        .agg(agg_fun)
        .reset_index(drop=True)
    )
    pix["pixel_idx"] = np.arange(len(pix), dtype=int)
    return pix


def nearest_match(
    pixels: pd.DataFrame,
    sx: float,
    sy: float,
    well_id: str,
) -> MatchResult:
    px = pixels["x"].to_numpy(dtype=float)
    py = pixels["y"].to_numpy(dtype=float)

    d_dir = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
    d_swp = np.sqrt((px - sy) ** 2 + (py - sx) ** 2)

    i_dir = int(np.argmin(d_dir))
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
    return MatchResult(
        well_id=str(well_id),
        match_mode=mode,
        site_x=float(sx),
        site_y=float(sy),
        pixel_x=float(row["x"]),
        pixel_y=float(row["y"]),
        distance_m=dist,
        pixel_idx=int(row["pixel_idx"]),
    )

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

# def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
#     x = np.asarray(x, dtype=float)
#     y = np.asarray(y, dtype=float)
#     if x.size != y.size or x.size < 2:
#         return float("nan")

#     xr = pd.Series(x).rank(method="average").to_numpy()
#     yr = pd.Series(y).rank(method="average").to_numpy()

#     xs = xr - xr.mean()
#     ys = yr - yr.mean()
#     den = math.sqrt(
#         float(np.sum(xs * xs) * np.sum(ys * ys))
#     )
#     if den <= 0.0:
#         return float("nan")
#     return float(np.sum(xs * ys) / den)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean(np.abs(x - y)))


def median_bias(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.median(pred - obs))


def compute_metrics(
    validation_csv: str,
    stage1_manifest_path: str,
    outdir: str,
    split: str = "test",
    inputs_npz: Optional[str] = None,
    physics_payload: Optional[str] = None,
    coord_scaler: Optional[str] = None,
    stage2_manifest_path: Optional[str] = None,
    productivity_col: str = (
        "step3_specific_capacity_Lps_per_m"
    ),
    horizon_reducer: str = "mean",
    site_reducer: str = "median",
) -> Tuple[pd.DataFrame, dict]:
    ensure_dir(outdir)

    stage1 = read_json(stage1_manifest_path)
    stage2 = None
    if stage2_manifest_path:
        stage2 = read_json(stage2_manifest_path)

    inputs_npz_path = resolve_inputs_npz(
        stage1,
        split=split,
        override=inputs_npz,
    )
    payload_npz_path = resolve_payload_path(
        physics_payload,
        stage2_manifest=stage2,
    )
    coord_scaler_obj = resolve_coord_scaler(
        stage1,
        override=coord_scaler,
    )

    pixels = build_pixel_table(
        stage1_manifest=stage1,
        inputs_npz=inputs_npz_path,
        payload_npz=payload_npz_path,
        coord_scaler=coord_scaler_obj,
        horizon_reducer=horizon_reducer,
        site_reducer=site_reducer,
    )

    sites = pd.read_csv(validation_csv)
    rows = []
    for _, rec in sites.iterrows():
        m = nearest_match(
            pixels=pixels,
            sx=float(rec["x"]),
            sy=float(rec["y"]),
            well_id=str(rec["well_id"]),
        )
        pix = pixels.loc[
            pixels["pixel_idx"] == m.pixel_idx
        ].iloc[0]

        out = dict(rec)
        out.update(
            {
                "match_mode": m.match_mode,
                "matched_pixel_x": m.pixel_x,
                "matched_pixel_y": m.pixel_y,
                "match_distance_m": m.distance_m,
                "pixel_idx": m.pixel_idx,
                "model_H_eff_m": float(pix["H_eff_input_m"]),
                "model_K_mps": float(pix["K_mps"]),
                "model_Hd_m": float(pix["Hd_m"]),
            }
        )
        if "H_payload_m" in pix.index:
            out["payload_H_m"] = float(pix["H_payload_m"])
        rows.append(out)

    site_df = pd.DataFrame(rows)
    
    def validate_site_matches(
        site_df: pd.DataFrame,
        max_distance_m: float = 5000.0,
        min_unique_pixels: int = 3,
    ) -> None:
        n_unique = int(site_df["pixel_idx"].nunique())
        max_dist = float(site_df["match_distance_m"].max())
    
        if n_unique < min_unique_pixels:
            raise RuntimeError(
                "Too few unique matched pixels for site validation "
                f"(unique={n_unique}). This usually means CRS/order mismatch."
            )
    
        if max_dist > max_distance_m:
            raise RuntimeError(
                "Site-to-pixel matching distance is too large "
                f"(max={max_dist:.1f} m > {max_distance_m:.1f} m). "
                "Coordinates are likely not in the same CRS/grid as Stage-1."
            )

    validate_site_matches(site_df)
    
    obs_h = site_df[
        "approx_compressible_thickness_m"
    ].to_numpy(dtype=float)
    mod_h = site_df["model_H_eff_m"].to_numpy(dtype=float)

    prod = site_df[productivity_col].to_numpy(dtype=float)
    mod_k = site_df["model_K_mps"].to_numpy(dtype=float)

    metrics = {
        "n_sites": int(len(site_df)),
        "split_used": split,
        "borehole_vs_H_eff": {
            "spearman_rho": spearman_rho(obs_h, mod_h),
            "mae_m": mae(obs_h, mod_h),
            "median_bias_m": median_bias(obs_h, mod_h),
        },
        "pumping_vs_K": {
            "productivity_column": productivity_col,
            "spearman_rho": spearman_rho(prod, mod_k),
        },
        "files": {
            "validation_csv": os.path.abspath(validation_csv),
            "stage1_manifest": os.path.abspath(
                stage1_manifest_path
            ),
            "stage2_manifest": (
                os.path.abspath(stage2_manifest_path)
                if stage2_manifest_path
                else None
            ),
            "inputs_npz": os.path.abspath(inputs_npz_path),
            "physics_payload": os.path.abspath(
                payload_npz_path
            ),
        },
        "reducers": {
            "horizon_reducer": horizon_reducer,
            "site_reducer": site_reducer,
        },
    }

    site_csv = os.path.join(
        outdir,
        "site_level_external_validation.csv",
    )
    metrics_json = os.path.join(
        outdir,
        "external_validation_metrics.json",
    )

    site_df.to_csv(site_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return site_df, metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute borehole/pumping validation "
            "metrics from Stage-1 inputs plus a "
            "saved physics payload."
        )
    )
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--stage1-manifest", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--inputs-npz", default=None)
    p.add_argument("--physics-payload", default=None)
    p.add_argument("--coord-scaler", default=None)
    p.add_argument("--stage2-manifest", default=None)
    p.add_argument(
        "--productivity-col",
        default="step3_specific_capacity_Lps_per_m",
    )
    p.add_argument(
        "--horizon-reducer",
        default="mean",
        choices=["first", "mean", "median"],
    )
    p.add_argument(
        "--site-reducer",
        default="median",
        choices=["mean", "median"],
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    site_df, metrics = compute_metrics(
        validation_csv=args.validation_csv,
        stage1_manifest_path=args.stage1_manifest,
        outdir=args.outdir,
        split=args.split,
        inputs_npz=args.inputs_npz,
        physics_payload=args.physics_payload,
        coord_scaler=args.coord_scaler,
        stage2_manifest_path=args.stage2_manifest,
        productivity_col=args.productivity_col,
        horizon_reducer=args.horizon_reducer,
        site_reducer=args.site_reducer,
    )
    print("\nPer-site summary\n")
    print(site_df.to_string(index=False))
    print("\nHeadline metrics\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
