# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os
import json
import glob
import datetime as dt
import csv
from typing import Sequence, Optional, Dict, Any, Callable, Tuple, List

import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model

from ....._optdeps import with_progress
from .....registry.utils import _find_stage1_manifest, reproject_dynamic_scale
from .....utils.generic_utils import ensure_directory_exists
from .....utils.scale_metrics import inverse_scale_target
from .....utils.forecast_utils import format_and_forecast
# v3.2: prefer compat loader (Keras2/Keras3 + TF SavedModel)
from .....compat.keras import load_inference_model
# v3.2: geoprior model path may have moved
from .....nn.pinn.geoprior.models import GeoPriorSubsNet
from .....params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
from .....nn.losses import make_weighted_pinball
from .....nn.keras_metrics import _to_py, coverage80_fn, sharpness80_fn
from .....nn.calibration import (
    IntervalCalibrator,
    fit_interval_calibrator_on_val,
    apply_calibrator_to_subs,
)
from .....nn.pinn.op import extract_physical_parameters

# ---------------------------------------------------------------------
# v3.2 store integration (optional)
# ---------------------------------------------------------------------
from ..config.store import GeoConfigStore

# ---------------------------------------------------------------------
# Small compat helpers (v3.0 NAT vs v3.2 GeoPriorConfig)
# ---------------------------------------------------------------------
def _cfg_get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Return the first existing key among `keys` from cfg, else default.
    """
    for k in keys:
        if not k:
            continue
        if k in cfg:
            return cfg[k]
    return default


def _cfg_features(cfg: Dict[str, Any], kind: str) -> List[str]:
    """
    kind in {"static","dynamic","future_known"}.

    v3.0: cfg["features"][kind]
    v3.2: cfg["static_features"], cfg["dynamic_features"], cfg["future_known_features"]
    """
    feats = cfg.get("features")
    if isinstance(feats, dict) and kind in feats:
        return list(feats.get(kind) or [])
    if kind == "static":
        return list(_cfg_get(cfg, "static_features", default=[] ) or [])
    if kind == "dynamic":
        return list(_cfg_get(cfg, "dynamic_features", default=[] ) or [])
    if kind in {"future_known", "future"}:
        return list(_cfg_get(cfg, "future_known_features", default=[] ) or [])
    return []


def _cfg_subs_col(cfg: Dict[str, Any]) -> str:
    # v3.2: subs_col, v3.0: cols["subsidence"]
    subs = _cfg_get(cfg, "subs_col", default=None)
    if isinstance(subs, str) and subs.strip():
        return subs.strip()
    cols = cfg.get("cols") or {}
    if isinstance(cols, dict):
        v = cols.get("subsidence") or cols.get("subsidence_col")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "subsidence"


def _cfg_mode(cfg: Dict[str, Any]) -> str:
    return str(_cfg_get(cfg, "mode", "MODE", default="tft_like") or "tft_like")


def _cfg_horizon(cfg: Dict[str, Any], fallback: int) -> int:
    v = _cfg_get(cfg, "forecast_horizon_years", "FORECAST_HORIZON_YEARS", default=fallback)
    try:
        return int(v)
    except Exception:
        return int(fallback)


def _cfg_quantiles(cfg: Dict[str, Any], override: Optional[Sequence[float]] = None) -> List[float]:
    if override is not None:
        return [float(q) for q in override]
    q = _cfg_get(cfg, "quantiles", "QUANTILES", default=[0.1, 0.5, 0.9])
    try:
        return [float(x) for x in (q or [0.1, 0.5, 0.9])]
    except Exception:
        return [0.1, 0.5, 0.9]


def _manifest_run_dir(M: Dict[str, Any]) -> str:
    paths = M.get("paths") or {}
    if isinstance(paths, dict):
        for k in ("run_dir", "stage1_dir", "root_dir"):
            v = paths.get(k)
            if isinstance(v, str) and v:
                return v
    # last resort: infer from manifest location if present
    mpath = M.get("manifest_path")
    if isinstance(mpath, str) and os.path.isfile(mpath):
        return os.path.dirname(mpath)
    return ""


def _latest_model_under(run_dir: str, *, prefer_tuned: bool = True) -> Optional[str]:
    """
    v3.2: still prefer tuned models if present.
    """
    if not run_dir or not os.path.isdir(run_dir):
        return None

    patterns: List[str] = []
    if prefer_tuned:
        patterns.append(os.path.join(run_dir, "tuning", "**", "*.keras"))
    patterns.append(os.path.join(run_dir, "train_*", "*.keras"))
    patterns.append(os.path.join(run_dir, "**", "*.keras"))

    cands: List[Tuple[float, str]] = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            try:
                cands.append((os.path.getmtime(p), p))
            except Exception:
                pass

    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def _pick_npz(M: Dict[str, Any], which: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    npzs = (M.get("artifacts") or {}).get("numpy") or {}
    if which == "val":
        return (
            dict(np.load(npzs["val_inputs_npz"])),
            dict(np.load(npzs["val_targets_npz"])),
        )
    if which == "test":
        ti, tt = npzs.get("test_inputs_npz"), npzs.get("test_targets_npz")
        if not ti:
            return None, None
        return dict(np.load(ti)), (dict(np.load(tt)) if tt else None)
    raise ValueError(which)


def _ensure_shapes(x: Dict[str, Any], mode: str, horizon: int) -> Dict[str, Any]:
    out = dict(x)
    N = out["dynamic_features"].shape[0]

    if out.get("static_features") is None:
        out["static_features"] = np.zeros((N, 0), np.float32)

    if out.get("future_features") is None:
        t_future = out["dynamic_features"].shape[1] if mode == "tft_like" else horizon
        out["future_features"] = np.zeros((N, t_future, 0), np.float32)

    return out


def _map_targets(y: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not y:
        return {}
    if "subsidence" in y and "gwl" in y:
        return {"subs_pred": y["subsidence"], "gwl_pred": y["gwl"]}
    if "subs_pred" in y and "gwl_pred" in y:
        return dict(y)
    return {}


def _load_source_calibrator(source_run_dir: str) -> Optional[IntervalCalibrator]:
    pats = glob.glob(os.path.join(source_run_dir, "train_*", "interval_factors_80.npy"))
    if not pats:
        return None
    p = sorted(pats, key=os.path.getmtime, reverse=True)[0]
    cal = IntervalCalibrator(target=0.80)
    cal.factors_ = np.load(p).astype(np.float32)
    return cal


def _infer_source_input_dims(M_src: Dict[str, Any]) -> Tuple[int, int]:
    """
    Infer (static_input_dim, dynamic_input_dim) from manifest.
    """
    arts = M_src.get("artifacts") or {}
    seq = arts.get("sequences") or {}
    dims = seq.get("dims") or {}

    s_src = dims.get("static_input_dim")
    d_src = dims.get("dynamic_input_dim")

    shapes = arts.get("shapes") or {}
    tr_in = shapes.get("train_inputs") or {}

    if s_src is None:
        sf_shape = tr_in.get("static_features")
        if isinstance(sf_shape, (list, tuple)) and len(sf_shape) >= 2:
            s_src = sf_shape[-1]

    if d_src is None:
        df_shape = tr_in.get("dynamic_features")
        if isinstance(df_shape, (list, tuple)) and len(df_shape) >= 3:
            d_src = df_shape[-1]

    cfg = M_src.get("config") or {}
    if s_src is None:
        s_src = len(_cfg_features(cfg, "static") or [])
    if d_src is None:
        d_src = len(_cfg_features(cfg, "dynamic") or [])

    return int(s_src or 0), int(d_src or 0)


def _align_last_axis_by_name(
    arr: np.ndarray,
    names_src: Sequence[str],
    names_tgt: Sequence[str],
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Reorder/pad `arr` last axis to match `names_src`, using `names_tgt`
    as the meaning of existing columns.
    """
    if arr.ndim < 2:
        return arr

    names_src = list(names_src or [])
    names_tgt = list(names_tgt or [])

    if not names_src:
        # model expects no features
        new_shape = list(arr.shape)
        new_shape[-1] = 0
        return np.zeros(new_shape, dtype=np.float32)

    out_shape = list(arr.shape)
    out_shape[-1] = len(names_src)
    out = np.full(out_shape, fill_value, dtype=np.float32)

    name2idx = {n: i for i, n in enumerate(names_tgt)}
    for j, n in enumerate(names_src):
        i = name2idx.get(n)
        if i is None or i >= arr.shape[-1]:
            continue
        out[..., j] = arr[..., i]

    return out


def _align_inputs_to_source(
    X_tgt: Dict[str, Any],
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
    *,
    policy: str,
) -> Dict[str, Any]:
    """
    policy:
      - "strict"              : no reordering; dims must match.
      - "align_by_name_pad"   : reorder by name and pad missing with zeros.
    """
    policy = (policy or "align_by_name_pad").strip().lower()

    cfg_s = M_src.get("config") or {}
    cfg_t = M_tgt.get("config") or {}

    static_src = _cfg_features(cfg_s, "static")
    static_tgt = _cfg_features(cfg_t, "static")
    dyn_src = _cfg_features(cfg_s, "dynamic")
    dyn_tgt = _cfg_features(cfg_t, "dynamic")

    if policy == "align_by_name_pad":
        X_tgt["static_features"] = _align_last_axis_by_name(
            X_tgt.get("static_features", np.zeros((X_tgt["dynamic_features"].shape[0], 0), np.float32)),
            static_src,
            static_tgt,
            fill_value=0.0,
        )
        X_tgt["dynamic_features"] = _align_last_axis_by_name(
            X_tgt["dynamic_features"],
            dyn_src,
            dyn_tgt,
            fill_value=0.0,
        )
        return X_tgt

    # strict: keep as-is
    return X_tgt


def _load_model_v32(
    model_path: str,
    *,
    custom_objects: Dict[str, Any],
    endpoint: str = "serve",
    compile_model: bool = True,
):
    """
    v3.2-friendly model loader. Prefers fusionlab.compat.keras if present.
    """
    if load_inference_model is not None:
        # Your compat loader should already handle:
        # - .keras / .h5
        # - TF SavedModel dir (endpoint)
        return load_inference_model(
            model_path,
            custom_objects=custom_objects,
            endpoint=endpoint,
            compile=compile_model,
        )

    # Fallback (older environments)
    with custom_object_scope(custom_objects):
        return load_model(model_path, compile=compile_model)


# ---------------------------------------------------------------------
# v3.2 manifest loading (still Stage-1 based)
# ---------------------------------------------------------------------
def _load_manifest_for_city(
    city: str,
    results_root: str,
    model_name: str = "GeoPriorSubsNet",
    manual: Optional[str] = None,
) -> Dict[str, Any]:
    mpath = _find_stage1_manifest(
        manual=manual,
        base_dir=results_root,
        city_hint=city,
        model_hint=model_name,
        prefer="timestamp",
        required_keys=("model", "stage"),
        verbose=0,
    )
    with open(mpath, "r", encoding="utf-8") as f:
        M = json.load(f)
    # keep manifest path for last-resort run_dir inference
    M["manifest_path"] = mpath
    return M


# ---------------------------------------------------------------------
# Public GUI entry point (v3.2)
# ---------------------------------------------------------------------
def run_xfer_matrix(
    city_a: str,
    city_b: str,
    *,
    store: Optional[GeoConfigStore] = None,
    results_root: Optional[str] = None,
    splits: Sequence[str] = ("val", "test"),
    calib_modes: Sequence[str] = ("none", "source", "target"),
    rescale_to_source: bool = False,
    batch_size: int = 32,
    quantiles_override: Optional[Sequence[float]] = None,
    out_dir: Optional[str] = None,
    write_json: bool = True,
    write_csv: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    model_name: str = "GeoPriorSubsNet",
    # v3.2: extra GUI-exposed knobs
    prefer_tuned: bool = True,
    align_policy: str = "align_by_name_pad",  # or "strict"
    interval_target: float = 0.80,
    load_endpoint: str = "serve",
    export_physics_payload: bool = True,
    export_physical_parameters_csv: bool = True,
    write_eval_future_csv: bool = True,
    **kws,
) -> Dict[str, Any]:
    """
    v3.2 GUI transfer matrix runner (A->B and B->A).

    Store-driven defaults (if `store` is provided):
      - results_root: store.get("results_root")
      - xfer.* knobs (optional): xfer.splits, xfer.calib_modes, xfer.batch_size,
        xfer.rescale_to_source, xfer.align_policy, xfer.interval_target, ...
    """

    def log(msg: str) -> None:
        (logger or print)(msg)

    def should_stop() -> bool:
        return bool(stop_check and stop_check())

    def _progress(value: float, message: str) -> None:
        if progress_callback is None:
            return
        try:
            v = float(value)
        except Exception:
            v = 0.0
        v = max(0.0, min(1.0, v))
        try:
            progress_callback(v, message)
        except Exception:
            pass

    # ---------------- store-driven overrides ----------------
    if store is not None:
        try:
            results_root = results_root or store.get("results_root", None)
            splits = tuple(store.get("xfer.splits", splits))  # type: ignore[assignment]
            calib_modes = tuple(store.get("xfer.calib_modes", calib_modes))  # type: ignore[assignment]
            batch_size = int(store.get("xfer.batch_size", batch_size))
            rescale_to_source = bool(store.get("xfer.rescale_to_source", rescale_to_source))
            prefer_tuned = bool(store.get("xfer.prefer_tuned", prefer_tuned))
            align_policy = str(store.get("xfer.align_policy", align_policy))
            interval_target = float(store.get("xfer.interval_target", interval_target))
            load_endpoint = str(store.get("xfer.load_endpoint", load_endpoint))
            export_physics_payload = bool(store.get("xfer.export_physics_payload", export_physics_payload))
            export_physical_parameters_csv = bool(
                store.get("xfer.export_physical_parameters_csv", export_physical_parameters_csv)
            )
            write_eval_future_csv = bool(store.get("xfer.write_eval_future_csv", write_eval_future_csv))
        except Exception:
            # never crash just because store had unexpected types
            pass

    results_root = results_root or "results"

    # ---------------- load manifests ----------------
    _progress(0.03, "XFER: locating Stage-1 manifests")
    M_A = _load_manifest_for_city(city_a, results_root, model_name=model_name)
    M_B = _load_manifest_for_city(city_b, results_root, model_name=model_name)

    if out_dir is None:
        out_dir = os.path.join(
            results_root,
            "xfer",
            f"{city_a}_to_{city_b}",
            dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    ensure_directory_exists(out_dir)
    log(f"[XFER] Output directory: {out_dir}")
    _progress(0.08, "XFER: manifests loaded and output dir ready")

    # ---------------- run all directions/modes ----------------
    directions = [
        ("A_to_B", M_A, M_B),
        ("B_to_A", M_B, M_A),
    ]
    total_jobs = len(directions) * len(splits) * len(calib_modes)
    done_jobs = 0
    base = 0.10
    span = 0.78

    results: List[Dict[str, Any]] = []

    for tag, M_src, M_tgt in directions:
        for split in splits:
            for cm in calib_modes:
                if should_stop():
                    log("[XFER] Cancelled by user.")
                    _progress(base + span * (done_jobs / max(1, total_jobs)), "XFER: cancelled")
                    return {
                        "out_dir": out_dir,
                        "results": results,
                        "json_path": None,
                        "csv_path": None,
                    }

                frac = base + span * (done_jobs / max(1, total_jobs))
                done_jobs += 1
                _progress(frac, f"XFER: {tag} split={split} calib={cm} ({done_jobs}/{total_jobs})")

                log(f"[XFER] direction={tag}, split={split}, calib={cm} ...")
                r = run_one_direction(
                    M_src=M_src,
                    M_tgt=M_tgt,
                    split=split,
                    calib_mode=cm,
                    rescale_to_source=rescale_to_source,
                    batch_size=batch_size,
                    quantiles_override=quantiles_override,
                    out_dir=out_dir,
                    prefer_tuned=prefer_tuned,
                    align_policy=align_policy,
                    interval_target=interval_target,
                    load_endpoint=load_endpoint,
                    export_physics_payload=export_physics_payload,
                    export_physical_parameters_csv=export_physical_parameters_csv,
                    write_eval_future_csv=write_eval_future_csv,
                    logger=logger,
                    stop_check=stop_check,
                )
                if r is not None:
                    r["direction"] = tag
                    r["source_city"] = M_src.get("city") or city_a
                    r["target_city"] = M_tgt.get("city") or city_b
                    results.append(r)

    _progress(0.92, "XFER: writing summary files")

    # ----------------- write JSON (optional) -----------------
    json_path = None
    if write_json:
        json_path = os.path.join(out_dir, "xfer_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log(f"[XFER] Saved transfer results JSON -> {json_path}")

    # ----------------- write CSV (optional) -----------------
    csv_path = None
    if write_csv:
        base_cols = [
            "direction",
            "source_city",
            "target_city",
            "split",
            "calibration",
            "overall_mae",
            "overall_mse",
            "overall_r2",
            "coverage80",
            "sharpness80",
            "physics.epsilon_prior",
            "physics.epsilon_cons",
            "keras_eval_scaled.loss",
            "keras_eval_scaled.subs_pred_mae",
            "keras_eval_scaled.gwl_pred_mae",
            "csv_eval",
            "csv_future",
        ]

        def _sorted_hkeys(keys):
            def _k(k):
                try:
                    return int(str(k).strip().split("H")[-1])
                except Exception:
                    return 9999
            return sorted(keys, key=_k)

        h_mae_keys, h_r2_keys = set(), set()
        for r in results:
            h_mae_keys |= set((r.get("per_horizon_mae") or {}).keys())
            h_r2_keys  |= set((r.get("per_horizon_r2")  or {}).keys())

        h_mae_keys = _sorted_hkeys(h_mae_keys)
        h_r2_keys  = _sorted_hkeys(h_r2_keys)

        cols = base_cols + [f"per_horizon_mae.{k}" for k in h_mae_keys] \
                        + [f"per_horizon_r2.{k}"  for k in h_r2_keys]

        csv_path = os.path.join(out_dir, "xfer_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in results:
                phys = r.get("physics") or {}
                ev = r.get("keras_eval_scaled") or {}
                row = [
                    r.get("direction"),
                    r.get("source_city"),
                    r.get("target_city"),
                    r.get("split"),
                    r.get("calibration"),
                    r.get("overall_mae"),
                    r.get("overall_mse"),
                    r.get("overall_r2"),
                    r.get("coverage80"),
                    r.get("sharpness80"),
                    phys.get("epsilon_prior"),
                    phys.get("epsilon_cons"),
                    ev.get("loss"),
                    ev.get("subs_pred_mae"),
                    ev.get("gwl_pred_mae"),
                    r.get("csv_eval"),
                    r.get("csv_future"),
                ]
                ph_mae = r.get("per_horizon_mae") or {}
                ph_r2 = r.get("per_horizon_r2") or {}
                row.extend([ph_mae.get(k, "NA") for k in h_mae_keys])
                row.extend([ph_r2.get(k, "NA") for k in h_r2_keys])
                w.writerow(row)

        log(f"[XFER] Saved transfer CSV -> {csv_path}")

    _progress(1.0, "XFER: done")
    return {"out_dir": out_dir, "results": results, "json_path": json_path, "csv_path": csv_path}


def run_one_direction(
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
    split: str,
    calib_mode: str,
    rescale_to_source: bool,
    batch_size: int,
    quantiles_override: Optional[Sequence[float]] = None,
    *,
    out_dir: Optional[str] = None,
    prefer_tuned: bool = True,
    align_policy: str = "align_by_name_pad",
    interval_target: float = 0.80,
    load_endpoint: str = "serve",
    export_physics_payload: bool = True,
    export_physical_parameters_csv: bool = True,
    write_eval_future_csv: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Optional[Dict[str, Any]]:

    def log(msg: str) -> None:
        (logger or print)(msg)

    def should_stop() -> bool:
        return bool(stop_check and stop_check())

    # -------- 1) Load NPZs for target city --------
    X_tgt, y_tgt = _pick_npz(M_tgt, split)
    if X_tgt is None:
        log(f"[XFER] No NPZs for target split={split!r}; skipping.")
        return None

    cfg_t = M_tgt.get("config") or {}
    mode = _cfg_mode(cfg_t)

    # fallback horizon from y if present; else 1
    horizon_fallback = 1
    if isinstance(y_tgt, dict) and "subsidence" in y_tgt:
        try:
            horizon_fallback = int(np.asarray(y_tgt["subsidence"]).shape[1])
        except Exception:
            pass

    H = _cfg_horizon(cfg_t, fallback=horizon_fallback)
    Q = _cfg_quantiles(cfg_t, override=quantiles_override)

    # shapes + target mapping
    X_tgt = _ensure_shapes(dict(X_tgt), mode, H)
    y_map = _map_targets(y_tgt)

    # -------- 2) Align inputs to source schema --------
    X_tgt = _align_inputs_to_source(
        X_tgt=X_tgt,
        M_src=M_src,
        M_tgt=M_tgt,
        policy=align_policy,
    )

    s_src, d_src = _infer_source_input_dims(M_src)
    s_tgt = int(X_tgt.get("static_features", np.zeros((X_tgt["dynamic_features"].shape[0], 0))).shape[-1])
    d_tgt = int(X_tgt["dynamic_features"].shape[-1])

    if align_policy.strip().lower() == "strict":
        if s_src != s_tgt:
            raise SystemExit(
                f"Static dim mismatch: source={s_src}, target={s_tgt}. "
                "Use align_policy='align_by_name_pad' or harmonize Stage-1."
            )
        if d_src != d_tgt:
            raise SystemExit(
                f"Dynamic dim mismatch: source={d_src}, target={d_tgt}. "
                "Use align_policy='align_by_name_pad' or harmonize Stage-1."
            )

    # After align_by_name_pad, dims should match the source expectation
    if s_src != s_tgt or d_src != d_tgt:
        log(
            "[XFER] WARNING: dims still mismatch after alignment: "
            f"static source={s_src} target={s_tgt}, "
            f"dynamic source={d_src} target={d_tgt}"
        )

    if should_stop():
        return None

    # -------- 3) Optional strict domain test: reproject dynamic scaling --------
    if rescale_to_source:
        enc_t = (M_tgt.get("artifacts") or {}).get("encoders") or {}
        scaler_info = enc_t.get("scaler_info")
        if isinstance(scaler_info, str) and os.path.exists(scaler_info):
            scaler_info = joblib.load(scaler_info)

        enc_s = (M_src.get("artifacts") or {}).get("encoders") or {}
        src_scaler_path = enc_s.get("main_scaler")
        if not src_scaler_path:
            raise SystemExit("Source 'main_scaler' path missing in manifest.")

        dyn_names = _cfg_features(cfg_t, "dynamic")
        X_tgt = reproject_dynamic_scale(
            X_np=X_tgt,
            target_scaler_info=scaler_info,
            source_scaler_path=src_scaler_path,
            dynamic_feature_order=dyn_names,
        )

    if should_stop():
        return None

    # -------- 4) Load latest model under *source* run dir --------
    run_dir = _manifest_run_dir(M_src)
    model_path = _latest_model_under(run_dir, prefer_tuned=prefer_tuned)
    if not model_path:
        raise SystemExit(f"No .keras found under {run_dir}")

    log(f"[XFER] Using source model: {model_path}")

    custom_objects = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
        "coverage80_fn": coverage80_fn,
        "sharpness80_fn": sharpness80_fn,
    }
    model = _load_model_v32(
        model_path,
        custom_objects=custom_objects,
        endpoint=load_endpoint,
        compile_model=True,
    )

    if should_stop():
        return None

    # -------- 5) Optional calibrator --------
    cal: Optional[IntervalCalibrator] = None
    calib_mode = (calib_mode or "none").strip().lower()

    if calib_mode == "source":
        cal = _load_source_calibrator(run_dir)
        if cal is not None:
            cal.target = float(interval_target)
            log("[XFER] Using source-city interval calibrator.")
    elif calib_mode == "target":
        try:
            vx, vy = _pick_npz(M_tgt, "val")
            if vx is not None:
                vx = _ensure_shapes(dict(vx), mode, H)
                vy = _map_targets(vy)
                ds_val = tf.data.Dataset.from_tensor_slices((vx, vy)).batch(batch_size)
                log("[XFER] Fitting target-city interval calibrator on VAL.")
                cal = fit_interval_calibrator_on_val(
                    model,
                    ds_val,
                    target=float(interval_target),
                )
        except Exception as e:
            log(f"[XFER] Target calibrator fit failed: {e}")
            cal = None

    # -------- 6) Export physical parameters (best-effort) --------
    if export_physical_parameters_csv and extract_physical_parameters is not None:
        try:
            save_dir = out_dir or os.path.dirname(model_path)
            ensure_directory_exists(save_dir)
            extract_physical_parameters(
                model,
                to_csv=True,
                filename=f"{M_tgt.get('city')}_xfer_physical_parameters.csv",
                save_dir=save_dir,
                model_name="geoprior",
            )
            log("[XFER] Exported physical parameters CSV.")
        except Exception as e:
            log(f"[XFER] Physical parameter export failed: {e}")

    if should_stop():
        return None

    # -------- 7) Predict (v3.2: y_pred dict) --------
    log("[XFER] Running predict(...)")
    pred_dict = model.predict(X_tgt, verbose=0)

    if not isinstance(pred_dict, dict):
        raise TypeError(
            "GeoPrior predict() must return a dict."
        )

    dims = (
        ((M_tgt.get("artifacts") or {})
         .get("sequences") or {})
        .get("dims") or {}
    )
    out_s = int(dims.get("output_subsidence_dim", 1))
    out_g = int(dims.get("output_gwl_dim", 1))

    def _as_np(v):
        if isinstance(v, tf.Tensor):
            return v.numpy()
        return np.asarray(v)

    def _pick(d, keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                return d[k]
        return None

    y_pack = pred_dict.get("y_pred", None)

    subs = _pick(pred_dict, ("subs_pred", "subsidence"))
    gwl = _pick(pred_dict, ("gwl_pred", "gwl"))

    if isinstance(y_pack, dict):
        if subs is None:
            subs = _pick(y_pack, ("subs_pred", "subsidence"))
        if gwl is None:
            gwl = _pick(y_pack, ("gwl_pred", "gwl"))

    if subs is None:
        y_arr = y_pack
        if y_arr is None:
            y_arr = pred_dict.get("data_final", None)

        if y_arr is None:
            raise KeyError(
                "Missing y_pred/subs_pred/gwl_pred keys."
            )

        y_arr = _as_np(y_arr)
        subs = y_arr[..., :out_s]
        gwl = y_arr[..., out_s:]

    subs = _as_np(subs)
    gwl = _as_np(gwl) if gwl is not None else None

    is_q = getattr(subs, "ndim", 0) == 4

    if gwl is None:
        shp = list(subs.shape)
        shp[-1] = max(0, out_g)
        gwl = np.zeros(shp, dtype=np.float32)

    if cal is not None and is_q:
        subs = apply_calibrator_to_subs(cal, subs)

    predictions = {
        "subs_pred": subs,
        "gwl_pred": gwl,
    }

    # -------- 8) Encoders / scalers (target city) --------
    enc = (
        (M_tgt.get("artifacts") or {})
        .get("encoders") or {}
    )

    coord_scaler = None
    if enc.get("coord_scaler"):
        try:
            coord_scaler = joblib.load(
                enc["coord_scaler"]
            )
        except Exception as e:
            log(
                f"[XFER] coord_scaler load fail: {e}"
            )

    scaler_info = enc.get("scaler_info")
    if isinstance(scaler_info, str):
        if os.path.exists(scaler_info):
            try:
                scaler_info = joblib.load(
                    scaler_info
                )
            except Exception as e:
                log(
                    f"[XFER] scaler_info load fail: {e}"
                )

    if isinstance(scaler_info, dict):
        for _, v in scaler_info.items():
            if not isinstance(v, dict):
                continue
            if "scaler" in v:
                continue
            p = v.get("scaler_path")
            if p and os.path.exists(p):
                try:
                    v["scaler"] = joblib.load(p)
                except Exception:
                    pass

    SUBS_COL = _cfg_subs_col(cfg_t)

    # -------- 9) Metrics (physical units) --------
    per_horizon_mae: Dict[str, float] = {}
    per_horizon_r2: Dict[str, float] = {}

    overall_mae = None
    overall_mse = None
    overall_r2 = None

    if y_map and ("subs_pred" in y_map):
        if scaler_info is not None:
            y_true = np.asarray(y_map["subs_pred"])

            if is_q:
                q_arr = np.asarray(Q, dtype=np.float32)
                med = np.abs(q_arr - 0.5)
                med_idx = int(np.argmin(med))
                y_pred = np.asarray(
                    predictions["subs_pred"][
                        :, :, med_idx, :
                    ]
                )
            else:
                y_pred = np.asarray(
                    predictions["subs_pred"]
                )

            y_true = y_true[..., :1]
            y_pred = y_pred[..., :1]

            y_true_p = inverse_scale_target(
                y_true,
                scaler_info=scaler_info,
                target_name=SUBS_COL,
            )
            y_pred_p = inverse_scale_target(
                y_pred,
                scaler_info=scaler_info,
                target_name=SUBS_COL,
            )

            h_eff = int(y_true_p.shape[1])
            for h in range(h_eff):
                yt = y_true_p[:, h, :].reshape(-1)
                yp = y_pred_p[:, h, :].reshape(-1)
                per_horizon_mae[f"H{h+1}"] = float(
                    mean_absolute_error(yt, yp)
                )
                per_horizon_r2[f"H{h+1}"] = float(
                    r2_score(yt, yp)
                )

            yt_all = y_true_p.reshape(-1)
            yp_all = y_pred_p.reshape(-1)

            overall_mae = float(
                mean_absolute_error(yt_all, yp_all)
            )
            overall_mse = float(
                np.mean((yt_all - yp_all) ** 2)
            )
            overall_r2 = float(
                r2_score(yt_all, yp_all)
            )

    # -------- 10) Scaled eval + physics diag --------
    eval_scaled = None
    physics_diag = None

    if y_map and not should_stop():
        ds = tf.data.Dataset.from_tensor_slices(
            (X_tgt, y_map)
        ).batch(batch_size)

        try:
            eval_scaled = model.evaluate(
                ds,
                return_dict=True,
                verbose=0,
            )
            log(
                "[XFER] Scaled eval ok."
            )
        except Exception as e:
            log(
                f"[XFER] evaluate() failed: {e}"
            )
            eval_scaled = None

    if eval_scaled is not None:
        keys = ("epsilon_prior", "epsilon_cons")
        try:
            physics_diag = {}
            for k in keys:
                if k in eval_scaled:
                    physics_diag[k] = float(
                        _to_py(eval_scaled[k])
                    )
        except Exception:
            physics_diag = None

    # -------- 11) Export physics payload --------
    if export_physics_payload:
        try:
            save_dir = out_dir
            if save_dir is None:
                save_dir = os.path.dirname(model_path)

            ensure_directory_exists(save_dir)

            fname = (
                f"{M_tgt.get('city')}"
                f"_xfer_physics_payload_{split}.npz"
            )
            fpath = os.path.join(save_dir, fname)

            model.export_physics_payload(
                X_tgt,
                max_batches=None,
                save_path=fpath,
                format="npz",
                overwrite=True,
                metadata={
                    "city": M_tgt.get("city"),
                    "split": split,
                },
            )
            log("[XFER] Physics payload saved.")
        except Exception as e:
            log(
                f"[XFER] Physics payload fail: {e}"
            )

    # -------- 12) Interval metrics (phys space) --------
    coverage80 = None
    sharpness80 = None

    if y_map and is_q and not should_stop():
        if scaler_info is not None:
            ds_calc = tf.data.Dataset.from_tensor_slices(
                (X_tgt, y_map)
            ).batch(batch_size)

            y_true_list = []
            s_q_list = []

            desc = (
                f"Diagnose {M_tgt.get('city')} "
                f"xfer-metrics"
            )

            for xb, yb in with_progress(ds_calc, desc=desc):
                out = model(xb, training=False)
                if not isinstance(out, dict):
                    raise TypeError(
                        "model(x) must return a dict."
                    )

                pack = out.get("y_pred", out)
                s_q_b = _pick(
                    pack,
                    ("subs_pred", "subsidence"),
                )

                if s_q_b is None:
                    y_arr = pack.get("data_final", None)
                    if y_arr is None:
                        raise KeyError(
                            "Missing subs_pred in model output."
                        )
                    s_q_b = y_arr[..., :out_s]

                s_q_b = tf.convert_to_tensor(s_q_b)
                y_true_list.append(yb["subs_pred"])
                s_q_list.append(s_q_b)

            if y_true_list and s_q_list:
                y_true_s = tf.concat(y_true_list, axis=0)
                s_q_s = tf.concat(s_q_list, axis=0)

                y_true_p = inverse_scale_target(
                    y_true_s,
                    scaler_info=scaler_info,
                    target_name=SUBS_COL,
                )
                s_q_p = inverse_scale_target(
                    s_q_s,
                    scaler_info=scaler_info,
                    target_name=SUBS_COL,
                )

                y_true_tf = tf.convert_to_tensor(
                    y_true_p,
                    dtype=tf.float32,
                )
                s_q_tf = tf.convert_to_tensor(
                    s_q_p,
                    dtype=tf.float32,
                )

                coverage80 = float(
                    coverage80_fn(y_true_tf, s_q_tf).numpy()
                )
                sharpness80 = float(
                    sharpness80_fn(y_true_tf, s_q_tf).numpy()
                )

    # -------- 13) Forecast CSVs (eval + future) --------
    csv_eval_path = None
    csv_future_path = None

    if write_eval_future_csv:
        y_true_for_fmt = None
        if y_map:
            y_true_for_fmt = {}
            if "subs_pred" in y_map:
                y_true_for_fmt["subsidence"] = (
                    y_map["subs_pred"]
                )
            if "gwl_pred" in y_map:
                y_true_for_fmt["gwl"] = y_map["gwl_pred"]

        train_end = _cfg_get(
            cfg_t,
            "train_end_year",
            "TRAIN_END_YEAR",
            default=None,
        )
        fcst_start = _cfg_get(
            cfg_t,
            "forecast_start_year",
            "FORECAST_START_YEAR",
            default=None,
        )

        future_grid = None
        if fcst_start is not None:
            try:
                future_grid = np.arange(
                    float(fcst_start),
                    float(fcst_start) + float(H),
                    dtype=float,
                )
            except Exception:
                future_grid = None

        save_dir = out_dir
        if save_dir is None:
            save_dir = os.path.dirname(model_path)

        ensure_directory_exists(save_dir)

        base = (
            f"{M_src.get('city')}"
            f"_to_{M_tgt.get('city')}"
            f"_xfer_{split}_{calib_mode}"
        )
        csv_eval_path = os.path.join(
            save_dir,
            base + "_eval.csv",
        )
        csv_future_path = os.path.join(
            save_dir,
            base + "_future.csv",
        )

        format_and_forecast(
            y_pred=predictions,
            y_true=y_true_for_fmt,
            coords=X_tgt.get("coords", None),
            quantiles=Q if is_q else None,
            target_name=SUBS_COL,
            target_key_pred="subs_pred",
            component_index=0,
            scaler_info=scaler_info,
            coord_scaler=coord_scaler,
            coord_columns=("coord_t", "coord_x", "coord_y"),
            train_end_time=train_end,
            forecast_start_time=fcst_start,
            forecast_horizon=H,
            future_time_grid=future_grid,
            eval_forecast_step=None,
            eval_export="all",
            sample_index_offset=0,
            city_name=M_tgt.get("city"),
            model_name=(
                M_src.get("model")
                or "GeoPriorSubsNet"
            ),
            dataset_name=(
                f"xfer_{split}_{calib_mode}"
            ),
            csv_eval_path=csv_eval_path,
            csv_future_path=csv_future_path,
            time_as_datetime=False,
            time_format=None,
            verbose=0,
            eval_metrics=False,
            metrics_time_as_str=True,
            value_mode="rate",
            logger=log,
        )

    return {
        "model_path": model_path,
        "split": split,
        "calibration": calib_mode,
        "quantiles": Q if is_q else None,
        "keras_eval_scaled": eval_scaled,
        "physics": physics_diag,
        "coverage80": coverage80,
        "sharpness80": sharpness80,
        "overall_mae": overall_mae,
        "overall_mse": overall_mse,
        "overall_r2": overall_r2,
        "per_horizon_mae": per_horizon_mae,
        "per_horizon_r2": per_horizon_r2,
        "csv_eval": csv_eval_path,
        "csv_future": csv_future_path,
    }

