# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Stage-1 helper: preprocessing & sequence export for GeoPriorSubsNet.

Expose a single entry point, :func:`run_stage1`, which wraps the
original NATCOM Stage-1 script into a reusable function usable from
both CLI and Qt GUI.
"""

from __future__ import annotations

import os
import json
import shutil
import joblib
import warnings
from pathlib import Path  
import datetime as dt
from typing import Dict, Tuple, Optional, Callable, Any

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .....api.util import get_table_size
from .....utils.data_utils import nan_ops
from .....utils.io_utils import save_job
from .....utils.nat_utils import load_nat_config
from .....utils.generic_utils import (
    normalize_time_column,
    ensure_directory_exists,
    print_config_table,
)
from ....utils.audit_utils import (
    audit_stage1_scaling,
    should_audit,
)

from ....utils.spatial_utils import deg_to_m_from_lat
from ....utils.subsidence_utils import (
    cumulative_to_rate,
    make_txy_coords,
    normalize_gwl_alias,
    rate_to_cumulative,
    resolve_gwl_for_physics,
    resolve_head_column,
)

from .....utils.sequence_utils import build_future_sequences_npz
from .....nn.pinn.utils import prepare_pinn_data_sequences

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")
if hasattr(tf, "autograph") and hasattr(tf.autograph, "set_verbosity"):
    tf.autograph.set_verbosity(0)
    

def _save_npz(path: str, arrays: Dict[str, np.ndarray]) -> str:
    """Save dict of numpy arrays to compressed NPZ."""
    np.savez_compressed(path, **arrays)
    return path


def _dataset_to_numpy_pair(
    ds: tf.data.Dataset, limit: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Unbatch a (x, y) dataset into stacked numpy dicts."""
    xs, ys = [], []
    count = 0
    for x_b, y_b in ds.unbatch():
        xs.append({k: v.numpy() for k, v in x_b.items()})
        ys.append({k: v.numpy() for k, v in y_b.items()})
        count += 1
        if limit is not None and count >= limit:
            break
    if not xs:
        raise RuntimeError("Empty dataset when converting to numpy.")

    x0 = xs[0]
    x_np = {k: np.stack([xi[k] for xi in xs], axis=0) for k in x0}
    y0 = ys[0]
    y_np = {k: np.stack([yi[k] for yi in ys], axis=0) for k in y0}
    return x_np, y_np


def _resolve_optional_columns(
    df: pd.DataFrame, spec_list
) -> tuple[list[str], dict]:
    """
    From a list like ["a", ("b","b2"), "c"], return:
      present_cols: concrete cols found in df (first match in tuples)
      mapping: {chosen_col: original_spec}.
    """
    present, mapping = [], {}
    cols = set(df.columns)
    for spec in spec_list:
        if isinstance(spec, (list, tuple, set)):
            chosen = next((s for s in spec if s in cols), None)
            if chosen is not None:
                present.append(chosen)
                mapping[chosen] = tuple(spec)
        else:
            if spec in cols:
                present.append(spec)
                mapping[spec] = spec
    return present, mapping


def _drop_missing_keep_order(cands: list[str], df: pd.DataFrame) -> list[str]:
    """Keep only those in df.columns preserving order."""
    cols = set(df.columns)
    return [c for c in cands if c in cols]


def _apply_censoring(
    df: pd.DataFrame, specs: list[dict]
) -> tuple[pd.DataFrame, dict]:
    """
    Add <col>_censored (bool) and <col>_eff (float) for each spec.
    Returns (df, report) with basic rates for the manifest.
    """
    report = {}
    for sp in specs or []:
        col = sp.get("col")
        if not col or col not in df.columns:
            continue

        cap = sp.get("cap")
        tol = float(sp.get("tol", 0.0))
        dirn = sp.get("direction", "right")
        fflag = col + sp.get("flag_suffix", "_censored")
        feff = col + sp.get("eff_suffix", "_eff")
        mode = sp.get("eff_mode", "clip")
        eps = float(sp.get("eps", 0.02))

        x = pd.to_numeric(df[col], errors="coerce").astype(float)

        if dirn == "right":
            m = np.isfinite(x) & (x >= (cap - tol))
        elif dirn == "left":
            m = np.isfinite(x) & (x <= (cap + tol))
        else:
            raise ValueError("Censoring 'direction' must be 'left' or 'right'.")

        df[fflag] = m.astype(bool)

        if mode == "clip" and cap is not None:
            eff = np.minimum(x, cap)
        elif mode == "cap_minus_eps" and cap is not None:
            eff = np.where(m, cap * (1.0 - eps), x)
        elif mode == "nan_if_censored":
            eff = x.copy()
            eff[m] = np.nan
            imp = sp.get("impute") or {}
            by = imp.get("by", [])
            func = imp.get("func", "median")
            if by:
                df[feff] = eff
                grp = df.groupby(by)[feff]
                if func == "median":
                    df[feff] = grp.transform(lambda s: s.fillna(s.median()))
                elif func == "mean":
                    df[feff] = grp.transform(lambda s: s.fillna(s.mean()))
                eff = df[feff].to_numpy(dtype=float)
        else:
            eff = x  # fallback

        df[feff] = eff
        report[col] = {
            "direction": dirn,
            "cap": cap,
            "flag_col": fflag,
            "eff_col": feff,
            "eff_mode": mode,
            "censored_rate": float(np.mean(m)),
        }
    return df, report

def _find_latest_gui_dataset(
    city_name: str,
    *,
    results_root: Optional[os.PathLike | str],
    logger: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """
    Try to locate the most recent CSV dataset saved by the GeoPrior GUI
    for a given city.

    Search order
    ------------
    1. ``results_root / "_datasets"``  (if ``results_root`` is given)
    2. ``Path.home() / ".fusionlab_runs" / "_datasets"``

    Files considered
    ----------------
    - ``{city_name}.csv``
    - ``{city_name}_<int>.csv``

    Returns
    -------
    str or None
        Absolute path to the latest CSV, or ``None`` if no match is found.
    """
    log = logger or (lambda msg: None)

    city_slug = (city_name or "").strip()
    if not city_slug:
        city_slug = "geoprior_city"

    search_roots: list[Path] = []
    if results_root is not None:
        search_roots.append(Path(results_root))

    user_root = Path.home() / ".fusionlab_runs"
    if user_root not in search_roots:
        search_roots.append(user_root)

    candidates: list[Path] = []

    for root in search_roots:
        ds_dir = root / "_datasets"
        if not ds_dir.is_dir():
            continue

        for p in ds_dir.glob("*.csv"):
            stem = p.stem
            if stem == city_slug or stem.startswith(f"{city_slug}_"):
                candidates.append(p)

    if not candidates:
        log(
            f"[Stage-1] No GUI datasets found for city '{city_slug}' "
            "under any _datasets directory in: "
            + ", ".join(str(r) for r in search_roots)
        )
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    log(
        "[Stage-1] Auto-detected GUI dataset: "
        f"{latest} (latest by mtime among {len(candidates)} candidate(s))"
    )
    return str(latest.resolve())

def _distinct_preserve_order(names: list[str]) -> list[str]:
    """Return a list with duplicates removed, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for n in names or []:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _filter_present_features(
    requested: list[str],
    df: pd.DataFrame,
    *,
    what: str,
    log: Callable[[str], None],
    banned: Optional[set[str]] = None,
) -> list[str]:
    """
    Keep only requested features that exist in df.columns,
    log missing / banned ones.
    """
    requested = _distinct_preserve_order(requested or [])
    if not requested:
        return []

    cols = set(df.columns)
    banned = banned or set()

    present: list[str] = []
    missing: list[str] = []
    skipped_banned: list[str] = []

    for name in requested:
        if name in banned:
            skipped_banned.append(name)
            continue
        if name in cols:
            present.append(name)
        else:
            missing.append(name)

    if missing:
        log(
            f"  [Info] {what} features not found in dataset and will be "
            f"skipped: {missing}"
        )
    if skipped_banned:
        log(
            f"  [Info] {what} features include target/H-field columns that "
            f"cannot be used as inputs and will be skipped: {skipped_banned}"
        )

    return present


def _resolve_feature_sets(
    df_scaled: pd.DataFrame,
    *,
    encoded_names: list[str],
    opt_num_cols: list[str],
    static_cfg: list[str],
    dynamic_cfg: list[str],
    future_cfg: list[str],
    gwl_col: str,
    subs_col: str,
    h_field_col: str,
    log: Callable[[str], None],
) -> tuple[list[str], list[str], list[str]]:
    """
    Resolve static / dynamic / future feature sets with user overrides.

    Rules
    -----
    - If STATIC_DRIVER_FEATURES non-empty:
        * Use intersection with df_scaled.columns.
        * Log missing ones.
        * If nothing valid → fall back to encoded_names.
    - If DYNAMIC_DRIVER_FEATURES non-empty:
        * Use intersection with df_scaled.columns, excluding subs/h_field.
        * Log missing / banned ones.
        * If nothing valid → fall back to auto dynamic set.
    - FUTURE_DRIVER_FEATURES:
        * Always treated as the declared list (from config or overrides).
        * Use intersection with df_scaled.columns, log missing.
        * If empty → it's allowed (no future drivers).
    """
    df_cols = list(df_scaled.columns)

    # --- auto dynamic baseline (current behaviour) --------------------
    banned_dynamic = {subs_col, h_field_col}
    dynamic_base = [gwl_col]
    dynamic_extra = [
        c for c in opt_num_cols if c not in banned_dynamic
    ]
    dynamic_auto = [
        c for c in dynamic_base + dynamic_extra if c in df_cols
    ]

    # --- static -------------------------------------------------------
    static_cfg = list(static_cfg or [])
    if static_cfg:
        static_features = _filter_present_features(
            static_cfg,
            df_scaled,
            what="Static",
            log=log,
        )
        if not static_features:
            log(
                "  [Info] No valid static features from "
                "STATIC_DRIVER_FEATURES; falling back to "
                f"auto static set (encoded categorical features): "
                f"{encoded_names}"
            )
            static_features = encoded_names[:]
    else:
        static_features = encoded_names[:]

    # --- dynamic ------------------------------------------------------
    dynamic_cfg = list(dynamic_cfg or [])
    if dynamic_cfg:
        dynamic_features = _filter_present_features(
            dynamic_cfg,
            df_scaled,
            what="Dynamic",
            log=log,
            banned=banned_dynamic,
        )
        if not dynamic_features:
            log(
                "  [Info] No valid dynamic features from "
                "DYNAMIC_DRIVER_FEATURES; falling back to "
                f"auto dynamic set: {dynamic_auto}"
            )
            dynamic_features = dynamic_auto
    else:
        dynamic_features = dynamic_auto

    # --- future -------------------------------------------------------
    future_cfg = list(future_cfg or [])
    future_features = _filter_present_features(
        future_cfg,
        df_scaled,
        what="Future driver",
        log=log,
    )
    # If empty, that's OK (no future drivers).

    return (
        _distinct_preserve_order(static_features),
        _distinct_preserve_order(dynamic_features),
        _distinct_preserve_order(future_features),
    )

def _any_exists(paths: list[str]) -> bool:
    return any(os.path.exists(p) for p in paths)


# ======================================================================
# Main entry point
# ======================================================================
def run_stage1(
    cfg_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    clean_run_dir: bool = False,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None, 
    base_cfg: Optional[Dict[str, Any]] = None,
    results_root: Optional[os.PathLike | str] = None,
    edited_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Run NATCOM GeoPrior Stage-1 for the current city.

    Parameters
    ----------
    cfg_overrides : dict, optional
        Keys to override from ``load_nat_config()``, e.g.
        ``{"CITY_NAME": "zhongshan"}``.
    logger : callable, optional
        Function ``f(msg: str) -> None`` used for logging instead of
        :func:`print`. In the GUI, pass something like
        ``logger=self.append_log``.
    clean_run_dir : bool, default True
        If True, delete any existing ``*_stage1`` folder before running.

    Returns
    -------
    dict
        ``{"manifest", "manifest_path", "run_dir", "artifacts_dir"}``.
    """
    def _seq_progress_train(local_frac: float) -> None:
        # Map [0, 1] in sequence-building → [0.40, 0.60] in Stage-1
        _progress(
            0.40 + 0.20 * float(local_frac),
            "Stage-1: building train sequences",
        )

    def _seq_progress_future(local_frac: float) -> None:
        # Map [0, 1] in future-building → [0.70, 0.90] in Stage-1 (for example)
        _progress(
            0.70 + 0.20 * float(local_frac),
            "Stage-1: building future sequences",
        )
        
    def _progress(fraction: float, msg: str) -> None:
        if progress_callback is None:
            return
        try:
            # clamp and forward
            f = max(0.0, min(1.0, float(fraction)))
            progress_callback(f, msg)
        except Exception:
            # never crash Stage-1 because the GUI callback misbehaved
            pass

    def _maybe_stop(stage: str) -> None:
        """
        Check cooperative stop flag and abort Stage-1 cleanly.

        Parameters
        ----------
        stage : str
            Short description of where we are (for logs/GUI).
        """
        if stop_check is not None:
            try:
                should_stop = bool(stop_check())
            except Exception:
                # Never crash if the callback misbehaves
                should_stop = False
            if should_stop:
                msg = f"[Stage-1] Stop requested → aborting at: {stage}."
                log(msg)
                _progress(1.0, f"Stage-1 aborted at: {stage}")
                raise InterruptedError("Stage-1 was interrupted by user.")


    log = logger or (lambda msg: print(msg, flush=True))

    # ===================== CONFIG =====================
    # 1) Start from a base NAT-style config.
    if base_cfg is not None:
        # Shallow copy so we don't mutate the caller's dict.
        cfg = dict(base_cfg)
    else:
        GUI_CONFIG_DIR = os.path.dirname(__file__)
        config_root = os.path.join( os.path.dirname (GUI_CONFIG_DIR), 'config')
        cfg = load_nat_config(root=config_root)

    # 2) Apply flat overrides (from GeoPriorConfig.to_cfg_overrides, plus
    #    CITY_NAME / DATA_DIR / BIG_FN / BASE_OUTPUT_DIR, etc.).
    if cfg_overrides:
        cfg.update(cfg_overrides)

    # 3) Decide where Stage-1 runs live.
    #    - If GUI passes results_root, we force BASE_OUTPUT_DIR to that.
    #    - Otherwise, fall back to NAT's BASE_OUTPUT_DIR or ./results.
    if results_root is not None:
        base_output_dir = os.fspath(results_root)
        cfg["BASE_OUTPUT_DIR"] = base_output_dir
    else:
        base_output_dir = cfg.get(
            "BASE_OUTPUT_DIR",
            os.path.join(os.getcwd(), "results"),
        )
        cfg["BASE_OUTPUT_DIR"] = base_output_dir


    CITY_NAME   = cfg.get("CITY_NAME", '')
    MODEL_NAME  = cfg.get("MODEL_NAME", 'GeoPriorSubsNet')
    DATA_DIR    = cfg.get("DATA_DIR", base_output_dir)
    BIG_FN      = cfg.get("BIG_FN", "")
    SMALL_FN    = cfg.get("SMALL_FN", "")

    TRAIN_END_YEAR         = cfg["TRAIN_END_YEAR"]
    FORECAST_START_YEAR    = cfg["FORECAST_START_YEAR"]
    FORECAST_HORIZON_YEARS = cfg["FORECAST_HORIZON_YEARS"]
    TIME_STEPS             = cfg["TIME_STEPS"]
    MODE                   = cfg["MODE"]
    BUILD_FUTURE_NPZ       = bool(cfg.get("BUILD_FUTURE_NPZ", False))

    TIME_COL         = cfg["TIME_COL"]
    LON_COL          = cfg["LON_COL"]
    LAT_COL          = cfg["LAT_COL"]
    SUBSIDENCE_COL   = cfg["SUBSIDENCE_COL"]
    GWL_COL          = cfg["GWL_COL"]
    H_FIELD_COL_NAME = cfg["H_FIELD_COL_NAME"]

    # --- v3.2 hydro/geometry columns ---
    Z_SURF_COL = cfg.get("Z_SURF_COL", None)
    HEAD_COL = cfg.get("HEAD_COL", None)
    USE_HEAD_PROXY = bool(cfg.get("USE_HEAD_PROXY", True))
    
    GWL_KIND = str(cfg.get("GWL_KIND", "depth_bgs")).strip().lower()
    GWL_SIGN = cfg.get("GWL_SIGN", None)
    
    # --- v3.2 units -> SI (meters) ---
    SUBS_UNIT_TO_SI = float(cfg.get("SUBS_UNIT_TO_SI", 1.0))
    HEAD_UNIT_TO_SI = float(cfg.get("HEAD_UNIT_TO_SI", 1.0))
    THICKNESS_UNIT_TO_SI = float(cfg.get("THICKNESS_UNIT_TO_SI", 1.0))
    Z_SURF_UNIT_TO_SI = float(cfg.get("Z_SURF_UNIT_TO_SI", 1.0))
    H_MIN_SI = float(cfg.get("H_MIN_SI", 1.0))
    
    # --- v3.2 behavior toggles ---
    SUBSIDENCE_KIND = str(
        cfg.get("SUBSIDENCE_KIND", "cumulative")
    ).strip().lower()
    
    ALLOW_SUBS_RESIDUAL = bool(cfg.get("ALLOW_SUBS_RESIDUAL", True))
    INCLUDE_Z_SURF_AS_STATIC = bool(
        cfg.get("INCLUDE_Z_SURF_AS_STATIC", True)
    )
    INCLUDE_SUBS_HIST_DYNAMIC = bool(
        cfg.get("INCLUDE_SUBS_HIST_DYNAMIC", True)
    )
    
    # --- v3.2 coordinate handling ---
    COORD_MODE = str(cfg.get("COORD_MODE", "degrees")).strip().lower()
    COORD_SRC_EPSG = int(cfg.get("COORD_SRC_EPSG", 4326))
    COORD_TARGET_EPSG = int(cfg.get("COORD_TARGET_EPSG", 32649))
    COORD_X_COL = str(cfg.get("COORD_X_COL", "x_m"))
    COORD_Y_COL = str(cfg.get("COORD_Y_COL", "y_m"))
    
    NORMALIZE_COORDS = bool(cfg.get("NORMALIZE_COORDS", True))
    KEEP_COORDS_RAW = bool(cfg.get("KEEP_COORDS_RAW", not NORMALIZE_COORDS))
    
    if "NORMALIZE_COORDS" in cfg:
        KEEP_COORDS_RAW = not bool(cfg.get("NORMALIZE_COORDS", True))
    else:
        NORMALIZE_COORDS = not KEEP_COORDS_RAW

    SHIFT_RAW_COORDS = bool(cfg.get("SHIFT_RAW_COORDS", True))
    
    AUDIT_STAGES = cfg.get("AUDIT_STAGES", []) or []


    OPTIONAL_NUMERIC_FEATURES = (
        cfg.get("OPTIONAL_NUMERIC_FEATURES_REGISTRY")
        or cfg.get("OPTIONAL_NUMERIC_FEATURES")
        or []
    )
    OPTIONAL_CATEGORICAL_FEATURES = (
        cfg.get("OPTIONAL_CATEGORICAL_FEATURES_REGISTRY")
        or cfg.get("OPTIONAL_CATEGORICAL_FEATURES")
        or []
    )
    ALREADY_NORMALIZED_FEATURES   = cfg["ALREADY_NORMALIZED_FEATURES"]
    FUTURE_DRIVER_FEATURES        = list(
        cfg.get("FUTURE_DRIVER_FEATURES", []) or []
    )

    # NEW: optional user-defined feature lists (from GeoPriorConfig)
    STATIC_DRIVER_FEATURES = list(
        cfg.get("STATIC_DRIVER_FEATURES", []) or []
    )
    DYNAMIC_DRIVER_FEATURES = list(
        cfg.get("DYNAMIC_DRIVER_FEATURES", []) or []
    )

    _censor_cfg = cfg.get("censoring", {}) or {}
    CENSORING_SPECS = _censor_cfg.get("specs") or cfg.get("CENSORING_SPECS", [])
    INCLUDE_CENSOR_FLAGS_AS_DYNAMIC = bool(
        _censor_cfg.get("flags_as_dynamic", cfg.get(
            "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC", True)
            )
    )
    # --- v3.2 censor flags ---
    INCLUDE_CENSOR_FLAGS_AS_FUTURE = bool(
        _censor_cfg.get(
            "flags_as_future",
            cfg.get("INCLUDE_CENSOR_FLAGS_AS_FUTURE", True),
        )
    )
    
    USE_EFFECTIVE_H_FIELD = bool(
        _censor_cfg.get("use_effective_h_field", cfg.get(
            "USE_EFFECTIVE_H_FIELD", True)
            )
    )
    TRACK_AUX_METRICS = bool(cfg.get("TRACK_AUX_METRICS", True))

    CONSOLIDATION_RESIDUAL_UNITS = str(
        cfg.get("CONSOLIDATION_RESIDUAL_UNITS", "second")
    )
    GW_RESIDUAL_UNITS = str(cfg.get("GW_RESIDUAL_UNITS", "time_unit"))
    CLIP_GLOBAL_NORM = float(cfg.get("CLIP_GLOBAL_NORM", 5.0))
    
    CONS_SCALE_FLOOR = cfg.get("CONS_SCALE_FLOOR", 1e-5)
    GW_SCALE_FLOOR = cfg.get("GW_SCALE_FLOOR", 1e-5)
    DT_MIN_UNITS = float(cfg.get("DT_MIN_UNITS", 1e-6))
    
    Q_WRT_NORMALIZED_TIME = bool(cfg.get("Q_WRT_NORMALIZED_TIME", False))
    Q_IN_SI = bool(cfg.get("Q_IN_SI", False))
    Q_IN_PER_SECOND = bool(cfg.get("Q_IN_PER_SECOND", False))
    Q_KIND = str(cfg.get("Q_kind", "per_volume"))
    Q_LENGTH_IN_SI = bool(cfg.get("Q_length_in_si", False))
    DRAINAGE_MODE = str(cfg.get("DRAINAGE_MODE", "double"))
    
    SCALING_ERROR_POLICY = str(cfg.get("SCALING_ERROR_POLICY", "warn"))
    DEBUG_PHYSICS_GRADS = bool(cfg.get("DEBUG_PHYSICS_GRADS", False))
    
    CONS_DRAWDOWN_MODE = str(
        cfg.get("CONS_DRAWDOWN_MODE", "smooth_relu")
    ).lower()
    CONS_DRAWDOWN_RULE = str(
        cfg.get("CONS_DRAWDOWN_RULE", "ref_minus_mean")
    ).lower()
    
    CONS_STOP_GRAD_REF = bool(cfg.get("CONS_STOP_GRAD_REF", True))
    CONS_DRAWDOWN_ZERO_AT_ORIGIN = bool(
        cfg.get("CONS_DRAWDOWN_ZERO_AT_ORIGIN", False)
    )
    CONS_RELU_BETA = float(cfg.get("CONS_RELU_BETA", 20.0))
    
    _cons_clip = cfg.get("CONS_DRAWDOWN_CLIP_MAX", None)
    if _cons_clip is None:
        CONS_DRAWDOWN_CLIP_MAX = None
    else:
        try:
            CONS_DRAWDOWN_CLIP_MAX = float(_cons_clip)
        except Exception:
            CONS_DRAWDOWN_CLIP_MAX = None
    
    MV_PRIOR_UNITS = str(cfg.get("MV_PRIOR_UNITS", "auto"))
    MV_ALPHA_DISP = float(cfg.get("MV_ALPHA_DISP", 0.1))
    MV_HUBER_DELTA = float(cfg.get("MV_HUBER_DELTA", 1.0))
    MV_PRIOR_MODE = str(cfg.get("MV_PRIOR_MODE", "calibrate"))
    MV_WEIGHT = float(cfg.get("MV_WEIGHT", 1e-3))
    
    MV_SCHEDULE_UNIT = str(
        cfg.get("MV_SCHEDULE_UNIT", "epoch")
    ).strip().lower()
    
    MV_DELAY_EPOCHS = int(cfg.get("MV_DELAY_EPOCHS", 1))
    MV_WARMUP_EPOCHS = int(cfg.get("MV_WARMUP_EPOCHS", 2))
    
    MV_DELAY_STEPS = cfg.get("MV_DELAY_STEPS", None)
    MV_WARMUP_STEPS = cfg.get("MV_WARMUP_STEPS", None)
    
    if MV_SCHEDULE_UNIT not in ("epoch", "step"):
        raise ValueError(
            "MV_SCHEDULE_UNIT must be 'epoch' or 'step'."
        )
    
    if MV_SCHEDULE_UNIT == "step" and MV_WARMUP_STEPS is None:
        MV_WARMUP_STEPS = 4780 * 2
    if MV_SCHEDULE_UNIT == "step" and MV_DELAY_STEPS is None:
        MV_DELAY_STEPS = 0
    
    TIME_UNITS = str(cfg.get("TIME_UNITS", "year"))


    BASE_OUTPUT_DIR = base_output_dir
    ensure_directory_exists(BASE_OUTPUT_DIR)

     # auto-detect a GUI dataset under _datasets/, if any
    auto_dataset = _find_latest_gui_dataset(
        CITY_NAME,
        results_root=BASE_OUTPUT_DIR,
        logger=log,
    )
    if auto_dataset is not None:
        # Keep config consistent with what we really use
        DATA_DIR = os.path.dirname(auto_dataset)
        BIG_FN = os.path.basename(auto_dataset)
        cfg["DATA_DIR"] = DATA_DIR
        cfg["BIG_FN"] = BIG_FN
        log(
            "[Stage-1] Overriding DATA_DIR/BIG_FN from GUI dataset: "
            f"DATA_DIR={DATA_DIR}, BIG_FN={BIG_FN}"
        )


    RUN_OUTPUT_PATH = os.path.join(
        BASE_OUTPUT_DIR, f"{CITY_NAME}_{MODEL_NAME}_stage1"
    )

    if clean_run_dir and os.path.isdir(RUN_OUTPUT_PATH):
        log(f"Cleaning existing Stage-1 directory: {RUN_OUTPUT_PATH}")
        shutil.rmtree(RUN_OUTPUT_PATH)
        
    os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)

    ARTIFACTS_DIR = os.path.join(RUN_OUTPUT_PATH, "artifacts")
    ensure_directory_exists(ARTIFACTS_DIR)

    try:
        _TW = get_table_size()
    except Exception:
        _TW = 80

    log(
        f"\n{'-'*_TW}\n{CITY_NAME.upper()} {MODEL_NAME} STAGE-1 "
        f"(Steps 1–6)\n{'-'*_TW}"
    )
    log(
        f"TIME_STEPS={TIME_STEPS}, "
        f"HORIZON={FORECAST_HORIZON_YEARS}, MODE={MODE}"
    )

    config_sections = [
        ("Run", {
            "CITY_NAME": CITY_NAME,
            "MODEL_NAME": MODEL_NAME,
            "DATA_DIR": DATA_DIR,
            "BIG_FN": BIG_FN,

        }),
        ("Time windows", {
            "TRAIN_END_YEAR": TRAIN_END_YEAR,
            "FORECAST_START_YEAR": FORECAST_START_YEAR,
            "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
            "TIME_STEPS": TIME_STEPS,
            "MODE": MODE,
        }),
        ("Columns", {
            "TIME_COL": TIME_COL,
            "LON_COL": LON_COL,
            "LAT_COL": LAT_COL,
            "SUBSIDENCE_COL": SUBSIDENCE_COL,
            "GWL_COL": GWL_COL,
            "H_FIELD_COL_NAME": H_FIELD_COL_NAME,
        }),
        ("Feature registry", {
            "OPTIONAL_NUMERIC_FEATURES": OPTIONAL_NUMERIC_FEATURES,
            "OPTIONAL_CATEGORICAL_FEATURES":
                OPTIONAL_CATEGORICAL_FEATURES,
            "ALREADY_NORMALIZED_FEATURES": ALREADY_NORMALIZED_FEATURES,
            "FUTURE_DRIVER_FEATURES": FUTURE_DRIVER_FEATURES,
            "STATIC_DRIVER_FEATURES": STATIC_DRIVER_FEATURES, 
            "DYNAMIC_DRIVER_FEATURES": DYNAMIC_DRIVER_FEATURES,
            
        }),
        ("Censoring", {
            "CENSORING_SPECS": CENSORING_SPECS,
            "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC":
                INCLUDE_CENSOR_FLAGS_AS_DYNAMIC,
            "USE_EFFECTIVE_H_FIELD": USE_EFFECTIVE_H_FIELD,
        }),
        ("Outputs", {
            "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
            "RUN_OUTPUT_PATH": RUN_OUTPUT_PATH,
            "ARTIFACTS_DIR": ARTIFACTS_DIR,
        }),
    ]
    # This still prints to stdout; GUI can capture stdout if desired.
    print_config_table(
        config_sections,
        table_width=_TW,
        title=f"{CITY_NAME.upper()} {MODEL_NAME} STAGE-1 CONFIG",
        print_fn =log
    )
    _maybe_stop("after config resolved")
    
    # ===================== STEP 1: LOAD =====================
    log(f"\n{'='*18} Step 1: Load Dataset {'='*18}")

    df_raw: pd.DataFrame | None = None
    used_path: str | None = None

    if edited_df is not None:
        # GUI provided an in-memory dataset (already edited)
        df_raw = edited_df.copy()
        log(f"  Using in-memory dataset provided by GUI -> {df_raw.shape}")
    else:
        # In GUI/CLI mode the CSV path is DATA_DIR + BIG_FN, with
        # optional fallback to SMALL_FN if BIG_FN is missing.
        candidates: list[str] = []

        if BIG_FN:
            candidates.append(os.path.join(DATA_DIR, BIG_FN))
        if SMALL_FN and SMALL_FN != BIG_FN:
            candidates.append(os.path.join(DATA_DIR, SMALL_FN))

        for p in candidates:
            abs_p = os.path.abspath(p)
            log(f"  Try: {abs_p}")
            if not os.path.exists(p):
                continue
            try:
                df_raw = pd.read_csv(p)
                used_path = abs_p
                log(f"    Loaded {os.path.basename(p)} -> {df_raw.shape}")
                break
            except Exception as e:
                log(f"    Error reading {abs_p}: {e}")

        if df_raw is None or df_raw.empty:
            msg = (
                "Failed to load dataset. Checked the following paths:\n  "
                + "\n  ".join(os.path.abspath(c) for c in candidates)
            )
            raise FileNotFoundError(msg)

        if used_path:
            log(f"  [Info] Using dataset from: {used_path}")

    # At this point df_raw is guaranteed non-None
    raw_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_01_raw.csv")
    df_raw.to_csv(raw_csv, index=False)
    log(f"  Saved: {raw_csv}")

    df_raw, GWL_COL = normalize_gwl_alias(
        df_raw,
        GWL_COL,
        prefer_depth_bgs=True,
    )
    
    if SUBSIDENCE_COL not in df_raw.columns:
        if SUBSIDENCE_COL.endswith("_cum") and (
            "subsidence" in df_raw.columns
        ):
            df_raw = rate_to_cumulative(
                df_raw,
                rate_col="subsidence",
                cum_col=SUBSIDENCE_COL,
                time_col=TIME_COL,
                group_cols=(LON_COL, LAT_COL, "city"),
                initial="first_equals_rate_dt",
                inplace=False,
            )
            log(
                f"  [Subs] Built cumulative '{SUBSIDENCE_COL}' "
                "from 'subsidence'."
            )
        elif (SUBSIDENCE_COL == "subsidence") and (
            "subsidence_cum" in df_raw.columns
        ):
            df_raw = cumulative_to_rate(
                df_raw,
                cum_col="subsidence_cum",
                rate_col="subsidence",
                time_col=TIME_COL,
                group_cols=(LON_COL, LAT_COL, "city"),
                first="cum_over_dtref",
                inplace=False,
            )
            log("  [Subs] Built rate 'subsidence' from 'subsidence_cum'.")
        else:
            raise ValueError(
                f"SUBSIDENCE_COL={SUBSIDENCE_COL!r} not in df and no "
                "convertible alternative was found."
            )

    
    # Resolve optional future features (tuples / alternatives)
    FUTURE_DRIVER_FEATURES, _ = _resolve_optional_columns(
        df_raw,
        FUTURE_DRIVER_FEATURES,
    )

    _maybe_stop("before preprocessing")
    # ===================== STEP 2: PREPROCESS =====================
    log(f"\n{'='*18} Step 2: Initial Preprocessing {'='*18}")
    
    opt_num_cols, opt_num_map = _resolve_optional_columns(
        df_raw, OPTIONAL_NUMERIC_FEATURES
    )
    opt_cat_cols, opt_cat_map = _resolve_optional_columns(
        df_raw, OPTIONAL_CATEGORICAL_FEATURES
    )
    
    GWL_METERS_COL, GWL_ZSCORE_COL = resolve_gwl_for_physics(
        df_raw,
        gwl_col_user=GWL_COL,
        prefer_depth_bgs=True,
        allow_keep_zscore_as_ml=True,
    )
    
    GWL_DEPTH_COL = GWL_METERS_COL
    
    HEAD_SRC_COL, Z_SURF_COL_USED = resolve_head_column(
        df_raw,
        depth_col=GWL_DEPTH_COL,
        head_col=HEAD_COL,
        z_surf_col=Z_SURF_COL,
        use_head_proxy=bool(USE_HEAD_PROXY),
    )
    
    _pde_mode = str(cfg.get("PDE_MODE_CONFIG", "off")).strip().lower()
    _physics_requested = _pde_mode not in ("off", "none", "no", "0")
    
    if (GWL_METERS_COL is None) and _physics_requested:
        log(
            "  [Warn] No groundwater-level column in meters resolved. "
            "Disabling physics (PDE_MODE_CONFIG='off') for this run."
        )
        cfg["PDE_MODE_CONFIG"] = "off"
        _physics_requested = False
    
    if GWL_METERS_COL is None:
        if (GWL_ZSCORE_COL is not None) and (GWL_ZSCORE_COL in df_raw.columns):
            GWL_METERS_COL = GWL_ZSCORE_COL
        elif (GWL_COL is not None) and (GWL_COL in df_raw.columns):
            GWL_METERS_COL = GWL_COL
        else:
            raise ValueError(
                "No usable groundwater column found. Provide at least "
                f"one of: {GWL_COL!r}, {GWL_ZSCORE_COL!r}, or meters GWL."
            )
    
    if H_FIELD_COL_NAME not in df_raw.columns:
        _thick_candidates = [
            "soil_thickness_eff",
            "soil_thickness_imputed",
            "soil_thickness",
            "H_field",
            "soil_thickness_m",
        ]
        _found = next(
            (c for c in _thick_candidates if c in df_raw.columns),
            None,
        )
        if _found is not None:
            log(
                f"  [Info] Using thickness column {_found!r} "
                f"as H_FIELD_COL_NAME (was {H_FIELD_COL_NAME!r})."
            )
            H_FIELD_COL_NAME = _found
        else:
            if _physics_requested:
                log(
                    "  [Warn] No thickness column found; disabling physics "
                    "(PDE_MODE_CONFIG='off') for this run."
                )
                cfg["PDE_MODE_CONFIG"] = "off"
                _physics_requested = False
            _H_default = float(cfg.get("H_FIELD_DEFAULT_VALUE", 1.0))
            log(
                f"  [Warn] No thickness column found; creating constant "
                f"{H_FIELD_COL_NAME!r}={_H_default} (m)."
            )
            df_raw[H_FIELD_COL_NAME] = _H_default
    
    GWL_SRC_COL = GWL_METERS_COL
    
    base_select = [
        LON_COL,
        LAT_COL,
        TIME_COL,
        SUBSIDENCE_COL,
        GWL_METERS_COL,
        H_FIELD_COL_NAME,
    ]
    
    if Z_SURF_COL and (Z_SURF_COL in df_raw.columns):
        base_select.append(Z_SURF_COL)
    if HEAD_COL and (HEAD_COL in df_raw.columns):
        base_select.append(HEAD_COL)
    
    if (GWL_ZSCORE_COL is not None) and (GWL_ZSCORE_COL in df_raw.columns):
        base_select.append(GWL_ZSCORE_COL)
    
    base_select += opt_num_cols + opt_cat_cols
    
    selected = _drop_missing_keep_order(base_select, df_raw)

    missing_required = [
        c for c in [
            LON_COL, LAT_COL, TIME_COL,
            SUBSIDENCE_COL, GWL_COL, H_FIELD_COL_NAME,
        ] if c not in selected
    ]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    skipped_optional = sorted(set(opt_num_cols + opt_cat_cols) - set(selected))
    if skipped_optional:
        log(f"  [Info] Optional columns not found (skipped): {skipped_optional}")

    df_sel = df_raw[selected].copy()
    DT_TMP = "datetime_temp"
    try:
        df_sel[DT_TMP] = pd.to_datetime(df_sel[TIME_COL], format="%Y")
    except Exception:
        df_sel = normalize_time_column(
            df_sel,
            time_col=TIME_COL,
            datetime_col=DT_TMP,
            year_col=TIME_COL,
            drop_orig=True,
        )

    log(f"  Shape after select: {df_sel.shape}")
    log(f"  NaNs before clean: {df_sel.isna().sum().sum()}")
    df_clean = nan_ops(df_sel, ops="sanitize", action="fill", verbose=0)
    log(f"  NaNs after clean : {df_clean.isna().sum().sum()}")

    clean_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_02_clean.csv")
    df_clean.to_csv(clean_csv, index=False)
    log(f"  Saved: {clean_csv}")

    log(f"\n{'='*18} Step 2.5: Censor-aware transforms {'='*18}")
    df_cens, censor_report = _apply_censoring(df_clean.copy(), CENSORING_SPECS)

    _progress(0.10, "Stage-1: CSV loaded & cleaned")
    
    _maybe_stop("after cleaning & censoring")
    # ===================== STEP 3: ENCODE & SCALE =====================
    log(f"\n{'='*18} Step 3: Encode & Scale {'='*18}")
    
    df_proc = df_cens.copy()
    
    # ------------------------------------------------------------------
    # 3.0 Choose thickness source column (raw vs effective)
    # ------------------------------------------------------------------
    H_FIELD_SRC_COL = H_FIELD_COL_NAME
    for sp in (CENSORING_SPECS or []):
        if sp.get("col") == H_FIELD_COL_NAME:
            eff = H_FIELD_COL_NAME + sp.get("eff_suffix", "_eff")
            if USE_EFFECTIVE_H_FIELD and (eff in df_proc.columns):
                H_FIELD_SRC_COL = eff
            break
    
    # ------------------------------------------------------------------
    # 3.1 Create explicit SI columns (meters) for physics-critical channels
    # ------------------------------------------------------------------
    _missing_src = [
        c
        for c in (
            SUBSIDENCE_COL,
            GWL_SRC_COL,
            H_FIELD_SRC_COL,
        )
        if c not in df_proc.columns
    ]
    if _missing_src:
        raise ValueError(f"Missing required columns: {_missing_src}")
    
    SUBS_SI_COL = f"{SUBSIDENCE_COL}__si"
    DEPTH_SI_COL = f"{GWL_DEPTH_COL}__si"
    HEAD_SI_COL = f"{HEAD_SRC_COL}__si"
    H_SI_COL = f"{H_FIELD_SRC_COL}__si"
    Z_SURF_SI_COL = (
        f"{Z_SURF_COL_USED}__si" if Z_SURF_COL_USED else None
    )
    
    df_proc[SUBS_SI_COL] = (
        pd.to_numeric(df_proc[SUBSIDENCE_COL], errors="coerce")
        .astype(float)
        * float(SUBS_UNIT_TO_SI)
    ).astype(np.float32)
    
    df_proc[DEPTH_SI_COL] = (
        pd.to_numeric(df_proc[GWL_DEPTH_COL], errors="coerce")
        .astype(float)
        * float(HEAD_UNIT_TO_SI)
    ).astype(np.float32)
    
    df_proc[HEAD_SI_COL] = (
        pd.to_numeric(df_proc[HEAD_SRC_COL], errors="coerce")
        .astype(float)
        * float(HEAD_UNIT_TO_SI)
    ).astype(np.float32)
    
    df_proc[H_SI_COL] = (
        pd.to_numeric(df_proc[H_FIELD_SRC_COL], errors="coerce")
        .astype(float)
        * float(THICKNESS_UNIT_TO_SI)
    ).astype(np.float32)
    
    df_proc[H_SI_COL] = np.clip(
        df_proc[H_SI_COL].to_numpy(np.float32),
        float(H_MIN_SI),
        np.inf,
    )
    
    if Z_SURF_SI_COL:
        df_proc[Z_SURF_SI_COL] = (
            pd.to_numeric(df_proc[Z_SURF_COL_USED], errors="coerce")
            .astype(float)
            * float(Z_SURF_UNIT_TO_SI)
        ).astype(np.float32)
    
    SUBS_MODEL_COL = SUBS_SI_COL
    GWL_DYN_COL = DEPTH_SI_COL
    GWL_TARGET_COL = HEAD_SI_COL
    H_FIELD_COL = H_SI_COL

    _phys_cols = [SUBS_SI_COL, DEPTH_SI_COL, HEAD_SI_COL, H_SI_COL]
    bad = {
        c: int((~np.isfinite(df_proc[c].to_numpy())).sum())
        for c in _phys_cols
    }
    if any(v > 0 for v in bad.values()):
        raise ValueError(
            f"[Stage1] Non-finite values in physics SI columns: {bad}"
        )

    units_provenance = {
        "subs_src": SUBSIDENCE_COL,
        "subs_si": SUBS_SI_COL,
        "depth_src": GWL_DEPTH_COL,
        "depth_si": DEPTH_SI_COL,
        "head_src": HEAD_SRC_COL,
        "head_si": HEAD_SI_COL,
        "h_field_src": H_FIELD_SRC_COL,
        "h_field_si": H_SI_COL,
        "z_surf_src": Z_SURF_COL_USED,
        "z_surf_si": Z_SURF_SI_COL,
        "subs_unit_to_si": SUBS_UNIT_TO_SI,
        "head_unit_to_si": HEAD_UNIT_TO_SI,
        "thickness_unit_to_si": THICKNESS_UNIT_TO_SI,
        "z_surf_unit_to_si": Z_SURF_UNIT_TO_SI,
        "h_min_si": H_MIN_SI,
        "gwl_kind": GWL_KIND,
        "gwl_sign": GWL_SIGN,
        "use_head_proxy": USE_HEAD_PROXY,
    }
    
    # ------------------------------------------------------------------
    # 3.2 Coordinates: degrees -> projected meters (optional)
    # ------------------------------------------------------------------
    coords_in_degrees = COORD_MODE == "degrees"
    coord_epsg_used = None
    
    if coords_in_degrees:
        try:
            from pyproj import Transformer  # local import
        except Exception as e:
            raise ImportError(
                "pyproj is required for COORD_MODE='degrees'."
            ) from e
    
        transformer = Transformer.from_crs(
            COORD_SRC_EPSG,
            COORD_TARGET_EPSG,
            always_xy=True,
        )
        x_m, y_m = transformer.transform(
            df_proc[LON_COL].to_numpy(),
            df_proc[LAT_COL].to_numpy(),
        )
        df_proc[COORD_X_COL] = x_m.astype(float)
        df_proc[COORD_Y_COL] = y_m.astype(float)
        coord_epsg_used = COORD_TARGET_EPSG
    else:
        if COORD_X_COL not in df_proc.columns:
            df_proc[COORD_X_COL] = df_proc[LON_COL].astype(float)
        if COORD_Y_COL not in df_proc.columns:
            df_proc[COORD_Y_COL] = df_proc[LAT_COL].astype(float)
        coord_epsg_used = COORD_SRC_EPSG
    
    # Heuristic degree->meter scale factors (for debug/audit only)
    deg2m_x = deg_to_m_from_lat(
        df_proc[LAT_COL].astype(float).to_numpy()
    ) if coords_in_degrees else None
    deg2m_y = 111_320.0 if coords_in_degrees else None
    
    # ------------------------------------------------------------------
    # 3.3 Normalize time -> numeric coord
    # ------------------------------------------------------------------
    DT_TMP = "datetime_temp"
    df_proc[DT_TMP] = pd.to_datetime(df_proc[TIME_COL], format="%Y")
    
    df_proc = normalize_time_column(
        df_proc,
        time_col=TIME_COL,
        datetime_col=DT_TMP,
        require_full_year=True,
        raise_on_error=True,
    )
    
    TIME_COL_NUM = f"{TIME_COL}_numeric_coord"
    df_proc[TIME_COL_NUM] = (
        df_proc[DT_TMP].dt.year
        + (df_proc[DT_TMP].dt.dayofyear - 1)
        / (365 + df_proc[DT_TMP].dt.is_leap_year.astype(int))
    )
    
    # ------------------------------------------------------------------
    # 3.4 Build effective coord columns (shift optional)
    # ------------------------------------------------------------------
    TIME_COL_USED = TIME_COL_NUM
    X_COL_USED = COORD_X_COL
    Y_COL_USED = COORD_Y_COL
    
    coord_shift_pack = None
    
    if KEEP_COORDS_RAW and SHIFT_RAW_COORDS:
        pack = make_txy_coords(
            t=df_proc[TIME_COL_NUM].to_numpy(float),
            x=df_proc[COORD_X_COL].to_numpy(float),
            y=df_proc[COORD_Y_COL].to_numpy(float),
            time_shift="min",
            xy_shift="min",
            dtype="float32",
        )
        df_proc[TIME_COL_NUM + "__shift"] = pack.coords[:, 0]
        df_proc[COORD_X_COL + "__shift"] = pack.coords[:, 1]
        df_proc[COORD_Y_COL + "__shift"] = pack.coords[:, 2]
    
        TIME_COL_USED = TIME_COL_NUM + "__shift"
        X_COL_USED = COORD_X_COL + "__shift"
        Y_COL_USED = COORD_Y_COL + "__shift"
    
        coord_shift_pack = pack

    # ------------------------------------------------------------------
    # 3.5 OneHotEncode categorical features (keep your existing OHE code,
    #     but do it on df_proc now)
    # ------------------------------------------------------------------
    # NOTE: reuse your current OHE code here, but read from df_proc
    # and set encoded_names accordingly.
    # # 3.1 OHE
    encoded_names: list[str] = []
    ohe_paths: dict[str, str] = {}
    for cat_col in _drop_missing_keep_order(opt_cat_cols, df_proc):
        ohe = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.float32,
        )
        enc = ohe.fit_transform(df_proc[[cat_col]])
        enc_cols = ohe.get_feature_names_out([cat_col]).tolist()
        encoded_names.extend(enc_cols)
        enc_df = pd.DataFrame(enc, columns=enc_cols, index=df_proc.index)
        df_proc = pd.concat([df_proc.drop(columns=[cat_col]), enc_df], axis=1)

        path = os.path.join(ARTIFACTS_DIR, f"{CITY_NAME}_ohe_{cat_col}.joblib")
        try:
            save_job(ohe, path, append_versions=True, append_date=True)
        except Exception:
            joblib.dump(ohe, path)
        ohe_paths[cat_col] = path
        log(f"  Saved OHE for '{cat_col}': {path}")
    # ------------------------------------------------------------------
    # 3.6 Scale ML numeric features on TRAIN ONLY (never physics locked)
    #
    # No leakage rule:
    #   - Fit scaler on TRAIN split only (<= TRAIN_END_YEAR)
    #   - Transform the full table after fitting
    # ------------------------------------------------------------------
    # # 3.3 scaling
    censor_numeric_additions: list[str] = []
    for sp in CENSORING_SPECS or []:
        col = sp["col"]
        feff  = col + sp.get("eff_suffix",  "_eff")
        fflag = col + sp.get("flag_suffix", "_censored")
        if feff in df_proc.columns:
            censor_numeric_additions.append(feff)
        if INCLUDE_CENSOR_FLAGS_AS_DYNAMIC and fflag in df_proc.columns:
            censor_numeric_additions.append(fflag)
            
    # Keep a single "gwl_model" for backward compatibility: treat it as TARGET
    GWL_MODEL_COL  = GWL_TARGET_COL
    
    ml_numeric_candidates = []
    ml_numeric_candidates += opt_num_cols[:]
    ml_numeric_candidates += _drop_missing_keep_order(
        censor_numeric_additions,
        df_proc,
    )
    
    physics_locked = {
        # targets/physics channels (SI == model)
        SUBS_MODEL_COL,
        GWL_MODEL_COL,
        H_FIELD_COL,
        SUBS_SI_COL,
        DEPTH_SI_COL,
        HEAD_SI_COL,
        H_SI_COL,
    
        # time/coords (raw/proj + possibly shifted)
        TIME_COL_NUM,
        COORD_X_COL,
        COORD_Y_COL,
        TIME_COL_USED,
        X_COL_USED,
        Y_COL_USED,
    
        # raw columns used to build SI
        SUBSIDENCE_COL,
        GWL_DEPTH_COL,
        HEAD_SRC_COL,
        H_FIELD_SRC_COL,
    
        # bookkeeping columns you don't want scaled
        TIME_COL,
        DT_TMP,
        LON_COL,
        LAT_COL,
    }
    
    if Z_SURF_SI_COL:
        physics_locked.add(Z_SURF_SI_COL)
    
    ml_numeric_cols = [
        c for c in _drop_missing_keep_order(
            ml_numeric_candidates,
            df_proc,
        )
        if c not in physics_locked
    ]
    
    ml_numeric_cols = [
        c for c in ml_numeric_cols
        if c not in set(ALREADY_NORMALIZED_FEATURES or [])
    ]
    
    df_scaled = df_proc.copy()
    scaler_path = None
    
    train_mask = df_proc[TIME_COL].astype(float) <= float(TRAIN_END_YEAR)
    if not np.any(train_mask):
        raise ValueError(
            f"No rows satisfy training mask: {TIME_COL} "
            f"<= {TRAIN_END_YEAR}"
        )
    
    if ml_numeric_cols:
        scaler = MinMaxScaler()
    
        scaler.fit(df_proc.loc[train_mask, ml_numeric_cols])
        df_scaled.loc[:, ml_numeric_cols] = scaler.transform(
            df_scaled.loc[:, ml_numeric_cols]
        )
    
        scaler_path = os.path.join(
            ARTIFACTS_DIR,
            f"{CITY_NAME}_main_scaler.joblib",
        )
        joblib.dump(scaler, scaler_path)
    
        log(f"  Saved scaler (fit on train only): {scaler_path}")
        log(f"  Scaled ML numeric cols: {ml_numeric_cols}")
    
    else:
        scaler = None
        log("  [Info] No ML numeric features to scale.")
    
    # --------------------------------------------------------------
    # Stage-1 scaler_info (for Stage-2 metrics inversion/debugging)
    # --------------------------------------------------------------
    scaler_info = {}
    if scaler_path and ml_numeric_cols:
        for j, feat in enumerate(ml_numeric_cols):
            scaler_info[feat] = {
                "scaler_path": scaler_path,
                "all_features": list(ml_numeric_cols),
                "idx": int(j),
            }
    
    scaled_ml_numeric_cols = ml_numeric_cols[:]
    
    scaled_csv = os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_03_scaled.csv",
    )
    df_scaled.to_csv(scaled_csv, index=False)
    log(f"  Saved: {scaled_csv}")
    
    _maybe_stop("after encode & scale")
    _progress(0.30, "Stage-1: features scaled")

    # ===================== STEP 4: FEATURE SETS =====================
    log(f"\n{'='*18} Step 4: Define Feature Sets {'='*18}")


    static_features = list(encoded_names or [])
    if INCLUDE_Z_SURF_AS_STATIC and (Z_SURF_SI_COL is not None):
        static_features.append(Z_SURF_SI_COL)
    
    dynamic_features = [GWL_DYN_COL]
    if INCLUDE_SUBS_HIST_DYNAMIC:
        dynamic_features.append(SUBS_MODEL_COL)
    
    for c in OPTIONAL_NUMERIC_FEATURES:
        if c in df_scaled.columns and (c not in dynamic_features):
            if c not in (FUTURE_DRIVER_FEATURES or []):
                dynamic_features.append(c)
    
    future_features = []
    for c in (FUTURE_DRIVER_FEATURES or []):
        if c in df_scaled.columns:
            future_features.append(c)
    
    # censor flags
    if CENSORING_SPECS:
        flag_cols = [
            sp.get("flag") or f"{sp.get('col')}_censored"
            for sp in CENSORING_SPECS
        ]
        if INCLUDE_CENSOR_FLAGS_AS_DYNAMIC:
            for fcol in flag_cols:
                if fcol in df_scaled.columns:
                    dynamic_features.append(fcol)
        if INCLUDE_CENSOR_FLAGS_AS_FUTURE:
            for fcol in flag_cols:
                if fcol in df_scaled.columns:
                    future_features.append(fcol)
    
    # optional GUI overrides: append only (do not reorder base)
    for c in (STATIC_DRIVER_FEATURES or []):
        if c in df_scaled.columns and (c not in static_features):
            static_features.append(c)
    
    for c in (DYNAMIC_DRIVER_FEATURES or []):
        if c in df_scaled.columns and (c not in dynamic_features):
            dynamic_features.append(c)
    
    # indices spec for Stage-2
    gwl_dyn_index = dynamic_features.index(GWL_DYN_COL)
    subs_dyn_index = (
        dynamic_features.index(SUBS_MODEL_COL)
        if SUBS_MODEL_COL in dynamic_features
        else None
    )
    z_surf_static_index = (
        static_features.index(Z_SURF_SI_COL)
        if (Z_SURF_SI_COL is not None and Z_SURF_SI_COL in static_features)
        else None
    )

    # Then decide whether to use effective H-field instead of raw
    H_FIELD_COL = H_FIELD_COL_NAME
    for sp in CENSORING_SPECS or []:
        if sp["col"] == H_FIELD_COL_NAME:
            eff = H_FIELD_COL_NAME + sp.get("eff_suffix", "_eff")
            if USE_EFFECTIVE_H_FIELD and eff in df_scaled.columns:
                H_FIELD_COL = eff
                break


    GROUP_ID_COLS = [LON_COL, LAT_COL]
    
    # Finally, add censor flags as dynamic if requested
    for sp in CENSORING_SPECS or []:
        fflag = sp["col"] + sp.get("flag_suffix", "_censored")
        if (
            INCLUDE_CENSOR_FLAGS_AS_DYNAMIC
            and fflag in df_scaled.columns
            and fflag not in dynamic_features
        ):
            dynamic_features.append(fflag)

    log(f"  Static : {static_features}")
    log(f"  Dynamic: {dynamic_features}")
    log(f"  Future : {future_features}")
    log(f"  H_field: {H_FIELD_COL}")

    _progress(0.35, "Stage-1: feature sets defined")
    
    _maybe_stop("after feature sets defined")
    # ===================== STEP 5: TRAIN SPLIT & SEQUENCES =====================
    log(f"\n{'='*18} Step 5: Train Split & Sequences {'='*18}")

    df_train = df_scaled[df_scaled[TIME_COL] <= TRAIN_END_YEAR].copy()
    if df_train.empty:
        raise ValueError(f"Empty train split at year={TRAIN_END_YEAR}.")

    seq_job = os.path.join(
        ARTIFACTS_DIR,
        f"{CITY_NAME}_train_sequences_T{TIME_STEPS}_H{FORECAST_HORIZON_YEARS}.joblib",  # noqa: E501
    )
    
    _maybe_stop("before train sequence generation")
    _progress(0.40, "Stage-1: data sequences preparing...")
    OUT_S_DIM, OUT_G_DIM = 1, 1
    inputs_train, targets_train, coord_scaler = prepare_pinn_data_sequences(
        df=df_train,
        time_col=TIME_COL_USED,
        lon_col=X_COL_USED,
        lat_col=Y_COL_USED,
        subsidence_col=SUBS_MODEL_COL,
        gwl_col=GWL_TARGET_COL,
        gwl_dyn_col=GWL_DYN_COL,
        h_field_col=H_FIELD_COL,
        dynamic_cols=dynamic_features,
        static_cols=static_features,
        future_cols=future_features,
        group_id_cols=GROUP_ID_COLS,
        time_steps=TIME_STEPS,
        forecast_horizon=FORECAST_HORIZON_YEARS,
        output_subsidence_dim=OUT_S_DIM,
        output_gwl_dim=OUT_G_DIM,
        normalize_coords=bool(NORMALIZE_COORDS),
        fit_coord_scaler=bool(NORMALIZE_COORDS),
        savefile=seq_job,
        return_coord_scaler=True,
        mode=MODE,
        model=MODEL_NAME,
        verbose=2,
        _logger=log,
        stop_check=stop_check,
        progress_hook=_seq_progress_train,
    )
    coord_scaler_path = None
    if coord_scaler is not None:
        coord_scaler_path = os.path.join(
            ARTIFACTS_DIR,
            f"{CITY_NAME}_coord_scaler.joblib",
        )
        joblib.dump(coord_scaler, coord_scaler_path)
        log(f"  Saved coord scaler: {coord_scaler_path}")

    if coord_scaler is not None:
        cmin = coord_scaler.data_min_
        cmax = coord_scaler.data_max_
        coord_ranges = {
            "t": float(cmax[0] - cmin[0]),
            "x": float(cmax[1] - cmin[1]),
            "y": float(cmax[2] - cmin[2]),
        }
    else:
        coord_ranges = {
            "t": float(df_train[TIME_COL_USED].max() - df_train[TIME_COL_USED].min()),
            "x": float(df_train[X_COL_USED].max() - df_train[X_COL_USED].min()),
            "y": float(df_train[Y_COL_USED].max() - df_train[Y_COL_USED].min()),
        }
    
    if should_audit(AUDIT_STAGES, "stage1"):
        audit = audit_stage1_scaling(
            df_train=df_train,
            df_all=df_scaled,
            city=CITY_NAME,
            cfg=cfg,
            time_col=TIME_COL_USED,
            x_col=X_COL_USED,
            y_col=Y_COL_USED,
            normalize_coords=NORMALIZE_COORDS,
            keep_coords_raw=KEEP_COORDS_RAW,
            shift_raw_coords=SHIFT_RAW_COORDS,
            coords_in_degrees=coords_in_degrees,
            coord_epsg_used=coord_epsg_used,
            coord_ranges=coord_ranges,
            scaler_info=scaler_info,
        )
        audit_path = os.path.join(
            RUN_OUTPUT_PATH,
            "stage1_handshake_audit.json",
        )
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        log(f"[Audit] Saved Stage-1 audit -> {audit_path}")

    _progress(0.60, "Stage-1: train sequences built")
    _maybe_stop("after train sequences built")
    
    if targets_train["subsidence"].shape[0] == 0:
        raise ValueError("No training sequences were generated.")


    for k, v in inputs_train.items():
        log(f"  Train input '{k}': {None if v is None else v.shape}")
    for k, v in targets_train.items():
        log(f"  Train target '{k}': {v.shape}")

    # ===================== STEP 6: TF DATASETS & EXPORT =====================
    log(f"\n{'='*18} Step 6: TF Datasets & Export {'='*18}")

    num_train = inputs_train["dynamic_features"].shape[0]
    future_time_dim = (
        TIME_STEPS + FORECAST_HORIZON_YEARS
        if MODE == "tft_like"
        else FORECAST_HORIZON_YEARS
    )

    # Dense inputs (replace Nones with correctly-shaped zeros)
    static_arr = inputs_train.get("static_features")
    if static_arr is None:
        static_arr = np.zeros((num_train, 0), dtype=np.float32)

    future_arr = inputs_train.get("future_features")
    if future_arr is None:
        future_arr = np.zeros(
            (num_train, future_time_dim, 0), dtype=np.float32
        )

    dataset_inputs = {
        "coords": inputs_train["coords"],
        "dynamic_features": inputs_train["dynamic_features"],
        "static_features": static_arr,
        "future_features": future_arr,
        "H_field": inputs_train["H_field"],
    }
    dataset_targets = {
        "subs_pred": targets_train["subsidence"],
        "gwl_pred": targets_train["gwl"],
    }

    full_ds = tf.data.Dataset.from_tensor_slices(
        (dataset_inputs, dataset_targets)
    )
    val_size = int(0.2 * num_train)
    train_size = num_train - val_size

    full_ds = full_ds.shuffle(
        buffer_size=num_train, seed=42, reshuffle_each_iteration=False
    )
    train_ds = full_ds.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = full_ds.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)

    # Export to numpy so Stage-2 can just np.load(...)
    train_np_x, train_np_y = _dataset_to_numpy_pair(train_ds)
    val_np_x, val_np_y = _dataset_to_numpy_pair(val_ds)

    train_inputs_npz = os.path.join(ARTIFACTS_DIR, "train_inputs.npz")
    train_targets_npz = os.path.join(ARTIFACTS_DIR, "train_targets.npz")
    val_inputs_npz = os.path.join(ARTIFACTS_DIR, "val_inputs.npz")
    val_targets_npz = os.path.join(ARTIFACTS_DIR, "val_targets.npz")

    _save_npz(train_inputs_npz, train_np_x)
    _save_npz(train_targets_npz, train_np_y)
    _save_npz(val_inputs_npz, val_np_x)
    _save_npz(val_targets_npz, val_np_y)

    log(
        "  Saved NPZs:\n"
        f"    {train_inputs_npz}\n"
        f"    {train_targets_npz}\n"
        f"    {val_inputs_npz}\n"
        f"    {val_targets_npz}"
    )

    _progress(0.80, "Stage-1: validation sequences built")
    _maybe_stop("before TF dataset export")
    # ===================== MANIFEST: ONE FILE TO RULE THEM ALL =================
    log(f"\n{'='*18} Build Manifest {'='*18}")

    run_dir_abs = os.path.abspath(RUN_OUTPUT_PATH)
    artifacts_dir_abs = os.path.abspath(ARTIFACTS_DIR)
    base_output_abs = os.path.abspath(BASE_OUTPUT_DIR)

    # ===================== MANIFEST (v3.2) =====================
    log(f"\n{'='*18} Build Manifest {'='*18}")
    
    run_dir_abs = os.path.abspath(RUN_OUTPUT_PATH)
    artifacts_dir_abs = os.path.abspath(ARTIFACTS_DIR)
    base_output_abs = os.path.abspath(BASE_OUTPUT_DIR)
    
    
    # v3.2 censoring payload (keep keys stable)
    manifest_censor = {
        "specs": CENSORING_SPECS,
        "report": censor_report,
        "use_effective_h_field": USE_EFFECTIVE_H_FIELD,
        "flags_as_dynamic": INCLUDE_CENSOR_FLAGS_AS_DYNAMIC,
        "flags_as_future": INCLUDE_CENSOR_FLAGS_AS_FUTURE,
    }
    
    # v3.2: list of ML-scaled numeric columns (if any)
    # (If you use the v3.2 leak-safe scaler, this is ML-only.)
    scaled_ml_numeric_cols = list(scaled_ml_numeric_cols or [])
    
    # ------------------------ scaling_kwargs ------------------------
    scaling_kwargs = {
        "subs_scale_si": 1.0,
        "subs_bias_si": 0.0,
        "head_scale_si": 1.0,
        "head_bias_si": 0.0,
        "H_scale_si": 1.0,
        "H_bias_si": 0.0,
        "subsidence_kind": str(SUBSIDENCE_KIND),
        "allow_subs_residual": bool(ALLOW_SUBS_RESIDUAL),
        "coords_normalized": bool(NORMALIZE_COORDS),
        "coord_order": ["t", "x", "y"],
        "coord_ranges": coord_ranges,
        "coord_mode": COORD_MODE,
        "coord_src_epsg": COORD_SRC_EPSG,
        "coord_target_epsg": COORD_TARGET_EPSG,
        "coord_epsg_used": coord_epsg_used,
        "coords_in_degrees": bool(coords_in_degrees),
        "gwl_dyn_index": int(gwl_dyn_index),
        "subs_dyn_index": (
            None if subs_dyn_index is None else int(subs_dyn_index)
        ),
        "z_surf_static_index": (
            None
            if z_surf_static_index is None
            else int(z_surf_static_index)
        ),
        "time_units": str(TIME_UNITS),
        "cons_residual_units": CONSOLIDATION_RESIDUAL_UNITS,
        "cons_scale_floor": CONS_SCALE_FLOOR,
        "gw_scale_floor": GW_SCALE_FLOOR,
        "dt_min_units": DT_MIN_UNITS,
        "Q_wrt_normalized_time": Q_WRT_NORMALIZED_TIME,
        "Q_in_si": Q_IN_SI,
        "Q_in_per_second": Q_IN_PER_SECOND,
        "Q_kind": Q_KIND,
        "Q_length_in_si": Q_LENGTH_IN_SI,
        "drainage_mode": DRAINAGE_MODE,
        "scaling_error_policy": SCALING_ERROR_POLICY,
        "debug_physics_grads": DEBUG_PHYSICS_GRADS,
        "gw_residual_units": GW_RESIDUAL_UNITS,
        "clip_global_norm": CLIP_GLOBAL_NORM,
        "cons_drawdown_mode": str(CONS_DRAWDOWN_MODE).lower(),
        "cons_drawdown_rule": str(CONS_DRAWDOWN_RULE).lower(),
        "cons_stop_grad_ref": bool(CONS_STOP_GRAD_REF),
        "cons_drawdown_zero_at_origin": bool(
            CONS_DRAWDOWN_ZERO_AT_ORIGIN
        ),
        "cons_drawdown_clip_max": (
            None
            if CONS_DRAWDOWN_CLIP_MAX is None
            else float(CONS_DRAWDOWN_CLIP_MAX)
        ),
        "cons_relu_beta": float(CONS_RELU_BETA),
        "mv_prior_units": MV_PRIOR_UNITS,
        "mv_alpha_disp": MV_ALPHA_DISP,
        "mv_huber_delta": MV_HUBER_DELTA,
        "mv_prior_mode": MV_PRIOR_MODE,
        "mv_weight": MV_WEIGHT,
        "mv_schedule_unit": MV_SCHEDULE_UNIT,
        "mv_delay_epochs": int(MV_DELAY_EPOCHS),
        "mv_warmup_epochs": int(MV_WARMUP_EPOCHS),
        "mv_delay_steps": (
            None if MV_DELAY_STEPS is None else int(MV_DELAY_STEPS)
        ),
        "mv_warmup_steps": (
            None if MV_WARMUP_STEPS is None else int(MV_WARMUP_STEPS)
        ),
        "track_aux_metrics": TRACK_AUX_METRICS,
    }
    # Only attach degree→meter factors if
    #  coords are actually degrees
    if coords_in_degrees:
        # Use the *raw* latitude column, not 
        # y_used if y_used got shifted/normalized
        lat_ref_deg = float(
            np.nanmean(df_proc[LAT_COL].to_numpy(dtype=float)))
        deg_to_m_lon, deg_to_m_lat = deg_to_m_from_lat(lat_ref_deg)
        scaling_kwargs.update({
            "lat_ref_deg": lat_ref_deg,      # handy for audit/fallback
            "deg_to_m_lon": deg_to_m_lon,
            "deg_to_m_lat": deg_to_m_lat,
        })

    # Optional: shift metadata only if used (kept here because Stage-2 needs it)
    if coord_shift_pack is not None:
        scaling_kwargs["coord_shift_mins"] = coord_shift_pack.coord_mins
        scaling_kwargs["coord_shift_meta"] = coord_shift_pack.meta

    
    # ------------------------ manifest ------------------------
    manifest = {
        "schema_version": "3.2",
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": CITY_NAME,
        "model": MODEL_NAME,
        "stage": "stage1",
        
        "run": {
            "city": CITY_NAME,
            "model": MODEL_NAME,
            "stage": 1,
            "created_at": dt.datetime.now().isoformat(),
        },
        "paths": {
            "base_output_dir": base_output_abs,
            "run_dir": run_dir_abs,
            "artifacts_dir": artifacts_dir_abs,
        },
        "config": {
            "TIME_STEPS": TIME_STEPS,
            "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
            "MODE": MODE,
            "TRAIN_END_YEAR": TRAIN_END_YEAR,
            "FORECAST_START_YEAR": FORECAST_START_YEAR,
        },
        "cols_spec": {
            "time_raw": TIME_COL,
            "time_numeric": TIME_COL_NUM,
            "time_used": TIME_COL_USED,
            "lon_raw": LON_COL,
            "lat_raw": LAT_COL,
            "x_raw": COORD_X_COL,
            "y_raw": COORD_Y_COL,
            "x_used": X_COL_USED,
            "y_used": Y_COL_USED,
            "subs_raw": SUBSIDENCE_COL,
            "subs_model": SUBS_MODEL_COL,
            "gwl_raw": GWL_COL,
            "gwl_meters": GWL_METERS_COL,
            "gwl_depth_model": GWL_DYN_COL,
            "head_raw": HEAD_SRC_COL,
            "head_model": GWL_TARGET_COL,
            "h_field_raw": H_FIELD_SRC_COL,
            "h_field_model": H_FIELD_COL,
            "z_surf_raw": Z_SURF_COL_USED,
            "z_surf_model": Z_SURF_SI_COL,
        },
        "features_spec": {
            "static": static_features,
            "dynamic": dynamic_features,
            "future": future_features,
        },
        "indices_spec": {
            "gwl_dyn_index": int(gwl_dyn_index),
            "subs_dyn_index": subs_dyn_index,
            "z_surf_static_index": z_surf_static_index,
        },
        "conventions_spec": {
            "coords_in_degrees": bool(coords_in_degrees),
            "coord_epsg_used": coord_epsg_used,
            "deg_to_m": {
                "x": None if deg2m_x is None else "per_lat_vector",
                "y": deg2m_y,
            },
            "positive_z": "down",
            "positive_subsidence": "down",
            "units": {
                "subs": "meter",
                "head": "meter",
                "depth": "meter",
                "thickness": "meter",
                "coords": (
                    "meter"
                    if not coords_in_degrees
                    else "degree"
                ),
                "time": str(TIME_UNITS),
            },
        },
        "scaling_kwargs": scaling_kwargs,
        "units_provenance": units_provenance,
        "censoring": manifest_censor,
        "artifacts": {
            "tables": {
                "raw_csv": raw_csv,
                "clean_csv": clean_csv,
                "scaled_csv": scaled_csv,
            },
            "encoders": {
                "ohe_path": ohe_paths,
                "main_scaler_path": (
                    scaler_path if ml_numeric_cols else None
                ),
                "coord_scaler_path": coord_scaler_path,
            },
            "numpy": {
                "train_inputs": train_inputs_npz,
                "train_targets": train_targets_npz,
                "val_inputs": val_inputs_npz,
                "val_targets": val_targets_npz,
            },
            "meta": {
                "scaled_ml_numeric_cols": scaled_ml_numeric_cols,
                "scaler_info": scaler_info,
            },
        },
        
    "versions": {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
      },
    }


    manifest["config"]["feature_registry"] = {
        "optional_numeric_declared": OPTIONAL_NUMERIC_FEATURES,
        "optional_categorical_declared": OPTIONAL_CATEGORICAL_FEATURES,
        "already_normalized": ALREADY_NORMALIZED_FEATURES,
        "future_drivers_declared": FUTURE_DRIVER_FEATURES,
        "static_drivers_declared": STATIC_DRIVER_FEATURES,
        "dynamic_drivers_declared": DYNAMIC_DRIVER_FEATURES,
        "resolved_optional_numeric": opt_num_cols,
        "resolved_optional_categorical": opt_cat_cols,
    }
    manifest["config"]["censoring"] = manifest_censor

    # ===================== STEP 5b: TEST SEQUENCES (TEMPORAL GEN) ==============
    df_test = df_scaled[df_scaled[TIME_COL] >= FORECAST_START_YEAR].copy()
    if not df_test.empty:

        _maybe_stop("before test sequence generation")
        log(
            f"\n{'='*18} Step 5b: Test Sequences (temporal generalisation) "
            f"{'='*18}"
        )
        try:
            test_inputs, test_targets = prepare_pinn_data_sequences(
                df=df_test,
                time_col=TIME_COL_USED,
                lon_col=X_COL_USED,
                lat_col=Y_COL_USED,
                subsidence_col=SUBS_MODEL_COL,
                gwl_col=GWL_TARGET_COL,   
                gwl_dyn_col=GWL_DYN_COL,     
                h_field_col=H_FIELD_COL,
                dynamic_cols=dynamic_features,
                static_cols=static_features,
                future_cols=future_features,
                group_id_cols=GROUP_ID_COLS,
                time_steps=TIME_STEPS,
                forecast_horizon=FORECAST_HORIZON_YEARS,
                output_subsidence_dim=OUT_S_DIM,
                output_gwl_dim=OUT_G_DIM,
                normalize_coords=bool(NORMALIZE_COORDS),
                coord_scaler=coord_scaler,
                fit_coord_scaler=False,
            
                mode=MODE,
                model=MODEL_NAME,
                verbose=2,
            )

            if test_inputs:
                N = test_inputs["dynamic_features"].shape[0]
                fut_T = (
                    TIME_STEPS + FORECAST_HORIZON_YEARS
                    if MODE == "tft_like"
                    else FORECAST_HORIZON_YEARS
                )
                static_arr = (
                    test_inputs.get("static_features")
                    or np.zeros((N, 0), np.float32)
                )
                future_arr = (
                    test_inputs.get("future_features")
                    or np.zeros((N, fut_T, 0), np.float32)
                )

                test_inputs_np = {
                    "coords": test_inputs["coords"],
                    "dynamic_features": test_inputs["dynamic_features"],
                    "static_features": static_arr,
                    "future_features": future_arr,
                    "H_field": test_inputs["H_field"],
                }
                test_targets_np = {
                    "subsidence": test_targets["subsidence"],
                    "gwl": test_targets["gwl"],
                }

                test_inputs_npz = os.path.join(
                    ARTIFACTS_DIR, "test_inputs.npz"
                )
                test_targets_npz = os.path.join(
                    ARTIFACTS_DIR, "test_targets.npz"
                )
                _save_npz(test_inputs_npz, test_inputs_np)
                _save_npz(test_targets_npz, test_targets_np)

                manifest["artifacts"]["numpy"]["test_inputs_npz"] = (
                    test_inputs_npz
                )
                manifest["artifacts"]["numpy"]["test_targets_npz"] = (
                    test_targets_npz
                )
                log(
                    "  Saved TEST NPZs:\n"
                    f"    {test_inputs_npz}\n"
                    f"    {test_targets_npz}"
                )
            _maybe_stop("after test sequences")
        except Exception as e:  # pragma: no cover
            log(
                f"  [Warn] Test sequence construction failed: {e}. "
                "Continuing without test_* NPZs."
            )
            if stop_check and stop_check():
                raise InterruptedError("Sequence generation aborted.")
                
    # ===================== STEP 7: TRUE FUTURE SEQUENCES (OPTIONAL) ============
    future_npz_paths = None
    if BUILD_FUTURE_NPZ:
        _maybe_stop("before future sequence generation")
        log(f"\n{'='*18} Step 7: Future Sequences {'='*18}")
        try:
            future_npz_paths = build_future_sequences_npz(
            df_scaled=df_scaled,
            time_col=TIME_COL,
            time_col_num=TIME_COL_NUM,
            lon_col=COORD_X_COL,
            lat_col=COORD_Y_COL,
            subs_col=SUBS_MODEL_COL,
            gwl_col=GWL_TARGET_COL,
            h_field_col=H_FIELD_COL,
            static_features=static_features,
            dynamic_features=dynamic_features,
            future_features=future_features,
            group_id_cols=GROUP_ID_COLS,
            train_end_time=TRAIN_END_YEAR,
            forecast_start_time=FORECAST_START_YEAR,
            forecast_horizon=FORECAST_HORIZON_YEARS,
            time_steps=TIME_STEPS,
            mode=MODE,
            model_name=MODEL_NAME,
            artifacts_dir=ARTIFACTS_DIR,
            prefix="future",
            normalize_coords=bool(NORMALIZE_COORDS),
            coord_scaler=coord_scaler,
            verbose=7,
            logger=log,
            stop_check=stop_check,
            progress_hook=_seq_progress_future,
        )

            manifest["artifacts"]["numpy"].update(future_npz_paths)
            log(f"  Future NPZs: {future_npz_paths}")
        except Exception as e:  # pragma: no cover
            log(
                "[Warn] BUILD_FUTURE_NPZ=True but construction failed: "
                f"{e}\n"
                "       Continuing without future_* NPZs."
            )
        _maybe_stop("after future sequences")

    _maybe_stop("before manifest write")
    
    manifest_path = os.path.join(RUN_OUTPUT_PATH, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"  Saved manifest: {manifest_path}")
    log(
        f"\n{'-'*_TW}\nSTAGE-1 COMPLETE. Artifacts in:\n"
        f"  {run_dir_abs}\n{'-'*_TW}"
    )

    _progress(1.0, "Stage-1: complete")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "run_dir": run_dir_abs,
        "artifacts_dir": artifacts_dir_abs,
    }

