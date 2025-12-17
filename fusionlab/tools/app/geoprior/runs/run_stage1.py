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
    USE_EFFECTIVE_H_FIELD = bool(
        _censor_cfg.get("use_effective_h_field", cfg.get(
            "USE_EFFECTIVE_H_FIELD", True)
            )
    )

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

    _maybe_stop("before preprocessing")
    # ===================== STEP 2: PREPROCESS =====================
    log(f"\n{'='*18} Step 2: Initial Preprocessing {'='*18}")
    
    opt_num_cols, opt_num_map = _resolve_optional_columns(
        df_raw, OPTIONAL_NUMERIC_FEATURES
    )
    opt_cat_cols, opt_cat_map = _resolve_optional_columns(
        df_raw, OPTIONAL_CATEGORICAL_FEATURES
    )

    base_select = [
        LON_COL, LAT_COL, TIME_COL,
        SUBSIDENCE_COL, GWL_COL, H_FIELD_COL_NAME,
    ]
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

    # 3.1 OHE
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

    # 3.2 numeric time coord
    TIME_COL_NUM = f"{TIME_COL}_numeric_coord"
    df_proc[TIME_COL_NUM] = (
        df_proc[DT_TMP].dt.year
        + (df_proc[DT_TMP].dt.dayofyear - 1)
        / (365 + df_proc[DT_TMP].dt.is_leap_year.astype(int))
    )

    # 3.3 scaling
    censor_numeric_additions: list[str] = []
    for sp in CENSORING_SPECS or []:
        col = sp["col"]
        feff  = col + sp.get("eff_suffix",  "_eff")
        fflag = col + sp.get("flag_suffix", "_censored")
        if feff in df_proc.columns:
            censor_numeric_additions.append(feff)
        if INCLUDE_CENSOR_FLAGS_AS_DYNAMIC and fflag in df_proc.columns:
            censor_numeric_additions.append(fflag)

    present_num = _drop_missing_keep_order(
        [GWL_COL, H_FIELD_COL_NAME, SUBSIDENCE_COL] + opt_num_cols, df_proc
    ) + _drop_missing_keep_order(censor_numeric_additions, df_proc)
    num_cols = [c for c in present_num if c not in set(ALREADY_NORMALIZED_FEATURES)]

    df_scaled = df_proc.copy()
    scaler_info, scaler_path = {}, None
    if num_cols:
        scaler = MinMaxScaler()
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
        scaler_path = os.path.join(ARTIFACTS_DIR, f"{CITY_NAME}_main_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        log(f"  Saved scaler (joblib): {scaler_path}")

        targets_to_inv = {SUBSIDENCE_COL: SUBSIDENCE_COL, GWL_COL: GWL_COL}
        for base_name, col_in_scaler in targets_to_inv.items():
            if col_in_scaler in num_cols:
                scaler_info[base_name] = {
                    "scaler_path": scaler_path,
                    "all_features": num_cols,
                    "idx": num_cols.index(col_in_scaler),
                }

    scaled_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_03_scaled.csv")
    df_scaled.to_csv(scaled_csv, index=False)
    log(f"  Saved: {scaled_csv}")

    _maybe_stop("after encode & scale")

    _progress(0.30, "Stage-1: features scaled")

    # ===================== STEP 4: FEATURE SETS =====================
    log(f"\n{'='*18} Step 4: Define Feature Sets {'='*18}")

    # static_features = encoded_names[:]
    # dynamic_base = [GWL_COL]
    # dynamic_extra = [
    #     c for c in opt_num_cols if c not in {SUBSIDENCE_COL, H_FIELD_COL_NAME}
    # ]
    # dynamic_features = [
    #     c for c in dynamic_base + dynamic_extra if c in df_scaled.columns
    # ]
    # future_features = [
    #     c for c in FUTURE_DRIVER_FEATURES if c in df_scaled.columns
    # ]
    
    # First resolve base static / dynamic / future sets
    
    static_features, dynamic_features, future_features = _resolve_feature_sets(
        df_scaled=df_scaled,
        encoded_names=encoded_names,
        opt_num_cols=opt_num_cols,
        static_cfg=STATIC_DRIVER_FEATURES,
        dynamic_cfg=DYNAMIC_DRIVER_FEATURES,
        future_cfg=FUTURE_DRIVER_FEATURES,
        gwl_col=GWL_COL,
        subs_col=SUBSIDENCE_COL,
        h_field_col=H_FIELD_COL_NAME,
        log=log,
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
        time_col=TIME_COL_NUM,
        lon_col=LON_COL,
        lat_col=LAT_COL,
        subsidence_col=SUBSIDENCE_COL,
        gwl_col=GWL_COL,
        h_field_col=H_FIELD_COL,
        dynamic_cols=dynamic_features,
        static_cols=static_features,
        future_cols=future_features,
        group_id_cols=GROUP_ID_COLS,
        time_steps=TIME_STEPS,
        forecast_horizon=FORECAST_HORIZON_YEARS,
        output_subsidence_dim=OUT_S_DIM,
        output_gwl_dim=OUT_G_DIM,
        normalize_coords=True,
        savefile=seq_job,
        return_coord_scaler=True,
        mode=MODE,
        model=MODEL_NAME,
        verbose=2,
        _logger=log,          
        stop_check=stop_check , 
        progress_hook=_seq_progress_train,
    )
    _progress(0.60, "Stage-1: train sequences built")
    _maybe_stop("after train sequences built")
    
    if targets_train["subsidence"].shape[0] == 0:
        raise ValueError("No training sequences were generated.")

    coord_scaler_path = os.path.join(
        ARTIFACTS_DIR, f"{CITY_NAME}_coord_scaler.joblib"
    )
    joblib.dump(coord_scaler, coord_scaler_path)
    log(f"  Saved coord scaler: {coord_scaler_path}")

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

    manifest_censor = {
        "specs": CENSORING_SPECS,
        "report": censor_report,
        "use_effective_h_field": USE_EFFECTIVE_H_FIELD,
        "flags_as_dynamic": INCLUDE_CENSOR_FLAGS_AS_DYNAMIC,
    }

    manifest = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": CITY_NAME,
        "model": MODEL_NAME,
        "stage": "stage1",
        "config": {
            "TIME_STEPS": TIME_STEPS,
            "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
            "MODE": MODE,
            "TRAIN_END_YEAR": TRAIN_END_YEAR,
            "FORECAST_START_YEAR": FORECAST_START_YEAR,
            "cols": {
                "time": TIME_COL,
                "time_numeric": TIME_COL_NUM,
                "lon": LON_COL,
                "lat": LAT_COL,
                "subsidence": SUBSIDENCE_COL,
                "gwl": GWL_COL,
                "h_field": H_FIELD_COL,
            },
            "features": {
                "static": static_features,
                "dynamic": dynamic_features,
                "future": future_features,
                "group_id_cols": GROUP_ID_COLS,
            },
        },
        "artifacts": {
            "csv": {
                "raw": raw_csv,
                "clean": clean_csv,
                "scaled": scaled_csv,
            },
            "encoders": {
                "ohe": ohe_paths,            # {categorical_col: path}
                "main_scaler": scaler_path if num_cols else None,
                "coord_scaler": coord_scaler_path,
                "scaler_info": scaler_info,  # indices for inverse-transform
            },
            "sequences": {
                "joblib_train_sequences": seq_job,
                "dims": {
                    "output_subsidence_dim": OUT_S_DIM,
                    "output_gwl_dim": OUT_G_DIM,
                },
            },
            "numpy": {
                "train_inputs_npz": train_inputs_npz,
                "train_targets_npz": train_targets_npz,
                "val_inputs_npz": val_inputs_npz,
                "val_targets_npz": val_targets_npz,
            },
            "shapes": {
                "train_inputs": {
                    k: list(v.shape) for k, v in train_np_x.items()
                },
                "train_targets": {
                    k: list(v.shape) for k, v in train_np_y.items()
                },
                "val_inputs": {
                    k: list(v.shape) for k, v in val_np_x.items()
                },
                "val_targets": {
                    k: list(v.shape) for k, v in val_np_y.items()
                },
            },
        },
        "paths": {
            "run_dir": run_dir_abs,
            "artifacts_dir": artifacts_dir_abs,
            "results_root": base_output_abs,  
        },
        "versions": {
            "python": (
                f"{os.sys.version_info.major}."
                f"{os.sys.version_info.minor}."
                f"{os.sys.version_info.micro}"
            ),
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
            test_inputs, test_targets, _ = prepare_pinn_data_sequences(
                df=df_test,
                time_col=TIME_COL_NUM,
                lon_col=LON_COL,
                lat_col=LAT_COL,
                subsidence_col=SUBSIDENCE_COL,
                gwl_col=GWL_COL,
                h_field_col=H_FIELD_COL,
                dynamic_cols=dynamic_features,
                static_cols=static_features,
                future_cols=future_features,
                group_id_cols=GROUP_ID_COLS,
                time_steps=TIME_STEPS,
                forecast_horizon=FORECAST_HORIZON_YEARS,
                output_subsidence_dim=OUT_S_DIM,
                output_gwl_dim=OUT_G_DIM,
                normalize_coords=True,
                savefile=None,
                return_coord_scaler=False,
                mode=MODE,
                model=MODEL_NAME,
                verbose=2,
                _logger=log,
                stop_check=stop_check,
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
                lon_col=LON_COL,
                lat_col=LAT_COL,
                subs_col=SUBSIDENCE_COL,
                gwl_col=GWL_COL,
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
                verbose=7,
                logger=log, 
                stop_check = stop_check, 
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

