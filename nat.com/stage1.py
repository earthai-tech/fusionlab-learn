# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Stage-1: Zhongshan/Nansha preprocessing & sequence export for GeoPriorSubsNet

This script runs Steps 1–6 of the NATCOM pipeline:
  1) Load dataset
  2) Clean & select features
  3) Encode & scale
  4) Define feature sets
  5) Split by year & build PINN sequences
  6) Build train/val tf.data and EXPORT all arrays & metadata

Outputs
-------
- CSVs: raw, cleaned, scaled
- Joblib: one-hot encoder, main scaler, coord scaler
- NPZ: train_inputs, train_targets, val_inputs, val_targets
- JSON: manifest.json describing everything (paths, shapes, dims, columns,
  config), so Stage-2 can load without recomputing.

Stage-2 only needs to:
  - read manifest.json
  - np.load(...) the NPZs
  - joblib.load(...) the scalers/encoders if needed
  - build/compile/tune/train the model
"""

from __future__ import annotations

import os
import json
import shutil
import joblib
import warnings
import datetime as dt
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from fusionlab.api.util import get_table_size
from fusionlab.datasets import fetch_zhongshan_data
from fusionlab.nn.pinn import prepare_pinn_data_sequences
from fusionlab.utils import (
    should_audit,
    audit_stage1_scaling, 
    nan_ops,
    save_job, 
    unpack_frames_from_file, 
    load_nat_config,
    normalize_time_column,
    ensure_directory_exists,
    print_config_table,
    deg_to_m_from_lat,
    cumulative_to_rate,
    normalize_gwl_alias,
    rate_to_cumulative,
    resolve_gwl_for_physics,
    resolve_head_column,
    make_txy_coords,
    build_future_sequences_npz,
    compute_group_masks,
    split_groups_holdout,
    filter_df_by_groups,
    default_results_dir
)

# --- Suppress common warnings/tf chatter ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
if hasattr(tf, "autograph") and hasattr(tf.autograph, "set_verbosity"):
    tf.autograph.set_verbosity(0)

# ==================================================================
# Step 0: CONFIGURATION (centralized via load_nat_config)
# ==================================================================
 
cfg = load_nat_config()

# --- Core identifiers & paths ---
CITY_NAME   = cfg["CITY_NAME"]
MODEL_NAME  = cfg["MODEL_NAME"]
DATA_DIR    = cfg["DATA_DIR"]
BIG_FN      = cfg["BIG_FN"]
SMALL_FN    = cfg["SMALL_FN"]

# optional multi-city parquet (if exists and CSV are missing)
ALL_CITIES_PARQUET = cfg.get("ALL_CITIES_PARQUET")

# Allow SEARCH/FALLBACK paths to be specified in cfg; otherwise, use defaults
SEARCH_PATHS = cfg.get("SEARCH_PATHS") or [
    os.path.join(DATA_DIR, "data", BIG_FN),
    os.path.join(DATA_DIR, BIG_FN),
    os.path.join(".", "data", BIG_FN),
    BIG_FN,
]
FALLBACK_PATHS = cfg.get("FALLBACK_PATHS") or [
    os.path.join(DATA_DIR, "data", SMALL_FN),
    os.path.join(DATA_DIR, SMALL_FN),
    os.path.join(".", "data", SMALL_FN),
    SMALL_FN,
]

# where to look for the merged parquet (if configured)
ALL_CITIES_SEARCH_PATHS = cfg.get("ALL_CITIES_SEARCH_PATHS") or [
    os.path.join(DATA_DIR, "data", ALL_CITIES_PARQUET) if ALL_CITIES_PARQUET else None,
    os.path.join(DATA_DIR, ALL_CITIES_PARQUET)         if ALL_CITIES_PARQUET else None,
    os.path.join(".", "data", ALL_CITIES_PARQUET)      if ALL_CITIES_PARQUET else None,
    ALL_CITIES_PARQUET,
]
# drop Nones
ALL_CITIES_SEARCH_PATHS = [p for p in ALL_CITIES_SEARCH_PATHS if p]

# --- Time windows ---
TRAIN_END_YEAR          = cfg["TRAIN_END_YEAR"]
FORECAST_START_YEAR     = cfg["FORECAST_START_YEAR"]
FORECAST_HORIZON_YEARS  = cfg["FORECAST_HORIZON_YEARS"]
TIME_STEPS              = cfg["TIME_STEPS"]
MODE                    = cfg["MODE"]          # {'pihal_like', 'tft_like'}

# Optional: whether Stage-1 should also pre-build future_* NPZ for Stage-3
BUILD_FUTURE_NPZ = bool(cfg.get("BUILD_FUTURE_NPZ", False))

# --- Column names ---
TIME_COL         = cfg["TIME_COL"]
LON_COL          = cfg["LON_COL"]
LAT_COL          = cfg["LAT_COL"]
SUBSIDENCE_COL   = cfg["SUBSIDENCE_COL"]
GWL_COL          = cfg["GWL_COL"]
H_FIELD_COL_NAME = cfg["H_FIELD_COL_NAME"]     # Required by GeoPriorSubsNet

# --- Feature registry (global knobs) ---
OPTIONAL_NUMERIC_FEATURES     = cfg["OPTIONAL_NUMERIC_FEATURES"]
OPTIONAL_CATEGORICAL_FEATURES = cfg["OPTIONAL_CATEGORICAL_FEATURES"]
ALREADY_NORMALIZED_FEATURES   = cfg["ALREADY_NORMALIZED_FEATURES"]
FUTURE_DRIVER_FEATURES        = cfg["FUTURE_DRIVER_FEATURES"]

# --- Censoring config (v3.2 flat config; keep legacy fallback) ---
CENSORING_SPECS = cfg.get("CENSORING_SPECS", None)
if CENSORING_SPECS is None:
    _c = cfg.get("censoring", {}) or {}
    CENSORING_SPECS = _c.get("specs", []) or []

INCLUDE_CENSOR_FLAGS_AS_DYNAMIC = bool(
    cfg.get(
        "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC",
        (cfg.get("censoring", {}) or {}).get("flags_as_dynamic", True),
    )
)
INCLUDE_CENSOR_FLAGS_AS_FUTURE = bool(
    cfg.get(
        "INCLUDE_CENSOR_FLAGS_AS_FUTURE",
        (cfg.get("censoring", {}) or {}).get("flags_as_future", False),
    )
)
USE_EFFECTIVE_H_FIELD = bool(
    cfg.get(
        "USE_EFFECTIVE_H_FIELD",
        (cfg.get("censoring", {}) or {}).get("use_effective_h_field", True),
    )
)

THICKNESS_UNIT_TO_SI = float(cfg.get("THICKNESS_UNIT_TO_SI", 1.0))
HEAD_UNIT_TO_SI = float(cfg.get("HEAD_UNIT_TO_SI", 1.0))
H_MIN_SI = float(cfg.get("H_FIELD_MIN_SI", 0.1))  # meters

# v3.2 z_surf/head conventions
Z_SURF_COL = cfg.get("Z_SURF_COL", None)
USE_HEAD_PROXY = bool(cfg.get("USE_HEAD_PROXY", True))
Z_SURF_UNIT_TO_SI= float(cfg.get("Z_SURF_UNIT_TO_SI", 1.0))
HEAD_COL = cfg.get("HEAD_COL", "head_m")
INCLUDE_Z_SURF_AS_STATIC = bool(cfg.get("INCLUDE_Z_SURF_AS_STATIC", True))

TRACK_AUX_METRICS = bool(cfg.get("TRACK_AUX_METRICS", True))
# --- Stage-1 physics-critical scaling controls ---
# Preferred knob:
NORMALIZE_COORDS = bool(cfg.get("NORMALIZE_COORDS", True))

# Backward-compat knob (legacy):
KEEP_COORDS_RAW = bool(cfg.get("KEEP_COORDS_RAW", not NORMALIZE_COORDS))

# Canonical reconciliation: NORMALIZE_COORDS wins if present
if "NORMALIZE_COORDS" in cfg:
    if "KEEP_COORDS_RAW" in cfg:
        # if inconsistent, warn once (optional)
        if bool(cfg["KEEP_COORDS_RAW"]) == NORMALIZE_COORDS:
            print(
                "  [Warn] BOTH NORMALIZE_COORDS and KEEP_COORDS_RAW were set but "
                "are inconsistent. NORMALIZE_COORDS takes precedence."
            )
    KEEP_COORDS_RAW = not NORMALIZE_COORDS
else:
    NORMALIZE_COORDS = not KEEP_COORDS_RAW

# Only matters when KEEP_COORDS_RAW=True
SHIFT_RAW_COORDS = bool(cfg.get("SHIFT_RAW_COORDS", True))

SCALE_H_FIELD   = bool(cfg.get("SCALE_H_FIELD", False))  # recommended False
SCALE_GWL       = bool(cfg.get("SCALE_GWL", False))      # recommended False

SUBS_UNIT_TO_SI = float(cfg.get("SUBS_UNIT_TO_SI", 1e-3))  # mm -> m
SUBSIDENCE_KIND = str(cfg.get("SUBSIDENCE_KIND", "cumulative")).lower()

CONSOLIDATION_RESIDUAL_UNITS = str(
    cfg.get("CONSOLIDATION_RESIDUAL_UNITS", "second")
    ).lower()

GW_RESIDUAL_UNITS = str(cfg.get ("GW_RESIDUAL_UNITS", "time_unit"))
CLIP_GLOBAL_NORM = float(cfg.get("CLIP_GLOBAL_NORM", 5.0))

# Optional but strongly recommended if GWL_COL is not in meters:
GWL_RAW_COL = cfg.get("GWL_RAW_COL", None)  # e.g. "GWL_depth_bgs"

CONS_SCALE_FLOOR = cfg.get("CONS_SCALE_FLOOR", 1e-5) 
GW_SCALE_FLOOR = cfg.get("GW_SCALE_FLOOR", 1e-5)  
DT_MIN_UNITS = float(cfg.get("DT_MIN_UNITS", 1e-6))  

Q_WRT_NORMALIZED_TIME = bool(cfg.get("Q_WRT_NORMALIZED_TIME", False))     
Q_IN_SI = bool(cfg.get("Q_IN_SI", False))      
Q_IN_PER_SECOND  = bool(cfg.get("Q_IN_PER_SECOND", False))      
Q_KIND =str(cfg.get("Q_kind", "per_volume"))
Q_LENGTH_IN_SI = bool (cfg.get('Q_length_in_si', False))  
DRAINAGE_MODE = str(cfg.get("DRAINAGE_MODE", "double"))

SCALING_ERROR_POLICY = str(cfg.get("SCALING_ERROR_POLICY", "warn"))
DEBUG_PHYSICS_GRADS =bool(cfg.get('DEBUG_PHYSICS_GRADS', False ))
# Coordinate system for physics
COORD_MODE = str(cfg.get("COORD_MODE", "degrees")).lower()
UTM_EPSG = int(cfg.get("UTM_EPSG", 32649))
COORD_SRC_EPSG    = cfg.get("COORD_SRC_EPSG", None)      # required for degrees
COORD_TARGET_EPSG = int(cfg.get("COORD_TARGET_EPSG", UTM_EPSG))

COORD_X_COL, COORD_Y_COL = LON_COL, LAT_COL
coord_epsg = None
coords_in_degrees = bool(COORD_MODE == "degrees")

ALLOW_SUBS_RESIDUAL = bool (cfg.get("allow_subs_residual", True)) 

# AUDIT stages 
AUDIT_STAGES = cfg.get("AUDIT_STAGES", '*')

# Consolidation drawdown / equilibrium compaction controls
CONS_DRAWDOWN_MODE = str(cfg.get("CONS_DRAWDOWN_MODE", "smooth_relu")).lower()
CONS_DRAWDOWN_RULE = str(cfg.get("CONS_DRAWDOWN_RULE", "ref_minus_mean")).lower()

CONS_STOP_GRAD_REF = bool(cfg.get("CONS_STOP_GRAD_REF", True))
CONS_DRAWDOWN_ZERO_AT_ORIGIN = bool(cfg.get("CONS_DRAWDOWN_ZERO_AT_ORIGIN", False))

CONS_RELU_BETA = float(cfg.get("CONS_RELU_BETA", 20.0))

# Optional clip max: allow None / "none" / "" etc.
_cons_clip = cfg.get("CONS_DRAWDOWN_CLIP_MAX", None)
if _cons_clip is None:
    CONS_DRAWDOWN_CLIP_MAX = None
else:
    s = str(_cons_clip).strip().lower()
    if s in {"", "none", "null"}:
        CONS_DRAWDOWN_CLIP_MAX = None
    else:
        CONS_DRAWDOWN_CLIP_MAX = float(_cons_clip)

MV_PRIOR_UNITS = str(cfg.get("MV_PRIOR_UNITS", "auto") )
MV_ALPHA_DISP = float(cfg.get("MV_ALPHA_DISP", 0.1))
MV_HUBER_DELTA = float(cfg.get("MV_HUBER_DELTA", 1.0))    
  
MV_PRIOR_MODE =str(cfg.get("MV_PRIOR_MODE", "calibrate") )
MV_WEIGHT = float(cfg.get("MV_WEIGHT", 1e-3))  

MV_SCHEDULE_UNIT = str(cfg.get("MV_SCHEDULE_UNIT", "epoch")).strip().lower()

MV_DELAY_EPOCHS  = int(cfg.get("MV_DELAY_EPOCHS", 1))
MV_WARMUP_EPOCHS = int(cfg.get("MV_WARMUP_EPOCHS", 2))

# legacy / optional step-based keys
MV_DELAY_STEPS   = cfg.get("MV_DELAY_STEPS", None)
MV_WARMUP_STEPS  = cfg.get("MV_WARMUP_STEPS", None)

if MV_SCHEDULE_UNIT not in ("epoch", "step"):
    raise ValueError("MV_SCHEDULE_UNIT must be 'epoch' or 'step'.")

# If user selects step mode but didn't provide step counts, fall back to legacy default
if MV_SCHEDULE_UNIT == "step" and MV_WARMUP_STEPS is None:
    MV_WARMUP_STEPS = 4780 * 2  # legacy fallback, but prefer explicit in config.py
if MV_SCHEDULE_UNIT == "step" and MV_DELAY_STEPS is None:
    MV_DELAY_STEPS = 0

# --- Output directories (optionally overridable from cfg) ---
# BASE_OUTPUT_DIR = cfg.get("BASE_OUTPUT_DIR", os.path.join(os.getcwd(), "results"))
BASE_OUTPUT_DIR = cfg.get("BASE_OUTPUT_DIR")
if not BASE_OUTPUT_DIR:
    BASE_OUTPUT_DIR = default_results_dir(create=True)
    
ensure_directory_exists(BASE_OUTPUT_DIR)

RUN_OUTPUT_PATH = os.path.join(
    BASE_OUTPUT_DIR, f"{CITY_NAME}_{MODEL_NAME}_stage1"
)
if os.path.isdir(RUN_OUTPUT_PATH):
    print(f"Cleaning existing Stage-1 directory: {RUN_OUTPUT_PATH}")
    shutil.rmtree(RUN_OUTPUT_PATH)
os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)

ARTIFACTS_DIR = os.path.join(RUN_OUTPUT_PATH, "artifacts")
ensure_directory_exists(ARTIFACTS_DIR)

TIME_UNITS = str(cfg.get("TIME_UNITS", "year"))

try:
    _TW = get_table_size()
except Exception:
    _TW = 80

print(f"\n{'-'*_TW}\n{CITY_NAME.upper()} {MODEL_NAME} STAGE-1 (Steps 1–6)\n{'-'*_TW}")
print(f"TIME_STEPS={TIME_STEPS}, HORIZON={FORECAST_HORIZON_YEARS}, MODE={MODE}")

# ==================================================================
# Small helpers
# ==================================================================
def _save_npz(path: str, arrays: Dict[str, np.ndarray]) -> str:
    """Save dict of numpy arrays to compressed NPZ."""
    np.savez_compressed(path, **arrays)
    return path


def _dataset_to_numpy_pair(ds: tf.data.Dataset, limit: Optional[int] = None
                           ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Unbatch a (x,y) dataset into stacked numpy dicts.
    """
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

    # Stack by key
    x0 = xs[0]
    x_np = {k: np.stack([xi[k] for xi in xs], axis=0) for k in x0}
    y0 = ys[0]
    y_np = {k: np.stack([yi[k] for yi in ys], axis=0) for k in y0}
    return x_np, y_np

def _resolve_optional_columns(df: pd.DataFrame, spec_list) -> tuple[list[str], dict]:
    """
    From a list like ["a", ("b","b2"), "c"], return:
      present_cols: concrete columns found in df (choose first match in tuples)
      mapping: {chosen_col: original_spec} for traceability
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

def _apply_censoring(df: pd.DataFrame, specs: list[dict]) -> tuple[pd.DataFrame, dict]:
    """
    Add <col>_censored (bool) and <col>_eff (float) for each spec.
    Returns (df, report) with basic rates for the manifest.
    """
    report = {}
    for sp in specs or []:
        col = sp.get("col")
        if not col or col not in df.columns:
            continue

        cap  = sp.get("cap")
        tol  = float(sp.get("tol", 0.0))
        dirn = sp.get("direction", "right")
        fflag = col + sp.get("flag_suffix", "_censored")
        feff  = col + sp.get("eff_suffix",  "_eff")
        mode  = sp.get("eff_mode", "clip")
        eps   = float(sp.get("eps", 0.02))

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
            by  = imp.get("by", [])
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
    print("==> Censoring done ...")
    return df, report

# ==================================================================
# Step 1: Load dataset
# ==================================================================
print(f"\n{'='*18} Step 1: Load Dataset {'='*18}")

def _any_exists(paths: list[str]) -> bool:
    """Return True if any candidate path exists on disk."""
    return any(p and os.path.exists(p) for p in paths)

def _first_existing(paths: list[str]) -> Optional[str]:
    """Return the first existing path from a list, else None."""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

# ------------------------------------------------------------------
# 1.0 If the city CSV is missing, optionally unpack it from a merged
#     parquet that contains both cities.
# ------------------------------------------------------------------
if (not _any_exists(SEARCH_PATHS)) and ALL_CITIES_PARQUET:
    merged_path = _first_existing(ALL_CITIES_SEARCH_PATHS)

    if merged_path is not None:
        print(
            f"  [Info] No '{BIG_FN}' found for city={CITY_NAME!r}; "
            f"unpacking from merged parquet: {os.path.abspath(merged_path)}"
        )
        # This will create e.g.:
        #   'nansha_final_main_std.harmonized.csv'
        #   'zhongshan_final_main_std.harmonized.csv'
        # in the parquet folder (or whatever 'source' column provides).
        unpack_frames_from_file(
            merged=merged_path,
            group_col="city",
            output_dir=os.path.dirname(merged_path),
            output_format="csv",
            use_source_col=True,
            source_col="source",
            drop_columns=("source",),
            save=True,
            return_dict=False,
            verbose=1,
        )
    else:
        print(
            "  [Info] ALL_CITIES_PARQUET is set, but the merged parquet "
            "was not found in ALL_CITIES_SEARCH_PATHS.\n"
            "         Will proceed with normal CSV search / "
            "fetch_zhongshan_data fallback."
        )

# ------------------------------------------------------------------
# 1.1 Load CSV from SEARCH_PATHS then FALLBACK_PATHS; otherwise use the
#     fusionlab dataset fetcher.
# ------------------------------------------------------------------
df_raw = None
used_path = None

for p in (SEARCH_PATHS + FALLBACK_PATHS):
    if not p:
        continue
    print(f"  Try: {os.path.abspath(p)}")
    if os.path.exists(p):
        try:
            df_raw = pd.read_csv(p)
            used_path = p
            print(f"    Loaded {os.path.basename(p)} -> {df_raw.shape}")
            # soft sanity check: some 500k-named files are smaller
            if (BIG_FN in os.path.basename(p)) and (df_raw.shape[0] < 400_000):
                print("    [Warn] BIG_FN filename but rows < 400k (check you loaded the expected file).")
            break
        except Exception as e:
            print(f"    [Warn] Error reading {p}: {e}")

if df_raw is None or df_raw.empty:
    print("  [Info] No CSV found. Trying fusionlab.datasets.fetch_zhongshan_data() ...")
    try:
        bunch = fetch_zhongshan_data(verbose=1)
        df_raw = bunch.frame
        used_path = "<fetch_zhongshan_data>"
        print(f"    Loaded via fetch_zhongshan_data -> {df_raw.shape}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load dataset from CSV and fetcher: {e}")

# Save a copy of the raw loaded table for traceability
raw_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_01_raw.csv")
df_raw.to_csv(raw_csv, index=False)
print(f"  Saved: {raw_csv}")
print(f"  Source: {used_path}")

# ------------------------------------------------------------------
# 1.2 GWL alias normalization (do this ONCE, early, before Step 2)
#     Goal:
#       - If config uses 'GWL' but dataset provides 'GWL_depth_bgs', map it.
#       - If dataset uses 'GWL' but model expects meters 'GWL_depth_bgs', map it.
#     IMPORTANT: This is ONLY a naming fix. Unit conversion is handled
#     later when creating __si columns.
# ------------------------------------------------------------------
df_raw, GWL_COL = normalize_gwl_alias(
    df_raw,
    GWL_COL,
    prefer_depth_bgs=True,
)

# ==================================================================
# Step 2: Initial preprocessing (clean + select + censor)
# ==================================================================
print(f"\n{'='*18} Step 2: Initial Preprocessing {'='*18}")
avail = df_raw.columns.tolist()

# ------------------------------------------------------------------
# 2.0 Subsidence column handshake (rate <-> cumulative)
# ------------------------------------------------------------------
if SUBSIDENCE_COL not in df_raw.columns:
    if SUBSIDENCE_COL.endswith("_cum") and ("subsidence" in df_raw.columns):
        df_raw = rate_to_cumulative(
            df_raw,
            rate_col="subsidence",
            cum_col=SUBSIDENCE_COL,
            time_col=TIME_COL,
            group_cols=(LON_COL, LAT_COL, "city"),
            initial="first_equals_rate_dt",
            inplace=False,
        )
        print(f"  [Subs] Built cumulative '{SUBSIDENCE_COL}' from 'subsidence'.")
    elif (SUBSIDENCE_COL == "subsidence") and ("subsidence_cum" in df_raw.columns):
        df_raw = cumulative_to_rate(
            df_raw,
            cum_col="subsidence_cum",
            rate_col="subsidence",
            time_col=TIME_COL,
            group_cols=(LON_COL, LAT_COL, "city"),
            first="cum_over_dtref",
            inplace=False,
        )
        print("  [Subs] Built rate 'subsidence' from 'subsidence_cum'.")
    else:
        raise ValueError(
            f"SUBSIDENCE_COL={SUBSIDENCE_COL!r} not in df and no "
            "convertible alternative was found."
        )

# ------------------------------------------------------------------
# 2.1 Resolve optional columns exactly as declared in config.
#     Supports tuples like ("rain", "rainfall") and chooses the first
#     present column in df_raw.
# ------------------------------------------------------------------
opt_num_cols, opt_num_map = _resolve_optional_columns(df_raw, OPTIONAL_NUMERIC_FEATURES)
opt_cat_cols, opt_cat_map = _resolve_optional_columns(df_raw, OPTIONAL_CATEGORICAL_FEATURES)

# Resolve future drivers too (if present)
FUTURE_DRIVER_FEATURES, _ = _resolve_optional_columns(df_raw, FUTURE_DRIVER_FEATURES)

# ------------------------------------------------------------------
# 2.2 Decide the "meters" groundwater source used for physics
#--------------------------------------------------------------------
# ---- Apply the resolver ----
GWL_METERS_COL, GWL_ZSCORE_COL = resolve_gwl_for_physics(
    df_raw,
    gwl_col_user=GWL_COL, # may be None/'' per  rule
    prefer_depth_bgs=True,
    allow_keep_zscore_as_ml=True,
)


# Depth for the dynamic driver (z_GWL)
GWL_DEPTH_COL = GWL_METERS_COL  # 

# Head for physics/target (h)
HEAD_SRC_COL, Z_SURF_COL_USED = resolve_head_column(
    df_raw,
    depth_col=GWL_DEPTH_COL,
    head_col=HEAD_COL,
    z_surf_col=Z_SURF_COL,
    use_head_proxy=bool(USE_HEAD_PROXY),
)

# ------------------------------------------------------------------
# v3.2 robustness: make some physics inputs optional when missing.
# If the user requests physics (PDE_MODE_CONFIG not "off") but the
# dataset does not contain physics-critical columns, we auto-disable
# physics for this Stage-1 run and proceed in ML-only mode.
# ------------------------------------------------------------------
_pde_mode = str(cfg.get("PDE_MODE_CONFIG", "off")).strip().lower()
_physics_requested = _pde_mode not in ("off", "none", "no", "0")

if (GWL_METERS_COL is None) and _physics_requested:
    print(
        "  [Warn] No groundwater-level column in meters could be resolved. "
        "Disabling physics (PDE_MODE_CONFIG='off') for this Stage-1 run."
    )
    cfg["PDE_MODE_CONFIG"] = "off"
    _physics_requested = False

# If meters column is missing, fall back to z-score or user-provided GWL
# so the ML driver can still run. (Physics stays off.)
if GWL_METERS_COL is None:
    if (GWL_ZSCORE_COL is not None) and (GWL_ZSCORE_COL in df_raw.columns):
        GWL_METERS_COL = GWL_ZSCORE_COL
    elif (GWL_COL is not None) and (GWL_COL in df_raw.columns):
        GWL_METERS_COL = GWL_COL
    else:
        raise ValueError(
            "No usable groundwater column found. Provide at least one of: "
            f"{GWL_COL!r}, {GWL_ZSCORE_COL!r}, or a resolvable meters column."
        )

# Soil thickness: try to resolve from common aliases; otherwise create a
# constant thickness so the pipeline can still run (physics stays off if
# thickness was required).
if H_FIELD_COL_NAME not in df_raw.columns:
    _thick_candidates = [
        "soil_thickness_eff", "soil_thickness_imputed", "soil_thickness",
        "H_field", "soil_thickness_m"
    ]
    _found = next((c for c in _thick_candidates if c in df_raw.columns), None)
    if _found is not None:
        print(
            f"  [Info] Using thickness column {_found!r} as H_FIELD_COL_NAME "
            f"(was {H_FIELD_COL_NAME!r})."
        )
        H_FIELD_COL_NAME = _found
    else:
        if _physics_requested:
            print(
                "  [Warn] No soil thickness column found; disabling physics "
                "(PDE_MODE_CONFIG='off') for this Stage-1 run."
            )
            cfg["PDE_MODE_CONFIG"] = "off"
            _physics_requested = False
        _H_default = float(cfg.get("H_FIELD_DEFAULT_VALUE", 1.0))
        print(
            f"  [Warn] No thickness column found; creating constant "
            f"{H_FIELD_COL_NAME!r}={_H_default} (m)."
        )
        df_raw[H_FIELD_COL_NAME] = _H_default

# The physics source used later to build __si (NEVER z-score)
GWL_SRC_COL = GWL_METERS_COL

# ------------------------------------------------------------------
# 2.3 Build selection list (required + resolved optional)
#     - Always include required physics/targets:
#         lon, lat, year, subsidence, meters_gwl, thickness
#     - Optionally include z-score GWL as an extra ML feature
#       (never used for physics; can help the network)
# ------------------------------------------------------------------
base_select = [
    LON_COL,
    LAT_COL,
    TIME_COL,
    SUBSIDENCE_COL,
    GWL_METERS_COL,       # <- meters (physics-critical)
    H_FIELD_COL_NAME,
]

if Z_SURF_COL and (Z_SURF_COL in df_raw.columns):
    base_select.append(Z_SURF_COL)
if HEAD_COL and (HEAD_COL in df_raw.columns):
    base_select.append(HEAD_COL)

# Optional: keep z-score as ML convenience column if present
if (GWL_ZSCORE_COL is not None) and (GWL_ZSCORE_COL in df_raw.columns):
    base_select.append(GWL_ZSCORE_COL)

# Add resolved optional columns (numeric + categorical)
base_select += opt_num_cols + opt_cat_cols

selected = _drop_missing_keep_order(base_select, df_raw)

missing_required = [
    c for c in [
        LON_COL, LAT_COL, TIME_COL, 
        SUBSIDENCE_COL, GWL_METERS_COL, 
        H_FIELD_COL_NAME
    ]
    if c not in selected
]
if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}\n"
        f"(Groundwater meters column resolved as {GWL_METERS_COL!r})"
    )

# Optional columns can be missing; report for transparency
skipped_optional = sorted(set(opt_num_cols + opt_cat_cols) - set(selected))
if skipped_optional:
    print(f"  [Info] Optional columns not found (skipped): {skipped_optional}")

df_sel = df_raw[selected].copy()

print("[Required columns check]")
print("  depth col:", GWL_DEPTH_COL)
print("  head  col:", HEAD_SRC_COL)
print("  z_surf col:", Z_SURF_COL_USED)
print("  H col:", H_FIELD_COL_NAME)

for c in (GWL_DEPTH_COL, HEAD_SRC_COL, H_FIELD_COL_NAME):
    if c not in df_raw.columns:
        raise ValueError(f"[Stage1] Missing critical column: {c}")

if not USE_HEAD_PROXY and (
        Z_SURF_COL is not None) and (
            Z_SURF_COL not in df_raw.columns):
    raise ValueError(
        "[Stage1] USE_HEAD_PROXY=False but Z_SURF_COL"
        f"={Z_SURF_COL!r} not found in dataset."
    )

# ------------------------------------------------------------------
# 2.4 Normalize time column to DT_TMP + ensure TIME_COL is numeric year
# ------------------------------------------------------------------
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

print(f"  Shape after select: {df_sel.shape}")
print(f"  NaNs before clean: {df_sel.isna().sum().sum()}")

# Sanitize/fill missing values using your pipeline utility
df_clean = nan_ops(df_sel, ops="sanitize", action="fill", verbose=0)

print(f"  NaNs after clean : {df_clean.isna().sum().sum()}")

clean_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_02_clean.csv")
df_clean.to_csv(clean_csv, index=False)
print(f"  Saved: {clean_csv}")

# after df_clean is created (and TIME_COL is year int)
df_clean[TIME_COL] = (
    pd.to_numeric(df_clean[TIME_COL], errors="raise")
    .astype(int)
)
masks = compute_group_masks(
    df_clean,
    group_cols=[LON_COL, LAT_COL],
    time_col=TIME_COL,
    train_end_year=TRAIN_END_YEAR,
    time_steps=TIME_STEPS,
    horizon=FORECAST_HORIZON_YEARS,
)

valid_train_groups = masks.valid_for_train
valid_fcst_groups = masks.valid_for_forecast

# keep union so you can still forecast pixels that can't train
df_clean = filter_df_by_groups(
    df_clean,
    group_cols=[LON_COL, LAT_COL],
    groups=masks.keep_for_processing,
)
# ------------------------------------------------------------------
# 2.5 Censor-aware transforms (adds *_censored flags and *_eff values)
# ------------------------------------------------------------------
print(f"\n{'='*18} Step 2.5: Censor-aware transforms {'='*18}")
df_cens, censor_report = _apply_censoring(df_clean.copy(), CENSORING_SPECS)

# ==================================================================
# Step 3: Encode & Scale
# ==================================================================
print(f"\n{'='*18} Step 3: Encode & Scale {'='*18}")
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
# 3.1 Create explicit SI columns (meters) for physics-critical channels.
#
# Key rule (avoid ambiguity):
#   - Physics + targets ALWAYS use SI columns (meters).
#   - We DO NOT scale these with ML scalers.
#   - We keep *_MODEL_COL == *_SI_COL for GeoPriorSubsNet.
#
# If you later want a "scaled GWL driver" for pure ML convenience, create
# a separate driver column and add it to dynamic_features, but NEVER
# replace the gwl target/physics channel.
# ------------------------------------------------------------------

# ---- Sanity: required source columns must exist now ----
_missing_src = [c for c in (
    SUBSIDENCE_COL, GWL_SRC_COL, H_FIELD_SRC_COL
    ) if c not in df_proc.columns]
if _missing_src:
    raise ValueError(
        "Missing required source columns before SI conversion: "
        f"{_missing_src}. Check earlier selection/censoring steps."
    )

# ---- SI column names (explicit, always meters) ----
SUBS_SI_COL   = f"{SUBSIDENCE_COL}__si"      # settlement in meters
DEPTH_SI_COL  = f"{GWL_DEPTH_COL}__si"       # depth-bgs in meters (z_GWL)
HEAD_SI_COL   = f"{HEAD_SRC_COL}__si"        # head in meters
H_SI_COL      = f"{H_FIELD_SRC_COL}__si"     # thickness in meters
Z_SURF_SI_COL = f"{Z_SURF_COL_USED}__si" if Z_SURF_COL_USED else None

# ---- Build SI columns ----
df_proc[SUBS_SI_COL] = (
    pd.to_numeric(df_proc[SUBSIDENCE_COL], errors="coerce").astype(float)
    * float(SUBS_UNIT_TO_SI)
).astype(np.float32)

df_proc[DEPTH_SI_COL] = (
    pd.to_numeric(df_proc[GWL_DEPTH_COL], errors="coerce").astype(float)
    * float(HEAD_UNIT_TO_SI)   # depth is meters in your datasets; keep 1.0
).astype(np.float32)

df_proc[HEAD_SI_COL] = (
    pd.to_numeric(df_proc[HEAD_SRC_COL], errors="coerce").astype(float)
    * float(HEAD_UNIT_TO_SI)   # head is meters; keep 1.0
).astype(np.float32)

if Z_SURF_SI_COL:
    df_proc[Z_SURF_SI_COL] = (
        pd.to_numeric(df_proc[Z_SURF_COL_USED], errors="coerce").astype(float)
        * float(Z_SURF_UNIT_TO_SI)
    ).astype(np.float32)

df_proc[H_SI_COL] = (
    pd.to_numeric(df_proc[H_FIELD_SRC_COL], errors="coerce").astype(float)
    * float(THICKNESS_UNIT_TO_SI)
).astype(np.float32)

df_proc[H_SI_COL] = np.clip(df_proc[H_SI_COL].to_numpy(np.float32), H_MIN_SI, np.inf)

# Optional: hard fail if anything non-finite survives in physics-critical cols
_phys_cols = [SUBS_SI_COL, DEPTH_SI_COL, HEAD_SI_COL, H_SI_COL]
bad = {c: int((~np.isfinite(df_proc[c].to_numpy())).sum()) for c in _phys_cols}
if any(v > 0 for v in bad.values()):
    raise ValueError(f"[Stage1] Non-finite values in physics SI columns: {bad}")


# ---- Model-space columns (GeoPrior v3.2 uses SI directly) ----
SUBS_MODEL_COL = SUBS_SI_COL

# IMPORTANT v3.2:
# - dynamic GWL channel is DEPTH (z_GWL)
# - predicted/target GWL channel is HEAD
GWL_DYN_COL    = DEPTH_SI_COL
GWL_TARGET_COL = HEAD_SI_COL

# Keep a single "gwl_model" for backward compatibility: treat it as TARGET
GWL_MODEL_COL  = GWL_TARGET_COL

H_MODEL_COL    = H_SI_COL
H_FIELD_COL    = H_MODEL_COL

print("[SI cols]")
print(f"  SUBS_MODEL_COL={SUBS_MODEL_COL}  <- {SUBSIDENCE_COL} * {SUBS_UNIT_TO_SI}")
print(f"  GWL_DYN_COL   ={GWL_DYN_COL}     <- {GWL_DEPTH_COL} (depth, meters)")
print(f"  GWL_TARGET_COL={GWL_TARGET_COL} <- {HEAD_SRC_COL} (head, meters)")
print(f"  H_FIELD_COL   ={H_FIELD_COL}    <- {H_FIELD_SRC_COL} * {THICKNESS_UNIT_TO_SI}")
if Z_SURF_SI_COL:
    print(f"  Z_SURF_SI_COL ={Z_SURF_SI_COL}  <- {Z_SURF_COL_USED} (meters)")

# Quick hard checks (debugging)
for _c in (SUBS_MODEL_COL, GWL_DYN_COL, GWL_TARGET_COL, H_FIELD_COL):
    if _c not in df_proc.columns:
        raise RuntimeError(f"[Stage1] SI column missing after build: {_c}")

# Optional guard: if someone sets these True, make it explicit that we ignore
# them for GeoPriorSubsNet to keep physics consistent.
if SCALE_GWL or SCALE_H_FIELD:
    print(
        "  [Warn] SCALE_GWL/SCALE_H_FIELD are set True, but GeoPrior Stage-1 "
        "keeps physics + targets in SI (meters). These flags are ignored here."
    )

# ------------------------------------------------------------------
# 3.2 Coordinate system for physics: ALWAYS meters internally
#     (this block unchanged except it now runs after SI columns exist)
# ------------------------------------------------------------------

if COORD_MODE in ("degrees", "lonlat", "geographic"):
    if COORD_SRC_EPSG is None:
        raise ValueError("COORD_MODE='degrees' requires COORD_SRC_EPSG (e.g., 4326).")
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs(
            f"EPSG:{int(COORD_SRC_EPSG)}",
            f"EPSG:{int(COORD_TARGET_EPSG)}",
            always_xy=True,
        )
        x_m, y_m = tr.transform(
            df_proc[LON_COL].to_numpy(dtype=float),
            df_proc[LAT_COL].to_numpy(dtype=float),
        )
        df_proc["x_m"] = x_m.astype(np.float32)
        df_proc["y_m"] = y_m.astype(np.float32)

        COORD_X_COL, COORD_Y_COL = "x_m", "y_m"
        coord_epsg = int(COORD_TARGET_EPSG)

        print(
            f"[Coords] degrees(EPSG:{COORD_SRC_EPSG}) -> meters(EPSG:{coord_epsg}) "
            f"using ({COORD_X_COL},{COORD_Y_COL})"
        )
        coords_in_degrees = False  # already projected

        # lon, lat = df_proc.loc[0, "longitude"], df_proc.loc[0, "latitude"]
        lon = float(df_proc.loc[df_proc.index[0], LON_COL])
        lat = float(df_proc.loc[df_proc.index[0], LAT_COL])

        x, y = tr.transform(lon, lat)
        
        print("lon/lat:", lon, lat)
        print("UTM32649 x/y:", x, y)
        print("stored x_m/y_m:", df_proc.loc[0, "x_m"], df_proc.loc[0, "y_m"])

    except Exception as e:
        raise RuntimeError(f"Projection failed for COORD_MODE='degrees': {e}")

elif COORD_MODE in ("utm", "projected", "meters"):
    coord_epsg = int(cfg.get("COORD_EPSG", COORD_TARGET_EPSG))
    print(
        f"[Coords] Using provided projected coords directly: "
        f"({COORD_X_COL},{COORD_Y_COL}) EPSG:{coord_epsg}"
    )
else:
    raise ValueError(
        f"Unknown COORD_MODE={COORD_MODE!r}. Use 'degrees' or 'utm'/'projected'/'meters'."
    )

proc_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_03_02_proc.csv")
df_proc.to_csv(proc_csv, index=False)
print(f"  Saved: {proc_csv}") 

# Build a unique groups table for splitting.
# Include x/y if you want spatial_block split.
gcols = [LON_COL, LAT_COL]
groups_for_split = df_proc[gcols + [COORD_X_COL, COORD_Y_COL]].drop_duplicates()

# Keep ONLY groups valid for supervised training
groups_for_split = filter_df_by_groups(
    groups_for_split,
    group_cols=gcols,
    groups=valid_train_groups,
)

split = split_groups_holdout(
    groups_for_split,
    seed=int(cfg.get("SPLIT_SEED", 42)),
    val_frac=float(cfg.get("VAL_FRAC", 0.2)),
    test_frac=float(cfg.get("TEST_FRAC", 0.1)),
    strategy=str(cfg.get("HOLDOUT_STRATEGY", "random")),
    x_col=COORD_X_COL,
    y_col=COORD_Y_COL,
    block_size=float(cfg.get("HOLDOUT_BLOCK_M", 2000.0)),
)
train_groups = split.train_groups[gcols]
val_groups = split.val_groups[gcols]
test_groups = split.test_groups[gcols]

holdout_dir = os.path.join(ARTIFACTS_DIR, "holdout")
os.makedirs(holdout_dir, exist_ok=True)

train_groups_csv = os.path.join(holdout_dir, "train_groups.csv")
val_groups_csv   = os.path.join(holdout_dir, "val_groups.csv")
test_groups_csv  = os.path.join(holdout_dir, "test_groups.csv")

train_groups.to_csv(train_groups_csv, index=False)
val_groups.to_csv(val_groups_csv, index=False)
test_groups.to_csv(test_groups_csv, index=False)

#%
# ------------------------------------------------------------------
# 3.3 One-hot encode ALL optional categoricals that are present
# ------------------------------------------------------------------
encoded_names = []
ohe_paths = {}
for cat_col in _drop_missing_keep_order(opt_cat_cols, df_proc):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
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
    print(f"  Saved OHE for '{cat_col}': {path}")

# ------------------------------------------------------------------
# 3.4 Numeric time coordinate for PINN / derivatives
# ------------------------------------------------------------------
TIME_COL_NUM = f"{TIME_COL}_numeric_coord"
df_proc[TIME_COL_NUM] = (
    df_proc[DT_TMP].dt.year
    + (df_proc[DT_TMP].dt.dayofyear - 1)
    / (365 + df_proc[DT_TMP].dt.is_leap_year.astype(int))
)

# If censoring created <col>_eff and/or <col>_censored, include them
censor_numeric_additions = []
for sp in (CENSORING_SPECS or []):
    col = sp["col"]
    feff  = col + sp.get("eff_suffix", "_eff")
    fflag = col + sp.get("flag_suffix", "_censored")
    if feff in df_proc.columns:
        censor_numeric_additions.append(feff)
    if INCLUDE_CENSOR_FLAGS_AS_DYNAMIC and (fflag in df_proc.columns):
        censor_numeric_additions.append(fflag)

# ------------------------------------------------------------------
# 3.5 Print config (now consistent: GWL_SRC_COL is meters, never None)
# ------------------------------------------------------------------
config_sections = [
    ("Run", {
        "CITY_NAME": CITY_NAME,
        "MODEL_NAME": MODEL_NAME,
        "DATA_DIR": DATA_DIR,
        "BIG_FN": BIG_FN,
        "SMALL_FN": SMALL_FN,
        "ALL_CITIES_PARQUET": ALL_CITIES_PARQUET,
    }),
    ("Data search paths", {
        "SEARCH_PATHS": SEARCH_PATHS,
        "FALLBACK_PATHS": FALLBACK_PATHS,
        "ALL_CITIES_SEARCH_PATHS": ALL_CITIES_SEARCH_PATHS,
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
        # Keep both for transparency:
        "GWL_COL (ML)": GWL_COL,
        "GWL_SRC_COL (meters)": GWL_SRC_COL,
        "H_FIELD_COL_NAME": H_FIELD_COL_NAME,
    }),
    ("Feature registry", {
        "OPTIONAL_NUMERIC_FEATURES": OPTIONAL_NUMERIC_FEATURES,
        "OPTIONAL_CATEGORICAL_FEATURES": OPTIONAL_CATEGORICAL_FEATURES,
        "ALREADY_NORMALIZED_FEATURES": ALREADY_NORMALIZED_FEATURES,
        "FUTURE_DRIVER_FEATURES": FUTURE_DRIVER_FEATURES,
    }),
    ("Censoring", {
        "CENSORING_SPECS": CENSORING_SPECS,
        "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC": INCLUDE_CENSOR_FLAGS_AS_DYNAMIC,
        "INCLUDE_CENSOR_FLAGS_AS_FUTURE": INCLUDE_CENSOR_FLAGS_AS_FUTURE,
        "USE_EFFECTIVE_H_FIELD": USE_EFFECTIVE_H_FIELD,
    }),
    ("Outputs", {
        "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
        "RUN_OUTPUT_PATH": RUN_OUTPUT_PATH,
        "ARTIFACTS_DIR": ARTIFACTS_DIR,
    }),
]

config_sections.insert(5, ("Coordinates", {
    "COORD_MODE": COORD_MODE,
    "COORD_SRC_EPSG": COORD_SRC_EPSG,
    "COORD_TARGET_EPSG": COORD_TARGET_EPSG,
    "COORD_EPSG": cfg.get("COORD_EPSG", None),
    "COORD_X_COL_USED": COORD_X_COL,
    "COORD_Y_COL_USED": COORD_Y_COL,
    "NORMALIZE_COORDS": bool(NORMALIZE_COORDS),
    "KEEP_COORDS_RAW": bool(KEEP_COORDS_RAW),
    "SHIFT_RAW_COORDS": bool(SHIFT_RAW_COORDS),
}))


config_sections.insert(6, ("Physics SI", {
    "SUBS_UNIT_TO_SI": SUBS_UNIT_TO_SI,
    "THICKNESS_UNIT_TO_SI": THICKNESS_UNIT_TO_SI,
    "SCALE_GWL": SCALE_GWL,
    "SUBS_MODEL_COL": SUBS_MODEL_COL,
    "GWL_MODEL_COL": GWL_MODEL_COL,
    "H_MODEL_COL": H_MODEL_COL,
}))

print_config_table(
    config_sections,
    table_width=_TW,
    title=f"{CITY_NAME.upper()} {MODEL_NAME} STAGE-1 CONFIG",
)

# ------------------------------------------------------------------
# 3.6 Decide which coord columns you will feed to the model
# ------------------------------------------------------------------
TIME_COL_USED = TIME_COL_NUM
X_COL_USED = COORD_X_COL
Y_COL_USED = COORD_Y_COL

SHIFT_RAW_COORDS = bool(cfg.get("SHIFT_RAW_COORDS", True))
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
    df_proc[COORD_X_COL + "__shift"]  = pack.coords[:, 1]
    df_proc[COORD_Y_COL + "__shift"]  = pack.coords[:, 2]

    TIME_COL_USED = TIME_COL_NUM + "__shift"
    X_COL_USED = COORD_X_COL + "__shift"
    Y_COL_USED = COORD_Y_COL + "__shift"

    coord_shift_pack = pack

# ------------------------------------------------------------------
# 3.7 Scale numeric ML drivers (NOT physics-critical channels)
#
# No leakage rule:
#   - Fit scaler on TRAIN split only (<= TRAIN_END_YEAR)
#   - Transform the full table after fitting
# ------------------------------------------------------------------

ml_numeric_candidates = []
ml_numeric_candidates += opt_num_cols[:]  # rainfall, urban_load, etc.
ml_numeric_candidates += _drop_missing_keep_order(censor_numeric_additions, df_proc)

# Anything that influences physics or targets must NEVER be scaled here.
# (We lock both the raw sources and the SI/model cols.)
physics_locked = {
    # targets/physics channels (SI == model)
    SUBS_MODEL_COL, GWL_MODEL_COL, H_FIELD_COL,
    SUBS_SI_COL, DEPTH_SI_COL, HEAD_SI_COL, H_SI_COL,

    # time/coords (raw/proj + possibly shifted)
    TIME_COL_NUM, COORD_X_COL, COORD_Y_COL,
    TIME_COL_USED, X_COL_USED, Y_COL_USED,

    # raw columns used to build SI
    SUBSIDENCE_COL, GWL_DEPTH_COL, HEAD_SRC_COL, H_FIELD_SRC_COL,

    # bookkeeping columns you don't want scaled
    TIME_COL, DT_TMP, LON_COL, LAT_COL,
}
if Z_SURF_SI_COL:
    physics_locked.add(Z_SURF_SI_COL)


ml_numeric_cols = [
    c for c in _drop_missing_keep_order(ml_numeric_candidates, df_proc)
    if c not in physics_locked
]
ml_numeric_cols = [
    c for c in ml_numeric_cols if c not in 
    set(ALREADY_NORMALIZED_FEATURES)
]

df_scaled = df_proc.copy()
scaler_path = None

train_mask = df_proc[TIME_COL] <= TRAIN_END_YEAR
df_tmp = df_proc.loc[train_mask, gcols].copy()
df_tmp = filter_df_by_groups(
    df_tmp,
    group_cols=gcols,
    groups=train_groups,
)
train_mask = df_proc.index.isin(df_tmp.index)

if not np.any(train_mask):
    raise ValueError(f"No rows satisfy training mask: {TIME_COL} <= {TRAIN_END_YEAR}")

if ml_numeric_cols:
    scaler = MinMaxScaler()

    # Fit on TRAIN only (avoid leakage), transform ALL
    scaler.fit(df_proc.loc[train_mask, ml_numeric_cols])
    df_scaled.loc[:, ml_numeric_cols] = scaler.transform(
        df_scaled.loc[:, ml_numeric_cols]
    )

    scaler_path = os.path.join(ARTIFACTS_DIR, f"{CITY_NAME}_main_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler (fit on train only): {scaler_path}")
    print(f"  Scaled ML numeric cols: {ml_numeric_cols}")
    
else:
    scaler = None
    print("  [Info] No ML numeric features to scale.")

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
            # "scaler": scaler  # DO NOT store object in JSON; Stage-2 may attach it at runtime
        }
else:
    scaler_info = {}


scaled_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_03_scaled.csv")
df_scaled.to_csv(scaled_csv, index=False)
print(f"  Saved: {scaled_csv}")

scaled_ml_numeric_cols = ml_numeric_cols[:]  # record for manifest

# ------------------------------------------------------------------
# 3.8 Physics SI affine mapping (model-space -> SI-space)
# Stage-1 created explicit __si columns, so identity maps apply.
# ------------------------------------------------------------------
subs_scale_si, subs_bias_si = 1.0, 0.0
head_scale_si, head_bias_si = 1.0, 0.0
H_scale_si, H_bias_si       = 1.0, 0.0
gwl_z_meta = None  # unused because we feed meters directly

print("[SI affine] Using identity (inputs/targets already SI):")
print("  subs: scale_si=1.0 bias_si=0.0")
print("  head: scale_si=1.0 bias_si=0.0")
print("  H   : scale_si=1.0 bias_si=0.0")

H_FIELD_COL = H_MODEL_COL
print(f"[SI affine] H_field identity (meters already): {H_FIELD_COL}")

# XXX FOR DEBUG 
# ---- DEBUG UNITS: Stage-1 sanity ----
cols = [
    SUBSIDENCE_COL, SUBS_SI_COL,
    GWL_DEPTH_COL, DEPTH_SI_COL,
    HEAD_SRC_COL, HEAD_SI_COL,
    H_FIELD_SRC_COL, H_SI_COL,
]
cols = [c for c in cols if c and c in df_proc.columns]

print("\n[Stage1][Units] head/depth/subs sample (raw vs __si):")
print(df_proc[cols].head(5))

print("\n[Stage1][Units] ranges:")
for c in cols:
    s = df_proc[c]
    print(f"  {c:20s} min={s.min(): .4g}  max={s.max(): .4g}  mean={s.mean(): .4g}")

#%
# ==================================================================
# Step 4: Feature sets (lists only)
# ==================================================================
def _dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

print(f"\n{'='*18} Step 4: Define Feature Sets {'='*18}")

GROUP_ID_COLS = [LON_COL, LAT_COL] 
# --- base features
static_features = encoded_names[:]
if bool(cfg.get("INCLUDE_Z_SURF_AS_STATIC", True)
        ) and Z_SURF_SI_COL and (Z_SURF_SI_COL in df_scaled.columns):
    static_features.append(Z_SURF_SI_COL)

# v3.2: optionally include historical subsidence as a dynamic driver so
# physics-driven settlement (which needs an initial condition) can use
# the last observed value without extra packing.
INCLUDE_SUBS_HIST_DYNAMIC = bool(cfg.get("INCLUDE_SUBS_HIST_DYNAMIC", True))

dynamic_base  = [GWL_DYN_COL] 
if INCLUDE_SUBS_HIST_DYNAMIC and (SUBS_MODEL_COL in df_scaled.columns):
    dynamic_base.append(SUBS_MODEL_COL)

forbidden_dyn = {GWL_SRC_COL, GWL_METERS_COL, GWL_COL, (GWL_ZSCORE_COL or "")}

dynamic_extra = [c for c in opt_num_cols if c not in {
    SUBSIDENCE_COL, H_FIELD_COL_NAME, *forbidden_dyn
    }
]
dynamic_features = [c for c in dynamic_base + dynamic_extra if c in df_scaled.columns]

future_features = [c for c in FUTURE_DRIVER_FEATURES if c in df_scaled.columns]

# --- add censor flags once
for sp in CENSORING_SPECS or []:
    fflag = sp["col"] + sp.get("flag_suffix", "_censored")
    if INCLUDE_CENSOR_FLAGS_AS_DYNAMIC and fflag in df_scaled.columns:
        dynamic_features.append(fflag)
    if INCLUDE_CENSOR_FLAGS_AS_FUTURE and fflag in df_scaled.columns:
        future_features.append(fflag)

# --- deduplicate (preserve order)
dynamic_features = _dedup_keep_order(dynamic_features)
future_features  = _dedup_keep_order(future_features)

# --- record indices after finalization
gwl_dyn_index = int(dynamic_features.index(GWL_DYN_COL))

if SUBS_MODEL_COL in dynamic_features:
    subs_dyn_index = int(dynamic_features.index(SUBS_MODEL_COL))
    
z_surf_static_index = None 
if Z_SURF_SI_COL in static_features:
    z_surf_static_index = int(static_features.index(Z_SURF_SI_COL)) 
    
print(f"  Static : {static_features}")
print(f"  Dynamic: {dynamic_features}")
print(f"  Future : {future_features}")
print(f"  H_field: {H_FIELD_COL}")

# ==================================================================
# Step 5: Split & build TRAIN sequences
# ==================================================================
print(f"\n{'='*18} Step 5: Train Split & Sequences {'='*18}")
# df_train = df_scaled[df_scaled[TIME_COL] <= TRAIN_END_YEAR].copy()
df_hist = df_scaled[df_scaled[TIME_COL] <= TRAIN_END_YEAR].copy()

df_train = filter_df_by_groups(
    df_hist, group_cols=gcols, groups=train_groups
)
df_val = filter_df_by_groups(
    df_hist, group_cols=gcols, groups=val_groups
)
df_test = filter_df_by_groups(
    df_hist, group_cols=gcols, groups=test_groups
)

if df_train.empty:
    raise ValueError(f"Empty train split at year={TRAIN_END_YEAR}.")

seq_job = os.path.join(
    ARTIFACTS_DIR, 
    f"{CITY_NAME}_train_sequences_T{TIME_STEPS}_H{FORECAST_HORIZON_YEARS}.joblib"
)

OUT_S_DIM, OUT_G_DIM = 1, 1
normalize_coords = bool(NORMALIZE_COORDS)

print("gwl_col(target):", GWL_TARGET_COL)
print("gwl_dyn_col(driver):", GWL_DYN_COL)
print("dynamic_features:", dynamic_features)

# --- helper: force targets to 3D (N, H, D) ---
def _ensure_target_3d(arr, name):
    a = np.asarray(arr)
    # squeeze trailing singleton dims
    while a.ndim > 3 and a.shape[-1] == 1:
        a = np.squeeze(a, axis=-1)
    while a.ndim > 3 and a.shape[-2] == 1:
        a = np.squeeze(a, axis=-2)
    # expand if (N, H)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3:
        raise ValueError(
            f"{name} must be 3D (N,H,D), got {a.shape}"
        )
    return a.astype(np.float32)

inputs_train, targets_train, coord_scaler = prepare_pinn_data_sequences(
    df=df_train,
    time_col=TIME_COL_USED,
    lon_col=X_COL_USED,
    lat_col=Y_COL_USED,

    # Use SI columns for physics
    subsidence_col=SUBS_MODEL_COL,
    gwl_col=GWL_TARGET_COL,   # head_m__si  (TARGET)
    gwl_dyn_col=GWL_DYN_COL,     # depth__si   (DRIVER)  
    h_field_col=H_FIELD_COL,

    dynamic_cols=dynamic_features,
    static_cols=static_features,
    future_cols=future_features,
    group_id_cols=GROUP_ID_COLS,
    time_steps=TIME_STEPS,
    forecast_horizon=FORECAST_HORIZON_YEARS,
    output_subsidence_dim=OUT_S_DIM,
    output_gwl_dim=OUT_G_DIM,

    normalize_coords=normalize_coords,
    fit_coord_scaler=normalize_coords,
    return_coord_scaler=True,

    savefile=seq_job,
    mode=MODE,
    model=MODEL_NAME,
    verbose=2,
)

# --- VAL ---
inputs_val, targets_val = prepare_pinn_data_sequences(
    df=df_val,
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
    normalize_coords=normalize_coords,
    coord_scaler=coord_scaler,
    fit_coord_scaler=False,
    return_coord_scaler=False,
    mode=MODE,
    model=MODEL_NAME,
    verbose=2,
)

# --- TEST ---
inputs_test, targets_test = prepare_pinn_data_sequences(
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
    normalize_coords=normalize_coords,
    coord_scaler=coord_scaler,
    fit_coord_scaler=False,
    return_coord_scaler=False,
    mode=MODE,
    model=MODEL_NAME,
    verbose=2,
)

# --- enforce target shapes (IMPORTANT) ---
targets_train["subsidence"] = _ensure_target_3d(
    targets_train["subsidence"], "train subsidence"
)
targets_train["gwl"] = _ensure_target_3d(
    targets_train["gwl"], "train gwl"
)
targets_val["subsidence"] = _ensure_target_3d(
    targets_val["subsidence"], "val subsidence"
)
targets_val["gwl"] = _ensure_target_3d(
    targets_val["gwl"], "val gwl"
)
targets_test["subsidence"] = _ensure_target_3d(
    targets_test["subsidence"], "test subsidence"
)
targets_test["gwl"] = _ensure_target_3d(
    targets_test["gwl"], "test gwl"
)


if targets_train["subsidence"].shape[0] == 0:
    raise ValueError("No training sequences were generated.")

coord_scaler_path = None
if normalize_coords:
    coord_scaler_path = os.path.join(ARTIFACTS_DIR, f"{CITY_NAME}_coord_scaler.joblib")
    joblib.dump(coord_scaler, coord_scaler_path)
    print(f"  Saved coord scaler: {coord_scaler_path}")
else:
    coord_scaler = None
    print("  [Coords] KEEP_COORDS_RAW=True -> coords NOT normalized; no coord_scaler saved.")
#%
# ---- DEBUG UNITS: Stage-1 target tensors ----

def _np_stats(name, a):
    a = np.asarray(a)
    print(f"[Stage1][NPZ] {name:12s} shape={a.shape} "
          f"min={np.nanmin(a):.4g} max={np.nanmax(a):.4g} mean={np.nanmean(a):.4g}")

_np_stats("y_subs", targets_train["subsidence"])
_np_stats("y_gwl",  targets_train["gwl"])

#%
# --------------------------------------------------------------
# coord ranges for chain-rule in Stage-2
# --------------------------------------------------------------

EPS_RNG = 1e-6
if normalize_coords:
    cmin = np.asarray(coord_scaler.data_min_, dtype=float)
    cmax = np.asarray(coord_scaler.data_max_, dtype=float)
    crng = np.maximum(cmax - cmin, EPS_RNG)
    coord_ranges = {
        "t": float(crng[0]), 
        "x": float(crng[1]), 
        "y": float(crng[2])
    }
else:
    coords_raw = np.asarray(inputs_train["coords"], dtype=float)
    coord_ranges = {
        "t": float(max(coords_raw[..., 0].max() - coords_raw[..., 0].min(), EPS_RNG)),
        "x": float(max(coords_raw[..., 1].max() - coords_raw[..., 1].min(), EPS_RNG)),
        "y": float(max(coords_raw[..., 2].max() - coords_raw[..., 2].min(), EPS_RNG)),
    }

for k, v in inputs_train.items():
    print(f"  Train input '{k}': {None if v is None else v.shape}")
for k, v in targets_train.items():
    print(f"  Train target '{k}': {v.shape}")

# --------------------------------------------------------------
# Stage-1 Audit (coords + scaling + feature split + SI sanity)
# --------------------------------------------------------------
if should_audit(AUDIT_STAGES, stage="stage1"):
    _ = audit_stage1_scaling(
        df_train=df_train,
        inputs_train=inputs_train,
        targets_train=targets_train,
        coord_scaler=coord_scaler,
        coord_ranges=coord_ranges,

        # provenance
        coord_mode=COORD_MODE,
        coords_in_degrees=bool(coords_in_degrees),
        coord_epsg_used=coord_epsg,
        coord_x_col_used=COORD_X_COL,
        coord_y_col_used=COORD_Y_COL,
        x_col_used=X_COL_USED,
        y_col_used=Y_COL_USED,
        time_col_used=TIME_COL_USED,
        normalize_coords=bool(NORMALIZE_COORDS),
        keep_coords_raw=bool(KEEP_COORDS_RAW),
        shift_raw_coords=bool(SHIFT_RAW_COORDS),

        # physics columns (SI/model)
        subs_model_col=SUBS_MODEL_COL,
        gwl_dyn_col=GWL_DYN_COL,
        gwl_target_col=GWL_TARGET_COL,
        h_field_col=H_FIELD_COL,

        # feature lists + scaler summary
        dynamic_features=dynamic_features,
        static_features=static_features,
        future_features=future_features,
        scaled_ml_numeric_cols=scaled_ml_numeric_cols,

        # UI + saving
        save_dir=RUN_OUTPUT_PATH,
        table_width=_TW,
        title_prefix="COORDINATE + FEATURE SCALING AUDIT (Stage-1)",
        city=CITY_NAME,
        model_name=MODEL_NAME,
        sample_rows=5,
        main_scaler_path=scaler_path,
        scaler_info=scaler_info,
    )

#%
# ==================================================================
# Step 6: Build train/val datasets & EXPORT numpy
# ==================================================================
print(f"\n{'='*18} Step 6: Split & Export NPZ {'='*18}")

num_train = inputs_train["dynamic_features"].shape[0]
future_time_dim = TIME_STEPS + FORECAST_HORIZON_YEARS if MODE == "tft_like" else FORECAST_HORIZON_YEARS

# Fill missing optional blocks with empty tensors (consistent shapes)
static_arr = inputs_train.get("static_features")
if static_arr is None:
    static_arr = np.zeros((num_train, 0), dtype=np.float32)

future_arr = inputs_train.get("future_features")
if future_arr is None:
    future_arr = np.zeros((num_train, future_time_dim, 0), dtype=np.float32)

# Build export dicts directly from arrays (no tf.data round-trip)
dataset_inputs = {
    "coords":            np.asarray(inputs_train["coords"], dtype=np.float32),
    "dynamic_features":  np.asarray(inputs_train["dynamic_features"], dtype=np.float32),
    "static_features":   np.asarray(static_arr, dtype=np.float32),
    "future_features":   np.asarray(future_arr, dtype=np.float32),
    "H_field":           np.asarray(inputs_train["H_field"], dtype=np.float32),
}
dataset_targets = {
    "subs_pred": np.asarray(targets_train["subsidence"], dtype=np.float32),
    "gwl_pred":  np.asarray(targets_train["gwl"], dtype=np.float32),
}

def _pack_inputs(inputs, future_time_dim):
    n = inputs["dynamic_features"].shape[0]

    st = inputs.get("static_features")
    if st is None:
        st = np.zeros((n, 0), np.float32)

    fu = inputs.get("future_features")
    if fu is None:
        fu = np.zeros((n, future_time_dim, 0), np.float32)

    return {
        "coords": inputs["coords"].astype(np.float32),
        "dynamic_features": inputs["dynamic_features"].astype(np.float32),
        "static_features": st.astype(np.float32),
        "future_features": fu.astype(np.float32),
        "H_field": inputs["H_field"].astype(np.float32),
    }


def _pack_targets(targets):
    return {
        "subs_pred": targets["subsidence"].astype(np.float32),
        "gwl_pred": targets["gwl"].astype(np.float32),
    }

future_time_dim = (
    TIME_STEPS + FORECAST_HORIZON_YEARS
    if MODE == "tft_like"
    else FORECAST_HORIZON_YEARS
)

train_np_x = _pack_inputs(inputs_train, future_time_dim)
train_np_y = _pack_targets(targets_train)

val_np_x = _pack_inputs(inputs_val, future_time_dim)
val_np_y = _pack_targets(targets_val)

test_np_x = _pack_inputs(inputs_test, future_time_dim)
test_np_y = _pack_targets(targets_test)

# # Deterministic split indices
# val_size = int(0.2 * num_train)
# val_size = max(1, min(val_size, num_train - 1))

# rng = np.random.default_rng(42)
# # perm = rng.permutation(num_train)
# # val_idx = perm[:val_size]
# # train_idx = perm[val_size:]

# # train_np_x = {k: v[train_idx] for k, v in dataset_inputs.items()}
# # train_np_y = {k: v[train_idx] for k, v in dataset_targets.items()}
# # val_np_x   = {k: v[val_idx]   for k, v in dataset_inputs.items()}
# # val_np_y   = {k: v[val_idx]   for k, v in dataset_targets.items()}

train_inputs_npz  = os.path.join(ARTIFACTS_DIR, "train_inputs.npz")
train_targets_npz = os.path.join(ARTIFACTS_DIR, "train_targets.npz")
val_inputs_npz    = os.path.join(ARTIFACTS_DIR, "val_inputs.npz")
val_targets_npz   = os.path.join(ARTIFACTS_DIR, "val_targets.npz")


test_inputs_npz  = os.path.join(ARTIFACTS_DIR, "test_inputs.npz")
test_targets_npz = os.path.join(ARTIFACTS_DIR, "test_targets.npz")

_save_npz(train_inputs_npz, train_np_x)
_save_npz(train_targets_npz, train_np_y)
_save_npz(val_inputs_npz, val_np_x)
_save_npz(val_targets_npz, val_np_y)

_save_npz(test_inputs_npz, test_np_x)
_save_npz(test_targets_npz, test_np_y)

print(f"  Saved NPZs:\n    {train_inputs_npz}\n    {train_targets_npz}\n"
      f"    {val_inputs_npz}\n    {val_targets_npz}")

# ==================================================================
# Manifest: single handshake for Stage-2 (no redundant truth source)
# ==================================================================
print(f"\n{'='*18} Build Manifest {'='*18}")

# Keep censoring details together (needed for audit + Stage-2 checks)
manifest_censor = {
    "specs": CENSORING_SPECS,
    "report": censor_report,
    "use_effective_h_field": USE_EFFECTIVE_H_FIELD,
    "flags_as_dynamic": INCLUDE_CENSOR_FLAGS_AS_DYNAMIC,
    "flags_as_future": INCLUDE_CENSOR_FLAGS_AS_FUTURE,
}

# ---- 1) Canonical naming/roles: ONE place for columns & feature order ----
cols_spec = {
    # raw identifiers
    "time": TIME_COL,
    "lon": LON_COL,
    "lat": LAT_COL,

    # time/coords actually fed to the model sequences (may be shifted/proj)
    "time_numeric": TIME_COL_NUM,
    "time_used": TIME_COL_USED,
    "x_base": COORD_X_COL,     # projected or raw, before shifting
    "y_base": COORD_Y_COL,
    "x_used": X_COL_USED,      # used in sequences (shifted or base)
    "y_used": Y_COL_USED,

    # subsidence: raw column + model-space column (SI in v3.2)
    "subs_raw": SUBSIDENCE_COL,
    "subs_model": SUBS_MODEL_COL,

    # groundwater: explicit split (driver=depth, target=head)
    "depth_raw": GWL_DEPTH_COL,
    "head_raw": HEAD_SRC_COL,
    "depth_model": GWL_DYN_COL,
    "head_model": GWL_TARGET_COL,

    # thickness: source and model column (SI in v3.2)
    "h_field_raw": H_FIELD_SRC_COL if "H_FIELD_SRC_COL" in globals() else H_FIELD_COL_NAME,
    "h_field_model": H_FIELD_COL,

    # optional surface elevation (static)
    "z_surf_raw": Z_SURF_COL_USED,
    "z_surf_static": Z_SURF_SI_COL,
}

features_spec = {
    "static": list(static_features),
    "dynamic": list(dynamic_features),
    "future": list(future_features),
    "group_id_cols": list(GROUP_ID_COLS),
}

# indices are the only “positional truth” Stage-2 needs
indices_spec = {
    "gwl_dyn_index": int(gwl_dyn_index),
    "subs_dyn_index": (int(subs_dyn_index) if subs_dyn_index is not None else None),
    "gwl_dyn_name": GWL_DYN_COL, 
    'z_surf_static_index': (
        int(z_surf_static_index) if z_surf_static_index is not None else None),
}

model_cols = {
    "gwl_col": GWL_DYN_COL,  # for backward compatibility
    "gwl_dyn_col": GWL_DYN_COL,
    "gwl_target_col": GWL_TARGET_COL,
    "subs_model_col": SUBS_MODEL_COL,
    "z_surf_col": Z_SURF_SI_COL,
}

kind = str(cfg.get("GWL_KIND", "depth_bgs"))
sign = cfg.get("GWL_SIGN", None)
if sign is None:
    sign = "up_positive" if kind == "head" else "down_positive"

conventions_spec = {
    # GWL conventions (Stage-2 physics depends on these)
    "gwl_kind": kind,
    "gwl_sign": sign, # str(cfg.get("GWL_SIGN", "down_positive")),  # declared/raw sign
    "use_head_proxy": bool(cfg.get("USE_HEAD_PROXY", True)),
    "time_units": str(TIME_UNITS),

    # what Stage-1 actually produced (explicit in v3.2)
    "gwl_driver_kind": "depth",
    "gwl_target_kind": "head",
    # explicit Stage-1 resolved roles (truth for Stage-2)
    "gwl_driver_sign": "down_positive",
    "gwl_target_sign": "up_positive",

}


# ---- 2) Numeric scaling/physics metadata: ONE place for scalars/ranges ----
# IMPORTANT: scaling_kwargs contains *no column names* (that lives in cols_spec).
scaling_kwargs = {
    # SI affine maps (identity here because Stage-1 produced SI columns)
    "subs_scale_si": float(subs_scale_si),
    "subs_bias_si": float(subs_bias_si),
    "head_scale_si": float(head_scale_si),
    "head_bias_si": float(head_bias_si),
    "H_scale_si": float(H_scale_si),
    "H_bias_si": float(H_bias_si),
    
    "subsidence_kind": SUBSIDENCE_KIND, 
    "allow_subs_residual": ALLOW_SUBS_RESIDUAL, 

    # coords scaling metadata (needed for chain-rule in Stage-2)
    "coords_normalized": bool(normalize_coords),
    "coord_order": ["t", "x", "y"],
    "coord_ranges": {k: float(v) for k, v in coord_ranges.items()},

    # coordinate provenance
    "coord_mode": str(COORD_MODE),
    "coord_src_epsg": (int(COORD_SRC_EPSG) if COORD_SRC_EPSG is not None else None),
    "coord_target_epsg": int(COORD_TARGET_EPSG),
    "coord_epsg_used": (int(coord_epsg) if coord_epsg is not None else None),
    "coords_in_degrees": bool(coords_in_degrees),
    
    "cons_residual_units": CONSOLIDATION_RESIDUAL_UNITS, 
    'cons_scale_floor':  CONS_SCALE_FLOOR,
    'gw_scale_floor' :GW_SCALE_FLOOR,
    'dt_min_units' :DT_MIN_UNITS,
    'Q_wrt_normalized_time':Q_WRT_NORMALIZED_TIME,
    'Q_in_si' : Q_IN_SI,
    'Q_in_per_second' : Q_IN_PER_SECOND,
    "Q_kind": Q_KIND,
    "Q_length_in_si": Q_LENGTH_IN_SI, 
    "drainage_mode": DRAINAGE_MODE, 
    
    "scaling_error_policy": SCALING_ERROR_POLICY, 
    'debug_physics_grads': DEBUG_PHYSICS_GRADS, 

    "gw_residual_units": GW_RESIDUAL_UNITS,
    "clip_global_norm": CLIP_GLOBAL_NORM, 
    
    # --- consolidation drawdown options ----------------------
    "cons_drawdown_mode": str(CONS_DRAWDOWN_MODE).lower(),
    "cons_drawdown_rule": str(CONS_DRAWDOWN_RULE).lower(),
    "cons_stop_grad_ref": bool(CONS_STOP_GRAD_REF),
    "cons_drawdown_zero_at_origin": bool(CONS_DRAWDOWN_ZERO_AT_ORIGIN),
    "cons_drawdown_clip_max": (
        None if CONS_DRAWDOWN_CLIP_MAX is None else float(CONS_DRAWDOWN_CLIP_MAX)
    ),
    "cons_relu_beta": float(CONS_RELU_BETA),
    
    "mv_prior_units": MV_PRIOR_UNITS, 
    "mv_alpha_disp": MV_ALPHA_DISP, 
    "mv_huber_delta":   MV_HUBER_DELTA,  
    
    "mv_prior_mode": MV_PRIOR_MODE,
    "mv_weight": MV_WEIGHT, 

    "mv_schedule_unit": MV_SCHEDULE_UNIT,
    "mv_delay_epochs": int(MV_DELAY_EPOCHS),
    "mv_warmup_epochs": int(MV_WARMUP_EPOCHS),
    "mv_delay_steps": (None if MV_DELAY_STEPS is None else int(MV_DELAY_STEPS)),
    "mv_warmup_steps": (None if MV_WARMUP_STEPS is None else int(MV_WARMUP_STEPS)),
    
    "track_aux_metrics": TRACK_AUX_METRICS, 

    **indices_spec, **model_cols
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

# ---- 3) Manifest object (minimal, stable schema) ----
manifest = {
    "schema_version": "3.2",  # bump when Stage-2 handshake changes
    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "city": CITY_NAME,
    "model": MODEL_NAME,
    "stage": "stage1",

    # Stage-2 handshake (minimal, canonical)
    "config": {
        "TIME_STEPS": int(TIME_STEPS),
        "FORECAST_HORIZON_YEARS": int(FORECAST_HORIZON_YEARS),
        "MODE": str(MODE),
        "TRAIN_END_YEAR": int(TRAIN_END_YEAR),
        "FORECAST_START_YEAR": int(FORECAST_START_YEAR),

        "cols": cols_spec,
        "features": features_spec,
        "indices": indices_spec,
        "conventions": conventions_spec,
        "scaler_info": scaler_info,
        "scaling_kwargs": scaling_kwargs,

        # “declared vs resolved” is optional but useful for audit/debug
        "feature_registry": {
            "optional_numeric_declared": OPTIONAL_NUMERIC_FEATURES,
            "optional_categorical_declared": OPTIONAL_CATEGORICAL_FEATURES,
            "already_normalized": ALREADY_NORMALIZED_FEATURES,
            "future_drivers_declared": FUTURE_DRIVER_FEATURES,
            "resolved_optional_numeric": opt_num_cols,
            "resolved_optional_categorical": opt_cat_cols,
        },
        "censoring": manifest_censor,

        # Units provenance: what Stage-1 applied to produce SI columns
        "units_provenance": {
            "subs_unit_to_si_applied_stage1": float(SUBS_UNIT_TO_SI),
            "thickness_unit_to_si_applied_stage1": float(THICKNESS_UNIT_TO_SI),
            "head_unit_to_si_assumed_stage1": float(HEAD_UNIT_TO_SI),
            "z_surf_unit_to_si_assumed_stage1": float(cfg.get("Z_SURF_UNIT_TO_SI", 1.0)),
        },
    },

    # Artifacts are just file pointers (Stage-2 loads from here)
    "artifacts": {
        "csv": {
            "raw": raw_csv,
            "clean": clean_csv,
            "scaled": scaled_csv,
        },
        "encoders": {
            "ohe": ohe_paths,                  # dict {col: path}
            "coord_scaler": coord_scaler_path, # None if KEEP_COORDS_RAW
            "main_scaler": scaler_path,        # may be None if no ML cols scaled
            "scaled_ml_numeric_cols": scaled_ml_numeric_cols,
        },
        "sequences": {
            "joblib_train_sequences": seq_job,
            "dims": {
                "output_subsidence_dim": int(OUT_S_DIM),
                "output_gwl_dim": int(OUT_G_DIM),
            },
        },
        "numpy": {
            "train_inputs_npz": train_inputs_npz,
            "train_targets_npz": train_targets_npz,
            "val_inputs_npz": val_inputs_npz,
            "val_targets_npz": val_targets_npz,
        },
        # shapes are helpful but do not define truth (derived from arrays)
        "shapes": {
            "train_inputs": {k: list(v.shape) for k, v in train_np_x.items()},
            "train_targets": {k: list(v.shape) for k, v in train_np_y.items()},
            "val_inputs": {k: list(v.shape) for k, v in val_np_x.items()},
            "val_targets": {k: list(v.shape) for k, v in val_np_y.items()},
        },
    },

    "paths": {
        "run_dir": RUN_OUTPUT_PATH,
        "artifacts_dir": ARTIFACTS_DIR,
    },

    "versions": {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    },
}

manifest["artifacts"]["numpy"]["test_inputs_npz"] = test_inputs_npz
manifest["artifacts"]["numpy"]["test_targets_npz"] = test_targets_npz

manifest["artifacts"]["shapes"]["test_inputs"] = {
    k: list(v.shape) for k, v in test_np_x.items()
}
manifest["artifacts"]["shapes"]["test_targets"] = {
    k: list(v.shape) for k, v in test_np_y.items()
}

manifest["config"]["holdout"] = {
    "strategy": str(cfg.get("HOLDOUT_STRATEGY", "random")),
    "seed": int(cfg.get("SPLIT_SEED", 42)),
    "val_frac": float(cfg.get("VAL_FRAC", 0.2)),
    "test_frac": float(cfg.get("TEST_FRAC", 0.1)),

    "group_cols": [LON_COL, LAT_COL],
    "coord_cols_for_blocking": {
        "x_col": COORD_X_COL,
        "y_col": COORD_Y_COL,
        "epsg_used": (
            None if coord_epsg is None else int(coord_epsg)
        ),
    },
    "block_size_m": (
        None
        if str(cfg.get("HOLDOUT_STRATEGY", "random")) != "spatial_block"
        else float(cfg.get("HOLDOUT_BLOCK_M", 2000.0))
    ),

    "requirements": {
        "time_col": TIME_COL,
        "train_end_year": int(TRAIN_END_YEAR),
        "forecast_start_year": int(FORECAST_START_YEAR),
        "time_steps": int(TIME_STEPS),
        "horizon": int(FORECAST_HORIZON_YEARS),

        "years_required_for_train_windows": int(
            TIME_STEPS + FORECAST_HORIZON_YEARS
        ),
        "note": (
            "Group validity enforced by compute_group_masks() "
            "before holdout splitting."
        ),
    },

    "group_counts": {
        "valid_for_train": int(len(valid_train_groups)),
        "valid_for_forecast": int(len(valid_fcst_groups)),
        "kept_for_processing": int(len(masks.keep_for_processing)),

        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "test_groups": int(len(test_groups)),
    },

    "row_counts_hist": {
        "train_rows": int(df_train.shape[0]),
        "val_rows": int(df_val.shape[0]),
        "test_rows": int(df_test.shape[0]),
    },

    "sequence_counts": {
        "train_seq": int(inputs_train["coords"].shape[0]),
        "val_seq": int(inputs_val["coords"].shape[0]),
        "test_seq": int(inputs_test["coords"].shape[0]),
    },

    "artifacts": {
        "train_groups_csv": train_groups_csv,
        "val_groups_csv": val_groups_csv,
        "test_groups_csv": test_groups_csv,
    },
}


# === Step 5b: Build TEST sequences (temporal generalization) ===
df_oos_time = df_scaled[df_scaled[TIME_COL] >= FORECAST_START_YEAR].copy()
df_oos_time = filter_df_by_groups(
    df_oos_time,
    group_cols=gcols,
    groups=valid_fcst_groups,
)

oos_inputs = oos_targets = None
if not df_oos_time.empty:
    try:
        oos_inputs, oos_targets = prepare_pinn_data_sequences(
            df=df_oos_time,
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
            normalize_coords=normalize_coords,
            coord_scaler=coord_scaler,
            fit_coord_scaler=False,
            mode=MODE,
            model=MODEL_NAME,
            verbose=2,
        )

        oos_targets["subsidence"] = _ensure_target_3d(
            oos_targets["subsidence"], "oos subsidence"
        )
        oos_targets["gwl"] = _ensure_target_3d(
            oos_targets["gwl"], "oos gwl"
        )

        # export
        fut_T = (
            TIME_STEPS + FORECAST_HORIZON_YEARS
            if MODE == "tft_like"
            else FORECAST_HORIZON_YEARS
        )
        oos_x = _pack_inputs(oos_inputs, fut_T)
        oos_y = _pack_targets(oos_targets)

        oos_x_npz = os.path.join(
            ARTIFACTS_DIR, "oos_time_inputs.npz"
        )
        oos_y_npz = os.path.join(
            ARTIFACTS_DIR, "oos_time_targets.npz"
        )
        _save_npz(oos_x_npz, oos_x)
        _save_npz(oos_y_npz, oos_y)

        manifest["artifacts"]["numpy"]["oos_time_inputs_npz"] = oos_x_npz
        manifest["artifacts"]["numpy"]["oos_time_targets_npz"] = oos_y_npz

    except Exception as e:
        print(f"[Warn] OOS-time build failed: {e}")


# === Step 7: Build *true future* sequences (for Stage-3) ===

future_npz_paths = None
if BUILD_FUTURE_NPZ:
    print(f"\n{'='*18} Step 7: Future Sequences {'='*18}")
    try:
        future_npz_paths = build_future_sequences_npz(
            df_scaled=df_scaled,
            time_col=TIME_COL,
            time_col_num=TIME_COL_USED,
            lon_col=X_COL_USED,
            lat_col=Y_COL_USED,
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
            normalize_coords=normalize_coords,     
            coord_scaler=coord_scaler,            
            verbose=7,
        )
        # Attach future_* NPZ paths into the manifest
        manifest["artifacts"]["numpy"].update(future_npz_paths)
    except Exception as e:
        print(
            f"  [Warn] BUILD_FUTURE_NPZ=True but construction failed: {e}\n"
            "        Continuing without future_* NPZs."
        )

def _chk(name, a, nd):
    a = np.asarray(a)
    print(name, a.shape, "ndim=", a.ndim)
    if a.ndim != nd:
        raise ValueError(f"{name} ndim != {nd}: {a.shape}")

_chk("X.dynamic", train_np_x["dynamic_features"], 3)
_chk("X.coords",  train_np_x["coords"], 3)
_chk("X.H_field", train_np_x["H_field"], 3)

_chk("y.subs", train_np_y["subs_pred"], 3)
_chk("y.gwl",  train_np_y["gwl_pred"], 3)

manifest_path = os.path.join(RUN_OUTPUT_PATH, "manifest.json")

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"  Saved manifest: {manifest_path}")

print(f"\n{'-'*_TW}\nSTAGE-1 COMPLETE. Artifacts in:\n  {RUN_OUTPUT_PATH}\n{'-'*_TW}")
