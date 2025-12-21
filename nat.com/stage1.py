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

# --- Suppress common warnings/tf chatter ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
if hasattr(tf, "autograph") and hasattr(tf.autograph, "set_verbosity"):
    tf.autograph.set_verbosity(0)

# ---- Project imports (keep these minimal for Stage-1) ----
try:
    from fusionlab.api.util import get_table_size
    from fusionlab.datasets import fetch_zhongshan_data
    from fusionlab.utils.data_utils import nan_ops
    from fusionlab.utils.io_utils import save_job
    from fusionlab.utils.geo_utils import unpack_frames_from_file
    from fusionlab.utils.nat_utils import load_nat_config
    from fusionlab.utils.generic_utils import (
        normalize_time_column,
        ensure_directory_exists,
        print_config_table
    )
    from fusionlab.utils.subsidence_utils import ( 
        rate_to_cumulative, cumulative_to_rate, 
        finalize_si_scaling_kwargs
    )
    from fusionlab.utils.subsidence_utils import make_txy_coords
    from fusionlab.utils.sequence_utils import build_future_sequences_npz
    from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences
    
    print("Successfully imported fusionlab modules.")
except Exception as e:
    print(f"Failed to import fusionlab modules: {e}")
    raise


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

# --- Censoring config (wired exactly like training/tuning) ---
_censor_cfg = cfg.get("censoring", {}) or {}
CENSORING_SPECS = _censor_cfg.get("specs", [])
INCLUDE_CENSOR_FLAGS_AS_DYNAMIC = bool(
    _censor_cfg.get("flags_as_dynamic", True)
)
INCLUDE_CENSOR_FLAGS_AS_FUTURE = bool(
    _censor_cfg.get("flags_as_future", False)
)

USE_EFFECTIVE_H_FIELD = bool(
    _censor_cfg.get("use_effective_h_field", True)
)
THICKNESS_UNIT_TO_SI = float(cfg.get("THICKNESS_UNIT_TO_SI", 1.0))

# --- Stage-1 physics-critical scaling controls ---
KEEP_COORDS_RAW = bool(cfg.get("KEEP_COORDS_RAW", True))
SCALE_H_FIELD   = bool(cfg.get("SCALE_H_FIELD", False))  # recommended False
SCALE_GWL       = bool(cfg.get("SCALE_GWL", False))      # recommended False

SUBS_UNIT_TO_SI = float(cfg.get("SUBS_UNIT_TO_SI", 1e-3))  # mm -> m
SUBSIDENCE_KIND = str(cfg.get("SUBSIDENCE_KIND", "cumulative")).lower()

# Optional but strongly recommended if GWL_COL is not in meters:
GWL_RAW_COL = cfg.get("GWL_RAW_COL", None)  # e.g. "GWL_depth_bgs"


# --- Output directories (optionally overridable from cfg) ---
BASE_OUTPUT_DIR = cfg.get("BASE_OUTPUT_DIR", os.path.join(os.getcwd(), "results"))
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
# If config says "GWL", treat it as an alias and standardize to depth_bgs.
if isinstance(GWL_COL, str) and (GWL_COL.strip().lower() == "gwl"):
    # Priority order: prefer the physically meaningful meters column
    if ("GWL_depth_bgs" in df_raw.columns) and ("GWL" in df_raw.columns):
        print("  [GWL] Config uses 'GWL'. Both 'GWL' and 'GWL_depth_bgs' exist; "
              "switching GWL_COL -> 'GWL_depth_bgs' (meters).")
        GWL_COL = "GWL_depth_bgs"

    elif ("GWL_depth_bgs" in df_raw.columns) and ("GWL" not in df_raw.columns):
        print("  [GWL] Config uses 'GWL' but dataset has 'GWL_depth_bgs'. "
              "Switching GWL_COL -> 'GWL_depth_bgs'.")
        GWL_COL = "GWL_depth_bgs"

    elif ("GWL" in df_raw.columns) and ("GWL_depth_bgs" not in df_raw.columns):
        # We only rename if the target name does not already exist.
        print("  [GWL] Dataset has 'GWL' but missing 'GWL_depth_bgs'. "
              "Renaming column 'GWL' -> 'GWL_depth_bgs' for consistency.")
        df_raw.rename(columns={"GWL": "GWL_depth_bgs"}, inplace=True)
        GWL_COL = "GWL_depth_bgs"

    else:
        print("  [GWL] Config uses 'GWL' but dataset has neither 'GWL' nor "
              "'GWL_depth_bgs'. Will error later in required-column checks.")

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
#
# Rule (physics-first):
#   - Physics MUST use meters. Prefer 'GWL_depth_bgs' or 'GWL'.
#   - If user explicitly provides a GWL_COL (non-None), we interpret it
#     as "this is the column I want as the PHYSICS groundwater input".
#       * If it exists and is NOT z-score -> accept.
#       * If it is z-score -> reject (physics requires meters).
#   - If user does NOT provide GWL_COL (None or ''), auto-detect:
#       * If meters exist -> choose 'GWL_depth_bgs' (preferred) else 'GWL'
#         and print a message.
#       * If meters exist AND z-score exists -> still choose meters and
#         print a message that z-score will be ignored for physics.
#       * If only z-score exists -> raise with a clear physics message.
#
# Optional:
#   - If z-score exists AND meters exists, keep z-score as an optional
#     ML convenience feature (if you want), but never use it for physics.
# ------------------------------------------------------------------

def _is_nonempty_str(x) -> bool:
    return isinstance(x, str) and x.strip() != ""

def _looks_like_zscore_name(col: str) -> bool:
    return isinstance(col, str) and col.lower().endswith("_z")

def resolve_gwl_for_physics(
    df: pd.DataFrame,
    gwl_col_user: Optional[str],
    *,
    prefer_depth_bgs: bool = True,
    allow_keep_zscore_as_ml: bool = True,
) -> tuple[str, Optional[str]]:
    """
    Resolve groundwater columns for GeoPrior Stage-1.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    gwl_col_user : str or None
        User-provided GWL_COL. If None/empty, auto-detect.
        If provided, it is treated as the intended PHYSICS input and
        must be in meters (NOT z-scored).
    prefer_depth_bgs : bool, default=True
        If auto-detect and both meters candidates exist, choose
        'GWL_depth_bgs' over 'GWL'.
    allow_keep_zscore_as_ml : bool, default=True
        If meters exist and z-score exists, return the z-score column
        as a "ML convenience" column name (optional).

    Returns
    -------
    gwl_meters_col : str
        The column name to use for physics and for building __si.
    gwl_zscore_col : str or None
        Optional z-score column name, returned only if present AND
        allow_keep_zscore_as_ml=True.

    Raises
    ------
    ValueError
        If only z-score exists, or if the user explicitly requests
        a z-score column for physics.
    """
    cols = set(df.columns)

    # Canonical candidate names used across your pipeline
    meters_candidates = []
    if prefer_depth_bgs:
        meters_candidates += ["GWL_depth_bgs", "GWL"]
    else:
        meters_candidates += ["GWL", "GWL_depth_bgs"]

    zscore_candidates = ["GWL_depth_bgs_z", "GWL_z"]

    # Discover availability
    meters_available = [c for c in meters_candidates if c in cols]
    zscore_available = [c for c in zscore_candidates if c in cols]

    # --- Case A: user explicitly provided a groundwater column ---
    if _is_nonempty_str(gwl_col_user):
        if gwl_col_user not in cols:
            raise ValueError(
                f"GWL_COL={gwl_col_user!r} was provided but does not exist in the dataset."
            )
        if _looks_like_zscore_name(gwl_col_user) or (gwl_col_user in zscore_candidates):
            raise ValueError(
                f"GWL_COL={gwl_col_user!r} looks like a z-score column. "
                "GeoPrior physics requires groundwater in meters. "
                "Please set GWL_COL to 'GWL_depth_bgs' or 'GWL' (meters)."
            )

        # If user points to a meters column, accept it.
        gwl_m = gwl_col_user

        # Optionally keep z-score for ML (never for physics)
        gwl_z = zscore_available[0] if (allow_keep_zscore_as_ml and zscore_available) else None

        # Informative messages
        if gwl_z is not None:
            print(
                f"  [GWL] Using user-provided meters column {gwl_m!r} for physics. "
                f"Z-score column {gwl_z!r} exists and will be kept only as an optional ML feature."
            )
        else:
            print(f"  [GWL] Using user-provided meters column {gwl_m!r} for physics.")
        return gwl_m, gwl_z

    # --- Case B: user did not provide GWL_COL -> auto detect meters ---
    if meters_available:
        # Choose preferred meters column
        gwl_m = meters_available[0]  # already ordered by preference above

        gwl_z = zscore_available[0] if (allow_keep_zscore_as_ml and zscore_available) else None

        if gwl_z is not None:
            print(
                f"  [GWL] Auto-selected meters column {gwl_m!r} for physics "
                f"(preferred). Z-score column {gwl_z!r} exists; it will NOT be used for physics."
            )
        else:
            print(f"  [GWL] Auto-selected meters column {gwl_m!r} for physics.")
        return gwl_m, gwl_z

    # --- Case C: no meters column exists ---
    if zscore_available:
        raise ValueError(
            "No groundwater column in meters was found (expected 'GWL_depth_bgs' or 'GWL'), "
            f"but z-score column(s) exist: {zscore_available}. "
            "GeoPrior physics constraints require groundwater in meters. "
            "Please provide/restore a meters column in the dataset."
        )

    raise ValueError(
        "No groundwater column found. Expected 'GWL_depth_bgs' or 'GWL' (meters) "
        "for physics (and optionally 'GWL_depth_bgs_z' for ML)."
    )


# ---- Apply the resolver ----
GWL_METERS_COL, GWL_ZSCORE_COL = resolve_gwl_for_physics(
    df_raw,
    gwl_col_user=GWL_COL,   # may be None/'' per your rule
    prefer_depth_bgs=True,
    allow_keep_zscore_as_ml=True,
)

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

# Optional: keep z-score as ML convenience column if present
if (GWL_ZSCORE_COL is not None) and (GWL_ZSCORE_COL in df_raw.columns):
    base_select.append(GWL_ZSCORE_COL)

# Add resolved optional columns (numeric + categorical)
base_select += opt_num_cols + opt_cat_cols

selected = _drop_missing_keep_order(base_select, df_raw)

missing_required = [
    c for c in [LON_COL, LAT_COL, TIME_COL, SUBSIDENCE_COL, GWL_METERS_COL, H_FIELD_COL_NAME]
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
#     These columns are NEVER scaled by ML scalers.
# ------------------------------------------------------------------
SUBS_COL_MODEL = f"{SUBSIDENCE_COL}__si"   # meters
GWL_COL_MODEL  = f"{GWL_SRC_COL}__si"      # meters  (GWL_SRC_COL is meters, never None)
H_COL_MODEL    = f"{H_FIELD_SRC_COL}__si"  # meters

# Subsidence: mm -> m (if SUBS_UNIT_TO_SI=1e-3), stored in __si
df_proc[SUBS_COL_MODEL] = (
    pd.to_numeric(df_proc[SUBSIDENCE_COL], errors="coerce").astype(float)
    * float(SUBS_UNIT_TO_SI)
).astype(np.float32)

# Groundwater: meters already (depth_bgs convention)
df_proc[GWL_COL_MODEL] = (
    pd.to_numeric(df_proc[GWL_SRC_COL], errors="coerce").astype(float)
).astype(np.float32)

# Thickness: apply THICKNESS_UNIT_TO_SI exactly once here
df_proc[H_COL_MODEL] = (
    pd.to_numeric(df_proc[H_FIELD_SRC_COL], errors="coerce").astype(float)
    * float(THICKNESS_UNIT_TO_SI)
).astype(np.float32)

# ------------------------------------------------------------------
# 3.2 Coordinate system for physics: ALWAYS meters internally
#     (this block unchanged except it now runs after SI columns exist)
# ------------------------------------------------------------------
COORD_MODE = str(cfg.get("COORD_MODE", "degrees")).lower()
UTM_EPSG = int(cfg.get("UTM_EPSG", 32649))
COORD_SRC_EPSG    = cfg.get("COORD_SRC_EPSG", None)      # required for degrees
COORD_TARGET_EPSG = int(cfg.get("COORD_TARGET_EPSG", UTM_EPSG))

COORD_X_COL, COORD_Y_COL = LON_COL, LAT_COL
coord_epsg = None
coords_in_degrees = bool(COORD_MODE == "degrees")

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
    "KEEP_COORDS_RAW": KEEP_COORDS_RAW,
}))

config_sections.insert(6, ("Physics SI", {
    "SUBS_UNIT_TO_SI": SUBS_UNIT_TO_SI,
    "THICKNESS_UNIT_TO_SI": THICKNESS_UNIT_TO_SI,
    "SCALE_GWL": SCALE_GWL,
    "SUBS_COL_MODEL": SUBS_COL_MODEL,
    "GWL_COL_MODEL": GWL_COL_MODEL,
    "H_COL_MODEL": H_COL_MODEL,
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
# ------------------------------------------------------------------
ml_numeric_candidates = []
ml_numeric_candidates += opt_num_cols[:]  # rainfall, urban_load, etc.
ml_numeric_candidates += _drop_missing_keep_order(censor_numeric_additions, df_proc)

# Anything that influences physics or targets must NEVER be scaled here.
physics_locked = {
    # SI columns (always locked)
    SUBS_COL_MODEL, GWL_COL_MODEL, H_COL_MODEL,
    # physics coords/time (raw or projected)
    TIME_COL_NUM, COORD_X_COL, COORD_Y_COL,
    # raw columns used to build SI
    SUBSIDENCE_COL, H_FIELD_COL_NAME, H_FIELD_SRC_COL,
    GWL_SRC_COL, GWL_METERS_COL,
    # plus the ML alias column if you want to keep it unscaled
    # (usually you DO want it scaled if it's a z-score / ML-only driver)
}

ml_numeric_cols = [
    c for c in _drop_missing_keep_order(ml_numeric_candidates, df_proc)
    if c not in physics_locked
]
ml_numeric_cols = [c for c in ml_numeric_cols if c not in set(ALREADY_NORMALIZED_FEATURES)]

df_scaled = df_proc.copy()
scaler_path = None

if ml_numeric_cols:
    scaler = MinMaxScaler()
    df_scaled[ml_numeric_cols] = scaler.fit_transform(df_scaled[ml_numeric_cols])
    scaler_path = os.path.join(ARTIFACTS_DIR, f"{CITY_NAME}_main_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler (ML drivers only): {scaler_path}")
else:
    scaler = None
    print("  [Info] No ML numeric features to scale.")

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

H_FIELD_COL = H_COL_MODEL
print(f"[SI affine] H_field identity (meters already): {H_FIELD_COL}")

      
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

GROUP_ID_COLS = [LON_COL, LAT_COL] # here for 
# --- base features
static_features = encoded_names[:]

dynamic_base  = [GWL_COL_MODEL]
dynamic_extra = [c for c in opt_num_cols if c not in {SUBSIDENCE_COL, H_FIELD_COL_NAME}]
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
gwl_dyn_index = int(dynamic_features.index(GWL_COL_MODEL))

print(f"  Static : {static_features}")
print(f"  Dynamic: {dynamic_features}")
print(f"  Future : {future_features}")
print(f"  H_field: {H_FIELD_COL}")

# ==================================================================
# Step 5: Split & build TRAIN sequences
# ==================================================================
print(f"\n{'='*18} Step 5: Train Split & Sequences {'='*18}")
df_train = df_scaled[df_scaled[TIME_COL] <= TRAIN_END_YEAR].copy()
if df_train.empty:
    raise ValueError(f"Empty train split at year={TRAIN_END_YEAR}.")

seq_job = os.path.join(
    ARTIFACTS_DIR, 
    f"{CITY_NAME}_train_sequences_T{TIME_STEPS}_H{FORECAST_HORIZON_YEARS}.joblib"
)

OUT_S_DIM, OUT_G_DIM = 1, 1
normalize_coords = not KEEP_COORDS_RAW

inputs_train, targets_train, coord_scaler = prepare_pinn_data_sequences(
    df=df_train,
    time_col=TIME_COL_USED,
    lon_col=X_COL_USED,
    lat_col=Y_COL_USED,

    # Use SI columns for physics
    subsidence_col=SUBS_COL_MODEL,
    gwl_col=GWL_COL_MODEL,
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
    return_coord_scaler=normalize_coords,

    savefile=seq_job,
    mode=MODE,
    model=MODEL_NAME,
    verbose=2,
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

# --------------------------------------------------------------
# coord ranges for chain-rule in Stage-2
# --------------------------------------------------------------
if normalize_coords:
    cmin = np.asarray(coord_scaler.data_min_, dtype=float)
    cmax = np.asarray(coord_scaler.data_max_, dtype=float)
    crng = (cmax - cmin)
    coord_ranges = {"t": float(crng[0]), "x": float(crng[1]), "y": float(crng[2])}
else:
    coords_raw = np.asarray(inputs_train["coords"], dtype=float)  # (N,H,3)
    coord_ranges = {
        "t": float(coords_raw[..., 0].max() - coords_raw[..., 0].min()),
        "x": float(coords_raw[..., 1].max() - coords_raw[..., 1].min()),
        "y": float(coords_raw[..., 2].max() - coords_raw[..., 2].min()),
    }


for k, v in inputs_train.items():
    print(f"  Train input '{k}': {None if v is None else v.shape}")
for k, v in targets_train.items():
    print(f"  Train target '{k}': {v.shape}")


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

# Deterministic split indices
val_size = int(0.2 * num_train)
val_size = max(1, min(val_size, num_train - 1))

rng = np.random.default_rng(42)
perm = rng.permutation(num_train)
val_idx = perm[:val_size]
train_idx = perm[val_size:]

train_np_x = {k: v[train_idx] for k, v in dataset_inputs.items()}
train_np_y = {k: v[train_idx] for k, v in dataset_targets.items()}
val_np_x   = {k: v[val_idx]   for k, v in dataset_inputs.items()}
val_np_y   = {k: v[val_idx]   for k, v in dataset_targets.items()}

train_inputs_npz  = os.path.join(ARTIFACTS_DIR, "train_inputs.npz")
train_targets_npz = os.path.join(ARTIFACTS_DIR, "train_targets.npz")
val_inputs_npz    = os.path.join(ARTIFACTS_DIR, "val_inputs.npz")
val_targets_npz   = os.path.join(ARTIFACTS_DIR, "val_targets.npz")

_save_npz(train_inputs_npz, train_np_x)
_save_npz(train_targets_npz, train_np_y)
_save_npz(val_inputs_npz, val_np_x)
_save_npz(val_targets_npz, val_np_y)

print(f"  Saved NPZs:\n    {train_inputs_npz}\n    {train_targets_npz}\n"
      f"    {val_inputs_npz}\n    {val_targets_npz}")

# (Optional) only if you still want tf.data for a tiny sanity check
# train_ds = tf.data.Dataset.from_tensor_slices((train_np_x, train_np_y)).batch(32).prefetch(1)
# val_ds   = tf.data.Dataset.from_tensor_slices((val_np_x, val_np_y)).batch(32).prefetch(1)

# ==================================================================
# Manifest: one file to rule them all
# ==================================================================
print(f"\n{'='*18} Build Manifest {'='*18}")

manifest_censor = {
    "specs": CENSORING_SPECS,
    "report": censor_report,
    "use_effective_h_field": USE_EFFECTIVE_H_FIELD,
    "flags_as_dynamic": INCLUDE_CENSOR_FLAGS_AS_DYNAMIC
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
            "ohe": ohe_paths,   # dict: {col: path}# ohe_path,
            "coord_scaler": coord_scaler_path,
            "main_scaler": scaler_path,
            "scaled_ml_numeric_cols": scaled_ml_numeric_cols,

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

manifest["config"].setdefault("scaling_kwargs", {})
manifest["config"]["scaling_kwargs"].update({
    "subs_scale_si": subs_scale_si,
    "subs_bias_si": subs_bias_si,
    "head_scale_si": head_scale_si,
    "head_bias_si": head_bias_si,
    "H_scale_si": H_scale_si,
    "H_bias_si": H_bias_si,
    
    # coords
    "coords_normalized":  bool(normalize_coords),
    "coord_order": ["t", "x", "y"],
    "coord_ranges": coord_ranges,
    "coord_x_col": COORD_X_COL,
    "coord_y_col": COORD_Y_COL,

    # coordinate provenance (NO deg-to-m hacks needed)
    "coord_mode": COORD_MODE,
    "coord_src_epsg": int(COORD_SRC_EPSG) if COORD_SRC_EPSG is not None else None,
    "coord_target_epsg": int(COORD_TARGET_EPSG),
    "coord_epsg_used": int(coord_epsg) if coord_epsg is not None else None,
    "coords_in_degrees": coords_in_degrees, 

    "gwl_z_meta": gwl_z_meta,
    "gwl_kind": str(cfg.get("GWL_KIND", "depth_bgs")),
    "gwl_sign": str(cfg.get("GWL_SIGN", "down_positive")),
    "use_head_proxy": bool(cfg.get("USE_HEAD_PROXY", True)),
    "z_surf_col": cfg.get("Z_SURF_COL", None),
    "gwl_col": GWL_COL,
    "gwl_raw_col": gwl_z_meta.get("raw_col") if gwl_z_meta else None,
    "gwl_dyn_name": GWL_COL_MODEL,
    "gwl_dyn_index": gwl_dyn_index,    

    "dynamic_feature_names": list(dynamic_features),
    "future_feature_names":  list(future_features),

})

# IMPORTANT: Stage-1 created __si columns (meters), so model inputs are SI.
manifest["config"]["scaling_kwargs"].update({
    # make the "unit conversions" explicit no-ops:
    "subs_unit_to_si": 1.0,
    "head_unit_to_si": 1.0,
    "thickness_unit_to_si": 1.0,
})

manifest["config"]["scaling_kwargs"] = finalize_si_scaling_kwargs(
    manifest["config"]["scaling_kwargs"],
    subs_in_si=True,        # subsidence_col = SUBS_COL_MODEL in meters
    head_in_si=True,        # gwl_col = GWL_COL_MODEL in meters (depth or head, still meters)
    thickness_in_si=True,   # h_field_col = H_COL_MODEL in meters
    force_identity_affine_if_si= True, 
    warn=True,
)

if coord_shift_pack is not None:
    manifest["config"]["scaling_kwargs"]["coord_shift_mins"] = coord_shift_pack.coord_mins
    manifest["config"]["scaling_kwargs"]["coord_shift_meta"] = coord_shift_pack.meta

manifest["config"]["feature_registry"] = {
    "optional_numeric_declared": OPTIONAL_NUMERIC_FEATURES,
    "optional_categorical_declared": OPTIONAL_CATEGORICAL_FEATURES,
    "already_normalized": ALREADY_NORMALIZED_FEATURES,
    "future_drivers_declared": FUTURE_DRIVER_FEATURES,
    "resolved_optional_numeric": opt_num_cols,
    "resolved_optional_categorical": opt_cat_cols,
}
manifest["config"]["censoring"] = manifest_censor

manifest["config"].setdefault("units_provenance", {})
manifest["config"]["units_provenance"].update({
    "subs_unit_to_si_applied_stage1": float(SUBS_UNIT_TO_SI),
    "thickness_unit_to_si_applied_stage1": float(THICKNESS_UNIT_TO_SI),
})

# === Step 5b: Build TEST sequences (temporal generalization) ===
df_test = df_scaled[df_scaled[TIME_COL] >= FORECAST_START_YEAR].copy()
test_inputs = test_targets = None
if not df_test.empty:
    try: 
        test_inputs, test_targets, _ = prepare_pinn_data_sequences(
            df=df_test,
            time_col=TIME_COL_USED,
            lon_col=X_COL_USED,
            lat_col=Y_COL_USED,
            subsidence_col=SUBS_COL_MODEL,
            gwl_col=GWL_COL_MODEL,
            h_field_col=H_COL_MODEL,
        
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

    
        # Export NPZs (mirror train/val)
        test_inputs_np, test_targets_np = {}, {}
        if test_inputs:
            # fill Nones to zeros like you do for train/val
            N = test_inputs["dynamic_features"].shape[0]
            static_arr = test_inputs.get("static_features")
            if static_arr is None:
                static_arr = np.zeros((N, 0), np.float32)
                
            fut_T = TIME_STEPS + FORECAST_HORIZON_YEARS if MODE == "tft_like" else FORECAST_HORIZON_YEARS
            future_arr = test_inputs.get("future_features")
            if future_arr is None:
                future_arr = np.zeros((N, fut_T, 0), np.float32)
            test_inputs_np = {
                "coords": test_inputs["coords"],
                "dynamic_features": test_inputs["dynamic_features"],
                "static_features": static_arr,
                "future_features": future_arr,
                "H_field": test_inputs["H_field"],
            }
            # test_targets_np = {
            #     "subsidence": test_targets["subsidence"],
            #     "gwl": test_targets["gwl"],
            # }
            test_targets_np = {
                "subs_pred": test_targets["subsidence"],
                "gwl_pred":  test_targets["gwl"],
            }
            test_inputs_npz  = os.path.join(ARTIFACTS_DIR, "test_inputs.npz")
            test_targets_npz = os.path.join(ARTIFACTS_DIR, "test_targets.npz")
            _save_npz(test_inputs_npz, test_inputs_np)
            _save_npz(test_targets_npz, test_targets_np)
            manifest["artifacts"]["numpy"]["test_inputs_npz"]  = test_inputs_npz
            manifest["artifacts"]["numpy"]["test_targets_npz"] = test_targets_npz
    except: 
         pass 

# === Step 7: Build *true future* sequences (for Stage-3) ===

future_npz_paths = None
if BUILD_FUTURE_NPZ:
    print(f"\n{'='*18} Step 7: Future Sequences {'='*18}")
    try:
        future_npz_paths = build_future_sequences_npz(
            df_scaled=df_scaled,
            time_col=TIME_COL,
            time_col_num=TIME_COL_NUM,
            lon_col=COORD_X_COL,
            lat_col=COORD_Y_COL,
            subs_col=SUBS_COL_MODEL,
            gwl_col=GWL_COL_MODEL,
            h_field_col=H_COL_MODEL,
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

manifest_path = os.path.join(RUN_OUTPUT_PATH, "manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"  Saved manifest: {manifest_path}")

print(f"\n{'-'*_TW}\nSTAGE-1 COMPLETE. Artifacts in:\n  {RUN_OUTPUT_PATH}\n{'-'*_TW}")
