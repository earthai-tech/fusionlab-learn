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
USE_EFFECTIVE_H_FIELD = bool(
    _censor_cfg.get("use_effective_h_field", True)
)

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
        "GWL_COL": GWL_COL,
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
        "USE_EFFECTIVE_H_FIELD": USE_EFFECTIVE_H_FIELD,
    }),
    ("Outputs", {
        "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
        "RUN_OUTPUT_PATH": RUN_OUTPUT_PATH,
        "ARTIFACTS_DIR": ARTIFACTS_DIR,
    }),
]

print_config_table(
    config_sections,table_width=_TW, 
    title=f"{CITY_NAME.upper()} {MODEL_NAME} STAGE-1 CONFIG",
)

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
    return df, report

# ==================================================================
# Step 1: Load dataset
# ==================================================================

print(f"\n{'='*18} Step 1: Load Dataset {'='*18}")

def _any_exists(paths: list[str]) -> bool:
    return any(os.path.exists(p) for p in paths)

# ---  ensure city CSV exists; if not, try to unpack from merged parquet ---
if not _any_exists(SEARCH_PATHS) and ALL_CITIES_PARQUET:
    merged_path = None
    for p in ALL_CITIES_SEARCH_PATHS:
        print(f"  [Check] Looking for merged parquet at: {os.path.abspath(p)}")
        if os.path.exists(p):
            merged_path = p
            break

    if merged_path is not None:
        print(
            f"  [Info] No '{BIG_FN}' found for city={CITY_NAME!r}; "
            f"unpacking from merged parquet: {merged_path}"
        )
        # This will create e.g. 'nansha_final_main_std.harmonized.csv',
        # 'zhongshan_final_main_std.harmonized.csv' in the same folder,
        # using the 'source' column recorded when you merged.
        unpack_frames_from_file(
            merged=merged_path,
            group_col="city",
            output_dir=os.path.dirname(merged_path),
            output_format="csv",      # default: CSV outputs
            use_source_col=True,      # reuse original file names if present
            source_col="source",
            drop_columns=("source",), # optional: strip bookkeeping column
            save=True,
            return_dict=False,
            verbose=1,
        )
    else:
        print(
            "  [Info] ALL_CITIES_PARQUET is set, but merged parquet not found.\n"
            "         Will proceed with normal CSV search / fetch_zhongshan_data."
        )

# --- ORIGINAL CSV loading logic  ---
df_raw = None
for p in SEARCH_PATHS + FALLBACK_PATHS:
    print(f"  Try: {os.path.abspath(p)}")
    if os.path.exists(p):
        try:
            df_raw = pd.read_csv(p)
            print(f"    Loaded {os.path.basename(p)} -> {df_raw.shape}")
            if BIG_FN in p and df_raw.shape[0] < 400_000:
                print("    [Warn] 500k filename but rows < 400k.")
            break
        except Exception as e:
            print(f"    Error reading {p}: {e}")

if df_raw is None or df_raw.empty:
    print("  No CSV found. Try fusionlab.datasets.fetch_zhongshan_data() ...")
    try:
        bunch = fetch_zhongshan_data(verbose=1)
        df_raw = bunch.frame
        print(f"    Loaded via fetch_zhongshan_data -> {df_raw.shape}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load dataset: {e}")

raw_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_01_raw.csv")
df_raw.to_csv(raw_csv, index=False)
print(f"  Saved: {raw_csv}")


# ==================================================================
# Step 2: Initial preprocessing
# ==================================================================

print(f"\n{'='*18} Step 2: Initial Preprocessing {'='*18}")
avail = df_raw.columns.tolist()

# 2.1 Resolve optional columns (as-declared in config)
opt_num_cols, opt_num_map = _resolve_optional_columns(
    df_raw, OPTIONAL_NUMERIC_FEATURES
    )
opt_cat_cols, opt_cat_map = _resolve_optional_columns(
    df_raw, OPTIONAL_CATEGORICAL_FEATURES
    )

# 2.2 Build selection list (required + resolved optional)
base_select = [
    LON_COL, LAT_COL, TIME_COL, SUBSIDENCE_COL, GWL_COL, H_FIELD_COL_NAME
]
base_select += opt_num_cols + opt_cat_cols

selected = _drop_missing_keep_order(base_select, df_raw)
missing_required = [
    c for c in [LON_COL, LAT_COL, 
                TIME_COL, 
                SUBSIDENCE_COL, 
                GWL_COL, 
                H_FIELD_COL_NAME]
              if c not in selected
        ]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# Note: Optional columns are fine to be absent – they’ll just be skipped downstream.
skipped_optional = sorted(set(opt_num_cols + opt_cat_cols) - set(selected))
if skipped_optional:
    print(f"  [Info] Optional columns not found (skipped): {skipped_optional}")

df_sel = df_raw[selected].copy()

DT_TMP = "datetime_temp"
try:
    df_sel[DT_TMP] = pd.to_datetime(df_sel[TIME_COL], format="%Y")
except Exception:
    df_sel = normalize_time_column(
        df_sel, time_col=TIME_COL, datetime_col=DT_TMP,
        year_col=TIME_COL, drop_orig=True
    )

print(f"  Shape after select: {df_sel.shape}")
print(f"  NaNs before clean: {df_sel.isna().sum().sum()}")
df_clean = nan_ops(df_sel, ops="sanitize", action="fill", verbose=0)
print(f"  NaNs after clean : {df_clean.isna().sum().sum()}")

clean_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_02_clean.csv")
df_clean.to_csv(clean_csv, index=False)
print(f"  Saved: {clean_csv}")

print(f"\n{'='*18} Step 2.5: Censor-aware transforms {'='*18}")
df_cens, censor_report = _apply_censoring(df_clean.copy(), CENSORING_SPECS)

# ==================================================================
# Step 3: Encode & Scale
# ==================================================================

print(f"\n{'='*18} Step 3: Encode & Scale {'='*18}")
df_proc = df_cens.copy()

# 3.1 One-hot encode ALL optional categoricals that are present
encoded_names = []
ohe_paths = {}  # support multiple encoders (one per categorical)
for cat_col in _drop_missing_keep_order(opt_cat_cols, df_proc):
    ohe = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore", dtype=np.float32)
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
    
    # joblib.dump(ohe, path)
    ohe_paths[cat_col] = path
    print(f"  Saved OHE for '{cat_col}': {path}")

# 3.2 Numeric time coordinate for PINN
TIME_COL_NUM = f"{TIME_COL}_numeric_coord"
df_proc[TIME_COL_NUM] = (
    df_proc[DT_TMP].dt.year
    + (df_proc[DT_TMP].dt.dayofyear - 1)
    / (365 + df_proc[DT_TMP].dt.is_leap_year.astype(int))
)

# If censoring created <col>_eff and/or <col>_censored, include them
censor_numeric_additions = []
for sp in CENSORING_SPECS or []:
    col = sp["col"]
    feff  = col + sp.get("eff_suffix",  "_eff")
    fflag = col + sp.get("flag_suffix", "_censored")
    if feff in df_proc.columns:
        censor_numeric_additions.append(feff)
    if INCLUDE_CENSOR_FLAGS_AS_DYNAMIC and fflag in df_proc.columns:
        censor_numeric_additions.append(fflag)


# 3.3 Scale numeric features (skip those declared already-normalized)
present_num = _drop_missing_keep_order(
    [GWL_COL, H_FIELD_COL_NAME, SUBSIDENCE_COL] + opt_num_cols, df_proc
) + _drop_missing_keep_order(censor_numeric_additions, df_proc)

# present_num = _drop_missing_keep_order(
#     [GWL_COL, H_FIELD_COL_NAME, SUBSIDENCE_COL] + opt_num_cols, df_proc
# )
num_cols = [c for c in present_num if c not in set(ALREADY_NORMALIZED_FEATURES)]

df_scaled = df_proc.copy()
scaler_info, scaler_path = {}, None
if num_cols:
    scaler = MinMaxScaler()
    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    scaler_path = os.path.join(ARTIFACTS_DIR, f"{CITY_NAME}_main_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler (joblib): {scaler_path}")

    # indices for inverse-transform of targets only
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
print(f"  Saved: {scaled_csv}")

# ==================================================================
# Step 4: Feature sets (lists only)
# ==================================================================

print(f"\n{'='*18} Step 4: Define Feature Sets {'='*18}")
static_features = encoded_names[:]  # all OHE columns added above

# dynamic = required GWL + all optional numeric that are present (except targets/H_field)
dynamic_base = [GWL_COL]
dynamic_extra = [c for c in opt_num_cols if c not in {SUBSIDENCE_COL, H_FIELD_COL_NAME}]
dynamic_features = [c for c in dynamic_base + dynamic_extra if c in df_scaled.columns]

# future = any of FUTURE_DRIVER_FEATURES that are present
future_features = [c for c in FUTURE_DRIVER_FEATURES if c in df_scaled.columns]

H_FIELD_COL = H_FIELD_COL_NAME
# If soil thickness is censored and we created an effective column, use it.
for sp in CENSORING_SPECS or []:
    if sp["col"] == H_FIELD_COL_NAME:
        eff = H_FIELD_COL_NAME + sp.get("eff_suffix", "_eff")
        if USE_EFFECTIVE_H_FIELD and eff in df_scaled.columns:
            H_FIELD_COL = eff    # this is what Stage-1 will feed as H_field
            break

GROUP_ID_COLS = [LON_COL, LAT_COL]

for sp in CENSORING_SPECS or []:
    fflag = sp["col"] + sp.get("flag_suffix", "_censored")
    if INCLUDE_CENSOR_FLAGS_AS_DYNAMIC and fflag in df_scaled.columns:
        dynamic_features = dynamic_features + [fflag]

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
)
if targets_train["subsidence"].shape[0] == 0:
    raise ValueError("No training sequences were generated.")

coord_scaler_path = os.path.join(ARTIFACTS_DIR, f"{CITY_NAME}_coord_scaler.joblib")
joblib.dump(coord_scaler, coord_scaler_path)
print(f"  Saved coord scaler: {coord_scaler_path}")

for k, v in inputs_train.items():
    print(f"  Train input '{k}': {None if v is None else v.shape}")
for k, v in targets_train.items():
    print(f"  Train target '{k}': {v.shape}")


# ==================================================================
# Step 6: Build train/val datasets & EXPORT numpy
# ==================================================================
print(f"\n{'='*18} Step 6: TF Datasets & Export {'='*18}")

num_train = inputs_train["dynamic_features"].shape[0]
future_time_dim = TIME_STEPS + FORECAST_HORIZON_YEARS if MODE == "tft_like" \
                  else FORECAST_HORIZON_YEARS

# Create dense inputs dict (replace Nones with valid shapes)
static_arr = inputs_train.get("static_features")
if static_arr is None:
    static_arr = np.zeros((num_train, 0), dtype=np.float32)

future_arr = inputs_train.get("future_features")
if future_arr is None:
    future_arr = np.zeros((num_train, future_time_dim, 0), dtype=np.float32)

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

full_ds = tf.data.Dataset.from_tensor_slices((dataset_inputs, dataset_targets))
val_size = int(0.2 * num_train)
train_size = num_train - val_size
full_ds = full_ds.shuffle(buffer_size=num_train, seed=42)
train_ds = full_ds.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = full_ds.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)

# Export to numpy so Stage-2 (tuner or trainer) can np.load
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

print(f"  Saved NPZs:\n    {train_inputs_npz}\n    {train_targets_npz}\n"
      f"    {val_inputs_npz}\n    {val_targets_npz}")


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
            "main_scaler": scaler_path if num_cols else None,
            "coord_scaler": coord_scaler_path,
            "scaler_info": scaler_info,  # indices for inverse transform
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

manifest["config"]["feature_registry"] = {
    "optional_numeric_declared": OPTIONAL_NUMERIC_FEATURES,
    "optional_categorical_declared": OPTIONAL_CATEGORICAL_FEATURES,
    "already_normalized": ALREADY_NORMALIZED_FEATURES,
    "future_drivers_declared": FUTURE_DRIVER_FEATURES,
    "resolved_optional_numeric": opt_num_cols,
    "resolved_optional_categorical": opt_cat_cols,
}
manifest["config"]["censoring"] = manifest_censor

# === Step 5b: Build TEST sequences (temporal generalization) ===
df_test = df_scaled[df_scaled[TIME_COL] >= FORECAST_START_YEAR].copy()
test_inputs = test_targets = None
if not df_test.empty:
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
        )
    
        # Export NPZs (mirror train/val)
        test_inputs_np, test_targets_np = {}, {}
        if test_inputs:
            # fill Nones to zeros like you do for train/val
            N = test_inputs["dynamic_features"].shape[0]
            static_arr = test_inputs.get("static_features") or np.zeros((N, 0), np.float32)
            fut_T = TIME_STEPS + FORECAST_HORIZON_YEARS if MODE == "tft_like" else FORECAST_HORIZON_YEARS
            future_arr = test_inputs.get("future_features") or np.zeros((N, fut_T, 0), np.float32)
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
