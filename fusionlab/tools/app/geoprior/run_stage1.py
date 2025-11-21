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
import datetime as dt
from typing import Dict, Tuple, Optional, Callable, Any

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# --- Suppress common warnings/tf chatter at import time ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
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
        print_config_table,
    )
    from fusionlab.utils.sequence_utils import build_future_sequences_npz
    from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences

    _IMPORT_OK = True
except Exception as e:  # pragma: no cover
    _IMPORT_OK = False
    _IMPORT_ERR = e


# ======================================================================
# Small helpers (same behaviour as in your script)
# ======================================================================
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


def _any_exists(paths: list[str]) -> bool:
    return any(os.path.exists(p) for p in paths)


# ======================================================================
# Main entry point
# ======================================================================
def run_stage1(
    cfg_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    clean_run_dir: bool = True,
    stop_check: Callable[[], bool] = None, 
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
    if not _IMPORT_OK:
        raise RuntimeError(
            "fusionlab imports failed in run_stage1: "
            f"{_IMPORT_ERR}"
        )

    log = logger or (lambda msg: print(msg, flush=True))

    # ===================== CONFIG =====================
    cfg = load_nat_config()
    if cfg_overrides:
        cfg = {**cfg, **cfg_overrides}

    CITY_NAME   = cfg["CITY_NAME"]
    MODEL_NAME  = cfg["MODEL_NAME"]
    DATA_DIR    = cfg["DATA_DIR"]
    BIG_FN      = cfg["BIG_FN"]
    SMALL_FN    = cfg["SMALL_FN"]
    ALL_CITIES_PARQUET = cfg.get("ALL_CITIES_PARQUET")

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
    ALL_CITIES_SEARCH_PATHS = cfg.get("ALL_CITIES_SEARCH_PATHS") or [
        os.path.join(DATA_DIR, "data", ALL_CITIES_PARQUET)
        if ALL_CITIES_PARQUET else None,
        os.path.join(DATA_DIR, ALL_CITIES_PARQUET)
        if ALL_CITIES_PARQUET else None,
        os.path.join(".", "data", ALL_CITIES_PARQUET)
        if ALL_CITIES_PARQUET else None,
        ALL_CITIES_PARQUET,
    ]
    ALL_CITIES_SEARCH_PATHS = [p for p in ALL_CITIES_SEARCH_PATHS if p]

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

    OPTIONAL_NUMERIC_FEATURES     = cfg["OPTIONAL_NUMERIC_FEATURES"]
    OPTIONAL_CATEGORICAL_FEATURES = cfg["OPTIONAL_CATEGORICAL_FEATURES"]
    ALREADY_NORMALIZED_FEATURES   = cfg["ALREADY_NORMALIZED_FEATURES"]
    FUTURE_DRIVER_FEATURES        = cfg["FUTURE_DRIVER_FEATURES"]

    _censor_cfg = cfg.get("censoring", {}) or {}
    CENSORING_SPECS = _censor_cfg.get("specs", [])
    INCLUDE_CENSOR_FLAGS_AS_DYNAMIC = bool(
        _censor_cfg.get("flags_as_dynamic", True)
    )
    USE_EFFECTIVE_H_FIELD = bool(
        _censor_cfg.get("use_effective_h_field", True)
    )

    BASE_OUTPUT_DIR = cfg.get(
        "BASE_OUTPUT_DIR", os.path.join(os.getcwd(), "results")
    )
    ensure_directory_exists(BASE_OUTPUT_DIR)

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
            "OPTIONAL_CATEGORICAL_FEATURES":
                OPTIONAL_CATEGORICAL_FEATURES,
            "ALREADY_NORMALIZED_FEATURES": ALREADY_NORMALIZED_FEATURES,
            "FUTURE_DRIVER_FEATURES": FUTURE_DRIVER_FEATURES,
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

    # ===================== STEP 1: LOAD =====================
    log(f"\n{'='*18} Step 1: Load Dataset {'='*18}")

    if not _any_exists(SEARCH_PATHS) and ALL_CITIES_PARQUET:
        merged_path = None
        for p in ALL_CITIES_SEARCH_PATHS:
            log(f"  [Check] Looking for merged parquet at: {os.path.abspath(p)}")
            if os.path.exists(p):
                merged_path = p
                break
        if merged_path is not None:
            log(
                f"  [Info] No '{BIG_FN}' found for city={CITY_NAME!r}; "
                f"unpacking from merged parquet: {merged_path}"
            )
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
                logger=log
            )
        else:
            log(
                "  [Info] ALL_CITIES_PARQUET is set, but merged parquet "
                "not found.\n"
                "         Will proceed with normal CSV search / "
                "fetch_zhongshan_data."
            )

    df_raw: pd.DataFrame | None = None
    for p in SEARCH_PATHS + FALLBACK_PATHS:
        log(f"  Try: {os.path.abspath(p)}")
        if os.path.exists(p):
            try:
                df_raw = pd.read_csv(p)
                log(f"    Loaded {os.path.basename(p)} -> {df_raw.shape}")
                if BIG_FN in p and df_raw.shape[0] < 400_000:
                    log("    [Warn] 500k filename but rows < 400k.")
                break
            except Exception as e:
                log(f"    Error reading {p}: {e}")

    if df_raw is None or df_raw.empty:
        log("  No CSV found. Try fusionlab.datasets.fetch_zhongshan_data() ...")
        try:
            bunch = fetch_zhongshan_data(verbose=1)
            df_raw = bunch.frame
            log(f"    Loaded via fetch_zhongshan_data -> {df_raw.shape}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset: {e}")

    raw_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_01_raw.csv")
    df_raw.to_csv(raw_csv, index=False)
    log(f"  Saved: {raw_csv}")

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

    if stop_check and stop_check():
        raise InterruptedError("Sequence generation aborted.")
            
    # ===================== STEP 4: FEATURE SETS =====================
    log(f"\n{'='*18} Step 4: Define Feature Sets {'='*18}")

    static_features = encoded_names[:]
    dynamic_base = [GWL_COL]
    dynamic_extra = [
        c for c in opt_num_cols if c not in {SUBSIDENCE_COL, H_FIELD_COL_NAME}
    ]
    dynamic_features = [
        c for c in dynamic_base + dynamic_extra if c in df_scaled.columns
    ]
    future_features = [
        c for c in FUTURE_DRIVER_FEATURES if c in df_scaled.columns
    ]

    H_FIELD_COL = H_FIELD_COL_NAME
    for sp in CENSORING_SPECS or []:
        if sp["col"] == H_FIELD_COL_NAME:
            eff = H_FIELD_COL_NAME + sp.get("eff_suffix", "_eff")
            if USE_EFFECTIVE_H_FIELD and eff in df_scaled.columns:
                H_FIELD_COL = eff
                break


    GROUP_ID_COLS = [LON_COL, LAT_COL]
    for sp in CENSORING_SPECS or []:
        fflag = sp["col"] + sp.get("flag_suffix", "_censored")
        if INCLUDE_CENSOR_FLAGS_AS_DYNAMIC and fflag in df_scaled.columns:
            dynamic_features = dynamic_features + [fflag]

    log(f"  Static : {static_features}")
    log(f"  Dynamic: {dynamic_features}")
    log(f"  Future : {future_features}")
    log(f"  H_field: {H_FIELD_COL}")

    # ===================== STEP 5: TRAIN SPLIT & SEQUENCES =====================
    log(f"\n{'='*18} Step 5: Train Split & Sequences {'='*18}")

    df_train = df_scaled[df_scaled[TIME_COL] <= TRAIN_END_YEAR].copy()
    if df_train.empty:
        raise ValueError(f"Empty train split at year={TRAIN_END_YEAR}.")

    seq_job = os.path.join(
        ARTIFACTS_DIR,
        f"{CITY_NAME}_train_sequences_T{TIME_STEPS}_H{FORECAST_HORIZON_YEARS}.joblib",  # noqa: E501
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
        _logger=log,          # <- route internal verbose logs to GUI/CLI
        stop_check=stop_check  # <- add when you extend run_stage1 signature
    )

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

    full_ds = full_ds.shuffle(buffer_size=num_train, seed=42)
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

    # ===================== MANIFEST: ONE FILE TO RULE THEM ALL =================
    log(f"\n{'='*18} Build Manifest {'='*18}")

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
            "run_dir": RUN_OUTPUT_PATH,
            "artifacts_dir": ARTIFACTS_DIR,
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
        "resolved_optional_numeric": opt_num_cols,
        "resolved_optional_categorical": opt_cat_cols,
    }
    manifest["config"]["censoring"] = manifest_censor

    # ===================== STEP 5b: TEST SEQUENCES (TEMPORAL GEN) ==============
    df_test = df_scaled[df_scaled[TIME_COL] >= FORECAST_START_YEAR].copy()
    if not df_test.empty:
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
            )
            manifest["artifacts"]["numpy"].update(future_npz_paths)
            log(f"  Future NPZs: {future_npz_paths}")
        except Exception as e:  # pragma: no cover
            log(
                "[Warn] BUILD_FUTURE_NPZ=True but construction failed: "
                f"{e}\n"
                "       Continuing without future_* NPZs."
            )

    manifest_path = os.path.join(RUN_OUTPUT_PATH, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"  Saved manifest: {manifest_path}")
    log(
        f"\n{'-'*_TW}\nSTAGE-1 COMPLETE. Artifacts in:\n"
        f"  {RUN_OUTPUT_PATH}\n{'-'*_TW}"
    )

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "run_dir": RUN_OUTPUT_PATH,
        "artifacts_dir": ARTIFACTS_DIR,
    }
