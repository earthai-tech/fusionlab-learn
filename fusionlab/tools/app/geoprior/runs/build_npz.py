# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
NPZ-only helper: build *future/inference* NPZs for GeoPriorSubsNet.

This is a trimmed-down variant of Stage-1 that:

- assumes the dataset has already been saved by the GUI
  (typically <gui_runs_root>/_datasets via ``open_dataset_with_editor``);
- reuses the same preprocessing / encoding / scaling / feature
  resolution as :func:`run_stage1`;
- only builds *future* NPZs using :func:`build_future_sequences_npz`;
- writes a small, separate manifest (``manifest_npz.json``) under
  ``<BASE_OUTPUT_DIR>/<CITY>_<MODEL>_npz/`` without touching the main
  Stage-1 run directory.

Intended usage from the Tools > "Build NPZ" tab:

    csv_path, df, city = open_dataset_with_editor(...)
    cfg_overrides = {
        "CITY_NAME": city,
        "TRAIN_END_YEAR": 2023,
        "FORECAST_START_YEAR": 2024,
        "FORECAST_HORIZON_YEARS": 7,
        "TIME_STEPS": 5,
        "MODE": "tft_like",
        # plus column names if not coming from base_cfg
    }
    run_build_npz(
        edited_df=df,
        csv_path=str(csv_path),
        cfg_overrides=cfg_overrides,
        results_root=gui_runs_root,
        logger=append_log,
        progress_callback=progress_cb,
        stop_check=stop_flag.is_set,
    )
"""

from __future__ import annotations

import os
import json
import joblib
import warnings
import datetime as dt
from typing import Dict, Optional, Callable, Any

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

from .run_stage1 import (
    _resolve_optional_columns,
    _drop_missing_keep_order,
    _apply_censoring,
    # _distinct_preserve_order,
    # _filter_present_features,
    _resolve_feature_sets,
    _find_latest_gui_dataset,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")
if hasattr(tf, "autograph") and hasattr(tf.autograph, "set_verbosity"):
    tf.autograph.set_verbosity(0)


def run_build_npz(
    cfg_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    base_cfg: Optional[Dict[str, Any]] = None,
    results_root: Optional[os.PathLike | str] = None,
    edited_df: Optional[pd.DataFrame] = None,
    csv_path: Optional[os.PathLike | str] = None,
    clean_run_dir: bool = False,
) -> Dict[str, Any]:
    """
    Build *future/inference* NPZs only (no train/val/test NPZs).

    This function mirrors the preprocessing and feature resolution
    of :func:`run_stage1` but skips training/validation sequence
    generation. It only calls :func:`build_future_sequences_npz` and
    writes a dedicated ``manifest_npz.json``.

    Parameters
    ----------
    cfg_overrides : dict, optional
        Flat overrides for the NAT-style config (same contract as
        in :func:`run_stage1`). Typically includes:

        - ``CITY_NAME``
        - ``TRAIN_END_YEAR``
        - ``FORECAST_START_YEAR``
        - ``FORECAST_HORIZON_YEARS``
        - ``TIME_STEPS``
        - ``MODE``

        and the relevant column names if they are not coming from
        ``base_cfg``.
    logger : callable, optional
        ``f(msg: str) -> None`` for logging (GUI uses
        ``self.append_log``).
    stop_check : callable, optional
        ``f() -> bool`` cooperative stop flag. If it returns ``True``
        at certain checkpoints, the run is aborted with
        :class:`InterruptedError`.
    progress_callback : callable, optional
        ``f(fraction: float, msg: str) -> None`` progress updates for
        the GUI progress bar.
    base_cfg : dict, optional
        Pre-merged NAT-style config to start from (e.g. from
        :class:`GeoPriorConfig`). If ``None``, this function will
        call :func:`load_nat_config` from the bundled ``config.py``.
    results_root : path-like, optional
        Root directory under which the NPZ run folder will be created.
        If ``None``, falls back to ``BASE_OUTPUT_DIR`` from the config
        or ``./results``.
    edited_df : DataFrame, optional
        In-memory dataset provided by the GUI (after "Open dataset…").
        If given, it takes precedence over ``csv_path`` and config
        ``DATA_DIR/BIG_FN``.
    csv_path : path-like, optional
        Explicit path to the dataset CSV (typically living under
        ``<gui_runs_root>/_datasets``). Used mainly for logging and
        reproducibility; the content is ignored if ``edited_df`` is
        also provided.
    clean_run_dir : bool, default False
        If True, delete any existing NPZ run directory before starting.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"manifest"``: the manifest dict;
        - ``"manifest_path"``: path to ``manifest_npz.json``;
        - ``"run_dir"``: absolute path to the NPZ run directory;
        - ``"artifacts_dir"``: absolute path to the artifacts folder;
        - ``"future_npz_paths"``: mapping returned by
          :func:`build_future_sequences_npz`.
    """
    log = logger or (lambda msg: print(msg, flush=True))

    def _progress(fraction: float, msg: str) -> None:
        if progress_callback is None:
            return
        try:
            f = max(0.0, min(1.0, float(fraction)))
            progress_callback(f, msg)
        except Exception:
            # Never crash the run because the GUI callback misbehaved
            pass

    def _seq_progress_future(local_frac: float) -> None:
        # Map [0, 1] in future-building → [0.60, 0.95] in NPZ build
        _progress(
            0.60 + 0.35 * float(local_frac),
            "Build NPZ: building future sequences",
        )

    def _maybe_stop(stage: str) -> None:
        if stop_check is None:
            return
        try:
            should_stop = bool(stop_check())
        except Exception:
            should_stop = False
        if should_stop:
            msg = f"[Build-NPZ] Stop requested → aborting at: {stage}."
            log(msg)
            _progress(1.0, f"Build NPZ aborted at: {stage}")
            raise InterruptedError("Build NPZ was interrupted by user.")

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------
    if base_cfg is not None:
        cfg = dict(base_cfg)
    else:
        GUI_CONFIG_DIR = os.path.dirname(__file__)
        config_root = os.path.join(os.path.dirname(GUI_CONFIG_DIR), "config")
        cfg = load_nat_config(root=config_root)

    if cfg_overrides:
        cfg.update(cfg_overrides)

    # Decide where the NPZ runs live
    if results_root is not None:
        base_output_dir = os.fspath(results_root)
        cfg["BASE_OUTPUT_DIR"] = base_output_dir
    else:
        base_output_dir = cfg.get(
            "BASE_OUTPUT_DIR", os.path.join(os.getcwd(), "results")
        )
        cfg["BASE_OUTPUT_DIR"] = base_output_dir

    CITY_NAME = cfg.get("CITY_NAME", "")
    MODEL_NAME = cfg.get("MODEL_NAME", "GeoPriorSubsNet")
    DATA_DIR = cfg.get("DATA_DIR", base_output_dir)
    BIG_FN = cfg.get("BIG_FN", "")
    SMALL_FN = cfg.get("SMALL_FN", "")

    TRAIN_END_YEAR = cfg["TRAIN_END_YEAR"]
    FORECAST_START_YEAR = cfg["FORECAST_START_YEAR"]
    FORECAST_HORIZON_YEARS = cfg["FORECAST_HORIZON_YEARS"]
    TIME_STEPS = cfg["TIME_STEPS"]
    MODE = cfg["MODE"]

    TIME_COL = cfg["TIME_COL"]
    LON_COL = cfg["LON_COL"]
    LAT_COL = cfg["LAT_COL"]
    SUBSIDENCE_COL = cfg["SUBSIDENCE_COL"]
    GWL_COL = cfg["GWL_COL"]
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
    ALREADY_NORMALIZED_FEATURES = cfg["ALREADY_NORMALIZED_FEATURES"]
    FUTURE_DRIVER_FEATURES = list(
        cfg.get("FUTURE_DRIVER_FEATURES", []) or []
    )

    STATIC_DRIVER_FEATURES = list(
        cfg.get("STATIC_DRIVER_FEATURES", []) or []
    )
    DYNAMIC_DRIVER_FEATURES = list(
        cfg.get("DYNAMIC_DRIVER_FEATURES", []) or []
    )

    _censor_cfg = cfg.get("censoring", {}) or {}
    CENSORING_SPECS = _censor_cfg.get("specs", [])
    INCLUDE_CENSOR_FLAGS_AS_DYNAMIC = bool(
        _censor_cfg.get("flags_as_dynamic", True)
    )
    USE_EFFECTIVE_H_FIELD = bool(
        _censor_cfg.get("use_effective_h_field", True)
    )

    BASE_OUTPUT_DIR = base_output_dir
    ensure_directory_exists(BASE_OUTPUT_DIR)

    # NPZ run folder is separate from Stage-1 folder
    RUN_OUTPUT_PATH = os.path.join(
        BASE_OUTPUT_DIR, f"{CITY_NAME}_{MODEL_NAME}_npz"
    )

    if clean_run_dir and os.path.isdir(RUN_OUTPUT_PATH):
        log(f"[Build-NPZ] Cleaning existing directory: {RUN_OUTPUT_PATH}")
        import shutil

        shutil.rmtree(RUN_OUTPUT_PATH)

    os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)
    ARTIFACTS_DIR = os.path.join(RUN_OUTPUT_PATH, "artifacts")
    ensure_directory_exists(ARTIFACTS_DIR)

    try:
        _TW = get_table_size()
    except Exception:
        _TW = 80

    log(
        f"\n{'-' * _TW}\n{CITY_NAME.upper()} {MODEL_NAME} BUILD-NPZ "
        f"(future-only)\n{'-' * _TW}"
    )
    log(
        f"TIME_STEPS={TIME_STEPS}, "
        f"HORIZON={FORECAST_HORIZON_YEARS}, MODE={MODE}"
    )

    config_sections = [
        (
            "Run",
            {
                "CITY_NAME": CITY_NAME,
                "MODEL_NAME": MODEL_NAME,
                "DATA_DIR": DATA_DIR,
                "BIG_FN": BIG_FN,
            },
        ),
        (
            "Time windows",
            {
                "TRAIN_END_YEAR": TRAIN_END_YEAR,
                "FORECAST_START_YEAR": FORECAST_START_YEAR,
                "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
                "TIME_STEPS": TIME_STEPS,
                "MODE": MODE,
            },
        ),
        (
            "Columns",
            {
                "TIME_COL": TIME_COL,
                "LON_COL": LON_COL,
                "LAT_COL": LAT_COL,
                "SUBSIDENCE_COL": SUBSIDENCE_COL,
                "GWL_COL": GWL_COL,
                "H_FIELD_COL_NAME": H_FIELD_COL_NAME,
            },
        ),
        (
            "Feature registry",
            {
                "OPTIONAL_NUMERIC_FEATURES": OPTIONAL_NUMERIC_FEATURES,
                "OPTIONAL_CATEGORICAL_FEATURES": OPTIONAL_CATEGORICAL_FEATURES,
                "ALREADY_NORMALIZED_FEATURES": ALREADY_NORMALIZED_FEATURES,
                "FUTURE_DRIVER_FEATURES": FUTURE_DRIVER_FEATURES,
                "STATIC_DRIVER_FEATURES": STATIC_DRIVER_FEATURES,
                "DYNAMIC_DRIVER_FEATURES": DYNAMIC_DRIVER_FEATURES,
            },
        ),
        (
            "Censoring",
            {
                "CENSORING_SPECS": CENSORING_SPECS,
                "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC": (
                    INCLUDE_CENSOR_FLAGS_AS_DYNAMIC
                ),
                "USE_EFFECTIVE_H_FIELD": USE_EFFECTIVE_H_FIELD,
            },
        ),
        (
            "Outputs",
            {
                "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
                "RUN_OUTPUT_PATH": RUN_OUTPUT_PATH,
                "ARTIFACTS_DIR": ARTIFACTS_DIR,
            },
        ),
    ]

    print_config_table(
        config_sections,
        table_width=_TW,
        title=f"{CITY_NAME.upper()} {MODEL_NAME} BUILD-NPZ CONFIG",
        print_fn=log,
    )
    _maybe_stop("after config resolved")

    # ------------------------------------------------------------------
    # STEP 1: LOAD DATASET
    # ------------------------------------------------------------------
    log(f"\n{'=' * 18} Step 1: Load Dataset {'=' * 18}")

    df_raw: Optional[pd.DataFrame] = None
    used_path: Optional[str] = None

    if edited_df is not None:
        df_raw = edited_df.copy()
        used_path = os.fspath(csv_path) if csv_path is not None else None
        log(f"  Using in-memory dataset from GUI -> {df_raw.shape}")
    else:
        # If csv_path provided, prefer it
        if csv_path is not None:
            abs_p = os.path.abspath(os.fspath(csv_path))
            log(f"  Reading dataset from explicit csv_path: {abs_p}")
            df_raw = pd.read_csv(abs_p)
            used_path = abs_p
        else:
            # Fallback: auto-detect GUI dataset, then config BIG_FN/SMALL_FN
            auto_dataset = _find_latest_gui_dataset(
                CITY_NAME,
                results_root=BASE_OUTPUT_DIR,
                logger=log,
            )
            if auto_dataset is not None:
                DATA_DIR = os.path.dirname(auto_dataset)
                BIG_FN = os.path.basename(auto_dataset)
                cfg["DATA_DIR"] = DATA_DIR
                cfg["BIG_FN"] = BIG_FN
                log(
                    "[Build-NPZ] Overriding DATA_DIR/BIG_FN from GUI dataset: "
                    f"DATA_DIR={DATA_DIR}, BIG_FN={BIG_FN}"
                )

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
        msg = "Failed to load dataset for Build-NPZ."
        raise FileNotFoundError(msg)

    if used_path:
        log(f"  [Info] Using dataset from: {used_path}")

    raw_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_01_raw_npz.csv")
    df_raw.to_csv(raw_csv, index=False)
    log(f"  Saved: {raw_csv}")

    _progress(0.10, "Build NPZ: dataset loaded")
    _maybe_stop("before preprocessing")

    # ------------------------------------------------------------------
    # STEP 2: PREPROCESS & CENSOR
    # ------------------------------------------------------------------
    log(f"\n{'=' * 18} Step 2: Initial Preprocessing {'=' * 18}")

    opt_num_cols, _ = _resolve_optional_columns(
        df_raw, OPTIONAL_NUMERIC_FEATURES
    )
    opt_cat_cols, _ = _resolve_optional_columns(
        df_raw, OPTIONAL_CATEGORICAL_FEATURES
    )

    base_select = [
        LON_COL,
        LAT_COL,
        TIME_COL,
        SUBSIDENCE_COL,
        GWL_COL,
        H_FIELD_COL_NAME,
    ]
    base_select += opt_num_cols + opt_cat_cols

    selected = _drop_missing_keep_order(base_select, df_raw)
    missing_required = [
        c
        for c in [
            LON_COL,
            LAT_COL,
            TIME_COL,
            SUBSIDENCE_COL,
            GWL_COL,
            H_FIELD_COL_NAME,
        ]
        if c not in selected
    ]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    skipped_optional = sorted(set(opt_num_cols + opt_cat_cols) - set(selected))
    if skipped_optional:
        log(
            f"  [Info] Optional columns not found (skipped): "
            f"{skipped_optional}"
        )

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

    clean_csv = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_02_clean_npz.csv")
    df_clean.to_csv(clean_csv, index=False)
    log(f"  Saved: {clean_csv}")

    log(f"\n{'=' * 18} Step 2.5: Censor-aware transforms {'=' * 18}")
    df_cens, censor_report = _apply_censoring(df_clean.copy(), CENSORING_SPECS)

    _progress(0.20, "Build NPZ: CSV cleaned & censoring applied")
    _maybe_stop("after cleaning & censoring")

    # ------------------------------------------------------------------
    # STEP 3: ENCODE & SCALE
    # ------------------------------------------------------------------
    log(f"\n{'=' * 18} Step 3: Encode & Scale {'=' * 18}")

    df_proc = df_cens.copy()

    # 3.1 One-hot encoding of categorical features
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

        path = os.path.join(
            ARTIFACTS_DIR, f"{CITY_NAME}_ohe_{cat_col}_npz.joblib"
        )
        try:
            save_job(ohe, path, append_versions=True, append_date=True)
        except Exception:
            joblib.dump(ohe, path)
        ohe_paths[cat_col] = path
        log(f"  Saved OHE for '{cat_col}': {path}")

    # 3.2 numeric time coordinate
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
        feff = col + sp.get("eff_suffix", "_eff")
        fflag = col + sp.get("flag_suffix", "_censored")
        if feff in df_proc.columns:
            censor_numeric_additions.append(feff)
        if INCLUDE_CENSOR_FLAGS_AS_DYNAMIC and fflag in df_proc.columns:
            censor_numeric_additions.append(fflag)

    present_num = _drop_missing_keep_order(
        [GWL_COL, H_FIELD_COL_NAME, SUBSIDENCE_COL] + opt_num_cols, df_proc
    ) + _drop_missing_keep_order(censor_numeric_additions, df_proc)
    num_cols = [
        c for c in present_num if c not in set(ALREADY_NORMALIZED_FEATURES)
    ]

    df_scaled = df_proc.copy()
    scaler_info, scaler_path = {}, None
    if num_cols:
        scaler = MinMaxScaler()
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
        scaler_path = os.path.join(
            ARTIFACTS_DIR, f"{CITY_NAME}_main_scaler_npz.joblib"
        )
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

    scaled_csv = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_03_scaled_npz.csv"
    )
    df_scaled.to_csv(scaled_csv, index=False)
    log(f"  Saved: {scaled_csv}")

    _progress(0.35, "Build NPZ: features scaled")
    _maybe_stop("after encode & scale")

    # ------------------------------------------------------------------
    # STEP 4: FEATURE SETS
    # ------------------------------------------------------------------
    log(f"\n{'=' * 18} Step 4: Define Feature Sets {'=' * 18}")

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

    # Decide whether to use effective H-field instead of raw
    H_FIELD_COL = H_FIELD_COL_NAME
    for sp in CENSORING_SPECS or []:
        if sp["col"] == H_FIELD_COL_NAME:
            eff = H_FIELD_COL_NAME + sp.get("eff_suffix", "_eff")
            if USE_EFFECTIVE_H_FIELD and eff in df_scaled.columns:
                H_FIELD_COL = eff
                break

    GROUP_ID_COLS = [LON_COL, LAT_COL]

    # Add censor flags as dynamic if requested
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

    _progress(0.45, "Build NPZ: feature sets defined")
    _maybe_stop("after feature sets defined")

    # ------------------------------------------------------------------
    # STEP 5: FUTURE / INFERENCE SEQUENCES ONLY
    # ------------------------------------------------------------------
    log(f"\n{'=' * 18} Step 5: Future Sequences (NPZ only) {'=' * 18}")

    _maybe_stop("before future sequence generation")

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
        stop_check=stop_check,
        progress_hook=_seq_progress_future,
    )

    log(f"  Future NPZs: {future_npz_paths}")
    _progress(0.95, "Build NPZ: future NPZs built")
    _maybe_stop("after future sequences")

    # ------------------------------------------------------------------
    # MANIFEST (NPZ-only)
    # ------------------------------------------------------------------
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
        "stage": "npz_only",
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
            "feature_registry": {
                "optional_numeric_declared": OPTIONAL_NUMERIC_FEATURES,
                "optional_categorical_declared": (
                    OPTIONAL_CATEGORICAL_FEATURES
                ),
                "already_normalized": ALREADY_NORMALIZED_FEATURES,
                "future_drivers_declared": FUTURE_DRIVER_FEATURES,
                "static_drivers_declared": STATIC_DRIVER_FEATURES,
                "dynamic_drivers_declared": DYNAMIC_DRIVER_FEATURES,
                "resolved_optional_numeric": opt_num_cols,
                "resolved_optional_categorical": opt_cat_cols,
            },
            "censoring": manifest_censor,
        },
        "artifacts": {
            "csv": {
                "raw": raw_csv,
                "clean": clean_csv,
                "scaled": scaled_csv,
            },
            "encoders": {
                "ohe": ohe_paths,
                "main_scaler": scaler_path if num_cols else None,
                "scaler_info": scaler_info,
            },
            "numpy": future_npz_paths,
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

    manifest_path = os.path.join(RUN_OUTPUT_PATH, "manifest_npz.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"  Saved NPZ manifest: {manifest_path}")
    log(
        f"\n{'-' * _TW}\nBUILD-NPZ COMPLETE. Artifacts in:\n"
        f"  {run_dir_abs}\n{'-' * _TW}"
    )

    _progress(1.0, "Build NPZ: complete")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "run_dir": run_dir_abs,
        "artifacts_dir": artifacts_dir_abs,
        "future_npz_paths": future_npz_paths,
    }
