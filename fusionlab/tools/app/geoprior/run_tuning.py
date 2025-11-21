# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
tune_NATCOM_GEOPRIOR.py
Stage-2: Hyperparameter tuning for GeoPriorSubsNet using GeoPriorTuner.

This script consumes artifacts produced by Stage-1 (manifest.json, NPZs,
encoders/scalers) and performs model tuning without re-running the data
preparation pipeline.

Pipeline
--------
1) Load Stage-1 manifest and NPZ arrays (train/val).
2) Infer fixed (non-HP) parameters from array shapes and Stage-1 config.
3) Define a compact but meaningful hyperparameter search space.
4) Create and run GeoPriorTuner.
5) Save best model and best hyperparameters to disk.

Requirements
------------
- fusionlab-learn with GeoPriorTuner & GeoPriorSubsNet
- tensorflow >= 2.12
- keras-tuner (if tuner_type='randomsearch'/'bayesian'/'hyperband')
"""

from __future__ import annotations

import os
import json
import joblib
import numpy as np
import datetime as dt
from typing import Optional, Dict, Any, Callable

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from fusionlab.api.util import get_table_size
from fusionlab.utils.generic_utils import (
    ensure_directory_exists,
    default_results_dir,
    getenv_stripped,
    print_config_table,
)
from fusionlab._optdeps import with_progress  # used later in Step 6

from fusionlab.registry.utils import _find_stage1_manifest

# Import the tuner
from fusionlab.nn.forecast_tuner import GeoPriorTuner
from fusionlab.nn.callbacks import NaNGuard

from fusionlab.utils.nat_utils import (
    load_nat_config,
    load_nat_config_payload,
    ensure_input_shapes,          # used in Step 6
    map_targets_for_training,     # used in Step 6
    make_tf_dataset,              # used in Step 6
    load_scaler_info,             # used in Step 6
    save_ablation_record,         # used in Step 6
    load_or_rebuild_geoprior_model,  # used in Step 6
    compile_for_eval,             # used in Step 6
)

from fusionlab.utils.forecast_utils import format_and_forecast   # used in Step 6
from fusionlab.utils.scale_metrics import (                      # used in Step 6
    inverse_scale_target,
    point_metrics,
    per_horizon_metrics,
)
from fusionlab.nn.keras_metrics import (                         # used in Step 6
    coverage80_fn,
    sharpness80_fn,
    _to_py,
)
from fusionlab.nn.calibration import (                           # used in Step 6
    fit_interval_calibrator_on_val,
    apply_calibrator_to_subs,
)

print("Successfully imported fusionlab modules for tuning.")


def run_tuning(
    manifest_path: Optional[str] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    evaluate_tuned: bool = False,
) -> Dict[str, Any]:
    """
    Run Stage-2 hyperparameter tuning for GeoPriorSubsNet (GeoPriorTuner).

    Parameters
    ----------
    manifest_path : str or None, default=None
        Explicit path to a Stage-1 manifest JSON. If provided, this
        overrides automatic discovery via `_find_stage1_manifest`.
        Intended for GUI use where the user selects the city/run.
    cfg_overrides : dict or None, default=None
        Optional overrides merged into the global NATCOM config
        loaded via `load_nat_config()`. Keys in `cfg_overrides`
        override the default config values.
    logger : callable or None, default=None
        Logging function. If None, falls back to built-in `print`.
        Signature should be `logger(msg: str) -> None`.
    stop_check : callable or None, default=None
        Optional callback for cooperative cancellation in GUI mode.
        If provided, should be a zero-argument callable returning
        True when tuning should stop. This is checked at coarse
        points (before running tuner).
    evaluate_tuned : bool, default=False
        Placeholder flag indicating whether the tuned model should
        be evaluated (Step 6). The actual evaluation logic is not
        yet wired into this function and will be added later.

    Returns
    -------
    info : dict
        Dictionary with key artifacts and configuration snapshot:

        - 'run_dir' : str
        - 'manifest_path' : str
        - 'city' : str
        - 'model' : str
        - 'best_model_path' : str or None
        - 'best_weights_path' : str or None
        - 'best_hps_path' : str
        - 'tuning_summary_json' : str
        - 'best_hps' : dict
    """
    # ------------------------------------------------------------------
    # Logger helper
    # ------------------------------------------------------------------
    if logger is None:
        def log(msg: str) -> None:  # noqa: D401
            print(msg)
    else:
        def log(msg: str) -> None:  # noqa: D401
            logger(msg)

    # Simple helper to check cancellation
    def should_stop() -> bool:
        return bool(stop_check and stop_check())

    # ------------------------------------------------------------------
    # 0) Load Stage-1 manifest and arrays
    # ------------------------------------------------------------------
    RESULTS_DIR = default_results_dir()  # auto-resolve

    # Desired city/model from NATCOM payload (for sanity checks/logging)
    cfg_payload = load_nat_config_payload()
    CFG_CITY = (cfg_payload.get("city") or "").strip().lower() or None
    CFG_MODEL = cfg_payload.get("model") or "GeoPriorSubsNet"

    # Load global NATCOM config (config.json) and apply overrides
    cfg_hp = load_nat_config()
    if cfg_overrides:
        cfg_hp.update(cfg_overrides)

    # Environment overrides (still honoured when manifest_path is None)
    CITY_ENV = getenv_stripped("CITY")
    MODEL_ENV = getenv_stripped("MODEL_NAME_OVERRIDE")

    CITY_HINT = CITY_ENV or CFG_CITY
    MODEL_HINT = MODEL_ENV or CFG_MODEL
    MANUAL_ENV = getenv_stripped("STAGE1_MANIFEST")

    # Decide manifest path
    if manifest_path is not None:
        MANIFEST_PATH = manifest_path
        log(f"[Stage-2] Using explicit manifest_path: {MANIFEST_PATH}")
    else:
        # If env provides STAGE1_MANIFEST, treat as manual hint
        manual = MANUAL_ENV or None
        MANIFEST_PATH = _find_stage1_manifest(
            manual=manual,                # exact manifest if provided
            base_dir=RESULTS_DIR,         # where to search
            city_hint=CITY_HINT,          # filter by city if set
            model_hint=MODEL_HINT,        # filter by model if set
            prefer="timestamp",           # or "mtime"
            required_keys=("model", "stage"),
            verbose=1,
        )

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"[Stage-2] Stage-1 manifest not found at: {MANIFEST_PATH}"
        )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        M = json.load(f)

    manifest_city = (M.get("city") or "").strip().lower()
    manifest_model = (M.get("model") or "").strip()

    log(f"[Manifest] Loaded city={manifest_city} model={manifest_model}")

    # If manifest was discovered automatically, enforce city consistency.
    # If manifest_path was passed explicitly, only warn on mismatch.
    if manifest_path is None:
        if CFG_CITY and manifest_city and manifest_city != CFG_CITY:
            raise RuntimeError(
                "[NATCOM] Stage-1 manifest city "
                f"{manifest_city!r} does not match config CITY_NAME "
                f"{CFG_CITY!r}. Run Stage-1 for this city first, or set "
                "CITY/STAGE1_MANIFEST to explicitly override.\n"
                "#>>> Windows cmd\n"
                "   $ set CITY=zhongshan\n"
                "   $ python nat.com/tune_NATCOM_GEOPRIOR.py\n"
                "\n"
                "#>>> or\n"
                "   $ set CITY=zhongshan\n"
                "   $ python nat.com/tune_NATCOM_GEOPRIOR.py\n"
            )
    else:
        if CFG_CITY and manifest_city and manifest_city != CFG_CITY:
            log(
                "[Warn] Explicit manifest_path city "
                f"{manifest_city!r} does not match NATCOM payload city "
                f"{CFG_CITY!r}. Proceeding with manifest city."
            )

    # Stage-1 config snapshot (used for shapes/time/horizon/etc.)
    cfg_stage1 = M["config"]        # what Stage-1 used to build sequences

    # Merge global config with Stage-1 snapshot (Stage-1 wins on conflicts)
    cfg = dict(cfg_hp)
    cfg.update(cfg_stage1)

    # Resolve NPZ paths from manifest
    train_inputs_npz = M["artifacts"]["numpy"]["train_inputs_npz"]
    train_targets_npz = M["artifacts"]["numpy"]["train_targets_npz"]
    val_inputs_npz = M["artifacts"]["numpy"]["val_inputs_npz"]
    val_targets_npz = M["artifacts"]["numpy"]["val_targets_npz"]

    X_train = dict(np.load(train_inputs_npz))
    y_train = dict(np.load(train_targets_npz))
    X_val = dict(np.load(val_inputs_npz))
    y_val = dict(np.load(val_targets_npz))

    def _sanitize_inputs(X: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in X.items():
            if v is None:
                continue
            v = np.asarray(v)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if v.ndim > 0 and np.isfinite(v).any():
                p99 = np.percentile(v, 99)
                if p99 > 0:
                    v = np.clip(v, -10 * p99, 10 * p99)
            X[k] = v
        if "H_field" in X:
            X["H_field"] = np.maximum(X["H_field"], 1e-3).astype(np.float32)
        return X

    def _ensure_input_shapes_np(
        X: Dict[str, Any],
        mode: str,
        horizon: int,
    ) -> Dict[str, Any]:
        """Guarantee zero-width placeholders for optional inputs."""
        X = dict(X)
        N = X["dynamic_features"].shape[0]
        T = X["dynamic_features"].shape[1]
        if X.get("static_features") is None:
            X["static_features"] = np.zeros((N, 0), dtype=np.float32)
        if X.get("future_features") is None:
            t_future = T if mode == "tft_like" else horizon
            X["future_features"] = np.zeros((N, t_future, 0), dtype=np.float32)
        return X

    X_train = _sanitize_inputs(X_train)
    X_val = _sanitize_inputs(X_val)

    CITY_NAME = M.get("city", cfg_hp.get("CITY_NAME", "unknown_city"))
    MODEL_NAME = M.get("model", cfg_hp.get("MODEL_NAME", "GeoPriorSubsNet"))

    # Time / horizon / mode MUST match Stage-1 sequences
    TIME_STEPS = cfg_stage1["TIME_STEPS"]
    FORECAST_HORIZON_YEARS = cfg_stage1["FORECAST_HORIZON_YEARS"]
    MODE = cfg_stage1["MODE"]

    # Censoring: merge Stage-1 (specs + report) and global config
    censor_stage1 = cfg_stage1.get("censoring", {}) or {}
    censor_global = cfg_hp.get("censoring", {}) or {}
    CENSOR = {**censor_stage1, **censor_global}
    USE_EFFECTIVE_H = CENSOR.get("use_effective_h_field", True)

    # Ensure placeholders exist even if Stage-1 NPZs are from an older run
    X_train = _ensure_input_shapes_np(X_train, MODE, FORECAST_HORIZON_YEARS)
    X_val = _ensure_input_shapes_np(X_val, MODE, FORECAST_HORIZON_YEARS)

    # Optional encoders/scalers (may be absent)
    enc = M["artifacts"]["encoders"]
    # main_scaler = None
    # ms_path = enc.get("main_scaler")
    # if ms_path and os.path.exists(ms_path):
    #     try:
    #         main_scaler = joblib.load(ms_path)
    #     except Exception:
    #         main_scaler = None

    coord_scaler = None
    cs_path = enc.get("coord_scaler")
    if cs_path and os.path.exists(cs_path):
        try:
            coord_scaler = joblib.load(cs_path)
        except Exception:
            coord_scaler = None

    # scaler_info = enc.get("scaler_info", {})
    scaler_info = load_scaler_info(enc)
    if isinstance(scaler_info, dict):
        for k, v in scaler_info.items():
            if isinstance(v, dict) and "scaler_path" in v and "scaler" not in v:
                p = v["scaler_path"]
                if p and os.path.exists(p):
                    try:
                        v["scaler"] = joblib.load(p)
                    except Exception:
                        pass
    # Output directory for Stage-2
    BASE_STAGE2_DIR = os.path.join(M["paths"]["run_dir"], "tuning")
    ensure_directory_exists(BASE_STAGE2_DIR)

    STAMP = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_OUTPUT_PATH = os.path.join(BASE_STAGE2_DIR, f"run_{STAMP}")
    ensure_directory_exists(RUN_OUTPUT_PATH)
    log(f"\nStage-2 outputs will be written to: {RUN_OUTPUT_PATH}")

    if should_stop():
        log("[Stage-2] stop_check=True before tuning; aborting early.")
        return {
            "run_dir": RUN_OUTPUT_PATH,
            "manifest_path": MANIFEST_PATH,
            "city": CITY_NAME,
            "model": MODEL_NAME,
            "best_model_path": None,
            "best_weights_path": None,
            "best_hps_path": None,
            "tuning_summary_json": None,
            "best_hps": {},
        }

    # ----------------------------------------------------------------------
    # 1) Sanity maps & shapes
    # ----------------------------------------------------------------------
    def map_targets_for_tuner(y_dict: dict) -> dict:
        """
        GeoPriorTuner expects 'subsidence' and 'gwl' as target keys.
        Stage-1 exported 'subs_pred' and 'gwl_pred'.
        """
        out = {}
        # prefer canonical keys if already present
        if "subsidence" in y_dict:
            out["subsidence"] = y_dict["subsidence"]
        elif "subs_pred" in y_dict:
            out["subsidence"] = y_dict["subs_pred"]
        else:
            raise KeyError("Missing 'subsidence' or 'subs_pred' in targets.")

        if "gwl" in y_dict:
            out["gwl"] = y_dict["gwl"]
        elif "gwl_pred" in y_dict:
            out["gwl"] = y_dict["gwl_pred"]
        else:
            raise KeyError("Missing 'gwl' or 'gwl_pred' in targets.")
        return out

    y_train_mapped = map_targets_for_tuner(y_train)
    y_val_mapped = map_targets_for_tuner(y_val)

    # Fixed dims from arrays
    def _dim_of(x: dict, key: str, fallback_lastdim: int = 0) -> int:
        arr = x.get(key)
        if arr is None:
            return 0
        if arr.ndim == 1:
            return 1
        return int(arr.shape[-1]) if arr.shape[-1] is not None else fallback_lastdim

    STATIC_DIM = _dim_of(X_train, "static_features", 0)
    DYNAMIC_DIM = _dim_of(X_train, "dynamic_features", 0)
    FUTURE_DIM = 0
    if "future_features" in X_train and X_train["future_features"] is not None:
        ff = X_train["future_features"]
        FUTURE_DIM = int(ff.shape[-1]) if ff.ndim >= 3 else _dim_of(
            X_train, "future_features", 0
        )

    OUT_S_DIM = M["artifacts"]["sequences"]["dims"]["output_subsidence_dim"]
    OUT_G_DIM = M["artifacts"]["sequences"]["dims"]["output_gwl_dim"]

    log("\nInferred input dims from Stage-1 arrays:")
    log(f"  static_input_dim  = {STATIC_DIM}")
    log(f"  dynamic_input_dim = {DYNAMIC_DIM}")
    log(f"  future_input_dim  = {FUTURE_DIM}")
    log(f"  OUT_S_DIM         = {OUT_S_DIM}")
    log(f"  OUT_G_DIM         = {OUT_G_DIM}")
    log(f"  TIME_STEPS        = {TIME_STEPS}")
    log(f"  HORIZON           = {FORECAST_HORIZON_YEARS}")
    log(f"  MODE              = {MODE}")

    # ----------------------------------------------------------------------
    # 2) Tuning configuration (non-HPs and HP space)
    # ----------------------------------------------------------------------
    # Training loop setup (from config.json + overrides)
    EPOCHS = cfg_hp.get("EPOCHS", 50)
    BATCH_SIZE = cfg_hp.get("BATCH_SIZE", 32)

    # Probabilistic outputs
    QUANTILES = cfg_hp.get("QUANTILES", [0.1, 0.5, 0.9])

    # PDE selection & attention levels
    PDE_MODE_CONFIG = cfg_hp.get("PDE_MODE_CONFIG", "both")
    ATTENTION_LEVELS = cfg_hp.get(
        "ATTENTION_LEVELS",
        ["cross", "hierarchical", "memory"],
    )
    SCALE_PDE_RESIDUALS = cfg_hp.get("SCALE_PDE_RESIDUALS", True)

    # 2.1 Non-HP fixed params (derived from shapes/config)
    fixed_params = {
        "static_input_dim": STATIC_DIM,
        "dynamic_input_dim": DYNAMIC_DIM,
        "future_input_dim": FUTURE_DIM,
        "output_subsidence_dim": OUT_S_DIM,
        "output_gwl_dim": OUT_G_DIM,
        "forecast_horizon": FORECAST_HORIZON_YEARS,
        "mode": MODE,
        "attention_levels": ATTENTION_LEVELS,
        "quantiles": QUANTILES,
        "loss_weights": {"subs_pred": 1.0, "gwl_pred": 0.5},
        "pde_mode": PDE_MODE_CONFIG,
        "scale_pde_residuals": SCALE_PDE_RESIDUALS,
        "use_effective_h": USE_EFFECTIVE_H,
        "architecture_config": {
            "encoder_type": "hybrid",
            "decoder_attention_stack": ATTENTION_LEVELS,
            "feature_processing": "vsn",
        },
    }

    # 2.2 Hyperparameter search space
    search_space_cfg = cfg_hp.get("TUNER_SEARCH_SPACE", None)

    if isinstance(search_space_cfg, dict) and search_space_cfg:
        search_space = search_space_cfg
    else:
        # Fallback: compact hard-coded space (centered around current good regime)
        search_space = {
            # --- Architecture (model.__init__) ---
            #
            # Centered around:
            #   EMBED_DIM      = 32
            #   HIDDEN_UNITS   = 64
            #   LSTM_UNITS     = 64
            #   ATTENTION_UNITS= 32
            #   NUMBER_HEADS   = 4
            #   DROPOUT_RATE   = 0.10
            #
            "embed_dim": [32, 48, 64],
            "hidden_units": [64, 96],
            "lstm_units": [64, 96],
            "attention_units": [32, 48],
            "num_heads": [2, 4],

            # Keep dropout near the good regime, but allow some exploration.
            "dropout_rate": {
                "type": "float",
                "min_value": 0.05,
                "max_value": 0.20,
            },

            # VSN & BatchNorm are *not* tuned here:
            #   USE_VSN       = True
            #   USE_BATCH_NORM= False
            # We keep them fixed via the main config, because
            # the preprocessing + scaling is designed for that.
            #
            # Still allow some variation of VSN width:
            "vsn_units": [24, 32, 40],

            # --- Physics switches ---
            #
            # Always keep full physics active by default.
            "pde_mode": ["both"],
            "scale_pde_residuals": {"type": "bool"},

            # Config default is "kb", but we can still let tuner
            # choose between bar/kb if useful.
            "kappa_mode": ["bar", "kb"],

            # Around GEOPRIOR_HD_FACTOR = 0.6
            "hd_factor": {
                "type": "float",
                "min_value": 0.50,
                "max_value": 0.70,
            },

            # --- Learnable scalar initials (model.__init__) ---
            #
            # Around GEOPRIOR_INIT_MV = 1e-7
            "mv": {
                "type": "float",
                "min_value": 5e-8,
                "max_value": 3e-7,
                "sampling": "log",
            },

            # Around GEOPRIOR_INIT_KAPPA = 1.0
            "kappa": {
                "type": "float",
                "min_value": 0.8,
                "max_value": 1.2,
            },

            # --- Compile-only (model.compile) ---
            #
            # Around LEARNING_RATE = 1e-4
            "learning_rate": {
                "type": "float",
                "min_value": 7e-5,
                "max_value": 2e-4,
                "sampling": "log",
            },

            # Around:
            #   LAMBDA_GW     = 0.01
            #   LAMBDA_CONS   = 0.10
            #   LAMBDA_PRIOR  = 0.10
            #   LAMBDA_SMOOTH = 0.01
            #   LAMBDA_MV     = 0.01
            #
            "lambda_gw": {
                "type": "float",
                "min_value": 0.005,
                "max_value": 0.03,
            },
            "lambda_cons": {
                "type": "float",
                "min_value": 0.05,
                "max_value": 0.20,
            },
            "lambda_prior": {
                "type": "float",
                "min_value": 0.05,
                "max_value": 0.20,
            },
            "lambda_smooth": {
                "type": "float",
                "min_value": 0.005,
                "max_value": 0.05,
            },
            "lambda_mv": {
                "type": "float",
                "min_value": 0.005,
                "max_value": 0.05,
            },

            # Around:
            #   MV_LR_MULT    = 1.0
            #   KAPPA_LR_MULT = 5.0
            #
            "mv_lr_mult": {
                "type": "float",
                "min_value": 0.5,
                "max_value": 2.0,
            },
            "kappa_lr_mult": {
                "type": "float",
                "min_value": 2.0,
                "max_value": 8.0,
            },
        }

    config_sections = [
        (
            "Run",
            {
                "CITY_NAME": CITY_NAME,
                "MODEL_NAME": MODEL_NAME,
                "RESULTS_DIR": RESULTS_DIR,
                "MANIFEST_PATH": MANIFEST_PATH,
            },
        ),
        (
            "Stage-1 sequences (snapshot)",
            {
                "TIME_STEPS": TIME_STEPS,
                "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
                "MODE": MODE,
            },
        ),
        (
            "Training loop (config.json)",
            {
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
            },
        ),
        (
            "Physics / architecture (non-HP)",
            {
                "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
                "ATTENTION_LEVELS": ATTENTION_LEVELS,
                "SCALE_PDE_RESIDUALS": SCALE_PDE_RESIDUALS,
                "USE_EFFECTIVE_H": USE_EFFECTIVE_H,
                "QUANTILES": QUANTILES,
            },
        ),
        (
            "Fixed dimensions (from NPZ)",
            {
                "STATIC_DIM": STATIC_DIM,
                "DYNAMIC_DIM": DYNAMIC_DIM,
                "FUTURE_DIM": FUTURE_DIM,
                "OUT_S_DIM": OUT_S_DIM,
                "OUT_G_DIM": OUT_G_DIM,
            },
        ),
        (
            "Tuner search space",
            {
                "n_hyperparameters": len(search_space),
                "keys": sorted(list(search_space.keys())),
            },
        ),
        (
            "Outputs",
            {
                "BASE_STAGE2_DIR": BASE_STAGE2_DIR,
                "RUN_OUTPUT_PATH": RUN_OUTPUT_PATH,
            },
        ),
    ]

    print_config_table(
        config_sections,
        table_width=get_table_size(),
        title=f"{CITY_NAME.upper()} {MODEL_NAME} TUNER CONFIG",
        print_fn =log,
    )

    if should_stop():
        log("[Stage-2] stop_check=True before creating tuner; aborting.")
        return {
            "run_dir": RUN_OUTPUT_PATH,
            "manifest_path": MANIFEST_PATH,
            "city": CITY_NAME,
            "model": MODEL_NAME,
            "best_model_path": None,
            "best_weights_path": None,
            "best_hps_path": None,
            "tuning_summary_json": None,
            "best_hps": {},
        }

    # ----------------------------------------------------------------------
    # 2.3 Package inputs/targets for tuner
    # ----------------------------------------------------------------------
    # Note: X_* were exported to always include the keys below
    # (with 0-width arrays if absent).
    train_inputs_np = {
        "coords": X_train["coords"],
        "dynamic_features": X_train["dynamic_features"],
        "static_features": X_train.get("static_features"),
        "future_features": X_train.get("future_features"),
        "H_field": X_train["H_field"],
    }
    train_targets_np = y_train_mapped

    val_inputs_np = {
        "coords": X_val["coords"],
        "dynamic_features": X_val["dynamic_features"],
        "static_features": X_val.get("static_features"),
        "future_features": X_val.get("future_features"),
        "H_field": X_val["H_field"],
    }
    val_targets_np = y_val_mapped
    validation_np = (val_inputs_np, val_targets_np)

    # ----------------------------------------------------------------------
    # 3) Create tuner
    # ----------------------------------------------------------------------
    tuner_logs_dir = os.path.join(RUN_OUTPUT_PATH, "kt_logs")
    ensure_directory_exists(tuner_logs_dir)

    tuner = GeoPriorTuner.create(
        inputs_data=train_inputs_np,
        targets_data=train_targets_np,
        search_space=search_space,
        fixed_params=fixed_params,
        project_name=f"{CITY_NAME}_GeoPrior_HP",
        directory=tuner_logs_dir,
        tuner_type="randomsearch",   # 'randomsearch' | 'bayesian' | 'hyperband'
        max_trials=20,
        overwrite=True,
    )
    try:
        tuner.tuner_.oracle._max_consecutive_failed_trials = 20  # private attr; pragmatic
    except Exception:
        pass

    # ----------------------------------------------------------------------
    # 4) Callbacks & Search
    # ----------------------------------------------------------------------
    early_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    nan_guard = NaNGuard(
        limit_to={
            "loss",
            "val_loss",
            "total_loss",
            "data_loss",
            "physics_loss",
            "consolidation_loss",
            "gw_flow_loss",
            "prior_loss",
            "smooth_loss",
        },
        raise_on_nan=False,
        verbose=1,
    )
    ton_cb = TerminateOnNaN()

    if should_stop():
        log("[Stage-2] stop_check=True before tuner.run(); aborting.")
        return {
            "run_dir": RUN_OUTPUT_PATH,
            "manifest_path": MANIFEST_PATH,
            "city": CITY_NAME,
            "model": MODEL_NAME,
            "best_model_path": None,
            "best_weights_path": None,
            "best_hps_path": None,
            "tuning_summary_json": None,
            "best_hps": {},
        }

    try:
        best_model, best_hps, kt = tuner.run(
            inputs=train_inputs_np,
            y=train_targets_np,
            validation_data=validation_np,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_cb, ton_cb, nan_guard],
            verbose=1,
        )
    except Exception as e:
        log(f"[WARN] Tuning crashed: {e}")
        # Try to recover best-so-far artifacts from the tuner
        best_model = getattr(tuner, "best_model_", None)
        best_hps = getattr(tuner, "best_hps_", None)
        # kt = getattr(tuner, "tuner_", None)

    # ----------------------------------------------------------------------
    # 5) Persist results
    # ----------------------------------------------------------------------
    best_model_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_GeoPrior_best.keras"
    )
    best_weights_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_GeoPrior_best.weights.h5"
    )
    best_hps_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_GeoPrior_best_hps.json"
    )

    if best_model is not None:
        # 1) Full SavedModel / .keras archive
        best_model.save(best_model_path)
        log(f"\nSaved best tuned model to: {best_model_path}")

        # 2) Weights-only checkpoint (robust fallback for inference)
        try:
            best_model.save_weights(best_weights_path)
            log(f"Saved best model weights to: {best_weights_path}")
        except Exception as e:
            log(f"[Warn] Could not save best model weights: {e}")
            best_weights_path = None
    else:
        log("\n[Warn] No best model returned by tuner.")
        best_weights_path = None

    best_hps_dict: Dict[str, Any] = {}
    if best_hps is not None:
        # Robustly serialize HyperParameters
        try:
            # KerasTuner HyperParameters exposes .values; otherwise fetch by name list
            names = list(getattr(best_hps, "values", {}).keys())
            if not names and hasattr(best_hps, "space"):
                names = [hp.name for hp in best_hps.space]
            best_hps_dict = {name: best_hps.get(name) for name in names}
        except Exception:
            # Fallback: try direct cast
            try:
                best_hps_dict = dict(best_hps.values)
            except Exception:
                best_hps_dict = {}

    with open(best_hps_path, "w", encoding="utf-8") as f:
        json.dump(best_hps_dict, f, indent=2)
    log(f"Saved best hyperparameters to: {best_hps_path}")

    # Save a tiny tuning summary
    tuning_summary_path = os.path.join(RUN_OUTPUT_PATH, "tuning_summary.json")
    summary_payload = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": CITY_NAME,
        "model": MODEL_NAME,
        "mode": MODE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "time_steps": TIME_STEPS,
        "horizon": FORECAST_HORIZON_YEARS,
        "dims": {
            "static_input_dim": STATIC_DIM,
            "dynamic_input_dim": DYNAMIC_DIM,
            "future_input_dim": FUTURE_DIM,
            "output_subsidence_dim": OUT_S_DIM,
            "output_gwl_dim": OUT_G_DIM,
        },
        "best_model_path": best_model_path if best_model is not None else None,
        "best_weights_path": (
            best_weights_path if best_model is not None else None
        ),
        "best_hps": best_hps_dict,
        "evaluate_tuned_flag": bool(evaluate_tuned),
    }
    with open(tuning_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    log(
        "\nTuning complete. You can now proceed to calibration/forecasting "
        "or connect this run to the Step-6 evaluation block.\n"
    )

    # ----------------------------------------------------------------------
    # 6) Evaluate tuned best model & save diagnostics / ablation record
    # ----------------------------------------------------------------------
    metrics_json_path: Optional[str] = None
    eval_csv: Optional[str] = None
    future_csv: Optional[str] = None

    if evaluate_tuned:
        log("\n[Eval] evaluate_tuned=True -> running"
            " diagnostics for tuned GeoPriorSubsNet...")

        if should_stop():
            log("[Eval] stop_check=True before tuned"
                " evaluation; skipping Step 6.")
        else:
            # 6.1 Decide which model we will actually evaluate
            model_for_eval = best_model
            best_hps_for_eval = dict(best_hps_dict) if best_hps_dict else {}

            # If tuner did not return a model object, rebuild from disk.
            if model_for_eval is None and best_model_path:
                log(f"[Eval] No in-memory model; trying to rebuild from {best_model_path}")
                try:
                    model_for_eval, best_hps_disk = load_or_rebuild_geoprior_model(
                        model_path=best_model_path,
                        manifest=M,
                        X_sample=X_val,   # already sanitized & shape-ensured
                        out_s_dim=OUT_S_DIM,
                        out_g_dim=OUT_G_DIM,
                        mode=MODE,
                        horizon=FORECAST_HORIZON_YEARS,
                        quantiles=QUANTILES if isinstance(QUANTILES, list) else None,
                        city_name=CITY_NAME,
                        compile_on_load=True,
                        verbose=1,
                    )
                    if best_hps_disk and not best_hps_for_eval:
                        best_hps_for_eval = dict(best_hps_disk)
                except Exception as e:
                    log(f"[Eval] Rebuild tuned model failed: {e}")
                    model_for_eval = None

            if model_for_eval is None:
                log("[Eval] No best tuned model available; skipping Step 6.")
            else:
                # 6.2 Load optional TEST NPZ; fall back to VAL if absent
                test_inputs_npz = M["artifacts"]["numpy"].get("test_inputs_npz")
                test_targets_npz = M["artifacts"]["numpy"].get("test_targets_npz")

                X_test = None
                y_test = None
                if test_inputs_npz and test_targets_npz:
                    try:
                        X_test = _sanitize_inputs(dict(np.load(test_inputs_npz)))
                        y_test = dict(np.load(test_targets_npz))
                        log("[Eval] Using TEST NPZ for tuned diagnostics.")
                    except Exception as e:
                        log(f"[Warn] Could not load test NPZ; falling back to val: {e}")
                        X_test, y_test = None, None

                if X_test is not None and y_test is not None:
                    X_fore = X_test
                    y_fore = y_test
                    dataset_name_for_forecast = "TestSet"
                else:
                    X_fore = X_val
                    y_fore = y_val
                    dataset_name_for_forecast = "ValidationSet_Fallback"

                # 6.3 Build validation tf.data.Dataset for calibrator & evaluate()
                val_dataset_tf = make_tf_dataset(
                    X_val,
                    y_val,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    mode=MODE,
                    forecast_horizon=FORECAST_HORIZON_YEARS,
                )

                # 6.4 Load scaler_info mapping (Stage-1 summary) in robust way
                enc_eval = M["artifacts"]["encoders"]
                scaler_info_dict = load_scaler_info(enc_eval)
                if isinstance(scaler_info_dict, dict):
                    for k, v in scaler_info_dict.items():
                        if (
                            isinstance(v, dict)
                            and "scaler_path" in v
                            and "scaler" not in v
                        ):
                            p = v["scaler_path"]
                            if p and os.path.exists(p):
                                try:
                                    v["scaler"] = joblib.load(p)
                                except Exception:
                                    pass

                # coord_scaler: reuse if already loaded, else try again
                coord_scaler_eval = coord_scaler
                cs_path_eval = enc_eval.get("coord_scaler")
                if coord_scaler_eval is None and cs_path_eval and os.path.exists(cs_path_eval):
                    try:
                        coord_scaler_eval = joblib.load(cs_path_eval)
                    except Exception as e:
                        log(f"[Warn] Could not load coord_scaler at {cs_path_eval}: {e}")

                # 6.5 Column names & forecast start year
                cols_cfg = (
                    cfg_stage1.get("cols")
                    or cfg_hp.get("cols")
                    or {}
                )
                SUBSIDENCE_COL = cols_cfg.get("subsidence", "subsidence")
                # GWL_COL = cols_cfg.get("gwl", "GWL")
                FORECAST_START_YEAR = cfg_stage1.get(
                    "FORECAST_START_YEAR",
                    cfg_hp.get("FORECAST_START_YEAR"),
                )

                if should_stop():
                    log("[Eval] stop_check=True before calibrator; skipping Step 6.")
                else:
                    # 6.6 Interval calibrator on tuned model (target 80%)
                    log("[Eval] Fitting interval calibrator (80%) on validation set...")
                    cal80 = fit_interval_calibrator_on_val(
                        model_for_eval,
                        val_dataset_tf,
                        target=0.80,
                    )
                    cal_factors_path = os.path.join(
                        RUN_OUTPUT_PATH, "interval_factors_80_tuned.npy"
                    )
                    np.save(cal_factors_path, getattr(cal80, "factors_", None))
                    log("[Eval] Calibrator (tuned) saved.")

                    # 6.7 Forecast on chosen split (TEST or VAL fallback)
                    X_fore_norm = ensure_input_shapes(
                        X_fore,
                        mode=MODE,
                        forecast_horizon=FORECAST_HORIZON_YEARS,
                    )
                    y_fore_fmt = map_targets_for_training(y_fore)

                    if should_stop():
                        log("[Eval] stop_check=True before tuned predict; skipping Step 6.")
                    else:
                        log(
                            f"[Eval] Predicting on {dataset_name_for_forecast} "
                            "with tuned model..."
                        )
                        pred_dict = model_for_eval.predict(X_fore_norm, verbose=0)
                        data_final = pred_dict["data_final"]

                        # Split subsidence / GWL heads and apply calibration to subsidence quantiles
                        if QUANTILES:
                            # (B, H, Q, O_total)
                            s_pred_q_raw = data_final[..., :OUT_S_DIM]
                            h_pred_q_raw = data_final[..., OUT_S_DIM:]
                            s_pred_q_cal = apply_calibrator_to_subs(cal80, s_pred_q_raw)
                            predictions_for_formatter = {
                                "subs_pred": s_pred_q_cal,
                                "gwl_pred": h_pred_q_raw,
                            }
                        else:
                            # Point forecasts
                            s_pred_raw = data_final[..., :OUT_S_DIM]
                            h_pred_raw = data_final[..., OUT_S_DIM:]
                            predictions_for_formatter = {
                                "subs_pred": s_pred_raw,
                                "gwl_pred": h_pred_raw,
                            }

                        # y_true mapping for formatter (back to canonical names)
                        y_true_for_format = {
                            "subsidence": y_fore_fmt["subs_pred"],
                            "gwl": y_fore_fmt["gwl_pred"],
                        }

                        eval_csv = os.path.join(
                            RUN_OUTPUT_PATH,
                            f"{CITY_NAME}_{MODEL_NAME}_tuned_forecast_"
                            f"{dataset_name_for_forecast}_H{FORECAST_HORIZON_YEARS}"
                            "_calibrated.csv",
                        )
                        future_csv = os.path.join(
                            RUN_OUTPUT_PATH,
                            f"{CITY_NAME}_{MODEL_NAME}_tuned_forecast_"
                            f"{dataset_name_for_forecast}_H{FORECAST_HORIZON_YEARS}"
                            "_future.csv",
                        )

                        future_grid = np.arange(
                            FORECAST_START_YEAR,
                            FORECAST_START_YEAR + FORECAST_HORIZON_YEARS,
                            dtype=float,
                        )

                        df_eval, df_future = format_and_forecast(
                            y_pred=predictions_for_formatter,
                            y_true=y_true_for_format,
                            coords=X_fore_norm.get("coords", None),
                            quantiles=QUANTILES if QUANTILES else None,
                            target_name=SUBSIDENCE_COL,
                            target_key_pred="subs_pred",
                            component_index=0,
                            scaler_info=scaler_info_dict,
                            coord_scaler=coord_scaler_eval,
                            coord_columns=("coord_t", "coord_x", "coord_y"),
                            train_end_time=cfg_stage1.get("TRAIN_END_YEAR"),
                            forecast_start_time=FORECAST_START_YEAR,
                            forecast_horizon=FORECAST_HORIZON_YEARS,
                            future_time_grid=future_grid,
                            eval_forecast_step=None,
                            sample_index_offset=0,
                            city_name=CITY_NAME,
                            model_name=MODEL_NAME,
                            dataset_name=dataset_name_for_forecast,
                            csv_eval_path=eval_csv,
                            csv_future_path=future_csv,
                            time_as_datetime=False,
                            time_format=None,
                            verbose=1,
                            # Extra diagnostics
                            eval_metrics=True,
                            metrics_column_map=None,
                            metrics_quantile_interval=(0.1, 0.9),
                            metrics_per_horizon=True,
                            metrics_extra=["pss"],
                            metrics_extra_kwargs=None,
                            metrics_savefile=os.path.join(
                                RUN_OUTPUT_PATH, "eval_diagnostics_tuned.json"
                            ),
                            metrics_save_format=".json",
                            metrics_time_as_str=True,
                            value_mode="rate",
                            logger = log, 
                        )

                        if df_eval is not None and not df_eval.empty:
                            log(f"[Eval] Saved tuned calibrated EVAL forecast -> {eval_csv}")
                        else:
                            log("[Warn] Tuned eval forecast DF is empty.")

                        if df_future is not None and not df_future.empty:
                            log(
                                "[Eval] Saved tuned calibrated FUTURE forecast "
                                f"-> {future_csv}"
                            )
                        else:
                            log("[Warn] Tuned future forecast DF is empty.")

                        # 6.8 Evaluate tuned model with Keras (includes physics losses)
                        model_for_eval = compile_for_eval(
                            model=model_for_eval,
                            manifest=M,
                            best_hps=best_hps_dict,  # from tuning section
                            quantiles=QUANTILES if isinstance(QUANTILES, list) else None,
                            include_metrics=True,
                        )

                        eval_results: Dict[str, Any] = {}
                        phys: Dict[str, float] = {}
                        ds_eval_tf = tf.data.Dataset.from_tensor_slices(
                            (X_fore_norm, y_fore_fmt)
                        ).batch(BATCH_SIZE)

                        try:
                            eval_results = model_for_eval.evaluate(
                                ds_eval_tf,
                                return_dict=True,
                                verbose=1,
                            )
                            log(f"[Eval] Tuned model evaluate(): {eval_results}")
                            for k in ("epsilon_prior", "epsilon_cons"):
                                if k in eval_results:
                                    phys[k] = float(_to_py(eval_results[k]))
                            if phys:
                                log(f"[Eval] Physics diagnostics (tuned): {phys}")
                        except Exception as e:
                            log(f"[Warn] Tuned evaluate() failed: {e}")
                            eval_results, phys = {}, {}

                        # 6.9 Save physics payload
                        try:
                            _ = model_for_eval.export_physics_payload(
                                ds_eval_tf,
                                max_batches=None,
                                save_path=os.path.join(
                                    RUN_OUTPUT_PATH,
                                    f"{CITY_NAME}_tuned_phys_payload_{dataset_name_for_forecast.lower()}.npz",
                                ),
                                format="npz",
                                overwrite=True,
                                metadata={
                                    "city": CITY_NAME,
                                    "split": dataset_name_for_forecast,
                                },
                            )
                            log("[Eval] Tuned physics payload saved.")
                        except Exception as e:
                            log(f"[Warn] Physics payload saving failed: {e}")

                        # 6.10 Interval metrics (coverage / sharpness) in PHYSICAL space
                        coverage80_for_abl = None
                        sharpness80_for_abl = None

                        y_true_int = None
                        s_q_int = None

                        if QUANTILES:
                            y_true_list_int = []
                            s_q_list_int = []
                            for xb, yb in with_progress(
                                ds_eval_tf, desc="Tuned interval diagnostics"
                            ):
                                out_b = model_for_eval(xb, training=False)
                                data_final_b = out_b["data_final"]
                                y_true_list_int.append(yb["subs_pred"])           # (B,H,1)
                                s_q_list_int.append(data_final_b[..., :OUT_S_DIM])  # (B,H,Q,1)

                            if y_true_list_int and s_q_list_int:
                                y_true_int = tf.concat(y_true_list_int, axis=0)
                                s_q_int = tf.concat(s_q_list_int, axis=0)

                                y_true_phys_np = inverse_scale_target(
                                    y_true_int,
                                    scaler_info=scaler_info_dict,
                                    target_name=SUBSIDENCE_COL,
                                )
                                s_q_phys_np = inverse_scale_target(
                                    s_q_int,
                                    scaler_info=scaler_info_dict,
                                    target_name=SUBSIDENCE_COL,
                                )

                                y_true_phys_tf = tf.convert_to_tensor(
                                    y_true_phys_np, dtype=tf.float32
                                )
                                s_q_phys_tf = tf.convert_to_tensor(
                                    s_q_phys_np, dtype=tf.float32
                                )

                                coverage80_for_abl = float(
                                    coverage80_fn(y_true_phys_tf, s_q_phys_tf).numpy()
                                )
                                sharpness80_for_abl = float(
                                    sharpness80_fn(y_true_phys_tf, s_q_phys_tf).numpy()
                                )

                        # 6.11 Point metrics + per-horizon metrics (physical units)
                        metrics_point: Dict[str, Any] = {}
                        per_h_mae_dict: Optional[Dict[str, float]] = None
                        per_h_r2_dict: Optional[Dict[str, float]] = None

                        s_med = None
                        if QUANTILES and (s_q_int is not None) and (y_true_int is not None):
                            qs = np.asarray(QUANTILES)
                            med_idx = int(np.argmin(np.abs(qs - 0.5)))
                            s_med = s_q_int[..., med_idx, :]    # (N,H,1)
                            y_true_for_point = y_true_int
                        else:
                            y_true_list2 = []
                            s_pred_list2 = []
                            for xb, yb in with_progress(
                                ds_eval_tf, desc="Tuned point diagnostics"
                            ):
                                out_b = model_for_eval(xb, training=False)
                                data_final_b = out_b["data_final"]
                                y_true_list2.append(yb["subs_pred"])
                                s_pred_list2.append(data_final_b[..., :OUT_S_DIM])

                            if s_pred_list2:
                                s_med = tf.concat(s_pred_list2, axis=0)
                                y_true_for_point = tf.concat(y_true_list2, axis=0)
                            else:
                                y_true_for_point = None

                        if s_med is not None and (y_true_for_point is not None):
                            metrics_point = point_metrics(
                                y_true_for_point,
                                s_med,
                                use_physical=True,
                                scaler_info=scaler_info_dict,
                                target_name=SUBSIDENCE_COL,
                            )
                            per_h_mae_dict, per_h_r2_dict = per_horizon_metrics(
                                y_true_for_point,
                                s_med,
                                use_physical=True,
                                scaler_info=scaler_info_dict,
                                target_name=SUBSIDENCE_COL,
                            )

                        # 6.12 Save tuned metrics JSON
                        stamp_eval = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                        metrics_json = os.path.join(
                            RUN_OUTPUT_PATH,
                            f"geoprior_eval_phys_tuned_{stamp_eval}.json",
                        )

                        payload_eval = {
                            "timestamp": stamp_eval,
                            "tf_version": tf.__version__,
                            "numpy_version": np.__version__,
                            "city": CITY_NAME,
                            "model": MODEL_NAME,
                            "horizon": FORECAST_HORIZON_YEARS,
                            "quantiles": QUANTILES,
                            "dataset_name": dataset_name_for_forecast,
                            "metrics_evaluate": {
                                k: _to_py(v)
                                for k, v in (eval_results or {}).items()
                            },
                            "physics_diagnostics": phys,
                            "point_metrics": metrics_point,
                            "per_horizon": {
                                "mae": per_h_mae_dict,
                                "r2": per_h_r2_dict,
                            },
                            "coverage80": coverage80_for_abl,
                            "sharpness80": sharpness80_for_abl,
                        }

                        with open(metrics_json, "w", encoding="utf-8") as f:
                            json.dump(payload_eval, f, indent=2)
                        log(
                            f"\n[Eval] Saved tuned metrics + physics JSON "
                            f"-> {metrics_json}"
                        )

                        metrics_json_path = metrics_json

                        # 6.13 Ablation record row for tuned model
                        ABLCFG_TUNED = {
                            "PDE_MODE_CONFIG": best_hps_for_eval.get(
                                "pde_mode", PDE_MODE_CONFIG
                            ),
                            "GEOPRIOR_USE_EFFECTIVE_H": USE_EFFECTIVE_H,
                            "GEOPRIOR_KAPPA_MODE": best_hps_for_eval.get("kappa_mode"),
                            "GEOPRIOR_HD_FACTOR": best_hps_for_eval.get("hd_factor"),
                            "LAMBDA_CONS": best_hps_for_eval.get("lambda_cons"),
                            "LAMBDA_GW": best_hps_for_eval.get("lambda_gw"),
                            "LAMBDA_PRIOR": best_hps_for_eval.get("lambda_prior"),
                            "LAMBDA_SMOOTH": best_hps_for_eval.get("lambda_smooth"),
                            "LAMBDA_MV": best_hps_for_eval.get("lambda_mv"),
                        }

                        try:
                            save_ablation_record(
                                outdir=RUN_OUTPUT_PATH,
                                city=CITY_NAME,
                                model_name=f"{MODEL_NAME}_tuned",
                                cfg=ABLCFG_TUNED,
                                eval_dict={
                                    "r2": (metrics_point or {}).get("r2"),
                                    "mse": (metrics_point or {}).get("mse"),
                                    "mae": (metrics_point or {}).get("mae"),
                                    "coverage80": coverage80_for_abl,
                                    "sharpness80": sharpness80_for_abl,
                                },
                                phys_diag=phys,
                                per_h_mae=per_h_mae_dict,
                                per_h_r2=per_h_r2_dict,
                            )
                            log("[Eval] Tuned ablation record appended.")
                        except Exception as e:
                            log(f"[Warn] Failed to append tuned ablation record: {e}")

    # ----------------------------------------------------------------------
    # Final return dict (extended with optional evaluation artifacts)
    # ----------------------------------------------------------------------
    log(
        f"---- {CITY_NAME.upper()} {MODEL_NAME} TUNING COMPLETE ----\n"
        f"Artifacts -> {RUN_OUTPUT_PATH}\n"
    )

    return {
        "run_dir": RUN_OUTPUT_PATH,
        "manifest_path": MANIFEST_PATH,
        "city": CITY_NAME,
        "model": MODEL_NAME,
        "best_model_path": best_model_path if best_model is not None else None,
        "best_weights_path": best_weights_path,
        "best_hps_path": best_hps_path,
        "tuning_summary_json": tuning_summary_path,
        "best_hps": best_hps_dict,
        "eval_csv": eval_csv,
        "future_csv": future_csv,
        "metrics_json": metrics_json_path,
    }



if __name__ == "__main__":
    # CLI usage (simple): run with default discovery and no overrides
    run_tuning()
