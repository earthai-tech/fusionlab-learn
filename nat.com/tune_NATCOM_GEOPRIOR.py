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

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

try:
    from fusionlab.api.util import get_table_size
    from fusionlab.utils.generic_utils import (
        ensure_directory_exists,
        default_results_dir,
        getenv_stripped,
        print_config_table,   
    )
    from fusionlab._optdeps import with_progress

    from fusionlab.registry.utils import _find_stage1_manifest

    # Import the tuner
    from fusionlab.nn.forecast_tuner import GeoPriorTuner
    from fusionlab.nn.callbacks import NaNGuard 

    from fusionlab.utils.nat_utils import (
        load_nat_config,
        ensure_input_shapes,
        map_targets_for_training,
        make_tf_dataset,
        load_scaler_info,
        save_ablation_record,
        load_or_rebuild_geoprior_model, 
        compile_for_eval,
    )


    from fusionlab.utils.forecast_utils import format_and_forecast
    from fusionlab.utils.scale_metrics import (
        inverse_scale_target,
        point_metrics,
        per_horizon_metrics,
    )
    from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn, _to_py
    from fusionlab.nn.calibration import (
        fit_interval_calibrator_on_val,
        apply_calibrator_to_subs,
    )

    print("Successfully imported fusionlab modules for tuning.")
except Exception as e:
    print(f"Failed to import fusionlab modules: {e}")
    raise

# =============================================================================
# 0) Load Stage-1 manifest and arrays
# =============================================================================
RESULTS_DIR = default_results_dir()  # auto-resolve
CITY_HINT   = getenv_stripped("CITY")  # -> None if unset/empty
MODEL_HINT  = getenv_stripped("MODEL_NAME_OVERRIDE", default="GeoPriorSubsNet")
MANUAL      = getenv_stripped("STAGE1_MANIFEST")  # exact path if provided

MANIFEST_PATH = _find_stage1_manifest(
    manual=MANUAL,                   # exact manifest if provided
    base_dir=RESULTS_DIR,            # where to search
    city_hint=CITY_HINT,             # filter by city if set
    model_hint=MODEL_HINT,           # filter by model if set
    prefer="timestamp",              # or "mtime"
    required_keys=("model", "stage"),
    verbose=1,
)

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    M = json.load(f)
print(f"[Manifest] Loaded city={M.get('city')} model={M.get('model')}")

cfg_stage1 = M["config"]        # what Stage-1 used to build sequences
cfg_hp = load_nat_config()      # global config from config.json (4.*)

# Merge global config with Stage-1 snapshot (Stage-1 wins on conflicts)
cfg = dict(cfg_hp)
cfg.update(cfg_stage1)

# Resolve NPZ paths from manifest
train_inputs_npz = M["artifacts"]["numpy"]["train_inputs_npz"]
train_targets_npz = M["artifacts"]["numpy"]["train_targets_npz"]
val_inputs_npz   = M["artifacts"]["numpy"]["val_inputs_npz"]
val_targets_npz  = M["artifacts"]["numpy"]["val_targets_npz"]

X_train = dict(np.load(train_inputs_npz))
y_train = dict(np.load(train_targets_npz))
X_val   = dict(np.load(val_inputs_npz))
y_val   = dict(np.load(val_targets_npz))

def _sanitize_inputs(X):
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

def _ensure_input_shapes_np(X, mode: str, horizon: int):
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
X_val   = _sanitize_inputs(X_val)

CITY_NAME  = M.get("city", cfg_hp.get("CITY_NAME", "unknown_city"))
MODEL_NAME = M.get("model", cfg_hp.get("MODEL_NAME", "GeoPriorSubsNet"))

# Time / horizon / mode MUST match Stage-1 sequences
TIME_STEPS             = cfg_stage1["TIME_STEPS"]
FORECAST_HORIZON_YEARS = cfg_stage1["FORECAST_HORIZON_YEARS"]
MODE                   = cfg_stage1["MODE"]

# Censoring: merge Stage-1 (specs + report) and global config
censor_stage1 = cfg_stage1.get("censoring", {}) or {}
censor_global = cfg_hp.get("censoring", {}) or {}
CENSOR = {**censor_stage1, **censor_global}
USE_EFFECTIVE_H = CENSOR.get("use_effective_h_field", True)


# Ensure placeholders exist even if Stage-1 NPZs are from an older run
X_train = _ensure_input_shapes_np(X_train, MODE, FORECAST_HORIZON_YEARS)
X_val   = _ensure_input_shapes_np(X_val,   MODE, FORECAST_HORIZON_YEARS)

# Optional encoders/scalers (may be absent)
enc = M["artifacts"]["encoders"]
main_scaler = None
ms_path = enc.get("main_scaler")
if ms_path and os.path.exists(ms_path):
    try:
        main_scaler = joblib.load(ms_path)
    except Exception:
        pass

coord_scaler = None
cs_path = enc.get("coord_scaler")
if cs_path and os.path.exists(cs_path):
    try:
        coord_scaler = joblib.load(cs_path)
    except Exception:
        pass

scaler_info = enc.get("scaler_info", {})

# Output directory for Stage-2
BASE_STAGE2_DIR = os.path.join(M["paths"]["run_dir"], "tuning")
ensure_directory_exists(BASE_STAGE2_DIR)

STAMP = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_OUTPUT_PATH = os.path.join(BASE_STAGE2_DIR, f"run_{STAMP}")
ensure_directory_exists(RUN_OUTPUT_PATH)
print(f"\nStage-2 outputs will be written to: {RUN_OUTPUT_PATH}")

# =============================================================================
# 1) Sanity maps & shapes
# =============================================================================
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
y_val_mapped   = map_targets_for_tuner(y_val)

# Fixed dims from arrays
def _dim_of(x, key, fallback_lastdim=0):
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
    FUTURE_DIM = int(ff.shape[-1]) if ff.ndim >= 3 else _dim_of(X_train, "future_features", 0)

OUT_S_DIM = M["artifacts"]["sequences"]["dims"]["output_subsidence_dim"]
OUT_G_DIM = M["artifacts"]["sequences"]["dims"]["output_gwl_dim"]

print("\nInferred input dims from Stage-1 arrays:")
print(f"  static_input_dim  = {STATIC_DIM}")
print(f"  dynamic_input_dim = {DYNAMIC_DIM}")
print(f"  future_input_dim  = {FUTURE_DIM}")
print(f"  OUT_S_DIM         = {OUT_S_DIM}")
print(f"  OUT_G_DIM         = {OUT_G_DIM}")
print(f"  TIME_STEPS        = {TIME_STEPS}")
print(f"  HORIZON           = {FORECAST_HORIZON_YEARS}")
print(f"  MODE              = {MODE}")


# =============================================================================
# 2) Tuning configuration (non-HPs and HP space)
# =============================================================================

# Training loop setup (from config.json)
EPOCHS     = cfg_hp.get("EPOCHS", 50)
BATCH_SIZE = cfg_hp.get("BATCH_SIZE", 32)

# Probabilistic outputs
QUANTILES = cfg_hp.get("QUANTILES", [0.1, 0.5, 0.9])

# PDE selection & attention levels
PDE_MODE_CONFIG   = cfg_hp.get("PDE_MODE_CONFIG", "both")
ATTENTION_LEVELS  = cfg_hp.get("ATTENTION_LEVELS",
                               ["cross", "hierarchical", "memory"])
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
# # 2.2 Hyperparameter search space

search_space_cfg = cfg_hp.get("TUNER_SEARCH_SPACE", None)

if isinstance(search_space_cfg, dict) and search_space_cfg:
    search_space = search_space_cfg
else:
    # Fallback: old hard-coded space (kept here as a backup)
    search_space = {
        # --- Architecture (model.__init__) ---
        "embed_dim": [32, 64],
        "hidden_units": [64, 96, 128],
        "lstm_units": [64, 96],
        "attention_units": [32, 64],
        "num_heads": [2, 4],
        "dropout_rate": {
            "type": "float", "min_value": 0.10, "max_value": 0.30
        },
        "use_vsn": {"type": "bool"},
        "vsn_units": [16, 32, 48],
        "use_batch_norm": {"type": "bool"},
        # Physics switches
        "pde_mode": ["both"],
        "scale_pde_residuals": {"type": "bool"},
        "kappa_mode": ["bar", "kb"],
        "hd_factor": {
            "type": "float", 
            "min_value": 0.50, 
            "max_value": 0.80
        },
        "mv": {
            "type": "float", 
            "min_value": 3e-7,
            "max_value": 1e-6,
            "sampling": "log",
        },
        "kappa": {
            "type": "float",
                "min_value": 0.7,
                "max_value": 1.5
        },
        # --- Compile-only (model.compile) ---
        "learning_rate": {
            "type": "float",
            "min_value": 5e-5, 
            "max_value": 3e-4,
            "sampling": "log",
        },
        "lambda_gw": {
            "type": "float", 
            "min_value": 0.1,
            "max_value": 1.0
        },
        "lambda_cons": {
            "type": "float",
            "min_value": 0.01, 
            "max_value": 1.0
        },
        "lambda_prior": {
            "type": "float", 
            "min_value": 0.1, 
            "max_value": 0.8
        },
        "lambda_smooth": {
            "type": "float",
            "min_value": 0.01, 
            "max_value": 1.0
        },
        "lambda_mv": {
            "type": "float", 
            "min_value": 0.01,
            "max_value": 0.5
        },
        "mv_lr_mult": {
            "type": "float", 
            "min_value": 0.5, 
            "max_value": 2.0
        },
        "kappa_lr_mult": {
            "type": "float", 
            "min_value": 1.0,
            "max_value": 10.0
        },
    }

config_sections = [
    ("Run", {
        "CITY_NAME": CITY_NAME,
        "MODEL_NAME": MODEL_NAME,
        "RESULTS_DIR": RESULTS_DIR,
        "MANIFEST_PATH": MANIFEST_PATH,
    }),
    ("Stage-1 sequences (snapshot)", {
        "TIME_STEPS": TIME_STEPS,
        "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
        "MODE": MODE,
    }),
    ("Training loop (config.json)", {
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
    }),
    ("Physics / architecture (non-HP)", {
        "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
        "ATTENTION_LEVELS": ATTENTION_LEVELS,
        "SCALE_PDE_RESIDUALS": SCALE_PDE_RESIDUALS,
        "USE_EFFECTIVE_H": USE_EFFECTIVE_H,
        "QUANTILES": QUANTILES,
    }),
    ("Fixed dimensions (from NPZ)", {
        "STATIC_DIM": STATIC_DIM,
        "DYNAMIC_DIM": DYNAMIC_DIM,
        "FUTURE_DIM": FUTURE_DIM,
        "OUT_S_DIM": OUT_S_DIM,
        "OUT_G_DIM": OUT_G_DIM,
    }),
    ("Tuner search space", {
        "n_hyperparameters": len(search_space),
        "keys": sorted(list(search_space.keys())),
    }),
    ("Outputs", {
        "BASE_STAGE2_DIR": BASE_STAGE2_DIR,
        "RUN_OUTPUT_PATH": RUN_OUTPUT_PATH,
    }),
]

print_config_table(
    config_sections, table_width = get_table_size (), 
    title=f"{CITY_NAME.upper()} {MODEL_NAME} TUNER CONFIG",
)

# 2.3 Package inputs/targets for tuner
# Note: X_* were exported to always include the keys below (with 0-width arrays if absent).
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

# =============================================================================
# 3) Create tuner
# =============================================================================
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
# =============================================================================
# 4) Callbacks & Search
# =============================================================================

early_cb = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

nan_guard = NaNGuard(
    limit_to={"loss","val_loss","total_loss","data_loss","physics_loss",
              "consolidation_loss","gw_flow_loss","prior_loss","smooth_loss"},
    raise_on_nan=False,
    verbose=1,
)
ton_cb = TerminateOnNaN()

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
    print(f"[WARN] Tuning crashed: {e}")
    # Try to recover best-so-far artifacts from the tuner
    best_model = getattr(tuner, "best_model_", None)
    best_hps   = getattr(tuner, "best_hps_",   None)
    kt         = getattr(tuner, "tuner_",      None)


# =============================================================================
# 5) Persist results
# =============================================================================
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
    print(f"\nSaved best tuned model to: {best_model_path}")

    # 2) Weights-only checkpoint (robust fallback for inference)
    try:
        best_model.save_weights(best_weights_path)
        print(f"Saved best model weights to: {best_weights_path}")
    except Exception as e:
        print(f"[Warn] Could not save best model weights: {e}")
else:
    print("\n[Warn] No best model returned by tuner.")
    best_weights_path = None

best_hps_dict = {}
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
print(f"Saved best hyperparameters to: {best_hps_path}")

# Save a tiny tuning summary
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
    "best_weights_path": best_weights_path if best_model is not None else None,
    "best_hps": best_hps_dict,
}
with open(os.path.join(RUN_OUTPUT_PATH, "tuning_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2)

print("\nTuning complete. You can now proceed to calibration/forecasting using "
      "the saved best model or continue experimentation with the KerasTuner logs.")

#
# =============================================================================
# 6) Evaluate tuned best model & save diagnostics / ablation record
# =============================================================================

# Decide which model we will actually evaluate
model_for_eval = best_model
best_hps_for_eval = dict(best_hps_dict) if best_hps_dict else {}

# If tuner did not return a model object (or script is re-used in a
# "diagnostics-only" mode later), try to load/rebuild from disk.
if model_for_eval is None:
    tuned_model_path = best_model_path  # path we just wrote in section 5

    if tuned_model_path:
        model_for_eval, best_hps_disk = load_or_rebuild_geoprior_model(
            model_path=tuned_model_path,
            manifest=M,
            X_sample=X_val,   # already sanitized & shape-ensured above
            out_s_dim=OUT_S_DIM,
            out_g_dim=OUT_G_DIM,
            mode=MODE,
            horizon=FORECAST_HORIZON_YEARS,
            quantiles=QUANTILES if isinstance(QUANTILES, list) else None,
            city_name=CITY_NAME,
            compile_on_load=True,
            verbose=1,
        )
        # If we didn't have HPs in-memory, reuse what the loader recovered
        if best_hps_disk and not best_hps_for_eval:
            best_hps_for_eval = dict(best_hps_disk)

if model_for_eval is None:
    print("\n[Warn] No best model; skipping tuned-model diagnostics.")
else:

    print("\n[Eval] Running diagnostics for tuned GeoPriorSubsNet...")

    # 6.1 Load optional TEST NPZ; fall back to VAL if absent
    test_inputs_npz  = M["artifacts"]["numpy"].get("test_inputs_npz")
    test_targets_npz = M["artifacts"]["numpy"].get("test_targets_npz")

    X_test = None
    y_test = None
    if test_inputs_npz and test_targets_npz:
        try:
            X_test = _sanitize_inputs(dict(np.load(test_inputs_npz)))
            y_test = dict(np.load(test_targets_npz))
        except Exception as e:
            print(f"[Warn] Could not load test NPZ; falling back to val: {e}")

    if X_test is not None and y_test is not None:
        X_fore = X_test
        y_fore = y_test
        dataset_name_for_forecast = "TestSet"
    else:
        X_fore = X_val
        y_fore = y_val
        dataset_name_for_forecast = "ValidationSet_Fallback"

    # 6.2 Build validation tf.data.Dataset for calibrator & evaluate()
    val_dataset_tf = make_tf_dataset(
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        mode=MODE,
        forecast_horizon=FORECAST_HORIZON_YEARS,
    )

    # 6.3 Load scaler_info mapping (Stage-1 summary) in robust way
    enc = M["artifacts"]["encoders"]
    scaler_info_dict = load_scaler_info(enc)
    if isinstance(scaler_info_dict, dict):
        for k, v in scaler_info_dict.items():
            if isinstance(v, dict) and "scaler_path" in v and "scaler" not in v:
                p = v["scaler_path"]
                if p and os.path.exists(p):
                    try:
                        v["scaler"] = joblib.load(p)
                    except Exception:
                        pass

    # coord_scaler (optional)
    coord_scaler = None
    cs_path = enc.get("coord_scaler")
    if cs_path and os.path.exists(cs_path):
        try:
            coord_scaler = joblib.load(cs_path)
        except Exception as e:
            print(f"[Warn] Could not load coord_scaler at {cs_path}: {e}")

    # 6.4 Column names & forecast start year
    cols_cfg = (
        cfg_stage1.get("cols")
        or cfg_hp.get("cols")
        or {}
    )
    SUBSIDENCE_COL = cols_cfg.get("subsidence", "subsidence")
    GWL_COL        = cols_cfg.get("gwl", "GWL")
    FORECAST_START_YEAR = cfg_stage1.get(
        "FORECAST_START_YEAR",
        cfg_hp.get("FORECAST_START_YEAR"),
    )

    # 6.5 Interval calibrator on tuned model (target 80%)
    print("[Eval] Fitting interval calibrator (80%) on validation set...")
    cal80 = fit_interval_calibrator_on_val(
        model_for_eval,
        val_dataset_tf,
        target=0.80,
    )
    np.save(
        os.path.join(RUN_OUTPUT_PATH, "interval_factors_80_tuned.npy"),
        getattr(cal80, "factors_", None),
    )
    print("\n[Eval]Calibrator (tuned) saved.")

    # 6.6 Forecast on chosen split (TEST or VAL fallback)
    X_fore_norm = ensure_input_shapes(
        X_fore,
        mode=MODE,
        forecast_horizon=FORECAST_HORIZON_YEARS,
    )
    y_fore_fmt = map_targets_for_training(y_fore)

    print(f"[Eval] Predicting on {dataset_name_for_forecast} with tuned model...")
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
            "gwl_pred":  h_pred_q_raw,
        }
    else:
        # Point forecasts
        s_pred_raw = data_final[..., :OUT_S_DIM]
        h_pred_raw = data_final[..., OUT_S_DIM:]
        predictions_for_formatter = {
            "subs_pred": s_pred_raw,
            "gwl_pred":  h_pred_raw,
        }

    # y_true mapping for formatter (back to canonical names)
    y_true_for_format = {
        "subsidence": y_fore_fmt["subs_pred"],
        "gwl":        y_fore_fmt["gwl_pred"],
    }

    csv_eval = os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_{MODEL_NAME}_tuned_forecast_"
        f"{dataset_name_for_forecast}_H{FORECAST_HORIZON_YEARS}_calibrated.csv",
    )
    csv_future = os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_{MODEL_NAME}_tuned_forecast_"
        f"{dataset_name_for_forecast}_H{FORECAST_HORIZON_YEARS}_future.csv",
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
        coord_scaler=coord_scaler,
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
        csv_eval_path=csv_eval,
        csv_future_path=csv_future,
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
    )


    if df_eval is not None and not df_eval.empty:
        print(f"[Eval] Saved tuned calibrated EVAL forecast -> {csv_eval}")
    else:
        print("[Warn] Tuned eval forecast DF is empty.")

    if df_future is not None and not df_future.empty:
        print(f"[Eval] Saved tuned calibrated FUTURE forecast -> {csv_future}")
    else:
        print("[Warn] Tuned future forecast DF is empty.")

    # 6.7 Evaluate tuned model with Keras (includes physics losses)
    # Ensure the tuned model is compiled before calling `evaluate`.
    # KerasTuner sometimes returns an uncompiled copy, and loading
    # from `.keras` with `compile=False` also leaves it uncompiled.
    model_for_eval = compile_for_eval(
        model=model_for_eval,
        manifest=M,
        best_hps=best_hps_dict,  # from the tuning summary section
        quantiles=QUANTILES if isinstance(QUANTILES, list) else None,
        include_metrics=True,
    )

    eval_results = {}
    phys = {}
    ds_eval = tf.data.Dataset.from_tensor_slices(
        (X_fore_norm, y_fore_fmt)
    ).batch(BATCH_SIZE)

    try:
        eval_results = model_for_eval.evaluate(
            ds_eval,
            return_dict=True,
            verbose=1,
        )
        print("[Eval] Tuned model evaluate():", eval_results)

        for k in ("epsilon_prior", "epsilon_cons"):
            if k in eval_results:
                phys[k] = float(_to_py(eval_results[k]))
        if phys:
            print("[Eval] Physics diagnostics (tuned):", phys)
    except Exception as e:
        print(f"[Warn] Tuned evaluate() failed: {e}")
        eval_results, phys = {}, {}
        
    # Save physic payload 
    try: 
        physics_payload = model_for_eval.export_physics_payload(
            ds_eval,
            max_batches=None,
            save_path=os.path.join(
                RUN_OUTPUT_PATH, f"{CITY_NAME}_tuned_phys_payload_run_val.npz"
            ),
            format="npz",
            overwrite=True,
            metadata={"city": CITY_NAME, "split": "val"},
        )
        print("[Eval] Tuned Physics payload saved (tuned)")
    except Exception as e: 
        print(f"[Warn] Physic payload saving failed: {e}")
        physics_payload =None 
        

    # 6.8 Interval metrics (coverage / sharpness) in PHYSICAL space
    coverage80_for_abl = None
    sharpness80_for_abl = None

    y_true = None
    s_q = None

    if QUANTILES:
        y_true_list = []
        s_q_list = []
        for xb, yb in with_progress(ds_eval, desc="Tuned interval diagnostics"):
            out_b = model_for_eval(xb, training=False)
            data_final_b = out_b["data_final"]

            y_true_list.append(yb["subs_pred"])               # (B,H,1)
            s_q_list.append(data_final_b[..., :OUT_S_DIM])    # (B,H,Q,1)

        if y_true_list and s_q_list:
            y_true = tf.concat(y_true_list, axis=0)
            s_q = tf.concat(s_q_list, axis=0)

            y_true_phys_np = inverse_scale_target(
                y_true,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )
            s_q_phys_np = inverse_scale_target(
                s_q,
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

    # 6.9 Point metrics + per-horizon metrics (physical units)
    metrics_point = {}
    per_h_mae_dict = None
    per_h_r2_dict = None

    s_med = None
    if QUANTILES and (s_q is not None) and (y_true is not None):
        qs = np.asarray(QUANTILES)
        med_idx = int(np.argmin(np.abs(qs - 0.5)))
        s_med = s_q[..., med_idx, :]    # (N,H,1)
    else:
        y_true_list2 = []
        s_pred_list = []
        for xb, yb in with_progress(ds_eval, desc="Tuned point diagnostics"):
            out_b = model_for_eval(xb, training=False)
            data_final_b = out_b["data_final"]
            y_true_list2.append(yb["subs_pred"])
            s_pred_list.append(data_final_b[..., :OUT_S_DIM])

        if s_pred_list:
            s_med = tf.concat(s_pred_list, axis=0)
            y_true = tf.concat(y_true_list2, axis=0)

    if s_med is not None and (y_true is not None):
        metrics_point = point_metrics(
            y_true,
            s_med,
            use_physical=True,
            scaler_info=scaler_info_dict,
            target_name=SUBSIDENCE_COL,
        )
        per_h_mae_dict, per_h_r2_dict = per_horizon_metrics(
            y_true,
            s_med,
            use_physical=True,
            scaler_info=scaler_info_dict,
            target_name=SUBSIDENCE_COL,
        )

    # 6.10 Save tuned metrics JSON
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_json = os.path.join(
        RUN_OUTPUT_PATH, f"geoprior_eval_phys_tuned_{stamp}.json"
    )

    payload = {
        "timestamp": stamp,
        "tf_version": tf.__version__,
        "numpy_version": np.__version__,
        "city": CITY_NAME,
        "model": MODEL_NAME,
        "horizon": FORECAST_HORIZON_YEARS,
        "quantiles": QUANTILES,
        "dataset_name": dataset_name_for_forecast,
        "metrics_evaluate": {
            k: _to_py(v) for k, v in (eval_results or {}).items()
        },
        "physics_diagnostics": phys,
        "point_metrics": metrics_point,
        "per_horizon": {
            "mae": per_h_mae_dict,
            "r2":  per_h_r2_dict,
        },
        "coverage80": coverage80_for_abl,
        "sharpness80": sharpness80_for_abl,
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Eval] Saved tuned metrics + physics JSON -> {metrics_json}")

    # 6.11 Ablation record row for tuned model
    ABLCFG_TUNED = {
        "PDE_MODE_CONFIG": best_hps_for_eval.get("pde_mode", PDE_MODE_CONFIG),
        "GEOPRIOR_USE_EFFECTIVE_H": USE_EFFECTIVE_H,
        "GEOPRIOR_KAPPA_MODE": best_hps_for_eval.get("kappa_mode"),
        "GEOPRIOR_HD_FACTOR": best_hps_for_eval.get("hd_factor"),
        "LAMBDA_CONS": best_hps_for_eval.get("lambda_cons"),
        "LAMBDA_GW": best_hps_for_eval.get("lambda_gw"),
        "LAMBDA_PRIOR": best_hps_for_eval.get("lambda_prior"),
        "LAMBDA_SMOOTH": best_hps_for_eval.get("lambda_smooth"),
        "LAMBDA_MV": best_hps_for_eval.get("lambda_mv"),
    }

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
    print("[Eval] Tuned ablation record appended.")

print(f"\n---- {CITY_NAME.upper()} {MODEL_NAME} TUNING COMPLETE ----\n"
      f"Artifacts -> {RUN_OUTPUT_PATH}\n")

