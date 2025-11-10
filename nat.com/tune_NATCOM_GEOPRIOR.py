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

# import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

# --- fusionlab imports ---
try:
    from fusionlab.utils.generic_utils import ensure_directory_exists
    # Import the tuner
    from fusionlab.nn.forecast_tuner import GeoPriorTuner
    from fusionlab.nn.callbacks import NaNGuard 
    
    print("Successfully imported fusionlab modules for tuning.")
except Exception as e:
    print(f"Failed to import fusionlab modules: {e}")
    raise


# =============================================================================
# 0) Load Stage-1 manifest and arrays
# =============================================================================
MANIFEST_PATH = "results/nansha_GeoPriorSubsNet_stage1/manifest.json"
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    M = json.load(f)

# Resolve paths from manifest (keeps the script relocatable)
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
        # Optional: clip extreme outliers to a robust range
        if v.ndim > 0:
            p99 = np.percentile(v, 99) if np.isfinite(v).any() else 0.0
            if p99 > 0:
                v = np.clip(v, -10*p99, 10*p99)
        X[k] = v
    # Thickness must be positive to avoid divisions later on
    if "H_field" in X:
        X["H_field"] = np.maximum(X["H_field"], 1e-3).astype(np.float32)
    return X

X_train = _sanitize_inputs(X_train)
X_val   = _sanitize_inputs(X_val)

# Optional scalers/encoders
main_scaler = None
if M["artifacts"]["encoders"].get("main_scaler"):
    main_scaler = joblib.load(M["artifacts"]["encoders"]["main_scaler"])
coord_scaler = joblib.load(M["artifacts"]["encoders"]["coord_scaler"])
scaler_info = M["artifacts"]["encoders"]["scaler_info"]

cfg = M["config"]
CITY_NAME = M.get("city", "nansha")
MODEL_NAME = M.get("model", "GeoPriorSubsNet")
TIME_STEPS = cfg["TIME_STEPS"]
FORECAST_HORIZON_YEARS = cfg["FORECAST_HORIZON_YEARS"]
MODE = cfg["MODE"]

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
# Training loop setup
EPOCHS = 50
BATCH_SIZE = 32

# Probabilistic outputs default
QUANTILES = [0.1, 0.5, 0.9]

# PDE selection
PDE_MODE_CONFIG = "both"  # can be overridden by search space

# Attention levels used in architecture
ATTENTION_LEVELS = ["cross", "hierarchical", "memory"]

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
    "scale_pde_residuals": True,
    "architecture_config": {
        "encoder_type": "hybrid",
        "decoder_attention_stack": ATTENTION_LEVELS,
        "feature_processing": "vsn",
    },
}

# 2.2 Hyperparameter search space
search_space = {
    # --- Architecture (model.__init__) ---
    "embed_dim": [32, 64],
    "hidden_units": [64, 96, 128],
    "lstm_units": [64, 96],
    "attention_units": [32, 64],
    "num_heads": [2, 4],
    "dropout_rate": {"type": "float", "min_value": 0.10, "max_value": 0.30},
    "use_vsn": {"type": "bool"},
    "vsn_units": [16, 32, 48],
    "use_batch_norm": {"type": "bool"},
    # Physics switches
    "pde_mode": ["both", "consolidation", "gw_flow"],
    "scale_pde_residuals": {"type": "bool"},
    "kappa_mode": ["bar", "kb"],
    "hd_factor": {"type": "float", "min_value": 0.50, "max_value": 0.80},
    # Learnable scalar initials (wrapped inside model __init__)
    "mv": {"type": "float", "min_value": 3e-7, "max_value": 1e-6, "sampling": "log"},
    "kappa": {"type": "float", "min_value": 0.7, "max_value": 1.5},
    # --- Compile-only (model.compile) ---
    "learning_rate": {"type": "float", "min_value": 5e-5, "max_value": 3e-4, "sampling": "log"},
    "lambda_gw": {"type": "float", "min_value": 0.1, "max_value": 1.0},
    "lambda_cons": {"type": "float", "min_value": 0.1, "max_value": 1.0},
    "lambda_prior": {"type": "float", "min_value": 0.1, "max_value": 0.8},
    "lambda_smooth": {"type": "float", "min_value": 0.0, "max_value": 1.0},
    "lambda_mv": {"type": "float", "min_value": 0.0, "max_value": 0.5},
    "mv_lr_mult": {"type": "float", "min_value": 0.5, "max_value": 2.0},
    "kappa_lr_mult": {"type": "float", "min_value": 1.0, "max_value": 10.0},
}


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
best_model_path = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_GeoPrior_best.keras")
best_hps_path = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_GeoPrior_best_hps.json")

if best_model is not None:
    best_model.save(best_model_path)
    print(f"\nSaved best tuned model to: {best_model_path}")
else:
    print("\n[Warn] No best model returned by tuner.")

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
    "best_hps": best_hps_dict,
}
with open(os.path.join(RUN_OUTPUT_PATH, "tuning_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2)

print("\nTuning complete. You can now proceed to calibration/forecasting using "
      "the saved best model or continue experimentation with the KerasTuner logs.")
