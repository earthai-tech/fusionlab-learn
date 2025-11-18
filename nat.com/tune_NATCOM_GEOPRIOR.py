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

from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
try:
    from fusionlab.api.util import get_table_size
    from fusionlab.utils.generic_utils import (
        ensure_directory_exists,
        default_results_dir,
        getenv_stripped,
        print_config_table,   
    )

    from fusionlab.registry.utils import _find_stage1_manifest
    # Import the tuner
    from fusionlab.nn.forecast_tuner import GeoPriorTuner
    from fusionlab.nn.callbacks import NaNGuard 
    
    from fusionlab.utils.nat_utils import load_nat_config
    
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
