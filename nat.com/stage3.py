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


from fusionlab.api.util import get_table_size
from fusionlab.backends.devices import configure_tf_from_cfg
from fusionlab.compat.keras import (
    load_inference_model,
    load_model_from_tfv2,
    save_model, 
)
from fusionlab.nn.pinn.models import GeoPriorSubsNet
from fusionlab.nn.losses import make_weighted_pinball  # if referenced in saved graphs
from fusionlab.nn.pinn.geoprior.models import (
    LearnableMV, LearnableKappa, FixedGammaW, FixedHRef,
)
from fusionlab.utils.audit_utils import (
    should_audit,
    audit_stage3_run,
)
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
    load_nat_config_payload, 
    ensure_input_shapes,
    map_targets_for_training,
    make_tf_dataset,
    load_scaler_info,
    save_ablation_record,
    load_or_rebuild_geoprior_model, 
    compile_for_eval,
)

from fusionlab.utils.shapes import canonicalize_BHQO
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
from fusionlab.nn.pinn.geoprior.scaling import (
    override_scaling_kwargs,
)
# =============================================================================
# 0) Load Stage-1 manifest and arrays
# =============================================================================

RESULTS_DIR = default_results_dir() # auto-resolve

# Desired city/model from NATCOM config payload
cfg_payload   = load_nat_config_payload()
CFG_CITY  = (cfg_payload.get("city") or "").strip().lower() or None
CFG_MODEL = cfg_payload.get("model") or "GeoPriorSubsNet"

# Optional advanced overrides from env
CITY_ENV   = getenv_stripped("CITY")
MODEL_ENV  = getenv_stripped("MODEL_NAME_OVERRIDE")

CITY_HINT  = CITY_ENV or CFG_CITY
MODEL_HINT = MODEL_ENV or CFG_MODEL
MANUAL     = getenv_stripped("STAGE1_MANIFEST")

MANIFEST_PATH = _find_stage1_manifest(
    manual=MANUAL,                   # exact manifest if provided
    base_dir=RESULTS_DIR,            # where to search
    city_hint=CITY_HINT,             # filter by city if set
    model_hint=MODEL_HINT,           # filter by model if set
    prefer="timestamp",              # or "mtime"
    required_keys=("model", "stage"),
    verbose=1,
)

# with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
#     M = json.load(f)
# print(f"[Manifest] Loaded city={M.get('city')} model={M.get('model')}")

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    M = json.load(f)

manifest_city = (M.get("city") or "").strip().lower()
print(f"[Manifest] Loaded city={manifest_city} model={M.get('model')}")

if CFG_CITY and manifest_city and manifest_city != CFG_CITY:
    raise RuntimeError(
        "[NATCOM] Stage-1 manifest city "
        f"{manifest_city!r} does not match config CITY_NAME {CFG_CITY!r}. "
        "Run Stage-1 for this city first, or set CITY/STAGE1_MANIFEST "
        "to explicitly override.\n"
        "#>>> Windows cmd\n"
        "   $ set CITY=zhongshan\n"
        "   $ python nat.com/tune_NATCOM_GEOPRIOR.py\n"
        "\n"
        "#>>> or\n"
        "   $ set CITY=zhongshan\n"
        "   $ python nat.com/tune_NATCOM_GEOPRIOR.py\n"

    )
    
cfg_stage1 = M["config"]        # what Stage-1 used to build sequences
cfg_hp = load_nat_config()      # global config from config.json (4.*)

# Merge global config with Stage-1 snapshot (Stage-1 wins on conflicts)
cfg = dict(cfg_hp)
cfg.update(cfg_stage1)
configure_tf_from_cfg(cfg_hp)

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

# Global physics bounds (from config.py / config.json)
PHYSICS_BOUNDS_CFG = cfg_hp.get("PHYSICS_BOUNDS", {}) or {}

_default_phys_bounds = {
    "H_min": 5.0,
    "H_max": 80.0,
    "K_min": 1e-8,
    "K_max": 1e-3,
    "Ss_min": 1e-7,
    "Ss_max": 1e-3,
    # tau bounds (seconds) — keep consistent with Stage2/config.py
    "tau_min": 7.0 * 86400.0,
    "tau_max": 300.0 * 31556952.0,
}

phys_bounds = dict(_default_phys_bounds)
phys_bounds.update(PHYSICS_BOUNDS_CFG)

bounds_for_scaling = {
    # thickness
    "H_min": float(phys_bounds["H_min"]),
    "H_max": float(phys_bounds["H_max"]),

    # linear bounds (canonical)
    "K_min": float(phys_bounds["K_min"]),
    "K_max": float(phys_bounds["K_max"]),
    "Ss_min": float(phys_bounds["Ss_min"]),
    "Ss_max": float(phys_bounds["Ss_max"]),
    "tau_min": float(phys_bounds["tau_min"]),
    "tau_max": float(phys_bounds["tau_max"]),

    # convenience log-space
    "logK_min": float(np.log(phys_bounds["K_min"])),
    "logK_max": float(np.log(phys_bounds["K_max"])),
    "logSs_min": float(np.log(phys_bounds["Ss_min"])),
    "logSs_max": float(np.log(phys_bounds["Ss_max"])),
    "logTau_min": float(np.log(phys_bounds["tau_min"])),
    "logTau_max": float(np.log(phys_bounds["tau_max"])),
}

# Merge Stage-1 scaling kwargs (critical in v3.2)
sk_stage1 = cfg.get("scaling_kwargs", {}) or {}
sk_model = dict(sk_stage1)

exp_names = sk_model.get("dynamic_feature_names", None)
exp_gwl = sk_model.get("gwl_dyn_index", None)

sk_model = override_scaling_kwargs(
    sk_model,
    cfg,
    dyn_names=exp_names,
    gwl_dyn_index=exp_gwl,
    base_dir=os.path.dirname(__file__),
    strict=True,
    log_fn=print,
)
sk_bounds0 = (sk_model.get("bounds", {}) or {})
sk_model["bounds"] = {**sk_bounds0, **bounds_for_scaling}

LOSS_WEIGHT_GWL = float(cfg.get("LOSS_WEIGHT_GWL", 0.5))
PHYSICS_BOUNDS_MODE = str(cfg.get("PHYSICS_BOUNDS_MODE", "soft") 
                          or "soft").strip().lower()
OFFSET_MODE = str(cfg.get("OFFSET_MODE", "mul") 
                  or "mul").strip().lower()
TIME_UNITS = cfg.get("TIME_UNITS", "year")
RESIDUAL_METHOD = str(cfg.get(
    "CONSOLIDATION_STEP_RESIDUAL_METHOD", "exact") 
    or "exact").strip().lower()

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
    "loss_weights": {"subs_pred": 1.0, "gwl_pred": LOSS_WEIGHT_GWL},
    "bounds_mode": PHYSICS_BOUNDS_MODE,
    "offset_mode": OFFSET_MODE,
    "residual_method": RESIDUAL_METHOD,
    "time_units": TIME_UNITS,
    "pde_mode": PDE_MODE_CONFIG,
    "scale_pde_residuals": SCALE_PDE_RESIDUALS,
    "use_effective_h": USE_EFFECTIVE_H,
    "scaling_kwargs": sk_model,
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

    # Keep dropout near the good regime, but allow a bit of exploration.
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
        "PHYSICS_BOUNDS": phys_bounds,
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
USE_TF_SAVEDMODEL   = bool(cfg.get("USE_TF_SAVEDMODEL", False))
USE_IN_MEMORY_MODEL = bool(cfg.get("USE_IN_MEMORY_MODEL", True))

best_keras_path = os.path.join(
    RUN_OUTPUT_PATH, f"{CITY_NAME}_GeoPrior_best.keras"
 )
best_tf_dir     = os.path.join(
    RUN_OUTPUT_PATH, f"{CITY_NAME}_GeoPrior_best_savedmodel"
)
best_model_path = best_tf_dir if USE_TF_SAVEDMODEL else best_keras_path

best_weights_path = os.path.join(
    RUN_OUTPUT_PATH, f"{CITY_NAME}_GeoPrior_best.weights.h5"
)
best_hps_path     = os.path.join(
    RUN_OUTPUT_PATH, f"{CITY_NAME}_GeoPrior_best_hps.json"
)

# Optional: write a lightweight init-manifest for this tuned model
model_init_manifest_path = os.path.join(
    RUN_OUTPUT_PATH, f"{CITY_NAME}_GeoPrior_best_manifest.json"
)
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


model_init_manifest = {
    "city": CITY_NAME,
    "model": "GeoPriorSubsNet",
    "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    "dims": {
        "static_input_dim": STATIC_DIM,
        "dynamic_input_dim": DYNAMIC_DIM,
        "future_input_dim": FUTURE_DIM,
        "output_subsidence_dim": OUT_S_DIM,
        "output_gwl_dim": OUT_G_DIM,
        "forecast_horizon": FORECAST_HORIZON_YEARS,
    },
    "config": {
        # keep JSON-safe only
        "fixed_params": fixed_params,
        "best_hps": (best_hps_dict or {}),
    },
}

if best_model is not None:
    save_model(
        model=best_model,
        keras_path=best_model_path,
        weights_path=best_weights_path,
        manifest_path=model_init_manifest_path,
        manifest=model_init_manifest,
        overwrite=True,
        use_tf_format=USE_TF_SAVEDMODEL,
    )
    print(f"\nSaved best tuned model to: {best_model_path}")
else:
    print("\n[Warn] No best model returned by tuner.")
    best_weights_path = None


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

# Optional: if you saved HPs to disk, prefer reading them (no "best_hps_disk" needed)
if (not best_hps_for_eval) and os.path.isfile(best_hps_path):
    with open(best_hps_path, "r", encoding="utf-8") as f:
        best_hps_for_eval.update(json.load(f) or {})

# If tuner did not return a model object, load robustly from disk
if model_for_eval is None:
    # Keep this aligned with Stage2 loader setup
    custom_objects_load = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
    }

    if USE_TF_SAVEDMODEL and os.path.isdir(best_tf_dir):
        model_for_eval = load_model_from_tfv2(
            best_tf_dir,
            endpoint="serve",
            custom_objects=custom_objects_load,
        )
        print("[OK] Loaded inference model from TF SavedModel:", best_tf_dir)
    else:
        # Build-inputs: small slice is enough to build subclassed models
        build_inputs = {k: (v[:2] if hasattr(v, "__len__") else v)
                        for k, v in val_inputs_np.items() if v is not None}

        # Minimal builder: use your known fixed_params (already v3.2-updated)
        def builder(_manifest: dict):
            # IMPORTANT: builder must return a fresh model instance with same init-params
            return GeoPriorSubsNet(**{
                k: fixed_params[k] for k in fixed_params
                if k in (
                    "static_input_dim", "dynamic_input_dim", "future_input_dim",
                    "output_subsidence_dim", "output_gwl_dim",
                    "forecast_horizon", "quantiles", "pde_mode",
                    "offset_mode", "bounds_mode", "residual_method",
                    "time_units", "scale_pde_residuals", "use_effective_h",
                    "scaling_kwargs", "attention_levels", "mode",
                    # + any other v3.2 init-controls passed in fixed_params
                )
            })
        try: 
            model_for_eval = load_inference_model(
                keras_path=best_keras_path,
                weights_path=best_weights_path,
                manifest_path=model_init_manifest_path,
                custom_objects=custom_objects_load,
                compile=False,                  # Stage3 diagnostics are manual
                builder=builder,
                build_inputs=build_inputs,
                prefer_full_model=False,        # allow rebuild+weights fallback
                log_fn=print,
                use_in_memory_model=False,
            )
            print("[OK] Loaded inference model from bundle:", type(model_for_eval))
        except: 
            model_for_eval, best_hps_disk = load_or_rebuild_geoprior_model(
                model_path=best_keras_path,
                manifest=M,
                X_sample=X_val,
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
        best_hps_for_eval.update(best_hps_disk or {})
        
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
    
    # robust: tolerate dict vs list outputs (recommended)
    # subs_pred_raw, h_pred_raw = extract_preds(model_for_eval, pred_dict)
    s_pred_raw = pred_dict["subs_pred"]
    h_pred_raw = pred_dict["gwl_pred"]
    
    # ---- BHQO canonicalize (uses y_true available here) ----
    y_true_scaled = None
    if "subs_pred" in y_fore_fmt:
        y_true_scaled = np.asarray(
            y_fore_fmt["subs_pred"][..., :1],
            np.float32,
        )
    
    if (
        (y_true_scaled is not None)
        and (s_pred_raw is not None)
        and (int(np.asarray(s_pred_raw).ndim) == 4)
    ):
        sp = tf.convert_to_tensor(s_pred_raw)
        yt = tf.convert_to_tensor(y_true_scaled)
    
        sp = canonicalize_BHQO(
            sp,
            y_true=yt,
            q_values=QUANTILES,
            n_q=len(QUANTILES),
            enforce_monotone=True,
            layout="BHQO",
            verbose=0,
            log_fn=None,
        )
        s_pred_raw = sp.numpy()
    elif (s_pred_raw is not None) and int(np.asarray(s_pred_raw).ndim) == 4:
        s_pred_raw = np.sort(np.asarray(s_pred_raw), axis=2)
    
    # ---- now calibration (expects correct Q axis) ----
    if cal80 is not None and QUANTILES:
        s_pred_cal = apply_calibrator_to_subs(cal80, s_pred_raw)
    else:
        s_pred_cal = s_pred_raw
    
    predictions_for_formatter = {
        "subs_pred": s_pred_cal,
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
        value_mode="cumulative", # set to "rate" to convert back to rate 
        input_value_mode="cumulative",
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
            
            yt_b = yb["subs_pred"][..., :1]
            sq_b = out_b["subs_pred"]
            
            sq_b = canonicalize_BHQO(
                sq_b,
                y_true=yt_b,
                q_values=QUANTILES,
                n_q=len(QUANTILES),
                enforce_monotone=True,
                layout="BHQO",
                verbose=0,
                log_fn=None,
            )
            
            # For coverage/sharpness, you typically want *calibrated* intervals:
            if cal80 is not None:
                sq_b = apply_calibrator_to_subs(cal80, sq_b)
            
            y_true_list.append(yb["subs_pred"])
            s_q_list.append(sq_b)

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
            sq_b = out_b["subs_pred"]
            yt_b = yb["subs_pred"][..., :1]
            
            sq_b = canonicalize_BHQO(
                sq_b,
                y_true=yt_b,
                q_values=QUANTILES,
                n_q=len(QUANTILES),
                enforce_monotone=True,
                layout="BHQO",
            )
            
            # median index
            qs = np.asarray(QUANTILES, dtype=float)
            mid = int(np.argmin(np.abs(qs - 0.5)))
            s_med_b = sq_b[..., mid, :]   # (B,H,1)

            y_true_list2.append(yb["subs_pred"])
            s_pred_list.append(out_b["subs_pred"])

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

    if should_audit(cfg.get("AUDIT_STAGES"), stage="stage3"):
        audit_stage3_run(
            manifest_path=MANIFEST_PATH,
            manifest=M,
            cfg=cfg,
            fixed_params=fixed_params,
            best_hps=best_hps_for_eval,
            run_dir=RUN_OUTPUT_PATH,
            best_model_path=best_model_path,
            best_weights_path=best_weights_path,
            use_tf_savedmodel=USE_TF_SAVEDMODEL,
            quantiles=QUANTILES,
            forecast_horizon=FORECAST_HORIZON_YEARS,
            mode=MODE,
            pred_shapes={
                "subs_pred": tuple(s_pred_raw.shape),
                "gwl_pred": tuple(h_pred_raw.shape),
            },
            eval_results=eval_results,
            phys_diag=phys,
            calibrator_factors=getattr(cal80, "factors_", None),
            forecast_csv_eval=csv_eval,
            forecast_csv_future=csv_future,
            metrics_json_path=metrics_json,
            physics_payload_path=os.path.join(
                RUN_OUTPUT_PATH,
                f"{CITY_NAME}_tuned_phys_payload_run_val.npz",
            ),
            save_dir=RUN_OUTPUT_PATH,
            city=CITY_NAME,
            model_name=MODEL_NAME,
        )


print(f"\n---- {CITY_NAME.upper()} {MODEL_NAME} TUNING COMPLETE ----\n"
      f"Artifacts -> {RUN_OUTPUT_PATH}\n")

