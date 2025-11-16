# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
training_NATCOM_GEOPRIOR.py
Stage-2 (training): GeoPriorSubsNet with physics-informed losses.
Consumes Stage-1 artifacts (manifest + NPZs), trains, calibrates, and
exports calibrated forecasts & diagnostics.

Key corrections:
- Calibrate subsidence quantiles on validation set FIRST, then format to
  DataFrame (inverse scaling, coords, etc.). Physics is enforced upstream
  during training via PINN loss terms.
"""

from __future__ import annotations

import os
import json
import joblib
import numpy as np
import datetime as dt
import warnings
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# Silence common warnings and TF logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
if hasattr(tf, "autograph") and hasattr(tf.autograph, "set_verbosity"):
    tf.autograph.set_verbosity(0)

# ---------------- fusionlab imports ----------------
try:
    from fusionlab.api.util import get_table_size
    from fusionlab.utils.generic_utils import ensure_directory_exists, save_all_figures
    from fusionlab.utils.generic_utils import default_results_dir, getenv_stripped
    from fusionlab.utils.generic_utils import print_config_table
    from fusionlab.registry.utils import _find_stage1_manifest
    from fusionlab.utils.nat_utils import load_nat_config  

    from fusionlab.nn.pinn.models import GeoPriorSubsNet
    from fusionlab.params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
    from fusionlab.nn.losses import make_weighted_pinball
    from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn, _to_py
    from fusionlab.nn.calibration import (
        fit_interval_calibrator_on_val,
        apply_calibrator_to_subs,
    )
    from fusionlab.nn.utils import plot_history_in
    from fusionlab.nn.pinn.op import extract_physical_parameters
    from fusionlab.nn.pinn.utils import format_pinn_predictions
    from fusionlab.plot.forecast import plot_forecasts, forecast_view

    print("Successfully imported fusionlab modules.")
except Exception as e:
    print(f"Critical Error: fusionlab imports failed: {e}")
    raise

# =============================================================================
# Config / Paths
# =============================================================================

RESULTS_DIR = default_results_dir()  # smart default auto-resolve
CITY_HINT   = getenv_stripped("CITY")  # -> None if unset/empty
MODEL_HINT  = getenv_stripped("MODEL_NAME_OVERRIDE", default="GeoPriorSubsNet")
MANUAL      = getenv_stripped("STAGE1_MANIFEST")  # exact path if provided

MANIFEST_PATH = _find_stage1_manifest(
    manual=MANUAL,
    base_dir=RESULTS_DIR,
    city_hint=CITY_HINT,         # e.g., "zhongshan"; None means "no filter"
    model_hint=MODEL_HINT,
    prefer="timestamp",          # or "mtime"
    required_keys=("model", "stage"),
    verbose=1,
)

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    M = json.load(f)

print(f"[Manifest] Loaded city={M.get('city')} model={M.get('model')}")

# -------------------------------------------------------------------------
# Merge global NATCOM config (config.json) with Stage-1 manifest config.
# - load_nat_config() → central config.json["config"] (model/physics/training)
# - M["config"]       → Stage-1 snapshot (TIME_STEPS/HORIZON/MODE + features,
#                        censoring report, etc.)
# We take config.json as base, then let manifest override where needed
# (especially TIME_STEPS / HORIZON / feature lists that were actually used).
# -------------------------------------------------------------------------
cfg_global   = load_nat_config()          # from config.json
cfg_manifest = M.get("config", {}) or {}  # from manifest.json

cfg = dict(cfg_global)
cfg.update(cfg_manifest)  # manifest wins on overlapping keys

CITY_NAME  = M.get("city",  cfg.get("CITY_NAME", "nansha"))
MODEL_NAME = M.get("model", cfg.get("MODEL_NAME", "GeoPriorSubsNet"))

# ----- Optional censoring config (from merged cfg) -----------------------
FEATURES   = cfg.get("features", {}) or {}
DYN_NAMES  = FEATURES.get("dynamic", []) or []

CENSOR     = cfg.get("censoring", {}) or cfg.get("censor", {}) or {}
CENSOR_SPECS  = CENSOR.get("specs", []) or []
CENSOR_THRESH = float(CENSOR.get("flag_threshold", 0.5))

# Resolve the first available censor flag column (if any) in dynamic_features
CENSOR_FLAG_IDX  = None
CENSOR_FLAG_NAME = None
for sp in CENSOR_SPECS:
    # Allow either explicit 'flag_col' or 'col' + optional 'flag_suffix'
    cand = sp.get("flag_col")
    if not cand:
        base = sp.get("col")
        if base:
            cand = base + sp.get("flag_suffix", "_censored")
    if cand and cand in DYN_NAMES:
        CENSOR_FLAG_NAME = cand
        CENSOR_FLAG_IDX  = DYN_NAMES.index(cand)
        print("[Info] Censor flags present in dynamic features:", cand)
        break

# ---- Model / physics / training config ---------------------------------
# NOTE: TIME_STEPS / FORECAST_HORIZON_YEARS / MODE are taken from cfg
# AFTER merging, so they reflect what Stage-1 actually used.
TIME_STEPS            = cfg["TIME_STEPS"]
FORECAST_HORIZON_YEARS = cfg["FORECAST_HORIZON_YEARS"]
MODE                  = cfg["MODE"]

ATTENTION_LEVELS      = cfg.get("ATTENTION_LEVELS", ["cross", "hierarchical", "memory"])
SCALE_PDE_RESIDUALS   = cfg.get("SCALE_PDE_RESIDUALS", True)

EMBED_DIM    = cfg.get("EMBED_DIM", 32)
HIDDEN_UNITS = cfg.get("HIDDEN_UNITS", 64)
LSTM_UNITS   = cfg.get("LSTM_UNITS", 64)
ATTENTION_UNITS = cfg.get("ATTENTION_UNITS", 64)
NUMBER_HEADS    = cfg.get("NUMBER_HEADS", 2)
DROPOUT_RATE    = cfg.get("DROPOUT_RATE", 0.10)

MEMORY_SIZE    = cfg.get("MEMORY_SIZE", 50)
SCALES         = cfg.get("SCALES", [1, 2])
USE_RESIDUALS  = cfg.get("USE_RESIDUALS", True)
USE_BATCH_NORM = cfg.get("USE_BATCH_NORM", False)   # FIXED key
USE_VSN        = cfg.get("USE_VSN", True)           # FIXED key
VSN_UNITS      = cfg.get("VSN_UNITS", 32)

# Helper: JSON has string keys for quantile weight dicts; coerce to float.
def _coerce_quantile_weights(d: dict, default: dict) -> dict:
    if not d:
        return default
    out = {}
    for k, v in d.items():
        try:
            q = float(k)
        except (TypeError, ValueError):
            q = k
        out[q] = float(v)
    return out

# Probabilistic outputs
QUANTILES = cfg.get("QUANTILES", [0.1, 0.5, 0.9])

SUBS_WEIGHTS_RAW = cfg.get(
    "SUBS_WEIGHTS",
    {0.1: 3.0, 0.5: 1.0, 0.9: 3.0},
)
GWL_WEIGHTS_RAW = cfg.get(
    "GWL_WEIGHTS",
    {0.1: 1.5, 0.5: 1.0, 0.9: 1.5},
)

SUBS_WEIGHTS = _coerce_quantile_weights(SUBS_WEIGHTS_RAW, {0.1: 3.0, 0.5: 1.0, 0.9: 3.0})
GWL_WEIGHTS  = _coerce_quantile_weights(GWL_WEIGHTS_RAW,  {0.1: 1.5, 0.5: 1.0, 0.9: 1.5})

# Physics loss weights
# Config uses e.g. "off", "both", "gw_flow", "consolidation"
PDE_MODE_CONFIG = cfg.get("PDE_MODE_CONFIG", "off")
LAMBDA_CONS   = cfg.get("LAMBDA_CONS",   0.10)
LAMBDA_GW     = cfg.get("LAMBDA_GW",     0.01)
LAMBDA_PRIOR  = cfg.get("LAMBDA_PRIOR",  0.10)
LAMBDA_SMOOTH = cfg.get("LAMBDA_SMOOTH", 0.01)
LAMBDA_MV     = cfg.get("LAMBDA_MV",     0.01)
MV_LR_MULT    = cfg.get("MV_LR_MULT",    1.0)
KAPPA_LR_MULT = cfg.get("KAPPA_LR_MULT", 5.0)

# GeoPrior scalar params
GEOPRIOR_INIT_MV    = cfg.get("GEOPRIOR_INIT_MV",    1e-7)
GEOPRIOR_INIT_KAPPA = cfg.get("GEOPRIOR_INIT_KAPPA", 1.0)
GEOPRIOR_GAMMA_W    = cfg.get("GEOPRIOR_GAMMA_W",    9810.0)
GEOPRIOR_H_REF      = cfg.get("GEOPRIOR_H_REF",      0.0)
GEOPRIOR_KAPPA_MODE = cfg.get("GEOPRIOR_KAPPA_MODE", "bar")
GEOPRIOR_USE_EFFECTIVE_H = cfg.get(
    "GEOPRIOR_USE_EFFECTIVE_H",
    CENSOR.get("use_effective_h_field", True),
)
GEOPRIOR_HD_FACTOR  = cfg.get("GEOPRIOR_HD_FACTOR", 0.6)

# Targets & columns (for formatter)
cols_cfg = cfg.get("cols", {})
SUBSIDENCE_COL = cols_cfg.get("subsidence", "subsidence")
GWL_COL        = cols_cfg.get("gwl", "GWL")

# Train options
EPOCHS        = cfg.get("EPOCHS", 50)
BATCH_SIZE    = cfg.get("BATCH_SIZE", 32)
LEARNING_RATE = cfg.get("LEARNING_RATE", 1e-4)

# Output directory (reuse Stage-1 run_dir)
BASE_OUTPUT_DIR = M["paths"]["run_dir"]
STAMP = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, f"train_{STAMP}")
ensure_directory_exists(RUN_OUTPUT_PATH)

config_sections = [
    ("Run", {
        "CITY_NAME": CITY_NAME,
        "MODEL_NAME": MODEL_NAME,
        "RESULTS_DIR": RESULTS_DIR,
        "MANIFEST_PATH": MANIFEST_PATH,
        "RUN_OUTPUT_PATH": RUN_OUTPUT_PATH,
    }),
    ("Architecture", {
        "TIME_STEPS": TIME_STEPS,
        "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
        "MODE": MODE,
        "ATTENTION_LEVELS": ATTENTION_LEVELS,
        "EMBED_DIM": EMBED_DIM,
        "HIDDEN_UNITS": HIDDEN_UNITS,
        "LSTM_UNITS": LSTM_UNITS,
        "ATTENTION_UNITS": ATTENTION_UNITS,
        "NUMBER_HEADS": NUMBER_HEADS,
        "DROPOUT_RATE": DROPOUT_RATE,
        "MEMORY_SIZE": MEMORY_SIZE,
        "SCALES": SCALES,
        "USE_RESIDUALS": USE_RESIDUALS,
        "USE_BATCH_NORM": USE_BATCH_NORM,
        "USE_VSN": USE_VSN,
        "VSN_UNITS": VSN_UNITS,
    }),
    ("Physics", {
        "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
        "SCALE_PDE_RESIDUALS": SCALE_PDE_RESIDUALS,
        "LAMBDA_CONS": LAMBDA_CONS,
        "LAMBDA_GW": LAMBDA_GW,
        "LAMBDA_PRIOR": LAMBDA_PRIOR,
        "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
        "LAMBDA_MV": LAMBDA_MV,
        "MV_LR_MULT": MV_LR_MULT,
        "KAPPA_LR_MULT": KAPPA_LR_MULT,
        "GEOPRIOR_INIT_MV": GEOPRIOR_INIT_MV,
        "GEOPRIOR_INIT_KAPPA": GEOPRIOR_INIT_KAPPA,
        "GEOPRIOR_GAMMA_W": GEOPRIOR_GAMMA_W,
        "GEOPRIOR_H_REF": GEOPRIOR_H_REF,
        "GEOPRIOR_KAPPA_MODE": GEOPRIOR_KAPPA_MODE,
        "GEOPRIOR_USE_EFFECTIVE_H": GEOPRIOR_USE_EFFECTIVE_H,
        "GEOPRIOR_HD_FACTOR": GEOPRIOR_HD_FACTOR,
    }),
    ("Training", {
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "QUANTILES": QUANTILES,
        "SUBS_WEIGHTS": SUBS_WEIGHTS,
        "GWL_WEIGHTS": GWL_WEIGHTS,
    }),
]

print_config_table(
    config_sections, table_width =get_table_size(), 
    title=f"{CITY_NAME.upper()} {MODEL_NAME} TRAINING CONFIG",
)

print(f"\nTraining outputs -> {RUN_OUTPUT_PATH}")

# Encoders / scalers
def _load_scaler_info(encoders_block: dict):
    """Load scaler_info (possibly a joblib path) so 
    formatters can inverse-transform."""
    si = encoders_block.get("scaler_info")
    if isinstance(si, str) and os.path.exists(si):
        try:
            return joblib.load(si)
        except Exception:
            pass
    # Could already be a plain dict with lightweight info (without the scaler object):
    return si

encoders = M["artifacts"]["encoders"]

# If OHE is present, it may be a single path or a {col:path} dict; we don't need it here.
ohe_block = encoders.get("ohe")
if isinstance(ohe_block, dict):
    print(f"[Info] {len(ohe_block)} OHE encoders recorded in manifest.")
elif isinstance(ohe_block, str):
    print("[Info] Single OHE encoder path recorded in manifest.")
else:
    print("[Info] No OHE encoders recorded (or not needed).")
    
# Try to load the plain scaler, but don't crash if missing
main_scaler = None
ms_path = encoders.get("main_scaler")
if ms_path and os.path.exists(ms_path):
    try:
        main_scaler = joblib.load(ms_path)
    except Exception as e:
        print(f"[Warn] Could not load main_scaler at {ms_path}: {e}")
else:
    print("[Warn] main_scaler path missing in manifest or file"
          " not found; continuing without it.")

# coord_scaler is optional but helpful for coords inverse-transform
coord_scaler = None
cs_path = encoders.get("coord_scaler")
if cs_path and os.path.exists(cs_path):
    try:
        coord_scaler = joblib.load(cs_path)
    except Exception as e:
        print(f"[Warn] Could not load coord_scaler at {cs_path}: {e}")

# Load scaler_info mapping (dict or path)
scaler_info_dict = _load_scaler_info(encoders)

# If scaler_info is a dict with only 'scaler_path',
#  proactively attach the loaded scaler objects
if isinstance(scaler_info_dict, dict):
    for k, v in scaler_info_dict.items():
        if isinstance(v, dict) and "scaler_path" in v and "scaler" not in v:
            p = v["scaler_path"]
            if p and os.path.exists(p):
                try:
                    v["scaler"] = joblib.load(p)
                except Exception:
                    pass
feat_reg = cfg.get("feature_registry", {})
if feat_reg:
    print("\n[Info] Stage-1 feature registry summary:")
    for k in ("resolved_optional_numeric", "resolved_optional_categorical",
              "already_normalized", "future_drivers_declared"):
        if k in feat_reg:
            print(f"  - {k}: {feat_reg[k]}")

# NPZs
train_inputs_npz  = M["artifacts"]["numpy"]["train_inputs_npz"]
train_targets_npz = M["artifacts"]["numpy"]["train_targets_npz"]
val_inputs_npz    = M["artifacts"]["numpy"]["val_inputs_npz"]
val_targets_npz   = M["artifacts"]["numpy"]["val_targets_npz"]
test_inputs_npz   = M["artifacts"]["numpy"].get("test_inputs_npz")   # optional
test_targets_npz  = M["artifacts"]["numpy"].get("test_targets_npz")  # optional

X_train = dict(np.load(train_inputs_npz))
y_train = dict(np.load(train_targets_npz))
X_val   = dict(np.load(val_inputs_npz))
y_val   = dict(np.load(val_targets_npz))
X_test  = dict(np.load(test_inputs_npz)) if test_inputs_npz else None
y_test  = dict(np.load(test_targets_npz)) if test_targets_npz else None

# Dims
OUT_S_DIM = M["artifacts"]["sequences"]["dims"]["output_subsidence_dim"]
OUT_G_DIM = M["artifacts"]["sequences"]["dims"]["output_gwl_dim"]

# =============================================================================
# Helpers
# =============================================================================
def _map_targets_for_training(y_dict: dict) -> dict:
    """Remap canonical keys to model compile keys."""
    if "subsidence" in y_dict and "gwl" in y_dict:
        return {"subs_pred": y_dict["subsidence"], "gwl_pred": y_dict["gwl"]}
    # Backward-compat if exported with 'subs_pred'/'gwl_pred'
    if "subs_pred" in y_dict and "gwl_pred" in y_dict:
        return y_dict
    raise KeyError("Targets must contain ('subsidence','gwl') or ('subs_pred','gwl_pred').")

def _ensure_input_shapes(x: dict) -> dict:
    """Ensure presence of zero-width placeholders if missing."""
    out = dict(x)
    N = out["dynamic_features"].shape[0]
    if out.get("static_features") is None:
        out["static_features"] = np.zeros((N, 0), dtype=np.float32)
    if out.get("future_features") is None:
        # Most Stage-1 exports already have the correct shape; if not, fall back.
        # Use past+future or horizon depending on MODE if truly absent.
        t_future = out["dynamic_features"].shape[1] if MODE == "tft_like" else FORECAST_HORIZON_YEARS
        out["future_features"] = np.zeros((N, t_future, 0), dtype=np.float32)
    return out

def make_tf_dataset(X_np: dict, y_np: dict, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    Xin = _ensure_input_shapes(X_np)
    Yin = _map_targets_for_training(y_np)
    ds = tf.data.Dataset.from_tensor_slices((Xin, Yin))
    if shuffle:
        ds = ds.shuffle(buffer_size=Xin["dynamic_features"].shape[0], seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# =============================================================================
# Build datasets
# =============================================================================
train_dataset = make_tf_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
val_dataset   = make_tf_dataset(X_val,   y_val,   BATCH_SIZE, shuffle=False)

print("\nDataset sample shapes:")
for xb, yb in train_dataset.take(1):
    for k, v in xb.items():
        print(f"  X[{k:>16}] -> {tuple(v.shape)}")
    for k, v in yb.items():
        print(f"  y[{k:>16}] -> {tuple(v.shape)}")

# =============================================================================
# Build & compile model
# =============================================================================
s_dim_model = _ensure_input_shapes(X_train)["static_features"].shape[-1]
d_dim_model = _ensure_input_shapes(X_train)["dynamic_features"].shape[-1]
f_dim_model = _ensure_input_shapes(X_train)["future_features"].shape[-1]

subsmodel_params = {
    "embed_dim": EMBED_DIM,
    "hidden_units": HIDDEN_UNITS,
    "lstm_units": LSTM_UNITS,
    "attention_units": ATTENTION_UNITS,
    "num_heads": NUMBER_HEADS,
    "dropout_rate": DROPOUT_RATE,
    "max_window_size": TIME_STEPS,
    "memory_size": MEMORY_SIZE,
    "scales": SCALES,
    "multi_scale_agg": "last",
    "final_agg": "last",
    "use_residuals": USE_RESIDUALS,
    "use_batch_norm": USE_BATCH_NORM,
    "use_vsn": USE_VSN,
    "vsn_units": VSN_UNITS,
    "mode": MODE,
    "attention_levels": ATTENTION_LEVELS,
    "scale_pde_residuals": SCALE_PDE_RESIDUALS,
    # GeoPrior scalar params
    "mv": LearnableMV(initial_value=GEOPRIOR_INIT_MV),
    "kappa": LearnableKappa(initial_value=GEOPRIOR_INIT_KAPPA),
    "gamma_w": FixedGammaW(value=GEOPRIOR_GAMMA_W),
    "h_ref": FixedHRef(value=GEOPRIOR_H_REF),
    "kappa_mode": GEOPRIOR_KAPPA_MODE,
    "use_effective_h": GEOPRIOR_USE_EFFECTIVE_H,
    "hd_factor": GEOPRIOR_HD_FACTOR,
}

subs_model_inst = GeoPriorSubsNet(
    static_input_dim=s_dim_model,
    dynamic_input_dim=d_dim_model,
    future_input_dim=f_dim_model,
    output_subsidence_dim=OUT_S_DIM,
    output_gwl_dim=OUT_G_DIM,
    forecast_horizon=FORECAST_HORIZON_YEARS,
    quantiles=QUANTILES,
    pde_mode=PDE_MODE_CONFIG,
    **subsmodel_params,
)
#%
# Build once
for xb, _ in train_dataset.take(1):
    subs_model_inst(xb)
    break
subs_model_inst.summary(line_length=110, expand_nested=True)

loss_dict = {
    "subs_pred": make_weighted_pinball(QUANTILES, SUBS_WEIGHTS) if QUANTILES else tf.keras.losses.MSE,
    "gwl_pred":  make_weighted_pinball(QUANTILES, GWL_WEIGHTS)  if QUANTILES else tf.keras.losses.MSE,
}
metrics_dict = {
    "subs_pred": ["mae", "mse"] + ([coverage80_fn, sharpness80_fn] if QUANTILES else []),
    "gwl_pred":  ["mae", "mse"],
}
loss_weights_dict = {"subs_pred": 1.0, "gwl_pred": 0.5}
physics_loss_weights = {
    "lambda_cons": LAMBDA_CONS,
    "lambda_gw": LAMBDA_GW,
    "lambda_prior": LAMBDA_PRIOR,
    "lambda_smooth": LAMBDA_SMOOTH,
    "lambda_mv": LAMBDA_MV,
    "mv_lr_mult": MV_LR_MULT,
    "kappa_lr_mult": KAPPA_LR_MULT,
}

subs_model_inst.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss=loss_dict,
    metrics=metrics_dict,
    loss_weights=loss_weights_dict,
    **physics_loss_weights,
)
print(f"{MODEL_NAME} compiled.")

# =============================================================================
# Train
# =============================================================================

ckpt_name = f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_HORIZON_YEARS}.keras"
ckpt_path = os.path.join(RUN_OUTPUT_PATH, ckpt_name)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", save_best_only=True,
                    save_weights_only=False, verbose=1),
]

print("\nTraining...")
history = subs_model_inst.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)
print(f"Best val_loss: {min(history.history.get('val_loss', [np.inf])):.4f}")

# Plots
history_groups = {
    "Total Loss": ["loss", "val_loss"],
    "Physics Loss": ["physics_loss", "val_physics_loss"],
    "Data Loss": ["data_loss", "val_data_loss"],
    "Component Losses": [
        "consolidation_loss", "val_consolidation_loss",
        "gw_flow_loss", "val_gw_flow_loss",
        "prior_loss", "val_prior_loss",
        "smooth_loss", "val_smooth_loss",
    ],
    "Subsidence MAE": ["subs_pred_mae", "val_subs_pred_mae"],
    "GWL MAE": ["gwl_pred_mae", "val_gwl_pred_mae"],
}
yscales = {
    "Total Loss": "log",
    "Physics Loss": "log",
    "Data Loss": "log",
    "Component Losses": "log",
    "Subsidence MAE": "linear",
    "GWL MAE": "linear",
}
plot_history_in(
    history.history, metrics=history_groups, 
    title=f"{MODEL_NAME} Training History",
    yscale_settings=yscales,
    layout="subplots",  
    savefig=os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_{MODEL_NAME.lower()}_training_history_plot"),
)

# Extract physical parameters
extract_physical_parameters(
    subs_model_inst, to_csv=True,
    filename=f"{CITY_NAME}_{MODEL_NAME.lower()}_physical_parameters.csv",
    save_dir=RUN_OUTPUT_PATH, model_name="geoprior",
)

# For inference (compile=False is fine)
custom_objects_load = {
    
    "GeoPriorSubsNet": GeoPriorSubsNet,
    "LearnableMV": LearnableMV,
    "LearnableKappa": LearnableKappa,
    "FixedGammaW": FixedGammaW,
    "FixedHRef": FixedHRef,
    "make_weighted_pinball": make_weighted_pinball,
}
try:
    with custom_object_scope(custom_objects_load):
        subs_model_loaded = load_model(ckpt_path, compile=False)
    print("Loaded best model for inference (compile=False).")
except Exception as e:
    print(f"[Warn] Could not load best checkpoint ({e}); using in-memory model.")
    subs_model_loaded = subs_model_inst

# =============================================================================
# Calibrate on validation set (BEFORE formatting)
# =============================================================================
print("\nFitting interval calibrator (target 80%) on validation set...")
cal80 = fit_interval_calibrator_on_val(subs_model_inst, val_dataset, target=0.80)
np.save(os.path.join(RUN_OUTPUT_PATH, "interval_factors_80.npy"), cal80.factors_)
print("Calibrator saved.")

# =============================================================================
# Forecasting (Test NPZ if available, otherwise validation fallback)
# =============================================================================
forecast_df = None
dataset_name_for_forecast = "ValidationSet_Fallback"
X_fore = None
y_fore = None

if X_test is not None and y_test is not None:
    X_fore = X_test
    y_fore = y_test
    dataset_name_for_forecast = "TestSet"
else:
    X_fore = X_val
    y_fore = y_val

X_fore = _ensure_input_shapes(X_fore)
y_fore_fmt = _map_targets_for_training(y_fore)

print(f"\nPredicting on {dataset_name_for_forecast}...")
pred_dict = subs_model_loaded.predict(X_fore, verbose=0)  # {'data_final': ...}

data_final = pred_dict["data_final"]
s_dim = subs_model_loaded.output_subsidence_dim

# Split into subsidence and GWL predictions
if QUANTILES:
    # (B, H, Q, O_total)
    s_pred_q_raw = data_final[..., :s_dim]   # (B,H,Q,O_s)
    h_pred_q_raw = data_final[..., s_dim:]   # (B,H,Q,O_g)
    # ---- CALIBRATION: apply on SUBSIDENCE quantiles first ----
    s_pred_q_cal = apply_calibrator_to_subs(cal80, s_pred_q_raw)
    # Prepared for formatter
    predictions_for_formatter = {
        "subs_pred": s_pred_q_cal,
        "gwl_pred": h_pred_q_raw,
    }
else:
    # (B, H, O_total) point forecasts
    s_pred_raw = data_final[..., :s_dim]
    h_pred_raw = data_final[..., s_dim:]
    predictions_for_formatter = {
        "subs_pred": s_pred_raw,
        "gwl_pred": h_pred_raw,
    }

# Format to DataFrame (inverse scaling, coords, coverage eval, save CSV)
target_mapping = {"subs_pred": SUBSIDENCE_COL, "gwl_pred": GWL_COL}
output_dims = {"subs_pred": OUT_S_DIM, "gwl_pred": OUT_G_DIM}

y_true_for_format = {
    "subsidence": y_fore_fmt["subs_pred"],
    "gwl": y_fore_fmt["gwl_pred"],
}

csv_name = (
    f"{CITY_NAME}_{MODEL_NAME}_forecast_{dataset_name_for_forecast}"
    f"_H{FORECAST_HORIZON_YEARS}_calibrated.csv"
)
csv_path = os.path.join(RUN_OUTPUT_PATH, csv_name)

forecast_df = format_pinn_predictions(
    predictions=predictions_for_formatter,
    y_true_dict=y_true_for_format,
    target_mapping=target_mapping,
    scaler_info=scaler_info_dict,
    quantiles=QUANTILES,
    forecast_horizon=FORECAST_HORIZON_YEARS,
    output_dims=output_dims,
    include_coords=True,
    include_gwl=False,  # change to True if you want to include GWL columns
    model_inputs=X_fore,
    evaluate_coverage=True if QUANTILES else False,
    savefile=csv_path,
    coord_scaler=coord_scaler,
    verbose=1,
)
if forecast_df is not None and not forecast_df.empty:
    print(f"Saved calibrated forecast CSV -> {csv_path}")
else:
    print("[Warn] Empty forecast DF.")

# =============================================================================
# Evaluate metrics & physics on the forecasting split (+ optional censoring)
# =============================================================================
eval_results = {}
phys = {}

ds_eval = tf.data.Dataset.from_tensor_slices((X_fore, y_fore_fmt)).batch(BATCH_SIZE)

# --- 2.1 Standard Keras evaluate() + physics metrics ---
try:
    eval_results = subs_model_inst.evaluate(
        ds_eval, return_dict=True, verbose=1
    )
    print("Evaluation:", eval_results)

    # Physics diagnostics are already aggregated in eval_results
    phys_keys = ("epsilon_prior", "epsilon_cons")
    phys = {
        k: float(_to_py(eval_results[k]))
        for k in phys_keys
        if k in eval_results
    }
    if phys:
        print("Physics diagnostics (from evaluate):", phys)

except Exception as e:
    print(f"[Warn] Evaluation failed (metrics + physics): {e}")
    eval_results, phys = {}, {}

#  2.2. Save physic payload 
# collect once and save
physics_payload = subs_model_loaded.export_physics_payload(
    val_dataset,
    max_batches=None,
    save_path=os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_phys_payload_run_val.npz"
    ),
    format="npz",
    overwrite=True,
    metadata={"city": CITY_NAME, "split": "val"},
)

# 2) later: load without re-evaluating the model
# payload2, meta = subs_model_inst.load_physics_payload(
#     os.path.join( RUN_OUTPUT_PATH, "phys_run_val.npz")
# )

def build_censor_mask_from_dynamic(xb, H, dyn_idx, thresh=0.5):
    """
    Returns a boolean mask (B, H, 1) where True means 'censored'.
    Prefers dynamic_features (where the index was computed).
    If dynamic time length != H, take the last H steps.
    If not available, returns all-False mask.
    """
    dyn = xb.get("dynamic_features", None)
    if dyn is not None:
        # defensive: make sure index fits the last dimension
        if dyn.shape[-1] and dyn_idx is not None and dyn_idx < dyn.shape[-1]:
            m_dyn = dyn[..., dyn_idx:dyn_idx+1] > thresh  # (B, T_dyn, 1)
            T_dyn = tf.shape(m_dyn)[1]
            return m_dyn[:, -H:, :] if tf.not_equal(T_dyn, H) else m_dyn

    # fallback: no flag available → no censoring
    B = tf.shape(xb["coords"])[0]
    return tf.zeros((B, H, 1), dtype=tf.bool)

# --- 2.3 Interval diagnostics + optional censor-stratified MAE ---
cov80_uncal = cov80_cal = sharp80_uncal = sharp80_cal = None
censor_metrics = None   # will become a dict if we have a flag

y_true_list, s_q_list, mask_list = [], [], []

for xb, yb in ds_eval:
    out = subs_model_inst(xb, training=False)
    data_final_b = out["data_final"]

    y_true_b = yb["subs_pred"]           # (B, H, 1)
    y_true_list.append(y_true_b)

    if QUANTILES:
        s_q_b, _ = subs_model_inst.split_data_predictions(data_final_b)  # (B, H, Q, 1)
        s_q_list.append(s_q_b)

    if CENSOR_FLAG_IDX is not None:
        H = tf.shape(y_true_b)[1]
        mask_b = build_censor_mask_from_dynamic(xb, H, CENSOR_FLAG_IDX, CENSOR_THRESH)  # (B, H, 1)
        mask_list.append(mask_b)

# # Stack what we collected
y_true = tf.concat(y_true_list, axis=0) if y_true_list else None  # (N,H,1)
s_q = tf.concat(s_q_list, axis=0) if s_q_list else None           # (N,H,Q,1)
mask = tf.concat(mask_list, axis=0) if mask_list else None        # (N,H,1) booleans


# --- 2.3.a Interval coverage/sharpness (uncalibrated vs calibrated) ---
if QUANTILES and (y_true is not None) and (s_q is not None):
    # Uncalibrated
    cov80_uncal   = float(coverage80_fn(y_true, s_q).numpy())
    sharp80_uncal = float(sharpness80_fn(y_true, s_q).numpy())
    # Calibrated (apply same calibrator to the whole tensor)
    s_q_cal = apply_calibrator_to_subs(cal80, s_q)                        # keeps (N, H, 3, 1)
    cov80_cal   = float(coverage80_fn(y_true, s_q_cal).numpy())
    sharp80_cal = float(sharpness80_fn(y_true, s_q_cal).numpy())

# --- 2.3.b Optional censor-stratified MAE on the same loop products ---
# Works for both quantile mode (use median) and point-forecast mode (fallback).
if (y_true is not None) and (mask is not None):
    if QUANTILES and (s_q is not None):
        med_idx = int(np.argmin(np.abs(np.asarray(QUANTILES) - 0.5)))
        s_med = s_q[..., med_idx, :]  # (N, H, 1)
    else:
        # point-forecast: take subsidence head from this pass
        s_pred_list = []
        for xb2, _ in ds_eval:
            out2 = subs_model_inst(xb2, training=False)
            s_pred_list.append(out2["data_final"][..., :1])  # (B, H, 1)
        s_med = tf.concat(s_pred_list, axis=0)

    mask_f = tf.cast(mask, tf.float32)                       # (N, H, 1)
    num_cens = tf.reduce_sum(mask_f) + 1e-8
    num_unc  = tf.reduce_sum(1.0 - mask_f) + 1e-8

    abs_err = tf.abs(y_true - s_med)                         # (N, H, 1)
    mae_cens = tf.reduce_sum(abs_err * mask_f) / num_cens
    mae_unc  = tf.reduce_sum(abs_err * (1.0 - mask_f)) / num_unc

    censor_metrics = {
        "flag_name": CENSOR_FLAG_NAME,
        "threshold": float(CENSOR_THRESH),
        "mae_censored": float(mae_cens.numpy()),
        "mae_uncensored": float(mae_unc.numpy()),
    }
    print(f"[CENSOR] MAE censored={censor_metrics['mae_censored']:.4f} | "
          f"uncensored={censor_metrics['mae_uncensored']:.4f}")

# Save summary JSON
stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
payload = {
    "timestamp": stamp,
    "tf_version": tf.__version__,
    "numpy_version": np.__version__,
    "quantiles": QUANTILES,
    "horizon": FORECAST_HORIZON_YEARS,
    "batch_size": BATCH_SIZE,
    "metrics_evaluate": {k: _to_py(v) for k, v in (eval_results or {}).items()},
    "physics_diagnostics": phys,
}
if QUANTILES:
    payload["interval_calibration"] = {
        "target": 0.80,
        "factors_per_horizon": getattr(cal80, "factors_", None).tolist(
            ) if hasattr(cal80, "factors_") else None,
        "coverage80_uncalibrated": cov80_uncal,
        "coverage80_calibrated": cov80_cal,
        "sharpness80_uncalibrated": sharp80_uncal,
        "sharpness80_calibrated": sharp80_cal,
    }
    
if censor_metrics is not None:
    payload["censor_stratified"] = censor_metrics


json_out = os.path.join(RUN_OUTPUT_PATH, f"geoprior_eval_phys_{stamp}.json")
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"Saved metrics + physics JSON -> {json_out}")
#%
# =============================================================================
# Visualization (optional)
# =============================================================================
print("\nPlotting forecast views...")
if forecast_df is not None and not forecast_df.empty:
    try:
        # quick spatial snapshots (subsidence only)
        horizon_steps = [1, FORECAST_HORIZON_YEARS] if FORECAST_HORIZON_YEARS > 1 else [1]
        plot_forecasts(
            forecast_df=forecast_df,
            target_name=SUBSIDENCE_COL,
            quantiles=QUANTILES,
            output_dim=OUT_S_DIM,
            kind="spatial",
            horizon_steps=horizon_steps,
            spatial_cols=("coord_x", "coord_y"),
            sample_ids="first_n",
            num_samples=min(3, BATCH_SIZE),
            max_cols=2,
            figsize=(7, 5.5),
            cbar="uniform",
            verbose=1,
            savefig = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_eval_plot"),
            show=False, 
        )
    except Exception as e:
        print(f"[Warn] plot_forecasts failed: {e}")

    try:
        forecast_view(
            forecast_df,
            spatial_cols=("coord_x", "coord_y"),
            time_col="coord_t",
            value_prefixes=[SUBSIDENCE_COL],
            verbose=1,
            view_quantiles=[0.5],
            savefig=os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_forecast_comparison_plot_"),
            save_fmts=[".png", ".pdf"],
        )
        print(f"Saved forecast view figures in: {RUN_OUTPUT_PATH}")
    except Exception as e:
        print(f"[Warn] forecast_view failed: {e}")

try:
    save_all_figures(
        output_dir=RUN_OUTPUT_PATH,
        prefix=f"{CITY_NAME}_{MODEL_NAME}_plot_",
        fmts=[".png", ".pdf"],
    )
    print(f"Saved all open Matplotlib figures in: {RUN_OUTPUT_PATH}")
except Exception as e:
    print(f"[Warn] save_all_figures failed: {e}")

print(f"\n---- {CITY_NAME.upper()} {MODEL_NAME} TRAINING COMPLETE ----\n"
      f"Artifacts -> {RUN_OUTPUT_PATH}\n")


