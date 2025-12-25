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
import sys 
import json
import joblib
import numpy as np
import pandas as pd 
import datetime as dt
import warnings
import tensorflow as tf
import  platform

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
# tf.debugging.enable_check_numerics()

# Silence common warnings and TF logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
if hasattr(tf, "autograph") and hasattr(tf.autograph, "set_verbosity"):
    tf.autograph.set_verbosity(0)

from fusionlab._optdeps import with_progress 
from fusionlab.backends.devices import configure_tf_from_cfg
from fusionlab.api.util import get_table_size
from fusionlab.utils.generic_utils import ensure_directory_exists, save_all_figures
from fusionlab.utils.generic_utils import default_results_dir, getenv_stripped
from fusionlab.utils.generic_utils import print_config_table
from fusionlab.registry.utils import _find_stage1_manifest
from fusionlab.utils.nat_utils import (
    load_nat_config, 
    load_nat_config_payload, 
    ensure_input_shapes,
    map_targets_for_training,
    make_tf_dataset,
    load_scaler_info,
    save_ablation_record,
    best_epoch_and_metrics,
    build_censor_mask, 
    name_of,
    serialize_subs_params,
    resolve_si_affine
)

from fusionlab.utils.forecast_utils import format_and_forecast
from fusionlab.utils.scale_metrics import (
    inverse_scale_target,
    point_metrics,
    per_horizon_metrics,
)
from fusionlab.utils.spatial_utils import deg_to_m_from_lat
from fusionlab.nn.pinn.models import GeoPriorSubsNet, PoroElasticSubsNet
from fusionlab.params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
from fusionlab.nn.losses import make_weighted_pinball
from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn, _to_py
from fusionlab.nn.calibration import (
    fit_interval_calibrator_on_val,
    apply_calibrator_to_subs,
)
from fusionlab.nn.utils import plot_history_in
from fusionlab.nn.pinn.op import extract_physical_parameters
from fusionlab.plot.forecast import plot_eval_future
from fusionlab.nn.callbacks import LambdaOffsetScheduler

# XXX NEW 
# =============================================================================
# Config / Paths
# =============================================================================

RESULTS_DIR = default_results_dir()

# Desired city/model from NATCOM config payload
cfg_payload   = load_nat_config_payload()
CFG_CITY  = (cfg_payload.get("city") or "").strip().lower() or None
CFG_MODEL = cfg_payload.get("model") or "GeoPriorSubsNet"

# Optional advanced overrides from env
CITY_ENV   = getenv_stripped("CITY")
MODEL_ENV  = getenv_stripped("MODEL_NAME_OVERRIDE")

CITY_HINT  = CITY_ENV or CFG_CITY
# MODEL_HINT = MODEL_ENV or CFG_MODEL
MANUAL     = getenv_stripped("STAGE1_MANIFEST")

# IMPORTANT: do not filter Stage-1 manifests by model, so we can
# reuse the same Stage-1 run for multiple model flavours.
MODEL_HINT = None

def _is_valid_stage1(m: dict, path: str) -> bool:
    if str(m.get("stage", "")).strip().lower() != "stage1":
        return False
    art = (m.get("artifacts") or {})
    npz = (art.get("numpy") or {})
    need = (
        "train_inputs_npz", "train_targets_npz",
        "val_inputs_npz", "val_targets_npz"
    )
    return all(k in npz and isinstance(
        npz[k], str) and npz[k] for k in need)

MANIFEST_PATH = _find_stage1_manifest(
    manual=MANUAL,
    base_dir=RESULTS_DIR,
    city_hint=CITY_HINT, # e.g., "zhongshan"; None means "no filter"
    model_hint=None,
    prefer="timestamp",
    required_keys=("model", "stage", "artifacts", "config", "paths"),
    filter_fn=_is_valid_stage1,
    verbose=1,
)

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    M = json.load(f)

stage = (M.get("stage") or "").strip().lower()
if stage not in ("stage1", "stage-1", "stage_1"):
    raise RuntimeError(
        "Expected a Stage-1 manifest, got"
        f" stage={stage!r} at {MANIFEST_PATH}"
    )

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
        "   $ python nat.com/training_NATCOM_GEOPRIOR.py\n"
    )
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

def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

cfg = deep_update(dict(cfg_global), cfg_manifest)  # manifest wins, but nested keys preserved


# cfg = dict(cfg_global)
# cfg.update(cfg_manifest)  # manifest wins on overlapping keys
device_info = configure_tf_from_cfg(cfg)

CITY_NAME  = M.get("city",  cfg.get("CITY_NAME", "nansha"))
# MODEL_NAME = M.get("model", cfg.get("MODEL_NAME", "GeoPriorSubsNet"))
MODEL_NAME = MODEL_ENV or cfg.get("MODEL_NAME", "GeoPriorSubsNet")
# ----- Optional censoring config (from merged cfg) -----------------------
FEATURES   = cfg.get("features", {}) or {}
DYN_NAMES  = FEATURES.get("dynamic", []) or []
FUT_NAMES  = FEATURES.get("future",  []) or []   

CENSOR     = cfg.get("censoring", {}) or cfg.get("censor", {}) or {}
CENSOR_SPECS  = CENSOR.get("specs", []) or []
CENSOR_THRESH = float(CENSOR.get("flag_threshold", 0.5))


CENSOR_FLAG_IDX_DYN  = None
CENSOR_FLAG_IDX_FUT  = None
CENSOR_FLAG_NAME     = None

for sp in CENSOR_SPECS:
    cand = sp.get("flag_col")
    if not cand:
        base = sp.get("col")
        if base:
            cand = base + sp.get("flag_suffix", "_censored")

    if cand:
        if cand in FUT_NAMES and CENSOR_FLAG_IDX_FUT is None:
            CENSOR_FLAG_IDX_FUT = FUT_NAMES.index(cand)
            CENSOR_FLAG_NAME = cand
        if cand in DYN_NAMES and CENSOR_FLAG_IDX_DYN is None:
            CENSOR_FLAG_IDX_DYN = DYN_NAMES.index(cand)
            CENSOR_FLAG_NAME = cand

# prefer FUT if available, else DYN
if CENSOR_FLAG_IDX_FUT is not None:
    CENSOR_MASK_SOURCE = "future"
    CENSOR_FLAG_IDX = CENSOR_FLAG_IDX_FUT
elif CENSOR_FLAG_IDX_DYN is not None:
    CENSOR_MASK_SOURCE = "dynamic"
    CENSOR_FLAG_IDX = CENSOR_FLAG_IDX_DYN
    print("[Info] Censor flags present in dynamic features:", CENSOR_FLAG_NAME)

else:
    CENSOR_MASK_SOURCE = None
    CENSOR_FLAG_IDX = None

print("[Info] Censor mask source:", CENSOR_MASK_SOURCE,
      "| flag:", CENSOR_FLAG_NAME, "| idx:", CENSOR_FLAG_IDX)


# ---- Model / physics / training config ---------------------------------
# NOTE: TIME_STEPS / FORECAST_HORIZON_YEARS / MODE are taken from cfg
# AFTER merging, so they reflect what Stage-1 actually used.
TIME_STEPS            = cfg["TIME_STEPS"]
FORECAST_HORIZON_YEARS = cfg["FORECAST_HORIZON_YEARS"]
FORECAST_START_YEAR = cfg['FORECAST_START_YEAR']
MODE                  = cfg["MODE"]

ATTENTION_LEVELS      = cfg.get("ATTENTION_LEVELS", ["cross", "hierarchical", "memory"])
SCALE_PDE_RESIDUALS   = cfg.get("SCALE_PDE_RESIDUALS", True)
CONSOLIDATION_STEP_RESIDUAL_METHOD =cfg.get(
    "CONSOLIDATION_STEP_RESIDUAL_METHOD", "exact"
)

ALLOW_SUBS_RESIDUAL =cfg.get(
    "allow_subs_residual", cfg.get(
        "ALLOW_SUBS_RESIDUAL", True
        )
)

EMBED_DIM    = cfg.get("EMBED_DIM", 32)
HIDDEN_UNITS = cfg.get("HIDDEN_UNITS", 64)
LSTM_UNITS   = cfg.get("LSTM_UNITS", 64)
ATTENTION_UNITS = cfg.get("ATTENTION_UNITS", 64)
NUMBER_HEADS    = cfg.get("NUMBER_HEADS", 2)
DROPOUT_RATE    = cfg.get("DROPOUT_RATE", 0.10)

MEMORY_SIZE    = cfg.get("MEMORY_SIZE", 50)
SCALES         = cfg.get("SCALES", [1, 2])
USE_RESIDUALS  = cfg.get("USE_RESIDUALS", True)
USE_BATCH_NORM = cfg.get("USE_BATCH_NORM", False)   
USE_VSN        = cfg.get("USE_VSN", True)           
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
if PDE_MODE_CONFIG in ("off", "none"):
    PDE_MODE_CONFIG = "none"

LAMBDA_CONS   = cfg.get("LAMBDA_CONS",   0.10)
LAMBDA_GW     = cfg.get("LAMBDA_GW",     0.01)
LAMBDA_PRIOR  = cfg.get("LAMBDA_PRIOR",  0.10)
LAMBDA_SMOOTH = cfg.get("LAMBDA_SMOOTH", 0.01)
LAMBDA_MV     = cfg.get("LAMBDA_MV",     0.01)
MV_LR_MULT    = cfg.get("MV_LR_MULT",    1.0)
KAPPA_LR_MULT = cfg.get("KAPPA_LR_MULT", 5.0)
OFFSET_MODE = cfg.get("OFFSET_MODE", "mul")
LAMBDA_OFFSET = float(cfg.get("LAMBDA_OFFSET", 1.0))

USE_LAMBDA_OFFSET_SCHEDULER = bool(cfg.get("USE_LAMBDA_OFFSET_SCHEDULER", False))
LAMBDA_OFFSET_UNIT = cfg.get("LAMBDA_OFFSET_UNIT", "epoch")
LAMBDA_OFFSET_WHEN = cfg.get("LAMBDA_OFFSET_WHEN", "begin")
LAMBDA_OFFSET_WARMUP = int(cfg.get("LAMBDA_OFFSET_WARMUP", 10))

LAMBDA_OFFSET_START = cfg.get("LAMBDA_OFFSET_START", None)
LAMBDA_OFFSET_END = cfg.get("LAMBDA_OFFSET_END", None)
LAMBDA_OFFSET_SCHEDULE = cfg.get("LAMBDA_OFFSET_SCHEDULE", None)

LAMBDA_BOUNDS = cfg.get("LAMBDA_BOUNDS", 0.0)
# Global physics bounds (from config.py)
PHYSICS_BOUNDS_CFG = cfg.get("PHYSICS_BOUNDS", {}) or {}
PHYSICS_BOUNDS_MODE = (cfg.get("PHYSICS_BOUNDS_MODE", "soft") or "soft").strip().lower()
# Time units for physics (controls d/dt scaling inside PINN residuals)
TIME_UNITS = (cfg.get("TIME_UNITS", "year") or "year").strip().lower()

# Prefer Stage-1 provenance (most trustworthy), fallback to config default.
units_prov = cfg.get("units_provenance", {}) or {}
SUBS_UNIT_TO_SI_APPLIED = float(
    units_prov.get("subs_unit_to_si_applied_stage1", cfg.get("SUBS_UNIT_TO_SI", 1e-3))
)

_default_phys_bounds = {
    "H_min": 5.0,
    "H_max": 80.0,
    "K_min": 1e-8,
    "K_max": 1e-3,
    "Ss_min": 1e-7,
    "Ss_max": 1e-3,
}

phys_bounds = dict(_default_phys_bounds)
phys_bounds.update(PHYSICS_BOUNDS_CFG)

# Convert to the form expected by GeoPriorSubsNet / default_scales(...)
bounds_for_scaling = {
    # thickness
    "H_min": float(phys_bounds["H_min"]),
    "H_max": float(phys_bounds["H_max"]),

    # ALSO keep linear (canonical from config.py)
    "K_min":  float(phys_bounds["K_min"]),
    "K_max":  float(phys_bounds["K_max"]),
    "Ss_min": float(phys_bounds["Ss_min"]),
    "Ss_max": float(phys_bounds["Ss_max"]),

    # convenience log-space
    "logK_min":  float(np.log(phys_bounds["K_min"])),
    "logK_max":  float(np.log(phys_bounds["K_max"])),
    "logSs_min": float(np.log(phys_bounds["Ss_min"])),
    "logSs_max": float(np.log(phys_bounds["Ss_max"])),
}

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

# ------------------------------------------------------------------
# Flavour-dependent physics tweaks
# ------------------------------------------------------------------
if MODEL_NAME == "HybridAttn-NoPhysics":
    # Full data-only baseline: disable physics entirely.
    PDE_MODE_CONFIG = "off"
    LAMBDA_CONS   = 0.0
    LAMBDA_GW     = 0.0
    LAMBDA_PRIOR  = 0.0
    LAMBDA_SMOOTH = 0.0
    LAMBDA_BOUNDS = 0.0
    LAMBDA_MV     = 0.0
    MV_LR_MULT    = 0.0
    KAPPA_LR_MULT = 0.0

elif MODEL_NAME == "PoroElasticSubsNet":
    # Poroelastic surrogate: consolidation only, no GW-flow equation.
    PDE_MODE_CONFIG = "consolidation"
    LAMBDA_GW       = 0.0

# For "GeoPriorSubsNet" we keep whatever is in config.py
# (typically PDE_MODE_CONFIG="both").

# Targets & columns (for formatter)
# cols_cfg = cfg.get("cols", {})
# SUBSIDENCE_COL = cols_cfg.get("subsidence", "subsidence")
# GWL_COL        = cols_cfg.get("gwl", "GWL")

cols_cfg = cfg.get("cols", {}) or {}

SUBS_MODEL_COL = cols_cfg.get("subs_model", cols_cfg.get("subsidence", "subsidence"))
GWL_MODEL_COL  = cols_cfg.get("gwl_model",  cols_cfg.get("gwl", "GWL"))
H_MODEL_COL    = cols_cfg.get("h_field_model", cols_cfg.get("h_field", "soil_thickness"))

# Keep requested names only for logging/UI if you want
# SUBS_REQUESTED_COL = cols_cfg.get("subsidence", "subsidence")
# GWL_REQUESTED_COL  = cols_cfg.get("gwl", "GWL")
SUBSIDENCE_COL = cols_cfg.get("subsidence", "subsidence")
GWL_COL        = cols_cfg.get("gwl", "GWL")
# -------------------------------------------------------------------------
# Resolve which *dynamic feature channel* corresponds to the GWL driver
# (depth-to-water or head proxy) and store its index for the model.
# -------------------------------------------------------------------------
sk_stage1 = cfg.get("scaling_kwargs", {}) or {}

# Prefer Stage-1 exported channel index (most robust)
if "gwl_dyn_index" in sk_stage1 and sk_stage1["gwl_dyn_index"] is not None:
    GWL_DYN_INDEX = int(sk_stage1["gwl_dyn_index"])
    if not (0 <= GWL_DYN_INDEX < len(DYN_NAMES)):
        raise RuntimeError(
            f"Stage-1 gwl_dyn_index={GWL_DYN_INDEX} out of bounds for "
            f"len(DYN_NAMES)={len(DYN_NAMES)}"
        )
    gwl_dyn_name = DYN_NAMES[GWL_DYN_INDEX]
else:

    # Prefer an explicit name if Stage-1 recorded it; else fall back to cols["gwl"]
    gwl_dyn_name = (
        sk_stage1.get("gwl_dyn_name")
        or sk_stage1.get("gwl_col")     # backward compat with earlier naming
        or GWL_COL
    )
    
    # If cols["gwl"] is a target name but the dynamic driver is e.g. "z_GWL",
    # try common alternatives.
    if gwl_dyn_name not in DYN_NAMES:
        for cand in (GWL_COL, "z_GWL", "Z_GWL", "gwl", "GWL", "depth_to_water"):
            if cand in DYN_NAMES:
                gwl_dyn_name = cand
                break
    
    if gwl_dyn_name not in DYN_NAMES:
        raise RuntimeError(
            "Cannot find the GWL driver column inside FEATURES['dynamic'].\n"
            f"  Requested/derived gwl_dyn_name: {gwl_dyn_name!r}\n"
            f"  Available dynamic features: {DYN_NAMES}\n"
            "Fix: ensure Stage-1 exports the GWL driver in dynamic_features, or set\n"
            "     cfg['scaling_kwargs']['gwl_dyn_name'] to the correct dynamic feature."
        )
    
    GWL_DYN_INDEX = int(DYN_NAMES.index(gwl_dyn_name))

print(f"[Info] GWL dynamic channel: name={gwl_dyn_name} | index={GWL_DYN_INDEX}")

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
        "TIME_UNITS": TIME_UNITS,
        "LAMBDA_CONS": LAMBDA_CONS,
        "LAMBDA_GW": LAMBDA_GW,
        "LAMBDA_PRIOR": LAMBDA_PRIOR,
        "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
        "LAMBDA_BOUNDS": LAMBDA_BOUNDS, 
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
        "PHYSICS_BOUNDS": phys_bounds,
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
#%

#-----------------------------------------------------------------------------

encoders = M["artifacts"]["encoders"]

scaled_cols = set(encoders.get("scaled_ml_numeric_cols") or [])

def _needs_inverse_affine(col: str) -> bool:
    return bool(col) and (col in scaled_cols)

def _warn_if_identity(col, a, b):
    if (a is not None and b is not None) and (float(a) == 1.0 and float(b) == 0.0):
        print(f"[Warn] {col}: manifest provides identity SI affine; "
              "will override if this column is scaled by main_scaler.")


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
        
# 
# -------------------------------------------------------------------------
# Stage-1 scaling metadata (single source of truth for physics chain-rule)
# -------------------------------------------------------------------------
sk = (cfg.get("scaling_kwargs") or {})

# -------------------------------------------------------------------------
# GWL semantics (depth vs head) from config / manifest
# -------------------------------------------------------------------------
GWL_KIND = str(sk.get("gwl_kind", cfg.get("GWL_KIND", "depth_bgs"))).lower()
GWL_SIGN = str(sk.get("gwl_sign", cfg.get("GWL_SIGN", "down_positive"))).lower()
USE_HEAD_PROXY = bool(sk.get("use_head_proxy", cfg.get("USE_HEAD_PROXY", True)))
Z_SURF_COL = sk.get("z_surf_col", cfg.get("Z_SURF_COL", None))

# -------------------------------------------------------------------------
# Canonical GWL truth table (single source for Stage-2 + physics code)
# -------------------------------------------------------------------------
conv = cfg.get("conventions", {}) or {}

# Promote manifest-level conventions into scaling_kwargs (sk) so downstream
# code only consults *one* dict.
for _k in (
    "gwl_kind",
    "gwl_sign",
    "gwl_driver_kind",
    "gwl_driver_sign",
    "gwl_target_kind",
    "gwl_target_sign",
    "use_head_proxy",
    "time_units",
):
    if _k not in sk and _k in conv:
        sk[_k] = conv[_k]

cols_spec = (cfg.get("cols", {}) or {})
z_surf_any = (
    Z_SURF_COL
    or cols_spec.get("z_surf_static")
    or cols_spec.get("z_surf_raw")
)

head_from_depth_rule = None
if z_surf_any:
    head_from_depth_rule = "z_surf - depth"
elif USE_HEAD_PROXY:
    head_from_depth_rule = "-depth (proxy)"

gwl_z_meta = {
    # declared/raw semantics (what user/config *meant*)
    "raw_kind": str(conv.get("gwl_kind", GWL_KIND)).lower(),
    "raw_sign": str(conv.get("gwl_sign", GWL_SIGN)).lower(),

    # Stage-1 resolved roles (what Stage-1 *produced*)
    "driver_kind": str(conv.get("gwl_driver_kind", "depth")).lower(),
    "driver_sign": str(conv.get("gwl_driver_sign", "down_positive")).lower(),
    "target_kind": str(conv.get("gwl_target_kind", "head")).lower(),
    "target_sign": str(conv.get("gwl_target_sign", "up_positive")).lower(),

    "use_head_proxy": bool(USE_HEAD_PROXY),
    "z_surf_col": z_surf_any,
    "head_from_depth_rule": head_from_depth_rule,

    # Column provenance (for audit/debug; physics should still use indices)
    "cols": {
        "depth_raw": cols_spec.get("depth_raw"),
        "head_raw": cols_spec.get("head_raw"),
        "z_surf_raw": cols_spec.get("z_surf_raw"),
        "depth_model": cols_spec.get("depth_model"),
        "head_model": cols_spec.get("head_model"),
        "z_surf_static": cols_spec.get("z_surf_static"),
    },
}

# Store back into scaling_kwargs so GeoPriorSubsNet only consults sk.
sk["gwl_z_meta"] = gwl_z_meta

print("[Info] GWL semantics:",
      "GWL_KIND=", GWL_KIND,
      "| GWL_SIGN=", GWL_SIGN,
      "| USE_HEAD_PROXY=", USE_HEAD_PROXY,
      "| Z_SURF_COL=", Z_SURF_COL)

# coords
# Stage-1 should be the source of truth
coords_normalized = bool(
    sk.get("coords_normalized", sk.get("normalize_coords", False))  # backward compat
)
coord_ranges = sk.get("coord_ranges") or None

# Infer ONLY if Stage-1 says normalized but didn’t record ranges
if coords_normalized and (not coord_ranges) and (coord_scaler is not None):
    if hasattr(coord_scaler, "data_min_") and hasattr(coord_scaler, "data_max_"):
        span = coord_scaler.data_max_ - coord_scaler.data_min_
        coord_ranges = {
            "t": float(span[0]),
            "x": float(span[1]), 
            "y": float(span[2])
        }
    elif hasattr(coord_scaler, "scale_"):
        sc = coord_scaler.scale_
        coord_ranges = {
            "t": float(sc[0]),
            "x": float(sc[1]),
            "y": float(sc[2])
        }

if coords_normalized and not coord_ranges:
    raise RuntimeError(
        "coords_normalized=True but coord_ranges missing"
        " and cannot infer from coord_scaler."
    )

coords_in_degrees = bool(sk.get("coords_in_degrees", False))
deg_to_m_lon = sk.get("deg_to_m_lon", None)
deg_to_m_lat = sk.get("deg_to_m_lat", None)
coord_order = sk.get("coord_order", ["t", "x", "y"])

# Robust fallback: if coords are degrees, we must know meters-per-degree
# for chain-rule rescaling. Stage-1 should provide these, but we can
# recover them from the Stage-1 scaled CSV if missing.
if coords_in_degrees and (deg_to_m_lon is None or deg_to_m_lat is None):
    lat_ref_deg = sk.get("lat_ref_deg", None)

    if lat_ref_deg is None or (
        isinstance(lat_ref_deg, str) and lat_ref_deg.strip().lower() == "auto"
    ):
        scaled_csv_path = (
            M.get("artifacts", {})
            .get("csv", {})
            .get("scaled", None)
        )
        lat_col = (cfg.get("cols", {}) or {}).get("lat", None)
        if scaled_csv_path and lat_col:
            try:
                _lat = pd.read_csv(
                    scaled_csv_path,
                    usecols=[lat_col],
                )[lat_col].to_numpy(dtype=float)
                lat_ref_deg = float(np.nanmean(_lat))
                print(
                    f"[Coords] Recovered lat_ref_deg={lat_ref_deg:.6f} "
                    f"from Stage-1 scaled CSV ({scaled_csv_path})."
                )
            except Exception:
                lat_ref_deg = None

    if lat_ref_deg is None or not np.isfinite(float(lat_ref_deg)):
        raise RuntimeError(
            "coords_in_degrees=True but deg_to_m_lon/deg_to_m_lat missing "
            "and could not infer a finite lat_ref_deg."
        )

    lat_ref_deg = float(lat_ref_deg)

    try:
          # type: ignore
        deg_to_m_lon, deg_to_m_lat = deg_to_m_from_lat(lat_ref_deg)
    except Exception:
        lat_rad = np.deg2rad(lat_ref_deg)
        deg_to_m_lat = (
            111132.92
            - 559.82 * np.cos(2.0 * lat_rad)
            + 1.175 * np.cos(4.0 * lat_rad)
            - 0.0023 * np.cos(6.0 * lat_rad)
        )
        deg_to_m_lon = (
            111412.84 * np.cos(lat_rad)
            - 93.5 * np.cos(3.0 * lat_rad)
            + 0.118 * np.cos(5.0 * lat_rad)
        )

    sk.update(
        {
            "lat_ref_deg": float(lat_ref_deg),
            "deg_to_m_lon": float(deg_to_m_lon),
            "deg_to_m_lat": float(deg_to_m_lat),
        }
    )

# thickness SI affine (model-space -> meters)
H_scale_si = sk.get("H_scale_si", None)
H_bias_si  = sk.get("H_bias_si",  None)

print("[Info] coords_normalized:", coords_normalized,
      "coord_ranges:", coord_ranges
     )
print("[Info] coords_in_degrees:", coords_in_degrees,
      "deg_to_m_lon:", deg_to_m_lon, "deg_to_m_lat:", deg_to_m_lat)
print("[Info] H_scale_si:", H_scale_si, "H_bias_si:", H_bias_si)


# ---END ADDED 

# Load scaler_info mapping (dict or path)
scaler_info_dict = load_scaler_info(encoders)

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

# Assert tensor/name consistency once

if "dynamic_features" in X_train and DYN_NAMES:
    f_dyn = X_train["dynamic_features"].shape[-1]
    if f_dyn != len(DYN_NAMES):
        raise RuntimeError(
            "Mismatch: NPZ dynamic_features last-dim != len(FEATURES['dynamic']).\n"
            f"  NPZ dynamic_features dim: {f_dyn}\n"
            f"  FEATURES['dynamic'] len : {len(DYN_NAMES)}\n"
            "This means Stage-1 feature list and exported NPZ are out of sync."
        )

if "future_features" in X_train and FUT_NAMES:
    f_fut = X_train["future_features"].shape[-1]
    if f_fut != len(FUT_NAMES):
        raise RuntimeError(
            "Mismatch: NPZ future_features last-dim != len(FEATURES['future']).\n"
            f"  NPZ future_features dim: {f_fut}\n"
            f"  FEATURES['future'] len : {len(FUT_NAMES)}\n"
            "This means Stage-1 feature list and exported NPZ are out of sync."
        )

# =============================================================================
# Build datasets
# =============================================================================

train_dataset = make_tf_dataset(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    mode=MODE,
    forecast_horizon=FORECAST_HORIZON_YEARS,
)
val_dataset = make_tf_dataset(
    X_val, y_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    mode=MODE,
    forecast_horizon=FORECAST_HORIZON_YEARS,
)


print("\nDataset sample shapes:")
for xb, yb in train_dataset.take(1):
    for k, v in xb.items():
        print(f"  X[{k:>16}] -> {tuple(v.shape)}")
    for k, v in yb.items():
        print(f"  y[{k:>16}] -> {tuple(v.shape)}")

# =============================================================================
# Build & compile model
# =============================================================================

X_train_norm = ensure_input_shapes(
    X_train,
    mode=MODE,
    forecast_horizon=FORECAST_HORIZON_YEARS,
)
s_dim_model = X_train_norm["static_features"].shape[-1]
d_dim_model = X_train_norm["dynamic_features"].shape[-1]
f_dim_model = X_train_norm["future_features"].shape[-1]

MODEL_CLASS_REGISTRY = {
    "GeoPriorSubsNet": GeoPriorSubsNet,
    "PoroElasticSubsNet": PoroElasticSubsNet,
    # HybridAttn-NoPhysics reuses the same architecture as GeoPriorSubsNet,
    # but with PDE_MODE_CONFIG="off" and all lambda_* = 0.
    "HybridAttn-NoPhysics": GeoPriorSubsNet,
}

model_cls = MODEL_CLASS_REGISTRY.get(MODEL_NAME, GeoPriorSubsNet)

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
    "scaling_kwargs": {
        # anything else default_scales(...) already expects can
        # also be passed here later
        "bounds": bounds_for_scaling,
        "time_units": TIME_UNITS,   
    },
    "bounds_mode": PHYSICS_BOUNDS_MODE,
    # GeoPrior scalar params
    "mv": LearnableMV(initial_value=GEOPRIOR_INIT_MV),
    "kappa": LearnableKappa(initial_value=GEOPRIOR_INIT_KAPPA),
    "gamma_w": FixedGammaW(value=GEOPRIOR_GAMMA_W),
    "h_ref": FixedHRef(mode=GEOPRIOR_H_REF),
    "kappa_mode": GEOPRIOR_KAPPA_MODE,
    "use_effective_h": GEOPRIOR_USE_EFFECTIVE_H,
    "hd_factor": GEOPRIOR_HD_FACTOR,
    "offset_mode": OFFSET_MODE,
    
    "residual_method": CONSOLIDATION_STEP_RESIDUAL_METHOD, 
    
    # For consistency
    "time_units": TIME_UNITS,
}

subsmodel_params["scaling_kwargs"].update({
    "coords_normalized": coords_normalized,
    "coord_ranges": coord_ranges or {},
    "coord_order": coord_order,

    # lon/lat degrees handling (only used if coords_in_degrees=True)
    "coords_in_degrees": coords_in_degrees,
    "deg_to_m_lon": (float(deg_to_m_lon) if deg_to_m_lon is not None else None),
    "deg_to_m_lat": (float(deg_to_m_lat) if deg_to_m_lat is not None else None),

    # thickness SI affine (used by _to_si_thickness patch)
    "H_scale_si": (float(H_scale_si) if H_scale_si is not None else 1.0),
    "H_bias_si":  (float(H_bias_si)  if H_bias_si  is not None else 0.0),
    
    "allow_subs_residual": ALLOW_SUBS_RESIDUAL, 
    
})

subsmodel_params["scaling_kwargs"].update({
    # names let you sanity-check tensors and debug
    "dynamic_feature_names": list(DYN_NAMES),
    "future_feature_names":  list(FUT_NAMES),

    # the important part: safe slicing instead of hard-coded channel 0
    "gwl_dyn_name":  gwl_dyn_name,
    "gwl_dyn_index": GWL_DYN_INDEX,
})



subs_scale_si = sk.get("subs_scale_si")
subs_bias_si  = sk.get("subs_bias_si")
head_scale_si = sk.get("head_scale_si")
head_bias_si  = sk.get("head_bias_si")

if subs_scale_si is None or subs_bias_si is None:
    subs_scale_si, subs_bias_si = resolve_si_affine(
        cfg, scaler_info_dict,
        target_name=SUBSIDENCE_COL,
        prefix="SUBS",
        unit_factor_key="SUBS_UNIT_TO_SI",
        scale_key="SUBS_SCALE_SI",
        bias_key="SUBS_BIAS_SI",
    )
if head_scale_si is None or head_bias_si is None:
    head_scale_si, head_bias_si = resolve_si_affine(
        cfg, scaler_info_dict,
        target_name=GWL_COL,
        prefix="HEAD",
        unit_factor_key="HEAD_UNIT_TO_SI",
        scale_key="HEAD_SCALE_SI",
        bias_key="HEAD_BIAS_SI",
    )

subsmodel_params["scaling_kwargs"].update({
    "subs_scale_si": subs_scale_si,
    "subs_bias_si": subs_bias_si,
    "head_scale_si": head_scale_si,
    "head_bias_si": head_bias_si,

    # --- semantics for interpreting the GWL variable ---
    "gwl_kind": GWL_KIND,                 # "depth_bgs" or "head"
    "gwl_sign": GWL_SIGN,                 # "down_positive" or "up_positive"
    "use_head_proxy": USE_HEAD_PROXY,     # if no z_surf -> head_proxy = -depth
    "z_surf_col": Z_SURF_COL,             # None or column name if you provide it
    "gwl_z_meta": sk.get("gwl_z_meta", None),  # optional traceability
})

# Optional: drop Nones to keep scaling_kwargs clean
subsmodel_params["scaling_kwargs"] = {
    k: v for k, v in subsmodel_params["scaling_kwargs"].items()
    if v is not None
}

print("=" * 72)
print("SCALES & UNITS (Stage-1 -> Stage-2 SI affine maps)")
print("-" * 72)

def _fmt(v):
    if v is None:
        return "None"
    try:
        return f"{float(v):.6g}"
    except Exception:
        return str(v)

print(f"{'subs_scale_si':<16}: {_fmt(subs_scale_si)}   [m / model_unit]")
print(f"{'subs_bias_si':<16}: {_fmt(subs_bias_si)}   [m]")
print(f"{'head_scale_si':<16}: {_fmt(head_scale_si)}   [m / model_unit]")
print(f"{'head_bias_si':<16}: {_fmt(head_bias_si)}   [m]")
print(f"{'time_units':<16}: {_fmt(TIME_UNITS)}   (e.g., 'years')")

print("-" * 72)
print("SI conversions:  s_si = s_model*subs_scale_si + subs_bias_si ; "
      "h_si = h_model*head_scale_si + head_bias_si")
print("=" * 72)


subs_model_inst = model_cls(
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
    "lambda_bounds": LAMBDA_BOUNDS, 
    "lambda_mv": LAMBDA_MV,
    "mv_lr_mult": MV_LR_MULT,
    # (global multiplier for the physics block)
    "lambda_offset": LAMBDA_OFFSET,
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
#%
ckpt_name = f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_HORIZON_YEARS}.keras"
ckpt_path = os.path.join(RUN_OUTPUT_PATH, ckpt_name)

callbacks = [
    EarlyStopping(
        monitor="val_loss", 
        patience=15, 
        restore_best_weights=True, 
        verbose=1
    ),
    ModelCheckpoint(
        filepath=ckpt_path, 
        monitor="val_loss", 
        save_best_only=True,
        save_weights_only=False, 
        verbose=1
    ),
]

csvlog_path = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_{MODEL_NAME}_train_log.csv")
callbacks.append(CSVLogger(csvlog_path, append=False))
callbacks.append(TerminateOnNaN())

if USE_LAMBDA_OFFSET_SCHEDULER and (not subs_model_inst._physics_off()):
    callbacks.append(
        LambdaOffsetScheduler(
            schedule=LAMBDA_OFFSET_SCHEDULE,
            unit=LAMBDA_OFFSET_UNIT,
            when=LAMBDA_OFFSET_WHEN,
            warmup=LAMBDA_OFFSET_WARMUP,
            start=LAMBDA_OFFSET_START,
            end=LAMBDA_OFFSET_END,
            clamp_positive=True,
            verbose=1,
        )
    )

print("\nTraining...")
history = subs_model_inst.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)
print(f"Best val_loss: {min(history.history.get('val_loss', [np.inf])):.4f}")
#%

# ---- files/paths
weights_path     = os.path.join(
    RUN_OUTPUT_PATH, 
    f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_HORIZON_YEARS}.weights.h5")
arch_json_path   = os.path.join(
    RUN_OUTPUT_PATH, 
    f"{CITY_NAME}_{MODEL_NAME}_architecture.json")
summary_json_path= os.path.join(
    RUN_OUTPUT_PATH, 
    f"{CITY_NAME}_{MODEL_NAME}_training_summary.json")

manifest_path= os.path.join(
    RUN_OUTPUT_PATH,
    f"{CITY_NAME}_{MODEL_NAME}_run_manifest.json")


# ---- 2.1 separate weights (useful for quick reloads / ablations)
#  save weights for rebuilding the model for caution. 
try:
    subs_model_inst.save_weights(weights_path)
    print(f"[OK] Saved HDF5 weights -> {weights_path}")
except Exception as e:
    print(
        f"[Warn] save_weights('{weights_path}') failed: {e}\n"
    )

# ---- 2.3 architecture JSON
try:
    with open(arch_json_path, "w", encoding="utf-8") as f:
        f.write(subs_model_inst.to_json())
except Exception as e:
    print(f"[Warn] to_json failed: {e}")

# ---- 2.4 best-epoch summary
best_epoch, metrics_at_best = best_epoch_and_metrics(history.history)

training_summary = {
    "timestamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
    "city": CITY_NAME,
    "model": MODEL_NAME,
    "horizon": int(FORECAST_HORIZON_YEARS),
    "best_epoch": (int(best_epoch) if best_epoch is not None else None),
    "metrics_at_best": metrics_at_best,       # includes loss/val_* to be tracked
    "final_epoch_metrics": {
        k: float(v[-1]) for k, v in history.history.items() if len(v)},
    "env": {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": device_info, 
    },
    "compile": {
        "optimizer": "Adam",
        "learning_rate": float(LEARNING_RATE),
        "loss_weights": loss_weights_dict,
        "metrics": {k: [name_of(m) for m in v] for k, v in metrics_dict.items()},
        "physics_loss_weights": physics_loss_weights,
        "lambda_offset": LAMBDA_OFFSET,
        
    },
    "hp_init": {
        "quantiles": QUANTILES,
        "subs_weights": SUBS_WEIGHTS,
        "gwl_weights": GWL_WEIGHTS,
        "attention_levels": ATTENTION_LEVELS,
        "pde_mode": PDE_MODE_CONFIG,
        "time_steps": int(TIME_STEPS),
        "use_batch_norm": bool(USE_BATCH_NORM),
        "use_vsn": bool(USE_VSN),
        "vsn_units": int(VSN_UNITS) if VSN_UNITS is not None else None,
        "mode": MODE,
        "model_init_params": serialize_subs_params(subsmodel_params, cfg),
        "offset_mode": OFFSET_MODE,
        "scaling_kwargs": {
            "bounds": bounds_for_scaling,
            "time_units": TIME_UNITS,   
            "coords_normalized": coords_normalized,
            "coord_ranges": coord_ranges or {},
        },
        
    },
    "paths": {
        "run_dir": RUN_OUTPUT_PATH,
        "checkpoint_keras": ckpt_path,
        "weights_h5": weights_path,
        # "saved_model": savedmodel_dir,
        "arch_json": arch_json_path,
        "csv_log": csvlog_path,
    },
}

final_model_path = os.path.join(
    RUN_OUTPUT_PATH,
    f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_HORIZON_YEARS}_final.keras",
)
try: 
    subs_model_inst.save(final_model_path)  # includes optimizer & compile config
    training_summary["paths"]["final_keras"] = final_model_path
    print(f"[OK] Saved Final keras model -> {final_model_path}")
except Exception as e: 
    print(
        f"[Warn] Saved Final keras model ('{final_model_path}') failed: {e}\n"
    )

with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(training_summary, f, indent=2)
#%
# ---- 2.5 a small run manifest that downstream scripts can read
run_manifest = {
    "stage": "stage-2-train",
    "city": CITY_NAME,
    "model": MODEL_NAME,
    "config": {  # keep only lightweight keys you’ll query later
        "TIME_STEPS": TIME_STEPS,
        "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
        "MODE": MODE,
        "ATTENTION_LEVELS": ATTENTION_LEVELS,
        "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
        "QUANTILES": QUANTILES,
        "scaling_kwargs": {
            "bounds": bounds_for_scaling,
            "time_units": TIME_UNITS,   
        },
    },
    "paths": training_summary["paths"],
    "artifacts": {
        "training_summary_json": summary_json_path,
        "train_log_csv": csvlog_path,
    }
}
run_manifest["config"]["scaling_kwargs"].update({
    "subs_scale_si": subs_scale_si,
    "subs_bias_si": subs_bias_si,
    "head_scale_si": head_scale_si,
    "head_bias_si": head_bias_si,
    "H_scale_si": float(H_scale_si) if H_scale_si is not None else None,
    "H_bias_si":  float(H_bias_si)  if H_bias_si  is not None else None,
    "coords_normalized": coords_normalized,
    "coord_ranges": coord_ranges,
    "coords_in_degrees": coords_in_degrees,
    "deg_to_m_lon": deg_to_m_lon,
    "deg_to_m_lat": deg_to_m_lat,
})

run_manifest["config"]["scaling_kwargs"].update({
    "dynamic_feature_names": list(DYN_NAMES),
    "future_feature_names":  list(FUT_NAMES),
    "gwl_dyn_name":  gwl_dyn_name,
    "gwl_dyn_index": int(GWL_DYN_INDEX),
})

run_manifest["config"]["scaling_kwargs"].update({
    "coord_order": coord_order,
    "gwl_kind": GWL_KIND,
    "gwl_sign": GWL_SIGN,
    "use_head_proxy": USE_HEAD_PROXY,
    "z_surf_col": Z_SURF_COL,
})

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(run_manifest, f, indent=2)

print("[OK] Persisted weights, architecture JSON,"
      " CSV log, training summary, and run manifest.")


# 2.6 run to save plots  (use ONLY train keys; val_* is auto-detected)
history_groups = {
    # What  actually optimize
    "Total Loss": ["total_loss"],

    # Decomposition: data + physics (scaled vs raw)
    "Data vs Physics": ["data_loss", "physics_loss_scaled", "physics_loss"],

    # Offset controls (helps debug scheduling)
    "Offset Controls": ["lambda_offset", "physics_mult"],

    # Physics components (raw components, before global offset)
    "Physics Components": [
        "consolidation_loss",
        "gw_flow_loss",
        "prior_loss",
        "smooth_loss",
        "mv_prior_loss",
        "bounds_loss",
    ],

    "Subsidence MAE": ["subs_pred_mae"],
    "GWL MAE": ["gwl_pred_mae"],
}
history_groups.update({
    "Physics Loss (Scaled)": ["physics_loss_scaled", "val_physics_loss_scaled"],
    "Offset & Multiplier": [
        "lambda_offset", "val_lambda_offset",
        "physics_mult", "val_physics_mult",
    ],
})

yscales = {
    "Total Loss": "log",
    "Data vs Physics": "log",
    "Physics Components": "log",
    "Offset Controls": "linear",   # lambda_offset can be negative in log10 mode
    "Subsidence MAE": "linear",
    "GWL MAE": "linear",
}

plot_history_in(
    history.history,
    metrics=history_groups,
    title=f"{MODEL_NAME} Training History",
    yscale_settings=yscales,
    layout="subplots",
    savefig=os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_{MODEL_NAME.lower()}_training_history_plot",
    ),
)


# Extract physical parameters

phys_model_tag = "geoprior"
if MODEL_NAME == "PoroElasticSubsNet":
    phys_model_tag = "poroelastic"
elif MODEL_NAME.startswith("HybridAttn"):
    phys_model_tag = "hybridattn"

extract_physical_parameters(
    subs_model_inst, to_csv=True,
    filename=f"{CITY_NAME}_{MODEL_NAME.lower()}_physical_parameters.csv",
    save_dir=RUN_OUTPUT_PATH, model_name=phys_model_tag,
)

#%
# For inference (compile=False is fine)
custom_objects_load = {
    "GeoPriorSubsNet": GeoPriorSubsNet,
    "PoroElasticSubsNet": PoroElasticSubsNet,
    "LearnableMV": LearnableMV,
    "LearnableKappa": LearnableKappa,
    "FixedGammaW": FixedGammaW,
    "FixedHRef": FixedHRef,
    "make_weighted_pinball": make_weighted_pinball,
}

# inference model (use checkpoint if available, else in-memory)
model_inf = subs_model_inst

try:
    with custom_object_scope(custom_objects_load):
        model_inf = load_model(ckpt_path, compile=False)
    print("Loaded best model for inference (compile=False).")
except Exception as e:
    print(f"[Warn] Could not load best checkpoint ({e}); using in-memory model.")
    model_inf = subs_model_inst

# =============================================================================
# Calibrate on validation set (BEFORE formatting)
# =============================================================================
print("\nFitting interval calibrator (target 80%) on validation set...")
cal80 = fit_interval_calibrator_on_val(model_inf, val_dataset, target=0.80)
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

X_fore = ensure_input_shapes(
    X_fore,
    mode=MODE,
    forecast_horizon=FORECAST_HORIZON_YEARS,
)
y_fore_fmt = map_targets_for_training(y_fore)

print(f"\nPredicting on {dataset_name_for_forecast}...")
pred_dict = model_inf.predict(X_fore, verbose=0)  # {'data_final': ...}

data_final = pred_dict["data_final"]
s_dim = model_inf.output_subsidence_dim

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
#%
# Format to DataFrame (inverse scaling, coords, coverage eval, save CSV)
target_mapping = {"subs_pred": SUBSIDENCE_COL, "gwl_pred": GWL_COL}
output_dims = {"subs_pred": OUT_S_DIM, "gwl_pred": OUT_G_DIM}

y_true_for_format = {
    "subsidence": y_fore_fmt["subs_pred"],
    "gwl": y_fore_fmt["gwl_pred"],
}

csv_eval = os.path.join(
    RUN_OUTPUT_PATH,
    f"{CITY_NAME}_{MODEL_NAME}_forecast_"
    f"{dataset_name_for_forecast}_H"
    f"{FORECAST_HORIZON_YEARS}_calibrated.csv",
)
csv_future = os.path.join(
    RUN_OUTPUT_PATH,
    f"{CITY_NAME}_{MODEL_NAME}_forecast_"
    f"{dataset_name_for_forecast}_H"
    f"{FORECAST_HORIZON_YEARS}_future.csv",
)


# Build future grid in physical units (years)
future_grid = np.arange(
    FORECAST_START_YEAR,
    FORECAST_START_YEAR + FORECAST_HORIZON_YEARS,
    dtype=float,
)

df_eval, df_future = format_and_forecast(
    y_pred=predictions_for_formatter,
    y_true=y_true_for_format,
    coords=X_fore.get("coords", None),
    quantiles=QUANTILES if QUANTILES else None,
    target_name=SUBSIDENCE_COL,             
    scaler_target_name=SUBSIDENCE_COL,       
    output_target_name="subsidence",         
    target_key_pred="subs_pred",
    component_index=0,
    scaler_info=scaler_info_dict,
    coord_scaler=coord_scaler,
    coord_columns=("coord_t", "coord_x", "coord_y"),
    train_end_time=cfg.get("TRAIN_END_YEAR"),
    forecast_start_time=FORECAST_START_YEAR,
    forecast_horizon=FORECAST_HORIZON_YEARS,
    future_time_grid=future_grid,
    eval_forecast_step=None,  # last horizon step
    sample_index_offset=0,
    city_name=CITY_NAME,
    model_name=MODEL_NAME,
    dataset_name=dataset_name_for_forecast,
    csv_eval_path=csv_eval,
    csv_future_path=csv_future,
    time_as_datetime=False,
    time_format=None,
    verbose=1,
    # New evaluation options
    eval_metrics=True,
    metrics_column_map=None,  # or custom map
    metrics_quantile_interval=(0.1, 0.9),
    metrics_per_horizon=True,
    metrics_extra=["pss"],  # uses get_metric
    metrics_extra_kwargs=None,
    metrics_savefile=os.path.join(
        RUN_OUTPUT_PATH, 
        "eval_diagnostics.json"),         # auto name, or give explicit path
    metrics_save_format=".json",
    metrics_time_as_str=True,
    value_mode="cumulative", # set to "rate" to convert back to rate 
    input_value_mode="cumulative",
    
    # --- export in mm directly ---
    output_unit="mm",
    output_unit_from="m",
    output_unit_mode="overwrite",
    output_unit_col="subsidence_unit",
)

if df_eval is not None and not df_eval.empty:
    print(
        "Saved calibrated EVAL forecast CSV -> "
        f"{csv_eval}"
    )
else:
    print("[Warn] Empty eval forecast DF.")

if df_future is not None and not df_future.empty:
    print(
        "Saved calibrated FUTURE forecast CSV -> "
        f"{csv_future}"
    )
else:
    print("[Warn] Empty future forecast DF.")
#%
#
# =============================================================================
# Evaluate metrics & physics on the forecasting split (+ optional censoring)
# =============================================================================
eval_results = {}
phys = {}

# Better: reuse make_tf_dataset so keys/shapes match training exactly
# (instead of tf.data.Dataset.from_tensor_slices)
ds_eval = make_tf_dataset(
    X_fore, y_fore,                 # <--- use ORIGINAL y_fore
    batch_size=BATCH_SIZE,
    shuffle=False,
    mode=MODE,
    forecast_horizon=FORECAST_HORIZON_YEARS,
)

# --- 2.1 Standard Keras evaluate() + physics metrics ---
try:
    eval_results = model_inf.evaluate(ds_eval, return_dict=True, verbose=1)
    print("Evaluation:", eval_results)

    # v3.2: include epsilon_gw if available
    phys_keys = ("epsilon_prior", "epsilon_cons", "epsilon_gw")
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

# --- 2.2 Save physics payload (use the same model & same ds split you evaluated) ---
phys_npz_path = os.path.join(
    RUN_OUTPUT_PATH,
    f"{CITY_NAME}_phys_payload_run_val.npz",
    # f"{CITY_NAME}_phys_payload_{dataset_name_for_forecast.lower()}.npz"
 )

_ = model_inf.export_physics_payload(
    ds_eval,
    max_batches=None,
    save_path=phys_npz_path,
    format="npz",
    overwrite=True,
    metadata={
        "city": CITY_NAME,
        "split": dataset_name_for_forecast,
        "time_units": TIME_UNITS,
        "gwl_kind": GWL_KIND,
        "gwl_sign": GWL_SIGN,
        "use_head_proxy": USE_HEAD_PROXY,
    },
)
print(f"[OK] Saved physics payload -> {phys_npz_path}")

# -------------------------------------------------------------------------
# SM3: log-offset diagnostics (δ_K, δ_Ss, δ_Hd, δ_tau)
# -------------------------------------------------------------------------

# --- 2.3 Interval diagnostics + optional censor-stratified MAE ---
cov80_uncal = cov80_cal = sharp80_uncal = sharp80_cal = None
censor_metrics = None   # will become a dict if we have a flag

y_true_list, s_q_list, mask_list = [], [], []

for xb, yb in with_progress(ds_eval, desc="Interval-Censoring Diagnostics"):
    out = model_inf(xb, training=False)
    data_final_b = out["data_final"]

    y_true_b = yb["subs_pred"]           # (B, H, 1)
    y_true_list.append(y_true_b)

    if QUANTILES:
        s_q_b, _ = model_inf.split_data_predictions(data_final_b)  # (B, H, Q, 1)
        s_q_list.append(s_q_b)

    if CENSOR_FLAG_IDX is not None:
        H = tf.shape(y_true_b)[1]
        mask_b = build_censor_mask(
            xb, H, CENSOR_FLAG_IDX, CENSOR_THRESH,
            source=CENSOR_MASK_SOURCE or "dynamic",
            reduce_time="any",       # best default for thickness censoring
            align="broadcast",       # safe fallback
        )
        mask_list.append(mask_b)

# # Stack what we collected
y_true = tf.concat(y_true_list, axis=0) if y_true_list else None  # (N,H,1)
s_q = tf.concat(s_q_list, axis=0) if s_q_list else None           # (N,H,Q,1)
mask = tf.concat(mask_list, axis=0) if mask_list else None        # (N,H,1) booleans

# --- 2.3.a Interval coverage/sharpness (scaled + physical) ---------------
cov80_uncal_phys = cov80_cal_phys = None
sharp80_uncal_phys = sharp80_cal_phys = None
s_q_cal = None

if QUANTILES and (y_true is not None) and (s_q is not None):
    # ---------- SCALED metrics (as before) ----------
    cov80_uncal   = float(coverage80_fn(y_true, s_q).numpy())
    sharp80_uncal = float(sharpness80_fn(y_true, s_q).numpy())
    # Calibrated (apply same calibrator to the whole tensor)
    s_q_cal = apply_calibrator_to_subs(cal80, s_q)  # (N, H, Q, 1) # or keeps (N, H, 3, 1)
    cov80_cal   = float(coverage80_fn(y_true, s_q_cal).numpy())
    sharp80_cal = float(sharpness80_fn(y_true, s_q_cal).numpy())

    # ---------- PHYSICAL metrics (inverse-scaled) ----------
    # 1) inverse-transform y_true and quantiles to physical units
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

    y_true_phys_tf = tf.convert_to_tensor(y_true_phys_np, dtype=tf.float32)
    s_q_phys_tf    = tf.convert_to_tensor(s_q_phys_np,    dtype=tf.float32)

    cov80_uncal_phys   = float(coverage80_fn(y_true_phys_tf, s_q_phys_tf).numpy())
    sharp80_uncal_phys = float(sharpness80_fn(y_true_phys_tf, s_q_phys_tf).numpy())

    if s_q_cal is not None:
        s_q_cal_phys_np = inverse_scale_target(
            s_q_cal,
            scaler_info=scaler_info_dict,
            target_name=SUBSIDENCE_COL,
        )
        s_q_cal_phys_tf = tf.convert_to_tensor(s_q_cal_phys_np, dtype=tf.float32)

        cov80_cal_phys   = float(coverage80_fn(y_true_phys_tf, s_q_cal_phys_tf).numpy())
        sharp80_cal_phys = float(sharpness80_fn(y_true_phys_tf, s_q_cal_phys_tf).numpy())

# --- 2.3.b Optional censor-stratified MAE on the same loop products ---
# Works for both quantile mode (use median) and point-forecast mode (fallback).
if (y_true is not None) and (mask is not None):
    if QUANTILES and (s_q is not None):
        med_idx = int(np.argmin(np.abs(np.asarray(QUANTILES) - 0.5)))
        s_med = s_q[..., med_idx, :]  # (N, H, 1) scaled
    else:
        # point-forecast: take subsidence head from this pass
        s_pred_list = []
        for xb2, _ in ds_eval:
            out2 = subs_model_inst(xb2, training=False)
            s_pred_list.append(out2["data_final"][..., :OUT_S_DIM])  # (B, H, 1)
        s_med = tf.concat(s_pred_list, axis=0)              # scaled

    # Convert both y_true and s_med to physical units using Stage-1 scaler_info
    y_true_phys_np = inverse_scale_target(
        y_true,
        scaler_info=scaler_info_dict,
        target_name=SUBSIDENCE_COL,
    )
    s_med_phys_np = inverse_scale_target(
        s_med,
        scaler_info=scaler_info_dict,
        target_name=SUBSIDENCE_COL,
    )

    y_true_phys = tf.convert_to_tensor(y_true_phys_np, dtype=tf.float32)
    s_med_phys  = tf.convert_to_tensor(s_med_phys_np, dtype=tf.float32)

    mask_f = tf.cast(mask, tf.float32)                       # (N, H, 1)
    num_cens = tf.reduce_sum(mask_f) + 1e-8
    num_unc  = tf.reduce_sum(1.0 - mask_f) + 1e-8

    abs_err = tf.abs(y_true_phys - s_med_phys)               # physical units
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


# --- 2.4 Point metrics (MAE/MSE/R²), overall + per-horizon -------------

metrics_point = {}
per_h_mae_dict, per_h_r2_dict = None, None

if (y_true is not None):
    if QUANTILES and (s_q is not None):
        # median index
        med_idx = int(np.argmin(np.abs(np.asarray(QUANTILES) - 0.5)))
        s_med_uncal = s_q[..., med_idx, :]           # (N,H,1)

        # Prefer calibrated median if available
        if s_q_cal is not None:
            s_med_cal = s_q_cal[..., med_idx, :]     # (N,H,1)
            metrics_point = point_metrics(
                y_true,
                s_med_cal,
                use_physical=True,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )
            per_h_mae_dict, per_h_r2_dict = per_horizon_metrics(
                y_true,
                s_med_cal,
                use_physical=True,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )
        else:
            metrics_point = point_metrics(
                y_true,
                s_med_uncal,
                use_physical=True,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )
            per_h_mae_dict, per_h_r2_dict = per_horizon_metrics(
                y_true,
                s_med_uncal,
                use_physical=True,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )

    else:
        # point-forecast branch
        if s_med is None:
            s_pred_list = []
            for xb2, _ in with_progress(ds_eval, desc="Point-forecast Diagnostics"):
                out2 = subs_model_inst(xb2, training=False)
                s_pred_list.append(out2["data_final"][..., :OUT_S_DIM])  # (B,H,1)
            s_med = tf.concat(s_pred_list, axis=0)

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

# Normalize coverage/sharpness choices for ablation record (prefer calibrated)
coverage80_for_abl = (
    cov80_cal_phys if (cov80_cal_phys is not None)
    else cov80_cal if (cov80_cal is not None)
    else cov80_uncal_phys if (cov80_uncal_phys is not None)
    else cov80_uncal
)

sharpness80_for_abl = (
    sharp80_cal_phys if (sharp80_cal_phys is not None)
    else sharp80_uncal_phys if (sharp80_uncal_phys is not None)
    else sharp80_cal if (sharp80_cal is not None)
    else sharp80_uncal
)

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
        "factors_per_horizon": getattr(cal80, "factors_", None).tolist()
        if hasattr(cal80, "factors_") else None,

        # scaled-space metrics (backward compatible)
        "coverage80_uncalibrated": cov80_uncal,
        "coverage80_calibrated":   cov80_cal,
        "sharpness80_uncalibrated": sharp80_uncal,
        "sharpness80_calibrated":   sharp80_cal,

        # physical-space metrics (new, recommended for the paper)
        "coverage80_uncalibrated_phys": cov80_uncal_phys,
        "coverage80_calibrated_phys":   cov80_cal_phys,
        "sharpness80_uncalibrated_phys": sharp80_uncal_phys,
        "sharpness80_calibrated_phys":   sharp80_cal_phys,
    }

if censor_metrics is not None:
    payload["censor_stratified"] = censor_metrics

# Attach point metrics & per-horizon into payload
if metrics_point:
    payload["point_metrics"] = {
        "mae": metrics_point.get("mae"),
        "mse": metrics_point.get("mse"),
        "r2":  metrics_point.get("r2"),
    }
if per_h_mae_dict:
    payload.setdefault("per_horizon", {})
    payload["per_horizon"]["mae"] = per_h_mae_dict
if per_h_r2_dict:
    payload.setdefault("per_horizon", {})
    payload["per_horizon"]["r2"] = per_h_r2_dict


json_out = os.path.join(RUN_OUTPUT_PATH, f"geoprior_eval_phys_{stamp}.json")
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"Saved metrics + physics JSON -> {json_out}")
#%
ABLCFG = {
    "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
    "GEOPRIOR_USE_EFFECTIVE_H": GEOPRIOR_USE_EFFECTIVE_H,
    "GEOPRIOR_KAPPA_MODE": GEOPRIOR_KAPPA_MODE,
    "GEOPRIOR_HD_FACTOR": GEOPRIOR_HD_FACTOR,
    "LAMBDA_CONS": LAMBDA_CONS,
    "LAMBDA_GW": LAMBDA_GW,
    "LAMBDA_PRIOR": LAMBDA_PRIOR,
    "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
    "LAMBDA_BOUNDS": LAMBDA_BOUNDS,
    "LAMBDA_MV": LAMBDA_MV,
}

# Prefer MAE/MSE directly from evaluate() if they exist; else use computed.
eval_mae = None
eval_mse = None
if isinstance(eval_results, dict):
    eval_mae = eval_results.get("subs_pred_mae")
    eval_mse = eval_results.get("subs_pred_mse")

save_ablation_record(
    outdir=RUN_OUTPUT_PATH,
    city=CITY_NAME,
    model_name=MODEL_NAME,
    cfg=ABLCFG,
    eval_dict={
        "r2": (metrics_point or {}).get("r2"),
        "mse": float(eval_mse) if eval_mse is not None else (metrics_point or {}).get("mse"),
        "mae": float(eval_mae) if eval_mae is not None else (metrics_point or {}).get("mae"),
        "coverage80": coverage80_for_abl,
        "sharpness80": sharpness80_for_abl,
    },
    phys_diag=(phys or {}),
    per_h_mae=per_h_mae_dict,   # optional but included when available
    per_h_r2=per_h_r2_dict      # optional but included when available
)
print("Ablation record saved.")
#
# =============================================================================
# Visualization (optional)
# =============================================================================

print("\nPlotting forecast views...")
try:
    plot_eval_future(
        df_eval=df_eval,
        df_future=df_future,
        target_name=SUBSIDENCE_COL,
        quantiles=QUANTILES,
        spatial_cols=("coord_x", "coord_y"),
        time_col="coord_t",
        # Eval: show last eval year (e.g. 2022)
        eval_years=[FORECAST_START_YEAR - 1],
        # Future: use the same grid you passed to format_and_forecast
        future_years=future_grid,
        # For eval: compare [actual] vs [q50] only
        eval_view_quantiles=[0.5],
        # For future: show full [q10, q50, q90]
        future_view_quantiles=QUANTILES,
        spatial_mode="hexbin",      # hotspot view
        hexbin_gridsize=40,
        savefig_prefix=os.path.join(
            RUN_OUTPUT_PATH,
            f"{CITY_NAME}_subsidence_view",
        ),
        save_fmts=[".png", ".pdf"],
        show=False,
        verbose=1,
    )
except Exception as e:
    print(f"[Warn] plot_eval_future failed: {e}")

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


