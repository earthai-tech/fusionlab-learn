# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author : LKouadio <etanoyau@gmail.com>
#
# Central configuration for the NATCOM subsidence experiments.
#
# SINGLE SOURCE OF TRUTH
# -------------------------
# This file is the ONLY place users should edit experiment settings.
#
# STRICT RULES FOR THIS FILE
# -----------------------------
# - Only:
#       NAME = VALUE
#   assignments and comments.
# - No imports.
# - No helper functions.
# - Keep values JSON-serializable (str/int/float/bool/list/dict/None).
#
#  HOW SCRIPTS MUST READ THIS CONFIG
# -----------------------------------
# All scripts (Stage-1 prepare, training, tuning) must obtain their
# configuration through:
#
#     from fusionlab.utils.nat_utils import load_nat_config
#     cfg = load_nat_config()
#     CITY_NAME = cfg["CITY_NAME"]
#
# The `nat_utils` module is responsible for:
#   - reading this file,
#   - generating / updating `nat.com/config.json`,
#   - returning a flat configuration dictionary.
#
# If `config.py` changes, `nat_utils` detects it and regenerates
# `config.json` automatically.


# ===================================================================
# 1) CORE EXPERIMENT SETUP
# ===================================================================

# -------------------------------------------------------------------
# 1.1 City / model identifiers
# -------------------------------------------------------------------
# CITY_NAME selects which city dataset is used.
# Typical values: "nansha", "zhongshan"
CITY_NAME = "nansha"

# MODEL_NAME selects the Stage-2 model flavour:
#   - "HybridAttn-NoPhysics" : HybridAttn encoder-decoder, physics OFF
#   - "PoroElasticSubsNet"   : poroelastic surrogate (consolidation-only)
#   - "GeoPriorSubsNet"      : full GeoPriorSubsNet (default)
MODEL_NAME = "GeoPriorSubsNet"


# -------------------------------------------------------------------
# 1.2 Data root and file patterns
# -------------------------------------------------------------------
DATA_DIR = ".."

# Dataset variant tag (v3.2 real-data harmonized export)
DATASET_VARIANT = "with_zsurf"

# File name templates. When CITY_NAME="nansha":
#   BIG_FN   -> "nansha_final_main_std.harmonized.with_zsurf.csv"
#   SMALL_FN -> "nansha_2000.with_zsurf.csv"  (only if you created it)
BIG_FN_TEMPLATE = "{city}_final_main_std.harmonized.{variant}.csv"
SMALL_FN_TEMPLATE = "{city}_2000.{variant}.csv"

# Resolved filenames (scripts may use these directly).
BIG_FN = BIG_FN_TEMPLATE.format(city=CITY_NAME, variant=DATASET_VARIANT)
SMALL_FN = SMALL_FN_TEMPLATE.format(city=CITY_NAME, variant=DATASET_VARIANT)

ALL_CITIES_PARQUET = "natcom_all_cities.parquet"


# -------------------------------------------------------------------
# 1.3 Temporal windows
# -------------------------------------------------------------------
# TRAIN_END_YEAR:
#   Last year included in the training set.
#
# FORECAST_START_YEAR:
#   First year included in the forecast window.
#
# FORECAST_HORIZON_YEARS:
#   Forecast length, expressed in years.
#
# TIME_STEPS:
#   Historical look-back window length (years).
#
# MODE:
#   Encoder–decoder sequence layout:
#     - "tft_like"   : history + future blocks (TFT style)
#     - "pihal_like" : legacy layout
TRAIN_END_YEAR = 2022
FORECAST_START_YEAR = 2023
FORECAST_HORIZON_YEARS = 3
TIME_STEPS = 5
MODE = "tft_like"   # {"pihal_like", "tft_like"}


# -------------------------------------------------------------------
# 1.4 Column names and groundwater conventions
# -------------------------------------------------------------------
# Column names must match your harmonized CSV headers.
#
# TIME_COL: temporal index column (typically "year")
# LON_COL / LAT_COL: spatial coordinates
# SUBSIDENCE_COL: subsidence target (often cumulative, e.g. "subsidence_cum")
# GWL_COL: groundwater observation column used by the model
# H_FIELD_COL_NAME: thickness proxy for GeoPrior physics (H-field)
TIME_COL = "year"
LON_COL = "longitude"
LAT_COL = "latitude"
SUBSIDENCE_COL = "subsidence_cum"
GWL_COL = "GWL_depth_bgs"   # preferred over "GWL" (if both exist)
H_FIELD_COL_NAME = "soil_thickness"

# Groundwater representation (critical for sign consistency):
# - GWL_KIND:
#     "depth_bgs" -> depth below ground surface (positive downward)
#     "head"      -> hydraulic head elevation (positive upward)
# - GWL_SIGN:
#     "down_positive" -> z increases downward (depth-like convention)
#     "up_positive"   -> z increases upward (head-like convention)
#
# If you do not have a reliable surface elevation (z_surf),
# USE_HEAD_PROXY=True uses a simple proxy:
#   head_proxy ≈ -depth
# This keeps head and depth linked for physics, but is approximate.
GWL_KIND = "depth_bgs"       # {"depth_bgs", "head"}
GWL_SIGN = "down_positive"   # {"down_positive", "up_positive"}

# With z_surf available, do NOT use the proxy
USE_HEAD_PROXY = False

# Column containing surface elevation z_surf in meters.
# IMPORTANT: set this to the actual column name you wrote into the CSV.
# Recommended name for v3.2: "z_surf"
Z_SURF_COL = "z_surf" # e.g. None :"dem_m" if available, else None

INCLUDE_Z_SURF_AS_STATIC = True   # new (recommended)
HEAD_COL ="head_m"
# IMPORTANT (recommended in new GeoPrior paths):
# If the model cannot resolve which channel inside dynamic_features is GWL,
# you MUST provide gwl_dyn_index in scaling_kwargs (Stage-2 uses this).
#
# - Set to an integer when your dynamic_features has a fixed order.
# - Leave None only if Stage-2 can reliably infer it from names.
GWL_DYN_INDEX = None         # e.g. 0 if z_GWL is the first dynamic channel

# Stage-1: physics-critical scaling controls

# Keep coords raw for physics:
KEEP_COORDS_RAW = True

# Keep H_field in meters (recommended):
SCALE_H_FIELD = False

# Keep GWL/head in meters (recommended):
SCALE_GWL = False

# NEW (recommended): keep z_surf in meters, unscaled
SCALE_Z_SURF = False
Z_SURF_UNIT_TO_SI = 1.0

# Convert subsidence to SI (recommended):
# If SUBSIDENCE_COL is in mm or mm/yr, convert to meters or meters/yr.
SUBS_UNIT_TO_SI = 1e-3

# If subsidence is "rate" (per year) or "cumulative":
SUBSIDENCE_KIND = "cumulative"   # {"cumulative", "rate"}


# ===================================================================
# 2) FEATURE REGISTRY (Stage-1 -> Stage-2 handshake)
# ===================================================================
# This section defines which OPTIONAL columns are used as:
#   - dynamic drivers (past/historical)
#   - static features (categorical one-hot)
#   - future-known drivers (forecast-window scenario inputs)
#
# Each entry may be:
#   - a string (exact column name), or
#   - a tuple/list of candidate names (first match is used).

# -------------------------------------------------------------------
# 2.1 Optional numeric (dynamic) drivers
# -------------------------------------------------------------------
OPTIONAL_NUMERIC_FEATURES = [
    ("rainfall_mm", "rainfall", "rain_mm", "precip_mm"),
    ("urban_load_global", "normalized_density", "urban_load"),
]

# -------------------------------------------------------------------
# 2.2 Optional categorical (static) features
# -------------------------------------------------------------------
OPTIONAL_CATEGORICAL_FEATURES = [
    ("lithology", "geology"),
    "lithology_class",
]

# -------------------------------------------------------------------
# 2.3 Already-normalized numeric features
# -------------------------------------------------------------------
# These columns are assumed already scaled (e.g. to [0, 1]).
# Stage-1 should NOT apply another MinMaxScaler to them.
ALREADY_NORMALIZED_FEATURES = [
    "urban_load_global",
]

# -------------------------------------------------------------------
# 2.4 Future-known drivers (TFT-like)
# -------------------------------------------------------------------
# Subset of numeric drivers that can be known (or scenario-provided)
# in the forecast horizon. These become `future_features`.
FUTURE_DRIVER_FEATURES = [
    ("rainfall_mm", "rainfall", "rain_mm", "precip_mm"),
]

# Optional explicit naming (helps Stage-2 build dynamic_feature_names):
# Keep as None unless you are fully controlling feature order.
DYNAMIC_FEATURE_NAMES = None   # e.g. ["z_GWL", "rainfall_mm", "urban_load_global"]
FUTURE_FEATURE_NAMES = None    # e.g. ["rainfall_mm"]


# ===================================================================
# 3) CENSORING / EFFECTIVE-THICKNESS (H_eff) CONFIGURATION
# ===================================================================
# Soil thickness (and possibly other variables) can be censored at a
# measurement/processing cap.
#
# For each spec, Stage-1 may derive:
#   - <col>_censored : boolean flag
#   - <col>_eff      : effective value used by the model
#
# NOTE: This is central to GeoPrior because H_eff influences tau prior,
#       s_eq, and the learned closures.

CENSORING_SPECS = [
    {
        "col": H_FIELD_COL_NAME,
        "direction": "right",        # "right" (>= cap) or "left" (<= cap)
        "cap": 30.0,                 # instrument / processing cap
        "tol": 1e-6,                 # tolerance for equality checks
        "flag_suffix": "_censored",  # derived indicator name
        "eff_suffix": "_eff",        # derived effective-value name

        # How to form the effective value:
        #   - "clip"          : min(x, cap)  (recommended default)
        #   - "cap_minus_eps" : use cap*(1-eps) when censored
        #   - "nan_if_censored": set NaN then impute (see "impute")
        "eff_mode": "clip",
        "eps": 0.02,                 # used only for "cap_minus_eps"

        # Used only if eff_mode == "nan_if_censored"
        "impute": {
            "by": ["year"],
            "func": "median",
        },

        # Optional probability threshold if flags come from soft values
        "flag_threshold": 0.5,
    },
]

# If True, include *_censored flags as extra dynamic drivers.
INCLUDE_CENSOR_FLAGS_AS_DYNAMIC = True

# If True, also include *_censored flags as extra future drivers.
# Usually False for thickness, because it is effectively static/sample-wise.
INCLUDE_CENSOR_FLAGS_AS_FUTURE = False

# If True, prefer "<col>_eff" (if created) as the thickness fed to the model.
USE_EFFECTIVE_H_FIELD = True

# Raw thickness unit conversion to SI meters.
# If your thickness is already meters, keep 1.0.
THICKNESS_UNIT_TO_SI = 1.0

# If True, Stage-1 may pre-build future_* NPZ blocks for Stage-3 scenarios.
BUILD_FUTURE_NPZ = False


# ===================================================================
# 4) MODEL ARCHITECTURE DEFAULTS (Stage-2)
# ===================================================================

# -------------------------------------------------------------------
# 4.1 Attention / architecture defaults
# -------------------------------------------------------------------
ATTENTION_LEVELS = ["cross", "hierarchical", "memory"]

EMBED_DIM = 32
HIDDEN_UNITS = 64
LSTM_UNITS = 64
ATTENTION_UNITS = 64
NUMBER_HEADS = 2
DROPOUT_RATE = 0.10

# Additional BaseAttentive / GeoPriorSubsNet knobs
MEMORY_SIZE = 50
SCALES = [1, 2]
USE_RESIDUALS = True
USE_BATCH_NORM = False
USE_VSN = True
VSN_UNITS = 32


# -------------------------------------------------------------------
# 4.2 Probabilistic outputs and asymmetric loss weights
# -------------------------------------------------------------------
# Quantiles for probabilistic predictions (subsidence + gwl heads).
QUANTILES = [0.1, 0.5, 0.9]

# Pinball weights: heavier tails encourages better uncertainty calibration.
SUBS_WEIGHTS = {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
GWL_WEIGHTS  = {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}


# ===================================================================
# 5) PHYSICS CONFIGURATION (GeoPrior PINN block)
# ===================================================================

# -------------------------------------------------------------------
# 5.1 Which residuals are active
# -------------------------------------------------------------------
# PDE_MODE_CONFIG:
#   - "both" or "on"   : consolidation + groundwater flow
#   - "consolidation"  : consolidation only
#   - "gw_flow"        : groundwater flow only
#   - "none" or "off"  : physics switched off
PDE_MODE_CONFIG = "both"

# For data-only baselines, scripts may ignore physics even if enabled above.
PHYSICS_BASELINE_MODE = "none"

# If True, use internal scale factors (c*, g*) so residual terms are comparable.
SCALE_PDE_RESIDUALS = True


# -------------------------------------------------------------------
# 5.2 Relative weights of each physics term (compile-time)
# -------------------------------------------------------------------
LAMBDA_CONS   = 0.10
LAMBDA_GW     = 0.005
LAMBDA_PRIOR  = 0.05
LAMBDA_SMOOTH = 0.01
LAMBDA_MV     = 0.005
LAMBDA_BOUNDS = 1e-4


# -------------------------------------------------------------------
# 5.3 Global physics-loss multiplier (offset)
# -------------------------------------------------------------------
# OFFSET_MODE controls how lambda_offset is interpreted:
#   - "mul"   : physics_mult = lambda_offset
#   - "log10" : physics_mult = 10 ** lambda_offset
OFFSET_MODE = "mul"     # {"mul", "log10"}

# Initial value used in model.compile(lambda_offset=...)
LAMBDA_OFFSET = 1.0

# Optional scheduler (recommended when physics can dominate early).
USE_LAMBDA_OFFSET_SCHEDULER = True

# Scheduler semantics:
# - LAMBDA_OFFSET_UNIT: step vs epoch schedule indexing
# - LAMBDA_OFFSET_WHEN: update at begin vs end
LAMBDA_OFFSET_UNIT = "epoch"   # {"epoch", "step"}
LAMBDA_OFFSET_WHEN = "begin"   # {"begin", "end"}

# If LAMBDA_OFFSET_SCHEDULE is None, callback uses warmup:
# start -> end over `LAMBDA_OFFSET_WARMUP` epochs/steps.
LAMBDA_OFFSET_WARMUP = 15

# Safe defaults:
# - start small so the model learns data scale before physics locks in
# - end at 1.0 (neutral)
LAMBDA_OFFSET_START = 0.05
LAMBDA_OFFSET_END = 1.0

# Optional explicit schedule:
# - dict  : {index: value} where index is epoch/step
# - list  : values[index]
LAMBDA_OFFSET_SCHEDULE = None
# Example:
# LAMBDA_OFFSET_SCHEDULE = {0: 0.1, 5: 0.5, 10: 1.0}

# Learning-rate multipliers for scalar physics parameters.
MV_LR_MULT = 1.0
KAPPA_LR_MULT = 5.0


# -------------------------------------------------------------------
# 5.4 Physics bounds (specified in LINEAR space here)
# -------------------------------------------------------------------
# Bounds are used for:
# - soft penalties (PHYSICS_BOUNDS_MODE)
# - residual scaling / prior anchoring (Stage-2 converts as needed)
PHYSICS_BOUNDS = {
    "H_min": 0.1,
    "H_max": 30.0,      # match censor cap for thickness

    "K_min": 1e-12,     # [m/s]
    "K_max": 1e-7,

    "Ss_min": 1e-6,     # [1/m]
    "Ss_max": 1e-3,
}

# Bounds penalty mode:
# - "soft" : penalize violations (recommended)
# - "hard" : clamp or reject (only if you know what you are doing)
PHYSICS_BOUNDS_MODE = "soft"

# Time coordinate units used by physics conversions (rate_to_per_second etc.)
# Must match what `TIME_COL` represents in your dataset.
TIME_UNITS = "year"



# -------------------------------------------------------------------
# 5.5 Model->SI affine mapping for physics residuals
# -------------------------------------------------------------------
# Physics residuals must run in SI-consistent units.
# The model outputs are often in Stage-1 scaled space.
#
# Convert with:
#   y_si = y_model * SCALE + BIAS
#
# If *_SCALE_SI / *_BIAS_SI are None and AUTO_SI_AFFINE_FROM_STAGE1=True,
# Stage-2 should infer them from Stage-1 scalers (recommended).
# SUBS_UNIT_TO_SI = 1e-3   # e.g. mm -> m
HEAD_UNIT_TO_SI = 1.0    # typically already meters

SUBS_SCALE_SI = 1.0
SUBS_BIAS_SI  = 0.0
HEAD_SCALE_SI = None
HEAD_BIAS_SI  = None

AUTO_SI_AFFINE_FROM_STAGE1 = True


# -------------------------------------------------------------------
# 5.6 Coordinate handling for physics (x,y)
# -------------------------------------------------------------------
# If coords are degrees, Stage-2 must convert degrees -> meters internally
# before computing spatial derivatives (or use ranges to rescale).
COORD_MODE = "degrees"     # {"utm", "degrees"}
UTM_EPSG = 32649           # if COORD_MODE="utm" then use it

COORD_SRC_EPSG = 4326 # (e.g. 4326 if lon/lat WGS84)
# COORD_TARGET_EPSG= 32649 # (your UTM EPSG, e.g. 32649)

# ===================================================================
# 6) GEOPRIOR SCALAR PARAMETERS (initialization / closures)
# ===================================================================

# GeoPrior scalar priors
GEOPRIOR_INIT_MV = 1e-7
GEOPRIOR_INIT_KAPPA = 1.0
GEOPRIOR_GAMMA_W = 9810.0

# Kappa mode:
#   - "kb"  : kappa_b
#   - "bar" : kappa_bar (if you use an effective compressibility mapping)
GEOPRIOR_KAPPA_MODE = "kb"   # {"bar", "kb"}

# Effective-thickness usage inside GeoPrior physics:
GEOPRIOR_USE_EFFECTIVE_H = True
GEOPRIOR_HD_FACTOR = 0.6

# Reference head for drawdown:
#   Δh = max(h_ref - h, 0)
#
# Recommended:
#   "auto" -> use last historical groundwater observation per sample as h_ref.
#
# Numeric fallback:
#   0.0 -> fixed datum (useful for synthetic 1-pixel tests)
GEOPRIOR_H_REF = "auto"   # or 0.0


# -------------------------------------------------------------------
# 5. Consolidation stepping / residual definition (v3.2)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# 5.x v3.2: residual discretization controls
# -------------------------------------------------------------------
# How to compute the per-step consolidation residual inside the batch:
#   - "exact" : v3.2 exact step formulation (preferred)
#   - "fd"    : finite-difference / legacy discretization
CONSOLIDATION_STEP_RESIDUAL_METHOD = "exact"


# ===================================================================
# 7) TRAINING LOOP DEFAULTS (non-tuner runs)
# ===================================================================
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


# ===================================================================
# 8) HARDWARE / RUNTIME (TensorFlow)
# ===================================================================

# TF_DEVICE_MODE:
#   - "auto" : use GPU if available, else CPU
#   - "cpu"  : force CPU only
#   - "gpu"  : force GPU only (first visible GPU)
TF_DEVICE_MODE = "auto"

# CPU threading (None -> TensorFlow decides)
TF_INTRA_THREADS = None
TF_INTER_THREADS = None

# GPU memory behaviour
TF_GPU_ALLOW_GROWTH = True
TF_GPU_MEMORY_LIMIT_MB = None   # e.g. 12000 for 12 GB, or None


# -------------------------------------------------------------------
# 5. TUNING SEARCH SPACE
# -------------------------------------------------------------------
# Hyperparameter search space for the GeoPriorTuner.  The tuner
# script imports this as a simple dictionary and passes it to
# the tuning API.

TUNER_SEARCH_SPACE = {
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
    
    "residual_method": ["exact"],  # or ["exact", "euler"]
    
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
    "lambda_offset": {
        "type": "float",
        "min_value": 0.1,
        "max_value": 50.0,
        "sampling": "log",
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
