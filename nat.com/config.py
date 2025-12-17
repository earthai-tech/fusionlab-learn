# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author : LKouadio <etanoyau@gmail.com>
#
# Central configuration for the NATCOM subsidence experiments.
#
# This file is the **single source of truth** that users edit.
# It should only contain:
#
#     NAME = VALUE
#
# assignments and comments.  No imports, no helper functions.
#
# All scripts (Stage-1 prepare, training, tuning) must obtain
# their configuration through:
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
# If `config.py` changes, `nat_utils` will detect it and
# regenerate `config.json` automatically.


# -------------------------------------------------------------------
# 1. CORE EXPERIMENT SETUP
# -------------------------------------------------------------------
# 1.1 City / model identifiers
# ----------------------------
# CITY_NAME selects which city dataset is used.
#
# MODEL_NAME selects the model *flavour* used in Stage-2:
#   - "HybridAttn-NoPhysics" : HybridAttn encoder-decoder, physics OFF
#   - "PoroElasticSubsNet"   : poroelastic surrogate (consolidation-only)
#   - "GeoPriorSubsNet"      : full GeoPriorSubsNet (default)
CITY_NAME = "zhongshan"
MODEL_NAME = "GeoPriorSubsNet"



# 1.2 Data root and file patterns
# -------------------------------
# DATA_DIR is the base path where the CSV files live, relative to
# the project root.  A typical layout is:
#
#   fusionlab-learn/
#       nat.com/
#           config.py
#           main_NATCOM_stage1_prepare.py
#       data/
#           nansha_final_main_std.harmonized.csv
#           nansha_2000.csv
#
# You can change DATA_DIR if your data folder is elsewhere.
# Advanced users can also override this via environment variables
# in `nat_utils` (for example `JUPYTER_PROJECT_ROOT`).
DATA_DIR = ".."

# File name templates.  When CITY_NAME = "nansha", this becomes:
#   BIG_FN   = "nansha_final_main_std.harmonized.csv"
#   SMALL_FN = "nansha_2000.csv"
BIG_FN_TEMPLATE = "{city}_final_main_std.harmonized.csv"
SMALL_FN_TEMPLATE = "{city}_2000.csv"

BIG_FN = BIG_FN_TEMPLATE.format(city=CITY_NAME)
SMALL_FN = SMALL_FN_TEMPLATE.format(city=CITY_NAME)

# Optional multi-city parquet (e.g. natcom_all_cities.parquet)
ALL_CITIES_PARQUET = "natcom_all_cities.parquet"

# 1.3 Temporal windows
# --------------------
# TRAIN_END_YEAR:
#   Last year included in the training set.
#
# FORECAST_START_YEAR:
#   First year included in the forecasting window.
#
# FORECAST_HORIZON_YEARS:
#   Forecast length, expressed in years.
#
# TIME_STEPS:
#   Length of the historical look-back window, in years.
#
# MODE:
#   Sequence layout for the encoder–decoder input:
#     - "tft_like"   : history + future blocks (TFT style),
#     - "pihal_like" : alternative legacy layout.
TRAIN_END_YEAR = 2022
FORECAST_START_YEAR = 2023
FORECAST_HORIZON_YEARS = 5 
TIME_STEPS = 3
MODE = "tft_like"   # {"pihal_like", "tft_like"}


# 1.4 Column names
# ----------------
# Column names must match the harmonized CSV headers.
#
# TIME_COL:
#   Temporal index, typically "year".
#
# LON_COL / LAT_COL:
#   Spatial coordinates.
#
# SUBSIDENCE_COL:
#   Yearly subsidence (or similar) used as target.
#
# GWL_COL:
#   Groundwater level.  We usually work with depth
#   below ground surface in a standardized z-system.
#
# H_FIELD_COL_NAME:
#   Soil thickness proxy fed to GeoPriorSubsNet as
#   the H-field in the physics block.
TIME_COL = "year"
LON_COL = "longitude"
LAT_COL = "latitude"
SUBSIDENCE_COL = "subsidence"
GWL_COL = "GWL_depth_bgs_z"
H_FIELD_COL_NAME = "soil_thickness"


# -------------------------------------------------------------------
# 2. FEATURE REGISTRY
# -------------------------------------------------------------------
# This section defines which *optional* columns are used as
# dynamic drivers, static (categorical) features, and
# future-known drivers for TFT-style models.


# 2.1 Optional numeric features
# -----------------------------
# Each entry is either:
#   - a string, interpreted as a column name, or
#   - a tuple/list of candidate names; the first that is present
#     in the dataframe is selected.
#
# Resolved numeric features are added to the dynamic drivers.
OPTIONAL_NUMERIC_FEATURES = [
    ("rainfall_mm", "rainfall", "rain_mm", "precip_mm"),
    ("urban_load_global", "normalized_density", "urban_load"),
    # Add more numeric candidates here if needed.
]


# 2.2 Optional categorical features
# ---------------------------------
# Columns that will be one-hot encoded and used as static
# features.  As above, tuples represent alternative names.
OPTIONAL_CATEGORICAL_FEATURES = [
    ("lithology", "geology"),
    "lithology_class",   # used only if present
    # Add more categorical candidates here if needed.
]


# 2.3 Already-normalized numeric features
# ---------------------------------------
# Numeric columns that are already scaled to [0, 1] and should
# *not* be rescaled by the global MinMax scaler.  This avoids
# double-normalization.
ALREADY_NORMALIZED_FEATURES = [
    # "normalized_urban_load_proxy",
    "urban_load_global",
    # Add more if needed.
]


# 2.4 Future-known drivers
# ------------------------
# Subset of numeric drivers that can have known (or scenario)
# values in the forecast window.  These are exposed as
# `future_features` for TFT-style models.
FUTURE_DRIVER_FEATURES = [
    ("rainfall_mm", "rainfall", "rain_mm", "precip_mm")
    # Add more if you want multi-driver futures.
]


# -------------------------------------------------------------------
# 3. CENSORING CONFIGURATION
# -------------------------------------------------------------------
# Soil thickness (and possibly other variables) can be censored
# at a measurement or processing cap.  Each entry in
# CENSORING_SPECS describes one numeric column that may be
# censored.
#
# For each spec we can derive:
#   - <col>_censored : boolean flag,
#   - <col>_eff      : effective value used by the model.
#
# nat_utils will also build a compact "censoring" block from
# these values for use in Stage-2 scripts.

CENSORING_SPECS = [
    {
        "col": H_FIELD_COL_NAME,     # for example "soil_thickness"
        "direction": "right",        # "right" (>= cap) or "left" (<= cap)
        "cap": 30.0,                 # instrument / processing cap
        "tol": 1e-6,                 # tolerance when testing equality
        "flag_suffix": "_censored",  # boolean indicator column
        "eff_suffix": "_eff",        # effective value column
        # Optional alternative naming:
        # "flag_col": "soil_thickness_censored",
        #
        # How to form the effective value:
        #   "clip"          : min(x, cap),
        #   "cap_minus_eps" : cap * (1 - eps) if censored,
        #   "nan_if_censored":
        #       set NaN, then impute using the "impute" rule.
        "eff_mode": "clip",
        "eps": 0.02,                 # used only for "cap_minus_eps"
        "impute": {
            "by": ["year"],
            "func": "median",
        },  # used only if eff_mode == "nan_if_censored"
        "flag_threshold": 0.5,
    },
]

# If True, add *_censored flags as extra dynamic drivers.
INCLUDE_CENSOR_FLAGS_AS_DYNAMIC = True

# If True, prefer the effective column "<col>_eff" (if created)
# when feeding the H-field into GeoPriorSubsNet.
USE_EFFECTIVE_H_FIELD = True

# Optional: whether Stage-1 should also pre-build future_* NPZ for Stage-3
BUILD_FUTURE_NPZ = True

# -------------------------------------------------------------------
# 4. MODEL / PHYSICS / TRAINING DEFAULTS
# -------------------------------------------------------------------
# These are defaults used when training without the tuner, or
# as central values around which the tuner explores.


# 4.1 Attention / architecture defaults
# -------------------------------------
ATTENTION_LEVELS = ["cross", "hierarchical", "memory"]

EMBED_DIM = 32
HIDDEN_UNITS = 64
LSTM_UNITS = 64
ATTENTION_UNITS = 64 #32 # 64
NUMBER_HEADS = 2 #4 #2
DROPOUT_RATE = 0.10

# Additional architectural knobs used inside BaseAttentive /
# GeoPriorSubsNet.
MEMORY_SIZE = 50
SCALES = [1, 2]
USE_RESIDUALS = True
USE_BATCH_NORM = False
USE_VSN = True
VSN_UNITS = 32


# 4.2 Probabilistic outputs and loss weights
# ------------------------------------------
# Quantiles for probabilistic predictions.
QUANTILES = [0.1, 0.5, 0.9]

# Asymmetric pinball loss weights per quantile for subsidence.
SUBS_WEIGHTS = {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}

# Asymmetric pinball loss weights per quantile for GWL.
GWL_WEIGHTS = {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}


# 4.3 Physics loss configuration
# ------------------------------
# PDE_MODE_CONFIG selects which physical residuals are active:
#   - "both" or "on"   : consolidation + groundwater flow,
#   - "consolidation"  : consolidation only,
#   - "gw_flow"        : groundwater flow only,
#   - "none" or "off"  : physics switched off.
PDE_MODE_CONFIG = "both"
PHYSICS_BASELINE_MODE = "none"  # used for data-only baseline

SCALE_PDE_RESIDUALS = True

# Relative weights for each physics term.
LAMBDA_CONS = 0.10
LAMBDA_GW = 0.01
LAMBDA_PRIOR = 0.10
LAMBDA_SMOOTH = 0.01
LAMBDA_MV = 0.01

LAMBDA_BOUNDS = 0.0

# 4.3ter Global physics-loss offset (scales the whole physics block)
# ------------------------------------------------------------------
# OFFSET_MODE controls how `model._lambda_offset` is interpreted:
#   - "mul"   : physics_mult = lambda_offset
#   - "log10" : physics_mult = 10 ** lambda_offset
OFFSET_MODE = "mul"   # {"mul", "log10"}

# Initial value assigned in model.compile(lambda_offset=...)
LAMBDA_OFFSET = 1.0

# Optional scheduler (OFF by default)
USE_LAMBDA_OFFSET_SCHEDULER = False

# Scheduler knobs (used only when USE_LAMBDA_OFFSET_SCHEDULER=True)
LAMBDA_OFFSET_UNIT = "epoch"   # {"epoch", "step"}
LAMBDA_OFFSET_WHEN = "begin"   # {"begin", "end"}

# If LAMBDA_OFFSET_SCHEDULE is None, callback uses linear warmup:
# start -> end over `warmup` epochs/steps.
LAMBDA_OFFSET_WARMUP = 10
LAMBDA_OFFSET_START = None
LAMBDA_OFFSET_END = None

# Optional explicit schedule:
# - dict  : {index: value} where index is epoch/step
# - list  : values[index]
LAMBDA_OFFSET_SCHEDULE = None
# Example:
# LAMBDA_OFFSET_SCHEDULE = {0: 0.1, 5: 0.5, 10: 1.0}


# Learning-rate multipliers for scalar physics parameters.
MV_LR_MULT = 1.0
KAPPA_LR_MULT = 5.0

# 4.3bis Physics bounds for scaling
# ---------------------------------
# Global, city-level ranges used when scaling PDE residuals and
# defining log-offset priors.
#
# Bounds are specified in *linear* space here (no numpy in config.py);
# the training script converts them to log-space as needed.

PHYSICS_BOUNDS = {
    # Effective thickness H [m]
    "H_min": 5.0,
    "H_max": 80.0,

    # Hydraulic conductivity K [m/s]
    "K_min": 1e-8,
    "K_max": 1e-3,

    # Specific storage Ss [Pa^-1]
    "Ss_min": 1e-7,
    "Ss_max": 1e-3,
}

TIME_UNITS ="year" 
# -------------------------------------------------------------------
# 4.3quater  Model->SI affine mapping for physics residuals
# -------------------------------------------------------------------
# Physics residuals should run in physical/SI-consistent units.
# The model outputs are usually in the Stage-1 scaled space.
#
# We convert with an affine map:
#   y_si = y_model * SCALE + BIAS
#
# If *_SCALE_SI / *_BIAS_SI are None, Stage-2 will infer them from
# the Stage-1 target scalers (recommended).
#
# Optional extra unit factors (e.g., if your raw subsidence is mm/yr but
# physics expects m/yr, set SUBS_UNIT_TO_SI=1e-3).
SUBS_UNIT_TO_SI = 1e-3   # mm -> m (set 1.0 if already meters)
HEAD_UNIT_TO_SI = 1.0    # head already meters in most cases

SUBS_SCALE_SI = None
SUBS_BIAS_SI  = None
HEAD_SCALE_SI = None
HEAD_BIAS_SI  = None

# Prefer auto-derive from Stage-1 scalers when None
AUTO_SI_AFFINE_FROM_STAGE1 = True

# 4.4 GeoPrior scalar parameters
# ------------------------------
# These control how the geomechanical prior is initialised and
# used inside GeoPriorSubsNet (see `_geoprior_subnet.py` and
# the Methods section of the revised manuscript).
GEOPRIOR_INIT_MV = 1e-7
GEOPRIOR_INIT_KAPPA = 1.0
GEOPRIOR_GAMMA_W = 9810.0
GEOPRIOR_H_REF = 0.0
GEOPRIOR_KAPPA_MODE = "kb"   # {"bar", "kb"}
GEOPRIOR_USE_EFFECTIVE_H = True
GEOPRIOR_HD_FACTOR = 0.6


# 4.5 Training loop defaults
# --------------------------
# Used when training directly (without tuner) and as defaults
# for compile / fit arguments.

EPOCHS = 100

BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# 4.6. Hardware / device configuration
# ----------------------------------
# TF_DEVICE_MODE:
#   - "auto" : use GPU if available, else CPU
#   - "cpu"  : force CPU only
#   - "gpu"  : force GPU only (first visible GPU, unless env overrides)
TF_DEVICE_MODE = "auto"

# CPU threading.  None → let TensorFlow decide.
TF_INTRA_THREADS = None
TF_INTER_THREADS = None

# GPU memory behaviour
TF_GPU_ALLOW_GROWTH = True        # True recommended for desktop GPUs
TF_GPU_MEMORY_LIMIT_MB = None     # e.g. 12000 to cap at 12 GB, or None

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
