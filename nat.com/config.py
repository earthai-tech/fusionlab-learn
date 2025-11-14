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
# CITY_NAME selects which city dataset is used.  Typical values:
#   - "nansha"
#   - "zhongshan"
#
# MODEL_NAME is the registered model identifier.  For now the
# main model is:
#   - "GeoPriorSubsNet"
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

# optional multi-city parquet (e.g. natcom_all_cities.parquet)
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
FORECAST_HORIZON_YEARS = 3 
TIME_STEPS = 5
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
    "rainfall_mm",
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
ATTENTION_UNITS = 64
NUMBER_HEADS = 2
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
#   - "both"         : consolidation + groundwater flow,
#   - "consolidation": consolidation only,
#   - "gw_flow"      : groundwater flow only,
#   - "none"         : physics switched off.
PDE_MODE_CONFIG = "off"

SCALE_PDE_RESIDUALS = True

# Relative weights for each physics term.
LAMBDA_CONS = 0.10
LAMBDA_GW = 0.01
LAMBDA_PRIOR = 0.10
LAMBDA_SMOOTH = 0.01
LAMBDA_MV = 0.01

# Learning-rate multipliers for scalar physics parameters.
MV_LR_MULT = 1.0
KAPPA_LR_MULT = 5.0


# 4.4 GeoPrior scalar parameters
# ------------------------------
# These control how the geomechanical prior is initialised and
# used inside GeoPriorSubsNet (see `_geoprior_subnet.py` and
# the Methods section of the revised manuscript).
GEOPRIOR_INIT_MV = 1e-7
GEOPRIOR_INIT_KAPPA = 1.0
GEOPRIOR_GAMMA_W = 9810.0
GEOPRIOR_H_REF = 0.0
GEOPRIOR_KAPPA_MODE = "bar"   # {"bar", "kb"}
GEOPRIOR_USE_EFFECTIVE_H = True
GEOPRIOR_HD_FACTOR = 0.6


# 4.5 Training loop defaults
# --------------------------
# Used when training directly (without tuner) and as defaults
# for compile / fit arguments.
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


# -------------------------------------------------------------------
# 5. TUNING SEARCH SPACE
# -------------------------------------------------------------------
# Hyperparameter search space for the GeoPriorTuner.  The tuner
# script imports this as a simple dictionary and passes it to
# the tuning API.

TUNER_SEARCH_SPACE = {
    # --- Architecture (model.__init__) ---
    "embed_dim": [32, 64],
    "hidden_units": [64, 96, 128],
    "lstm_units": [64, 96],
    "attention_units": [32, 64],
    "num_heads": [2, 4],
    "dropout_rate": {
        "type": "float",
        "min_value": 0.10,
        "max_value": 0.30,
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
        "max_value": 0.80,
    },

    # Learnable scalar initials (model.__init__)
    "mv": {
        "type": "float",
        "min_value": 3e-7,
        "max_value": 1e-6,
        "sampling": "log",
    },
    "kappa": {
        "type": "float",
        "min_value": 0.7,
        "max_value": 1.5,
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
        "max_value": 1.0,
    },
    "lambda_cons": {
        "type": "float",
        "min_value": 0.01,
        "max_value": 1.0,
    },
    "lambda_prior": {
        "type": "float",
        "min_value": 0.1,
        "max_value": 0.8,
    },
    "lambda_smooth": {
        "type": "float",
        "min_value": 0.01,
        "max_value": 1.0,
    },
    "lambda_mv": {
        "type": "float",
        "min_value": 0.01,
        "max_value": 0.5,
    },
    "mv_lr_mult": {
        "type": "float",
        "min_value": 0.5,
        "max_value": 2.0,
    },
    "kappa_lr_mult": {
        "type": "float",
        "min_value": 1.0,
        "max_value": 10.0,
    },
}
