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
BIG_FN_TEMPLATE = "{city}_final_main_std.harmonized.cleaned.{variant}.csv"
SMALL_FN_TEMPLATE = "{city}_2000.cleaned.{variant}.csv"

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

# XXX: Optimize: preset:
# TRAIN_END_YEAR = 2022
# FORECAST_START_YEAR = 2023
# FORECAST_HORIZON_YEARS = 3

# # real data: slightly longer memory helps (still cheap)
# TIME_STEPS = 6
# MODE = "tft_like"

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
H_FIELD_COL_NAME = "soil_thickness"

GWL_COL = "GWL_depth_bgs_m"   # preferred over "GWL" (if both exist)
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

GWL_KIND =  "depth_bgs"  # or None ->depth_bgs defaults to "down_positive" for depth_bgs
GWL_SIGN = "down_positive"   # {"down_positive", "up_positive"}

# With z_surf available, do NOT use the proxy
USE_HEAD_PROXY = False

# Column containing surface elevation z_surf in meters.
# IMPORTANT: set this to the actual column name you wrote into the CSV.
# Recommended name for v3.2: "z_surf"
Z_SURF_COL = "z_surf_m" # e.g. None :"dem_m" if available, else None

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
NORMALIZE_COORDS = True          # preferred knob
KEEP_COORDS_RAW  = False         # legacy knob, keep for backward compat
SHIFT_RAW_COORDS = True          # only matters when KEEP_COORDS_RAW=True


# XXX: Optimize: preset:
# GWL_KIND = "depth_bgs"
# GWL_SIGN = "down_positive"
# USE_HEAD_PROXY = False

# Z_SURF_COL = "z_surf_m"
# INCLUDE_Z_SURF_AS_STATIC = True

# SCALE_GWL = False
# SCALE_H_FIELD = False
# SCALE_Z_SURF = False
# NORMALIZE_COORDS = True
# KEEP_COORDS_RAW = False



# Keep H_field in meters (recommended):
SCALE_H_FIELD = False

# Keep GWL/head in meters (recommended):
SCALE_GWL = False

# NEW (recommended): keep z_surf in meters, unscaled
SCALE_Z_SURF = False

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

# If True, Stage-1 may pre-build future_* NPZ blocks for Stage-3 scenarios.
BUILD_FUTURE_NPZ = False

# ========================================================
#   HOLD OUT STRATEGY 
# =========================================================

SPLIT_SEED = 42 
VAL_FRAC = 0.2 
TEST_FRAC = 0.1 
HOLDOUT_STRATEGY = "random" #  # (or "spatial_block" )
HOLDOUT_BLOCK_M = 2000.0 

# SPLIT_SEED = 42
# VAL_FRAC = 0.15
# TEST_FRAC = 0.15

# HOLDOUT_STRATEGY = "block"   # (or "spatial_block" if that’s your accepted token)
# HOLDOUT_BLOCK_M = 2500.0

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

# XXX: Optimize: preset:
# # Additional BaseAttentive / GeoPriorSubsNet knobs
# MEMORY_SIZE = 50
# SCALES = [1, 2]
# USE_RESIDUALS = True
# USE_BATCH_NORM = False
# USE_VSN = True
# VSN_UNITS = 32

# EMBED_DIM = 48
# HIDDEN_UNITS = 96
# LSTM_UNITS = 96
# ATTENTION_UNITS = 64
# NUMBER_HEADS = 4
# DROPOUT_RATE = 0.15

# MEMORY_SIZE = 80
# SCALES = [1, 2]
# USE_RESIDUALS = True
# USE_BATCH_NORM = False

# USE_VSN = True
# VSN_UNITS = 48

# -------------------------------------------------------------------
# 4.2 Probabilistic outputs and asymmetric loss weights
# -------------------------------------------------------------------
# Quantiles for probabilistic predictions (subsidence + gwl heads).
QUANTILES = [0.1, 0.5, 0.9]

# Pinball weights: heavier tails encourages better uncertainty calibration.
SUBS_WEIGHTS = {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
GWL_WEIGHTS  = {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}

# XXX: optimize preset
# QUANTILES = [0.1, 0.5, 0.9]

# SUBS_WEIGHTS = {0.1: 2.0, 0.5: 1.0, 0.9: 2.0}
# GWL_WEIGHTS  = {0.1: 1.2, 0.5: 1.0, 0.9: 1.2}

# ===================================================================
# 5) PHYSICS + TRAINING SCHEDULE (GeoPrior PINN block)
# ===================================================================
# This section is the SINGLE place where you control:
#   (A) what physics is enabled,
#   (B) how strong each physics term is,
#   (C) how physics strength is scheduled during training,
#   (D) when auxiliary constraints (Q, subs-residual) turn on,
#   (E) bounds / guards / diagnostics.
#
# Mental model
# ------------
# Total loss is conceptually:
#   L_total =
#       L_data(subs) + LOSS_WEIGHT_GWL * L_data(head)
#     + physics_mult(step) * (
#           LAMBDA_CONS   * L_cons
#         + LAMBDA_GW     * L_gw
#         + LAMBDA_PRIOR  * L_prior
#         + LAMBDA_SMOOTH * L_smooth
#         + LAMBDA_BOUNDS * L_bounds
#         + LAMBDA_MV     * L_mvprior
#         + LAMBDA_Q      * L_Q_reg * gate_Q(step)
#         + (optional)    L_subs_resid * gate_subs_resid(step)
#       )
#
# where:
#   - physics_mult(step) is the GLOBAL physics strength scheduler
#     (either PHYSICS_*_STEPS OR lambda_offset scheduler).
#   - gate_Q and gate_subs_resid are ON/OFF/RAMP gates controlled by
#     TRAINING_STRATEGY-specific policies.
#
# IMPORTANT RULE: avoid double scheduling
# --------------------------------------
# You should use ONE global scheduling mechanism:
#   - Preferred: PHYSICS_WARMUP_STEPS / PHYSICS_RAMP_STEPS (step-based).
#   - Alternative: USE_LAMBDA_OFFSET_SCHEDULER (epoch/step based).
# If you set PHYSICS_*_STEPS, keep USE_LAMBDA_OFFSET_SCHEDULER = False.


# -------------------------------------------------------------------
# 5.1 Physics switches (what residuals are active)
# -------------------------------------------------------------------
# PDE_MODE_CONFIG selects which PDE residual blocks are included.
# Accepted values (case-insensitive):
#   - "on" / "both"          : consolidation + groundwater flow
#   - "consolidation"        : consolidation only
#   - "gw_flow"              : groundwater flow only
#   - "off" / "none"         : physics disabled
PDE_MODE_CONFIG = "on"

# PHYSICS_BASELINE_MODE is for "data-only baselines" in scripts.
# Typical:
#   - "none" : do not override (normal behavior)
#   - other values may be used by your training script to force-disable
PHYSICS_BASELINE_MODE = "none"

# If True, internally scale PDE residuals (c*, g*) so magnitudes are comparable.
# Recommended: True (especially for real data, mixed units, and stability).
SCALE_PDE_RESIDUALS = True


# -------------------------------------------------------------------
# 5.2 Compile-time physics weights (relative importance INSIDE physics)
# -------------------------------------------------------------------
# These are multipliers for each physics component BEFORE applying the
# global physics multiplier (physics_mult).
#
# Guidance:
# - LAMBDA_CONS / LAMBDA_GW control how much the PDE residuals matter.
# - LAMBDA_PRIOR stabilizes the closure (e.g., tau prior / physical coherence).
# - LAMBDA_SMOOTH suppresses noisy spatial fields (K, Ss).
# - LAMBDA_BOUNDS penalizes constraint violations (soft bounds).
# - LAMBDA_MV controls MV-prior penalty (scalar regularization).
# - LAMBDA_Q is the base weight for Q regularization; it can still be
#   effectively OFF early via gating (TRAINING_STRATEGY section).
LAMBDA_CONS = 1.0
LAMBDA_GW = 0.10
LAMBDA_PRIOR = 0.20
LAMBDA_SMOOTH = 0.01
LAMBDA_BOUNDS = 0.05
LAMBDA_MV = 0.01
LAMBDA_Q = 0.0


# -------------------------------------------------------------------
# 5.3 Global physics strength (choose ONE scheduling mechanism)
# -------------------------------------------------------------------
# Option A (RECOMMENDED): step-based global warmup/ramp inside the model.
# -------------------------------------------------------------------
# PHYSICS_WARMUP_STEPS:
#   Number of optimizer steps where global physics is "soft" / warming up.
# PHYSICS_RAMP_STEPS:
#   Number of optimizer steps used to ramp to full physics strength.
#
# Because config.py cannot reference runtime variables (like steps_per_epoch),
# you MUST paste numeric values here.
#
# Typical values:
#   - data_first   : warmup ~ 2–3 epochs, ramp ~ 5–10 epochs
#   - physics_first: warmup ~ 0–1 epoch, ramp ~ 2–5 epochs
#
# Example (if steps_per_epoch = 600):
#   data_first warmup=3*600=1800, ramp=8*600=4800
PHYSICS_WARMUP_STEPS = 1800
PHYSICS_RAMP_STEPS = 4800

# Option B (ALTERNATIVE): lambda_offset scheduler (do NOT combine with Option A)
# -------------------------------------------------------------------
# OFFSET_MODE controls how LAMBDA_OFFSET is interpreted:
#   - "mul"   : physics_mult = LAMBDA_OFFSET
#   - "log10" : physics_mult = 10 ** LAMBDA_OFFSET
OFFSET_MODE = "mul"

# Base value used in model.compile(lambda_offset=...)
LAMBDA_OFFSET = 1.0

# If you use PHYSICS_*_STEPS above, keep this False (avoid double scheduling).
USE_LAMBDA_OFFSET_SCHEDULER = False

# If you enable the scheduler, it can run over epochs or steps.
LAMBDA_OFFSET_UNIT = "epoch"   # {"epoch", "step"}
LAMBDA_OFFSET_WHEN = "begin"   # {"begin", "end"}

# If LAMBDA_OFFSET_SCHEDULE is None, use warmup-based interpolation:
# start -> end over LAMBDA_OFFSET_WARMUP epochs/steps.
LAMBDA_OFFSET_WARMUP = 5
LAMBDA_OFFSET_START = 0.1
LAMBDA_OFFSET_END = 1.0

# Optional explicit schedule:
#   - dict: {index: value}   (index is epoch/step depending on UNIT)
#   - list: values[index]
LAMBDA_OFFSET_SCHEDULE = None


# -------------------------------------------------------------------
# 5.4 Learning-rate multipliers for scalar physics parameters
# -------------------------------------------------------------------
# These apply to specific scalar parameters in the physics block.
MV_LR_MULT = 1.0
KAPPA_LR_MULT = 5.0


# -------------------------------------------------------------------
# 5.5 MV prior (scalar regularization) scheduling
# -------------------------------------------------------------------
# MV_PRIOR_MODE controls how MV prior interacts with learned fields:
#   - "calibrate" : typical mode; may stop gradients through Ss_field
#   - other modes depend on your implementation
MV_PRIOR_MODE = "calibrate"

# MV prior weight is usually small (it is also inside physics_mult scaling).
MV_WEIGHT = 1e-3

# Units/shape controls for MV prior penalty (implementation-specific).
MV_PRIOR_UNITS = "strict"
MV_ALPHA_DISP = 0.1
MV_HUBER_DELTA = 1.0

# MV scheduler is independent from the global physics scheduler.
# Recommended: epoch-based (robust to batch size).
MV_SCHEDULE_UNIT = "epoch"   # {"epoch", "step"}

# Epoch-based ramp:
MV_DELAY_EPOCHS = 1
MV_WARMUP_EPOCHS = 2

# Step-based ramp (used only if MV_SCHEDULE_UNIT="step"):
MV_DELAY_STEPS = None
MV_WARMUP_STEPS = None

# If True, log extra auxiliary diagnostics/metrics (more verbose logs).
TRACK_AUX_METRICS = False


# -------------------------------------------------------------------
# 5.6 Training strategy: when constraints turn on (Q + subs residual)
# -------------------------------------------------------------------
# TRAINING_STRATEGY controls *gating* of:
#   - Q regularization (lambda_q term)
#   - subsidence residual contribution (if ALLOW_SUBS_RESIDUAL=True)
# and also controls which LOSS_WEIGHT_GWL / LAMBDA_Q values are used.
#
# Choose ONE:
#   - "data_first"   : fit observations early, then tighten constraints
#   - "physics_first": enforce constraints early, then let data refine
TRAINING_STRATEGY = "data_first"   # {"data_first", "physics_first"}

# Gate policies (for Q and subs residual):
#   - "always_on"  : gate = 1 from step 0
#   - "always_off" : gate = 0 forever (and lambda_q is forced to 0)
#   - "warmup_off" : gate = 0 during warmup, then ramps to 1
#
# NOTE:
# The actual conversion of epochs -> steps uses steps_per_epoch inside
# the training script (not here). Here you define warmup/ramp in epochs.
#
# ----------------------------
# A) PHYSICS-FIRST preset
# ----------------------------
# In physics-first, you often keep Q + subs residual OFF early so the PDE
# residuals/priors shape the solution manifold first.
Q_POLICY_PHYSICS_FIRST = "warmup_off"     # {"always_on","always_off","warmup_off"}
Q_WARMUP_EPOCHS_PHYSICS_FIRST = 5
Q_RAMP_EPOCHS_PHYSICS_FIRST = 5

SUBS_RESID_POLICY_PHYSICS_FIRST = "warmup_off"
SUBS_RESID_WARMUP_EPOCHS_PHYSICS_FIRST = 5
SUBS_RESID_RAMP_EPOCHS_PHYSICS_FIRST = 5

# Head (GWL) observation weighting and Q regularization used in physics-first.
LOSS_WEIGHT_GWL_PHYSICS_FIRST = 0.5
LAMBDA_Q_PHYSICS_FIRST = 1e-5

# ----------------------------
# B) DATA-FIRST preset (DEFAULT)
# ----------------------------
# In data-first, observations dominate early, then constraints ramp in.
LOSS_WEIGHT_GWL_DATA_FIRST = 0.8
LAMBDA_Q_DATA_FIRST = 5e-4

Q_POLICY_DATA_FIRST = "warmup_off"
Q_WARMUP_EPOCHS_DATA_FIRST = 2
Q_RAMP_EPOCHS_DATA_FIRST = 6

SUBS_RESID_POLICY_DATA_FIRST = "warmup_off"
SUBS_RESID_WARMUP_EPOCHS_DATA_FIRST = 1
SUBS_RESID_RAMP_EPOCHS_DATA_FIRST = 4

# Log extra Q/subs-residual diagnostics in train/val logs.
LOG_Q_DIAGNOSTICS = True


# -------------------------------------------------------------------
# 5.7 Bounds (constraints) for learned physical fields
# -------------------------------------------------------------------
# Bounds are specified in LINEAR space here (Stage-2 converts as needed).
# Used for:
#   - soft penalties (PHYSICS_BOUNDS_MODE="soft")
#   - optional hard clamping/rejection (PHYSICS_BOUNDS_MODE="hard")
PHYSICS_BOUNDS = {
    # Effective thickness H [m]
    "H_min": 0.1,
    "H_max": 30.0,

    # Hydraulic conductivity K [m/s]
    "K_min": 1e-12,
    "K_max": 1e-7,

    # Specific storage Ss [1/m]
    "Ss_min": 1e-6,
    "Ss_max": 1e-3,

    # Time constant tau [s]
    "tau_min": 7.0 * 86400.0,
    "tau_max": 300.0 * 31556952.0,
}

# Bounds enforcement mode:
#   - "soft" : penalize violations (recommended; keeps gradients alive)
#   - "hard" : clamp/reject (only if you know exactly why)
PHYSICS_BOUNDS_MODE = "soft"


# -------------------------------------------------------------------
# 5.8 Bounds-loss shaping (v3.2)
# -------------------------------------------------------------------
# These parameters shape the *form* of the bounds penalty (not its weight).
# The overall weight is still controlled by LAMBDA_BOUNDS.
BOUNDS_LOSS_KIND = "both"   # {"both","K","Ss","tau",...}
BOUNDS_BETA = 20.0          # barrier sharpness (larger -> harder wall)
BOUNDS_GUARD = 5.0          # guard margin (linear/log depending on impl)
BOUNDS_W = 1.0              # base weight inside bounds loss
BOUNDS_INCLUDE_TAU = True
BOUNDS_TAU_W = 1.0


# -------------------------------------------------------------------
# 5.9 Time/units hooks for physics conversions
# -------------------------------------------------------------------
# Must match the meaning of TIME_COL in your dataset.
TIME_UNITS = "year"


# -------------------------------------------------------------------
# 5.10 Debug hooks
# -------------------------------------------------------------------
DEBUG_PHYSICS_GRADS = False


# ------------------------------------------------------------
# Consolidation drawdown gating (s_eq = Ss * delta_h * H)
# ------------------------------------------------------------
# These options control how drawdown (delta_h) is computed and gated.
# They are used by:
# - equilibrium_compaction_si(...)
# - integrate_consolidation_mean(...)
# - compute_consolidation_step_residual(...)
#
# CONS_DRAWDOWN_RULE:
# - "ref_minus_mean" : delta_h = h_ref - h_mean   (default; head-loss drawdown)
# - "mean_minus_ref" : delta_h = h_mean - h_ref   (use if you are actually in depth-space)
#
# CONS_DRAWDOWN_MODE:
# - "smooth_relu" : smooth positive-part gate via softplus(beta*x)/beta (recommended)
# - "relu"        : hard relu max(x,0)
# - "softplus"    : softplus(x) (always > 0; not zero at x=0)
# - "none"        : no gating (delta_h can be negative; use carefully)
#
# CONS_STOP_GRAD_REF:
# - True  : stop-gradient on h_ref to prevent collapsing drawdown by moving the reference
# - False : allow gradients through h_ref (only if you know why)
#
# CONS_DRAWDOWN_ZERO_AT_ORIGIN:
# - If True and mode="smooth_relu", shift so output is ~0 at x=0
#   (note: can become slightly negative for x<0).
#
# CONS_DRAWDOWN_CLIP_MAX:
# - Optional float to clip delta_h after gating: delta_h = clip(delta_h, 0, clip_max)
# - None disables clipping.
#
# CONS_RELU_BETA:
# - Curvature control for smooth_relu. Larger -> closer to hard relu.
CONS_DRAWDOWN_MODE = "smooth_relu"  # "softplus"
CONS_DRAWDOWN_RULE = "ref_minus_mean"
CONS_STOP_GRAD_REF = True
CONS_DRAWDOWN_ZERO_AT_ORIGIN = False
CONS_DRAWDOWN_CLIP_MAX = None
CONS_RELU_BETA = 20.0

#XXX: Preset:
# CONS_SCALE_FLOOR = 1e-10
# GW_SCALE_FLOOR = 1e-11

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

Z_SURF_UNIT_TO_SI = 1.0
# Convert subsidence to SI (recommended):
# If SUBSIDENCE_COL is in mm or mm/yr, convert to meters or meters/yr.
SUBS_UNIT_TO_SI = 1e-3

# Raw thickness unit conversion to SI meters.
# If your thickness is already meters, keep 1.0.
THICKNESS_UNIT_TO_SI = 1.0
# Guard against zero/negative thickness & non-finite SI columns ( in meters) 

H_FIELD_MIN_SI = 0.1 

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
CONSOLIDATION_RESIDUAL_UNITS ="second"

CONS_SCALE_FLOOR =3e-11      # ~1 mm/year in m/s # "auto"
# GW_SCALE_FLOOR ="auto"
GW_RESIDUAL_UNITS = "second"
GW_SCALE_FLOOR = 1e-12        # safer than 1e-9; keeps GW residual visible

ALLOW_SUBS_RESIDUAL =True 

DT_MIN_UNITS = 1e-6

Q_WRT_NORMALIZED_TIME = False 
Q_IN_SI = False  
Q_IN_PER_SECOND = False
Q_KIND ="per_volume" 
Q_LENGTH_IN_SI=False 

DRAINAGE_MODE ="double" 
SCALING_ERROR_POLICY ="raise" 
GW_RESIDUAL_UNITS="second"

CLIP_GLOBAL_NORM = 5.0 

# ===================================================================
# 7) TRAINING LOOP DEFAULTS (non-tuner runs)
# ===================================================================
EPOCHS = 20           # Recommended: 50 to 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3   # Slightly higher start, let Adam decay it

# XXX preset:
# EPOCHS = 150
# BATCH_SIZE = 32
# LEARNING_RATE = 5e-4

VERBOSE = 1 
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

# ===================================================================
# 8) MODEL FORMAT CONFIGURATION
# ===================================================================
# If True, the model will be saved in TensorFlow format (SavedModel).
USE_TF_SAVEDMODEL = False  # Set to False to use the default weight-based saving

# ===================================================================
# 9) MODEL LOADING / DEBUGGING CONFIGURATION
# ===================================================================
# If True, load the model directly into memory rather than from disk.
USE_IN_MEMORY_MODEL = False # True  # Change to True for in-memory usage

# If True, enable debug information during training and evaluation.
DEBUG = True  # Enable or disable debugging

# ---------------------------------------------------------------------
# Auditing (Stage-1 / Stage-2 pipeline sanity checks)
# ---------------------------------------------------------------------
# Controls which audit helpers run and print/save diagnostics.
#
# Accepted values (flexible):
#   - None / "" / False / "off" / "0" :
#       Disable all audits.
#
#   - "*" / "all" / "both" / True :
#       Audit all stages (Stage-1 + Stage-2).
#
#   - "stage1" / "stage_1" / "stage-1" / "1" / "s1" :
#       Audit Stage-1 only.
#
#   - "stage2" / "stage_2" / "stage-2" / "2" / "s2" :
#       Audit Stage-2 only.
#
#   - Combined forms like:
#       "stage1,stage2", "stage1/2", "1+2", ["stage1", "stage2"]
#       Audit both Stage-1 and Stage-2.
#
# Notes
# -----
# - Stage-1 audit focuses on coordinate provenance, scaling/leakage,
#   SI-channel sanity, and feature splits.
# - Stage-2 audit focuses on NPZ tensor shapes, finiteness, coordinate
#   normalization/inversion checks, and scaling_kwargs consistency.
AUDIT_STAGES = "*"

EVAL_JSON_UNITS_MODE = "si"              # or "si / interpretable"
EVAL_JSON_UNITS_SCOPE = "all"            # "subsidence" / "physics" / "all"

# -------------------------------------------------------------------
# Optional: force scaling_kwargs from an external JSON file.
# If provided, this JSON takes precedence over any Stage-2/Stage-3
# auto-computed scaling_kwargs values.
# -------------------------------------------------------------------
SCALING_KWARGS_JSON_PATH = None   # e.g. r"F:\...\results\...\scaling_kwargs.json"

# -------------------------------------------------------------------
# 5. TUNING SEARCH SPACE
# -------------------------------------------------------------------
# Hyperparameter search space for the GeoPriorTuner.  The tuner
# script imports this as a simple dictionary and passes it to
# the tuning API.

# Notes:
# - v3.2 adds compile-time: lambda_bounds, lambda_q, lambda_offset,
#   scale_mv_with_offset, scale_q_with_offset.
# - OFFSET_MODE affects lambda_offset semantics:
#     * "mul"   : physics_mult = lambda_offset                  (must be > 0)
#     * "log10" : physics_mult = 10 ** lambda_offset            (can be < 0)
#   We set ranges so physics_mult roughly spans ~[0.1, 50] in both regimes.

TUNER_SEARCH_SPACE_BASE_V32 = {
    # ----------------------------
    # Architecture (model.__init__)
    # ----------------------------
    "embed_dim": [32, 48, 64],
    "hidden_units": [64, 96, 128],
    "lstm_units": [64, 96],
    "attention_units": [32, 48],
    "num_heads": [2, 4],
    "dropout_rate": {
        "type": "float",
        "min_value": 0.05,
        "max_value": 0.25,
        "step": 0.05,
    },
    # Optional memory knobs (keep modest; Stage2 defaults are stable)
    "max_window_size": [8, 10, 12],
    "memory_size": [50, 100],

    # ----------------------------
    # Optimizer / training
    # ----------------------------
    "learning_rate": {
        "type": "float",
        "min_value": 3e-4,
        "max_value": 3e-3,
        "sampling": "log",
    },

    # ----------------------------
    # Physics loss weights (compile-time)
    # ----------------------------
    "lambda_gw": {
        "type": "float",
        "min_value": 0.01,
        "max_value": 0.20,
    },
    "lambda_cons": {
        "type": "float",
        "min_value": 0.05,
        "max_value": 0.30,
    },
    "lambda_prior": {
        "type": "float",
        "min_value": 0.02,
        "max_value": 0.30,
    },

    # Smoothness is often tiny; log-range helps
    "lambda_smooth": {
        "type": "float",
        "min_value": 1e-5,
        "max_value": 5e-3,
        "sampling": "log",
    },

    # v3.2: bounds penalty (soft bounds). Use wide log-range.
    "lambda_bounds": {
        "type": "float",
        "min_value": 1e-4,
        "max_value": 1.0,
        "sampling": "log",
    },

    # MV prior penalty usually small (or even 0)
    "lambda_mv": {
        "type": "float",
        "min_value": 0.0,
        "max_value": 0.05,
    },

    # v3.2: Q forcing penalty (often 0; include a few “on” options)
    "lambda_q": {
        "type": "choice",
        "values": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    },

    # v3.2: per-parameter LR multipliers
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

    # v3.2: whether MV/Q terms scale with the global physics offset
    # (I usually keep these fixed in fixed_params, but they can be tuned)
    "scale_mv_with_offset": {
        "type": "choice",
        "values": [False, True],
    },
    "scale_q_with_offset": {
        "type": "choice",
        "values": [True, False],
    },
}

# Offset-mode-aware lambda_offset
if OFFSET_MODE == "mul":
    # physics_mult = lambda_offset in ~[0.1, 50] (log-sampled)
    TUNER_SEARCH_SPACE = dict(TUNER_SEARCH_SPACE_BASE_V32)
    TUNER_SEARCH_SPACE["lambda_offset"] = {
        "type": "float",
        "min_value": 0.1,
        "max_value": 50.0,
        "sampling": "log",
    }
else:
    # physics_mult = 10**lambda_offset in ~[0.1, 50]
    # lambda_offset ~ log10(mult) in [-1, 1.7]
    TUNER_SEARCH_SPACE = dict(TUNER_SEARCH_SPACE_BASE_V32)
    TUNER_SEARCH_SPACE["lambda_offset"] = {
        "type": "float",
        "min_value": -1.0,
        "max_value": 1.7,
        "step": 0.05,
    }

