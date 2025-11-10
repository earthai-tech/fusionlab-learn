# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
main_NATCOM.py: Zhongshan Land Subsidence Forecasting with GeoPriorSubsNet

This script performs physics-informed, quantile-based subsidence and GWL
prediction for Zhongshan City using the GeoPriorSubsNet model from the
fusionlab-learn library, incorporating advanced geomechanical priors
as per the revised manuscript.

Dataset Notes for Reproducibility (Zhongshan - 500k samples):
1. The script will primarily attempt to load 'zhongshan_500k.csv'
   from a 'data/' or '../data/' directory, or a specified JUPYTER_PROJECT_ROOT.
2. As a fallback for development/testing without the large dataset,
   it can use `fusionlab.datasets.fetch_zhongshan_data()` (2k samples)
   or a smaller local CSV.

Workflow:
---------
1. Configuration and Library Imports.
2. Load and Inspect Zhongshan Dataset.
3. Preprocessing: Feature Selection, Cleaning, Encoding, Scaling.
4. Define Feature Sets (Static, Dynamic, Future, Coordinates, H_field, Targets).
5. Split Master Data by Year (Train/Test).
6. Sequence Generation for PINN using `prepare_pinn_data_sequences`.
7. Create tf.data.Dataset for training and validation.
8. GeoPriorSubsNet Model Definition, Compilation, and Training with new PINN loss.
9. Forecasting on Test Data Sequences.
10. Formatting Predictions and Visualization.
11. Saving Artifacts.
"""

import os
import shutil
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Explicit import for save_all_figures # noqa
import json
import datetime as dt

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings

# --- Suppress Common Warnings ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')
if hasattr(tf, 'autograph') and hasattr(tf.autograph, 'set_verbosity'):
    tf.autograph.set_verbosity(0)

try:
    from fusionlab.api.util import get_table_size
    from fusionlab.datasets import fetch_zhongshan_data # For Zhongshan

    from fusionlab.nn.calibration import ( 
        fit_interval_calibrator_on_val, _stack_subs_quantiles, 
        apply_calibrator_to_subs, 
        )
    
    from fusionlab.nn.pinn.models import GeoPriorSubsNet
    from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences # PINN data prep
    from fusionlab.nn.pinn.op import extract_physical_parameters 
    # Import new parameter classes for GeoPriorSubsNet
    from fusionlab.params import (
        LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
    )
    from fusionlab.nn.utils import extract_batches_from_dataset 
    from fusionlab.nn.losses import combined_quantile_loss, make_weighted_pinball
    from fusionlab.nn.utils import plot_history_in 
    from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn, _to_py 
    from fusionlab.nn.pinn.utils import format_pinn_predictions 
    
    from fusionlab.plot.forecast import plot_forecasts, forecast_view 
    from fusionlab.utils.data_utils import nan_ops
    from fusionlab.utils.io_utils import save_job
    from fusionlab.utils.forecast_utils import get_test_data_from, adjust_time_predictions 
    from fusionlab.utils.generic_utils import save_all_figures, normalize_time_column 
    from fusionlab.utils.generic_utils import ensure_directory_exists
    
    
    print("Successfully imported fusionlab modules.")
except ImportError as e:
    print(f"Critical Error: Failed to import fusionlab modules: {e}. "
          "Please ensure 'fusionlab-learn' is correctly installed.")
    raise
    
#%
# ==================================================================
# ** Step 0: CONFIGURATION PARAMETERS **
# ==================================================================
CITY_NAME = 'nansha' # or "zhongshan"
MODEL_NAME ='GeoPriorSubsNet' # Use the new model
H_FIELD_COL_NAME = 'soil_thickness' # Required input for GeoPriorSubsNet

# Data loading: Prioritize 500k sample file
DATA_DIR = os.getenv("JUPYTER_PROJECT_ROOT", "..") # Go up one level from script if not set
ZHONGSHAN_500K_FILENAME = f"{CITY_NAME}_500k.csv" # Target file
ZHONGSHAN_2K_FILENAME = f"{CITY_NAME}_2000.csv" # Smaller fallback

data_paths_to_check = [
    os.path.join(DATA_DIR, "data", ZHONGSHAN_500K_FILENAME),
    os.path.join(DATA_DIR, ZHONGSHAN_500K_FILENAME),
    os.path.join(".", "data", ZHONGSHAN_500K_FILENAME),
    ZHONGSHAN_500K_FILENAME,
]

# Training and Forecasting Periods
TRAIN_END_YEAR = 2021        # Example: Use data up to 2022 for training
FORECAST_START_YEAR = 2022   # Example: Start forecasting for 2023
FORECAST_HORIZON_YEARS = 3 # 4   # Example: Predict 4 years ahead (2023, 2024, 2025, 2026)

# --- Time Series Sequence Configuration ---
# The look-back window (`TIME_STEPS`) is a critical hyperparameter.
# A larger window allows the model to capture longer-term trends
# and seasonality, but it requires that each individual time series
# in the dataset (e.g., for each unique longitude/latitude pair)
# has a sufficient number of historical data points.
#
# Dataset Context:
# - The Nansha dataset spans 8 years (2015-2022).
# - The Zhongshan dataset spans 9 years (2015-2023).
#
# For the full datasets used in our paper, the optimal values
# were determined through hyperparameter tuning:
# - Zhongshan (4.5M samples): TIME_STEPS = 6
# - Nansha (2.4M samples):   TIME_STEPS = 5
#
# For your own dataset, you can programmatically find the maximum
# valid look-back window using the `resolve_time_steps` utility
# from the library to avoid generating empty training sets: 
#   >>> from fusionlab.utils.ts_utils import resolve_time_steps
#

# Use TIME_STEPS = 6 for Zhongshan as per paper tuning
TIME_STEPS = 4# 6           # Lookback window (in years) for dynamic features

# PINN Configuration
PDE_MODE_CONFIG ='both' # 'consolidation', 'gw_flow', 'both', or 'none'

# New GeoPriorSubsNet Physics Loss Weights 
LAMBDA_CONS = 1.0
LAMBDA_GW = 1.0
# a much more stable starting point
LAMBDA_PRIOR = 1.0 # 0.5 → 1.0  #0.001  # Drastically reduce the unstable term
LAMBDA_SMOOTH = 1.0  # Reduce smoothing a bit as well

LAMBDA_MV=0.1 
MV_LR_MULT=1.0
KAPPA_LR_MULT=5.0 #    # 5–20  
    
# New GeoPriorSubsNet Scalar Parameters 
GEOPRIOR_INIT_MV = 1e-7      # Learnable m_v
GEOPRIOR_INIT_KAPPA = 1.0    # Learnable kappa_bar
GEOPRIOR_GAMMA_W = 9810.0    # Fixed gamma_w
GEOPRIOR_H_REF = 0.0         # Fixed h_ref

GEOPRIOR_KAPPA_MODE ="bar" 
GEOPRIOR_USE_EFFECTIVE_H= True # False 
GEOPRIOR_HD_FACTOR =0.6 # 1.0 # 0.6–0.8

# Model Hyperparameters
QUANTILES = [0.1, 0.5, 0.9] # For probabilistic forecast
# QUANTILES = None # For point forecast

# Example: heavier tails than the median (tweak as you like)
SUBS_WEIGHTS = {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
GWL_WEIGHTS  = {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}

EPOCHS = 50 # 100 # For demonstration; increase for robust results (e.g., 100-200)
LEARNING_RATE = 0.0001
BATCH_SIZE = 32 # 256 # Adjusted for potentially larger dataset

NUM_BATCHES_TO_EXTRACT = "auto" # 'auto' to extract all from val_dataset if test seq fails
AGG = True # Aggregate batches extracted from val_dataset

MODE ='tft_like' # Model operation mode: {'pihal_like', 'tft_like'}

# Attention levels to use for the model
ATTENTION_LEVELS = ['cross', 'hierarchical', 'memory'] # Use all three

# Output Directories
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "results") # For Code Ocean compatibility
ensure_directory_exists(BASE_OUTPUT_DIR)
RUN_OUTPUT_PATH = os.path.join(
    BASE_OUTPUT_DIR, f"{CITY_NAME}_{MODEL_NAME}_run"
)
if os.path.isdir(RUN_OUTPUT_PATH) and os.path.exists(RUN_OUTPUT_PATH):
    print(f"Cleaning up existing run directory: {RUN_OUTPUT_PATH}")
    shutil.rmtree(RUN_OUTPUT_PATH)
os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)
print(f"Output artifacts will be saved in: {RUN_OUTPUT_PATH}")

SAVE_INTERMEDIATE_ARTEFACTS = True
SAVE_MODEL_AS_SINGLE_FILE = True # .keras format (preferred)
INCLUDE_GWL_IN_DF = False # Include Gwl in prediction results
# --- Display Settings ---
try: _TW = get_table_size()
except: _TW = 80 # Default terminal width
print(f"\n{'-'*_TW}\n{CITY_NAME.upper()} {MODEL_NAME} FORECASTING (PINN)\n{'-'*_TW}")
print(f"Configuration: TIME_STEPS={TIME_STEPS}, "
      f"FORECAST_HORIZON={FORECAST_HORIZON_YEARS} years.")
FORECAST_YEARS_LIST = [
    FORECAST_START_YEAR + i for i in range(FORECAST_HORIZON_YEARS)
]
print(f"Forecasting for years: {FORECAST_YEARS_LIST}")


# ==================================================================
# ** Step 1: Load Zhongshan Dataset **
# ==================================================================
print(f"\n{'='*20} Step 1: Load Zhongshan Dataset {'='*20}")
zhongshan_df_raw = None
fallback_paths = [
    os.path.join(DATA_DIR, "data", ZHONGSHAN_2K_FILENAME),
    os.path.join(DATA_DIR, ZHONGSHAN_2K_FILENAME),
    os.path.join(".", "data", ZHONGSHAN_2K_FILENAME),
    ZHONGSHAN_2K_FILENAME
]

for path_option in data_paths_to_check + fallback_paths:
    print(f"Attempting to load data from: {os.path.abspath(path_option)}")
    if os.path.exists(path_option):
        try:
            zhongshan_df_raw = pd.read_csv(path_option)
            print(f"  Successfully loaded '{os.path.basename(path_option)}'. "
                  f"Shape: {zhongshan_df_raw.shape}")
            if ZHONGSHAN_500K_FILENAME in path_option and zhongshan_df_raw.shape[0] < 400_000: # Basic check
                print(f"  Warning: Loaded file named {ZHONGSHAN_500K_FILENAME} but "
                      f"it has only {zhongshan_df_raw.shape[0]} rows.")
            break
        except Exception as e_load:
            print(f"  Error loading '{path_option}': {e_load}")
    else:
        print(f"  Path not found: {os.path.abspath(path_option)}")

if zhongshan_df_raw is None or zhongshan_df_raw.empty:
    print("No local CSV found. Attempting to fetch via fusionlab.datasets...")
    try:
        data_bunch = fetch_zhongshan_data(verbose=1)
        zhongshan_df_raw = data_bunch.frame
        print(f"  Zhongshan data loaded via fetch_zhongshan_data. "
              f"Shape: {zhongshan_df_raw.shape}")
    except Exception as e_fetch:
        raise FileNotFoundError(
            "Zhongshan data could not be loaded from local paths or fetched. "
            f"Last fetch error: {e_fetch}. Please ensure data is available."
        )

if SAVE_INTERMEDIATE_ARTEFACTS:
    raw_data_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_01_raw_data.csv")
    zhongshan_df_raw.to_csv(raw_data_path, index=False)
    print(f"  Raw data saved to: {raw_data_path}")

# ==================================================================
# ** Step 2: Preprocessing - Feature Selection & Initial Cleaning **
# ==================================================================
print(f"\n{'='*20} Step 2: Preprocessing - Initial Steps {'='*20}")

TIME_COL = 'year' # Will be converted to numeric for PINN 't'
LON_COL = 'longitude'
LAT_COL = 'latitude'
SUBSIDENCE_COL = 'subsidence'
GWL_COL = 'GWL'
# H_FIELD_COL_NAME is defined in Step 0

# Select relevant features for the model
available_cols = zhongshan_df_raw.columns.tolist()
selected_features_base = [
    LON_COL, LAT_COL, TIME_COL, SUBSIDENCE_COL, GWL_COL, 'rainfall_mm',
    'geology', 
    H_FIELD_COL_NAME, # CRITICAL: Ensure soil_thickness is selected
    'normalized_density', # Example, building density
]
selected_features = [
    col for col in selected_features_base if col in available_cols
    ]
missing_selection = set(selected_features_base) - set(selected_features)
if missing_selection:
    print(f"  Warning: Some initially selected features were not found "
          f"in the loaded data and will be skipped: {missing_selection}")
    if H_FIELD_COL_NAME in missing_selection:
        raise ValueError(
            f"CRITICAL: Required soil thickness column "
            f"'{H_FIELD_COL_NAME}' not found in the data."
        )

df_selected = zhongshan_df_raw[selected_features].copy()

# Convert year to datetime for consistent processing, then to numeric for PINN
DT_COL_NAME_TEMP = 'datetime_temp' # Temporary datetime column

try: 
    df_selected[DT_COL_NAME_TEMP] = pd.to_datetime(
        df_selected[TIME_COL], format='%Y'
    )
except: 
    df_selected = normalize_time_column(
        df_selected, time_col= TIME_COL, 
        datetime_col= DT_COL_NAME_TEMP, 
        year_col= TIME_COL,
        drop_orig= True 
    )

print(f"  Initial features selected. Shape: {df_selected.shape}")

# Cleaning NaNs
print(f"NaNs before cleaning: {df_selected.isna().sum().sum()}")
cols_for_nan_ops = [
    c for c in df_selected.columns if c != DT_COL_NAME_TEMP
    ] 
df_cleaned = nan_ops(
    df_selected, ops='sanitize', action='fill', verbose=0,
    # subset=cols_for_nan_ops # Apply only to relevant columns
    )
print(f"NaNs after cleaning: {df_cleaned.isna().sum().sum()}")

if SAVE_INTERMEDIATE_ARTEFACTS:
    cleaned_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_02_cleaned_data.csv")
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"  Cleaned data saved to: {cleaned_path}")

# ==================================================================
# ** Step 3: Preprocessing - Encoding & Scaling **
# ==================================================================
print(f"\n{'='*20} Step 3: Preprocessing - Encoding & Scaling {'='*20}")
df_for_processing = df_cleaned.copy()

# --- Encoding Categorical Features ---
categorical_cols_to_encode = ['geology']
categorical_cols_to_encode = [
    c for c in categorical_cols_to_encode if c in df_for_processing.columns
    ]
encoded_feature_names_list = []

if categorical_cols_to_encode:
    encoder_ohe = OneHotEncoder(
        sparse_output=False, handle_unknown='ignore', dtype=np.float32
    )
    encoded_data_parts = encoder_ohe.fit_transform(
        df_for_processing[categorical_cols_to_encode]
    )
    encoder_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_ohe_encoder.joblib")
    try: 
        save_job(encoder_ohe, encoder_path, append_versions=True, 
            append_date=True)
    except: 
        joblib.dump(encoder_ohe, encoder_path)
        
    print(f"  OneHotEncoder saved to: {encoder_path}")

    new_ohe_cols = encoder_ohe.get_feature_names_out(
        categorical_cols_to_encode
    )
    encoded_feature_names_list.extend(new_ohe_cols)
    encoded_df_part = pd.DataFrame(
        encoded_data_parts, columns=new_ohe_cols, index=df_for_processing.index
    )
    df_for_processing = pd.concat(
        [df_for_processing.drop(columns=categorical_cols_to_encode), 
         encoded_df_part], axis=1
    )
    print(f"  Encoded features created: {new_ohe_cols.tolist()}")

# --- Numerical Time Coordinate for PINN ---
df_for_processing[f"{TIME_COL}_numeric_coord"] = (
    df_for_processing[DT_COL_NAME_TEMP].dt.year +
    (df_for_processing[DT_COL_NAME_TEMP].dt.dayofyear - 1) /
    (365 + df_for_processing[DT_COL_NAME_TEMP].dt.is_leap_year.astype(int))
)
TIME_COL_NUMERIC_PINN = f"{TIME_COL}_numeric_coord"

# --- Scaling Numerical Features ---
# numerical_cols_for_scaling_model = [
#     GWL_COL, 
#     'rainfall_mm',
#     H_FIELD_COL_NAME, 
#     'normalized_density', 
#     SUBSIDENCE_COL, 
# ]
# numerical_cols_for_scaling_model = list(set( # Unique existing cols
#     [c for c in numerical_cols_for_scaling_model if c in df_for_processing.columns]
# ))

# df_scaled = df_for_processing.copy()
# scaler_main = MinMaxScaler()
# if numerical_cols_for_scaling_model:
#     df_scaled[numerical_cols_for_scaling_model] = scaler_main.fit_transform(
#         df_scaled[numerical_cols_for_scaling_model]
#     )
#     scaler_path = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_main_scaler.joblib")
#     try: 
#         save_job( scaler_main, scaler_path)
#     except: 
#         joblib.dump(scaler_main, scaler_path)
        
#     print(f"  Numerical features scaled. Scaler saved to: {scaler_path}")
# --- Scaling Numerical Features ---
numerical_cols_for_scaling_model = [
    GWL_COL, 
    'rainfall_mm',
    H_FIELD_COL_NAME, 
    'normalized_density', 
    SUBSIDENCE_COL, 
]
# Filter to existing columns *preserving order* (list(set()) is dangerous)
numerical_cols_for_scaling_model = [
    c for c in numerical_cols_for_scaling_model if c in df_for_processing.columns
]

df_scaled = df_for_processing.copy()
scaler_main = MinMaxScaler()
scaler_info_dict = None # Initialize
if numerical_cols_for_scaling_model:
    df_scaled[numerical_cols_for_scaling_model] = scaler_main.fit_transform(
        df_scaled[numerical_cols_for_scaling_model]
    )
    scaler_path = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_main_scaler.joblib")
    try: 
        save_job(scaler_main, scaler_path)
    except: 
        joblib.dump(scaler_main, scaler_path)
    print(f"  Numerical features scaled. Scaler saved to: {scaler_path}")

    # --- *** ADD THIS BLOCK *** ---
    # Create the dictionary for the formatter
    print("  Constructing scaler_info for inverse transform...")
    scaler_info_dict = {}
    feature_list_for_scaler = numerical_cols_for_scaling_model
    
    # These base names MUST match the *values* in target_mapping in Step 8
    targets_to_inverse = {
        SUBSIDENCE_COL: SUBSIDENCE_COL, # 'subsidence': 'subsidence'
        GWL_COL: GWL_COL                # 'GWL': 'GWL'
    }
    
    for base_name, col_name_in_scaler in targets_to_inverse.items():
        if col_name_in_scaler in feature_list_for_scaler:
            col_index = feature_list_for_scaler.index(col_name_in_scaler)
            scaler_info_dict[base_name] = {
                'scaler': scaler_main,
                'all_features': feature_list_for_scaler,
                'idx': col_index
            }
            print(f"    - Mapping target '{base_name}' to feature "
                  f"'{col_name_in_scaler}' at index {col_index}")
        else:
            print(f"    - [WARN] Target '{base_name}' (from "
                  f"'{col_name_in_scaler}') not found in scaler's feature list.")
            
    # --- *** END OF ADDED BLOCK *** ---
    
if SAVE_INTERMEDIATE_ARTEFACTS:
    scaled_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_03_processed_scaled_data.csv")
    df_scaled.to_csv(scaled_path, index=False)
    print(f"  Processed and scaled data saved to: {scaled_path}")

# ==================================================================
# ** Step 4: Define Feature Sets for `prepare_pinn_data_sequences` **
# ==================================================================
print(f"\n{'='*20} Step 4: Define Feature Sets for PINN Data Prep {'='*20}")

# Static features
static_features_list = encoded_feature_names_list # Geology, etc.

# Dynamic features: vary over `time_steps` (past observed)
dynamic_features_list = [
    GWL_COL, 'rainfall_mm', 'normalized_density', 
]
dynamic_features_list = [
    c for c in dynamic_features_list if c in df_scaled.columns
    ]

# Future known features: known for the `forecast_horizon`
future_features_list = ['rainfall_mm'] # Example
future_features_list = [
    c for c in future_features_list if c in df_scaled.columns
    ]

# Soil thickness column (H_field)
H_FIELD_COL = H_FIELD_COL_NAME # 'soil_thickness'

# Grouping columns for sequence generation
GROUP_ID_COLS_SEQ = [LON_COL, LAT_COL] 

print(f"  PINN Time Coordinate: {TIME_COL_NUMERIC_PINN}")
print(f"  PINN Lon Coordinate: {LON_COL}")
print(f"  PINN Lat Coordinate: {LAT_COL}")
print(f"  Target 1 (Subsidence): {SUBSIDENCE_COL}")
print(f"  Target 2 (GWL): {GWL_COL}")
print(f"  H_field (Soil Thickness): {H_FIELD_COL}")
print(f"  Static Features: {static_features_list}")
print(f"  Dynamic Features: {dynamic_features_list}")
print(f"  Future Features: {future_features_list}")
print(f"  Group ID Cols for Sequencing: {GROUP_ID_COLS_SEQ}")

# ==================================================================
# ** Step 5: Split Master Data & Generate Sequences for Training **
# ==================================================================
print(f"\n{'='*20} Step 5: Split Master Data & PINN Sequence Generation {'='*20}")
df_train_master = df_scaled[
    df_scaled[TIME_COL] <= TRAIN_END_YEAR
].copy()

df_test_master = get_test_data_from(
    df_scaled.copy(), 
    time_col=TIME_COL, 
    time_steps=TIME_STEPS , 
    train_end_year= TRAIN_END_YEAR, 
    forecast_horizon= FORECAST_HORIZON_YEARS ,
    verbose = 3, 
    strategy='onwards',
    objective="forecasting", 
)
print(f"Master train data shape (<= {TRAIN_END_YEAR}): {df_train_master.shape}")
print(f"Master test data shape (for {FORECAST_START_YEAR} onwards): {df_test_master.shape}")

if df_train_master.empty:
    raise ValueError(f"Training data empty after split at year {TRAIN_END_YEAR}.")

sequence_file_path_train = os.path.join(
    RUN_OUTPUT_PATH,
    f'{CITY_NAME}_train_pinn_sequences_T{TIME_STEPS}_H{FORECAST_HORIZON_YEARS}.joblib'
)

print(f"Generating PINN training sequences (T={TIME_STEPS}, H={FORECAST_HORIZON_YEARS})...")

OUT_S_DIM = 1 
OUT_G_DIM = 1
inputs_train_dict, targets_train_dict, coord_scaler = prepare_pinn_data_sequences(
    df=df_train_master,
    time_col=TIME_COL_NUMERIC_PINN,
    lon_col=LON_COL,
    lat_col=LAT_COL,
    subsidence_col=SUBSIDENCE_COL,
    gwl_col=GWL_COL,
    h_field_col=H_FIELD_COL, # *** NEW: Pass H_field column ***
    dynamic_cols=dynamic_features_list,
    static_cols=static_features_list,
    future_cols=future_features_list,
    group_id_cols=GROUP_ID_COLS_SEQ,
    time_steps=TIME_STEPS,
    forecast_horizon=FORECAST_HORIZON_YEARS,
    output_subsidence_dim=OUT_S_DIM,
    output_gwl_dim=OUT_G_DIM,    
    normalize_coords=True, 
    savefile=sequence_file_path_train, 
    return_coord_scaler= True, 
    mode=MODE, 
    model=MODEL_NAME, # *** NEW: Specify model type ***
    verbose=7 
)

if targets_train_dict['subsidence'].shape[0] == 0:
    raise ValueError("Sequence generation produced no training samples.")

print("Training sequences generated successfully.")
for name, arr in inputs_train_dict.items():
    print(f"  Train Input '{name}' shape: {arr.shape if arr is not None else 'None'}")
for name, arr in targets_train_dict.items():
    print(f"  Train Target '{name}' shape: {arr.shape}")

# ==================================================================
# ** Step 6: Create tf.data.Datasets for Training and Validation **
# ==================================================================
print(f"\n{'='*20} Step 6: Create TensorFlow Datasets {'='*20}")

num_train_samples = inputs_train_dict['dynamic_features'].shape[0]

# *** CRITICAL FIX for 'tft_like' mode future features shape ***
future_time_dim_dummy = TIME_STEPS + FORECAST_HORIZON_YEARS if MODE == 'tft_like' \
                        else FORECAST_HORIZON_YEARS
                        
dataset_inputs = {
    'coords': inputs_train_dict['coords'],
    'dynamic_features': inputs_train_dict['dynamic_features'],
    'static_features': inputs_train_dict.get('static_features') if inputs_train_dict.get(
        'static_features') is not None \
        else np.zeros((num_train_samples, 0), dtype=np.float32),
    'future_features': inputs_train_dict.get('future_features') if inputs_train_dict.get(
        'future_features') is not None \
        else np.zeros((num_train_samples, future_time_dim_dummy, 0), dtype=np.float32),
    'H_field': inputs_train_dict['H_field'] # *** NEW: Add H_field ***
}

dataset_targets = {
    'subs_pred': targets_train_dict['subsidence'], # Key for subsidence output
    'gwl_pred': targets_train_dict['gwl']          # Key for GWL output
}

full_dataset = tf.data.Dataset.from_tensor_slices(
    (dataset_inputs, dataset_targets)
)

total_size = num_train_samples
val_size = int(0.2 * total_size)
train_size = total_size - val_size

full_dataset = full_dataset.shuffle(buffer_size=total_size, seed=42)
train_dataset = full_dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = full_dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"  Training dataset created with {train_size} samples.")
print(f"  Validation dataset created with {val_size} samples.")
for x_batch, y_batch in train_dataset.take(1):
    print("  Sample batch input keys:", list(x_batch.keys()))
    for k,v in x_batch.items(): print(f"    {k}: {v.shape}")
    print("  Sample batch target keys:", list(y_batch.keys()))
    for k,v in y_batch.items(): print(f"    {k}: {v.shape}")
    break

# ==================================================================
# ** Step 7: GeoPriorSubsNet Model Definition, Compilation & Training **
# ==================================================================
print(f"\n{'='*20} Step 7: {MODEL_NAME} Model Training {'='*20}")

s_dim_model = dataset_inputs['static_features'].shape[-1]
d_dim_model = dataset_inputs['dynamic_features'].shape[-1]
f_dim_model = dataset_inputs['future_features'].shape[-1]

subsmodel_params = {
    'embed_dim': 32,
    'hidden_units': 64,
    'lstm_units': 64,
    'attention_units': 64,
    'num_heads': 2,
    'dropout_rate': 0.1,
    'max_window_size': TIME_STEPS,
    'memory_size': 50,
    'scales': [1, 2], 
    'multi_scale_agg': 'last',
    'final_agg': 'last',
    'use_residuals': True,
    'use_batch_norm': False,
    'use_vsn': True, 
    'vsn_units': 32, 
    'mode': MODE, 
    'attention_levels': ATTENTION_LEVELS,
    'scale_pde_residuals': True, # For GeoPriorSubsNet
}

if MODEL_NAME == "GeoPriorSubsNet": 
    SubsModel = GeoPriorSubsNet 
    # Add new scalar physics parameters to the params dict
    subsmodel_params.update({ 
        "mv": LearnableMV(initial_value=GEOPRIOR_INIT_MV), 
        "kappa": LearnableKappa(initial_value=GEOPRIOR_INIT_KAPPA), 
        "gamma_w": FixedGammaW(value=GEOPRIOR_GAMMA_W), 
        "h_ref": FixedHRef(value=GEOPRIOR_H_REF),
        "kappa_mode": GEOPRIOR_KAPPA_MODE,  
        "use_effective_h": GEOPRIOR_USE_EFFECTIVE_H, 
        "hd_factor": GEOPRIOR_HD_FACTOR,  

    })
    
    # Define the new loss weights for compilation
    physics_loss_weights = {
        "lambda_cons": LAMBDA_CONS, 
        "lambda_gw": LAMBDA_GW,
        "lambda_prior": LAMBDA_PRIOR,
        "lambda_smooth": LAMBDA_SMOOTH,
        'lambda_mv':LAMBDA_MV,         
        'mv_lr_mult':MV_LR_MULT,         
        'kappa_lr_mult': KAPPA_LR_MULT,       
    }

else: 
    raise ValueError(f"This script is configured for GeoPriorSubsNet, "
                     f"but MODEL_NAME is set to '{MODEL_NAME}'.")

subs_model_inst = SubsModel(
    static_input_dim=s_dim_model,
    dynamic_input_dim=d_dim_model,
    future_input_dim=f_dim_model,
    output_subsidence_dim=OUT_S_DIM,
    output_gwl_dim=OUT_G_DIM,     
    forecast_horizon=FORECAST_HORIZON_YEARS,
    quantiles=QUANTILES,
    pde_mode=PDE_MODE_CONFIG,
    **subsmodel_params
)

# Build the model
for x_sample_build, _ in train_dataset.take(1):
    subs_model_inst(x_sample_build) 
    break 
subs_model_inst.summary(line_length=110, expand_nested=True)


# Compile SubsModel
loss_dict = {
    'subs_pred': tf.keras.losses.MeanSquaredError() if QUANTILES is None \
                 else make_weighted_pinball(QUANTILES, SUBS_WEIGHTS), # combined_quantile_loss(QUANTILES),
    'gwl_pred': tf.keras.losses.MeanSquaredError() if QUANTILES is None \
                else make_weighted_pinball(QUANTILES, GWL_WEIGHTS), # combined_quantile_loss(QUANTILES)
}
metrics_dict = {
    'subs_pred': ['mae', 'mse'],
    'gwl_pred': ['mae', 'mse']
}
if QUANTILES:
    # Add probabilistic metrics on the quantile output tensor
    metrics_dict['subs_pred'] += [coverage80_fn, sharpness80_fn]
    # If you also want them for GWL, uncomment:
    # metrics_dict['gwl_pred']  += [coverage80_fn, sharpness80_fn]
    
loss_weights_dict = { 
    'subs_pred': 1.0,
    'gwl_pred': 0.5 # Example: GWL data loss is half as important
}

subs_model_inst.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss=loss_dict,
    metrics=metrics_dict,
    loss_weights=loss_weights_dict,
    **physics_loss_weights # Pass new loss weights
)
print(f"{MODEL_NAME} model compiled successfully.")

# Callbacks
early_stopping_cb = EarlyStopping(
    monitor='val_loss', # Keras default total val loss
    patience=15, 
    restore_best_weights=True,
    verbose=1
)
checkpoint_filename = (
    f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_HORIZON_YEARS}"
)
if SAVE_MODEL_AS_SINGLE_FILE:
    checkpoint_filename += ".keras"
model_checkpoint_path = os.path.join(RUN_OUTPUT_PATH, checkpoint_filename)

model_checkpoint_cb = ModelCheckpoint(
    filepath=model_checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False, # Save entire model
    verbose=1
)
print(f"Model checkpoints will be saved to: {model_checkpoint_path}")

print(f"\nStarting {MODEL_NAME} model training...")
history = subs_model_inst.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping_cb, model_checkpoint_cb],
    verbose=1 
)
print(f"Best validation total_loss achieved: "
      f"{min(history.history.get('val_loss', [np.inf])):.4f}")



# 2a) Fit calibrator on validation split (target 80% coverage)
cal80 = fit_interval_calibrator_on_val(
    subs_model_inst, val_dataset, target=0.80)

# Optionally save factors per horizon so runs are reproducible
np.save(os.path.join(RUN_OUTPUT_PATH, "interval_factors_80.npy"), cal80.factors_)


# --- Optional Post-Training Utilities ---
subsmodel_metrics = {
    "Total Loss": ["loss", "val_loss"],
    "Physics Loss": ["physics_loss", "val_physics_loss"],
    "Data Loss": ["data_loss", "val_data_loss"],
    "Component Losses": [
        "consolidation_loss", "val_consolidation_loss",
        "gw_flow_loss", "val_gw_flow_loss",
        "prior_loss", "val_prior_loss",
        "smooth_loss", "val_smooth_loss"
    ],
    "Subsidence MAE": ["subs_pred_mae", "val_subs_pred_mae"],
    "GWL MAE": ["gwl_pred_mae", "val_gwl_pred_mae"],
}

# Define the scale for each plot ---
plot_scales = {
    "Total Loss": "log",
    "Physics Loss": "log",
    "Data Loss": "log",
    "Component Losses": "log",
    "Subsidence MAE": "linear", # Keep MAE linear to see it approach 0
    "GWL MAE": "linear"        # Keep MAE linear
}

print("\n--- SubsModel Training History on Separate Subplots ---")
plot_history_in(
    history.history,
    metrics=subsmodel_metrics,
    layout='single',
    title=f"{MODEL_NAME} Training History",
    yscale_settings=plot_scales, 
    savefig=os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_{MODEL_NAME.lower()}_training_history_plot",
    ),
)
# 2) Extract Physical Parameters to CSV
extract_physical_parameters(
    subs_model_inst,
    to_csv=True,
    filename=f"{CITY_NAME}_{MODEL_NAME.lower()}_physical_parameters.csv",
    save_dir=RUN_OUTPUT_PATH,
    model_name='geoprior' # Specify new model type
)

# Load the best model saved by ModelCheckpoint
print(f"\nLoading best model from checkpoint: {model_checkpoint_path}")

# --- Define custom objects needed for loading GeoPriorSubsNet ---
custom_objects_load = {
    'GeoPriorSubsNet': GeoPriorSubsNet,
    'LearnableMV': LearnableMV,
    'LearnableKappa': LearnableKappa,
    'FixedGammaW': FixedGammaW,
    'FixedHRef': FixedHRef,
}
if QUANTILES:
    # # Add the *specific instance* of the loss function used
    # custom_objects_load['combined_quantile_loss'] = combined_quantile_loss(QUANTILES)
    custom_objects_load['make_weighted_pinball'] = make_weighted_pinball


# try:
#     with custom_object_scope(custom_objects_load):
#         subs_model_loaded = load_model(model_checkpoint_path)
#     print("Best GeoPriorSubsNet model loaded successfully for forecasting.")
# except Exception as e_load:
#     print(f"Error loading saved GeoPriorSubsNet model: {e_load}. "
#           "Using model from end of training.")
#     subs_model_loaded = subs_model_inst
try:
    # Load for inference only; keep training instance for evaluation
    with custom_object_scope(custom_objects_load):
        subs_model_loaded = load_model(model_checkpoint_path, compile=False)
    print("Best GeoPriorSubsNet model loaded successfully for forecasting (compile=False).")
except Exception as e_load:
    print(f"Error loading saved GeoPriorSubsNet model: {e_load}. Using model from end of training.")
    subs_model_loaded = subs_model_inst
    
# =============================================================================
# ** Step 8: Forecasting on Test Data **
# =============================================================================
forecast_df = None
print(f"\n{'='*20} Step 8: Forecasting on Test Data {'='*20}")

inputs_test_dict = None
targets_test_dict_for_eval = None
dataset_name_for_forecast = "Unknown"
test_coord_scaler = None # Placeholder for recovering coordinate scaler.
try:
    if df_test_master.empty:
        raise ValueError("Test data (df_test_master) is empty.")

    print(f"Attempting to generate PINN sequences from test data (year {FORECAST_START_YEAR})...")
    inputs_test_dict, targets_test_dict_raw, test_coord_scaler = prepare_pinn_data_sequences(
        df=df_scaled, # df_test_master, # Use the future-dated data
        time_col=TIME_COL_NUMERIC_PINN,
        lon_col=LON_COL, lat_col=LAT_COL,
        subsidence_col=SUBSIDENCE_COL,
        gwl_col=GWL_COL,
        h_field_col=H_FIELD_COL, # *** NEW ***
        dynamic_cols=dynamic_features_list,
        static_cols=static_features_list,
        future_cols=future_features_list,
        group_id_cols=GROUP_ID_COLS_SEQ,
        time_steps=TIME_STEPS,
        forecast_horizon=FORECAST_HORIZON_YEARS,
        output_subsidence_dim=OUT_S_DIM,
        output_gwl_dim=OUT_G_DIM,
        normalize_coords=True, 
        return_coord_scaler= True, 
        mode=MODE, 
        model=MODEL_NAME, # *** NEW ***
        verbose=1
    )

    if targets_test_dict_raw['subsidence'].shape[0] == 0:
        raise ValueError("No test sequences were generated from the provided test data.")

    print("Test sequences generated successfully.")
    dataset_name_for_forecast = f"TestSet (Year {FORECAST_START_YEAR})"
    targets_test_dict_for_eval = {
        'subs_pred': targets_test_dict_raw['subsidence'],
        'gwl_pred': targets_test_dict_raw['gwl']
    }

except Exception as e:
    print(f"\n[WARNING] Could not generate test sequences due to an error: {e}")
    print("Falling back to use the validation dataset for forecasting demonstration.")
    
    dataset_name_for_forecast = "ValidationSet_Fallback"
    print(f"Attempting to extract batches from validation set (AGG={AGG})...")

    extracted_validation_data = extract_batches_from_dataset(
        val_dataset,
        num_batches_to_extract=NUM_BATCHES_TO_EXTRACT,
        errors='warn', 
        agg=AGG, 
    )

    inputs_test_dict = None
    targets_test_dict_for_eval = None

    if AGG:
        if extracted_validation_data is not None:
            print("Successfully extracted and aggregated validation batch(es).")
            if isinstance(extracted_validation_data, tuple) and \
               len(extracted_validation_data) >= 2:
                inputs_test_dict = extracted_validation_data[0]
                targets_test_dict_for_eval = extracted_validation_data[1]
            elif isinstance(extracted_validation_data, tuple) and \
                 len(extracted_validation_data) == 1:
                inputs_test_dict = extracted_validation_data[0]
                print("[WARNING] Aggregated data only contained inputs. Targets will be None.")
            else:
                print(f"[ERROR] Aggregated data has unexpected structure: "
                      f"{type(extracted_validation_data)}. Cannot unpack.")
        else:
            print("[WARNING] Aggregation returned None. No validation data extracted.")
    else:
        if extracted_validation_data: 
            print(f"Successfully extracted {len(extracted_validation_data)} "
                  "validation batch(es) (no aggregation). Using the first one.")
            first_batch = extracted_validation_data[0]
            if isinstance(first_batch, tuple) and len(first_batch) >= 2:
                inputs_test_dict = first_batch[0]
                targets_test_dict_for_eval = first_batch[1]
            elif isinstance(first_batch, tuple) and len(first_batch) == 1:
                inputs_test_dict = first_batch[0]
                print("[WARNING] First extracted batch only contained inputs. "
                      "Targets will be None.")
            else:
                print(f"[ERROR] First extracted batch has unexpected structure: "
                      f"{type(first_batch)}. Cannot unpack.")
        else:
            print("[WARNING] Extraction returned an empty list. No validation data extracted.")

    if inputs_test_dict is None:
        print("[INFO] Fallback to val_dataset.take(1) as primary extraction failed.")
        for x_val_sample_fallback, y_val_sample_fallback in val_dataset.take(1):
            inputs_test_dict = x_val_sample_fallback
            targets_test_dict_for_eval = y_val_sample_fallback
            print("[INFO] Successfully extracted one batch using val_dataset.take(1).")
            break
        else: 
            print("[ERROR] Critical Fallback failed: Could not extract any "
                  "batches from the validation dataset.")

if inputs_test_dict is not None:
    print(f"\nGenerating {MODEL_NAME} predictions on: {dataset_name_for_forecast}...")
    
    # Ensure H_field is present if using validation fallback
    if 'H_field' not in inputs_test_dict:
        print("[ERROR] 'H_field' is missing from the test/validation input dictionary. "
              "GeoPriorSubsNet cannot predict without it.")
    else:

        # 1. Get the nested dictionary from the model
        subsmodel_test_outputs_dict = subs_model_loaded.predict(
            inputs_test_dict, verbose=0
        )
        
        # 2. Extract the combined data tensor
        data_final_tensor = subsmodel_test_outputs_dict['data_final']
        
        # 3. Manually split the tensor, just like in the model's train_step
        #    We use the dimensions stored in the *loaded model* instance.
        s_dim = subs_model_loaded.output_subsidence_dim
        
        if QUANTILES:
            # Shape is (B, H, Q, O_total)
            s_pred_tensor = data_final_tensor[..., :s_dim]
            h_pred_tensor = data_final_tensor[..., s_dim:]
        else:
            # Shape is (B, H, O_total)
            s_pred_tensor = data_final_tensor[..., :s_dim]
            h_pred_tensor = data_final_tensor[..., s_dim:]

        # 4. Create the flat dictionary that format_pinn_predictions expects
        predictions_for_formatter = {
            'subs_pred': s_pred_tensor,
            'gwl_pred': h_pred_tensor
        }


        y_true_for_format = None
        if targets_test_dict_for_eval:
            y_true_for_format = {
                'subsidence': targets_test_dict_for_eval['subs_pred'],
                'gwl': targets_test_dict_for_eval['gwl_pred']
            }
            
        forecast_csv_filename = (
            f"{CITY_NAME}_{MODEL_NAME}_forecast_{dataset_name_for_forecast.replace(' ', '_')}"
            f"_{FORECAST_YEARS_LIST[0]}-{FORECAST_YEARS_LIST[-1]}.csv"
        )
        forecast_csv_path = os.path.join(RUN_OUTPUT_PATH, forecast_csv_filename)

        forecast_df = format_pinn_predictions(
            predictions=predictions_for_formatter,
            y_true_dict=y_true_for_format,
            target_mapping={'subs_pred': SUBSIDENCE_COL, 'gwl_pred': GWL_COL},
            scaler_info=scaler_info_dict,
            quantiles=QUANTILES,
            forecast_horizon=FORECAST_HORIZON_YEARS,
            output_dims={'subs_pred': OUT_S_DIM, 'gwl_pred': OUT_G_DIM},
            include_coords=True,
            include_gwl=INCLUDE_GWL_IN_DF, # Use config flag
            model_inputs=inputs_test_dict, # Pass inputs to get coords
            evaluate_coverage=True if QUANTILES and y_true_for_format else False,
            savefile=forecast_csv_path, 
            coord_scaler= test_coord_scaler or coord_scaler,
            verbose=1
        )

        if forecast_df is not None and not forecast_df.empty:
            forecast_df2 = adjust_time_predictions(
                forecast_df.copy(), 
                time_col='coord_t', 
                forecast_horizon=0, # Already future years, just inverse transform
                inverse_transformed=True, # It was already inversed by format_pinn_predictions
            )
            forecast_csv_filename = (
                f"{CITY_NAME}_{MODEL_NAME}_forecast_{dataset_name_for_forecast.replace(' ', '_')}"
                f"_{FORECAST_YEARS_LIST[0]}-{FORECAST_YEARS_LIST[-1]}_t_adjusted.csv"
            )
            forecast_csv_path = os.path.join(RUN_OUTPUT_PATH, forecast_csv_filename)
            forecast_df2.to_csv(forecast_csv_path, index=False)
            print(f"\n{MODEL_NAME} forecast results for {dataset_name_for_forecast}"
                  f" saved to: {forecast_csv_path}")
            print("\nSample of {MODEL_NAME} forecast DataFrame:")
            print(forecast_df2.head())
        else:
            print(f"\nNo final {MODEL_NAME} forecast DataFrame was generated.")
else:
    print("\nSkipping forecasting as no valid test or"
          " validation input sequences could be prepared.")

# === Evaluate on test (or fallback validation) with coverage & sharpness ===
# if inputs_test_dict is not None and targets_test_dict_for_eval is not None:
#     test_dataset = tf.data.Dataset.from_tensor_slices(
#         (inputs_test_dict, targets_test_dict_for_eval)
#     ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#     print("\nEvaluating model on test set (includes coverage80"
#           " & sharpness80 when QUANTILES is set)...")
#     eval_results = subs_model_loaded.evaluate(
#         test_dataset, return_dict=True, verbose=1)
#     print("Test evaluation:", eval_results)
    
if inputs_test_dict is not None and targets_test_dict_for_eval is not None:
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (inputs_test_dict, targets_test_dict_for_eval)
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("\nEvaluating model on test set (includes coverage80 & sharpness80 when QUANTILES is set)...")
    eval_results = subs_model_inst.evaluate(test_dataset, return_dict=True, verbose=1)
    print("Test evaluation:", eval_results)
else:
    eval_results = {}

# note :
    # subs_pred_sharpness80 (average interval width, in the same units as 
    # subsidence after the inverse-scaling step inside the formatter isn’t
    # applied here; this is in scaled model space—keep that in mind if you 
    # compare directly to physical units).

if inputs_test_dict is not None:
    phys = subs_model_inst.evaluate_physics(inputs_test_dict)
    print("Physics diagnostics:", {k: float(v.numpy()) for k, v in phys.items()})
else:
    phys = {}
    
# phys = subs_model_loaded.evaluate_physics(inputs_test_dict)
# print("Physics diagnostics:", {k: float(v.numpy()) for k, v in phys.items()})


stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
payload = {
    "timestamp": stamp,
    "tf_version": tf.__version__,
    "numpy_version": np.__version__,
    "quantiles": QUANTILES,
    "horizon": FORECAST_HORIZON_YEARS,
    "batch_size": BATCH_SIZE,
    "metrics_evaluate": {k: _to_py(v) for k, v in (eval_results or {}).items()},
    "physics_diagnostics": {k: _to_py(v) for k, v in (phys or {}).items()},
}

# === Interval calibration diagnostics (subsidence) ===
if inputs_test_dict is not None and targets_test_dict_for_eval is not None:
    ds_for_metrics = tf.data.Dataset.from_tensor_slices(
        (inputs_test_dict, targets_test_dict_for_eval)
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    y_true_list, s_q_list = [], []
    for x, y in ds_for_metrics:
        out = subs_model_inst(x, training=False)  # compiled model
        s_pred_q, _ = subs_model_inst.split_data_predictions(out["data_final"])  # (B,H,Q,1)
        y_true_list.append(y["subs_pred"])
        s_q_list.append(s_pred_q)

    y_true = tf.concat(y_true_list, axis=0)
    s_q    = tf.concat(s_q_list,    axis=0)

    s_lo, s_med, s_hi = _stack_subs_quantiles(s_q)
    cov80_uncal   = float(coverage80_fn(y_true, s_lo, s_hi).numpy())
    sharp80_uncal = float(sharpness80_fn(s_lo, s_hi).numpy())

    s_q_cal = apply_calibrator_to_subs(cal80, s_q)
    s_lo_c, s_med_c, s_hi_c = _stack_subs_quantiles(s_q_cal)

    cov80_cal   = float(coverage80_fn(y_true, s_lo_c, s_hi_c).numpy())
    sharp80_cal = float(sharpness80_fn(s_lo_c, s_hi_c).numpy())

    payload["interval_calibration"] = {
        "target": 0.80,
        "factors_per_horizon": cal80.factors_.tolist(),
        "coverage80_uncalibrated": cov80_uncal,
        "coverage80_calibrated":   cov80_cal,
        "sharpness80_uncalibrated": sharp80_uncal,
        "sharpness80_calibrated":   sharp80_cal,
    }

out_file = os.path.join(RUN_OUTPUT_PATH, f"geoprior_eval_phys_{stamp}.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"Saved metrics + physics JSON -> {out_file}")

# ==================================================================
# ** Step 9: Visualize Forecasts **
# ==================================================================
print(f"\n{'='*20} Step 9: Visualize {MODEL_NAME} Forecasts {'='*20}")
if forecast_df is not None and not forecast_df.empty:
    print(f"Visualizing {MODEL_NAME} forecasts for {dataset_name_for_forecast}...")
    
    coord_plot_cols = ['coord_x', 'coord_y'] 
    if not all(c in forecast_df.columns for c in coord_plot_cols):
        print(f"Warning: Coordinate columns {coord_plot_cols} not in DataFrame. "
              "Spatial plot may fail or be incorrect.")

    print("\n--- Plotting Subsidence Forecasts ---")
    horizon_steps = [1, FORECAST_HORIZON_YEARS] if FORECAST_HORIZON_YEARS > 1 else [1]
    view_years = [FORECAST_YEARS_LIST[idx - 1] for idx in horizon_steps]
    
    plot_forecasts(
        forecast_df=forecast_df,
        target_name=SUBSIDENCE_COL, 
        quantiles=QUANTILES,
        output_dim=OUT_S_DIM,
        kind="spatial", 
        horizon_steps=horizon_steps,
        spatial_cols=coord_plot_cols if all(
            c in forecast_df.columns for c in coord_plot_cols) else None,
        sample_ids="first_n", 
        num_samples=min(3, BATCH_SIZE), 
        max_cols=2, 
        figsize=(7, 5.5), 
        cbar="uniform", 
        step_names = {
            f"step {step}": f'Subsidence: Year {y}' 
            for step, y in zip (horizon_steps, view_years)
            }, 
        verbose=1
    )
    if (INCLUDE_GWL_IN_DF and f"{GWL_COL}_pred" in forecast_df.columns) \
        or (QUANTILES and f"{GWL_COL}_q50" in forecast_df.columns):
        print("\n--- Plotting GWL Forecasts ---")
        plot_forecasts(
            forecast_df=forecast_df,
            target_name=GWL_COL,
            quantiles=QUANTILES,
            output_dim=OUT_G_DIM,
            kind="spatial",
            horizon_steps=horizon_steps,
            spatial_cols=coord_plot_cols if all(
                c in forecast_df.columns for c in coord_plot_cols) else None,
            sample_ids="first_n",
            num_samples=min(3, BATCH_SIZE),
            max_cols=2, 
            figsize=(7, 5.5),
            titles=[f'GWL: Year {y}' for y in view_years],
            verbose=1, 
            cbar =None, 
        ) 
else: 
    print(f"No {MODEL_NAME} forecast data to visualize.")

# ==================================================================
# ** Step 10: Save All Generated Figures **
# ==================================================================
print(f"\n{'='*20} Step 10: Forecast View & Save All Figures {'='*20}")
try: 
    forecast_view (
        forecast_df, 
        spatial_cols = ('coord_x', 'coord_y'), 
        time_col ='coord_t', 
        value_prefixes=['subsidence'], # view subisidence only
        verbose =7,  
        view_quantiles =[0.5], 
        savefig=os.path.join(
            RUN_OUTPUT_PATH, f"{CITY_NAME}_forecast_comparison_plot_"), 
        save_fmts=['.png', '.pdf'] 
      )
    print(f"Forecast view figures saved to: {RUN_OUTPUT_PATH}")
except Exception as e : 
    print(f"Could not plot forecast view: {e}")

try:
    save_all_figures(
        output_dir=RUN_OUTPUT_PATH,
        prefix=f"{CITY_NAME}_{MODEL_NAME}_plot_",
        fmts=['.png', '.pdf'] 
    )
    print(f"All open Matplotlib figures saved to: {RUN_OUTPUT_PATH}")
except Exception as e_save_fig:
    print(f"Could not save all figures: {e_save_fig}")

print(f"\n{'-'*_TW}\n{CITY_NAME.upper()} {MODEL_NAME} SCRIPT COMPLETED.\n"
      f"Outputs are in: {RUN_OUTPUT_PATH}\n{'-'*_TW}")