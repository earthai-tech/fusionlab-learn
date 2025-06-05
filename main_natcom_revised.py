# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com> & Gemini AI
"""
main_zhongshan_pihalnet.py: Zhongshan Land Subsidence Forecasting with PIHALNet

This script performs physics-informed, quantile-based subsidence and GWL
prediction for Zhongshan City using the PIHALNet model from the
fusionlab-learn library.

Dataset Notes for Reproducibility (Zhongshan - 500k samples):
1. The script will primarily attempt to load 'zhongshan_500000.csv'
   from a 'data/' or '../data/' directory, or a specified JUPYTER_PROJECT_ROOT.
2. As a fallback for development/testing without the large dataset,
   it can use `fusionlab.datasets.fetch_zhongshan_data()` (2k samples)
   or a smaller local CSV.

Workflow:
---------
1. Configuration and Library Imports.
2. Load and Inspect Zhongshan Dataset.
3. Preprocessing: Feature Selection, Cleaning, Encoding, Scaling.
4. Define Feature Sets (Static, Dynamic, Future, Coordinates, Targets).
5. Split Master Data by Year (Train/Test).
6. Sequence Generation for PINN using `prepare_pinn_data_sequences`.
7. Create tf.data.Dataset for training and validation.
8. PIHALNet Model Definition, Compilation, and Training with PINN loss.
9. Forecasting on Test Data Sequences.
10. Formatting Predictions and Visualization.
11. Saving Artifacts.
"""

# ==========================================
#  SECTION 0: PREAMBLE & CONFIGURATION
# ==========================================
import os
import shutil
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Explicit import for save_all_figures # noqa

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

# --- FusionLab Imports ---
try:
    from fusionlab.api.util import get_table_size
    from fusionlab.datasets import fetch_zhongshan_data # For Zhongshan
    from fusionlab.nn.pinn.models import PIHALNet      # Our PINN model
    from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences # PINN data prep
    from fusionlab.nn.pinn.op import ( # PINN physics helpers : Noqa
        compute_consolidation_residual, 
        # calculate_gw_flow_pde_residual_from_derivs # For future use
    )
    from fusionlab.nn.utils import extract_batches_from_dataset 
    from fusionlab.nn.losses import combined_quantile_loss
    from fusionlab.nn.pinn.utils import format_pihalnet_predictions
    from fusionlab.plot.forecast import plot_forecasts
    from fusionlab.utils.data_utils import nan_ops
    from fusionlab.utils.io_utils import save_job #, fetch_joblib_data
    from fusionlab.utils.generic_utils import save_all_figures, normalize_time_column 
    from fusionlab.utils.generic_utils import ensure_directory_exists

    print("Successfully imported fusionlab modules.")
except ImportError as e:
    print(f"Critical Error: Failed to import fusionlab modules: {e}. "
          "Please ensure 'fusionlab-learn' is correctly installed.")
    raise
#%
# --- Configuration Parameters ---
CITY_NAME = 'zhongshan'
MODEL_NAME = 'PIHALNet' # Using our new model name

# Data loading: Prioritize 500k sample file
# For Code Ocean, data is typically in ../data or /data
# JUPYTER_PROJECT_ROOT can be set as an environment variable
# For local runs, adjust DATA_DIR as needed.
DATA_DIR = os.getenv("JUPYTER_PROJECT_ROOT", "..") # Go up one level from script if not set
ZHONGSHAN_500K_FILENAME = "zhongshan_500k.csv" # Target file
ZHONGSHAN_2K_FILENAME = "zhongshan_2000.csv"    # Smaller fallback

# Training and Forecasting Periods
TRAIN_END_YEAR = 2022        # Example: Use data up to 2020 for training
FORECAST_START_YEAR = 2023   # Example: Start forecasting for 2021
FORECAST_HORIZON_YEARS = 4   # Example: Predict 3 years ahead (2021, 2022, 2023) (2023, 2024, 2025)
TIME_STEPS = 4               # Lookback window (in years) for dynamic features

# PINN Configuration
PDE_MODE_CONFIG = 'consolidation' # Focus on consolidation
PINN_COEFF_C_CONFIG = 'learnable' # Learn the consolidation coefficient
LAMBDA_PDE_CONFIG = 1.0           # Weight for the PDE loss term in compile

# Model Hyperparameters (can be tuned later with PIHALTuner)
QUANTILES = [0.1, 0.5, 0.9] # For probabilistic forecast
# QUANTILES = None # For point forecast

EPOCHS = 50 # For demonstration; increase for robust results (e.g., 100-200)
LEARNING_RATE = 0.001
BATCH_SIZE = 256 # Adjusted for potentially larger dataset

NUM_BATCHES_TO_EXTRACT = "auto" # Number of batch to extract if there is not enough data
                                # in df_test_master, auto extract all batches.  
AGG = True # Set this to True or False as needed for your specific test case

# Output Directories
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "results_pinn") # For Code Ocean compatibility
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
SAVE_MODEL_AS_SINGLE_FILE = True # .keras format (preferred) or .h5
INCLUDE_GWL_IN_DF =False # Include Gwl in prediction results
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
data_paths_to_check = [
    os.path.join(DATA_DIR, "data", ZHONGSHAN_500K_FILENAME),
    os.path.join(DATA_DIR, ZHONGSHAN_500K_FILENAME),
    os.path.join(".", "data", ZHONGSHAN_500K_FILENAME), # Current dir/data
    ZHONGSHAN_500K_FILENAME # Current dir
]
# Add paths for smaller dataset if 500k is not found
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
            if ZHONGSHAN_500K_FILENAME in path_option and zhongshan_df_raw.shape[0] < 400000: # Basic check
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
# Define columns based on Zhongshan dataset structure
# ['longitude', 'latitude', 'year', 'GWL', 'seismic_risk_score',
#  'rainfall_mm', 'geology', 'normalized_density', 'density_tier',
#  'subsidence_intensity', 'density_concentration',
#  'normalized_seismic_risk_score', 'rainfall_category', 'subsidence']

# For PIHALNet, we need distinct coordinate columns for the `coords` input.
# The model also needs `subsidence` and `GWL` as targets.
# Other features will be dynamic, static, or future.

TIME_COL = 'year' # Will be converted to numeric for PINN 't'
LON_COL = 'longitude'
LAT_COL = 'latitude'
SUBSIDENCE_COL = 'subsidence'
GWL_COL = 'GWL'

# Select relevant features for the model
# Ensure all selected columns actually exist in the loaded DataFrame
available_cols = zhongshan_df_raw.columns.tolist()
selected_features_base = [
    LON_COL, LAT_COL, TIME_COL, SUBSIDENCE_COL, GWL_COL, 'rainfall_mm',
    'geology', # Will be one-hot encoded
    # 'soil_thickness', # Not in Zhongshan sample, add if for Nansha dataset
    'normalized_density', # Example, might be building density
    'normalized_seismic_risk_score',
    # Add other features for nansha_dataset  500k dataset
]
selected_features = [
    col for col in selected_features_base if col in available_cols
    ]
missing_selection = set(selected_features_base) - set(selected_features)
if missing_selection:
    print(f"  Warning: Some initially selected features were not found "
          f"in the loaded data and will be skipped: {missing_selection}")

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
# For `nan_ops`, ensure all columns passed to it exist
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
categorical_cols_to_encode = ['geology'] # Add others depend on you for other tests.
# Ensure only existing columns are processed
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
    # Save encoder
    encoder_path = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_ohe_encoder.joblib")
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
# Create a normalized numerical time from 'year' for PINN's 't' coord
# This was already done if using prepare_pinn_data_sequences, but good to have explicitly
df_for_processing[f"{TIME_COL}_numeric_coord"] = (
    df_for_processing[DT_COL_NAME_TEMP].dt.year +
    (df_for_processing[DT_COL_NAME_TEMP].dt.dayofyear - 1) /
    (365 + df_for_processing[DT_COL_NAME_TEMP].dt.is_leap_year.astype(int))
)
TIME_COL_NUMERIC_PINN = f"{TIME_COL}_numeric_coord"


# --- Scaling Numerical Features ---
# All numerical features including targets and coordinates might be scaled
# `prepare_pinn_data_sequences` has its own `normalize_coords` option.
# Here, we scale features that go into the data-driven part.
# XXX TO FIX : Optimize
numerical_cols_for_scaling_model = [
    # LON_COL, 
    # LAT_COL, 
    GWL_COL, 
    'rainfall_mm',
    # 'soil_thickness', # If Nanshan
    'normalized_density', # If present and numerical # because not use in Nansha
    'normalized_seismic_risk_score',
    SUBSIDENCE_COL, # Scale target as well for stable training
    # TIME_COL_NUMERIC_PINN # Scale numeric time
]
# Add OHE features (already 0 or 1, but good practice if other numerical derived)
# numerical_cols_for_scaling_model.extend(encoded_feature_names_list)
numerical_cols_for_scaling_model = list(set( # Unique existing cols
    [c for c in numerical_cols_for_scaling_model if c in df_for_processing.columns]
))

df_scaled = df_for_processing.copy()
scaler_main = MinMaxScaler() # Or StandardScaler
if numerical_cols_for_scaling_model:
    df_scaled[numerical_cols_for_scaling_model] = scaler_main.fit_transform(
        df_scaled[numerical_cols_for_scaling_model]
    )
    scaler_path = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_main_scaler.joblib")
    try: 
        save_job( scaler_main, scaler_path
            # append_date= True      # Date and version of main package 
            # append_versions=True,  # have already been appended earlier, can skipped.
        )
    except: 
        joblib.dump(scaler_main, scaler_path)
        
    print(f"  Numerical features scaled. Scaler saved to: {scaler_path}")

if SAVE_INTERMEDIATE_ARTEFACTS:
    scaled_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_03_processed_scaled_data.csv")
    df_scaled.to_csv(scaled_path, index=False)
    print(f"  Processed and scaled data saved to: {scaled_path}")

# ==================================================================
# ** Step 4: Define Feature Sets for `prepare_pinn_data_sequences` **
# ==================================================================
print(f"\n{'='*20} Step 4: Define Feature Sets for PINN Data Prep {'='*20}")

# Static features for PIHALNet (after VSN, these become context)
# These are features that do NOT change over the `time_steps` for a given sample.
# If grouping by lon/lat, geology might be static per group.
static_features_list = encoded_feature_names_list # Geology, etc.
# Add other known static features if any, e.g. soil_thickness if it's one value per (lon,lat)
# For PIHALNet, static_features are per-sample, not per-timestep.
# If `group_id_cols` are lon/lat, these static features are those constant per lon/lat.

# Dynamic features: vary over `time_steps` (past observed)
dynamic_features_list = [
    GWL_COL, 'rainfall_mm', 'normalized_density', 
    'normalized_seismic_risk_score'
    # Add other dynamic features from df_scaled.columns
]
dynamic_features_list = [
    c for c in dynamic_features_list if c in df_scaled.columns
    ]

# Future known features: known for the `forecast_horizon`
future_features_list = ['rainfall_mm'] # Example: if rainfall forecast is available
future_features_list = [
    c for c in future_features_list if c in df_scaled.columns
    ]

# Grouping columns for sequence generation
# If each (lon, lat) pair represents a unique time series
GROUP_ID_COLS_SEQ = [LON_COL, LAT_COL] 

print(f"  PINN Time Coordinate: {TIME_COL_NUMERIC_PINN}")
print(f"  PINN Lon Coordinate: {LON_COL}")
print(f"  PINN Lat Coordinate: {LAT_COL}")
print(f"  Target 1 (Subsidence): {SUBSIDENCE_COL}")
print(f"  Target 2 (GWL): {GWL_COL}")
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
# For test data, PIHALNet needs future knowns for its "future_features" input
# and also target coords for its "coords" input.
# The `prepare_pinn_data_sequences` will handle slicing.
df_test_master = df_scaled[
    df_scaled[TIME_COL] >= FORECAST_START_YEAR - (TIME_STEPS / 12) # Include lookback for test seq
].copy() 
# Ensure test master starts early enough to create first sequence including lookback

print(f"Master train data shape (<= {TRAIN_END_YEAR}): {df_train_master.shape}")
print(f"Master test data shape (for {FORECAST_START_YEAR} onwards): {df_test_master.shape}")

if df_train_master.empty:
    raise ValueError(f"Training data empty after split at year {TRAIN_END_YEAR}.")

sequence_file_path_train = os.path.join(
    RUN_OUTPUT_PATH,
    f'{CITY_NAME}_train_pinn_sequences_T{TIME_STEPS}_H{FORECAST_HORIZON_YEARS}.joblib'
)

print(f"Generating PINN training sequences (T={TIME_STEPS}, H={FORECAST_HORIZON_YEARS})...")
# Note: `prepare_pinn_data_sequences` requires lon_col, lat_col explicitly
OUT_S_DIM =1 
OUT_G_DIM =1
inputs_train_dict, targets_train_dict, coord_scaler = prepare_pinn_data_sequences(
    df=df_train_master,
    time_col=TIME_COL_NUMERIC_PINN, # Use the numeric time for 't' coord
    lon_col=LON_COL,
    lat_col=LAT_COL,
    subsidence_col=SUBSIDENCE_COL,
    gwl_col=GWL_COL,
    dynamic_cols=dynamic_features_list,
    static_cols=static_features_list,
    future_cols=future_features_list,
    group_id_cols=GROUP_ID_COLS_SEQ,
    time_steps=TIME_STEPS,
    forecast_horizon=FORECAST_HORIZON_YEARS,
    output_subsidence_dim=OUT_S_DIM, # PIHALNet default is 1
    output_gwl_dim=OUT_G_DIM,       # PIHALNet default is 1
    normalize_coords=True, # Recommended for PINN coordinates
    savefile=sequence_file_path_train, # Save the prepared sequences
    return_coord_scaler= True, # return corrd-scaler for inverse_transform. 
    verbose=7 # High verbosity for sequence generation
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

# Create a combined dataset for easier splitting
# Need to handle potential None for static/future features in inputs_train_dict
num_train_samples = inputs_train_dict['dynamic_features'].shape[0]
dataset_inputs = {
    'coords': inputs_train_dict['coords'],
    'dynamic_features': inputs_train_dict['dynamic_features'],
    'static_features': inputs_train_dict.get('static_features') if inputs_train_dict.get('static_features') is not None \
                       else np.zeros((num_train_samples, 0), dtype=np.float32),
    'future_features': inputs_train_dict.get('future_features') if inputs_train_dict.get('future_features') is not None \
                       else np.zeros((num_train_samples, FORECAST_HORIZON_YEARS, 0), dtype=np.float32)
}
# Ensure keys in targets_train_dict match PIHALNet.compile loss keys
dataset_targets = {
    'subs_pred': targets_train_dict['subsidence'], # Key for subsidence output
    'gwl_pred': targets_train_dict['gwl']          # Key for GWL output
}

full_dataset = tf.data.Dataset.from_tensor_slices(
    (dataset_inputs, dataset_targets)
)

# Split into training and validation
# Define sizes first
total_size = num_train_samples
val_size = int(0.2 * total_size)
train_size = total_size - val_size

full_dataset = full_dataset.shuffle(buffer_size=total_size, seed=42)
train_dataset = full_dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = full_dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"  Training dataset created with {train_size} samples.")
print(f"  Validation dataset created with {val_size} samples.")
# Print one batch spec for verification
for x_batch, y_batch in train_dataset.take(1):
    print("  Sample batch input keys:", list(x_batch.keys()))
    for k,v in x_batch.items(): print(f"    {k}: {v.shape}")
    print("  Sample batch target keys:", list(y_batch.keys()))
    for k,v in y_batch.items(): print(f"    {k}: {v.shape}")
    break
#%
# ==================================================================
# ** Step 7: PIHALNet Model Definition, Compilation & Training **
# ==================================================================
print(f"\n{'='*20} Step 7: PIHALNet Model Training {'='*20}")

# Infer input dimensions from the prepared data for PIHALNet __init__
s_dim_model = dataset_inputs['static_features'].shape[-1]
d_dim_model = dataset_inputs['dynamic_features'].shape[-1]
f_dim_model = dataset_inputs['future_features'].shape[-1]

pihalnet_params = {
    'embed_dim': 32,
    'hidden_units': 64,
    'lstm_units': 64,
    'attention_units': 64,
    'num_heads': 2,
    'dropout_rate': 0.1,
    'max_window_size': TIME_STEPS,
    'memory_size': 50, # Example
    'scales': [1, 2],  # Example
    'multi_scale_agg': 'last',
    'final_agg': 'last',
    'use_residuals': True,
    'use_batch_norm': False,
    'use_vsn': True, # Enable VSN
    'vsn_units': 32, # Units for VSN internal GRNs
}

pihal_model_inst = PIHALNet(
    static_input_dim=s_dim_model,
    dynamic_input_dim=d_dim_model,
    future_input_dim=f_dim_model,
    output_subsidence_dim=OUT_S_DIM, # As defined earlier
    output_gwl_dim=OUT_G_DIM,         # As defined earlier
    forecast_horizon=FORECAST_HORIZON_YEARS,
    quantiles=QUANTILES,
    pde_mode=PDE_MODE_CONFIG,
    pinn_coefficient_C=PINN_COEFF_C_CONFIG,
    gw_flow_coeffs=None, # Keep None for consolidation focus
    **pihalnet_params
)

# Build the model with a sample batch to initialize weights and allow summary
for x_sample_build, _ in train_dataset.take(1):
    pihal_model_inst(x_sample_build) 
    break # Only need one batch to build
pihal_model_inst.summary(line_length=110, expand_nested=True)

# Compile PIHALNet
# Loss dictionary keys MUST match the keys PIHALNet.call() output dict for predictions
loss_dict = {
    'subs_pred': tf.keras.losses.MeanSquaredError() if QUANTILES is None \
                 else combined_quantile_loss(QUANTILES),
    'gwl_pred': tf.keras.losses.MeanSquaredError() if QUANTILES is None \
                else combined_quantile_loss(QUANTILES) # Or a different loss for GWL
}
metrics_dict = {
    'subs_pred': ['mae', 'mse'],
    'gwl_pred': ['mae', 'mse']
}
loss_weights_dict = { # Weights for the data loss terms
    'subs_pred': 1.0,
    'gwl_pred': 0.5 # Example: GWL data loss is half as important
}

pihal_model_inst.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss_dict,
    metrics=metrics_dict,
    loss_weights=loss_weights_dict,
    lambda_pde=LAMBDA_PDE_CONFIG # Weight for the physics loss component
)
print("PIHALNet model compiled successfully.")

# Callbacks
early_stopping_cb = EarlyStopping(
    monitor='val_total_loss', # PIHALNet train_step returns 'total_loss'
    patience=15, # Increased patience
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
    monitor='val_total_loss',
    save_best_only=True,
    save_weights_only=False, # Save entire model
    verbose=1
)
print(f"Model checkpoints will be saved to: {model_checkpoint_path}")

print(f"\nStarting {MODEL_NAME} model training...")
history = pihal_model_inst.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping_cb, model_checkpoint_cb],
    verbose=1 # Or 2 for more detail per epoch
)
print(f"Best validation total_loss achieved: "
      f"{min(history.history.get('val_total_loss', [np.inf])):.4f}")
# %
# Load the best model saved by ModelCheckpoint
print(f"\nLoading best model from checkpoint: {model_checkpoint_path}")

loss_to_use = combined_quantile_loss(QUANTILES) if QUANTILES else 'mse'
try:
    # If combined_quantile_loss is a function used directly
    # custom_objects_load['combined_quantile_loss'] = combined_quantile_loss
    # If it's part of a Keras Loss class that's registered, not needed.
    # For PIHALNet, the losses are applied in compile.
    custom_objects_load = {}
    if QUANTILES:
        # If combined_quantile_loss is a function used directly
        # custom_objects_load['combined_quantile_loss'] = combined_quantile_loss
        # If it's part of a Keras Loss class that's registered, not needed.
        # For PIHALNet, the losses are applied in compile.
        custom_objects_load = {'combined_quantile_loss': loss_to_use} 
        
    with custom_object_scope(custom_objects_load):
        pihal_model_loaded = load_model(model_checkpoint_path)
    print("Best PIHALNet model loaded successfully for forecasting.")
except Exception as e_load:
    print(f"Error loading saved PIHALNet model: {e_load}. "
          "Using model from end of training.")
    pihal_model_loaded = pihal_model_inst
#%
# =============================================================================
# ** Step 8: Forecasting on Test Data **
# =============================================================================
print(f"\n{'='*20} Step 8: Forecasting on Test Data {'='*20}")
# For PIHALNet, we need to prepare test inputs similar to training
# This includes 'coords', 'static_features', 'dynamic_features', 'future_features'
# Initialize variables
inputs_test_dict = None
targets_test_dict_for_eval = None
dataset_name_for_forecast = "Unknown"
test_coord_scaler =None # Placeholder for recovering coordinate scaler.
try:
    # 1. Attempt to generate sequences from the test dataset
    if df_test_master.empty:
        print("Warning: Test data (df_test_master) is empty. Cannot generate test sequences.")
        # Raise an exception if the dataframe is empty to trigger the fallback
        raise ValueError("Test data (df_test_master) is empty.")

    print(f"Attempting to generate PINN sequences from test data (year {FORECAST_START_YEAR})...")
    inputs_test_dict, targets_test_dict_raw, test_coord_scaler = prepare_pinn_data_sequences(
        df=df_train_master, #df_test_master,
        time_col=TIME_COL_NUMERIC_PINN,
        lon_col=LON_COL, lat_col=LAT_COL,
        subsidence_col=SUBSIDENCE_COL,
        gwl_col=GWL_COL,
        dynamic_cols=dynamic_features_list,
        static_cols=static_features_list,
        future_cols=future_features_list,
        group_id_cols=GROUP_ID_COLS_SEQ,
        time_steps=TIME_STEPS,
        forecast_horizon=FORECAST_HORIZON_YEARS,
        output_subsidence_dim=OUT_S_DIM,
        output_gwl_dim=OUT_G_DIM,
        normalize_coords=True, # Match training
        return_coord_scaler= True, 
        verbose=1
    )

    # Check if any sequences were actually created
    if targets_test_dict_raw['subsidence'].shape[0] == 0:
        # Raise an exception to trigger the fallback
        raise ValueError("No test sequences were generated from the provided test data.")

    # If successful, prepare the target dictionary for evaluation
    print("Test sequences generated successfully.")
    dataset_name_for_forecast = f"TestSet (Year {FORECAST_START_YEAR})"
    targets_test_dict_for_eval = {
        'subs_pred': targets_test_dict_raw['subsidence'],
        'gwl_pred': targets_test_dict_raw['gwl']
    }

except Exception as e:
    # 2. If any error occurs, fall back to using the validation set
    print(f"\n[WARNING] Could not generate test sequences due to an error: {e}")
    print(
        "Falling back to use the validation dataset for forecasting "
        "demonstration."
    )
    
    dataset_name_for_forecast = "ValidationSet_Fallback"

    print(
        f"Attempting to extract batches from validation set (AGG={AGG})..."
    )
    # Use the new utility function
    # train_dataset 
    extracted_validation_data = extract_batches_from_dataset(
        val_dataset,
        num_batches_to_extract=NUM_BATCHES_TO_EXTRACT,
        errors='warn', 
        agg=AGG, 
    )

    inputs_test_dict = None
    targets_test_dict_for_eval = None

    if AGG:
        # If agg is True, extracted_validation_data is a single tuple 
        # (inputs, targets) or None
        if extracted_validation_data is not None:
            print(
                "Successfully extracted and aggregated validation batch(es)."
            )
            # Ensure the tuple has at least two elements before unpacking
            if isinstance(extracted_validation_data, tuple) and \
               len(extracted_validation_data) >= 2:
                inputs_test_dict = extracted_validation_data[0]
                targets_test_dict_for_eval = extracted_validation_data[1]
                # If more elements in tuple, you might need to handle them:
                # e.g., 
                # inputs_test_dict, targets_test_dict_for_eval, 
                # *other_aggregated_parts = extracted_validation_data
            elif isinstance(extracted_validation_data, tuple) and \
                 len(extracted_validation_data) == 1:
                inputs_test_dict = extracted_validation_data[0]
                # targets_test_dict_for_eval would remain None or handle
                # as appropriate
                print(
                    "[WARNING] Aggregated data only contained one component "
                    "(inputs). Targets will be None."
                )
            else:
                print(
                    f"[ERROR] Aggregated data is not a tuple or has "
                    f"unexpected structure: {type(extracted_validation_data)}. "
                    "Cannot unpack."
                )
        else:
            print(
                "[WARNING] Aggregation returned None. No validation data "
                "extracted."
            )
    else:
        # If agg is False, extracted_validation_data is a list of batch tuples
        if extracted_validation_data: # Check if the list is not empty
            print(
                f"Successfully extracted {len(extracted_validation_data)} "
                "validation batch(es) (no aggregation). Using the first one."
            )
            # Ensure the first batch tuple has at least two elements
            first_batch = extracted_validation_data[0]
            if isinstance(first_batch, tuple) and len(first_batch) >= 2:
                inputs_test_dict = first_batch[0]
                targets_test_dict_for_eval = first_batch[1]
            elif isinstance(first_batch, tuple) and len(first_batch) == 1:
                inputs_test_dict = first_batch[0]
                print(
                    "[WARNING] First extracted batch only contained one "
                    "component (inputs). Targets will be None."
                )
            else:
                print(
                    f"[ERROR] First extracted batch is not a tuple or has "
                    f"unexpected structure: {type(first_batch)}. "
                    "Cannot unpack."
                )
        else:
            print(
                "[WARNING] Extraction returned an empty list. No validation "
                "data extracted."
            )

    # Fallback if the utility function didn't yield usable data
    if inputs_test_dict is None:
        print(
            "[INFO] Fallback to val_dataset.take(1) as primary extraction "
            "failed or yielded no data."
        )
        # Attempt to take one batch directly if previous attempts failed
        for x_val_sample_fallback, y_val_sample_fallback in val_dataset.take(1):
            inputs_test_dict = x_val_sample_fallback
            targets_test_dict_for_eval = y_val_sample_fallback
            print(
                "[INFO] Successfully extracted one batch using "
                "val_dataset.take(1)."
            )
            break # Use only this first batch
        else: 
            # This else belongs to the for loop, executed if the loop 
            # completes without a break (i.e., val_dataset.take(1) was empty)
            print(
                "[ERROR] Critical Fallback failed: Could not extract any "
                "batches from the validation dataset even with .take(1)."
            )
            # inputs_test_dict and targets_test_dict_for_eval will remain None

# 3. Proceed with forecasting if input data (from test or validation) is available
if inputs_test_dict is not None:
    print(f"\nGenerating PIHALNet predictions on: {dataset_name_for_forecast}...")
    
    # Get model predictions
    pihalnet_test_outputs = pihal_model_loaded.predict(inputs_test_dict, verbose=0)
    
    # Format true values for comparison
    y_true_for_format = {
        'subsidence': targets_test_dict_for_eval['subs_pred'],
        'gwl': targets_test_dict_for_eval['gwl_pred']
    }

    # Format predictions into a clear DataFrame
    forecast_df_pihalnet = format_pihalnet_predictions(
        pihalnet_outputs=pihalnet_test_outputs,
        y_true_dict=y_true_for_format,
        target_mapping={'subs_pred': SUBSIDENCE_COL, 'gwl_pred': GWL_COL},
        quantiles=QUANTILES,
        forecast_horizon=FORECAST_HORIZON_YEARS,
        output_dims={'subs_pred': OUT_S_DIM, 'gwl_pred': OUT_G_DIM},
        include_coords_in_df=True,
        model_inputs=inputs_test_dict,
        evaluate_coverage=True if QUANTILES else False,
        coord_scaler= test_coord_scaler or coord_scaler, # either test is passed or not
        verbose=1
    )

    # Save the results to a CSV file
    if forecast_df_pihalnet is not None and not forecast_df_pihalnet.empty:
        forecast_csv_filename = (
            f"{CITY_NAME}_{MODEL_NAME}_forecast_{dataset_name_for_forecast.replace(' ', '_')}"
            f"_{FORECAST_YEARS_LIST[0]}-{FORECAST_YEARS_LIST[-1]}.csv"
        )
        forecast_csv_path = os.path.join(RUN_OUTPUT_PATH, forecast_csv_filename)
        forecast_df_pihalnet.to_csv(forecast_csv_path, index=False)
        print(f"\nPIHALNet forecast results for {dataset_name_for_forecast} saved to: {forecast_csv_path}")
        print("\nSample of PIHALNet forecast DataFrame:")
        print(forecast_df_pihalnet.head())
    else:
        print("\nNo final PIHALNet forecast DataFrame was generated.")
else:
    print("\nSkipping forecasting as no valid test or validation input sequences could be prepared.")

# ==================================================================
# ** Step 9: Visualize Forecasts **
# ==================================================================
print(f"\n{'='*20} Step 9: Visualize PIHALNet Forecasts {'='*20}")
if forecast_df_pihalnet is not None and not forecast_df_pihalnet.empty:
    print(f"Visualizing PIHALNet forecasts for {dataset_name_for_forecast}...")
    
    # Check if coordinate columns exist for spatial plot
    coord_plot_cols = ['coord_x', 'coord_y'] # From format_pihalnet_predictions
    if not all(c in forecast_df_pihalnet.columns for c in coord_plot_cols):
        print(f"Warning: Coordinate columns {coord_plot_cols} not in DataFrame. "
              "Spatial plot may fail or be incorrect.")

    # Plot for Subsidence
    print("\n--- Plotting Subsidence Forecasts ---")
    plot_forecasts(
        forecast_df=forecast_df_pihalnet,
        target_name=SUBSIDENCE_COL, # Use the base name used in format_pihalnet_predictions
        quantiles=QUANTILES,
        output_dim=OUT_S_DIM,
        kind="spatial", # Or "temporal"
        horizon_steps=[1, FORECAST_HORIZON_YEARS] if FORECAST_HORIZON_YEARS > 1 else 1,
        spatial_cols=coord_plot_cols if all(
            c in forecast_df_pihalnet.columns for c in coord_plot_cols) else None,
        sample_ids="first_n", num_samples=min(3, BATCH_SIZE), # For temporal
        max_cols=2, # For spatial or temporal multi-sample
        figsize_per_subplot=(7, 5.5), # Adjusted for potentially 2 plots
        # titles=[f'Subsidence: Year {y}' for y in FORECAST_YEARS_LIST[:2]], # Example
        verbose=1
    )
    # Plot for GWL
    if INCLUDE_GWL_IN_DF and f"{GWL_COL}_pred" in forecast_df_pihalnet.columns \
        or (QUANTILES and f"{GWL_COL}_q50" in forecast_df_pihalnet.columns):
        print("\n--- Plotting GWL Forecasts ---")
        plot_forecasts(
            forecast_df=forecast_df_pihalnet,
            target_name=GWL_COL,
            quantiles=QUANTILES,
            output_dim=OUT_G_DIM,
            kind="spatial",
            horizon_steps=[1, FORECAST_HORIZON_YEARS] if FORECAST_HORIZON_YEARS > 1 else 1,
            spatial_cols=coord_plot_cols if all(
                c in forecast_df_pihalnet.columns for c in coord_plot_cols) else None,
            sample_ids="first_n", num_samples=min(3, BATCH_SIZE),
            max_cols=2,
            figsize_per_subplot=(7, 5.5),
            # titles=[f'GWL: Year {y}' for y in FORECAST_YEARS_LIST[:2]],
            verbose=1
        )
else:
    print("No PIHALNet forecast data to visualize.")

# ==================================================================
# ** Step 10: Save All Generated Figures **
# ==================================================================
print(f"\n{'='*20} Step 10: Save All Figures {'='*20}")
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