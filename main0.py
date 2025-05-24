# -*- coding: utf-8 -*-
# License : BSD-3-Clause
"""
main.py: Nansha Land Subsidence Forecasting with XTFT

This script performs quantile-based subsidence prediction for the
Nansha district using the XTFT deep learning model from the
fusionlab-learn library.

The Nansha dataset typically spans multiple years (e.g., 2015-2022).
For this demonstration with sampled data:
- Training data: Up to and including TRAIN_END_YEAR.
- Test/Validation data for forecast generation: Year FORECAST_START_YEAR.
- Forecasting period: Configurable, e.g., 2-year horizon for sample.

Note on Data Import:
This script is designed for reproducibility.
1. It first attempts to load the Nansha dataset using
   `fusionlab.datasets.fetch_nansha_data()` (which provides a
   2,000-sample subset by default).
2. If that fails, it tries to load 'nansha_500_000.csv' from a local
   'data/' or '../data/' directory.
3. If that also fails, it tries to load 'nansha_2000.csv' from
   local 'data/' or '../data/' directory.
For the full dataset, please contact the authors.

Workflow:
---------
1. Configuration and Library Imports.
2. Load and Inspect Nansha Dataset with fallbacks.
3. Preprocessing: Feature Selection, Cleaning, Encoding, Scaling.
4. Define Feature Sets (Static, Dynamic, Future).
5. Split Master Data by Year (Train/Test).
6. Sequence Generation for Training Data using `reshape_xtft_data`.
7. Train/Validation Split of Sequence Data.
8. XTFT Model Definition, Compilation, and Training.
9. Forecasting on Test Data Sequences (if available).
10. Visualization of Forecasts (if forecasts generated).
11. Saving All Generated Figures.

Paper Reference:
XTFT: A Next-Generation Temporal Fusion Transformer for
Uncertainty-Rich Time Series Forecasting (Submitted to IEEE PAMI)

Author: [Anonymous] for Double-blind review
"""

# ==========================================
#  SECTION 0: PREAMBLE & CONFIGURATION
# ==========================================
import os
import shutil
import joblib
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress C++ TF logs
tf.get_logger().setLevel('ERROR')
if hasattr(tf, 'autograph') and hasattr(tf.autograph, 'set_verbosity'):
    tf.autograph.set_verbosity(0)

# FusionLab Imports
try:
    from fusionlab.api.util import get_table_size
    from fusionlab.datasets import fetch_nansha_data
    # from fusionlab.datasets._property import get_data # Not used if path is explicit
    from fusionlab.nn.losses import combined_quantile_loss
    from fusionlab.nn.transformers import XTFT, SuperXTFT 
    from fusionlab.nn.utils import (
        reshape_xtft_data,
        format_predictions_to_dataframe,
    )
    from fusionlab.plot.forecast import plot_forecasts
    from fusionlab.utils.data_utils import nan_ops
    from fusionlab.utils.io_utils import fetch_joblib_data, save_job # noqa ( can use for robust saver)
    from fusionlab.utils.generic_utils import save_all_figures
except ImportError as e:
    print(f"Critical Error: Failed to import fusionlab modules: {e}. "
          "Please ensure 'fusionlab-learn' (version >= 0.2.1) is "
          "installed correctly in your environment.")
    raise

# --- Configuration Parameters ---
CITY_NAME = 'nansha'
USE_SUPER_XTFT = False # Set to False for standard XTFT for the paper
MODEL_SUFFIX = '_super' if USE_SUPER_XTFT else ''
TRANSFORMER_CLASS = SuperXTFT if USE_SUPER_XTFT else XTFT

TRAIN_END_YEAR = 2021
FORECAST_START_YEAR = 2022
# For the paper with FULL data, use FORECAST_HORIZON_YEARS = 4.
# For SAMPLE data (2k or 500k), a shorter horizon is more robust.
FORECAST_HORIZON_YEARS = 2
TIME_STEPS = 2 # Lookback window (in years)
# Min length needed per group = TIME_STEPS + FORECAST_HORIZON_YEARS

FORECAST_YEARS = [
    FORECAST_START_YEAR + i for i in range(FORECAST_HORIZON_YEARS)
    ]
print(f"Configuration: TIME_STEPS={TIME_STEPS}, "
      f"FORECAST_HORIZON={FORECAST_HORIZON_YEARS} years.")
print(f"Forecasting for years: {FORECAST_YEARS}")

SPATIAL_COLS = ['longitude', 'latitude']
QUANTILES = [0.1, 0.5, 0.9]
# QUANTILES = None # For point forecast

EPOCHS = 50 # For demo; increase for paper (e.g., 100-200)
LEARNING_RATE = 0.001
BATCH_SIZE = 32

BASE_OUTPUT_DIR = "results" # For Code Ocean
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
RUN_OUTPUT_PATH = os.path.join(
    BASE_OUTPUT_DIR, f"{CITY_NAME}_xtft_run{MODEL_SUFFIX}"
    )
if os.path.isdir(RUN_OUTPUT_PATH): shutil.rmtree(RUN_OUTPUT_PATH)
os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)
print(f"Output artifacts will be saved in: {RUN_OUTPUT_PATH}")

SAVE_INTERMEDIATE_ARTEFACTS = True
SAVE_MODEL_AS_SINGLE_FILE = True # .keras format

try: _TW = get_table_size()
except: _TW = 80
print(f"\n{'-'*_TW}\n{CITY_NAME.upper()} XTFT FORECASTING (IEEE PAMI)\n{'-'*_TW}")

# ==================================================================
# ** Step 1: Load Nansha Dataset **
# ==================================================================
print(f"\n{'='*20} Step 1: Load Nansha Dataset {'='*20}")
nansha_df_raw = None
try:
    raise # to fallback to 500_000_samples for testing  
    # Attempt 1: Fetch from fusionlab.datasets (default 2000 samples)
    print("Attempt 1: Fetching Nansha data via fusionlab.datasets...")
    nansha_df_raw = fetch_nansha_data(as_frame=True, verbose=1,
                                      download_if_missing=True)
    print(f"  Nansha data loaded via fetch_nansha_data. Shape: {nansha_df_raw.shape}")
    
except Exception as e_fetch:
    print(f"  Warning: Failed to load via fetch_nansha_data: {e_fetch}")
    # Fallback paths for Code Ocean or local setup
    # Assumes data folder is at the root of the capsule or project
    paths_to_try = [
        # os.path.join("data", "nansha_500_000.csv"),
        # os.path.join("..", "data", "nansha_500_000.csv"), # If script is in a subdir
        # os.path.join("data", "nansha_2000.csv"),
        # os.path.join("..", "data", "nansha_2000.csv"),
        os.path.join( r'J:\nature_data\final', "nansha_500_000.csv") # test this sample of data 
    ]
    for local_path in paths_to_try:
        print(f"Attempt 2: Trying local path '{local_path}'...")
        if os.path.exists(local_path):
            try:
                nansha_df_raw = pd.read_csv(local_path)
                print(f"  Nansha data loaded from local CSV '{local_path}'. "
                      f"Shape: {nansha_df_raw.shape}")
                break # Stop if successfully loaded
            except Exception as e_local:
                print(f"  ERROR: Failed to load from '{local_path}': {e_local}")
        else:
            print(f"  Path not found: '{local_path}'")

if nansha_df_raw is None or nansha_df_raw.empty:
    raise FileNotFoundError(
        "Nansha data CSV could not be loaded from any specified path. "
        "Please ensure the data is available."
    )
# ...  save artefacts  ...
if SAVE_INTERMEDIATE_ARTEFACTS:
    raw_data_path = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_01_raw_data.csv")
    nansha_df_raw.to_csv(raw_data_path, index=False)
    print(f"  Raw data saved to: {raw_data_path}")

# ==================================================================
# ** Step 2: Preprocessing - Feature Selection & Initial Cleaning **
# ==================================================================
print(f"\n{'='*20} Step 2: Preprocessing - Initial Steps {'='*20}")
selected_features = [
    'longitude', 'latitude', 'year', 'subsidence', 'GWL', 'rainfall_mm',
    'geology', 'soil_thickness', 'building_concentration',
    'normalized_seismic_risk_score'
]
nansha_df_selected = nansha_df_raw[[
    col for col in selected_features if col in nansha_df_raw.columns
    ]].copy()

DT_COL_NAME = 'Date'
nansha_df_selected[DT_COL_NAME] = pd.to_datetime(
    nansha_df_selected['year'], format='%Y')
print(f"NaNs before cleaning: {nansha_df_selected.isna().sum().sum()}")
nansha_df_cleaned = nan_ops(nansha_df_selected, ops='sanitize', action='fill', verbose=0)
print(f"NaNs after cleaning: {nansha_df_cleaned.isna().sum().sum()}")
if SAVE_INTERMEDIATE_ARTEFACTS:
    cleaned_path = os.path.join(RUN_OUTPUT_PATH, f"{CITY_NAME}_02_cleaned_data.csv")
    nansha_df_cleaned.to_csv(cleaned_path, index=False)
    print(f"  Cleaned data saved to: {cleaned_path}")

# ==================================================================
# ** Step 3: Preprocessing - Encoding & Scaling **
# ==================================================================
print(f"\n{'='*20} Step 3: Preprocessing - Encoding & Scaling {'='*20}")
df_for_processing = nansha_df_cleaned.copy()
TARGET_COL_NAME = 'subsidence'
# ...  encoding and scaling  ...
categorical_cols_to_encode = ['geology']
if 'building_concentration' in df_for_processing.columns and \
   df_for_processing['building_concentration'].dtype == 'object':
    categorical_cols_to_encode.append('building_concentration')

encoded_feature_names_list = []
if categorical_cols_to_encode:
    encoder_ohe = OneHotEncoder(
        sparse_output=False, 
        handle_unknown='ignore', 
        dtype=np.float32
    )
    encoded_data_parts = encoder_ohe.fit_transform(
        df_for_processing[categorical_cols_to_encode]
        )
    new_ohe_cols = encoder_ohe.get_feature_names_out(categorical_cols_to_encode)
    encoded_feature_names_list.extend(new_ohe_cols)
    encoded_df_part = pd.DataFrame(
        encoded_data_parts, columns=new_ohe_cols,
        index=df_for_processing.index
    )
    df_for_processing = pd.concat(
        [df_for_processing.drop(
            columns=categorical_cols_to_encode), 
            encoded_df_part], axis=1
    )
    print(f"  Encoded features: {new_ohe_cols.tolist()}")

numerical_cols_to_scale = [
    'longitude', 'latitude', 'GWL', 'rainfall_mm', 'soil_thickness',
    'normalized_seismic_risk_score', TARGET_COL_NAME
]
if 'building_concentration' in df_for_processing.columns and \
   'building_concentration' not in categorical_cols_to_encode:
    numerical_cols_to_scale.append('building_concentration')
numerical_cols_to_scale = [
    col for col in numerical_cols_to_scale 
    if col in df_for_processing.columns]

df_scaled = df_for_processing.copy()
scaler_main = MinMaxScaler()
if numerical_cols_to_scale:
    df_scaled[numerical_cols_to_scale] = scaler_main.fit_transform(
        df_scaled[numerical_cols_to_scale]
        )
    scaler_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_main_scaler.joblib"
        )
    joblib.dump(scaler_main, scaler_path)
    print(f"  Numerical features scaled. Scaler saved to: {scaler_path}")

if SAVE_INTERMEDIATE_ARTEFACTS:
    scaled_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_03_processed_scaled_data.csv")
    df_scaled.to_csv(scaled_path, index=False)
    print(f"  Processed and scaled data saved to: {scaled_path}")

# ==================================================================
# ** Step 4: Define Feature Sets for Model Input **
# ==================================================================
print(f"\n{'='*20} Step 4: Define Feature Sets {'='*20}")
# ... (rest of step 4: defining static_features_model etc. as before) ...
static_features_model = ['longitude', 'latitude'] + encoded_feature_names_list
static_features_model = [
    c for c in static_features_model if c in df_scaled.columns]
dynamic_features_model = [
    'GWL', 
    'rainfall_mm', 
    'soil_thickness', 
    'normalized_seismic_risk_score'
   ]
if ( 
        'building_concentration' in numerical_cols_to_scale 
        and 'building_concentration' not in categorical_cols_to_encode
    ):
    dynamic_features_model.append('building_concentration')
dynamic_features_model = [
    c for c in dynamic_features_model if c in df_scaled.columns]
future_features_model = ['rainfall_mm']
future_features_model = [
    c for c in future_features_model if c in df_scaled.columns]

print(f"  Static features: {static_features_model}")
print(f"  Dynamic features: {dynamic_features_model}")
print(f"  Future features: {future_features_model}")

# ==================================================================
# ** Step 5: Split Master Data & Generate Sequences for Training **
# ==================================================================
print(f"\n{'='*20} Step 5: Split Master Data & Sequence Generation {'='*20}")
# ... (rest of step 5: splitting df_scaled into df_train_master, df_test_master) ...
df_train_master = df_scaled[df_scaled['year'] <= TRAIN_END_YEAR].copy()
df_test_master = df_scaled[df_scaled['year'] == FORECAST_START_YEAR].copy()
print(f"Master train data shape (<= {TRAIN_END_YEAR}): {df_train_master.shape}")
print(f"Master test data shape ({FORECAST_START_YEAR}): {df_test_master.shape}")

if df_train_master.empty:
    raise ValueError(
        f"Training data empty after split at year {TRAIN_END_YEAR}.")

sequence_file_path = os.path.join(
    RUN_OUTPUT_PATH,
    f'{CITY_NAME}_sequences_T{TIME_STEPS}_H{FORECAST_HORIZON_YEARS}{MODEL_SUFFIX}.joblib'
)
# Always generate for this script to ensure consistency with parameters
print(f"Generating training sequences (T={TIME_STEPS}, H={FORECAST_HORIZON_YEARS})...")
X_static_seq, X_dynamic_seq, X_future_seq, y_target_seq = reshape_xtft_data(
    df=df_train_master, 
    dt_col=DT_COL_NAME, 
    target_col=TARGET_COL_NAME,
    static_cols=static_features_model,
    dynamic_cols=dynamic_features_model,
    future_cols=future_features_model, 
    time_steps=TIME_STEPS,
    forecast_horizons=FORECAST_HORIZON_YEARS,
    spatial_cols=SPATIAL_COLS,
    savefile=sequence_file_path,
    verbose=7 # or 1 for minimum verbosity
)
if y_target_seq is None or X_dynamic_seq is None or y_target_seq.shape[0] == 0:
    min_len_needed = TIME_STEPS + FORECAST_HORIZON_YEARS
    raise ValueError(
        "Sequence generation failed or produced no sequences for training. "
        "Check data length per spatial group vs."
        f" (TIME_STEPS + FORECAST_HORIZON_YEARS = {min_len_needed}). "
        "For sampled data, ensure TIME_STEPS and"
        " FORECAST_HORIZON_YEARS are small enough."
    )
print(f"Training Sequences: S={X_static_seq.shape}, D={X_dynamic_seq.shape}, "
      f"F={X_future_seq.shape}, Y={y_target_seq.shape}")

# ==================================================================
# ** Step 6: Train-Validation Split of Sequence Data **
# ==================================================================
print(f"\n{'='*20} Step 6: Train-Validation Split of Sequences {'='*20}")
# ... train_test_split  ...
X_s_train, X_s_val, X_d_train, X_d_val, X_f_train, X_f_val, y_train, y_val = \
    train_test_split(
        X_static_seq, X_dynamic_seq, X_future_seq, y_target_seq,
        test_size=0.2, random_state=42, shuffle=True
    )
train_inputs = [X_s_train, X_d_train, X_f_train]
val_inputs = [X_s_val, X_d_val, X_f_val]
print(f"Train inputs: S={X_s_train.shape}, D={X_d_train.shape},"
      f" F={X_f_train.shape}, Y={y_train.shape}")
print(f"Val inputs  : S={X_s_val.shape}, D={X_d_val.shape},"
      " F={X_f_val.shape}, Y={y_val.shape}")

# ==================================================================
# ** Step 7: XTFT Model Definition, Compilation & Training **
# ==================================================================
print(f"\n{'='*20} Step 7: Model Training {'='*20}")
# ...  model definition, compile, fit ...
xtft_params = {
    'embed_dim': 32,
    'max_window_size': TIME_STEPS, 
    'memory_size': 50,
    'num_heads': 2, 
    'dropout_rate': 0.1, 
    'lstm_units': 32,
    'attention_units': 32, 
    'hidden_units': 32, 
    'scales': None,
    'multi_scale_agg': 'last', 
    'final_agg': 'last',
    'use_residuals': True,
    'use_batch_norm': False,
    'anomaly_detection_strategy': None
}
model_output_dim = y_train.shape[-1]
xtft_model_inst = TRANSFORMER_CLASS(
    static_input_dim=X_s_train.shape[-1], 
    dynamic_input_dim=X_d_train.shape[-1],
    future_input_dim=X_f_train.shape[-1],
    forecast_horizon=FORECAST_HORIZON_YEARS,
    quantiles=QUANTILES, 
    output_dim=model_output_dim,
    **xtft_params
)
loss_to_use = combined_quantile_loss(QUANTILES) if QUANTILES else 'mse'
xtft_model_inst.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE), loss=loss_to_use)
try:
    dummy_s = tf.zeros((1, X_s_train.shape[-1]), dtype=tf.float32)
    dummy_d = tf.zeros((1, TIME_STEPS, X_d_train.shape[-1]), 
                       dtype=tf.float32)
    dummy_f_shape_time = TIME_STEPS + FORECAST_HORIZON_YEARS
    dummy_f = tf.zeros((1, dummy_f_shape_time, X_f_train.shape[-1]),
                       dtype=tf.float32)
    xtft_model_inst([dummy_s, dummy_d, dummy_f])
    xtft_model_inst.summary(line_length=100)
except Exception as e_build: 
    print(f"Error during model build/summary: {e_build}")
    
early_stopping_cb = EarlyStopping(
    monitor='val_loss',
    patience=10, 
    restore_best_weights=True, 
    verbose=1
  )
checkpoint_filename = ( 
    f"{CITY_NAME}_xtft_model_H{FORECAST_HORIZON_YEARS}{MODEL_SUFFIX}"
    )
if SAVE_MODEL_AS_SINGLE_FILE:
    checkpoint_filename += ".keras"
    
model_checkpoint_path = os.path.join(RUN_OUTPUT_PATH, checkpoint_filename)

model_checkpoint_cb = ModelCheckpoint(
    filepath=model_checkpoint_path,
    monitor='val_loss', 
    save_best_only=True, 
    save_weights_only=False,
    verbose=1
   )

print(f"Model checkpoints will be saved to: {model_checkpoint_path}")
print("\nStarting XTFT model training...")
history = xtft_model_inst.fit(
    train_inputs, 
    y_train, 
    validation_data=(val_inputs, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping_cb, model_checkpoint_cb], 
    verbose=1
)
print("Best validation loss achieved:"
      f" {min(history.history.get('val_loss', [np.inf])):.4f}")
print(f"\nLoading best model from checkpoint: {model_checkpoint_path}")

try:
    custom_objects = {
        'combined_quantile_loss': loss_to_use} if QUANTILES else {}
    with custom_object_scope(custom_objects):
        xtft_model_loaded = load_model(model_checkpoint_path)
    print("Best model loaded successfully for forecasting.") 
    
except Exception as e_load:
    print(f"Error loading saved model: {e_load}."
          " Using model from end of training.")
    xtft_model_loaded = xtft_model_inst

# =============================================================================
# ** Step 8: Forecasting on Test Data (or Validation Data if Test is Empty) **
# =============================================================================
print(f"\n{'='*20} Step 8: Forecasting {'='*20}")
forecast_df_final = None
s_test_seq, d_test_seq, f_test_seq, y_test_actual_seq = None, None, None, None
inputs_for_final_forecast = val_inputs # Default to validation inputs
actuals_for_final_forecast = y_val     # Default to validation actuals
dataset_name_for_forecast = "ValidationSet"

if not df_test_master.empty:
    print(f"Attempting to generate sequences from test data (year {FORECAST_START_YEAR})...")
    try:
        s_test_seq, d_test_seq, f_test_seq, y_test_actual_seq = reshape_xtft_data(
            df=df_test_master, 
            dt_col=DT_COL_NAME, 
            target_col=TARGET_COL_NAME,
            static_cols=static_features_model, 
            dynamic_cols=dynamic_features_model,
            future_cols=future_features_model, 
            spatial_cols=SPATIAL_COLS,
            time_steps=TIME_STEPS, 
            forecast_horizons=FORECAST_HORIZON_YEARS,
            verbose=0 # Suppress logs for this reshape
        )
        if ( 
                y_test_actual_seq is not None 
                and d_test_seq is not None 
                and y_test_actual_seq.shape[0] > 0
        ):
            print(f"  Test sequences generated: S={s_test_seq.shape},"
                  f" D={d_test_seq.shape},F={f_test_seq.shape}"
                  f", Y={y_test_actual_seq.shape}")
            # Ensure dummy arrays have correct batch size for test sequences
            s_dummy_test = ( 
                s_test_seq if s_test_seq is not None 
                else np.zeros((d_test_seq.shape[0], 0), 
                              dtype=np.float32)
            )
            f_dummy_test = ( 
                f_test_seq if f_test_seq is not None 
                else np.zeros(
                    (d_test_seq.shape[0], 
                     d_test_seq.shape[1] + FORECAST_HORIZON_YEARS, 0), 
                    dtype=np.float32)
                )
            inputs_for_final_forecast = [
                s_dummy_test, d_test_seq, f_dummy_test]
            actuals_for_final_forecast = y_test_actual_seq
            dataset_name_for_forecast = f"TestSet (Year {FORECAST_START_YEAR})"
        else:
            print("  Warning: No sequences generated from test"
                  f" data (year {FORECAST_START_YEAR}). "
                  "This can happen if the test year has insufficient"
                  " data points per spatial group "
                  f"for T={TIME_STEPS}, H={FORECAST_HORIZON_YEARS}. "
                  "Forecasting on validation set instead.")
    except ValueError as e_reshape_test: # Catch error from reshape_xtft_data
        print("  Warning: Could not generate sequences"
              f" from test data: {e_reshape_test}. "
              "Forecasting on validation set instead.")
else:
    print(f"Test data (df_test_master for year {FORECAST_START_YEAR}) is empty. "
          "Forecasting on validation set.")

print(f"Generating predictions on {dataset_name_for_forecast} sequences...")
predictions_scaled = xtft_model_loaded.predict(
    inputs_for_final_forecast, verbose=0)

# Format predictions into a DataFrame
# Prepare spatial_data_array for format_predictions_to_dataframe
# It should correspond to the samples being predicted (inputs_for_final_forecast[0])
spatial_data_for_formatting = inputs_for_final_forecast[0] # This is s_test_seq or X_s_val

forecast_df_final = format_predictions_to_dataframe(
    predictions=predictions_scaled,
    y_true_sequences=actuals_for_final_forecast,
    target_name=TARGET_COL_NAME,
    quantiles=QUANTILES,
    forecast_horizon=FORECAST_HORIZON_YEARS,
    output_dim=model_output_dim,
    spatial_data_array=spatial_data_for_formatting,
    spatial_cols_names=SPATIAL_COLS,
    spatial_cols_indices=[
        static_features_model.index(col) 
        for col in SPATIAL_COLS 
        if col in static_features_model
        ],
    scaler= scaler_main if scaler_main is not None else None,
    scaler_feature_names=( 
        numerical_cols_to_scale 
        if numerical_cols_to_scale is not None else None),
    target_idx_in_scaler=(
        numerical_cols_to_scale.index(TARGET_COL_NAME) 
        if numerical_cols_to_scale is not None 
        and TARGET_COL_NAME in numerical_cols_to_scale 
        else None),
    evaluate_coverage = True if QUANTILES else False,
    verbose=1
)
if forecast_df_final is not None and not forecast_df_final.empty:
    forecast_csv_filename = ( 
        f"{CITY_NAME}_forecast_{dataset_name_for_forecast.replace(' ', '_')}"
        f"_{FORECAST_YEARS[0]}-{FORECAST_YEARS[-1]}{MODEL_SUFFIX}.csv"
        )
    forecast_csv_path = os.path.join(
        RUN_OUTPUT_PATH, forecast_csv_filename)
    forecast_df_final.to_csv(forecast_csv_path, index=False)
    print(f"Forecast results for {dataset_name_for_forecast}"
          f" saved to: {forecast_csv_path}")
    print("\nSample of final forecast DataFrame:")
    print(forecast_df_final.head())
else:
    print("No final forecast DataFrame generated.")

#%
# ==================================================================
# ** Step 9: Visualize Forecasts **
# ==================================================================
print(f"\n{'='*20} Step 9: Visualize Forecasts {'='*20}")
if forecast_df_final is not None and not forecast_df_final.empty:
    print(f"Visualizing forecasts for {dataset_name_for_forecast}"
          f" (spatial plot for first forecast year)...")
    # Ensure SPATIAL_COLS are present for plotting
    if not all(col in forecast_df_final.columns for col in SPATIAL_COLS):
        print(f"Warning: Spatial columns {SPATIAL_COLS}"
              " not found in final forecast DataFrame. "
              "Spatial plot might be incorrect or fail."
             )
        
        # Attempt to add them if they were part of the static features used for prediction
        if inputs_for_final_forecast[0] is not None and \
           all(scol in static_features_model for scol in SPATIAL_COLS):
            s_indices_viz = [
                static_features_model.index(col) for col in SPATIAL_COLS]
            spatial_vals_from_s_input = inputs_for_final_forecast[0][:, s_indices_viz]
            
            # We need to repeat these spatial values for each forecast step
            # The forecast_df_final is long: (num_samples * horizon_steps, features)
            # The spatial_vals_from_s_input is (num_samples, num_spatial_features)
            if ( 
                len(spatial_vals_from_s_input) * FORECAST_HORIZON_YEARS == len(forecast_df_final)
                ):
                repeated_spatial_vals = ( 
                    np.repeat(spatial_vals_from_s_input, 
                              FORECAST_HORIZON_YEARS, axis=0)
                    )
                for i, col_name in enumerate(SPATIAL_COLS):
                    forecast_df_final[col_name] = repeated_spatial_vals[:, i]
                print("  Added spatial columns to"
                      " forecast_df_final from prediction inputs.")
            else:
                 print("  Could not align spatial data"
                       f" (len {len(spatial_vals_from_s_input)}) "
                       f"with forecast_df_final (len {len(forecast_df_final)}).")
        else:
            print("  Cannot add spatial columns as static input"
                  " data is missing or incomplete for SPATIAL_COLS.")

    plot_forecasts(
        forecast_df=forecast_df_final,
        target_name=TARGET_COL_NAME,
        quantiles=QUANTILES,
        output_dim=model_output_dim,
        kind="spatial",
        horizon_steps=[1, 2,],  # 1/2 Plot for the first/second year of the horizon
                                # [1, 2] # Plot the two as subplots row1/row2
        spatial_cols=SPATIAL_COLS,
        max_cols=1,
        figsize_per_subplot=(8, 7),
        titles = [
            f'Forecast_step{i+1}: year = {y}' 
            for i, y in enumerate(FORECAST_YEARS)], 
        verbose=1,
        s=20, 
        cmap="jet", 
        
    )
else:
    print("No forecast data to visualize.")

# ==================================================================
# ** Step 10: Save All Generated Figures **
# ==================================================================
print(f"\n{'='*20} Step 10: Save All Figures {'='*20}")
# ... (save_all_figures logic as before) ...
try:
    save_all_figures(
        output_dir=RUN_OUTPUT_PATH,
        prefix=f"{CITY_NAME}_xtft_plot_",
        fmts=['.png', '.pdf'] # Save as PNG and PDF
        )
    print(f"All open Matplotlib figures saved to: {RUN_OUTPUT_PATH}")
except Exception as e_save_fig:
    print(f"Could not save all figures: {e_save_fig}")

print(f"\n{'-'*_TW}\n{CITY_NAME.upper()} XTFT FORECASTING SCRIPT COMPLETED.\n"
      f"Outputs are in: {RUN_OUTPUT_PATH}\n{'-'*_TW}")

