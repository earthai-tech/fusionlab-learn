
"""
nansha_forecast_2022_2025.py

This script performs quantile-based subsidence prediction for the years
2022-2025 using the XTFT deep learning model. The dataset for Nansha
is used. For training, data up to 2021 is used, while 2022 might be
reserved for validation if a train/test split on the original DataFrame
is performed before sequencing. The model forecasts subsidence for
2022-2025 with a forecast horizon of 4 years.

Note on Data Import: This script (main.py) is primarily used for
reproducibility purposes and loads a subset of 2,000 samples from the
full Nansha subsidence data. The complete dataset is available upon
request. The 2,000 samples are sufficient for demonstrating the
forecasting process.

Workflow:
---------
1. Data preprocessing and feature engineering.
2. Encoding categorical features and normalizing numerical data.
3. Splitting the dataset into training and testing sets (conceptually,
   actual split happens after sequencing).
4. Reshaping data for XTFT model input using `reshape_xtft_data`.
5. Splitting the sequence data into train-validation sets.
6. Training and evaluating the XTFT model.
7. Generating forecasts for the specified future years.
8. Visualizing the actual versus predicted subsidence.

Author: LKouadio

"""

# ==========================================
#  SECTION 1: LIBRARY IMPORTS & CONFIG
# ==========================================
import os
import shutil
import time
from datetime import datetime
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Ensure numpy is imported
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# FusionLab - Custom Modules
try:
    from fusionlab.api.util import get_table_size
    from fusionlab.datasets import fetch_nansha_data 
    from fusionlab.datasets._property import get_data
    from fusionlab.nn.losses import combined_quantile_loss
    from fusionlab.nn.transformers import SuperXTFT, XTFT
    from fusionlab.nn.utils import (
        forecast_multi_step,
        reshape_xtft_data_in,
    )
    from fusionlab.plot.forecast import visualize_forecasts
    from fusionlab.utils.data_utils import nan_ops
    from fusionlab.utils.io_utils import fetch_joblib_data
except ImportError as e:
    print(f"Error importing fusionlab modules: {e}. "
          "Please ensure fusionlab-learn is installed correctly.")
    raise

# Suppress TensorFlow/Keras logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress C++ level logs
tf.get_logger().setLevel('ERROR')
if hasattr(tf, 'autograph') and hasattr(tf.autograph, 'set_verbosity'):
    tf.autograph.set_verbosity(0)

# Get terminal width for pretty printing, default to 80
try:
    _TW = get_table_size()
except Exception:
    _TW = 80

# =================== CONFIGURATION PARAMETERS ===================
# Toggle to use the SuperXTFT variant; if False, use standard XTFT.
USE_SUPER = False

# Suffix for file naming based on transformer type.
super_ext = '_super' if USE_SUPER else ''

# Transformer model selection.
TRANSFORMER_ = SuperXTFT if USE_SUPER else XTFT

# Years to forecast.
forecast_years = [2022, 2023, 2024, 2025] # Nansha forecast period

# Number of past time steps for input sequence length.
# This should be carefully chosen based on data characteristics.
# For this example, using 3 as per the script's original setting.
time_steps = 3

# Spatial columns for grouping data. Set to None if not applicable
# or if data is already for a single spatial entity.
spatial_cols = None # Example: ["longitude", "latitude"]

# Quantile values for probabilistic forecasting.
# Set to None for deterministic (point) forecasts.
quantiles = [0.1, 0.5, 0.9]

# Anomaly detection strategy.
# Options: None, 'feature_based', 'prediction_based', 'from_config'.
anomaly_detection_strategy = None

# Training hyperparameters
EPOCHS = 50 # Number of epochs for training
LEARNING_RATE = 0.001 # Learning rate for the Adam optimizer
BATCH_SIZE = 32 # Batch size for training

city_name ='nansha'

# Paths
# Default data path from fusionlab.datasets._property
# This is typically ~/fusionlab_data when using #get_data()
data_path = get_data() #'/results/' # get_data() # use it for fetching nansha data remotely and use software path  to save results 
# Create a subdirectory within data_path for this specific run's artifacts
run_output_path = os.path.join(data_path,  f"nansha_forecast_run{super_ext}")

# Clean any previous contents for a fresh run
if os.path.isdir(run_output_path):
    shutil.rmtree(run_output_path)

os.makedirs(run_output_path, exist_ok=True)
print(f"Output artifacts will be saved in: {run_output_path}")

# ------------------------------------------------------------------
# RUNTIME OPTIONS
# ------------------------------------------------------------------
# save_intermediate_artefacts  –  If True, the script exports all
#     mid‑pipeline artefacts (cleaned/encoded dataframe, train–test
#     master CSVs, fitted scaler) to the current run directory.  
#     Enables easy debugging and guarantees full reproducibility.

save_intermediate_artefacts = True

# save_single_file_archive     –  Governs the checkpoint format:  
#       - True   →  write a compact single‑file “.keras” archive  
#                   (portable, but omits variables/ and assets/ dirs).  
#       - False  →  write a full TensorFlow SavedModel directory  
#                   containing saved_model.pb, variables/, assets/, etc.

save_single_file_archive    = True

# ==================================================================
# ** Step 1: Load Nansha Dataset **
# ------------------------------------------------------------------
print(f"\n{'='*20} Step 1: Load Nansha Dataset {'='*20}")
# First, attempt to fetch the Nansha data using the original method.
try:
    nansha_df = fetch_nansha_data(as_frame=True, verbose=1)
    print(f"Nansha data loaded successfully. Shape: {nansha_df.shape}")
except Exception as e:
    print(f"ERROR: Failed to load Nansha data via fetch_nansha_data: {e}")
    print("Attempting to load the dataset from the 'data/' folder...")

    # Fallback to loading the data from the local 'data/nansha_2000.csv' in the Code Ocean capsule
    import os
    print(f"Current working directory: {os.getcwd()}")
    try:
        nansha_df = pd.read_csv('../data/nansha_2000.csv')
        print(f"Nansha data loaded from local CSV. Shape: {nansha_df.shape}")
    except Exception as e:
        print(f"ERROR: Failed to load Nansha data from local CSV: {e}")
        raise  # Raising error to stop execution if the dataset cannot be loaded

# Backup original dataset (optional, for reference)
# nansha_data_original = nansha_df.copy()

# ==================================================================
# ** Step 2: Feature Selection & Initial Cleaning **
# ------------------------------------------------------------------
print(f"\n{'='*20} Step 2: Feature Selection & Cleaning {'='*20}")
# Define features to be used based on the Nansha dataset description
# Ensure these column names match exactly those in nansha_2000.csv
selected_features = [
    'longitude', 'latitude', 'year',
    'GWL', 'rainfall_mm', 'geology', # 'geological_category' was renamed
    'soil_thickness', 'building_concentration',
    'normalized_seismic_risk_score', 'subsidence' # Target
]
# Check if all selected features exist
missing_cols = [f for f in selected_features if f not in nansha_df.columns]
if missing_cols:
    print(f"Warning: The following selected columns are missing "
          f"from the loaded data: {missing_cols}")
    print(f"Available columns: {nansha_df.columns.tolist()}")
    # Filter selected_features to only include available columns
    selected_features = [f for f in selected_features if f in nansha_df.columns]
    if not selected_features:
        raise ValueError("No selected features remain after checking availability.")

nansha_df_selected = nansha_df[selected_features].copy()
print(f"Selected features. Shape: {nansha_df_selected.shape}")

# Check and handle NaN values
print(f"NaN exists before processing? "
      f"{nansha_df_selected.isna().any().any()}")
# Using nan_ops to fill NaNs (e.g., ffill then bfill)
nansha_df_cleaned = nan_ops(
    nansha_df_selected, ops='sanitize',
    action='fill', process="do_anyway"
    )
print(f"NaN exists after processing? "
      f"{nansha_df_cleaned.isna().any().any()}")

# ==================================================================
# ** Step 3: Feature Engineering & Normalization **
# ------------------------------------------------------------------
print(f"\n{'='*20} Step 3: Feature Engineering & Normalization {'='*20}")

# 3a. Encode Categorical Features
# 'geology' is categorical. 'building_concentration' might be if it's binned.
# The script uses OneHotEncoder for 'geology' and 'building_concentration'.
# Let's assume 'building_concentration' in nansha_2000.csv is numerical
# and was intended to be binned or is treated as such by OHE.
# If it's already numerical and continuous, it shouldn't be one-hot encoded.
# For this script, we follow the original logic.

df_for_encoding = nansha_df_cleaned.copy()
encoded_dfs = []
categorical_to_encode = ['geology'] # Explicitly from Nansha columns
# Add 'building_concentration' if it's meant to be categorical
# For now, assume it is for consistency with script's OHE usage
if 'building_concentration' in df_for_encoding.columns:
    categorical_to_encode.append('building_concentration')

encoder_info = {} # To store fitted encoders or column names

for col in categorical_to_encode:
    if col in df_for_encoding.columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore',
                                dtype=np.float32)
        encoded_data = encoder.fit_transform(df_for_encoding[[col]])
        new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
        encoded_df = pd.DataFrame(
            encoded_data, columns=new_cols, index=df_for_encoding.index
            )
        encoded_dfs.append(encoded_df)
        encoder_info[col] = {'encoder': encoder, 'columns': new_cols}
        print(f"  Encoded '{col}' into {len(new_cols)} columns.")
    else:
        print(f"  Warning: Categorical column '{col}' not found for encoding.")

# Drop original categorical columns and concatenate encoded ones
df_processed = df_for_encoding.drop(columns=categorical_to_encode, errors='ignore')
if encoded_dfs:
    df_processed = pd.concat([df_processed] + encoded_dfs, axis=1)

print(f"Shape after encoding: {df_processed.shape}")

# 3b. Normalize Numerical Features
# Exclude coordinates, year, and already one-hot encoded features.
# Target ('subsidence') should also be scaled.
cols_to_scale = [
    'GWL', 'rainfall_mm', 'normalized_seismic_risk_score',
    'soil_thickness', 'subsidence'
]
# If 'building_concentration' was NOT one-hot encoded and is numerical:
if 'building_concentration' in df_processed.columns and \
   'building_concentration' not in categorical_to_encode:
    cols_to_scale.append('building_concentration')

# Filter to only columns present in df_processed
cols_to_scale = [col for col in cols_to_scale if col in df_processed.columns]

if cols_to_scale:
    scaler = MinMaxScaler() # As per script
    df_processed[cols_to_scale] = scaler.fit_transform(
        df_processed[cols_to_scale]
        )
    # Save the scaler
    scaler_path = os.path.join(run_output_path, "nansha_minmax_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Numerical features scaled and scaler saved to: {scaler_path}")
else:
    print("No numerical columns found or specified for scaling.")


# ==================================================================
# ** Step 4: Define Feature Sets for Reshaping **
# ------------------------------------------------------------------
print(f"\n{'='*20} Step 4: Define Feature Sets {'='*20}")

target_col_name = 'subsidence'
dt_col_name = 'year' # 'year' acts as the time index

# Static features: coordinates + encoded categoricals
static_feature_names = ['longitude', 'latitude']
for col_info in encoder_info.values():
    static_feature_names.extend(col_info['columns'])
# Ensure only existing columns are included
static_feature_names = [c for c in static_feature_names if c in df_processed.columns]

# Dynamic features: scaled numericals (excluding target) + time-varying like 'year'
# The script logic implies 'year' itself is not a dynamic input feature for the sequence,
# but rather the time index for creating sequences.
dynamic_feature_names = [
    'GWL', 'rainfall_mm', 'normalized_seismic_risk_score',
    'soil_thickness'
    # Add 'building_concentration' if it was scaled and not encoded
]
if 'building_concentration' in cols_to_scale and \
   'building_concentration' not in categorical_to_encode:
    dynamic_feature_names.append('building_concentration')
dynamic_feature_names = [c for c in dynamic_feature_names if c in df_processed.columns]

# Future features: rainfall_mm (as per script)
future_feature_names = ['rainfall_mm']
future_feature_names = [c for c in future_feature_names if c in df_processed.columns]

print(f"  Static features for model: {static_feature_names}")
print(f"  Dynamic features for model: {dynamic_feature_names}")
print(f"  Future features for model: {future_feature_names}")

# ==================================================================
# ** Step 5: Data Splitting (Conceptual) & Sequencing **
# ------------------------------------------------------------------
print(f"\n{'='*20} Step 5: Data Splitting & Sequencing {'='*20}")
# The script splits by year: train <= 2021, test = 2022 for Nansha.
# We apply this split on df_processed BEFORE reshaping.

df_train_master = df_processed[df_processed['year'] <= 2021].copy()
df_test_master = df_processed[df_processed['year'] == 2022].copy()

print(f"Master train data shape (years <= 2021): {df_train_master.shape}")
print(f"Master test data shape (year 2022): {df_test_master.shape}")

if save_intermediate_artefacts:
    # --------------------------------------------------------------
    # Persist intermediate artefacts for debugging / reuse
    # --------------------------------------------------------------
    print("Saving intermediate artefacts …")

    #  Cleaned & encoded frame used for the split
    cleaned_csv = os.path.join(data_path, f"{city_name}_processed_full.csv")
    df_processed.to_csv(cleaned_csv, index=False)

    # Train / test master sets
    train_csv = os.path.join(data_path, f"{city_name}_train_master.csv")
    test_csv  = os.path.join(data_path, f"{city_name}_test_master.csv")
    df_train_master.to_csv(train_csv, index=False)
    df_test_master.to_csv(test_csv,  index=False)

    # Scaler object (re‑dump in case user skipped earlier cell)
    scaler_path = os.path.join(data_path, f"{city_name}_minmax_scaler.joblib")
    if not os.path.isfile(scaler_path):
        joblib.dump(scaler, scaler_path)

    # Log summary
    print(f"    Processed data  ➜ {cleaned_csv}")
    print(f"    Train master    ➜ {train_csv}")
    print(f"    Test  master    ➜ {test_csv}")
    print(f"    Scaler object   ➜ {scaler_path}")
    print(" Artefact saving complete.")

# Sequence generation parameters
# forecast_horizon derived from forecast_years
current_forecast_horizon = len(forecast_years)
# time_steps is defined in CONFIG
current_time_steps = time_steps

print(f"Reshaping training data (T={current_time_steps}, H={current_forecast_horizon})...")

# Attempt to load cached sequences first
sequence_file = os.path.join(
    run_output_path,
    f'nansha_sequences_T{current_time_steps}_H{current_forecast_horizon}{super_ext}.joblib'
)

if os.path.isfile(sequence_file):
    print(f"Loading preprocessed sequences from: {sequence_file}")
    (X_static_seq, X_dynamic_seq, X_future_seq, y_target_seq
     ) = fetch_joblib_data(
        sequence_file,
        'static_data', 'dynamic_data', 'future_data', 'target_data',
        verbose=1,
    )
else:
    print("Generating sequences from scratch...")
    X_static_seq, X_dynamic_seq, X_future_seq, y_target_seq = reshape_xtft_data_in(
        df_train_master, # Use the master training data
        dt_col=dt_col_name,
        target_col=target_col_name,
        static_cols=static_feature_names,
        dynamic_cols=dynamic_feature_names,
        future_cols=future_feature_names,
        time_steps=current_time_steps,
        forecast_horizons=current_forecast_horizon,
        spatial_cols=spatial_cols, # None in this script's config
        # savefile=sequence_file, # Save the generated sequences ; NOTE: uncomment this for Python>=3.10 ( more robust) to auto-save file and comment the explicit saved data;
        verbose=1
    )
    #  Explicitly save with joblib (avoids importlib.metadata) # Comment this for Python 3>=3.10 
    seq_dict = {
        "static":  X_static_seq,
        "dynamic": X_dynamic_seq,
        "future":  X_future_seq,
        "target":  y_target_seq,
    }
    joblib.dump(seq_dict, sequence_file)
    print(f"[INFO] Sequences saved to: {sequence_file}")

print("\nSequence shapes for training input:")
print(f"  Static : {X_static_seq.shape if X_static_seq is not None else 'None'}")
print(f"  Dynamic: {X_dynamic_seq.shape}")
print(f"  Future : {X_future_seq.shape if X_future_seq is not None else 'None'}")
print(f"  Target : {y_target_seq.shape}")

# ==================================================================
# ** Step 6: Train-Validation Split of Sequences **
# ------------------------------------------------------------------
print(f"\n{'='*20} Step 6: Train-Validation Split {'='*20}")
# Split the generated sequences for model training
X_static_train, X_static_val, \
X_dynamic_train, X_dynamic_val, \
X_future_train, X_future_val, \
y_train, y_val = train_test_split(
    X_static_seq, X_dynamic_seq, X_future_seq, y_target_seq,
    test_size=0.2, # 20% for validation
    random_state=42, # For reproducibility
    shuffle=True # Shuffle sequences, as they are already time-ordered windows
)

print("Training set shapes:")
print(f"  Static: {X_static_train.shape if X_static_train is not None else 'None'}")
print(f"  Dynamic: {X_dynamic_train.shape}")
print(f"  Future: {X_future_train.shape if X_future_train is not None else 'None'}")
print(f"  Target: {y_train.shape}")
print("\nValidation set shapes:")
print(f"  Static: {X_static_val.shape if X_static_val is not None else 'None'}")
print(f"  Dynamic: {X_dynamic_val.shape}")
print(f"  Future: {X_future_val.shape if X_future_val is not None else 'None'}")
print(f"  Target: {y_val.shape}")

# Package inputs for model.fit
train_inputs = [X_static_train, X_dynamic_train, X_future_train]
val_inputs = [X_static_val, X_dynamic_val, X_future_val]

# ==================================================================
# ** Step 7: Model Training **
# ------------------------------------------------------------------
print(f"\n{'='*20} Step 7: Model Training {'='*20}")

# Best hyperparameters (example, should be from tuning)
best_params = {
    'embed_dim': 32,
    'max_window_size': current_time_steps, # Match data prep
    'memory_size': 100,
    'num_heads': 4,
    'dropout_rate': 0.1,
    'lstm_units': 64,
    'attention_units': 64,
    'hidden_units': 32,
    'multi_scale_agg': 'auto',
    'anomaly_detection_strategy': anomaly_detection_strategy,
    'use_residuals': True,
    'use_batch_norm': True, # As per script's example
    'final_agg': 'last',
}

# Instantiate the selected XTFT model (XTFT or SuperXTFT)
xtft_model = TRANSFORMER_(
    static_input_dim=X_static_train.shape[-1] if X_static_train is not None else 0,
    dynamic_input_dim=X_dynamic_train.shape[-1],
    future_input_dim=X_future_train.shape[-1] if X_future_train is not None else 0,
    forecast_horizon=current_forecast_horizon,
    quantiles=quantiles,
    output_dim=y_train.shape[-1], # Should be 1 for subsidence
    **best_params
)

# Select appropriate loss
if quantiles is not None:
    loss_fn = combined_quantile_loss(quantiles)
else:
    loss_fn = 'mse' # For point forecasts

# Compile the model
xtft_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn
)

# Dummy call to build the model and print summary
# Ensure inputs match expected structure (list of 3, some can be None if model handles)
# For XTFT, all three are typically expected by default constructor after validation
dummy_call_inputs = [
    tf.zeros_like(X_static_train[:1]) if X_static_train is not None else None,
    tf.zeros_like(X_dynamic_train[:1]),
    tf.zeros_like(X_future_train[:1]) if X_future_train is not None else None,
]
# Filter out None inputs if the model's call method expects only non-None
# For XTFT/SuperXTFT, the call method expects a list of 3.
# The internal `validate_xtft_inputs` handles None checks based on *_input_dim.
if dummy_call_inputs[0] is None and xtft_model.static_input_dim is not None:
    print("Warning: Static input is None but model expects static_input_dim.")
if dummy_call_inputs[2] is None and xtft_model.future_input_dim is not None:
    print("Warning: Future input is None but model expects future_input_dim.")

try:
    xtft_model(dummy_call_inputs) # Build the model
    xtft_model.summary(line_length=90)
except Exception as e:
    print(f"Error during model build/summary: {e}")


# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, # Increased patience
    restore_best_weights=True, verbose=1
)
# -----------------------------------------------------------------
# Model‑checkpoint path
#   - If you append **“.keras”** (or “.h5”), Keras writes a single‑file
#     archive.  Handy for quick portability, but it omits the standard
#     SavedModel sub‑folders (`variables/`, `assets/`, `fingerprint.pb`).
#   - Leaving **no extension** (or adding a trailing slash) tells Keras
#     to save a full TensorFlow SavedModel directory that includes:
#         saved_model.pb
#         variables/            ← all trainable / non‑trainable weights
#         assets/ (optional)    ← vocab files, lookup tables, etc.
#         fingerprint.pb        ← integrity hash (TF ≥ 2.14)
#   • The paper’s reproducibility package expects the *directory* form
#     so that downstream scripts (e.g., TFLite, TF‑Serving) can load
#     the model without extra conversion steps.
# -----------------------------------------------------------------
if save_single_file_archive:
    # single portable file  →  *.keras
    model_checkpoint_path = os.path.join(
        run_output_path,
        f"nansha_xtft_model_H{current_forecast_horizon}{super_ext}.keras"
    )
else:
    # full TensorFlow SavedModel directory  →  no extension
    model_checkpoint_path = os.path.join(
        run_output_path,
        f"nansha_xtft_model_H{current_forecast_horizon}{super_ext}"
    )


model_checkpoint = ModelCheckpoint(
    filepath=model_checkpoint_path, monitor='val_loss',
    save_best_only=True, save_weights_only=False,
    verbose=1, 
    save_format='tf' # Save entire model in TF format # commnent
)

print("\nTraining the XTFT model...")
start_time = time.time()
history = xtft_model.fit(
    train_inputs,
    y_train,
    validation_data=(val_inputs, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)
training_time = (time.time() - start_time) / 60
print(f"Training completed in {training_time:.2f} minutes.")
print(f"Best validation loss: {min(history.history.get('val_loss', [np.inf])):.4f}")

# Load the best model saved by checkpoint
print(f"\nLoading best model from: {model_checkpoint_path}")
try:
    with custom_object_scope({'combined_quantile_loss': loss_fn} if quantiles else {}):
        loaded_model = load_model(model_checkpoint_path)
    print("Best model loaded successfully.")
    xtft_model = loaded_model # Use the best saved model
except Exception as e:
    print(f"Error loading saved model: {e}. Using model from end of training.")

#%
# ==================================================================
# ** Step 8: Forecasting & Evaluation **
# ------------------------------------------------------------------
print(f"\n{'='*20} Step 8: Forecasting & Evaluation {'='*20}")

# Ensure the last_train_date is a proper datetime object
last_train_date = pd.to_datetime(df_train_master[dt_col_name].max(), format='%Y')
# forecast_start_date should now work correctly with DateOffset
forecast_start_date = last_train_date + pd.DateOffset(years=1)  # Since dt_col is 'year'

# Print to check the forecast start date
print(f"Forecast start date: {forecast_start_date}")

# Generate forecast dates based on horizon and original data frequency
# forecast_years is predefined as the list [2022, 2023, 2024, 2025]
forecast_years = [forecast_start_date.year + i for i in range(current_forecast_horizon)]

print(f"Generating forecasts for years: {forecast_years}")

forecast_csv_path = os.path.join(
    run_output_path, f"nansha_forecast_{forecast_years[0]}-{forecast_years[-1]}{super_ext}.csv"
)

# The `inputs` for forecast_multi_step should be the complete set of
# sequences you want to predict on (e.g., X_static_seq, X_dynamic_seq, X_future_seq).
# The `y` argument is for evaluation against true values.
# Here, we'll use the full `y_target_seq` for evaluation.

# If `df_test_master` is the hold-out set, we'd need to reshape it too.
# For this example, let's predict on the validation sequences `val_inputs`
# and evaluate against `y_val`.

# To evaluate on the conceptual "test_data" (year 2022):
# Reshape df_test_master into sequences
if not df_test_master.empty:
    print("Reshaping master test data (year 2022) for evaluation...")
    s_test_master, d_test_master, f_test_master, y_test_master = reshape_xtft_data_in(
       df_test_master, dt_col=dt_col_name, target_col=target_col_name,
       static_cols=static_feature_names, dynamic_cols=dynamic_feature_names,
       future_cols=future_feature_names, spatial_cols=spatial_cols,
       time_steps=current_time_steps, forecast_horizons=current_forecast_horizon,
       verbose=0 # Suppress verbose for this internal reshape
    )
    test_inputs_for_forecast = [s_test_master, d_test_master, f_test_master]
    y_for_eval = y_test_master
    eval_forecast_years = [df_test_master[dt_col_name].unique()[0]] * current_forecast_horizon
    # This assumes forecast_horizon matches the number of years in test set.
    # For multi-year test set, forecast_dt needs careful construction.
    # For single year 2022 test, horizon 1:
    if current_forecast_horizon == 1 and len(df_test_master[dt_col_name].unique()) == 1:
        eval_forecast_years = [df_test_master[dt_col_name].unique()[0]]
    else: # For multi-horizon prediction on 2022 data, need to define future years
        # This logic is tricky if test_data is only one year but horizon is > 1
        # For now, let's assume we are forecasting from the end of training data
        print("Using validation set for multi-step forecast demonstration.")
        test_inputs_for_forecast = val_inputs
        y_for_eval = y_val
        # Create future dates for the validation set's forecast
        # This requires knowing the last date of the *input sequences* for val_inputs
        # For simplicity, we'll use the global forecast_years for output formatting.

else:
    print("Skipping visualization as test_data (year 2022) is empty.")

forecast_df = forecast_multi_step(
    xtft_model=xtft_model,
    inputs=test_inputs_for_forecast,  # Predict on validation or test sequences
    forecast_horizon=current_forecast_horizon,
    y=y_for_eval,  # Evaluate against corresponding true values
    dt_col=dt_col_name,  # For output DataFrame structure
    mode="quantile" if quantiles else "point",
    q=quantiles,
    tname=target_col_name,
    forecast_dt=forecast_years,  # For labeling output columns if needed by step_to_long
    apply_mask=False,  # Example:  No masking ; if 'True', consider masking the delta river locations
    # mask_values=0, 
    # mask_fill_value=0.0, # ignore the delta river part.
    savefile=forecast_csv_path,
    spatial_cols=spatial_cols,  # Pass spatial_cols for correct DataFrame structure
    verbose=7
)

print(f"Forecast results saved to: {forecast_csv_path}")

# ==================================================================
# ** Step 9: Visualize Forecasts **
# ------------------------------------------------------------------
print(f"\n{'='*20} Step 9: Visualize Forecasts {'='*20}")
# Visualize the forecasts against actuals (if available in test_data for eval_periods)
# `forecast_df` from forecast_multi_step is already in long format.
# `test_data` for visualize_forecasts should be the original unsequenced test data
# for the periods being visualized.

# For visualization, we need to merge original test data features with predictions
# or ensure visualize_forecasts can handle the long format from forecast_multi_step.
# The current visualize_forecasts expects a forecast_df that might need
# coordinates if kind='spatial'. The forecast_df from forecast_multi_step
# should already include spatial_cols if they were passed.

# We need the original df_test_master for actual values if evaluating visually
if not df_test_master.empty:
    print("Visualizing forecasts against actual test data (year 2022)...")
    # Ensure forecast_df has longitude and latitude columns for visualization
    ll_df = pd.DataFrame(test_inputs_for_forecast[0][:, :2], columns=["longitude", "latitude"])
    ll_dupl = pd.concat([ll_df] * len(forecast_years), ignore_index=True)
    assert len(ll_dupl) == len(forecast_df), " Mismatch in coordinate and forecast length."

    # Merge coordinates with forecast
    forecast_df_full = pd.concat([ll_dupl, forecast_df], axis=1)

    # Optional: visualize coordinate grid (for sanity check)
    plt.scatter(forecast_df_full["longitude"], forecast_df_full["latitude"], alpha=0.5)
    plt.title("Forecast Coordinate Grid")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()

    # Visualize final spatial-temporal forecast
    print("Visualizing forecast results...\n")
    visualize_forecasts(
        forecast_df_full,  # Already merged with coordinates
        dt_col="year",
        tname="subsidence",
        eval_periods=forecast_years,
        test_data=df_test_master,
        mode="quantile" if quantiles else "point",
        kind="spatial",
        x="longitude",
        y="latitude",
        max_cols=2,
        axis="off",
        verbose=7, 
        cmap="jet_r", 
        s=10, 
    )
else:
    print("Skipping visualization as test_data (year 2022) is empty.")

print("\n" + "=" * _TW)
print("Nansha Case Study Workflow Completed.")
print("WARNING: This script uses sampled data (nansha_2000.csv) for "
      "demonstration.")
print("For optimal results and research, use the full dataset and "
      "perform thorough hyperparameter tuning.")
print("=" * _TW)

try: 
    from fusionlab.utils.generic_utils import save_all_figures 
except ImportError: # for previous version 

    # Dummy Function to catch and save all figures automatically
    def save_all_figures():
        # Get the list of all open figure numbers
        all_figs = plt.get_fignums()
        
        for fig_num in all_figs:
            # Set the current figure by figure number
            fig = plt.figure(fig_num)
            
            # Define the path to save the figure
            filename = f"figure_{fig_num}.png"  # You can customize the filename format
            plot_path = os.path.join(run_output_path, filename)
            
            # Save the current figure
            fig.savefig(plot_path)
            print(f"Figure {fig_num} saved to: {plot_path}")
            
            # Close the figure to free up memory
            plt.close(fig)
finally: 
    save_all_figures () 
    


save_all_figures()