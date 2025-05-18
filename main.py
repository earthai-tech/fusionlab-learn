# -*- coding: utf-8 -*-
"""
nansha_forecast_2022_2025.py

This script performs quantile-based subsidence prediction for the years 
2022-2025 using the XTFT deep learning model. The dataset for Zhongshan 
spans from 2015 to 2023. For training, data from 2015 to 2021 is used, 
while 2022 is reserved for validation. The model forecasts subsidence for 
2022-2025 with a forecast horizon of 4 years and a rolling window (time_steps) of 3.

Note on Data Import: This script (main.py) is primarily used for reproducibility 
purposes and only loads a subset of 2,000 samples from the full 2,449,321 records 
of Nansha subsidence data. The complete dataset is available upon request. 
However, the 2,000 samples are sufficient for demonstrating the forecasting process 
and validating predictions.

Workflow:
---------
1. Data preprocessing and feature engineering.
2. Encoding categorical features and normalizing numerical data.
3. Splitting the dataset into training (2015-2022) and testing (2023) sets.
4. Reshaping data for XTFT model input using `reshape_xtft_data`.
5. Splitting the data into train-validation sets for model training.
6. Training and evaluating the XTFT model.
7. Generating forecasts for 2023-2026 and validating the 2023 prediction against 
   actual observations.
8. Visualizing the actual versus predicted subsidence.

Author: LKouadio
"""


# ==========================================
#  SECTION 1: LIBRARY IMPORTS & DATA LOADING
# ==========================================

# Standard Library
import os
import time
from datetime import datetime

# Third-Party Libraries
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# FusionLab - Custom Modules
from fusionlab.api.util import get_table_size
from fusionlab.datasets import fetch_nansha_data
from fusionlab.datasets._property import get_data
from fusionlab.nn.losses import combined_quantile_loss
from fusionlab.nn.transformers import SuperXTFT, XTFT
from fusionlab.nn.utils import (
    forecast_multi_step,
    reshape_xtft_data,
    visualize_forecasts
)
from fusionlab.utils.data_utils import nan_ops
from fusionlab.utils.io_utils import fetch_joblib_data

_TW = get_table_size()
# =================== CONFIGURATION PARAMETERS =============================

# Toggle to use the SuperXTFT variant; if False, use the standard XTFT.
USE_SUPER = False  

# Suffix for file naming or model labeling depending on transformer type.
super_ext = '' if not USE_SUPER else '_super'

# Transformer model selection based on the USE_SUPER flag.
# TRANSF_ refers to the selected fusion transformer model.
TRANSF_ = SuperXTFT if USE_SUPER else XTFT  

# Years to forecast; modify based on the prediction scope (e.g., Nansha).
forecast_years = [2022, 2023, 2024, 2025]  # e.g., [2023, 2024, 2025, 2026] for Nansha

# Number of time steps to consider for input sequence length.
time_steps = 3
# NOTE: Spatial columns are required for models involving spatial input.
# For reproducibility or limited datasets, you may set this to None.
spatial_cols =None # Example: ("longitude", "latitude")

# Quantile values for probabilistic forecasting; use None for point prediction.
quantiles = [0.1, 0.5, 0.9]  # Set to None for deterministic forecasts

# Strategy used for anomaly detection; 'feature_based' is one of the supported modes.
anomaly_detection_strategy = None #'feature_based'

# Training hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.001

# ==========================================================================
# ** Step 1: Define Data Paths**
# ------------------------------------------

data_path = get_data()

# data_path = os.path.join(main_path, 'qt_forecast_2023_2026')

# # Load dataset
# nansha_file = os.path.join(main_path, 'nansha_filtered_final_data.csv')

# nansha_data = fetch_zhongshan_data().frame
nansha_data = fetch_nansha_data().frame

# nansha_data = pd.read_csv(r'J:\nature_data\final\nansha_data.csv')
# Rename geological category column for consistency
nansha_data.rename(columns={"geological_category": "geology"}, inplace=True)

# Backup original dataset
nansha_data_original = nansha_data.copy()

# ------------------------------------------
# ** Step 2: Feature Selection**
# ------------------------------------------
selected_features = [
    'longitude', 'latitude', 'year',
    'GWL', 'rainfall_mm', 'geology',

    # --> density_tier and normalized_density are only valid for Zhongshan 
    # 'density_tier', 'normalized_density',  # (only in Zhongshan)

    # --> Now soil_thickness and building_concentration are only valid for Nansha
    'soil_thickness', 'building_concentration',  # (only for Nansha)

    'normalized_seismic_risk_score', 'subsidence'
]
nansha_data = nansha_data[selected_features].copy()

# Check and fix NaN values
print(f"NaN exists before processing? "
      f"{nansha_data.isna().any().any()}")
nansha_data = nan_ops(nansha_data, ops='sanitize',
                      action='fill', process="do_anyway")
print(f"NaN exists after processing? "
      f"{nansha_data.isna().any().any()}")
 

#  SECTION 2: FEATURE ENGINEERING & NORMALIZATION
# =================================================

# Step 3: Encode Categorical Features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Encode 'geology'
geology_encoded = encoder.fit_transform(nansha_data[['geology']])
geology_columns = [f'geology_{cat}' for cat in encoder.categories_[0]]

# Encode either 'density_tier' (for Zhongshan) or 'building_concentration' (for Nansha)
if 'density_tier' in selected_features:
    density_tier_encoded = encoder.fit_transform(nansha_data[['density_tier']])
    density_tier_columns = [f'density_tier_{cat}' for cat in encoder.categories_[0]]
else:
    density_tier_encoded = encoder.fit_transform(nansha_data[['building_concentration']])
    density_tier_columns = [f'building_concentration_{cat}' for cat in encoder.categories_[0]]

# Convert encoded arrays to DataFrames
geology_df = pd.DataFrame(geology_encoded, columns=geology_columns)
density_df = pd.DataFrame(density_tier_encoded, columns=density_tier_columns)

# Step 4: Normalize Numerical Features (excluding geo-coordinates, year, and already-normalized fields)
scaler = MinMaxScaler()
columns_to_normalize = ['GWL', 'rainfall_mm']  # 'normalized_density' and 'normalized_seismic_risk_score' are already normalized
nansha_data[columns_to_normalize] = scaler.fit_transform(nansha_data[columns_to_normalize])
joblib.dump(scaler, os.path.join(data_path, 'nansha_scaler.joblib'))

print("Columns before dropping categorical features:\n", list(nansha_data.columns))

# Step 5: Merge Encoded Features & Drop Original Categorical Columns
nansha_data = pd.concat([nansha_data, geology_df, density_df], axis=1)

if 'density_tier' in selected_features:
    nansha_data.drop(columns=['geology', 'density_tier'], inplace=True)
else:
    nansha_data.drop(columns=['geology', 'building_concentration'], inplace=True)

print("Columns after encoding and cleanup:\n", list(nansha_data.columns))


# ==========================================
# SECTION 3: DATA SPLITTING & SEQUENCING
# ==========================================

# Step 6: Split data for training and testing
if 'density_tier' in selected_features:
    # Zhongshan: Train on 2015–2022, Test on 2023
    train_data = nansha_data[nansha_data['year'] <= 2022].copy()
    test_data  = nansha_data[nansha_data['year'] == 2023].copy()
else:
    # Nansha: Train on 2015–2021, Test on 2022
    train_data = nansha_data[nansha_data['year'] <= 2021].copy()
    test_data  = nansha_data[nansha_data['year'] == 2022].copy()

# Ensure training data is ordered by time
train_data.sort_values('year', inplace=True)

# Step 7: Define feature groups
static_features = ['longitude', 'latitude'] + list(geology_df.columns) + list(density_df.columns)

dynamic_features = [
    'GWL', 
    'rainfall_mm', 
    'normalized_seismic_risk_score',
    # 'normalized_density'  # Only available for Zhongshan
]

future_features = ['rainfall_mm']

print("Actual subsidence values in test set:")
print(test_data["subsidence"].head())

# Step 8: Sequence generation parameters
forecast_horizon = len(forecast_years)
time_steps = forecast_horizon - 1 if time_steps is None else time_steps
time_steps = min(time_steps, forecast_horizon - 1) if forecast_horizon > 1 else 1

print("Reshaping training data for XTFT input...")

sequence_file = os.path.join(data_path, 'qt.2023_2026.train_data_v3.joblib')

if os.path.isfile(sequence_file):
    # Load preprocessed sequences if available
    X_static, X_dynamic, X_future, y_train_seq = fetch_joblib_data(
        sequence_file,
        'static_data', 'dynamic_data', 'future_data', 'target_data',
        verbose=7,
    )
else:
    # Otherwise, generate sequences from scratch
    X_static, X_dynamic, X_future, y_train_seq = reshape_xtft_data(
        train_data,
        dt_col="year",
        target_col="subsidence",
        static_cols=static_features,
        dynamic_cols=dynamic_features,
        future_cols=future_features,
        time_steps=time_steps,
        forecast_horizons=forecast_horizon,
        spatial_cols=spatial_cols,
        savefile=sequence_file,
        verbose=7
    )
#%
# ==========================================
# SECTION 4: Train-Validation Split & Saving
# ==========================================

# Debug prints
print("Test data preview:")
print(test_data.head())
print("Train data years:", train_data['year'].unique())

# Input shapes
print(f"\nStatic input shape:  {X_static.shape}")
print(f"Dynamic input shape: {X_dynamic.shape}")
print(f"Future input shape:  {X_future.shape}")
print(f"Target sequence shape: {y_train_seq.shape}")

# Step 9: Train-validation split (80-20)
print("\nSplitting training sequences into train and validation sets...\n")

X_static_train, X_static_val, \
X_dynamic_train, X_dynamic_val, \
X_future_train, X_future_val, \
y_train, y_val = train_test_split(
    X_static, X_dynamic, X_future, y_train_seq,
    test_size=0.2,
    random_state=42
)

# Display shapes
print("Training set shapes:")
print(f"  Static:  {X_static_train.shape}")
print(f"  Dynamic: {X_dynamic_train.shape}")
print(f"  Future:  {X_future_train.shape}")
print(f"  Target:  {y_train.shape}\n")

print("Validation set shapes:")
print(f"  Static:  {X_static_val.shape}")
print(f"  Dynamic: {X_dynamic_val.shape}")
print(f"  Future:  {X_future_val.shape}")
print(f"  Target:  {y_val.shape}")

# Save prepared raw data if not already saved
prepared_data_file = os.path.join(data_path, 'qt.prepared_data_2023_2026_v3.joblib')
if os.path.isfile(prepared_data_file):
    test_data = fetch_joblib_data(prepared_data_file, 'test_data')
else:
    data_prepared = {
        "data": nansha_data,  # <-- Corrected from zhongshan_data
        "train_data": train_data,
        "test_data": test_data,
        "selected_features": selected_features,
    }
    joblib.dump(data_prepared, prepared_data_file)
    print(f"\nPrepared data saved at: {prepared_data_file}")

# Save processed sequence data
xtft_data_file = os.path.join(data_path, 'qt.xtft_data_2023_2026_v3.joblib')
if not os.path.isfile(xtft_data_file):
    data_dict = {
        'X_static_train': X_static_train,
        'X_dynamic_train': X_dynamic_train,
        'X_future_train': X_future_train,
        'y_train': y_train,
        'X_static_val': X_static_val,
        'X_dynamic_val': X_dynamic_val,
        'X_future_val': X_future_val,
        'y_val': y_val,
        'X_static': X_static,
        'X_dynamic': X_dynamic,
        'X_future': X_future,
        'y_train_seq': y_train_seq
    }
    joblib.dump(data_dict, xtft_data_file)
    print(f"Processed XTFT data saved at: {xtft_data_file}")


# ==========================================
# SECTION 4: TRAINING XTFT MODEL
# ==========================================

# Best hyperparameters (e.g. from prior tuning)
best_params = {
    'embed_dim'       : 32,
    'max_window_size' : 3,
    'memory_size'     : 100,
    'num_heads'       : 4,
    'dropout_rate'    : 0.1,
    'lstm_units'      : 64,
    'attention_units' : 64,
    'hidden_units'    : 32,
    'multi_scale_agg' : 'auto',
    'anomaly_detection_strategy': anomaly_detection_strategy,
    'use_residuals': True,
    'use_batch_norm':  True,
    'final_agg':'last',
}

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint to save the best model
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(data_path, f'qt.xtft_best_model_2023_2026_v3{super_ext}'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
    save_format='tf'
)

# Instantiate the XTFT model
xtft_model = TRANSF_(
    static_input_dim   = X_static_train.shape[1],
    dynamic_input_dim  = X_dynamic_train.shape[2],
    future_input_dim   = X_future_train.shape[2],
    forecast_horizon   = forecast_horizon,
    quantiles          = quantiles,
    **best_params
)

# Select appropriate loss
if quantiles is not None:
    loss_fn = combined_quantile_loss(quantiles)
else:
    loss_fn = 'mse'

# Compile the model
xtft_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn
)
xtft_model([X_static_train, X_dynamic_train, X_future_train])
xtft_model.summary()

start = time.time()

# Train the model
print("\nTraining the XTFT model...\n")
xtft_model.fit(
    x              = [X_static_train, X_dynamic_train, X_future_train],
    y              = y_train,
    validation_data= ([X_static_val, X_dynamic_val, X_future_val], y_val),
    epochs         = EPOCHS,
    batch_size     = 32,
    callbacks      = [early_stopping, model_checkpoint],
    verbose        = 1
)
print(f"Training time: {(time.time() - start)/60:.2f} minutes")


print("Attempting to fetch trained XTFT model...")

model_paths = [
    os.path.join(data_path, f'qt.xtft_best_model_fixed{super_ext}'),
    os.path.join(data_path, f'qt.xtft_best_model_2023_2026_v3{super_ext}')
]

xtft_model = None

# Try direct loading first
try:
    print(f"Trying direct load from: {model_paths[0]}")
    xtft_model = load_model(model_paths[0])
    print("✅ Model successfully loaded via direct method.")
except Exception as e_direct:
    print(f"❌ Direct load failed: {e_direct}")
    
    # Fallback: Use custom object scope to load model with custom loss
    try:
        print(f"Trying fallback load from: {model_paths[1]} with custom objects...")
        with custom_object_scope({'combined_quantile_loss': combined_quantile_loss}):
            xtft_model = load_model(model_paths[1])
        print("✅ Model successfully loaded using custom_object_scope.")
    except Exception as e_custom:
        print(f"❌ Fallback load also failed: {e_custom}")
        xtft_model = None

# Check if loading was successful
if xtft_model is None:
    print("⚠️ Model loading failed. Please check file paths and custom loss function definition.")
else:
    print("🎉 Model ready for inference or evaluation.")


if xtft_model is None:
    raise RuntimeError("XTFT model is not loaded. Forecasting aborted.")

#%%
# ==========================================
#  SECTION 5: FORECASTING & EVALUATION
# ==========================================
print("\n🚀 Generating Super Quantile Forecast for 2023–2026...\n")


# Check that model is loaded
if xtft_model is None:
    raise RuntimeError("❌ XTFT model is not loaded. Forecasting aborted.")

# Print basic info
print(f"✅ Forecasting years: {forecast_years}")
print(f"🛰️ Forecasting horizon: {forecast_horizon}")
print(f"📊 Forecasting mode: {'quantile' if quantiles else 'point'}")
print(f"🧩 Forecast points per year: {len(X_static)}")
print(f"📈 Total forecast points: {len(forecast_years) * len(X_static)}")

#%%
# Run forecast
forecast_path = os.path.join(data_path, f"qt.forecast_results_2023_2026_v3{super_ext}.csv")
forecast_df = forecast_multi_step(
    xtft_model=xtft_model,
    inputs=[X_static, X_dynamic, X_future],
    forecast_horizon=forecast_horizon,
    y=y_train_seq,
    dt_col="year",
    mode="quantile" if quantiles else "point",
    q=quantiles,
    tname="subsidence",
    forecast_dt=forecast_years,
    apply_mask= True if quantiles else False, 
    mask_values=0,
    mask_fill_value=0,
    savefile=forecast_path,
    spatial_cols= spatial_cols, 
    verbose=7
)

print(f"✅ Forecast saved to: {forecast_path}")
#%%
# Backup forecast file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = os.path.join(data_path, f"backup_forecast_{timestamp}.csv")
forecast_df.to_csv(backup_path, index=False)
print(f"🗃️ Forecast backup saved to: {backup_path}")

# Recover longitude-latitude manually
ll_df = pd.DataFrame(X_static[:, :2], columns=["longitude", "latitude"])
ll_dupl = pd.concat([ll_df] * len(forecast_years), ignore_index=True)
assert len(ll_dupl) == len(forecast_df), "❌ Mismatch in coordinate and forecast length."

# Merge coordinates with forecast
forecast_df_full = pd.concat([ll_dupl, forecast_df], axis=1)

# Optional: visualize coordinate grid (for sanity check)

plt.scatter(forecast_df_full["longitude"], forecast_df_full["latitude"], alpha=0.5)
plt.title("🗺️ Forecast Coordinate Grid")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# Visualize final spatial-temporal forecast
print("🖼️ Visualizing forecast results...\n")
visualize_forecasts(
    forecast_df_full,
    dt_col="year",
    tname="subsidence",
    eval_periods=forecast_years,
    test_data=test_data,
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
print("+" * _TW)
print(
    """
    WARNING: For optimal results, ensure you have sufficient data and 
    set `spatial_cols` to spatial coordinates (e.g., ('longitude', 'latitude')). 
    This script is intended for code-ocean test reproducibility only.

    You can toggle the anomaly detection strategy via the config. 
    Available options include 'from_config', 'feature_based', 'prediction_based', 
    or None (default). Please retrain the model after making any adjustments.
    
    For more information on leveraging the anomaly detection features integrated 
    within the XTFT model, visit:
    https://fusion-lab.readthedocs.io/en/latest/user_guide/examples/xtft_with_anomaly_detection.html
    
    For details on computing anomalies when 'from_config' is selected, see:
    https://fusion-lab.readthedocs.io/en/latest/user_guide/anomaly_detection.html#using-anomaly-detection-with-xtft
    """
)
print("+" * _TW)
