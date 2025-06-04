import os
import numpy as np
import pandas as pd
import tensorflow as tf
# import keras_tuner as kt # PIHALTuner will handle kt internally
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# For splitting, a time-series aware or group-aware split is often best.
# Using train_test_split for simplicity here, assuming groups are handled.
from sklearn.model_selection import train_test_split
import joblib
import logging # Using standard logging, can be replaced by fusionlog
from typing import Dict, List, Tuple, Optional, Any

# --- FUSIONLAB Imports (IMPORTANT: Adjust these paths to your project structure) ---
try:
    from fusionlab.nn.pinn.tuning import PIHALTuner
    from fusionlab.nn.pinn.models import PIHALNet # Required by PIHALTuner
    from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences
    from fusionlab.nn.losses import combined_quantile_loss # If used and custom
    # from fusionlab.utils import fusionlog # If using your custom logger
    logger = logging.getLogger(__name__) # Default logger
    # logger = fusionlog().get_fusionlog_logger(__name__) # If using fusionlog
except ImportError as e:
    print(f"Error importing fusionlab modules: {e}. Ensure fusionlab is in PYTHONPATH.")
    print("Using placeholder functions/classes for critical missing parts.")
    # --- Fallback Placeholders (REMOVE THESE IF FUSIONLAB IS CORRECTLY INSTALLED/IMPORTED) ---

# --- Basic Logger Setup (if not using fusionlog's setup) ---
if not hasattr(logger, 'handlers') or not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
#%
# --- Configuration Constants ---
# Paths
DATA_FILE_PATH = "data/zhongshan_500k.csv" # IMPORTANT: Update this
RUN_OUTPUT_PATH = "zhongshan_tuning_pinn"
CITY_NAME = "zhongshan"
MODEL_NAME_TUNED = "PIHALNet_Zhongshan_Tuned"

# Data Parameters (specific to Zhongshan dataset)
TIME_COL = "year"             # Original time column name in your CSV
DT_COL_NAME_TEMP = "datetime_temp" # Temp column for datetime objects
LON_COL = "longitude"
LAT_COL = "latitude"
SUBSIDENCE_COL = "subsidence"
GWL_COL = "GWL"
CATEGORICAL_COLS = ['geology'] # Example
NUMERICAL_COLS_FOR_MAIN_SCALING = [ # Features to scale before sequence prep
    'rainfall_mm', 
    # Add other numerical features specific to Zhongshan data that need scaling
    # e.g., 'pumping_rate', 'temperature', etc.
    # DO NOT include LON_COL, LAT_COL, TIME_COL_NUMERIC_PINN, SUBSIDENCE_COL, GWL_COL here
]

# Sequence Parameters
TIME_STEPS = 2        # Lookback window (e.g., 2 years of monthly data)
FORECAST_HORIZON = 3 # Prediction horizon (e.g., 3 years)
# Output dimensions for PIHALNet (usually 1 for each target value)
OUT_S_DIM = 1
OUT_G_DIM = 1

BATCH_SIZE =256#32 

# Splitting Data
# Example: Use data up to year X for training/validation, rest for a final hold-out test
# For tuning, we split the "training/validation" part further.
TRAIN_VAL_END_YEAR = 2020 # Example: data up to this year for tuning
VAL_SPLIT_RATIO = 0.2     # 20% of train_val_data for tuner's validation

# Tuning Parameters
TUNER_TYPE = 'hyperband' # 'randomsearch', 'bayesian', 'hyperband'
MAX_TRIALS = 30          # Total number of hyperparameter combinations to test
EXECUTIONS_PER_TRIAL = 1 # How many times to train each model configuration
EPOCHS_PER_TRIAL = 50    # Max epochs for each trial in the tuner
TUNER_OBJECTIVE = "val_loss" # Metric to optimize
TUNER_SEED = 42

# PIHALNet Fixed Parameters (some will be inferred, others set here)
# These are parameters NOT tuned by the `param_space`
# Refer to PIHALTuner.DEFAULT_PIHALNET_FIXED_PARAMS for a base
BASE_FIXED_PARAMS = {
    "quantiles": [0.1, 0.5, 0.9], # Or None for point estimates
    "max_window_size": TIME_STEPS, # Should match sequence parameter
    # "memory_size": 50, # Can be fixed or tuned
    # "scales": [1, 2],  # Can be fixed or tuned
    "multi_scale_agg": 'last',
    "final_agg": 'last',
    "use_residuals": True,
    "use_batch_norm": False, # Often False if other normalizations are used
    "use_vsn": True,
    # "vsn_units": 32, # Can be fixed or tuned
    "pde_mode": "consolidation", # Or 'gw_flow', 'coupled', 'none'
    "pinn_coefficient_C": "learnable", # Or a fixed float
    "gw_flow_coeffs": None, # Or a dictionary of coefficients
    "loss_weights": { # Weights for different parts of the loss in PIHALNet
        'subs_pred': 1.0, # Weight for subsidence data loss
        'gwl_pred': 0.8,  # Weight for GWL data loss
        # 'physics_loss': 0.1 # If PIHALNet's compile/train_step uses this key
    }
}

#%
# --- Data Loading and Preprocessing Function ---
def load_and_preprocess_zhongshan_data(file_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the Zhongshan dataset.
    ADAPT THIS FUNCTION TO YOUR ACTUAL DATA AND PREPROCESSING NEEDS.
    """
    logger.info(f"Loading Zhongshan data from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(
            f"Data file NOT FOUND: {file_path}. "
            "Please update DATA_FILE_PATH. Exiting."
        )
        # For demonstration, generating dummy data if file not found.
        # In a real scenario, you'd raise an error or handle this.
        n_samples = 500_000
        years = np.random.randint(2000, TRAIN_VAL_END_YEAR + 4, n_samples)
        df = pd.DataFrame({
            TIME_COL: years,
            LON_COL: np.random.uniform(113.0, 113.8, n_samples), # Zhongshan approx
            LAT_COL: np.random.uniform(22.3, 22.8, n_samples),  # Zhongshan approx
            SUBSIDENCE_COL: np.random.normal(loc=-20, scale=15, size=n_samples),
            GWL_COL: np.random.normal(loc=2.5, scale=1, size=n_samples),
            'rainfall_mm': np.random.uniform(500, 2500, n_samples),
            CATEGORICAL_COLS[0]: np.random.choice(
                ['Silty Clay', 'Fine Sand', 'Gravel', 'Bedrock'], size=n_samples
            )
            # Add other relevant Zhongshan features
        })
        logger.warning(f"Generated dummy Zhongshan data with {n_samples} samples.")
    else:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data. Shape: {df.shape}")

    # --- Round Spatial Coordinates to a Fixed Precision ---
    # Define the precision (number of decimal places)
    # This value needs careful consideration based on your data's spatial resolution
    # and the scale of the phenomenon you're modeling.
    # For example, 4-5 decimal places for lon/lat is often a good starting point.
    # 1 degree = ~111 km
    # 0.1 degree = ~11.1 km
    # 0.01 degree = ~1.11 km
    # 0.001 degree = ~111 m
    # 0.0001 degree = ~11.1 m
    # 0.00001 degree = ~1.11 m
    coordinate_precision = 5 # Example: 5 decimal places

    if LON_COL in df.columns and LAT_COL in df.columns:
        logger.info(
            f"Rounding '{LON_COL}' and '{LAT_COL}' to "
            f"{coordinate_precision} decimal places for grouping."
        )
        df[LON_COL] = df[LON_COL].round(coordinate_precision)
        df[LAT_COL] = df[LAT_COL].round(coordinate_precision)
    else:
        logger.warning(
            f"Longitude ('{LON_COL}') or Latitude ('{LAT_COL}') columns "
            "not found for rounding."
        )

    # Now df_processed will use these rounded coordinates when it's created
    # from df, and subsequent grouping will be based on these rounded values.
    df_processed = df # Continue with df_processed = df.copy() if you prefer
    
    # Basic Cleaning (examples, adapt as needed)
    required_cols_check = [
        TIME_COL, LON_COL, LAT_COL, SUBSIDENCE_COL, GWL_COL
    ] + CATEGORICAL_COLS + [feat for feat in NUMERICAL_COLS_FOR_MAIN_SCALING if feat in df.columns]
    
    missing_cols = [col for col in required_cols_check if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in the dataset: {missing_cols}")
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    df = df.dropna(subset=[
        TIME_COL, LON_COL, LAT_COL, SUBSIDENCE_COL, GWL_COL
    ]).copy() # Use .copy() to avoid SettingWithCopyWarning

    # Convert TIME_COL to datetime objects
    # Assuming TIME_COL contains year integers or parsable date strings
    try:
        if pd.api.types.is_numeric_dtype(df[TIME_COL]):
             # If TIME_COL is like 2015, 2016 (numeric years)
            df.loc[:, DT_COL_NAME_TEMP] = pd.to_datetime(
                df[TIME_COL], format='%Y', errors='coerce'
            )
        else:
            # If TIME_COL is like '2015-01-01'
            df.loc[:, DT_COL_NAME_TEMP] = pd.to_datetime(
                df[TIME_COL], errors='coerce'
            )
    except Exception as e:
        logger.error(f"Error converting '{TIME_COL}' to datetime: {e}")
        raise
        
    df = df.dropna(subset=[DT_COL_NAME_TEMP])
    logger.info(f"Data shape after initial cleaning: {df.shape}")

    # Create RUN_OUTPUT_PATH
    os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)

    # --- Encoding Categorical Features ---
    df_processed = df # Work on this copy
    categorical_cols_to_encode = [
        c for c in CATEGORICAL_COLS if c in df_processed.columns
    ]
    global encoded_feature_names_list # Used later for feature set definition
    encoded_feature_names_list = []

    if categorical_cols_to_encode:
        encoder_ohe = OneHotEncoder(
            sparse_output=False, handle_unknown='ignore', dtype=np.float32
        )
        # Fit on unique values to handle potential large data more efficiently
        # For very large datasets, consider feature hashing or other techniques
        encoded_data = encoder_ohe.fit_transform(
            df_processed[categorical_cols_to_encode]
        )
        encoder_path = os.path.join(
            RUN_OUTPUT_PATH, f"{CITY_NAME}_ohe_encoder_tuning.joblib"
        )
        joblib.dump(encoder_ohe, encoder_path)
        logger.info(f"OneHotEncoder saved to: {encoder_path}")

        new_ohe_cols = encoder_ohe.get_feature_names_out(
            categorical_cols_to_encode
        )
        encoded_feature_names_list.extend(new_ohe_cols)
        encoded_df_part = pd.DataFrame(
            encoded_data, columns=new_ohe_cols, index=df_processed.index
        )
        # Drop original categorical columns and concatenate encoded ones
        df_processed = df_processed.drop(columns=categorical_cols_to_encode)
        df_processed = pd.concat([df_processed, encoded_df_part], axis=1)
        logger.info(f"Encoded features created: {new_ohe_cols.tolist()}")

    # --- Numerical Time Coordinate for PINN ---
    global TIME_COL_NUMERIC_PINN # Used by prepare_pinn_data_sequences
    TIME_COL_NUMERIC_PINN = f"{TIME_COL}_numeric_coord_tuning"
    df_processed[TIME_COL_NUMERIC_PINN] = (
        df_processed[DT_COL_NAME_TEMP].dt.year +
        (df_processed[DT_COL_NAME_TEMP].dt.dayofyear - 1) /
        (365 + df_processed[DT_COL_NAME_TEMP].dt.is_leap_year.astype(int))
    )

    # --- Scaling Numerical Features (for model inputs, not coords/targets) ---
    numerical_cols_to_scale_final = [
        c for c in NUMERICAL_COLS_FOR_MAIN_SCALING if c in df_processed.columns
    ]
    if numerical_cols_to_scale_final:
        scaler_main = MinMaxScaler()
        df_processed[numerical_cols_to_scale_final] = scaler_main.fit_transform(
            df_processed[numerical_cols_to_scale_final]
        )
        scaler_path = os.path.join(
            RUN_OUTPUT_PATH, f"{CITY_NAME}_main_scaler_tuning.joblib"
        )
        joblib.dump(scaler_main, scaler_path)
        logger.info(
            f"Scaled numerical features: {numerical_cols_to_scale_final}. "
            f"Main scaler saved to: {scaler_path}"
        )
    else:
        logger.info("No numerical features specified for main scaling.")
    
    return df_processed

# --- Main Tuning Script ---
if __name__ == "__main__":
    logger.info(f"--- Starting PIHALNet Tuning for {CITY_NAME} ---")

    # 1. Load and Preprocess Zhongshan Data
    df_master_processed = load_and_preprocess_zhongshan_data(DATA_FILE_PATH)

    # 2. Split data for tuning (train_val)
    logger.info(f"Splitting data for tuning (up to year {TRAIN_VAL_END_YEAR})...")
    df_train_val_master = df_master_processed[
        df_master_processed[DT_COL_NAME_TEMP].dt.year <= TRAIN_VAL_END_YEAR
    ].copy()

    if df_train_val_master.empty:
        raise ValueError(
            f"Train/Validation data is empty after split at year "
            f"{TRAIN_VAL_END_YEAR}. Check data and TRAIN_VAL_END_YEAR."
        )
    logger.info(f"Total data for tuning (train+val): {df_train_val_master.shape}")

    # Further split train_val_master into training and validation sets for the tuner.
    # A robust split for time-series with spatial groups would:
    #  1. Identify unique spatial groups (lon, lat).
    #  2. Split these groups into train and val sets.
    #  3. Construct df_tune_train and df_tune_val based on these group assignments.
    unique_locations = df_train_val_master[[LON_COL, LAT_COL]].drop_duplicates()
    
    if len(unique_locations) < 2 : # Need at least two groups to split
        logger.warning(
            "Less than 2 unique locations found for splitting. "
            "Using a simple time-based split if possible, or random split. "
            "This might lead to data leakage if not careful."
        )
        # Fallback to a simple random split if too few locations
        df_tune_train, df_tune_val = train_test_split(
            df_train_val_master, test_size=VAL_SPLIT_RATIO, random_state=TUNER_SEED,
            shuffle=True # Shuffle before splitting if not time-ordered
        )
    else:
        train_loc_groups, val_loc_groups = train_test_split(
            unique_locations, test_size=VAL_SPLIT_RATIO, random_state=TUNER_SEED
        )
        df_tune_train = df_train_val_master.merge(
            train_loc_groups, on=[LON_COL, LAT_COL], how='inner'
        )
        df_tune_val = df_train_val_master.merge(
            val_loc_groups, on=[LON_COL, LAT_COL], how='inner'
        )

    if df_tune_train.empty or df_tune_val.empty:
        logger.error(
            "Tuner training or validation DataFrame is empty after splitting. "
            "Check data distribution, number of unique locations, and split ratio."
        )
        exit()
        
    logger.info(f"Tuner training data portion shape: {df_tune_train.shape}")
    logger.info(f"Tuner validation data portion shape: {df_tune_val.shape}")

    # 3. Prepare data sequences for training and validation
    logger.info("Preparing data sequences for tuner training...")
    inputs_train_dict, targets_train_dict, coord_scaler_train = \
        prepare_pinn_data_sequences(
            df=df_tune_train,
            time_col=TIME_COL_NUMERIC_PINN,
            lon_col=LON_COL,
            lat_col=LAT_COL,
            subsidence_col=SUBSIDENCE_COL, 
            gwl_col=GWL_COL,
            dynamic_cols=[ # Define explicitly based on available scaled columns
                c for c in [GWL_COL, 'rainfall_mm', 'normalized_density', 'normalized_seismic_risk_score'] 
                if c in df_tune_train.columns
            ],
            static_cols=list(encoded_feature_names_list), # From global
            future_cols=[c for c in ['rainfall_mm'] if c in df_tune_train.columns],
            group_id_cols=[LON_COL, LAT_COL],
            time_steps=TIME_STEPS, 
            forecast_horizon=FORECAST_HORIZON,
            output_subsidence_dim=OUT_S_DIM,
            output_gwl_dim=OUT_G_DIM,
            normalize_coords=True, # Let prepare_pinn_data_sequences handle this
            return_coord_scaler=True,
            cols_to_scale='auto', # scale other numerical features  
            verbose=1
        )

    if inputs_train_dict['coords'].shape[0] == 0:
        raise ValueError("Sequence generation produced no training samples for the tuner.")

    logger.info("Preparing data sequences for tuner validation...")
    inputs_val_dict, targets_val_dict = \
        prepare_pinn_data_sequences(
            df=df_tune_val, # Use the same coord_scaler from training if needed for consistency
            time_col=TIME_COL_NUMERIC_PINN,
            lon_col=LON_COL, lat_col=LAT_COL,
            subsidence_col=SUBSIDENCE_COL, gwl_col=GWL_COL,
            dynamic_cols=[
                c for c in [GWL_COL, 'rainfall_mm', 'normalized_density', 'normalized_seismic_risk_score']
                if c in df_tune_val.columns
            ],
            static_cols=list(encoded_feature_names_list),
            future_cols=[c for c in ['rainfall_mm'] if c in df_tune_val.columns],
            group_id_cols=[LON_COL, LAT_COL],
            time_steps=TIME_STEPS, 
            forecast_horizon=FORECAST_HORIZON,
            output_subsidence_dim=OUT_S_DIM, 
            output_gwl_dim=OUT_G_DIM,
            normalize_coords=True, # Apply same normalization strategy
            return_coord_scaler=False, # Not strictly needed from validation
            verbose=1
        )
    
    if inputs_val_dict['coords'].shape[0] == 0:
        raise ValueError("Sequence generation produced no validation samples for the tuner.")


#%
    # 4. Define Fixed Model Parameters and Hyperparameter Space for PIHALTuner
    # Infer dimensions from the training data sequences
    # These are passed to PIHALTuner.create or PIHALTuner constructor
    
    # Start with a copy of base fixed parameters
    current_fixed_params = BASE_FIXED_PARAMS.copy() 
    
    # Update with inferred dimensions and specific settings for this run
    current_fixed_params.update({
        "static_input_dim": inputs_train_dict.get('static_features', np.zeros((0,0))).shape[-1],
        "dynamic_input_dim": inputs_train_dict['dynamic_features'].shape[-1],
        "future_input_dim": inputs_train_dict.get('future_features', np.zeros((0,0,0))).shape[-1],
        "output_subsidence_dim": OUT_S_DIM,
        "output_gwl_dim": OUT_G_DIM,
        "forecast_horizon": FORECAST_HORIZON,
        "quantiles": BASE_FIXED_PARAMS.get("quantiles"), # Use from BASE_FIXED_PARAMS
        "max_window_size": 5, #TIME_STEPS,
    })
#%
    # Define the hyperparameter search space for PIHALTuner
    # PIHALTuner's _get_hp_* methods will use these if provided,
    # otherwise they use their own defaults.
    param_space_config = {
        'embed_dim': {'min_value': 32, 'max_value': 128, 'step': 32},
        'hidden_units': {'min_value': 64, 'max_value': 256, 'step': 64},
        'lstm_units': {'min_value': 64, 'max_value': 256, 'step': 64},
        'attention_units': {'min_value': 32, 'max_value': 128, 'step': 32},
        'num_heads': [2, 4], # Choices
        'dropout_rate': {'min_value': 0.05, 'max_value': 0.3, 'step': 0.05},
        'vsn_units': {'min_value': 16, 'max_value': 64, 'step': 16},
        'activation': ['relu', 'gelu'], # Choices
        'learning_rate': [1e-4, 5e-4, 1e-3], # Choices
        #'lambda_pde': {'min_value': 1, 'max_value': 10, 'step': 0.05, 'sampling': 'log'}, # Tune PDE weight
        'pinn_coefficient_C_type': ['learnable',],
        # 'pinn_coefficient_C_value': {'min_value': 1, 'max_value': 5, 'sampling': 'log'},
        'pde_mode': ['consolidation', 'none'], # Example: tune if PDE is used
    }
#%
    # 5. Instantiate PIHALTuner
    logger.info("Instantiating PIHALTuner...")
    tuner = PIHALTuner(
        fixed_model_params=current_fixed_params,
        param_space=param_space_config,
        objective=TUNER_OBJECTIVE,
        max_trials=MAX_TRIALS,
        project_name=f"{CITY_NAME}_{MODEL_NAME_TUNED}",
        directory=os.path.join(RUN_OUTPUT_PATH, "tuner_dir_zhongshan"),
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        tuner_type=TUNER_TYPE, # e.g., 'hyperband', 'randomsearch'
        seed=TUNER_SEED,
        overwrite_tuner=True # Set to False to resume
    )

    # 6. Run Hyperparameter Search
    logger.info("Creating TensorFlow datasets for Keras Tuner...")

    # Create tf.data.Dataset for training
    if not inputs_train_dict or not targets_train_dict:
        raise ValueError("inputs_train_dict or targets_train_dict is None"
                         " or empty before dataset creation.")

    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (inputs_train_dict, targets_train_dict)
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    logger.info(f"Training tf.data.Dataset created. Cardinality:"
                f" {train_tf_dataset.cardinality().numpy()}")


    # Create tf.data.Dataset for validation
    val_tf_dataset = None
    if inputs_val_dict and targets_val_dict: # Ensure validation data exists
        val_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs_val_dict, targets_val_dict)
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        logger.info(f"Validation tf.data.Dataset created. Cardinality:"
                    f" {val_tf_dataset.cardinality().numpy()}")
    else:
        logger.warning("Validation data (inputs_val_dict or targets_val_dict)"
                       " is not available. Validation will be skipped by"
                       " the tuner if val_tf_dataset is None.")


    # 6. Run Hyperparameter Search
    logger.info(f"Starting hyperparameter search with {TUNER_TYPE} tuner...")
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=TUNER_OBJECTIVE,
        patience=8, 
        restore_best_weights=True,
        verbose=1
    )
    print(tuner.fixed_model_params)
    # % 
    try: 
        print("Leverage keras HyperModel search... ")
        # Call tuner.search with tf.data.Dataset objects
        tuner.search(
            train_data=train_tf_dataset,       # Pass the tf.data.Dataset
            validation_data=val_tf_dataset,  # Pass the tf.data.Dataset (or None)
            epochs=EPOCHS_PER_TRIAL,
            callbacks=[early_stopping_cb],
            # batch_size is now part of the dataset, so Keras/Tuner will use that.
            # Passing it here might be redundant or ignored.
            verbose=1, 
        )
    except: 
        print("Implement Run for robust data parsing ...")
        # Call tuner.search with tf.data.Dataset objects
        tuner.run(
            inputs = inputs_train_dict,        # Pass the tf.data.Dataset
            y= targets_train_dict, 
            validation_data=(inputs_val_dict, targets_val_dict),  # Pass the tf.data.Dataset (or None)
            epochs=EPOCHS_PER_TRIAL,
            callbacks=[early_stopping_cb],
            batch_size =BATCH_SIZE, 
            # batch_size is now part of the dataset, so Keras/Tuner will use that.
            # Passing it here might be redundant or ignored.
            verbose=1, 
        )
    
    logger.info("Hyperparameter search completed.")
    

    # logger.info(f"Starting hyperparameter search with {TUNER_TYPE} tuner...")
    # # Callbacks for the search
    # early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    #     monitor=TUNER_OBJECTIVE,
    #     patience=8, # Patience for early stopping within a trial
    #     restore_best_weights=True, # Restore best weights at the end of trial
    #     verbose=1
    # )
    
    # tuner.search(
    #     train_data=(inputs_train_dict, targets_train_dict), 
    #     validation_data=(inputs_val_dict, targets_val_dict),
    #     epochs=EPOCHS_PER_TRIAL,
    #     callbacks=[early_stopping_cb],
    #     batch_size=BATCH_SIZE, 
    #     verbose=1, 
    
    # )
    # tuner.search( # PIHALTuner's fit method
    #     inputs=( inputs_train_dict, input_target,
    #     # y=targets_train_dict,
    #     validation_data=(inputs_val_dict, targets_val_dict),
    #     epochs=EPOCHS_PER_TRIAL, # Max epochs for each trial
    #     callbacks=[early_stopping_cb],
    #     batch_size=BATCH_SIZE, # PIHALTuner's fit handles dataset creation
    #     verbose=1 # Verbosity of search progress
    # )
    logger.info("Hyperparameter search completed.")

    # 7. Retrieve and Display Results
    try:
        best_hps_list = tuner.get_best_hyperparameters(num_trials=1)
        if not best_hps_list:
            logger.error("Tuner could not find any best hyperparameters.")
            exit()
        best_hps = best_hps_list[0] # KerasTuner returns a list

        logger.info("\n--- Best Hyperparameters Found ---")
        for param_name, value in best_hps.values.items():
            logger.info(f"{param_name}: {value}")

        # Save best HPs to a file
        best_hps_path = os.path.join(
            RUN_OUTPUT_PATH, f"{MODEL_NAME_TUNED}_best_hyperparameters.txt"
        )
        with open(best_hps_path, 'w') as f:
            for param_name, value in best_hps.values.items():
                f.write(f"{param_name}: {value}\n")
        logger.info(f"Best hyperparameters saved to: {best_hps_path}")

        # Get the best model(s)
        logger.info("Retrieving the best model(s)...")
        best_models = tuner.get_best_models(num_models=1)
        if not best_models or best_models[0] is None:
            logger.error("Tuner could not retrieve the best model.")
            exit()
        best_model = best_models[0]
        
        logger.info("\n--- Best Model Summary ---")
        best_model.summary(line_length=110)

        # Save the best model
        best_model_path = os.path.join(
            RUN_OUTPUT_PATH, f"{MODEL_NAME_TUNED}_best_model.keras"
        )
        best_model.save(best_model_path)
        logger.info(f"Best model saved to: {best_model_path}")

    except Exception as ex_results:
        logger.error(f"Error retrieving or saving tuning results: {ex_results}")

    logger.info(f"--- PIHALNet Tuning for {CITY_NAME} Finished ---")
    logger.info(
        f"Results and artifacts are in: {os.path.join(RUN_OUTPUT_PATH, 'tuner_dir_zhongshan')}")

