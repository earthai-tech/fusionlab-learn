# pihalnet_tuner.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause


"""
This script runs a hyperparameter tuning session for PIHALNet (a
Physics-Informed HalNet) on Zhongshan data. It handles:
  1. Data loading and preprocessing
  2. Sequence generation for training and validation
  3. Defining the search space and fixed parameters
  4. Instantiating and running PIHALTuner
  5. Saving and summarizing results

Usage:
    python pihalnet_tuner.py

All long lines have been wrapped at ~70 characters for readability.
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging

try:
    import tensorflow as tf
    from fusionlab.nn.pinn.tuning import PIHALTuner
    from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences
    # from fusionlab.nn.losses import combined_quantile_loss  # optional
    from fusionlab.utils.generic_utils import rename_dict_keys
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"Error importing fusionlab modules: {e}. "
          "Ensure fusionlab is in PYTHONPATH.")
    print("Using placeholder functions/classes for critical missing parts.")
    # If needed, you can define dummy placeholders here.

# --- Basic Logger Setup (if not using fusionlog) ---
if not hasattr(logger, 'handlers') or not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - '
               '%(message)s'
    )


# --- Configuration Constants ---
# In pihalnet_tuner.py, replace the hard‐coded DATA_FILE_PATH with:
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
# Paths and filenames
DATA_FILE_PATH = os.path.join(
    SCRIPT_DIR,          # e.g. fusionlab/tools
    os.pardir,           # go up to fusionlab/
    os.pardir,           # go up to fusionlab-learn/
    "data",              # into the data/ folder
    "zhongshan_500k.csv" # the actual file
)
# DATA_FILE_PATH = "../../data/zhongshan_500k.csv"
RUN_OUTPUT_PATH = "zhongshan_tuning_pinn"
CITY_NAME = "zhongshan"
MODEL_NAME_TUNED = "PIHALNet_Zhongshan_Tuned"

# Data column names
TIME_COL = "year"                  # Original time column in CSV
DT_COL_NAME_TEMP = "datetime_temp" # Temp column holding datetime objects
LON_COL = "longitude"
LAT_COL = "latitude"
SUBSIDENCE_COL = "subsidence"
GWL_COL = "GWL"
CATEGORICAL_COLS = ['geology']     # Example categorical column
NUMERICAL_COLS_FOR_MAIN_SCALING = [
    'rainfall_mm',
    # Add any other numerical features you want to scale
    # Do NOT include lon/lat, time, subsidence, or GWL here
]

# Sequence parameters (for PINN data sequences)
TIME_STEPS = 4           # Lookback window (e.g., 2 years)
FORECAST_HORIZON = 4     # Prediction horizon (e.g., 3 years)
OUT_S_DIM = 1            # Output dimension for subsidence
OUT_G_DIM = 1            # Output dimension for groundwater level
BATCH_SIZE = 256

# Splitting parameters
TRAIN_VAL_END_YEAR = 2020  # Year cutoff for tuning data
VAL_SPLIT_RATIO = 0.2      # 20% for validation inside tuner

# Tuner parameters
TUNER_TYPE = 'hyperband'     # 'randomsearch', 'bayesian', or 'hyperband'
MAX_TRIALS = 30              # Number of HP combinations to try
EXECUTIONS_PER_TRIAL = 1      # Repeats per combination
EPOCHS_PER_TRIAL = 50         # Max epochs per trial
TUNER_OBJECTIVE = "val_loss"  # Metric to optimize
TUNER_SEED = 42              # Random seed for reproducibility

# PIHALNet fixed parameters (not tuned by the tuner)
BASE_FIXED_PARAMS = {
    "quantiles": [0.1, 0.5, 0.9],  # Or None for point estimates
    "max_window_size": TIME_STEPS,
    "multi_scale_agg": 'last',
    "final_agg": 'last',
    "use_residuals": True,
    "use_batch_norm": False,
    "use_vsn": True,
    # "vsn_units": 32,  # Can be fixed or tuned
    "pde_mode": "consolidation",  # 'gw_flow', 'coupled', 'none'
    "pinn_coefficient_C": "learnable",  # Or a fixed float
    "gw_flow_coeffs": None,  # Or a dict of coefficients
    "loss_weights": {
        'subs_pred': 1.0,
        'gwl_pred': 0.8,
        # 'physics_loss': 0.1  # Example if used
    }
}


# --- Data Loading and Preprocessing Function ---
def load_and_preprocess_zhongshan_data(file_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses Zhongshan data. Adapts to missing files by
    generating a dummy DataFrame. Rounds lon/lat, converts time to
    datetime, encodes categoricals, and scales main numerical features.
    """
    logger.info(f"Loading Zhongshan data from: {file_path}")
    # Test the script with generated data if real data is missing.
    
    if not os.path.exists(file_path): 
        logger.error(
            f"Data file NOT FOUND: {file_path}. Please update "
            "DATA_FILE_PATH. Exiting."
        )
        # For demonstration, generate dummy data
        n_samples = 500_000
        years = np.random.randint(
            2000, TRAIN_VAL_END_YEAR + 4, n_samples
        )
        df = pd.DataFrame({
            TIME_COL: years,
            LON_COL: np.random.uniform(113.0, 113.8, n_samples),
            LAT_COL: np.random.uniform(22.3, 22.8, n_samples),
            SUBSIDENCE_COL: np.random.normal(
                loc=-20, scale=15, size=n_samples
            ),
            GWL_COL: np.random.normal(loc=2.5, scale=1, size=n_samples),
            'rainfall_mm': np.random.uniform(500, 2500, n_samples),
            CATEGORICAL_COLS[0]: np.random.choice(
                ['Silty Clay', 'Fine Sand', 'Gravel', 'Bedrock'],
                size=n_samples
            )
            # Other relevant features can added here
        })
        logger.warning(f"Generated dummy data with {n_samples} samples.")
    else:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data. Shape: {df.shape}")

    # --- Round lon/lat for grouping ---
    coordinate_precision = 5
    if LON_COL in df.columns and LAT_COL in df.columns:
        logger.info(
            f"Rounding '{LON_COL}' and '{LAT_COL}' to "
            f"{coordinate_precision} decimal places."
        )
        df[LON_COL] = df[LON_COL].round(coordinate_precision)
        df[LAT_COL] = df[LAT_COL].round(coordinate_precision)
    else:
        logger.warning(
            f"Columns '{LON_COL}' or '{LAT_COL}' not found for rounding."
        )

    df_processed = df  # You can do df.copy() if you prefer

    # --- Check for required columns ---
    required_cols_check = (
        [TIME_COL, LON_COL, LAT_COL, SUBSIDENCE_COL, GWL_COL]
        + CATEGORICAL_COLS
        + [
            feat for feat in NUMERICAL_COLS_FOR_MAIN_SCALING
            if feat in df.columns
        ]
    )
    missing_cols = [col for col in required_cols_check
                    if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Dataset missing: {missing_cols}")

    # Drop rows with NA in critical columns
    df = df.dropna(
        subset=[TIME_COL, LON_COL, LAT_COL, SUBSIDENCE_COL, GWL_COL]
    ).copy()

    # --- Convert TIME_COL to datetime ---
    try:
        if pd.api.types.is_numeric_dtype(df[TIME_COL]):
            # Example: 2015, 2016 (numeric years)
            df.loc[:, DT_COL_NAME_TEMP] = pd.to_datetime(
                df[TIME_COL], format='%Y', errors='coerce'
            )
        else:
            # Example: '2015-01-01' (string date)
            df.loc[:, DT_COL_NAME_TEMP] = pd.to_datetime(
                df[TIME_COL], errors='coerce'
            )
    except Exception as e:
        logger.error(f"Error converting '{TIME_COL}' to datetime: {e}")
        raise

    df = df.dropna(subset=[DT_COL_NAME_TEMP])
    logger.info(f"Shape after cleaning: {df.shape}")

    # Ensure the run output directory exists
    os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)

    # --- One-Hot Encode Categorical Features ---
    df_processed = df
    categorical_cols_to_encode = [
        c for c in CATEGORICAL_COLS if c in df_processed.columns
    ]
    global encoded_feature_names_list
    encoded_feature_names_list = []

    if categorical_cols_to_encode:
        encoder_ohe = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            dtype=np.float32
        )
        encoded_data = encoder_ohe.fit_transform(
            df_processed[categorical_cols_to_encode]
        )
        encoder_path = os.path.join(
            RUN_OUTPUT_PATH,
            f"{CITY_NAME}_ohe_encoder_tuning.joblib"
        )
        joblib.dump(encoder_ohe, encoder_path)
        logger.info(f"OHE saved to: {encoder_path}")

        new_ohe_cols = encoder_ohe.get_feature_names_out(
            categorical_cols_to_encode
        )
        encoded_feature_names_list.extend(new_ohe_cols)
        encoded_df_part = pd.DataFrame(
            encoded_data,
            columns=new_ohe_cols,
            index=df_processed.index
        )
        df_processed = df_processed.drop(
            columns=categorical_cols_to_encode
        )
        df_processed = pd.concat(
            [df_processed, encoded_df_part], axis=1
        )
        logger.info(
            f"Encoded features created: {new_ohe_cols.tolist()}"
        )

    # --- Compute numerical time coordinate for PINN ---
    global TIME_COL_NUMERIC_PINN
    TIME_COL_NUMERIC_PINN = f"{TIME_COL}_numeric_coord_tuning"
    df_processed[TIME_COL_NUMERIC_PINN] = (
        df_processed[DT_COL_NAME_TEMP].dt.year +
        (df_processed[DT_COL_NAME_TEMP].dt.dayofyear - 1) /
        (365 + df_processed[DT_COL_NAME_TEMP].dt.is_leap_year
         .astype(int))
    )

    # --- Scale main numerical features (excluding coords/targets) ---
    numerical_cols_to_scale_final = [
        c for c in NUMERICAL_COLS_FOR_MAIN_SCALING
        if c in df_processed.columns
    ]
    if numerical_cols_to_scale_final:
        scaler_main = MinMaxScaler()
        df_processed[numerical_cols_to_scale_final] = scaler_main.fit_transform(
            df_processed[numerical_cols_to_scale_final]
        )
        scaler_path = os.path.join(
            RUN_OUTPUT_PATH,
            f"{CITY_NAME}_main_scaler_tuning.joblib"
        )
        joblib.dump(scaler_main, scaler_path)
        logger.info(
            f"Scaled features: {numerical_cols_to_scale_final}. "
            f"Scaler saved to: {scaler_path}"
        )
    else:
        logger.info("No numerical features to scale.")

    return df_processed


# --- Main Tuning Script ---
if __name__ == "__main__":
    logger.info(f"--- Starting PIHALNet Tuning for {CITY_NAME} ---")

    # 1. Load & preprocess Zhongshan data
    df_master_processed = load_and_preprocess_zhongshan_data(
        DATA_FILE_PATH
    )
    # 2. Split data for tuning (year ≤ TRAIN_VAL_END_YEAR)
    logger.info(
        f"Splitting data for tuning (up to year {TRAIN_VAL_END_YEAR})..."
    )
    df_train_val_master = df_master_processed[
        df_master_processed[DT_COL_NAME_TEMP].dt.year
        <= TRAIN_VAL_END_YEAR
    ].copy()

    if df_train_val_master.empty:
        raise ValueError(
            f"Train/Val data is empty after split at year "
            f"{TRAIN_VAL_END_YEAR}."
        )
    logger.info(
        f"Data for tuning (train+val) shape: {df_train_val_master.shape}"
    )

    # 2a. Further split into training & validation for tuner
    unique_locations = df_train_val_master[[LON_COL, LAT_COL]].drop_duplicates()

    if len(unique_locations) < 2:
        logger.warning(
            "Fewer than 2 unique locations. Using random split, "
            "may cause leakage."
        )
        df_tune_train, df_tune_val = train_test_split(
            df_train_val_master,
            test_size=VAL_SPLIT_RATIO,
            random_state=TUNER_SEED,
            shuffle=True
        )
    else:
        train_loc_groups, val_loc_groups = train_test_split(
            unique_locations,
            test_size=VAL_SPLIT_RATIO,
            random_state=TUNER_SEED
        )
        df_tune_train = df_train_val_master.merge(
            train_loc_groups, on=[LON_COL, LAT_COL], how='inner'
        )
        df_tune_val = df_train_val_master.merge(
            val_loc_groups, on=[LON_COL, LAT_COL], how='inner'
        )

    if df_tune_train.empty or df_tune_val.empty:
        logger.error(
            "Training or validation DataFrame is empty after split. "
            "Check data distribution and VAL_SPLIT_RATIO."
        )
        exit()

    logger.info(
        f"Tuner training data shape: {df_tune_train.shape}"
    )
    logger.info(
        f"Tuner validation data shape: {df_tune_val.shape}"
    )

    # 3. Prepare PINN data sequences for training & validation
    logger.info("Preparing sequences for tuner training...")
    inputs_train_dict, targets_train_dict, coord_scaler_train = \
        prepare_pinn_data_sequences(
            df=df_tune_train,
            time_col=TIME_COL_NUMERIC_PINN,
            lon_col=LON_COL,
            lat_col=LAT_COL,
            subsidence_col=SUBSIDENCE_COL,
            gwl_col=GWL_COL,
            dynamic_cols=[
                c for c in [
                    GWL_COL,
                    'rainfall_mm',
                    'normalized_density',
                    'normalized_seismic_risk_score'
                ] if c in df_tune_train.columns
            ],
            static_cols=list(encoded_feature_names_list),
            future_cols=[c for c in ['rainfall_mm']
                         if c in df_tune_train.columns],
            group_id_cols=[LON_COL, LAT_COL],
            time_steps=TIME_STEPS,
            forecast_horizon=FORECAST_HORIZON,
            output_subsidence_dim=OUT_S_DIM,
            output_gwl_dim=OUT_G_DIM,
            normalize_coords=True,
            return_coord_scaler=True,
            cols_to_scale='auto',
            verbose=1
        )

    if inputs_train_dict['coords'].shape[0] == 0:
        raise ValueError(
            "Sequence generation produced no training samples."
        )

    logger.info("Preparing sequences for tuner validation...")
    inputs_val_dict, targets_val_dict = \
        prepare_pinn_data_sequences(
            df=df_tune_val,
            time_col=TIME_COL_NUMERIC_PINN,
            lon_col=LON_COL,
            lat_col=LAT_COL,
            subsidence_col=SUBSIDENCE_COL,
            gwl_col=GWL_COL,
            dynamic_cols=[
                c for c in [
                    GWL_COL,
                    'rainfall_mm',
                    'normalized_density',
                    'normalized_seismic_risk_score'
                ] if c in df_tune_val.columns
            ],
            static_cols=list(encoded_feature_names_list),
            future_cols=[c for c in ['rainfall_mm']
                         if c in df_tune_val.columns],
            group_id_cols=[LON_COL, LAT_COL],
            time_steps=TIME_STEPS,
            forecast_horizon=FORECAST_HORIZON,
            output_subsidence_dim=OUT_S_DIM,
            output_gwl_dim=OUT_G_DIM,
            normalize_coords=True,
            return_coord_scaler=False,
            cols_to_scale="auto",
            verbose=1
        )

    if inputs_val_dict['coords'].shape[0] == 0:
        raise ValueError(
            "Sequence generation produced no validation samples."
        )

    # 4. Define fixed model parameters & hyperparameter space
    current_fixed_params = BASE_FIXED_PARAMS.copy()
    current_fixed_params.update({
        "static_input_dim": inputs_train_dict.get(
            'static_features', np.zeros((0, 0))
        ).shape[-1],
        "dynamic_input_dim": inputs_train_dict[
            'dynamic_features'
        ].shape[-1],
        "future_input_dim": inputs_train_dict.get(
            'future_features', np.zeros((0, 0, 0))
        ).shape[-1],
        "output_subsidence_dim": OUT_S_DIM,
        "output_gwl_dim": OUT_G_DIM,
        "forecast_horizon": FORECAST_HORIZON,
        "quantiles": BASE_FIXED_PARAMS.get("quantiles"),
        "max_window_size": TIME_STEPS,
    })

    param_space_config = {
        'embed_dim': {
            'min_value': 32,
            'max_value': 64,
            'step': 32
        },
        'hidden_units': {
            'min_value': 32,
            'max_value': 128,
            'step': 32
        },
        'lstm_units': {
            'min_value': 32,
            'max_value': 128,
            'step': 32
        },
        'attention_units': {
            'min_value': 16,
            'max_value': 64,
            'step': 16
        },
        'num_heads': [2, 4],
        'dropout_rate': {
            'min_value': 0.05,
            'max_value': 0.30,
            'step': 0.05
        },
        'vsn_units': {
            'min_value': 16,
            'max_value': 64,
            'step': 16
        },
        'activation': ['relu', 'gelu'],
        'learning_rate': [1e-4, 5e-5, 1e-5],
        'lambda_pde': {
            'min_value': 0.1,
            'max_value': 0.5,
            'step': 0.05,
            'sampling': 'linear'
        },
        'pinn_coefficient_C_type': ['learnable', 'fixed'],
        'pinn_coefficient_C_value': {
            'min_value': 1e-3,
            'max_value': 1e-1,
            'sampling': 'log'
        },
        'pde_mode': ['consolidation', 'none']
    }

    # 5. Instantiate PIHALTuner
    logger.info("Instantiating PIHALTuner...")
    tuner = PIHALTuner(
        fixed_model_params=current_fixed_params,
        param_space=param_space_config,
        objective=TUNER_OBJECTIVE,
        max_trials=MAX_TRIALS,
        project_name=f"{CITY_NAME}_{MODEL_NAME_TUNED}",
        directory=os.path.join(RUN_OUTPUT_PATH,
                               "tuner_dir_zhongshan"),
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        tuner_type=TUNER_TYPE,
        seed=TUNER_SEED,
        overwrite_tuner=True
    )

    # 6. Run Hyperparameter Search
    logger.info("Creating TensorFlow datasets for Keras Tuner...")

    
    targets_val_dict = rename_dict_keys(
        targets_val_dict,
        param_to_rename={"subsidence": 'subs_pred',
                         'gwl': "gwl_pred"}
    )
    targets_train_dict = rename_dict_keys(
        targets_train_dict,
        param_to_rename={"subsidence": 'subs_pred',
                         'gwl': "gwl_pred"}
    )

    if not inputs_train_dict or not targets_train_dict:
        raise ValueError(
            "inputs_train_dict or targets_train_dict is None "
            "or empty before creating datasets."
        )

    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (inputs_train_dict, targets_train_dict)
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    logger.info(
        f"Training dataset created. Cardinality: "
        f"{train_tf_dataset.cardinality().numpy()}"
    )

    val_tf_dataset = None
    if inputs_val_dict and targets_val_dict:
        val_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs_val_dict, targets_val_dict)
        ).batch(BATCH_SIZE).prefetch(
            tf.data.AUTOTUNE
        )
        logger.info(
            f"Validation dataset created. Cardinality: "
            f"{val_tf_dataset.cardinality().numpy()}"
        )
    else:
        logger.warning(
            "Validation data not available. Tuner will skip validation."
        )

    logger.info(
        f"Starting hyperparameter search with {TUNER_TYPE} tuner..."
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=TUNER_OBJECTIVE,
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    print(tuner.fixed_model_params)

    try:
        print("Leverage Keras HyperModel search...")
        tuner.search(
            train_data=train_tf_dataset,
            validation_data=val_tf_dataset,
            epochs=EPOCHS_PER_TRIAL,
            callbacks=[early_stopping_cb],
            verbose=1
        )
    except Exception:
        print("Falling back to tuner.run(...)")
        tuner.run(
            inputs=inputs_train_dict,
            y=targets_train_dict,
            validation_data=(inputs_val_dict, targets_val_dict),
            epochs=EPOCHS_PER_TRIAL,
            callbacks=[early_stopping_cb],
            batch_size=BATCH_SIZE,
            verbose=1
        )

    logger.info("Hyperparameter search completed.")

    # 7. Retrieve and display results
    try:
        best_hps_list = tuner.get_best_hyperparameters(num_trials=1)
        if not best_hps_list:
            logger.error("No best hyperparameters found.")
            exit()
        best_hps = best_hps_list[0]

        logger.info("\n--- Best Hyperparameters Found ---")
        for param_name, value in best_hps.values.items():
            logger.info(f"{param_name}: {value}")

        # Save best hyperparameters to file
        best_hps_path = os.path.join(
            RUN_OUTPUT_PATH,
            f"{MODEL_NAME_TUNED}_best_hyperparameters.txt"
        )
        with open(best_hps_path, 'w') as f:
            for param_name, value in best_hps.values.items():
                f.write(f"{param_name}: {value}\n")
        logger.info(f"Best HPs saved to: {best_hps_path}")

        # Retrieve best model
        logger.info("Retrieving the best model(s)...")
        best_models = tuner.get_best_models(num_models=1)
        if not best_models or best_models[0] is None:
            logger.error("Could not retrieve the best model.")
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
        logger.error(f"Error retrieving/saving tuning results: {ex_results}")

    logger.info(f"--- PIHALNet Tuning for {CITY_NAME} Finished ---")
    logger.info(
        f"Results & artifacts in: "
        f"{os.path.join(RUN_OUTPUT_PATH, 'tuner_dir_zhongshan')}"
    )
