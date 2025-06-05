.. _tuning_pihalnet_example:

=========================================
Example: Tuning PIHALNet with PIHALTuner
=========================================

This page provides a practical example of how to use the
:class:`~fusionlab.nn.pinn.tuning.PIHALTuner` to perform
hyperparameter tuning for the :class:`~fusionlab.nn.pinn.models.PIHALNet`
model. We'll cover the typical workflow, from data preparation to
retrieving the best model.

Prerequisites
-------------
Ensure you have ``fusionlab`` installed with its dependencies, including
``tensorflow`` and ``keras-tuner``. You will also need your dataset
(e.g., Zhongshan or Nansha subsidence data).

.. code-block:: python

   import os
   import numpy as np
   import pandas as pd
   import tensorflow as tf
   from sklearn.model_selection import train_test_split # For splitting data
   import joblib # For saving/loading scalers/encoders

   # FusionLab imports (adjust paths based on your project structure)
   from fusionlab.nn.pinn.tuning import PIHALTuner
   from fusionlab.nn.pinn.models import PIHALNet # PIHALTuner needs to build this
   from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences
   # from fusionlab.datasets.load import load_subsidence_pinn_data # If using this loader
   # from fusionlab.nn.losses import combined_quantile_loss # If using custom quantile loss
   from fusionlab._fusionlog import fusionlog # Assuming your logger

   logger = fusionlog().get_fusionlog_logger(__name__)


Step 1: Configuration
---------------------
Define essential configurations for your tuning run.

.. code-block:: python
   :linenos:

   # --- Configuration Constants ---
   DATA_FILE_PATH = "path/to/your/city_data.csv" # IMPORTANT: Update this!
   CITY_NAME = "your_city" # e.g., "zhongshan" or "nansha"
   RUN_OUTPUT_PATH = f"{CITY_NAME}_tuning_example_output"
   MODEL_NAME_TUNED = f"PIHALNet_{CITY_NAME}_TunedExample"

   # Data Parameters (adapt these to your dataset)
   TIME_COL = "year"
   DT_COL_NAME_TEMP = "datetime_temp" # Temporary column for datetime objects
   LON_COL = "longitude"
   LAT_COL = "latitude"
   SUBSIDENCE_COL = "subsidence" # Your primary target
   GWL_COL = "GWL"               # Your secondary target

   # Example feature columns (customize for your dataset)
   CATEGORICAL_COLS = ['geology']
   NUMERICAL_DRIVER_COLS = [ # Features to scale and use as drivers
       'rainfall_mm', 'pumping_rate', 'river_level'
       # Add other relevant numerical features for your city
       # Exclude LON_COL, LAT_COL, TIME_COL (handled by prepare_pinn_data_sequences)
       # Exclude target columns (SUBSIDENCE_COL, GWL_COL)
   ]

   # Sequence Parameters for PIHALNet
   TIME_STEPS = 12        # Lookback window
   FORECAST_HORIZON = 3 # Prediction horizon
   OUTPUT_SUBSIDENCE_DIM = 1
   OUTPUT_GWL_DIM = 1

   # Data Splitting for Tuner
   # Data up to this year for training/validation by the tuner
   TRAIN_VAL_END_YEAR_TUNER = 2018 # Example
   # Proportion of the above data to use for tuner's internal validation
   VALIDATION_SPLIT_TUNER = 0.2
   BATCH_SIZE_TUNER = 32

   # Tuner Configuration
   TUNER_OBJECTIVE = 'val_total_loss' # Metric PIHALNet reports
   MAX_TRIALS_TUNER = 10 # Keep low for example, increase for real tuning
   EPOCHS_PER_TRIAL_TUNER = 25 # Max epochs per trial
   TUNER_TYPE = 'hyperband' # 'randomsearch', 'bayesianoptimization', or 'hyperband'
   TUNER_SEED = 42

Step 2: Data Loading and Preprocessing
--------------------------------------
Load your dataset and perform necessary preprocessing steps like cleaning,
encoding categorical features, and scaling numerical features. The
``load_subsidence_pinn_data`` function (if you're using it from
``fusionlab.datasets``) can handle some of this. Here, we show a
manual example.

.. code-block:: python
   :linenos:
   
   def load_and_preprocess_city_data(
       file_path: str,
       time_col: str,
       dt_col_name: str,
       categorical_cols: List[str],
       numerical_cols: List[str],
       run_output_path: str,
       city_name: str
   ) -> pd.DataFrame:
       logger.info(f"Loading data from: {file_path}")
       if not os.path.exists(file_path):
           # For this example, we'll raise an error.
           # In your script, you might generate dummy data as a fallback.
           raise FileNotFoundError(
               f"Data file NOT FOUND: {file_path}. Please update."
           )
       df = pd.read_csv(file_path)
       logger.info(f"Original data shape: {df.shape}")

       # Basic Cleaning (adapt to your data)
       essential_cols = [time_col, LON_COL, LAT_COL, SUBSIDENCE_COL, GWL_COL]
       df = df.dropna(subset=essential_cols).copy()

       # Convert time column to datetime
       try:
           if pd.api.types.is_numeric_dtype(df[time_col]):
               df[dt_col_name] = pd.to_datetime(df[time_col], format='%Y')
           else:
               df[dt_col_name] = pd.to_datetime(df[time_col])
       except Exception as e:
           logger.error(f"Error converting time column '{time_col}': {e}")
           raise
       df = df.dropna(subset=[dt_col_name])

       os.makedirs(run_output_path, exist_ok=True)

       # Encode Categorical Features
       global encoded_feature_names_list # Make accessible for feature definition
       encoded_feature_names_list = []
       cats_to_encode = [c for c in categorical_cols if c in df.columns]
       if cats_to_encode:
           encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
           encoded_data = encoder.fit_transform(df[cats_to_encode])
           ohe_cols = encoder.get_feature_names_out(cats_to_encode)
           encoded_feature_names_list.extend(ohe_cols)
           enc_df = pd.DataFrame(encoded_data, columns=ohe_cols, index=df.index)
           df = pd.concat([df.drop(columns=cats_to_encode), enc_df], axis=1)
           joblib.dump(encoder, os.path.join(run_output_path, f"{city_name}_ohe.joblib"))
           logger.info(f"Categorical features encoded: {cats_to_encode}")

       # Scale Numerical Driver Features
       num_to_scale = [c for c in numerical_cols if c in df.columns]
       if num_to_scale:
           scaler = MinMaxScaler()
           df[num_to_scale] = scaler.fit_transform(df[num_to_scale])
           joblib.dump(scaler, os.path.join(run_output_path, f"{city_name}_scaler.joblib"))
           logger.info(f"Numerical driver features scaled: {num_to_scale}")
       
       # Create the numerical time column for PINN sequences
       # Ensure this happens after all row manipulations (like dropna)
       global TIME_COL_NUMERIC_PINN # Make accessible
       TIME_COL_NUMERIC_PINN = f"{time_col}_numeric_pinn"
       df[TIME_COL_NUMERIC_PINN] = (
           df[dt_col_name].dt.year +
           (df[dt_col_name].dt.dayofyear - 1) /
           (365 + df[dt_col_name].dt.is_leap_year.astype(int))
       )
       logger.info(f"Processed data shape: {df.shape}")
       return df

   # Load and preprocess your data
   df_processed = load_and_preprocess_city_data(
       DATA_FILE_PATH, TIME_COL, DT_COL_NAME_TEMP,
       CATEGORICAL_COLS, NUMERICAL_DRIVER_COLS,
       RUN_OUTPUT_PATH, CITY_NAME
   )

Step 3: Prepare Data for Tuner
--------------------------------
Split the processed data into training and validation sets for the tuner.
Then, use ``prepare_pinn_data_sequences`` to format this data into the
structure required by ``PIHALNet``.

.. code-block:: python
   :linenos:
   
   # Split data for tuner
   df_for_tuner_train_val = df_processed[
       df_processed[DT_COL_NAME_TEMP].dt.year <= TRAIN_VAL_END_YEAR_TUNER
   ].copy()

   if df_for_tuner_train_val.empty:
       raise ValueError(f"No data available up to year {TRAIN_VAL_END_YEAR_TUNER}")

   # Split unique locations to avoid data leakage between train and val
   unique_locs = df_for_tuner_train_val[[LON_COL, LAT_COL]].drop_duplicates()
   if len(unique_locs) < 2:
       logger.warning("Very few unique locations for robust train/val split. Using random split on all data.")
       df_tuner_train, df_tuner_val = train_test_split(
           df_for_tuner_train_val, test_size=VALIDATION_SPLIT_TUNER, random_state=TUNER_SEED
       )
   else:
       train_locations, val_locations = train_test_split(
           unique_locs, test_size=VALIDATION_SPLIT_TUNER, random_state=TUNER_SEED
       )
       df_tuner_train = df_for_tuner_train_val.merge(train_locations, on=[LON_COL, LAT_COL], how='inner')
       df_tuner_val = df_for_tuner_train_val.merge(val_locations, on=[LON_COL, LAT_COL], how='inner')

   if df_tuner_train.empty or df_tuner_val.empty:
       raise ValueError("Tuner train or validation set is empty after location-based split.")

   logger.info(f"Tuner training data part shape: {df_tuner_train.shape}")
   logger.info(f"Tuner validation data part shape: {df_tuner_val.shape}")

   # Define feature sets for prepare_pinn_data_sequences
   # Static features often include one-hot encoded categoricals
   static_features_list = list(encoded_feature_names_list)
   # Dynamic features are typically scaled numerical drivers and targets (like GWL)
   dynamic_features_list = [GWL_COL] + [
       c for c in NUMERICAL_DRIVER_COLS if c in df_tuner_train.columns
   ]
   # Future features might include forecasted drivers like rainfall
   future_features_list = ['rainfall_mm'] # Example, if available and scaled
   future_features_list = [c for c in future_features_list if c in df_tuner_train.columns]


   # Prepare training sequences for the tuner
   logger.info("Preparing training sequences for tuner...")
   inputs_train_np, targets_train_np, coord_scaler = prepare_pinn_data_sequences(
       df=df_tuner_train,
       time_col=TIME_COL_NUMERIC_PINN, # Use the generated numeric time
       lon_col=LON_COL, lat_col=LAT_COL,
       subsidence_col=SUBSIDENCE_COL, gwl_col=GWL_COL,
       dynamic_cols=dynamic_features_list,
       static_cols=static_features_list,
       future_cols=future_features_list,
       group_id_cols=[LON_COL, LAT_COL],
       time_steps=TIME_STEPS,
       forecast_horizon=FORECAST_HORIZON,
       output_subsidence_dim=OUTPUT_SUBSIDENCE_DIM,
       output_gwl_dim=OUTPUT_GWL_DIM,
       normalize_coords=True, # Let sequence prep handle coordinate normalization
       return_coord_scaler=True, # We might need this scaler later
       # cols_to_scale ='auto', # scale  numeric data except one hot encoding 
       verbose=7
   )

   # Prepare validation sequences for the tuner
   logger.info("Preparing validation sequences for tuner...")
   inputs_val_np, targets_val_np, _ = prepare_pinn_data_sequences(
       df=df_tuner_val,
       time_col=TIME_COL_NUMERIC_PINN,
       lon_col=LON_COL, lat_col=LAT_COL,
       subsidence_col=SUBSIDENCE_COL, gwl_col=GWL_COL,
       dynamic_cols=dynamic_features_list,
       static_cols=static_features_list,
       future_cols=future_features_list,
       group_id_cols=[LON_COL, LAT_COL],
       time_steps=TIME_STEPS,
       forecast_horizon=FORECAST_HORIZON,
       output_subsidence_dim=OUTPUT_SUBSIDENCE_DIM,
       output_gwl_dim=OUTPUT_GWL_DIM,
       normalize_coords=True, # Use same strategy
       return_coord_scaler=False, # Scaler from training data is usually sufficient
       verbose=1
   )

   if inputs_train_np['coords'].shape[0] == 0 or inputs_val_np['coords'].shape[0] == 0:
       raise ValueError("Sequence preparation resulted in empty training or validation data for the tuner.")

Step 4: Configure and Run PIHALTuner
------------------------------------
Define the fixed parameters for ``PIHALNet`` (many are inferred from data)
and the hyperparameter search space. Then, instantiate and run ``PIHALTuner``.

.. code-block:: python
   :linenos:
   
   # Define fixed parameters for PIHALTuner
   # These are derived from data or set as constants for this tuning run
   fixed_model_params_for_tuner = {
       "static_input_dim": inputs_train_np.get('static_features', np.zeros((0,0))).shape[-1],
       "dynamic_input_dim": inputs_train_np['dynamic_features'].shape[-1],
       "future_input_dim": inputs_train_np.get('future_features', np.zeros((0,0,0))).shape[-1],
       "output_subsidence_dim": OUTPUT_SUBSIDENCE_DIM,
       "output_gwl_dim": OUTPUT_GWL_DIM,
       "forecast_horizon": FORECAST_HORIZON,
       "quantiles": [0.1, 0.5, 0.9], # Example, or None
       "max_window_size": TIME_STEPS,
       "pde_mode": "consolidation", # Example fixed PDE mode
       "pinn_coefficient_C": "learnable",
       "loss_weights": {'subs_pred': 1.0, 'gwl_pred': 0.8},
       # Add other PIHALNet parameters that should be fixed during tuning
       "scales": [1, 2], # Example fixed scales
       "memory_size": 50,
       "use_vsn": True,
   }

   # Define the hyperparameter search space
   param_space_config = {
       'embed_dim': {'min_value': 32, 'max_value': 64, 'step': 16},
       'hidden_units': {'min_value': 32, 'max_value': 128, 'step': 32},
       'lstm_units': {'min_value': 32, 'max_value': 128, 'step': 32},
       'attention_units': {'min_value': 16, 'max_value': 64, 'step': 16},
       'num_heads': [2, 4],
       'dropout_rate': {'min_value': 0.0, 'max_value': 0.2, 'step': 0.1},
       'vsn_units': {'min_value': 16, 'max_value': 32, 'step': 16},
       'activation': ['relu', 'gelu'],
       'learning_rate': [1e-4, 5e-4, 1e-3],
       'lambda_pde': {'min_value': 0.01, 'max_value': 0.5, 'sampling': 'linear'},
       # pinn_coefficient_C_type can also be tuned if 'pinn_coefficient_C' is not fixed
   }

   logger.info("Instantiating PIHALTuner...")
   tuner = PIHALTuner(
       fixed_model_params=fixed_model_params_for_tuner,
       param_space=param_space_config,
       objective=TUNER_OBJECTIVE,
       max_trials=MAX_TRIALS_TUNER,
       project_name=MODEL_NAME_TUNED,
       directory=os.path.join(RUN_OUTPUT_PATH, "tuner_results"),
       executions_per_trial=EXECUTIONS_PER_TRIAL,
       tuner_type=TUNER_TYPE,
       seed=TUNER_SEED,
       overwrite_tuner=True # Set to False to resume previous tuning
   )

   # Callbacks for the search
   early_stopping_cb = tf.keras.callbacks.EarlyStopping(
       monitor=TUNER_OBJECTIVE,
       patience=5, # Shorter patience for faster example
       restore_best_weights=True,
       verbose=1
   )

   logger.info(f"Starting hyperparameter search ({TUNER_TYPE})...")
   
   # PIHALTuner's `run` method expects NumPy dicts and handles tf.data.Dataset creation
   tuner.run( # Or use tuner.search if PINNTunerBase directly defines it
       inputs=inputs_train_np,
       y=targets_train_np, # Ensure keys are "subs_pred", "gwl_pred" or will be renamed
       validation_data=(inputs_val_np, targets_val_np),
       epochs=EPOCHS_PER_TRIAL_TUNER,
       batch_size=BATCH_SIZE_TUNER,
       callbacks=[early_stopping_cb],
       verbose=1
   )
   logger.info("Hyperparameter search completed.")

Step 5: Retrieve and Use Results
--------------------------------
After the search, you can get the best hyperparameters and the
best model instance.

.. code-block:: python
   :linenos:
   
   try:
       best_hps_list = tuner.get_best_hyperparameters(num_trials=1)
       if not best_hps_list:
           logger.error("Tuner could not retrieve best hyperparameters.")
       else:
           best_hps = best_hps_list[0]
           logger.info("\n--- Best Hyperparameters Found ---")
           for param_name, value in best_hps.values.items():
               logger.info(f"  {param_name}: {value}")

           # Save best HPs
           best_hps_path = os.path.join(
               RUN_OUTPUT_PATH, f"{MODEL_NAME_TUNED}_best_hps.txt"
           )
           with open(best_hps_path, 'w') as f:
               for param, val in best_hps.values.items():
                   f.write(f"{param}: {val}\n")
           logger.info(f"Best hyperparameters saved to: {best_hps_path}")

           # Get the best model
           best_models = tuner.get_best_models(num_models=1)
           if best_models and best_models[0] is not None:
               best_pihalnet_model = best_models[0]
               logger.info("\n--- Best Model Summary ---")
               best_pihalnet_model.summary(line_length=110)

               # Save the best model
               best_model_path = os.path.join(
                   RUN_OUTPUT_PATH, f"{MODEL_NAME_TUNED}_best_model.keras"
               )
               best_pihalnet_model.save(best_model_path)
               logger.info(f"Best PIHALNet model saved to: {best_model_path}")

               # Optionally, retrain the best model on more data or for more epochs
               # logger.info("Retraining best model on full train_val data...")
               # ... (prepare full train_val dataset) ...
               # best_pihalnet_model.fit(full_train_val_dataset, epochs=50, ...)

           else:
               logger.error("Tuner could not retrieve the best model instance.")

   except Exception as e_results:
       logger.error(f"Error during result retrieval or saving: {e_results}")

   logger.info(
       f"Tuning process finished. Check results in: "
       f"{os.path.join(RUN_OUTPUT_PATH, 'tuner_results')}"
   )

This example provides a template. You'll need to:
- **Update `DATA_FILE_PATH`** and other path/name configurations.
- **Customize `load_and_preprocess_city_data`** for your specific dataset's cleaning and feature engineering needs.
- **Adjust feature lists** (`CATEGORICAL_COLS`, `NUMERICAL_DRIVER_COLS`, `static_features_list`, etc.) to match your data.
- **Refine `fixed_model_params_for_tuner`** and **`param_space_config`** to suit the aspects of `PIHALNet` you want to fix versus tune.
- **Increase `MAX_TRIALS_TUNER` and `EPOCHS_PER_TRIAL_TUNER`** for a more thorough search in a real application.
