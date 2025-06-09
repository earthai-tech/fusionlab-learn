.. _tuning_pihalnet_example:

=========================================
Example: Tuning PIHALNet with PIHALTuner
=========================================

This page provides practical examples of how to use the
:class:`~fusionlab.nn.pinn.tuning.PIHALTuner` to perform
hyperparameter tuning for the :class:`~fusionlab.nn.pinn.models.PIHALNet`
model. We'll cover the typical workflow, from data preparation to
retrieving the best model, first with synthetic data for a clear
demonstration, and then outlining the steps for a real application.

Prerequisites
-------------
Ensure you have ``fusionlab`` installed with its dependencies, including
``tensorflow`` and ``keras-tuner``.

.. code-block:: python
   :linenos:
   
   # Common imports for the examples
   import os
   import logging
   import numpy as np
   import pandas as pd
   import tensorflow as tf
   from sklearn.model_selection import train_test_split # For splitting data
   import joblib # For saving/loading scalers/encoders

   from fusionlab.nn.pinn.tuning import PIHALTuner
   from fusionlab.nn.pinn.models import PIHALNet # PIHALTuner needs to build this
   from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences
   # from fusionlab.datasets.load import load_subsidence_pinn_data # For real data
   # from fusionlab.nn.losses import combined_quantile_loss # If using custom quantile loss

   # Basic configuration for logging 
   logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


Section 1: Tuning with Synthetic Data
---------------------------------------
This section walks through the process using synthetically generated
data. This allows you to understand the mechanics of ``PIHALTuner``
without needing a specific large dataset immediately. Each step includes
code to run and a placeholder for its expected output.

Step 1.1: Configuration for Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, we define configurations for our synthetic data generation and
the tuning process. These are simplified for demonstration.

.. code-block:: python
   :linenos: 
   
   # --- Configuration for Synthetic Data Example ---
   SYNTHETIC_CITY_NAME = "SyntheticCity"
   SYNTHETIC_RUN_OUTPUT_PATH = f"{SYNTHETIC_CITY_NAME}_tuning_synthetic_output"
   SYNTHETIC_MODEL_NAME_TUNED = f"PIHALNet_{SYNTHETIC_CITY_NAME}_TunedSynthetic"

   # Synthetic Data Parameters
   SYNTHETIC_N_SAMPLES_TOTAL = 1000  # Total raw data points to generate
   SYNTHETIC_N_LOCATIONS = 10     # Number of unique spatial locations
   SYNTHETIC_YEARS_PER_LOCATION = SYNTHETIC_N_SAMPLES_TOTAL // SYNTHETIC_N_LOCATIONS

   # Column names for synthetic data
   SYNTHETIC_TIME_COL = "year"
   SYNTHETIC_DT_COL_NAME_TEMP = "datetime_temp_synth"
   SYNTHETIC_LON_COL = "longitude"
   SYNTHETIC_LAT_COL = "latitude"
   SYNTHETIC_SUBSIDENCE_COL = "subsidence"
   SYNTHETIC_GWL_COL = "gwl"
   SYNTHETIC_CAT_COL = ['geology_s'] # Using a different name to avoid global conflicts
   SYNTHETIC_NUM_DRIVER_COLS = ['rainfall_s', 'pumping_s']

   # Sequence Parameters for PIHALNet (can be small for synthetic example)
   SYNTHETIC_TIME_STEPS = 5
   SYNTHETIC_FORECAST_HORIZON = 5
   SYNTHETIC_OUTPUT_S_DIM = 1
   SYNTHETIC_OUTPUT_G_DIM = 1

   # Data Splitting & Batching for Tuner
   SYNTHETIC_TRAIN_VAL_END_YEAR = 2020 # Not strictly needed if all data is used
   SYNTHETIC_VAL_SPLIT = 0.25
   SYNTHETIC_BATCH_SIZE = 16 # Smaller batch size for smaller data

   # Tuner Configuration (minimal for quick example)
   SYNTHETIC_TUNER_OBJECTIVE = 'val_total_loss'
   SYNTHETIC_MAX_TRIALS = 3 # Very few trials for a quick run
   SYNTHETIC_EPOCHS_PER_TRIAL = 2 # Very few epochs
   SYNTHETIC_TUNER_TYPE = 'randomsearch' # Faster than hyperband for few trials
   SYNTHETIC_TUNER_SEED = 123

   print("Synthetic data configurations set.")
   # Ensure output directory exists
   os.makedirs(SYNTHETIC_RUN_OUTPUT_PATH, exist_ok=True)

**Expected Output:**

.. code-block:: text

   Synthetic data configurations set.

Step 1.2: Synthetic Data Generation and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, we generate a simple synthetic dataset that mimics the structure
needed by ``PIHALNet``. This includes time series for multiple locations,
categorical features, and numerical features. We also perform basic
preprocessing like encoding and scaling.

.. code-block:: python
   :linenos: 
   
   def generate_synthetic_city_data(
       n_locations: int,
       years_per_location: int,
       time_col: str,
       dt_col_name: str,
       lon_col: str, lat_col: str,
       subs_col: str, gwl_col: str,
       cat_col_names: List[str],
       num_driver_col_names: List[str],
       output_path: str,
       city_name: str
   ) -> pd.DataFrame:
       logger.info(f"Generating synthetic data for {n_locations} locations, "
                   f"{years_per_location} years each.")
       all_rows = []
       start_year = 2000
       for i in range(n_locations):
           loc_lon = 113.0 + i * 0.01
           loc_lat = 22.0 + i * 0.01
           for year_offset in range(years_per_location):
               current_year = start_year + year_offset
               row = {
                   time_col: current_year,
                   lon_col: loc_lon, lat_col: loc_lat,
                   subs_col: -10 - i*0.5 - year_offset * 0.2 + np.random.randn()*2,
                   gwl_col: 5 - i*0.1 + year_offset * 0.1 + np.random.randn()*0.5,
               }
               for cat_c in cat_col_names:
                   row[cat_c] = f"Type{np.random.choice(['A', 'B'])}"
               for num_c in num_driver_col_names:
                   row[num_c] = np.random.rand() * 100
               all_rows.append(row)
       
       df = pd.DataFrame(all_rows)
       df[dt_col_name] = pd.to_datetime(df[time_col], format='%Y')
       
       # Encode Categorical
       global synthetic_encoded_feature_names
       synthetic_encoded_feature_names = []
       cats_to_encode = [c for c in cat_col_names if c in df.columns]
       if cats_to_encode:
           encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
           encoded_data = encoder.fit_transform(df[cats_to_encode])
           ohe_cols = encoder.get_feature_names_out(cats_to_encode)
           synthetic_encoded_feature_names.extend(ohe_cols)
           enc_df = pd.DataFrame(encoded_data, columns=ohe_cols, index=df.index)
           df = pd.concat([df.drop(columns=cats_to_encode), enc_df], axis=1)
           joblib.dump(encoder, os.path.join(output_path, f"{city_name}_synth_ohe.joblib"))

       # Scale Numerical Drivers
       num_to_scale = [c for c in num_driver_col_names if c in df.columns]
       if num_to_scale:
           scaler = MinMaxScaler()
           df[num_to_scale] = scaler.fit_transform(df[num_to_scale])
           joblib.dump(scaler, os.path.join(output_path, f"{city_name}_synth_scaler.joblib"))

       global SYNTHETIC_TIME_COL_NUMERIC_PINN
       SYNTHETIC_TIME_COL_NUMERIC_PINN = f"{time_col}_numeric_pinn_synth"
       df[SYNTHETIC_TIME_COL_NUMERIC_PINN] = (
           df[dt_col_name].dt.year +
           (df[dt_col_name].dt.dayofyear - 1) /
           (365 + df[dt_col_name].dt.is_leap_year.astype(int))
       )
       logger.info(f"Synthetic data generated and preprocessed. Shape: {df.shape}")
       return df

   df_synthetic_processed = generate_synthetic_city_data(
       SYNTHETIC_N_LOCATIONS, SYNTHETIC_YEARS_PER_LOCATION,
       SYNTHETIC_TIME_COL, SYNTHETIC_DT_COL_NAME_TEMP,
       SYNTHETIC_LON_COL, SYNTHETIC_LAT_COL,
       SYNTHETIC_SUBSIDENCE_COL, SYNTHETIC_GWL_COL,
       SYNTHETIC_CAT_COL, SYNTHETIC_NUM_DRIVER_COLS,
       SYNTHETIC_RUN_OUTPUT_PATH, SYNTHETIC_CITY_NAME
   )
   print(df_synthetic_processed.head())

**Expected Output (will vary due to randomness):**

.. code-block:: text

   Synthetic data generated and preprocessed. Shape: (100, 12)
      year  longitude  ...  geology_s_TypeB  year_numeric_pinn_synth
   0  2000      113.0  ...              1.0                   2000.0
   1  2001      113.0  ...              0.0                   2001.0
   2  2002      113.0  ...              1.0                   2002.0
   3  2003      113.0  ...              1.0                   2003.0
   4  2004      113.0  ...              1.0                   2004.0

   [5 rows x 11 columns]

Step 1.3: Prepare Synthetic Data for Tuner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We split the synthetic data and use ``prepare_pinn_data_sequences``
to create the input/target dictionaries for ``PIHALNet``.

.. code-block:: python
   :linenos: 
   
   # Split synthetic data (can use all for train/val in this simple case or split by location)
   synth_unique_locs = df_synthetic_processed[[SYNTHETIC_LON_COL, SYNTHETIC_LAT_COL]].drop_duplicates()
   synth_train_locs, synth_val_locs = train_test_split(
       synth_unique_locs, test_size=SYNTHETIC_VAL_SPLIT, random_state=SYNTHETIC_TUNER_SEED
   )
   df_synth_tuner_train = df_synthetic_processed.merge(
       synth_train_locs, on=[SYNTHETIC_LON_COL, SYNTHETIC_LAT_COL], how='inner'
   )
   df_synth_tuner_val = df_synthetic_processed.merge(
       synth_val_locs, on=[SYNTHETIC_LON_COL, SYNTHETIC_LAT_COL], how='inner'
   )

   logger.info(f"Synthetic tuner training data part shape: {df_synth_tuner_train.shape}")
   logger.info(f"Synthetic tuner validation data part shape: {df_synth_tuner_val.shape}")

   # Define feature lists for synthetic data
   synth_static_features_list = list(synthetic_encoded_feature_names)
   synth_dynamic_features_list = [SYNTHETIC_GWL_COL] + [
       c for c in SYNTHETIC_NUM_DRIVER_COLS if c in df_synth_tuner_train.columns
   ]
   synth_future_features_list = [ # Example: use one of the drivers as a "known future"
       c for c in SYNTHETIC_NUM_DRIVER_COLS[:1] if c in df_synth_tuner_train.columns
   ]

   # Prepare training sequences
   inputs_train_np_s, targets_train_np_s = prepare_pinn_data_sequences(
       df=df_synth_tuner_train, time_col=SYNTHETIC_TIME_COL_NUMERIC_PINN,
       lon_col=SYNTHETIC_LON_COL, lat_col=SYNTHETIC_LAT_COL,
       subsidence_col=SYNTHETIC_SUBSIDENCE_COL, gwl_col=SYNTHETIC_GWL_COL,
       dynamic_cols=synth_dynamic_features_list, static_cols=synth_static_features_list,
       future_cols=synth_future_features_list, group_id_cols=[SYNTHETIC_LON_COL, SYNTHETIC_LAT_COL],
       time_steps=SYNTHETIC_TIME_STEPS, forecast_horizon=SYNTHETIC_FORECAST_HORIZON,
       output_subsidence_dim=SYNTHETIC_OUTPUT_S_DIM, output_gwl_dim=SYNTHETIC_OUTPUT_G_DIM,
       normalize_coords=True, return_coord_scaler=False, verbose=0
   )

   # Prepare validation sequences
   inputs_val_np_s, targets_val_np_s = prepare_pinn_data_sequences(
       df=df_synth_tuner_val, time_col=SYNTHETIC_TIME_COL_NUMERIC_PINN,
       lon_col=SYNTHETIC_LON_COL, lat_col=SYNTHETIC_LAT_COL,
       subsidence_col=SYNTHETIC_SUBSIDENCE_COL, gwl_col=SYNTHETIC_GWL_COL,
       dynamic_cols=synth_dynamic_features_list, static_cols=synth_static_features_list,
       future_cols=synth_future_features_list, group_id_cols=[SYNTHETIC_LON_COL, SYNTHETIC_LAT_COL],
       time_steps=SYNTHETIC_TIME_STEPS, forecast_horizon=SYNTHETIC_FORECAST_HORIZON,
       output_subsidence_dim=SYNTHETIC_OUTPUT_S_DIM, output_gwl_dim=SYNTHETIC_OUTPUT_G_DIM,
       normalize_coords=True, return_coord_scaler=False, verbose=0
   )

   print(f"Num training sequences: {inputs_train_np_s['coords'].shape[0]}")
   print(f"Num validation sequences: {inputs_val_np_s['coords'].shape[0]}")
   if inputs_train_np_s['coords'].shape[0] == 0 or inputs_val_np_s['coords'].shape[0] == 0:
       print("WARNING: Empty train or val sequences for synthetic data. Adjust generation params.")

**Expected Output (will vary based on sequence generation success):**

.. code-block:: text

   Num training sequences: 658
   Num validation sequences: 282

Step 1.4: Configure and Run PIHALTuner with Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We set up ``PIHALTuner`` with fixed parameters derived from our synthetic
data and a simplified hyperparameter search space.

.. code-block:: python
   :linenos: 
   
   # Define fixed parameters for PIHALTuner using synthetic data shapes
   fixed_params_synth = {
       "static_input_dim": inputs_train_np_s.get('static_features', np.zeros((0,0))).shape[-1],
       "dynamic_input_dim": inputs_train_np_s['dynamic_features'].shape[-1],
       "future_input_dim": inputs_train_np_s.get('future_features', np.zeros((0,0,0))).shape[-1],
       "output_subsidence_dim": SYNTHETIC_OUTPUT_S_DIM,
       "output_gwl_dim": SYNTHETIC_OUTPUT_G_DIM,
       "forecast_horizon": SYNTHETIC_FORECAST_HORIZON,
       "quantiles": None, # Point predictions for simpler synthetic example
       "max_window_size": SYNTHETIC_TIME_STEPS,
       "pde_mode": "none", # No PDE for simple synthetic example
       "pinn_coefficient_C": None,
       "loss_weights": {'subs_pred': 1.0, 'gwl_pred': 1.0},
       "use_vsn": False, # Simpler model without VSN for quick test
       "scales": [1], # Single scale LSTM
       "memory_size": 10, # Small memory
   }

   # Simplified hyperparameter space for synthetic example
   param_space_synth = {
       'embed_dim': {'min_value': 8, 'max_value': 16, 'step': 8},
       'hidden_units': {'min_value': 16, 'max_value': 32, 'step': 16},
       'lstm_units': {'min_value': 16, 'max_value': 32, 'step': 16},
       'attention_units': {'min_value': 8, 'max_value': 16, 'step': 8},
       'num_heads': [1, 2],
       'dropout_rate': [0.0, 0.1],
       'learning_rate': [1e-3, 5e-3],
       # 'lambda_pde': [0.0] # Not tuning if pde_mode is none
   }

   logger.info("Instantiating PIHALTuner for synthetic data...")
   synthetic_tuner = PIHALTuner(
       fixed_model_params=fixed_params_synth,
       param_space=param_space_synth,
       objective=SYNTHETIC_TUNER_OBJECTIVE,
       max_trials=SYNTHETIC_MAX_TRIALS,
       project_name=SYNTHETIC_MODEL_NAME_TUNED,
       directory=os.path.join(SYNTHETIC_RUN_OUTPUT_PATH, "tuner_synth_results"),
       executions_per_trial=1,
       tuner_type=SYNTHETIC_TUNER_TYPE,
       seed=SYNTHETIC_TUNER_SEED,
       overwrite_tuner=True
   )

   # Callbacks
   synthetic_early_stopping = tf.keras.callbacks.EarlyStopping(
       monitor=SYNTHETIC_TUNER_OBJECTIVE, patience=2, restore_best_weights=True, verbose=0
   )

   logger.info(f"Starting synthetic data hyperparameter search ({SYNTHETIC_TUNER_TYPE})...")
   
   if inputs_train_np_s['coords'].shape[0] > 0 and inputs_val_np_s['coords'].shape[0] > 0:
       synthetic_tuner.run(
           inputs=inputs_train_np_s,
           y=targets_train_np_s,
           validation_data=(inputs_val_np_s, targets_val_np_s),
           epochs=SYNTHETIC_EPOCHS_PER_TRIAL,
           batch_size=SYNTHETIC_BATCH_SIZE,
           callbacks=[synthetic_early_stopping],
           verbose=1
       )
       logger.info("Synthetic data hyperparameter search completed.")
   else:
       logger.error("Cannot start tuner search: training or validation sequences are empty for synthetic data.")

**Expected Output (will be verbose from Keras Tuner):**

.. code-block:: text

   ...(logger messages for instantiation)...
   Starting synthetic data hyperparameter search (randomsearch)...
   Trial 1 Complete [...]
   Best val_total_loss So Far: ...
   ...(more trials)...
   Synthetic data hyperparameter search completed.

Step 1.5: Retrieve and Interpret Results (Synthetic Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the search, we inspect the best hyperparameters found for our
synthetic data tuning run.

.. code-block:: python

   try:
       best_hps_list_s = synthetic_tuner.get_best_hyperparameters(num_trials=1)
       if not best_hps_list_s:
           logger.error("Synthetic tuner could not retrieve best hyperparameters.")
       else:
           best_hps_s = best_hps_list_s[0]
           logger.info("\n--- Best Hyperparameters (Synthetic Data) ---")
           for param_name, value in best_hps_s.values.items():
               logger.info(f"  {param_name}: {value}")

           best_models_s = synthetic_tuner.get_best_models(num_models=1)
           if best_models_s and best_models_s[0] is not None:
               logger.info("Best model from synthetic tuning retrieved.")
               # best_models_s[0].summary() # Optional: print summary
           else:
               logger.error("Synthetic tuner could not retrieve the best model instance.")
   except Exception as e_results_s:
       logger.error(f"Error during synthetic result retrieval: {e_results_s}")

   logger.info(
       f"Synthetic tuning finished. Check results in: "
       f"{os.path.join(SYNTHETIC_RUN_OUTPUT_PATH, 'tuner_synth_results')}"
   )

**Expected Output:**

.. code-block:: text

   ...(logger messages)...
   --- Best Hyperparameters (Synthetic Data) ---
     embed_dim: ...
     hidden_units: ...
     ...(other hyperparameters and their values)...
   Best model from synthetic tuning retrieved.
   Synthetic tuning finished. Check results in: SyntheticCity_tuning_synthetic_output/tuner_synth_results

This completes the synthetic data example. It demonstrates the full pipeline,
allowing users to test ``PIHALTuner`` and understand its operation with
controllable data.

Section 2: Real Application Case - Example Workflow
-------------------------------------------------------
For tuning ``PIHALNet`` on a real-world dataset like Zhongshan or Nansha,
the workflow follows the same fundamental steps as the synthetic data example,
but with careful attention to actual data characteristics, more extensive
preprocessing, and a more thorough hyperparameter search.

The detailed steps, as previously outlined (and which formed the original content
of this page), would involve:
1.  **Configuration**: Setting up paths, city-specific parameters, feature definitions, sequence parameters, and tuner settings appropriate for the real dataset.
2.  **Data Loading and Preprocessing**: Loading the actual city data (e.g., from a CSV file), performing robust cleaning, handling missing values, encoding categorical features (e.g., 'geology'), and scaling numerical driver features. This step is crucial and dataset-specific.
3.  **Prepare Data for Tuner**: Splitting the processed data into training and validation sets suitable for the tuner (e.g., using a time-based or location-aware split to prevent data leakage) and then using ``prepare_pinn_data_sequences`` to generate the sequence dictionaries.
4.  **Configure and Run PIHALTuner**: Defining the ``fixed_model_params`` (with dimensions inferred from the real prepared data) and a comprehensive ``param_space_config`` for the hyperparameters. Instantiating and running ``PIHALTuner`` for a significant number of trials and epochs.
5.  **Retrieve, Analyze, and Use Results**: Extracting the best hyperparameters, retrieving the best model, saving these artifacts, and potentially retraining the best model on a larger portion of the data before deployment or further evaluation.

Please refer to the code blocks from Step 1 to Step 5 in the initial version of this
document for a detailed code structure for a real application. The key is to adapt
the data loading, preprocessing, feature engineering, and fixed/hyperparameter
configurations to the specifics of your chosen real-world dataset.

Step 2.1: Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Step 2.2: Data Loading and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Step 2.3: Prepare Data for Tuner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Step 2.4: Configure and Run PIHALTuner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Step 2.5: Retrieve and Use Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
