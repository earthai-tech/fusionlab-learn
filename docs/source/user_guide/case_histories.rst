.. _user_guide_case_histories:

==============
Case Histories
==============

This section showcases practical applications of the ``fusionlab``
library on real-world-inspired datasets, demonstrating common
workflows from data loading and preprocessing to model training and
forecasting.

These case studies focus on land subsidence prediction, using datasets
from the Nansha and Zhongshan areas in China. They illustrate how
to leverage ``fusionlab`` utilities and models like
:class:`~fusionlab.nn.XTFT` to handle complex spatio-temporal data
with static, dynamic, and future features.

.. note::
   The datasets used (`nansha_2000.csv`, `zhongshan_2000.csv`) are
   spatially sampled versions (approx. 2000 points each) derived from
   larger datasets used in research (e.g., [Liu24]_). While suitable
   for demonstrating workflows, results on these smaller samples may
   differ from those obtained on the full datasets.

.. [Liu24] Liu, J., Liu, W., Allechy, F. B., Zheng, Z., Liu, R.,
   & Kouadio, K. L. (2024). Machine learning-based techniques for
   land subsidence simulation in an urban area. *Journal of
   Environmental Management*, 352, 120078.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Nansha Land Subsidence Forecasting (Using XTFT)
-----------------------------------------------

This case study demonstrates a multi-step quantile forecasting workflow
for land subsidence in the Nansha district using the
:class:`~fusionlab.nn.XTFT` model. Research indicates that factors like
Groundwater Level (GWL) and Building Concentration (BC) are significant
drivers in this region [Liu24]_. We will follow a standard pipeline,
assuming the goal is to predict future subsidence based on historical
data and known future indicators.

*(Note: This example uses the smaller sampled dataset `nansha_2000.csv`
for demonstration purposes.)*

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Import necessary libraries, including `pandas`, `numpy`, `sklearn`
utilities, and relevant `fusionlab` functions for data loading,
preprocessing, modeling, and loss calculation.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import os
   import joblib # For saving scalers/encoders
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   from sklearn.model_selection import train_test_split # For simple splitting later

   # Fusionlab imports
   from fusionlab.datasets.load import fetch_nansha_data # Specific loader
   from fusionlab.utils.ts_utils import reshape_xtft_data, to_dt
   try:
       # Assume nan_ops handles missing values appropriately
       from fusionlab.utils.preprocessing import nan_ops
   except ImportError:
       def nan_ops(df, **kwargs): # Dummy if not found
           print("Warning: nan_ops not found, filling NaNs with ffill/bfill.")
           return df.ffill().bfill()
   from fusionlab.nn.transformers import XTFT
   from fusionlab.nn.losses import combined_quantile_loss

   # Suppress warnings and TF logs
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # Configuration
   output_dir_nansha = "./nansha_case_study_output"
   os.makedirs(output_dir_nansha, exist_ok=True)
   print("Nansha Case Study: Setup complete.")


Step 2: Load Raw Nansha Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Load the sampled Nansha dataset using the dedicated fetch function.
We load it as a DataFrame for initial inspection and preprocessing.

.. code-block:: python
   :linenos:

   # Load the nansha_2000.csv dataset
   try:
       nansha_df_raw = fetch_nansha_data(
           as_frame=True,
           verbose=True,
           download_if_missing=True # Allow download if needed
       )
       print("\nNansha data loaded successfully.")
       nansha_df_raw.info()
   except Exception as e:
       print(f"ERROR: Could not load Nansha data. Please ensure the file"
             f" 'nansha_2000.csv' is available or downloadable. Error: {e}")
       # Exit or raise further if data is essential
       raise


Step 3: Initial Cleaning and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ensure the time column ('year' in this dataset) is treated correctly.
Handle any immediate missing values using a defined strategy (e.g.,
filling).

.. code-block:: python
   :linenos:

   # Assume 'year' is the primary time column based on data description
   dt_col = 'year'
   df_clean = nansha_df_raw.copy()

   # Although 'year' is integer, treat it as the time index marker later
   # If actual datetime conversion needed:
   # df_clean = to_dt(df_clean, dt_col=dt_col, ...)

   # Handle missing values (example: using nan_ops utility)
   df_clean = nan_ops(df_clean, ops='sanitize', action='fill')
   print("\nInitial NaN check passed (or NaNs filled).")


Step 4: Preprocessing (Encoding & Scaling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Apply preprocessing steps necessary for the model. This typically
involves encoding categorical features and scaling numerical features.

.. code-block:: python
   :linenos:

   # 4a. Define Categorical and Numerical Columns for Nansha
   # Based on user description: 'geology' is categorical
   categorical_cols_n = ['geology']
   # Identify remaining numerical columns (excluding target/coords/year)
   numerical_cols_n = [
       'building_concentration', 'GWL', 'rainfall_mm',
       'normalized_seismic_risk_score', 'soil_thickness'
   ]
   target_col_n = 'subsidence'
   coord_cols_n = ['longitude', 'latitude']

   # 4b. Encode Categorical Features
   df_processed = df_clean.copy()
   encoder_info_n = {'columns': {}, 'names': {}}
   encoded_cols_generated = []

   if categorical_cols_n:
       print("\nEncoding categorical features...")
       cols_to_keep_temp = df_processed.columns.difference(
           categorical_cols_n).tolist()
       df_encoded_list = [df_processed[cols_to_keep_temp]]
       for col in categorical_cols_n:
           if col in df_processed.columns:
               encoder = OneHotEncoder(sparse_output=False,
                                       handle_unknown='ignore',
                                       dtype=np.float32)
               encoded_data = encoder.fit_transform(df_processed[[col]])
               new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
               encoded_df = pd.DataFrame(encoded_data, columns=new_cols,
                                         index=df_processed.index)
               df_encoded_list.append(encoded_df)
               encoder_info_n['columns'][col] = new_cols
               encoder_info_n['names'][col] = encoder.categories_[0]
               encoded_cols_generated.extend(new_cols) # Keep track
               print(f"  Encoded '{col}' -> {len(new_cols)} columns")
           else:
               warnings.warn(f"Categorical column '{col}' not found.")
       df_processed = pd.concat(df_encoded_list, axis=1)

   # 4c. Scale Numerical Features (including Target)
   scaler_n = StandardScaler() # Or MinMaxScaler()
   cols_to_scale_n = [c for c in numerical_cols_n if c in df_processed.columns]
   if target_col_n in df_processed.columns:
       cols_to_scale_n.append(target_col_n)

   if cols_to_scale_n:
       print(f"\nScaling numerical features: {cols_to_scale_n}...")
       df_processed[cols_to_scale_n] = scaler_n.fit_transform(
           df_processed[cols_to_scale_n]
       )
       # Save the scaler
       scaler_n_path = os.path.join(output_dir_nansha, "nansha_scaler.joblib")
       joblib.dump(scaler_n, scaler_n_path)
       print(f"Scaler saved to {scaler_n_path}")
   else:
       print("No numerical columns found to scale.")


Step 5: Define Feature Sets for Reshaping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Define the lists of column names corresponding to static, dynamic (past),
and future known features *after* the preprocessing steps (encoding,
scaling). This requires mapping original concepts to the potentially
new column names.

.. code-block:: python
   :linenos:

   # Define based on available columns in df_processed
   # Static: Coords + Encoded Categoricals + Base numericals?
   final_static_cols_n = list(coord_cols_n)
   final_static_cols_n.extend(encoded_cols_generated) # Add encoded geology
   # Add original static numericals if applicable (e.g., maybe soil_thickness varies slowly)
   # final_static_cols_n.append('soil_thickness') # Example if static

   # Dynamic: Non-static numericals + Time varying categoricals (if encoded differently)
   # Here: GWL, rainfall_mm, building_concentration, norm_seismic_score?
   # Assuming 'year' is the time index dt_col
   final_dynamic_cols_n = [
       'GWL', 'rainfall_mm', 'building_concentration',
       'normalized_seismic_risk_score', 'soil_thickness' # Assume dynamic for now
       ]
   final_dynamic_cols_n = [c for c in final_dynamic_cols_n if c in df_processed.columns]


   # Future: Known future events or time features
   # Here: rainfall_mm (assuming known forecast), year?
   final_future_cols_n = ['rainfall_mm'] # Example, needs domain knowledge
   final_future_cols_n = [c for c in final_future_cols_n if c in df_processed.columns]

   # Ensure all columns exist
   print("\nFinal Feature Sets for Reshaping:")
   print("  Static:", final_static_cols_n)
   print("  Dynamic:", final_dynamic_cols_n)
   print("  Future:", final_future_cols_n)
   print("  Target:", target_col_n)
   print("  DateTime:", dt_col)
   print("  Spatial:", spatial_cols)


Step 6: Generate Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~
Use :func:`~fusionlab.nn.utils.reshape_xtft_data` to create the final
sequence arrays needed by the XTFT model.

.. code-block:: python
   :linenos:

   time_steps = 4          # Example lookback (needs tuning)
   forecast_horizon = 4    # Example horizon (e.g., predict 4 years)

   print(f"\nReshaping data (T={time_steps}, H={forecast_horizon})...")
   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df_processed,
       dt_col=dt_col, # 'year' acts as time index here
       target_col=target_col_n,
       static_cols=final_static_cols_n,
       dynamic_cols=final_dynamic_cols_n,
       future_cols=final_future_cols_n,
       spatial_cols=spatial_cols, # ['longitude', 'latitude']
       time_steps=time_steps,
       forecast_horizons=forecast_horizon,
       verbose=1
   )

Step 7: Data Splitting for Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Split the generated sequence arrays into training and validation sets.
A simple chronological split based on the sequence order is shown.

.. code-block:: python
   :linenos:

   # Split sequences (example: 80% train, 20% validation)
   val_split_frac = 0.2
   n_seq_samples = static_data.shape[0]
   split_seq_idx = int(n_seq_samples * (1 - val_split_frac))

   X_train_static, X_val_static = static_data[:split_seq_idx], static_data[split_seq_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_seq_idx], dynamic_data[split_seq_idx:]
   X_train_future, X_val_future = future_data[:split_seq_idx], future_data[split_seq_idx:]
   y_train, y_val = target_data[:split_seq_idx], target_data[split_seq_idx:]

   # Package inputs into lists
   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future]

   print("\nSequence data split into Train/Validation sets.")
   print(f"  Train sequences: {len(y_train)}")
   print(f"  Validation sequences: {len(y_val)}")


Step 8: Define and Train XTFT Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate the XTFT model, configure it for quantile forecasting,
compile it with the appropriate loss function, and train it on the
prepared sequence data.

.. code-block:: python
   :linenos:

   quantiles = [0.1, 0.5, 0.9] # Define target quantiles
   output_dim = 1             # Predicting single target 'subsidence'

   # Instantiate XTFT
   xtft_model = XTFT(
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1],
       forecast_horizon=forecast_horizon,
       quantiles=quantiles,
       output_dim=output_dim,
       # --- Example Hyperparameters (Tune these) ---
       hidden_units=32,
       embed_dim=16,
       attention_units=16,
       lstm_units=32,
       num_heads=4,
       max_window_size=time_steps, # Typically match lookback
       memory_size=50,
       dropout_rate=0.1,
       # --- Anomaly Detection (Optional) ---
       # anomaly_detection_strategy=None, # Example: No anomaly det.
   )
   print("\nXTFT model instantiated.")

   # Compile with quantile loss
   loss_fn = combined_quantile_loss(quantiles=quantiles)
   xtft_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                      loss=loss_fn)
   print("Model compiled.")

   # Train the model
   print("Starting model training...")
   history = xtft_model.fit(
       train_inputs,
       y_train,
       validation_data=(val_inputs, y_val),
       epochs=10, # Increase significantly for real training
       batch_size=32,
       verbose=1
   )
   print("Training finished.")


Step 9: Forecasting and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use the trained model to make predictions on the validation set (or
new data). Inverse-transform the scaled predictions and evaluate
performance using relevant metrics (e.g., quantile loss, R², coverage).

.. code-block:: python
   :linenos:

   print("\nGenerating predictions on validation set...")
   predictions_scaled = xtft_model.predict(val_inputs)

   # Inverse transform predictions
   # (Requires loading/using the 'scaler' saved in Step 4c)
   # loaded_scaler = joblib.load(scaler_n_path)
   scaler_target = scalers[target_col_n] # Use scaler from memory

   num_val_samples = X_val_static.shape[0]
   pred_reshaped = predictions_scaled.reshape(-1, len(quantiles))
   predictions_inv = scaler_target.inverse_transform(pred_reshaped)
   predictions_final = predictions_inv.reshape(
       num_val_samples, forecast_horizon, len(quantiles)
       )

   # Inverse transform actual validation targets
   y_val_reshaped = y_val.reshape(-1, output_dim)
   y_val_inv = scaler_target.inverse_transform(y_val_reshaped)
   y_val_final = y_val_inv.reshape(
       num_val_samples, forecast_horizon, output_dim
       )

   print("Predictions inverse transformed.")
   print("Sample inverse predictions (median, q=0.5):")
   print(predictions_final[:2, :, 1]) # Show median for first 2 samples

   # Evaluation (Example: R2 score on median forecast)
   # Note: Proper evaluation uses appropriate metrics like pinball loss
   from sklearn.metrics import r2_score
   # Flatten horizon for basic R2
   r2 = r2_score(y_val_final.flatten(), predictions_final[:, :, 1].flatten())
   print(f"\nApprox R2 Score (Median vs Actual): {r2:.4f}")


Step 10: Visualization (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Visualize the quantile forecasts against actual values for specific
items or time periods using matplotlib or :func:`~fusionlab.nn.utils.visualize_forecasts`.

.. code-block:: python
   :linenos:

   # (Visualization code would go here, similar to previous examples,
   #  plotting y_val_final against columns of predictions_final)
   print("\nVisualization step placeholder.")


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _case_history_zhongshan:

Zhongshan Land Subsidence Forecasting (Using XTFT)
-----------------------------------------------------

This case study demonstrates a multi-step quantile forecasting workflow
for land subsidence in Zhongshan, China, using the
:class:`~fusionlab.nn.XTFT` model. Zhongshan, located in the Pearl
River Delta, experiences significant land subsidence challenges due to
urbanization and groundwater extraction, making it a relevant area for
advanced forecasting techniques [Liu25NS]_.

We will follow a pipeline that includes:
1. Loading and preprocessing the Zhongshan dataset using a dedicated utility.
2. Generating sequences suitable for XTFT.
3. Splitting data for training and validation.
4. Defining, compiling, and training the XTFT model for quantile output.
5. Generating out-of-sample forecasts using a helper function.
6. Visualizing the spatial forecast results.

*(Note: This example uses the smaller sampled dataset `zhongshan_2000.csv`
for demonstration purposes.)*

.. [Liu25NS] Liu, R., Kouadio, K. L., et al. (2025). Forecasting Urban
   Land Subsidence in the Era of Rapid Urbanization and Climate Stress.
   *Nature Sustainability* (Submitted).

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~
Import necessary libraries and the specific ``fusionlab`` functions for
data loading/processing (`load_processed_subsidence_data`), modeling
(`XTFT`), loss (`combined_quantile_loss`), forecasting
(`generate_forecast`), and visualization (`visualize_forecasts`).

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   import os
   import joblib # For loading scaler if saved by processing func

   # Fusionlab imports
   from fusionlab.datasets.load import load_processed_subsidence_data
   from fusionlab.nn.transformers import XTFT
   from fusionlab.nn.utils import (
       reshape_xtft_data, # Used internally by load_processed if needed
       generate_forecast,
       visualize_forecasts
   )
   from fusionlab.nn.losses import combined_quantile_loss
   from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

   # Suppress warnings and TF logs
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # Configuration
   output_dir_zhongshan = "./zhongshan_case_study_output"
   os.makedirs(output_dir_zhongshan, exist_ok=True)
   print("Zhongshan Case Study: Setup complete.")


Step 2: Load and Preprocess Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use the :func:`~fusionlab.datasets.load.load_processed_subsidence_data`
function. This helper encapsulates loading the raw `zhongshan_2000.csv`
data, applying the preprocessing steps (feature selection, NaN filling,
OneHotEncoding for 'geology' and 'density_tier', MinMaxScaler), and
returns the processed DataFrame. We disable sequence generation at this
stage (`return_sequences=False`).

.. code-block:: python
   :linenos:

   print("Loading and preprocessing Zhongshan data...")
   try:
       # Use the loader to get the processed frame
       # It handles raw data fetching, feature selection, nan ops,
       # encoding, and scaling internally based on defaults for zhongshan
       df_processed = load_processed_subsidence_data(
           dataset_name='zhongshan',
           return_sequences=False, # We want the processed DataFrame first
           as_frame=True,
           scaler_type='minmax', # Match paper example script
           use_processed_cache=True, # Use cache if available
           save_processed_frame=True, # Save if reprocessed
           cache_suffix="_paper_proc", # Suffix for this specific processing
           verbose=True
       )
       print("\nZhongshan data loaded and processed.")
       df_processed.info()

   except FileNotFoundError:
       print("\nERROR: Raw data file 'zhongshan_2000.csv' not found and"
             " could not be downloaded. Cannot proceed.")
       # Handle error appropriately in a real script
       raise
   except Exception as e:
       print(f"\nERROR during data loading/processing: {e}")
       raise


Step 3: Define Feature Sets for Sequencing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Based on the columns present *after* preprocessing (including the new
one-hot encoded columns), we define the lists required by
:func:`~fusionlab.nn.utils.reshape_xtft_data`.

.. code-block:: python
   :linenos:

   target_col = 'subsidence'
   dt_col = 'year' # Time index column
   spatial_cols = ['longitude', 'latitude'] # Grouping and also static

   # Identify encoded columns automatically (example)
   encoded_cols = [c for c in df_processed.columns if
                   c.startswith('geology_') or c.startswith('density_tier_')]

   # Define final feature sets
   final_static_cols = list(spatial_cols) + encoded_cols
   # Numerical columns from paper (excluding target, year) after scaling
   final_dynamic_cols = [
       'GWL', 'rainfall_mm', 'normalized_density',
       'normalized_seismic_risk_score'
       ]
   # Future columns from paper
   final_future_cols = ['rainfall_mm']

   # Verify all defined columns exist in the processed dataframe
   all_needed_cols = (
        [dt_col, target_col] + spatial_cols + final_static_cols +
        final_dynamic_cols + final_future_cols
   )
   missing_cols = [c for c in set(all_needed_cols) if c not in df_processed.columns]
   if missing_cols:
       raise ValueError(f"Columns required for sequencing missing from"
                        f" processed data: {missing_cols}")

   print("\nFeature sets defined for sequence reshaping.")
   print(f"  Static : {final_static_cols}")
   print(f"  Dynamic: {final_dynamic_cols}")
   print(f"  Future : {final_future_cols}")


Step 4: Generate Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, use :func:`~fusionlab.nn.utils.reshape_xtft_data` with the
processed DataFrame and defined column lists to create the sequence
arrays. We use `time_steps=4` and `forecast_horizon=4` based on the
paper's example script configuration.

.. code-block:: python
   :linenos:

   time_steps = 4          # Lookback window from script
   forecast_horizon = 4    # Prediction horizon from script (2023-2026)

   print(f"\nReshaping data (T={time_steps}, H={forecast_horizon})...")
   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df_processed,
       dt_col=dt_col,
       target_col=target_col,
       static_cols=final_static_cols,
       dynamic_cols=final_dynamic_cols,
       future_cols=final_future_cols,
       spatial_cols=spatial_cols,
       time_steps=time_steps,
       forecast_horizons=forecast_horizon,
       verbose=1
   )


Step 5: Data Splitting for Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Split the generated sequences into training and validation sets. The
paper used 2015-2022 for training and 2023 for testing. Here, we split
the generated *sequences* (which implicitly capture this time split if
`reshape_xtft_data` processed chronologically per group) using a
standard ratio (e.g., 80/20).

.. code-block:: python
   :linenos:

   # Split sequences (example: 80% train, 20% validation)
   X_static_train, X_static_val, \
   X_dynamic_train, X_dynamic_val, \
   X_future_train, X_future_val, \
   y_train, y_val = train_test_split(
       static_data, dynamic_data, future_data, target_data,
       test_size=0.2, # 20% for validation
       random_state=42 # For reproducible split
   )

   # Package inputs into lists
   train_inputs = [X_static_train, X_dynamic_train, X_train_future]
   val_inputs = [X_static_val, X_dynamic_val, X_val_future]

   print("\nSequence data split into Train/Validation sets:")
   print(f"  Train sequences: {len(y_train)}")
   print(f"  Validation sequences: {len(y_val)}")


Step 6: Define and Train XTFT Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate the :class:`~fusionlab.nn.XTFT` model (note: the script used
`SuperXTFT`, but we use `XTFT` as requested). Configure it for quantile
forecasting using hyperparameters potentially derived from tuning or
the paper's example. Compile with quantile loss and train using `.fit()`.

.. code-block:: python
   :linenos:

   quantiles = [0.1, 0.5, 0.9]
   output_dim = 1 # Predicting 'subsidence'

   # Example Hyperparameters (adjust based on tuning/paper)
   best_params = {
       'embed_dim': 32, 'max_window_size': time_steps, 'memory_size': 100,
       'num_heads': 4, 'dropout_rate': 0.1, 'lstm_units': 64,
       'attention_units': 64, 'hidden_units': 32, 'multi_scale_agg': 'auto',
   }

   # Instantiate XTFT model
   xtft_model = XTFT(
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1],
       forecast_horizon=forecast_horizon,
       quantiles=quantiles,
       output_dim=output_dim,
       **best_params
   )
   print("\nXTFT model instantiated.")

   # Compile with quantile loss
   loss_fn = combined_quantile_loss(quantiles=quantiles)
   xtft_model.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
       loss=loss_fn
   )
   print("Model compiled.")

   # Define Callbacks
   early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                                  restore_best_weights=True)
   model_checkpoint_path = os.path.join(
       output_dir_zhongshan, 'xtft_zhongshan_best.keras' # Use .keras format
       )
   model_checkpoint = ModelCheckpoint(
       model_checkpoint_path, monitor='val_loss', save_best_only=True,
       save_weights_only=False, verbose=1
   )

   # Train the model
   print("Starting model training...")
   history = xtft_model.fit(
       train_inputs,
       y_train,
       validation_data=(val_inputs, y_val),
       epochs=5, # Increase significantly for real training (e.g., 50)
       batch_size=32,
       callbacks=[early_stopping, model_checkpoint],
       verbose=1
   )
   print("Training finished.")

   # Optional: Load the best model saved by checkpoint
   # try:
   #     print("Loading best model from checkpoint...")
   #     best_xtft_model = tf.keras.models.load_model(
   #         model_checkpoint_path,
   #         custom_objects={"combined_quantile_loss": loss_fn} # Pass loss
   #     )
   # except Exception as e:
   #     print(f"Could not load checkpoint, using last model state. Error: {e}")
   #     best_xtft_model = xtft_model # Fallback to last state


Step 7: Generate Forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use the trained model and the :func:`~fusionlab.nn.utils.generate_forecast`
utility to generate predictions for future years (e.g., 2023-2026, matching
the `forecast_horizon`). This function handles preparing the necessary
input sequences from the end of the training data. We also provide the
hold-out test data (actual 2023 data) for evaluation within the function.

.. code-block:: python
   :linenos:

   # Need the processed DataFrame BEFORE sequencing for generate_forecast
   # Also need the original unprocessed test data for evaluation comparison
   # Re-load/split original data if not kept from Step 2 of the script
   # For simplicity, assume df_processed contains data up to 2022
   # and we have a separate test_df_raw for 2023 actuals.

   # Placeholder for actual 2023 test data (load this properly)
   # test_data = df_raw[df_raw['year'] == 2023].copy()
   # Ensure test_data has same columns as needed for eval inside generate_forecast
   # For demonstration, we'll use the validation set derived from sequences
   # NOTE: generate_forecast expects the *training* data to find the last sequence
   # and test data for evaluation. Need df_processed up to end of training.
   df_processed_train = df_processed[df_processed[dt_col] <= 2022] # Example filter

   print("\nGenerating forecast using generate_forecast...")
   # Define the forecast years explicitly
   forecast_years = [2023, 2024, 2025, 2026] # Matches horizon=4

   # Use generate_forecast
   forecast_df = generate_forecast(
       xtft_model=xtft_model, # Use the trained model
       train_data=df_processed_train, # Data model was trained on (for last seq)
       dt_col=dt_col,
       time_steps=time_steps,
       forecast_horizon=forecast_horizon,
       static_features=final_static_cols,
       dynamic_features=final_dynamic_cols,
       future_features=final_future_cols,
       spatial_cols=spatial_cols,
       # test_data=test_data, # Provide actual 2023 data for evaluation
       mode="quantile",
       q=quantiles, # Pass the quantiles used
       tname=target_col,
       forecast_dt=forecast_years, # Specify exact output years
       scaler=scaler_n, # Pass the scaler used for numerical cols
       num_cols=cols_to_scale_n, # List of cols that were scaled
       savefile=os.path.join(output_dir_zhongshan, "zhongshan_forecast.csv"),
       verbose=1
   )

   print("\nForecast DataFrame head:")
   print(forecast_df.head())


Step 8: Visualize Forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use the :func:`~fusionlab.nn.utils.visualize_forecasts` utility to plot
the spatial distribution of actual (if available) and predicted
subsidence for specific forecast periods (years).

.. code-block:: python
   :linenos:

   # Need actual test data (e.g., for 2023) for comparison plots
   # Assuming test_data DataFrame for 2023 exists from script's Step 6
   # test_data = df_raw[df_raw['year'] == 2023] # Example

   print("\nGenerating forecast visualization...")
   try:
       # Visualize forecast for 2023 (requires actual 2023 data in test_data)
       # Create dummy test_data if needed for visualization code to run
       if 'test_data' not in locals():
            warnings.warn("Actual test_data for 2023 not available,"
                          " visualization will only show predictions.")
            test_data_vis = None
       else:
           test_data_vis = test_data

       visualize_forecasts(
           forecast_df=forecast_df,
           test_data=test_data_vis, # Provide actuals if available
           dt_col=dt_col,
           tname=target_col,
           eval_periods=[2023], # Evaluate/plot only 2023
           mode="quantile", # Match forecast mode
           kind="spatial",
           x="longitude", # Coordinate columns
           y="latitude",
           verbose=1
       )
   except Exception as e:
       print(f"Visualization failed. Error: {e}")