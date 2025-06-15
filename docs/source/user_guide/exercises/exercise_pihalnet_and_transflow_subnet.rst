.. _exercise_pihalnet_transflow_subsnet_guide:

=========================================================================
Exercise: Physics-Informed Forecasting with PIHALNet & TransFlowSubsNet 
=========================================================================

Welcome to this comprehensive exercise on using the ``PIHALNet`` and
``TransFlowSubsNet`` models. This guide will walk you through a
complete, real-world workflow for performing a multi-horizon,
probabilistic, physics-informed forecast for land subsidence and
groundwater level.

This is a deep-dive tutorial that covers every step from raw data
ingestion to final visualization, demonstrating how to handle the
complex data requirements of these advanced hybrid models.

**Learning Objectives:**

* Load and preprocess a complex, real-world style dataset.
* Perform feature engineering, including one-hot encoding and numerical
    scaling.
* Use the ``prepare_pinn_data_sequences`` utility to transform a flat
    DataFrame into the specialized sequence format required by PINNs.
* Configure and train a hybrid model (``PIHALNet`` or ``TransFlowSubsNet``)
    with both a data-driven loss and a physics-informed loss component.
* Set up callbacks for model checkpointing and early stopping.
* Generate predictions on new data and format them into an
    interpretable DataFrame.
* Visualize the multi-target, multi-horizon forecast results using
    the library's specialized plotting functions.

Let's get started!

Prerequisites
-------------
Ensure you have ``fusionlab-learn`` and its dependencies installed.

.. code-block:: bash

   pip install fusionlab-learn "keras-tuner>=1.4.0" matplotlib scikit-learn

---

Step 0: Preamble & Configuration
-----------------------------------------
First, we import all necessary libraries and define the main
configuration parameters for our experiment. This central config block
allows you to easily change the model, forecast period, or key
hyperparameters for your own tests.

.. code-block:: python
   :linenos:

   import os
   import pandas as pd
   import numpy as np
   import tensorflow as tf
   from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
   from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
   
   # --- FusionLab Imports ---
   from fusionlab.datasets import fetch_zhongshan_data
   from fusionlab.nn.pinn.models import PIHALNet, TransFlowSubsNet
   from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences
   from fusionlab.nn.losses import combined_quantile_loss
   from fusionlab.nn.models.utils import plot_history_in
   from fusionlab.nn.pinn.utils import format_pihalnet_predictions
   from fusionlab.plot.forecast import plot_forecasts
   
   # --- Main Configuration ---
   MODEL_NAME = 'TransFlowSubsNet'  # or 'PIHALNet'
   TRAIN_END_YEAR = 2022
   FORECAST_START_YEAR = 2023
   FORECAST_HORIZON_YEARS = 3
   TIME_STEPS = 5  # Lookback window
   QUANTILES = [0.1, 0.5, 0.9]
   EPOCHS = 20 # Use more epochs (e.g., 100+) for real results
   BATCH_SIZE = 256
   
   # --- PINN Configuration ---
   PDE_MODE_CONFIG = 'both' if MODEL_NAME == 'TransFlowSubsNet' else 'consolidation'
   LAMBDA_PDE_CONS = 1.0 # Weight for consolidation loss
   LAMBDA_PDE_GW = 1.0   # Weight for groundwater flow loss

   # --- Output Directory ---
   RUN_OUTPUT_PATH = f"./pihalnet_exercise_outputs/{MODEL_NAME}_run"
   os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)
   print(f"Configuration set for model: {MODEL_NAME}")
   print(f"Output artifacts will be saved in: {RUN_OUTPUT_PATH}")

Step 1: Load and Inspect the Dataset
-----------------------------------------
We will load the Zhongshan dataset. The logic first tries to find a
local, larger version of the file but will fall back to fetching the
smaller, built-in version if it's not found.

.. code-block:: python
   :linenos:

   try:
       # For this exercise, we directly use the fetch utility
       print("Fetching Zhongshan dataset...")
       data_bunch = fetch_zhongshan_data()
       df_raw = data_bunch.frame
       print(f"Successfully loaded dataset. Shape: {df_raw.shape}")
   except Exception as e:
       raise RuntimeError(f"Failed to fetch dataset: {e}")
   
   print(df_raw.head())

Step 2: Preprocessing - Feature Selection & Cleaning
-------------------------------------------------------------
We select the features relevant to our model and handle any missing
values. For these models, we need coordinates (`longitude`, `latitude`),
a time column (`year`), targets (`subsidence`, `GWL`), and other
covariates.

.. code-block:: python
   :linenos:

   from fusionlab.utils.data_utils import nan_ops

   TIME_COL, LON_COL, LAT_COL = 'year', 'longitude', 'latitude'
   SUBSIDENCE_COL, GWL_COL = 'subsidence', 'GWL'

   # Select relevant features
   features_to_use = [
       LON_COL, LAT_COL, TIME_COL, SUBSIDENCE_COL, GWL_COL,
       'rainfall_mm', 'geology', 'normalized_density'
   ]
   df_selected = df_raw[features_to_use].copy()
   df_cleaned = nan_ops(df_selected, ops='sanitize', action='fill')
   print(f"NaNs after cleaning: {df_cleaned.isna().sum().sum()}")

Step 3: Preprocessing - Encoding & Scaling
-----------------------------------------------
Next, we convert categorical features like `geology` into a numerical
format using one-hot encoding and scale all numerical features to a
common range (0-1) using `MinMaxScaler` for stable training.

.. code-block:: python
   :linenos:

   # --- Encoding Categorical Features ---
   ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
   encoded_data = ohe.fit_transform(df_cleaned[['geology']])
   encoded_cols = ohe.get_feature_names_out(['geology'])
   df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_cleaned.index)
   df_processed = pd.concat([df_cleaned.drop('geology', axis=1), df_encoded], axis=1)

   # --- Create a numeric time coordinate for the PINN ---
   TIME_COL_NUMERIC = "time_numeric"
   df_processed[TIME_COL_NUMERIC] = df_processed[TIME_COL] - df_processed[TIME_COL].min()

   # --- Scaling Numerical Features ---
   cols_to_scale = [c for c in df_processed.columns if c != TIME_COL]
   scaler = MinMaxScaler()
   df_scaled = df_processed.copy()
   df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
   print("Categorical features encoded and numerical features scaled.")
   print(f"Final processed data shape: {df_scaled.shape}")

Step 4: Define Feature Sets & Generate Sequences
-------------------------------------------------------
This is a critical step where we tell our data preparation utility,
``prepare_pinn_data_sequences``, which columns to use for which input
stream (static, dynamic, etc.) and then generate the sequences.

.. code-block:: python
   :linenos:

   # 1. Split data into training and testing periods
   df_train_master = df_scaled[df_scaled[TIME_COL] <= TRAIN_END_YEAR]
   df_test_master = df_scaled[df_scaled[TIME_COL] >= TRAIN_END_YEAR]

   # 2. Define feature lists
   static_features_list = encoded_cols.tolist()
   dynamic_features_list = [GWL_COL, 'rainfall_mm', 'normalized_density']
   future_features_list = ['rainfall_mm'] # Assume rainfall forecast is known

   # 3. Generate sequences
   inputs_train, targets_train, _ = prepare_pinn_data_sequences(
       df=df_train_master,
       time_col=TIME_COL_NUMERIC,
       lon_col=LON_COL, lat_col=LAT_COL,
       subsidence_col=SUBSIDENCE_COL, gwl_col=GWL_COL,
       dynamic_cols=dynamic_features_list,
       static_cols=static_features_list,
       future_cols=future_features_list,
       group_id_cols=[LON_COL, LAT_COL],
       time_steps=TIME_STEPS,
       forecast_horizon=FORECAST_HORIZON_YEARS,
   )
   print("\nTraining sequences generated.")
   for name, arr in inputs_train.items():
       print(f"  Train Input '{name}' shape: {arr.shape if arr is not None else 'None'}")

Step 5: Create tf.data.Dataset
-----------------------------------------
We convert our NumPy sequence arrays into ``tf.data.Dataset`` objects
for efficient, high-performance training with TensorFlow.

.. code-block:: python
   :linenos:

   # Standardize target keys to match model output names
   targets_train_std = {
       'subs_pred': targets_train['subsidence'],
       'gwl_pred': targets_train['gwl']
   }
   
   # Create the full dataset
   full_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train_std))
   
   # Create train/validation split
   total_size = len(inputs_train['coords'])
   val_size = int(0.2 * total_size)
   train_size = total_size - val_size
   
   train_dataset = full_dataset.take(train_size).shuffle(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
   val_dataset = full_dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
   
   print(f"\nCreated training dataset ({train_size} samples) and validation dataset ({val_size} samples).")

Step 6: Model Training
----------------------------
We now instantiate our chosen model, compile it with our composite loss
function, and begin training.

.. code-block:: python
   :linenos:

   # 1. Instantiate Model
   ModelClass = TransFlowSubsNet if MODEL_NAME == 'TransFlowSubsNet' else PIHALNet
   model = ModelClass(
       static_input_dim=inputs_train['static_features'].shape[-1],
       dynamic_input_dim=inputs_train['dynamic_features'].shape[-1],
       future_input_dim=inputs_train['future_features'].shape[-1],
       output_subsidence_dim=1, output_gwl_dim=1,
       forecast_horizon=FORECAST_HORIZON_YEARS,
       max_window_size=TIME_STEPS,
       quantiles=QUANTILES,
       pde_mode=PDE_MODE_CONFIG
   )

   # 2. Define Losses and Weights
   loss_dict = {'subs_pred': 'mse', 'gwl_pred': 'mse'}
   if QUANTILES:
       loss_dict = {k: combined_quantile_loss(QUANTILES) for k in loss_dict}
   
   physics_loss_weights = {"lambda_cons": LAMBDA_PDE_CONS, "lambda_gw": LAMBDA_PDE_GW} \
       if MODEL_NAME == 'TransFlowSubsNet' else {"lambda_physics": LAMBDA_PDE_CONFIG}

   # 3. Compile Model
   model.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
       loss=loss_dict,
       loss_weights={'subs_pred': 1.0, 'gwl_pred': 0.5},
       **physics_loss_weights
   )

   # 4. Train
   print(f"\nStarting {MODEL_NAME} training...")
   history = model.fit(
       train_dataset,
       validation_data=val_dataset,
       epochs=EPOCHS,
       callbacks=[EarlyStopping('val_loss', patience=10, restore_best_weights=True)]
   )

Step 7: Visualize Results
-----------------------------
Finally, we can use the plotting utilities to visualize the training
history and the forecast results.

.. code-block:: python
   :linenos:

   # Plot training history
   plot_history_in(history, title=f'{MODEL_NAME} Training History')

   # Generate and plot forecasts on the validation set
   val_inputs_batch, val_targets_batch = next(iter(val_dataset))
   predictions = model.predict(val_inputs_batch)

   df_forecast = format_pihalnet_predictions(
       pihalnet_outputs=predictions,
       y_true_dict=val_targets_batch,
       target_mapping={'subs_pred': SUBSIDENCE_COL, 'gwl_pred': GWL_COL},
       quantiles=QUANTILES,
       forecast_horizon=FORECAST_HORIZON_YEARS,
       model_inputs=val_inputs_batch,
   )

   if not df_forecast.empty:
       plot_forecasts(
           df_forecast,
           target_name=SUBSIDENCE_COL,
           quantiles=QUANTILES,
           kind="temporal",
           sample_ids="first_n",
           num_samples=4
       )

Discussion of Exercise
--------------------------
Congratulations! You have completed a full end-to-end workflow for a
complex, hybrid physics-informed model. You have learned how to:

* Take raw, tabular data and perform all necessary preprocessing steps.
* Use the specialized ``prepare_pinn_data_sequences`` utility to create
    the multi-part input required by the models.
* Configure, compile, and train a ``PIHALNet`` or ``TransFlowSubsNet``
    model with a composite loss function.
* Evaluate the training process and visualize the final probabilistic
    forecasts.

This comprehensive process is a powerful template for applying these
advanced models to your own scientific machine learning challenges.