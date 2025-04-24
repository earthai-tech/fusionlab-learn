.. _example_tft_no_flex:

===========================================
TFT Forecasting (Required Inputs Version)
===========================================

This example demonstrates how to use the revised
:class:`~fusionlab.nn.transformers.TFT` class implementation. Unlike
the potentially more flexible `TemporalFusionTransformer`, this version
strictly requires **static**, **dynamic (past)**, and **known future**
features as inputs during initialization and calls.

We will walk through the following steps for point forecasting:

1.  Imports and Setup.
2.  Generate synthetic data including all three feature types.
3.  Define feature roles and scale numerical features.
4.  Use :func:`~fusionlab.nn.utils.reshape_xtft_data` to prepare the
    required input arrays.
5.  Split data and package inputs for the model.
6.  Define the stricter `TFT` model.
7.  Compile the model.
8.  Train the model using the required three-part input list.
9.  Make predictions.
10. (Optional) Inverse transform and visualize.

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~
Import necessary libraries: TensorFlow, Pandas, NumPy, Scikit-learn's
`StandardScaler`, and the required ``fusionlab`` components (`TFT`,
`reshape_xtft_data`). We also suppress warnings for clarity.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import StandardScaler
   # from sklearn.model_selection import train_test_split # Not used directly
   import os
   import joblib # For saving scalers

   # Import the stricter TFT class and the appropriate reshape util
   from fusionlab.nn.transformers import TFT # The revised class
   from fusionlab.nn.utils import reshape_xtft_data
   # Loss not needed explicitly for point forecast compilation with 'mse'

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # Setup output directory (optional)
   output_dir = "./tft_required_output"
   os.makedirs(output_dir, exist_ok=True)
   print("Setup complete.")


Step 2: Generate Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We create a synthetic dataset simulating monthly sales for multiple items.
This includes static features (`ItemID`, `Category`), dynamic features
(`Month`, `PrevMonthSales`), a known future feature (`PlannedEvent`),
and the target `Sales`.

.. code-block:: python
   :linenos:

   n_items = 2
   n_timesteps = 48 # 4 years of monthly data
   date_rng = pd.date_range(start='2018-01-01', periods=n_timesteps, freq='MS')
   df_list = []

   for item_id in range(n_items):
       time = np.arange(n_timesteps)
       # Base sales with trend, seasonality, and noise
       sales = (
           50 + item_id * 20 + time * (1.5 + item_id * 0.5) +
           15 * np.sin(2 * np.pi * time / 12) +
           np.random.normal(0, 5, n_timesteps)
       )
       # Static feature: Item Category (numerical representation)
       category = item_id + 1
       # Dynamic feature: Month
       month = date_rng.month
       # Future feature: Planned Event (binary)
       event = np.random.randint(0, 2, n_timesteps)

       item_df = pd.DataFrame({
           'Date': date_rng, 'ItemID': item_id, 'Category': category,
           'Month': month, 'PlannedEvent': event, 'Sales': sales
       })
       # Add lagged sales as another dynamic feature
       item_df['PrevMonthSales'] = item_df['Sales'].shift(1)
       df_list.append(item_df)

   df = pd.concat(df_list).dropna().reset_index(drop=True)
   print(f"Generated data shape: {df.shape}")
   print("Sample data:")
   print(df.head())


Step 3: Define Features & Scale Numerics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assign columns to their roles (static, dynamic, future, target, etc.)
and apply scaling (here, `StandardScaler`) to the numerical columns
that will be used as inputs or targets for the model. Saving the scaler
is important for later inverse transformation.

.. code-block:: python
   :linenos:

   target_col = 'Sales'
   dt_col = 'Date'
   static_cols = ['ItemID', 'Category'] # Static features
   dynamic_cols = ['Month', 'PrevMonthSales'] # Dynamic past features
   future_cols = ['PlannedEvent', 'Month'] # Known future features
   spatial_cols = ['ItemID'] # For grouping by item

   # Scale numerical features (excluding IDs/Month/Binary Event)
   scalers = {}
   num_cols_to_scale = ['PrevMonthSales', 'Sales'] # Scale lag and target
   for col in num_cols_to_scale:
       scaler = StandardScaler()
       df[col] = scaler.fit_transform(df[[col]])
       scalers[col] = scaler
       print(f"Scaled column: {col}")

   # Save the scaler (using joblib)
   scaler_path = os.path.join(output_dir, "tft_scaler.joblib")
   joblib.dump(scalers, scaler_path)
   print(f"Scalers saved to {scaler_path}")


Step 4: Prepare Sequences using `reshape_xtft_data`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use the :func:`~fusionlab.nn.utils.reshape_xtft_data` utility. This
function takes the processed DataFrame and creates the rolling window
sequences, automatically separating features into the static, dynamic,
future, and target arrays based on the provided column lists.

.. code-block:: python
   :linenos:

   time_steps = 12         # 1 year lookback
   forecast_horizons = 3   # Predict next 3 months (multi-step point)

   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df, # Use scaled data
       dt_col=dt_col,
       target_col=target_col,
       dynamic_cols=dynamic_cols,
       static_cols=static_cols,
       future_cols=future_cols,
       spatial_cols=spatial_cols,
       time_steps=time_steps,
       forecast_horizons=forecast_horizons,
       verbose=1 # Show resulting shapes
   )
   # Target data needs shape (Samples, Horizon) for MSE loss if O=1
   targets = target_data.reshape(-1, forecast_horizons)


Step 5: Train/Validation Split & Input Packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Split the generated sequence arrays chronologically into training and
validation sets. Then, package the input arrays into a list in the
specific order required by this `TFT` class: `[static, dynamic, future]`.

.. code-block:: python
   :linenos:

   val_split_fraction = 0.2
   n_samples = static_data.shape[0]
   split_idx = int(n_samples * (1 - val_split_fraction))

   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   X_train_future, X_val_future = future_data[:split_idx], future_data[split_idx:]
   y_train, y_val = targets[:split_idx], targets[split_idx:]

   # Package inputs in the REQUIRED list order [static, dynamic, future]
   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future]

   print("\nData prepared and split into Train/Validation.")
   print(f"  Train samples: {X_train_static.shape[0]}")
   print(f"  Validation samples: {X_val_static.shape[0]}")


Step 6: Define Required-Inputs TFT Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate the revised :class:`~fusionlab.nn.transformers.TFT` class.
All three input dimensions (`static_input_dim`, `dynamic_input_dim`,
`future_input_dim`) must be provided. We set `quantiles=None` for
point forecasting.

.. code-block:: python
   :linenos:

   model = TFT( # Using the revised TFT class
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1], # Must provide all dims
       forecast_horizon=forecast_horizons,
       hidden_units=16, # Smaller for demo
       num_heads=2,
       num_lstm_layers=1,
       quantiles=None # Point forecast
   )
   print("\nRequired-Inputs TFT model instantiated.")
   # model.summary() # Call after build


Step 7: Compile the Model
~~~~~~~~~~~~~~~~~~~~~~~~~
Compile the model using an optimizer and Mean Squared Error ('mse')
loss, suitable for point forecasting.

.. code-block:: python
   :linenos:

   model.compile(optimizer='adam', loss='mse')
   print("Model compiled successfully with MSE loss.")


Step 8: Train the Model
~~~~~~~~~~~~~~~~~~~~~~~
Train the model using the `.fit()` method. Pass the packaged 3-element
list `train_inputs` as `x`.

.. code-block:: python
   :linenos:

   print("Starting model training (few epochs for demo)...")
   history = model.fit(
       train_inputs, # Pass the list [static, dynamic, future]
       y_train,      # Target shape (Samples, Horizon) or (Samples, H, O=1)
       validation_data=(val_inputs, y_val),
       epochs=5,       # Increase epochs for actual training
       batch_size=16,
       verbose=1       # Show progress
   )
   print("Training finished.")


Step 9: Make Predictions
~~~~~~~~~~~~~~~~~~~~~~~~
Use the trained model's `.predict()` method with the 3-element
validation input list to generate point forecasts.

.. code-block:: python
   :linenos:

   print("\nMaking prediction on validation set...")
   predictions_scaled = model.predict(val_inputs, verbose=0)
   print(f"Prediction output shape: {predictions_scaled.shape}")
   # Expected: (Batch, Horizon, OutputDim=1) -> (N_val, 3, 1)


Step 10: Inverse Transform & Visualize (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To interpret the results, inverse transform the scaled predictions
(and actuals) using the scaler saved in Step 3. Visualize the results
(e.g., actuals vs. predictions for one item).

.. code-block:: python
   :linenos:

   print("\nInverse transforming predictions...")
   # Reshape predictions and actuals for scaler
   num_val_samples = X_val_static.shape[0]
   output_dim = model.output_dim # Should be 1 here
   pred_reshaped = predictions_scaled.reshape(-1, output_dim)
   y_val_reshaped = y_val.reshape(-1, output_dim)

   # Load scaler and inverse transform
   # loaded_scalers = joblib.load(scaler_path)
   # scaler_target = loaded_scalers['Sales'] # Get the correct scaler
   scaler_target = scalers['Sales'] # Use scaler from memory in this example

   predictions_inv = scaler_target.inverse_transform(pred_reshaped)
   y_val_inv = scaler_target.inverse_transform(y_val_reshaped)

   # Reshape back to (Samples, Horizon)
   predictions_final = predictions_inv.reshape(num_val_samples, forecast_horizons)
   y_val_final = y_val_inv.reshape(num_val_samples, forecast_horizons)
   print("Predictions inverse transformed.")

   # --- Visualization (Example for one item) ---
   item_to_plot = 0
   item_mask_val = (X_val_static[:, 0] == item_to_plot)
   if np.sum(item_mask_val) > 0:
       first_val_seq_idx = np.where(item_mask_val)[0][0]
       actual_vals_item = y_val_final[first_val_seq_idx, :]
       pred_vals_item = predictions_final[first_val_seq_idx, :]

       # Create time axis (approximate)
       last_train_date_item = df[df['ItemID']==item_to_plot].iloc[split_idx + time_steps - 1]['Date']
       pred_time_axis = pd.date_range(
           last_train_date_item + pd.DateOffset(months=1),
           periods=forecast_horizons, freq='MS' # Use MS for monthly data
       )

       plt.figure(figsize=(10, 5))
       plt.plot(pred_time_axis, actual_vals_item, label='Actual Sales', marker='o', linestyle='--')
       plt.plot(pred_time_axis, pred_vals_item, label='Predicted Sales (Point)', marker='x')
       plt.title(f'TFT Point Forecast (ItemID {item_to_plot})')
       plt.xlabel('Date'); plt.ylabel('Sales (Inverse Scaled)'); plt.legend(); plt.grid(True)
       plt.show()