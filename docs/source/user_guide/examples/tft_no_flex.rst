.. _example_tft_no_flex:

===========================================
TFT Forecasting (Required Inputs Version)
===========================================

This example demonstrates how to use the revised
:class:`~fusionlab.nn.transformers.TFT` class implementation. Unlike
the potentially more flexible `TemporalFusionTransformer`, this version
strictly requires **static**, **dynamic (past)**, and **known future**
features as inputs during initialization and calls.

We will show how to:
1. Generate synthetic data including all three feature types.
2. Use the :func:`~fusionlab.nn.utils.reshape_xtft_data` utility
   to prepare the required input arrays.
3. Define, compile, and train the revised `TFT` model for point
   forecasting.
4. Make predictions using the mandatory three-part input structure.

Code Example
------------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import train_test_split

   # Assuming fusionlab components are importable
   # *** Import the standard TFT class ***
   from fusionlab.nn.transformers import TFT 
   from fusionlab.nn.utils import reshape_xtft_data
   # Loss not needed explicitly for point forecast compilation with 'mse'

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # 1. Generate Synthetic Data (Static, Dynamic, Future)
   # ----------------------------------------------------
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
   print("Generated data shape:", df.shape)

   # 2. Define Features & Scale
   # --------------------------
   target_col = 'Sales'
   dt_col = 'Date'
   static_cols = ['ItemID', 'Category'] # Static features
   dynamic_cols = ['Month', 'PrevMonthSales'] # Dynamic past features
   future_cols = ['PlannedEvent', 'Month'] # Known future features
   spatial_cols = ['ItemID'] # For grouping

   # Scale numerical features (excluding IDs/Month/Binary Event)
   scalers = {}
   num_cols_to_scale = ['PrevMonthSales', 'Sales'] # Scale lag and target
   for col in num_cols_to_scale:
       scaler = StandardScaler()
       df[col] = scaler.fit_transform(df[[col]])
       scalers[col] = scaler
   print("Numerical features scaled.")

   # 3. Prepare Sequences using reshape_xtft_data
   # ----------------------------------------------
   time_steps = 12 # Lookback window
   forecast_horizons = 3 # Predict next 3 months

   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df, dt_col=dt_col, target_col=target_col,
       dynamic_cols=dynamic_cols, static_cols=static_cols,
       future_cols=future_cols, spatial_cols=spatial_cols,
       time_steps=time_steps, forecast_horizons=forecast_horizons,
       verbose=0
   )
   print(f"\nReshaped Data Shapes:")
   print(f"  Static : {static_data.shape}")    # (Samples, NumStatic)
   print(f"  Dynamic: {dynamic_data.shape}")   # (Samples, T, NumDynamic)
   print(f"  Future : {future_data.shape}")    # (Samples, H, NumFuture) -> Note: reshape_xtft_data output needs checking
   print(f"  Target : {target_data.shape}")    # (Samples, H, 1)

   # Note: Ensure the output shapes from reshape_xtft_data match the
   #       expectations of the revised TFT. Adjust reshape_xtft_data
   #       or add reshaping steps here if needed (e.g., for future data length).
   # Assuming here future_data has shape (Samples, forecast_horizons, NumFuture)
   # and dynamic_data has shape (Samples, time_steps, NumDynamic)
   # The internal TFT call might need adjustment for time dimension concat

   # 4. Train/Validation Split
   # ---------------------------
   val_split_fraction = 0.2
   n_samples = static_data.shape[0]
   split_idx = int(n_samples * (1 - val_split_fraction))

   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   X_train_future, X_val_future = future_data[:split_idx], future_data[split_idx:]
   y_train, y_val = target_data[:split_idx], target_data[split_idx:]

   # Package inputs as the REQUIRED list [static, dynamic, future]
   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future]
   print("Data prepared and split.")

   # 5. Define Revised TFT Model for Point Forecast
   # -----------------------------------------------
   model = TFT( # Using the revised TFT class
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1],
       forecast_horizon=forecast_horizons,
       hidden_units=16, # Smaller for demo
       num_heads=2,
       num_lstm_layers=1,
       quantiles=None # Point forecast
   )
   print("Revised TFT model instantiated.")

   # 6. Compile the Model
   # --------------------
   model.compile(optimizer='adam', loss='mse')
   print("Model compiled successfully.")

   # 7. Train the Model
   # ------------------
   print("Starting model training (few epochs for demo)...")
   history = model.fit(
       train_inputs, # Pass the list [static, dynamic, future]
       y_train,
       validation_data=(val_inputs, y_val),
       epochs=5,
       batch_size=16,
       verbose=0
   )
   print("Training finished.")
   print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

   # 8. Make a Prediction
   # --------------------
   print("Making prediction on validation set...")
   predictions_scaled = model.predict(val_inputs, verbose=0)
   print("Prediction output shape:", predictions_scaled.shape)
   # Expected: (Batch, Horizon, OutputDim=1) -> (N_val, 3, 1)

   # 9. Inverse Transform & Visualize (Optional)
   # -----------------------------------------
   # (Add inverse transform using saved scalers['Sales'] and plotting
   #  similar to previous examples if desired)


.. topic:: Explanations

   1.  **Data Generation:** We create a more structured dataset with
       multiple items (`ItemID`), including a static feature (`Category`),
       dynamic features (`Month`, `PrevMonthSales`), and a known future
       feature (`PlannedEvent`).
   2.  **Feature Definition:** Explicit lists define the role of each
       column (`static_cols`, `dynamic_cols`, `future_cols`, `target_col`,
       `dt_col`, `spatial_cols`).
   3.  **Scaling:** Numerical features used in dynamic inputs or as the
       target are scaled using `StandardScaler`. The scalers should be
       saved for inverse transformation during prediction. Static identifiers
       or features used as categories (like `ItemID`, `Month`) are usually not
       scaled.
   4.  **Sequence Preparation:** :func:`~fusionlab.utils.ts_utils.reshape_xtft_data`
       is used. This utility is specifically designed to handle static,
       dynamic, and future features, grouping by `spatial_cols` if needed,
       and creating the correctly structured NumPy arrays (`static_data`,
       `dynamic_data`, `future_data`, `target_data`) required by models
       like TFT and XTFT. *(Note: Verify that the time dimension of the
       `future_data` array output by this function aligns with the expectations
       of the TFT model's internal `call` method, especially regarding how
       past dynamic and future features are combined before the LSTM).*
   5.  **Train/Validation Split:** The generated sequence arrays are split
       chronologically. The inputs for `fit` and `predict` are packaged
       into a list `[static, dynamic, future]` in that specific order.
   6.  **Model Definition:** We instantiate the revised
       :class:`~fusionlab.nn.transformers.TFT` class. Crucially, we **must**
       provide valid integer dimensions for `static_input_dim`,
       `dynamic_input_dim`, and `future_input_dim`. We set `quantiles=None`
       to specify point forecasting.
   7.  **Model Compilation:** The model is compiled with 'adam' optimizer and
       'mse' loss, appropriate for point forecasting.
   8.  **Model Training:** The `.fit()` method is called with the **required
       three-element input list** `train_inputs`.
   9.  **Prediction:** Similarly, `.predict()` is called with the three-element
       `val_inputs` list. The output shape reflects the batch size, forecast
       horizon, and single output dimension for the point forecast.

This example highlights how to use the stricter `TFT` implementation when
you have all three types of features and prefer explicit input requirements,
leveraging `reshape_xtft_data` for convenient data preparation.