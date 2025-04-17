.. _example_advanced_xtft:

=============================
Advanced Forecasting with XTFT
=============================

This example demonstrates using the more advanced
:class:`~fusionlab.nn.XTFT` model for a multi-step quantile
forecasting task. XTFT is designed to handle complex scenarios involving
static features (e.g., item ID, location), dynamic historical features
(e.g., past sales, temperature), and known future inputs (e.g.,
planned promotions).

We will:
1. Generate synthetic multi-variate time series data for multiple items.
2. Define static, dynamic, future, and target features.
3. Use the :func:`~fusionlab.nn.utils.reshape_xtft_data` utility
   to prepare sequences suitable for XTFT.
4. Define, compile, and train an XTFT model with quantile outputs.
5. Make and visualize quantile predictions.

Code Example
------------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler

   from fusionlab.nn.transformers import XTFT
   from fusionlab.nn.utils import reshape_xtft_data
   from fusionlab.nn.losses import combined_quantile_loss

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # 1. Generate Synthetic Multi-Item Data
   # -------------------------------------
   n_items = 3
   n_timesteps = 36 # 3 years of monthly data
   date_rng = pd.date_range(start='2020-01-01', periods=n_timesteps, freq='MS')
   df_list = []

   for item_id in range(n_items):
       # Static feature: Item ID
       # Dynamic features: Month, Temperature, Lagged Sales
       # Future feature: Planned Promotion (binary)
       # Target: Sales
       time = np.arange(n_timesteps)
       # Base sales with trend and seasonality
       sales = (
           100 + item_id * 50 + time * (2 + item_id) +
           20 * np.sin(2 * np.pi * time / 12) +
           np.random.normal(0, 10, n_timesteps)
       )
       temp = 15 + 10 * np.sin(2 * np.pi * (time % 12) / 12 + np.pi) + np.random.normal(0, 2)
       promo = np.random.randint(0, 2, n_timesteps) # Future known promotions

       item_df = pd.DataFrame({
           'Date': date_rng,
           'ItemID': item_id,
           'Month': date_rng.month,
           'Temperature': temp,
           'PlannedPromotion': promo,
           'Sales': sales
       })
       # Create lagged sales (dynamic history)
       item_df['PrevMonthSales'] = item_df['Sales'].shift(1)
       df_list.append(item_df)

   df = pd.concat(df_list).dropna().reset_index(drop=True)
   print("Generated data shape:", df.shape)
   # print(df.head()) # Optional: view data

   # 2. Define Features
   # ------------------
   target_col = 'Sales'
   dt_col = 'Date'
   static_cols = ['ItemID'] # Could add more item attributes here
   # Past dynamic inputs (can include lags of target or other features)
   dynamic_cols = ['Month', 'Temperature', 'PrevMonthSales']
   # Known future inputs
   future_cols = ['PlannedPromotion', 'Month'] # Can reuse 'Month' if known ahead
   # Spatial/Grouping columns (used by reshape_xtft_data)
   spatial_cols = ['ItemID']

   # 3. Scale Numerical Features (excluding categoricals like ItemID, Month)
   # --------------------------------------------------------------------
   # Important: Scale before reshaping into sequences
   scalers = {}
   num_cols_to_scale = ['Temperature', 'PrevMonthSales', 'Sales'] # Only scale these
   for col in num_cols_to_scale:
       scaler = StandardScaler()
       df[col] = scaler.fit_transform(df[[col]])
       scalers[col] = scaler # Store scaler for inverse transform later
   print("Numerical features scaled.")

   # 4. Prepare Sequences using reshape_xtft_data
   # --------------------------------------------
   time_steps = 12 # Use 1 year of history
   forecast_horizons = 6 # Predict next 6 months

   # This utility handles grouping by spatial_cols and creating sequences
   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df,
       dt_col=dt_col,
       target_col=target_col,
       dynamic_cols=dynamic_cols,
       static_cols=static_cols,
       future_cols=future_cols,
       spatial_cols=spatial_cols, # Group data by ItemID
       time_steps=time_steps,
       forecast_horizons=forecast_horizons,
       verbose=0 # Keep output clean
   )

   print(f"\nReshaped Data Shapes:")
   print(f"  Static : {static_data.shape}")
   print(f"  Dynamic: {dynamic_data.shape}")
   print(f"  Future : {future_data.shape}")
   print(f"  Target : {target_data.shape}")
   # Expected e.g., Static: (NumSeq, NumStatic=1), Dynamic: (NumSeq, 12, NumDyn=3)
   # Future: (NumSeq, 12, NumFut=2), Target: (NumSeq, 6, 1)

   # 5. Train/Validation Split (Chronological within groups implicitly handled by reshape)
   # -----------------------------------------------------------------------------------
   # Simple split on the generated sequences (maintaining order)
   val_split_fraction = 0.2
   n_samples = static_data.shape[0]
   split_idx = int(n_samples * (1 - val_split_fraction))

   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   X_train_future, X_val_future = future_data[:split_idx], future_data[split_idx:]
   y_train, y_val = target_data[:split_idx], target_data[split_idx:]

   print(f"\nTrain Shapes: Static={X_train_static.shape}, Dynamic={X_train_dynamic.shape},"
         f" Future={X_train_future.shape}, Target={y_train.shape}")
   print(f"Val Shapes: Static={X_val_static.shape}, Dynamic={X_val_dynamic.shape},"
         f" Future={X_val_future.shape}, Target={y_val.shape}")

   # Package inputs as lists for the model
   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future]

   # 6. Define XTFT Model for Quantile Forecast
   # ------------------------------------------
   quantiles_to_predict = [0.1, 0.5, 0.9]

   model = XTFT(
       static_input_dim=static_data.shape[-1],   # Num static features
       dynamic_input_dim=dynamic_data.shape[-1], # Num dynamic features
       future_input_dim=future_data.shape[-1],   # Num future features
       forecast_horizon=forecast_horizons,
       quantiles=quantiles_to_predict,
       # Example XTFT Hyperparameters (adjust as needed)
       embed_dim=16,
       lstm_units=32,
       attention_units=16,
       hidden_units=32,
       num_heads=4,
       dropout_rate=0.1,
       max_window_size=time_steps, # Can be different from time_steps
       memory_size=50,
       # scales=[1, 6] # Optional: example multi-scale config
   )

   # 7. Compile the Model with Quantile Loss
   # ---------------------------------------
   loss_fn = combined_quantile_loss(quantiles=quantiles_to_predict)
   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=loss_fn)
   print("\nXTFT model compiled successfully.")

   # 8. Train the Model
   # ------------------
   print("Starting XTFT model training (few epochs for demo)...")
   history = model.fit(
       train_inputs,
       y_train,
       validation_data=(val_inputs, y_val),
       epochs=5, # Increase for real training
       batch_size=16, # Adjust based on memory
       verbose=0 # Suppress epoch progress
   )
   print("Training finished.")
   print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

   # 9. Make Predictions (Quantiles)
   # -------------------------------
   print("\nMaking predictions on validation set...")
   predictions_scaled = model.predict(val_inputs, verbose=0)
   print("Scaled prediction output shape:", predictions_scaled.shape)
   # Expected: (NumValSamples, Horizon, NumQuantiles) -> (e.g., N, 6, 3)

   # 10. Inverse Transform Predictions
   # ---------------------------------
   # Reshape for scaler: (Samples*Horizon, Quantiles)
   pred_reshaped = predictions_scaled.reshape(-1, len(quantiles_to_predict))
   predictions_inv = scalers['Sales'].inverse_transform(pred_reshaped)
   # Reshape back: (Samples, Horizon, Quantiles)
   predictions_final = predictions_inv.reshape(
       X_val_static.shape[0], forecast_horizons, len(quantiles_to_predict)
   )
   # Also inverse transform actuals for plotting
   y_val_inv = scalers['Sales'].inverse_transform(y_val.reshape(-1, 1))
   y_val_final = y_val_inv.reshape(X_val_static.shape[0], forecast_horizons)

   print("Predictions inverse transformed.")

   # 11. Visualize Forecast for one Item
   # -----------------------------------
   item_to_plot = 0 # Plot results for the first item in validation set
   item_mask_val = (X_val_static[:, 0] == item_to_plot) # Find sequences for this item

   if np.sum(item_mask_val) > 0:
       # Find the first sequence index for this item in the validation set
       first_val_seq_idx = np.where(item_mask_val)[0][0]

       actual_vals_item = y_val_final[first_val_seq_idx, :]
       pred_quantiles_item = predictions_final[first_val_seq_idx, :, :]

       # Create approximate time axis for the forecast period
       last_train_date = df['Date'].iloc[split_idx + time_steps -1]
       pred_time_axis = pd.date_range(
           last_train_date + pd.DateOffset(months=1),
           periods=forecast_horizons, freq='MS'
       )

       plt.figure(figsize=(12, 6))
       plt.plot(pred_time_axis, actual_vals_item, label='Actual Sales', marker='o', linestyle='--')
       plt.plot(pred_time_axis, pred_quantiles_item[:, 1], label='Median Forecast (q=0.5)', marker='x')
       plt.fill_between(
           pred_time_axis,
           pred_quantiles_item[:, 0], # Lower quantile (q=0.1)
           pred_quantiles_item[:, 2], # Upper quantile (q=0.9)
           color='gray', alpha=0.3, label='Prediction Interval (q=0.1 to q=0.9)'
       )
       plt.title(f'XTFT Quantile Forecast (ItemID {item_to_plot})')
       plt.xlabel('Date')
       plt.ylabel('Sales (Inverse Scaled)')
       plt.legend()
       plt.grid(True)
       plt.show()
   else:
       print(f"No validation data found for ItemID {item_to_plot} to plot.")


.. topic:: Explanations

   1.  **Data Generation:** We create a synthetic dataset simulating
       monthly sales for multiple items (`n_items`). Each item has:
       * A unique `ItemID` (static feature).
       * Time-varying features: `Month`, `Temperature`, and lagged sales
         `PrevMonthSales` (dynamic features).
       * A binary `PlannedPromotion` flag, assumed to be known in
         advance (future feature).
       * The target variable `Sales`.
   2.  **Feature Definition:** Lists (`static_cols`, `dynamic_cols`, etc.)
       are created to clearly assign each column to its role.
   3.  **Scaling:** Numerical features, including the target, are scaled
       using `StandardScaler`. It's crucial to scale features before
       feeding them into neural networks. The scalers are stored for
       later inverse transformation of predictions.
   4.  **Sequence Preparation:** The
       :func:`~fusionlab.utils.ts_utils.reshape_xtft_data` utility is
       used here. This function is ideal for XTFT/TFT as it handles
       grouping data (by `spatial_cols='ItemID'`) and automatically
       creates the separate NumPy arrays required by the model:
       `static_data`, `dynamic_data`, `future_data`, and `target_data`,
       based on the provided column lists, `time_steps`, and
       `forecast_horizons`.
   5.  **Train/Validation Split:** The reshaped sequence arrays are split
       chronologically to create training and validation sets. We take
       the first 80% for training and the last 20% for validation.
       Input arrays are packaged into lists (`train_inputs`, `val_inputs`)
       in the expected order `[static, dynamic, future]`.
   6.  **Model Definition:** We instantiate the :class:`~fusionlab.nn.XTFT`
       model.
       * Input dimensions are derived from the shapes of the prepared
         data arrays (`static_data.shape[-1]`, etc.).
       * `forecast_horizon` matches the sequence preparation.
       * `quantiles=[0.1, 0.5, 0.9]` enables probabilistic forecasting.
       * Key XTFT hyperparameters like `embed_dim`, `lstm_units`,
         `attention_units`, `memory_size`, `num_heads` are set (these
         would typically be tuned using tools like
         :func:`~fusionlab.nn.forecast_tuner.xtft_tuner`).
   7.  **Model Compilation:** The model is compiled with Adam optimizer
       and the :func:`~fusionlab.nn.losses.combined_quantile_loss`
       function, using the same quantiles defined for the model.
   8.  **Model Training:** The `.fit()` method is called with the
       packaged `train_inputs` and `y_train`, using `val_inputs` and
       `y_val` for validation.
   9.  **Prediction:** `model.predict()` is called on the validation
       inputs (`val_inputs`). The output shape is
       `(NumValSamples, Horizon, NumQuantiles)`.
   10. **Inverse Transform:** Predictions are scaled back to the original
       'Sales' units using the saved `target_scaler`. Actual validation
       targets (`y_val`) are also inverse-transformed for comparison.
   11. **Visualization:** The actual sales and the predicted quantiles
       (median line plus shaded interval) are plotted for one sample item
       over the forecast horizon, providing a visual check of the forecast
       accuracy and uncertainty estimation.