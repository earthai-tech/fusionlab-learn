.. _example_quantile_tft:

========================
TFT Quantile Forecasting
========================

This guide demonstrates how to configure and train Temporal Fusion
Transformer models available in ``fusionlab`` to produce **quantile
forecasts**. Instead of predicting a single point value, the model
predicts multiple quantiles (e.g., 10th, 50th, 90th percentiles),
providing an estimate of the prediction uncertainty.

We will show examples using both the flexible
:class:`~fusionlab.nn.TemporalFusionTransformer` (handling optional
inputs) and the stricter :class:`~fusionlab.nn.transformers.TFT`
(requiring all inputs).

Example 1: Using Flexible `TemporalFusionTransformer`
-------------------------------------------------------
This example builds upon the :doc:`basic_tft_forecasting` example, using
only dynamic features but modifying the model to output quantile predictions.

We will:
1. Generate simple synthetic time series data.
2. Prepare sequences and multi-step targets using
   :func:`~fusionlab.nn.utils.create_sequences`.
3. Instantiate the flexible `TemporalFusionTransformer` with specified
   `quantiles`.
4. Compile the model using `combined_quantile_loss`.
5. Train the model.
6. Interpret and visualize the multi-quantile output.

**Code Example (Flexible TFT):**

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt

   # Assuming fusionlab components are importable
   from fusionlab.nn.transformers import TemporalFusionTransformer # Flexible version
   from fusionlab.nn.utils import create_sequences
   from fusionlab.nn.losses import combined_quantile_loss

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # 1. Generate Synthetic Data (same as basic example)
   # --------------------------------------------------
   time = np.arange(0, 100, 0.1)
   amplitude = np.sin(time) + np.random.normal(0, 0.15, len(time))
   df = pd.DataFrame({'Value': amplitude})
   print("Generated data shape:", df.shape)

   # 2. Prepare Sequences for Multi-Step Forecasting
   # -----------------------------------------------
   sequence_length = 10
   forecast_horizon = 5 # Predict next 5 steps

   sequences, targets = create_sequences(
       df=df,
       sequence_length=sequence_length,
       target_col='Value',
       forecast_horizon=forecast_horizon, # Predict 5 steps ahead
       verbose=0
   )

   # Reshape targets for Keras: (Samples, Horizon, OutputDim=1)
   targets = targets.reshape(-1, forecast_horizon, 1).astype(np.float32)
   # Ensure sequences are float32 as well
   sequences = sequences.astype(np.float32)

   print(f"\nInput sequences shape (X): {sequences.shape}")
   print(f"Target values shape (y): {targets.shape}")

   # 3. Define Flexible TFT Model for Quantile Forecast
   # --------------------------------------------------
   quantiles_to_predict = [0.1, 0.5, 0.9] # 10th, 50th (Median), 90th

   model_flex = TemporalFusionTransformer( # Using the flexible model
       dynamic_input_dim=sequences.shape[-1], # Only dynamic needed
       static_input_dim=None, # Explicitly None
       future_input_dim=None, # Explicitly None
       forecast_horizon=forecast_horizon,
       hidden_units=16,
       num_heads=2,
       quantiles=quantiles_to_predict # Provide the list of quantiles
   )

   # 4. Compile the Model with Quantile Loss
   # ---------------------------------------
   loss_fn = combined_quantile_loss(quantiles=quantiles_to_predict)
   model_flex.compile(optimizer='adam', loss=loss_fn)
   print("Flexible TFT compiled with combined quantile loss.")

   # 5. Train the Model
   # ------------------
   # Input is list with only dynamic sequences
   train_inputs = [sequences]

   print("Starting flexible TFT training (few epochs)...")
   history = model_flex.fit(
       train_inputs,
       targets, # Shape (Samples, Horizon, 1)
       epochs=5,
       batch_size=32,
       validation_split=0.2,
       verbose=0
   )
   print("Training finished.")

   # 6. Make Predictions (Quantiles)
   # -------------------------------
   val_start_index = int(len(sequences) * (1 - 0.2))
   sample_input_dynamic = np.expand_dims(sequences[val_start_index], axis=0)
   sample_input = [sample_input_dynamic]

   print("\nMaking quantile predictions (flexible TFT)...")
   predictions_quantiles = model_flex.predict(sample_input, verbose=0)
   print("Prediction output shape:", predictions_quantiles.shape)
   # Expected: (Batch, Horizon, NumQuantiles) -> (1, 5, 3)

   # 7. Visualize Quantile Forecast (code omitted for brevity, same as before)
   # ... (Plotting code using 'targets' and 'predictions_quantiles') ...
   print("Visualization would show prediction intervals.")


.. topic:: Explanations (Flexible TemporalFusionTransformer)

   1.  **Imports & Data:** Standard setup using the flexible
       :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`.
   2.  **Sequence Preparation:** :func:`~fusionlab.nn.utils.create_sequences`
       is used with ``forecast_horizon=5`` to get multi-step targets.
       Targets are reshaped to `(Samples, Horizon, 1)`.
   3.  **Model Definition:** The flexible `TemporalFusionTransformer` is
       instantiated. Only ``dynamic_input_dim`` is required. Crucially,
       the ``quantiles`` parameter is set to the list `[0.1, 0.5, 0.9]`.
   4.  **Model Compilation:** The model is compiled using the loss function
       returned by :func:`~fusionlab.nn.losses.combined_quantile_loss`,
       passing the same list of quantiles.
   5.  **Training:** The model is trained using `.fit`. The input `X` is
       provided as a single-element list `[sequences]` because only
       dynamic inputs were configured.
   6.  **Prediction:** `.predict` returns an output shape corresponding to
       `(Batch, Horizon, NumQuantiles)`.
   7.  **Visualization:** Shows the median prediction and the uncertainty
       interval derived from the lower and upper quantiles against the
       actual values.


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Example 2: Using Stricter `TFT` (Required Inputs Version)
---------------------------------------------------------
This example uses the revised :class:`~fusionlab.nn.transformers.TFT`
class, which requires static, dynamic, and future inputs. We adapt the
data preparation to include these and demonstrate quantile forecasting.

We will:
1. Generate synthetic data with static, dynamic, and future features.
2. Use :func:`~fusionlab.nn.utils.reshape_xtft_data` to prepare the
   three separate input arrays and multi-step targets.
3. Define and compile the stricter `TFT` model with quantile outputs.
4. Train the model using the required three-part input list.
5. Make and visualize quantile predictions.

**Code Example (Stricter TFT):**

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import StandardScaler

   # Import the stricter TFT class and reshape util
   from fusionlab.nn.transformers import TFT # Stricter version
   from fusionlab.nn.utils import reshape_xtft_data
   from fusionlab.nn.losses import combined_quantile_loss

   # Suppress warnings and TF logs
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # 1. Generate Synthetic Data (Static, Dynamic, Future)
   # ----------------------------------------------------
   # (Using same generation as tft_no_flex.rst example)
   n_items = 2; n_timesteps = 50
   date_rng = pd.date_range('2021-01-01', periods=n_timesteps, freq='D')
   df_list = []
   for item_id in range(n_items):
       time_idx = np.arange(n_timesteps)
       value = 50 + item_id*10 + time_idx*0.5 + np.sin(time_idx/3)*5 + np.random.randn(n_timesteps)*2
       static_cat = item_id + 1
       future_event = (date_rng.dayofweek >= 5).astype(int) # Weekend flag
       item_df = pd.DataFrame({'Date': date_rng, 'ItemID': item_id,
           'Category': static_cat, 'DayOfWeek': date_rng.dayofweek,
           'FutureEvent': future_event, 'Value': value })
       item_df['ValueLag1'] = item_df['Value'].shift(1)
       df_list.append(item_df)
   df = pd.concat(df_list).dropna().reset_index(drop=True)
   print("Generated data shape:", df.shape)

   # 2. Define Features & Scale
   # --------------------------
   target_col = 'Value'; dt_col = 'Date'; static_cols = ['ItemID', 'Category']
   dynamic_cols = ['DayOfWeek', 'ValueLag1']; future_cols = ['FutureEvent', 'DayOfWeek']
   spatial_cols = ['ItemID']
   scaler = StandardScaler()
   num_cols_to_scale = ['Value', 'ValueLag1']
   df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])
   print("Numerical features scaled.")

   # 3. Prepare Sequences using reshape_xtft_data
   # --------------------------------------------
   time_steps = 7; forecast_horizon = 5 # Predict 5 steps
   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df, dt_col=dt_col, target_col=target_col, dynamic_cols=dynamic_cols,
       static_cols=static_cols, future_cols=future_cols, spatial_cols=spatial_cols,
       time_steps=time_steps, forecast_horizons=forecast_horizon, verbose=0
   )
   # Reshape target for loss: (Samples, Horizon, OutputDim=1)
   targets = target_data.astype(np.float32) # Already has OutputDim=1
   print("\nReshaped Data Shapes:")
   print(f"  Static : {static_data.shape}, Dynamic: {dynamic_data.shape}")
   print(f"  Future : {future_data.shape}, Target : {targets.shape}")

   # 4. Train/Validation Split
   # -------------------------
   val_split_fraction = 0.3
   n_samples = static_data.shape[0]
   split_idx = int(n_samples * (1 - val_split_fraction))
   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   X_train_future, X_val_future = future_data[:split_idx], future_data[split_idx:]
   y_train, y_val = targets[:split_idx], targets[split_idx:]
   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future]
   print(f"Data split. Train samples: {split_idx}")

   # 5. Define Stricter TFT Model for Quantile Forecast
   # --------------------------------------------------
   quantiles_to_predict = [0.1, 0.5, 0.9]
   model_strict = TFT( # Using the stricter TFT class
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1], # Must provide all dims
       forecast_horizon=forecast_horizon,
       quantiles=quantiles_to_predict, # Set quantiles
       hidden_units=16, num_heads=2, num_lstm_layers=1, output_dim=1
   )
   print("Stricter TFT model instantiated for quantiles.")

   # 6. Compile the Model with Quantile Loss
   # ---------------------------------------
   loss_fn_q = combined_quantile_loss(quantiles=quantiles_to_predict)
   model_strict.compile(optimizer='adam', loss=loss_fn_q)
   print("Model compiled.")

   # 7. Train the Model
   # ------------------
   print("Starting stricter TFT training...")
   history_strict = model_strict.fit(
       train_inputs, # Pass list [static, dynamic, future]
       y_train,      # Target shape (Samples, Horizon, 1)
       validation_data=(val_inputs, y_val),
       epochs=5, batch_size=16, verbose=0
   )
   print("Training finished.")

   # 8. Make Predictions (Quantiles)
   # -------------------------------
   print("\nMaking quantile predictions (stricter TFT)...")
   predictions_strict = model_strict.predict(val_inputs, verbose=0)
   print("Prediction output shape:", predictions_strict.shape)
   # Expected: (Batch, Horizon, NumQuantiles) -> (N_val, 5, 3)

   # 9. Visualize (Optional - similar plotting code as Example 1)
   # ...


.. topic:: Explanations (Stricter TFT)

   1.  **Data Generation:** We create data containing static, dynamic,
       and future features, as this model requires all three.
   2.  **Feature Definition & Scaling:** Roles are assigned, and numerical
       features scaled as usual.
   3.  **Sequence Preparation:** We **must** use a utility like
       :func:`~fusionlab.nn.utils.reshape_xtft_data` that can separate
       the different feature types into distinct arrays: `static_data`,
       `dynamic_data`, `future_data`. Targets are also generated for the
       multi-step horizon and reshaped to `(Samples, Horizon, 1)`.
   4.  **Train/Validation Split:** The split is performed on the generated
       sequence arrays. The input for `fit`/`predict` **must** be a list
       containing the three arrays in the order
       `[static, dynamic, future]`.
   5.  **Model Definition:** We instantiate the stricter
       :class:`~fusionlab.nn.transformers.TFT` class. All three input
       dimensions (`static_input_dim`, `dynamic_input_dim`,
       `future_input_dim`) are required arguments. We pass the desired
       `quantiles` list.
   6.  **Compilation & Training:** Compilation uses
       :func:`~fusionlab.nn.losses.combined_quantile_loss`. Training uses
       the 3-element input list.
   7.  **Prediction:** The model predicts the specified quantiles across
       the forecast horizon.
   8.  **Visualization:** Similar to the first example, results can be
       plotted showing the median and prediction intervals.