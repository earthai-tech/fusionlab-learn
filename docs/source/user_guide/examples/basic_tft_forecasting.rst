.. _example_basic_tft:

=============================
Basic TFT Point Forecasting
=============================

This example demonstrates how to train a standard
:class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
for a basic single-step, point forecasting task using only
dynamic (past observed) features.

We will:
1. Generate simple synthetic time series data.
2. Prepare input sequences and targets using
   :func:`~fusionlab.nn.utils.create_sequences`.
3. Define and compile a basic TFT model.
4. Train the model for a few epochs.
5. Make a sample prediction.

Code Example
------------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt

   from fusionlab.nn.transformers import TemporalFusionTransformer
   from fusionlab.nn.utils import create_sequences
   from fusionlab.nn.losses import combined_quantile_loss # Although not used here let keras register the function

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # 1. Generate Synthetic Data
   # --------------------------
   # Create a simple sine wave with noise
   time = np.arange(0, 100, 0.1)
   amplitude = np.sin(time) + np.random.normal(0, 0.15, len(time))
   df = pd.DataFrame({'Value': amplitude})
   print("Generated data shape:", df.shape)
   # print(df.head()) # Optional: view data

   # 2. Prepare Sequences
   # --------------------
   # Use past 10 steps to predict the next 1 step
   sequence_length = 10
   forecast_horizon = 1 # For single-step point forecast

   # create_sequences expects df, seq_len, target_col name
   # Note: It includes all columns in the sequences by default
   sequences, targets = create_sequences(
       df=df,
       sequence_length=sequence_length,
       target_col='Value',
       forecast_horizon=forecast_horizon, # Predict 1 step ahead
       verbose=0 # Keep output clean for example
   )

   # Reshape targets for Keras MSE loss (samples, horizon)
   targets = targets.reshape(-1, forecast_horizon)

   print(f"Input sequences shape (X): {sequences.shape}")
   # Expected: (NumSamples, SequenceLength, NumFeatures) -> e.g., (990, 10, 1)
   print(f"Target values shape (y): {targets.shape}")
   # Expected: (NumSamples, ForecastHorizon) -> e.g., (990, 1)

   # 3. Define TFT Model for Point Forecast
   # ---------------------------------------
   # We only have one dynamic feature ('Value') in this simple case.
   # We set quantiles=None for point forecasting.
   model = TemporalFusionTransformer(
       dynamic_input_dim=sequences.shape[-1], # Num features in X
       forecast_horizon=forecast_horizon,     # Predict 1 step
       hidden_units=16,                       # Smaller for demo
       num_heads=2,                           # Fewer heads for demo
       quantiles=None,                        # Ensures point forecast
       # Other params use defaults (e.g., no static/future inputs)
   )

   # 4. Compile the Model
   # --------------------
   # Use Mean Squared Error for point forecasting
   model.compile(optimizer='adam', loss='mse')
   print("Model compiled successfully.")

   # 5. Train the Model
   # ------------------
   # For TFT, inputs should be a list/tuple: [static, dynamic, future]
   # Since we only have dynamic, provide None or handle inside model/utils
   # Here, assume model handles list input, provide only dynamic:
   # (Adjust if your specific TFT requires explicit None placeholders)
   # Let's assume the simplest case where only dynamic is needed
   train_inputs = [sequences] # Pass dynamic sequences as a list element
   # For models strictly requiring 3 inputs, might need:
   # train_inputs = [None, sequences, None] # Needs model/validation flexibility

   print("Starting model training (few epochs for demo)...")
   history = model.fit(
       train_inputs,
       targets,
       epochs=5,
       batch_size=32,
       validation_split=0.2, # Use last 20% for validation
       verbose=0 # Suppress epoch progress for example clarity
   )
   print("Training finished.")
   print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

   # 6. Make a Prediction
   # --------------------
   # Use the first validation sample as input for prediction
   # Need to reshape sample input for the model (add batch dim)
   # And package it as a list
   sample_input_dynamic = np.expand_dims(sequences[-1], axis=0)
   sample_input = [sample_input_dynamic]
   # or sample_input = [None, sample_input_dynamic, None] if needed

   print("Making prediction on a sample input...")
   prediction = model.predict(sample_input, verbose=0)
   print("Prediction output shape:", prediction.shape)
   # Expected: (Batch, Horizon, NumOutputs=1) -> (1, 1, 1)
   print("Sample Prediction:", prediction.flatten())


   # 7. Visualize (Optional)
   # -----------------------
   plt.figure(figsize=(12, 6))
   plt.plot(time, amplitude, label='Original Data', alpha=0.7)
   # Plot predictions on validation part for context
   val_start_index = int(len(sequences) * (1 - 0.2)) # Approx start of val
   val_pred_time = time[val_start_index + sequence_length :
                        val_start_index + sequence_length + len(history.epoch)] # Crude time alignment
   # Need to run predict on the whole validation set for a meaningful plot
   val_inputs_dynamic = sequences[val_start_index:]
   val_inputs_list = [val_inputs_dynamic]
   val_predictions = model.predict(val_inputs_list, verbose=0).flatten()
   val_actuals = targets[val_start_index:].flatten()
   val_time = time[val_start_index + sequence_length :
                   val_start_index + sequence_length + len(val_actuals)]

   plt.plot(val_time, val_actuals, label='Actual Validation Data', linestyle='--', marker='.')
   plt.plot(val_time, val_predictions, label='Predicted Validation Data', marker='x')
   plt.title('Basic TFT Point Forecast Example')
   plt.xlabel('Time')
   plt.ylabel('Value')
   plt.legend()
   plt.grid(True)
   plt.show()


.. topic:: Explanations

   1.  **Imports:** We import standard libraries (`numpy`, `pandas`,
       `tensorflow`, `matplotlib`) along with the main model
       :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
       and the :func:`~fusionlab.nn.utils.create_sequences` utility
       for data preparation.
   2.  **Data Generation:** A simple sine wave with added noise is
       created using NumPy and stored in a Pandas DataFrame. This
       serves as our univariate time series data.
   3.  **Sequence Preparation:** The `create_sequences` function is
       used to transform the flat time series into input-output pairs
       suitable for supervised learning.
       * `sequence_length=10`: Each input sample (`X`) will consist
         of 10 consecutive time steps.
       * `target_col='Value'`: The 'Value' column is used as the
         source for target values.
       * `forecast_horizon=1`: We aim to predict only the single
         next time step immediately following each input sequence.
       * The output `sequences` contains the input windows, and
         `targets` contains the corresponding single future value for
         each sequence. Targets are reshaped for compatibility with
         Keras loss functions.
   4.  **Model Definition:** We instantiate the `TemporalFusionTransformer`.
       * `dynamic_input_dim`: Set to the number of features in our
         input sequences (`sequences.shape[-1]`, which is 1 in this
         case).
       * `forecast_horizon=1`: Matches the target preparation.
       * `quantiles=None`: This is key for **point forecasting**. It
         tells the model to output a single value per horizon step
         and configures it internally for a loss like MSE.
       * `hidden_units` and `num_heads` are reduced for faster demo
         training. Other parameters like `static_input_dim` and
         `future_input_dim` default to `None`, indicating they are not
         used in this basic example.
   5.  **Model Compilation:** The model is compiled using the 'adam'
       optimizer and 'mse' (Mean Squared Error) loss, which is
       appropriate for point forecasting (regression).
   6.  **Model Training:** The `.fit()` method trains the model.
       * **Input Format:** The input `X` is passed as a list
         `[sequences]`. While TFT can handle static/future inputs, in
         this case, we only provide the dynamic sequence. *(Note: Some
         model implementations might strictly require a list of 3
         elements, potentially with `None` placeholders for unused
         inputs like `[None, sequences, None]`)*.
       * `targets`: The prepared target array.
       * `epochs=5`: We train only for a few epochs for speed.
       * `validation_split=0.2`: Keras automatically uses the last 20%
         of the data (in the order provided) for validation during
         training.
   7.  **Prediction:** We demonstrate `.predict()` on a single sample
       (the last sequence from the dataset). The input needs to be
       reshaped to include a batch dimension (`np.expand_dims`) and
       passed as a list. The output shape reflects (Batch, Horizon,
       OutputsPerStep=1).
   8.  **Visualization:** A simple plot shows the original data, the
       actual validation data, and the model's predictions on the
       validation set to give a visual sense of the fit. Note that
       aligning prediction time steps correctly requires careful index
       management.
       


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


.. _example_tft_required_inputs:

Example using TFT (Required Inputs Version)
-------------------------------------------

This example uses the stricter :class:`~fusionlab.nn.transformers.TFT`
implementation, which mandates that static, dynamic (past), and future
inputs are always provided. We will again perform a single-step point
forecast, but the data preparation and model call differ slightly.

We will:
1. Generate synthetic data with static, dynamic, and future features.
2. Use :func:`~fusionlab.nn.utils.reshape_xtft_data` to prepare the
   three separate input arrays.
3. Define and compile the stricter `TFT` model.
4. Train the model using the required three-part input list.
5. Make a sample prediction.

Code Example (Required Inputs)
------------------------------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import StandardScaler

   # Import the stricter TFT class and the appropriate reshape util
   from fusionlab.nn.transformers import TFT # The revised class
   from fusionlab.nn.utils import reshape_xtft_data

   # Suppress warnings and TF logs
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # 1. Generate Synthetic Data (Static, Dynamic, Future)
   # ----------------------------------------------------
   n_items = 2
   n_timesteps = 50 # Increase data points
   date_rng = pd.date_range(start='2021-01-01', periods=n_timesteps, freq='D')
   df_list = []
   for item_id in range(n_items):
       time_idx = np.arange(n_timesteps)
       value = 50 + item_id * 10 + time_idx * 0.5 + np.random.randn(n_timesteps) * 2
       static_cat = item_id # Example static feature
       future_event = (time_idx % 7 == 0).astype(int) # Example future event (e.g., Sunday)
       item_df = pd.DataFrame({
           'Date': date_rng, 'ItemID': item_id, 'Category': static_cat,
           'DayOfWeek': date_rng.dayofweek, # Dynamic
           'FutureEvent': future_event, # Known Future
           'Value': value
       })
       item_df['ValueLag1'] = item_df['Value'].shift(1) # Dynamic
       df_list.append(item_df)
   df = pd.concat(df_list).dropna().reset_index(drop=True)
   print("Generated data shape:", df.shape)

   # 2. Define Features & Scale
   # --------------------------
   target_col = 'Value'
   dt_col = 'Date'
   static_cols = ['ItemID', 'Category']
   dynamic_cols = ['DayOfWeek', 'ValueLag1']
   # Future features known for lookback + horizon
   future_cols = ['FutureEvent', 'DayOfWeek'] # Use DayOfWeek also as future known
   spatial_cols = ['ItemID']

   # Scale relevant columns (Value, ValueLag1)
   scaler = StandardScaler()
   num_cols_to_scale = ['Value', 'ValueLag1']
   df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])
   print("Numerical features scaled.")

   # 3. Prepare Sequences using reshape_xtft_data
   # --------------------------------------------
   time_steps = 7          # Lookback window
   forecast_horizon = 1    # Single step point forecast

   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df, dt_col=dt_col, target_col=target_col,
       dynamic_cols=dynamic_cols, static_cols=static_cols,
       future_cols=future_cols, spatial_cols=spatial_cols,
       time_steps=time_steps, forecast_horizons=forecast_horizon,
       verbose=0
   )
   print(f"\nReshaped Data Shapes:")
   print(f"  Static : {static_data.shape}")
   print(f"  Dynamic: {dynamic_data.shape}")
   print(f"  Future : {future_data.shape}")
   print(f"  Target : {target_data.shape}")
   # Target shape needs to be (Samples, Horizon=1) for MSE loss
   targets = target_data.reshape(-1, forecast_horizon)

   # 4. Train/Validation Split (Simple for demo)
   # -------------------------------------------
   val_split_fraction = 0.2
   n_samples = static_data.shape[0]
   split_idx = int(n_samples * (1 - val_split_fraction))
   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   X_train_future, X_val_future = future_data[:split_idx], future_data[split_idx:]
   y_train, y_val = targets[:split_idx], targets[split_idx:]

   # Package inputs as the REQUIRED list [static, dynamic, future]
   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future]
   print("Data prepared and split.")

   # 5. Define Required-Inputs TFT Model
   # -----------------------------------
   model_req = TFT( # Using the revised TFT class
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1], # Must provide all dims
       forecast_horizon=forecast_horizon,
       hidden_units=16, num_heads=2, num_lstm_layers=1,
       quantiles=None # Point forecast
   )
   print("Required-Inputs TFT model instantiated.")

   # 6. Compile the Model
   # --------------------
   model_req.compile(optimizer='adam', loss='mse')
   print("Model compiled successfully.")

   # 7. Train the Model
   # ------------------
   print("Starting model training...")
   history_req = model_req.fit(
       train_inputs, # Pass the list [static, dynamic, future]
       y_train,
       validation_data=(val_inputs, y_val),
       epochs=5,
       batch_size=16,
       verbose=0
   )
   print("Training finished.")
   print(f"Final validation loss: {history_req.history['val_loss'][-1]:.4f}")

   # 8. Make a Prediction
   # --------------------
   # Use first validation sample
   sample_input = [X_val_static[0:1], X_val_dynamic[0:1], X_val_future[0:1]]

   print("Making prediction on a sample input...")
   prediction = model_req.predict(sample_input, verbose=0)
   print("Prediction output shape:", prediction.shape)
   # Expected: (Batch=1, Horizon=1, OutputDim=1) -> (1, 1, 1)
   print("Sample Prediction:", prediction.flatten())


.. topic:: Explanations (Required Inputs TFT)

   1.  **Data Generation:** We create data that includes columns
       explicitly intended as static (`ItemID`, `Category`), dynamic
       (`DayOfWeek`, `ValueLag1`), and future (`PlannedEvent`,
       `DayOfWeek`).
   2.  **Feature Definition & Scaling:** We define lists for each
       feature type required by the model and apply scaling only to
       the continuous numerical features.
   3.  **Sequence Preparation:** We use
       :func:`~fusionlab.nn.utils.reshape_xtft_data` because it is
       designed to handle the separation of static, dynamic, and
       future features based on the provided column lists, creating
       the distinct NumPy arrays needed. The target shape is adjusted
       for the MSE loss.
   4.  **Train/Validation Split:** The resulting sequence arrays
       (`static_data`, `dynamic_data`, `future_data`, `targets`) are
       split. Note that the input for fitting/predicting is packaged
       as a **list of three arrays** in the specific order
       `[static, dynamic, future]`.
   5.  **Model Definition:** We instantiate the revised
       :class:`~fusionlab.nn.transformers.TFT` class. It **requires**
       integer dimensions to be provided for `static_input_dim`,
       `dynamic_input_dim`, and `future_input_dim`. We set
       `quantiles=None` for point forecasting.
   6.  **Compilation:** Standard compilation with 'adam' and 'mse'.
   7.  **Training:** The `.fit` method is called with the 3-element
       `train_inputs` list.
   8.  **Prediction:** The `.predict` method is called with a sample
       input, also structured as a 3-element list. The output shape
       reflects the single-step point forecast.