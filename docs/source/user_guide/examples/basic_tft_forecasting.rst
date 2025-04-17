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

   # Assuming fusionlab components are importable
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