.. _example_quantile_tft:

========================
TFT Quantile Forecasting
========================

Building upon the :doc:`basic_tft_forecasting` example, this guide
demonstrates how to configure and train a
:class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
to produce **quantile forecasts**. Instead of predicting a single
point value, the model predicts multiple quantiles (e.g., 10th, 50th,
90th percentiles), providing an estimate of the prediction uncertainty.

We will modify the previous example to:
1. Generate multi-step targets.
2. Instantiate TFT with specified `quantiles`.
3. Compile the model using `combined_quantile_loss`.
4. Interpret and visualize the multi-quantile output.

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
   from fusionlab.nn.losses import combined_quantile_loss

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # 1. Generate Synthetic Data (same as before)
   # --------------------------
   time = np.arange(0, 100, 0.1)
   amplitude = np.sin(time) + np.random.normal(0, 0.15, len(time))
   df = pd.DataFrame({'Value': amplitude})
   print("Generated data shape:", df.shape)

   # 2. Prepare Sequences for Multi-Step Forecasting
   # ------------------------------------------------
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
   # OutputDim is 1 because we predict one target variable ('Value')
   targets = targets.reshape(-1, forecast_horizon, 1).astype(np.float32)
   # Ensure sequences are float32 as well
   sequences = sequences.astype(np.float32)

   print(f"Input sequences shape (X): {sequences.shape}")
   # Expected: (NumSamples, SequenceLength, NumFeatures) -> (e.g., 986, 10, 1)
   print(f"Target values shape (y): {targets.shape}")
   # Expected: (NumSamples, ForecastHorizon, OutputDim) -> (e.g., 986, 5, 1)

   # 3. Define TFT Model for Quantile Forecast
   # -----------------------------------------
   # Specify the desired quantiles
   quantiles_to_predict = [0.1, 0.5, 0.9] # 10th, 50th (Median), 90th

   model = TemporalFusionTransformer(
       dynamic_input_dim=sequences.shape[-1],
       forecast_horizon=forecast_horizon,
       hidden_units=16,
       num_heads=2,
       # *** Key change: Provide the list of quantiles ***
       quantiles=quantiles_to_predict,
       # Other params use defaults
   )

   # 4. Compile the Model with Quantile Loss
   # ---------------------------------------
   # Use the dedicated combined_quantile_loss function
   loss_fn = combined_quantile_loss(quantiles=quantiles_to_predict)
   model.compile(optimizer='adam', loss=loss_fn)
   print("Model compiled successfully with combined quantile loss.")

   # 5. Train the Model
   # ------------------
   # Inputs are the same as before (only dynamic in this case)
   train_inputs = [sequences]
   # Or potentially [None, sequences, None]

   print("Starting model training (few epochs for demo)...")
   history = model.fit(
       train_inputs,
       targets, # Shape (Samples, Horizon, 1)
       epochs=5,
       batch_size=32,
       validation_split=0.2,
       verbose=0
   )
   print("Training finished.")
   print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

   # 6. Make Predictions (Quantiles)
   # -------------------------------
   # Use the first validation sample as input
   val_start_index = int(len(sequences) * (1 - 0.2))
   sample_input_dynamic = np.expand_dims(sequences[val_start_index], axis=0)
   sample_input = [sample_input_dynamic]
   # or sample_input = [None, sample_input_dynamic, None]

   print("Making quantile predictions on a sample input...")
   predictions_quantiles = model.predict(sample_input, verbose=0)
   print("Prediction output shape:", predictions_quantiles.shape)
   # Expected: (Batch, Horizon, NumQuantiles) -> (1, 5, 3)
   print("Sample Predictions (Quantiles 0.1, 0.5, 0.9):")
   # Print predictions for each step in the horizon
   for step in range(forecast_horizon):
       print(f"  Step {step+1}: {predictions_quantiles[0, step, :]}")


   # 7. Visualize Quantile Forecast
   # ------------------------------
   # Predict on the whole validation set
   val_inputs_dynamic = sequences[val_start_index:]
   val_inputs_list = [val_inputs_dynamic]
   val_predictions = model.predict(val_inputs_list, verbose=0)
   # Shape: (NumValSamples, Horizon, NumQuantiles) -> e.g., (198, 5, 3)

   # Select a sample from validation set to plot (e.g., the first one)
   sample_to_plot = 0
   actual_vals = targets[val_start_index + sample_to_plot, :, 0]
   pred_quantiles = val_predictions[sample_to_plot, :, :] # (Horizon, NumQuantiles)

   # Time axis for plotting
   start_time_index = val_start_index + sequence_length + sample_to_plot
   pred_time = time[start_time_index : start_time_index + forecast_horizon]

   plt.figure(figsize=(12, 6))
   plt.plot(pred_time, actual_vals, label='Actual Value', marker='o', linestyle='--')
   # Plot median (0.5 quantile, index 1 in our list)
   plt.plot(pred_time, pred_quantiles[:, 1], label='Predicted Median (q=0.5)', marker='x')
   # Fill between lower and upper quantiles (0.1 and 0.9, indices 0 and 2)
   plt.fill_between(
       pred_time,
       pred_quantiles[:, 0], # Lower quantile (q=0.1)
       pred_quantiles[:, 2], # Upper quantile (q=0.9)
       color='gray', alpha=0.3, label='Prediction Interval (q=0.1 to q=0.9)'
   )

   plt.title(f'TFT Quantile Forecast (Sample {sample_to_plot} from Validation Set)')
   plt.xlabel('Time')
   plt.ylabel('Value')
   plt.legend()
   plt.grid(True)
   plt.show()


.. topic:: Explanations

   1.  **Imports & Data:** Same as the basic example.
   2.  **Sequence Preparation:**
       * We set ``forecast_horizon=5`` in `create_sequences` to
         predict 5 steps ahead.
       * The resulting `targets` array now has shape
         `(NumSamples, 5)`.
       * **Crucially**, we reshape `targets` to `(NumSamples, 5, 1)`
         before feeding it to the model during training. Keras loss
         functions often expect the target to have a trailing dimension
         even if it's just 1 (representing the output feature dim).
   3.  **Model Definition:**
       * We define `quantiles_to_predict = [0.1, 0.5, 0.9]`.
       * This list is passed to the `TemporalFusionTransformer` via the
         ``quantiles`` argument during instantiation. This tells the
         model to configure its final output layer(s) to produce
         predictions for these specific quantiles.
   4.  **Model Compilation:**
       * We create the appropriate loss function using
         :func:`~fusionlab.nn.losses.combined_quantile_loss`, passing
         the *same* list of quantiles (`quantiles_to_predict`). This
         ensures the loss calculation aligns with the model's output.
       * The model is compiled with this specific loss function.
   5.  **Model Training:** Training proceeds similarly, but the model
       now learns to minimize the quantile loss, effectively learning
       to predict the specified percentiles of the target distribution
       for each future step.
   6.  **Prediction:**
       * The `model.predict()` output shape changes to
         `(Batch, Horizon, NumQuantiles)`. In this case, `(1, 5, 3)`,
         reflecting predictions for 5 steps ahead across 3 quantiles.
   7.  **Visualization:** The plot is adapted to show the probabilistic
       nature of the forecast. We plot the actual values, the predicted
       median (0.5 quantile), and shade the area between the lower (0.1)
       and upper (0.9) predicted quantiles to visualize the prediction
       interval or uncertainty range.