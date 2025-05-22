.. _exercise_basic_forecasting:

===============================================================
Exercise: Basic Point Forecasting with Flexible TFT
===============================================================

Welcome! This exercise will guide you through the fundamentals of
point forecasting using the flexible
:class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
from ``fusionlab-learn``. We'll focus on a simple scenario using only
dynamic (past observed) features to predict a single future time step.

**Learning Objectives:**

* Generate a simple synthetic time series.
* Prepare input sequences and targets for a forecasting model using
  :func:`~fusionlab.nn.utils.create_sequences`.
* Understand how to instantiate and compile the flexible
  `TemporalFusionTransformer` for point forecasting (i.e., with
  `quantiles=None`).
* Correctly format inputs for the model when only dynamic features
  are used.
* Train the model and make basic predictions.
* Visualize the forecast against actual values.

Let's get started!

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

Before you begin, ensure you have ``fusionlab-learn`` and its
common dependencies installed. For visualizations, `matplotlib` is
also needed.

.. code-block:: bash

   pip install fusionlab-learn matplotlib

---

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~
First, we import all the necessary libraries. This includes ``pandas``
for data handling, ``numpy`` for numerical operations, ``tensorflow``
for the model, ``matplotlib`` for plotting, and the relevant components
from ``fusionlab``.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   import warnings
   import os

   # FusionLab imports
   from fusionlab.nn.transformers import TemporalFusionTransformer
   from fusionlab.nn.utils import create_sequences
   # Import for Keras to recognize custom loss if model was saved with it
   from fusionlab.nn.losses import combined_quantile_loss

   # Suppress warnings and TF logs for cleaner output
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   if hasattr(tf, 'autograph'): # Check for autograph availability
       tf.autograph.set_verbosity(0)

   # Directory for saving any output images from this exercise
   exercise_output_dir_basic = "./basic_forecasting_exercise_outputs"
   os.makedirs(exercise_output_dir_basic, exist_ok=True)

   print("Libraries imported and setup complete for basic forecasting exercise.")

**Expected Output 1.1:**

.. code-block:: text

   Libraries imported and setup complete for basic forecasting exercise.

Step 2: Generate Synthetic Time Series Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We'll create a simple sine wave with some added random noise. This will
be our univariate time series that we want to forecast.

.. code-block:: python
   :linenos:

   # For reproducibility
   np.random.seed(42)
   tf.random.set_seed(42)

   time_ex = np.arange(0, 100, 0.1)
   amplitude_ex = np.sin(time_ex) + np.random.normal(
       0, 0.15, len(time_ex)
       )
   df_ex = pd.DataFrame({'Value': amplitude_ex})
   print(f"Generated data shape for exercise: {df_ex.shape}")
   print("Sample of generated data:")
   print(df_ex.head())

**Expected Output 2.2:**

.. code-block:: text

   Generated data shape for exercise: (1000, 1)
   Sample of generated data:
         Value
   0  0.074540
   1  0.070004
   2  0.140878
   3  0.312668
   4  0.208073

Step 3: Prepare Input Sequences and Targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Forecasting models learn from sequences of past data to predict future
values. We'll use the :func:`~fusionlab.nn.utils.create_sequences`
utility to transform our time series into these input-output pairs.
We will use the past 10 time steps to predict the single next time step.

.. code-block:: python
   :linenos:

   sequence_length_ex = 10    # How many past steps to look at (lookback window)
   forecast_horizon_ex = 1    # How many steps ahead to predict
   target_col_name_ex = 'Value'

   # Create sequences
   # `sequences_ex` will be our X (input features)
   # `targets_ex` will be our y (what we want to predict)
   sequences_ex, targets_ex = create_sequences(
       df=df_ex,
       sequence_length=sequence_length_ex,
       target_col=target_col_name_ex,
       forecast_horizon=forecast_horizon_ex,
       verbose=0 # Keep output clean
   )

   # Ensure data types are float32 for TensorFlow
   sequences_ex = sequences_ex.astype(np.float32)
   # Reshape targets for Keras: (Samples, Horizon, OutputDim)
   # Here, OutputDim is 1 as we predict one feature ('Value')
   targets_ex = targets_ex.reshape(
       -1, forecast_horizon_ex, 1
       ).astype(np.float32)

   print(f"\nInput sequences shape (X): {sequences_ex.shape}")
   print(f"Target values shape (y): {targets_ex.shape}")

**Expected Output 3.3:**
   *(The number of samples will be `len(df_ex) - sequence_length_ex - forecast_horizon_ex + 1`
   if `forecast_horizon > 0` in `create_sequences` logic, or `len(df_ex) - sequence_length_ex`
   if `forecast_horizon=0` means reconstruct. For `forecast_horizon=1`, it's typically
   `len(df_ex) - sequence_length_ex`)*
   *For `create_sequences` as typically implemented for forecasting, it should be `len(df) - sequence_length - forecast_horizon + 1`.*
   *So, 1000 - 10 - 1 + 1 = 990 samples.*

.. code-block:: text

   Input sequences shape (X): (990, 10, 1)
   Target values shape (y): (990, 1, 1)

Step 4: Define and Compile the Flexible TFT Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, we instantiate the
:class:`~fusionlab.nn.transformers.TemporalFusionTransformer`.
Since this exercise uses only dynamic past features:
* `dynamic_input_dim` is set to the number of features in our `sequences_ex`.
* `static_input_dim` and `future_input_dim` are left as `None` (their defaults).
* `quantiles` is set to `None` for point forecasting.
* `output_dim=1` as we are predicting a single target value.

We compile the model with 'adam' optimizer and 'mse' (Mean Squared Error)
loss, suitable for regression tasks like point forecasting.

.. code-block:: python
   :linenos:

   num_dynamic_features_ex = sequences_ex.shape[-1] # Should be 1

   tft_model_ex = TemporalFusionTransformer(
       dynamic_input_dim=num_dynamic_features_ex,
       # static_input_dim and future_input_dim default to None
       forecast_horizon=forecast_horizon_ex,
       output_dim=1,
       hidden_units=16,        # Using smaller values for faster demo
       num_heads=2,
       num_lstm_layers=1,      # A single LSTM layer in the encoder
       lstm_units=16,
       quantiles=None          # Crucial for point forecasting
   )
   print("\nFlexible TemporalFusionTransformer instantiated for point forecast.")

   tft_model_ex.compile(optimizer='adam', loss='mse')
   print("Model compiled successfully with MSE loss.")

**Expected Output 4.4:**

.. code-block:: text

   Flexible TemporalFusionTransformer instantiated for point forecast.
   Model compiled successfully with MSE loss.

Step 5: Train the Model
~~~~~~~~~~~~~~~~~~~~~~~
We train the model using the `.fit()` method. The
`TemporalFusionTransformer` expects its inputs as a list of three
elements: `[static_inputs, dynamic_inputs, future_inputs]`.
Since we only have dynamic inputs for this exercise, the static and
future inputs will be `None` in this list.

.. code-block:: python
   :linenos:

   # Prepare inputs for the model's fit method in the correct order
   # [Static, Dynamic, Future]
   # since static and Future are None, then 
   # pass directly sequences_ex as dynamic only. 
   train_inputs_list_ex = [sequences_ex]

   print("\nStarting model training (this may take a few moments)...")
   history_obj_ex = tft_model_ex.fit(
       train_inputs_list_ex, # Pass the 3-element list
       targets_ex,           # Target shape (Samples, Horizon, OutputDim)
       epochs=10,            # Train for more epochs for better results
       batch_size=32,
       validation_split=0.2, # Use last 20% of data for validation
       verbose=1             # Show training progress per epoch
   )
   print("Training finished.")
   if history_obj_ex and history_obj_ex.history.get('val_loss'):
       final_val_loss = history_obj_ex.history['val_loss'][-1]
       print(f"Final validation loss: {final_val_loss:.4f}")

**Expected Output 5.5:**
   *(Output will show Keras training progress for 5 epochs. The final
   validation loss will vary.)*

.. code-block:: text

   Starting model training (this may take a few moments)...
   Epoch 1/10
   25/25 [==============================] - 5s 37ms/step - loss: 0.4707 - val_loss: 0.2404
   Epoch 2/10
   25/25 [==============================] - 0s 8ms/step - loss: 0.2550 - val_loss: 0.1804
   Epoch 3/10
   25/25 [==============================] - 0s 8ms/step - loss: 0.2153 - val_loss: 0.1285
   Epoch 4/10
   25/25 [==============================] - 0s 8ms/step - loss: 0.1804 - val_loss: 0.0970
   Epoch 5/10
   25/25 [==============================] - 0s 8ms/step - loss: 0.1599 - val_loss: 0.0901
   Epoch 6/10
   25/25 [==============================] - 0s 9ms/step - loss: 0.1536 - val_loss: 0.0911
   Epoch 7/10
   25/25 [==============================] - 0s 9ms/step - loss: 0.1449 - val_loss: 0.0924
   Epoch 8/10
   25/25 [==============================] - 0s 9ms/step - loss: 0.1366 - val_loss: 0.0919
   Epoch 9/10
   25/25 [==============================] - 0s 8ms/step - loss: 0.1298 - val_loss: 0.0907
   Epoch 10/10
   25/25 [==============================] - 0s 8ms/step - loss: 0.1306 - val_loss: 0.0869
   Training finished.
   Final validation loss: 0.0869

Step 6: Make Predictions and Visualize Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's use the trained model to predict on the validation portion of our
data and plot these predictions against the actual values.

.. code-block:: python
   :linenos:

   # Prepare validation data for prediction
   num_total_samples = sequences_ex.shape[0]
   val_start_index_ex = int(num_total_samples * (1 - 0.2)) # From validation_split

   val_dynamic_data = sequences_ex[val_start_index_ex:]
   val_actual_targets = targets_ex[val_start_index_ex:]
   val_actuals_for_plot = val_actual_targets
   
   # Package validation inputs in the [Static, Dynamic, Future] format
   # Static and Future are None then pass Dynamic only as 
   val_inputs_list = [val_dynamic_data] # this way , the flexible can handle 

   print("\nMaking predictions on the validation set...")
   val_predictions = tft_model_ex.predict(val_inputs_list, verbose=0)
   # val_predictions shape: (NumValSamples, Horizon, OutputDim)

   print(f"Validation predictions shape: {val_predictions.shape}")
   # For H=1, O=1, this will be (NumValSamples, 1, 1)
   print("Sample prediction (first validation sample):",
         val_predictions[0, 0, 0]) # Accessing the scalar value

   # --- Visualization ---
   # Align time axis for plotting the validation results
   # The target for sequence `i` corresponds to data point `time_ex[i + sequence_length_ex]`
   plot_val_time_axis_ex = time_ex[
       val_start_index_ex + sequence_length_ex : \
       val_start_index_ex + sequence_length_ex + len(val_actuals_for_plot)
       ]
   # Ensure the time axis matches the number of validation predictions/actuals
   num_plot_points_ex = min(len(plot_val_time_axis_ex), len(val_actuals_for_plot))

   plt.figure(figsize=(14, 7))
   # Plot a portion of original data for context
   context_end_idx_ex = val_start_index_ex + sequence_length_ex + \
                        num_plot_points_ex + forecast_horizon_ex
   plt.plot(time_ex[:context_end_idx_ex],
            amplitude_ex[:context_end_idx_ex],
            label='Original Data Context', alpha=0.6, color='lightblue')

   # Plot actuals from validation set (H=1, O=1)
   plt.plot(plot_val_time_axis_ex[:num_plot_points_ex],
            val_actuals_for_plot[:num_plot_points_ex, 0, 0],
            label=f'Actual Validation Data (H={forecast_horizon_ex})',
            linestyle=':', marker='o', color='blue')

   # Plot predictions on validation set (H=1, O=1)
   plt.plot(plot_val_time_axis_ex[:num_plot_points_ex],
            val_predictions[:num_plot_points_ex, 0, 0],
            label=f'Predicted Validation Data (H={forecast_horizon_ex})',
            marker='x', color='red')

   plt.title('Flexible TFT Point Forecast Exercise (Dynamic Input Only)')
   plt.xlabel('Time')
   plt.ylabel('Value')
   plt.legend()
   plt.grid(True)
   plt.tight_layout()
   # To save the figure for documentation:
   # fig_path_ex = os.path.join(
   # exercise_output_dir_basic,
   # "exercise_basic_tft_point_forecast.png"
   # )
   # plt.savefig(fig_path_ex)
   # print(f"\nPlot saved to {fig_path_ex}")
   plt.show() # Display plot

   print("\nBasic TFT point forecasting exercise complete.")

**Expected Plot 6.6:**

.. figure:: ../../images/exercise_basic_tft_point_forecast.png
   :alt: Basic TFT Point Forecast Exercise Results
   :align: center
   :width: 80%

   Visualization of the point forecast from the flexible
   `TemporalFusionTransformer` against actual validation data.

Discussion of Exercise
----------------------
In this exercise, you learned how to:
* Prepare simple time series data for a forecasting model using
  ``create_sequences``.
* Instantiate the flexible
  :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
  for a point forecasting task, specifying only the
  ``dynamic_input_dim`` and setting ``quantiles=None``.
* Correctly provide inputs to the model's ``fit`` and ``predict``
  methods as a list `[None, dynamic_array, None]` when only
  dynamic features are used, adhering to the expected
  `[static, dynamic, future]` order.
* Compile the model with an appropriate loss function (`mse`) for
  point forecasts.
* Train the model and generate predictions.

This forms the foundation for more complex forecasting tasks. You can
extend this by adding static and future known covariates, exploring
multi-step forecasting, or moving to probabilistic (quantile)
forecasts as shown in other examples.


