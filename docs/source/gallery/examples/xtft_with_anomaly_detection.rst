.. _example_xtft_anomaly:

=============================
XTFT with Anomaly Detection
=============================

This example demonstrates how to leverage the anomaly detection
features integrated within the :class:`~fusionlab.nn.XTFT` model.
Incorporating anomaly information during training can potentially
make the model more robust to irregularities and improve forecasting
performance, especially on noisy real-world data.

We will show two main approaches:

1.  **Using Pre-computed Scores:** (Detailed below) Calculate anomaly scores
    externally and incorporate them into the training loss using a
    combined loss function.
2.  **Using Prediction-Based Errors:** (Coming Soon) Configure XTFT to
    use the `'prediction_based'` strategy, deriving anomaly signals
    from prediction errors during training.

We adapt the setup from the :doc:`advanced_forecasting_xtft` example.

Common Setup Steps (Data Generation & Preprocessing)
-------------------------------------------------------

*(This section remains the same as before, explaining the assumed
variables: `train_inputs`, `val_inputs`, `y_train`, `y_val`, `scalers`,
`quantiles_to_predict`, `forecast_horizons`). We will repeat the code
here within Strategy 1 for completeness.*

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Strategy 1: Using Pre-computed Scores
----------------------------------------

This approach involves calculating anomaly scores based on some method
before training, and then using a special loss function during training
that incorporates these fixed scores alongside the primary forecasting
objective.

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Import standard libraries, ``XTFT`` model, relevant utilities like
``reshape_xtft_data`` and ``compute_anomaly_scores``, the
``AnomalyLoss`` component, and the ``combined_total_loss`` factory
function. Suppress warnings for clarity.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   import os
   import joblib # For potential scaler saving/loading

   # Assuming fusionlab components are importable
   from fusionlab.nn.transformers import XTFT
   from fusionlab.nn.utils import (
       reshape_xtft_data,
       compute_anomaly_scores # Needed to pre-calculate scores
   )
   from fusionlab.nn.losses import (
       combined_quantile_loss, # Base for comparison/quantiles
       combined_total_loss, # Used to combine quantile + anomaly
       prediction_based_loss # Used in Strategy 2
   )
   # AnomalyLoss component is needed by combined_total_loss
   from fusionlab.nn.components import AnomalyLoss

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # Setup output directory (optional)
   output_dir = "./xtft_anomaly_output"
   os.makedirs(output_dir, exist_ok=True)
   print("Setup complete.")


Step 2: Generate Synthetic Data (with Anomalies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We create multi-item time series data similar to the advanced XTFT
example, but intentionally inject some anomalous spikes/dips into the
'Sales' target variable for one of the items.

.. code-block:: python
   :linenos:

   n_items = 3
   n_timesteps = 36 # 3 years of monthly data
   date_rng = pd.date_range(start='2020-01-01', periods=n_timesteps, freq='MS')
   df_list = []

   for item_id in range(n_items):
       time = np.arange(n_timesteps)
       sales = (
           100 + item_id * 50 + time * (2 + item_id) +
           20 * np.sin(2 * np.pi * time / 12) +
           np.random.normal(0, 10, n_timesteps) # Base noise
       )
       # Inject anomalies for item_id 1
       if item_id == 1:
           sales[15] *= 2.5 # Positive spike
           sales[25] *= 0.2 # Negative dip
           print(f"Injected anomalies for ItemID {item_id}")

       temp = 15 + 10 * np.sin(2 * np.pi * (time % 12) / 12 + np.pi) + np.random.normal(0, 2)
       promo = np.random.randint(0, 2, n_timesteps)

       item_df = pd.DataFrame({
           'Date': date_rng, 'ItemID': item_id, 'Month': date_rng.month,
           'Temperature': temp, 'PlannedPromotion': promo, 'Sales': sales
       })
       item_df['PrevMonthSales'] = item_df['Sales'].shift(1)
       df_list.append(item_df)

   df = pd.concat(df_list).dropna().reset_index(drop=True)
   print(f"\nGenerated data shape (with anomalies): {df.shape}")


Step 3: Define Features & Scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Define the roles of each column (static, dynamic, future, target, etc.)
and scale the numerical features using `StandardScaler`. The scaler for
the target variable is stored to inverse-transform predictions later.

.. code-block:: python
   :linenos:

   target_col = 'Sales'
   dt_col = 'Date'
   static_cols = ['ItemID']
   dynamic_cols = ['Month', 'Temperature', 'PrevMonthSales']
   future_cols = ['PlannedPromotion', 'Month']
   spatial_cols = ['ItemID']
   scalers = {}
   num_cols_to_scale = ['Temperature', 'PrevMonthSales', 'Sales']

   for col in num_cols_to_scale:
       scaler = StandardScaler()
       df[col] = scaler.fit_transform(df[[col]])
       scalers[col] = scaler # Store scaler
       print(f"Scaled column: {col}")


Step 4: Prepare Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use :func:`~fusionlab.nn.utils.reshape_xtft_data` to transform the
DataFrame into the sequence arrays (static, dynamic, future, target)
required by XTFT.

.. code-block:: python
   :linenos:

   time_steps = 12 # Lookback window
   forecast_horizons = 6 # Prediction horizon

   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df, dt_col=dt_col, target_col=target_col,
       dynamic_cols=dynamic_cols, static_cols=static_cols,
       future_cols=future_cols, spatial_cols=spatial_cols,
       time_steps=time_steps, forecast_horizons=forecast_horizons,
       verbose=1 # Show shapes
   )


Step 5: Pre-compute Anomaly Scores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For the 'from_config' strategy (or equivalent using combined loss),
we need anomaly scores *before* training. We use
:func:`~fusionlab.nn.utils.compute_anomaly_scores` on the target
sequence data. Here, we use the 'statistical' method (squared
Z-score) for simplicity.

.. code-block:: python
   :linenos:

   print("\nCalculating anomaly scores...")
   anomaly_scores_array = compute_anomaly_scores(
       y_true=target_data, # Use the ground truth target sequences
       method='statistical',
       verbose=0
   )
   print(f"Computed anomaly scores shape: {anomaly_scores_array.shape}")
   # Should match target_data shape: (NumSequences, Horizon, 1)


Step 6: Train/Validation Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Split *all* the generated arrays (static, dynamic, future, target,
*and* anomaly scores) into training and validation sets using a
chronological split. Package the model inputs into lists.

.. code-block:: python
   :linenos:

   val_split_fraction = 0.2
   n_samples = static_data.shape[0]
   split_idx = int(n_samples * (1 - val_split_fraction))

   # Split all arrays
   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   X_train_future, X_val_future = future_data[:split_idx], future_data[split_idx:]
   y_train, y_val = target_data[:split_idx], target_data[split_idx:]
   anomaly_scores_train = anomaly_scores_array[:split_idx]
   anomaly_scores_val = anomaly_scores_array[split_idx:] # For potential validation

   # Package inputs
   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future]

   print("\nData split into Train/Validation sets.")
   print(f"  Train samples: {split_idx}")
   print(f"  Validation samples: {n_samples - split_idx}")
   print(f"  Anomaly scores train shape: {anomaly_scores_train.shape}")


Step 7: Define XTFT Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate the :class:`~fusionlab.nn.XTFT` model. Note that when using
`combined_total_loss`, the `anomaly_detection_strategy` parameter
in the model itself might not be strictly necessary, as the loss
handles the anomaly component. We still pass `anomaly_loss_weight`
as it's needed by the `AnomalyLoss` component.

.. code-block:: python
   :linenos:

   quantiles_to_predict = [0.1, 0.5, 0.9]
   anomaly_weight = 0.05 # Weight for anomaly loss component

   model = XTFT(
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1],
       forecast_horizon=forecast_horizons,
       quantiles=quantiles_to_predict,
       # Example Hyperparameters
       embed_dim=16, lstm_units=32, attention_units=16,
       hidden_units=32, num_heads=4, dropout_rate=0.1,
       max_window_size=time_steps, memory_size=50,
       # Pass weight for potential internal use or logging
       anomaly_loss_weight=anomaly_weight,
       # anomaly_detection_strategy='from_config', # Not strictly needed here
       # anomaly_config=None # Scores passed via loss function below
   )
   print("\nXTFT model instantiated.")


Step 8: Define Combined Loss Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create the combined loss function using
:func:`~fusionlab.nn.losses.combined_total_loss`. This requires an
instance of the :class:`~fusionlab.nn.components.AnomalyLoss` layer
(configured with the desired weight) and the **fixed training anomaly scores**
computed in Step 5.

.. code-block:: python
   :linenos:

   # 1. Create the AnomalyLoss component
   anomaly_loss_layer = AnomalyLoss(weight=anomaly_weight)

   # 2. Create the combined loss, capturing the training scores
   # Ensure scores are a TensorFlow constant for graph compatibility
   combined_loss = combined_total_loss(
       quantiles=quantiles_to_predict,
       anomaly_layer=anomaly_loss_layer,
       anomaly_scores=tf.constant(anomaly_scores_train, dtype=tf.float32)
   )
   print("Combined quantile + pre-computed anomaly loss defined.")


Step 9: Compile Model
~~~~~~~~~~~~~~~~~~~~~~~
Compile the XTFT model using the combined loss function created in the
previous step and an optimizer like Adam.

.. code-block:: python
   :linenos:

   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                 loss=combined_loss)
   print("XTFT model compiled with combined loss.")


Step 10: Train Model
~~~~~~~~~~~~~~~~~~~~~~
Train the model using `.fit()`. The optimizer will minimize the
combined loss, balancing quantile prediction accuracy and alignment
with the provided anomaly score signal.

.. code-block:: python
   :linenos:

   print("Starting XTFT model training with anomaly objective...")
   history = model.fit(
       train_inputs,
       y_train, # Target shape (B, H, 1)
       validation_data=(val_inputs, y_val),
       epochs=5, # Increase for real training
       batch_size=16,
       verbose=1 # Show epoch progress
   )
   print("Training finished.")


Step 11: Prediction & Visualization (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate predictions and inverse-transform them using the saved scaler.
Visualize the results for one item, showing the prediction interval.
The process is the same as the advanced XTFT example.

.. code-block:: python
   :linenos:

   # (Code for prediction, inverse transform, and visualization
   #  would go here, similar to the advanced_forecasting_xtft example)
   print("\nPrediction and visualization steps would follow.")
   # Example:
   # predictions_scaled = model.predict(val_inputs)
   # ... inverse transform predictions_scaled and y_val ...
   # ... plot actuals vs predicted quantiles for one item ...


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Strategy 2: Using Prediction-Based Errors
-------------------------------------------

This approach configures the model and loss function to derive anomaly
signals directly from prediction errors during training. Anomalies are
implicitly defined as points or sequences where the model's own
predictions deviate significantly from the true values.

*(We assume the data preparation steps (1-5 from Strategy 1, excluding
anomaly score calculation/split) have been run, providing `train_inputs`,
`val_inputs`, `y_train`, `y_val`, `scalers`, `quantiles_to_predict`,
`forecast_horizons`, `time_steps`)*

Step 6a: Define XTFT Model (Prediction-Based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate the :class:`~fusionlab.nn.XTFT` model, crucially setting
the ``anomaly_detection_strategy`` parameter to ``'prediction_based'``.
Also provide the ``anomaly_loss_weight`` to control the balance
between the forecasting loss and the anomaly (prediction error) penalty.

.. code-block:: python
   :linenos:

   # (Assuming train_inputs, val_inputs, y_train, y_val etc. exist)
   # (Assuming quantiles_to_predict, forecast_horizons, time_steps exist)
   # (Assuming X_train_static, etc. exist for getting input dims)

   print("\n--- Configuring for 'prediction_based' Strategy ---")

   anomaly_weight_pb = 0.05 # Define weight for this strategy

   model_pred_based = XTFT(
       static_input_dim=X_train_static.shape[-1],
       dynamic_input_dim=X_train_dynamic.shape[-1],
       future_input_dim=X_train_future.shape[-1],
       forecast_horizon=forecast_horizons,
       quantiles=quantiles_to_predict, # Can still predict quantiles
       # Example Hyperparameters
       embed_dim=16, lstm_units=32, attention_units=16,
       hidden_units=32, num_heads=4, dropout_rate=0.1,
       max_window_size=time_steps, memory_size=50,
       # *** Set the strategy explicitly ***
       anomaly_detection_strategy='prediction_based',
       anomaly_loss_weight=anomaly_weight_pb # Pass weight
   )
   print("XTFT model instantiated with strategy='prediction_based'.")


Step 7a: Compile Model with Prediction-Based Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compile the model using the loss function generated by the
:func:`~fusionlab.nn.losses.prediction_based_loss` factory. This
factory creates a loss function that internally calculates both the
primary prediction loss (quantile or MSE) and an anomaly loss term
based on the prediction error magnitude, combining them using the
provided `anomaly_loss_weight`.

.. code-block:: python
   :linenos:

   # Create the combined loss using the factory
   loss_pred_based = prediction_based_loss(
       quantiles=quantiles_to_predict, # Base loss uses quantiles
       anomaly_loss_weight=anomaly_weight_pb # Weight for error term
   )

   # Compile the model with this specific loss
   model_pred_based.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
       loss=loss_pred_based
   )
   print("Model compiled with prediction_based_loss.")


Step 8a: Train the Model (Prediction-Based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Train the model using the standard `.fit()` method. The combined loss
calculation (prediction + anomaly penalty) happens automatically within
the `prediction_based_loss` function called by Keras during each training
step. No external anomaly scores need to be provided. The model learns
to simultaneously make accurate forecasts and minimize large prediction
errors.

.. code-block:: python
   :linenos:

   print("\nStarting model training (Strategy 2)...")
   # Train using the standard fit method
   # The custom loss handles the combined objective
   history_pred_based = model_pred_based.fit(
       train_inputs, # [Static, Dynamic, Future]
       y_train,      # Target shape (B, H, 1)
       validation_data=(val_inputs, y_val),
       epochs=5,     # Increase for real training
       batch_size=16,
       verbose=1     # Show epoch progress
   )
   print("Training finished.")


Step 9a: Prediction & Visualization (Prediction-Based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generating predictions and visualizing the results after training is
the same as for Strategy 1. The anomaly detection strategy primarily
affects the training objective and the learned model weights, not the
prediction process itself.

.. code-block:: python
   :linenos:

   print("\nMaking predictions (Strategy 2 model)...")
   predictions_scaled_pb = model_pred_based.predict(val_inputs, verbose=0)
   print(f"Prediction output shape: {predictions_scaled_pb.shape}")

   # --- Inverse Transform (Example) ---
   # (Requires 'scalers' dictionary from data prep step)
   num_val_samples = X_val_static.shape[0]
   num_quantiles = len(quantiles_to_predict)
   output_dim = 1 # From config

   pred_reshaped_pb = predictions_scaled_pb.reshape(-1, num_quantiles)
   predictions_inv_pb = scalers['Sales'].inverse_transform(pred_reshaped_pb)
   predictions_final_pb = predictions_inv_pb.reshape(
       num_val_samples, forecast_horizons, num_quantiles
   )
   # Inverse transform y_val for comparison
   y_val_reshaped = y_val.reshape(-1, output_dim)
   y_val_inv = scalers['Sales'].inverse_transform(y_val_reshaped)
   y_val_final = y_val_inv.reshape(num_val_samples, forecast_horizons, output_dim)

   print("Predictions inverse transformed.")

   # --- Visualization (Example for one item) ---
   # (Plotting code similar to Strategy 1, using predictions_final_pb
   #  and y_val_final. Omitted here for brevity, see previous example.)
   print("Visualization would show prediction intervals.")


.. topic:: Explanations Summary

   **Common Setup (Steps 1-5):**
   These steps involve generating synthetic data (optionally with
   injected anomalies), defining feature roles, scaling numerical
   features, preparing sequences using
   :func:`~fusionlab.utils.ts_utils.reshape_xtft_data`, and splitting
   data.

   **Strategy 1: Using Pre-computed Scores (`from_config` Logic)**
   This strategy calculates anomaly scores *before* training (Step 5)
   using a method like `'statistical'`. The model is then compiled
   (Step 8) with a loss created by
   :func:`~fusionlab.nn.losses.combined_total_loss`, which takes these
   *fixed* scores as input. The model training (Step 10) minimizes a
   combination of forecasting loss and a penalty based on these fixed
   external scores.

   **Strategy 2: Using Prediction-Based Errors (`prediction_based`)**
   This strategy defines anomalies based on the model's own prediction
   errors *during* training.
   * The model is instantiated with
     ``anomaly_detection_strategy='prediction_based'`` (Step 6a).
   * It's compiled using the loss from
     :func:`~fusionlab.nn.losses.prediction_based_loss` (Step 7a).
   * During training (Step 8a), this loss function automatically
     calculates both the forecasting loss (e.g., quantile) and an
     anomaly penalty based on the current prediction error, combining
     them using ``anomaly_loss_weight``. No external scores are needed.

   **Choosing a Strategy:**
   * Use **Strategy 1** when you have reliable external anomaly scores
     or want to define anomalies based on specific domain knowledge.
   * Use **Strategy 2** when you want the model to implicitly learn
     to identify anomalies as points/sequences where its own predictions
     are poor, making it sensitive to unexpected deviations.