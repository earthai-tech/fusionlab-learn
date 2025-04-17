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

1.  **Using Pre-computed Scores:** We'll first calculate anomaly scores
    externally (using :func:`~fusionlab.nn.utils.compute_anomaly_scores`)
    and then incorporate them into the training loss using a combined
    loss function (similar to the `'from_config'` strategy logic).
2.  **Using Prediction-Based Errors:** We'll configure XTFT to use the
    `'prediction_based'` strategy, where the anomaly signal is derived
    directly from prediction errors during training via a specialized
    loss function.

We adapt the setup from the :doc:`advanced_forecasting_xtft` example.


Common Setup Steps (Data Generation & Preprocessing)
----------------------------------------------------

The initial steps for data generation, feature definition, scaling,
sequence preparation, and splitting are common to both strategies. We
assume these steps have been performed, resulting in the following
variables available for use in the subsequent strategy examples:

* `train_inputs`: List `[X_train_static, X_train_dynamic, X_train_future]`
* `val_inputs`: List `[X_val_static, X_val_dynamic, X_val_future]`
* `y_train`, `y_val`: Target arrays for training and validation.
* `scalers`: Dictionary containing fitted scalers (e.g., `scalers['Sales']`).
* `quantiles_to_predict`: List of target quantiles (e.g., `[0.1, 0.5, 0.9]`).
* `forecast_horizons`: Number of steps to predict.
* `anomaly_scores_train`, `anomaly_scores_val`: (Optional) Pre-calculated
    anomaly scores aligned with train/val sets, needed for Strategy 1.

*(Refer to the full code in Strategy 1 below for an example of generating
these variables)*.

.. raw:: html

    <hr>
    
Code Example (Strategy 1: Using Pre-computed Scores)
----------------------------------------------------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler

   # Assuming fusionlab components are importable
   from fusionlab.nn.transformers import XTFT
   from fusionlab.nn.utils import (
       reshape_xtft_data,
       compute_anomaly_scores # Needed to pre-calculate scores
   )
   from fusionlab.nn.losses import (
       combined_quantile_loss,
       combined_total_loss, # Used to combine quantile + anomaly
       prediction_based_loss # Alternative for strategy 2
   )
   from fusionlab.nn.components import AnomalyLoss # Needed for combined_total_loss

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # 1. Generate Synthetic Data (with some anomalies)
   # -------------------------------------------------
   n_items = 3
   n_timesteps = 36
   date_rng = pd.date_range(start='2020-01-01', periods=n_timesteps, freq='MS')
   df_list = []

   for item_id in range(n_items):
       time = np.arange(n_timesteps)
       sales = (
           100 + item_id * 50 + time * (2 + item_id) +
           20 * np.sin(2 * np.pi * time / 12) +
           np.random.normal(0, 10, n_timesteps)
       )
       # Inject some anomalies (e.g., sudden spikes/dips)
       if item_id == 1:
           sales[15] *= 2.5 # Spike
           sales[25] *= 0.2 # Dip
       temp = 15 + 10 * np.sin(2 * np.pi * (time % 12) / 12 + np.pi) + np.random.normal(0, 2)
       promo = np.random.randint(0, 2, n_timesteps)

       item_df = pd.DataFrame({
           'Date': date_rng, 'ItemID': item_id, 'Month': date_rng.month,
           'Temperature': temp, 'PlannedPromotion': promo, 'Sales': sales
       })
       item_df['PrevMonthSales'] = item_df['Sales'].shift(1)
       df_list.append(item_df)

   df = pd.concat(df_list).dropna().reset_index(drop=True)
   print("Generated data shape:", df.shape)

   # 2. Define Features & Scale (same as previous example)
   # ------------------------------------------------------
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
       scalers[col] = scaler
   print("Numerical features scaled.")

   # 3. Prepare Sequences (same as previous example)
   # -------------------------------------------------
   time_steps = 12
   forecast_horizons = 6
   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df, dt_col=dt_col, target_col=target_col,
       dynamic_cols=dynamic_cols, static_cols=static_cols,
       future_cols=future_cols, spatial_cols=spatial_cols,
       time_steps=time_steps, forecast_horizons=forecast_horizons,
       verbose=0
   )
   print(f"Sequence shapes: Target={target_data.shape}") # e.g., (N, 6, 1)

   # 4. Pre-compute Anomaly Scores ('from_config' strategy)
   # ------------------------------------------------------
   # Calculate scores based on the *target* data itself before splitting
   # Using statistical method for simplicity here
   # Output shape should match target_data: (NumSequences, Horizon, 1)
   anomaly_scores_array = compute_anomaly_scores(
       target_data, method='statistical', verbose=0
   )
   print(f"Computed anomaly scores shape: {anomaly_scores_array.shape}")

   # 5. Train/Validation Split (include anomaly scores)
   # --------------------------------------------------
   val_split_fraction = 0.2
   n_samples = static_data.shape[0]
   split_idx = int(n_samples * (1 - val_split_fraction))

   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   X_train_future, X_val_future = future_data[:split_idx], future_data[split_idx:]
   y_train, y_val = target_data[:split_idx], target_data[split_idx:]
   # Split anomaly scores as well
   anomaly_scores_train = anomaly_scores_array[:split_idx]
   anomaly_scores_val = anomaly_scores_array[split_idx:] # Used for potential eval

   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future]
   print("Data split into Train/Validation sets.")

   # 6. Define XTFT Model (with anomaly settings)
   # --------------------------------------------
   quantiles_to_predict = [0.1, 0.5, 0.9]
   anomaly_weight = 0.05 # Weight for the anomaly loss component

   # When using combined_total_loss, strategy isn't set in model __init__
   # but implicitly handled by the loss function choice.
   # If using feature_based, set strategy here.
   model = XTFT(
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1],
       forecast_horizon=forecast_horizons,
       quantiles=quantiles_to_predict,
       embed_dim=16, lstm_units=32, attention_units=16,
       hidden_units=32, num_heads=4, dropout_rate=0.1,
       max_window_size=time_steps, memory_size=50,
       # ** Anomaly specific params **
       # 'anomaly_detection_strategy': 'from_config', # Not needed if using combined_total_loss
       anomaly_loss_weight=anomaly_weight, # Can still pass for potential internal use
       # 'anomaly_config': {'anomaly_scores': anomaly_scores_train}, # Pass scores if model uses it directly
   )

   # 7. Compile with Combined Quantile + Anomaly Loss
   # ------------------------------------------------
   # Create the anomaly loss layer instance
   anomaly_loss_layer = AnomalyLoss(weight=anomaly_weight)

   # Create the combined loss function, providing the fixed training scores
   # Note: combined_total_loss captures anomaly_scores_train at definition time
   combined_loss = combined_total_loss(
       quantiles=quantiles_to_predict,
       anomaly_layer=anomaly_loss_layer,
       anomaly_scores=tf.constant(anomaly_scores_train, dtype=tf.float32) # Must be TF tensor
   )

   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                 loss=combined_loss)
   print("XTFT model compiled with combined quantile and anomaly loss.")

   # 8. Train the Model
   # ------------------
   print("Starting XTFT model training with anomaly objective...")
   history = model.fit(
       train_inputs,
       y_train,
       validation_data=(val_inputs, y_val),
       epochs=5, # Increase for real training
       batch_size=16,
       verbose=0
   )
   print("Training finished.")
   print(f"Final validation loss (combined): {history.history['val_loss'][-1]:.4f}")

   # 9. Prediction & Visualization (similar to previous example)
   # ---------------------------------------------------------
   # (Prediction uses the trained model; visualization remains the same)
   # ... add prediction and visualization code from previous example ...


   # --- Alternative: Strategy 2 ('prediction_based') ---
   print("\n--- Example Setup for 'prediction_based' Strategy ---")
   # Note: Requires modification of model instantiation and compilation

   # 6a. Define XTFT Model with 'prediction_based' strategy
   # model_pred_based = XTFT(
   #     # ... other parameters same as above ...
   #     anomaly_detection_strategy='prediction_based',
   #     anomaly_loss_weight=0.05 # Control weight
   # )

   # 7a. Compile with prediction_based_loss factory
   # loss_pred_based = prediction_based_loss(
   #     quantiles=quantiles_to_predict, # Use quantiles for prediction part
   #     anomaly_loss_weight=0.05
   # )
   # model_pred_based.compile(optimizer='adam', loss=loss_pred_based)
   # print("Model configured for 'prediction_based' anomaly detection.")

   # 8a. Train the Model (uses custom train_step internally)
   # history_pred_based = model_pred_based.fit(...) # Train as usual


.. topic:: Explanations

   1.  **Data Generation:** We use the multi-item setup but inject
       some artificial anomalies (a spike and a dip) into the 'Sales'
       data for one item to simulate irregularities.
   2.  **Features & Scaling:** Definitions and scaling remain the same
       as the previous XTFT example.
   3.  **Sequence Preparation:** We use `reshape_xtft_data` as before to
       get `static_data`, `dynamic_data`, `future_data`, `target_data`.
   4.  **Anomaly Score Calculation (`from_config` strategy):**
       * We demonstrate the *concept* behind the `'from_config'`
         strategy by pre-calculating anomaly scores *before* training.
       * :func:`~fusionlab.nn.utils.compute_anomaly_scores` is called on
         the `target_data` (before train/val split) using the
         `'statistical'` method (deviation from mean). This generates
         an array `anomaly_scores_array` aligned with the target sequences.
         *(Note: In practice, scores might be derived from domain knowledge,
         other models, or different features).*
   5.  **Train/Validation Split:** We split *all* generated arrays,
       including the `anomaly_scores_array`, maintaining the chronological
       order or random state consistency.
   6.  **Model Definition (`from_config` strategy):**
       * `XTFT` is instantiated similarly to before.
       * The `anomaly_detection_strategy` parameter might not be strictly
         needed here if we use `combined_total_loss`, as the loss function
         itself handles the anomaly component based on the provided scores.
         However, setting `anomaly_loss_weight` can still be useful.
       * Passing scores via `anomaly_config` is another way XTFT might
         be designed to handle this, depending on its internal implementation.
   7.  **Model Compilation (`from_config` strategy):**
       * This is the key step for integrating pre-computed scores.
       * We instantiate the :class:`~fusionlab.nn.components.AnomalyLoss`
         layer, passing the desired `weight`.
       * We use the :func:`~fusionlab.nn.losses.combined_total_loss`
         factory function. It takes the `quantiles`, the instantiated
         `AnomalyLoss` layer, and importantly, the **training anomaly scores**
         (`anomaly_scores_train`, converted to a `tf.constant`).
       * This `combined_loss` now calculates both the quantile loss (on
         `y_true`, `y_pred`) and the anomaly loss (based *only* on the
         provided `anomaly_scores_train` captured when the loss was created)
         and sums them.
       * The model is compiled with this combined loss.
   8.  **Model Training:** Training proceeds as usual with `.fit()`. The
       optimizer now minimizes the combined objective, encouraging both
       accurate quantile predictions and alignment with the provided
       anomaly score signal (e.g., implicitly penalizing predictions
       that would lead to generating features similar to those associated
       with high pre-computed anomaly scores, although the link is
       indirect here as scores are fixed).
   9.  **Alternative (`prediction_based` strategy):**
       * The commented-out section shows the alternative setup.
       * You would set `anomaly_detection_strategy='prediction_based'`
         when creating the `XTFT` model.
       * You would compile using the
         :func:`~fusionlab.nn.losses.prediction_based_loss` factory,
         which returns a loss function that internally calculates both
         quantile/MSE loss and an anomaly term based on prediction errors.
       * The model's custom `train_step` (if implemented in `XTFT` for this
         strategy) would handle the combined loss calculation.

This example illustrates how anomaly information, either pre-computed or
derived from prediction errors, can be integrated into the XTFT training
process to potentially improve model robustness.


.. raw:: html

    <hr>
    
Strategy 2: Using Prediction-Based Errors
-----------------------------------------

This approach configures the model and loss function to derive anomaly
signals directly from prediction errors during training.

.. code-block:: python
   :linenos:

   # --- Strategy 2 Code Example ---

   # (Steps 1-5 remain the same: Imports, Data Gen, Feature Def, Scaling,
   #  Sequence Prep, Train/Val Split are done as above, resulting in
   #  train_inputs, val_inputs, y_train, y_val, scalers,
   #  quantiles_to_predict, forecast_horizons)

   print("\n--- Running Example for 'prediction_based' Strategy ---")

   # 6a. Define XTFT Model with 'prediction_based' strategy
   # ------------------------------------------------------
   anomaly_weight_pb = 0.05 # Define weight for this strategy

   model_pred_based = XTFT(
       static_input_dim=X_train_static.shape[-1],
       dynamic_input_dim=X_train_dynamic.shape[-1],
       future_input_dim=X_train_future.shape[-1],
       forecast_horizon=forecast_horizons,
       quantiles=quantiles_to_predict, # Can still predict quantiles
       embed_dim=16, lstm_units=32, attention_units=16,
       hidden_units=32, num_heads=4, dropout_rate=0.1,
       max_window_size=time_steps, memory_size=50,
       # *** Set the strategy explicitly ***
       anomaly_detection_strategy='prediction_based',
       anomaly_loss_weight=anomaly_weight_pb # Pass weight
   )
   print("XTFT model instantiated with strategy='prediction_based'.")

   # 7a. Compile with prediction_based_loss factory
   # ----------------------------------------------
   # Use the factory to create the combined loss
   loss_pred_based = prediction_based_loss(
       quantiles=quantiles_to_predict, # Base loss uses quantiles
       anomaly_loss_weight=anomaly_weight_pb # Weight for error term
   )
   model_pred_based.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
       loss=loss_pred_based
   )
   print("Model compiled with prediction_based_loss.")

   # 8a. Train the Model (Strategy 2)
   # --------------------------------
   print("Starting model training (Strategy 2)...")
   history_pred_based = model_pred_based.fit(
       train_inputs,
       y_train,
       validation_data=(val_inputs, y_val),
       epochs=5,
       batch_size=16,
       verbose=0
   )
   print("Training finished.")
   print(f"Final validation loss (prediction-based combined): "
         f"{history_pred_based.history['val_loss'][-1]:.4f}")

   # 9a. Prediction & Visualization (Strategy 2)
   # -------------------------------------------
   # Prediction/visualization code is the same as Strategy 1,
   # just uses model_pred_based
   print("\nMaking predictions (Strategy 2)...")
   predictions_scaled_pb = model_pred_based.predict(val_inputs, verbose=0)
   # ... (Add inverse scaling and plotting code similar to previous example,
   #      using predictions_scaled_pb and maybe distinct plot titles/colors) ...
   print("Prediction and visualization for Strategy 2 would follow here.")


.. topic:: Explanations

   **Common Setup (Steps 1-5):**
   These steps involve generating synthetic data (optionally with
   injected anomalies), defining feature roles (static, dynamic, future),
   scaling numerical features (saving scalers), preparing sequences with
   :func:`~fusionlab.utils.ts_utils.reshape_xtft_data`, and splitting the
   sequence arrays into training and validation sets. For Strategy 1,
   anomaly scores are also computed and split here.

   **Strategy 1: Using Pre-computed Scores (`from_config` Logic)**

   6.  **Anomaly Score Calculation:** We demonstrate pre-calculating
       scores using :func:`~fusionlab.nn.utils.compute_anomaly_scores`
       on the target data *before* splitting. These scores (`anomaly_scores_train`)
       represent an external assessment of anomaly likelihood for each
       target point.
   7.  **Model Definition:** `XTFT` is instantiated. The key is how the loss
       will be defined in the next step. The `anomaly_loss_weight` parameter
       is passed to the :class:`~fusionlab.nn.components.AnomalyLoss` layer.
   8.  **Model Compilation:** The :func:`~fusionlab.nn.losses.combined_total_loss`
       factory function is used. It requires the target `quantiles`, an
       instance of :class:`~fusionlab.nn.components.AnomalyLoss` (configured
       with its weight), and the pre-computed `anomaly_scores_train` tensor.
       This creates a single loss function that calculates both quantile loss
       (based on `y_true`, `y_pred`) and anomaly loss (based *only* on the
       provided `anomaly_scores_train`).
   9.  **Model Training:** The model minimizes the combined loss, learning
       to predict quantiles accurately while implicitly being penalized based
       on the fixed anomaly scores associated with the training targets.

   **Strategy 2: Using Prediction-Based Errors (`prediction_based`)**

   6a. **Model Definition:** `XTFT` is instantiated with the crucial parameter
       `anomaly_detection_strategy='prediction_based'`. The
       `anomaly_loss_weight` is also provided here to control the balance.
   7a. **Model Compilation:** We use the
       :func:`~fusionlab.nn.losses.prediction_based_loss` factory. This
       function takes the `quantiles` (to define the base prediction loss)
       and the `anomaly_loss_weight`. It returns a loss function that
       *internally* computes both the prediction loss (e.g., quantile loss)
       and an anomaly term based on the magnitude of the prediction error
       ($|y_{true} - y_{pred}|$ or similar), then sums them with the given weight.
   8a. **Model Training:** Training uses the standard `.fit()` method. The
       combined loss calculation happens transparently within the custom
       `prediction_based_loss` function called by Keras during each training step.
       No external anomaly scores are needed.
   9a. **Prediction/Visualization:** The process of making predictions and
       visualizing them after training is identical for both strategies, as
       the anomaly detection component primarily affects the training objective.

   **Choosing a Strategy:**
   * Use the **pre-computed score** approach (Strategy 1 / `'from_config'`)
     when you have reliable external anomaly scores or want to define anomalies
     based on specific features or domain knowledge calculated beforehand.
   * Use the **prediction-based** approach (Strategy 2) when you want the
     model to implicitly identify anomalies as points where its own predictions
     deviate significantly from the actual values, making it sensitive to
     unexpected deviations or periods where the model struggles.