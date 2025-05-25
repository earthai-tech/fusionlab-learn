.. _exercise_tft_required_inputs:

=============================================================
Exercise: Forecasting with Stricter TFT (All Inputs Required)
=============================================================

Welcome to this exercise on using the stricter version of the
Temporal Fusion Transformer, :class:`~fusionlab.nn.transformers.TFT`,
available in ``fusionlab-learn``. This model implementation requires
that **static**, **dynamic (past observed)**, and **known future**
features are all provided as inputs.

We will perform a single-step point forecast to illustrate the specific
data preparation and model interaction for this TFT variant.

**Learning Objectives:**

* Generate synthetic multi-item time series data with distinct static,
  dynamic, and future features.
* Understand how to define feature roles, numerically encode
  categorical static features (like item identifiers), and scale
  numerical data.
* Utilize the :func:`~fusionlab.nn.utils.reshape_xtft_data` utility
  to prepare the three separate input arrays (static, dynamic, future)
  and targets.
* Correctly structure the input list `[static, dynamic, future]` for
  training and prediction with the stricter `TFT`.
* Define, compile, and train the `TFT` model.
* Make predictions and visualize the results, including inverse
  transformation of scaled values.

Let's get started!

Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and its common dependencies
installed. For visualizations, `matplotlib` is also needed.

.. code-block:: bash

   pip install fusionlab-learn matplotlib scikit-learn

---

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~
First, we import all necessary libraries.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   from sklearn.model_selection import train_test_split
   import warnings
   import os

   # FusionLab imports
   from fusionlab.nn.transformers import TFT # The stricter TFT class
   from fusionlab.nn.utils import reshape_xtft_data
   # Import for Keras to recognize custom loss if model was saved with it
   from fusionlab.nn.losses import combined_quantile_loss

   # Suppress warnings and TF logs for cleaner output
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   if hasattr(tf, 'autograph'): # Check for autograph availability
       tf.autograph.set_verbosity(0)

   # Directory for saving any output images from this exercise
   exercise_output_dir_tft_strict = "./tft_strict_exercise_outputs"
   os.makedirs(exercise_output_dir_tft_strict, exist_ok=True)

   print("Libraries imported and setup complete for stricter TFT exercise.")

**Expected Output 1.1:**

.. code-block:: text

   Libraries imported and setup complete for stricter TFT exercise.

Step 2: Generate Synthetic Multi-Feature Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We'll create a synthetic dataset for multiple items. Each item will
have:
* Static features: `ItemID_str` (a string identifier) and `Category` (a numerical category).
* Dynamic past features: `DayOfWeek` and `ValueLag1` (lagged target).
* Known future features: `FutureEvent` (a binary indicator) and `DayOfWeek`.
* Target: `Value`.

.. code-block:: python
   :linenos:

   n_items_ex_strict = 2
   n_timesteps_per_item_ex_strict = 50
   rng_seed_ex_strict = 42
   np.random.seed(rng_seed_ex_strict)
   tf.random.set_seed(rng_seed_ex_strict)

   date_rng_ex_strict = pd.date_range(
       start='2021-01-01',
       periods=n_timesteps_per_item_ex_strict, freq='D'
       )
   df_list_ex_strict = []

   for item_id_num in range(n_items_ex_strict):
       time_idx = np.arange(n_timesteps_per_item_ex_strict)
       value = (50 + item_id_num * 10 + time_idx * 0.5 +
                np.sin(time_idx / 7) * 5 + # Weekly seasonality
                np.random.normal(0, 2, n_timesteps_per_item_ex_strict))
       static_category_val = item_id_num + 1
       future_event_val = (date_rng_ex_strict.dayofweek >= 5).astype(int) # Weekend

       item_df = pd.DataFrame({
           'Date': date_rng_ex_strict,
           'ItemID_str': f'item_{item_id_num}', # String ID
           'Category': static_category_val,    # Numerical static
           'DayOfWeek': date_rng_ex_strict.dayofweek,
           'FutureEvent': future_event_val,
           'Value': value
       })
       item_df['ValueLag1'] = item_df['Value'].shift(1)
       df_list_ex_strict.append(item_df)

   df_raw_ex_strict = pd.concat(
       df_list_ex_strict).dropna().reset_index(drop=True)
   print(f"Generated raw data shape: {df_raw_ex_strict.shape}")
   print("Sample of generated data:")
   print(df_raw_ex_strict.head(3))

**Expected Output 2.2:**

.. code-block:: text

   Generated raw data shape: (98, 7)
   Sample of generated data:
          Date ItemID_str  Category  DayOfWeek  FutureEvent      Value  ValueLag1
   0 2021-01-02     item_0         1          5            1  50.935330  50.993428
   1 2021-01-03     item_0         1          6            1  53.704591  50.935330
   2 2021-01-04     item_0         1          0            0  56.623919  53.704591

Step 3: Define Features, Encode Static, and Scale Numerics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We assign columns to their roles. Since the stricter `TFT` model (and
`reshape_xtft_data`) expects numerical inputs for static features,
we'll LabelEncode the string-based `ItemID_str`. Then, we scale relevant
numerical features.

.. code-block:: python
   :linenos:

   target_col_strict = 'Value'
   dt_col_strict = 'Date'

   # Initial column definitions
   static_cols_def_strict = ['ItemID_str', 'Category']
   dynamic_cols_def_strict = ['DayOfWeek', 'ValueLag1']
   future_cols_def_strict = ['FutureEvent', 'DayOfWeek']
   # For reshape_xtft_data, spatial_cols are used for grouping
   spatial_cols_for_grouping = ['ItemID_str']

   df_processed_strict = df_raw_ex_strict.copy()

   # --- Encode ItemID_str (Categorical Static Feature) ---
   le_item_id_ex_strict = LabelEncoder()
   df_processed_strict['ItemID_encoded'] = \
       le_item_id_ex_strict.fit_transform(df_processed_strict['ItemID_str'])
   print(f"\nEncoded 'ItemID_str' into 'ItemID_encoded'. "
         f"Classes: {le_item_id_ex_strict.classes_}")

   # --- Update static_cols to use the encoded version for the model ---
   static_cols_for_model_strict = ['ItemID_encoded', 'Category']
   # For reshape_xtft_data, grouping can still use original string ID,
   # or you can group by the encoded ID if preferred.
   # If grouping by encoded, ensure it's in df_processed_strict.
   # Here, we'll pass the original string ItemID for grouping to reshape,
   # but use ItemID_encoded as a static *feature*.

   # --- Scale Numerical Features ---
   scaler_strict = StandardScaler()
   num_cols_to_scale_strict = ['Value', 'ValueLag1']
   # Ensure columns exist
   num_cols_to_scale_strict = [
       c for c in num_cols_to_scale_strict if c in df_processed_strict.columns
       ]
   if num_cols_to_scale_strict:
       df_processed_strict[num_cols_to_scale_strict] = \
           scaler_strict.fit_transform(
               df_processed_strict[num_cols_to_scale_strict]
               )
       print("\nNumerical features scaled.")
   else:
       print("\nNo numerical features found for scaling.")

**Expected Output 3.3:**

.. code-block:: text

   Encoded 'ItemID_str' into 'ItemID_encoded'. Classes: ['item_0' 'item_1']

   Numerical features scaled.

Step 4: Prepare Sequences with `reshape_xtft_data`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use :func:`~fusionlab.nn.utils.reshape_xtft_data` to transform the
DataFrame. It will use `spatial_cols_for_grouping` (original `ItemID_str`)
for grouping and `static_cols_for_model_strict` (including
`ItemID_encoded`) to create the `static_data` array.

.. code-block:: python
   :linenos:

   time_steps_strict = 7
   forecast_horizon_strict = 1 # Single-step point forecast

   static_data_s, dynamic_data_s, future_data_s, target_data_s = \
       reshape_xtft_data(
           df=df_processed_strict, # Contains ItemID_encoded
           dt_col=dt_col_strict,
           target_col=target_col_strict,
           dynamic_cols=dynamic_cols_def_strict,
           static_cols=static_cols_for_model_strict, # Use encoded static
           future_cols=future_cols_def_strict,
           spatial_cols=spatial_cols_for_grouping, # Group by original ItemID_str
           time_steps=time_steps_strict,
           forecast_horizons=forecast_horizon_strict,
           verbose=0
       )
   targets_s = target_data_s.astype(np.float32) # Already (N,H,1)

   print(f"\nReshaped Data Shapes for Stricter TFT:")
   print(f"  Static : {static_data_s.shape}")
   print(f"  Dynamic: {dynamic_data_s.shape}")
   print(f"  Future : {future_data_s.shape}")
   print(f"  Target : {targets_s.shape}")

**Expected Output 4.4:**
   *(Shapes depend on n_items, n_timesteps, time_steps, forecast_horizon)*

.. code-block:: text

   Reshaped Data Shapes for Stricter TFT:
     Static : (84, 2)
     Dynamic: (84, 7, 2)
     Future : (84, 8, 2)
     Target : (84, 1, 1)

Step 5: Train/Validation Split of Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Split the generated sequence arrays. The input for the model will be a
list of three non-None arrays: `[X_static, X_dynamic, X_future]`.

.. code-block:: python
   :linenos:

   val_split_s_frac = 0.2
   n_samples_s_total = static_data_s.shape[0]
   split_idx_s_val = int(n_samples_s_total * (1 - val_split_s_frac))

   X_s_train_s, X_s_val_s = static_data_s[:split_idx_s_val], static_data_s[split_idx_s_val:]
   X_d_train_s, X_d_val_s = dynamic_data_s[:split_idx_s_val], dynamic_data_s[split_idx_s_val:]
   X_f_train_s, X_f_val_s = future_data_s[:split_idx_s_val], future_data_s[split_idx_s_val:]
   y_t_train_s, y_t_val_s = targets_s[:split_idx_s_val], targets_s[split_idx_s_val:]

   # Package inputs as the REQUIRED list [static, dynamic, future]
   train_inputs_strict = [X_s_train_s, X_d_train_s, X_f_train_s]
   val_inputs_strict = [X_s_val_s, X_d_val_s, X_f_val_s]

   print("\nSequence data split for stricter TFT.")
   print(f"  Train samples: {len(y_t_train_s)}")
   print(f"  Validation samples: {len(y_t_val_s)}")

**Expected Output 5.5:**

.. code-block:: text

   Sequence data split for stricter TFT.
     Train samples: 67
     Validation samples: 17

Step 6: Define and Compile Stricter `TFT` Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate the :class:`~fusionlab.nn.transformers.TFT` class. All
three input dimensions (`static_input_dim`, `dynamic_input_dim`,
`future_input_dim`) must be provided and must be > 0.

.. code-block:: python
   :linenos:

   model_strict_ex = TFT( # Using the stricter TFT class
       static_input_dim=static_data_s.shape[-1],
       dynamic_input_dim=dynamic_data_s.shape[-1],
       future_input_dim=future_data_s.shape[-1],
       forecast_horizon=forecast_horizon_strict,
       output_dim=1, # Predicting a single value
       hidden_units=16, num_heads=2,
       num_lstm_layers=1, lstm_units=16,
       quantiles=None # Point forecast
   )
   print("\nStricter TFT model instantiated for point forecast.")

   model_strict_ex.compile(optimizer='adam', loss='mse')
   print("Model compiled successfully.")

**Expected Output 6.6:**

.. code-block:: text

   Stricter TFT model instantiated for point forecast.
   Model compiled successfully.

Step 7: Train the Stricter `TFT` Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   print("\nStarting stricter TFT model training...")
   history_strict_ex = model_strict_ex.fit(
       train_inputs_strict, # Pass the list [static, dynamic, future]
       y_t_train_s,
       validation_data=(val_inputs_strict, y_t_val_s),
       epochs=5, batch_size=16, verbose=1
   )
   print("Training finished.")
   if history_strict_ex and history_strict_ex.history.get('val_loss'):
       val_loss = history_strict_ex.history['val_loss'][-1]
       print(f"Final validation loss: {val_loss:.4f}")

**Expected Output 7.7:**
   *(Output will show Keras training progress)*

.. code-block:: text

   Starting stricter TFT model training...
   Epoch 1/5
   5/5 [==============================] - 13s 511ms/step - loss: 1.5969 - val_loss: 0.8108
   Epoch 2/5
   5/5 [==============================] - 0s 16ms/step - loss: 0.7010 - val_loss: 1.9081
   Epoch 3/5
   5/5 [==============================] - 0s 17ms/step - loss: 0.4777 - val_loss: 1.8109
   Epoch 4/5
   5/5 [==============================] - 0s 16ms/step - loss: 0.4485 - val_loss: 1.0865
   Epoch 5/5
   5/5 [==============================] - 0s 17ms/step - loss: 0.4132 - val_loss: 0.7321
   Training finished.
   Final validation loss: 0.7321

Step 8: Make Predictions and Visualize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use the trained model to predict and then visualize the results after
inverse transforming.

.. code-block:: python
   :linenos:

   print("\nMaking predictions with stricter TFT...")
   val_predictions_scaled_s = model_strict_ex.predict(
       val_inputs_strict, verbose=0
       )

   # Inverse transform predictions and actuals
   target_scaler_s = scalers_ex.get(target_col_strict)
   if target_scaler_s:
       dummy_pred_s = np.zeros((len(val_predictions_scaled_s.flatten()),
                                len(num_cols_to_scale_strict)))
       target_idx_s = num_cols_to_scale_strict.index(target_col_strict)
       dummy_pred_s[:, target_idx_s] = val_predictions_scaled_s.flatten()
       val_pred_inv_s = target_scaler_s.inverse_transform(
           dummy_pred_s)[:, target_idx_s]
       val_pred_final_s = val_pred_inv_s.reshape(val_predictions_scaled_s.shape)

       dummy_actual_s = np.zeros((len(y_t_val_s.flatten()),
                                  len(num_cols_to_scale_strict)))
       dummy_actual_s[:, target_idx_s] = y_t_val_s.flatten()
       val_actual_inv_s = target_scaler_s.inverse_transform(
           dummy_actual_s)[:, target_idx_s]
       val_actual_final_s = val_actual_inv_s.reshape(y_t_val_s.shape)
       print("Predictions and actuals inverse transformed.")
   else:
       print("Warning: Target scaler not found. Plotting scaled values.")
       val_pred_final_s = val_predictions_scaled_s
       val_actual_final_s = y_t_val_s

   # --- Visualization (for the first item in validation set) ---
   first_val_item_id_enc = X_s_val_s[0, static_cols_for_model_strict.index('ItemID_encoded')]
   item_mask_val_s = (X_s_val_s[:, static_cols_for_model_strict.index('ItemID_encoded')] == \
                      first_val_item_id_enc)

   item_preds_s = val_pred_final_s[item_mask_val_s, 0, 0]
   item_actuals_s = val_actual_final_s[item_mask_val_s, 0, 0]

   plt.figure(figsize=(12, 6))
   plt.plot(item_actuals_s,
            label=f'Actual (Item Encoded: {int(first_val_item_id_enc)})',
            marker='o', linestyle='--')
   plt.plot(item_preds_s,
            label=f'Predicted (Item Encoded: {int(first_val_item_id_enc)})',
            marker='x')
   plt.title(f'Stricter TFT Point Forecast (Validation Item - Inverse Transformed)')
   plt.xlabel('Sequence Index in Validation Set for this Item')
   plt.ylabel('Value (Inverse Transformed)')
   plt.legend(); plt.grid(True); plt.tight_layout()
   # fig_path_strict_ex = os.path.join(
   # exercise_output_dir_tft_strict,
   # "exercise_tft_required_inputs.png"
   # )
   # plt.savefig(fig_path_strict_ex)
   # print(f"\nPlot saved to {fig_path_strict_ex}")
   plt.show()
   print("Plot generated for stricter TFT.")

**Expected Plot 8.8:**

.. figure:: ../../images/exercise_tft_required_inputs.png
   :alt: Stricter TFT Point Forecast Exercise Results
   :align: center
   :width: 80%

   Visualization of the point forecast from the stricter `TFT` model
   against actual validation data for a specific item.

Discussion of Exercise
----------------------
In this exercise, you learned how to:
* Prepare a multi-item dataset with distinct static, dynamic, and
  future features.
* Numerically encode categorical static identifiers like `ItemID` using
  `LabelEncoder`.
* Use :func:`~fusionlab.nn.utils.reshape_xtft_data` to generate the
  three required input arrays (`static_data`, `dynamic_data`,
  `future_data`) for the stricter
  :class:`~fusionlab.nn.transformers.TFT` model.
* Instantiate and train the stricter `TFT`, ensuring all three
  `*_input_dim` parameters are provided.
* Correctly structure the input to `fit` and `predict` as a list
  `[static_array, dynamic_array, future_array]`.

This example highlights the data preparation and usage pattern for
the `TFT` model variant that mandates all three types of input features.

