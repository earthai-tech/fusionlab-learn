.. _exercise_advanced_xtft:

==================================================
Exercise: Advanced Quantile Forecasting with XTFT
==================================================

Welcome to this exercise on advanced time series forecasting using
the :class:`~fusionlab.nn.XTFT` (Extreme Temporal Fusion Transformer)
model from ``fusionlab-learn``. XTFT is designed for complex scenarios,
handling static, dynamic past, and known future features to produce
multi-horizon quantile forecasts.

**Learning Objectives:**

* Understand the data preparation steps for XTFT, including feature
    definition and sequence generation.
* Learn how to instantiate, compile, and train an XTFT model for
    quantile forecasting using all three input types (static, dynamic, future).
* Practice making multi-step predictions and interpreting the
    quantile outputs.
* Visualize probabilistic forecasts to understand prediction
    uncertainty.

Let's begin!

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and its common dependencies
installed. For visualizations, `matplotlib` is also needed.

.. code-block:: bash

   pip install fusionlab-learn matplotlib scikit-learn joblib

---

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, we import all necessary libraries.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   import os
   import joblib
   import warnings

   # FusionLab imports
   from fusionlab.nn.transformers import XTFT
   from fusionlab.nn.utils import reshape_xtft_data
   from fusionlab.nn.losses import combined_quantile_loss
   from fusionlab.datasets.make import make_multi_feature_time_series

   warnings.filterwarnings('ignore')
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   tf.get_logger().setLevel('ERROR')
   if hasattr(tf, 'autograph'):
       tf.autograph.set_verbosity(0)

   exercise_output_dir_xtft = "./xtft_advanced_exercise_outputs"
   os.makedirs(exercise_output_dir_xtft, exist_ok=True)
   print("Libraries imported for XTFT exercise.")

**Expected Output 1.1:**

.. code-block:: text

   Libraries imported for XTFT exercise.

Step 2: Generate Synthetic Time Series Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use :func:`~fusionlab.datasets.make.make_multi_feature_time_series`
to generate data with static, dynamic, and future features.

.. code-block:: python
   :linenos:

   n_items_ex = 3
   n_timesteps_ex = 36
   rng_seed_ex = 42
   np.random.seed(rng_seed_ex)

   # Generate data using the fusionlab utility
   data_bunch_ex = make_multi_feature_time_series(
       n_series=n_items_ex,
       n_timesteps=n_timesteps_ex,
       freq='MS', # Monthly data
       seasonality_period=12, # Yearly seasonality
       seed=rng_seed_ex,
       as_frame=False # Get Bunch object to access feature lists
   )
   df_raw_ex = data_bunch_ex.frame.copy() # Work with a copy

   print(f"Generated raw data shape for exercise: {df_raw_ex.shape}")
   print(f"Columns: {df_raw_ex.columns.tolist()}")
   print("Sample of generated data:")
   print(df_raw_ex.head(3))

**Expected Output 2.2:**
   *(Shape and sample data will be consistent due to random seed.
   Column names will match those from `make_multi_feature_time_series`)*

.. code-block:: text

   Generated raw data shape for exercise: (108, 9)
   Columns: ['date', 'series_id', 'base_level', 'month', 'dayofweek', 'dynamic_cov', 'target_lag1', 'future_event', 'target']
   Sample of generated data:
        date  series_id  base_level  ...  dayofweek  dynamic_cov     target
0 2020-01-01          0   50.049671  ...          2    -0.069132  63.055435
1 2020-02-01          0   50.049671  ...          5     0.841482  68.394497
2 2020-03-01          0   50.049671  ...          6     1.761515  70.075474

[3 rows x 9 columns]

Step 3: Define Feature Roles and Scale Numerical Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use the feature lists provided by `data_bunch_ex`.
Numerical features are scaled. `series_id` is already numerical.

.. code-block:: python
   :linenos:

   target_col_ex = data_bunch_ex.target_col
   dt_col_ex = data_bunch_ex.dt_col
   # Use feature lists from data_bunch
   static_cols_ex = data_bunch_ex.static_features
   dynamic_cols_ex = data_bunch_ex.dynamic_features
   future_cols_ex = data_bunch_ex.future_features
   spatial_cols_ex = [data_bunch_ex.spatial_id_col]

   scalers_ex = {}
   # Define numerical columns to scale (excluding IDs and time components
   # that might be treated as categorical by the model's embeddings)
   num_cols_to_scale_ex = ['base_level', 'dynamic_cov', 'target_lag1', target_col_ex]
   # Ensure 'month' and 'dayofweek' are not scaled if they are to be embedded
   # or treated as categorical by the model.

   df_scaled_ex = df_raw_ex.copy()
   for col in num_cols_to_scale_ex:
       if col in df_scaled_ex.columns:
           scaler = StandardScaler()
           df_scaled_ex[col] = scaler.fit_transform(df_scaled_ex[[col]])
           scalers_ex[col] = scaler
           print(f"Scaled column: {col}")
       else:
           print(f"Warning: Column '{col}' for scaling not found in DataFrame.")

   scalers_path_ex = os.path.join(
       exercise_output_dir_xtft, "xtft_exercise_scalers.joblib"
       )
   joblib.dump(scalers_ex, scalers_path_ex)
   print(f"\nScalers saved to {scalers_path_ex}")

**Expected Output 3.3:**

.. code-block:: text

   Scaled column: base_level
   Scaled column: dynamic_cov
   Scaled column: target_lag1
   Scaled column: target

   Scalers saved to ./xtft_advanced_exercise_outputs/xtft_exercise_scalers.joblib

Step 4: Prepare Sequences using `reshape_xtft_data`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, we use the `static_cols_ex` (which includes `series_id` and
`base_level`) when calling `reshape_xtft_data`. This will ensure
`static_data_ex` has features.

.. code-block:: python
   :linenos:

   time_steps_ex = 12
   forecast_horizons_ex = 6

   # `static_cols_ex` from data_bunch is ['series_id', 'base_level']
   # Both are numerical and can be used as static features.
   static_data_ex, dynamic_data_ex, future_data_ex, target_data_ex = \
       reshape_xtft_data(
           df=df_scaled_ex,
           dt_col=dt_col_ex,
           target_col=target_col_ex,
           dynamic_cols=dynamic_cols_ex,
           static_cols=static_cols_ex, # Use actual static features
           future_cols=future_cols_ex,
           spatial_cols=spatial_cols_ex, # Group by 'series_id'
           time_steps=time_steps_ex,
           forecast_horizons=forecast_horizons_ex,
           verbose=1
       )

**Expected Output 4.4:**
   *(Shapes will reflect actual static features being used)*

.. code-block:: text

   [INFO] Reshaping time‑series data into rolling sequences...

   [INFO] Data grouped by ['series_id'] into 3 groups.

   [INFO] Total valid sequences to be generated: 57

   [INFO] Final data shapes after reshaping:
     [DEBUG] Static Data : (57, 2)
     [DEBUG] Dynamic Data: (57, 12, 4)
     [DEBUG] Future Data : (57, 18, 3)
     [DEBUG] Target Data : (57, 6, 1)

   [INFO] Time‑series data successfully reshaped into rolling sequences.

Step 5: Train/Validation Split of Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Split the generated sequence arrays.

.. code-block:: python
   :linenos:

   val_split_fraction_ex = 0.2
   if target_data_ex is None or target_data_ex.shape[0] == 0:
       raise ValueError("No sequences generated.")
   
   n_samples_ex = target_data_ex.shape[0]
   split_idx_ex = int(n_samples_ex * (1 - val_split_fraction_ex))

   X_s_train, X_s_val = static_data_ex[:split_idx_ex], static_data_ex[split_idx_ex:]
   X_d_train, X_d_val = dynamic_data_ex[:split_idx_ex], dynamic_data_ex[split_idx_ex:]
   X_f_train, X_f_val = future_data_ex[:split_idx_ex], future_data_ex[split_idx_ex:]
   y_t_train, y_t_val = target_data_ex[:split_idx_ex], target_data_ex[split_idx_ex:]

   train_inputs_ex = [X_s_train, X_d_train, X_f_train]
   val_inputs_ex = [X_s_val, X_d_val, X_f_val]

   print(f"\nData split into Train/Validation sequences:")
   print(f"  Train samples: {X_d_train.shape[0]}")
   print(f"  Validation samples: {X_d_val.shape[0]}")
   print(f"  Train Static Shape : {X_s_train.shape}")
   print(f"  Train Dynamic Shape: {X_d_train.shape}")
   print(f"  Train Future Shape : {X_f_train.shape}")
   print(f"  Train Target Shape : {y_t_train.shape}")

**Expected Output 5.5:**

.. code-block:: text

   Data split into Train/Validation sequences:
     Train samples: 45
     Validation samples: 12
     Train Static Shape : (45, 2)
     Train Dynamic Shape: (45, 12, 4)
     Train Future Shape : (45, 18, 3)
     Train Target Shape : (45, 6, 1)

Step 6: Define XTFT Model for Quantile Forecast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate :class:`~fusionlab.nn.XTFT`. `static_input_dim` will now
be greater than 0. Explicitly set `anomaly_detection_strategy=None`.

.. code-block:: python
   :linenos:

   quantiles_ex = [0.1, 0.5, 0.9]
   output_dim_ex = 1

   s_dim_ex = X_s_train.shape[-1] # Will be > 0 now
   d_dim_ex = X_d_train.shape[-1]
   f_dim_ex = X_f_train.shape[-1]

   model_ex = XTFT(
       static_input_dim=s_dim_ex,
       dynamic_input_dim=d_dim_ex,
       future_input_dim=f_dim_ex,
       forecast_horizon=forecast_horizons_ex,
       quantiles=quantiles_ex,
       output_dim=output_dim_ex,
       embed_dim=16, lstm_units=32, attention_units=16,
       hidden_units=32, num_heads=2, dropout_rate=0.1,
       max_window_size=time_steps_ex, memory_size=20,
       scales=None,
       anomaly_detection_strategy=None, # Explicitly disable
       anomaly_loss_weight=0.0
   )
   print("\nXTFT model instantiated (anomaly detection disabled).")

Step 7: Compile and Train the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*(This step remains the same as in the previous version of the artifact)*

.. code-block:: python
   :linenos:

   loss_fn_ex = combined_quantile_loss(quantiles=quantiles_ex)
   model_ex.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
       loss=loss_fn_ex
       )
   print("XTFT model compiled with combined quantile loss.")

   # Dummy call to build model (optional)
   try:
       dummy_s_ex = tf.zeros((1, s_dim_ex)) # s_dim_ex > 0
       dummy_d_ex = tf.zeros((1, time_steps_ex, d_dim_ex))
       dummy_f_ex = tf.zeros((1, time_steps_ex + forecast_horizons_ex, f_dim_ex))
       # model_ex([dummy_s_ex, dummy_d_ex, dummy_f_ex]) # Build
       # model_ex.summary(line_length=90)
   except Exception as e:
       print(f"Model build/summary failed: {e}")

   print("\nStarting XTFT model training (few epochs for demo)...")
   history_ex = model_ex.fit(
       train_inputs_ex, y_t_train,
       validation_data=(val_inputs_ex, y_t_val),
       epochs=3, batch_size=4, verbose=1 # Reduced for gallery speed
   )
   print("Training finished.")
   if history_ex and history_ex.history.get('val_loss'):
       val_loss = history_ex.history['val_loss'][-1]
       print(f"Final validation loss (quantile): {val_loss:.4f}")
       
       
**Expected Output 7:**

.. code-block:: text

   XTFT model compiled with combined quantile loss.

   Starting XTFT model training (few epochs for demo)...
   Epoch 1/3
   12/12 [==============================] - 8s 86ms/step - loss: 0.3010 - val_loss: 0.4640
   Epoch 2/3
   12/12 [==============================] - 0s 8ms/step - loss: 0.1919 - val_loss: 0.5092
   Epoch 3/3
   12/12 [==============================] - 0s 9ms/step - loss: 0.1450 - val_loss: 0.4088
   Training finished.
   Final validation loss (quantile): 0.4088
     

Step 8: Make Predictions and Inverse Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*(This step remains the same as in the previous version of the artifact)*

.. code-block:: python
   :linenos:

   print("\nMaking quantile predictions on validation set...")
   predictions_scaled_ex = model_ex.predict(val_inputs_ex, verbose=0)
   print(f"Scaled prediction output shape: {predictions_scaled_ex.shape}")

   target_scaler_ex = scalers_ex.get(target_col_ex)
   if target_scaler_ex is None:
       print("Warning: Target scaler not found. Plotting scaled values.")
       predictions_final_ex = predictions_scaled_ex
       y_val_final_ex = y_t_val
   else:
       num_val_samples_ex = X_s_val.shape[0]
       num_quantiles_ex = len(quantiles_ex)
       if output_dim_ex == 1:
           pred_reshaped_ex = predictions_scaled_ex.reshape(-1, num_quantiles_ex)
           predictions_inv_ex = target_scaler_ex.inverse_transform(pred_reshaped_ex)
           predictions_final_ex = predictions_inv_ex.reshape(
               num_val_samples_ex, forecast_horizons_ex, num_quantiles_ex
           )
           y_val_reshaped_ex = y_t_val.reshape(-1, output_dim_ex)
           y_val_inv_ex = target_scaler_ex.inverse_transform(y_val_reshaped_ex)
           y_val_final_ex = y_val_inv_ex.reshape(
               num_val_samples_ex, forecast_horizons_ex, output_dim_ex
           )
           print("Predictions and actuals inverse transformed.")
       else:
           print("Multi-output inverse transform not shown, plotting scaled.")
           predictions_final_ex = predictions_scaled_ex
           y_val_final_ex = y_t_val

**Expected Output 8:**

.. code-block:: text

   Making quantile predictions on validation set...
   Scaled prediction output shape: (12, 6, 3)
   Predictions and actuals inverse transformed.
   
Step 9: Visualize Forecast for One Item
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*(This step remains the same. The visualization will now use the actual
`X_val_static` to identify the item, as it contains features.)*

.. code-block:: python
   :linenos:

   sample_to_plot_idx_ex = 0 # Plot the first validation sequence's forecast

   if y_val_final_ex is not None and predictions_final_ex is not None and \
      len(y_val_final_ex) > sample_to_plot_idx_ex:
       actual_vals_item_ex = y_val_final_ex[sample_to_plot_idx_ex, :, 0]
       pred_quantiles_item_ex = predictions_final_ex[sample_to_plot_idx_ex, :, :]
       forecast_steps_axis_ex = np.arange(1, forecast_horizons_ex + 1)

       # Get the ItemID for the plotted sample from X_val_static
       # Assuming 'series_id' is the first column in static_cols_ex
       item_id_plotted = X_s_val[sample_to_plot_idx_ex, 0]
       # If 'series_id' was label encoded, you might want to inverse_transform it here
       # For this example, make_multi_feature_time_series provides integer series_id

       plt.figure(figsize=(12, 6))
       plt.plot(forecast_steps_axis_ex, actual_vals_item_ex,
                label='Actual Sales', marker='o', linestyle='--')
       plt.plot(forecast_steps_axis_ex, pred_quantiles_item_ex[:, 1],
                label='Median Forecast (q=0.5)', marker='x')
       plt.fill_between(
           forecast_steps_axis_ex,
           pred_quantiles_item_ex[:, 0], pred_quantiles_item_ex[:, 2],
           color='gray', alpha=0.3,
           label='Prediction Interval (q=0.1 to q=0.9)'
       )
       plt.title(f'XTFT Quantile Forecast (Item ID from Static: {item_id_plotted:.0f}, Sample {sample_to_plot_idx_ex})')
       plt.xlabel('Forecast Step into Horizon')
       plt.ylabel(f'{target_col_ex} (Units after Inverse Transform if applied)')
       plt.legend(); plt.grid(True); plt.tight_layout()
       fig_path_ex = os.path.join(
           exercise_output_dir_xtft,
           "exercise_advanced_xtft_quantile_forecast.png"
           )
       # plt.savefig(fig_path_ex) # Uncomment to save
       # print(f"\nPlot saved to {fig_path_ex}")
       plt.show()
   else:
       print("\nSkipping plot: Not enough data or predictions missing.")


**Example Output Plot:**

.. figure:: ../../images/exercise_advanced_xtft_quantile_forecast.png
   :alt: Advanced XTFT Quantile Forecast Example
   :align: center
   :width: 80%

   Visualization of the XTFT quantile forecast (median and interval)
   against actual validation data for a sample item.

**Discussion of Exercise:**

   This exercise walked through a complete workflow for using the
   :class:`~fusionlab.nn.XTFT` model for multi-step quantile
   forecasting using all three input types: static, dynamic, and future
   features. Key takeaways include:
   * The use of :func:`~fusionlab.datasets.make.make_multi_feature_time_series`
     to generate rich synthetic data.
   * The importance of defining feature roles and appropriately scaling
     numerical inputs.
   * Ensuring that static features (like `series_id` and `base_level`
     from `make_multi_feature_time_series`) are included when calling
     :func:`~fusionlab.nn.utils.reshape_xtft_data` if they are to be
     used by the model. This results in `static_input_dim > 0`.
   * Configuring XTFT for quantile output and using
     :func:`~fusionlab.nn.losses.combined_quantile_loss`.
   * The ability to inverse-transform predictions for interpretation.
   * Visualizing quantile forecasts to assess prediction uncertainty.

   For real-world applications, extensive hyperparameter tuning (see
   :doc:`../hyperparameter_tuning/index`) and more sophisticated
   validation strategies would be necessary.

