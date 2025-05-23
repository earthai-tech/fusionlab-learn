.. _exercise_experimental_super_tft:

==================================================
Exercise: Forecasting with Experimental SuperXTFT
==================================================

Welcome to this exercise on using the experimental
:class:`~fusionlab.nn.SuperXTFT` model from ``fusionlab-learn``.
`SuperXTFT` builds upon the :class:`~fusionlab.nn.XTFT` architecture
by incorporating additional components like input Variable Selection
Networks (VSNs) and post-processing Gated Residual Networks (GRNs)
for several internal stages.

.. warning::
   ``SuperXTFT`` is currently considered **experimental**. Its API and
   behavior may change in future releases, or it might be merged into
   the main ``XTFT`` or deprecated. It is **not recommended for
   production use** at this time. This exercise is for exploration
   and understanding its structure. For stable deployments, please use
   :class:`~fusionlab.nn.XTFT`.

**Learning Objectives:**

* Understand how to instantiate and use the `SuperXTFT` model.
* Recognize that the data preparation and overall workflow are very
  similar to those for `XTFT`.
* Perform a multi-step quantile forecast using `SuperXTFT`.
* Visualize the probabilistic forecasts.

Let's begin!

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and its common dependencies
installed.

.. code-block:: bash

   pip install fusionlab-learn matplotlib scikit-learn joblib

---

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~
We start by importing necessary libraries and ``fusionlab`` components.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   import os
   import joblib
   import warnings

   # FusionLab imports
   from fusionlab.nn.transformers import SuperXTFT # Import SuperXTFT
   from fusionlab.nn.utils import reshape_xtft_data
   from fusionlab.nn.losses import combined_quantile_loss
   from fusionlab.datasets.make import make_multi_feature_time_series

   # Suppress warnings and TF logs
   warnings.filterwarnings('ignore')
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   tf.get_logger().setLevel('ERROR')
   if hasattr(tf, 'autograph'):
       tf.autograph.set_verbosity(0)

   exercise_output_dir_super_xtft = "./super_xtft_exercise_outputs"
   os.makedirs(exercise_output_dir_super_xtft, exist_ok=True)

   print("Libraries imported for SuperXTFT exercise.")

**Expected Output 1.1:**

.. code-block:: text

   Libraries imported for SuperXTFT exercise.

Step 2: Generate Synthetic Multi-Feature Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We'll use the same data generation setup as the advanced XTFT exercise,
as `SuperXTFT` also expects static, dynamic, and future inputs.

.. code-block:: python
   :linenos:

   n_items_sxtft = 2
   n_timesteps_sxtft = 36 # Shorter for quicker run
   rng_seed_sxtft = 42
   np.random.seed(rng_seed_sxtft)
   tf.random.set_seed(rng_seed_sxtft)

   data_bunch_sxtft = make_multi_feature_time_series(
       n_series=n_items_sxtft, n_timesteps=n_timesteps_sxtft,
       freq='MS', seasonality_period=12,
       seed=rng_seed_sxtft, as_frame=False
   )
   df_raw_sxtft = data_bunch_sxtft.frame.copy()
   print(f"Generated raw data shape for SuperXTFT exercise: {df_raw_sxtft.shape}")
   print(df_raw_sxtft.head(3))

**Expected Output 2.2:**

.. code-block:: text

   Generated raw data shape for SuperXTFT exercise: (72, 9)
           date  series_id  base_level  ...  month  future_event     target
   0 2020-01-01          0   50.049671  ...      1             1  63.055435
   1 2020-02-01          0   50.049671  ...      2             1  68.394497
   2 2020-03-01          0   50.049671  ...      3             1  70.075474

   [3 rows x 9 columns]
   
Step 3: Define Feature Roles and Scale Numerical Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use feature lists from the `data_bunch` and scale numerical features.
`series_id` is numerical and will be used as a static feature.

.. code-block:: python
   :linenos:

   target_col_sxtft = data_bunch_sxtft.target_col
   dt_col_sxtft = data_bunch_sxtft.dt_col
   static_cols_sxtft = data_bunch_sxtft.static_features
   dynamic_cols_sxtft = data_bunch_sxtft.dynamic_features
   future_cols_sxtft = data_bunch_sxtft.future_features
   spatial_cols_sxtft = [data_bunch_sxtft.spatial_id_col]

   scalers_sxtft = {}
   num_cols_to_scale_sxtft = ['base_level', 'dynamic_cov',
                              'target_lag1', target_col_sxtft]
   df_scaled_sxtft = df_raw_sxtft.copy()

   for col in num_cols_to_scale_sxtft:
       if col in df_scaled_sxtft.columns and \
          pd.api.types.is_numeric_dtype(df_scaled_sxtft[col]):
           scaler = StandardScaler()
           df_scaled_sxtft[col] = scaler.fit_transform(df_scaled_sxtft[[col]])
           scalers_sxtft[col] = scaler
   print(f"\nNumerical features scaled: {num_cols_to_scale_sxtft}")

**Expected Output 3.3:**

.. code-block:: text

   Numerical features scaled: ['base_level', 'dynamic_cov', 'target_lag1', 'target']

Step 4: Prepare Sequences using `reshape_xtft_data`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Transform the DataFrame into structured arrays for `SuperXTFT`.

.. code-block:: python
   :linenos:

   time_steps_sxtft = 12
   forecast_horizons_sxtft = 6

   s_data_sxtft, d_data_sxtft, f_data_sxtft, t_data_sxtft = \
       reshape_xtft_data(
           df=df_scaled_sxtft, dt_col=dt_col_sxtft,
           target_col=target_col_sxtft,
           dynamic_cols=dynamic_cols_sxtft,
           static_cols=static_cols_sxtft, # Includes series_id, base_level
           future_cols=future_cols_sxtft,
           spatial_cols=spatial_cols_sxtft,
           time_steps=time_steps_sxtft,
           forecast_horizons=forecast_horizons_sxtft,
           verbose=0 # Suppress reshape logs for brevity
       )
   print(f"\nReshaped Data Shapes for SuperXTFT:")
   print(f"  Static : {s_data_sxtft.shape}")
   print(f"  Dynamic: {d_data_sxtft.shape}")
   print(f"  Future : {f_data_sxtft.shape}")
   print(f"  Target : {t_data_sxtft.shape}")

**Expected Output 4.4:**
   *(For N_series=2, N_timesteps=36, T=12, H=6:
   Seq/series = 36-12-6+1 = 19. Total = 2*19 = 38)*

.. code-block:: text

   Reshaped Data Shapes for SuperXTFT:
     Static : (38, 2)
     Dynamic: (38, 12, 4)
     Future : (38, 18, 3)
     Target : (38, 6, 1)

Step 5: Train/Validation Split of Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Split sequence arrays for training and validation.

.. code-block:: python
   :linenos:

   val_split_sxtft_frac = 0.25 # Using a bit more for validation
   n_samples_sxtft = s_data_sxtft.shape[0]
   split_idx_sxtft = int(n_samples_sxtft * (1 - val_split_sxtft_frac))

   X_s_train_sxtft, X_s_val_sxtft = s_data_sxtft[:split_idx_sxtft], s_data_sxtft[split_idx_sxtft:]
   X_d_train_sxtft, X_d_val_sxtft = d_data_sxtft[:split_idx_sxtft], d_data_sxtft[split_idx_sxtft:]
   X_f_train_sxtft, X_f_val_sxtft = f_data_sxtft[:split_idx_sxtft], f_data_sxtft[split_idx_sxtft:]
   y_t_train_sxtft, y_t_val_sxtft = t_data_sxtft[:split_idx_sxtft], t_data_sxtft[split_idx_sxtft:]

   train_inputs_sxtft = [X_s_train_sxtft, X_d_train_sxtft, X_f_train_sxtft]
   val_inputs_sxtft = [X_s_val_sxtft, X_d_val_sxtft, X_f_val_sxtft]

   print(f"\nData split for SuperXTFT. Train: {len(y_t_train_sxtft)}, "
         f"Val: {len(y_t_val_sxtft)}")

**Expected Output 5.5:**

.. code-block:: text

   Data split for SuperXTFT. Train: 28, Val: 10

Step 6: Define SuperXTFT Model for Quantile Forecast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate the :class:`~fusionlab.nn.SuperXTFT` model. Its parameters
are similar to `XTFT`. We'll explicitly disable anomaly detection for
this exercise.

.. code-block:: python
   :linenos:

   quantiles_sxtft = [0.1, 0.5, 0.9]
   output_dim_sxtft = 1

   s_dim_sxtft = X_s_train_sxtft.shape[-1]
   d_dim_sxtft = X_d_train_sxtft.shape[-1]
   f_dim_sxtft = X_f_train_sxtft.shape[-1]

   super_xtft_model_ex = SuperXTFT(
       static_input_dim=s_dim_sxtft,
       dynamic_input_dim=d_dim_sxtft,
       future_input_dim=f_dim_sxtft,
       forecast_horizon=forecast_horizons_sxtft,
       quantiles=quantiles_sxtft,
       output_dim=output_dim_sxtft,
       # Minimal HPs for faster demo
       embed_dim=8, lstm_units=16, attention_units=8,
       hidden_units=16, num_heads=1, dropout_rate=0.0,
       max_window_size=time_steps_sxtft, memory_size=10,
       scales=None,
       anomaly_detection_strategy=None, # Explicitly disable
       anomaly_loss_weight=0.0
   )
   print("\nSuperXTFT model instantiated (anomaly detection disabled).")

Step 7: Compile and Train the SuperXTFT Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compile with quantile loss and train for a few epochs.

.. code-block:: python
   :linenos:

   loss_fn_sxtft = combined_quantile_loss(quantiles=quantiles_sxtft)
   super_xtft_model_ex.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
       loss=loss_fn_sxtft
       )
   print("SuperXTFT model compiled.")

   # Optional: Build model with dummy inputs to print summary
   # try:
   #     dummy_s = tf.zeros((1, s_dim_sxtft))
   #     dummy_d = tf.zeros((1, time_steps_sxtft, d_dim_sxtft))
   #     dummy_f = tf.zeros((1, time_steps_sxtft + forecast_horizons_sxtft, f_dim_sxtft))
   #     super_xtft_model_ex([dummy_s, dummy_d, dummy_f])
   #     super_xtft_model_ex.summary(line_length=90)
   # except Exception as e:
   #     print(f"Model build/summary error: {e}")

   print("\nStarting SuperXTFT model training...")
   history_sxtft = super_xtft_model_ex.fit(
       train_inputs_sxtft, y_t_train_sxtft,
       validation_data=(val_inputs_sxtft, y_t_val_sxtft),
       epochs=3, batch_size=4, verbose=1 # Short run for demo
   )
   print("SuperXTFT Training finished.")
   if history_sxtft and history_sxtft.history.get('val_loss'):
       val_loss_sxtft = history_sxtft.history['val_loss'][-1]
       print(f"Final validation loss: {val_loss_sxtft:.4f}")

**Expected Output 7.7:**
   *(Keras training logs and final validation loss)*

.. code-block:: text

   SuperXTFT model compiled.

   Starting SuperXTFT model training...
   Epoch 1/3
   7/7 [==============================] - 17s 329ms/step - loss: 0.4341 - val_loss: 0.5377
   Epoch 2/3
   7/7 [==============================] - 0s 12ms/step - loss: 0.4233 - val_loss: 0.5354
   Epoch 3/3
   7/7 [==============================] - 0s 12ms/step - loss: 0.4135 - val_loss: 0.5387
   SuperXTFT Training finished.
   Final validation loss: 0.5387

Step 8: Make Predictions and Visualize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Predict on the validation set and visualize the quantile forecast for
a sample item, similar to the XTFT example.

.. code-block:: python
   :linenos:

   print("\nMaking quantile predictions with SuperXTFT...")
   val_pred_scaled_sxtft = super_xtft_model_ex.predict(
       val_inputs_sxtft, verbose=0
       )
   print(f"Scaled prediction output shape: {val_pred_scaled_sxtft.shape}")

   # Inverse Transform (simplified, assumes target was scaled)
   target_scaler_sxtft = scalers_sxtft.get(target_col_sxtft)
   if target_scaler_sxtft:
       num_val_sxtft = X_s_val_sxtft.shape[0]
       num_q_sxtft = len(quantiles_sxtft)

       pred_flat_sxtft = val_pred_scaled_sxtft.reshape(-1, num_q_sxtft)
       actual_flat_sxtft = y_t_val_sxtft.reshape(-1, 1)

       pred_inv_sxtft = target_scaler_sxtft.inverse_transform(pred_flat_sxtft)
       actual_inv_sxtft = target_scaler_sxtft.inverse_transform(actual_flat_sxtft)

       pred_final_sxtft = pred_inv_sxtft.reshape(val_pred_scaled_sxtft.shape)
       actual_final_sxtft = actual_inv_sxtft.reshape(y_t_val_sxtft.shape)
       print("Predictions and actuals inverse transformed.")
   else:
       print("Warning: Target scaler not found. Plotting scaled values.")
       pred_final_sxtft = val_pred_scaled_sxtft
       actual_final_sxtft = y_t_val_sxtft

   # --- Visualization for one sample item ---
   sample_idx_sxtft = 0 # Plot the first validation sequence
   if len(actual_final_sxtft) > sample_idx_sxtft:
       actual_sxtft_item = actual_final_sxtft[sample_idx_sxtft, :, 0]
       pred_q_sxtft_item = pred_final_sxtft[sample_idx_sxtft, :, :]
       steps_axis_sxtft = np.arange(1, forecast_horizons_sxtft + 1)

       plt.figure(figsize=(12, 6))
       plt.plot(steps_axis_sxtft, actual_sxtft_item,
                label='Actual Sales', marker='o', linestyle='--')
       plt.plot(steps_axis_sxtft, pred_q_sxtft_item[:, 1], # Median
                label='Median Forecast (q=0.5)', marker='x')
       plt.fill_between(
           steps_axis_sxtft, pred_q_sxtft_item[:, 0], pred_q_sxtft_item[:, 2],
           color='lightcoral', alpha=0.4,
           label='Interval (q0.1-q0.9)'
       )
       plt.title(f'SuperXTFT Quantile Forecast (Sample {sample_idx_sxtft})')
       plt.xlabel('Forecast Step'); plt.ylabel(target_col_sxtft)
       plt.legend(); plt.grid(True); plt.tight_layout()
       # fig_path_sxtft = os.path.join(
       # exercise_output_dir_super_xtft,
       # "exercise_super_xtft_forecast.png")
       # plt.savefig(fig_path_sxtft)
       plt.show()
       print("\nSuperXTFT quantile forecast plot generated.")
   else:
       print("\nNot enough validation samples to plot.")


**Expected Plot 8.8:**

.. figure:: ../../images/exercise_super_xtft_forecast.png
   :alt: SuperXTFT Quantile Forecast Exercise
   :align: center
   :width: 80%

   Visualization of the SuperXTFT quantile forecast (median and
   interval) against actual validation data.

Discussion of Exercise
-------------------------
This exercise demonstrated the usage of the experimental
:class:`~fusionlab.nn.SuperXTFT` model. You observed that:
* The data preparation steps (feature definition, scaling, sequence
  generation with `reshape_xtft_data`) are identical to those for
  the standard `XTFT` model, as `SuperXTFT` expects the same
  `[static, dynamic, future]` input structure.
* Instantiation and compilation are also very similar, using the
  same set of core hyperparameters.
* The main differences of `SuperXTFT` (input VSNs, additional GRNs)
  are internal to its architecture. From a user's perspective, the
  interaction pattern for training and prediction is largely the same
  as with `XTFT`.

Remember that `SuperXTFT` is experimental. For production or stable
research, :class:`~fusionlab.nn.XTFT` is the recommended choice. This
exercise serves to illustrate how one might explore such experimental
variants within the ``fusionlab-learn`` framework.

