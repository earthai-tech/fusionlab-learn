. _example_hyperparameter_tuning:

======================================================
Hyperparameter Tuning with Keras Tuner
======================================================

Finding optimal hyperparameters is crucial for maximizing the
performance of complex models like
:class:`~fusionlab.nn.XTFT`,
:class:`~fusionlab.nn.transformers.TFT` (stricter version), and
the flexible :class:`~fusionlab.nn.TemporalFusionTransformer`.
This example demonstrates how to use the
:func:`~fusionlab.nn.forecast_tuner.xtft_tuner` and
:func:`~fusionlab.nn.forecast_tuner.tft_tuner` utilities
from ``fusionlab-learn`` to automate this search process.

We will cover:

1.  Tuning :class:`~fusionlab.nn.XTFT` for quantile forecasting.
2.  Tuning the stricter :class:`~fusionlab.nn.transformers.TFT`
    (all inputs required) for point forecasting.
3.  Tuning the flexible
    :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
    (using `model_name="tft_flex"`) for point forecasting,
    demonstrating its ability to handle optional inputs (e.g., only
    dynamic features).

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and `keras-tuner` installed:

.. code-block:: bash

   pip install fusionlab-learn keras-tuner matplotlib

Common Setup for All Examples
-----------------------------
The following imports and directory setup are common.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import os
   import shutil # For cleaning up tuner directories
   import warnings

   # FusionLab imports
   from fusionlab.datasets.make import make_multi_feature_time_series
   from fusionlab.nn.forecast_tuner import xtft_tuner, tft_tuner
   from fusionlab.nn.transformers import (
       XTFT,
       TFT as TFTStricter, # Alias for the stricter TFT
       TemporalFusionTransformer as TFTFlexible # Alias for flexible TFT
   )
   from fusionlab.nn.losses import combined_quantile_loss
   from fusionlab.nn.utils import reshape_xtft_data
   import keras_tuner as kt

   # Suppress warnings and TF logs for cleaner output
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   if hasattr(tf, 'autograph'):
       tf.autograph.set_verbosity(0)

   # Configuration for outputs
   base_output_dir_tuning = "./gallery_tuning_runs"
   if not os.path.exists(base_output_dir_tuning):
       os.makedirs(base_output_dir_tuning, exist_ok=True)

   print("Libraries imported and base setup complete for tuning examples.")


Example 1: Tuning XTFT for Quantile Forecasting
-----------------------------------------------
This section demonstrates tuning :class:`~fusionlab.nn.XTFT`.

Step 1.1: Generate Synthetic Data for XTFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use :func:`~fusionlab.datasets.make.make_multi_feature_time_series`
to create a dataset with static, dynamic, and future features.

.. code-block:: python
   :linenos:

   # Data generation parameters for XTFT
   N_SERIES_XTFT = 2
   N_TIMESTEPS_XTFT = 60
   FREQ_XTFT = 'MS'
   SEED_XTFT = 42

   data_bunch_xtft = make_multi_feature_time_series(
       n_series=N_SERIES_XTFT, n_timesteps=N_TIMESTEPS_XTFT,
       freq=FREQ_XTFT, seasonality_period=12,
       seed=SEED_XTFT, as_frame=False
   )
   df_for_xtft_tuning = data_bunch_xtft.frame
   print(f"Generated data for XTFT tuning. Shape: {df_for_xtft_tuning.shape}")

   # Prepare data for reshape_xtft_data (assuming numerical readiness)
   dt_col_xtft = data_bunch_xtft.dt_col
   target_col_xtft = data_bunch_xtft.target_col
   static_cols_xtft = data_bunch_xtft.static_features
   dynamic_cols_xtft = data_bunch_xtft.dynamic_features
   future_cols_xtft = data_bunch_xtft.future_features
   spatial_cols_xtft = [data_bunch_xtft.spatial_id_col]

   time_steps_xtft = 12
   forecast_horizon_xtft = 6

   s_data_xtft, d_data_xtft, f_data_xtft, t_data_xtft = reshape_xtft_data(
       df=df_for_xtft_tuning, dt_col=dt_col_xtft,
       target_col=target_col_xtft,
       dynamic_cols=dynamic_cols_xtft, static_cols=static_cols_xtft,
       future_cols=future_cols_xtft, spatial_cols=spatial_cols_xtft,
       time_steps=time_steps_xtft,
       forecast_horizons=forecast_horizon_xtft,
       verbose=0
   )
   train_inputs_xtft = [
       tf.constant(s_data_xtft, dtype=tf.float32),
       tf.constant(d_data_xtft, dtype=tf.float32),
       tf.constant(f_data_xtft, dtype=tf.float32)
   ]
   y_train_xtft = tf.constant(t_data_xtft, dtype=tf.float32)
   print(f"XTFT Reshaped: S={s_data_xtft.shape}, D={d_data_xtft.shape}, "
         f"F={f_data_xtft.shape}, T={t_data_xtft.shape}")

Step 1.2: Define XTFT Search Space and Case Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Define quantiles, a custom search space, and fixed `case_info`.

.. code-block:: python
   :linenos:

   quantiles_xtft = [0.1, 0.5, 0.9]
   custom_param_space_xtft = {
       'hidden_units': [16, 32], 'num_heads': [1, 2],
       'lstm_units': [16], 'dropout_rate': [0.05, 0.1],
       'learning_rate': [5e-4, 1e-3]
   }
   case_info_xtft = {
       'quantiles': quantiles_xtft,
       'forecast_horizon': forecast_horizon_xtft,
       'output_dim': y_train_xtft.shape[-1],
       'static_input_dim': train_inputs_xtft[0].shape[-1],
       'dynamic_input_dim': train_inputs_xtft[1].shape[-1],
       'future_input_dim': train_inputs_xtft[2].shape[-1],
       'embed_dim': 16, 'max_window_size': time_steps_xtft,
       'memory_size': 20, 'attention_units': 16,
       'recurrent_dropout_rate': 0.0,
       'use_residuals_choices': [True], 'final_agg': 'last',
       'multi_scale_agg': 'last', 'scales_options': ['no_scales'],
       'use_batch_norm_choices': [False], 'verbose_build': 0
   }

Step 1.3: Run the XTFT Tuner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   output_dir_xtft = os.path.join(base_output_dir_tuning, "xtft_run")
   project_name_xtft = "XTFT_Gallery_Quantile_Tuning"
   if os.path.exists(os.path.join(output_dir_xtft, project_name_xtft)):
       shutil.rmtree(os.path.join(output_dir_xtft, project_name_xtft))

   print("\nStarting XTFT hyperparameter tuning...")
   best_hps_xtft, best_model_xtft, tuner_xtft = xtft_tuner(
       inputs=train_inputs_xtft, y=y_train_xtft,
       param_space=custom_param_space_xtft,
       forecast_horizon=forecast_horizon_xtft,
       quantiles=quantiles_xtft,
       case_info=case_info_xtft,
       max_trials=1, epochs=1, batch_sizes=[4], # Minimal for demo
       validation_split=0.5,
       tuner_dir=output_dir_xtft, project_name=project_name_xtft,
       tuner_type='random', model_name="xtft", verbose=0
   )
   print("\nXTFT Tuning complete.")
   if best_hps_xtft:
       print("--- Best Hyperparameters (XTFT) ---")
       print(best_hps_xtft)
       # if best_model_xtft: best_model_xtft.summary()
   else:
       print("XTFT Tuning did not yield best HPs.")


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Tuning Standard TFT Variants
------------------------------

This section covers tuning the stricter :class:`~fusionlab.nn.transformers.TFT`
and the flexible :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
(referred to as `tft_flex`). We use the
:func:`~fusionlab.nn.forecast_tuner.tft_tuner` function, which is a
wrapper around :func:`~fusionlab.nn.forecast_tuner.xtft_tuner`,
setting the `model_name` appropriately.

.. _example_tuning_tft_stricter:

Tuning Stricter TFT (All Inputs Required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The stricter :class:`~fusionlab.nn.transformers.TFT` requires static,
dynamic, and future inputs to be non-None.

Step 2.1: Prepare Data for Stricter TFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use the same data generation as for XTFT, as it includes all three
input types.

.. code-block:: python
   :linenos:

   # Re-use data from XTFT example (s_data_xtft, d_data_xtft, etc.)
   # Or generate new if needed, ensuring all D_s, D_d, D_f are > 0
   train_inputs_strict_tft = [
       tf.constant(s_data_xtft, dtype=tf.float32),
       tf.constant(d_data_xtft, dtype=tf.float32),
       tf.constant(f_data_xtft, dtype=tf.float32)
   ]
   y_train_strict_tft = tf.constant(t_data_xtft, dtype=tf.float32)
   print("\nData prepared for Stricter TFT tuning.")

Step 2.2: Define Stricter TFT Search Space and Case Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The search space will focus on parameters relevant to the standard TFT.

.. code-block:: python
   :linenos:

   # Point forecast for this example
   param_space_strict_tft = {
       'hidden_units': [16, 32],
       'num_heads': [1, 2],
       'num_lstm_layers': [1], # Tune number of LSTM layers
       'lstm_units': [16, 32],   # Tune LSTM units
       'dropout_rate': [0.0, 0.1],
       'recurrent_dropout_rate': [0.0], # Often fixed or small
       'learning_rate': [1e-3]
   }
   case_info_strict_tft = {
       'quantiles': None, # Point forecast
       'forecast_horizon': forecast_horizon_xtft, # Use same as XTFT example
       'output_dim': y_train_strict_tft.shape[-1],
       'static_input_dim': train_inputs_strict_tft[0].shape[-1],
       'dynamic_input_dim': train_inputs_strict_tft[1].shape[-1],
       'future_input_dim': train_inputs_strict_tft[2].shape[-1],
       'activation': 'relu', # Fixed activation
       'use_batch_norm_choices': [False], # Fixed
       'verbose_build': 0
   }

Step 2.3: Run the Tuner for Stricter TFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   output_dir_strict_tft = os.path.join(base_output_dir_tuning, "tft_strict_run")
   project_name_strict_tft = "TFT_Strict_Gallery_Point_Tuning"
   if os.path.exists(os.path.join(output_dir_strict_tft, project_name_strict_tft)):
       shutil.rmtree(os.path.join(output_dir_strict_tft, project_name_strict_tft))

   print("\nStarting Stricter TFT hyperparameter tuning...")
   best_hps_tft_s, _, _ = tft_tuner( # Use tft_tuner
       inputs=train_inputs_strict_tft,
       y=y_train_strict_tft,
       param_space=param_space_strict_tft,
       forecast_horizon=forecast_horizon_xtft,
       quantiles=None,
       case_info=case_info_strict_tft,
       max_trials=1, epochs=1, batch_sizes=[4],
       validation_split=0.5,
       tuner_dir=output_dir_strict_tft,
       project_name=project_name_strict_tft,
       model_name="tft", # Key: specifies the stricter TFT
       verbose=0
   )
   print("\nStricter TFT Tuning complete.")
   if best_hps_tft_s:
       print("--- Best Hyperparameters (Stricter TFT) ---")
       print(best_hps_tft_s)


.. raw:: html

   <hr>

.. _example_tuning_tft_flexible:

Tuning Flexible TemporalFusionTransformer (`tft_flex`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This demonstrates tuning the flexible
:class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
using only dynamic inputs.

Step 3.1: Prepare Data for Flexible TFT (Dynamic Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We'll use only the dynamic part of the previously generated data.

.. code-block:: python
   :linenos:

   # Use d_data_xtft and t_data_xtft from the XTFT data prep
   # Inputs for flexible TFT: [Static, Dynamic, Future]
   # Here, Static and Future will be None.
   # rather to pass this: 
   train_inputs_flex_tft = [
       None, # No static input
       tf.constant(d_data_xtft, dtype=tf.float32), # Only dynamic
       None  # No future input
   ]
   # pass only the dynamic , and TemporalFusionTransformer will 
   # handle it 
   train_inputs_flex_tft = [
       tf.constant(d_data_xtft, dtype=tf.float32), # Only dynamic
   ]
   y_train_flex_tft = tf.constant(t_data_xtft, dtype=tf.float32)
   print("\nData prepared for Flexible TFT (Dynamic Only) tuning.")
   print(f"  Dynamic Input Shape: {train_inputs_flex_tft[0].shape}")

Step 3.2: Define Flexible TFT Search Space and Case Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `case_info` will reflect that static and future dimensions are `None`.

.. code-block:: python
   :linenos:

   # Point forecast for this example
   param_space_flex_tft = {
       'hidden_units': [8, 16], # Smaller search space
       'num_heads': [1],
       'num_lstm_layers': [1],
       'lstm_units': [16],
       'dropout_rate': [0.0],
       'learning_rate': [1e-3]
   }
   case_info_flex_tft = {
       'quantiles': None, # Point forecast
       'forecast_horizon': forecast_horizon_xtft,
       'output_dim': y_train_flex_tft.shape[-1],
       'static_input_dim': None, # Explicitly None
       'dynamic_input_dim': train_inputs_flex_tft[0].shape[-1],
       'future_input_dim': None, # Explicitly None
       'activation': 'relu',
       'use_batch_norm_choices': [False],
       'verbose_build': 0
   }

Step 3.3: Run the Tuner for Flexible TFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   output_dir_flex_tft = os.path.join(base_output_dir_tuning, "tft_flex_run")
   project_name_flex_tft = "TFT_Flexible_Gallery_Point_Tuning"
   if os.path.exists(os.path.join(output_dir_flex_tft, project_name_flex_tft)):
       shutil.rmtree(os.path.join(output_dir_flex_tft, project_name_flex_tft))

   print("\nStarting Flexible TFT (tft_flex) hyperparameter tuning...")
   best_hps_tft_f, _, _ = tft_tuner( # Use tft_tuner
       inputs=train_inputs_flex_tft, # [None, Dynamic, None]
       y=y_train_flex_tft,
       param_space=param_space_flex_tft,
       forecast_horizon=forecast_horizon_xtft,
       quantiles=None,
       case_info=case_info_flex_tft,
       max_trials=1, epochs=1, batch_sizes=[4],
       validation_split=0.5,
       tuner_dir=output_dir_flex_tft,
       project_name=project_name_flex_tft,
       model_name="tft_flex", # Key: specifies flexible TemporalFusionTransformer
       verbose=0
   )
   print("\nFlexible TFT (tft_flex) Tuning complete.")
   if best_hps_tft_f:
       print("--- Best Hyperparameters (Flexible TFT) ---")
       print(best_hps_tft_f)
