.. _example_xtft_hyperparameter_tuning:

==============
XTFT Tuning
==============

Finding optimal hyperparameters is crucial for maximizing the
performance of complex models like
:class:`~fusionlab.nn.XTFT`. This example demonstrates how to
use the :func:`~fusionlab.nn.forecast_tuner.xtft_tuner` utility
from ``fusionlab-learn`` to automate this search process for an
XTFT model configured for quantile forecasting.

The workflow includes:

1.  Generating synthetic multi-feature time series data suitable
    for XTFT.
2.  Defining a hyperparameter search space.
3.  Configuring and running the `xtft_tuner`.
4.  Retrieving and inspecting the best hyperparameters and model.


Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and `keras-tuner` installed:

.. code-block:: bash

   pip install fusionlab-learn keras-tuner matplotlib

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Import necessary libraries, including ``fusionlab-learn`` components
for data generation, model tuning, and the XTFT model itself.

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
   from fusionlab.nn.forecast_tuner import xtft_tuner
   from fusionlab.nn.transformers import XTFT # For type context
   from fusionlab.nn.losses import combined_quantile_loss # For loss definition
   from fusionlab.nn.utils import reshape_xtft_data
   import keras_tuner as kt # Keras Tuner

   # Suppress warnings and TF logs for cleaner output
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   if hasattr(tf, 'autograph'):
       tf.autograph.set_verbosity(0)

   # Configuration for outputs
   output_dir_tuning = "./gallery_tuning_output"
   # Clean up previous run if it exists, for a fresh start
   if os.path.exists(output_dir_tuning):
       shutil.rmtree(output_dir_tuning)
   os.makedirs(output_dir_tuning, exist_ok=True)

   print("Libraries imported and setup complete for tuning example.")

Step 2: Generate Synthetic Data for XTFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use :func:`~fusionlab.datasets.make.make_multi_feature_time_series`
to create a dataset with static, dynamic, and future features. This
data will be used for training and validation during the tuning process.

.. code-block:: python
   :linenos:

   # Data generation parameters
   N_SERIES_TUNE = 2
   N_TIMESTEPS_TUNE = 60 # Approx 5 years of monthly data
   FREQ_TUNE = 'MS'
   SEED_TUNE = 42

   # Generate data as a Bunch object
   data_bunch = make_multi_feature_time_series(
       n_series=N_SERIES_TUNE,
       n_timesteps=N_TIMESTEPS_TUNE,
       freq=FREQ_TUNE,
       seasonality_period=12, # Yearly seasonality for monthly data
       seed=SEED_TUNE,
       as_frame=False # Get Bunch to easily access column names
   )
   df_for_tuning = data_bunch.frame
   print(f"Generated data for tuning. Shape: {df_for_tuning.shape}")

   # --- Prepare data for reshape_xtft_data ---
   # This step would normally involve scaling, encoding etc.
   # For this example, we assume data is numerically ready.
   # In a real workflow, use load_processed_subsidence_data or similar.

   dt_col_tune = data_bunch.dt_col
   target_col_tune = data_bunch.target_col
   static_cols_tune = data_bunch.static_features
   dynamic_cols_tune = data_bunch.dynamic_features
   future_cols_tune = data_bunch.future_features
   spatial_cols_tune = [data_bunch.spatial_id_col]

   # Reshape data into sequences

  
   time_steps_tune = 12 # 1 year lookback
   forecast_horizon_tune = 6 # Predict 6 months ahead

   s_data, d_data, f_data, t_data = reshape_xtft_data(
       df=df_for_tuning, dt_col=dt_col_tune, target_col=target_col_tune,
       dynamic_cols=dynamic_cols_tune, static_cols=static_cols_tune,
       future_cols=future_cols_tune, spatial_cols=spatial_cols_tune,
       time_steps=time_steps_tune, forecast_horizons=forecast_horizon_tune,
       verbose=0
   )
   print(f"\nReshaped data for tuning:")
   print(f"  Static : {s_data.shape}, Dynamic: {d_data.shape}")
   print(f"  Future : {f_data.shape}, Target : {t_data.shape}")

   # For tuner, inputs are [Static, Dynamic, Future]
   # All inputs are required by XTFT
   if s_data is None or d_data is None or f_data is None:
       raise ValueError("XTFT requires static, dynamic, and future inputs.")

   train_inputs_tune = [
       tf.constant(s_data, dtype=tf.float32),
       tf.constant(d_data, dtype=tf.float32),
       tf.constant(f_data, dtype=tf.float32)
   ]
   y_train_tune = tf.constant(t_data, dtype=tf.float32)


Step 3: Define Hyperparameter Search Space and Case Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We define a `custom_param_space` to explore a few hyperparameters.
`case_info` provides fixed parameters required by the model builder.

.. code-block:: python
   :linenos:

   # Define quantiles for probabilistic forecast
   quantiles_tune = [0.1, 0.5, 0.9]

   # Custom search space (subset of DEFAULT_PS in forecast_tuner)
   custom_param_space_tune = {
       'hidden_units': [16, 32],       # Try these hidden unit sizes
       'num_heads': [1, 2],            # Try 1 or 2 attention heads
       'lstm_units': [16],             # Fix LSTM units for this demo
       'dropout_rate': [0.05, 0.1],
       'learning_rate': [5e-4, 1e-3] # Try two learning rates
   }

   # Case info provides fixed parameters for the model builder
   # It must include all required dimensions for the model
   case_info_tune = {
       'quantiles': quantiles_tune,
       'forecast_horizon': forecast_horizon_tune,
       'output_dim': y_train_tune.shape[-1], # Should be 1 for this example
       'static_input_dim': train_inputs_tune[0].shape[-1],
       'dynamic_input_dim': train_inputs_tune[1].shape[-1],
       'future_input_dim': train_inputs_tune[2].shape[-1],
       # Pass other fixed XTFT params if not tuning them:
       'embed_dim': 16, # Example fixed value
       'max_window_size': time_steps_tune,
       'memory_size': 20,
       'attention_units': 16,
       'recurrent_dropout_rate': 0.0,
       'use_residuals_choices': [True], # Fix use_residuals to True
       'final_agg': 'last',
       'multi_scale_agg': 'last',
       'scales_options': ['no_scales'], # Fix scales to None
       'use_batch_norm_choices': [False], # Fix use_batch_norm
       'verbose_build': 0 # Suppress model builder logs
   }
   print("\nHyperparameter search space and case info defined.")

Step 4: Run the XTFT Tuner
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Call :func:`~fusionlab.nn.forecast_tuner.xtft_tuner` with the prepared
data, search space, and tuning configurations. We use a small number
of `max_trials` and `epochs` for a quick demonstration.

.. code-block:: python
   :linenos:

   project_name_tune = "XTFT_Gallery_Quantile_Tuning"
   # Clean up previous project directory if it exists
   project_path = os.path.join(output_dir_tuning, project_name_tune)
   if os.path.exists(project_path):
       shutil.rmtree(project_path)

   print("\nStarting XTFT hyperparameter tuning...")
   best_hps, best_model, tuner = xtft_tuner(
       inputs=train_inputs_tune,
       y=y_train_tune,
       param_space=custom_param_space_tune,
       # forecast_horizon and quantiles are now primarily passed via case_info
       # for the model builder, but also needed by tuner func for defaults
       forecast_horizon=forecast_horizon_tune,
       quantiles=quantiles_tune,
       case_info=case_info_tune, # Crucial for model instantiation
       max_trials=2,        # Number of HP combinations to try per batch size
       objective='val_loss',
       epochs=3,            # Epochs for FULL training of best HP per batch
       batch_sizes=[8],     # Test with a single small batch size for demo
       validation_split=0.3, # Use 30% of data for validation during search
       tuner_dir=output_dir_tuning,
       project_name=project_name_tune,
       tuner_type='random', # 'random' or 'bayesian'
       model_name="xtft",   # Specify XTFT for the default builder 
       # ; change to model_name='super_xtft', for SuperXFT tuning 
       verbose=1            # Show some tuner progress
   )
   print("\nXTFT Tuning complete.")

Step 5: Display Results
~~~~~~~~~~~~~~~~~~~~~~~~~
The tuner returns the best hyperparameters found, the corresponding
fully trained model, and the Keras Tuner object.

.. code-block:: python
   :linenos:

   if best_hps:
       print("\n--- Best Hyperparameters Found ---")
       for param, value in best_hps.items():
           print(f"  {param}: {value}")
       print(f"\nOptimal Batch Size (among tested): "
             f"{best_hps.get('batch_size', 'N/A')}")

       print("\n--- Summary of Best Model Architecture ---")
       if best_model:
           best_model.summary(line_length=100)
       else:
           print("Best model was not returned from tuning.")
   else:
       print("Tuning did not yield best hyperparameters (e.g., all trials failed).")

   # For more details, you can inspect the tuner object:
   # if tuner:
   #     tuner.results_summary()

   # The `best_model` can now be used for forecasting or saved.


