.. _example_hyperparameter_tuning:

==============================
Hyperparameter Tuning Example
==============================

Finding optimal hyperparameters is crucial for getting the best
performance from models like :class:`~fusionlab.nn.XTFT` and
:class:`~fusionlab.nn.TemporalFusionTransformer`. ``fusionlab`` provides
convenient wrappers around the `Keras Tuner <https://keras.io/keras_tuner/>`_
library to automate this search process.

This example demonstrates how to use
:func:`~fusionlab.nn.forecast_tuner.xtft_tuner` to tune an XTFT model
for quantile forecasting. The process for using
:func:`~fusionlab.nn.forecast_tuner.tft_tuner` is analogous.

Prerequisites
-------------

Ensure you have installed `keras-tuner`:

.. code-block:: bash

   pip install keras-tuner -q

Step 1: Imports and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Import necessary libraries including the model to tune (e.g., `XTFT`),
the corresponding tuner function (`xtft_tuner`), data preparation
utilities (`reshape_xtft_data`), Keras Tuner (`kt`), and standard data
science libraries. Set up an output directory for tuner results.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import os
   import joblib
   import keras_tuner as kt # Keras Tuner

   from fusionlab.nn.transformers import XTFT # Model to tune
   from fusionlab.nn.forecast_tuner import xtft_tuner # Tuner function
   from fusionlab.nn.utils import reshape_xtft_data # Data prep
   from fusionlab.nn.losses import combined_quantile_loss # For context

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # --- Configuration ---
   output_dir = "./tuning_example_output" # For tuner logs/results
   os.makedirs(output_dir, exist_ok=True)
   print(f"Tuner logs will be saved to: {output_dir}")


Step 2: Prepare Data
~~~~~~~~~~~~~~~~~~~~~~~
Load, preprocess, scale, and reshape your data into the required
sequence format (`static_data`, `dynamic_data`, `future_data`,
`target_data`) using appropriate utilities like
:func:`~fusionlab.utils.ts_utils.reshape_xtft_data`. Split the data
into training and validation sets **before** passing it to the tuner.
The tuner will use the training portions for its search and internal
final model training, evaluating on the validation split you define
within the tuner arguments.

*(Note: Placeholder data generation is used here for brevity. Replace
this with your actual data pipeline, similar to the Data Preparation
Workflow example)*.

.. code-block:: python
   :linenos:

   print("Preparing data (using placeholder logic)...")
   # Placeholder shapes & data
   B, T, H = 8, 12, 6
   D_stat, D_dyn, D_fut = 3, 5, 2
   T_future_total = T + H # Example shape for future inputs
   n_samples_total = 50 # Fewer samples for faster demo

   static_data = np.random.rand(n_samples_total, D_stat).astype(np.float32)
   dynamic_data = np.random.rand(n_samples_total, T, D_dyn).astype(np.float32)
   future_data = np.random.rand(n_samples_total, T_future_total, D_fut).astype(np.float32)
   target_data = np.random.rand(n_samples_total, H, 1).astype(np.float32)

   # Split into Train/Validation (simple split for demo)
   val_split_fraction = 0.3 # Use 30% for final validation by tuner
   split_idx = int(n_samples_total * (1 - val_split_fraction))

   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   # IMPORTANT: Ensure future data passed to tuner has correct time dim expected by model builder
   # Assuming builder needs T for future input context during LSTM phase
   X_train_future, X_val_future = future_data[:split_idx, :T, :], future_data[split_idx:, :T, :]
   y_train, y_val = target_data[:split_idx], target_data[split_idx:]

   # Package inputs for the tuner function
   # Order needs to match tuner's internal model builder
   # Assuming [Static, Dynamic, Future] order for this example
   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   # Validation data (X_val_*, y_val) is used internally by tuner if validation_split is set

   print(f"Data prepared and split. Training samples: {split_idx}")


Step 3: Define Quantiles and Case Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specify the `quantiles` for probabilistic forecasting (or `None` for
point forecasting). The `case_info` dictionary passes essential fixed
parameters like `forecast_horizon` and `quantiles` to the tuner's
internal model builder function.

.. code-block:: python
   :linenos:

   quantiles_to_predict = [0.1, 0.5, 0.9]
   forecast_horizons = H # From data prep step

   case_info = {
       'quantiles': quantiles_to_predict,
       'forecast_horizon': forecast_horizons,
       # Add any other FIXED parameters the model builder needs
       'static_input_dim': D_stat,
       'dynamic_input_dim': D_dyn,
       'future_input_dim': D_fut,
       'output_dim': 1
   }
   print(f"Defined case info: Quantiles={case_info['quantiles']}, "
         f"Horizon={case_info['forecast_horizon']}")


Step 4: Define Hyperparameter Search Space (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The tuner uses a default search space. You can override parts of it by
providing a `param_space` dictionary. Keys should match the names of
hyperparameters accepted by the target model (e.g., `XTFT`) or the
optimizer (e.g., `learning_rate`). Use lists for `hp.Choice`.

.. code-block:: python
   :linenos:

   # Override or narrow down default search ranges
   custom_param_space = {
       'hidden_units': [16, 32],    # Try only 16 or 32 units
       'num_heads': [1, 2],         # Try 1 or 2 heads
       'learning_rate': [1e-3, 5e-4] # Try two specific learning rates
       # 'dropout_rate': [0.1]     # Example: Fix dropout rate
   }
   print("Defined custom hyperparameter search space (subset).")


Step 5: Run the Tuner
~~~~~~~~~~~~~~~~~~~~~~
Call the appropriate tuner function (`xtft_tuner` or `tft_tuner`).
Provide the training data (`inputs`, `y`), the search space, case info,
and tuning configuration like `max_trials` (per batch size), `epochs`
(for final training run per batch size), `batch_sizes` (list to try),
`validation_split` (used on provided training data), `objective`,
output directory, project name, and tuner type (`'random'` or `'bayesian'`).

.. code-block:: python
   :linenos:

   # Tuning Parameters
   output_dir = "./xtft_tuning_output"
   project_name = "XTFT_Quantile_Tuning_Example"
   max_trials = 4         # Low for demo (try more combinations)
   epochs_per_run = 5     # Low for demo (epochs for final train of best HP per batch)
   batch_sizes_to_try = [8, 16] # Example batch sizes

   print(f"\nStarting XTFT tuning (Max Trials={max_trials} per batch size)...")
   best_hps, best_model, tuner = xtft_tuner(
       inputs=train_inputs,        # Training data (list)
       y=y_train,                  # Training targets
       param_space=custom_param_space, # Optional custom search space
       # forecast_horizon=forecast_horizons, # Now in case_info
       # quantiles=quantiles_to_predict,   # Now in case_info
       case_info=case_info,        # Pass fixed info
       max_trials=max_trials,
       objective='val_loss',       # Optimize validation loss
       epochs=epochs_per_run,
       batch_sizes=batch_sizes_to_try,
       validation_split=val_split_fraction, # Fraction of train data for tuner validation
       tuner_dir=output_dir,
       project_name=project_name,
       tuner_type='random',        # 'random' or 'bayesian'
       model_name="xtft",          # Ensures XTFT is built internally
       verbose=0                   # Set > 0 for more Keras Tuner logs
   )
   print("\nHyperparameter tuning finished.")


Step 6: Show Results
~~~~~~~~~~~~~~~~~~~~~
The tuner function returns the best hyperparameters found (`best_hps`
dictionary), the corresponding fully trained model (`best_model`), and
the Keras Tuner object (`tuner`) for further inspection.

.. code-block:: python
   :linenos:

   print("\n--- Best Hyperparameters Found ---")
   # best_hps is a dictionary combining model HPs and batch size
   if best_hps:
       for param, value in best_hps.items():
           print(f"  {param}: {value}")
       print(f"\nOptimal Batch Size (among tested): {best_hps.get('batch_size', 'N/A')}")
   else:
       print("Tuning did not complete successfully or find best HPs.")

   # Display summary of the best model found and trained
   print("\n--- Summary of the Best Trained Model ---")
   if best_model:
       best_model.summary()
       # This model is ready for evaluation on a hold-out test set
       # e.g., test_loss = best_model.evaluate(val_inputs, y_val)
   else:
       print("Tuning did not return a best model.")

   # You can explore more results via the tuner object if needed
   # Example: Display top N trials
   # if tuner:
   #     tuner.results_summary(num_trials=3)