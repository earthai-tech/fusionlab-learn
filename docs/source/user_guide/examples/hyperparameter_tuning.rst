.. _example_hyperparameter_tuning:

============================
Hyperparameter Tuning Example
============================

Finding optimal hyperparameters is crucial for getting the best
performance from models like :class:`~fusionlab.nn.XTFT` and
:class:`~fusionlab.nn.TemporalFusionTransformer`. ``fusionlab`` provides
convenient wrappers around the `Keras Tuner <https://keras.io/keras_tuner/>`_
library to automate this search process.

This example demonstrates how to use :func:`~fusionlab.nn.forecast_tuner.xtft_tuner`
to tune an XTFT model for quantile forecasting.

Prerequisites
-------------

Ensure you have installed `keras-tuner`:

.. code-block:: bash

   pip install keras-tuner -q

Code Example
------------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import os
   import joblib
   import keras_tuner as kt # Import Keras Tuner

   # Assuming fusionlab components are importable
   from fusionlab.nn.transformers import XTFT # Model to tune
   from fusionlab.nn.forecast_tuner import xtft_tuner # Tuner function
   from fusionlab.utils.ts_utils import reshape_xtft_data # Data prep utility
   from fusionlab.nn.losses import combined_quantile_loss # Loss for model

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)

   # --- Configuration ---
   output_dir = "./tuning_example_output" # For tuner logs/results
   os.makedirs(output_dir, exist_ok=True)

   # 1. Prepare Data (Similar to advanced_forecasting_xtft example)
   # ------------------------------------------------------------
   # Generate/Load data, define features, scale, reshape into sequences
   # (Using placeholder generation for brevity - replace with full prep)

   print("Preparing data (using placeholder logic)...")
   # Placeholder shapes - replace with actual data loading and reshaping
   n_samples_total = 100
   time_steps = 12
   forecast_horizons = 6
   # Assume these result from reshape_xtft_data
   static_data = np.random.rand(n_samples_total, 2).astype(np.float32) # (Samples, StaticFeats)
   dynamic_data = np.random.rand(n_samples_total, time_steps, 5).astype(np.float32) # (Samples, T, DynFeats)
   future_data = np.random.rand(n_samples_total, time_steps, 3).astype(np.float32) # (Samples, T, FutFeats)
   target_data = np.random.rand(n_samples_total, forecast_horizons, 1).astype(np.float32) # (Samples, H, 1)

   # Split into Train/Validation (simple split for demo)
   val_split_fraction = 0.25
   split_idx = int(n_samples_total * (1 - val_split_fraction))
   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   X_train_future, X_val_future = future_data[:split_idx], future_data[split_idx:]
   y_train, y_val = target_data[:split_idx], target_data[split_idx:]

   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future] # Needed for internal validation fit
   print("Data prepared and split.")

   # 2. Define Quantiles and Base Case Info
   # --------------------------------------
   quantiles_to_predict = [0.1, 0.5, 0.9]
   case_info = { # Passed to tuner, used by default builder
       'quantiles': quantiles_to_predict,
       'forecast_horizon': forecast_horizons
   }

   # 3. Define Hyperparameter Search Space (Optional)
   # -----------------------------------------------
   # Override or narrow down default search ranges if desired
   # Keys should match parameters of XTFT or Adam optimizer
   custom_param_space = {
       'hidden_units': [32, 64], # Try only 32 or 64 units
       'num_heads': [2, 4],      # Try 2 or 4 heads
       'learning_rate': [0.01, 0.005, 0.001], # Specific learning rates
       # 'dropout_rate': [0.1], # Fix dropout if needed
       # Other parameters will use defaults from forecast_tuner.DEFAULT_PS
       # For example: 'embed_dim', 'lstm_units', 'attention_units', etc.
   }
   print("Defined custom hyperparameter search space (subset).")

   # 4. Run the Tuner
   # ----------------
   print("Starting hyperparameter tuning...")
   best_hps, best_model, tuner = xtft_tuner(
       inputs=train_inputs, # Use training data for search
       y=y_train,
       param_space=custom_param_space, # Provide custom space
       forecast_horizon=forecast_horizons, # Needed by builder
       quantiles=quantiles_to_predict,   # Needed by builder for loss
       case_info=case_info, # Pass case info
       max_trials=4,        # Number of HP combinations to try (low for demo)
       objective='val_loss', # Metric to optimize
       epochs=10,           # Epochs for FULL training run AFTER search for a batch size
                            # Note: Keras Tuner might use fewer epochs during the SEARCH phase itself
       batch_sizes=[32, 64],# List of batch sizes to try
       validation_split=val_split_fraction, # Use same split for internal validation fit
       tuner_dir=output_dir, # Directory to store results
       project_name="XTFT_Quantile_Tuning_Example",
       tuner_type='random', # Use 'random' or 'bayesian'
       verbose=1 # Show basic tuner progress logs
   )

   print("\nHyperparameter tuning finished.")

   # 5. Show Results
   # ---------------
   print("\n--- Best Hyperparameters Found ---")
   # best_hps is a dictionary
   for param, value in best_hps.items():
       print(f"  {param}: {value}")

   print(f"\nAchieved best validation loss: {tuner.oracle.get_best_trials(1)[0].score:.4f}")
   print(f"Optimal Batch Size: {best_hps.get('batch_size', 'N/A')} "
         f"(Note: Tuner iterates batches; best overall HP set is returned)")


   print("\n--- Summary of the Best Trained Model ---")
   best_model.summary() # Display architecture of the best model

   # You can explore more results via the tuner object
   # tuner.results_summary()

   # The 'best_model' is fully trained and ready for evaluation or prediction
   # Example: Predict on validation set with the best model
   # predictions = best_model.predict(val_inputs)


.. topic:: Explanations

   1.  **Imports & Config:** Import standard libraries, `XTFT` model,
       the `xtft_tuner` function (or `tft_tuner` if tuning TFT),
       data utilities, Keras Tuner (`kt`), and loss functions. Define
       an output directory for tuner logs.
   2.  **Data Preparation:** Load, preprocess, scale, and reshape your
       time series data into the required sequence format
       (`static_data`, `dynamic_data`, `future_data`, `target_data`).
       This typically involves using functions like
       :func:`~fusionlab.utils.ts_utils.reshape_xtft_data`. Split the
       data into training and validation sets *before* passing the
       training portions (`train_inputs`, `y_train`) to the tuner.
       The `validation_split` argument within the tuner function is
       used for internal validation during the hyperparameter search
       and the final training run for each batch size candidate.
   3.  **Define Quantiles & Case Info:** Specify the `quantiles` if doing
       quantile forecasting. The `case_info` dictionary passes essential
       information like `forecast_horizon` and `quantiles` to the
       internal default model builder (`_model_builder_factory`).
   4.  **Define Search Space (Optional):** The tuner functions use a
       default search space (`DEFAULT_PS`) for common hyperparameters.
       You can provide a `param_space` dictionary to override or add
       to this space. Use Keras Tuner syntax implicitly (e.g., a list
       like `[16, 32, 64]` implies `hp.Choice`).
   5.  **Run Tuner:** Call
