.. _user_guide_forecast_tuner:

=======================
Hyperparameter Tuning
=======================

Finding the optimal set of hyperparameters for deep learning models
like Temporal Fusion Transformer (TFT) and its variants (XTFT,
SuperXTFT) is crucial for achieving the best possible forecasting
performance. Hyperparameters control aspects of the model
architecture (e.g., number of hidden units, attention heads) and the
training process (e.g., learning rate, batch size).

``fusionlab`` provides utility functions that leverage the powerful
**Keras Tuner** library (`keras-tuner`) to automate this search
process for its forecasting models.

Prerequisites
-------------

To use the tuning functions, you must have Keras Tuner installed in
your environment:

.. code-block:: bash

   pip install keras-tuner -q

Tuner Functions
-----------------

The `fusionlab.nn.forecast_tuner` module offers dedicated functions
to tune different model types.

.. _xtft_tuner:

xtft_tuner
~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.forecast_tuner.xtft_tuner`

**Purpose:** To perform hyperparameter optimization for the
:class:`~fusionlab.nn.XTFT` and :class:`~fusionlab.nn.SuperXTFT`
models using Keras Tuner.

**Functionality:**
This function orchestrates the tuning process:

1.  **Inputs:** Takes the prepared input data as a list of NumPy
    arrays `inputs = [X_static, X_dynamic, X_future]` and the
    corresponding target array `y`.
2.  **Search Space:** Defines a search space for hyperparameters.
    It uses a default space (`DEFAULT_PS` defined internally) for
    parameters like `embed_dim`, `lstm_units`, `num_heads`,
    `dropout_rate`, `learning_rate`, etc. Users can provide their
    own `param_space` dictionary to override or extend these defaults.
3.  **Model Builder:** Uses a function (either a user-provided
    `model_builder` or the internal default `_model_builder_factory`)
    that defines how to construct an XTFT or SuperXTFT model instance
    for a given set of hyperparameters (`hp`). The builder uses
    Keras Tuner's `hp` object (e.g., `hp.Choice`, `hp.Boolean`) to
    sample values from the defined search space. The model is
    compiled within the builder, typically using Adam optimizer and
    an appropriate loss (MSE for point forecasts, or
    :func:`~fusionlab.nn.losses.combined_quantile_loss` if `quantiles`
    are provided).
4.  **Tuner Initialization:** Creates a Keras Tuner instance based on
    the `tuner_type` parameter:
    * `'bayesian'`: Uses `keras_tuner.BayesianOptimization`.
    * `'random'` (Default): Uses `keras_tuner.RandomSearch`.
    The tuner is configured with the `objective` to optimize (e.g.,
    `'val_loss'`), the maximum number of trials (`max_trials`), and
    storage locations (`tuner_dir`, `project_name`).
5.  **Search Execution:** Iterates through the specified list of
    `batch_sizes`. For each batch size:
    * Runs `tuner.search()` using the provided data (`inputs`, `y`),
        `epochs`, `validation_split`, and `callbacks` (defaults to
        `EarlyStopping`). Keras Tuner repeatedly calls the model
        builder, trains the model for a few epochs (as defined within
        the search or implicitly by Keras Tuner), and evaluates its
        performance on the validation set.
    * After the search for a given batch size, it retrieves the best
        hyperparameters found (`current_hps`) for that batch size.
    * It then builds and fully trains a model using these `current_hps`
        and the current `batch_size` for the specified number of `epochs`.
6.  **Best Model Selection:** Compares the final validation loss
    achieved across all tested `batch_sizes`. It keeps track of the
    hyperparameters (`best_hps`), the fully trained model (`best_model`),
    and the batch size (`best_batch`) that resulted in the overall lowest
    validation loss.
7.  **Output:** Returns a tuple containing the overall `best_hps`
    (dictionary), the corresponding fully trained `best_model`
    (Keras model object), and the `tuner` object itself (which can be
    used for further inspection). Tuning results are also logged to a
    file in the `tuner_dir`.

**Usage Context:** Call this function after preparing your training data
into the required `[X_static, X_dynamic, X_future]` format. Provide
your data, specify the `forecast_horizon` and `quantiles` (if any),
and optionally customize the `param_space`, `max_trials`, `epochs`,
etc. The function automates the search for good hyperparameters for
XTFT or SuperXTFT models tailored to your data.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   import os
   # Assuming tuner and model are importable
   from fusionlab.nn.forecast_tuner import xtft_tuner
   from fusionlab.nn import XTFT # Needed for type hints/context

   # 1. Prepare Dummy Data (Replace with your actual data prep)
   # Needs static, dynamic (past), and future known inputs
   B, T, H = 8, 12, 6  # Batch, TimeSteps, Horizon
   D_stat, D_dyn, D_fut = 3, 5, 2 # Feature dimensions
   T_future_total = T + H # Example length for future features

   X_static_train = np.random.rand(B, D_stat).astype(np.float32)
   X_dynamic_train = np.random.rand(B, T, D_dyn).astype(np.float32)
   # Future data needs correct shape for internal builder
   X_future_train = np.random.rand(B, T_future_total, D_fut).astype(np.float32)
   y_train = np.random.rand(B, H, 1).astype(np.float32) # Point forecast target

   # Tuner expects inputs as list [Static, Dynamic, Future]
   # Note: Order depends on internal model builder assumptions
   # Assuming [Static, Dynamic, Future] order for this example
   train_inputs = [X_static_train, X_dynamic_train, X_future_train]

   # 2. Define Search Space (Optional - overrides defaults)
   custom_param_space = {
       'hidden_units': [16, 32],    # Try fewer units
       'num_heads': [2],            # Fix number of heads
       'lstm_units': [16],          # Fix LSTM units
       'learning_rate': [1e-3, 5e-4] # Try two learning rates
   }

   # 3. Define Tuning Parameters
   output_dir = "./xtft_tuning_output"
   project_name = "XTFT_Point_Forecast_Tuning"
   max_trials = 5 # Reduce for quick demo
   epochs_per_run = 10 # Epochs for final training of best HP per batch size
   batch_sizes_to_try = [16, 32] # Try two batch sizes

   # 4. Run the Tuner
   print("Starting XTFT tuning...")
   best_hps, best_model, tuner = xtft_tuner(
       inputs=train_inputs,
       y=y_train,
       param_space=custom_param_space,
       forecast_horizon=H,
       quantiles=None, # Point forecast for this example
       max_trials=max_trials,
       objective='val_loss', # Optimize validation loss
       epochs=epochs_per_run,
       batch_sizes=batch_sizes_to_try,
       validation_split=0.25, # Use 25% of input data for validation
       tuner_dir=output_dir,
       project_name=project_name,
       tuner_type='random', # Use 'random' or 'bayesian' search
       model_name="xtft", # Specify model type for default builder
       verbose=0 # Set to 1 or higher for detailed logs
   )

   # 5. Display Results
   print("\nTuning complete.")
   print("\n--- Best Hyperparameters Found ---")
   print(best_hps) # Dictionary of best HPs
   print(f"\nOptimal Batch Size Found (among tested): {best_hps.get('batch_size', 'N/A')}")

   print("\n--- Summary of Best Model Architecture ---")
   if best_model:
       best_model.summary()
   else:
       print("Tuning failed to find a best model.")

   # Further analysis can be done using the 'tuner' object
   # tuner.results_summary()

.. raw:: html

   <hr>

.. _tft_tuner:

tft_tuner
~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.forecast_tuner.tft_tuner`

**Purpose:** To perform hyperparameter optimization specifically for
the :class:`~fusionlab.nn.TemporalFusionTransformer` model using
Keras Tuner.

**Functionality:**
This function acts as a convenient wrapper around :func:`xtft_tuner`.
It accepts the same parameters but internally calls `xtft_tuner` with
`model_name="tft"`.

This ensures that the internal default model builder
(`_model_builder_factory`) constructs a `TemporalFusionTransformer`
instance and uses hyperparameters relevant to it (e.g., sampling
`num_lstm_layers` instead of XTFT-specific parameters like
`embed_dim` or `memory_size`, although some overlap exists in the
default search space).

**Usage Context:** Use this function similarly to `xtft_tuner`, but
when your goal is specifically to tune a standard
`TemporalFusionTransformer` model.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   import os
   from fusionlab.nn.forecast_tuner import tft_tuner # Use TFT tuner
   from fusionlab.nn import TemporalFusionTransformer # Model context

   # 1. Prepare Dummy Data (Same as XTFT example for consistency)
   print("Preparing data (using placeholder logic)...")
   B, T, H = 8, 12, 6
   D_stat, D_dyn, D_fut = 3, 5, 2
   T_future_total = T + H
   X_static_train = np.random.rand(B, D_stat).astype(np.float32)
   X_dynamic_train = np.random.rand(B, T, D_dyn).astype(np.float32)
   X_future_train = np.random.rand(B, T_future_total, D_fut).astype(np.float32)
   # Example for quantile forecast
   y_train_quant = np.random.rand(B, H, 1).astype(np.float32)
   quantiles_to_predict = [0.1, 0.5, 0.9]

   # Assuming [Static, Dynamic, Future] input order for tuner's default builder
   train_inputs_tft = [X_static_train, X_dynamic_train, X_future_train]

   # 2. Define Search Space (Optional - focus on TFT params)
   tft_param_space = {
       'hidden_units': [16, 32],
       'num_heads': [2],
       'num_lstm_layers': [1, 2], # Specific to standard TFT builder
       'learning_rate': [1e-3, 5e-4]
   }

   # 3. Define Tuning Parameters
   output_dir_tft = "./tft_tuning_output"
   project_name_tft = "TFT_Quantile_Tuning_Example"
   max_trials_tft = 4
   epochs_per_run_tft = 10
   batch_sizes_tft = [32] # Test single batch size

   # 4. Run the TFT Tuner
   print("Starting TFT tuning...")
   best_hps_tft, best_model_tft, tuner_tft = tft_tuner(
       inputs=train_inputs_tft,
       y=y_train_quant,
       param_space=tft_param_space,
       forecast_horizon=H,
       quantiles=quantiles_to_predict, # Quantile forecast
       max_trials=max_trials_tft,
       objective='val_loss',
       epochs=epochs_per_run_tft,
       batch_sizes=batch_sizes_tft,
       validation_split=0.25,
       tuner_dir=output_dir_tft,
       project_name=project_name_tft,
       tuner_type='random',
       verbose=0
   )

   # 5. Display Results
   print("\nTFT Tuning complete.")
   print("\n--- Best TFT Hyperparameters Found ---")
   print(best_hps_tft)
   print(f"\nOptimal Batch Size Found (among tested): {best_hps_tft.get('batch_size', 'N/A')}")

   print("\n--- Summary of Best TFT Model Architecture ---")
   if best_model_tft:
       best_model_tft.summary()
   else:
       print("Tuning failed to find a best TFT model.")


.. raw:: html

   <hr>
   
Internal Model Builder (`_model_builder_factory`)
--------------------------------------------------

*(Note: Users typically do not interact with this function directly,
but understanding its role is helpful).*

This internal helper function is used by default if no custom
`model_builder` is provided to the tuner functions. Its responsibilities
are:

1.  Accepts the Keras Tuner `hp` object.
2.  Determines the correct model class to instantiate (`XTFT`,
    `SuperXTFT`, or `TemporalFusionTransformer`) based on the
    `model_name`.
3.  Defines the range or set of choices for each hyperparameter
    relevant to the chosen model class, using `hp.Choice`, `hp.Boolean`,
    etc., based on the `param_space` provided to the tuner or the
    internal `DEFAULT_PS`.
4.  Instantiates the model class with the sampled hyperparameters.
5.  Compiles the model with an Adam optimizer (learning rate is also
    tuned) and an appropriate loss function (MSE or quantile loss).
6.  Returns the compiled model instance to the Keras Tuner for
    evaluation during the search process.

By providing a custom `model_builder` function to `xtft_tuner` or
`tft_tuner`, users can gain finer control over the architecture
variations or compilation settings explored during tuning.