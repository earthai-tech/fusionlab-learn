.. _pihal_tuner_legacy_guide:

===============================================
Hyperparameter Tuning (Legacy PiHALTuner)
===============================================

This section details the legacy ``PiHALTuner``, a class designed
specifically for hyperparameter optimization of the original
:class:`~fusionlab.nn.pinn.models.legacy.PiHALNet` model.

.. warning::
   This tuner is specific to the legacy ``PiHALNet`` model. For tuning
   the modern, ``BaseAttentive``-based models like ``PIHALNet`` and
   ``TransFlowSubsNet``, please use the more flexible
   :class:`~fusionlab.nn.forecast_tuner.HydroTuner`.

The ``PiHALTuner`` automates the process of searching for the best
architectural and physics-informed hyperparameters using the
`Keras Tuner <https://keras.io/keras_tuner/>`_ backend.

Core Concepts
-------------

The design of ``PiHALTuner`` is centered on separating the static,
unchanging aspects of a model from the parameters you wish to optimize.

* **Fixed Parameters (`fixed_model_params`):** This is a dictionary
    that defines the "problem." It holds all the parameters that will
    **not** be tuned, primarily the data-dependent dimensions like
    `static_input_dim`, `dynamic_input_dim`, and `forecast_horizon`.
    These are constant for a given tuning job.

* **Hyperparameter Space (`param_space`):** This dictionary allows you
    to **override** the tuner's default search space for any given
    hyperparameter. For example, while the tuner might search for a
    `dropout_rate` between 0.0 and 0.3 by default, you can provide
    `param_space={'dropout_rate': [0.1, 0.15]}` to test only those
    two specific values.

End-to-End Workflow
-------------------

The primary workflow involves using the ``PiHALTuner.create()`` factory
method. This simplifies setup by automatically inferring data dimensions
and merging them with defaults and user-provided parameters.

### Step 1: Prepare Data
First, prepare your input features and target variables as dictionaries
of NumPy arrays.

.. code-block:: python
   :linenos:

   import numpy as np

   # Define data dimensions for the example
   B, T, H = 128, 12, 5
   S_DIM, D_DIM, F_DIM = 4, 6, 3
   O_DIM = 1

   # Generate dummy data arrays
   inputs = {
       "coords": np.random.rand(B, H, 3).astype(np.float32),
       "static_features": np.random.rand(B, S_DIM).astype(np.float32),
       "dynamic_features": np.random.rand(B, T, D_DIM).astype(np.float32),
       "future_features": np.random.rand(B, H, F_DIM).astype(np.float32),
   }
   targets = {
       "subsidence": np.random.rand(B, H, O_DIM).astype(np.float32),
       "gwl": np.random.rand(B, H, O_DIM).astype(np.float32)
   }

### Step 2: Create the Tuner with `.create()`

The ``.create()`` method is the recommended way to instantiate the
tuner. It inspects your data to determine the necessary dimensions,
combines them with sensible defaults, and prepares the tuner for the
search. You can still provide a `fixed_model_params` dictionary to
override any specific default or inferred values.

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.tuning import PiHALTuner

   # The .create() method infers dimensions from data
   tuner = PiHALTuner.create(
       inputs_data=inputs,
       targets_data=targets,
       # You can override specific fixed params if needed
       fixed_model_params={'quantiles': [0.1, 0.5, 0.9]},
       # Define a custom search space for specific HPs
       param_space={
           'learning_rate': [1e-3, 5e-4],
           'lambda_pde': {'type': 'float', 'min_value': 0.05, 'max_value': 0.5}
       },
       # Keras Tuner settings
       objective='val_loss',
       max_trials=10,
       project_name="Legacy_PIHALNet_Tuning",
       directory="./pihal_tuner_results",
       overwrite=True
   )
   print("Tuner created and configured.")

### Step 3: Run the Hyperparameter Search
Call the ``.run()`` method (an alias for `.fit()`) to start the tuning
process. This method handles the creation of `tf.data.Dataset` objects
internally and executes the Keras Tuner search loop.

.. code-block:: python
   :linenos:
   
   import tensorflow as tf

   # Split data for validation
   val_split = -20
   train_inputs = {k: v[:val_split] for k, v in inputs.items()}
   val_inputs = {k: v[val_split:] for k, v in inputs.items()}
   train_targets = {k: v[:val_split] for k, v in targets.items()}
   val_targets = {k: v[val_split:] for k, v in targets.items()}
   
   # Run the search
   best_model, best_hps, tuner_instance = tuner.run(
       inputs=train_inputs,
       y=train_targets,
       validation_data=(val_inputs, val_targets),
       epochs=20, # Max epochs for each trial
       batch_size=32,
       callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)]
   )

### Step 4: Analyze Results
After the search, the tuner object contains the best hyperparameters
and a model instance retrained on the full dataset using those settings.

.. code-block:: python
   :linenos:

   print("\n--- Tuning Complete: Best Hyperparameters ---")
   if best_hps:
       for hp, value in best_hps.values.items():
           if isinstance(value, float):
               print(f"  - {hp}: {value:.5f}")
           else:
               print(f"  - {hp}: {value}")
   
       # The best model is ready to be saved or used for prediction
       # best_model.save("best_legacy_pihalnet_model.keras")

API Reference
-------------

.. autoclass:: fusionlab.nn.pinn.tuning.PiHALTuner
   :members: create, run, build
   :undoc-members: