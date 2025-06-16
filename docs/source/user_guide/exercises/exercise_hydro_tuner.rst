.. _exercise_hydro_tuner_guide:

==================================================
Exercise: Hyperparameter Tuning with HydroTuner
==================================================

Welcome to this hands-on exercise on using the
:class:`~fusionlab.nn.forecast_tuner.HydroTuner`. This guide will
walk you through the complete, end-to-end process of finding an
optimal set of hyperparameters for a sophisticated physics-informed
model, ``PIHALNet``.

This tutorial will demonstrate the full power and flexibility of the
tuner, including how to define a custom search space and how to launch
the tuning process using both the high-level convenience methods and
the lower-level core API.

**Learning Objectives:**

* Generate a synthetic hybrid dataset with features and coordinates suitable
  for a PINN model.
* Define a custom hyperparameter ``search_space``, including architectural,
  physical, and optimization parameters.
* Use the recommended ``HydroTuner.create()`` factory method to
  automatically infer data dimensions and configure the tuner.
* Launch the tuning process with the high-level ``.run()`` method, which
  accepts NumPy arrays directly.
* (Advanced) Understand how to manually prepare a ``tf.data.Dataset`` and
  use the lower-level ``.search()`` method.
* Retrieve the best hyperparameters and the best-performing model
  after the search is complete.

Let's begin!

Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and its dependencies, including
``keras-tuner``, installed.

.. code-block:: bash

   pip install fusionlab-learn "keras-tuner>=1.4.0" matplotlib scikit-learn



Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we import all necessary libraries and set up our environment.

.. code-block:: python
   :linenos:

   import os
   import numpy as np
   import tensorflow as tf
   import matplotlib.pyplot as plt

   # FusionLab imports
   from fusionlab.nn.forecast_tuner import HydroTuner
   from fusionlab.nn.pinn.models import PIHALNet # The model we'll tune

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')

   # Directory for saving tuner results and plots
   EXERCISE_OUTPUT_DIR = "./hydrotuner_exercise_outputs"
   os.makedirs(EXERCISE_OUTPUT_DIR, exist_ok=True)

   print("Libraries imported and setup complete.")


**Expected Output:**

.. code-block:: text

   Libraries imported and setup complete.

Step 2: Generate Synthetic Hybrid Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will create a synthetic dataset suitable for ``PIHALNet``, which
requires both feature-based inputs (static, dynamic, future) and a
`coords` tensor for the physics module. We will keep these as raw
NumPy arrays, as this is the expected format for the tuner's `.run()`
method.

.. code-block:: python
   :linenos:

   # Configuration
   N_SAMPLES = 800
   PAST_STEPS = 12
   HORIZON = 5
   STATIC_DIM, DYNAMIC_DIM, FUTURE_DIM = 4, 6, 3
   SEED = 42
   np.random.seed(SEED)
   tf.random.set_seed(SEED)

   # --- Generate Data Arrays ---
   inputs = {
       "coords": np.random.rand(N_SAMPLES, HORIZON, 3).astype(np.float32),
       "static_features": np.random.rand(N_SAMPLES, STATIC_DIM).astype(np.float32),
       "dynamic_features": np.random.rand(N_SAMPLES, PAST_STEPS, DYNAMIC_DIM).astype(np.float32),
       "future_features": np.random.rand(N_SAMPLES, HORIZON, FUTURE_DIM).astype(np.float32),
   }
   targets = {
       "subsidence": np.random.rand(N_SAMPLES, HORIZON, 1).astype(np.float32),
       "gwl": np.random.rand(N_SAMPLES, HORIZON, 1).astype(np.float32)
   }

   # Create a validation split
   val_split = -100
   train_inputs = {k: v[:val_split] for k, v in inputs.items()}
   val_inputs = {k: v[val_split:] for k, v in inputs.items()}
   train_targets = {k: v[:val_split] for k, v in targets.items()}
   val_targets = {k: v[val_split:] for k, v in targets.items()}

   print(f"Generated {len(train_inputs['static_features'])} training and "
         f"{len(val_inputs['static_features'])} validation samples.")

**Expected Output:**

.. code-block:: text

   Generated 700 training and 100 validation samples.

Step 3: Define the Tuning Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is where we tell the tuner *what* to optimize. We define a
``search_space`` dictionary containing all the hyperparameters we want
to explore.

.. code-block:: python
   :linenos:

   search_space = {
       # --- Architectural Hyperparameters ---
       "embed_dim": [16, 32, 64], # Discrete choice
       "num_heads": [2, 4],
       "dropout_rate": {"type": "float", "min_value": 0.05, "max_value": 0.3},

       # --- Physics-Informed Hyperparameters ---
       # Tune whether the coefficient is fixed or learned
       "pinn_coefficient_C": ["learnable", 1e-3, 5e-3],
       # The lambda weight for the physics loss
       "lambda_physics": {"type": "float", "min_value": 0.05, "max_value": 0.5},

       # --- Optimization Hyperparameters ---
       "learning_rate": {"type": "choice", "values": [1e-3, 5e-4, 1e-4]}
   }
   print("Hyperparameter search space defined.")

Step 4: Launch the Search with the High-Level `.run()` Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the **recommended workflow**. We use the ``HydroTuner.create()``
factory method, which automatically inspects our data to determine fixed
parameters like input/output dimensions. We then call ``.run()``, which
handles the conversion of our NumPy arrays into `tf.data.Dataset`
objects internally.

.. code-block:: python
   :linenos:

   # 1. Create the tuner using the factory method
   tuner = HydroTuner.create(
       model_name_or_cls=PIHALNet,
       inputs_data=train_inputs,
       targets_data=train_targets,
       search_space=search_space,
       # Keras Tuner configuration
       objective="val_loss",
       max_trials=5, # Keep low for this example
       project_name="PIHALNet_Tuning_Exercise_Run",
       directory=EXERCISE_OUTPUT_DIR,
       overwrite=True
   )

   # 2. Start the search by calling .run()
   print("\nStarting hyperparameter search with the .run() method...")
   best_model, best_hps, tuner_instance = tuner.run(
       inputs=train_inputs,
       y=train_targets,
       validation_data=(val_inputs, val_targets),
       epochs=5, # Train each trial for 5 epochs
       batch_size=64,
       callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)]
   )

   print("\n--- Search via .run() Complete ---")
   if best_hps:
       print("Best learning rate found:", best_hps.get('learning_rate'))

Step 5 (Advanced): Using the Low-Level `.search()` Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section demonstrates the alternative workflow for users who prefer
to manage their data pipelines manually. Here, we first create
`tf.data.Dataset` objects ourselves and then instantiate the tuner
using its direct ``__init__``, which requires us to provide the complete
set of `fixed_params` manually.

.. code-block:: python
   :linenos:

   # 1. Manually prepare tf.data.Dataset objects
   # Note: The tuner's internal logic renames target keys automatically
   train_dataset = tf.data.Dataset.from_tensor_slices(
       (train_inputs, train_targets)).batch(64)
   val_dataset = tf.data.Dataset.from_tensor_slices(
       (val_inputs, val_targets)).batch(64)

   # 2. Manually define ALL fixed parameters (what .create() does for us)
   manual_fixed_params = {
       "static_input_dim": STATIC_DIM,
       "dynamic_input_dim": DYNAMIC_DIM,
       "future_input_dim": FUTURE_DIM,
       "output_subsidence_dim": 1,
       "output_gwl_dim": 1,
       "forecast_horizon": HORIZON,
       "mode": 'pihal_like' # An example of another fixed param
   }

   # 3. Instantiate the tuner directly
   tuner_adv = HydroTuner(
       model_name_or_cls=PIHALNet,
       fixed_params=manual_fixed_params,
       search_space=search_space,
       objective="val_loss",
       max_trials=5,
       project_name="PIHALNet_Tuning_Exercise_Search",
       directory=EXERCISE_OUTPUT_DIR,
       overwrite=True
   )

   # 4. Start the search by calling the base .search() method
   print("\nStarting hyperparameter search with the .search() method...")
   # Note: .search() is called by the inherited .fit() from PINNTunerBase
   _, _, tuner_instance_adv = tuner_adv.run(
       inputs=train_inputs, # Still needed for case info
       y=train_targets,
       validation_data=(val_inputs, val_targets), # passed to search()
       epochs=5,
       batch_size=64 # used by the wrapper
   )
   print("\n--- Search via .search() Complete ---")

Step 6: Analyze Results and Retrieve the Best Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the search completes (either way), the tuner object contains the
results. You can retrieve the best hyperparameters and the best model,
which has been automatically retrained on the full dataset.

.. code-block:: python
   :linenos:

   print("\n--- Summary of Best Hyperparameters ---")
   # Get the best hyperparameters from the first tuner run
   best_hps_found = tuner.get_best_hyperparameters(num_trials=1)[0]
   for hp, value in best_hps_found.values.items():
       print(f"- {hp}: {value}")

   # Get the best model instance
   best_pihalnet_model = tuner.get_best_models(num_models=1)[0]

   # You can now use this model for prediction
   print(f"\nBest model summary:")
   best_pihalnet_model.summary(line_length=100)

Discussion of Exercise
~~~~~~~~~~~~~~~~~~~~~~~~
Congratulations! You have successfully performed a full hyperparameter
tuning workflow for an advanced physics-informed model. In this exercise,
you have learned to:

* Define a flexible `search_space` to control which parameters are
    tuned.
* Use the high-level `HydroTuner.create()` and `.run()` methods for a
    convenient, automated workflow with NumPy data.
* (Advanced) Understand the lower-level process of manually creating
    datasets and using the core `.search()` method.
* Retrieve the final, optimized model and its hyperparameters from the
    tuner instance.

This process is fundamental to achieving peak performance with complex
deep learning architectures and allows you to systematically find the
best configuration for your specific problem.