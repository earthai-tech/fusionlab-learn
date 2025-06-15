.. _hydro_tuner_examples_guide:

=============================
HydroTuner: Usage Examples
=============================

This page provides practical, end-to-end examples for using the
:class:`~fusionlab.nn.forecast_tuner.HydroTuner` to find optimal
hyperparameters for the library's hydrogeological PINN models.

The core workflow involves three main steps:
1. Preparing your data as NumPy arrays.
2. Defining a `search_space` dictionary that specifies which
   hyperparameters to tune.
3. Using the ``HydroTuner.create()`` factory method to instantiate
   and run the tuner.

We will start with a comprehensive workflow for the fully-coupled
``TransFlowSubsNet`` model.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Example 1: A Comprehensive Workflow for Tuning `TransFlowSubsNet`
-----------------------------------------------------------------
In this example, we'll configure the ``HydroTuner`` to find the best
hyperparameters for a ``TransFlowSubsNet`` model. Our search space will
be comprehensive, including architectural parameters for the data-driven
core, physical parameters for the PINN module, and optimization
parameters for the training process.

Step 1: Imports and Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, we set up our environment and generate a synthetic dataset. This
dataset mimics a real-world scenario by including all the data types
that ``TransFlowSubsNet`` is designed to handle.

.. code-block:: python
   :linenos:

   import os
   import numpy as np
   import tensorflow as tf

   # FusionLab imports
   from fusionlab.nn.forecast_tuner import HydroTuner
   from fusionlab.nn.pinn.models import TransFlowSubsNet

   # --- Configuration ---
   N_SAMPLES, T_PAST, HORIZON = 500, 15, 7
   S_DIM, D_DIM, F_DIM = 4, 6, 3
   RUN_DIR = "./hydrotuner_examples"

   # --- Generate Dummy Data as NumPy Arrays ---
   print("Generating synthetic data for tuning...")
   inputs = {
       # Coords (t,x,y) for the physics loss calculation
       "coords": np.random.rand(N_SAMPLES, HORIZON, 3).astype(np.float32),
       # Time-invariant features
       "static_features": np.random.rand(N_SAMPLES, S_DIM).astype(np.float32),
       # Historical time-varying features
       "dynamic_features": np.random.rand(N_SAMPLES, T_PAST, D_DIM).astype(np.float32),
       # Known future time-varying features
       "future_features": np.random.rand(N_SAMPLES, HORIZON, F_DIM).astype(np.float32),
   }
   targets = {
       # The two target variables to forecast
       "subsidence": np.random.rand(N_SAMPLES, HORIZON, 1).astype(np.float32),
       "gwl": np.random.rand(N_SAMPLES, HORIZON, 1).astype(np.float32)
   }
   print(f"Generated {N_SAMPLES} data samples.")

Step 2: Define the Hyperparameter Search Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This dictionary is the core of our tuning experiment. It tells the
``HydroTuner`` exactly which parameters to optimize and what values or
ranges to explore for each one. We define three types of hyperparameters:

* **Architectural HPs:** Control the size and capacity of the
    data-driven ``BaseAttentive`` core.
* **Physics HPs:** Control the physical coefficients in the PDEs,
    allowing the tuner to test fixed vs. learnable assumptions.
* **Compile-time HPs:** Control the optimization process, including
    the learning rate and the weights of the physics loss terms.

.. code-block:: python
   :linenos:

   transflow_search_space = {
       # --- Architectural Hyperparameters ---
       "embed_dim": [32, 64],
       "attention_units": [32, 64],
       "num_heads": [2, 4],
       "dropout_rate": {"type": "float", "min_value": 0.05, "max_value": 0.3},

       # --- Physics Hyperparameters for TransFlowSubsNet ---
       # Test if making K learnable is better than two fixed values
       "K": ["learnable", 1e-5, 1e-4],
       # Search for the best Ss value in a log space
       "Ss": {"type": "float", "min_value": 1e-6, "max_value": 1e-4, "sampling": "log"},
       # For this experiment, we'll fix C as learnable
       "pinn_coefficient_C": ["learnable"],

       # --- Compile-time Hyperparameters ---
       "learning_rate": {"type": "choice", "values": [1e-3, 5e-4, 1e-4]},
       # Search for the best weights for the two physics losses
       "lambda_gw": {"type": "float", "min_value": 0.5, "max_value": 1.5},
       "lambda_cons": {"type": "float", "min_value": 0.1, "max_value": 1.0}
   }
   print("Hyperparameter search space for TransFlowSubsNet defined.")

Step 3: Create and Run the Tuner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use the ``HydroTuner.create()`` factory method. This is the
recommended approach as it simplifies setup by automatically inspecting
our data to determine the fixed parameters (like input/output
dimensions). We then call the high-level ``.run()`` method, which handles
all the underlying data conversion and launches the Keras Tuner search.

.. code-block:: python
   :linenos:

   # 1. Create the tuner instance using the factory method
   tuner = HydroTuner.create(
       model_name_or_cls=TransFlowSubsNet,
       inputs_data=inputs,
       targets_data=targets,
       search_space=transflow_search_space,
       # Keras Tuner configuration
       max_trials=5, # Keep low for this example; use 30-50 for real tasks
       project_name="TransFlowSubsNet_Comprehensive_Tuning",
       directory=RUN_DIR,
       overwrite=True,
       objective="val_loss" # Monitor the total validation loss
   )

   # 2. Run the search process
   print("\nStarting hyperparameter search for TransFlowSubsNet...")
   best_model, best_hps, tuner_instance = tuner.run(
       inputs=inputs,
       y=targets,
       validation_data=(inputs, targets), # Use same data for example
       epochs=5, # Train each trial for a few epochs
       batch_size=64,
       callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)]
   )

Step 4: Analyze Results and Use the Best Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the search is complete, the tuner object provides access to the
best hyperparameters and the best model, which has been automatically
retrained on the full dataset for you.

.. code-block:: python
   :linenos:

   print("\n--- Hyperparameter Search Complete ---")

   if best_hps:
       print("\nBest Hyperparameters Found:")
       for hp, value in best_hps.values.items():
           # Format floats for readability
           if isinstance(value, float):
               print(f"  - {hp}: {value:.5f}")
           else:
               print(f"  - {hp}: {value}")

       # The best_model is ready for prediction or saving
       # best_model.save(os.path.join(RUN_DIR, "best_transflow_model.keras"))
       # print("\nBest model saved.")
   else:
       print("Search finished, but no best hyperparameters were found.")

**Expected Output:**

.. code-block:: text

   Starting hyperparameter search for TransFlowSubsNet...
   Trial 5 Complete [00h 00m 45s]
   val_loss: 0.168...

   Results summary
   [...]
   --- Hyperparameter Search Complete ---

   Best Hyperparameters Found:
     - embed_dim: 32
     - attention_units: 64
     - num_heads: 4
     - dropout_rate: 0.17581
     - K: learnable
     - Ss: 0.00003
     - pinn_coefficient_C: learnable
     - learning_rate: 0.00100
     - lambda_gw: 1.25678
     - lambda_cons: 0.45901

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Example 2: Tuning PIHALNet
----------------------------
This example showcases the power and flexibility of the ``HydroTuner``. We will
now tune the ``PIHALNet`` model, which has a different set of physical
parameters than ``TransFlowSubsNet``.

The core workflow remains identical. We will reuse the same dataset from
Example 1, but we will define a new ``search_space`` tailored
specifically to the hyperparameters available in ``PIHALNet``.

Step 1: Define a PIHALNet-Specific Search Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The key to adapting the tuner is creating a `search_space` that matches
the target model. For ``PIHALNet``, we are interested in tuning its
unique consolidation coefficient (:math:`C`) and its single physics loss
weight (:math:`\lambda_{physics}`).

Note that we **omit** the hyperparameters for groundwater flow (`K`, `Ss`,
`lambda_gw`), as they are not relevant to the `PIHALNet` model.

.. code-block:: python
   :linenos:

   pihalnet_search_space = {
       # --- Architectural Hyperparameters (can be different) ---
       # For this run, let's fix the embedding dimension and explore others.
       "embed_dim": [32],
       "num_heads": [4, 8],
       "dropout_rate": {"type": "float", "min_value": 0.1, "max_value": 0.5},
       
       # --- Physics Hyperparameters for PIHALNet ---
       # Test a few different fixed values for the consolidation coefficient C
       "pinn_coefficient_C": [1e-3, 5e-3, 1e-2],
       
       # --- Compile-time Hyperparameters ---
       "learning_rate": [1e-3, 5e-4],
       # PIHALNet uses a single physics weight, which we name `lambda_physics`
       # Note: The tuner's `build` method will correctly pass this to the
       # model's `compile` method, which may expect a different name
       # like `lambda_pde`. This mapping is handled internally.
       "lambda_physics": {"type": "float", "min_value": 0.1, "max_value": 1.0}
   }
   print("Defined a new search space tailored for PIHALNet.")

Step 2: Create and Run the Tuner for PIHALNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The process is exactly the same as before. The only changes are in the
arguments passed to ``HydroTuner.create()``: we now specify
``model_name_or_cls=PIHALNet`` and pass our new
``pihalnet_search_space``.

.. code-block:: python
   :linenos:
   
   from fusionlab.nn.pinn.models import PIHALNet
   from tensorflow.keras.callbacks import EarlyStopping

   # 1. Create the tuner instance for PIHALNet
   tuner_pihal = HydroTuner.create(
       model_name_or_cls=PIHALNet, # <-- The primary change
       inputs_data=inputs,
       targets_data=targets,
       search_space=pihalnet_search_space, # <-- Use the new search space
       max_trials=3,
       project_name="PIHALNet_Example_Tuning",
       directory=RUN_DIR,
       overwrite=True
   )

   # 2. Run the search
   print("\nStarting tuning for PIHALNet...")
   best_model_pihal, best_hps_pihal, _ = tuner_pihal.run(
       inputs=inputs,
       y=targets,
       validation_data=(inputs, targets),
       epochs=3,
       batch_size=64,
       callbacks=[EarlyStopping('val_loss', patience=2)]
   )

Step 3: Analyze the PIHALNet Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Finally, we inspect the output to see the best combination of
hyperparameters that the tuner discovered for the ``PIHALNet`` model.

.. code-block:: python
   :linenos:
   
   print("\n--- Best Hyperparameters for PIHALNet ---")
   if best_hps_pihal:
       for hp, value in best_hps_pihal.values.items():
           if isinstance(value, float):
               print(f"  - {hp}: {value:.5f}")
           else:
               print(f"  - {hp}: {value}")
   else:
       print("Search finished, but no best hyperparameters were found.")

**Expected Output:**

.. code-block:: text

   --- Best Hyperparameters for PIHALNet ---
     - embed_dim: 32
     - num_heads: 4
     - dropout_rate: 0.25123
     - pinn_coefficient_C: 0.00500
     - learning_rate: 0.00100
     - lambda_physics: 0.67890

These examples illustrate the power of the ``HydroTuner``'s generic
design. The core workflow remains consistent, and you can easily adapt
it to different models by providing the appropriate model class and
defining a relevant `search_space`. This dramatically accelerates the
process of experimentation and optimization.