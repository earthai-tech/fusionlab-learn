.. _hydro_tuner_guide:

==========================================
Hyperparameter Tuning with HydroTuner
==========================================

:API Reference: :class:`~fusionlab.nn.forecast_tuner.HydroTuner`

Finding the optimal set of hyperparameters for complex models like
``PIHALNet`` and ``TransFlowSubsNet`` can be a challenging and
time-consuming task. The ``HydroTuner`` class is a powerful utility
designed to automate this process, enabling you to efficiently search
for the best model architecture and training configuration for your
specific dataset.

Built on top of the industry-standard `Keras Tuner <https://keras.io/keras_tuner/>`_
library, ``HydroTuner`` provides a streamlined interface tailored for
the unique requirements of hybrid physics-informed models.

Core Concepts
-------------

The design of ``HydroTuner`` is centered on flexibility and a clear
separation of concerns between the problem definition and the tuning
experiment.

**Model-Agnostic Design**
***************************
The tuner is not hardcoded to a single model. By simply passing a
model class (e.g., ``PIHALNet`` or ``TransFlowSubsNet``) to its
constructor, it dynamically adapts its internal `build` process to
construct and tune that specific model. This makes the tuner
extensible to new models in the future.

**Separation of Concerns: `fixed_params` vs. `search_space`**
***************************************************************
Understanding the difference between these two key parameters is
essential for using the tuner effectively:

* **`fixed_params` (The Problem Definition):** This dictionary holds
    all parameters that are **not** tuned and remain constant for an
    entire tuning job. It primarily contains data-dependent dimensions
    that define the model's static structure, such as:
    * `static_input_dim`, `dynamic_input_dim`, `future_input_dim`
    * `output_subsidence_dim`, `output_gwl_dim`
    * `forecast_horizon`
    These values define the specific problem you are trying to solve.

* **`search_space` (The Experiment Definition):** This dictionary
    defines the universe of all hyperparameters you want to optimize.
    This includes everything from network architecture (`embed_dim`,
    `num_heads`, `dropout_rate`) to physics parameters (`K`, `Ss`, `C`)
    and training parameters (`learning_rate`, `lambda_gw`).

This separation allows you to define a single problem (via `fixed_params`)
and run many different tuning experiments on it by simply changing the
`search_space`.

Defining the Search Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``search_space`` dictionary is where you define what to tune. The
tuner supports several flexible formats for defining the range of each
hyperparameter.

* **List for Discrete Choices:**

  Provides a list of explicit values to test.
  
  .. code-block:: python

     {'num_heads': [2, 4, 8], 'activation': ['relu', 'gelu']}

* **Dictionary for Ranges:**

  For integers or floats, you can specify a range. The dictionary
  must include a `type` key.
  
  .. code-block:: python

     # Integer range with a step
     {'lstm_units': {'type': 'int', 'min_value': 32, 'max_value': 128, 'step': 32}}

     # Float range with logarithmic sampling
     {'learning_rate': {'type': 'float', 'min_value': 1e-4, 'max_value': 1e-2, 'sampling': 'log'}}

* **Dictionary for Booleans:**

  To tune whether a feature should be enabled or disabled.
  .. code-block:: python

     {'use_residuals': {'type': 'bool'}}

This flexible format gives you complete control over the scope of the
hyperparameter search.

End-to-End Workflow
-------------------
The recommended workflow involves using the ``HydroTuner.create()``
factory method, which simplifies the setup process significantly.

**Step 1: Prepare Data**
************************
First, load your input features and target variables as dictionaries
of NumPy arrays.

.. code-block:: python
   :linenos:
   
   import numpy as np
   
   # Example dummy data
   B, T, H = 128, 15, 7
   S_DIM, D_DIM, F_DIM = 3, 5, 2
   
   inputs = {
       "coords": np.random.rand(B, H, 3),
       "static_features": np.random.rand(B, S_DIM),
       "dynamic_features": np.random.rand(B, T, D_DIM),
       "future_features": np.random.rand(B, H, F_DIM),
   }
   targets = {
       "subsidence": np.random.rand(B, H, 1),
       "gwl": np.random.rand(B, H, 1)
   }

**Step 2: Define the Search Space**
***********************************
Create the dictionary that defines your tuning experiment. Here, we'll
define a space for tuning a ``TransFlowSubsNet`` model, including its
physics parameters.

.. code-block:: python
   :linenos:

   search_space = {
       # Architectural HPs
       "embed_dim": [32, 64],
       "dropout_rate": {"type": "float", "min_value": 0.0, "max_value": 0.4},
       
       # Physics HPs for TransFlowSubsNet
       "K": ["learnable", 1e-5, 1e-4], # Tune between learnable or fixed values
       "Ss": {"type": "float", "min_value": 1e-6, "max_value": 1e-4, "sampling": "log"},
       
       # Compile-time HPs
       "learning_rate": [1e-3, 5e-4, 1e-4],
       "lambda_gw": {"type": "float", "min_value": 0.5, "max_value": 1.5},
       "lambda_cons": {"type": "float", "min_value": 0.1, "max_value": 1.0}
   }

**Step 3: Create the Tuner with `.create()`**
*********************************************
Use the factory method to instantiate the tuner. It will automatically
infer the data dimensions from your NumPy arrays. You can still provide
manual ``fixed_params`` to override any defaults or inferred values.

.. code-block:: python
   :linenos:

   from fusionlab.nn.forecast_tuner import HydroTuner
   from fusionlab.nn.pinn import TransFlowSubsNet

   tuner = HydroTuner.create(
       model_name_or_cls=TransFlowSubsNet,
       inputs_data=inputs,
       targets_data=targets,
       search_space=search_space,
       fixed_params={"quantiles": None}, # Manually specify no quantiles
       
       # Keras Tuner settings
       objective="val_loss",
       max_trials=25,
       project_name="TransFlowSubsNet_Optimization"
   )

   print(f"Tuner created for model: {tuner.model_class.__name__}")
   print(f"Inferred forecast horizon: {tuner.fixed_params['forecast_horizon']}")

**Step 4: Run the Search**
**************************
Call the ``.run()`` method to start the hyperparameter search. You will
typically want to include an ``EarlyStopping`` callback.

.. code-block:: python
   :linenos:

   from tensorflow.keras.callbacks import EarlyStopping

   # Note: The 'run' method is an alias for the base 'fit' method.
   best_model, best_hps, tuner_instance = tuner.run(
       inputs=inputs,
       y=targets,
       validation_data=(inputs, targets), # Use same data for example
       epochs=50,
       batch_size=32,
       callbacks=[EarlyStopping(patience=5, monitor='val_loss')]
   )

**Step 5: Retrieve and Use the Best Model**
*******************************************
After the search is complete, the tuner object holds the best
hyperparameters and a retrained model ready for use.

.. code-block:: python

   print("\n--- Best Hyperparameters Found ---")
   for hp, value in best_hps.values.items():
       print(f"{hp}: {value}")

   # The best_model is already retrained and ready for prediction or saving.
   # best_model.save("my_best_hydro_model.keras")

API Reference
-------------
.. autoclass:: fusionlab.nn.forecast_tuner.HydroTuner
   :members: create, run, build
   :undoc-members: