.. _pihal_tuner_guide:

================================
PIHALTuner for PIHALNet Tuning
================================

:API Reference: :class:`~fusionlab.nn.pinn.tuning.PIHALTuner`

The ``PIHALTuner`` class is a specialized hyperparameter tuning
utility designed specifically for the
:class:`~fusionlab.nn.pinn.models.PIHALNet` model. It inherits from
:class:`~fusionlab.nn.pinn.tuning.PINNTunerBase` and leverages the
`Keras Tuner <https://keras.io/keras_tuner/>`_ library to
efficiently search for optimal hyperparameter configurations for your
physics-informed land subsidence forecasting tasks.

Key Responsibilities
----------------------

* **PIHALNet-Specific Configuration**: Manages both fixed parameters
    (like input/output dimensions, forecast horizon) and the
    hyperparameter search space relevant to ``PIHALNet``'s architecture
    and PINN components.
* **Parameter Inference**: Includes logic (via the static method
    ``_infer_dims_and_prepare_fixed_params``) to infer necessary
    fixed parameters (e.g., input feature dimensions) directly
    from the provided training data if they are not explicitly set
    during instantiation.
* **Model Building**: Implements the ``build`` method required by
    Keras Tuner. This method constructs and compiles a ``PIHALNet``
    instance for each tuning trial, using a combination of the fixed
    parameters and the hyperparameters sampled by the tuner for that
    trial.
* **Orchestration of Tuning Process**: The ``run`` method provides
    a high-level interface to start the hyperparameter search. It
    handles data preparation (converting NumPy arrays to
    ``tf.data.Dataset`` objects), ensures fixed parameters are
    correctly set up, and then invokes the underlying Keras Tuner's
    search mechanism (from ``PINNTunerBase``).

Core Components and Workflow
------------------------------

1.  **Initialization (`__init__`)**:
    When you create a ``PIHALTuner`` instance, you typically provide:

    * ``fixed_model_params``: A dictionary containing parameters for
        :class:`~fusionlab.nn.pinn.models.PIHALNet` that will remain
        constant throughout the tuning process. This **must** include
        data-dependent dimensions like ``static_input_dim``,
        ``dynamic_input_dim``, ``future_input_dim``,
        ``output_subsidence_dim``, ``output_gwl_dim``, and
        ``forecast_horizon``.
    * ``param_space`` (optional): A dictionary defining the search
        space for hyperparameters you wish to tune. If ``None``, the
        ``PIHALTuner.build()`` method uses a predefined default search
        space.
    * Standard Keras Tuner configurations: ``objective``,
        ``max_trials``, ``project_name``, ``directory``, ``tuner_type``
        (e.g., 'randomsearch', 'hyperband'), ``seed``, etc.

2.  **Factory Method (`create` class method)**:
    The ``PIHALTuner.create(...)`` class method offers a convenient
    way to instantiate the tuner by inferring dimensions and other fixed
    parameters directly from your data arrays.

    .. code-block:: python
       :linenos:

       from fusionlab.nn.pinn.tuning import PIHALTuner
       import numpy as np

       # Example: Dummy input and target data
       inputs_np = {
           "coords": np.random.rand(100, 3, 3),
           "static_features": np.random.rand(100, 5),
           "dynamic_features": np.random.rand(100, 7, 4),
           "future_features": np.random.rand(100, 3, 1)
       }
       targets_np = {
           "subsidence": np.random.rand(100, 3, 1),
           "gwl": np.random.rand(100, 3, 1)
       }

       tuner = PIHALTuner.create(
           inputs_data=inputs_np,
           targets_data=targets_np,
           forecast_horizon=3,
           max_trials=5,
           project_name="MyPIHALNetTuning_Create",
           tuner_type="randomsearch"
       )

3.  **Building the Model (`build` method)**:
    This method is automatically called by the Keras Tuner backend
    for each trial. It samples hyperparameters from the defined
    search space and uses them, along with the fixed parameters, to
    construct and compile a :class:`~fusionlab.nn.pinn.models.PIHALNet`
    instance. Tunable parameters include ``embed_dim``, ``lstm_units``,
    ``learning_rate``, ``lambda_pde``, and many others.

4.  **Running the Hyperparameter Search (`run` method)**:
    The ``PIHALTuner.run(...)`` method is the primary way to start the
    tuning process. It handles data preparation and invokes the core
    Keras Tuner search.

    .. code-block:: python
       :linenos:

       import tensorflow as tf
       
       # Continuing from the tuner instantiation example
       # Assume inputs_train_np, targets_train_np, etc. are prepared

       early_stopping = tf.keras.callbacks.EarlyStopping(
           monitor='val_total_loss',
           patience=5
       )

       best_model, best_hps, tuner_instance = tuner.run(
           inputs=inputs_train_np,
           y=targets_train_np,
           validation_data=(inputs_val_np, targets_val_np),
           epochs=20,
           batch_size=32,
           callbacks=[early_stopping]
       )

       if best_hps:
           print("Best Hyperparameters found:")
           print(best_hps.values)

Default Fixed Parameters
------------------------
``PIHALTuner`` uses ``DEFAULT_PIHALNET_FIXED_PARAMS`` for fallback
values if certain fixed parameters are not provided or inferred. This
dictionary includes sensible defaults for aspects like output
dimensions, aggregation methods, and physics-related settings.

.. code-block:: python
   :emphasize-lines: 1-23

   DEFAULT_PIHALNET_FIXED_PARAMS = {
       "output_subsidence_dim": 1,
       "output_gwl_dim": 1,
       "forecast_horizon": 1,
       "quantiles": None,
       "max_window_size": 10,
       "memory_size": 100,
       "scales": [1],
       "multi_scale_agg": 'last',
       "final_agg": 'last',
       "use_residuals": True,
       "use_batch_norm": False,
       "use_vsn": True,
       "vsn_units": 32,
       "activation": "relu",
       "pde_mode": "consolidation",
       "pinn_coefficient_C": "learnable",
       "gw_flow_coeffs": None,
       "loss_weights": {
           'subs_pred': 1.0,
           'gwl_pred': 0.8
       }
   }

Customizing the Hyperparameter Search Space
-------------------------------------------
You can supply a custom ``param_space`` dictionary during tuner
instantiation to precisely control which hyperparameters are tuned
and over what ranges or choices.

.. code-block:: python

   custom_param_space = {
       'embed_dim': {'min_value': 64, 'max_value': 128, 'step': 32},
       'lstm_units': [128, 256], # hp.Choice
       'learning_rate': {'min_value': 1e-5, 'max_value': 1e-3, 'sampling': 'log'},
       'lambda_pde': {'min_value': 0.1, 'max_value': 1.0},
       'pde_mode': ['consolidation'] # Fixed choice for this tuning run
   }

   tuner = PIHALTuner(
       fixed_model_params=my_data_derived_fixed_params,
       param_space=custom_param_space,
       # ... other tuner settings ...
   )

This allows for focused tuning on specific aspects of the ``PIHALNet`` model.
For a practical example of using ``PIHALTuner``, see the
:ref:`tuning_pihalnet_example` page.
