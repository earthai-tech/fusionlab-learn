.. _pihal_tuner_guide:

================================
PIHALTuner for PIHALNet Tuning
================================

:API Reference: :class:`~fusionlab.nn.pinn.tuning.PIHALTuner`

The ``PIHALTuner`` class is a specialized hyperparameter tuning utility
designed specifically for the :class:`~fusionlab.nn.pinn.models.PIHALNet` model.
It inherits from :class:`~fusionlab.nn.pinn.tuning.PINNTunerBase` and
leverages the `Keras Tuner <https://keras.io/keras_tuner/>`_ library to
efficiently search for optimal hyperparameter configurations for your
physics-informed land subsidence forecasting tasks.

Key Responsibilities
--------------------

* **PIHALNet-Specific Configuration**: Manages both fixed parameters
    (like input/output dimensions, forecast horizon) and the hyperparameter
    search space relevant to ``PIHALNet``'s architecture and PINN components.
* **Parameter Inference**: Includes logic (via the static method
    ``_infer_dims_and_prepare_fixed_params``) to infer necessary fixed parameters
    (e.g., input feature dimensions) directly from the provided training data if
    they are not explicitly set during instantiation.
* **Model Building**: Implements the ``build`` method required by Keras Tuner.
    This method constructs and compiles a ``PIHALNet`` instance for each
    tuning trial, using a combination of the fixed parameters and the
    hyperparameters sampled by the tuner for that trial.
* **Orchestration of Tuning Process**: The ``run`` method provides a high-level
    interface to start the hyperparameter search. It handles data preparation
    (converting NumPy arrays to ``tf.data.Dataset`` objects), ensures fixed
    parameters are correctly set up, and then invokes the underlying Keras Tuner's
    search mechanism (from ``PINNTunerBase``).

Core Components and Workflow
-------------------------------

1.  **Initialization (`__init__`)**:
    When you create a ``PIHALTuner`` instance, you typically provide:
    * ``fixed_model_params``: A dictionary containing parameters for
        :class:`~fusionlab.nn.pinn.models.PIHALNet` that will remain constant
        throughout the tuning process. This **must** include data-dependent
        dimensions like ``static_input_dim``, ``dynamic_input_dim``,
        ``future_input_dim``, ``output_subsidence_dim``, ``output_gwl_dim``,
        and ``forecast_horizon``. Other ``PIHALNet`` parameters (e.g.,
        ``quantiles``, ``pde_mode``, ``max_window_size``) can also be set here
        if they are not part of the hyperparameter search defined in ``param_space``.
    * ``param_space`` (optional): A dictionary defining the search space for
        hyperparameters you wish to tune. The keys are hyperparameter names
        (e.g., ``'embed_dim'``, ``'learning_rate'``), and the values are
        configurations for Keras Tuner's sampling methods (e.g.,
        ``{'min_value': 32, 'max_value': 128, 'step': 32}`` for an integer,
        or a list of choices like ``['relu', 'gelu']``). If ``None``, the
        ``PIHALTuner.build()`` method uses a predefined default search space for
        common ``PIHALNet`` hyperparameters.
    * Standard Keras Tuner configurations: ``objective``, ``max_trials``,
        ``project_name``, ``directory``, ``tuner_type`` (e.g., 'randomsearch',
        'hyperband', 'bayesianoptimization'), ``seed``, ``overwrite_tuner``, etc.

2.  **Parameter Inference (`_infer_dims_and_prepare_fixed_params` static method)**:
    This internal static method is key to ``PIHALTuner``'s flexibility. It
    determines the final set of ``fixed_model_params`` by following a priority:
    user-provided parameters first, then values inferred from data (if ``inputs_data``
    and ``targets_data`` are given), and finally, defaults from
    ``DEFAULT_PIHALNET_FIXED_PARAMS``. This method is called by ``PIHALTuner.create()``
    and also by ``PIHALTuner.run()`` if fixed parameters were not fully specified
    at initialization.

3.  **Factory Method (`create` class method)**:
    The ``PIHALTuner.create(...)`` class method offers a convenient way to
    instantiate the tuner, particularly when you want to infer dimensions and
    other fixed parameters directly from your data arrays.

    .. code-block:: python
       :linenos:
       
       from fusionlab.nn.pinn.tuning import PIHALTuner
       import numpy as np # For dummy data
       import tensorflow as tf # For EarlyStopping

       # Example: Dummy input and target data (dictionaries of NumPy arrays)
       inputs_np = {
           "coords": np.random.rand(100, 3, 3).astype(np.float32), # B, H, (t,x,y)
           "static_features": np.random.rand(100, 5).astype(np.float32), # B, Ds
           "dynamic_features": np.random.rand(100, 7, 4).astype(np.float32), # B, T_past, Dd
           "future_features": np.random.rand(100, 3, 1).astype(np.float32)  # B, H, Df
       }
       targets_np = {
           "subsidence": np.random.rand(100, 3, 1).astype(np.float32), # B, H, Os
           "gwl": np.random.rand(100, 3, 1).astype(np.float32)         # B, H, Og
       }

       tuner = PIHALTuner.create(
           inputs_data=inputs_np,
           targets_data=targets_np,
           # forecast_horizon & quantiles will be inferred or taken from fixed_model_params/defaults
           # fixed_model_params can be provided here to override defaults/inference for specific items
           max_trials=5, # Keep low for example
           project_name="MyPIHALNetTuning_Create",
           tuner_type="randomsearch",
           verbose=1
       )

4.  **Building the Model (`build` method)**:
    This method is automatically called by the Keras Tuner backend for each trial.
    Based on the ``PIHALTuner`` class definition:
    * It takes a ``hp: kt.HyperParameters`` object.
    * It uses helper methods like ``_get_hp_int()``, ``_get_hp_float()``,
        ``_get_hp_choice()`` to sample values for tunable ``PIHALNet``
        hyperparameters. These helpers consult the ``self.param_space``
        dictionary if provided, otherwise they use pre-defined default ranges
        and choices (as seen in the `build` method's docstring).
    * Tunable parameters include: ``embed_dim``, ``hidden_units``,
        ``lstm_units``, ``attention_units``, ``num_heads``, ``dropout_rate``,
        ``activation``, ``use_vsn``, ``vsn_units``, ``pde_mode``,
        ``pinn_coefficient_C_type`` (and its value if fixed), ``lambda_pde`` (PDE loss weight),
        and ``learning_rate``.
    * These sampled hyperparameters are merged with ``self.fixed_model_params``.
    * A :class:`~fusionlab.nn.pinn.models.PIHALNet` instance is created.
    * The model is compiled with an Adam optimizer (with ``clipnorm=1.0``),
        Mean Squared Error losses for 'subs_pred' and 'gwl_pred' (or
        :func:`~fusionlab.nn.losses.combined_quantile_loss` if quantiles
        are used), and MAE metrics.

5.  **Running the Hyperparameter Search (`run` method)**:
    The ``PIHALTuner.run(...)`` method is the primary way to start the tuning process.
    * It accepts ``inputs`` (dictionary of NumPy input arrays) and ``y``
        (dictionary of NumPy target arrays).
    * It handles the necessary setup for ``self.fixed_model_params`` by calling
        ``_infer_dims_and_prepare_fixed_params`` if these weren't fully set
        during ``__init__``.
    * It renames target keys in `y` (and `validation_data`) from common names
        like "subsidence" to model output names like "subs_pred".
    * It converts the NumPy input/target dictionaries into ``tf.data.Dataset``
        objects, including batching and prefetching.
    * Finally, it calls ``super().search(...)`` (from ``PINNTunerBase``),
        which manages the instantiation of the Keras Tuner backend (e.g.,
        `kt.RandomSearch`, `kt.Hyperband`) and runs the search loop, fitting
        models for each trial.

    .. code-block:: python
       :linenos:

       # Continuing from the tuner instantiation example:
       # Assume inputs_train_np, targets_train_np, inputs_val_np, targets_val_np are prepared
       
       # Define early stopping callback for the search
       early_stopping = tf.keras.callbacks.EarlyStopping(
           monitor='val_total_loss', # Or your chosen objective
           patience=5,
           restore_best_weights=True
       )

       # Start the search using PIHALTuner's run method
       best_model, best_hps, tuner_instance = tuner.run(
           inputs=inputs_train_np,
           y=targets_train_np, # e.g., {'subsidence': s_train, 'gwl': h_train}
           validation_data=(inputs_val_np, targets_val_np),
           epochs=20, # Max epochs for each trial
           batch_size=32,
           callbacks=[early_stopping]
       )

       if best_hps:
           print("Best Hyperparameters found:")
           for hp_name, hp_value in best_hps.values.items():
               print(f"  {hp_name}: {hp_value}")
       if best_model:
           best_model.summary()

Default Fixed Parameters and Case Information
---------------------------------------------
``PIHALTuner`` relies on ``DEFAULT_PIHALNET_FIXED_PARAMS`` for fallback values
if certain fixed parameters are not provided or inferred. This dictionary includes
sensible defaults for aspects like output dimensions, horizon, aggregation methods,
and physics-related settings.

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
       "loss_weights": { # Used during model.compile() in PIHALTuner.build
           'subs_pred': 1.0,
           'gwl_pred': 0.8
       }
   }

Note that critical data-dependent dimensions (``static_input_dim``,
``dynamic_input_dim``, ``future_input_dim``) are **not** in these defaults and
**must** be either provided directly in `fixed_model_params` at tuner
instantiation or inferred from data via `PIHALTuner.create()` or during the
`PIHALTuner.run()` call.

Customizing the Hyperparameter Search Space
-------------------------------------------
While ``PIHALTuner`` provides a comprehensive default search space within its
``build`` method, you can supply a custom ``param_space`` dictionary during
tuner instantiation to precisely control which hyperparameters are tuned and
over what ranges or choices.

Example of a custom ``param_space`` dictionary:

.. code-block:: python
   :linenos:

   custom_param_space = {
       'embed_dim': {'min_value': 64, 'max_value': 128, 'step': 32},
       'lstm_units': [128, 256], # hp.Choice
       'learning_rate': {'min_value': 1e-5, 'max_value': 1e-3, 'sampling': 'log'},
       'lambda_pde': {'min_value': 0.1, 'max_value': 1.0, 'step': 0.1},
       'pde_mode': ['consolidation'] # Fixed choice for this tuning run
   }

   # Pass to PIHALTuner constructor or create method
   tuner = PIHALTuner(
       fixed_model_params=my_data_derived_fixed_params,
       param_space=custom_param_space,
       # ... other tuner settings ...
   )

This allows for focused tuning on specific aspects of the ``PIHALNet`` model.

For a practical example of using ``PIHALTuner``, including data preparation,
see the :ref:`tuning_pihalnet_example` page.
