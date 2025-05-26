.. _user_guide_forecast_tuner:

=======================
Forecast Tuner Guide
=======================

Finding the optimal set of hyperparameters for deep learning models
like :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`,
:class:`~fusionlab.nn.transformers.TFT` (stricter version),
:class:`~fusionlab.nn.XTFT`, and
:class:`~fusionlab.nn.SuperXTFT` is crucial for achieving the best
possible forecasting performance. Hyperparameters control aspects of
the model architecture (e.g., number of hidden units, attention
heads) and the training process (e.g., learning rate, batch size).

``fusionlab`` provides utility functions and classes within the
:mod:`~fusionlab.nn.forecast_tuner` module that leverage the
powerful **Keras Tuner** library (`keras-tuner`) to automate this
search process.

Prerequisites
-------------

To use the tuning functions, you must have Keras Tuner installed:

.. code-block:: bash

    pip install keras-tuner -q

.. _tuning_approach_choice:

Choosing Your Tuning Approach: Functions vs. Classes
----------------------------------------------------

As of version 0.2.3, ``fusionlab-learn`` offers two primary ways to
perform hyperparameter tuning:

1.  **Class-Based Tuners (Recommended for New Projects):**
    This modern, object-oriented approach utilizes dedicated tuner
    classes like :class:`~fusionlab.nn.forecast_tuner.XTFTTuner`
    and :class:`~fusionlab.nn.forecast_tuner.TFTTuner`.
    It offers improved structure, reusability (configure once,
    `fit` multiple times), and flexibility.
    For detailed information and examples, please refer to the
    :doc:`Class-Based Forecast Tuner Guide <./forecast_tuner_class_based>`.

2.  **Function-Based Tuners (Legacy, Still Supported):**
    The original approach uses standalone functions like
    :func:`~fusionlab.nn.forecast_tuner.xtft_tuner` and
    :func:`~fusionlab.nn.forecast_tuner.tft_tuner`. These
    functions remain fully supported for backward compatibility and
    are detailed in the sections below.

For new development, we recommend exploring the class-based tuners
for a more streamlined and maintainable workflow.

Tuner Functions (Legacy Approach)
---------------------------------

The :mod:`fusionlab.nn.forecast_tuner` module offers dedicated
functions to tune different model types. These are suitable for
quick experiments or maintaining existing codebases.

.. _xtft_tuner_doc:

xtft_tuner
~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.forecast_tuner.xtft_tuner`

**Purpose:**
To perform hyperparameter optimization for the
:class:`~fusionlab.nn.XTFT` and
:class:`~fusionlab.nn.SuperXTFT` models. It can also be used
to tune :class:`~fusionlab.nn.transformers.TFT` (stricter) and
:class:`~fusionlab.nn.TemporalFusionTransformer` (flexible) by
specifying the appropriate `model_name`.

**Functionality:**
This function orchestrates the tuning process:

1.  **Inputs:** Takes prepared input data as a list
    `inputs = [X_static, X_dynamic, X_future]` (where static or
    future can be `None` if `model_name='tft_flex'`) and the
    target array `y`.
2.  **Search Space:** Uses a default space (`DEFAULT_PS`) for
    common hyperparameters. Users can provide their own
    `param_space` dictionary to override or extend these.
3.  **Model Builder:** Employs an internal default
    :func:`~fusionlab.nn.forecast_tuner._model_builder_factory`
    (or a user-provided `model_builder`) to construct model
    instances for given hyperparameters (`hp`). The builder samples
    values using Keras Tuner's `hp` object. Models are compiled
    with Adam optimizer and an appropriate loss (MSE or quantile).
4.  **Tuner Initialization:** Creates a Keras Tuner instance
    (`RandomSearch` or `BayesianOptimization`) configured with the
    `objective`, `max_trials`, `tuner_dir`, and `project_name`.
5.  **Search Execution:** Iterates through `batch_sizes`. For each:
    * Runs `tuner.search()` using the data, `epochs` (for trials),
      `validation_split`, and `callbacks`.
    * Retrieves the best hyperparameters for that batch size.
    * Builds and fully trains a model using these HPs and batch
      size for the user-specified `epochs`.
6.  **Best Model Selection:** Compares validation loss across all
    tested `batch_sizes` to find the overall `best_hps`,
    `best_model`, and `best_batch_size`.
7.  **Output:** Returns `(best_hps, best_model, tuner_object)`.
    Results are logged to a JSON file.

**Usage Context:**
Use after preparing training data into the required list format.
Provide data, `forecast_horizon`, `quantiles` (if any), and
optionally customize `param_space`, `max_trials`, `epochs`, etc.
Crucially, set `model_name` to `"xtft"`, `"superxtft"`, `"tft"`,
or `"tft_flex"` to guide the internal model builder.

**Code Example (Tuning XTFT):**

.. code-block:: python
    :linenos:

    import numpy as np
    import os
    import tensorflow as tf
    from fusionlab.nn.forecast_tuner import xtft_tuner
    # from fusionlab.nn import XTFT # For context

    # 1. Prepare Dummy Data (Static, Dynamic, Future)
    B, T_past, H_out = 8, 12, 6
    D_s, D_d, D_f = 3, 5, 2
    T_future_total = T_past + H_out

    X_static_train = np.random.rand(B, D_s).astype(np.float32)
    X_dynamic_train = np.random.rand(B, T_past, D_d).astype(np.float32)
    X_future_train = np.random.rand(
        B, T_future_total, D_f).astype(np.float32)
    y_train = np.random.rand(B, H_out, 1).astype(np.float32)

    # Inputs for tuner: [Static, Dynamic, Future]
    train_inputs = [X_static_train, X_dynamic_train, X_future_train]

    # 2. Define Minimal Search Space & Case Info
    custom_param_space = {
        'hidden_units': [16], # Fixed for speed
        'num_heads': [2],
        'learning_rate': [1e-3]
    }
    case_info_xtft = {
        'quantiles': None, # Point forecast
        'forecast_horizon': H_out,
        'static_input_dim': D_s,
        'dynamic_input_dim': D_d,
        'future_input_dim': D_f,
        'output_dim': 1
    }

    # 3. Define Tuning Parameters
    output_dir = "./xtft_tuning_example_output"
    project_name = "XTFT_Point_Tuning"

    # 4. Run the Tuner for XTFT
    print("Starting XTFT tuning...")
    best_hps, best_model, tuner = xtft_tuner(
        inputs=train_inputs,
        y=y_train,
        param_space=custom_param_space,
        forecast_horizon=H_out, # Passed directly to tuner
        quantiles=None,         # Passed directly to tuner
        case_info=case_info_xtft, # For model builder
        max_trials=1,           # Minimal for demo
        objective='val_loss',
        epochs=2,               # Minimal for demo
        batch_sizes=[8],        # Single small batch
        validation_split=0.25,
        tuner_dir=output_dir,
        project_name=project_name,
        tuner_type='random',
        model_name="xtft", # Crucial: tells builder to make XTFT
        verbose=0
    )

    # 5. Display Results
    print("\nXTFT Tuning complete.")
    if best_hps:
        print("--- Best Hyperparameters (XTFT) ---")
        print(best_hps)
        # best_model.summary()
    else:
        print("XTFT Tuning failed to find a best model.")
    # tuner.results_summary(num_trials=1)


.. raw:: html

    <hr>

.. _tft_tuner_doc:

tft_tuner
~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.forecast_tuner.tft_tuner`

**Purpose:**
A convenience wrapper for tuning Temporal Fusion Transformer models.
It calls :func:`xtft_tuner` internally, passing the `model_name`
parameter to differentiate between the stricter
:class:`~fusionlab.nn.transformers.TFT` (which requires all static,
dynamic, and future inputs) and the more flexible
:class:`~fusionlab.nn.TemporalFusionTransformer` (which can handle
optional static and/or future inputs).

**Functionality:**
Accepts the same parameters as :func:`xtft_tuner`. The key is the
`model_name` argument:
* Set `model_name="tft"` to tune the stricter `TFT` class.
    In this case, `inputs` must be a list of three non-None tensors
    `[X_static, X_dynamic, X_future]`.
* Set `model_name="tft_flex"` to tune the flexible
    `TemporalFusionTransformer`. In this case, `inputs` can be
    `[X_static, X_dynamic, X_future]` where `X_static` and/or
    `X_future` can be `None` (or even a single tensor for dynamic-only).

The internal default model builder
(:func:`~fusionlab.nn.forecast_tuner._model_builder_factory`)
constructs the appropriate TFT variant and uses relevant
hyperparameters.

**Usage Context:**
Use this when your primary goal is to tune a TFT model. Choose
`model_name="tft"` for the standard three-input architecture or
`model_name="tft_flex"` if you are working with scenarios that
might not include all input types.

**Code Example 1 (Tuning Stricter `TFT`):**

.. code-block:: python
    :linenos:

    import numpy as np
    import os
    import tensorflow as tf
    from fusionlab.nn.forecast_tuner import tft_tuner
    # from fusionlab.nn.transformers import TFT # For context

    # 1. Prepare Dummy Data (ALL inputs required for stricter TFT)
    B, T_past, H_out = 8, 12, 6
    D_s, D_d, D_f = 3, 5, 2
    T_future_total = T_past + H_out

    X_s_train = np.random.rand(B, D_s).astype(np.float32)
    X_d_train = np.random.rand(B, T_past, D_d).astype(np.float32)
    X_f_train = np.random.rand(
        B, T_future_total, D_f).astype(np.float32)
    y_train_tft = np.random.rand(B, H_out, 1).astype(np.float32)

    train_inputs_strict_tft = [X_s_train, X_d_train, X_f_train]

    # 2. Define Case Info & Minimal Param Space
    case_info_strict_tft = {
        'quantiles': None, 'forecast_horizon': H_out,
        'static_input_dim': D_s, 'dynamic_input_dim': D_d,
        'future_input_dim': D_f, 'output_dim': 1
    }
    param_space_tft = {'hidden_units': [16], 'learning_rate': [1e-3]}

    # 3. Run Tuner for Stricter TFT
    print("\nStarting stricter TFT tuning...")
    best_hps_s, _, _ = tft_tuner(
        inputs=train_inputs_strict_tft, y=y_train_tft,
        param_space=param_space_tft,
        forecast_horizon=H_out, quantiles=None,
        case_info=case_info_strict_tft,
        max_trials=1, epochs=1, batch_sizes=[4],
        validation_split=0.5, tuner_dir="./tft_strict_tuning",
        project_name="TFT_Strict_Tune", model_name="tft", # Key
        verbose=0
    )
    print("Stricter TFT Tuning complete.")
    if best_hps_s: print("  Best HPs (Stricter TFT):", best_hps_s)

**Code Example 2 (Tuning Flexible `TemporalFusionTransformer`):**

This example tunes the flexible TFT, providing only dynamic inputs.

.. code-block:: python
    :linenos:

    import numpy as np
    import os
    import tensorflow as tf
    from fusionlab.nn.forecast_tuner import tft_tuner
    # from fusionlab.nn import TemporalFusionTransformer # For context

    # 1. Prepare Dummy Data (Dynamic inputs only)
    B, T_past, H_out = 8, 12, 6
    D_d = 5 # Dynamic features
    X_d_train_flex = np.random.rand(B, T_past, D_d).astype(np.float32)
    y_train_flex = np.random.rand(B, H_out, 1).astype(np.float32)

    # Inputs for flexible TFT (static and future are None)
    train_inputs_flex = [None, X_d_train_flex, None]

    # 2. Define Case Info & Minimal Param Space
    case_info_flex_tft = {
        'quantiles': None, 'forecast_horizon': H_out,
        'dynamic_input_dim': D_d, # Static/Future dims are None
        'static_input_dim': None,
        'future_input_dim': None,
        'output_dim': 1
    }
    param_space_flex = {'hidden_units': [16], 'learning_rate': [1e-3]}

    # 3. Run Tuner for Flexible TFT
    print("\nStarting flexible TFT (tft_flex) tuning...")
    best_hps_f, _, _ = tft_tuner(
        inputs=train_inputs_flex, y=y_train_flex,
        param_space=param_space_flex,
        forecast_horizon=H_out, quantiles=None,
        case_info=case_info_flex_tft,
        max_trials=1, epochs=1, batch_sizes=[4],
        validation_split=0.5, tuner_dir="./tft_flex_tuning",
        project_name="TFT_Flex_Tune", model_name="tft_flex", # Key
        verbose=0
    )
    print("Flexible TFT Tuning complete.")
    if best_hps_f: print("  Best HPs (Flexible TFT):", best_hps_f)


.. raw:: html

    <hr>

Internal Model Builder
-------------------------
:API Reference: :func:`~fusionlab.nn.forecast_tuner._model_builder_factory` (Note: private function)


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

