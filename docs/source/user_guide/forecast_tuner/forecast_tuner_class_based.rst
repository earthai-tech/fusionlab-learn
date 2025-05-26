.. _user_guide_forecast_tuner_class:

====================================
Class-Based Tuner Guide 
====================================

Optimizing hyperparameters is essential for maximizing the performance
of advanced forecasting models like
:class:`~fusionlab.nn.transformers.TemporalFusionTransformer`,
:class:`~fusionlab.nn.transformers.TFT`,
:class:`~fusionlab.nn.XTFT`, and
:class:`~fusionlab.nn.SuperXTFT`. These parameters define the model's
architecture and its training dynamics.

``fusionlab-learn`` introduces an object-oriented approach for hyperparameter
tuning, building upon the **Keras Tuner** library. This approach
utilizes dedicated tuner classes, `XTFTTuner` and `TFTTuner`, which
inherit common functionalities from a `_BaseTuner` class (Note: `_BaseTuner`
is internal; users interact with `XTFTTuner` and `TFTTuner`).

This class-based approach provides a more structured and reusable way
to set up and execute tuning experiments compared to the previous
function-based method (which remains available for backward compatibility).

Prerequisites
-------------

Ensure you have Keras Tuner installed:

.. code-block:: bash

    pip install keras-tuner -q

The Class-Based Approach
------------------------

The core idea is to instantiate a tuner class (`XTFTTuner` or `TFTTuner`)
with parameters defining the *tuning process* itself (like `max_trials`,
`objective`, `param_space`), and then call its `fit` method with the
*data and task-specific parameters* (like `inputs`, `y`, `forecast_horizon`).

**Key Advantages:**

* **Reusability:** Configure a tuner once and use its `fit` method
    multiple times with different datasets or forecast horizons.
* **Structure:** Encapsulates tuning logic within objects, leading to
    cleaner code.
* **Flexibility:** Easily extend or customize by inheriting or providing
    custom model builders.

.. _xtft_tuner_class_doc:

XTFTTuner
~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.forecast_tuner.XTFTTuner`

**Purpose:**
Designed specifically to perform hyperparameter optimization for
:class:`~fusionlab.nn.XTFT` and :class:`~fusionlab.nn.SuperXTFT`
models. It ensures that the `model_name` passed during initialization
is one of these supported variants ("xtft", "superxtft", or "super_xtft").

**Functionality:**

1.  **Initialization (`__init__`)**: You create an instance by providing
    tuning-process parameters (e.g., `max_trials`, `epochs`, `batch_sizes`,
    `param_space`). You must specify a supported `model_name` (defaults
    to "xtft").
2.  **Fitting (`fit`)**: You call the ``fit`` method with your input tensors
    (``inputs = [X_static, X_dynamic, X_future]``), target `y`, and crucial
    task parameters like `forecast_horizon` and `quantiles`.
3.  **Tuning**: The `fit` method executes the Keras Tuner search loop,
    building models using the internal factory or a custom builder,
    evaluating them, and finding the best set of hyperparameters across
    the specified `batch_sizes`.
4.  **Results**: After fitting, the tuner instance holds the results in
    its attributes: `best_hps_` (dictionary), `best_model_` (Keras model),
    and `tuner_` (Keras Tuner object). Results are also saved to a JSON file.

**Code Example (Tuning XTFT):**

.. code-block:: python
    :linenos:

    import numpy as np
    import os
    from fusionlab.nn.forecast_tuner import XTFTTuner

    # 1. Prepare Dummy Data (Static, Dynamic, Future)
    B, T_past, H_out = 8, 12, 6
    D_s, D_d, D_f = 3, 5, 2
    T_future_total = T_past + H_out

    X_static_train = np.random.rand(B, D_s).astype(np.float32)
    X_dynamic_train = np.random.rand(B, T_past, D_d).astype(np.float32)
    X_future_train = np.random.rand(
        B, T_future_total, D_f).astype(np.float32)
    y_train = np.random.rand(B, H_out, 1).astype(np.float32)

    train_inputs = [X_static_train, X_dynamic_train, X_future_train]

    # 2. Define Minimal Search Space
    custom_param_space = {
        'hidden_units': [16], # Fixed for speed
        'num_heads': [2],
        'learning_rate': [1e-3]
    }

    # 3. Instantiate the Tuner
    xtft_tuner_obj = XTFTTuner(
        model_name="xtft",
        param_space=custom_param_space,
        max_trials=1,       # Minimal for demo
        epochs=2,           # Minimal for demo
        batch_sizes=[8],    # Single small batch
        tuner_dir="./xtft_class_tuning",
        project_name="XTFT_Class_Tune",
        tuner_type='random',
        verbose=0
    )

    # 4. Run the Tuning by Calling fit()
    print("Starting XTFT tuning (Class-Based)...")
    best_hps, best_model, tuner = xtft_tuner_obj.fit(
        inputs=train_inputs,
        y=y_train,
        forecast_horizon=H_out,
        quantiles=None, # Point forecast
        case_info={ # Can still pass case_info for extra details
           'description': "My XTFT Point Forecast"
        }
    )

    # 5. Display Results
    print("\nXTFT Tuning complete.")
    if best_hps:
        print("--- Best Hyperparameters (XTFT) ---")
        print(best_hps)
    else:
        print("XTFT Tuning failed to find a best model.")

**Expected Output:**

.. code-block:: text

    Starting XTFT tuning (Class-Based)...
    XTFT Tuning complete.
    --- Best Hyperparameters (XTFT) ---
    {'hidden_units': 16, 'num_heads': 2, 'dropout_rate': 0.3, 'activation': 'gelu', 
    'use_batch_norm': 0, 'embed_dim': 64, 'max_window_size': 10, 'memory_size': 100, 
    'lstm_units': 64, 'attention_units': 128, 'recurrent_dropout_rate': 0.0, 
    'use_residuals': 1, 'final_agg': 'average', 'multi_scale_agg': 'last', 
    'scales_options': 'no_scales', 'learning_rate': 0.001, 'batch_size': 8}

.. raw:: html

    <hr>

.. _tft_tuner_class_doc:

TFTTuner
~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.forecast_tuner.TFTTuner`

**Purpose:**
Provides a dedicated tuner for Temporal Fusion Transformer models.
It supports both the stricter `TFT` (requires all inputs, set
`model_name="tft"`) and the flexible `TemporalFusionTransformer`
(handles optional inputs, set `model_name="tft_flex"`).

**Functionality:**
Similar to :class:`fusionlab.nn.forecast_tuner.XTFTTuner`:

1.  **Initialization (`__init__`)**: Configure the tuning process. Crucially,
    set `model_name` to either `"tft"` or `"tft_flex"`.
2.  **Fitting (`fit`)**: Call `fit` with the data and task parameters.
    * If `model_name="tft"`, `inputs` *must* be `[X_s, X_d, X_f]` with
      non-None tensors.
    * If `model_name="tft_flex"`, `inputs` can be `[X_s, X_d, X_f]`
      where `X_s` and `X_f` can be `None`.
3.  **Tuning**: Executes the search process.
4.  **Results**: Access via `best_hps_`, `best_model_`, `tuner_`.

**Code Example (Tuning Flexible TFT - `tft_flex`):**

.. code-block:: python
    :linenos:

    import numpy as np
    import os
    from fusionlab.nn.forecast_tuner import TFTTuner

    # 1. Prepare Dummy Data (Dynamic only for this example)
    B, T_past, H_out = 8, 12, 6
    D_d = 5
    X_d_train_flex = np.random.rand(B, T_past, D_d).astype(np.float32)
    y_train_flex = np.random.rand(B, H_out, 1).astype(np.float32)

    # Inputs for flexible TFT (static and future are None)
    train_inputs_flex = [None, X_d_train_flex, None]

    # 2. Define Minimal Search Space
    param_space_flex = {'hidden_units': [16], 'learning_rate': [1e-3]}

    # 3. Instantiate the Tuner
    tft_tuner_obj = TFTTuner(
        model_name="tft_flex", # Key: Using the flexible version
        param_space=param_space_flex,
        max_trials=1,
        epochs=1,
        batch_sizes=[4],
        tuner_dir="./tft_flex_class_tuning",
        project_name="TFT_Flex_Class_Tune",
        verbose=0
    )

    # 4. Run Tuning
    print("\nStarting flexible TFT (tft_flex) tuning (Class-Based)...")
    best_hps_f, _, _ = tft_tuner_obj.fit(
        inputs=train_inputs_flex,
        y=y_train_flex,
        forecast_horizon=H_out,
        quantiles=None
    )

    # 5. Display Results
    print("Flexible TFT Tuning complete.")
    if best_hps_f:
        print("  Best HPs (Flexible TFT):", best_hps_f)


**Expected Output:**

.. code-block:: text

    Starting flexible TFT (tft_flex) tuning (Class-Based)...
    Flexible TFT Tuning complete.
      Best HPs (Flexible TFT): {'hidden_units': 16, 'num_heads': 4, 'dropout_rate': 0.3,
       'activation': 'relu', 'use_batch_norm': 1, 'num_lstm_layers': 2, 'lstm_units': 128,
        'learning_rate': 0.001, 'batch_size': 4}

.. raw:: html

    <hr>

Customizing the Tuning
----------------------

While the default settings and model builders are powerful, you
can customize the process:

* **param_space**: Provide a dictionary in the `__init__` method to
    define specific ranges or choices for any hyperparameter used by the
    internal `_model_builder_factory`. See Keras Tuner documentation for
    how to define choices.
* **model_builder**: For complete control, you can write your own
    function that takes `hp` (Keras Tuner's `HyperParameters` object)
    and returns a compiled Keras model. Pass this function to the
    `__init__` method. This allows you to explore entirely different
    architectures or compilation strategies.

This class-based approach in ``fusionlab-learn`` provides a robust and
flexible framework for efficiently finding optimal hyperparameters for
your time-series forecasting models.