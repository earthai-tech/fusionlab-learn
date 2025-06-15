.. _halnet_tuner_guide:

=====================================
Tuning HALNet with the XTFTTuner
=====================================

:API Reference: :class:`~fusionlab.nn.forecast_tuner.XTFTTuner`
:Model Reference: :class:`~fusionlab.nn.models.HALNet`

The :class:`~fusionlab.nn.models.HALNet` model, with its rich set of
components like ``MultiScaleLSTM`` and multiple attention layers, has
many hyperparameters that can be optimized to achieve peak performance
on a given dataset.

This guide explains how to use the ``XTFTTuner`` to automatically find
the best set of hyperparameters for your ``HALNet`` model.

.. note::
   Because ``HALNet`` shares its core data-driven architecture with
   the more advanced ``XTFT`` model, the same **``XTFTTuner``** can be
   used to tune both. The process is nearly identical, with the primary
   difference being the set of hyperparameters defined in the
   `search_space`.

End-to-End Workflow
-------------------
The process involves preparing your data, defining a search space
specific to ``HALNet``, and then using the ``XTFTTuner`` to run the
optimization process.

Step 1: Prepare Data
~~~~~~~~~~~~~~~~~~~~~
First, ensure your data is prepared in the three-part format required
by ``HALNet``: dictionaries of NumPy arrays for static, dynamic past,
and known future features, along with a corresponding targets array.

.. code-block:: python
   :linenos:

   import numpy as np

   # --- Define Data Dimensions for the Example ---
   B, T_PAST, HORIZON = 256, 21, 7
   S_DIM, D_DIM, F_DIM = 4, 6, 3
   O_DIM = 1

   # --- Generate Dummy Data Arrays ---
   static_features = np.random.rand(B, S_DIM)
   dynamic_features = np.random.rand(B, T_PAST, D_DIM)
   # For 'tft_like' mode, future features span past + horizon
   future_features = np.random.rand(B, T_PAST + HORIZON, F_DIM)
   targets = np.random.rand(B, HORIZON, O_DIM)

   # Create a validation split
   val_split = -50
   train_inputs = [arr[:val_split] for arr in [static_features, dynamic_features, future_features]]
   val_inputs = [arr[val_split:] for arr in [static_features, dynamic_features, future_features]]
   train_targets, val_targets = targets[:val_split], targets[val_split:]

   print("Generated dummy data for HALNet tuning.")

Step 2: Define the `HALNet` Search Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This dictionary is the core of the tuning experiment. Here, we define
all the architectural and optimization hyperparameters we want the tuner
to explore.

When tuning ``HALNet``, we simply **omit** any hyperparameters that are
specific to ``XTFT``, such as ``anomaly_detection_strategy`` or
``anomaly_loss_weight``.

.. code-block:: python
   :linenos:

   halnet_search_space = {
       # --- Architectural Hyperparameters ---
       "embed_dim": [32, 64],
       "hidden_units": [32, 64],
       "lstm_units": [32, 64],
       "attention_units": [16, 32],
       "num_heads": {"type": "choice", "values": [2, 4]},
       "dropout_rate": {"type": "float", "min_value": 0.1, "max_value": 0.4},
       "use_vsn": {"type": "bool"}, # Tune whether to use VSN or not

       # --- Compile-time Hyperparameters ---
       "learning_rate": [1e-3, 5e-4]
   }
   print("Defined hyperparameter search space for HALNet.")

### Step 3: Create and Run the Tuner
Now, we use the ``XTFTTuner.create()`` factory method. The key step is
to pass the **``HALNet``** class to the `model_name_or_cls` argument.
The tuner will intelligently adapt and build `HALNet` instances during
the search.

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.forecast_tuner import XTFTTuner # Use the XTFT Tuner
   from fusionlab.nn.models import HALNet            # But for the HALNet model

   # 1. Create the tuner instance, passing the HALNet class
   tuner = XTFTTuner.create(
       model_name_or_cls=HALNet, # <-- Specify HALNet here
       inputs_data={"static": static_features, "dynamic": dynamic_features},
       targets_data=targets,
       search_space=halnet_search_space,
       # Provide any fixed params that shouldn't be tuned
       fixed_params={
           "future_input_dim": F_DIM,
           "mode": "tft_like",
           "max_window_size": T_PAST
       },
       # Keras Tuner settings
       objective="val_loss",
       max_trials=5, # Use a small number for this example
       project_name="HALNet_Tuning_Example",
       directory="./halnet_tuner_results",
       overwrite=True
   )

   # 2. Run the search process
   print("\nStarting hyperparameter search for HALNet...")
   best_model, best_hps, _ = tuner.run(
       inputs=train_inputs,
       y=train_targets,
       validation_data=(val_inputs, val_targets),
       epochs=5,
       batch_size=64,
       callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)]
   )

Step 4: Analyze the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the search completes, you can inspect the best hyperparameters
found for your `HALNet` model.

.. code-block:: python
   :linenos:

   print("\n--- Tuning Complete: Best Hyperparameters for HALNet ---")
   if best_hps:
       for hp, value in best_hps.values.items():
           if isinstance(value, float):
               print(f"  - {hp}: {value:.4f}")
           else:
               print(f"  - {hp}: {value}")
   else:
       print("Search did not find any best hyperparameters.")

**Expected Output:**

.. code-block:: text

   --- Tuning Complete: Best Hyperparameters for HALNet ---
     - embed_dim: 32
     - hidden_units: 64
     - lstm_units: 32
     - attention_units: 16
     - num_heads: 2
     - dropout_rate: 0.1827
     - use_vsn: True
     - learning_rate: 0.0010

This workflow demonstrates that the modular design of the tuning utilities
allows them to be flexibly applied to different but related model
architectures, accelerating the path to an optimized model.