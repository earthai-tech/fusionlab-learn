.. _halnet_guide:

===========================================
HALNet (Hybrid Attentive LSTM Network)
===========================================

:API Reference: :class:`~fusionlab.nn.models.HALNet`

The Hybrid Attentive LSTM Network (``HALNet``) is a powerful,
data-driven model designed for complex, multi-horizon time series
forecasting. It forms the architectural core of the
:class:`~fusionlab.nn.pinn.models.PIHALNet` model but is provided
as a standalone tool for general-purpose forecasting tasks that do
not require physics-informed constraints.

``HALNet`` leverages a sophisticated encoder-decoder framework,
integrating multi-scale LSTMs and a suite of advanced attention
mechanisms to capture complex temporal patterns from static, dynamic
past, and known future inputs.

Key Features
------------
* **Flexible Encoder-Decoder Architecture**: The model can operate
  in two distinct modes via the ``mode`` parameter:
    * **`pihal_like`**: A standard sequence-to-sequence architecture
      where the encoder processes past data and the decoder uses
      future data.
    * **`tft_like`**: An architecture inspired by the Temporal
      Fusion Transformer where known future inputs are used to enrich
      both the historical context (encoder) and the future context
      (decoder).

* **Encoder-Decoder Architecture:** Correctly processes historical
  data (in the encoder) and future context (in the decoder)
  separately, making it robust to differing lookback and forecast
  horizon lengths.
  
* **Advanced Input Handling:** Accepts three distinct types of
  inputs: static, dynamic (past observed), and known future
  features. It can optionally use
  :class:`~fusionlab.nn.components.VariableSelectionNetwork` (VSN)
  for intelligent, learnable feature selection and embedding for
  each input type.
  
* **Multi-Scale Temporal Processing:** Employs a
  :class:`~fusionlab.nn.components.MultiScaleLSTM` in the encoder
  to capture temporal dependencies at various user-defined
  resolutions (via the ``scales`` parameter).
  
* **Rich Attention Mechanisms:** Uses a suite of attention layers to
  effectively fuse information from different sources:
    * :class:`~fusionlab.nn.components.CrossAttention` allows the
      decoder to focus on the most relevant parts of the encoded
      historical context.
    * :class:`~fusionlab.nn.components.HierarchicalAttention` and
      :class:`~fusionlab.nn.components.MemoryAugmentedAttention`
      further refine the decoder's context.
    * :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`
      integrates the final set of features before prediction.
      
* **Probabilistic Forecasting:** Employs
  :class:`~fusionlab.nn.components.QuantileDistributionModeling`
  to output forecasts for specified ``quantiles``, enabling the
  estimation of prediction uncertainty. It produces standard point
  forecasts if ``quantiles`` is ``None``.

When to Use HALNet
------------------
``HALNet`` is an excellent choice for complex forecasting problems
where:

* You have rich inputs, including static metadata, historical time
  series, and information about the future.
* The underlying temporal dynamics are complex and may exist at
  multiple time scales.
* You need to forecast multiple time steps into the future
  (multi-horizon forecasting).
* Capturing long-range dependencies and complex interactions between
  different features is important for accuracy.

Architectural Workflow
~~~~~~~~~~~~~~~~~~~~~~~~
``HALNet``'s architecture is organized into an encoder-decoder
structure. The key difference between its operational modes lies in
how it handles the ``future_input`` tensor.

**Input Modes: `tft_like` vs. `pihal_like`**

* **`mode='pihal_like'` (Standard Encoder-Decoder):**
    * In this mode, ``future_input`` is expected to have a time
      dimension equal to the ``forecast_horizon``.
    * **Encoder**: Processes only the `dynamic_input` (of length
      :math:`T_{past}`) to create a summary of the past.
    * **Decoder**: Uses the encoder's summary along with the
      `static_input` and the entire `future_input` to generate
      the forecast. This is a clean and robust separation of concerns.

* **`mode='tft_like'` (TFT-Style Inputs):**
    * This mode requires the ``future_input`` tensor to span both
      the lookback and forecast periods, with a time dimension of
      :math:`T_{past} + T_{future}`.
    * **Encoder**: The `future_input` is sliced. Its historical part
      (length :math:`T_{past}`) is concatenated with the
      `dynamic_input` and fed into the encoder. This provides the
      encoder with richer context about past events.
    * **Decoder**: The future part of the `future_input` (length
      :math:`T_{future}`) is used as context for generating the
      prediction.

**Subsequent Steps (Common to Both Modes):**

1.  **Initial Feature Processing:**

    Both static and time-varying inputs (`dynamic` and `future`) are
    first processed to create feature representations. If
    :py:attr:`use_vsn` is ``True``, each input type is passed through
    its own :class:`~fusionlab.nn.components.VariableSelectionNetwork`
    and a subsequent :class:`~fusionlab.nn.components.GatedResidualNetwork`
    (GRN). If ``False``, they are processed by standard :class:`~keras.layers.Dense`
    layers.

2.  **Encoder Path:**

    The encoder's role is to create a rich, contextualized summary of
    all past information.

    * The historical parts of the `dynamic_input` and `future_input`
      (a slice of length :math:`T_{past}`) are combined.
    * This combined tensor is passed through a
      :class:`~fusionlab.nn.components.MultiScaleLSTM`.
    * The outputs from different LSTM scales are aggregated by
      :func:`~fusionlab.nn.components.aggregate_multiscale` into a
      single 3D tensor, :math:`\mathbf{E} \in \mathbb{R}^{B \times T' \times D_{enc}}`,
      which represents the complete encoded history. :math:`T'` is the
      (potentially sliced) time dimension of the past.

3.  **Decoder Path:**

    The decoder prepares the context for the forecast window
    (:math:`T_{future}` or :py:attr:`forecast_horizon`).

    * The static context vector is tiled across the forecast horizon.
    * The future part of the `future_input` tensor (of length
      :math:`T_{future}`) is combined with the tiled static context.
    * This combined tensor is projected by a :class:`~keras.layers.Dense`
      layer to create the initial decoder context,
      :math:`\mathbf{D}_{init} \in \mathbb{R}^{B \times T_{future} \times D_{attn}}`.

4.  **Attention-Based Fusion:**

    The decoder context acts as a *query* to the encoder's output sequences
    (which serve as *keys* and *values*) via :class:`~fusionlab.nn.components.CrossAttention`.
    This allows the model to focus on the most relevant historical information
    for each future time step it predicts. This is where the model intelligently 
    combines the past and future.

    * **Cross-Attention:** The decoder context :math:`\mathbf{D}_{init}`
      acts as the *query* to attend to the encoded history
      :math:`\mathbf{E}` (which serves as the *key* and *value*).
      .. math::
          \mathbf{A}_{cross} = \text{CrossAttention}(\mathbf{D}_{init}, \mathbf{E})

    * **Context Refinement:** The output of the cross-attention is
        further processed through residual connections, normalization, and
        other self-attention layers (`HierarchicalAttention`,
        `MemoryAugmentedAttention`, `MultiResolutionAttentionFusion`) to
        build a highly refined feature representation for the forecast period.
        
        * **Residual Connection:** The output of the cross-attention is added
          to the initial decoder input and normalized, a standard technique
          for stabilizing deep models.
          .. math::
              \mathbf{D}' = \text{LayerNorm}(\mathbf{D}_{init} + \text{GRN}(\mathbf{A}_{cross}))
    
        * **Self-Attention:** Further attention layers (Hierarchical, Memory,
          Multi-Resolution Fusion) refine this fused context :math:`\mathbf{D}'`
          through self-attention mechanisms.

5.  **Final Aggregation and Output:**
    * The final feature tensor from the attention blocks, which has a
      shape of :math:`(B, T_{future}, D_{feat})`, is aggregated along the
      time dimension using the specified ``final_agg`` strategy (e.g.,
      taking the 'last' step or 'average'). This produces a single
      vector per sample.
    * This vector is passed to the :class:`~fusionlab.nn.components.MultiDecoder`
      to generate predictions for each step in the horizon.
    * Finally, :class:`~fusionlab.nn.components.QuantileDistributionModeling`
      maps the decoder's output to the final point or quantile forecasts.


Complete Example
----------------
This example demonstrates a complete workflow for ``HALNet`` using the
`tft_like` mode, which has the more complex data requirement.

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~
First, we import all necessary libraries and set up the environment.

.. code-block:: python
   :linenos:

   import os
   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   import warnings

   # FusionLab imports
   from fusionlab.nn.models import HALNet
   from fusionlab.nn.utils import reshape_xtft_data
   from fusionlab.nn.models.utils import plot_history_in
   from fusionlab._fusionlog import fusionlog

   logger = fusionlog().get_fusionlab_logger(__name__)
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')

   EXERCISE_OUTPUT_DIR = "./halnet_exercise_outputs"
   os.makedirs(EXERCISE_OUTPUT_DIR, exist_ok=True)


Step 2: Generate and Prepare Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We generate a synthetic dataset and use `reshape_xtft_data` to create
the three required input arrays (`static`, `dynamic`, `future`).

.. code-block:: python
   :linenos:

   # Configuration
   N_ITEMS = 3
   N_TIMESTEPS_PER_ITEM = 100
   TIME_STEPS = 14
   FORECAST_HORIZON = 7
   TARGET_COL = 'Value'
   DT_COL = 'Date'

   # Generate synthetic data (code omitted for brevity, see exercise page)
   # ...
   # Preprocessing (LabelEncoding, Scaling)
   # ...

   # For this example, we'll create dummy arrays with the correct shapes
   # that `reshape_xtft_data` would output.
   n_sequences = N_ITEMS * (N_TIMESTEPS_PER_ITEM - TIME_STEPS - FORECAST_HORIZON + 1)
   
   static_data = np.random.rand(n_sequences, 2)  # e.g., ItemID, Category
   dynamic_data = np.random.rand(n_sequences, TIME_STEPS, 3) # e.g., ValueLag1, DayOfWeek
   # Future data spans both past and future windows for 'tft_like' mode
   future_data = np.random.rand(n_sequences, TIME_STEPS + FORECAST_HORIZON, 2)
   targets = np.random.rand(n_sequences, FORECAST_HORIZON, 1)

   print(f"Generated data shapes for 'tft_like' mode:")
   print(f"  Static:  {static_data.shape}")
   print(f"  Dynamic: {dynamic_data.shape}")
   print(f"  Future:  {future_data.shape}")
   print(f"  Target:  {targets.shape}")


Step 3: Define, Compile, and Train HALNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We instantiate the model, specifying `mode='tft_like'`, and then
compile and train it.

.. code-block:: python
   :linenos:

   # Split data into training and validation sets
   train_inputs = [arr[:-20] for arr in [static_data, dynamic_data, future_data]]
   val_inputs = [arr[-20:] for arr in [static_data, dynamic_data, future_data]]
   train_targets, val_targets = targets[:-20], targets[-20:]

   # Instantiate HALNet for 'tft_like' operation
   halnet_model = HALNet(
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1],
       output_dim=1,
       forecast_horizon=FORECAST_HORIZON,
       max_window_size=TIME_STEPS,
       mode='tft_like', # Specify the mode
       use_vsn=False,
       hidden_units=16,
       attention_units=16
   )

   # Compile and train
   halnet_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
   print("\\nTraining HALNet model...")
   history = halnet_model.fit(
       train_inputs,
       train_targets,
       validation_data=(val_inputs, val_targets),
       epochs=10,
       batch_size=32,
       verbose=0 # Set to 1 to see progress
   )
   print("Training complete.")


Step 4: Visualize Training History
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use the ``plot_history_in`` utility to visualize the loss curves.

.. code-block:: python
   :linenos:

   print("\\nPlotting training history...")
   plot_history_in(
       history,
       metrics={"Loss": ["loss"], "MAE": ["mae"]},
       layout='subplots',
       title="HALNet Training and Validation History"
   )


**Example Output Plot:**

.. figure:: ../images/halnet_history_plot.png
   :alt: HALNet Training History Plot
   :align: center
   :width: 90%

   An example plot showing the training and validation loss and Mean
   Absolute Error (MAE) over epochs. This helps in diagnosing model
   fit and convergence.

