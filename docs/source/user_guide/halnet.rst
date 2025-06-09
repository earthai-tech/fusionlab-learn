.. _halnet_guide:

===========================================
HALNet (Hybrid Attentive LSTM Network)
===========================================

:API Reference: :class:`~fusionlab.nn.models.HALNet`

The Hybrid Attentive LSTM Network (``HALNet``) is a powerful,
data-driven model for multi-horizon time series forecasting. It is
the core data-driven architecture of
:class:`~fusionlab.nn.pinn.models.PIHALNet` but is provided as a
standalone model for general-purpose forecasting tasks that do not
require physics-informed constraints.

``HALNet`` leverages a sophisticated encoder-decoder framework,
integrating multi-scale LSTMs and a suite of advanced attention
mechanisms to capture complex temporal patterns from static, dynamic
past, and known future inputs.

**Key Features:**

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

**When to Use HALNet:**

``HALNet`` is an excellent choice for complex forecasting problems where:

* You have rich inputs, including static metadata, historical time
  series, and information about the future.
* The underlying temporal dynamics are complex and may exist at
  multiple time scales.
* You need to forecast multiple time steps into the future
  (multi-horizon forecasting).
* Capturing long-range dependencies and complex interactions between
  different features is important for accuracy.
* You require a high-performance, data-driven model without the need
  for explicit physical constraints.

Architectural Workflow and Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``HALNet``'s architecture is organized into an encoder-decoder
structure, ensuring that past and future information are handled in
their appropriate contexts.

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
    This is where the model intelligently combines the past and future.

    * **Cross-Attention:** The decoder context :math:`\mathbf{D}_{init}`
      acts as the *query* to attend to the encoded history
      :math:`\mathbf{E}` (which serves as the *key* and *value*).
      .. math::
          \mathbf{A}_{cross} = \text{CrossAttention}(\mathbf{D}_{init}, \mathbf{E})

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


Code Example
------------
The following example shows how to instantiate, compile, and train
a `HALNet` model on synthetic data.

.. code-block:: python
   :linenos:

   import numpy as np
   import tensorflow as tf
   from fusionlab.nn.models import HALNet
   from tensorflow.keras.losses import MeanSquaredError
   from tensorflow.keras.optimizers import Adam

   # 1. Define model and data configuration
   BATCH_SIZE = 4
   TIME_STEPS = 8          # Lookback window for dynamic features
   FORECAST_HORIZON = 4    # How many steps to predict
   STATIC_DIM = 3
   DYNAMIC_DIM = 5
   FUTURE_DIM = 2
   OUTPUT_DIM = 1

   # 2. Instantiate the HALNet model
   halnet_model = HALNet(
       static_input_dim=STATIC_DIM,
       dynamic_input_dim=DYNAMIC_DIM,
       future_input_dim=FUTURE_DIM,
       output_dim=OUTPUT_DIM,
       forecast_horizon=FORECAST_HORIZON,
       max_window_size=TIME_STEPS,
       quantiles=None, # For a simple point forecast
       embed_dim=16,
       hidden_units=16,
       lstm_units=16,
       attention_units=16, # Kept consistent for simplicity
       num_heads=2,
       use_vsn=False # Use simpler Dense layers for this example
   )

   # 3. Generate correctly shaped dummy data
   # The future_features tensor must span both the lookback and forecast periods
   future_span = TIME_STEPS + FORECAST_HORIZON

   static_data = tf.random.normal((BATCH_SIZE, STATIC_DIM))
   dynamic_data = tf.random.normal((BATCH_SIZE, TIME_STEPS, DYNAMIC_DIM))
   future_data = tf.random.normal((BATCH_SIZE, future_span, FUTURE_DIM))

   # The input to the model is a list of these three tensors
   model_inputs = [static_data, dynamic_data, future_data]

   # The target data shape must match the forecast horizon
   target_data = tf.random.normal((BATCH_SIZE, FORECAST_HORIZON, OUTPUT_DIM))

   # 4. Compile the model
   halnet_model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

   # 5. Train the model
   print("Training HALNet model...")
   halnet_model.fit(
       model_inputs,
       target_data,
       epochs=3,
       batch_size=BATCH_SIZE,
       verbose=1
   )
   print("Training complete.")

   # 6. Make a prediction
   predictions = halnet_model.predict(model_inputs)
   print(f"Shape of predictions: {predictions.shape}")

