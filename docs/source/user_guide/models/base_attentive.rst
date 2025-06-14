.. _base_attentive_guide:

=================================
Base Attentive Model Architecture
=================================

:API Reference: :class:`~fusionlab.nn.models.BaseAttentive`

The ``BaseAttentive`` class is not a model intended for direct,
standalone use, but rather the powerful and flexible architectural
foundation for a suite of advanced, multi-horizon time series
forecasting models like ``HALNet`` and ``PIHALNet``. It encapsulates a
sophisticated data-driven, encoder-decoder framework designed to
capture complex temporal patterns from diverse data sources.

Understanding ``BaseAttentive`` is key to understanding how its child
models work. It provides a highly configurable engine that processes
three distinct input streams—**static**, **dynamic (past observed)**,
and **known future** features—using a combination of LSTMs,
self-attention, and cross-attention mechanisms.

Key Features
------------

* **Advanced Input Handling:** Natively processes three types of
  inputs. It can optionally use a
  :class:`~fusionlab.nn.components.VariableSelectionNetwork` (VSN)
  for intelligent, learnable feature selection and embedding for
  each input type.

* **Configurable Encoder Architecture:** The internal encoder can
  be dynamically configured to operate in two modes:
  * **`hybrid`**: A powerful combination of LSTMs and attention,
    ideal for capturing patterns at multiple time scales.
  * **`transformer`**: A pure self-attention architecture,
    excellent for problems with very long-range dependencies.

* **Modular Attention Stack:** The decoder uses a sequence of
  attention layers to fuse information. The user can fully
  customize this stack, choosing the order and type of attention
  mechanisms applied (e.g., cross-attention, hierarchical,
  memory-augmented).

* **Multi-Scale Temporal Processing:** When using the `hybrid`
  encoder, it employs a
  :class:`~fusionlab.nn.components.MultiScaleLSTM` to capture
  temporal dependencies at various user-defined resolutions (via
  the ``scales`` parameter).

* **Probabilistic Forecasting:** Natively supports uncertainty
  quantification by using a
  :class:`~fusionlab.nn.components.QuantileDistributionModeling`
  head to output forecasts for a specified list of ``quantiles``.

Architectural Deep Dive
~~~~~~~~~~~~~~~~~~~~~~~~~
The ``BaseAttentive`` architecture is organized into a five-stage
encoder-decoder pipeline.

**1. Initial Feature Processing**
----------------------------------
Each of the three input streams (static, dynamic, future) is first
passed through a feature processing block. This can be configured
via the `architecture_config`.

* **VSN Path (`feature_processing='vsn'`)**:
  Each input is passed through its own
  :class:`~fusionlab.nn.components.VariableSelectionNetwork`, which
  learns the relevance of each input variable. The output is then
  processed by a
  :class:`~fusionlab.nn.components.GatedResidualNetwork` (GRN) to
  create a robust feature embedding. This is the default and
  recommended path.

* **Dense Path (`feature_processing='dense'`)**:
  If VSN is disabled, features are processed through standard
  :class:`~keras.layers.Dense` layers.

**2. The Encoder Path**
--------------------------
The encoder's role is to create a rich, contextualized summary of all
past information. First, positional information is added to the time-
varying inputs via :class:`~fusionlab.nn.components.PositionalEncoding`.
Then, the core processing begins, as defined by `encoder_type`.

* **Hybrid Encoder (`encoder_type='hybrid'`)**:
  The processed dynamic inputs are fed into a
  :class:`~fusionlab.nn.components.MultiScaleLSTM`. This layer
  contains multiple parallel LSTMs, each processing the input sequence
  at a different temporal resolution (e.g., looking at every step,
  every 3rd step, etc.). This allows the model to capture both
  short-term and long-term patterns simultaneously. The outputs from
  all scales are then concatenated to form the final encoder memory.

* **Transformer Encoder (`encoder_type='transformer'`)**:
  This path bypasses LSTMs entirely. The input sequence is instead
  processed by a stack of standard transformer encoder blocks, each
  consisting of a :class:`~fusionlab.nn.components.MultiHeadAttention`
  layer followed by a residual connection and layer normalization.
  .. math::
     \mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}))

**3. The Decoder Path & Context Preparation**
-------------------------------------------------
The decoder prepares the context for the forecast window. The static
context vector (from Step 1) is tiled (repeated) across the forecast
horizon. This is combined with the processed known future features to
form the initial decoder input, which serves as the **query** for the
attention mechanisms.

**4. The Attention Stack**
-----------------------------
This is the heart of the model, where information from the past (encoder)
and future (decoder) is intelligently fused. The `decoder_attention_stack`
config controls which layers are used and in what order.

* **Cross-Attention (`'cross'`)**: This is the crucial encoder-decoder
  interaction. The decoder context from Step 3 acts as the *query*,
  while the encoder's output memory (from Step 2) serves as the
  *keys* and *values*. The model learns to "pay attention" to the most
  relevant historical time steps for each future step it needs to predict.
  .. math::
     \mathbf{A}_{cross} = \text{Attention}(\mathbf{Q}_{decoder}, \mathbf{K}_{encoder}, \mathbf{V}_{encoder})

* **Self-Attention (`'hierarchical'`, `'memory'`)**: After cross-attention,
  the resulting context is further refined using self-attention
  mechanisms. Hierarchical attention helps find structural patterns, while
  memory-augmented attention is designed to capture very long-range
  dependencies.

* **Residual Connections**: Throughout the stack, GRNs and residual
  connections are used to ensure stable training of this deep
  architecture.

**5. Final Output Generation**
-------------------------------
The highly-refined feature tensor from the attention stack is passed
to a :class:`~fusionlab.nn.components.MultiDecoder`, which has separate
output heads to generate a prediction for each step in the forecast
horizon. If quantiles are requested, these point forecasts are finally
passed to the :class:`~fusionlab.nn.components.QuantileDistributionModeling`
layer to produce the final probabilistic forecast.

Smart Configuration System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To provide maximum flexibility, ``BaseAttentive`` uses a dedicated
``architecture_config`` dictionary to control its internal structure.
This separates the model's data shape definitions from its internal
architectural choices.

The primary keys are:

* **`encoder_type`**: `'hybrid'` (default) or `'transformer'`.
* **`decoder_attention_stack`**: A list from `['cross', 'hierarchical', 'memory']`.
* **`feature_processing`**: `'vsn'` (default) or `'dense'`.

You can also create variants of an existing model using the
``.reconfigure()`` method.

**Usage Examples for Configuration**

.. code-block:: python
   :linenos:

   from fusionlab.nn.models import BaseAttentive
   import warnings

   # Suppress the deprecation warning for the example
   warnings.filterwarnings("ignore", category=FutureWarning)

   # --- 1. Default Model (Hybrid Encoder, Full Attention) ---
   model_default = BaseAttentive(
       static_input_dim=2, dynamic_input_dim=3, future_input_dim=2,
       output_dim=1, forecast_horizon=7
   )
   print("Default Config:", model_default.architecture_config)


   # --- 2. Create a Pure Transformer Model from Scratch ---
   tfmr_config = {
       'encoder_type': 'transformer',
       'feature_processing': 'dense'
   }
   model_transformer = BaseAttentive(
       static_input_dim=2, dynamic_input_dim=3, future_input_dim=2,
       output_dim=1, forecast_horizon=7,
       architecture_config=tfmr_config
   )
   print("\nTransformer Config:", model_transformer.architecture_config)


   # --- 3. Use reconfigure() to create a lightweight variant ---
   model_lightweight = model_default.reconfigure({
       'decoder_attention_stack': ['cross'] # Simpler decoder
   })
   print("\nLightweight Reconfigured:", model_lightweight.architecture_config)


   # --- 4. Using a deprecated key (triggers a warning) ---
   # This shows how backward compatibility is handled.
   deprecated_config = {'objective': 'transformer'}
   print("\nInstantiating with deprecated 'objective' key...")
   model_deprecated = BaseAttentive(
       static_input_dim=2, dynamic_input_dim=3, future_input_dim=2,
       output_dim=1, forecast_horizon=7,
       architecture_config=deprecated_config
   )

**Expected Output:**

.. code-block:: text

   Default Config: {'encoder_type': 'hybrid', 'decoder_attention_stack': ['cross', 'hierarchical', 'memory'], 'feature_processing': 'vsn'}

   Transformer Config: {'encoder_type': 'transformer', 'decoder_attention_stack': ['cross', 'hierarchical', 'memory'], 'feature_processing': 'dense'}

   Lightweight Reconfigured: {'encoder_type': 'hybrid', 'decoder_attention_stack': ['cross'], 'feature_processing': 'vsn'}

   Instantiating with deprecated 'objective' key...

Complete Usage Example
----------------------
This example shows the end-to-end workflow for using the
``BaseAttentive`` model (or any of its children).

.. code-block:: python
   :linenos:

   import tensorflow as tf

   # 1. Model Configuration
   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       max_window_size=10,
       mode='tft_like', # Requires future_input to span past and future
       quantiles=[0.1, 0.5, 0.9],
       architecture_config={
           'encoder_type': 'hybrid',
           'feature_processing': 'vsn'
       }
   )

   # 2. Prepare Dummy Input Data
   BATCH_SIZE = 32
   PAST_STEPS = 10
   HORIZON = 24

   x_static  = tf.random.normal([BATCH_SIZE, 4])
   x_dynamic = tf.random.normal([BATCH_SIZE, PAST_STEPS, 8])
   # For 'tft_like', future input has length PAST_STEPS + HORIZON
   x_future  = tf.random.normal([BATCH_SIZE, PAST_STEPS + HORIZON, 6])

   # 3. Get Model Output
   # Inputs must be a list: [static, dynamic, future]
   y_hat = model([x_static, x_dynamic, x_future])

   # 4. Check Output Shape
   print(f"Model Input Shapes:")
   print(f"  Static: {x_static.shape}")
   print(f"  Dynamic: {x_dynamic.shape}")
   print(f"  Future: {x_future.shape}")
   print(f"\nModel Output Shape: {y_hat.shape}")
   print("(Batch, Horizon, Quantiles, Output_Dim)")

**Expected Output:**

.. code-block:: text

   Model Input Shapes:
     Static: (32, 4)
     Dynamic: (32, 10, 8)
     Future: (32, 34, 6)

   Model Output Shape: (32, 24, 3, 2)
   (Batch, Horizon, Quantiles, Output_Dim)