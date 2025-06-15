.. _pure_transformer_models_guide:

=========================
Pure Transformer Models
=========================

:API Reference: :class:`~fusionlab.nn.transformers.TimeSeriesTransformer`

This section of the user guide focuses on "pure" transformer models,
which rely exclusively on attention mechanisms to process sequential
data. Unlike hybrid models that combine LSTMs and attention, these
architectures adhere to the original "Attention Is All You Need"
paradigm, making them exceptionally powerful for capturing long-range
dependencies and complex feature interactions.

The primary model in this category is the ``TimeSeriesTransformer``, a
versatile implementation of the classic encoder-decoder transformer,
specifically adapted for multi-horizon, multi-variate time series
forecasting.

TimeSeriesTransformer
-----------------------

The ``TimeSeriesTransformer`` is a canonical implementation of the
transformer architecture for forecasting. It is designed to be a
robust, general-purpose tool capable of handling diverse time series
data by processing static, dynamic past, and known future features.

Key Features
--------------

* **Classic Encoder-Decoder Architecture:** Faithfully implements the
  standard transformer structure with separate stacks of encoder and
  decoder layers, providing a strong and well-understood foundation.

* **Solely Attention-Based:** Relies entirely on self-attention and
  cross-attention to model temporal dependencies, making it distinct
  from recurrent or convolutional approaches.

* **Multi-Feature Input:** Natively handles three distinct input streams:

  * **Static features:** Time-invariant context.
  * **Dynamic past features:** Historical time series data.
  * **Known future features:** Covariates with known future values.

* **Flexible Static Feature Integration:** Provides multiple strategies
  (via ``static_integration_mode``) for injecting static context
  into the model, either at the encoder input, the decoder input, or
  not at all.

* **Causal Masking for Autoregressive Decoding:** Correctly applies a
  look-ahead mask in the decoder's self-attention layer to ensure
  that predictions for a given time step are only conditioned on
  previous steps, which is crucial for preventing data leakage during
  training.

* **Probabilistic Forecasting:** Can be configured to produce
  quantile forecasts by providing a list of ``quantiles``, enabling
  the model to output prediction intervals and quantify uncertainty.

Architectural Deep Dive
~~~~~~~~~~~~~~~~~~~~~~~~~
The ``TimeSeriesTransformer`` follows the classic architecture composed
of an encoder, a decoder, and input processing layers.

**1. Input Processing and Embedding**

Before entering the main transformer blocks, each input tensor is
processed to have the same core dimensionality, :math:`d_{model}`
(defined by ``embed_dim``).

* **Dynamic and Future Inputs:** Are passed through separate
  :class:`~keras.layers.Dense` layers to project them into the
  :math:`d_{model}` space.
* **Static Inputs:** Are processed by either a `Dense` layer or a
  :class:`~fusionlab.nn.components.GatedResidualNetwork` (GRN).
* **Positional Encoding:** Crucially, sinusoidal positional encodings
  are added to the dynamic and future embeddings to provide the
  model with information about the order of the sequence, since the
  attention mechanism itself has no inherent sense of position.

.. math::
   \mathbf{X}_{emb} = \text{Embedding}(\mathbf{X}_{input}) + \text{PositionalEncoding}

**2. The Encoder Stack**

The encoder's job is to build a rich, contextualized representation of
the historical (dynamic past) input sequence. It consists of a stack
of ``num_encoder_layers``, where each layer performs two main operations:

* **Multi-Head Self-Attention:** This allows every time step in the
  input sequence to "look at" and weigh the importance of every other
  time step in the sequence. This is how the model captures long-range
  dependencies and complex interactions within the historical data.
* **Position-wise Feed-Forward Network (FFN):** A simple, fully
  connected feed-forward network applied independently to each time step.

A residual connection followed by layer normalization is applied after
each operation to ensure stable training.

**3. The Decoder Stack**

The decoder's job is to take the encoded historical context and the known
future inputs to generate the output forecast sequence. It consists of
a stack of ``num_decoder_layers``, where each layer has **three** main
operations:

* **Masked Multi-Head Self-Attention:** This layer performs self-attention
  on the decoder's inputs (the known future features). A **causal
  (look-ahead) mask** is applied to ensure that the prediction for
  time step :math:`i` can only attend to outputs at positions less than
  :math:`i`. This prevents the model from cheating by looking at future
  answers during training and preserves the autoregressive property.

* **Multi-Head Cross-Attention:** This is the bridge between the
  encoder and decoder. Here, the decoder's output from the previous
  sub-layer acts as the *query*, while the encoder's output memory serves
  as the *keys* and *values*. This allows the decoder to "focus" on the
  most relevant parts of the historical input sequence for generating
  each step of the forecast.

* **Position-wise Feed-Forward Network (FFN):** Identical in structure
  to the FFN in the encoder.

**4. Final Output Layer**

After passing through the decoder stack, a final linear (Dense) layer
projects the output of the last decoder layer into the desired output
shape (:math:`output\_dim`). If ``quantiles`` are specified, this is
then passed to a :class:`~fusionlab.nn.components.QuantileDistributionModeling`
layer to produce the final probabilistic forecast.

Complete Usage Example
------------------------
This example demonstrates a complete workflow for using the
``TimeSeriesTransformer`` for a multi-horizon, probabilistic forecast.

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.transformers import TimeSeriesTransformer

   # 1. Model & Data Configuration
   BATCH_SIZE = 16
   PAST_STEPS = 30
   HORIZON = 10
   STATIC_DIM, DYNAMIC_DIM, FUTURE_DIM = 4, 5, 3

   # 2. Instantiate the TimeSeriesTransformer
   model = TimeSeriesTransformer(
       static_input_dim=STATIC_DIM,
       dynamic_input_dim=DYNAMIC_DIM,
       future_input_dim=FUTURE_DIM,
       embed_dim=32,
       num_heads=4,
       ffn_dim=64,
       num_encoder_layers=2,
       num_decoder_layers=2,
       forecast_horizon=HORIZON,
       output_dim=1,
       quantiles=[0.1, 0.5, 0.9],
       static_integration_mode='add_to_decoder_input'
   )

   # 3. Prepare Dummy Input Data
   static_input = tf.random.normal([BATCH_SIZE, STATIC_DIM])
   dynamic_input = tf.random.normal([BATCH_SIZE, PAST_STEPS, DYNAMIC_DIM])
   future_input = tf.random.normal([BATCH_SIZE, HORIZON, FUTURE_DIM])

   # 4. Get Model Output
   # The model expects inputs as a list: [static, dynamic, future]
   predictions = model([static_input, dynamic_input, future_input])

   # 5. Check Output Shape
   print(f"Model Input Shapes:")
   print(f"  Static: {static_input.shape}")
   print(f"  Dynamic: {dynamic_input.shape}")
   print(f"  Future: {future_input.shape}")
   print(f"\nModel Output Shape: {predictions.shape}")
   print("(Batch, Horizon, Quantiles)")

**Expected Output:**

.. code-block:: text

   Model Input Shapes:
     Static: (16, 4)
     Dynamic: (16, 30, 5)
     Future: (16, 10, 3)

   Model Output Shape: (16, 10, 3)
   (Batch, Horizon, Quantiles)
   
Next Steps
----------

.. note::

   Now that you understand the theory and the complete workflow for
   ``TimeSeriesTransformer``, you can proceed to the exercises for more hands-on practice:
   :doc:`../../exercises/exercise_pure_transformer`