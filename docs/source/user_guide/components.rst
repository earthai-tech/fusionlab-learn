.. _user_guide_components:

=================
Model Components
=================

The forecasting models in ``fusionlab``, such as TFT and XTFT, are
built from a collection of specialized, reusable components. These
components handle tasks like feature selection, temporal encoding,
attention calculation, and non-linear transformations.

Understanding these building blocks can help you:

* Gain deeper insight into how the main models work.
* Interpret model behavior by examining intermediate outputs or
  component configurations.
* Customize existing models or build novel architectures using
  these components.

This section provides an overview of the key components available
in ``fusionlab.nn.components``.

Architectural Components
--------------------------

Activation
~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.Activation`

This is a simple utility layer that wraps standard Keras activation
functions (e.g., 'relu', 'elu', 'sigmoid'). Its primary purpose is
to ensure consistent handling and serialization of activation
functions within the ``fusionlab`` framework, particularly when
models or custom layers using activations are saved and loaded. It
internally uses `tf.keras.activations.get()` to resolve string
names to callable functions.

While it can be used directly, users typically specify activations
as strings (e.g., ``activation='relu'``) when initializing other
layers (like :class:`~fusionlab.nn.components.GatedResidualNetwork`),
which then utilize this `Activation` layer or similar internal logic.
Therefore, a direct code example is less illustrative here.

Positional Encoding
~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.PositionalEncoding`

**Purpose:** To inject information about the relative or absolute
position of tokens (time steps) in a sequence. This is crucial for
models like Transformers (and TFT/XTFT which use attention) because
standard self-attention mechanisms are permutation-invariant – they
don't inherently know the order of inputs.

**Functionality:** This layer adds a representation of the time
step index directly to the input feature embeddings at each
position.

.. math::

   Output_t = InputEmbed_t + PositionalInfo_t

The specific implementation here adds the numerical time index
(e.g., :math:`0, 1, 2, ..., T-1`) to each feature dimension,
broadcast across the batch. Other forms of positional encoding
(e.g., sinusoidal) exist, but this simple additive index is used
in this implementation.

**Usage Context:** Applied to the sequence of temporal embeddings
(derived from dynamic past and future inputs) before they are fed
into attention layers or LSTMs in models like
:class:`~fusionlab.nn.TemporalFusionTransformer` and
:class:`~fusionlab.nn.XTFT`.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import PositionalEncoding

   # Dummy input tensor (Batch, TimeSteps, Features)
   B, T, F = 4, 20, 16
   input_sequence = tf.random.normal((B, T, F))

   # Instantiate the layer
   pos_encoding_layer = PositionalEncoding()

   # Apply the layer
   output_sequence = pos_encoding_layer(input_sequence)

   print(f"Input shape: {input_sequence.shape}")
   print(f"Output shape after Positional Encoding: {output_sequence.shape}")
   # Note: Output shape is the same as input shape

   # You can inspect the difference:
   # diff = output_sequence - input_sequence
   # print("Example difference (should show added positional values):")
   # print(diff[0, :5, 0]) # First batch, first 5 steps, first feature

Gated Residual Network (GRN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.GatedResidualNetwork`

**Purpose:** The GRN is arguably one of the most fundamental
building blocks in TFT and related architectures. It provides a
flexible way to apply non-linear transformations to inputs,
optionally conditioned by context, while incorporating gating and
residual connections for stable training of deep networks.

**Functionality:** A GRN typically involves these steps:

1.  **(Optional) Context Addition:** If context information :math:`c`
    is provided, it's linearly transformed (:math:`Linear_c(c)`)
    and added to the primary input :math:`a`.
    Let :math:`a' = a` if no context, or
    :math:`a' = a + Linear_c(c)` if context exists.
2.  **Non-linear Transformation:** The (potentially contextualized)
    input :math:`a'` goes through a main path, often involving a
    Dense layer with activation (:math:`Layer_0`, :math:`act`),
    optional normalization (BN), dropout, and another Dense layer
    (:math:`Layer_1`).
3.  **Gating Mechanism:** A separate dense layer with a sigmoid
    activation (:math:`Layer_g`) processes the initial :math:`a'`
    to calculate a "gate" :math:`g = \sigma(Layer_g(a'))`.
4.  **Gating:** The output of the main transformation path is
    element-wise multiplied by the gate :math:`g`.
5.  **Residual Connection:** The gated output is added back to the
    original input :math:`a` (or a linearly projected version of it
    :math:`Linear_p(a)` if dimensions need matching).
6.  **Layer Normalization:** The final result is normalized using
    Layer Normalization (:math:`LN`).

A simplified representation (details vary slightly based on
implementation, e.g., where exactly dropout/BN are applied):

.. math::

   GRN(a, [c]) \approx LN\left( \text{proj}(a) + \text{Dense}_1(dropout(act(\text{Dense}_0(a')))) \odot \sigma(\text{Dense}_g(a')) \right)

*(See the API reference or the internal code for the precise layer
ordering and transformations)*

**Usage Context:** GRNs are used extensively throughout TFT and XTFT:

* Processing static features to generate context vectors.
* Applying transformations within Variable Selection Networks.
* Processing outputs of attention layers (position-wise feed-forward).
* Static enrichment of temporal features (see :class:`~fusionlab.nn.components.StaticEnrichmentLayer`).

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   # Assuming GatedResidualNetwork is importable
   from fusionlab.nn.components import GatedResidualNetwork

   # Config
   batch_size = 4
   input_features = 32
   units = 16 # GRN output dimension

   # Dummy input and context
   dummy_input = tf.random.normal((batch_size, input_features))
   dummy_context = tf.random.normal((batch_size, units)) # Context matches output units

   # Instantiate GRN
   grn_layer = GatedResidualNetwork(
       units=units,
       dropout_rate=0.1,
       activation='elu',
       use_batch_norm=False
   )

   # Call without context
   output_no_context = grn_layer(dummy_input, training=False)
   print(f"GRN output shape (no context): {output_no_context.shape}")
   # Expected: (4, 16)

   # Call with context
   output_with_context = grn_layer(dummy_input, context=dummy_context, training=False)
   print(f"GRN output shape (with context): {output_with_context.shape}")
   # Expected: (4, 16)

   # Example with input dim != units (triggers projection)
   dummy_input_diff_dim = tf.random.normal((batch_size, input_features + 10))
   output_proj = grn_layer(dummy_input_diff_dim, training=False)
   print(f"GRN output shape (projection): {output_proj.shape}")
   # Expected: (4, 16)


StaticEnrichmentLayer
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.StaticEnrichmentLayer`

**Purpose:** To effectively infuse time-invariant static context
into time-varying temporal features. This allows the model's
processing of temporal patterns (e.g., seasonality, trends learned
by an LSTM) to be conditioned by static attributes (e.g., sensor
location, product category).

**Functionality:**
The layer typically performs the following:

1. Takes a *static context vector* (shape :math:`(B, U)`) and
   *temporal features* (shape :math:`(B, T, U)`), where :math:`U`
   is the number of hidden units.
2. Adds the static context to the temporal features. This often
   requires expanding the static context to match the time
   dimension of the temporal features (e.g., :math:`(B, U) \rightarrow
   (B, 1, U)`) and relying on broadcasting during addition.
3. Passes the combined tensor through an internal
   :class:`~fusionlab.nn.components.GatedResidualNetwork` (GRN) for
   non-linear transformation and gating, producing the final enriched
   temporal features.

**Usage Context:** A standard component in TFT architectures, typically
applied after the sequence encoder (like an LSTM) and before the
main temporal self-attention layer. It injects static information
at a key point in the temporal processing pipeline.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import StaticEnrichmentLayer

   # Config
   batch_size = 4
   time_steps = 20
   units = 16 # Dimension of features and context

   # Dummy temporal features (e.g., LSTM output)
   temporal_features = tf.random.normal((batch_size, time_steps, units))
   # Dummy static context (e.g., GRN output from static features)
   static_context = tf.random.normal((batch_size, units))

   # Instantiate the layer
   enrichment_layer = StaticEnrichmentLayer(units=units, activation='relu')

   # Apply the layer
   # call signature is call(temporal_features, static_context)
   # Adjust if the signature is different (e.g., call([temporal, static]))
   enriched_features = enrichment_layer(temporal_features, static_context)

   # Some implementations might use call(inputs, context=...)
   # enriched_features = enrichment_layer(temporal_features, context=static_context)

   print(f"Input temporal shape: {temporal_features.shape}")
   print(f"Input static context shape: {static_context.shape}")
   print(f"Output enriched shape: {enriched_features.shape}")
   # Expected: (4, 20, 16)


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Input Processing & Embedding Layers
-------------------------------------

These layers handle the initial transformation and embedding of
various input types before they enter the main processing stream of
the models.

LearnedNormalization
~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.LearnedNormalization`

**Purpose:** To normalize input features using scaling parameters
(mean and standard deviation) that are **learned** during model
training, rather than being pre-calculated from the dataset
statistics (like `StandardScaler`).

**Functionality:**

1.  Maintains two trainable weight vectors: `mean` and `stddev`,
    initialized typically near 0 and 1 respectively. Their size
    matches the number of input features (last dimension).
2.  During the forward pass, it applies standard normalization
    to the input tensor :math:`x`:

    .. math::
       x_{norm} = \frac{x - \mu_{learned}}{\sigma_{learned} + \epsilon}

    where :math:`\mu_{learned}` and :math:`\sigma_{learned}` are the
    learned mean and standard deviation weights, and :math:`\epsilon`
    is a small constant (e.g., 1e-6) added for numerical stability,
    preventing division by zero.

**Usage Context:** Used in :class:`~fusionlab.nn.XTFT` as an
initial processing step, often applied to static inputs. This
allows the model to adaptively determine the appropriate
normalization scale and shift for these features based on the data
distribution encountered during training, potentially offering more
flexibility than fixed pre-processing normalization, especially if
input distributions change or differ significantly across deployment
scenarios.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   # Assuming LearnedNormalization is importable
   from fusionlab.nn.components import LearnedNormalization

   # Config
   batch_size = 4
   num_features = 8

   # Dummy input tensor (e.g., static features)
   dummy_input = tf.random.normal((batch_size, num_features))

   # Instantiate the layer
   learned_norm_layer = LearnedNormalization()

   # Build the layer (important for weights to be created)
   # Option 1: Call with data
   _ = learned_norm_layer(dummy_input)
   # Option 2: Explicit build
   # learned_norm_layer.build(dummy_input.shape)

   print(f"Layer trainable weights (mean, stddev): "
         f"{len(learned_norm_layer.trainable_weights)}")
   print(f"Initial Mean (example): {learned_norm_layer.mean.numpy()[:3]}") # Show a few
   print(f"Initial Stddev (example): {learned_norm_layer.stddev.numpy()[:3]}")

   # Apply the layer (e.g., during model call)
   normalized_output = learned_norm_layer(dummy_input, training=True)

   print(f"\nInput shape: {dummy_input.shape}")
   print(f"Normalized output shape: {normalized_output.shape}")
   # Note: Shape remains the same, values are normalized


MultiModalEmbedding
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MultiModalEmbedding`

**Purpose:** To process multiple sequences (representing different
modalities or feature groups), potentially having different numbers
of features initially, by projecting each into a **common embedding
space** and then combining them (typically via concatenation).

**Functionality:**

1.  Takes a *list* of input tensors (e.g.,
    `[dynamic_inputs, future_inputs]`). Each tensor must share the
    same batch and time dimensions (e.g., :math:`(B, T)`) but can
    have a different number of features (:math:`D_i`).
2.  For each input tensor (modality) in the list, it applies a
    separate :class:`~tf.keras.layers.Dense` layer to project that
    modality's features into a common target dimension specified by
    `embed_dim`. A non-linear activation (like ReLU) is often
    applied within or after this projection.
3.  Concatenates the resulting embeddings (each now having shape
    :math:`(B, T, \text{embed_dim})`) along the last (feature)
    dimension.
4.  The final output is a single tensor containing the combined
    embeddings, with shape
    :math:`(B, T, \text{num_modalities} \times \text{embed_dim})`.

**Usage Context:** Used in :class:`~fusionlab.nn.XTFT` to unify
different time-varying inputs, like selected dynamic past features
and selected known future covariates, into a single sequence
representation *before* applying positional encoding and subsequent
attention or recurrent layers. This ensures that features from
different sources are processed in a shared dimensional space.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import MultiModalEmbedding

   # Config
   batch_size = 4
   time_steps = 20
   embed_dim = 16 # Target embedding dimension per modality

   # Dummy input tensors (list) with different feature dimensions
   dynamic_features = tf.random.normal((batch_size, time_steps, 10)) # D1=10
   future_features = tf.random.normal((batch_size, time_steps, 5)) # D2=5

   # Instantiate the layer
   embedding_layer = MultiModalEmbedding(embed_dim=embed_dim)

   # Apply the layer to the list of inputs
   combined_embeddings = embedding_layer([dynamic_features, future_features])

   print(f"Input shapes: {[t.shape for t in [dynamic_features, future_features]]}")
   print(f"Output combined embedding shape: {combined_embeddings.shape}")
   # Expected: (4, 20, 2 * 16) = (4, 20, 32)

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Sequence Processing Layers
----------------------------

These layers process sequences to capture temporal dependencies or
patterns at different scales.

MultiScaleLSTM
~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MultiScaleLSTM`

**Purpose:** To analyze temporal patterns in a sequence at multiple
time resolutions simultaneously by applying parallel LSTM layers to
sub-sampled versions of the input. This allows the model to capture
both short-term and longer-term dynamics within the data.

**Functionality:**

1. Takes a single input time series tensor (shape :math:`(B, T, D)`).
2. Initializes multiple standard Keras LSTM layers, one for each
   `scale` factor provided (e.g., ``scales=[1, 3, 7]``). All LSTMs
   share the same `lstm_units`.
3. For each `scale` :math:`s`, it creates a sub-sampled version of
   the input sequence by taking every :math:`s`-th time step
   (e.g., ``input[:, ::s, :]``).
4. Feeds each sub-sampled sequence into its corresponding LSTM layer.
5. **Output Handling (controlled by `return_sequences`):**

   * If ``return_sequences=False``: Each LSTM returns only its final
     hidden state (shape :math:`(B, \text{lstm_units})`). These final
     states from all scales are concatenated along the feature
     dimension, yielding a single output tensor of shape
     :math:`(B, \text{lstm_units} \times \text{num_scales})`.
   * If `return_sequences=True`: Each LSTM returns its full output
     sequence. Since sub-sampling changes the sequence length, the
     result is a **list** of output tensors. Each tensor in the list
     has shape :math:`(B, T', \text{lstm_units})`, where the time
     dimension :math:`T'` depends on the corresponding scale
     (approximately :math:`T/s`).

**Usage Context:** Used within :class:`~fusionlab.nn.XTFT` to capture
dynamics occurring at different frequencies (e.g., daily patterns with
scale 1, weekly patterns with scale 7) from the dynamic input features.
The utility function :func:`~fusionlab.nn.components.aggregate_multiscale`
is often used subsequently to combine the outputs (especially when
`return_sequences=True`) before further processing.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import MultiScaleLSTM

   # Config
   batch_size = 4
   time_steps = 30
   features = 8
   lstm_units = 16
   scales = [1, 5, 10] # Analyze original, 5-step, 10-step patterns

   # Dummy input tensor
   dummy_input = tf.random.normal((batch_size, time_steps, features))

   # --- Example 1: Return only final states ---
   ms_lstm_final_state = MultiScaleLSTM(
       lstm_units=lstm_units,
       scales=scales,
       return_sequences=False # Default
   )
   final_states_concat = ms_lstm_final_state(dummy_input)
   print(f"Input shape: {dummy_input.shape}")
   print(f"Output shape (return_sequences=False): {final_states_concat.shape}")
   # Expected: (B, units * num_scales) -> (4, 16 * 3) = (4, 48)

   # --- Example 2: Return full sequences ---
   ms_lstm_sequences = MultiScaleLSTM(
       lstm_units=lstm_units,
       scales=scales,
       return_sequences=True
   )
   output_sequences_list = ms_lstm_sequences(dummy_input)
   print(f"\nOutput type (return_sequences=True): {type(output_sequences_list)}")
   print(f"Number of output sequences: {len(output_sequences_list)}")
   for i, seq in enumerate(output_sequences_list):
       print(f"  Shape of sequence for scale={scales[i]}: {seq.shape}")
   # Expected shapes (approx): (4, 30, 16), (4, 6, 16), (4, 3, 16)

DynamicTimeWindow
~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.DynamicTimeWindow`

**Purpose:** To select a fixed-size window containing only the most
recent time steps from an input sequence.

**Functionality:** This layer performs a simple slicing operation.
Given an input tensor representing a time series with :math:`T` steps
(shape :math:`(B, T, F)`), it returns only the last
`max_window_size` (:math:`W`) steps along the time dimension.

.. math::
   Output = Input[:, -W:, :]

If the input sequence length :math:`T` is less than or equal to
:math:`W`, the entire sequence is returned.

**Usage Context:** Used within the :class:`~fusionlab.nn.XTFT` model,
typically after attention fusion stages. It helps focus subsequent
decoding or output layers on the most recent temporal context,
which can be beneficial if long-range dependencies have already
been captured by other mechanisms (like LSTMs or memory attention)
and the final prediction relies more heavily on recent patterns.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   # Assuming DynamicTimeWindow is importable
   from fusionlab.nn.components import DynamicTimeWindow

   # Config
   batch_size = 4
   time_steps = 30
   features = 8
   window_size = 10 # Select last 10 steps

   # Dummy input tensor
   dummy_input = tf.random.normal((batch_size, time_steps, features))

   # Instantiate the layer
   time_window_layer = DynamicTimeWindow(max_window_size=window_size)

   # Apply the layer
   windowed_output = time_window_layer(dummy_input)

   print(f"Input shape: {dummy_input.shape}")
   print(f"Output windowed shape: {windowed_output.shape}")
   # Expected: (4, 10, 8)


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   


Attention Mechanisms
----------------------

Attention layers are a powerful tool in modern deep learning,
allowing models to dynamically weigh the importance of different
parts of the input when producing an output or representation.
Instead of treating all inputs equally, attention mechanisms learn
to focus on the most relevant information for the task at hand.
``fusionlab`` utilizes several specialized attention components,
often based on the core concepts described below.

**Core Concept: Scaled Dot-Product Attention**

The fundamental building block for many attention mechanisms is the
scaled dot-product attention [Vaswani17]_. It operates on three sets of
vectors: Queries (:math:`\mathbf{Q}`), Keys (:math:`\mathbf{K}`), and
Values (:math:`\mathbf{V}`).

1.  **Similarity Scoring:** The relevance or similarity between each
    Query vector and all Key vectors is computed using the dot
    product.
2.  **Scaling:** The scores are scaled down by dividing by the
    square root of the key dimension (:math:`d_k`) to stabilize
    gradients during training.
3.  **Weighting (Softmax):** A softmax function is applied to the
    scaled scores to obtain attention weights, which sum to 1. These
    weights indicate how much focus should be placed on each Value
    vector.
4.  **Weighted Sum:** The final output is the weighted sum of the
    Value vectors, using the computed attention weights.

The formula is:

.. math::
   Attention(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}

Here, :math:`\mathbf{Q} \in \mathbb{R}^{T_q \times d_q}`,
:math:`\mathbf{K} \in \mathbb{R}^{T_k \times d_k}`, and
:math:`\mathbf{V} \in \mathbb{R}^{T_v \times d_v}` (where
:math:`T_k = T_v` usually holds, and often :math:`d_q = d_k`).
The output has dimensions :math:`\mathbb{R}^{T_q \times d_v}`.

**Multi-Head Attention**

Instead of performing a single attention calculation, Multi-Head
Attention [Vaswani17]_ allows the model to jointly attend to information
from different representational subspaces at different positions.

1.  **Projection:** The original Queries, Keys, and Values are
    linearly projected :math:`h` times (where :math:`h` is the number
    of heads) using different, learned linear projections
    (:math:`\mathbf{W}^Q_i, \mathbf{W}^K_i, \mathbf{W}^V_i`
    for head :math:`i=1...h`).
2.  **Parallel Attention:** Scaled dot-product attention is applied
    in parallel to each of these projected versions, yielding :math:`h`
    different output vectors (:math:`head_i`).

    .. math::
       head_i = Attention(\mathbf{Q}\mathbf{W}^Q_i, \mathbf{K}\mathbf{W}^K_i, \mathbf{V}\mathbf{W}^V_i)

3.  **Concatenation:** The outputs from all heads are concatenated
    together.
4.  **Final Projection:** The concatenated output is passed through a
    final linear projection (:math:`\mathbf{W}^O`) to produce the
    final Multi-Head Attention output.

.. math::
   MultiHead(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(head_1, ..., head_h)\mathbf{W}^O

This allows each head to potentially focus on different aspects or
relationships within the data.

**Self-Attention vs. Cross-Attention**

* **Self-Attention:** When :math:`\mathbf{Q}, \mathbf{K}, \mathbf{V}`
  are all derived from the *same* input sequence (e.g., finding
  relationships within a single time series).
* **Cross-Attention:** When the Query comes from one sequence and the
  Keys/Values come from a *different* sequence (e.g., finding
  relationships between past inputs and future inputs, or between
  dynamic and static features).

The specific attention components provided by ``fusionlab`` build upon
or adapt these fundamental concepts for various purposes within time
series modeling.


ExplainableAttention
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.ExplainableAttention`

**Purpose:** To facilitate model interpretability by providing direct
access to the raw attention weights computed by a multi-head attention
mechanism, rather than the weighted values.

**Functionality:** This layer wraps the standard Keras
:class:`~tf.keras.layers.MultiHeadAttention`. However, its `call`
method is configured (by setting `return_attention_scores=True`
internally or during the call) to return only the computed
`attention_scores` tensor (typically shape
:math:`(B, H, T_q, T_k)`), where :math:`H` is `num_heads`,
:math:`T_q` is the query sequence length, and :math:`T_k` is the
key sequence length.

**Usage Context:** Primarily intended for model analysis, debugging,
and visualization during development. By examining the attention scores,
one can infer which parts of the key/value sequences the model focused
on for each element in the query sequence. It's generally *not* used
in the main predictive pathway of a deployed model because it doesn't
return the contextually weighted features needed for subsequent layers.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import ExplainableAttention

   # Config
   batch_size = 4
   query_seq_len = 10
   key_val_seq_len = 15
   units = 16
   num_heads = 2

   # Dummy Query, Key, Value tensors
   query = tf.random.normal((batch_size, query_seq_len, units))
   key = tf.random.normal((batch_size, key_val_seq_len, units))
   value = tf.random.normal((batch_size, key_val_seq_len, units))

   # Instantiate the layer
   explainable_attn_layer = ExplainableAttention(
       num_heads=num_heads,
       key_dim=units # key_dim usually matches units
   )

   # Apply the layer (implicitly returns only scores)
   # Note: Standard MHA requires return_attention_scores=True in call
   # Assuming this layer is hardcoded or configured to do so.
   attention_scores = explainable_attn_layer(query, value, key)

   print(f"Query shape: {query.shape}")
   print(f"Key/Value shape: {key.shape}")
   print(f"Output Attention Scores shape: {attention_scores.shape}")
   # Expected: (B, NumHeads, T_q, T_k) -> (4, 2, 10, 15)


CrossAttention
~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.CrossAttention`

**Purpose:** To model the interaction between two distinct input
sequences. It allows one sequence (the "query") to attend to another
sequence (the "key" and "value").

**Functionality:**

1. Takes a list of two tensors: `[source1, source2]`.
2. Applies separate dense layers to project each source to `units`.
3. Performs multi-head attention: `query=projected_source1`,
   `key=projected_source2`, `value=projected_source2`.
4. Returns the context vector representing `source2` information
   relevant to `source1`.

**Usage Context:** Useful for fusing information between different
modalities, like attending dynamic features to static context, or
future inputs to historical inputs. Used in :class:`~fusionlab.nn.XTFT`.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   # Assuming CrossAttention is importable
   from fusionlab.nn.components import CrossAttention

   # Config
   batch_size = 4
   seq_len_1 = 10
   seq_len_2 = 15
   features_1 = 8
   features_2 = 12
   units = 16 # Target dimension for attention
   num_heads = 2

   # Dummy input tensors
   source1 = tf.random.normal((batch_size, seq_len_1, features_1))
   source2 = tf.random.normal((batch_size, seq_len_2, features_2))

   # Instantiate
   cross_attn_layer = CrossAttention(units=units, num_heads=num_heads)

   # Apply layer (Input is a list)
   output_context = cross_attn_layer([source1, source2])

   print(f"Source 1 shape: {source1.shape}")
   print(f"Source 2 shape: {source2.shape}")
   print(f"Output Context shape: {output_context.shape}")
   # Expected: (B, T_q, U) -> (4, 10, 16)


TemporalAttentionLayer
~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.TemporalAttentionLayer`

**Purpose:** Implements the interpretable multi-head self-attention
block from the standard TFT architecture. It weights past time steps
based on relevance to the current step, conditioned by static context.

**Functionality:**

1. Takes temporal features (`inputs`, shape :math:`(B, T, U)`) and
   a static context vector (shape :math:`(B, U)`).
2. Transforms static context via a GRN, expands it across time, and
   adds it to `inputs` to form the `query`.
3. Applies :class:`~tf.keras.layers.MultiHeadAttention` using the
   generated `query`, with original `inputs` as `key` and `value`.
4. Applies dropout, a residual connection (`inputs` + attention output),
   and Layer Normalization.
5. Passes the result through a final position-wise GRN.

**Usage Context:** The core self-attention mechanism within the
standard :class:`~fusionlab.nn.TemporalFusionTransformer`.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import TemporalAttentionLayer

   # Config
   batch_size = 4
   time_steps = 10
   units = 16
   num_heads = 4

   # Dummy inputs
   temporal_input = tf.random.normal((batch_size, time_steps, units))
   static_context = tf.random.normal((batch_size, units))

   # Instantiate
   tal_layer = TemporalAttentionLayer(
       units=units, num_heads=num_heads, dropout_rate=0.1
       )

   # Apply layer
   output = tal_layer(temporal_input, context_vector=static_context)

   print(f"Input shape: {temporal_input.shape}")
   print(f"Context shape: {static_context.shape}")
   print(f"Output shape: {output.shape}")
   # Expected: (4, 10, 16)


MemoryAugmentedAttention
~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MemoryAugmentedAttention`

**Purpose:** Enhances attention by allowing the input sequence to
attend to an external, trainable `memory` matrix, potentially capturing
longer-range dependencies or learned prototypes.

**Functionality:**

1. Maintains an internal trainable `memory` matrix
   (shape :math:`(M, U)`, where :math:`M` = `memory_size`).
2. Input sequence (shape :math:`(B, T, U)`) serves as the `query`.
3. Multi-head attention uses the `query` to attend to the `memory`
   matrix (tiled across batch, acting as `key` and `value`).
4. The output context vector (derived from memory) is added
   residually to the original input sequence.

**Usage Context:** Used in :class:`~fusionlab.nn.XTFT` to integrate
a persistent learned context, potentially spanning longer horizons
than standard sequence processing.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import MemoryAugmentedAttention

   # Config
   batch_size = 4
   time_steps = 15
   units = 16
   num_heads = 2
   memory_size = 30 # Number of memory slots

   # Dummy input
   input_sequence = tf.random.normal((batch_size, time_steps, units))

   # Instantiate
   mem_attn_layer = MemoryAugmentedAttention(
       units=units, memory_size=memory_size, num_heads=num_heads
       )

   # Apply layer
   output = mem_attn_layer(input_sequence)

   print(f"Input shape: {input_sequence.shape}")
   print(f"Output shape: {output.shape}")
   # Expected: (4, 15, 16) (Shape is preserved by residual connection)


HierarchicalAttention
~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.HierarchicalAttention`

**Purpose:** Processes two related input sequences in parallel using
independent self-attention mechanisms, then combines their refined
representations (typically via addition).

**Functionality:**

1. Takes a list of two input tensors `[seq1, seq2]`, both usually
   shape :math:`(B, T, D)`.
2. Projects each sequence independently to `units` dimension.
3. Applies multi-head self-attention independently to `projected_seq1`.
4. Applies a separate multi-head self-attention independently to
   `projected_seq2`.
5. Adds the outputs of the two self-attention layers element-wise.

**Usage Context:** Used in :class:`~fusionlab.nn.XTFT` to model
potentially different temporal views (e.g., short vs. long term,
or different feature groups) in parallel before merging them.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   # Assuming HierarchicalAttention is importable
   from fusionlab.nn.components import HierarchicalAttention

   # Config
   batch_size = 4
   time_steps = 15
   features = 12 # Input features
   units = 16 # Target dimension
   num_heads = 2

   # Dummy inputs (list of two)
   input_seq1 = tf.random.normal((batch_size, time_steps, features))
   input_seq2 = tf.random.normal((batch_size, time_steps, features))

   # Instantiate
   hier_attn_layer = HierarchicalAttention(units=units, num_heads=num_heads)

   # Apply layer
   combined_output = hier_attn_layer([input_seq1, input_seq2])

   print(f"Input shapes: {[t.shape for t in [input_seq1, input_seq2]]}")
   print(f"Combined output shape: {combined_output.shape}")
   # Expected: (4, 15, 16)


MultiResolutionAttentionFusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`

**Purpose:** Fuses a combined feature tensor (potentially derived from
multiple sources or scales) using a standard multi-head self-attention
mechanism.

**Functionality:** Essentially a wrapper around Keras's
:class:`~tf.keras.layers.MultiHeadAttention` configured for
self-attention. It takes a single input tensor (e.g., shape
:math:`(B, T, F_{combined})`) and applies MHA where the input serves as
query, key, and value.

**Usage Context:** Used in :class:`~fusionlab.nn.XTFT` after combining
features from static context, multi-scale LSTMs, and other attention
layers. It allows the different feature streams within the combined
tensor to interact and be weighted before final processing stages.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   # Assuming MultiResolutionAttentionFusion is importable
   from fusionlab.nn.components import MultiResolutionAttentionFusion

   # Config
   batch_size = 4
   time_steps = 15
   combined_features = 64 # Dimension after concatenation
   units = 32 # Target dimension for attention/output
   num_heads = 4

   # Dummy combined features tensor
   fused_input = tf.random.normal((batch_size, time_steps, combined_features))

   # Instantiate
   fusion_attn_layer = MultiResolutionAttentionFusion(
       units=units, num_heads=num_heads
       )

   # Apply layer
   output = fusion_attn_layer(fused_input)

   print(f"Input shape: {fused_input.shape}")
   print(f"Output shape: {output.shape}")
   # Expected: (4, 15, 32)


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   

Output & Decoding Layers
--------------------------

These layers are typically used at the end of the model architecture
to transform the final feature representations into the desired
forecast format (point or quantile, across multiple horizons).

MultiDecoder
~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MultiDecoder`

**Purpose:** To generate multi-horizon forecasts where each future
time step (horizon) is predicted using its own dedicated set of
parameters (a separate dense layer), enabling step-specific predictions
from a shared context.

**Functionality:**

1. Takes a feature vector representing the aggregated context learned
   by the preceding parts of the model (typically shape
   :math:`(B, F)`, where :math:`B` is Batch, :math:`F` is Features).
2. Initializes a list of independent :class:`~tf.keras.layers.Dense`
   layers, one for each step in the forecast horizon (defined by
   `num_horizons`). Each dense layer maps the input features :math:`F`
   to the desired `output_dim` (:math:`O`).
3. Applies each horizon-specific Dense layer independently to the
   input feature vector.
4. Stacks the outputs from these layers along a new time (horizon)
   dimension to create the final output tensor of shape
   :math:`(B, H, O)`, where :math:`H` is `num_horizons`.

**Usage Context:** Employed in :class:`~fusionlab.nn.XTFT` after the
final feature aggregation step (e.g., after dynamic time windowing
and aggregation). It allows the model to learn different mappings
from the context vector to the prediction for each future step,
offering more flexibility than using a single shared output layer
across all horizons. The output :math:`(B, H, O)` is often fed into the
:class:`~fusionlab.nn.components.QuantileDistributionModeling` layer
if quantile outputs are needed.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import MultiDecoder

   # Config
   batch_size = 4
   features = 32 # Dimension of aggregated features
   num_horizons = 6 # Number of future steps to predict
   output_dim = 1 # Univariate forecast

   # Dummy input feature vector (e.g., output of final aggregation)
   aggregated_features = tf.random.normal((batch_size, features))

   # Instantiate the layer
   multi_decoder_layer = MultiDecoder(
       output_dim=output_dim,
       num_horizons=num_horizons
   )

   # Apply the layer
   horizon_outputs = multi_decoder_layer(aggregated_features)

   print(f"Input features shape: {aggregated_features.shape}")
   print(f"Output shape (multi-horizon): {horizon_outputs.shape}")
   # Expected: (B, H, O) -> (4, 6, 1)


QuantileDistributionModeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.QuantileDistributionModeling`

**Purpose:** To project the final feature representations generated
by the model's decoder stage into either point predictions or specific
quantile predictions, forming the final output tensor of the forecasting
model.

**Functionality:**

1. Takes the output features from a preceding layer (like
   :class:`~fusionlab.nn.components.MultiDecoder` or the final GRN in
   TFT), typically representing processed features for each forecast
   horizon step (shape :math:`(B, H, F)`).
2. **If quantiles were specified** during initialization (e.g.,
   ``[0.1, 0.5, 0.9]``):
   
   * It uses a separate :class:`~tf.keras.layers.Dense` layer for each quantile :math:`q`.
   * Each dense layer projects the input features :math:`F` to the
     target `output_dim` (:math:`O`).
   * The outputs for all quantiles are stacked along a new dimension,
     resulting in a shape of :math:`(B, H, Q, O)`, where :math:`Q`
     is the number of quantiles. (Note: If :math:`O=1`, the final
     output might be squeezed to :math:`(B, H, Q)` depending on model
     configuration or subsequent steps).
3. **If quantiles** is ``None``:

   * It uses a single :class:`~tf.keras.layers.Dense` layer.
   * This layer projects the input features :math:`F` to the target `output_dim` (:math:`O`).
   * The output shape is :math:`(B, H, O)`.

**Usage Context:** This is typically the **very last layer** in TFT and
XTFT architectures. It transforms the final internal representations
into the actual forecast values (either point estimates or specific
quantiles) that can be compared against ground truth using appropriate
loss functions (like MSE for point forecasts or quantile/pinball loss
like :class:`~fusionlab.nn.components.AdaptiveQuantileLoss` or
:func:`~fusionlab.nn.losses.combined_quantile_loss` for quantile
forecasts).

**Code Examples:**

*Example 1: Quantile Output*

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import QuantileDistributionModeling

   # Config
   batch_size = 4
   horizon = 6
   features = 32 # Features per horizon step
   output_dim = 1 # Univariate target
   quantiles = [0.1, 0.5, 0.9]

   # Dummy input (e.g., output of MultiDecoder)
   decoder_output = tf.random.normal((batch_size, horizon, features))

   # Instantiate for quantile output
   quantile_layer = QuantileDistributionModeling(
       quantiles=quantiles,
       output_dim=output_dim
   )

   # Apply the layer
   quantile_predictions = quantile_layer(decoder_output)

   print("--- Quantile Example ---")
   print(f"Input decoder features shape: {decoder_output.shape}")
   print(f"Quantile predictions shape: {quantile_predictions.shape}")
   # Expected: (B, H, Q, O) -> (4, 6, 3, 1)

*Example 2: Point Output*

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import QuantileDistributionModeling

   # Config
   batch_size = 4
   horizon = 6
   features = 32
   output_dim = 1

   # Dummy input
   decoder_output = tf.random.normal((batch_size, horizon, features))

   # Instantiate for point output (quantiles=None)
   point_layer = QuantileDistributionModeling(
       quantiles=None,
       output_dim=output_dim
   )

   # Apply the layer
   point_predictions = point_layer(decoder_output)

   print("\n--- Point Forecast Example ---")
   print(f"Input decoder features shape: {decoder_output.shape}")
   print(f"Point predictions shape: {point_predictions.shape}")
   # Expected: (B, H, O) -> (4, 6, 1)


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   

Loss Function Components
----------------------------

These components are specialized Keras Loss layers or related utilities
used for training the forecasting models, particularly for
probabilistic forecasting and incorporating anomaly detection
objectives.

AdaptiveQuantileLoss
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.AdaptiveQuantileLoss`

**Purpose:** To compute the quantile loss (also known as pinball
loss), which is essential for training models to produce quantile
forecasts. Predicting quantiles allows for estimating the
uncertainty around a point forecast.

**Functionality:** For a given quantile :math:`q`, the loss penalizes
prediction errors :math:`(y - \hat{y})` asymmetrically:

.. math::
   \text{Loss}_q(y, \hat{y}) =
   \begin{cases}
       q \cdot |y - \hat{y}| & \text{if } y \ge \hat{y} \\
       (1 - q) \cdot |y - \hat{y}| & \text{if } y < \hat{y}
   \end{cases}

This can also be written as
:math:`\max(q \cdot (y - \hat{y}),\, (q - 1) \cdot (y - \hat{y}))`.
The layer calculates this loss for each specified quantile in the
``quantiles`` list provided during initialization and averages the
result across all applicable dimensions (batch, horizon, quantiles,
output features).

**Usage Context:** This loss function (or the factory function
:func:`~fusionlab.nn.losses.combined_quantile_loss`) is typically
passed to `model.compile` when training models like
:class:`~fusionlab.nn.TemporalFusionTransformer` or
:class:`~fusionlab.nn.XTFT` that are configured to output quantile
predictions (i.e., when the ``quantiles`` parameter is set during
model initialization).

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import AdaptiveQuantileLoss

   # Config
   batch_size = 4
   horizon = 6
   quantiles = [0.1, 0.5, 0.9]
   num_quantiles = len(quantiles)

   # Dummy true values (B, H, 1) - usually single output dim
   y_true = tf.random.normal((batch_size, horizon, 1))
   # Dummy predicted quantiles (B, H, Q)
   y_pred_quantiles = tf.random.normal((batch_size, horizon, num_quantiles))

   # Instantiate the loss layer
   quantile_loss = AdaptiveQuantileLoss(quantiles=quantiles)

   # Calculate loss
   loss_value = quantile_loss(y_true, y_pred_quantiles)

   print(f"y_true shape: {y_true.shape}")
   print(f"y_pred shape: {y_pred_quantiles.shape}")
   print(f"Calculated Quantile Loss: {loss_value.numpy():.4f}")


AnomalyLoss
~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.AnomalyLoss`

**Purpose:** To provide a differentiable loss component based on
computed or provided anomaly scores. This allows models like
:class:`~fusionlab.nn.XTFT` to incorporate an auxiliary anomaly-related
objective during training.

**Functionality:** This Keras Loss layer calculates the mean of the
squared values of the input `anomaly_scores` tensor. The result is
then multiplied by a configurable `weight`. It expects the standard
Keras loss signature `call(y_true, y_pred)`, but typically uses only
one of the inputs (e.g., `y_true`) as the source of the anomaly scores.

.. math::
   \text{Loss}_{anomaly}(scores) = \text{weight} \cdot \text{mean}(\text{scores}^2)

The underlying assumption is that higher anomaly scores indicate greater
abnormality, and minimizing this loss (often alongside a primary task
loss) encourages the model to produce representations or predictions
associated with lower anomaly scores for the training data.

**Usage Context:** Primarily used as part of a combined loss strategy
within :class:`~fusionlab.nn.transformers.XTFT`, especially for `'feature_based'` or
`'from_config'` anomaly detection.
* In `'feature_based'`, the model might output scores internally, which
  are then fed to this loss (often added via `model.add_loss`).
* In `'from_config'`, pre-computed scores might be used with this loss
  layer within a :class:`~fusionlab.nn.components.MultiObjectiveLoss`
  or the :func:`~fusionlab.nn.losses.combined_total_loss` factory.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   # Assuming AnomalyLoss is importable
   from fusionlab.nn.components import AnomalyLoss

   # Config
   batch_size = 4
   horizon = 6
   anomaly_weight = 0.1

   # Dummy anomaly scores (e.g., pre-calculated or output by model)
   # Shape might vary, e.g., (B, H, 1) or (B, H) or (B, Features)
   dummy_scores = tf.random.uniform((batch_size, horizon, 1))

   # Instantiate the loss layer
   anomaly_loss_layer = AnomalyLoss(weight=anomaly_weight)

   # Calculate loss - Keras requires y_true, y_pred signature.
   # Pass scores as y_true and dummy zeros as y_pred (or vice-versa).
   loss_value = anomaly_loss_layer(dummy_scores, tf.zeros_like(dummy_scores))

   print(f"Input scores shape: {dummy_scores.shape}")
   print(f"Calculated Anomaly Loss: {loss_value.numpy():.4f}")


MultiObjectiveLoss
~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MultiObjectiveLoss`

**Purpose:** To combine multiple individual Keras loss function layers
into a single callable loss object, facilitating multi-task learning or
training with combined objectives (like forecasting + anomaly detection).

**Functionality:**

1.  Initialized with instances of other Keras loss layers, such as
    `quantile_loss_fn` (e.g., `AdaptiveQuantileLoss`) and
    `anomaly_loss_fn` (e.g., `AnomalyLoss`).
2.  Its `call(y_true, y_pred)` method internally calls the respective
    loss functions it holds.
3.  **Crucially**, it needs a mechanism to know how to map the potentially
    complex `y_true` and `y_pred` arguments to the inputs expected by
    each internal loss function. The provided description suggests it
    might expect `y_true` or `y_pred` to be tuples/dictionaries containing
    both target values and anomaly scores if used for combined quantile +
    anomaly loss. Alternatively, it might only compute the `quantile_loss`
    from `y_true`/`y_pred` and assume the anomaly loss is added separately
    via `model.add_loss` (driven by `anomaly_scores` computed elsewhere
    in the model's forward pass).

**Usage Context:** Intended for scenarios where a single optimization step
needs to minimize a weighted sum of different loss criteria. For example,
compiling an :class:`~fusionlab.nn.XTFT` model with this loss allows
jointly optimizing for quantile prediction accuracy and low anomaly scores
(if the data pipeline and `call` method correctly handle the multiple
targets/scores). *(Consult the API Reference and potentially source code
for the exact expected input format for `y_true` and `y_pred` when using
this layer).*

**Code Example (Instantiation):**

*(Note: Calling this loss requires careful setup of `y_true` and `y_pred`
or integration within a custom `train_step`, so only instantiation is shown)*

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import (
       MultiObjectiveLoss, AdaptiveQuantileLoss, AnomalyLoss
   )

   # Config
   quantiles = [0.1, 0.5, 0.9]
   anomaly_weight = 0.05

   # 1. Instantiate individual loss components
   quantile_loss_fn = AdaptiveQuantileLoss(quantiles=quantiles)
   anomaly_loss_fn = AnomalyLoss(weight=anomaly_weight)

   # 2. Instantiate the multi-objective loss
   multi_loss = MultiObjectiveLoss(
       quantile_loss_fn=quantile_loss_fn,
       anomaly_loss_fn=anomaly_loss_fn
   )

   print("MultiObjectiveLoss instantiated.")
   # To use: model.compile(optimizer='adam', loss=multi_loss)
   # Requires model's train_step or data pipeline to provide
   # compatible y_true / y_pred for both internal losses.


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   

Utility Functions
-----------------

These Python functions provide common aggregation or processing steps
used internally within model components or potentially useful for custom
model building.

aggregate_multiscale
~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.components.aggregate_multiscale`

**Purpose:** To combine the outputs from a
:class:`~fusionlab.nn.components.MultiScaleLSTM` layer into a single
tensor representation. This is necessary because `MultiScaleLSTM`
can produce a list of tensors (when `return_sequences=True`), each
potentially having a different length in the time dimension due to
different scaling factors.

**Functionality / Modes:**
The function takes the `lstm_output` (a list of 3D tensors
:math:`(B, T'_s, U)` where :math:`T'_s` can vary with scale :math:`s`)
and applies an aggregation strategy specified by the `mode`:

* **`'auto'` or `'last'` (Default):** Extracts the features from the
  *last time step* of each individual sequence in the input list
  and concatenates these feature vectors along the feature dimension.
  This is robust to varying sequence lengths (:math:`T'_s`) across scales.
  Output shape: :math:`(B, U \times N_{scales})`.
* **`'sum'`:** For each sequence in the input list, it sums the
  features across the time dimension (:math:`T'_s`). The resulting sum
  vectors (one per scale, shape :math:`(B, U)`) are then concatenated.
  Output shape: :math:`(B, U \times N_{scales})`.
* **`'average'`:** For each sequence in the input list, it averages
  the features across the time dimension (:math:`T'_s`). The resulting
  mean vectors are concatenated. Output shape:
  :math:`(B, U \times N_{scales})`.
* **`'concat'`:** *Requires all input sequences to have the **same**
  time dimension* (:math:`T'`). Concatenates the sequences along the
  feature dimension first (creating :math:`(B, T', U \times N_{scales})`),
  then takes only the features from the *last time step* of this
  combined tensor. Output shape: :math:`(B, U \times N_{scales})`.
* **`'flatten'`:** *Requires all input sequences to have the **same**
  time dimension* (:math:`T'`). Concatenates the sequences along the
  feature dimension first (creating :math:`(B, T', U \times N_{scales})`),
  then flattens the time and feature dimensions together. Output shape:
  :math:`(B, T' \times U \times N_{scales})`.

*(Refer to the function's docstring for more details).*

**Usage Context:** Used within :class:`~fusionlab.nn.XTFT` immediately
after the :class:`~fusionlab.nn.components.MultiScaleLSTM` layer (when
`return_sequences=True` is used) to aggregate its multi-resolution
outputs into a single feature vector suitable for combining with other
features before attention fusion. The default `'last'` mode is often
preferred for its robustness to varying sequence lengths.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import aggregate_multiscale

   # Config
   batch_size = 4
   units = 16
   num_scales = 3

   # Dummy MultiScaleLSTM output (list of tensors with different time steps)
   lstm_outputs_list = [
       tf.random.normal((batch_size, 20, units)), # Scale 1 (T'=20)
       tf.random.normal((batch_size, 10, units)), # Scale 2 (T'=10)
       tf.random.normal((batch_size, 5, units))   # Scale 3 (T'=5)
   ]
   print(f"Input is a list of {len(lstm_outputs_list)} tensors with shapes:")
   for i, t in enumerate(lstm_outputs_list):
       print(f"  Scale {i+1}: {t.shape}")

   # Aggregate using 'last' mode (default)
   agg_last = aggregate_multiscale(lstm_outputs_list, mode='last')
   print(f"\nOutput shape (mode='last'): {agg_last.shape}")
   # Expected: (B, U * N_scales) -> (4, 16 * 3) = (4, 48)

   # Aggregate using 'average' mode
   agg_avg = aggregate_multiscale(lstm_outputs_list, mode='average')
   print(f"Output shape (mode='average'): {agg_avg.shape}")
   # Expected: (B, U * N_scales) -> (4, 48)

   # Aggregate using 'sum' mode
   agg_sum = aggregate_multiscale(lstm_outputs_list, mode='sum')
   print(f"Output shape (mode='sum'): {agg_sum.shape}")
   # Expected: (B, U * N_scales) -> (4, 48)

   # --- Flatten/Concat require same time dimension ---
   time_steps_same = 10
   lstm_outputs_same_t = [
       tf.random.normal((batch_size, time_steps_same, units)),
       tf.random.normal((batch_size, time_steps_same, units)),
       tf.random.normal((batch_size, time_steps_same, units))
   ]
   print("\n--- Testing modes requiring same T' ---")
   print(f"Input tensors shape: {(batch_size, time_steps_same, units)}")

   # Aggregate using 'concat' mode
   agg_concat = aggregate_multiscale(lstm_outputs_same_t, mode='concat')
   print(f"Output shape (mode='concat'): {agg_concat.shape}")
   # Expected: (B, U * N_scales) -> (4, 48)

   # Aggregate using 'flatten' mode
   agg_flatten = aggregate_multiscale(lstm_outputs_same_t, mode='flatten')
   print(f"Output shape (mode='flatten'): {agg_flatten.shape}")
   # Expected: (B, T' * U * N_scales) -> (4, 10 * 16 * 3) = (4, 480)


aggregate_time_window_output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.components.aggregate_time_window_output`

**Purpose:** To perform a final aggregation step along the time
dimension of a sequence of features, typically after attention or
dynamic windowing, producing a single feature vector per item in the
batch that summarizes the temporal information.

**Functionality / Modes:**
Takes a 3D input tensor `time_window_output` with shape
:math:`(B, T, F)` (Batch, TimeSteps, Features) and applies an
aggregation method based on the `mode`:

* **`'last'`:** Selects only the feature vector from the very **last**
  time step (:math:`t=T`). Output shape: :math:`(B, F)`.
* **`'average'`:** Computes the mean of the feature vectors across
  the `TimeSteps` dimension (:math:`T`). Output shape: :math:`(B, F)`.
* **`'flatten'` (Default if `mode` is `None`):** Flattens the
  `TimeSteps` and `Features` dimensions together. Output shape:
  :math:`(B, T \times F)`.

**Usage Context:** Used within :class:`~fusionlab.nn.XTFT` after the
:class:`~fusionlab.nn.components.DynamicTimeWindow` layer (or other
sequence-producing layers like attention fusion). It collapses the
temporal dimension according to the chosen strategy, producing a
single context vector per batch item that summarizes the relevant
temporal information from the window. This aggregated vector is then
typically fed into the :class:`~fusionlab.nn.components.MultiDecoder`
for generating multi-horizon predictions.

**Code Example:**

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.components import aggregate_time_window_output

   # Config
   batch_size = 4
   time_steps = 10 # e.g., after dynamic windowing
   features = 32

   # Dummy input tensor (output from previous layer)
   dummy_input = tf.random.normal((batch_size, time_steps, features))
   print(f"Input shape: {dummy_input.shape}")

   # Aggregate using 'last' mode
   agg_last = aggregate_time_window_output(dummy_input, mode='last')
   print(f"\nOutput shape (mode='last'): {agg_last.shape}")
   # Expected: (B, F) -> (4, 32)

   # Aggregate using 'average' mode
   agg_avg = aggregate_time_window_output(dummy_input, mode='average')
   print(f"Output shape (mode='average'): {agg_avg.shape}")
   # Expected: (B, F) -> (4, 32)

   # Aggregate using 'flatten' mode (default)
   agg_flatten = aggregate_time_window_output(dummy_input, mode='flatten')
   print(f"Output shape (mode='flatten'): {agg_flatten.shape}")
   # Expected: (B, T * F) -> (4, 10 * 32) = (4, 320)

