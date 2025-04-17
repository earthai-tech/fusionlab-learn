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
~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.Activation`

This is a simple utility layer that wraps standard Keras activation
functions (like 'relu', 'elu', 'sigmoid'). Its primary purpose is
to ensure consistent handling and serialization of activation
functions within the ``fusionlab`` framework, particularly when
models or custom layers are saved and loaded.

While it can be used directly, users typically specify activations
as strings (e.g., ``activation='relu'``) in other layers
(like :class:`~fusionlab.nn.components.GatedResidualNetwork`),
which often handle the activation internally.

Positional Encoding
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.PositionalEncoding`

**Purpose:** To inject information about the relative or absolute
position of tokens in a sequence. This is crucial for models
like Transformers (and TFT/XTFT which use attention) because
standard self-attention mechanisms are permutation-invariant – they
don't inherently know the order of inputs.

**Functionality:** This layer adds a representation of the time
step index directly to the input feature embeddings at each
position.

.. math::
    Output_t = InputEmbed_t + PositionalInfo_t

The specific implementation here adds the actual numerical index
($0, 1, 2, ..., T-1$) to each feature dimension, broadcast across
the batch. Other forms of positional encoding (e.g., sinusoidal)
exist but this simple additive index is used here.

**Usage Context:** Applied to the sequence of temporal embeddings
(derived from dynamic past and future inputs) before they are fed
into attention layers or LSTMs in models like
:class:`~fusionlab.nn.TemporalFusionTransformer` and
:class:`~fusionlab.nn.XTFT`.

Gated Residual Network (GRN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.GatedResidualNetwork`

**Purpose:** The GRN is arguably one of the most fundamental
building blocks in TFT and related architectures. It provides a
flexible way to apply non-linear transformations to inputs,
optionally conditioned by context, while incorporating gating and
residual connections for stable training of deep networks.

**Functionality:** A GRN typically involves these steps:

1.  **(Optional) Context Addition:** If context information is
    provided, it's linearly transformed and added to the primary
    input.
2.  **Non-linear Transformation:** The (potentially contextualized)
    input goes through two dense layers with an activation function
    (e.g., ELU, ReLU). Dropout and Batch Normalization can be
    optionally applied.
3.  **Gating Mechanism:** A separate dense layer with a sigmoid
    activation calculates a "gate". The output of the non-linear
    transformation is then element-wise multiplied by this gate. This
    allows the network to dynamically control the flow of information.
4.  **Residual Connection:** The gated output is added back to the
    original input (or a linearly projected version of it if
    dimensions need matching).
5.  **Layer Normalization:** The final result is normalized using
    Layer Normalization.

.. math::
   GRN(a, [c]) = LN(a' + GLU(Layer_1(act(Layer_0(a'))), Layer_2(a')))

*(See the API reference or the TFT mathematical formulation for the
precise definition of terms)*

**Usage Context:** GRNs are used extensively throughout TFT and XTFT:
    * Processing static context features.
    * Applying transformations within Variable Selection Networks.
    * Processing outputs of attention layers (position-wise feed-forward).
    * Static enrichment of temporal features.


These components represent key architectural layers used within the
TFT and XTFT models to process and transform features.

StaticEnrichmentLayer
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.StaticEnrichmentLayer`

**Purpose:** To effectively infuse time-invariant static context
into time-varying temporal features. This allows the model's
processing of temporal patterns (e.g., seasonality, trends learned
by an LSTM) to be conditioned by static attributes (e.g., sensor
location, product category).

**Functionality:**
1. Takes a *static context vector* (typically derived from static
   inputs via GRNs, shape `(Batch, Units)`) and *temporal features*
   (often the output of an LSTM, shape `(Batch, TimeSteps, Units)`).
2. Expands and tiles the static context vector along the time
   dimension to match the temporal features' shape
   (`(Batch, TimeSteps, Units)`).
3. Concatenates the tiled static context and the original temporal
   features along the feature dimension.
4. Passes this combined tensor through an internal
   :class:`~fusionlab.nn.components.GatedResidualNetwork` (GRN) for
   non-linear transformation and gating, producing the enriched
   temporal features.

**Usage Context:** A standard component in TFT architectures, typically
applied after the sequence encoder (like an LSTM) and before the
main temporal attention layer.

Input Processing & Embedding Layers
-------------------------------------

These layers handle the initial transformation and embedding of
various input types.

LearnedNormalization
~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.LearnedNormalization`

**Purpose:** To normalize input features using scaling parameters
(mean and standard deviation) that are learned during model training,
rather than being pre-calculated from the dataset statistics.

**Functionality:**
1. Maintains two trainable weight vectors: `mean` and `stddev`, with
   a size equal to the number of input features (last dimension of
   the input tensor).
2. During the forward pass, it applies the standard normalization
   formula to the input tensor `x`:

   .. math::
      x_{norm} = \frac{x - \mu_{learned}}{\sigma_{learned} + \epsilon}

   where $\mu_{learned}$ and $\sigma_{learned}$ are the learned mean
   and standard deviation weights, and $\epsilon$ is a small constant
   (e.g., 1e-6) added for numerical stability.

**Usage Context:** Used in the :class:`~fusionlab.nn.XTFT` model as an
initial processing step, typically applied to static inputs. This
allows the model to adaptively determine the appropriate normalization
for these features based on the data distribution encountered during
training, potentially offering more flexibility than fixed
pre-processing normalization.

MultiModalEmbedding
~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MultiModalEmbedding`

**Purpose:** To process multiple input sequences (modalities), which
may have different feature dimensions initially, by projecting each
into a common embedding space and then combining them.

**Functionality:**
1. Takes a *list* of input tensors (e.g., `[dynamic_inputs, future_inputs]`).
   Each tensor must share the same batch and time dimensions
   (`B`, `T`) but can have a different number of features (`D_i`).
2. For each input tensor (modality) in the list, it applies a
   separate Dense layer to project that modality's features into
   a common target dimension specified by `embed_dim`. A ReLU
   activation is typically applied.
3. Concatenates the resulting embeddings (each now having shape
   `(B, T, embed_dim)`) along the last (feature) dimension.
4. The final output is a single tensor containing the combined
   embeddings, with shape `(B, T, num_modalities * embed_dim)`.

**Usage Context:** Used in :class:`~fusionlab.nn.XTFT` to unify
different time-varying inputs, like dynamic past features and known
future covariates, into a single sequence representation before
applying positional encoding and subsequent attention or recurrent
layers.


Sequence Processing Layers
----------------------------

These layers process sequences to capture temporal dependencies or
patterns at different scales.

MultiScaleLSTM
~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MultiScaleLSTM`

**Purpose:** To analyze temporal patterns in a sequence at multiple
time resolutions simultaneously by applying parallel LSTM layers to
sub-sampled versions of the input.

**Functionality:**
1. Takes a single input time series tensor (shape `(B, T, D)`).
2. Initializes multiple standard Keras LSTM layers, one for each
   `scale` factor provided (e.g., `scales=[1, 3, 7]`).
3. For each `scale`, it creates a sub-sampled version of the input
   sequence by taking every `scale`-th time step
   (`input[:, ::scale, :]`).
4. Feeds each sub-sampled sequence into its corresponding LSTM layer.
5. **Output Handling (controlled by `return_sequences`):**
   * If `return_sequences=False`: Each LSTM returns only its final
     hidden state (shape `(B, lstm_units)`). These final states from
     all scales are concatenated along the feature dimension, yielding
     a single output tensor of shape `(B, lstm_units * num_scales)`.
   * If `return_sequences=True`: Each LSTM returns its full output
     sequence. Since sub-sampling changes the sequence length, the
     result is a *list* of output tensors, where each tensor has shape
     `(B, T', lstm_units)` and `T'` depends on the corresponding scale.

**Usage Context:** Used within :class:`~fusionlab.nn.XTFT` to capture
dynamics occurring at different frequencies (e.g., daily patterns with
scale 1, weekly patterns with scale 7) from the dynamic input features.
The utility function :func:`~fusionlab.nn.components.aggregate_multiscale`
is often used subsequently to combine the outputs if needed.


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
scaled dot-product attention [1]_. It operates on three sets of
vectors: Queries ($Q$), Keys ($K$), and Values ($V$).

1.  **Similarity Scoring:** The relevance or similarity between each
    Query vector and all Key vectors is computed using the dot
    product.
2.  **Scaling:** The scores are scaled down by dividing by the
    square root of the key dimension ($d_k$) to stabilize gradients
    during training.
3.  **Weighting (Softmax):** A softmax function is applied to the
    scaled scores to obtain attention weights, which sum to 1. These
    weights indicate how much focus should be placed on each Value
    vector.
4.  **Weighted Sum:** The final output is the weighted sum of the
    Value vectors, using the computed attention weights.

The formula is:

.. math::
   Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

Here, $Q \in \mathbb{R}^{T_q \times d_q}$,
$K \in \mathbb{R}^{T_k \times d_k}$, and
$V \in \mathbb{R}^{T_v \times d_v}$ (where $T_k = T_v$ usually holds).
The output has dimensions $\mathbb{R}^{T_q \times d_v}$.

**Multi-Head Attention**

Instead of performing a single attention calculation, Multi-Head
Attention [1]_ allows the model to jointly attend to information
from different representational subspaces at different positions.

1.  **Projection:** The original Queries, Keys, and Values are
    linearly projected $h$ times (where $h$ is the number of heads)
    using different, learned linear projections ($W^Q_i, W^K_i, W^V_i$
    for head $i=1...h$).
2.  **Parallel Attention:** Scaled dot-product attention is applied
    in parallel to each of these projected versions, yielding $h$
    different output vectors ($head_i$).

   .. math::
      head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

3.  **Concatenation:** The outputs from all heads are concatenated
    together.
4.  **Final Projection:** The concatenated output is passed through a
    final linear projection ($W^O$) to produce the final Multi-Head
    Attention output.

.. math::
   MultiHead(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O

This allows each head to potentially focus on different aspects or
relationships within the data.

**Self-Attention vs. Cross-Attention**

* **Self-Attention:** When $Q, K, V$ are all derived from the *same*
    input sequence (e.g., finding relationships within a single time
    series).
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
access to the attention weights computed by a multi-head attention
mechanism.

**Functionality:** This layer wraps the standard Keras
:class:`~tf.keras.layers.MultiHeadAttention`. However, instead of
returning the weighted sum of values (the context vector), its `call`
method is configured to return only the computed `attention_scores`
tensor (typically shape `(Batch, NumHeads, TimeSteps, TimeSteps)`).

**Usage Context:** Primarily intended for model analysis, debugging,
and visualization. By examining the attention scores, one can infer
which parts of the input sequence(s) the model focused on when
making predictions or generating representations. It's generally not
used in the main predictive pathway of a deployed model unless
runtime interpretability is specifically required.

CrossAttention
~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.CrossAttention`

**Purpose:** To model the interaction between two distinct input
sequences. It allows one sequence (the "query") to attend to another
sequence (the "key" and "value"), effectively asking: "Based on
sequence 1, what information from sequence 2 is most relevant?"

**Functionality:**
1. Takes a list containing two input tensors, `source1` (query) and
   `source2` (key/value), typically of shape `(B, T, D)`.
2. Applies separate dense layers to project `source1` and `source2`
   to the required internal dimensionality (`units`).
3. Performs multi-head attention where the projected `source1` acts
   as the query, and the projected `source2` acts as both the key
   and the value.
4. Returns the resulting context vector, which represents information
   from `source2` weighted according to its relevance to `source1`.

**Usage Context:** Useful in scenarios involving multiple input
modalities or feature sets. For example, attending dynamic features
to static features, or attending known future inputs to historical
inputs. Used within the :class:`~fusionlab.nn.XTFT` model.

TemporalAttentionLayer
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.TemporalAttentionLayer`

**Purpose:** Implements the interpretable multi-head self-attention
mechanism specific to the Temporal Fusion Transformer. It allows the
model to weigh the importance of different past time steps when
processing information at a given time step, while also being
conditioned by static context.

**Functionality:**
1. Takes *temporal features* (`inputs`, shape `(B, T, U)`) and a
   *static context vector* (shape `(B, U)`).
2. Transforms the static context using a GRN, expands it to match
   the time dimension, and adds it to the `inputs`. This result
   forms the `query` for the attention mechanism.
3. Applies standard :class:`~tf.keras.layers.MultiHeadAttention`
   using the generated `query`, with `inputs` serving as both the
   `key` and `value`. This calculates attention scores over the
   temporal sequence.
4. Applies dropout to the attention output.
5. Adds the attention output back to the original `inputs` (residual
   connection) and applies Layer Normalization.
6. Passes the result through a final GRN for further processing.

**Usage Context:** This is the core self-attention block used within
the :class:`~fusionlab.nn.TemporalFusionTransformer` model to capture
temporal dependencies and enable interpretability via attention
weights.

MemoryAugmentedAttention
~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MemoryAugmentedAttention`

**Purpose:** To enhance the model's ability to capture long-range
dependencies or access a persistent, learned context by incorporating
an external, trainable memory matrix into the attention process.
Inspired by concepts like Neural Turing Machines.

**Functionality:**
1. Maintains an internal, trainable `memory` matrix of shape
   `(memory_size, units)`.
2. During the forward pass, the input sequence serves as the `query`.
3. Multi-head attention is computed where the `query` attends to the
   `memory` matrix (which is tiled across the batch dimension and
   acts as both `key` and `value`).
4. The resulting attention output (context vector derived from the
   memory) is added residually back to the original input sequence.

**Usage Context:** Employed in :class:`~fusionlab.nn.XTFT` to provide
a mechanism for integrating information potentially spanning longer
time horizons than what might be captured by standard recurrent layers
or self-attention over the input sequence alone.


HierarchicalAttention
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.HierarchicalAttention`

**Purpose:** To process two related input sequences (conceptually
representing different views, like short-term vs. long-term dynamics,
or different feature groups) independently using self-attention, and
then combine their processed representations.

**Functionality:**
1. Takes a list containing two input tensors, conceptually
   `short_term` and `long_term`, each typically of shape
   `(B, T, D)`.
2. Applies separate dense layers to project `short_term` and
   `long_term` sequences individually to the target `units`
   dimension.
3. Applies a multi-head self-attention mechanism independently
   to the projected `short_term` sequence (query, key, and value
   are all derived from `short_term`).
4. Similarly, applies a separate multi-head self-attention
   mechanism independently to the projected `long_term` sequence.
5. Adds the outputs of the two independent self-attention layers
   element-wise to produce the final combined output tensor.

**Usage Context:** This layer allows the model to capture temporal
patterns within two potentially distinct but related sequences in
parallel before merging them. It differs from cross-attention, which
models direct interactions *between* sequences. It's used within the
:class:`~fusionlab.nn.XTFT` architecture to handle complex temporal
interactions.


MultiResolutionAttentionFusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`

**Purpose:** To fuse a combined set of features, potentially derived
from different sources or representing different temporal resolutions,
using a standard self-attention mechanism.

**Functionality:** This layer is a direct application of Keras's
:class:`~tf.keras.layers.MultiHeadAttention`. It takes a single input
tensor (which is often the result of concatenating features from
various previous layers) and performs self-attention, where the input
serves as the query, key, and value. This allows different elements
within the combined feature representation to interact and weigh each
other's importance.

**Usage Context:** Used within :class:`~fusionlab.nn.XTFT` at a later
stage in the network, after features from static inputs, multi-scale
LSTMs, and other attention mechanisms have been computed and
concatenated. This layer serves to integrate these diverse feature
streams into a unified representation before further processing like
dynamic time windowing or decoding.

DynamicTimeWindow
~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.DynamicTimeWindow`

**Purpose:** To select a fixed-size window containing only the most
recent time steps from an input sequence.

**Functionality:** This layer performs a simple slicing operation.
Given an input tensor representing a time series with $T$ steps
(shape `(Batch, TimeSteps, Features)`), it returns only the last
`max_window_size` steps along the time dimension.

.. math::
   Output = Input[:, -W:, :]

where $W$ is the specified `max_window_size`. If the input sequence
length $T$ is less than or equal to $W$, the entire sequence is
returned.

**Usage Context:** Used within the :class:`~fusionlab.nn.XTFT` model,
typically after attention fusion stages. It helps focus subsequent
decoding or output layers on the most recent temporal context,
which can be beneficial if long-range dependencies have already
been captured by other mechanisms (like LSTMs or memory attention)
and the final prediction relies more heavily on recent patterns.

Output & Decoding Layers
--------------------------

These layers are typically used at the end of the model architecture
to transform the final feature representations into the desired
forecast format (point or quantile, across multiple horizons).

MultiDecoder
~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MultiDecoder`

**Purpose:** To generate multi-horizon forecasts where each future
time step (horizon) is predicted using its own dedicated set of
parameters (a separate dense layer).

**Functionality:**
1. Takes a feature vector representing the aggregated context learned
   by the preceding parts of the model (typically shape
   `(Batch, Features)`).
2. Initializes a list of independent Dense layers, one for each step
   in the forecast horizon (defined by `num_horizons`). Each dense
   layer maps the input features to the desired `output_dim`.
3. Applies each horizon-specific Dense layer to the input feature
   vector.
4. Stacks the outputs from these layers along a new dimension to
   create the final output tensor of shape
   `(Batch, NumHorizons, OutputDim)`.

**Usage Context:** Employed in :class:`~fusionlab.nn.XTFT` after the
final feature aggregation step (e.g., after dynamic time windowing
and aggregation). It allows the model to learn different mappings
from the context vector to the prediction for each future step,
offering more flexibility than using a single shared output layer
across all horizons. The output is often fed into the
:class:`~fusionlab.nn.components.QuantileDistributionModeling` layer.

QuantileDistributionModeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.QuantileDistributionModeling`

**Purpose:** To project the final feature representations generated
by the model's decoder stage into either point predictions or specific
quantile predictions, forming the final output of the forecasting model.

**Functionality:**
1. Takes the output from a preceding layer (like
   :class:`~fusionlab.nn.components.MultiDecoder`), typically representing
   features for each forecast horizon step (shape `(B, H, F)`).
2. **If `quantiles` were specified** during initialization (e.g.,
   `[0.1, 0.5, 0.9]`):
   * It uses a separate Dense layer for each quantile $q$.
   * Each dense layer projects the input features to the target
     `output_dim`.
   * The outputs for all quantiles are stacked along a new dimension,
     resulting in a shape of `(Batch, Horizon, NumQuantiles, OutputDim)`.
3. **If `quantiles` is `None`:**
   * It uses a single Dense layer.
   * This layer projects the input features to the target `output_dim`.
   * The output shape is `(Batch, Horizon, OutputDim)`.

**Usage Context:** This is typically the very last layer in TFT and
XTFT architectures. It transforms the final internal representations
into the actual forecast values that can be compared against ground
truth using appropriate loss functions (like MSE for point forecasts
or :class:`~fusionlab.nn.components.AdaptiveQuantileLoss` for quantile
forecasts).


Loss Function Components
--------------------------

These components are specialized Keras Loss layers used for training
the forecasting models, particularly for probabilistic forecasting
and incorporating anomaly detection objectives.

AdaptiveQuantileLoss
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.AdaptiveQuantileLoss`

**Purpose:** To compute the quantile loss (also known as pinball
loss), which is essential for training models to produce quantile
forecasts. Predicting quantiles allows for estimating the
uncertainty around a point forecast.

**Functionality:** For a given quantile $q$, the loss penalizes
prediction errors $(y - \hat{y})$ asymmetrically:

.. math::
   \text{Loss}_q(y, \hat{y}) =
   \begin{cases}
       q \cdot |y - \hat{y}| & \text{if } y \ge \hat{y} \\
       (1 - q) \cdot |y - \hat{y}| & \text{if } y < \hat{y}
   \end{cases}

This can also be written as $\max(q \cdot (y - \hat{y}),\, (q - 1) \cdot (y - \hat{y}))$.
The layer calculates this loss for each specified quantile in the
``quantiles`` list provided during initialization and averages the
result across all dimensions (batch, horizon, quantiles, output).

**Usage Context:** This loss function is typically used when
compiling a model (like :class:`~fusionlab.nn.TemporalFusionTransformer`
or :class:`~fusionlab.nn.XTFT`) that is configured to output
quantile predictions (i.e., when the ``quantiles`` parameter is
set during model initialization).

AnomalyLoss
~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.AnomalyLoss`

**Purpose:** To provide a loss signal based on computed or provided
anomaly scores. This allows models like :class:`~fusionlab.nn.XTFT`
to be trained with an auxiliary objective related to anomaly
detection alongside the primary forecasting task.

**Functionality:** This layer calculates the mean of the squared
values of the input `anomaly_scores` tensor and multiplies the result
by a configurable `weight`.

.. math::
   \text{Loss}_{anomaly} = \text{weight} \cdot \text{mean}(\text{anomaly\_scores}^2)

The assumption is that higher anomaly scores indicate a greater
likelihood of an anomaly, and minimizing this loss (when combined
with other objectives) encourages the model to either predict low
anomaly scores or internal representations that lead to low scores.

**Usage Context:** Used internally by :class:`~fusionlab.nn.XTFT` when
``anomaly_detection_strategy`` is set to ``'feature_based'`` or
``'from_config'``. The calculated loss is typically added to the
main model loss via ``model.add_loss`` during the forward pass if
anomaly scores are available. It can also be used as part of a
:class:`~fusionlab.nn.components.MultiObjectiveLoss`.

MultiObjectiveLoss
~~~~~~~~~~~~~~~~~~~~
:API Reference: :class:`~fusionlab.nn.components.MultiObjectiveLoss`

**Purpose:** To combine multiple individual loss functions into a
single objective, facilitating multi-task learning. This
implementation specifically combines a quantile forecasting loss
with an anomaly detection loss.

**Functionality:** This layer takes two pre-initialized loss function
layers as input during its own initialization:
`quantile_loss_fn` (e.g., an instance of `AdaptiveQuantileLoss`) and
`anomaly_loss_fn` (e.g., an instance of `AnomalyLoss`).

During the `call` method, it computes:
1. The quantile loss using `y_true` and `y_pred`.
2. The anomaly loss using `anomaly_scores` (if provided).

The total loss returned is the sum of these two components. If
`anomaly_scores` are not provided to the `call` method, only the
quantile loss is returned.

**Usage Context:** This loss function can be passed to `model.compile`
when training a model like :class:`~fusionlab.nn.XTFT` configured for
both quantile forecasting and anomaly detection (using the
`'from_config'` or potentially `'feature_based'` strategies where
anomaly scores are explicitly handled). It allows the optimizer to
jointly minimize both forecasting error and anomaly scores. *(Note:
The code comments suggest this specific multi-objective combination
might be subject to change in future versions).*


Utility Functions
-------------------

These functions provide common aggregation or processing steps used
within the model components.

aggregate_multiscale
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.components.aggregate_multiscale`

**Purpose:** To combine the outputs from a
:class:`~fusionlab.nn.components.MultiScaleLSTM` layer into a single
tensor representation. This is necessary because `MultiScaleLSTM`
can produce a list of tensors (when `return_sequences=True`), each
potentially having a different length in the time dimension due to
different scaling factors.

**Functionality / Modes:**
The function takes the `lstm_output` (either a list of tensors or
potentially a single tensor if `return_sequences=False` was used)
and applies an aggregation strategy specified by the `mode`
parameter:

* **`'auto'` or `'last'` (Default):** Extracts the features from the
    *last time step* of each individual sequence in the input list
    and concatenates these feature vectors. This is robust to
    varying sequence lengths ($T'$) across scales. Output shape:
    `(Batch, LSTMUnits * NumScales)`.
* **`'sum'`:** For each sequence in the input list, it sums the
    features across the time dimension. The resulting sum vectors
    (one per scale) are then concatenated. Output shape:
    `(Batch, LSTMUnits * NumScales)`.
* **`'average'`:** For each sequence in the input list, it averages
    the features across the time dimension. The resulting mean
    vectors are concatenated. Output shape:
    `(Batch, LSTMUnits * NumScales)`.
* **`'concat'`:** *Requires all input sequences to have the same
    time dimension ($T'$)*. Concatenates the sequences along the
    feature dimension first (creating `(B, T', U*S)`), then takes
    only the features from the *last time step* of this combined
    tensor. Output shape: `(Batch, LSTMUnits * NumScales)`.
* **`'flatten'`:** *Requires all input sequences to have the same
    time dimension ($T'$)*. Concatenates the sequences along the
    feature dimension first (creating `(B, T', U*S)`), then flattens
    the time and feature dimensions together. Output shape:
    `(Batch, T' * LSTMUnits * NumScales)`.

*(Refer to the function's docstring for comparison tables and precise
mathematical formulations).*

**Usage Context:** Used within :class:`~fusionlab.nn.XTFT` immediately
after the `MultiScaleLSTM` layer to aggregate its potentially
multi-resolution outputs into a single tensor suitable for combining
with other features before attention fusion.

aggregate_time_window_output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.components.aggregate_time_window_output`

**Purpose:** To perform a final aggregation step along the time
dimension of a sequence of features, typically after attention or
dynamic windowing, producing a single feature vector per item in the
batch.

**Functionality / Modes:**
Takes a 3D input tensor `time_window_output` with shape
`(Batch, TimeSteps, Features)` and applies an aggregation method
based on the `mode`:

* **`'last'`:** Selects only the feature vector from the very last
    time step. Output shape: `(Batch, Features)`.
* **`'average'`:** Computes the mean of the feature vectors across
    the `TimeSteps` dimension. Output shape: `(Batch, Features)`.
* **`'flatten'` (Default if `mode` is `None`):** Flattens the
    `TimeSteps` and `Features` dimensions together. Output shape:
    `(Batch, TimeSteps * Features)`.

**Usage Context:** Used within :class:`~fusionlab.nn.XTFT` after the
:class:`~fusionlab.nn.components.DynamicTimeWindow` layer. It collapses
the temporal dimension according to the chosen strategy, producing a
single context vector per batch item that summarizes the relevant
temporal information. This aggregated vector is then typically fed
into the :class:`~fusionlab.nn.components.MultiDecoder` for generating
multi-horizon predictions.

