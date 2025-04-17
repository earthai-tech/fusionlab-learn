.. _user_guide_models:

=================
Available Models
=================

``fusionlab`` provides several implementations of advanced time
series forecasting models. This section details the main model
classes available, their characteristics, and typical use cases.

.. raw:: html

    <hr>
    
TemporalFusionTransformer
--------------------------

:API Reference: :class:`~fusionlab.nn.TemporalFusionTransformer`

The ``TemporalFusionTransformer`` class is the primary, flexible
implementation of the Temporal Fusion Transformer architecture [1]_
within ``fusionlab``. It is designed to handle a variety of input
configurations and forecasting tasks.

**Key Features:**

* **Flexible Inputs:** Supports dynamic (past) inputs (required),
    optional static inputs (time-invariant metadata), and optional
    future inputs (known exogenous covariates).
* **Multi-Horizon Forecasting:** Directly outputs predictions for
    multiple future time steps defined by the ``forecast_horizon``
    parameter.
* **Probabilistic Forecasts:** Can generate quantile forecasts by
    specifying the ``quantiles`` parameter (e.g., ``[0.1, 0.5, 0.9]``)
    to estimate prediction intervals and uncertainty. If
    ``quantiles`` is ``None``, it produces deterministic (point)
    forecasts.
* **Standard TFT Components:** Built using core TFT blocks like
    Variable Selection Networks (optional for static/future),
    LSTM encoders, Static Enrichment, Temporal Self-Attention,
    and Gated Residual Networks.

**When to Use:**

This is generally the recommended starting point for applying the
TFT architecture. Use it when:

* You need a standard, well-understood TFT implementation.
* You have various combinations of dynamic, static, or future
    inputs.
* You require either point forecasts or probabilistic (quantile)
    forecasts.


Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we describe the core mathematical concepts behind the
Temporal Fusion Transformer, following the architecture outlined
in the original paper [1]_. This provides insight into how different
inputs are processed and transformed to generate forecasts.

**Notation:**

We define the following notation for clarity:

* **Inputs:**
    * $s \in \mathbb{R}^{d_s}$: Static (time-invariant) covariates.
    * $z_t \in \mathbb{R}^{d_z}$: Known future inputs at time $t$.
    * $x_t \in \mathbb{R}^{d_x}$: Observed past dynamic inputs at time $t$.
    * $y_t \in \mathbb{R}^{d_y}$: Past target variable(s) at time $t$
        (often included in $x_t$).
* **Time Indices:**
    * $t \in [T-k+1, T]$: Past time steps within the lookback
        window of size $k$.
    * $t \in [T+1, T+\tau]$: Future time steps for the forecast
        horizon $\tau$.
* **Dimensions:**
    * $d_s, d_x, d_z, d_y$: Dimensionalities of respective inputs.
    * $d_{model}$: The main hidden state dimension of the model
        (e.g., ``hidden_units``).
* **Common Functions:**
    * $LN(\cdot)$: Layer Normalization.
    * $\sigma(\cdot)$: Sigmoid activation function.
    * $ReLU(\cdot), ELU(\cdot)$: Rectified Linear Unit / Exponential
        Linear Unit activation.
    * $Linear(\cdot)$: A standard dense (fully-connected) layer.
    * $GLU(a, b) = a \odot \sigma(b)$: Gated Linear Unit, where
        $\odot$ is element-wise multiplication.
    * $GRN(a, [c])$: Gated Residual Network. A key block defined as:
      $GRN(a, c) = LN(a' + GLU(Linear_1(act(Linear_0(a'))), Linear_2(a')))$,
      where $a' = a+Linear_c(c)$ if context $c$ is provided, else $a'=a$,
      and $act$ is typically $ELU$ or $ReLU$. The GRN applies gating,
      non-linearity, and a residual connection.

**Architectural Flow:**

1.  **Input Transformations & Variable Selection:**
    Different types of inputs (categorical, continuous) are first
    transformed into numerical vectors (e.g., via embeddings for
    categorical, linear layers for continuous). A crucial step is
    applying **Variable Selection Networks (VSNs)** to each input type
    (static $s$, past dynamic $x_t$, known future $z_t$).

    * A VSN takes the transformed input features $\mathbf{\chi} = [\chi^1, ..., \chi^N]$
      and an optional static context vector $c$.
    * It computes feature-wise weights $\alpha_\chi = Softmax(GRN(Linear(\mathbf{\chi}), c))$.
    * It applies feature-wise GRNs: $\tilde{\chi}^j = GRN(\chi^j)$.
    * The output embedding is a weighted sum: $\xi = \sum_{j=1}^N \alpha_\chi^j \tilde{\chi}^j$.

    This process yields embeddings for static features ($\zeta$),
    past dynamic features ($\xi_t$ for $t \le T$), and known future
    features ($\xi_t$ for $t > T$).

2.  **Static Covariate Encoders:**
    The static VSN output $\zeta$ is processed through dedicated GRNs
    to produce four context vectors, which condition different parts
    of the temporal processing:
    * $c_s$: Static context for temporal variable selection.
    * $c_e$: Static context for static enrichment.
    * $c_h$: Static context for initializing the LSTM hidden state.
    * $c_c$: Static context for initializing the LSTM cell state.
    Each context vector is generated by applying a separate GRN to
    $\zeta$. For example, $c_s = GRN_{c_s}(\zeta)$.

3.  **Locality Enhancement (LSTM Encoder):**
    The sequence of combined past and future embeddings
    $\{\xi_t\}_{t=T-k+1}^{T+\tau}$ is fed into a sequence processing layer,
    typically a multi-layer LSTM (or similar RNN/Transformer encoder),
    initialized with contexts $c_h, c_c$. This captures temporal
    dependencies.
    * $(h_t, cell_t) = LSTM((h_{t-1}, cell_{t-1}), \xi_t)$.
    The LSTM outputs a sequence of hidden states $\{h_t\}$.

4.  **Static Enrichment:**
    The output from the LSTM $\{h_t\}$ is enriched with static context
    $c_e$ using another GRN, applied element-wise across time:
    * $\phi_t = GRN_{enrich}(h_t, c_e)$
    This step ensures that static information influences the
    temporal features before the attention mechanism.

5.  **Temporal Self-Attention:**
    TFT uses an interpretable multi-head attention mechanism, adapted
    from standard Transformer architectures. It calculates attention
    weights over the past time steps ($t \le T$) to produce a
    context vector relevant for predicting future steps.

    * **Attention Calculation:** For each head $h$, attention weights
        are computed, typically focusing on past time steps relative
        to the current forecast time step $t$:
        $\alpha_t^{(h)} = Softmax\left( \frac{Q_t^{(h)} K_{\le T}^{(h)\top}}{\sqrt{d_{attn}}} \right)$,
        where $Q_t^{(h)}$ is derived from $\phi_t$ and $K_{\le T}^{(h)}$
        from $\{\phi_{t'}\}_{t' \le T}$.
    * **Weighted Sum:** The output for head $h$ at time $t$ is
        $Attn_t^{(h)} = \alpha_t^{(h)} V_{\le T}^{(h)}$, where $V_{\le T}^{(h)}$
        is also derived from $\{\phi_{t'}\}_{t' \le T}$.
    * **Multi-Head Aggregation:** Outputs from all heads are
        concatenated and passed through a linear layer:
        $Attn_t = Linear([Attn_t^{(1)}, ..., Attn_t^{(H)}])$.
    * **Gating & Residual:** The attention output is typically gated
        and added back to the enriched features $\phi_t$, followed by
        Layer Normalization:
        $\beta_t = LN( \phi_t + GLU(Linear_3(Attn_t), Linear_4(Attn_t)) )$

    This attention mechanism allows the model to focus on relevant
    past time steps and provides interpretability through the
    learned attention weights $\alpha_t^{(h)}$.

6.  **Position-wise Feed-forward:**
    The output from the attention layer $\beta_t$ is further processed
    by another GRN, applied independently at each time step, to
    produce the final temporal features:
    * $\delta_t = GRN_{final}(\beta_t)$

7.  **Output Layer:**
    Finally, the features corresponding to the forecast horizon
    ($\{\delta_t\}_{t=T+1}^{T+\tau}$) are passed through linear layers
    to produce the final predictions.

    * **Quantile Forecasts:** If quantiles $q \in \{q_1, ..., q_N\}$
        are specified, separate linear layers predict the value for
        each quantile at each horizon step:
        $\hat{y}_{t, q} = Linear_q(\delta_t)$ for $t \in [T+1, T+\tau]$.
    * **Point Forecasts:** If no quantiles are specified, a single
        linear layer predicts the point forecast:
        $\hat{y}_t = Linear_{point}(\delta_t)$ for $t \in [T+1, T+\tau]$.

This detailed flow illustrates how TFT integrates various components
to handle diverse inputs, capture temporal patterns, incorporate
static context, and generate interpretable multi-horizon forecasts
with uncertainty estimates.

.. raw:: html

    <hr>
    
NTemporalFusionTransformer
------------------------------

:API Reference: :class:`~fusionlab.nn.NTemporalFusionTransformer`

The ``NTemporalFusionTransformer`` is a variant of the TFT model
available in ``fusionlab``, characterized by its specific input
requirements and current output capabilities.

**Key Features & Differences:**

* **Mandatory Static & Dynamic Inputs:** Unlike the main
    ``TemporalFusionTransformer``, this class **requires** both
    ``static_input_dim`` and ``dynamic_input_dim`` to be specified
    during initialization. It expects corresponding static and
    dynamic (past) tensors as input during the forward pass.
* **No Future Inputs:** This variant is designed specifically for
    scenarios where known future covariates are not available or
    not used. It does not include processing pathways for future
    inputs.
* **Point Forecasts Only (Current Status):** Based on the current
    implementation (which includes a mechanism to override the
    ``quantiles`` parameter), this class effectively produces only
    deterministic (point) forecasts. Even if quantile values are
    provided, they are currently ignored, and the output represents
    a single predicted value per forecast horizon step.
* **Core TFT Architecture:** It utilizes the fundamental TFT
    components like Variable Selection Networks (VSNs), LSTM
    encoders, Static Enrichment, Temporal Self-Attention, and
    Gated Residual Networks (GRNs), configured for its specific
    input structure.

**When to Use:**

Consider using ``NTemporalFusionTransformer`` primarily when:

* Your forecasting problem involves **only** static metadata and
    dynamic (past) observed features.
* You explicitly **do not** have or need to use known future
    covariates.
* You only require single-value **point forecasts** for each
    future time step.
* You might be working with specific examples or integrations
    designed around this particular variant.

*(Note: For more general use cases, especially those involving
future inputs or requiring probabilistic (quantile) forecasts, the
primary ``TemporalFusionTransformer`` class offers greater
flexibility.)*

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``NTemporalFusionTransformer`` follows the core mathematical
principles of the standard Temporal Fusion Transformer described in
the previous section. It employs the same key components:

* **Variable Selection Networks (VSNs):** Applied to both the
    mandatory static inputs ($s$) and the dynamic past inputs ($x_t$).
* **Static Covariate Encoders:** Process the selected static
    features ($\zeta$) through GRNs to generate context vectors
    ($c_s, c_e, c_h, c_c$).
* **Locality Enhancement (LSTM Encoder):** Processes the sequence of
    selected dynamic embeddings ($\{\xi_t\}_{t \le T}$), initialized
    using static contexts ($c_h, c_c$). Note that only past inputs
    ($t \le T$) are fed into the LSTM sequence.
* **Static Enrichment:** Combines LSTM outputs ($h_t$) with static
    context ($c_e$) using a GRN ($\phi_t = GRN_{enrich}(h_t, c_e)$).
* **Temporal Self-Attention:** Calculates attention weights over the
    enriched past sequence ($\{\phi_{t'}\}_{t' \le T}$) to produce
    contextualized features ($\beta_t$).
* **Position-wise Feed-forward:** Applies a final GRN to the
    attention output ($\delta_t = GRN_{final}(\beta_t)$).

The main distinctions in the formulation compared to the general
description are:

1.  **No Future Input Path:** The architecture omits the processing
    path for known future inputs ($z_t$). VSNs are not applied to
    them, and they are not included in the sequence fed to the LSTM
    or attention mechanisms.
2.  **Point Output Layer:** The final output layer consists of a single
    dense layer applied to the features corresponding to the forecast
    horizon ($\{\delta_t\}_{t=T+1}^{T+\tau}$), producing a single
    value per step: $\hat{y}_t = Linear_{point}(\delta_t)$. It does
    not generate separate outputs for different quantiles.

Essentially, it implements the standard TFT flow but is specialized
for a scenario limited to static/past inputs and point predictions.

.. raw:: html

    <hr>
    
XTFT (Extreme Temporal Fusion Transformer)
---------------------------------------------

:API Reference: :class:`~fusionlab.nn.XTFT`

The ``XTFT`` model represents a significant evolution of the Temporal
Fusion Transformer, designed to tackle highly complex time series
forecasting tasks with enhanced capabilities for representation
learning, multi-scale analysis, and anomaly detection.

**Key Features:**

* **Advanced Input Handling:** Requires static, dynamic (past), and
    future known inputs. Utilizes :class:`~fusionlab.nn.components.LearnedNormalization`
    and :class:`~fusionlab.nn.components.MultiModalEmbedding` for sophisticated
    input processing and fusion.
* **Multi-Scale Temporal Processing:** Employs
    :class:`~fusionlab.nn.components.MultiScaleLSTM` to analyze temporal
    dependencies at different user-defined resolutions (``scales``).
* **Sophisticated Attention Mechanisms:** Incorporates multiple
    specialized attention layers:
    * :class:`~fusionlab.nn.components.HierarchicalAttention`: Captures
        patterns potentially across different input groups or levels.
    * :class:`~fusionlab.nn.components.CrossAttention`: Models
        interactions between different input sequences (e.g., dynamic
        history and combined embeddings).
    * :class:`~fusionlab.nn.components.MemoryAugmentedAttention`: Uses
        an external memory bank to potentially recall longer-range
        patterns.
    * :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`:
        Combines the outputs of various LSTM and attention pathways.
* **Dynamic Temporal Focus:** Uses a
    :class:`~fusionlab.nn.components.DynamicTimeWindow` component to adaptively
    focus on the most relevant recent time steps from the fused
    features.
* **Integrated Anomaly Detection:** Offers multiple strategies
    (``anomaly_detection_strategy`` parameter) for identifying and
    incorporating anomaly information:
    * ``'feature_based'``: Learns to detect anomalies from feature
        interactions using dedicated attention layers.
    * ``'prediction_based'``: Identifies anomalies based on deviations
        between predictions and actuals during training (requires
        a custom loss function).
    * ``'from_config'``: Allows providing pre-computed anomaly scores.
    An :class:`~fusionlab.nn.components.AnomalyLoss` component is used
    to integrate this into the training objective, weighted by
    ``anomaly_loss_weight``.
* **Flexible Output:** Features a :class:`~fusionlab.nn.components.MultiDecoder`
    and :class:`~fusionlab.nn.components.QuantileDistributionModeling` layer
    to generate multi-horizon forecasts for specified ``quantiles``
    (or point forecasts if ``quantiles`` is None).

**When to Use:**

XTFT is designed for challenging forecasting problems where:

* The underlying temporal dynamics are highly complex and potentially
    span multiple time scales.
* Rich static, dynamic, and future information needs to be
    integrated effectively.
* Capturing long-range dependencies is important (leveraging memory
    attention).
* Identifying or accounting for anomalies within the time series is
    a requirement.
* Maximum predictive performance is desired, potentially at the cost
    of increased model complexity and computational resources compared
    to standard TFT.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~

XTFT significantly extends the standard TFT architecture. While it
builds upon core concepts like GRNs and attention, it introduces
many specialized components. We highlight the key additions and
modifications here, using notation consistent with the previous TFT
description. For full details, please refer to the source code and
the documentation of individual components (linked above).

1.  **Input Processing:**
    * Static inputs ($s$) undergo :class:`~fusionlab.nn.components.LearnedNormalization`
        and are processed by GRNs (similar to TFT static context).
    * Dynamic ($x_t$) and Future ($z_t$) inputs are jointly processed
        by :class:`~fusionlab.nn.components.MultiModalEmbedding` to create
        initial combined embeddings.
    * :class:`~fusionlab.nn.components.PositionalEncoding` is added to
        these embeddings.
    * Optional residual connections can be applied.

2.  **Multi-Scale LSTM:**
    * The dynamic inputs $x_t$ are processed by
        :class:`~fusionlab.nn.components.MultiScaleLSTM` using different
        temporal ``scales``.
    * The outputs from different scales are aggregated based on the
        ``multi_scale_agg`` method (e.g., taking the 'last' step,
        averaging) into `lstm_features`.

3.  **Advanced Attention Layers:**
    * :class:`~fusionlab.nn.components.HierarchicalAttention` processes
        dynamic and future inputs.
    * :class:`~fusionlab.nn.components.CrossAttention` models interactions
        between dynamic inputs and the combined embeddings.
    * :class:`~fusionlab.nn.components.MemoryAugmentedAttention` takes
        hierarchical attention output and interacts with an external
        memory matrix.

4.  **Feature Fusion:**
    * The processed static features, aggregated multi-scale LSTM
        features, and outputs from the various attention mechanisms
        (Hierarchical, Cross, Memory-Augmented) are concatenated.
    * :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`
        is applied to this combined feature set to produce a unified
        temporal representation (`attention_fusion_output`).

5.  **Dynamic Windowing & Aggregation:**
    * :class:`~fusionlab.nn.components.DynamicTimeWindow` selects or weights
        recent time steps from the `attention_fusion_output`.
    * The result is aggregated (e.g., 'last' step, 'average') based
        on `final_agg` into `final_features`.

6.  **Decoding and Output:**
    * :class:`~fusionlab.nn.components.MultiDecoder` transforms the
        `final_features` across the `forecast_horizon`.
    * :class:`~fusionlab.nn.components.QuantileDistributionModeling` maps
        the decoder outputs to the final quantile (or point)
        predictions $\hat{y}_{t, q}$ / $\hat{y}_t$.

7.  **Anomaly Detection Integration:**
    * **Feature-Based:** If strategy is ``'feature_based'``, the
        `attention_fusion_output` is passed through dedicated
        `anomaly_attention`, `anomaly_projection`, and `anomaly_scorer`
        layers during the forward pass to compute internal
        `anomaly_scores`.
    * **Config-Based:** If strategy is ``'from_config'``, pre-computed
        `anomaly_scores` (provided via `anomaly_config`) are used.
    * **Loss Calculation:** If `anomaly_scores` are available (either
        computed or provided), the :class:`~fusionlab.nn.components.AnomalyLoss`
        layer calculates a loss based on these scores, which is added
        to the model's total loss during training via ``model.add_loss``.
    * **Prediction-Based:** If strategy is ``'prediction_based'``, a
        special combined loss function is used during `compile` and
        the custom `train_step` handles calculating loss based on
        prediction errors and anomaly considerations simultaneously.

XTFT orchestrates these advanced components to create a powerful and
flexible architecture capable of handling very complex time series
dynamics and incorporating domain-specific features like anomaly
detection.

.. raw:: html

    <hr>

SuperXTFT
-----------

:API Reference: :class:`~fusionlab.nn.SuperXTFT`

.. warning::
   ``SuperXTFT`` is currently considered **experimental** and is under
   maintenance. It is **not recommended for production use** at this
   time. Please use the standard :class:`~fusionlab.nn.XTFT` for stable
   deployments. Stay tuned for future updates regarding the status
   of ``SuperXTFT``.

The ``SuperXTFT`` class inherits from :class:`~fusionlab.nn.XTFT` and
introduces specific architectural modifications aimed at potentially
enhancing feature representation and processing flow.

**Key Features & Differences (from XTFT):**

* **Inherits XTFT Features:** Includes all the advanced components
    and capabilities of the base ``XTFT`` model (Multi-Scale LSTM,
    advanced attention, anomaly detection, etc.).
* **Adds Variable Selection Networks (VSNs):** Re-introduces VSNs
    (similar to standard TFT) applied directly to the static,
    dynamic (past), and future inputs at the beginning of the
    forward pass. The outputs of these VSNs (selected/weighted
    features) are then fed into the rest of the XTFT architecture.
* **Adds Post-Processing GRNs:** Integrates dedicated Gated
    Residual Network (GRN) layers immediately following several key
    components:
    * After Hierarchical Attention
    * After Cross Attention
    * After Memory-Augmented Attention
    * After the Multi-Decoder stage (before quantile modeling)
    These GRNs apply further non-linear processing and gating to the
    outputs of these specific stages.

**When to Use:**

* **Currently:** Primarily for internal development, testing, or
    research purposes within the ``fusionlab`` project itself.
* **Future:** Intended as a potentially enhanced alternative to
    ``XTFT`` once development is complete and the model is stable.
* **Avoid for production or general use until officially released
    and undeprecated.**

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~

``SuperXTFT`` modifies the data flow of the base ``XTFT`` model in
two main ways:

1.  **Input Variable Selection:**
    Unlike ``XTFT`` where inputs might go directly into normalization
    or embedding layers, ``SuperXTFT`` first processes each input
    type through a dedicated :class:`~fusionlab.nn.components.VariableSelectionNetwork`:
    * Static inputs $s \rightarrow s' = VSN_{static}(s)$
    * Dynamic inputs $x_t \rightarrow x'_t = VSN_{dynamic}(x_t)$
    * Future inputs $z_t \rightarrow z'_t = VSN_{future}(z_t)$
    These *selected* features ($s', x'_t, z'_t$) are then used as
    inputs to the subsequent stages described in the XTFT
    formulation (e.g., $s'$ goes to Learned Normalization, $x'_t$
    and $z'_t$ go to MultiModal Embedding).

2.  **Integrated Post-Processing GRNs:**
    After specific intermediate outputs are computed within the main
    XTFT flow, ``SuperXTFT`` applies an additional GRN transformation.
    Conceptually:
    * Hierarchical Attention Output $Attn_{hier} \rightarrow GRN_{hier}(Attn_{hier})$
    * Cross Attention Output $Attn_{cross} \rightarrow GRN_{cross}(Attn_{cross})$
    * Memory Attention Output $Attn_{mem} \rightarrow GRN_{mem}(Attn_{mem})$
    * Multi-Decoder Output $Dec_{out} \rightarrow GRN_{dec}(Dec_{out})$
    The output of these dedicated GRNs then replaces the original
    output in the subsequent steps of the network (e.g., the output
    of $GRN_{cross}$ is used in the feature concatenation step
    instead of the raw $Attn_{cross}$). This adds extra processing
    steps within the main architectural graph.

These modifications aim to potentially improve feature selection and
refine the representations generated by key components of the XTFT
architecture, but the model is currently experimental.