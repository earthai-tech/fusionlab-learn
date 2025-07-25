.. _tft_models_guide:

================================================
Temporal Fusion Transformer (TFT) and Variants
================================================

This section of the user guide provides a detailed overview of the
**Temporal Fusion Transformer (TFT)** architecture and its various
implementations available within the ``fusionlab-learn`` library. The
TFT is a landmark architecture for multi-horizon time series
forecasting, renowned for its ability to deliver high performance while
maintaining interpretability.

Its core innovation lies in its hybrid design, which skillfully
combines recurrent layers for processing local sequences, self-attention
for capturing long-range dependencies, and gating mechanisms to
intelligently filter information from static, dynamic, and future-known
inputs.

``fusionlab-learn`` offers several implementations of the TFT, each
tailored for different use cases and input requirements:

* **TemporalFusionTransformer**: The primary, flexible, general-purpose
  implementation that can gracefully handle scenarios where some input
  types (like static or future features) might be missing.
* **TFT (Stricter)** & **DummyTFT**: More specialized variants that
  enforce rigid input patterns (e.g., requiring all inputs, or only
  static and dynamic), which can be useful for ensuring data integrity
  in well-defined pipelines.

This guide details each of these models, their unique features, and
their typical applications.

.. toctree::
   :hidden:

.. raw:: html

   <hr>

.. _temporal_fusion_transformer_model:

TemporalFusionTransformer
--------------------------

:API Reference: :class:`~fusionlab.nn.TemporalFusionTransformer`

The ``TemporalFusionTransformer`` class is the primary, flexible
implementation of the Temporal Fusion Transformer architecture
[1]_ within ``fusionlab``. It is designed to handle a variety of
input configurations and forecasting tasks.

**Key Features:**

* **Flexible Inputs:** Supports dynamic (past) inputs (required),
  optional static inputs (time-invariant metadata), and optional
  future inputs (known exogenous covariates). The model adapts
  internally based on which inputs are provided.
* **Multi-Horizon Forecasting:** Directly outputs predictions for
  multiple future time steps defined by the ``forecast_horizon``
  parameter.
* **Probabilistic Forecasts:** Can generate quantile forecasts by
  specifying the ``quantiles`` parameter (e.g., ``[0.1, 0.5, 0.9]``)
  to estimate prediction intervals and uncertainty. If
  ``quantiles`` is ``None``, it produces deterministic (point)
  forecasts.
* **Standard TFT Components:** Built using core TFT blocks like
  :class:`~fusionlab.nn.components.VariableSelectionNetwork`
  (optional for static/future), LSTM encoders, Static Enrichment
  (using :class:`~fusionlab.nn.components.GatedResidualNetwork`),
  Temporal Self-Attention
  (:class:`~fusionlab.nn.components.TemporalAttentionLayer`),
  and :class:`~fusionlab.nn.components.GatedResidualNetwork`.

**When to Use:**

This is generally the recommended starting point for applying the
TFT architecture. Use it when:

* You need a standard, well-understood TFT implementation.
* You have various combinations of dynamic, static, or future
  inputs (and don't necessarily have all three).
* You require either point forecasts or probabilistic (quantile)
  forecasts.

**Code Example (Instantiation and Call)**

This example shows how to instantiate the flexible
``TemporalFusionTransformer`` with different combinations of inputs.

.. code-block:: python
   :linenos:

   import numpy as np
   import tensorflow as tf
   from fusionlab.nn import TemporalFusionTransformer # Flexible version

   # --- Dummy Data Dimensions ---
   B, T_past, H = 4, 12, 6 # Batch, Lookback, Forecast Horizon
   D_dyn, D_stat, D_fut = 5, 3, 2 # Feature dimensions
   T_future_total = T_past + H # Future inputs span lookback + horizon

   # Create Dummy Input Tensors
   static_in = tf.random.normal((B, D_stat), dtype=tf.float32)
   dynamic_in = tf.random.normal((B, T_past, D_dyn), dtype=tf.float32)
   future_in = tf.random.normal((B, T_future_total, D_fut), dtype=tf.float32)

   # --- Example 1: Dynamic Inputs Only (Point Forecast) ---
   print("--- Example 1: Dynamic Only ---")
   model_dyn_only = TemporalFusionTransformer(
       dynamic_input_dim=D_dyn,
       static_input_dim=None,  # Explicitly None
       future_input_dim=None,  # Explicitly None
       forecast_horizon=H,
       output_dim=1, # Specify output dimension
       hidden_units=8, num_heads=1 # Small params for example
   )
   # Input is a list: [Static, Dynamic, Future]
   # For dynamic only, static and future are None
   inputs1 = [None, dynamic_in, None]
   try:
       output1 = model_dyn_only(inputs1, training=False)
       print(f"Input: [None, Dynamic ({dynamic_in.shape}), None]")
       print(f"Output Shape (Point): {output1.shape}")
       # Expected: (B, H, OutputDim=1) -> (4, 6, 1)
   except Exception as e:
       print(f"Call failed for Dynamic Only: {e}")


   # --- Example 2: All Inputs (Quantile Forecast) ---
   print("\n--- Example 2: All Inputs (Quantile) ---")
   my_quantiles = [0.1, 0.5, 0.9]
   model_all_inputs = TemporalFusionTransformer(
       dynamic_input_dim=D_dyn,
       static_input_dim=D_stat,   # Provide static dim
       future_input_dim=D_fut,   # Provide future dim
       forecast_horizon=H,
       quantiles=my_quantiles,   # Set quantiles
       output_dim=1,             # Specify output dimension
       hidden_units=8, num_heads=1
   )
   # Input is a list [Static, Dynamic, Future]
   inputs2 = [static_in, dynamic_in, future_in]
   try:
       output2 = model_all_inputs(inputs2, training=False)
       print(f"Input: [Stat ({static_in.shape}), Dyn ({dynamic_in.shape}), "
             f"Fut ({future_in.shape})]")
       print(f"Output Shape (Quantile): {output2.shape}")
       # Expected: (B, H, NumQuantiles) if output_dim=1 -> (4, 6, 3)
   except Exception as e:
       print(f"Call failed for All Inputs: {e}")

.. important:: Input Data Order and Format for `TemporalFusionTransformer`

   The flexible ``TemporalFusionTransformer`` expects its `inputs` argument
   in the ``call`` method (and subsequently for ``.fit()``, ``.predict()``)
   to be a **list or tuple of three elements**:
   ``[static_features, dynamic_features, future_features]``.

   * If a particular input type is not used (e.g., no static features),
     pass ``None`` for that element in the list.
     For example:
     
     * Dynamic only: ``[None, dynamic_array, None]``
     * Dynamic + Static: ``[static_array, dynamic_array, None]``
     * Dynamic + Future: ``[None, dynamic_array, future_array]``
     * All three: ``[static_array, dynamic_array, future_array]``
   * The ``dynamic_features`` input is always required.
   * The model's ``__init__`` parameters (`static_input_dim`,
     `dynamic_input_dim`, `future_input_dim`) determine which of these
     inputs are expected to be actual tensors versus ``None``.
   * This order is handled by the internal validation logic
     (:func:`~fusionlab.nn._tensor_validation.validate_model_inputs`
     when ``model_name='tft_flex'``).


Formulation
~~~~~~~~~~~~~~~~

Here, we describe the core mathematical concepts behind the
Temporal Fusion Transformer, following the architecture outlined
in the original paper [1]_. This provides insight into how different
inputs are processed and transformed to generate forecasts.

**Notation:**

* **Inputs:**
    * :math:`s \in \mathbb{R}^{d_s}`: Static (time-invariant) covariates.
    * :math:`z_t \in \mathbb{R}^{d_z}`: Known future inputs at time :math:`t`.
    * :math:`x_t \in \mathbb{R}^{d_x}`: Observed past dynamic inputs at time :math:`t`.
    * :math:`y_t \in \mathbb{R}^{d_y}`: Past target variable(s) at time :math:`t`
      (often included in :math:`x_t`).
* **Time Indices:**
    * :math:`t \in [T-k+1, T]`: Past time steps within the lookback
      window of size :math:`k`.
    * :math:`t \in [T+1, T+\tau]`: Future time steps for the forecast
      horizon :math:`\tau`.
* **Dimensions:**
    * :math:`d_s, d_x, d_z, d_y`: Dimensionalities of respective inputs.
    * :math:`d_{model}`: The main hidden state dimension of the model
      (e.g., ``hidden_units``).
* **Common Functions:**
    * :math:`LN(\cdot)`: Layer Normalization.
    * :math:`\sigma(\cdot)`: Sigmoid activation function.
    * :math:`ReLU(\cdot), ELU(\cdot)`: Activation functions.
    * :math:`Linear(\cdot)`: A dense (fully-connected) layer.
    * :math:`GLU(a, b) = a \odot \sigma(b)`: Gated Linear Unit, where
      :math:`\odot` is element-wise multiplication.
    * :math:`GRN(a, [c])`: Gated Residual Network. A key block roughly defined as:
      :math:`GRN(a, c) = LN(a' + GLU(Linear_1(act(Linear_0(a'))), Linear_2(a')))`,
      where :math:`a' = a+Linear_c(c)` if context :math:`c` is provided, else :math:`a'=a`.

**Architectural Flow:**

1.  **Input Transformations & Variable Selection:**
    Inputs (categorical/continuous) are transformed into numerical
    vectors (e.g., via embeddings or linear layers). **Variable
    Selection Networks (VSNs)** are applied to each input type
    (static :math:`s`, past dynamic :math:`x_t`, known future :math:`z_t`),
    potentially conditioned on static context :math:`c_s`.

    * VSN computes feature weights :math:`\alpha_\chi` and applies
      feature-wise GRNs (:math:`\tilde{\chi}^j = GRN(\chi^j)`).
    * Output is a weighted sum: :math:`\xi = \sum_{j} \alpha_\chi^j \tilde{\chi}^j`.

    This yields embeddings: static :math:`\zeta`, past dynamic
    :math:`\xi_t` (:math:`t \le T`), and future :math:`\xi_t` (:math:`t > T`).

2.  **Static Covariate Encoders:**
    The static embedding :math:`\zeta` is processed through dedicated
    GRNs to produce four context vectors for conditioning different parts
    of the temporal processing: :math:`c_s` (VSN context), :math:`c_e`
    (enrichment context), :math:`c_h` (LSTM initial hidden state),
    :math:`c_c` (LSTM initial cell state).
    E.g., :math:`c_s = GRN_{vs}(\zeta)`.

3.  **Locality Enhancement (LSTM Encoder):**
    The sequence of combined past and future VSN embeddings
    :math:`\{\xi_t\}_{t=T-k+1}^{T+\tau}` is fed into a sequence
    processing layer (typically multi-layer LSTM), initialized with
    contexts :math:`c_h, c_c`.
    :math:`(h_t, cell_t) = LSTM((h_{t-1}, cell_{t-1}), \xi_t)`.
    The output is a sequence of hidden states :math:`\{h_t\}`.

4.  **Static Enrichment:**
    The LSTM output sequence :math:`\{h_t\}` is enriched with static
    context :math:`c_e` using another GRN applied time-wise:
    :math:`\phi_t = GRN_{enrich}(h_t, c_e)`.

5.  **Temporal Self-Attention:**
    An interpretable multi-head attention mechanism processes the
    enriched sequence :math:`\{\phi_t\}`. The static context :math:`c_s`
    may condition the query generation or internal GRNs. It computes
    attention weights over past time steps relative to the current
    forecast time step.

    * **Attention Calculation (Simplified):**
      Weights :math:`\alpha_t^{(h)}` for head :math:`h` at step :math:`t`
      are computed via scaled dot-product attention, typically using
      :math:`\phi_t` to form Queries and :math:`\{\phi_{t'}\}_{t' \le T}`
      to form Keys and Values.
      :math:`\alpha_t^{(h)} = \text{Softmax}\left( \dots \right)`.
    * **Output & Gating:** The attention output :math:`Attn_t` is combined
      with :math:`\phi_t` using gating (GLU) and a residual connection,
      followed by Layer Normalization:
      :math:`\beta_t = LN( \phi_t + GLU(..., Attn_t))`.

6.  **Position-wise Feed-forward:**
    The attention output :math:`\beta_t` is processed by another GRN
    applied independently at each time step: :math:`\delta_t = GRN_{final}(\beta_t)`.

7.  **Output Layer:**
    The final features corresponding to the forecast horizon
    :math:`\{\delta_t\}_{t=T+1}^{T+\tau}` are passed through linear layers
    to produce predictions.
    
    * **Quantiles:** Separate linear layers for each quantile :math:`q`:
      
      .. math:: 
         \hat{y}_{t, q} = Linear_q(\delta_t)
         
    * **Point:** A single linear layer: :math:`\hat{y}_t = Linear_{point}(\delta_t)`.

This detailed flow illustrates how TFT integrates various components
to handle diverse inputs, capture temporal patterns, incorporate
static context, and generate interpretable multi-horizon forecasts
with uncertainty estimates.


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _tft_model_stricter:

TFT (Stricter Implementation - All Inputs Required)
---------------------------------------------------
:API Reference: :class:`~fusionlab.nn.transformers.TFT`

*(Note: This refers to a specific* ``TFT`` *class implementation within
fusionlab that enforces stricter input requirements compared to the
more flexible* ``TemporalFusionTransformer`` *described above. It assumes
static, dynamic past, and known future inputs are always provided and
are not* ``None``).

This class implements the Temporal Fusion Transformer (TFT)
architecture, closely following the structure described in the
original paper [1]_. It is designed for multi-horizon time
series forecasting and explicitly requires static covariates,
dynamic (historical) covariates, and known future covariates as
inputs.


Compared to implementations allowing optional inputs, this version
mandates all input types, simplifying the internal input handling
logic. It incorporates key TFT components like
:class:`~fusionlab.nn.components.VariableSelectionNetwork` (VSNs),
:class:`~fusionlab.nn.components.GatedResidualNetwork` (GRNs) for
static context generation and feature processing, LSTM encoding,
static enrichment, interpretable multi-head attention
(:class:`~fusionlab.nn.components.TemporalAttentionLayer`),
and position-wise feedforward layers.

**Use Case and Importance**

This `TFT` class provides a structured implementation useful
when all feature types (static, dynamic past, known future) are
readily available and adherence to the paper's component structure
(like distinct static contexts) is desired. It serves as a strong
baseline for complex forecasting tasks demanding interpretability
and handling of heterogeneous data. Its requirement for all inputs
simplifies the `call` method, making the internal flow potentially
easier to follow.

**Parameters**

* **dynamic_input_dim** (`int`):
  Total number of features in the dynamic (past) input tensor.
* **static_input_dim** (`int`):
  Total number of features in the static (time-invariant) input tensor.
* **future_input_dim** (`int`):
  Total number of features in the known future input tensor.
* **hidden_units** (`int`, default: `32`):
  Main dimensionality of hidden layers (VSNs, GRNs, Attention).
* **num_heads** (`int`, default: `4`):
  Number of attention heads in the Temporal Attention Layer.
* **dropout_rate** (`float`, default: `0.1`):
  Dropout rate for non-recurrent connections (0 to 1).
* **recurrent_dropout_rate** (`float`, default: `0.0`):
  Dropout rate for LSTM recurrent connections (0 to 1).
* **forecast_horizon** (`int`, default: `1`):
  Number of future time steps to predict.
* **quantiles** (`Optional[List[float]]`, default: `None`):
  List of quantiles (e.g., `[0.1, 0.5, 0.9]`) for probabilistic
  forecasting. If `None`, performs point forecasting.
* **activation** (`str`, default: `'elu'`):
  Activation function for GRNs (e.g., 'relu', 'gelu').
* **use_batch_norm** (`bool`, default: `False`):
  If True, use Batch Normalization in GRNs.
* **num_lstm_layers** (`int`, default: `1`):
  Number of stacked LSTM layers in the encoder.
* **lstm_units** (`Optional[Union[int, List[int]]]`, default: `None`):
  Units per LSTM layer. If `int`, used for all layers. If `list`,
  length must match `num_lstm_layers`. Defaults to `hidden_units`.
* **output_dim** (`int`, default: `1`):
  Number of target variables predicted per step.

**Notes**

* **Input Format:** This implementation **requires** inputs to the `call`
  method as a list or tuple containing exactly three tensors in the
  order: ``[static_inputs, dynamic_inputs, future_inputs]``.
  Expected shapes are generally:
  
  * `static_inputs`: :math:`(B, D_s)`
  * `dynamic_inputs`: :math:`(B, T_{past}, D_{dyn})`
  * `future_inputs`: :math:`(B, T_{future}, D_{fut})` *(Note: The
    required length* :math:`T_{future}` *depends on how inputs are
    combined internally before the LSTM. Ensure data preparation
    aligns, e.g., using* :func:`~fusionlab.nn.utils.reshape_xtft_data`).
* **Categorical Features:** This implementation assumes inputs
  are *numeric*. Handling categorical features requires modifications
  (e.g., adding embedding layers before VSNs).

Formulation
~~~~~~~~~~~~~~
*(This section describes the flow assuming numeric inputs)*

1.  **Variable Selection:** Separate
    :class:`~fusionlab.nn.components.VariableSelectionNetwork` (VSNs)
    process static (:math:`\mathbf{s}`), dynamic past (:math:`\mathbf{x}_t`),
    and known future (:math:`\mathbf{z}_t`) inputs, potentially conditioned
    by static context (:math:`c_s`). Outputs: :math:`\zeta`, :math:`\xi^{dyn}_t`,
    :math:`\xi^{fut}_t`.

2.  **Static Context Generation:** Four distinct
    :class:`~fusionlab.nn.components.GatedResidualNetwork` (GRNs) process
    the static VSN output :math:`\zeta` to produce context vectors:
    :math:`c_s` (for VSNs), :math:`c_e` (for enrichment), :math:`c_h`
    (LSTM initial hidden state), :math:`c_c` (LSTM initial cell state).

3.  **Temporal Processing Input:** Selected dynamic (:math:`\xi^{dyn}_t`) and
    future (:math:`\xi^{fut}_t`) embeddings are combined (e.g., concatenated
    along features) and augmented with
    :class:`~fusionlab.nn.components.PositionalEncoding` (:math:`\psi_t`).

4.  **LSTM Encoder:** A stack of LSTMs processes :math:`\psi_t`, initialized
    with :math:`[c_h, c_c]`, outputting hidden states :math:`\{h_t\}`.

    .. math::
       \{h_t\} = \text{LSTMStack}(\{\psi_t\}, \text{init}=[c_h, c_c])

5.  **Static Enrichment:** A time-distributed GRN combines LSTM outputs
    :math:`h_t` with the static enrichment context :math:`c_e`.

    .. math::
       \phi_t = GRN_{enrich}(h_t, c_e)

6.  **Temporal Self-Attention:** :class:`~fusionlab.nn.components.TemporalAttentionLayer`
    processes the enriched sequence :math:`\{\phi_t\}` using :math:`c_s` as context,
    outputting :math:`\beta_t` after internal gating/residuals.

    .. math::
       \beta_t = \text{TemporalAttention}(\{\phi_t\}, c_s)

7.  **Position-wise Feed-Forward:** A final time-distributed GRN processes :math:`\beta_t`.

    .. math::
       \delta_t = GRN_{final}(\beta_t)

8.  **Output Projection:** Features for the forecast horizon (:math:`t > T`)
    are selected from :math:`\{\delta_t\}` (typically the last :math:`H` steps)
    and passed through output :class:`~tf.keras.layers.Dense` layer(s) for point
    (:math:`\hat{y}_t`) or quantile (:math:`\hat{y}_{t, q}`) predictions.

**Code Example (Instantiation & Call):**

.. code-block:: python
   :linenos:

   import numpy as np
   import tensorflow as tf
   from fusionlab.nn.transformers import TFT 

   # Dummy Data Dimensions
   B, T_past, H = 4, 12, 6
   D_dyn, D_stat, D_fut = 5, 3, 2
   T_future = T_past + H # Example: Future covers lookback + horizon

   # Create Dummy Input Tensors (ALL REQUIRED)
   static_in = tf.random.normal((B, D_stat), dtype=tf.float32)
   dynamic_in = tf.random.normal((B, T_past, D_dyn), dtype=tf.float32)
   future_in = tf.random.normal((B, T_future, D_fut), dtype=tf.float32)

   # Instantiate the revised TFT Model (Point Forecast)
   model = TFT(
       dynamic_input_dim=D_dyn,
       static_input_dim=D_stat,
       future_input_dim=D_fut,
       forecast_horizon=H,
       hidden_units=16,
       num_heads=2,
       quantiles=None # Point forecast
   )

   # Prepare input list in REQUIRED order: [static, dynamic, future]
   model_inputs = [static_in, dynamic_in, future_in]

   # Call the model (builds on first call)
   predictions = model(model_inputs)

   print(f"Input Shapes: S={static_in.shape}, D={dynamic_in.shape}, F={future_in.shape}")
   print(f"Output shape (Point): {predictions.shape}")
   # Expected: (B, H, OutputDim=1) -> (4, 6, 1)


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

.. _dummy_tft_model:

DummyTFT (Static & Dynamic Inputs Only)
---------------------------------------
:API Reference: :class:`~fusionlab.nn.transformers.DummyTFT`

The ``DummyTFT`` (formerly ``NTemporalFusionTransformer``) is a
variant of the TFT model available in ``fusionlab``. It is
characterized by its specific input requirements, focusing on scenarios
where only static and dynamic (past) features are available.

.. param_deprecated_message::
   :conditions_params_mappings:
     - param: future_input_dim
       condition: "lambda v: v is not None"
       message: |
         The 'future_input_dim' parameter is accepted by DummyTFT for API
         consistency with other TFT models, but DummyTFT **does not
         utilize future covariates**. This parameter will be internally
         ignored and effectively treated as None. If you need to use
         known future covariates, please consider using the standard
         :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
         (for flexible input handling) or the stricter
         :class:`~fusionlab.nn.transformers.TFT` (which requires future inputs).
       default: None

**Key Features & Differences:**

* **Mandatory Static & Dynamic Inputs:** This class **requires** both
  ``static_input_dim`` and ``dynamic_input_dim`` to be specified
  during initialization and expects corresponding non-None static and
  dynamic (past) tensors as input.
* **No Future Inputs Used:** This variant is designed specifically for
  scenarios where known future covariates are not available or
  not utilized. The architecture omits pathways for processing
  future inputs.
* **Point or Quantile Forecasts:** Can produce deterministic (point)
  forecasts or probabilistic (quantile) forecasts if the ``quantiles``
  parameter is specified.
* **Core TFT Architecture:** It utilizes fundamental TFT components like
  :class:`~fusionlab.nn.components.VariableSelectionNetwork` (VSNs for
  static and dynamic inputs), LSTM encoders, Static Enrichment,
  Temporal Self-Attention
  (:class:`~fusionlab.nn.components.TemporalAttentionLayer`), and
  :class:`~fusionlab.nn.components.GatedResidualNetwork` (GRNs),
  configured for its two-input structure.

**When to Use:**

Consider using ``DummyTFT`` primarily when:

* Your forecasting problem involves **only** static metadata and
  dynamic (past) observed features.
* You explicitly **do not** have or require known future covariates.
* You need point or quantile forecasts based on this two-input setup.

Formulation
~~~~~~~~~~~~~
The ``DummyTFT`` follows the core mathematical principles of the
standard Temporal Fusion Transformer [1]_, employing components like
VSNs, static context GRNs, LSTM encoding, static enrichment,
temporal self-attention, and position-wise feed-forward GRNs.

The main distinctions in the formulation are:

1.  **No Future Input Path:** The architecture **omits** the processing
    pathway for known future inputs (:math:`z_t`). VSNs are not applied
    to them, and they are not included in the sequence fed to the LSTM
    or attention mechanisms. Only static (:math:`s`) and past dynamic
    (:math:`x_t`) inputs are processed.
2.  **Output Layer:** The final output layer processes features derived
    from static and dynamic inputs to produce point or quantile
    predictions for the forecast horizon.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   import tensorflow as tf
   from fusionlab.nn.transformers import DummyTFT

   # Dummy Data Dimensions
   B, T_past, H = 4, 12, 6 # Batch, Lookback, Horizon
   D_dyn, D_stat = 5, 3    # Dynamic, Static feature dimensions
   output_dim = 1          # Univariate target

   # Create Dummy Input Tensors (Static and Dynamic ONLY)
   static_in = tf.random.normal((B, D_stat), dtype=tf.float32)
   dynamic_in = tf.random.normal((B, T_past, D_dyn), dtype=tf.float32)

   # Instantiate the DummyTFT Model (Point Forecast)
   model_point = DummyTFT(
       static_input_dim=D_stat,
       dynamic_input_dim=D_dyn,
       forecast_horizon=H,
       output_dim=output_dim,
       hidden_units=16, num_heads=2,
       quantiles=None # Point forecast
   )

   # Prepare input list: [static, dynamic]
   model_inputs_point = [static_in, dynamic_in]

   # Call the model
   try:
       predictions_point = model_point(model_inputs_point, training=False)
       print("--- DummyTFT Point Forecast ---")
       print(f"Input Shapes: S={static_in.shape}, D={dynamic_in.shape}")
       print(f"Output shape (Point): {predictions_point.shape}")
       # Expected: (B, H, O) -> (4, 6, 1)
   except Exception as e:
       print(f"DummyTFT (Point) call failed: {e}")

   # Instantiate for Quantile Forecast
   my_quantiles = [0.2, 0.5, 0.8]
   model_quant = DummyTFT(
       static_input_dim=D_stat,
       dynamic_input_dim=D_dyn,
       forecast_horizon=H,
       output_dim=output_dim,
       quantiles=my_quantiles,
       hidden_units=16, num_heads=2
   )
   model_inputs_quant = [static_in, dynamic_in]
   try:
       predictions_quant = model_quant(model_inputs_quant, training=False)
       print("\n--- DummyTFT Quantile Forecast ---")
       print(f"Output shape (Quantile): {predictions_quant.shape}")
       # Expected for output_dim=1: (B, H, NumQuantiles) -> (4, 6, 3)
   except Exception as e:
       print(f"DummyTFT (Quantile) call failed: {e}")


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Next Steps
----------

.. note::

   You now understand the theory and the complete workflow for 
   ``TFT`` and its variants, you can proceed to the exercises for more hands-on practice:
   
   * :doc:`../../exercises/exercise_tft_required_inputs`
   * :doc:`../../exercises/exercise_basic_forecasting`
   * :doc:`../../exercises/exercise_quantile_forecast`.
   
   
.. rubric:: References

.. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
   Temporal fusion transformers for interpretable multi-horizon
   time series forecasting. *International Journal of Forecasting*,
   37(4), 1748-1764. (Also arXiv:1912.09363)