



.. _xtft_model:

XTFT (Extreme Temporal Fusion Transformer)
--------------------------------------------
:API Reference: :class:`~fusionlab.nn.transformers.XTFT`

The ``XTFT`` model represents a significant evolution of the Temporal
Fusion Transformer, designed to tackle highly complex time series
forecasting tasks with enhanced capabilities for representation
learning, multi-scale analysis, and integrated anomaly detection.

**Key Features:**

* **Advanced Input Handling:** Requires static, dynamic (past), and
  future known inputs. Utilizes components like
  :class:`~fusionlab.nn.components.LearnedNormalization` and
  :class:`~fusionlab.nn.components.MultiModalEmbedding` for input
  processing. *Note: Unlike the revised TFT, XTFT internally uses
  these components and doesn't rely on VSNs directly at the input stage.*
* **Multi-Scale Temporal Processing:** Employs
  :class:`~fusionlab.nn.components.MultiScaleLSTM` to analyze temporal
  dependencies at different user-defined resolutions (via ``scales``).
  Output aggregation is handled by
  :func:`~fusionlab.nn.components.aggregate_multiscale`.
* **Sophisticated Attention Mechanisms:** Incorporates multiple
  specialized attention layers for richer context modeling:
  
  * :class:`~fusionlab.nn.components.HierarchicalAttention`
  * :class:`~fusionlab.nn.components.CrossAttention`
  * :class:`~fusionlab.nn.components.MemoryAugmentedAttention`
  * :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`
    
* **Dynamic Temporal Focus:** Uses a
  :class:`~fusionlab.nn.components.DynamicTimeWindow` component to potentially
  focus on the most relevant recent time steps before final aggregation.
* **Flexible Aggregation:** Aggregates final temporal features using
  different strategies (``final_agg`` parameter, handled by
  :func:`~fusionlab.nn.components.aggregate_time_window_output`).
* **Integrated Anomaly Detection:** Offers multiple strategies
  (via ``anomaly_detection_strategy`` parameter) for incorporating
  anomaly information into the training process:
  
  * **'feature_based':** Learns anomaly scores from internal features
    using dedicated attention/scoring layers.
  * **'prediction_based':** Calculates anomaly scores based on
    prediction errors using a specialized loss function
    (:func:`~fusionlab.nn.losses.prediction_based_loss`).
  * **'from_config':** Uses pre-computed anomaly scores provided via
    the ``anomaly_config`` dictionary, integrated into the loss via
    :class:`~fusionlab.nn.components.AnomalyLoss` and potentially
    :func:`~fusionlab.nn.losses.combined_total_loss`.
    The contribution of anomaly loss is controlled by ``anomaly_loss_weight``.
* **Flexible Output:** Features a :class:`~fusionlab.nn.components.MultiDecoder`
  (generating horizon-specific features) and
  :class:`~fusionlab.nn.components.QuantileDistributionModeling` layer
  to produce multi-horizon forecasts for specified ``quantiles``
  (or point forecasts if ``quantiles`` is ``None``).

**When to Use:**

XTFT is designed for challenging forecasting problems where:

* Underlying temporal dynamics are highly complex and potentially
  span **multiple time scales**.
* Rich static, dynamic, and future information needs to be
  **integrated effectively** using advanced fusion techniques.
* Capturing **long-range dependencies** is important (leveraging memory
  attention).
* Identifying or accounting for **anomalies** within the time series is
  a requirement.
* **Maximum predictive performance** is desired, potentially at the cost
  of increased model complexity and computational resources compared
  to standard TFT.

Formulation
~~~~~~~~~~~~~~

XTFT significantly extends the standard TFT architecture. While it
builds upon core concepts like GRNs and attention, it introduces
many specialized components. We highlight the key additions and
modifications here. For full details, please refer to the source code
and the documentation of individual components (linked above).

1.  **Input Processing:**

    * Static inputs (:math:`s`) undergo :class:`~fusionlab.nn.components.LearnedNormalization` 
      and are processed by internal GRNs/Dense layers (`static_dense`,
      `static_dropout`, `grn_static`).
    * Dynamic (:math:`x_t`) and Future (:math:`z_t`) inputs are jointly
      processed by :class:`~fusionlab.nn.components.MultiModalEmbedding`.
    * :class:`~fusionlab.nn.components.PositionalEncoding` is added.
    * Optional residual connections enhance gradient flow.

2.  **Multi-Scale LSTM:**

    * Dynamic inputs (:math:`x_t` or embeddings derived from them) are
      processed by :class:`~fusionlab.nn.components.MultiScaleLSTM` using
      different temporal ``scales``.
    * Outputs are aggregated (e.g., 'last' step) into `lstm_features`.

3.  **Advanced Attention Layers:**

    * :class:`~fusionlab.nn.components.HierarchicalAttention` processes dynamic and future inputs.
    * :class:`~fusionlab.nn.components.CrossAttention` models interactions between dynamic inputs and combined embeddings.
    * :class:`~fusionlab.nn.components.MemoryAugmentedAttention` uses
      hierarchical attention output to query an external memory.
    * GRNs are applied after each attention block (`grn_attention_*`).

4.  **Feature Fusion:**

    * Processed static features, aggregated `lstm_features`, and outputs
    from the various attention mechanisms are concatenated.
    * :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`
    is applied to integrate these diverse feature streams.

5.  **Dynamic Windowing & Aggregation:**

    * :class:`~fusionlab.nn.components.DynamicTimeWindow` selects recent
      time steps from the fused features.
    * :func:`~fusionlab.nn.components.aggregate_time_window_output`
      collapses the time dimension based on `final_agg` strategy.

6.  **Decoding and Output:**

    * :class:`~fusionlab.nn.components.MultiDecoder` transforms the aggregated features for each horizon step.
    * A final GRN pipeline (`grn_decoder`) processes decoder outputs.
    * :class:`~fusionlab.nn.components.QuantileDistributionModeling` maps
      these features to the final quantile or point predictions
      (:math:`\hat{y}_{t, q}` / :math:`\hat{y}_t`).

7.  **Anomaly Detection Integration:**

    * **Feature-Based:** Internal `anomaly_attention`, `anomaly_projection`,
      and `anomaly_scorer` layers compute `anomaly_scores` during the forward pass.
    * **Config-Based:** Pre-computed `anomaly_scores` are provided via `anomaly_config`.
    * **Loss Calculation:** If `anomaly_scores` exist,
      :class:`~fusionlab.nn.components.AnomalyLoss` calculates an anomaly term,
      which is added via ``model.add_loss`` (used in feature/config modes).
    * **Prediction-Based:** A specialized combined loss function is used
      during `compile`, and the custom `train_step` handles calculations.

**Code Example (Instantiation):**

.. code-block:: python
   :linenos:

   import numpy as np
   # Assuming XTFT is importable
   from fusionlab.nn.transformers import XTFT

   # Example Configuration
   static_dim, dynamic_dim, future_dim = 5, 7, 3
   horizon = 12
   output_dim = 1
   my_quantiles = [0.1, 0.5, 0.9]
   my_scales = [1, 3, 6] # Example scales for MultiScaleLSTM

   # Instantiate XTFT with various parameters
   xtft_model = XTFT(
       static_input_dim=static_dim,
       dynamic_input_dim=dynamic_dim,
       future_input_dim=future_dim,
       forecast_horizon=horizon,
       quantiles=my_quantiles,
       output_dim=output_dim,
       embed_dim=16,
       hidden_units=32,
       attention_units=16,
       lstm_units=32,
       num_heads=4,
       scales=my_scales,
       multi_scale_agg='last', # Aggregation for MultiScaleLSTM
       memory_size=50,
       max_window_size=24, # For DynamicTimeWindow
       final_agg='average', # Aggregation after DynamicTimeWindow
       anomaly_detection_strategy='prediction_based', # Example strategy
       anomaly_loss_weight=0.05,
       dropout_rate=0.1
   )

   # Build the model (e.g., by providing dummy input shapes)
   # Note: Actual shapes depend on data preprocessing
   dummy_batch_size = 4
   dummy_time_steps = 24 # Should match or exceed max_window_size

   # Example shapes (adjust T_future as needed)
   static_shape = (dummy_batch_size, static_dim)
   dynamic_shape = (dummy_batch_size, dummy_time_steps, dynamic_dim)
   future_shape = (dummy_batch_size, dummy_time_steps + horizon, future_dim)

   # Build using dummy shapes (or use model.fit/predict later)
   # xtft_model.build(input_shape=[static_shape, dynamic_shape, future_shape])
   # print("XTFT Model Built (example).")

   xtft_model.summary() # Display model architecture summary (after build)

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


.. _superxtft_model:

SuperXTFT
-----------
:API Reference: :class:`~fusionlab.nn.transformers.SuperXTFT`

.. warning::
   ``SuperXTFT`` is currently considered **experimental** and may be
   subject to significant changes or removal in future versions.
   It is **not recommended for production use** at this time. Please
   use the standard :class:`~fusionlab.nn.XTFT` for stable
   deployments.

The ``SuperXTFT`` class inherits from :class:`~fusionlab.nn.XTFT` and
introduces specific architectural modifications aimed at potentially
enhancing feature representation and the internal processing flow.

**Key Features & Differences (from XTFT):**

* **Inherits XTFT Features:** Includes all the advanced components
  and capabilities of the base ``XTFT`` model (Multi-Scale LSTM,
  advanced attention, anomaly detection capabilities, etc.).
* **Adds Input Variable Selection Networks (VSNs):** Unlike ``XTFT``
  which processes inputs via embeddings/normalization first,
  ``SuperXTFT`` re-introduces VSNs applied directly to the *raw*
  static, dynamic (past), and future inputs at the beginning of
  the forward pass. The outputs of these VSNs (selected/weighted
  features) are then fed into the subsequent stages inherited from
  the XTFT architecture.
* **Adds Post-Processing GRNs:** Integrates dedicated
  :class:`~fusionlab.nn.components.GatedResidualNetwork` (GRN)
  layers immediately following several key attention/decoder
  components (Hierarchical Attention, Cross Attention,
  Memory-Augmented Attention, Multi-Decoder). These apply further
  non-linear processing to the outputs of these specific stages.

**When to Use:**

* **Currently:** Primarily for internal development, testing, or
  research purposes within the ``fusionlab`` project itself due to
  its experimental status.
* **Future:** Intended as a potentially enhanced alternative to
  ``XTFT`` once development stabilizes.
* **Avoid for production or general use until officially recommended.**

Formulation
~~~~~~~~~~~~~

``SuperXTFT`` modifies the data flow of the base ``XTFT`` model in
two main ways:

1.  **Input Variable Selection:**
    Inputs (:math:`s, x_t, z_t`) are first processed through dedicated
    :class:`~fusionlab.nn.components.VariableSelectionNetwork` layers
    *before* subsequent XTFT components like normalization or embedding.

    .. math::
       s' = VSN_{static}(s) \\
       x'_t = VSN_{dynamic}(x_t) \\
       z'_t = VSN_{future}(z_t)

    These *selected* features (:math:`s', x'_t, z'_t`) then replace the
    original inputs in the downstream XTFT pipeline (e.g., :math:`s'`
    goes to Learned Normalization, :math:`x'_t` / :math:`z'_t` go to
    MultiModal Embedding).

2.  **Integrated Post-Processing GRNs:**
    After specific intermediate outputs (:math:`Attn_{...}` or
    :math:`Dec_{out}`) are computed within the main XTFT flow,
    ``SuperXTFT`` applies an additional GRN transformation before the
    result is used in subsequent steps.

    .. math::
       Output'_{component} = GRN_{component}(Output_{component})

    This adds extra non-linear processing within the architecture.

These modifications aim to potentially improve feature selection and
refine representations, but require further validation.

**Code Example (Instantiation Only):**

*(Note: Due to the experimental status, only instantiation is shown.
Use with caution.)*

.. code-block:: python
   :linenos:

   import numpy as np
   # Assuming SuperXTFT is importable
   from fusionlab.nn.transformers import SuperXTFT

   # Example Configuration (must provide all required dims)
   static_dim, dynamic_dim, future_dim = 5, 7, 3
   horizon = 12
   output_dim = 1

   # Instantiate SuperXTFT
   # Uses the same parameters as XTFT
   try:
       super_xtft_model = SuperXTFT(
           static_input_dim=static_dim,
           dynamic_input_dim=dynamic_dim,
           future_input_dim=future_dim,
           forecast_horizon=horizon,
           output_dim=output_dim,
           hidden_units=32, # Example other params
           num_heads=4
       )
       print("SuperXTFT model instantiated successfully.")
       # super_xtft_model.summary() # Can view summary after building
   except Exception as e:
       print(f"Error instantiating SuperXTFT: {e}")


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


.. rubric:: References

.. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
   Temporal fusion transformers for interpretable multi-horizon
   time series forecasting. *International Journal of Forecasting*,
   37(4), 1748-1764. (Also arXiv:1912.09363)
