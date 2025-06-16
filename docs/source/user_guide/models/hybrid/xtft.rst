.. _hybrid_transformer_models_guide:

==============================================
Hybrid Transformer Models: XTFT & SuperXTFT
==============================================

This section of the user guide covers the ``XTFT`` (Extreme Temporal
Fusion Transformer) family of models. These are advanced, hybrid
architectures designed for the most demanding multi-horizon time
series forecasting tasks.

Building upon the foundational concepts of the original Temporal Fusion
Transformer (TFT), these models integrate multi-scale recurrent
processing using LSTMs with a sophisticated, multi-layered attention
framework. This hybrid approach allows them to capture an exceptionally
rich set of temporal patterns, from short-term dependencies to very
long-range, complex interactions.

This guide details two models in this family:

* **XTFT**: The main, stable implementation, which includes numerous
  enhancements over the standard TFT, such as advanced attention
  mechanisms and integrated anomaly detection.
* **SuperXTFT**: An experimental variant of ``XTFT`` that introduces
  additional feature selection and processing layers.

.. toctree::
   :hidden:


.. _xtft_model:

XTFT (Extreme Temporal Fusion Transformer)
--------------------------------------------
:API Reference: :class:`~fusionlab.nn.models.XTFT`

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

When to Use XTFT
~~~~~~~~~~~~~~~~~~
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
   from fusionlab.nn.models import XTFT

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

SuperXTFT: An Enhanced Hybrid Transformer
------------------------------------------
:API Reference: :class:`~fusionlab.nn.models.SuperXTFT`

The ``SuperXTFT`` is the most advanced and powerful implementation in the
TFT family available in ``fusionlab-learn``. It inherits the entire
robust feature set of the standard :class:`~fusionlab.nn.transformers.XTFT`
and enhances it with two significant architectural modifications, designed
to maximize representation learning and predictive performance.

It should be considered the expert choice for tackling the most complex
forecasting problems where fine-grained feature selection and deep
contextual processing are paramount.

Key Architectural Enhancements (from XTFT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``SuperXTFT`` improves upon the standard ``XTFT`` architecture in two
primary ways:

**1. Integrated Input Variable Selection (VSNs)**

Unlike the standard ``XTFT`` which processes inputs directly into
embeddings, ``SuperXTFT`` first passes all three raw input streams
(static, dynamic past, and future) through their own dedicated
:class:`~fusionlab.nn.components.VariableSelectionNetwork` (VSN) layers.

.. math::
   \mathbf{s}' = VSN_{static}(\mathbf{s}) \\
   \mathbf{x}'_t = VSN_{dynamic}(\mathbf{x}_t) \\
   \mathbf{z}'_t = VSN_{future}(\mathbf{z}_t)

**Benefit:** This allows the model to learn the relative importance
of each input feature *at the very beginning* of the pipeline, before
they are mixed and processed by downstream components. This can lead to
more robust and interpretable feature representations, especially in
datasets with a large number of potentially redundant or noisy features.
The selected features (:math:`\mathbf{s}', \mathbf{x}'_t, \mathbf{z}'_t`)
are then fed into the rest of the standard XTFT architecture.

**2. Post-Component Gated Processing (GRNs)**

``SuperXTFT`` strategically inserts additional
:class:`~fusionlab.nn.components.GatedResidualNetwork` (GRN) layers
immediately after each major attention and decoder block. A GRN is
applied to the outputs of:

* Hierarchical Attention
* Cross-Attention
* Memory-Augmented Attention
* The Multi-Decoder layer

.. math::
   Output'_{component} = GRN_{component}(Output_{component})

**Benefit:** This adds another layer of deep, non-linear processing
and feature gating at critical junctures within the architecture. It
allows the model to further refine the contextual representations
generated by each attention mechanism before they are fused together,
potentially capturing more complex and subtle interactions.

When to Use SuperXTFT 
~~~~~~~~~~~~~~~~~~~~~~~~
``SuperXTFT`` is the recommended choice for challenging forecasting
problems where:

* You are working with a **large number of input features** of
  varying importance and want the model to learn which ones to
  prioritize (leveraging the input VSNs).
* You hypothesize that there are **complex, non-linear interactions**
  between the different contexts (static, temporal, memory) that
  could benefit from the additional deep processing offered by the
  post-component GRNs.
* You are aiming for **maximum predictive performance** and have the
  computational resources for a deeper, more parameter-rich model.
* For standard use cases, the :class:`~fusionlab.nn.transformers.XTFT`
  remains a powerful and efficient baseline.

Code Example
~~~~~~~~~~~~

The instantiation of ``SuperXTFT`` is identical to ``XTFT``. The
additional VSN and GRN layers are created and integrated automatically
within the model's constructor and forward pass.

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.nn.models import SuperXTFT

   # Configuration is the same as for XTFT
   static_dim, dynamic_dim, future_dim = 5, 7, 3
   horizon = 12
   output_dim = 1

   # Instantiate the SuperXTFT model
   super_xtft_model = SuperXTFT(
       static_input_dim=static_dim,
       dynamic_input_dim=dynamic_dim,
       future_input_dim=future_dim,
       forecast_horizon=horizon,
       output_dim=output_dim,
       # Other architectural parameters
       hidden_units=32,
       num_heads=4,
       lstm_units=32,
       attention_units=16
   )

   print("SuperXTFT model instantiated successfully.")
   # You can view the deeper architecture with .summary() after building
   # super_xtft_model.summary(line_length=110)


.. note::

   ``SuperXTFT`` is a production-ready model and represents the most
   powerful, feature-rich version in the TFT family.

   It also serves as a development platform where new, cutting-edge
   features may be introduced first. Future releases might include
   experimental options aimed at lightening the architecture or
   improving computational efficiency. While any such new features may be
   subject to change, the core ``SuperXTFT`` architecture is stable,
   has been thoroughly tested, and can be confidently used in
   production environments where maximum performance is desired.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Next Steps
----------

.. note::

   You now have a deep understanding of the theory and architecture
   of the ``XTFT`` and ``SuperXTFT`` models. To apply these concepts,
   you can proceed to the hands-on exercises:

   * :doc:`../../exercises/exercise_advanced_xtft`
   * :doc:`../../exercises/exercise_experimental_super_tft`
   

.. rubric:: References

.. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
   Temporal fusion transformers for interpretable multi-horizon
   time series forecasting. *International Journal of Forecasting*,
   37(4), 1748-1764. (Also arXiv:1912.09363)