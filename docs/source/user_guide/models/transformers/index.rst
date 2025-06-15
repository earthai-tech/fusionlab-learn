.. _transformer_models_guide:

===========================
Transformer-Based Models
===========================

Welcome to the Transformer-Based Models section of the ``fusionlab-learn``
user guide. The models detailed here are all built upon the revolutionary
**attention mechanism**, which allows them to capture complex, long-range
dependencies in sequential data.

This powerful mechanism enables a model to dynamically weigh the
importance of different input features at different time steps, making
transformers exceptionally effective for a wide range of time series
forecasting tasks.

This section covers two main categories of transformer architectures available
in the library:

* **Pure Transformers**: These models, like the ``TimeSeriesTransformer``,
  adhere to the original "Attention Is All You Need" paradigm, relying
  exclusively on self-attention and cross-attention to process
  temporal information.

* **Temporal Fusion Transformers (TFTs)**: While these are technically
  hybrid models because they also use LSTMs, their core innovation lies
  in their specialized attention mechanisms and gating layers. They are
  grouped here due to their strong reliance on transformer principles.

Please select a model guide below to learn more.

.. toctree::
   :maxdepth: 2
   :caption: Transformer-Based Models:

   pure_transformers
   tft