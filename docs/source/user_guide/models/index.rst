.. _models_guide:

====================
Forecasting Models
====================

Welcome to the main models section of the ``fusionlab-learn`` user
guide. This section provides detailed documentation for the advanced
time series forecasting architectures available in the library.

The models are categorized based on their core architectural
principles, from purely data-driven engines to sophisticated hybrids
that integrate physical laws.

* **Hybrid Models** combine the strengths of Recurrent Neural Networks
  (LSTMs) for sequential processing with the power of Transformers
  for capturing long-range dependencies.

* **Physics-Informed Neural Networks (PINNs)** represent a cutting-edge
  approach, fusing data-driven models with the governing equations
  of physical systems to produce more robust and consistent results.

* **Transformer-Based Models** leverage the attention mechanism as
  their core component, including both pure transformer architectures
  and variants of the influential Temporal Fusion Transformer (TFT).

The ``BaseAttentive`` model serves as the powerful, modular foundation
for many of the advanced hybrid and PINN architectures.

Please select a category below to explore the available models.

.. toctree::
   :maxdepth: 2
   :caption: Model Categories:

   base_attentive
   hybrid/index
   pinn/index
   transformers/index