.. _hybrid_models_guide:

===============
Hybrid Models
===============

Welcome to the Hybrid Models section of the ``fusionlab-learn`` user
guide. The models detailed here represent a powerful architectural
paradigm that combines the strengths of different deep learning
components to tackle complex time series forecasting tasks.

Specifically, these "hybrid" models fuse the sequential processing
capabilities of **Recurrent Neural Networks (LSTMs)** with the
sophisticated context-modeling power of **Transformers and their
attention mechanisms**.

This approach allows the models to:
* Capture short-term temporal dependencies and sequential patterns
    effectively using LSTMs.
* Model long-range dependencies and complex feature interactions
    using multi-headed attention.
* Integrate diverse data sources (static, dynamic past, and known
    future) into a cohesive and rich representation.

This section provides detailed guides for each of the hybrid models
available in the library.

.. toctree::
   :maxdepth: 2
   :caption: Hybrid Models:

   halnet
   xtft