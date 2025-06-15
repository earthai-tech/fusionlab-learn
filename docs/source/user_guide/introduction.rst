.. _user_guide_introduction:

===============
Introduction 
===============

Welcome to the ``fusionlab-learn`` user guide! This library is a
comprehensive, research-oriented toolkit for building, training, and
experimenting with advanced deep learning models for time series
forecasting.

This introduction provides a high-level overview of the challenges in
modern forecasting and the core philosophies and capabilities that
``fusionlab-learn`` offers to address them.

The Modern Forecasting Challenge
----------------------------------
Real-world time series data presents significant hurdles that go
beyond simple trend extrapolation:

* **Complex Temporal Patterns:** Data often exhibits a mix of intricate
  seasonality, long-term trends, and irregular, hard-to-modelcycles.
* **Heterogeneous Data Sources:** Effective forecasting requires fusing
  information from various input types:
    
  * **Dynamic Past Inputs:** Historical target values and observed covariates.
  * **Known Future Inputs:** Events or values known in advance,
    such as holidays, promotions, or weather forecasts.
  * **Static Metadata:** Time-invariant features that provide
    context, like a sensor's location or a product's category.
* **Multi-Horizon Requirements:** Predictions are often needed for an
  entire sequence of future steps, not just the very next one.
* **Quantifying Uncertainty:** A single point forecast is often
  insufficient. For robust decision-making, it's crucial to
  understand the forecast's uncertainty by generating prediction
  intervals.

Our Philosophy: A Unified and Modular Approach
------------------------------------------------
``fusionlab-learn`` is built on a philosophy of **modularity** and
**architectural diversity**.

1.  **A Spectrum of Architectures:** We believe there is no
    one-size-fits-all model. The library provides state-of-the-art
    implementations across the three main paradigms of modern deep
    learning for time series.

2.  **Modular Components:** The models are constructed from a rich set
    of reusable, interchangeable building blocks (available in
    :doc:`components`). This design allows researchers and
    practitioners to easily experiment with novel architectures and
    build custom models tailored to specific problems.

A Spectrum of Forecasting Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``fusionlab-learn`` provides expert implementations of three distinct
families of models, allowing you to choose the right tool for your task.

**1. Pure Transformer Models**
********************************
Based on the original "Attention Is All You Need" paper [1]_, these
models rely exclusively on self-attention and cross-attention
mechanisms. They excel at capturing very long-range dependencies and
complex inter-feature relationships without the inductive biases of
recurrent layers.

.. seealso::
   See the :doc:`models/transformers/index` guide for details.

**2. Hybrid Models**
**********************
These models, including the Temporal Fusion Transformer (TFT) [2]_
and its advanced successor ``XTFT``, represent a powerful fusion of
architectures. They combine the strengths of **Recurrent Neural
Networks (LSTMs)** for processing local sequential information with
the **global context** provided by transformer-based attention. This
makes them exceptionally powerful and robust general-purpose
forecasters.

.. seealso::
   See the :doc:`models/hybrid/index` guide for details.

**3. Physics-Informed Neural Networks (PINNs)**
**************************************************
This is the most advanced category, designed for scientific machine
learning. PINNs are hybrid models that are regularized by physical
laws. They integrate **Partial Differential Equations (PDEs)** into
the loss function, forcing the model to produce predictions that are
not only accurate with respect to data but also physically consistent.
This approach can dramatically improve generalization in data-scarce
environments and even allow for the discovery of physical parameters.

.. seealso::
   See the :doc:`models/pinn/index` guide for details.

Key Cross-Cutting Features
-----------------------------
Across these architectures, ``fusionlab-learn`` emphasizes a common set
of powerful features:

* **Multi-Step-Ahead Forecasting:** All primary models are designed as
  sequence-to-sequence architectures capable of producing
  multi-horizon forecasts in a single forward pass.
* **Probabilistic Outputs:** Native support for quantile regression
  allows models to output prediction intervals, providing a crucial
  measure of forecast uncertainty.
* **Flexible Input Handling:** A unified data pipeline allows all
  models to seamlessly handle static, dynamic past, and known future inputs.

Next Steps
------------
Now that you have a conceptual overview, we recommend you proceed to:

* :doc:`../installation` to set up your environment.
* :doc:`../quickstart` for a fast, hands-on example.
* :doc:`models/index` to take a deep dive into the specific model
  architectures.

References
----------
.. [1] Vaswani, A., et al. (2017). "Attention Is All You Need."
   *Advances in Neural Information Processing Systems 30*.
.. [2] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
   "Temporal Fusion Transformers for interpretable multi-horizon
   time series forecasting." *International Journal of Forecasting,
   37*(4), 1748-1764.