.. _user_guide_introduction:

==============
Introduction
==============

Welcome to the ``fusionlab`` user guide! This section provides a
conceptual overview of the models and techniques implemented in the
library, focusing on advanced time series forecasting.

The Challenge of Time Series Forecasting
------------------------------------------

Predicting the future based on the past is a fundamental challenge
across many domains, from finance and retail to weather and IoT.
Real-world time series data often presents significant hurdles:

* **Complex Patterns:** Data can exhibit intricate seasonality,
    trends, and irregular cycles that are hard to model.
* **Multiple Inputs:** Effective forecasting often requires using
    various types of features:
    * *Past Values:* Historical target values and covariates.
    * *Known Future Inputs:* Events or values known in advance
        (e.g., holidays, promotions).
    * *Static Metadata:* Time-invariant features (e.g., store ID,
        sensor location).
* **Multi-Horizon Needs:** Often, predictions are needed for
    multiple steps into the future, not just the next one.
* **Uncertainty:** Providing not just a single point forecast, but
    also an estimate of the prediction uncertainty (prediction
    intervals) is crucial for decision-making.

Enter the Temporal Fusion Transformer (TFT)
---------------------------------------------

The Temporal Fusion Transformer (TFT) [1]_ was a significant
advancement designed specifically to address these challenges. It
combines ideas from sequence-to-sequence learning (like LSTMs and
attention) with specialized components for time series:

* **Gated Residual Networks (GRNs):** Flexible blocks for
    processing features.
* **Variable Selection Networks (VSNs):** To identify and weight
    the importance of different input features (static, past,
    future).
* **Static Enrichment:** Mechanisms to effectively incorporate static
    metadata into the temporal dynamics.
* **Temporal Self-Attention:** To learn long-range dependencies
    across time steps, inspired by Transformers, but adapted for
    time series interpretability.
* **Multi-Horizon Forecasting:** Directly outputs predictions for
    multiple future steps simultaneously.
* **Quantile Regression:** Natively supports predicting multiple
    quantiles to estimate uncertainty intervals.

``fusionlab``: Beyond the Standard TFT
----------------------------------------

While TFT provides a powerful baseline, real-world data can push
its limits. ``fusionlab`` provides robust implementations of TFT
and goes further by offering **XTFT (Extreme Temporal Fusion
Transformer)**, an enhanced architecture designed for even greater
complexity and performance.

XTFT builds upon TFT by incorporating several advanced modules:

* **Enhanced Attention:** Uses more sophisticated attention layers
    like Hierarchical Attention, Cross-Attention, and
    Memory-Augmented Attention to capture intricate relationships
    across different inputs and time scales.
* **Multi-Scale Processing:** Employs techniques like Multi-Scale
    LSTMs and Multi-Resolution Attention Fusion to analyze temporal
    patterns at different frequencies (e.g., daily, weekly).
* **Dynamic Time Windowing:** Adapts the focus on recent history
    dynamically.
* **Integrated Anomaly Detection:** Includes mechanisms to identify
    and optionally incorporate information about anomalies directly
    into the forecasting process.

Modularity at its Core
------------------------

A key philosophy of ``fusionlab`` is **modularity**. Both TFT and
XTFT are constructed from reusable building blocks (available in
:doc:`components`). This design allows researchers and practitioners
to:

* Understand the contribution of each part of the model.
* Easily experiment by swapping or modifying components.
* Build custom variants tailored to specific problems.

Next Steps
------------

Now that you have a conceptual overview, you can explore:

* :doc:`models`: Details on the specific model classes available
    (``TemporalFusionTransformer``, ``XTFT``, etc.).
* :doc:`components`: A closer look at the building blocks used
    within the models.
* Or dive into the :doc:`/quickstart` for a hands-on example.

References
------------

.. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2019).
   Temporal Fusion Transformers for Interpretable Multi-horizon
   Time Series Forecasting. arXiv preprint arXiv:1912.09363.
   https://arxiv.org/abs/1912.09363