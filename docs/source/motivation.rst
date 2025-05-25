.. _motivation:

============
Motivation
============

Time series forecasting is fundamental across countless domains, yet
predicting complex real-world systems remains a significant challenge.
From urban planning scenarios like land subsidence monitoring in rapidly
developing areas [1]_ to financial modeling and resource management,
decision-makers increasingly require forecasts that are not only
accurate but also provide reliable estimates of uncertainty.

The Advent of Transformers
----------------------------
The landscape of sequence modeling was revolutionized by Transformer
architectures [2]_, initially excelling in natural language
processing. Their adaptation to time series, notably through models like
the **Temporal Fusion Transformer (TFT)** [3]_, marked a major step
forward. TFT introduced powerful mechanisms for multi-horizon
forecasting by integrating static metadata, dynamic historical inputs,
and known future covariates using specialized gating and attention layers
[4]_.

Persistent Challenges in Forecasting
--------------------------------------
Despite these advancements, several critical challenges hinder the
development and deployment of truly robust and interpretable forecasting
systems, particularly for complex spatiotemporal or multivariate data:

1.  **Multiscale Temporal Dynamics:** Real-world processes often exhibit
    patterns across vastly different timescales (e.g., daily fluctuations,
    weekly cycles, annual seasonality). Standard architectures frequently
    struggle to capture these interacting dynamics simultaneously and
    efficiently [5]_. While hierarchical or multiresolution
    models exist [6]_, [7]_, they often add
    complexity [8]_.
2.  **Heterogeneous Data Fusion:** Integrating diverse data types—static
    attributes, time-varying historical data (potentially with varying
    sampling rates), and known future inputs—remains complex. Achieving
    synergy between these modalities, rather than simple concatenation,
    is often difficult, especially when semantic contexts differ
    [9]_, [10]_.
3.  **Actionable Uncertainty Quantification:** Many advanced models still
    prioritize point forecast accuracy over providing reliable and
    well-calibrated uncertainty estimates (e.g., prediction intervals via
    quantiles). For high-stakes decisions (like geohazard mitigation or
    financial risk assessment), understanding the *range* of possible
    outcomes is paramount, yet often inadequately addressed
    [11]_, [12]_.
4.  **Interpretability and Scalability:** As models become more complex
    to handle intricate data, maintaining interpretability (understanding
    *why* a prediction was made) and ensuring scalability to large
    datasets become increasingly challenging [9]_, [13]_.

The FusionLab Vision: Addressing the Gaps
---------------------------------------------
``fusionlab-learn`` was born from the need to address these persistent gaps.
Motivated by complex real-world forecasting problems, such as
understanding the uncertainty in **land subsidence predictions** for
urban planning [1]_, we aim to provide a framework for building,
experimenting with, and deploying next-generation temporal fusion models.

Our core philosophy is **modularity and targeted enhancement**. We provide
reusable, well-defined components alongside advanced, pre-configured models
like :class:`~fusionlab.nn.XTFT` (Extreme Temporal Fusion Transformer) that
specifically incorporate features designed to tackle the challenges above:

* **Multi-Scale Processing:** Incorporating components like
  :class:`~fusionlab.nn.components.MultiScaleLSTM` to analyze temporal
  patterns at different resolutions.
* **Advanced Fusion & Attention:** Employing sophisticated attention mechanisms
  (like those in :class:`~fusionlab.nn.XTFT`) to better integrate
  heterogeneous inputs and capture complex dependencies.
* **Probabilistic Focus:** Natively supporting multi-horizon quantile
  forecasting to treat uncertainty not just as noise, but as a critical
  output signal.
* **Integrated Capabilities:** Building in features like anomaly detection
  within the forecasting pipeline itself.
* **Extensibility:** Providing a foundation (currently based on
  TensorFlow/Keras) for researchers and practitioners to easily
  experiment with new ideas and build custom model variants.

Ultimately, ``fusionlab-learn`` strives to facilitate the development of more
robust, interpretable, and uncertainty-aware forecasting solutions for
complex, real-world time series challenges.

.. rubric:: References

.. [1] Liu, J., Liu, W., Allechy, F. B., Zheng, Z., Liu, R.,
       & Kouadio, K. L. (2024). Machine learning-based techniques for
       land subsidence simulation in an urban area. *Journal of
       Environmental Management*, 352, 120078.
.. [2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
       Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
       *Attention is all you need*. Advances in Neural Information
       Processing Systems, 30.
.. [3] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
       Temporal fusion transformers for interpretable multi-horizon
       time series forecasting. *International Journal of Forecasting*,
       37(4), 1748-1764. (Also arXiv:1912.09363)
.. [4] Liu, L., Wang, X., Dong, X., Chen, K., Chen, Q.,
       & Li, B. (2024). Interpretable feature-temporal transformer for
       short-term wind power forecasting with multivariate time series.
       *Applied Energy*, 374, 124035.
.. [5] Hittawe, M. M., Harrou, F., Togou, M. A., Sun, Y.,
       & Knio, O. (2024). Time-series weather prediction in the Red sea
       using ensemble transformers. *Applied Soft Computing*, 164, 111926.
.. [6] Huang, X., Wu, D., & Boulet, B. (2023).
       Metaprobformer for charging load probabilistic forecasting of
       electric vehicle charging stations. *IEEE Transactions on
       Intelligent Transportation Systems*, 24(10), 10445-10455.
.. [7] Shu, M., Chen, G., Zhang, Z., & Xu, L. (2022). Indoor
       geomagnetic positioning using direction-aware multiscale recurrent
       neural networks. *IEEE Sensors Journal*, 23(3), 3321-3333.
.. [8] Deihim, A., Alonso, E., & Apostolopoulou, D. (2023).
       STTRE: A Spatio-Temporal Transformer with Relative Embeddings for
       multivariate time series forecasting. *Neural Networks*, 168,
       549-559.
.. [9] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are
       transformers effective for time series forecasting?. *AAAI
       Conference on Artificial Intelligence*, 37(9), 11121-11128.
.. [10] Peruzzo, E., Sangineto, E., Liu, Y., De Nadai, M.,
        Bi, W., Lepri, B., & Sebe, N. (2024). Spatial entropy as an
        inductive bias for vision transformers. *Machine Learning*,
        113(9), 6945-6975.
.. [11] Xu, C., Li, J., Feng, B., & Lu, B. (2023). A financial
        time-series prediction model based on multiplex attention and
        linear transformer structure. *Applied Sciences*, 13(8), 5175.
.. [12] Wu, N., Green, B., Ben, X., & O'Banion, S. (2022).
        Interpretable Deep Learning for Time Series Forecasting:
        Taxonomy, Methods, and Challenges. *arXiv preprint arXiv:2201.13010*.
.. [13] Chen, Z., Ma, M., Li, T., Wang, H., & Li, C. (2023).
        Long sequence time-series forecasting with deep learning: A survey.
        *Information Fusion*, 97, 101819.