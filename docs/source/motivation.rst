.. _motivation:

======================
Motivation and Genesis
======================

Time series forecasting is fundamental across countless domains, yet
predicting complex real-world systems remains a significant challenge.
From urban planning scenarios like land subsidence monitoring in rapidly
developing areas [Liu24]_ to financial modeling and resource management,
decision-makers increasingly require forecasts that are not only
accurate but also provide reliable estimates of uncertainty.

The Advent of Transformers
--------------------------
The landscape of sequence modeling was revolutionized by Transformer
architectures [Vaswani17]_, initially excelling in natural language
processing. Their adaptation to time series, notably through models like
the **Temporal Fusion Transformer (TFT)** [Lim21]_, marked a major step
forward. TFT introduced powerful mechanisms for multi-horizon
forecasting by integrating static metadata, dynamic historical inputs,
and known future covariates using specialized gating and attention layers
[liu2024interpretable]_.

Persistent Challenges in Forecasting
------------------------------------
Despite these advancements, several critical challenges hinder the
development and deployment of truly robust and interpretable forecasting
systems, particularly for complex spatiotemporal or multivariate data:

1.  **Multiscale Temporal Dynamics:** Real-world processes often exhibit
    patterns across vastly different timescales (e.g., daily fluctuations,
    weekly cycles, annual seasonality). Standard architectures frequently
    struggle to capture these interacting dynamics simultaneously and
    efficiently [hittawe2024time]_. While hierarchical or multiresolution
    models exist [huang2023metaprobformer]_, [shu2022indoor]_, they often add
    complexity [deihim2023sttre]_.
2.  **Heterogeneous Data Fusion:** Integrating diverse data types—static
    attributes, time-varying historical data (potentially with varying
    sampling rates), and known future inputs—remains complex. Achieving
    synergy between these modalities, rather than simple concatenation,
    is often difficult, especially when semantic contexts differ
    [Zeng23a]_, [peruzzo2024spatial]_.
3.  **Actionable Uncertainty Quantification:** Many advanced models still
    prioritize point forecast accuracy over providing reliable and
    well-calibrated uncertainty estimates (e.g., prediction intervals via
    quantiles). For high-stakes decisions (like geohazard mitigation or
    financial risk assessment), understanding the *range* of possible
    outcomes is paramount, yet often inadequately addressed
    [Xu23]_, [wu2022interpretable]_.
4.  **Interpretability and Scalability:** As models become more complex
    to handle intricate data, maintaining interpretability (understanding
    *why* a prediction was made) and ensuring scalability to large
    datasets become increasingly challenging [Zeng23a]_, [Chen23]_.

The FusionLab Vision: Addressing the Gaps
---------------------------------------------
``fusionlab`` was born from the need to address these persistent gaps.
Motivated by complex real-world forecasting problems, such as
understanding the uncertainty in **land subsidence predictions** for
urban planning [Liu24]_, we aim to provide a framework for building,
experimenting with, and deploying next-generation temporal fusion models.

Our core philosophy is **modularity and targeted enhancement**. We provide
reusable, well-defined components alongside advanced, pre-configured models
like :class:`~fusionlab.nn.XTFT` (Extreme Temporal Fusion Transformer) that
specifically incorporate features designed to tackle the challenges above:

* **Multi-Scale Processing:** Incorporating components like Multi-Scale LSTMs
    to analyze temporal patterns at different resolutions.
* **Advanced Fusion & Attention:** Employing sophisticated attention mechanisms
    (Hierarchical, Cross, Memory-Augmented, Multi-Resolution Fusion) to better
    integrate heterogeneous inputs and capture complex dependencies.
* **Probabilistic Focus:** Natively supporting multi-horizon quantile forecasting
    to treat uncertainty not just as noise, but as a critical output signal.
* **Integrated Capabilities:** Building in features like anomaly detection within
    the forecasting pipeline itself.
* **Extensibility:** Providing a foundation (currently based on TensorFlow/Keras)
    for researchers and practitioners to easily experiment with new ideas and
    build custom model variants.

Ultimately, ``fusionlab`` strives to facilitate the development of more robust,
interpretable, and uncertainty-aware forecasting solutions for complex,
real-world time series challenges.

References
----------

.. [Vaswani17] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
   Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
   *Attention is all you need*. Advances in Neural Information
   Processing Systems, 30.

.. [Lim21] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
   Temporal fusion transformers for interpretable multi-horizon
   time series forecasting. *International Journal of Forecasting*,
   37(4), 1748-1764. (Also arXiv:1912.09363)

.. [hittawe2024time] Hittawe, M. M., Harrou, F., Togou, M. A., Sun, Y.,
   & Knio, O. (2024). Time-series weather prediction in the Red sea
   using ensemble transformers. *Applied Soft Computing*, 164, 111926.
   *(Note: Check if this is the intended citation for multiscale challenges)*

.. [Zeng23a] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are
   transformers effective for time series forecasting?. *AAAI
   Conference on Artificial Intelligence*, 37(9), 11121-11128.

.. [peruzzo2024spatial] Peruzzo, E., Sangineto, E., Liu, Y., De Nadai, M.,
   Bi, W., Lepri, B., & Sebe, N. (2024). Spatial entropy as an
   inductive bias for vision transformers. *Machine Learning*,
   113(9), 6945-6975. *(Note: Cited in context of heterogeneous data)*

.. [Xu23] Xu, C., Li, J., Feng, B., & Lu, B. (2023). A financial
   time-series prediction model based on multiplex attention and
   linear transformer structure. *Applied Sciences*, 13(8), 5175.

.. [wu2022interpretable] Wu, N., Green, B., Ben, X., & O'Banion, S. (2022).
   Interpretable Deep Learning for Time Series Forecasting: Taxonomy,
   Methods, and Challenges. *arXiv preprint arXiv:2201.13010*.

.. [Chen23] Chen, Z., Ma, M., Li, T., Wang, H., & Li, C. (2023).
   Long sequence time-series forecasting with deep learning: A survey.
   *Information Fusion*, 97, 101819.

.. [Liu24] Liu, J., Liu, W., Allechy, F. B., Zheng, Z., Liu, R.,
   & Kouadio, K. L. (2024). Machine learning-based techniques for
   land subsidence simulation in an urban area. *Journal of
   Environmental Management*, 352, 120078.

.. [liu2024interpretable] Liu, L., Wang, X., Dong, X., Chen, K., Chen, Q.,
   & Li, B. (2024). Interpretable feature-temporal transformer for
   short-term wind power forecasting with multivariate time series.
   *Applied Energy*, 374, 124035. *(Note: Cited for TFT's data modality handling)*

.. [huang2023metaprobformer] Huang, X., Wu, D., & Boulet, B. (2023).
   Metaprobformer for charging load probabilistic forecasting of electric
   vehicle charging stations. *IEEE Transactions on Intelligent
   Transportation Systems*, 24(10), 10445-10455. *(Note: Cited as example
   of hierarchical transformer)*

.. [shu2022indoor] Shu, M., Chen, G., Zhang, Z., & Xu, L. (2022). Indoor
   geomagnetic positioning using direction-aware multiscale recurrent
   neural networks. *IEEE Sensors Journal*, 23(3), 3321-3333. *(Note: Cited
   as example of multiresolution RNN)*

.. [deihim2023sttre] Deihim, A., Alonso, E., & Apostolopoulou, D. (2023).
   STTRE: A Spatio-Temporal Transformer with Relative Embeddings for
   multivariate time series forecasting. *Neural Networks*, 168, 549-559.
   *(Note: Cited regarding multiscale interaction complexity)*

