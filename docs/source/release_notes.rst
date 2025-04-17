.. _release_notes:

===============
Release Notes
===============

This document tracks the history of changes, new features, and bug
fixes for each release of the ``fusionlab`` library.

---
Version 0.1.0
---
*(Release Date: April 17, 2025)* *(Note: Update with actual date)*

**Initial Public Release**

This is the first public release of ``fusionlab``, establishing the
core framework for building and experimenting with advanced time
series forecasting models based on Temporal Fusion Transformer
architectures.

**✨ Key Features & Modules Included:**

* **Core Forecasting Models (`fusionlab.nn.transformers`):**
    * :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`:
        A flexible implementation of the standard TFT model.
    * :class:`~fusionlab.nn.transformers.XTFT`: The Extreme Temporal
        Fusion Transformer, featuring enhanced components for complex
        time series, including multi-scale processing, advanced
        attention, and integrated anomaly detection capabilities.
    * :class:`~fusionlab.nn.transformers.NTemporalFusionTransformer`:
        A variant requiring static/dynamic inputs (point forecast only).
    * :class:`~fusionlab.nn.transformers.SuperXTFT`: An experimental
        variant of XTFT (currently marked as deprecated/experimental).

* **Modular Components (`fusionlab.nn.components`):**
    * Core building blocks:
        :class:`~fusionlab.nn.components.GatedResidualNetwork`,
        :class:`~fusionlab.nn.components.VariableSelectionNetwork`,
        :class:`~fusionlab.nn.components.PositionalEncoding`.
    * Sequence processing:
        :class:`~fusionlab.nn.components.MultiScaleLSTM`,
        :class:`~fusionlab.nn.components.DynamicTimeWindow`,
        :func:`~fusionlab.nn.components.aggregate_multiscale`,
        :func:`~fusionlab.nn.components.aggregate_time_window_output`.
    * Attention mechanisms:
        :class:`~fusionlab.nn.components.TemporalAttentionLayer`,
        :class:`~fusionlab.nn.components.CrossAttention`,
        :class:`~fusionlab.nn.components.HierarchicalAttention`,
        :class:`~fusionlab.nn.components.MemoryAugmentedAttention`,
        :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`,
        :class:`~fusionlab.nn.components.ExplainableAttention`.
    * Input/Output layers:
        :class:`~fusionlab.nn.components.MultiModalEmbedding`,
        :class:`~fusionlab.nn.components.LearnedNormalization`,
        :class:`~fusionlab.nn.components.MultiDecoder`,
        :class:`~fusionlab.nn.components.QuantileDistributionModeling`.

* **Loss Functions (`fusionlab.nn.losses`, `fusionlab.nn.components`):**
    * Support for point forecasting (MSE) and quantile forecasting
        (Pinball/Quantile Loss) via factory functions like
        :func:`~fusionlab.nn.losses.combined_quantile_loss`.
    * Components and factories for combined objectives including anomaly
        scores: :class:`~fusionlab.nn.components.AnomalyLoss`,
        :class:`~fusionlab.nn.components.MultiObjectiveLoss`,
        :func:`~fusionlab.nn.losses.prediction_based_loss`,
        :func:`~fusionlab.nn.losses.combined_total_loss`.

* **Anomaly Detection (`fusionlab.nn.anomaly_detection`):**
    * Initial components for unsupervised and feature-based anomaly
        detection:
        :class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`,
        :class:`~fusionlab.nn.anomaly_detection.SequenceAnomalyScoreLayer`.

* **Hyperparameter Tuning (`fusionlab.nn.forecast_tuner`):**
    * Utilities (:func:`~fusionlab.nn.forecast_tuner.xtft_tuner`,
        :func:`~fusionlab.nn.forecast_tuner.tft_tuner`) leveraging
        `keras-tuner` for automated hyperparameter search.

* **Utilities (`fusionlab.utils`, `fusionlab.nn.utils`):**
    * Time series specific helpers (`ts_utils`) for validation,
        feature engineering (:func:`~fusionlab.utils.ts_utils.ts_engineering`),
        analysis (:func:`~fusionlab.utils.ts_utils.trend_analysis`,
        :func:`~fusionlab.utils.ts_utils.decompose_ts`, etc.), outlier
        detection, and splitting.
    * Neural network specific helpers (`nn.utils`) for sequence
        preparation (:func:`~fusionlab.nn.utils.reshape_xtft_data`,
        :func:`~fusionlab.nn.utils.create_sequences`), prediction input
        generation (:func:`~fusionlab.nn.utils.prepare_spatial_future_data`),
        forecasting execution (:func:`~fusionlab.nn.utils.generate_forecast`),
        and visualization (:func:`~fusionlab.nn.utils.visualize_forecasts`).

* **Tools (`fusionlab.tools`):**
    * Initial command-line applications for running XTFT/TFT workflows
        (e.g., `tft_cli.py`, `xtft_proba_app.py`).

* **Documentation:**
    * Initial setup of Sphinx documentation including User Guide,
        Examples, API Reference, and Glossary.

**⚠️ Breaking Changes:**

* Initial release. No breaking changes from previous versions.

**❗ Known Issues / Limitations:**

* :class:`~fusionlab.nn.SuperXTFT` is experimental and currently
    marked as deprecated.
* Backend support is currently focused on TensorFlow/Keras.
* Some utility functions might require specific optional dependencies
    (e.g., `statsmodels`).

**Contributors:**

* earthai-tech (Lead Developer: Laurent Kouadio)
