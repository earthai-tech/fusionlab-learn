.. _api_reference:
.. default-role:: obj

===============
API Reference
===============

Welcome to the ``fusionlab-learn`` API reference. This section provides detailed
specifications for the public functions, classes, and modules included
in the package.

The documentation here is largely auto-generated from the docstrings
within the ``fusionlab-learn`` source code. For narrative explanations and usage
examples, please consult the :doc:`User Guide </user_guide/index>`.

.. note::
   Ensure ``fusionlab-learn`` is installed in your documentation build
   environment (see :doc:`installation`) for these links and summaries
   to be generated correctly. You also need `sphinx.ext.autosummary`
   enabled in your `conf.py` with `autosummary_generate = True`.

Datasets (`fusionlab.datasets`)
-------------------------------
Utilities for loading included sample datasets and generating synthetic
time series data for testing and demonstration.

**Loading Functions** (`fusionlab.datasets.load`)

.. autosummary::
   :toctree: _autosummary/datasets_load
   :nosignatures:

   ~fusionlab.datasets.load.fetch_zhongshan_data
   ~fusionlab.datasets.load.fetch_nansha_data
   ~fusionlab.datasets.load.load_processed_subsidence_data

**Generation Functions** (`fusionlab.datasets.make`)

.. autosummary::
   :toctree: _autosummary/datasets_make
   :nosignatures:

   ~fusionlab.datasets.make.make_multi_feature_time_series
   ~fusionlab.datasets.make.make_quantile_prediction_data
   ~fusionlab.datasets.make.make_anomaly_data
   ~fusionlab.datasets.make.make_trend_seasonal_data
   ~fusionlab.datasets.make.make_multivariate_target_data

Metrics (`fusionlab.metrics`)
-------------------------------
A collection of metrics for evaluating forecast accuracy, calibration,
sharpness, and stability, particularly suited for probabilistic and
time-series forecasting.

.. autosummary::
   :toctree: _autosummary/metrics
   :nosignatures:

   ~fusionlab.metrics.coverage_score
   ~fusionlab.metrics.continuous_ranked_probability_score
   ~fusionlab.metrics.mean_interval_width_score
   ~fusionlab.metrics.prediction_stability_score
   ~fusionlab.metrics.quantile_calibration_error
   ~fusionlab.metrics.theils_u_score
   ~fusionlab.metrics.time_weighted_accuracy_score
   ~fusionlab.metrics.time_weighted_interval_score
   ~fusionlab.metrics.time_weighted_mean_absolute_error
   ~fusionlab.metrics.weighted_interval_score
   
Forecasting Models (`fusionlab.nn.transformers`)
-------------------------------------------------
Core implementations of the Temporal Fusion Transformer and its variants.

.. autosummary::
   :toctree: _autosummary/models
   :nosignatures:

   ~fusionlab.nn.transformers.TemporalFusionTransformer
   ~fusionlab.nn.transformers.TFT
   ~fusionlab.nn.transformers.XTFT
   ~fusionlab.nn.transformers.DummyTFT
   ~fusionlab.nn.transformers.SuperXTFT


Core Model Components (`fusionlab.nn.components`)
-------------------------------------------------
Reusable building blocks used within the forecasting models.

.. autosummary::
   :toctree: _autosummary/components_core
   :nosignatures:

   ~fusionlab.nn.components.GatedResidualNetwork
   ~fusionlab.nn.components.VariableSelectionNetwork
   ~fusionlab.nn.components.PositionalEncoding
   ~fusionlab.nn.components.StaticEnrichmentLayer
   ~fusionlab.nn.components.LearnedNormalization


Sequence Processing Components (`fusionlab.nn.components`)
-----------------------------------------------------------
Components primarily focused on processing temporal sequences.

.. autosummary::
   :toctree: _autosummary/components_seq
   :nosignatures:

   ~fusionlab.nn.components.MultiScaleLSTM
   ~fusionlab.nn.components.DynamicTimeWindow
   ~fusionlab.nn.components.aggregate_multiscale
   ~fusionlab.nn.components.aggregate_time_window_output


Attention Mechanisms (`fusionlab.nn.components`)
-------------------------------------------------
Various attention layers used in TFT and XTFT architectures.

.. autosummary::
   :toctree: _autosummary/components_attn
   :nosignatures:

   ~fusionlab.nn.components.TemporalAttentionLayer
   ~fusionlab.nn.components.CrossAttention
   ~fusionlab.nn.components.HierarchicalAttention
   ~fusionlab.nn.components.MemoryAugmentedAttention
   ~fusionlab.nn.components.MultiResolutionAttentionFusion
   ~fusionlab.nn.components.ExplainableAttention


Embedding & Output Components (`fusionlab.nn.components`)
---------------------------------------------------------
Layers for input embedding and generating final model outputs.

.. autosummary::
   :toctree: _autosummary/components_io
   :nosignatures:

   ~fusionlab.nn.components.MultiModalEmbedding
   ~fusionlab.nn.components.MultiDecoder
   ~fusionlab.nn.components.QuantileDistributionModeling


Loss Functions (`fusionlab.nn.losses` & `fusionlab.nn.components`)
--------------------------------------------------------------------
Loss functions tailored for time series forecasting and anomaly detection.

.. autosummary::
   :toctree: _autosummary/losses
   :nosignatures:

   ~fusionlab.nn.losses.combined_quantile_loss
   ~fusionlab.nn.losses.prediction_based_loss
   ~fusionlab.nn.losses.combined_total_loss
   ~fusionlab.nn.losses.objective_loss
   ~fusionlab.nn.losses.quantile_loss
   ~fusionlab.nn.losses.quantile_loss_multi
   ~fusionlab.nn.losses.anomaly_loss
   ~fusionlab.nn.components.AdaptiveQuantileLoss
   ~fusionlab.nn.components.AnomalyLoss
   ~fusionlab.nn.components.MultiObjectiveLoss


Anomaly Detection (`fusionlab.nn.anomaly_detection`)
-----------------------------------------------------
Components specifically designed for anomaly detection tasks.

.. autosummary::
   :toctree: _autosummary/anomaly
   :nosignatures:

   ~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly
   ~fusionlab.nn.anomaly_detection.SequenceAnomalyScoreLayer
   ~fusionlab.nn.anomaly_detection.PredictionErrorAnomalyScore


Hyperparameter Tuning (`fusionlab.nn.forecast_tuner`)
------------------------------------------------------
Utilities for optimizing model hyperparameters using Keras Tuner.

.. autosummary::
   :toctree: _autosummary/tuning
   :nosignatures:

   ~fusionlab.nn.forecast_tuner.xtft_tuner
   ~fusionlab.nn.forecast_tuner.tft_tuner
   ~fusionlab.nn.forecast_tuner.XTFTTuner
   ~fusionlab.nn.forecast_tuner.TFTTuner


Neural Network Utilities (`fusionlab.nn.utils`)
------------------------------------------------
Utilities specifically for preparing data for or interacting with neural network models.

.. autosummary::
   :toctree: _autosummary/nn_utils
   :nosignatures:

   ~fusionlab.nn.utils.create_sequences
   ~fusionlab.nn.utils.split_static_dynamic
   ~fusionlab.nn.utils.reshape_xtft_data
   ~fusionlab.nn.utils.compute_forecast_horizon
   ~fusionlab.nn.utils.prepare_spatial_future_data
   ~fusionlab.nn.utils.compute_anomaly_scores
   ~fusionlab.nn.utils.generate_forecast
   ~fusionlab.nn.utils.generate_forecast_with
   ~fusionlab.nn.utils.forecast_single_step
   ~fusionlab.nn.utils.forecast_multi_step
   ~fusionlab.nn.utils.step_to_long
   ~fusionlab.nn.utils.format_predictions_to_dataframe 
   ~fusionlab.nn.utils.prepare_model_inputs  


Visual‑metric helpers (`fusionlab.plot.evaluation`)
------------------------------------------------------
A curated set of plotting utilities that turn the raw numbers returned  
by `fusionlab.metrics` into clear, publication‑quality figures.  
They cover point‑forecast accuracy, interval **sharpness & coverage**,  
ensemble calibration, temporal stability, and more – all tailored to  
time‑series / probabilistic‑forecast workflows.

.. autosummary::
   :toctree: _autosummary/metrics
   :nosignatures:

   ~fusionlab.plot.evaluation.plot_coverage
   ~fusionlab.plot.evaluation.plot_crps
   ~fusionlab.plot.evaluation.plot_forecast_comparison
   ~fusionlab.plot.evaluation.plot_mean_interval_width
   ~fusionlab.plot.evaluation.plot_metric_over_horizon
   ~fusionlab.plot.evaluation.plot_metric_radar
   ~fusionlab.plot.evaluation.plot_prediction_stability
   ~fusionlab.plot.evaluation.plot_quantile_calibration
   ~fusionlab.plot.evaluation.plot_theils_u_score
   ~fusionlab.plot.evaluation.plot_time_weighted_metric
   ~fusionlab.plot.evaluation.plot_weighted_interval_score


Quick‑look forecast helpers (`fusionlab.plot.forecast`)
---------------------------------------------------------
Light‑weight plotting utilities that turn a long‑format forecast
DataFrame (as returned by
:func:fusionlab.nn.utils.format_predictions_to_dataframe) into clear,
side‑by‑side figures for rapid inspection.
 
.. autosummary::
   :toctree: _autosummary/forecast
   :nosignatures:

   ~fusionlab.plot.forecast.plot_forecasts
   ~fusionlab.plot.forecast.visualize_forecasts

   
Time Series Utilities (`fusionlab.utils.ts_utils`)
---------------------------------------------------
General utilities for time series data processing, analysis, and feature engineering.

.. autosummary::
   :toctree: _autosummary/ts_utils
   :nosignatures:

   ~fusionlab.utils.ts_utils.ts_validator
   ~fusionlab.utils.ts_utils.to_dt
   ~fusionlab.utils.ts_utils.filter_by_period
   ~fusionlab.utils.ts_utils.ts_engineering
   ~fusionlab.utils.ts_utils.create_lag_features
   ~fusionlab.utils.ts_utils.trend_analysis
   ~fusionlab.utils.ts_utils.trend_ops
   ~fusionlab.utils.ts_utils.decompose_ts
   ~fusionlab.utils.ts_utils.get_decomposition_method
   ~fusionlab.utils.ts_utils.infer_decomposition_method
   ~fusionlab.utils.ts_utils.ts_corr_analysis
   ~fusionlab.utils.ts_utils.transform_stationarity
   ~fusionlab.utils.ts_utils.ts_split
   ~fusionlab.utils.ts_utils.ts_outlier_detector
   ~fusionlab.utils.ts_utils.select_and_reduce_features


Command-Line Tools (`fusionlab.tools`)
---------------------------------------
High-level applications for common workflows. For usage details, see the
:doc:`Command-Line Tools guide </user_guide/tools>`.

.. rubric:: References

.. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
       Temporal fusion transformers for interpretable multi-horizon
       time series forecasting. *International Journal of Forecasting*,
       37(4), 1748-1764. (Also arXiv:1912.09363)