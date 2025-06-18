.. _api_reference:
.. default-role:: obj

===============
API Reference
===============

Welcome to the ``fusionlab-learn`` API reference. This section provides detailed
specifications for the public functions, classes, and modules included
in the package.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   
Datasets (`fusionlab.datasets`)
---------------------------------
Utilities for loading included sample datasets and generating synthetic
time series data for testing and demonstration.

**Loading Functions** (`fusionlab.datasets`)

.. autosummary::
   :toctree: _autosummary/datasets
   :nosignatures:

   ~fusionlab.datasets.fetch_zhongshan_data
   ~fusionlab.datasets.fetch_nansha_data
   ~fusionlab.datasets.load_processed_subsidence_data
   ~fusionlab.datasets.load_subsidence_pinn_data
   ~fusionlab.datasets.make_multi_feature_time_series
   ~fusionlab.datasets.make_quantile_prediction_data
   ~fusionlab.datasets.make_anomaly_data
   ~fusionlab.datasets.make_trend_seasonal_data
   ~fusionlab.datasets.make_multivariate_target_data
   
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
   
.. raw:: html

   <hr>

Transformer Models (`fusionlab.nn.transformers`)
-------------------------------------------------
Implementations of pure-transformer and Temporal Fusion Transformer architectures.

.. autosummary::
   :toctree: _autosummary/models
   :nosignatures:

   ~fusionlab.nn.transformers.TimeSeriesTransformer
   ~fusionlab.nn.transformers.TemporalFusionTransformer
   ~fusionlab.nn.transformers.TFT
   ~fusionlab.nn.transformers.DummyTFT

Fusion-Attentive Models (`fusionlab.nn.models`)
-------------------------------------------------
Core implementations of the Hybrid-Attentive Fusion and its variants.

.. autosummary::
   :toctree: _autosummary/models
   :nosignatures:

   ~fusionlab.nn.models.BaseAttentive
   ~fusionlab.nn.models.HALNet
   ~fusionlab.nn.models.XTFT
   ~fusionlab.nn.models.SuperXTFT

Physic-Informed Models (`fusionlab.nn.pinn`)
--------------------------------------------------------
Fusion models that integrate physical laws into the training process.

.. autosummary::
   :toctree: _autosummary/models
   :nosignatures:

   ~fusionlab.nn.pinn.TransFlowSubsNet
   ~fusionlab.nn.pinn.models.PIHALNet
   ~fusionlab.nn.pinn.PiHALNet
   ~fusionlab.nn.pinn.PiTGWFlow
   
.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   
Core Neural Network Components (`fusionlab.nn.components`)
-----------------------------------------------------------
Reusable building blocks for feature selection, sequence processing,
attention, and output generation used within the forecasting models.

.. autosummary::
   :toctree: _autosummary/components_core
   :nosignatures:

   ~fusionlab.nn.components.GatedResidualNetwork
   ~fusionlab.nn.components.VariableSelectionNetwork
   ~fusionlab.nn.components.PositionalEncoding
   ~fusionlab.nn.components.StaticEnrichmentLayer
   ~fusionlab.nn.components.LearnedNormalization
   ~fusionlab.nn.components.PositionwiseFeedForward

Sequence Processing Components (`fusionlab.nn.components`)
-----------------------------------------------------------
Components primarily focused on processing temporal sequences.

.. autosummary::
   :toctree: _autosummary/components_seq
   :nosignatures:

   ~fusionlab.nn.components.MultiScaleLSTM
   ~fusionlab.nn.components.DynamicTimeWindow
   ~fusionlab.nn.components.aggregate_multiscale
   ~fusionlab.nn.components.aggregate_multiscale_on_3d
   ~fusionlab.nn.components.aggregate_time_window_output
   ~fusionlab.nn.components.create_causal_mask


Attention Mechanisms (`fusionlab.nn.components`)
-------------------------------------------------
Various attention layers used in Fusion Model architectures.

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

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   
Losses (`fusionlab.nn.components` & `fusionlab.nn.losses`)
-------------------------------------------------------------
Losses tailored for time series forecasting and anomaly detection.

**Loss Components** (`fusionlab.nn.components`)

These classes implement composite or parameterized loss behaviors.

.. autosummary::
   :toctree: _autosummary/losses
   :nosignatures:

   ~fusionlab.nn.components.AdaptiveQuantileLoss
   ~fusionlab.nn.components.AnomalyLoss
   ~fusionlab.nn.components.MultiObjectiveLoss


**Loss Functions** (`fusionlab.nn.losses`)

Pure functions for computing scalar losses on predictions.

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
  
.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

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
   ~fusionlab.nn.utils.format_predictions
   ~fusionlab.nn.utils.format_predictions_to_dataframe 
   ~fusionlab.nn.utils.prepare_model_inputs
   ~fusionlab.nn.utils.format_pihalnet_predictions 
   ~fusionlab.nn.utils.prepare_pinn_data_sequences
   ~fusionlab.nn.utils.format_pinn_predictions 
   
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

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Quick‑look forecast helpers (`fusionlab.plot.forecast`)
---------------------------------------------------------
Light‑weight plotting utilities that turn a long‑format forecast
DataFrame (as returned by
:func:fusionlab.nn.utils.format_predictions_to_dataframe) into clear,
side‑by‑side figures for rapid inspection.
 
.. autosummary::
   :toctree: _autosummary/forecast
   :nosignatures:

   ~fusionlab.plot.forecast.forecast_view
   ~fusionlab.plot.forecast.plot_forecasts
   ~fusionlab.plot.forecast.plot_forecast_by_step
   ~fusionlab.plot.forecast.visualize_forecasts

Time Series Utilities (`fusionlab.utils.ts_utils`)
-----------------------------------------------------
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

Data Processing Utilities (`fusionlab.utils`)
-------------------------------------------------
A collection of helpers for data manipulation, feature engineering,
and preparing data for models.

.. autosummary::
   :toctree: _autosummary/utils
   :nosignatures:

   ~fusionlab.utils.nan_ops
   ~fusionlab.utils.widen_temporal_columns
   ~fusionlab.utils.pivot_forecast_dataframe
   ~fusionlab.utils.create_spatial_clusters
   ~fusionlab.utils.spatial_sampling
   ~fusionlab.utils.augment_series_features 
   ~fusionlab.utils.generate_dummy_pinn_data 
   ~fusionlab.utils.augment_spatiotemporal_data
   ~fusionlab.utils.mask_by_reference
   ~fusionlab.utils.fetch_joblib_data 
   ~fusionlab.utils.save_job 
   
Command-Line Tools (`fusionlab.tools`)
---------------------------------------
High-level applications for common workflows. For usage details, see the
:doc:`Command-Line Tools guide </user_guide/tools>`.

.. rubric:: References

.. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
       Temporal fusion transformers for interpretable multi-horizon
       time series forecasting. *International Journal of Forecasting*,
       37(4), 1748-1764. (Also arXiv:1912.09363)
