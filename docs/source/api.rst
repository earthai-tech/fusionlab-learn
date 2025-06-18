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


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   
Hyperparameter Tuning (`fusionlab.nn.forecast_tuner`)
------------------------------------------------------
Utilities for optimizing model hyperparameters using Keras Tuner.

.. autosummary::
   :toctree: _autosummary/tuning
   :nosignatures:

   ~fusionlab.nn.forecast_tuner.HydroTuner
   ~fusionlab.nn.forecast_tuner.HALTuner
   ~fusionlab.nn.forecast_tuner.XTFTTuner
   ~fusionlab.nn.forecast_tuner.TFTTuner
   ~fusionlab.nn.forecast_tuner.PiHALTuner
   ~fusionlab.nn.forecast_tuner.xtft_tuner
   ~fusionlab.nn.forecast_tuner.tft_tuner
   
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
   ~fusionlab.nn.pinn.utils.format_pihalnet_predictions 
   ~fusionlab.nn.pinn.utils.prepare_pinn_data_sequences 


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
   ~fusionlab.nn.models.utils.plot_history_in

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

   ~fusionlab.utils.data_utils.nan_ops
   ~fusionlab.utils.data_utils.widen_temporal_columns
   ~fusionlab.utils.forecast_utils.pivot_forecast_dataframe
   ~fusionlab.utils.spatial_utils.create_spatial_clusters
   ~fusionlab.utils.spatial_utils.batch_spatial_sampling
   ~fusionlab.utils.spatial_utils.spatial_sampling
   ~fusionlab.nn.utils.create_sequences
   ~fusionlab.nn.pinn.utils.prepare_pinn_data_sequences
   ~fusionlab.nn.pinn.utils.format_pinn_predictions
   
Command-Line Tools (`fusionlab.tools`)
---------------------------------------
High-level applications for common workflows. For usage details, see the
:doc:`Command-Line Tools guide </user_guide/tools>`.

.. rubric:: References

.. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
       Temporal fusion transformers for interpretable multi-horizon
       time series forecasting. *International Journal of Forecasting*,
       37(4), 1748-1764. (Also arXiv:1912.09363)
