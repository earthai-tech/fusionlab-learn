.. _user_guide_evaluation_plotting:

======================================
Evaluating and Visualizing Forecasts
======================================

Effective evaluation and clear visualization are key to understanding
the performance of your forecasting models and communicating their
results. ``fusionlab-learn`` provides utilities in
:mod:`fusionlab.plot.evaluation` to help with this process,
working seamlessly with forecast data structured by
:func:`~fusionlab.nn.utils.format_predictions_to_dataframe`.

This guide demonstrates how to use the primary plotting functions:

* :func:`~fusionlab.plot.evaluation.plot_forecast_comparison`: For
  visualizing actual vs. predicted values, including quantile intervals.
* :func:`~fusionlab.plot.evaluation.plot_metric_over_horizon`: For
  analyzing how performance metrics change across the forecast horizon.
* :func:`~fusionlab.plot.evaluation.plot_metric_radar`: For comparing
  a metric across different segments or categories.

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and its common dependencies
installed. For visualizations, `matplotlib` is essential.

.. code-block:: bash

   pip install fusionlab-learn matplotlib scikit-learn

Common Setup for Examples
-------------------------
The following imports and basic data generation will be used across
the examples. We'll simulate a forecast DataFrame that might be
produced after running a model and formatting its output.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf # For Tensor type hint if needed
   import matplotlib.pyplot as plt
   import os
   import warnings

   # FusionLab imports
   from fusionlab.nn.utils import format_predictions_to_dataframe
   from fusionlab.plot.evaluation import (
       plot_forecast_comparison,
       plot_metric_over_horizon,
       plot_metric_radar
   )
   # For dummy scaler and metrics if needed by plot functions
   from sklearn.preprocessing import StandardScaler
   from sklearn.metrics import mean_absolute_error

   # Suppress warnings and TF logs for cleaner output
   warnings.filterwarnings('ignore')
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   tf.get_logger().setLevel('ERROR')
   if hasattr(tf, 'autograph'):
       tf.autograph.set_verbosity(0)

   # Directory for saving any output images from this guide
   evaluation_plot_dir = "./evaluation_plots_output"
   os.makedirs(evaluation_plot_dir, exist_ok=True)

   print("Libraries imported and setup complete for evaluation plotting.")

   # --- Generate Base Dummy Forecast Data ---
   # This data will be used as input to format_predictions_to_dataframe
   # to create the forecast_df for our plotting functions.
   B, H, O_SINGLE, O_MULTI = 10, 6, 1, 2 # Batch, Horizon, OutputDims
   Q_LIST_VIZ = [0.1, 0.5, 0.9]
   N_Q_VIZ = len(Q_LIST_VIZ)
   SAMPLES_VIZ = B # Number of sequences

   np.random.seed(42) # For reproducibility
   base_y_true_single = 50 + np.cumsum(
       np.random.randn(SAMPLES_VIZ, H, O_SINGLE) * 2, axis=1)
   base_preds_point_single = base_y_true_single * \
       np.random.uniform(0.9, 1.1, size=base_y_true_single.shape) + \
       np.random.normal(0, 2, size=base_y_true_single.shape)
   
   # For quantile, ensure median is somewhat centered, and bounds span it
   base_preds_q_median = base_y_true_single * \
       np.random.uniform(0.95, 1.05, size=base_y_true_single.shape) + \
       np.random.normal(0, 1, size=base_y_true_single.shape)
   interval_spread = np.abs(np.random.normal(2, 1, size=base_y_true_single.shape))
   base_preds_q_lower = base_preds_q_median - interval_spread
   base_preds_q_upper = base_preds_q_median + interval_spread
   
   # Stack quantiles for single output: (Samples, Horizon, NumQuantiles)
   base_preds_quant_single = np.stack([
       base_preds_q_lower, base_preds_q_median, base_preds_q_upper
   ], axis=-1).reshape(SAMPLES_VIZ, H, N_Q_VIZ)


   # Create a sample forecast_df for point forecasts
   forecast_df_point_viz = format_predictions_to_dataframe(
       predictions=base_preds_point_single.astype(np.float32),
       y_true_sequences=base_y_true_single.astype(np.float32),
       target_name="value",
       forecast_horizon=H,
       output_dim=O_SINGLE
   )
   # Add a segment column for radar plot example
   forecast_df_point_viz['category'] = np.random.choice(
       ['CatA', 'CatB', 'CatC'], size=len(forecast_df_point_viz)
       )
   # Add spatial columns for spatial plot example
   forecast_df_point_viz['longitude'] = np.tile(
       np.linspace(110, 111, SAMPLES_VIZ), H)
   forecast_df_point_viz['latitude'] = np.tile(
       np.linspace(22, 23, SAMPLES_VIZ), H)


   # Create a sample forecast_df for quantile forecasts
   forecast_df_quant_viz = format_predictions_to_dataframe(
       predictions=base_preds_quant_single.astype(np.float32),
       y_true_sequences=base_y_true_single.astype(np.float32),
       target_name="value",
       quantiles=Q_LIST_VIZ,
       forecast_horizon=H,
       output_dim=O_SINGLE
   )
   forecast_df_quant_viz['category'] = np.random.choice(
       ['CatX', 'CatY', 'CatZ'], size=len(forecast_df_quant_viz)
       )
   forecast_df_quant_viz['longitude'] = np.tile(
       np.linspace(110, 111, SAMPLES_VIZ), H)
   forecast_df_quant_viz['latitude'] = np.tile(
       np.linspace(22, 23, SAMPLES_VIZ), H)


   print("Base data and sample DataFrames prepared for plotting examples.")

**Expected Output (Common Setup):**

.. code-block:: text

   Libraries imported and setup complete for evaluation plotting.
   Base data and sample DataFrames prepared for plotting examples.

---

1. Visualizing Forecast Comparisons (`plot_forecast_comparison`)
-----------------------------------------------------------------
:API Reference: :func:`~fusionlab.plot.evaluation.plot_forecast_comparison`

This function is your primary tool for visually comparing model
predictions against actual values. It supports both temporal line plots
(showing forecasts over the horizon for specific samples) and spatial
scatter plots (showing forecast values across geographical coordinates
for a specific horizon step).

**Key Use Cases:**
* **Temporal Point Forecasts:** Plot actual vs. predicted lines for
  selected time series samples.
* **Temporal Quantile Forecasts:** Plot actuals, the median prediction,
  and the uncertainty interval (e.g., between 10th and 90th quantiles).
* **Spatial Forecasts:** Visualize predicted values (e.g., median for
  quantiles) on a map for a specific forecast step.

**Example 1.1: Temporal Point Forecast Visualization**

.. code-block:: python
   :linenos:

   print("\nPlotting Temporal Point Forecast Comparison...")
   plot_forecast_comparison(
       forecast_df=forecast_df_point_viz,
       target_name="value",
       kind="temporal",
       sample_ids="first_n", # Plot for the first N samples
       num_samples=2,        # Plot for 2 samples
       max_cols=1,           # Each sample in its own row
       figsize_per_subplot=(10, 4),
       verbose=0
   )
   # To save:
   # fig_path = os.path.join(evaluation_plot_dir, "eval_temporal_point.png")
   # plt.savefig(fig_path) # Call before plt.show() if saving

**Expected Plot 1.1:**

.. figure:: ../../images/evaluation_forecast_comparison_temporal_point.png
   :alt: Temporal Point Forecast Comparison
   :align: center
   :width: 70%

   Line plot showing actual vs. predicted values over the forecast
   horizon for selected samples (point forecast).

**Example 1.2: Temporal Quantile Forecast Visualization**

.. code-block:: python
   :linenos:

   print("\nPlotting Temporal Quantile Forecast Comparison...")
   plot_forecast_comparison(
       forecast_df=forecast_df_quant_viz,
       target_name="value",
       quantiles=Q_LIST_VIZ,
       kind="temporal",
       sample_ids=[0, 1], # Plot for specific sample_idx 0 and 1
       max_cols=2,
       figsize_per_subplot=(9, 4.5),
       verbose=0
   )
   # To save:
   # fig_path = os.path.join(evaluation_plot_dir, "eval_temporal_quantile.png")
   # plt.savefig(fig_path)

**Expected Plot 1.2:**

.. figure:: ../../images/evaluation_forecast_comparison_temporal_quantile.png
   :alt: Temporal Quantile Forecast Comparison
   :align: center
   :width: 90%

   Line plot showing actual values, median prediction, and the
   prediction interval for selected samples (quantile forecast).

**Example 1.3: Spatial Quantile Forecast Visualization**
   This requires `spatial_cols` (e.g., 'longitude', 'latitude') to be
   present in `forecast_df`.

.. code-block:: python
   :linenos:

   print("\nPlotting Spatial Quantile Forecast Comparison...")
   plot_forecast_comparison(
       forecast_df=forecast_df_quant_viz,
       target_name="value",
       quantiles=Q_LIST_VIZ,
       kind="spatial",
       horizon_steps=1, # Visualize the first step of the horizon
       spatial_cols=['longitude', 'latitude'],
       figsize_per_subplot=(7, 6), # Single plot, so this is figure size
       verbose=0
   )
   # To save:
   # fig_path = os.path.join(evaluation_plot_dir, "eval_spatial_quantile.png")
   # plt.savefig(fig_path)

**Expected Plot 1.3:**

.. figure:: ../../images/evaluation_forecast_comparison_spatial.png
   :alt: Spatial Quantile Forecast Comparison
   :align: center
   :width: 70%

   Scatter plot showing the median predicted values across spatial
   coordinates for a specific forecast horizon step.

---

2. Visualizing Metrics Over the Forecast Horizon (`plot_metric_over_horizon`)
------------------------------------------------------------------------------
:API Reference: :func:`~fusionlab.plot.evaluation.plot_metric_over_horizon`

This function helps understand how a model's performance, measured by
one or more metrics, changes as the forecast lead time increases.
It's useful for identifying if a model's accuracy degrades
significantly for longer horizons.

**Key Use Cases:**
* Plotting MAE, RMSE, MAPE, etc., for each step of the horizon.
* For quantile forecasts, plotting coverage or pinball loss over the
  horizon.
* Comparing horizon-wise metrics across different segments if
  `group_by_cols` is used.

**Example 2.1: MAE of Point Forecasts Over Horizon**

.. code-block:: python
   :linenos:

   print("\nPlotting MAE of Point Forecast Over Horizon...")
   plot_metric_over_horizon(
       forecast_df=forecast_df_point_viz,
       target_name="value",
       metrics='mae', # Calculate Mean Absolute Error
       plot_kind='bar', # Display as a bar chart
       figsize_per_subplot=(8, 5),
       verbose=0
   )
   # To save:
   # fig_path = os.path.join(evaluation_plot_dir, "eval_moh_mae_point.png")
   # plt.savefig(fig_path)

**Expected Plot 2.1:**

.. figure:: ../../images/evaluation_metric_over_horizon_mae.png
   :alt: MAE of Point Forecast Over Horizon
   :align: center
   :width: 70%

   Bar chart showing Mean Absolute Error for each step of the
   forecast horizon.

**Example 2.2: Coverage of Quantile Forecasts Over Horizon (Grouped)**

.. code-block:: python
   :linenos:

   # Ensure coverage_score is available for this example
   
   print("\nPlotting Coverage of Quantile Forecast Over Horizon (Grouped)...")
   plot_metric_over_horizon(
        forecast_df=forecast_df_quant_viz,
        target_name="value",
        metrics='coverage',
        quantiles=Q_LIST_VIZ, # Required for coverage
        group_by_cols=['category'], # Show coverage per category
        plot_kind='line',
        figsize_per_subplot=(9, 5),
        verbose=0
   )
    # To save:
    # fig_path = os.path.join(evaluation_plot_dir, "eval_moh_coverage_quant.png")
    # plt.savefig(fig_path)
   
**Expected Plot 2.2:**

.. figure:: ../../images/evaluation_metric_over_horizon_coverage.png
   :alt: Coverage of Quantile Forecast Over Horizon
   :align: center
   :width: 70%

   Line plot showing prediction interval coverage for each forecast
   step, potentially with separate lines for different categories.

---

3. Visualizing Metrics Across Segments with Radar Plots (`plot_metric_radar`)
-----------------------------------------------------------------------------
:API Reference: :func:`~fusionlab.plot.evaluation.plot_metric_radar`

Radar charts provide a way to compare a single performance metric
across different categorical segments (e.g., item types, regions,
months). Each segment forms an axis on the radar, and the metric's
value for that segment is plotted along it.

**Key Use Cases:**
* Comparing model performance (e.g., MAE, RMSE) across different
  product categories.
* Identifying if a model performs consistently across different
  days of the week or months.

**Example 3.1: MAE of Median Forecast by Category (Radar Plot)**

.. code-block:: python
   :linenos:

   print("\nPlotting MAE by Category (Radar Plot)...")
   # Using forecast_df_quant_viz which has a 'category' column
   plot_metric_radar(
       forecast_df=forecast_df_quant_viz,
       segment_col='category', # Column defining the radar axes
       metric='mae',
       target_name="value",
       quantiles=Q_LIST_VIZ, # MAE will be on the median
       figsize=(7, 7),
       verbose=0
   )
   # To save:
   # fig_path = os.path.join(evaluation_plot_dir, "eval_radar_mae_category.png")
   # plt.savefig(fig_path)

**Expected Plot 3.1:**

.. figure:: ../../images/evaluation_metric_radar_mae.png
   :alt: MAE by Category Radar Plot
   :align: center
   :width: 70%

   Radar chart showing the Mean Absolute Error (of the median
   forecast) for different categories.

---

Further Exploration
-------------------

These examples provide a starting point for visualizing your
``fusionlab-learn`` model outputs. For a detailed understanding of
the metrics themselves, including their mathematical formulations and
calculation examples, please refer to the :doc:`metrics` page.

Experiment with different parameters of these plotting functions to
customize the visualizations for your specific analysis needs.

