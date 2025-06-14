.. _advanced_forecast_plots_guide:

==========================
Forecast Visualization
==========================

Visualizing forecast results is a critical step for evaluating
model performance, understanding prediction uncertainty, and
communicating findings. The ``fusionlab.plot.forecast`` module
provides a suite of functions designed to create insightful and
customizable plots for various forecasting scenarios.

This guide covers the main plotting utilities, from high-level,
automated functions to more specific tools.


forecast_view
=============
:API Reference: :func:`~fusionlab.plot.forecast.forecast_view`

The ``forecast_view`` function is a powerful, high-level utility
designed for the spatial visualization of multi-horizon forecasts.
It automatically handles both long and wide-format prediction
DataFrames, arranges plots in a structured grid by year and
metric, and provides extensive customization options.

Key Features
------------
* **Automatic Format Handling**: Intelligently detects whether the
    input DataFrame is in a long or wide format and pivots it if
    necessary.
* **Structured Grid Layout**: Organizes plots in a grid where each
    row typically represents a forecast year and each column
    represents a different metric (e.g., actuals, q10, q50, q90).
* **Dual Mode Comparison**: Can display actual values alongside
    predictions for direct comparison (`kind='dual'`).
* **Filtering**: Allows users to select specific years
    (`view_years`) and quantiles (`view_quantiles`) to visualize.
* **Uniform Color Scaling**: Can apply a uniform color scale
    (`cbar_type='uniform'`) across all subplots for consistent
    comparison of magnitudes.
* **Customization**: Offers control over figure size, colormaps,
    axes, grids, and saving options.

When to Use
-----------
Use ``forecast_view`` when you have a forecast DataFrame (long or
wide) containing spatial coordinates (`coord_x`, `coord_y`) and
you want to create a comprehensive spatial overview of your model's
predictions across multiple years and uncertainty levels. It is the
recommended starting point for visualizing PINN model outputs.

Example
~~~~~~~
This example demonstrates how to use ``forecast_view`` to visualize
a forecast for 'subsidence', showing both actuals and predictions
for the q10, q50, and q90 quantiles.

.. code-block:: python
   :linenos:

   import pandas as pd
   from fusionlab.plot.forecast import forecast_view
   from fusionlab.utils.forecast_utils import get_value_prefixes

   # Example 1: 
   # Assume `df_forecast_long` is a long-format DataFrame from a model
   # It must contain spatial_cols, a time_col, and value columns.
   # For this example, we'll create a dummy one.
   data = {
       'sample_idx':      [0, 0, 1, 1],
       'coord_t':         [2023.0, 2024.0, 2023.0, 2024.0],
       'coord_x':         [113.1, 113.1, 113.5, 113.5],
       'coord_y':         [22.5, 22.5, 22.8, 22.8],
       'subsidence_q10':  [-10, -11, -15, -16],
       'subsidence_q50':  [-8, -9, -13, -14],
       'subsidence_q90':  [-6, -7, -11, -12],
       'subsidence_actual': [-8.5, -8.5, -13.2, -13.2],
       'GWL_q50':         [1.2, 1.3, 2.2, 2.3], # Another metric
   }
   df_long_example = pd.DataFrame(data)

   # Auto-detect the prefixes to plot
   prefixes_to_plot = get_value_prefixes(df_long_example)
   # -> This would return ['GWL', 'subsidence']

   # Visualize only the 'subsidence' forecast
   forecast_view(
       forecast_df=df_long_example,
       value_prefixes=['subsidence'],
       kind='dual',
       spatial_cols=('coord_x', 'coord_y'),
       time_col='coord_t',
       max_cols='auto',
       cmap='viridis',
       axis_off=False,
       savefig='output/subsidence_forecast_view.png',
       verbose=1
   )

   # --- Generate a substantial, reproducible dummy dataset ---
   # Set a seed for reproducibility
   np.random.seed(42)

   # Define parameters for data generation
   n_unique_samples = 50000
   forecast_years = [2023.0, 2024.0]
   n_years = len(forecast_years)
   total_rows = n_unique_samples * n_years

   # Create base sample and coordinate data
   sample_ids = np.arange(n_unique_samples)
   coord_x_base = np.random.uniform(113.0, 113.8, n_unique_samples)
   coord_y_base = np.random.uniform(22.3, 22.8, n_unique_samples)
   subsidence_actual_base = np.random.normal(-25, 8, n_unique_samples)

   # Create the long-format DataFrame by repeating the base data for each year
   data = {
       'sample_idx': np.repeat(sample_ids, n_years),
       'coord_t': np.tile(forecast_years, n_unique_samples),
       'coord_x': np.repeat(coord_x_base, n_years),
       'coord_y': np.repeat(coord_y_base, n_years),
       'subsidence_actual': np.repeat(subsidence_actual_base, n_years)
   }

   # Generate forecast values that have some relation to the actuals
   base_preds = data['subsidence_actual'] + np.random.normal(0, 2, total_rows)
   data['subsidence_q50'] = base_preds
   data['subsidence_q10'] = base_preds - np.random.uniform(2, 5, total_rows)
   data['subsidence_q90'] = base_preds + np.random.uniform(2, 5, total_rows)
   data['GWL_q50'] = np.random.uniform(1.0, 5.0, total_rows) # Another metric

   df_long_example = pd.DataFrame(data)
   # -> This creates a DataFrame with 100,000 rows

   # Auto-detect the prefixes to plot
   prefixes_to_plot = get_value_prefixes(df_long_example)
   # -> This would return ['GWL', 'subsidence']

   # Visualize only the 'subsidence' forecast
   forecast_view(
       forecast_df=df_long_example,
       value_prefixes=['subsidence'],
       kind='dual',
       spatial_cols=('coord_x', 'coord_y'),
       time_col='coord_t',
       max_cols='auto',
       cmap='viridis',
       axis_off=False,
       savefig='output/subsidence_forecast_view.png',
       verbose=1
   )
   
**Example Output Plot:**

.. figure:: ../images/forecast_view_example.png
   :alt: Example grid plot from forecast_view
   :align: center
   :width: 90%

   Example output from ``forecast_view`` showing a grid of spatial subsidence
   forecasts. Each row represents a forecast year, and columns show the
   actual ground truth alongside predictions for different quantiles (q10,
   q50, q90).

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

plot_forecast_by_step
=======================
:API Reference: :func:`~fusionlab.plot.forecast.plot_forecast_by_step`

This function is tailored to visualize forecast data by organizing
plots based on the `forecast_step` (e.g., 1-step ahead, 2-steps
ahead). It is particularly useful for analyzing how a model's
performance changes over the forecast horizon.

Key Features
------------
* **Step-Based Organization**: Creates a grid of plots where each
  row corresponds to a forecast step.
* **Flexible Plotting**: Automatically generates spatial scatter plots
  if `spatial_cols` are provided and valid. If not, it gracefully
  falls back to creating temporal line plots.
* **Custom Step Naming**: Allows custom labels for each step via the
  `step_names` parameter, making plots more interpretable (e.g.,
  mapping Step 1 to 'Year 2023').
* **Comprehensive Customization**: Includes the same rich set of
  customization options as ``forecast_view``, such as `kind`, `cmap`,
  `cbar_type`, `savefig`, etc.

When to Use
-----------
Use ``plot_forecast_by_step`` when your primary goal is to analyze
the model's predictive capability at each step into the future. It
helps answer questions like: "Does the model's accuracy degrade as
it predicts further out?" or "Are there spatial patterns in the error
at specific forecast horizons?"

Example
~~~~~~~
.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   from fusionlab.plot.forecast import plot_forecast_by_step

   # Create a dummy long-format DataFrame
   n_samples_per_step = 100
   steps = [1, 2, 3]
   df_list = []
   for step in steps:
       df_list.append(pd.DataFrame({
           'sample_idx': np.arange(n_samples_per_step),
           'forecast_step': step,
           'coord_x': np.random.rand(n_samples_per_step),
           'coord_y': np.random.rand(n_samples_per_step),
           'subsidence_pred': np.random.randn(n_samples_per_step) * step,
           'subsidence_actual': np.random.randn(n_samples_per_step) * step + 0.5
       }))
   df_step_example = pd.concat(df_list, ignore_index=True)

   # Visualize the forecast by step
   # plot_forecast_by_step(
   #     df=df_step_example,
   #     value_prefixes=['subsidence'],
   #     spatial_cols=('coord_x', 'coord_y'),
   #     step_names={1: '1-Year Ahead', 2: '2-Years Ahead', 3: '3-Years Ahead'},
   #     kind='dual',
   #     max_cols=2 # Show 'Actual' and 'Prediction'
   # )

**Example Output Plot:**

.. figure:: ../images/plot_forecast_by_step_example.png
   :alt: Example grid plot from plot_forecast_by_step
   :align: center
   :width: 70%

   Example output from ``plot_forecast_by_step``. Each row shows the
   spatial distribution of the forecast for a specific step into the
   future (e.g., 1-Year Ahead, 2-Years Ahead), allowing for direct
   comparison of performance over the forecast horizon.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

plot_forecasts
==============
:API Reference: :func:`~fusionlab.plot.forecast.plot_forecasts`

The ``plot_forecasts`` function is a versatile utility for creating
both temporal and spatial plots of model forecasts. It is designed
to work with a specific long-format DataFrame structure, typically
one created by :func:`~fusionlab.nn.utils.format_predictions_to_dataframe`.

Key Features
------------
* **Temporal and Spatial Plots**: Can generate line plots over time
    (`kind='temporal'`) or scatter plots over space
    (`kind='spatial'`).
* **Sample Selection**: Easily visualize forecasts for specific
    samples using `sample_ids` (e.g., plot the first 3 samples).
* **Quantile Shading**: In temporal plots, it automatically shades
    the area between quantile predictions to represent uncertainty.
* **Inverse Scaling**: Can apply an inverse transformation to scaled
    predictions if a fitted `scaler` object is provided.

Example
~~~~~~~
.. code-block:: python
   :linenos:

   from fusionlab.nn.utils import format_predictions_to_dataframe
   from fusionlab.plot.forecast import plot_forecasts
   import pandas as pd
   import numpy as np

   # Assume preds_quant (B,H,Q) and y_true_seq (B,H,O) are available
   B, H, O, Q_len = 4, 3, 1, 3
   preds_quant = np.random.rand(B, H, Q_len)
   y_true_seq = np.random.rand(B, H, O)
   quantiles_list = [0.1, 0.5, 0.9]

   # Create DataFrame using the formatting utility
   df_quant = format_predictions_to_dataframe(
       predictions=preds_quant, y_true_sequences=y_true_seq,
       target_name="value", quantiles=quantiles_list,
       forecast_horizon=H, output_dim=O
   )

   # Plot temporal quantile forecast for the first 2 samples
   # plot_forecasts(
   #     df_quant,
   #     target_name="value",
   #     quantiles=quantiles_list,
   #     sample_ids="first_n",
   #     num_samples=2,
   #     max_cols=1
   # )

**Example Output Plot:**

.. figure:: ../images/plot_forecasts_temporal_example.png
   :alt: Temporal forecast plot with quantile shading
   :align: center
   :width: 90%

   A temporal forecast plot generated by ``plot_forecasts``. The solid
   line represents the actual data, the dashed line is the median
   prediction (q50), and the shaded area represents the uncertainty
   range between the q10 and q90 quantiles.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

visualize_forecasts
===================
:API Reference: :func:`~fusionlab.plot.forecast.visualize_forecasts`

The ``visualize_forecasts`` function is another powerful tool for
creating spatial and non-spatial forecast plots. It is distinct in
that it often takes separate DataFrames for forecast results and
actual test data, merging them for visualization.

Key Features
------------
* **Separate DataFrames**: Designed to work with a `forecast_df`
    containing predictions and an optional `test_data` DataFrame
    containing ground truth.
* **Flexible Plotting**: Supports both 'spatial' and 'non-spatial'
    kinds of plots.
* **Time Period Filtering**: Can focus the visualization on
    specific years or time periods using the `eval_periods` parameter.

Example
~~~~~~~
.. code-block:: python
   :linenos:

   from fusionlab.plot.forecast import visualize_forecasts
   import pandas as pd
   import numpy as np
   rng = np.random.default_rng(seed=42)
   n_periods = 3

   dates = pd.date_range("2023-01-01", periods=n_periods, freq="D")
   lon, lat = -103.8, 0.47
    
   forecast_results = pd.DataFrame(
        {
        "longitude": np.full(n_periods, lon),
        "latitude": np.full(n_periods, lat),
        "subsidence_q50": rng.uniform(0.30, 0.50, n_periods),
        "subsidence": rng.uniform(0.33, 0.49, n_periods),
        "date": dates,
        }
    )
    
   test_data = pd.DataFrame(
        {
        "longitude": np.full(n_periods, lon),
        "latitude": np.full(n_periods, lat),
        "subsidence": rng.uniform(0.34, 0.50, n_periods),
        "date": dates,
        }
   )
   # Example data
   # forecast_results = pd.DataFrame({
   #     'longitude': [-103.8, -103.8, -103.8],
   #     'latitude': [0.47, 0.47, 0.47],
   #     'subsidence_q50': [0.3, 0.4, 0.5],
   #     'subsidence': [0.35, 0.42, 0.49], # Actuals can be in forecast_df
   #     'date': ['2023-01-01', '2023-01-02', '2023-01-03']
   # })
   # test_data = pd.DataFrame({ # Or actuals can be in a separate df
   #     'longitude': [-103.8, -103.8, -103.8],
   #     'latitude': [0.47, 0.47, 0.47],
   #     'subsidence': [0.36, 0.42, 0.50],
   #     'date': ['2023-01-01', '2023-01-02', '2023-01-03']
   # })

   # Generate a spatial quantile plot for the year 2023
    visualize_forecasts(
        forecast_df=forecast_results,
        test_data=test_data,
        dt_col="date",
        tname="subsidence",
        eval_periods=[2023],
        mode="quantile",
        kind="spatial",
        cmap="coolwarm",
        max_cols=2
    )

**Example Output Plot:**

.. figure:: ../images/visualize_forecasts_spatial_example.png
   :alt: Spatial forecast plot from visualize_forecasts
   :align: center
   :width: 90%

   Spatial visualization created by ``visualize_forecasts``. The left
   panel shows the actual ground truth subsidence for a given year,
   and the right panel shows the model's median (q50) prediction
   for comparison.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Further Examples
------------------

For more advanced and detailed examples of visualizations, including
animations and integration with specific model outputs, please visit
the :ref:`gallery_index`.

