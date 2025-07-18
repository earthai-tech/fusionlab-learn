.. _forecast_plotting_guide:

===================================
Forecast Visualization Utilities
===================================

``fusionlab-learn`` provides a powerful suite of plotting functions
specifically designed to visualize time series forecast results. These
utilities are built on ``pandas`` and ``matplotlib``, offering a
flexible and intuitive way to inspect model predictions, compare them
against actual values, and analyze performance across different
dimensions like time, space, or forecast horizon.

This guide covers the main functions for generating temporal line plots,
spatial scatter plots, and structured grid visualizations of your
forecast data.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

General-Purpose Forecasting Plots (`plot_forecasts`)
------------------------------------------------------
:API Reference: :func:`~fusionlab.plot.forecast.plot_forecasts`

This is the primary and most versatile plotting function. It is designed
to handle a wide variety of forecasting visualization tasks from a
single, standardized long-format DataFrame (typically generated by
:func:`~fusionlab.nn.utils.format_predictions_to_dataframe`).

**Key Capabilities:**

* **Temporal & Spatial Modes:** Can generate both time series line
  plots (``kind='temporal'``) for individual samples and spatial
  scatter plots (``kind='spatial'``) for specific forecast steps.
* **Probabilistic Visualization:** Natively handles quantile forecasts,
  plotting the median prediction as a line and the outer quantiles
  as a shaded uncertainty interval.
* **Actuals Comparison:** Can automatically overlay true actual
  values for direct comparison with model predictions.
* **Scaler Integration:** Can automatically apply the
  ``inverse_transform`` method of a provided ``scikit-learn``-like
  scaler to display results in their original physical units.

Example 1: Temporal Quantile Forecast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to plot the forecast for a few individual samples
over the entire forecast horizon, complete with uncertainty bounds.

.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   from fusionlab.plot.forecast import plot_forecasts
   from fusionlab.nn.utils import format_predictions_to_dataframe

   # 1. Assume we have model predictions and true values
   B, H, O, Q_len = 10, 7, 1, 3
   preds_quant = np.random.rand(B, H, Q_len) * 2
   y_true_seq = preds_quant[:, :, 1, np.newaxis] + np.random.randn(B, H, O) * 0.2
   quantiles_list = [0.1, 0.5, 0.9]

   # 2. Format data into the required long-format DataFrame
   df_quant = format_predictions_to_dataframe(
       predictions=preds_quant, y_true_sequences=y_true_seq,
       target_name="subsidence", quantiles=quantiles_list,
       forecast_horizon=H, output_dim=O
   )

   # 3. Generate the plot for the first 2 samples
   plot_forecasts(
       df_quant,
       target_name="subsidence",
       quantiles=quantiles_list,
       kind="temporal",
       sample_ids="first_n", # Plot the first N samples
       num_samples=2,
       max_cols=2 # Arrange plots in 2 columns
   )

**Expected Output:**

.. figure:: ../../../images/forecast_plot_temporal.png
   :alt: Temporal Quantile Forecast Plot
   :align: center
   :width: 95%

   A figure with two subplots, each showing the forecast for a single
   sample. The plots include the true actual values (dashed line), the
   median prediction (solid line), and the shaded uncertainty interval.

Example 2: Spatial Forecast Snapshot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example shows how to visualize the spatial distribution of a forecast
at a single step into the future. This requires the DataFrame to have
columns containing spatial coordinates.

.. code-block:: python
   :linenos:

   # Add spatial coordinates to our previous DataFrame
   df_quant['longitude'] = np.random.rand(len(df_quant)) * 50
   df_quant['latitude'] = np.random.rand(len(df_quant)) * 50

   # 3. Generate the spatial plot for the 3rd forecast step
   plot_forecasts(
       df_quant,
       target_name="subsidence",
       quantiles=quantiles_list,
       kind="spatial",
       horizon_steps=3, # Visualize the 3rd step of the horizon
       spatial_cols=['longitude', 'latitude'],
       cbar='uniform' # Use a uniform color bar for all plots
   )

**Expected Output:**

.. figure:: ../../../images/forecast_plot_spatial.png
   :alt: Spatial Forecast Plot
   :align: center
   :width: 60%

   A single spatial scatter plot for the 3rd forecast step. The color
   of each point represents the predicted median subsidence value at that
   location.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Visualizing by Forecast Step (`plot_forecast_by_step`)
-------------------------------------------------------
:API Reference: :func:`~fusionlab.plot.forecast.plot_forecast_by_step`

This function is specifically designed to analyze how a model's forecast
evolves across its prediction horizon. It creates a grid of plots where
each **row represents a forecast step** (e.g., 1-step-ahead,
2-steps-ahead, etc.). This is extremely useful for diagnosing error
propagation and understanding how far into the future the model remains
reliable.

.. code-block:: python
   :linenos:

   from fusionlab.plot.forecast import plot_forecast_by_step

   # Use the same DataFrame from the previous example
   plot_forecast_by_step(
       df=df_quant,
       value_prefixes=['subsidence'],
       spatial_cols=('longitude', 'latitude'),
       # Create side-by-side plots for actual vs. predicted
       kind='dual',
       # Provide custom names for the steps
       step_names={
           1: "1-Step Ahead", 3: "3-Steps Ahead", 5: "5-Steps Ahead"
       },
       # Only plot steps 1, 3, and 5
       steps=[1, 3, 5]
   )

**Expected Output:**

.. figure:: ../../../images/forecast_plot_by_step.png
   :alt: Forecast by Step Plot
   :align: center
   :width: 95%

   A grid of plots with three rows, one for each forecast step (1, 3,
   and 5). Each row contains multiple columns showing the spatial
   distribution of the actual values and the predicted quantiles for
   that specific step.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Yearly/Periodic Spatial Views (`forecast_view`)
-----------------------------------------------
:API Reference: :func:`~fusionlab.plot.forecast.forecast_view`

This function is a powerful tool for creating a grid of spatial
forecast plots organized by a specific time period, such as a **year**.
It is ideal for comparing how the spatial distribution of a forecast
evolves across different evaluation periods (e.g., comparing the
predicted subsidence map for 2023 vs. 2024).

The function is designed to work with a "wide-format" DataFrame where
columns represent different years and metrics. It includes an
internal helper to automatically pivot data from a long format if
necessary, making it user-friendly.

**Key Capabilities:**

* **Period-Based Grid:** Automatically creates rows in the output
  figure for each unique period (e.g., year) specified in
  ``view_years``.
* **Metric-Based Columns:** Arranges plots column-wise to compare
  different metrics (e.g., "Actuals", "p50 Forecast", "p90 Forecast")
  side-by-side for the same period.
* **Uniform Color Scaling:** Can enforce a single, uniform color scale
  (via ``cbar='uniform'``) across all subplots, making it easy to
  visually compare magnitudes between different years and metrics.
* **Automatic Data Handling:** Intelligently detects value prefixes
  (like 'subsidence' or 'gwl') and quantile levels from the DataFrame
  columns.

Usage Example
~~~~~~~~~~~~~~
This example demonstrates how to use ``forecast_view`` to compare the
spatial forecast for two different years, showing the actual values
alongside two predicted quantiles.

.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   from fusionlab.plot.forecast import forecast_view

   # 1. Create a sample long-format DataFrame with a 'year' column
   n_points_per_year = 50
   data = {
       'coord_x': np.random.rand(n_points_per_year * 2),
       'coord_y': np.random.rand(n_points_per_year * 2),
       'year': [2023] * n_points_per_year + [2024] * n_points_per_year,
       'subsidence_actual': np.random.randn(n_points_per_year * 2),
       'subsidence_q10': np.random.randn(n_points_per_year * 2) * 0.8,
       'subsidence_q90': np.random.randn(n_points_per_year * 2) * 1.2,
   }
   df_yearly = pd.DataFrame(data)
   
   # 2. add the `sample_idx` column (one id per row here; adjust as needed)
   df_yearly["sample_idx"] = np.arange(len(df_yearly))
   # ── or, if you have repeated measurements per point:
   # df_yearly["sample_idx"] = (df_yearly["coord_x"].round(3).astype(str)
   #                            + "_" +
   #                            df_yearly["coord_y"].round(3).astype(str))

   # 2. Generate the plot
   forecast_view(
       forecast_df=df_yearly,
       dt_col="year",
       value_prefixes=['subsidence'],
       spatial_cols=('coord_x', 'coord_y'),
       kind='dual', # Show actuals and predictions
       # Specify which years and quantiles to visualize
       view_years=[2023, 2024],
       view_quantiles=[0.1, 0.9],
       cmap ='seismic', 
       s=100, 
       cbar='uniform'
   )

**Expected Output:**

.. figure:: ../../../images/forecast_view_yearly.png
   :alt: Yearly Forecast View Plot
   :align: center
   :width: 95%

   A grid of plots with two rows (for years 2023 and 2024). Each row
   shows the actual spatial distribution for that year, followed by the
   predicted spatial distributions for the 10th and 90th quantiles.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Legacy Comparison Plot (`visualize_forecasts`)
----------------------------------------------
:API Reference: :func:`~fusionlab.plot.forecast.visualize_forecasts`

The ``visualize_forecasts`` function is an earlier utility for creating
a grid of scatter plots that compare actual values from a ``test_data``
DataFrame with predicted values from a ``forecast_df``.

.. note::
   For most use cases, the newer and more flexible
   :func:`~fusionlab.plot.forecast.plot_forecasts` and
   :func:`~fusionlab.plot.forecast.forecast_view` functions are
   recommended. ``visualize_forecasts`` is maintained for specific
   workflows where forecast and test data are stored in separate
   DataFrames.

Its primary design is to create a plot grid where each specified
evaluation period gets its own **pair** of subplots: one for the actual
data and one for the predicted data, facilitating direct visual
comparison.

.. code-block:: python
   :linenos:

   from fusionlab.plot.forecast import visualize_forecasts

   # 1. Create separate forecast and test DataFrames
   forecast_data = pd.DataFrame({
       'longitude': np.random.rand(50),
       'latitude': np.random.rand(50),
       'subsidence_q50': np.random.randn(50),
       'date': pd.to_datetime(['2023-06-01'] * 50)
   })
   test_data_actuals = pd.DataFrame({
       'longitude': np.random.rand(50),
       'latitude': np.random.rand(50),
       'subsidence': np.random.randn(50),
       'date': pd.to_datetime(['2023-06-01'] * 50)
   })

   # 2. Generate the comparison plot
   visualize_forecasts(
       forecast_df=forecast_data,
       test_data=test_data_actuals,
       dt_col="date",
       tname="subsidence",
       eval_periods=['2023'], # Must match the year in the data
       mode="quantile",
       kind="spatial",
       x="longitude",
       y="latitude",
       max_cols=1, 
       s=100,
   )

**Expected Output:**

.. figure:: ../../../images/visualize_forecasts_comparison.png
   :alt: Legacy Forecast Comparison Plot
   :align: center
   :width: 70%

   A figure with two subplots stacked vertically. The top plot shows
   the spatial distribution of the actual `subsidence` values from the
   `test_data` DataFrame for the year 2023. The bottom plot shows the
   predicted median (`q50`) `subsidence` values from the `forecast_df`.