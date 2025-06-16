.. _forecast_utils_guide:

====================================
Forecast Data Formatting Utilities
====================================

After generating forecasts, the raw model output—often a long-format
DataFrame where each row represents a single time step—needs to be
reshaped and structured for effective analysis and visualization. The
``fusionlab.utils.forecast_utils`` module provides specialized
functions for these data transformation tasks.

This guide covers the primary utilities for pivoting your forecast
results from a long to a wide format.


Pivoting from Long to Wide Format (`pivot_forecast_dataframe`)
--------------------------------------------------------------
:API Reference: :func:`~fusionlab.utils.forecast_utils.pivot_forecast_dataframe`

This is the core utility for transforming time series forecast data. It
converts a **long-format** DataFrame into a **wide-format** DataFrame.

* **Long Format**: Each row represents a single observation at a
  specific time for a specific sample (e.g., one row for the
  forecast of "site A" at "step 1", another for "site A" at "step 2").
* **Wide Format**: Each row represents a single unique sample, and the
  time steps are spread across the columns (e.g., one row for "site A"
  with columns like `prediction_step1`, `prediction_step2`, etc.).

This transformation is often necessary for certain types of analysis or
for plotting libraries that expect data in a wide structure.

**Key Parameters:**

* **data**: The input long-format DataFrame.
* **id_vars**: The list of columns that uniquely identify each
  sample (e.g., `['sample_idx', 'longitude', 'latitude']`). These
  remain as index-like columns in the output.
* **time_col**: The name of the column whose values will become the
  new column headers (e.g., `forecast_step` or `year`).
* **value_prefixes**: A list of base names (e.g., `['subsidence', 'GWL']`)
  that the function uses to identify all the value columns that need
  to be pivoted.

**Usage Example:**

.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   from fusionlab.utils.forecast_utils import pivot_forecast_dataframe

   # 1. Create a sample long-format DataFrame
   data = {
       'sample_idx':      [0, 0, 1, 1],
       'coord_t':         [2018.0, 2019.0, 2018.0, 2019.0],
       'coord_x':         [113.5, 113.5, 113.8, 113.8],
       'subsidence_q50':  [-8, -9, -13, -14],
       'subsidence_actual': [-8.5, -8.5, -13.2, -13.2],
       'GWL_q50':         [1.2, 1.3, 2.2, 2.3],
   }
   df_long = pd.DataFrame(data)
   print("--- Original Long-Format DataFrame ---")
   print(df_long)

   # 2. Pivot the DataFrame to wide format
   df_wide = pivot_forecast_dataframe(
       data=df_long,
       id_vars=['sample_idx', 'coord_x'],
       time_col='coord_t',
       value_prefixes=['subsidence', 'GWL'],
       static_actuals_cols=['subsidence_actual'],
       # Treat the float time column as integer years for naming
       time_col_is_float_year=True
   )

   # 3. Display the result
   print("\n--- Pivoted Wide-Format DataFrame ---")
   print(df_wide)

**Expected Output:**

.. code-block:: text

   --- Original Long-Format DataFrame ---
      sample_idx  coord_t  coord_x  subsidence_q50  subsidence_actual  GWL_q50
   0           0   2018.0    113.5              -8               -8.5      1.2
   1           0   2019.0    113.5              -9               -8.5      1.3
   2           1   2018.0    113.8             -13              -13.2      2.2
   3           1   2019.0    113.8             -14              -13.2      2.3

   --- Pivoted Wide-Format DataFrame ---
      sample_idx  coord_x  subsidence_actual  GWL_2018_q50  GWL_2019_q50  subsidence_2018_q50  subsidence_2019_q50
   0           0    113.5               -8.5           1.2           1.3                   -8                   -9
   1           1    113.8              -13.2           2.2           2.3                  -13                  -14

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Smart Format Detection and Conversion (`format_forecast_dataframe`)
-------------------------------------------------------------------
:API Reference: :func:`~fusionlab.utils.forecast_utils.format_forecast_dataframe`

This function is a high-level "smart wrapper" around the core pivoting
utility. Its main purpose is to **auto-detect** the format of your
DataFrame (either 'long' or 'wide') and then conditionally pivot it
only if necessary.

This is useful in data pipelines where you might receive data in
either format and want to ensure it conforms to a specific structure
(typically 'wide') before the next step.

**Usage Example:**

.. code-block:: python
   :linenos:

   from fusionlab.utils.forecast_utils import format_forecast_dataframe

   # Use the same long-format DataFrame from the previous example
   df_long = pd.DataFrame({
       'sample_idx': [0, 0, 1, 1],
       'coord_t': [2018.0, 2019.0, 2018.0, 2019.0],
       'subsidence_q50':  [-8, -9, -13, -14],
   })

   # --- Case 1: Detect the format without converting ---
   detected_format = format_forecast_dataframe(df_long, to_wide=False)
   print(f"Detected format: '{detected_format}'")

   # --- Case 2: Ensure the output is wide format ---
   # Since the input is long, this will call the pivot function internally.
   df_wide_smart = format_forecast_dataframe(
       df_long,
       to_wide=True,
       id_vars=['sample_idx'],
       time_col='coord_t',
       value_prefixes=['subsidence']
   )
   print("\n--- DataFrame after ensuring wide format ---")
   print(df_wide_smart)

**Expected Output:**

.. code-block:: text

   Detected format: 'long'

   --- DataFrame after ensuring wide format ---
      sample_idx  subsidence_2018.0_q50  subsidence_2019.0_q50
   0           0                     -8                     -9
   1           1                    -13                    -14