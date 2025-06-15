.. _data_utils_guide:

=============================
Data Manipulation Utilities
=============================

The ``fusionlab.utils.data_utils`` module provides a collection of
powerful helper functions for common and often complex data manipulation
tasks encountered in time series and forecasting workflows.

This guide covers the primary utilities for handling missing values,
reshaping temporal data from long to wide formats, and performing
conditional data masking.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Advanced NaN Handling (`nan_ops`)
---------------------------------
:API Reference: :func:`~fusionlab.utils.data_utils.nan_ops`

This is a powerful, centralized function for managing missing values
(:math:`NaN`). It goes beyond a simple ``.dropna()`` or ``.fillna()``
by providing a robust framework for checking, validating, and sanitizing
data, with special handling for keeping feature and target arrays aligned.

Its three main modes of operation (`ops` parameter) are:
* **`'check_only'`**: Safely checks if `NaN` values exist and returns ``True`` or ``False``.
* **`'validate'`**: Asserts that no `NaN`s are present, raising an error if they are, which is useful for ensuring data integrity in a pipeline.
* **`'sanitize'`**: The main action mode, which can either ``'fill'`` or ``'drop'`` `NaN`s based on user-defined rules.

A key feature is its handling of auxiliary or "witness" data via the
`auxi_data` parameter. When rows containing `NaN`s are dropped from the
primary `data`, the **exact same rows** are dropped from `auxi_data`,
perfectly preserving the alignment between, for example, a target vector
and its corresponding feature matrix.

**Usage Example:**

In this example, we have a target `Series` and a feature `DataFrame` with
`NaN` values in different locations. We use `nan_ops` to drop rows from
both structures wherever the **target** has a `NaN`, keeping them aligned.

.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   from fusionlab.utils.data_utils import nan_ops

   # 1. Create a target Series and a feature DataFrame with NaNs
   target = pd.Series([10, 20, np.nan, 40, 50])
   features = pd.DataFrame({
       'A': [1, 2, 3, 4, 5],
       'B': [0.1, 0.2, 0.3, np.nan, 0.5] # NaN is in a different row
   })

   # 2. Sanitize by dropping rows where the target is NaN
   #    and apply the same row removal to the features.
   cleaned_target, cleaned_features = nan_ops(
       data=target,
       auxi_data=features,
       data_kind='target', # Specifies that `data` is the target
       ops='sanitize',
       action='drop'
   )

   # 3. Display the results
   print("--- Cleaned Target ---")
   print(cleaned_target)
   print("\n--- Aligned Features ---")
   print(cleaned_features)

**Expected Output:**

.. code-block:: text

   --- Cleaned Target ---
   0    10.0
   1    20.0
   3    40.0
   4    50.0
   dtype: float64

   --- Aligned Features ---
        A    B
   0  1.0  0.1
   1  2.0  0.2
   3  4.0  NaN
   4  5.0  0.5

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Reshaping Time Series Data (`widen_temporal_columns`)
------------------------------------------------------
:API Reference: :func:`~fusionlab.utils.data_utils.widen_temporal_columns`

This utility transforms a **long-format** forecast DataFrame into a
**wide-format** one. This is a specific type of pivot operation where
temporal information is encoded into the column names. For example, a
column `subsidence_q50` with a time value of `2019` becomes a new
column named `subsidence_2019_q50`.

This reshaping is particularly useful for creating summary tables or for
analysis methods that require each time series sample to be represented
by a single row.

**Usage Example:**

.. code-block:: python
   :linenos:

   from fusionlab.utils.data_utils import widen_temporal_columns

   # 1. Create a sample long-format forecast DataFrame
   df_long = pd.DataFrame({
       "coord_x": [113.15, 113.15, 113.20, 113.20],
       "coord_y": [22.63, 22.63, 22.80, 22.80],
       "coord_t": [2019, 2020, 2019, 2020],
       "subsidence_q50": [9.1, 10.2, 12.5, 13.6],
       "subsidence_actual": [9.0, 10.5, 12.4, 13.8],
       "GWL_q50": [5.5, 5.3, 4.1, 4.0],
       "region": ["A", "A", "B", "B"]
   })

   # 2. Convert to wide format
   df_wide = widen_temporal_columns(
       df_long,
       dt_col="coord_t",
       spatial_cols=("coord_x", "coord_y"),
       ignore_cols=["region"], # Carry this static column through
       verbose=1
   )

   # 3. Display the wide DataFrame
   print(df_wide)

**Expected Output:**

.. code-block:: text

   [INFO] Initial rows: 4, columns: 2
   [INFO] Widening base 'GWL' (1 columns)
   [INFO] Widening base 'subsidence' (2 columns)
   [DONE] Final wide shape: (2, 9)
      coord_x  coord_y region  GWL_2019_q50  GWL_2020_q50  subsidence_2019_actual  subsidence_2020_actual  subsidence_2019_q50  subsidence_2020_q50
   0   113.15    22.63      A           5.5           5.3                     9.0                    10.5                  9.1                 10.2
   1   113.20    22.80      B           4.1           4.0                    12.4                    13.8                 12.5                 13.6

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Conditional Data Masking (`mask_by_reference`)
-----------------------------------------------
:API Reference: :func:`~fusionlab.utils.data_utils.mask_by_reference`

This function provides a powerful way to conditionally modify data. It
finds rows in a DataFrame based on a value in a `ref_col` and then
replaces the values in *other* columns of those matched rows with a
specified `fill_value`.

It is highly flexible, supporting both exact matching and approximate
(closest value) matching for numeric data. This is useful for data
cleaning, scenario creation, or preparing data for specific modeling
techniques.

**Usage Example:**

.. code-block:: python
   :linenos:

   from fusionlab.utils.data_utils import mask_by_reference

   # 1. Create a sample DataFrame
   df = pd.DataFrame({
       "A_code": [10, 0, 8, 0],
       "B_value": [2.0, 0.5, 18.0, 85.0],
       "C_value": [34.0, 0.8, 12.0, 4.5],
   })
   print("--- Original DataFrame ---")
   print(df)

   # 2. Example: Find rows where 'A_code' is exactly 0 and
   #    mask only the 'C_value' column with -999.
   df_masked = mask_by_reference(
       data=df,
       ref_col="A_code",
       values=0,
       fill_value=-999,
       mask_columns=["C_value"] # Only affect this column
   )

   print("\n--- Masked DataFrame ---")
   print(df_masked)

**Expected Output:**

.. code-block:: text

   --- Original DataFrame ---
      A_code  B_value  C_value
   0      10      2.0     34.0
   1       0      0.5      0.8
   2       8     18.0     12.0
   3       0     85.0      4.5

   --- Masked DataFrame ---
      A_code  B_value  C_value
   0      10      2.0     34.0
   1       0      0.5   -999.0
   2       8     18.0     12.0
   3       0     85.0   -999.0

