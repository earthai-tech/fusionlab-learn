.. _geo_utils_guide:

===========================================
Geospatial & Time Series Data Utilities
===========================================

This section covers the specialized utility functions available in
``fusionlab.utils.geo_utils``. These helpers are designed to handle
common data preparation and augmentation tasks required for building
robust spatiotemporal forecasting models.

The utilities focus on handling missing data in time series, augmenting
feature sets to improve model generalization, and generating synthetic
data for rapid prototyping and testing of Physics-Informed Neural
Networks (PINNs).


Temporal Gap Interpolation (`interpolate_temporal_gaps`)
---------------------------------------------------------
:API Reference: :func:`~fusionlab.utils.geo_utils.interpolate_temporal_gaps`

When working with real-world sensor data, you often encounter two
types of missing information: individual `NaN` values within an
existing timestamp, and entire missing timestamps (e.g., a sensor was
offline for a day). This function is designed to handle both cases
gracefully.

Its key feature is the optional `freq` parameter. When provided (e.g.,
`freq='D'` for daily), it first reindexes the DataFrame to a complete,
regular time index. This action creates rows for any missing
timestamps, filled with `NaN` values. The function then applies a
chosen interpolation strategy to fill all `NaN` values in the specified
columns.

**Usage Example:**

In this example, our initial DataFrame is missing the date `2020-01-02`
entirely, and the value for `2020-01-03` is `NaN`. By setting
`freq='D'`, we create a complete daily index, and `method='linear'`
fills all the gaps.

.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   from fusionlab.utils.geo_utils import interpolate_temporal_gaps

   # 1. Create a sample DataFrame with missing dates and values
   df = pd.DataFrame({
       'date': ['2020-01-01', '2020-01-03', '2020-01-06'],
       'value_a': [10.0, np.nan, 19.0],
       'value_b': [100, 120, 150]
   })
   print("--- Original DataFrame ---")
   print(df)

   # 2. Interpolate on a daily frequency
   result_df = interpolate_temporal_gaps(
       df,
       time_col='date',
       value_cols=['value_a'], # Only interpolate this column
       freq='D', # Reindex to daily frequency
       method='linear'
   )

   # 3. Display the result
   # Note: value_b is forward-filled into the new empty rows.
   print("\n--- Interpolated DataFrame ---")
   print(result_df)

**Expected Output:**

.. code-block:: text

   --- Original DataFrame ---
           date  value_a  value_b
   0  2020-01-01     10.0      100
   1  2020-01-03      NaN      120
   2  2020-01-06     19.0      150

   --- Interpolated DataFrame ---
           date  value_a  value_b
   0 2020-01-01     10.0      100
   1 2020-01-02     13.0      100
   2 2020-01-03     16.0      120
   3 2020-01-04     17.0      120
   4 2020-01-05     18.0      120
   5 2020-01-06     19.0      150

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Data Augmentation via Noise Injection (`augment_series_features`)
-----------------------------------------------------------------
:API Reference: :func:`~fusionlab.utils.geo_utils.augment_series_features`

Data augmentation is a common technique to increase the diversity of a
training set, which can help a model generalize better and reduce
overfitting. This function provides a simple way to augment time series
data by adding a small amount of random noise to specified feature columns.

The magnitude of the noise is controlled by `noise_level` and is scaled
relative to the standard deviation (`gaussian`) or range (`uniform`)
of the original feature, ensuring the augmentation is proportional to
the feature's natural variance.

**Usage Example:**

.. code-block:: python
   :linenos:

   from fusionlab.utils.geo_utils import augment_series_features

   # 1. Create a sample DataFrame
   df = pd.DataFrame({
       'feature_1': [100.0, 102.0, 105.0, 103.0],
       'feature_2': [10.0, 11.0, 9.0, 12.0]
   })

   # 2. Add Gaussian noise with a level of 0.05 (5%) to 'feature_1'
   df_augmented = augment_series_features(
       df,
       feature_cols=['feature_1'],
       noise_level=0.05,
       noise_type='gaussian',
       random_seed=42 # For reproducibility
   )

   # 3. Display the result
   print("--- Original vs. Augmented ---")
   print(df.rename(columns={'feature_1': 'original_f1'}))
   print("\n")
   print(df_augmented.rename(columns={'feature_1': 'augmented_f1'}))

**Expected Output:**

.. code-block:: text

   --- Original vs. Augmented ---
      original_f1  feature_2
   0        100.0       10.0
   1        102.0       11.0
   2        105.0        9.0
   3        103.0       12.0


      augmented_f1  feature_2
   0    100.111225       10.0
   1    101.970119       11.0
   2    105.145021        9.0
   3    102.934005       12.0

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Spatiotemporal Data Augmentation Pipeline (`augment_spatiotemporal_data`)
--------------------------------------------------------------------------
:API Reference: :func:`~fusionlab.utils.geo_utils.augment_spatiotemporal_data`

This is a high-level wrapper function that orchestrates a complete
data augmentation pipeline for spatiotemporal datasets. It is particularly
useful when you have data from multiple locations (or groups) and need
to apply processing steps to each group independently.

* **`interpolate`**: Fills in missing temporal gaps for each group.
* **`augment_features`**: Adds noise to specified features.
* **`both`**: Performs interpolation first, then adds noise.

**Usage Example:**

Here, we have a dataset for two sites, each with missing data. We use
the pipeline to first interpolate the missing values for each site,
and then add noise to the `rainfall` feature.

.. code-block:: python
   :linenos:

   from fusionlab.utils.geo_utils import augment_spatiotemporal_data

   # 1. Create data for two sites with missing values
   df_multi_site = pd.DataFrame({
       'site_id': ['A', 'A', 'B', 'B', 'B'],
       'date': ['2021-01-01', '2021-01-03', '2021-01-01', '2021-01-02', '2021-01-04'],
       'rainfall': [10, np.nan, 50, 55, np.nan],
       'temperature': [15, 17, 25, 26, 28]
   })

   # 2. Run the full pipeline: interpolate then augment
   df_processed = augment_spatiotemporal_data(
       df=df_multi_site,
       mode='both',
       group_by_cols=['site_id'],
       time_col='date',
       value_cols_interpolate=['rainfall', 'temperature'],
       feature_cols_augment=['rainfall'],
       interpolation_kwargs={'freq': 'D'},
       augmentation_kwargs={'noise_level': 0.1, 'random_seed': 42}
   )

   # 3. Display the result, grouped by site to see the effect
   print("--- Augmented Data by Site ---")
   for site, group in df_processed.groupby('site_id'):
       print(f"\nSite: {site}")
       print(group)

**Expected Output:**

.. code-block:: text

   --- Augmented Data by Site ---

   Site: A
     site_id       date   rainfall  temperature
   0       A 2021-01-01   9.751643         15.0
   1       A 2021-01-02  13.430868         16.0
   2       A 2021-01-03  16.623844         17.0

   Site: B
     site_id       date   rainfall  temperature
   3       B 2021-01-01  49.654826         25.0
   4       B 2021-01-02  55.513335         26.0
   5       B 2021-01-03  56.096958         27.0
   6       B 2021-01-04  55.626883         28.0

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Generating Synthetic PINN Data (`generate_dummy_pinn_data`)
--------------------------------------------------------------
:API Reference: :func:`~fusionlab.utils.geo_utils.generate_dummy_pinn_data`

This is a convenience utility for quickly generating a synthetic dataset
that mimics the structure required for the library's PINN models. It's
invaluable for creating reproducible examples, writing tests, or
prototyping a model before using real data.

The function creates a dictionary of NumPy arrays with standard keys
(e.g., `year`, `longitude`, `subsidence`, `GWL`) and populates them
with random data drawn from plausible ranges.

**Usage Example:**

.. code-block:: python
   :linenos:
   
   from fusionlab.utils.geo_utils import generate_dummy_pinn_data

   # 1. Generate 5 samples of dummy data
   dummy_pinn_data = generate_dummy_pinn_data(n_samples=5)

   # 2. Convert to DataFrame for easy inspection
   df_dummy = pd.DataFrame(dummy_pinn_data)

   # 3. Display the result
   print(df_dummy)

**Expected Output:**

.. code-block:: text

      year   longitude   latitude  subsidence       GWL  rainfall_mm
   0  2007  113.722687  22.756353  -13.486241  3.013589  1694.373535
   1  2012  113.111870  22.569485  -20.914291  2.529849  1263.818237
   2  2024  113.477722  22.392769  -18.136894  2.646533  1442.929810
   3  2003  113.424492  22.427742  -34.121510  2.249431  2415.699707
   4  2004  113.615173  22.575974  -17.764954  2.343292  1489.193237