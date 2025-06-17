.. _user_guide_datasets:

==========
Datasets
==========

The ``fusionlab.datasets`` module provides access to sample datasets
relevant to the forecasting tasks addressed by the library (like land
subsidence) and includes tools for generating synthetic datasets. These
can be useful for testing models, demonstrating utilities, and running
examples.

Loading Included Datasets
---------------------------

These functions load pre-packaged (or downloadable) datasets, often
derived from real-world studies but potentially sampled or processed
for convenience. They typically handle locating the data file (checking
local cache, package data, and optionally downloading).

.. _fetch_zhongshan_data_doc:

fetch_zhongshan_data
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.datasets.fetch_zhongshan_data`

Loads the sampled Zhongshan land subsidence dataset
(`zhongshan_2000.csv`). This file contains ~2000 data points spatially
sampled from a larger dataset [Liu24]_, including raw features like
coordinates, year, GWL, rainfall, geology, seismic risk, density
metrics, and the target `subsidence` variable.

The function can return the data as a pandas DataFrame or a scikit-learn
style :class:`~fusionlab.api.bunch.Bunch` object containing the data
and metadata. It also supports further sub-sampling via the `n_samples`
parameter.

**Basic Usage (Load as Bunch):**

.. code-block:: python
   :linenos:

   from fusionlab.datasets import fetch_zhongshan_data

   # Load the full dataset (~2000 samples) as a Bunch
   zhongshan_bunch = fetch_zhongshan_data(as_frame=False)

   # Access the DataFrame
   print("Zhongshan DataFrame shape:", zhongshan_bunch.frame.shape)

   # Access target values
   print("Target shape:", zhongshan_bunch.target.shape)

   # Access feature names
   print("Feature names:", zhongshan_bunch.feature_names)

.. _fetch_nansha_data_doc:

fetch_nansha_data
~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.datasets.fetch_nansha_data`

Loads the sampled Nansha land subsidence dataset (`nansha_2000.csv`).
Similar to the Zhongshan dataset, this contains ~2000 spatially
sampled data points with features relevant to subsidence in the
Nansha area, including coordinates, year, geological information,
hydrogeology, building concentration, soil thickness, risk scores,
and the target `subsidence`.

It provides the same options as `Workspace_zhongshan_data` for returning
a DataFrame or Bunch, sub-sampling (`n_samples`), and controlling
data loading/caching.

**Basic Usage (Load Sample as DataFrame):**

.. code-block:: python
   :linenos:

   from fusionlab.datasets import fetch_nansha_data

   # Load a random spatial sample of 500 points as a DataFrame
   nansha_df_sample = fetch_nansha_data(
       n_samples=500,
       as_frame=True,
       random_state=42, # for reproducibility
       verbose=False # suppress messages
       )

   print(f"Loaded Nansha sample shape: {nansha_df_sample.shape}")
   print(nansha_df_sample.head())


.. _load_processed_subsidence_data_doc:

load_processed_subsidence_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.datasets.load_processed_subsidence_data`

This function provides a higher-level pipeline that **loads** one of the
raw datasets (`zhongshan_2000.csv` or `nansha_2000.csv` via the `Workspace_`
functions), applies a predefined **preprocessing workflow** (feature
selection, NaN handling, categorical encoding, numerical scaling), and
optionally **reshapes** the data into sequences suitable for TFT/XTFT
models using :func:`~fusionlab.utils.ts_utils.reshape_xtft_data`.

It includes options to control which preprocessing steps are applied
and utilizes caching for processed DataFrames and generated sequences
to speed up repeated calls.

**Basic Usage (Get Processed Frame):**

.. code-block:: python
   :linenos:

   from fusionlab.datasets import load_processed_subsidence_data

   # Load and preprocess Zhongshan data, return as DataFrame
   # Applies default preprocessing: feature select, nan fill, one-hot, minmax scale
   df_processed = load_processed_subsidence_data(
       dataset_name='zhongshan',
       return_sequences=False, # Get the processed DataFrame
       as_frame=True,
       use_processed_cache=True, # Try to load from cache first
       save_processed_frame=True # Save if reprocessed
   )
   print("Processed Zhongshan DataFrame info:")
   df_processed.info()

**Usage for Model Training (Get Sequences):**

.. code-block:: python
   :linenos:

   from fusionlab.datasets import load_processed_subsidence_data

   # Load Zhongshan, preprocess, and generate sequences
   static, dynamic, future, target = load_processed_subsidence_data(
       dataset_name='zhongshan',
       return_sequences=True, # Request sequence arrays
       time_steps=12,         # Example lookback
       forecast_horizon=6,    # Example horizon
       use_sequence_cache=True,
       save_sequences=True
   )
   print("\nGenerated sequences for model training:")
   print(f"Static shape: {static.shape}")
   print(f"Dynamic shape: {dynamic.shape}")
   print(f"Future shape: {future.shape}")
   print(f"Target shape: {target.shape}")

.. _load_subsidence_pinn_data_guide:

load_subsidence_pinn_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.datasets.load.load_subsidence_pinn_data`

This function is the recommended **all-in-one data preparation
pipeline** for any project using the library's PINN models,
such as ``PIHALNet`` and ``TransFlowSubsNet``.

It is designed to handle the entire data ingestion and preprocessing
workflow, which is particularly complex for physics-informed models.
By abstracting away common steps like data loading, cleaning, encoding,
scaling, and optional augmentation, it saves a significant amount of
boilerplate code and helps prevent common errors, allowing you to focus
on the modeling itself.

**End-to-End Workflow**

The function executes a comprehensive, multi-stage workflow, with each
stage being configurable through the function's parameters.

**1. Data Sourcing & Caching**

The first step is to get the data. The function employs an
efficient strategy:

* **Caching:** If ``use_cache=True``, it first checks for a pre-processed
  version of the data in your local cache directory. If found, it loads
  this file instantly, skipping all subsequent processing steps and
  saving significant time on repeated runs.
* **Data Loading:** If no cache is found, it proceeds to load the raw
  data according to the ``strategy`` parameter: ``'load'`` (requires the
  file to exist), ``'fallback'`` (loads the file if present, otherwise
  generates dummy data), or ``'generate'`` (always creates a new dummy
  dataset, great for testing).
* **Saving Cache:** After processing the data for the first time, you
  can set ``save_cache=True`` to save the fully processed results,
  including the DataFrame and any fitted scalers/encoders, for fast
  retrieval in the future.

**2. Automated Core Preprocessing**

Once the raw data is loaded, the function performs a full preprocessing
pipeline:

* It ensures essential columns (coordinates, time, targets) exist and
  drops rows where they are missing.
* It robustly converts the time column (e.g., integer years) into a
  proper datetime object for internal calculations.
* It **one-hot encodes** specified categorical columns (like `geology`).
* It creates a continuous **numerical time coordinate**, which is
  essential for computing derivatives in the PINN loss function.
* It **scales** specified numerical features to a [0, 1] range to ensure
  stable model training. The fitted `scaler` and `encoder` objects are
  saved along with the data.

**3. Optional Data Augmentation**

By setting ``augment_data=True``, you can invoke the
:func:`~fusionlab.utils.geo_utils.augment_city_spatiotemporal_data`
pipeline. This can perform two types of augmentation on the data before
it is returned:

* **Temporal Interpolation:** Fills in missing time steps in your data
  (e.g., missing years) for each location.
* **Feature Augmentation:** Adds a small amount of random noise to
  feature columns to create a larger, more diverse training set, which
  can improve model robustness.

**4. Flexible Output Format**

The function can return its results in two convenient formats,
controlled by ``return_dataframe``:

* A single, fully processed ``pandas.DataFrame``.
* A :mod:`~fusionlab.api.bunch` **``XBunch``** object. This is often the preferred
  output, as it's a self-contained object that bundles the processed
  DataFrame (`.frame`) with crucial metadata like feature names
  (`.feature_names`), target names (`.target_names`), and a
  human-readable description of all the processing steps that were
  applied (`.DESCR`).

**Usage Example**

This example demonstrates how to use the function to load, process,
and augment a dataset in a single call. For reproducibility, we first
create a dummy CSV file to simulate a raw data source.

.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   import os
   from fusionlab.datasets.load import load_subsidence_pinn_data
   from fusionlab.utils.geo_utils import generate_dummy_pinn_data

   # --- 1. Create a dummy raw data file for the example ---
   DUMMY_DATA_DIR = "./dummy_data"
   # The function will look inside a 'data' subdirectory of data_home
   os.makedirs(DUMMY_DATA_DIR, exist_ok=True)
   dummy_data_path = os.path.join(DUMMY_DATA_DIR, "zhongshan_2000.csv")

   dummy_dict = generate_dummy_pinn_data(n_samples=100)
   dummy_dict['geology'] = np.random.choice(['Clay', 'Sand'], 100)
   pd.DataFrame(dummy_dict).to_csv(dummy_data_path, index=False)
   print(f"Dummy data created at: {dummy_data_path}")

   # --- 2. Use the pipeline to load and process the data ---
   # We will load from the file, encode 'geology', scale numericals,
   # and perform augmentation, returning a rich Bunch object.
   processed_bunch = load_subsidence_pinn_data(
       data_name='zhongshan',        # This configures internal column names
       strategy='load',              # Explicitly load from file
       data_home=DUMMY_DATA_DIR,     # Tell the function where to look
       encode_categoricals=True,     # Enable one-hot encoding
       scale_numericals=True,        # Enable MinMax scaling
       augment_data=True,            # Enable augmentation
       augment_mode='interpolate',   # Specify interpolation mode
       use_cache=False,              # Disable caching for this demo
       as_frame=False                # Return the rich XBunch object
   )

   # --- 3. Inspect the output ---
   print("\n--- Processed DataFrame (from Bunch) ---")
   # The XBunch contains the processed DataFrame in the 'frame' attribute
   print(processed_bunch.frame.head())

   print("\n--- Description of Processing (from Bunch) ---")
   print(processed_bunch.DESCR)


**Expected Output:**

.. code-block:: text

   Dummy data created at: ./dummy_data/data/zhongshan_2000.csv
   ... (Log messages from the function will appear here) ...

   --- Processed DataFrame (from Bunch) ---
           year  longitude  latitude  ...  geology_Clay  geology_Sand  year_numeric
   0 2008-01-01   113.0084   22.3616  ...           1.0           0.0        2008.0
   1 2017-01-01   113.0172   22.3231  ...           1.0           0.0        2017.0
   2 2010-01-01   113.0226   22.7769  ...           1.0           0.0        2010.0
   3 2008-01-01   113.0289   22.4596  ...           1.0           0.0        2008.0
   4 2021-01-01   113.0308   22.7131  ...           0.0           1.0        2021.0

   [5 rows x 10 columns]

   --- Description of Processing (from Bunch) ---
   Processed Zhongshan PINN data.
   Load Strategy: load.
   Cache Used: No, Cache Path: N/A.
   Categorical Encoding: Applied.
   Numerical Scaling: minmax.
   Augmentation: Applied (Mode: interpolate).
   Rows: 100, Features: 7 (in 'data' array).
   Targets: ['subsidence', 'GWL'].
   Coordinate Precision: 4 decimal places.
   Time Column (numeric): year_numeric.
   ...
   
.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Generating Synthetic Datasets
-----------------------------

The ``fusionlab.datasets.make`` module provides functions to create
synthetic datasets with specific characteristics. These are useful for:

* Testing model implementations (TFT, XTFT, etc.).
* Demonstrating specific features or components.
* Creating reproducible examples for documentation or tutorials.
* Evaluating algorithms under controlled conditions (e.g., specific
    trend types, anomaly patterns).

.. _make_multi_feature_time_series_util:

make_multi_feature_time_series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.datasets.make.make_multi_feature_time_series`

**Purpose:** Generates a multi-variate dataset across multiple
independent series (e.g., items, locations), including static,
dynamic (past), and known future features, along with a target variable
influenced by these components plus trend, seasonality, and noise.

**Functionality:** Creates a DataFrame simulating data suitable for
models like :class:`~fusionlab.nn.transformers.TFT` and
:class:`~fusionlab.nn.XTFT`. Key generated features include:
* ``series_id`` (static)
* `base_level` (static, noisy per series)
* `month`, `dayofweek` (dynamic/future)
* `dynamic_cov` (simulated dynamic covariate)
* `target_lag1` (dynamic)
* `future_event` (simulated binary future covariate)
* `target` (combination of inputs + trend + seasonality + noise)

**Usage Context:** Ideal for creating a complete, structured dataset
from scratch to test the end-to-end workflow of TFT/XTFT models,
including data preparation with
:func:`~fusionlab.nn.utils.reshape_xtft_data`.

**Code Example:**

.. code-block:: python
   :linenos:

   from fusionlab.datasets.make import make_multi_feature_time_series

   # Generate data for 3 series, 50 steps each
   data_bunch = make_multi_feature_time_series(
       n_series=3,
       n_timesteps=50,
       freq='D',        # Daily frequency
       seasonality_period=7, # Weekly seasonality
       seed=42,
       as_frame=False   # Return Bunch object
   )

   print("--- Multi-Feature Time Series Bunch ---")
   print("Generated DataFrame shape:", data_bunch.frame.shape)
   print("Static features:", data_bunch.static_features)
   print("Dynamic features:", data_bunch.dynamic_features)
   print("Future features:", data_bunch.future_features)
   print("Target column:", data_bunch.target_col)
   print("\nSample Data:")
   print(data_bunch.frame.head())


.. _make_quantile_prediction_data_util:

make_quantile_prediction_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.datasets.make.make_quantile_prediction_data`

**Purpose:** Generates a dataset simulating the typical **output** of a
multi-horizon quantile forecasting model. It includes columns for
actual target values and corresponding predicted quantiles for multiple
steps ahead.

**Functionality:** Creates a DataFrame in a "wide" format where columns
represent different forecast horizons (:math:`h`) and quantiles (:math:`q`).

* Target columns: `target_h1`, `target_h2`, ...
* Prediction columns: `pred_qX_h1`, `pred_qY_h1`, ..., `pred_qX_h2`, ...
Actual values are drawn from a normal distribution, and predictions are
generated around a potentially biased version of the actuals, with spread
controlled by parameters.

**Usage Context:** Useful for testing or demonstrating evaluation metrics
and visualization functions that operate on quantile forecast results (e.g.,
calculating pinball loss, coverage scores, or plotting prediction intervals
against actuals).

**Code Example:**

.. code-block:: python
   :linenos:

   from fusionlab.datasets.make import make_quantile_prediction_data

   # Generate data for 10 samples, 5 horizons, 3 quantiles
   quantiles = [0.1, 0.5, 0.9]
   pred_data_bunch = make_quantile_prediction_data(
       n_samples=10,
       n_horizons=5,
       quantiles=quantiles,
       seed=123,
       as_frame=False # Return Bunch
   )

   print("\n--- Quantile Prediction Data Bunch ---")
   print("Generated DataFrame shape:", pred_data_bunch.frame.shape)
   print("Available quantiles:", pred_data_bunch.quantiles)
   print("Target columns:", pred_data_bunch.target_cols)
   print("Prediction columns for q0.5:",
         pred_data_bunch.prediction_cols.get('q0.5', 'N/A'))
   print("\nSample DataFrame:")
   print(pred_data_bunch.frame.head(3))


.. _make_anomaly_data_util:

make_anomaly_data
~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.datasets.make.make_anomaly_data`

**Purpose:** Generates univariate sequence data where a specified
fraction of the sequences contain injected anomalies (either spikes or
level shifts).

**Functionality:** Creates normal sequences (e.g., sine wave + noise)
and injects anomalies into a subset based on `anomaly_fraction` and
`anomaly_type`. Returns the sequence data and corresponding binary
labels (0=normal, 1=anomaly).

**Usage Context:** Designed for creating simple datasets to test anomaly
detection algorithms (like
:class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`) or
anomaly-aware training strategies.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.datasets.make import make_anomaly_data

   # Generate 100 sequences, 20% with spike anomalies
   sequences, labels = make_anomaly_data(
       n_sequences=100,
       sequence_length=50,
       n_features=1, # Required
       anomaly_fraction=0.2,
       anomaly_type='spike',
       anomaly_magnitude=10.0,
       seed=42,
       as_frame=False # Return numpy arrays
   )

   print("\n--- Anomaly Data ---")
   print(f"Generated sequences shape: {sequences.shape}")
   print(f"Generated labels shape: {labels.shape}")
   print(f"Number of normal sequences: {np.sum(labels == 0)}")
   print(f"Number of anomalous sequences: {np.sum(labels == 1)}")


.. _make_trend_seasonal_data_util:

make_trend_seasonal_data
~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.datasets.make.make_trend_seasonal_data`

**Purpose:** Generates a univariate time series with clearly defined
and controllable polynomial trend and multiple sinusoidal seasonal
components, plus noise.

**Functionality:** Combines a polynomial trend (order specified by
`trend_order` and coefficients by `trend_coeffs`) with one or more
sine waves (defined by `seasonal_periods` and `seasonal_amplitudes`)
and adds Gaussian noise (`noise_level`).

**Usage Context:** Useful for testing specific aspects of time series
models, such as their ability to capture linear or non-linear trends,
handle multiple overlapping seasonalities, or for demonstrating time
series decomposition utilities like
:func:`~fusionlab.utils.ts_utils.decompose_ts`.

**Code Example:**

.. code-block:: python
   :linenos:

   from fusionlab.datasets.make import make_trend_seasonal_data
   import matplotlib.pyplot as plt

   # Generate data with quadratic trend, weekly + monthly seasonality
   data_bunch = make_trend_seasonal_data(
       n_timesteps=90, # 3 months daily
       freq='D',
       trend_order=2, trend_coeffs=[20, 0.1, 0.01], # Quadratic
       seasonal_periods=[7, 30.5], # Weekly & approx Monthly
       seasonal_amplitudes=[3, 8],
       noise_level=0.5,
       seed=99
   )

   print("\n--- Trend + Seasonal Data ---")
   print("Generated DataFrame shape:", data_bunch.frame.shape)
   print(data_bunch.frame.head())

   # Simple plot
   # data_bunch.frame.plot(x='date', y='value', figsize=(10,4),
   #                       title="Generated Trend + Seasonal Data")
   # plt.show()



.. _make_multivariate_target_data_util:

make_multivariate_target_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.datasets.make.make_multivariate_target_data`

**Purpose:** Generates synthetic data simulating multiple time series
(e.g., different items or locations) where each series has not only
static, dynamic, and future features but also **multiple target
variables** that may exhibit some interdependencies.

**Functionality:**
Extends the logic of :func:`make_multi_feature_time_series` to
generate `n_targets` distinct target columns (e.g., 'target_1',
'target_2', ...). The generation process includes:

* A base signal incorporating trend and seasonality.
* Noise specific to each target.
* An optional lagged dependency where `target_N` is influenced by
  `target_{N-1}` from a previous time step (`cross_target_lag`),
  controlled by `cross_target_factor`.

**Usage Context:** Useful for developing, testing, or demonstrating
forecasting models that are capable of performing **multivariate
forecasting**, i.e., predicting multiple related target variables
simultaneously (e.g., predicting sales and inventory for multiple
stores). The generated data mimics scenarios where target variables might
influence each other over time.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   # Assuming make_multivariate_target_data is importable
   from fusionlab.datasets.make import make_multivariate_target_data

   # Generate data for 2 series, 50 steps, 3 target variables
   multi_target_bunch = make_multivariate_target_data(
       n_series=2,
       n_timesteps=50,
       n_targets=3,
       freq='D',
       seasonality_period=7,
       cross_target_factor=0.4, # Example dependency
       seed=123,
       as_frame=False # Return Bunch object
   )

   print("\n--- Multi-Target Data Bunch ---")
   print("Generated DataFrame shape:", multi_target_bunch.frame.shape)
   print("Static features:", multi_target_bunch.static_features)
   print("Dynamic features:", multi_target_bunch.dynamic_features)
   print("Future features:", multi_target_bunch.future_features)
   # Check multiple target names and target array shape
   print("Target names:", multi_target_bunch.target_names)
   print("Target array shape:", multi_target_bunch.target.shape)
   print("\nSample Data:")
   # Display target columns
   print(multi_target_bunch.frame[
       ['date', 'series_id'] + multi_target_bunch.target_names
       ].head())


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
