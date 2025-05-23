.. _exercise_case_history_data_prep:

===========================================================
Exercise: Data Preparation Workflow for Case History Data
===========================================================

Welcome to this hands-on exercise! We will walk through a typical
data preparation pipeline for time series forecasting, using the
Zhongshan subsidence dataset available in ``fusionlab-learn``.
This exercise focuses on cleaning, feature engineering, encoding,
and scaling the data to make it ready for sequence generation
(e.g., using :func:`~fusionlab.nn.utils.reshape_xtft_data`) and
subsequent model training.

**Learning Objectives:**

* Load a sample dataset using ``fusionlab.datasets``.
* Perform initial data cleaning and validation, including datetime
  conversion and NaN handling.
* Generate relevant time series features (lags, rolling statistics,
  calendar features) using
  :func:`~fusionlab.utils.ts_utils.ts_engineering`.
* Encode categorical features for model consumption.
* Scale numerical features to improve model training.
* Define distinct feature sets (static, dynamic, future, target)
  for advanced forecasting models.
* Save processed data and scalers for reusability.

Let's begin!

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and its common dependencies
installed. We'll also use `joblib` for saving artifacts.

.. code-block:: bash

   pip install fusionlab-learn scikit-learn joblib matplotlib

---

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~
We start by importing the necessary libraries and ``fusionlab``
utilities. An output directory is also created for any artifacts
we generate, like scalers or processed data.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   import joblib
   import os
   import warnings
   import matplotlib.pyplot as plt # For initial data viz

   # FusionLab imports
   from fusionlab.datasets import fetch_zhongshan_data
   from fusionlab.utils.ts_utils import ts_engineering, to_dt
   from fusionlab.utils.data_utils import nan_ops

   # Suppress warnings and TF logs for cleaner output
   warnings.filterwarnings('ignore')
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress C++ level TF logs
   try:
       import tensorflow as tf
       tf.get_logger().setLevel('ERROR')
       if hasattr(tf, 'autograph'):
           tf.autograph.set_verbosity(0)
   except ImportError:
       print("TensorFlow not found, skipping TF log suppression.")

   # Configuration for outputs from this exercise
   exercise_output_dir_dataprep = "./zhongshan_data_prep_exercise_outputs"
   os.makedirs(exercise_output_dir_dataprep, exist_ok=True)

   print("Libraries imported and setup complete for data prep exercise.")

**Expected Output 1.1:**

.. code-block:: text

   Libraries imported and setup complete for data prep exercise.

Step 2: Load Zhongshan Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We'll use the :func:`~fusionlab.datasets.fetch_zhongshan_data`
function to load the sample Zhongshan subsidence dataset. This
function returns a ``Bunch`` object containing the DataFrame and
metadata about the columns.

.. code-block:: python
   :linenos:

   # Fetch the data as a Bunch object
   zhongshan_bunch = fetch_zhongshan_data(as_frame=False, verbose=0)
   df_raw = zhongshan_bunch.frame.copy() # Work with a copy

   print(f"Loaded Zhongshan data. Shape: {df_raw.shape}")
   print(f"Available columns: {df_raw.columns.tolist()}")
   print("\nSample of raw data:")
   print(df_raw.head())
   print("\nData types:")
   print(df_raw.info())

**Expected Output 2.2:**
   *(Details will match the `zhongshan_2000.csv` structure)*

.. code-block:: text

   Loaded Zhongshan data. Shape: (1999, 14)
   Available columns: ['longitude', 'latitude', 'year', 'GWL', 'seismic_risk_score', 'rainfall_mm', 'geology', 'normalized_density', 'density_tier', 'subsidence_intensity', 'density_concentration', 'normalized_seismic_risk_score', 'rainfall_category', 'subsidence']

   Sample of raw data:
       longitude   latitude  ...  rainfall_category  subsidence
   0  113.240334  22.476652  ...             Medium       15.51
   1  113.215866  22.510025  ...             Medium       31.60
   2  113.237984  22.494591  ...             Medium        8.09
   3  113.219109  22.513433  ...             Medium       15.49
   4  113.210678  22.536232  ...             Medium       14.02

   [5 rows x 14 columns]

   Data types:
   <class 'pandas.core.frame.DataFrame'>
   RangeIndex: 1999 entries, 0 to 1998
   Data columns (total 14 columns):
    #   Column                         Non-Null Count  Dtype  
   ---  ------                         --------------  -----  
    0   longitude                      1999 non-null   float64
    1   latitude                       1999 non-null   float64
    2   year                           1999 non-null   int64  
    3   GWL                            1999 non-null   float64
    4   seismic_risk_score             1999 non-null   float64
    5   rainfall_mm                    1999 non-null   float64
    6   geology                        1999 non-null   object 
    7   normalized_density             1999 non-null   float64
    8   density_tier                   1998 non-null   object 
    9   subsidence_intensity           1998 non-null   object 
    10  density_concentration          1999 non-null   object 
    11  normalized_seismic_risk_score  1999 non-null   float64
    12  rainfall_category              1999 non-null   object 
    13  subsidence                     1999 non-null   float64
   dtypes: float64(8), int64(1), object(5)
   memory usage: 218.8+ KB
   None

Step 3: Initial Data Cleaning and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The 'year' column is currently an integer. For time series operations,
it's often better to have a proper datetime index or column. We'll
convert 'year' to a datetime object representing the start of each
year. Then, we'll handle any missing values.

.. code-block:: python
   :linenos:

   dt_col_exercise = 'Date' # We will create this column
   df_clean = df_raw.copy()

   # Convert 'year' to a datetime column (start of year)
   df_clean[dt_col_exercise] = pd.to_datetime(
       df_clean['year'], format='%Y'
       )
   print(f"\nConverted 'year' to datetime column '{dt_col_exercise}'.")

   # Handle missing values using nan_ops for robust ffill/bfill
   print(f"NaNs before cleaning: "
         f"{df_clean.isna().any().sum()} columns have NaNs.")
   df_clean = nan_ops(df_clean, ops='sanitize', action='fill', # Uses ffill then bfill
                      verbose=0)
   print(f"NaNs after cleaning: "
         f"{df_clean.isna().any().sum()} columns have NaNs.")
   print("Cleaned data sample (with new 'Date' column):")
   print(df_clean[['Date', 'year', 'subsidence', 'GWL']].head())

**Expected Output 3.3:**

.. code-block:: text

   Converted 'year' to datetime column 'Date'.
   NaNs before cleaning: 2 columns have NaNs.
   NaNs after cleaning: 0 columns have NaNs.
   Cleaned data sample (with new 'Date' column):
           Date    year  subsidence       GWL
   0 2015-01-01  2015.0       15.51  2.865853
   1 2023-01-01  2023.0       31.60  1.924022
   2 2018-01-01  2018.0        8.09  0.752556
   3 2019-01-01  2019.0       15.49  1.043998
   4 2015-01-01  2015.0       14.02  1.700558

Step 4: Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate new time-based features (lags, rolling statistics, calendar
features) from the 'subsidence' column using
:func:`~fusionlab.utils.ts_utils.ts_engineering`.
This function requires data to be sorted by time for each group if
grouping is implicit. Since our data has multiple spatial points
per year, we should ideally group by spatial identifiers before
applying `ts_engineering` if lags/rolling stats are per location.
For this exercise, we'll apply it globally first, then drop NaNs.
A more advanced workflow would group by `['longitude', 'latitude']`
before this step.

.. code-block:: python
   :linenos:

   target_col_exercise = 'subsidence' # Target and base for engineering

   # For ts_engineering, ensure data is sorted if applying globally
   # or group by spatial identifiers first.
   # Here, we sort by ItemID (if exists) then Date.
   # Zhongshan data has 'longitude', 'latitude' as identifiers.
   # For simplicity, we'll sort by Date and assume global features.
   # A production workflow would group by 'longitude', 'latitude'.
   df_for_eng = df_clean.sort_values(by=[dt_col_exercise]).copy()

   df_featured = ts_engineering(
       df=df_for_eng,
       value_col=target_col_exercise,
       dt_col=dt_col_exercise, # Use the new 'Date' column
       lags=2,             # Create subsidence_lag_1, _lag_2
       window=3,               # Rolling mean/std over 3 periods (years)
       window_type='triang',   # Example window type
       seasonal_period=0,      # No explicit seasonal decomp here
       diff_order=0,
       apply_fourier=False,
       time_features=['year', 'month', 'quarter', 'day_of_week'],
       scaler=None             # Scale later
   )
   print(f"\nShape after feature engineering (before dropna): "
         f"{df_featured.shape}")

   # Drop rows with NaNs introduced by lags/rolling features
   df_featured.dropna(inplace=True) 
   df_featured.reset_index (inplace =True)
   print(f"Shape after dropna: {df_featured.shape}")
   print("Sample of engineered features (new columns):")
   print(df_featured[['Date', target_col_exercise, 'lag_1',
                      'rolling_mean_3', 'month', 'quarter']].head())

**Expected Output 4.4:**
   *(Shapes and new columns will reflect `ts_engineering` output)*

.. code-block:: text

   Shape after feature engineering (before dropna): (1997, 25)
   Shape after dropna: (1997, 26)
   Sample of engineered features (new columns):
           Date  subsidence  lag_1  rolling_mean_3  month  quarter
   0 2015-01-01        5.13  13.71       11.450000      1        1
   1 2015-01-01       22.17   5.13       13.670000      1        1
   2 2015-01-01       22.65  22.17       16.650000      1        1
   3 2015-01-01       10.98  22.65       18.600000      1        1
   4 2015-01-01        8.98  10.98       14.203333      1        1


Step 5: Categorical Feature Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Models require numerical inputs. Categorical features like 'geology'
and 'density_tier' from the Zhongshan dataset need to be encoded.
We'll use One-Hot Encoding.

.. code-block:: python
   :linenos:

   df_to_encode = df_featured.copy()
   categorical_cols_zhongshan = ['geology', 'density_tier']
   encoded_feature_names = [] # To store names of new one-hot columns

   for col in categorical_cols_zhongshan:
       if col in df_to_encode.columns:
           encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore',
                                   dtype=np.float32)
           encoded_data = encoder.fit_transform(df_to_encode[[col]])
           new_cols = [f"{col}_{cat.replace(' ', '_')}" for cat in encoder.categories_[0]]
           encoded_df_part = pd.DataFrame(
               encoded_data, columns=new_cols, index=df_to_encode.index
               )
           df_to_encode = pd.concat([df_to_encode, encoded_df_part], axis=1)
           df_to_encode.drop(columns=[col], inplace=True)
           encoded_feature_names.extend(new_cols)
           print(f"  Encoded '{col}' into: {new_cols}")
       else:
           print(f"  Warning: Categorical column '{col}' not found for encoding.")

   df_encoded = df_to_encode
   print(f"\nShape after one-hot encoding: {df_encoded.shape}")
   print(f"Added one-hot encoded columns: {encoded_feature_names}")

**Expected Output 5.5:**

.. code-block:: text

     Encoded 'geology' into: ['geology_Cohesive_Soil', 'geology_Gravelly_Soil', 'geology_Residual_Soil', 'geology_Rock', 'geology_Sand']
     Encoded 'density_tier' into: ['density_tier_High', 'density_tier_Low', 'density_tier_Medium']

   Shape after one-hot encoding: (1997, 32)
   Added one-hot encoded columns: ['geology_Cohesive_Soil', 'geology_Gravelly_Soil', 'geology_Residual_Soil', 'geology_Rock', 'geology_Sand', 'density_tier_High', 'density_tier_Low', 'density_tier_Medium']

Step 6: Define Final Feature Sets and Scale Numerical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now that all features are generated and encoded, we define our final
sets of static, dynamic, and future features. Then, we scale all
numerical features (including the target) that will be fed into the
model.

.. code-block:: python
   :linenos:

   df_for_scaling = df_encoded.copy()

   # Define feature sets based on available columns
   # Static: longitude, latitude, and one-hot encoded categoricals
   static_feature_names_ex = ['longitude', 'latitude'] + encoded_feature_names
   static_feature_names_ex = [
       c for c in static_feature_names_ex if c in df_for_scaling.columns
       ]

   # Dynamic: Original numericals + engineered numericals + calendar features
   dynamic_feature_names_ex = [
       'GWL', 'rainfall_mm', 'normalized_density',
       'normalized_seismic_risk_score',
       'subsidence_lag_1', 'subsidence_lag_2', # Check if lag_2 was created
       'rolling_mean_3', 'rolling_std_3',      # Check if _std_3 was created
       'month', 'quarter', 'day_of_week'     # Calendar features
   ]
   # Filter to existing columns
   dynamic_feature_names_ex = [
       c for c in dynamic_feature_names_ex if c in df_for_scaling.columns
       ]
   # Ensure target is not in dynamic features if it's handled separately
   if target_col_exercise in dynamic_feature_names_ex:
       dynamic_feature_names_ex.remove(target_col_exercise)

   # Future: For this example, assume some calendar features are "known future"
   # In a real scenario, these would be genuinely known ahead of time.
   future_feature_names_ex = ['month', 'quarter', 'day_of_week']
   future_feature_names_ex = [
       c for c in future_feature_names_ex if c in df_for_scaling.columns
       ]

   # Columns to be scaled: all numerical features including target
   # Exclude already one-hot encoded and simple calendar integers if
   # they are to be embedded or treated as categorical by the model.
   # For this exercise, we scale most numericals.
   numerical_cols_to_scale_ex = [
       'longitude', 'latitude', 'GWL', 'rainfall_mm',
       'normalized_density', 'normalized_seismic_risk_score',
       target_col_exercise # Include target
   ]
   # Add engineered numerical features if they exist
   engineered_to_scale = [
       'subsidence_lag_1', 'subsidence_lag_2',
       'rolling_mean_3', 'rolling_std_3'
       ]
   numerical_cols_to_scale_ex.extend(
       [c for c in engineered_to_scale if c in df_for_scaling.columns]
       )
   # Ensure unique and existing columns
   numerical_cols_to_scale_ex = list(set(
       c for c in numerical_cols_to_scale_ex if c in df_for_scaling.columns
       ))

   print(f"\nStatic features for model: {static_feature_names_ex}")
   print(f"Dynamic features for model: {dynamic_feature_names_ex}")
   print(f"Future features for model: {future_feature_names_ex}")
   print(f"Numerical columns to be scaled: {numerical_cols_to_scale_ex}")

   df_final_scaled = df_for_scaling.copy()
   if numerical_cols_to_scale_ex:
       scaler_final = StandardScaler()
       df_final_scaled[numerical_cols_to_scale_ex] = \
           scaler_final.fit_transform(
               df_final_scaled[numerical_cols_to_scale_ex]
               )
       scaler_path_final = os.path.join(
           exercise_output_dir_dataprep, "zhongshan_final_scaler.joblib"
           )
       joblib.dump(scaler_final, scaler_path_final)
       print(f"\nFinal numerical features scaled. Scaler saved to {scaler_path_final}")
   else:
       print("\nNo numerical columns identified for final scaling.")

   print("\nSample of fully processed data (first 5 rows, selected columns):")
   cols_to_show = static_feature_names_ex[:2] + \
                  dynamic_feature_names_ex[:2] + \
                  future_feature_names_ex[:1] + [target_col_exercise]
   cols_to_show = [c for c in cols_to_show if c in df_final_scaled.columns]
   print(df_final_scaled[cols_to_show].head())

**Expected Output 6.6:**
   *(Column lists and sample data will reflect the processing)*

.. code-block:: text

   Static features for model: ['longitude', 'latitude', 'geology_Cohesive_Soil', 'geology_Gravelly_Soil', 'geology_Residual_Soil', 'geology_Rock', 'geology_Sand', 'density_tier_High', 'density_tier_Low', 'density_tier_Medium']
   Dynamic features for model: ['GWL', 'rainfall_mm', 'normalized_density', 'normalized_seismic_risk_score', 'lag_1', 'lag_2', 'rolling_mean_3', 'rolling_std_3', 'month', 'quarter', 'day_of_week']
   Future features for model: ['month', 'quarter', 'day_of_week']
   Numerical columns to be scaled: ['GWL', 'latitude', 'normalized_seismic_risk_score', 'longitude', 'rolling_std_3', 'rainfall_mm', 'rolling_mean_3', 'normalized_density', 'lag_1', 'subsidence', 'lag_2']

   Final numerical features scaled. Scaler saved to ./zhongshan_data_prep_exercise_outputs\zhongshan_final_scaler.joblib

   Sample of fully processed data (first 5 rows, selected columns):
      longitude  latitude       GWL  rainfall_mm  month  subsidence
   0   0.379708 -1.794134 -0.556755    -1.135748      1   -0.804539
   1   0.750644  0.812747  0.390138    -0.973036      1    0.382239
   2   0.814110  0.736287  0.240789    -0.929659      1    0.415669
   3  -0.583759 -0.571861  0.551307    -1.252451      1   -0.397107
   4   0.225787 -2.117750 -0.392659    -1.088000      1   -0.536400

Step 7: Save Processed DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It's good practice to save the fully processed DataFrame. This allows
you to load it directly for sequence reshaping and model training in
future sessions without repeating all preprocessing steps.

.. code-block:: python
   :linenos:

   processed_df_path = os.path.join(
       exercise_output_dir_dataprep, "zhongshan_fully_processed_data.csv"
       )
   df_final_scaled.to_csv(processed_df_path, index=False)
   print(f"\nFully processed DataFrame saved to: {processed_df_path}")
   print(f"Final DataFrame shape: {df_final_scaled.shape}")

**Expected Output 7.7:**

.. code-block:: text

   Fully processed DataFrame saved to: ./zhongshan_data_prep_exercise_outputs\zhongshan_fully_processed_data.csv
   Final DataFrame shape: (1997, 32)

Discussion of Exercise
------------------------
In this exercise, we performed a comprehensive data preparation
workflow for the Zhongshan dataset:
1.  **Loaded** the raw data.
2.  **Cleaned** it by ensuring correct datetime formatting and handling
    missing values.
3.  **Engineered new features** like lags, rolling statistics, and
    calendar attributes using `ts_engineering`.
4.  **Encoded categorical features** (`geology`, `density_tier`) into
    a numerical format (one-hot encoding) suitable for machine
    learning models.
5.  **Scaled numerical features** (including the target variable) using
    `StandardScaler` to normalize their ranges.
6.  **Defined distinct sets of features** (static, dynamic, future)
    based on the processed data, ready for input into advanced
    forecasting models like TFT or XTFT (via
    `reshape_xtft_data`).

This prepared DataFrame (`df_final_scaled`) is now in a state where it
can be passed to :func:`~fusionlab.nn.utils.reshape_xtft_data` to
create sequences for training models. The saved scaler
(`zhongshan_final_scaler.joblib`) is essential for inverse-transforming
predictions back to their original scale.

This workflow demonstrates key steps in transforming raw time series
data from multiple sources/locations into a structured format suitable
for sophisticated deep learning forecasting models.

