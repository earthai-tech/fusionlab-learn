.. _example_zhongshan_data_prep:

===========================
Data Preparation Workflow
===========================

Preparing time series data correctly is crucial for training effective
forecasting models like TFT and XTFT. This example demonstrates a
typical workflow using utilities from ``fusionlab-learn`` to transform raw
time series data into the structured sequence format required by these
models.

We will cover the following steps:

1.  Imports and Configuration Setup.
2.  Loading raw data.
3.  Basic cleaning and datetime validation.
4.  Generating time series features (lags, rolling stats, etc.).
5.  (Optional) Feature Selection.
6.  Defining feature sets and scaling numerical features.
7.  Reshaping the data into sequences using ``reshape_xtft_data``.
8.  Splitting the sequences into training, validation, and test sets.
9.  (Optional) Saving the processed data.

.. contents::
   :local:
   :depth: 2

Step 1: Imports and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, we import the necessary libraries: ``pandas`` for data handling,
``numpy`` for numerical operations, ``StandardScaler`` from
``scikit-learn`` for feature scaling, ``joblib`` for saving artifacts,
and the relevant utilities from ``fusionlab``. We also set up an output
directory and suppress warnings.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   from sklearn.preprocessing import StandardScaler
   # from sklearn.model_selection import train_test_split # Not used directly here
   import joblib # For saving scalers
   import os
   import warnings

   # FusionLab imports
   # Corrected import for ts_utils
   from fusionlab.utils.ts_utils import (
       to_dt,
       ts_engineering
       # select_and_reduce_features # Import if using this optional step
   )
   from fusionlab.nn.utils import reshape_xtft_data
   # Assuming fetch functions are in fusionlab.datasets.load
   from fusionlab.datasets import fetch_zhongshan_data # Example

   # Suppress warnings for cleaner output
   warnings.filterwarnings('ignore')
   # Suppress TensorFlow logs if TF is used later
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   try:
       import tensorflow as tf
       tf.get_logger().setLevel('ERROR')
       if hasattr(tf, 'autograph'):
            tf.autograph.set_verbosity(0)
   except ImportError:
       pass # TensorFlow not used directly in this script yet

   # --- Configuration ---
   output_dir = "./data_prep_workflow_output" # Unique name
   os.makedirs(output_dir, exist_ok=True)
   print(f"Artifacts will be saved to: {output_dir}")


Step 2: Load Raw Data
~~~~~~~~~~~~~~~~~~~~~~~
Load your initial dataset. For this example, we use
:func:`~fusionlab.datasets.fetch_zhongshan_data` to get a sample
dataset. In a real scenario, you would replace this with loading your
own data file (e.g., using `pd.read_csv`).

.. code-block:: python
   :linenos:

   # Using fetch_zhongshan_data for a reproducible example
   # This loads the zhongshan_2000.csv file
   try:
       df_raw_bunch = fetch_zhongshan_data(as_frame=False, verbose=False)
       df_raw = df_raw_bunch.frame
       print(f"Loaded raw data using fetch_zhongshan_data. Shape: {df_raw.shape}")
   except Exception as e:
       print(f"Could not load data using fetch_zhongshan_data: {e}")
       print("Falling back to generating synthetic data for demonstration.")
       # Fallback synthetic data generation (from original script)
       n_items = 3
       n_timesteps = 48
       date_rng = pd.date_range(start='2018-01-01', periods=n_timesteps, freq='MS')
       df_list = []
       for item_id in range(n_items):
           time = np.arange(n_timesteps)
           sales = (
               100 + item_id * 50 + time * (2 + item_id) +
               30 * np.sin(2 * np.pi * time / 12) +
               np.random.normal(0, 15, n_timesteps)
           )
           temp = 15 + 10 * np.sin(2*np.pi*(time % 12)/12 + np.pi) + np.random.normal(0,2)
           promo = np.random.randint(0, 2, n_timesteps)
           item_df = pd.DataFrame({
               'Date': date_rng, 'ItemID': f'item_{item_id}', # Use string ItemID
               'Temperature': temp, 'PlannedPromotion': promo, 'Sales': sales
           })
           df_list.append(item_df)
       df_raw = pd.concat(df_list).reset_index(drop=True)
       # Add 'geology' and 'density_tier' for Zhongshan example consistency
       df_raw['geology'] = np.random.choice(['Sand', 'Clay', 'Rock'], size=len(df_raw))
       df_raw['density_tier'] = np.random.choice(['Low', 'Medium', 'High'], size=len(df_raw))
       df_raw['normalized_density'] = np.random.rand(len(df_raw))
       df_raw['normalized_seismic_risk_score'] = np.random.rand(len(df_raw))
       df_raw['GWL'] = np.random.rand(len(df_raw)) * 10
       df_raw['rainfall_mm'] = np.random.rand(len(df_raw)) * 100
       # Rename 'Sales' to 'subsidence' to match Zhongshan context if needed
       # For this generic workflow, we'll keep 'Sales' as target_col.
       # If using fetch_zhongshan_data, target is 'subsidence'.
       if 'subsidence' in df_raw.columns and 'Sales' not in df_raw.columns:
           df_raw.rename(columns={'subsidence': 'Sales'}, inplace=True)


   print(f"Using data with shape: {df_raw.shape}")
   print("Raw data sample:")
   print(df_raw.head(3))


Step 3: Initial Cleaning & Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ensure the time column is in the correct datetime format using
:func:`~fusionlab.utils.ts_utils.to_dt`. Handle any initial missing
values using an appropriate strategy (here, forward fill `ffill` then
backward fill `bfill` for robustness).

.. code-block:: python
   :linenos:

   dt_col = 'year' # Datetime column
   # Ensure datetime column is correct format
   df_clean = to_dt(df_raw.copy(), dt_col=dt_col, error='raise')

   # Handle missing values (example: forward fill then backward fill)
   print(f"\nNaNs before ffill/bfill: {df_clean.isna().any().sum()} columns")
   df_clean = df_clean.ffill().bfill()
   print(f"NaNs after ffill/bfill: {df_clean.isna().any().sum()} columns")
   print("Initial cleaning complete.")


Step 4: Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate additional time series features using
:func:`~fusionlab.utils.ts_utils.ts_engineering`.
The `value_col` parameter specifies the column based on which lags
and rolling statistics are computed.

.. code-block:: python
   :linenos:

   target_col = 'subsidence' # This is the column we want to predict
                        # and also the base for lags/rolling stats.

   # Ensure target_col exists before feature engineering
   if target_col not in df_clean.columns:
       raise KeyError(
           f"Target column '{target_col}' for feature engineering "
           f"not found in cleaned DataFrame. Available columns: "
           f"{df_clean.columns.tolist()}"
       )

   df_featured = ts_engineering(
       df=df_clean.copy(),
       value_col=target_col, # Generate features based on 'Sales'
       dt_col=dt_col,        # Use 'Date' for time-based features
       lags=3,               # Create Sales_lag_1, _lag_2, _lag_3
       window=6,             # Create rolling_mean_6, rolling_std_6
       diff_order=0,         # No differencing in this example
       apply_fourier=False,  # No Fourier features
       scaler=None            # Scaling will be applied later
   )
   print("\nGenerated time series features using ts_engineering.")
   print(f"Shape before dropna from engineering: {df_featured.shape}")
   print(f"Columns after ts_engineering: {df_featured.columns.tolist()}")

   # Drop rows with NaNs introduced by lags/rolling features
   # This is crucial as these rows cannot be used for training.
   df_featured.dropna(inplace=True)
   print(f"Shape after dropna: {df_featured.shape}")

   if df_featured.empty:
       warnings.warn(
           "DataFrame became empty after ts_engineering and dropna. "
           "This might happen if the original data is too short for "
           "the specified lags/window, or if data is not properly "
           "grouped before these operations if ts_engineering expects it."
           "Check your data and ts_engineering parameters."
       )
   else:
       print("NaNs dropped from engineered features.")


Step 5: Feature Selection / Reduction (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This step is optional. If you have generated many features, you might
apply techniques to reduce dimensionality or remove redundancy.

.. code-block:: python
   :linenos:

   # For this example, we'll use all features generated by ts_engineering
   # that remain after dropna.
   df_selected = df_featured.copy()
   print(f"\nSkipped optional feature selection. "
         f"Using shape: {df_selected.shape}")


Step 6: Define Feature Sets & Scale Numerics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Define the final lists of static, dynamic, and future columns based on
the features now present in the `df_selected` DataFrame. Then, apply
scaling to the numerical features.

.. code-block:: python
   :linenos:

   # Define feature sets AFTER engineering/selection
   # ItemID should be label encoded if used as a static feature.
   # For this generic workflow, assume ItemID is for grouping in reshape_xtft_data
   # and not directly a feature passed to the model unless encoded.
   static_cols = ['ItemID'] if 'ItemID' in df_selected.columns else []

   # Dynamic cols include original numerics and engineered time features
   # (lags, rolling stats, calendar features from ts_engineering)
   # Start with known numericals from original data (excluding target if scaled separately)
   dynamic_cols = ['Temperature', 'GWL', 'rainfall_mm',
                   'normalized_density', 'normalized_seismic_risk_score']
   # Add engineered features if they exist
   engineered_dyn_features = [
       'lag_1', 'lag_2', 'lag_3',
       'rolling_mean_6', 'rolling_std_6',
       'year', 'month', 'day', 'day_of_week', 'is_weekend', 'quarter'
   ]
   dynamic_cols.extend(
       [col for col in engineered_dyn_features if col in df_selected.columns]
       )
   # Filter to only existing columns in df_selected
   dynamic_cols = [col for col in dynamic_cols if col in df_selected.columns]
   # Remove target_col from dynamic_cols if it's there
   if target_col in dynamic_cols:
       dynamic_cols.remove(target_col)


   # Future cols include known future promotions and calendar features
   future_cols = [ 'year', 'month', 'day',
                  'day_of_week', 'is_weekend', 'quarter']
   future_cols = [col for col in future_cols if col in df_selected.columns]

   # Columns to be scaled: original numerics + engineered numericals + target
   numerical_cols_for_scaling = [
       'GWL', 'rainfall_mm',
       'normalized_density', 'normalized_seismic_risk_score',
       target_col # Ensure target is scaled
       ]
   engineered_num_to_scale = [
       'lag_1', 'lag_2', 'lag_3',
       'rolling_mean_6', 'rolling_std_6'
       ]
   numerical_cols_for_scaling.extend(
       [col for col in engineered_num_to_scale if col in df_selected.columns]
       )
   # Ensure only existing columns are scaled
   numerical_cols_for_scaling = list(set(
       col for col in numerical_cols_for_scaling if col in df_selected.columns
       ))

   df_scaled = df_selected.copy()
   if not df_scaled.empty and numerical_cols_for_scaling:
       scaler = StandardScaler()
       df_scaled[numerical_cols_for_scaling] = scaler.fit_transform(
           df_scaled[numerical_cols_for_scaling]
       )
       scaler_path = os.path.join(output_dir, "feature_scaler.joblib")
       joblib.dump(scaler, scaler_path)
       print(f"\nScaled numerical features: {numerical_cols_for_scaling}")
       print(f"Scaler saved to {scaler_path}")
   elif df_scaled.empty:
       print("\nDataFrame is empty, skipping scaling.")
   else:
       print("\nNo numerical columns to scale.")

   # Note on categorical features:
   # 'ItemID' (if used as static feature) and other categoricals like 'geology'
   # (from Zhongshan example) or time features ('Month', 'DayOfWeek')
   # would typically be label encoded or passed to model embedding layers.
   # This script assumes they are handled by model embeddings if not scaled.
   # If OneHotEncoding is needed, it should be done before this scaling step,
   # and the one-hot encoded columns (which are numeric) should NOT be scaled.


Step 7: Reshape into Sequences using `reshape_xtft_data`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use :func:`~fusionlab.nn.utils.reshape_xtft_data` to transform the
processed, scaled DataFrame into structured NumPy arrays.

.. code-block:: python
   :linenos:

   time_steps = 12
   forecast_horizons = 6
   # Use 'ItemID' for grouping if it exists, otherwise no spatial grouping
   current_spatial_cols = ['ItemID'] if 'ItemID' in df_scaled.columns else None

   print(f"\nReshaping data (T={time_steps}, H={forecast_horizons})...")
   if df_scaled.empty:
       print("DataFrame is empty. Cannot reshape into sequences.")
       static_data, dynamic_data, future_data, target_data = (
           None, None, None, None # Or empty arrays with 0 samples
           )
   else:
       # Ensure all column lists passed to reshape_xtft_data only contain
       # columns that actually exist in df_scaled.
       final_static_cols = [c for c in static_cols if c in df_scaled.columns]
       final_dynamic_cols = [c for c in dynamic_cols if c in df_scaled.columns]
       final_future_cols = [c for c in future_cols if c in df_scaled.columns]
       final_spatial_cols = [c for c in current_spatial_cols if c in df_scaled.columns] \
           if current_spatial_cols else None

       static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
           df=df_scaled.reset_index(drop=True), # drop one index to avoid collision
           dt_col=dt_col,
           target_col=target_col,
           dynamic_cols=final_dynamic_cols,
           static_cols=final_static_cols,
           future_cols=final_future_cols,
           spatial_cols=final_spatial_cols,
           time_steps=time_steps,
           forecast_horizons=forecast_horizons,
           verbose=1
       )
   if static_data is not None: # Check if sequence generation was successful
        print(f"  Static sequences shape: {static_data.shape}")
        print(f"  Dynamic sequences shape: {dynamic_data.shape}")
        print(f"  Future sequences shape: {future_data.shape}")
        print(f"  Target sequences shape: {target_data.shape}")
        


Step 8: Train / Validation / Test Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Split the generated sequence arrays chronologically.

.. code-block:: python
   :linenos:

   if target_data is not None and target_data.shape[0] > 0:
       n_samples = target_data.shape[0] # Use target_data for num samples
       n_val = int(n_samples * 0.15)
       n_test = int(n_samples * 0.15)
       n_train = n_samples - n_val - n_test

       if n_train <=0 or n_val <=0: # Basic check for enough samples
           print(f"Warning: Not enough samples for train/val/test split. "
                 f"Total samples: {n_samples}, Train: {n_train}, Val: {n_val}")
           # Handle this case, e.g., by adjusting split or raising error
           X_train_static, X_val_static, X_test_static = (None, None, None)
           X_train_dynamic, X_val_dynamic, X_test_dynamic = (None, None, None)
           X_train_future, X_val_future, X_test_future = (None, None, None)
           y_train, y_val, y_test = (None, None, None)
       else:
           X_train_static, X_val_static, X_test_static = (
               static_data[:n_train],
               static_data[n_train:n_train + n_val],
               static_data[n_train + n_val:]
           )
           X_train_dynamic, X_val_dynamic, X_test_dynamic = (
               dynamic_data[:n_train],
               dynamic_data[n_train:n_train + n_val],
               dynamic_data[n_train + n_val:]
           )
           X_train_future, X_val_future, X_test_future = (
               future_data[:n_train],
               future_data[n_train:n_train + n_val],
               future_data[n_train + n_val:]
           )
           y_train, y_val, y_test = (
               target_data[:n_train],
               target_data[n_train:n_train + n_val],
               target_data[n_train + n_val:]
           )

       print("\nData split into Train/Validation/Test sets:")
       print(f"  Train Samples : {n_train if n_train > 0 else 0}")
       print(f"  Val.  Samples : {n_val if n_val > 0 else 0}")
       print(f"  Test  Samples : {n_test if n_test > 0 else 0}")
       if X_train_dynamic is not None:
           print(f"  Example Train Dynamic Shape: {X_train_dynamic.shape}")
   else:
       print("\nNo sequence data to split.")


Step 9: Save Processed Data (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optionally, save the final processed sequence arrays.

.. code-block:: python
   :linenos:

   if y_train is not None: # Check if training data exists
       processed_data_path = os.path.join(output_dir, "processed_sequences.npz")
       np.savez(
           processed_data_path,
           X_train_static=X_train_static, X_val_static=X_val_static,
           X_train_dynamic=X_train_dynamic, X_val_dynamic=X_val_dynamic,
           X_train_future=X_train_future, X_val_future=X_val_future,
           y_train=y_train, y_val=y_val,
           X_test_static=X_test_static, X_test_dynamic=X_test_dynamic, # Save test too
           X_test_future=X_test_future, y_test=y_test
       )
       print(f"\nProcessed sequence data saved to {processed_data_path}")
   else:
       print("\nNo training data to save.")

