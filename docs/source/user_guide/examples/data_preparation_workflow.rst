.. _example_data_prep:

===========================
Data Preparation Workflow
===========================

Preparing time series data correctly is crucial for training effective
forecasting models like TFT and XTFT. This example demonstrates a
typical workflow using utilities from ``fusionlab`` to transform raw
time series data into the structured sequence format required by these
models.

We will cover:
1. Loading raw data.
2. Basic cleaning and datetime validation.
3. Generating time series features (lags, rolling stats, etc.).
4. Scaling numerical features.
5. Reshaping the data into sequences suitable for models handling
   static, dynamic, and future inputs.
6. Splitting the sequences into training, validation, and test sets.

Code Example
------------

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import train_test_split
   import joblib # For saving scalers
   import os

   # Assuming fusionlab components are importable
   from fusionlab.core.io import read_data # Example loader
   from fusionlab.utils.ts_utils import (
       to_dt,
       ts_engineering,
       reshape_xtft_data,
       # select_and_reduce_features # Optional step
   )

   # Suppress warnings for cleaner output
   import warnings
   warnings.filterwarnings('ignore')

   # --- Configuration ---
   output_dir = "./data_prep_output" # Directory to save artifacts
   os.makedirs(output_dir, exist_ok=True)

   # 1. Load Raw Data
   # ----------------
   # Using synthetic data generation from previous XTFT example for demo
   # In practice, load your raw CSV/data file here
   # df_raw = read_data("path/to/your/raw_data.csv")
   n_items = 3
   n_timesteps = 48 # 4 years of monthly data
   date_rng = pd.date_range(start='2018-01-01', periods=n_timesteps, freq='MS')
   df_list = []
   for item_id in range(n_items):
       time = np.arange(n_timesteps)
       sales = (
           100 + item_id * 50 + time * (2 + item_id) +
           30 * np.sin(2 * np.pi * time / 12) +
           np.random.normal(0, 15, n_timesteps)
       )
       temp = 15 + 10 * np.sin(2 * np.pi * (time % 12) / 12 + np.pi) + np.random.normal(0, 2)
       promo = np.random.randint(0, 2, n_timesteps)

       item_df = pd.DataFrame({
           'Date': date_rng, 'ItemID': item_id,
           'Temperature': temp, 'PlannedPromotion': promo, 'Sales': sales
       })
       df_list.append(item_df)
   df_raw = pd.concat(df_list).reset_index(drop=True)
   print(f"Loaded raw data shape: {df_raw.shape}")

   # 2. Initial Cleaning & Validation
   # --------------------------------
   # Ensure datetime column is correct format
   dt_col = 'Date'
   df_clean = to_dt(df_raw, dt_col=dt_col, error='raise')

   # Handle missing values (example: forward fill)
   # Production workflows might need more sophisticated imputation
   df_clean = df_clean.ffill()
   print("Performed initial cleaning (datetime check, ffill).")

   # 3. Feature Engineering
   # ----------------------
   # Generate lag, rolling, and time-based features
   target_col = 'Sales'
   df_featured = ts_engineering(
       df=df_clean,
       value_col=target_col, # Generate features based on Sales
       dt_col=dt_col,        # Use Date for time features
       lags=3,               # Create Sales_lag_1, _lag_2, _lag_3
       window=6,             # Create rolling mean/std over 6 months
       diff_order=0,         # No differencing in this example
       apply_fourier=False,  # No Fourier features
       scaler=None            # Apply scaling later after selecting final features
   )
   # Drop rows with NaNs introduced by lags/rolling features
   df_featured.dropna(inplace=True)
   print("Generated time series features (lags, rolling, time-based).")
   # print(df_featured.columns) # View all created features

   # 4. Feature Selection / Reduction (Optional)
   # -------------------------------------------
   # Example: Remove highly correlated features (threshold 0.95)
   # exclude_cols = [dt_col, 'ItemID'] # Keep identifiers
   # df_selected, _ = select_and_reduce_features(
   #     df=df_featured,
   #     target_col=target_col,
   #     exclude_cols=exclude_cols,
   #     method='corr',
   #     corr_threshold=0.95,
   #     verbose=1
   # )
   # For this example, we'll manually select features instead
   df_selected = df_featured # Use all generated features for now

   # 5. Scaling & Encoding
   # ---------------------
   # Define final feature sets AFTER engineering/selection
   static_cols = ['ItemID'] # Should be categorical or already encoded
   dynamic_cols = ['Month', 'Temperature', 'PrevMonthSales', # From ts_engineering
                   'Sales_lag_1', 'Sales_lag_2', 'Sales_lag_3',
                   'rolling_mean_6', 'rolling_std_6',
                   'year', 'day', 'day_of_week', 'is_weekend', 'quarter']
   future_cols = ['PlannedPromotion', 'Month'] # Use future promo & month
   numerical_cols = ['Temperature', 'PrevMonthSales', target_col,
                     'Sales_lag_1', 'Sales_lag_2', 'Sales_lag_3',
                     'rolling_mean_6', 'rolling_std_6'] # Cols to scale

   # Scale numerical features
   scaler = StandardScaler()
   df_scaled = df_selected.copy()
   df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
   # Save the scaler
   scaler_path = os.path.join(output_dir, "feature_scaler.joblib")
   joblib.dump(scaler, scaler_path)
   print(f"Scaled numerical features. Scaler saved to {scaler_path}")

   # Note: Categorical features ('ItemID', 'Month', 'PlannedPromotion')
   # are assumed to be handled by the model's embedding layers or
   # should be one-hot encoded here if the model requires it.

   # 6. Reshape into Sequences for XTFT/TFT
   # --------------------------------------
   time_steps = 12         # 1 year lookback
   forecast_horizons = 6   # Predict 6 months ahead
   spatial_cols = ['ItemID'] # Group sequences by ItemID

   print(f"\nReshaping data into sequences (T={time_steps}, H={forecast_horizons})...")
   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df_scaled, # Use scaled data
       dt_col=dt_col,
       target_col=target_col,
       dynamic_cols=dynamic_cols,
       static_cols=static_cols,
       future_cols=future_cols,
       spatial_cols=spatial_cols,
       time_steps=time_steps,
       forecast_horizons=forecast_horizons,
       verbose=1 # Show shapes
   )

   # 7. Train / Validation / Test Split
   # ----------------------------------
   # Example: 70% Train, 15% Validation, 15% Test (Chronological)
   n_samples = static_data.shape[0]
   n_val = int(n_samples * 0.15)
   n_test = int(n_samples * 0.15)
   n_train = n_samples - n_val - n_test

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
   print(f"  Train: {X_train_dynamic.shape[0]} samples")
   print(f"  Val  : {X_val_dynamic.shape[0]} samples")
   print(f"  Test : {X_test_dynamic.shape[0]} samples")

   # Save processed data (optional)
   processed_data_path = os.path.join(output_dir, "processed_sequences.npz")
   np.savez(
       processed_data_path,
       X_train_static=X_train_static, X_val_static=X_val_static, X_test_static=X_test_static,
       X_train_dynamic=X_train_dynamic, X_val_dynamic=X_val_dynamic, X_test_dynamic=X_test_dynamic,
       X_train_future=X_train_future, X_val_future=X_val_future, X_test_future=X_test_future,
       y_train=y_train, y_val=y_val, y_test=y_test
   )
   print(f"Processed sequence data saved to {processed_data_path}")


.. topic:: Explanations

   1.  **Imports & Config:** Import necessary libraries including pandas,
       numpy, sklearn for scaling, and relevant ``fusionlab`` utilities.
       Define an output directory for saving artifacts like scalers.
   2.  **Load Raw Data:** Load your initial dataset. Here, we generate
       synthetic multi-item sales data for demonstration. Replace this
       with loading your own data file (e.g., using `pd.read_csv` or
       :func:`~fusionlab.core.io.read_data`).
   3.  **Initial Cleaning & Validation:**
       * Use :func:`~fusionlab.utils.ts_utils.to_dt` or
         :func:`~fusionlab.utils.ts_utils.ts_validator` to ensure your
         time column is in the correct `datetime` format.
       * Apply basic missing value handling (here, forward fill `ffill`).
         More complex imputation might be needed for real data.
   4.  **Feature Engineering:**
       * Use :func:`~fusionlab.utils.ts_utils.ts_engineering` to
         automatically generate useful time series features. We create:
         * Lag features for 'Sales' (`lags=3`).
         * Rolling mean/std over a 6-month `window`.
         * Standard time-based features (year, month, day, etc.).
       * Rows with NaNs created by lagging/rolling are dropped.
   5.  **Feature Selection/Reduction (Optional):** After potentially
       creating many features, you might apply techniques to reduce
       dimensionality or remove redundancy. We show where you *could*
       use :func:`~fusionlab.utils.ts_utils.select_and_reduce_features`
       (e.g., for correlation removal), but skip the actual reduction
       in this example for simplicity, using all engineered features.
   6.  **Scaling & Encoding:**
       * Define the final lists of `static_cols`, `dynamic_cols`, and
         `future_cols` based on the available columns *after* feature
         engineering and selection.
       * Identify purely `numerical_cols` that require scaling.
       * Apply `StandardScaler` (or `MinMaxScaler`) to these numerical
         columns. **Crucially, save the fitted scaler** (using `joblib`)
         so you can apply the *same* transformation to new data during
         prediction and inverse transform the model's output.
       * Categorical features (`ItemID`, `Month`, `PlannedPromotion`) are
         listed but not explicitly encoded here, assuming the downstream
         TFT/XTFT model will handle them internally via embedding layers.
         If your model requires one-hot encoding, perform it here and save
         the encoder.
   7.  **Reshape into Sequences:**
       * Use the :func:`~fusionlab.utils.ts_utils.reshape_xtft_data`
         utility. This is ideal for TFT/XTFT as it handles static, dynamic,
         and future features based on the provided column lists.
       * It takes the *processed* and *scaled* DataFrame.
       * It requires `time_steps` (lookback length) and
         `forecast_horizons` (prediction length).
       * It handles grouping by `spatial_cols` (`ItemID`) automatically.
       * It returns the four required NumPy arrays: `static_data`,
         `dynamic_data`, `future_data`, `target_data`.
   8.  **Train/Val/Test Split:**
       * Split the *sequence arrays* chronologically. A simple split
         based on percentages is shown here. For more robust evaluation,
         consider using :func:`~fusionlab.utils.ts_utils.ts_split` with
         `split_type='cv'` before reshaping, although that requires more
         complex data handling across folds.
       * The final arrays (e.g., `X_train_static`, `y_train`) are ready
         to be fed into the model's `fit` method.
       * Optionally save the final processed sequence arrays using
         `np.savez` for easy reloading later.

This workflow provides a robust template for preparing time series data
for `fusionlab`'s advanced forecasting models.