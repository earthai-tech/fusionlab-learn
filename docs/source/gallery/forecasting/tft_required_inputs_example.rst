.. _example_tft_required_inputs:

======================================================
Point Forecasting with Stricter TFT (Required Inputs)
======================================================

This example demonstrates how to use the stricter
:class:`~fusionlab.nn.transformers.TFT` class implementation.
Unlike the more flexible
:class:`~fusionlab.nn.transformers.TemporalFusionTransformer`, this
version strictly requires **static**, **dynamic (past)**, and
**known future** features as inputs during initialization and for
model calls.

We will perform a single-step point forecast, showcasing the specific
data preparation and model interaction for this TFT variant.

The workflow includes:
1.  Generating synthetic data with distinct static, dynamic, and
    future features for multiple items.
2.  Defining feature roles, encoding categorical static features, and
    scaling numerical data.
3.  Using the :func:`~fusionlab.nn.utils.reshape_xtft_data` utility
    to prepare the three separate input arrays (static, dynamic, future)
    and targets.
4.  Splitting the sequence data into training and validation sets.
5.  Defining, compiling, and training the stricter `TFT` model for point
    forecasting.
6.  Making predictions using the mandatory three-part input structure.
7.  Visualizing the results.

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and its dependencies installed:

.. code-block:: bash

   pip install fusionlab-learn matplotlib scikit-learn

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Import standard libraries and the necessary components from
``fusionlab``, including the stricter `TFT` model and
`reshape_xtft_data`.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.preprocessing import StandardScaler, LabelEncoder # Added LabelEncoder
   from sklearn.model_selection import train_test_split
   import warnings
   import os

   # FusionLab imports
   from fusionlab.nn.transformers import TFT # The stricter TFT class
   from fusionlab.nn.utils import reshape_xtft_data
   # For Keras to recognize custom loss if model was saved with it
   from fusionlab.nn.losses import combined_quantile_loss

   # Suppress warnings and TF logs for cleaner output
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   if hasattr(tf, 'autograph'):
       tf.autograph.set_verbosity(0)

   print("Libraries imported and TensorFlow logs configured.")

Step 2: Generate Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We create a synthetic dataset for multiple items, including static
features (like `ItemID`, `Category`), dynamic past features
(`DayOfWeek`, `ValueLag1`), and known future features (`FutureEvent`,
`DayOfWeek`). `ItemID` is generated as a string.

.. code-block:: python
   :linenos:

   n_items = 2
   n_timesteps_per_item = 50
   rng_seed = 42
   np.random.seed(rng_seed)

   date_rng = pd.date_range(
       start='2021-01-01', periods=n_timesteps_per_item, freq='D'
       )
   df_list = []

   for item_id_num in range(n_items): # Use numerical id for generation
       time_idx = np.arange(n_timesteps_per_item)
       value = (50 + item_id_num * 10 + time_idx * 0.5 +
                np.sin(time_idx / 7) * 5 +
                np.random.normal(0, 2, n_timesteps_per_item))
       static_category_val = item_id_num + 1
       future_event_val = (date_rng.dayofweek >= 5).astype(int)

       item_df = pd.DataFrame({
           'Date': date_rng,
           'ItemID_str': f'item_{item_id_num}', # String ID for raw data
           'Category': static_category_val,    # Numerical static
           'DayOfWeek': date_rng.dayofweek,
           'FutureEvent': future_event_val,
           'Value': value
       })
       item_df['ValueLag1'] = item_df['Value'].shift(1)
       df_list.append(item_df)

   df_raw = pd.concat(df_list).dropna().reset_index(drop=True)
   print(f"Generated raw data shape: {df_raw.shape}")
   print("Sample of generated data:")
   print(df_raw.head())

Step 3: Define Features, Encode Categorical Static, & Scale Numerics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assign columns to their roles. **Crucially, encode string-based static
features like `ItemID_str` into numerical representations before scaling
and reshaping.**

.. code-block:: python
   :linenos:

   target_col = 'Value'
   dt_col = 'Date'
   
   # Initial column definitions
   # ItemID_str is categorical, Category is already numerical static
   static_cols_def = ['ItemID_str', 'Category']
   dynamic_cols_def = ['DayOfWeek', 'ValueLag1']
   future_cols_def = ['FutureEvent', 'DayOfWeek']
   spatial_cols_def = ['ItemID_str'] # Group by original string ID

   df_processed = df_raw.copy()

   # --- Encode ItemID_str (Categorical Static Feature) ---
   le_item_id = LabelEncoder()
   # Create a new numerical column for ItemID
   df_processed['ItemID_encoded'] = le_item_id.fit_transform(
       df_processed['ItemID_str']
   )
   print(f"\nEncoded 'ItemID_str' into 'ItemID_encoded'. "
         f"Classes: {le_item_id.classes_}")

   # --- Update static_cols to use the encoded version ---
   # 'Category' is already numeric. We'll use 'ItemID_encoded'.
   static_cols_for_model = ['ItemID_encoded', 'Category']
   # Update spatial_cols if grouping should now be by the encoded ID
   # For reshape_xtft_data, spatial_cols are used for grouping and
   # are often also part of static_cols if they are static identifiers.
   # If ItemID_encoded is the primary key for grouping sequences:
   spatial_cols_for_model = ['ItemID_encoded']


   # --- Scale Numerical Features ---
   # Target 'Value' and 'ValueLag1' are scaled.
   # 'Category', 'DayOfWeek', 'FutureEvent', 'ItemID_encoded' are not scaled here
   # as they are categorical or already identifiers.
   scaler = StandardScaler()
   num_cols_to_scale = ['Value', 'ValueLag1']
   # Ensure these columns exist before trying to scale
   num_cols_to_scale = [c for c in num_cols_to_scale if c in df_processed.columns]

   if num_cols_to_scale:
       df_processed[num_cols_to_scale] = scaler.fit_transform(
           df_processed[num_cols_to_scale]
       )
       print("\nNumerical features scaled.")
   else:
       print("\nNo numerical features specified or found for scaling.")


Step 4: Prepare Sequences with `reshape_xtft_data`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use :func:`~fusionlab.nn.utils.reshape_xtft_data` with the
**processed DataFrame** (which now has `ItemID_encoded`) and the
updated column lists.

.. code-block:: python
   :linenos:

   time_steps = 7
   forecast_horizon = 1

   # Use the updated column lists for model input features
   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df_processed, # Use the DataFrame with ItemID_encoded
       dt_col=dt_col,
       target_col=target_col,
       dynamic_cols=dynamic_cols_def, # Original dynamic cols
       static_cols=static_cols_for_model, # Use encoded static cols
       future_cols=future_cols_def,   # Original future cols
       spatial_cols=spatial_cols_for_model, # Group by encoded ItemID
       time_steps=time_steps,
       forecast_horizons=forecast_horizon,
       verbose=0
   )
   targets = target_data.astype(np.float32)

   print(f"\nReshaped Data Shapes:")
   print(f"  Static : {static_data.shape if static_data is not None else 'None'}")
   print(f"  Dynamic: {dynamic_data.shape if dynamic_data is not None else 'None'}")
   print(f"  Future : {future_data.shape if future_data is not None else 'None'}")
   print(f"  Target : {targets.shape if targets is not None else 'None'}")

Step 5: Train/Validation Split of Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Split the generated sequence arrays. The input for the model will be
`[X_static, X_dynamic, X_future]`.

.. code-block:: python
   :linenos:

   val_split_fraction = 0.2
   if static_data is None or dynamic_data is None or \
      future_data is None or targets is None:
       raise ValueError("Data reshaping did not produce all required arrays.")

   n_samples = static_data.shape[0]
   split_idx = int(n_samples * (1 - val_split_fraction))

   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   X_train_future, X_val_future = future_data[:split_idx], future_data[split_idx:]
   y_train, y_val = targets[:split_idx], targets[split_idx:]

   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future]

   print("\nSequence data split into Train/Validation sets.")
   print(f"  Train samples: {len(y_train)}")
   print(f"  Validation samples: {len(y_val)}")

Step 6: Define and Compile Stricter `TFT` Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instantiate the :class:`~fusionlab.nn.transformers.TFT` class.
All three input dimensions must be provided.

.. code-block:: python
   :linenos:

   model = TFT(
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1],
       forecast_horizon=forecast_horizon,
       output_dim=1,
       hidden_units=16, num_heads=2,
       num_lstm_layers=1, lstm_units=16,
       quantiles=None # Point forecast
   )
   print("\nStricter TFT model instantiated.")
   model.compile(optimizer='adam', loss='mse')
   print("Model compiled.")

Step 7: Train the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Train using the 3-element `train_inputs` list.

.. code-block:: python
   :linenos:

   print("\nStarting model training...")
   history = model.fit(
       train_inputs, y_train,
       validation_data=(val_inputs, y_val),
       epochs=5, batch_size=16, verbose=1
   )
   print("Training finished.")
   if history and history.history.get('val_loss'):
       print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

Step 8: Make Predictions and Visualize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Predict on the validation set and visualize. Inverse transform for
interpretable results.

.. code-block:: python
   :linenos:

   print("\nMaking predictions on the validation set...")
   val_predictions_scaled = model.predict(val_inputs, verbose=0)

   # Inverse transform (simplified for target only)
   # Create a dummy array matching the shape scaler was fit on
   # (assuming scaler was fit on multiple columns from num_cols_to_scale)
   dummy_for_inv_transform = np.zeros((len(val_predictions_scaled.flatten()), len(num_cols_to_scale)))
   
   # Find the index of the target column in the original list of scaled columns
   target_idx_in_scaler = num_cols_to_scale.index(target_col)

   # Populate the target column in the dummy array for inverse transform
   dummy_for_inv_transform[:, target_idx_in_scaler] = val_predictions_scaled.flatten()
   val_predictions_inv = scaler.inverse_transform(dummy_for_inv_transform)[:, target_idx_in_scaler]
   val_predictions_final = val_predictions_inv.reshape(val_predictions_scaled.shape)

   # Inverse transform actuals
   dummy_for_inv_transform_actual = np.zeros((len(y_val.flatten()), len(num_cols_to_scale)))
   dummy_for_inv_transform_actual[:, target_idx_in_scaler] = y_val.flatten()
   val_actuals_inv = scaler.inverse_transform(dummy_for_inv_transform_actual)[:, target_idx_in_scaler]
   val_actuals_final = val_actuals_inv.reshape(y_val.shape)

   print("Predictions and actuals inverse transformed.")

   # --- Visualization (for the first item ID in validation set) ---
   # Get the encoded ItemID from the validation static data
   first_val_item_id_encoded = X_val_static[0, static_cols_for_model.index('ItemID_encoded')]
   # Convert back to original string ID for display if desired
   # original_item_id_str = le_item_id.inverse_transform([int(first_val_item_id_encoded)])[0]

   item_mask_val = (X_val_static[:, static_cols_for_model.index('ItemID_encoded')] == first_val_item_id_encoded)
   item_preds = val_predictions_final[item_mask_val, 0, 0]
   item_actuals = val_actuals_final[item_mask_val, 0, 0]

   plt.figure(figsize=(12, 6))
   plt.plot(item_actuals,
            label=f'Actual (Item Encoded: {int(first_val_item_id_encoded)})',
            marker='o', linestyle='--')
   plt.plot(item_preds,
            label=f'Predicted (Item Encoded: {int(first_val_item_id_encoded)})',
            marker='x')
   plt.title(f'Stricter TFT Point Forecast (Validation Item - Inverse Transformed)')
   plt.xlabel('Sequence Index in Validation Set for this Item')
   plt.ylabel('Value (Inverse Transformed)')
   plt.legend(); plt.grid(True); plt.tight_layout()
   # plt.savefig("docs/source/images/forecasting_tft_required_inputs.png")
   plt.show()
   print("Plot generated.")

**Example Output Plot:**

.. figure:: ../../images/forecasting_tft_required_inputs.png
   :alt: Stricter TFT Point Forecast
   :align: center
   :width: 80%

   Visualization of the point forecast against actual validation data
   using the stricter `TFT` model.

