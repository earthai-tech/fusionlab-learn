.. _example_advanced_xtft:

================================
Advanced Forecasting with XTFT
================================

This example demonstrates using the more advanced
:class:`~fusionlab.nn.XTFT` model for a multi-step quantile
forecasting task. XTFT is designed to handle complex scenarios involving
static features (e.g., item ID, location), dynamic historical features
(e.g., past sales, temperature), and known future inputs (e.g.,
planned promotions).

.. _superxtft_model:

XTFT 
------
:API Reference: :class:`~fusionlab.nn.XTFT`

We will walk through the process step-by-step:

1.  Generate synthetic multi-variate time series data for multiple items.
2.  Define static, dynamic, future, and target features.
3.  Scale numerical features.
4.  Use the :func:`~fusionlab.nn.utils.reshape_xtft_data` utility
    to prepare sequences suitable for XTFT.
5.  Split the data into training and validation sets.
6.  Define and compile an XTFT model with quantile outputs.
7.  Train the model.
8.  Make predictions and inverse transform them.
9.  Visualize the quantile predictions.

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, we import the necessary libraries, including TensorFlow, Pandas,
NumPy, scikit-learn for scaling, Matplotlib for plotting, and the
required components from ``fusionlab``. We also suppress common
warnings and logs for cleaner output during the example run.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   import os
   import joblib # For potential scaler saving/loading

   from fusionlab.nn.transformers import XTFT
   from fusionlab.nn.utils import reshape_xtft_data
   from fusionlab.nn.losses import combined_quantile_loss

   # Suppress warnings and TF logs for cleaner output
   import warnings
   warnings.filterwarnings('ignore')
   tf.get_logger().setLevel('ERROR')
   tf.autograph.set_verbosity(0)


Step 2: Generate Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We create a sample dataset simulating monthly sales for multiple items
over several years. This dataset includes static features (`ItemID`),
dynamic features (`Month`, `Temperature`, `PrevMonthSales`), known
future features (`PlannedPromotion`), and the target (`Sales`).

.. code-block:: python
   :linenos:

   n_items = 3
   n_timesteps = 36 # 3 years of monthly data
   date_rng = pd.date_range(start='2020-01-01', periods=n_timesteps, freq='MS')
   df_list = []

   for item_id in range(n_items):
       # Static feature: Item ID (used for grouping)
       # Dynamic features: Month, Temperature, Lagged Sales
       # Future feature: Planned Promotion (binary)
       # Target: Sales
       time = np.arange(n_timesteps)
       # Base sales with trend and seasonality
       sales = (
           100 + item_id * 50 + time * (2 + item_id) +
           20 * np.sin(2 * np.pi * time / 12) +
           np.random.normal(0, 10, n_timesteps)
       )
       temp = 15 + 10 * np.sin(2 * np.pi * (time % 12) / 12 + np.pi) + np.random.normal(0, 2)
       promo = np.random.randint(0, 2, n_timesteps) # Future known promotions

       item_df = pd.DataFrame({
           'Date': date_rng,
           'ItemID': item_id, # Static identifier
           'Month': date_rng.month, # Can be dynamic and future
           'Temperature': temp, # Dynamic
           'PlannedPromotion': promo, # Future
           'Sales': sales # Target
       })
       # Create lagged sales (dynamic history)
       item_df['PrevMonthSales'] = item_df['Sales'].shift(1)
       df_list.append(item_df)

   df = pd.concat(df_list).dropna().reset_index(drop=True)
   print(f"Generated data shape: {df.shape}")
   print("Sample data:")
   print(df.head())


Step 3: Define Features and Scale Numerics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We explicitly define which columns correspond to static, dynamic past,
known future, and target roles. We also identify columns used for
grouping (`spatial_cols`). Importantly, numerical features that will be
fed into the neural network are scaled (here using `StandardScaler`)
to improve training stability. Scalers are typically saved so predictions
can be inverse-transformed later.

.. code-block:: python
   :linenos:

   target_col = 'Sales'
   dt_col = 'Date'
   static_cols = ['ItemID'] # Could add more attributes here
   dynamic_cols = ['Month', 'Temperature', 'PrevMonthSales']
   future_cols = ['PlannedPromotion', 'Month'] # 'Month' is known ahead
   spatial_cols = ['ItemID'] # Group data by ItemID

   # Scale numerical features (excluding IDs/Month/Event)
   scalers = {}
   num_cols_to_scale = ['Temperature', 'PrevMonthSales', 'Sales']
   for col in num_cols_to_scale:
       scaler = StandardScaler()
       df[col] = scaler.fit_transform(df[[col]])
       scalers[col] = scaler # Store scaler for inverse transform
       print(f"Scaled column: {col}")
   # Example: Save scalers
   # joblib.dump(scalers, 'scalers.joblib')


Step 4: Prepare Sequences using `reshape_xtft_data`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :func:`~fusionlab.nn.utils.reshape_xtft_data` utility transforms
the processed DataFrame into the specific input arrays required by XTFT.
It creates rolling windows based on `time_steps` (lookback) and
`forecast_horizons`, groups by `spatial_cols`, and separates features
into static, dynamic, future, and target arrays.

.. code-block:: python
   :linenos:

   time_steps = 12 # Use 1 year of history
   forecast_horizons = 6 # Predict next 6 months

   static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
       df=df,
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


Step 5: Train/Validation Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The generated sequence arrays are split into training and validation sets.
A simple chronological split is used here, taking the first part for
training and the latter part for validation. Inputs for the model are
packaged into lists in the standard order `[static, dynamic, future]`.

.. code-block:: python
   :linenos:

   val_split_fraction = 0.2
   n_samples = static_data.shape[0]
   split_idx = int(n_samples * (1 - val_split_fraction))

   X_train_static, X_val_static = static_data[:split_idx], static_data[split_idx:]
   X_train_dynamic, X_val_dynamic = dynamic_data[:split_idx], dynamic_data[split_idx:]
   X_train_future, X_val_future = future_data[:split_idx], future_data[split_idx:]
   y_train, y_val = target_data[:split_idx], target_data[split_idx:]

   # Package inputs as lists
   train_inputs = [X_train_static, X_train_dynamic, X_train_future]
   val_inputs = [X_val_static, X_val_dynamic, X_val_future]

   print(f"\nData split into Train/Validation:")
   print(f"  Train samples: {X_train_static.shape[0]}")
   print(f"  Validation samples: {X_val_static.shape[0]}")


Step 6: Define XTFT Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We instantiate the :class:`~fusionlab.nn.XTFT` model. Input dimensions
are derived from the prepared data arrays. We configure it for quantile
forecasting by providing the `quantiles` list and specify other relevant
hyperparameters (these would typically be tuned).

.. code-block:: python
   :linenos:

   quantiles_to_predict = [0.1, 0.5, 0.9]
   output_dim = 1 # Predicting univariate 'Sales'

   model = XTFT(
       static_input_dim=static_data.shape[-1],
       dynamic_input_dim=dynamic_data.shape[-1],
       future_input_dim=future_data.shape[-1],
       forecast_horizon=forecast_horizons,
       quantiles=quantiles_to_predict,
       output_dim=output_dim,
       # Example XTFT Hyperparameters
       embed_dim=16, lstm_units=32, attention_units=16,
       hidden_units=32, num_heads=4, dropout_rate=0.1,
       max_window_size=time_steps, memory_size=50,
       # scales=[1, 6] # Optional multi-scale example
   )
   print("XTFT model instantiated.")
   # model.summary() # Call after model is built


Step 7: Compile the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The model is compiled using an optimizer (like Adam) and an appropriate
loss function. For quantile forecasting, we use the
:func:`~fusionlab.nn.losses.combined_quantile_loss` function.

.. code-block:: python
   :linenos:

   loss_fn = combined_quantile_loss(quantiles=quantiles_to_predict)
   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                 loss=loss_fn)
   print("XTFT model compiled successfully.")


Step 8: Train the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~
We train the model using the `.fit()` method, providing the prepared
training inputs (`train_inputs`) and targets (`y_train`), along with
validation data (`val_inputs`, `y_val`). Training runs for a small number
of epochs for demonstration.

.. code-block:: python
   :linenos:

   print("Starting XTFT model training (few epochs for demo)...")
   history = model.fit(
       train_inputs,
       y_train,
       validation_data=(val_inputs, y_val),
       epochs=5, # Increase for real training
       batch_size=16, # Adjust based on memory
       verbose=1 # Show epoch progress
   )
   print("Training finished.")


Step 9: Make Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use the trained model's `.predict()` method on the validation input
data (`val_inputs`) to generate scaled quantile forecasts.

.. code-block:: python
   :linenos:

   print("\nMaking predictions on validation set...")
   predictions_scaled = model.predict(val_inputs, verbose=0)
   print(f"Scaled prediction output shape: {predictions_scaled.shape}")
   # Expected: (NumValSamples, Horizon, NumQuantiles) -> (N_val, 6, 3)


Step 10: Inverse Transform Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The model predicts scaled values. We use the scaler saved during
preprocessing (Step 3) to transform the predictions and the actual
validation targets back to their original 'Sales' units for easier
interpretation and visualization.

.. code-block:: python
   :linenos:

   # Reshape for scaler: (Samples*Horizon, Quantiles/OutputDim)
   num_val_samples = X_val_static.shape[0]
   num_quantiles = len(quantiles_to_predict)

   pred_reshaped = predictions_scaled.reshape(-1, num_quantiles)
   predictions_inv = scalers['Sales'].inverse_transform(pred_reshaped)
   # Reshape back: (Samples, Horizon, Quantiles)
   predictions_final = predictions_inv.reshape(
       num_val_samples, forecast_horizons, num_quantiles
   )

   # Also inverse transform actuals for plotting
   y_val_reshaped = y_val.reshape(-1, output_dim) # Reshape y_val (B, H, O)
   y_val_inv = scalers['Sales'].inverse_transform(y_val_reshaped)
   y_val_final = y_val_inv.reshape(
       num_val_samples, forecast_horizons, output_dim
       ) # Reshape back (B, H, O)

   print("Predictions inverse transformed.")


Step 11: Visualize Forecast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Finally, we visualize the forecast for a single item from the
validation set. We plot the actual sales against the predicted median
(0.5 quantile) and shade the area between the lower (0.1 quantile) and
upper (0.9 quantile) predictions to represent the uncertainty interval.

.. code-block:: python
   :linenos:

   item_to_plot = 0 # Plot results for ItemID 0
   # Find indices in validation set corresponding to this item
   item_mask_val = (X_val_static[:, 0] == item_to_plot)

   if np.sum(item_mask_val) > 0:
       # Find the first sequence index for this item in validation
       first_val_seq_idx = np.where(item_mask_val)[0][0]

       actual_vals_item = y_val_final[first_val_seq_idx, :, 0] # O=1
       pred_quantiles_item = predictions_final[first_val_seq_idx, :, :]

       # Create approximate time axis for the forecast period
       # Find last date in training data for this item ID
       last_train_date_item = df[(df['ItemID']==item_to_plot)].iloc[split_idx + time_steps - 1]['Date']
       pred_time_axis = pd.date_range(
           last_train_date_item + pd.DateOffset(months=1),
           periods=forecast_horizons, freq='MS'
       )

       plt.figure(figsize=(12, 6))
       plt.plot(pred_time_axis, actual_vals_item, label='Actual Sales', marker='o', linestyle='--')
       # Assuming quantiles are [0.1, 0.5, 0.9] -> index 1 is median
       plt.plot(pred_time_axis, pred_quantiles_item[:, 1], label='Median Forecast (q=0.5)', marker='x')
       plt.fill_between(
           pred_time_axis,
           pred_quantiles_item[:, 0], # Lower quantile (q=0.1)
           pred_quantiles_item[:, 2], # Upper quantile (q=0.9)
           color='gray', alpha=0.3, label='Prediction Interval (q=0.1 to q=0.9)'
       )
       plt.title(f'XTFT Quantile Forecast (ItemID {item_to_plot})')
       plt.xlabel('Date')
       plt.ylabel('Sales (Inverse Scaled)')
       plt.legend()
       plt.grid(True)
       plt.show()
   else:
       print(f"No validation data found for ItemID {item_to_plot} to plot.")
       

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   
.. _superxtft_model:

SuperXTFT (Experimental)
--------------------------
:API Reference: :class:`~fusionlab.nn.SuperXTFT`

.. warning::
   ``SuperXTFT`` is currently **experimental** and may undergo
   significant changes or removal. It is **not recommended for
   production use**. Please use :class:`~fusionlab.nn.XTFT` for
   stable deployments.

The ``SuperXTFT`` class inherits from and extends the
:class:`~fusionlab.nn.XTFT` model. It incorporates additional
components aimed at potentially improving input feature selection and
refining intermediate representations within the attention pathways.

**Key Enhancements & Differences (from XTFT):**

* **Inherits XTFT Features:** Includes all advanced capabilities of the
  base ``XTFT`` (Multi-Scale LSTM, diverse attention mechanisms,
  anomaly detection integration, etc.).
* **Input Variable Selection Networks (VSNs):** ``SuperXTFT`` introduces
  dedicated :class:`~fusionlab.nn.components.VariableSelectionNetwork`
  layers applied directly to the raw static, dynamic (past), and
  future inputs. This aims to perform feature selection *before* the
  inputs enter the main embedding and temporal processing stages,
  allowing the model to focus on the most relevant raw features early on.
* **Post-Attention GRN Processing:** Adds dedicated
  :class:`~fusionlab.nn.components.GatedResidualNetwork` layers that
  process the outputs of the main attention blocks (Hierarchical, Cross,
  Memory-Augmented). This allows for further non-linear transformation
  and refinement of the context vectors generated by these attention
  mechanisms before they are fused.
* **Post-Decoder GRN:** Includes an additional GRN applied to the output
  of the :class:`~fusionlab.nn.components.MultiDecoder` stage, potentially
  refining the horizon-specific features before the final quantile
  modeling step.

**When to Use:**

* **Currently:** Intended for research, experimentation, and internal
  development within ``fusionlab`` to evaluate the benefits of the
  added components.
* **Not Recommended For:** General use or production deployments until
  its stability and performance advantages over XTFT are confirmed and
  its status moves beyond experimental.

Mathematical Formulation Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SuperXTFT`` modifies the ``XTFT`` data flow primarily by:

1.  **Applying Input VSNs:** Instead of initial normalization/embedding,
    inputs (:math:`s, x_t, z_t`) first pass through respective VSNs:

    .. math::
       s' = VSN_{static}(s) \\
       x'_t = VSN_{dynamic}(x_t) \\
       z'_t = VSN_{future}(z_t)

    These selected features (:math:`s', x'_t, z'_t`) then feed into the
    subsequent XTFT stages (normalization, embedding, etc.).

2.  **Applying Post-Component GRNs:** Specific intermediate outputs within
    the XTFT flow (:math:`Attn_{...}` or :math:`Dec_{out}`) are immediately
    processed by an additional dedicated GRN before proceeding:

    .. math::
       Output'_{component} = GRN_{component}(Output_{component})

    This interleaves extra GRN processing within the attention and
    decoding stages.

**Code Example (Instantiation Only):**

*(Note: Due to the experimental status, only instantiation is shown.
Use with caution and refer to source code for precise implementation.)*

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.nn.transformers import SuperXTFT

   # Example Configuration (must provide all required dims)
   static_dim, dynamic_dim, future_dim = 5, 7, 3
   horizon = 12
   output_dim = 1

   # Instantiate SuperXTFT
   # Accepts the same parameters as XTFT
   try:
       super_xtft_model = SuperXTFT(
           static_input_dim=static_dim,
           dynamic_input_dim=dynamic_dim,
           future_input_dim=future_dim,
           forecast_horizon=horizon,
           output_dim=output_dim,
           hidden_units=32, # Example other params
           num_heads=4,
           # anomaly_detection_strategy=None # Example
       )
       print("SuperXTFT model instantiated successfully.")
       # Build the model to see the structure (e.g., via dummy call)
       # dummy_s = tf.zeros((1, static_dim))
       # dummy_d = tf.zeros((1, 10, dynamic_dim)) # T=10 example
       # dummy_f = tf.zeros((1, 10 + horizon, future_dim)) # T+H example
       # super_xtft_model([dummy_s, dummy_d, dummy_f])
       # super_xtft_model.summary()
   except Exception as e:
       print(f"Error instantiating SuperXTFT: {e}")


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

