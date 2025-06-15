.. _user_guide_forecasting_workflow_utils:

=================================
Forecasting Workflow Utilities
=================================

``fusionlab-learn`` provides a set of powerful utility functions within
the :mod:`fusionlab.nn.utils` module to streamline common tasks in a
time series forecasting pipeline. This guide demonstrates a typical
workflow using three key utilities:

1.  :func:`~fusionlab.nn.utils.prepare_model_inputs`: Standardizes
    the creation of the input list `[static, dynamic, future]` for
    various model types, handling optional inputs gracefully.
2.  :func:`~fusionlab.nn.utils.format_predictions_to_dataframe`:
    Transforms raw model predictions (point or quantile) into a
    structured, long-format pandas DataFrame, suitable for analysis,
    storage, and further visualization.
3.  :func:`~fusionlab.plot.forecast.plot_forecasts`: Visualizes the
    formatted forecast DataFrame, allowing comparison of predictions
    against actuals in both temporal and spatial dimensions.
    *(Note: This function resides in `fusionlab.plot.evaluation` but
    is often used in conjunction with `nn.utils`)*.

By using these utilities together, you can significantly simplify your
forecasting code, making it more robust and easier to manage.


Prerequisites
-------------

Ensure you have ``fusionlab-learn`` and its common dependencies
installed. For visualizations, `matplotlib` is also needed.

.. code-block:: bash

   pip install fusionlab-learn matplotlib scikit-learn

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Common Setup for Examples
----------------------------
We'll start with common imports and generate some basic dummy data
that simulates static, dynamic, and future features, along with
target values.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   import os
   import warnings

   # FusionLab imports
   from fusionlab.nn.utils import (
       prepare_model_inputs,
       format_predictions_to_dataframe
   )
   try:
       from fusionlab.plot.forecast import plot_forecasts
   except ImportError:
       # Fallback if plot_forecasts is in nn.utils for some versions
       from fusionlab.nn.utils import plot_forecasts
       warnings.warn("Imported plot_forecasts from fusionlab.nn.utils. "
                     "Consider moving to fusionlab.plot.evaluation.")


   # Suppress warnings and TF logs for cleaner output
   warnings.filterwarnings('ignore')
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   tf.get_logger().setLevel('ERROR')
   if hasattr(tf, 'autograph'):
       tf.autograph.set_verbosity(0)

   # Base dimensions for dummy data
   B, T_PAST, H_OUT = 5, 12, 4 # Batch, Past Timesteps, Horizon
   D_S, D_D, D_F, D_O = 2, 3, 2, 1 # Static, Dynamic, Future, Output Dims
   T_FUTURE_TOTAL = T_PAST + H_OUT
   SEED = 42
   np.random.seed(SEED)
   tf.random.set_seed(SEED)

   # Generate dummy data components
   raw_static_data = tf.random.normal((B, D_S), dtype=tf.float32, seed=SEED)
   raw_dynamic_data = tf.random.normal((B, T_PAST, D_D), dtype=tf.float32, seed=SEED+1)
   raw_future_data = tf.random.normal((B, T_FUTURE_TOTAL, D_F), dtype=tf.float32, seed=SEED+2)
   raw_y_true_sequences = tf.random.normal((B, H_OUT, D_O), dtype=tf.float32, seed=SEED+3)

   # Simulate some spatial identifiers for later use
   spatial_ids_df = pd.DataFrame({
       'location_id': [f'L{i}' for i in range(B)],
       'region': [f'R{i%2}' for i in range(B)]
   })

   print("Common setup complete. Dummy data generated.")
   print(f"  Static shape : {raw_static_data.shape}")
   print(f"  Dynamic shape: {raw_dynamic_data.shape}")
   print(f"  Future shape : {raw_future_data.shape}")
   print(f"  Target shape : {raw_y_true_sequences.shape}")

**Expected Output (Common Setup):**

.. code-block:: text

   Common setup complete. Dummy data generated.
     Static shape : (5, 2)
     Dynamic shape: (5, 12, 3)
     Future shape : (5, 16, 2)
     Target shape : (5, 4, 1)

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Step 1: Preparing Model Inputs with `prepare_model_inputs`
----------------------------------------------------------
:API Reference: :func:`~fusionlab.nn.utils.prepare_model_inputs`

The first step in a forecasting pipeline after loading/generating raw
features is to package them correctly for your specific model.
`prepare_model_inputs` helps create the standard 3-element list
`[static_input, dynamic_input, future_input]` that many ``fusionlab-learn``
models expect for their `call` method.

**Scenario 1.1: Stricter Model (e.g., XTFT, TFTStricter)**
   These models typically require all three input types (static,
   dynamic, future) to be actual tensors. If an input type is not
   semantically present for your data, `prepare_model_inputs` with
   `model_type='strict'` will create a dummy tensor with zero features
   for that slot.

.. code-block:: python
   :linenos:

   print("\n--- Preparing inputs for a 'strict' model ---")
   # Example 1: All inputs provided
   inputs_strict_all = prepare_model_inputs(
       dynamic_input=raw_dynamic_data,
       static_input=raw_static_data,
       future_input=raw_future_data,
       model_type='strict',
       forecast_horizon=H_OUT, # Used for dummy future if future_input is None
       verbose=1
   )
   print(f"Strict (all provided): S={inputs_strict_all[0].shape}, "
         f"D={inputs_strict_all[1].shape}, F={inputs_strict_all[2].shape}")

   # Example 2: Static input is conceptually absent
   inputs_strict_no_static = prepare_model_inputs(
       dynamic_input=raw_dynamic_data,
       static_input=None, # Static features are not available
       future_input=raw_future_data,
       model_type='strict',
       forecast_horizon=H_OUT,
       verbose=1
   )
   print(f"Strict (no static): S={inputs_strict_no_static[0].shape}, "
         f"D={inputs_strict_no_static[1].shape}, "
         f"F={inputs_strict_no_static[2].shape}")

**Expected Output 1.1:**

.. code-block:: text

   --- Preparing inputs for a 'strict' model ---
     prepare_model_inputs (strict): Passing inputs as is.
   Strict (all provided): S=(5, 2), D=(5, 12, 3), F=(5, 16, 2)
     prepare_model_inputs (strict): Created dummy static input with shape (5, 0)
     prepare_model_inputs (strict): Passing inputs as is.
   Strict (no static): S=(5, 0), D=(5, 12, 3), F=(5, 16, 2)

**Scenario 1.2: Flexible Model (e.g., TemporalFusionTransformer)**
   Flexible models can handle `None` for optional inputs (static, future).
   `prepare_model_inputs` with `model_type='flexible'` will pass these
   `None` values through.

.. code-block:: python
   :linenos:

   print("\n--- Preparing inputs for a 'flexible' model ---")
   # Example 1: Dynamic input only
   inputs_flex_dyn_only = prepare_model_inputs(
       dynamic_input=raw_dynamic_data,
       static_input=None,
       future_input=None,
       model_type='flexible',
       verbose=1
   )
   s_shape = inputs_flex_dyn_only[0].shape if inputs_flex_dyn_only[0] is not None else "None"
   d_shape = inputs_flex_dyn_only[1].shape
   f_shape = inputs_flex_dyn_only[2].shape if inputs_flex_dyn_only[2] is not None else "None"
   print(f"Flexible (dynamic only): S={s_shape}, D={d_shape}, F={f_shape}")

**Expected Output 1.2:**

.. code-block:: text

   --- Preparing inputs for a 'flexible' model ---
     prepare_model_inputs (flexible): Passing inputs as is (Static: <class 'NoneType'>, Dynamic: <class 'tensorflow.python.framework.ops.EagerTensor'>, Future: <class 'NoneType'>)
   Flexible (dynamic only): S=None, D=(5, 12, 3), F=None


Step 2: Simulate Model Prediction
------------------------------------
For this exercise, we won't train a full model. Instead, we'll simulate
the kind of output a forecasting model might produce.
Let's assume we are doing a quantile forecast.

.. code-block:: python
   :linenos:

   # Simulate predictions (e.g., from an XTFT model)
   # Shape: (Batch, Horizon, NumQuantiles * OutputDim)
   # For this example, OutputDim=1, NumQuantiles=3
   simulated_predictions_quant = tf.random.normal(
       (B, H_OUT, len(Q_LIST_VIZ) * D_O), dtype=tf.float32, seed=SEED+4
   )
   print(f"\nSimulated quantile predictions shape: {simulated_predictions_quant.shape}")

**Expected Output 2.1:**

.. code-block:: text

   Simulated quantile predictions shape: (5, 4, 3)


Step 3: Format Predictions with `format_predictions_to_dataframe`
-----------------------------------------------------------------
:API Reference: :func:`~fusionlab.nn.utils.format_predictions_to_dataframe`

This utility takes the raw prediction tensor (and optionally actuals,
spatial data, etc.) and converts it into a well-structured, long-format
pandas DataFrame. This DataFrame is then easy to analyze, save, or
pass to plotting functions.

**Scenario 3.1: Formatting Quantile Forecasts with Actuals and Spatial Data**

.. code-block:: python
   :linenos:

   print("\n--- Formatting quantile predictions to DataFrame ---")
   # Use the spatial_ids_df created in common setup
   # Ensure it has the same number of samples (B) as predictions
   spatial_data_for_format = spatial_ids_df # Shape (B, NumSpatialFeatures)

   forecast_df_viz = format_predictions_to_dataframe(
       predictions=simulated_predictions_quant,
       y_true_sequences=raw_y_true_sequences,
       target_name="sales", # Base name for columns
       quantiles=Q_LIST_VIZ,
       forecast_horizon=H_OUT, # Helps structure the DataFrame
       output_dim=D_O,         # Number of target variables
       spatial_data_array=spatial_data_for_format, # DataFrame with B rows
       spatial_cols_names=['location_id', 'region_code'], # Names for these cols
       verbose=1
   )
   print("\nFormatted DataFrame head (Quantile Forecast):")
   print(forecast_df_viz.head(H_OUT * 2)) # Show for first two samples
   print(f"\nFormatted DataFrame shape: {forecast_df_viz.shape}")
   print(f"Formatted DataFrame columns: {forecast_df_viz.columns.tolist()}")

**Expected Output 3.1:**
   *(DataFrame structure with sample_idx, forecast_step, spatial cols,
   sales_q10, sales_q50, sales_q90, sales_actual)*

.. code-block:: text

   --- Formatting quantile predictions to DataFrame ---
   [INFO] Starting prediction formatting to DataFrame.
       [INFO]   Raw predictions shape: (5, 4, 3)
       [INFO]   Inferred/Validated: Samples=5, Horizon=4, OutputDim=1, NumQuantiles=3
       [INFO]   Added prediction columns: ['sales_q10', 'sales_q50', 'sales_q90']
       [INFO]   Added actual value columns: ['sales_actual']
   [INFO] Prediction formatting to DataFrame complete.

   Formatted DataFrame head (Quantile Forecast):
      sample_idx  forecast_step  sales_q10  sales_q50  sales_q90  sales_actual
   0           0              1  -0.492519   0.314352  -0.939723     -0.019795
   1           0              2  -0.489788   1.087007   0.165282      0.407925
   2           0              3   0.692570  -0.101750  -0.165129     -0.115735
   3           0              4   0.622007   0.223282   0.049389     -0.308791
   4           1              1  -1.499012  -0.228126  -0.840142      0.445111
   5           1              2  -0.401215   1.823693   1.008885     -0.407488
   6           1              3   1.087821   0.155696  -0.351913      2.175023
   7           1              4  -0.040999  -1.583362   1.056865      0.755576

   Formatted DataFrame shape: (20, 6)
   Formatted DataFrame columns: ['sample_idx', 'forecast_step', 'sales_q10', 'sales_q50', 'sales_q90', 'sales_actual']


Step 4: Visualizing Formatted Predictions with `plot_forecasts`
---------------------------------------------------------------
:API Reference: :func:`~fusionlab.plot.evaluation.plot_forecasts`

Once your predictions are in a structured DataFrame (thanks to
`format_predictions_to_dataframe`), `plot_forecasts` can easily
visualize them.

**Scenario 4.1: Temporal Quantile Forecast for Selected Samples**

.. code-block:: python
   :linenos:

   print("\n--- Visualizing Temporal Quantile Forecast ---")
   plot_forecasts(
       forecast_df=forecast_df_viz,
       target_name="sales",
       quantiles=Q_LIST_VIZ,
       output_dim=D_O,
       kind="temporal",
       sample_ids=[0, 1], # Plot for first two samples
       max_cols=1,         # Each sample plot in a new row
       figsize_per_subplot=(10, 4),
       verbose=1
   )
   # To save:
   # fig_path = os.path.join(evaluation_plot_dir, "workflow_temporal_quantile.png")
   # plt.savefig(fig_path)

**Expected Plot 4.1:**
   *(Two subplots, each showing actual vs. median and prediction interval
   for sample_idx 0 and 1 respectively)*

.. figure:: ../../images/workflow_utils_temporal_quantile.png
   :alt: Temporal Quantile Forecast from Workflow Utilities
   :align: center
   :width: 70%

   Temporal plot showing actuals, median forecast, and prediction
   intervals for selected samples.

**Scenario 4.2: Spatial Point Forecast for a Specific Horizon Step**
   First, let's create a point forecast DataFrame for this.

.. code-block:: python
   :linenos:

   # Simulate point predictions (e.g., just the median from quantiles)
   simulated_predictions_point = simulated_predictions_quant[:, :, 1:2] # Take median

   forecast_df_point_for_spatial = format_predictions_to_dataframe(
       predictions=simulated_predictions_point,
       y_true_sequences=raw_y_true_sequences,
       target_name="sales",
       # No quantiles for point forecast
       forecast_horizon=H_OUT,
       output_dim=D_O,
       spatial_data_array=spatial_ids_df,
       spatial_cols_names=['location_id', 'region_code'],
       verbose=0
   )
   # Add dummy longitude/latitude for spatial plotting
   # In a real case, these would come from your spatial_data_array
   
   # 1. Work out how many rows the DF actually contains
   n_rows = len(forecast_df_point_for_spatial)      # → B * H_OUT (= 20)
   
   # 2. Create a base vector of length B (one per sample)
   base_lon = np.linspace(-100, -90, B)             #  [-100 … -90] 5 points
   base_lat = np.linspace(30, 35,  B)               #   [30 … 35]   5 points
   
   # -------------------------------------------------------
   # 3. Repeat each value H_OUT times so the final length is n_rows
   forecast_df_point_for_spatial["longitude"] = np.repeat(base_lon, H_OUT)
   forecast_df_point_for_spatial["latitude"]  = np.repeat(base_lat, H_OUT)
   # -------------------------------------------------------
 
   # If you prefer to keep the tile idiom you can do: 
   # forecast_df_point_for_spatial["longitude"] = np.tile(base_lon, H_OUT)
   # forecast_df_point_for_spatial["latitude"]  = np.tile(base_lat,  H_OUT)

   print("\n--- Visualizing Spatial Point Forecast ---")
   plot_forecasts(
       forecast_df=forecast_df_point_for_spatial,
       target_name="sales",
       # No quantiles
       output_dim=D_O,
       kind="spatial",
       horizon_steps=1, # Plot the first forecast step
       spatial_cols=['longitude', 'latitude'],
       figsize_per_subplot=(7, 6),
       verbose=1,
       # Additional kwargs for scatter plot
       s=50, cmap='coolwarm' # Marker size and colormap
   )
   # To save:
   # fig_path = os.path.join(evaluation_plot_dir, "workflow_spatial_point.png")
   # plt.savefig(fig_path)

**Expected Plot 4.2:**
   *(A scatter plot showing predicted 'sales_pred' values at different
   longitude/latitude points for the first forecast horizon step.)*

.. figure:: ../../images/workflow_utils_spatial_point.png
   :alt: Spatial Point Forecast from Workflow Utilities
   :align: center
   :width: 70%

   Spatial plot showing point forecast values across coordinates for a
   specific horizon step.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Conclusion
----------

This guide demonstrated a streamlined workflow using key utilities
from ``fusionlab.nn.utils`` and ``fusionlab.plot.evaluation``:

* **`prepare_model_inputs`** helps in correctly structuring the
  potentially complex list of inputs (static, dynamic, future) that
  forecasting models require, handling optional inputs gracefully.
* **`format_predictions_to_dataframe`** transforms raw model outputs
  (point or quantile, single or multi-target) into a standardized
  long-format DataFrame, which is essential for systematic analysis,
  storage, and as input to other evaluation tools.
* **`plot_forecasts`** offers a versatile way to quickly visualize
  these formatted predictions, allowing for temporal inspection of
  individual series and spatial distribution of forecasts.

By leveraging these functions, users can significantly reduce boilerplate
code, ensure data consistency, and focus more on model development
and interpretation. For more detailed evaluation metrics, please refer
to the :doc:`metrics` page.

