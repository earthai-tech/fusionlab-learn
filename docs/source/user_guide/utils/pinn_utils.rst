.. _pinn_utils_guide:

============================
PINN Data Utilities
============================

The Physics-Informed Neural Network (PINN) models in ``fusionlab-learn``,
such as ``PIHALNet`` and ``TransFlowSubsNet``, have unique data
requirements. They need not only the standard feature-based inputs
(static, dynamic, future) but also spatio-temporal coordinates
(:math:`t, x, y`) to compute the physics-based loss.

This section details the specialized utility functions designed to
handle the complex data preparation and results formatting tasks
associated with these models.


Sequence Generation for PINNs (`prepare_pinn_data_sequences`)
-------------------------------------------------------------
:API Reference: :func:`~fusionlab.nn.pinn.utils.prepare_pinn_data_sequences`

This is the most critical data preparation utility for PINN models. Its
primary role is to transform a flat, time-series `pandas.DataFrame`
into the complex, multi-part sequence format required for training.

The function iterates through the DataFrame, creating rolling windows
to generate sequences of a specified length (`time_steps`) and
forecasting predictions for a given `forecast_horizon`. Its key
distinction is that it generates **both** the feature-based input
tensors and the crucial **`coords`** tensor needed by the physics module.

**Key Parameters:**

* **`df`**: The input DataFrame containing all features, targets, and
    coordinates in a long format.
* **`group_id_cols`**: A list of columns (e.g., `['longitude', 'latitude']`)
    used to identify and separate individual time series within the
    DataFrame. The function generates sequences independently for each group.
* **`time_steps`**: The length of the historical lookback window for the
    dynamic features.
* **`forecast_horizon`**: The number of future steps to predict.
* **`*_cols` arguments**: Various arguments (`dynamic_cols`,
    `static_cols`, `future_cols`, `subsidence_col`, etc.) that map
    column names in the DataFrame to their respective roles.
* **`normalize_coords`**: A boolean flag that, when ``True``, scales the
    spatio-temporal coordinate values (:math:`t, x, y`) to a 0-1 range,
    which is highly recommended for stable gradient calculations in the
    PINN loss.

**Workflow and Outputs:**

The function returns two dictionaries containing NumPy arrays:

1.  **`inputs_dict`**: Contains all the input tensors required by the
    model's ``call`` method.
    * ``'coords'``: The spatio-temporal coordinates for the forecast
        horizon, shape: :math:`(N, H, 3)`.
    * ``'static_features'``: Shape: :math:`(N, D_s)`.
    * ``'dynamic_features'``: Shape: :math:`(N, T, D_d)`.
    * ``'future_features'``: Shape: :math:`(N, H, D_f)`.
2.  **`targets_dict`**: Contains the ground-truth target tensors.
    * ``'subsidence'``: Shape: :math:`(N, H, O_s)`.
    * ``'gwl'``: Shape: :math:`(N, H, O_g)`.

Here, :math:`N` is the total number of sequences generated, :math:`T` is
`time_steps`, :math:`H` is `forecast_horizon`, and :math:`D` and :math:`O`
are feature/output dimensions.

Usage Example
~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences

   # 1. Create a sample DataFrame with multiple time series groups
   data = []
   for group_id in range(2): # 2 locations
       for year in range(2010, 2020): # 10 years of data
           data.append({
               'year': year,
               'lon': 113.5 + group_id,
               'lat': 22.5 + group_id,
               'geology_type': f'type_{group_id}',
               'subsidence': -10 - (year - 2010) * 2 - group_id,
               'GWL': 5 - (year - 2010) * 0.5 - group_id,
               'rainfall': 1500 + np.random.randn() * 100
           })
   df = pd.DataFrame(data)

   # 2. Define feature sets
   static_cols = ['geology_type'] # This will need to be one-hot encoded first
   dynamic_cols = ['GWL', 'rainfall']
   df = pd.get_dummies(df, columns=static_cols, dtype=float)
   static_cols_encoded = [c for c in df.columns if 'geology_type' in c]

   # 3. Generate sequences
   inputs, targets, _ = prepare_pinn_data_sequences(
       df=df,
       time_col='year',
       lon_col='lon', lat_col='lat',
       subsidence_col='subsidence', gwl_col='GWL',
       dynamic_cols=dynamic_cols,
       static_cols=static_cols_encoded,
       group_id_cols=['lon', 'lat'],
       time_steps=5,
       forecast_horizon=3
   )

   # 4. Print the shapes of the output
   print("--- Generated Input Shapes ---")
   for name, arr in inputs.items():
       print(f"  '{name}': {arr.shape if arr is not None else 'None'}")
   print("\n--- Generated Target Shapes ---")
   for name, arr in targets.items():
       print(f"  '{name}': {arr.shape}")

**Expected Output:**

.. code-block:: text

   --- Generated Input Shapes ---
     'coords': (4, 3, 3)
     'static_features': (4, 2)
     'dynamic_features': (4, 5, 2)
     'future_features': None

   --- Generated Target Shapes ---
     'subsidence': (4, 3, 1)
     'gwl': (4, 3, 1)
     
.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Formatting Model Outputs (`format_pinn_outputs`)
------------------------------------------------
:API Reference: :func:`~fusionlab.nn.pinn.utils.format_pinn_outputs`

This function is the counterpart to the preparation utility. It takes
the raw dictionary of prediction tensors from a model's `.predict()`
call and transforms it into a clean, long-format ``pandas.DataFrame``
that is easy to analyze, visualize, or export.

It robustly handles multi-target outputs, point or quantile forecasts,
and can merge the predictions with ground-truth values, coordinates,
and other static metadata for a complete results summary.

.. note::
   The function ``format_pihalnet_predictions`` is a deprecated alias
   for ``format_pinn_outputs`` and is maintained for backward
   compatibility. New code should use ``format_pinn_outputs``.

**Usage Example:**

.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   from fusionlab.nn.pinn.utils import format_pinn_outputs

   # 1. Create dummy model outputs and true values
   B, H, Q_len = 4, 3, 3 # Batch, Horizon, Num Quantiles
   quantiles = [0.1, 0.5, 0.9]
   
   predictions = {
       'subs_pred': np.random.rand(B, H, Q_len),
       'gwl_pred': np.random.rand(B, H, Q_len)
   }
   y_true = {
       'subsidence': np.random.rand(B, H, 1),
       'gwl': np.random.rand(B, H, 1)
   }
   # Dummy coordinates and static IDs
   coords = np.random.rand(B, H, 3)
   ids = pd.DataFrame({'site_id': [f'site_{i}' for i in range(B)]})

   # 2. Format the predictions into a DataFrame
   df_results = format_pinn_outputs(
       predictions=predictions,
       y_true_dict=y_true,
       quantiles=quantiles,
       model_inputs={'coords': coords}, # Provide coords for inclusion
       ids_data_array=ids,
       target_mapping={'subs_pred': 'subsidence', 'gwl_pred': 'gwl'}
   )

   # 3. Display the head of the resulting DataFrame
   print(df_results.head())

**Expected Output:**

.. code-block:: text

      sample_idx  forecast_step   coord_t   coord_x   coord_y     site_id  subsidence_q10  subsidence_q50  subsidence_q90  subsidence_actual   gwl_q10   gwl_q50   gwl_q90  gwl_actual
   0           0              1  0.722839  0.906981  0.887213      site_0        0.038539        0.155893        0.521225           0.822144  0.518956  0.957137  0.301275    0.287298
   1           0              2  0.648172  0.439833  0.413819      site_0        0.839845        0.536343        0.995442           0.199451  0.347561  0.569412  0.607545    0.409383
   2           0              3  0.944208  0.343521  0.364445      site_0        0.398938        0.473836        0.472256           0.024223  0.395424  0.787321  0.523829    0.231295
   3           1              1  0.592753  0.134589  0.334085      site_1        0.784036        0.563038        0.286823           0.917769  0.923229  0.182537  0.425501    0.984734
   4           1              2  0.340941  0.933219  0.030580      site_1        0.201277        0.875417        0.053173           0.755497  0.775355  0.871836  0.536763    0.159397


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Coordinate and Feature Scaling (`normalize_for_pinn`)
------------------------------------------------------
:API Reference: :func:`~fusionlab.nn.pinn.utils.normalize_for_pinn`

Normalization is crucial for training PINNs. The coordinate inputs
(:math:`t, x, y`) that are used to compute PDE derivatives should
ideally be scaled to a standard range (e.g., [0, 1]) to ensure the
gradients are well-behaved and stable.

This utility function provides a centralized way to handle this scaling.

* **`scale_coords=True`**: This primary option applies a ``MinMaxScaler``
    to the `time_col`, `lon_col`, and `lat_col` together, preserving
    their relative relationships while scaling them to the [0, 1] range.
* **`cols_to_scale='auto'`**: This feature automatically detects all other
    numerical columns in the DataFrame (excluding booleans/one-hot
    encoded columns) and applies a separate scaler to them.

**Usage Example:**

.. code-block:: python
   :linenos:

   import pandas as pd
   from fusionlab.nn.pinn.utils import normalize_for_pinn

   # 1. Create a sample DataFrame
   df = pd.DataFrame({
       'time': [2020.0, 2021.0, 2022.0],
       'lon': [-122.4, -122.3, -122.2],
       'lat': [37.7, 37.8, 37.9],
       'rainfall': [500, 600, 550],
       'is_event': [0, 1, 0] # A one-hot style column
   })

   # 2. Normalize coordinates and auto-selected features
   df_scaled, coord_scaler, other_scaler = normalize_for_pinn(
       df,
       time_col='time',
       lon_col='lon',
       lat_col='lat',
       scale_coords=True,
       cols_to_scale='auto' # Auto-detect 'rainfall'
   )

   # 3. Display results
   print("--- Original DataFrame ---")
   print(df)
   print("\n--- Scaled DataFrame ---")
   print(df_scaled)
   print(f"\nCoordinate Scaler Range: {coord_scaler.data_range_}")
   print(f"Feature Scaler Range: {other_scaler.data_range_}")

**Expected Output:**

.. code-block:: text

   --- Original DataFrame ---
       time     lon    lat  rainfall  is_event
   0  2020.0  -122.4   37.7       500         0
   1  2021.0  -122.3   37.8       600         1
   2  2022.0  -122.2   37.9       550         0

   --- Scaled DataFrame ---
      time   lon   lat  rainfall  is_event
   0   0.0   0.0   0.0       0.0         0
   1   0.5   0.5   0.5       1.0         1
   2   1.0   1.0   1.0       0.5         0

   Coordinate Scaler Range: [2.  0.2 0.2]
   Feature Scaler Range: [100.]

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Coordinate Extraction Utilities
-------------------------------
The library includes two low-level helpers, ``extract_txy_in`` and
``extract_txy``, used internally to robustly parse the :math:`t, x, y`
coordinate tensors from different input structures (e.g., a single
concatenated tensor vs. a dictionary).

While you may not need to call these directly, understanding their
difference is useful for advanced customization.

**The Difference:**

The key difference lies in how they handle the dimensionality of the
output tensors.

**extract_txy_in** (Internal, Strict)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.pinn.utils.extract_txy_in`

This version is stricter and is designed for internal model components
that always expect a 3D spatio-temporal tensor. It **always**
ensures the output tensors have a rank of 3. If it receives a 2D
input of shape `(batch, 3)`, it will automatically expand it to
`(batch, 1, 3)` before slicing, ensuring a consistent 3D output like
`(batch, 1, 1)`.

.. code-block:: python
   :linenos:
   
   from fusionlab.nn.pinn.utils import extract_txy_in
   
   # A 2D spatial tensor (batch, features)
   coords_2d = tf.random.normal((4, 3))
   # A 3D spatio-temporal tensor (batch, time, features)
   coords_3d = tf.random.normal((4, 10, 3))
   
   t2, x2, y2 = extract_txy_in(coords_2d)
   t3, x3, y3 = extract_txy_in(coords_3d)

   print(f"Input 2D shape: {coords_2d.shape}")
   print(f"Output shape from 2D input: {t2.shape}")
   print(f"\nInput 3D shape: {coords_3d.shape}")
   print(f"Output shape from 3D input: {t3.shape}")

**Expected Output:**

.. code-block:: text

   Input 2D shape: (4, 3)
   Output shape from 2D input: (4, 1, 1)

   Input 3D shape: (4, 10, 3)
   Output shape from 3D input: (4, 10, 1)
   
   
**extract_txy** (Flexible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.pinn.utils.extract_txy`

This version is more flexible and is controlled by the `expect_dim`
parameter. It can return 2D or 3D tensors based on the input and
the desired output format, making it suitable for different parts
of a model that may operate on data with or without a time
dimension.

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.utils import extract_txy

   # Using the same 2D and 3D tensors
   
   # Case 1: expect_dim=None (preserves rank)
   t, x, y = extract_txy(coords_2d, expect_dim=None)
   print(f"With expect_dim=None, 2D input gives output shape: {t.shape}")
   
   # Case 2: expect_dim='3d' (expands 2D to 3D)
   t, x, y = extract_txy(coords_2d, expect_dim='3d')
   print(f"With expect_dim='3d', 2D input gives output shape: {t.shape}")

**Expected Output:**

.. code-block:: text

   With expect_dim=None, 2D input gives output shape: (4, 1)
   With expect_dim='3d', 2D input gives output shape: (4, 1, 1)