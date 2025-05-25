.. _user_guide_nn_utils:

==========================
Neural Network Utilities
==========================

The ``fusionlab.nn.utils`` module provides helpful functions for
working specifically with the neural network models in ``fusionlab``.
These utilities assist with tasks such as data preprocessing tailored
for models like TFT and XTFT, computing anomaly scores, generating
forecasts, and reshaping data arrays.

Anomaly Score Calculation
---------------------------

.. _compute_anomaly_scores_util:

compute_anomaly_scores
~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.compute_anomaly_scores`

**Purpose:** To calculate anomaly scores for time series data using
various statistical or algorithmic methods. These scores quantify the
"unusualness" of data points or sequences and can be used to inform
model training (e.g., with the `'from_config'` strategy in
:class:`~fusionlab.nn.XTFT`) or for post-hoc analysis and evaluation.

**Functionality / Methods:**
This function computes scores based on the chosen `method`. Let :math:`y`
denote a value from ``y_true``, :math:`\mu` its mean, :math:`\sigma` its
standard deviation, and :math:`\epsilon` a small constant:

* **`'statistical'` (or `'stats'`):** Calculates scores based on
  the squared normalized deviation from the mean (squared Z-score).
  Higher scores indicate larger deviations.

  .. math::
     Score(y) = \left(\frac{y - \mu}{\sigma + \epsilon}\right)^2

* **`'domain'`:** Uses a user-provided callable `domain_func(y)` or a
  default heuristic (e.g., assigning higher scores to negative or zero
  values if only positive values are expected).

* **`'isolation_forest'` (or `'if'`):** Uses the
  :class:`sklearn.ensemble.IsolationForest` algorithm. Scores are
  derived from the negative average path length required to isolate
  a sample (e.g., ``-iso.score_samples(y)``). Lower original scores
  (more negative) indicate higher anomaly likelihood; the function
  may transform these. Requires :mod:`sklearn`.

* **`'residual'`:** Requires providing corresponding predictions `y_pred`.
  Scores are based on the prediction error :math:`e = y_{true} - y_{pred}`:
  * `'mae'` sub-method: :math:`Score = |e|`
  * `'mse'` sub-method: :math:`Score = e^2`
  * `'rmse'` sub-method: :math:`Score = \sqrt{e^2 + \epsilon}`

*(Refer to the API documentation for details on parameters like
`threshold`, `contamination`, `sklearn_params`, `error_metric` etc.)*

**Usage Context:** This function is typically used *outside* the main
model training loop, for instance, to **pre-calculate** anomaly scores
from historical data, baseline model predictions, or domain rules.
These pre-calculated scores can then be passed to the
:class:`~fusionlab.nn.XTFT` model via the ``anomaly_config`` parameter
when using the ``anomaly_detection_strategy='from_config'``. It offers
a flexible way to define anomaly signals based on various approaches
before integrating them into an anomaly-aware training process.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   from fusionlab.nn.utils import compute_anomaly_scores
   # IsolationForest needed for 'if' method
   # from sklearn.ensemble import IsolationForest

   # Config
   batch_size = 4
   time_steps = 10
   features = 1

   # Dummy data (e.g., target values)
   y_true = np.random.randn(batch_size, time_steps, features).astype(np.float32)
   # Inject an anomaly
   y_true[1, 5, 0] = 10.0

   # Dummy predictions (for residual method)
   y_pred = y_true + np.random.normal(0, 0.5, y_true.shape).astype(np.float32)

   # 1. Calculate using 'statistical' method
   stat_scores = compute_anomaly_scores(
       y_true=y_true,
       method='statistical'
   )
   print("--- Statistical Scores ---")
   print(f"Input y_true shape: {y_true.shape}")
   print(f"Output scores shape: {stat_scores.shape}")
   # Expected shape: (4, 10, 1)
   print(f"Example score for anomalous point: {stat_scores[1, 5, 0]:.2f}")
   print(f"Example score for normal point: {stat_scores[0, 5, 0]:.2f}")

   # 2. Calculate using 'residual' (MAE) method
   resid_scores = compute_anomaly_scores(
       y_true=y_true,
       y_pred=y_pred,
       method='residual',
       error_metric='mae' # Use MAE for residuals
   )
   print("\n--- Residual (MAE) Scores ---")
   print(f"Input y_true shape: {y_true.shape}")
   print(f"Input y_pred shape: {y_pred.shape}")
   print(f"Output scores shape: {resid_scores.shape}")
   # Expected shape: (4, 10, 1)
   print(f"Example score for anomalous point: {resid_scores[1, 5, 0]:.2f}")
   print(f"Example score for normal point: {resid_scores[0, 5, 0]:.2f}")

   # 3. Calculate using 'isolation_forest' (requires sklearn)
   # Need to reshape data for Isolation Forest (Samples, Features)
   # For time series, might apply IF per timestep or on sequence features
   # Example: Apply per timestep (treat each B*T point independently)
   # try:
   #     y_true_flat = y_true.reshape(-1, features)
   #     if_scores_flat = compute_anomaly_scores(
   #         y_true=y_true_flat,
   #         method='isolation_forest',
   #         contamination=0.05 # Expected anomaly rate
   #     )
   #     if_scores = if_scores_flat.reshape(batch_size, time_steps, features)
   #     print("\n--- Isolation Forest Scores ---")
   #     print(f"Output scores shape: {if_scores.shape}")
   # except ImportError:
   #     print("\nSkipping Isolation Forest example (sklearn not found).")


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Data Preparation & Preprocessing
----------------------------------

These functions help prepare raw time series data into the specific
formats expected by models like TFT and XTFT.

.. _split_static_dynamic_util:

split_static_dynamic
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.split_static_dynamic`

**Purpose:** To separate an input array containing sequences of
combined features into two distinct arrays: one for static
(time-invariant extracted from a single time step) features and one
for dynamic (time-varying) features. This is often needed when a
simpler sequence generation tool creates a combined array first.

**Functionality:**
Given an input sequence tensor
:math:`\mathbf{X} \in \mathbb{R}^{B \times T \times N}`
(Batch, TimeSteps, NumCombinedFeatures), static feature indices
:math:`I_{static}`, dynamic feature indices :math:`I_{dynamic}`,
and a specific time step :math:`t_{static}` (usually 0) for
extracting static values:

1.  **Extract Static Features:** Selects features :math:`I_{static}`
    at time step :math:`t_{static}`.

    .. math::
       \mathbf{S}_{raw} = \mathbf{X}_{:, t_{static}, I_{static}} \in \mathbb{R}^{B \times |I_{static}|}

2.  **Extract Dynamic Features:** Selects features :math:`I_{dynamic}`
    across *all* time steps :math:`T`.

    .. math::
       \mathbf{D}_{raw} = \mathbf{X}_{:, :, I_{dynamic}} \in \mathbb{R}^{B \times T \times |I_{dynamic}|}

3.  **Reshape (Optional):** If ``reshape_static`` or ``reshape_dynamic``
    are True (default), adds a trailing dimension of 1:
    * :math:`\mathbf{S} \in \mathbb{R}^{B \times |I_{static}| \times 1}`
    * :math:`\mathbf{D} \in \mathbb{R}^{B \times T \times |I_{dynamic}| \times 1}`

**Usage Context:** Use this function after creating combined sequences
(e.g., using :func:`create_sequences` on a DataFrame containing both
static and dynamic columns) when you need to separate them into the
distinct static and dynamic input arrays required by models like
:class:`~fusionlab.nn.TemporalFusionTransformer` or
:class:`~fusionlab.nn.NTemporalFusionTransformer`. It assumes static
values are repeated across time in the input sequence.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   # Assuming split_static_dynamic is importable
   from fusionlab.nn.utils import split_static_dynamic

   # Config
   B, T, N = 4, 10, 5 # Batch, Time, Features (2 static, 3 dynamic)
   static_indices = [0, 1]
   dynamic_indices = [2, 3, 4]
   static_timestep_idx = 0 # Extract static from first step

   # Dummy combined sequence input
   combined_sequences = np.random.rand(B, T, N).astype(np.float32)

   # Split the sequences
   static_array, dynamic_array = split_static_dynamic(
       X=combined_sequences,
       static_features_indices=static_indices,
       dynamic_features_indices=dynamic_indices,
       static_timestep=static_timestep_idx,
       reshape_static=True, # Default
       reshape_dynamic=True # Default
   )

   print(f"Input combined sequence shape: {combined_sequences.shape}")
   print(f"Output static array shape: {static_array.shape}")
   print(f"Output dynamic array shape: {dynamic_array.shape}")
   # Expected: (4, 10, 5), (4, 2, 1), (4, 10, 3, 1)


.. _create_sequences_util:

create_sequences
~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.create_sequences`

**Purpose:** To transform a time series dataset (typically in a
Pandas DataFrame) into a format suitable for supervised learning
with sequence models. It creates input sequences (windows of past
data, including *all* available features) and their corresponding
target values (future data to predict from a specific column).

**Functionality:**
This function slides a window of a specified `sequence_length`
(:math:`T`) across the input DataFrame `df`. For each window, it
extracts:

1.  **Input Sequence** (:math:`\mathbf{X}^{(i)}`): A segment of the
    DataFrame containing **all feature columns** over :math:`T`
    consecutive time steps starting at index :math:`i`.

    .. math::
       \mathbf{X}^{(i)} = [\mathbf{df}_{i}, \mathbf{df}_{i+1}, ..., \mathbf{df}_{i+T-1}]

2.  **Target Value(s)** (:math:`y^{(i)}`): The value(s) from the
    specified `target_col` that occur immediately after the input
    sequence.
    
    * **Single-step** (`forecast_horizon=None` or 1): Target is
      :math:`\text{target\_{value}}_{i+T}`.
    * **Multi-step** (`forecast_horizon=H`): Target is the sequence
      :math:`[\text{target\_{value}}_{i+T}, ..., \text{target\_{value}}_{i+T+H-1}]`.

The function iterates through the DataFrame with a given `step` size
(stride=1 creates overlapping sequences). The `drop_last` parameter
controls handling of sequences near the end without full targets.

**Output:** Returns two NumPy arrays:
* `sequences`: Shape :math:`(\text{NumSeq}, T, \text{NumFeatures})`
* `targets`: Shape :math:`(\text{NumSeq},)` for single-step or :math:`(\text{NumSeq}, H)` for multi-step.

**Usage Context:** A fundamental preprocessing step. Use it after
cleaning and feature engineering your DataFrame to generate the
`(X, y)` pairs needed to train basic sequence models or as an
intermediate step before further processing (like using
:func:`split_static_dynamic`) for more complex models like TFT/XTFT.
It's simpler than :func:`reshape_xtft_data` as it doesn't automatically
separate static/dynamic/future types.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   from fusionlab.nn.utils import create_sequences

   # Dummy DataFrame
   data = {
       'Time': pd.to_datetime(pd.date_range('2023-01-01', periods=50)),
       'Feature1': np.random.rand(50) * 10,
       'Target': np.sin(np.arange(50) * 0.5) + 5
   }
   df = pd.DataFrame(data).set_index('Time')

   # Config
   sequence_length = 10 # Lookback window
   forecast_horizon = 5 # Predict 5 steps ahead
   target_column = 'Target'

   # Create sequences and multi-step targets
   X, y = create_sequences(
       df=df,
       sequence_length=sequence_length,
       forecast_horizon=forecast_horizon,
       target_col=target_column,
       step=1 # Default stride
   )

   print(f"Original DataFrame shape: {df.shape}")
   print(f"Output sequences (X) shape: {X.shape}")
   print(f"Output targets (y) shape: {y.shape}")
   # Expected shapes (approx): (50, 2), (36, 10, 2), (36, 5)


.. _compute_forecast_horizon_util:

compute_forecast_horizon
~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.compute_forecast_horizon`

**Purpose:** To determine the number of time steps (`forecast_horizon`)
between a specified prediction start date/time and end date/time,
optionally using the inferred frequency of provided time series data.

**Functionality:**
1.  **Frequency Inference:** Optionally infers the time series frequency
    (e.g., 'D', 'H', 'MS') from input `data` using `pandas.infer_freq`.
2.  **Date Parsing:** Converts `start_pred` and `end_pred` (strings,
    datetimes, or integer years) into pandas Timestamps.
3.  **Horizon Calculation:**

    * **With Frequency:** Calculates the number of steps by generating
    a date range between start and end using the inferred frequency.
    * **Without Frequency:** Estimates the horizon based on the time
    delta in the largest applicable unit (years, months, weeks, days).

**Usage Context:** Useful before creating sequences or configuring models
when the forecast period is defined by start/end dates rather than a
fixed number of steps. Helps ensure consistency between the desired
prediction range and parameters like ``forecast_horizon`` used in
:func:`create_sequences` or model initialization.

**Code Example:**

.. code-block:: python
   :linenos:

   import pandas as pd
   from fusionlab.nn.utils import compute_forecast_horizon

   # Example 1: Using frequency inference
   dates = pd.date_range('2023-01-01', periods=60, freq='D')
   df_daily = pd.DataFrame({'date': dates})
   horizon1 = compute_forecast_horizon(
       data=df_daily, # Provide data to infer frequency
       dt_col='date',
       start_pred='2023-03-01',
       end_pred='2023-03-10'
   )
   print(f"Horizon (daily data, Mar 1 to Mar 10): {horizon1}")
   # Expected: 10

   # Example 2: Using integer years (no frequency)
   horizon2 = compute_forecast_horizon(
       start_pred=2024,
       end_pred=2026
       # No data/freq provided, calculates based on years
   )
   print(f"Horizon (years 2024 to 2026): {horizon2}")
   # Expected: 3 (2024, 2025, 2026)


.. _prepare_spatial_future_data_util:

prepare_spatial_future_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.prepare_spatial_future_data`

**Purpose:** To prepare the specific static and dynamic input arrays
needed to generate **out-of-sample future predictions** using a trained
sequence model, particularly designed for datasets structured with
spatial groupings (e.g., locations, sensors).

**Functionality:**
Processes a DataFrame containing historical data, grouped by location,
to construct model inputs for time steps beyond the training data range.

1.  **Grouping & Sorting:** Groups data by `spatial_cols` and sorts
    by time (`dt_col`).
2.  **Last Sequence Extraction:** Extracts the most recent sequence
    (length `sequence_length`) for each group/location.
3.  **Input Preparation:** From the last sequence, it extracts:
    * **Static Inputs:** Values from `static_feature_names`.
    * **Dynamic Inputs:** Values from `dynamic_feature_indices`. This
      forms a template for future dynamic inputs.
4.  **Future Time Step Projection:** For each required future step (up
    to `forecast_horizon`, based on `future_years`):
    * It **updates the time feature(s)** within the dynamic template
    sequence to reflect the future time step, potentially applying
    inverse scaling if necessary using provided `scaling_params`
    (:math:`\mu, \sigma`). Other dynamic features are usually carried
    forward from the last known state.

    .. math::
       scaled\_{future}\_{time} = \frac{\text{future\_{time}} - \mu_{time}}{\sigma_{time} + \epsilon}

**Output:** Returns prepared NumPy arrays for static and future dynamic
inputs, ready for the model's `.predict()` method, along with metadata
like future time steps and location identifiers.

**Usage Context:** This is a crucial function for **generating actual
forecasts** after a model has been trained. Use it to create the input
arrays needed to predict future values not seen during training, based
on the last available historical data for each spatial group. See the
CLI tools (:doc:`/user_guide/tools`) or forecasting examples
(:doc:`/user_guide/examples/index`) for contextual usage. *(A direct code
example here would require significant setup; refer to full workflow
examples).*

.. _reshape_xtft_data_util:

reshape_xtft_data
~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.reshape_xtft_data`

**Purpose:** A comprehensive utility to transform a time series
DataFrame into the structured sequence format with **separate arrays**
for static, dynamic (past), future (known), and target features, as
required for *training* and *evaluating* models like XTFT and TFT.

**Functionality:**

1.  **Validation & Grouping:** Validates inputs, handles datetime column,
    optionally groups by `spatial_cols`, sorts by time.
2.  **Rolling Window:** Slides a window (length `time_steps` +
    `forecast_horizons`) across the data within each group.
3.  **Feature Extraction per Window:** For each window, it extracts and
    separates sequences for:
    * **Static Features:** From `static_cols` (value usually taken once per group).
    * **Dynamic Features:** From `dynamic_cols` for the lookback period (`time_steps`).
    * **Future Features:** From `future_cols` for a period relevant to the model
    (often lookback + horizon). *(Note: Verify precise time window used for
    future features based on implementation/model needs).*
    * **Target Features:** From `target_col` for the forecast period (`forecast_horizons`).
4.  **Output:** Returns a tuple of NumPy arrays:
    `(static_data, dynamic_data, future_data, target_data)`. Arrays for
    optional inputs (static/future) will be `None` if no corresponding columns
    are provided. Static data typically has shape :math:`(NumSeq, NumStatic)`, while
    others are 3D: :math:`(NumSeq, Time, Features)`.

**Mathematical Concept (Rolling Window):**
Generates pairs of input sequences (:math:`\mathbf{X}^{(i)}`, possibly split by type)
and target sequences (:math:`\mathbf{Y}^{(i)}`) for supervised training.

.. math::
   \mathbf{X}^{(i)} = \text{Features}_{i \dots i+T-1} \quad , \quad
   \mathbf{Y}^{(i)} = \text{Targets}_{i+T \dots i+T+H-1}

**Usage Context:** This is the **recommended primary tool** for preparing
datasets directly from DataFrames for training or evaluating `fusionlab`'s
TFT and XTFT models. It handles sequence creation, feature type separation,
and spatial grouping in one step, producing the exact array formats needed
by the models' `call` methods. See examples like
:doc:`/user_guide/examples/advanced_forecasting_xtft` for usage. *(A direct
code example here would be very similar to the full model examples; refer
to those for context).*

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">



Forecasting & Visualization
---------------------------

These functions assist with generating predictions from trained models
and visualizing the forecast results.

.. _generate_forecast_util:

generate_forecast
~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.generate_forecast`

**Purpose:** To generate future predictions using a pre-trained
``fusionlab`` model (like :class:`~fusionlab.nn.XTFT` or
:class:`~fusionlab.nn.transformers.TFT`). This function acts as a
high-level wrapper that handles preparing the necessary model inputs
from the end of the provided training data and formats the model's
output into a structured DataFrame.

**Functionality:**
1.  **Model Validation:** Ensures ``xtft_model`` is a valid Keras model.
2.  **Input Preparation:** Groups ``train_data`` by ``spatial_cols`` 
    (if provided). For each group, extracts the last sequence of
    length ``time_steps`` and constructs the input arrays
    ``[X_static, X_dynamic, X_future]`` needed for prediction, using
    logic similar to :func:`prepare_spatial_future_data`.
3.  **Prediction:** Calls ``xtft_model.predict()`` with the prepared
    arrays for each group. Conceptually:

    .. math::
       \hat{\mathbf{y}}_{t+1...t+H} = f_{model}(\mathbf{X}_{\text{static}}, \mathbf{X}_{\text{dynamic}}, \mathbf{X}_{\text{future}})

    where :math:`H` is the ``forecast_horizon``.
4.  **Output Formatting:** Organizes predictions into a Pandas
    DataFrame, including spatial identifiers and forecast dates/periods
    (``forecast_dt``). Creates columns for point predictions
    (``<tname>_pred``) or quantile predictions (``<tname>_qXX``).
5.  **Evaluation (Optional):** If ``test_data`` is provided, aligns
    forecasts with actuals and calculates/prints R² and Coverage
    Scores for the overlapping periods within the horizon.
6.  **Saving (Optional):** Saves the forecast DataFrame if ``savefile``
    is specified.

**Usage Context:** This is the primary function for **generating
out-of-sample forecasts** after model training. It simplifies input
preparation based on historical data and structures the results. See
the CLI tools (:doc:`/user_guide/tools`) or forecasting examples
(:doc:`/user_guide/examples/index`) for contextual usage. *(A direct
code example here would require significant setup; refer to full
workflow examples).*

.. _visualize_forecasts_util:

visualize_forecasts
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.plot.forecast.visualize_forecasts`

**Purpose:** To create visualizations comparing forecasted values
against actual values (if available), particularly useful for
spatial data or analyzing performance across different time periods.

**Functionality:**

1.  **Data Filtering:** Selects data for specified ``eval_periods`` from
    ``forecast_df`` and optional ``test_data``.
2.  **Column Identification:** Determines prediction, actual, and
    coordinate column names.
3.  **Plot Grid Setup:** Creates a `matplotlib` grid showing actual vs.
    predicted plots for each period.
4.  **Plotting:** Generates scatter plots for actuals (if available)
    and predictions, colored by value, using specified coordinates.
    Applies consistent colormap and range. Adds titles, labels,
    colorbars.
5.  **Display:** Shows the `matplotlib` figure.

**Usage Context:** Use this after generating forecasts (e.g., via
:func:`generate_forecast`) to visually inspect spatial patterns, compare
predictions to actuals over time, or assess quantile forecast spreads.
See the forecasting examples (:doc:`/user_guide/examples/index`) for
contextual usage. *(A direct code example here requires forecast data;
refer to full workflow examples).*

.. _forecast_single_step_util:

forecast_single_step
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.forecast_single_step`

**Purpose:** To generate a forecast for only the **next single time step**
(:math:`H=1`) using a pre-trained ``fusionlab`` model and
**pre-prepared** input arrays.

**Functionality:**

1.  **Input:** Takes a validated Keras ``xtft_model`` and ``inputs``
    (a list/tuple `[X_static, X_dynamic, X_future]`).
2.  **Prediction:** Calls ``xtft_model.predict(inputs)``. Assumes the
    model outputs multiple horizon steps and extracts the prediction
    for the first step (:math:`t+1`).
3.  **Output Formatting:** Creates a Pandas DataFrame including spatial
    columns (if specified), optional datetime, optional actuals (`y`),
    and prediction columns (``<tname>_pred`` or ``<tname>_qXX``).
4.  **Masking (Optional):** Masks predictions based on `mask_values` in `y`.
5.  **Evaluation (Optional):** Calculates R²/Coverage if `y` provided.
6.  **Saving (Optional):** Saves DataFrame if `savefile` specified.

**Usage Context:** Use when you only need the immediate next prediction
and have already manually prepared the required model input arrays
(`X_static`, `X_dynamic`, `X_future`). Useful in scenarios like
real-time single-step updates or when integrating into systems where
input preparation is handled separately.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   from fusionlab.nn.utils import forecast_single_step
   # from fusionlab.nn.transformers import XTFT 

   # Dummy Model
   class DummyModel(tf.keras.Model):
       def __init__(self, horizon=1, num_outputs=1):
           super().__init__()
           self.horizon = horizon
           self.num_outputs=num_outputs
           # Dummy layer to ensure model is callable
           self.dense = tf.keras.layers.Dense(horizon * num_outputs)
       def call(self, inputs):
           # Simulate output shape (B, H, O) or (B, H, Q)
           batch_size = tf.shape(inputs[1])[0] # Get from dynamic
           # Flatten and project to simulate processing
           flat_in = tf.keras.layers.Flatten()(inputs[1][:,-1,:]) # Use last step dynamic
           out_flat = self.dense(flat_in) # Shape (B, H*O)
           return tf.reshape(out_flat, (batch_size, self.horizon, self.num_outputs))

   # Config & Dummy Data
   B, T, H_model = 4, 12, 6 # Model trained for H=6
   D_dyn, D_stat, D_fut = 5, 3, 2
   static_in = tf.random.normal((B, D_stat))
   dynamic_in = tf.random.normal((B, T, D_dyn))
   future_in = tf.random.normal((B, T + H_model, D_fut)) # Future for model call
   # Dummy target for NEXT SINGLE STEP ONLY (H=1)
   y_true_single = tf.random.normal((B, 1)) # Needs to match output dim (usually 1)

   # Instantiate Dummy Model (trained for H=6, output O=1)
   model = DummyModel(horizon=H_model, num_outputs=1)
   # Dummy call to build model
   _ = model([static_in, dynamic_in, future_in])

   # Prepare inputs for forecast_single_step
   model_inputs = [static_in, dynamic_in, future_in]

   # Generate single step forecast
   forecast_df = forecast_single_step(
       xtft_model=model,
       inputs=model_inputs,
       target_col='Value',
       mode='point', # Point forecast
       y=y_true_single, # Provide actuals for next step
       evaluate=True, # Ask for evaluation
       spatial_cols=['ID'], # Assume static_in had ID column (needs adjustment)
       # dt_col='Timestamp' # Optional
   )

   print("\n--- Single Step Forecast ---")
   print(forecast_df.head())


.. _forecast_multi_step_util:

forecast_multi_step
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.forecast_multi_step`

**Purpose:** To generate forecasts for **multiple future time steps**
(up to a specified `forecast_horizon`) using a pre-trained
``fusionlab`` model and pre-prepared input arrays.

**Functionality:**
1.  **Input:** Takes ``xtft_model``, ``inputs = [X_s, X_d, X_f]``, and
    ``forecast_horizon``.
2.  **Prediction:** Calls ``xtft_model.predict(inputs)``, expecting an
    output covering the full horizon (shape :math:`(B, H, Outputs)`).
3.  **Output Formatting (Wide -> Long):** Organizes the multi-step
    predictions. It first creates a wide-format DataFrame (columns
    like `<tname>_pred_step1`, `<tname>_pred_step2`, etc.) using an
    internal `BatchDataFrameBuilder`, then likely converts it to a
    long format using :func:`step_to_long`, where each row represents
    a specific sample, forecast step, and prediction value.
4.  **Masking (Optional):** Masks predictions based on `mask_values` in `y`.
5.  **Evaluation (Optional):** Calculates R²/Coverage across all horizon
    steps if `y` (with shape :math:`(B, H, O)`) is provided.
6.  **Saving (Optional):** Saves the final DataFrame if `savefile` given.

**Usage Context:** Use when you need multi-step forecasts based on a
specific set of pre-prepared input arrays. It handles the organization
of the model's multi-step output into a structured DataFrame.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   # Assuming forecast_multi_step and a dummy model class are available
   from fusionlab.nn.utils import forecast_multi_step
   # from fusionlab.nn import XTFT # Replace with your actual model class

   # Dummy Model (same as single-step example)
   class DummyModel(tf.keras.Model):
       def __init__(self, horizon=1, num_outputs=1):
           super().__init__()
           self.horizon = horizon; self.num_outputs = num_outputs
           self.dense = tf.keras.layers.Dense(horizon * num_outputs)
       def call(self, inputs):
           batch_size = tf.shape(inputs[1])[0]
           flat_in = tf.keras.layers.Flatten()(inputs[1][:,-1,:])
           out_flat = self.dense(flat_in)
           return tf.reshape(out_flat, (batch_size, self.horizon, self.num_outputs))

   # Config & Dummy Data
   B, T, H = 4, 12, 6 # Horizon H=6
   D_dyn, D_stat, D_fut = 5, 3, 2
   output_dim = 1
   static_in = tf.random.normal((B, D_stat))
   dynamic_in = tf.random.normal((B, T, D_dyn))
   future_in = tf.random.normal((B, T + H, D_fut)) # Future for model call
   # Dummy target for MULTIPLE steps (H=6)
   y_true_multi = tf.random.normal((B, H, output_dim))

   # Instantiate Dummy Model (trained for H=6, output O=1)
   model = DummyModel(horizon=H, num_outputs=output_dim)
   _ = model([static_in, dynamic_in, future_in]) # Build

   # Prepare inputs for forecast_multi_step
   model_inputs = [static_in, dynamic_in, future_in]

   # Generate multi-step forecast
   forecast_df_multi = forecast_multi_step(
       xtft_model=model,
       inputs=model_inputs,
       target_col='Value',
       forecast_horizon=H, # Specify horizon
       mode='point',
       y=y_true_multi, # Provide multi-step actuals
       evaluate=True,
       spatial_cols=['ID'], # Assume static_in had ID
       # dt_col='Timestamp' # Optional
   )

   print("\n--- Multi Step Forecast (Long Format) ---")
   print(forecast_df_multi.head()) # Display long format


.. _generate_forecast_with_util:

generate_forecast_with
~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.generate_forecast_with`

**Purpose:** A convenient wrapper function that automatically calls
either :func:`forecast_single_step` or :func:`forecast_multi_step`
based on the specified ``forecast_horizon``.

**Functionality:**
1. Takes all the same arguments as :func:`forecast_single_step` and
   :func:`forecast_multi_step`.
2. Checks ``forecast_horizon``:
   * If ``forecast_horizon == 1``, calls :func:`forecast_single_step`.
   * If ``forecast_horizon > 1``, calls :func:`forecast_multi_step`.
3. Returns the DataFrame produced by the called function.

**Usage Context:** Provides a unified interface for generating forecasts
from pre-prepared input arrays, regardless of whether you need one step
or multiple steps ahead. Simplifies workflows where the forecast length
might be a variable parameter.

**Code Example:**

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import tensorflow as tf
   # Assuming generate_forecast_with and dummy model are available
   from fusionlab.nn.utils import generate_forecast_with
   # from my_models import DummyModel # Use same dummy model as above

   # Use Dummy Model and Data from previous examples
   B, T, H = 4, 12, 6
   D_dyn, D_stat, D_fut = 5, 3, 2
   output_dim = 1
   static_in = tf.random.normal((B, D_stat))
   dynamic_in = tf.random.normal((B, T, D_dyn))
   future_in = tf.random.normal((B, T + H, D_fut))
   y_true_multi = tf.random.normal((B, H, output_dim))
   model = DummyModel(horizon=H, num_outputs=output_dim)
   _ = model([static_in, dynamic_in, future_in]) # Build
   model_inputs = [static_in, dynamic_in, future_in]

   # Example 1: Generate single step (H=1 passed implicitly)
   print("--- generate_forecast_with (H=1) ---")
   df_single = generate_forecast_with(
       xtft_model=model,
       inputs=model_inputs,
       target_col='Value',
       # forecast_horizon=1 # (Default or set to 1)
       mode='point',
       y=y_true_multi[:, :1, :] # Provide only first step actuals
   )
   print(df_single.head())

   # Example 2: Generate multi step
   print("\n--- generate_forecast_with (H=6) ---")
   df_multi = generate_forecast_with(
       xtft_model=model,
       inputs=model_inputs,
       target_col='Value',
       forecast_horizon=H, # Explicitly set > 1
       mode='point',
       y=y_true_multi
   )
   print(df_multi.head())


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


Data Reshaping Utilities
------------------------

These functions assist in transforming data between different formats
commonly encountered in multi-step time series forecasting workflows.

.. _step_to_long_util:

step_to_long
~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.step_to_long`

**Purpose:** To transform a DataFrame containing multi-step forecast
results from a "wide" format into a "long" format. In the wide
format, each forecast step typically occupies separate columns (e.g.,
`target_q50_step1`, `target_q50_step2`). The long format reshapes
this so that each row represents a single prediction for a specific
sample (identified by original index or identifier columns), time
step into the future, and possibly quantile level.

**Functionality:**
1. Takes a wide-format DataFrame `df` as input, along with metadata
   like `tname` (target variable base name), `dt_col` (datetime/period
   column), `spatial_cols` (identifier columns), and `mode`
   ('quantile' or 'point').
2. Identifies the columns corresponding to different forecast steps
   and quantiles based on naming conventions. It typically looks for
   patterns like `_stepX` and `_qYY` appended to the `tname`.
3. Uses internal helper functions (likely employing Pandas melting,
   stacking, or pivoting operations) to unpivot the step-based columns.
4. Reshapes the data, creating new columns usually named 'step' (for
   the forecast horizon step) and potentially 'quantile'. It
   consolidates the prediction values into a single column (e.g.,
   'predicted_value').
5. Identifier columns (`dt_col`, `spatial_cols`), and any actual target
   value columns present in the wide DataFrame are typically preserved
   and duplicated appropriately across the newly created long-format rows.
6. Optionally sorts the final long-format DataFrame based on
   identifiers and step.

**Usage Context:** This function is primarily used as an internal
helper within :func:`forecast_multi_step` to convert the initially
generated wide-format predictions into a more standardized long format,
which is often easier for plotting or subsequent analysis (e.g.,
evaluating performance per step). Users might also find it useful if
they have wide-format forecast data from other sources and want to
reshape it.

**Code Example:**

.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   from fusionlab.nn.utils import step_to_long

   # 1. Create Dummy Wide-Format DataFrame (simulating output)
   # (e.g., as might be initially created by forecast_multi_step)
   data_wide = {
       'ID': [1, 1, 2, 2],
       'ForecastStartDate': pd.to_datetime(['2023-01-01', '2023-01-02',
                                            '2023-01-01', '2023-01-02']),
       'Actual_step1': [10, 11, 20, 21],
       'Actual_step2': [12, 13, 22, 23],
       # Point predictions
       'Value_pred_step1': [9.8, 11.2, 19.5, 21.3],
       'Value_pred_step2': [11.5, 13.1, 21.8, 23.2],
       # Quantile predictions
       'Value_q10_step1': [8.8, 10.2, 18.5, 20.3],
       'Value_q50_step1': [9.8, 11.2, 19.5, 21.3], # Same as point
       'Value_q90_step1': [10.8, 12.2, 20.5, 22.3],
       'Value_q10_step2': [10.5, 12.1, 20.8, 22.2],
       'Value_q50_step2': [11.5, 13.1, 21.8, 23.2], # Same as point
       'Value_q90_step2': [12.5, 14.1, 22.8, 24.2],
   }
   df_wide = pd.DataFrame(data_wide)
   print("--- Original Wide DataFrame ---")
   print(df_wide)

   # 2. Convert Point Forecast Columns to Long Format
   df_long_point = step_to_long(
       df=df_wide.drop(columns=[c for c in df_wide if '_q' in c]), # Keep only pred cols
       tname='Value',
       dt_col='ForecastStartDate',
       mode='point',
       spatial_cols=['ID'],
       pred_colname='Value_pred' # Name for the prediction column
   )
   print("\n--- Long Format DataFrame (Point Mode) ---")
   print(df_long_point)

   # 3. Convert Quantile Forecast Columns to Long Format
   df_long_quantile = step_to_long(
       df=df_wide.drop(columns=[c for c in df_wide if '_pred_' in c]), # Keep only quantile cols
       tname='Value',
       dt_col='ForecastStartDate',
       mode='quantile',
       spatial_cols=['ID'],
       pred_colname='Value_quantile' # Name for the prediction column
   )
   print("\n--- Long Format DataFrame (Quantile Mode) ---")
   print(df_long_quantile)


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
