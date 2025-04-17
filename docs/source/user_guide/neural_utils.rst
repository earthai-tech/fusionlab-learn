.. _user_guide_neural_utils:

===========================
Neural Network Utilities
===========================

The ``fusionlab.nn.utils`` module provides helpful functions for
working with the neural network models in ``fusionlab``. These
utilities assist with tasks such as data preprocessing tailored for
models like TFT and XTFT, computing anomaly scores, generating
forecasts, and reshaping data arrays.

Anomaly Score Calculation
-------------------------

.. _compute_anomaly_scores:

compute_anomaly_scores
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.compute_anomaly_scores`

**Purpose:** To calculate anomaly scores for time series data using
various methods. These scores quantify the "unusualness" of data
points and can be used to inform model training (e.g., with the
`'from_config'` strategy in :class:`~fusionlab.nn.XTFT`) or for
post-hoc analysis.

**Functionality / Methods:**
This function computes scores based on the chosen `method`. Let $y$
denote a value from `y_true`, $\mu$ its mean, $\sigma$ its standard
deviation, and $\epsilon$ a small constant:

* **`'statistical'` (or `'stats'`):** Calculates scores based on
    normalized deviation from the mean.

    .. math::
       Score(y) = \left(\frac{y - \mu}{\sigma + \epsilon}\right)^2

* **`'domain'`:** Uses a user-provided `domain_func(y)` or a
    default heuristic (e.g., penalizing negative values).

* **`'isolation_forest'` (or `'if'`):** Uses the Isolation Forest
    algorithm. Scores are derived from `-iso.score_samples(y)`.
    Requires :mod:`sklearn`.

* **`'residual'`:** Requires `y_pred`. Scores based on the error
    $e = y_{true} - y_{pred}$:
    * `'mse'`: $Score = e^2$
    * `'mae'`: $Score = |e|$
    * `'rmse'`: $Score = \sqrt{e^2 + \epsilon}$

*(Refer to the API documentation for details on parameters like
`threshold`, `contamination`, etc.)*

**Usage Context:** This function is typically used *outside* the main
model training loop, for instance, to pre-calculate anomaly scores
that can then be fed into the :class:`~fusionlab.nn.XTFT` model via
the `anomaly_config` parameter when using the `'from_config'`
strategy. It provides flexibility in defining how anomalies are
quantified based on different statistical, algorithmic, or
domain-specific approaches.

.. raw:: html

    <hr>
    
Data Preparation & Preprocessing
----------------------------------

These functions help prepare raw time series data into the specific
formats expected by models like TFT and XTFT.

.. _split_static_dynamic:

split_static_dynamic
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.split_static_dynamic`

**Purpose:** To separate an input array containing sequences of
features into two distinct arrays: one for static (time-invariant)
features and one for dynamic (time-varying) features. This is a
common preprocessing step for models that handle these input types
separately.

**Functionality:**
Given an input sequence tensor
$\mathbf{X} \in \mathbb{R}^{B \times T \times N}$ (Batch, TimeSteps,
NumFeatures), static feature indices $I_{static}$, dynamic feature
indices $I_{dynamic}$, and a specific time step $t_{static}$ (usually
0):

1.  **Extract Static Features:** Selects features $I_{static}$ at the
    single time step $t_{static}$.

    .. math::
       \mathbf{S}_{raw} = \mathbf{X}_{:, t_{static}, I_{static}} \in \mathbb{R}^{B \times |I_{static}|}

2.  **Extract Dynamic Features:** Selects features $I_{dynamic}$ across
    *all* time steps $T$.

    .. math::
       \mathbf{D}_{raw} = \mathbf{X}_{:, :, I_{dynamic}} \in \mathbb{R}^{B \times T \times |I_{dynamic}|}

3.  **Reshape (Optional):** If `reshape_static` or `reshape_dynamic`
    are True (default), the extracted arrays are reshaped. The default
    reshaping adds a trailing dimension of 1, suitable for some
    Keras layers:
    * $\mathbf{S} \in \mathbb{R}^{B \times |I_{static}| \times 1}$
    * $\mathbf{D} \in \mathbb{R}^{B \times T \times |I_{dynamic}| \times 1}$
    Custom shapes can also be provided via parameters.

**Usage Context:** Use this function during data preparation when your
raw sequence data (with combined static and dynamic columns) needs
to be split into the separate input arrays required by models like
:class:`~fusionlab.nn.TemporalFusionTransformer`,
:class:`~fusionlab.nn.NTemporalFusionTransformer`, or
:class:`~fusionlab.nn.XTFT`.



.. _create_sequences:

create_sequences
~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.create_sequences`

**Purpose:** To transform a time series dataset (typically in a
Pandas DataFrame) into a format suitable for supervised learning
with sequence models. It creates input sequences (windows of past
data) and their corresponding target values (future data to predict).

**Functionality:**
This function slides a window of a specified `sequence_length` ($T$)
across the input DataFrame `df`. For each window, it extracts:

1.  **Input Sequence ($\mathbf{X}^{(i)}$):** A segment of the DataFrame
    containing all features over $T$ consecutive time steps starting
    at index $i$.

    .. math::
       \mathbf{X}^{(i)} = [\mathbf{df}_{i}, \mathbf{df}_{i+1}, ..., \mathbf{df}_{i+T-1}]

2.  **Target Value(s) ($y^{(i)}$):** The value(s) from the specified
    `target_col` that occur immediately after the input sequence.
    * **Single-step forecasting** (`forecast_horizon=None` or 1):
        The target is the single value at time step $i+T$.

        .. math::
           y^{(i)} = \text{target\_value}_{i+T}

    * **Multi-step forecasting** (`forecast_horizon=H`): The target
        is the sequence of $H$ values from the `target_col` starting
        at time step $i+T$.

        .. math::
           y^{(i)} = [\text{target\_value}_{i+T}, ..., \text{target\_value}_{i+T+H-1}]

The function iterates through the DataFrame with a given `step` size
(stride). Setting `step=1` (default) creates maximally overlapping
sequences. The `drop_last` parameter controls whether sequences at
the very end of the DataFrame, which might not have a complete
corresponding target, are included.

**Output:** Returns two NumPy arrays:
* `sequences`: Shape `(NumSequences, SequenceLength, NumFeatures)`
* `targets`: Shape `(NumSequences,)` for single-step or
    `(NumSequences, ForecastHorizon)` for multi-step.

**Usage Context:** This is a fundamental preprocessing step for time
series forecasting. Use it after cleaning and feature engineering your
DataFrame to generate the `(X, y)` pairs needed to train sequence
models like LSTMs, GRUs, TFT, and XTFT. The output `sequences` array
might then be further processed (e.g., using
:func:`split_static_dynamic`) depending on the specific model's input
requirements.

.. _compute_forecast_horizon:

compute_forecast_horizon
~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.compute_forecast_horizon`

**Purpose:** To determine the number of time steps (`forecast_horizon`)
between a specified prediction start date/time and end date/time,
often based on the frequency of the time series data.

**Functionality:**
1.  **Frequency Inference:** If `data` (e.g., DataFrame with `dt_col`,
    Series, list of datetimes) is provided, the function attempts to
    infer the time series frequency (e.g., 'D' for daily, 'H' for
    hourly) using `pandas.infer_freq`.
2.  **Date Parsing:** Converts `start_pred` and `end_pred` inputs
    (which can be strings, datetime objects, or integers representing
    years) into pandas Timestamp objects.
3.  **Horizon Calculation (with Frequency):** If a frequency `freq`
    was successfully inferred, it calculates the number of steps by
    generating a date range between `start_pred` and `end_pred` using
    that frequency: `len(pd.date_range(start, end, freq=freq))`.
4.  **Horizon Calculation (without Frequency):** If frequency cannot
    be inferred or no data is provided, it calculates the time delta
    between `start_pred` and `end_pred` and estimates the horizon
    based on the largest applicable time unit (years, months, weeks,
    or days). For example, if the difference is 400 days, it might
    return $400 // 365 + 1 = 2$ years (depending on exact logic).
5.  **Error Handling:** Manages invalid inputs or inability to parse
    dates based on the `error` parameter ('raise', 'warn').

**Usage Context:** Useful before creating sequences or configuring models
when you know the desired start and end dates of your forecast period
but need to determine the corresponding number of steps (`forecast_horizon`)
based on your data's time frequency. Helps ensure consistency between
the desired prediction range and model parameters or data generation
steps like :func:`create_sequences`.


.. _prepare_spatial_future_data:

prepare_spatial_future_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.prepare_spatial_future_data`

**Purpose:** To prepare the static and dynamic input arrays needed
to make *future predictions* using a trained sequence model,
especially when data is organized by spatial locations (e.g.,
different sensors or stores).

**Functionality:**
This function processes a dataset containing historical data, potentially
grouped by location, to construct the inputs required for forecasting
beyond the last known time point.

1.  **Grouping:** Groups the input `final_processed_data` DataFrame
    by spatial identifiers (e.g., 'longitude', 'latitude' specified
    via `spatial_cols`). If no spatial columns are given, treats the
    entire dataset as one group.
2.  **Last Sequence Extraction:** For each group (location), it sorts
    the data by time (`dt_col`) and extracts the *most recent* sequence
    of length `sequence_length`.
3.  **Static Input Preparation:** Extracts the static feature values
    (defined by `static_feature_names` and `encoded_cat_columns`)
    from this last sequence. These static features are assumed to remain
    constant for future predictions for that location.
4.  **Dynamic Input Preparation:** Extracts the dynamic feature values
    (defined by `dynamic_feature_indices`) from the last sequence.
5.  **Future Time Step Generation:** Determines the future time steps
    to predict based on `forecast_horizon` and optionally provided
    `future_years`.
6.  **Future Dynamic Input Construction:** For each future time step:
    * It takes the *last known dynamic sequence* as a template.
    * It **updates the time feature** within this template sequence
      to reflect the specific future time step being predicted. This
      update often involves scaling the future time value using
      provided or computed `scaling_params` ($\mu, \sigma$):

      .. math::
         scaled\_time = \frac{\text{future\_time} - \mu}{\sigma + \epsilon}

    * Other dynamic features in the template sequence (from the last
      known data) are typically carried forward.
7.  **Output Collection:** Collects the prepared static inputs (repeated
    for each future step of each location) and the corresponding
    time-updated future dynamic inputs into NumPy arrays. It also
    returns lists containing metadata like the future time steps,
    location IDs, and coordinates for traceability.

**Usage Context:** Use this function *after* training a model, when
you want to generate forecasts for future periods not present in the
original dataset. It constructs the specific input arrays needed for
the model's `.predict()` method by using the last known state for
each location and projecting it forward by updating the time feature.

.. _reshape_xtft_data:

reshape_xtft_data
~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.reshape_xtft_data`

**Purpose:** A comprehensive utility to transform a time series
DataFrame into the structured sequence format required for *training*
and *evaluating* complex sequence models like XTFT and TFT. It handles
the creation of rolling windows and separates static, dynamic, future,
and target features.

**Functionality:**
1.  **Validation & Grouping:** Validates input columns and converts the
    datetime column (`dt_col`). Optionally groups the DataFrame by
    `spatial_cols`; otherwise processes the entire DataFrame as one group.
    Sorts data by time within each group.
2.  **Rolling Window Generation:** Iterates through each group using a
    sliding window approach based on `time_steps` ($T$) and
    `forecast_horizons` ($H$).
3.  **Feature Extraction per Window:** For each window starting at index $i$:
    * **Static Features:** Extracts values from `static_cols`. Typically
        takes the value from the first row of the group (assuming static
        within a group).
    * **Dynamic Features:** Extracts the sequence from `dynamic_cols` for
        time steps $i$ to $i+T-1$.
    * **Future Features:** Extracts values from `future_cols`. The current
        implementation appears to take the future feature values from the
        *start* of the input window ($i$) and repeats them across all $T$
        time steps of the input sequence. *(Note: This specific handling
        of future features might differ from other conventions and should
        be considered during model design and interpretation).*
    * **Target Features:** Extracts the sequence from `target_col` for the
        *prediction* window, i.e., time steps $i+T$ to $i+T+H-1$.
4.  **Data Aggregation:** Collects the extracted static, dynamic, future,
    and target sequences from all windows and groups into separate lists.
5.  **Output Conversion:** Converts the lists into NumPy arrays.
    The function returns a tuple:
    `(static_data, dynamic_data, future_data, target_data)`. Static and
    future data arrays will be `None` if the corresponding columns are not
    provided.
6.  **Saving (Optional):** If `savefile` is provided, saves the processed
    arrays and feature names to a file using `joblib`.

**Mathematical Concept (Rolling Window):**
The core idea is creating pairs of input ($\mathbf{X}^{(i)}$) and target
($\mathbf{Y}^{(i)}$) sequences. For a window starting at index $i$:

.. math::
   \mathbf{X}^{(i)} =
   \begin{bmatrix}
      \mathbf{features}_{i} \\
      \mathbf{features}_{i+1} \\
      \vdots \\
      \mathbf{features}_{i+T-1}
   \end{bmatrix}
   \quad , \quad
   \mathbf{Y}^{(i)} =
   \begin{bmatrix}
      \text{target}_{i+T} \\
      \text{target}_{i+T+1} \\
      \vdots \\
      \text{target}_{i+T+H-1}
   \end{bmatrix}

where $\mathbf{features}_t$ includes the relevant static (repeated),
dynamic, and future features for time $t$.

**Usage Context:** This function is designed to be a primary tool for
preparing complete datasets for training or evaluating TFT/XTFT models
directly from a Pandas DataFrame. It handles the complexities of
sequence generation, feature type separation, and optional spatial
grouping.


.. raw:: html

    <hr>
    
Forecasting & Visualization
---------------------------

These functions assist with generating predictions from trained models
and visualizing the forecast results.

.. _generate_forecast:

generate_forecast
~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.generate_forecast`

**Purpose:** To generate future predictions using a pre-trained
``fusionlab`` model (like XTFT or TFT). This function acts as a
high-level wrapper that handles preparing the necessary model inputs
from the end of the provided training data and formats the model's
output into a structured DataFrame.

**Functionality:**
1.  **Model Validation:** Ensures the provided `xtft_model` is a
    valid, trained Keras model.
2.  **Input Preparation:**
    * Optionally groups `train_data` by `spatial_cols`.
    * For each group (or the entire dataset):
        * Extracts the *last* sequence of length `time_steps` from
            the sorted `train_data`.
        * Constructs the required input arrays for the model's
            `.predict()` method:
            * `X_static`: Uses static features from the last record.
                (Uses zeros if `static_features` not provided).
            * `X_dynamic`: Uses dynamic features from the last
                `time_steps` records.
            * `X_future`: Uses future features. The current implementation
                takes values from the *first* record of the last
                sequence and tiles them across `time_steps`. (Uses zeros
                if `future_features` not provided).
3.  **Prediction:** Calls `xtft_model.predict()` with the prepared
    `[X_static, X_dynamic, X_future]` arrays for each group.
    Conceptually:

    .. math::
       \hat{y}_{t+1...t+H} = f_{model}(X_{\text{static}}, X_{\text{dynamic}}, X_{\text{future}})

    where $H$ is the `forecast_horizon`.
4.  **Output Formatting:**
    * Determines the future dates/periods (`forecast_dt`), inferring
        automatically if set to `"auto"`.
    * Organizes the raw predictions into a Pandas DataFrame.
    * Includes spatial identifiers (if used) and the corresponding
        forecast date/period for each prediction.
    * Creates columns for point predictions (`<tname>_pred`) or
        quantile predictions (`<tname>_qXX`) based on the `mode` and
        `q` parameters.
5.  **Evaluation (Optional):** If `test_data` is provided, it aligns
    the forecasts with the actual values based on dates and spatial
    columns (if applicable) for the common periods within the
    `forecast_horizon`. It then calculates and prints the R² score
    (comparing actuals to median/point forecast) and, if in
    `'quantile'` mode, the coverage score (using the lowest and
    highest specified quantiles).
6.  **Saving (Optional):** Saves the resulting forecast DataFrame to
    a CSV file if `savefile` is specified.

**Usage Context:** This is the primary function to use after training
a model to generate out-of-sample forecasts. It simplifies the process
of preparing the specific inputs needed for prediction directly from
the training dataset and provides the results in an easily usable
DataFrame format, optionally including basic evaluation metrics if
test data is available.

.. _visualize_forecasts:

visualize_forecasts
~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.visualize_forecasts`

**Purpose:** To create visualizations comparing forecasted values
against actual values (if available), particularly useful for spatial
data or analyzing performance across different time periods.

**Functionality:**
1.  **Data Filtering:** Selects data from `forecast_df` and optional
    `test_data` corresponding to the specified `eval_periods` (or
    infers up to 3 periods if `eval_periods` is `None`). Ensures only
    common periods present in both forecast and test data (if provided)
    are used.
2.  **Column Identification:** Determines the column names for predicted
    values (e.g., `<tname>_q50` or `<tname>_pred` based on `mode`)
    and actual values (`tname` or `actual_name` in `test_data`).
    Identifies coordinate columns (`x`, `y`) based on `kind` ('spatial'
    defaults to 'longitude'/'latitude').
3.  **Plot Grid Setup:** Creates a `matplotlib` subplot grid. The size
    depends on the number of `eval_periods` and `max_cols`. If
    `test_data` is provided, it creates two plots per period (Actual
    vs. Predicted); otherwise, just one plot (Predicted).
4.  **Plotting:** For each evaluation period:
    * Creates a scatter plot of actual values (if `test_data` given),
        coloring points by the actual value.
    * Creates a scatter plot of predicted values, coloring points by
        the predicted value.
    * Uses `x` and `y` columns for plot coordinates.
    * Applies a consistent color map (`cmap`) and value range (`vmin`,
        `vmax`) across all plots for comparability.
    * Adds titles indicating the period and whether it's actual or
        predicted data, labels, optional grid, and color bars.
5.  **Display:** Shows the generated `matplotlib` figure.

**Usage Context:** Use this function after generating forecasts (e.g.,
using :func:`generate_forecast`) to visually assess the model's
performance. It's particularly helpful for:
* Comparing predicted patterns to actual patterns spatially.
* Observing how forecast accuracy changes over different periods.
* Checking the spread and median of quantile forecasts against actuals.


.. _forecast_single_step:

forecast_single_step
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.forecast_single_step`

**Purpose:** To generate a forecast for only the *next single time step*
using a pre-trained ``fusionlab`` model and pre-prepared input arrays.

**Functionality:**
1.  **Input:** Takes a validated Keras `xtft_model` and a list/tuple
    `inputs` containing the NumPy arrays `[X_static, X_dynamic, X_future]`
    ready for prediction.
2.  **Prediction:** Calls `xtft_model.predict(inputs)` to get the raw
    model output. It assumes the prediction for the first step ahead
    is the relevant one.

    .. math::
       \hat{y}_{t+1} = f_{model}(X_{\text{static}}, X_{\text{dynamic}}, X_{\text{future}})_{step=1}

3.  **Output Formatting:** Creates a Pandas DataFrame containing the
    predictions.
    * Includes spatial columns (e.g., longitude, latitude) if
        `spatial_cols` are provided (extracted from the first columns
        of `X_static`).
    * Optionally adds a `dt_col` column (values need external context).
    * Optionally adds actual target values (`y`) if provided.
    * Adds prediction columns:
        * **Quantile Mode:** `<tname>_qXX` for each quantile `q`.
        * **Point Mode:** `<tname>_pred`.
4.  **Masking (Optional):** If `apply_mask=True`, uses the provided
    `mask_values` in the actual target column (`y`) to mask corresponding
    predictions (setting them to `mask_fill_value`). Requires `y`.
5.  **Evaluation (Optional):** If actual target values `y` are provided,
    computes and prints the R² score and (for quantile mode) the
    coverage score between `y` and the relevant prediction column(s).
6.  **Saving (Optional):** Saves the resulting DataFrame to a CSV file
    if `savefile` is specified.

**Usage Context:** Use this function when you have already prepared the
specific `X_static`, `X_dynamic`, and `X_future` input arrays needed
to predict the immediate next time step for a batch of samples/locations.
This is useful for direct prediction tasks where the input preparation
is handled separately, unlike :func:`generate_forecast` which prepares
inputs from historical data internally.

.. _forecast_multi_step:

forecast_multi_step
~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.forecast_multi_step`

**Purpose:** To generate forecasts for *multiple future time steps*
(up to a specified `forecast_horizon`) using a pre-trained
``fusionlab`` model and pre-prepared input arrays.

**Functionality:**
1.  **Input:** Similar to `forecast_single_step`, takes a validated
    `xtft_model` and `inputs = [X_static, X_dynamic, X_future]`. Requires
    `forecast_horizon` to be specified.
2.  **Prediction:** Calls `xtft_model.predict(inputs)`. The model is
    expected to output predictions for the entire horizon, typically
    with shape `(Batch, Horizon, NumOutputs)`.
3.  **Initial Output Formatting (Wide):** Iterates through each sample
    and each forecast step `i` (from 1 to `forecast_horizon`). Creates
    a wide-format DataFrame where each row corresponds to a sample,
    and columns represent predictions for specific steps and quantiles
    (e.g., `<tname>_q10_step1`, `<tname>_q50_step1`,
    `<tname>_q10_step2`, etc.) or points (`<tname>_pred_step1`,
    `<tname>_pred_step2`, etc.). Includes spatial columns and optional
    `dt_col` placeholders if specified. Actual values (`y`) are also
    added if provided. A `BatchDataFrameBuilder` is used internally
    for memory efficiency with large numbers of samples.
4.  **Reshaping to Long Format:** Calls the internal utility `step_to_long`
    to likely transform the wide-format DataFrame into a long format,
    where each row represents a single prediction for a specific sample,
    time step, and potentially quantile. *(Note: The exact output format
    depends on the implementation of `step_to_long`, but long format is
    common for multi-step results).*
5.  **Masking (Optional):** If `apply_mask=True`, masks predictions based
    on `mask_values` in the actual target column (`y`). Requires `y`.
6.  **Evaluation (Optional):** If actual target values `y` are provided,
    computes and prints R² and Coverage Scores, comparing predictions
    against actuals across *all* available forecast steps (up to
    `forecast_horizon` or the length of `y`).
7.  **Saving (Optional):** Saves the final (likely long-format)
    DataFrame to a CSV file if `savefile` is specified.

**Usage Context:** Use when you need predictions spanning multiple time
steps ahead, based on a specific set of prepared input arrays. Like
`forecast_single_step`, it assumes input preparation is done externally.
It handles the complexity of organizing and potentially reshaping
multi-step outputs from the model.

.. _generate_forecast_with:

generate_forecast_with
~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.generate_forecast_with`

**Purpose:** A convenient wrapper function that automatically calls
either :func:`forecast_single_step` or :func:`forecast_multi_step`
based on the specified `forecast_horizon`.

**Functionality:**
1.  Takes all the same arguments as `forecast_single_step` and
    `forecast_multi_step` (including the `xtft_model`, prepared
    `inputs`, `forecast_horizon`, etc.).
2.  Checks the value of `forecast_horizon`:
    * If `forecast_horizon == 1`, it internally calls
      :func:`forecast_single_step`, passing along all the other
      arguments.
    * If `forecast_horizon > 1`, it internally calls
      :func:`forecast_multi_step`, passing along all the other
      arguments.
3.  Returns the DataFrame produced by the called function (either
    single-step or multi-step results).

**Usage Context:** This function provides a unified interface for
generating forecasts when using pre-prepared input arrays. Instead of
manually choosing between the single-step and multi-step functions,
users can simply call `generate_forecast_with` and let it dispatch
the task based on the desired `forecast_horizon`. This can simplify
workflows where the forecast length might vary.


.. raw:: html

    <hr>
    
Data Reshaping Utilities
------------------------

.. _step_to_long:

step_to_long
~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.nn.utils.step_to_long`

**Purpose:** To transform a DataFrame containing multi-step forecast
results from a "wide" format into a "long" format. In the wide
format, each forecast step typically occupies separate columns (e.g.,
`target_q50_step1`, `target_q50_step2`). The long format reshapes
this so that each row represents a single prediction for a specific
sample, time step, and possibly quantile.

**Functionality:**
1.  Takes a wide-format DataFrame `df` as input, along with metadata
    like `tname`, `dt_col`, `spatial_cols`, and `mode` ('quantile' or
    'point').
2.  Identifies the columns corresponding to different forecast steps
    and quantiles based on naming conventions (e.g., `_stepX`, `_qYY`).
3.  Uses internal helper functions (`_step_to_long_q` for quantile,
    `_step_to_long_pred` for point) which likely employ Pandas
    melting or stacking operations.
4.  Reshapes the data, creating new columns for the forecast step
    (e.g., 'step') and quantile (e.g., 'quantile'), and consolidating
    the prediction values into a single column (e.g., 'predicted_value').
5.  Identifier columns (`dt_col`, `spatial_cols`, actual values if
    present) are typically preserved and duplicated across the reshaped
    rows.
6.  Optionally sorts the final long-format DataFrame.

**Usage Context:** This function is primarily used as an internal
helper within :func:`forecast_multi_step` to convert the initially
generated wide-format predictions into a more standardized long format.
Users might also find it useful if they have wide-format forecast data
from other sources and want to reshape it for easier plotting or analysis.

