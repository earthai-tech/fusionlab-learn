.. _user_guide_ts_utils:

=======================
Time Series Utilities
=======================

The ``fusionlab.utils.ts_utils`` module provides a collection of
utility functions designed to facilitate common time series data
manipulation, analysis, and preprocessing tasks. These functions can
be helpful when preparing data for use with ``fusionlab``'s
forecasting models or for general time series analysis workflows.

Datetime Handling & Filtering
-----------------------------

These utilities focus on converting, validating, and filtering time
series data based on its datetime index or columns.

.. _filter_by_period:

filter_by_period
~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.filter_by_period`

**Purpose:** To filter rows in a Pandas DataFrame based on whether their
datetime values fall within specified evaluation periods (e.g.,
specific years, months, days, weeks).

**Functionality:**
1.  **Datetime Validation:** Ensures the specified `dt_col` (or the
    DataFrame index if `dt_col` is None) is converted to a proper
    Pandas datetime format using the internal :func:`ts_validator`.
2.  **Period Granularity Detection:** Automatically determines the
    granularity of the filter based on the format of the strings
    provided in `eval_periods` (e.g., 'YYYY' for year, 'YYYY-MM' for
    month, 'YYYY-MM-DD' for day, 'YYYY-Www' for week, etc.).
3.  **Filtering:** Formats the datetime column of the DataFrame to
    match the detected granularity and then uses the Pandas `.isin()`
    method to select only those rows where the formatted datetime
    matches one of the periods specified in `eval_periods`.

    Conceptually:

    .. math::
       filtered\_df = df[\text{format}(dt_{col}).isin(\text{eval\_periods})]

    where `format` depends on the detected granularity.

**Usage Context:** Useful for selecting specific subsets of your time
series data corresponding to particular years, months, weeks, or days.
This can be valuable for creating training/validation/test splits based
on time periods, or for analyzing model performance during specific
intervals.

.. _to_dt:

to_dt
~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.to_dt`

**Purpose:** To robustly convert a specific column or the index of a
Pandas DataFrame into the standard Pandas datetime format. It includes
special handling for columns/indices that might store date information
as integers (e.g., year as `2023`).

**Functionality:**
1.  Takes an input DataFrame `df` and an optional `dt_col` name. If
    `dt_col` is `None`, the function targets the DataFrame's index.
2.  Uses :func:`pandas.to_datetime` for the core conversion logic,
    allowing passthrough of additional arguments (`format`, `error`,
    etc.).
3.  **Integer Handling:** If the target column or index is detected as
    having an integer dtype, it is first explicitly converted to a
    string type before being passed to `pd.to_datetime`. This allows
    correct parsing of integer years or other integer date formats
    when an appropriate `format` string is provided (or sometimes
    inferred).
4.  **Error Handling:** Manages conversion errors based on the `error`
    parameter ('raise', 'warn', 'ignore').
5.  **Return Value:** Returns the modified DataFrame. If
    `return_dt_col=True`, it also returns the name of the processed
    column (or `None` if the index was converted).

**Usage Context:** An essential utility for ensuring that your time-related
columns or indices are in the proper `datetime64[ns]` dtype required
by most Pandas time series operations and many other functions within
``fusionlab``. Use it early in your data preprocessing pipeline to
standardize date/time representations.

ts_split
~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.ts_split`

**Purpose:** To split time series data into training and testing sets
while respecting chronological order, or to generate time-series-aware
cross-validation splits. This prevents lookahead bias, where future
information inadvertently leaks into the training process.

**Functionality:**
Takes a DataFrame `df` and parameters controlling the split type.

* **`split_type='simple'` (or `'base'`)**: Performs a single
    chronological split into two sets: `train_df` and `test_df`.
    * **Date-Based:** If `train_start` and/or `train_end` are given,
        the split point is determined by these dates.
    * **Ratio-Based:** If `test_ratio` (e.g., 0.2 for 20%) is given,
        the last fraction of the data becomes the test set.
        Conceptually, splits at index $k = N \times (1 - \text{test_ratio})$:

        .. math::
           \text{Train} = \{X_t | t \le k \}, \quad \text{Test} = \{X_t | t > k \}

    * Returns `(train_df, test_df)`.

* **`split_type='cv'`**: Creates time series cross-validation splits
    using `sklearn.model_selection.TimeSeriesSplit`.
    * Generates `n_splits` pairs of (train_indices, test_indices).
    * Each split consists of a training set containing earlier data
        and a test set containing later data. Subsequent folds use
        progressively larger training sets (expanding window).
    * A `gap` can be introduced between the train and test sets in
        each fold.
    * Returns a generator yielding `(train_indices, test_indices)` for
        each split.

**Usage Context:** Essential for properly evaluating time series models.
Use `'simple'` for creating a single hold-out test set. Use `'cv'` for
more robust model evaluation and hyperparameter tuning using time
series cross-validation, ensuring that the model is always tested on
data that comes after its training data within each fold. Requires
`scikit-learn` for the 'cv' option.

.. _ts_outlier_detector:

ts_outlier_detector
~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.ts_outlier_detector`

**Purpose:** To identify potential outliers within a specified time
series column (`value_col`) using standard statistical methods.
Optionally removes detected outliers.

**Functionality:**
Uses one of two common methods based on the `method` parameter:

* **`method='zscore'`:** Calculates the Z-score for each data point
    $X_t$ relative to the series mean ($\mu$) and standard
    deviation ($\sigma$):

    .. math::
       Z_t = \frac{X_t - \mu}{\sigma}

    Points where $|Z_t|$ exceeds the specified `threshold` (default 3)
    are flagged as outliers. This method assumes data is approximately
    normally distributed.

* **`method='iqr'`:** Uses the Interquartile Range (IQR = Q3 - Q1).
    Calculates outlier bounds:
    * Lower Bound = $Q1 - threshold \times IQR$
    * Upper Bound = $Q3 + threshold \times IQR$
    Points falling outside these bounds are flagged as outliers. The
    `threshold` (default 1.5 in typical IQR definitions, but customizable
    here) controls sensitivity. This method is less sensitive to extreme
    values than the Z-score method.

The function adds a boolean column `'is_outlier'` to the input
DataFrame `df`. If `drop=True`, it removes the rows marked as
outliers and does *not* add the `'is_outlier'` column. If `view=True`,
it displays a plot of the time series with detected outliers highlighted.

**Usage Context:** Can be used as a data cleaning step before analysis
or modeling to identify or remove data points that might unduly
influence results (e.g., measurement errors, anomalous events). The
choice between 'zscore' and 'iqr' depends on the expected distribution
of the data. Requires `scipy` for Z-score calculation.

.. raw:: html

    <hr>

Trend & Seasonality Analysis
----------------------------

These utilities help in analyzing, transforming, and visualizing
trends and seasonal patterns within time series data, often leveraging
the `statsmodels` library.

.. _trend_analysis:

trend_analysis
~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.trend_analysis`

**Purpose:** To perform a basic analysis of a time series to identify
its overall trend direction (upward, downward, or stationary) and
optionally assess its stationarity using statistical tests.

**Functionality:**
1.  **Stationarity Test (Optional):** If `check_stationarity=True`,
    it performs either the Augmented Dickey-Fuller (ADF, `strategy='adf'`)
    or KPSS (`strategy='kpss'`) test on the `value_col`.
    * ADF Test: Null Hypothesis = Non-stationary. A low p-value (< 0.05)
        suggests stationarity.
    * KPSS Test: Null Hypothesis = Stationary around a constant (or
        linear trend). A low p-value (< 0.05) suggests non-stationarity.
    Based on the test result, an initial trend status ('stationary' or
    'non-stationary') is determined.
2.  **Linear Trend Fitting:** If the series is initially deemed
    non-stationary (or if `trend_type='both'`), it fits a simple
    Ordinary Least Squares (OLS) linear regression model:

    .. math::
       y_t = \beta_0 + \beta_1 \cdot t + \epsilon_t

    where $y_t$ is the value at time $t$, and $t$ is a simple time
    index (0, 1, 2,...).
3.  **Trend Classification:** The final trend classification is based
    on the slope $\beta_1$ of the fitted line, potentially overriding
    the initial stationarity test result if `trend_type='both'`:
    * `'upward'`: if $\beta_1 > 0$.
    * `'downward'`: if $\beta_1 < 0$.
    * `'stationary'`: if $\beta_1 \approx 0$ or if the initial
        stationarity test indicated stationarity and no conflicting
        trend was found (depending on `trend_type`).
4.  **Visualization (Optional):** If `view=True`, displays a plot
    of the time series, annotated with the detected trend and the
    p-value from the stationarity test (if performed). A mean line
    is shown for stationary series, and the fitted OLS line for
    trending series.

**Usage Context:** A useful first step in time series EDA to get a
quick assessment of stationarity and the dominant linear trend. The
results can guide subsequent preprocessing steps, such as detrending
or differencing using :func:`trend_ops` or
:func:`transform_stationarity`. Requires `statsmodels`.

.. _trend_ops:

trend_ops
~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.trend_ops`

**Purpose:** To apply specific transformations to a time series aimed
at removing or mitigating trends, based on an automatic trend analysis.

**Functionality:**
1.  **Trend Detection:** Internally calls :func:`trend_analysis` to
    determine the trend ('upward', 'downward', 'stationary') of the
    `value_col`, optionally using stationarity tests (`check_stationarity`).
2.  **Transformation Application:** Based on the detected `trend` and
    the user-specified operation `ops`:
    * `ops='remove_upward'/'remove_downward'/'remove_both'`: If the
        corresponding trend is detected, fits a linear OLS trend
        $\hat{Y}_t = \beta_0 + \beta_1 \cdot t$ and subtracts it from
        the original series $Y_t$: $Y'_{t} = Y_t - \hat{Y}_t$.
    * `ops='detrend'`: If the series is detected as 'non-stationary',
        applies first-order differencing: $\nabla Y_t = Y_t - Y_{t-1}$.
        The first value becomes NaN.
    * `ops='none'`: No transformation is applied.
3.  **DataFrame Update:** Replaces the original `value_col` in the
    DataFrame with the transformed data (or leaves it unchanged if
    `ops='none'` or no relevant trend was detected for removal).
4.  **Visualization (Optional):** If `view=True`, displays side-by-side
    plots of the original and transformed time series.

**Usage Context:** Use this function to automate the process of making
a time series (more) stationary by removing linear trends or applying
differencing, based on preliminary analysis. This is often a necessary
preprocessing step for classical time series models like ARIMA.
Requires `statsmodels`.

.. _visual_inspection:

visual_inspection
~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.visual_inspection`

**Purpose:** To generate a comprehensive set of diagnostic plots for
visually exploring the characteristics of a time series, including
trend, seasonality, and autocorrelation.

**Functionality:**
Creates a grid of `matplotlib` plots displaying:

1.  **Original Time Series:** The raw `value_col` data over time.
2.  **Rolling Mean (Trend):** If `show_trend=True`, plots the rolling
    mean calculated over a specified `window`. This helps visualize the
    underlying trend.
    .. math::
       \text{RollingMean}_t = \frac{1}{W}\sum_{i=0}^{W-1} X_{t-i}
3.  **Rolling Standard Deviation (Seasonality/Volatility):** If
    `show_seasonal=True`, plots the rolling standard deviation over
    the `window`. This can indicate periods of changing volatility or
    provide clues about seasonality.
4.  **Autocorrelation Function (ACF):** If `show_acf=True`, plots the
    ACF using `statsmodels.graphics.tsaplots.plot_acf` up to a
    specified number of `lags`. This helps identify correlation between
    a data point and its past values.
5.  **Seasonal Decomposition (Optional):** If `show_decomposition=True`
    and a valid `seasonal_period` is provided, performs classical
    decomposition (e.g., additive) using
    `statsmodels.tsa.seasonal.seasonal_decompose` and plots the
    observed, trend, seasonal, and residual components. Can be plotted
    in the main grid or a separate figure (`decompose_on_sep=True`).
    The residual component can also be plotted individually if
    `show_residual=True`.

**Usage Context:** An essential tool for Exploratory Data Analysis (EDA)
of time series data. It provides quick visual insights into key
properties like trend, potential seasonality, autocorrelation structure,
and stationarity, helping to inform subsequent modeling choices and
preprocessing steps. Requires `statsmodels` and `matplotlib`.


.. _get_decomposition_method:

get_decomposition_method
~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.get_decomposition_method`

**Purpose:** To provide a *heuristic* estimate of a suitable
decomposition model type ('additive' or 'multiplicative') and a
basic guess for the seasonal period for a time series.

**Functionality:**
1.  Takes a DataFrame `df`, the `value_col`, and optional `dt_col`.
2.  **Method Inference:**
    * If `method='auto'`, it checks if all values in `value_col` are
        strictly positive (> 0). If yes, it suggests `'multiplicative'`,
        assuming multiplicative effects are more likely with positive data
        where seasonality might scale with the trend. Otherwise, it
        suggests `'additive'`.
    * If `method` is explicitly set to `'additive'` or `'multiplicative'`,
        it returns the specified method.
3.  **Period Inference:** The current implementation uses a basic
    placeholder logic. It returns a default period (e.g., 1 or
    `min_period`), rather than performing sophisticated analysis like
    autocorrelation or spectral analysis to detect the actual period.
    *(Note: This period inference is very simplistic in the current
    code version).*

**Usage Context:** This function offers a very quick, rule-based first
guess for decomposition parameters, primarily distinguishing between
additive and multiplicative based on data positivity. Due to the basic
period inference, its utility for determining the correct seasonal
period is limited in its current form. It might be used in automated
pipelines where a rough initial setting is needed before more rigorous
analysis or decomposition. Requires `statsmodels`.

.. _infer_decomposition_method:

infer_decomposition_method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.infer_decomposition_method`

**Purpose:** To determine the more appropriate decomposition method
('additive' or 'multiplicative') for a time series using either a
positivity heuristic or by comparing the variance of residuals from
both decomposition types.

**Functionality:**
Takes a DataFrame `df` (using `dt_col` as index), the first column
as the `value_col`, a required seasonal `period`, and the inference
`method`.

1.  **`method='heuristic'`:** Checks if all values in the time series
    are strictly positive (> 0). Returns `'multiplicative'` if true,
    `'additive'` otherwise. This does not perform actual decomposition.
2.  **`method='variance_comparison'`:**
    * Performs *both* additive and multiplicative classical seasonal
        decomposition using `statsmodels.tsa.seasonal.seasonal_decompose`
        with the specified `period`.
    * Calculates the variance of the residual component ($\epsilon_t$)
        from both decompositions.
    * Selects and returns the method ('additive' or 'multiplicative')
        that resulted in the *lower* residual variance, assuming that
        model provides a better fit (less unexplained variation).
    * If `view=True`, displays histograms of the residuals from both
        methods for visual comparison.
    * If `return_components=True`, also returns the trend, seasonal,
        and residual series from the chosen best-fitting decomposition.

**Usage Context:** Provides a more data-driven way (via variance
comparison) to choose between additive and multiplicative decomposition
models compared to simple heuristics, assuming the provided `period` is
correct. Useful when deciding which model form better captures the
structure of the time series before performing the final decomposition
with :func:`decompose_ts`. Requires `statsmodels`.

.. _decompose_ts:

decompose_ts
~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.decompose_ts`

**Purpose:** To perform time series decomposition, explicitly separating
a given time series (`value_col`) into its constituent Trend ($T_t$),
Seasonal ($S_t$), and Residual ($R_t$) components using methods from
`statsmodels`.

**Functionality:**
1.  Takes a DataFrame `df`, the `value_col` to decompose, optional
    `dt_col`, the decomposition `method` ('additive' or
    'multiplicative'), the decomposition `strategy` ('STL' or 'SDT'),
    the `seasonal_period`, and a `robust` flag (for STL).
2.  **Selects Algorithm:**
    * `strategy='STL'`: Uses `statsmodels.tsa.seasonal.STL` (Seasonal-
        Trend decomposition using LOESS). This method is generally more
        flexible and can be made robust to outliers (`robust=True`). Requires
        `seasonal_period` to be odd and >= 3 (adjusted automatically if
        even). Cannot perform multiplicative decomposition directly.
    * `strategy='SDT'`: Uses the classical decomposition method
        `statsmodels.tsa.seasonal.seasonal_decompose`, supporting both
        `method='additive'` and `method='multiplicative'`.
3.  **Performs Decomposition:** Applies the chosen algorithm to the
    `value_col` data using the specified `seasonal_period`.
4.  **Returns Augmented DataFrame:** Creates a new DataFrame containing
    the calculated `trend`, `seasonal`, and `residual` components as
    new columns. It also includes the original `value_col` and all
    other columns from the input `df`, preserving the original index.

**Mathematical Models:**
* Additive: $Y_t = T_t + S_t + R_t$
* Multiplicative: $Y_t = T_t \times S_t \times R_t$

**Usage Context:** This function is used when you need to explicitly
extract and analyze the underlying components of a time series. The
resulting trend component can show long-term direction, the seasonal
component reveals periodic patterns, and the residual represents the
irregular noise or unexplained part. These components can be analyzed
individually, forecasted separately, or used as features for other
models. Requires `statsmodels`.

.. _transform_stationarity:

transform_stationarity
~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.transform_stationarity`

**Purpose:** To apply common transformations to a time series aimed at
achieving or improving stationarity (i.e., stabilizing the mean,
variance, and autocorrelation structure over time).

**Functionality:**
Applies a transformation to the `value_col` based on the `method`:

* **`'differencing'`:** Computes the difference between consecutive
    observations. Can be applied multiple times (`order` parameter)
    or seasonally (`seasonal_period` parameter).
    * First Difference: $\nabla Y_t = Y_t - Y_{t-1}$
    * Seasonal Difference: $\nabla_S Y_t = Y_t - Y_{t-S}$
* **`'log'`:** Applies the natural logarithm. Useful for stabilizing
    variance when it increases with the level of the series. Requires
    all values to be positive.
    .. math:: Y'_t = \ln(Y_t)
* **`'sqrt'`:** Applies the square root transform. Also used for
    variance stabilization. Requires non-negative values.
    .. math:: Y'_t = \sqrt{Y_t}
* **`'detrending'`:** Removes a trend component.
    * `detrend_method='linear'`: Fits a linear OLS trend
        ($\hat{Y}_t = \beta_0 + \beta_1 t$) and subtracts it:
        $Y'_t = Y_t - \hat{Y}_t$.
    * `detrend_method='stl'`: Performs STL decomposition and returns
        the residual component ($R_t$) as the transformed series.
        Requires `statsmodels`.

The function adds the transformed series as a new column
`'<value_col>_transformed'`. If `drop_original=True` (default), the
original `value_col` is removed. Optionally plots the original vs.
transformed series (`view=True`).

**Usage Context:** A crucial preprocessing step for many classical time
series models (like ARIMA) that assume stationarity. Differencing is
common for removing trends and seasonality. Log/sqrt transforms address
heteroscedasticity (non-constant variance). Detrending provides
alternative ways to remove trend components. Requires `statsmodels` if
using `'stl'` detrending.

.. _ts_corr_analysis:

ts_corr_analysis
~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.ts_corr_analysis`

**Purpose:** To analyze and visualize the correlation structure of a
time series, including its relationship with its own past values
(autocorrelation) and with other external features (cross-correlation).

**Functionality:**
1.  **Inputs:** Takes a DataFrame `df`, the target `value_col`, the
    datetime column `dt_col`, number of `lags` for ACF/PACF, and an
    optional list of external `features`.
2.  **Autocorrelation (ACF/PACF):** If `view_acf_pacf=True`, it plots:
    * **ACF Plot:** Shows the correlation of the series with its
        lagged values ($\rho(h)$ vs $h$). Helps identify Moving
        Average (MA) order in ARIMA models.
    * **PACF Plot:** Shows the correlation of the series with its
        lagged values after removing the effects of intermediate lags.
        Helps identify Autoregressive (AR) order.
    Plots are generated using `statsmodels`. The function currently
    returns placeholder `None` for ACF/PACF values, relying on the
    visualization.

    .. math::
       \rho(h) = \frac{Cov(Y_t, Y_{t-h})}{\sqrt{Var(Y_t)Var(Y_{t-h})}}

3.  **Cross-Correlation:** If `features` are provided (or inferred), it
    calculates the Pearson correlation coefficient (zero-lag) between
    the `value_col` and each specified external `feature` using
    `scipy.stats.pearsonr`. This measures the linear relationship
    between the target and potential predictors at the same time step.
4.  **Visualization:** Besides ACF/PACF plots, if `view_cross_corr=True`,
    it displays a bar chart visualizing the cross-correlation
    coefficients between the target and external features. This plot
    can be combined with the ACF/PACF plots or shown separately
    (`cross_corr_on_sep=True`).
5.  **Output:** Returns a dictionary containing the calculated
    cross-correlation coefficients and p-values for each external feature.

**Usage Context:** An important EDA tool. ACF/PACF plots help understand
the internal memory or persistence of the time series. Cross-correlation
analysis helps identify potentially relevant exogenous variables (static,
dynamic past, or future known inputs) that could be included as
predictors in a forecasting model. Requires `statsmodels`, `scipy`,
and `matplotlib`.

.. raw:: html

    <hr>
    
Feature Engineering
-------------------

These utilities focus on creating new features from time series data
that can be beneficial for machine learning models.

.. _ts_engineering:

ts_engineering
~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.ts_engineering`

**Purpose:** To automatically generate a variety of common and useful
time series features from a DataFrame, augmenting it with predictors
that capture temporal dynamics, seasonality, and other patterns.

**Functionality:**
Takes a DataFrame `df` (with a datetime index or `dt_col`), the primary
`value_col`, and various parameters controlling feature generation:

1.  **Time-Based Features:** Extracts standard calendar features from
    the datetime index: 'year', 'month', 'day', 'day_of_week',
    'is_weekend', 'quarter', 'hour' (if applicable).
2.  **Holiday Indicator:** Creates a binary 'is_holiday' feature if a
    `holiday_df` (containing holiday dates) is provided.
3.  **Lag Features:** Creates `lags` number of lag features by shifting
    the `value_col` (e.g., `lag_1` is $Y_{t-1}$, `lag_2` is $Y_{t-2}$).
4.  **Rolling Statistics:** Calculates rolling mean and standard
    deviation of the `value_col` over a specified `window` size ($W$).

    .. math::
       \text{RollingMean}_t = \frac{1}{W}\sum_{i=0}^{W-1} Y_{t-i}
       \; ; \;
       \text{RollingStd}_t = \sqrt{\frac{1}{W-1}\sum_{i=0}^{W-1} (Y_{t-i} - \text{RollingMean}_t)^2}

5.  **Differencing:** Creates a differenced series of order `diff_order`.
    For order 1: $\nabla Y_t = Y_t - Y_{t-1}$.
6.  **Seasonal Differencing:** If `seasonal_period` ($S$) is provided,
    creates a seasonally differenced series: $Y_t - Y_{t-S}$.
7.  **Fourier Features (Optional):** If `apply_fourier=True`, computes
    the FFT of the `value_col` and adds the magnitudes of the frequency
    components as features ('fft_1', 'fft_2', ...). Useful for
    capturing complex periodicities.
8.  **Missing Value Handling:** Fills NaNs resulting from lags,
    differencing, or rolling windows using forward fill (`ffill`), then
    drops any remaining rows with NaNs.
9.  **Scaling (Optional):** If `scaler` is specified ('z-norm' or
    'minmax'), applies StandardScaler or MinMaxScaler to all numeric
    columns in the augmented DataFrame.

**Usage Context:** A powerful utility for automating the creation of a
rich feature set for time series forecasting models. It generates
features commonly found effective in capturing trend, seasonality,
autocorrelation, and calendar effects. The resulting DataFrame can be
used directly for training simpler models or serve as input for further
processing steps like :func:`create_sequences` or
:func:`reshape_xtft_data`.


.. _create_lag_features:

create_lag_features
~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.create_lag_features`

**Purpose:** To generate lagged features for one or more time series
columns in a DataFrame. Lag features represent the values of a series
at previous time steps and are fundamental predictors for many time
series models.

**Functionality:**
1.  Takes a DataFrame `df`, the primary `value_col`, optional `dt_col`,
    an optional list of additional `lag_features` (if None, only uses
    `value_col`), and a list of integer `lags`.
2.  Ensures the DataFrame has a datetime index using `ts_validator`.
3.  For each specified `feature` (from `value_col` and `lag_features`)
    and for each lag interval $k$ in the `lags` list:
    * Creates a new column named `<feature>_lag_<k>`.
    * Populates this column by shifting the original `feature` column
        down by $k$ steps using `df[feature].shift(k)`. This aligns
        the value from time $t-k$ with the row for time $t$.

    .. math::
       \text{Feature}_{lag\_k}(t) = \text{Feature}(t-k)

4.  Optionally concatenates the original columns back with the new lag
    columns (`include_original=True`).
5.  Optionally drops rows containing NaN values introduced by the shift
    operation (`dropna=True`). This typically removes the first
    `max(lags)` rows.
6.  Optionally resets the index if it was modified (`reset_index=True`).

**Usage Context:** A core feature engineering step for time series.
Generating lags allows models (from simple linear regression to complex
neural networks) to learn autoregressive patterns, i.e., how past values
influence future values. This function provides a convenient way to
create multiple lags for multiple features simultaneously.

.. raw:: html

    <hr>
    
Feature Selection & Reduction
-----------------------------

After potentially generating many features (e.g., via lags, rolling
stats, etc.), these utilities can help select the most relevant ones
or reduce the dimensionality of the feature space.

.. _select_and_reduce_features:

select_and_reduce_features
~~~~~~~~~~~~~~~~~~~~~~~~~~
:API Reference: :func:`~fusionlab.utils.ts_utils.select_and_reduce_features`

**Purpose:** To perform feature selection by removing highly correlated
features or to reduce dimensionality using Principal Component Analysis
(PCA).

**Functionality:**
Takes a DataFrame `df` and optional `target_col` / `exclude_cols`
(which are ignored during processing but can be re-attached). Operates
based on the `method` parameter:

* **`method='corr'` (or `'correlation')`:**
    1. Calculates the pairwise Pearson correlation matrix for all
        included numeric features.
    2. Identifies pairs of features whose absolute correlation exceeds
        `corr_threshold`.
    3. Systematically drops one feature from each highly correlated pair
        to reduce multicollinearity. (Specifically, drops columns where
        any value in the upper triangle of the absolute correlation
        matrix exceeds the threshold).
* **`method='pca'`:**
    1. Optionally standardizes the features using
        `sklearn.preprocessing.StandardScaler` (`scale_data=True`).
        Standardization is generally recommended for PCA.
    2. Applies PCA using `sklearn.decomposition.PCA`. The number of
        components is determined by `n_components`:
        * If `int`: Keeps the top `n_components`.
        * If `float` (0 < float <= 1): Keeps the minimum number of
            components required to explain at least that proportion of
            the variance.
    3. Replaces the original features with the calculated principal
        components (typically named 'PC1', 'PC2', ...). PCA transforms
        the data into a new set of uncorrelated variables (principal
        components) that capture the maximum variance.

        .. math::
           \text{ExplainedVarianceRatio}(\text{PC}_i) = \frac{\lambda_i}{\sum_j \lambda_j}

        where $\lambda_i$ are the eigenvalues of the covariance matrix.

The function returns the transformed DataFrame. If `target_col` was
specified, it's appended to the result. If `method='pca'` and
`return_pca=True`, it also returns the fitted `sklearn` PCA object.

**Usage Context:** Apply this function after creating a potentially
large number of features (e.g., via :func:`ts_engineering`) to either
remove redundant features (correlation method) or create a lower-
dimensional representation (PCA). Correlation removal can sometimes
improve model stability and interpretability. PCA is a powerful
dimensionality reduction technique but results in less interpretable
features (principal components instead of original variables). Requires
`scikit-learn` for PCA.

