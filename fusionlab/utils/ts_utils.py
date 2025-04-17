# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Times-series utilities (ts_utils).
"""

import warnings
from numbers import Real, Integral
from typing import Union, List
import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.stats import pearsonr, zscore
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
except ImportError:
    pass

from ..api.summary import ResultSummary
from ..compat.sklearn import Interval, StrOptions, validate_params
from ..core.array_manager import smart_ts_detector
from ..core.checks import exist_features, validate_ratio
from ..core.checks import is_in_if, check_params
from ..core.handlers import columns_manager
from ..core.io import to_frame_if
from ..utils.base_utils import validate_target_in, select_features
from ..utils.deps_utils import ensure_pkg
from ..utils.validator import is_time_series, is_frame


__all__= [ 
    'decompose_ts','infer_decomposition_method',
    'prepare_ts_df','trend_analysis','trend_ops',
    'ts_engineering','ts_validator','visual_inspection', 
    'ts_corr_analysis', 'transform_stationarity','ts_split', 
    'ts_outlier_detector', 'create_lag_features', 
    'select_and_reduce_features', 'get_decomposition_method', 
    'filter_by_period', 'to_dt', 
 ]

def filter_by_period(
    df, eval_periods, 
    dt_col =None, 
    ):
    """
    Filter a DataFrame based on the provided evaluation periods.

    The function filters the rows of the DataFrame where the values in the
    :param:`dt_col` match the provided :param:`eval_periods`. The function 
    supports filtering by various time granularities, such as year, month, 
    day, week, hour, minute, and second, based on the provided periods.

    The function performs the following operation:
    
    .. math::
        filtered\_df = df[dt_{col}.isin(eval_{periods})]

    It checks whether the values in the :math:`dt_{col}` match any of the 
    values in the :math:`eval_{periods}` and filters the DataFrame accordingly.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be filtered. It must contain a column with 
        date/time information specified in :param:`dt_col`.
  
    eval_periods : list of str or str
        A list or single string containing the periods to filter
        the DataFrame. The periods can be of varying granularity:
        - Year (`'YYYY'`)
        - Month (`'YYYY-MM'`)
        - Exact Date (`'YYYY-MM-DD'`)
        - Week (`'YYYY-Www'`)
        - Hour (`'YYYY-MM-DD HH'`)
        - Minute (`'YYYY-MM-DD HH:MM'`)
        - Second (`'YYYY-MM-DD HH:MM:SS'`)
    dt_col : str, optional
        The name of the column containing date or time information. 
        The values in this column are used to match the periods in 
        :param:`eval_periods`. This column should be of type `datetime`.
        If ``None`` index is infered as `dt_col`. If not datetime, an 
        error is raised. 
        
    Returns
    -------
    pandas.DataFrame
        The filtered DataFrame containing only the rows that match the 
        evaluation periods specified in :param:`eval_periods`.

    Examples
    --------
    >>> from gofast.utils.ts_utils import filter_by_period
    >>> df = pd.DataFrame({
    ...     'dt_col': ['2023-01-01', '2023-01-02', '2023-03-03', '2022-05-05'],
    ...     'value': [10, 20, 30, 40]
    ... })
    >>> df['dt_col'] = pd.to_datetime(df['dt_col'])
    >>> eval_periods = ['2023']
    >>> filtered_df = filter_by_period(df, eval_periods, 'dt_col')
    >>> print(filtered_df)
        dt_col  value
    0 2023-01-01     10
    1 2023-01-02     20
    2 2023-03-03     30

    >>> eval_periods = ['2023-01']
    >>> filtered_df = filter_by_period(df, eval_periods, 'dt_col')
    >>> print(filtered_df)
        dt_col  value
    0 2023-01-01     10
    1 2023-01-02     20

    >>> eval_periods = ['2023-01-01']
    >>> filtered_df = filter_by_period(df, eval_periods, 'dt_col')
    >>> print(filtered_df)
        dt_col  value
    0 2023-01-01     10

    >>> eval_periods = ['2023-W01']
    >>> filtered_df = filter_by_period(df, eval_periods, 'dt_col')
    >>> print(filtered_df)
        dt_col  value
    0 2023-01-01     10
    1 2023-01-02     20

    Notes
    -----
    - This function is flexible and can handle a wide range of 
      datetime formats.
    - The function uses the pandas `.isin()` method to perform
      the filtering based on the provided :param:`eval_periods`.
    - The :param:`eval_periods` parameter can contain periods 
      of various granularities (e.g., year, month, exact date),
      and the function will handle these correctly 
      by comparing the appropriate level of detail 
      (e.g., only the year, month, or exact date).
    - The :param:`dt_col` in the DataFrame must be of type
      `datetime`. If it's not, it will be converted to 
      `datetime` internally.
    - If :param:`eval_periods` is a single string, it is
      automatically converted into a list.

    See Also
    --------
    pandas.DataFrame: The pandas DataFrame object that provides
    methods such as `isin` to filter data.
    
    References
    ----------
    .. [1] Kouadio L. et al., "Time Series Filtering in DataFrames", 
       Journal of Data Processing, 2025. (In review)
    """
    # Validate the datetime column and
    # ensure it is in the correct format
    df, dt_col =ts_validator(
        df.copy(), dt_col=dt_col, 
        to_datetime="auto", 
        as_index=False, 
        error="raise", 
        verbose=1, 
        return_dt_col=True, 
    )
 
    # Ensure dt_col is of datetime type
    if pd.api.types.is_numeric_dtype(df[dt_col]):
        # Check if all values end in ".0" (are whole numbers)
        if (df[dt_col] % 1 == 0).all():
            # if dt_col.lower() != "year":  # Avoid modifying "year" column
            df[dt_col] = df[dt_col].astype(int)
            # 
    if pd.api.types.is_integer_dtype(df[dt_col]):
       # Convert the integer values to strings, then to datetime using the '%Y' format.
       df[dt_col] = pd.to_datetime(df[dt_col].astype(str), format='%Y')
    else:
        df[dt_col] = pd.to_datetime(df[dt_col])

    # If eval_periods is a single 
    # string, convert it to a list
    eval_periods= columns_manager(
        eval_periods, to_string=True)
    # Prepare the filtered DataFrame
    filtered_df = df.copy()
    
    # Filter by year (e.g., '2023')
        # Year
    if all(len(period) == 4 for period in eval_periods):  
        filtered_df = filtered_df[filtered_df[dt_col].dt.year.isin(
            [int(period) for period in eval_periods])]
    
    # Filter by month (e.g., '2023-01')
        # Year-Month
    elif all(len(period) == 7 for period in eval_periods):  
        filtered_df = filtered_df[
            filtered_df[dt_col].dt.strftime(
                '%Y-%m').isin(eval_periods)]
    
    # Filter by exact date (e.g., '2023-01-01')
        # Exact Date
    elif all(len(period) == 10 for period in eval_periods):  
        filtered_df = filtered_df[
            filtered_df[dt_col].dt.strftime(
                '%Y-%m-%d').isin(eval_periods)]
    
    # Filter by week (e.g., '2023-W01')
    elif all(len(period) == 7 and period[4] == 'W' 
             for period in eval_periods):  # Year-Wk
        filtered_df = filtered_df[
            filtered_df[dt_col].dt.strftime(
                '%Y-W%U').isin(eval_periods)]
    
    # Filter by hour (e.g., '2023-01-01 10')
        # Year-Month-Day Hour
    elif all(len(period) == 13 for period in eval_periods):  
        filtered_df = filtered_df[
            filtered_df[dt_col].dt.strftime(
                '%Y-%m-%d %H').isin(eval_periods)]
    
    # Filter by minute (e.g., '2023-01-01 10:30')
        # Year-Month-Day Hour:Min
    elif all(len(period) == 16 for period in eval_periods):  
        filtered_df = filtered_df[
            filtered_df[dt_col].dt.strftime(
                '%Y-%m-%d %H:%M').isin(eval_periods)]
    
    # Filter by second (e.g., '2023-01-01 10:30:00')
        # Year-Month-Day Hour:Min:Sec
    elif all(len(period) == 19 for period in eval_periods):  
        filtered_df = filtered_df
        [filtered_df[dt_col].dt.strftime(
            '%Y-%m-%d %H:%M:%S').isin(eval_periods)]
    
    else:
        raise ValueError(
            "eval_periods should contain valid year, month, day, "
            "week, hour, minute, or second format."
        )
    
    return filtered_df
    
def to_dt(
    df,
    dt_col=None,
    return_dt_col=False,
    format=None,
    error='raise',
    verbose=0,
    **kwargs
):
    r"""
    Converts a given DataFrame's column or index to datetime
    format using pandas' ``to_datetime`` functionality. This
    method `to_dt` handles integer-based columns or index by
    converting them to string before parsing. It can also
    return the name of the processed datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame that will be converted.

    dt_col : str, optional
        The name of the column to convert to datetime. If
        `dt_col` is None, then the DataFrame index will be
        converted.

    return_dt_col : bool, default=False
        If True, returns a tuple of the processed DataFrame
        and the name of the datetime column (or None if the
        index was converted). If False, returns only the
        processed DataFrame.

    format : str, optional
        The ``strftime`` format to use for parsing the
        datetime strings. If None, pandas attempts to infer
        the format automatically.

    error : {'raise', 'warn', 'ignore'}, default 'raise'
        The strategy to handle parsing errors. If 'raise',
        an exception is raised. If 'warn', a warning is
        issued. If 'ignore', the original DataFrame is
        returned unmodified without raising or warning.

    verbose : int, default=0
        The verbosity level of log messages:
        * 0 : No messages.
        * 1 : Basic messages.
        * 2 : More detailed messages.
        * 3 : Most verbose messages.

    **kwargs
        Additional keyword arguments passed directly to
        ``pd.to_datetime``.

    Returns
    -------
    pd.DataFrame or tuple
        The processed DataFrame with the specified column or
        index converted to datetime. If `return_dt_col` is
        True, a tuple of (processed DataFrame, datetime
        column name/None) is returned.

    Examples
    --------
    >>> from gofast.utils.ts_utils import to_dt
    >>> import pandas as pd

    >>> data = {
    ...     'Date': ['2021-01-01', '2021-01-02', '2021-01-03'],
    ...     'Value': [100, 200, 300]
    ... }
    >>> df = pd.DataFrame(data)
    >>> # Convert a column to datetime
    >>> df_dt = to_dt(df, dt_col='Date')
    >>> df_dt.info()

    >>> # Convert the index to datetime (example index as strings)
    >>> df_idx = df.set_index('Date')
    >>> df_idx_converted = to_dt(df_idx)
    >>> df_idx_converted.index

    Notes
    -----
    Internally, `to_dt` calls ``pd.to_datetime`` to perform
    the actual conversion. If an integer column or index is
    supplied, it is first cast to string and then parsed as
    datetime. This can be useful when timestamps are stored
    as integer values representing YYYYMMDD or similar
    formats.
    
    .. math::
        \text{Let } X \in \{\text{column, index}\}, \quad
        X_{\text{dt}} = pd.to\_datetime(X, \ldots)

    Here, :math:`X_{\text{dt}}` is the converted datetime
    representation. If ``dt_col`` is not provided, the index
    of the DataFrame is converted.

    See Also
    --------
    pandas.to_datetime : Pandas function for converting
        objects to datetime.
    pandas.DataFrame.astype : Cast object to a specified dtype.

    References
    ----------
    .. [1] Wes McKinney. *Python for Data Analysis: Data
       Wrangling with Pandas, NumPy, and IPython.* O'Reilly
       Media, 2nd Edition, 2017.
    """
    is_frame(df, df_only= True, objname="Data 'df'")
    
    # Create a copy of the original DataFrame to avoid
    # side effects on the user's data.
    df_copy  = df.copy()
    processed_dt_col = None

    try:
        if dt_col is None:
            # Process the index if no column is specified.
            if not pd.api.types.is_datetime64_any_dtype(
                df_copy.index
            ):
                if verbose >= 1:
                    print("Converting index to datetime.")
                # If index is integer, convert to string first.
                if pd.api.types.is_integer_dtype(
                    df_copy.index
                ):
                    if verbose >= 2:
                        print("Index is integer. Converting "
                              "to string first.")
                    index_str = df_copy.index.astype(str)
                    new_index = pd.to_datetime(
                        index_str,
                        format=format,
                        errors='raise',
                        **kwargs
                    )
                else:
                    new_index = pd.to_datetime(
                        df_copy.index,
                        format=format,
                        errors='raise',
                        **kwargs
                    )
                df_copy.index = new_index
                processed_dt_col = None
                if verbose >= 1:
                    print("Index converted to datetime.")
            else:
                if verbose >= 2:
                    print("Index is already datetime. "
                          "No conversion needed.")
        else:
            # Process the specified column.
            if dt_col not in df_copy.columns:
                raise ValueError(
                    f"Column '{dt_col}' not found in "
                    f"DataFrame."
                )
            col = df_copy[dt_col]
            # Convert integer column to string first.
            if pd.api.types.is_integer_dtype(col):
                if verbose >= 1:
                    print(f"Converting integer column "
                          f"'{dt_col}' to datetime with "
                          f"format {format}.")
                col_str = col.astype(str)
                converted_col = pd.to_datetime(
                    col_str,
                    format=format,
                    errors='raise',
                    **kwargs
                )
            else:
                if verbose >= 1:
                    print(f"Converting column '{dt_col}' "
                          f"to datetime.")
                converted_col = pd.to_datetime(
                    col,
                    format=format,
                    errors='raise',
                    **kwargs
                )
            df_copy[dt_col] = converted_col
            processed_dt_col = dt_col
            if verbose >= 1:
                print(f"Column '{dt_col}' "
                      f"converted to datetime.")

    except Exception as e:
        if error == 'raise':
            raise
        elif error == 'warn':
            warnings.warn(
                f"DateTime conversion failed: {e}"
            )
            # Return original DataFrame if error
            # occurs in 'warn' mode.
            return (
                (df, None) if return_dt_col else df
            )
        elif error == 'ignore':
            # Return original DataFrame if error
            # occurs in 'ignore' mode.
            return (
                (df, None) if return_dt_col else df
            )
        else:
            raise ValueError(
                "error must be 'raise', 'warn', "
                "or 'ignore'."
            )

    if return_dt_col:
        return df_copy, processed_dt_col
    else:
        return df_copy

def ts_validator(
    df,
    dt_col=None,
    to_datetime=None,
    as_index="auto",
    error='raise',
    return_dt_col=False,
    ensure_order=False,
    verbose=0,
):
    r"""
    Validate and preprocess time series data, ensuring the presence of
    a properly formatted datetime column or index. This function can
    automatically convert a given column or the DataFrame index to a
    datetime type, sort the data by time order, and optionally set the
    datetime column as the index.

    In a more formal sense, let :math:`\{x_t\}_{t=1}^N` represent a
    time series with :math:`t` denoting the time index and :math:`N`
    the number of observations [1]_. The role of this function is to
    ensure the alignment:

    .. math::
        t_1 < t_2 < \ldots < t_N

    so that any subsequent operations or modeling steps assume valid
    temporal ordering.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame which must contain time-related
        information. If ``dt_col`` is not provided, the index of
        ``df`` may be used or converted if possible.
    dt_col : str, optional
        The column name representing date/time. If ``dt_col`` is
        not found in the DataFrame columns or index, behavior
        depends on the ``error`` parameter.
    to_datetime : {None, 'auto', 'Y', 'M', 'W', 'D', 'H', 'min', 's'}, optional
        Controls how to convert the detected time column if it is
        not already in a datetime format. Examples:
        
        * ``None``: No forced conversion; only format detection.
        * ``'auto'``: Automatic inference of the date/time format.
        * ``'D'``: Convert using daily periods, etc.
    as_index : bool or {"auto"}, optional
        Whether to set the time column as index after conversion:

        * ``True``: Set ``dt_col`` as the index in the returned
          DataFrame.
        * ``False``: Retain ``dt_col`` as a column.
        * ``"auto"``: Keep the current structure if the column is
          already in the index; else, do not change it.
    error : {'raise', 'ignore', 'warn'}, optional
        Defines how to handle potential errors such as an invalid
        format or missing time column:

        * ``'raise'``: Raise a :class:`ValueError`.
        * ``'warn'``: Issue a warning and return the unmodified
          data if it cannot be converted.
        * ``'ignore'``: Silently ignore conversion failures.
    return_dt_col : bool, optional
        If ``True``, return a tuple ``(df, dt_col)`` with the
        final validated DataFrame and the name of the detected
        time column.
    ensure_order : bool, optional
        If ``True``, sorts the DataFrame in ascending time order
        based on the detected or provided ``dt_col``. For time
        series modeling, ensuring chronological ordering can be
        critical.
    verbose : int, optional
        Verbosity level. The higher the value, the more
        information is printed during execution:

        * ``0``: No output.
        * ``1``: Basic info messages.
        * ``2``: Detailed messages on steps taken.
        * ``3``: Very detailed internal states for debugging.

    Returns
    -------
    df : pandas.DataFrame
        The validated and possibly re-indexed DataFrame with
        correctly formatted datetime information.
    (df, dt_col) : (pandas.DataFrame, str)
        Returned if ``return_dt_col=True``. The first element is
        the processed DataFrame, and the second element is the
        detected time column name.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import ts_validator

    >>> # Example DataFrame with a 'Date' column
    >>> data = {
    ...     'Date': ['2020-01-01', '2020-01-02', '2020-01-03'],
    ...     'Value': [10, 15, 20]
    ... }
    >>> df = pd.DataFrame(data)
    >>> validated_df = ts_validator(df, dt_col='Date',
    ...                             to_datetime='auto',
    ...                             ensure_order=True,
    ...                             verbose=1)
    Datetime column detected: 'Date' ...
    Datetime conversion applied successfully ...
    >>> validated_df
               Value
    Date
    2020-01-01     10
    2020-01-02     15
    2020-01-03     20

    Notes
    -----
    Proper time series validation and ordering is crucial for
    reliable forecasting and analysis. If the column or index
    cannot be converted to a valid datetime format, consider
    adjusting your time field or using the correct format codes.

    See Also
    --------
    ts_engineering : Higher-level features engineering on time
        series data.
    ts_corr_analysis : Analyze correlation structures in time
        series.

    References
    ----------
    .. [1] Brockwell, P.J., & Davis, R.A. (2016). *Introduction to
           Time Series and Forecasting*. Springer.
    .. [2] Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M.
           (2015). *Time Series Analysis: Forecasting and Control*.
           John Wiley & Sons.
    """

    # Convert the input to a DataFrame if needed
    df = to_frame_if(df, df_only=True)

    # If no datetime column is explicitly specified, attempt to use
    # or detect a datetime index or convert it
    if dt_col is None:
        # Check if the DataFrame index is already datetime
        if pd.api.types.is_datetime64_any_dtype(df.index):
            if verbose >= 1:
                print(
                    "Datetime index detected. No further datetime "
                    "conversion is required."
                )
            dt_col = df.index.name
            # If user wants columns only, reset index
            if not as_index:
                df.reset_index(inplace=True)
            # Return as needed
            return df if not return_dt_col else (df, dt_col)
        else:
            # Not a datetime index; attempt conversion if requested
            if verbose >= 1:
                print(
                    "Index is not a datetime type. Checking datetime "
                    "conversion settings."
                )

            # If the user specifies a format or "auto" for index conv
            if to_datetime is not None:
                try:
                    df.index = pd.to_datetime(
                        df.index,
                        errors='coerce',
                        format=to_datetime
                    )
                    # Check if any null results from conversion
                    if df.index.isnull().any():
                        raise ValueError(
                            "Some index values could not be converted "
                            "to datetime."
                        )
                    if verbose >= 2:
                        print(
                            f"Index converted to datetime using: "
                            f"{to_datetime}"
                        )
                except Exception as e:
                    if error == 'raise':
                        raise ValueError(
                            "Failed to convert index to datetime "
                            f"({e})."
                        )
                    elif error == 'warn':
                        warnings.warn(
                            "Failed to convert index to datetime. "
                            f"{e}"
                        )
                dt_col = df.index.name
                # Return as needed
                return df if not return_dt_col else (df, dt_col)
            else:
                # No dt_col, no forced conversion
                if error == 'raise':
                    raise ValueError(
                        "Index is not datetime and no dt_col "
                        "or to_datetime was provided."
                    )
                elif error == 'warn':
                    warnings.warn(
                        "Index is not datetime and no to_datetime "
                        "was provided."
                    )
                # Return if ignoring the problem
                return df if not return_dt_col else (df, dt_col)
    else:
        # dt_col is specified
        if dt_col in df.index:
            # Move dt_col from index back to columns
            df.reset_index(inplace=True, drop=False)
        # Validate that dt_col is in columns
        exist_features(df, features=dt_col, name="Datetime column")
        # Use the built-in time series detector
        df = smart_ts_detector(
            df=df,
            dt_col=dt_col,
            return_types="df",
            as_index=as_index,
            error=error,
            verbose=verbose,
            to_datetime=to_datetime,
        )

    # Optionally ensure ascending time order
    if ensure_order:
        # Temporarily remove from index if set
        is_index_already = False
        if dt_col in df.index:
            df.reset_index(inplace=True)
            is_index_already = True
        df = df.sort_values(by=dt_col)
        if is_index_already:
            df.set_index(dt_col, inplace=True)

    return df if not return_dt_col else (df, dt_col)

@validate_params({ 
    "trend_type": [StrOptions({'both', 'upward', 'downward'})], 
    "strategy": [StrOptions({'adf', 'kpss'})]
    })
@ensure_pkg(
    "statsmodels", 
    extra="'stasmodels' is required for 'trend_analysis' to proceed."
)
def trend_analysis(
    df,
    value_col,
    dt_col=None,
    view=False,
    check_stationarity=True,
    trend_type='both',
    strategy="adf", 
    stationnay_color='green',
    linestyle='--',
    fig_size=(10, 6),
    trend_color='red',
    show_grid=True,
    error='raise',
    verbose=0,
    **kw
):
    r"""
    Perform trend analysis on a given time series, combining a
    stationarity test and a linear trend detection. The function
    checks whether the series is stationary via the specified
    test (ADF or KPSS) and, if non-stationary, fits a simple
    linear regression model to infer the direction of trend.

    Mathematically, the trend detection relies on fitting:

    .. math::
        y_t = \beta_0 + \beta_1 \cdot t + \epsilon_t,

    where :math:`y_t` is the value at time :math:`t`. The slope
    :math:`\beta_1` determines whether the trend is upward
    (:math:`\beta_1 > 0`), downward (:math:`\beta_1 < 0`), or
    stationary (:math:`\beta_1 \approx 0`).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing time series data.
    value_col : str or array-like
        The name (or direct series) of the target variable
        to analyze.
    dt_col : str, optional
        The column name representing datetime information.
        If ``None``, attempts to detect or convert the index
        to datetime.
    view : bool, optional
        If ``True``, displays a plot showing the original time
        series along with the fitted trend line (if applicable).
    check_stationarity : bool, optional
        If ``True``, performs a stationarity test (ADF or KPSS)
        before trend detection. If the data is found stationary
        and ``trend_type`` is not 'both', the function may not
        fit a trend line.
    trend_type : {'both', 'upward', 'downward'}, optional
        Type of trend to detect. If ``'both'``, considers all
        possibilities. If ``'upward'``, checks only for a
        positive slope. If ``'downward'``, checks only for a
        negative slope.
    strategy : {'adf', 'kpss'}, optional
        Stationarity test to use:

        * ``'adf'``: Augmented Dickey-Fuller test.
        * ``'kpss'``: Kwiatkowski–Phillips–Schmidt–Shin test.
    stationnay_color : str, optional
        Color for the mean line if the series is found to be
        stationary.
    linestyle : str, optional
        The line style for the stationary mean line.
    fig_size : tuple of (int, int), optional
        The width and height of the plot in inches.
    trend_color : str, optional
        Color for the fitted trend line.
    show_grid : bool, optional
        Whether to display grid lines on the plot.
    error : {'raise', 'warn', 'ignore'}, optional
        Behavior to adopt when encountering errors:

        * ``'raise'``: Raises a ValueError.
        * ``'warn'``: Issues a warning message.
        * ``'ignore'``: Silently ignores errors.
    verbose : int, optional
        Verbosity level controlling console output:

        * ``0``: No messages.
        * ``1``: Basic information.
        * ``2``: More detailed status updates.
    **kw : dict, optional
        Additional keyword arguments passed to matplotlib's
        plotting function (e.g., marker styles, alpha).

    Returns
    -------
    trend : str
        Detected trend type:

        * ``'upward'``: If the slope is strictly positive.
        * ``'downward'``: If the slope is strictly negative.
        * ``'stationary'``: If no clear slope or series is
          identified as stationary.
    p_value : float or None
        p-value from the stationarity test. If
        ``check_stationarity=False``, returns ``None``.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import trend_analysis
    >>> data = {
    ...     'Date': [
    ...         '2020-01-01', '2020-01-02', '2020-01-03',
    ...         '2020-01-04', '2020-01-05'
    ...     ],
    ...     'Value': [10, 12, 15, 13, 14]
    ... }
    >>> df = pd.DataFrame(data)
    >>> trend, p_val = trend_analysis(
    ...     df, value_col='Value',
    ...     dt_col='Date', view=True,
    ...     check_stationarity=True, strategy='adf'
    ... )
    >>> trend
    'non-stationary'  # or 'upward', 'downward', or 'stationary'

    Notes
    -----
    Identifying trends is a central component of time series
    analysis. Non-stationary behavior can lead to misleading
    statistical results if not handled properly [1]_ [2]_.
    This function thus helps detect both stationarity and
    monotonic tendencies in the data.

    See Also
    --------
    ts_validator : Validate and preprocess time series data.
    transform_stationarity : Convert a non-stationary series
        into a stationary one through differencing or other
        transformations.

    References
    ----------
    .. [1] Brockwell, P.J., & Davis, R.A. (2016). *Introduction to
           Time Series and Forecasting*. Springer.
    .. [2] Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M.
           (2015). *Time Series Analysis: Forecasting and Control*.
           John Wiley & Sons.
    """

    # Validate and process the datetime column
    # (set index, format to datetime, etc.)
    df, dt_col = ts_validator(
        df,
        dt_col=dt_col,
        to_datetime='auto',
        as_index=False,
        error=error,
        return_dt_col=True,
        verbose=verbose
    )

    # Validate presence of target column or series
    target, _ = validate_target_in(
        df,
        value_col,
        error=error,
        verbose=verbose
    )

    # Initialize p_value and trend
    p_value = None
    trend = 'non-stationary'

    # Step 1: Check stationarity (optional)
    if check_stationarity:
        if strategy == "adf":
            # Augmented Dickey-Fuller test
            result = adfuller(df[target.name].dropna())
            p_value = result[1]
            if p_value < 0.05:
                trend = 'stationary'
            else:
                trend = 'non-stationary'
        elif strategy == "kpss":
            # KPSS test (level stationarity)
            result = kpss(
                df[target.name].dropna(),
                regression='c'
            )
            p_value = result[1]
            if p_value < 0.05:
                trend = 'non-stationary'
            else:
                trend = 'stationary'

    # Step 2: Apply trend detection if non-stationary or forced
    if trend == 'non-stationary' or trend_type == 'both':
        # Fit a linear regression to detect slope
        X = np.arange(len(df)).reshape(-1, 1)
        y = target.values
        ols_model = sm.OLS(y, sm.add_constant(X)).fit()
        slope = ols_model.params[1]

        # Classify slope direction
        if slope > 0:
            detected_trend = 'upward'
        elif slope < 0:
            detected_trend = 'downward'
        else:
            detected_trend = 'stationary'

        # Update if it contradicts earlier stationarity
        if trend == 'stationary' and detected_trend != 'stationary':
            trend = detected_trend
        else:
            trend = detected_trend

    # Step 3: Visualization
    if view:
        plt.figure(figsize=fig_size)
        # Plot original data. If needed, pass extra
        # keywords in **kw (e.g., markers, alpha, etc.)
        plt.plot(
            df[dt_col],
            df[target.name],
            label='Original Data',
            color='blue',
            **kw
        )
        plt.title(
            f"Trend Analysis for {dt_col}",
            fontsize=14,
            fontweight='bold'
        )
        plt.xlabel('Time', fontsize=12)
        plt.ylabel(target.name, fontsize=12)

        # For stationary, draw mean line
        if trend == 'stationary':
            plt.axhline(
                y=target.mean(),
                color=stationnay_color,
                linestyle=linestyle,
                label='Mean Line'
            )
        else:
            # Plot fitted OLS trend line
            plt.plot(
                df[dt_col],
                ols_model.fittedvalues,
                color=trend_color,
                label='Fitted Trend',
                linewidth=2
            )

        # Toggle grid lines
        if not show_grid:
            plt.grid(False)
        else:
            plt.grid(True, linestyle=':', alpha=0.7)

        # Annotate the detected trend
        plt.text(
            0.05,
            0.95,
            f"Detected Trend: {trend.capitalize()}",
            transform=plt.gca().transAxes,
            fontsize=12,
            color=trend_color,
            fontweight='bold',
            ha='left',
            va='top'
        )

        # Annotate stationarity test results
        if p_value is not None:
            plt.text(
                0.05,
                0.85,
                (
                    f"Stationarity Test "
                    f"({strategy.upper()}) "
                    f"p-value: {p_value:.3f}"
                ),
                transform=plt.gca().transAxes,
                fontsize=10,
                color='black',
                fontweight='normal',
                ha='left',
                va='top'
            )

        plt.legend()
        plt.tight_layout()
        plt.show()

    return trend, p_value

@validate_params({ 
    "trend_type": [StrOptions({'both', 'upward', 'downward'})], 
    "strategy": [StrOptions({'adf', 'kpss'})], 
    "ops": [StrOptions({'remove_upward', 'remove_downward', 'remove_both',
           'detrend', 'none'}), None], 
    })
@ensure_pkg(
    "statsmodels", 
    extra="'stasmodels' is required for 'trend_ops' to proceed."
)
def trend_ops(
    df,
    dt_col,
    value_col,
    ops=None,
    check_stationarity=True,
    trend_type='both',
    error='raise',
    strategy="adf",  
    verbose=0,
    view=False,
    fig_size=(10, 4),
    show_grid=False,
    **kw
):
    r"""
    Perform transformations on a time series (e.g., removing
    upward/downward trends or applying differencing) based on
    automatically detected trends. The function first determines
    if the series is stationary or non-stationary and then applies
    the specified operation to remove or mitigate the detected
    trend.

    In particular, when differencing is applied (for example in the
    ``'detrend'`` option), the mathematical operator is:

    .. math::
        \nabla Y_t = Y_t - Y_{t-1},

    which removes first-order trends. For a linear trend removal
    through ordinary least squares (OLS), the model:

    .. math::
        Y_t = \beta_0 + \beta_1 \cdot t + \epsilon_t

    is fitted and subtracted from the original series if the slope
    :math:`\beta_1` indicates an upward or downward trend.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the time series data.
    dt_col : str
        The column name representing datetime in the DataFrame.
        Must be a valid time-like column or convertible to one.
    value_col : str or array-like
        The name (or direct series) of the target variable to
        transform.
    ops : {'remove_upward', 'remove_downward', 'remove_both',
           'detrend', 'none'}, optional
        The transformation operation to perform:

        * ``'remove_upward'``: Detect and remove upward trend only.
        * ``'remove_downward'``: Detect and remove downward trend only.
        * ``'remove_both'``: Remove any identified trend
          (upward/downward).
        * ``'detrend'``: Apply differencing if the series is
          non-stationary.
        * ``'none'``: No transformation is performed.
    check_stationarity : bool, optional
        Whether to apply a stationarity test (ADF or KPSS) before
        deciding on transformations. If ``False``, transformations
        rely solely on the linear trend detection.
    trend_type : {'both', 'upward', 'downward'}, optional
        Type of trend detection applied in conjunction with
        stationarity checks:

        * ``'both'``: Check for both upward and downward slopes.
        * ``'upward'``: Focus on detecting a positive slope only.
        * ``'downward'``: Focus on detecting a negative slope only.
    error : {'raise', 'warn', 'ignore'}, optional
        Behavior to adopt if transformation yields invalid data
        (e.g., all NaNs):

        * ``'raise'``: Raises a ValueError.
        * ``'warn'``: Issues a warning.
        * ``'ignore'``: Silently ignores it.
    strategy : {'adf', 'kpss'}, optional
        Stationarity test used if ``check_stationarity=True``:

        * ``'adf'``: Augmented Dickey-Fuller.
        * ``'kpss'``: Kwiatkowski–Phillips–Schmidt–Shin test.
    verbose : int, optional
        Verbosity level controlling console output:

        * ``0``: No printing.
        * ``1``: Basic info.
        * ``2``: More detailed logs.
        * ``3``: Very detailed logs (debug mode).
    view : bool, optional
        If ``True``, displays a plot comparing original vs.
        transformed data after trend removal or differencing.
    fig_size : tuple of (int, int), optional
        Figure size for the resulting plot in inches
        (width, height).
    show_grid : bool, optional
        Whether to display grid lines on the generated plots.
    **kw : dict, optional
        Additional keyword arguments forwarded to lower-level
        functions or the plotting calls (e.g., markers, line
        widths, alpha, etc.).

    Returns
    -------
    df : pandas.DataFrame
        The original DataFrame with its target column potentially
        replaced by the transformed version.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import trend_ops
    >>> data = {
    ...     'Date': [
    ...         '2020-01-01', '2020-01-02', '2020-01-03',
    ...         '2020-01-04', '2020-01-05'
    ...     ],
    ...     'Value': [10, 12, 15, 13, 14]
    ... }
    >>> df = pd.DataFrame(data)
    >>> # Remove both upward or downward linear trends if found
    >>> transformed_df = trend_ops(
    ...     df=df, dt_col='Date',
    ...     value_col='Value',
    ...     ops='remove_both',
    ...     view=True,
    ...     verbose=1
    ... )
    Detected Trend: upward
    Stationarity Test p-value: 0.3147
    Both upward and downward trends removed.

    See Also
    --------
    trend_analysis : Combine stationarity tests and linear slope
        detection to classify the series as 'upward', 'downward',
        or 'stationary'.
    transform_stationarity : Convert a time series to stationary
        via various transformations (differencing, logging, etc.).

    Notes
    -----
    Properly removing or mitigating trends can be a critical step
    before applying certain time series models, especially ARIMA-
    type models that assume stationarity [1]_ [2]_. This function
    helps automate that process based on statistical tests and
    OLS-based slope detection.

    References
    ----------
    .. [1] Brockwell, P.J., & Davis, R.A. (2016). *Introduction
           to Time Series and Forecasting*. Springer.
    .. [2] Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M.
           (2015). *Time Series Analysis: Forecasting and Control*.
           John Wiley & Sons.
    """

    # Validate the input DataFrame (raises exception if not valid)
    is_frame(
        df,
        df_only=True,
        raise_exception=True,
        objname="Dataframe 'df'"
    )
    # Validate it is a time series but skip checking intervals
    is_time_series(
        df,
        time_col=dt_col,
        check_time_interval=False
    )

    # Step 1: Perform trend analysis (stationarity + slope detection)
    trend, p_value = trend_analysis(
        df,
        value_col=value_col,
        dt_col=dt_col,
        check_stationarity=check_stationarity,
        trend_type=trend_type,
        view=False,  # no immediate plot
        strategy=strategy,
        **kw
    )

    # Step 2: Validate that target column exists
    target, _ = validate_target_in(
        df,
        value_col,
        error=error,
        verbose=verbose
    )
    tname = target.name

    # Set datetime column as index for transformations
    df.set_index(dt_col, inplace=True)

    # Prepare container for transformed series
    detrended_data = pd.DataFrame()

    # Optional console logging about detected trend
    if verbose >= 1:
        print(f"Detected Trend: {trend}")
        if check_stationarity:
            print(
                f"Stationarity Test p-value: {p_value:.4f}"
            )

    # Step 3: Conditional transformations based on 'ops'
    if ops == 'remove_upward':
        # Remove upward trend only if detected
        if trend != 'upward':
            if verbose >= 1:
                print(
                    "No upward trend detected. "
                    "Skipping upward removal."
                )
        else:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[tname].values
            model = sm.OLS(y, sm.add_constant(X)).fit()
            detrended_data = df[tname] - model.fittedvalues
            if verbose >= 1:
                print("Upward trend removed.")

    elif ops == 'remove_downward':
        # Remove downward trend only if detected
        if trend != 'downward':
            if verbose >= 1:
                print(
                    "No downward trend detected. "
                    "Skipping downward removal."
                )
        else:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[tname].values
            model = sm.OLS(y, sm.add_constant(X)).fit()
            detrended_data = df[tname] - model.fittedvalues
            if verbose >= 1:
                print("Downward trend removed.")

    elif ops == 'remove_both':
        # Remove linear trend whether upward or downward
        if trend == 'stationary' and verbose >= 1:
            print(
                "Data is already stationary. "
                "Skipping trend removal."
            )
        else:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[tname].values
            model = sm.OLS(y, sm.add_constant(X)).fit()
            detrended_data = df[tname] - model.fittedvalues
            if verbose >= 1:
                print("Both upward and downward trends removed.")

    elif ops == 'detrend':
        # Differencing if series is non-stationary
        if trend == 'stationary':
            if verbose >= 1:
                print(
                    "Data is already stationary. "
                    "Skipping differencing."
                )
        else:
            # Apply simple differencing
            detrended_data = df[tname].diff().dropna()
            if verbose >= 1:
                print("Data detrended using differencing.")

    elif ops == 'none':
        # Do nothing
        if verbose >= 1:
            print("No transformation applied.")

    # Handle potential errors if the transformation yields all NaNs
    if ops is not None and df[tname].isnull().all():
        if error == 'raise':
            raise ValueError(
                f"After {ops}, the data became entirely null."
            )
        elif error == 'warn':
            warnings.warn(
                f"After {ops}, the data became entirely null."
            )

    # Step 4: Visualization if requested
    if view:
        if detrended_data.empty:
            if verbose >= 1:
                print(
                    f"No transformation applied for trend='{trend}' "
                    f"and ops='{ops}'. Skipping visualization."
                )
        else:
            fig, axes = plt.subplots(1, 2, figsize=fig_size)
            # Original data
            axes[0].plot(
                df.index,
                df[tname],
                label="Original Data",
                color='blue'
            )
            axes[0].set_title("Original Data with Trend")
            axes[0].set_xlabel("Time")
            axes[0].set_ylabel(tname)
            axes[0].grid(show_grid)

            # Transformed data
            axes[1].plot(
                df.index,
                detrended_data,
                label="Transformed Data",
                color='green'
            )
            axes[1].set_title(f"Transformed Data (After {ops})")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel(tname)
            axes[1].grid(show_grid)

            plt.tight_layout()
            plt.show()

    # If no transformation was applied, return original df
    # Otherwise, replace original column with the transformation
    if detrended_data.empty:
        return df

    df[tname] = detrended_data
    return df

@validate_params ({
    "window": [Interval(Integral, 0, None, closed="neither")], 
    "max_col":  [Interval(Integral, 1, None, closed="left")], 
    "lags": [Interval(Integral, 1, None, closed="left")], 
    })
@ensure_pkg(
    "statsmodels", 
    extra="'statsmodels' is required for 'visual_inspection' to proceed."
)
def visual_inspection(
    df,
    value_col,
    dt_col=None,
    window=12,
    seasonal_period=None,
    show_acf=True,
    show_decomposition=True,
    show_trend=True,
    show_seasonal=True,
    show_residual=True,
    lags=2, 
    figsize=(14, 8),
    show_grid=True,
    max_cols=3,
    decompose_on_sep=False,
    title=None,
    **kwargs
):
    r"""
    Perform visual inspection of a time series by plotting its
    original form and various diagnostic plots, including rolling
    statistics (trend, seasonality), autocorrelation (ACF), and
    optional seasonal decomposition.

    Rolling statistics (mean or standard deviation) typically take
    the form:

    .. math::
        \text{RollingStat}_t = \frac{1}{W}\sum_{i=0}^{W-1}
        X_{t-i},

    where :math:`W` is the rolling window size. Seasonal
    decomposition, if requested, is performed via
    :func:`statsmodels.tsa.seasonal.seasonal_decompose` with a
    user-specified period.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing time series data.
    value_col : str or array-like
        The name (or direct series) of the target variable
        for inspection.
    dt_col : str, optional
        The column name corresponding to the time dimension. If
        ``None``, the function labels the x-axis as
        "Date/Time" generically.
    window : int, optional
        Window size for rolling mean or standard deviation
        calculations.
    seasonal_period : int, optional
        Defines the frequency of the seasonality for the
        decomposition. For example, ``12`` for monthly data
        with yearly seasonality.
    figsize : tuple of (float, float), optional
        The figure width and height in inches for the main
        plotting layout.
    show_acf : bool, optional
        If ``True``, plots the Autocorrelation Function (ACF).
    show_decomposition : bool, optional
        If ``True``, includes a seasonal decomposition plot of
        the data.
    show_trend : bool, optional
        If ``True``, plots the rolling mean to visualize the
        trend component.
    show_seasonal : bool, optional
        If ``True``, plots the rolling standard deviation to
        provide a rough view of seasonality.
    show_residual : bool, optional
        If ``True``, plots the residuals from the seasonal
        decomposition (requires ``show_decomposition=True`` and
        a valid ``seasonal_period``).
    show_grid : bool, optional
        Controls whether grid lines are added to the plots.
    max_cols : int, optional
        The maximum number of subplot columns in the composite
        figure.
    lags : int, optional
        Number of lag features to create. For example,
        ``lags=5`` yields columns for
        :math:`X_{t-1}, X_{t-2}, \ldots, X_{t-5}`.
    decompose_on_sep : bool, optional
        If ``True``, plots the seasonal decomposition in a
        separate figure instead of the composite layout.
    title : str, optional
        Main title for the first subplot featuring the original
        time series.
    **kwargs : dict, optional
        Additional keyword arguments passed to the main plot
        function (e.g., line style, marker style, transparency).

    Notes
    -----
    Visual inspection of rolling statistics, autocorrelation,
    and seasonal decomposition provides valuable insight into
    stationarity and potential seasonal effects [1]_[2]_.

    Returns
    -------
    None
        The function only displays the generated plots. It does
        not return a value.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import visual_inspection
    >>> data = {
    ...     'Date': [
    ...         '2021-01-01','2021-02-01','2021-03-01',
    ...         '2021-04-01','2021-05-01','2021-06-01'
    ...     ],
    ...     'Sales': [100, 120, 130, 115, 150, 170]
    ... }
    >>> df = pd.DataFrame(data)
    >>> df['Date'] = pd.to_datetime(df['Date'])
    >>> df.set_index('Date', inplace=True)
    >>> visual_inspection(
    ...     df, value_col='Sales',
    ...     window=2, seasonal_period=3,
    ...     show_acf=True, show_decomposition=True,
    ...     title="Sales Over Time"
    ... )

    See Also
    --------
    trend_analysis : Detect stationarity and linear trends
        within a time series.
    trend_ops : Remove or mitigate upward/downward trends
        or apply differencing to a time series.
    transform_stationarity : Convert a time series to
        stationary through differencing and other methods.

    References
    ----------
    .. [1] Brockwell, P.J., & Davis, R.A. (2016). *Introduction
           to Time Series and Forecasting*. Springer.
    .. [2] Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M.
           (2015). *Time Series Analysis: Forecasting and Control*.
           John Wiley & Sons.
    """
    title = title or "Time Series Visual Inspection" 
    # Validate the input DataFrame
    is_frame(
        df,
        df_only=True,
        raise_exception=True,
        objname="Dataframe 'df'"
    )

    # Extract and validate the target column
    ts, _ = validate_target_in(
        df,
        value_col,
        error='raise',
        verbose=0
    )
    tname = ts.name

    # Count the number of subplots needed
    num_plots = 1  # Original series is always plotted
    if show_trend:
        num_plots += 1
    if show_seasonal:
        num_plots += 1
    if show_acf:
        num_plots += 1
    # Decomposition can add up to 4 subplots (Observed, Trend,
    # Seasonal, Residual) in the same figure
    if seasonal_period and show_decomposition and not decompose_on_sep:
        num_plots += 4
    # If decomposition is separate but user wants residual
    # explicitly in the same figure, it adds 1 more subplot
    if seasonal_period and show_decomposition and show_residual:
        num_plots += 1

    # Determine rows and columns for subplots
    num_rows = (
        num_plots // max_cols
        + (num_plots % max_cols > 0)
    )
    num_cols = min(max_cols, num_plots)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=figsize
    )

    # Flatten the axes for consistent indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # For labeling x-axis
    if dt_col is None:
        x_label = "Date/Time"
    else:
        x_label = str(dt_col).title()

    # Plot 1: Original Time Series
    plot_idx = 0
    axes[plot_idx].plot(
        df.index,
        ts,
        label="Original Data",
        color='blue',
        **kwargs
    )
    axes[plot_idx].set_title(
        f"{title}: Original Time Series"
    )
    axes[plot_idx].set_xlabel(x_label)
    axes[plot_idx].set_ylabel(tname)
    if show_grid:
        axes[plot_idx].grid(True, linestyle=':', alpha=0.7)
    else:
        axes[plot_idx].grid(False)
    plot_idx += 1

    # Plot 2: Rolling Mean (Trend)
    if show_trend:
        rolling_mean = ts.rolling(window=window).mean()
        axes[plot_idx].plot(
            df.index,
            ts,
            label="Original Data",
            color='blue',
            alpha=0.5
        )
        axes[plot_idx].plot(
            df.index,
            rolling_mean,
            label="Rolling Mean (Trend)",
            color='red'
        )
        axes[plot_idx].set_title(
            f"Rolling Mean (Trend) - Window={window}"
        )
        axes[plot_idx].set_xlabel(x_label)
        axes[plot_idx].set_ylabel(tname)
        if show_grid:
            axes[plot_idx].grid(True, linestyle=':', alpha=0.7)
        else:
            axes[plot_idx].grid(False)
        plot_idx += 1

    # Plot 3: Rolling Standard Deviation (Seasonality proxy)
    if show_seasonal:
        rolling_std = ts.rolling(window=window).std()
        axes[plot_idx].plot(
            df.index,
            rolling_std,
            label="Rolling Std (Seasonality)",
            color='green'
        )
        axes[plot_idx].set_title(
            f"Rolling Standard Deviation - Window={window}"
        )
        axes[plot_idx].set_xlabel(x_label)
        axes[plot_idx].set_ylabel("Rolling Std")
        if show_grid:
            axes[plot_idx].grid(True, linestyle=':', alpha=0.7)
        else:
            axes[plot_idx].grid(False)
        plot_idx += 1

    # Plot 4: Autocorrelation Function (ACF)
    if show_acf:
        # Raise an error if 'lags' is too large for the current data length
        if lags >= len(ts):
            raise ValueError(
                f"Cannot compute ACF with lags={lags} for a series "
                f"of length={len(ts)}. Please reduce 'lags'."
            )
        plot_acf(ts, ax=axes[plot_idx], lags=lags)
        axes[plot_idx].set_title("Autocorrelation (ACF)")
        plot_idx += 1

    # Seasonal Decomposition if a seasonal_period is provided
    if seasonal_period and show_decomposition:
        decomposition = seasonal_decompose(
            ts,
            model='additive',
            period=seasonal_period
        )

        # Plot decomposition in a separate figure, if requested
        if decompose_on_sep:
            # Create new figure
            fig_decomp, axes_decomp = plt.subplots(
                4,
                1,
                figsize=(
                    figsize[0],
                    figsize[0]
                )
            )
            decomposition.observed.plot(
                ax=axes_decomp[0],
                label='Observed',
                color='blue'
            )
            decomposition.trend.plot(
                ax=axes_decomp[1],
                label='Trend',
                color='red'
            )
            decomposition.seasonal.plot(
                ax=axes_decomp[2],
                label='Seasonal',
                color='green'
            )
            decomposition.resid.plot(
                ax=axes_decomp[3],
                label='Residuals',
                color='purple'
            )
            # Subplot settings
            axes_decomp[0].set_title("Observed")
            axes_decomp[1].set_title("Trend")
            axes_decomp[2].set_title("Seasonal")
            axes_decomp[3].set_title("Residuals")
            for ax in axes_decomp:
                ax.set_xlabel(x_label)
                ax.set_ylabel(tname)
                if show_grid:
                    ax.grid(True, linestyle=':', alpha=0.7)
                else:
                    ax.grid(False)
            plt.suptitle("Seasonal Decomposition", fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.show()
        else:
            # Plot decomposition in the existing figure layout
            decomposition.observed.plot(
                ax=axes[plot_idx],
                label='Observed',
                color='blue'
            )
            decomposition.trend.plot(
                ax=axes[plot_idx + 1],
                label='Trend',
                color='red'
            )
            decomposition.seasonal.plot(
                ax=axes[plot_idx + 2],
                label='Seasonal',
                color='green'
            )
            decomposition.resid.plot(
                ax=axes[plot_idx + 3],
                label='Residuals',
                color='purple'
            )
            # Subplot titles
            axes[plot_idx].set_title("Observed")
            axes[plot_idx + 1].set_title("Trend")
            axes[plot_idx + 2].set_title("Seasonal")
            axes[plot_idx + 3].set_title("Residuals")

            for ax in axes[plot_idx:plot_idx + 4]:
                ax.set_xlabel("Time")
                ax.set_ylabel(x_label)
                if show_grid:
                    ax.grid(True, linestyle=':', alpha=0.7)
                else:
                    ax.grid(False)
            plot_idx += 4

        # Optionally plot residuals in the same figure if user wants
        if show_residual and not decompose_on_sep:
            residual = decomposition.resid.dropna()
            axes[plot_idx].plot(
                df.index[:len(residual)],
                residual,
                label="Residuals",
                color='purple'
            )
            axes[plot_idx].set_title(
                "Residuals (After Trend and Seasonality)"
            )
            axes[plot_idx].set_xlabel("Time")
            axes[plot_idx].set_ylabel("Residuals")
            if show_grid:
                axes[plot_idx].grid(True, linestyle=':', alpha=0.7)
            else:
                axes[plot_idx].grid(False)
            plot_idx += 1

    # Hide any remaining subplots if they exist
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    # Final layout adjustments and show
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

@validate_params({
    "method":[StrOptions({'auto','additive','multiplicative'})]
    })
def get_decomposition_method(
    df,
    value_col,
    dt_col=None,
    max_period=24,
    method='auto',
    min_period=2,
    verbose=0
):
    r"""
    Infer the suitable decomposition method for a given time
    series, based on certain heuristics or user preferences.
    This function helps decide whether to apply an additive
    or multiplicative model, and what seasonal period to use
    for subsequent decomposition steps.

    .. math::
        Y_t = T_t + S_t + \epsilon_t, \quad
        \text{(Additive Model)}

    .. math::
        \log(Y_t) = \log(T_t) + \log(S_t) + \epsilon_t, \quad
        \text{(Multiplicative Model)}

    Here, :math:`T_t` is the trend component, :math:`S_t`
    the seasonal component, and :math:`\epsilon_t` the
    irregular (residual) component [1]_.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing time series data. Must include
        the target variable in `<value_col>` and optionally a
        datetime column `<dt_col>`.
    value_col : str
        The column name of the target time series variable to
        decompose.
    dt_col : str, optional
        The column name representing datetime. If ``None``, the
        index of `df` is assumed to be the time dimension.
    max_period : int, optional
        The maximum seasonal period to check. If
        ``method='auto'``, the function may inspect data for
        possible seasonality up to this limit.
    method : {'auto','additive','multiplicative'}, optional
        The approach for decomposition. If ``'auto'``, the
        function tries to detect whether data is strictly
        positive (favoring multiplicative) or can be
        well-modeled additively. If ``'additive'`` or
        ``'multiplicative'``, uses that model directly.
    min_period : int, optional
        The minimum seasonal period to consider. For instance,
        setting it to ``2`` prevents using a period of ``1``
        (no real seasonality).
    verbose : int, optional
        The level of logging:

        * ``0``: No output.
        * ``1``: Basic info messages.
        * ``2``: More diagnostic details.

    Returns
    -------
    best_method : str
        The inferred model type: ``'additive'`` or
        ``'multiplicative'``.
    best_period : int
        The inferred seasonal period. If the data shows
        minimal seasonality or the detection fails, returns
        a default value of ``1`` or a recognized fallback.

    Notes
    -----
    Choosing between additive and multiplicative models can
    hinge on data behavior. If the time series is strictly
    positive and exhibits increasing variance with time, a
    multiplicative approach is often more suitable [2]_.
    When seasonality does not scale with level, an additive
    approach may suffice.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import get_decomposition_method
    >>> data = {
    ...     'Date': [
    ...         '2020-01-01','2020-02-01','2020-03-01',
    ...         '2020-04-01','2020-05-01'
    ...     ],
    ...     'Sales': [100, 120, 140, 135, 150]
    ... }
    >>> df = pd.DataFrame(data)
    >>> df['Date'] = pd.to_datetime(df['Date'])
    >>> df.set_index('Date', inplace=True)
    >>> mtype, speriod = get_decomposition_method(
    ...     df,
    ...     value_col='Sales',
    ...     method='auto',
    ...     verbose=1
    ... )
    Detected model type: additive
    Detected seasonal period: 1

    See Also
    --------
    seasonal_decompose : Decompose a time series into trend,
        seasonal, and residual components.
    trend_analysis : Detect stationarity and linear trend in
        time series data.

    References
    ----------
    .. [1] Brockwell, P.J., & Davis, R.A. (2016). *Introduction
           to Time Series and Forecasting*. Springer.
    .. [2] Hyndman, R.J., & Athanasopoulos, G. (2021).
           *Forecasting: Principles and Practice* (3rd ed).
           OTexts.
    """

    # Basic validation
    if dt_col is not None and dt_col in df.columns:
        df = df.sort_values(by=dt_col).copy()
    else:
        df = df.sort_index().copy()

    # Extract series
    y = df[value_col]

    # If method is 'additive' or 'multiplicative', no inference needed
    if method.lower() in ['additive','multiplicative']:
        best_method = method.lower()
    else:
        # Attempt 'auto' detection
        # If all positive values, prefer multiplicative
        if (y > 0).all():
            best_method = 'multiplicative'
        else:
            best_method = 'additive'

    # Dummy approach for best seasonal period
    # Real logic may involve spectral analysis or autocorrelation.
    # For demonstration, we just clamp it to 1 or max_period.
    best_period = 1
    if max_period >= min_period and len(y) > max_period:
        # Some naive approach to guess seasonality:
        best_period = min_period

    # Verbosity
    if verbose >= 1:
        print(f"Detected model type: {best_method}")
        print(f"Detected seasonal period: {best_period}")

    return best_method, best_period

@validate_params({
    "method":[StrOptions({'heuristic','variance_comparison'})]
    })
@ensure_pkg(
    "statsmodels", 
    extra="'statsmodels' is required for 'infer_decomposition_method' to proceed."
)
def infer_decomposition_method(
    df,
    dt_col,
    period=12,
    return_components=False,
    view=False,
    figsize=(10, 8),
    method='heuristic',
    verbose=0
):
    r"""
    Determine the best decomposition approach for a time series,
    offering two modes:

    1) ``method='heuristic'``:
       Checks if all data points are strictly positive and
       decides on *multiplicative* if they are, or *additive*
       otherwise. This approach does not evaluate the fit.

    2) ``method='variance_comparison'``:
       Performs both additive and multiplicative decompositions,
       compares residual variances, and chooses the method with
       the smaller residual variance.

    .. math::
        \text{Additive: } Y_t = T_t + S_t + \epsilon_t

    .. math::
        \text{Multiplicative: } Y_t = T_t \times S_t \times
        \epsilon_t \quad\text{or}\quad
        \log(Y_t) = \log(T_t) + \log(S_t) + \epsilon_t.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing time series data. Must include
        the datetime column ``dt_col`` and at least one column
        of values to decompose.
    dt_col : str
        The column name representing datetime. This column is
        set as the index for decomposition.
    period : int, optional
        The seasonal period (frequency) for decomposition.
        Commonly, ``12`` for monthly data showing yearly seasonality.
    return_components : bool, optional
        If ``True``, returns a dictionary of decomposition
        components (``trend``, ``seasonal``, ``residual``).
        Otherwise, returns only the chosen model.
    view : bool, optional
        If ``True``, displays histograms of residuals in the
        ``variance_comparison`` mode to facilitate comparison.
    figsize : tuple of (float, float), optional
        Figure dimensions for residual plots.
    method : {'heuristic','variance_comparison'}, optional
        Strategy for deciding on the decomposition approach:

        * ``'heuristic'``: If all data points are positive,
          uses ``'multiplicative'``; else ``'additive'``.
        * ``'variance_comparison'``: Tries both models,
          compares the variance of residuals, and picks the
          one with smaller residual variance.
    verbose : {0, 1, 2, 3}, optional
        Control the amount of logging:

        * 0 : No messages printed.
        * 1 : Basic info about chosen model and decomposition.
        * 2 : Additional details about data checks.
        * 3 : Very detailed logs, including internal states
          and partial results.

    Returns
    -------
    best_method : str
        The chosen decomposition type: ``'additive'`` or
        ``'multiplicative'``.
    components : dict, optional
        Returned only if ``return_components=True``. Contains
        the keys ``'trend'``, ``'seasonal'``, and ``'residual'``
        mapped to :class:`pandas.Series` objects from the best
        decomposition.

    Notes
    -----
    Selecting an appropriate decomposition model can be crucial
    for capturing both trend and seasonality accurately [1]_.
    In particular, the variance comparison approach ensures a
    more data-driven selection [2]_.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import infer_decomposition_method
    >>> data = {
    ...     'Date': [
    ...         '2020-01-01','2020-02-01','2020-03-01',
    ...         '2020-04-01','2020-05-01'
    ...     ],
    ...     'Sales': [100, 120, 140, 135, 150]
    ... }
    >>> df = pd.DataFrame(data)
    >>> df['Date'] = pd.to_datetime(df['Date'])
    >>> best_model = infer_decomposition_method(
    ...     df, dt_col='Date', period=12,
    ...     method='heuristic', verbose=2
    ... )
    Checking positivity for heuristic method...
    All values are > 0. Using 'multiplicative' model.
    >>> best_model
    'multiplicative'

    See Also
    --------
    seasonal_decompose : Decompose a time series into trend,
        seasonal, and residual components.

    References
    ----------
    .. [1] Brockwell, P.J., & Davis, R.A. (2016). *Introduction
           to Time Series and Forecasting*. Springer.
    .. [2] Hyndman, R.J., & Athanasopoulos, G. (2021).
           *Forecasting: Principles and Practice* (3rd ed).
           OTexts.
    """

    # Validate input: check if df is a proper DataFrame and
    #    that dt_col is a time-series column.
    if verbose >= 2:
        print(
            "Validating DataFrame and time series column..."
        )
    is_frame(
        df,
        df_only=True,
        raise_exception=True,
        objname="Dataframe 'df'"
    )
    is_time_series(
        df,
        time_col=dt_col,
        check_time_interval=False
    )

    # Convert dt_col to datetime if needed, then set as index.
    if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
        if verbose >= 2:
            print(f"Converting {dt_col} to datetime...")
        df[dt_col] = pd.to_datetime(df[dt_col])

    df = df.sort_values(by=dt_col).copy()
    df.set_index(dt_col, inplace=True)

    # Drop missing values before decomposition.
    null_count = df.isnull().sum().sum()
    if null_count > 0 and verbose >= 2:
        print(
            f"Dropping {null_count} null values from the DataFrame."
        )
    df.dropna(inplace=True)

    # Extract the primary data column to decompose.
    #    We'll assume the user has only one main data column
    #    or that they want the first column if there are many.
    #    If there's only dt_col + 1 column, use that. Otherwise,
    #    adapt to user needs (here we keep it simple).
    series = df.iloc[:, 0]
    if verbose >= 3:
        print(f"Series length: {len(series)}")

    # Heuristic method: checks positivity.
    # If the user selected 'heuristic' method, simply check
    #    positivity of the data. If all positive, we use
    #    'multiplicative'; otherwise 'additive'.
    
    if method == 'heuristic':
        if verbose >= 1:
            print(
                "Using 'heuristic' approach to pick between "
                "additive and multiplicative decomposition..."
            )
        if verbose >= 2:
            print("Checking positivity for heuristic method...")

        if (series > 0).all():
            if verbose >= 2:
                print(
                    "All values are > 0. Using 'multiplicative' model."
                )
            best_method = 'multiplicative'
        else:
            if verbose >= 2:
                print(
                    "Some values are <= 0. Using 'additive' model."
                )
            best_method = 'additive'

        # If return_components=False, user only needs method.
        if not return_components:
            if verbose >= 1:
                print(f"Chosen method: {best_method}")
            return best_method

        # Perform actual decomposition with chosen model.
        if verbose >= 2:
            print(
                f"Decomposing series with {best_method} model, period={period}."
            )
        best_decomp = seasonal_decompose(
            series,
            model=best_method,
            period=period
        )

        if verbose >= 1:
            print(f"Chosen method: {best_method}")

        return best_method, {
            'trend': best_decomp.trend,
            'seasonal': best_decomp.seasonal,
            'residual': best_decomp.resid
        }

    # Variance comparison method: do both, pick the best by residual variance.
    # If method == 'variance_comparison', decompose using both
    #    'additive' and 'multiplicative', then compare residual
    #    variances. The model with the lower residual variance
    #    is chosen.
    elif method == 'variance_comparison':
        if verbose >= 1:
            print(
                "Using 'variance_comparison' approach: "
                "Additive vs. Multiplicative..."
            )
        if verbose >= 2:
            print("Decomposing additively...")
        additive_decomp = seasonal_decompose(
            series,
            model='additive',
            period=period
        )

        if verbose >= 2:
            print("Decomposing multiplicatively...")
        multiplicative_decomp = seasonal_decompose(
            series,
            model='multiplicative',
            period=period
        )

        resid_add = additive_decomp.resid.dropna()
        resid_mul = multiplicative_decomp.resid.dropna()

        var_add = np.var(resid_add)
        var_mul = np.var(resid_mul)

        if verbose >= 2:
            print(
                f"Additive residual variance: {var_add:.4f}"
            )
            print(
                f"Multiplicative residual variance: {var_mul:.4f}"
            )

        if var_add < var_mul:
            best_method = 'additive'
            best_decomp = additive_decomp
        else:
            best_method = 'multiplicative'
            best_decomp = multiplicative_decomp

        if verbose >= 1:
            print(
                f"Chosen method by residual variance: {best_method}"
            )

        # If view=True, show histograms of both sets of residuals.
        # Optionally, display a plot of the residual histograms
        # for both methods to visually inspect them.
        if view:
            if verbose >= 1:
                print(
                    "Displaying residual histograms for comparison..."
                )
            fig, axes = plt.subplots(2, 1, figsize=figsize)
            axes[0].hist(
                resid_add,
                bins='auto',
                color='blue',
                alpha=0.7
            )
            axes[0].set_title(
                "Residuals (Additive Decomposition)"
            )

            axes[1].hist(
                resid_mul,
                bins='auto',
                color='green',
                alpha=0.7
            )
            axes[1].set_title(
                "Residuals (Multiplicative Decomposition)"
            )
            plt.tight_layout()
            plt.show()

        # Return either just the best method or also the components.
        if return_components:
            return best_method, {
                'trend': best_decomp.trend,
                'seasonal': best_decomp.seasonal,
                'residual': best_decomp.resid
            }
        else:
            return best_method

    # If the user passes an invalid method.
    else:
        err_msg = (
            f"Unknown 'method'. Must be 'heuristic' or "
            f"'variance_comparison'. Got: {method}"
        )
        if verbose >= 1:
            print(err_msg)
        raise ValueError(err_msg)

@validate_params ({ 
    "method": [StrOptions({'additive', 'multiplicative'})], 
    "strategy": [StrOptions({'STL', 'SDT', 'stl', 'sdt'})]
    })
@ensure_pkg(
    "statsmodels", 
    extra="'statsmodels' is required for 'decompose_ts' to proceed."
)
def decompose_ts(
    df,
    value_col,
    dt_col=None,
    method='additive',
    strategy='STL',
    seasonal_period=12,
    robust=True
):
    r"""
    Decompose a time series into *trend*, *seasonal*, and *residual*
    components while keeping the other features intact.

    In practice, the time series :math:`Y_t` is broken down into
    three main components [1]_ [2]_:

    .. math::
        Y_t = T_t + S_t + R_t

    where :math:`T_t` is the *trend*, :math:`S_t` is the
    *seasonal* component, and :math:`R_t` is the *residual*
    or irregular term. If a *multiplicative* method is used,
    the decomposition can be modeled as:

    .. math::
        Y_t = T_t \times S_t \times R_t,

    or equivalently in logarithms:

    .. math::
        \log(Y_t) = \log(T_t) + \log(S_t) + \log(R_t).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the time series data
        along with potential additional features.
    value_col : str
        The name of the column holding the primary time
        series to be decomposed. This column is used to
        derive :math:`T_t, S_t, R_t`.
    dt_col : str, optional
        The column holding datetime information, if needed
        for validations or indexing. If ``None``, the function
        assumes the time series is already aligned or
        validated.
    method : {'additive', 'multiplicative'}, optional
        The type of decomposition model:

        * ``'additive'``: Assumes data can be decomposed as
          a sum of its components.
        * ``'multiplicative'``: Assumes the product of
          components. Useful if the amplitude of seasonality
          scales with the level of the series.
    strategy : {'STL', 'SDT'}, optional
        Determines how the decomposition is performed:

        * ``'STL'``: Uses :class:`statsmodels.tsa.seasonal.STL`
          (Seasonal-Trend decomposition using LOESS).
        * ``'SDT'``: Uses classic
          :func:`statsmodels.tsa.seasonal.seasonal_decompose`.
    seasonal_period : int, optional
        Defines the periodicity or frequency of the seasonality.
        For example, ``12`` for monthly data exhibiting yearly
        seasonality. Must be an odd integer >= 3.
    robust : bool, optional
        Whether to perform a *robust* STL decomposition
        (only valid for ``strategy='STL'``). With robust
        set to ``True``, the algorithm can better handle
        outliers.

    Returns
    -------
    decomposed_df : pandas.DataFrame
        A new DataFrame containing columns for ``trend``,
        ``seasonal``, and ``residual``, along with the
        original time series column and any other existing
        features in ``df``. This allows further analysis
        without losing context of the other data.

    Notes
    -----
    STL decomposition (``strategy='STL'``) is typically more
    flexible than the classical approach, particularly for
    handling complex seasonal patterns or outliers. The
    seasonal period must be an odd integer >= 3 in STL.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import decompose_ts

    >>> # Generate 100 days of synthetic data
    >>> df = pd.DataFrame({
    ...     'time': pd.date_range(start='2020-01-01',
    ...                           periods=100,
    ...                           freq='D'),
    ...     'value': np.random.randn(100).cumsum() + 5
    ... })
    >>> df.set_index('time', inplace=True)

    >>> # Decompose using STL (Seasonal-Trend decomposition)
    >>> decomposed_df = decompose_ts(
    ...     df,
    ...     value_col='value',
    ...     method='additive',
    ...     strategy='STL',
    ...     seasonal_period=12
    ... )
    >>> print(decomposed_df.head())

    >>> # Decompose using SDT (Seasonal Decomposition of Time Series)
    >>> decomposed_df_sdt = decompose_ts(
    ...     df,
    ...     value_col='value',
    ...     method='multiplicative',
    ...     strategy='SDT',
    ...     seasonal_period=12
    ... )
    >>> print(decomposed_df_sdt.head())

    See Also
    --------
    STL : Seasonal and Trend decomposition using LOESS
        from :mod:`statsmodels.tsa.seasonal`.
    seasonal_decompose : Classic decomposition method
        from :mod:`statsmodels.tsa.seasonal`.

    References
    ----------
    .. [1] Cleveland, R.B., Cleveland, W.S., McRae, J.E., & Terpenning, I.
           (1990). STL: A Seasonal-Trend Decomposition Procedure Based
           on LOESS. *Journal of Official Statistics*, 6(1), 3-73.
    .. [2] Brockwell, P.J., & Davis, R.A. (2016). *Introduction to Time
           Series and Forecasting*. Springer.
    """

    # Validate and extract the target time series
    # from the user-specified column <value_col>.
    ts, _ = validate_target_in(
        df,
        value_col,
        error='raise',
        verbose=0
    )
    tname = ts.name

    # Ensure that the seasonal period is an odd integer >= 3.
    # If user provides an even number, increment it by 1.
    if seasonal_period % 2 == 0:
        seasonal_period += 1
    if seasonal_period < 3:
        raise ValueError(
            "The seasonal period must be an odd integer >= 3. "
            f"Got {seasonal_period}."
        )

    # Lowercase the strategy for consistency.
    strategy = str(strategy).lower()

    # Perform the decomposition according to the chosen strategy.
    if strategy == 'stl':
        # STL decomposition:
        # This algorithm uses LOESS to separately model seasonality
        # and trend. It is more robust to outliers when 'robust=True'.
        stl = STL(
            ts,
            seasonal=seasonal_period,
            trend=seasonal_period,
            robust=robust
        )
        result = stl.fit()
        decomposed_df = pd.DataFrame({
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        })
    elif strategy == 'sdt':
        # "SDT" stands for the classical Seasonal Decomposition
        # of Time series using either an 'additive' or
        # 'multiplicative' model.
        result = seasonal_decompose(
            ts,
            model=method,
            period=seasonal_period
        )
        decomposed_df = pd.DataFrame({
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        })
    else:
        # If strategy is invalid, raise an error.
        raise ValueError(
            "Invalid strategy. Choose either 'STL' or 'SDT'. "
            f"Got: {strategy}"
        )

    # Combine the newly derived components with the original
    # data features. We place them first in 'decomposed_df',
    # then add the original time series to keep a direct
    # reference, and finally append all other columns.
    decomposed_df[tname] = df[tname]

    # Iterate through original columns and add them if
    # they are not the target time series column.
    for col in df.columns:
        if col != tname:
            decomposed_df[col] = df[col]

    return decomposed_df

@validate_params ({
    "window": [Interval(Integral, 0, None, closed="neither")], 
    "scaler":  [StrOptions({'z-norm', 'minmax'}), None], 
    "lags": [Interval(Integral, 1, None, closed="left")], 
    "holiday_df": ['array-like', None]
    })
def ts_engineering(
    df,
    value_col,
    dt_col=None,
    lags=5,
    window=7,
    diff_order=1,
    seasonal_period=None,
    apply_fourier=False,
    holiday_df=None,
    robust_diff=True,
    scaler='z-norm',
    **kwargs
):
    r"""
    Perform feature engineering on a time series to create
    relevant predictors for machine learning models. The
    function can generate lag features, rolling statistics,
    differences, Fourier transforms, holiday indicators, and
    applies optional scaling.

    Specifically, let :math:`X_t` be the time series at time
    :math:`t`. This function will create features such as
    :math:`X_{t-1}, \dots, X_{t-l}`, rolling means
    :math:`\frac{1}{w}\sum_{i=0}^{w-1}X_{t-i}`, and so on,
    enabling predictive models to capture temporal
    dependencies [1]_.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the time series data
        plus any additional columns.
    value_col : str
        The name of the column in ``df`` representing the
        primary time series for which features are derived.
    dt_col : str, optional
        The name of the datetime column, if the DataFrame
        index is not already a time index. If provided, it
        may be used for indexing or validations.
    lags : int, optional
        Number of lag features to create. For example,
        ``lags=5`` yields columns for
        :math:`X_{t-1}, X_{t-2}, \ldots, X_{t-5}`.
    window : int, optional
        Window size for rolling statistics. For example,
        a 7-day rolling average if ``window=7``.
    diff_order : int, optional
        Order of differencing to apply to the target time
        series. ``0`` means no differencing, ``1`` means
        first differencing, etc.
    seasonal_period : int, optional
        Specifies the seasonal period for seasonal
        differencing. For example, ``12`` for monthly data
        exhibiting yearly seasonality.
    apply_fourier : bool, optional
        If ``True``, computes a discrete Fourier transform
        of the time series and includes its magnitudes as
        additional features.
    holiday_df : pandas.DataFrame, optional
        DataFrame containing holiday dates for adding a
        holiday indicator feature. The DataFrame should
        have a column named, for instance, ``'date'``
        listing holiday dates.
    robust_diff : bool, optional
        Placeholder flag indicating whether robust
        differencing should be used. Implementation details
        may vary, but in this snippet it is not used
        explicitly.
    scaler : {'z-norm', 'minmax', None}, optional
        The scaling approach for numeric features:

        * ``'z-norm'``: Apply standard normalization
          :math:`Z = (X - \mu)/\sigma`.
        * ``'minmax'``: Rescale to [0, 1].
        * ``None``: No scaling is applied.
    **kwargs : dict, optional
        Additional parameters for customization, e.g.,
        methods to handle missing values.

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with newly created time-series features.
        This includes lag columns, rolling statistics,
        differenced series, Fourier terms, holiday indicator,
        time-based components (year, month, day, etc.), and
        optionally scaled numeric features.

    Notes
    -----
    Feature engineering in time series is crucial for capturing
    temporal dependencies and seasonality. Lag features help
    machine learning models exploit autocorrelations, rolling
    windows expose local trends, and Fourier terms can capture
    complex seasonality beyond simple differencing [2]_.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gofast.utils.ts_utils import ts_engineering

    >>> # Generate synthetic daily data
    >>> idx = pd.date_range(
    ...     start='2020-01-01',
    ...     periods=30,
    ...     freq='D'
    ... )
    >>> data = {
    ...     'Date': idx,
    ...     'Sales': np.random.randint(50, 150, len(idx))
    ... }
    >>> df = pd.DataFrame(data)
    >>> df.set_index('Date', inplace=True)

    >>> # Perform feature engineering with 3 lags, 7-day window,
    ... # first differencing, and scaled features
    >>> df_features = ts_engineering(
    ...     df,
    ...     value_col='Sales',
    ...     lags=3,
    ...     window=7,
    ...     diff_order=1,
    ...     scaler='z-norm'
    ... )
    >>> df_features.head()

    See Also
    --------
    pandas.DataFrame.shift : Used for lagging.
    pandas.DataFrame.rolling : Used for rolling statistics.
    scipy.fft.fft : Discrete Fourier transform for capturing
        high-frequency seasonalities.

    References
    ----------
    .. [1] Box, G.E.P., Jenkins, G.M., Reinsel, G.C., &
           Ljung, G.M. (2015). *Time Series Analysis:
           Forecasting and Control*. John Wiley & Sons.
    .. [2] Hyndman, R.J., & Athanasopoulos, G. (2021).
           *Forecasting: Principles and Practice* (3rd ed).
           OTexts.
    """

    # 1)Validate and process the datetime column
    # (set index, format to datetime, etc.)
    # If a datetime column is specified but not the index,
    #  we could process it.
    df, dt_col = ts_validator(
        df,
        dt_col=dt_col,
        to_datetime='auto',
        as_index=True,
        error="raise",
        return_dt_col=True,
    )
    # 2) Validate and extract the target time series using
    #    the helper. This ensures <value_col> exists.
    ts, _ = validate_target_in(
        df,
        value_col,
        error='raise',
        verbose=0
    )
    tname = ts.name

    # 3) Create time-based features from the index for
    #    daily or other frequencies. This helps many ML
    #    models to capture cyclical patterns:
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_week'] = df.index.dayofweek  # 0=Monday,...,6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['quarter'] = df.index.quarter
    # For hourly data only if index.freq='H'. If not, default=0.
    df['hour'] = df.index.hour if hasattr(df.index, 'freq') and \
        str(df.index.freq) == 'H' else 0

    # 4) Create holiday indicator if holiday_df is provided.
    df['is_holiday'] = 0
    if holiday_df is not None:
        is_frame(holiday_df,
                 df_only=True,
                 raise_exception=True, 
                 objname="Holiday df"
                )
        # Check if user-supplied holiday_df has a 'date' column.
        # Mark row as holiday if index in holiday_df['date'].
        df['is_holiday'] = df.index.isin(holiday_df['date']).astype(int)

    # 5) Generate lag features up to <lags>.
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = ts.shift(lag)

    # 6) Compute rolling statistics over a specified window.
    #    e.g. rolling mean and rolling std for capturing
    #    local trend and volatility.
    df[f'rolling_mean_{window}'] = ts.rolling(window=window).mean()
    df[f'rolling_std_{window}'] = ts.rolling(window=window).std()

    # 7) Differencing: Remove certain non-stationary behavior.
    #    diff_order=1 does X[t] - X[t-1].
    if diff_order > 0:
        df[f'{tname}_diff'] = ts.diff(diff_order)

    # 8) Seasonal differencing: If a known seasonal period is
    #    provided, create a differenced series over that lag.
    if seasonal_period and seasonal_period > 0:
        df[f'{tname}_seasonal_diff'] = ts.diff(seasonal_period)

    # 9) Optional Fourier transform to capture periodicities
    #    that differ from the simpler approach. We apply FFT
    #    to fill missing with 0 for stability.
    if apply_fourier:
        fft_values = fft(ts.fillna(0))
        # The magnitude of the first half (since the second
        # half is often symmetric for real signals).
        half_len = len(ts) // 2
        fft_features = np.abs(fft_values[:half_len])
        # Create columns named fft_1, fft_2, ...
        fft_columns = [f'fft_{i}' for i in range(1, half_len + 1)]
        df[fft_columns] = pd.DataFrame(fft_features).T

    # 10) Handle missing values. By default, we do a forward
    #     fill, then drop any leftover rows if needed.
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    # 11) Apply optional scaling to numeric columns.
    #     By default 'z-norm' uses StandardScaler,
    #     'minmax' uses MinMaxScaler, or None means no scaling.
    if scaler is not None:
        scaler = (StandardScaler() if scaler == 'z-norm'
                  else MinMaxScaler())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Fit and transform only numeric columns
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index
        )
        df[numeric_cols] = df_scaled

    return df


def prepare_ts_df(
    df,
    dt_col=None,
    set_index=True,
    error='raise',
    use_smart_ts_formatter=False,
    verbose=0
):
    r"""
    Prepare a DataFrame for time series operations by ensuring it
    has a valid datetime index or column. The function checks
    whether the index is already datetime or, if not, whether a
    specified datetime column exists. Under the hood, it can also
    rely on ``ts_validator`` to auto-convert or raise errors as
    needed.

    .. math::
        \text{Time Series DF} \to \text{Datetime Index} \,|\,
        \text{Datetime Column}

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the series data. Must
        either have a datetime index or a valid datetime column
        specified via ``dt_col``.
    dt_col : str, optional
        The name of the column to be used as datetime if the
        index is not already datetime. If ``None``, checks
        whether the existing index is datetime-like.
    set_index : bool, optional
        If ``True``, sets the specified datetime column as
        the index after conversion. Otherwise, the DataFrame’s
        structure remains unchanged.
    error : {'raise', 'warn', 'ignore'}, optional
        Behavior when no valid datetime is found:

        * ``'raise'``: Raises a ValueError.
        * ``'warn'``: Issues a warning and tries to proceed.
        * ``'ignore'``: Silently continues, returning ``df``
          unchanged.
    use_smart_ts_formatter : bool, optional
        If ``True``, calls the internal
        :func:`ts_validator` function to automatically detect
        or convert datetime columns. This can handle various
        formats, but if it fails, it depends on the ``error``
        setting.
    verbose : int, optional
        Verbosity level:

        * ``0``: No messages.
        * ``1``: Basic info messages.
        * ``2+``: More detailed debug messages (implementation
          dependent).

    Returns
    -------
    df : pandas.DataFrame
        The resulting DataFrame, ensuring that its index or
        a specified column is a proper datetime type. If
        ``set_index=True``, the DataFrame index becomes
        datetime-based.

    Notes
    -----
    This function is critical in pipelines where subsequent
    transformations or model fitting expect a time-based
    index [1]_. If the index is not a valid datetime or if
    no column is provided, it can raise warnings or errors
    depending on the configuration.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import prepare_ts_df
    >>> data = {
    ...     'Date': [
    ...         '2021-01-01', '2021-01-02', '2021-01-03'
    ...     ],
    ...     'Value': [10, 15, 20]
    ... }
    >>> df = pd.DataFrame(data)
    >>> # The index is not yet datetime
    >>> df_out = prepare_ts_df(
    ...     df,
    ...     dt_col='Date',
    ...     set_index=True,
    ...     error='raise'
    ... )
    >>> df_out.index
    DatetimeIndex(['2021-01-01', '2021-01-02', '2021-01-03'],
                  dtype='datetime64[ns]', freq=None)

    See Also
    --------
    ts_validator : More comprehensive time series validation
        and conversion utility.

    References
    ----------
    .. [1] Box, G.E.P., Jenkins, G.M., Reinsel, G.C., &
           Ljung, G.M. (2015). *Time Series Analysis:
           Forecasting and Control*. John Wiley & Sons.
    """

    # 1) If the index is already a datetime, we may do nothing.
    if pd.api.types.is_datetime64_any_dtype(df.index):
        if verbose >= 1:
            print("Index is already a datetime object.")
        if not set_index:
            # If user doesn't want it as index, revert the index
            # to a column. This is rare but supported.
            df.reset_index(inplace=True)
        return df

    # 2) If user wants the 'smart_ts_formatter', call ts_validator.
    if use_smart_ts_formatter:
        # Pass relevant parameters for auto-conversion. This can
        # handle or raise errors as needed.
        df = ts_validator(
            df=df,
            dt_col=dt_col,
            to_datetime='auto',
            as_index=set_index,
            error=error,
            return_dt_col=False,
            verbose=verbose
        )
        return df

    # 3) If the index is not datetime but a datetime column is
    # provided, set it. If no column is found, raise or handle
    # per the 'error' param.
    if dt_col is not None:
        if dt_col not in df.columns:
            raise ValueError(
                f"Column '{dt_col}' not found in DataFrame."
            )
        # Convert to datetime if needed
        df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
        if df[dt_col].isnull().any():
            raise ValueError(
                f"Column '{dt_col}' contains invalid date "
                "formats that could not be converted."
            )
        # Optionally set it as index
        if set_index:
            df.set_index(dt_col, inplace=True)
        if verbose >= 1:
            print(
                f"Column '{dt_col}' has been set as the index "
                "and converted to datetime."
            )
        return df

    # 4) If no dt_col is specified, decide action based on 'error'.
    if error == 'raise':
        raise ValueError(
            "Index is not a datetime object, and no 'dt_col' "
            "was specified."
        )
    elif error == 'warn':
        warnings.warn(
            "Index is not a datetime object, and no 'dt_col' "
            "was specified. Returning DataFrame unchanged."
        )
        return df
    elif error == 'ignore':
        # No action taken, just return as is.
        return df

    # 5) If for some reason no valid approach was found, return df.
    if verbose >= 1:
        print(
            "No valid datetime index or column found. "
            "Returning the DataFrame as is."
        )
    return df

@ensure_pkg(
    "statsmodels", 
    extra="'stasmodels' is required for 'ts_corr_analysis' to proceed."
)
def ts_corr_analysis(
    df,
    dt_col,
    value_col,
    lags=2,
    features=None,
    view_acf_pacf=True,
    view_cross_corr=True,
    fig_size=(14, 6),
    show_grid=True,
    cross_corr_on_sep=False,
    verbose=0,
):
    r"""
    Perform correlation analysis on a time series dataset,
    including autocorrelation (ACF), partial autocorrelation
    (PACF), and cross-correlation with external features.

    .. math::
        \rho(h) = \frac{E\big[(X_t - \mu)(X_{t+h} - \mu)\big]}
        {\sigma^2},

    where :math:`h` denotes the lag, :math:`\mu` the mean, and
    :math:`\sigma^2` the variance of the time series [1]_.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing time series data.
        Must contain at least one time-like column or index.
    dt_col : str
        Column name representing the datetime dimension (e.g.
        "DateTime" or "timestamp").
    value_col : str
        Name of the primary target variable column (e.g.
        "sales").
    lags : int, optional
        Number of time lags for ACF/PACF analysis. Default is 2.
    features : list of str, optional
        List of external feature columns to analyze for
        cross-correlation with ``value_col``. If ``None``,
        uses all non-target, non-datetime columns in ``df``.
    view_acf_pacf : bool, optional
        Whether to generate and display ACF and PACF plots.
    view_cross_corr : bool, optional
        Whether to visualize cross-correlations for selected
        external features.
    fig_size : tuple of (float, float), optional
        Figure dimension for ACF/PACF plots and optionally
        cross-correlation bars. Default is (14, 6).
    show_grid : bool, optional
        Whether to display gridlines in the plots. Default
        is True.
    cross_corr_on_sep : bool, optional
        If ``True``, plots cross-correlation results in a
        separate figure. If ``False`` and
        ``view_cross_corr=True``, it appends the cross-corr
        plot to the same figure containing ACF/PACF (if
        feasible).
    verbose : int, optional
        Verbosity level:

        * ``0``: No console messages.
        * ``1``: Basic info messages.
        * ``2``: More detailed logs.

    Returns
    -------
    results : dict
        Dictionary of correlation metrics:

        * ``'acf_values'``: ACF values up to ``lags``.
        * ``'pacf_values'``: PACF values up to ``lags``.
        * ``'cross_corr'``: Cross-correlation coefficients
          (and p-values) for external features.

    Notes
    -----
    This function can aid in both univariate and multivariate
    time series analysis. By assessing ACF and PACF, users
    glean insights about autocorrelation structure (e.g.
    potential AR or MA terms in ARIMA). Cross-correlation
    helps identify external predictors correlated with the
    target [2]_.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import ts_corr_analysis
    >>> data = {
    ...     'Date': [
    ...         '2021-01-01','2021-01-02','2021-01-03',
    ...         '2021-01-04','2021-01-05'
    ...     ],
    ...     'Sales': [10, 12, 14, 13, 15],
    ...     'Promo': [0, 1, 0, 1, 1]
    ... }
    >>> df = pd.DataFrame(data)
    >>> results = ts_corr_analysis(
    ...     df,
    ...     dt_col='Date',
    ...     value_col='Sales',
    ...     lags=1,
    ...     features=['Promo'],
    ...     view_acf_pacf=True,
    ...     view_cross_corr=True,
    ...     verbose=1
    ... )
    Performing ACF and PACF analysis...
    Target variable: Sales
    Datetime column: Date
    Cross-correlation features: ['Promo']
    Performing cross-correlation analysis...
    CrossCorrResults > item 1: correlation=0.2890, p_value=0.6367

    See Also
    --------
    statsmodels.graphics.tsaplots.plot_acf : Plot the
        autocorrelation function.
    statsmodels.graphics.tsaplots.plot_pacf : Plot the
        partial autocorrelation function.

    References
    ----------
    .. [1] Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M.
           (2015). *Time Series Analysis: Forecasting and Control*.
           John Wiley & Sons.
    .. [2] Wei, W.W.S. (2006). *Time Series Analysis: Univariate
           and Multivariate Methods*. Addison Wesley.
    """

    # Step 1: Validate DataFrame structure.
    is_frame(
        df,
        df_only=True,
        raise_exception=True,
        objname="DataFrame 'df'"
    )

    # Step 2: Validate target column and extract it.
    target, df = validate_target_in(
        df,
        value_col,
        verbose=verbose
    )

    # Step 3: Ensure <dt_col> is valid and possibly set as index.
    df, dt_col = ts_validator(
        df=df,
        dt_col=dt_col,
        to_datetime='auto',
        as_index=False,
        error="raise",
        return_dt_col=True,
        verbose=verbose
    )

    # If verbose=1 or higher, let the user know.
    if verbose >= 1:
        print("Performing ACF and PACF analysis...")

    # Step 4: Manage the external features. If none are
    # provided, use all columns except the target and dt_col.
    features = columns_manager(features, empty_as_none=True)
    if features is not None:
        exist_features(
            df,
            features=features,
            name="Features for cross-correlation"
        )
    else:
        features = [
            col for col in df.columns
            if col not in [value_col, dt_col]
        ]

    if verbose >= 1:
        print(f"Target variable: {value_col}")
        print(f"Datetime column: {dt_col}")
        if features:
            print(f"Cross-correlation features: {features}")

    # ACF/PACF placeholders
    acf_values = None
    pacf_values = None

    # Step 5: Plot ACF/PACF if requested.
    if view_acf_pacf:
        # Check if cross-corr is on the same figure or separate.
        if view_cross_corr and not cross_corr_on_sep:
            fig = plt.figure(
                figsize=(fig_size[0], fig_size[1] * 1.5)
            )
            gs = fig.add_gridspec(
                2,
                2,
                height_ratios=[1, 0.7]
            )
            ax_acf = fig.add_subplot(gs[0, 0])
            ax_pacf = fig.add_subplot(gs[0, 1])
            ax_cross_corr = fig.add_subplot(gs[1, :])
        else:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=fig_size
            )
            ax_acf, ax_pacf = axes
            ax_cross_corr = None

        # ACF plot
        plot_acf(
            target,
            lags=lags,
            ax=ax_acf
        )
        ax_acf.set_title("Autocorrelation Function (ACF)")
        ax_acf.set_xlabel("Lags")
        ax_acf.set_ylabel("ACF")
        ax_acf.grid(
            show_grid,
            linestyle=":",
            alpha=0.7
        )

        # PACF plot
        plot_pacf(
            target,
            lags=lags,
            ax=ax_pacf,
            method='ywm'
        )
        ax_pacf.set_title("Partial Autocorrelation (PACF)")
        ax_pacf.set_xlabel("Lags")
        ax_pacf.set_ylabel("PACF")
        ax_pacf.grid(
            show_grid,
            linestyle=":",
            alpha=0.7
        )

        # Dummy placeholders for ACF/PACF values. You can
        # refine by computing them numerically, but
        # statsmodels also provides them if needed.
        acf_values = None
        pacf_values = None

    # Step 6: Cross-correlation analysis for external features.
    cross_corr_results = {}
    if features:
        if verbose >= 1:
            print("Performing cross-correlation analysis...")

        # For each feature, compute Pearson correlation with
        # the target. This is a zero-lag cross-correlation.
        for feat in features:
            correlation, p_value = pearsonr(
                target, df[feat]
            )
            cross_corr_results[feat] = {
                'correlation': correlation,
                'p_value': p_value
            }
            if verbose >= 2:
                print(
                    f"Cross-correlation with {feat}: "
                    f"r={correlation:.4f}, p={p_value:.4f}"
                )

        # Plot cross-correlation if requested.
        if view_cross_corr:
            if cross_corr_on_sep:
                # Separate figure for cross-corr bar chart.
                fig_cc, ax_cc_sep = plt.subplots(
                    figsize=(fig_size[0], fig_size[1] // 2)
                )
                ax_cc_sep.bar(
                    features,
                    [
                        cross_corr_results[f]['correlation']
                        for f in features
                    ],
                    color='skyblue'
                )
                ax_cc_sep.set_title(
                    "Cross-Correlation with External Features"
                )
                ax_cc_sep.set_xlabel("Features")
                ax_cc_sep.set_ylabel("Correlation Coefficient")
                ax_cc_sep.grid(
                    show_grid,
                    linestyle=":",
                    alpha=0.7
                )
                plt.xticks(rotation=45)
            elif ax_cross_corr is not None:
                # Plot cross-corr on the same figure if the
                # axes is defined.
                ax_cross_corr.bar(
                    features,
                    [
                        cross_corr_results[f]['correlation']
                        for f in features
                    ],
                    color='skyblue'
                )
                ax_cross_corr.set_title(
                    "Cross-Correlation with External Features"
                )
                ax_cross_corr.set_xlabel("Features")
                ax_cross_corr.set_ylabel("Correlation Coefficient")
                ax_cross_corr.grid(
                    show_grid,
                    linestyle=":",
                    alpha=0.7
                )
                plt.xticks(rotation=45)

    # If we plotted ACF/PACF, finalize layout.
    if view_acf_pacf:
        plt.tight_layout()
        plt.show()

    # Step 7: Compile and display results.
    results = {
        'acf_values': acf_values,
        'pacf_values': pacf_values,
        'cross_corr': cross_corr_results
    }
    summary = ResultSummary(
        "CrossCorrResults",
        flatten_nested_dicts=False
    )
    summary.add_results(results['cross_corr'])

    if verbose >= 1:
        print(summary)

    return results

@ensure_pkg(
    "statsmodels", 
    extra="'statsmodels' is required for 'infer_decomposition_method' to proceed.", 
    partial_check=True, 
    condition=lambda *args, **kws: ( 
        kws.get("method")=='detrending' 
        and kws.get("method")=='stl'
        )
)
def transform_stationarity(
    df,
    dt_col=None,
    value_col=None,
    method="differencing",
    order=1,
    seasonal_period=None,
    detrend_method="linear",
    view=True,
    fig_size=(12, 6),
    show_grid=True,
    drop_original=True,
    reset_index=False,
    verbose=0
):
    r"""
    Perform stationarity transformations on a time series
    dataset by applying differencing, variance stabilization,
    or detrending. This function helps reduce non-stationary
    components (trends, seasonal effects) to align the data
    with time-series modeling assumptions [1]_.

    .. math::
        \Delta^d (X_t) = X_t - X_{t-d},

    for differencing, and

    .. math::
        Y_t = \log(X_t),

    for a logarithmic transform.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing time series data. The
        index or a column should correspond to time.
    dt_col : str, optional
        Column name representing the datetime dimension
        (e.g., "Date"). If ``None``, the function assumes the
        index is already datetime-like.
    value_col : str, optional
        Name of the target variable column (e.g., "Sales").
        This column is transformed to promote stationarity.
    method : {'differencing', 'log', 'sqrt', 'detrending'}, optional
        The transformation method:

        * ``'differencing'``: Remove trends or cycles by
          subtracting lagged values.
        * ``'log'``: Apply a log transform for variance
          stabilization (positive values only).
        * ``'sqrt'``: Apply a square-root transform (non-negative
          values only).
        * ``'detrending'``: Remove trend either by linear
          regression or STL decomposition.
    order : int, optional
        Order of differencing if ``method='differencing'``.
        For example, ``1`` for first differencing,
        ``2`` for second differencing, etc.
    seasonal_period : int, optional
        Seasonal period for seasonal differencing or STL
        decomposition. For instance, 12 in monthly data
        with annual seasonality.
    detrend_method : {'linear', 'stl'}, optional
        Method for detrending if ``method='detrending'``:

        * ``'linear'``: Fit a linear regression to the series
          and subtract the fitted line.
        * ``'stl'``: Use STL (Seasonal and Trend decomposition)
          to remove the estimated trend component.
    view : bool, optional
        If ``True``, displays plots of original and transformed
        data in a 2-row subplot.
    fig_size : tuple of (float, float), optional
        The figure width and height in inches for the optional
        plots. Default is (12, 6).
    show_grid : bool, optional
        Whether to show gridlines in the plots. Default True.
    drop_original : bool, optional
        Whether to keep the original column in the returned
        DataFrame. If True, only the transformed column is
        kept (besides other unrelated columns).
    reset_index : bool, optional
        If True, resets the DataFrame index before returning.
    verbose : int, optional
        Verbosity level:

        * 0 : No output
        * 1 : Basic info about transformations
        * 2+ : More detailed logs (not fully implemented here).

    Returns
    -------
    transformed_df : pandas.DataFrame
        A DataFrame containing the transformed series in
        a new column named ``'<value_col>_transformed'``.
        If ``drop_original=False``, it also includes the
        original series in column ``'<value_col>'``.

    Notes
    -----
    Stationarity transformations aim to remove or lessen
    trends and periodic components, aligning data with the
    assumptions of many time-series models such as ARIMA [2]_.
    Log and square-root transforms assume positive values,
    so care must be taken with zero or negative data.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import transform_stationarity
    >>> data = {
    ...     'Date': [
    ...         '2021-01-01', '2021-01-02', '2021-01-03',
    ...         '2021-01-04', '2021-01-05'
    ...     ],
    ...     'Sales': [10, 12, 14, 13, 15]
    ... }
    >>> df = pd.DataFrame(data)
    >>> df['Date'] = pd.to_datetime(df['Date'])
    >>> df.set_index('Date', inplace=True)
    >>> # Perform first-order differencing and plot
    >>> df_trans = transform_stationarity(
    ...     df,
    ...     value_col='Sales',
    ...     method='differencing',
    ...     order=1,
    ...     view=True,
    ...     verbose=1
    ... )
    Target variable: Sales
    Datetime column: Date
    Transformation method: differencing
    Applying first-order differencing with order=1.

    See Also
    --------
    STL : A robust method for seasonal-trend decomposition.
    ts_engineering : Broader feature creation for time-series,
        including lags and rolling windows.

    References
    ----------
    .. [1] Brockwell, P.J. & Davis, R.A. (2016). *Introduction to
           Time Series and Forecasting*. Springer.
    .. [2] Hyndman, R.J. & Athanasopoulos, G. (2021).
           *Forecasting: Principles and Practice* (3rd ed).
           OTexts.
    """

    # 1) Validate input DataFrame structure.
    is_frame(
        df,
        df_only=True,
        raise_exception=True,
        objname="DataFrame 'df'"
    )

    # 2) Validate and extract the target column.
    target, df = validate_target_in(df, value_col)
    tname = target.name

    # 3) Ensure the datetime column is valid and set as index.
    #    ts_validator can raise an error if it fails.
    df, dt_col = ts_validator(
        df=df,
        dt_col=dt_col,
        to_datetime='auto',
        as_index=True,
        error="raise",
        return_dt_col=True,
        verbose=verbose
    )
    # Align the extracted target with the updated index
    target.index = df.index

    # Optional logging if verbose >=1
    if verbose >= 1:
        print(f"Target variable: {tname}")
        print(f"Datetime column: {dt_col}")
        print(f"Transformation method: {method}")

    # 4) Apply the transformation
    if method == "differencing":
        # Seasonal differencing if seasonal_period is given
        if seasonal_period:
            if verbose >= 1:
                print(
                    f"Applying seasonal differencing "
                    f"with period={seasonal_period}."
                )
            transformed_data = target.diff(seasonal_period).dropna()
        else:
            if verbose >= 1:
                print(
                    f"Applying first-order differencing "
                    f"with order={order}."
                )
            transformed_data = target.diff(order).dropna()

    elif method == "log":
        if verbose >= 1:
            print("Applying logarithmic transformation.")
        if (target <= 0).any():
            raise ValueError(
                "Log transformation cannot be applied "
                "to non-positive values."
            )
        transformed_data = np.log(target)

    elif method == "sqrt":
        if verbose >= 1:
            print("Applying square root transformation.")
        if (target < 0).any():
            raise ValueError(
                "Square root transformation cannot be "
                "applied to negative values."
            )
        transformed_data = np.sqrt(target)

    elif method == "detrending":
        if detrend_method == "linear":
            if verbose >= 1:
                print("Applying linear detrending.")
            time_index = np.arange(len(target)).reshape(-1, 1)
            # Fit a linear polynomial to the data
            trend = np.polyfit(
                time_index.flatten(),
                target.values,
                deg=1
            )
            # Evaluate the polynomial
            trend_line = np.polyval(
                trend,
                time_index
            )
            transformed_data = target - trend_line.flatten()
        elif detrend_method == "stl":
            if verbose >= 1:
                print("Applying STL detrending.")
            # If user doesn't specify a seasonal_period, assume 7
            # (weekly) or some fallback
            stl = STL(
                target,
                period=seasonal_period or 7
            )
            result = stl.fit()
            transformed_data = result.resid
        else:
            raise ValueError(
                f"Invalid detrend_method: {detrend_method}"
            )
    else:
        raise ValueError(f"Invalid method: {method}")

    # 5) Visualize if requested
    if view:
        plt.figure(figsize=fig_size)

        # Original data
        plt.subplot(2, 1, 1)
        plt.plot(
            target,
            label="Original Data",
            color="blue"
        )
        plt.title("Original Time Series")
        plt.xlabel("Time")
        plt.ylabel(tname)
        plt.grid(
            show_grid,
            linestyle=":",
            alpha=0.7
        ) if show_grid else plt.grid(False)

        # Transformed data
        plt.subplot(2, 1, 2)
        plt.plot(
            transformed_data,
            label=f"Transformed ({method})",
            color="green"
        )
        plt.title(f"Transformed Series ({method})")
        plt.xlabel("Time")
        plt.ylabel(f"{tname} (Transformed)")
        plt.grid(
            show_grid,
            linestyle=":",
            alpha=0.7
        ) if show_grid else plt.grid(False)

        plt.tight_layout()
        plt.show()

    # 6) Return a DataFrame with the transformed column
    transformed_df = df.copy()

    # If user wants to keep original column
    if not drop_original:
        transformed_df[tname] = target

    # Create a new column storing the transformed data
    transformed_df[f"{tname}_transformed"] = transformed_data

    if reset_index:
        transformed_df.reset_index(inplace=True)

    return transformed_df

@validate_params ({
    "split_type": [StrOptions({"simple", "base", "cv"})], 
    "test_ratio": [str, Interval(Real, 0, 1, closed='both'), None], 
    "n_splits": [Integral], 
    "gap": [Interval(Integral, 0, None, closed="left")]
    })
def ts_split(
    df,
    dt_col=None,
    value_col=None,
    split_type="simple",
    test_ratio=None,
    n_splits=5,
    gap=0,
    train_start=None,
    train_end=None,
    verbose=0
):
    r"""
    Perform a time-based split on a time series dataset
    for either a simple train-test partition or
    cross-validation.

    In time-series modeling, it is critical to maintain
    chronological ordering [1]_. Let :math:`\{x_t\}_{t=1}^N`
    be the time-ordered observations. A simple time-based
    split partitions the data at some time index
    :math:`k`:

    .. math::
        \text{Train set}: \{x_t | t \le k \}, \quad
        \text{Test set}: \{x_t | t > k \}.

    Cross-validation (`"cv"`) uses multiple splits,
    iteratively moving the boundary to create overlapping
    train sets for model training and test sets for
    validation [2]_.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing time series data.
        Must include a column (or index) for time
        references.
    dt_col : str, optional
        The name of the datetime column if the index is
        not already datetime. If provided, the function
        ensures it is valid and can parse it as datetime
        if needed.
    value_col : str, optional
        The name of the target variable column (e.g.,
        "sales"). Primarily for logging or reference;
        it is not required for the split logic itself.
    split_type : {'simple', 'cv'}, optional
        Type of split:

        * ``'simple'`` or ``'base'``: Splits the DataFrame into a
          single train and test set based on time or
          specified rows.
        * ``'cv'``: Constructs a generator for
          time-series cross-validation using
          :class:`sklearn.model_selection.TimeSeriesSplit`.
    test_ratio : float, optional
        For a simple split, if set, this denotes the
        fraction of rows allocated to the test set
        (:math:`0 < \text{test_ratio} < 1`). If not
        specified, ``train_end`` can determine the
        boundary. Not used for cross-validation.
    n_splits : int, optional
        Number of splits for cross-validation if
        ``split_type='cv'``. Defaults to 5.
    gap : int, optional
        Gap (number of points) between train and test
        sets in cross-validation. Defaults to 0.
    train_start : str, optional
        If set, the earliest date to include in the
        training set for a simple split. Should be a
        string convertible by pandas to a datetime, e.g.,
        "2021-01-01".
    train_end : str, optional
        If set, the last date to include in the training
        set for a simple split. The subsequent rows
        become the test set if older than ``train_end``.
    verbose : int, optional
        Verbosity level:

        * 0: No messages.
        * 1: Basic logs on split info.
        * 2: More detailed logs (including indices for
          cross-validation splits).

    Returns
    -------
    splits : tuple or generator
        * If ``split_type='simple'``, returns a tuple
          ``(train_df, test_df)``.
        * If ``split_type='cv'``, returns a
          :class:`TimeSeriesSplit` generator yielding
          indices for train/test.

    Notes
    -----
    Maintaining time order in training and testing sets
    is essential to avoid leakage of future information
    into model training. Cross-validation further
    generalizes the idea by repeated train-test
    sub-sampling in an expanding window manner, shifting
    the boundary forward for each split.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import ts_split
    >>> data = {
    ...     'Date': [
    ...         '2021-01-01','2021-01-02','2021-01-03',
    ...         '2021-01-04','2021-01-05'
    ...     ],
    ...     'Sales': [10, 12, 14, 13, 15]
    ... }
    >>> df = pd.DataFrame(data)
    >>> # Simple split using 60% train and 40% test
    >>> train_df, test_df = ts_split(
    ...     df,
    ...     dt_col='Date',
    ...     split_type='simple',
    ...     test_ratio=0.4,
    ...     verbose=1
    ... )
    Performing simple split: Train size=3, Test size=2.

    >>> # Cross-validation with 2 splits and gap=0
    >>> splits = ts_split(
    ...     df,
    ...     dt_col='Date',
    ...     split_type='cv',
    ...     n_splits=2,
    ...     verbose=1
    ... )
    Performing cross-validation split with n_splits=2,
    gap=0.

    See Also
    --------
    sklearn.model_selection.TimeSeriesSplit : Cross-validation
        splits for time-series data.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021).
           *Forecasting: Principles and Practice* (3rd ed).
           OTexts.
    .. [2] Bergmeir, C., Hyndman, R.J., & Koo, B. (2018).
           A note on the validity of cross-validation for
           evaluating autoregressive time series prediction.
           *Computational Statistics & Data Analysis*,
           120, 70-83.
    """

    # 1) Validate the input DataFrame.
    is_frame(
        df,
        df_only=True,
        raise_exception=True,
        objname="DataFrame 'df'"
    )

    # 2) Validate and/or parse the datetime column using
    #    ts_validator to ensure correct ordering.
    df, dt_col = ts_validator(
        df=df,
        dt_col=dt_col,
        to_datetime='auto',
        as_index=False,
        error="raise",
        return_dt_col=True,
        verbose=verbose
    )

    # 3) Depending on split_type, perform the desired split logic.
    if split_type in ["simple", "base"]:
        # A single train-test partition.
        # Option A: Use date-based filtering with train_start
        # and train_end.
        if train_start or train_end:
            if verbose >= 1:
                print(
                    "Performing simple split with "
                    f"train_start={train_start}, "
                    f"train_end={train_end}."
                )
            # Filter train set
            if train_start and train_end:
                train_mask = (
                    (df[dt_col] >= pd.to_datetime(train_start))
                    & (df[dt_col] <= pd.to_datetime(train_end))
                )
                train_df = df.loc[train_mask]
                test_df = df.loc[~train_mask]
            elif train_end:
                train_mask = df[dt_col] <= pd.to_datetime(train_end)
                train_df = df.loc[train_mask]
                test_df = df.loc[~train_mask]
            else:
                # If only train_start is provided, up to user logic
                # Not fully specified, but we can handle similarly
                train_mask = df[dt_col] >= pd.to_datetime(train_start)
                train_df = df.loc[train_mask]
                test_df = df.loc[~train_mask]

        # Option B: Use 'test_ratio' for fraction-based split.
        elif test_ratio is not None:
            # Convert test_ratio to integer row count if in (0,1).
            test_ratio = validate_ratio(
                test_ratio,
                bounds=(0, 1),
                param_name="Test Ratio",
                exclude=0
            )
            n_test = int(len(df) * test_ratio)
            split_idx = len(df) - n_test
            if verbose >= 1:
                print(
                    f"Performing simple split: "
                    f"Train size={split_idx}, "
                    f"Test size={n_test}."
                )
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:
            # If neither 'train_end' nor 'test_ratio' is provided
            # for the simple approach, raise an error.
            raise ValueError(
                "`test_ratio` or `train_end` must be specified "
                "for a simple split."
            )

        return train_df, test_df

    elif split_type == "cv":
        # 4) Cross-validation approach using TimeSeriesSplit
        if verbose >= 1:
            print(
                f"Performing cross-validation split with "
                f"n_splits={n_splits}, gap={gap}."
            )
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap
        )
        splits = tscv.split(df)
        if verbose >= 2:
            # Show the actual indices for each split
            for i, (train_idx, test_idx) in enumerate(splits):
                print(
                    f"Split {i}: Train indices={train_idx}, "
                    f"Test indices={test_idx}"
                )
            # We need to re-generate, so return a new split iterator
            splits = tscv.split(df)
        return splits

    else:
        # 5) Invalid split type
        raise ValueError(
            f"Invalid split_type: {split_type}. "
            "Choose 'simple' or 'cv'."
        )

@validate_params ({
    "method": [StrOptions({'zscore', 'iqr'})]
    })
def ts_outlier_detector(
    df,
    dt_col=None,
    value_col=None,
    method="zscore",
    threshold=3,
    view=False,
    fig_size=(10, 5),
    show_grid=True,
    drop=False,
    verbose=0
):
    r"""
    Detect outliers in a time series using either Z-Score
    or Interquartile Range (IQR). Outliers can optionally
    be removed from the DataFrame.

    In many time-series analyses, anomalous points can
    distort model training or skew statistical inferences.
    Common outlier detection approaches include the
    Z-Score:

    .. math::
        Z_t = \frac{X_t - \mu}{\sigma},

    which flags points for which :math:`|Z_t| > \text{threshold}`.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the time series data.
        Must include a datetime column or index.
    dt_col : str, optional
        Column name representing the datetime dimension.
        If ``None``, the function assumes the index is
        datetime-like or uses `ts_validator`.
    value_col : str, optional
        Name of the target variable in the DataFrame (e.g.,
        "Sales").
    method : {'zscore', 'iqr'}, optional
        * ``'zscore'``: Use Z-Scores to detect outliers.
        * ``'iqr'``: Use the Interquartile Range method,
          :math:`Q_1` and :math:`Q_3` scaled by
          ``threshold * IQR``.
    threshold : int or float, optional
        Threshold multiplier for the chosen method. For
        Z-Scores, it represents how many standard
        deviations above/below the mean qualifies as
        an outlier (default=3). For IQR, it is the
        multiplier applied to the IQR to define
        lower and upper bounds.
    view : bool, optional
        If ``True``, displays a plot marking outliers
        in red over the original time series.
    fig_size : tuple of (float, float), optional
        The size of the figure (width, height) if
        visualizing.
    show_grid : bool, optional
        Whether to display gridlines in the plot.
    drop : bool, optional
        If ``True``, removes the rows flagged as outliers
        from ``df``.
    verbose : int, optional
        Verbosity level:

        * 0 : No console messages.
        * 1 : Basic information about outlier counts.
        * 2+ : (Not implemented here, but can be extended).

    Returns
    -------
    result : pandas.DataFrame
        The original DataFrame with a new column
        ``'is_outlier'`` marking outlier rows (True/False),
        unless ``drop=True``. In that case, it returns
        the DataFrame after removing these rows (and
        without the extra column).

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import ts_outlier_detector
    >>> data = {
    ...     'Date': [
    ...         '2021-01-01','2021-01-02','2021-01-03',
    ...         '2021-01-04','2021-01-05','2021-01-06'
    ...     ],
    ...     'Sales': [10, 100, 12, 13, 200, 15]
    ... }
    >>> df = pd.DataFrame(data)
    >>> df['Date'] = pd.to_datetime(df['Date'])
    >>> df_out = ts_outlier_detector(
    ...     df,
    ...     dt_col='Date',
    ...     value_col='Sales',
    ...     method='zscore',
    ...     threshold=2.5,
    ...     view=True,
    ...     drop=False,
    ...     verbose=1
    ... )
    Target variable: Sales
    Datetime column: Date
    Outlier detection method: zscore, Threshold: 2.5
    Detecting outliers using Z-Score...
    Number of outliers detected: 2
    Outliers retained in the DataFrame.

    Notes
    -----
    The choice of outlier detection (Z-Score vs. IQR) can be
    context dependent. Z-Scores assume a somewhat normal
    distribution of data [1]_ while IQR is more robust
    to skewed distributions [2]_.

    See Also
    --------
    ts_engineering : Broader time-series feature engineering
        (lags, rolling statistics, etc.).
    transform_stationarity : Techniques for removing trends
        or stabilizing variance.

    References
    ----------
    .. [1] Barnett, V., & Lewis, T. (1994). *Outliers in
           Statistical Data*. John Wiley & Sons.
    .. [2] Rousseeuw, P.J., & Croux, C. (1993). Alternatives
           to the median absolute deviation. *Journal of
           the American Statistical Association*,
           88(424), 1273-1283.
    """

    # 1) Validate the input DataFrame
    is_frame(
        df,
        df_only=True,
        raise_exception=True,
        objname="DataFrame 'df'"
    )

    # 2) Parse/validate the datetime column if provided.
    df, dt_col = ts_validator(
        df=df,
        dt_col=dt_col,
        to_datetime='auto',
        as_index=False,
        error="raise",
        return_dt_col=True,
        verbose=verbose
    )

    # 3) Validate and extract the target series.
    target, _ = validate_target_in(df, value_col)
    tname = target.name

    if verbose >= 1:
        print(f"Target variable: {tname}")
        print(f"Datetime column: {dt_col}")
        print(
            f"Outlier detection method: {method}, "
            f"Threshold: {threshold}"
        )

    # 4) Detect outliers using the chosen approach
    if method == "zscore":
        if verbose >= 1:
            print("Detecting outliers using Z-Score...")
        z_scores = zscore(target)
        outliers = np.abs(z_scores) > threshold
    elif method == "iqr":
        if verbose >= 1:
            print("Detecting outliers using IQR...")
        q1 = target.quantile(0.25)
        q3 = target.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)
        outliers = (target < lower_bound) | (target > upper_bound)
    else:
        raise ValueError(
            f"Invalid method: {method}. "
            "Choose 'zscore' or 'iqr'."
        )

    # Flag the outliers in the DataFrame
    df['is_outlier'] = outliers

    # Provide info on outlier counts
    if verbose >= 1:
        num_outliers = outliers.sum()
        print(f"Number of outliers detected: {num_outliers}")

    # 5) Visualization if `view=True`
    if view:
        plt.figure(figsize=fig_size)
        # Plot the main series
        plt.plot(
            df[dt_col],
            target,
            label="Original Data",
            color="blue",
            alpha=0.8
        )
        # Mark outliers in red
        plt.scatter(
            df[dt_col][outliers],
            target[outliers],
            color="red",
            label="Outliers",
            zorder=5
        )
        plt.title(
            f"Outlier Detection ({method.capitalize()} Method)"
        )
        plt.xlabel("Time")
        plt.ylabel(tname)
        if show_grid:
            plt.grid(
                True,
                linestyle=":",
                alpha=0.7
            )
        else:
            plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 6) If `drop=True`, remove outliers from the DataFrame
    if drop:
        df = df[~df['is_outlier']].drop(columns=['is_outlier'])
        if verbose >= 1:
            print(
                f"Outliers dropped. "
                f"Remaining data points: {len(df)}"
            )
    else:
        if verbose >= 1:
            print("Outliers retained in the DataFrame.")

    return df

@check_params ({ 
    "lags": Union[int, List[int]]
    })
def create_lag_features(
    df,
    value_col,
    dt_col=None,
    lag_features=None,
    lags=[1, 2], 
    dropna=True,
    include_original=True,
    reset_index=True,
    verbose=0
):
    r"""
    Generate lag features for a time series to capture temporal
    dependencies. Lag features are delayed copies of an original
    variable, enabling predictive models to learn from previous
    values.

    Formally, if :math:`X_t` denotes the value at time :math:`t`,
    then for a given lag :math:`\ell`, the lag feature
    :math:`X_{t-\ell}` provides the value of :math:`X` at
    :math:`t-\ell` [1]_. For multiple lags, the output DataFrame
    includes columns like:

    .. math::
        X_{t-1}, \; X_{t-3}, \; X_{t-7}, \ldots

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing time series data.
        Must have at least one time-like column or index.
    value_col : str
        The name of the target column (time series variable)
        for which lag features are created.
    dt_col : str, optional
        Name of the datetime column if not using the index.
        If ``None``, the function assumes the DataFrame index
        is datetime or validated by :func:`ts_validator`.
    lag_features : list of str, optional
        Additional feature columns (besides ``value_col``)
        for which to create lag features. If ``None``, only
        the target column (``value_col``) is used.
    lags : list of int, optional
        List of lag intervals to create. For example,
        ``[1, 3, 7]`` generates columns
        ``<feature>_lag_1, <feature>_lag_3, <feature>_lag_7``.
    dropna : bool, optional
        If ``True``, drops rows with any NaN introduced by
        shifting (i.e. the first few rows that cannot have
        lag values).
    include_original : bool, optional
        If ``True``, concatenates the original columns with
        the newly created lag columns in the output.
    reset_index : bool, optional
        If ``True``, resets the index of the resulting DataFrame.
        This can be helpful if lagging or sorting modifies the
        index alignment.
    verbose : int, optional
        Verbosity level. Higher values print more info about
        the process:

        * 0 : No printing.
        * 1 : Basic logs.

    Returns
    -------
    lagged_df : pandas.DataFrame
        DataFrame containing the newly generated lag features,
        and optionally the original features (depending on
        ``include_original``). If ``dropna=True``, rows
        lacking sufficient history are removed.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import create_lag_features
    >>> data = {
    ...     'Date': [
    ...         '2021-01-01','2021-01-02','2021-01-03',
    ...         '2021-01-04','2021-01-05'
    ...     ],
    ...     'Sales': [10, 12, 14, 13, 15]
    ... }
    >>> df = pd.DataFrame(data)
    >>> df['Date'] = pd.to_datetime(df['Date'])
    >>> lagged_df = create_lag_features(
    ...     df, value_col='Sales',
    ...     dt_col='Date',
    ...     lags=[1, 2],
    ...     dropna=True,
    ...     verbose=1
    ... )
    Target variable: Sales
    Datetime column: Date
    Lag intervals: [1, 2]
    Creating lag features for: Sales
    Rows dropped due to NaN values: 2

    Notes
    -----
    By introducing lagged versions of the target (and possibly
    other columns), models can learn from past states of the
    system. However, each additional lag typically reduces
    the row count if ``dropna=True``, because the first
    :math:`\max(lags)` observations cannot have complete lag
    values.

    See Also
    --------
    ts_engineering : A broader utility for generating lag
        features, rolling stats, and other transformations.
    transform_stationarity : Convert non-stationary series to
        stationary (e.g. differencing).

    References
    ----------
    .. [1] Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M.
           (2015). *Time Series Analysis: Forecasting and Control*.
           John Wiley & Sons.
    """

    # 1) Validate that df is a DataFrame.
    is_frame(
        df,
        df_only=True,
        raise_exception=True,
        objname="DataFrame 'df'"
    )

    # 2) Convert or validate datetime usage via ts_validator.
    #    This ensures the DataFrame has a proper time axis
    #    needed for shifting logic.
    df, dt_col = ts_validator(
        df=df,
        dt_col=dt_col,
        to_datetime='auto',
        as_index=True,
        error="raise",
        return_dt_col=True,
        verbose=verbose
    )

    # 3) Ensure the target column is present.
    target, _ = validate_target_in(df, value_col)
    tname = target.name

    lags = columns_manager(lags, empty_as_none=False)
    if verbose >= 1:
        print(f"Target variable: {tname}")
        print(f"Datetime column: {dt_col}")
        print(f"Lag intervals: {lags}")

    # 4) Determine which columns we create lag features for.
    #    If not specified, default to the target alone.
    lag_features = columns_manager(
        lag_features,
        empty_as_none=False
    )
    if value_col not in lag_features:
        lag_features.append(value_col)

    exist_features(
        df,
        features=lag_features,
        name="Lag features"
    )

    # 5) Build an empty DataFrame (indexed by the same index
    #    as df) to hold new columns.
    lagged_df = pd.DataFrame(index=df.index)

    # If dt_col remains in df.columns, keep a copy so we can
    # reference it in the final result (e.g. for plotting).
    if dt_col in df.columns:
        lagged_df[dt_col] = df[dt_col]

    # Create columns for each feature-lag combination.
    for feature in lag_features:
        if verbose >= 1:
            print(f"Creating lag features for: {feature}")
        for lag_k in lags:
            lagged_df[f"{feature}_lag_{lag_k}"] = df[feature].shift(lag_k)

    # 6) Optionally concatenate the original columns with the
    #    newly created lags.
    if include_original:
        lagged_df = pd.concat(
            [lagged_df, df],
            axis=1
        )
        # Avoid duplicating columns
        lagged_df = lagged_df.loc[:, ~lagged_df.columns.duplicated()]

    # 7) If dropna=True, remove rows lacking required lags.
    if dropna:
        if verbose >= 1:
            num_rows_before = len(lagged_df)
        lagged_df.dropna(inplace=True)
        if verbose >= 1:
            num_rows_after = len(lagged_df)
            print(
                "Rows dropped due to NaN values: "
                f"{num_rows_before - num_rows_after}"
            )

    # 8) If reset_index=True, restore the index to a column
    #    for subsequent usage in many modeling pipelines.
    if reset_index and dt_col == lagged_df.index.name:
        lagged_df.reset_index(inplace=True)

    return lagged_df

@validate_params({
    "method": [StrOptions( {'corr', 'correlation', 'pca'})], 
    "corr_threshold": [Interval(Real, 0, 1, closed="both")]
    })
def select_and_reduce_features(
    df,
    target_col=None,
    exclude_cols=None,
    method="corr",
    corr_threshold=0.9,
    n_components=None,
    scale_data=True,
    return_pca=False,
    verbose=0
):
    r"""
    Perform feature selection or dimensionality reduction
    on a dataset, using either correlation-based filtering
    or Principal Component Analysis (PCA).

    .. math::
        \text{Var}_{\text{explained}}(\text{PC}_i) =
        \frac{\lambda_i}{\sum_j \lambda_j},

    where :math:`\lambda_i` are the eigenvalues from
    the covariance matrix in PCA [1]_.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the dataset. Typically,
        it includes both feature columns and optionally a
        target column.
    target_col : str, list, optional
        The name(s) of the target column(s) which should be
        excluded from feature selection or reduction. If a
        list is provided, these columns are excluded as well.
        If ``None``, no column is excluded as target.
    exclude_cols : list of str, optional
        Additional columns to exclude from feature selection
        and PCA transformations (e.g. ID columns, date-time
        columns). Defaults to an empty list.
    method : {'corr', 'correlation', 'pca'}, optional
        The approach for feature reduction:

        * ``'corr'`` or ``'correlation'``:
          Use correlation-based feature selection. Features
          exceeding a specified correlation threshold are
          dropped.
        * ``'pca'``:
          Use Principal Component Analysis to reduce the
          dimensionality.
    corr_threshold : float, optional
        The correlation threshold for correlation-based
        feature selection. Any pair of features with
        absolute correlation above this value leads to
        dropping one of them. Defaults to 0.9.
    n_components : int or float, optional
        Number of PCA components to keep. If an integer,
        keeps that many components. If a float in range
        ``(0,1]``, it indicates the proportion of variance
        to retain. Only used if ``method='pca'``.
    scale_data : bool, optional
        If ``True``, standardizes the features before PCA
        using :class:`sklearn.preprocessing.StandardScaler`.
        Ignored for correlation-based selection. Default is
        True.
    return_pca : bool, optional
        If ``True`` and ``method='pca'``, returns the fitted
        PCA model along with the transformed DataFrame.
    verbose : int, optional
        Verbosity level:

        * 0 : No output.
        * 1 : Basic logs of feature counts and steps.
        * 2 : More detailed information such as correlation
          matrix or explained variance ratio.

    Returns
    -------
    transformed_df : pandas.DataFrame
        The resulting DataFrame after feature selection or
        PCA-based dimensionality reduction. If a target was
        specified, it is re-appended at the end.
    pca_model : sklearn.decomposition.PCA or None
        If ``method='pca'`` and ``return_pca=True``, returns
        the fitted PCA model. Otherwise ``None``.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ts_utils import select_and_reduce_features
    >>> data = {
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [2, 4, 6, 8, 10],
    ...     'C': [5, 3, 6, 2, 11],
    ...     'Target': [0, 1, 0, 1, 0]
    ... }
    >>> df = pd.DataFrame(data)
    >>> # Correlation-based selection
    >>> out_df = select_and_reduce_features(
    ...     df, target_col='Target',
    ...     method='corr', corr_threshold=0.8,
    ...     verbose=1
    ... )
    Number of features before selection: 3
    Excluded columns: []
    Performing correlation-based feature selection...

    >>> # PCA-based reduction
    >>> pca_df, pca_model = select_and_reduce_features(
    ...     df, target_col='Target', method='pca',
    ...     n_components=2, scale_data=True,
    ...     return_pca=True, verbose=1
    ... )
    Number of features before selection: 3
    Excluded columns: []
    Performing Principal Component Analysis (PCA)...
    Standardizing data before PCA.
    Explained variance ratio: [0.63717928 0.29160977 0.07121096]
    Number of components selected: 2

    Notes
    -----
    * Correlation-based selection can be efficient if many
      features are highly correlated, but it might discard
      relevant signals if multiple correlated features
      collectively provide synergy [2]_.
    * PCA transforms the data to orthogonal principal
      components, which can simplify many ML models but
      complicate interpretability.

    See Also
    --------
    PCA : The scikit-learn PCA class used for dimension
        reduction.
    transform_stationarity : Stabilize time-series data
        prior to certain modeling approaches.

    References
    ----------
    .. [1] Jolliffe, I.T., & Cadima, J. (2016). Principal
           component analysis: a review and recent
           developments. *Philosophical Transactions of the
           Royal Society A*, 374(2065), 20150202.
    .. [2] Guyon, I., & Elisseeff, A. (2003). *An introduction
           to variable and feature selection*. Journal of
           Machine Learning Research, 3(Mar), 1157-1182.
    """

    # Step 1: Validate the input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("`df` must be a pandas DataFrame.")

    # Convert target_col and exclude_cols to list-like
    target_col = columns_manager(
        target_col,
        empty_as_none=False
    )
    exclude_cols = columns_manager(
        exclude_cols,
        empty_as_none=False
    )

    # Filter out excluded columns from the DataFrame
    valid_cols = is_in_if(
        df.columns,
        items=exclude_cols,
        return_diff=True
    )
    # Now select from df only the "valid_cols"
    features = select_features(df, features=valid_cols)

    # Separate target if provided
    target = None
    if target_col is not None:
        # This also removes the target from 'features'
        target, features = validate_target_in(
            features,
            target_col
        )

    pca_model = None

    if verbose >= 1:
        print(
            f"Number of features before selection: {features.shape[1]}"
        )
        print(f"Excluded columns: {exclude_cols}")

    # Step 2: Check the method for correlation-based selection
    if method in ["correlation", "corr"]:
        if verbose >= 1:
            print("Performing correlation-based feature selection...")

        # Build correlation matrix
        corr_matrix = features.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        # Identify columns exceeding threshold
        to_drop = [
            col for col in upper_triangle.columns
            if any(upper_triangle[col] > corr_threshold)
        ]
        if verbose >= 2:
            print(f"Correlation matrix:\n{corr_matrix}")
            print(
                "Highly correlated features to drop "
                f"(threshold={corr_threshold}): {to_drop}"
            )
        # Drop those correlated columns
        reduced_features = features.drop(
            columns=to_drop,
            errors='ignore'
        )

        # Reattach target if needed
        if target_col:
            transformed_df = pd.concat(
                [reduced_features, target],
                axis=1
            )
        else:
            transformed_df = reduced_features

        if return_pca:
            # Warn if user requested PCA but method is correlation
            warnings.warn(
                "PCA is not selected as the method for dimensionality"
                " reduction. Returning correlation-based result only."
            )
        return transformed_df

    # Step 3: If method='pca', apply Principal Component Analysis
    elif method == "pca":
        if verbose >= 1:
            print("Performing Principal Component Analysis (PCA)...")
            if scale_data:
                print("Standardizing data before PCA.")

        # Scale data if requested
        if scale_data:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
        else:
            scaled_features = features.values

        # Instantiate PCA
        pca_model = PCA(n_components=n_components)
        principal_components = pca_model.fit_transform(scaled_features)

        # Name the principal components
        if isinstance(n_components, int):
            pca_columns = [
                f"PC{i+1}"
                for i in range(n_components)
            ]
        else:
            # If user set n_components as float => proportion of variance
            pca_columns = [
                f"PC{i+1}"
                for i in range(pca_model.n_components_)
            ]

        pca_df = pd.DataFrame(
            principal_components,
            columns=pca_columns,
            index=features.index
        )

        if verbose >= 1:
            print(
                f"Explained variance ratio: {pca_model.explained_variance_ratio_}"
            )
            print(
                f"Number of components selected: {pca_model.n_components_}"
            )

        # Attach target if present
        if target_col:
            transformed_df = pd.concat(
                [pca_df, target],
                axis=1
            )
        else:
            transformed_df = pca_df

        # Return PCA model if user requests it
        if return_pca:
            return transformed_df, pca_model

        return transformed_df

    else:
        # Step 4: Invalid method
        raise ValueError(
            f"Invalid method: {method}. "
            "Choose 'corr' (or 'correlation') or 'pca'."
        )
