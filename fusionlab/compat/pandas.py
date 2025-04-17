# -*- coding: utf-8 -*-

"""
Provides compatibility utilities for different versions of pandas.
This module includes functions and feature flags to ensure smooth
operation across various pandas versions, handling breaking changes
and deprecated features.

Key functionalities include:
- DataFrame and Series assertion utilities
- Resampling and validation utilities
- Compatibility checks for different pandas versions
- Utility functions for working with DataFrames and Series

The module ensures compatibility with pandas versions less than
2.2.0, 2.1.0, 2.0.0, 1.4.0, and 1.0.0.

Attributes
----------
version : packaging.version.Version
    The installed pandas version.
PD_LT_2_2_0 : bool
    True if the installed pandas version is less than 2.2.0.
PD_LT_2_1_0 : bool
    True if the installed pandas version is less than 2.1.0.
PD_LT_2_0_0 : bool
    True if the installed pandas version is less than 2.0.0.
PD_LT_1_0_0 : bool
    True if the installed pandas version is less than 1.0.0.
PD_LT_1_4 : bool
    True if the installed pandas version is less than 1.4.0.
PD_LT_2 : bool
    True if the installed pandas version is less than 2.0.

Functions
---------
assert_frame_equal
    Check if two DataFrame objects are equal.
assert_index_equal
    Check if two Index objects are equal.
assert_series_equal
    Check if two Series objects are equal.
describe_dataframe
    Describe a DataFrame with compatibility for pandas versions <2 and >=2.
data_klasses
    A tuple of pandas data structures (Series, DataFrame).
frequencies
    Date offset aliases for frequencies.
is_numeric_dtype
    Check if the dtype is numeric.
testing
    pandas testing utilities.
cache_readonly
    Decorator to cache readonly properties.
deprecate_kwarg
    Decorator to deprecate a keyword argument.
Appender
    Decorator to append an addendum to a docstring.
Substitution
    Decorator to perform string substitution on a docstring.
is_int_index
    Check if an index is of integer type.
is_float_index
    Check if an index is of float type.
make_dataframe
    Create a sample DataFrame.
to_numpy
    Convert a DataFrame or Series to a numpy array.
get_cached_func
    Get a cached function.
get_cached_doc
    Get a cached docstring.
call_cached_func
    Call a cached function.
"""

from typing import Optional

import numpy as np
from collections.abc import Iterable 
from packaging.version import Version, parse
import pandas as pd
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
    deprecate_kwarg,
)

__all__ = [
    "assert_frame_equal",
    "assert_index_equal",
    "assert_series_equal",
    "data_klasses",
    "frequencies",
    "is_numeric_dtype",
    "describe_dataframe", 
    "select_dtypes", 
    "testing",
    "cache_readonly",
    "deprecate_kwarg",
    "Appender",
    "Substitution",
    "is_int_index",
    "is_float_index",
    "make_dataframe",
    "to_numpy",
    "PD_LT_1_0_0",
    "get_cached_func",
    "get_cached_doc",
    "call_cached_func",
    "PD_LT_1_4",
    "PD_LT_2",
    "MONTH_END",
    "QUARTER_END",
    "YEAR_END",
    "FUTURE_STACK",
]

version = parse(pd.__version__)

PD_LT_2_2_0 = version < Version("2.1.99")
PD_LT_2_1_0 = version < Version("2.0.99")
PD_LT_2_0_0 = version < Version("2.0.0")
PD_LT_1_0_0 = version < Version("0.99.0")
PD_LT_1_4 = version < Version("1.3.99")
PD_LT_2 = version < Version("1.9.99")

try:
    from pandas.api.types import is_numeric_dtype
except ImportError:
    from pandas.core.common import is_numeric_dtype

try:
    from pandas.tseries import offsets as frequencies
except ImportError:
    from pandas.tseries import frequencies

data_klasses = (pd.Series, pd.DataFrame)

try:
    import pandas.testing as testing
except ImportError:
    import pandas.util.testing as testing

assert_frame_equal = testing.assert_frame_equal
assert_index_equal = testing.assert_index_equal
assert_series_equal = testing.assert_series_equal

def select_dtypes(
    df: pd.DataFrame, 
    dtypes: 'str | list[str]'=None, 
    incl: 'str | list[str]' = None, 
    excl: 'str | list[str]' = None, 
    return_columns: bool = False, 
    return_dtype: bool = False,
    include_nan: bool = False
) -> pd.DataFrame:
    """
    Selects columns from a pandas DataFrame based on data types or 
    includes/excludes certain column types. This function allows for 
    greater flexibility and control over the selection of columns based 
    on their data types. It supports inclusion and exclusion of specific 
    data types, and can also return the column names or a DataFrame with 
    selected data types.

    The function also accommodates numeric types and handles optional 
    arguments like `return_columns` (to return column names) and 
    `include_nan` (to include columns with NaN values). This function 
    aims to provide more control in environments where specific data types 
    need to be filtered, such as during pre-processing or data analysis.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The DataFrame from which to select columns based on data type. 
        This argument is mandatory, and the function will raise a 
        `TypeError` if the argument is not a valid DataFrame.

    dtypes : `str` or `list[str]`, optional
        The data type(s) to select from the DataFrame. Can be a 
        single type (e.g., `'int64'`) or a list of types 
        (e.g., `['int64', 'float64']`). Special case: If `dtypes` is 
        'numeric', it automatically includes `['int64', 'float64']`. For 
        non-string arguments, `incl` or `excl` should be used instead.

    incl : `str | list[str]`, optional, default: `None`
        Specifies the data types to include when selecting columns. 
        If provided, this will override the `dtypes` parameter to filter 
        columns based on the included types. Can be a single type or a 
        list of types.

    excl : `str | list[str]`, optional, default: `None`
        Specifies the data types to exclude from selection. If provided, 
        this will exclude columns matching the types in the list from 
        the selection. Can be a single type or a list of types.

    return_columns : `bool`, optional, default: `False`
        If `True`, returns the column names of the selected DataFrame 
        as a list. If `False`, returns the full DataFrame of selected 
        columns.

    return_dtype : `bool`, optional, default: `False`
        If `True`, the function will return a DataFrame with both the 
        column names and the corresponding data types for the selected 
        columns. This can be useful for examining the data types of 
        selected columns.

    include_nan : `bool`, optional, default: `False`
        If `True`, columns that contain NaN values will be included in 
        the selection, even if the columns' data types would otherwise 
        exclude them. If `False`, columns with NaN values are excluded 
        based on their data types.

    Returns
    -------
    `pandas.DataFrame`
        Returns a DataFrame containing the selected columns based on the 
        specified data types, or a list of column names if `return_columns` 
        is `True`. If `return_dtype` is `True`, a DataFrame with column names 
        and data types is returned instead.

    Examples
    --------
    1. Select all numeric columns from the DataFrame:
    
    >>> from gofast.compat.pandas import select_dtypes 
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.1, 2.2, 3.3], 'c': ['x', 'y', 'z']})
    >>> from gofast.compat.pandas import select_dtypes
    >>> select_dtypes(df, 'numeric')
       a    b
    0  1  1.1
    1  2  2.2
    2  3  3.3

    2. Select specific data types and return column names:
    
    >>> select_dtypes(df, ['int64', 'float64'], return_columns=True)
    ['a', 'b']

    3. Include only `float64` columns and exclude `int64` columns:
    
    >>> select_dtypes(df, 'float64', excl='int64')
       b
    0  1.1
    1  2.2
    2  3.3

    4. Select columns that include NaN values:
    
    >>> df = pd.DataFrame({'a': [1, 2, None], 'b': [4, 5, 6]})
    >>> select_dtypes(df, 'float64', include_nan=True)
       a    b
    0  1  4
    1  2  5
    2 NaN  6

    Notes
    -----
    - The `dtypes` argument can be used to select columns by their data 
      type, including numeric types (e.g., `int64`, `float64`) or any 
      other specific data types (e.g., `object` for string columns).
    - The `include` and `exclude` parameters provide additional flexibility 
      to selectively include or exclude specific data types from the selection.
    - This function is particularly useful for handling large DataFrames where 
      column selection based on data types is necessary, such as data preprocessing 
      or feature selection tasks in machine learning pipelines.

    See Also
    --------
    `pandas.DataFrame.select_dtypes` : The underlying function used for column 
    selection based on data types.
    
    References
    ----------
    .. [1] pandas documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
    """

    # Ensure that df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame, but got a {}.".format(
            type(df).__name__))

    # If dtypes is 'numeric', default to ['int64', 'float64']
    if dtypes == 'numeric':
        dtypes = ['int64', 'float64', "int32", 'float32']

    # If dtypes is a string, convert it to a list
    if isinstance(dtypes, str):
        dtypes = [dtypes]

    # Ensure dtypes is an iterable if not already
    if dtypes is not None:
        if not isinstance(dtypes, Iterable):
            raise TypeError(
                "`dtypes` must be a string or a list of strings. Use parameters"
                " `incl` or `excl` for non-string arguments instead."
            )
    
    # Prepare include/exclude arguments
    include = incl if incl is not None else []
    exclude = excl if excl is not None else []

    # Handle NaN inclusion/exclusion logic for numeric types
    if dtypes is not None:
        if include_nan and 'float64' in dtypes: 
            dtypes = list(set(dtypes) - {'float64'})  # Remove 'float64' if NaN is included
    
    # Select columns based on specified dtypes
    selected_df = df.copy() 
    if include:
        selected_df = df.select_dtypes(include=include)
    elif dtypes is not None:
        selected_df = df.select_dtypes(include=dtypes)

    # Exclude columns with specified data types
    if exclude:
        selected_df = selected_df.select_dtypes(exclude=exclude)

    # If return_columns is True, return only column names
    if return_columns:
        return selected_df.columns.tolist()

    # If return_dtype is True, return columns with their data types
    if return_dtype:
        return selected_df.dtypes

    return selected_df

def describe_dataframe(
        df, numeric_only=True, include_all=False, percentiles=None, 
        datetime_is_numeric=True):
    """
    Describe a DataFrame with compatibility for pandas versions <2 and >=2.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to describe. This parameter accepts any
        DataFrame object containing the data you wish to
        summarize. Each column in the DataFrame will be 
        described based on its type and the options selected.
        
    numeric_only : bool, optional
        Whether to include only numeric columns. If `True`, 
        the description will only include numeric columns.
        If `False`, all columns will be included in the 
        description, including non-numeric columns.
        Default is `True`.
    
    include_all : bool, optional
        If `True`, include all columns regardless of their
        data type. Overrides `numeric_only`. Default is `False`.

    percentiles : list-like of numbers, optional
        The percentiles to include in the output. All should
        fall between 0 and 1. By default, [0.25, 0.5, 0.75]
        are included if not provided.

    datetime_is_numeric : bool, optional
        Whether to treat datetime columns as numeric. This is
        only applicable for pandas versions >= 2.0.0. Default is `True`.

    Returns
    -------
    pd.DataFrame
        The description of the DataFrame. The returned DataFrame
        contains summary statistics for each column of the input
        DataFrame. For numeric columns, this includes metrics 
        such as count, mean, standard deviation, min, and max.
        For non-numeric columns, this includes metrics such as
        count, unique, top, and frequency.

    Notes
    -----
    The `describe_dataframe` function provides a flexible way to
    generate summary statistics for a DataFrame, ensuring
    compatibility across different versions of pandas. For pandas
    versions >= 2.0.0, the function includes the `datetime_is_numeric`
    parameter to handle datetime columns as numeric types. For 
    versions < 2.0.0, this parameter is omitted to maintain 
    compatibility.

    The mathematical formulations used in the summary statistics
    are as follows:

    .. math::

        \text{mean} = \frac{1}{n} \sum_{i=1}^n x_i

        \text{std} = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (x_i - \text{mean})^2}

    where :math:`x_i` are the data points, :math:`n` is the number of 
    data points, :math:`\text{mean}` is the average value, and 
    :math:`\text{std}` is the standard deviation.

    Examples
    --------
    >>> from gofast.compat.pandas import describe_dataframe
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [4, 3, 2, 1],
    ...     'C': pd.date_range('20230101', periods=4)
    ... })
    >>> df_descr = describe_dataframe(df, numeric_only=False)
    >>> print(df_descr)

    See Also
    --------
    pandas.DataFrame.describe : Generate descriptive statistics 
        for a DataFrame.

    References
    ----------
    .. [1] pandas.DataFrame.describe documentation. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
    """
    if not isinstance (df, pd.DataFrame): 
        raise TypeError ("Dataframe is expected for `describe_dataframe` to proceed.")
        
    if include_all:
        include = 'all'
    else:
        include = 'all' if not numeric_only else None

    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    
    if PD_LT_2_0_0:
        df_descr = df.describe(include=include, percentiles=percentiles, 
                               datetime_is_numeric=datetime_is_numeric)
    else:
        df_descr = df.describe(include=include, percentiles=percentiles)
    
    return df_descr


def is_int_index(index: pd.Index) -> bool:
    """
    Check if an index is integral

    Parameters
    ----------
    index : pd.Index
        Any numeric index

    Returns
    -------
    bool
        True if is an index with a standard integral type
    """
    return (
        isinstance(index, pd.Index)
        and isinstance(index.dtype, np.dtype)
        and np.issubdtype(index.dtype, np.integer)
    )


def is_float_index(index: pd.Index) -> bool:
    """
    Check if an index is floating

    Parameters
    ----------
    index : pd.Index
        Any numeric index

    Returns
    -------
    bool
        True if an index with a standard numpy floating dtype
    """
    return (
        isinstance(index, pd.Index)
        and isinstance(index.dtype, np.dtype)
        and np.issubdtype(index.dtype, np.floating)
    )


try:
    from pandas._testing import makeDataFrame as make_dataframe
except ImportError:
    import string

    def rands_array(nchars, size, dtype="O"):
        """
        Generate an array of byte strings.
        """
        rands_chars = np.array(
            list(string.ascii_letters + string.digits), dtype=(np.str_, 1)
        )
        retval = (
            np.random.choice(rands_chars, size=nchars * np.prod(size))
            .view((np.str_, nchars))
            .reshape(size)
        )
        if dtype is None:
            return retval
        else:
            return retval.astype(dtype)

    def make_dataframe():
        """
        Simple verion of pandas._testing.makeDataFrame
        """
        n = 30
        k = 4
        index = pd.Index(rands_array(nchars=10, size=n), name=None)
        data = {
            c: pd.Series(np.random.randn(n), index=index)
            for c in string.ascii_uppercase[:k]
        }

        return pd.DataFrame(data)

def iteritems_compat(series: pd.Series):
    """
    Compatibility function for iterating over Series items.

    Parameters
    ----------
    series : pd.Series
        The pandas Series to iterate over.

    Returns
    -------
    iterator
        An iterator over the (index, value) pairs of the series.
    
    Example
    --------
    from gofast.compat.pandas import iteritems_compat

    # Example usage of iteritems_compat
    series = pd.Series([1, 2, 3])
    for index, value in iteritems_compat(series):
        print(f"Index: {index}, Value: {value}")
    """
    if PD_LT_2:
        return series.iteritems()
    else:
        return series.items()

def make_dataframe_compat():
    """
    Compatibility function for creating a sample DataFrame.

    Returns
    -------
    DataFrame
        A pandas DataFrame with sample data.
        
    Example 
    --------
    from gofast.compat.pandas import make_dataframe_compat

    # Example usage of make_dataframe_compat
    df = make_dataframe_compat()
    print(df)
    """
    if PD_LT_2:
        return make_dataframe()
    else:
        import string

        def rands_array(nchars, size, dtype="O"):
            """
            Generate an array of byte strings.
            """
            rands_chars = np.array(
                list(string.ascii_letters + string.digits), dtype=(np.str_, 1)
            )
            retval = (
                np.random.choice(rands_chars, size=nchars * np.prod(size))
                .view((np.str_, nchars))
                .reshape(size)
            )
            if dtype is None:
                return retval
            else:
                return retval.astype(dtype)

        n = 30
        k = 4
        index = pd.Index(rands_array(nchars=10, size=n), name=None)
        data = {
            c: pd.Series(np.random.randn(n), index=index)
            for c in string.ascii_uppercase[:k]
        }

        return pd.DataFrame(data)

def is_pandas_version_less_than(version: str) -> bool:
    """
    Check if the current pandas version is less than the specified version.

    Parameters
    ----------
    version : str
        The version to compare against.

    Returns
    -------
    bool
        True if the current pandas version is less than the specified version.
        
    Example 
    -------
    from gofast.compat.pandas import is_pandas_version_less_than

    # Example usage of is_pandas_version_less_than
    if is_pandas_version_less_than("2.0.0"):
        print("Pandas version is less than 2.0.0")
    else:
        print("Pandas version is 2.0.0 or greater")

    """
    return pd.__version__ < version

def to_numpy(po: pd.DataFrame) -> np.ndarray:
    """
    Workaround legacy pandas lacking to_numpy

    Parameters
    ----------
    po : Pandas obkect

    Returns
    -------
    ndarray
        A numpy array
    """
    try:
        return po.to_numpy()
    except AttributeError:
        return po.values


def get_cached_func(cached_prop):
    try:
        return cached_prop.fget
    except AttributeError:
        return cached_prop.func

def call_cached_func(cached_prop, *args, **kwargs):
    f = get_cached_func(cached_prop)
    return f(*args, **kwargs)


def get_cached_doc(cached_prop) -> Optional[str]:
    return get_cached_func(cached_prop).__doc__


MONTH_END = "M" if PD_LT_2_2_0 else "ME"
QUARTER_END = "Q" if PD_LT_2_2_0 else "QE"
YEAR_END = "Y" if PD_LT_2_2_0 else "YE"
FUTURE_STACK = {} if PD_LT_2_1_0 else {"future_stack": True}

