# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

# Source: Adapted from earthai-tech/gofast/utils/base_utils (https://github.com/earthai-tech/gofast)
# This module is included in the FusionLab package, with modifications
# to fit FusionLab’s utility conventions.

"""
Essential utilities for data processing and analysis in FusionLab,
offering functions for normalization, interpolation, feature selection,
outlier removal, and various data manipulation tasks.

Adapted for FusionLab from the original gofast.utils.base_utils.
"""

from __future__ import annotations
import os
import re
import copy
import time
import shutil
import inspect
import pathlib
import warnings
import functools
import threading
import subprocess
from datetime import datetime
from collections.abc import Iterable as IterableInstance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import stats
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d, griddata
from matplotlib.ticker import FixedLocator
from sklearn.utils import all_estimators

from ..api.property import PandasDataHandlers
from ..api.types import (
    Union, List, Optional, Tuple, Iterable, Any, Set, Pattern, Dict,
    _T, _F, DataFrame, ArrayLike, Series, Callable, NDArray,
)
from ..compat.pandas import select_dtypes
from ..core.array_manager import (
    to_numeric_dtypes, reshape, to_array, array_preserver,
    drop_nan_in, to_series
)
from ..core.checks import (
    _assert_all_types, is_iterable, exist_features, validate_feature,
    is_numeric_dtype, check_datetime
)
from ..core.handlers import get_batch_size
from ..core.io import is_data_readable
from ..core.utils import ellipsis2false, smart_format, error_policy
from ..compat.scipy import check_scipy_interpolate
from ..decorators import Dataify
from ..exceptions import FileHandlingError
from .deps_utils import import_optional_dependency
from .validator import (
    check_consistent_length, get_estimator_name, _is_arraylike_1d,
    array_to_frame, build_data_if, _is_numeric_dtype, check_y,
    check_consistency_size, is_categorical, is_valid_policies,
    contains_nested_objects, parameter_validator, normalize_array,
    is_frame
)

__all__ = [
    'array2hdf5', 'binning_statistic', 'categorize_target',
    'category_count', 'detect_categorical_columns', 'normalizer',
    'extract_target', 'fancier_downloader', 'fillNaN', 'get_target',
    'interpolate_grid', 'interpolate_data', 'labels_validator',
    'make_df', 'remove_outliers', 'remove_target_from_array',
    'rename_labels_in', 'scale_y', 'select_features',
    'smooth1d', 'smoothing', 'soft_bin_stat', 'speed_rowwise_process',
    'nan_to_mode', 'handle_outliers', 'fill_NaN', 'map_values',
    'validate_target_in',
]

@is_data_readable 
def map_values(
    data : Union[DataFrame, Series, dict],
    map_dict: Dict[Any, Any],
    action: str = None,
    suffix : str = '_map',
    error : str = 'warn',
    coerce : bool = False
):
    """
    Map values in ``data`` using a dictionary of substitutions.

    The `<map_values>` function applies ``map_dict`` to each value in
    the columns of ``data``, handling unmapped items according to
    `<error>` (e.g., `'raise'`, `'warn'`, or `'ignore'`). When
    `<coerce>` is True, an attempt is made to convert columns to
    numeric or string depending on the type of the keys in
    ``map_dict``. The result is either appended as new columns if
    `<action>` is `'append'`, or replaces the original columns
    otherwise.

    Parameters
    ----------
    data : DataFrame, Series, or dict
        The input data on which to perform value mapping. A dict is
        converted to a DataFrame; a Series is treated as a single
        column DataFrame for uniform processing.

    map_dict : dict of Any to Any
        A mapping dictionary specifying how to transform existing
        values. Each key in ``map_dict`` represents an original value
        and the corresponding dict value is the substituted result.

    action : {None, 'append'}, optional
        Determines how the mapped results are integrated:
        - ``None``: Overwrite the existing columns with mapped
          values.
        - ``'append'``: Create new columns named
          `<original_column><suffix>` and append them.

    suffix : str, default='_map'
        Suffix appended to column names if `<action>` is `'append'`.
        Ignored if `<action>` is None.

    error : {'raise', 'warn', 'ignore'}, default='warn'
        Controls behavior when a value in ``data`` is not found in
        ``map_dict``:
        - ``'raise'``: Raises a ValueError for missing mappings.
        - ``'warn'``: Issues a warning but leaves the value unmapped.
        - ``'ignore'``: Silently leaves any unmapped value unchanged.

    coerce : bool, default=False
        If True, attempts to unify types so that items in each column
        align with the keys of ``map_dict``. Typically, columns are
        converted to string if keys are string, or numeric if keys are
        numbers. Fails partially if the column has incompatible values.

    Returns
    -------
    DataFrame or Series
        The updated data structure with mapped values. If the input
        was a Series and `<action>` is not `'append'`, returns a
        single Series. Otherwise, returns a DataFrame.

    Notes
    -----
    This function follows a simple mapping rule:
    .. math::
       mapped\\_value = \\begin{cases}
       \\text{map\\_dict}[x] &\\text{if } x \\in
       \\text{map\\_dict keys} \\\\
       x &\\text{otherwise}
       \\end{cases}
    When `<coerce>` is True, it attempts to cast data columns to a
    unified type for more consistent lookups.

    Examples
    --------
    >>> from gofast.utils.base_utils import map_values
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['X', 'Y', 'Z']})
    >>> mapping = {1: 100, 'X': 'Alpha'}
    >>> # Overwrite columns in df with mapped values
    >>> mapped_df = map_values(df, mapping, coerce=False)
    >>> mapped_df
       A      B
    0  100  Alpha
    1    2      Y
    2    3      Z

    >>> # Append new columns without overwriting
    >>> appended_df = map_values(df, mapping, action='append',
    ...                          suffix='_mapped')
    >>> appended_df
       A  B  A_mapped B_mapped
    0  1  X        100   Alpha
    1  2  Y          2       Y
    2  3  Z          3       Z

    See also
    --------
    `parameter_validator` : Validates parameter choices for
    consistent usage.
    `error_policy` : Manages error handling logic to handle
    'raise', 'warn', or 'ignore'.

    References
    ----------
    .. [1] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning
           in Python. Journal of Machine Learning Research,
           12, 2825-2830.
    .. [2] Gofast Documentation. Available at:
           https://gofast.readthedocs.io/en/latest/
    """

    error = error_policy(error, base='warn') 
    emsg = f"Invalid input <action>. Expect one of {None, 'append'}. Got '{action}'"
    action = parameter_validator(
        "action", target_strs={None, 'append'}, error_msg=emsg)(action)
    # 1) If user passed a dict instead of DataFrame/Series,
    # convert it to DataFrame.
    if isinstance(data, dict):
        data = pd.DataFrame(data)

    # 2) If data is a Series, convert to a single-col
    # DataFrame for uniform processing.
    is_series = False
    if isinstance(data, pd.Series):
        data = data.to_frame()
        is_series = True

    df = data.copy() # important to not change the original data 
    
    # 3) For each column in the DataFrame, attempt to map using `map_dict`.
    #    We'll store mapped results in new_cols dict to assemble after processing.
    new_cols = {}

    for col in df.columns:
        col_data = df[col]

        # 3a) If coerce=True, try to convert col_data to numeric if possible,
        #     or to string. We'll do a best-effort approach:
        if coerce:
            # Attempt to unify types so that dict keys match.
            # We'll guess numeric => string or string => numeric,
            # depending on map_dict.
            # Minimal approach: convert col_data to str if map_dict keys
            # are str, else numeric.
            # More robust logic might inspect the types in map_dict.
            # For now, let's do a naive approach: if the first key is str,
            # we do col_data = col_data.astype(str)
            # else we do numeric. We'll handle exceptions or partial conversions.
            first_key = next(iter(map_dict))
            if isinstance(first_key, str):
                try:
                    col_data = col_data.astype(str)
                except ValueError as e:
                    # If it fails, we skip forcing
                    warnings.warn(
                        f"coerce=True but converting column '{col}'"
                        f" to string failed: {e}")
            else:
                try:
                    col_data = pd.to_numeric(col_data, errors='coerce')
                except ValueError as e:
                    warnings.warn(
                        f"coerce=True but numeric conversion"
                        f" for column '{col}' failed: {e}")

        # 3b) We'll build a function that handles 'not found'
        # keys depending on `error`.
        def mapper_fn(x):
            # If x is in map_dict => return map_dict[x], else handle not found
            if x in map_dict:
                return map_dict[x]
            else:
                # If value not found, handle error/warn/ignore
                if error == 'raise':
                    raise ValueError(
                        f"Value '{x}' in column '{col}' not found in map_dict."
                    )
                elif error == 'warn':
                    warnings.warn(
                        f"Value '{x}' in column '{col}' not found in map_dict. "
                        f"It will be left unmapped."
                    )
                # error='ignore' => do nothing silently
                return x  # keep original if not found

        # 3c) Apply the mapper to each value in this column
        mapped_col = col_data.map(mapper_fn)

        # 3d) If action=='append', store in a new column named col+suffix
        #     else, we overwrite the original column name.
        if action == 'append':
            new_col_name = f"{col}{suffix}"
        else:
            new_col_name = col

        new_cols[new_col_name] = mapped_col

    # 4) If action=='append', we add these new columns to the existing DataFrame
    #    otherwise we replace the old columns with newly mapped columns.
    if action == 'append':
        for nc in new_cols:
            df[nc] = new_cols[nc]
    else:
        # Overwrite only the columns that exist in new_cols
        # if the user had more columns not in the old data,
        # we only keep them if they are in new_cols?
        # but logically we only mapped existing columns, so safe to do:
        for c in df.columns:
            if c in new_cols:
                df[c] = new_cols[c]
        # If action=None, the user might want to rename columns
        # to old name => we did that above.

    # 5) If data was originally a Series, return the relevant column as a Series
    #    If action='append' and is_series, we have two columns now => decide?
    #    We'll do minimal approach: if is_series and not append => return the single col
    if is_series and action != 'append':
        # There's only one column, let's return it as a series
        col_name = df.columns[0]
        return df[col_name]

    # 6) Return the DataFrame
    return df

@is_data_readable
def detect_categorical_columns(
    data,
    integer_as_cat=True,
    float0_as_cat=True,
    min_unique_values=None,
    max_unique_values=None,
    handle_nan=None,
    return_frame=False,
    consider_dt_as=None, 
    verbose=0
):
    r"""
    Detect categorical columns in a dataset by examining column
    types and user-defined criteria. Columns with integer type
    or float values ending with .0 can be categorized as
    categorical, depending on settings. Also handles user-defined
    thresholds for minimum and maximum unique values.
    
    .. math::
       \forall x \in X,\; x = \lfloor x \rfloor
    
    Above equation indicates that for float columns to be treated
    as categorical, each value :math:`x` must be an integer when
    cast from float. This function leverages the inline methods
    `build_data_if`, `drop_nan_in`, `fill_NaN`, `parameter_validator`,
    and `smart_format` (excluding those prefixed with `_`).
    
    Parameters
    ----------
    data : DataFrame or array-like
        The input data to analyze. If not a DataFrame,
        it will be converted internally.
    integer_as_cat : bool, optional
        If ``True``, integer-type columns are considered
        categorical. Default is ``True``.
    float0_as_cat : bool, optional
        If ``True``, float columns whose values can be
        cast to integer without remainder are considered
        categorical. Default is ``True``.
    min_unique_values : int or None, optional
        Minimum number of unique values in a column to
        qualify as categorical. If ``None``, no minimum
        check is applied.
    max_unique_values : int or ``'auto'`` or None, optional
        Maximum number of unique values allowed for a
        column to be considered categorical. If ``'auto'``,
        set the limit to the column's own unique count.
        If ``None``, no maximum check is applied.
    handle_nan : str or None, optional
        Handling method for missing data. Can be ``'drop'``
        to remove rows with NaNs, ``'fill'`` to impute
        them via forward/backward fill, or ``None`` for
        no change. 
    return_frame : bool, optional
        If ``True``, returns a DataFrame of detected
        categorical columns; otherwise returns a list of
        column names. Default is ``False``.
    consider_dt_as : str, optional
        Indicates how to handle or convert datetime columns when
        ``ops='validate'``:
        - `None`: Do not convert; if datetime is not accepted, 
          handle according to `accept_dt` and `error`.
        - `'numeric'`: Convert date columns to a numeric format
          (like timestamps).
        - `'float'`, `'float32'`, `'float64'`: Convert date columns
          to float representation.
        - `'int'`, `'int32'`, `'int64'`: Convert date columns to
          integer representation.
        - `'object'` or `'category'`: Convert date columns to Python 
          objects (strings, etc.). If conversion fails, raise or warn 
          per `error` policy.
    verbose : int, optional
        Verbosity level. If greater than 0, a summary of
        detected columns is printed.
    
    Returns
    -------
    list or DataFrame
        Either a list of column names or a DataFrame
        containing the categorical columns, depending on
        the value of ``return_frame``.
    
    Examples
    --------
    >>> from gofast.utils.base_utils import detect_categorical_columns
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [1.0, 2.0, 3.0],
    ...     'C': ['cat', 'dog', 'mouse']
    ... })
    >>> detect_categorical_columns(df)
    ['A', 'B', 'C']
    
    Notes
    -----
    - This function focuses on flexible treatment of
      integer and float columns. Combined with 
      `<verbose>` settings, it can provide detailed
      feedback.
    - Utilizing ``'drop'`` or ``'fill'`` in 
      ``handle_nan`` ensures minimal disruptions due
      to missing data.
 
    The function uses flexible criteria for determining whether a column should
    be treated as categorical, allowing for detection of columns with integer 
    values or float values ending in `.0` as categorical columns. The method is
    useful when preparing data for machine learning algorithms that expect 
    categorical inputs, such as decision trees or classification models.
    
    This method uses the helper function `build_data_if` from 
    `gofast.utils.validator` to ensure that the input `data` is a DataFrame. 
    If the input is not a DataFrame, it creates one, giving column names that 
    start with `input_name`.
    
    See Also
    --------
    build_data_if : Validates and converts input into a 
        DataFrame if needed.
    drop_nan_in : Drops NaN values from a DataFrame along 
        axis=0.
    fill_NaN : Fills missing data in a DataFrame using 
        forward and backward fill.

    References
    ----------
    .. [1] Harris, C.R., et al. "Array Programming
       with NumPy." *Nature*, 585(7825), 357–362 (2020).
    """

    # ensure input data is a DataFrame or convert it to one
    data = build_data_if(
        data, 
        to_frame=True, 
        force=True, 
        raise_exception=True, 
        input_name='col'
    )

    # validate handle_nan parameter
    handle_nan = parameter_validator(
        "handle_nan",
        target_strs={"fill", "drop", None}
    )(handle_nan)

    # optionally drop or fill NaN values
    if handle_nan == 'drop':
        data = drop_nan_in(data, solo_return=True)
    elif handle_nan == 'fill':
        data = fill_NaN(data, method='both')

    #Check if datetime columns exist in the data.
    has_dt_cols = check_datetime(data, error='ignore')
    if has_dt_cols: 
        if consider_dt_as is None: 
            # If no explicit instruction is provided 
            # via `consider_dt_as`, warn the user
            # that datetime columns will be treated
            # as numeric by default.
            warnings.warn(
                "Datetime columns detected. Defaulting"
                " to treating datetime columns as numeric."
                " If this behavior is not desired, please "
                "specify the `consider_dt_as` parameter"
                " accordingly."
            )
        else: 
            # If `consider_dt_as` is provided and True,
            # validate datetime columns 
            # according to the specified handling.
            data =check_datetime(
                data, 
                ops='validate', 
                accept_dt=True, 
                consider_dt_as=consider_dt_as, 
                error="warn", 
            )
    
    # user-specified limit might be set to 'auto' or a numeric value
    # store the original for reference
    original_max_unique = max_unique_values

    # prepare list to store detected categorical columns
    categorical_columns = []

    # iterate over the columns to determine if 
    # they meet conditions to be categorical
    for col in data.columns:
        unique_values = data[col].nunique()

        # if the user set max_unique_values to 'auto',
        # just use the column's own unique count
        if original_max_unique == 'auto':
            max_unique_values = unique_values

        # always consider object dtype as categorical
        if pd.api.types.is_object_dtype(data[col]):
            # check optional unique-value thresholds
            # no need, so go straight for collection.
            # if (
            #     (min_unique_values is None or unique_values >= min_unique_values)
            #     and (max_unique_values is None or unique_values <= max_unique_values)
            # ):
                categorical_columns.append(col)

        # also consider boolean dtype columns as categorical
        elif pd.api.types.is_bool_dtype(data[col]):
            # no need to apply condition, usually consider as a 
            # binary so categorical col.
            # if (
            #     (min_unique_values is None or unique_values >= min_unique_values)
            #     and (max_unique_values is None or unique_values <= max_unique_values)
            # ):
                categorical_columns.append(col)

        # consider integer columns as categorical if flagged
        elif integer_as_cat and pd.api.types.is_integer_dtype(data[col]):
            if (
                (min_unique_values is None or unique_values >= min_unique_values)
                and (max_unique_values is None or unique_values <= max_unique_values)
            ):
                categorical_columns.append(col)

        # consider float columns with all .0 values as categorical if flagged
        elif float0_as_cat and pd.api.types.is_float_dtype(data[col]):
            try:
                # check if all float values can be cast to int without remainder
                if np.all(data[col] == data[col].astype(int)):
                    if (
                        (min_unique_values is None or unique_values >= min_unique_values)
                        and (max_unique_values is None or unique_values <= max_unique_values)
                    ):
                        categorical_columns.append(col)
            except pd.errors.IntCastingNaNError as e:
                raise ValueError(
                    f"NaN detected in the data: {e}. Consider resetting "
                    "integer_as_cat=False or float0_as_cat=False, or handle NaN "
                    "via 'drop' or 'fill'."
                )

    # optionally print a summary of what was found or not found
    if verbose:
        if len(categorical_columns) == 0:
            print(
                "No categorical columns detected based on conditions. "
                "Consider adjusting min_unique_values or max_unique_values."
            )
        else:
            print(
                f"Categorical columns detected ({len(categorical_columns)}): "
                f"{smart_format(categorical_columns)}"
            )

    # return either the DataFrame subset of just 
    # the categorical columns or the list of names
    if return_frame:
        return data[categorical_columns]
    
    return categorical_columns

def handle_outliers(
    ar: Union[np.ndarray, Series, DataFrame],
    method: str = 'iqr',
    threshold: float = 1.5,
    fill_strategy: str = 'replace',  
    fill_value: Optional[float] = np.nan,
    axis: Optional[int] = None,  
    interpolate_method: str = 'linear',  
    inplace: bool = False,
    verbose: bool = False, 
    batch_size : Union[int, str] ='auto', 
    batch_processor: bool=False, 
) -> Union[np.ndarray, pd.Series, pd.DataFrame, None]:
    """
    Handle outliers in data structures with flexible strategies.

    This function detects and handles outliers in NumPy arrays,
    pandas Series, or pandas DataFrames using various statistical
    methods. It provides options to replace, interpolate, or drop
    outliers based on the specified strategy.

    Parameters
    ----------
    ar : np.ndarray, pd.Series, pd.DataFrame
        The input data to process for outliers.

    method : str, optional
        The method to use for outlier detection. Options are ``'iqr'``,
         ``'zscore'``, and ``'modified_zscore'``. Default is ``'iqr'``.

        - ``'iqr'``: Uses the Interquartile Range method.
          Outliers are points outside of
          :math:`[Q_1 - k \times IQR, \ Q_3 + k \times IQR]`,
          where :math:`Q_1` and :math:`Q_3` are the first and
          third quartiles, :math:`IQR = Q_3 - Q_1`, and
          :math:`k` is the ``threshold``.

        - ``'zscore'``: Uses the standard Z-score method. Outliers are points
          with a Z-score greater than the ``threshold``.

        - ``'modified_zscore'``: Uses the modified Z-score,
          which is more robust to outliers in small samples [1]_. Outliers are
          points with a modified Z-score greater than the ``threshold``.

    threshold : float, optional
        The threshold to use with the chosen ``method``. Its interpretation 
        depends on the method:

        - For ``'iqr'``, it is the multiplier of the IQR. Common values are
          1.5 (mild outliers) or 3.0
          (extreme outliers).

        - For ``'zscore'`` and ``'modified_zscore'``, it is the Z-score cutoff.
          Common values are 2.0 or 3.0.

        Default is 1.5.

    fill_strategy : str, optional
        The strategy to handle detected outliers.
        Options are ``'replace'``, ``'interpolate'``, and ``'drop'``
        . Default is ``'replace'``.

        - ``'replace'``: Replaces outliers with ``fill_value``.

        - ``'interpolate'``: Replaces outliers with interpolated values using
           the method specified in ``interpolate_method``.

        - ``'drop'``: Removes outliers from the data.

    fill_value : float, optional
        The value to replace outliers with when ``fill_strategy`` is
         ``'replace'``. Default is ``np.nan``.

    axis : int or None, optional
        The axis along which to detect outliers:

        - For ``DataFrame`` and 2D ``ndarray``, ``0`` applies the method 
         to each column, ``1`` to each row, and ``None`` to the flattened 
         array.

        - For ``Series`` and 1D ``ndarray``, this parameter is ignored.

        Default is ``None``.

    interpolate_method : str, optional
        The interpolation method to use when ``fill_strategy``
        is ``'interpolate'``. Options include methods supported
        by ``pandas.Series.interpolate`` or
        ``pandas.DataFrame.interpolate``, such as ``'linear'``, ``'nearest'``,
        ``'spline'``, etc. Default is ``'linear'``.

    inplace : bool, optional
        If ``True``, perform the operation in-place and return ``None``. 
        Default is ``False``.

    verbose : bool, optional
        If ``True``, print the number of outliers detected in each column 
        or row. Default is ``False``.

    Returns
    -------
    np.ndarray, pd.Series, pd.DataFrame, or None
        The data with outliers handled according to the specified strategy.
        Returns ``None`` if ``inplace`` is ``True``.

    Notes
    -----
    **Outlier Detection Methods:**

    - *Interquartile Range (IQR) Method*:

      The IQR method identifies outliers based on the spread of the middle 50%
      of the data. It is robust to non-normal distributions.

      .. math::

          IQR = Q_3 - Q_1

          \text{Lower Bound} = Q_1 - k \times IQR

          \text{Upper Bound} = Q_3 + k \times IQR

      Where :math:`Q_1` and :math:`Q_3` are the first and third quartiles,
      and :math:`k` is the ``threshold``.

    - *Z-score Method*:

      Assumes a normal distribution and identifies outliers based on standard 
      deviations from the mean.

      .. math::

          Z = \frac{X - \mu}{\sigma}

      Where :math:`\mu` is the mean and :math:`\sigma` is the standard 
      deviation.
      

    - *Modified Z-score Method*:

      Uses the Median Absolute Deviation (MAD) instead of standard
      deviation, making it more robust to outliers, especially in smaller 
      datasets [1]_.

      .. math::

          \text{Modified Z} = 0.6745 \times \frac{X - \tilde{X}}{\text{MAD}}

      Where :math:`\tilde{X}` is the median and

      .. math::

          \text{MAD} = \text{median}(|X - \tilde{X}|)

    **Handling Strategies:**

    - *Replace*: Outliers are replaced with ``fill_value``.

    - *Interpolate*: Outliers are replaced using interpolation
      methods. This is useful for time-series data.

    - *Drop*: Outliers are removed from the dataset.
    
    **Additional Notes:**

    - **Data Types and NaN Representation:**
    
      NumPy integer arrays cannot represent `np.nan`. When working with 
      missing values or outlier replacement that involves `np.nan`, always 
      ensure your array is of a floating-point type.
    
    - **Interpolation at Edges:**
    
      By default, `pandas` interpolation methods won't fill NaNs at the start 
      or end of a series. The `limit_direction` parameter controls the 
      direction in which to fill missing values:
    
      - `'forward'`: Only fill NaNs forward.
      - `'backward'`: Only fill NaNs backward.
      - `'both'`: Fill NaNs in both directions.

    Examples
    --------
    Detect and replace outliers in a pandas Series:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.utils.base_utils import handle_outliers
    >>> data = pd.Series([1, 2, 2, 3, 4, 100])
    >>> clean_data = handle_outliers(data, method='iqr', threshold=1.5)
    >>> print(clean_data)
    0      1.0
    1      2.0
    2      2.0
    3      3.0
    4      4.0
    5      NaN
    dtype: float64

    Handle outliers in a DataFrame along columns:

    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 2, 3, 4, 100],
    ...     'B': [10, 12, 12, 13, 14, -100]
    ... })
    >>> clean_df = handle_outliers(df, method='zscore', threshold=2, axis=0)
    >>> print(clean_df)
          A     B
    0   1.0  10.0
    1   2.0  12.0
    2   2.0  12.0
    3   3.0  13.0
    4   4.0  14.0
    5   NaN   NaN

    Interpolate outliers in a NumPy array:

    >>> arr = np.array([1, 2, 2, 3, 4, 100])
    >>> clean_arr = handle_outliers(
    ...     arr, method='modified_zscore', threshold=3.5,
    ...     fill_strategy='interpolate', interpolate_method='linear'
    ... )
    >>> print(clean_arr)
    [1. 2. 2. 3. 4. 4.]

    See Also
    --------
    interpolate_grid : Interpolate missing values in a 2D grid.

    References
    ----------
    .. [1] Iglewicz, Boris, and David Hoaglin. "Volume 16: How to Detect and
        Handle Outliers." The ASQC Basic References in Quality Control:
       Statistical Techniques (1993).

    """
    # Validate parameters (external function)
    supported_methods = {'iqr', 'zscore', 'modified_zscore'}
    supported_fill_strategies = {'replace', 'interpolate', 'drop'}
    method, fill_strategy = _validate_parameters(
        method,
        fill_strategy,
        supported_methods,
        supported_fill_strategies
    )
    if fill_strategy == 'interpolate' and not np.isnan(fill_value):
        raise ValueError(
            "When `fill_strategy` is 'interpolate', `fill_value` must be np.nan."
        )
    if not isinstance(ar, (np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError(
            "Input `ar` must be a NumPy array, pandas Series, or pandas DataFrame."
        )
    if isinstance(ar, pd.DataFrame):
        return _process_dataframe(
            df=ar,
            method=method,
            threshold=threshold,
            fill_strategy=fill_strategy,
            fill_value=fill_value,
            axis=axis,
            interpolate_method=interpolate_method,
            inplace=inplace,
            verbose=verbose,
            batch_size=batch_size,
        )
    elif isinstance(ar, pd.Series):
        return _process_series(
            series=ar,
            method=method,
            threshold=threshold,
            fill_strategy=fill_strategy,
            fill_value=fill_value,
            interpolate_method=interpolate_method,
            inplace=inplace,
            verbose=verbose,
            batch_size=batch_size,
        )
    elif isinstance(ar, np.ndarray):
        return _process_ndarray(
            array=ar,
            method=method,
            threshold=threshold,
            fill_strategy=fill_strategy,
            fill_value=fill_value,
            interpolate_method=interpolate_method,
            axis=axis,
            inplace=inplace,
            batch_size=batch_size,
        )

def _process_dataframe(
    df: pd.DataFrame,
    method: str,
    threshold: float,
    fill_strategy: str,
    fill_value: Optional[float],
    axis: Optional[int],
    interpolate_method: str,
    inplace: bool,
    verbose: bool,
    batch_size: Optional[Union[str, int]] = None,
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    if batch_size is not None:
        if batch_size == 'auto':
            batch_size = get_batch_size(df, default_size= 10000 )
            # batch_size = 10000  
        else:
            batch_size = int(batch_size)
        total_rows = len(df)

        def batch_generator():
            for start in range(0, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                df_batch = df.iloc[start:end]
                processed_batch = _process_dataframe(
                    df=df_batch,
                    method=method,
                    threshold=threshold,
                    fill_strategy=fill_strategy,
                    fill_value=fill_value,
                    axis=axis,
                    interpolate_method=interpolate_method,
                    inplace=False,
                    verbose=verbose,
                    batch_size=None  # Avoid recursion
                )
                yield processed_batch
                if verbose:
                    print(f"Processed batch {start} to {end}")
        result = pd.concat(batch_generator(), ignore_index=True)
        return result
    else:
        if method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (df < lower_bound) | (df > upper_bound)
        elif method == 'zscore':
            mean = df.mean()
            std = df.std()
            z_scores = (df - mean) / std
            outlier_mask = (z_scores < -threshold) | (z_scores > threshold)
        else:  # modified_zscore
            median = df.median()
            mad = df.mad()
            modified_z_scores = 0.6745 * (df - median) / mad
            outlier_mask = (modified_z_scores < -threshold) | (modified_z_scores > threshold)
        # Apply outlier strategy
        if fill_strategy == 'replace':
            df[outlier_mask] = fill_value
        elif fill_strategy == 'interpolate':
            df[outlier_mask] = np.nan
            df = df.interpolate(method=interpolate_method, axis=axis)
        elif fill_strategy == 'drop':
            df = df[~outlier_mask.any(axis=1)]
        return df

def _process_series(
    series: pd.Series,
    method: str,
    threshold: float,
    fill_strategy: str,
    fill_value: Optional[float],
    interpolate_method: str,
    inplace: bool,
    verbose: bool,
    batch_size: Optional[Union[str, int]] = None,
) -> pd.Series:
    if not inplace:
        series = series.copy()

    if batch_size is not None:
        if batch_size == 'auto':
            batch_size = get_batch_size(series, default_size= 10000 )
        else:
            batch_size = int(batch_size)
        total_length = len(series)

        def batch_generator():
            for start in range(0, total_length, batch_size):
                end = min(start + batch_size, total_length)
                series_batch = series.iloc[start:end]
                processed_batch = _process_series(
                    series=series_batch,
                    method=method,
                    threshold=threshold,
                    fill_strategy=fill_strategy,
                    fill_value=fill_value,
                    interpolate_method=interpolate_method,
                    inplace=False,
                    verbose=verbose,
                    batch_size=None  # Avoid recursion
                )
                yield processed_batch
                if verbose:
                    print(f"Processed batch {start} to {end}")
        result = pd.concat(batch_generator(), ignore_index=True)
        return result
    else:
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (series < lower_bound) | (series > upper_bound)
        elif method == 'zscore':
            mean = series.mean()
            std = series.std()
            z_scores = (series - mean) / std
            outlier_mask = (z_scores < -threshold) | (z_scores > threshold)
        else:  # modified_zscore
            median = series.median()
            mad = series.mad()
            modified_z_scores = 0.6745 * (series - median) / mad
            outlier_mask = (
                modified_z_scores < -threshold) | (modified_z_scores > threshold)
        # Apply outlier strategy
        if fill_strategy == 'replace':
            series[outlier_mask] = fill_value
        elif fill_strategy == 'interpolate':
            series[outlier_mask] = np.nan
            series = series.interpolate(method=interpolate_method)
        elif fill_strategy == 'drop':
            series = series[~outlier_mask]
        return series

def _process_ndarray(
    array: np.ndarray,
    method: str,
    threshold: float,
    fill_strategy: str,
    fill_value: Optional[float],
    interpolate_method: str,
    axis: Optional[int],
    inplace: bool,
    batch_size: Optional[Union[str, int]] = None,
) -> np.ndarray:
    if not inplace:
        array = array.copy()

    if batch_size is not None:
        if batch_size == 'auto':
            batch_size = get_batch_size(array, default_size= 10000 )
        else:
            batch_size = int(batch_size)
        if axis == 0 or axis is None:
            total_rows = array.shape[0]

            def batch_generator():
                for start in range(0, total_rows, batch_size):
                    end = min(start + batch_size, total_rows)
                    array_batch = array[start:end]
                    processed_batch = _process_ndarray(
                        array=array_batch,
                        method=method,
                        threshold=threshold,
                        fill_strategy=fill_strategy,
                        fill_value=fill_value,
                        interpolate_method=interpolate_method,
                        axis=axis,
                        inplace=False,
                        batch_size=None  # Avoid recursion
                    )
                    yield processed_batch
            result = np.concatenate(list(batch_generator()), axis=0)
            return result
        else:
            # Implement batch processing along other axes if needed
            raise NotImplementedError(
                "Batch processing for axis > 0 is not implemented.")
    else:
        original_dtype = array.dtype
        if method == 'iqr':
            Q1 = np.percentile(array, 25, axis=axis, keepdims=True)
            Q3 = np.percentile(array, 75, axis=axis, keepdims=True)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (array < lower_bound) | (array > upper_bound)
        elif method == 'zscore':
            mean = np.mean(array, axis=axis, keepdims=True)
            std = np.std(array, axis=axis, keepdims=True)
            z_scores = (array - mean) / std
            outlier_mask = (z_scores < -threshold) | (z_scores > threshold)
        else:  # modified_zscore
            median = np.median(array, axis=axis, keepdims=True)
            mad = np.median(np.abs(array - median), axis=axis, keepdims=True)
            modified_z_scores = 0.6745 * (array - median) / mad
            outlier_mask = (
                modified_z_scores < -threshold) | (modified_z_scores > threshold)
        # Apply outlier strategy
        if fill_strategy == 'replace':
            array = array.astype(np.float64)  # To accommodate np.nan if necessary
            array[outlier_mask] = fill_value
            array = array.astype(original_dtype)
        elif fill_strategy == 'interpolate':
            if not np.isnan(fill_value):
                raise ValueError(
                    "Interpolate is feasible only if `fill_value=np.nan`.")
            array = array.astype(np.float64)
            array[outlier_mask] = np.nan
            array = interpolate_grid(array, method=interpolate_method)
            array = array.astype(original_dtype)
        elif fill_strategy == 'drop':
            if array.ndim == 1:
                axis = None
            if axis is None:
                array = array[~outlier_mask]
            elif axis == 0:
                rows_to_keep = ~np.any(outlier_mask, axis=1)
                array = array[rows_to_keep, :]
            elif axis == 1:
                cols_to_keep = ~np.any(outlier_mask, axis=0)
                array = array[:, cols_to_keep]
            else:
                raise ValueError("Invalid axis for drop strategy in ndarray.")
        return array

def _validate_parameters(
    method: str,
    fill_strategy: str,
    supported_methods: set,
    supported_fill_strategies: set
) -> Tuple[str, str]:

    method = method.lower()
    fill_strategy = fill_strategy.lower()

    method = parameter_validator(
        "method", target_strs=supported_methods, 
        error_msg= ( 
            f"Unsupported method '{method}'."
            " Supported methods are: {supported_methods}"
            )
        ) (method)
    # validate method 
    fill_strategy = parameter_validator(
        "method", target_strs=supported_fill_strategies, 
        error_msg=  (
            f"Unsupported fill_strategy '{fill_strategy}'."
            f" Supported strategies are: {supported_fill_strategies}"
            )
        ) (fill_strategy)
    

    return method, fill_strategy

def remove_outliers(
    ar: Union[ArrayLike,DataFrame],  
    method: str = 'IQR',
    threshold: float = 3.0,
    fill_value: Optional[float] = None,
    axis: int = 1,
    interpolate: bool = False,
    kind: str = 'linear'
) -> Union[ArrayLike, DataFrame]:
    """
    Efficient strategy to remove outliers in the data. 
    
    An outlier is a data point in a sample, observation, or distribution 
    that lies outside the overall pattern. A commonly used rule is to 
    consider a data point an outlier if it is more than 1.5 * IQR below 
    the first quartile or above the third quartile.
    
    Two approaches are used to identify and remove outliers:

    - Inter Quartile Range (``IQR``)
      IQR is the most commonly used and most trusted approach in 
      research. Outliers are defined as points lying below Q1 - 1.5 * IQR 
      or above Q3 + 1.5 * IQR. The quartiles and IQR are calculated as follows:
      
      .. math::
          
        Q1 = \frac{1}{4}(n + 1),\\
        Q3 = \frac{3}{4}(n + 1),\\
        IQR = Q3 - Q1,\\
        \text{Upper} = Q3 + 1.5 \times IQR,\\
        \text{Lower} = Q1 - 1.5 \times IQR.
    
    - Z-score 
      Also known as a standard score, this value helps to understand how 
      far a data point is from the mean. After setting a threshold, data 
      points with Z-scores beyond this threshold are considered outliers.
      
      .. math::
          
          \text{Zscore} = \frac{(\text{data\_point} - \text{mean})}{\text{std\_deviation}}
      
    A threshold value of generally 3.0 is chosen as 99.7% of data points 
    lie within +/- 3 standard deviations in a Gaussian Distribution.

    Parameters
    ----------
    ar : Union[np.ndarray, pd.DataFrame]
        The input data from which to remove outliers, either a numpy array 
        or pandas DataFrame.
    method : str, default 'IQR'
        Method to detect and remove outliers:
        - ``'IQR'`` uses the Inter Quartile Range to define outliers.
        - ``'Z-score'`` identifies outliers based on standard deviation.
        See detailed explanation on how each method works in the function's 
        description.
    threshold : float, default 3.0
        For ``'Z-score'``, this is the number of standard deviations a data point
        must be from the mean to be considered an outlier. For 'IQR', 
        this multiplies the IQR range.
    fill_value : float, optional
        Value used to replace outliers. If None, outliers are removed from 
        the dataset.
    axis : int, default 1
        Specifies the axis along which to remove outliers, applicable to 
        multi-dimensional data.
    interpolate : bool, default False
        Enables interpolation to estimate and replace outliers, only applicable
        if `fill_value` is NaN.
    kind : str, default 'linear'
        Type of interpolation used if interpolation is True. Options include
        ``'nearest'``, ``'linear'``, ``'cubic'``.
    
    Returns
    -------
    Union[np.ndarray, DataFrame]
        Array or DataFrame with outliers removed or replaced.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.utils.base_utils import remove_outliers 
    >>> np.random.seed(42)
    >>> data = np.random.randn(7, 3)
    >>> data_r = remove_outliers(data)
    >>> data.shape, data_r.shape
    (7, 3), (5, 3)
    >>> remove_outliers(data, fill_value=np.nan)
    array([[ 0.49671415, -0.1382643 ,  0.64768854],
           [ 1.52302986, -0.23415337, -0.23413696],
           [ 1.57921282,  0.76743473, -0.46947439],
           [ 0.54256004, -0.46341769, -0.46572975],
           [ 0.24196227,         nan,         nan],
           [-0.56228753, -1.01283112,  0.31424733],
           [-0.90802408,         nan,  1.46564877]])
    >>> remove_outliers(data[:, 0], fill_value=np.nan, interpolate=True)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(np.arange(len(data)), data, 'r-')
    """

    method = str(method).lower()
    # Validation of inputs
    method = parameter_validator(
        "method", target_strs={"iqr", 'z-score'}) (method)
    
    if isinstance (ar, pd.DataFrame):
        return _remove_outliers( ar, n_std= threshold )
    
    is_series = isinstance ( ar, pd.Series)
    arr =np.array (ar, dtype = float)

    if method =='iqr': 
        Q1 = np.percentile(arr[~np.isnan(arr)], 25,) 
        Q3 = np.percentile(arr[~np.isnan(arr)], 75)
        IQR = Q3 - Q1
        
        upper = Q3 + 1.5 * IQR  
        
        upper_arr = np.array (arr >= upper) 
        lower = Q3 - 1.5 * IQR 
        lower_arr =  np.array ( arr <= lower )
        # replace the oulier by nan 
        arr [upper_arr]= fill_value if fill_value else np.nan  
        arr[ lower_arr]= fill_value if fill_value else np.nan 
        
    if method =='z-score': 
        z = np.abs(stats.zscore(arr[~np.isnan(arr)]))
        zmask  = np.array ( z > threshold )
        arr [zmask]= fill_value if fill_value else np.nan
        
    if fill_value is None: 
        # delete nan if fill value is not provided 
        arr = arr[ ~np.isnan (arr ).any(axis =1)
                  ]  if np.ndim (arr) > 1 else arr [~np.isnan(arr)]

    if interpolate: 
        if fill_value !=np.nan: 
            raise ValueError(
                "Interpolate is feasible only if ``fill_value=np.nan``.")
        arr = interpolate_grid (arr, method = kind)
        
    if is_series: 
        arr =pd.Series (arr.squeeze(), name =ar.name )
        
    return arr 

def _remove_outliers(data, n_std=3):
    """Remove outliers from a dataframe."""
    # separate cat feature and numeric features 
    # if exists 
    df, numf, catf = to_numeric_dtypes(
        data , return_feature_types= True,  drop_nan_columns =True )
    # get on;y the numeric 
    df = df[numf]
    for col in df.columns:
        # print('Working on column: {}'.format(col))
        mean = df[col].mean()
        sd = df[col].std()
        df = df[(df[col] <= mean+(n_std*sd))]
    # get the index and select only the index 
    index = df.index 
    # get the data by index then 
    # concatenate 
    df_cat = data [catf].iloc [ index ]
    df = pd.concat ( [df_cat, df ], axis = 1 )
    
    return df

def interpolate_grid(
    arr,
    method ='cubic', 
    fill_value='auto', 
    view = False,
    ): 
    """
    Interpolate data containing missing values. 

    Parameters 
    -----------
    arr: ArrayLike2D 
       Two dimensional array for interpolation 
    method: str, default='cubic'
      kind of interpolation. It could be ['nearest'|'linear'|'cubic']. 
     
    fill_value: float, str, default='auto' 
       Fill the interpolated grid at the egdes or surrounding NaN with 
       a filled value. The ``auto`` uses the forward and backward 
       fill strategy. 
       
    view: bool, default=False, 
       Quick visualize the interpolated grid. 
 
    Returns 
    ---------
    arri: ArrayLike2d 
       Interpolated 2D grid. 
       
    See also 
    ---------
    spi.griddata: 
        Scipy interpolate Grid data 
    fillNaN: 
        Fill missing data strategy. 
        
    Examples
    ---------
    >>> import numpy as np
    >>> from gofast.utils.base_utils import interpolate_grid 
    >>> x = [28, np.nan, 50, 60] ; y = [np.nan, 1000, 2000, 3000]
    >>> xy = np.vstack ((x, y))._T
    >>> xyi = interpolate_grid (xy, view=True ) 
    >>> xyi 
    array([[  28.        ,   28.        ],
           [  22.78880663, 1000.        ],
           [  50.        , 2000.        ],
           [  60.        , 3000.        ]])

    """
    spi = check_scipy_interpolate()
    if spi is None:
        return None
    
    is2d = True 
    arr = np.asarray (arr) 
    
    if arr.ndim==1: 
        #convert to two dimension array
        arr = np.vstack ((arr, arr ))
        is2d =False 
        # raise TypeError(
        #     "Expect two dimensional array for grid interpolation.")
        
    # make x, y array for mapping 
    x = np.arange(0, arr.shape[1])
    y = np.arange(0, arr.shape[0])
    #mask invalid values
    arr= np.ma.masked_invalid(arr) 
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~arr.mask]
    y1 = yy[~arr.mask]
    newarr = arr[~arr.mask]
    
    arri = spi.griddata(
        (x1, y1),
        newarr.ravel(),
        (xx, yy), 
        method=method
        )
    
    if fill_value =='auto': 
        arri = fillNaN(arri, method ='both ')
    else:
        arri [np.isnan(arri)] = float( _assert_all_types(
            fill_value, float, int, objname ="'fill_value'" )
            ) 

    if view : 
        fig, ax  = plt.subplots (nrows = 1, ncols = 2 , sharey= True, )
        ax[0].imshow(arr ,interpolation='nearest', label ='Raw Grid')
        ax[1].imshow (arri, interpolation ='nearest', 
                      label = 'Interpolate Grid')
        
        ax[0].set_title ('Raw Grid') 
        ax[1].set_title ('Interpolate Grid') 
        
        plt.show () 
        
    if not is2d: 
        arri = arri[0, :]
        
    return arri 

def interpolate_grid_in(
    arr: np.ndarray,
    method: str = 'linear',
    fill_value: str = 'auto',
    view: bool = False
) -> np.ndarray:
    """
    Interpolate missing values in a 2D grid.

    Parameters
    ----------
    arr : np.ndarray
        Two-dimensional array with missing values (np.nan).
    method : str, default='linear'
        Interpolation method: 'nearest', 'linear', 'cubic', etc.
    fill_value : str or float, default='auto'
        How to fill values outside the convex hull:
        - 'auto': Use nearest-neighbor interpolation.
        - Any float: Use the specified value.
    view : bool, default=False
        If True, display the original and interpolated grids.

    Returns
    -------
    np.ndarray
        Interpolated 2D array.
    """
    spi = check_scipy_interpolate()
    if spi is None:
        return None
    
    if arr.ndim != 2:
        raise ValueError("`arr` must be a two-dimensional array.")

    x = np.arange(arr.shape[1])
    y = np.arange(arr.shape[0])
    xx, yy = np.meshgrid(x, y)

    # Mask invalid values
    mask = np.isnan(arr)
    valid = ~mask
    if not np.any(valid):
        raise ValueError("Array contains only NaNs.")

    # Perform interpolation
    interpolated = spi.griddata(
        points=(xx[valid], yy[valid]),
        values=arr[valid],
        xi=(xx, yy),
        method=method,
        fill_value=np.nan
    )

    if fill_value == 'auto':
        # Simple fill: forward and backward fill along both axes
        df_interp = pd.DataFrame(interpolated)
        df_interp = df_interp.fillna(method='ffill').fillna(method='bfill')
        df_interp = df_interp.fillna(axis=1, method='ffill').fillna(
            axis=1, method='bfill')
        interpolated = df_interp.to_numpy()
    elif isinstance(fill_value, (int, float)):
        interpolated[np.isnan(interpolated)] = fill_value

    if view:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(arr, interpolation='nearest', cmap='viridis')
        axes[0].set_title('Original Grid')
        axes[1].imshow(interpolated, interpolation='nearest', cmap='viridis')
        axes[1].set_title('Interpolated Grid')
        plt.show()

    return interpolated

def _select_fill_method(method):
    """
    Helper function to standardize the fill method input.
    Maps various user-provided method aliases to standardized method keys.
    
    Parameters
    ----------
    method : str
        The fill method specified by the user. Can be one of the following:
        - Forward fill: 'forward', 'ff', 'fwd'
        - Backward fill: 'backward', 'bf', 'bwd'
        - Both: 'both', 'ffbf', 'fbwf', 'bff', 'full'
    
    Returns
    -------
    str
        Standardized fill method key: 'ff', 'bf', or 'both'.
    
    Raises
    ------
    ValueError
        If the provided method is not recognized.
    """
    # Convert method to lowercase to ensure case-insensitive matching
    method_lower = method.lower()
    
    # Define mappings for forward fill aliases
    forward_aliases = {'forward', 'ff', 'fwd'}
    # Define mappings for backward fill aliases
    backward_aliases = {'backward', 'bf', 'bwd'}
    # Define mappings for both forward and backward fill aliases
    both_aliases = {'both', 'ffbf', 'fbwf', 'bff', 'full'}
    
    # Determine the standardized method based on aliases
    if method_lower in forward_aliases:
        return 'ff'   # Forward fill
    elif method_lower in backward_aliases:
        return 'bf'   # Backward fill
    elif method_lower in both_aliases:
        return 'both' # Both forward and backward fill
    else:
        # Raise an error if the method is not recognized
        raise ValueError(
            f"Invalid fill method '{method}'. "
            "Choose from 'forward', 'ff', 'fwd', 'backward', 'bf', 'bwd', "
            "'both', 'ffbf', 'fbwf', 'bff', or 'full'."
        )
        
def fill_NaN(arr, method='ff'):
    """
    Fill NaN values in an array-like structure using specified methods.
    Handles numeric and non-numeric data separately to preserve data
    integrity.

    Parameters
    ----------
    arr : array-like, pandas.DataFrame, or pandas.Series
        The input data structure containing NaN values to be filled.
    method : str, default ``'ff'``
        The method to use for filling NaN values. Accepted values:

        - Forward fill: ``'forward'``, ``'ff'``, ``'fwd'``
        - Backward fill: ``'backward'``, ``'bf'``, ``'bwd'``
        - Both: ``'both'``, ``'ffbf'``, ``'fbwf'``, ``'bff'``, ``'full'``

    Returns
    -------
    array-like, pandas.DataFrame, or pandas.Series
        The input data structure with NaN values filled according to the specified
        method.

    Raises
    ------
    ValueError
        If the provided fill method is not recognized.

    Notes
    -----
    Mathematically, the function performs:

    .. math::
        \text{Filled\_array} =
        \begin{cases}
            \text{fillNaN(arr, method)} & \text{if arr is numeric} \\
            \text{concat(fillNaN(numeric\_parts, method), non\_numeric\_parts)} &
            \text{otherwise}
        \end{cases}

    This ensures that non-numeric data remains unaltered while NaN values in
    numeric columns are appropriately filled.

    Examples
    --------
    >>> from gofast.utils.base_utils import fill_NaN
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, np.nan, 4],
    ...     'B': ['x', np.nan, 'y', 'z']
    ... })
    >>> fill_NaN(df, method='ff')
         A    B
    0  1.0    x
    1  2.0    x
    2  2.0    y
    3  4.0    z

    Notes
    -----
    The function preserves the original structure of the input array by utilizing
    ``array_preserver``. Numeric columns are filled using the specified method,
    while non-numeric columns remain unchanged.

    See Also
    --------
    gofast.core.array_manager.array_preserver:
        Preserves and restores array structures.
    gofast.core.array_manager.to_array:
        Converts input to a pandas-compatible array-like structure.
    gofast.core.checks.is_numeric_dtype:
        Checks if the array has numeric data types.
    gofast.utils.base_utils.fillNaN :
        Core function to fill NaN values in numeric data.

    """

    # Step 1: Standardize the fill method using the helper function
    standardized_method = _select_fill_method(method)
    
    # Step 2: Collect the original array's properties to preserve its structure
    collected = array_preserver(arr, action='collect')
    
    # Step 3: Convert the input to a pandas-compatible array-like structure
    # This ensures consistent handling across different input types
    arr_converted = to_array(arr)
    arr_converted = to_numeric_dtypes(arr_converted)
    # Step 4: Check if the entire array has a numeric dtype
    if is_numeric_dtype(arr_converted):
        # If all data is numeric, apply the fillNaN function directly
        array_filled = fillNaN(arr_converted, method=standardized_method)
    else:
        if not isinstance (arr_converted, (pd.Series, pd.DataFrame)): 
            # For other array-like types (e.g., lists, tuples), 
            # convert to pandas Series
            # to leverage pandas' fill capabilities
            try: 
                arr_converted = pd.Series(arr_converted) # expert one1d 
            except : 
                arr_converted = pd.DataFrame (arr_converted) # two dimensional
        
        # If there are non-numeric data types, handle numeric
        # and non-numeric separately
        if isinstance(arr_converted, pd.DataFrame):
            # Identify numeric columns
            numeric_cols = select_dtypes(
                arr_converted, incl=[np.number], 
                return_columns=True 
                )
            # numeric_cols = arr_converted.select_dtypes(
            #     include=[np.number]).columns
            # Identify non-numeric columns
            non_numeric_cols = arr_converted.columns.difference(
                numeric_cols)
            
            # Apply fillNaN to numeric columns
            if numeric_cols: 
                filled_numeric = fillNaN(
                    arr_converted[numeric_cols], method=standardized_method)
            else: 
                filled_numeric= pd.DataFrame()
            
            # Fill non-numeric columns with forward and backward fill (if requested)
            filled_non_numeric = arr_converted[non_numeric_cols]
            if non_numeric_cols.any():
                if 'ff' in standardized_method:
                    filled_non_numeric = filled_non_numeric.ffill(axis=0)
                elif 'bf' in standardized_method:
                    filled_non_numeric = filled_non_numeric.bfill(axis=0) 
                else: # both 
                    filled_non_numeric = filled_non_numeric.ffill(axis=0)
                    filled_non_numeric = filled_non_numeric.bfill(axis=0) 
                    
            # Combine the filled numeric data with the untouched non-numeric data
            array_filled = pd.concat(
                [filled_numeric, filled_non_numeric], axis=1)
            # Ensure the original column order is preserved
            array_filled = array_filled[arr_converted.columns]
        
        elif isinstance(arr_converted, pd.Series):
            
            if is_numeric_dtype(arr_converted):
               # If the Series is numeric, apply fillNaN
                array_filled = fillNaN(
                    arr_converted, method=standardized_method
                    )
            else:
                # If the Series is not numeric
                # Fill non-numeric Series with forward 
                # and backward fill (if requested)
                array_filled = arr_converted.copy()
                if 'ff' in standardized_method:
                    array_filled = array_filled.ffill()
                elif 'bf' in standardized_method:
                    array_filled = array_filled.bfill()
                else:
                    array_filled = array_filled.ffill()
                    array_filled = array_filled.bffill()
                    
    # Step 5: Attempt to restore the original array
    # structure using the collected properties
    collected['processed'] = [array_filled]
    try:
        # Restore the original structure 
        # (e.g., DataFrame, Series) with filled data
        array_restored = array_preserver(
            collected, action='restore',
            solo_return= True 
            )
    except Exception:
        # If restoration fails, return the filled
        # array without structure preservation
        array_restored = array_filled
    
    # Step 6: Return the filled and structure-preserved array
    return array_restored

def fillNaN0(
    arr: Union[ArrayLike, Series, DataFrame], 
    method: str = 'ff'
    ) -> Union[ArrayLike, Series, DataFrame]:
    """
    Fill NaN values in a numpy array, pandas Series, or pandas DataFrame 
    using specified methods for forward filling, backward filling, or both.

    Parameters
    ----------
    arr : Union[np.ndarray, pd.Series, pd.DataFrame]
        The input data containing NaN values to be filled. This can be a numpy
        array, pandas Series,  or DataFrame expected to contain numeric data
        types.
    method : str, optional
        The method used for filling NaN values. Valid options are:
        - 'ff': forward fill (default)
        - 'bf': backward fill
        - 'both': applies both forward and backward fill sequentially

    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        The array with NaN values filled according to the specified method. 
        The return type matches the input type (numpy array, Series, or DataFrame).

    Raises
    ------
    ValueError
        If an unsupported filling method is specified.

    Notes
    -----
    The function is designed to handle scenarios where NaN values are 
    framed between valid numbersand at the ends of the dataset. Forward fill 
    (``ff``) is preferred when NaNs are at the end of the data, while backward 
    fill (``bf``) is better suited for NaNs at the beginning. The `both` method
    combines both fills but at a higher computation cost.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.utils.base_utils import fillNaN 
    >>> arr2d = np.random.randn(7, 3)
    >>> arr2d[[0, 2, 3, 3], [0, 2, 1, 2]] = np.nan
    >>> print(arr2d)
    [[       nan -0.74636104  1.12731613]
     [0.48178017 -0.18593812 -0.67673698]
     [0.17143421 -2.15184895        nan]
     [-0.6839212         nan        nan]]
    >>> print(fillNaN(arr2d))
    [[       nan -0.74636104  1.12731613]
     [0.48178017 -0.18593812 -0.67673698]
     [0.17143421 -2.15184895 -2.15184895]
     [-0.6839212 -0.6839212 -0.6839212]]
    >>> print(fillNaN(arr2d, 'bf'))
    [[-0.74636104 -0.74636104  1.12731613]
     [0.48178017 -0.18593812 -0.67673698]
     [0.17143421 -2.15184895        nan]
     [-0.6839212         nan        nan]]
    >>> print(fillNaN(arr2d, 'both'))
    [[-0.74636104 -0.74636104  1.12731613]
     [0.48178017 -0.18593812 -0.67673698]
     [0.17143421 -2.15184895 -2.15184895]
     [-0.6839212 -0.6839212 -0.6839212]]

    References
    ----------
    Further details can be found at:
    https://pyquestions.com/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """

    # Store column or series name for restoration later if needed
    name_or_columns = None 
    
    # Convert the input array to numpy if it doesn't already support numpy-like operations
    if not hasattr(arr, '__array__'): 
        arr = np.array(arr)
    
    # Convert to a pandas-compatible structure if necessary
    # and ensure numeric dtype
    arr = to_array(arr)
    has_numeric_dtype = is_numeric_dtype(arr, to_array=True)
    
    if not has_numeric_dtype: 
        warnings.warn(
            "Non-numeric data detected. Note `fillNaN` operates only with "
            "numeric data. To deal with non-numeric data or both,"
            " use 'fill_NaN' instead."
        )
    
    # Handle non-numeric data if needed
    arr = _handle_non_numeric(arr, action='fill missing values NaN')
    
    # If the array is a pandas DataFrame or Series, store column names for later restoration
    if isinstance(arr, (pd.Series, pd.DataFrame)): 
        name_or_columns = arr.name if isinstance(arr, pd.Series) else arr.columns
        arr = arr.to_numpy()  # Convert to numpy array for easier manipulation
    
    # Define the forward fill function
    def ffill(arr): 
        """ Apply forward fill. """
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        return arr[np.arange(idx.shape[0])[:, None], idx]
    
    # Define the backward fill function
    def bfill(arr): 
        """ Apply backward fill. """
        idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
        idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
        return arr[np.arange(idx.shape[0])[:, None], idx]
    
    # Standardize the fill method (i.e., ensure method
    # is lowercase and stripped of extra spaces)
    method = _select_fill_method(str(method).lower().strip())
    
    # Reshape if array is one-dimensional
    if arr.ndim == 1: 
        arr = reshape(arr, axis=1)  
    
    # Create a mask identifying NaN values
    mask = np.isnan(arr)
    
    # Apply both forward and backward fill if requested
    if method == 'both': 
        arr = ffill(arr) 
        arr = bfill(arr)
    
    # Apply forward or backward fill depending on the method
    elif method in ('bf', 'ff'): 
        arr = ffill(arr) if method == 'ff' else bfill(arr)
    
    # Restore the original structure (Series/DataFrame) if necessary
    if name_or_columns is not None: 
        if isinstance(name_or_columns, str):
            arr = pd.Series(arr.squeeze(), name=name_or_columns)
        else:
            arr = pd.DataFrame(arr, columns=name_or_columns)
    
    return arr

#XXX TODO: FIX NaN 
def fillNaN(
    arr: Union[ArrayLike, Series, DataFrame], 
    method: str = 'ff'
) -> Union[ArrayLike, Series, DataFrame]:
    """
    Fill NaN values in a numpy array, pandas Series, or pandas DataFrame 
    using specified methods for forward filling, backward filling, or both.

    Parameters
    ----------
    arr : Union[np.ndarray, pd.Series, pd.DataFrame]
        The input data containing NaN values to be filled. This can be a numpy
        array, pandas Series, or DataFrame expected to contain numeric data.
        
    method : str, optional
        The method used for filling NaN values. Valid options are:
        - 'ff': forward fill (default)
        - 'bf': backward fill
        - 'both': applies both forward and backward fill sequentially

    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        The array with NaN values filled according to the specified method. 
        The return type matches the input type (numpy array, Series, or DataFrame).
    """
    
    name_or_columns = None 
    
    # Convert to numpy array if it doesn't have numpy-like methods
    if not hasattr(arr, '__array__'): 
        arr = np.array(arr)
    
    arr = to_array(arr) 
    has_numeric_dtype = is_numeric_dtype(arr, to_array=True)
    
    # Handle non-numeric data and issue a warning if necessary
    if not has_numeric_dtype:
        warnings.warn(
            "Non-numeric data detected. Note `fillNaN`"
            " operates only with numeric data. "
            "To deal with non-numeric data or both,"
            " use 'fill_NaN' instead."
        )
        
    arr = _handle_non_numeric(arr, action='fill missing values NaN')

    if isinstance(arr, (pd.Series, pd.DataFrame)): 
        # Preserve column names for restoration 
        # if it's a pandas Series or DataFrame
        name_or_columns = arr.name if isinstance(
            arr, pd.Series) else arr.columns
        # Convert to numpy array for easier manipulation
        arr = arr.to_numpy()  
    
    # Forward fill function
    def ffill(arr): 
        """ Apply forward fill. """
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        return arr[np.arange(idx.shape[0])[:, None], idx]
    
    # Backward fill function
    def bfill(arr): 
        """ Apply backward fill. """
        idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
        idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
        return arr[np.arange(idx.shape[0])[:, None], idx]
    
    # Standardize method (ensure lowercase and stripped of extra spaces)
    method = _select_fill_method(str(method).lower().strip())
    
    # Reshape if array is one-dimensional
    if arr.ndim == 1: 
        arr = reshape(arr, axis=1)  
    
    # Create a mask identifying NaN values
    mask = np.isnan(arr)
    
    # Apply both forward and backward fill if requested
    if method == 'both': 
        arr = ffill(arr) 
        arr = bfill(arr)
    
    # Apply forward or backward fill depending on the method
    elif method in ('bf', 'ff'): 
        arr = ffill(arr) if method == 'ff' else bfill(arr)
    
    # Handle DataFrame/Series restoration
    if name_or_columns is not None: 
        if isinstance(name_or_columns, str):
            arr = pd.Series(arr.squeeze(), name=name_or_columns)
        else:
            arr = pd.DataFrame(arr, columns=name_or_columns)
    
    return arr

def convert_array_dimensions(
        *arrays, target_dim=1, new_shape=None, orient='row'):
    """
    Convert arrays between 1D, 2D, and higher dimensions. 
    
    Function dynamically adjusts the dimensions of the input arrays based on 
    the target dimension specified and can reshape arrays according to a new 
    shape.

    Parameters:
    -----------
    *arrays : tuple of array-like
        Variable number of array-like structures (lists, tuples, np.ndarray).
    target_dim : int, optional
        The target dimension to which the arrays should be converted. Options 
        are primarily 1 or 2.
        Default is 1, which flattens arrays to 1D.
    new_shape : tuple of ints, optional
        The new shape for the array when converting to 2D or reshaping a
        higher-dimensional array.
        If None, defaults are used (e.g., flattening to 1D or reshaping to 
                                    one row per array in 2D).
    orient : str or int, optional
        Specifies the orientation for reshaping the array into 2D when no 
        new_shape is provided. 
        Accepts 'row' or 0 to reshape the array into a single row 
        (default behavior), and 'column' or 1 
        to reshape the array into a single column. This parameter determines 
        the structure of the 2D array:
        - 'row' or 0: The array is reshaped to have one row with multiple 
          columns.
        - 'column' or 1: The array is reshaped to have one column with 
           multiple rows.
        If an invalid option is provided, a ValueError is raised.

    Returns:
    --------
    list
        A list of arrays converted to the specified target dimension.

    Raises:
    -------
    ValueError
        If the target_dim is not supported or if conversion is not feasible 
        for the given dimensions.
        
    Examples
    --------
    >>> from gofast.utils.base_utils import convert_array_dimensions
    >>> import numpy as np

    # Example 1: Convert a 1D array to a 2D array with a specific shape
    >>> array_1d = np.array([1, 2, 3, 4, 5])
    >>> convert_array_dimensions(array_1d, target_dim=2, new_shape=(5, 1))
    [array([[1],
            [2],
            [3],
            [4],
            [5]])]

    # Example 2: Flatten a 2D array to 1D
    >>> array_2d = np.array([[1, 2, 3], [4, 5, 6]])
    >>> convert_array_dimensions(array_2d, target_dim=1)
    [array([1, 2, 3, 4, 5, 6])]

    # Example 3: Convert a 1D array to a default 2D array (one row)
    >>> array_1d = np.array([7, 8, 9, 10])
    >>> convert_array_dimensions(array_1d, target_dim=2)
    [array([[ 7,  8,  9, 10]])]

    # Example 4: Attempt to reshape a 1D array into an 
    # incompatible 2D shape (should raise an error)
    >>> array_1d = np.array([1, 2, 3, 4, 5, 6])
    >>> convert_array_dimensions(array_1d, target_dim=2, new_shape=(2, 4))
    Traceback (most recent call last):
      ...
    ValueError: Cannot reshape array of size 6 into shape (2, 4)
    """
    converted_arrays = []
    for arr in arrays:
        # Ensure input is converted to a NumPy array for manipulation
        array = np.array(arr) 

        if target_dim == 1:
            # Flatten the array to 1D
            converted_arrays.append(array.ravel())
        elif target_dim == 2:
            if new_shape is not None:
                # Reshape according to the provided new_shape
                try:
                    converted_arrays.append(array.reshape(new_shape))
                except ValueError as e:
                    raise ValueError(f"Cannot reshape array of size {array.size}"
                                     f" into shape {new_shape}") from e
            else:
                # Skip reshaping if the array is already 2D and no new shape is provided
                if array.ndim == 2 and array.shape == (1, -1) or array.shape == (-1, 1):
                    converted_arrays.append(array)
                else:
                    # Default 2D shape based on orientation
                    if orient == 'row' or orient == 0:
                        converted_arrays.append(array.reshape(1, -1))
                    elif orient == 'column' or orient == 1:
                        converted_arrays.append(array.reshape(-1, 1))
                    else:
                        raise ValueError("orient must be 'row', 'column', 0, or 1.")
        else:
            raise ValueError(
                f"Invalid target dimension {target_dim}. Only 1 or 2 are supported.")

    return converted_arrays

def filter_nan_entries(
    nan_policy, *listof, 
    sample_weights=None, 
    mode='strict', 
    trim_weights=False
    ):
    """
    Filters out NaN values from multiple lists of lists, or arrays, 
    based on the specified NaN handling policy ('omit', 'propagate', 'raise'), 
    and adjusts the sample weights accordingly if provided.

    This function is particularly useful when preprocessing data for
    machine learning algorithms that do not support NaN values or when NaN values
    signify missing data that should be excluded from analysis.

    Parameters
    ----------
    nan_policy : {'omit', 'propagate', 'raise'}
        The policy for handling NaN values.
        - 'omit': Exclude NaN values from all lists in `listof`. 
        - 'propagate': Keep NaN values, which may result in NaN values in output.
        - 'raise': If NaN values are detected, raise a ValueError.
    *listof : array-like sequences
        Variable number of list-like sequences from which NaN values are to be 
        filtered out.  Each sequence in `listof` must have the same length.
    sample_weights : array-like, optional
        Weights corresponding to the elements in each sequence in `listof`.
        Must have the same length as the sequences. If `nan_policy` is 'omit',
        weights are adjusted to match the filtered sequences.
    mode : str, optional
        Specifies the mode of NaN filtering:
        - 'strict': Indices are retained only if all corresponding elements 
          across sequences are non-NaN.
        - 'soft': Indices are retained if any corresponding element across 
          sequences is non-NaN.
    trim_weights : bool, optional
        If True and `sample_weights` is provided, trims the sample_weights to
        match the length of the filtered data when the 'soft' mode results in
        fewer data points than there are weights.
        
    Returns
    -------
    tuple of lists
        A tuple containing the filtered list-like sequences as per the specified
        `nan_policy`.
        If `nan_policy` is 'omit', sequences with NaN values removed are 
        returned.
    np.ndarray or None
        The adjusted sample weights matching the filtered sequences if 
        `nan_policy` is 'omit'.
        If `sample_weights` is not provided, None is returned.

    Raises
    ------
    ValueError
        If `nan_policy` is 'raise' and NaN values are present in any input 
        sequence.

    Examples
    --------
    >>> from gofast.utils.base_utils import filter_nan_entries
    >>> list1 = [1, 2, np.nan, 4]
    >>> list2 = [np.nan, 2, 3, 4]
    >>> weights = [0.5, 1.0, 1.5, 2.0]
    >>> filter_nan_entries('omit', list1, list2,sample_weights=weights,
                           mode="soft" ,)
    ([1.0, 2.0, 4.0], [2.0, 3.0, 4.0], array([0.5, 1. , 1.5, 2. ]))

    >>> filter_nan_entries('omit', list1, list2,sample_weights=weights, )
    ([2.0, 4.0], [2.0, 4.0], array([1., 2.]))
    >>> filter_nan_entries(
    ...     'omit', list1, list2,sample_weights=weights, mode="soft" ,
    ... trim_weights=True)
    ([1.0, 2.0, 4.0], [2.0, 3.0, 4.0], array([0.5, 1. , 1.5])) 
    >>> filter_nan_entries('raise', list1, list2)
    ValueError: NaN values present and nan_policy is 'raise'.

    Notes
    -----
    This function is designed to work with numerical data where NaN values
    may indicate missing data. It allows for flexible preprocessing by supporting
    multiple NaN handling strategies, making it suitable for various machine learning
    and data analysis workflows.

    When using 'omit' in 'soft' mode, it's important to ensure that all sequences
    in `listof`and the corresponding `sample_weights` (if provided) are 
    correctly aligned so that filtering does not introduce misalignments in the data.
    """
    # Validate nan_policy and check if any list contains nested objects
    nan_policy = is_valid_policies(nan_policy)
    
    for d in listof: 
        if contains_nested_objects(d): 
            # write a professionnal error message 
            raise ValueError ("filter_nan_entries does not support nested items.")
        
    # Prepare the data arrays
    arrays = [np.array(lst, dtype=float) for lst in listof]

    # Apply NaN filtering based on the selected policy
    filtered_arrays, non_nan_mask = _filter_nan_policies(arrays, nan_policy, mode)

    # Adjust sample weights if necessary
    if sample_weights is not None:
        sample_weights = _adjust_weights(
            sample_weights, non_nan_mask, mode, trim_weights, filtered_arrays
            )

    # Prepare the output to be returned
    filtered_listof = [arr.tolist() for arr in filtered_arrays]
    return (*filtered_listof, sample_weights
            ) if sample_weights is not None else tuple(filtered_listof)


def _filter_nan_policies(arrays, policy, mode):
    """Apply the NaN policy to filter the data arrays."""
    mode = parameter_validator("mode", target_strs={"soft", "strict"})(mode)
    if policy == 'omit':
        if mode == "strict":
            non_nan_mask = np.logical_and.reduce(~np.isnan(arrays))
            # Filter arrays using the computed mask
            filtered_arrays = [arr[non_nan_mask] for arr in arrays]
        elif mode == "soft":
           non_nan_mask = np.logical_or.reduce(~np.isnan(arrays))
           # Ensure that elements are NaN only if they are NaN across all arrays
           filtered_arrays =[arr[~np.isnan(arr)] for arr in  arrays]
    
    elif policy == 'raise':
        # Raise an error if any NaN values are detected
        if any(np.isnan(arr).any() for arr in arrays):
            raise ValueError("NaN values present and nan_policy is 'raise'.")
    else:
        # If 'propagate' is selected, return the original arrays
        filtered_arrays = arrays
    return filtered_arrays, non_nan_mask


def _adjust_weights(weights, mask, mode, adjusted_weights, arrays):
    """Adjust sample weights according to the filtered data."""
    if mode == 'soft' and adjusted_weights:
        # Determine the minimum length among the filtered arrays 
        # to ensure weights consistency
        min_length = min(len(arr) for arr in arrays)
    
        # Validate and adjust the non_nan_mask to ensure it aligns 
        # with the smallest array length
        if len(weights) > min_length:
            # Generate a warning if the original weights exceed the size
            # of available data
            warnings.warn(f"Adjusting sample weights from {len(weights)}"
                          f" to {min_length} due to mismatched data lengths.")
            
            # Resize the non_nan_mask to match the minimum length of the filtered data
            #       Indices where non_nan_mask is True
            valid_indices = np.where(mask)[0]  
            if valid_indices.size > min_length:
                # Truncate valid_indices to match the size of the shortest
                # filtered array
                valid_indices = valid_indices[:min_length]
            
            # Create a new mask that only includes the valid indices 
            # adjusted to min_length
            adjusted_non_nan_mask = np.zeros_like(mask, dtype=bool)
            adjusted_non_nan_mask[valid_indices] = True
            mask = adjusted_non_nan_mask
    
    # Update sample weights using the final adjusted non_nan_mask
    return np.asarray(weights)[mask]

def _flatten(items):
    """Helper function to flatten complex nested structures into a flat list."""
    for x in items:
        if isinstance(x, IterableInstance) and not isinstance(x, (str, bytes)):
            for sub_x in _flatten(x):
                yield sub_x
        else:
            yield x
            
def filter_nan_values(
    nan_policy, *data_lists, 
    sample_weights=None, error='raise',
    flatten=False, 
    preserve_type=False
    ):
    """
    Filters out NaN values from provided lists based on a specified policy,
    adjusts sample weights if necessary, and can optionally flatten the input lists.

    Parameters
    ----------
    nan_policy : {'omit', 'propagate', 'raise'}
        Determines how NaN values are handled:
        - 'omit': Removes NaN values from the lists.
        - 'propagate': Keeps NaN values in the lists.
        - 'raise': Raises an error if NaN values are found.
    data_lists : list of lists or arrays
        Variable number of list-like or array-like sequences from which NaN
        values are to be filtered based on the `nan_policy`.
    sample_weights : array-like, optional
        Weights corresponding to each entry in `data_lists`. If provided,
        they are adjusted to match the filtering operation.
    error : {'raise', 'warn'}, default 'raise'
        Error handling strategy if sample weights do not match the number
        of entries after filtering.
    flatten : bool, default False
        If True, flattens each list in `data_lists` before applying the filter.
    preserve_type : bool, default False
        If True, preserves the original type of nested structures within `data_lists`.

    Returns
    -------
    tuple
        A tuple containing the filtered lists. If `sample_weights` is provided,
        it is included as the last element of the tuple.

    Raises
    ------
    ValueError
        If `nan_policy` is 'raise' and NaN values are present.

    Examples
    --------
    >>> from gofast.utils.base_utils import filter_nan_values
    >>> list1 = [{2, 3}, {1, 2, np.nan}]
    >>> list2 = [{1, 2, 3}, {1, 2, 3, np.nan}]
    >>> weights = [0.5, 1.0, 1.5, 2.0]
    >>> print(filter_nan_values('omit', list1, list2, sample_weights=weights,
    ...             error="warn", flatten=True, preserve_type=True))
    ({1, 2, 3}, {1, 2, 3}, [0.5, 1.0, 1.5, 2.0])

    >>> filter_nan_values('raise', data1, error='warn')
    ValueError: NaN values present and nan_policy is 'raise'.

    Notes
    -----
    This function is useful for pre-processing data for algorithms that do not
    support NaN values or require array-like input with optional weighting. The
    `flatten` option is particularly useful for handling nested lists or arrays.
    """
    # Validate nan_policy
    if nan_policy not in ['omit', 'propagate', 'raise']:
        raise ValueError(f"Invalid nan_policy: {nan_policy}. Must be one of"
                         " 'omit', 'propagate', 'raise'.")
    
    for listof in data_lists: 
        if not contains_nested_objects(listof, strict=True):
            raise ValueError(
                "filter_nan_values expects each item in the data_lists"
                " to be a nested structure (e.g., list, set, or dict)."
                " Please ensure all elements are nested."
                )

    # Prepare arrays by flattening if requested
    arrays = _prepare_arrays(data_lists, flatten)

    # Apply NaN policy to arrays
    filtered_arrays, non_nan_mask = _apply_nan_policy(
        arrays, nan_policy, flatten)

    # Adjust sample weights according to the filtered data
    if sample_weights is not None:
        sample_weights = _adjust_sample_weights(
            sample_weights, non_nan_mask, error)

    # Prepare the output list, preserving types if required
    filtered_listof = _prepare_output(
        filtered_arrays, preserve_type, data_lists, flatten)

    # Return the filtered data, including sample weights if provided
    return (*filtered_listof, sample_weights) if sample_weights is not None else tuple(
        filtered_listof)

def adjust_weights(
    data_lengths, 
    original_weights, 
    mode='auto', 
    fill_value=None,
    normalize=False, 
    normalize_method='01'
    ):
    """
    Adjusts sample weights to match the lengths of filtered or transformed datasets.
    This function can handle scenarios where the filtered data is shorter than the
    original weights array, either by truncating or by filling the remaining weights.

    Normalization can be applied to adjust the scale of the weights after 
    adjusting their length.
    
    Parameters
    ----------
    data_lengths : list or array-like
        A list of integers representing the lengths of the filtered datasets.
        This is used to determine how to adjust the sample weights array.
    original_weights : array-like
        The original weights array that needs adjustment to match the filtered data.
    mode : str, optional
        Specifies the method for adjusting the weights:
        - 'auto': Automatically decides based on the data_lengths. If any length in
          data_lengths is less than the length of original_weights, it trims the weights;
          otherwise, it fills or repeats weights if fill_value is not None.
        - 'trim': Trims the weights to match the shortest length in data_lengths.
        - 'fill': Extends the weights to match the longest length in data_lengths,
          using fill_value for the new elements.
    fill_value : float or None, optional
        The value used to fill the weights array if 'fill' mode is active and the
        weights need to be extended. If None and 'fill' is required, an error will
        be raised.
    normalize : bool or str, optional
        If True normalizes the weights after adjusting their length.
        If False, no normalization is applied.
    normalize_method : str, optional
        Normalization method ('01' for 0-1 scaling, 'zscore' for Z-score, or 'sum'
        to scale by the sum of weights).
    Returns
    -------
    np.ndarray
        The adjusted weights array that matches the specified requirements based on the
        filtered data lengths.

    Raises
    ------
    ValueError
        If 'fill' mode is selected but fill_value is None and the weights need
        extending.

    Examples
    --------
    >>> from gofast.utils.base_utils import adjust_weights
    >>> weights = [1, 2, 3, 4, 5]
    >>> data_lengths = [3, 4]
    >>> adjust_weights(data_lengths, weights, match_mode='trim')
    array([1, 2, 3])

    >>> adjust_weights(data_lengths, weights, match_mode='fill', fill_value=0)
    array([1, 2, 3, 4, 5, 0])

    >>> adjust_weights(data_lengths, weights, mode='auto', fill_value=0, 
    ...                normalize=True, normalize_method='01')
    array([0.2, 0.4, 0.6, 0.8, 1. , 0. ])
    
    Notes
    -----
    Normalization is applied after adjusting the weights to ensure that the 
    processed weights are ready for use in weighted statistical analyses or models.
    The choice between trimming and filling depends on the analysis needs and 
    the nature of the data. Care should be taken with 'fill' mode as it 
    introduces artificial values.
    """
    # Validate input types and values
    if not isinstance(data_lengths, (list, tuple, np.ndarray)):
        raise TypeError("data_lengths must be a list, tuple, or numpy array.")
    if not isinstance(original_weights, (list, np.ndarray)):
        raise TypeError("original_weights must be a list or numpy array.")
    
    # Convert inputs to numpy arrays for uniformity
    data_lengths = np.array(data_lengths)
    original_weights = np.array(original_weights)
    
    # Determine the required length based on the match mode
    if mode == 'auto':
        max_length = max(data_lengths)
        min_length = min(data_lengths)
        required_length = max_length if len(original_weights) < max_length else min_length
    elif mode == 'trim':
        required_length = min(data_lengths)
    elif mode == 'fill':
        required_length = max(data_lengths)
    else:
        raise ValueError("Invalid match_mode. Choose from 'auto', 'trim', or 'fill'.")

    # Adjust weights according to the required length
    if len(original_weights) > required_length:
        adjusted_weights = original_weights[:required_length]
    elif len(original_weights) < required_length and mode == 'fill':
        if fill_value is None:
            raise ValueError("fill_value must be provided when extending"
                             " weights in 'fill' mode.")
        extra_length = required_length - len(original_weights)
        adjusted_weights = np.concatenate([original_weights, np.full(
            extra_length, fill_value)])
    else:
        adjusted_weights = original_weights

    # Normalize weights if requested
    if normalize:
        adjusted_weights = normalize_array(adjusted_weights, method=normalize_method)
    
    return adjusted_weights


def _prepare_output(filtered_arrays, preserve_type, original_lists, flatten):
    """
    Prepares the output list from filtered arrays, optionally preserving the 
    original data types of the nested structures.

    Parameters
    ----------
    filtered_arrays : list of arrays
        Arrays that have been filtered based on NaN policies.
    preserve_type : bool
        Whether to preserve the original types of nested structures in the output.
    original_lists : list of lists
        The original list of lists provided by the user, used for type reference
        if `preserve_type` is True.
   flatten : bool, default False
       If True, flattens each list in `data_lists` before applying the filter.
    Returns
    -------
    list
        The list of filtered arrays, with types preserved if `preserve_type` is True.

    Notes
    -----
    This function aids in maintaining the integrity of the data types in nested 
    structures when required by the user. If `preserve_type` is False, all nested
    structures are converted to lists. If True, the original data structure type 
    (e.g., set, list) is preserved based on the type of the first element in each
    corresponding original list.
    """
    filtered_listof = []
    for arr, original in zip(filtered_arrays, original_lists):
        original_type = type(original[0]) if original else list
        if preserve_type and not flatten:
            # Determine the type of the original nested structure
            filtered_sublistof = [original_type(subarr) for subarr in arr]
        else:
            # Convert all sub-arrays to lists
            filtered_sublistof = original_type(arr)
        filtered_listof.append(filtered_sublistof)
    return filtered_listof

def _prepare_arrays(listof, flatten_lst):
    """Prepares and converts lists to numpy arrays, handling flattening if specified."""
    arrays = []
    for lst in listof:
        if flatten_lst:
            # Flatten and then convert to numpy array
            flattened_list = list(_flatten(lst))
            arrays.append(np.array(flattened_list, dtype=object))
        else:
            # Apply filtering to each nested element directly
            arrays.append(np.array([np.array(list(item), dtype=object) if isinstance(
                item, (list, set)) else item for item in lst], dtype=object))
    
    return arrays

def _apply_nan_policy(arrays, nan_policy, flatten_lst):
    """Applies the specified NaN policy to the arrays."""
    filtered_arrays = []
    all_non_nan_mask = []
    for arr in arrays:
        if nan_policy == 'omit':
            if flatten_lst: 
                # Create a mask for non-NaN entries
                if arr.dtype == object:
                    non_nan_mask = np.array([not isinstance(
                        x, float) or not np.isnan(x) for x in arr])
                else:
                    non_nan_mask = ~np.isnan(arr)
                filtered_array = arr[non_nan_mask]
                filtered_arrays.append(filtered_array)
                
            else: 
               filtered_subarrays = []
               for subarr in arr:
                   if isinstance(subarr, np.ndarray):
                       # Apply the non-NaN mask to each sub-array
                       non_nan_mask = ~np.isnan(subarr.astype(float))
                       filtered_subarrays.append(subarr[non_nan_mask])
                   else:
                       # If it's not an array, just append as is
                       filtered_subarrays.append(subarr)
               filtered_arrays.append(filtered_subarrays) 

            all_non_nan_mask.append(non_nan_mask)
        elif nan_policy == 'raise' and np.isnan(arr.astype(float)).any():
            raise ValueError("NaN values present and nan_policy is 'raise'.")
        else:
            filtered_arrays.append(arr)
            
    return filtered_arrays, all_non_nan_mask

def _adjust_sample_weights(sample_weights, non_nan_mask, error):
    """Adjusts sample weights to match the filtered data's length."""
    try: 
        total_entries = sum(non_nan_mask) # if flatten_lst 
    except: 
        total_entries = sum(len(arr) for arr in non_nan_mask )
    if len(sample_weights) == total_entries:
        return np.asarray(sample_weights)[non_nan_mask]
    elif error == 'warn':
        # If the length of the sample weights does not match but is greater,
        # reduce the sample weights array to match the filtered data length.
        # Note: This may not always be the desired behavior, as it can lead to
        #       unexpected results if data misalignment occurs.
        #       Use with caution or consider whether this should raise an error instead.
        warnings.warn("Length of sample_weights does not match the number"
                      " of entries after NaN filtering.")
    elif error == 'raise':
        raise ValueError("Length of sample_weights must match the number of entries in listof.")
    return sample_weights


def filter_nan_from( *listof, sample_weights=None):
    """
    Filters out NaN values from multiple lists of lists, adjusting 
    sample_weights accordingly.

    Parameters
    ----------
    *listof : tuple of list of lists
        Variable number of list of lists from which NaN values need to be 
        filtered out.
    sample_weights : list or np.ndarray, optional
        Sample weights corresponding to each sublist in the input list of lists.
        Must have the same outer length as each list in *listof.

    Returns
    -------
    filtered_listof : tuple of list of lists
        The input list of lists with NaN values removed.
    adjusted_sample_weights : np.ndarray or None
        The sample weights adjusted to match the filtered data. Same length as
        the filtered list of lists.

    Examples
    --------
    >>> from gofast.utils.base_utils import filter_nan_from
    >>> list1 = [[1, 2, np.nan], [4, np.nan, 6]]
    >>> list2 = [[np.nan, 8, 9], [10, 11, np.nan]]
    >>> weights = [0.5, 1.0]
    >>> filtered_lists, adjusted_weights = filter_nan_from(
        list1, list2, sample_weights=weights)
    >>> print(filtered_lists)
    ([[1, 2], [4, 6]], [[8, 9], [10, 11]])
    >>> print(adjusted_weights)
    [0.5 1. ]

    Notes
    -----
    This function assumes that all lists in *listof and sample_weights have 
    compatible shapes. Each sublist is expected to correspond to a set of 
    sample_weights, which are adjusted based on the presence of NaNs in 
    the sublist.
    """
    import math 
    if sample_weights is not None and len(sample_weights) != len(listof[0]):
        raise ValueError(
            "sample_weights length must match the number of sublists in listof.")

    # Convert sample_weights to a numpy array for easier manipulation
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    filtered_listof = []
    valid_indices = set(range(len(listof[0])))  # Initialize with all indices as valid

    # Identify indices with NaNs across all lists
    for lst in listof:
        for idx, sublist in enumerate(lst):
            if any(math.isnan(item) if isinstance(item, (
                    float, np.floating)) else False for item in sublist):
                valid_indices.discard(idx)

    # Filter lists based on valid indices
    for lst in listof:
        filtered_list = [lst[idx] for idx in sorted(valid_indices)]
        filtered_listof.append(filtered_list)

    # Adjust sample_weights based on valid indices
    adjusted_sample_weights = sample_weights[
        sorted(valid_indices)] if sample_weights is not None else None

    return tuple(filtered_listof), adjusted_sample_weights

def standardize_input(*arrays):
    """
    Standardizes input formats for comparison metrics, converting input data
    into a uniform format of lists of sets. This function can handle a variety
    of input formats, including 1D and 2D numpy arrays, lists of lists, and
    tuples, making it versatile for tasks that involve comparing lists of items
    like ranking or recommendation systems.

    Parameters
    ----------
    *arrays : variable number of array-like or list of lists
        Each array-like argument represents a set of labels or items, such as 
        `y_true` and `y_pred`. The function is designed to handle:
        
        - 1D arrays where each element represents a single item.
        - 2D arrays where rows represent samples and columns represent items
          (for multi-output scenarios).
        - Lists of lists or tuples, where each inner list or tuple represents 
          a set of items for a sample.

    Returns
    -------
    standardized : list of lists of set
        A list containing the standardized inputs as lists of sets. Each outer
        list corresponds to one of the input arrays, and each inner list 
        corresponds to a sample within that array.

    Raises
    ------
    ValueError
        If the lengths of the input arrays are inconsistent.
    TypeError
        If the inputs are not array-like, lists of lists, or lists of tuples,
        or if an ndarray has more than 2 dimensions.

    Examples
    --------
    >>> from numpy import array
    >>> from gofast.utils.base_utils import standardize_input
    >>> y_true = [[1, 2], [3]]
    >>> y_pred = array([[2, 1], [3]])
    >>> standardized_inputs = standardize_input(y_true, y_pred)
    >>> for standardized in standardized_inputs:
    ...     print([list(s) for s in standardized])
    [[1, 2], [3]]
    [[2, 1], [3]]

    >>> y_true_1d = array([1, 2, 3])
    >>> y_pred_1d = [4, 5, 6]
    >>> standardized_inputs = standardize_input(y_true_1d, y_pred_1d)
    >>> for standardized in standardized_inputs:
    ...     print([list(s) for s in standardized])
    [[1], [2], [3]]
    [[4], [5], [6]]

    Notes
    -----
    The function is particularly useful for preprocessing inputs to metrics 
    that require comparison of sets of items across samples, such as precision
    at K, recall at K, or NDCG. By standardizing the inputs to lists of sets,
    the function facilitates consistent handling of these computations
    regardless of the original format of the input data. This standardization
    is critical when working with real-world data, which can vary widely in
    format and structure.
    """
    standardized = []
    for data in arrays:
        # Transform ndarray based on its dimensions
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                standardized.append([set([item]) for item in data])
            elif data.ndim == 2:
                standardized.append([set(row) for row in data])
            else:
                raise TypeError("Unsupported ndarray shape. Must be 1D or 2D.")
        # Transform lists or tuples
        elif isinstance(data, (list, tuple)):
            if all(isinstance(item, (list, tuple, np.ndarray)) for item in data):
                standardized.append([set(item) for item in data])
            else:
                standardized.append([set([item]) for item in data])
        else:
            raise TypeError(
                "Inputs must be array-like, lists of lists, or lists of tuples.")
    
    # Check consistent length across all transformed inputs
    if any(len(standardized[0]) != len(arr) for arr in standardized[1:]):
        raise ValueError("All inputs must have the same length.")
    
    return standardized

def smart_rotation(ax):
    """
    Automatically adjusts the rotation of x-axis tick labels on a matplotlib
    axis object based on the overlap of labels. This function assesses the
    overlap by comparing the horizontal extents of adjacent tick labels. If
    any overlap is detected, it rotates the labels by 45 degrees to reduce
    or eliminate the overlap. If no overlap is detected, labels remain
    horizontal.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object for which to adjust the tick label rotation.

    Examples
    --------
    # Example of creating a simple time series plot with date overlap handling
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.dates import DateFormatter
    >>> from gofast.utils.base_utils import smart_rotation

    # Generate a date range and some random data
    >>> dates = pd.date_range(start="2020-01-01", periods=100, freq='D')
    >>> values = np.random.rand(100)

    # Create a DataFrame
    >>> df = pd.DataFrame({'Date': dates, 'Value': values})

    # Create a plot
    >>> fig, ax = plt.subplots()
    >>> ax.plot(df['Date'], df['Value'])
    >>> ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    # Apply smart rotation to adjust tick labels dynamically
    >>> smart_rotation(ax)

    # Show the plot
    >>> plt.show()

    Notes
    -----
    This function needs to be used in conjunction with matplotlib plots where
    the axis ('ax') is already set up with tick labels. It is especially useful
    in time series and other plots where the x-axis labels are dates or other
    large strings that may easily overlap. Drawing the canvas (plt.gcf().canvas.draw())
    is necessary to render the labels and calculate their positions, which may
    impact performance for very large plots or in tight loops.
    
    """
    
    # Draw the canvas to get the labels rendered, which is necessary for calculating overlap
    plt.gcf().canvas.draw()

    # Retrieve the x-axis tick labels and their extents
    labels = [label.get_text() for label in ax.get_xticklabels()]
    tick_locs = ax.get_xticks()  # get the locations of the current ticks
    label_extents = [label.get_window_extent() for label in ax.get_xticklabels()]

    # Check for overlap by examining the extents
    overlap = False
    num_labels = len(label_extents)
    for i in range(num_labels - 1):
        if label_extents[i].xmax > label_extents[i + 1].xmin:
            overlap = True
            break

    # Apply rotation if overlap is detected
    rotation = 45 if overlap else 0

    # Set the locator before setting labels
    ax.xaxis.set_major_locator(FixedLocator(tick_locs))
    ax.set_xticklabels(labels, rotation=rotation)
    
def select_features(
    data: Union[DataFrame, dict, np.ndarray, list],
    features: Optional[Union[List[str], Pattern, Callable[[str], bool]]] = None,
    dtypes_inc: Optional[Union[str, List[str]]] = None,
    dtypes_exc: Optional[Union[str, List[str]]] = None,
    coerce: bool = False,
    columns: Optional[List[str]] = None,
    verify_integrity: bool = False,
    parse_features: bool = False,
    include_missing: Optional[bool] = None,
    exclude_missing: Optional[bool] = None,
    transform: Optional[Union[Callable[[pd.Series], Any],
                              Dict[str, Callable[[pd.Series], Any]]]] = None,
    regex: Optional[Union[str, Pattern]] = None,
    callable_selector: Optional[Callable[[str], bool]] = None,
    inplace: bool = False,
    **astype_kwargs: Any
) -> DataFrame:
    """
    Selects features from a dataset based on various criteria and returns
    a new DataFrame.

    .. math::
        \text{Selected Columns} = 
        \text{Features Selection Criteria Applied to } C

    Where:
    - \( C = \{c_1, c_2, \dots, c_n\} \) is the set of columns in the data.
    - Features Selection Criteria include feature names, data types, regex patterns,
      callable selectors, and missing data conditions.

    Parameters
    ----------
    data : Union[pd.DataFrame, dict, np.ndarray, list]
        The dataset from which to select features. Can be a pandas DataFrame, a 
        dictionary, a NumPy array, or a list of dictionaries/lists.
    features : Optional[Union[List[str], Pattern, Callable[[str], bool]]], default=None
        Specific feature names to select. Can also be a regex pattern or a callable
        that takes a column name and returns ``True`` if the column should be selected.
    dtypes_inc : Optional[Union[str, List[str]]], default=None
        The data type(s) to include in the selection. Possible values are the same 
        as for the pandas ``include`` parameter in ``select_dtypes``.
    dtypes_exc : Optional[Union[str, List[str]]], default=None
        The data type(s) to exclude from the selection. Possible values are the same 
        as for the pandas ``exclude`` parameter in ``select_dtypes``.
    coerce : bool, default=False
        If ``True``, numeric columns are coerced to the appropriate types without
        selection, ignoring ``features``, ``dtypes_inc``, and ``dtypes_exc`` parameters.
    columns : Optional[List[str]], default=None
        Column names to use if ``data`` is a NumPy array or a list without column
        names.
    verify_integrity : bool, default=False
        Verifies the data type integrity and converts data to the correct types if 
        necessary.
    parse_features : bool, default=False
        Parses string features and converts them to an iterable object (e.g., lists).
    include_missing : Optional[bool], default=None
        If ``True``, includes only columns with missing values.
        If ``False``, excludes columns with missing values.
    exclude_missing : Optional[bool], default=None
        If ``True``, excludes columns with any missing values.
    transform : Optional, default=None
        Function or dictionary of functions to apply to the selected columns.
        If a dictionary is provided, keys should correspond to column names.
    regex : Optional[Union[str, Pattern]], default=None
        Regular expression pattern to select columns.
    callable_selector : Optional[Callable[[str], bool]], default=None
        A callable that takes a column name and returns ``True`` if the column should
        be selected.
    inplace : bool, default=False
        If ``True``, modifies the data in place. Otherwise, returns a new DataFrame.
    **astype_kwargs : Any
        Additional keyword arguments for ``pandas.DataFrame.astype``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the selected features.

    Raises
    ------
    ValueError
        If no columns match the selection criteria and ``coerce`` is ``False``.
    TypeError
        If ``regex`` is not a string or compiled regex pattern.
        If ``callable_selector`` is not a callable.
        If ``transform`` is not a callable or a dictionary of callables.
        If provided parameters are of incorrect types.

    Examples
    --------
    >>> from gofast.utils.base_utils import select_features
    >>> import pandas as pd
    >>> import re
    >>> import numpy as np
    >>> data = {
    ...     "Color": ['Blue', 'Red', 'Green'],
    ...     "Name": ['Mary', "Daniel", "Augustine"],
    ...     "Price ($)": ['200', "300", "100"],
    ...     "Discount": [20, 30, np.nan]
    ... }
    >>> select_features(data, dtypes_inc='number', verify_integrity=True)
       Price ($)  Discount
    0      200.0      20.0
    1      300.0      30.0
    2      100.0       NaN

    >>> select_features(data, features=['Color', 'Price ($)'])
       Color Price ($)
    0   Blue       200
    1    Red       300
    2  Green       100

    >>> select_features(
    ...     data,
    ...     regex='^Price|Discount$',
    ...     transform={'Price ($)': lambda x: x / 100}
    ... )
       Price ($)  Discount
    0        2.0        20
    1        3.0        30
    2        1.0         NaN

    >>> select_features(
    ...     data,
    ...     callable_selector=lambda col: col.startswith('C')
    ... )
       Color
    0   Blue
    1    Red
    2  Green

    Notes
    -----
    - This function is particularly useful in data preprocessing pipelines
      where the presence of certain features is critical for subsequent
      analysis or modeling steps.
    - When using regex patterns, ensure that the pattern accurately reflects
      the intended column names to avoid unintended matches.
    - The callable provided to ``callable_selector`` should accept a single string
      argument (the column name) and return a boolean indicating whether the column
      should be selected.
    - Transformation functions should be designed to handle the data types of
      the respective columns to avoid runtime errors.

    See Also
    --------
    validate_feature : Validates the existence of specified features in data.
    pandas.DataFrame.select_dtypes : For more information on how to use ``include`` and
        ``exclude`` parameters.
    pandas.DataFrame.astype : For information on data type conversion.

    References
    ----------
    .. [1] Pandas Documentation. "DataFrame.select_dtypes." 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
    .. [2] Python `re` Module. 
       https://docs.python.org/3/library/re.html
    .. [3] Pandas Documentation. "DataFrame.astype."
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    .. [4] Pandas Documentation. "DataFrame."
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    """

    # Convert input data to DataFrame if necessary
    df = build_data_if(
        data, columns =columns, force =True, raise_exception= True )

    # Handle coercion
    if coerce:
        numeric_cols = select_dtypes (df, dtypes ='number', return_columns =True)
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df

    # Handle verify_integrity
    if verify_integrity:
        df = to_numeric_dtypes(df )

    # Handle parse_features
    if parse_features:
        for col in df.select_dtypes(['object', 'string']):
            df[col] = df[col].apply(lambda x: x.split(',') if isinstance(x, str) else x)

    # Initialize mask for column selection
    mask = pd.Series([True] * df.shape[1], index=df.columns)

    # Select features by names, regex, or callable
    if features is not None:
        return validate_feature(df, features, ops='validate', error ='raise')
        
    # Select features by regex separately if provided
    if regex is not None:
        if isinstance(regex, str):
            pattern = re.compile(regex)
        elif isinstance(regex, re.Pattern):
            pattern = regex
        else:
            raise TypeError(
                "`regex` must be a string or a compiled regex pattern.")
        mask &= df.columns.str.match(pattern)

    # Select features by callable_selector separately if provided
    if callable_selector is not None:
        if not callable(callable_selector):
            raise TypeError("`callable_selector` must be a callable.")
        mask &= df.columns.to_series().apply(callable_selector)

    # Select features by data types to include
    if dtypes_inc is not None:
        included = select_dtypes(df, dtypes=dtypes_inc, return_columns=True) 
        mask &= df.columns.isin(included)

    # Select features by data types to exclude
    if dtypes_exc is not None:
        excluded = df.select_dtypes(exclude=dtypes_exc).columns
        mask &= df.columns.isin(excluded)

    # Handle missing data inclusion/exclusion
    if include_missing is True:
        cols_with_missing = df.columns[df.isnull().any()]
        mask &= df.columns.isin(cols_with_missing)
    if exclude_missing is True:
        cols_without_missing = df.columns[~df.isnull().any()]
        mask &= df.columns.isin(cols_without_missing)

    # Apply the mask to select columns
    selected_columns = df.columns[mask]
    if selected_columns.empty:
        if coerce:
            return df
        else:
            raise ValueError("No columns match the selection criteria.")

    df_selected = df[selected_columns].copy() 

    # Apply transformations if specified
    if transform is not None:
        if callable(transform):
            df_selected = transform(df_selected)
        elif isinstance(transform, dict):
            for col, func in transform.items():
                if col in df_selected.columns:
                    df_selected[col] = df_selected[col].apply(func)
                else:
                    raise KeyError(
                        f"Column '{col}' not found in the selected DataFrame.")
        else:
            raise TypeError(
                "`transform` must be a callable or a dictionary of callables.")

    # Change data types as specified
    if astype_kwargs:
        df_selected = df_selected.astype(**astype_kwargs)

    return df_selected

def speed_rowwise_process(
    data, 
    func, 
    n_jobs=-1
    ):
    """
    Processes a large dataset by applying a complex function to each row. 
    
    Function utilizes parallel processing to optimize for speed.

    Parameters
    ----------
    data : pd.DataFrames
        The large dataset to be processed. Assumes the 
        dataset is a Pandas DataFrame.

    func : function
        A complex function to apply to each row of the dataset. 
        This function should take a row of the DataFrame as 
        input and return a processed row.

    n_jobs : int, optional
        The number of jobs to run in parallel. -1 means using 
        all processors. Default is -1.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    Example
    -------
    >>> def complex_calculation(row):
    >>>     # Example of a complex row-wise calculation
    >>>     return row * 2  # This is a simple placeholder for demonstration.
    >>>
    >>> large_data = pd.DataFrame(np.random.rand(10000, 10))
    >>> processed_data = speed_rowwise_process(large_data, complex_calculation)

    """
    # Function to apply `func` to each row in parallel
    def process_row(row):
        return func(row)

    # Using Joblib's Parallel and delayed to apply the function in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_row)(row) 
                                      for row in data.itertuples(index=False))

    # Converting results back to DataFrame
    processed_data = pd.DataFrame(results, columns=data.columns)
    return processed_data
    
def run_shell_command(command, progress_bar_duration=30, pkg=None):
    """
    Run a shell command with an indeterminate progress bar.

    This function will display a progress bar for a predefined duration while 
    the package installation command runs in a separate thread. The progress 
    bar is purely for visual effect and does not reflect the actual 
    progress of the installation.

    Keep in mind:
    
    This function assumes that you have tqdm installed (pip install tqdm).
    The actual progress of the installation isn't tracked; the progress bar 
    is merely for aesthetics.
    The function assumes the command is a blocking one 
    (like most pip install commands) and waits for it to complete.
    Adjust progress_bar_duration based on how long you expect the installation
    to take. If the installation finishes before the progress bar, the bar
    will stop early. If the installation takes longer, the bar will complete, 
    but the function will continue to wait until the installation is done.
    
    Parameters:
    -----------
    command : list
        The command to run, provided as a list of strings.

    progress_bar_duration : int
        The maximum duration to display the progress bar for, in seconds.
        Defaults to 30 seconds.
    pkg: str, optional 
        The name of package to install for customizing bar description. 

    Returns:
    --------
    None
    
    Example 
    -------
    >>> from gofast.utils.base_utils import run_shell_command 
    >>> run_shell_command(["pip", "install", "gofast"])
    """
    def run_command(command):
        subprocess.run(command, check=True)

    def show_progress_bar(duration):
        with tqdm(total=duration, desc="Installing{}".format( 
                '' if pkg is None else f" {str(pkg)}"), 
                  bar_format="{l_bar}{bar}", ncols=77, ascii=True)  as pbar:
            for i in range(duration):
                time.sleep(1)
                pbar.update(1)

    # Start running the command
    thread = threading.Thread(target=run_command, args=(command,))
    thread.start()

    # Start the progress bar
    show_progress_bar(progress_bar_duration)

    # Wait for the command to finish
    thread.join()


def download_file(url, filename , dstpath =None ):
    """download a remote file. 
    
    Parameters 
    -----------
    url: str, 
      Url to where the file is stored. 
    loadl_filename: str,
      Name of the local file 
      
    dstpath: Optional 
      The destination path to save the downloaded file. 
      
    Return 
    --------
    None, local_filename
       None if the `dstpath` is supplied and `local_filename` otherwise. 
       
    Example 
    ---------
    >>> from gofast.utils.base_utils import download_file
    >>> url = 'https://raw.githubusercontent.com/WEgeophysics/gofast/master/gofast/datasets/data/h.h5'
    >>> local_filename = 'h.h5'
    >>> download_file(url, local_filename, test_directory)    
    
    """
    import_optional_dependency("requests")
    import requests 
    print("{:-^70}".format(f" Please, Wait while {os.path.basename(filename)}"
                          " is downloading. ")) 
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    filename = os.path.join( os.getcwd(), filename) 
    
    if dstpath: 
         move_file( filename,  dstpath)
         
    print("{:-^70}".format(" ok! "))
    
    return None if dstpath else filename

def fancier_downloader(
    url: str,
    filename: str,
    dstpath: Optional[str] = None,
    check_size: bool = False,
    error: str = 'raise',
    verbose: bool = True
) -> Optional[str]:
    """
    Download a remote file with a progress bar and optional size verification.
    
    This function downloads a file from the specified ``url`` and saves it locally
    with the given ``filename``. It provides a visual progress bar during the
    download process and offers an option to verify the downloaded file's size
    against the expected size to ensure data integrity. Additionally, the function
    allows for moving the downloaded file to a specified destination directory.
    
    .. math::
        |S_{downloaded} - S_{expected}| < \epsilon
    
    where :math:`S_{downloaded}` is the size of the downloaded file,
    :math:`S_{expected}` is the size specified by the server,
    and :math:`\epsilon` is a small tolerance value.
    
    Parameters
    ----------
    url : str
        The URL from which to download the remote file.
    
    filename : str
        The desired name for the local file. This is the name under which the
        file will be saved after downloading.
    
    dstpath : Optional[str], default=None
        The destination directory path where the downloaded file should be saved.
        If ``None``, the file is saved in the current working directory.
    
    check_size : bool, default=False
        Whether to verify the size of the downloaded file against the expected
        size obtained from the server. This is useful for ensuring the integrity
        of the downloaded file. When ``True``, the function checks:
        
        .. math::
            |S_{downloaded} - S_{expected}| < \epsilon
        
        If the size check fails:
        
        - If ``error='raise'``, an exception is raised.
        - If ``error='warn'``, a warning is emitted.
        - If ``error='ignore'``, the discrepancy is ignored, and the function
          continues.
    
    error : str, default='raise'
        Specifies how to handle errors during the size verification process.
        
        - ``'raise'``: Raises an exception if the file size does not match.
        - ``'warn'``: Emits a warning and continues execution.
        - ``'ignore'``: Silently ignores the size discrepancy and proceeds.
    
    verbose : bool, default=True
        Controls the verbosity of the function. If ``True``, the function will
        print informative messages about the download status, including progress
        updates and success or failure notifications.
    
    Returns
    -------
    Optional[str]
        Returns ``None`` if ``dstpath`` is provided and the file is moved to the
        destination. Otherwise, returns the local filename as a string.
    
    Raises
    ------
    RuntimeError
        If the download fails and ``error`` is set to ``'raise'``.
    
    ValueError
        If an invalid value is provided for the ``error`` parameter.
    
    Examples
    --------
    >>> from gofast.utils.base_utils import fancier_downloader
    >>> url = 'https://example.com/data/file.h5'
    >>> local_filename = 'file.h5'
    >>> # Download to current directory without size check
    >>> fancier_downloader(url, local_filename)
    >>> 
    >>> # Download to a specific directory with size verification
    >>> fancier_downloader(
    ...     url, 
    ...     local_filename, 
    ...     dstpath='/path/to/save/', 
    ...     check_size=True, 
    ...     error='warn', 
    ...     verbose=True
    ... )
    >>> 
    >>> # Handle size mismatch by raising an exception
    >>> fancier_downloader(
    ...     url, 
    ...     local_filename, 
    ...     check_size=True, 
    ...     error='raise'
    ... )
    
    Notes
    -----
    - **Progress Bar**: The function uses the `tqdm` library to display a
      progress bar during the download. If `tqdm` is not installed, it falls
      back to a basic downloader without a progress bar.
    - **Directory Creation**: If the specified ``dstpath`` does not exist,
      the function will attempt to create it to ensure the file is saved
      correctly.
    - **File Integrity**: Enabling ``check_size`` helps in verifying that the
      downloaded file is complete and uncorrupted. However, it does not perform
      a checksum verification.
    
    See Also
    --------
    - :func:`requests.get` : Function to perform HTTP GET requests.
    - :func:`tqdm` : A library for creating progress bars.
    - :func:`os.makedirs` : Function to create directories.
    - :func:`gofast.utils.base_utils.check_file_exists` : Utility to check file
      existence.
    
    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M.,
           & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python.
           *Journal of Machine Learning Research*, 12, 2825-2830.
    .. [2] tqdm documentation. https://tqdm.github.io/
    """

    # Import necessary dependencies
    import_optional_dependency("requests")
    import requests

    if error not in ['ignore', 'warn', 'raise']: 
        raise ValueError("`error` parameter must be 'raise', 'warn', or 'ignore'.")
        
    try:
        from tqdm import tqdm  # Import tqdm for progress bar visualization
    except ImportError:
        # If tqdm is not installed, fallback to the basic download_file function
        if verbose:
            warnings.warn(
                "tqdm is not installed. Falling back"
                " to basic downloader without progress bar."
            )
        return download_file(url, filename, dstpath)
    
    try:
        # Initiate the HTTP GET request with streaming enabled
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Retrieve the total size of the file from the 'Content-Length' header
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # Define the chunk size (1 Kibibyte)
            
            # Initialize the progress bar with the total file size
            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit='iB',
                unit_scale=True,
                ncols=77,
                ascii=True,
                desc=f"Downloading {filename}"
            )
            
            # Open the target file in binary write mode
            with open(filename, 'wb') as file:
                # Iterate over the response stream in chunks
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))  # Update the progress bar
                    file.write(data)  # Write the chunk to the file
            progress_bar.close()  # Close the progress bar once download is complete
        
        # Optional: Verify the size of the downloaded file
        if check_size:
            # Get the actual size of the downloaded file
            downloaded_size = os.path.getsize(filename)
            expected_size = total_size_in_bytes
            
            # Define a tolerance level (e.g., 1%) for size discrepancy
            tolerance = expected_size * 0.01
            # for consistency if  
            if downloaded_size >= expected_size: 
                expected_size = downloaded_size 
                
            # Check if the downloaded file size is within the acceptable range
            if not (expected_size - tolerance <= downloaded_size <= expected_size + tolerance):
                # Prepare an informative message about the size mismatch
                size_mismatch_msg = (
                    f"Downloaded file size for '{filename}' ({downloaded_size} bytes) "
                    f"does not match the expected size ({expected_size} bytes)."
                )
                
                # Handle the discrepancy based on the 'error' parameter
                if error == 'raise':
                    raise RuntimeError(size_mismatch_msg)
                elif error == 'warn':
                    warnings.warn(size_mismatch_msg)
                elif error == 'ignore':
                    pass  # Do nothing and continue
              
            elif verbose:
                print(f"File size for '{filename}' verified successfully.")
        
        # Move the file to the destination path if 'dstpath' is provided
        if dstpath:
            try:
                # Ensure the destination directory exists
                os.makedirs(dstpath, exist_ok=True)
                
                # Define the full destination path
                destination_file = os.path.join(dstpath, filename)
                
                # Move the downloaded file to the destination directory
                os.replace(filename, destination_file)
                
                if verbose:
                    print(f"File '{filename}' moved to '{destination_file}'.")
            except Exception as move_error:
                # Handle any errors that occur during the file move
                move_error_msg = (
                    f"Failed to move '{filename}' to '{dstpath}'. Error: {move_error}"
                )
                if error == 'raise':
                    raise RuntimeError(move_error_msg) from move_error
                elif error == 'warn':
                    warnings.warn(move_error_msg)
                elif error == 'ignore':
                    pass  # Do nothing and continue
          
            return None  # Return None since the file has been moved
        else:
            if verbose:
                print(f"File '{filename}' downloaded successfully.")
            return filename  # Return the filename if no destination path is provided
    
    except Exception as download_error:
        # Handle any exceptions that occur during the download process
        download_error_msg = (
            f"Failed to download '{filename}' from '{url}'. Error: {download_error}"
        )
        if error == 'raise':
            raise RuntimeError(download_error_msg) from download_error
        elif error == 'warn':
            warnings.warn(download_error_msg)
        elif error == 'ignore':
            pass  # Do nothing and continue
        
    return None  # Return None as a fallback

def fancier_downloader0(url, filename, dstpath =None ):
    """ Download remote file with a bar progression. 
    
    Parameters 
    -----------
    url: str, 
      Url to where the file is stored. 
    loadl_filename: str,
      Name of the local file 
      
    dstpath: Optional 
      The destination path to save the downloaded file. 
      
    Return 
    --------
    None, local_filename
       None if the `dstpath` is supplied and `local_filename` otherwise. 
    Example
    --------
    >>> from gofast.utils.base_utils import fancier_downloader
    >>> url = 'https://raw.githubusercontent.com/WEgeophysics/gofast/master/gofast/datasets/data/h.h5'
    >>> local_filename = 'h.h5'
    >>> download_file(url, local_filename)

    """
    import_optional_dependency("requests")
    import requests 
    try : 
        from tqdm import tqdm
    except: 
        # if tqm is not install  # this is simple downloading 
        return download_file (url, filename, dstpath  )
        
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        # Get the total file size from header
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', 
                            unit_scale=True, ncols=77, ascii=True)
        with open(filename, 'wb') as f:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        
    filename = os.path.join( os.getcwd(), filename) 
    
    if dstpath: 
         move_file ( filename,  dstpath)
         
    return filename


def move_file(file_path, directory):
    """ Move file to a directory. 
    
    Create a directory if not exists. 
    
    Parameters 
    -----------
    file_path: str, 
       Path to the local file 
    directory: str, 
       Path to locate the directory.
    
    Example 
    ---------
    >>> from gofast.utils.base_utils import move_file
    >>> file_path = 'path/to/your/file.txt'  # Replace with your file's path
    >>> directory = 'path/to/your/directory'  # Replace with your directory's path
    >>> move_file(file_path, directory)
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Move the file to the directory
    shutil.move(file_path, os.path.join(directory, os.path.basename(file_path)))

def check_file_exists(package, resource):
    """
    Check if a file exists in a package's directory with 
    importlib.resources.

    :param package: The package containing the resource.
    :param resource: The resource (file) to check.
    :return: Boolean indicating if the resource exists.
    
    :example: 
        >>> from gofast.utils.base_utils import check_file_exists
        >>> package_name = 'gofast.datasets.data'  # Replace with your package name
        >>> file_name = 'h.h5'    # Replace with your file name

        >>> file_exists = check_file_exists(package_name, file_name)
        >>> print(f"File exists: {file_exists}")
    """

    import importlib.resources as pkg_resources
    return pkg_resources.is_resource(package, resource)

def is_readable (
        f:str, 
        *, 
        as_frame:bool=False, 
        columns:List[str]=None,
        input_name='f', 
        **kws
 ) -> DataFrame: 
    """ Assert and read specific files and url allowed by the package
    
    Readable files are systematically convert to a pandas frame.  
    
    Parameters 
    -----------
    f: Path-like object -Should be a readable files or url  
    columns: str or list of str 
        Series name or columns names for pandas.Series and DataFrame. 
        
    to_frame: str, default=False
        If ``True`` , reconvert the array to frame using the columns orthewise 
        no-action is performed and return the same array.
    input_name : str, default=""
        The data name used to construct the error message. 
        
    raise_warning : bool, default=True
        If True then raise a warning if conversion is required.
        If ``ignore``, warnings silence mode is triggered.
    raise_exception : bool, default=False
        If True then raise an exception if array is not symmetric.
        
    force:bool, default=False
        Force conversion array to a frame is columns is not supplied.
        Use the combinaison, `input_name` and `X.shape[1]` range.
        
    kws: dict, 
        Pandas readableformats additional keywords arguments. 
    Returns
    ---------
    f: pandas dataframe 
         A dataframe with head contents... 
    
    """
    def _check_readable_file (f): 
        """ Return file name from path objects """
        msg =(f"Expects a Path-like object or URL. Please, check your"
              f" file: {os.path.basename(f)!r}")
        if not os.path.isfile (f): # force pandas read html etc 
            if not ('http://'  in f or 'https://' in f ):  
                raise TypeError (msg)
        elif not isinstance (f,  (str , pathlib.PurePath)): 
             raise TypeError (msg)
        if isinstance(f, str): f =f.strip() # for consistency 
        return f 
    
    if hasattr (f, '__array__' ) : 
        f = array_to_frame(
            f, 
            to_frame= True , 
            columns =columns, 
            input_name=input_name , 
            raise_exception= True, 
            force= True, 
            )
        return f 

    cpObj= PandasDataHandlers().parsers 
    
    f= _check_readable_file(f)
    _, ex = os.path.splitext(f) 
    if ex.lower() not in tuple (cpObj.keys()):
        raise TypeError(f"Can only parse the {smart_format(cpObj.keys(), 'or')} "
                        f" files not {ex!r}.")
    try : 
        f = cpObj[ex](f, **kws)
    except FileNotFoundError:
        raise FileNotFoundError (
            f"No such file in directory: {os.path.basename (f)!r}")
    except: 
        raise FileHandlingError (
            f" Can not parse the file : {os.path.basename (f)!r}")

    return f 

def array2hdf5 (
    filename: str, 
    arr: NDArray=None , 
    dataname: str='data',  
    task: str='store', 
    as_frame: bool =..., 
    columns: List[str]=None, 
)-> Union [NDArray , DataFrame]: 
    """ Load or write array to hdf5.
    
    Parameters 
    -----------
    arr: Arraylike ( m_samples, n_features) 
      Data to load or write 
    filename: str, 
      Hdf5 disk file name whether to write or to load 
    task: str, {"store", "load", "save", default='store'}
       Action to perform. user can use ['write'|'store'] interchnageably. Both 
       does the same task. 
    as_frame: bool, default=False 
       Concert loaded array to data frame. `Columns` can be supplied 
       to construct the datafame. 
    columns: List, Optional 
       Columns used to construct the dataframe. When its given, it must be 
       consistent with the shape of the `arr` along axis 1 
       
    Returns 
    ---------
    None| data: ArrayLike or pd.DataFrame 
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from gofast.utils.base_utils import array2hdf5
    >>> data = np.random.randn (100, 27 ) 
    >>> array2hdf5 ('test.h5', data   )
    >>> load_data = array2hdf5 ( 'test.h5', data, task ='load')
    >>> load_data.shape 
    Out[177]: (100, 27)
    """
    import_optional_dependency("h5py")
    import h5py 
    
    arr = is_iterable( arr, exclude_string =True, transform =True )
    act = copy.deepcopy(task)
    task = str(task).lower().strip() 
    
    if task in ("write", "store", "save"): 
        task ='store'
    assert task in {"store", "load"}, ("Expects ['store'|'load'] as task."
                                         f" Got {act!r}")
    # for consistency 
    arr = np.array ( arr )
    h5fname = str(filename).replace ('.h5', '')
    if task =='store': 
        if arr is None: 
            raise TypeError ("Array cannot be None when the task"
                             " consists to write a file.")
        with h5py.File(h5fname + '.h5', 'w') as hf:
            hf.create_dataset(dataname,  data=arr)
            
    elif task=='load': 
        with h5py.File(h5fname +".h5", 'r') as hf:
            data = hf[dataname][:]
            
        if  ellipsis2false( as_frame )[0]: 
            data = pd.DataFrame ( data , columns = columns )
            
    return data if task=='load' else None 

def remove_target_from_array(arr,  target_indices):
    """
    Remove specified columns from a 2D array based on target indices.

    This function extracts columns at specified indices from a 2D array, 
    returning the modified array without these columns and a separate array 
    containing the extracted columns. It raises an error if any of the indices
    are out of bounds.

    Parameters
    ----------
    arr : ndarray
        A 2D numpy array from which columns are to be removed.
    target_indices : list or ndarray
        Indices of the columns in `arr` that need to be extracted and removed.

    Returns
    -------
    modified_arr : ndarray
        The array obtained after removing the specified columns.
    target_arr : ndarray
        An array consisting of the columns extracted from `arr`.

    Raises
    ------
    ValueError
        If any of the target indices are out of the range of the array dimensions.

    Examples
    --------
    >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> target_indices = [1, 2]
    >>> modified_arr, target_arr = remove_target_from_array(arr, target_indices)
    >>> modified_arr
    array([[1],
           [4],
           [7]])
    >>> target_arr
    array([[2, 3],
           [5, 6],
           [7, 8]])
    """
    if any(idx >= arr.shape[1] for idx in target_indices):
        raise ValueError("One or more indices are out of the array's bounds.")

    target_arr = arr[:, target_indices]
    modified_arr = np.delete(arr, target_indices, axis=1)
    return modified_arr, target_arr

def extract_target(
    data: Union[ArrayLike, DataFrame], 
    target_names: Union[str, int, List[Union[str, int]]],
    drop: bool = True,
    columns: Optional[List[str]] = None,
    return_y_X: bool = False
) -> Union[ArrayLike, Series, DataFrame, Tuple[ArrayLike, pd.DataFrame]]:
    """
    Extracts specified target column(s) from a multidimensional numpy array
    or pandas DataFrame. 
    
    with options to rename columns in a DataFrame and control over whether the 
    extracted columns are dropped from the original data.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame]
        The input data from which target columns are to be extracted. Can be a 
        NumPy array or a pandas DataFrame.
    target_names : Union[str, int, List[Union[str, int]]]
        The name(s) or integer index/indices of the column(s) to extract. 
        If `data` is a DataFrame, this can be a mix of column names and indices. 
        If `data` is a NumPy array, only integer indices are allowed.
    drop : bool, default True
        If True, the extracted columns are removed from the original `data`. 
        If False, the original `data` remains unchanged.
    columns : Optional[List[str]], default None
        If provided and `data` is a DataFrame, specifies new names for the 
        columns in `data`. The length of `columns` must match the number of 
        columns in `data`. This parameter is ignored if `data` is a NumPy array.
    return_y_X : bool, default False
        If True, returns a tuple (y, X) where X is the data with the target columns
        removed and y is the target columns. If False, returns only y.

    Returns
    -------
    Union[ArrayLike, pd.Series, pd.DataFrame, Tuple[ pd.DataFrame, ArrayLike]]
        If return_X_y is True, returns a tuple (X, y) where X is the data with the 
        target columns removed and y is the target columns. If return_X_y is False, 
        returns only y.

    Raises
    ------
    ValueError
        If `columns` is provided and its length does not match the number of 
        columns in `data`.
        If any of the specified `target_names` do not exist in `data`.
        If `target_names` includes a mix of strings and integers for a NumPy 
        array input.

    Examples
    --------
    >>> import pandas as pd 
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> target = extract_target(df, 'B', drop=True, return_y_X=False)
    >>> print(target)
    0    4
    1    5
    2    6
    Name: B, dtype: int64
    >>> target, remaining = extract_target(df, 'B', drop=True, return_y_X=True)
    >>> print(target)
    0    4
    1    5
    2    6
    Name: B, dtype: int64
    >>> print(remaining)
       A  C
    0  1  7
    1  2  8
    2  3  9
    >>> arr = np.random.rand(5, 3)
    >>> target, modified_arr = extract_target(arr, 2, return_X_y=True)
    >>> print(target)
    >>> print(modified_arr)
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if np.ndim(data) == 1:
        data = np.expand_dims(data, axis=1)

    is_frame = isinstance(data, pd.DataFrame)

    if is_frame and columns is not None:
        columns = is_iterable(columns, exclude_string= True, transform= True)
        if len(columns) != data.shape[1]:
            raise ValueError(
                "`columns` must match the number of columns in"" `data`."
                f" Expected {data.shape[1]}, got {len(columns)}.")
        data.columns = columns

    if isinstance(target_names, (int, str)):
        target_names = [target_names]

    if all(isinstance(name, int) for name in target_names):
        if max(target_names, default=-1) >= data.shape[1]:
            raise ValueError(
                "All integer indices must be within the column range of the data.")
    elif any(isinstance(name, int) for name in target_names) and is_frame:
        target_names = [data.columns[name] if isinstance(name, int) 
                        else name for name in target_names]

    if is_frame:
        missing_cols = [name for name in target_names if name not in data.columns]
        if missing_cols:
            raise ValueError(f"Column names {missing_cols} do not match any"
                             " column in the DataFrame.")
        target = data.loc[:, target_names]
        if drop:
            data = data.drop(columns=target_names)
    else:
        if any(isinstance(name, str) for name in target_names):
            raise ValueError("String names are not allowed for target names"
                             " when data is a NumPy array.")
        target = data[:, target_names]
        if drop:
            data = np.delete(data, target_names, axis=1)

    if isinstance(target, np.ndarray):
        target = np.squeeze(target)

    target = to_series (target, handle_2d="passthrough")
    if return_y_X:
        return target, data
    return target

def _extract_target(
        X, target: Union[ArrayLike, int, str, List[Union[int, str]]]):
    """
    Extracts and validates the target variable(s) from the dataset.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The dataset from which to extract the target variable(s).
    target : ArrayLike, int, str, or list of int/str
        The target variable(s) to be used. If an array-like or DataFrame, 
        it's directly used as `y`. If an int or str (or list of them), it 
        indicates the column(s) in `X` to be used as `y`.

    Returns
    -------
    X : pd.DataFrame or np.ndarray
        The dataset without the target column(s).
    y : pd.Series, np.ndarray, pd.DataFrame
        The target variable(s).
    target_names : list of str
        The names of the target variable(s) for labeling purposes.
    """
    target_names = []

    if isinstance(target, (list, pd.DataFrame)) or (
            isinstance(target, pd.Series) and not isinstance(X, np.ndarray)):
        if isinstance(target, list):  # List of column names or indexes
            if all(isinstance(t, str) for t in target):
                y = X[target]
                target_names = target
            elif all(isinstance(t, int) for t in target):
                y = X.iloc[:, target]
                target_names = [X.columns[i] for i in target]
            X = X.drop(columns=target_names)
        elif isinstance(target, pd.DataFrame):
            y = target
            target_names = target.columns.tolist()
            # Assuming target DataFrame is not part of X
        elif isinstance(target, pd.Series):
            y = target
            target_names = [target.name] if target.name else ["target"]
            if target.name and target.name in X.columns:
                X = X.drop(columns=target.name)
                
    elif isinstance(target, (int, str)):
        if isinstance(target, str):
            y = X.pop(target)
            target_names = [target]
        elif isinstance(target, int):
            y = X.iloc[:, target]
            target_names = [X.columns[target]]
            X = X.drop(columns=X.columns[target])
    elif isinstance(target, np.ndarray) or (
            isinstance(target, pd.Series) and isinstance(X, np.ndarray)):
        y = np.array(target)
        target_names = ["target"]
    else:
        raise ValueError("Unsupported target type or target does not match X dimensions.")
    
    check_consistent_length(X, y)
    
    return X, y, target_names

def categorize_target(
    arr : Union [ArrayLike , Series] ,  
    func: _F = None,  
    labels: Union [int, List[int]] = None, 
    rename_labels: Optional[str] = None, 
    coerce:bool=False,
    order:str='strict',
    ): 
    """ Categorize array to hold the given identifier labels. 
    
    Classifier numerical values according to the given label values. Labels 
    are a list of integers where each integer is a group of unique identifier  
    of a sample in the dataset. 
    
    Parameters 
    -----------
    arr: array-like |pandas.Series 
        array or series containing numerical values. If a non-numerical values 
        is given , an errors will raises. 
    func: Callable, 
        Function to categorize the target y.  
    labels: int, list of int, 
        if an integer value is given, it should be considered as the number 
        of category to split 'y'. For instance ``label=3`` applied on 
        the first ten number, the labels values should be ``[0, 1, 2]``. 
        If labels are given as a list, items must be self-contain in the 
        target 'y'.
    rename_labels: list of str; 
        list of string or values to replace the label integer identifier. 
    coerce: bool, default =False, 
        force the new label names passed to `rename_labels` to appear in the 
        target including or not some integer identifier class label. If 
        `coerce` is ``True``, the target array holds the dtype of new_array. 

    Return
    --------
    arr: Arraylike |pandas.Series
        The category array with unique identifer labels 
        
    Examples 
    --------

    >>> from gofast.utils.base_utils import categorize_target 
    >>> def binfunc(v): 
            if v < 3 : return 0 
            else : return 1 
    >>> arr = np.arange (10 )
    >>> arr 
    ... array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> target = categorize_target(arr, func =binfunc)
    ... array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    >>> categorize_target(arr, labels =3 )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    >>> array([2, 2, 2, 2, 1, 1, 1, 0, 0, 0]) 
    >>> categorize_target(arr, labels =3 , order =None )
    ... array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> categorize_target(arr[::-1], labels =3 , order =None )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) # reverse does not change
    >>> categorize_target(arr, labels =[0 , 2,  4]  )
    ... array([0, 0, 0, 2, 2, 4, 4, 4, 4, 4])

    """
    arr = _assert_all_types(arr, np.ndarray, pd.Series) 
    is_arr =False 
    if isinstance (arr, np.ndarray ) :
        arr = pd.Series (arr  , name = 'none') 
        is_arr =True 
        
    if func is not None: 
        if not  inspect.isfunction (func): 
            raise TypeError (
                f'Expect a function but got {type(func).__name__!r}')
            
        arr= arr.apply (func )
        
        return  arr.values  if is_arr else arr   
    
    name = arr.name 
    arr = arr.values 

    if labels is not None: 
        arr = _cattarget (arr , labels, order =order)
        if rename_labels is not None: 
            arr = rename_labels_in( arr , rename_labels , coerce =coerce ) 

    return arr  if is_arr else pd.Series (arr, name =name  )

def rename_labels_in (
        arr, new_names, coerce = False): 
    """ Rename label by a new names 
    
    :param arr: arr: array-like |pandas.Series 
         array or series containing numerical values. If a non-numerical values 
         is given , an errors will raises. 
    :param new_names: list of str; 
        list of string or values to replace the label integer identifier. 
    :param coerce: bool, default =False, 
        force the 'new_names' to appear in the target including or not some 
        integer identifier class label. `coerce` is ``True``, the target array 
        hold the dtype of new_array; coercing the label names will not yield 
        error. Consequently can introduce an unexpected results.
    :return: array-like, 
        An array-like with full new label names. 
    """
    
    if not is_iterable(new_names): 
        new_names= [new_names]
    true_labels = np.unique (arr) 
    
    if labels_validator(arr, new_names, return_bool= True): 
        return arr 

    if len(true_labels) != len(new_names):
        if not coerce: 
            raise ValueError(
                "Can't rename labels; the new names and unique label" 
                " identifiers size must be consistent; expect {}, got " 
                "{} label(s).".format(len(true_labels), len(new_names))
                             )
        if len(true_labels) < len(new_names) : 
            new_names = new_names [: len(new_names)]
        else: 
            new_names = list(new_names)  + list(
                true_labels)[len(new_names):]
            warnings.warn("Number of the given labels '{}' and values '{}'"
                          " are not consistent. Be aware that this could "
                          "yield an expected results.".format(
                              len(new_names), len(true_labels)))
            
    new_names = np.array(new_names)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # hold the type of arr to operate the 
    # element wise comparaison if not a 
    # ValueError:' invalid literal for int() with base 10' 
    # will appear. 
    if not np.issubdtype(np.array(new_names).dtype, np.number): 
        arr= arr.astype (np.array(new_names).dtype)
        true_labels = true_labels.astype (np.array(new_names).dtype)

    for el , nel in zip (true_labels, new_names ): 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # element comparison throws a future warning here 
        # because of a disagreement between Numpy and native python 
        # Numpy version ='1.22.4' while python version = 3.9.12
        # this code is brittle and requires these versions above. 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # suppress element wise comparison warning locally 
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            arr [arr == el ] = nel 
            
    return arr 

    
def _cattarget (ar , labels , order=None): 
    """ A shadow function of :func:`gofast.utils.base_utils.cattarget`. 
    
    :param ar: array-like of numerical values 
    :param labels: int or list of int, 
        the number of category to split 'ar'into. 
    :param order: str, optional, 
        the order of label to be categorized. If None or any other values, 
        the categorization of labels considers only the length of array. 
        For instance a reverse array and non-reverse array yield the same 
        categorization samples. When order is set to ``strict``, the 
        categorization  strictly considers the value of each element. 
        
    :return: array-like of int , array of categorized values.  
    """
    # assert labels
    if is_iterable (labels):
        labels =[int (_assert_all_types(lab, int, float)) 
                 for lab in labels ]
        labels = np.array (labels , dtype = np.int32 ) 
        cc = labels 
        # assert whether element is on the array 
        s = set (ar).intersection(labels) 
        if len(s) != len(labels): 
            mv = set(labels).difference (s) 
            
            fmt = [f"{'s' if len(mv) >1 else''} ", mv,
                   f"{'is' if len(mv) <=1 else'are'}"]
            warnings.warn("Label values must be array self-contain item. "
                           "Label{0} {1} {2} missing in the array.".format(
                               *fmt)
                          )
            raise ValueError (
                "label value{0} {1} {2} missing in the array.".format(*fmt))
    else : 
        labels = int (_assert_all_types(labels , int, float))
        labels = np.linspace ( min(ar), max (ar), labels + 1 ) #+ .00000001 
        #array([ 0.,  6., 12., 18.])
        # split arr and get the range of with max bound 
        cc = np.arange (len(labels)) #[0, 1, 3]
        # we expect three classes [ 0, 1, 3 ] while maximum 
        # value is 18 . we want the value value to be >= 12 which 
        # include 18 , so remove the 18 in the list 
        labels = labels [:-1] # remove the last items a
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]) # 3 classes 
        #  array([ 0.        ,  3.33333333,  6.66666667, 10. ]) + 
    # to avoid the index bound error 
    # append nan value to lengthen arr 
    r = np.append (labels , np.nan ) 
    new_arr = np.zeros_like(ar) 
    # print(labels)
    ar = ar.astype (np.float32)

    if order =='strict': 
        for i in range (len(r)):
            if i == len(r) -2 : 
                ix = np.argwhere ( (ar >= r[i]) & (ar != np.inf ))
                new_arr[ix ]= cc[i]
                break 
            
            if i ==0 : 
                ix = np.argwhere (ar < r[i +1])
                new_arr [ix] == cc[i] 
                ar [ix ] = np.inf # replace by a big number than it was 
                # rather than delete it 
            else :
                ix = np.argwhere( (r[i] <= ar) & (ar < r[i +1]) )
                new_arr [ix ]= cc[i] 
                ar [ix ] = np.inf 
    else: 
        l= list() 
        for i in range (len(r)): 
            if i == len(r) -2 : 
                l.append (np.repeat ( cc[i], len(ar))) 
                
                break
            ix = np.argwhere ( (ar < r [ i + 1 ] ))
            l.append (np.repeat (cc[i], len (ar[ix ])))  
            # remove the value ready for i label 
            # categorization 
            ar = np.delete (ar, ix  )
            
        new_arr= np.hstack (l).astype (np.int32)  
        
    return new_arr.astype (np.int32)    
   

def labels_validator(
    target: ArrayLike, 
    labels: Union[int, str, List[Union[int, str]]], 
    return_bool: bool = False
    ) -> Union[bool, List[Union[int, str]]]:
    """
    Validates if specified labels are present in the target array and 
    optionally returns a boolean indicating the presence of all labels or 
    the list of labels themselves.
    
    Parameters
    ----------
    target : np.ndarray
        The target array expected to contain the labels.
    labels : int, str, or list of int or str
        The label(s) supposed to be in the target array.
    return_bool : bool, default=False
        If True, returns a boolean indicating whether all specified 
        labels are present. If False, returns the list of labels.

    Returns
    -------
    bool or List[Union[int, str]]
        If `return_bool` is True, returns True if all labels are present, 
        False otherwise.
        If `return_bool` is False, returns the list of labels if all are present.
    
    Raises
    ------
    ValueError
        If any of the specified labels are missing in the target array and 
        `return_bool` is False.
    
    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.utils.base_utils import labels_validator
    >>> target = np.array([1, 2, 3, 4, 5])
    >>> labels_validator(target, [1, 2, 3])
    [1, 2, 3]
    >>> labels_validator(target, [0, 1], return_bool=True)
    False
    >>> labels_validator(target, 1)
    [1]
    >>> labels_validator(target, [6], return_bool=True)
    False
    """
    if isinstance(labels, (int, str)):
        labels = [labels]
    
    labels_present = np.unique([label for label in labels if label in target])
    missing_labels = [label for label in labels if label not in labels_present]

    if missing_labels:
        if return_bool:
            return False
        raise ValueError(f"Label{'s' if len(missing_labels) > 1 else ''}"
                        f" {', '.join(map(str, missing_labels))}"
                        f" {'are' if len(missing_labels) > 1 else 'is'}"
                        " missing in the target."
                    )

    return True if return_bool else labels

def generate_placeholders(
        iterable_obj: Iterable[_T]) -> List[str]:
    """
    Generates a list of string placeholders for each item in the input
    iterable. This can be useful for creating formatted string
    representations where each item's index is used within braces.

    :param iterable_obj: An iterable object (e.g., list, set, or any
        iterable collection) whose length determines the number of
        placeholders generated.
    :return: A list of strings, each representing a placeholder in
        the format "{n}", where n is the index of the placeholder.
        
    :Example:
        >>> from gofast.utils.base_utils import generate_placeholders
        >>> generate_placeholders_for_iterable({'ohmS', 'lwi', 'power', 'id', 
        ...                                     'sfi', 'magnitude'})
        ['{0}', '{1}', '{2}', '{3}', '{4}', '{5}']
    """
    return [f"{{{index}}}" for index in range(len(iterable_obj))]


def compute_set_operation( 
    iterable1: Iterable[Any],
    iterable2: Iterable[Any],
    operation: str = "intersection"
) -> Set[Any]:
    """
    Computes the intersection or difference between two iterable objects,
    returning the result as a set. This function is flexible and works
    with any iterable types, including lists, sets, and dictionaries.

    Parameters
    ----------
    iterable1 : Iterable[Any]
        The first iterable object from which to compute the operation.
    iterable2 : Iterable[Any]
        The second iterable object.
    operation : str, optional
        The operation to perform, either 'intersection' or 'difference'.
        Defaults to 'intersection'.

    Returns
    -------
    Set[Any]
        A set of either common elements (intersection) or unique elements
        (difference) from the two iterables.

    Examples
    --------
    Intersection example:
    >>> compute_set_operation(
    ...     ['a', 'b', 'c'], 
    ...     {'b', 'c', 'd'}
    ... )
    {'b', 'c'}

    Difference example:
    >>> compute_set_operation(
    ...     ['a', 'b', 'c'], 
    ...     {'b', 'c', 'd'},
    ...     operation='difference'
    ... )
    {'a', 'd'}

    Notes
    -----
    The function supports only 'intersection' and 'difference' operations.
    It ensures the result is always returned as a set, regardless of the
    input iterable types.
    """
    
    set1 = set(iterable1)
    set2 = set(iterable2)

    if operation == "intersection":
        return set1 & set2  # Using & for intersection
    elif operation == "difference":
        # Returning symmetric difference
        return set1 ^ set2  # Using ^ for symmetric difference
    else:
        raise ValueError("Invalid operation specified. Choose either"
                         " 'intersection' or 'difference'.")

def find_intersection(
    iterable1: Iterable[Any],
    iterable2: Iterable[Any]
) -> Set[Any]:
    """
    Computes the intersection of two iterable objects, returning a set
    of elements common to both. This function is designed to work with
    various iterable types, including lists, sets, and dictionaries.

    Parameters
    ----------
    iterable1 : Iterable[Any]
        The first iterable object.
    iterable2 : Iterable[Any]
        The second iterable object.

    Returns
    -------
    Set[Any]
        A set of elements common to both `iterable1` and `iterable2`.

    Example
    -------
    >>> from gofast.utils.base_utils import find_intersection_between_generics
    >>> compute_intersection(
    ...     ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
    ...     {'ohmS', 'lwi', 'power'}
    ... )
    {'ohmS', 'lwi', 'power'}

    Notes
    -----
    The result is always a set, regardless of the input types, ensuring
    that each element is unique and present in both iterables.
    """

    # Utilize set intersection operation (&) for clarity and conciseness
    return set(iterable1) & set(iterable2)

def find_unique_elements(
    iterable1: Iterable[Any],
    iterable2: Iterable[Any]
) -> Optional[Set[Any]]:
    """
    Computes the difference between two iterable objects, returning a set
    containing elements unique to the iterable with more unique elements.
    If both iterables contain an equal number of unique elements, the function
    returns None.

    This function is designed to work with various iterable types, including
    lists, sets, and dictionaries. The focus is on the count of unique elements
    rather than the total length, which allows for more consistent results
    across different types of iterables.

    Parameters
    ----------
    iterable1 : Iterable[Any]
        The first iterable object.
    iterable2 : Iterable[Any]
        The second iterable object.

    Returns
    -------
    Optional[Set[Any]]
        A set of elements unique to the iterable with more unique elements,
        or None if both have an equal number of unique elements.

    Example
    -------
    >>> find_unique_elements(
    ...     ['a', 'b', 'c', 'c'],
    ...     {'a', 'b'}
    ... )
    {'c'}

    Notes
    -----
    The comparison is based on the number of unique elements, not the
    iterable size. This approach ensures a more meaningful comparison
    when the iterables are of different types or when duplicates are present.
    """

    set1 = set(iterable1)
    set2 = set(iterable2)

    # Adjust the logic to focus on the uniqueness rather than size
    if len(set1) == len(set2):
        return None
    elif len(set1) > len(set2):
        return set1 - set2
    else:
        return set2 - set1

def validate_feature_existence(supervised_features: Iterable[_T], 
                               features: Iterable[_T]) -> None:
    """
    Validates the existence of supervised features within a list of all features.
    This is typically used to ensure that certain expected features (columns) are
    present in a pandas DataFrame.

    Parameters
    ----------
    supervised_features : Iterable[_T]
        An iterable of features presumed to be controlled or supervised.
        
    features : Iterable[_T]
        An iterable of all features, such as pd.DataFrame.columns.

    Raises
    ------
    ValueError
        If `supervised_features` are not found within `features`.
    """
    # Ensure input is in list format if strings are passed
    if isinstance(supervised_features, str):
        supervised_features = [supervised_features]
    if isinstance(features, str):
        features = [features]
    
    # Check for feature existence
    if not cfexist(features_to=supervised_features, features=list(features)):
        raise ValueError(f"Features {supervised_features} not found in {list(features)}")

def cfexist(features_to: List[Any], features: List[Any]) -> bool:
    """
    Checks if all elements of one list (features_to) exist within another list (features).

    Parameters
    ----------
    features_to : List[Any]
        List or array to be checked for existence within `features`.
        
    features : List[Any]
        List of whole features, e.g., as in pd.DataFrame.columns.

    Returns
    -------
    bool
        True if all elements in `features_to` exist in `features`, False otherwise.
    """
    # Normalize input to lists, handle string inputs
    if isinstance(features_to, str):
        features_to = [features_to]
    if isinstance(features, str):
        features = [features]

    # Check for existence
    return set(features_to).issubset(features)

def control_existing_estimator(
    estimator_name: str, 
    predefined_estimators=None, 
    raise_error: bool = False
) -> Union[Tuple[str, str], None]:
    """
    Validates and retrieves the corresponding prefix for a given estimator name.

    This function checks if the provided estimator name exists in a predefined
    list of estimators or in scikit-learn. If found, it returns the corresponding
    prefix and full name. Otherwise, it either raises an error or returns None,
    based on the 'raise_error' flag.

    Parameters
    ----------
    estimator_name : str
        The name of the estimator to check.
    predefined_estimators : dict, default _predefined_estimators
        A dictionary of predefined estimators.
    raise_error : bool, default False
        If True, raises an error when the estimator is not found. Otherwise, 
        emits a warning.

    Returns
    -------
    Tuple[str, str] or None
        A tuple containing the prefix and full name of the estimator, or 
        None if not found.

    Example
    -------
    >>> from gofast.utils.base_utils import control_existing_estimator
    >>> test_est = control_existing_estimator('svm')
    >>> print(test_est)
    ('svc', 'SupportVectorClassifier')
    """
    
    from ..exceptions import EstimatorError 
    # Define a dictionary of predefined estimators
    _predefined_estimators ={
            'dtc': ['DecisionTreeClassifier', 'dtc', 'dec', 'dt'],
            'svc': ['SupportVectorClassifier', 'svc', 'sup', 'svm'],
            'sdg': ['SGDClassifier','sdg', 'sd', 'sdg'],
            'knn': ['KNeighborsClassifier','knn', 'kne', 'knr'],
            'rdf': ['RandomForestClassifier', 'rdf', 'rf', 'rfc',],
            'ada': ['AdaBoostClassifier','ada', 'adc', 'adboost'],
            'vtc': ['VotingClassifier','vtc', 'vot', 'voting'],
            'bag': ['BaggingClassifier', 'bag', 'bag', 'bagg'],
            'stc': ['StackingClassifier','stc', 'sta', 'stack'],
            'xgb': ['ExtremeGradientBoosting', 'xgboost', 'gboost', 'gbdm', 'xgb'], 
          'logit': ['LogisticRegression', 'logit', 'lr', 'logreg'], 
          'extree': ['ExtraTreesClassifier', 'extree', 'xtree', 'xtr']
            }
    predefined_estimators = predefined_estimators or _predefined_estimators
    
    estimator_name= estimator_name.lower().strip() if isinstance (
        estimator_name, str) else get_estimator_name(estimator_name)
    
    # Check if the estimator is in the predefined list
    for prefix, names in predefined_estimators.items():
        lower_names = [name.lower() for name in names]
        
        if estimator_name in lower_names:
            return prefix, names[0]

    # If not found in predefined list, check if it's a valid scikit-learn estimator
    if estimator_name in _get_sklearn_estimator_names():
        return estimator_name, estimator_name

    # If XGBoost is installed, check if it's an XGBoost estimator
    if 'xgb' in predefined_estimators and estimator_name.startswith('xgb'):
        return 'xgb', estimator_name

    # If raise_error is True, raise an error; otherwise, emit a warning
    if raise_error:
        valid_names = [name for names in predefined_estimators.values() for name in names]
        raise EstimatorError(f'Unsupported estimator {estimator_name!r}. '
                             f'Expected one of {valid_names}.')
    else:
        available_estimators = _get_available_estimators(predefined_estimators)
        warning_msg = (f"Estimator {estimator_name!r} not found. "
                       f"Expected one of: {available_estimators}.")
        warnings.warn(warning_msg)

    return None

def _get_sklearn_estimator_names():
    
    # Retrieve all scikit-learn estimator names using all_estimators
    sklearn_estimators = [name for name, _ in all_estimators(type_filter='classifier')]
    sklearn_estimators += [name for name, _ in all_estimators(type_filter='regressor')]
    return sklearn_estimators

def _get_available_estimators(predefined_estimators):
    # Combine scikit-learn and predefined estimators
    sklearn_estimators = _get_sklearn_estimator_names()
    xgboost_estimators = ['xgb' + name for name in predefined_estimators['xgb']]
    
    available_estimators = sklearn_estimators + xgboost_estimators
    return available_estimators

def get_target(df, tname, inplace=True):
    """
    Extracts one or more target columns from a DataFrame and optionally
    modifies the original DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to extract the target(s).
    tname : str or list of str
        The name(s) of the target column(s) to extract. These must be present
        in the DataFrame.
    inplace : bool, optional
        If True, the DataFrame is modified in place by removing the target
        column(s). Defaults to True.

    Returns
    -------
    tuple
        A tuple containing:
        - pd.Series or pd.DataFrame: The extracted target column(s).
        - pd.DataFrame: The modified or unmodified DataFrame depending on the
          `inplace` parameter.

    Raises
    ------
    ValueError
        If any of the specified target names are not in the DataFrame columns.
    TypeError
        If `df` is not a pandas DataFrame.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gofast.base_utils import get_target 
    >>> data = load_iris(as_frame=True).frame
    >>> targets, modified_df = get_target(data, 'target', inplace=False)
    >>> print(targets.head())
    >>> print(modified_df.columns)

    Notes
    -----
    This function is particularly useful when preparing data for machine
    learning models, where separating features from labels is a common task.

    See Also
    --------
    extract_target : Similar function with enhanced capabilities for handling
                     more complex scenarios.
    
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if isinstance(tname, str):
        tname = [tname]  # Convert string to list for uniform processing

    missing_columns = [name for name in tname if name not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Target name(s) not found in DataFrame columns: {missing_columns}")

    target_data = df[tname]
    if inplace:
        df.drop(tname, axis=1, inplace=True)

    return target_data, df

@Dataify(auto_columns=True )
def binning_statistic(
    data, categorical_column, 
    value_column, 
    statistic='mean'
    ):
    """
    Compute a statistic for each category in a categorical column of a dataset.

    This function categorizes the data into bins based on a categorical variable and then
    applies a statistical function to the values of another column for each category.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    categorical_column : str
        Name of the column in `data` which contains the categorical variable.
    value_column : str
        Name of the column in `data` from which the statistic will be calculated.
    statistic : str, optional
        The statistic to compute (default is 'mean'). Other options include 
        'sum', 'count','median', 'min', 'max', etc.

    Returns
    -------
    result : DataFrame
        A DataFrame with each category and the corresponding computed statistic.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.utils.base_utils import binning_statistic
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    ...     'Value': [1, 2, 3, 4, 5, 6, 7]
    ... })
    >>> binning_statistic(df, 'Category', 'Value', statistic='mean')
       Category  Mean_Value
    0        A         3.33
    1        B         3.50
    2        C         5.50
    """
    if statistic not in ('mean', 'sum', 'count', 'median', 'min',
                         'max', 'proportion'):
        raise ValueError(
            "Unsupported statistic. Please choose from 'mean',"
            " 'sum', 'count', 'median', 'min', 'max', 'proportion'.")

    is_categorical(data, categorical_column)
    exist_features(data, features =value_column, name ="value_column")
    grouped_data = data.groupby(categorical_column)[value_column]
    
    if statistic == 'mean':
        result = grouped_data.mean().reset_index(name=f'Mean_{value_column}')
    elif statistic == 'sum':
        result = grouped_data.sum().reset_index(name=f'Sum_{value_column}')
    elif statistic == 'count':
        result = grouped_data.count().reset_index(name=f'Count_{value_column}')
    elif statistic == 'median':
        result = grouped_data.median().reset_index(name=f'Median_{value_column}')
    elif statistic == 'min':
        result = grouped_data.min().reset_index(name=f'Min_{value_column}')
    elif statistic == 'max':
        result = grouped_data.max().reset_index(name=f'Max_{value_column}')
    elif statistic == 'proportion':
        total_count = data[value_column].count()
        proportion = grouped_data.sum() / total_count
        result = proportion.reset_index(name=f'Proportion_{value_column}')
        
    return result

@Dataify(auto_columns=True)
def category_count(data,  *categorical_columns, error='raise'):
    """
    Count occurrences of each category in one or more categorical columns 
    of a dataset.
    
    This function computes the frequency of each unique category in the specified
    categorical columns of a pandas DataFrame and handles different ways of error
    reporting including raising an error, warning, or ignoring the error when a
    specified column is not found.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    *categorical_columns : str
        One or multiple names of the columns in `data` which contain the 
        categorical variables.
    error : str, optional
        Error handling strategy - 'raise' (default), 'warn', or 'ignore' which
        dictates the action when a categorical column is not found.

    Returns
    -------
    counts : DataFrame
        A DataFrame with each category and the corresponding count from each
        categorical column. If multiple columns are provided, columns are named as
        'Category_i' and 'Count_i'.

    Raises
    ------
    ValueError
        If any categorical column is not found in the DataFrame and error is 'raise'.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.utils.base_utils import category_count
    >>> df = pd.DataFrame({
    ...     'Fruit': ['Apple', 'Banana', 'Apple', 'Cherry', 'Banana', 'Apple'],
    ...     'Color': ['Red', 'Yellow', 'Green', 'Red', 'Yellow', 'Green']
    ... })
    >>> category_count(df, 'Fruit', 'Color')
       Category_1  Count_1 Category_2  Count_2
    0      Apple        3        Red        2
    1     Banana        2     Yellow        2
    2     Cherry        1      Green        2
    >>> category_count(df, 'NonExistentColumn', error='warn')
    Warning: Column 'NonExistentColumn' not found in the dataframe.
    Empty DataFrame
    Columns: []
    Index: []
    """
    results = []
    for i, column in enumerate(categorical_columns, 1):
        if column not in data.columns:
            message = f"Column '{column}' not found in the dataframe."
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
                continue
            elif error == 'ignore':
                continue

        count = data[column].value_counts().reset_index()
        count.columns = [f'Category_{i}', f'Count_{i}']
        results.append(count)

    if not results:
        return pd.DataFrame()

    # Merge all results into a single DataFrame
    final_df = functools.reduce(lambda left, right: pd.merge(
        left, right, left_index=True, right_index=True, how='outer'), results)
    final_df.fillna(value=np.nan, inplace=True)
    
    if len( results)==1: 
        final_df.columns =['Category', 'Count']
    return final_df

@Dataify(auto_columns=True) 
def soft_bin_stat(
    data,  categorical_column, 
    target_column, 
    statistic='mean', 
    update=False, 
    ):
    """
    Compute a statistic for each category in a categorical 
    column based on a binary target.

    This function calculates statistics like mean, sum, or proportion 
    for a binary target variable, grouped by categories in a 
    specified column.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    categorical_column : str
        Name of the column in `data` which contains the categorical variable.
    target_column : str
        Name of the column in `data` which contains the binary target variable.
    statistic : str, optional
        The statistic to compute for the binary target (default is 'mean').
        Other options include 'sum' and 'proportion'.

    Returns
    -------
    result : DataFrame
        A DataFrame with each category and the corresponding 
        computed statistic.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.utils.base_utils import soft_bin_stat
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    ...     'Target': [1, 0, 1, 0, 1, 0, 1]
    ... })
    >>> soft_bin_stat(df, 'Category', 'Target', statistic='mean')
       Category  Mean_Target
    0        A     0.666667
    1        B     0.500000
    2        C     0.500000

    >>> soft_bin_stat(df.values, 'col_0', 'col_1', statistic='mean')
      col_0  Mean_col_1
    0     A    0.666667
    1     B    0.500000
    2     C    0.500000
    """
    if statistic not in ['mean', 'sum', 'proportion']:
        raise ValueError("Unsupported statistic. Please choose from "
                         "'mean', 'sum', 'proportion'.")
    
    is_categorical(data, categorical_column)
    exist_features(data, features= target_column, name ='Target')
    grouped_data = data.groupby(categorical_column)[target_column]
    
    if statistic == 'mean':
        result = grouped_data.mean().reset_index(name=f'Mean_{target_column}')
    elif statistic == 'sum':
        result = grouped_data.sum().reset_index(name=f'Sum_{target_column}')
    elif statistic == 'proportion':
        total_count = data[target_column].count()
        proportion = grouped_data.sum() / total_count
        result = proportion.reset_index(name=f'Proportion_{target_column}')

    return result

def reshape_to_dataframe(flattened_array, columns, error ='raise'):
    """
    Reshapes a flattened array into a pandas DataFrame or Series based on the
    provided column names. If the number of columns does not allow reshaping
    to match the array length, it raises an error.

    Parameters
    ----------
    flattened_array : array-like
        The flattened array to reshape.
    columns : list of str
        The list of column names for the DataFrame. If a single name is provided,
        a Series is returned.
        
    error : {'raise', 'warn', 'ignore'}, default 'raise'
        Specifies how to handle the situation when the number of elements in the
        flattened array is not compatible with the number of columns required for
        reshaping. Options are:
        
        - 'raise': Raises a ValueError. This is the default behavior.
        - 'warn': Emits a warning, but still returns the original flattened array.
        - 'ignore': Does nothing about the error, just returns the original
          flattened array.
        
    Returns
    -------
    pandas.DataFrame or pandas.Series
        A DataFrame or Series reshaped according to the specified columns.

    Raises
    ------
    ValueError
        If the total number of elements in the flattened array does not match
        the required number for a complete reshaping.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.utils.base_utils import reshape_to_dataframe
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> print(reshape_to_dataframe(data, ['A', 'B', 'C']))  # DataFrame with 2 rows and 3 columns
    >>> print(reshape_to_dataframe(data, 'A'))  # Series with 6 elements
    >>> print(reshape_to_dataframe(data, ['A']))  # DataFrame with 6 rows and 1 column
    """
    # Check if the reshaping is possible
    is_string = isinstance ( columns, str )
    # Convert single string column name to list
    if isinstance(columns, str):
        columns = [columns]
        
    num_elements = len(flattened_array)
    num_columns = len(columns)
    if num_elements % num_columns != 0:
        message = ("The number of elements in the flattened array is not"
                   " compatible with the number of columns.")
        if error =="raise": 
            raise ValueError(message)
        elif error =='warn': 
            warnings.warn(message, UserWarning)
        return flattened_array
    # Calculate the number of rows that will be needed
    num_rows = num_elements // num_columns

    # Reshape the array
    reshaped_array = np.reshape(flattened_array, (num_rows, num_columns))

    # Check if we need to return a DataFrame or a Series
    if num_columns == 1 and is_string:
        return pd.Series(reshaped_array[:, 0], name=columns[0])
    else:
        return pd.DataFrame(reshaped_array, columns=columns)

def save_figure(fig, filename=None, dpi=300, close=True, ax=None, 
                tight_layout=False, bbox_inches='tight'):
    """
    Saves a matplotlib figure to a file and optionally closes it. 
    Automatically generates a unique filename if not provided.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str, optional
        The name of the file to save the figure to. If None, a unique name
        is generated based on the current date-time.
    dpi : int, optional
        The resolution of the output file in dots per inch.
    close : bool, optional
        Whether to close the figure after saving.
    ax : matplotlib.axes.Axes or array-like of Axes, optional
        Axes object(s) to perform operations on before saving.
    tight_layout : bool, optional
        Whether to adjust subplot parameters to give specified padding.
    bbox_inches : str, optional
        Bounding box in inches: 'tight' or a specific value.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from gofast.utils.base_utils import save_figure
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> ax.plot(x, y)
    >>> save_figure(fig, close=True, ax=ax)

    Notes
    -----
    If the filename is not specified, this function generates a filename that
    is unique to the second, using the pattern 'figure_YYYYMMDD_HHMMSS.png'.
    If two figures are saved within the same second, it appends microseconds
    to ensure uniqueness.
    """
    # Generate a unique filename if not provided
    if filename is None:
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"figure_{date_time}.png"
        while os.path.exists(filename):
            date_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"figure_{date_time}.png"

    # Adjust layout if requested
    if tight_layout:
        fig.tight_layout()

    # Optionally adjust axis properties
    if ax is not None:
        if isinstance(ax, (list, tuple, np.ndarray)):
            for a in ax:
                a.grid(True)
        else:
            ax.grid(True)
    
    # Save the figure
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved as '{filename}' with dpi={dpi}")

    # Optionally close the figure
    if close:
        plt.close(fig)
        print("Figure closed.")

def _handle_non_numeric(data, action='normalize'):
    """Process input data (Series, DataFrame, or ndarray) to ensure 
    it contains only numeric data.
    
    Parameters:
    data (pandas.Series, pandas.DataFrame, numpy.ndarray):
        Input data to process.
    
    Returns:
    numpy.ndarray: An array containing only numeric data.
    
    Raises:
    ValueError: If the processed data is empty after removing non-numeric types.
    TypeError: If the input is not a Series, DataFrame, or ndarray.
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        if isinstance(data, pd.Series):
            # Convert Series to DataFrame to use select_dtypes
            data = data.to_frame()
            # Convert back to Series if needed
            numeric_data = data.select_dtypes([np.number]).squeeze()  
        elif isinstance(data, pd.DataFrame):
            # For DataFrame, use select_dtypes to filter numeric data.
            numeric_data = data.select_dtypes([np.number])
        # For pandas data structures, select only numeric data types.
        if numeric_data.empty:
            raise ValueError(f"No numeric data to {action}.")
            
    elif isinstance(data, np.ndarray):
        # For numpy arrays, ensure the dtype is numeric.
        if not np.issubdtype(data.dtype, np.number):
            # Attempt to convert non-numeric numpy
            # array to a numeric one by coercion
            try:
                numeric_data = data.astype(np.float64)
            except ValueError:
                raise ValueError("Array contains non-numeric data that cannot"
                                 " be converted to numeric type.")
        else:
            numeric_data = data
    else:
        raise TypeError(
            "Input must be a pandas Series,"
            " DataFrame, or a numpy array."
        )
    
    # Check if resulting numeric data is empty
    if numeric_data.size == 0:
        raise ValueError("No numeric data available after processing.")
    
    return numeric_data

def _nan_checker(arr, allow_nan=False):
    """Check and handle NaN values in a numpy array, pandas Series, 
    or pandas DataFrame.

    Parameters:
    arr (numpy.ndarray, pandas.Series, pandas.DataFrame): The data to check
    for NaNs.
    allow_nan (bool): If False, raises an error if NaNs are found. If True, 
    replaces NaNs with zero.

    Returns:
    numpy.ndarray, pandas.Series, pandas.DataFrame: Data with NaNs handled 
    according to allow_nan.

    Raises:
    ValueError: If NaNs are found and allow_nan is False.
    """
    # Check for NaNs across different types
    if not allow_nan:
        if isinstance(arr, (np.ndarray, pd.Series, pd.DataFrame)): 
            contain_nans = np.isnan(arr).any() if isinstance (
                arr, np.ndarray)  else pd.isnull(arr).values.any()
            if contain_nans:
                raise ValueError("NaN values found, set allow_nan=True to handle them.")
    if allow_nan:
        if isinstance(arr, np.ndarray):
            arr = nan_to_mode(arr)  # Replace NaNs with zero for numpy arrays
        elif isinstance(arr, (pd.Series, pd.DataFrame)):
            arr = arr.fillna(0)  # Replace NaNs with zero for pandas Series or DataFrame
    
    return arr

def nan_to_mode(
    arr: np.ndarray,
    nan_policy: str = 'omit',
    axis: int = None,
    keepdims: bool = False,
    fill_value: float = None
) -> np.ndarray:
    """
    Replace NaN values in a numpy array with the mode (most frequent value).
    
    This function calculates the mode (the most frequent value) of the given 
    array, and replaces NaN values with this mode [1]_. The mode is computed 
    across the array, and the behavior can be customized based on the axis, 
    nan_policy, and whether to keep the dimensions of the array after 
    the operation [2]_.

    Parameters
    ----------
    arr : ndarray
        The input array which may contain NaN values. It can be a 
        one-dimensional or multi-dimensional numpy array. NaN values will
        be replaced with the computed mode.
    
    nan_policy : {'omit', 'raise'}, optional, default 'omit'
        Defines how to handle NaN values while computing the mode. If 'omit', 
        the NaN values are ignored during the computation. If 'raise', a 
        `ValueError` is raised if NaN values are encountered in the array.
        
    axis : int, optional, default None
        The axis along which to compute the mode. If `None`, the mode is 
        computed over the flattened array. If an integer is provided, it 
        computes the mode along that axis.
    
    keepdims : bool, optional, default False
        If True, the reduced dimensions will be retained as dimensions of 
        size one. If False, the reduced dimensions are removed.

    fill_value : float, optional, default None
        If provided, it replaces NaN values with the specified `fill_value` 
        instead of the mode.

    Returns
    -------
    ndarray
        The array with NaN values replaced by the mode (or the specified `fill_value`). 
        The shape of the output array will match the input array, except for 
        the dimensions reduced by the `axis` if specified.

    Notes
    -----
    - The mode is computed using `scipy.stats.mode` with the `nan_policy='omit'`
      argument.
    - If `nan_policy` is set to `'raise'`, an error is thrown if NaN values 
      are encountered.
    - `fill_value` can be used to specify a custom replacement for NaN values
      instead of the mode.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.utils.base_utils import nan_to_mode

    >>> arr = np.array([1, 2, 2, 3, np.nan, 4, np.nan, 2])
    >>> nan_to_mode(arr)
    array([1., 2., 2., 3., 2., 4., 2., 2.])

    >>> arr2 = np.array([[1, 2], [np.nan, 4]])
    >>> nan_to_mode(arr2, axis=0)
    array([[1., 2.],
           [2., 4.]])
           
    >>> nan_to_mode(arr, nan_policy='raise')
    ValueError: Input array contains NaN values

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., et al.
           "Array programming with NumPy." Nature 585, 357–362 (2019).
           https://doi.org/10.1038/s41586-019-1556-0

    .. [2] Virtanen, P., Gommers, R., Oliphant, T. E., et al. "SciPy 1.0: 
           fundamental algorithms for scientific computing in Python." 
           Nature Methods 17, 261–272 (2020). https://doi.org/10.1038/s41592-019-0686-2

    See Also
    --------
    numpy.nan_to_num : Replace NaN with a specified value
    scipy.stats.mode : Compute the mode of an array
    """
    arr = np.asarray( arr)

    # Calculate the mode, ignoring NaN values
    mode_result = stats.mode(
        arr, nan_policy=nan_policy, 
        axis=axis, 
        keepdims=keepdims
    )
    # mode_result is a tuple: (mode_values, count_values)
    mode_value = mode_result.mode
    
    # If a fill_value is provided, replace NaN values with
    # the fill_value instead of the mode
    if fill_value is not None:
        arr = np.where(np.isnan(arr), fill_value, arr)
    else:
        # Replace NaNs with the mode value
        arr = np.where(np.isnan(arr), mode_value, arr)

    return arr


def smooth1d(
    ar,
    drop_outliers:bool=True, 
    ma:bool=True, 
    absolute:bool=False,
    interpolate:bool=False, 
    view:bool=False , 
    x: ArrayLike=None, 
    xlabel:str =None, 
    ylabel:str =None, 
    fig_size:tuple = ( 10, 5) 
    )-> ArrayLike[float]: 
    """ Smooth one-dimensional array. 
    
    Parameters 
    -----------
    ar: ArrayLike 1d 
       Array of one-dimensional 
       
    drop_outliers: bool, default=True 
       Remove the outliers in the data before smoothing 
       
    ma: bool, default=True, 
       Use the moving average for smoothing array value. This seems more 
       realistic.
       
    interpolate: bool, default=False 
       Interpolate value to fit the original data size after NaN filling. 
       
    absolute: bool, default=False, 
       keep postive the extrapolated scaled values. Indeed, when scaling data, 
       negative value can be appear due to the polyfit function. to absolute 
       this value, set ``absolute=True``. Note that converting to values to 
       positive must be considered as the last option when values in the 
       array must be positive.
       
    view: bool, default =False 
       Display curves 
    x: ArrayLike, optional 
       Abscissa array for visualization. If given, it must be consistent 
       with the given array `ar`. Raises error otherwise. 
    xlabel: str, optional 
       Label of x 
    ylabel:str, optional 
       label of y  
    fig_size: tuple , default=(10, 5)
       Matplotlib figure size
       
    Returns 
    --------
    yc: ArrayLike 
       Smoothed array value. 
       
    Examples 
    ---------
    >>> import numpy as np 
    >>> from gofast.utils.base_utils  import smooth1d 
    >>> # add Guassian Noise 
    >>> np.random.seed (42)
    >>> ar = np.random.randn (20 ) * 20 + np.random.normal ( 20 )
    >>> ar [:7 ]
    array([6.42891445e+00, 3.75072493e-02, 1.82905357e+01, 2.92957265e+01,
           6.20589038e+01, 2.26399535e+01, 1.12596434e+01])
    >>> arc = smooth1d (ar, view =True , ma =False )
    >>> arc [:7 ]
    array([12.08603102, 15.29819907, 18.017749  , 20.27968322, 22.11900412,
           23.5707141 , 24.66981557])
    >>> arc = smooth1d (ar, view =True )# ma=True by default 
    array([ 5.0071604 ,  5.90839339,  9.6264018 , 13.94679804, 17.67369252,
           20.34922943, 22.00836725])
    """
    from .mathex import moving_average 
    # convert data into an iterable object 
    ar = np.array(
        is_iterable(ar, exclude_string = True , transform =True )) 
    
    if not _is_arraylike_1d(ar): 
        raise TypeError("Expect one-dimensional array. Use `gofast.smoothing`"
                        " for handling two-dimensional array.")
    if not _is_numeric_dtype(ar): 
        raise ValueError (f"{ar.dtype.name!r} is not allowed. Expect a numeric"
                          " array")
        
    arr = ar.copy() 
    if drop_outliers: 
        arr = remove_outliers( 
            arr, fill_value = np.nan , interpolate = interpolate )
    # Nan is not allow so fill NaN if exists in array 
    # is arraylike 1d 
    if not interpolate:
        # fill NaN 
        arr = reshape ( fillNaN( arr , method ='both') ) 
    if ma: 
        arr = moving_average(arr, method ='sma')
    # if extrapolation give negative  values
    # whether to keep as it was or convert to positive values. 
    # note that converting to positive values is 
    arr, *_  = scale_y ( arr ) 
    # if extrapolation gives negative values
    # convert to positive values or keep it intact. 
    # note that converting to positive values is 
    # can be used as the last option when array 
    # data must be positive.
    if absolute: 
        arr = np.abs (arr )
    if view: 
        x = np.arange ( len(ar )) if x is None else np.array (x )

        check_consistency_size( x, ar )
            
        fig,  ax = plt.subplots (1, 1, figsize = fig_size)
        ax.plot (x, 
                 ar , 
                 'ok-', 
                 label ='raw curve'
                 )
        ax.plot (x, 
                 arr, 
                 c='#0A4CEE',
                 marker = 'o', 
                 label ='smooth curve'
                 ) 
        
        ax.legend ( ) 
        ax.set_xlabel (xlabel or '')
        ax.set_ylabel ( ylabel or '') 
        
    return arr 

def smoothing (
    ar,  
    drop_outliers = True ,
    ma=True,
    absolute =False,
    interpolate=False, 
    axis = 0, 
    view = False, 
    fig_size =(7, 7), 
    xlabel =None, 
    ylabel =None , 
    cmap ='binary'
    ): 
    """ Smooth data along axis. 
    
    Parameters 
    -----------
    ar: ArrayLike 1d or 2d 
       One dimensional or two dimensional array. 
       
    drop_outliers: bool, default=True 
       Remove the outliers in the data before smoothing along the given axis 
       
    ma: bool, default=True, 
       Use the moving average for smoothing array value along axis. This seems 
       more realistic rather than using only the scaling method. 
       
    absolute: bool, default=False, 
       keep positive the extrapolated scaled values. Indeed, when scaling data, 
       negative value can be appear due to the polyfit function. to absolute 
       this value, set ``absolute=True``. Note that converting to values to 
       positive must be considered as the last option when values in the 
       array must be positive.
       
    axis: int, default=0 
       Axis along with the data must be smoothed. The default is the along  
       the row. 
       
    view: bool, default =False 
       Visualize the two dimensional raw and smoothing grid. 
       
    xlabel: str, optional 
       Label of x 
       
    ylabel:str, optional 
    
       label of y  
    fig_size: tuple , default=(7, 5)
       Matplotlib figure size 
       
    cmap: str, default='binary'
       Matplotlib.colormap to manage the `view` color 
      
    Return 
    --------
    arr0: ArrayLike 
       Smoothed array value. 
    
    Examples 
    ---------
    >>> import numpy as np 
    >>> from gofast.utils.base_utils  import smoothing
    >>> # add Guassian Noises 
    >>> np.random.seed (42)
    >>> ar = np.random.randn (20, 7 ) * 20 + np.random.normal ( 20, 7 )
    >>> ar [:3, :3 ]
    array([[ 31.5265026 ,  18.82693352,  34.5459903 ],
           [ 36.94091413,  12.20273182,  32.44342041],
           [-12.90613711,  10.34646896,   1.33559714]])
    >>> arc = smoothing (ar, view =True , ma =False )
    >>> arc [:3, :3 ]
    array([[32.20356863, 17.18624398, 41.22258603],
           [33.46353806, 15.56839464, 19.20963317],
           [23.22466498, 13.8985316 ,  5.04748584]])
    >>> arcma = smoothing (ar, view =True )# ma=True by default
    >>> arcma [:3, :3 ]
    array([[23.96547827,  8.48064226, 31.81490918],
           [26.21374675, 13.33233065, 12.29345026],
           [22.60143346, 16.77242118,  2.07931194]])
    >>> arcma_1 = smoothing (ar, view =True, axis =1 )
    >>> arcma_1 [:3, :3 ]
    array([[18.74017857, 26.91532187, 32.02914421],
           [18.4056216 , 21.81293014, 21.98535213],
           [-1.44359989,  3.49228057,  7.51734762]])
    """
    ar = np.array ( 
        is_iterable(ar, exclude_string = True , transform =True )
        ) 
    if ( 
            str (axis).lower().find('1')>=0 
            or str(axis).lower().find('column')>=0
            ): 
        axis = 1 
    else : axis =0 
    
    if _is_arraylike_1d(ar): 
        ar = reshape ( ar, axis = 0 ) 
    # make a copy
    arr = ar.copy() 
    along_axis = arr.shape [1] if axis == 0 else len(ar) 
    arr0 = np.zeros_like (arr)
    for ix in range (along_axis): 
        value = arr [:, ix ] if axis ==0 else arr[ix , :]
        yc = smooth1d(value, drop_outliers = drop_outliers , 
                      ma= ma, view =False , absolute =absolute , 
                      interpolate= interpolate, 
                      ) 
        if axis ==0: 
            arr0[:, ix ] = yc 
        else : arr0[ix, :] = yc 
        
    if view: 
        fig, ax  = plt.subplots (nrows = 1, ncols = 2 , sharey= True,
                                 figsize = fig_size )
        ax[0].imshow(arr ,interpolation='nearest', label ='Raw Grid', 
                     cmap = cmap )
        ax[1].imshow (arr0, interpolation ='nearest', label = 'Smooth Grid', 
                      cmap =cmap  )
        
        ax[0].set_title ('Raw Grid') 
        ax[0].set_xlabel (xlabel or '')
        ax[0].set_ylabel ( ylabel or '')
        ax[1].set_title ('Smooth Grid') 
        ax[1].set_xlabel (xlabel or '')
        ax[1].set_ylabel ( ylabel or '')
        plt.legend
        plt.show () 
        
    if 1 in ar.shape: 
        arr0 = reshape (arr0 )
        
    return arr0 
    
def _count_local_minima(arr: ArrayLike, method: str = 'robust') -> int:
    if method == 'base':
        return sum(
            1 for i in range(1, len(arr) - 1) if arr[i] < arr[i - 1] 
            and arr[i] < arr[i + 1]
            )
    else:
        return len(argrelextrema(np.array(arr), np.less)[0])

def scale_y(
    y: ArrayLike, 
    x: ArrayLike = None, 
    deg: Union [int, str] = 'auto', 
    func: _F = None, 
    return_xf: bool = False, 
    view: bool = False
) -> Tuple[ArrayLike, ArrayLike, _F]:
    """
    Scaling value using a fitting curve.

    Create polyfit function from specific data points `x` to correct `y` 
    values.

    Parameters
    ----------
    y : ArrayLike
        Array-like of y-axis. This is the array of values to be scaled.
    x : ArrayLike, optional
        Array-like of x-axis. If `x` is given, it should be the same length 
        as `y`, otherwise an error will occur. Default is ``None``.
    deg : int, optional
        Polynomial degree. If value is ``auto`` or ``None``, it will be computed 
        using the length of extrema (local and/or global) values.
    func : _F, optional
        Callable - The model function, ``f(x, ...)``. It must take the 
        independent variable as the first argument and the parameters to 
        fit as separate remaining arguments. `func` can be a ``linear`` 
        function i.e. for ``f(x)= ax + b`` where `a` is slope and `b` is 
        the intercept value. It is recommended to set up a custom function 
        according to the `y` value distribution for better fitting. If 
        `func` is given, `deg` is not needed.
    return_xf : bool, optional
        If True, returns the new x-axis and the fitting function. Default 
        is ``False``.
    view : bool, optional
        If True, visualizes the original and scaled data. Default is 
        ``False``.

    Returns
    -------
    yc : ArrayLike
        Array of scaled/projected sample values obtained from `f`.
    x : ArrayLike
        New x-axis generated from the samples (if `return_xf` is True).
    f : _F
        Linear or polynomial function `f` (if `return_xf` is True).

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.random.seed(42)
    >>> x0 = 10 * np.random.rand(50)
    >>> y = 2 * x0 + np.random.randn(50) - 1
    >>> plt.scatter(x0, y)
    >>> yc, x, f = scale_y(y, return_xf=True, view=True)
    >>> plt.plot(x, y, label='Original Data')
    >>> plt.plot(x, yc, label='Scaled Data')
    >>> plt.legend()
    >>> plt.show()

    Notes
    -----
    - The function checks if `x` and `y` are of the same length.
    - If `func` is provided, `deg` is ignored.
    - The function raises a `TypeError` if `func` is not callable.
    - The degree of the polynomial is determined by the number of local 
      minima plus one.
    - In case of fitting errors, a `ValueError` is raised with a suggestion 
      to check the polynomial degree.
      
    References 
    -----------
    Wikipedia, Curve fitting, https://en.wikipedia.org/wiki/Curve_fitting
    Wikipedia, Polynomial interpolation, https://en.wikipedia.org/wiki/Polynomial_interpolation
    """
    _, name_or_columns, index = pandas_manager(y )
    
    y = check_y(y, y_numeric =True )
    if func is not None and (not callable(func) or not hasattr(func, '__call__')):
        raise TypeError(f"`func` argument is not a callable; got {type(func).__name__!r}")

    degree = _count_local_minima(y) + 1
    if x is None:
        x = np.arange(len(y))
        
    x = check_y(x, input_name="x")
    
    if len(x) != len(y):
        raise ValueError("`x` and `y` arrays must have the same length."
                         f" Got lengths {len(x)} and {len(y)}.")
    
    deg= None if deg =='auto' else deg 
    
    try: 
        coeff = np.polyfit(x, y, int(deg) if deg is not None else degree)
        f = np.poly1d(coeff) if func is None else func
        yc = f(x)
    except np.linalg.LinAlgError:
        raise ValueError("Check the number of degrees. SVD did not converge"
                         " in Linear Least Squares.")

    if view: 
        plt.plot(x, y, "-ok", label='Original Data')
        plt.plot(x, yc, "-or", label='Scaled Data')
        plt.xlabel ("x") ; plt.ylabel("y")
        plt.legend()
        plt.show()
    
    yc = pandas_manager(yc, todo="set", name_or_columns=name_or_columns,
                        index =index ) 
    
    return (yc, x, f) if return_xf else yc

def _visualize_interpolation(
    original_data: Union[np.ndarray, pd.Series, pd.DataFrame], 
    interpolated_data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> None:
    """
    Helper function to visualize original and interpolated data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    if isinstance(original_data, (pd.Series, np.ndarray)
                  ) and original_data.ndim == 1:
        axes[0].plot(original_data, label='Original Data', marker='o')
        axes[1].plot(interpolated_data, label='Interpolated Data', marker='o')
    elif isinstance(original_data, (pd.DataFrame, np.ndarray)
                    ) and original_data.ndim == 2:
        axes[0].imshow(original_data, aspect='auto', interpolation='none')
        axes[1].imshow(interpolated_data, aspect='auto', interpolation='none')
    
    axes[0].set_title('Original Data')
    axes[1].set_title('Interpolated Data')
    plt.show()

def interpolate_data(
    data: Union[ArrayLike, Series, DataFrame], 
    method: str = 'slinear', 
    order: Optional[int] = None, 
    fill_value: str = 'extrapolate', 
    axis: int = 0, 
    drop_outliers: bool = False, 
    outlier_method: str = "IQR", 
    view: bool = False, 
    **kwargs
) -> Union[ArrayLike, Series, DataFrame]:
    """
    Interpolates 1D or 2D data, allowing for NaN values, and visualizes 
    the result if requested.

    Parameters
    ----------
    data : Union[np.ndarray, pd.Series, pd.DataFrame]
        Input data to be interpolated. Can be a 1D or 2D numpy array, 
        pandas Series, or pandas DataFrame.
    method : str, optional
        Method of interpolation. Options include 'linear', 'nearest', 
        'zero', 'slinear', 'quadratic', and 'cubic'. Default is 'slinear'.
    order : Optional[int], optional
        The order of the spline interpolation. Only applicable for some 
        methods. Default is None.
    fill_value : str, optional
        Specifies the fill value for points outside the interpolation 
        domain. Default is 'extrapolate'.
    axis : int, optional
        Axis along which to interpolate, if data is a DataFrame. Default 
        is 0.
    drop_outliers : bool, optional
        If True, outliers will be removed before interpolation. Default 
        is False.
    outlier_method : str, optional
        Method for outlier detection if drop_outliers is True. Options 
        are 'IQR' and 'z-score'. Default is 'IQR'.
    view : bool, optional
        If True, visualizes the original and interpolated data in two 
        panels. Default is False.
    **kwargs
        Additional keyword arguments to pass to the interpolation 
        function.

    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Interpolated data, returned in the same type as the input data.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd 
    >>> from gofast.utils.base_utils import interpolate_data
    >>> s = pd.Series([1, np.nan, 3, 4, np.nan, 6])
    >>> interpolate_data(s, view=True)
    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    5    6.0
    dtype: float64

    >>> df = pd.DataFrame([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 8, 9]])
    >>> interpolate_data(df, view=True)
         0    1    2
    0  1.0  6.5  3.0
    1  4.0  5.0  6.0
    2  4.0  8.0  9.0

    Notes
    -----
    - The 'slinear' method is not available for 2D interpolation. It 
      defaults to 'linear' for 2D data.
    - The 'extrapolate' fill_value is not available for 2D interpolation 
      and defaults to 0.
    - If drop_outliers is True, the specified outlier detection method 
      will be applied before interpolation.
    """
    valid_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
    method = str(method).lower() 
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method. Expected one of {valid_methods}, got {method}.")
    
    
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        data = np.asarray(data)
    
    data = _handle_non_numeric(data, action="interpolate")
    data, name_or_columns, index = pandas_manager(data)
    if drop_outliers: 
        data = remove_outliers(data, method=outlier_method)
        
    if isinstance(data, pd.Series):
        x = np.arange(len(data))
        y = data.values
        mask = ~np.isnan(y)
        f = interp1d(x[mask], y[mask], kind=method, fill_value=fill_value,
                     bounds_error=False, **kwargs)
        data_interp = f(x)
        result = pd.Series(data_interp, index=data.index)
    
    elif isinstance(data, pd.DataFrame):
        data_interp = data.apply(lambda col: interpolate_data(
            col, method=method, 
            order=order, fill_value=fill_value, axis=axis, 
            **kwargs), axis=axis)
        result = data_interp
    
    elif data.ndim == 1:
        x = np.arange(len(data))
        y = data
        mask = ~np.isnan(y)
        f = interp1d(x[mask], y[mask], kind=method, fill_value=fill_value,
                     bounds_error=False, **kwargs)
        data_interp = f(x)
        result = data_interp
    
    elif data.ndim == 2:
        # slinear not available for two dimensional interpolation 
        method = 'linear' if method.find("linear")>=0 else method 
        
        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        x_grid, y_grid = np.meshgrid(x, y)
        mask = ~np.isnan(data)
        points = np.array((y_grid[mask], x_grid[mask])).T
        values = data[mask]
        x_new, y_new = np.meshgrid(x, y)
        # 'extrapolate fill_value' not available.
        fill_value = 0. if str(fill_value).find("extra")>=0 else fill_value
        data_interp = griddata(points, values, (
            y_new, x_new), method=method, fill_value=fill_value, **kwargs)
        result = data_interp
 
    if view:
        _visualize_interpolation(data, result)
    
    if name_or_columns is not None: 
        result = pandas_manager(
            result, 
            name_or_columns = name_or_columns,
            index = index, 
            todo='set'
        )
    
    return result

def normalizer(
    *arrays: tuple[np.ndarray], 
    method: str = '01', 
    scaler: str = 'naive', 
    allow_nan: bool = False, 
    axis: Optional[int] = None) -> List[np.ndarray]:
    """
    Normalize given arrays using a specified method and scaler, optionally 
    along a specified axis. 
    
    Handles non-numeric data and NaNs according to the parameters.

    Parameters
    ----------
    arrays : tuple of np.ndarray
        A tuple containing one or more arrays (either 1D or 2D) to be normalized.
    method : str, default '01'
        Specifies the normalization method to apply. Options include:
        - '01' : Normalizes the data to the range [0, 1].
        - 'zscore' : Uses Z-score normalization (standardization).
        - 'sum' : Scales the data so that the sum is 1.
        Note that the 'sum' method is not compatible with the 'sklearn' scaler.
    scaler : str, default 'naive'
        Specifies the type of scaling technique to use. Options include:
        - 'naive' : Simple mathematical operations based on the method.
        - 'sklearn' : Utilizes scikit-learn's MinMaxScaler or StandardScaler,
          depending on the `method` specified.
    allow_nan : bool, default False
        Determines how NaN values should be handled. If False, the function 
        will raise an error if NaN values are present. If True, NaNs will be 
        replaced with zero or handled by an imputer,
        depending on the context.
    axis : int, optional
        The axis along which to normalize the data. By default (None), the 
        data is normalized based on all elements. If specified, normalization 
        is done along the axis for 2D arrays:
        - axis=0 : Normalize each column.
        - axis=1 : Normalize each row.

    Returns
    -------
    list of np.ndarray
        A list of normalized arrays. If only one array is provided, a single 
        normalized array is returned.

    Raises
    ------
    ValueError
        If `allow_nan` is False and NaNs are detected.
        If an invalid normalization method or combination of scaler and method 
        is specified.

    Notes
    -----
    - The function internally converts pandas DataFrames and Series to numpy 
      arrays for processing.
    - Non-numeric data types within arrays are filtered out before normalization.
    - It's important to consider the scale of values and distribution of data
      when choosing a normalization method,
      as each method can significantly affect the outcome and interpretation 
      of results.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.utils.base_utils import normalizer 
    >>> arr = np.array([10, 20, 30, 40, 50])
    >>> normalizer(arr, method='01', scaler='naive')
    array([0. , 0.25, 0.5 , 0.75, 1. ])
    
    >>> arr2d = np.array([[1, 2], [3, 4]])
    >>> normalizer(arr2d, method='zscore', axis=0)
    array([[-1., -1.], [ 1.,  1.]])
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    def _normalize_array(arr, method, scaler):
        arr = _nan_checker(arr, allow_nan = allow_nan )
        if scaler in ['sklearn', 'scikit-learn']:
            is_array_1d =False 
            if method == 'sum':
                raise ValueError(
                    "`sum` method is not valid with `scaler`='sklearn'.")
            scaler_object = MinMaxScaler() if method == '01' else StandardScaler()
            
            if arr.ndim ==1: 
                arr = ( 
                    np.asarray(arr).reshape(-1, 0) if axis ==0 
                    else np.asarray(arr).reshape(1, -1)  
                    )
                is_array_1d =True 
            scaled = scaler_object.fit_transform(
                arr if axis == 0 else arr.T).T if axis == 0 else\
                scaler_object.fit_transform(arr.T).T
            if is_array_1d: 
                scaled = scaled.flatten() 
        else:  # naive scaling
            arr, name_or_columns, index  = pandas_manager(arr )
            if axis is None:
                arr_min = np.min(arr)
                arr_max = np.max(arr)
            else:
                arr_min = np.min(arr, axis=axis, keepdims=True)
                arr_max = np.max(arr, axis=axis, keepdims=True)

            if method == '01':
                scaled = (arr - arr_min) / (arr_max - arr_min)
            elif method in [ 'z-score', 'zscore']:
                mean = np.mean(arr, axis=axis, keepdims=True)
                std = np.std(arr, axis=axis, keepdims=True)
                scaled = (arr - mean) / std
            elif method == 'sum':
                sum_val = np.sum(arr, axis=axis, keepdims=True)
                scaled = arr / sum_val
            else:
                raise ValueError(f"Unknown method '{method}'. Valid methods"
                                 " are '01', 'zscore', and 'sum'.")
                
        # revert back to series or dataframe is given as it 
        if name_or_columns is not None: 
            scaled = pandas_manager(
                scaled, todo='set', 
                name_or_columns=name_or_columns, 
                index= index 
                )

        return scaled
    # Normalize each array
    normalized_arrays = []
    for arr in arrays:
        if not hasattr(arr, '__array__'):
            arr = np.asarray(arr)
        arr = _handle_non_numeric(arr)
        normalized = _normalize_array(arr, method, scaler)
        normalized_arrays.append(normalized)
    
    return normalized_arrays[0] if len(
        normalized_arrays) == 1 else normalized_arrays   

def _handle_get_action(
        data: Any, action: str, error: str) -> Union[bool, Tuple[np.ndarray, Any]]:
    is_pandas = isinstance(data, (pd.Series, pd.DataFrame))
    if action == 'check_only':
        return is_pandas

    arr = data.to_numpy() if is_pandas else data
    index = data.index if is_pandas else None  
    if error == 'raise' and not is_pandas:
        raise TypeError("Expected a pandas Series or DataFrame, but got a non-pandas type.")
    elif error == 'warn' and not is_pandas:
        warnings.warn("Expected a pandas Series or DataFrame, but got a non-pandas type.")

    name_or_columns = data.name if isinstance(data, pd.Series) else\
        data.columns if is_pandas else None
        
    if action=='keep_frame': 
        arr= data 
    return arr, name_or_columns, index 

def _handle_set_action(data: Any, name_or_columns: Any, action: str, error: str, 
                       index=None, 
                       ) -> Union[pd.Series, pd.DataFrame]:
    if name_or_columns is None and action != 'default':
        raise ValueError(
            "The 'name_or_columns' parameter cannot be None when setting data. "
            "Provide a valid 'name_or_columns' or set the 'action' to 'default'."
        )
    is_pandas = isinstance(data, (pd.Series, pd.DataFrame))
    
    if is_pandas and name_or_columns is None:
        # Not set anymore when is already a series or dataframe.
        name_or_columns =  data.name if isinstance(data, pd.Series) else data.columns 
        
    data = np.asarray(data).squeeze()
    if data.ndim == 1:
        if isinstance(name_or_columns, (list, tuple)):
            if error == 'raise':
                raise ValueError("1D array provided; 'name_or_columns' should be a single string.")
            name_or_columns = list(name_or_columns)[0]
        return pd.Series(data, name=name_or_columns, index =index  )
    else:
        if isinstance(name_or_columns, str):
            name_or_columns = [name_or_columns]
        return pd.DataFrame(data, columns=name_or_columns, index =index )

def pandas_manager(
    data: Any, todo: str = 'get', 
    name_or_columns: Any = None, 
    action: str = 'as_array', 
    error: str = 'passthrough', 
    index: bool = None
) -> Union[np.ndarray, Series, DataFrame, Tuple[np.ndarray, Any], bool]:
    """
    Manages pandas objects by getting or setting data, with error handling.

    Parameters
    ----------
    data : Any
        The input data which can be a numpy array, pandas Series, or DataFrame.
    todo : str, optional
        Specifies the action to perform: 'get' or 'set'. Default is 'get'.
    name_or_columns : Any, optional
        Name for Series or columns for DataFrame when setting data.
    action : str, optional
        Specifies the sub-action for 'get': 'as_array' or 'check_only' or 'keep_frame'. 
        - `as_array` converts data to Numpy array 
        - `check_only` checks only whether the data is passed as pandas Series or 
          DataFrame
        - `keep_frame` keeps the dataframe but returns additionals frame attributes 
          like columns/name  and index. 
        Default is 'as_array'.
    error : str, optional
        Error handling strategy: 'passthrough', 'raise', or 'warn'. 
        Default is 'passthrough'.
    index : bool, optional
        Whether to include the index when getting or setting data. Default is None.

    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame, Tuple[np.ndarray, Any], bool]
        Depending on the action, returns different types of results:
        - If `todo` is 'get' and `action` is 'check_only': returns a boolean 
          indicating if the data is a pandas object.
        - If `todo` is 'get' and `action` is 'as_array': returns the data as 
          a NumPy array along with the original pandas attributes.
        - If `todo` is 'set': returns the data as a pandas Series or DataFrame.

    Raises
    ------
    ValueError
        If the 'todo' parameter is not 'get' or 'set'.
        If 'name_or_columns' is None when setting data and action is not 'default'.
    TypeError
        If the data is not a pandas Series or DataFrame when expected.

 
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.utils.base_utils import pandas_manager

    Get action with pandas DataFrame:
    >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    >>> array, columns, index = pandas_manager(df, todo='get', action='as_array')
    >>> print(array)
    [[1 2]
     [3 4]]
    >>> print(columns)
    Index(['A', 'B'], dtype='object')

    Set action with 1D numpy array:
    >>> arr = np.array([1, 2, 3])
    >>> series = pandas_manager(arr, todo='set', name_or_columns='Value')
    >>> print(series)
    0    1
    1    2
    2    3
    Name: Value, dtype: int64

    Set action with 2D numpy array:
    >>> arr_2d = np.array([[1, 2], [3, 4]])
    >>> df_2d = pandas_manager(arr_2d, todo='set', name_or_columns=['A', 'B'])
    >>> print(df_2d)
       A  B
    0  1  2
    1  3  4
    """
    todo = todo.lower()
    if todo not in ('get', 'set'):
        raise ValueError("Invalid 'todo' parameter. Must be 'get' or 'set'.")

    if todo == 'get':
        return _handle_get_action(data, action, error)
    elif todo == 'set':
        return _handle_set_action(
            data, name_or_columns, action , error, index)

def make_df(
    X, y=None, 
    prefix=None, 
    target_names=None,
    coerce=False, 
    error='raise', 
    fill_value=np.nan,
    truncate_X=False
):
    """
    Prepare a DataFrame from input data `X` and `y` with appropriate naming
    conventions and length checks.

    Parameters
    ----------
    X : array-like or DataFrame
        Input features. If not a DataFrame, `X` will be converted into one with 
        column names generated using the `prefix` parameter.

    y : array-like or Series, optional
        Target values. If provided, `y` will be converted into a DataFrame. 
        If `y` is a 1D array and `target_names` is not given, 'target' will be 
        used as the default name. For a 2D array, `target_names` will be used 
        to name the columns.

    prefix : str, optional
        Prefix for feature column names if `X` is not a DataFrame. Default is 'feature_'.

    target_names : list of str, optional
        Names for target columns if `y` is 2D. If `target_names` is not provided, 
        columns will be named using 'target_' prefix.

    coerce : bool, optional
        Whether to coerce the lengths of `X` and `y` to match if they are inconsistent.
        Default is False. If True and `len(X) < len(y)`, `X` will be repeated to 
        match `y`'s length. If `len(y) < len(X)`, `y` will be forward-filled to 
        match `X`'s length.

    error : str, optional
        Error handling strategy. Options are:
        - 'raise': Raises an error if lengths of `X` and `y` are inconsistent 
          and `coerce` is False.
        - 'warn': Issues a warning instead of raising an error.
        - 'ignore': Ignores the inconsistency and proceeds. Default is 'raise'.

    fill_value : scalar, optional
        Value to use for filling missing values if `coerce` is True and 
        `len(y) < len(X)`. Default is `np.nan`.

    truncate_X : bool, optional
        If True and `len(y) < len(X)`, truncate `X` to match the length of `y`.
        Default is False.

    Returns
    -------
    DataFrame
        Combined DataFrame with features and target(s).

    Notes
    -----
    This function is useful for preparing data for machine learning tasks, 
    ensuring consistency between features and targets. The function handles 
    common issues such as inconsistent lengths and missing values, providing 
    options to coerce or truncate data as needed.

    Examples
    --------
    >>> from gofast.utils.base_utils import make_df
    >>> X = np.random.rand(90, 5)
    >>> y = np.random.rand(100)
    >>> df = make_df(X, y, coerce=True, error='ignore')
    >>> print(df.head())

    See Also
    --------
    pd.DataFrame : Pandas DataFrame constructor.
    np.asarray : Convert input to an array.

    References
    ----------
    .. [1] Pandas Documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    .. [2] NumPy Documentation: https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
    """
    
    def to_dataframe(data, prefix, default_name=None):
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, pd.Series):
            return data.to_frame(name=default_name)
        data = np.asarray(data)
        if data.ndim == 1:
            default_name ='feature' if default_name=='feature_' else default_name 
            return pd.DataFrame(data, columns=[default_name])
        column_names = [f"{prefix}{i}" for i in range(data.shape[1])]
        
        default_name = column_names if (
            default_name is None or default_name == f'{prefix}') else default_name
        
        if isinstance(default_name, str):
            default_name = [default_name]
        
        if isinstance(default_name, (list, tuple)) and len(default_name) != len(column_names):
            if error == 'warn':
                warnings.warn("Provided default names must match the number"
                              " of columns in data. Using generated names instead.")
            elif error == 'raise':
                raise ValueError("Provided default names must match the number of columns in data.")
            default_name = column_names
        
        return pd.DataFrame(data, columns=default_name)

    prefix = prefix or 'feature_'
    
    # Convert X to DataFrame
    X = to_dataframe(X, prefix, f"{prefix}")
    
    # Convert y to DataFrame if it is provided
    if y is not None:
        if isinstance(target_names, str):
            target_names = [target_names]
            
        y = to_dataframe(y, 'target_', target_names if target_names else 'target')

    # Check for length consistency
    if y is not None:
        if len(X) != len(y):
            msg = f"Inconsistent lengths: len(X)={len(X)}, len(y)={len(y)}."
            if len(X) < len(y):
                if error == 'raise' and not coerce:
                    raise ValueError(msg)
                elif coerce:
                    multiplier = (len(y) // len(X)) + 1
                    X = pd.concat([X] * multiplier).reset_index(drop=True).iloc[:len(y)]
                    warnings.warn(f"{msg} Coercing X to match y length.")
                elif error == 'ignore':
                    pass
            elif len(y) < len(X):
                if truncate_X:
                    X = X.iloc[:len(y)]
                    if error =='warn': 
                        warnings.warn(f"{msg} Truncating X to match y length.")
                elif coerce:
                    y = y.reindex(np.arange(len(X)), method='ffill',
                                  fill_value=fill_value)
                    warnings.warn(f"{msg} Forward filling y to match X length.")
                else:
                    raise ValueError(
                        msg + " y cannot have fewer rows than X. Please check the input.")

    # Combine X and y
    if y is not None:
        return pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    
    return X

def update_df(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    return_common_dfs=False, 
    return_common_columns=False,
    error='warn'
):
    """
    `update_df` function is designed to update a given DataFrame (`old_df`) 
    by replacing the common columns with the corresponding values from a 
    new DataFrame (`new_df`). This function supports a variety of use cases, 
    such as returning updated DataFrames, extracting common columns, or 
    handling potential errors when the common columns are missing.

    Parameters
    ----------
    old_df : pandas.DataFrame
        The original DataFrame to be updated. It must contain columns 
        that can be matched with those in `new_df`.

    new_df : pandas.DataFrame
        The DataFrame that contains updated values for the common columns 
        with `old_df`. It should have columns that overlap with `old_df`.

    return_common_dfs : bool, optional, default=False
        If set to True, the function will return two DataFrames containing 
        only the common columns between `old_df` and `new_df`:
        - The first DataFrame will be from `old_df`.
        - The second DataFrame will be from `new_df`.
        If False (default), the function will update the common columns in 
        `old_df` with the corresponding values from `new_df`.

    return_common_columns : bool, optional, default=False
        If set to True, the function will return a list of common column 
        names between `old_df` and `new_df`. This can be useful to 
        inspect which columns will be updated or matched between the DataFrames.

    error: {'warn', 'raise'}, optional, default='warn'
        Defines the action to take if no common columns are found between 
        `old_df` and `new_df`. 
        - 'warn' (default) will display a warning message.
        - 'raise' will raise an exception (`ValueError`).

    Returns
    -------
    updated_df : pandas.DataFrame
        If neither `return_common_columns` nor `return_common_dfs` is 
        specified, this function returns the original `old_df` with its 
        common columns updated to the values from `new_df`.

    common_columns : list
        If `return_common_columns` is set to True, returns a list of 
        column names common to both `old_df` and `new_df`.

    common_old_df : pandas.DataFrame
        If `return_common_dfs` is set to True, returns a DataFrame 
        containing the common columns of `old_df`.

    common_new_df : pandas.DataFrame
        If `return_common_dfs` is set to True, returns a DataFrame 
        containing the common columns of `new_df`.

    Examples
    --------
    >>> from gofast.utils.base_utils import update_df

    # Example 1: Return only common columns
    >>> updated_common = update_df(old_df, new_df, return_common_columns=True)
    >>> print(updated_common)
    ['A', 'B']

    # Example 2: Return DataFrames for common columns
    >>> common_old, common_new = update_df(
    ...    old_df, new_df, return_common_dfs=True)
    >>> print(common_old)
       A  B
    0  1  4
    1  2  5
    2  3  6
    >>> print(common_new)
        A   B
    0  10  40
    1  20  50
    2  30  60

    # Example 3: Update full DataFrame with common columns
    >>> updated_full = update_df(
    ...    old_df, new_df, return_common_columns=False, return_common_dfs=False)
    >>> print(updated_full)
        A   B  C
    0  10  40  7
    1  20  50  8
    2  30  60  9

    Notes
    -----

    Given two DataFrames `old_df` and `new_df`, we define a set of common 
    columns:

    .. math:: 
        C = \text{columns}(old\_df) \cap \text{columns}(new\_df)

    The function then replaces the values in the common columns of `old_df` 
    with the corresponding values from `new_df`:

    .. math::
        \text{updated\_df}[C] = \text{new\_df}[C]

    Where `C` is the set of common columns between `old_df` and `new_df`. 
    This operation is done only for the common columns, and other columns 
    remain unchanged.

    - The function allows flexibility in handling errors and selecting 
      what to return: the updated DataFrame, common columns, or common 
      DataFrames.
    - The error policy can be customized to either raise an exception or 
      warn the user when no common columns are found.
    - The function assumes that `old_df` and `new_df` are pandas DataFrames 
      with at least some overlapping column names.


    See Also
    --------
    - :func:`pandas.DataFrame`
    - :func:`pandas.DataFrame.intersection`

    References
    ----------
    .. [1] McKinney, W. "Data Structures for Statistical Computing in 
       Python." Proceedings of the 9th Python in Science Conference, 
       2010.
       https://conference.scipy.org/proceedings/scipy2010/pdfs/mckinney.pdf
    """

    # Check if the DataFrames are valid
    if not isinstance(old_df, pd.DataFrame) or not isinstance(new_df, pd.DataFrame):
        raise ValueError("Both old_df and new_df must be pandas DataFrame.")

    # Find common columns
    common_columns = old_df.columns.intersection(new_df.columns)
    
    # Handle error policy for missing common columns
    if len(common_columns) == 0:
        if error== "warn":
            print("Warning: No common columns between the two DataFrames.")
        elif error== "raise":
            raise ValueError("No common columns found between the two DataFrames.")
    
    # Case 1: Return only the common columns
    if return_common_columns:
        return common_columns.tolist()

    # Case 2: Return DataFrames for common columns
    if return_common_dfs:
        common_old_df = old_df[common_columns]
        common_new_df = new_df[common_columns]
        return common_old_df, common_new_df
    
    # Case 3: Return the full DataFrame with updated common columns
    # Update the common columns from new_df to old_df
    updated_df = old_df.copy()
    updated_df[common_columns] = new_df[common_columns]
    
    return updated_df

def validate_target_in(df, target, error='raise', verbose=0): 
    """
    Validate and process the target variable, ensuring it is consistent
    with the features in the DataFrame.
    
    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the features (X) and possibly the target column.
    - target: str, pandas Series, or pandas DataFrame
        The target variable to validate and process.
    - error: {'raise', 'warn', 'ignore'}, optional (default: 'raise')
        Defines behavior if there are issues with target validation.
        - 'raise': Raise an error if validation fails.
        - 'warn': Issue a warning and continue.
        - 'ignore': Ignore any issues.
    - verbose: int, optional (default: 0)
        Verbosity level for logging.
        - 0: No output.
        - 1: Basic info.
        - 2: Detailed info.

    Returns:
    - target: pandas Series
        The processed target variable.
    - df: pandas DataFrame
        The DataFrame containing the features and target.
    """
    is_frame(
        df, df_only=True, 
        raise_exception =True,
        objname="Data 'df'" 
    )
    # If target is a string, try to extract
    # the corresponding column from the DataFrame
    if isinstance(target, (str, list, tuple)):
        if verbose >= 1:
            print(f"Target is a string: Extracting '{target}'"
                  " column from the DataFrame.")
        target, df = extract_target(
            df, target_names=target, 
            return_y_X=True
        )
        
    # If target is a DataFrame, attempt to convert it 
    # to a pandas Series (if it has a single column)
    if isinstance(target, pd.DataFrame):
        if target.shape[1] == 1:
            if verbose >= 1:
                print("Target is a DataFrame with a single column."
                      " Converting to Series.")
            target = to_series(target)
        else:
            if error == 'raise':
                raise ValueError("If 'target' is a DataFrame, it"
                                 " must have a single column.")
            elif error == 'warn':
                warnings.warn("Target DataFrame has more than one column."
                      " Using the first column.")
                target = target.iloc[:, 0]  # Use the first column as the target
            else:
                # Default behavior: use the" first column if there are multiple columns
                target = target.iloc[:, 0]  

    # If target is a pandas Series, just use it as-is
    if isinstance(target, pd.Series):
        if verbose >= 1:
            print("Target is a pandas Series. Proceeding with it directly.")
    
    # Check that the length of the target matches the length of the DataFrame
    check_consistent_length(df, target)
    
    return target, df

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    