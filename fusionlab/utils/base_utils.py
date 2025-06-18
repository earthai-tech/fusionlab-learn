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

Adapted for FusionLab from the original fusionlab.utils.base_utils.
"""

from __future__ import annotations
import os
import re
import shutil
import warnings

import numpy as np
import pandas as pd

from ..api.types import (
    Union, List, Optional, Tuple,  Any,  Pattern, Dict,
    DataFrame, ArrayLike, Series, Callable, 
)
from ..compat.pandas import select_dtypes
from ..core.array_manager import (
    to_numeric_dtypes, reshape, 
    to_array, array_preserver,
    drop_nan_in, to_series
)
from ..core.checks import (
    is_iterable, validate_feature,
    is_numeric_dtype, check_datetime
)
from ..core.io import is_data_readable
from ..core.utils import smart_format
from .deps_utils import import_optional_dependency
from .validator import (
    check_consistent_length,  
     build_data_if, 
    parameter_validator,
    is_frame
)

__all__ = [
    'detect_categorical_columns', 
    'extract_target', 
    'fancier_downloader',
    'fillNaN', 
    'select_features',
    'fill_NaN',
    'validate_target_in',
]

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
    >>> from fusionlab.utils.base_utils import detect_categorical_columns
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
    `fusionlab.utils.validator` to ensure that the input `data` is a DataFrame. 
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
    >>> from fusionlab.utils.base_utils import fill_NaN
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
    fusionlab.core.array_manager.array_preserver:
        Preserves and restores array structures.
    fusionlab.core.array_manager.to_array:
        Converts input to a pandas-compatible array-like structure.
    fusionlab.core.checks.is_numeric_dtype:
        Checks if the array has numeric data types.
    fusionlab.utils.base_utils.fillNaN :
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
    >>> from fusionlab.utils.base_utils import select_features
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
    >>> from fusionlab.utils.base_utils import download_file
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
    >>> from fusionlab.utils.base_utils import fancier_downloader
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
    - :func:`fusionlab.utils.base_utils.check_file_exists` : Utility to check file
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
        from tqdm import tqdm  #NOQA
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
    >>> from fusionlab.utils.base_utils import move_file
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
        >>> from fusionlab.utils.base_utils import check_file_exists
        >>> package_name = 'fusionlab.datasets.data'  # Replace with your package name
        >>> file_name = 'h.h5'    # Replace with your file name

        >>> file_exists = check_file_exists(package_name, file_name)
        >>> print(f"File exists: {file_exists}")
    """

    import importlib.resources as pkg_resources
    return pkg_resources.is_resource(package, resource)



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

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    