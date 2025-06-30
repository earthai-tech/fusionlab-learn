# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio  <etanoyau@gmail.com>
# Source: Adapted from earthai-tech/gofast/core utilities
# This module is included in the FusionLab package, with modifications
# to fit FusionLab’s API and coding conventions.

"""
Core utilities for array and DataFrame manipulation in FusionLab,
including data reshaping, conversion, and sparse matrix handling.
Ensures consistent array-like formats across the package.

Adapted from the original fusionlab.core module.
"""

from __future__ import print_function, annotations
import re
import copy
import hashlib
import itertools
import warnings
import logging

import numpy as np
import pandas as pd

from ..api.types import (
    Any, Dict, Union, Series, Tuple, Optional, Set,
    _T, _F, ArrayLike, List, DataFrame, NDArray
)
from .utils import (
    is_iterable,
    _assert_all_types,
    sanitize_frame_cols,
    listing_items_format,
    smart_format,
    error_policy,
)
from .checks import (
    assert_ratio, is_in_if,
    str2columns,
    is_numeric_dtype,
    are_all_frames_valid,
    ensure_same_shape,
    validate_axis,
)

__all__ = [
    'convert_to_structured_format',
    'denormalize',
    'to_numeric_dtypes',
    'reshape',
    'squeeze_specific_dim',
    'split_list',
    'split_train_test',
    'split_train_test_by_id',
    'make_arr_consistent',
    'process_and_extract_data',
    'to_series_if',
    'test_set_check_id',
    'decode_sparse_data',
    'map_specific_columns',
    'reduce_dimensions',
    'smart_ts_detector',
    'extract_array_from',
    'drop_nan_in',
    'to_array',
    'to_arrays',
    'array_preserver',
    'return_if_preserver_failed',
]


def to_array(
    arr: Any,
    accept: Optional[str] = None,
    error: str = 'raise',
    force_conversion: bool = False,
    axis: int = 0,
    mode: str = "keep_origin",
    as_frame: bool=False, 
    verbose: int = 0,
    **kwargs
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Convert various array-like objects to the desired dimensionality.

    The ``to_array`` function standardizes different array-like inputs 
    (such as lists, tuples, NumPy arrays, Pandas Series, and DataFrames) 
    by converting them to a specified dimensionality. It provides flexibility 
    through parameters like ``accept``, ``force_conversion``, and 
    ``mode`` to ensure the output meets the user's requirements. Additionally, 
    the ``verbose`` parameter allows users to control the level of informational 
    messages during the conversion process.

    Parameters
    ----------
    arr : Any
        The input array-like object to be converted. Supported types include 
        lists, tuples, NumPy arrays, Pandas Series, and Pandas DataFrames.
    
    accept : str, optional
        Specifies the desired dimensionality of the output array. Accepted 
        values include:
        
        - ``'1d'``: Ensure the output is a 1-dimensional array.
        - ``'2d'``: Ensure the output is a 2-dimensional array.
        - ``'3d'``: Ensure the output is a 3-dimensional array.
        - ``'only_1d'``: Only accept 1-dimensional arrays without conversion.
        - ``'only_2d'``: Only accept 2-dimensional arrays without conversion.
        - ``'only_3d'``: Only accept 3-dimensional arrays without conversion.
        - ``'>3d'``: Accept arrays with more than 3 dimensions.
        - ``'only_>3d'``: Only accept arrays with more than 3 dimensions without conversion.
        
        If ``accept`` is ``None``, the function returns the input as-is 
        without any dimensionality enforcement.
    
    error : str, default='raise'
        Defines the behavior when the input array does not meet the 
        specified ``accept`` criteria. Options include:
        
        - ``'raise'``: Raise a ``ValueError`` when the input does not match 
          the expected dimensionality.
        - ``'warn'``: Issue a warning but return the input array as-is.
        - ``'ignore'``: Silently return the input array without any checks 
          or modifications.
    
    force_conversion : bool, default=False
        If ``True``, the function will attempt to automatically convert the 
        input array to match the desired dimensionality specified by 
        ``accept``. This includes reshaping, flattening, or expanding 
        dimensions as necessary.
    
    axis : int, default=0
        Specifies the axis along which to reshape the array when converting 
        to higher dimensions. For example, when converting a 1D array to a 
        2D array, ``axis=0`` will reshape it to ``(x, 1)`` and 
        ``axis=1`` will reshape it to ``(1, x)``.
    
    mode : str, default="keep_origin"
        Determines the operation mode for handling original data types. 
        Accepted values include:
        
        - ``"keep_origin"``: Retains the original data types (e.g., 
          keeps a Pandas DataFrame as is when converting to 2D).
        - ``"numpy_only"``: Forces all conversions to NumPy arrays, 
          disregarding original data types.
        
        When ``accept`` is ``'2d'`` or ``'only_2d'`` and a Pandas DataFrame 
        is passed, it will not be converted to a NumPy array if ``mode`` 
        is set to ``"keep_origin"``. Similarly, a Pandas Series will not be 
        converted to a NumPy array when ``accept`` is ``'1d'`` or 
        ``'only_1d'`` and ``mode`` is ``"keep_origin"``.
    
    as_frame : bool, default=False
        If ``True``, attempts to convert the array back to a Pandas 
        DataFrame or Series after reshaping. This is useful when the 
        user prefers to work with Pandas objects instead of NumPy arrays.
    
    verbose : int, default=0
        Controls the verbosity level of informational messages:
        
        - ``0``: No messages.
        - ``1``: Basic messages about conversion steps.
        - ``2``: Detailed messages including successful conversions.
        - ``3``: Debug-level messages with extensive details.
    
    **kwargs : dict
        Additional keyword arguments for future flexibility and to handle 
        specific cases as needed.
    
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        The converted array-like object adhering to the specified 
        dimensionality constraints. The output type depends on the 
        ``mode`` and the nature of the input.
    
    Examples
    --------
    >>> from fusionlab.core.array_manager import to_array
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Convert a list to a 2D NumPy array
    >>> list_input = [1, 2, 3]
    >>> array_2d = to_array(
    ...     list_input, 
    ...     accept='2d', 
    ...     force_conversion=True, 
    ...     axis=0, 
    ...     verbose=1
    ... )
    >>> print(array_2d)
    [[1]
     [2]
     [3]]
    
    >>> # Convert a Pandas Series to a 1D NumPy array
    >>> series_input = pd.Series([4, 5, 6])
    >>> array_1d = to_array(
    ...     series_input, 
    ...     accept='1d', 
    ...     mode='numpy_only', 
    ...     verbose=2
    ... )
    >>> print(array_1d)
    [4 5 6]
    
    >>> # Attempt to convert a 3D NumPy array to 2D without forcing conversion
    >>> np_array_3d = np.random.rand(2, 3, 4)
    >>> try:
    ...     array_2d_fail = to_array(
    ...         np_array_3d, 
    ...         accept='2d', 
    ...         force_conversion=False, 
    ...         error='warn', 
    ...         verbose=3
    ...     )
    ... except ValueError as e:
    ...     print(e)
    Input array has 3 dimensions, but at least 2 dimensions are required.
    
    >>> # Force conversion of a 3D array to 2D
    >>> array_2d_force = to_array(
    ...     np_array_3d, 
    ...     accept='2d', 
    ...     force_conversion=True, 
    ...     axis=1, 
    ...     verbose=1
    ... )
    >>> print(array_2d_force.shape)
    (2, 12)
    
    >>> # Convert a 1D NumPy array to a Pandas Series with as_frame=True
    >>> np_array_1d = np.array([7, 8, 9])
    >>> series_output = to_array(
    ...     np_array_1d, 
    ...     accept='1d', 
    ...     mode='keep_origin', 
    ...     as_frame=True, 
    ...     verbose=2
    ... )
    >>> print(series_output)
    0    7
    1    8
    2    9
    dtype: int64
    
    Notes
    -----
    The ``to_array`` function is essential for standardizing data structures 
    before analysis or modeling. By accommodating various input types and 
    providing flexible conversion options, it ensures that downstream processes 
    receive data in the expected format. The ``force_conversion`` parameter 
    allows users to override default behaviors, enabling customized data handling 
    strategies. Additionally, the ``mode`` parameter offers control over 
    whether to preserve original data types or enforce conversions to NumPy arrays, 
    enhancing the function's adaptability to different use cases.
    
    When ``as_frame`` is set to ``True``, the function attempts to convert 
    NumPy arrays back to Pandas DataFrames or Series, facilitating seamless 
    integration with Pandas-centric workflows. Users should be cautious when 
    forcing conversions, as automatic reshaping may lead to unintended data 
    transformations.
    
    The ``verbose`` parameter is instrumental for debugging and monitoring 
    the conversion process. Higher verbosity levels provide more granular 
    insights, which can be invaluable when troubleshooting complex data structures.
    

    .. math::
        \text{Converted Array} = 
        \begin{cases}
            \text{arr.flatten()} & \text{if converting to 1D} \\
            \text{arr.reshape(-1, 1)} & \text{if converting to 2D along axis 0} \\
            \text{arr.reshape(1, -1)} & \text{if converting to 2D along axis 1} \\
            \text{arr.reshape(arr.shape[0], -1, 1)} & \text{if converting to 3D} \\
            \text{arr.reshape(*arr.shape, 1, \dots)} & \text{if converting to } >3\text{D}
        \end{cases}
        
    See Also
    --------
    ``to_arrays`` : Converts multiple array-like objects to desired dimensionality 
                   using ``to_array``.
    ``PandasDataHandlers`` : Manages Pandas-based data parsing and writing functions.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. In *Proceedings of the 9th Python in Science Conference*, 
           51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. *Nature*, 585(7825), 357-362.
    .. [3] Pandas Development Team. (2023). *pandas documentation*. 
           https://pandas.pydata.org/pandas-docs/stable/
    """
    # Helper function to determine the number of dimensions
    def _get_ndim(obj: Any) -> int:
        """Return the number of dimensions of the input object."""
        if isinstance(obj, pd.DataFrame):
            return 2
        elif isinstance(obj, pd.Series):
            return 1
        elif isinstance(obj, np.ndarray):
            return obj.ndim
        else:
            return 1  # Default to 1 for list, tuple, etc.

    # Helper function to print messages based on verbosity level
    def _verbose_print(message: str, level: int = 1):
        """Print messages based on the verbosity level."""
        if verbose >= level:
            print(message)

    # Helper function to handle errors
    def _handle_error(message: str):
        """Handle errors based on the 'error' parameter."""
        if error == 'raise':
            raise ValueError(message)
        elif error == 'warn':
            warnings.warn(message)
        elif error == 'ignore':
            pass
        else:
            raise ValueError(
                f"Invalid error handling mode: '{error}'. "
                "Choose from 'raise', 'warn', or 'ignore'."
            )

    # Helper function to convert to numpy array
    def _convert_to_numpy(obj: Any) -> np.ndarray:
        """Convert list or tuple to NumPy array."""
        if isinstance(obj, (list, tuple)):
            _verbose_print("Converting list/tuple to NumPy array.", level=1)
            return np.array(obj)
        return obj

    # Helper function to reshape NumPy array
    def _reshape_numpy(arr_np: np.ndarray, target_dim: int) -> np.ndarray:
        """Reshape NumPy array to the target dimension."""
        try:
            if target_dim == 1:
                return arr_np.flatten()
            elif target_dim == 2:
                return arr_np.reshape(-1, 1) if axis == 0 else arr_np.reshape(1, -1)
            elif target_dim == 3:
                return arr_np.reshape(arr_np.shape[0], -1, 1)
            elif target_dim > 3:
                extra_dims = target_dim - arr_np.ndim
                return arr_np.reshape(*arr_np.shape, *(1,) * extra_dims)
        except Exception as e:
            _handle_error(f"Failed to reshape array to {target_dim}D. Error: {e}")
        return arr_np
    
    # Helper function to Pandas Series to dataframe if target_dim is 2d 
    # else return Numpy array for reshaping in higher dimension. 
    def _reshape_series(series: pd.Series, target_dim: int, axis: int = 0):
        """
        Reshape a Pandas Series into a DataFrame or NumPy array based 
        on the target dimension.
    
        Parameters:
        - series (pd.Series): The Pandas Series to reshape.
        - target_dim (int): The desired number of dimensions after reshaping.
        - axis (int, optional): The axis along which to reshape. Defaults to 0.
    
        Returns:
        - pd.DataFrame or np.ndarray: The reshaped Series as a DataFrame 
         if target_dim is 2,
          or as a NumPy array for higher dimensions.
        """
        if isinstance (series, pd.Series): 
            if axis == 0:
                if target_dim == 2:
                    _verbose_print(
                        "Reshaping Series to DataFrame.", level=2
                    )
                    return series.to_frame() # Convert Series to DataFrame
                
                elif target_dim > 2:
                    _verbose_print(
                        "Series detected. Converting to NumPy array for"
                        " dimensions exceeding 2.", level=2
                    )
                    raise # force to ball back to numpy array conversion 
                else: # for 1d dimension, no conversion is performed. 
                    return series 
                
            elif axis == 1:
                _verbose_print(
                    "Series detected with conversion axis=1. Conversion to"
                    " DataFrame is only supported with axis=0. "
                    "Falling back to NumPy reshaping.", level=2
                )
                raise # Fallback to numpy array conversion.
        else: 
            raise 
    
    # Convert list or tuple to NumPy array if necessary
    arr = _convert_to_numpy(arr)
    
    # keep original types for futher conversion 
    collected = array_preserver(arr, action='collect')
   
    # Handle 'keep_origin' operation mode
    if mode == "keep_origin":
        if accept in ['2d', 'only_2d'] and isinstance(arr, pd.DataFrame):
            _verbose_print("Keeping original Pandas DataFrame as it is 2D.", level=2)
            return arr
        elif accept in ['1d', 'only_1d'] and isinstance(arr, pd.Series):
            _verbose_print("Keeping original Pandas Series as it is 1D.", level=2)
            return arr

    # Determine current number of dimensions
    current_ndim = _get_ndim(arr)
    _verbose_print(f"Current number of dimensions: {current_ndim}", level=3)

    # Define acceptable dimensions based on 'accept' parameter
    if accept:
        # Mapping for 'accept' values to desired dimensions
        dim_map = {
            '1d': 1,
            '2d': 2,
            '3d': 3,
            '>3d': 4  # Any dimension greater than 3
        }

        # Check for 'only_' prefix to enforce strict acceptance
        only = False
        if accept.startswith('only_'):
            only = True
            accept_dim = accept.split('_', 1)[1]
        elif accept.startswith('>only_'):
            # Handle cases like 'only_>3d'
            only = True
            accept_dim = accept.split('_', 1)[1]
        else:
            accept_dim = accept

        # Determine the required dimension
        min_dim = dim_map.get(accept_dim, None)

        # Handle cases where 'accept' specifies dimensions greater than a threshold
        if accept_dim.startswith('>'):
            min_dim = int(accept_dim.strip('>d'))
            condition = current_ndim > min_dim
            if only and not condition:
                message = (
                    f"Input array has {current_ndim} dimensions, but only arrays "
                    f"with more than {min_dim} dimensions are accepted."
                )
                _handle_error(message)
        else:
            condition = current_ndim == min_dim
            if only:
                if not condition:
                    message = (
                        f"Input array has {current_ndim} dimensions, but only "
                        f"{min_dim}-dimensional arrays are accepted."
                    )
                    _handle_error(message)
            else:
                
                # Allow dimensions greater than or equal to the minimum 
                # required when 'only_' is not specified
                if min_dim and current_ndim < min_dim:
                    if force_conversion:
                        # Attempt to reshape using _reshape_series
                        # for intelligent handling
                        _verbose_print(
                            f"Attempting to reshape array to {accept_dim}"
                            " using _reshape_series.", level=2
                        )
                        try:
                            arr = _reshape_series(
                                arr, target_dim=dim_map[accept_dim], axis=axis)
                            current_ndim = _get_ndim(arr)
                            _verbose_print(
                                "Reshaping successful. Current dimensions:"
                                f" {current_ndim}D.", level=2
                            )
                        except Exception as e:
                            if isinstance (arr, pd.Series) :
                                arr= arr.values 
                            _verbose_print(
                                f"Reshaping with _reshape_series failed: {e}."
                                " Falling back to NumPy reshaping.", level=2
                            )
                            arr = _reshape_numpy(arr, dim_map[accept_dim])
                            current_ndim = _get_ndim(arr)
                    else:
                        error_message = (
                            f"Input array has {current_ndim} dimension(s), but at least "
                            f"{min_dim} dimension(s) are required."
                        )
                        _handle_error(error_message)

    # Handle conversion based on 'accept' parameter
    if accept:
        # Convert Pandas DataFrame to NumPy array if needed
        if isinstance(arr, pd.DataFrame):
            arr = arr.values
            _verbose_print("Converted Pandas DataFrame to NumPy array.", level=2)

        # Convert Pandas Series to NumPy array if needed
        elif isinstance(arr, pd.Series):
            arr = arr.values
            _verbose_print("Converted Pandas Series to NumPy array.", level=2)

        # Handle NumPy array dimensionality adjustments
        if isinstance(arr, np.ndarray):
            target_dim =min_dim 
            # if only: 
            #     target_dim = dim_map.get(accept.split('_')[1], None)
            # else : 
            #     target_dim = dim_map.get(accept.split('_')[0], None)
           
            if target_dim:
                if accept in ['1d', 'only_1d']:
                    if arr.ndim == 2:
                        if arr.shape[0] == 1 or arr.shape[1] == 1:
                            arr = arr.flatten()
                            _verbose_print("Flattened 2D NumPy array to 1D.", 
                                           level=3)
                        elif force_conversion:
                            arr = arr.ravel()
                            _verbose_print("Raveled NumPy array to 1D.", 
                                           level=3)
                    elif arr.ndim > 1 and force_conversion:
                        arr = arr.flatten()
                        _verbose_print(
                            "Flattened higher-dimensional NumPy array to 1D.",
                            level=3)

                elif accept in ['2d', 'only_2d']:
                    if arr.ndim == 1 and force_conversion:
                        arr = arr.reshape(-1, 1) if axis == 0 else arr.reshape(1, -1)
                        _verbose_print(
                            f"Reshaped 1D NumPy array to 2D with shape {arr.shape}.",
                            level=3
                        )

                elif accept in ['3d', 'only_3d']:
                    if arr.ndim < 3 and force_conversion:
                        arr = _reshape_numpy(arr, 3)
                        _verbose_print(
                            f"Reshaped array to 3D with shape {arr.shape}.",
                            level=3)

                elif accept.startswith('>'):
                    # Handle greater than dimensions
                    required_dim = int(accept.strip('>d'))
                    if arr.ndim <= required_dim and force_conversion:
                        arr = _reshape_numpy(arr, required_dim + 1)
                        _verbose_print(
                            f"Reshaped array to exceed {required_dim}"
                            f" dimensions with shape {arr.shape}.",
                            level=3
                        )

    # Final validation of dimensions
    if accept:
        final_ndim = _get_ndim(arr)
        expected_dim = min_dim # dim_map.get(accept.split('_')[0], None)
        
        if accept.startswith('>'):
            min_dim = int(accept.strip('>d'))
            if not final_ndim > min_dim:
                message = (
                    f"Final array has {final_ndim} dimensions, which does not exceed "
                    f"{min_dim} dimensions as required by `accept='{accept}'`."
                )
                _handle_error(message)
        else:
            if 'only_' in accept:
                if final_ndim != expected_dim:
                    message = (
                        f"Final array has {final_ndim} dimensions, expected exactly "
                        f"{expected_dim} dimensions as specified by `accept='{accept}'`."
                    )
                    _handle_error(message)
            else:
                if final_ndim < expected_dim:
                    message = (
                        f"Final array has {final_ndim} dimensions, which is less than "
                        f"the required {expected_dim} dimensions."
                    )
                    _handle_error(message)

    # Maintain original type if ops_mode is 'keep_origin' and no conversion was done
    if mode == "keep_origin":
        collected['processed'] = [arr]
        arr = array_preserver(collected, action='restore', solo_return= True)
    
    if isinstance(arr, np.ndarray) and as_frame: 
        # Then try to convert array to frame 
        if accept in ['2d', 'only_2d'] and arr.ndim == 2:
            # Attempt to keep as DataFrame if possible
            try:
                arr = pd.DataFrame(arr)
                _verbose_print("Converted NumPy array back to Pandas DataFrame.", 
                               level=2)
            except Exception as e:
                _handle_error(
                    f"Failed to convert NumPy array to DataFrame. Error: {e}")
        elif accept in ['1d', 'only_1d'] and arr.ndim == 1:
            # Attempt to keep as Series if possible
            try:
                arr = pd.Series(arr)
                _verbose_print(
                    "Converted NumPy array back to Pandas Series.", level=2)
            except Exception as e:
                _handle_error(
                    f"Failed to convert NumPy array to Series. Error: {e}")

    return arr

def to_arrays(
    *arrays: Any,
    accept: Optional[str] = None,
    error: str = 'raise',
    force_conversion: bool = False,
    axis: int = 0,
    verbose: int = 0,
    mode: str = "keep_origin",
    as_frame: bool=False, 
    **kwargs
) -> Tuple[Union[np.ndarray, pd.Series, pd.DataFrame], ...]:
    """
    Convert multiple array-like objects to desired dimensionality 
    using ``to_array``.

    The `to_arrays` function processes each input array using the 
    ``to_array`` function, ensuring that all arrays conform to the 
    specified dimensionality requirements. This approach promotes 
    consistency and efficiency when handling batches of data, making 
    it a versatile tool for data preprocessing within the Gofast package.

    Parameters
    ----------
    *arrays : Any
        Variable number of array-like objects to be converted. Supported 
        types include lists, tuples, NumPy arrays, Pandas Series, and 
        Pandas DataFrames. Each array is processed individually to 
        match the specified dimensionality.
    
    accept : str, optional
        Specifies the desired dimensionality of the output arrays. 
        Accepted values include:
        
        - ``'1d'``: Ensure the output is a 1-dimensional array.
        - ``'2d'``: Ensure the output is a 2-dimensional array.
        - ``'3d'``: Ensure the output is a 3-dimensional array.
        - ``'only_1d'``: Only accept 1-dimensional arrays without conversion.
        - ``'only_2d'``: Only accept 2-dimensional arrays without conversion.
        - ``'only_3d'``: Only accept 3-dimensional arrays without conversion.
        - ``'>3d'``: Accept arrays with more than 3 dimensions.
        - ``'only_>3d'``: Only accept arrays with more than 3 dimensions without 
          conversion.
        
        If ``accept`` is ``None``, the function returns the inputs as-is 
        without any dimensionality enforcement.
    
    error : str, default='raise'
        Defines the behavior when an input array does not meet the 
        specified ``accept`` criteria. Options include:
        
        - ``'raise'``: Raise a `ValueError` when the input does not match the 
          expected dimensionality.
        - ``'warn'``: Issue a warning but append the original array without 
          conversion.
        - ``'ignore'``: Silently append the original array without any checks 
          or modifications.
    
    force_conversion : bool, default=False
        If ``True``, the function will attempt to automatically convert the 
        input arrays to match the desired dimensionality specified by 
        ``accept``. This includes reshaping, flattening, or expanding 
        dimensions as necessary.
    
    axis : int, default=0
        Specifies the axis along which to reshape the array when converting 
        to higher dimensions. For example, when converting a 1D array to a 
        2D array, ``axis=0`` will reshape it to ``(x, 1)`` and 
        ``axis=1`` will reshape it to ``(1, x)``.
    
    verbose : int, default=0
        Controls the verbosity level of the function's output. Accepts integer 
        values from 1 to 3, where:
        
        - ``1``: Minimal output, only essential messages.
        - ``2``: Moderate output, including warnings and important information.
        - ``3``: High verbosity, detailed debugging information.
        
        Setting ``verbose`` to a higher value can aid in debugging by providing 
        step-by-step insights into the conversion process.
    
    mode : str, default="keep_origin"
        Determines the operation mode for handling original data types during 
        conversion. Accepted values include:
        
        - ``"keep_origin"``: Retains the original data types (e.g., 
          Pandas DataFrame remains a DataFrame). Conversion to NumPy arrays 
          is avoided unless absolutely necessary.
        - ``"numpy_only"``: Forces all inputs to be converted to NumPy arrays, 
          regardless of their original types.
        
        **Behavior Details:**
        
        - If ``accept`` is ``'2d'`` or ``'only_2d'`` and ``mode`` is 
          ``"keep_origin"``, Pandas DataFrames remain as DataFrames without 
          conversion to NumPy arrays.
        - If ``accept`` is ``'1d'`` or ``'only_1d'`` and a Pandas Series is 
          passed, it remains a Series without conversion to a NumPy array.
        - When ``mode`` is ``"keep_origin"`` and ``force_conversion`` 
          is enabled, the function attempts conversions without altering the 
          original data types unless necessary. If conversion isn't possible 
          without changing the type, it handles the situation based on the 
          ``error`` parameter.
    
    **kwargs : dict
        Additional keyword arguments passed to the underlying ``to_array`` 
        function. This allows for extended flexibility and handling of 
        specific cases as needed.

    Returns
    -------
    Tuple[Union[np.ndarray, pd.Series, pd.DataFrame], ...]
        A tuple containing the converted array-like objects, each adhering 
        to the specified dimensionality constraints. The order of the 
        returned arrays corresponds to the order of the input arrays.

    .. math::
        \text{Converted Arrays} =\\
            \left( \text{to\_array}(arr_1), \text{to\_array}(arr_2),\\
                  \ldots, \text{to\_array}(arr_n) \right)

    Examples
    --------
    >>> from fusionlab.core.array_manager import to_arrays
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Define multiple array-like inputs
    >>> list_input = [1, 2, 3]
    >>> tuple_input = (4, 5, 6)
    >>> np_array = np.array([7, 8, 9])
    >>> pd_series = pd.Series([10, 11, 12])
    >>> # Convert all inputs to 2D arrays
    >>> converted = to_arrays(
    ...     list_input, tuple_input, np_array, pd_series,
    ...     accept='2d', force_conversion=True, axis=0, verbose=2, mode='numpy_only'
    ... )
    Converted list/tuple to NumPy array.
    Reshaped 1D NumPy array to 2D with shape (3, 1).
    Reshaped 1D NumPy array to 2D with shape (3, 1).
    Reshaped 1D NumPy array to 2D with shape (3, 1).
    >>> for arr in converted:
    ...     print(arr)
    [[1]
     [2]
     [3]]
    [[4]
     [5]
     [6]]
    [[7]
     [8]
     [9]]
    [[10]
     [11]
     [12]]
    
    >>> # Convert multiple arrays to 1D, raising errors for mismatches
    >>> converted_1d = to_arrays(
    ...     np_array, pd_series, list_input,
    ...     accept='1d', error='raise', verbose=1, mode='keep_origin'
    ... )
    >>> for arr in converted_1d:
    ...     print(arr)
    [7 8 9]
    0    10
    1    11
    2    12
    dtype: int64
    [1, 2, 3]
    
    >>> # Attempt to convert arrays to 3D without forcing conversion
    >>> converted_3d = to_arrays(
    ...     np_array, list_input,
    ...     accept='3d', error='warn', verbose=3, mode='keep_origin'
    ... )
    Converted list/tuple to NumPy array.
    Input array has 1 dimensions, but at least 3 dimensions are required.
    >>> print(converted_3d)
    [7 8 9]
    [1 2 3]

    Notes
    -----
    The `to_arrays` function ensures that all input arrays are processed 
    uniformly, adhering to the specified dimensionality rules. By leveraging 
    the ``to_array`` function, it maintains consistency and reduces redundancy 
    when handling multiple arrays. This function is particularly useful in 
    scenarios where batch processing of data is required, such as in machine 
    learning pipelines or data preprocessing workflows.

    When using the ``force_conversion`` parameter, the function attempts to 
    reshape or flatten arrays to meet the desired dimensionality. However, 
    this automatic adjustment may lead to unintended data transformations 
    if not used cautiously. It is recommended to verify the shape of the 
    output arrays, especially when working with complex data structures.

    The ``error`` parameter provides flexibility in how the function responds 
    to mismatched dimensions. Using ``'raise'`` ensures strict adherence 
    to the specified ``accept`` criteria, while ``'warn'`` and ``'ignore'`` 
    offer more lenient handling, allowing the processing to continue even 
    when some arrays do not meet the requirements.

    The ``ops_mode`` parameter adds an additional layer of control over 
    how original data types are handled during conversion. By setting 
    ``ops_mode`` to ``"keep_origin"``, the function preserves the original 
    types of inputs like Pandas DataFrames and Series, avoiding unnecessary 
    conversions to NumPy arrays. Conversely, setting it to ``"numpy_only"`` 
    enforces conversion to NumPy arrays, providing uniformity across all 
    processed arrays.

    See Also
    --------
    ``to_array`` : Converts a single array-like object to the desired dimensionality.
    ``PandasDataHandlers`` : Manages Pandas-based data parsing and writing functions.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. In *Proceedings of the 9th Python in Science Conference*, 
           51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. *Nature*, 585(7825), 357-362.
    .. [3] Pandas Development Team. (2023). *pandas documentation*. 
           https://pandas.pydata.org/pandas-docs/stable/
    """

    # Initialize a list to store the converted arrays
    converted_arrays = []
    
    # Iterate over each input array
    for idx, arr in enumerate(arrays):
        try:
            # Convert the current array using the to_array function
            converted = to_array(
                arr,
                accept=accept,
                error=error,
                force_conversion=force_conversion,
                axis=axis,
                mode=mode,
                **kwargs
            )
            # Append the converted array to the list
            converted_arrays.append(converted)
        except Exception as e:
            # Handle exceptions based on the 'error' parameter
            if error == 'raise':
                # Raise the exception to halt execution
                raise ValueError(
                    f"Error converting array at position {idx}: {e}"
                ) from e
            elif error == 'warn':
                # Issue a warning and append the original array
                warnings.warn(
                    f"Warning converting array at position {idx}: {e}. "
                    "Appending the original array without conversion."
                )
                converted_arrays.append(arr)
            elif error == 'ignore':
                # Silently append the original array without conversion
                converted_arrays.append(arr)
            else:
                # If an unknown error mode is specified, raise a ValueError
                raise ValueError(
                    f"Invalid error handling mode: '{error}'. "
                    "Choose from 'raise', 'warn', or 'ignore'."
                )
    
    # Return the converted arrays as a tuple to maintain immutability
    return tuple(converted_arrays)

def smart_ts_detector(
    df,
    dt_col,
    return_types='format',
    to_datetime=None,
    error='raise',
    as_index=False, 
    verbose=0
):
    r"""
    Intelligently determine the temporal resolution or format of a 
    given date/time column in a DataFrame, and optionally convert 
    it to a proper datetime representation. The function can detect 
    if `<dt_col>` is already a datetime-like column, infer time 
    frequency if possible, or guess the temporal granularity from 
    numeric values (e.g., treating them as years, months, weeks, 
    minutes, etc.) when no datetime format is found.

    Let the date column be represented by :math:`d = \{d_1, d_2, 
    \ldots, d_n\}`. The goal is to determine the format type 
    :math:`f(d)`, such as `year`, `month`, `week`, `day`, `minute`, 
    or `second`, based on the range and nature of these values. 
    Formally, if `d` is datetime-like, we attempt to infer frequency 
    using heuristics. If numeric, we decide the format by the value 
    ranges (e.g., values = 12 might suggest months, values = 52 
    might suggest weeks, etc.). If `to_datetime` is provided, the 
    function attempts to convert the column accordingly.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the `<dt_col>`. This column is 
        expected to either be datetime-like, numeric, or convertible 
        to a known temporal format.
    dt_col : str
        The name of the column in `df` representing date or 
        time-related data. If not found, handling depends on `<error>`.
    return_types: {'format', 'dt_col', 'df'}, optional
        Determines the return value:
        
        - ``'format'``: Returns a string representing the inferred 
          date/time format (e.g., 'years', 'months', 'weeks', 'datetime').
        - ``'dt_col'``: Returns the transformed or original date column.
        - ``'df'``: Returns the entire DataFrame with the `<dt_col>` 
          modified if necessary.
          
    to_datetime : {None, 'auto', 'Y', 'M', 'W', 'D', 'H', 'min', 's'}, optional
    
        Controls how the column is converted if not already datetime:
        - None: No conversion, only format detection.
        - 'auto': Automatically infer and convert based on rules.
        - Explicit codes like 'Y', 'M', 'W', 'min', 's' attempt to 
          convert according to those units.
    error : {'raise', 'ignore', 'warn'}, optional
        Defines behavior if `<dt_col>` is not found or cannot be 
        interpreted:
        
        - 'raise': Raise a ValueError.
        - 'ignore': Return without modification or raise.
        - 'warn': Issue a warning and proceed (if possible).
    as_index: bool, 
       Whether to return the entire dataset and set as index the `dt_col`. This 
       is done when `return_types='df'. 
       
    verbose : int, optional
        Verbosity level for logging:
        
        - 0: No output.
        - 1: Basic info.
        - 2: More details on reasoning steps.
        - 3: Very detailed internal states.

    Returns
    -------
    str or pandas.Series or pandas.DataFrame
        Depending on `<return_types>`:
        
        - If `'format'`, returns a string like `'years'`, `'months'`,
          `'weeks'`, `'datetime'`, etc.
        - If `'dt_col'`, returns the possibly converted date column.
        - If `'df'`, returns the entire DataFrame with `<dt_col>` 
          modified accordingly.

    Notes
    -----
    If `<dt_col>` is already a datetime-like column (np.datetime64),
    this function attempts to infer frequency using `pd.infer_freq` 
    or heuristics. If `<to_datetime>` is 'auto', it tries to guess 
    the best format. If numeric, the function deduces format based 
    on value ranges. If strings or other types are found, it attempts 
    conversion if `<to_datetime>` is specified or, otherwise, 
    handles them according to `<error>`.

    Handling missing or non-convertible values depends on `<error>`. 
    If 'raise', errors are raised when conversion fails or `<dt_col>` 
    is absent. If 'warn', issues a warning. If 'ignore', quietly 
    returns what is possible.

    Examples
    --------
    >>> from fusionlab.core.array_manager import smart_ts_detector
    >>> import pandas as pd
    >>> df = pd.DataFrame({'year': [2020, 2021, 2022]})
    >>> # Detect format from year-like integers:
    >>> fmt = smart_ts_detector(df, 'year', return_types='format')
    >>> print(fmt)
    'years'

    >>> # Convert to datetime assuming years:
    >>> df2 = smart_ts_detector(df, 'year', return_types='df',
    ...                         to_datetime='Y')
    >>> print(df2['year'])

    See Also
    --------
    pandas.to_datetime : For conversion of values to datetime.
    pandas.infer_freq : For inferring frequency of datetime series.

    References
    ----------
    .. [1] McKinney, W. "Data Structures for Statistical Computing
           in Python." Proceedings of the 9th Python in Science Conf.
           (2010): 51–56.
    """

    # Helper function to raise/warn/ignore errors
    def handle_error(msg, e=error):
        if e == 'raise':
            raise ValueError(msg)
        elif e == 'warn':
            warnings.warn(msg)
        # if ignore, do nothing
        
    are_all_frames_valid(df)
    # Check if dt_col in df
    if dt_col not in df.columns:
        handle_error(f"Column {dt_col!r} not found in DataFrame.", e='raise')
        # If ignoring, just return None or df as is
        if return_types=='df':
            return df
        elif return_types=='dt_col':
            return df[dt_col] if dt_col in df else None
        else:
            return None

    series = df[dt_col]
    
    # validate to_datetime format is passed
    valid_formats ={'auto', 'Y', 'M', 'W', 'D', 'H', 'min', 's'}
    
    if to_datetime is not None and to_datetime not in valid_formats: 
        raise ValueError(
           f"Invalid `to_datetime` format '{to_datetime}'."
            f" Expect one of {smart_format(valid_formats,'or')}"
        )
    # Check if already datetime
    if np.issubdtype(series.dtype, np.datetime64):
        # already datetime
        # detect resolution
        # We try to infer what kind of resolution:
        # We'll check the range and differences
        # freqs = series.dropna().diff().dropna()
        # Just a heuristic to guess resolution
        # If diffs are large (in days), consider daily or yearly, etc.
        # But let's simplify: if all values have same year and differ
        #  in month -> months
        # If all differ at daily scale -> daily,
        # If differ at weekly scale -> weekly
        # If differ by year boundaries -> year
        # This could be complex. Let's just return 'datetime' if no 
        # to_datetime specified unless we are forced to guess granularity.

        # If to_datetime is not None and is not 'auto', try converting
        # but we already have datetime, so no need actually
        if to_datetime is None or to_datetime=='auto':
            # Just guess a rough format by checking frequency
            # Let's convert to period and see the smallest freq
            try:
                inferred = pd.infer_freq(series.dropna().sort_values())
                # infer_freq might return 'M','W','A-DEC' for annual, etc.
                # We'll map this to a format
                if inferred is None:
                    # no freq inferred, just call it 'datetime'
                    dt_format = 'datetime'
                else:
                    # map freq code to something nicer
                    # For simplicity just return inferred
                    dt_format = inferred
            except:
                dt_format = 'datetime'
        else:
            # if to_datetime is explicitly something like 'Y','M','W' 
            # we won't reconvert since already datetime
            # Just set dt_format to that
            dt_format = to_datetime
    else:
        # Not datetime
        # Check if numeric
        if np.issubdtype(series.dtype, np.number):
            # numeric
            # Decide format based on values
            sdrop = series.dropna()
            if sdrop.empty:
                # empty? can't infer
                dt_format = 'unknown'
            else:
                min_val = sdrop.min()
                max_val = sdrop.max()

                # If to_datetime='auto', we guess from range
                # if to_datetime is None, we just guess format and not convert
                # If to_datetime given explicitly, just convert to that

                if to_datetime is None or to_datetime=='auto':
                    # guess
                    # if max_val <=12 and min_val>=1 -> months
                    # if max_val<=52 -> weeks
                    # if looks like a year range: (e.g. between 1900 and 2100)
                    # if max_val<=60 -> could be minutes or seconds
                    # guess priority: 
                    # If all values less or equal 12 and >=1 -> months
                    if max_val<=12 and min_val>=1:
                        dt_format='months'
                    elif max_val<=52 and min_val>=1:
                        dt_format='weeks'
                    elif max_val>1900 and max_val<2100:
                        dt_format='years'
                    elif max_val<=60:
                        # could be minutes or seconds, let's say 'minutes'
                        dt_format='minutes'
                    else:
                        dt_format='unknown'
                else:
                    # to_datetime given explicitly
                    dt_format = to_datetime
        else:
            # Not numeric, not datetime -> probably string or something
            # Try convert to datetime if to_datetime is given
            # else error
            if to_datetime is None:
                handle_error(
                    "dt_col is not datetime or numeric and to_datetime is None.")
                dt_format='unknown'
            else:
                dt_format=to_datetime

        # If we have dt_format and to_datetime='auto' or explicit format,
        # convert if possible
        if to_datetime is not None and to_datetime=='auto':
            # Try convert based on dt_format
            # Let's implement a simple converter:
            # if dt_format='years' and numeric: convert as year: pd.to_datetime(series, format='%Y')
            # if dt_format='months' and numeric: assume year=2000 and that months
            # represent month of that year
            # if dt_format='weeks' and numeric: treat as week numbers in a given year?
            # if dt_format='minutes' and numeric: treat as minutes from a base date?

            # For simplicity:
            # if years: format='%Y'
            # if months: treat integers as month numbers in year 2000
            # if weeks: treat as weeks in year 2000
            # if minutes: treat as minutes from '2000-01-01'
            # if seconds: likewise

            try:
                if dt_format=='years':
                    # year as int: e.g. 2020 -> '2020'
                    series_dt = pd.to_datetime(
                        series.astype(int).astype(str), format='%Y', errors='coerce')
                    if series_dt.isna().any():
                        handle_error("Cannot convert to years datetime from given values.")
                    series = series_dt
                elif dt_format=='months':
                    # month as int 1-12
                    # create a dummy date: '2000-{month}-01'
                    series_dt = pd.to_datetime(
                        '2000-'+series.astype(int).astype(str)+'-01',
                        format='%Y-%m-%d', errors='coerce')
                    if series_dt.isna().any():
                        handle_error("Cannot convert to months datetime from given values.")
                    series = series_dt
                elif dt_format=='weeks':
                    # weeks as int 1-52
                    # interpret as week number in year 2000 starting from first Monday of 2000
                    # This is trickier
                    # Let's assume '2000' + week number *7 days
                    base = pd.Timestamp('2000-01-01')
                    # series as int *7 days
                    series_dt = [base + pd.Timedelta(weeks=int(w)) for w in series]
                    series = pd.to_datetime(series_dt)
                elif dt_format=='minutes':
                    # interpret as minutes since 2000-01-01
                    base = pd.Timestamp('2000-01-01')
                    series_dt = [base + pd.Timedelta(minutes=float(m)) for m in series]
                    series = pd.to_datetime(series_dt)
                elif dt_format=='seconds':
                    # same logic as minutes but seconds
                    base = pd.Timestamp('2000-01-01')
                    series_dt = [base + pd.Timedelta(seconds=float(s)) for s in series]
                    series = pd.to_datetime(series_dt)
                else:
                    # unknown format, try to_datetime directly
                    series_dt = pd.to_datetime(series, errors='coerce')
                    if series_dt.isna().any():
                        handle_error("Cannot convert to datetime with given format guess.")
                    series = series_dt
            except Exception as e:
                handle_error(f"Conversion failed: {e}")
    # if return_types is 'format', return dt_format
    if return_types=='format':
        return dt_format
    elif return_types=='dt_col':
        return series
    elif return_types=='df':
        df = df.copy()
        df[dt_col] = series
        if as_index: 
            df.set_index (dt_col, inplace=True)
            
        return df
    else:
        return dt_format

def extract_array_from(
    df,
    *col_names,
    handle_unknown='passthrough',
    asarray=False,
    check_size=False,
    ravel=None,
    error='raise'
):
    r"""
    Extract one or multiple arrays from a pandas DataFrame based on 
    specified column names or arrays. This function provides flexible 
    handling of column names, nested lists of column names, and direct 
    array-like objects. By default, it returns the extracted data as 
    DataFrame, Series, or the original objects unless `asarray` is True, 
    in which case arrays are returned as NumPy arrays. 
    
    The mathematical essence of extraction involves indexing the 
    DataFrame or passing through objects. Let:
    
    .. math::
       X = \text{DataFrame}
    
    and suppose we have a set of column names :math:`C = \{c_1, \ldots, c_k\}` 
    or nested structures. The extraction aims to produce:
    
    .. math::
       X_{\text{subset}} = X[:, C_{\text{valid}}]
    
    for valid column names. For array-like inputs not corresponding to 
    columns, behavior depends on `<handle_unknown>`.
    
    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame from which columns are extracted. If `names` 
        includes references to columns not in `df`, handling depends 
        on `<handle_unknown>` and `<error>`.
    *col_names : str, list of str, or array-like
        The column names or arrays to extract. This can be:
        
        - A single string `name` representing a single column.
        - A list of strings `[col1, col2, ...]` representing multiple 
          columns.
        - An array-like object (non-string) that is passed through if 
          `<handle_unknown>` is `'passthrough'`.
        - A nested combination like `[col1, [col2, col3], col4, ...]`.
        
        If `<asarray>` is False, columns are returned as DataFrame or 
        Series. If `<asarray>` is True, they are converted to NumPy 
        arrays.
    handle_unknown : {'passthrough', None}, optional
        Controls how to handle unknown columns or non-column arrays.
        
        - `'passthrough'`: Return the unknown entries as-is (if arrays) 
          or skip missing columns.
        - `None`: Ignore unknown columns silently. If `<error>` is 
          `'raise'`, an error is raised for missing columns.
    asarray : bool, optional
        If True, convert extracted results to NumPy arrays. If False, 
        return them as DataFrame/Series or as provided arrays.
    check_size : bool, optional
        If True, checks that all extracted arrays have the same length. 
        If a mismatch is found, a ValueError is raised.
    ravel : {None, '1d', '2d', 'all'}, optional
        Controls reshaping of arrays if `<asarray>` is True:
        
        - `None`: No reshaping performed.
        - `'1d'`: If an array has shape (n,1), it is raveled to (n,).
        - `'2d'`: If an array has shape (n,), it is reshaped to (n,1).
        - `'all'` or `'*'`: Applies '1d' logic to (n,1) arrays. 
          Leaves (n,) as is.
    error : {'raise', ...}, optional
        If `'raise'`, a ValueError is raised if expected columns are 
        missing. If otherwise (not specified), silently skip or 
        passthrough.

    Returns
    -------
    object or list
        The extracted arrays. If multiple items are extracted, a list 
        is returned. If a single item is extracted, that item is 
        returned directly. If no items are extracted, `None` is 
        returned.
    
    Notes
    -----
    When `<names>` contain nested lists, each element is resolved. 
    Numeric columns are extracted directly. Non-numeric or unknown 
    columns handling depends on `<handle_unknown>`. The `<ravel>` 
    parameter applies only when `<asarray>` is True, adjusting the 
    shape of extracted arrays.
    
    If `<check_size>` is True and multiple arrays are extracted, all 
    must share the same first dimension length. Otherwise, a ValueError 
    is raised.
    
    Examples
    --------
    >>> from fusionlab.core.array_manager import extract_array_from
    >>> import pandas as pd
    >>> import numpy as np

    >>> df = pd.DataFrame({'A':[1,2,3],'B':[4,5,6],'C':[7,8,9]})
    >>> # Extract a single column as is:
    >>> result = extract_array_from(df, 'A')
    >>> print(result)
       A
    0  1
    1  2
    2  3

    >>> # Extract multiple columns and convert to array:
    >>> arr = extract_array_from(df, ['A','B'], asarray=True)
    >>> arr.shape
    (3, 2)

    >>> # Handle unknown column with 'passthrough':
    >>> result = extract_array_from(df, 'Z', handle_unknown='passthrough')
    >>> print(result)
    Z
    
    >>> # Check size consistency:
    >>> arrs = extract_array_from(df, 'A', ['B','C'], asarray=True,
    ...                           check_size=True)
    >>> # arrs will be a list of arrays all with length 3.

    See Also
    --------
    pandas.DataFrame : The input DataFrame structure.
    numpy.array : Arrays returned if `asarray=True`.

    References
    ----------
    .. [1] McKinney, W. "Data Structures for Statistical Computing
           in Python." Proceedings of the 9th Python in Science Conf.
           (2010): 51–56.
    """
    # handle scenario:
    # names could be single col names or lists of col names or arrays
    # if name is a string, must exist in df or handle_unknown scenario
    # if name is a list of strings, extract them from df (multicol)
    # if name is array-like (not string), just pass through if allowed by handle_unknown
    # ravel='1d', '2d', 'all', None means how to reshape arrays if asarray=True
    # check_size means ensure all extracted have same length
    # check the dataframe 
    are_all_frames_valid(df)
    
    # Helper to check if something is array-like but not a string
    def is_array_like(x):
        return hasattr(x, '__iter__') and not isinstance(x, str)

    extracted = []
    all_lengths = []

    for nm in col_names:
        if isinstance(nm, str):
            # single column name
            if nm in df.columns:
                res = df[[nm]] if asarray is False else df[[nm]].values
                # if single col and asarray=True, res shape (n,1)
                # if asarray=False, res is a dataframe with 1 col
            else:
                # column not found
                if handle_unknown == 'passthrough':
                    # return as is the name
                    res = nm
                elif handle_unknown is None:
                    # ignore silently?
                    # The instructions: if None return only valid names?
                    # Let's interpret: if None then we do not add this invalid col
                    # and if warn?
                    # There's no warn param given actually. Let's just skip col
                    # or if error='raise', raise
                    if error=='raise':
                        raise ValueError(f"Column {nm!r} not found in df.")
                    else:
                        # skip silently
                        continue
                else:
                    # if handle_unknown='passthrough', done above
                    # no other logic given, let's default to skipping if unknown scenario
                    if error=='raise':
                        raise ValueError(f"Column {nm!r} not found in df.")
                    else:
                        continue
        elif is_array_like(nm):
            # nm could be list of strings or array
            # check if list of strings
            if all(isinstance(x, str) for x in nm):
                # multiple column extraction
                valid_cols = [c for c in nm if c in df.columns]
                missing_cols = [c for c in nm if c not in df.columns]
                if missing_cols:
                    if handle_unknown == 'passthrough':
                        # keep the missing as they are?
                        # The instructions are not super clear
                        # Just return array with found and missing as original strings?
                        # Not consistent, but let's guess: if unknown and passthrough,
                        # we can just return the subset we found and add the missing as is?
                        # Or we skip missing?
                        # Let's skip missing silently or if error='raise', raise
                        if error=='raise':
                            raise ValueError(f"Columns {missing_cols} not found in df.")
                        elif error=='warn': 
                            warnings.warn(f"Columns {missing_cols} not found in df."
                                          " Skip them.")
                        # else skip them
                    else:
                        # if None then just skip them
                        if error=='raise':
                            raise ValueError(f"Columns {missing_cols} not found in df.")
                        elif error=='warn': 
                            warnings.warn(f"Columns {missing_cols} not found in df."
                                          " Skip them.")
                        # else skip them
                if valid_cols:
                    res = df[valid_cols] if asarray is False else df[valid_cols].values
                else:
                    # no valid cols found
                    # return empty?
                    res = np.array([]) if asarray else df.iloc[[], :]
            else:
                # nm is array-like but not all strings - treat as passthrough if allowed
                # If handle_unknown='passthrough', just return as array (if asarray=True)
                # else return nm as is
                if handle_unknown == 'passthrough':
                    res = np.array(nm) if asarray else nm
                else:
                    # if None or others?
                    # if error='raise', raise an error that we have unknown data
                    if error=='raise':
                        raise ValueError("Found array-like non-string not handled.")
                    elif error=='warn': 
                        warnings.warn(
                            "Found array-like non-string not handled. Skip it."
                        )
                    # skip
                    continue
        else:
            # nm is not string, not iterable means unknown
            # handle_unknown='passthrough'?
            if handle_unknown == 'passthrough':
                res = np.array([nm]) if asarray else nm
            else:
                if error=='raise':
                    raise ValueError(f"Unsupported type {type(nm)}.")
                else:
                    continue

        # Now handle ravel if asarray=True
        if asarray and isinstance(res, np.ndarray):
            if ravel is not None:
                # ravel logic
                if ravel == '1d':
                    # if shape is (n,1), ravel to (n,)
                    if len(res.shape)==2 and res.shape[1]==1:
                        res = res.ravel()
                elif ravel == '2d':
                    # if shape is (n,), expand to (n,1)
                    if len(res.shape)==1:
                        res = res.reshape(-1, 1)
                elif ravel in ('all','*'):
                    # if (n,1)->(n,), if (n,)->(n,1)
                    # this contradict each other though
                    # Let's assume 'all' means just ravel 1D arrays
                    if len(res.shape)==2 and res.shape[1]==1:
                        res = res.ravel()
                    elif len(res.shape)==1:
                        # keep as is if 1D
                        pass
                # else no action

        extracted.append(res)
        # store length for check_size
        if check_size:
            if asarray and isinstance(res, np.ndarray):
                all_lengths.append(res.shape[0])
            else:
                # if not array, if dataframe or series
                if hasattr(res, 'shape'):
                    all_lengths.append(res.shape[0])
                elif isinstance(res, (list, tuple)):
                    all_lengths.append(len(res))
                else:
                    # single element maybe?
                    all_lengths.append(1)

    # After extraction, if check_size=True verify lengths
    if check_size and len(all_lengths)>1:
        if len(set(all_lengths))>1:
            # mismatch
            raise ValueError(f"Mismatched lengths in extracted arrays: {all_lengths}")

    return extracted if len(extracted)>1 else extracted[0] if extracted else None

# Check if something is array-like but not a string
def is_array_like(x):
    """Check if object `x` is array-like but not a string."""
    return hasattr(x, '__iter__') and not isinstance(x, str)

def reduce_dimensions(
    arr: np.ndarray,
    z: Union[List, np.ndarray],
    x: Union[List, np.ndarray],
    ops: str = 'reduce',
    axis_names: Tuple[str, str] = ('Z', 'X'),
    error: str ='raise', 
    strict: bool = False,
    logger: Optional[logging.Logger] = None
) -> Union[
    bool, 
    Tuple[np.ndarray, Union[List, np.ndarray], Union[List, np.ndarray]]
]:
    """
    Reduce or Check the Dimensionality of a 2D Data Array.
    
    The ``reduce_dimensions`` function offers a robust mechanism to verify 
    and adjust the dimensionality of a 2D NumPy array ``data`` based on the 
    lengths of the provided axes ``z`` and ``x``. It supports two primary 
    operations:
    
    - **Check Only**: Validates whether the dimensions of ``data`` align with 
      the lengths of ``z`` and ``x``.
    - **Reduce**: Adjusts the dimensions of ``data``, ``z``, and ``x`` to 
      ensure consistency by truncating or padding as necessary.
    
    This utility is particularly beneficial in data preprocessing pipelines 
    where dimensional consistency is crucial for downstream analyses.
    
    .. math::
        \text{Operation} = 
        \begin{cases}
            \text{Check Dimensions} & \text{if } \text{ops} = \text{'check_only'} \\
            \text{Adjust Dimensions} & \text{if } \text{ops} = \text{'reduce'}
        \end{cases}
    
    Parameters
    ----------
    arr : `numpy.ndarray`
        The input 2D array with dimensions corresponding to ``z`` and ``x``.
    
    z : Union[`List`, `numpy.ndarray`]
        Array-like object representing the Z-axis. Should be reduced or checked 
        against the first dimension of ``data``.
    
    x : Union[`List`, `numpy.ndarray`]
        Array-like object representing the X-axis. Should be reduced or checked 
        against the second dimension of ``data``.
    
    ops : `str`, default=`'reduce'`
        Operation mode.
        
        - ``'check_only'``: Validate if ``z`` and ``x`` can be reduced to match 
          ``data``.
        - ``'reduce'``: Perform the reduction of ``z``, ``x``, and ``data`` to 
          match ``data``'s dimensions.
    
    axis_names : `Tuple[str, str]`, default=(``'Z'``, ``'X'``)
        Names of the axes for logging purposes.
    
    error : str, default='raise'
        The strategy for handling validation errors.
        - ``'warn'``: Issues warnings and continues processing.
        - ``'raise'``: Raises exceptions upon encountering errors.
    
    strict : `bool`, default=`False`
        If ``True``, enforces that no padding occurs during reduction. Only 
        truncation is allowed. If padding is required and ``strict`` is ``True``, 
        an error is raised.
        
    logger : `logging.Logger`, optional
        Logger instance for debug messages. If ``None``, a default logger is used.
    
    Returns
    -------
    Union[
        `bool`, 
        Tuple[`numpy.ndarray`, Union[`List`, `numpy.ndarray`], 
              Union[`List`, `numpy.ndarray`]]
    ]
        - If ``ops='check_only'``, returns ``True`` if dimensions match or can 
          be reduced, otherwise ``False``.
        - If ``ops='reduce'``, returns a tuple containing the reduced ``data``, 
          ``z``, and ``x``.
    
    Raises
    ------
    ValueError
        - If ``ops`` is not ``'check_only'`` or ``'reduce'``.
        - If reduction is not possible and ``ops='check_only'``.
        - If ``strict`` is ``True`` and padding is required.
    
    TypeError
        - If ``data`` is not a 2D ``numpy.ndarray``.
        - If ``z`` and ``x`` are not list-like.
    
    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.core.array_manager import reduce_dimensions
    >>> 
    >>> # Sample data array
    >>> data = np.array([
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9]
    ... ])
    >>> 
    >>> # Corresponding axes
    >>> z = [10, 20, 30]
    >>> x = [100, 200, 300]
    >>> 
    >>> # Check dimensionality
    >>> result = reduce_dimensions(data, z, x, ops='check_only')
    >>> print(result)
    True
    >>> 
    >>> # Reduce dimensions (no change needed in this case)
    >>> reduced_data, reduced_z, reduced_x = reduce_dimensions(
    ...     data, z, x, ops='reduce'
    ... )
    >>> print(reduced_data)
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    >>> print(reduced_z)
    [10, 20, 30]
    >>> print(reduced_x)
    [100, 200, 300]
    >>> 
    >>> # Dimensionality mismatch
    >>> z_mismatch = [10, 20]
    >>> result = reduce_dimensions(data, z_mismatch, x, ops='check_only')
    >>> print(result)
    False
    >>> 
    >>> # Reduce dimensions with mismatch
    >>> reduced_data, reduced_z, reduced_x = reduce_dimensions(
    ...     data, z_mismatch, x, ops='reduce'
    ... )
    >>> print(reduced_data)
    [[1 2 3]
     [4 5 6]]
    >>> print(reduced_z)
    [10, 20]
    >>> print(reduced_x)
    [100, 200, 300]
    
    Notes
    -----
    - **Logging**: The function utilizes a logger to provide debug information 
      during the dimensionality check and reduction processes. Users can 
      supply their own logger or rely on the default logger.
    
    - **Axis Adjustment**: When reducing dimensions, if the length of an 
      axis exceeds the corresponding dimension in ``data``, the axis is 
      truncated. If it is shorter, the axis is either padded with ``None`` 
      values or an error is raised based on the ``strict`` parameter.
    
    - **Error Handling**: By default, the function raises exceptions for any 
      mismatches or invalid parameters. Users can modify the behavior by 
      adjusting the ``ops`` and ``strict`` parameters.
    
    - **Flexibility**: The function is designed to handle various 
      dimensionality scenarios, making it suitable for preprocessing steps 
      in data analysis workflows.
    
    See Also
    --------
    numpy.ndarray : The primary array object in NumPy.
    logging.Logger : Logger class for debug messages.
    
    References
    ----------
    .. [1] NumPy Documentation: numpy.ndarray.  
       https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html  
    .. [2] Python Documentation: logging.  
       https://docs.python.org/3/library/logging.html  
    .. [3] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """
    
    # Set up logger
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s:%(name)s:%(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
    
    # Validate input types
    arr = np.asarray(arr)
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"'data' must be a numpy.ndarray, got {type(arr).__name__}."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"'data' must be a 2D array, got {arr.ndim}D array."
        )

    if not isinstance(z, (list, np.ndarray, pd.Series)):
        raise TypeError(
            f"'z' must be a list or numpy.ndarray, got {type(z).__name__}."
        )
    if not isinstance(x, (list, np.ndarray, pd.Series)):
        raise TypeError(
            f"'x' must be a list or numpy.ndarray, got {type(x).__name__}."
        )
    
    # Validate 'ops' parameter
    if ops not in {'check_only', 'reduce'}:
        raise ValueError(
            f"Invalid 'ops' value '{ops}'. Expected 'check_only' or 'reduce'."
        )
    
    # Define helper function to adjust axis lengths
    def adjust_axis_length(
        desired_length: int, 
        current_axis: Union[List, np.ndarray], 
        axis_name: str
    ) -> Tuple[int, Union[List, np.ndarray]]:
        """
        Adjust the length of an axis to match the desired length.
        
        Parameters
        ----------
        desired_length : `int`
            The target length for the axis.
        
        current_axis : Union[`List`, `numpy.ndarray`]
            The current axis data to be adjusted.
        
        axis_name : `str`
            The name of the axis for logging purposes.
        
        Returns
        -------
        Tuple[int, Union[`List`, `numpy.ndarray`]]
            A tuple containing the new length and the adjusted axis.
        """
        original_length = len(current_axis)
        if original_length > desired_length:
            adjusted_axis = current_axis[:desired_length]
            logger.debug(
                f"Resizing ``{axis_name}`` from {original_length} to "
                f"{desired_length}."
            )
        elif original_length < desired_length:
            if strict:
                msg = (
                    f"Cannot pad ``{axis_name}`` to match desired length "
                    f"{desired_length}."
                )
                if ops == 'reduce':
                    raise ValueError(msg)
                elif error == 'raise':
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
                    return desired_length, current_axis
            adjusted_axis = list(current_axis) + [None] * (
                desired_length - original_length
            )
            logger.debug(
                f"Resizing ``{axis_name}`` from {original_length} to "
                f"{desired_length} by padding."
            )
        else:
            adjusted_axis = current_axis
            logger.debug(
                f"No resizing needed for ``{axis_name}``; already at desired "
                f"length {desired_length}."
            )
        return desired_length, adjusted_axis
    
    if ops == 'check_only':
        z_matches = len(z) == arr.shape[0]
        x_matches = len(x) == arr.shape[1]
        if z_matches and x_matches:
            logger.debug(
                "Data dimensions match ``z`` and ``x`` dimensions."
            )
            return True
        else:
            msg = "Data dimensionality does not match ``z`` and/or ``x`` dimensions."
            if not z_matches and not x_matches:
                msg += (
                    f" ``z`` has length {len(z)} vs {arr.shape[0]}, and "
                    f"``x`` has length {len(x)} vs {arr.shape[1]}."
                )
            elif not z_matches:
                msg += f" ``z`` has length {len(z)} vs {arr.shape[0]}."
            else:
                msg += f" ``x`` has length {len(x)} vs {arr.shape[1]}."
            if error == 'raise':
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return False
    
    elif ops == 'reduce':
        # Determine new lengths based on current data shape and axis lengths
        new_z_length = min(len(z), arr.shape[0])
        new_x_length = min(len(x), arr.shape[1])
        
        # Adjust 'z' axis
        sz0, z_adjusted = adjust_axis_length(
            desired_length=new_z_length, 
            current_axis=z, 
            axis_name=axis_names[0]
        )
        
        # Adjust 'x' axis
        sx0, x_adjusted = adjust_axis_length(
            desired_length=new_x_length, 
            current_axis=x, 
            axis_name=axis_names[1]
        )
        
        # Slice the data accordingly
        data_reduced = arr[:sz0, :sx0]
        logger.debug(
            "Data, ``z``, and ``x`` have been reduced to match dimensions."
        )
        return data_reduced, z_adjusted, x_adjusted
    
def decode_sparse_data(sparse_data: pd.Series) -> pd.DataFrame:
    """
    Decode a sparse matrix represented as strings in a pandas Series 
    back into a dense pandas DataFrame.
    
    Each entry in the `sparse_data` Series should contain multiple lines, 
    where each line represents a non-zero entry in the format:
    `(row_index, column_index)\tvalue`.
    
    **Note:** This function assumes that the row indices within the strings 
    are always `(0, column_index)`. The Series index is used as the actual 
    row index in the decoded DataFrame.
    
    Parameters
    ----------
    sparse_data : pd.Series
        A pandas Series where each element is a string representing a 
        sparse matrix row. Each string contains entries separated by 
        newline characters (`\n`), and each entry is in the format 
        `(row, col)\tvalue`.
    
    Returns
    -------
    pd.DataFrame
        A dense pandas DataFrame reconstructed from the sparse representation.
    
    Raises
    ------
    ValueError
        If an entry in `sparse_data` is not a string or does not follow the 
        expected format.
    
    Examples
    --------
    >>> from fusionlab.core.array_manager import decode_sparse_data
    >>> import pandas as pd
    >>> 
    >>> # Sample sparse data as a pandas Series
    >>> sparse_data = pd.Series([
    ...     "(0, 0)\t-1.6752467319482305\n(0, 1)\t1.515...",
    ...     "(0, 0)\t-1.5597124745724904\n(0, 1)\t-0.00...",
    ...     "(0, 0)\t-1.4441782171967503\n(0, 1)\t-1.41...",
    ...     "(0, 0)\t-1.3286439598210102\n(0, 1)\t0.912...",
    ...     "(0, 0)\t-1.2131097024452704\n(0, 1)\t-0.41..."
    ... ])
    >>> 
    >>> # Decode the sparse data
    >>> decoded_df = decode_sparse_data(sparse_data)
    >>> print(decoded_df)
               0     1
        0 -1.675  1.515
        1 -1.560 -0.000
        2 -1.444 -1.410
        3 -1.329  0.912
        4 -1.213 -0.410
    
    Notes
    -----
    - **Input Structure:** Each entry in the `sparse_data` Series should be a 
      string containing multiple `(row, column)\tvalue` pairs separated by 
      newline characters.
    - **Row Mapping:** The Series index is used as the row index in the 
      decoded DataFrame. The row index specified within each string is 
      ignored and assumed to be `0`.
    - **Memory Consideration:** Decoding large sparse matrices into dense 
      DataFrames can consume significant memory. Consider using sparse 
      DataFrame representations if memory usage is a concern.
    
    References
    ----------
    - pandas.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    - scipy.sparse.coo_matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
    """
    from scipy.sparse import coo_matrix
    
    if isinstance ( sparse_data, pd.DataFrame): 
        # try to squeeze the dataframe if has a single column 
        sparse_data = sparse_data.squeeze() 
        
    if not isinstance(sparse_data, pd.Series):
        raise ValueError("Input `sparse_data` must be a pandas Series.")
    
    rows = []
    cols = []
    data = []
    
    for series_idx, row in sparse_data.items():
        if not isinstance(row, str):
            row = str(row)  # Convert to string if possible
        
        for entry in row.split('\n'):
            entry = entry.strip()
            if entry:
                try:
                    col_row, value = entry.split('\t')
                    # Remove parentheses and split into row and column
                    _, col_idx = map(int, col_row.strip('()').split(','))
                    rows.append(series_idx)  # Use Series index as row index
                    cols.append(col_idx)
                    data.append(float(value))
                except Exception as e:
                    raise ValueError(
                        f"Error parsing entry '{entry}'"
                        f" in Series index {series_idx}: {e}"
                    )
    
    if not rows:
        raise ValueError("No data found to decode.")
    
    # Determine the size of the matrix
    max_row = max(rows) + 1
    max_col = max(cols) + 1
    
    # Create a COO sparse matrix
    sparse_matrix = coo_matrix(
        (data, (rows, cols)), shape=(max_row, max_col)
    )
    
    # Convert to a dense NumPy array
    dense_array = sparse_matrix.toarray()
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame(dense_array)
    
    return df

def process_and_extract_data(
    *args: ArrayLike, 
    columns: Optional[List[Union[str, int]]] = None,
    enforce_extraction: bool = True, 
    allow_split: bool = False, 
    search_multiple: bool = False,
    ensure_uniform_length: bool = False, 
    to_array: bool = False,
    on_error: str = 'raise',
) -> List[np.ndarray]:
    """
    Extracts and processes data from various input types, focusing on column 
    extraction from pandas DataFrames and conversion of inputs to numpy 
    arrays or pandas Series.

    Parameters
    ----------
    *args : ArrayLike
        A variable number of inputs, each can be a list, numpy array, pandas 
        Series,dictionary, or pandas DataFrame.
    columns : List[Union[str, int]], optional
        Specific columns to extract from pandas DataFrames. If not provided, 
        the function behaves differently based on `allow_split`.
    enforce_extraction : bool, default=True
        Forces the function to try extracting `columns` from DataFrames. 
        If False, DataFrames are returned without column extraction unless 
        `allow_split` is True.
        Removing non-conforming elements if True.
    allow_split : bool, default=False
        If True and a DataFrame is provided without `columns`, splits the 
        DataFrame into its constituent columns.
    search_multiple : bool, default=False
        Allows searching for `columns` across multiple DataFrame inputs. Once 
        a column is found, it is not searched for in subsequent DataFrames.
    ensure_uniform_length : bool, default=False
        Checks that all extracted arrays have the same length. Raises an error
        if they don't.
    to_array : bool, default=False
        Converts all extracted pandas Series to numpy arrays.
    on_error : str, {'raise', 'ignore'}, default='raise'
        Determines how to handle errors during column extraction or when 
        enforcing uniform length. 'raise' will raise an error, 'ignore' will 
        skip the problematic input.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays or pandas Series extracted based 
        on the specified conditions.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from fusionlab.core.array_manager import process_and_extract_data
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> process_and_extract_data(data, columns=['A'], to_array=True)
    [array([1, 2, 3])]

    Splitting DataFrame into individual arrays:

    >>> process_and_extract_data(data, allow_split=True, to_array=True)
    [array([1, 2, 3]), array([4, 5, 6])]

    Extracting columns from multiple DataFrames:

    >>> data2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
    >>> process_and_extract_data(data, data2, columns=['A', 'C'], 
                                  search_multiple=True, to_array=True)
    [array([1, 2, 3]), array([7, 8, 9])]

    Handling mixed data types:

    >>> process_and_extract_data([1, 2, 3], {'E': [13, 14, 15]}, to_array=True)
    [array([1, 2, 3]), array([13, 14, 15])]
    
    Extracting columns from multiple DataFrames and enforcing uniform length:
    >>> data2 = pd.DataFrame({'C': [7, 8, 9, 10], 'D': [11, 12, 13, 14]})
    >>> result = process_and_extract_data(
        data, data2, columns=['A', 'C'],search_multiple=True,
        ensure_uniform_length=True, to_array=True)
    ValueError: Extracted data arrays do not have uniform length.
    """
    extracted_data = []
    columns_found: Set[Union[str, int]] = set()

    def _process_input(
            input_data: ArrayLike,
            target_columns: Optional[List[Union[str, int]]], 
            to_array: bool) -> Optional[np.ndarray]:
        """
        Processes each input based on its type, extracting specified columns 
        if necessary, and converting to numpy array if specified.
        """
        if isinstance(input_data, (list, tuple)):
            input_data = np.array(input_data)
            return input_data if len(input_data.shape
                                     ) == 1 or not enforce_extraction else None

        elif isinstance(input_data, dict):
            input_data = pd.DataFrame(input_data)

        if isinstance(input_data, pd.DataFrame):
            if target_columns:
                for col in target_columns:
                    if col in input_data.columns and (
                            search_multiple or col not in columns_found):
                        data_to_add = input_data[col].to_numpy(
                            ) if to_array else input_data[col]
                        extracted_data.append(data_to_add)
                        columns_found.add(col)
                    elif on_error == 'raise':
                        raise ValueError(f"Column {col} not found in DataFrame.")
            elif allow_split:
                for col in input_data.columns:
                    data_to_add = input_data[col].to_numpy(
                        ) if to_array else input_data[col]
                    extracted_data.append(data_to_add)
            return None

        if isinstance(input_data, np.ndarray):
            if input_data.ndim > 1 and allow_split:
                input_data = np.hsplit(input_data, input_data.shape[1])
                for arr in input_data:
                    extracted_data.append(arr.squeeze())
                return None
            elif input_data.ndim > 1 and enforce_extraction and on_error == 'raise':
                raise ValueError("Multidimensional array found while "
                                 "`enforce_extraction` is True.")
            return input_data if to_array else np.squeeze(input_data)

        return input_data.to_numpy() if to_array and isinstance(
            input_data, pd.Series) else input_data

    for arg in args:
        result = _process_input(arg, columns, to_array)
        if result is not None:
            extracted_data.append(result)

    if ensure_uniform_length and not all(len(x) == len(
            extracted_data[0]) for x in extracted_data):
        if on_error == 'raise':
            raise ValueError("Extracted data arrays do not have uniform length.")
        else:
            return []

    return extracted_data

def to_series_if(
    *values: Any, 
    value_names: Optional[List[str]] = None, 
    name: Optional[str] = None,
    error: str = 'ignore',
    **kws
) -> Series:
    """
    Constructs a pandas Series from given values, optionally naming 
    the series and its index.

    Parameters
    ----------
    *values : Any
        A variable number of inputs, each can be a scalar, float, int,\
            or array-like object.
    value_names : Optional[List[str]]
        Names to be used for the index of the series. If not provided or 
        if its length doesn't match the number of values, default numeric
        index is used.
    name : Optional[str]
        Name of the series.
    error : str, default 'ignore'
        Error handling strategy ('ignore' or 'raise'). If 'raise', errors 
        during series
        construction lead to an exception.
    **kws : dict
        Additional keyword arguments passed to `pd.Series` constructor.

    Returns
    -------
    pd.Series or original values
        A pandas Series constructed from the inputs if successful, otherwise,
        the original values if the series construction is not applicable.

    Examples
    --------
    >>> from fusionlab.core.array_manager import to_series_if
    >>> series = to_series_if(0.5, 8, np.array(
        [6.3]), [5], 2, value_names=['a', 'b', 'c', 'd', 'e'])
    >>> print(series)
    a    0.5
    b    8.0
    c    6.3
    d    5.0
    e    2.0
    dtype: float64
    >>> series = to_series_if(0.5, 8, np.array([6.3, 7]), [5], 2,
                              value_names=['a', 'b', 'c', 'd', 'e'], error='raise')
    ValueError: Failed to construct series, input types vary.
    """
    # Validate input lengths and types
    if value_names and len(value_names) != len(values):
        if error == 'raise':
            raise ValueError("Length of `value_names` does not"
                             " match the number of values.")
        value_names = None  # Reset to default indexing
    # Attempt to construct series
    try:
        # Flatten array-like inputs to avoid creating Series of lists/arrays
        flattened_values = [val[0] if isinstance(
            val, (list,tuple,  np.ndarray, pd.Series)) 
            and len(val) == 1 else val for val in values]
        series = pd.Series(
            flattened_values, index=value_names, name=name, **kws)
    except Exception as e:
        if error == 'raise':
            raise ValueError(f"Failed to construct series due to: {e}")
        return values  # Return the original values if series construction fails

    return series

def make_arr_consistent (
        refarr, arr, fill_value = np.nan, return_index = False, 
        method='naive'): 
    """
    Make `arr` to be consistent with the reference array `refarr`. Fill the 
    missing value with param `fill_value`. 
    
    Note that it does care of the position of the value in the array. Use 
    Numpy digitize to compute the bins. The array caveat here is the bins 
    must be monotonically decreasing or increasing.
    
    If the values in `arr` are present in `refarr`, the position of `arr` 
    in new consistent array should be located decreasing or increasing order. 
    
    Parameters 
    ------------
    arr: array-like 1d, 
        Array to extended with fill value. It should be  shorter than the 
        `refarr`.
        
    refarr: array-like- the reference array. It should have a greater 
        length than the array `arr`.  
    fill_value: float, 
        Value to fill the `arr` to match the length of the `refarr`. 
    return_index: bool or str, default=True 
         index of the position of the  elements in `refarr`.
         Default is ``False``. If ``mask`` should  return the 
        mask of existing element in reference array
    method: str, default="naive"
        Is the method used to find the right position of items in `arr`
        based on the reference array. 
        - ``naive``, considers the length of ``arr`` must fit the number of 
            items that should be visible in the consistent array. This method 
            erases the remaining bins values out of length of `arr`. 
        - ``strict` did the same but rather than considering the length, 
            it considers the maximum values in the `arr`. It assumes that `arr`
            is sorted in ascending order. This methods is usefull for plotting 
            a specific stations since the station loactions are sorted in 
            ascending order. 
        
    Returns 
    ---------
    non_zero_index , mask or t  
        index: indices of the position of `arr` items in ``refarr``. 
        mask: bool of the position `arr` items in ``refarr``
        t: new consistent array with the same length as ``refarr``
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from fusionlab.core.array_manager import make_arr_consistent
    >>> refarr = np.arange (12) 
    >>> arr = np.arange (7, 10) 
    >>> make_arr_consistent (refarr, arr ) 
    Out[84]: array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9., nan, nan])
    >>> make_arr_consistent (refarr, arr , return_index =True )
    Out[104]: array([7, 8, 9], dtype=int64)
    >>> make_arr_consistent (refarr, arr , return_index ="mask" )
    Out[105]: 
    array([False, False, False, False, False, False, False,  True,  True,
            True, False, False])
    >>> a = np.arange ( 12 ); b = np.linspace (7, 10 , 7) 
    >>> make_arr_consistent (a, b ) 
    Out[112]: array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9., 10., 11.])
    >>> make_arr_consistent (a, b ,method='strict') 
    Out[114]: array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9., 10., nan])
    """
    try : 
        refarr = reshape( refarr).shape[1] 
        arr= reshape( arr).shape[1] 
    except :pass 
    else: raise TypeError ("Expects one-dimensional arrays for both arrays.")

    t = np.full_like( refarr, fill_value = np.nan, dtype =float )
    temp_arr = np.digitize( refarr, arr) 
    non_zero_index = reshape (np.argwhere (temp_arr!=0 ) ) 
    t[non_zero_index] = refarr [non_zero_index] 
    # force value to keep only 
    # value in array 
    if method=='strict':
        index = reshape ( np.argwhere (  (max( arr)  - t) < 0 ) ) 
        t [index ]= np.nan 
    else: 
        if len (t[~np.isnan (t)]) > len(arr): 
            t [ - (len(t[~np.isnan (t)])-len(arr)):]= np.nan 
    # update the non_zeros index 
    non_zero_index= reshape ( np.argwhere (~np.isnan (t)))
    # now replace all NaN value by filled value 
    t [np.isnan(t)] = fill_value 

    return  refarr == t  if return_index =='mask' else (
        non_zero_index if return_index else t )

def split_train_test(
        data: DataFrame, test_ratio: float = 0.2
        ) -> Tuple[DataFrame, DataFrame]:
    """
    Split a DataFrame into train and test sets based on a given ratio.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the features.
    test_ratio : float, optional
        The ratio of the test set, ranging from 0 to 1. Default is 0.2 (20%).

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        A tuple of the train set and test set DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> from fusionlab.core.array_manager import split_train_test
    >>> data = load_iris(as_frame=True)['data']
    >>> train_set, test_set = split_train_test(data, test_ratio=0.2)
    >>> len(train_set), len(test_set)
    ... (120, 30)
    """

    test_ratio = assert_ratio(test_ratio)

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check_id(
        identifier: int, test_ratio: float, hash: _F[_T]) -> bool:
    """
    Check if an instance should be in the test set based on its unique identifier.

    Parameters
    ----------
    identifier : int
        A unique identifier for the instance.
    test_ratio : float, optional
        The ratio of instances to put in the test set. Default is 0.2 (20%).
    hash : callable
        A hash function to generate a hash from the identifier.
        Secure hashes and message digests algorithm. Can be 
        SHA1, SHA224, SHA256, SHA384, and SHA512 (defined in FIPS 180-2) 
        as well as RSA’s MD5 algorithm (defined in Internet RFC 1321). 
        
        Please refer to :ref:`<https://docs.python.org/3/library/hashlib.html>` 
        for futher details.

    Returns
    -------
    bool
        True if the instance should be in the test set, False otherwise.

    Examples
    --------
    >>> from fusionlab.core.array_manager import test_set_check_id
    >>> test_set_check_id(42, test_ratio=0.2, hash=hashlib.md5)
    ... False
    """
    # def test_set_check_id(identifier: str, ratio: float, hash_function: _F) -> bool:
    #     """Determines if an identifier belongs to the test set using the hash value."""
    #     # Convert identifier to string and hash
    #     hash_val = int(hash_function(str(identifier).encode()).hexdigest(), 16)
    #     # Use the hash value to decide test set membership
    #     return hash_val % 10000 / 10000.0 < ratio
    
    #     hashed_id = hash_function(identifier.encode('utf-8')).digest()
    #     return np.frombuffer(hashed_id, dtype=np.uint8).sum() < 256 * test_ratio
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(
    data: DataFrame, test_ratio: float, id_column: Optional[List[str]] = None,
    keep_colindex: bool = True, hash: _F = hashlib.md5
) -> Tuple[DataFrame, DataFrame]:
    """
    Split a DataFrame into train and test sets while ensuring data consistency
    by using specified id columns or the DataFrame's index as unique identifiers.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the features.
    test_ratio : float
        The ratio of instances to include in the test set.
    id_column : list of str, optional
        Column names to use as unique identifiers. If None, the DataFrame's index
        is used as the identifier.
    keep_colindex : bool, optional
        Determines whether to keep or drop the index column after resetting.
        This parameter is only applicable if id_column is None and the DataFrame's
        index is reset. Default is True.
    hash : callable
        A hash function to generate a hash from the identifier.

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        A tuple containing the train and test set DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.core.array_manager import split_train_test_by_id
    >>> data = pd.DataFrame({'ID': [1, 2, 3, 4, 5], 'Value': [10, 20, 30, 40, 50]})
    >>> train_set, test_set = split_train_test_by_id(data, test_ratio=0.2, id_column=['ID'])
    >>> len(train_set), len(test_set)
    (4, 1)
    """
    drop_tmp_index=False
    if id_column is None:
        # Check if the index is integer-based; if not, create a temporary integer index.
        if not data.index.is_integer():
            data['_tmp_hash_index'] = np.arange(len(data))
            ids = data['_tmp_hash_index']
            drop_tmp_index = True
        else:
            ids = data.index.to_series()
            drop_tmp_index = False
    else:
        # Use specified id columns as unique identifiers, combining them if necessary.
        ids = data[id_column].astype(str).apply(
            lambda row: '_'.join(row), axis=1) if isinstance(
                id_column, list) else data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check_id(id_, test_ratio, hash))

    train_set = data.loc[~in_test_set].copy()
    test_set = data.loc[in_test_set].copy()

    if drop_tmp_index or (id_column is None and not keep_colindex):
        # Remove the temporary index or reset the index as needed
        train_set.drop(columns=['_tmp_hash_index'], errors='ignore', inplace=True)
        test_set.drop(columns=['_tmp_hash_index'], errors='ignore', inplace=True)
        # for consistency if '_tmp_has_index' 
        if '_tmp_hash_index' in data.columns: 
            data.drop (columns='_tmp_hash_index', inplace =True)
    elif id_column is None and keep_colindex:
        # If keeping the original index and it was integer-based, no action needed
        pass

    return train_set, test_set


def split_list(
    lst: List[Any],
    val: int,
    fill_value: Optional[Any] = None
) -> List[List[Any]]:
    """
    Split a one‑dimensional list into contiguous chunks of length
    ``val`` while preserving order.

    A chunk boundary is determined by the integer division

    .. math::
        k = \\bigl\\lfloor \\tfrac{(i - 1)}{\\text{val}} \\bigr\\rfloor,

    where :math:`i` is the *1‑based* index of the element in
    ``lst``.  All elements sharing the same :math:`k` belong to
    the same sub‑list.  When non‑numeric items prevent this
    arithmetic grouping, the function falls back to
    :pyfunc:`itertools.zip_longest`, optionally padding with
    ``fill_value``.

    Parameters
    ----------
    lst : list of Any
        Input sequence to be partitioned.  The helper
        `is_iterable` coerces array‑likes (except strings) into
        a list.
    val : int
        Size of each group.  Must be a positive integer.  It is
        validated with `_assert_all_types` and cast to ``int``.
    fill_value : Any, optional
        Placeholder used when the fallback strategy produces
        ragged groups.  Only relevant when ``lst`` contains
        non‑numeric items and the final chunk is shorter than
        ``val``.

    Returns
    -------
    list of list
        Nested list in which each inner list holds at most
        ``val`` items, e.g. ``[[e₁, …, e_val], …]``.

    Notes
    -----
    * The primary path uses :pyfunc:`itertools.groupby` with the
      grouping key shown above.  This requires elements to be
      *numeric* (support subtraction and division).
    * If the numeric assumption fails (e.g. strings), a safer
      but slightly slower zip‑based approach is used.
    * No copy of the items themselves is made—only the list
      structure is rebuilt.

    Examples
    --------
    >>> from fusionlab.core.array_manager import split_list
    >>> data = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> split_list(data, 3)
    [[1, 2, 3], [4, 5, 6], [7, 8]]

    >>> # Strings trigger the fallback mechanism
    >>> split_list(list("abcdefg"), 3, fill_value="*")
    [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', '*', '*']]

    See Also
    --------
    numpy.array_split : N‑way splitting for NumPy arrays.
    itertools.groupby  : Group consecutive items by a key
        function.
    itertools.zip_longest : Pad shorter iterables to equal
        length.

    References
    ----------
    .. [1] Python Software Foundation. *itertools — Functions
       creating iterators for efficient looping*.
       https://docs.python.org/3/library/itertools.html
    """
    # 1. Ensure *lst* is a proper list (exclude bare strings).   
    lst = is_iterable(
        lst,
        exclude_string=True,
        transform=True
    )

    # 2. Validate and normalise *val*.                           
    val = int(
        _assert_all_types(
            val,
            int,
            float
        )
    )

    # 3. Attempt numeric grouping via itertools.groupby.         
    try:
        grouped = [
            list(group)
            for _, group in itertools.groupby(
                lst,
                key=lambda x: (x - 1) // val
            )
        ]
    except Exception:

        # Non‑numeric elements: fallback to zip_longest,         
        # padding with *fill_value* when the last chunk          
        # under‑flows.                                           
        grouped = list(
            itertools.zip_longest(
                *(
                    iter(lst),
                ) * val,
                fillvalue=fill_value
            )
        )
        # Remove trailing pad if present in the final chunk
        if fill_value is None:
            grouped[-1] = [
                item for item in grouped[-1]
                if item is not None
            ]

    return grouped

def squeeze_specific_dim(
    arr: np.ndarray,
    axis: Optional[int] = -1
) -> np.ndarray:
    """
    Remove single‑dimension axes from *arr* along a chosen
    `axis`.

    If `axis` is ``None``, every axis with length 1 is removed,
    mimicking :pyfunc:`numpy.squeeze`.  Otherwise, only the
    indicated axis is targeted and dropped when its size is 1.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array to be reshaped.
    axis : int or None, default ``-1``
        * If ``None`` all singleton axes are removed.  
        * If an integer, only that axis is squeezed provided
          its length equals 1.  Negative indices follow NumPy
          convention.

    Returns
    -------
    numpy.ndarray
        A view of *arr* with the specified singleton axis
        removed, or *arr* unchanged when no squeeze is
        possible.

    Notes
    -----
    The squeeze condition is

    .. math::
        \\text{shape}[\\text{axis}] = 1.

    When this is false the original array is returned.

    Examples
    --------
    >>> from fusionlab.core.array_manager import squeeze_specific_dim
    >>> a = np.arange(6).reshape(3, 2, 1)
    >>> squeeze_specific_dim(a).shape
    (3, 2)

    >>> squeeze_specific_dim(a, axis=1).shape
    (3, 2, 1)          # axis 1 has length 2 → no change

    See Also
    --------
    numpy.squeeze : Built‑in NumPy equivalent.

    References
    ----------
    .. [1] Harris, C. R. *et al.* "Array programming with NumPy",
       *Nature*, 585, 357‑362 (2020).
    """
    if axis is None:
        # Remove *all* singleton axes
        return np.squeeze(arr)
    try:
        # Remove the requested singleton axis
        return np.squeeze(arr, axis=axis)
    except ValueError:
        # Axis length ≠ 1 → return original array
        return arr

def reshape(
    arr: Any,
    axis: Optional[int] = None
) -> np.ndarray:
    """
    Reshape a 1‑D or 2‑D array to match a desired orientation
    given by `axis`.

    The function is a convenience wrapper that flips between
    row‑vector ``(1, N)``, column‑vector ``(N, 1)`` and flat
    shape ``(N,)`` depending on `axis`.

    Parameters
    ----------
    arr : array_like
        Input data with at most two dimensions.
    axis : {0, 1, -1, None}, optional
        * ``0``   → ensure column‑vector shape ``(N, 1)``  
        * ``1``   → ensure row‑vector shape ``(1, N)``  
        * ``None`` or ``-1`` → flatten to ``(N,)``

    Returns
    -------
    numpy.ndarray
        Reshaped view of *arr*.

    Raises
    ------
    ValueError
        If *arr* has more than two dimensions or if `axis`
        takes an unsupported value.

    Examples
    --------
    >>> from fusionlab.core.array_manager import reshape
    >>> x = np.arange(5)
    >>> reshape(x, 1).shape
    (1, 5)

    >>> reshape(_, 0).shape
    (5, 1)

    >>> reshape(_, None).shape
    (5,)

    See Also
    --------
    numpy.reshape : General‑purpose reshaping routine.

    """
    arr = np.asarray(arr)

    # 1. Sanity checks                                       
    if arr.ndim > 2:
        raise ValueError(
            "Expect an array with ndim ≤ 2, got "
            f"{arr.ndim}."
        )
    if axis not in (0, 1, -1, None):
        raise ValueError(
            f"Unsupported axis value {axis!r}."
        )
    if axis == -1:
        axis = None

    # 2. Branch on original dimensionality                  
    if arr.ndim == 1:
        n = arr.shape[0]
        return (
            arr.reshape(1, n) if axis == 1 else
            arr.reshape(n, 1) if axis == 0 else
            arr
        )

    # arr.ndim == 2
    n0, n1 = arr.shape
    if axis is None:
        # Flatten to (N,)
        return arr.reshape(n0 * n1)
    if axis == 1 and n0 != 1:
        return arr.reshape(1, n0 * n1)
    if axis == 0 and n1 != 1:
        return arr.reshape(n0 * n1, 1)

    # If reshape is not feasible, return original
    return arr

def to_numeric_dtypes(
    arr: Union[NDArray, DataFrame], *, 
    columns: Optional[List[str]] = None, 
    return_feature_types: bool = False, 
    missing_values: float = np.nan, 
    pop_cat_features: bool = False, 
    sanitize_columns: bool = False, 
    regex: Optional[re.Pattern] = None, 
    fill_pattern: str = '_', 
    drop_nan_columns: bool = ..., 
    how: str = 'all', 
    reset_index: bool = False, 
    drop_index: bool = ..., 
    verbose: bool = False
) -> Union[DataFrame, Tuple[DataFrame, List[str], List[str]]]:
    """
    Convert input array or DataFrame into a numeric-friendly DataFrame by 
    coercing values to numeric data types where applicable and identifying 
    feature types.

    This utility function is essential for pre-processing mixed-type data, 
    such as datasets containing both numeric and categorical values, often 
    encountered in data science pipelines. It offers advanced options for 
    sanitizing column names, handling missing values, and dynamically 
    identifying and extracting numeric versus categorical features.

    The core mechanism iteratively attempts to convert each column of the 
    input array to :class:`float64` using a fallback strategy to preserve 
    categorical columns that cannot be converted. The classification of 
    features is guided by :func:`is_numeric_dtype`.

    .. math::

        \text{ProcessedDF} = f(\text{OriginalDF}, \text{cast}, \text{drop}_{NaN}, \text{sanitize}_{col})

    where:
        - :math:`f` is a transformation function;
        - :math:`cast` is the numeric coercion;
        - :math:`\text{drop}_{NaN}` denotes dropping columns/rows with NaNs;
        - :math:`\text{sanitize}_{col}` refers to column name formatting.

    Parameters
    ----------
    arr : NDArray or DataFrame
        Input data to be processed. If not already a DataFrame, the input is 
        coerced into one. Each row corresponds to a sample and each column to 
        a feature. Arrays must be two-dimensional.
    
    columns : list of str, optional
        Names to assign to the DataFrame columns when the input `arr` is a 
        NumPy array. The list must have the same length as the number of 
        columns in `arr`. If `arr` is already a DataFrame, this parameter 
        is ignored unless explicitly specified.
    
    return_feature_types : bool, default=False
        Whether to return feature type classification. If ``True``, the 
        function returns a tuple of:
        
            - Processed DataFrame
            - List of numeric feature names
            - List of categorical feature names

        This is useful for downstream feature engineering tasks.

    missing_values : float, default=np.nan
        Placeholder value to assign to cells that contain empty strings or 
        whitespace-only values. Typically set to :attr:`np.nan`.

    pop_cat_features : bool, default=False
        Whether to exclude all categorical features from the output. If 
        ``True``, the returned DataFrame will only contain numeric columns.

    sanitize_columns : bool, default=False
        If ``True``, column names are cleaned using the pattern specified by 
        `regex`. This replaces non-alphanumeric characters and ensures 
        standardized column labels, especially useful when importing data 
        from varied sources.

    regex : re.Pattern or str, optional
        Regular expression pattern for identifying non-alphanumeric 
        characters to be replaced in column names. If not provided, the 
        default pattern is::

            >>> import re
            >>> re.compile(r'[_#&.)(*@!_,;\\s-]\\s*', flags=re.IGNORECASE)

        This pattern captures special characters and whitespace sequences.

    fill_pattern : str, default='_'
        Replacement string used for matching patterns in `regex`. Common 
        choices are ``'_'`` or ``'-'`` depending on the naming convention.

    drop_nan_columns : bool, default=True
        If ``True``, columns containing only NaN values are removed. Combined 
        with the `how` parameter, it also enables dropping fully-NaN rows.

    how : {'all', 'any'}, default='all'
        Strategy to drop rows with NaN values:
        
            - ``'all'``: Drop rows where *all* elements are NaN.
            - ``'any'``: Drop rows with *any* NaN.

        Used in conjunction with `drop_nan_columns`.

    reset_index : bool, default=False
        Whether to reset the index of the DataFrame. If enabled, the old 
        index is dropped or kept based on the `drop_index` flag.

    drop_index : bool, default=True
        Specifies whether the original index should be dropped when 
        `reset_index` is applied. If ``False``, the index becomes a column 
        in the new DataFrame.

    verbose : bool, default=False
        If ``True``, prints detailed logs during processing including any 
        dropped columns or type conversions.

    Returns
    -------
    DataFrame or tuple of DataFrame, List[str], List[str]
        A cleaned DataFrame with numeric data types where applicable. If 
        `return_feature_types` is ``True``, a 3-element tuple is returned:
        
            - Cleaned DataFrame
            - List of numeric column names
            - List of categorical column names

    Notes
    -----
    This function facilitates safe type coercion and preprocessing before 
    applying ML algorithms which typically require numeric data. By isolating 
    numeric and categorical types, it supports modular feature transformation 
    pipelines and helps prevent common data-related errors.

    Empty string values are interpreted as missing and replaced by the 
    `missing_values` value before type coercion. Columns are attempted to be 
    casted to :class:`float64`, and non-convertible columns are preserved as 
    categorical or string types.

    Examples
    --------
    >>> from fusionlab.datasets.dload import load_bagoue
    >>> from fusionlab.core.array_manager import to_numeric_dtypes
    >>> X = load_bagoue(as_frame=True)
    >>> X0 = X[['shape', 'power', 'magnitude']]

    >>> # Get full output with separated feature types
    >>> df, nf, cf = to_numeric_dtypes(X0, return_feature_types=True)
    >>> print(df.dtypes)
    shape         object
    power        float64
    magnitude    float64
    dtype: object
    >>> print("Numeric Features:", nf)
    >>> print("Categorical Features:", cf)

    >>> # Remove categorical columns and keep only numeric ones
    >>> df_numeric = to_numeric_dtypes(X0, pop_cat_features=True)
    >>> print(df_numeric.columns)
    Index(['power', 'magnitude'], dtype='object')

    See Also
    --------
    fusionlab.preprocessing.sanitize_frame_cols : Sanitize DataFrame column names.
    fusionlab.tools.validator.is_numeric_dtype : Check for numeric column types.
    fusionlab.core.array_manager.to_dataframe : Convert arrays to structured DataFrames.

    References
    ----------
    .. [1] VanderPlas, J. (2016). *Python Data Science Handbook: Essential 
           Tools for Working with Data*. O'Reilly Media.
    .. [2] McKinney, W. (2012). *Python for Data Analysis: Data Wrangling with 
           Pandas, NumPy, and IPython*. O'Reilly Media.
    .. [3] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in 
           Python. *Journal of Machine Learning Research*, 12, 2825–2830.
    """
    if not is_iterable (arr, exclude_string=True): 
        raise TypeError(f"Expect array. Got {type (arr).__name__!r}")

    if hasattr ( arr, '__array__') and hasattr ( arr, 'columns'): 
        df = arr.copy()
        if columns is not None: 
            if verbose: 
                print("Dataframe is passed. Columns should be replaced.")
            df =pd.DataFrame ( np.array (arr), columns =columns )
            
    else: df = pd.DataFrame (arr, columns =columns  ) 
        
    # sanitize columns 
    if sanitize_columns: 
        # Pass in the case columns are all integer values. 
        if not is_numeric_dtype(df.columns , to_array=True): 
           # for consistency reconvert to str 
           df.columns = df.columns.astype(str) 
           df = sanitize_frame_cols(
               df, regex=regex, fill_pattern=fill_pattern ) 

    #replace empty string by Nan if NaN exist in dataframe  
    df= df.replace(r'^\s*$', missing_values, regex=True)
    
    # check the possibililty to cast all 
    # the numerical data 
    for serie in df.columns: 
        try: 
            df= df.astype(
                {serie:np.float64})
        except:continue
    
    # drop nan  columns if exists 
    if drop_nan_columns: 
        if verbose: 
            nan_columns = df.columns [ df.isna().all()].tolist() 
            print("No NaN column found.") if len(
                nan_columns)==0 else listing_items_format (nan_columns, 
                    "NaN columns found in the data",
                    " ", inline =True, lstyle='.')                               
        # drop rows and columns with NaN values everywhere.                                                   
        df.dropna (axis=1, how='all', inplace =True)
        if str(how).lower()=='all': 
            df.dropna ( axis=0, how='all', inplace =True)
    
    # reset_index of the dataframe
    # This is useful after droping rows
    if reset_index: 
        df.reset_index (inplace =True, drop = drop_index )
    # collect numeric and non-numeric data 
    nf, cf =[], []    
    for serie in df.columns: 
        if is_numeric_dtype(df[serie], to_array =True ): 
            nf.append(serie)
        else: cf.append(serie)

    if pop_cat_features: 
        [ df.pop(item) for item in cf ] 
        if verbose: 
            msg ="Dataframe does not contain any categorial features."
            b= f"Feature{'s' if len(cf)>1 else ''}"
            e = (f"{'have' if len(cf) >1 else 'has'} been dropped"
                 " from the dataframe.")
            print(msg) if len(cf)==0 else listing_items_format (
                cf , b, e ,lstyle ='.', inline=True)
            
        return df 
    
    return (df, nf, cf) if return_feature_types else df 

def denormalize(
    data: ArrayLike, min_value: float, max_value: float
    ) -> ArrayLike:
    """
    Denormalizes data from a normalized scale back to its original scale.

    This function is useful when data has been normalized to a different 
    scale (e.g., [0, 1]) and needs to be converted back to its original scale 
    for interpretation or further processing.

    Parameters
    ----------
    data : np.ndarray
        The data to be denormalized, assumed to be a NumPy array.
    min_value : float
        The minimum value of the original scale before normalization.
    max_value : float
        The maximum value of the original scale before normalization.

    Returns
    -------
    np.ndarray
        The denormalized data, converted back to its original scale.

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.core.array_manager import denormalize
    >>> normalized_data = np.array([0, 0.5, 1])
    >>> min_value = 10
    >>> max_value = 20
    >>> denormalized_data = denormalize(normalized_data, min_value, max_value)
    >>> print(denormalized_data)
    [10. 15. 20.]

    Note
    ----
    The denormalization process is the inverse of normalization and is applied
    to data that was previously normalized according to the formula:
        `data_norm = (data - min_value) / (max_value - min_value)`
    The denormalize function uses the inverse of this formula to restore the data.
    """
    if not isinstance (data, (pd.Series, pd.DataFrame)): 
        data = np.asarray( data )
        
    return data * (max_value - min_value) + min_value

 
def convert_to_structured_format(
    *arrays: Any, as_frame: bool = True, 
    skip_sparse: bool =True,
    cols_as_str: bool=False, 
    solo_return: bool=False, 
    ) -> List[Union[ArrayLike, DataFrame, Series]]:
    """
    Converts input objects to structured numpy arrays or pandas DataFrame/Series
    based on their shapes and the `as_frame` flag. If conversion to a structured
    format fails, the original objects are returned. When `as_frame` is False,
    attempts are made to convert inputs to numpy arrays.
    
    Parameters
    ----------
    *arrays : Any
        A variable number of objects to potentially convert. These can be lists,
        tuples, or numpy arrays.
    as_frame : bool, default=True
        If True, attempts to convert arrays to DataFrame or Series; otherwise,
        attempts to standardize as numpy arrays.
    skip_sparse: bool, default=True 
        Dont convert any sparse matrix and keept it as is. 
    cols_as_str: bool, default=False 
       If ``True``, convert numeric columns to string when pandas dataframe 
       is created. This is useful to avoid operations with litteral columns 
       already set as string or object dtype. 
    solo_return : bool, default=False
        If ``True`` and exactly one array, the function returns that array
        directly rather than as a tuple of length 1. If multiple
        arrays are provided then `solo_return` does no work even is ``True``.
       
    Returns
    -------
    List[Union[np.ndarray, pd.DataFrame, pd.Series]]
        A list containing the original objects, numpy arrays, DataFrames, or
        Series, depending on each object's structure and the `as_frame` flag.
    
    Examples
    --------
    Converting to pandas DataFrame/Series:
    >>> from fusionlab.core.array_manager import convert_to_structured_format
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> features= {"feature_1": range (7), "feature_2":['1', 2, 9, 35, "0", "76", 'r']}
    >>> target= pd.Series(data=range(10), name="target")
    >>> convert_to_structured_format( features, target, as_frame=True)
    >>> arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> arr2 = np.array([7, 8, 9])
    >>> convert_to_structured_format(arr1, arr2, as_frame=True)
    [   DataFrame:
            0  1  2
        0   1  2  3
        1   4  5  6,
        Series:
        0    7
        1    8
        2    9
    ]

    Standardizing as numpy arrays:
    >>> list1 = [10, 11, 12]
    >>> tuple1 = (13, 14, 15)
    >>> convert_to_structured_format(list1, tuple1, as_frame=False)
    [   array([10, 11, 12]),
        array([13, 14, 15])
    ]
    """

    def attempt_conversion_to_numpy(arr: Any) -> np.ndarray:
        """Attempts to convert an object to a numpy array."""
        try:
            arr= np.array(arr)
        except Exception:
            return arr
        
        if arr.ndim==2 and arr.shape[1] ==1: 
            arr = arr.flatten () 
        
        return arr 

    def attempt_conversion_to_pandas(
            arr: np.ndarray) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """Attempts to convert an array to a DataFrame or Series based on shape."""
        from scipy.sparse import issparse
        try:
            if issparse(arr) and skip_sparse: 
                raise # dont perform any convertion 
            if hasattr(arr, '__array__'): 
                if arr.ndim == 1:
                    return pd.Series(arr)
                elif arr.ndim == 2:
                    if arr.shape[1] == 1:
                        return pd.Series(arr.squeeze())
                    else:
                        arr= pd.DataFrame(arr)
                        if cols_as_str: 
                            arr.columns = arr.columns.astype(str) 
                        return arr 
            else: 
                arr= pd.DataFrame(arr)
                if cols_as_str: 
                    arr.columns = arr.columns.astype(str) 
                    
        except Exception:
            pass
        return arr

    if as_frame:
        arrays= [attempt_conversion_to_pandas(arr) for arr in arrays]
    else:
        # Try to convert everything to numpy arrays, return as is if it fails
        arrays= [attempt_conversion_to_numpy(attempt_conversion_to_pandas(arr)
                                            ) for arr in arrays]
    if solo_return and len(arrays)==1: 
        return arrays[0]
    
    return arrays 

def map_specific_columns ( 
    data: DataFrame, 
    ufunc:_F , 
    columns_to_skip:List[str]=None,   
    pattern:str=None, 
    inplace:bool= False, 
    **kws
    ): 
    """ Apply function to a specific columns is the dataframe. 
    
    It is possible to skip some columns that we want operation to not be 
    performed.
    
    Parameters 
    -----------
    data: dataframe, 
        pandas dataframe with valid columns 
    ufunc: callable, 
        Universal function that can be applying to the dataframe. 
    columns_to_skip: list or str , 
        List of columns to skip. If given as string and separed by the default
        pattern items, it should be converted to a list and make sure the 
        columns name exist in the dataframe. Otherwise an error with 
        raise.
        
    pattern: str, default = '[#&*@!,;\s]\s*'
        The base pattern to split the text in `column2skip` into a columns
        For instance, the following string coulb be splitted to:: 
            
            'depth_top, thickness, sp, gamma_gamma' -> 
            ['depth_top', 'thickness', 'sp', 'gamma_gamma']
        
        Refer to :func:`~.str2columns` for further details. 
    inplace: bool, default=True 
        Modified dataframe in place and return None, otherwise return a 
        new dataframe 
    kws: dict, 
        Keywords argument passed to :func: `pandas.DataFrame.apply` function 
        
    Returns 
    ---------
    X: Dataframe or None 
        Dataframe modified inplace with values computed using the given 
        `func`except the skipped columns, or ``None`` if `inplace` is ``True``. 
        
    Examples 
    ---------
    >>> from fusionlab.datasets import load_hlogs 
    >>> from fusionlab.utils.plotutils import map_specific_columns 
    >>> X0, _= load_hlogs (as_frame =True ) 
    >>> # let visualize the  first3 values of `sp` and `resistivity` keys 
    >>> X0['sp'][:3] , X0['resistivity'][:3]  
    ... (0   -1.580000
         1   -1.580000
         2   -1.922632
         Name: sp, dtype: float64,
         0    15.919130
         1    16.000000
         2    24.422316
         Name: resistivity, dtype: float64)
    >>> column2skip = ['hole_id','depth_top', 'depth_bottom', 
                      'strata_name', 'rock_name', 'well_diameter', 'sp']
    >>> map_specific_columns (X0, ufunc = np.log10, column2skip)
    >>> # now let visualize the same keys values 
    >>> X0['sp'][:3] , X0['resistivity'][:3]
    ... (0   -1.580000
         1   -1.580000
         2   -1.922632
         Name: sp, dtype: float64,
         0    1.201919
         1    1.204120
         2    1.387787
         Name: resistivity, dtype: float64)
    >>> # it is obvious the `resistiviy` values is log10 
    >>> # while `sp` stil remains the same 
      
    """
    X = _assert_all_types(data, pd.DataFrame)
    if not callable(ufunc): 
        raise TypeError ("Expect a function for `ufunc`; "
                         f"got {type(ufunc).__name__!r}")
        
    pattern = pattern or r'[#&*@!,;\s]\s*'
    if not is_iterable( columns_to_skip): 
        raise TypeError ("Columns  to skip expect an iterable object;"
                         f" got {type(columns_to_skip).__name__!r}")
        
    if isinstance(columns_to_skip, str):
        columns_to_skip = str2columns (columns_to_skip, pattern=pattern  )
    #assert whether column to skip is in 
    if columns_to_skip:
        cskip = copy.deepcopy(columns_to_skip)
        columns_to_skip = is_in_if(X.columns, columns_to_skip, return_diff= True)
        if len(columns_to_skip) ==len (X.columns): 
            warnings.warn("Value(s) to skip are not detected.")
    elif columns_to_skip is None: 
        columns_to_skip = list(X.columns) 
        
    if inplace : 
        X[columns_to_skip] = X[columns_to_skip].apply (
            ufunc , **kws)
        X.drop (columns = cskip , inplace =True )
        return 
    if not inplace: 
        X0 = X.copy() 
        X0[columns_to_skip] = X0[columns_to_skip].apply (
            ufunc , **kws)
    
        return  X0   
    
def concat_array_from_list(list_of_array, concat_axis=0):
    """
    Concatenate arrays from a list and replace `None` values with `NaN`.

    Parameters
    ----------
    list_of_array : list of array-like
        A list containing arrays that will be concatenated. If an element 
        is `None`, it will be replaced with a `NaN` array.
    concat_axis : int, optional
        The axis along which to concatenate the arrays. Must be either `0` 
        (rows) or `1` (columns). Default is `0`.

    Returns
    -------
    np.ndarray
        The concatenated array with shape determined by the length of the 
        arrays in the list and the specified `concat_axis`.

    Raises
    ------
    ValueError
        If `concat_axis` is not either `0` or `1`.

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.core.array_manager import concat_array_from_list
    >>> np.random.seed(0)
    >>> ass = np.random.randn(10)
    >>> ass2 = np.linspace(0, 15, 10)
    >>> concat_array_from_list([ass, ass2])
    array([[ 1.76405235,  0.        ],
           [ 0.40015721,  1.66666667],
           [ 0.97873798,  3.33333333],
           [ 2.2408932 ,  5.        ],
           [ 1.86755799,  6.66666667],
           [ -0.97727788,  8.33333333],
           [ 0.95008842, 10.        ],
           [ -0.15135721, 11.66666667],
           [ -0.10321885, 13.33333333],
           [ 0.4105985 , 15.        ]])

    Notes
    -----
    - The function will reshape one-dimensional arrays to two 
      dimensions if necessary, depending on the `concat_axis` value.
    - If the list contains only one array, it will return the reshaped 
      array without concatenation.
    """
    concat_axis = int(_assert_all_types(concat_axis, int, float))
    
    if concat_axis not in (0, 1):
        raise ValueError(f'Unable to understand axis: {str(concat_axis)!r}')
    
    # Replace None with NaN arrays
    list_of_array = list(map(
        lambda e: np.array([np.nan]) if e is None else np.array(e), 
        list_of_array
    ))

    # If the list has only one element, reshape it accordingly
    if len(list_of_array) == 1:
        ar = (list_of_array[0].reshape(
                (1, len(list_of_array[0]))
            ) if concat_axis == 0 else list_of_array[0].reshape(
                (len(list_of_array[0]), 1)
            )
        ) if list_of_array[0].ndim == 1 else list_of_array[0]
        return ar

    # Reshape arrays if necessary before concatenation
    list_of_array = list(map(
        lambda e: e.reshape(e.shape[0], 1) if e.ndim == 1 else e, 
        list_of_array
    )) if concat_axis == 1 else list(map(
        lambda e: e.reshape(1, e.shape[0]) if e.ndim == 1 else e, 
        list_of_array
    ))

    return np.concatenate(list_of_array, axis=concat_axis)

def drop_nan_in(
    *arrays: Union[np.ndarray, pd.DataFrame],
    ref_array: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    columns: Optional[List[str]] = None,
    reset_index: bool = True,
    axis: Union[int, str] = 0,
    error: str = 'raise', 
    solo_return=False, 
) -> List[Union[np.ndarray, pd.DataFrame]]:
    """
    Drop rows or columns containing NaNs across multiple arrays consistently.

    This function ensures that all provided arrays (either NumPy
    ndarrays or pandas DataFrames) have rows or columns with missing
    values (`NaN`) removed in a synchronized manner. If a `ref_array` is
    specified, the function bases the removal of rows or columns on the
    presence of `NaN` values within the reference array alone. Otherwise,
    it considers all provided arrays to determine where `NaN` values
    exist.

    Parameters
    ----------
    *arrays : Union[np.ndarray, pd.DataFrame]
        Multiple arrays to process. Each array must be either a NumPy
        ndarray or a pandas DataFrame. All arrays must have the same
        shape along the specified `axis`.
    
    ref_array : Union[np.ndarray, pd.DataFrame], optional
        Reference array to determine which rows or columns to drop based
        on `NaN` values. If provided, `NaN` values are checked only in
        this array. All arrays, including `ref_array`, must share the
        same columns if `axis=1` or the same index if `axis=0`.
    
    columns : List[str], optional
        Specific columns to check for `NaN` values. Applicable only if
        the arrays are pandas DataFrames. If specified, `NaN` values are
        checked only within these columns. All specified columns must
        exist in each DataFrame.
    
    reset_index : bool, default=True
        Whether to reset the index of the DataFrames before and after dropping
        rows. This parameter is ignored for NumPy ndarrays after dropping rows.
        If set to `True`, the resulting DataFrames will have a new integer 
        index.
    
    axis : int or str, default=0
        Axis along which to drop `NaN` values.
        
        - `0` or `'rows'`: Drop entire rows that contain `NaN` values.
        - `1` or `'columns'`: Drop entire columns that contain `NaN` values.
        
        .. math::
            \text{axis} = 
            \begin{cases}
                0 & \text{if dropping rows} \\
                1 & \text{if dropping columns}
            \end{cases}
    
    error : str, default='raise'
        Specifies how to handle errors when indices or columns to be
        dropped are not found in some arrays.
        
        - `'raise'`: Raise a `ValueError` if an inconsistency is found.
        - `'warn'`: Issue a warning and skip dropping for arrays where
          indices or columns are not found.
        - `'ignore'`: Silently skip dropping without raising errors or
          issuing warnings.
        
        .. math::
            \text{error} = 
            \begin{cases}
                \text{'raise'} & \text{if strict error handling is desired} \\
                \text{'warn'} & \text{to notify about inconsistencies} \\
                \text{'ignore'} & \text{to proceed without notifications}
            \end{cases}
            
    solo_return : bool, default=False
        If ``True`` and exactly one array, the function returns that array
        directly rather than as a tuple of length 1. If multiple
        arrays are provided or `solo_return` does no work. 
        
    Returns
    -------
    List[Union[np.ndarray, pd.DataFrame]]
        A list of cleaned arrays with `NaN` rows or columns removed
        according to the specified parameters. The returned arrays retain
        their original data types (NumPy ndarray or pandas DataFrame).
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from fusionlab.core.array_manager import drop_nan_in
    >>> 
    >>> # Sample DataFrames and ndarray
    >>> df1 = pd.DataFrame({
    ...     'A': [1, 2, np.nan, 4],
    ...     'B': [5, np.nan, 7, 8],
    ...     'C': [9, 10, 11, 12]
    ... })
    >>> 
    >>> df2 = pd.DataFrame({
    ...     'A': [13, 14, 15, 16],
    ...     'B': [17, 18, np.nan, 20],
    ...     'C': [21, 22, 23, 24]
    ... })
    >>> 
    >>> array1 = np.array([
    ...     [25, 26, 27],
    ...     [28, np.nan, 30],
    ...     [31, 32, 33],
    ...     [34, 35, 36]
    ... ])
    >>> 
    >>> # Drop rows with NaNs in columns 'A' and 'B' across all arrays
    >>> cleaned_df1, cleaned_df2, cleaned_array1 = drop_nan_in(
    ...     df1, df2, array1,
    ...     axis=0,
    ...     error='warn'
    ... )
    >>> 
    >>> print("Cleaned df1:")
    >>> print(cleaned_df1)
         A    B   C
    0  1.0  5.0   9
    3  4.0  8.0  12
    >>> 
    >>> print("\nCleaned df2:")
    >>> print(cleaned_df2)
        A     B   C
    0 13  17.0  21
    3 16  20.0  24
    >>> 
    >>> print("\nCleaned array1:")
    >>> print(cleaned_array1)
    [[25. 26. 27.]
     [34. 35. 36.]]
    
    Notes
    -----
    - **Consistency Across Arrays**: When multiple arrays are provided,
      `drop_nan_in` ensures that rows or columns are dropped uniformly
      across all arrays based on the presence of `NaN` values. This
      consistency is crucial for maintaining data integrity, especially
      when the arrays represent related datasets.
    
    - **Reference Array Usage**: By specifying a `ref_array`, users can
      dictate that `NaN` values are only checked within this array.
      Consequently, only the rows or columns identified as containing
      `NaN` in the `ref_array` will be removed from all other arrays.
    
    - **Selective Column Checking**: The `columns` parameter allows
      for targeted `NaN` checks within specific columns of DataFrames.
      This feature is beneficial when only certain variables are
      critical for analysis, and `NaN` values in other columns can be
      tolerated.
    
    - **Index Resetting**: Setting `reset_index=True` ensures that
      DataFrames have a clean, sequential index after rows are dropped.
      This is particularly useful for downstream processing and analysis.
    
    - **Error Handling Flexibility**: The `error` parameter provides
      users with control over how the function responds to potential
      inconsistencies, such as mismatched indices or missing columns
      across arrays. This flexibility allows for smoother integration
      into various data processing pipelines.
    
    See Also
    --------
    - :func:`pandas.DataFrame.drop` : Drop specified labels from rows or
      columns.
    - :func:`numpy.isnan` : Return a boolean array indicating whether
      each element is `NaN`.
    
    References
    ----------
    .. [1] McKinney, Wes. "Data Structures for Statistical Computing in
       Python." Proceedings of the 9th Python in Science Conference, 
       2010.
    .. [2] Harris, Charles R., et al. "Array programming with NumPy."
       Nature, vol. 585, no. 7825, 2020, pp. 357–362.
    """
    # Validate and standardize the axis parameter
    axis = validate_axis(axis , accept_axis_none= False)

    # Validate the error handling parameter
    error = error_policy(error)
    
    # Ensure all arrays have the same size along the specified axis
    arrays = ensure_same_shape(
        *arrays, axis = axis, ops ='validate', solo_return=False )
    
    # Process the reference array if provided
    ref_df, ref_is_df = _process_reference_array(ref_array, axis, reset_index)

    # Convert all input arrays to DataFrames for uniform processing
    df_arrays, original_types = _convert_arrays_to_df(arrays, reset_index)

    # If ref_array is provided and not a DataFrame, convert it to DataFrame
    if ref_df is not None and not ref_is_df:
        ref_df = pd.DataFrame(ref_df)

    # Determine which rows or columns contain NaNs based on the criteria
    na_mask = _determine_na_mask(
        df_arrays, ref_df, ref_is_df, columns, axis
    )
    # Identify the specific labels (indices or columns) to drop
    labels_to_drop = na_mask[na_mask].index 

    # Drop the identified labels from each DataFrame
    processed_dfs = [
        _drop_labels_from_df(
            df, labels_to_drop, 
            axis, 
            reset_index, 
            ref_df, 
            ref_is_df, 
            error, 
            df_arrays, 
        ) for df in df_arrays
    ]

    # Convert the processed DataFrames back to their original types
    final_arrays = _convert_back_to_original_types(
        processed_dfs, original_types, solo_return, 
    )

    return final_arrays

def _process_reference_array(
    ref_array: Optional[Union[np.ndarray, pd.DataFrame]],
    axis: int, 
    reset_index : bool, 
) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Process the reference array and determine its type.
    """
    if isinstance (ref_array, (list, tuple)): 
        ref_array = np.asarray( ref_array)
        
    if ref_array is not None:
        if isinstance(ref_array, pd.DataFrame):
            if reset_index: 
                ref_array.reset_index(drop=True, inplace =True)
            return ref_array.copy(), True
        elif isinstance(ref_array, np.ndarray):
            flattened = ref_array.flatten() if axis == 0 else ref_array
            return flattened, False
        else:
            raise TypeError(
                "ref_array must be a pandas DataFrame or a numpy ndarray"
            )
    return None, False


def _determine_na_mask(
    df_arrays: List[pd.DataFrame],
    ref_df: Optional[pd.DataFrame],
    ref_is_df: bool,
    columns: Optional[List[str]],
    axis: int
) -> pd.Series:
    """
    Determine which rows or columns contain NaNs based on the criteria.
    """
    # Map the main axis to pandas axis
    pandas_axis = 1 if axis == 0 else 0

    if ref_df is not None:
        if ref_is_df:
            if columns is not None:
                # Check if all specified columns exist in the reference DataFrame
                missing_cols = [
                    col for col in columns if col not in ref_df.columns
                ]
                if missing_cols:
                    raise ValueError(
                        f"Columns {missing_cols} not found in reference DataFrame"
                    )
                # Identify rows or columns with NaNs in specified columns
                na_mask = ref_df[columns].isna().any(axis=pandas_axis)
            else:
                # Identify rows or columns with any NaNs in the reference DataFrame
                na_mask = ref_df.isna().any(axis=pandas_axis)
        else:
            if columns is not None:
                raise ValueError(
                    "columns parameter is only applicable when ref_array is a DataFrame"
                )
            # Identify rows or columns with NaNs in the reference ndarray
            na_mask = pd.Series(
                np.isnan(ref_df) if axis == 0 else np.isnan(ref_df).any(axis=0)
            )
    else:
        # Combine NaN masks from all arrays
        na_masks = []
        for df in df_arrays:
            if columns is not None:
                # Check if all specified columns exist in the current DataFrame
                missing_cols = [
                    col for col in columns if col not in df.columns
                ]
                if missing_cols:
                    raise ValueError(
                        f"Columns {missing_cols} not found in one of the DataFrames"
                    )
                # Identify rows or columns with NaNs in specified columns
                na_masks.append(df[columns].isna().any(axis=pandas_axis))
            else:
                # Identify rows or columns with any NaNs in the current DataFrame
                na_masks.append(df.isna().any(axis=pandas_axis))
        # Combine all masks using logical OR to identify overall NaNs
        na_mask = pd.concat(na_masks, axis=1).any(axis=0 if axis == 1 else 1)
    return na_mask

def _drop_labels_from_df(
    df: pd.DataFrame,
    labels_to_drop: pd.Index,
    axis: int,
    reset_index: bool,
    ref_df: Optional[pd.DataFrame],
    ref_is_df: bool,
    error: str,
    df_arrays:List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Drop the specified labels from the DataFrame based on the axis.
    """
    try:
        if axis == 0:
            # Dropping rows
            if reset_index:
                # Drop rows and reset index
                return df.drop(labels=labels_to_drop, axis=0).reset_index(drop=True)
     
            if ref_df is not None:
                # Define the expected index based on the reference
                expected_index = ref_df.index if ref_is_df else pd.RangeIndex(
                    start=0, stop=len(df), step=1
                )
                if not df.index.equals(expected_index):
                    if error == 'raise':
                        raise ValueError(
                            "Indices of all arrays must match when "
                            "reset_index is False and ref_array is provided"
                        )
                    elif error == 'warn':
                        warnings.warn(
                            "Indices do not match. Skipping drop for this array.",
                            UserWarning
                        )
                        return df.copy()
                    elif error == 'ignore':
                        return df.copy()
                # Drop rows based on labels_to_drop
                return df.drop(labels=labels_to_drop, axis=0)
            else:
                # Without ref_array, assume all DataFrames have the same index
                if not df.index.equals(df_arrays[0].index):
                    if error == 'raise':
                        raise ValueError(
                            "All arrays must have the same index when reset_index is False"
                        )
                    elif error == 'warn':
                        warnings.warn(
                            "Indices do not match. Skipping drop for this array.",
                            UserWarning
                        )
                        return df.copy()
                    elif error == 'ignore':
                        return df.copy()
                # Drop rows based on labels_to_drop
                return df.drop(labels=labels_to_drop, axis=0)
        else:
            # Dropping columns
            if reset_index:
                # Drop columns and reset index
                return df.drop(labels=labels_to_drop, axis=1).reset_index(drop=True)
     
            if ref_df is not None:
                # Ensure columns to drop exist in the current DataFrame
                existing_cols = labels_to_drop.intersection(df.columns)
                missing_cols = labels_to_drop.difference(df.columns)
                if missing_cols:
                    if error == 'raise':
                        raise ValueError(
                            "Columns to drop not found in one of the "
                            "DataFrames when reset_index is False"
                        )
                    elif error == 'warn':
                        warnings.warn(
                            f"Some columns to drop not found: {missing_cols}. "
                            f"Dropping existing columns: {existing_cols}",
                            UserWarning
                        )
                # Drop existing columns based on labels_to_drop
                return df.drop(labels=existing_cols, axis=1)
            else:
                # Without ref_array, assume all DataFrames have the same columns
                if not df.columns.equals(df_arrays[0].columns):
                    if error == 'raise':
                        raise ValueError(
                            "All arrays must have the same columns when reset_index is False"
                        )
                    elif error == 'warn':
                        warnings.warn(
                            "Columns do not match. Skipping drop for this array.",
                            UserWarning
                        )
                        return df.copy()
                    elif error == 'ignore':
                        return df.copy()
                # Drop columns based on labels_to_drop
                return df.drop(labels=labels_to_drop, axis=1)
    except KeyError as e:
        # Handle KeyError based on the error parameter
        if error == 'raise':
            raise ValueError(f"Key error while dropping: {e}")
        elif error == 'warn':
            warnings.warn(
                f"Key error while dropping: {e}. Skipping drop for this array.",
                UserWarning
            )
            return df.copy()
        elif error == 'ignore':
            return df.copy()

def _convert_arrays_to_df(
    arrays: Tuple[Union[np.ndarray, pd.DataFrame, pd.Series], ...], 
    reset_index: bool=True , 
) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Convert all input arrays to DataFrames and track their original types.

    Parameters
    ----------
    arrays : Tuple[Union[np.ndarray, pd.DataFrame, pd.Series], ...]
        A tuple of arrays to be converted. Each array can be a NumPy ndarray,
        pandas DataFrame, or pandas Series.

    Returns
    -------
    Tuple[List[pd.DataFrame], List[str]]
        A tuple containing:
        - A list of pandas DataFrames converted from the input arrays.
        - A list of strings indicating the original type of each array
          ('df' for DataFrame, 'ndarray' for NumPy ndarray, 'series' for Series).
    """
    df_arrays = []
    original_types = []
    for arr in arrays:
        if isinstance (arr, (list, tuple)): 
            arr = np.asarray(arr)
            
        if isinstance(arr, pd.DataFrame):
            # If the array is already a DataFrame, make a copy to avoid
            # modifying the original data.
            df_arrays.append(arr.copy())
            original_types.append('df')
        elif isinstance(arr, pd.Series):
            # Convert Series to DataFrame with the Series name as the column name.
            df_arrays.append(arr.to_frame())
            original_types.append('series')
        elif isinstance(arr, np.ndarray):
            # Convert NumPy ndarray to DataFrame.
            df = pd.DataFrame(arr)
            df_arrays.append(df)
            original_types.append('ndarray')
        else:
            # Raise an error if the array type is unsupported.
            raise TypeError(
                "All arrays must be pandas DataFrames,"
                " Series, or numpy ndarrays"
            )
    if reset_index: 
        # If reset index, reset index to avoid mismatching 
        df_arrays = [df.reset_index (drop =True ) for df in df_arrays]
        
    return df_arrays, original_types

def _convert_back_to_original_types(
    processed_dfs: List[pd.DataFrame],
    original_types: List[str], 
    solo_return: bool=False, 
) -> List[Union[np.ndarray, pd.DataFrame, pd.Series]]:
    """
    Convert the processed DataFrames back to their original types.

    Parameters
    ----------
    processed_dfs : List[pd.DataFrame]
        A list of pandas DataFrames after processing (dropping NaNs).

    original_types : List[str]
        A list indicating the original type of each DataFrame
        ('df' for DataFrame, 'ndarray' for NumPy ndarray, 'series' for Series).
        
    solo_return: bool, Default=False 
       If ``True``, remove  the single array from the final processed list 
       and return it.

    Returns
    -------
    List[Union[np.ndarray, pd.DataFrame, pd.Series]]
        A list of arrays converted back to their original types.
        - DataFrames remain as pandas DataFrames.
        - NumPy ndarrays are converted back to ndarrays.
        - Series are converted back to pandas Series.

    Raises
    ------
    ValueError
        If a DataFrame cannot be converted back to a Series due to multiple
        columns.
    """
    final_arrays = []
    for df, original_type in zip(processed_dfs, original_types):
        if original_type == 'df':
            # If original was DataFrame, retain as DataFrame.
            final_arrays.append(df)
        elif original_type == 'ndarray':
            # If original was ndarray, convert DataFrame back to ndarray.
            final_arrays.append(df.to_numpy())
        elif original_type == 'series':
            # If original was Series, convert DataFrame back to Series.
            if df.shape[1] == 1:
                # Successfully convert to Series if only one column exists.
                final_arrays.append(df.iloc[:, 0])
            elif df.shape[1] == 0:
                # If all columns were dropped, return an empty
                # Series with the original name.
                final_arrays.append(
                    pd.Series([], name=df.columns[0] if not df.empty else None))
            else:
                # Raise an error if multiple columns exist, cannot convert to Series.
                raise ValueError(
                    "Cannot convert DataFrame with multiple columns back to Series."
                )
        else:
            # This condition should not occur; added for safety.
            raise TypeError(
                f"Unsupported original type '{original_type}'."
            )
    if solo_return and len(final_arrays)==1: 
        return final_arrays [0]
    
    return final_arrays

def array_preserver(
    *arrays: Any,
    action: str = 'collect',
    error: str = 'warn',
    deep_restore: bool = False, 
    solo_return: bool=True, 
) -> Union[Dict[str, List[Any]], List[Any]]:
    """
    Collect and restore array-like objects while preserving their 
    original properties.

    The ``array_preserver`` function facilitates the management of multiple 
    array-like objects by enabling their collection and restoration. This 
    ensures that the original data types and properties are maintained 
    throughout processing workflows.

    .. math::
        \text{Collected Data} = \text{array\_preserver}\\
            (\text{arrays}, \text{action}='collect')

    .. math::
        \text{Restored Data} = \text{array\_preserver}\\
            (\text{Collected Data}, \text{action}='restore')

    Parameters
    ----------
    *arrays : Any
        Variable length argument list of array-like objects to be collected 
        or restored. Supported types include lists, tuples, NumPy 
        arrays, Pandas Series, and Pandas DataFrames.

    action : str, default='collect'
        Specifies the operation mode of the function. Accepted values are:
        
        - ``'collect'``: Collects the input arrays, preserving their original 
          properties and storing associated metadata.
        - ``'restore'``: Restores the processed arrays back to their original 
          types using the provided metadata.
        
        The ``action`` parameter dictates whether the function is in collection 
        mode or restoration mode.

    error : str, default='warn'
        Defines the behavior when restoration fails or encounters inconsistencies. 
        Options include:
        
        - ``'raise'``: Raise a ``ValueError`` when restoration fails.
        - ``'warn'``: Issue a warning but continue execution, returning the 
          partially restored data.
        - ``'ignore'``: Silently ignore errors and proceed without raising or 
          warning.
        
        This parameter allows users to control the strictness of error handling 
        during the restoration process.

    deep_restore : bool, default=False
        Applicable only when ``action='restore'``. If ``True``, the function 
        attempts to restore additional properties such as indices and column 
        names for Pandas DataFrames and Series. This ensures a more comprehensive 
        restoration of the original objects.
        
        When ``deep_restore`` is enabled, the restored DataFrames and Series 
        will retain their original indices and column names, providing a 
        faithful reconstruction of the original data structures.
        
    solo_return : bool, default=True
        If ``True`` and exactly one array, the function returns that array
        directly rather than as a tuple of length 1. If multiple
        arrays are provided or `solo_return` does no work. 
        
    Returns
    -------
    Union[Dict[str, List[Any]], List[Any]]
        - When ``action='collect'``: Returns a dictionary containing two keys:
        
            - ``'processed'``: A list of the original array-like objects.
            - ``'metadata'``: A list of metadata dictionaries corresponding 
              to each processed object, preserving original properties.
        
        - When ``action='restore'``: Returns a list of restored array-like 
          objects in their original types and with their original properties.

    .. math::
        \text{Restored Array} = f(\text{Processed Array}, \text{Metadata})

    Examples
    --------
    >>> from fusionlab.core.array_manager import array_preserver
    >>> import numpy as np
    >>> import pandas as pd
    >>> 
    >>> # Collect multiple array-like objects
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> series = pd.Series([5, 6], name='C')
    >>> list_input = [7, 8, 9]
    >>> collected = array_preserver(df, series, list_input, action='collect')
    >>> 
    >>> # External processing (e.g., converting DataFrame to NumPy array)
    >>> processed_df = collected['processed'][0].values * 2  # Example processing
    >>> processed_series = collected['processed'][1].values + 1
    >>> processed_list = [x * 3 for x in collected['processed'][2]]
    >>> collected['processed'] = [processed_df, processed_series, processed_list]
    >>> 
    >>> # Restore the original array-like objects
    >>> restored = array_preserver(collected, action='restore', deep_restore=True)
    >>> 
    >>> print(restored[0])  # Restored DataFrame
       A  B
    0  2  6
    1  4  8
    >>> print(restored[1])  # Restored Series
    0    6
    1    7
    Name: C, dtype: int64
    >>> print(restored[2])  # Restored list
    [21, 24, 27]
    
    >>> # Handling restoration errors with 'raise'
    >>> try:
    ...     invalid_collected = {'processed': [np.array([10, 11])], 'metadata': [{'type': 'Unknown'}]}
    ...     restored_invalid = array_preserver(invalid_collected, action='restore', error='raise')
    ... except ValueError as e:
    ...     print(e)
    Failed to restore Unknown: Cannot restore to original type 'Unknown'.
    
    Notes
    -----
    The ``array_preserver`` function is designed to streamline the management of 
    array-like data structures in data processing workflows. By separating the 
    collection and restoration phases, it ensures that the original data types 
    and properties are maintained, facilitating seamless integration with 
    various data manipulation and analysis tasks.
    
    - **Collection Phase**:
        - Gathers multiple array-like objects and records their essential 
          metadata, such as type, columns, indices, and shapes.
        - Stores the original arrays without altering their structure or content.
    
    - **Restoration Phase**:
        - Utilizes the stored metadata to accurately reconstruct the original 
          array-like objects.
        - Supports deep restoration to retain complex properties like indices 
          and column names for Pandas DataFrames and Series.
    
    This function is particularly useful in scenarios where data undergoes 
    transformations or processing steps that may alter its original structure. 
    By preserving the original properties, ``array_preserver`` ensures that 
    data can be accurately reverted to its initial state post-processing.

    See Also
    --------
    ``to_array`` : Converts a single array-like object to the desired dimensionality.
    ``PandasDataHandlers`` : Manages Pandas-based data parsing and writing functions.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. In *Proceedings of the 9th Python in Science Conference*, 
           51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. *Nature*, 585(7825), 357-362.
    .. [3] Pandas Development Team. (2023). *pandas documentation*. 
           https://pandas.pydata.org/pandas-docs/stable/
    """

    # Validate the 'action' parameter
    if action not in ['collect', 'restore']:
        raise ValueError("Invalid action. Choose 'collect' or 'restore'.")

    # Define acceptable error handling modes
    if error not in ['raise', 'warn', 'ignore']:
        raise ValueError(
            "Invalid error handling mode. Choose from 'raise', 'warn', or 'ignore'."
        )

    if action == 'collect':
        processed: List[Any] = []
        metadata: List[Dict[str, Any]] = []

        for arr in arrays:
            meta: Dict[str, Any] = {}

            # Determine the type and capture relevant metadata
            if isinstance(arr, pd.DataFrame):
                meta['type'] = 'DataFrame'
                meta['columns'] = arr.columns.tolist()
                meta['index'] = arr.index.tolist()

            elif isinstance(arr, pd.Series):
                meta['type'] = 'Series'
                meta['name'] = arr.name
                meta['index'] = arr.index.tolist()

            elif isinstance(arr, list):
                meta['type'] = 'list'

            elif isinstance(arr, tuple):
                meta['type'] = 'tuple'

            elif isinstance(arr, np.ndarray):
                meta['type'] = 'ndarray'
                meta['shape'] = arr.shape

            else:
                meta['type'] = type(arr).__name__

            # Store the array as-is
            processed_arr = arr.copy() if hasattr(arr, 'copy') else arr
            processed.append(processed_arr)
            metadata.append(meta)

        return {'processed': processed, 'metadata': metadata}

    elif action == 'restore':
        # Expect exactly one argument: the container with 'processed' and 'metadata'
        if len(arrays) != 1:
            raise ValueError(
                "For 'restore' action, provide a single container"
                " with 'processed' data and 'metadata'."
            )
        container = arrays[0]

        # Validate the container structure
        if not isinstance(container, dict):
            raise ValueError(
                "For 'restore' action, the container must be a dictionary"
                " with 'processed' and 'metadata' keys."
            )
        if 'processed' not in container or 'metadata' not in container:
            raise ValueError(
                "The container must have 'processed' and 'metadata' keys.")

        processed: List[Any] = container['processed']
        metadata: List[Dict[str, Any]] = container['metadata']

        if not isinstance(processed, list) or not isinstance(metadata, list):
            raise ValueError(
                "'processed' and 'metadata' must be lists within the container.")

        restored: List[Any] = []

        for proc_arr, meta in zip(processed, metadata):
            orig_type = meta.get('type')

            try:
                if orig_type == 'DataFrame':
                    # Restore Pandas DataFrame
                    columns = meta.get('columns')
                    index = meta.get('index') if deep_restore else None
                    df = pd.DataFrame(proc_arr, columns=columns)
                    if deep_restore and index is not None:
                        df.index = index
                    restored.append(df)

                elif orig_type == 'Series':
                    # Restore Pandas Series
                    name = meta.get('name')
                    index = meta.get('index') if deep_restore else None

                    if isinstance(proc_arr, pd.DataFrame) and proc_arr.shape[1] == 1:
                        # Convert single-column DataFrame back to Series
                        s = proc_arr.iloc[:, 0]
                        if deep_restore and index is not None:
                            s.index = index
                        s.name = name
                        restored.append(s)

                    elif isinstance(proc_arr, pd.Series):
                        # Directly copy Series
                        s = proc_arr.copy()
                        if deep_restore and index is not None:
                            s.index = index
                        s.name = name
                        restored.append(s)

                    elif isinstance(proc_arr, np.ndarray):
                        # Convert NumPy array to Series
                        s = pd.Series(proc_arr, name=name)
                        if deep_restore and index is not None:
                            s.index = index
                        restored.append(s)

                    elif isinstance(proc_arr, list):
                        # Convert list to Series
                        s = pd.Series(proc_arr, name=name)
                        if deep_restore and 'index' in meta:
                            s.index = meta['index']
                        restored.append(s)

                    else:
                        # Attempt to convert other types to Series
                        s = pd.Series(proc_arr, name=name)
                        if deep_restore and index is not None:
                            s.index = index
                        restored.append(s)

                elif orig_type == 'list':
                    # Restore list
                    restored.append(list(proc_arr))

                elif orig_type == 'tuple':
                    # Restore tuple
                    restored.append(tuple(proc_arr))

                elif orig_type == 'ndarray':
                    # Restore NumPy ndarray
                    restored.append(proc_arr)

                else:
                    # Attempt to restore unknown types as-is
                    restored.append(proc_arr)

            except Exception as e:
                message = f"Failed to restore {orig_type}: {e}"
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
                    restored.append(proc_arr)
                elif error == 'ignore':
                    restored.append(proc_arr)
        
        if solo_return and len(restored) ==1: 
            return restored[0] 
        
        return restored

def index_based_selector(
    *dfs,
    ref_df,
    reset_index: bool = False,
    error: str = 'raise',
    as_series: bool = True,
    return_ref: bool = False,
    check_size: bool = False
):
    """
    Align multiple dataframes to the index of a reference dataframe
    and return the indexed subsets.

    The ``index_based_selector`` function subsets each dataframe in
    ``*dfs`` based on the index of ``ref_df``. It ensures consistent
    indexing across multiple dataframes for further aligned analyses.
    Optionally, it resets indexes before selection and converts
    single-column dataframes to Series.

    .. math::
       \\text{Indexed Subset} = D \\bigl[ I_{ref} \\bigr]

    where:
    - :math:`D` represents a dataframe from ``*dfs``.
    - :math:`I_{ref}` is the index of the reference dataframe :math:`R`.
    - The operation selects rows from :math:`D` whose indices intersect
      with :math:`I_{ref}`.

    Parameters
    ----------
    *dfs : pandas.DataFrame
        One or more dataframes to be aligned to the index of
        ``ref_df``. These dataframes must be validated as real
        DataFrame objects. An error is raised if any is invalid
        unless otherwise specified by `check_size`.
    
    ref_df : pandas.DataFrame
        The reference dataframe whose index determines the row
        selection. Rows in each dataframe from ``*dfs`` are
        sub-selected to match this index.

    reset_index : bool, default=False
        If ``True``, reset the index of ``ref_df`` and all dataframes
        in ``*dfs`` before alignment. Useful when indexes need to
        start from zero or be standardized.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        Controls how to handle missing indices. If some reference
        indices are not found in a dataframe:
        
        - ``'raise'``: Halt and raise a ValueError.
        - ``'warn'``: Warn the user, skip the missing indices,
          and continue.
        - ``'ignore'``: Silently skip missing indices.

    as_series : bool, default=True
        If ``True``, any resulting dataframe with exactly one
        column is converted to a Series. The Series name is set
        to the column name, preserving data clarity.

    return_ref : bool, default=False
        If ``True``, the reference dataframe (potentially with a
        reset index) is included in the returned tuple. By default,
        only the aligned subsets of ``*dfs`` are returned.

    check_size : bool, default=False
        If ``True``, enforces that all dataframes in ``*dfs`` have
        the same length. An error is raised if any mismatch is
        detected. If ``False``, no size check is enforced.

    Returns
    -------
    pandas.DataFrame or pandas.Series or tuple
        - If only one dataframe is provided (and ``return_ref=False``),
          the function returns its indexed subset as a single object.
          If that subset has one column and ``as_series=True``, a
          Series is returned.
        - If multiple dataframes are provided, returns a tuple of
          their indexed subsets. Each subset is either a DataFrame
          or Series (if it had one column and ``as_series=True``).
        - If ``return_ref=True``, the reference dataframe (possibly
          reset) is appended to the tuple as the last element.

    Raises
    ------
    ValueError
        - If any input in ``*dfs`` is not a valid DataFrame when
          `check_size=True`.
        - If the reference index is not fully found in a dataframe
          under ``error='raise'``.
        - If a size mismatch is found among dataframes when
          `check_size=True`.

    Examples
    --------
    >>> from fusionlab.core.array_manager import index_based_selector
    >>> import pandas as pd
    >>> df1 = pd.DataFrame(
    ...     {'A': [1, 2, 3]},
    ...     index=[10, 11, 12]
    ... )
    >>> df2 = pd.DataFrame(
    ...     {'B': [4, 5, 6]},
    ...     index=[11, 12, 13]
    ... )
    >>> ref_df = pd.DataFrame(
    ...     {'C': [7, 8]},
    ...     index=[11, 12]
    ... )
    >>> aligned = index_based_selector(
    ...     df1, df2,
    ...     ref_df=ref_df,
    ...     reset_index=False,
    ...     error='warn',
    ...     as_series=True,
    ...     return_ref=True
    ... )
    >>> for item in aligned:
    ...     print(item)
    A
    11    2
    12    3
    Name: A, dtype: int64

    B
    11    5
    12    6
    Name: B, dtype: int64

       C
    11  7
    12  8

    Notes
    -----
    - Use `reset_index=True` when dataframes have misaligned or
      duplicate index values that need standardization.
    - Converting single-column DataFrames to Series can streamline
      further processing steps but may remove multi-column structure.

    See Also
    --------
    SomeOtherSelector : A related selector function that also
        works with indexes in Gofast package.

    References
    ----------
    .. [1] McKinney, W. "Python for Data Analysis: Data Wrangling with
           Pandas, NumPy, and IPython." O'Reilly, 2017.
    """
    # 1) Check if all provided `dfs` are valid DataFrames.
    #    Use `are_all_frame_valid` from `fusionlab.core.checks` to ensure
    #    they're DataFrames and optionally check if they have the same
    #    length if `check_size=True`. The `error='raise'` here means
    #    that any invalid condition immediately raises an error.
    
    dfs= are_all_frames_valid(
        *dfs,
        df_only=True,
        check_size=check_size, 
        to_df =True, 
        ops='validate', 
    )

    # 2) If `reset_index` is True, reset the index of `ref_df` and
    #    all dataframes in `dfs`.
    #    This step aligns indexes starting from zero before performing
    #    any index-based subset selection.
    if reset_index:
        ref_df = ref_df.reset_index(drop=True)
        new_dfs = []
        for df in dfs:
            new_dfs.append(df.reset_index(drop=True))
        dfs = tuple(new_dfs)

    # 3) Validate the `error` parameter for the subset selection logic.
    #    Acceptable values are `'raise'`, `'warn'`, `'ignore'`.
    error = error_policy(error)

    # 4) Subset each dataframe in `dfs` based on `ref_df`'s index.
    #    - If `error='raise'`, all indexes in `ref_df`'s index must be
    #      present in the target dataframe's index. If not, raise.
    #    - If `error='warn'`, warn user about missing indexes and
    #      skip them.
    #    - If `error='ignore'`, silently skip missing indexes.
    out_dfs = []
    ref_index = ref_df.index

    for idx, df in enumerate(dfs):
        df_idx = df.index
        missing_idx = ref_index.difference(df_idx)

        # If there are missing indexes, handle according to `error`.
        if not missing_idx.empty:
            if error == 'raise':
                raise ValueError(
                    f"Reference index not fully found in "
                    f"DataFrame at position {idx}. Missing "
                    f"indices: {list(missing_idx)}."
                )
            elif error == 'warn':
                warnings.warn(
                    f"Some indices from reference dataframe not "
                    f"found in DataFrame at position {idx}. "
                    f"Missing: {list(missing_idx)}. Those indices "
                    f"are skipped."
                )

        # Subset the dataframe by intersection with reference index.
        valid_index = ref_index.intersection(df_idx)
        df_subset = df.loc[valid_index]

        # If `as_series=True` and this dataframe has only one column,
        # convert it to a Series while preserving its name if possible.
        if as_series and df_subset.shape[1] == 1:
            col_name = df_subset.columns[0]
            s = df_subset[col_name]
            s.name = col_name
            out_dfs.append(s)
        else:
            out_dfs.append(df_subset)

    # 5) If `return_ref` is True, append the (possibly reset) `ref_df`
    #    to the results. Return it as the last item.
    if return_ref:
        out_dfs = tuple(list(out_dfs) + [ref_df])
    else:
        out_dfs = tuple(out_dfs)

    # 6) If there's only one dataframe in the input (`dfs` length == 1)
    #    and `return_ref=False`, return it directly to avoid returning
    #    a tuple of length one. Otherwise, return the tuple of results.
    if not return_ref and len(out_dfs) == 1:
        return out_dfs[0]

    return out_dfs

def to_series(
    data,
    name=None,
    handle_2d="raise", 
):
    """
    Convert the provided data to a one-dimensional pandas Series,
    respecting shapes and optional `name` assignment. This function,
    named `to_series`, aims to unify various array-like inputs
    (lists, tuples, NumPy arrays, or single-column pandas DataFrame)
    into a single coherent Series structure.

    .. math::
       y = \\alpha x + \\beta

    Here, :math:`y` represents the resulting Series of length
    :math:`x`, and :math:`\\alpha, \\beta` are conceptual
    scaling factors in transformations of the input data [1]_.

    Parameters
    ----------
    data : array-like, pandas.DataFrame, list, or tuple
        The input data to be converted into a one-dimensional
        Series. Supported formats include:
          - Python lists or tuples (converted to NumPy arrays first)
          - NumPy arrays of shape (n,) or reshaped from (1, n) or
            (n, 1) into (n,)
          - A pandas DataFrame with a single column
    name : str, optional
        A string used to rename the resulting Series. If not
        provided, the name is inferred from the DataFrame column
        (if applicable) or left as None.
    handle_2d : {"raise", "passthrough"}, default="raise"
        Determines how 2D inputs are handled when they do not
        meet the single-column requirement or remain
        reshaped to a single dimension:

        - ``"raise"`` : Raise a ValueError for data that is
          strictly two-dimensional but doesn't match the
          expected shape of ``(1, n)`` or ``(n, 1)``.
        - ``"passthrough"`` : Return the unmodified 2D data
          instead of raising an exception, allowing the caller
          to decide how to handle multi-column or multi-row
          data.

    Returns
    -------
    pandas.Series
        A one-dimensional Series containing the input data. If the
        conversion fails or if multiple columns are detected when
        a single-column structure was expected, a descriptive error
        is raised.

    Raises
    ------
    ValueError
        If the input shape or format is incompatible with a single
        Series. For instance, if the DataFrame contains more than
        one column, or if the NumPy array has more than one
        dimension that cannot be reduced to (n,).

    Examples
    --------
    >>> from fusionlab.core.array_manager import to_series
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Convert a list to Series
    >>> my_list = [1, 2, 3, 4]
    >>> s = to_series(my_list, name='my_series')
    >>> s
    0    1
    1    2
    2    3
    3    4
    Name: my_series, dtype: int64

    >>> # Convert a single-column DataFrame
    >>> df_single = pd.DataFrame({'A': [10, 20, 30]})
    >>> series_A = to_series(df_single)
    >>> series_A
    0    10
    1    20
    2    30
    Name: A, dtype: int64

    Notes
    -----
    The `to_series` function is primarily designed to simplify
    downstream processing by ensuring that any valid one-dimensional
    input is uniformly handled as a pandas Series. This is especially
    useful in data cleaning and feature engineering workflows, where
    consistency of data types and shapes is paramount.

    See Also
    --------
    to_array :
        Converts input to a NumPy array, providing an intermediate
        step for uniform handling of data shapes.

    References
    ----------
    .. [1] Slatt, R.M. "Stratigraphic reservoir characterization
       for petroleum geologists, geophysicists, and engineers",
       2nd Edition, Elsevier, 2013.
    """
    # If data is already a Series, just rename if requested.
    if isinstance(data, pd.Series):
        if name is not None:
            data = data.rename(name)
        return data

    # If data is a DataFrame, ensure it has exactly one column,
    # then extract that column as a Series.
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            if handle_2d=="passthrough": 
                return data 
            
            raise ValueError(
                "DataFrame must have exactly one"
                " column to be converted to Series."
            )
        series_col = data.columns[0]
        s = data.iloc[:, 0]
        # If user provided a name, use it; otherwise use the column name.
        if name is not None:
            s = s.rename(name)
        else: 
            s = pd.Series (s, name = series_col, index=s.index)
        return s

    # If data is a list or tuple, convert to a numpy array for uniform handling.
    if isinstance(data, (list, tuple)):
        data = np.array(data)

    # If data is a numpy array, ensure it is one-dimensional or reshaped to (n,).
    if isinstance(data, np.ndarray):
        # Reshape if it's (1, n) or (n, 1).
        if len(data.shape) == 2:
            if data.shape[0] == 1 and data.shape[1] >= 1:
                data = data.reshape(-1)
            elif data.shape[1] == 1 and data.shape[0] >= 1:
                data = data.reshape(-1)
            else:
                if handle_2d=="passthrough": 
                    return data 
                
                raise ValueError(
                    "NumPy array must be one-dimensional or reshapeable to (n,). "
                    f"Current shape: {data.shape}"
                )
        elif len(data.shape) > 2:
            if handle_2d=="passthrough": 
                return data 
            
            raise ValueError(
                "NumPy array must be one-dimensional or reshapeable to (n,). "
                f"Current shape: {data.shape}"
            )
        # Convert to Series
        s = pd.Series(data)
        if name is not None:
            s = s.rename(name)
        return s

    raise ValueError(
        f"Cannot convert data of type {type(data)} to a Series."
    )
    
def return_if_preserver_failed(
    d, to_numpy=False, 
    error="ignore", 
    verbose=0
    ):
    """ Return processed data types as is if failed to convert to its original 
    types with :func:`array_preserver`."""
    
    
    emsg =("Array preserver failed to properly revert the processed"
          f" data type {type(d).__name__!r} to its original types.")
    
    if error =='raise': 
        raise TypeError(emsg) 
    elif error=='warn': 
        warnings.warn(emsg) 
        
    # Check for DataFrame input    
    if isinstance(d, pd.DataFrame): 
        if d.shape[1] == 1: 
            if verbose > 0:
                print("Converting DataFrame with 1 column to Series.")
            d = to_series(d)  # Convert DataFrame with 1 column to Series
            
        # Handle numpy conversion
        if to_numpy:
            if verbose > 0:
                print("Converting to numpy array.")
            return d.to_numpy() if not isinstance(d, np.ndarray) else d
        return d
    
    # Handle non-DataFrame inputs and conversion to numpy
    if to_numpy:
        if verbose > 0:
            print("Converting input to numpy array.")
        return np.asarray(d) if not isinstance(d, np.ndarray) else d
    
    # Handle invalid types with error handling
    if not isinstance(d, (np.ndarray, pd.Series, pd.DataFrame, list, tuple)):
        msg = "Invalid input type: Expected a DataFrame, Series, ndarray, or list."
        
        if verbose > 0:
            print(f"{msg}")
        
        if error == "raise":
            raise ValueError(msg)
        elif error == "warn":
            warnings.warn(msg, UserWarning)
        elif error == "ignore":
            return d  # Return the original value if error is ignored
    
    return d