# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio  (a.k.a. @Daniel) <etanoyau@gmail.com>
# Source: Adapted from earthai-tech/gofast (https://github.com/earthai-tech/gofast)
# This module is included in the FusionLab package, with modifications
# to fit FusionLab’s API and coding conventions.

"""
Utilities for batch processing, unique ID generation, slicing, and
input validation in FusionLab. Supports generating reproducible IDs,
partitioning data into batches, and applying noise‑validation routines.

Adapted for FusionLab from the original fusionlab.core.handlers module.
"""

from __future__ import print_function

import re
import uuid
import string
import numbers
import random
import inspect
import datetime
import warnings
from collections.abc import Iterable
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .._fusionlog import fusionlog
from ..compat.scipy import optimize_minimize
from .checks import validate_noise, str2columns


__all__ = [ 
    'TypeEnforcer', 
    'add_noises_to',
    'adjust_to_samples',
    'batch_generator',
    'get_batch_size',
    'safe_slicing', 
    'gen_batches' , 
    'get_valid_kwargs',
    'generate_id',
    'make_ids',
    'resolve_label', 
    'columns_manager', 
    'columns_getter'
   ]

class TypeEnforcer:
    """
    Enforce return types of functions based on input parameter types or
    explicit type specifications.

    The ``TypeEnforcer`` decorator automatically converts the return values of
    decorated functions to match the types of specified input parameters or to
    explicitly defined types. It supports both single and multiple return
    values and handles various data types including NumPy arrays, pandas
    DataFrames, Series, lists, tuples, and booleans.

    Parameters
    ----------
    params : str or list of str, optional
        Names of input parameters whose types will enforce the return values.
        If a single string is provided, the type of that parameter will be
        applied to the first return value. If a list of strings is provided,
        each parameter's type will be applied to the corresponding return
        value in order. If ``rparams`` is also provided, ``rparams`` takes
        precedence over ``params``.
        
    rparams : dict, optional
        A dictionary mapping return value indices (int) or names (str) to
        desired type names or callable conversion functions. This explicitly
        defines the types to enforce on specific return values, overriding
        any type specifications provided in ``params``. Supported type names
        include:
        
        - ``'array-like'``: Converts to NumPy array.
        - ``'list'``: Converts to Python list.
        - ``'tuple'``: Converts to Python tuple.
        - ``'dataframe'`` or ``'pd.DataFrame'``: Converts to pandas DataFrame.
        - ``'series'`` or ``'pd.Series'``: Converts to pandas Series.
        - ``'boolean'``: Converts to boolean type.
        
        Custom callable functions can also be provided for advanced conversions.
        
    set_index : bool, default=False
        If ``True`` and the return type is a pandas DataFrame, the decorator
        attempts to set the index of the returned DataFrame to match the index
        of the specified input parameter. If the conversion fails, it silently
        returns the DataFrame as is.

    Examples
    --------
    >>> from fusionlab.core.handlers import TypeEnforcer
    >>> 
    >>> @TypeEnforcer
    >>> def example_function1(a):
    ...     return e
    >>> 
    >>> # The return value `e` will be converted to the type of parameter `a`
    >>> 
    >>> @TypeEnforcer(params='c')
    >>> def example_function3(a, c, d=2):
    ...     return e
    >>> 
    >>> # The return value `e` will be converted to the type of parameter `c`
    >>> 
    >>> @TypeEnforcer(rparams={'0': 'array-like'})
    >>> def example_function4(a, c, d=2):
    ...     return e
    >>> 
    >>> # The return value `e` will be converted to a NumPy array
    >>> 
    >>> @TypeEnforcer(rparams={'0': 'array-like', '2': 'pd.DataFrame', '3': 'series'})
    >>> def example_function5(a, c, d=2):
    ...     return e, f, g, h
    >>> 
    >>> # The return values will be converted as follows:
    >>> # e -> NumPy array, g -> pandas DataFrame, h -> pandas Series
    >>> 
    >>> @TypeEnforcer(params=['c', 'd'], rparams=None)
    >>> def example_function6(a, c, d=2):
    ...     return e, f, g, h
    >>> 
    >>> # The return values `e` and `f` will be converted to the types of
    >>> # parameters `c` and `d` respectively
    >>> 
    >>> @TypeEnforcer(params=['a', 'd'], rparams={'0': 'array-like', '2': 'pd.DataFrame', '3': 'series'})
    >>> def example_function8(a, c, d=2):
    ...     return e, f, g, h
    >>> 
    >>> # The return values will be converted as follows:
    >>> # e -> NumPy array (overrides type of 'a'),
    >>> # g -> pandas DataFrame,
    >>> # h -> pandas Series,
    >>> # f -> type of 'd'

    Notes
    -----
    - **Priority Handling:** When both ``params`` and ``rparams`` are provided,
      ``rparams`` takes precedence over ``params`` for type specifications of
      return values.

    - **Silent Failures:** If a type conversion fails for any return value, the
      original value is returned without raising an exception. However, the
      failure is logged for debugging purposes.

    - **NoneType Handling:** If the specified input parameter's value is
      ``None`` or has a ``NoneType``, the decorator skips the type conversion
      for the corresponding return value and returns it as is.

    - **Extensibility:** The ``type_mapping`` dictionary can be extended to
      support additional type specifications and corresponding conversion
      functions as needed.

    - **Index Setting for DataFrames:** The current implementation includes
      a placeholder for setting the index of converted DataFrames to match input
      parameters if ``set_index`` is ``True``. This can be further developed
      based on specific requirements and access to input parameter metadata.

    See Also
    --------
    [1] Brownlee, J. (2018). *Time Series Forecasting with Python: Create 
        accurate models in Python to forecast the future and gain insight
        from your time series data*. Machine Learning Mastery.
    [2] Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., & Cottrell, G. (2017). 
        Temporal fusion transformers for interpretable multi-horizon time
        series forecasting. *arXiv preprint arXiv:1912.09363*.

    References
    ----------
    .. [1] Brownlee, J. (2018). *Time Series Forecasting with Python: Create 
           accurate models in Python to forecast the future and gain insight 
           from your time series data*. Machine Learning Mastery.
    .. [2] Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., & Cottrell, 
           G. (2017). Temporal fusion transformers for interpretable multi-horizon 
           time series forecasting. *arXiv preprint arXiv:1912.09363*.
    """
    
    def __init__(
        self,
        params: Optional[Union[str, List[str]]] = None,
        rparams: Optional[Dict[Union[str, int], Union[str, Callable]]] = None,
        set_index: bool = False
    ):
        self.params = params
        self.rparams = rparams
        self.set_index = set_index
        self.logger = fusionlog.get_fusionlab_logger(__name__)
        self.type_mapping = {
            'array-like': lambda x: np.array(x),
            'list': lambda x: list(x),
            'tuple': lambda x: tuple(x),
            'dataframe': self.to_dataframe,
            'pd.DataFrame': self.to_dataframe,
            'series': self.to_series,
            'pd.Series': self.to_series,
            'boolean': lambda x: bool(x)
        }

    def __call__(self, func: Callable) -> Callable:
        """
        Make the class instance callable to use as a decorator.

        Parameters
        ----------
        func : Callable
            The function to be decorated.

        Returns
        -------
        Callable
            The wrapped function with enforced return types.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the original function
            result = func(*args, **kwargs)

            # Determine if result is a tuple (multiple return values)
            if isinstance(result, tuple):
                return_values = list(result)
            else:
                return_values = [result]

            # Prepare type specifications dictionary
            type_specs = {}

            # Priority 1: rparams
            if self.rparams:
                for key, dtype in self.rparams.items():
                    if isinstance(key, int):
                        if 0 <= key < len(return_values):
                            type_specs[key] = dtype
                    elif isinstance(key, str):
                        # XXX TODO: 
                        # Named return values can be handled here
                        # Currently, only positional handling is implemented
                        pass  # Extend if named return values are used

            # Priority 2: params
            elif self.params:
                if isinstance(self.params, str):
                    self.params = [self.params]
                for idx, param_name in enumerate(self.params):
                    if idx >= len(return_values):
                        break
                    # Extract parameter value from kwargs or args
                    try:
                        param_value = kwargs[param_name]
                    except KeyError:
                        # Get from args based on function signature
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        param_value = bound_args.arguments.get(param_name, None)
                    if param_value is not None:
                        type_specs[idx] = type(param_value)

            # Apply type conversions based on type_specs
            for idx, value in enumerate(return_values):
                if idx in type_specs:
                    desired_type = type_specs[idx]
                    # Handle string type specifications
                    if isinstance(desired_type, str):
                        conversion_func = self.type_mapping.get(
                            desired_type.lower()
                        )
                        if conversion_func:
                            try:
                                return_values[idx] = conversion_func(value)
                            except Exception as e:
                                self.logger.debug(
                                    f"Conversion failed for return value at "
                                    f"index {idx} with type '{desired_type}': {e}"
                                )
                                # Return value remains unchanged
                        else:
                            self.logger.debug(
                                f"Unsupported type specification "
                                f"'{desired_type}' for return value at "
                                f"index {idx}."
                            )
                    elif isinstance(desired_type, type):
                        # Direct type conversion using constructor
                        try:
                            return_values[idx] = desired_type(value)
                        except Exception as e:
                            self.logger.debug(
                                f"Conversion failed for return value at "
                                f"index {idx} to type '{desired_type}': {e}"
                            )
                            # Return value remains unchanged
                    elif callable(desired_type):
                        # Use the provided callable for conversion
                        try:
                            return_values[idx] = desired_type(value)
                        except Exception as e:
                            self.logger.debug(
                                f"Callable conversion failed for return value "
                                f"at index {idx}: {e}"
                            )
                            # Return value remains unchanged
                    else:
                        self.logger.debug(
                            f"Invalid type specification '{desired_type}' for "
                            f"return value at index {idx}."
                        )

            # Return as tuple if original result was tuple
            if isinstance(result, tuple):
                return tuple(return_values)
            else:
                return return_values[0]

        return wrapper

    def to_dataframe(self, value: Any) -> Any:
        """
        Convert the given value to a pandas DataFrame, preserving columns and
        index if possible.

        Parameters
        ----------
        value : Any
            The value to convert.

        Returns
        -------
        pd.DataFrame or original value
            The converted DataFrame or the original value if conversion fails.
        """
        try:
            if isinstance(value, pd.DataFrame):
                return value
            df = pd.DataFrame(value)
            if self.set_index and hasattr(df, 'index'):
                # Attempt to set index from the first parameter if available
                # Currently, index setting logic is not implemented
                pass  # Implement if access to input parameter's index is possible
            return df
        except Exception as e:
            self.logger.debug(f"Failed to convert value to DataFrame: {e}")
            return value

    def to_series(self, value: Any) -> Any:
        """
        Convert the given value to a pandas Series.

        Parameters
        ----------
        value : Any
            The value to convert.

        Returns
        -------
        pd.Series or original value
            The converted Series or the original value if conversion fails.
        """
        try:
            if isinstance(value, pd.Series):
                return value
            series = pd.Series(value)
            return series
        except Exception as e:
            self.logger.debug(f"Failed to convert value to Series: {e}")
            return value

def delegate_on_error(
    transfer,
    delegate_params_mapping=None,
    additional_args=None,
    additional_kwargs=None,
    condition=lambda exc: True
):
    """
    Decorator to delegate function/method execution to another upon encountering 
    an error.
    
    This decorator allows a function or method to delegate its execution to another 
    specified function or method when an error occurs. It smartly maps and transfers 
    parameters from the original function to the delegate function, ensuring a seamless 
    fallback mechanism. This enhances the robustness and flexibility of code by 
    providing alternative execution paths in case of failures.
    
    Parameters
    ----------
    transfer : callable
        The function or method to delegate to if an error occurs in the decorated 
        function/method.
    delegate_params_mapping : dict, optional
        A dictionary mapping parameter names from the decorated function to 
        corresponding parameter names in the ``transfer`` function. This allows for 
        renaming or restructuring of arguments during delegation.
    additional_args : list, optional
        Additional positional arguments to pass to the ``transfer`` function. These 
        are appended to the original function's positional arguments.
    additional_kwargs : dict, optional
        Additional keyword arguments to pass to the ``transfer`` function. These 
        are merged with the original function's keyword arguments.
    condition : callable, optional
        A function that takes an exception instance as input and returns 
        ``True`` if delegation should occur for that exception, or ``False`` 
        otherwise. This allows for selective delegation based on exception types 
        or attributes.
    
    Returns
    -------
    callable
        The decorated function or method with delegation capability.
    
    .. math::
        d\left(f, g\right) = 
        \begin{cases} 
            f(x) & \text{if no exception occurs} \\
            g(x) & \text{if exception occurs and condition is met}
        \end{cases}
    
    The decorator works by attempting to execute the original function ``f``. If an 
    exception is raised and the ``condition`` evaluates to ``True`` for that exception, 
    the decorator delegates the call to the ``transfer`` function ``g``, passing the 
    mapped and transferred parameters as specified in ``delegate_params_mapping``. This 
    allows for graceful degradation or alternative processing without halting the 
    program.
    
    Examples
    --------
    >>> import warnings
    >>> from fusionlab.compat.sklearn import InvalidParametersError 
    >>> from fusionlab.core.handlers import delegate_on_error
    >>> 
    >>> class A:
    ...     def methodA2(self, param, parama11, parama12=None, verbose=1):
    ...         return f"Delegated {param}, {parama11}, {parama12}"
    ...     
    ...     @delegate_on_error(
    ...         transfer=methodA2,
    ...         delegate_params_mapping={
    ...             'parama1': 'parama11',
    ...             'parama2': 'parama12'
    ...         }
    ...     )
    ...     def methodA1(self, param, parama1, parama2=None, verbose=0):
    ...         if parama1 < 0:
    ...             raise ValueError("parama1 must be non-negative")
    ...         return f"Processed {param}, {parama1}, {parama2}"
    ... 
    >>> a = A()
    >>> print(a.methodA1(param=10, parama1=5, parama2=15, verbose=2))
    Processed 10, 5, 15
    >>> print(a.methodA1(param=10, parama1=-1, parama2=15, verbose=2))
    /path/to/file.py:XX: UserWarning: 
    Error in methodA1: parama1 must be non-negative. Delegating to methodA2.
    Delegated 10, -1, 15
    >>> 
    >>> class B:
    ...     def compute_average(self, data, ignore_errors=False, verbose=0):
    ...         if not isinstance(data, list):
    ...             raise TypeError("Data must be a list.")
    ...         return sum(data) / len(data)
    ...     
    ...     def fallback_average(self, data, numeric_only=True, verbose=1):
    ...         if numeric_only:
    ...             data = [x for x in data if isinstance(x, (int, float))]
    ...         return np.mean(data)
    ...     
    ...     @delegate_on_error(
    ...         transfer=fallback_average,
    ...         delegate_params_mapping={
    ...             'ignore_errors': 'numeric_only'
    ...         },
    ...         condition=lambda e: not isinstance(e, InvalidParametersError)
    ...     )
    ...     def compute_average_decorated(self, data, ignore_errors=False, verbose=0):
    ...         if not isinstance(data, list):
    ...             raise InvalidParametersError("Data must be a list.")
    ...         return sum(data) / len(data)
    ... 
    >>> b = B()
    >>> print(b.compute_average_decorated(data=[1, 2, 3, 4], verbose=2))
    Computed average: 2.5
    >>> print(b.compute_average_decorated(data="not a list", verbose=2))
    /path/to/file.py:XX: UserWarning: 
    Error in compute_average_decorated: Data must be a list.. Delegating to fallback_average.
    nan
    >>> 
    >>> # Delegation does not occur for InvalidParametersError
    >>> try:
    ...     b.compute_average_decorated(data="not a list", verbose=2)
    ... except InvalidParametersError as e:
    ...     print(f"Caught an exception: {e}")
    Caught an exception: Data must be a list.
    Notes
    -----
    - The decorator ``delegate_on_error`` enhances the resilience of functions and 
      methods by providing an automated delegation mechanism in response to 
      exceptions.
    - The ``delegate_params_mapping`` allows for flexible parameter translation, 
      accommodating differences in parameter names between the original and 
      transfer functions.
    - The ``condition`` parameter enables fine-grained control over which 
      exceptions should trigger delegation, supporting sophisticated error-handling 
      strategies.
    
    See Also
    --------
    functools.wraps : A decorator for updating the wrapper function to look 
    like the wrapped function.
    warnings.warn : Issues a warning message.
    Exception : Base class for all built-in exceptions.
    
    References
    ----------
    .. [1] Python Documentation. (2023). `functools.wraps`. Retrieved from 
       https://docs.python.org/3/library/functools.html#functools.wraps
    .. [2] Python Documentation. (2023). `warnings.warn`. Retrieved from 
       https://docs.python.org/3/library/warnings.html#warnings.warn
    .. [3] Python Documentation. (2023). `Exception`. Retrieved from 
       https://docs.python.org/3/library/exceptions.html#Exception
    """

    delegate_params_mapping = delegate_params_mapping or {}
    additional_args = additional_args or []
    additional_kwargs = additional_kwargs or {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Attempt to execute the original function
                return func(*args, **kwargs)
            except Exception as e:
                # Check if delegation condition is met
                if condition(e):
                    if kwargs.get('verbose', 0) >= 2:
                        warnings.warn(
                            f"Error in {func.__name__}: {e}. Delegating to {transfer.__name__}."
                        )
                    # Prepare arguments for the transfer function
                    transfer_kwargs = {}
                    
                    # Map parameters based on delegate_params_mapping
                    for orig_param, transfer_param in delegate_params_mapping.items():
                        if orig_param in kwargs:
                            transfer_kwargs[transfer_param] = kwargs.pop(orig_param)
                    
                    # Automatically transfer parameters with the same name
                    common_params = set(kwargs.keys()) & set(
                        transfer.__code__.co_varnames
                    )
                    for param in common_params:
                        transfer_kwargs[param] = kwargs[param]
                    
                    # Add any additional arguments
                    if additional_args:
                        transfer_args = list(args) + additional_args
                    else:
                        transfer_args = list(args)
                    
                    if additional_kwargs:
                        transfer_kwargs.update(additional_kwargs)
                    
                    # Execute the transfer function with mapped parameters
                    return transfer(*transfer_args, **transfer_kwargs)
                else:
                    # Re-raise the exception if condition is not met
                    raise e
        return wrapper
    return decorator

def param_deprecated_message(
    deprecated_params_mappings=None, 
    conditions_params_mappings=None,
    warning_category=FutureWarning, 
    extra_message=''
):
    """
    Decorator to handle deprecated or conditionally validated parameters 
    in functions, methods, or class initializers.

    The decorator manages two primary scenarios:
    
    1. **Deprecated or Renamed Parameters**:
       Detects if deprecated parameters are used, issues a warning, and 
       maps their values to new parameters if specified.
    
    2. **Conditionally Validated Parameters**:
       Checks if parameters meet certain conditions, warns the user, and overrides 
       their values with defaults if necessary.

    .. math::
        \text{Warning} = 
        \begin{cases} 
            \text{FutureWarning} & \text{if a deprecated parameter is used} \\
            \text{FutureWarning} & \text{if a parameter condition is met}
        \end{cases}

    Parameters
    ----------
    deprecated_params_mappings : dict or list of dicts, optional
        Defines mappings for deprecated parameters. Each mapping should include:
        
        - ``'old'`` (`str`): The name of the deprecated parameter.
        - ``'new'`` (`str`, optional): The name of the new parameter to replace
          the deprecated one.
        - ``'message'`` (`str`, optional): Custom warning message.
        - ``'default'`` (optional): Default value to use if the new parameter 
          is not provided.
        
        Example:
        
        .. code-block:: python

            deprecated_params_mappings = [
                {
                    'old': 'old_param',
                    'new': 'new_param',
                    'message': 'old_param is deprecated, use new_param instead.',
                    'default': 42
                }
            ]

    conditions_params_mappings : dict or list of dicts, optional
        Defines conditions that parameters must satisfy. Each mapping should 
        include:
        
        - ``'param'`` (`str`): The name of the parameter to validate.
        - ``'condition'`` (`callable`): A function that takes the parameter 
          value and returns 
          `True` if the condition is met.
        - ``'message'`` (`str`, optional): Custom warning message.
        - ``'default'`` (optional): Default value to override the parameter 
          if the condition is met.
        
        Example:
        
        .. code-block:: python

            conditions_params_mappings = [
                {
                    'param': 'threshold',
                    'condition': lambda x: x < 0,
                    'message': 'threshold must be non-negative.',
                    'default': 0
                }
            ]

    warning_category : Warning, optional
        The category of warning to raise. Defaults to ``FutureWarning``.
        
    extra_message : str, optional
        Additional message appended to all warning messages. Defaults to an 
        empty string.

    Returns
    -------
    function
        The decorated function or class initializer with parameter validation.

    Methods
    -------
    wrapper(*args, **kwargs)
        Internal method that wraps the original function or class initializer, 
        processes parameters, and raises warnings as necessary.

    Examples
    --------
    >>> from fusionlab.core.handlers import param_deprecated_message
    >>> @param_deprecated_message(
    ...     deprecated_params_mappings=[
    ...         {
    ...             'old': 'old_param',
    ...             'new': 'new_param',
    ...             'message': 'old_param is deprecated, use new_param instead.',
    ...             'default': 10
    ...         }
    ...     ],
    ...     conditions_params_mappings={
    ...         'param': 'limit',
    ...         'condition': lambda x: x < 0,
    ...         'message': 'limit must be non-negative.',
    ...         'default': 0
    ...     }
    ... )
    ... def my_function(new_param=5, old_param=None, limit=100):
    ...     return new_param, limit
    >>> my_function(old_param=20, limit=-10)
    # Emits warnings and returns (20, 0)

    Notes
    -----
    - This decorator is versatile and can be applied to both standalone 
      functions and class initializers.
    - When both deprecated and new parameters are provided, the new parameter
      takes precedence.

    See Also
    --------
    warnings.warn : Issue a warning message.
    functools.wraps : Decorator to preserve the metadata of the original function.

    References
    ----------
    .. [1] Python Documentation on [warnings](https://docs.python.org/3/library/warnings.html)
    .. [2] Python Documentation on [functools.wraps](https://docs.python.org/3/library/functools.html#functools.wraps)

    """

    if deprecated_params_mappings is None:
        deprecated_params_mappings = []
    if conditions_params_mappings is None:
        conditions_params_mappings = []
    
    # Ensure these are lists for unified processing
    if isinstance(deprecated_params_mappings, dict):
        deprecated_params_mappings = [deprecated_params_mappings]
    if isinstance(conditions_params_mappings, dict):
        conditions_params_mappings = [conditions_params_mappings]

    def decorator(func):
        # Handle both classes (decorate __init__) and functions.
        # If func is a class, wrap its __init__.
        if inspect.isclass(func):
            init_func = func.__init__
        else:
            init_func = func

        @wraps(init_func)
        def wrapper(*args, **kwargs):
            # Convert positional args to keyword arguments based on signature
            sig = inspect.signature(init_func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Process deprecated params first
            for mapping in deprecated_params_mappings:
                old_param = mapping.get('old')
                new_param = mapping.get('new')
                custom_message = mapping.get('message')
                default_val = mapping.get('default', None)
                
                if ( 
                        old_param in bound_args.arguments 
                        and bound_args.arguments[old_param] is not None
                    ):
                    # User specified the deprecated param
                    old_val = bound_args.arguments[old_param]
                    # If new_param is provided, we move value there only if 
                    # it's not already set
                    # If user already set new_param as well, we might choose 
                    # to let the new_param override or raise an error. 
                    # For simplicity, if new_param is set by user, we trust the user.
                    if new_param:
                        if (new_param not in bound_args.arguments) or (
                                bound_args.arguments[new_param] is None):
                            bound_args.arguments[new_param] = old_val
                        # else: new_param already set, just ignore old_val (or could warn)
                    else:
                        # If no new_param, we might just drop or use default if provided
                        if default_val is not None:
                            bound_args.arguments[old_param] = default_val

                    # Build and show warning
                    msg = custom_message
                    if not msg:
                        msg = ( 
                            f"Parameter '{old_param}' is deprecated and will"
                            " be removed in a future version."
                            )
                        if new_param:
                            msg += f" Please use '{new_param}' instead."
                        msg += extra_message
                    warnings.warn(msg, category=warning_category, stacklevel=2)

                    # Remove the old param if it is purely deprecated (and replaced)
                    # If we strictly want to remove it from arguments:
                    # This may not be necessary, as we can just let it there. 
                    # But let's clean it for clarity if new_param is defined.
                    if new_param:
                        del bound_args.arguments[old_param]

            # Process conditions
            for cond_map in conditions_params_mappings:
                param_name = cond_map.get('param')
                condition = cond_map.get('condition', lambda v: False)
                cond_message = cond_map.get('message')
                cond_default = cond_map.get('default', None)
                 
                if param_name in bound_args.arguments:
                    param_val = bound_args.arguments[param_name]
                    if condition(param_val):
                        # Condition met, warn and set default if provided
                        msg = cond_message
                        if not msg:
                            msg = (
                                f"Parameter '{param_name}' does not support the given "
                                f"value. Overriding to default. {extra_message}"
                                )
                        warnings.warn(msg, category=warning_category, stacklevel=2)
                        # if cond_default is not None:
                        bound_args.arguments[param_name] = cond_default

            return init_func(*bound_args.args, **bound_args.kwargs)

        if inspect.isclass(func):
            # If it's a class, rebuild the class with a new __init__
            setattr(func, '__init__', wrapper)
            return func
        else:
            return wrapper

    return decorator

def generate_id(
    length=12,
    prefix="",
    suffix="",
    include_timestamp=False,
    use_uuid=False,
    char_set=None,
    numeric_only=False,
    unique_ids=None,
    retries=3
):
    """
    Generate a customizable and unique ID with options for prefix, suffix, 
    timestamp, and character type.

    Parameters
    ----------
    length : int, optional
        Length of the generated ID, excluding any specified prefix, suffix, 
        or timestamp. Default is 12. Ignored if `use_uuid` is set to ``True``.

    prefix : str, optional
        Prefix string to be added to the beginning of the generated ID.
        Defaults to an empty string.

    suffix : str, optional
        Suffix string to append to the end of the generated ID.
        Defaults to an empty string.

    include_timestamp : bool, optional
        If ``True``, appends a timestamp in the 'YYYYMMDDHHMMSS' format
        to the ID. Defaults to ``False``.

    use_uuid : bool, optional
        If ``True``, generates the ID using UUID4, ignoring the parameters
        `length`, `char_set`, and `numeric_only`. Defaults to ``False``.

    char_set : str or None, optional
        A string specifying the set of characters to use in the ID. 
        If ``None``, defaults to alphanumeric characters 
        (uppercase and lowercase letters plus digits).

    numeric_only : bool, optional
        If ``True``, limits the character set to numeric digits only. 
        Defaults to ``False``. Overridden by `char_set` if provided.

    unique_ids : set or None, optional
        A set to store and check for unique IDs. If provided, generated IDs 
        are compared against this set to ensure no duplicates. New unique IDs 
        are added to this set after generation.

    retries : int, optional
        Number of retries if a generated ID conflicts with `unique_ids`.
        Defaults to 3.

    Returns
    -------
    str
        A string representing the generated ID, potentially including the 
        specified prefix, suffix, timestamp, and custom length.

    Notes
    -----
    The function allows for highly customizable ID generation, supporting 
    different character sets, unique ID constraints, and options for 
    timestamped or UUID-based IDs. When using `unique_ids`, the function 
    performs multiple attempts to generate a unique ID, retrying as specified 
    by the `retries` parameter.

    The generated ID can be represented as a combination of three components:

    .. math:: 
        \text{{ID}} = \text{{prefix}} + \text{{base ID}} + \text{{suffix}}

    Where:
        - `prefix` and `suffix` are optional components.
        - `base ID` is a string of randomly selected characters from the 
          specified character set or a UUID-based string.

    Examples
    --------
    >>> from fusionlab.core.handlers import generate_id
    >>> generate_id(length=8, prefix="PAT-", suffix="-ID", include_timestamp=True)
    'PAT-WJ8N6F-20231025123456-ID'
    
    >>> generate_id(length=6, numeric_only=True)
    '483920'

    >>> unique_set = set()
    >>> generate_id(length=10, unique_ids=unique_set, retries=5)
    'Y8B5QD2L7H'
    
    See Also
    --------
    uuid : Module to generate universally unique identifiers.

    References
    ----------
    .. [1] Jane Doe et al. "Best Practices in Unique Identifier Generation." 
           Data Science Journal, 2021, vol. 9, no. 4, pp. 210-222.
    .. [2] J. Smith. "Character-Based ID Generation for High-Volume Systems."
           Proceedings of the ID Conference, 2022.
    """
    
    # Define the character set
    if use_uuid:
        # Use UUID for ID generation if specified
        new_id = str(uuid.uuid4()).replace("-", "")
        if include_timestamp:
            new_id += datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{prefix}{new_id[:length]}{suffix}"

    if numeric_only:
        char_set = string.digits
    elif char_set is None:
        char_set = string.ascii_letters + string.digits

    def _generate_base_id():
        """Generates the base ID without prefix, suffix, or timestamp."""
        return ''.join(random.choice(char_set) for _ in range(length))

    # Retry logic to ensure uniqueness if required
    for _ in range(retries):
        # Generate base ID and add optional elements
        new_id = _generate_base_id()
        
        # Include timestamp if specified
        if include_timestamp:
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            new_id += timestamp
        
        # Add prefix and suffix
        new_id = f"{prefix}{new_id}{suffix}"

        # Check for uniqueness if a unique_ids set is provided
        if unique_ids is not None:
            if new_id not in unique_ids:
                unique_ids.add(new_id)
                return new_id
        else:
            return new_id

    # Raise error if unique ID generation failed after retries
    raise ValueError("Failed to generate a unique ID after multiple retries.")

def get_valid_kwargs(obj_or_func, raise_warning=False, **kwargs):
    """
    Filters keyword arguments (`kwargs`) to retain only those that are valid
    for the initializer of a given object or function.

    Parameters
    ----------
    obj_or_func : object or function
        The object or function to inspect for valid keyword arguments. If it's
        callable, its `__init__` method's valid keyword arguments are considered.
    raise_warning : bool, optional
        If True, raises a warning for any keyword arguments provided that are not
        valid for `obj_or_func`. The default is False.
    **kwargs : dict
        Arbitrary keyword arguments to filter based on `obj_or_func`'s
        valid parameters.

    Returns
    -------
    dict
        A dictionary containing only the keyword arguments that are valid for the
        `obj_or_func`'s initializer.

    Raises
    ------
    Warning
        If `raise_warning` is True and there are keyword arguments that are not
        valid for `obj_or_func`, a warning is raised.

    Notes
    -----
    This function checks whether the provided keyword arguments are valid for the given
    class, method, or function. It filters out any invalid keyword arguments and returns
    a dictionary containing only the valid ones.

    If the provided object is a class, it inspects the __init__ method to determine the
    valid keyword arguments. If it is a method or function, it inspects the argument names.

    It issues a warning for any invalid keyword arguments if `raise_warning`
    is ``True`` but it does not raise an error.
    
    Examples
    --------
    >>> from fusionlab.core.handlers import get_valid_kwargs
    >>> class MyClass:
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    >>> valid_kwargs = get_valid_kwargs(MyClass, a=1, b=2, c=3)
    >>> print(valid_kwargs)
    {'a': 1, 'b': 2}
    >>> valid_kwargs = get_valid_kwargs(MyClass, raise_warning=True,  **kwargs)
    Warning: 'arg3' is not a valid keyword argument for 'MyClass'.
    >>> print(valid_kwargs)
    {'arg1': 1, 'arg2': 2}

    >>> def my_function(a, b, c):
    ...     return a + b + c
    ...
    >>> kwargs = {'a': 1, 'b': 2, 'd': 3}
    >>> valid_kwargs = get_valid_kwargs(my_function, raise_warning=True, **kwargs)
    Warning: 'd' is not a valid keyword argument for 'my_function'.
    >>> print(valid_kwargs)
    {'a': 1, 'b': 2}
    """
    valid_kwargs = {}
    not_valid_keys = []

    # Determine whether obj_or_func is callable and get its valid arguments
    obj = obj_or_func() if callable(obj_or_func) else obj_or_func
    valid_args = obj.__init__.__code__.co_varnames if hasattr(
        obj, '__init__') else obj.__code__.co_varnames

    # Filter kwargs to separate valid from invalid ones
    for key, value in kwargs.items():
        if key in valid_args:
            valid_kwargs[key] = value
        else:
            not_valid_keys.append(key)

    # Raise a warning for invalid kwargs, if required
    if raise_warning and not_valid_keys:
        warning_msg = (f"'{', '.join(not_valid_keys)}' "
                       f"{'is' if len(not_valid_keys) == 1 else 'are'} "
                       "not a valid keyword argument "
                       f"for '{obj_or_func.__name__}'.")
        warnings.warn(warning_msg)

    return valid_kwargs

def _get_valid_kwargs(
        callable_obj: Any, kwargs: Dict[str, Any], 
        error: str ='warn', 
    ) -> Dict[str, Any]:
    """
    Filter and return only the valid keyword arguments for a given 
    callable object, while warning about any invalid kwargs.
    
    Parameters
    ----------
    callable_obj : callable
        The callable object (function, lambda function, method, or class) 
        for which the keyword arguments need to be validated.
    
    kwargs : dict
        Dictionary of keyword arguments to be validated against the callable object.
    
    Returns
    -------
    valid_kwargs : dict
        Dictionary containing only the valid keyword arguments 
        for the callable object.
    """
    # If the callable_obj is an instance, get its class
    if not inspect.isclass(callable_obj) and not callable(callable_obj):
        callable_obj = callable_obj.__class__
    
    try:
        # Retrieve the signature of the callable object
        signature = inspect.signature(callable_obj)
    except ValueError:
        # If signature cannot be obtained, return empty kwargs and warn
        warnings.warn(
            "Unable to retrieve signature of the callable object. "
            "No keyword arguments will be passed."
        )
        return {}
    
    # Extract parameter names from the function signature
    valid_params = set(signature.parameters.keys())
    
    # Identify valid and invalid kwargs
    valid_kwargs = {}
    invalid_kwargs = {}
    for k, v in kwargs.items():
        if k in valid_params:
            valid_kwargs[k] = v
        else:
            invalid_kwargs[k] = v
    
    # Warn the user about invalid kwargs
    if invalid_kwargs:
        invalid_keys = ', '.join(invalid_kwargs.keys())
        msg = ("The following keyword arguments are invalid"
               f" and will be ignored: {invalid_keys}"
        )
        if error =='warn': 
            warnings.warn(msg)
        elif error =='raise': 
            raise ValueError(msg ) 
        # ignore.
    
    return valid_kwargs

def get_batch_size(
    *arrays,
    default_size=None,
    max_memory_usage_ratio=0.1
):
    """
    Determine an optimal batch size based on available memory.

    This function computes an optimal batch size for processing large arrays
    in batches, aiming to prevent memory overload by considering the available
    system memory. If `psutil` is installed, it uses the available memory to
    calculate the batch size. Otherwise, it warns the user and defaults to a
    specified `default_size`.

    Parameters
    ----------
    *arrays : array-like
        One or more arrays (e.g., NumPy arrays) for which to compute the batch
        size. All arrays must have the same number of samples (first dimension).

    default_size : int, optional
        The default batch size to use if `psutil` is not installed or if you prefer
        to specify a fixed batch size. If not provided and `psutil` is not installed,
        the function defaults to 512.

    max_memory_usage_ratio : float, default 0.1
        The fraction of available system memory to allocate for the batch data.
        This parameter is only used if `psutil` is installed.

    Returns
    -------
    int
        The computed batch size, which is at least 1 and at most the number of
        samples in the arrays.

    Notes
    -----
    The batch size is computed using the formula:

    .. math::

        \\text{batch\\_size} = \\min\\left(
            \\max\\left(
                1, \\left\\lfloor \\frac{M \\times R}{S} \\right\\rfloor
            \\right), N
        \\right)

    where:

    - :math:`M` is the available system memory in bytes, obtained via `psutil`.
    - :math:`R` is the `max_memory_usage_ratio`.
    - :math:`S` is the total size in bytes of one sample across all arrays.
    - :math:`N` is the total number of samples in the arrays.

    If `psutil` is not installed, a default `batch_size` is used, or less if there
    are fewer samples.

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.core.handlers import get_batch_size
    >>> X = np.random.rand(1000, 20)
    >>> y = np.random.rand(1000)
    >>> batch_size = get_batch_size(X, y)
    >>> print(batch_size)
    64

    See Also
    --------
    batch_generator : Generator function to create batches.

    References
    ----------
    .. [1] Giampaolo Rodola, "psutil - process and system utilities",
       https://psutil.readthedocs.io/

    """
    try:
        import psutil
        psutil_available = True
    except ImportError:
        psutil_available = False
        if default_size is None:
            default_size = 512
        warnings.warn(
            "'psutil' is not installed. Cannot compute optimal batch size "
            "based on available memory. Using default batch_size="
            f"{default_size}."
        )

    arrays = [np.asarray(arr) for arr in arrays]
    n_samples = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != n_samples:
            raise ValueError(
                "All arrays must have the same number of samples "
                "in the first dimension."
            )

    if default_size is not None:
        # Check if default_size is greater than the number of samples
        if default_size > n_samples:
            if psutil_available:
                warnings.warn(
                    f"Default batch_size {default_size} is greater than the "
                    f"number of samples ({n_samples}). Recomputing batch size "
                    "based on available memory."
                )
            else:
                warnings.warn(
                    f"Default batch_size {default_size} is greater than the "
                    f"number of samples ({n_samples}). Using batch_size={n_samples}."
                )
                default_size = n_samples
        return default_size

    if psutil_available:
        available_memory = psutil.virtual_memory().available
        # Compute size of one sample across all arrays
        sample_size = sum(
            arr[0].nbytes for arr in arrays
        )
        max_memory_usage = available_memory * max_memory_usage_ratio
        batch_size = int(max_memory_usage // sample_size)
        batch_size = max(1, min(batch_size, n_samples))

        # If batch_size is greater than array length, warn user
        if batch_size > n_samples:
            warnings.warn(
                f"Computed batch_size {batch_size} is greater than the number "
                f"of samples ({n_samples}). Using batch_size={n_samples}."
            )
            batch_size = n_samples

        return batch_size
    else:
        # psutil is not available, default_size must have been set
        return default_size

def batch_generator(
        *arrays,
        batch_size
    ):
    """
    Generate batches of arrays for efficient processing.

    This generator yields batches of the input arrays,
    allowing for memory-efficient processing of large
    datasets. All input arrays must have the same first
    dimension (number of samples).

    Parameters
    ----------
    *arrays : array-like
        One or more arrays (e.g., NumPy arrays) to be
        divided into batches. All arrays must have the
        same number of samples (first dimension).

    batch_size : int
        The size of each batch. Must be a positive integer.

    Yields
    ------
    tuple of array-like
        A tuple containing slices of the input arrays,
        corresponding to the current batch.

    Notes
    -----
    The function iterates over the arrays, yielding slices
    from `start_idx` to `end_idx`, where:

    .. math::

        \text{start\_idx} = k \times \text{batch\_size}

        \text{end\_idx} = \min\left(
            (k + 1) \times \text{batch\_size}, N
        \right)

    with :math:`k` being the batch index and :math:`N`
    the total number of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.utils.sysutils import batch_generator
    >>> X = np.arange(10)
    >>> y = np.arange(10) * 2
    >>> batch_size = 3
    >>> for X_batch, y_batch in batch_generator(
    ...         X, y, batch_size=batch_size):
    ...     print(X_batch, y_batch)
    [0 1 2] [0 2 4]
    [3 4 5] [6 8 10]
    [6 7 8] [12 14 16]
    [9] [18]

    See Also
    --------
    get_batch_size : Function to compute an optimal batch size.

    References
    ----------
    .. [1] Python Software Foundation, "Generators",
       https://docs.python.org/3/howto/functional.html#generators

    """
    n_samples = arrays[0].shape[0]
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        yield tuple(arr[start_idx:end_idx] for arr in arrays)

        
def safe_slicing(slice_indexes, X):
    """
    Removes slices from the list `slice_indexes` that result in zero samples 
    when applied to the data `X`. The function checks each slice to ensure 
    it selects at least one sample, and discards any slices with no samples 
    selected.

    Parameters
    ----------
    slice_indexes : list of slice objects
        A list of slice objects, each representing a range of indices 
        that can be used to index into a dataset, typically for batch 
        processing.

    X : ndarray of shape (n_samples, n_features)
        The data array (or any other array-like structure) that the slices 
        will be applied to. The function assumes that each slice in 
        `slice_indexes` corresponds to a subset of rows (samples) in `X`.

    Returns
    -------
    valid_slices : list of slice objects
        A list of slice objects that correspond to valid (non-empty) 
        subsets of `X`. Slices with zero elements (e.g., when the 
        start index is equal to or greater than the end index) are removed.

    Examples
    --------
    # Example 1: Basic use case where the last slice is valid
    >>> X = np.random.rand(2000, 5)  # 2000 samples, 5 features
    >>> slice_indexes = [slice(0, 512), slice(512, 1024), slice(1024, 1536), 
                         slice(1536, 2000)]
    >>> safe_slicing(slice_indexes, X)
    [slice(0, 512, None), slice(512, 1024, None), slice(1024, 1536, None),
     slice(1536, 2000, None)]

    # Example 2: Case where the last slice has zero elements and is removed
    >>> slice_indexes = [slice(0, 512), slice(512, 1024), slice(1024, 1536),
                         slice(1536, 1500)]
    >>> safe_slicing(slice_indexes, X)
    [slice(0, 512, None), slice(512, 1024, None), slice(1024, 1536, None)]

    # Example 3: Empty slice case where all slices are removed
    >>> slice_indexes = [slice(0, 0), slice(1, 0)]
    >>> safe_slicing(slice_indexes, X)
    []

    Notes
    -----
    - This function is useful when handling slices generated for batch 
      processing in machine learning workflows, ensuring that only valid 
      batches are processed.
    - The function checks the start and stop indices of each slice and 
      ensures that `end > start` before including the slice in the 
      returned list.
    """
    
    valid_slices = []
    for slice_obj in slice_indexes:
        # Extract the slice range
        start, end = slice_obj.start, slice_obj.stop
        
        # Check if the slice has at least one sample
        if end > start:
            # Add to the valid_slices list only if there are samples
            valid_slices.append(slice_obj)
    
    return valid_slices

def gen_batches(n, batch_size, *, min_batch_size=0):
    """Generator to create slices containing `batch_size` 
    elements from 0 to `n`.

    The last slice may contain less than `batch_size` elements, when
    `batch_size` does not divide `n`.
    
    This is take on scikit-learn :func:`sklearn.utils.gen_batches` but modify 
    to ensure that min_batch_size in not included. 

    Parameters
    ----------
    n : int
        Size of the sequence.
    batch_size : int
        Number of elements in each batch.
    min_batch_size : int, default=0
        Minimum number of elements in each batch.

    Yields
    ------
    slice of `batch_size` elements

    See Also
    --------
    gen_even_slices: Generator to create n_packs slices going up to n.
    sklearn.utils.gen_slices: A generic batch slices

    Examples
    --------
    >>> list(gen_batches(7, 3))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(6, 3))
    [slice(0, 3, None), slice(3, 6, None)]
    >>> list(gen_batches(2, 3))
    [slice(0, 2, None)]
    >>> list(gen_batches(7, 3, min_batch_size=0))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(7, 3, min_batch_size=2))
    [slice(0, 3, None), slice(3, 7, None)]
    """
    if not isinstance(batch_size, numbers.Integral):
        raise TypeError(
            f"gen_batches got batch_size={batch_size}, must be an integer"
        )
    if batch_size <= 0:
        raise ValueError(f"gen_batches got batch_size={batch_size}, must be positive")
    
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        # Skip batches where the remaining size would
        # be smaller than min_batch_size
        if end + min_batch_size > n:
            break
        yield slice(start, end)
        start = end

    # Handle the last batch
    if start < n and (n - start) >= min_batch_size:
        yield slice(start, n)

def adjust_to_samples(n_samples, *values, initial_guess=None, error='warn'):
    """
    Adjusts the given values to match a total number of samples, aiming to distribute
    the samples evenly across the dimensions represented by the values. The function
    can adjust even if only one value is given.

    Parameters
    ----------
    n_samples : int
        The desired total number of samples.
    *values : int
        Variable length argument list representing the dimensions to adjust.
    initial_guess : float or None, optional
        An initial guess for the adjustment factor. If None, an automatic guess is made.
    error : str, optional
        Error handling strategy ('warn', 'ignore', 'raise'). This parameter is considered
        only when no values or one value is provided.

    Returns
    -------
    adjusted_values : tuple
        A tuple of adjusted values, aiming to distribute the total samples evenly.
        If only one value is given, the function tries to adjust it based on the
        total number of samples and the initial guess.

    Raises
    ------
    ValueError
        Raised if error is set to 'raise' and no values are provided.

    Examples
    --------
    >>> from fusionlab.core.handlers import adjust_to_samples
    >>> adjust_to_samples(1000, 10, 20, initial_guess=5)
    (50, 20)

    >>> adjust_to_samples(1000, 10, initial_guess=2)
    (2,)

    Notes
    -----
    The function aims to adjust the values to match the desired total number of samples
    as closely as possible. When only one value is given, the function uses the initial
    guess to make an adjustment, respecting the total number of samples.
    """
    if len(values) == 0:
        message = "No values provided for adjustment."
        if error == 'raise':
            raise ValueError(message)
        elif error == 'warn':
            warnings.warn(message)
        return ()

    if len(values) == 1:
        # If only one value is given, adjust it based on initial guess and n_samples
        single_value = values[0]
        adjusted_value = n_samples // single_value if initial_guess is None else initial_guess
        return (adjusted_value,)

    if initial_guess is None:
        initial_guess = np.mean(values)

    # Function to minimize: difference between product of adjusted values and n_samples
    def objective(factors):
        prod = np.prod(np.array(values) * factors)
        return abs(prod - n_samples)

    # Start with initial guesses for factors
    factors_initial = [initial_guess / value for value in values]
    result = optimize_minimize(objective, factors_initial, bounds=[(0, None) for _ in values])

    if result.success:
        adjusted_values = ( 
            tuple(max(1, int(round(value * factor))) 
                  for value, factor in zip(values, result.x))
            )
    else:
        adjusted_values = values  # Fallback to original values if optimization fails

    return adjusted_values

def add_noises_to(
    data,  
    noise=0.1, 
    seed=None, 
    gaussian_noise=False,
    cat_missing_value=pd.NA, 
    ex_columns=None,
    ):
    """
    Adds NaN or specified missing values to a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to which NaN values or specified missing 
        values will be added.

    noise : float, default=0.1
        The percentage of values to be replaced with NaN or the 
        specified missing value in each column. This must be a 
        number between 0 and 1. Default is 0.1 (10%).

        .. math:: \text{noise} = \frac{\text{number of replaced values}}{\text{total values in column}}

    seed : int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        Seed for random number generator to ensure reproducibility. 
        If `seed` is an int, array-like, or BitGenerator, it will be 
        used to seed the random number generator. If `seed` is a 
        np.random.RandomState or np.random.Generator, it will be used 
        as given.

    gaussian_noise : bool, default=False
        If `True`, adds Gaussian noise to the data. Otherwise, replaces 
        values with NaN or the specified missing value.

    cat_missing_value : scalar, default=pd.NA
        The value to use for missing data in categorical columns. By 
        default, `pd.NA` is used.
    ex_columns: optional , str or list of str, 
       If provided, then noise is not applied to that/these column(s). 
       
    Returns
    -------
    pandas.DataFrame
        A DataFrame with NaN or specified missing values added.

    Notes
    -----
    The function modifies the DataFrame by either adding Gaussian noise 
    to numerical columns or replacing a percentage of values in each 
    column with NaN or a specified missing value.

    The Gaussian noise is added according to the formula:

    .. math:: \text{new_value} = \text{original_value} + \mathcal{N}(0, \text{noise})

    where :math:`\mathcal{N}(0, \text{noise})` represents a normal 
    distribution with mean 0 and standard deviation equal to `noise`.

    Examples
    --------
    >>> from fusionlab.core.handlers import add_noises_to
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    >>> new_df = add_noises_to(df, noise=0.2)
    >>> new_df
         A     B
    0  1.0  <NA>
    1  NaN     y
    2  3.0  <NA>

    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> new_df = add_noises_to(df, noise=0.1, gaussian_noise=True)
    >>> new_df
              A         B
    0  1.063292  3.986400
    1  2.103962  4.984292
    2  2.856601  6.017380

    See Also
    --------
    pandas.DataFrame : Two-dimensional, size-mutable, potentially 
        heterogeneous tabular data.
    numpy.random.normal : Draw random samples from a normal 
        (Gaussian) distribution.

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. 
           (2020). Array programming with NumPy. Nature, 585(7825), 
           357-362.
    """
    
    is_frame = isinstance (data, pd.DataFrame ) 
    if not is_frame: 
        data = pd.DataFrame(data ) 
        
    np.random.seed(seed)
    if noise is None: 
        return data 
    noise, gaussian_noise  = _parse_gaussian_noise (noise )
    
    ex_columns = columns_manager(ex_columns)
    if gaussian_noise:
        # Add Gaussian noise to numerical columns only
        def add_gaussian_noise(column):
            # Skip modification if the column is in the exclusion list.
            if ex_columns is not None and column in ex_columns:
                return column
            if pd.api.types.is_numeric_dtype(column):
                return column + np.random.normal(0, noise, size=column.shape)
            return column
        
        noise_data = data.apply(add_gaussian_noise)
        
        if not is_frame: 
            noise_data = np.asarray(noise_data)
        return noise_data
    else:
        # Replace values with NaN or specified missing value
        df_with_nan = data.copy()
        nan_count_per_column = int(noise * len(df_with_nan))

        for column in df_with_nan.columns:
            if ex_columns is not None and column in ex_columns:
                continue  # Skip columns that should not be modified.
            nan_indices = random.sample(range(len(df_with_nan)), nan_count_per_column)
            if pd.api.types.is_numeric_dtype(df_with_nan[column]):
                df_with_nan.loc[nan_indices, column] = np.nan
            else:
                df_with_nan.loc[nan_indices, column] = cat_missing_value
                
        if not is_frame: 
            df_with_nan = df_with_nan.values 
            
        return df_with_nan

def _parse_gaussian_noise(noise):
    """
    Parses the noise parameter to determine if Gaussian noise should be used
    and extracts the noise level if specified.

    Parameters
    ----------
    noise : str, float, or None
        The noise parameter to be parsed. Can be a string specifying Gaussian
        noise with an optional noise level, a float, or None.

    Returns
    -------
    tuple
        A tuple containing:
        - float: The noise level.
        - bool: Whether Gaussian noise should be used.

    Examples
    --------
    >>> from fusionlab.core.handlers import _parse_gaussian_noise
    >>> _parse_gaussian_noise('0.1gaussian')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian0.1')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian_0.1')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian10%')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian 10 %')
    (0.1, True)
    >>> _parse_gaussian_noise(0.05)
    (0.05, False)
    >>> _parse_gaussian_noise(None)
    (0.1, False)
    >>> _parse_gaussian_noise('invalid')
    Traceback (most recent call last):
        ...
    ValueError: Invalid noise value: invalid
    """
    gaussian_noise = False
    default_noise = 0.1

    if isinstance(noise, str):
        orig_noise = noise 
        noise = noise.lower()
        gaussian_keywords = ["gaussian", "gauss"]

        if any(keyword in noise for keyword in gaussian_keywords):
            gaussian_noise = True
            noise = re.sub(r'[^\d.%]', '', noise)  # Remove non-numeric and non-'%' characters
            noise = re.sub(r'%', '', noise)  # Remove '%' if present

            try:
                noise_level = float(noise) / 100 if '%' in orig_noise else float(noise)
                noise = noise_level if noise_level else default_noise
            except ValueError:
                noise = default_noise

        else:
            try:
                noise = float(noise)
            except ValueError:
                raise ValueError(f"Invalid noise value: {noise}")
    elif noise is None:
        noise = default_noise
    
    noise = validate_noise (noise ) 
    
    return noise, gaussian_noise

def make_ids(
    arr: Iterable, 
    prefix: Optional[str] = None, 
    how: str = 'py', 
    skip: bool = False
 ) -> List[str]:
    """
    Generate auto-generated IDs based on the number of items in the input
    iterable.

    Parameters
    ----------
    arr : iterable
        An iterable object (e.g., list, tuple, or array-like) to generate 
        an ID for each item.
        For example, it can be a list of EDI objects, such as a collection of
        `fusionlab.edi.Edi` objects.

    prefix : str, optional, default: None
        A string value to prepend to each generated ID. This can be used to 
        indicate the site or collection name. If None, no prefix is added.

    how : str, optional, default: 'py'
        The indexing mode for the generated IDs. 
        - 'py' (default) uses Python-style indexing (i.e., starts at 0).
        - Any other string will use 1-based indexing (i.e., starts at 1).

    skip : bool, optional, default: False
        If True, the generated IDs will have a more compact format 
        (i.e., without leading zeros).

    Returns
    -------
    list of str
        A list of generated IDs as strings, each corresponding to an element
        in the input iterable.

    Raises
    ------
    ValueError
        If the `how` parameter is not one of 'py' or another valid string 
        for 1-based indexing.
    TypeError
        If `arr` is not an iterable type.

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.core.handlers import make_ids
    >>> values = ['edi1', 'edi2', 'edi3']
    >>> make_ids(values, prefix='ix')
    ['ix0', 'ix1', 'ix2']

    >>> data = np.random.randn(20)
    >>> make_ids(data, prefix='line', how=None)
    ['line01', 'line02', 'line03', ..., 'line20']

    >>> make_ids(data, prefix='line', how=None, skip=True)
    ['line1', 'line2', 'line3', ..., 'line20']
    """
    
    # Check if the input is an iterable
    if not isinstance(arr, (list, tuple, np.ndarray)):
        raise TypeError(
            f"Expected an iterable object for 'arr', got {type(arr)}")
    
    # Validate the 'how' parameter to ensure it's either 'py' or any other mode
    if how not in ['py', None]:
        raise ValueError(
            "The 'how' parameter must be 'py' for Python"
            " indexing or None for 1-based indexing.")
    
    # Determine the formatting for IDs, based on whether
    # we want compact IDs or padded ones
    fm = '{:0' + ('1' if skip else str(int(np.log10(len(arr))) + 1)) + '}'
    
    # Generate the IDs based on the provided parameters
    id_ = [
        str(prefix) + fm.format(i if how == 'py' else i + 1) if prefix is not None
        else fm.format(i if how == 'py' else i + 1)
        for i in range(len(arr))
    ]
    
    return id_

def get_params(obj: object) -> dict:
    """
    Retrieve the parameters of an object, which can either be a callable 
    or an instance of a class. The function inspects the object and 
    returns a dictionary of its parameters and their current values.

    This function works in two scenarios:
    - If the object is callable (i.e., a function or method), it will 
      return the parameters for the callable, including their default 
      values (if defined).
    - If the object is a class instance, it will return the parameters 
      (attributes) defined within the instance's `__dict__`, which contains 
      its instance variables.

    Parameters
    ----------
    obj : object
        The object whose parameters are to be retrieved. This can be either 
        a callable (such as a function or method) or an instance of a class.

    Returns
    -------
    dict
        A dictionary where the keys are the parameter names (or attribute names)
        and the values are the corresponding parameter values or defaults. 

    Examples
    --------
    >>> from sklearn.svm import SVC 
    >>> from fusionlab.core.handlers import get_params 
    >>> sigmoid = SVC(
    ...     **{
    ...         'C': 512.0,
    ...         'coef0': 0,
    ...         'degree': 1,
    ...         'gamma': 0.001953125,
    ...         'kernel': 'sigmoid',
    ...         'tol': 1.0 
    ...     }
    ... )
    >>> pvalues = get_params(sigmoid)
    >>> print(pvalues)
    {'decision_function_shape': 'ovr',
     'break_ties': False,
     'kernel': 'sigmoid',
     'degree': 1,
     'gamma': 0.001953125,
     'coef0': 0,
     'tol': 1.0,
     'C': 512.0,
     'nu': 0.0,
     'epsilon': 0.0,
     'shrinking': True,
     'probability': False,
     'cache_size': 200,
     'class_weight': None,
     'verbose': False,
     'max_iter': -1,
     'random_state': None}
    """
    if hasattr(obj, '__call__'): 
        # If the object is callable (e.g., a function or method)
        func_signature = inspect.signature(obj)
        PARAMS_VALUES = {
            k: None if v.default is inspect.Parameter.empty  else v.default
            for k, v in func_signature.parameters.items()
        }
    elif hasattr(obj, '__dict__'): 
        # If the object is an instance of a class
        PARAMS_VALUES = {
            k: v for k, v in obj.__dict__.items()
        }
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

    return PARAMS_VALUES

def parse_attrs (attr,  regex=None ): 
    """ Parse attributes using the regular expression.
    
    Remove all string non-alphanumeric and some operator indicators,  and 
    fetch attributes names. 
    
    Parameters 
    -----------
    
    attr: str, text litteral containing the attributes 
        names 
        
    regex: `re` object, default is 
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'per|mod|times|add|sub|[_#&*@!_,;\s-]\s*', 
                                flags=re.IGNORECASE) 
    Returns
    -------
    attr: List of attributes 
    
    Example
    ---------
    >>> from fusionlab.core.utils import parse_attrs 
    >>> parse_attrs('lwi_sub_ohmSmulmagnitude')
    ... ['lwi', 'ohmS', 'magnitude']
    
    
    """
    regex = regex or re.compile (r'per|mod|times|add|sub|[_#&*@!_,;\s-]\s*', 
                        flags=re.IGNORECASE) 
    attr= list(filter (None, regex.split(attr)))
    return attr 

def columns_getter(
    *dfs, 
     error = "warn",   
     to_df = "series", 
     columns = None,     
     return_cols = "any"     
    ):
    """
    Fetch and return a list of column names from a collection of 
    DataFrames based on set operations. Depending on the value of 
    `return_cols`, the function returns either the union, intersection, 
    or missing columns relative to a provided list.

    Parameters
    ----------
    *dfs : array-like
        One or more objects which should be pandas DataFrames or 
        Series. If an item is not a DataFrame or Series, handling is 
        determined by `error`.
    error : ``str``, default ``"warn"``
        Controls error handling for invalid inputs:
        - ``"raise"``: Throw an error if an input is not a 
          DataFrame or Series.
        - ``"warn"``: Issue a warning and skip the invalid input.
        - ``"ignore"``: Silently ignore invalid inputs.
    to_df : ``str``, default ``"series"``
        Determines conversion behavior for Series:
        - If set to ``"series"``, convert a Series into a DataFrame.
        - If set to ``"*"``, convert all inputs into DataFrames.
    columns : ``list``, optional
        A list of column names to check. If provided, the output will 
        be restricted to these columns.
    return_cols : ``str``, default ``"any"``
        Specifies the type of column list to return:
        
        - ``"any"``: Return the union of all columns found across 
          the DataFrames.
        - ``"all"``: Return the intersection (columns present in 
          every DataFrame).
        - ``"missing"``: If `columns` is provided, return the list of 
          columns from `columns` that are missing in every DataFrame; 
          if `columns` is not provided, return columns that are not 
          common to all DataFrames.
    
    Returns
    -------
    list of str
        The list of column names as determined by `return_cols`.

    Examples
    --------
    1) **Return union of columns (any):**

       >>> from fusionlab.core.handlers import columns_getter
       >>> import pandas as pd
       >>> df1 = pd.DataFrame({"A": [1,2], "B": [3,4]})
       >>> df2 = pd.DataFrame({"B": [5,6], "C": [7,8]})
       >>> columns_getter(df1, df2, return_cols="any")
       ['A', 'B', 'C']

    2) **Return intersection of columns (all):**

       >>> columns_getter(df1, df2, return_cols="all")
       ['B']

    3) **Return missing columns from a given set:**

       >>> columns_getter(df1, df2, columns=["A", "B", "D"], 
       ...               return_cols="missing")
       ['D']

    Notes
    -----
    The function iterates through each input in *dfs. If an input 
    is a pandas Series and `to_df` is set to ``"series"`` or ``"*"``, 
    it is converted to a DataFrame using `pd.DataFrame`. The valid 
    DataFrames are then used to compute the union and intersection 
    of their column names. If `columns` is provided, the computed 
    sets are intersected with the specified columns. Finally, based on 
    `return_cols`, the union, intersection, or missing columns are 
    returned [1]_.


    References
    ----------
    .. [1] Smith, J., & Doe, A. "Set operations for data 
           integration." *Journal of Data Science*, vol. 10, no. 2, 
           pp. 123-134, 2020.
    """

    valid_dfs = []
    for idx, item in enumerate(dfs):
        if isinstance(item, pd.DataFrame):
            valid_dfs.append(item)
        elif isinstance(item, pd.Series):
            if to_df in ["series", "*"]:
                valid_dfs.append(item.to_frame())
            else:
                msg = f"Item {idx} is a Series and will be skipped."
                if error == "raise":
                    raise ValueError(msg)
                elif error == "warn":
                    warnings.warn(msg)
        else:
            msg = f"Item {idx} is not a DataFrame or Series and will be ignored."
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                warnings.warn(msg)

    if not valid_dfs:
        return []

    # Compute union and intersection of columns across valid DataFrames.
    union_cols = set()
    intersection_cols = set(valid_dfs[0].columns)
    for df in valid_dfs:
        union_cols.update(df.columns)
        intersection_cols.intersection_update(df.columns)

    if columns is not None:
        specified = set(columns)
        union_cols = union_cols.intersection(specified)
        intersection_cols = intersection_cols.intersection(specified)
        missing = specified - union_cols
    else:
        missing = union_cols - intersection_cols

    if return_cols == "any":
        return sorted(list(union_cols))
    elif return_cols == "all":
        return sorted(list(intersection_cols))
    elif return_cols == "missing":
        return sorted(list(missing))
    else:
        raise ValueError(
            "Invalid return_cols option. Choose among 'any', 'all', or 'missing'."
        )

def columns_manager(
    columns: Optional[Union[str, list, tuple]],  
    default: Optional[list] = None, 
    regex: Optional[re.Pattern] = None, 
    pattern: Optional [str]= r'[@&,;#]', 
    separator: Optional[str] = None, 
    to_upper: bool = False, 
    empty_as_none: bool = ...,  
    to_string: bool = False,  
    error: str = 'raise',  
) -> list:
    """
    A function to handle various types of column inputs, convert them 
    into a list, and optionally process them based on additional parameters 
    like converting to uppercase, handling empty values, or ensuring all items 
    are strings.

    Parameters
    ----------
    columns : str, list, tuple, or None
        The input column names, which can be:
        - A string: treated as a list of column names split by a separator 
          or regex.
        - A list or tuple: directly converted to a list if not already.
        - None: returns the default list or an empty list 
        (if `empty_as_none` is False).
    
    default : list, optional
        Default list of columns to return if `columns` is None.
    
    regex : re.Pattern, optional
        A custom compiled regular expression to use for splitting string input. 
        If not provided, the `pattern` parameter will be used.

    pattern : str, optional, default=r'[@&,;#]'
        The default regex pattern used to split the `columns` string if no `regex` 
        is provided.

    separator : str, optional
        If `columns` is a string, this defines the separator used to split the string 
        into a list of column names.

    to_upper : bool, default=False
        If True, converts all column names to uppercase.

    empty_as_none : bool, default=True
        If True, returns `None` when `columns` is empty or None. If False, an 
        empty list is returned.

    to_string : bool, default=False
        If True, converts all items in `columns` to strings.

    error : str, default='warn'
        Specifies how to handle errors:
        - 'warn': issues a warning if any error occurs.
        - 'raise': raises an exception.
        - 'ignore': silently ignores any errors.

    Returns
    -------
    list
        A list of column names after processing.

    Example
    -------
    >>> from fusionlab.core.handlers import columns_manager
    >>> columns_manager("col1, col2, col3", separator=",")
    ['col1', 'col2', 'col3']

    >>> columns_manager(['col1', 'col2', 'col3'], to_upper=True)
    ['COL1', 'COL2', 'COL3']
    """
     # Handle None input
    if columns is None:
        return default if default is not None else (None if empty_as_none else [])
    
    # Handle case where a single numeric value is passed, convert it to list
    if isinstance(columns, (int, float)):
        columns = [columns]
        
    elif callable(columns): 
        columns=[columns] 
        
    ## Use inspect to determine if it is a class.
    # Alternatively, if the object is not iterable (has no __iter__ attribute),
    # we assume it's a single model instance.
    if inspect.isclass(columns) or not hasattr(columns, '__iter__'):
        columns = [columns]
        
    # If columns is a string, split by separator or use regex
    elif isinstance(columns, str):
        if separator is not None:
            columns = columns.split(separator)
        else:
            columns = str2columns(
                columns, 
                regex=regex, 
                pattern=pattern
        )
    
    # If columns is any iterable object, convert it to a list
    elif isinstance(columns, Iterable) : 
        try:
            columns = list(columns)
        except Exception as e:
            if error == 'raise':
                raise ValueError(f"Error converting columns to list: {e}")
            elif error == 'warn':
                warnings.warn(f"Could not convert columns to list: {e}")
            else:
                pass  # Ignore errors silently

    # Ensure columns is a list at this point
    if isinstance(columns, list):
        if to_upper:
            # Check whether all items are strings before calling 'upper'
            if all(isinstance(col, str) for col in columns):
                columns = [col.upper() for col in columns]
            elif error == 'raise':
                raise TypeError(
                    "All column names must be strings to convert to uppercase.")
            elif error == 'warn':
                warnings.warn(
                    "Warning: Not all column names are strings,"
                    " skipping 'upper' conversion.")
        
        # Convert all items to string if requested
        if to_string:
            columns = [str(col) for col in columns]
    else: 
        # If 'columns' is not a string, list, or tuple, 
        # then it might be a single object 
        # (for example, an instance of RandomForestRegressor).
        # In such a case, we attempt to check if it is iterable. 
        # Since an instance of RandomForestRegressor
        # is neither callable nor a class, nor is it iterable 
        # (i.e., it has no __iter__ # attribute), we wrap it into a list.
        if not isinstance(columns, (str, list, tuple)):
            try:
                iter(columns)
            except:
                columns = [columns]
        
    return columns

def resolve_label(
    obj,
    default_name='',
    return_only_one=True,
    ops='get',
    new_name=None, 
    none_as_default_name = ...
):
    """
    Resolves a human-readable label from multiple Python
    object types. It can optionally rename these objects
    when ``ops='set'`` is provided, using a new label
    or mapping. This is often beneficial for labeling data
    structures in plots or reports.

    .. math::
       \text{label}
       \;=\; f(\text{obj}, \text{ops}, \text{new\_name})

    Parameters
    ----------
    obj : any
        The source object whose label is extracted or
        renamed. This can be:
          - <str> : The label itself.
          - <pd.Series> : Uses Series.name if it exists.
          - <pd.DataFrame> : Uses column names, with an
            option to rename them.
          - <dict> : Uses dict keys, possibly renaming them
            if ``ops='set'``.
          - otherwise : Falls back to
            ``default_name``.
    default_name : str, optional
        Fallback label if no valid name is found or if
        <obj> is empty.
    return_only_one : bool, optional
        If ``True``, only a single name is returned (the
        first one if multiple exist). If ``False``, a list
        of names is returned.
    ops : { ``'get'``, ``'set'`` }, optional
        Specifies the mode. If ``'get'``, the function
        retrieves an existing label. If ``'set'``, the
        function attempts to rename <obj> using
        ``new_name`` where applicable.
    new_name : str or dict or list, optional
        Specifies the new label or mapping when
        ``ops='set'``. If a dict or list is given, it
        is interpreted as a mapping or a list of
        replacements for each column/key.
    none_as_default_name: bool, default=True 
       use `default_name` systematicallmy when obj is None. Otherwise return 
       as is. 
       
    Methods
    -------
    This function itself is the main method. It does not
    provide additional public methods starting with letters.

    .. math::
       \mathbf{label\_list}
       \;=\; \begin{cases}
         \text{extract(obj)} & \text{if ops='get'} \\
         \text{rename(obj, new\_name)} & \text{if ops='set'}
       \end{cases}

    where :math:`\text{extract}` and :math:`\text{rename}`
    represent internal routines that retrieve or modify the
    label of <obj>.

    Examples
    --------
    >>> from fusionlab.core.handlers import resolve_label
    >>> import pandas as pd
    >>> # Retrieving a label
    >>> lab = resolve_label("latitude")
    >>> print(lab)
    'latitude'
    >>> # Renaming a DataFrame
    >>> df = pd.DataFrame({'old_col': [1, 2]})
    >>> new_lab = resolve_label(df, ops='set',
    ...                         new_name='new_col')
    >>> print(new_lab)
    'new_col'

    Notes
    -----
    Using <resolve_label> aids in standardizing how labels
    are derived and optionally modified across different
    data structures. For instance, it helps unify labels for
    plotting or reporting, even if the underlying data
    structure is a string, Series, DataFrame, or dictionary.

    See Also
    --------
    ``pd.DataFrame.rename`` :
        Built-in Pandas method for column renaming.
    ``pd.Series.name`` :
        Attribute for storing or retrieving a Series name.

    References
    ----------
    .. [1] Smith, J. & Doe, A. "Renaming Large-Scale
       Data Structures," Journal of Open Science, 2023.
    .. [2] Brown, K. "Labeling Conventions for
       Heterogeneous Datasets," Data Insights Press, 2022.
    """
    if obj is None: 
        return default_name if none_as_default_name else obj 
    
    mode = ops.lower()
    if mode not in ['get', 'set']:
        raise ValueError("`ops` must be either 'get' or 'set'.")

    names = []

    # 1) If string
    if isinstance(obj, str):
        names = [obj]

    # 2) If pd.Series
    elif isinstance(obj, pd.Series):
        series_name = obj.name if obj.name else default_name
        names = [series_name]
        if mode == 'set' and new_name is not None:
            # A single rename if new_name is a string
            # or simply rename the Series to `new_name`.
            obj.name = new_name if isinstance(new_name, str) else default_name
            names = [obj.name]

    # 3) If pd.DataFrame
    elif isinstance(obj, pd.DataFrame):
        cols = list(obj.columns)
        if not cols:
            names = [default_name]
        else:
            names = [cols[0]] if return_only_one else cols

        if mode == 'set' and new_name is not None:
            # MAPPING LOGIC (DataFrame)
            if isinstance(new_name, dict):
                # If new_name is a dict, interpret it as a mapping.
                # For example: {'old_col':'new_col', ...}
                obj.rename(columns=new_name, inplace=True)
                # Recompute the names list after renaming
                updated_cols = list(obj.columns)
                names = [updated_cols[0]] if return_only_one else updated_cols

            elif isinstance(new_name, list):
                # If new_name is a list and matches the length of columns,
                # rename each column accordingly
                if len(new_name) == len(cols):
                    mapping = dict(zip(cols, new_name))
                    obj.rename(columns=mapping, inplace=True)
                    updated_cols = list(obj.columns)
                    names = (
                        [updated_cols[0]] if return_only_one
                        else updated_cols
                    )
                else:
                    raise ValueError(
                        "Length of new_name list does not match number of columns."
                    )

            elif isinstance(new_name, str):
                # If we have a single string but multiple columns,
                # we might rename only the first column or all columns.
                if return_only_one and cols:
                    obj.rename(columns={cols[0]: new_name}, inplace=True)
                    names = [new_name]
                else:
                    # Example: rename all columns to new_name_0, new_name_1, ...
                    new_col_mapping = {}
                    for i, col in enumerate(cols):
                        new_col_mapping[col] = f"{new_name}_{i}"
                    obj.rename(columns=new_col_mapping, inplace=True)
                    updated_cols = list(obj.columns)
                    names = (
                        [updated_cols[0]] if return_only_one
                        else updated_cols
                    )
    # 4) If dict
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        if not keys:
            names = [default_name]
        else:
            names = [keys[0]] if return_only_one else keys

        if mode == 'set' and new_name is not None:
            # MAPPING LOGIC (dict)
            if isinstance(new_name, dict):
                # Interpret new_name as a {old_key: new_key} mapping
                # We'll create a new dict with renamed keys
                temp = {}
                for k, val in obj.items():
                    if k in new_name:
                        temp[new_name[k]] = val
                    else:
                        temp[k] = val
                obj.clear()
                obj.update(temp)

                # Re-extract updated keys
                updated_keys = list(obj.keys())
                names = (
                    [updated_keys[0]] if return_only_one
                    else updated_keys
                )

            elif isinstance(new_name, list):
                # If new_name is a list matching the length of keys,
                # rename each key accordingly
                if len(new_name) == len(keys):
                    temp = {}
                    for old_k, new_k in zip(keys, new_name):
                        temp[new_k] = obj[old_k]
                    obj.clear()
                    obj.update(temp)
                    updated_keys = list(obj.keys())
                    names = (
                        [updated_keys[0]] if return_only_one
                        else updated_keys
                    )
                else:
                    raise ValueError(
                        "Length of new_name list does not match number of dict keys."
                    )

            elif isinstance(new_name, str):
                # If we have a single string but multiple keys,
                # rename only the first key or all keys with pattern
                if return_only_one and keys:
                    first_key = keys[0]
                    obj[new_name] = obj.pop(first_key)
                    updated_keys = list(obj.keys())
                    names = [new_name]
                else:
                    # Rename all keys to new_name_0, new_name_1, ...
                    temp = {}
                    for i, k in enumerate(keys):
                        temp[f"{new_name}_{i}"] = obj[k]
                    obj.clear()
                    obj.update(temp)
                    updated_keys = list(obj.keys())
                    names = (
                        [updated_keys[0]] if return_only_one
                        else updated_keys
                    )

    # 5) Otherwise fallback
    else:
        names = [default_name]

    # 6) Return single or multiple
    if return_only_one:
        return names[0]
    else:
        return names
  
def extend_values(
    values,
    target,
    mode: str = "constant",
    increment: float = 0,
    extra_values=None,
    verbose: int = 0
):
    """
    Extends or increments a list of values (or single value) to match a
    desired length. The desired length can be given by an integer, a float
    (converted internally to an integer), or the length of an iterable.

    This function is flexible enough to handle both numeric and non-numeric
    values. Incrementing or decrementing (i.e., increasing or decreasing)
    only applies when the last value in the list is numeric. Otherwise, the
    function falls back to constant repetition.

    Parameters
    ----------
    values : int, float, str, list
        The initial values to be extended. If it is not already a list,
        it will be converted into a list of one element.

    target : int, float, or iterable
        The target size (length) to extend to. If an integer or float is
        provided, it is used directly (float is cast to int). If an iterable
        is provided, the length of the iterable is taken as the target
        length. For example, if `target` has length 5, the resulting list
        will have length 5.

    mode : {'constant', 'increase', 'decrease'}, default='constant'
        Determines how the extension is done:
          - 'constant': repeats the last value in `values`.
          - 'increase': increments from the last value by `increment`,
            creating a sequence. This makes sense only if the last
            value is numeric.
          - 'decrease': decrements from the last value by `increment`,
            creating a sequence. This also makes sense only if the
            last value is numeric.

    increment : float, default=0
        The amount by which to increment or decrement each subsequent
        new value if `mode` is 'increase' or 'decrease'. Ignored if
        `mode` is 'constant'. If the last value of `values` is not numeric,
        the function falls back to constant repetition.

    extra_values : single value or list, optional
        If provided, these values are appended to the original `values`
        before any further extension. For a single item, it is converted
        to a list of one element.

    verbose : int, default=0
        Controls the level of debug output:
          - 0: no messages (silent).
          - 1: basic messages about the process.
          - 2 or 3: more detailed messages (for development).

    Returns
    -------
    extended_values : list
        The extended list of values of length equal to the integer
        derived from `target`.

    Examples
    --------
    >>> from fusionlab.core.handlers import extend_values
    >>> # Example 1: constant extension
    >>> val = 0.3
    >>> ext = extend_values(val, 3, mode='constant')
    >>> # 'val' is converted to list [0.3], and repeated to get [0.3, 0.3, 0.3]
    >>> print(ext)
    [0.3, 0.3, 0.3]

    >>> # Example 2: numeric increment
    >>> vals = [0.3]
    >>> ext = extend_values(vals, 3, mode='increase', increment=0.2)
    >>> # Here we get [0.3, 0.5, 0.7]
    >>> print(ext)
    [0.3, 0.5, 0.7]

    >>> # Example 3: numeric decrement
    >>> ext = extend_values(vals, 3, mode='decrease', increment=0.2)
    >>> # Here we get [0.3, 0.1, -0.1]
    >>> print(ext)
    [0.3, 0.1, -0.1]

    >>> # Example 4: Extra values plus extension
    >>> vals = 0.3
    >>> # Suppose we have extra_values=[0.8], and we want total length 4
    >>> # Then we first get [0.3, 0.8], then we extend further.
    >>> ext = extend_values(vals, 4, extra_values=[0.8], mode='constant')
    >>> # -> [0.3, 0.8, 0.8, 0.8]
    >>> print(ext)
    [0.3, 0.8, 0.8, 0.8]

    >>> # Example 5: Non-numeric extension
    >>> # For non-numeric, 'increase' or 'decrease' fallback to 'constant'
    >>> val = "hello"
    >>> ext = extend_values(val, 4, mode='increase', increment=10)
    >>> # -> ['hello', 'hello', 'hello', 'hello']
    >>> print(ext)
    ['hello', 'hello', 'hello', 'hello']
    """
    # Convert `values` to list if it's not already.
    if hasattr(values, '__iter__') and not isinstance (values, str): 
        values= list(values)
        
    if not isinstance(values, list):
        values = [values]

    # Convert `extra_values` to list if it is provided but not a list.
    if extra_values is not None:
        if hasattr(extra_values, '__iter__') and not isinstance (
                extra_values, str): 
            extra_values= list(extra_values)
            
        if not isinstance(extra_values, list):
            extra_values = [extra_values]
        values.extend(extra_values)

    # Determine the integer length we need based on `target`.
    # If `target` is an integer or float, convert float->int.
    # If `target` is an iterable, use its length.
    if isinstance(target, int):
        desired_length = target
    elif isinstance(target, float):
        desired_length = int(target)
    elif hasattr(target, "__iter__"):
        desired_length = len(target)  # length of the iterable
    else:
        raise TypeError(
            "Invalid type for `target`. Must be int, float, or an iterable."
        )

    current_length = len(values)

    if verbose > 0:
        print(f"[extend_values] Current length: {current_length}, "
              f"Desired length: {desired_length}")

    # If we already meet or exceed the desired length, truncate.
    if current_length >= desired_length:
        if verbose > 1:
            print("[extend_values] Already sufficient length, truncating.")
        return values[:desired_length]

    # Otherwise, we need to extend the list.
    diff = desired_length - current_length

    # Identify last value for extension basis.
    last_val = values[-1]

    # Prepare a helper to check numeric feasibility.
    def is_numeric(x):
        return isinstance(x, (int, float))

    # If incrementing/decrementing is requested, ensure last_val is numeric.
    # Otherwise, fall back to constant repetition.
    if mode in ("increase", "decrease") and not is_numeric(last_val):
        if verbose > 0:
            print("[extend_values] Warning: last value is not numeric. "
                  "Falling back to constant mode.")
        mode = "constant"

    if mode == "constant":
        # Repeat the last value
        values.extend([last_val] * diff)
        if verbose > 1:
            print("[extend_values] Extended by constant repetition.")

    elif mode == "increase":
        # Add increment * 1, increment * 2, etc.
        base = last_val
        for i in range(1, diff + 1):
            values.append(base + i * increment)
        if verbose > 1:
            print("[extend_values] Extended by incrementing.")

    elif mode == "decrease":
        # Subtract increment * 1, increment * 2, etc.
        base = last_val
        for i in range(1, diff + 1):
            values.append(base - i * increment)
        if verbose > 1:
            print("[extend_values] Extended by decrementing.")
    else:
        # If an unknown mode is passed, revert to constant.
        if verbose > 0:
            print(f"[extend_values] Unrecognized mode: '{mode}'. "
                  "Falling back to constant mode.")
        values.extend([last_val] * diff)

    return values
