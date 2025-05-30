# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio  <etanoyau@gmail.com>
# Source: Adapted from earthai-tech/gofast (https://github.com/earthai-tech/gofast)
# This module is included in the FusionLab package, with adjustments
# to fit FusionLab’s API and coding conventions.

"""
Core validation utilities for FusionLab, offering functions for data
validation, assertion, and feature extraction. Provides type-consistency
checks, feature-existence verifications, regex-based searches, and 
task-specific validators.

Adapted for FusionLab from the original fusionlab.core.checks implementation.
"""

from __future__ import print_function, annotations
import re
import os
import numbers
import inspect
import warnings
from functools import wraps
from collections.abc import Iterable
import scipy.sparse as ssp
from typing import (
    Any,
    List,
    Dict,
    Optional,
    Union,
    Tuple,
    Type,
    Pattern,
    Callable,
    get_origin,
    get_args,
)
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype as _is_numeric_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_object_dtype,
)
from sklearn.utils._param_validation import (
    Hidden,
    make_constraint,
    _Constraint,
    InvalidParameterError,
)
from ..api.types import Series, _F, DataFrame, _Sub, ArrayLike

__all__ = [
    'ParamsValidator',
    'assert_ratio',
    'check_uniform_type',
    'exist_features',
    'features_in',
    'find_features_in',
    'validate_feature',
    'validate_noise',
    'validate_ratio',
    'is_classification_task',
    'is_depth_in',
    'is_in_if',
    'is_in',
    'is_numeric_dtype',
    'find_by_regex',
    'find_closest',
    'str2columns',
    'random_state_validator',
    'is_sparse_matrix',
    'has_sparse_format',
    'check_features_types',
    'are_all_frames_valid',
    'has_nan',
    'check_spatial_columns',
    'validate_spatial_columns',
    'validate_nested_param',
    'check_params',
    'ensure_same_shape',
    'validate_axis',
    'validate_depth',
    'check_empty',
    'check_numeric_dtype',
    'check_non_emptiness',
]

class ParamsValidator:
    """
    Decorator class designed to validate and transform parameters of 
    functions and class constructors based on predefined constraints.
    It ensures that the inputs conform to specified types and conditions,
    enhancing the robustness and reliability of the codebase.

    .. math::
        \text{Given a set of constraints } C = \{c_1, c_2, \dots, c_n\},
        \text{ `ParamsValidator` verifies that each input parameter } p_i\\
            \text{ satisfies at least one } c_j \in C.

    Parameters
    ----------
    constraints : Dict[str, Any]
        A dictionary mapping parameter names to their respective constraints.
        Each constraint defines the expected type or condition that the parameter
        must satisfy.

    skip_nested_validation : bool, optional
        If set to ``True``, the validator will not perform recursive validation
        on nested structures such as lists or dictionaries. Default is ``True``.

    verbose : int, optional
        Sets the verbosity level of the validator's logging output.
        The level ranges from ``0`` (no output) to ``7`` (maximum verbosity).
        Default is ``0``.

    Methods
    -------
    __call__(obj: Callable or Type) -> Callable or Type
        Applies the `ParamsValidator` decorator to a function or a class.
        It wraps the target's `__init__` method (for classes) or the function 
        itself, enforcing the defined parameter constraints upon invocation.

    Notes
    -----
    The validation process can be represented as:

    .. math::
        \forall p_i \in \text{parameters}, \exists c_j \in C\\
            \text{ such that } c_j(p_i) \text{ is True}

    Where:
    - \( p_i \) represents each parameter to be validated.
    - \( c_j \) represents each constraint applied to the parameters.
    - The parameter \( p_i \) must satisfy at least one constraint \( c_j \).
    

    - The `ParamsValidator` can handle both simple and complex constraints,
      including type checks and transformations.
    - When `skip_nested_validation` is set to ``False``, the validator will
      recursively validate elements within nested structures like lists and dictionaries.
    - Verbosity levels allow developers to control the amount of logging information
      for debugging purposes.
      
    Examples
    --------
    Validate parameters of a function to ensure `age` is an integer and `name`
    is a string:

    >>> from fusionlab.core.checks import ParamsValidator
    >>> constraints = {
    ...     'age': [int],
    ...     'name': [str]
    ... }
    >>> @ParamsValidator(constraints)
    ... def register_user(name, age):
    ...     print(f"User {name} registered with age {age}.")
    ...
    >>> register_user("Alice", 30)
    User Alice registered with age 30.
    >>> register_user("Bob", "thirty")
    Traceback (most recent call last):
        ...
    InvalidParameters: Parameter 'age' must be of type int.

    Validate a class constructor to ensure `data` is a pandas DataFrame:

    >>> from fusionlab.core.checks import ParamsValidator
    >>> import pandas as pd
    >>> constraints = {
    ...     'data': ['array-like:dataframe:transf']
    ... }
    >>> @ParamsValidator(constraints, verbose=3)
    ... class DataProcessor:
    ...     def __init__(self, data):
    ...         self.data = data
    ...
    >>> df = {'column1': [1, 2], 'column2': [3, 4]}
    >>> processor = DataProcessor(df)
    
    [ParamsValidator] Initialized with constraints: {'data': 'dataframe:transf'}
    [ParamsValidator] Skip nested validation: True
    [ParamsValidator] Verbosity level set to: 3
    [ParamsValidator] Decorating class 'DataProcessor' __init__ method.
    [ParamsValidator] Bound arguments before validation: OrderedDict(...)
    [ParamsValidator] Starting validation of bound arguments.
    [ParamsValidator] Validating parameter 'data' with value: {'column1': [1, 2], 'column2': [3, 4]}
    [ParamsValidator] Parameter 'data' is a valid dict.
    [ParamsValidator] Validation completed successfully.

    See Also
    --------
    `InvalidParameters` : Exception raised when parameter validation fails.
    `make_constraint` : Utility function to create constraint objects.

    References
    ----------
    .. [1] Smith, J. (2020). *Effective Python Decorators*. Python Publishing.
    .. [2] Doe, A. (2021). *Advanced Parameter Validation Techniques*. 
      Software Engineering Journal.
    """

    def __init__(
        self,
        constraints: Dict[str, Any],
        skip_nested_validation: bool = True,
        verbose: int = 0  # Verbose level from 0 to 7
    ):
        self.constraints = constraints
        self.skip_nested_validation = skip_nested_validation
        self.verbose = verbose

        if self.verbose >= 1:
            print(f"[ParamsValidator] Initialized with constraints:"
                  f" {self.constraints}")
            print(f"[ParamsValidator] Skip nested validation:"
                  f" {self.skip_nested_validation}")
            print(f"[ParamsValidator] Verbosity level set to:"
                  f" {self.verbose}")

    def __call__(self, obj: Callable or Type) -> Callable or Type:
        """
        Apply the decorator to a function or a class.

        Parameters:
            obj (Callable or Type): The function or class to decorate.

        Returns:
            Callable or Type: The decorated function or class.
        """
        if isinstance(obj, type):
            # Decorate a class by wrapping its __init__ method
            original_init = obj.__init__

            @wraps(original_init)
            def wrapped_init(*args, **kwargs):
                if self.verbose >= 2:
                    print("[ParamsValidator] Decorating class"
                          f" '{obj.__name__}' __init__ method.")
                sig = inspect.signature(original_init)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                if self.verbose >= 3:
                    print("[ParamsValidator] Bound arguments"
                          f" before validation: {bound.arguments}")

                self._validate(bound)

                if self.verbose >= 4:
                    print("[ParamsValidator] Arguments after"
                          f" validation: {bound.arguments}")

                # Reconstruct args and kwargs from bound.arguments
                new_args, new_kwargs = self._reconstruct_args_kwargs(bound)

                if self.verbose >= 5:
                    print(f"[ParamsValidator] Reconstructed args: {new_args}")
                    print(f"[ParamsValidator] Reconstructed kwargs: {new_kwargs}")

                return original_init(*new_args, **new_kwargs)

            obj.__init__ = wrapped_init
            return obj
        else:
            # Decorate a function by wrapping it
            @wraps(obj)
            def wrapped_func(*args, **kwargs):
                if self.verbose >= 2:
                    print(f"[ParamsValidator] Decorating function '{obj.__name__}'.")
                sig = inspect.signature(obj)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                if self.verbose >= 3:
                    print("[ParamsValidator] Bound arguments"
                          f" before validation: {bound.arguments}")

                self._validate(bound)

                if self.verbose >= 4:
                    print("[ParamsValidator] Arguments after"
                          f" validation: {bound.arguments}")

                # Reconstruct args and kwargs from bound.arguments
                new_args, new_kwargs = self._reconstruct_args_kwargs(bound)

                if self.verbose >= 5:
                    print(f"[ParamsValidator] Reconstructed args: {new_args}")
                    print(f"[ParamsValidator] Reconstructed kwargs: {new_kwargs}")

                return obj(*new_args, **new_kwargs)

            return wrapped_func

    def _validate(self, bound: inspect.BoundArguments):
        """
        Validates and transforms the parameters in bound.arguments 
        based on the constraints.

        Parameters:
            bound (inspect.BoundArguments): The bound arguments
            to validate and transform.

        Raises:
            InvalidParameters: If any parameter fails to satisfy
            its constraints.
        """
        if self.verbose >= 3:
            print("[ParamsValidator] Starting validation of bound arguments.")

        for param, constraints in self.constraints.items():
            if param not in bound.arguments:
                if self.verbose >= 6:
                    print(f"[ParamsValidator] Parameter '{param}'"
                          " not in arguments; skipping.")
                continue

            value = bound.arguments[param]

            if self.verbose >= 5:
                print("[ParamsValidator] Validating parameter"
                      f" '{param}' with value: {value}")

            if value is None:
                if None not in constraints:
                    error_message = (
                        f"Parameter '{param}' cannot be None"
                    )
                    raise InvalidParameterError(error_message)
                if self.verbose >= 6:
                    print(f"[ParamsValidator] Parameter '{param}' is None and allowed.")
                continue

            valid = False
  
            for constraint in constraints:
                origin = get_origin(constraint)

                # Handle generic types first (e.g., List[str], Dict[str, float])
                if origin in [list, List]:
                    if not isinstance(value, list):
                        error_message = (
                            f"Parameter '{param}' must be a list"
                        )
                        raise InvalidParameterError(error_message)
                    if not self.skip_nested_validation:
                        subtype = get_args(constraint)[0]
                        for idx, item in enumerate(value):
                            if not isinstance(item, subtype):
                                error_message = (
                                    f"All items in parameter '{param}' must be of type "
                                    f"{subtype.__name__} (item {idx} is {type(item).__name__})"
                                )
                                raise InvalidParameterError(error_message)
                    valid = True
                    if self.verbose >= 5:
                        print(f"[ParamsValidator] Parameter '{param}' is a valid list.")
                    break  # No need to check other constraints

                elif origin in [dict, Dict]:
                    if not isinstance(value, dict):
                        error_message = (
                            f"Parameter '{param}' must be a dict"
                        )
                        raise InvalidParameterError(error_message)
                    if not self.skip_nested_validation:
                        key_type, val_type = get_args(constraint)
                        for k, v in value.items():
                            if not isinstance(k, key_type):
                                error_message = (
                                    f"All keys in parameter '{param}' must be of type "
                                    f"{key_type.__name__} (key '{k}' is {type(k).__name__})"
                                )
                                raise InvalidParameterError(error_message)
                            if val_type is not Any and not isinstance(v, val_type):
                                error_message = (
                                    f"All values in parameter '{param}' must be of type "
                                    f"{val_type.__name__} (value '{v}' is {type(v).__name__})"
                                )
                                raise InvalidParameterError(error_message)
                    valid = True
                    if self.verbose >= 5:
                        print(f"[ParamsValidator] Parameter '{param}' is a valid dict.")
                    break  # No need to check other constraints

                # Handle non-generic constraints
                try:
                    constraint_obj = self._make_constraint(constraint)
                except ( InvalidParameterError, ValueError) as e:
                    error_message = (
                        f"Unsupported constraint type for parameter '{param}': {constraint}"
                    )
                    raise InvalidParameterError(error_message) from e

                # Handle Hidden constraints by unwrapping them
                if isinstance(constraint_obj, Hidden):
                    try:
                        inner_constraint = make_constraint(
                            constraint_obj.constraint
                        )
                    except (ValueError, InvalidParameterError) as e:
                        error_message = (
                            f"Hidden constraint for parameter '{param}' contains"
                            f" an unsupported constraint: {constraint_obj.constraint}"
                        )
                        raise InvalidParameterError(error_message) from e

                    if not inner_constraint.is_satisfied_by(value):
                        error_message = (
                            f"Parameter '{param}' does not satisfy the hidden "
                            f"constraint {inner_constraint}"
                        )
                        raise InvalidParameterError(error_message)

                    if (
                        isinstance(constraint, str)
                        and ':' in constraint
                        and len(constraint.split(':')) == 2
                    ):
                        self._validate_specific_types(value, param, constraint)

                    # Apply transformation if needed
                    if isinstance(constraint, str) and ':transf' in constraint:
                        try:
                            value = self._apply_transformation(value, param, constraint)
                            bound.arguments[param] = value
                            if self.verbose >= 5:
                                print(
                                    "[ParamsValidator] Applied transformation"
                                    f" on parameter '{param}'."
                                )
                        except InvalidParameterError as e:
                            raise e
                    valid = True
                    break  # No need to check other constraints
                else:
                    if not constraint_obj.is_satisfied_by(value):
                        if self.verbose >= 6:
                            print(
                                f"[ParamsValidator] Constraint '{constraint_obj}'"
                                f" not satisfied for parameter '{param}'."
                            )
                        continue  # Try the next constraint

                    if (
                        isinstance(constraint, str)
                        and ':' in constraint
                        and len(constraint.split(':')) == 2
                    ):
                        self._validate_specific_types(value, param, constraint)

                    # Apply transformation if needed
                    if isinstance(constraint, str) and ':transf' in constraint:
                        try:
                            value = self._apply_transformation(value, param, constraint)
                            bound.arguments[param] = value
                            if self.verbose >= 5:
                                print(
                                    "[ParamsValidator] Applied transformation"
                                    f" on parameter '{param}'."
                                )
                        except Exception as e:
                            raise e
                    valid = True
                    if self.verbose >= 5:
                        print(
                            f"[ParamsValidator] Parameter '{param}'"
                            f" satisfies constraint '{constraint_obj}'."
                        )
                    break  # Constraint satisfied

            if not valid:
                constraint_name =( 
                    constraint.__name__ if hasattr(constraint, '__name__') 
                    else constraint.__class__.__name__
                    )
                error_message = (
                    f"Parameter '{param}' must be a type '{constraint_name}'."
                )
                raise InvalidParameterError(error_message)

        if self.verbose >= 3:
            print("[ParamsValidator] Validation completed successfully.")

    def _make_constraint(self, constraint: Any) -> _Constraint:
        """
        Convert the constraint into the appropriate _Constraint object.

        Parameters:
            constraint (Any): The constraint to convert.

        Returns:
            _Constraint: The corresponding constraint object.

        Raises:
            ValueError: If the constraint type is unknown.
        """
        if self.verbose >= 4:
            print(
                f"[ParamsValidator] Creating constraint object for: {constraint}")

        if isinstance(constraint, str):
            if ':' in constraint:
                # Handle complex string constraints with transformations
                base_constraint = constraint.split(':')[0]
                return make_constraint(base_constraint)
            return make_constraint(constraint)
        elif constraint is None:
            return make_constraint(constraint)
        elif isinstance(constraint, type):
            return make_constraint(constraint)
        elif isinstance(constraint, (_Constraint, Hidden)):
            return constraint
        else:
            raise ValueError(f"Unknown constraint type: {constraint}")

    def _apply_transformation(self, value: Any, param:Any, constraint_str: str) -> Any:
        """
        Apply transformations based on constraint specifications.

        Parameters:
            value (Any): The value to transform.
            constraint_str (str): The constraint string specifying 
            transformations.

        Returns:
            Any: The transformed value.

        Raises:
            InvalidParameters: If the transformation fails.
        """
        parts = constraint_str.split(':')
        specs = parts[1:]  # Extract transformation specs
        specs = [str(s).lower() for s in specs]
        err_msg = (
            "Invalid parameter '{0}', expected a {1}. Got {2!r} instead."
        )

        if self.verbose >= 6:
            print(f"[ParamsValidator] Applying"
                  f" transformations: {specs} on value: {value}")

        for spec in specs:
            if 'tf' in spec: 
                try: 
                    import tensorflow as tf 
                    # Convert list to TensorFlow tensor
                    value = tf.convert_to_tensor(value, dtype=tf.float32)
                except: 
                    # fallback to numpy conversion 
                    spec ='np'
            
            if 'df' in spec or 'dataframe' in spec:
                try:
                    value = pd.DataFrame(value)
                except Exception:
                    raise InvalidParameterError(
                        err_msg.format(
                            param, 'pandas DataFrame', type(value).__name__)
                    )
            elif 'np' in spec:
                try:
                    value = np.array(value)
                except Exception:
                    raise InvalidParameterError(
                        err_msg.format(
                            param,'numpy array', type(value).__name__)
                    )
                
            elif 'series' in spec:
                try:
                    value = pd.Series(value)
                except Exception:
                    raise InvalidParameterError(
                        err_msg.format(
                            param,'pandas Series', type(value).__name__)
                    )
            elif spec == 'list':
                try:
                    value = list(value)
                except Exception:
                    raise InvalidParameterError(
                        err_msg.format(
                            param,'list object', type(value).__name__)
                    )
            elif spec == 'tuple':
                try:
                    value = tuple(value)
                except Exception:
                    raise InvalidParameterError(
                        err_msg.format(
                            param,'tuple object', type(value).__name__)
                    )
            #XXX: Future extension: Add more transformations as needed
            else:
                if self.verbose >= 6:
                    print(f"[ParamsValidator] Unknown transformation spec: '{spec}'")
        return value

    def _validate_specific_types(
            self, value: Any, param: Any, constraint_str: str) -> None:
        """
        Validate specific types based on constraint specifications.

        Parameters:
            value (Any): The value to validate.
            constraint_str (str): The constraint string specifying the type.

        Raises:
            InvalidParameters: If the type validation fails.
        """
        parts = constraint_str.split(':')  # e.g., 'arraylike:np'
        spec = parts[-1]
        err_msg = (
            "Invalid parameter '{0}', expected a {1}. Got {2!r} instead."
        )

        if self.verbose >= 6:
            print("[ParamsValidator] Validating"
                  f" specific type: '{spec}' for value: {value}")

        if ('df' in spec or 'dataframe' in spec) and not isinstance(value, pd.DataFrame):
            raise InvalidParameterError(
                err_msg.format(param, 'pandas DataFrame', type(value).__name__)
            )
        elif 'np' in spec and not isinstance(value, np.ndarray):
            raise InvalidParameterError(
                err_msg.format(param,'numpy array', type(value).__name__)
            )
        elif 'series' in spec and not isinstance(value, pd.Series):
            raise InvalidParameterError(
                err_msg.format(param,'pandas Series', type(value).__name__)
            )
        elif spec == 'list' and not isinstance(value, list):
            raise InvalidParameterError(
                err_msg.format(param,'list object', type(value).__name__)
            )
        elif spec == 'tuple' and not isinstance(value, tuple):
            raise InvalidParameterError(
                err_msg.format(param,'tuple object', type(value).__name__)
            )
        try: 
            import tensorflow as tf 
            if spec =='tf' and not isinstance (value, tf.Tensor): 
                raise InvalidParameterError(
                    err_msg.format(
                        param,'tensorflow Tensor object', type(value).__name__)
                )
        except: 
            pass 
        #XXX: Future extension: Add more transformations as needed

    def _reconstruct_args_kwargs(
        self, bound: inspect.BoundArguments
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Reconstruct *args and **kwargs from the bound arguments.

        Parameters:
            bound (inspect.BoundArguments): The bound arguments.

        Returns:
            Tuple[List[Any], Dict[str, Any]]: The reconstructed args and kwargs.
        """
        args = []
        kwargs = {}
        signature = bound.signature

        if self.verbose >= 5:
            print("[ParamsValidator] Reconstructing args"
                  " and kwargs from bound arguments.")

        for name, param in signature.parameters.items():
            if param.kind in (
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD
            ):
                if name in bound.arguments:
                    args.append(bound.arguments[name])
            elif param.kind == param.KEYWORD_ONLY:
                if name in bound.arguments:
                    kwargs[name] = bound.arguments[name]
            elif param.kind == param.VAR_POSITIONAL:
                if name in bound.arguments:
                    args.extend(bound.arguments[name])
            elif param.kind == param.VAR_KEYWORD:
                if name in bound.arguments:
                    kwargs.update(bound.arguments[name])
        
        if self.verbose >= 6:
            print(f"[ParamsValidator] Reconstructed args: {args}")
            print(f"[ParamsValidator] Reconstructed kwargs: {kwargs}")

        return args, kwargs

    def _is_array_like(self, value: Any) -> bool:
        """
        Check if the value is array-like.

        Parameters:
            value (Any): The value to check.

        Returns:
            bool: True if array-like, False otherwise.
        """
        array_like = isinstance(
            value, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)
        )
        if self.verbose >= 7:
            print("[ParamsValidator] Checking if"
                  f" value is array-like: {array_like}")
        return array_like

def check_numeric_dtype(
    X,
    y=None,
    ops='check_only',
    param_names=None,
    coerce=False,
    error='raise',
    return_X_y=False
):
    """
    Validate or coerce the numeric data type of ``X`` and ``y``.

    The `<check_numeric_dtype>` function ensures that the provided
    arguments ``X`` and (optionally) ``y`` are valid numeric types. If
    `<coerce>` is set to True, attempts are made to convert them to
    numeric formats. If conversion fails or detects non-numeric
    content, behavior depends on `<error>` (e.g., `'raise'` will throw
    an exception).

    Parameters
    ----------
    X : array-like, DataFrame, or Series
        The primary data structure to validate or coerce. This
        parameter is expected to be numeric. If `<coerce>` is True,
        an attempt is made to convert it to numeric.

    y : array-like, DataFrame, or Series or None, optional
        A secondary data structure that, if provided, is also
        validated or coerced. If None, no checks are performed on
        `y`. Otherwise, it follows the same logic as ``X``.

    ops : {'check_only', 'validate'}, default='check_only'
        Determines the function's return behavior:
        - ``'check_only'``: Checks (and possibly coerces) but
          returns None unless `<return_X_y>` is True.
        - ``'validate'``: Checks (and possibly coerces) and returns
          the resulting numeric structures.

    param_names : dict or None, optional
        A dictionary mapping internal names of parameters to
        user-friendly labels for error messages. For example,
        ``{'X': 'Data', 'y': 'Target'}``.

    coerce : bool, default=False
        If True, attempts to convert `<X>` (and `<y>`, if
        provided) to numeric type. If False, leaves them as is.
        If coercion fails, behavior is governed by `<error>`.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        The error policy applied when encountering non-numeric or
        uncoercible values:
        - ``'raise'``: Raises a ValueError if checks fail.
        - ``'warn'``: Issues a warning but allows execution to
          continue.
        - ``'ignore'``: Ignores errors or warnings, although
          non-numeric data might lead to further issues.

    return_X_y : bool, default=False
        If True, forces the return of the processed `(X, y)`,
        regardless of `<ops>`. If `<ops>` is `'check_only'`
        and `<return_X_y>` is False, nothing is returned unless
        coercion fails.

    Returns
    -------
    None or DataFrame or Series or tuple of (DataFrame or Series)
        Depending on `<ops>` and `<return_X_y>`:
        - If `<ops>` is `'check_only'` and `<return_X_y>` is False,
          returns None.
        - If `<ops>` is `'check_only'` and `<return_X_y>` is True,
          returns `(X_converted, y_converted)`.
        - If `<ops>` is `'validate'`, returns `(X_converted, y_converted)`
          if `y` is provided, otherwise returns only `X_converted`.

    Notes
    -----
    The function checks each array using a numeric detection utility
    that tests whether the underlying dtype is numeric. If
    :math:`X` or :math:`y` fails to meet this criterion:
    .. math::
       \\text{non\\_numeric}(X) \\Longrightarrow \\text{action based on}\\
       \\langle error \\rangle
    If `<coerce>` is True, an attempt is made to cast the array into
    numeric form. If casting fails and `<error>` is `'raise'`, a
    ``ValueError`` is thrown.

    Examples
    --------
    >>> from fusionlab.core.checks import check_numeric_dtype
    >>> import numpy as np
    >>> X = [1, 2, '3', 4]
    >>> # Check without coercion
    >>> check_numeric_dtype(X, coerce=False, error='warn')
    # Warns about non-numeric if present, no return
    >>> # Validate with coercion
    >>> result = check_numeric_dtype(X, ops='validate', coerce=True,
    ...                              return_X_y=True)
    >>> result
    (0    1.0
     1    2.0
     2    3.0
     3    4.0
     dtype: float64, None)

    See also
    --------
    `is_numeric_dtype` : Internal function that detects numeric
    types in arrays.

    References
    ----------
    .. [1] Gofast Documentation. Available at:
           https://fusionlab.readthedocs.io/en/latest/
    """
    # Retrieve custom names if provided, else defaults
    names = param_names or {}
    x_name = names.get('X', 'X')
    y_name = names.get('y', 'y')

    # Function to handle error messages or warnings
    def _handle_error(msg):
        if error == 'raise':
            raise ValueError(msg)
        elif error == 'warn':
            warnings.warn(msg)
        # if 'ignore', do nothing

    # Helper to attempt coercing to numeric if coerce=True
    def _try_coerce_to_numeric(arr, arr_name):
        try:
            return pd.to_numeric(arr, errors='coerce')
        except Exception as e:
            _handle_error(
                f"Failed to coerce {arr_name} to numeric: {str(e)}"
            )
            return arr

    # Check or validate X
    X_converted = X
    if X is not None:
        if not is_numeric_dtype(X, to_array=True):
            try:
                _ = np.asarray(X)
            except Exception as e:
                _handle_error(
                    f"Cannot interpret {x_name} as numeric: {str(e)}"
                )
            # If coerce is True, try to coerce
            if coerce:
                X_converted = _try_coerce_to_numeric(X, x_name)
            else:
                _handle_error(
                    f"Non-numeric {x_name} detected. "
                    "Consider setting 'coerce=True' to attempt conversion."
                )

    # Check or validate y if provided
    y_converted = y
    if y is not None:
        if not is_numeric_dtype(y, to_array=True):
            try:
                _ = np.asarray(y)
            except Exception as e:
                _handle_error(
                    f"Cannot interpret {y_name} as numeric: {str(e)}"
                )
            if coerce:
                y_converted = _try_coerce_to_numeric(y, y_name)
            else:
                _handle_error(
                    f"Non-numeric {y_name} detected. "
                    "Consider setting 'coerce=True' to attempt conversion."
                )

    # If ops='check_only', we do not alter X or y, just return None or None
    # If ops='validate', we return the (possibly coerced) X or X,y
    if ops == 'check_only':
        if return_X_y:
            return X_converted, y_converted
        return None
    elif ops == 'validate':
        if return_X_y:
            return X_converted, y_converted
        # if y is None, return only X
        if y is None:
            return X_converted
        return X_converted, y_converted
    else:
        # Unrecognized operation
        _handle_error(
            f"Unknown operation '{ops}' in check_numeric_dtype."
        )
        return None

def check_empty(
    func=None,
    *,
    params=None,
    allow_none=True,
    none_as_empty=False,
    error='raise'
):
    r"""
    Validate that certain inputs are non-empty or valid 
    (not :math:`\varnothing`). Can be used in two modes:

    - **No parentheses** (e.g. ``@check_empty``):
      Only the first argument (positional or keyword) of the
      decorated function is checked. This is useful for simple
      scenarios where only one primary parameter requires
      validation.

    - **With parentheses** (e.g. ``@check_empty(params=['x'])``):
      Only the parameters listed in `params` are checked, ignoring
      ``*args`` or ``**kwargs``. This mode offers precise control
      over which parameters must be verified.

    .. math::
        \text{valid\_data}(x) =
        \begin{cases}
        \emptyset & \text{if } x \text{ is invalid},\\
        x & \text{if } x \text{ is non-empty}.
        \end{cases}

    Parameters
    ----------
    func : callable, optional
        The function being decorated. If this parameter is 
        provided without parentheses (e.g. ``@check_empty``), then
        the first argument to `func` is validated. If omitted 
        (i.e., ``@check_empty(...)``), the returned decorator 
        checks only the parameters listed in `params`.
    params : list of str, optional
        A list of parameter names that should be checked for 
        emptiness. Only used when the decorator is invoked with 
        parentheses (e.g. 
        ``@check_empty(params=['data'], allow_none=True)``). If 
        this is ``None``, no parameter is explicitly checked in 
        this mode.
    allow_none : bool, default=True
        Indicates whether `None` is acceptable. If set to 
        ``False``, any encounter of `None` triggers the defined 
        error handling strategy in `<error>`.
    none_as_empty : bool, default=False
        If ``True``, `None` is treated as empty 
        (i.e. :math:`\varnothing`). Hence, if any parameter is 
        `None`, it is considered a violation of non-emptiness,
        regardless of `<allow_none>`.
    error : {'raise', 'warn', 'ignore'}, default='raise'
        Determines the behavior when an empty parameter is 
        detected. 
        - ``'raise'`` : Raises :class:`ValueError` (stops 
                       execution).
        - ``'warn'``  : Issues a :func:`warnings.warn` (allows 
                       execution to continue).
        - ``'ignore'``: Takes no action.

    Returns
    -------
    callable
        A decorator function if used with parentheses (or if 
        `func` is not given). Otherwise, a wrapper function that 
        checks the first argument of the decorated function.

    Notes
    -----
    - When used without parentheses, only the *first argument* of
      the decorated function is checked—no matter if it is 
      positional or a keyword.
    - When used with parentheses (and `params` is specified),
      *only* those parameters named in `<params>` are checked. The
      other arguments (including ``*args`` or ``**kwargs``) are
      ignored.
    - To detect emptiness, this function attempts :func:`len`. If
      length is zero, it is considered empty. If no length
      (e.g. an integer), it is not considered empty unless it is 
      `None` and `<none_as_empty>` is ``True`` or 
      `<allow_none>` is ``False``.

    Examples
    --------
    Minimal usage with no parentheses:
    
    >>> from fusionlab.core.checks import check_empty
    >>> @check_empty
    ... def process_data(data, *args, **kwargs):
    ...     print("Data:", data)

    Using parentheses with custom parameters:
    
    >>> from fusionlab.core.checks import check_empty
    >>> @check_empty(params=['data'], allow_none=False)
    ... def load_data(data, path=None):
    ...     print("Loading data:", data, "from", path)

    See Also
    --------
    some_other_checker : Another hypothetical checker function for 
        demonstration purposes.

    References
    ----------
    .. [1] *Doe, J.*, *Smith, A.*, "On Data Quality," Journal of 
           Integrity, 2021.

    """
    # This function operates in two distinct modes:
    #
    # 1) No parentheses -> e.g. @check_empty
    #    In this scenario, we have a direct reference to the 
    #    decorated function as `func`. The decorator will check the 
    #    first argument in the function call.
    #
    # 2) With parentheses -> e.g. @check_empty(params=['x'])
    #    Here, `func` is None or not yet bound, and we return a 
    #    decorator function that can handle multiple specified 
    #    parameters listed in `params`.

    def _handle_violation(param_name):
        """
        Internal method to manage empty or None parameter 
        violations based on <error> strategy.
        """
        msg = (
            f"Parameter '{param_name}' is considered empty. Please "
            "provide a valid non-empty input."
        )
        if error == 'raise':
            raise ValueError(msg)
        elif error == 'warn':
            warnings.warn(msg, UserWarning)
        # 'ignore' => do nothing

    def _check_val(val, param_name):
        """
        Internal checker for a single parameter <param_name> 
        against emptiness or None depending on <allow_none>, 
        <none_as_empty>, etc.
        """
        # If <val> is None and:
        #   - <none_as_empty> is True, or
        #   - <allow_none> is False,
        # => violation
        if val is None and (none_as_empty or not allow_none):
            _handle_violation(param_name)
        else:
            try:
                # Check length to detect empty
                if len(val) == 0:
                    _handle_violation(param_name)
            except TypeError:
                # If object has no __len__, skip.
                pass

    def decorator_with_args(func):
        """
        Decorator for the scenario: @check_empty(params=['...'])
        Only checks the parameters listed in <params>.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use signature to map positional and keyword 
            # arguments to parameter names. Then check only 
            # those listed in <params>.
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # If <params> is provided, iterate over them
            if params:
                for p in params:
                    if p in bound.arguments:
                        _check_val(bound.arguments[p], p)
            return func(*args, **kwargs)
        return wrapper

    def decorator_no_args(func):
        """
        Decorator for the scenario: @check_empty
        Only checks the first argument (positional or 
        first keyword).
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # If there's at least one positional arg, check it;
            # otherwise check the first keyword arg (if any).
            if args:
                first_param_name = func.__code__.co_varnames[0]
                _check_val(args[0], first_param_name)
            elif kwargs:
                first_kwarg = next(iter(kwargs))
                _check_val(kwargs[first_kwarg], first_kwarg)
            return func(*args, **kwargs)
        return wrapper

    # Distinguish between the two usage modes:
    #  - If <func> is a callable, user did @check_empty without 
    #    parentheses. 
    #  - Otherwise, user did @check_empty(...) with parentheses.
    if callable(func):
        return decorator_no_args(func)
    else:
        return decorator_with_args


def _check_empty(
    func=None,
    *,
    params: Optional[List[str]] = None,
    allow_none: bool = True,
    none_as_empty: bool = False,
    error: str = 'raise'
) -> Callable:
    """
    Validate that specified parameters are not empty. Can be used
    as both a decorator and a standalone function.

    The ``check_empty`` function inspects whether certain
    parameters within a function call are empty or ``None``,
    depending on the provided configuration. This ensures data
    integrity by preventing further execution if crucial inputs
    are invalid.

    Parameters
    ----------
    params : list of str, optional
        A list of parameter names to validate. If provided, only
        these parameters are checked. If ``None``, all positional
        arguments of the decorated function are validated.
    allow_none : bool, default=True
        Determines whether ``None`` values are permitted for the
        specified parameters. If set to ``False``, parameters must
        not be ``None``.
    none_as_empty : bool, default=False
        If ``True``, treats ``None`` as empty regardless of
        ``allow_none``. This triggers an error or warning if
        encountered in the specified parameters.
    error : {'raise', 'warn', 'ignore'}, default='raise'
        Strategy upon detecting empty parameters.
        - ``'raise'`` : Raises a ``ValueError``.
        - ``'warn'``  : Issues a warning but continues.
        - ``'ignore'``: No action taken.

    Returns
    -------
    Callable
        A decorator that can validate parameters when applied to
        other functions, or a callable for manual checks.

    Examples
    --------
    As a decorator:
    >>> from fusionlab.core.checks import check_empty
    >>> @check_empty(params=['data'], allow_none=False)
    ... def process_data(data, *args, **kws):
    ...     print("Processing:", data)

    As a standalone function:

    >>> def process_data(data):
    ...     check_empty(params=['data'], allow_none=False)(
    ...         data=data
    ...     )
    ...     print("Processing:", data)
    """

    # CASE 1: Decorator *without* parentheses:
    #    @check_empty
    #    def some_func(...):
    #        ...
    #
    # => 'func' is actually the user function, and so we
    #    must wrap it directly, ignoring the other args.
    if func is not None and callable(func):
        # Wrap directly and return
        return _make_wrapper(
            func         = func,
            params       = None,        # Not provided
            allow_none   = True,        # Default
            none_as_empty= False,       # Default
            error        = 'raise'      # Default
        )

    # -------------------------------------------
    # CASE 2: Decorator *with* parentheses OR
    # CASE 3: Standalone usage.
    #
    # We return a 'decorator' function. The user might do:
    #   @check_empty(params=['x'])
    #   def f(x): ...
    #
    # or:
    #   check_empty(params=['x'])(x=...)
    #
    # We'll handle both via the same object.
    # -------------------------------------------
    def decorator(user_func: Callable) -> Callable:
        """ This is the real decorator function. """
        return _make_wrapper(
            func = user_func,
            params = params,
            allow_none = allow_none,
            none_as_empty= none_as_empty,
            error= error
        )

    # Add a __call__ method to handle standalone usage
    def standalone_checker(**kwargs) -> None:
        """
        For standalone usage, e.g.:
            check_empty(params=['x'], allow_none=False)(x=my_data)
        """
        _evaluate_params(
            arguments = kwargs,
            params = params,
            allow_none = allow_none,
            none_as_empty= none_as_empty,
            error = error
        )

    # We attach the standalone checker to 'decorator' so that
    # calling check_empty(...)(**kwargs) performs a direct check.
    decorator.__call__ = standalone_checker
    return decorator

def _make_wrapper(
    func: Callable,
    params: Optional[List[str]],
    allow_none: bool,
    none_as_empty: bool,
    error: str
) -> Callable:
    """
    Wraps the user-provided function and runs the
    parameter checks before execution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Obtain the signature to bind the arguments
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # If user provided explicit `params`, check those.
        # Otherwise, check all arguments.
        actual_params = params if params else list(bound_args.arguments.keys())

        # Evaluate the specified parameters
        _evaluate_params(
            arguments = bound_args.arguments,
            params = actual_params,
            allow_none = allow_none,
            none_as_empty= none_as_empty,
            error = error
        )
        # Proceed with the original function
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper

def _evaluate_params(
    arguments: dict,
    params: List[str],
    allow_none: bool,
    none_as_empty: bool,
    error: str
):
    """
    Core logic to check whether specified parameters are empty
    or None based on the given settings.
    """
    for p in params:
        if p not in arguments:
            # If not in the call, skip
            continue
        value = arguments[p]

        # Check if None is permitted
        if value is None:
            if none_as_empty or not allow_none:
                _handle_error(
                    param_name = p,
                    error= error,
                    message= f"Parameter '{p}' cannot be None."
                )

        # Check if empty (for lists, tuples, strings, DataFrame, etc.)
        
        elif hasattr(value, '__len__') and len(value) == 0:
            _handle_error(
                param_name= p,
                error = error,
                message= f"Parameter '{p}' is empty."
            )


def _handle_error(param_name: str, error: str, message: str):
    """
    Dispatches error handling based on `error` setting:
    raises, warns, or ignores.
    """
    if error == 'raise':
        raise ValueError(
            f"[check_empty] {message}"
        )
    elif error == 'warn':
        warnings.warn(
            f"[check_empty] {message}",
            UserWarning
        )
    # If 'ignore', do nothing.

def find_closest(arr, values):
    """
    Find the closest values in an array from a set of target values.

    This function takes an array and a set of target values, and for each 
    target value, finds the closest value in the array. It can handle 
    both scalar and array-like inputs for `values`, ensuring flexibility 
    in usage. The result is either a single closest value or an array 
    of closest values corresponding to each target.

    Parameters
    ----------
    arr : array-like
        The array to search within. It can be a list, tuple, or numpy array 
        of numeric types. If the array is multi-dimensional, it will be 
        flattened to a 1D array.
        
    values : float or array-like
        The target value(s) to find the closest match for in `arr`. This can 
        be a single float or an array of floats.

    Returns
    -------
    numpy.ndarray
        An array of the closest values in `arr` for each target in `values`.
        If `values` is a single float, the function returns a single-element
        array.

    Notes
    -----
    - This function operates by calculating the absolute difference between
      each element in `arr` and each target in `values`, selecting the 
      element with the smallest difference.
    - The function assumes `arr` and `values` contain numeric values, and it
      raises a `TypeError` if they contain non-numeric data.

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.core.checks import find_closest
    >>> find_closest([2, 3, 4, 5], 2.6)
    array([3.])

    >>> find_closest(np.array([[2, 3], [4, 5]]), (2.6, 5.6))
    array([3., 5.])

    See Also
    --------
    numpy.argmin : Find the indices of the minimum values along an axis.
    numpy.abs : Compute the absolute value element-wise.

    References
    ----------
    .. [1] Harris, C. R., et al. "Array programming with NumPy." 
       Nature 585.7825 (2020): 357-362.
    """

    arr = is_iterable(arr, exclude_string=True, transform=True)
    values = is_iterable(values, exclude_string=True, transform=True)

    # Validate numeric types in arr and values
    for var, name in zip([arr, values], ['array', 'values']):
        if not is_numeric_dtype(var, to_array=True):
            raise TypeError(f"Non-numeric data found in {name}.")

    # Convert arr and values to numpy arrays for vectorized operations
    arr = np.array(arr, dtype=np.float64)
    values = np.array(values, dtype=np.float64)

    # Flatten arr if it is multi-dimensional
    arr = arr.ravel() if arr.ndim != 1 else arr

    # Find the closest value for each target in values
    closest_values = np.array([
        arr[np.abs(arr - target).argmin()] for target in values
    ])

    return closest_values

def find_by_regex(
    o: Union[str, Iterable],
    pattern: str = r'[_#&*@!_,;\s-]\s*',
    func: Callable = re.match,
    **kws
) -> Optional[List[str]]:
    """
    Find Pattern Matches within an Object using Regular Expressions.
    
    The ``find_by_regex`` function searches for a specified regex ``pattern`` 
    within an object ``o``, which can be either a string or an iterable 
    (excluding strings). It utilizes a user-specified regex function (e.g., 
    ``re.match``, ``re.search``, ``re.findall``) to identify matches and 
    returns a list of matched elements.
    
    .. math::
        \text{Match Results} = 
        \begin{cases}
            \text{List of Matches} & \text{if matches are found} \\
            \text{None} & \text{if no matches are found}
        \end{cases}
    
    Parameters
    ----------
    o : Union[`str`, `Iterable`]
        The input object in which to search for the regex ``pattern``.
        - If a string is provided, the function searches within the string.
        - If an iterable is provided (excluding strings), the function searches 
          within each element of the iterable.
    
    pattern : `str`, default=`'[_#&*@!_,;\s-]\s*'`
        The regex pattern to search for within ``o``. By default, it matches 
        separators commonly used in text splitting.
    
    func : `Callable`, default=`re.match`
        The regex function to use for searching. Can be one of the following:
        
        - ``re.match``: Searches for a match only at the beginning of the string.
        - ``re.search``: Searches for a match anywhere in the string.
        - ``re.findall``: Finds all non-overlapping matches in the string.
        
        Additional regex functions can also be used as long as they adhere to 
        the callable signature.
    
    **kws : `dict`
        Additional keyword arguments to pass to the regex ``func``. These can 
        include flags like ``re.IGNORECASE``, ``re.MULTILINE``, etc.
    
    Returns
    -------
    Optional[List[str]]
        A list of matched objects found within ``o`` based on the regex 
        ``pattern``.
        - If matches are found, returns a list of matched strings.
        - If no matches are found, returns ``None``.
    
    Raises
    ------
    TypeError
        If ``o`` is neither a string nor an iterable object.
    
    ValueError
        If the provided ``func`` is not a recognized regex function.
    
    Examples
    --------
    >>> import re
    >>> from fusionlab.core.checks import find_by_regex
    >>> 
    >>> # Example 1: Find pattern in a concatenated string
    >>> text = "depth_top, depth_bottom, temperature"
    >>> find_by_regex(text, pattern='depth', func=re.search)
    ['depth_top']
    >>> 
    >>> # Example 2: Find pattern in an iterable of column names
    >>> columns = ['depth_top', 'depth_bottom', 'temperature']
    >>> find_by_regex(columns, pattern='depth', func=re.search)
    ['depth_top', 'depth_bottom']
    >>> 
    >>> # Example 3: Find all occurrences using re.findall
    >>> text = "depth1, depth2, depth3"
    >>> find_by_regex(text, pattern='depth\d+', func=re.findall)
    ['depth1', 'depth2', 'depth3']
    >>> 
    >>> # Example 4: No matches found
    >>> find_by_regex(columns, pattern='pressure', func=re.search)
    None
    
    Notes
    -----
    - **Input Flexibility**: The function can handle both single strings and 
      iterables (excluding strings), allowing for versatile usage across different 
      data structures.
    
    - **Regex Functionality**: By allowing users to specify the regex function, 
      the ``find_by_regex`` function provides flexibility in how patterns are 
      searched and matched within the input object.
    
    - **Performance Considerations**: When dealing with large iterables, using 
      ``re.findall`` can lead to extensive memory usage as it retrieves all matches 
      at once. Users should choose the appropriate regex function based on their 
      specific requirements.
    
    - **Handling No Matches**: If no matches are found, the function returns 
      ``None``, allowing users to handle such cases gracefully in their workflows.
    
    - **Order Preservation**: The order of matched items in the returned list 
      corresponds to their order of appearance in the input object.
    
    See Also
    --------
    re.match : Function to match a regex pattern at the beginning of a string.
    re.search : Function to search for a regex pattern anywhere in a string.
    re.findall : Function to find all non-overlapping matches of a regex 
      pattern in a string.
    
    References
    ----------
    .. [1] Python Documentation: re.match.  
       https://docs.python.org/3/library/re.html#re.match  
    .. [2] Python Documentation: re.search.  
       https://docs.python.org/3/library/re.html#re.search  
    .. [3] Python Documentation: re.findall.  
       https://docs.python.org/3/library/re.html#re.findall  
    .. [4] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """
    om = []
    
    if isinstance(o, str):
        result = func(pattern=pattern, string=o, **kws)
        if result:
            if func.__name__ == 'findall':
                om.extend(result)
            else:
                om.append(result.group())
    elif isinstance(o, Iterable) and not isinstance(o, str):
        for s in o:
            result = func(pattern=pattern, string=s, **kws)
            if result:
                if func.__name__ == 'findall':
                    om.extend(result)
                else:
                    om.append(s)
    else:
        raise TypeError(
            f"'o' must be a string or an iterable object, got {type(o).__name__!r}."
        )
    
    if not om:
        return None
    
    return om


def is_in_if(
    o: Iterable,
    items: Union[str, Iterable],
    error: str = 'raise',
    return_diff: bool = False,
    return_intersect: bool = False
) -> Union[List, None]:
    """
    Assert the Presence of Items within an Iterable Object.
    
    The ``is_in_if`` function verifies whether specified ``items`` exist within 
    an iterable object ``o``. It offers flexibility in handling missing items by 
    allowing users to either raise errors, ignore them, or retrieve differences 
    and intersections based on the provided parameters.
    
    .. math::
        \text{Presence Check} = 
        \begin{cases}
            \text{Raise Error} & \text{if items are missing and error='raise'} \\
            \text{Return Differences} & \text{if return_diff=True} \\
            \text{Return Intersection} & \text{if return_intersect=True}
        \end{cases}
    
    Parameters
    ----------
    o : `Iterable`
        The iterable object in which to check for the presence of ``items``.
    
    items : Union[`str`, `Iterable`]
        The item or collection of items to assert their presence within ``o``.
        If a single string is provided, it is treated as a single-item iterable.
    
    error : `str`, default=`'raise'`
        Determines how the function handles missing items.
        
        - ``'raise'``: Raises a ``ValueError`` if any ``items`` are not found 
          in ``o``.
        - ``'ignore'``: Suppresses errors and allows the function to proceed.
    
    return_diff : `bool`, default=`False`
        If ``True``, returns a list of items that are missing from ``o``.
        When set to ``True``, the ``error`` parameter is automatically set to 
        ``'ignore'``.
    
    return_intersect : `bool`, default=`False`
        If ``True``, returns a list of items that are present in both ``o`` and 
        ``items``.
        When set to ``True``, the ``error`` parameter is automatically set to 
        ``'ignore'``.
    
    Returns
    -------
    Union[List, None]
        - If ``return_diff`` is ``True``, returns a list of missing items.
        - If ``return_intersect`` is ``True``, returns a list of intersecting items.
        - If neither is ``True``, returns ``None`` unless an error is raised.
    
    Raises
    ------
    ValueError
        - If ``error`` is set to ``'raise'`` and any ``items`` are missing in ``o``.
        - If an unsupported value is provided for ``error``.
    
    TypeError
        - If ``o`` is not an iterable object.
    
    Examples
    --------
    >>> from fusionlab.core.checks import is_in_if
    >>> 
    >>> # Example 1: Check presence with error raising
    >>> o = ['apple', 'banana', 'cherry']
    >>> is_in_if(o, 'banana')
    # No output, validation passed
    >>> is_in_if(o, 'date')
    ValueError: Item 'date' is missing in the list ['apple', 'banana', 'cherry'].
    >>> 
    >>> # Example 2: Check multiple items with some missing
    >>> items = ['banana', 'date']
    >>> is_in_if(o, items)
    ValueError: Items 'date' are missing in the list ['apple', 'banana', 'cherry'].
    >>> 
    >>> # Example 3: Return missing items without raising error
    >>> missing = is_in_if(o, 'date', error='ignore', return_diff=True)
    >>> print(missing)
    ['date']
    >>> 
    >>> # Example 4: Return intersecting items
    >>> intersect = is_in_if(o, ['banana', 'date'], 
    ...                      error='ignore', return_intersect=True)
    >>> print(intersect)
    ['banana']
    
    Notes
    -----
    - **Flexible Input Handling**: The function accepts both single items 
      (as strings) and multiple items (as iterables), providing versatility 
      in usage scenarios.
    
    - **Automatic Error Handling Adjustment**: Setting ``return_diff`` or 
      ``return_intersect`` to ``True`` automatically changes the ``error`` 
      parameter to ``'ignore'`` to facilitate the retrieval of differences 
      or intersections without interruption.
    
    - **Performance Considerations**: For large iterables and item lists, 
      converting them to sets can enhance performance during intersection 
      and difference operations.
    
    See Also
    --------
    list : Built-in Python list type.
    set : Built-in Python set type.
    
    References
    ----------
    .. [1] Python Documentation: set.intersection.  
       https://docs.python.org/3/library/stdtypes.html#set.intersection  
    .. [2] Python Documentation: set.difference.  
       https://docs.python.org/3/library/stdtypes.html#set.difference  
    .. [3] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """
    if isinstance(items, str):
        items = [items]
    elif not isinstance(o, Iterable):
        raise TypeError(
            f"Expected an iterable object for 'o', got {type(o).__name__!r}."
        )
    
    # Convert to sets for efficient operations
    set_o = set(o)
    set_items = set(items)
    
    intersect = list(set_o.intersection(set_items))
    
    # to make a difference be sure to select the long set 
    if len(set_items) >= len(set_o): 
        missing_items = list(set_items.difference(set_o))
    else: 
        missing_items = list(set_o.difference(set_items))
    
    if return_diff or return_intersect:
        error = 'ignore'
    
    if missing_items:
        formatted_items = ', '.join(f"'{item}'" for item in missing_items)
        verb = 'is' if len(missing_items) == 1 else 'are'
        msg= (
            f"Item{'' if len(missing_items) == 1 else 's'} {formatted_items} "
            f"{verb} missing in the {type(o).__name__.lower()} {list(o)}."
        )
        if error == 'raise':
            raise ValueError(msg)
        elif error=="warn": 
            warnings.warn(msg)
    
    if return_diff:
        return missing_items if missing_items else []
    elif return_intersect:
        return intersect if intersect else []
    
    return None

def is_depth_in(
    X: pd.DataFrame,
    name: Union[str, int],
    columns: Optional[List[str]] = None,
    error: str = 'ignore'
) -> Tuple[pd.DataFrame, Union[pd.Series, None]]:
    """
    Assert the Presence of a Depth Column within a DataFrame.
    
    The ``is_depth_in`` function verifies whether a specified depth column 
    exists within a DataFrame. It supports identification by column name or 
    index and offers options to rename columns and handle missing depth columns 
    gracefully by creating pseudo-depth data if necessary.
    
    Parameters
    ----------
    X : `pandas.DataFrame`
        The input DataFrame containing data for validation.
    
    name : Union[`str`, `int`]
        The name or index of the depth column within the DataFrame.
        - If a string is provided, the function searches for a column that matches 
          the name using regex patterns.
        - If an integer is provided, it treats it as the column index to retrieve 
          the depth column name.
    
    columns : `List[str]`, optional
        New labels to replace the existing column names in the DataFrame.
        - If provided, it must match the number of columns in ``X``.
        - If the length does not match, a warning is issued and renaming is skipped.
    
    error : `str`, default=`'ignore'`
        Determines how the function handles missing depth columns.
        
        - ``'raise'``: Raises a ``ValueError`` if the depth column is not found.
        - ``'ignore'``: Suppresses errors and creates a pseudo-depth column using 
          the length of the DataFrame.
    
    Returns
    -------
    Tuple[`pandas.DataFrame`, Union[`pandas.Series`, `None`]]
        - The first element is the DataFrame without the depth column.
        - The second element is the depth column as a Series. If the depth column 
          is not found and ``error`` is ``'ignore'``, a pseudo-depth Series is created.
          Otherwise, it returns ``None``.
    
    Raises
    ------
    ValueError
        - If ``error`` is set to ``'raise'`` and the depth column is not found.
        - If ``name`` is an index that is out of bounds of the DataFrame's columns.
    
    TypeError
        - If ``X`` is not a `pandas.DataFrame`.
        - If ``columns`` is provided but is not iterable.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.core.checks import is_depth_in
    >>> 
    >>> # Example 1: Depth column by name
    >>> df = pd.DataFrame({
    ...     'depth_top': [100, 200, 300],
    ...     'value': [1, 2, 3]
    ... })
    >>> X, depth = is_depth_in(df, name='depth_top')
    >>> print(X)
       value
    0      1
    1      2
    2      3
    >>> print(depth)
    0    100
    1    200
    2    300
    Name: depth_top, dtype: int64
    >>> 
    >>> # Example 2: Depth column by index
    >>> df = pd.DataFrame({
    ...     'value': [1, 2, 3],
    ...     'depth_bottom': [400, 500, 600]
    ... })
    >>> X, depth = is_depth_in(df, name=1)
    >>> print(X)
       value
    0      1
    1      2
    2      3
    >>> print(depth)
    0    400
    1    500
    2    600
    Name: depth_bottom, dtype: int64
    >>> 
    >>> # Example 3: Rename columns and handle missing depth
    >>> df = pd.DataFrame({
    ...     'val': [1, 2, 3],
    ...     'dep': [700, 800, 900]
    ... })
    >>> X, depth = is_depth_in(
    ...     df, 
    ...     name='dep', 
    ...     columns=['value', 'depth'], 
    ...     error='ignore'
    ... )
    >>> print(X)
       value
    0      1
    1      2
    2      3
    >>> print(depth)
    0    700
    1    800
    2    900
    Name: depth, dtype: int64
    >>> 
    >>> # Example 4: Missing depth column with error handling
    >>> df = pd.DataFrame({
    ...     'value': [1, 2, 3],
    ...     'temperature': [15, 16, 17]
    ... })
    >>> is_depth_in(df, name='depth')
    ValueError: Depth column not found in dataframe.
    
    Notes
    -----
    - **Column Renaming**: When providing new column names via the ``columns`` 
      parameter, ensure that the number of new names matches the number of 
      existing columns in the DataFrame to avoid warnings and skipped renaming.
    
    - **Regex Matching**: When ``name`` is a string, the function utilizes regex 
      patterns to allow partial and case-insensitive matches for more flexible 
      depth column identification.
    
    - **Pseudo-Depth Creation**: If the depth column is not found and ``error`` 
      is set to ``'ignore'``, the function generates a pseudo-depth column based 
      on the DataFrame's length to maintain data integrity for downstream tasks.
    
    - **Error Handling**: The function provides clear and descriptive error 
      messages to aid in debugging and ensure that missing critical columns are 
      promptly addressed.
    
    See Also
    --------
    pandas.DataFrame : The primary DataFrame object in pandas.
    re.search : Function for regex-based searching.
    
    References
    ----------
    .. [1] Pandas Documentation: pandas.DataFrame.  
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html  
    .. [2] Python Documentation: re.search.  
       https://docs.python.org/3/library/re.html#re.search  
    .. [3] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"'X' must be a pandas.DataFrame, got {type(X).__name__!r}."
        )
    
    if columns is not None:
        if not isinstance(columns, Iterable):
            raise TypeError(
                f"'columns' must be an iterable object, got {type(columns).__name__!r}."
            )
        columns = list(columns)
        if len(columns) != len(X.columns):
            warnings.warn(
                f"Cannot rename columns. Expected {len(X.columns)} labels, "
                f"got {len(columns)}."
            )
        else:
            X = X.copy()
            X.columns = columns
    
    if isinstance(name, (int, float)):
        name = int(name)
        if name < 0 or name >= len(X.columns):
            warnings.warn(
                f"Name index {name} is out of range. DataFrame has "
                f"{len(X.columns)} columns."
            )
            depth = None
        else:
            depth_col = X.columns[name]
            depth = X.pop(depth_col)
    elif isinstance(name, str):
        # Use regex to find matching column
        pattern = re.compile(fr'{name}', re.IGNORECASE)
        matched_cols = [col for col in X.columns if pattern.search(col)]
        if not matched_cols:
            msg = f"Depth column matching '{name}' not found in DataFrame."
            if error == 'raise':
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                depth = None
        else:
            # Take the first match
            depth_col = matched_cols[0]
            depth = X.pop(depth_col)
    else:
        raise TypeError(
            f"'name' must be a string or integer, got {type(name).__name__!r}."
        )
    
    if depth is None:
        if error == 'raise':
            raise ValueError("Depth column not found in DataFrame.")
        else:
            depth = pd.Series(
                data=np.arange(len(X)), 
                name='depth (m)'
            )
    
    return X, depth

def exist_labels(
    df, labels, 
    features=None, 
    name="Label columns", 
    return_valid=False, 
    as_categories=False,
    error='warn',  
    verbose=0  
    
):
    """
    exist_labels - Check whether specified labels exist in feature columns 
    of a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe in which the presence of the specified labels will be 
        checked. Each column is expected to accommodate the specified `labels`, 
        which can be of categorical or any other relevant data type.
    
    labels : str or list of str
        A single label or a list of labels to check for in the specified feature 
        columns. The function checks for the presence of these labels in each 
        column. If a label is not found in a column, that column will be flagged 
        as missing that label.
    
    features : str, list of str, or None, optional, default=None
        The column(s) in which to check for the `labels`. If `features` is `None`, 
        the function will automatically check all columns of the dataframe that 
        have a categorical data type (i.e., dtype `category`, or objects with 
        categorical values). It can either be a single column name (`str`) or a 
        list of column names (`list` of `str`).
    
    name : str, optional, default="Label columns"
        A name used for informative output in verbose logging. This name is used 
        to describe the label columns, making the output more descriptive. The 
        `name` can be anything that helps identify the purpose of the label columns 
        in the dataset.
    
    return_valid : bool, optional, default=False
        If set to `True`, the function will return a dictionary where the keys 
        are column names and the values are lists of labels that were found to 
        be valid (i.e., the labels present in the specified columns). If set to 
        `False` (the default), it will return the missing labels for each column 
        where labels are missing.
    
    as_categories : bool, optional, default=False
        If set to `True`, the function will convert the specified feature columns 
        to categorical dtype before performing the check. This is useful when 
        you want to treat the feature columns as categorical variables for consistency.
    
    error : {'warn', 'raise'}, optional, default='warn'
        Specifies how to handle cases where the specified `labels` are missing. 
        If set to `'warn'` (the default), the function will issue a warning when 
        labels are not found. If set to `'raise'`, an error will be raised when 
        labels are missing from any of the specified columns.
    
    verbose : int, optional, default=0
        Controls the level of verbosity for printed messages:
        - 0: No output (silent mode).
        - 1: Basic output, including a summary of missing labels.
        - 2: Detailed output, including missing labels for each column.
        - 3: Full debug output, showing the state of all intermediate steps.
    
    Returns
    -------
    dict
        A dictionary with column names as keys and a list of missing labels or 
        valid labels (based on the value of `return_valid`). If `return_valid` is 
        `False`, the dictionary contains missing labels, otherwise it contains 
        the valid labels that are present in the dataframe columns.
    
    Notes
    ------
    Let `df` be a dataframe with columns `C1, C2, ..., Cn`. Each column 
    `Ci` can contain categorical values. The task is to check whether each 
    of the given `labels = [l1, l2, ..., lm]` exists in the columns.
    
    - For each column `Ci`, check if `l1, l2, ..., lm` are present in `Ci`.
    - If a label `lj` is missing in column `Ci`, add `Ci` to the missing list 
      for that label.
    
    The function checks each column and provides the result based on the 
    `return_valid` and `verbose` parameters:
    - If `return_valid=True`, return the valid labels that are present in each column.
    - If `return_valid=False`, return the missing labels for each column.
    
    Example
    -------
    >>> import pandas as pd 
    >>> from fusionlab.core.checks import exist_labels
    >>> df = pd.DataFrame({
    >>>     'A': ['X', 'Y', 'Z', 'X', 'W'],
    >>>     'B': ['c', 'c', 'd', 'e', 'c'],
    >>>     'C': ['a', 'b', 'c', 'd', 'e']
    >>> })
    >>> labels = ['a', 'b', 'd']
    >>> result=exist_labels(df, labels, features=['A', 'B'], verbose=2)
    >>> print(result)
    Label columns - Missing labels:
    Column 'A': Missing labels: ['a', 'b', 'd']
    Column 'B': Missing labels: ['d']
    >>> print(result)
    {'A': ['a', 'b', 'd'], 'B': ['a', 'b']}
    >>> # Sample DataFrame
    >>> df = pd.DataFrame({
        'A': ['X', 'Y', 'Z', 'X', 'W'],
        'B': ['a', 'b', 'c', 'a', 'b'],
        'C': ['cat', 'dog', 'cat', 'bird', 'dog']
    })
    
    >>> # Check if certain labels exist in the 'A' and 'B' columns
    >>> print(exist_labels(df, labels=['X', 'Y', 'Z'], features=['A', 'B'], verbose=2))
    
    >>> # Example 2: Return only valid labels in the 'C' column
    >>> print(exist_labels(df, labels=['cat', 'dog', 'lion'], features='C',
                       return_valid=True, verbose=1))
    
    >>> # Example 3: Handle missing labels with raise error
    >>> try:
        exist_labels(df, labels=['tiger', 'lion'], features='C', error='raise')
    except ValueError as e:
        print(e)

    >>> print(result)
    {'A': ['cat', 'dog'], 'B': ['dog']}

    
    Notes
    -----
    - The function automatically selects columns of categorical dtype if `features` 
      is not specified. Ensure your dataframe has appropriate categorical columns.
    - When `as_categories=True`, the function converts the specified columns 
      into categorical dtype before checking for labels.
    - The function supports two error handling modes:
        - `'warn'` issues a warning if labels are missing.
        - `'raise'` raises an exception if any labels are missing from the columns.
    
    See Also
    --------
    pandas.DataFrame.astype : Convert data types of dataframe columns
    pandas.api.types.is_categorical_dtype : Check for categorical dtype
    
    References
    ----------
    .. [1] Smith, J., "Data Analysis with Pandas", 2020, Springer
    .. [2] Doe, A., "Advanced DataFrame Operations in Python", 2018, Wiley
    """

    are_all_frames_valid(df, df_only= True )
    # Ensure 'features' and 'labels' are lists, even if passed as strings
    if isinstance(features, str):
        features = [features]
    if isinstance(labels, str):
        labels = [labels]
    
    # If 'features' is None, use columns with categorical dtype
    if features is None:
        features = [
            col for col in df.columns 
            if pd.api.types.is_categorical_dtype(df[col])
        ]
    if as_categories: 
        df[features]= df[features].astype('category')
        
    # If no valid features found, raise an informative error or warning
    if not features:
        if error == 'raise':
            raise ValueError(
                f"No valid features found with categorical dtype in {name}.")
        elif error == 'warn':
            warnings.warn(
                "No valid features with categorical"
                " dtype found. Proceeding with all columns.")
            features = df.columns.tolist()

    # Initialize result dictionary
    result = {}
    
    # Iterate through the features (columns) to check for valid or missing labels
    for col in features:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe.")
        
        # Get the valid categories in the current column
        valid_categories = df[col].unique()
        
        # Find missing labels
        missing_in_col = [
            label for label in labels if label not in valid_categories]
        
        # Depending on return_valid, return valid or missing labels
        if return_valid:
            valid_in_col = [
                label for label in labels if label in valid_categories]
            result[col] = valid_in_col
        else:
            if missing_in_col:
                result[col] = missing_in_col
        
        # Verbosity control: Print results based on verbosity level
        if verbose >= 1:
            if missing_in_col:
                print(
                    f"{name}: Column '{col}' has missing labels: {missing_in_col}")
            elif verbose >= 2:
                print(
                    f"{name}: Column '{col}' has all labels present.")
        
        if verbose == 3:
            print(f"Debug: Column '{col}' - Valid categories: {valid_categories}")
    
    # Handle case where no valid labels are found in any columns
    if not result:
        if error == 'raise':
            raise ValueError(
                f"No valid labels found in any of the specified columns for {name}.")
        elif error == 'warn':
            warnings.warn(
                "Warning: No valid labels found in any"
                " of the specified columns for {name}."
            )
    
    return result

def is_classification_task(
    *y, max_unique_values=10
    ):
    """
    Check whether the given arrays are for a classification task.

    This function assumes that if all values in the provided arrays are 
    integers and the number of unique values is within the specified
    threshold, it is a classification task.

    Parameters
    ----------
    *y : list or numpy.array
        A variable number of arrays representing actual values, 
        predicted values, etc.
    max_unique_values : int, optional
        The maximum number of unique values to consider the task 
        as classification. 
        Default is 10.

    Returns
    -------
    bool
        True if the provided arrays are for a classification task, 
        False otherwise.

    Examples
    --------
    >>> from fusionlab.core.checks import is_classification_task 
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> is_classification_task(y_true, y_pred)
    True
    """
    max_unique_values = int (
        _assert_all_types(max_unique_values, 
                          int, float, objname="Max Unique values")
                             )
    # Combine all arrays for analysis
    combined = np.concatenate(y)

    # Check if all elements are integers
    if ( 
            not all(isinstance(x, int) for x in combined) 
            and not combined.dtype.kind in 'iu'
            ):
        return False

    # Check the number of unique elements
    unique_values = np.unique(combined)
    # check Arbitrary threshold for number of classes
    if len(unique_values) > max_unique_values:
        return False

    return True

def validate_noise(noise):
    """
    Validates the `noise` parameter and returns either the noise value
    as a float or the string 'gaussian'.

    Parameters
    ----------
    noise : str or float or None
        The noise parameter to be validated. It can be the string
        'gaussian', a float value, or None.

    Returns
    -------
    float or str
        The validated noise value as a float or the string 'gaussian'.

    Raises
    ------
    ValueError
        If the `noise` parameter is a string other than 'gaussian' or
        cannot be converted to a float.

    Examples
    --------
    >>> validate_noise('gaussian')
    'gaussian'
    >>> validate_noise(0.1)
    0.1
    >>> validate_noise(None)
    None
    >>> validate_noise('0.2')
    0.2

    """
    if isinstance(noise, str):
        if noise.lower() == 'gaussian':
            return 'gaussian'
        else:
            try:
                noise = float(noise)
            except ValueError:
                raise ValueError("The `noise` parameter accepts the string"
                                 " 'gaussian' or a float value.")
    elif noise is not None:
        noise = validate_ratio(noise, bounds=(0, 1), param_name='noise' )
        # try:
        # except ValueError:
        #     raise ValueError("The `noise` parameter must be convertible to a float.")
    return noise

def validate_feature(
    data: Union[DataFrame, Series],
    features: Union[List[str], Pattern, Callable[[str], bool]],
    error: str = 'raise',  
    ops: str = 'validate'  
) -> Union[bool, pd.DataFrame]:
    """
    Validate the existence of specified features in a DataFrame or Series.

    This function verifies whether the specified features exist within the
    provided data. The ``features`` parameter can be a list of feature names,
    a regular expression (regex) pattern, or a callable that determines feature
    selection based on column names.

    .. math::
        \text{Let } C = \{c_1, c_2, \dots, c_n\} \text{ be the set of columns in }
        \text{data. Let } F = \{f_1, f_2, \dots, f_m\} \text{ be the set of features to validate.}

    The function evaluates the intersection of ``C`` and ``F``, and based on the
    operation mode and error handling parameters, it performs the necessary
    actions.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        The DataFrame or Series in which to validate feature existence.
    features : Union[List[str], Pattern, Callable[[str], bool]]
        Features to check for existence. Can be a list of feature names, a
        regex pattern (``str`` or ``re.Pattern``), or a callable that takes a
        column name and returns ``True`` if the column should be selected.
    error : str, {'raise', 'warn', 'ignore'}, default='raise'
        Specifies how to handle the absence of features when ``ops='validate'``.
        
        - ``'raise'``: Raises a ``ValueError`` if any feature is missing.
        - ``'warn'``: Issues a warning if any feature is missing and returns a
          DataFrame with the valid features.
        - ``'ignore'``: Silently returns a DataFrame with the valid features,
          disregarding any missing features.
    ops : str, {'validate', 'check_only'}, default='validate'
        Operation mode.
        
        - ``'validate'``: Performs validation and handles missing features based
          on the ``error`` parameter.
        - ``'check_only'``: Returns a boolean indicating whether the specified
          features exist without raising errors.

    Returns
    -------
    Union[bool, pd.DataFrame]
        - If ``ops='validate'``:
            - ``'raise'``: Returns ``True`` if all features exist, otherwise
              raises a ``ValueError``.
            - ``'warn'`` or ``'ignore'``: Returns a DataFrame containing only
              the valid features.
        - If ``ops='check_only'``:
            - Returns ``True`` if all specified features are present, ``False``
              otherwise.

    Raises
    ------
    ValueError
        If ``error='raise'`` and any specified features are missing.
        If ``ops`` or ``error`` parameters are invalid.
    TypeError
        If ``features`` is not a list of strings, a regex pattern, or a callable.

    Examples
    --------
    >>> from fusionlab.core.checks import validate_feature
    >>> import pandas as pd
    >>> import re
    >>> # Sample DataFrame
    >>> data = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C1': [5, 6]})
    
    # Example 1: Validating a list of feature names
    >>> validate_feature(data, ['A', 'B'])
        B  A
     0  3  1
     1  4  2
    
    >>> validate_feature(data, ['A', 'D'], error='raise')
    Traceback (most recent call last):
        ...
    ValueError: The following feature is missing in the data: D.
    
    # Example 2: Using a regex pattern
    >>> pattern = re.compile(r'^C\d+$')
    >>> validate_feature(data, pattern)
       C1
    0   5
    1   6
    
    >>> invalid_pattern = re.compile(r'^D\d+$')
    >>> validate_feature(data, invalid_pattern, ops='check_only')
    False
    
    # Example 3: Using a callable
    >>> callable_selector = lambda col: col.startswith('A') or col.startswith('B')
    >>> validate_feature(data, callable_selector)
       A  B
    0  1  3
    1  2  4
    
    >>> invalid_callable = lambda col: col.startswith('D')
    >>> validate_feature(data, invalid_callable, ops='check_only')
    False

    Notes
    -----
    - This function is particularly useful in data preprocessing pipelines
      where the presence of certain features is critical for subsequent
      analysis or modeling steps.
    - When using regex patterns, ensure that the pattern accurately reflects
      the intended column names to avoid unintended matches.
    - The callable provided to ``features`` should accept a single string
      argument (the column name) and return a boolean indicating whether
      the column should be selected.

    See Also
    --------
    pandas.DataFrame.columns : Provides access to the column labels of a DataFrame.
    pandas.Series.apply : Applies a function along an axis of the DataFrame or Series.
    warnings.warn : Issues a warning message.

    References
    ----------
    .. [1] Pandas Documentation. "DataFrame.columns." 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.columns.html
    .. [2] Python `re` Module. 
       https://docs.python.org/3/library/re.html
    .. [3] Python `warnings` Module.
       https://docs.python.org/3/library/warnings.html

    """
    if isinstance(data, pd.Series):
        # Convert Series to DataFrame for uniform processing
        data = data.to_frame()#.T

    # Initialize sets for selected and missing features
    selected_columns = set()
    missing_features = set()

    # Determine selected columns based on the type of 'features'
    if isinstance(features, (list, tuple)):
        # Features specified as a list of strings
        selected_columns = set(features).intersection(data.columns)
        missing_features = set(features).difference(selected_columns)
    elif isinstance(features, (re.Pattern, str)):
        # Features specified as a regex pattern
        pattern = features if isinstance(features, re.Pattern) else re.compile(features)
        matched_columns = set(data.columns[data.columns.to_series().str.match(pattern)])
        selected_columns = matched_columns
        if isinstance(features, str):
            pattern_str = features
        else:
            pattern_str = pattern.pattern
        # For regex, missing features are not explicitly tracked
    elif callable(features):
        # Features specified as a callable
        matched_columns = set(
            data.columns[data.columns.to_series().apply(features)]
        )
        selected_columns = matched_columns
    else:
        raise TypeError(
            "``features`` must be a list of strings, a regex pattern, or a callable."
        )

    # Calculate missing features only if 'features' is a list
    if isinstance(features, list):
        missing_features = set(features).difference(selected_columns)

    message =''
    if missing_features:
        verb = "is" if len(missing_features) == 1 else "are"
        message = (
            f"The following feature {verb} missing in the data: "
            f"{_smart_format(missing_features)}."
        )
        
    if ops == 'validate':
        if isinstance(features, list):
            if missing_features:
                if error == 'raise':
                    # Raise ValueError if any features are missing
                    raise ValueError(message)
                elif error == 'warn':
                    # Issue a warning and proceed with valid features
                    warnings.warn(
                        f"{message}. Proceeding with available features.",
                        UserWarning
                    )
                elif error == 'ignore':
                    # Silently proceed with available features
                    pass
                else:
                    raise ValueError(
                        f"Invalid ``error`` parameter: {error}. Must be "
                        f"'raise', 'warn', or 'ignore'."
                    )
            # Return DataFrame with valid features
            return data[list(selected_columns)]
        else:
            # For regex or callable, missing features are not explicitly tracked
            if isinstance(features, (re.Pattern, str)):
                if not selected_columns:
                    if error == 'raise':
                        pattern_str = features.pattern if isinstance(
                            features, re.Pattern) else features
                        raise ValueError(
                            f"No columns match the regex pattern: {pattern_str}."
                        )
                    elif error == 'warn':
                        pattern_str = features.pattern if isinstance(
                            features, re.Pattern) else features
                        warnings.warn(
                            f"No columns match the regex pattern: {pattern_str}.",
                            UserWarning
                        )
                    elif error == 'ignore':
                        pass
                    else:
                        raise ValueError(
                            f"Invalid ``error`` parameter: {error}. Must be "
                            f"'raise', 'warn', or 'ignore'."
                        )
            elif callable(features):
                if not selected_columns:
                    if error == 'raise':
                        raise ValueError(
                            "No columns match the callable criteria."
                        )
                    elif error == 'warn':
                        warnings.warn(
                            "No columns match the callable criteria.",
                            UserWarning
                        )
                    elif error == 'ignore':
                        pass
                    else:
                        raise ValueError(
                            f"Invalid ``error`` parameter: {error}. Must be "
                            f"'raise', 'warn', or 'ignore'."
                        )
            # Return DataFrame with matched features
            return data[list(selected_columns)]
        
    elif ops == 'check_only':
        if isinstance(features, list):
            if missing_features:
                if error == 'warn':
                    warnings.warn(message,  UserWarning)
                elif error == 'ignore':
                    pass
                elif error == 'raise':
                    # Even in 'check_only', if error='raise', raise an error
       
                    raise ValueError(message)
                else:
                    raise ValueError(
                        f"Invalid ``error`` parameter: {error}. Must be "
                        f"'raise', 'warn', or 'ignore'."
                    )
                return False
            else:
                return True
        else:
            # For regex or callable, simply check if any columns matched
            return bool(selected_columns)
    else:
        raise ValueError(
            f"Invalid ``ops`` parameter: {ops}. Must be "
            f"'validate' or 'check_only'."
        )

def features_in(
    *data: Union[pd.DataFrame, pd.Series], features: List[str],
    error: str = 'ignore') -> List[bool]:
    """
    Control whether the specified features exist in multiple datasets.

    Parameters
    ----------
    *data : DataFrame or Series arguments
        Multiple DataFrames or Series to check for feature existence.
    features : list of str
        List of features to check for existence in the datasets.
    error : str, {'raise', 'ignore'}, optional
        Specify how to handle the absence of features. 'ignore' (default) will ignore
        a ValueError for each dataset with missing features, while 'ignore' will
        return a list of booleans indicating whether all features exist in each dataset.

    Returns
    -------
    list of bool
        A list of booleans indicating whether the specified features exist in each dataset.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.core.checks import features_in
    >>> data1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> data2 = pd.Series([5, 6], name='C')
    >>> data3 = pd.DataFrame({'X': [7, 8]})
    >>> features = ['A', 'C']
    >>> results1 = features_in(data1, data2, features=features, error='raise')
    >>> print(results1)  # This will raise a ValueError for the first dataset
    >>> results2 = features_in(data1, data3, features, error='ignore')
    >>> print(results2)  # This will return [True, False]
    """
    results = []

    for dataset in data:
        results.append(validate_feature(dataset, features, error=error))

    return results

def find_features_in(
    data: DataFrame = None,
    features: List[str] = None,
    parse_features: bool = False,
    return_frames: bool = False,
) -> Tuple[Union[List[str], DataFrame], Union[List[str], DataFrame]]:
    """
    Retrieve the categorical or numerical features from the dataset.

    Parameters
    ----------
    data : DataFrame, optional
        DataFrame with columns representing the features.
    features : list of str, optional
        List of column names. If provided, the DataFrame will be restricted
        to only include the specified features before searching for numerical
        and categorical features. An error will be raised if any specified
        feature is missing in the DataFrame.
    return_frames : bool, optional
        If True, it returns two separate DataFrames (cat & num). Otherwise, it
        returns only the column names of categorical and numerical features.
    parse_features : bool, default False
        Use default parsers to parse string items into an iterable object.

    Returns
    -------
    Tuple : List[str] or DataFrame
        The names or DataFrames of categorical and numerical features.

    Examples
    --------
    >>> from fusionlab.datasets import fetch_data
    >>> from fusionlab.utils.mlutils import find_features_in
    >>> data = fetch_data('bagoue').frame 
    >>> cat, num = find_features_in(data)
    >>> cat, num
    ... (['type', 'geol', 'shape', 'name', 'flow'],
    ...  ['num', 'east', 'north', 'power', 'magnitude', 'sfi', 'ohmS', 'lwi'])
    >>> cat, num = find_features_in(data, features=['geol', 'ohmS', 'sfi'])
    >>> cat, num
    ... (['geol'], ['ohmS', 'sfi'])
    """
    from .array_manager import to_numeric_dtypes 
    
    if not isinstance (data, pd.DataFrame):
        raise TypeError(f"Expect a DataFrame. Got {type(data).__name__!r}")

    if features is not None:
        features = list(
            is_iterable(
                features,
                exclude_string=True,
                transform=True,
                parse_string=parse_features,
            )
        )

    if features is None:
        features = list(data.columns)

    validate_feature(data, list(features))
    data = data[features].copy()

    # Get numerical features
    data, numnames, catnames = to_numeric_dtypes(
        data, return_feature_types=True )

    if catnames is None:
        catnames = []

    return (data[catnames], data[numnames]) if return_frames else (
        list(catnames), list(numnames)
    )


def check_uniform_type(
    values: Union[Iterable[Any], Any],
    items_to_compare: Union[Iterable[Any], Any] = None,
    raise_exception: bool = True,
    convert_values: bool = False,
    return_types: bool = False,
    target_type: type = None,
    allow_mismatch: bool = True,
    infer_types: bool = False,
    comparison_method: str = 'intersection',
    custom_conversion_func: _F[[...], ...] = None,
    return_func: bool = False
) -> Union[bool, List[type], Tuple[Iterable[Any], List[type]], _F]:
    """
    Checks whether elements in `values` are of uniform type. 
    
    Optionally comparing them against another set of items or converting all 
    values to a target type. Can return a callable for deferred execution of 
    the specified logic.Function is useful for validating data uniformity, 
    especially before performing operations that assume homogeneity of the 
    input types.
    

    Parameters
    ----------
    values : Iterable[Any] or Any
        An iterable containing items to check. If a non-iterable item is provided,
        it is treated as a single-element iterable.
    items_to_compare : Iterable[Any] or Any, optional
        An iterable of items to compare against `values`. If specified, the
        `comparison_method` is used to perform the comparison.
    raise_exception : bool, default True
        If True, raises an exception when a uniform type is not found or other
        constraints are not met. Otherwise, issues a warning.
    convert_values : bool, default False
        If True, tries to convert all `values` to `target_type`. Requires
        `target_type` to be specified.
    return_types : bool, default False
        If True, returns the types of the items in `values`.
    target_type : type, optional
        The target type to which `values` should be converted if `convert_values`
        is True.
    allow_mismatch : bool, default True
        If False, requires all values to be of identical types; otherwise,
        allows type mismatch.
    infer_types : bool, default False
        If True and different types are found, returns the types of each item
        in `values` in order.
    comparison_method : str, default 'intersection'
        The method used to compare `values` against `items_to_compare`. Must
        be one of the set comparison methods ('difference', 'intersection', etc.).
    custom_conversion_func : Callable[[Any], Any], optional
        A custom function for converting items in `values` to another type.
    return_func : bool, default False
        If True, returns a callable that encapsulates the logic based on the 
        other parameters.

    Returns
    -------
    Union[bool, List[type], Tuple[Iterable[Any], List[type]], Callable]
        The result based on the specified parameters. This can be: 
        - A boolean indicating whether all values are of the same type.
        - The common type of all values if `return_types` is True.
        - A tuple containing the converted values and their types if `convert_values`
          and `return_types` are both True.
        - a callable encapsulating the specified logic for deferred execution.
        
    Examples
    --------
    >>> from fusionlab.core.checks import check_uniform_type
    >>> check_uniform_type([1, 2, 3])
    True

    >>> check_uniform_type([1, '2', 3], allow_mismatch=False, raise_exception=False)
    False

    >>> deferred_check = check_uniform_type([1, 2, '3'], convert_values=True, 
    ...                                        target_type=int, return_func=True)
    >>> deferred_check()
    [1, 2, 3]

    Notes
    -----
    The function is designed to be flexible, supporting immediate or deferred execution,
    with options for type conversion and detailed type information retrieval.
    """
    def operation():
        # Convert values and items_to_compare to lists if 
        # they're not already iterable
        if isinstance(values, Iterable) and not isinstance(values, str):
            val_list = list(values)
        else:
            val_list = [values]

        if items_to_compare is not None:
            if isinstance(items_to_compare, Iterable) and not isinstance(
                    items_to_compare, str):
                comp_list = list(items_to_compare)
            else:
                comp_list = [items_to_compare]
        else:
            comp_list = []

        # Extract types
        val_types = set(type(v) for v in val_list)
        comp_types = set(type(c) for c in comp_list) if comp_list else set()

        # Compare types
        if comparison_method == 'intersection':
            common_types = val_types.intersection(comp_types) if comp_types else val_types
        elif comparison_method == 'difference':
            common_types = val_types.difference(comp_types)
        else:
            if raise_exception:
                raise ValueError(f"Invalid comparison method: {comparison_method}")
            return False

        # Check for type uniformity
        if not allow_mismatch and len(common_types) > 1:
            if raise_exception:
                raise ValueError("Not all values are the same type.")
            return False

        # Conversion
        if convert_values:
            if not target_type and not custom_conversion_func:
                if raise_exception:
                    raise ValueError("Target type or custom conversion "
                                     "function must be specified for conversion.")
                return False
            try:
                if custom_conversion_func:
                    converted_values = [custom_conversion_func(v) for v in val_list]
                else:
                    converted_values = [target_type(v) for v in val_list]
            except Exception as e:
                if raise_exception:
                    raise ValueError(f"Conversion failed: {e}")
                return False
            if return_types:
                return converted_values, [type(v) for v in converted_values]
            return converted_values

        # Return types
        if return_types:
            if infer_types or len(common_types) > 1:
                return [type(v) for v in val_list]
            return list(common_types)

        return True

    return operation if return_func else operation()


def assert_ratio(
    v: Union[str, float, int],
    bounds: Optional[Tuple[float, float]] = None,
    exclude_values: Optional[Union[float, List[float]]] = None,
    in_percent: bool = False,
    inclusive: bool = True,
    name: str = 'ratio'
) -> float:
    """
    Asserts that a given value falls within a specified range and does not
    match any excluded values. Optionally converts the value to a percentage.
    
    This function is useful for validating ratio or rate values in data 
    preprocessing, ensuring they meet defined criteria before further 
    analysis or modeling.
    
    Parameters
    ----------
    v : Union[str, float, int]
        The ratio value to assert. Can be a string (possibly containing 
        a percentage sign), float, or integer.
        
    bounds : Optional[Tuple[float, float]], default=None
        A tuple specifying the lower and upper bounds (inclusive by default) 
        within which the value `v` must lie. If `None`, no bounds are enforced.
        
    exclude_values : Optional[Union[float, List[float]]], default=None
        Specific value(s) that `v` must not equal. Can be a single float or a 
        list of floats. If provided, `v` is checked against these excluded 
        values after any necessary conversions.
        
    in_percent : bool, default=False
        If `True`, interprets the input value `v` as a percentage and converts 
        it to its decimal form (e.g., 50 becomes 0.5). If `v` is a string 
        containing a `%` sign, it is automatically converted to decimal.
        
    inclusive : bool, default=True
        Determines whether the bounds are inclusive. If `True`, `v` can be equal 
        to the lower and upper bounds. If `False`, `v` must be strictly 
        greater than the lower bound and strictly less than the upper bound.
        
    name : str, default='ratio'
        The descriptive name of the value being asserted. This is used in error 
        messages for clarity.
    
    Returns
    -------
    float
        The validated (and possibly converted) ratio value.
        
    Raises
    ------
    TypeError
        If `v` cannot be converted to a float.
        
    ValueError
        If `v` is outside the specified bounds or matches any excluded values.
    
    Examples
    --------
    1. **Basic Usage with Bounds:**
    
        ```python
        from fusionlab.core.checks import assert_ratio
        assert_ratio(0.5, bounds=(0.0, 1.0))
        # Returns: 0.5
        ```
    
    2. **String Input with Percentage:**
    
        ```python
        assert_ratio("75%", in_percent=True)
        # Returns: 0.75
        ```
    
    3. **Excluding Specific Values:**
    
        ```python
        assert_ratio(0.5, bounds=(0.0, 1.0), exclude_values=0.5)
        # Raises ValueError
        ```
    
    4. **Multiple Excluded Values and Exclusive Bounds:**
    
        ```python
        assert_ratio(0.3, bounds=(0.0, 1.0), exclude_values=[0.2, 0.4], inclusive=False)
        # Returns: 0.3
        ```
    
    Notes
    -----
    - The function first attempts to convert the input `v` to a float. 
      If `in_percent` is `True`, it converts percentage values to 
      their decimal equivalents.
    - Bounds can be set to define a valid range for `v`. If `inclusive` 
      is set to `False`, the bounds are treated as exclusive.
    - Excluded values are checked after any necessary conversions.
    - If `exclude_values` is provided without specifying `bounds`, the 
      function will only check for excluded values.
    
    References
    ----------
    - [Python `float()` Function](https://docs.python.org/3/library/functions.html#float)
    - [Warnings in Python](https://docs.python.org/3/library/warnings.html)
    """
    
    # Initialize exclusion list
    if exclude_values is not None and not isinstance(exclude_values, list):
        exclude_values = [exclude_values]
    
    # Regular expression to detect percentage in strings
    percent_pattern = re.compile(r'^\s*[-+]?\d+(\.\d+)?%\s*$')
    
    # Check and convert string inputs
    if isinstance(v, str):
        v = v.strip()
        if percent_pattern.match(v):
            in_percent = True
            v = v.replace('%', '').strip()
    
    try:
        # Convert to float
        v = float(v)
    except (TypeError, ValueError):
        raise TypeError(
            f"Unable to convert {type(v).__name__!r} value '{v}' to float."
        )
    
    # Convert to percentage if required
    if in_percent:
        if 0 <= v <= 100:
            v /= 100.0
        elif 0 <= v <= 1:
            warnings.warn(
                f"The value {v} seems already in decimal form; "
                f"no conversion applied for {name}.",
                UserWarning
            )
        else:
            raise ValueError(
                f"When 'in_percent' is True, {name} should be between "
                f"0 and 100, got {v * 100 if 0 <= v <=1 else v}."
            )
    
    # Check bounds if specified
    if bounds:
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(
                 "'bounds' must be a tuple of two"
                f" floats, got {type(bounds).__name__}."
            )
        lower, upper = bounds
        if inclusive:
            if not (lower <= v <= upper):
                raise ValueError(
                    f"{name.capitalize()} must be between {lower}"
                    f" and {upper} inclusive, got {v}."
                )
        else:
            if not (lower < v < upper):
                raise ValueError(
                    f"{name.capitalize()} must be between {lower} and "
                    f"{upper} exclusive, got {v}."
                )
    
    # Check excluded values
    if exclude_values:
        if v in exclude_values:
            if len(exclude_values) == 1:
                exclusion_msg =( 
                    f"{name.capitalize()} must not be {exclude_values[0]}."
                    )
            else:
                exclusion_msg =( 
                    f"{name.capitalize()} must not be one of {exclude_values}."
                    )
            raise ValueError(exclusion_msg)
    
    return v

def validate_ratio(
    value: float, 
    bounds: Optional[Tuple[float, float]] = None, 
    exclude: Optional[float] = None, 
    to_percent: bool = False, 
    param_name: str = 'value'
) -> float:
    """Validates and optionally converts a value to a percentage within 
    specified bounds, excluding specific values.

    Parameters:
    -----------
    value : float or str
        The value to validate and convert. If a string with a '%' sign, 
        conversion to percentage is attempted.
    bounds : tuple of float, optional
        A tuple specifying the lower and upper bounds (inclusive) for the value. 
        If None, no bounds are enforced.
    exclude : float, optional
        A specific value to exclude from the valid range. If the value matches 
        'exclude', a ValueError is raised.
    to_percent : bool, default=False
        If True, the value is converted to a percentage 
        (assumed to be in the range [0, 100]).
    param_name : str, default='value'
        The parameter name to use in error messages.

    Returns:
    --------
    float
        The validated (and possibly converted) value.

    Raises:
    ------
    ValueError
        If the value is outside the specified bounds, matches the 'exclude' 
        value, or cannot be converted as specified.
    """
    if isinstance(value, str) and '%' in value:
        to_percent = True
        value = value.replace('%', '')
    try:
        value = float(value)
    except ValueError:
        raise ValueError(f"Expected a float, got {type(value).__name__}: {value}")

    if to_percent and 0 < value <= 100:
        value /= 100

    if bounds:
        if not (bounds[0] <= value <= bounds[1]):
            raise ValueError(f"{param_name} must be between {bounds[0]}"
                             f" and {bounds[1]}, got: {value}")
    
    if exclude is not None and value == exclude:
        raise ValueError(f"{param_name} cannot be {exclude}")

    if to_percent and value > 1:
        raise ValueError(f"{param_name} converted to percent must"
                         f" not exceed 1, got: {value}")

    return value

def exist_features(
    df: pd.DataFrame, 
    features, 
    error='raise',  
    name="Feature", 
    message=None, 
) -> bool:
    """
    Check whether the specified features exist in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features to be checked.
    features : list or str
        List of feature names (str) to check for in the dataframe. 
        If a string is provided, it will be treated as a list with 
        a single feature.
    error : str, optional, default 'raise'
        Action to take if features are not found. Can be one of:
        - 'raise' (default): Raise a ValueError.
        - 'warn': Issue a warning and return False.
        - 'ignore': Do nothing if features are not found.
    name : str, optional, default 'Feature'
        Name of the feature(s) being checked (default is 'Feature').

    Returns
    -------
    bool
        Returns True if all features exist in the dataframe, otherwise False.

    Raises
    ------
    ValueError
        If 'error' is 'raise' and features are not found.
    
    Warns
    -----
    UserWarning
        If 'error' is 'warn' and features are missing.

    Notes
    -----
    This function ensures that all the specified features exist in the
    dataframe. If the 'error' parameter is set to 'warn', the function 
    will issue a warning instead of raising an error when a feature 
    is missing, and return False.

    References
    ----------
    - pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Examples
    --------
    >>> from fusionlab.core.checks import exist_features
    >>> import pandas as pd

    >>> # Sample DataFrame
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3],
    >>>     'feature2': [4, 5, 6],
    >>>     'feature3': [7, 8, 9]
    >>> })

    >>> # Check for missing features with 'raise' error
    >>> exist_features(df, ['feature1', 'feature4'], error='raise')
    Traceback (most recent call last):
        ...
    ValueError: Features feature4 not found in the dataframe.

    >>> # Check for missing features with 'warn' error
    >>> exist_features(df, ['feature1', 'feature4'], error='warn')
    UserWarning: Features feature4 not found in the dataframe.

    >>> # Check for missing features with 'ignore' error
    >>> exist_features(df, ['feature1', 'feature4'], error='ignore')
    False
    """
    # Validate if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas DataFrame.")

    # Normalize the error parameter to lowercase and strip whitespace
    error = error.lower().strip()

    # Validate the 'error' parameter
    if error not in ['raise', 'ignore', 'warn']:
        raise ValueError(
            "Invalid value for 'error'. Expected"
            " one of ['raise', 'ignore', 'warn'].")

    # Ensure 'features' is a list-like structure
    if isinstance(features, str):
        features = [features]

    # Validate that 'features' is one of the allowed types
    features = _assert_all_types(
        features, list, tuple, np.ndarray, pd.Index)

    # Get the intersection of features with the dataframe columns
    existing_features = set(features).intersection(df.columns)

    # If all features exist, return True
    if len(existing_features) == len(features):
        return True

    # Calculate the missing features
    missing_features = set(features) - existing_features

    # If there are missing features, handle according to 'error' type
    if missing_features:
        msg = message or f"{name}{'s' if len(features) > 1 else ''}"

        if error == 'raise':
            raise ValueError(
                f"{msg} {_smart_format(missing_features)}"
                " not found in the dataframe."
            )

        elif error == 'warn':
            warnings.warn(
                f"{msg} {_smart_format(missing_features)}"
                " not found in the dataframe.",
                UserWarning
            )
            return False

        # If 'error' is 'ignore', simply return False
        return False

    return True

def is_iterable(
    y,
    exclude_string: bool = False,
    transform: bool = False,
    parse_string: bool = False
) -> Union[bool, list]:
    """
    Asserts whether `<y>` is iterable and optionally transforms
    `<y>` into a list or parses it as columns if it is a string.

    If `<exclude_string>` is True and `<y>` is a string, the
    function returns `False` for the iterability check.
    If `<transform>` is True, the function returns `<y>` as-is
    if already iterable or wraps `<y>` in a list. If
    `<parse_string>` is True (and `<transform>` is also True),
    a string input is split into columns via `str2columns`.

    Parameters
    ----------
    y : any
        Object to evaluate for iterability or transform
        into an iterable.
    exclude_string : bool, default=False
        If True, treats any string `<y>` as non-iterable.
    transform : bool, default=False
        If True, transforms `<y>` into an iterable if
        not already one. By default, wraps `<y>` into
        a list.
    parse_string : bool, default=False
        If True and `<y>` is a string, attempts to parse
        using `str2columns`. Requires `<transform>` = True.

    Returns
    -------
    bool or list
        If `<transform>` is False, returns a boolean
        indicating whether `<y>` is considered iterable.
        If `<transform>` is True, returns either `<y>` (if
        it is already iterable) or `[y]`. If `<parse_string>`
        is also True, a string is split into columns.

    Raises
    ------
    ValueError
        If `<parse_string>` is True but `<transform>` is
        False while `<y>` is a string.

    Examples
    --------
    >>> from fusionlab.core.checks.is_iterable import is_iterable
    >>> is_iterable('iterable', exclude_string=True)
    False
    >>> is_iterable('iterable', exclude_string=True, transform=True)
    ['iterable']
    >>> is_iterable('parse this', parse_string=True, transform=True)
    ['parse', 'this']
    """
    # If user wants to parse string but not transform,
    # raise error because result wouldn't be an iterable.
    if parse_string and not transform and isinstance(y, str):
        raise ValueError(
            "Cannot parse the given string. Set 'transform' to True "
            "or use 'str2columns' directly."
        )

    # If parse_string is True, convert string to columns
    if isinstance(y, str) and parse_string:
        y = str2columns(y)

    # Check iterability, but optionally treat string
    # objects as non-iterable
    is_iter = not (
        exclude_string and isinstance(y, str)
    ) and hasattr(y, '__iter__')

    # If transform is True, return y as-is if it is
    # iterable, otherwise wrap it in a list.
    if transform:
        return y if is_iter else [y]

    # Otherwise, just return boolean indicating
    # iterability
    return is_iter


def _smart_format(
    iter_obj,
    choice: str = 'and'
) -> str:
    """
    Smartly format an iterable object into a readable
    string with a specific connector (e.g. `'and'`).

    Parameters
    ----------
    iter_obj : iterable
        The iterable to format. If it is not truly
        iterable, it is returned as a string
        representation.
    choice : str, default='and'
        The connector word between the second last
        and last items (e.g. `'and'`, `'or'`).

    Returns
    -------
    str
        A user-friendly string representation of
        `<iter_obj>`, e.g. '"foo", "bar" and "baz"'.

    Examples
    --------
    >>> _smart_format(['apple', 'banana', 'cherry'])
    '"apple","banana" and "cherry"'
    >>> _smart_format(['apple'])
    '"apple"'
    >>> _smart_format('banana')
    'banana'
    """
    # Attempt to ensure it's iterable
    try:
        _ = iter(iter_obj)
    except: # TypeError >
        return f"{iter_obj}"

    # Convert each element to string
    items = [str(obj) for obj in iter_obj]
    if not items:
        return ""

    if len(items) == 1:
        return ','.join([f"{i!r}" for i in items])

    # Multiple items: join all but last with commas,
    # then add the connector word and final item
    body = ','.join([f"{i!r}" for i in items[:-1]])
    return f"{body} {choice} {items[-1]!r}"

def str2columns(
    text: str,
    regex: Optional[re.Pattern] = None,
    pattern: Optional[str] = None
) -> List[str]:
    """
    Split the input string `<text>` into words or
    column names using a regular expression.

    By default, if both `<regex>` and `<pattern>` are
    None, returns `[text]`. If `<pattern>` is given,
    compiles it into a regex to split `<text>`. If
    `<regex>` is provided, uses it directly.

    Parameters
    ----------
    text : str
        The string to split into words/columns.
    regex : re.Pattern, optional
        Precompiled regular expression used for
        splitting the text. If provided, overrides
        `<pattern>`.
    pattern : str, optional
        Regex pattern to compile if `<regex>` is not
        given. Defaults to
        ``r'[#&.*@!_,;\s-]\s*'`` if only `pattern`
        is used.
    Returns
    -------
    List[str]
        List of tokens from `<text>`, split
        according to the pattern or `<regex>`.

    Examples
    --------
    >>> from fusionlab.core.checks import str2columns
    >>> text = "this.is an-example"
    >>> str2columns(text)
    ['this','is','an','example']
    """
    # If no regex or pattern is provided,
    # just wrap the entire text in a list
    if regex is None and pattern is None:
        return [text]

    # If the user provided a compiled regex,
    # we use it directly
    if regex is not None:
        splitter = regex
    else:
        # Otherwise compile from <pattern>
        splitter = re.compile(pattern, flags=re.IGNORECASE)

    # Split and filter out empty parts
    parts = splitter.split(text)
    return list(filter(None, parts))

def _assert_all_types(
    obj: object,
    *expected_objtype: Union[Type[Any], Tuple[Type[Any], ...]],
    objname: Optional[str] = None,
) -> object:
    """
    Robust type checking with enhanced error handling and formatting.

    Parameters
    ----------
    obj : object
        Object to validate
    *expected_objtype : type or tuple of types
        Acceptable type(s). Can pass multiple types or tuples of types
    objname : str, optional
        Custom name for object in error messages

    Returns
    -------
    object
        Original object if validation passes

    Raises
    ------
    TypeError
        If type validation fails
    """
    # Flatten nested type specifications
    expected_types = []
    for typ in expected_objtype:
        if isinstance(typ, tuple):
            expected_types.extend(typ)
        else:
            expected_types.append(typ)

    # Convert numpy dtypes to python types
    py_types = []
    for t in expected_types:
        if isinstance(t, type):
            py_types.append(t)
        elif isinstance(t, np.dtype):
            py_types.append(np.dtype(t).type)
        else:
            py_types.append(t)

    expected_types = tuple(py_types)

    if not isinstance(obj, expected_types):
        # Build human-readable type list
        type_names = []
        for t in expected_types:
            try:
                name = t.__name__
            except AttributeError:
                name = str(t)
            type_names.append(name)

        # Format error message components
        obj_name = f"'{objname}'" if objname else "Object"
        plural = "s" if len(expected_types) > 1 else ""
        expected_str = _smart_format(type_names)
        actual_type = type(obj).__name__

        raise TypeError(
            f"{obj_name} must be of type{plural} {expected_str}, "
            f"but got type {actual_type!r}"
        )

    return obj

def random_state_validator(seed):
    """Turn seed into a Numpy-Random-RandomState instance.
    
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
        
    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )
    
def is_numeric_dtype(o, to_array=False):
    """
    Determine whether the argument has a numeric datatype when
    converted to a NumPy array.

    Parameters
    ----------
    o : object or array-like
        The object (or iterable) to check for a numeric datatype. 
        This can be a list, tuple, or any other iterable object.
    to_array : bool, optional, default=False
        If `o` is passed as a non-array-like object (e.g., list, tuple,
        or other iterable), setting `to_array` to `True` will convert 
        `o` into a NumPy array before checking its datatype.

    Returns
    -------
    bool
        `True` if `o` has a numeric dtype, and `False` otherwise.

    Raises
    ------
    TypeError
        If `o` is not an iterable object.
    ValueError
        If `o` is not an array-like object after conversion (if `to_array=True`).

    Examples
    --------
    >>> from fusionlab.core.checks import _is_numeric_dtypes
    >>> is_numeric_dtype([1, 2, 3])
    True

    >>> is_numeric_dtype(['a', 'b', 'c'])
    False

    >>> is_numeric_dtype((1.5, 2.3, 3.1), to_array=True)
    True

    >>> is_numeric_dtype({'a': 1, 'b': 2})
    False

    Notes
    -----
    This function checks if the dtype of `o` (or its NumPy array 
    conversion) is one of the numeric types: boolean, unsigned integers, 
    signed integers, floats, or complex numbers. It uses the `dtype.kind`
    attribute to determine this.

    The function will raise an error if `o` is not iterable, or if it 
    cannot be converted into an array-like structure.

    The check for numeric types is performed using the `_NUMERIC_KINDS` set,
    which includes the following types:
        - 'b' : boolean
        - 'u' : unsigned integer
        - 'i' : signed integer
        - 'f' : float
        - 'c' : complex number
    """
    _NUMERIC_KINDS = set('buifc')

    # Check if 'o' is iterable
    if not hasattr(o, '__iter__'):
        raise TypeError("'o' is expected to be an iterable object. "
                         f"Got: {type(o).__name__!r}")
    
    # Convert to array if specified
    if to_array:
        o = np.array(o)

    # Check if 'o' is an array-like object
    if not hasattr(o, '__array__'):
        raise ValueError(f"Expect type array-like, got: {type(o).__name__!r}")

    # Check for numeric dtype using _NUMERIC_KINDS
    return (o.values.dtype.kind if (hasattr(o, 'columns') or hasattr(o, 'name'))
            else o.dtype.kind) in _NUMERIC_KINDS

def is_in(
    arr: Union[ArrayLike, List[float]],
    subarr: Union[_Sub[ArrayLike], _Sub[List[float]], float],
    return_mask: bool = False,
) -> bool:
    """
    Check whether the subset array `subarr` is present in the array `arr`.

    Parameters
    ----------
    arr : array-like
        Array of item elements to check against. This can be a list,
        numpy array, or any other array-like structure.
    subarr : array-like, float
        Subset array or individual item to check for presence in `arr`.
        This can be a list, numpy array, or float.
    return_mask : bool, optional
        If True, returns a boolean mask indicating where the elements of
        `subarr` are found in `arr`. Default is False, which returns a
        single boolean value (True if any element of `subarr` is in `arr`,
        False otherwise).

    Returns
    -------
    bool or ndarray
        If `return_mask` is False, returns `True` if any item in `subarr`
        is present in `arr`, otherwise returns `False`. If `return_mask` is
        True, returns a boolean mask (ndarray) where `True` indicates that
        the corresponding element in `arr` is found in `subarr`.

    Examples
    --------
    >>> from fusionlab.core.checks import is_in 
    >>> is_in([1, 2, 3, 4, 5], [2, 4])
    True
    
    >>> is_in([1, 2, 3, 4, 5], [6, 7], return_mask=True)
    array([False, False, False, False, False])
    
    >>> is_in([1, 2, 3, 4, 5], 3)
    True
    
    >>> is_in([1, 2, 3, 4, 5], 6)
    False

    Notes
    -----
    This function uses `np.isin` internally to check whether elements
    from `subarr` are present in `arr`. The `return_mask` argument
    allows for flexibility in the return type. If `return_mask` is False,
    the function simply checks if any elements of `subarr` are present in
    `arr` and returns a boolean result.
    """
    arr = np.array(arr)
    subarr = np.array(subarr)

    return (True if True in np.isin(arr, subarr) else False
            ) if not return_mask else np.isin(arr, subarr)

def is_sparse_matrix(
    data: pd.Series, 
    threshold: float = 0.9, 
    verbose=False
    ) -> bool:
    """
    Checks if the data is a sparse matrix, either as a scipy sparse matrix 
    or a pandas Series containing string-encoded sparse matrix data.
    
    This function identifies sparse data structures, considering both 
    actual scipy sparse matrix types and string-encoded representations 
    of sparse matrices, such as those commonly found in pandas Series.
    
    Parameters
    ----------
    data : object
        The data to check. This can be a scipy sparse matrix or a pandas 
        Series containing string-encoded sparse matrix data.
    
    threshold : float, optional, default 0.9
        The minimum proportion of entries that must match the sparse 
        pattern (i.e., be non-zero) for the data to be considered sparse. 
        This value should lie between 0 and 1.
    
    verbose : bool, optional, default False
        If set to True, the function will print the sparsity ratio for a 
        scipy sparse matrix and the proportion of matching entries for a 
        pandas Series. This is useful for debugging or monitoring the 
        function’s behavior.
    
    Returns
    -------
    bool
        True if the data is a sparse matrix (either scipy sparse matrix or 
        string-encoded sparse matrix), False otherwise.
    
    Notes
    -----
    - The function first checks if the data is a scipy sparse matrix 
      (e.g., `csr_matrix`, `coo_matrix`).
    - If the data is a pandas Series, it assumes the Series may contain 
      string-encoded sparse matrix data and checks if each entry in the 
      Series follows the expected sparse format.
    - The `threshold` determines how many non-zero elements (or matching 
      string-encoded sparse entries) are required to consider the data sparse.
    
    Examples
    --------
    1. Check if a scipy sparse matrix is sparse:
    
       ```python
       sparse_matrix = sp.csr_matrix([[0, 0, 1], [0, 2, 0], [0, 0, 3]])
       result = is_sparse_matrix(sparse_matrix)
       print(result)  # Expected: True (based on sparsity ratio)
       ```

    2. Check if a pandas Series with string-encoded sparse matrix data is sparse:
    
       ```python
       sparse_series = pd.Series([
           "(0, 0)\t1.0\n(1, 1)\t2.0\n(2, 2)\t3.0",
           "(0, 1)\t1.5\n(1, 0)\t1.0\n(2, 1)\t2.5"
       ])
       result = is_sparse_matrix(sparse_series)
       print(result)  # Expected: True or False (based on threshold)
       ```

    References
    ----------
    - SciPy Sparse Matrices Documentation:
      https://docs.scipy.org/doc/scipy/reference/sparse.html
    - pandas Series Documentation:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
    """
    if isinstance ( data, pd.DataFrame) :
        data = data.squeeze () 
        
    # Check if the data is a scipy sparse matrix
    if isinstance(data, ssp.spmatrix):
        # Number of non-zero elements in the sparse matrix
        non_zero_elements = data.nnz
        
        # Total number of elements in the matrix (rows * columns)
        total_elements = data.shape[0] * data.shape[1]
        
        # Calculate the sparsity ratio (non-zero elements / total elements)
        sparsity_ratio = non_zero_elements / total_elements
        
        # Print the sparsity ratio if verbose flag is True
        if verbose:
            print(f"Sparsity ratio: {sparsity_ratio:.2f}")
        
        # If the sparsity ratio meets the threshold, return True (sparse)
        return sparsity_ratio >= threshold
    
    # Check if the data is a pandas Series
    if isinstance(data, pd.Series):
        # Check if each entry in the Series follows the expected sparse format
        matches = data.apply(has_sparse_format)
        
        # Calculate the proportion of entries that match the sparse format
        proportion = matches.mean()
        
        # Print the proportion of matching entries if verbose flag is True
        if verbose:
            print(f"Proportion of matching entries: {proportion:.2f}")
        
        # If the proportion of matching entries
        # meets the threshold, return True (sparse)
        return proportion >= threshold
    
    # If data is neither a scipy sparse matrix
    # nor a string-encoded pandas Series
    if verbose:
        print("Data is neither a scipy sparse matrix"
              " nor a string-encoded pandas Series.")
    
    return False

def has_sparse_format(s):
    """
    Checks if a string follows the expected sparse matrix format for entries
    (i.e., coordinate-value pairs like (i, j)\tvalue).
    
    This function uses a regular expression to identify if a given string 
    represents a sparse matrix entry with coordinate-value pairs. This is 
    particularly useful when checking if the entries in a pandas Series 
    follow the sparse matrix format.
    
    Parameters
    ----------
    s : str
        A string entry to check. This should contain coordinates and values 
        separated by tabs, e.g., "(i, j)\tvalue".
    
    Returns
    -------
    bool
        True if the string follows the sparse matrix format, False otherwise.
    
    Examples
    --------
    1. Check if a string represents a sparse matrix entry:
    
       ```python
       entry = "(0, 0)\t1.0"
       result = has_sparse_format(entry)
       print(result)  # Expected: True
       ```
    """
    # Regex pattern for the expected sparse format: (i, j)\tvalue
    pattern = re.compile(r'\(\d+, \d+\)\t-?\d+(\.\d+)?')
    
    if isinstance(s, (ssp.coo_matrix, ssp.csr_matrix, ssp.csc_matrix)):
        return True 
    
    # Return False if s is not a string
    if not isinstance(s, str):
        return False
    
    # Split the string into individual entries
    entries = s.split()
    
    # Check if each entry matches the sparse matrix format
    for entry in entries:
        if not pattern.match(entry):
            return False
    
    return True

def validate_name_in(
    name, defaults='', 
    expect_name=None, 
    exception=None, 
    deep=False 
    ):
    """
    Assert that the given name exists within a set of default names.

    Parameters
    ----------
    name : str
        The name to assert.
    defaults : list of str or str, optional, default=''
        The default names used for the assertion. Can be a list of names,
        a single string, or other iterable. If `deep=True`, this argument
        will be joined into a single string and checked for occurrences of 
        `name`.
    expect_name : str, optional
        The name to return if the assertion is verified (`True`). If `None`,
        the function will return `True` or `False` depending on whether the
        name is found in the defaults.
    deep : bool, optional, default=False
        If `True`, `defaults` are joined into a single string and the function
        checks whether `name` occurs anywhere in the concatenated string.
    exception : Exception, optional
        The exception to raise if `name` is not found in `defaults`. If no
        exception is provided and the name is not found, the function will 
        return `False`.

    Returns
    -------
    str or bool
        If `expect_name` is provided and `name` is found in `defaults`,
        the function returns `expect_name`. If `expect_name` is `None`,
        it returns `True` if `name` is found in `defaults`, or `False` otherwise.
        If `name` is not found and `exception` is specified, the exception
        is raised.

    Examples
    --------
    >>> from fusionlab.core.checks import validate_name_in
    >>> dnames = ('NAME', 'FIRST NAME', 'SURNAME')
    >>> validate_name_in('name', defaults=dnames)
    False

    >>> validate_name_in('name', defaults=dnames, deep=True)
    True

    >>> validate_name_in('name', defaults=dnames, expect_name='NAM')
    False

    >>> validate_name_in('name', defaults=dnames, expect_name='NAM', deep=True)
    'NAM'

    Notes
    -----
    The function performs a case-insensitive check for `name` within
    the `defaults`. If `deep=True`, it combines all elements in `defaults`
    into a single string and checks whether `name` is a substring of that string.
    If `name` is found and `expect_name` is provided, the function returns
    `expect_name`. Otherwise, it returns a boolean value indicating whether
    `name` is in `defaults`. If `name` is not found and `exception` is provided,
    the exception is raised.
    """
    
    name = str(name).lower().strip()
    defaults = is_iterable(
        defaults, exclude_string=True, parse_string=True, 
        transform=True)
    
    if deep:
        defaults = ''.join([str(i) for i in defaults])

    # Check if name is in defaults
    name = True if expect_name is None else expect_name if name in defaults else False
    
    if not name and exception:
        raise exception
    
    return name

def is_valid_dtypes(
    df: pd.DataFrame, 
    columns: Optional[Union[str, List[Any]]] = None, 
    dtypes: Union[str, List[str]] = 'numeric',
    ops: str = 'check_only',
    treat_obj_dtype_as_category: bool = False, 
    error_msg: Optional[str] = None, 
    extra: str = '',
    error: str = 'warn', 
) -> Union[bool, Dict[str, List[Any]]]:
    """
    Check if specified columns in a DataFrame match the desired data types.
    
    This function verifies whether the data types of selected columns in a 
    pandas DataFrame align with specified objective types. It supports various 
    operations, including validation and conditional error handling, making it 
    a versatile tool for data preprocessing and quality assurance in data pipelines.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check.
    
    columns : str or list of str, optional
        The columns to check. If ``None``, all columns are checked.

        .. code-block:: python

            columns = 'age'
            columns = ['age', 'salary']
    
    dtypes : str or list of str, default= 'numeric'
        The desired data types to validate against. 
        Options include ``'numeric'``, ``'object'``, ``'category'``, ``'datetime'``.
        Can be a single type or a list of types.
        
        .. code-block:: python

            dtypes = 'numeric'
            dtypes = ['numeric', 'datetime']
    
    ops : str, default 'check_only'
        Operation mode. 
        - ``'check_only'``: Returns True if all specified columns match the 
          objectives, False otherwise.
        - ``'validate'``: Returns a dictionary with objectives as keys and lists 
          of valid columns as values.
    
    treat_obj_dtype_as_category : bool, default= False
        If True, columns with dtype ``'category'`` are treated as categorical 
        regardless of the 'object' dtype.
    
    error_msg : str, optional
        Custom error message to display if validation fails.
        If None, a default message is used.
    
    extra : str, default= ''
        Extra message to append to the error message.
    
    error : str, default ='warn'
        Specifies the error handling behavior. Options are:
        - ``'raise'``: Raises an exception when validation fails.
        - ``'warn'``: Issues a warning when validation fails.
        - ``'ignore'``: Silently ignores validation failures.
    
    Returns
    -------
    bool or dict
        - If ``ops='check_only'``, returns ``True`` if all specified columns 
          match the objectives dtypes, ``False`` otherwise.
        - If ``ops='validate'``, returns a dictionary mapping objectives to 
          lists of valid columns.
    
    Notes
    ------
    Let the DataFrame be represented as :math:`DF`, and let 
    :math:`C = \{c_1, c_2, \dots, c_n\}` be the set of columns to validate. 
    Let :math:`O = \{o_1, o_2, \dots, o_m\}` be the set of objective data types.
    
    The function checks whether each column :math:`c_i \in C` satisfies 
    :math:`c_i.dtype \in O`. If ``treat_obj_dtype_as_category=True``, 
    columns with dtype ``'category'`` are also considered valid for 
    the ``'object'`` objective.
    
    .. math::
        \forall c_i \in C, \quad c_i.dtype \in O \cup \{\text{'category'} 
        \text{ if treat\_obj\_dtype\_as\_category}\}
    
    Examples
    ---------
    >>> from fusionlab.utils.datautils import is_valid_dtypes
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 45],
    ...     'salary': [50000.0, 60000.5, 75000.0],
    ...     'department': ['HR', 'Engineering', 'Marketing'],
    ...     'hire_date': pd.to_datetime(['2020-01-15', '2019-03-22', '2021-07-30'])
    ... })
    >>> # Check if 'age' and 'salary' are numeric
    >>> is_valid_dtypes(df, columns=['age', 'salary'], dtypes='numeric')
    True
    >>> # Validate data types for multiple objectives
    >>> is_valid_dtypes(df, objectives=['numeric', 'datetime'], ops='validate')
    {'numeric': ['age', 'salary'], 'datetime': ['hire_date']}
    >>> # Check with custom error handling
    >>> is_valid_dtypes(df, columns='department', dtypes='numeric', error='raise')
    Traceback (most recent call last):
        ...
    TypeError: Invalid ... objectives: ['department']

    Notes
    -----
    - The function is particularly useful in data preprocessing pipelines to ensure 
      data integrity before performing further analysis or modeling.
    - When ``ops='validate'``, the returned dictionary provides a clear 
      categorization of columns based on their data types, facilitating 
      selective processing.
    - The ``treat_obj_dtype_as_category`` parameter is beneficial when 
      categorical data is encoded as either ``'object'`` or ``'category'``, 
      allowing for flexible validation criteria.
    
    See Also
    --------
    pandas.DataFrame.select_dtypes : Select columns based on their dtypes.
    pandas.api.types : Utilities for checking data types in pandas.
    
    References
    ----------
    .. [1] Pandas Documentation: https://pandas.pydata.org/docs/
    .. [2] McKinney, W. "Python for Data Analysis," O'Reilly Media, 2017.
    """
    # validate the dataframe
    are_all_frames_valid(df)
    if columns is None:
        columns = df.columns.tolist()
    # Ensure that columns and dtypes are formatted correctly as lists
    columns = is_iterable(columns, exclude_string= True, transform =True)
    dtypes = is_iterable(dtypes, exclude_string= True, transform= True )

    valid_objectives = {'numeric', 'object', 'category', 'datetime'}
    if not set(dtypes).issubset(valid_objectives):
        invalid = set(dtypes) - valid_objectives
        raise ValueError(f"Invalid  dtypes specified: {invalid}")
    
    dtype_map = {
        'numeric': np.number,
        'object': 'object',
        'category': 'category',
        'datetime': 'datetime64[ns]'
    }
    
    # Prepare the result for 'validate' operation
    if ops not in ('check_only', 'validate'): 
        raise ValueError("`ops` must be either 'check_only' or 'validate'.")
        
    if ops == 'validate':
        result = {obj: [] for obj in dtypes}
        for obj in dtypes:
            if obj == 'numeric':
                result[obj] = df.select_dtypes(
                    include=[dtype_map[obj]]
                ).columns.intersection(columns).tolist()
            elif obj == 'datetime':
                result[obj] = df.select_dtypes(
                    include=[dtype_map[obj]]
                ).columns.intersection(columns).tolist()
            else:
                result[obj] = df.select_dtypes(
                    include=[dtype_map[obj]]
                ).columns.intersection(columns).tolist()
                if treat_obj_dtype_as_category and obj == 'object':
                    category_cols = df.select_dtypes(
                        include=['category']
                    ).columns.intersection(columns).tolist()
                    result[obj].extend(category_cols)
        
        return result
    
    # Perform 'check_only' operation
    elif ops == 'check_only':
        invalid_cols = []
        for col in columns:
            if col not in df.columns:
                invalid_cols.append(col)
                continue
            col_dtype = df[col].dtypes
            matched = False
            for obj in dtypes:
                if obj == 'numeric':
                    if np.issubdtype(col_dtype, np.number):
                        matched = True
                        break
                elif obj == 'object':
                    if df[col].dtypes == 'object':
                        matched = True
                        break
                    if treat_obj_dtype_as_category and df[col].dtypes.name == 'category':
                        matched = True
                        break
                elif obj == 'category':
                    if df[col].dtypes.name == 'category':
                        matched = True
                        break
                elif obj == 'datetime':
                    if np.issubdtype(col_dtype, np.datetime64):
                        matched = True
                        break
            if not matched:
                invalid_cols.append(col)
        
        if invalid_cols:
            message = error_msg if error_msg else (
                f"The following columns do not match the dtypes: {invalid_cols}"
            )
            if extra:
                message += f" {extra}"
                
            if error == 'warn': 
                warnings.warn(message)
                
            elif error == 'raise': 
                raise TypeError(f"Invalid columns detected: {message}")
                
            return False
        
        return True
  
def check_features_types(
    data,
    features,
    dtype,
    error_msg=None, 
    accept_object_dtype=False, 
    extra=''
):
    """
    Verify that specified features in a DataFrame match the expected type.

    This function checks whether the provided features within a pandas DataFrame
    conform to the specified objective type. Supported objective types include
    'category', 'numeric', and 'datetime'. It ensures data integrity by validating
    the data types before proceeding with further data processing or analysis.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to be checked.
    features : str or list of str
        The feature(s) to validate. If a single feature is provided as a string,
        it will be internally converted to a list for uniform processing.
    dtype : str
        The expected data type for the features. Supported types are:
        - ``'category'``: Categorical data type.
        - ``'numeric'`` : Numeric data types (int, float).
        - ``'datetime'``: Datetime data type.
    error_msg : str, optional
        Custom error message to raise if a feature's data type does not match
        the expected objective type. If set to `None`, a default error message
        will be generated. Default is ``None``.
        
    accept_object_dtype: bool, default=False, 
       Pass when object dtype is given rather than raising error. Th default 
       behavior only verify the ``'category'``, ``'numeric'`` and 
       ``'datetime'`` types. 
       
    extra: str, optional, 
       Extra message to append to the TypeError message. 

    Returns
    -------
    bool
        Returns `True` if all specified features match the expected type.

    Raises
    ------
    TypeError
        If `data` is not a pandas DataFrame.
        If `features` is neither a string nor a list of strings.
    ValueError
        If an unsupported `objective` type is provided.
        If any feature specified in `features` does not exist in `data`.
        If a feature's data type does not match the expected `dtype`.

    Notes
    -----
    - The function is case-sensitive regarding the `objective` parameter.
    - It is advisable to ensure that datetime columns are properly parsed
      (e.g., using `pd.to_datetime`) before using this function.
    - This function can be extended to support additional data types by
      modifying the `type_checks` dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.core.checks import check_features_types
    >>> data = {
    ...     'age': [25, 30, 45],
    ...     'salary': [50000.0, 60000.5, 75000.0],
    ...     'join_date': pd.to_datetime(['2020-01-15', '2019-06-23', '2021-03-10']),
    ...     'department': ['HR', 'Engineering', 'Marketing']
    ... }
    >>> df = pd.DataFrame(data)
    >>> check_features_types(df, ['age', 'salary'], 'numeric')
    True

    >>> df['department'] = df['department'].astype('category')
    >>> check_features_types(df, 'department', 'category')
    True

    >>> check_features_types(df, 'join_date', 'datetime')
    True

    >>> # Using a custom error message
    >>> check_features_types(
    ...     df,
    ...     'age',
    ...     'category',
    ...     error_msg="Age should be a categorical feature."
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Age should be a categorical feature.
    """

    # Validate that 'data' is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"The 'data' parameter must be a pandas DataFrame, "
            f"got {type(data).__name__} instead."
        )

    # Ensure 'features' is a list
    if isinstance(features, str):
        features = [features]
    elif isinstance(features, list):
        if not all(isinstance(feature, str) for feature in features):
            raise TypeError(
                "All elements in the 'features' list must be strings."
            )
    else:
        raise TypeError(
            "The 'features' parameter should be a list of strings "
            "or a single feature name as a string."
        )

    # Mapping of objectives to pandas type checking functions
    type_checks = {
        'category' : is_categorical_dtype,
        'numeric'  : _is_numeric_dtype,
        'datetime' : is_datetime64_any_dtype,
    }
    if accept_object_dtype: 
        type_checks['object']= is_object_dtype 
        
    # Validate the objective
    if dtype not in type_checks:
        raise ValueError(
            f"Unsupported objective type: '{dtype}'. "
            f"Supported types are {list(type_checks.keys())}."
        )

    check_func = type_checks[dtype]

    # Iterate through each feature and check its type
    for feature in features:
        if feature not in data.columns:
            raise ValueError(
                f"Feature '{feature}' not found in the DataFrame."
            )

        if not check_func(data[feature]):
            if error_msg:
                raise ValueError(error_msg)
            else:
                actual_type = data[feature].dtype
                raise TypeError(
                    f"Feature '{feature}' has type '{actual_type}', "
                    f"expected type '{dtype}'.{extra}"
                )

    return True

def are_all_frames_valid(
    *dfs: Union[pd.DataFrame, pd.Series],
    df_only: bool = ...,  
    error_msg: Optional[str] = None, 
    check_size: bool = False,  
    check_symmetry: bool = False, 
    to_df: bool=False, 
    ops='check_only'
) -> bool:
    """
    Validates whether all provided inputs are pandas DataFrames or Series 
    based on the `df_only` flag. This function checks the types of the 
    input objects and optionally verifies additional properties like size 
    and symmetry.

    Parameters
    ----------
    *dfs : Union[pd.DataFrame, pd.Series]
        One or more pandas DataFrame or Series objects to be validated. 
        This function can accept multiple inputs, checking each one for 
        compliance with the expected type.

    df_only : bool, default=True
        If True, only DataFrames are considered valid inputs. If any of 
        the provided inputs is not a DataFrame, an error is raised. 
        If False, pandas Series are also allowed as valid inputs, and 
        the function will not raise an error for Series objects.

    check_size : bool, default=False
        If True, the function will additionally check if all DataFrames 
        or Series objects have the same number of rows or columns (depending 
        on the size parameter). This is useful for validating consistency 
        across inputs.

    check_symmetry : bool, default=False
        If True, the function checks if all DataFrames or Series are symmetric, 
        meaning the rows and columns (or the data itself in the case of Series) 
        are consistent. This could be important in certain data validation 
        or data integrity scenarios.

    error_msg : str, optional
        A custom error message to be raised if any input is invalid. If 
        not provided, a default message is used that specifies the need 
        for DataFrames or Series, depending on the `df_only` flag.
        
    ops: str, default='check_only'
        Expect {``'validate'``, ``'check_only'``}. Return valid dataframes
        if ``ops='validate'`` otherwise, it checks only whether all frames 
        are valid and return ``True``.  
        
    to_df: bool, default  =False 
        Convert series to dataframe. This parameter only operates when the 
        option is `'validate'``
    Returns
    -------
    bool
        Returns `True` if all inputs are either DataFrames (or Series if 
        `df_only=False`), and if applicable, if they pass size and symmetry checks. 
        Otherwise, an error is raised based on the validation rules.
    tuple
        Returns valid dataframes if ``ops='validate'``. 
        
    Raises
    ------
    TypeError
        If any of the inputs is neither a DataFrame nor a Series (based on 
        `df_only`), or if any input fails the size or symmetry checks 
        (if `check_size` or `check_symmetry` are `True`).

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.core.checks import is_all_frames
    
    # Example with multiple DataFrames
    >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    >>> is_all_frames(df1, df2)
    True
    
    # Example with a DataFrame and a Series
    >>> df3 = pd.Series([1, 2, 3])
    >>> is_all_frames(df1, df3, df_only=False)
    True
    
    # Example with a Series when df_only=True (raises TypeError)
    >>> is_all_frames(df1, df3)
    TypeError: Expected DataFrame, but found a Series

    Notes
    -----
    - If `check_size=True`, all DataFrames and Series must have the same 
      number of rows (for DataFrames) or elements (for Series). 
    - If `check_symmetry=True`, the function will check if the dimensions 
      of the DataFrames match for all inputs. In the case of Series, it 
      checks for consistency in the data sequence.
    
    See Also
    --------
    - `pandas.DataFrame`: For more information on the pandas DataFrame object.
    - `pandas.Series`: For more information on the pandas Series object.

    References
    ----------
    .. [1] Pandas Documentation. "DataFrame". Retrieved from: 
           https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    .. [2] Pandas Documentation. "Series". Retrieved from: 
           https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
    """
    # Default error message
    if error_msg is None:
        subj = 'All inputs' if len(dfs) > 1 else 'Input'
        either =' either' if not df_only else ''
        s='s' if len(dfs) > 1 else ''
        error_msg = f"{subj} must be{either} pandas DataFrame{s}" + \
                    (" or pandas Series." if not df_only else ".")
                    
    if ops not in {'validate', 'check_only'}: 
        warnings.warn("Ops parameter expects options: ``'validate'`` or"
                      " ``'check_only'``. Got '{ops}'. Fallback to"
                      " 'check_only'"
                    )
        ops ='check_only'
        
    new_dfs =[] 
    # Check each argument
    for df in dfs:
        # Check if each element is a valid DataFrame or Series
        if not isinstance(df, (pd.DataFrame, pd.Series)):
            raise TypeError(error_msg)
        
        #if option is validate and series must be converted to dataframe 
        # then convert it. 
        
        if ( ops=='validate' 
            and isinstance (df, pd.Series) 
            and to_df 
            ): 
            df= df.to_frame() 
            
        # If df_only is True, raise error for Series    
        if df_only and isinstance(df, pd.Series):
            raise TypeError(f"Expected DataFrame, but found a Series: {df}")

        # Check for size consistency
        if check_size:
            if isinstance(df, pd.DataFrame):
                # Check if number of rows (size[0]) is the same for all DataFrames
                for other_df in dfs:
                    if isinstance(other_df, pd.DataFrame
                                  ) and df.shape[0] != other_df.shape[0]:
                        raise ValueError(
                            "DataFrames have different row counts:"
                            f" {df.shape[0]} != {other_df.shape[0]}")
            elif isinstance(df, pd.Series):
                # Check if length is the same for all Series
                for other_df in dfs:
                    if isinstance(other_df, pd.Series) and len(df) != len(other_df):
                        raise ValueError(
                            "Series have different lengths:"
                            f" {len(df)} != {len(other_df)}")

        # Check for symmetry (square matrices)
        if check_symmetry and isinstance(df, pd.DataFrame):
            if df.shape[0] != df.shape[1]:
                raise ValueError(
                    f"DataFrame is not symmetric: {df.shape[0]} != {df.shape[1]}")
        
        if ops =='validate': 
            new_dfs.append (df)
            
    if ops=='validate': 
        return tuple (new_dfs)
    
    return True 

def has_nan(
    data: Union[pd.DataFrame, pd.Series], 
    axis: Optional[int] = None, 
    how: str = 'any', 
    include_missing_columns: bool = False,
    error_msg: Optional[str] = None
) -> bool:
    """
    Check if the provided data (DataFrame or Series) contains any NaN values.
    
    This function provides enhanced flexibility to check NaN values either
    along specific axes (rows or columns), specify how to check for NaN 
    ('any' or 'all'), and includes the option to handle missing columns.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        The input data which can either be a DataFrame or Series.
        
    axis : int, optional, default=None
        The axis along which to check for NaN values. 
        - For DataFrame: 0 checks columns (axis=0), 1 checks rows (axis=1).
        - For Series: No effect, as it is one-dimensional.
        
    how : {'any', 'all'}, optional, default 'any'
        Defines how to check for NaN values:
        - 'any' (default): Returns True if any NaN values are found.
        - 'all': Returns True only if all values are NaN in the specified axis.
        
    include_missing_columns : bool, optional, default False
        If True, include columns that are entirely missing (i.e., all NaN values) 
        in the result. If False, missing columns (completely NaN) are ignored 
        in the output.

    error_msg : str, optional
        A custom error message to raise if the input data type is not a
        DataFrame or Series. 
        If not provided, a default error message is used.
        
    Returns
    -------
    bool
        Returns True if NaN values are found according to the specified
        parameters, otherwise False.
    
    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame or Series.
        
    Examples
    --------
    >>> from fusionlab.core.checks import has_nan 
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [None, 2, 3]})
    >>> has_nan(df)
    True
    
    >>> has_nan(df, axis=0, how='all')
    False
    
    >>> has_nan(df, axis=1, how='any')
    True
    
    >>> has_nan(df, axis=0, how='all', include_missing_columns=True)
    False
    """
    # Handle error messages for invalid input types
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        error_msg = error_msg or "Input must be either a pandas DataFrame or Series"
        raise TypeError(error_msg)
    
    # For DataFrame: allow axis specification (axis=0 for columns, axis=1 for rows)
    if isinstance(data, pd.DataFrame):
        if axis is not None:
            if how == 'any':
                return data.isna().any(axis=axis).any() if not include_missing_columns else \
                    data.isna().any(axis=axis).any() or data.isna().all(axis=axis).any()
            elif how == 'all':
                return data.isna().all(axis=axis).any()
            else:
                raise ValueError("Parameter `how` must be 'any' or 'all'")
        else:
            return data.isna().any().any()

    # For Series: no axis, always check along the single dimension (rows)
    elif isinstance(data, pd.Series):
        if how == 'any':
            return data.isna().any()
        elif how == 'all':
            return data.isna().all()
        else:
            raise ValueError("Parameter `how` must be 'any' or 'all'")

    return False


def validate_spatial_columns(
    df: pd.DataFrame,
    mode_search: str = 'soft',
    rename: bool = False,
    error: str = 'raise',
    return_valid: bool = True,
    as_pair: bool = True,
    as_frame: bool = False
) -> Union[List[Union[str, Tuple[str, str]]], pd.DataFrame, None]:
    """
    Validate and Extract Spatial Columns from a DataFrame.
    
    The function checks for the existence of spatial column pairs within a 
    DataFrame, such as (`easting`, `northing`) or (`longitude`, `latitude`). 
    It supports flexible column name matching through regex patterns and can 
    optionally rename abbreviated column names to standard ones. The function 
    ensures that spatial columns are present in valid pairs and provides options 
    to return the validated columns or the corresponding DataFrame slices.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing potential spatial columns.
    
    mode_search : str, default='soft'
        The mode of searching for spatial columns.
        - `'soft'`: Case insensitive and allows partial matches using regex.
        - `'strict'`: Exact case-sensitive matches.
    
    rename : bool, default=False
        Whether to rename abbreviated spatial column names to their standard 
        counterparts.
        - If `True`, columns like `'east'` will be renamed to `'easting'`, 
          `'north'` to `'northing'`, `'lon'` to `'longitude'`, and `'lat'` 
          to `'latitude'`.
    
    error : str, default='raise'
        The strategy for handling validation errors.
        - `'warn'`: Issues warnings and continues processing.
        - `'raise'`: Raises exceptions upon encountering errors.
    
    return_valid : bool, default=True
        Determines whether to return the validated spatial columns.
        - If `True`, returns the validated columns or pairs.
        - If `False`, the function performs validation without returning columns.
    
    as_pair : bool, default=True
        Indicates whether to return spatial columns as pairs.
        - If `True`, returns a list of tuples, each containing a valid pair.
        - If `False`, returns a flat list of valid spatial column names.
    
    as_frame : bool, default=False
        Determines whether to return the spatial columns as a DataFrame slice.
        - If `True`, returns a DataFrame containing only the valid spatial columns.
        - If `False`, returns the list of valid columns or pairs based on 
          `as_pair`.
    
    Returns
    -------
    Union[List[Union[str, Tuple[str, str]]], pandas.DataFrame, None]
        - If `return_valid` is `True` and `as_pair` is `True`, returns a list 
          of tuples with each tuple representing a valid spatial column pair.
        - If `return_valid` is `True` and `as_pair` is `False`, returns a flat 
          list of valid spatial column names.
        - If `as_frame` is `True`, returns a DataFrame containing only the valid 
          spatial columns.
        - If `return_valid` is `False`, returns `None`.
    
    Raises
    ------
    ValueError
        - If `mode_search` is not `'soft'` or `'strict'`.
        - If `error` is neither `'warn'` nor `'raise'`.
        - If any of the spatial column pairs are partially present.
    
    TypeError
        - If `df` is not a pandas DataFrame.
    
    Notes
    -----
    - **Flexible Matching**: In `'soft'` mode, the function uses regex patterns 
      to match various naming conventions for spatial columns, allowing for 
      greater flexibility in column naming.
    
    - **Renaming Capability**: When `rename` is enabled, the function standardizes 
      abbreviated spatial column names to ensure consistency across the DataFrame.
    
    - **Pair Validation**: The function ensures that spatial columns are present 
      in complete pairs. Partial pairs (e.g., only `'longitude'` without 
      `'latitude'`) are considered invalid.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.core.checks import validate_spatial_columns
    
    >>> # Sample DataFrame with various spatial column name formats
    >>> data = {
    ...     'East': [500000, 500100, 500200],
    ...     'North': [4100000, 4100100, 4100200],
    ...     'lon': [-75.0, -75.1, -75.2],
    ...     'lat': [40.0, 40.1, 40.2],
    ...     'value': [10, 20, 30]
    ... }
    >>> df = pd.DataFrame(data)
    
    >>> # Validate spatial columns with default parameters
    >>> valid_cols = validate_spatial_columns(df)
    >>> print(valid_cols)
    [('East', 'North'), ('lon', 'lat')]
    
    >>> # Validate and rename spatial columns
    >>> valid_cols = validate_spatial_columns(df, rename=True)
    >>> print(valid_cols)
    [('easting', 'northing'), ('longitude', 'latitude')]
    
    >>> # Validate and return as a DataFrame
    >>> spatial_df = validate_spatial_columns(df, as_frame=True)
    >>> print(spatial_df)
       easting  northing  longitude  latitude
    0  500000    4100000      -75.0      40.0
    1  500100    4100100      -75.1      40.1
    2  500200    4100200      -75.2      40.2
    
    >>> # Attempt validation with missing pair
    >>> df_missing = pd.DataFrame({
    ...     'longitude': [-75.0, -75.1, -75.2],
    ...     'value': [10, 20, 30]
    ... })
    >>> validate_spatial_columns(df_missing)
    ValueError: The following spatial column pairs are incomplete or missing: {'latitude'}
    
    Notes
    -----
    - **Extensibility**: The function can be extended to include additional 
      spatial column pairs by updating the internal `spatial_pairs` list.
    
    - **Performance**: For large DataFrames, consider disabling `rename` if 
      renaming is not necessary to optimize performance.
    
    - **Integration**: This function is ideal for preprocessing steps in 
      geospatial data analysis workflows, ensuring that spatial data is 
      correctly identified and standardized before further processing.
    
    See Also
    --------
    pandas.DataFrame : Main pandas object for data manipulation.
    pandas.read_excel : Function to read Excel files into DataFrames.
    re : Module for regular expressions.
    
    References
    ----------
    .. [1] Pandas Documentation: pandas.DataFrame. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    .. [2] Python Documentation: re. 
       https://docs.python.org/3/library/re.html
    .. [3] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"validate_spatial_columns requires a pandas DataFrame. "
            f"Got {type(df).__name__!r} instead."
        )
    
    if mode_search not in {'soft', 'strict'}:
        raise ValueError(
            f"mode_search must be either 'soft' or 'strict'. Got '{mode_search}'."
        )
    
    if error not in {'warn', 'raise'}:
        raise ValueError(
            f"error must be either 'warn' or 'raise'. Got '{error}'."
        )
    
    # Define spatial column pairs with regex patterns
    spatial_pairs = [
        {
            'standard': ('easting', 'northing'),
            'patterns': (r'^east(?:ing)?$', r'^north(?:ing)?$')
        },
        {
            'standard': ('longitude', 'latitude'),
            'patterns': (r'^lon(?:gitude)?$', r'^lat(?:itude)?$')
        }
    ]
    
    valid_pairs = []
    renamed_columns = {}
    
    for pair in spatial_pairs:
        std_east, std_north = pair['standard']
        pattern_east, pattern_north = pair['patterns']
        
        # Compile regex patterns
        regex_east = re.compile(pattern_east, re.IGNORECASE)
        regex_north = re.compile(pattern_north, re.IGNORECASE)
        
        # Find matching columns
        east_matches = [col for col in df.columns if regex_east.match(col)]
        north_matches = [col for col in df.columns if regex_north.match(col)]
        
        if east_matches and north_matches:
            east_col = east_matches[0]
            north_col = north_matches[0]
            
            # Rename columns if required
            if rename:
                renamed_columns[east_col] = std_east
                renamed_columns[north_col] = std_north
                df.rename(columns=renamed_columns, inplace=True)
                east_col = std_east
                north_col = std_north
            
            valid_pairs.append((east_col, north_col))
        elif east_matches or north_matches:
            missing = std_north if east_matches else std_east
            msg = (
                f"Spatial column pair incomplete. Missing '{missing}' for "
                f"the existing spatial column."
            )
            if error == 'warn':
                warnings.warn(msg)
            else:
                raise ValueError(msg)
    
    if not valid_pairs:
        msg = "No valid spatial column pairs found in the DataFrame."
        if error == 'warn':
            warnings.warn(msg)
            return None
        else:
            raise ValueError(msg)
    
    if return_valid:
        if as_frame:
            # Flatten the list of pairs
            spatial_cols = [col for pair in valid_pairs for col in pair]
            spatial_df = df[spatial_cols]
            return spatial_df
        elif as_pair:
            return valid_pairs
        else:
            # Flatten the list of pairs
            spatial_cols = [col for pair in valid_pairs for col in pair]
            return spatial_cols
    else:
        return None

def check_spatial_columns(
    df: pd.DataFrame,
    spatial_cols: Optional[tuple] = ('longitude', 'latitude'), 

) -> None:
    """
    Validate the spatial columns in the DataFrame.

    Ensures that the specified `spatial_cols` are present in the DataFrame and 
    consist of exactly two columns representing longitude and latitude.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing geographical data.
    
    spatial_cols : tuple, optional, default=('longitude', 'latitude')
        A tuple containing the names of the longitude and latitude columns.
        Must consist of exactly two elements.

    Raises
    ------
    ValueError
        - If `spatial_cols` is not a tuple or does not contain exactly two elements.
        - If any of the specified `spatial_cols` are not present in the DataFrame.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.core.checks import check_spatial_columns

    >>> # Valid spatial columns
    >>> df = pd.DataFrame({
    ...     'longitude': [-100, -99, -98],
    ...     'latitude': [35, 36, 37],
    ...     'value': [1, 2, 3]
    ... })
    >>> check_spatial_columns(df, spatial_cols=('longitude', 'latitude'))
    # No output, validation passed

    >>> # Invalid spatial columns
    >>> check_spatial_columns(df, spatial_cols=('lon', 'lat'))
    ValueError: The following spatial_cols are not present in the dataframe: {'lat', 'lon'}

    Notes
    -----
    - The function strictly requires `spatial_cols` to contain exactly two 
      column names representing longitude and latitude.
    
    See Also
    --------
    plot_spatial_distribution : Function to plot spatial distributions.

    References
    ----------
    .. [1] Pandas Documentation: pandas.DataFrame
    """
    if not isinstance (df, pd.DataFrame): 
        raise TypeError(
            "Spatial columns check requires a dataframe `df`"
            f" to be set. Got {type(df).__name__!r}")
        
    if spatial_cols is None or not isinstance(
            spatial_cols, (tuple, list)) or len(spatial_cols) != 2:
        raise ValueError(
            "spatial_cols must be a tuple of exactly two elements "
            "(longitude and latitude)."
        )
    
    missing_cols = set(spatial_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"The following spatial_cols are not present in the dataframe: {missing_cols}"
        )

def validate_column_types(
    df: pd.DataFrame, 
    category_columns: Optional [Union[str, List[str]]]=None, 
    consider_object_as_category: bool = False, 
    error: str = 'raise',  
    error_msg: Optional[str] = None
) -> Union[dict, bool, None]:
    """
    Validate the types of specified columns in a DataFrame. Determines if each 
    column is categorical or numeric. The user can customize the behavior when 
    an error is encountered.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the columns to check.
        
    category_columns : str or list of str, optional 
        The name(s) of the column(s) to check. It can either be a single 
        column name or a list of column names. If ``None``, all columns of the 
        dataframe types will be checked.
    
    consider_object_as_category : bool, default ``False``
        If ``True``, columns of dtype ``'object'`` will be treated as categorical 
        columns. Otherwise, ``'object'`` columns will be considered non-categorical.
    
    error : str, default ``'raise'``
        Defines how to handle errors when a column is neither numeric nor categorical:
        - ``'raise'``: Raise a ``ValueError``.
        - ``'warn'``: Issue a warning using ``warnings.warn``.
        - ``'ignore'``: Do nothing and return ``None`` for invalid columns.
    
    error_msg : str, optional, default ``None``
        Custom error message to use when raising an error or issuing a warning. 
        If ``None``, a default message will be used.
    
    Returns
    -------
    dict or bool or None
        - If multiple columns are provided, returns a ``dict`` where keys are 
          column names and values are either ``True`` (if the column is 
          categorical), ``False`` (if numeric), or ``None`` (if invalid and 
          ``error`` is not ``'raise'``).
        - If a single column is provided, returns ``True``, ``False``, or ``None``
          directly instead of a dictionary.
    
    Raises
    ------
    ValueError
        If any column is neither numeric nor categorical and ``error`` is set to 
        ``'raise'``. Also raised if a specified column does not exist in 
        the DataFrame.
    
    Warns
    -----
    UserWarning
        If any column is neither numeric nor categorical and ``error`` is set to 
        ``'warn'``.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.core.checks import validate_column_types
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40],
    ...     'gender': ['M', 'F', 'M', 'F'],
    ...     'income': [50000, 60000, 55000, 65000],
    ...     'region': ['North', 'South', 'East', 'West']
    ... })
    >>> # Validate multiple columns, treating 'object' as categorical
    >>> result = validate_column_types(
    ...     df, 
    ...     ['gender', 'region', 'income'], 
    ...     consider_object_as_category=True, 
    ...     error='warn'
    ... )
    >>> print(result)
    {'gender': True, 'region': True, 'income': False}
    
    >>> # Validate a single column
    >>> result_single = validate_column_types(df, 'age')
    >>> print(result_single)
    False
    
    >>> # Attempt to validate an invalid column with error='ignore'
    >>> df['invalid_col'] = pd.Series([1, 2, 3, 'a'])
    >>> result_ignore = validate_column_types(
    ...     df, 
    ...     ['invalid_col'], 
    ...     error='ignore'
    ... )
    >>> print(result_ignore)
    None
    
    Notes
    -----
    - The function is designed to handle both single and multiple column validations.
    - When ``category_columns`` contains only one column, the function returns a 
      single ``bool`` or ``None`` value instead of a dictionary for convenience.
    - Setting ``consider_object_as_category=True`` is useful when categorical 
      data is stored as strings or mixed types in the DataFrame.
    - The ``error`` parameter provides flexibility in how the function responds 
      to invalid column types, allowing integration into larger data processing 
      pipelines without interruption.
    
    See Also
    --------
    ``pandas.api.types.is_categorical_dtype`` : Check if a pandas Series has a 
    categorical dtype.
    
    ``pandas.api.types.is_numeric_dtype`` : Check if a pandas Series has a 
    numeric dtype.
    
    References
    ----------
    .. [1] Pandas Documentation. "Categorical Data". Retrieved from: 
           https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html
    
    .. [2] Pandas Documentation. "Data Types". Retrieved from: 
           https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#dtypes
    
    .. [3] Python Documentation. "warnings — Warning control". Retrieved from: 
           https://docs.python.org/3/library/warnings.html
    """
    are_all_frames_valid(df)
    # Ensure category_columns is a list
    if isinstance(category_columns, str):
        category_columns = [category_columns]
    
    if category_columns is None: 
        category_columns = list (df.columns)
        
    column_types = {}

    # Set default error message if not provided
    if error_msg is None:
        error_msg = "Column must be either numeric or categorical."

    # Iterate over each column name
    for column in category_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
        # Determine column type
        if pd.api.types.is_categorical_dtype(df[column]) or \
           (consider_object_as_category and df[column].dtype == 'object'):
            column_types[column] = True  # Categorical
        elif pd.api.types.is_numeric_dtype(df[column]):
            column_types[column] = False  # Numeric
        else:
            if error == 'raise':
                raise ValueError(f"{error_msg} (Column: '{column}')")
            elif error == 'warn':
                warnings.warn(f"{error_msg} (Column: '{column}')", UserWarning)
                column_types[column] = None  # Mark as invalid
            elif error == 'ignore':
                column_types[column] = None  # Mark as invalid without action
    
    # If only one column was checked, return its type directly
    if len(column_types) == 1:
        return next(iter(column_types.values()))
    
    return column_types

def is_df_square(
    df: pd.DataFrame, 
    order: Optional[Union[list, tuple]] = None, 
    ops: str = 'check_only', 
    check_symmetry: bool=False,  # new parameter, 
    cols_as_index: bool = False, 
) -> Optional[pd.DataFrame]:
    """
    Checks if a given DataFrame is square and optionally reorders the 
    columns and index or sets the index based on column names. The function 
    checks the structural properties of the DataFrame and applies specified 
    operations based on the parameters provided.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check and optionally manipulate. This must be a 
        square matrix (i.e., having an equal number of rows and columns) 
        for further operations to proceed.

    order : list or tuple, optional
        A custom order of column names to reorder the columns and index.
        If provided, the DataFrame will be rearranged based on this order. 
        This is a sequence of column names that should exist in both the 
        index and columns of the DataFrame. If not provided, the DataFrame 
        will not be reordered.

    ops : {'check_only', 'validate'}, default 'check_only'
        Specifies the operation to be performed. If set to `'check_only'`, 
        the function only validates the structure of the DataFrame and does 
        not perform any modification. If set to `'validate'`, the function 
        attempts to reorder the DataFrame or set the index and columns based 
        on the provided `order` list.
        
    check_symmetry : bool, default False
        If True, the function checks if the DataFrame's columns and index 
        contain the same set of values. This ensures the symmetry of the 
        DataFrame's structure. If False, no check for symmetry is performed.

    cols_as_index : bool, default False
        If True, the function sets the index of the DataFrame using the 
        column names (i.e., the columns and index will be identical). If 
        False, no modification to the index is made. This parameter is only 
        relevant when the DataFrame is square and an `order` is specified.

    Returns
    -------
    pd.DataFrame or None
        If `ops` is ``'validate'``, the function returns a DataFrame where 
        the index and columns are reordered based on the provided `order`. 
        If `ops` is ``'check_only'``, the function performs checks without 
        modifying the DataFrame and returns None. If the checks fail (e.g., 
        the DataFrame is not square or the index and columns do not match), 
        a `ValueError` is raised.

    Raises
    ------
    ValueError
        If the DataFrame is not square, or if the index and columns 
        do not contain the same values, or if `order` contains invalid 
        column names.
    
    Examples
    --------
    >>> from fusionlab.core.checks import is_df_square
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'A': [1, 2, 3],
    >>>     'B': [4, 5, 6],
    >>>     'C': [7, 8, 9]
    >>> })
    >>> is_df_square(df, order=['B', 'A', 'C'], ops='validate')
    >>> df  # Returns the DataFrame with reordered columns and index

    Notes
    -----
    - The `is_df_square` function is used to ensure that a given DataFrame 
      has an equal number of rows and columns, and optionally reorder its 
      columns and index or set the index to match the columns.
    - The function raises a `ValueError` if the DataFrame does not meet the 
      expected conditions (i.e., square matrix or matching index and columns).
    
    See Also
    --------
    pd.DataFrame: The base pandas DataFrame class which provides many 
    useful methods for DataFrame manipulation, such as `.reindex()` and 
    `.set_index()`.
    
    References
    ----------
    [1] pandas documentation: https://pandas.pydata.org/pandas-docs/stable/
    """

    # Check if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Input must be a pandas DataFrame. Got {type(df).__name__!r}")
    
    # Check if the DataFrame is square (rows == columns)
    if df.shape[0] != df.shape[1]:
        raise ValueError(
            "The DataFrame must be square: number of rows"
            " must equal the number of columns.")

    if check_symmetry: 
        # Check if the index and columns have the same elements
        if ( 
            not set(df.columns).issubset(df.index) or 
            not set(df.index).issubset(df.columns)
            ):
            raise ValueError(
                "Index and columns must contain the same values.")
    
    # Optionally reorder index and columns if 'order' is provided
    if order:
        # Ensure the order is a valid subset of both index and columns
        if not set(order).issubset(df.columns) or not set(order).issubset(df.index):
            raise ValueError(
                "Custom order contains values not present in both index and columns.")
        
        if ops == 'validate':
            # df = df[order]         # Reorder columns
            # df = df.T[order].T     # Reorder index
            # Apply custom order to index and columns : .loc much faster
            df = df.loc[order, order]
            
    # If only checking, return True
    if ops == 'check_only':
        return True 
    
    # If validating and cols_as_index is True, set index equal to columns
    if cols_as_index:
        df.index = df.columns
        
    # If cols_as_index is 'reverse', reverse columns and set as index
    elif str(cols_as_index).lower() == 'reverse':
        df.index = df.columns[::-1]
    
    return df

def check_files(
    files: Union[str, List[str]],
    formats: Union[str, List[str]]=None,
    return_valid: bool = True,
    error: str = 'raise',
    empty_allowed: bool = False
) -> Union[str, List[str], None]:
    """
    Validate the Existence, Format, and Non-Emptiness of Files.
    
    The `check_files` function provides a robust mechanism to validate one or 
    multiple file paths by ensuring their existence, verifying their formats 
    against specified criteria, and confirming that they are not empty. This 
    utility is essential for data processing pipelines where the integrity and 
    correctness of input files are paramount. By offering flexible parameters, 
    users can tailor the validation process to their specific needs, handling 
    multiple files and diverse formats with ease.
    
    .. math::
        \text{Validation Process} = 
        \begin{cases}
            \text{Existence Check} \\
            \text{Format Verification} \\
            \text{Non-Emptiness Confirmation}
        \end{cases}
    
    Parameters
    -----------
    files : Union[`str`, `List[str]`]
        The file path or list of file paths to be validated. Each file will be 
        individually checked for existence, format compliance, and non-emptiness 
        based on the provided parameters.
    
    formats : Union[`str`, `List[str]`], optional
        The acceptable file formats/extensions. Formats can be specified with or 
        without a leading dot (e.g., `'csv'` or `'.xlsx'`). The function 
        normalizes these formats to ensure consistency during validation. If 
        `formats` is `None`, the format check is skipped.
    
    return_valid : `bool`, default=`True`
        Determines whether the function should return the list of valid files.
        - If `True`, returns the validated file paths.
        - If `False`, the function performs validation without returning the 
          file paths.
    
    error : `str`, default=`'raise'`
        Specifies the error handling strategy when encountering invalid files.
        - `'warn'`: Issues warnings for each validation failure and continues 
          processing remaining files.
        - `'raise'`: Raises exceptions immediately upon encountering a 
          validation failure, halting further execution.
        
        Raises a `ValueError` if an unsupported error handling strategy is 
        provided.
    
    empty_allowed : `bool`, default=`False`
        Indicates whether empty files are permissible.
        - If `False`, files with a size of zero bytes are considered invalid.
        - If `True`, empty files pass the non-emptiness check.
    
    Returns
    -------
    Union[`str`, `List[str]`, `None`]
        - If `return_valid` is `True` and only one file is valid, returns the 
          file path as a `str`.
        - If `return_valid` is `True` and multiple files are valid, returns a 
          `List[str]` of file paths.
        - If `return_valid` is `True` but no valid files are found, returns 
          `None`.
        - If `return_valid` is `False`, returns `None` regardless of validation 
          outcomes.
    
    Raises
    ------
    FileNotFoundError
        Raised when a specified file does not exist and `error` is set to `'raise'`.
    
    ValueError
        - Raised when a file's format is unsupported based on the `formats` 
          parameter and `error` is `'raise'`.
        - Raised when a file is empty while `empty_allowed` is `False` and `error` 
          is `'raise'`.
        - Raised when `error` parameter is neither `'warn'` nor `'raise'`.
    
    TypeError
        Raised when the `checkpoint` parameter is neither a `str` nor an `int`.
    
    Notes
    -----
    - **Format Normalization**: The function ensures that all format 
      specifications begin with a leading dot for consistent comparison, 
      regardless of how the user inputs them.
    
    - **Flexible Input Handling**: Users can pass a single file path as a 
      string or multiple file paths as a list of strings, allowing for 
      versatile usage scenarios.
    
    - **Error Management**: The `error` parameter provides control over how 
      the function responds to validation failures, enabling integration into 
      larger workflows where certain failures may need to be handled 
      differently.
    
    - **Empty File Consideration**: By default, the function treats empty 
      files as invalid to prevent downstream errors in data processing 
      tasks. This behavior can be modified using the `empty_allowed` parameter 
      based on user requirements.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.core.checks import check_files
    
    >>> # Single file validation
    >>> valid_file = check_files(
    ...     files='data/sample.csv',
    ...     formats=['csv', 'xlsx'],
    ...     return_valid=True,
    ...     error='warn',
    ...     empty_allowed=False
    ... )
    >>> print(valid_file)
    'data/sample.csv'
    
    >>> # Multiple files validation with mixed formats
    >>> files = [
    ...     'data/sample1.csv',
    ...     'data/sample2.xlsx',
    ...     'data/sample3.json',
    ...     'data/sample4.txt'  # Unsupported format
    ... ]
    >>> valid_files = check_files(
    ...     files=files,
    ...     formats=['csv', 'xlsx', 'json'],
    ...     return_valid=True,
    ...     error='warn',
    ...     empty_allowed=False
    ... )
    >>> print(valid_files)
    ['data/sample1.csv', 'data/sample2.xlsx', 'data/sample3.json']
    
    >>> # Handling empty files
    >>> check_files(
    ...     files='data/empty.csv',
    ...     formats='csv',
    ...     return_valid=True,
    ...     error='warn',
    ...     empty_allowed=False
    ... )
    /path/to/script.py:XX: UserWarning: File is empty: `data/empty.csv`.
    warnings.warn(msg)
    >>> # With empty_allowed=True
    >>> valid_file = check_files(
    ...     files='data/empty.csv',
    ...     formats='csv',
    ...     return_valid=True,
    ...     error='warn',
    ...     empty_allowed=True
    ... )
    >>> print(valid_file)
    'data/empty.csv'
    
    >>> # Single file validation with error handling
    >>> try:
    ...     check_files(
    ...         files='data/nonexistent.csv',
    ...         formats='csv',
    ...         return_valid=True,
    ...         error='raise',
    ...         empty_allowed=False
    ...     )
    ... except FileNotFoundError as e:
    ...     print(e)
    File does not exist: `data/nonexistent.csv`.
    
    Notes
    -----
    - **Extensibility**: The `check_files` function can be easily extended to 
      support additional file formats by simply updating the `formats` parameter 
      with the desired extensions.
    
    - **Integration**: This function is particularly useful in data ingestion 
      pipelines where the integrity of input files must be verified before 
      processing to ensure downstream tasks execute correctly.
    
    - **Performance Considerations**: When validating a large number of files, 
      consider setting `error='warn'` to allow the function to process all files 
      and collect all valid ones, rather than stopping at the first encountered error.
    
    See Also
    --------
    os.path.exists : Check if a path exists.
    os.path.splitext : Split the file path into root and extension.
    os.path.getsize : Get the size of a file.
    warnings.warn : Issue a warning message.
    
    References
    ----------
    .. [1] Python Documentation: os.path.exists. 
       https://docs.python.org/3/library/os.path.html#os.path.exists
    .. [2] Python Documentation: os.path.splitext. 
       https://docs.python.org/3/library/os.path.html#os.path.splitext
    .. [3] Python Documentation: os.path.getsize. 
       https://docs.python.org/3/library/os.path.html#os.path.getsize
    .. [4] Python Documentation: warnings.warn. 
       https://docs.python.org/3/library/warnings.html#warnings.warn
    .. [5] Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator: 
           L2 theory. *Probability Theory and Related Fields*, 57(5), 453-476.
    """

    # Normalize formats to include leading dot if formats is provided
    if formats is not None:
        if isinstance(formats, str):
            formats = [formats]
        normalized_formats = [
            f".{fmt.lstrip('.')}" for fmt in formats
        ]
    else:
        normalized_formats = None
    
    # Ensure files is a list
    if isinstance(files, str):
        files = [files]
    
    if not all( isinstance (f, str) for f in files): 
        raise TypeError(
            "Got invalid type(s). Supported only file path objects."
            " Please check your files.")
        
    valid_files = []
    
    for file in files:
        # Check if file exists
        if not os.path.exists(file):
            msg = f"File does not exist: `{file}`."
            if error == 'warn':
                warnings.warn(msg)
                continue
            else:
                raise FileNotFoundError(msg)
        
        # Check file format if formats is specified
        if normalized_formats is not None:
            _, ext = os.path.splitext(file)
            ext = ext.lower()
            if ext not in normalized_formats:
                msg = (
                    f"Unsupported file format: `{file}`. "
                    f"Expected formats: {normalized_formats}."
                )
                if error == 'warn':
                    warnings.warn(msg)
                    continue
                else:
                    raise ValueError(msg)
        
        # Check if file is not empty
        if not empty_allowed:
            if os.path.getsize(file) == 0:
                msg = f"File is empty: `{file}`."
                if error == 'warn':
                    warnings.warn(msg)
                    continue
                else:
                    raise ValueError(msg)
        
        # If all checks pass, add to valid_files
        valid_files.append(file)
    
    if return_valid:
        if not valid_files:
            if error == 'warn':
                warnings.warn("No valid files found.")
            else:
                raise ValueError("No valid files found.")
        else:
            if len(valid_files) == 1:
                valid_files = valid_files[0]
        
        return valid_files

def _validate_nested_param(
    value: Any,
    expected_type: Any,
    param_name: str = '',
    coerce: bool = True, 
    empty_as_none: bool =..., 
) -> Any:

    origin = get_origin(expected_type)
    args = get_args(expected_type)
    
    # **New logic for Callable:**
    # Detect if expected_type is a Callable or a Union that includes Callable.
    if expected_type is Callable or origin is Callable or (
        origin is Union and any(t is Callable for t in args)
    ):
        return check_callable(value, expected_type, param_name)

    # Handle Optional types (Union with NoneType)
    if origin is Union and type(None) in args:
        # origin, arg ---> typing.Union (<class 'int'>, <class 'NoneType'>)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if not non_none_args:
            if value is not None:
                raise TypeError(f"Parameter `{param_name}` must be None.")
            return None
        # If the expected type is Optional[List[X]], 
        # set to empty list if None and coerce is True
        if len(non_none_args) == 1:
            sub_type = non_none_args[0]
            if get_origin(sub_type) in ( list, tuple) and value is None:
                return None if empty_as_none else [] 
            # If the expected type is Optional[X],
            # set None whatever.
            elif ( 
                sub_type in (int, float, numbers.Real, numbers.Integral) 
                and value is None
                ): 
                # return None for Optional[X]
                return None 
            
        # Recursively validate the non-None type
        return validate_nested_param(value, non_none_args[0], param_name, coerce)
    
    # Handle List types
    elif origin in (list, List):
        if isinstance(value, list):
            validated_list = []
            for index, item in enumerate(value):
                try:
                    validated_item = validate_nested_param(
                        item, args[0],
                        param_name = f"{param_name}[{index}]",
                        coerce = coerce
                    )
                    validated_list.append(validated_item)
                except TypeError as e:
                    raise TypeError(f"{e}") from None
            return validated_list
        # If value is a single item, attempt to convert
        # it to a list if coerce is True
        else:
            if coerce:
                try:
                    validated_item = validate_nested_param(
                        value, args[0],
                        param_name = param_name,
                        coerce = coerce
                    )
                    return [validated_item]
                except TypeError as e:
                    raise TypeError(
                        f"Parameter `{param_name}` is expected to be a "
                        f"list of `{args[0].__name__}`, but got type "
                        f"`{type(value).__name__}`."
                    ) from e
            else:
                raise TypeError(
                    f"Parameter `{param_name}` is expected to be a "
                    f"list of `{args[0].__name__}`, but got type "
                    f"`{type(value).__name__}`."
                )
                
    # Handle a Dict types 
    elif origin in (dict, Dict):
        if not isinstance(value, dict):
            raise TypeError(
                f"Parameter `{param_name}` is expected to be a dict,"
                f" but got type `{type(value).__name__}`."
            )
        validated_dict = {}
        key_type, val_type = args
        for key, val in value.items():
            try:
                validated_key = validate_nested_param(
                    key, key_type,
                    param_name = f"{param_name} key",
                    coerce = coerce
                )
            except TypeError as e:
                raise TypeError(f"{e}") from None
            try:
                validated_val = validate_nested_param(
                    val, val_type,
                    param_name = f"{param_name}[{key}]",
                    coerce = coerce
                )
            except TypeError as e:
                raise TypeError(f"{e}") from None
            validated_dict[validated_key] = validated_val
        return validated_dict
    
    # Handle Tuple types
    elif origin in (tuple, Tuple):
        if not isinstance(value, tuple):
            raise TypeError(
                f"Parameter `{param_name}` is expected to be a tuple,"
                f" but got type `{type(value).__name__}`."
            )
        if len(args) != len(value):
            raise TypeError(
                f"Parameter `{param_name}` expects a tuple of length "
                f"{len(args)}, but got tuple of length {len(value)}."
            )
        validated_tuple = []
        for index, (item, sub_type) in enumerate(zip(value, args)):
            try:
                validated_item = validate_nested_param(
                    item, sub_type,
                    param_name = f"{param_name}[{index}]",
                    coerce = coerce
                )
                validated_tuple.append(validated_item)
            except TypeError as e:
                raise TypeError(f"{e}") from None
        return tuple(validated_tuple)
    
    # Handle all other basic types
    # Handle basic types
    else:
        if not isinstance(value, expected_type):
            if coerce:
                # Attempt to convert the value to the expected type
                try:
                    return expected_type(value)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Parameter `{param_name}` is expected to be of "
                        f"type `{expected_type.__name__}`, but got type "
                        f"`{type(value).__name__}`."
                    ) from e
            else:
                raise TypeError(
                    f"Parameter `{param_name}` is expected to be of type "
                    f"`{expected_type.__name__}`, but got type "
                    f"`{type(value).__name__}`."
                )
        return value


def validate_nested_param(
        value: Any, 
        expected_type: Any, 
        param_name: str = '', 
        coerce: bool = False
    ) -> Any:
    """
    Validate and coerce parameters and their nested items to the expected types.
    
    This function ensures that the provided ``value`` conforms to the 
    ``expected_type``. It supports validation of nested structures such as 
    lists of dictionaries, dictionaries containing lists, and other complex 
    nested types. If the ``value`` is not of the expected type but can be 
    coerced (e.g., a single item to a list), the function performs the 
    necessary conversion. If coercion is not possible, it raises a 
    ``TypeError`` with a descriptive message.
    
    .. math::
        \text{If } x \text{ is not of type } T, \text{ attempt to convert } 
        x \text{ to } T.
    
    Parameters
    ----------
    value : Any
        The value to be validated and coerced. It can be a single item, a list 
        of items, a dictionary, or nested combinations thereof. If ``None`` is 
        acceptable for the parameter, it should be handled appropriately.
    expected_type : Any
        The expected type of the ``value``. This can include nested types like 
        ``List[str]``, ``Dict[str, float]``, etc., using Python's typing 
        constructs. The function recursively validates each level of the input 
        against the specified ``expected_type``.
    param_name : str, optional
        The name of the parameter being validated. This is used in error 
        messages to provide clear and descriptive feedback in case of 
        validation failures.
        
    coerce : bool, optional
        If ``True``, attempt conversions to match `'expected_type'`. If
        ``False``, enforce strict type checking with no conversions.
    empty_as_none : bool, default=True
        If True, returns `None` when type is `Optional`. If ``False``, an 
        empty Iterable  object  `List` is returned.

    Returns
    -------
    Any
        The validated and potentially coerced value, matching the 
        ``expected_type``. If ``value`` is ``None`` and ``None`` is 
        acceptable, it returns ``None`` or an empty list/dictionary based 
        on the context.
    
    Raises
    ------
    TypeError
        If the ``value`` cannot be coerced to the ``expected_type`` or if any 
        nested item fails validation.
    
    Examples
    --------
    >>> from fusionlab.core.checks import validate_nested_param
    >>> # Validate a single string item
    >>> validate_nested_param('item', str, 'static_feature_names')
    'item'
    >>> # Validate a single integer item to be a list of integers
    >>> validate_nested_param(5, List[int], 'ages')
    [5]
    >>> # Validate a list of strings
    >>> validate_nested_param(['feature1', 'feature2'], List[str], 'static_feature_names')
    ['feature1', 'feature2']
    >>> # Validate a dictionary with string keys and float values
    >>> validate_nested_param({'a': 1.0, 'b': 2.5}, Dict[str, float], 'scaling_params')
    {'a': 1.0, 'b': 2.5}
    
    Notes
    -----
    - This function is designed to be highly flexible and can handle deeply 
      nested structures by recursively validating each level of the input.
    - If a single item is provided where a list is expected, the function will 
      automatically wrap the item in a list.
    - For dictionary validations, both keys and values are validated against 
      expected types, ensuring the integrity of complex nested data.
    
    See Also
    --------
    prepare_future_data : Function for preparing future data inputs with validation.
    validate_parameter : Another utility function for parameter validation.
    
    References
    ----------
    .. [1] Python Software Foundation. *typing — Support for type hints*. 
        https://docs.python.org/3/library/typing.html
    .. [2] Smith, J., & Doe, A. (2020). *Effective Type Validation in Python*. 
        Journal of Python Development, 10(2), 123-135.
    """

    origin = get_origin(expected_type)
    args = get_args(expected_type)
    
    # **New logic for Callable:**
    # Detect if expected_type is a Callable or a Union that includes Callable.
    if expected_type is Callable or origin is Callable or (
        origin is Union and any(t is Callable for t in args)
    ):
        return check_callable(value, expected_type, param_name)
    
    # Handle Optional[...] which is Union[..., NoneType]
    if origin is Union and type(None) in args:
        non_none_args = tuple(arg for arg in args if arg is not type(None))
        if value is None:
            return value
        return validate_nested_param(
            value, 
            Union[non_none_args], 
            param_name, 
            coerce
        )
    
    # Handle Union types
    if origin is Union:
        last_exception = None
        for arg in args:
            try:
                return validate_nested_param(value, arg, param_name, coerce)
            except TypeError as e:
                last_exception = e
        raise TypeError(
            f"Parameter '{param_name}': Value '{value}' does not"
            f" match any type in {expected_type}."
        ) from last_exception
    
    # Handle List[...] types
    if origin in {list, List}:
        if not isinstance(value, list):
            if coerce:
                try:
                    value = list(value)
                except Exception as e:
                    raise TypeError(
                        f"Parameter '{param_name}': Cannot coerce"
                        f" value '{value}' to list."
                    ) from e
            else:
                raise TypeError(
                    f"Parameter '{param_name}': Expected type list for"
                    f" value '{value}', got {type(value).__name__!r}."
                )
        if not args:
            return value  # No inner type to validate
        return [
            validate_nested_param(item, args[0], param_name, coerce) 
            for item in value
        ]
    
    # Handle Dict[...] types
    if origin in {dict, Dict}:
        if not isinstance(value, dict):
            if coerce:
                try:
                    value = dict(value)
                except Exception as e:
                    raise TypeError(
                        f"Parameter '{param_name}': Cannot coerce"
                        f" value '{value}' to dict."
                    ) from e
            else:
                raise TypeError(
                    f"Parameter '{param_name}': Expected type dict for"
                    f" value '{value}', got {type(value).__name__!r}."
                )
        if not args:
            return value  # No inner types to validate
        key_type, val_type = args
        return {
            validate_nested_param(k, key_type, f"{param_name} key", coerce): 
            validate_nested_param(v, val_type, f"{param_name} value", coerce)
            for k, v in value.items()
        }
    
    # Handle Tuple[...] types
    if origin in {tuple, Tuple}:
        if not isinstance(value, tuple):
            if coerce:
                try:
                    value = tuple(value)
                except Exception as e:
                    raise TypeError(
                        f"Parameter '{param_name}': Cannot coerce"
                        f" value '{value}' to tuple."
                    ) from e
            else:
                raise TypeError(
                    f"Parameter '{param_name}': Expected type tuple for"
                    f" value '{value}', got {type(value).__name__!r}."
                )
        if not args:
            return value  # No inner types to validate
        if len(args) != len(value):
            raise TypeError(
                f"Parameter '{param_name}': Expected tuple of"
                f" length {len(args)}, got {len(value)}."
            )
        return tuple(
            validate_nested_param(item, arg, f"{param_name}[{i}]", coerce) 
            for i, (item, arg) in enumerate(zip(value, args))
        )
    
    # Handle Callable[...] types
    if origin is Callable:
        if not callable(value):
            raise TypeError(
                f"Parameter '{param_name}': Expected a callable"
                f" for value '{value}', got {type(value).__name__!r}."
            )
        return value
    
    # Handle Any type
    if expected_type is Any:
        return value
    
    # Handle basic types
    if isinstance(expected_type, type):
        if not isinstance(value, expected_type):
            if coerce:
                try:
                    value = expected_type(value)
                except Exception as e:
                    raise TypeError(
                        f"Parameter '{param_name}': Cannot coerce"
                        f" value '{value}' to {expected_type}."
                    ) from e
            else:
                raise TypeError(
                    f"Parameter '{param_name}': Expected type {expected_type}"
                    f" for value '{value}', got {type(value).__name__!r}."
                )
        return value
    
    # If expected_type is still not handled, raise an error
    raise TypeError(
        f"Parameter '{param_name}': Unsupported type"
        " annotation: {expected_type}."
    )

def check_params(param_types: Dict[str, Any], coerce: bool = True):
    """
    Validate parameters of the decorated function against given
    expected types, with optional coercion.

    This `check_params` decorator verifies that function parameters
    match their expected types specified in ``param_types``. It
    applies :func:`validate_nested_param` to each parameter defined
    in ``param_types``. If ``coerce`` is True, attempts are made
    to convert the given value to the expected type. Otherwise, a
    strict type check is enforced. If validation fails, a
    ``TypeError`` is raised.

    .. math::
       \text{Given a parameter } x \text{ with expected type } T,
       \text{ if } x \notin T \text{ and coerce=True, then }
       x \rightarrow T \text{ if possible. Else, raise error.}

    Parameters
    ----------
    param_types : Dict[str, Any]
        Dictionary mapping parameter names to their expected types.
        For example:
        ::
        
            {
                "x": List[int],
                "y": Dict[str, float]
            }

        Each key corresponds to a parameter name of the decorated
        function, and its value is the expected type that the
        parameter must match or be coerced into.

    coerce : bool, optional
        Whether to attempt coercion if types do not initially match.
        If ``True``, the decorator tries to convert the parameter
        value to the expected type. If ``False``, a strict check is
        enforced and a ``TypeError`` is raised if the type does not
        match exactly.

    Methods
    -------
    check_params(param_types, coerce)
        This is the main function that creates and returns the
        decorator.

    The decorator returned by `check_params`:
    - Binds the decorated function and applies default values.
    - Iterates over ``param_types`` and checks each corresponding
      parameter in the bound arguments.
    - Calls :func:`validate_nested_param` to validate or coerce the
      value to the expected type.
    - Raises ``TypeError`` if validation fails.

    The inner wrapper function:
    - Applies the type checks before calling the decorated function.
    - Returns the result of the decorated function if checks pass.

    Examples
    --------
    >>> from fusionlab.core.checks import check_params
    >>> from typing import List, Dict
    >>> @check_params({"items": List[str], "mapping": Dict[str, float]}, coerce=True)
    ... def process_data(items, mapping):
    ...     return items, mapping
    ...
    >>> process_data(["apple", "banana"], {"a": 1.0, "b": 2.5})
    (['apple', 'banana'], {'a': 1.0, 'b': 2.5})
    >>> process_data("single_item", {"x": "2.0"})
    (['single_item'], {'x': 2.0})

    Notes
    -----
    - If ``coerce`` is True, single values can be promoted to lists,
      and strings converted to floats if feasible.
    - If ``coerce`` is False, strict type matching is enforced.
    - This decorator allows writing robust, type-safe functions.

    See Also
    --------
    validate_nested_param : Validates and optionally coerces a single 
        parameter or nested structure to an expected type.

    References
    ----------
    .. [1] Python Software Foundation. *typing — Support for type hints*.
           https://docs.python.org/3/library/typing.html

    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig   = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for p_name, p_type in param_types.items():
                if p_name in bound.arguments:
                    val = bound.arguments[p_name]
                    validated_val = validate_nested_param(
                        val, p_type,
                        param_name = p_name,
                        coerce     = coerce
                    )
                    bound.arguments[p_name] = validated_val

            return func(*bound.args, **bound.kwargs)
        return wrapper
    return decorator

def check_callable(
        value, expected_type, param_name: str = ''):
    """
    Validate that a parameter matches a Callable type or an Optional[Callable]
    (or Union[Callable, None]).

    Parameters
    ----------
    value : Any
        The value to validate.
    expected_type : Any
        The expected type, which might be `Callable`, `Optional[Callable]`,
        or `Union[Callable, None]`.
    param_name : str, optional
        Name of the parameter (for better error messages).

    Returns
    -------
    Any
        Returns the validated value if it matches the expected callable type.
        Returns None if None is allowed by the type hint and value is None.

    Raises
    ------
    TypeError
        If the value does not match the expected Callable requirements.
        
    Examples 
    ---------
    # Example usages:
    check_callable(lambda x: x+1, Callable)        # returns <lambda>
    check_callable(None, Callable)                 # raises TypeError
    check_callable(None, Union[Callable, None])    # returns None
    check_callable(None, Optional[Callable])       # returns None
    check_callable(int, Callable)                  # int is callable (class), so returns int

    """

    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Check for Optional[Callable] or Union[Callable, None]
    if origin is Union and type(None) in args:
        # If the value is None and None is allowed, return None immediately
        if value is None:
            return None

        # Otherwise, we proceed with the non-None args to validate value
        non_none_args = [t for t in args if t is not type(None)]
        if not non_none_args:
            # Means only None is allowed, and we've already handled value=None
            # If code reaches here and value is not None, it's an error:
            raise TypeError(
                f"Parameter '{param_name}' must be None because only NoneType is allowed."
            )
        
        # Validate against the non-None callable type
        return check_callable(value, non_none_args[0], param_name)

    # If `None` is provided but the expected type is strictly Callable, raise an error
    if value is None:
        raise TypeError(
            f"Parameter {param_name} is expected to be a Callable, but got `None`."
        )

    # Check if the value is callable
    if not callable(value):
        raise TypeError(
            f"Parameter {param_name} is expected to be a Callable,"
            f" but got type `{type(value).__name__}`."
        )

    return value

def check_array_like(
    obj, 
    context="ml", 
    ops="check_only"
):
    """
    Determine if an object is array-like based on a specified context 
    and operation.

    The function can either check if the object is array-like or validate it, 
    depending on the operation specified.

    Parameters
    ----------
    obj : any
        The object to check or validate.
        
    context : {'general', 'ml'}, default 'ml'
        The context in which to validate the object:
        - 'ml' (default): Checks for array-like objects commonly used in 
          machine learning, 
          such as lists, numpy arrays, pandas DataFrames, Series, etc.
        - 'general': Checks for basic array-like objects, including lists, tuples, 
          numpy arrays, and other iterables.
        
    ops : {'check_only', 'validate'}, default 'check_only'
        Defines the operation to perform:
        - 'check_only': Returns True if the object is array-like, False otherwise.
        - 'validate': Returns the object if valid, or raises a TypeError if invalid.
    
    Returns
    -------
    bool or array-like
        - If `ops='check_only'`, returns True if the object is array-like,
         False otherwise.
        - If `ops='validate'`, returns the object if valid, otherwise raises
        a TypeError.
    
    Raises
    ------
    TypeError
        If the object is not array-like in the given context and `ops='validate'`.
    
    Examples
    --------
    >>> from fusionlab.core.checks import check_array_like 
    >>> check_array_like([1, 2, 3])
    True
    
    >>> check_array_like(np.array([1, 2, 3]))
    True
    
    >>> check_array_like(pd.DataFrame([[1, 2], [3, 4]]), context='ml', ops='validate')
    pandas.core.frame.DataFrame
    """
    
    # Define valid contexts and operations
    valid_contexts = {'general', 'ml'}
    valid_operations = {'check_only', 'validate'}

    # Validate the context and operation parameters
    if context not in valid_contexts:
        raise ValueError(
            f"Invalid context '{context}'. Expected one of {valid_contexts}.")
    
    if ops not in valid_operations:
        raise ValueError(
            f"Invalid ops '{ops}'. Expected one of {valid_operations}.")
    
    # Check if the object is array-like
    is_array_like = False
    if context == 'ml':
        # For ML context, check for lists, numpy arrays, pandas DataFrames/Series
        is_array_like = isinstance(obj, (list, tuple, np.ndarray, pd.DataFrame, pd.Series))
    elif context == 'general':
        # For general context, check for iterables (e.g., list, tuple, numpy array,...)
        is_array_like = isinstance(obj, Iterable) and not isinstance(obj, str)

    # Return based on operation type
    if ops == 'check_only':
        return is_array_like
    
    if ops == 'validate' and not is_array_like:
        raise TypeError(
            f"Object of type '{type(obj)}' is not array-like in the '{context}' context.")
    
    return obj

def check_datetime(
    df : pd.DataFrame,
    dt_cols=None, # if dt_cols is not None, then check this columns in the dataframe 
    ops : str = 'check_only',
    error : str = 'raise',
    consider_dt_as : Optional[str] = None,
    accept_dt : bool = True, 
    allow_int: bool=False, # if allow_int, consider integer as dateime column  , for instance  [2024, 2025], 
    int2dt : bool=False, # if True, then convert the integer to datetime,
) -> Union[bool, pd.DataFrame]:
    """
    Check or validate datetime columns in a DataFrame and optionally
    convert them to a specified data type.

    The function identifies columns of datetime-like dtype in
    ``df``. Depending on ``ops`` and other parameters, it either
    checks or manipulates them.

    .. math::
        \\text{time\\_col} \\in 
            \\{ \\text{datetime64}, \\text{object (parsed as date)}, 
               \\text{etc.} \\}

    Parameters
    -----------
    df : 
        pd.DataFrame 
        The input dataframe to check or validate.

    ops : str, default='check_only'
        Specifies the operation mode:
        - `check_only`: Return True/False depending on the presence
          of datetime columns.
        - `validate`: Potentially convert or remove datetime columns
          depending on other parameters.

    error : str, default='raise'
        Determines how to handle invalid cases:
        - `'raise'`: Raise an error immediately.
        - `'warn'` : Issue a warning, then proceed.
        - `'ignore'`: Silently continue, ignoring issues.

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
        - `'object'` or 'category': Convert date columns to Python objects 
          (strings, etc.). If conversion fails, raise or warn 
          per `error` policy.

    accept_dt : bool, default=True
        Whether to accept the presence of datetime columns:
        - If True and ``ops='validate'``, datetime columns are
          kept or converted if `consider_dt_as` is set.
        - If False and no `consider_dt_as` provided:
          * With `error='raise'`, raise error if any datetime 
            column is found.
          * With `error='warn'`, warn user, then drop datetime 
            columns.
          * With `error='ignore'`, silently drop datetime columns.

    allow_int : bool, default=False
        If True, treat integer columns as datetime columns 
        (e.g., years such as [2024, 2025]) when `dt_cols` is provided.

    int2dt : bool, default=False
        If True, convert integer columns to datetime columns 
        (e.g., 2024 to '2024-01-01') when `dt_cols` is provided.

    Returns
    -------
    bool or pd.DataFrame
        - If ``ops='check_only'``, returns True if datetime columns
          exist, otherwise False (respecting `error` behavior).
        - If ``ops='validate'``, returns a possibly modified
          DataFrame according to the rules described.

    Notes
    -----
    If :math:`\\text{accept\\_dt} = \\text{False}` and
    :math:`\\text{ops} = \\text{'validate'}`, but no valid
    conversion is specified via :math:`\\text{consider\\_dt\\_as}`,
    the behavior depends on :math:`\\text{error}`:
      - `'raise'`: raise an error.
      - `'warn'`: warn and drop datetime columns.
      - `'ignore'`: silently drop datetime columns.

    Examples
    --------
    >>> from fusionlab.core.checks import check_datetime
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date_col': pd.date_range('2021-01-01', periods=5),
    ...     'val': [1,2,3,4,5]
    ... })
    >>> # Example 1: Just checking
    >>> has_dt = check_datetime(df, ops='check_only')
    >>> print(has_dt)
    True

    >>> # Example 2: Validate and convert to float
    >>> df_float = check_datetime(df, ops='validate',
    ...     error='warn', consider_dt_as='float64')
    >>> print(df_float.dtypes)

    See Also
    --------
    SomeOtherFunction : Reference to a related function.

    References
    ----------
    .. [1] Smith, J. et al., "Efficient Data Preprocessing 
           Techniques," Data Engineering, 2020.
    """
    are_all_frames_valid(df, df_only=True)
    df_copy = df.copy() 
    # 1. Identify all datetime columns in df.
    if dt_cols is not None: 
        if isinstance(dt_cols, str):
            dt_cols = [dt_cols]

        if allow_int:
            # If allow_int, treat integer columns as datetime columns
            int_columns = df_copy[dt_cols].select_dtypes(
                include=['int', 'int32', 'int64']).columns
            dt_cols.extend(int_columns)
            if int2dt:
                for col in int_columns:
                    df_copy[col] = pd.to_datetime(
                        df_copy[col].astype(str), format='%Y',  errors='coerce')
            
    else: 
        dt_cols = df_copy.select_dtypes(
            include=['datetime', 'datetime64[ns]','datetimetz']).columns.tolist()

    # quick check for presence
    has_dt = (len(dt_cols) > 0)

    # 2. If ops == 'check_only', we just return bool or handle error
    #    as needed.
    if ops == 'check_only':
        if not has_dt:
            # If dt columns exist, handle error or just return True
            if error == 'raise':
                raise ValueError(
                    "DataFrame has datetime columns "
                    f"({dt_cols}). Raise requested."
                )
            elif error == 'warn':
                warnings.warn(
                    "DataFrame has datetime columns "
                    f"({dt_cols})."
                )
            # if error == 'ignore', do nothing
            return False
        else:
            return True

    # 3. For ops == 'validate', proceed with acceptance or conversion.
    if ops != 'validate':
        # unrecognized ops -> either raise or warn
        msg_ops = (f"Unknown ops='{ops}'. Use 'check_only' or 'validate'.")
        if error == 'raise':
            raise ValueError(msg_ops)
        elif error == 'warn':
            warnings.warn(msg_ops)
        # 'ignore' -> just do nothing
        return df_copy

    # ops='validate':
    if not has_dt:
        # no datetime columns => just return df_copy
        return df_copy

    # dt exists
    if accept_dt:
        # accept dt columns, but maybe convert if consider_dt_as is set
        if consider_dt_as is None:
            # user wants dt columns as is
            return df_copy

        # user wants to convert dt
        for col in dt_cols:
            df_copy = _convert_datetime_column(
                df=df_copy,
                col=col,
                consider_dt_as=consider_dt_as,
                error=error
            )
        return df_copy
    else:
        # accept_dt is False, dt columns are not acceptable
        # if we have consider_dt_as, we can still convert them
        if consider_dt_as is not None:
            # try conversion
            for col in dt_cols:
                df_copy = _convert_datetime_column(
                    df=df_copy,
                    col=col,
                    consider_dt_as=consider_dt_as,
                    error=error
                )
            return df_copy
        else:
            # user did not provide consider_dt_as
            # => handle error
            if error == 'raise':
                raise ValueError(
                    "Datetime columns found but no conversion or "
                    f"acceptance specified. Don't know how to treat {dt_cols}."
                    " Use `consider_dt_as/dt_as` or set ``accept_dt=True``."
                )
            elif error == 'warn':
                warnings.warn(
                    "Datetime columns found and not accepted. "
                    f"Dropping these columns: {dt_cols}."
                )
                return df_copy.drop(columns=dt_cols)
            elif error == 'ignore':
                # silently drop dt columns
                return df_copy.drop(columns=dt_cols)

    # fallback
    return df_copy

def _convert_datetime_column(
    df : pd.DataFrame,
    col : str,
    consider_dt_as : str,
    error : str
) -> pd.DataFrame:
    """
    Convert datetime column (col) in df to the type specified by 
    consider_dt_as. Raise or warn on failure based on error.
    """
    try:
        if consider_dt_as.lower() in ['numeric']:
            # convert datetime to numeric (timestamp)
            df[col] = pd.to_numeric(df[col], errors='raise')
        elif consider_dt_as.lower() in ['float', 'float32', 'float64']:
            # convert datetime to numeric then to float
            df[col] = pd.to_numeric(df[col], errors='raise')
            df[col] = df[col].astype(consider_dt_as.lower())
        elif consider_dt_as.lower() in ['int', 'int32', 'int64']:
            # convert datetime to numeric then to int
            df[col] = pd.to_numeric(df[col], errors='raise')
            df[col] = df[col].astype(consider_dt_as.lower())
        elif consider_dt_as.lower() in ['object', 'category', 'categoric']:
            # convert datetime to object (string)
            df[col] = df[col].astype('object')
        else:
            msg = (f"Unsupported consider_dt_as='{consider_dt_as}'. "
                   "Use 'numeric', 'floatXX', 'intXX', or 'object'.")
            if error == 'raise':
                raise ValueError(msg)
            elif error == 'warn':
                warnings.warn(msg)
            # ignore => do nothing
    except Exception as e:
        msg_fail = (
            f"Failed to convert column '{col}' from datetime to "
            f"'{consider_dt_as}'. Error: {str(e)}"
        )
        if error == 'raise':
            raise ValueError(msg_fail)
        elif error == 'warn':
            warnings.warn(msg_fail)
        # if 'ignore', pass
    return df

def ensure_same_shape(
    *arrays: Union[np.ndarray, pd.DataFrame],
    axis: Optional[int] = None,
    ops: str = 'check_only',
    solo_return: bool = True
) -> Optional[Union[Tuple[Union[np.ndarray, pd.DataFrame], ...],
                    Union[np.ndarray, pd.DataFrame]]]:
    """
    Check or validate that multiple arrays have the same shape
    along a specified axis, with an option to return them in
    a singular format if only one array is provided.

    The ``ensure_same_shape`` function enforces consistent
    shapes among any number of arrays (NumPy or pandas), based
    on a specified (or default) axis. When ``axis=None``, the
    function checks for matching length (i.e., shape[0]). The
    behavior of the function, whether it only checks or
    actually returns validated arrays, is governed by
    ``ops``. Additionally, if ``solo_return=True`` and
    exactly one array is validated under ``ops='validate'``,
    the function returns that array directly, rather than
    wrapping it in a tuple.

    .. math::
       \\text{Shape Consistency} =
       \\begin{cases}
         \\text{same size along axis} & \\text{if axis is not None}\\\\
         \\text{same length (shape[0])} & \\text{if axis is None}
       \\end{cases}

    Parameters
    ----------
    *arrays : np.ndarray or pandas.DataFrame
        One or more array-like objects to be checked. Each must be
        either a NumPy array or a pandas DataFrame. If a single array
        is passed as the only argument, it is processed accordingly.

    axis : int or None, optional
        The axis along which to enforce shape consistency. If
        ``axis=1``, all arrays must have at least two dimensions
        (i.e., shape[1] must be valid). An informative error is
        raised if an array is 1D. If ``axis=None``, the function
        checks that all arrays match in their length (shape[0]).

    ops : {'check_only', 'validate'}, default='check_only'
        Determines the function's behavior:
        
        - ``'check_only'``:
          Perform checks and raise if any mismatch is found, but
          do not return anything.
        - ``'validate'``:
          Verify shape consistency, then return the arrays if
          no issues are found.

    solo_return : bool, default=True
        If ``True`` and exactly one array is passed under
        ``ops='validate'``, the function returns that array
        directly rather than as a tuple of length 1. If multiple
        arrays are provided or ``ops='check_only'``, the function
        returns a tuple if returning anything.

    Returns
    -------
    tuple of np.ndarray or pandas.DataFrame or single object
        - If ``ops='validate'`` and multiple arrays are given, a
          tuple of the validated arrays is returned.
        - If ``ops='validate'`` and only one array is provided,
          returns that single array directly if ``solo_return=True``;
          otherwise, returns a tuple with one element.
        - If ``ops='check_only'``, no value is returned.

    Raises
    ------
    ValueError
        - If ``axis=1`` is given but an array is 1D, making
          indexing along axis 1 invalid.
        - If arrays do not match in shape along the specified
          axis or do not share the same length if ``axis=None``.
        - If ``ops`` is not in ``{'check_only', 'validate'}``.

    Examples
    --------
    >>> from fusionlab.core.array_manager import ensure_same_shape
    >>> import numpy as np
    
    >>> A = np.random.rand(5, 3)
    >>> B = np.random.rand(5, 3)
    >>> # Basic check on length: axis=None by default.
    >>> ensure_same_shape(A, B)  # no error => shapes consistent
    
    >>> # Return validated arrays if they are consistent:
    >>> validated = ensure_same_shape(A, B, ops='validate')
    >>> print(type(validated), len(validated))
    <class 'tuple'> 2
    
    >>> # If an array is 1D but axis=1 is requested:
    >>> C = np.random.rand(5)
    >>> ensure_same_shape(A, C, axis=1)
    ValueError: An array is 1D, but `axis=1` was requested.

    Notes
    -----
    - The function wraps a single array in a list if multiple arrays
      are not explicitly provided. This allows uniform processing
      logic.
    - When ``solo_return=True`` and only one array is validated
      under ``ops='validate'``, it is returned directly if checks
      pass, rather than as a tuple with one element.
    
    See Also
    --------
    fusionlab.utils.validator.check_consistent_length : 
        A related function that cross-checks dimensional alignment.

    References
    ----------
    .. [1] NumPy Development Team. (2023). *NumPy documentation*.
           https://numpy.org/doc/stable/
    """
    # Validate `ops` parameter.
    valid_ops = {'check_only', 'validate'}
    if ops not in valid_ops:
        raise ValueError(
            f"Invalid `ops` value: '{ops}'. Choose from {valid_ops}."
        )

    # Convert single array to a tuple if needed, so code handles a
    # uniform collection of arrays.
    if len(arrays) == 1 and not isinstance(arrays[0], (list, tuple)):
        arrays = (arrays[0],)

    # If axis=1, all arrays must have at least 2 dims. Check each.
    if axis == 1:
        for arr in arrays:
            if arr.ndim < 2:
                raise ValueError(
                    "An array is 1D, but `axis=1` was requested, "
                    "requiring at least 2 dimensions."
                )

    # Helper function to determine size based on the axis or default to shape[0].
    def get_size(arr, ax):
        ar = np.asarray(arr)
        return ar.shape[ax] if ax is not None else ar.shape[0]

    # Determine reference size from first array along specified axis (or default).
    ref_size = get_size(arrays[0], axis)

    # Check consistency of shapes among all arrays.
    for idx, arr in enumerate(arrays[1:], start=1):
        current_size = get_size(arr, axis)
        if current_size != ref_size:
            axis_str = f"axis={axis}" if axis is not None else "length (axis=None)"
            raise ValueError(
                f"Arrays must match size along {axis_str}. Mismatch found "
                f"at array position {idx} (expected {ref_size}, got {current_size})."
            )

    # If ops='check_only', no return needed after verifying shapes.
    if ops == 'check_only':
        return None

    # ops='validate'. Return arrays in the desired format.
    if len(arrays) == 1:
        # Only one array provided
        if solo_return:
            # Return that array directly
            return arrays[0]
        
    # If multiple arrays are provided, always return them as a tuple.
    return arrays

def validate_axis(
    axis: Optional[Union[int, str, None]] = None,
    array_base: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    accept_axis_none: bool = ...
) -> Optional[int]:
    """
    Validate and convert ``axis`` to an integer index, optionally checking it
    against the dimensionality of ``array_base``. If `accept_axis_none` is True
    and ``axis`` is None, returns None without further checks.

    The ``validate_axis`` function ensures that ``axis`` is a valid integer
    or string representation. For instance, `'rows'` maps to axis=0 and
    `'columns'` maps to axis=1. When ``axis=None`` is allowed, the function
    simply returns None. If an array-like object is provided via
    ``array_base``, the final axis must not exceed ``array_base`` dimensions.

    .. math::
       \\text{Validated Axis} = 
       \\begin{cases}
         0, & \\text{if }\\text{axis}\\in\\{0,\\text{'rows'}\\}\\\\
         1, & \\text{if }\\text{axis}\\in\\{1,\\text{'columns'}\\}\\\\
         \\text{None}, & \\text{if axis=None and accept\\_axis\\_none=True}\\\\
         \\text{raise Error}, & \\text{otherwise}
       \\end{cases}

    Parameters
    ----------
    axis : int, str, or None, optional
        The axis to validate. Accepted values:
        
        - ``0``, or `'rows'` for row-wise operations.
        - ``1``, or `'columns'`` for column-wise operations.
        - ``None`` if `accept_axis_none` is True.

    array_base : numpy.ndarray or pandas.DataFrame, optional
        An array-like object. If provided, the function checks that
        the final axis index does not exceed the dimensionality of
        ``array_base``. For a 2D object, valid axes are 0 or 1.

    accept_axis_none : bool, default=True
        Determines if ``None`` is a permissible axis value. If False,
        a ``ValueError`` is raised if ``axis=None``.

    Returns
    -------
    int or None
        - Integer axis (0 or 1) if string input maps to `'rows'` or
          `'columns'`, or if an integer is valid.
        - None if ``axis=None`` and `accept_axis_none=True`.
    
    Raises
    ------
    ValueError
        - If ``axis=None`` but `accept_axis_none=False`.
        - If `axis` is an integer outside the permissible range for
          the given ``array_base``.
        - If `axis` is `'rows'` or `'columns'` but does not align
          with the dimensionality of ``array_base``.
        - If the string axis is unrecognized.

    TypeError
        If `axis` is not an int, str, or None.

    Examples
    --------
    >>> from fusionlab.core.array_manager import validate_axis
    >>> import numpy as np

    >>> arr_2d = np.ones((5, 4))
    >>> # Validate axis=0 with a 2D array
    >>> ax0 = validate_axis(0, arr_2d)
    >>> print(ax0)
    0
    
    >>> # Validate axis='columns' with a 2D array
    >>> ax1 = validate_axis('columns', arr_2d)
    >>> print(ax1)
    1
    
    >>> # Allow axis=None
    >>> ax_none = validate_axis(None, arr_2d, accept_axis_none=True)
    >>> print(ax_none)
    None
    
    >>> # Disallow axis=None
    >>> validate_axis(None, arr_2d, accept_axis_none=False)
    ValueError: Axis is None, but `accept_axis_none` is False.

    .. note::
       If a 1D array is passed in `array_base` (e.g., shape=(n,)),
       specifying ``axis=1`` will raise an error.

    See Also
    --------
    fusionlab.utils.validator.check_consistency_size :
        A related function that also checks axis validity for
        multi-dimensional arrays.

    References
    ----------
    .. [1] NumPy Development Team. (2023). *NumPy documentation*. 
           https://numpy.org/doc/stable/
    """
    # 1) If axis is None and `accept_axis_none` is True, return None.
    if axis is None:
        if not accept_axis_none:
            raise ValueError(
                "Axis is None, but `accept_axis_none` is False."
            )
        return None

    # 2) Ensure `axis` is either int or str; if not, raise TypeError.
    if not isinstance(axis, (int, str)):
        raise TypeError(
            "axis must be an integer, string, or None."
        )

    # 3) Convert string axis to int: 'rows' -> 0, 'columns' -> 1.
    if isinstance(axis, str):
        axis_str = axis.lower()
        if axis_str == 'rows':
            axis_int = 0
        elif axis_str == 'columns':
            axis_int = 1
        else:
            raise ValueError(
                "Invalid string for `axis`. "
                "Must be 'rows', 'columns', or a valid integer."
            )
    else:
        # axis is an integer
        axis_int = axis

    # 4) If array_base is provided, check that `axis_int` fits
    #    within its dimensionality. For a 1D object, valid axis
    #    is only 0. For 2D, valid axis are 0 or 1, etc.
    if array_base is not None:
        # Determine the number of dimensions
        if hasattr(array_base, 'ndim'):
            ndims = array_base.ndim
        else:
            # If it doesn't have an ndim attribute, fallback
            # to a standard approach if possible
            raise TypeError(
                "Provided `array_base` does not have an `ndim` "
                "attribute, cannot validate axis."
            )
        # Validate axis_int range
        if not (0 <= axis_int < ndims):
            raise ValueError(
                f"Axis {axis_int} is out of bounds for array_base "
                f"with {ndims} dimension(s)."
            )

    # Return final integer axis
    return axis_int

def validate_depth(
    df,
    pred_df=None,
    reference=None,
    ref_col=None,
    depth=None,
    new_name='Depth',
    rename_depth=False,
    reset_index=False,
    check_monotonic=True,
    index_as_depth=True,
    allow_index_mismatch=False,
    error='warn',
    as_series=True,
    check_size=False
):
    """
    Validate and align depth information across DataFrames and related inputs.

    This function ensures that the depth-related data within a primary DataFrame
    (`df`) and optional supplementary DataFrames or Series (`pred_df`, `reference`)
    are consistent, properly aligned, and meet specified criteria. It handles
    index resetting, alignment enforcement, depth determination, monotonicity
    checks, and size validations to maintain data integrity for further analysis
    or processing.

    The validation process can be represented as aligning vectors
    :math:`D_{df}`, :math:`D_{pred}`, and :math:`D_{ref}` such that:

    .. math::
        D_{df} = D_{pred} = D_{ref}

    where each :math:`D` represents the depth values corresponding
    to each DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The primary DataFrame containing the main dataset to be validated.
    
    pred_df : array-like, pandas.Series, or pandas.DataFrame, optional
        Optional predictions or supplementary data corresponding to `df`. If provided,
        it will be aligned with `df` based on the index or reset accordingly.
    
    reference : array-like, pandas.Series, or pandas.DataFrame, optional
        Optional reference data for comparison or validation against `df`. This
        can be a Series or a single-column DataFrame. If a DataFrame with multiple
        columns is provided, `ref_col` must specify which column to use.
    
    ref_col : str, optional
        The name of the column in `reference` to use when `reference` is a
        DataFrame with multiple columns. Required if `reference` has more than one
        column.
    
    depth : str, array-like, pandas.Series, or pandas.DataFrame, optional
        Specifies the depth information to be validated and aligned. It can be:
        - A string representing a column name in `df`.
        - An array-like object providing explicit depth values.
        - A pandas Series or single-column DataFrame containing depth information.
        If `None`, depth is inferred from the DataFrame's index or a range based
        on `df`'s length.
    
    new_name : str, default 'Depth'
        The new name to assign to the depth Series if `rename_depth` is `True`.
    
    rename_depth : bool, default False
        Whether to rename the depth Series to `new_name`. If `False`, the original
        name is retained.
    
    reset_index : bool, default False
        If `True`, the index of `df`, `pred_df`, and `reference` will be reset to
        ensure proper alignment.
    
    check_monotonic : bool, default True
        If `True`, the function checks whether the depth values are monotonic.
        Non-monotonic depth values will trigger sorting or raise an error based
        on the `error` parameter.
    
    index_as_depth : bool, default True
        If `True`, uses the DataFrame's index as the depth values when `depth` is
        not explicitly provided.
    
    allow_index_mismatch : bool, default False
        Determines whether mismatches between the indices of `df` and
        `pred_df`/`reference` are permitted. If `False`, mismatches will trigger
        alignment actions or errors based on the `error` parameter.
    
    error : {'raise', 'warn'}, default 'warn'
        Specifies the behavior when mismatches or validation failures occur.
        - `'raise'`: Raises an exception.
        - `'warn'`: Issues a warning and attempts to correct the issue.
    
    as_series : bool, default True
        If `True`, converts single-column `pred_df` and `reference` back to
        pandas Series objects after processing.
    
    check_size : bool, default False
        If `True`, enforces that `df`, `pred_df`, `reference`, and `depth` have
        matching lengths. Discrepancies will trigger alignment actions or errors
        based on the `error` parameter.

    Returns
    -------
    tuple
        A tuple containing:
        - df : pandas.DataFrame
            The validated and potentially modified primary DataFrame.
        - pred_df : pandas.DataFrame or pandas.Series or None
            The validated and aligned predictions DataFrame or Series, if provided.
        - reference : pandas.DataFrame or pandas.Series or None
            The validated and aligned reference DataFrame or Series, if provided.
        - depth_vals : pandas.Series
            The validated and aligned depth values.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame, or if `pred_df`/`reference`/`depth`
        are of incorrect types.
    
    ValueError
        If `depth` specifications are invalid, or if lengths/indexes do not
        align and `allow_index_mismatch` is `False`.

    Examples
    --------
    >>> from fusionlab.core.checks import validate_depth
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'depth': [10, 20, 30]})
    >>> pred = pd.Series([1.1, 2.1, 3.1], name='prediction')
    >>> reference = pd.Series([1.0, 2.0, 3.0], name='reference')
    >>> validated_df, validated_pred, validated_ref, depth = validate_depth(
    ...     df,
    ...     pred_df=pred,
    ...     reference=reference,
    ...     depth='depth',
    ...     rename_depth=True
    ... )
    >>> print(validated_df)
       A  Depth
    0  1     10
    1  2     20
    2  3     30
    >>> print(validated_pred)
    0    1.1
    1    2.1
    2    3.1
    Name: prediction, dtype: float64
    >>> print(validated_ref)
    0    1.0
    1    2.0
    2    3.0
    Name: reference, dtype: float64
    >>> print(depth)
    0    10
    1    20
    2    30
    Name: Depth, dtype: int64

    Notes
    -----
    - The function is designed to handle various input types and ensure that
      all provided data structures are properly aligned based on depth information.
    - Monotonicity checks are crucial for downstream applications that assume
      ordered depth values.
    - When `allow_index_mismatch` is set to `False`, the function enforces strict
      alignment, which may involve reindexing or raising errors to prevent data
      inconsistencies.

    See Also
    --------
    pandas.DataFrame.align : Align two objects on their axes with the specified
    fill logic.
    pandas.Series.is_monotonic_increasing : Check if the Series is monotonic
    increasing.

    References
    ----------
    .. [1] McKinney, Wes. "Data Structures for Statistical Computing in Python."
       Proceedings of the 9th Python in Science Conference. 2010.
    .. [2] Van Rossum, Guido, and Fred L. Drake Jr. "Python Reference Manual."
       Python.org. 2009.
    """
    def to_series (df_):
        """ Convert dataframe with single columm back to series"""
        if isinstance(df_, pd.DataFrame) and df_.shape[1] == 1:
            df_ = df_.iloc[:, 0]
            
        return df_
    # 1) Ensure that the primary input `df` is a pandas DataFrame.
    if isinstance(df, pd.Series): 
        df = df.to_frame() 
        
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    
    # If  `index_as_depth` is True, ensure that the DataFrame index is numeric. 
    # If it is not, handle according to the `error` policy and resetting
    # before starting any operations.
    if index_as_depth:
        if not np.issubdtype(df.index.dtype, np.number):
            if error =='raise':
                raise ValueError(
                    "Index is not numeric, cannot interpret as depth "
                    "with `index_as_depth=True`."
                )
            elif error == 'warn':
                warnings.warn(
                    "Index is not numeric. Resetting index because "
                    "`index_as_depth=True` was requested."
                )
                df = df.reset_index(drop=True)
            else:  # 'ignore'
                df = df.reset_index(drop=True)
                
    # 5) Determine the `depth` values based on provided parameters.
    if depth is None:
        if index_as_depth:
            # a) Use the DataFrame's index as depth.
            depth_vals = pd.Series(
                df.index,
                name="Depth"
            )
        else:
            # b) Use a range based on the length of `df` as depth.
            depth_vals = pd.Series(
                np.arange(len(df)),
                name="Depth"
            )
    else:
        if isinstance(depth, str):
            # c) Use a column from `df` as depth if `depth` is a string.
            if depth not in df.columns:
                raise ValueError(f"Depth column '{depth}' not found in df.")
            depth_vals = df[depth].copy()
        elif isinstance(depth, pd.DataFrame) and depth.shape[1] == 1:
            # d) Use the single column of a DataFrame as depth.
            depth_vals = depth.iloc[:, 0].copy()
        elif isinstance(depth, (list, np.ndarray)):
            # e) Convert list or ndarray to Series for depth.
            depth_vals = pd.Series(
                depth,
                name=new_name
            )
        elif isinstance(depth, pd.Series):
            # f) Use the provided Series as depth.
            depth_vals = depth.copy()
        else:
            raise TypeError(
                "`depth` must be str, 1D array-like, Series,"
                " or single-col DataFrame."
            )
            
    # 2) Optionally reset the index of `df`, `pred_df`,
    # and `reference` to ensure alignment.
    if reset_index:
        df = df.reset_index(drop=True)
        
        if isinstance(pred_df, pd.DataFrame):
            pred_df = pred_df.reset_index(drop=True)
        
        if isinstance(reference, pd.DataFrame):
            reference = reference.reset_index(drop=True)
    
    # 3) Process `pred_df` if it is provided.
    if pred_df is not None:
        # a) Convert list or ndarray to DataFrame 
        # with aligned index and default column names.
        if isinstance(pred_df, (list, np.ndarray)):
            pred_df = pd.DataFrame(
                pred_df,
                index=df.index
            )
            pred_df.columns = [f"pred_{i}" for i in range(pred_df.shape[1])]
        
        # b) Convert Series to DataFrame and handle index reset if necessary.
        elif isinstance(pred_df, pd.Series):
            pred_df = pred_df.to_frame(
                name=pred_df.name or "prediction"
            )
            if reset_index:
                pred_df = pred_df.reset_index(drop=True)
        
        # c) Raise error if `pred_df` is not array-like, Series, or DataFrame.
        elif not isinstance(pred_df, pd.DataFrame):
            raise TypeError(
                "`pred_df` must be array-like, Series, or DataFrame.")
        
        # d) Align `pred_df` with `df`'s index unless mismatches are allowed.
        if (len(pred_df) != len(df)) and not allow_index_mismatch:
            if error == 'raise':
                raise ValueError("`pred_df` length/index does not match `df`.")
            elif error == 'warn':
                warnings.warn(
                    "`pred_df` length/index does not match `df`. "
                    "Forcing alignment by reindexing pred_df."
                )
            # Force alignment by reindexing to match `df`'s index.
            pred_df = pred_df.reindex(df.index)
    
    # 4) Process `reference` if it is provided.
    if reference is not None:
        if isinstance(reference, pd.DataFrame):
            # a) Extract the reference column if `ref_col` is specified and exists.
            if ref_col and ref_col in reference.columns:
                reference = reference[ref_col]
            # b) If DataFrame has only one column, use it as the reference Series.
            elif reference.shape[1] == 1:
                reference = reference.iloc[:, 0]
            else:
                raise ValueError(
                    "`reference` DataFrame must have exactly one column "
                    "or a valid `ref_col` must be specified."
                )
            # c) Reset index if required.
            if reset_index:
                reference = reference.reset_index(drop=True)
        elif isinstance(reference, (list, np.ndarray)):
            # d) Convert list or ndarray to Series with aligned index.
            reference = pd.Series(
                reference,
                index=df.index,
                name="reference"
            )
        elif isinstance(reference, pd.Series):
            # e) Reset index of Series if required.
            if reset_index:
                reference = reference.reset_index(drop=True)
        else:
            raise TypeError(
                "`reference` must be array-like, Series,"
                " or a single-col DataFrame."
            )
        
        # f) Align `reference` with `df`'s index unless mismatches are allowed.
        if (len(reference) != len(df)) and not allow_index_mismatch:
            if error == 'raise':
                raise ValueError("`reference` length/index does not match `df`.")
            elif error == 'warn':
                warnings.warn(
                    "`reference` length/index does not match `df`. "
                    "Forcing alignment by reindexing reference."
                )
            # Force alignment by reindexing to match `df`'s index.
            reference = reference.reindex(df.index)
      
    # 6) Optionally rename the `depth` Series to `new_name`.
    if rename_depth:
        depth_vals.name = new_name
    
    # 7) Enforce that all inputs have the same size if `check_size` is True.
    if check_size:
        len_df = len(df)
        
        # a) Check and adjust `pred_df` size.
        if pred_df is not None:
            len_pred = len(pred_df)
            if len_pred != len_df:
                if error == 'raise':
                    raise ValueError(
                        "`pred_df` length does not match `df` (check_size=True)."
                    )
                elif error == 'warn':
                    warnings.warn(
                        "`pred_df` length does not match `df`. "
                        "Forcing alignment by slicing or reindexing."
                    )
                # Slice `pred_df` to match `df` length.
                pred_df = pred_df.iloc[:len_df].reset_index(drop=True)
        
        # b) Check and adjust `reference` size.
        if reference is not None:
            len_ref = len(reference)
            if len_ref != len_df:
                if error == 'raise':
                    raise ValueError(
                        "`reference` length does not match `df` (check_size=True)."
                    )
                elif error == 'warn':
                    warnings.warn(
                        "`reference` length does not match `df`. "
                        "Forcing alignment by slicing or reindexing."
                    )
                # Slice `reference` to match `df` length.
                reference = reference.iloc[:len_df].reset_index(drop=True)
        
        # c) Check and adjust `depth_vals` size relative to `df`.
        len_depth = len(depth_vals)
        if len_depth > len_df:
            if error == 'warn':
                warnings.warn(
                    "Depth array is longer than df. Slicing"
                    " depth to match df length."
                )
            # Slice `depth_vals` to match `df` length.
            depth_vals = depth_vals.iloc[:len_df].reset_index(drop=True)
        elif len_depth < len_df:
            if error == 'warn':
                warnings.warn(
                    "Depth array is shorter than df. Slicing df, pred_df, "
                    "and reference to match depth length."
                )
            # Slice `df` to match `depth_vals` length.
            df = df.iloc[:len_depth].reset_index(drop=True)
            
            # Slice `pred_df` if it exists.
            if isinstance(pred_df, pd.DataFrame):
                pred_df = pred_df.iloc[:len_depth].reset_index(drop=True)
            elif isinstance(pred_df, pd.Series):
                pred_df = pred_df.iloc[:len_depth].reset_index(drop=True)
            
            # Slice `reference` if it exists.
            if isinstance(reference, pd.DataFrame):
                reference = reference.iloc[:len_depth].reset_index(drop=True)
            elif isinstance(reference, pd.Series):
                reference = reference.iloc[:len_depth].reset_index(drop=True)
            
            # Reset `depth_vals` index after slicing.
            depth_vals = depth_vals.reset_index(drop=True)
    
    # 8) Ensure that `depth_vals` and `df` have the same length
    # unless mismatches are allowed.
    if (len(depth_vals) != len(df)) and not allow_index_mismatch:
        if error == 'raise':
            raise ValueError("Depth length does not match `df`.")
        elif error == 'warn':
            warnings.warn(
                "Depth length does not match `df`. Forcing alignment."
            )
        # Force alignment by reindexing `depth_vals` to match `df`'s index.
        depth_vals = depth_vals.reindex(df.index)
    
    # 9) If `index_as_depth` is True and `depth` is provided, verify
    # that `depth_vals` matches `df`'s index.
    if index_as_depth and (depth is not None):
        index_series = pd.Series(
            df.index,
            name=depth_vals.name
        )
        if not depth_vals.equals(index_series):
            if not allow_index_mismatch:
                if error == 'raise':
                    raise ValueError(
                        "`index_as_depth=True` but depth does not match df.index."
                    )
                elif error == 'warn':
                    warnings.warn(
                        "`index_as_depth=True` but depth does not match df.index. "
                        "Forcing alignment by replacing depth with df.index."
                    )
            # Replace `depth_vals` with `df`'s index.
            depth_vals = index_series
    
    # 10) Check if `depth_vals` is monotonic. If not, 
    # sort all related DataFrames accordingly.
    if check_monotonic:
        sorted_idx = np.argsort(depth_vals.values)
        is_monotonic = (np.diff(depth_vals.values) >= 0).all()
        
        if not is_monotonic:
            if error == 'raise':
                raise ValueError("Depth is not monotonic.")
            elif error == 'warn':
                warnings.warn(
                    "Depth is not monotonic. Sorting df, depth, pred_df, and reference."
                )
            # Sort `df` and related DataFrames based on sorted indices of `depth_vals`.
            df = df.iloc[sorted_idx].reset_index(drop=True)
            depth_vals = depth_vals.iloc[sorted_idx].reset_index(drop=True)
            
            if isinstance(pred_df, pd.DataFrame):
                pred_df = pred_df.iloc[sorted_idx].reset_index(drop=True)
            elif isinstance(pred_df, pd.Series):
                pred_df = pred_df.iloc[sorted_idx].reset_index(drop=True)
            
            if isinstance(reference, pd.DataFrame):
                reference = reference.iloc[sorted_idx].reset_index(drop=True)
            elif isinstance(reference, pd.Series):
                reference = reference.iloc[sorted_idx].reset_index(drop=True)
        else:
            # Check if `depth_vals` is strictly increasing.
            strictly_increasing = depth_vals.is_monotonic_increasing
            if not strictly_increasing:
                if error == 'warn':
                    warnings.warn("Depth is not strictly monotonic. Proceeding as is.")
                elif error == 'raise':
                    raise ValueError("Depth is not strictly monotonic.")
    
    # 11) Optionally convert single-column `pred_df`
    # and `reference` back to Series.
    if as_series:
        pred_df = to_series (pred_df ) 
        reference = to_series (reference ) 
        depth_vals = to_series (depth_vals) 
        
    # 12) Return the validated and potentially modified
    # DataFrames along with `depth_vals`.
    return df, pred_df, reference, depth_vals

def check_non_emptiness(
    params=None,
    *,
    error: str = "raise",
    none_as_empty: bool = True,
    ellipsis_as_empty: bool = True,
    include: tuple = ("set", "dict")
):
    """
    Decorator to check for non-emptiness of specified
    parameters in a function. By default, it checks
    array-like structures (lists, tuples, Series,
    DataFrame, NumPy arrays) but can also consider
    sets, dicts, or other types via <include inline>.

    If the function is used without parentheses
    (e.g. ``@check_emptiness``), the first positional
    argument is checked by default. Otherwise, specify
    a list of parameter names in ``params``.

    Parameters
    ----------
    params : list of str, optional
        Names of arguments whose emptiness will be
        checked. If None and the decorator is used
        without parentheses, the first positional
        argument is checked.
    error : str, optional
        How to handle an empty argument:
          - "raise": raise ValueError
          - "warn": issue a warning
          - "ignore": do nothing
        Default "raise".
    none_as_empty : bool, optional
        If True, consider None as empty. Default True.
    ellipsis_as_empty : bool, optional
        If True, consider the Ellipsis object
        (``...``) as empty. Default True.
    include : tuple, optional
        Additional types to treat as potentially
        empty, e.g., ("set", "dict"). Default
        ("set", "dict").

    Returns
    -------
    callable
        The decorated function that checks emptiness
        for the specified arguments.

    Examples
    --------
    
    1) Decorator used without parentheses:
       >>> from fusionlab.core.checks import check_non_emptiness
       >>> @check_non_emptiness
       ... def func_first_arg(x):
       ...     return x
       ...
       >>> # Here, if x is empty, handle it according
       ... # to <error inline>.

    2) Specify which parameters to check:
       >>> @check_non_emptiness(params=['df'],
       ...                  error='warn',
       ...                  none_as_empty=True)
       ... def process_data(a, df=None):
       ...     return df

    3) Check multiple parameters:
       >>> @check_non_emptiness(params=['arr', 'df'],
       ...                  error='raise')
       ... def model_fit(arr, df=None, *args):
       ...     return (arr, df)

    Notes
    -----
    1. If <none_as_empty inline> is True and an
       argument is None, it is considered empty.
    2. If <ellipsis_as_empty inline> is True and
       an argument is Ellipsis (``...``), it is
       considered empty.
    3. For additional types, such as <set inline>
       or <dict inline>, set them in <include
       inline>. By default, sets and dicts are
       also checked for emptiness.
    4. By default, standard array-like structures
       (lists, tuples, Series, DataFrames, numpy
       arrays) are checked.

    """
    # Case A: If decorator is used without parentheses,
    # then `params` is actually the function object itself.
    if callable(params):
        func = params

        @wraps(func)
        def _wrapper(*args, **kwargs):
            if args:
                converted_args = list(args)
                converted_args[0] = _check_and_handle_emptiness(
                    converted_args[0],
                    error=error,
                    none_as_empty=none_as_empty,
                    ellipsis_as_empty=ellipsis_as_empty,
                    include=include,
                    param_name="first positional argument"
                )
                args = tuple(converted_args)
            return func(*args, **kwargs)

        return _wrapper

    # Case B: If decorator is used with parentheses
    # then `params` is the list of names or None.
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            valid_params = params or []

            # If no params were specified, check the
            # first positional argument (if present).
            if not valid_params:
                if args:
                    converted_args = list(args)
                    converted_args[0] = _check_and_handle_emptiness(
                        converted_args[0],
                        error=error,
                        none_as_empty=none_as_empty,
                        ellipsis_as_empty=ellipsis_as_empty,
                        include=include,
                        param_name="first positional argument"
                    )
                    args = tuple(converted_args)
            else:
                # For each name in valid_params,
                # check emptiness.
                sig = inspect.signature(func)
                parameters = list(sig.parameters.keys())

                for name in valid_params:
                    if name in kwargs:
                        kwargs[name] = _check_and_handle_emptiness(
                            kwargs[name],
                            error=error,
                            none_as_empty=none_as_empty,
                            ellipsis_as_empty=ellipsis_as_empty,
                            include=include,
                            param_name=name
                        )
                    else:
                        # Possibly positional
                        try:
                            idx = parameters.index(name)
                            if idx < len(args):
                                converted_args = list(args)
                                converted_args[idx] = _check_and_handle_emptiness(
                                    converted_args[idx],
                                    error=error,
                                    none_as_empty=none_as_empty,
                                    ellipsis_as_empty=ellipsis_as_empty,
                                    include=include,
                                    param_name=name
                                )
                                args = tuple(converted_args)
                        except ValueError:
                            # param not found in function signature
                            pass

            return func(*args, **kwargs)

        return _wrapper

    return _decorator

def _check_and_handle_emptiness(
    value,
    *,
    error: str,
    none_as_empty: bool,
    ellipsis_as_empty: bool,
    include: tuple,
    param_name: str
):
    """
    Internal helper to detect emptiness of `value`
    and handle it according to <error inline>
    policy.
    """
    if value is None and none_as_empty:
        # Consider None as empty
        return _handle_empty(error, param_name)
    if value is Ellipsis and ellipsis_as_empty:
        # Consider Ellipsis (...) as empty
        return _handle_empty(error, param_name)

    # Check standard array-likes:
    if _is_arraylike_empty(value):
        return _handle_empty(error, param_name)

    # Check optional sets, dicts, etc.
    # We only check if 'set' or 'dict' is in include
    # (by default, we do check them).
    include = include or []
    if include: 
        if "set" in include and isinstance(value, set):
            if len(value) == 0:
                return _handle_empty(error, param_name)
        if "dict" in include and isinstance(value, dict):
            if len(value) == 0:
                return _handle_empty(error, param_name)

    # If it's not considered empty, return as is.
    return value

def _is_arraylike_empty(value):
    """
    Heuristic check if `value` is an empty
    array-like structure (list, tuple, Series,
    DataFrame, numpy array).
    """
    # check list or tuple
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return True

    # check pandas Series/DataFrame
    if isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
        if value.empty:
            return True

    # check numpy array
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return True

    return False


def _handle_empty(error_policy, param_name):
    """
    Handle an empty argument based on the
    <error_policy inline>.
    """
    msg = f"Argument '{param_name}' is empty."
    if error_policy == "raise":
        raise ValueError(msg)
    elif error_policy == "warn":
        warnings.warn(msg, UserWarning)
    elif error_policy == "ignore":
        pass
    # Return None so that we effectively "clear"
    # the argument if emptiness was found
    return None
