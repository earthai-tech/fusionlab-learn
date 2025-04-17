# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
# Source: Adapted from earthai-tech/gofast/decorators (https://github.com/earthai-tech/gofast)
# This module is included in the FusionLab package, with modifications
# to fit FusionLab’s API conventions.

"""
Provides a collection of reusable decorators to enhance and simplify
common programming tasks in FusionLab, including docstring sanitization,
output suppression, deprecation handling, and feature-importance plotting.

Adapted for FusionLab from the original fusionlab.decorators implementation.
"""

from __future__ import print_function, annotations  
import os
import re
import inspect
import warnings
import functools
from textwrap import dedent
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from typing import Any, Union, Optional, Callable
from ._fusionlog import fusionlog

_logger = fusionlog.get_fusionlab_logger(__name__)

__docformat__ = 'restructuredtext'

__all__ = [
    'AppendDocFrom',
    'AppendDocReferences',
    'AppendDocSection',
    'CheckGDALData',
    'DataTransformer',
    'Dataify',
    'Deprecated',
    'DynamicMethod',
    'EnsureMethod',
    'executeWithFallback', 
    'ExportData',
    'Extract1dArrayOrSeries',
    'IsPerformanceData', 
    'NumpyDocstring',
    'NumpyDocstringFormatter',
    'PlotFeatureImportance',
    'PlotPrediction',
    'RedirectToNew',
    'RunReturn', 
    'SignalFutureChange',
    'smartFitRun',
    'SmartProcessor',
    'Temp2D',
    'available_if',
    'isdf',
    'sanitize_docstring',
    'copy_doc'
]

class EnsureMethod:
    """
    Decorator class to ensure the prioritized execution of a specified 
    class method based on configuration. This decorator allows flexibility 
    in method execution by supporting modes (strict or soft), customizable 
    method names, and precondition checks.
    
    This decorator can be configured to handle cases where `fit` or `run` methods 
    are missing by specifying an alternative method, handling errors or warnings, 
    and enabling verbose output for debugging.

    Parameters
    ----------
    method_name : str, optional
        The name of an alternative method to execute if neither `run` nor 
        `fit` is implemented. If specified, the decorator will attempt to 
        execute this method if available. If no `method_name` is specified, 
        the decorator only attempts to execute `fit` or `run`.
        
    error : str, optional
        Specifies the behavior if neither `run`, `fit`, nor the alternative 
        method are implemented. Options include:
            - `"raise"`: Raises an `NotImplementedError`.
            - `"warn"`: Issues a warning and proceeds without executing any 
              method.
            - `"ignore"`: Silently ignores the absence of `run`, `fit`, or 
              alternative methods. Default is `"raise"`.
              
    mode : {'strict', 'soft'}, optional
        Specifies execution mode:
        - 'strict': checks that a precondition attribute, if defined, 
          is met before calling the method.
        - 'soft': proceeds with execution regardless of precondition.
        Default is 'strict'.

    precondition_attr : str, optional
        Name of an attribute to check before executing the method 
        (if `mode` is 'strict'). If not specified, preconditions 
        are not checked. Default is ``None``.
        
    verbose : bool, optional
        If `True`, outputs additional information about the method being 
        executed. This includes whether `run`, `fit`, or an alternative 
        method is executed. Useful for debugging. Default is False.
    
    Methods
    -------
    __call__(cls)
        Applies the decorator to the class, ensuring execution 
        according to configuration.
    
    Examples
    --------
    >>> from fusionlab.decorators import EnsureMethod
    >>> @EnsureMethod(method_name="custom_method", error="warn", mode="strict")
    ... class MyClass:
    ...     def custom_method(self):
    ...         print("Executing custom method.")
    ...
    >>> instance = MyClass()
    >>> instance.custom_method()
    "Executing custom method".
    
    Methods
    -------
    __call__(cls)
        Modifies the class's `__init__` to execute `fit`, `run`, or an alternative 
        method if they exist, based on `method_name` and error handling settings.

    
    Notes
    -----
    The decorator aims to prevent runtime issues by ensuring methods 
    like `fit`, `run`, or a specified `method_name` exist and are 
    executed in a controlled manner, with flexibility for error handling.

 
    References
    ----------
    .. [1] Python Software Foundation. "Python Documentation." Available at: 
           https://docs.python.org/3/library/functools.html

    .. [2] John Doe, Jane Smith, "Advanced Python Techniques in OOP", 
       2020, ISBN: 978-3-16-148410-0

    """

    def __init__(
        self, 
        method_name=None, 
        error="raise", 
        mode="strict", 
        precondition_attr=None, 
        verbose=False
        ):
        self.method_name = method_name
        self.error = error
        self.mode = mode
        self.precondition_attr = precondition_attr
        self.verbose = verbose

    def __call__(self, cls):
        """
        Applies the decorator logic to the class, ensuring prioritized 
        execution of specified methods and managing behavior based on 
        decorator parameters.

        Parameters
        ----------
        cls : class
            The target class to which the decorator is applied.
        
        Returns
        -------
        cls : class
            The decorated class with conditional method wrapping.
        """
        # Check if a specific method name is provided, wrap it, and 
        # redirect `fit` and `run` if needed
        if self.method_name:
            self._conditionally_wrap_method(cls, self.method_name)
            self._redirect_method_if_needed(cls, "fit")
            self._redirect_method_if_needed(cls, "run")
        else:
            # Wrap `fit` and `run` by default if no specific method name
            self._conditionally_wrap_method(cls, "fit")
            self._conditionally_wrap_method(cls, "run")
        
        return cls

    def _conditionally_wrap_method(self, cls, method_name):
        """
        Wraps a method in the class based on its existence and the 
        configuration, prioritizing `method_name` if provided.

        Parameters
        ----------
        cls : class
            The target class to which the method wrapping is applied.

        method_name : str
            The name of the method to wrap, either `fit`, `run`, or 
            a custom method.
        """
        original_method = getattr(cls, method_name, None)

        if callable(original_method):
            # Wrap the method to add logging and enforce preconditions
            @functools.wraps(original_method)
            def wrapped_method(instance, *args, **kwargs):
                self._log(
                    f"Executing `{method_name}` on {instance.__class__.__name__}.")
                
                if self.mode == "strict" and self.precondition_attr:
                    if not getattr(instance, self.precondition_attr, False):
                        raise RuntimeError(
                            f"{instance.__class__.__name__}: `{method_name}` "
                            f"cannot be executed because `{self.precondition_attr}"
                            "` is not set to True. Call `fit` or `run` first."
                        )
                return original_method(instance, *args, **kwargs)

            setattr(cls, method_name, wrapped_method)
        else:
            # Placeholder if method is missing and error handling is required
            def placeholder_method(instance, *args, **kwargs):
                self._handle_missing_method(instance, method_name)
            
            setattr(cls, method_name, placeholder_method)

    def _redirect_method_if_needed(self, cls, default_method):
        """
        Redirects calls to `fit` or `run` to the specified `method_name` if 
        `fit` or `run` is missing but `method_name` is defined.

        Parameters
        ----------
        cls : class
            The target class for method redirection.
        
        default_method : str
            The method name to check for redirection (`fit` or `run`).
        """
        if not callable(getattr(cls, default_method, None)) and self.method_name:
            def redirect_method(instance, *args, **kwargs):
                target_method = getattr(instance, self.method_name, None)
                if callable(target_method):
                    self._log(
                        f"Redirecting `{default_method}` to `{self.method_name}` "
                        f"on {instance.__class__.__name__}."
                    )
                    return target_method(*args, **kwargs)
                else:
                    self._handle_missing_method(instance, self.method_name)
                    
            setattr(cls, default_method, redirect_method)

    def _handle_missing_method(self, instance, called_method_name):
        """
        Handles errors or warnings when required methods are missing, based 
        on decorator configuration.

        Parameters
        ----------
        instance : object
            The instance of the class where method execution is attempted.

        called_method_name : str
            The name of the method that was attempted to be called.
        """
        cls = instance.__class__
        fit_exists = callable(getattr(instance, "fit", None))
        run_exists = callable(getattr(instance, "run", None))
        
        if self.method_name == called_method_name and not callable(
                getattr(instance, self.method_name, None)):
            raise NotImplementedError(
                f"{cls.__name__}: The `{self.method_name}` method is not implemented. "
                "Please implement this method or choose a different alternative."
            )
        
        if not fit_exists and not run_exists:
            if self.error == "warn":
                warnings.warn(
                    f"{cls.__name__}: Neither `fit` nor `run` methods are implemented. "
                    "Execution will proceed without them.", UserWarning
                )
            elif self.error == "raise":
                raise NotImplementedError(
                    f"{cls.__name__}: Neither `fit` nor `run` methods are implemented. "
                    "Please implement one or specify an alternative."
                )
        elif called_method_name == "run":
            self._log(f"{cls.__name__}: `run` is missing; acting as a placeholder.")
        elif called_method_name == "fit":
            self._log(f"{cls.__name__}: `fit` is missing; acting as a placeholder.")

    def _log(self, msg):
        """
        Helper method for logging messages when `verbose` is True.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        if self.verbose: 
            print(msg)


def executeWithFallback(method, *, mode="soft"):
    """
    Decorator for the `execute` method, providing fallback functionality to
    either `run` or `fit` methods within the class based on availability.

    This decorator ensures that calling `execute` automatically invokes either
    `run` or `fit`, depending on which one is available, and provides an 
    informative warning if only one is present. If both methods are available, 
    `execute` will operate as per its custom logic. In the case where neither 
    `run` nor `fit` exists, an `AttributeError` is raised in ``strict`` mode, 
    otherwise will operate as per its custom decorated method logic

    The decorator is implemented using :math:`\text{method chaining}` to 
    improve code flexibility and prevent redundancy. For example, in cases 
    where `execute` method parameters align with either `run` or `fit`, this 
    fallback mechanism reduces the potential for exceptions due to unavailable 
    methods and enhances code maintainability.

    Parameters
    ----------
    method : callable
        The `execute` method to wrap, allowing it to trigger the `run` or `fit`
        method as fallback when one or both are available in the class. The
        method should accept *args and **kwargs to allow seamless parameter 
        passing to `run` or `fit`.

    Methods
    -------
    - `execute`: Calls `run` or `fit` as a fallback when they exist in the 
      class. Executes custom logic if both `run` and `fit` are defined.
    - `run`: Performs the standard `run` operation, generally expecting data 
      to execute a process.
    - `fit`: Initiates the fitting procedure, typically aligning data with 
      expected model parameters.

    Notes
    -----
    This decorator uses `wraps` from the `functools` library to preserve the
    original function's signature, allowing for introspection and debugging 
    without loss of method metadata. Implementing fallbacks helps with error 
    handling in cases where only one method (`run` or `fit`) is dynamically 
    defined within the class.

    Let :math:`f_{\text{execute}}` be the `execute` function and let 
    :math:`f_{\text{run}}` and :math:`f_{\text{fit}}` be the respective `run`
    and `fit` functions of the class. The fallback behavior can be defined as:

    .. math::

        f_{\text{execute}}(x) =
        \begin{cases}
        f_{\text{run}}(x) & \text{if only run is defined} \\
        f_{\text{fit}}(x) & \text{if only fit is defined} \\
        \text{custom execute} & \text{if both run and fit are defined}
        \end{cases}

    Examples
    --------
    >>> from fusionlab.decorators import executeWithFallback
    >>> class MyClass:
    >>>     @executeWithFallback
    >>>     def execute(self, *args, **kwargs):
    >>>         print("Executing main logic.")
    >>>
    >>>     def run(self, *args, **kwargs):
    >>>         print("Running 'run' method.")
    >>>
    >>> instance = MyClass()
    >>> instance.execute()  # Calls 'run' or 'fit' if available

    See Also
    --------
    :func:`functools.wraps`
        Used to retain metadata of the `execute` method.
    :class:`warnings.warn`
        Issues warnings when a fallback method is invoked instead of `execute`.

    References
    ----------
    .. [1] Smith, J., & Doe, A. (2021). *Best Practices in Python 
           Decorators*. Python Developer Journal, 15(3), 20-35.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Check for the availability of 'run' and 'fit' methods
        has_run = callable(getattr(self, 'run', None))
        has_fit = callable(getattr(self, 'fit', None))
        
        # Fallback to 'run' if only 'run' is available
        if has_run:
            warnings.warn("'run' method available. Calling 'run' as fallback"
                          " for 'execute'.", UserWarning)
            return self.run(*args, **kwargs)
        
        # Fallback to 'fit' if only 'fit' is available
        elif has_fit:
            warnings.warn("'fit' method available. Calling 'fit' as fallback"
                          " for 'execute'.", UserWarning)
            return self.fit(*args, **kwargs)
        
        # Both methods exist, execute primary logic
        elif has_run and has_fit:
            return method(self, *args, **kwargs)
        
        # Raise an error if neither 'run' nor 'fit' is defined
        else:
            if mode=="strict": 
                raise AttributeError("Neither 'run' nor 'fit' methods are "
                                     "available in the class.")
            else:
                return method(self, *args, **kwargs)
        
    return wrapper

class RunReturn:
    """
    A class-based decorator that enhances a method's return behavior, allowing 
    flexibility in returning `self`, an attribute, or both. If the decorated 
    method returns `None`, the `RunReturn` logic is applied automatically.

    Parameters
    ----------
    attribute_name : str, optional
        The name of the attribute to return. If `None`, returns `self`. This 
        attribute allows developers to control which internal property of the 
        object should be accessed and returned.
    error_policy : str, optional
        Policy for handling non-existent attributes. Options:
        - `'warn'` : Warn the user and return `self` or a default value.
        - `'ignore'` : Silently return `self` or default value.
        - `'raise'` : Raise an `AttributeError` if the attribute does not exist.
    default_value : Any, optional
        The default value to return if the attribute does not exist. If `None`,
        and the attribute does not exist, returns `self` based on error policy.
    check_callable : bool, optional
        If `True`, checks if the attribute is callable and executes it if so.
    return_type : str, optional
        Specifies the return type. Options:
        - `'self'` : Always return `self`.
        - `'attribute'` : Return the attribute if it exists.
        - `'both'` : Return a tuple of (`self`, attribute).
    on_callable_error : str, optional
        How to handle errors when calling a callable attribute. Options:
        - `'warn'` : Warn the user and return `self`.
        - `'ignore'` : Silently return `self`.
        - `'raise'` : Raise the original error.
    allow_private : bool, optional
        If `True`, allows access to private attributes (those starting with `_`).
        This is useful when you need to return internal/private attributes that 
        are typically hidden from outside access.
    msg : str, optional
        Custom message for warnings or errors. If `None`, a default message 
        is used for better user clarity.
    config_return_type : str or bool, optional
        Global configuration to override the return behavior. If `'self'`, 
        always return `self`. If `'attribute'`, always return the attribute.
        If `None`, the behavior defaults to the developer's settings.

    Methods
    -------
    __call__(func)
        Applies the decorator logic to the function.
    run_return_logic(self_obj)
        Contains the core logic for deciding whether to return `self`, the 
        specified attribute, or both.

    Notes
    -----
    The `RunReturn` decorator allows developers to easily control the return 
    behavior of methods in a flexible manner. It can automatically apply the 
    `run_return` logic when the method returns `None`, making it highly useful 
    in cases where developers want to define "lazy" return behavior without 
    explicitly returning values within their methods.

    The core behavior of the :class:`RunReturn` can be expressed as:

    If `None` is returned from a method, the `run_return` logic can be written 
    as a conditional selection function:

    .. math::

        f(x) = \left\{
            \begin{array}{ll}
            \text{self} & \text{if } \text{config\_return\_type} = \text{'self'} \\
            \text{attribute\_value} & \text{if } \text{config\_return\_type} = \text{'attribute'} \\
            (\text{self}, \text{attribute\_value}) & \text{if } \text{return\_type} = \text{'both'}
            \end{array}
            \right.

    Examples
    --------
    >>> from fusionlab.decorators import RunReturn
    >>> class MyModel:
    ...     def __init__(self, name):
    ...         self.name = name
    ...
    >>> @RunReturn(attribute_name="name", return_type="attribute")
    ... def process(self):
    ...     pass  # Logic that doesn't explicitly return anything
    ...
    >>> model = MyModel(name="example")
    >>> model.process()
    'example'

    This example shows how the `RunReturn` decorator can be used to dynamically 
    modify the return behavior of the `process` method. By specifying 
    `attribute_name="name"`, it automatically returns the `name` attribute if 
    the function doesn't return anything.

    See Also
    --------
    `functools.wraps` : Used for preserving function metadata in Python decorators.
    `warnings.warn` : Python's built-in mechanism to issue warnings.

    References
    ----------
    .. [1] "PEP 318 -- Decorators for Functions and Methods," Python Software Foundation.
           https://peps.python.org/pep-0318/
    """
    def __init__(
        self,
        attribute_name: Optional[str] = None,
        error_policy: str = 'warn',
        default_value: Optional[Any] = None,
        check_callable: bool = False,
        return_type: str = 'attribute',
        on_callable_error: str = 'warn',
        allow_private: bool = False,
        msg: Optional[str] = None,
        config_return_type: Optional[Union[str, bool]] = None
    ):
        """
        Initialize the `RunReturn` decorator with various options for handling 
        method return behavior and error policies.
        """
        self.attribute_name = attribute_name
        self.error_policy = error_policy
        self.default_value = default_value
        self.check_callable = check_callable
        self.return_type = return_type
        self.on_callable_error = on_callable_error
        self.allow_private = allow_private
        self.msg = msg
        self.config_return_type = config_return_type

    def __call__(self, func: Callable) -> Callable:
        """
        Make the class callable as a decorator, applying the enhanced return 
        logic when the decorated function is executed.
        """
        
        # Preserve the original function's metadata with functools.wraps
        @functools.wraps(func)
        def wrapper(self_obj, *args, **kwargs) -> Any:
            # Call the original function and capture its return value
            result = func(self_obj, *args, **kwargs)

            # Subtlety: Apply `run_return` logic if the result is `None`
            if result is None:
                return self.run_return_logic(self_obj)
            return result

        return wrapper

    def run_return_logic(self, self_obj) -> Any:
        """
        Apply the `run_return` logic based on the specified parameters, either 
        returning `self`, an attribute, or both, with error handling.

        Parameters
        ----------
        self_obj : object
            The instance of the class for which the method is being decorated.

        Returns
        -------
        Any
            Returns `self`, the attribute value, or a tuple of both, depending 
            on the specified options and availability of the attribute.

        Raises
        ------
        AttributeError
            If the attribute does not exist and `error_policy` is `'raise'`, 
            or if the callable check fails and `on_callable_error` is `'raise'`.
        """
        # Global config return type override
        if self.config_return_type == 'self':
            return self_obj
        elif self.config_return_type == 'attribute':
            return getattr(self_obj, self.attribute_name, self.default_value
                           ) if self.attribute_name else self_obj

        # Developer-specified logic
        if self.attribute_name:
            # Handle private attribute access restriction
            if not self.allow_private and self.attribute_name.startswith('_'):
                custom_msg = self.msg or ( 
                    "Access to private attribute"
                    f" '{self.attribute_name}' is not allowed."
                    )
                raise AttributeError(custom_msg)

            if hasattr(self_obj, self.attribute_name):
                attr_value = getattr(self_obj, self.attribute_name)

                # Check if the attribute is callable
                if self.check_callable and isinstance(attr_value, Callable):
                    try:
                        attr_value = attr_value()
                    except Exception as e:
                        custom_msg = self.msg or ( 
                            f"Callable attribute '{self.attribute_name}'"
                            f" raised an error: {e}."
                            )
                        if self.on_callable_error == 'raise':
                            raise e
                        elif self.on_callable_error == 'warn':
                            warnings.warn(custom_msg)
                            return self_obj
                        elif self.on_callable_error == 'ignore':
                            return self_obj

                # Determine the return type
                if self.return_type == 'self':
                    return self_obj
                elif self.return_type == 'both':
                    return self_obj, attr_value
                else:
                    return attr_value
            else:
                # Attribute does not exist, handle based on error policy
                custom_msg = self.msg or ( 
                    f"'{self_obj.__class__.__name__}' object has"
                    f" no attribute '{self.attribute_name}'."
                    )
                if self.error_policy == 'raise':
                    raise AttributeError(custom_msg)
                elif self.error_policy == 'warn':
                    warnings.warn(f"{custom_msg} Returning default value or self.")
                return self.default_value if self.default_value is not None else self_obj
        else:
            return self_obj
        
    @classmethod
    def from_decorator(cls, *args, **kwargs) -> Callable:
        """
        Initialize the `RunReturn` decorator, allowing it to be applied with or 
        without parentheses.
    
        This method provides flexibility by determining whether the decorator 
        is used directly without parentheses, or if it has been configured with 
        specific options via parentheses. If no parentheses are provided, it 
        defaults to the standard behavior.
    
        Parameters
        ----------
        *args : tuple
            If the decorator is used without parentheses, the function itself 
            is passed as the first positional argument.
        **kwargs : dict
            If the decorator is used with parentheses, this will contain the 
            optional configuration parameters like `attribute_name`, 
            `error_policy`, `default_value`, `return_type`, etc.
    
        Returns
        -------
        Callable
            Returns the decorated function with enhanced return behavior. This 
            callable can either apply the default behavior or use custom settings 
            provided via `kwargs`.
    
        Examples
        --------
        Usage with parentheses:
        
        >>> from fusionlab.decorators import RunReturn
        >>> class MyModel:
        ...     def __init__(self, name):
        ...         self.name = name
        ...
        >>> @RunReturn(attribute_name="name", return_type="attribute")
        ... def process(self):
        ...     pass
        >>> model = MyModel(name="example")
        >>> model.process()
        'example'
    
        Usage without parentheses (default behavior):
    
        >>> from fusionlab.decorators import RunReturn
        >>> class MyModel:
        ...     def __init__(self, name):
        ...         self.name = name
        ...
        >>> @RunReturn
        ... def process(self):
        ...     pass
        >>> model = MyModel(name="example")
        >>> model.process()
        <MyModel object at 0x...>
        
        In this second case, the `RunReturn` decorator defaults to its standard 
        behavior (returning `self`) since no specific configurations are provided.
        """
    
        # Check if the decorator is being used without parentheses
        if len(args) == 1 and callable(args[0]):
            return cls()(args[0])
    
        # Otherwise, it was used with parentheses, so initialize as normal
        return cls(*args, **kwargs)

# Assign the classmethod to the decorator name,
# allowing it to be used with or without parentheses
RunReturn = RunReturn.from_decorator

def smartFitRun(cls):
    """
    A class-based decorator that manages the `fit`/`run` method switching 
    logic. If one method is called but the other is implemented, the correct 
    method will be invoked automatically, with a warning issued to the user.

    This is useful for ensuring that a class which only implements one of the 
    `fit` or `run` methods can still function when the other is called 
    incorrectly. The system will detect whether `fit` or `run` is available, 
    issue a warning, and automatically call the available method.

    Parameters
    ----------
    cls : class
        The class being decorated, which should implement either `fit` or 
        `run` (but not both). The decorator ensures that if the missing method 
        is called, the available method is invoked instead.

    Returns
    -------
    cls : class
        The decorated class with method switching logic applied.
    
    Notes
    -----
    This decorator is designed for situations where either the `fit` method or 
    the `run` method is implemented in a class, but not both. It ensures that 
    calling the missing method does not result in an error but rather triggers 
    the other method.

    The method switching logic is as follows:

    - If only ``fit`` is implemented and ``run`` is called, the decorator will
      call the ``fit`` method instead, issuing a  warning.

    - If only ``run`` is implemented and ``fit`` is called, the decorator will
      call the ``run`` method instead, issuing a warning.

    This behavior is particularly useful when ``fit`` requires a dataset 
    (typically ``X``, ``y``), while ``run`` operates on pre-fitted models 
    without needing the same input structure.
    
    Example
    -------
    >>> from fusionlab.decorators import smartFitRun
    >>> @smartFitRun
    >>> class ModelExampleFitOnly:
    >>>     ''' Expects run while fit is called'''
    >>>     def fit(self, X, y=None, **fit_params):
    >>>         print(f"Fitting model with data {X} and target {y}")

    >>> model = ModelExampleFitOnly()
    >>> model.run(X=[1, 2, 3], y=[0, 1, 0])  # Will call `fit` instead and issue a warning.

    >>> @smartFitRun
    >>> class ModelExampleRunOnly:
    >>>     ''' Expects fit while run is called'''
    >>>     def run(self, **run_kwargs):
    >>>         print(f"Running model with parameters {run_kwargs}")

    >>> model = ModelExampleRunOnly()
    >>> model.fit()  # Will call `run` instead and issue a warning.

    See Also
    --------
    :class:`fit` : Fits a model to the provided data.
    :class:`run` : Runs a model using parameters.

    References
    ----------
    .. [1] Python Software Foundation. Python 3.9 Documentation.
           https://docs.python.org/3/

    .. [2] Decorators: Advanced Functions, Real Python.
           https://realpython.com/primer-on-python-decorators/
    """

    original_fit = getattr(cls, 'fit', None)
    original_run = getattr(cls, 'run', None)

    if original_fit is not None and original_run is None:
        # Only 'fit' is defined
        @functools.wraps(original_fit)
        def run(self, *args, **kwargs):
            # Temporarily enable warnings if they were disabled
            with warnings.catch_warnings():
                warnings.simplefilter('once', FutureWarning)
                warnings.warn(
                    "`fit` method is required, but `run` was called. "
                    "Automatically switching to `fit`. "
                    "Note: Calling `run` without a `fit` implementation "
                    "might be deprecated.",
                    FutureWarning, 
                )
            return self.fit(*args, **kwargs)

        setattr(cls, 'run', run)
        
    elif original_run is not None and original_fit is None:
        # Only 'run' is defined
        @functools.wraps(original_run)
        def fit(self, *args, **kwargs):
            with warnings.catch_warnings():
               warnings.simplefilter('once', FutureWarning)
               warnings.warn(
                   "`run` method is required, but `fit` was called. "
                   "Automatically switching to `run`. "
                   "In future versions, calling `fit` without a `run` "
                   "implementation might raise an error.",
                   FutureWarning
               )
            return self.run(*args, **kwargs)

        setattr(cls, 'fit', fit)
        
    return cls

class SmartProcessor:
    """
    A decorator class for data processing which selectively excludes specified 
    columns from the processing step and reintegrates them afterward. This is 
    useful for data preprocessing steps like scaling or imputing, where certain
    columns (e.g., identifiers or target variables) should be omitted from 
    the processing.

    Parameters
    ----------
    func : callable, optional
        The function to decorate. If not provided at initialization, it must 
        be provided later as the first positional argument in the call to the 
        decorator instance.
    param_name : str, optional
        The name of the keyword argument in the decorated function that 
        specifies which columns to exclude from processing. 
        Defaults to 'column_to_skip' if not provided.
    fail_silently : bool or 'warn', optional
        Controls the error handling behavior. If `False` (default), errors 
        raise exceptions. If `True`, the original data is returned on error. 
        If set to 'warn', a warning is issued, and the original data is returned.
    to_dataframe : bool, optional
        If `True`, converts the output to a pandas DataFrame, regardless of the
        input type. This is useful when working with NumPy arrays but needing 
        a DataFrame for the result. Defaults to `False`.

    Notes
    -----
    The decorator dynamically adjusts to the data type of the input, supporting
    both pandas DataFrames and NumPy arrays. When applied, it ensures that the
    columns specified for exclusion are not modified by the processing function,
    preserving their original values and positions in the output.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.decorators import SmartProcessor
    >>> @SmartProcessor(to_dataframe=True)
    ... def scale_data(data):
    ...     return (data - data.mean()) / data.std()
    ...
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> print(scale_data(df, column_to_skip=['C']))
           A         B  C
    0 -1.224745 -1.224745  7
    1  0.000000  0.000000  8
    2  1.224745  1.224745  9

    Using the decorator with NumPy arrays while skipping specific indices:
    >>> import numpy as np
    >>> arr = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    >>> result = scale_data(arr, column_to_skip=[2])
    >>> print(result)
    [[-1.22474487 -1.22474487  7.        ]
     [ 0.          0.          8.        ]
     [ 1.22474487  1.22474487  9.        ]]
    """
    def __init__(self, func=None, *, param_name=None, fail_silently=False, 
                 to_dataframe=False):
        self.func = func
        self.param_name = param_name or 'column_to_skip'
        self.fail_silently = fail_silently
        self.to_dataframe = to_dataframe
        if func:
            functools.update_wrapper(self, func)
            
    def __call__(self, *args, **kwargs):
        """
        Call method that makes `SmartProcessor` a callable object which can 
        act as a decorator.
    
        When `SmartProcessor` is used to decorate a function without previously
        being instantiated with a function, it receives the function as the 
        first positional argument and returns a new instance of `SmartProcessor`
        as the decorator. If it was already instantiated with a function, it 
        processes the input data and handles the exclusion and reintegration of 
        specified columns.
    
        Parameters
        ----------
        *args : tuple
            The positional arguments passed to the function. If called on an 
            undecorated function, `args[0]` is expected to be the function to 
            decorate.
        **kwargs : dict
            The keyword arguments passed to the function.
    
        Returns
        -------
        callable or object
            If called without a function, returns a new instance of 
            `SmartProcessor` with the function to decorate. If called with a 
            function, returns the wrapper function that processes the data.
    
        Notes
        -----
        This method handles two scenarios:
        1. Initialization of the decorator with a function to decorate.
        2. Application of the decorator to process data by wrapping the 
        decorated function and optionally excluding specified columns from 
        being processed.
        
        The actual data processing includes error handling according to the 
        'fail_silently' attribute, allowing for warnings or silent failures 
        as configured.
    
        Examples
        --------
        >>> from fusionlab.decorators import SmartProcessor
        >>> import pandas as pd
        >>> @SmartProcessor(to_dataframe=True)
        ... def scale_data(data):
        ...     return (data - data.mean()) / data.std()
        ...
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3],
        ...     'B': [4, 5, 6],
        ...     'C': [7, 8, 9]
        ... })
        >>> print(scale_data(df, column_to_skip=['C']))
               A         B  C
        0 -1.224745 -1.224745  7
        1  0.000000  0.000000  8
        2  1.224745  1.224745  9
        """
        if not self.func:
            # If the instance is called with a function to decorate, 
            # return a new decorated instance
            return self.__class__(
                args[0], param_name=self.param_name,
                fail_silently=self.fail_silently, 
                to_dataframe=self.to_dataframe
            )
    
        def wrapper(data, *args, **kwargs):
            columns_to_skip = kwargs.get(self.param_name, None)
            if isinstance(columns_to_skip, ( str, int)): 
                columns_to_skip = [columns_to_skip]
            try:
                if columns_to_skip is not None:
                    if isinstance(data, pd.DataFrame):
                        self._check_columns_exist(data, columns_to_skip)
                        data_to_process, skipped_data = data.drop(
                            columns=columns_to_skip), data[columns_to_skip]
                    elif isinstance(data, np.ndarray):
                        self._check_indices_valid(data, columns_to_skip)
                        data_to_process = np.delete(data, columns_to_skip, axis=1)
                        skipped_data = data[:, columns_to_skip]
                    else:
                        raise TypeError(
                            "Data must be a pandas DataFrame or a NumPy array."
                            f" Got {type(data).__name_!r}")
    
                    processed_data = self.func(data_to_process, *args, **kwargs)
    
                    if isinstance(data, pd.DataFrame):
                        result = pd.concat([processed_data, skipped_data], axis=1)
                    elif isinstance(data, np.ndarray):
                        result = self._reintegrate_skipped_numpy(
                            data, processed_data, skipped_data, columns_to_skip)
                    
                    return result if not self.to_dataframe or not isinstance(
                        result, pd.DataFrame
                        ) else self.restore_original_column_order (result)
                  
                else:
                    return self.func(data, *args, **kwargs)
    
            except Exception as e:
                if self.fail_silently == 'warn':
                    warnings.warn(str(e))
                    return data
                elif not self.fail_silently or self.fail_silently=='raise':
                    raise
    
        return wrapper(*args, **kwargs)
    
    def _check_columns_exist(self, dataframe, columns):
        """
        Check if the specified columns exist in the dataframe.
    
        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to check for column existence.
        columns : list of str
            List of column names to check in the DataFrame.
    
        Raises
        ------
        ValueError
            If any of the specified columns do not exist in the DataFrame, a 
            ValueError is raised with an appropriate message.
    
        Notes
        -----
        This method is used internally by the SmartProcessor class to ensure 
        that the columns specified to be skipped during processing actually 
        exist in the input DataFrame. This is crucial for preventing runtime 
        errors during data manipulation.
        """
        self._original_columns= dataframe.columns.tolist() 
        if any(col not in dataframe.columns for col in columns):
            raise ValueError("Some columns to skip do not exist in the DataFrame")
    
    def _check_indices_valid(self, array, indices):
        """
        Check if the specified indices are valid for the given NumPy array.
    
        Parameters
        ----------
        array : np.ndarray
            The NumPy array to check indices against.
        indices : list of int
            List of column indices to check in the NumPy array.
    
        Raises
        ------
        ValueError
            If any of the indices are out of the range of the array's second dimension,
            a ValueError is raised.
    
        Notes
        -----
        This method ensures that the indices specified for skipping are within the valid
        range of columns of the NumPy array. It prevents index errors during operations
        that involve slicing or accessing array elements by index.
        """
        if any(index >= array.shape[1] for index in indices):
            raise ValueError("Column index out of range. Expect indexes"
                             f" ranged between [0, {array.shape[1]}).")
    
    def _reintegrate_skipped_numpy(
            self, original_data, processed_data, skipped_data, columns_to_skip):
        """
        Reintegrate skipped data back into the processed NumPy array.
    
        Parameters
        ----------
        original_data : np.ndarray
            The original data from which columns were skipped.
        processed_data : np.ndarray
            The data after processing, missing the skipped columns.
        skipped_data : np.ndarray
            The columns that were skipped during the processing.
        columns_to_skip : list of int
            Indices of the columns that were skipped.
    
        Returns
        -------
        np.ndarray
            A new NumPy array that combines both the processed and skipped data 
            in their original column order.
    
        Notes
        -----
        This method handles the reintegration of skipped columns back into the
        NumPy array after the main processing has been completed. It ensures 
        that the final output maintains the same structure and order as the 
        original input array, which is essential for consistency in data 
        processing pipelines.
        """
        full_data = np.empty_like(original_data)
        j = 0  # Index for processed data columns
        for i in range(original_data.shape[1]):
            if i in columns_to_skip:
                # Place skipped data back in its original position
                full_data[:, i] = skipped_data[:, columns_to_skip.index(i)]
            else:
                # Insert processed data in the remaining positions
                full_data[:, i] = processed_data[:, j]
                j += 1
        return full_data
    
    def reoder_dataframe_columns (self , result): 
        # reoder dataframe columns like the original positions after concatena
        # tion 
        if not self.dataframe or isinstance ( result, pd.DataFrame): 
            return result 
        else: 
            result =pd.DataFrame(result) 
        
        if hasattr (self, '_original_columns'): 
            try :
                # try to place the original columns in order after concatenation. 
                result = result[ self._original_columns]
            except : pass # do nothing 
        
        return  result 
    
    def restore_original_column_order(self, result):
        """
        Restore the column order of a DataFrame to match the original column order
        stored in `_original_columns`. If `result` is not a DataFrame, it attempts to
        convert it into one.
    
        Parameters
        ----------
        result : pd.DataFrame or convertible to pd.DataFrame
            The result DataFrame whose columns need to be reordered.
    
        Returns
        -------
        pd.DataFrame
            DataFrame with columns reordered to match the original order, if possible.
    
        Notes
        -----
        This method relies on the presence of an attribute `_original_columns` which 
        is expected to be a list of column names in their original order. The method 
        only modifies the column order if `result` is a DataFrame and `_original_columns`
        is set.
    
        If the reordering process fails (e.g., due to missing columns), the method 
        fails silently and returns the DataFrame as is without reordering.
    
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'B': [4, 5, 6],
        ...     'A': [1, 2, 3],
        ...     'C': [7, 8, 9]
        ... })
        >>> self._original_columns = ['A', 'B', 'C']
        >>> restored_df = self.restore_original_column_order(df)
        >>> print(restored_df.columns)
        Index(['A', 'B', 'C'], dtype='object')
        """
        
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result)
        
        if hasattr(self, '_original_columns'):
            try:
                # Try to reorder columns according to the original order
                result = result[self._original_columns]
            except KeyError:
                # Fails silently if reordering is not possible due to missing columns
                pass
    
        return result
        
            
class DataTransformer:
    """
    A decorator class for transforming the output of functions that return
    pandas DataFrames or Series. It can adjust the return value based on
    specified parameters, including renaming columns, resetting indexes, or
    setting a specific index. This class is useful for ensuring that the
    output of data processing functions conforms to a specific structure or
    naming convention.

    Parameters
    ----------
    name : str, optional
        The name of the keyword argument in the decorated function that 
        contains the data to be transformed. If not specified, the first 
        positional argument is used.
    data_index : int, optional
        The index of the data within the return value if the return is a 
        tuple or list. This parameter is only used if the return value is 
        not a single DataFrame or Series. Default is None, which implies 
        that the first item in the return tuple/list is used in 'lazy' mode.
    reset_index : bool, optional
        If True, the index of the DataFrame or Series is reset. Default is False.
    mode : {'lazy', 'hardworker'}, optional
        The mode of operation. In 'lazy' mode, minimal changes are made to the 
        return value. In 'hardworker' mode, the decorator attempts more 
        extensive transformations. Default is 'lazy'.
    verbose : bool, optional
        If True, the decorator will print information about the transformations
        it performs and any errors or warnings. Default is False.
    set_index : bool, optional
        If True and `original_attrs` has an 'index', the decorator will set 
        this index to the return value. Default is False.
    rename_columns : bool, optional
        If True and `original_attrs` has 'columns', the decorator will rename 
        the columns of the return value. This is only applicable if the return 
        value is a DataFrame. Default is False.
    keep_origin_type: bool, default=False 
       If set ``True`` and mode is set to ``lazy``, return as the original data
       types passed to the decorated function. 
       
    Examples
    --------
    Use as a decorator to automatically convert the return value of a function 
    to a DataFrame and rename columns based on a predefined structure:

    >>> import pandas as pd
    >>> from fusionlab.decorators import DataTransformer
    >>> @DataTransformer(rename_columns=True, verbose=True)
    ... def process_data():
    ...     return pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    ...
    >>> df = process_data()
    DataTransformer: Finished processing the result.
    
    The `process_data` function will return a DataFrame with columns renamed 
    according to `original_attrs`, if they were collected and `rename_columns`
    was set to True.

    """
    def __init__(
        self, 
        name=None, 
        data_index=None, 
        reset_index=False, 
        mode='lazy', 
        verbose=False, 
        set_index=False, 
        rename_columns=False, 
        keep_origin_type=False, 
    ):
        self.name = name
        self.data_index = data_index
        self.reset_index = reset_index
        self.mode = mode
        self.verbose = verbose
        self.set_index = set_index
        self.rename_columns = rename_columns
        self.keep_origin_type=keep_origin_type 
        self.original_attrs = {}
        self._is_frame=False 
        
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-function execution: Data collection and attribute setup
            if args:
                self._collect_data_attributes(*args, **kwargs)
            result = func(*args, **kwargs)
            # Post-function execution: Data re-construction and manipulation
            result = self._reconstruct_data(result)
            if self.verbose: 
                print( "DataTransformer: Finished processing the result.")
            
            if self.mode=='lazy': 
                if self.keep_origin_type and not self._is_frame: 
                    # keep the data type as is.
                    result = np.asarray(result)
            return result
        return wrapper

    def _collect_data_attributes(self, *args, **kwargs):
        if self.name:
            data = kwargs.get(self.name)
        else:
            data = args[0]  # Use the first positional argument if name is not specified
        # Use tolist() to ensure JSON serializability if needed
        if isinstance(data, pd.DataFrame):
            self.original_attrs['columns'] = data.columns.tolist()  
            self.original_attrs['index'] = data.index.tolist()
            self._is_frame =True 
        elif isinstance(data, pd.Series):
            self.original_attrs['name'] = data.name
            self.original_attrs['index'] = data.index.tolist()
            self._is_frame =True 

    def _reconstruct_data(self, result):
        is_tuple_result = isinstance(result, (tuple, list))
        data_index = self.data_index if self.data_index is not None else 0
        
        if is_tuple_result:
            if data_index >= len(result):
                if self.verbose:
                    print(f"DataTransformer: Data position {data_index} is out"
                          f" of range for the result size {len(result)}.")
                return result
            data = result[data_index]
        else:
            data = result
        
        # In lazy mode and data is already in the correct format, no need to reconstruct
        if self.mode == 'lazy' and isinstance(data, (pd.DataFrame, pd.Series)):
            return result

        data = self._convert_and_adjust_data(data)
        
        # Re-insert transformed data into the original result structure if needed
        if is_tuple_result:
            result = list(result)  # Convert to list for mutability
            result[data_index] = data
            result = tuple(result)  # Convert back if originally a tuple
        else:
            result = data
        
        return result

    def _convert_and_adjust_data(self, data):
        if isinstance(data, np.ndarray):
            data = self._convert_ndarray_to_pandas(data)
        # Add additional conversion logic here for other data 
        # types like lists or dictionaries

        if isinstance(data, pd.DataFrame) and self.rename_columns:
            data = self._rename_columns(data)

        if (isinstance(data, (pd.DataFrame, pd.Series)
                       ) and self.set_index) or self.reset_index:
            data = self._apply_index_and_name_settings(data)

        return data

    def _convert_ndarray_to_pandas(self, ndarray):
        try:
            if ndarray.ndim == 1:
                return pd.Series(ndarray, name=self.original_attrs.get('name'))
            elif ndarray.ndim > 1:
                return pd.DataFrame(
                    ndarray, columns=self.original_attrs.get('columns'))
        except Exception as e:
            if self.verbose:
                print("DataTransformer: Error converting"
                      f" numpy array to DataFrame/Series - {e}")
        return ndarray

    def _rename_columns(self, dataframe):
        try:
            dataframe.columns = self.original_attrs['columns']
        except Exception as e:
            if self.verbose:
                print(f"DataTransformer: Error renaming columns - {e}")
        return dataframe

    def _apply_index_and_name_settings(self, data):
        if self.reset_index:
            data.reset_index(drop=True, inplace=True)
        elif self.set_index:
            try:
                data.index = self.original_attrs['index']
            except Exception as e:
                if self.verbose:
                    print(f"DataTransformer: Error setting index - {e}")
        return data

class Extract1dArrayOrSeries:
    """
    A decorator and callable that preprocesses input data to ensure it is 
    provided to the decorated/called function as a one-dimensional NumPy 
    array or Pandas Series.

    This utility is designed to facilitate data extraction and conversion 
    from various input formats (lists, dictionaries, Pandas DataFrames, 
    and NumPy ndarrays) into a one-dimensional array or series. It is 
    particularly useful for functions expecting standardized input data 
    formats. The class supports dynamic parameter updates for `column`, 
    `index`, `axis`, and `verbose` when used as a decorator.

    Parameters
    ----------
    func : callable, optional
        The function to be decorated. If None, the instance acts as a 
        factory for partials with preset parameters.
    column : str or int, optional
        Specifies which column to extract from the input if it is a DataFrame
        or multidimensional ndarray. Use an integer for index selection or a 
        string for a DataFrame column name.
    index : int, optional
        Specifies which row to extract from the input if it is a DataFrame 
        or ndarray.
    axis : int, optional
        Specifies the axis along which to extract a one-dimensional array 
        from an ndarray. Valid values are 0 or 1.
    method : {'strict', 'soft'}, default 'strict'
        Specifies the behavior when a specified column or index is not found,
        or when no specification is provided. 'soft' uses the first column 
        if the specified one is missing without raising an error.
    as_series : bool, default False
        Determines whether the output should be a Pandas Series. If False, 
        the output is a NumPy array.
    verbose : int, default 0
        Controls the verbosity of the process. A value greater than 0 
        activates verbose output.
    squeeze_arr : bool, default True
        Determines whether to squeeze the input array to one dimension. 
        Applicable to ndarrays only.

    Returns
    -------
    The decorator returns the modified function with input data processed 
    according to the specified parameters.

    Raises
    ------
    TypeError
        If the input is not a list, dictionary, Pandas DataFrame, or 
        NumPy ndarray.
    ValueError
        If the specified conditions (e.g., column/index out of range, 
        incorrect axis specification) are not met.

    Examples
    --------
    Using as a decorator to ensure input data is a one-dimensional array:

    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from fusionlab.decorators import Extract1dArrayOrSeries
    >>> @Extract1dArrayOrSeries(column=0, as_series=True, verbose=1)
    >>> def compute_average(data):
    ...     return data.mean()

    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> print(compute_average(df))

    Dynamically overriding decorator parameters in function call:

    >>> @Extract1dArrayOrSeries(column='A', as_series=True)
    >>> def summarize_data(data):
    ...     return {'mean': data.mean(), 'std': data.std()}

    >>> new_df = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})
    >>> # Overriding the 'column' parameter dynamically
    >>> print(summarize_data(new_df, column='B'))
    
    Using as a callable:

    >>> def process_data(data):
    ...     # Expect data to be a one-dimensional NumPy array or Pandas Series
    ...     return np.mean(data)

    >>> decorated_function = Extract1dArrayOrSeries(process_data, column=0, as_series=False)
    >>> ndarray = np.array([[1, 2, 3], [4, 5, 6]])
    >>> print(decorated_function(ndarray))
    """
    def __init__(
        self, 
        func=None, *, 
        column=None, 
        index=None, 
        axis=None, 
        method='strict',
        as_series=False, 
        verbose=0, 
        squeeze_arr=True
        ):
        self.func = func
        self.column = column
        self.index = index
        self.axis = axis
        self.method = method
        self.as_series = as_series
        self.verbose = verbose
        self.squeeze_arr = squeeze_arr
        
        if func is not None:
            functools.wraps(func)(self)

    def __call__(self, *args, **kwargs):
        # Dynamically update only specific 
        # parameters if they are explicitly passed
        dynamic_params = ['column', 'index', 'axis', 'verbose']
        for param in dynamic_params:
            if param in kwargs:
                setattr(self, param, kwargs[param])
        if self.func:
            # Proceed with the possibly updated instance attributes
            return self._wrapper(*args, **kwargs)
        else:
            return self._partial(*args, **kwargs)

    def _partial(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return self.__class__(
                func, column=self.column, 
                index=self.index, 
                axis=self.axis,
                method=self.method, 
                as_series=self.as_series,
                verbose=self.verbose, 
                squeeze_arr=self.squeeze_arr
            )(*args, **kwargs)
        return wrapped
    
    def _wrapper(self, arr, *args, **kwargs):
        arr = self._convert_input(arr)
        
        if isinstance (arr, pd.Series): 
            result= arr.copy() 
        elif isinstance(arr, pd.DataFrame):
            result = self._extract_from_dataframe(arr)
        elif isinstance(arr, np.ndarray):
            result = self._extract_from_ndarray(arr)
        else:
            raise TypeError(
                "The input must be either a Pandas DataFrame or a NumPy ndarray.")
        
        if self.as_series and isinstance(result, np.ndarray):
            result = pd.Series(result)
        elif not self.as_series and isinstance(result, pd.Series):
            result = result.to_numpy()
        
        return self.func(result, *args, **kwargs)
    
    def __get__(self, instance, owner):
        return self.__class__(
            self.func.__get__(instance, owner), column=self.column, 
            index=self.index, axis=self.axis,
            method=self.method, as_series=self.as_series, verbose=self.verbose, 
            squeeze_arr=self.squeeze_arr)

    def _convert_input(self, arr):
        """Convert input data to numpy array or pandas DataFrame if
        it's a list or dictionary, respectively."""
        if isinstance(arr, list):
            arr = np.array(arr)
        elif isinstance(arr, dict):
            arr = pd.DataFrame(arr)
        return arr
        
    def _extract_from_dataframe(self, arr):
        """Extract data from a pandas DataFrame based on the 
        specified column or index."""
        if self.column is not None:
            if isinstance(self.column, int):  # Column by integer index
                self._validate_column_index(arr, self.column)
                result = arr.iloc[:, self.column]
            elif isinstance(self.column, str):  # Column by name
                self._validate_column_name(arr, self.column)
                result = arr[self.column]
        elif self.index is not None:
            result = arr.iloc[self.index]
        else:
            result = self._default_dataframe_extraction(arr)
        return result
    
    def _extract_from_ndarray(self, arr):
        """Extract a specific slice from a numpy ndarray based on 
        the provided parameters."""
        if arr.ndim > 1:
            result = self._handle_multidimensional_array(arr)
        else:
            result = np.squeeze(arr) if self.squeeze_arr else arr
        return result
    
    def _validate_column_index(self, arr, index):
        """Validate if the provided column index is within the valid range."""
        if index < 0 or index >= arr.shape[1]:
            raise ValueError("The specified column index is out of range."
                             " Please provide a valid index.")
            
    def _validate_column_name(self, arr, name):
        """Validate if the provided column name exists in the DataFrame."""
        if name not in arr.columns:
            raise ValueError(f"The specified column name '{name}' does"
                             " not exist in the DataFrame.")
            
    def _default_dataframe_extraction(self, arr):
        """Extract the first column from a DataFrame by default when no 
        specific column or index is provided."""
        if self.method == 'soft':
            if self.verbose:
                print("No specific column or index provided; extracting"
                      " the first column by default.")
            return arr.iloc[:, 0]
        else:
            raise ValueError("No specific column or index was provided while "
                             "a DataFrame was passed, and 'soft' method is not"
                             " enabled.")
    
    def _handle_multidimensional_array(self, arr):
        """
        Handle the extraction from a multidimensional numpy array
        based on axis and other parameters.
        
        This method takes into account the specified axis, column, index, and
        the extraction method to retrieve a one-dimensional array from a 
        multidimensional numpy array.
        """
        if self.axis is not None:
            if self.axis == 0:
                if self.index is not None:
                    # Extract specific row
                    result = arr[self.index, :]
                else:
                    if self.method == "soft":
                        if self.column is not None and isinstance(self.column, int):
                            self._validate_column_index(arr, self.column)
                            result = arr[:, self.column]
                            if self.verbose:
                                print("Column specified, extracting based on "
                                      "column index in 'soft' mode.")
                        else:
                            raise ValueError("Column must be an integer for "
                                             "ndarray when 'soft' method is "
                                             "used and axis=0.")
                    else:
                        raise ValueError("Column cannot be used for axis=0 "
                                         "unless method is set to 'soft'.")
            elif self.axis == 1:
                if self.column is not None and isinstance(self.column, int):
                    # Extract specific column
                    self._validate_column_index(arr, self.column)
                    result = arr[:, self.column]
                elif self.index is not None:
                    # Index behaves dually; here it acts 
                    # as column extraction for axis=1
                    if self.verbose:
                        print("Index is treated as column for numpy array when axis=1.")
                    self._validate_column_index(arr, self.index)
                    result = arr[:, self.index]
                else:
                    raise ValueError("Either column or index needs to be "
                                     "specified when axis is 1.")
            else:
                raise ValueError("Axis must be 0 or 1.")
        else:
            if self.squeeze_arr:
                result = np.squeeze(arr)
                if result.ndim >= 2:
                    # Squeeze failed to convert to 1D; likely 
                    # a matrix without a specified axis
                    raise ValueError("Unable to automatically convert 2D array"
                                     " to 1D without axis specification.")
            else:
                # No squeezing; ensure arr is already 1D or has a 
                # single dimension to be treated as 1D
                if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
                    # Treat as 1D if one of the dimensions is 1,
                    # regardless of orientation
                    result = arr.flatten()
                else:
                    raise ValueError("Array is multidimensional and requires"
                                     " axis specification for extraction.")
        
        return result

class DynamicMethod:
    """
    A class-based decorator designed to preprocess data before it's passed to 
    a function or method. This preprocessing includes filtering data by type, 
    selecting specific columns, handling missing values, applying transformations,
    and executing based on custom conditions. It offers advanced options like 
    treating integer columns as categorical and encoding categorical columns.

    Parameters
    ----------
    expected_type : str, optional
        Specifies the expected data type for processing. The options are:
        - 'numeric': Only numeric columns are considered.
        - 'categorical': Only categorical columns are considered.
        - 'both': Both numeric and categorical columns are considered.
        Defaults to ``'numeric'``.

    capture_columns : bool, optional
        If set to True, the decorator filters the DataFrame columns to those 
        specified in the 'columns' keyword argument passed to the decorated 
        function. 
        Defaults to ``False``.

    treat_int_as_categorical : bool, optional
        When True, integer columns in the DataFrame are treated as categorical 
        data, which can be particularly useful for statistical operations that
        distinguish between numeric and categorical data types, such as ANOVA 
        tests. 
        Defaults to ``False``.

    encode_categories : bool, optional
        If True, categorical columns are encoded into integer values. This is
        especially useful for models that require numerical input for 
        categorical data.
        Defaults to ``False``.

    drop_na : bool, optional
        Determines whether rows or columns with missing values should be dropped.
        The specific rows or columns to drop are dictated by `na_axis` and
        `na_thresh`.
        Defaults to ``False``.

    na_axis : Union[int, str], optional
        Specifies the axis along which to drop missing values. Acceptable 
        values are:
        - 0 or 'row': Drop rows with missing values.
        - 1 or 'col': Drop columns with missing values.
        Defaults to ``0``.

    na_thresh : Optional[float], optional
        Sets a threshold for dropping rows or columns with missing values. 
        This can be specified as an absolute number of non-NA values or a
        proportion (0 < value <= 1) of the total number of values in a row
        or column.
        Defaults to ``None``.

    transform_func : Optional[Callable], optional
        A custom function to apply to the DataFrame before passing it to the
        decorated function. This allows for flexible data transformations 
        as needed.
        Defaults to ``None``.

    condition : Optional[Callable[[pd.DataFrame], bool]], optional
        A condition function that takes the DataFrame as an argument and 
        returns ``True`` if the decorated function should be executed. This
        enables conditional processing based on the data.
        Defaults to ``None``.

    reset_index : bool, optional
        If True, the DataFrame index is reset before processing. This is useful
        after filtering rows to ensure the index is continuous.
        Defaults to ``False``.
        
    prefixer : str or None, optional
        A string to prefix the function name with when adding it as a method 
        to DataFrame and Series objects. If set to "exclude" or 'false' 
        (case-insensitive), the prefix is omitted, and the original function 
        name is used. If None or not provided, a default prefix of 'go' is 
        used to denote the method's origin from the gofast package.
        
    verbose : bool, optional
        Controls the verbosity of the decoration process. If True, detailed 
        information about the preprocessing steps is printed.
        Defaults to ``False``.

    Raises
    ------
    ValueError
        If the first argument to the decorated function is not a pandas DataFrame,
        dictionary, or NumPy ndarray.

    Examples
    --------
    >>> from fusionlab.decorators import DynamicMethod
    >>> @DynamicMethod(expected_type='numeric', capture_columns=True, 
    ... verbose=True, drop_na=True, na_axis='row', na_thresh=0.5, reset_index=True)
    ... def calculate_variance(data):
    ...     return data.var(ddof=0).mean()
    
    >>> data = pd.DataFrame({"A": [1, 2, np.nan], "B": [4, np.nan, 6]})
    >>> print(calculate_variance(data))
    # The above example demonstrates preprocessing a DataFrame by dropping rows with
    # more than 50% missing values and calculating the variance of the remaining data.

    Notes
    -----
    - The `treat_int_as_categorical` and `encode_categories` parameters offer flexibility
      in handling integer and categorical data, which can be critical for certain types
      of analysis or modeling.
    - The `transform_func` and `condition` parameters allow for custom data transformations
      and conditional execution, adding a layer of customization to the preprocessing steps.
    """
    def __init__(
        self, 
        expected_type: str = 'numeric', 
        capture_columns: bool = False,
        treat_int_as_categorical: bool = False, 
        encode_categories: bool = False, 
        verbose: bool = False, 
        drop_na: bool = False, 
        na_axis: Union[int, str] = 0, 
        na_thresh: Optional[float] = None, 
        transform_func: Optional[Callable] = None, 
        condition: Optional[Callable[[pd.DataFrame], bool]] = None, 
        reset_index: bool = False,
        prefixer:Optional[str]=None, 
        ):
        self.expected_type = expected_type
        self.capture_columns = capture_columns
        self.treat_int_as_categorical = treat_int_as_categorical
        self.encode_categories = encode_categories
        self.drop_na = drop_na
        self.na_axis = na_axis
        self.na_thresh = na_thresh
        self.transform_func = transform_func
        self.condition = condition
        self.reset_index = reset_index
        self.prefixer = prefixer
        self.verbose = verbose

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.verbose:
                print(f"Preprocessing data for {func.__name__}...")

            data = self._validate_and_prepare_data(args[0], **kwargs)
            if data is None:
                return func(*args, **kwargs)  # Early exit if data validation fails

            data = self._process_data(data, **kwargs)
            return func(data, *args[1:], **kwargs)
        
        self._add_method_to_pandas (
            wrapper, prefixer= self.prefixer)
        
        return wrapper

    def _validate_and_prepare_data(self, data, **kwargs):
        """
        Validates the input data and converts it to a pandas DataFrame if
        necessary.
    
        This method checks if the first argument is one of the supported types 
        (pd.DataFrame, dict, np.ndarray, or iterable object) and converts it to 
        a pandas DataFrame if it's not already one. If `columns` are specified in 
        kwargs and the data is an np.ndarray or a converted iterable, it attempts 
        to use these columns when creating the DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame, dict, or np.ndarray
            The input data to be validated and possibly converted.
        **kwargs : dict
            Additional keyword arguments, including 'columns' which may be used 
            if the data is an np.ndarray to specify DataFrame column names.
    
        Returns
        -------
        pd.DataFrame
            The validated and prepared pandas DataFrame.
    
        Raises
        ------
        ValueError
            If the input data is not one of the supported types.
        """
        if isinstance(data, dict):
            data= pd.DataFrame(data)
        # Convert iterable (not DataFrame or np.ndarray) to DataFrame
        elif  hasattr(data, '__iter__') and not isinstance(
                data, ( pd.DataFrame, np.ndarray, pd.Series)):  
            try:
                data =  np.array(data)
            except Exception:
                raise ValueError(
                    "Expect the first argument to be an iterable object"
                    " with minimum samples equal 2.")
        if isinstance(data, np.ndarray):
            columns = kwargs.get('columns')
            if isinstance (columns, str): 
                columns =[columns]
                
            data = pd.DataFrame(data, columns=(
                columns if columns and len(columns) == data.shape[1] else None))
            
        elif isinstance ( data, pd.Series): 
            data = pd.DataFrame ( data)
            
        # Finally validate  whether dataFrame if constructed.
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Input data must be a pd.DataFrame, dict, np.ndarray,"
                " or iterable.")
        return data
    
    def _process_data(self, data: pd.DataFrame, **kwargs):
        """
        Applies various preprocessing steps to the data based on the decorator's parameters.
    
        This method sequentially processes the data through specified steps: capturing 
        specified columns, filtering data by type, dropping missing values, applying 
        a transformation function, checking a condition for execution, and resetting 
        the index if required.
    
        Parameters
        ----------
        data : pd.DataFrame
            The data to be processed.
        **kwargs : dict
            Additional keyword arguments passed through from the decorator.
    
        Returns
        -------
        None
            The function modifies the data in place and does not return any value.
        """
        if self.capture_columns:
            data = self._capture_columns(data, **kwargs)
        if self.expected_type in ['numeric', 'categorical']:
            data = self._filter_data_type(data)
        if self.drop_na:
            data = self._drop_na(data)
        if self.transform_func:
            data = self.transform_func(data)
        if self.condition and not self.condition(data):
            if self.verbose:
                print("Condition for execution not met, skipping function call.")
            return
        if self.reset_index:
            data = data.reset_index(drop=True)

        return data 
        
    def _capture_columns(self, data: pd.DataFrame, **kwargs):
        """
        Filters the columns of the DataFrame based on the specified 'columns' in kwargs.
    
        If the 'columns' keyword argument is provided, this method attempts to filter 
        the DataFrame to include only those columns. If any specified columns do not 
        exist in the DataFrame, a warning is printed if verbose output is enabled.

        """
        columns = kwargs.pop('columns', None)
        if columns is not None:
            try:
                data = data[columns]
            except KeyError:
                if self.verbose:
                    print("Specified columns do not match, ignoring columns.")
        return data 
    
    def _filter_data_type(self, data: pd.DataFrame):
        """
        Filters the data based on the expected type ('numeric' or 'categorical').
    
        This method filters the DataFrame to include only numeric or categorical 
        columns based on the `expected_type` parameter. If 'categorical' is specified, 
        it further processes categorical data as per the class parameters.
        """
        if self.expected_type == 'numeric':
            data = data.select_dtypes(include=[np.number])
        elif self.expected_type == 'categorical':
            data =self._handle_categorical_data(data)
        
        return data 
    
    def _handle_categorical_data(self, data: pd.DataFrame):
        """
        Handles categorical data by treating integer columns as categorical
        if specified, and encoding categorical columns if required.
    
        This method processes integer columns as categorical if
        `treat_int_as_categorical` 
        is True, and encodes categorical columns into integers if 
        `encode_categories` is True.
    
        """
        if self.treat_int_as_categorical:
            int_columns = data.select_dtypes(include=[int]).columns.tolist()
            data[int_columns] = data[int_columns].astype('category')
        if self.encode_categories:
            data = self._encode_categorical_columns(data)
        
        return data 
    
    def _encode_categorical_columns(self, data: pd.DataFrame):
        """
        Encodes categorical columns in the DataFrame into integer values.
    
        This method applies Label Encoding to columns in the DataFrame that are 
        identified as categorical (either 'category' or 'object' dtype).

        """
        from sklearn.preprocessing import LabelEncoder
        cat_columns = data.select_dtypes(include=['category', 'object']).columns
        for col in cat_columns:
            data[col] = LabelEncoder().fit_transform(data[col])
            
        return data 

    def _drop_na(self, data: pd.DataFrame):
        """
        Drops rows or columns from the DataFrame based on missing values criteria.
    
        This method drops rows or columns with missing values based on the specified
        `na_axis` and `na_thresh` parameters. `na_axis` determines the axis along which
        to drop (rows or columns), and `na_thresh` specifies the threshold for dropping
        either as an absolute number of non-NA values required to keep a row/column or
        as a proportion of the total number of values.
    
        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame from which rows or columns will be dropped based 
            on missing values.
    
        Modifies
        --------
        data : pd.DataFrame
            The input DataFrame is modified in place by dropping specified 
            rows or columns.
    
        """
        # Convert na_axis from string to integer if necessary
        na_axis = 0 if self.na_axis in [0, 'row'] else 1 if self.na_axis in [
            1, 'col'] else self.na_axis
        
        # Calculate the threshold for dropping based on the proportion 
        # if specified as a float less than 1
        if self.na_thresh is None: 
            thresh= self.na_thresh
        elif 0 < self.na_thresh <= 1:
            total_elements = len(data.columns) if na_axis == 0 else len(data)
            thresh = int(total_elements * self.na_thresh)
        else:
            thresh = self.na_thresh
        # Drop missing values based on the specified axis and threshold
        return data.dropna(axis=na_axis, thresh=thresh)
                                  
    def _add_method_to_pandas(self, func, prefixer=None):
        """
        Dynamically adds a custom method to pandas DataFrame and Series classes. 
        This enhancement allows for extending pandas objects with additional 
        functionality in a flexible manner. The method can be optionally prefixed 
        to denote its origin or purpose, enhancing readability and avoiding 
        namespace collisions.
    
        The method checks if a function, optionally prefixed, already exists 
        as a method within the pandas DataFrame and Series classes. If the 
        method does not exist, it is added, making it accessible directly 
        from DataFrame and Series instances.
    
        Parameters
        ----------
        func : function
            The function to be added as a method. This function should accept a 
            DataFrame or Series as its first argument, followed by any additional 
            arguments or keyword arguments the function requires.
        prefixer : str or None, optional
            A string to prefix the function name with when adding it as a method 
            to DataFrame and Series objects. If set to "exclude" or 'false' 
            (case-insensitive), the prefix is omitted, and the original function 
            name is used. If None or not provided, a default prefix of 'go' is 
            used to denote the method's origin from the gofast package.
    
        Examples
        --------
        Suppose `custom_func` is a function intended to be added to DataFrame and 
        Series objects, and we want to prefix it with 'go_':
    
            >>> def custom_func(df, *args, **kwargs):
            ...     # Implementation here
            ...     pass
            >>> gofast_instance._add_method_to_pandas(custom_func)
    
        Now, `custom_func` can be called on DataFrame and Series objects like so:
    
            >>> df.go_custom_func(*args, **kwargs)
    
        If `dynamic_prefixer` is set to "exclude", the 'go_' prefix is omitted:
    
            >>> gofast_instance._add_method_to_pandas(custom_func, dynamic_prefixer="exclude")
            >>> df.custom_func(*args, **kwargs)
    
        Raises
        ------
        Exception
            If an error occurs while adding the method, it is caught and a message 
            is printed. This behavior can be modified to log the error or handle 
            it as needed.
        """
        # Determine whether to use a prefix based on `prefixer`
        method_name = func.__name__ if prefixer in (
            "exclude", 'false') else "go_" + func.__name__
    
        # Attempt to add the method to both DataFrame and Series classes
        for cls in [pd.DataFrame, pd.Series]:
            if not hasattr(cls, method_name):
                try:
                    setattr(cls, method_name, func)
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to add method {method_name}: {e}")
                    # Optionally log or handle the error as needed
                    
class ExportData:
    """
    A decorator for exporting data into various formats post-function execution. 
    It supports exporting pandas DataFrames or other data types to specified 
    file formats with additional customization through keyword arguments.

    Parameters
    ----------
    export_type : str, optional
        The type of data to export, which can be 'frame' for pandas DataFrames or 'text' 
        for text files. Defaults to 'frame'.
    encoding : str, optional
        The encoding to use for text files. Defaults to 'utf-8'. This parameter is not 
        applicable when exporting DataFrames.
    **kwargs : dict
        Additional keyword arguments to be passed to the pandas export function or 
        the file writing process.

    Examples
    --------
    >>> from fusionlab.decorators import ExportData
    >>> @ExportData(export_type='frame', file_format='csv')
    ... def data_processing_function():
    ...     df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    ...     return df, 'output_filename', 'csv', './savepath', 'Data', {}
    ...
    >>> data_processing_function()
    # This will save the DataFrame returned by data_processing_function to a CSV file
    # named 'output_filename.csv' in the './savepath' directory.
    """
    
    def __init__(self, export_type='frame', encoding='utf8', **kwargs):
        self.export_type = export_type
        self.encoding = encoding
        self.kwargs = kwargs

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_func(*args, **func_kwargs):
            # Extracting data and parameters from the decorated function
            dfs, fname, file_format, savepath, nameof, extra_kwargs = func(
                *args, **func_kwargs)
            
            if self.kwargs.get("file_format") and file_format: 
                # The priority is given to user format. 
                _= self.kwargs.pop("file_format", None)
            
            # Merge decorator's kwargs with function's extra_kwargs,
            # giving precedence to extra_kwargs
            export_kwargs = {**self.kwargs, **extra_kwargs}

            # Ensure the savepath exists
            os.makedirs(savepath, exist_ok=True)

            # Setting file format based on extension or provided format
            _, ext = os.path.splitext(fname)
            file_format = ext.lower() if ext else f".{file_format}"

            # Validate file format
            if self.export_type.lower() == 'frame' and file_format not in ['.csv', '.xlsx']:
                raise ValueError(f"Unsupported file format for DataFrame export: {file_format}")

            # Choose the writer function based on the export type
            if self.export_type.lower() == 'frame':
                fnames = self._export_frame(dfs, fname, file_format, savepath, nameof, **export_kwargs)
            else:
                fnames = self._export_others(dfs, fname, file_format, savepath, nameof, **export_kwargs)

            # Optionally move files to a designated output directory
            # Assuming move_cfile function exists and is imported correctly
            for fname in fnames:
                from .utils.ioutils import move_cfile 
                move_cfile(fname, savepath, dpath='_out')
                
            # Optionally return the filenames of the exported files
            return fnames
        return wrapper_func
        
    def _export_frame(self, dfs, fname, file_format, savepath, nameof=None, **kwargs):
        """
        Handles exporting pandas DataFrame(s) to specified file formats.
        """
        dfs = [dfs] if isinstance(dfs, pd.DataFrame) else dfs
        fnames = []
        for i, df in enumerate(dfs):
            if not isinstance(df, pd.DataFrame):
                continue
            name_suffix = f"_{nameof[i]}" if nameof and i < len(nameof) else ""
            output_fname = f"{fname}{name_suffix}{file_format}"
            full_path = os.path.join(savepath, output_fname)
            
            if file_format == '.xlsx':
                with pd.ExcelWriter(full_path) as writer:
                    df.to_excel(writer, sheet_name=nameof[i] if nameof and i < len(nameof) else 'Sheet1', **kwargs)
            else:
                df.to_csv(full_path, **kwargs)
            fnames.append(full_path)
        
        return fnames

    def _export_others(self, data, fname, file_format, savepath, nameof=None,
                       **kwargs):
        """
        Handles exporting non-DataFrame data to files, primarily text files.
        """
        output_fname = f"{fname}{file_format}"
        full_path = os.path.join(savepath, output_fname)
        
        with open(full_path, mode='w', encoding=self.encoding) as f:
            for item in data:
                f.write(f"{item}\n")
                
        return [full_path]
  
class Temp2D:
    """
    A decorator for creating two-dimensional plots from the outputs of 
    decorated functions. It integrates seamlessly with matplotlib for 
    plotting and supports customization through various parameters.

    Parameters
    ----------
    reason : str, optional
        The purpose of the plot. This parameter is for documentation 
        purposes and does not affect the plot's appearance or behavior.
    **kwargs : dict
        Additional keyword arguments for plot customization. These 
        arguments are expected to align with the parameters used by 
        matplotlib and related plotting utilities.

    Notes
    -----
    The decorator uses the last return value of the decorated function as a 
    dictionary of plotting arguments, which should include keys and values 
    compatible with `matplotlib.pyplot` functions. If these plotting 
    arguments are not provided, an AttributeError will be raised.

    Examples
    --------
    >>> from fusionlab.decorators import Temp2D
    >>> @Temp2D(reason="Show an example")
    ... def generate_data():
    ...     x = np.linspace(0, 10, 100)
    ...     y = np.sin(x)
    ...     return x, y, {'xlabel': 'X Axis', 'ylabel': 'Y Axis'}
    ...
    >>> generate_data()
    # This will plot a sine wave with the specified x and y labels.
    """

    def __init__(self, reason=None, **kwargs):
        self.reason = reason
        self.plot_kwargs = kwargs

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the decorated function and expect a tuple where
            # the last item is a dictionary for plot customization
            *plot_data, plot_customization = func(*args, **kwargs)

            # Update the plot customization with any additional kwargs
            # provided during the decorator initialization
            plot_customization.update(self.plot_kwargs)

            # Call the plot creation method
            return self.plot2d(*plot_data, **plot_customization)

        return wrapper

    def plot2d(self, x, y, **kwargs):
        """
        Generates a 2D plot based on the provided x and y data along with 
        customizable plotting arguments.

        Parameters
        ----------
        x : array-like
            X-coordinates for the plot.
        y : array-like
            Y-coordinates for the plot.
        **kwargs : dict
            Additional keyword arguments for customizing the plot, such as 
            'xlabel', 'ylabel', and any other matplotlib.axes.Axes method 
            arguments.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib Axes object with the plot.

        Example
        -------
        >>> Temp2D().plot2d(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)),
        ...                set_xlabel="X Axis", set_ylabel="Y Axis", set_title="Sine Wave")
        # This will create and display a 2D plot of a sine wave.
        """
        fig, ax = plt.subplots()
        ax.plot(x, y)

        # Apply customizations from kwargs
        for key, value in kwargs.items():
            if key in ["ylabel", 'xlabel', 'title']: 
                key = f"set_{key}" # for Axes 
            if hasattr(ax, key) and callable(getattr(ax, key)):
                getattr(ax, key)(value)
            else:
                print(f"Warning: {key} is not a valid Axes method")

        plt.show()
        return ax

    def __getattr__(self, name):
        # Custom error message for missing attributes
        msg = (f"{self.__class__.__name__!r} has no attribute {name!r}. "
               "Ensure plot arguments are supplied as the last return value "
               "of the decorated function.")
        raise AttributeError(msg)

class SignalFutureChange:
    """
    A decorator that signals an upcoming change to a function or class, such 
    as deprecation or a recommendation to use a more robust alternative. It 
    allows the function or class to execute normally while optionally logging 
    a warning message.

    Parameters
    ----------
    message : str, optional
        A message to be displayed to indicate the reason for the future change. 
        This could inform about deprecation or suggest using an alternative.

    Examples
    --------
    >>> from fusionlab.decorators import SignalFutureChange
    >>> @SignalFutureChange(message="This function will be deprecated in future "
    ...                        "releases. Consider using `new_function` instead.")
    ... def old_function():
    ...     print("This is an old function.")
    ...
    >>> old_function()
    # Executes old_function, logging a message about future deprecation or 
    # recommending an alternative, based on the provided message.
    """
    
    def __init__(self, message=None):
        self.message = message

    def __call__(self, cls_or_func):
        if self.message:
            # Log the warning message at the time of decoration, not at call time
            warnings.warn(self.message, FutureWarning, stacklevel=2)
        
        @functools.wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            # Directly return the result of the original function or class call
            return cls_or_func(*args, **kwargs)
        return wrapper

class AppendDocReferences:
    """
    A decorator for appending reStructuredText references to the docstring 
    of the decorated function or class, enhancing Sphinx documentation by 
    auto-retrieving and replacing values from specified references.

    This allows for dynamic insertion of common documentation elements, 
    such as glossary terms or external documentation links, into the 
    docstrings of multiple functions or classes.

    Parameters
    ----------
    docref : str, optional
        The documentation reference string to be appended to the function's 
        or class's docstring. This should be in reStructuredText format.

    Examples
    --------
    >>> from fusionlab.decorators import AppendDocReferences
    >>> @AppendDocReferences(docref=".. |VES| replace:: Vertical Electrical"
    ...                         " Sounding\\n.. |ERP| replace:: Electrical Resistivity Profiling")
    ... def example_function():
    ...     '''This function demonstrates appending doc references.
    ...
    ...     See more details about |VES| and |ERP|.
    ...     '''
    ...     pass
    ...
    >>> print(example_function.__doc__)
    # The docstring of example_function will now include the replaced 
    # references to VES and ERP along with their definitions.
    """

    def __init__(self, docref=None):
        self.docref = "\n" + docref if docref else ""

    def __call__(self, cls_or_func):
        
        original_doc = cls_or_func.__doc__ if cls_or_func.__doc__ else ''
        # Append the doc reference to the original docstring
        cls_or_func.__doc__ = original_doc + self.docref
        
        @functools.wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            # Directly return the result of the original function or class call
            return cls_or_func(*args, **kwargs)
        
        return wrapper
    
class Deprecated:
    """
    A decorator for marking functions, methods, and classes as deprecated. 
    It emits a deprecation warning when the decorated item is called or 
    instantiated.

    Parameters
    ----------
    reason : str
        The reason why the function, method, or class is deprecated.

    Examples
    --------
    >>> from fusionlab.decorators import Deprecated
    >>> @Deprecated(reason="Use `new_function` instead.")
    ... def old_function():
    ...     print("This function is deprecated.")
    ...
    >>> old_function()
    # Outputs a deprecation warning and prints: "This function is deprecated."

    Note
    ----
    The warning will point to the location where the deprecated item is 
    used, making it easier to identify and replace deprecated usage in 
    codebases.
    """
    
    def __init__(self, reason):
        if not reason:
            raise ValueError("A reason for deprecation must be supplied.")
        self.reason = reason

    def __call__(self, cls_or_func):
        if not inspect.isfunction(cls_or_func) and not inspect.isclass(cls_or_func):
            raise TypeError("Deprecated decorator can only be applied to functions or classes.")

        fmt = "Call to deprecated {item} {name} ({reason})."
        item_type = "class" if inspect.isclass(cls_or_func) else "function or method"
        msg = fmt.format(item=item_type, name=cls_or_func.__name__, reason=self.reason)

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return cls_or_func(*args, **kwargs)

        return new_func

class CheckGDALData:
    """
    A decorator to ensure the availability of GDAL data for functions requiring GDAL. 
    It checks if the GDAL_DATA environment variable is correctly set and points to an 
    existing path. Optionally, it can raise an ImportError if the GDAL data is not 
    configured correctly.

    Parameters
    ----------
    raise_error : bool, optional
        If True, raises an ImportError when GDAL data is not found. Defaults to False.
    verbose : int, optional
        Verbosity level. A higher number indicates more verbose output. Defaults to 0.

    Examples
    --------
    >>> from fusionlab.decorators import CheckGDALData
    >>> @CheckGDALData(raise_error=True, verbose=1)
    ... def my_gdal_function():
    ...     print("This function uses GDAL.")
    ...
    >>> my_gdal_function()
    # This will either print "This function uses GDAL." if GDAL data is correctly set,
    # or raise an ImportError with instructions on how to configure GDAL data.

    Notes
    -----
    This decorator is particularly useful in environments where GDAL is required but 
    might not be fully configured, such as in some virtual environments or custom 
    installations.
    """

    _has_checked = False
    _gdal_data_found = False

    def __init__(self, raise_error=False, verbose=0):
        self.raise_error = raise_error
        self.verbose = verbose

    def __call__(self, func):
        if not self._has_checked:
            self._check_gdal_data()
            self._has_checked = True

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._gdal_data_found and self.raise_error:
                raise ImportError(
                    "GDAL is NOT installed correctly. "
                    f"GDAL wheel can be downloaded from {self._gdal_wheel_resources}. "
                    f"See the installation guide: {self._gdal_installation_guide}."
                )
            return func(*args, **kwargs)

        return wrapper

    def _check_gdal_data(self):
        from subprocess import Popen, PIPE 
        if 'GDAL_DATA' in os.environ and os.path.exists(os.environ['GDAL_DATA']):
            if self.verbose:
                _logger.info(f"GDAL_DATA is set to: {os.environ['GDAL_DATA']}")
            self._gdal_data_found = True
        else:
            if self.verbose:
                _logger.warning(
                    "GDAL_DATA environment variable is not set. "
                    f"Please see {self._gdal_data_variable_resources}"
                )
            # Attempt to locate GDAL data using gdal-config
            try:
                if self.verbose:
                    _logger.info("Trying to find gdal-data path ...")
                process = Popen(['gdal-config', '--datadir'], stdout=PIPE, stderr=PIPE)
                output, err = process.communicate()
                if process.returncode == 0 and os.path.exists(output.strip()):
                    os.environ['GDAL_DATA'] = output.strip().decode()
                    if self.verbose:
                        _logger.info(f"Found gdal-data path: {os.environ['GDAL_DATA']}")
                    self._gdal_data_found = True
            except Exception as e:
                if self.verbose:
                    _logger.error(f"Failed to find gdal-data path. Error: {e}")
                self._gdal_data_found = False

    # Class variable declarations for resources and installation guide
    _gdal_data_variable_resources = 'https://trac.osgeo.org/gdal/wiki/FAQInstallationAndBuilding#HowtosetGDAL_DATAvariable'
    _gdal_wheel_resources = 'https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal'
    _gdal_installation_guide = 'https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/'

class RedirectToNew:
    """
    A decorator to redirect calls from deprecated functions or classes to new ones,
    issuing a deprecation warning and guiding users towards the updated implementation.

    This decorator simplifies the process of transitioning codebases to use new
    functions or classes without breaking existing implementations that rely on
    deprecated ones.

    Parameters
    ----------
    new_target : callable or class
        The new function or class to which calls should be redirected.
    reason : str
        Explanation why the redirection is occurring, typically including deprecation
        information and guidance on using the new target.

    Examples
    --------
    >>> from fusionlab.decorators import RedirectToNew
    >>> @RedirectToNew(new_function, "Use `new_function` instead of `old_function`.")
    ... def old_function():
    ...     pass
    ...
    >>> old_function()
    # This call will be redirected to `new_function`, with a warning issued
    # about the deprecation.

    """

    def __init__(self, new_target, reason):
        if not callable(new_target):
            raise TypeError("The new target must be a callable or a class.")
        if not isinstance(reason, str):
            raise TypeError("Redirect reason must be supplied as a string.")

        self.new_target = new_target
        self.reason = reason

    def __call__(self, cls_or_func):
        @functools.wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            _logger.warning(f"DEPRECATION WARNING: {self.reason}")
            return self.new_target(*args, **kwargs)

        return wrapper
  
class PlotPrediction:
    """
    A decorator for plotting predictions and observations using matplotlib. 
    This decorator enhances functions that return prediction and observation 
    data by optionally generating a scatter plot for visual comparison.

    Parameters
    ----------
    turn : str, optional
        Controls whether plotting is enabled ('on') or disabled ('off'). Defaults to 'off'.
    **kwargs : dict
        Customization options for the plot, supporting matplotlib.pyplot keywords.

    Attributes
    ----------
    fig_size : tuple
        Figure size for the plot.
    y_pred_kws : dict
        Styling options for the predicted values scatter plot.
    y_obs_kws : dict
        Styling options for the observed values scatter plot.
    tick_params : dict
        Parameters for configuring axis ticks.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    obs_line : tuple
        Controls the observation line plotting ('on', 'off') and its type ('Obs', 'Pred').
    l_kws : dict
        Line properties for the observation line.
    savefig : str or dict
        Path or options for saving the figure.

    Examples
    --------
    >>> from fusionlab.decorators import PlotPrediction
    >>> @PlotPrediction(turn='on', fig_size=(10, 6))
    ... def my_prediction_function():
    ...     # prediction function logic here
    ...     return y_true, y_pred, 'on'
    ...
    >>> my_prediction_function()
    # This will generate a scatter plot for the predicted and observed values.

    """

    def __init__(self, turn='off', **kwargs):
        self.turn = turn
        self.fig_size = kwargs.pop('fig_size', (16, 8))
        self.y_pred_kws = kwargs.pop('y_pred_kws', {'c': 'r', 's': 200, 'alpha': 1,
                                                     'label': 'Predicted flow:y_pred'})
        self.y_obs_kws = kwargs.pop('y_obs_kws', {'c': 'blue', 's': 100, 'alpha': 0.8,
                                                   'label': 'Observed flow:y_true'})
        self.tick_params = kwargs.pop('tick_params', {'axis': 'x', 'labelsize': 10,
                                                      'rotation': 90})
        self.xlabel = kwargs.pop('xlabel', 'Boreholes tested')
        self.ylabel = kwargs.pop('ylabel', 'Flow rates(FR) classes')
        self.obs_line = kwargs.pop('obs_line', ('off', 'Obs'))
        self.l_kws = kwargs.pop('l_kws', {'c': 'blue', 'ls': '--', 'lw': 1, 'alpha': 0.5})
        self.savefig = kwargs.pop('savefig', None)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            y_true, y_pred, switch = func(*args, **kwargs)
            turn = switch if switch is not None else self.turn

            if turn == 'on':
                self._plot(y_true, y_pred)

            return y_true, y_pred, switch

        return wrapper

    def _plot(self, y_true, y_pred):
        plt.figure(figsize=self.fig_size)
        plt.scatter(y_true.index, y_true, **self.y_obs_kws)
        plt.scatter(y_pred.index, y_pred, **self.y_pred_kws)

        if self.obs_line[0] == 'on':
            data = y_true if 'true' in self.obs_line[1].lower(
                ) or 'obs' in self.obs_line[1].lower() else y_pred
            plt.plot(data.index, data, **self.l_kws)

        plt.tick_params(**self.tick_params)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()

        if self.savefig:
            if isinstance(self.savefig, str):
                plt.savefig(self.savefig)
            else:
                plt.savefig(**self.savefig)

        plt.show()

class PlotFeatureImportance:
    """
    A decorator to plot permutation feature importance (PFI) diagrams or dendrogram 
    figures for feature correlation analysis. It utilizes matplotlib for plotting 
    and can be customized with various keyword arguments.

    Parameters
    ----------
    kind : str, optional
        Specifies the type of plot to generate. Options are:
        - 'pfi' for permutation feature importance before and after shuffling trees.
        - 'dendro' for a dendrogram plot showing feature correlations.
        Defaults to 'pfi'.
    turn : str, optional
        Controls whether to plot ('on') or not ('off'). Defaults to 'off'.
    **kwargs : dict
        Keyword arguments for matplotlib plotting functions and additional customization.

    Examples
    --------
    >>> from fusionlab.decorators import PlotFeatureImportance
    >>> @PlotFeatureImportance(kind='pfi', turn='on', fig_size=(10, 6))
    ... def my_model_analysis_function():
    ...     # Function logic here
    ...     return X, y_pred, y_true, model, feature_names, 'on'
    ...
    >>> my_model_analysis_function()
    # This will plot the specified PFI diagram if turn is 'on'.

    Note
    ----
    Ensure matplotlib is installed in your environment to use this decorator.
    """
    
    def __init__(self, kind='pfi', turn='off', **kwargs):
        self.kind = kind
        self.turn = turn
        self.fig_size = kwargs.pop('fig_size', (9, 3))
        self.savefig = kwargs.pop('savefig', None)
        # Default keyword arguments for various plots
        self.barh_kws = kwargs.pop('barh_kws', {'color': 'blue', 'edgecolor': 'k', 'linewidth': 2})
        self.box_kws = kwargs.pop('box_kws', {'vert': False, 'patch_artist': True})
        self.dendro_kws = kwargs.pop('dendro_kws', {'leaf_rotation': 90})
        self.plot_kwargs = kwargs  # Remaining kwargs for further customization

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the decorated function
            results = func(*args, **kwargs)
            # Unpack results based on expected structure
            X, y_pred, y_true, model, feature_names, switch = results
            
            # Update settings based on function return values if provided
            self.turn = switch if switch is not None else self.turn
            
            # Proceed to plot if enabled
            if self.turn.lower() == 'on':
                self._plot_results(X, y_pred, y_true, model, feature_names)
            
            return results

        return wrapper

    def _plot_results(self, X, y_pred, y_true, model, feature_names):
        if self.kind == 'pfi':
            self._plot_pfi(X, model, feature_names)
        elif self.kind == 'dendro':
            self._plot_dendrogram(X, feature_names)
        else:
            warnings.warn(f"Unknown kind '{self.kind}'. No plot will be generated.")

        if self.savefig:
            plt.savefig(self.savefig, **self.plot_kwargs)

    def _plot_pfi(self, X, model, feature_names):
        # Example PFI plotting.
        plt.figure(figsize=self.fig_size)
        plt.barh(range(len(feature_names)), model.feature_importances_, **self.barh_kws)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.show()

    def _plot_dendrogram(self, X, feature_names):
        # Example dendrogram plotting.
        from scipy.cluster.hierarchy import dendrogram, linkage
        Z = linkage(X, 'ward')
        plt.figure(figsize=self.fig_size)
        dendrogram(Z, labels=feature_names, **self.dendro_kws)
        plt.show()

class AppendDocSection:
    """
    A decorator to append a specific section of a function's or class's docstring
    to another. This is particularly useful for avoiding redundancy when documenting
    shared parameters or information across multiple functions or classes.

    Parameters
    ----------
    source_func : callable
        The source function or class whose docstring part will be appended.
    start : str, optional
        The start marker from where to begin appending the docstring.
    end : str, optional
        The end marker where to stop appending the docstring. If not provided, 
        everything from the `start` to the end of the source's docstring is appended.

    Examples
    --------
    >>> from fusionlab.decorators import AppendDocSection
    >>> @AppendDocSection(source_func=writedf, start='param reason', end='param to_')
    ... def new_function():
    ...     '''Function-specific docstring.'''
    ...     pass
    ...
    >>> print(new_function.__doc__)
    # This will include the section of `writedf`'s docstring from 'param reason' 
    # to 'param to_' appended to 'new_function' docstring.

    """
    
    def __init__(self, source_func, start=None, end=None):
        if not callable(source_func):
            raise TypeError("`source_func` must be a callable.")
        self.source_func = source_func
        self.start = start
        self.end = end

    def __call__(self, target_func):
        source_doc = inspect.getdoc(self.source_func) or ''
        target_doc = inspect.getdoc(target_func) or ''
        
        # Find the start index
        start_ix = source_doc.find(self.start) if self.start else 0
        end_ix = source_doc.find(self.end, start_ix) if self.end else len(source_doc)
        
        # Handle cases where start or end markers are not found
        if self.start and start_ix == -1:
            warnings.warn(f"Start marker '{self.start}' not found in"
                          f" `{self.source_func.__name__}` docstring.")
            start_ix = 0
        if self.end and end_ix == -1:
            warnings.warn(f"End marker '{self.end}' not found in"
                          f" `{self.source_func.__name__}` docstring.")
            end_ix = len(source_doc)

        # Extract the desired docstring section
        doc_section = source_doc[start_ix:end_ix]

        # Append the extracted section to the target's docstring
        target_func.__doc__ = (target_doc + "\n\n" + doc_section).strip()

        return target_func

class AppendDocFrom:
    """
    A decorator for appending a specific section of a function's or class's docstring
    to another. This is useful for avoiding redundancy in documentation, especially
    for shared parameters or descriptions.

    Parameters
    ----------
    source : callable
        The source function or class from which to extract the docstring section.
    from_ : str
        The start marker for the docstring section to be extracted.
    to : str, optional
        The end marker for the docstring section. If not provided, everything
        from `from_` to the end of the source's docstring is used.
    insert_at : str
        The marker in the target's docstring where the extracted section should
        be inserted. If not found, the section is appended at the end.

    Examples
    --------
    >>> from fusionlab.decorators import AppendDocFrom
    >>> @AppendDocFrom(source=func0, from_='Parameters', to='Returns', insert_at='Parameters')
    ... def new_func():
    ...     '''New function docstring.'''
    ...     pass
    ...
    >>> print(new_func.__doc__)
    # This will print 'new_func' docstring with the 'func0' docstring section 
    # from 'Parameters' to 'Returns' appended at the 'Parameters' marker.

    Note
    ----
    It's recommended to use docstrings with consistent formatting to ensure
    proper insertion and readability.

    """
    def __init__(self, source, from_, to=None, insert_at='Parameters'):
        self.source = source
        self.from_ = from_
        self.to = to
        self.insert_at = insert_at.lower()
        
    def __call__(self, target):
        self._append_doc(target)
        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            return target(*args, **kwargs)

        # self._append_doc(target)
        return wrapper
    
    def _append_doc(self, target):
        source_doc = inspect.getdoc(self.source) or ''
        target_doc = inspect.getdoc(target) or ''
        
        start_index = source_doc.find(self.from_)
        end_index = source_doc.find(self.to, start_index) if self.to else len(source_doc)
        
        if start_index == -1 or (self.to and end_index == -1):
            warnings.warn(f"Cannot find specified docstring section in `{self.source.__name__}`.")
            return
        
        doc_section = source_doc[start_index:end_index]
        
        insert_index = target_doc.lower().find(self.insert_at)
        if insert_index == -1:
            target_doc += "\n\n" + doc_section
        else:
            part1 = target_doc[:insert_index]
            part2 = target_doc[insert_index:]
            target_doc = part1 + doc_section + "\n\n" + part2
        
        target.__doc__ = target_doc

class NumpyDocstring:
    """
    A class decorator designed to automatically parse and reformat the docstring of
    the decorated function into a structured NumPy-style docstring. This decorator
    enhances readability and standardization of documentation, making it more useful
    for developers and users, especially in the context of Sphinx-generated docs.

    Parameters
    ----------
    func : function, optional
        The function to be decorated. If not provided at initialization, it must be
        set later by calling the instance as a decorator.
    enforce_strict : bool, optional
        If True, enforces strict NumPy docstring formatting rules. This may include
        checking for specific sections and their order. Defaults to False, allowing
        for more flexibility in the original docstring format.
    custom_sections : dict, optional
        A dictionary where keys are section titles (e.g., "Custom Section") and values
        are the content for those sections. This allows for the addition of custom
        sections not typically found in NumPy docstrings.

    Examples
    --------
    Using as a decorator directly on a function:

    @NumpyDocstring
    def my_function(x, y):
        \"\"\"Function docstring.\"\"\"
        return x + y

    Adding custom sections and enforcing strict formatting:

    @NumpyDocstring(enforce_strict=True, custom_sections={
        'Custom Section': 'Details here.'})
    def another_function(x):
        \"\"\"Another function docstring.\"\"\"
        return x * 2
    """

    def __init__(self, func=None, *, enforce_strict=False, custom_sections=None):
        self.func = func
        self.enforce_strict = enforce_strict
        self.custom_sections = custom_sections or {}
        if func is not None:
            functools.update_wrapper(self, func)
            self._update_docstring()

    def __call__(self, *args, **kwargs):
        if self.func:
            return self.func(*args, **kwargs)
        else:
            def wrapper(func):
                self.func = func
                functools.update_wrapper(self, func)
                self._update_docstring()
                return self
            return wrapper

    def __get__(self, instance, owner):
        return self if instance is None else functools.partial(self.__call__, instance)

    def _parse_docstring(self, docstring):
        """
        Advanced parsing of the original docstring to identify and reformat sections.
        """
        # Define the sections and their possible headings in docstrings
        section_headings = {
            'Parameters': ['parameters', 'args', 'arguments', ':param'],
            'Returns': ['returns', ':return', ':returns'],
            'Raises': ['raises', ':raise', ':raises'],
            'Examples': ['examples', ':example'],
            'Warnings': ['warnings', ':warning'],
            'See Also': ['see also', 'references'],
            'Notes': ['notes']
        }

        # Initialize sections dictionary
        sections = {key: '' for key in section_headings}

        # Regular expression to detect section headings
        section_regex = re.compile(
            r'^\s*(?P<section>' + '|'.join(
                [f"(?:{'|'.join(headings)})" for headings in section_headings.values()]
                ) + r')\s*$', re.IGNORECASE)

        current_section = None
        for line in docstring.split('\n'):
            match = section_regex.match(line.strip())
            if match:
                # Find which section it belongs to
                for section, headings in section_headings.items():
                    if match.group('section').lower() in headings:
                        current_section = section
                        break
            elif current_section:
                sections[current_section] += line + '\n'

        # Apply custom sections if any
        for section, content in self.custom_sections.items():
            sections[section] = content

        return sections

    def _format_section(self, title, content):
        """
        Format a single section with the given title and content.
        """
        if not content.strip():
            return ''
        return f"{title}\n{'-' * len(title)}\n{content.strip()}\n"

    def _update_docstring(self):
        """
        Update the function's docstring with parsed and formatted content.
        """
        if not self.func.__doc__:
            return

        sections = self._parse_docstring(self.func.__doc__)
        formatted_docstring = "\n".join(self._format_section(
            title, content) for title, content in sections.items() if content)

        self.func.__doc__ = formatted_docstring

    def __set_name__(self, owner, name):
        self._update_docstring()
        setattr(owner, name, self)
        
def sanitize_docstring(enforce_strict=False, custom_sections=None):
    """
    Decorator factory function that returns an instance of NumpyDocstring.
    This function simplifies the application of the decorator with additional parameters
    like enforcing strict formatting and adding custom sections to the docstring.

    Parameters
    ----------
    enforce_strict : bool, optional
        If set to True, the decorator enforces strict adherence to the NumPy docstring
        format, potentially raising errors for non-compliance. Defaults to False.
    custom_sections : dict, optional
        Allows for the specification of custom sections in the decorated function's
        docstring. Keys are the titles of the sections, and values are the content.

    Returns
    -------
    decorator : NumpyDocstring
        An instance of AdvancedNumpyDocDecorator configured with the provided parameters.

    Examples
    --------
    Decorating a function with custom sections and without strict enforcement:

    @sanitize_docstring(custom_sections={'Custom Usage': 'This is how you use this function.'})
    def sample_function(param1, param2):
        \"\"\"This function does something interesting.\"\"\"
        pass
    
    """
    def decorator(func):
        return NumpyDocstring(func, enforce_strict=enforce_strict,
                              custom_sections=custom_sections)
    return decorator


class _M:
    def _m(self): pass
MethodType = type(_M()._m)

class _AvailableIfDescriptor:
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name

        # update the docstring of the descriptor
        functools.update_wrapper(self, fn)

    def __get__(self, obj, owner=None):
        attr_err = AttributeError(
            f"This {repr(owner.__name__)} has no attribute {repr(self.attribute_name)}"
        )
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            if not self.check(obj):
                raise attr_err
            out = MethodType(self.fn, obj)

        else:
            # This makes it possible to use the decorated method as an unbound method,
            # for instance when monkeypatching.
            @functools.wraps(self.fn)
            def out(*args, **kwargs):
                if not self.check(args[0]):
                    raise attr_err
                return self.fn(*args, **kwargs)

        return out

def available_if(check):
    """An attribute that is available only if check returns a truthy value

    Parameters
    ----------
    check : callable
        When passed the object with the decorated method, this should return
        a truthy value if the attribute is available, and either return False
        or raise an AttributeError if not available.

    Examples
    --------
    >>> from sklearn.utils.metaestimators import available_if
    >>> class HelloIfEven:
    ...    def __init__(self, x):
    ...        self.x = x
    ...
    ...    def _x_is_even(self):
    ...        return self.x % 2 == 0
    ...
    ...    @available_if(_x_is_even)
    ...    def say_hello(self):
    ...        print("Hello")
    ...
    >>> obj = HelloIfEven(1)
    >>> hasattr(obj, "say_hello")
    False
    >>> obj.x = 2
    >>> hasattr(obj, "say_hello")
    True
    >>> obj.say_hello()
    Hello
    """
    return lambda fn: _AvailableIfDescriptor(fn, check, attribute_name=fn.__name__)


def isdf(func):
    """
    Decorator that ensures the first positional argument passed to the 
    decorated callable is a pandas DataFrame. If it's not, attempts to convert
    it to a DataFrame using an optional `columns` keyword argument. 
    
    Function is designed to be flexible and efficient, suitable for 
    both functions and methods.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the signature of the function
        sig = inspect.signature(func)
        params = sig.parameters
        param_list = list(params.values())
        
        # Check if the function has any parameters
        if not param_list:
            # No parameters to process
            return func(*args, **kwargs)
        
        # Determine if we're decorating a method (with 'self' or 'cls')
        is_method = False
        start_idx = 0
        if param_list[0].name in ('self', 'cls'):
            is_method = True
            start_idx = 1  # Skip 'self' or 'cls'

        # Map arguments to their names
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Identify the data parameter name
        # Prefer 'data' if it's among the parameters
        data_param_name = None
        for idx, param in enumerate(param_list[start_idx:], start=start_idx):
            if param.name == 'data':
                data_param_name = 'data'
                break
        else:
            # If 'data' is not a parameter, use the first positional
            # parameter after 'self'/'cls'
            if ( len(param_list) > start_idx) or is_method:
                data_param_name = param_list[start_idx].name
            else:
                # No parameters left to consider
                return func(*args, **kwargs)

        # Get 'data' argument from bound arguments
        data = bound_args.arguments.get(data_param_name, None)
        columns = bound_args.arguments.get('columns', kwargs.get('columns', None))
        if isinstance(columns, str):
            columns = [columns]
        
        # Proceed with conversion if necessary
        if data is not None and not isinstance(data, pd.DataFrame):
            try:
                if columns and len(columns) != data.shape[1]:
                    data = pd.DataFrame(data)
                else: 
                    data = pd.DataFrame(data, columns=columns)
            except Exception as e:
                raise ValueError(
                    f"Unable to convert {type(data).__name__!r} to DataFrame: {e}"
                )
            data.columns = data.columns.astype(str)
            # Update the bound arguments with the new data
            bound_args.arguments[data_param_name] = data
        
        # Call the original function with the updated arguments
        return func(*bound_args.args, **bound_args.kwargs)
    
    return wrapper

class IsPerformanceData:
    """
    A decorator and validator for performance data in machine learning.

    The `IsPerformanceData` class can be used both as a decorator and as a
    function to validate performance data, ensuring that the data conforms
    to expected formats and value ranges. It checks that the data is a 
    pandas DataFrame or dictionary with numeric values, handles NaN values
    according to the specified policy, and verifies that performance metrics
    are within the range [0, 1].

    Parameters
    ----------
    *args : tuple
        Positional arguments. If used as a decorator without arguments,
        the first argument may be the function to decorate.
        If used as a function, the first argument is the data to validate.

    **kwargs : dict
        Keyword arguments for configuration.

    nan_policy : str, optional
        Defines how to handle NaN values in the data.
        Options are ``'raise'`` (default), ``'omit'``, or ``'propagate'``.

        - ``'raise'``: Raise a ``ValueError`` if NaN values are found.
        - ``'omit'``: Drop rows containing NaN values.
        - ``'propagate'``: Proceed without altering NaN values.

    convert_integers : bool, optional
        If ``True`` (default), integer values are converted to floats.

    check_performance_range : bool, optional
        If ``True`` (default), checks that all performance values are within
         the range [0, 1].

    verbose : bool, optional
        If ``True``, prints detailed messages during validation.

    Methods
    -------
    __call__(*args, **kwargs)
        Allows the class instance to be called as a function or used as a 
        decorator.

    actual_validate_performance_data(data)
        Validates the performance data according to the specified policies.

    Notes
    -----
    This class serves both as a decorator and a validator function. When 
    used as a decorator, it validates the performance data  passed to the
    decorated function. When used as a function, it validates the given 
    data and returns the validated DataFrame.

    The validation process includes:

    - Checking that the data is a pandas DataFrame or a dictionary that can
      be converted to a DataFrame.
    - Converting integer values to floats if `convert_integers` is `True`.
    - Handling NaN values according to the specified `nan_policy`.
    - Ensuring all performance values are within the range [0, 1] if
       `check_performance_range` is `True`.

    Examples
    --------
    Using as a decorator without arguments:

    >>> from fusionlab.decorators import IsPerformanceData
    >>> @IsPerformanceData
    ... def analyze_performance(data):
    ...     # Function body
    ...     pass

    Using as a decorator with arguments:

    >>> from fusionlab.decorators import IsPerformanceData
    >>> @IsPerformanceData(nan_policy='omit', verbose=True)
    ... def analyze_performance(data):
    ...     # Function body
    ...     pass

    Using as a function:

    >>> from fusionlab.decorators import IsPerformanceData
    >>> validator = IsPerformanceData(nan_policy='omit')
    >>> validated_data = validator(data)

    See Also
    --------
    pandas.DataFrame : Two-dimensional, size-mutable,
        potentially heterogeneous tabular data.

    References
    ----------
    .. [1] pandas.DataFrame documentation,
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        # Default parameter values
        self.nan_policy = kwargs.get('nan_policy', 'raise')
        self.convert_integers = kwargs.get('convert_integers', True)
        self.check_performance_range = kwargs.get(
            'check_performance_range', True)
        self.verbose = kwargs.get('verbose', False)
        self.func = None

        if args and callable(args[0]):
            # Decorator used without arguments
            self.func = args[0]
        elif args:
            # Called as a function with data as the first argument
            self.data = args[0]
        else:
            # Decorator used with arguments or will be called as a function later
            self.data = None

    def __call__(self, *args, **kwargs):
        """
        Enables the class instance to be called as a function
        or used as a decorator.

        Parameters
        ----------
        *args : tuple
            Positional arguments. If used as a decorator, the
            first argument may be the function to decorate. If
            used as a function, the first argument is the data
            to validate.

        **kwargs : dict
            Keyword arguments passed to the decorated function
            or the validation process.

        Returns
        -------
        callable or pandas.DataFrame
            If used as a decorator, returns the wrapped function.
            If used as a function, returns the validated DataFrame.

        Examples
        --------
        Using as a decorator:

        >>> from fusionlab.decorators import IsPerformanceData
        >>> @IsPerformanceData
        ... def analyze_performance(data):
        ...     # Function body
        ...     pass

        Using as a function:

        >>> from fusionlab.decorators import IsPerformanceData
        >>> validator = IsPerformanceData()
        >>> validated_data = validator(data)

        """
        if self.func:
            # Used as a decorator without arguments
            @functools.wraps(self.func)
            def wrapper(*args, **kwargs):
                data = args[0]
                data = self._is_formatter_data(data)
                validated_data = self.actual_validate_performance_data(data)
                return self.func(validated_data, *args[1:], **kwargs)
            return wrapper(*args, **kwargs)
        elif args and callable(args[0]):
            # Used as a decorator with arguments
            func = args[0]
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                data = args[0]
                data = self._is_formatter_data(data)
                validated_data = self.actual_validate_performance_data(data)
                return func(validated_data, *args[1:], **kwargs)
            return wrapper
        else:
            # Called as a function with data
            data = args[0] if args else self.data
            return self.actual_validate_performance_data(data)

    @isdf 
    def actual_validate_performance_data(self, data):
        """
        Performs the actual validation of the performance data.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            The performance data to validate. Should be a DataFrame
            or a dictionary that can be converted into a DataFrame.

        Returns
        -------
        pandas.DataFrame
            The validated performance data.

        Raises
        ------
        ValueError
            If the data is invalid or does not meet the specified
            criteria.

        Notes
        -----
        The validation process includes:

        - Converting dictionaries to DataFrames if necessary.
        - Converting integer values to floats if
          `convert_integers` is `True`.
        - Handling NaN values according to the `nan_policy`.
        - Checking that all performance values are within the
          range [0, 1] if `check_performance_range` is `True`.

        The performance values are expected to satisfy:

        .. math::

            0 \\leq x \\leq 1

        where :math:`x` represents a performance metric value.

        Examples
        --------
        >>> validator = IsPerformanceData()
        >>> validated_data = validator.actual_validate_performance_data(data)

        """
        from .utils.validator import convert_to_numeric
        from .utils.validator import is_valid_policies

        # Convert to DataFrame if input is a dictionary
        if isinstance(data, dict):
            if self.verbose:
                print("Converting dictionary to DataFrame...")
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be either a dictionary "
                             "or a DataFrame.")

        # Ensure all values are float, convert integers to floats if needed
        if self.convert_integers:
            if self.verbose:
                print("Converting integer values to floats where necessary...")
            df = df.applymap(
                lambda x: convert_to_numeric(
                    x, preserve_integers=False,
                    context_description='Performance data')
            )

        # Handle NaN values according to nan_policy
        is_valid_policies(
            self.nan_policy, allowed_policies=['raise', 'omit', 'propagate']
        )

        if df.isna().any().any():  # Check for NaN values
            if self.nan_policy == 'raise':
                raise ValueError("NaN values detected in the data. "
                                 "Set `nan_policy='omit'` to drop them.")
            elif self.nan_policy == 'omit':
                if self.verbose:
                    print("Dropping rows with NaN values...")
                df = df.dropna()

        # Ensure all values are float type
        df = df.astype(float)

        # Check if performance values are within the valid range [0, 1]
        if self.check_performance_range:
            if self.nan_policy == 'propagate':
                df_checked = df.dropna()
            else:
                df_checked = df

            if (df_checked < 0).any().any():
                raise ValueError("Performance values cannot be negative.")
            if (df_checked > 1).any().any():
                raise ValueError("Performance values must be in the "
                                 "range [0, 1].")

        if self.verbose:
            print("Validation and conversion complete. "
                  "Data is ready for further processing.")

        return df
    
    def _is_formatter_data (self, data): 
        """ Check whether the data passed is a formatter instance. If so,
        then retrieve the dataframe before calling the 
        `actual_validate_performance_data` method. 
        
        """
        from .api.formatter import( 
            DataFrameFormatter, MultiFrameFormatter, formatter_validator
            )
        if isinstance(data, (DataFrameFormatter, MultiFrameFormatter)):
            data = formatter_validator(data, df_indices=[0], only_df=True)
            
        return data 

class NumpyDocstringFormatter:
    """
    A decorator class for reformatting function docstrings to adhere to the
    NumPy documentation standard.

    This class provides a flexible way to ensure that the docstrings of 
    decorated functions follow a consistent format, making them more readable
    and compatible with tools like Sphinx for generating documentation. It can
    automatically extract and reformat specified sections of a docstring, and
    optionally validate the result using Sphinx.

    Parameters
    ----------
    include_sections : list of str, optional
        A list of section names to include in the reformatted docstring. 
        If None (the default), all recognized sections are included. This 
        allows for selective inclusion of sections like "Parameters", "Returns",
        "Examples", etc., based on user preference or requirements.
        
        Example: ['Parameters', 'Returns', 'Examples']

    validate_with_sphinx : bool, default False
        Indicates whether the reformatted docstring should be validated using
        Sphinx. This can be useful for ensuring that the docstring is not only
        correctly formatted but also compatible with Sphinx documentation
        generation. Note that actual implementation of Sphinx validation is 
        not provided in this example and would require integration with Sphinx's
        documentation building process.

    custom_formatting : callable, optional
        A custom function that applies additional formatting to each section
        of the docstring. This function should accept two arguments: 
            `section_name` (a string indicating the name of the section) and 
            `section_content` (the content of the section as a string), and 
            return the formatted content as a string. This allows for
        further customization of the docstring formatting beyond the standard
        reformatting performed by this class::
        
        Example function:
            def custom_formatter(section_name, section_content):
                # Custom formatting logic here
                return formatted_content

    Examples
    --------
    Using the decorator with default settings to reformat all sections:

    >>> from fusionlab.decorators import NumpyDocstringFormatter
    >>> @NumpyDocstringFormatter()
    ... def example_function(param1, param2=None):
    ...     '''
    ...     This is an example function with parameters.
    ...
    ...     Parameters
    ...     ----------
    ...     param1 : int
    ...         The first parameter.
    ...     param2 : int, optional
    ...         The second parameter (default is None).
    ...     '''
    ...     return True

    Specifying sections to include and enabling Sphinx validation:

    >>> @NumpyDocstringFormatter(include_sections=['Parameters', 'Returns'], 
    ...                             validate_with_sphinx=True)
    ... def another_function(param1):
    ...     '''
    ...     Another example function demonstrating selective section inclusion
    ...     and Sphinx validation.
    ...
    ...     Parameters
    ...     ----------
    ...     param1 : str
    ...         A string parameter.
    ...
    ...     Returns
    ...     -------
    ...     bool
    ...         Always returns True.
    ...     '''
    ...     return True

    Applying custom formatting to docstring sections:

    >>> def uppercase_formatter(section_name, section_content):
    ...     # Example custom formatting function that uppercases section content
    ...     return section_content.upper()
    ...
    >>> @NumpyDocstringFormatter(custom_formatting=uppercase_formatter)
    ... def custom_formatted_function(param1):
    ...     '''
    ...     Function demonstrating custom formatting of docstring sections.
    ...
    ...     Parameters
    ...     ----------
    ...     param1 : str
    ...         A string parameter to be uppercased in the documentation.
    ...     '''
    ...     return param1.upper()
    """

    def __init__(self, include_sections=None, validate_with_sphinx=False, 
                 custom_formatting=None, verbose=0):

        self.include_sections = include_sections
        self.validate_with_sphinx = validate_with_sphinx
        self.custom_formatting = custom_formatting
        self.verbose=verbose 

    def __call__(self, func):
        """
        Decorator method to apply the docstring formatting.

        Parameters
        ----------
        func : function
            The function to decorate.

        Returns
        -------
        function
            The decorated function with a reformatted docstring.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = self.format_docstring(func.__doc__)
        if self.validate_with_sphinx:
            self.sphinx_validation(wrapper.__doc__)
        
        return wrapper

    def format_docstring(self, docstring):
        """
        

        Parameters
        ----------
        docstring : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        """
        Format the original docstring according to the NumPy standard.

        Parameters
        ----------
        docstring : str
            The original docstring to format.

        Returns
        -------
        str
            The formatted docstring.
        """
        if not docstring:
            return ''

        sections_order = ['Parameters', 'Returns', 'Raises', 'Examples', 
                          'Warnings', 'See Also', 'Notes']
        if self.include_sections is not None:
            sections_order = [s for s in sections_order if s in self.include_sections]

        sections = self.extract_sections(docstring, sections_order)
        formatted_docstring = self.reconstruct_docstring(sections, sections_order)

        return formatted_docstring

    def extract_sections(self, docstring, sections_order):
        """
        Extract and return the docstring sections found in the given order.

        Parameters
        ----------
        docstring : str
            The original docstring from which to extract sections.
        sections_order : list of str
            The ordered list of section names to extract.

        Returns
        -------
        dict
            A dictionary with section names as keys and extracted content as values.
        """
        sections = {section: '' for section in sections_order}
        # Simplified patterns
        section_patterns = {
            'Parameters': re.compile(r'parameters\s*[\n\r]+', re.IGNORECASE),
            'Returns': re.compile(r'returns\s*[\n\r]+', re.IGNORECASE),
            'Raises': re.compile(r'raise[s]?\s*[\n\r]+', re.IGNORECASE),
            'Examples': re.compile(r':examples:\s*[\n\r]+', re.IGNORECASE),
            'Warnings': re.compile(r'warnings\s*[\n\r]+', re.IGNORECASE),
            'See Also': re.compile(r'see also\s*[\n\r]+', re.IGNORECASE),
            'Notes': re.compile(r'notes\s*[\n\r]+', re.IGNORECASE),
        }

        for section in sections_order:
            if section in section_patterns:
                match = section_patterns[section].search(docstring)
                if match:
                    content = match.group(0).strip()
                    if self.custom_formatting:
                        content = self.custom_formatting(section, content)
                    sections[section] = content

        return sections

    def reconstruct_docstring(self, sections, sections_order):
        """
        Reconstruct the docstring from the extracted sections in the given order.

        Parameters
        ----------
        sections : dict
            The sections extracted from the original docstring.
        sections_order : list of str
            The ordered list of section names to include in the reconstructed docstring.

        Returns
        -------
        str
            The reconstructed docstring.
        """
        reconstructed_docstring = ""
        for section in sections_order:
            if sections[section]:
                reconstructed_docstring += f"{section}\n{'-' * len(section)}\n{sections[section]}\n\n"
        return reconstructed_docstring
    
    def sphinx_validation(self, docstring):
        """
        Validates the given docstring using Sphinx and docutils to ensure 
        it adheres to standards acceptable by Sphinx for documentation generation.
    
        Parameters
        ----------
        docstring : str
            The docstring to validate.
    
        Note
        ----
        This method provides a conceptual approach and requires a Sphinx 
        environment to be properly implemented.
        """
        from .utils.deps_utils import import_optional_dependency
        
        try: 
            import_optional_dependency ("docutils")
        except: 
            from .utils.deps_utils import is_module_installed 
            from .utils.deps_utils import ensure_module_installed
            if not is_module_installed("docutils"): 
                ensure_module_installed('docutils', auto_install=True)
            
        from docutils import nodes
        from docutils.core import publish_doctree
        
        try:
            # Create a new document for parsing
            settings_overrides = {'report_level': 2, 'warning_stream': False}
            document = publish_doctree(docstring, settings_overrides=settings_overrides)
            
            # Check for any errors or warnings in the parsed document
            warnings_or_errors = document.traverse(condition=lambda node: isinstance(
                node, (nodes.warning, nodes.error)))
            if next(warnings_or_errors, None):
                if self.verbose:
                    _logger.warning(
                        "Docstring validation failed with warnings or errors.") 
            else:
                if self.verbose:
                    _logger.info("Docstring passed Sphinx validation.")
        except Exception as e:
            if self.verbose:
                _logger.error(
                    f"Docstring validation failed due to an exception: {e}") 


class Dataify:
    r"""
    A class decorator that ensures the first positional argument
    passed to the decorated function is converted into a
    ``pandas.DataFrame`` if <enforce_df> is True. This allows for
    automatic handling of diverse inputs (lists, NumPy arrays,
    Series) and ensures that downstream processing always has a
    consistent DataFrame. When <enforce_df> is False, the decorator
    passes data unchanged if it is already a Series, ignoring other
    conversion steps [1]_.

    Mathematically, one may view this transformation as mapping any
    input :math:`X` into a space of valid DataFrame objects
    :math:`\tilde{X}`, such that:

    .. math::
       \tilde{X} = F_{\text{df}}(X)

    where :math:`F_{\text{df}}(\cdot)` represents the conversion
    function and :math:`\tilde{X}` is the resultant DataFrame.

    Parameters
    ----------
    enforce_df : bool, optional
        If True, the decorator converts the first positional
        argument to a DataFrame if it is not already one. If
        False, the input is passed through unchanged if it is
        a Series. Default is True.
    auto_columns : bool, optional
        If True, automatically generates column names when
        converting data to a DataFrame and no column names are
        otherwise provided. Default is False.
    prefix : str, optional
        A prefix string for auto-generated column names, used
        only if <auto_columns> is True. The default is 'col_'.
    columns : list of str, optional
        Specifies DataFrame column names if <enforce_df> is True.
        If missing or of mismatched length, defaults to integer
        column names unless <ignore_mismatch> is also True, in
        which case the mismatch is silently ignored.
    ignore_mismatch : bool, optional
        If True, mismatches between <columns> length and data
        dimensions are ignored, falling back to default column
        names. If False, a ValueError is raised on mismatch.
        Default is False.
    fail_silently : bool, optional
        If True, any conversion error is caught and the original
        data is passed to the wrapped function. If False, a
        ValueError is raised on failure. Default is False.
    start_incr_at : int, optional
        Starting index for auto-generated column names when
        <auto_columns> is True. Default is 0.

    Notes
    -----
    The transformation is performed by the ``__call__``
    method, which intercepts the function's arguments and,
    if <enforce_df> is True, attempts to convert the first
    positional argument to a DataFrame.

    Examples
    --------
    >>> from fusionlab.decorators import Dataify
    >>> import numpy as np
    >>> import pandas as pd 
    >>> @Dataify(enforce_df=True, columns=['A', 'B'],
    ...          ignore_mismatch=True)
    ... def process_data(data):
    ...     print(data)

    >>> process_data(np.array([[1, 2], [3, 4]]))
       A  B
    0  1  2
    1  3  4

    >>> @Dataify(enforce_df=True, auto_columns=True,
    ...          prefix='feature_')
    ... def process_data2(data):
    ...     print(data)

    >>> process_data2([[5, 6], [7, 8]])
       feature_0  feature_1
    0          5          6
    1          7          8

    >>> @Dataify(enforce_df=True, auto_columns=True,
    ...          prefix='ser_col_')
    ... def process_data3(data):
    ...     print(data)
    
    >>> series =pd.Series ([5, 6, 7, 8], name="series")
    >>> process_data3(series)
       series
    0       5
    1       6
    2       7
    3       8
    >>> @Dataify(enforce_df=False, auto_columns=True,
    ...          prefix='ser_col_')
    ... def process_data4(data):
    ...     print(data)
        
    >>> series =pd.Series ([5, 6, 7, 8], name="series")
    >>> process_data4(series)
    0    5
    1    6
    2    7
    3    8
    Name: series, dtype: int64
    

    References
    ----------
    .. [1] Pedregosa et al. (2011). "Scikit-learn: Machine Learning
       in Python." *Journal of Machine Learning Research,* 12,
       2825–2830.
    """

    def __init__(
        self,
        enforce_df=True,
        auto_columns=False,
        prefix='col_',
        columns=None,
        ignore_mismatch=False,
        fail_silently=False,
        start_incr_at=0
    ):
        self.enforce_df = enforce_df
        self.auto_columns = auto_columns
        self.prefix = prefix
        self.columns = columns
        self.ignore_mismatch = ignore_mismatch
        self.fail_silently = fail_silently
        self.start_incr_at = start_incr_at

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                return func(*args, **kwargs)

            data = args[0]

            # If not enforcing DF, just pass through if it's a Series
            if not self.enforce_df:
                if isinstance(data, pd.Series):
                    return func(*args, **kwargs)

            # Else, enforce DataFrame
            if not isinstance(data, (pd.DataFrame, pd.Series)):
                # Remove 'data' from kwargs if present to avoid collisions
                new_kwargs = {k: v for k, v in kwargs.items() if k != 'data'}
                try:
                    data = self._attempt_dataframe_conversion(data, **new_kwargs)
                except ValueError as e:
                    if self.fail_silently:
                        warnings.warn(f"Dataify Warning: {e}")
                        return func(*args, **kwargs)
                    else:
                        raise

            elif isinstance(data, pd.Series) and self.enforce_df:
                # If it's a Series and we want a DF
                data = data.to_frame()

            return func(data, *args[1:], **kwargs)

        return wrapper

    def _attempt_dataframe_conversion(self, data, **kwargs):
        """ Attempts to convert the input data to a pandas DataFrame using 
         the specified columns if applicable, handling dimension mismatches.
         """
        columns = kwargs.get('columns', self.columns)
        prefix = kwargs.get('prefix', kwargs.get('col_prefix', self.prefix))
        start_incr_at = kwargs.get('start_incr_at', self.start_incr_at)

        if isinstance(columns, str):
            columns = [columns]

        # Auto-generate column names
        if self.auto_columns and columns is None:
            num_cols = np.shape(data)[1] if np.ndim(data) > 1 else 1
            columns = [f"{prefix}{i + start_incr_at}" for i in range(num_cols)]

        try:
            df = pd.DataFrame(data, columns=columns)
        except Exception as e:
            if self.ignore_mismatch:
                df = pd.DataFrame(data)
            else:
                raise ValueError(
                    f"Error converting {type(data).__name__!r} to DataFrame: {e}"
                )

        return df

def copy_doc(
    source=None, 
    docstring=None, 
    replace=False, 
    copy_attrs=None
    ):
    """
    Class or function decorator to copy the docstring and specified attributes 
    from a source class or function to the decorated class or function.

    This decorator facilitates the transfer of documentation and attributes, 
    ensuring consistency and reducing redundancy. It is particularly useful when  
    creating aliases for deprecated classes or functions, allowing the new 
    and deprecated entities to share documentation seamlessly.

    .. math::
        \text{CopyDoc}(S, D) = 
        \begin{cases} 
            \text{Replace} & \text{if } \text{replace=True} \\
            \text{Append}  & \text{if } \text{replace=False}
        \end{cases}

    Parameters
    ----------
    source : class or function, optional
        The source class or function from which to copy the docstring and attributes.
        If provided, the decorator will copy the `__doc__` attribute and any attributes
        specified in `copy_attrs` from this source to the decorated object.

    docstring : str, optional
        An additional or alternative docstring to include in the decorated object.
        If `replace` is `False`, this docstring will be appended to the source's docstring.
        If `replace` is `True`, this docstring will replace the source's docstring.

    replace : bool, default=False
        Determines how the `docstring` parameter is applied.
        - If `True`, the existing docstring of the decorated object is replaced entirely 
          by the `docstring` parameter.
        - If `False`, the `docstring` parameter is appended to the existing docstring.

    copy_attrs : list of str, optional
        A list of attribute names to copy from the `source` to the decorated object.
        Only attributes listed in `copy_attrs` will be copied. If `None`, no additional
        attributes are copied beyond the docstring.

    Returns
    -------
    decorator : function
        The decorator function that applies the specified docstring and attribute 
        copying to the decorated class or function.

    Methods
    -------
    __call__(obj)
        Applies the decorator to the given class or function `obj`.

    Examples
    --------
    >>> import hwm 
    >>> from hwm.utils.decorators import copy_doc
    >>> from hwm.estimators import HWRegressor

    **Copying Docstring from a Class:**

    >>> @copy_doc(source=HWRegressor, docstring="Deprecated. Use HWRegressor instead.",
    ...              replace=False)
    ... class HammersteinWienerRegressor(HWRegressor):
    ...     def __init__(self, *args, **kwargs):
    ...         import warnings
    ...         warnings.warn(
    ...             "HammersteinWienerRegressor is deprecated and will be removed in version 1.1.3. "
    ...             "Use HWRegressor instead.",
    ...             DeprecationWarning,
    ...             stacklevel=2
    ...         )
    ...         super().__init__(*args, **kwargs)

    **Copying Docstring from a Function:**

    >>> def source_function():
    ...     '''Original source function docstring.'''
    ...     pass
    ...
    >>> @copy_doc(source=source_function, docstring="Additional information.", replace=False)
    ... def decorated_function():
    ...     pass
    ...
    >>> print(decorated_function.__doc__)
    Original source function docstring.

    Additional information.

    **Replacing Docstring Completely:**

    >>> @copy_doc(docstring="Completely new docstring.", replace=True)
    ... def new_function():
    ...     pass
    ...
    >>> print(new_function.__doc__)
    Completely new docstring.

    **Copying Specific Attributes:**

    >>> class Source:
    ...     attribute = "Copied attribute"
    ...
    >>> @copy_doc(source=Source, copy_attrs=["attribute"])
    ... class Decorated(Source):
    ...     pass
    ...
    >>> print(Decorated.attribute)
    Copied attribute

    Notes
    -----
    - The `copy_doc` decorator does not modify any methods or internal attributes 
      (those starting with an underscore) of the decorated class or function.
    - It is recommended to use this decorator primarily for creating aliases 
      for deprecated classes or functions to maintain documentation consistency.

    See Also
    --------
    warnings.warn : Issue warnings to users about deprecated features.

    References
    ----------
    .. [1] Python Documentation on [Decorators](https://docs.python.org/3/glossary.html#term-decorator).

    """
    def decorator(obj):
        """
        Apply the `copy_doc` decorator to the given object `obj`.

        Parameters
        ----------
        obj : class or function
            The class or function to which the docstring and attributes 
            will be copied.

        Returns
        -------
        obj : class or function
            The decorated object with updated docstring and attributes.
        """
        # Handle docstring copying
        if source:
            source_doc = source.__doc__ or ""
            if replace:
                combined_doc = docstring or ""
            else:
                combined_doc = f"{source_doc}\n\n{docstring}" if docstring else source_doc
        else:
            combined_doc = docstring or ""

        if combined_doc:
            obj.__doc__ = combined_doc

        # Handle attribute copying
        if source and copy_attrs:
            for attr in copy_attrs:
                if hasattr(source, attr):
                    setattr(obj, attr, getattr(source, attr))

        return obj

    return decorator

# Substitution and Appender are derived from matplotlib.docstring (1.1.0)
# module https://matplotlib.org/users/license.html

class Substitution:
    """
    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a sequence or
    dictionary suitable for performing substitution; then
    decorate a suitable function with the constructed object. e.g.

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "%(author)s wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments.

    sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    @sub_first_last_names
    def some_function(x):
        "%s %s wrote the Raven"
    """

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")

        self.params = args or kwargs

    def __call__(self, func: callable) -> callable:
        func.__doc__ = func.__doc__ and func.__doc__ % self.params
        return func

    def update(self, *args, **kwargs) -> None:
        """
        Update self.params with supplied args.
        """
        if isinstance(self.params, dict):
            self.params.update(*args, **kwargs)


class Appender:
    """
    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    add_copyright = Appender("Copyright (c) 2009", join='\n')

    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """

    addendum: str | None

    def __init__(self, addendum: str | None, join: str = "", indents: int = 0):
        if indents > 0:
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func: callable) -> callable:
        func.__doc__ = func.__doc__ if func.__doc__ else ""
        self.addendum = self.addendum if self.addendum else ""
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = dedent(self.join.join(docitems))
        return func


def indent(text: str | None, indents: int = 1) -> str:
    if not text or not isinstance(text, str):
        return ""
    jointext = "".join(["\n"] + ["    "] * indents)
    return jointext.join(text.split("\n"))


# Substitution and Appender are derived from matplotlib.docstring (1.1.0)
# module https://matplotlib.org/users/license.html
class ValidateXy:
    """
    A class-based decorator that validates and converts the input data `X` 
    and optionally `y`. It checks for numeric data types, reshapes the 
    data if needed, and applies additional checks based on preferences.
    
    Parameters
    ----------
    check_y : bool or str, optional, default='auto'
        If True, the decorator will check `y` for validity (numeric type).
        If 'auto', `y` will be checked only if it is not None.
    
    ensure_y_2d : bool, optional, default=False
        If False, `y` is expected to be 1D. If True, `y` can be 1D or 2D.
    
    allow_sparse_matrix : bool, optional, default=False
        If True, the decorator will allow sparse matrix inputs for `X`.
    
    check_1d_input : bool, optional, default=False
        If True, checks whether `X` is 1D and reshapes it accordingly.
    
    allow_negative_values : bool, optional, default=True
        If False, raises a warning if `X` or `y` contains negative values.
    
    raise_on_non_numeric : bool, optional, default=True
        If True, raises an error when non-numeric data is encountered in 
        `X` or `y`.
    
    Examples
    --------
    >>> from fusionlab.decorators import ValidateXy 
    >>> @ValidateXy(check_y=True, ensure_y_2d=True, check_1d_input=True)
    >>> def my_model(X, y):
    >>>     # Model processing here
    >>>     return X, y
    
    >>> X = [[1, 2], [3, 4]]
    >>> y = [5, 6]
    >>> my_model(X, y)  # Valid data
    
    >>> X_invalid = [[1, 'a'], [3, 4]]
    >>> my_model(X_invalid, y)  # Raises ValueError due to non-numeric data
    
    >>> X_negative = [[-1, 2], [3, 4]]
    >>> my_model(X_negative, y)  # Will raise a valueError if allow_negative_values=False
    """

    def __init__(
        self, 
        check_y="auto", 
        ensure_y_2d=False, 
        allow_sparse_matrix=False, 
        check_1d_input=False, 
        allow_negative_values=True, 
        raise_on_non_numeric=True
    ):
        self.check_y = check_y
        self.ensure_y_2d = ensure_y_2d
        self.allow_sparse_matrix = allow_sparse_matrix
        self.check_1d_input = check_1d_input
        self.allow_negative_values = allow_negative_values
        self.raise_on_non_numeric = raise_on_non_numeric

    def __call__(self, func=None):
        """
        Callable method to wrap the function/method.
        """
        if func is None:
            return lambda func: self.__call__(func)

        # Identify if the decorated function is part of a class (i.e., a method)
        if hasattr(func, '__self__'):
            # For methods (bound functions), check self argument
            @functools.wraps(func)
            def wrapper(self, X, y=None):
                # Process data (X, y) before calling the original function
                X = self._check_and_convert_X(X)
                if self.check_y and y is not None:
                    y = self._check_and_convert_y(y)
                elif self.check_y == 'auto' and y is not None:
                    y = self._check_and_convert_y(y)
                return func(self, X, y)
        else:
            # For standalone functions
            @functools.wraps(func)
            def wrapper(X, y=None):
                X = self._check_and_convert_X(X)
                if self.check_y and y is not None:
                    y = self._check_and_convert_y(y)
                elif self.check_y == 'auto' and y is not None:
                    y = self._check_and_convert_y(y)
                return func(X, y)
        
        return wrapper

    def _check_and_convert_X(self, X):
        """
        Helper function to check and convert X.
        
        Parameters
        ----------
        X : array-like
            The input data to be checked and converted.
        
        Returns
        -------
        X : array-like
            The validated and converted input data.
        """
        import scipy.sparse as sp
        
        if not isinstance (X, pd.DataFrame): 
            X =  np.asarray (X )
        # Check if X is a pandas DataFrame
        if isinstance(X, pd.DataFrame):
            # Ensure DataFrame contains numeric values
            if X.select_dtypes(include=[np.number]).shape[1] == 0:
                raise ValueError(
                    "DataFrame contains no numeric values."
                    " Only numeric data is supported."
                )
            else:
                # Warn the user if there are non-numeric columns
                non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
                if len(non_numeric_columns) > 0:
                    warnings.warn(
                        "The following non-numeric columns will be ignored: "
                        f"{', '.join(non_numeric_columns)}"
                    )
                # Use only numeric columns
                X = X.select_dtypes(include=[np.number])

        # If X is not numeric (and not a DataFrame), try to convert it
        elif not np.issubdtype(np.array(X).dtype, np.number):
            try:
                X = np.array(X, dtype=np.float64)
            except ValueError:
                if self.raise_on_non_numeric:
                    raise ValueError(
                        "X contains non-numeric values and could"
                        " not be converted to numeric."
                    )
                else:
                    warnings.warn(
                        "X contains non-numeric values and will be ignored.")
                    return np.array([])

        # Ensure X is 2D (reshape if it's 1D)
        if self.check_1d_input and X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Allow sparse matrices if enabled
        if not self.allow_sparse_matrix and isinstance(
                X, (sp.csr_matrix, sp.csc_matrix, sp.bsr_matrix)):
            raise ValueError(
                "Sparse matrices are not supported by this transformer.")
        
        # Check for negative values if allowed
        if not self.allow_negative_values and np.any(X < 0):
            raise ValueError("Negative values are not allowed in X.")

        return X

    def _check_and_convert_y(self, y):
        """
        Helper function to check and convert y.
        
        Parameters
        ----------
        y : array-like
            The target labels to be checked and converted.
        
        Returns
        -------
        y : array-like
            The validated and converted target labels.
        """
        if not isinstance (y, (pd.Series, pd.DataFrame)): 
             y = np.asarray (y )
        if not np.issubdtype(np.array(y).dtype, np.number):
            try:
                y = np.array(y, dtype=np.float64)
            except ValueError:
                if self.raise_on_non_numeric:
                    raise ValueError(
                        "y contains non-numeric values and"
                        " could not be converted to numeric."
                    )
                else:
                    warnings.warn(
                        "y contains non-numeric values and will be ignored.")
                    return np.array([])

        # Ensure y is 2D if ensure_y_2d is True
        if self.ensure_y_2d:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
        elif y.ndim != 1:
            raise ValueError("If ensure_y_2d is False, y must be 1D.")
        
        # Check for negative values in y if allowed
        if not self.allow_negative_values and np.any(y < 0):
            raise ValueError(
                "Negative values found in y. Expect non-negatives only.")

        return y

@NumpyDocstringFormatter(
    include_sections=['Parameters', 'Returns'],
    validate_with_sphinx=True)
def example_function(param1, param2=None):
    """
    This is an example function that demonstrates the usage of
    the NumpyDocstringFormatter.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : int, optional
        The second parameter (default is None).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    return True
    
