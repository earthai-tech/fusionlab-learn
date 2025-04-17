# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides utilities to check and validate Keras models,
including optional deep inspection of essential Keras
capabilities.
"""
import warnings 
from typing import Any, Callable, Optional

from ..utils.validator import has_required_attributes
from ..utils.deps_utils import ensure_pkg
from . import KERAS_DEPS,  KERAS_BACKEND, dependency_message

if KERAS_BACKEND: 
    Layer=KERAS_DEPS.Layer
    Loss=KERAS_DEPS.Loss 
    Model= KERAS_DEPS.Model 
    Sequential=KERAS_DEPS.Sequential
    
DEP_MSG = dependency_message('nn.keras_validator') 
    
__all__=[
    "is_keras_model", "validate_keras_model",
    "check_keras_model_status", "validate_keras_layer", 
    "is_keras_layer", "validate_keras_loss", "is_keras_loss"
 ]

def validate_keras_loss(
    loss_obj: Any,
    custom_check: Optional[Callable[[Any], bool]] = None,
    deep_check: bool = False,
    error: str = "raise",
    ops: str = "check_only"
) -> bool:
    """
    Validate an object as a Keras Loss, optionally with a custom check
    and/or deep inspection of required functionalities.

    Parameters
    ----------
    loss_obj : Any
        The object to validate as a Keras loss.
    custom_check : Callable[[Any], bool], optional
        A user-provided callable returning True if `loss_obj`
        meets certain criteria. If None, no custom check is performed.
    deep_check : bool, optional
        If True, verify that the object has attributes
        typically expected of Keras `Loss` subclasses (e.g.,
        'call', 'from_config', etc.). Defaults to False.
    error : {'warn', 'raise'}, optional
        When `ops='validate'`, indicates how to handle
        validation failure:
          - 'warn': issue a warning but return False
          - 'raise': raise an exception
        Defaults to 'raise'.
    ops : {'check_only', 'validate'}, optional
        - 'check_only': Return True/False if the object
          is valid. No exceptions unless from custom_check.
        - 'validate': Return the loss object if valid,
          otherwise warn or raise (per `error`). Defaults
          to 'check_only'.

    Returns
    -------
    bool or loss_obj
        If `ops='check_only'`, returns True if all checks pass,
        False otherwise.
        If `ops='validate'`, returns the `loss_obj` if it passes,
        otherwise warns or raises an exception, returning False.

    Raises
    ------
    TypeError
        If the object is not a Keras `Loss`, or if `deep_check` fails
        and `error='raise'`.
    ValueError
        If `custom_check` is provided and fails, and `error='raise'`.
    ImportError
        If TensorFlow is not installed.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from tensorflow.keras.losses import MeanSquaredError
    >>> from gofast.nn.keras_validator import validate_keras_loss
    >>> mse = MeanSquaredError()
    >>> validate_keras_loss(mse)
    True

    >>> # Force deep_check
    ... validate_keras_loss(mse, deep_check=True)
    True

    >>> # Provide a custom check
    ... custom_name_check = lambda l: l.name == 'mean_squared_error'
    >>> validate_keras_loss(mse, custom_check=custom_name_check)
    True

    >>> # If user wants to raise an error if checks fail
    ... validate_keras_loss(mse, deep_check=True,
    ...                     error='raise', ops='validate')
    <tensorflow.python.keras.losses.MeanSquaredError object at 0x...>
    """
    # 1. Check if the object is a recognized Keras Loss
    if not is_keras_loss(loss_obj):
        msg = "Provided object is not a Keras Loss."
        if ops == "validate":
            if error == "raise":
                raise TypeError(msg)
            else:
                warnings.warn(msg, category=UserWarning)
                return False
        return False

    # 2. Optional deep check for required attributes
    if deep_check:
        required = ["call", "from_config"]  
        # or any others that can be considered essential
        if not has_required_attributes(loss_obj, required):
            msg = (f"Loss object lacks some required attributes: {required}.")
            if ops == "validate":
                if error == "raise":
                    raise TypeError(msg)
                else:
                    warnings.warn(msg, category=UserWarning)
                    return False
            return False

    # 3. Optional custom check
    if custom_check is not None:
        try:
            if not custom_check(loss_obj):
                msg = "Custom check returned False."
                if ops == "validate":
                    if error == "raise":
                        raise ValueError(msg)
                    else:
                        warnings.warn(msg, category=UserWarning)
                        return False
                return False
        except Exception as exc:
            msg = f"Custom check raised an exception: {exc}"
            if ops == "validate":
                if error == "raise":
                    raise ValueError(msg) from exc
                else:
                    warnings.warn(msg, category=UserWarning)
                    return False
            return False

    # Passed all checks
    if ops == "validate":
        return loss_obj
    return True

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def is_keras_loss(loss_obj: Any) -> bool:
    """
    Check if the given object is a Keras Loss instance.

    This function attempts a lazy import of TensorFlow or
    Keras dependencies, then returns True if `loss_obj`
    is an instance of `tf.keras.losses.Loss`.

    Parameters
    ----------
    loss_obj : Any
        The object to check.

    Returns
    -------
    bool
        True if `loss_obj` is a recognized Keras loss,
        otherwise False.

    Raises
    ------
    ImportError
        If TensorFlow or Keras is not installed.
    """
    return isinstance(loss_obj, Loss)

def validate_keras_layer(
    layer: Any,
    custom_check: Optional[Callable[[Any], bool]] = None,
    deep_check: bool = False,
    error: str = "raise",
    ops: str = "check_only",
) -> bool:
    """
    Validate an object as a Keras Layer, optionally with a custom check 
    and/or deep inspection of essential layer functionalities.

    Parameters
    ----------
    layer : Any
        The object to validate as a Keras layer.
    custom_check : Callable[[Any], bool], optional
        A callback that returns True if `layer` passes custom 
        criteria. If None, no custom check is done.
    deep_check : bool, optional
        If True, verify that `layer` has certain expected 
        attributes (e.g. 'build', 'call'). Defaults to False.
    error : {'warn', 'raise'}, optional
        How to handle validation failures when 
        ``ops='validate'``:
          - 'warn': issue a warning but return False
          - 'raise': raise an exception
        Defaults to 'raise'.
    ops : {'check_only', 'validate'}, optional
        The mode of operation:
          - 'check_only': Return True/False if the object is 
                          valid. (No exceptions raised 
                          unless from the custom_check.)
          - 'validate':   Return the layer itself if all checks 
                          pass; otherwise warn or raise an error 
                          (depending on `error`). Defaults to 
                          'check_only'.

    Returns
    -------
    bool or layer
        If ops='check_only', returns True if `layer` passes 
        all checks, False otherwise.
        If ops='validate':
          - If the layer fails any check, it either warns or 
            raises, returning False or None if 'warn', or 
            raising if 'raise'.
          - If it passes, returns the actual layer object.

    Raises
    ------
    TypeError
        If the object is not a Keras layer, or if `deep_check` 
        fails in 'raise' mode.
    ValueError
        If `custom_check` is provided and fails (in 'raise' mode).
    ImportError
        If TensorFlow is not installed.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from gofast.nn.keras_validator import validate_keras_layer
    >>> layer = tf.keras.layers.Dense(10)
    >>> validate_keras_layer(layer) 
    True

    >>> # Force deep check
    ... validate_keras_layer(layer, deep_check=True)
    True

    >>> # Provide a custom check
    ... custom_units_check = lambda lyr: getattr(lyr, 'units', 0) >= 10
    >>> validate_keras_layer(layer, custom_check=custom_units_check)
    True

    >>> # If user wants to raise an error if checks fail
    ... validate_keras_layer(layer, deep_check=True, 
    ...                      error='raise', ops='validate')
    <tf.keras.layers.core.Dense object at ...>
    """
    # 1. Check if it's a Keras layer
    if not is_keras_layer(layer):
        msg = "Provided object is not a Keras layer."
        if ops == "validate":
            if error == "raise":
                raise TypeError(msg)
            warnings.warn(msg, category=UserWarning)
            return False
        return False  # ops='check_only'

    # 2. If deep_check is True, ensure essential attributes
    if deep_check:
        required = ["build", "call"]
        if not has_required_attributes(layer, required):
            msg = ("Layer lacks one or more essential methods: " 
                   f"{required}")
            if ops == "validate":
                if error == "raise":
                    raise TypeError(msg)
                warnings.warn(msg, category=UserWarning)
                return False
            return False

    # 3. If custom_check is provided, invoke it
    if custom_check is not None:
        try:
            if not custom_check(layer):
                # The custom check returned False
                msg = "Custom layer check returned False."
                if ops == "validate":
                    if error == "raise":
                        raise ValueError(msg)
                    warnings.warn(msg, category=UserWarning)
                    return False
                return False
        except Exception as exc:
            # The custom check raised an Exception
            msg = f"Custom check raised an exception: {exc}"
            if ops == "validate":
                if error == "raise":
                    raise ValueError(msg) from exc
                warnings.warn(msg, category=UserWarning)
                return False
            return False

    # If we reach here, the layer passes all checks
    if ops == "validate":
        # Return the layer itself if all checks pass
        return layer
    
    # ops == "check_only"
    return True

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def is_keras_layer(layer: Any) -> bool:
    """
    Check if the provided object is a Keras Layer.

    This function attempts a lazy import of TensorFlow (or 
    a Keras backend) to verify that ``layer`` inherits from
    ``tf.keras.layers.Layer``.

    Parameters
    ----------
    layer : Any
        The object to check.

    Returns
    -------
    bool
        True if `layer` is an instance of 
        ``tf.keras.layers.Layer``, otherwise False.

    Raises
    ------
    ImportError
        If TensorFlow (or Keras) is not installed.
    """

    return isinstance(layer, Layer)


@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def is_keras_model(model: Any) -> bool:
    """
    Check if the provided object is an instance of a Keras model.

    This function imports TensorFlow (if installed) and returns
    True if ``model`` is an instance of ``tf.keras.models.Model``
    or ``tf.keras.Sequential``, False otherwise.

    Parameters
    ----------
    model : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a recognized Keras model,
        False otherwise.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.

    Notes
    -----
    - Internally performs a lazy import of TensorFlow
      to check model types.
    """
    return isinstance(model, (Model, Sequential))

def validate_keras_model(
    model: Any,
    custom_check: Optional[Callable[[Any], bool]] = None,
    deep_check: bool = False,
    raise_exception: bool = False
) -> bool:
    """
    Validate an object as a Keras model, optionally with a custom check
    and/or deep inspection of Keras functionalities.

    This function ensures the object is recognized as a Keras model
    and can optionally:
      1. Check essential methods (``fit``, ``predict``, ``compile``,
         ``summary``) if `deep_check` is True.
      2. Run a user-supplied `custom_check` callback.
      3. Raise exceptions instead of returning False, if
         `raise_exception` is True.

    Parameters
    ----------
    model : Any
        The object to validate as a Keras model.
    custom_check : Callable[[Any], bool], optional
        A callback that returns True if the model passes the custom
        criteria, or False (or raises) if it fails. If None, no custom
        check is done. Defaults to None.
    deep_check : bool, optional
        If True, verifies that `model` has key methods associated with
        Keras (`fit`, `predict`, `compile`, and `summary`). Defaults
        to False.
    raise_exception : bool, optional
        If True, raises an exception upon validation failure instead
        of returning False. Defaults to False.

    Returns
    -------
    bool or Model 
        True if the model passes all checks, False otherwise. If
        `raise_exception` is True and a check fails, this function
        raises instead of returning a model.

    Raises
    ------
    TypeError
        If the object is not a Keras model, or if the deep check
        fails (when `raise_exception` is True).
    ValueError
        If the `custom_check` is provided and fails
        (when `raise_exception` is True).
    ImportError
        If TensorFlow is not installed.

    Examples
    --------
    >>> from tensorflow.keras.layers import Dense
    >>> from tensorflow.keras.models import Sequential
    >>> from gofast.nn.keras_validator import validate_keras_model
    >>> # Suppose we have:
    ... model = Sequential([Dense(2)])
    >>> # Basic validation:
    ... validate_keras_model(model)
    True

    >>> # Custom check: model must have more than 1 layer
    ... custom_layer_check = lambda m: len(m.layers) > 1
    >>> validate_keras_model(model, custom_check=custom_layer_check)
    False

    >>> # Enforce deep inspection:
    ... validate_keras_model(model, deep_check=True)
    True
    """
    if not is_keras_model(model):
        msg = "Provided object is not a Keras model."
        if raise_exception:
            raise TypeError(msg)
        return False

    if deep_check:
        # Ensure essential Keras methods exist
        required = ['fit', 'predict', 'compile', 'summary']
        if not has_required_attributes(model, required):
            msg = (
                "Model lacks one or more essential Keras functionalities: "
                f"{required}."
            )
            if raise_exception:
                raise TypeError(msg)
            return False

    if custom_check is not None:
        try:
            if not custom_check(model):
                # The custom check returned False
                msg = "Custom check returned False."
                if raise_exception:
                    raise ValueError(msg)
                return False
        except Exception as exc:
            # The custom check raised an Exception
            msg = f"Custom check raised an exception: {exc}"
            if raise_exception:
                raise ValueError(msg) from exc
            return False

    return model

def check_keras_model_status(
    model,
    mode: str = "build",
    ops: str = "check_only",
    error: str = "warn",
    err_msg: str = None
) -> bool:
    """
    Check whether a Keras model has been compiled, built, or trained,
    and optionally validate that status, raising a warning or error
    if not satisfied.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to check.
    mode : {'fit', 'build', 'compile'}, optional
        The status to verify:
          - 'compile': Checks if the model is compiled
                       (i.e., has a non-None optimizer).
          - 'build':   Checks if the model was built
                       (i.e., ``model.built`` is True).
          - 'fit':     Checks if the model has been
                       trained (i.e., ``optimizer.iterations > 0``).
        Defaults to ``'build'``.
    ops : {'check_only', 'validate'}, optional
        The action to take with the result:
          - 'check_only': Return True/False. (No side effects.)
          - 'validate':   If the check fails, either raise an error
                          or issue a warning, depending on ``error``.
        Defaults to ``'check_only'``.
    error : {'warn', 'raise'}, optional
        If ``ops='validate'`` and the check fails:
          - 'warn':  issue a warning
          - 'raise': raise a ValueError
        Defaults to ``'warn'``.
    err_msg : str, optional
        A custom message used if the status fails
        under validation. If None, a default message is used.

    Returns
    -------
    bool
        If ``ops='check_only'``, returns True if the model
        passes the requested status check, False otherwise.
        If ``ops='validate'``:
          - If the check fails, it either warns or raises,
            so this function won't return. 
          - If the check passes, returns True.

    Notes
    -----
    1. For ``mode='compile'``, we check that
       ``model.optimizer`` is not None, which typically
       means the model was compiled with an optimizer.
    2. For ``mode='build'``, we check the boolean
       ``model.built`` property. This becomes True when the
       model has a known input shape (e.g., from explicit
       build or a forward pass).
    3. For ``mode='fit'``, we look at
       ``model.optimizer.iterations``. If it's greater
       than zero, at least one training step occurred.
       This may not detect custom training loops or custom
       optimizers that do not increment that variable.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from tensorflow.keras import layers
    >>> from gofast.nn.keras_validator import check_keras_model_status
    >>> # Minimal model
    ... model = tf.keras.Sequential([
    ...     layers.Dense(1, input_shape=(3,))
    ... ])
    >>> # Check 'build' status
    ... check_keras_model_status(model, mode='build')
    False

    >>> # Build the model by calling it once
    ... model(tf.random.normal((1, 3)))
    >>> check_keras_model_status(model, mode='build')
    True

    >>> # Not compiled yet
    ... check_keras_model_status(model, mode='compile')
    False

    >>> # Compile
    ... model.compile(optimizer='adam', loss='mse')
    >>> check_keras_model_status(model, mode='compile')
    True

    >>> # Not fit yet
    ... check_keras_model_status(model, mode='fit')
    False

    >>> # Train it
    ... model.fit(tf.random.normal((10, 3)),
    ...           tf.random.normal((10, 1)),
    ...           epochs=1, verbose=0)
    >>> check_keras_model_status(model, mode='fit')
    True
    """
    # Try importing TensorFlow (if not already imported)
    if not KERAS_BACKEND: 
    # try:
    #     import tensorflow as tf  # noqa: F401
    # except ImportError as e:
        raise ImportError(
            "TensorFlow(prefered)/Keras is not installed. Please install"
            " one of them to proceed."
        ) # from e

    # Validate mode
    mode = mode.lower().strip()
    if mode not in {"compile", "build", "fit"}:
        raise ValueError(
            "`mode` must be one of {'compile', 'build', 'fit'}."
        )

    # Determine the model's status according to mode
    if mode == "compile":
        # Compiled if there's a non-None optimizer
        status = (model.optimizer is not None)

    elif mode == "build":
        # 'built' if model.built is True
        status = bool(model.built)

    else:  # mode == "fit"
        # Must have an optimizer and show iteration > 0
        if getattr(model, "optimizer", None) is None:
            status = False
        else:
            try:
                iterations = model.optimizer.iterations.numpy()
                status = (iterations > 0)
            except AttributeError:
                # If a custom optimizer has no .iterations
                status = False

    # Handle validation if ops == 'validate'
    if ops == "validate":
        if not status:
            # Construct default or custom error/warn message
            if err_msg is None:
                err_msg = f"Model does not satisfy '{mode}' status."

            if error == "warn":
                warnings.warn(err_msg)
    
            elif error == "raise":
                raise ValueError(err_msg)
                
        return model  # If passes, or if we didn't fail, return the model 

    # If ops == 'check_only', just return the boolean
    return status