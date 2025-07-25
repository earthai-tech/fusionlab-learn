# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides utilities for managing compatibility between TensorFlow's Keras and 
standalone Keras. It includes functions and classes for dynamically importing 
Keras dependencies and checking the availability of TensorFlow or Keras.
"""
import os
import re
import logging
import warnings 
import importlib
import numpy as np 
from functools import wraps
from contextlib import contextmanager 
from typing import Callable, Optional, Any, Union 
from .._configs import TENSORFLOW_CONFIG

# --- TensorFlow Setup ---

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 0 = all messages are logged (default)
# 1 = filter out INFO messages
# 2 = filter out INFO and WARNING messages
# 3 = filter out INFO, WARNING, and ERROR messages

# Disable OneDNN logs or usage (Optional):
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# # '3' shows only errors, suppressing warnings and infos
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Convenience flag: should *our* warnings be shown?
_tf_logs_suppressed = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "0") != "0"

# Placeholder for tf.Tensor type hint
tf = None
Tensor = None 

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
else: 
    # If TF is imported successfully, detect the version:
    version_str = tf.__version__
    major_version = int(version_str.split('.')[0])

    if major_version >= 2:
        # For TF 2.x:
        tf.get_logger().setLevel('ERROR')
    else:
        # For older TF 1.x style (still works in tf.compat.v1):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
try:
    # First, try the new location (2.15+):
    from tensorflow.keras.saving import register_keras_serializable
    saving_module = "saving"
except ImportError:
    # Fallback: In older TF/Keras, `register_keras_serializable` is in `utils`.
    from tensorflow.keras.utils import register_keras_serializable # noqa: F401
    saving_module = "utils"

# I want to use :
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # 0 = all messages are logged (default)
    # 1 = filter out INFO messages
    # 2 = filter out INFO and WARNING messages
    # 3 = filter out INFO, WARNING, and ERROR messages

    # Disable OneDNN logs or usage (Optional):
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    # # '3' shows only errors, suppressing warnings and infos
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # so when value is 0 warnings .warn is triggered below 
try:
    tf_spec = importlib.util.find_spec("tensorflow")
    if tf_spec is not None:
        tf = importlib.util.module_from_spec(tf_spec)
        tf_spec.loader.exec_module(tf)
        if hasattr(tf, 'Tensor'): # Basic check
            Tensor = tf.Tensor
        HAS_TF = True
except ImportError:
    if not _tf_logs_suppressed:
        warnings.warn(
            "TensorFlow is not installed. Some functionalities in "
            "fusionlab.compat.tf will be limited or fall back to NumPy.",
            ImportWarning
        )
except Exception as e:
    if not _tf_logs_suppressed:
        warnings.warn(
            f"An error occurred during TensorFlow import: {e}. "
            "TensorFlow-dependent functionalities might not work.",
            ImportWarning
        )

__all__ = [
    'KerasDependencies',
    'check_keras_backend',
    'import_keras_dependencies',
    'import_keras_function',
    'standalone_keras', 
    'optional_tf_function', 
    'suppress_tf_warnings', 
    'has_wrappers'
]

class KerasDependencies:
    def __init__(
        self,
        extra_msg=None,
        error='warn'
    ):
        self.extra_msg = extra_msg
        self.error = error
        self._dependencies = {}
        self._check_tensorflow()

    def _check_tensorflow(self):
        try:
            import_optional_dependency(
                'tensorflow',
                extra=self.extra_msg
            )
            if self.error == 'warn':
                warnings.warn(
                    "TensorFlow is installed.",
                    UserWarning
                )
        except ImportError as e:
            warnings.warn(f"{self.extra_msg}: {e}")

    def __getattr__(
        self,
        name
    ):
        if name not in self._dependencies:
            self._dependencies[name] = self._import_dependency(name)
        return self._dependencies[name]

    def _import_dependency(
        self,
        name
    ):
        """Imports a dependency using the central TENSORFLOW_CONFIG."""
        if name in TENSORFLOW_CONFIG:
            module_info = TENSORFLOW_CONFIG[name]

            # If the entry is a single tuple, it has one path.
            # If it's a tuple of tuples, it has multiple fallback paths.
            if isinstance(
                    module_info, tuple
                    ) and isinstance(module_info[0], str):
                module_name, function_name = module_info
                return import_keras_function(
                    module_name, function_name, error=self.error)
            else: # It's a tuple of potential paths
                 return import_keras_function(
                     module_info, name, error=self.error)

        raise AttributeError(
            f"'KerasDependencies' object has no attribute '{name}'. "
            "Ensure it is defined in `fusionlab/_configs.py`."
        )
        
class TFConfig:
    """
    A configuration class that manages TensorFlow's dimension compatibility 
    by toggling the `ndim` attribute behavior based on the TensorFlow version.
    
    This class provides a means to control whether the `ndim` attribute should 
    function as it did in older versions of TensorFlow or if it should utilize 
    the newer `shape` attribute to determine the number of dimensions.
    
    Attributes:
    -----------
    compat_ndim_enabled : bool
        A flag to enable or disable compatibility mode for the `ndim` attribute.
        When enabled, the `ndim` behavior is overridden to use `get_ndim` instead.
        
    _original_ndim : callable or None
        A reference to the original `ndim` method of the TensorFlow `Tensor` 
        class, if available. This is used for restoring the original
        behavior when compatibility mode is disabled. If `ndim` is not
        present (new TensorFlow version), it will be `None`.
    
    Methods:
    --------
    compat_ndim_enabled(value)
        Sets the flag to enable or disable compatibility mode based
        on the value provided.
        
    enable_ndim_compatibility()
        Replaces TensorFlow's `ndim` with a compatibility method that
        uses `get_ndim`.
        
    disable_ndim_compatibility()
        Restores the original `ndim` method of TensorFlow if available.
    """
    
    def __init__(self):
        """
        Initializes the TFConfig instance and checks the TensorFlow version.
        
        It verifies if the `ndim` attribute exists on TensorFlow's 
        `Tensor` class. If the attribute exists, it stores the original
        method. If not, it prepares for newer TensorFlow versions where
        `ndim` is not available.
        
        Attributes:
        -----------
        _original_ndim : callable or None
            Stores the original `ndim` method, or `None` if it doesn't exist.
        _set_compat_ndim_tensor : bool
            A flag to toggle compatibility mode for `ndim`.
        """
  
        # self.compat_ndim_enabled was removed to prevent recursion.
        # Any version-specific TF logic should be handled within
        # individual utility functions using tf.version.VERSION if needed.
        if HAS_TF:
            # Example: Log TF version or set TF-specific flags
            # print(f"TFConfig initialized with TensorFlow version: {tf.version.VERSION}")
            pass
        else:
            # print("TFConfig initialized, but TensorFlow not found.")
            pass
        
        # # Check TensorFlow version before accessing tf.Tensor.ndim
        # if hasattr(tf.Tensor, 'ndim'):
        #     self._original_ndim = tf.Tensor.ndim
        # else:
        #     # Indicate that ndim doesn't exist in TensorFlow
        #     self._original_ndim = None  

        # self._set_compat_ndim_tensor = False

    @property
    def compat_ndim_enabled(self):
        """
        Retrieves the current status of the `ndim` compatibility flag.
        
        Returns:
        --------
        bool
            The current status of the compatibility mode. `True` means that 
            `get_ndim` is used, and `False` means the original behavior
            is restored.
        """
        return self._set_compat_ndim_tensor

    @compat_ndim_enabled.setter
    def compat_ndim_enabled(self, value):
        """
        Sets the compatibility mode for TensorFlow's `ndim` attribute.
        
        Parameters:
        -----------
        value : bool
            If `True`, enable the compatibility mode where `get_ndim` is used. 
            If `False`, restore the original `ndim` behavior.
        """
        self._set_compat_ndim_tensor = value
        # Apply changes when the flag is set or unset
        if value:
            self.enable_ndim_compatibility()
        else:
            self.disable_ndim_compatibility()

    def enable_ndim_compatibility(self):
        """
        Enables compatibility mode for TensorFlow's `ndim` attribute.
        
        This method overrides TensorFlow's default `ndim` property with
        a custom method that uses the `get_ndim` function from the
        `fusionlab.compat.tf` module, allowing compatibility with both
        old and new TensorFlow versions.
        """
        # Define a compatibility method for ndim that uses get_ndim
        def compat_ndim(self):
            return get_ndim(self)

        # Override TensorFlow's ndim method with compat_ndim
        tf.Tensor.ndim = property(compat_ndim)

    def disable_ndim_compatibility(self):
        """
        Disables compatibility mode and restores the original `ndim` method.
        
        If the original `ndim` method is available 
        (for older TensorFlow versions), 
        it will be restored. Otherwise, if `ndim` was not present,
        no changes are made.
        """
        # Restore the original ndim method if it exists
        if self._original_ndim:
            tf.Tensor.ndim = self._original_ndim
        else:
            # If the original ndim was never there, don't restore anything
            # since it does not exist, no neeed. Just for consistency.
            delattr(tf.Tensor, 'ndim')  # Optional, remove any existing ndim definition
            
#--------------------------------------------------
# Instantiate the global configuration object
Config = TFConfig()
# ------------------------------------------------

def import_keras_function(
        module_path_info, 
        function_name,
        error='warn'
    ):
    """
    Attempts to import a function from a list of possible module paths.
    """
    if isinstance(module_path_info, str):
        # If only one path is provided, wrap it in a list
        paths_to_try = [(module_path_info, function_name)]
    elif isinstance(module_path_info, tuple) and isinstance(
            module_path_info[0], str):
        # Handle single tuple case
        paths_to_try = [module_path_info]
    else: # It's a tuple of tuples
        paths_to_try = module_path_info

    for module_name, func_name in paths_to_try:
        try:
            # Handle standalone TensorFlow functions
            if module_name.startswith('tensorflow'):
                tf_module = importlib.import_module(module_name)
                return getattr(tf_module, func_name)
            
            # Try importing from tensorflow.keras first
            tf_keras_module = importlib.import_module(
                f'tensorflow.keras.{module_name}')
            return getattr(tf_keras_module, func_name)
            
        except ImportError:
            # Fallback to standalone Keras
            try:
                keras_module = importlib.import_module(
                    f'keras.{module_name}')
                return getattr(keras_module, func_name)
            except ImportError:
                continue # Try the next path in the list
                
    # If all paths fail
    msg = (f"Cannot import '{function_name}' from any of the specified paths:"
           f" {paths_to_try}. Ensure TensorFlow or Keras is installed.")
    if error == 'raise':
        raise ImportError(msg)
    elif error == 'warn':
        warnings.warn(msg)
    return None


def _import_keras_function(
    module_name,
    function_name,
    error='warn'
):
    try:
        # Import directly from TensorFlow for TensorFlow-specific functions
        if module_name.startswith('tensorflow'):
            tf_module = importlib.import_module(module_name)
            function = getattr(tf_module, function_name)
            if error == 'warn':
                warnings.warn(
                    f"Using TensorFlow for {function_name} from {module_name}"
                )
            return function

        # Import from tensorflow.keras
        tf_keras_module = __import__(
            'tensorflow.keras.' + module_name,
            fromlist=[function_name]
        )
        function = getattr(tf_keras_module, function_name)
        if error == 'warn':
            warnings.warn(
                f"Using TensorFlow Keras for {function_name} from {module_name}"
            )
        return function
    except ImportError:
        try:
            # Fallback to standalone Keras
            keras_module = __import__(
                'keras.' + module_name,
                fromlist=[function_name]
            )
            function = getattr(keras_module, function_name)
            if error == 'warn':
                warnings.warn(
                    f"Using Keras for {function_name} from {module_name}"
                )
            return function
        except ImportError:
            raise ImportError(
                f"Cannot import {function_name} from {module_name}. "
                "Ensure TensorFlow or Keras is installed."
            )

def import_optional_dependency(
    package_name,
    error='warn',
    extra=None
):
    try:
        module = importlib.import_module(package_name)
        return module
    except ImportError as e:
        message = f"{package_name} is not installed"
        if extra:
            message = f"{extra}: {e}"
        else:
            message = (
                f"{message}: {e}. Use pip or conda to install it."
            )
        if error == 'warn':
            warnings.warn(message)
            return None
        elif error == 'raise':
            raise ImportError(message)
        else:
            raise ValueError(
                "Parameter 'error' must be either 'warn' or 'raise'."
            )

def import_keras_dependencies(
    extra_msg=None,
    error='warn'
):
    return KerasDependencies(extra_msg, error)

def check_keras_backend(
    error='warn'
):
    try:
        importlib.import_module('tensorflow')
        return 'tensorflow'
    except ImportError:
        try:
            importlib.import_module('keras')
            return 'keras'
        except ImportError as e:
            message = (
                "Neither TensorFlow nor Keras is installed."
            )
            if error == 'warn':
                warnings.warn(message)
            elif error == 'raise':
                raise ImportError(message) from e
            return None

def tf_debugging_assert_equal(
    x: Any,
    y: Any,
    msg_fmt: str,
    *args,
    summarize: Optional[int] = None
) -> Any:
    """
    Wrapper for tf.debugging.assert_equal that supports
    both percent-style and {}-style formatting in
    graph/eager modes, with Python fallback.
    """
    # Fallback when TensorFlow is not available
    if not HAS_TF:
        # Static message formatting
        if '%' in msg_fmt:
            message = msg_fmt % args
        else:
            curly_fmt = re.sub(r'%[ds]', '{}', msg_fmt)
            message   = curly_fmt.format(*args)

        # Compare arrays or scalars via numpy
        try:
            equal = np.array_equal(x, y)
        except Exception:
            equal = x == y

        if not equal:
            raise AssertionError(message)
        return None

    # TensorFlow available: import debugging ops
    import tensorflow as tf
    # from tensorflow.python.ops import debugging_ops

    # Detect if any arg is a TF tensor for dynamic formatting
    use_tf_fmt = any(isinstance(a, tf.Tensor) for a in args)

    # Static formatting when no TF tensors present
    if '%' in msg_fmt and not use_tf_fmt:
        message = msg_fmt % args
    else:
        # Convert %d/%s placeholders to {} for tf.strings.format()
        curly_fmt = re.sub(r'%[ds]', '{}', msg_fmt)
        fmt_args  = []
        for a in args:
            if isinstance(a, tf.Tensor):
                fmt_args.append(tf.strings.as_string(a))
            else:
                fmt_args.append(
                    tf.constant(str(a), dtype=tf.string)
                )
        # Build dynamic message tensor
        message = tf.strings.format(curly_fmt, tuple(fmt_args))

    # Execute the TensorFlow assertion
    return tf.debugging.assert_equal(
        x,
        y,
        message   = message,
        summarize = summarize
    )


def standalone_keras(module_name):
    """
    Tries to import the specified module from tensorflow.keras or 
    standalone keras.

    Parameters
    ----------
    module_name : str
        The name of the module to import (e.g., 'activations', 'layers', etc.).

    Returns
    -------
    module
        The imported module from tensorflow.keras or keras.

    Raises
    ------
    ImportError
        If neither tensorflow.keras nor standalone keras is installed or if
        the specified module does not exist in both frameworks.
        
    Examples
    ---------
    # Usage example
    try:
        activations = import_keras_module("activations")
        print("Successfully loaded activations module from:", activations)
    except ImportError as e:
        print(e)
            
    """
    try:
        # Try importing from tensorflow.keras
        import tensorflow.keras as tf_keras
        return getattr(tf_keras, module_name)
    except (ImportError, AttributeError):
        try:
            # Fallback to standalone keras
            import keras
            return getattr(keras, module_name)
        except (ImportError, AttributeError):
            raise ImportError(
                f"Module '{module_name}' could not be imported from either "
                f"tensorflow.keras or standalone keras. Ensure that TensorFlow "
                f"or standalone Keras is installed and the module exists."
            )

def optional_tf_function(func: Callable) -> Callable:
    """
    A decorator that applies @tf.function if TensorFlow is available.
    Otherwise, it wraps the function to raise an ImportError when called.

    Parameters:
    - ``func`` (`Callable`): The function to decorate.

    Returns:
    - `Callable`: The decorated function.
    """
    if HAS_TF:
        return tf.function(func)
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            raise ImportError(
                "TensorFlow is required to use this function. "
                "Please install TensorFlow to proceed."
            )
        return wrapper

@contextmanager
def suppress_tf_warnings():
    """
    A context manager to temporarily suppress TensorFlow warnings.
    
    Usage:
        with suppress_tf_warnings():
            # TensorFlow operations that may generate warnings
            ...
    """
    if HAS_TF: 
        tf_logger = tf.get_logger()
        original_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)  # Suppress WARNING and INFO messages
        try:
            yield
        finally:
            tf_logger.setLevel(original_level)  # Restore original logging level
         
def _compat_add_weight(self, *args, **kwargs):
    
    if HAS_TF: 
        import tensorflow as tf
        from packaging.version import parse

        _TF_V = parse(tf.__version__)
        _NEEDS_KW = _TF_V >= parse("2.16.0")

        # Grab the original method once
        _orig_add_weight = tf.keras.layers.Layer.add_weight
        
    # TF>=2.16 wants only keyword args for shape/name
    if _NEEDS_KW and args:
        # args could be (shape,) or (name, shape)
        if len(args) == 1:
            kwargs.setdefault("shape", args[0])
        elif len(args) == 2 and isinstance(args[0], str):
            kwargs.setdefault("name", args[0])
            kwargs.setdefault("shape", args[1])
        else:
            raise ValueError(f"Unexpected args for add_weight: {args}")
        return _orig_add_weight(self, **kwargs)
    else:
        # TF≤2.15 works with positional
        return _orig_add_weight(self, *args, **kwargs)

# if HAS_TF: 
#     # Monkey‑patch the Layer base class
#     tf.keras.layers.Layer.add_weight = _compat_add_weight

# --- Compatibility Functions ---

def get_ndim(tensor: Union[Tensor, np.ndarray, Any]) -> int:
    """
    Reliably get the number of dimensions (rank) of a tensor-like object.

    Uses tf.rank for TensorFlow tensors and converts to a Python integer.
    Uses .ndim for NumPy arrays. Falls back to len(tensor.shape) for
    other objects with a .shape attribute.

    Parameters
    ----------
    tensor : tf.Tensor, np.ndarray, or object with .shape
        The input tensor-like object.

    Returns
    -------
    int
        The number of dimensions of the tensor.

    Raises
    ------
    TypeError
        If the input object type is not supported or lacks shape info.
    tf.errors.InvalidArgumentError
        May be raised by tf.rank if the tensor is invalid, though
        this function attempts to get a static rank first.
    """
    if HAS_TF and isinstance(tensor, tf.Tensor):
        # For TensorFlow tensors, tf.rank() is robust.
        # Try to get static rank first if available.
        if tensor.shape.rank is not None:
            return tensor.shape.rank
        # Fallback to dynamic rank (will be a 0-D Tensor)
        # This path is more for graph execution; direct int needed for Python logic
        # If called outside tf.function, .numpy() would work on rank_tensor.
        # For validation logic that needs an int *before* graph construction,
        # relying on static shape (tensor.shape.rank) is preferred.
        # If in graph and dynamic, the caller should use tf.rank directly.
        # This function aims to return a Python int.
        try:
            # This will work in eager mode or if rank is statically known
            return tf.rank(tensor).numpy()
        except AttributeError: # Not in eager, .numpy() fails
             # This indicates a symbolic tensor where static rank isn't known.
             # Returning a symbolic tensor from here would change API.
             # For now, raise if static rank isn't available for Python int return.
            raise ValueError(
                "Cannot determine static rank for symbolic TensorFlow tensor "
                "without eager execution. Use tf.rank() directly in graph "
                "operations if a symbolic rank tensor is acceptable."
            )
    elif isinstance(tensor, np.ndarray):
        return tensor.ndim
    elif hasattr(tensor, 'shape') and hasattr(tensor.shape, '__len__'):
        return len(tensor.shape)
    elif hasattr(tensor, "ndim"): # Fallback for other array-like
        return tensor.ndim

    raise TypeError(
        f"Input object of type {type(tensor)} must be a TensorFlow tensor,"
        f" a NumPy array, or have a 'shape' or 'ndim' attribute."
    )
def _get_ndim(tensor):
    """
    Compatibility function to retrieve the number of dimensions
    of a TensorFlow tensor.
    
    This function checks if the object is a TensorFlow tensor and whether
    it has the 'ndim'attribute. If not, it falls back to using the 
    `len(object.shape)` to get the number of dimensions.
    If TensorFlow is not available, it provides a generic approach for
    non-TensorFlow objects.

    Parameters
    ----------
    tensor : `tf.Tensor` or `np.ndarray` or any object with a `.shape` attribute
        The input tensor object whose number of dimensions is to be retrieved. 
        It can be a TensorFlow tensor, a NumPy array, or any object that exposes a 
        `.shape` attribute representing its dimensions.

    Returns
    -------
    int
        The number of dimensions of the tensor-like object.
        
    Notes
    -----
    - If TensorFlow is installed and the object is a TensorFlow tensor, 
      it uses the `ndim` attribute to retrieve the number of dimensions. 
    - If the `ndim` attribute is unavailable, the function falls back to using the 
      length of the `shape` attribute (i.e., `len(tensor.shape)`).
    - If TensorFlow is not installed, the function checks if the object 
      is a NumPy array or an object that exposes a `shape` attribute,
      and it uses `len(tensor.shape)` to retrieve the number of dimensions.

    Examples
    --------
    >>> from fusionlab.compat.tf import get_ndim
    >>> import tensorflow as tf
    >>> tensor = tf.constant([[1, 2], [3, 4]])
    >>> get_ndim(tensor)
    2

    >>> import numpy as np
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> get_ndim(arr)
    2

    See Also
    --------
    TensorFlow Documentation: https://www.tensorflow.org/api_docs/python/tf/Tensor
    NumPy Documentation: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html

    References
    ----------
    .. [1] TensorFlow API Documentation. 
           TensorFlow 2.x: https://www.tensorflow.org/api_docs/python/tf/Tensor.
    .. [2] NumPy Documentation.
           https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html.
    """
    
    # Check if TensorFlow is available
    if HAS_TF:
        if isinstance(tensor, tf.Tensor):
            # If TensorFlow tensor, return the number 
            # of dimensions using 'ndim' or 'shape'
            return getattr(tensor, 'ndim', len(tensor.shape))
    
    # If TensorFlow is not available, check if it is a NumPy
    # array or another object with a shape attribute
    if isinstance(tensor, np.ndarray):
        # For NumPy arrays, use len(tensor.shape)
        return len(tensor.shape)
    
    if hasattr(tensor, 'shape'):
        # For objects with a shape attribute, return the length of shape
        return len(tensor.shape)
    elif hasattr(tensor, "ndim"): 
        return tensor.ndim 
    # if we reach here then raise the eror 
    # Raise an exception if the input does not have the necessary attributes
    raise ValueError(
        "Input object must be a TensorFlow tensor,"
        " a NumPy array, or an object with a 'shape' attribute."
    )

def has_wrappers(
        error="warn", model=None, ops="check_only", 
        estimator_type="classifier", **kw):
    """
    Function to check if Keras wrappers are available 
    (either from scikeras or keras).

    Parameters:
    -----------
    error : str, default='warn'
        Specifies the behavior if the wrappers are not found:
        - 'raise': Raises an ImportError if neither scikeras nor
          keras.wrappers.scikit_learn is available.
        - 'warn' (default): Warns the user and returns True or False.
        - 'ignore': Returns True or False without any warning or error.

    model : estimator object, optional, default=None
        The model to be used if the operation is to 'build'.

    ops : str, default='check_only'
        Specifies the operation to perform:
        - 'check_only': Only checks if wrappers are available.
        - 'build': Checks the wrappers and builds the model 
          based on the provided estimator type.

    estimator_type : str, default='classifier'
        Type of estimator to build, either 'classifier' or 'regressor'.

    **kw : additional keyword arguments
        Additional arguments passed to the model's build function.

    Returns:
    --------
    bool or estimator object
        - If ops='check_only', returns True if wrappers are available, 
          otherwise False.
        - If ops='build', returns the built model (KerasClassifier or 
          KerasRegressor).
    """
    # Validate 'error' parameter
    if error not in {"raise", "warn", "ignore"}:
        raise ValueError(
            "Invalid error parameter. Choose 'raise', 'warn', or 'ignore'."
        )

    # Attempt to import from scikeras (the newer package)
    try:
        from scikeras.wrappers import KerasClassifier, KerasRegressor

        if error == "warn":
            warnings.warn(
                "Using scikeras.wrappers for KerasClassifier and "
                "KerasRegressor. Ensure you have 'scikeras' installed.",
                UserWarning
            )
        # If only checking, return True since wrappers exist
        if ops == "check_only":
            return True

    except ImportError:
        # Fallback to older Keras wrapper if scikeras is not available
        try:
            from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor  # noqa
            if ops == "check_only":
                return True

        except ImportError:
            # If both wrappers are not found, handle based on 'error' parameter
            if error == "raise":
                raise ImportError(
                    "Neither scikeras nor keras.wrappers.scikit_learn is "
                    "available. Please install scikeras or ensure Keras is "
                    "properly installed."
                )
            elif error == "warn":
                warnings.warn(
                    "Neither scikeras nor keras.wrappers.scikit_learn is "
                    "available. Please install scikeras or ensure Keras is "
                    "properly installed.",
                    UserWarning
                )
                return False
            elif error == "ignore":
                return False

    if estimator_type not in {'classifier', 'regressor'}: 
        raise ValueError(
            "Invalid estimator_type. Choose 'classifier' or 'regressor'."
            )
    # If ops == 'build', proceed to create and return the model based on estimator type
    if ops == "build":
        if model is None:
            raise ValueError("A model must be provided when ops='build'.")

        # Build model for classifier
        if estimator_type == "classifier":
            return KerasClassifier(build_fn=model, **kw)
        
        # Build model for regressor
        elif estimator_type == "regressor":
            return KerasRegressor(build_fn=model, **kw)
        
     
# ---------------------- class and func documentations ----------------------

KerasDependencies.__doc__="""\ 
Lazy-loads Keras dependencies from `tensorflow.keras` or `keras`.

Parameters
----------
extra_msg : str, optional
    Additional message to display in the warning if TensorFlow is 
    not found. This message will be appended to the default warning 
    message. Default is `None`.
error : str, optional
    Defines the error handling strategy. If `'warn'` (default), a 
    warning message will be displayed if TensorFlow is not found. 
    If `'raise'`, an `ImportError` will be raised. Default is 
    `'warn'`.

Notes
-----
This class attempts to dynamically import necessary Keras 
dependencies from either `tensorflow.keras` or `keras` on demand.

Mathematically, the import of each function or class can be 
represented as:

.. math::

    f = 
    \begin{cases} 
    f_{\text{tf.keras}} & \text{if TensorFlow is installed} \\
    f_{\text{keras}} & \text{if only Keras is installed} \\
    \end{cases}

where :math:`f` is the function or class to be imported, 
:math:`f_{\text{tf.keras}}` is the function or class from 
`tensorflow.keras`, and :math:`f_{\text{keras}}` is the function or 
class from `keras`.

Examples
--------
>>> from fusionlab.compat.tf import KerasDependencies
>>> deps = KerasDependencies(extra_msg="Custom warning message")
>>> Adam = deps.Adam  # Only imports Adam optimizer when accessed
>>> Sequential = deps.Sequential  # Only imports Sequential model when accessed

See Also
--------
importlib.import_module : Import a module programmatically.
keras.models.Model : The `Model` class from `keras.models`.
keras.models.Sequential : The `Sequential` class from `keras.models`.
tensorflow.keras.models.Model : The `Model` class from `tensorflow.keras.models`.
tensorflow.keras.models.Sequential : The `Sequential` class from 
                                      `tensorflow.keras.models`.

References
----------
.. [1] Chollet, François. *Deep Learning with Python*. Manning, 2017.
.. [2] Abadi, Martín et al. *TensorFlow: Large-Scale Machine Learning 
        # on Heterogeneous Systems*. 2015.    
    
    
"""

import_optional_dependency.__doc__="""\
Checks if a package is installed and optionally imports it.

Parameters
----------
package_name : str
    The name of the package to check and optionally import.
error : str, optional
    Defines the error handling strategy. If `'warn'` (default), a warning 
    message will be displayed if the package is not found. If `'raise'`,
    an ImportError will be raised.
extra : str, optional
    Additional message to display in the warning or error if the package 
    is not found.

Returns
-------
module or None
    The imported module if available, otherwise None if `error='warn'` is used.

Raises
------
ImportError
    If the package is not installed and `error='raise'` is used.

Notes
-----
This function attempts to check for the presence of a specified package and 
optionally import it. 
If the package is not found, the behavior depends on the `error` parameter.

Examples
--------
>>> import_optional_dependency('numpy')
Using numpy
>>> import_optional_dependency('nonexistent_package',
                                extra="Custom warning message", error='warn')
Custom warning message: No module named 'nonexistent_package'
>>> import_optional_dependency('nonexistent_package', error='raise')
Traceback (most recent call last):
  ...
ImportError: nonexistent_package is not installed. Use pip or conda to install it.

See Also
--------
importlib.import_module : Import a module programmatically.

References
----------
.. [1] Python documentation on importlib: https://docs.python.org/3/library/importlib.html
"""


import_keras_function.__doc__="""\
Tries to import a function from `tensorflow.keras`, falling back to 
`keras` if necessary.

Parameters
----------
module_name : str
    The name of the module from which to import the function. 
    This could be a submodule within `tensorflow.keras` or `keras`. 
    For example, `'optimizers'` or `'models'`.
function_name : str
    The name of the function to import from the specified module. 
    For example, `'Adam'` or `'load_model'`.
error : str, optional
    Defines the error handling strategy. If `'warn'` (default), 
    a warning message will be displayed when the function is 
    imported. If `'ignore'`, no warnings will be shown.

Returns
-------
function
    The imported function from either `tensorflow.keras` or `keras`, 
    depending on what is available in the environment.

Raises
------
ImportError
    If neither `tensorflow.keras` nor `keras` is installed, or if 
    the specified function cannot be found in the respective modules.

Notes
-----
This function attempts to dynamically import a specified function 
from either `tensorflow.keras` or `keras`. It first tries to import 
from `tensorflow.keras`, and if that fails (due to `tensorflow` not 
being installed), it falls back to `keras`.

Mathematically, the import can be considered as a selection 
operation:

.. math::

    f = 
    \begin{cases} 
    f_{\text{tf.keras}} & \text{if TensorFlow is installed} \\
    f_{\text{keras}} & \text{if only Keras is installed} \\
    \end{cases}

where :math:`f` is the function to be imported, :math:`f_{\text{tf.keras}}` 
is the function from `tensorflow.keras`, and :math:`f_{\text{keras}}` 
is the function from `keras`.

Examples
--------
>>> from fusionlab.compat.nn import import_keras_function
>>> Adam = import_keras_function('optimizers', 'Adam')
>>> load_model = import_keras_function('models', 'load_model')

See Also
--------
keras.optimizers.Adam
keras.models.load_model
tensorflow.keras.optimizers.Adam
tensorflow.keras.models.load_model

References
----------
.. [1] Chollet, François. *Deep Learning with Python*. Manning, 2017.
.. [2] Abadi, Martn et al. *TensorFlow: Large-Scale Machine Learning 
        on Heterogeneous Systems*. 2015.
"""


import_keras_dependencies.__doc__="""\
Create a `KerasDependencies` instance for lazy-loading Keras dependencies.

Parameters
----------
extra_msg : str, optional
    Additional message to display in the warning if TensorFlow is 
    not found. This message will be appended to the default warning 
    message. Default is `None`.
error : str, optional
    Defines the error handling strategy. If `'warn'` (default), a 
    warning message will be displayed if TensorFlow is not found. 
    If `'raise'`, an `ImportError` will be raised. Default is 
    `'warn'`.

Returns
-------
KerasDependencies
    An instance of the `KerasDependencies` class for lazy-loading 
    Keras dependencies.

Notes
-----
This function returns an instance of the `KerasDependencies` class, 
which attempts to dynamically import necessary Keras dependencies 
from either `tensorflow.keras` or `keras` on demand.

Mathematically, the import of each function or class can be 
represented as:

.. math::

    f = 
    \begin{cases} 
    f_{\text{tf.keras}} & \text{if TensorFlow is installed} \\
    f_{\text{keras}} & \text{if only Keras is installed} \\
    \end{cases}

where :math:`f` is the function or class to be imported, 
:math:`f_{\text{tf.keras}}` is the function or class from 
`tensorflow.keras`, and :math:`f_{\text{keras}}` is the function or 
class from `keras`.

Examples
--------
>>> from fusionlab.compat.tf import import_keras_dependencies
>>> deps = import_keras_dependencies(extra_msg="Custom warning message")
>>> Adam = deps.Adam  # Only imports Adam optimizer when accessed
>>> Sequential = deps.Sequential  # Only imports Sequential model when accessed

See Also
--------
importlib.import_module : Import a module programmatically.
keras.models.Model : The `Model` class from `keras.models`.
keras.models.Sequential : The `Sequential` class from `keras.models`.
tensorflow.keras.models.Model : The `Model` class from 
                                `tensorflow.keras.models`.
tensorflow.keras.models.Sequential : The `Sequential` class from 
                                      `tensorflow.keras.models`.

References
----------
.. [1] Chollet, François. *Deep Learning with Python*. Manning, 2017.
.. [2] Abadi, Martín et al. *TensorFlow: Large-Scale Machine Learning 
        on Heterogeneous Systems*. 2015.
"""


check_keras_backend.__doc__="""\
Check if `tensorflow` or `keras` is installed.

Parameters
----------
error : str, optional
    Defines the error handling strategy. If `'warn'` (default), a 
    warning message will be displayed if neither `tensorflow` nor 
    `keras` is installed. If `'raise'`, an `ImportError` will be 
    raised. If `'ignore'`, no action will be taken. Default is `'warn'`.

Returns
-------
str or None
    Returns 'tensorflow' if TensorFlow is installed, 'keras' if Keras 
    is installed, or None if neither is installed.

Examples
--------
>>> backend = check_keras_backend()
>>> if backend:
>>>     print(f"{backend} is installed.")
>>> else:
>>>     print("Neither tensorflow nor keras is installed.")

Notes
-----
This function first checks for TensorFlow installation and then for 
Keras. The behavior on missing packages depends on the `error` 
parameter.
"""

