# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
This module defines common type aliases used throughout the FusionLab package.
It includes types for handling pandas DataFrames, Series, numpy NDArray, and 
Array-like objects, which are frequently encountered in data science and machine 
learning tasks.

These types are designed to aid static type checking, ensuring compatibility 
across different functions and classes within the package.
"""

from typing import Union, Any, Callable, List, Optional, Iterable
from typing import Tuple, Set, Pattern, TypeVar, SupportsInt 
from typing import Dict, TYPE_CHECKING, Iterator, Literal
import pandas as pd
import numpy as np

try:
    import jax.numpy as jnp
except ImportError: 
    jnp=None 

# --- PyTorch types ---
try:
    import torch
    from torch import Tensor as TorchTensor
    from torch.utils.data import Dataset as TorchDataset
    from torch.optim import Optimizer as TorchOptimizer
    import torch.nn as torch_nn

    TorchModel      = torch_nn.Module
    TorchSequential = torch_nn.Sequential

except ImportError:
    torch = None

    class TorchTensor:
        def __init__(*args, **kwargs):
            raise ImportError(
                "PyTorch is required for TorchTensor"
                " but is not installed.")

    TorchDataset = None
    TorchOptimizer = None

    class TorchModel:
        def __init__(*args, **kwargs):
            raise ImportError(
                "PyTorch is required for TorchModel"
                " but is not installed.")

    class TorchSequential:
        def __init__(*args, **kwargs):
            raise ImportError(
                "PyTorch is required for TorchSequential"
                " but is not installed.")


# --- TensorFlow types ---
try:
    import tensorflow as tf
    from tensorflow.data import Dataset as TFDataset
    from tensorflow.keras.optimizers import Optimizer as TFOptimizer
    from tensorflow.keras.callbacks import Callback as TFCallback
    from tensorflow.keras import Model as TFModel
    from tensorflow.keras import Sequential as TFSequential

    TFTensor = tf.Tensor

except ImportError:
    tf = None

    class TFTensor:
        def __init__(*args, **kwargs):
            raise ImportError("TensorFlow is required for TFTensor"
                              " but is not installed.")

    TFDataset   = None
    TFOptimizer = None
    TFCallback  = None

    class TFModel:
        def __init__(*args, **kwargs):
            raise ImportError("TensorFlow is required for TFModel"
                              " but is not installed.")

    class TFSequential:
        def __init__(*args, **kwargs):
            raise ImportError(
                "TensorFlow is required for TFSequential but"
                " is not installed.")


# Type aliases for common data structures
DataFrame = pd.DataFrame  
Series = pd.Series        
NDArray = np.ndarray       
_T = TypeVar('_T')
_V = TypeVar('_V')

# Type aliases for callable functions and operations
if TYPE_CHECKING:

    ArrayLike = Union[NDArray, Series, list, tuple]
    _Sub      = Callable[[Any], Any]
    _F        = Callable[[ArrayLike], Any]
else:
    # at runtime we avoid the subscript, so Callable[[…], …] isn't evaluated
    from collections.abc import Callable as _RuntimeCallable

    ArrayLike = (list, tuple, np.ndarray, pd.Series)  # just a marker
    _Sub      = _RuntimeCallable
    _F        = _RuntimeCallable


# --- Multi‑framework type aliases ---
JNPNDArray  = jnp.ndarray if jnp else None 
_Tensor     = Union[TorchTensor, TFTensor, JNPNDArray]
_Dataset    = Union[TorchDataset, TFDataset]
_Optimizer  = Union[TorchOptimizer, TFOptimizer]
_Callback   = Union[TFCallback]
_Model      = Union[TorchModel, TFModel]
_Sequential = Union[TorchSequential, TFSequential]

# Type aliases for additional Python built-in types
Iterator = Iterator[Any] 

# Define MultioutputLiteral for type hinting
#  if not using StrOptions directly in hints
MultioutputLiteral = Literal['raw_values', 'uniform_average']
NanPolicyLiteral = Literal['omit', 'propagate', 'raise']
MetricFunctionType = Callable[..., Union[float, np.ndarray]]
MetricType = Literal['mae', 'accuracy', 'interval_score']
PlotKind = Literal['time_profile', 'summary_bar']
PlotKindWIS = Literal['scores_histogram', 'summary_bar']
PlotKindTheilU = Literal['summary_bar']

def is_dataframe(obj: Any) -> bool:
    """
    Check if an object is a pandas DataFrame.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a pandas DataFrame, False otherwise.
    """
    return isinstance(obj, pd.DataFrame)

def is_series(obj: Any) -> bool:
    """
    Check if an object is a pandas Series.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a pandas Series, False otherwise.
    """
    return isinstance(obj, pd.Series)

def is_ndarray(obj: Any) -> bool:
    """
    Check if an object is a numpy ndarray.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a numpy ndarray, False otherwise.
    """
    return isinstance(obj, np.ndarray)

def is_array_like(obj: Any) -> bool:
    """
    Check if an object is array-like (e.g., numpy ndarray, pandas Series, list, or tuple).

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is array-like, False otherwise.
    """
    return isinstance(obj, (np.ndarray, pd.Series, list, tuple))

# Example callable function types
def apply_function(f: _F, data: ArrayLike) -> Any:
    """
    Apply a callable function (e.g., np.mean, np.sum) to an array-like structure.

    Parameters
    ----------
    f : _F
        The callable function (e.g., np.mean, np.sum).
    
    data : ArrayLike
        The data to which the function is applied.

    Returns
    -------
    Any
        The result of applying the function to the data.
    """
    return f(data)

def transform_data(f: _Sub, data: Any) -> Any:
    """
    Apply a transformation to the data using a callable function.

    Parameters
    ----------
    f : _Sub
        The callable function to transform the data (e.g., a lambda function).

    data : Any
        The data to which the transformation is applied.

    Returns
    -------
    Any
        The transformed data.
    """
    return f(data)

# Define __all__ to specify what gets imported when * is used
__all__ = [
    "DataFrame", "Series", "NDArray", "ArrayLike", 
    "is_dataframe", "is_series", "is_ndarray", "is_array_like", 
    "_Sub", "_F", "apply_function", "transform_data", 
    "_Tensor", "_Dataset", "_Optimizer",
    "_Callback", "_Model", "_Sequential",
    "_T", "_V", 
    
    "Union", "Any", "Callable", "List", "Optional", "Iterable", 
    "Union", "Dict" , "Tuple", "Set", "Pattern", "SupportsInt", 
    "Iterator"
]
