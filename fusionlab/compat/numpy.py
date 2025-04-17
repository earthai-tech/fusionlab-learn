# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Ensuring compatibility with different versions of NumPy.
Includes fallback functions for missing attributes or functions
in older versions of NumPy.
"""

from packaging.version import Version, parse
import warnings
import numpy as np
from .._gofastlog import gofastlog

# Setup logging
_logger = gofastlog().get_gofast_logger(__name__)

__all__ = [
    "safe_erf",
    "ensure_numpy_compatibility",
    "gelu_numpy",
    "NP_LT_1_19",
    "NP_LT_1_18",
    "NP_LT_1_17",
    "NP_LT_1_16",
    "NP_LT_1_15",
]

# Version checks
numpy_version = parse(np.__version__)
NP_LT_1_19 = numpy_version < Version("1.19.0")
NP_LT_1_18 = numpy_version < Version("1.18.0")
NP_LT_1_17 = numpy_version < Version("1.17.0")
NP_LT_1_16 = numpy_version < Version("1.16.0")
NP_LT_1_15 = numpy_version < Version("1.15.0")

def safe_erf(x):
    """Return the error function, ensuring compatibility with 
    older numpy versions."""
    try:
        return np.erf(x)
    except AttributeError:
        # Fallback for older versions of NumPy that don't have np.erf
        from scipy.special import erf
        return erf(x)

def ensure_numpy_compatibility():
    """Ensures that the current NumPy version is compatible 
    with the required functions."""
    if NP_LT_1_15:
        _logger.warning(
            "You are using an old version of NumPy (v<1.15)."
            " Some functions may not work as expected.")
        warnings.warn(
            "You are using an old version of NumPy. Please upgrade"
            " to a newer version for full functionality.")
    
    if NP_LT_1_18:
        _logger.warning(
            "NumPy version < 1.18 detected. Consider upgrading"
            " for better performance and features.")
    
    if NP_LT_1_19:
        _logger.warning(
            "NumPy version < 1.19 detected. Some advanced"
            " features may not be available.")


def gelu_numpy(x):
    return 0.5 * x * (1 + safe_erf(x / np.sqrt(2)))
