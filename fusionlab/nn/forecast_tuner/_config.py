# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides configuration and dependency checking for the forecast_tuner
subpackage.
"""
from __future__ import annotations
import importlib.util
import warnings

def check_keras_tuner_is_available(error: str = 'warn') -> bool:
    """
    Checks if the 'keras-tuner' package is installed.

    This helper function is used to determine if the optional tuning
    dependencies are present in the user's environment.

    Parameters
    ----------
    error : {'raise', 'warn', 'ignore'}, default='warn'
        Policy for handling the case where 'keras-tuner' is not found.
        - 'raise': Raises an ImportError.
        - 'warn': Issues an ImportWarning.
        - 'ignore': Does nothing.

    Returns
    -------
    bool
        True if 'keras-tuner' is installed, False otherwise.
    """
    if importlib.util.find_spec("keras_tuner"):
        return True
    
    message = (
        "Hyperparameter tuning features require `keras-tuner` to be "
        "installed. Please run `pip install keras-tuner`."
    )
    if error == 'raise':
        raise ImportError(message)
    elif error == 'warn':
        warnings.warn(message, ImportWarning, stacklevel=2)
        
    return False

