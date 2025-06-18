# fusionlab/_deps.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides centralized, robust utilities for checking and importing optional
heavy dependencies like TensorFlow and Keras Tuner.

This module acts as the single source of truth for dependency management,
allowing other parts of the library to gracefully handle cases where a
backend might not be installed.
"""
from __future__ import annotations
import importlib.util
import warnings
from typing import Dict, Optional, Literal 

# --- Standardized Error and Warning Messages ---

_DEPENDENCY_MESSAGES = {
    "tensorflow": (
        "This feature requires a full TensorFlow installation,"
        " but it was not found.\n\n"
        "Please install TensorFlow by running:\n"
        "    pip install tensorflow\n\n"
        "For more details, see: https://www.tensorflow.org/install"
    ),
    "keras_tuner": (
        "Hyperparameter tuning features require the `keras-tuner`"
        " package, but it was not found.\n\n"
        "Please install it by running:\n"
        "    pip install keras-tuner"
    ),
}

_KERAS_FALLBACK_WARNING = (
    "Standalone Keras is installed but TensorFlow is not. While some "
    "functionalities may work, the primary backend for `fusionlab-learn` "
    "is `tensorflow.keras`. For full functionality and future "
    "compatibility, it is highly recommended to install TensorFlow."
)

def check_backends(names: str) -> Dict[str, bool]:
    """
    Checks for the presence of one or more specified packages.

    This utility can check for a single package or multiple packages by
    providing their names separated by '__'.

    Parameters
    ----------
    names : str
        The name of the package(s) to check. For multiple packages,
        separate the names with a double underscore, e.g.,
        ``'tensorflow__keras_tuner'``.

    Returns
    -------
    dict
        A dictionary where keys are the package names and values are
        booleans indicating whether the package is installed.

    Examples
    --------
    >>> from fusionlab._deps import check_backends
    >>> check_backends('tensorflow')
    {'tensorflow': True}
    >>> check_backends('non_existent_package')
    {'non_existent_package': False}
    >>> check_backends('numpy__pandas')
    {'numpy': True, 'pandas': True}
    """
    package_list = [pkg.strip() for pkg in names.split('__')]
    status = {
        pkg: importlib.util.find_spec(pkg) is not None for pkg in package_list
    }
    return status

def import_dependencies(
    name: str = 'tensorflow',
    extra_msg: Optional[str] = None,
    error: str = 'warn',
    allow_keras_fallback: bool = True,
) -> Literal['KerasDependencies', 'DummyKerasDeps',
             'KerasTunerDependencies','DummyKT']:
    """
    Dynamically loads backend dependencies or returns a dummy object.

    This factory function is the central point for accessing backend
    functionality. It checks if the required package is installed and returns
    either a real dependency loader class or a safe dummy placeholder.

    Parameters
    ----------
    name : {'tensorflow', 'keras_tuner'}, default='tensorflow'
        The name of the core dependency to load.
    extra_msg : str, optional
        An additional message to append to any warnings or errors.
    error : {'raise', 'warn', 'ignore'}, default='warn'
        The policy for handling a missing dependency.
    allow_keras_fallback : bool, default=True
        If `name='tensorflow'` and it is not found, this flag allows
        the function to check for a standalone `keras` installation
        as a fallback. A warning will be issued if this occurs.

    Returns
    -------
    Union[KerasDependencies, DummyKerasDeps, KerasTunerDependencies, DummyKT]
        An instance of the appropriate real dependency loader or its
        dummy counterpart.
    """
    if name == 'tensorflow':
        tf_found = check_backends('tensorflow')['tensorflow']
        keras_found = check_backends('keras')['keras']

        if tf_found:
            from .compat.tf import KerasDependencies
            return KerasDependencies(extra_msg=extra_msg, error=error)
        
        elif allow_keras_fallback and keras_found:
            warnings.warn(_KERAS_FALLBACK_WARNING, UserWarning, stacklevel=2)
            from .compat.tf import KerasDependencies
            return KerasDependencies(extra_msg=extra_msg, error=error)
        
        else: # Neither TensorFlow nor Keras is found
            from ._dummies import DummyKerasDeps
            if extra_msg:
                warnings.warn(str(extra_msg), ImportWarning, stacklevel=2)
            return DummyKerasDeps()

    elif name == 'keras_tuner':
        if check_backends('keras_tuner')['keras_tuner']:
            from .compat.kt import KerasTunerDependencies
            return KerasTunerDependencies(extra_msg=extra_msg, error=error)
        else:
            from ._dummies import DummyKT
            if extra_msg:
                 warnings.warn(str(extra_msg), ImportWarning, stacklevel=2)
            return DummyKT()
            
    else:
        raise ValueError(
            f"Unsupported dependency name: '{name}'. "
            "Choose from 'tensorflow' or 'keras_tuner'."
        )
