
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides a compatibility layer for Keras Tuner, allowing for lazy
loading of its components and graceful handling of its absence.
"""
from __future__ import annotations
import importlib
import warnings
from typing import Optional 

from .._configs import KERAS_TUNER_CONFIG

class KerasTunerDependencies:
    """
    A lazy loader for Keras Tuner components.

    This class provides attribute-based access to Keras Tuner objects
    (e.g., Tuner, HyperParameters). It imports them on-demand, only when
    they are first accessed. This prevents ImportErrors if Keras Tuner
    is not installed but the library is imported.
    """
    def __init__(self, extra_msg: Optional[str]= None, error: str = 'warn'):
        self.extra_msg = extra_msg or (
            "`keras-tuner` is required for this feature but is not installed."
            " Please run `pip install keras-tuner`."
        )
        self.error = error
        self._dependencies = {}
        self._check_keras_tuner()

    def _check_keras_tuner(self):
        """Checks for Keras Tuner and warns if it's missing."""
        try:
            importlib.import_module('keras_tuner')
        except ImportError:
            if self.error == 'warn':
                warnings.warn(self.extra_msg, ImportWarning, stacklevel=2)
            elif self.error == 'raise':
                raise ImportError(self.extra_msg)

    def __getattr__(self, name: str):
        """Lazily import and return a Keras Tuner attribute."""
        if name not in self._dependencies:
            self._dependencies[name] = self._import_dependency(name)
        return self._dependencies[name]

    # def _import_dependency(self, name: str):
    #     """
    #     Imports a specific Keras Tuner object based on the central
    #     KERAS_TUNER_CONFIG.
    #     """
    #     if name in KERAS_TUNER_CONFIG:
    #         module_name, function_name = KERAS_TUNER_CONFIG[name]
    #         try:
    #             module = importlib.import_module(module_name)
    #             return getattr(module, function_name)
    #         except ImportError as e:
    #             if self.error == 'raise':
    #                 raise ImportError(
    #                     f"Failed to import `{name}` from `{module_name}`. "
    #                     f"{self.extra_msg}"
    #                 ) from e
    #             # In 'warn' or 'ignore' mode, we return None if it fails
    #             return None
        
    #     raise AttributeError(
    #         f"'KerasTunerDependencies' object has no attribute '{name}'. "
    #         "Ensure it is defined in `fusionlab/_configs.py`."
    #     )
    def _import_dependency(self, name: str):
        """
        Imports a specific Keras Tuner object based on the central
        KERAS_TUNER_CONFIG. Supports either a single (module, attr)
        tuple or an iterable of such tuples for fallbacks.
        """
        if name not in KERAS_TUNER_CONFIG:
            raise AttributeError(
                f"'KerasTunerDependencies' object has no attribute '{name}'. "
                "Ensure it is defined in `fusionlab/_configs.py`."
            )
        entry = KERAS_TUNER_CONFIG[name]
        # Normalize to a list of (module_name, function_name) pairs
        if isinstance(entry[0], (tuple, list)):
            candidates = entry        # already a sequence of pairs
        else:
            candidates = [entry]      # single pair
    
        last_err = None
        for module_name, function_name in candidates:
            try:
                module = importlib.import_module(module_name)
                return getattr(module, function_name)
            except (ImportError, AttributeError) as e:
                # record the last exception in case we need to raise later
                last_err = e
                continue
    
        # If we get here, all import attempts failed
        if self.error == 'raise':
            raise ImportError(
                f"Failed to import `{name}`; attempted modules: "
                f"{', '.join(m for m, _ in candidates)}. {self.extra_msg}"
            ) from last_err
    
        # Warn or ignore mode: return None
        if self.error == 'warn':
            warnings.warn(
                f"Could not import `{name}`; tried: "
                f"{', '.join(m for m, _ in candidates)}. "
                f"{self.extra_msg}",
                ImportWarning,
                stacklevel=2
            )



def import_kt_dependencies(extra_msg=None, error='warn'):
    """
    Factory function to create an instance of KerasTunerDependencies.
    """
    return KerasTunerDependencies(extra_msg, error)