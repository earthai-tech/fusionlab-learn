# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
This internal module provides placeholder "dummy" objects for optional
dependencies like TensorFlow, Keras, and KerasTuner.

It dynamically generates these dummies based on the central configuration
in `fusionlab._configs`, ensuring it is always in sync with the real
dependency loader.
"""
from __future__ import annotations
from typing import Any

try:
    from ._configs import TENSORFLOW_CONFIG, KERAS_TUNER_CONFIG
except ImportError:
    TENSORFLOW_CONFIG = {}
    KERAS_TUNER_CONFIG = {}

class _DummyObject:
    """
    A generic base class for all dummy objects. It raises a helpful
    ImportError when instantiated or called, guiding the user to install
    the missing package.
    """
    _dependency_name = "A required package"
    _is_callable = True

    def __init__(self, *args, **kwargs):
        # This __init__ is for dummy classes (e.g., Model, Dense).
        # It raises an error upon instantiation.
        if self._is_callable:
            raise ImportError(
                f"The `{self.__class__.__name__}` component"
                f" requires {self._dependency_name},"
                f" but it is not installed. Please install"
                " it to use this feature."
            )

    def __call__(self, *args, **kwargs):
        # This __call__ is for dummy functions (e.g., reduce_mean).
        # It raises an error upon being called.
        raise ImportError(
            f"The `{self.__class__.__name__}` function"
            f" requires {self._dependency_name},"
            f" but it is not installed."
            " Please install it to use this feature."
        )

def _create_dummy(
        name: str, dependency: str = "TensorFlow/Keras", 
        is_callable=True
    ):
    """
    Factory function to create specific dummy classes on the 
    fly with tailored error messages.
    """
    return type(
        name,
        (_DummyObject,),
        {
            "__doc__": f"Dummy placeholder for `{name}`. Requires {dependency}.",
            "_dependency_name": dependency,
            "_is_callable": is_callable
        }
    )

# --- Dummy Namespace for KerasTuner ---
class DummyKT:
    """
    A dummy module to stand in for the `keras_tuner` package.
     It dynamically creates dummy objects for its attributes
     based on KERAS_TUNER_CONFIG.
    """
    def __getattr__(self, name: str) -> Any:
        if name in KERAS_TUNER_CONFIG:
            # Most KerasTuner objects are classes.
            return _create_dummy(name, "KerasTuner")()
        # Allow access to generic objects for type checking if needed
        if name == "HyperModel":
            return object
            
        raise AttributeError(
            f"'DummyKT' has no attribute '{name}'. Please add it to the "
            "KERAS_TUNER_CONFIG in `fusionlab/_configs.py`."
        )

# --- Dummy Namespace for TensorFlow/Keras ---
class DummyKerasDeps:
    """
    A lazy-loading, dynamic namespace that mimics the real KERAS_DEPS object.

    It generates dummy objects on-the-fly for any attribute that is
    requested, based on the central TENSORFLOW_CONFIG. This ensures it
    is always in sync with the real dependency loader.
    """
    def __getattr__(self, name: str) -> Any:
        # Check if the requested name exists in the central config file.
        if name in TENSORFLOW_CONFIG:
            config_value = TENSORFLOW_CONFIG[name]

            # For attributes that are just strings (like dtypes), return them directly.
            if isinstance(config_value, str):
                return config_value
            
            # For objects that can be instantiated or called, create a dummy.
            # We determine if it's a class/function primarily by capitalization.
            is_class_or_func = not name.islower()
            return _create_dummy(name, is_callable=is_class_or_func)()

        # Fallback for any unexpected attribute.
        raise AttributeError(
            f"'DummyKerasDeps' has no attribute '{name}'. Please add it to "
            "the TENSORFLOW_CONFIG in `fusionlab/_configs.py`."
        )


