# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
fusionlab.tools
===============

This subpackage provides command-line tools and high-level scripts for
executing common forecasting and analysis workflows.

The tools in this package require a full TensorFlow installation. This
__init__ script checks for this dependency and configures the necessary
backend objects.
"""
# --- Centralized Dependency Setup ---
# Import the dependency management utilities from the central module.
from .._deps import import_dependencies, check_backends

# 1. Check if the required 'tensorflow' backend is available.
#    The `tools` subpackage considers TensorFlow a hard requirement.
KERAS_BACKEND = check_backends('tensorflow')['tensorflow']

# 2. Get the dependency loader object.
#    This will be the real KerasDependencies if TF is installed. If not,
#    it will be a dummy object that raises an informative error upon use.
#    We disable the standalone Keras fallback to enforce the TensorFlow
#    requirement for this part of the library.
KERAS_DEPS = import_dependencies(
    name='tensorflow',
    error='ignore', # raise will fail loudly if TensorFlow is missing.
    # we kept 'warn' for let the package loaded
    allow_keras_fallback=False
)

# --- Public API for the 'tools' Subpackage ---
# Define the objects that will be available when a user does
# `from fusionlab.tools import ...`

__all__ = [
    "KERAS_DEPS",
    "KERAS_BACKEND",
]

# XXX TODO: Design GUI instead. 
# Only expose the tool modules if the backend is available. This prevents
# ImportErrors if a user imports the `tools` package in an environment
# without TensorFlow.
# if KERAS_BACKEND:
#     from . import xtft_point # noqa
#     from . import xtft_proba_p # noqa
#     from . import xtft_proba # noqa

#     # Add the successfully imported modules to the public API.
#     __all__.extend([
#         "xtft_point",
#         "xtft_proba_p",
#         "xtft_proba",
#     ])
