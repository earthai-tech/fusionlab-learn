# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Initializes the `nn` subpackage, dynamically selecting the backend.

This module checks for the presence of TensorFlow/Keras and configures
a central `KERAS_DEPS` object. If the backend is available, `KERAS_DEPS`
becomes a lazy loader for real Keras/TensorFlow components. If not, it
becomes a dummy object generator that raises helpful `ImportError`
messages at runtime.

This allows other modules in the `nn` subpackage to be imported without
crashing, even if heavy dependencies are not installed.
"""

import os 
# filter out TF INFO and WARNING messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or "3"
# Disable oneDNN custom operations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from ._config import ( 
    import_keras_dependencies, 
    check_keras_backend , 
    configure_dependencies, 
    Config as config
)

# Set default configuration
config.INSTALL_DEPS = False
config.WARN_STATUS = 'warn'

# Custom message for missing dependencies
EXTRA_MSG = ( 
    "`nn` sub-package expects the `tensorflow` or"
    " `keras` library to be installed."
    )
# Configure and install dependencies if needed
configure_dependencies(install_dependencies=config.INSTALL_DEPS)

# Lazy-load Keras dependencies
KERAS_DEPS = import_keras_dependencies(extra_msg=EXTRA_MSG, error='ignore')

# Check if TensorFlow or Keras is installed
KERAS_BACKEND = check_keras_backend(error='ignore')

def dependency_message(module_name):
    """
    Generate a custom message for missing dependencies.

    Parameters
    ----------
    module_name : str
        The name of the module that requires the dependencies.

    Returns
    -------
    str
        A message indicating the required dependencies.
    """
    return (
        f"`{module_name}` needs either the `tensorflow` or `keras` package to be "
        "installed. Please install one of these packages to use this function."
    )


    
