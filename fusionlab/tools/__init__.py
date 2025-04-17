# -*- coding: utf-8 -*-
"""
gofast.tools
==============

**Description:**
This initialization script for the `gofast.tools` package ensures that 
essential dependencies, such as TensorFlow, are available when specific 
applications or scripts within the package are invoked. If the required 
dependencies are missing, the script raises informative errors
to guide the user in resolving the issue.

**Functionality:**
- **Dependency Checks:** Automatically checks for the presence of TensorFlow
   when certain modules or scripts are accessed.
- **Informative Errors:** Provides clear and actionable error messages 
  if dependencies are not installed.
- **Flexibility:** Easily extendable to include additional dependencies 
  and modules as needed.

**Modules Monitored:**
- `xtft_point_p.py`
- `tft_batch_p.py`
- `xtft_proba_p.py`
- `xtft_proba.py`

**Usage:**
When a user attempts to import any of the monitored modules without having 
TensorFlow installed, an `ImportError` with a detailed message will be raised.

**Example:**
```python
from gofast.tools import xtft_proba_p
# If TensorFlow is not installed, an ImportError will be raised with an
 informative message.
```

**Author:** Daniel
**Date:** 2024-12-17
"""

import sys # noqa
import importlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# List of modules that require TensorFlow
_TENSORFLOW_REQUIRED_MODULES = {
    'xtft_point',
    # 'tft_batch_p',
    'xtft_proba_p',
    'xtft_proba',
}

def _check_tensorflow():
    """
    Checks if TensorFlow is installed. If not, raises an ImportError
    with an informative message.
    """
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        error_message = (
            "TensorFlow is required to use this module but is not installed.\n"
            "Please install TensorFlow by running:\n"
            "    pip install tensorflow\n"
            "or refer to the TensorFlow installation guide: "
            "https://www.tensorflow.org/install"
        )
        logger.error(error_message)
        raise ImportError(error_message) from e

def __getattr__(name):
    """
    Custom attribute access for the gofast.tools package. Checks 
    for required dependencies
    when specific modules are accessed.

    :param name: Name of the attribute/module being accessed.
    :return: The imported module if it exists and dependencies are met.
    :raises ImportError: If dependencies are missing.
    """
    if name in _TENSORFLOW_REQUIRED_MODULES:
        _check_tensorflow()
        try:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module  # Cache the imported module
            return module
        except ImportError as e:
            error_message = (
                f"Failed to import module '{name}'. Ensure it exists"
                " within the 'gofast.tools' package."
            )
            logger.error(error_message)
            raise ImportError(error_message) from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    """
    Custom directory listing for the gofast.tools package.
    Includes monitored modules.
    """
    standard_attrs = list(globals().keys())
    return standard_attrs + list(_TENSORFLOW_REQUIRED_MODULES)

