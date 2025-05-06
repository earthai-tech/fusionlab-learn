# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Fusionlab-learn 🔥🧪: Igniting Next‑Gen Temporal Fusion Architectures
========================================================================

A modular library for building, experimenting with, and fusing state‑of‑the‑art
Temporal Fusion Transformer (TFT) variants. FusionLab streamlines every step of
your time‑series modeling workflow, enhancing productivity, flexibility, and
community‑driven innovation.
"""

import os
import logging
import warnings
import importlib

# Configure basic logging and suppress certain third‑party library warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger("matplotlib.font_manager").disabled = True

# Lazy‑import helper to defer heavy imports until needed
def _lazy_import(module_name, alias=None):
    """Lazily import a module to reduce initial package load time."""
    def _lazy_loader():
        return importlib.import_module(module_name)
    if alias:
        globals()[alias] = _lazy_loader
    else:
        globals()[module_name] = _lazy_loader

# Package version
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.1"

# Core dependencies
_required_dependencies = [
    ("numpy", None),
    ("pandas", None),
    ("scipy", None),
    ("matplotlib", None),
    ("tqdm", None),
    ("scikit-learn", "sklearn"),
    ("joblib", None), 
    #("jax", None),
    ("tensorflow", "tensorflow"),
    ("joblib", None), 
    ("statsmodels", None), 
    # ("torch", "torch"),
]

_missing = []
for pkg, import_name in _required_dependencies:
    try:
        if import_name:
            _lazy_import(import_name, pkg)
        else:
            _lazy_import(pkg)
    except ImportError as e:
        _missing.append(f"{pkg}: {e}")

if _missing:
    warnings.warn(
        "Some FusionLab dependencies are missing; functionality may be limited:\n"
        + "\n".join(_missing),
        ImportWarning
    )

# Warning controls
_WARNING_CATEGORIES = {
    "FutureWarning": FutureWarning,
    "SyntaxWarning": SyntaxWarning,
}
_WARN_ACTIONS = {
    "FutureWarning": "ignore",
    "SyntaxWarning": "ignore",
}

def suppress_warnings(suppress: bool = True):
    """
    Globally suppress or re-enable FutureWarning and SyntaxWarning.

    Parameters
    ----------
    suppress : bool, default=True
        If True, filters warnings according to `_WARN_ACTIONS`; if False,
        restores default warning behavior.
    """
    for name, cat in _WARNING_CATEGORIES.items():
        action = _WARN_ACTIONS.get(name, "default") if suppress else "default"
        warnings.filterwarnings(action, category=cat)

# Suppress by default on import
suppress_warnings()

# Disable OneDNN logs in TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Initialize structured logging for FusionLab
# fusionlab/__init__.py

from ._util import initialize_logging

# Suppress and safely initialize structured logging
try:
    initialize_logging()
except Exception:
    pass

__all__ = ["__version__"]

# Append version to module docstring
__doc__ += f"\nVersion: {__version__}\n"
