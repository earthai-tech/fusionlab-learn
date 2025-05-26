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
import sys 
import logging
import warnings
import importlib
import types

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
    __version__ = "0.2.3"

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
    ("pyyaml", None)
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

try:
    _kd = importlib.import_module("kdiagram")
    # Alias it so users can do:
    #    import fusionlab.kdiagram as fkd
    #    from fusionlab.kdiagram.plot import plot_model_drift
    sys.modules[__name__ + ".kdiagram"] = _kd
    # Also expose it in our namespace:
    kdiagram = _kd
    __all__.append("kdiagram")
except ImportError:
    # dont need to warn just ignore ..
    pass 

# If kdiagram import failed, we set up a dummy module entry so
# that `import fusionlab.kdiagram` doesn’t immediately crash,
# but accessing it does:
if __name__ + ".kdiagram" not in sys.modules:
    # Create an empty placeholder module for fusionlab.kdiagram
    _dummy_kd = types.ModuleType(__name__ + ".kdiagram")
    sys.modules[__name__ + ".kdiagram"] = _dummy_kd

def __getattr__(name: str):
    """
    Called when someone does `fusionlab.<name>` and <name> isn't found
    in the normal attributes. We intercept 'kdiagram' to provide
    a friendly hint.
    """

    if name == "kdiagram":
        hint = (
            "The submodule 'fusionlab.kdiagram' is unavailable because "
            "the optional dependency `k-diagram` is not installed.\n\n"
            "To install it alongside fusionlab-learn run:\n\n"
            "    pip install fusionlab-learn[kdiagram]\n\n"
            "Or directly:\n"
            "    pip install k-diagram\n\n"
            "After that, you can:\n\n"
            "    import fusionlab.kdiagram as fkd\n"
            "    fkd.plot.plot_model_drift(...)\n"
        )
        warnings.warn(hint, ImportWarning, stacklevel=2)

        raise AttributeError(
            "fusionlab.kdiagram is not available. " 
            "See warning above for install instructions."
        )
    # Fallback for any other missing attribute
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Append version to module docstring
__doc__ += f"\nVersion: {__version__}\n"

