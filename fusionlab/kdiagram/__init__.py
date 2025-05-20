# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Proxy module for optional k-diagram integration.

This module attempts to import the `k-diagram` package. If `k-diagram`
is installed, its public API (or selected submodules/functions) will
be available under `fusionlab.kdiagram`. If `kdiagram` is not
installed, importing from `fusionlab.kdiagram` will result in an
ImportError with instructions on how to install it.
"""
import importlib
import warnings # noqa 

_KDIAGRAM_INSTALLED = False
_KDIAGRAM_ERROR_MSG = (
    "The 'kdiagram' package is not installed, but is required for "
    "this functionality within `fusionlab.kdiagram`. Please install it "
    "using:\n  pip install k-diagram\nor install fusionlab-learn with "
    "the kdiagram extra:\n  pip install fusionlab-learn[kdiagram]"
)

try:
    # Attempt to import the actual kdiagram package
    kdiagram_actual = importlib.import_module('kdiagram')
    _KDIAGRAM_INSTALLED = True

    # Expose specific submodules or functions you want to make
    # directly available via fusionlab.kdiagram, e.g.:
    # from kdiagram import plot as kd_plot # Example if kdiagram has plot
    # from kdiagram.utils import some_util as kd_some_util

    # Or, more generally, you can try to expose its __all__ if defined
    # This is more dynamic but less explicit.
    if hasattr(kdiagram_actual, '__all__'):
        for _name in kdiagram_actual.__all__:
            globals()[_name] = getattr(kdiagram_actual, _name)
    else:
        # If no __all__, expose common submodules if they exist
        _common_submodules = ['plot', 'utils', 'datasets'] 
        for _submodule_name in _common_submodules:
            if hasattr(kdiagram_actual, _submodule_name):
                globals()[_submodule_name] = getattr(kdiagram_actual, _submodule_name)
                
    # Make the top-level kdiagram module itself available
    # so `fusionlab.kdiagram.some_function` can work if `some_function`
    # is at the top level of the kdiagram package.
    # This also allows `import fusionlab.kdiagram as flkd` and then `flkd.kdiagram_actual.something`.
    # For a cleaner API, explicitly re-exporting is better.

except ImportError:
    # kdiagram is not installed.
    # We can define placeholders or a custom __getattr__ to raise
    # a helpful error when fusionlab.kdiagram.something is accessed.
    pass # _KDIAGRAM_INSTALLED remains False

def __getattr__(name):
    """
    Called when an attribute is not found in this module.
    Used to raise an informative error if kdiagram is not installed.
    """
    if not _KDIAGRAM_INSTALLED:
        # Check if the user is trying to access something that would
        # exist if kdiagram were installed.
        # This is a bit heuristic.
        potential_kdiagram_members = ['plot', 'utils', 'datasets'] 
        if name in potential_kdiagram_members or name.startswith("plot_"):
            raise ImportError(_KDIAGRAM_ERROR_MSG)
    # If kdiagram is installed but the attribute is still not found,
    # or if the name is not a likely kdiagram member.
    raise AttributeError(
        f"module 'fusionlab.kdiagram' has no attribute '{name}'"
        )

# Optionally, define __all__ for fusionlab.kdiagram
# This would list what you *intend* to expose if kdiagram is installed.
# It's tricky with dynamic imports. For now, __getattr__ handles access.
# __all__ = [] # Populate if you explicitly re-export specific items