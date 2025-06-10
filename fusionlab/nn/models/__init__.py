# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import warnings
from fusionlab.compat.tf import HAS_TF 

# Core model definitions
from ._halnet import HALNet
from .utils import plot_history_in 
__all__ = [
    "HALNet",
    "plot_history_in", 
]

# XTFT variants relocated to models package
if HAS_TF:
    # Import and expose XTFT and SuperXTFT
    from ._xtft import XTFT, SuperXTFT  # noqa: F401
    __all__.extend([
        "XTFT",
        "SuperXTFT",
    ])
else:
    warnings.warn(
        "TensorFlow not installed. XTFT and SuperXTFT "
        "require TF and will not be available.",
        ImportWarning
    )