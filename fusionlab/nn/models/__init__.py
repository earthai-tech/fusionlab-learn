# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import warnings
from .. import KERAS_BACKEND 

if KERAS_BACKEND:
    from .._base_attentive import BaseAttentive # noqa
    from ._halnet import HALNet
    from ._xtft import XTFT, SuperXTFT  # noqa: F401
    from .utils import plot_history_in
    
else:
    warnings.warn(
        "TensorFlow not installed. XTFT and SuperXTFT "
        "require TF and will not be available.",
        ImportWarning
    )
    
__all__=[
    "BaseAttentive", 
    "HALNet", 
    "XTFT",
    "SuperXTFT",
    "HALNet",
    "BaseAttentive", 
    "plot_history_in"
]