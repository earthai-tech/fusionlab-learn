# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import warnings
from . import KERAS_BACKEND 

if KERAS_BACKEND:
    from ._base_attentive import BaseAttentive 
    from .hybrid._halnet import HALNet 
    from .hybrid._xtft import XTFT , SuperXTFT 
    from .transformers._ts_transformers import TimeSeriesTransformer 
    from .transformers._adj_tft import TFT
    from .transformers._tft import TemporalFusionTransformer, DummyTFT 

    from .pinn._pihal import PIHALNet 
    from .pinn._pihalnet import PiHALNet 
    from .pinn._transflow_subnet import TransFlowSubsNet 
    from .pinn._gw_models import PiTGWFlow 

else:
    warnings.warn(
        "TensorFlow not installed. XTFT and SuperXTFT "
        "require TF and will not be available.",
        ImportWarning
    )
    
__all__=[
    "HALNet", 
    "XTFT",
    "SuperXTFT",
    "HALNet",
    "BaseAttentive", 
    "BaseAttentive", 
    "TimeSeriesTransformer",
    "TemporalFusionTransformer",
    "TFT",
    "DummyTFT",
    "PIHALNet",
    "PiHALNet", 
    "TransFlowSubsNet",
    "PiTGWFlow" 
    
]