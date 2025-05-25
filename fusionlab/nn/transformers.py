# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import warnings
from fusionlab.compat.tf import HAS_TF

if not HAS_TF:
    warnings.warn(
        "TensorFlow is not installed. 'TemporalFusionTransformer',"
        " 'DummyTFT','XTFT', 'SuperXTFT',"
        " 'TFT' require tensorflow to be available."
    )

# If TF is available, import the actual classes
if HAS_TF:
    from ._tft import TemporalFusionTransformer, DummyTFT
    from ._adj_tft import TFT
    from ._xtft import XTFT, SuperXTFT

    __all__ = [
        "TemporalFusionTransformer",
        "DummyTFT",
        "XTFT",
        "SuperXTFT",
        "TFT"
    ]
else:
    # Provide stubs that do nothing if user tries to import them 
    # but we have already warned that TF is not installed.
    __all__ = [
        "TemporalFusionTransformer",
        "DummyTFT",
        "XTFT",
        "SuperXTFT",
        "TFT"
    ]

