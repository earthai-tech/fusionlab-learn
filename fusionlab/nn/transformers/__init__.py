# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Lazy loader for Transformer models in fusionlab.nn.
"""
import warnings
from fusionlab.compat.tf import HAS_TF

# Warn if TF missing
if not HAS_TF:
    warnings.warn(
        "TensorFlow not installed. "
        "TemporalFusionTransformer, DummyTFT, "
        "TFT, require tensorflow.",
        ImportWarning
    )
else:
    # Always provide core transformer
    from ._ts_transformers import TimeSeriesTransformer
    from ._tft import (
        TemporalFusionTransformer, DummyTFT
        )
    from ._adj_tft import TFT

__all__ = [
    "TimeSeriesTransformer",
    "TemporalFusionTransformer",
    "DummyTFT",
    "TFT",
]


