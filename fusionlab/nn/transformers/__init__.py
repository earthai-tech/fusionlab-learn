# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Lazy loader for Transformer models in fusionlab.nn.
"""
import warnings
from fusionlab.compat.tf import HAS_TF

# Always provide core transformer
from ._ts_transformers import TimeSeriesTransformer

# Warn if TF missing
if not HAS_TF:
    warnings.warn(
        "TensorFlow not installed. "
        "TemporalFusionTransformer, DummyTFT, "
        "TFT, require tensorflow.",
        ImportWarning
    )
else:
    # Eagerly import core wrappers
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


# def __getattr__(name):
#     """
#     Lazy import and warn relocation only when users
#     access XTFT or SuperXTFT.
#     """
#     if name in ("XTFT", "SuperXTFT"):
#         if not HAS_TF:
#             raise ImportError(
#                 f"Cannot import {name}: TensorFlow is missing."
#             )
#         warnings.warn(
#             "XTFT and SuperXTFT will be moved to "
#             "'fusionlab.nn.models' in future releases.",
#             FutureWarning
#         )
#         # Lazy import
#         from ..models._xtft import XTFT as _XT, \
#             SuperXTFT as _SuperXT
#         return _XT if name == "XTFT" else _SuperXT
#     raise AttributeError(
#         f"module {__name__!r} has no attribute {name!r}"
#     )


# def __dir__():
#     """Include lazy attributes in dir()."""
#     return sorted(__all__ + list(globals().keys()))
