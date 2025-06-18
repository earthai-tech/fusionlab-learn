# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
A subpackage for hyperparameter tuning of fusionlab models using Keras Tuner.
"""

from ._config import check_keras_tuner_is_available

HAS_KT = check_keras_tuner_is_available(error='ignore')

if HAS_KT:
    from ...compat.kt import KerasTunerDependencies
    KT_DEPS = KerasTunerDependencies()
else:
    from ..._dummies import DummyKT
    KT_DEPS = DummyKT()

from .tuners import ( 
    XTFTTuner, 
    TFTTuner, 
    PiHALTuner, # as LegacyPiHALTuner
    HALTuner, 
    HydroTuner, 
    xtft_tuner, 
    tft_tuner 
    
)

__all__= [
    'HAS_KT',
    'KT_DEPS',
    'HydroTuner', 
    'XTFTTuner', 
    'TFTTuner' , 
    'PiHALTuner', 
    'HALTuner', 
    'xtft_tuner', 
    'tft_tuner', 
    ]

