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

if HAS_KT: 
    from ._tft_tuner import ( 
        XTFTTuner, 
        TFTTuner, 
        xtft_tuner, 
        tft_tuner 
    )
    from ._pihal_tuner import PiHALTuner 
    from ._hal_tuner import HALTuner 
    from ._hydro_tuner import HydroTuner 
    
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

