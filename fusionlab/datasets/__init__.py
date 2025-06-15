# -*- coding: utf-8 -*-
# License: BSD-3-Clause 
# Author: LKouadio <etanoyau@gmail.com>

"""
Datasets submodule for fusionlab, including data generation tools
and loading APIs for included sample datasets.
"""
# Import generation functions from make.py
from .make import (
    make_multi_feature_time_series,
    make_quantile_prediction_data,
    make_anomaly_data,
    make_trend_seasonal_data,
    make_multivariate_target_data, 
    )

from .load import (
    fetch_zhongshan_data,
    fetch_nansha_data,
    load_processed_subsidence_data, 
    load_subsidence_pinn_data
    )

__all__ = [

    'make_multi_feature_time_series',
    'make_quantile_prediction_data',
    'make_anomaly_data',
    'make_trend_seasonal_data',
    'make_multivariate_target_data',
    
    'fetch_zhongshan_data',
    'fetch_nansha_data',
    'load_processed_subsidence_data', 
    'load_subsidence_pinn_data'
    ]

