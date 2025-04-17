# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Implements the Temporal Fusion Transformer (TFT) and the extreme
TFT (XTFT) architectures for multi-horizon time-series forecasting.
These models integrate static and dynamic covariates and can
optionally use future covariates for predictions several steps
ahead. They employ attention mechanisms and can produce
probabilistic forecasts via quantiles.

These architectures allow flexible configuration, including
residual connections, batch normalization, and scaling
strategies. Users can customize the number of attention heads,
LSTM units, and activation functions.

Below are concise examples demonstrating how to define and use
XTFT and TFT models, including how to prepare input data,
compile with an optimizer and loss function, fit, and then
perform predictions. Users can copy these examples outside of
this script for testing.

Example with XTFT
-----------------
Assume:
- ``static_input_dim = 10`` (static features dimension)
- ``dynamic_input_dim = 45`` (dynamic features dimension)
- ``future_covariate_dim = 5`` (future covariates dimension)
- ``forecast_horizons = 20`` (steps ahead to forecast)
- ``output_dim = 1`` (univariate forecast)
- ``quantiles = [0.1, 0.5, 0.9]`` for probabilistic forecasts

For illustration, we create synthetic data:

>>> import numpy as np
>>> from gofast.nn.transformers import XTFT
>>> batch_size = 32
>>> forecast_horizons = 20
>>> static_input_dim = 10
>>> dynamic_input_dim = 45
>>> future_covariate_dim = 5
>>> output_dim = 1
>>> quantiles = [0.1, 0.5, 0.9]

>>> static_in = np.random.randn(batch_size, static_input_dim).astype(np.float32)
>>> dynamic_in = np.random.randn(batch_size, forecast_horizons,
...                              dynamic_input_dim).astype(np.float32)
>>> fut_cov_in = np.random.randn(batch_size, forecast_horizons,
...                              future_covariate_dim).astype(np.float32)
>>> y_true = np.random.randn(batch_size, forecast_horizons,
...                          output_dim).astype(np.float32)

>>> xtft_model = XTFT(
...     static_input_dim=static_input_dim,
...     dynamic_input_dim=dynamic_input_dim,
...     future_input_dim=future_covariate_dim,
...     embed_dim=32,
...     forecast_horizons=forecast_horizons,
...     quantiles=quantiles,
...     max_window_size=10,
...     memory_size=100,
...     num_heads=4,
...     dropout_rate=0.1,
...     output_dim=output_dim,
...     attention_units=32,
...     hidden_units=64,
...     lstm_units=64,
...     scales=None,
...     multi_scale_agg=None,
...     activation='relu',
...     use_residuals=True,
...     use_batch_norm=False,
...     final_agg='last'
... )

>>> xtft_model.compile(optimizer='adam', loss='mse')
>>> history = xtft_model.fit(
...     [static_in, dynamic_in, fut_cov_in],
...     y_true,
...     epochs=2,
...     verbose=2
... )
>>> y_pred = xtft_model.predict([static_in, dynamic_in, fut_cov_in])
>>> print(y_pred.shape)
# shape: (batch_size, forecast_horizons, num_quantiles, output_dim)

Example with TemporalFusionTransformer
--------------------------------------
Assume:
- ``static_input_dim = 10``
- ``dynamic_input_dim = 45``
- ``forecast_horizon = 20``
- ``output_dim = 1``
- ``quantiles = [0.1, 0.5, 0.9]``

>>> from gofast.nn.transformers import TemporalFusionTransformer
>>> batch_size = 32
>>> static_input_dim = 10
>>> dynamic_input_dim = 45
>>> forecast_horizon = 20
>>> output_dim = 1
>>> quantiles = [0.1, 0.5, 0.9]

>>> static_in = np.random.randn(batch_size, static_input_dim).astype(np.float32)
>>> dynamic_in = np.random.randn(batch_size, forecast_horizon,
...                              dynamic_input_dim).astype(np.float32)
>>> y_true = np.random.randn(batch_size, forecast_horizon,
...                          output_dim).astype(np.float32)

>>> tft_model = TemporalFusionTransformer(
...     static_input_dim=static_input_dim,
...     dynamic_input_dim=dynamic_input_dim,
...     hidden_units=64,
...     num_heads=4,
...     dropout_rate=0.1,
...     forecast_horizons=forecast_horizon,
...     quantiles=quantiles,
...     activation='elu',
...     use_batch_norm=False,
...     num_lstm_layers=1,
...     lstm_units=64
... )
>>> tft_model.compile(optimizer='adam', loss='mse')
>>> history = tft_model.fit(
...     [static_in, dynamic_in],
...     y_true,
...     epochs=2,
...     verbose=2
... )
>>> y_pred = tft_model.predict([static_in, dynamic_in])
>>> print(y_pred.shape)
# shape: (batch_size, forecast_horizon, num_quantiles)

Below code provides the classes from `_tft.py`, `_adj_tft.py`,
and `_xtft.py`. They define the above models, letting you explore
multi-horizon forecasting with attention-based architectures.
"""

import warnings
from gofast.compat.tf import HAS_TF

if not HAS_TF:
    warnings.warn(
        "TensorFlow is not installed. 'TemporalFusionTransformer',"
        " 'NTemporalFusionTransformer','XTFT', 'SuperXTFT',"
        " 'TFT' require tensorflow to be available."
    )

# If TF is available, import the actual classes
if HAS_TF:
    from ._tft import TemporalFusionTransformer, NTemporalFusionTransformer
    from ._adj_tft import TFT
    from ._xtft import XTFT, SuperXTFT

    __all__ = [
        "TemporalFusionTransformer",
        "NTemporalFusionTransformer",
        "XTFT",
        "SuperXTFT",
        "TFT"
    ]
else:
    # Provide stubs that do nothing if user tries to import them 
    # but we have already warned that TF is not installed.
    __all__ = [
        "TemporalFusionTransformer",
        "NTemporalFusionTransformer",
        "XTFT",
        "SuperXTFT",
        "TFT"
    ]

