# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Centralized Keras/TensorFlow symbols used by fusionlab.nn.components.

Import from here instead of repeating `KERAS_DEPS.*` everywhere.

Exports:
- Layers: Dense, Dropout, MultiHeadAttention, ...
- TF ops: tf_shape, tf_pad, tf_debugging, tf_bool, ...
- Utils: activations, register_keras_serializable, DEP_MSG, _logger

This file assumes `KERAS_BACKEND` was resolved upstream.
"""

from ..._fusionlog import fusionlog 
from .. import KERAS_DEPS, KERAS_BACKEND, dependency_message
from ...compat.tf import standalone_keras

if KERAS_BACKEND:
    try:
        # Equivalent to: from tensorflow.keras import activations
        activations = KERAS_DEPS.activations  
    except (ImportError, AttributeError) as e: 
        try: 
            activations = standalone_keras('activations')
        except: 
            raise ImportError (str(e))
    except: 
        raise ImportError(
                "Module 'activations' could not be"
                " imported from either tensorflow.keras"
                " or standalone keras. Ensure that TensorFlow "
                "or standalone Keras is installed and the"
                " module exists."
        )

LSTM = KERAS_DEPS.LSTM
LayerNormalization = KERAS_DEPS.LayerNormalization 
TimeDistributed = KERAS_DEPS.TimeDistributed
MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
Model = KERAS_DEPS.Model 
BatchNormalization = KERAS_DEPS.BatchNormalization
Input = KERAS_DEPS.Input
Softmax = KERAS_DEPS.Softmax
Flatten = KERAS_DEPS.Flatten
Dropout = KERAS_DEPS.Dropout 
Dense = KERAS_DEPS.Dense
Embedding =KERAS_DEPS.Embedding 
Concatenate=KERAS_DEPS.Concatenate 
Layer = KERAS_DEPS.Layer 
Loss=KERAS_DEPS.Loss
Tensor=KERAS_DEPS.Tensor
Sequential =KERAS_DEPS.Sequential
TensorShape =KERAS_DEPS.TensorShape 
Reduction =KERAS_DEPS.Reduction 

register_keras_serializable=KERAS_DEPS.register_keras_serializable
get_loss = KERAS_DEPS.get 

tf_Assert= KERAS_DEPS.Assert
tf_TensorShape= KERAS_DEPS.TensorShape
tf_concat = KERAS_DEPS.concat
tf_shape = KERAS_DEPS.shape
tf_reshape=KERAS_DEPS.reshape
tf_repeat =KERAS_DEPS.repeat
tf_add = KERAS_DEPS.add
tf_cast=KERAS_DEPS.cast
tf_maximum = KERAS_DEPS.maximum
tf_reduce_mean = KERAS_DEPS.reduce_mean
tf_add_n = KERAS_DEPS.add_n
tf_float32=KERAS_DEPS.float32
tf_constant=KERAS_DEPS.constant 
tf_square=KERAS_DEPS.square 
tf_transpose=KERAS_DEPS.transpose 
tf_logical_and=KERAS_DEPS.logical_and 
tf_logical_not = KERAS_DEPS.logical_not 
tf_logical_or = KERAS_DEPS.logical_or
tf_get_static_value =KERAS_DEPS.get_static_value
tf_reduce_sum = KERAS_DEPS.reduce_sum
tf_stack = KERAS_DEPS.stack
tf_unstack =KERAS_DEPS.stack
tf_expand_dims = KERAS_DEPS.expand_dims
tf_tile = KERAS_DEPS.tile
tf_range=KERAS_DEPS.range 
tf_rank=KERAS_DEPS.rank
tf_split = KERAS_DEPS.split
tf_multiply=KERAS_DEPS.multiply
tf_cond=KERAS_DEPS.cond
tf_constant =KERAS_DEPS.constant 
tf_equal =KERAS_DEPS.equal 
tf_int32=KERAS_DEPS.int32 
tf_debugging =KERAS_DEPS.debugging 
tf_autograph=KERAS_DEPS.autograph
tf_pad =KERAS_DEPS.pad 
tf_maximum =KERAS_DEPS.maximum 
tf_ones_like = KERAS_DEPS.ones_like 
tf_bool =KERAS_DEPS.bool 
tf_newaxis = KERAS_DEPS.newaxis 
tf_abs =KERAS_DEPS.abs 
tf_pow = KERAS_DEPS.pow
tf_sin = KERAS_DEPS.sin
tf_cos = KERAS_DEPS.cos
tf_exp = KERAS_DEPS.exp 
tf_log = KERAS_DEPS.log
tf_sigmoid =KERAS_DEPS.sigmoid 
tf_cumsum =KERAS_DEPS.cumsum 
tf_gather =KERAS_DEPS.gather 
tf_random = KERAS_DEPS.random 
tf_softplus = KERAS_DEPS.softplus
tf_reduce_logsumexp = KERAS_DEPS.reduce_logsumexp
tf_sqrt = KERAS_DEPS.sqrt 
tf_erf =KERAS_DEPS.erf 
tf_ones = KERAS_DEPS.ones 
tf_linalg = KERAS_DEPS.linalg
tf_floordiv = KERAS_DEPS.floordiv
tf_greater =KERAS_DEPS.greater 
tf_float32 = KERAS_DEPS.float32
tf_reduce_max = KERAS_DEPS.reduce_max 


_logger = fusionlog().get_fusionlab_logger(__name__)

DEP_MSG = dependency_message('components._config') 

__all__ = [
    "activations",
    "LSTM",
    "LayerNormalization",
    "TimeDistributed",
    "MultiHeadAttention",
    "Model",
    "BatchNormalization",
    "Input",
    "Softmax",
    "Flatten",
    "Dropout",
    "Dense",
    "Embedding",
    "Concatenate",
    "Layer",
    "Loss",
    "Tensor",
    "Sequential",
    "TensorShape",
    "register_keras_serializable",
    "get_loss", 
    "tf_Assert",
    "tf_TensorShape",
    "tf_concat",
    "tf_shape",
    "tf_reshape",
    "tf_repeat",
    "tf_add",
    "tf_cast",
    "tf_maximum",
    "tf_reduce_mean",
    "tf_add_n",
    "tf_float32",
    "tf_constant",
    "tf_square",
    "tf_transpose",
    "tf_logical_and",
    "tf_logical_not",
    "tf_logical_or",
    "tf_get_static_value",
    "tf_reduce_sum",
    "tf_stack",
    "tf_expand_dims",
    "tf_tile",
    "tf_range",
    "tf_rank",
    "tf_split",
    "tf_multiply",
    "tf_cond",
    "tf_equal",
    "tf_int32",
    "tf_debugging",
    "tf_autograph",
    "tf_pad",
    "tf_ones_like",
    "tf_bool",
    "tf_newaxis",
    "tf_pow",
    "tf_sin",
    "tf_cos",
    "tf_exp",
    "tf_log",
    "tf_ones",
    "tf_linalg",
    "tf_floordiv",
    "tf_greater",
    "tf_erf", 
    "tf_sqrt", 
    "tf_sigmoid", 
    "tf_cumsum", 
    "tf_reduce_logsumexp", 
    "tf_softplus", 
    "tf_ones", 
    "tf_ones_like", 
    "tf_unstack", 
    "_logger",
    "DEP_MSG",
    "KERAS_DEPS",
    "KERAS_BACKEND",
]
