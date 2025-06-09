# -*- coding: utf-8 -*-
# File: fusionlab/nn/models/halnet.py
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Implements the Hybrid Attentive LSTM Network (HAL-Net), a state-of-the-art
architecture for multi-horizon time-series forecasting.
"""
from __future__ import annotations

from textwrap import dedent 
from numbers import Real, Integral  
from typing import List, Optional, Union, Dict, Any  
import numpy as np 

from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.docs import _shared_docs, doc 
from ...api.property import NNLearner 
from ...compat.sklearn import validate_params, Interval, StrOptions 
from ...core.handlers import param_deprecated_message 
from ...utils.deps_utils import ensure_pkg
# from ..decorators import Appender

from .. import KERAS_DEPS, KERAS_BACKEND, dependency_message
 
if KERAS_BACKEND:
    LSTM = KERAS_DEPS.LSTM
    Dense = KERAS_DEPS.Dense
    Flatten = KERAS_DEPS.Flatten
    Dropout = KERAS_DEPS.Dropout 
    Layer = KERAS_DEPS.Layer 
    LayerNormalization = KERAS_DEPS.LayerNormalization 
    MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
    Model= KERAS_DEPS.Model 
    Input=KERAS_DEPS.Input
    Concatenate=KERAS_DEPS.Concatenate 
    Tensor=KERAS_DEPS.Tensor
    Add = KERAS_DEPS.Add 
    Constant =KERAS_DEPS.Constant
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    
    tf_reduce_sum = KERAS_DEPS.reduce_sum
    tf_stack = KERAS_DEPS.stack
    tf_expand_dims = KERAS_DEPS.expand_dims
    tf_tile = KERAS_DEPS.tile
    tf_range_=KERAS_DEPS.range 
    tf_concat = KERAS_DEPS.concat
    tf_shape = KERAS_DEPS.shape
    tf_reshape=KERAS_DEPS.reshape
    tf_add = KERAS_DEPS.add
    tf_maximum = KERAS_DEPS.maximum
    tf_reduce_mean = KERAS_DEPS.reduce_mean
    tf_add_n = KERAS_DEPS.add_n
    tf_float32=KERAS_DEPS.float32
    tf_constant=KERAS_DEPS.constant 
    tf_square=KERAS_DEPS.square 
    tf_GradientTape=KERAS_DEPS.GradientTape
    tf_unstack =KERAS_DEPS.unstack
    tf_errors=KERAS_DEPS.errors 
    tf_is_nan =KERAS_DEPS.is_nan 
    tf_reduce_all=KERAS_DEPS.reduce_all
    tf_zeros_like=KERAS_DEPS.zeros_like
    tf_squeeze = KERAS_DEPS.squeeze
    tf_zeros =KERAS_DEPS.zeros 
    
    tf_autograph=KERAS_DEPS.autograph
    tf_autograph.set_verbosity(0)
    
    from ...compat.tf import optional_tf_function 
    from .._tensor_validation import validate_anomaly_scores 
    from .._tensor_validation import validate_model_inputs
    from .._tensor_validation import validate_anomaly_config 
    from .._tensor_validation import align_temporal_dimensions
    
    from ..losses import ( 
        combined_quantile_loss, 
        combined_total_loss, 
        prediction_based_loss
    )
    from ..utils import set_default_params
    from ..components import (
            Activation, 
            AdaptiveQuantileLoss,
            AnomalyLoss,
            CrossAttention,
            DynamicTimeWindow,
            GatedResidualNetwork,
            HierarchicalAttention,
            LearnedNormalization,
            MemoryAugmentedAttention,
            MultiDecoder,
            MultiModalEmbedding,
            MultiObjectiveLoss,
            MultiResolutionAttentionFusion,
            MultiScaleLSTM,
            QuantileDistributionModeling,
            VariableSelectionNetwork,
            PositionalEncoding, 
            aggregate_multiscale, 
            aggregate_multiscale_on_3d, 
            aggregate_time_window_output
        )
    

    # from ..components import aggregate_multiscale_on_3d#, aggregate_time_window_output
    # from .._tensor_validation import validate_model_inputs
    # from tensorflow.keras.initializers import Constant
    # import tensorflow as tf

    # --- Type Hinting ---
    Tensor = KERAS_DEPS.Tensor
else:
    # Define fallback types for type hinting if Keras is not available
    Tensor = Any
    Model = object
    Layer = object

DEP_MSG = dependency_message('nn.models')
logger = fusionlog().get_fusionlab_logger(__name__)
DEP_MSG = dependency_message('nn.transformers') 

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

__all__=['HALNet']

@register_keras_serializable('fusionlab.nn.transformers', name="HALNet")
# @doc (
#     key_improvements= dedent(_shared_docs['xtft_key_improvements']), 
#     key_functions= dedent(_shared_docs['xtft_key_functions']), 
#     methods= dedent( _shared_docs['xtft_methods']
#     )
#  )
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'multi_scale_agg',
            'condition': lambda v: v == "concat",
            'message': (
                "The 'concat' mode for multi-scale aggregation requires identical "
                "time dimensions across scales, which is rarely practical. "
                "This mode will fall back to the robust last-timestep approach "
                "in real applications. For true multi-scale handling, use 'last' "
                "mode instead (automatically set).\n"
                "Why change?\n"
                "- 'concat' mixes features across scales at the same timestep\n"
                "- Requires manual time alignment between scales\n" 
                "- 'last' preserves scale independence & handles variable lengths"
            ),
            'default': "last"
        }
    ],
    warning_category=UserWarning
)

class HALNet(Model, NNLearner):
    """Hybrid Attentive LSTM Network (HAL-Net).

    A powerful, data-driven model for multi-horizon time series
    forecasting, based on the architecture of PIHALNet but without the
    physics-informed or anomaly detection components.

    It leverages an encoder-decoder framework with multi-scale LSTMs
    and a suite of advanced attention mechanisms to capture complex
    temporal patterns from static, dynamic past, and known future inputs.
    """
    @validate_params({
        "static_input_dim": [Interval(Integral, 0, None, closed='left')],
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "future_input_dim": [Interval(Integral, 0, None, closed='left')],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "embed_dim": [Interval(Integral, 1, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')],
        "attention_units": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
    })
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_dim: int = 1,
        forecast_horizon: int = 1,
        quantiles: Optional[List[float]] = None,
        embed_dim: int = 32,
        hidden_units: int = 64,
        lstm_units: int = 64,
        attention_units: int = 32,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        max_window_size: int = 10,
        memory_size: int = 100,
        scales: Optional[List[int]] = None,
        multi_scale_agg: str = 'last',
        final_agg: str = 'last',
        activation: str = 'relu',
        use_residuals: bool = True,
        use_batch_norm: bool = False,
        use_vsn: bool = True,
        vsn_units: Optional[int] = None,
        name: str = "HALNet",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_window_size = max_window_size
        self.memory_size = memory_size
        self.final_agg = final_agg
        self.activation_fn_str = Activation(activation).activation_str
        self.use_residuals = use_residuals
        self.use_batch_norm = use_batch_norm
        self.use_vsn = use_vsn
        self.vsn_units = vsn_units if vsn_units is not None else self.hidden_units

        (self.quantiles, self.scales,
         self.lstm_return_sequences) = set_default_params(
            quantiles, scales, multi_scale_agg
        )
        self.multi_scale_agg_mode = multi_scale_agg
        
        self._build_halnet_layers()

    def _build_halnet_layers(self):
        """Instantiates all layers for the HALNet architecture."""
        # This is where all Keras layers are created to avoid issues
        # with tf.function recompilation.
        if self.use_vsn:
            if self.static_input_dim > 0:
                self.static_vsn = VariableSelectionNetwork(
                    num_inputs=self.static_input_dim,
                    units=self.vsn_units,
                    dropout_rate=self.dropout_rate, name="static_vsn")
                self.static_vsn_grn = GatedResidualNetwork(
                    units=self.hidden_units,
                    dropout_rate=self.dropout_rate, name="static_vsn_grn")
            else:
                self.static_vsn, self.static_vsn_grn = None, None

            if self.dynamic_input_dim > 0:
                self.dynamic_vsn = VariableSelectionNetwork(
                    num_inputs=self.dynamic_input_dim,
                    units=self.vsn_units, use_time_distributed=True,
                    dropout_rate=self.dropout_rate, name="dynamic_vsn")
                self.dynamic_vsn_grn = GatedResidualNetwork(
                    units=self.embed_dim,
                    dropout_rate=self.dropout_rate, name="dynamic_vsn_grn")
            else:
                self.dynamic_vsn, self.dynamic_vsn_grn = None, None

            if self.future_input_dim > 0:
                self.future_vsn = VariableSelectionNetwork(
                    num_inputs=self.future_input_dim,
                    units=self.vsn_units, use_time_distributed=True,
                    dropout_rate=self.dropout_rate, name="future_vsn")
                self.future_vsn_grn = GatedResidualNetwork(
                    units=self.embed_dim,
                    dropout_rate=self.dropout_rate, name="future_vsn_grn")
            else:
                self.future_vsn, self.future_vsn_grn = None, None
        else:
            # If not using VSN, ensure all related attributes are None.
            self.static_vsn, self.static_vsn_grn = None, None
            self.dynamic_vsn, self.dynamic_vsn_grn = None, None
            self.future_vsn, self.future_vsn_grn = None, None

        # A GRN for processing attention outputs.
        self.attention_processing_grn = GatedResidualNetwork(
            units=self.attention_units,
            dropout_rate=self.dropout_rate,
            activation=self.activation_fn_str,
            name="attention_processing_grn")
            
        # A projection layer for the decoder input.
        self.decoder_input_projection = Dense(
            self.attention_units,
            activation=self.activation_fn_str,
            name="decoder_input_projection")

        if not self.use_vsn:
            # Create dense layers for non-VSN path.
            if self.static_input_dim > 0:
                self.static_dense = Dense(
                    self.hidden_units, activation=self.activation_fn_str)
                self.grn_static_non_vsn = GatedResidualNetwork(
                    units=self.hidden_units,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation_fn_str,
                    name="grn_static_non_vsn")
            else:
                self.static_dense, self.grn_static_non_vsn = None, None
            
            # Create dense layers for dynamic and future features for non-VSN path
            self.dynamic_dense = Dense(self.embed_dim)
            self.future_dense = Dense(self.embed_dim)
        else:
            self.static_dense, self.grn_static_non_vsn = None, None
            self.dynamic_dense, self.future_dense = None, None
        
        # --- Core Architectural Layers (Always Created) ---
        self.positional_encoding = PositionalEncoding()
        self.multi_scale_lstm = MultiScaleLSTM(
            lstm_units=self.lstm_units, scales=self.scales,
            return_sequences=True)
        self.cross_attention = CrossAttention(
            units=self.attention_units, num_heads=self.num_heads)
        self.hierarchical_attention = HierarchicalAttention(
            units=self.attention_units, num_heads=self.num_heads)
        self.memory_augmented_attention = MemoryAugmentedAttention(
            units=self.attention_units,
            memory_size=self.memory_size, num_heads=self.num_heads)
        self.multi_resolution_attention_fusion = \
            MultiResolutionAttentionFusion(
                units=self.attention_units, num_heads=self.num_heads)
        
        # Final output layers
        self.multi_decoder = MultiDecoder(
            output_dim=self.output_dim,
            num_horizons=self.forecast_horizon)
        self.quantile_distribution_modeling = QuantileDistributionModeling(
            quantiles=self.quantiles, output_dim=self.output_dim)

        # --- Layers for Residual Connections ---
        if self.use_residuals:
            self.decoder_add_norm = [Add(), LayerNormalization()]
            self.final_add_norm = [Add(), LayerNormalization()]
            self.residual_dense = Dense(self.attention_units)
        else:
            self.decoder_add_norm, self.final_add_norm, self.residual_dense = \
                None, None, None

    def run_halnet_core(self, static_input, dynamic_input, future_input, training):
        """Executes data-driven pipeline using an encoder-decoder."""
        time_steps = tf_shape(dynamic_input)[1]

        # --- 1. Initial Feature Processing ---
        # Process inputs through VSNs or simple Dense layers to create
        # initial feature representations.
        
        # Static context processing
        static_context = None
        if self.use_vsn and self.static_vsn is not None:
            static_out = self.static_vsn(static_input, training=training)
            static_context = self.static_vsn_grn(static_out, training=training)
        elif self.static_dense:
            static_out = self.static_dense(static_input)
            static_context = self.grn_static_non_vsn(static_out, training=training)
        
        logger.debug(f"Static context shape: {getattr(static_context, 'shape', 'None')}")

        # --- 2. Encoder Path (Processes Past Data) ---
        # Slice the historical part of the `future_input`
        future_for_encoder = future_input[:, :time_steps, :]
        
        # Process dynamic and future historical features
        if self.use_vsn:
            dyn_proc = self.dynamic_vsn_grn(self.dynamic_vsn(
                dynamic_input, training=training), training=training)
            fut_enc_proc = self.future_vsn_grn(self.future_vsn(
                future_for_encoder, training=training), training=training)
        else:
            dyn_proc = self.dynamic_dense(dynamic_input)
            fut_enc_proc = self.future_dense(future_for_encoder)
        
        # Combine and encode historical information
        # _, encoder_input_raw = align_temporal_dimensions(
        #     tensor_ref=dyn_proc, 
        #     tensor_to_align= fut_enc_proc, 
        #     mode='pad_to_ref', 
        #     name ="future_enc_emb"
        # )
        encoder_input_raw = tf_concat([dyn_proc, fut_enc_proc], axis=-1)
        encoder_input = self.positional_encoding(encoder_input_raw, training=training)
        lstm_out = self.multi_scale_lstm(encoder_input, training=training)
        encoder_sequences = aggregate_multiscale_on_3d(lstm_out, mode='concat')
        
        logger.debug(f"Encoder output sequence shape: {encoder_sequences.shape}")

        # --- 3. Decoder Path (Prepares Context for Forecasting) ---
        # Slice the forecast-window part of the `future_input`
        future_for_decoder = future_input[:, time_steps:, :]
        
        # Process future decoder features
        if self.use_vsn:
            fut_dec_proc = self.future_vsn_grn(self.future_vsn(
                future_for_decoder, training=training), training=training)
        else:
            fut_dec_proc = self.future_dense(future_for_decoder)
            
        future_with_pos = self.positional_encoding(fut_dec_proc, training=training)
        
        # Combine static context with future features for the decoder
        decoder_parts = [future_with_pos]
        if static_context is not None:
            static_expanded = tf_expand_dims(static_context, 1)
            static_expanded = tf_tile(
                static_expanded, [1, self.forecast_horizon, 1]
            )
            decoder_parts.append(static_expanded)
            
        raw_decoder_input = tf_concat(decoder_parts, axis=-1)
        projected_decoder_input = self.decoder_input_projection(raw_decoder_input)
        
        logger.debug(f"Projected decoder input shape: {projected_decoder_input.shape}")

        # --- 4. Attention-based Fusion ---
        cross_att_out = self.cross_attention(
            [projected_decoder_input, encoder_sequences], training=training)
        
        if self.use_residuals and self.decoder_add_norm:
            att_proc = self.attention_processing_grn(cross_att_out, training=training)
            context_att = self.decoder_add_norm[0]([projected_decoder_input, att_proc])
            context_att = self.decoder_add_norm[1](context_att)
        else:
            context_att = cross_att_out
            
        hier_att_out = self.hierarchical_attention([context_att, context_att], training=training)
        mem_att_out = self.memory_augmented_attention(hier_att_out, training=training)
        
        # --- 5. Final Combination and Aggregation ---
        final_features = self.multi_resolution_attention_fusion(mem_att_out, training=training)

        if self.use_residuals and self.final_add_norm:
            res_base = self.residual_dense(context_att)
            final_features = self.final_add_norm[0]([final_features, res_base])
            final_features = self.final_add_norm[1](final_features)
        
        logger.debug(f"Shape after final fusion: {final_features.shape}")

        return aggregate_time_window_output(final_features, self.final_agg)
    
    def call(self, inputs: List[Optional[Tensor]], training: bool = False) -> Tensor:
        """Forward pass for the HALNet model."""
        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=inputs, static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            mode='strict', 
            model_name='xtft' # Re-use strict validation
        )
        
        final_features = self.run_halnet_core(
            static_p, dynamic_p, future_p, training=training)
        
        decoded_outputs = self.multi_decoder(final_features, training=training)
        
        if self.quantiles is not None:
            return self.quantile_distribution_modeling(decoded_outputs)
        
        return decoded_outputs

    def get_config(self):
        """Returns the configuration of the HALNet model."""
        config = super().get_config()
        config.update({
            'static_input_dim': self.static_input_dim,
            'dynamic_input_dim': self.dynamic_input_dim,
            'future_input_dim': self.future_input_dim,
            'output_dim': self.output_dim,
            'forecast_horizon': self.forecast_horizon,
            'quantiles': self.quantiles,
            'embed_dim': self.embed_dim,
            'hidden_units': self.hidden_units,
            'lstm_units': self.lstm_units,
            'attention_units': self.attention_units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'max_window_size': self.max_window_size,
            'memory_size': self.memory_size,
            'scales': self.scales,
            'multi_scale_agg': self.multi_scale_agg_mode,
            'final_agg': self.final_agg,
            'activation': self.activation_fn_str,
            'use_residuals': self.use_residuals,
            'use_batch_norm': self.use_batch_norm,
            'use_vsn': self.use_vsn,
            'vsn_units': self.vsn_units,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
