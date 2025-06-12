# -*- coding: utf-8 -*-
# File: fusionlab/nn/models/_base_attentive.py
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Base class for advanced, attentive sequence-to-sequence models
like HALNet and PIHALNet.
"""

from __future__ import annotations
from typing import List, Optional, Union, Dict, Any

from ..._fusionlog import fusionlog
from ...api.property import NNLearner
from ...compat.sklearn import validate_params, Interval, StrOptions
from ...utils.generic_utils import select_mode
from ...utils.deps_utils import ensure_pkg

from .. import KERAS_BACKEND, KERAS_DEPS, dependency_message

if KERAS_BACKEND:
    from ..components import (
        Activation, Add, Concatenate, CrossAttention, Dense, Dropout,
        DynamicTimeWindow, GatedResidualNetwork, HierarchicalAttention,
        Layer, LayerNormalization, LSTM, MemoryAugmentedAttention, Model,
        MultiDecoder, MultiHeadAttention, MultiResolutionAttentionFusion,
        MultiScaleLSTM, PositionalEncoding, QuantileDistributionModeling, 
        VariableSelectionNetwork, 
    )
    from ..utils import set_default_params
    from ..components import aggregate_multiscale_on_3d, aggregate_time_window_output
    from .._tensor_validation import validate_model_inputs
    import tensorflow as tf
    Tensor = KERAS_DEPS.Tensor
    tf_shape = tf.shape
    tf_concat = tf.concat
    tf_zeros = tf.zeros
    tf_expand_dims = tf.expand_dims
    tf_tile = tf.tile
    tf_convert_to_tensor = tf.convert_to_tensor
    tf_assert_equal = tf.debugging.assert_equal
else:
    class Model: pass
    class Layer: pass
    Tensor = Any

logger = fusionlog().get_fusionlab_logger(__name__)

@KERAS_DEPS.register_keras_serializable(
    'fusionlab.nn.models', name="BaseAttentive"
)
class BaseAttentive(Model, NNLearner):
    """
    A base class for hybrid and transformer-based attentive
    forecasting models.
    """
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_dim: int = 1,
        forecast_horizon: int = 1,
        architecture: str = 'hybrid',
        mode: Optional[str] = None,
        num_encoder_layers: int = 2,
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
        use_vsn: bool = True,
        vsn_units: Optional[int] = None,
        name: str = "BaseAttentiveModel",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        # Store all configuration parameters
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.architecture = select_mode(
            architecture, default='hybrid',
            canonical=['hybrid', 'transformer'])
        self.mode = select_mode(mode, default='tft_like')
        self.num_encoder_layers = num_encoder_layers
        self.quantiles = quantiles
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
        self.use_vsn = use_vsn
        self.vsn_units = (vsn_units if vsn_units is not None
                          else self.hidden_units)

        (self.quantiles, self.scales,
         self.lstm_return_sequences) = set_default_params(
            quantiles, scales, multi_scale_agg)
        self.lstm_return_sequences = True
        self.multi_scale_agg_mode = multi_scale_agg
        
        self._build_attentive_layers()

    def _build_attentive_layers(self):
        """Instantiates all shared layers for the attentive architecture."""
        # VSN Layers
        if self.use_vsn:
            if self.static_input_dim > 0:
                self.static_vsn = VariableSelectionNetwork(
                    self.static_input_dim, self.vsn_units,
                    self.dropout_rate, name="static_vsn")
                self.static_vsn_grn = GatedResidualNetwork(
                    self.hidden_units, self.dropout_rate,
                    name="static_vsn_grn")
            else: self.static_vsn, self.static_vsn_grn = None, None
            if self.dynamic_input_dim > 0:
                self.dynamic_vsn = VariableSelectionNetwork(
                    self.dynamic_input_dim, self.vsn_units,
                    self.dropout_rate,
                    use_time_distributed=True, name="dynamic_vsn")
                self.dynamic_vsn_grn = GatedResidualNetwork(
                    self.embed_dim, self.dropout_rate, name="dynamic_vsn_grn")
            else: self.dynamic_vsn, self.dynamic_vsn_grn = None, None
            if self.future_input_dim > 0:
                self.future_vsn = VariableSelectionNetwork(
                    self.future_input_dim, self.vsn_units, self.dropout_rate,
                    use_time_distributed=True, name="future_vsn")
                self.future_vsn_grn = GatedResidualNetwork(
                    self.embed_dim, self.dropout_rate, name="future_vsn_grn")
            else: self.future_vsn, self.future_vsn_grn = None, None
        else:
            self.static_vsn, self.dynamic_vsn, self.future_vsn = None, None, None
            self.static_vsn_grn, self.dynamic_vsn_grn, self.future_vsn_grn = None, None, None

        # Shared & Non-VSN Path Layers
        self.attention_processing_grn = GatedResidualNetwork(
            self.attention_units, self.dropout_rate, self.activation_fn_str,
            name="attention_processing_grn")
        self.decoder_input_projection = Dense(
            self.attention_units, self.activation_fn_str, name="decoder_input_projection")

        if not self.use_vsn:
            if self.static_input_dim > 0:
                self.static_dense = Dense(self.hidden_units, self.activation_fn_str)
                self.grn_static_non_vsn = GatedResidualNetwork(
                    self.hidden_units, self.dropout_rate, self.activation_fn_str,
                    name="grn_static_non_vsn")
            else: self.static_dense, self.grn_static_non_vsn = None, None
            self.dynamic_dense = Dense(self.embed_dim)
            self.future_dense = Dense(self.embed_dim)
        else:
            self.static_dense, self.grn_static_non_vsn, self.dynamic_dense, self.future_dense = \
                None, None, None, None

        # Encoder-specific Layers
        if self.architecture == 'hybrid':
            self.multi_scale_lstm = MultiScaleLSTM(
                self.lstm_units, self.scales, return_sequences=True)
            self.encoder_self_attention = None
        elif self.architecture == 'transformer':
            self.encoder_self_attention = [
                (MultiHeadAttention(num_heads=self.num_heads, key_dim=self.attention_units),
                 LayerNormalization()) for _ in range(
                     self.num_encoder_layers)
            ]
            self.multi_scale_lstm = None

        # Core Architectural Layers
        self.positional_encoding = PositionalEncoding()
        self.cross_attention = CrossAttention(
            self.attention_units, self.num_heads)
        self.hierarchical_attention = HierarchicalAttention(self.attention_units, self.num_heads)
        self.memory_augmented_attention = MemoryAugmentedAttention(
            self.attention_units, self.memory_size, self.num_heads)
        self.multi_resolution_attention_fusion = MultiResolutionAttentionFusion(
            self.attention_units, self.num_heads)
        self.multi_decoder = MultiDecoder(self.output_dim, self.forecast_horizon)
        self.quantile_distribution_modeling = QuantileDistributionModeling(
            self.quantiles, self.output_dim)

        if self.use_residuals:
            self.decoder_add_norm = [Add(), LayerNormalization()]
            self.final_add_norm = [Add(), LayerNormalization()]
            self.residual_dense = Dense(self.attention_units)
        else:
            self.decoder_add_norm, self.final_add_norm, self.residual_dense = \
                None, None, None

    def run_encoder_decoder_core(
            self, static_input, dynamic_input, 
            future_input, training):
        """Executes the data-driven pipeline with selectable encoder architecture."""
        time_steps = tf_shape(dynamic_input)[1]
        
        # 1. Initial Feature Processing
        static_context = self.static_vsn_grn(
            self.static_vsn(static_input, training), training) \
            if self.use_vsn and self.static_vsn else (
                self.grn_static_non_vsn(
                    self.static_dense(static_input), training) 
                if self.static_dense else None)
        
        dyn_proc = self.dynamic_vsn_grn(
            self.dynamic_vsn(dynamic_input, training), training) \
            if self.use_vsn else self.dynamic_dense(dynamic_input)
        
        fut_proc = self.future_vsn_grn(
            self.future_vsn(future_input, training), training) \
            if self.use_vsn else self.future_dense(future_input)

        # 2. Encoder Path
        encoder_input_parts = [dyn_proc]
        if self.mode == 'tft_like':
            fut_enc_proc = fut_proc[:, :time_steps, :]
            encoder_input_parts.append(fut_enc_proc)
        
        encoder_raw = tf_concat(encoder_input_parts, axis=-1)
        encoder_input = self.positional_encoding(encoder_raw, training=training)

        if self.architecture == 'hybrid':
            lstm_out = self.multi_scale_lstm(
                encoder_input, training=training)
            encoder_sequences = aggregate_multiscale_on_3d(lstm_out, mode='concat')
        else: # transformer
            encoder_sequences = encoder_input
            for mha, norm in self.encoder_self_attention:
                attn_out = mha(encoder_sequences, encoder_sequences)
                encoder_sequences = norm(encoder_sequences + attn_out)

        # 3. Decoder Path
        fut_dec_proc = fut_proc[:, time_steps:, :] if self.mode == 'tft_like' else fut_proc
        
        decoder_parts = []
        if static_context is not None:
            static_expanded = tf_expand_dims(static_context, 1)
            static_expanded = tf_tile(static_expanded, [1, self.forecast_horizon, 1])
            decoder_parts.append(static_expanded)
        
        future_with_pos = self.positional_encoding(fut_dec_proc, training)
        decoder_parts.append(future_with_pos)

        raw_decoder_input = tf_concat(decoder_parts, axis=-1)
        projected_decoder_input = self.decoder_input_projection(raw_decoder_input)

        # 4. Attention Fusion & Final Processing
        cross_att_out = self.cross_attention(
            [projected_decoder_input, encoder_sequences], 
            training=training)
        att_proc = self.attention_processing_grn(cross_att_out, training)
        
        context_att = self.decoder_add_norm[0]([projected_decoder_input, att_proc]) \
            if self.use_residuals else att_proc
        if self.use_residuals: context_att = self.decoder_add_norm[1](context_att)

        final_features = self.multi_resolution_attention_fusion(context_att, training)

        if self.use_residuals:
            res_base = self.residual_dense(context_att)
            final_features = self.final_add_norm[0]([final_features, res_base])
            final_features = self.final_add_norm[1](final_features)
        
        return aggregate_time_window_output(final_features, self.final_agg)

    def call(self, inputs, training=False):
        """Forward pass for the attentive model."""
        expected_span = (
            self.max_window_size + self.forecast_horizon 
            if self.mode == 'tft_like' 
           else self.forecast_horizon
          )
        
        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=inputs, static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            mode='strict', model_name='xtft'
            )
        
        tf_assert_equal(
            tf_shape(future_p)[1], expected_span, 
            message=f"Incorrect future_input span for mode='{self.mode}'"
            )
        
        final_features = self.run_encoder_decoder_core(
            static_p, dynamic_p, future_p, training)
        decoded_outputs = self.multi_decoder(final_features, training)
        
        return self.quantile_distribution_modeling(
            decoded_outputs) if self.quantiles else decoded_outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "static_input_dim": self.static_input_dim,
            "dynamic_input_dim": self.dynamic_input_dim,
            "future_input_dim": self.future_input_dim,
            "output_dim": self.output_dim,
            "forecast_horizon": self.forecast_horizon,
            "architecture": self.architecture,
            "mode": self.mode,
            "num_encoder_layers": self.num_encoder_layers,
            "quantiles": self.quantiles,
            "embed_dim": self.embed_dim,
            "hidden_units": self.hidden_units,
            "lstm_units": self.lstm_units,
            "attention_units": self.attention_units,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "max_window_size": self.max_window_size,
            "memory_size": self.memory_size,
            "scales": self.scales,
            "multi_scale_agg": self.multi_scale_agg_mode,
            "final_agg": self.final_agg,
            "activation": self.activation_fn_str,
            "use_residuals": self.use_residuals,
            "use_vsn": self.use_vsn,
            "vsn_units": self.vsn_units,
        })
        return config

