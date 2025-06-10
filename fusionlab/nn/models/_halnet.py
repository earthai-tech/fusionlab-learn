# -*- coding: utf-8 -*-
# File: fusionlab/nn/models/halnet.py
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Implements the Hybrid Attentive LSTM Network (HAL-Net), a state-of-the-art
architecture for multi-horizon time-series forecasting.
"""
from __future__ import annotations

from numbers import Real, Integral  
from typing import List, Optional, Any  

from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.docs import DocstringComponents, _halnet_core_params
from ...api.property import NNLearner 
from ...compat.sklearn import validate_params, Interval, StrOptions 
from ...core.handlers import param_deprecated_message 
from ...utils.deps_utils import ensure_pkg


from .. import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .utils import select_mode 

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
    tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor 
    tf_assert_equal=KERAS_DEPS.assert_equal
    
    tf_autograph=KERAS_DEPS.autograph
    tf_autograph.set_verbosity(0)
    
    from .._tensor_validation import validate_model_inputs

    from ..utils import set_default_params
    from ..components import (
            Activation, 
            CrossAttention,
            DynamicTimeWindow,
            GatedResidualNetwork,
            HierarchicalAttention,
            MemoryAugmentedAttention,
            MultiDecoder,
            MultiResolutionAttentionFusion,
            MultiScaleLSTM,
            QuantileDistributionModeling,
            VariableSelectionNetwork,
            PositionalEncoding, 
            aggregate_multiscale_on_3d, 
            aggregate_time_window_output
        )
    
else:
    # Define fallback types for type hinting if Keras is not available
    Tensor = Any
    Model = object
    Layer = object

DEP_MSG = dependency_message('nn._halnet')

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_halnet_core_params), 
)

__all__=['HALNet']

@register_keras_serializable('fusionlab.nn.models._halnet', name="HALNet")
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
    @validate_params({
        "static_input_dim": [Interval(Integral, 0, None, closed='left')],
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "future_input_dim": [Interval(Integral, 0, None, closed='left')],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "embed_dim": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "attention_units": [
            'array-like', 
            Interval(Integral, 1, None, closed='left')
        ], 
        "hidden_units": [
            'array-like', 
            Interval(Integral, 1, None, closed='left')
          ], 
        "lstm_units": [
            'array-like', 
            Interval(Integral, 1, None, closed='left'), 
            None
        ], 
        "activation": [
            StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", "gelu"}),
            callable 
            ],
        "multi_scale_agg": [
            StrOptions({"last", "average",  "flatten", "auto", "sum", "concat"}),
            None
        ],
        "scales": ['array-like', StrOptions({"auto"}),  None],
        "use_residuals": [bool, Interval(Integral, 0, 1, closed="both")],
        "final_agg": [StrOptions({"last", "average",  "flatten"})],
        "mode": [
            StrOptions({'tft', 'pihal', 'tft_like', 'pihal_like'}), 
            None # if None, behave like tft
            ]
    
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
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
        use_vsn: bool = True,
        vsn_units: Optional[int] = None,
        mode: Optional [str]=None, 
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
        self.use_vsn = use_vsn
        self.vsn_units = vsn_units if vsn_units is not None else self.hidden_units

        (self.quantiles, self.scales,
         self.lstm_return_sequences) = set_default_params(
            quantiles, scales, multi_scale_agg
        )
        self.mode= select_mode(mode, default='tft_like')
        
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
                
            # Create dense layers for dynamic and future features
            # for non-VSN path
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
        
        self.dynamic_time_window = DynamicTimeWindow(
            max_window_size=self.max_window_size
        )
        
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
                
    def run_halnet_core_(
        self,
        static_input:Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        training: bool
    ) -> Tensor:
        """Executes data-driven pipeline using an encoder-decoder.

        This revised method correctly separates the processing of past
        (dynamic) and future inputs to handle cases where `time_steps`
        and `forecast_horizon` are different.

        Args:
            static_input: Processed static features.
            dynamic_input: Processed dynamic historical features.
                           Shape: (batch, time_steps, features).
            future_input: Processed known future features.
                          Shape: (batch, forecast_horizon, features).
            training: Python boolean indicating training mode.

        Returns:
            A feature tensor for the forecast horizon, ready for the
            MultiDecoder. Shape is determined by `self.final_agg`.
        """
        # Get the length of the historical time window
        #ctime_steps = tf_shape(dynamic_input)[1]

        # --- 1. Initial Feature Processing ---
        # Process static inputs first.
        static_context = None
        if self.use_vsn and self.static_vsn is not None:
            static_out = self.static_vsn(static_input, training=training)
            static_context = self.static_vsn_grn(
                static_out, training=training
            )
        elif self.static_dense is not None:
            static_out = self.static_dense(static_input)
            static_context = self.grn_static_non_vsn(
                static_out, training=training
            )

        logger.debug(
            f"Static context shape: {getattr(static_context, 'shape', 'None')}"
        )

        # --- 2. Encoder Path (Processes Past Data) ---
        # The encoder uses `dynamic_input` and the historical part of
        # `future_input`.
        
        if self.mode =='tft_like': 
            # In models like TFT, "future" inputs are known for both the past
            # and future. We slice the part corresponding to the encoder's timeline.
            # This assumes `future_input` has length `time_steps + forecast_horizon`.
            # If it only has length `forecast_horizon`, this logic needs adjustment.
            # Let's assume for now `future_input` is just for the decoder.
            # The encoder will only use `dynamic_input`.
            future_for_encoder = None
            time_steps = tf_shape(dynamic_input)[1]
            # For TFT-style, slice the historical part of future inputs
            future_for_encoder = future_input[:, :time_steps, :]
        
            dyn_proc = self.dynamic_dense(
                dynamic_input) if not self.use_vsn else self.dynamic_vsn_grn(
                    self.dynamic_vsn(dynamic_input))
            
            encoder_input_parts = [dyn_proc]
            if future_for_encoder is not None:
                fut_enc_proc = self.future_dense(
                    future_for_encoder) if not self.use_vsn else self.future_vsn_grn(
                        self.future_vsn(future_for_encoder))
                        
                encoder_input_parts.append(fut_enc_proc)
                
            encoder_inputs = tf_concat(encoder_input_parts, axis=-1)
        
            fut_proc = None
            if self.future_input_dim > 0:
                # For TFT-style, slice the forecast part of future inputs
                fut_proc = future_input[:, time_steps:, :]
            else:
                # For standard encoder-decoder, 
                # the entire future_input is for the decoder
                fut_proc = future_input
        else: 
            
            dyn_proc = dynamic_input
            if self.use_vsn and self.dynamic_vsn is not None:
                dyn_proc = self.dynamic_vsn_grn(self.dynamic_vsn(
                    dynamic_input, training=training), training=training)
          
            fut_proc = future_input
            # Process future features for the decoder.
            if self.use_vsn and self.future_vsn is not None:
                fut_proc = self.future_vsn_grn(self.future_vsn(
                    future_input, training=training), training=training)
                
            encoder_inputs = dyn_proc # Self encoder-decoder architecture. 
           
        logger.debug(f"Shape after VSN/initial processing: "
                     f"Dynamic={getattr(dyn_proc, 'shape', 'N/A')}, "
                     f"Future={getattr(fut_proc, 'shape', 'N/A')}")
        
        encoder_input = self.positional_encoding(
            encoder_inputs, training=training
            )
        
        lstm_out = self.multi_scale_lstm(
            encoder_input, training=training
        )
        # `aggregate_multiscale` with 'concat' now correctly returns a 3D tensor
        # by padding and concatenating features.
        encoder_sequences = aggregate_multiscale_on_3d(
            lstm_out, mode='concat'
        )
        
        if self.dynamic_time_window is not None:
            encoder_sequences = self.dynamic_time_window(
                encoder_sequences, training=training)
            
        logger.debug(
            f"Encoder output sequence shape: {encoder_sequences.shape}"
        )
        # --- 3. Decoder Path (Prepares Context for Forecasting) ---
        # The decoder uses `static_context` and `future_input` over the
        # `forecast_horizon`.
        # Combine static context with future features for the decoder.
        static_expanded =None 
        if static_context is not None:
            static_expanded = tf_expand_dims(static_context, 1)
            static_expanded = tf_tile(
                static_expanded, [1, self.forecast_horizon, 1]
            )
            
        future_with_pos = self.positional_encoding(
            fut_proc, training=training
        )
        
        decoder_parts = []
        if static_expanded is not None :
            decoder_parts.append(static_expanded)
        if self.future_input_dim > 0: 
            decoder_parts.append(future_with_pos)
            
        if not decoder_parts: 
            batch_size = tf_shape(dynamic_input)[0]
            raw_decoder_input = tf_zeros(
                (batch_size, self.forecast_horizon, self.attention_units))
        else: 
            raw_decoder_input = tf_concat(decoder_parts, axis=-1)
            
        # Project decoder input to the right dimension for attention.
        projected_decoder_input = self.decoder_input_projection(
            raw_decoder_input
        )
        
        logger.debug(
            f"Projected decoder input shape: {projected_decoder_input.shape}"
        )

        # --- 4. Attention-based Fusion ---
        # The decoder context (query) attends to the encoder sequences (key/value).
        cross_att_out = self.cross_attention(
            [projected_decoder_input, encoder_sequences], 
            training=training
        )
        
        att_proc = self.attention_processing_grn(
            cross_att_out, training=training
        )
        # Process attention output and add residual connection.
        if self.use_residuals and self.decoder_add_norm:
            context_att = self.decoder_add_norm[0](
                [projected_decoder_input, att_proc]
            )
            context_att = self.decoder_add_norm[1](context_att)
        else:
            context_att = cross_att_out
            
        # Apply further attention layers to refine the context.
        hier_att_out = self.hierarchical_attention(
            [context_att, context_att], training=training # Self-attention
        )
        mem_att_out = self.memory_augmented_attention(
            hier_att_out, training=training
        )
        
        # --- 5. Final Combination and Aggregation ---
        final_features = self.multi_resolution_attention_fusion(
            mem_att_out, training=training
        )

        if self.use_residuals and self.final_add_norm:
            res_base = self.residual_dense(context_att)
            final_features = self.final_add_norm[0](
                [final_features, res_base]
            )
            final_features = self.final_add_norm[1](final_features)
        
        logger.debug(
            f"Shape after final fusion: {final_features.shape}"
        )

        # Collapse the time dimension to get a single vector for the decoder.
        return aggregate_time_window_output(final_features, self.final_agg)


    
    def call_(self, inputs: List[Optional[Tensor]], training: bool = False) -> Tensor:
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
    
    def run_halnet_core(self, static_input, dynamic_input, future_input, training):
        """Executes data-driven pipeline using an encoder-decoder."""
        time_steps = tf_shape(dynamic_input)[1]

        # 1. Initial Feature Processing & Slicing based on mode
        static_context, dyn_proc, fut_enc_proc, fut_dec_proc = (
            None, dynamic_input, None, future_input
        )
        if self.use_vsn:
            if self.static_vsn:
                static_context = self.static_vsn_grn(self.static_vsn(
                    static_input, training=training), training=training)
            if self.dynamic_vsn:
                dyn_proc = self.dynamic_vsn_grn(self.dynamic_vsn(
                    dynamic_input, training=training), training=training)
            if self.future_vsn:
                # Process the entire future tensor first
                future_processed = self.future_vsn_grn(self.future_vsn(
                    future_input, training=training), training=training)
        else: # Non-VSN path
            if self.static_dense:
                static_context = self.grn_static_non_vsn(self.static_dense(
                    static_input), training=training)
            dyn_proc = self.dynamic_dense(dynamic_input)
            future_processed = self.future_dense(future_input)

        # Handle TFT-like input slicing
        if self.mode == 'tft_like':
            fut_enc_proc = future_processed[:, :time_steps, :]
            fut_dec_proc = future_processed[:, time_steps:, :]
        else: # For pihal_like, encoder does not use future inputs
            fut_enc_proc = None
            fut_dec_proc = future_processed
            
        # 2. Encoder Path
        encoder_input_parts = [dyn_proc]
        if fut_enc_proc is not None:
            encoder_input_parts.append(fut_enc_proc)
        encoder_raw = tf_concat(encoder_input_parts, axis=-1)
        encoder_input = self.positional_encoding(encoder_raw, training=training)
        lstm_out = self.multi_scale_lstm(encoder_input, training=training)
        encoder_sequences = aggregate_multiscale_on_3d(lstm_out, mode='concat')
        
        # 3. Decoder Path
        static_expanded = None
        if static_context is not None:
            static_expanded = tf_expand_dims(static_context, 1)
            static_expanded = tf_tile(
                static_expanded, [1, self.forecast_horizon, 1])
        future_with_pos = self.positional_encoding(fut_dec_proc, training=training)
        
        decoder_parts = [future_with_pos]
        if static_expanded is not None:
            decoder_parts.append(static_expanded)
        raw_decoder_input = tf_concat(decoder_parts, axis=-1)
        projected_decoder_input = self.decoder_input_projection(raw_decoder_input)

        # 4. Attention Fusion
        cross_att_out = self.cross_attention(
            [projected_decoder_input, encoder_sequences], training=training)
        att_proc = self.attention_processing_grn(cross_att_out, training=training)
        
        if self.use_residuals and self.decoder_add_norm:
            context_att = self.decoder_add_norm[0]([projected_decoder_input, att_proc])
            context_att = self.decoder_add_norm[1](context_att)
        else:
            context_att = att_proc
            
        # 5. Final Processing
        # Apply further attention layers to refine the context.
        hier_att_out = self.hierarchical_attention(
            [context_att, context_att], training=training # Self-attention
        )
        mem_att_out = self.memory_augmented_attention(
            hier_att_out, training=training
        )
        
        final_features = self.multi_resolution_attention_fusion(mem_att_out)
        
        if self.use_residuals and self.final_add_norm:
            res_base = self.residual_dense(context_att)
            final_features = self.final_add_norm[0]([final_features, res_base])
            final_features = self.final_add_norm[1](final_features)
        
        return aggregate_time_window_output(final_features, self.final_agg)

    def call(
            self, inputs: List[Optional[Tensor]], training: bool = False) -> Tensor:
        """Forward pass for the HALNet model."""
        # Adjust validation based on mode
        expected_future_span = None
        if self.mode == 'tft_like':
            expected_future_span = self.max_window_size + self.forecast_horizon
        else: # pihal_like
            expected_future_span = self.forecast_horizon

        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=inputs, static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            # Pass the expected span to the validator if it supports it
            # Or perform a check here.
            mode='strict', 
            model_name='xtft'
        )
        # Convert the Python int to a tensor so the comparison is graph‑safe
        # Check future_input shape based on mode
        actual_future_span = tf_shape(future_p)[1]
        expected_span_tensor = tf_convert_to_tensor(
            expected_future_span, dtype=actual_future_span.dtype
        )
    
        tf_assert_equal(   # raises InvalidArgumentError in graph mode
            actual_future_span,
            expected_span_tensor ,
            message=(
                f"For mode='{self.mode}', `future_input` time dimension "
                f"must be {expected_future_span} but is "
                f"{actual_future_span}."
            ),
        )
        
        # if actual_future_span != expected_future_span:
        #     raise ValueError(
        #         f"For mode='{self.mode}', `future_input` time dimension must be "
        #         f"{expected_future_span}, but got {actual_future_span}."
        #     )
        
        final_features = self.run_halnet_core(
            static_p, dynamic_p, future_p, training=training)
        decoded_outputs = self.multi_decoder(
            final_features, training=training)
        
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
            'use_vsn': self.use_vsn,
            'vsn_units': self.vsn_units,
            'mode': self.mode 
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)



HALNet.__doc__ = r"""
Hybrid Attentive LSTM Network (HAL-Net).

A **data‑driven** encoder–decoder architecture that couples
multi‑scale LSTMs with hierarchical attention to deliver accurate
multi‑horizon forecasts from static, dynamic‑past, and known‑future
covariates.  HAL‑Net is a pared‑down variant of PIHALNet:
it omits physics constraints and anomaly modules, making it suitable
for purely statistical settings.

See more in :ref:`User Guide <user_guide>`. 

Parameters
----------
{params.base.static_input_dim}
{params.base.dynamic_input_dim}
{params.base.future_input_dim}

output_dim : int, default 1  
    Number of target variables produced at each forecast step.  
    The model outputs a tensor of shape  
    :math:`(B, \, H, \, Q, \, \text{{output\_dim}})` when *quantiles* are  
    provided, or :math:`(B, \, H, \, \text{{output\_dim}})` for point  
    forecasts, where  

    .. math::  
       B = \text{{batch size}},\qquad  
       H = \text{{forecast horizon}},\qquad  
       Q = |\text{{quantiles}}|.  

forecast_horizon : int, default 1  
    Length of the prediction window into the future.  The dynamic  
    encoder ingests *max_window_size* past steps and the decoder emits  
    :math:`H` steps ahead, where :math:`H=\text{{forecast\_horizon}}`.  
    Setting :math:`H > 1` enables multi‑horizon sequence‑to‑sequence  
    forecasts.  

quantiles : list[float] or None, default None  
    Optional quantile levels :math:`0 < q_1 < \dots < q_Q < 1`.  
    When supplied, a  
    :class:`fusionlab.nn.components.QuantileDistributionModeling` head  
    scales the point forecast :math:`\hat{{y}}` into quantile estimates  

    .. math::  
       \hat{{y}}^{{(q)}} = \hat{{y}} + \sigma \,\Phi^{{-1}}(q),  

    where :math:`\sigma` is a learned spread parameter and  
    :math:`\Phi^{{-1}}` is the probit function.  Omit or set to *None* to  
    obtain deterministic forecasts.  

{params.base.embed_dim}
{params.base.hidden_units}
{params.base.lstm_units}
{params.base.attention_units}
{params.base.num_heads}
{params.base.dropout_rate}
{params.base.max_window_size}
{params.base.memory_size}
{params.base.scales}
{params.base.multi_scale_agg}
{params.base.final_agg}
{params.base.activation}
{params.base.use_residuals}
{params.base.use_vsn}
{params.base.vsn_units}


name : str, default ``"HALNet"``  
    Model identifier passed to :pyclass:`tf.keras.Model`.  Appears in  
    weight filenames and TensorBoard scopes.  

**kwargs  
    Additional keyword arguments forwarded verbatim to the  
    :pyclass:`tf.keras.Model` constructor—e.g. ``dtype="float64"`` or  
    ``run_eagerly=True``.  

Notes  
-----  
The composite latent size produced by the cross‑attention block is  
:math:`d_\text{{model}} = \text{{attention\_units}}`.  For stable training  
ensure :math:`d_\text{{model}}` is divisible by *num_heads*.

See Also  
--------  
* :class:`fusionlab.nn.pinn.PIHALNet` – physics‑informed extension.  
* :func:`fusionlab.utils.data_utils.widen_temporal_columns` – prepares  
  wide data frames for plotting forecasts.  

Examples
--------
>>> from fusionlab.nn.pinn import HALNet  
>>> model = HALNet(  
...     static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,  
...     output_dim=2, forecast_horizon=24, quantiles=[0.1, 0.5, 0.9],  
...     scales=[1, 3], multi_scale_agg="concat", final_agg="last",  
...     attention_units=64, num_heads=8, dropout_rate=0.15,  
... )  
>>> x_static  = tf.random.normal([32, 4])              # B × S  
>>> x_dynamic = tf.random.normal([32, 10, 8])          # B × T × D  
>>> x_future  = tf.random.normal([32, 24, 6])          # B × H × F  
>>> y_hat = model({{  
...     "static_features": x_static,  
...     "dynamic_features": x_dynamic,  
...     "future_features": x_future,  
...     "coords": tf.zeros([32, 24, 3]),               # dummy (t, x, y)  
... }})  
>>> y_hat["subs_pred"].shape  
TensorShape([32, 24, 3, 2])  # B × H × Q × output_dim

See Also
--------
fusionlab.nn.pinn.PIHALNet
fusionlab.nn.components.MultiScaleLSTM
fusionlab.nn.components.VariableSelectionNetwork

References  
----------  
.. [1] Vaswani et al., “Attention Is All You Need,” *NeurIPS 2017*.  
.. [2] Lim et al., “Temporal Fusion Transformers for Interpretable  
       Multi‑Horizon Time Series Forecasting,” *IJCAI 2021*.  
""".format(params=_param_docs)
