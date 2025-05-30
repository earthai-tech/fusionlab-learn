# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com> 

"""
Implements a standard Transformer architecture tailored for multi-horizon 
time-series forecasting.
"""
from __future__ import annotations

from numbers import Real, Integral 
from textwrap import dedent # noqa
from typing import List, Optional, Union, Tuple

from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...compat.sklearn import validate_params, Interval, StrOptions 

from ...api.property import NNLearner
from .. import KERAS_DEPS, KERAS_BACKEND, dependency_message 

if KERAS_BACKEND:
    Layer = KERAS_DEPS.Layer
    Model = KERAS_DEPS.Model
    Input = KERAS_DEPS.Input 
    Dense = KERAS_DEPS.Dense
    Dropout = KERAS_DEPS.Dropout
    LayerNormalization = KERAS_DEPS.LayerNormalization
    MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
    Concatenate = KERAS_DEPS.Concatenate
    Add = KERAS_DEPS.Add
    Tensor = KERAS_DEPS.Tensor
    register_keras_serializable = KERAS_DEPS.register_keras_serializable
    
    tf_expand_dims = KERAS_DEPS.expand_dims
    tf_tile = KERAS_DEPS.tile
    tf_shape = KERAS_DEPS.shape
    tf_squeeze = KERAS_DEPS.squeeze
    tf_cast = KERAS_DEPS.cast
    tf_float32 = KERAS_DEPS.float32
    tf_ones = KERAS_DEPS.ones 
    tf_zeros =KERAS_DEPS.zeros
    tf_linalg = KERAS_DEPS.linalg
    tf_autograph =KERAS_DEPS.autograph 
    tf_logical_and = KERAS_DEPS.logical_and 
    tf_greater =KERAS_DEPS.greater 
    tf_constant =KERAS_DEPS.constant 
    tf_bool = KERAS_DEPS.bool 
    tf_cond =KERAS_DEPS.cond 
    tf_rank = KERAS_DEPS.rank
    tf_where =KERAS_DEPS.where
    tf_stack =KERAS_DEPS.stack 
    tf_int32 =KERAS_DEPS.int32

    from ..components import (
        PositionalEncodingTF, 
        QuantileDistributionModeling,
        GatedResidualNetwork,  
        TransformerEncoderLayer, 
        TransformerDecoderLayer, 
        create_causal_mask, 
    )
    from ..utils import prepare_model_inputs_in
    from .._tensor_validation import validate_model_inputs


DEP_MSG = dependency_message('nn._transformers') 

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())


@register_keras_serializable('fusionlab.nn.transformers', name="TimeSeriesTransformer")
class TimeSeriesTransformer(Model, NNLearner):
    @validate_params({ 
        "static_input_dim": [Interval(Integral, 0, None, closed='left')], 
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "future_input_dim": [Interval(Integral, 0, None, closed='left')], 
        "embed_dim": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "ffn_dim": [Interval(Integral, 1, None, closed='left')],
        "num_encoder_layers": [Interval(Integral, 1, None, closed='left')],
        "num_decoder_layers": [Interval(Integral, 1, None, closed='left')],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "input_dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "max_seq_len_encoder": [Interval(Integral, 1, None, closed='left')],
        "max_seq_len_decoder": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', StrOptions({'auto'}), None],
        "use_grn_for_static": [bool],
        "static_integration_mode": [StrOptions({
            'add_to_encoder_input', 'add_to_decoder_input', 'none'
            })],
        "ffn_activation": [str, callable],
        "layer_norm_epsilon": [Real],
    })
    def __init__(
        self,
        static_input_dim: int, 
        dynamic_input_dim: int,
        future_input_dim: int, 
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        forecast_horizon: int = 1,
        output_dim: int = 1, 
        dropout_rate: float = 0.1,
        input_dropout_rate: float = 0.1,
        max_seq_len_encoder: int = 100,
        max_seq_len_decoder: int = 50,
        quantiles: Optional[List[float]] = None,
        use_grn_for_static: bool = False,
        static_integration_mode: str = 'add_to_decoder_input',
        ffn_activation: str = 'relu',
        layer_norm_epsilon: float = 1e-6,
        name: Optional[str] = "TimeSeriesTransformer",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        if future_input_dim > 0 and forecast_horizon <= 0:
            raise ValueError(
                "forecast_horizon must be > 0 if future_input_dim > 0"
                )

        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.input_dropout_rate = input_dropout_rate
        self.max_seq_len_encoder = max_seq_len_encoder
        self.max_seq_len_decoder = max_seq_len_decoder
        self.quantiles = quantiles
        self.use_grn_for_static = use_grn_for_static
        self.static_integration_mode = static_integration_mode
        self.ffn_activation = ffn_activation
        self.layer_norm_epsilon = layer_norm_epsilon

        self.dynamic_embed = Dense(embed_dim, name="dynamic_embedding")
        if self.future_input_dim > 0:
            self.future_embed = Dense(embed_dim, name="future_embedding")
        
        if self.static_input_dim > 0:
            if self.use_grn_for_static:
                self.static_processor = GatedResidualNetwork(
                    units=embed_dim, dropout_rate=dropout_rate, 
                    activation=ffn_activation, 
                    name="static_grn_processor"
                )
            else:
                self.static_processor = Dense(
                    embed_dim, activation=ffn_activation, 
                    name="static_dense_processor"
                )

        self.pos_encoding_encoder = PositionalEncodingTF(
            max_seq_len_encoder, embed_dim, name="pos_encoder"
        )
        self.pos_encoding_decoder = PositionalEncodingTF(
            max_seq_len_decoder, embed_dim, name="pos_decoder"
        )
        self.input_dropout = Dropout(input_dropout_rate)

        self.encoder_layers = [
            TransformerEncoderLayer(
                embed_dim, num_heads, ffn_dim, dropout_rate, 
                ffn_activation, layer_norm_epsilon, 
                name=f"encoder_layer_{i}"
            ) for i in range(num_encoder_layers)
        ]
        self.decoder_layers = [
            TransformerDecoderLayer(
                embed_dim, num_heads, ffn_dim, dropout_rate,
                ffn_activation, layer_norm_epsilon, 
                name=f"decoder_layer_{i}"
            ) for i in range(num_decoder_layers)
        ]

        self.final_dense = Dense(output_dim, name="final_projection")
        if self.quantiles:
            self.quantile_modeling = QuantileDistributionModeling(
                quantiles=self.quantiles, output_dim=output_dim
            )
        else:
            self.quantile_modeling = None
    

    @tf_autograph.experimental.do_not_convert 
    def call(
        self, 
        inputs: Union[List[Optional[Tensor]], Tuple[Optional[Tensor], ...]], 
        training: bool = False
    ) -> Tensor:
        """
        Forward pass for the TimeSeriesTransformer.
        
        Args:
            inputs: A list or tuple of tensors. The elements are:
                1. static_input (Batch, static_input_dim)
                   (Can be None if self.static_input_dim is 0).
                2. dynamic_input (Batch, T_past, dynamic_input_dim)
                3. future_input (Batch, T_decode_seq, future_input_dim)
                   (T_decode_seq is typically self.forecast_horizon.
                    Can be None if self.future_input_dim is 0).
                
                The order must be consistent if some inputs are None.
                It's safer if the model expects a dict or if caller
                ensures correct list even with Nones.
                This `call` method expects a list/tuple that will be
                passed to `prepare_model_inputs`.
            training: Boolean, whether the model is in training mode.

        Returns:
            A tensor with forecast predictions.
        """
        # 1. Initial Unpacking for prepare_model_inputs
 
        _static_in, _dynamic_in, _future_in = validate_model_inputs (
            inputs = inputs, 
            static_input_dim= self.static_input_dim, 
            dynamic_input_dim= self.dynamic_input_dim , 
            future_covariate_dim= self.future_input_dim, 
            forecast_horizon= self.forecast_horizon, 
            mode='soft', 
            model_name="tft_flex"
        )

        # Use the utility to prepare and validate inputs
        # Note: `prepare_model_inputs` expects (dynamic, static, future)

        static_input_p, dynamic_input_p, future_input_p = prepare_model_inputs_in(
            dynamic_input=_dynamic_in,
            static_input=_static_in,    # Swapped order
            future_input=_future_in,    # Swapped order
            model_type='strict', # Transformer expects all tensor inputs
            verbose=0 # Or pass from a model config
        )
        logger.debug(
            "Prepared shapes: static=%s, dyn=%s, fut=%s",
            static_input_p.shape, dynamic_input_p.shape, future_input_p.shape
        )

  
        # --- 1. Process Static Features ---
        static_context_vector = None
        static_context_vector_expanded = None
        # Check static_input_p is not None AND its feature dim > 0
        
        if self.static_input_dim is not None and self.static_input_dim > 0:
            static_context_vector = self.static_processor(
                static_input_p, training=training
            ) 
            static_context_vector_expanded = tf_expand_dims(
                static_context_vector, 1
            ) 
            logger.debug(
                "Static context shape: %s",
                static_context_vector.shape
            )
        # else: 
        #     static_context_vector = static_input_p # not context add 

        # --- 2. Encoder Path ---
        dynamic_emb = self.dynamic_embed(dynamic_input_p)
        enc_input = self.pos_encoding_encoder(dynamic_emb)
        
        if (static_context_vector_expanded is not None 
            and self.static_integration_mode == 'add_to_encoder_input'):
            enc_input = Add()([enc_input, tf_tile(
                static_context_vector_expanded, 
                [1, tf_shape(enc_input)[1], 1])
                ])
    
        
        enc_input = self.input_dropout(enc_input, training=training)

        enc_output = enc_input
        for i in range(self.num_encoder_layers):
            enc_output = self.encoder_layers[i](
                enc_output, training=training, attention_mask=None
            )
            
            logger.debug(
                "Encoder output shape: %s",
                enc_output.shape
            )
        # --- 3. Decoder Path ---
        # Decoder input: `future_input_p` should be (B, H, F_future_dim)
        # or (B, H, 0) if future_input_dim is effectively zero.
        # The embedding layer handles feature_dim > 0.
        # If feature_dim is 0, `future_embed` might not be called.
        decoder_seq_len = self.forecast_horizon
        
        if self.future_input_dim is not None and self.future_input_dim > 0:
            # Take only up to forecast_horizon for decoder input sequence
            dec_emb = self.future_embed(
                future_input_p[:, :decoder_seq_len, :]
            ) 
        else:
            # If no actual future features, decoder starts from zeros 
            # or learned embeddings. Using zeros for simplicity here.
            batch_size = tf_shape(dynamic_input_p)[0]
            dec_emb = tf_zeros(
                (batch_size, decoder_seq_len, self.embed_dim), 
                dtype=tf_float32
            )

        dec_input = self.pos_encoding_decoder(dec_emb)
        
        if (static_context_vector_expanded is not None 
            and self.static_integration_mode == 'add_to_decoder_input'):
            dec_input = Add()([dec_input, tf_tile(
                static_context_vector_expanded, 
                [1, tf_shape(dec_input)[1], 1])
                ])

        dec_input = self.input_dropout(
            dec_input, training=training)
        
        logger.debug(
            "Decoder input shape: %s", dec_input.shape
        )
        
        look_ahead_mask = create_causal_mask(tf_shape(dec_input)[1])

        dec_output = dec_input
        for i in range(self.num_decoder_layers):
            dec_output = self.decoder_layers[i](
                dec_output, enc_output, training=training,
                look_ahead_mask=look_ahead_mask, padding_mask=None
            ) 
        
        logger.debug(
            "Decoder output shape: %s", dec_output.shape
        )
        # --- 4. Final Output ---
        predictions = self.final_dense(dec_output) 

        if self.quantile_modeling:
            predictions = self.quantile_modeling(predictions)
            # get the static (build‐time) rank
            rank = predictions.shape.ndims
            if (
                    self.output_dim == 1 
                    and self.quantiles
                    and rank == 4
                ): # static Python int comparison
                predictions = tf_squeeze(predictions, axis=-1)
                logger.debug(
                    "Predictions shape after quantiles: %s",
                    predictions.shape
                )

        logger.debug(
            "Exiting call(), output shape: %s", predictions.shape
        )
 
        return predictions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "static_input_dim": self.static_input_dim,
            "dynamic_input_dim": self.dynamic_input_dim,
            "future_input_dim": self.future_input_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "forecast_horizon": self.forecast_horizon,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "input_dropout_rate": self.input_dropout_rate,
            "max_seq_len_encoder": self.max_seq_len_encoder,
            "max_seq_len_decoder": self.max_seq_len_decoder,
            "quantiles": self.quantiles,
            "use_grn_for_static": self.use_grn_for_static,
            "static_integration_mode": self.static_integration_mode,
            "ffn_activation": self.ffn_activation,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

