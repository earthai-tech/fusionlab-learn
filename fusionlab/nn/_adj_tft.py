# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Temporal Fusion Transformer (TFT), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""
from textwrap import dedent 
from numbers import Real, Integral  

from .._fusionlog import fusionlog 
from ..api.property import NNLearner 
from ..decorators import Appender 
from ..utils.deps_utils import ensure_pkg 
from ..compat.sklearn import validate_params, Interval, StrOptions
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message 

if KERAS_BACKEND:
    LSTM = KERAS_DEPS.LSTM
    LSTMCell=KERAS_DEPS.LSTMCell
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
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    
    tf_reduce_sum =KERAS_DEPS.reduce_sum
    tf_stack =KERAS_DEPS.stack
    tf_expand_dims =KERAS_DEPS.expand_dims
    tf_tile =KERAS_DEPS.tile
    tf_range =KERAS_DEPS.range 
    tf_concat =KERAS_DEPS.concat
    tf_shape =KERAS_DEPS.shape
    tf_zeros=KERAS_DEPS.zeros
    tf_float32=KERAS_DEPS.float32
    tf_reshape=KERAS_DEPS.reshape
    tf_autograph=KERAS_DEPS.autograph
    tf_multiply=KERAS_DEPS.multiply
    tf_reduce_mean = KERAS_DEPS.reduce_mean
    tf_get_static_value=KERAS_DEPS.get_static_value
    
    from ._tensor_validation import validate_tft_inputs
    from ._tft import TemporalFusionTransformer
    from .losses import combined_quantile_loss 
    from .utils import set_default_params

    
DEP_MSG = dependency_message('transformers.tft') 

# ------------------------ TFT implementation --------------------------------

@Appender(dedent(
    TemporalFusionTransformer.__doc__.replace ('TemporalFusionTransformer', 'TFT')
    ), join='\n'
 )
@register_keras_serializable('gofast.nn.transformers', name="TFT")
class TFT(Model, NNLearner):
    @validate_params({
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "static_input_dim": [Interval(Integral, 1, None, closed='left'), None], 
        "future_input_dim": [Interval(Integral, 1, None, closed='left'), None], 
        "hidden_units": [Interval(Integral, 1, None, closed='left')], 
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', StrOptions({'auto'}), None],
        "activation": [
            StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", "gelu"}),
            callable],
        "use_batch_norm": [bool],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": ['array-like', Interval(Integral, 1, None, closed='left'), None]
        },
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        dynamic_input_dim,      
        future_input_dim=None,    
        static_input_dim=None,  
        hidden_units=32, 
        num_heads=4,            
        dropout_rate=0.1,       
        forecast_horizon=1,     
        quantiles=None,         
        activation='elu',       
        use_batch_norm=False,   
        lstm_units=None,        
        output_dim=1,           
    ):
        super().__init__()
        self.dynamic_input_dim = dynamic_input_dim
        self.hidden_units = hidden_units
        self.future_input_dim = future_input_dim
        self.static_input_dim = static_input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.forecast_horizon = forecast_horizon
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.lstm_units = lstm_units if lstm_units is not None else hidden_units
        self.output_dim = output_dim

        # Initialize logger
        self.logger = fusionlog().get_fusionlab_logger(__name__)
        self.logger.debug(
            "Initializing TemporalFusionTransformer with parameters: "
            f"static_input_dim={static_input_dim}, "
            f"dynamic_input_dim={dynamic_input_dim}, "
            f"future_input_dim={future_input_dim}, "
            f"hidden_units={hidden_units}, "
            f"num_heads={num_heads}, "
            f"dropout_rate={dropout_rate}, "
            f"forecast_horizon={forecast_horizon}, "
            f"quantiles={quantiles}, "
            f"activation={activation}, "
            f"use_batch_norm={use_batch_norm}, "
            f"lstm_units={lstm_units}"
            f"output_dim={output_dim}"
        )
        # Handle default quantiles, scales and multi_scale_agg 
        self.quantiles, _, _ = set_default_params(quantiles) 
        
        # Embedding layers for dynamic (past) inputs
        self.past_embedding = TimeDistributed(
            Dense(hidden_units, activation=None),
            name='past_embedding'
        )
        
        # Embedding layers for future inputs if provided
        self.future_embedding = (
            TimeDistributed(
                Dense(hidden_units, activation=None),
                name='future_embedding'
            ) 
            if self.future_input_dim is not None else None
        )
        
        # Embedding layer for static inputs using TimeDistributed
        self.static_embedding = (
            TimeDistributed(
                Dense(hidden_units, activation=None),
                name='static_embedding'
            ) 
            if self.static_input_dim is not None else None
        )
        
        # Variable Selection Network for dynamic (past) inputs
        self.past_varsel = VariableSelectionNetwork(
            num_inputs=self.dynamic_input_dim, 
            hidden_units=hidden_units, 
            dropout_rate=dropout_rate, 
            activation=activation, 
            use_batch_norm=use_batch_norm
        )
        
        # LSTM Encoder
        self.encoder = LSTMEncoder(
            lstm_units=self.lstm_units, 
            dropout_rate=dropout_rate
        )
        
        # Variable Selection Network for future inputs if provided
        self.future_varsel = (
            VariableSelectionNetwork(
                num_inputs=self.future_input_dim, 
                hidden_units=hidden_units, 
                dropout_rate=dropout_rate, 
                activation=activation, 
                use_batch_norm=use_batch_norm
            ) 
            if self.future_input_dim is not None else None
        )
        
        # LSTM Decoder
        self.decoder = LSTMDecoder(
            lstm_units=self.lstm_units, 
            dropout_rate=dropout_rate
        )
        
        # Variable Selection Network for static inputs if provided
        self.static_varsel = (
            VariableSelectionNetwork(
                num_inputs=self.static_input_dim, 
                hidden_units=hidden_units, 
                dropout_rate=dropout_rate, 
                activation=activation, 
                use_batch_norm=use_batch_norm
            ) 
            if self.static_input_dim is not None else None
        )
        
        # Temporal Fusion Decoder (includes attention and gating)
        self.temporal_fusion_decoder = TemporalFusionDecoder(
            hidden_units=hidden_units, 
            num_heads=num_heads, 
            dropout_rate=dropout_rate, 
            activation=activation, 
            use_batch_norm=use_batch_norm
        )
        
        # Final Dense layer(s) for output
        self.num_quantiles =None 
        if self.quantiles is not None:
            # If quantiles are provided, adjust output size accordingly
            self.num_quantiles = len(self.quantiles)
        self.output_layer = (
            Dense(
                self.output_dim * self.num_quantiles, 
                name='output_layer'
            ) 
            if self.quantiles is not None else 
            Dense(
                self.output_dim, 
                name='output_layer'
            )
        )
    
    def call(self, inputs, training=None):
        # Unpack inputs
        past_inputs, future_inputs, static_inputs=validate_tft_inputs(
            inputs =inputs,
            static_input_dim=self.static_input_dim, 
            dynamic_input_dim= self.dynamic_input_dim, 
            future_covariate_dim= self.future_input_dim, 
        )
        # 1) Embed past inputs
        # ---> (batch_size, past_steps, dynamic_input_dim, 1)
        past_inputs_expanded = tf_expand_dims(past_inputs, axis=-1)  
        # ---> (batch_size, past_steps, dynamic_input_dim, hidden_units)
        past_embedded = self.past_embedding(past_inputs_expanded) 
        
        self.logger.debug(f"Past inputs shape: {past_inputs.shape}")
        self.logger.debug(f"Past embedded shape: {past_embedded.shape}")
        
        # 2) Variable Selection for past inputs
        past_selected, past_weights = self.past_varsel(
            past_embedded, 
            training=training
        )  # (batch_size, past_steps, hidden_units)
        self.logger.debug(f"Past selected shape: {past_selected.shape}")
        
        # 3) Encode with LSTM
        encoder_out, state_h, state_c = self.encoder(
            past_selected, 
            training=training
        )  # (batch_size, past_steps, lstm_units)
        self.logger.debug(f"Encoder output shape: {encoder_out.shape}")
        
        # 4) Embed future inputs if provided
        if self.future_varsel is not None:
            # ---> (batch_size, forecast_horizon, future_input_dim, 1)
            future_inputs_expanded = tf_expand_dims(future_inputs, axis=-1)  
            # ---> (batch_size, forecast_horizon, future_input_dim, hidden_units)
            future_embedded = self.future_embedding(future_inputs_expanded)  
            
            self.logger.debug(f"Future inputs shape: {future_inputs.shape}")
            self.logger.debug(f"Future embedded shape: {future_embedded.shape}")
            
            # Variable Selection for future inputs
            future_selected, future_weights = self.future_varsel(
                future_embedded, 
                training=training
            )  # (batch_size, forecast_horizon, hidden_units)
            self.logger.debug(f"Future selected shape: {future_selected.shape}")
        else:
            # Placeholder for no future inputs
            future_selected = tf_zeros(
                shape=(tf_shape(past_inputs)[0], self.forecast_horizon, self.hidden_units),
                dtype=tf_float32
            )
            self.logger.info("No future inputs provided; using zero placeholder.")
        
        # 5) Decode with LSTM
        decoder_out, decoder_state = self.decoder(
            future_selected, 
            initial_state=(state_h, state_c), 
            training=training
        )  # (batch_size, forecast_horizon, lstm_units)
        self.logger.debug(f"Decoder output shape: {decoder_out.shape}")
        
        # 6) Embed and Variable Selection for static inputs if provided
        if self.static_varsel is not None:
            #  ---> (batch_size, static_input_dim, 1)
            static_inputs_expanded = tf_expand_dims(static_inputs, axis=-1)  
            # ---> (batch_size, static_input_dim, hidden_units)
            static_embedded = self.static_embedding(static_inputs_expanded)  
            
            self.logger.debug(f"Static inputs shape: {static_inputs.shape}")
            self.logger.debug(f"Static embedded shape: {static_embedded.shape}")
            
            # Variable Selection for static inputs
            static_selected, static_weights = self.static_varsel(
                static_embedded, 
                training=training
            )  # (batch_size, hidden_units)
            self.logger.debug(f"Static selected shape: {static_selected.shape}")
        else:
            static_selected = None
        
        # 7) Temporal Fusion Decoder
        fused_output = self.temporal_fusion_decoder(
            decoder_out, 
            static_context=static_selected, 
            training=training
        )  # (batch_size, forecast_horizon, hidden_units)
        self.logger.debug(f"Fused output shape: {fused_output.shape}")
        
        # 8) Final Projection
        outputs = self.output_layer(fused_output)  
        # (batch_size, forecast_horizon, output_dim * num_quantiles) 
        # or (batch_size, forecast_horizon, output_dim)
        self.logger.debug(f"Output layer shape: {outputs.shape}")
        
        # Reshape if quantiles are provided
        if self.quantiles is not None:
            outputs = tf_reshape(
                outputs, 
                (-1, self.forecast_horizon, self.num_quantiles, self.output_dim)
            )
            self.logger.debug(f"Reshaped outputs shape: {outputs.shape}")
        
        return outputs
    
    def compilex(self, optimizer, loss=None, **kwargs):
        """
        Custom compile method to handle quantile loss.
        
        Args:
            optimizer: Optimizer to use.
            loss: Loss function (optional).
            **kwargs: Additional keyword arguments.
        """
        if self.quantiles is None:
            # Deterministic scenario with Mean Squared Error
            super().compile(
                optimizer=optimizer, 
                loss=loss or 'mse', 
                **kwargs
            )
            self.logger.info("Compiled with MSE loss for deterministic predictions.")
        else:
            # Probabilistic scenario with combined quantile loss
            quantile_loss_fn = combined_quantile_loss(self.quantiles)
            super().compile(
                optimizer=optimizer, 
                loss=quantile_loss_fn, 
                **kwargs
            )
            self.logger.info(
                f"Compiled with combined quantile loss for quantiles: {self.quantiles}"
            )
    
    def get_config(self):
        """
        Returns the config of the model for serialization.
        """
        config = super().get_config().copy()
        config.update({
            'dynamic_input_dim': self.dynamic_input_dim, 
            'static_input_dim' : self.static_input_dim,
            'future_input_dim' : self.future_input_dim,
            'hidden_units'     : self.hidden_units,
            'num_heads'        : self.num_heads,
            'dropout_rate'     : self.dropout_rate,
            'forecast_horizon': self.forecast_horizon,
            'quantiles'        : self.quantiles,
            'activation'       : self.activation,
            'use_batch_norm'   : self.use_batch_norm,
            'lstm_units'       : self.lstm_units,
            'output_dim'       : self.output_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Creates a model from its config.
        """
        return cls(**config)

# -------------------------------------- TFT components -----------------------
# LSTM Encoder 
class LSTMEncoder(Layer):
    def __init__(self, lstm_units, dropout_rate=0.0):
        super().__init__()
        self.lstm_units = lstm_units
        self.lstm = LSTM(
            units=self.lstm_units, 
            return_sequences=True, 
            return_state=True, 
            dropout=dropout_rate
        )
    
    @tf_autograph.experimental.do_not_convert
    def call(self, x, initial_state=None, training=None):
        # x shape = (batch_size, timesteps, hidden_units)
        whole_seq_output, state_h, state_c = self.lstm(
            x, initial_state=initial_state, training=training
        )
        return whole_seq_output, state_h, state_c

# LSTM Decoder 
class LSTMDecoder(Layer):
    def __init__(self, lstm_units, dropout_rate=0.0):
        super().__init__()
        self.lstm_units = lstm_units
        self.lstm_cell = LSTMCell(
            units=self.lstm_units, 
            dropout=dropout_rate
        )
    
    @tf_autograph.experimental.do_not_convert
    def call(self, x, initial_state, training=None):
        # x shape = (batch_size, timesteps, hidden_units)
        outputs = []
        state_h, state_c = initial_state
        
        # Iterate over timesteps
        for t in range(x.shape[1]):
            xt = x[:, t, :]  # Shape: (batch_size, hidden_units)
            out, [state_h, state_c] = self.lstm_cell(
                xt, states=[state_h, state_c], training=training
            )
            outputs.append(out)
        
        # Stack outputs: (batch_size, timesteps, lstm_units)
        outputs = tf_stack(outputs, axis=1)
        return outputs, (state_h, state_c)

# Temporal Self-Attention
class TemporalSelfAttention(Layer):
    def __init__(self, hidden_units, num_heads, dropout_rate=0.0):
        super().__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=hidden_units, 
            dropout=dropout_rate
        )
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)
        self.grn = GatedResidualNetwork(
            hidden_units=hidden_units, 
            dropout_rate=dropout_rate
        )

    @tf_autograph.experimental.do_not_convert
    def call(self, x, mask=None, training=None):
        # Multi-Head Attention
        attn_output = self.mha(
            query=x, 
            value=x, 
            key=x, 
            attention_mask=mask, 
            training=training
        )
        attn_output = self.dropout(attn_output, training=training)
        
        # Residual connection and Layer Normalization
        out1 = self.layer_norm(x + attn_output)
        
        # Gated Residual Network
        out2 = self.grn(out1, training=training)
        return out2

# Temporal Fusion Decoder 
class TemporalFusionDecoder(Layer):
    def __init__(self, hidden_units, num_heads, dropout_rate=0.0,
                 activation='elu', use_batch_norm=False):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        
        # Temporal Self-Attention layer
        self.attention = TemporalSelfAttention(
            hidden_units=hidden_units, 
            num_heads=num_heads, 
            dropout_rate=dropout_rate
        )
        
        # Static enrichment via GRN
        self.static_enrichment = GatedResidualNetwork(
            hidden_units=hidden_units, 
            dropout_rate=dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        
        # Final GRN after attention
        self.post_attention_grn = GatedResidualNetwork(
            hidden_units=hidden_units, 
            dropout_rate=dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm
        )

    @tf_autograph.experimental.do_not_convert
    def call(self, decoder_seq, static_context=None, training=None):
        # Static enrichment: Incorporate static context into decoder sequence
        if static_context is not None:
            # Broadcast static context across time dimension
            time_steps = tf_shape(decoder_seq)[1]
            static_context_expanded = tf_tile(
                tf_expand_dims(static_context, axis=1), 
                [1, time_steps, 1]
            )
            enriched_seq = self.static_enrichment(
                decoder_seq, 
                context=static_context_expanded, 
                training=training
            )
        else:
            enriched_seq = decoder_seq
        
        # Temporal Self-Attention
        attn_out = self.attention(enriched_seq, training=training)
        
        # Post-Attention GRN
        out = self.post_attention_grn(attn_out, training=training)
        return out

# Gated Residual Network (GRN)
class GatedResidualNetwork(Layer):
    def __init__(
        self, 
        hidden_units, 
        output_units=None, 
        dropout_rate=0.0, 
        activation='elu', 
        use_batch_norm=False
    ):
        super().__init__()
        self.hidden_units = hidden_units
        self.output_units = output_units if output_units is not None else hidden_units
        self.dropout_rate= dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        
        # Define layers
        self.fc1 = Dense(self.hidden_units, activation=None)
        #self.activation_fn = Activation(self.activation)
        self.activation = activation

        self.dropout= Dropout(self.dropout_rate)
        self.fc2 = Dense(self.output_units, activation=None)
        
        # Optional Batch Normalization
        self.batch_norm= BatchNormalization() if self.use_batch_norm else None
        
        # Gating mechanism
        self.gate_dense = Dense(self.output_units, activation='sigmoid')
        
        # Skip connection adjustment if necessary
        self.skip_dense= (
            Dense(self.output_units, activation=None) 
            if self.output_units != self.hidden_units else None
        )
        
        # Layer Normalization for residual connection
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    @tf_autograph.experimental.do_not_convert
    def call(self, x, context=None, training=None):
        # Concatenate context if provided
        x_in = tf_concat([x, context], axis=-1) if context is not None else x
        
        # First Dense layer
        x_fc1 = self.fc1(x_in)
        
        # Activation function
        x_act = self.activation(x_fc1)
        
        # Optional Batch Normalization
        if self.batch_norm:
            x_act = self.batch_norm(x_act, training=training)
        
        # Dropout for regularization
        x_drp = self.dropout(x_act, training=training)
        
        # Second Dense layer
        x_fc2 = self.fc2(x_drp)
        
        # Gating mechanism
        gating = self.gate_dense(x_in)
        x_gated = tf_multiply(x_fc2, gating)
        
        # Adjust skip connection if output dimensions differ
        x_skip = self.skip_dense(x) if self.skip_dense else x
        
        # Residual connection
        x_res = x_skip + x_gated
        
        # Layer Normalization
        return self.layer_norm(x_res)

class VariableSelectionNetwork(Layer):
    def __init__(
        self, 
        num_inputs, 
        hidden_units, 
        dropout_rate=0.0, 
        activation='elu', 
        use_batch_norm=False
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm  = use_batch_norm
        
        # Initialize a GRN for each input variable
        self.grns = [
            GatedResidualNetwork(
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                activation=activation,
                use_batch_norm=use_batch_norm
            ) 
            for _ in range(num_inputs)
        ]
        
        # Softmax layer to compute variable weights
        self.softmax = Softmax(axis=-1)
    
    @tf_autograph.experimental.do_not_convert
    def call(self, x, training=None):
        """
        Handles both time-varying and static inputs by adjusting input dimensions.
        
        Args:
            x: 
                - Time-varying inputs: (batch_size, timesteps, num_inputs, embed_dim)
                - Static inputs: (batch_size, num_inputs, embed_dim)
            training: Boolean indicating training mode.
        
        Returns:
            weighted_sum: Tensor after variable selection.
            weights: Tensor of variable weights.
        """
        if len(x.shape) == 4:
            # Time-varying inputs
            grn_outputs = [
                self.grns[i](x[:, :, i, :], training=training) 
                for i in range(self.num_inputs)
            ]  # List of (batch_size, timesteps, hidden_units)
            # ---->  (batch_size, timesteps, num_inputs, hidden_units)
            grn_outputs = tf_stack(grn_outputs, axis=2) 
            
            # Compute weights across hidden units
                 # (batch_size, timesteps, num_inputs)
            flattened = tf_reduce_mean(grn_outputs, axis=-1) 
                 # (batch_size, timesteps, num_inputs)
            weights    = self.softmax(flattened)               
            
            # Weighted sum of GRN outputs
                  # (batch_size, timesteps, num_inputs, 1)
            w_expanded   = tf_expand_dims(weights, axis=-1)     
                  # (batch_size, timesteps, hidden_units)
            weighted_sum = tf_reduce_sum(grn_outputs * w_expanded, axis=2)  
            return weighted_sum, weights
        
        elif len(x.shape) == 3:
            # Static inputs
            grn_outputs = [
                self.grns[i](x[:, i, :], training=training) 
                for i in range(self.num_inputs)
            ]  # List of (batch_size, hidden_units)
                       # (batch_size, num_inputs, hidden_units)
            grn_outputs = tf_stack(grn_outputs, axis=1)  
            
            # Compute weights
                 # (batch_size, num_inputs)
            flattened = tf_reduce_mean(grn_outputs, axis=-1)  
                 # (batch_size, num_inputs)
            weights    = self.softmax(flattened)               
            
            # Weighted sum of GRN outputs
                   # (batch_size, num_inputs, 1)
            w_expanded   = tf_expand_dims(weights, axis=-1)    
                   # (batch_size, hidden_units)
            weighted_sum = tf_reduce_sum(grn_outputs * w_expanded, axis=1)  
            return weighted_sum, weights
        
        else:
            # Unsupported input dimensions: break and stop. 
            raise ValueError(
                "Input tensor must have 3 or 4 dimensions for "
                f"VariableSelectionNetwork. Got shape {x.shape}."
            )
            
            # TODO: We used for static metadata ,the Timedistributed for embedding 
            # which work perfectly. The next work, should be to remove the 
            # TimeDistributed... Below a bit trick to fix this issue .. 
            
            # If shape is (batch_size, embed_dim), we assume user has flattened
            # static features into a single dimension.
            # We can interpret that as a single "feature" => num_inputs=1
            # Alternatively, if shape is (batch_size, num_inputs), we can 
            # interpret embed_dim=1 (rare usage).
            # We'll attempt a best guess approach by checking 
            # if num_inputs=1 or embed_dim=1.

            bsz, dim2 = tf_shape(x)[0], tf_shape(x)[1]

            # If user has declared num_inputs=1, treat dim2 as the embedding dimension
            # => reshape x to (batch_size, num_inputs, embed_dim) => (bsz, 1, dim2)
            # If user has declared embed_dim=1, treat dim2 as the number of inputs
            # => reshape x to (batch_size, num_inputs, 1) => (bsz, dim2, 1)

            # We'll attempt a logic that if num_inputs == dim2, then embed_dim=1
            # else if num_inputs == 1, then embed_dim=dim2
            # else we can't fix automatically.

            num_inputs = self.num_inputs  # from constructor
            # Convert dims to actual python ints for logic
            dim2_py = tf_get_static_value(dim2)

            # If we can't get a static value from dim2, we forcibly 
            # raise an error or attempt dynamic approach
            if dim2_py is None:
                raise ValueError(
                    "VariableSelectionNetwork received a 2D tensor with"
                    " an unknown dimension. Cannot automatically reshape."
                    " Provide explicit shape info or embed your static"
                    " inputs to 3D."
                )

            if num_inputs == dim2_py:
                # shape => (batch_size, num_inputs)
                # interpret embed_dim=1
                x_reshaped = tf_reshape(x, (bsz, num_inputs, 1))
                # Now shape => (batch_size, num_inputs, 1)
                # Reuse the 3D path
                return self.call(x_reshaped, training=training)
            elif num_inputs == 1:
                # shape => (batch_size, embed_dim)
                # interpret that as only 1 feature => embed_dim=dim2
                x_reshaped = tf_reshape(x, (bsz, 1, dim2_py))
                # Now shape => (batch_size, 1, embed_dim)
                return self.call(x_reshaped, training=training)
            else:
                raise ValueError(
                    f"VariableSelectionNetwork got a 2D input of shape"
                    f" (batch_size, {dim2_py}), but num_inputs={num_inputs}."
                    " Provide consistent shapes or embed the inputs to 3D."
                )
       

# if __name__=='__main__': 
#     import tensorflow as tf 
#     import numpy as np
    
#     # Define model parameters
#     batch_size       = 2
#     past_steps       = 4
#     forecast_horizon  = 3
#     dynamic_input_dim = 5      # Number of dynamic (past) input features
#     static_input_dim  = 6      # Number of static (metadata) input features
#     future_input_dim  = 4      # Number of future (covariate) input features
#     hidden_units      = 16     # Embedding and hidden layer size
#     num_heads         = 4       # Number of attention heads
#     dropout_rate      = 0.1     # Dropout rate
#     quantiles         = [0.1, 0.5, 0.9]  # Quantiles for probabilistic forecasting
#     activation        = 'elu'   # Activation function for GRNs
#     use_batch_norm    = False   # Whether to use Batch Normalization in GRNs
#     lstm_units        = None    # Number of units in LSTM layers (defaults to hidden_units)
#     output_dim        = 1       # Number of output variables
    
#     # Instantiate the model
#     model = TFT(
#         dynamic_input_dim=dynamic_input_dim,
#         hidden_units=hidden_units,
#         num_heads=num_heads,
#         dropout_rate=dropout_rate,
#         forecast_horizon=forecast_horizon,
#         quantiles=quantiles,          # Set to None for deterministic predictions
#         activation=activation,
#         use_batch_norm=use_batch_norm,
#         lstm_units=lstm_units,
#         output_dim=output_dim,
#         static_input_dim=static_input_dim,
#         future_input_dim=future_input_dim
#     )
    
#     # Compile the model with Mean Squared Error loss for deterministic predictions
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(
#         optimizer=optimizer,
#         loss='mse'
#     )
    
#     # Generate dummy data
#     num_samples = 100
    
#     # Past inputs: (batch_size, past_steps, dynamic_input_dim)
#     past_in = np.random.randn(num_samples, past_steps, dynamic_input_dim).astype(np.float32)
    
#     # Future inputs: (batch_size, forecast_horizon, future_input_dim)
#     future_in = np.random.randn(num_samples, forecast_horizon, future_input_dim).astype(np.float32)
    
#     # Static inputs: (batch_size, static_input_dim)
#     static_in = np.random.randn(num_samples, static_input_dim).astype(np.float32)
    
#     # Targets: (batch_size, forecast_horizon, output_dim)
#     y_true = np.random.randn(num_samples, forecast_horizon, output_dim).astype(np.float32)
    
#     # Fit the model
#     history = model.fit(
#         x=(past_in, future_in, static_in),
#         y=y_true,
#         epochs=5,
#         batch_size=16
#     )
    
#     # Display training history
#     print(history.history)
    
#     # Predict using the model
#     y_pred = model.predict((past_in, future_in, static_in))
#     print(y_pred.shape)  # Expected: (batch_size, forecast_horizon, num_quantiles, output_dim) or (batch_size, forecast_horizon, output_dim)
