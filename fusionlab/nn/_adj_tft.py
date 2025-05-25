# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Temporal Fusion Transformer (TFT), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""
from numbers import Real, Integral  
from typing import List, Optional, Union 

from .._fusionlog import fusionlog 
from ..api.property import NNLearner 
from ..core.checks import is_iterable
from ..core.diagnose_q import validate_quantiles
from ..compat.sklearn import validate_params, Interval, StrOptions
from ..utils.deps_utils import ensure_pkg 
from ..utils.validator import validate_positive_integer

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
    tf_rank = KERAS_DEPS.rank
    tf_squeeze= KERAS_DEPS.squeeze 
    tf_concat =KERAS_DEPS.concat
    tf_shape =KERAS_DEPS.shape
    tf_zeros=KERAS_DEPS.zeros
    tf_float32=KERAS_DEPS.float32
    tf_reshape=KERAS_DEPS.reshape
    tf_autograph=KERAS_DEPS.autograph
    tf_multiply=KERAS_DEPS.multiply
    tf_reduce_mean = KERAS_DEPS.reduce_mean
    tf_get_static_value=KERAS_DEPS.get_static_value
    tf_gather=KERAS_DEPS.gather 
    
    from ._tensor_validation import ( 
        validate_model_inputs, combine_temporal_inputs_for_lstm
        )
    from .losses import combined_quantile_loss 
    from .components import (
        VariableSelectionNetwork,
        PositionalEncoding,
        GatedResidualNetwork,
        TemporalAttentionLayer, 
    )

    
DEP_MSG = dependency_message('transformers.tft') 
logger = fusionlog().get_fusionlab_logger(__name__) 

__all__= ['TFT']

# ------------------------ TFT implementation --------------------------------

@register_keras_serializable('fusionlab.nn.transformers', name="TFT")
class TFT(Model, NNLearner): 
    """Temporal Fusion Transformer (TFT) requiring static, dynamic(past), and future inputs. """

    @validate_params({
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "static_input_dim": [Interval(Integral, 0, None, closed='left')],
        "future_input_dim": [Interval(Integral, 0, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "recurrent_dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', None],
        "activation": [StrOptions(
            {"elu", "relu", "tanh", "sigmoid", "linear", "gelu"})],
        "use_batch_norm": [bool],
        "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": ['array-like', Interval(Integral, 1, None, closed='left'), None],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        future_input_dim: int,
        hidden_units: int = 32,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        recurrent_dropout_rate: float = 0.0,
        forecast_horizon: int = 1,
        quantiles: Optional[List[float]] = None,
        activation: str = 'elu',
        use_batch_norm: bool = False,
        num_lstm_layers: int = 1,
        lstm_units: Optional[Union[int, List[int]]] = None,
        output_dim: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.dynamic_input_dim = dynamic_input_dim
        self.static_input_dim = static_input_dim
        self.future_input_dim = future_input_dim
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.forecast_horizon = forecast_horizon
        self.activation = activation 
        self.use_batch_norm = use_batch_norm
        self.num_lstm_layers = num_lstm_layers
        self.output_dim = output_dim
        self.quantiles = validate_quantiles(
            quantiles) if quantiles else None
        self.num_quantiles = len(
            self.quantiles) if self.quantiles else 1
        self._lstm_units = lstm_units 
        
        # Process LSTM units list
        _lstm_units_resolved = lstm_units or hidden_units
        self.lstm_units_list = (
             [_lstm_units_resolved] * num_lstm_layers
             if isinstance(_lstm_units_resolved, int)
             else is_iterable(
                 _lstm_units_resolved, 
                 exclude_string=True, 
                 transform=True
                 )
         )
        self.lstm_units_list = [
            validate_positive_integer(v, "LSTM units")
            for v in self.lstm_units_list 
        ]
        if len(self.lstm_units_list) != num_lstm_layers:
             raise ValueError(
                 "'lstm_units' length must match 'num_lstm_layers'.")

        # --- Initialize Core TFT Components ---
        # 1. Variable Selection Networks
        self.static_vsn = VariableSelectionNetwork(
            num_inputs=self.static_input_dim, 
            units=self.hidden_units,
            dropout_rate=self.dropout_rate, 
            activation=self.activation,
            use_batch_norm=self.use_batch_norm, 
            name="static_vsn"
        )
        self.dynamic_vsn = VariableSelectionNetwork(
            num_inputs=self.dynamic_input_dim, 
            units=self.hidden_units,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            activation=self.activation, 
            use_batch_norm=self.use_batch_norm,
            name="dynamic_vsn"
        )
        self.future_vsn = VariableSelectionNetwork(
            num_inputs=self.future_input_dim, 
            units=self.hidden_units,
            dropout_rate=self.dropout_rate, 
            use_time_distributed=True,
            activation=self.activation, 
            use_batch_norm=self.use_batch_norm,
            name="future_vsn"
        )
        
        # 2. Static Context GRNs
        self.static_grn_for_vsns = GatedResidualNetwork(
            units=self.hidden_units, 
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            name="static_grn_for_vsns"
        )
        self.static_grn_for_enrichment = GatedResidualNetwork(
            units=self.hidden_units, 
            dropout_rate=self.dropout_rate,
            activation=self.activation, 
            use_batch_norm=self.use_batch_norm,
            name="static_grn_for_enrichment"
        )
        self.static_grn_for_state_h = GatedResidualNetwork(
            units=self.lstm_units_list[0], 
            dropout_rate=self.dropout_rate,
            activation=self.activation, 
            use_batch_norm=self.use_batch_norm,
            name="static_grn_for_state_h"
        )
        self.static_grn_for_state_c = GatedResidualNetwork(
            units=self.lstm_units_list[0], 
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            name="static_grn_for_state_c"
        )
        
        # 3. LSTM Encoder Layers
        self.lstm_layers = [
            LSTM(
                units=units, return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout_rate,
                name=f'encoder_lstm_{i+1}'
            ) for i, units in enumerate(self.lstm_units_list)
        ]
        # 4. Static Enrichment GRN
        self.static_enrichment_grn = GatedResidualNetwork(
             units=self.hidden_units, 
             dropout_rate=self.dropout_rate,
             activation=self.activation, 
             use_batch_norm=self.use_batch_norm,
             name="static_enrichment_grn"
        )
        # 5. Temporal Self-Attention Layer
        self.temporal_attention_layer = TemporalAttentionLayer(
            units=self.hidden_units, 
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate, 
            activation=self.activation,
            use_batch_norm=self.use_batch_norm, 
            name="temporal_self_attention"
        )
        
        # 6. Position-wise Feedforward GRN
        self.positionwise_grn = GatedResidualNetwork(
            units=self.hidden_units, 
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm, 
            name="pos_wise_ff_grn"
        )
        
        # 7. Output Layer(s)
        if self.quantiles:
            self.output_layers = [
                TimeDistributed(
                    Dense(self.output_dim), name=f'q_{int(q*100)}_td'
                    )
                for q in self.quantiles
            ]
        else:
            self.output_layer = TimeDistributed(
                Dense(self.output_dim), name='point_td'
            )
        # 8. Positional Encoding Layer
        self.positional_encoding = PositionalEncoding(name="pos_enc")


    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        """Forward pass for the revised TFT with numerical inputs."""
        logger.debug(f"TFT '{self.name}': Entering call method.")
        logger.debug(f"  Received {len(inputs)} inputs.")

        # --- Input Validation and Reordering ---
        # User provides [static, dynamic, future]
        # Validator expects [dynamic, future, static]
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError(
                "TFT expects inputs as list/tuple of 3 elements: "
                "[static_inputs, dynamic_inputs, future_inputs]."
            )
        static_inputs_user, dynamic_inputs_user, future_inputs_user = inputs
        logger.debug(
            f"  User inputs shapes: Static={static_inputs_user.shape}, "
            f"Dynamic={dynamic_inputs_user.shape}, "
            f"Future={future_inputs_user.shape}"
            )

        # Reorder for internal validation function
        validator_input_order = [
             static_inputs_user, dynamic_inputs_user, future_inputs_user 
             ]
        # Call validator: returns (dynamic, future, static) tensors
        # Performs type checks, float32 conversion, dimension checks.
        static_inputs, dynamic_inputs, future_inputs = validate_model_inputs(
             validator_input_order,
             dynamic_input_dim=self.dynamic_input_dim,
             static_input_dim=self.static_input_dim,
             future_covariate_dim=self.future_input_dim,
         )
        logger.debug(
            "  Inputs validated and assigned internally."
            f" Shapes: Dyn={dynamic_inputs.shape},"
            f" Fut={future_inputs.shape}, Stat={static_inputs.shape}"
            )

        # --- Static Pathway ---
        logger.debug("  Processing Static Pathway...")
        # 1a. Reshape Static Input for VSN if needed (B, N) -> (B, N, 1)
        # why use static_inputs.shape.rank rather than tf_rank (static_inputs)? 
        # to avoid issue of unknow rank for autograph conversion 
        # when it's removed.
        if static_inputs.shape.rank == 2:
            static_inputs_r = tf_expand_dims(static_inputs, axis=-1)
            logger.debug(
                "    Expanded static input rank to 3:"
                f" {static_inputs_r.shape}")
        else:
            static_inputs_r = static_inputs # already (B, N, F)

        # 1b. Static VSN
        # Processes static features, potentially
        # using context if VSN modified
        static_selected = self.static_vsn(
            static_inputs_r,
            training=training,
            # context=None # Context for static VSN usually not needed
            )
        # Output shape: (B, hidden_units)
        logger.debug(
            f"    Static VSN output shape: {static_selected.shape}")

        # 1c. Static Context Vector Generation using GRNs
        # Context for conditioning VSNs (passed if VSNs accept context)
        context_for_vsns = self.static_grn_for_vsns(
            static_selected, training=training)
        # Context for enriching temporal features after LSTM
        context_for_enrichment = self.static_grn_for_enrichment(
            static_selected, training=training)
        # Contexts for initializing LSTM states
        context_state_h = self.static_grn_for_state_h(
            static_selected, training=training)
        context_state_c = self.static_grn_for_state_c(
            static_selected, training=training)
        initial_state = [context_state_h, context_state_c]
        logger.debug(
            f"    Generated static contexts:"
            f" VSN={context_for_vsns.shape},"
            f" Enrich={context_for_enrichment.shape},"
            f" StateH={context_state_h.shape},"
            f" StateC={context_state_c.shape}"
            )

        # --- Temporal Pathway ---
        logger.debug("  Processing Temporal Pathway...")
        # 3a. Reshape Dynamic/Future Inputs for VSNs if needed
        if dynamic_inputs.shape.rank == 3:
             dynamic_inputs_r = tf_expand_dims(dynamic_inputs, axis=-1)
        else: 
            dynamic_inputs_r = dynamic_inputs # Assume (B, T, N, F)
        if future_inputs.shape.rank == 3:
             future_inputs_r = tf_expand_dims(future_inputs, axis=-1)
        else: 
            future_inputs_r = future_inputs # Assume (B, T_fut, N, F)
        logger.debug(
            f"    Temporal input shapes for VSN: Dyn={dynamic_inputs_r.shape},"
            f" Fut={future_inputs_r.shape}"
            )

        # 3b. Dynamic/Future VSNs
        # Pass static context derived earlier
        dynamic_selected = self.dynamic_vsn(
             dynamic_inputs_r, training=training, 
             context=context_for_vsns)
        future_selected = self.future_vsn(
             future_inputs_r, training=training, context=context_for_vsns)
        # Shapes: (B, T_past, H_units), (B, T_future_total, H_units)
        logger.debug(
            f"    Temporal VSN outputs shapes: Dyn={dynamic_selected.shape},"
            f" Fut={future_selected.shape}"
            )

        # 4. Combine Features for LSTM Input using helper
        # Handles slicing future_selected to match T_past and concatenates
        logger.debug(
            "  Combining dynamic and future features for LSTM...")
        temporal_features = combine_temporal_inputs_for_lstm(
            dynamic_selected, future_selected, 
            mode='soft' # Use soft? or strict?
            )
        # Shape: (B, T_past, combined_features = D_dyn_emb + D_fut_emb)
        # Assuming VSN outputs hidden_units: (B, T_past, 2 * hidden_units)
        logger.debug(
            f"    Combined temporal features shape:"
            f" {temporal_features.shape}")

        # 5. Positional Encoding
        temporal_features_pos = self.positional_encoding(
            temporal_features
            )
        logger.debug("    Applied positional encoding.")

        # 6. LSTM Encoder
        logger.debug("  Running LSTM encoder...")
        lstm_output = temporal_features_pos
        current_state = initial_state
        for i, layer in enumerate(self.lstm_layers):
             layer_input_shape = lstm_output.shape
             if i == 0:
                 lstm_output = layer(
                     lstm_output, initial_state=current_state,
                     training=training
                     )
             else:
                 lstm_output = layer(lstm_output, training=training)
             logger.debug(
                 f"    LSTM layer {i+1} output shape: {layer_input_shape}")
        # Final LSTM output shape: (B, T_past, lstm_units)

        # 7. Static Enrichment
        logger.debug("  Applying static enrichment...")
        enriched_output = self.static_enrichment_grn(
            lstm_output, context=context_for_enrichment, 
            training=training
        )
        # Shape: (B, T_past, hidden_units)
        logger.debug(
            f"    Enriched output shape: {enriched_output.shape}")

        # 8. Temporal Self-Attention
        logger.debug("  Applying temporal attention...")
        attention_output = self.temporal_attention_layer(
            enriched_output, 
            context_vector=context_for_vsns, 
            training=training
        )
        # Shape: (B, T_past, hidden_units)
        logger.debug(
            f"    Attention output shape: {attention_output.shape}")

        # 9. Position-wise Feedforward
        logger.debug("  Applying position-wise feedforward...")
        final_temporal_repr = self.positionwise_grn(
            attention_output, training=training
        )
        # Shape: (B, T_past, hidden_units)
        logger.debug(
            "    Final temporal representation shape:"
            f" {final_temporal_repr.shape}")

        # --- 10. Output Slice and Projection ---
        logger.debug("  Generating final predictions...")
        # Slice features corresponding to the forecast horizon
        output_features_sliced = final_temporal_repr[
            :, -self.forecast_horizon:, :]
        logger.debug(
            "    Sliced features for output shape:"
            f" {output_features_sliced.shape}")
        # Shape: (B, H, hidden_units)

        # Apply the final TimeDistributed output layer(s)
        if self.quantiles:
            quantile_outputs = []
            if not hasattr(self, 'output_layers'):
                 raise AttributeError(
                     "Quantile output layers not initialized."
                     )
            for i, layer in enumerate(self.output_layers):
                out_i = layer(output_features_sliced, training=training)
                quantile_outputs.append(out_i)
                logger.debug(
                    f"      Quantile output {i} shape: {out_i.shape}")

            outputs = tf_stack(quantile_outputs, axis=2) # (B, H, Q, O)
            logger.debug(
                f"      Stacked quantile output shape: {outputs.shape}")
            if self.output_dim == 1:
                outputs = tf_squeeze(outputs, axis=-1) # (B, H, Q)
                logger.debug(
                    "      Squeezed final dimension (output_dim=1).")
        else:
            # Point Forecast
            if not hasattr(self, 'output_layer'):
                 raise AttributeError("Point output layer not initialized.")
            outputs = self.output_layer(
                output_features_sliced, training=training
                )
            # Shape (B, H, O)

        logger.debug(
            f"TFT '{self.name}': Final output shape: {outputs.shape}")
        logger.debug(
            f"TFT '{self.name}': Exiting call method.")
        return outputs
    

    def compile(self, optimizer, loss=None, **kwargs):
        if self.quantiles is None:
            effective_loss = loss or 'mean_squared_error'
        else:
            effective_loss = loss or combined_quantile_loss(
                self.quantiles)
        super().compile(
            optimizer=optimizer, 
            loss=effective_loss, 
            **kwargs
        )
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'dynamic_input_dim': self.dynamic_input_dim,
            'static_input_dim': self.static_input_dim,
            'future_input_dim': self.future_input_dim,
            'hidden_units': self.hidden_units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout_rate': self.recurrent_dropout_rate,
            'forecast_horizon': self.forecast_horizon,
            'quantiles': self.quantiles,
            'activation': self.activation, 
            'use_batch_norm': self.use_batch_norm,
            'num_lstm_layers': self.num_lstm_layers,
            'lstm_units': self._lstm_units, 
            'output_dim': self.output_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
TFT.__doc__+=r"""\
This class implements the Temporal Fusion Transformer (TFT)
architecture, closely following the structure described in the
original paper [Lim21]_. It is designed for multi-horizon time
series forecasting and explicitly requires static covariates,
dynamic (historical) covariates, and known future covariates as
inputs.

Compared to more flexible implementations, this version mandates
all input types, simplifying the internal input handling logic. It
incorporates key TFT components like Variable Selection Networks
(VSNs), Gated Residual Networks (GRNs) for static context generation
and feature processing, LSTM encoding, static enrichment, interpretable
multi-head attention, and position-wise feedforward layers.

Parameters
----------
dynamic_input_dim : int
    The total number of features present in the dynamic (past)
    input tensor. These are features that vary across the lookback
    time steps.
static_input_dim : int
    The total number of features present in the static
    (time-invariant) input tensor. These features provide context
    that does not change over time for a given series.
future_input_dim : int
    The total number of features present in the known future input
    tensor. These features provide information about future events
    or conditions known at the time of prediction.
hidden_units : int, default=32
    The main dimensionality of the hidden layers used throughout
    the network, including VSN outputs, GRN hidden states,
    enrichment layers, and attention mechanisms.
num_heads : int, default=4
    Number of attention heads used in the
    :class:`~fusionlab.nn.components.TemporalAttentionLayer`. More heads
    allow attending to different representation subspaces.
dropout_rate : float, default=0.1
    Dropout rate applied to non-recurrent connections in LSTMs,
    VSNs, GRNs, and Attention layers. Value between 0 and 1.
recurrent_dropout_rate : float, default=0.0
    Dropout rate applied specifically to the recurrent connections
    within the LSTM layers. Helps regularize the recurrent state
    updates. Value between 0 and 1. Note: May impact performance
    on GPUs.
forecast_horizon : int, default=1
    The number of future time steps the model is trained to predict
    simultaneously (multi-horizon forecasting).
quantiles : list[float], optional, default=None
    A list of quantiles (e.g., ``[0.1, 0.5, 0.9]``) for probabilistic
    forecasting.
    * If provided, the model outputs predictions for each specified
        quantile, and the :func:`~fusionlab.nn.losses.combined_quantile_loss`
        should typically be used for training.
    * If ``None``, the model performs point forecasting (outputting a
        single value per step), typically trained with MSE loss.
activation : str, default='elu'
    Activation function used within GRNs and potentially other dense
    layers. Supported options include 'elu', 'relu', 'gelu', 'tanh',
    'sigmoid', 'linear'.
use_batch_norm : bool, default=False
    If True, applies Batch Normalization within the Gated Residual
    Networks (GRNs). Layer Normalization is typically used by default
    within GRN implementations as per the TFT paper.
num_lstm_layers : int, default=1
    Number of LSTM layers stacked in the sequence encoder module.
lstm_units : int or list[int], optional, default=None
    Number of hidden units in each LSTM encoder layer.
    * If ``int``: The same number of units is used for all layers
        specified by `num_lstm_layers`.
    * If ``list[int]``: Specifies the number of units for each LSTM
        layer sequentially. The length must match `num_lstm_layers`.
    * If ``None``: Defaults to using `hidden_units` for all LSTM layers.
output_dim : int, default=1
    The number of target variables the model predicts at each time
    step. Typically 1 for univariate forecasting.
**kwargs
    Additional keyword arguments passed to the parent Keras `Model`.

Notes
-----
This implementation requires inputs to the `call` method as a list
or tuple containing exactly three tensors in the order:
``[static_inputs, dynamic_inputs, future_inputs]``. The shapes
should be:
* `static_inputs`: `(Batch, StaticFeatures)`
* `dynamic_inputs`: `(Batch, PastTimeSteps, DynamicFeatures)`
* `future_inputs`: `(Batch, TotalTimeSteps, FutureFeatures)` where
    `TotalTimeSteps` includes past and future steps relevant for
    the LSTM processing.

**Use Case and Importance**

This revised `TFT` class provides a structured implementation that
closely follows the component architecture described in the original
TFT paper, including the distinct GRNs for generating static context
vectors. By requiring all input types (static, dynamic past, known
future), it simplifies the input handling logic compared to versions
allowing optional inputs. This makes it a suitable choice when you
have all three types of features available and want a robust baseline
TFT implementation that explicitly leverages static context for VSNs,
LSTM initialization, and temporal processing enrichment. It serves as
a strong foundation for complex multi-horizon forecasting tasks that
benefit from diverse data integration and interpretability components
like VSNs and attention.

**Mathematical Formulation**

The model processes inputs through the following key stages:

1.  **Variable Selection:** Separate Variable Selection Networks (VSNs)
    are applied to the static ($\mathbf{s}$), dynamic past
    ($\mathbf{x}_t, t \le T$), and known future ($\mathbf{z}_t, t > T$)
    inputs. This step identifies relevant features within each input type
    and transforms them into embeddings of dimension `hidden_units`. Let
    the outputs be $\zeta$ (static embedding), $\xi^{dyn}_t$ (dynamic),
    and $\xi^{fut}_t$ (future). VSNs may be conditioned by a static
    context vector $c_s$.

    .. math::
       \zeta &= \text{VSN}_{static}(\mathbf{s}, [c_s]) \\
       \xi^{dyn}_t &= \text{VSN}_{dyn}(\mathbf{x}_t, [c_s]) \\
       \xi^{fut}_t &= \text{VSN}_{fut}(\mathbf{z}_t, [c_s])

2.  **Static Context Generation:** Four distinct Gated Residual
    Networks (GRNs) process the static embedding $\zeta$ to produce
    context vectors: $c_s$ (for VSNs), $c_e$ (for enrichment),
    $c_h$ (LSTM initial hidden state), $c_c$ (LSTM initial cell state).

    .. math::
       c_s = GRN_{vs}(\zeta) \quad ... \quad c_c = GRN_{c}(\zeta)

3.  **Temporal Processing Input:** The selected dynamic and future
    embeddings are potentially combined (e.g., concatenated along time
    or features, depending on preprocessing) and augmented with
    positional encoding to form the input sequence for the LSTM.
    Let this sequence be $\psi_t$.

    .. math::
       \psi_t = \text{Combine}(\xi^{dyn}_t, \xi^{fut}_t) + \text{PosEncode}(t)

4.  **LSTM Encoder:** A stack of `num_lstm_layers` LSTMs processes
    $\psi_t$, initialized with $[c_h, c_c]$.

    .. math::
       \{h_t\} = \text{LSTMStack}(\{\psi_t\}, \text{initial_state}=[c_h, c_c])

5.  **Static Enrichment:** The LSTM outputs $h_t$ are combined with the
    static enrichment context $c_e$ using a time-distributed GRN.

    .. math::
       \phi_t = GRN_{enrich}(h_t, c_e)

6.  **Temporal Self-Attention:** Interpretable Multi-Head Attention is
    applied to the enriched sequence $\{\phi_t\}$, potentially using
    $c_s$ as context within the attention mechanism's internal GRNs.
    This results in context vectors $\beta_t$ after residual connection,
    gating (GLU), and normalization.

    .. math::
       \beta_t = \text{TemporalAttention}(\{\phi_t\}, c_s)

7.  **Position-wise Feed-Forward:** A final time-distributed GRN is
    applied to the attention output.

    .. math::
       \delta_t = GRN_{final}(\beta_t)

8.  **Output Projection:** The features corresponding to the forecast
    horizon ($t > T$) are selected from $\{\delta_t\}$ and passed through
    a final Dense layer (or multiple layers for quantiles) to produce
    the predictions $\hat{y}_{t+1}, ..., \hat{y}_{t+\tau}$.

Methods
-------
call(inputs, training=False)
    Performs the forward pass. Expects `inputs` as a list/tuple:
    `[static_inputs, dynamic_inputs, future_inputs]`.
compile(optimizer, loss=None, **kwargs)
    Compiles the model, automatically selecting 'mse' or quantile
    loss based on `quantiles` initialization if `loss` is not given.

Examples
--------
>>> import numpy as np
>>> import tensorflow as tf
>>> from fusionlab.nn.transformers import TFT
>>> from fusionlab.nn.losses import combined_quantile_loss
>>>
>>> # Dummy Data Dimensions
>>> B, T_past, H = 4, 12, 6 # Batch, Lookback, Horizon
>>> D_dyn, D_stat, D_fut = 5, 3, 2
>>> T_future = H # Assume future inputs cover horizon only for LSTM input
>>>
>>> # Create Dummy Input Tensors (Ensure correct shapes and types)
>>> static_in = tf.random.normal((B, D_stat), dtype=tf.float32)
>>> dynamic_in = tf.random.normal((B, T_past, D_dyn), dtype=tf.float32)
>>> # Future input needs shape (B, T_past + T_future, D_fut) for VSN
>>> # or (B, T_future, D_fut) if handled differently before LSTM concat.
>>> # Let's assume preprocessed to match horizon T_future for simplicity here
>>> future_in = tf.random.normal((B, T_future, D_fut), dtype=tf.float32)
>>>
>>> # Instantiate Model for Quantile Forecasting
>>> model = TFT(
...     dynamic_input_dim=D_dyn, static_input_dim=D_stat,
...     future_input_dim=D_fut, forecast_horizon=H,
...     hidden_units=16, num_heads=2, num_lstm_layers=1,
...     quantiles=[0.1, 0.5, 0.9], output_dim=1
... )
>>>
>>> # Compile with appropriate loss
>>> loss_fn = combined_quantile_loss([0.1, 0.5, 0.9])
>>> model.compile(optimizer='adam', loss=loss_fn)
>>>
>>> # Prepare input list in correct order: [static, dynamic, future]
>>> model_inputs = [static_in, dynamic_in, future_in]
>>>
>>> # Make a prediction (forward pass)
>>> # Note: Need to build the model first, e.g., by calling it once
>>> # or specifying input_shape in build method if using subclassing.
>>> # Alternatively, fit for one step. For direct call:
>>> # output_shape = model.compute_output_shape(
>>> #    [t.shape for t in model_inputs]) # Requires TF >= 2.8 approx
>>> # For simplicity, assume model builds on first call
>>> predictions = model(model_inputs, training=False)
>>> print(f"Output shape: {predictions.shape}")
Output shape: (4, 6, 3)

See Also
--------
fusionlab.nn.components.VariableSelectionNetwork : Core component for VSN.
fusionlab.nn.components.GatedResidualNetwork : Core component for GRN.
fusionlab.nn.components.TemporalAttentionLayer : Core attention block.
tensorflow.keras.layers.LSTM : Recurrent layer used internally.
fusionlab.nn.losses.combined_quantile_loss : Default loss for quantiles.
fusionlab.nn.utils.reshape_xtft_data : Utility to prepare inputs.
fusionlab.nn.XTFT : More advanced related architecture.
tensorflow.keras.Model : Base Keras model class.

References
----------
.. [Lim21] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
   Temporal fusion transformers for interpretable multi-horizon
   time series forecasting. *International Journal of Forecasting*,
   37(4), 1748-1764.
"""

