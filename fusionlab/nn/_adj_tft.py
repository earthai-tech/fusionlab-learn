# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Temporal Fusion Transformer (TFT), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""
from textwrap import dedent 
from numbers import Real, Integral  
from typing import List, Optional, Union, Dict, Tuple

from .._fusionlog import fusionlog 
from ..api.property import NNLearner 
from ..core.checks import is_iterable
from ..core.diagnose_q import validate_quantiles
from ..compat.sklearn import validate_params, Interval, StrOptions
from ..decorators import Appender 
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
        validate_tft_inputs, combine_temporal_inputs_for_lstm
        )
    from .losses import combined_quantile_loss 

    from .components import (
        VariableSelectionNetwork,
        PositionalEncoding,
        GatedResidualNetwork,
        TemporalAttentionLayer, 
        CategoricalEmbeddingProcessor 
    )

    
DEP_MSG = dependency_message('transformers.tft') 
logger = fusionlog().get_fusionlab_logger(__name__) 

__all__= ['TFT']

# ------------------------ TFT implementation --------------------------------

@register_keras_serializable('fusionlab.nn.transformers', name="TFT")
class TFT(Model, NNLearner): 

    @validate_params({
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "static_input_dim": [Interval(Integral, 1, None, closed='left')],
        "future_input_dim": [Interval(Integral, 1, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "recurrent_dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', None],
        "activation": [StrOptions(
            {"elu", "relu", "tanh", "sigmoid", "linear", "gelu"}
            )],
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
        
        # --- Store parameters ---
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
             dynamic_inputs_user, future_inputs_user, static_inputs_user
             ]
        # Call validator: returns (dynamic, future, static) tensors
        # Performs type checks, float32 conversion, dimension checks.
        dynamic_inputs, future_inputs, static_inputs = validate_tft_inputs(
             validator_input_order,
             dynamic_input_dim=self.dynamic_input_dim,
             static_input_dim=self.static_input_dim,
             future_covariate_dim=self.future_input_dim,
             error='raise'
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
   
TFT.__doc__=r"""\
    Temporal Fusion Transformer (TFT) requiring static,
dynamic (past), and future inputs.

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

# XXX TODO: 
@register_keras_serializable('fusionlab.nn.transformers', name="_TFTPlus")
class _TFTPlus(Model):
    """Revised Temporal Fusion Transformer (TFT) requiring static,
    dynamic (past), and future inputs, with categorical handling.
    """
    @validate_params({
        # --- Existing Validations ---
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "static_input_dim": [Interval(Integral, 1, None, closed='left')],
        "future_input_dim": [Interval(Integral, 1, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "recurrent_dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', None],
        "activation": [StrOptions(
            {"elu", "relu", "tanh", "sigmoid", "linear", "gelu"}
            )],
        "use_batch_norm": [bool],
        "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": ['array-like', Interval(Integral, 1, None, closed='left'), None],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
        # --- New Validations for Categorical Handling ---
        "static_categorical_info": [dict, None],
        "dynamic_categorical_info": [dict, None],
        "future_categorical_info": [dict, None],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        future_input_dim: int,
        # --- New parameters for categorical features ---
        static_categorical_info: Optional[Dict[int, Tuple[int, int]]] = None,
        dynamic_categorical_info: Optional[Dict[int, Tuple[int, int]]] = None,
        future_categorical_info: Optional[Dict[int, Tuple[int, int]]] = None,
        # --- Existing parameters ---
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
        # Store all parameters
        self.dynamic_input_dim = dynamic_input_dim
        self.static_input_dim = static_input_dim
        self.future_input_dim = future_input_dim
        self.static_categorical_info = static_categorical_info or {}
        self.dynamic_categorical_info = dynamic_categorical_info or {}
        self.future_categorical_info = future_categorical_info or {}
        # ... (store other existing params like hidden_units, num_heads, etc.) ...
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.forecast_horizon = forecast_horizon
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.num_lstm_layers = num_lstm_layers
        self.output_dim = output_dim
        self.quantiles = validate_quantiles(quantiles) if quantiles else None
        self.num_quantiles = len(self.quantiles) if self.quantiles else 1
        # Process LSTM units
        _lstm_units = lstm_units or hidden_units
        self._lstm_units = lstm_units 
        self.lstm_units_list = ([_lstm_units] * num_lstm_layers
                                if isinstance(_lstm_units, int) else _lstm_units)
        if len(self.lstm_units_list) != num_lstm_layers:
            raise ValueError("LSTM units list length mismatch.")

        # --- Calculate number of continuous features ---
        self.static_cont_dim = self.static_input_dim - len(
            self.static_categorical_info)
        self.dynamic_cont_dim = self.dynamic_input_dim - len(
            self.dynamic_categorical_info)
        self.future_cont_dim = self.future_input_dim - len(
            self.future_categorical_info)

        # Ensure non-negative dimensions
        if any(d < 0 for d in [self.static_cont_dim, self.dynamic_cont_dim,
                               self.future_cont_dim]):
             raise ValueError("Number of categorical features cannot exceed"
                              " total features for an input type.")

        # --- Initialize Components ---

        # 1. Categorical Embeddings (if any)
        self.static_cat_processor = CategoricalEmbeddingProcessor(
            self.static_categorical_info) if self.static_categorical_info else None
        self.dynamic_cat_processor = CategoricalEmbeddingProcessor(
            self.dynamic_categorical_info) if self.dynamic_categorical_info else None
        self.future_cat_processor = CategoricalEmbeddingProcessor(
            self.future_categorical_info) if self.future_categorical_info else None

        # 2. Linear projections for continuous features (to hidden_units)
        # Applied before VSN to have all inputs at same dimension
        if self.static_cont_dim > 0:
             self.static_cont_linear = Dense( # Not TimeDistributed
                 self.hidden_units, name="static_cont_linear")
        else: self.static_cont_linear = None

        if self.dynamic_cont_dim > 0:
            self.dynamic_cont_linear = TimeDistributed(
                Dense(self.hidden_units, name="dynamic_cont_linear"),
                name="dynamic_cont_linear_td")
        else: self.dynamic_cont_linear = None

        if self.future_cont_dim > 0:
            self.future_cont_linear = TimeDistributed(
                Dense(self.hidden_units, name="future_cont_linear"),
                name="future_cont_linear_td")
        else: self.future_cont_linear = None

        # 3. Variable Selection Networks
        # VSNs now operate on the processed features (all potentially dim=hidden_units)
        # num_inputs for VSN is the TOTAL number of original features (cat + cont)
        self.static_vsn = VariableSelectionNetwork(
            num_inputs=self.static_input_dim, units=self.hidden_units,
            dropout_rate=self.dropout_rate, activation=self.activation,
            use_batch_norm=self.use_batch_norm, name="static_vsn")
        self.dynamic_vsn = VariableSelectionNetwork(
            num_inputs=self.dynamic_input_dim, units=self.hidden_units,
            dropout_rate=self.dropout_rate, use_time_distributed=True,
            activation=self.activation, use_batch_norm=self.use_batch_norm,
            name="dynamic_vsn")
        self.future_vsn = VariableSelectionNetwork(
            num_inputs=self.future_input_dim, units=self.hidden_units,
            dropout_rate=self.dropout_rate, use_time_distributed=True,
            activation=self.activation, use_batch_norm=self.use_batch_norm,
            name="future_vsn")

        # 4. Static Context GRNs (process static VSN output)

        # 2. Static Context GRNs (processing VSN output)
        # Context for VSNs (used within VSN GRNs)
        # Names align more closely with original TFT paper roles.
        self.static_grn_for_vsns = GatedResidualNetwork(
             units=self.hidden_units, dropout_rate=self.dropout_rate,
             activation=self.activation, use_batch_norm=self.use_batch_norm,
             name="static_grn_for_vsns")
        # Context for Enrichment GRN
        self.static_grn_for_enrichment = GatedResidualNetwork(
             units=self.hidden_units, dropout_rate=self.dropout_rate,
             activation=self.activation, use_batch_norm=self.use_batch_norm,
             name="static_grn_for_enrichment")
        # Context for LSTM Hidden State Init
        self.static_grn_for_state_h = GatedResidualNetwork(
             units=self.lstm_units_list[0], # Match first LSTM layer units
             dropout_rate=self.dropout_rate, activation=self.activation,
             use_batch_norm=self.use_batch_norm, name="static_grn_for_state_h")
        # Context for LSTM Cell State Init
        self.static_grn_for_state_c = GatedResidualNetwork(
             units=self.lstm_units_list[0], # Match first LSTM layer units
             dropout_rate=self.dropout_rate, activation=self.activation,
             use_batch_norm=self.use_batch_norm, name="static_grn_for_state_c")
        
        
        # 5. Positional Encoding
        self.positional_encoding = PositionalEncoding(name="pos_enc")

        
        # 3. LSTM Encoder Layers
        self.lstm_layers = []
        for i in range(self.num_lstm_layers):
            self.lstm_layers.append(LSTM(
                units=self.lstm_units_list[i],
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout_rate, # Use new param
                name=f'encoder_lstm_{i+1}'
            ))

        # 4. Static Enrichment GRN (applied after LSTM)
        # Note: StaticEnrichmentLayer component might encapsulate this GRN
        # Here we define the GRN directly as per paper diagram structure.
        self.static_enrichment_grn = GatedResidualNetwork(
             units=self.hidden_units, dropout_rate=self.dropout_rate,
             activation=self.activation, use_batch_norm=self.use_batch_norm,
             use_time_distributed=True, name="static_enrichment_grn"
             )

        # 5. Temporal Self-Attention (Interpretable Multi-Head Attention)
        # This component internally contains GRNs for Q, K, V projections
        # and the final output gating, potentially conditioned by static context.
        self.temporal_attention_layer = TemporalAttentionLayer(
            units=self.hidden_units, num_heads=self.num_heads,
            dropout_rate=self.dropout_rate, activation=self.activation,
            use_batch_norm=self.use_batch_norm, name="temporal_self_attention")

        # 6. Position-wise Feedforward GRN (applied after attention)
        self.positionwise_grn = GatedResidualNetwork(
            units=self.hidden_units, dropout_rate=self.dropout_rate,
            use_time_distributed=True, activation=self.activation,
            use_batch_norm=self.use_batch_norm, name="pos_wise_ff_grn")

        # 7. Output Layer(s)
        if self.quantiles:
            self.output_layers = [
                TimeDistributed(Dense(self.output_dim), name=f'q_{int(q*100)}_td')
                for q in self.quantiles
            ]
        else:
            self.output_layer = TimeDistributed(
                Dense(self.output_dim), name='point_td'
            )


    def _process_input_type(
        self, inputs, categorical_info, cat_processor, cont_linear,
        input_type_name # For error messages
        ):
        """Helper to process static, dynamic, or future inputs."""
        # Separates categorical/continuous, applies embeddings/linear, concatenates
        if not categorical_info and cont_linear is None:
             # No processing needed if no categoricals and no continuous
             # (should not happen if respective input_dim > 0)
             return inputs # Or maybe raise error?

        cat_inputs, cont_inputs = None, None
        cat_indices = sorted(list(categorical_info.keys()))
        all_indices = set(range(inputs.shape[-1]))
        cont_indices = sorted(list(all_indices - set(cat_indices)))

        # Split based on rank (Static vs Temporal)
        input_rank = tf_rank(inputs)
        if input_rank == 2: # Static (B, F)
            if cat_indices:
                cat_inputs = tf_gather(inputs, cat_indices, axis=1)
            if cont_indices:
                cont_inputs = tf_gather(inputs, cont_indices, axis=1)
        elif input_rank == 3: # Temporal (B, T, F)
            if cat_indices:
                 # Gather along the last axis (features)
                 cat_inputs = tf_gather(inputs, cat_indices, axis=2)
            if cont_indices:
                 cont_inputs = tf_gather(inputs, cont_indices, axis=2)
        else:
            raise ValueError(f"Invalid rank {input_rank} for {input_type_name} inputs.")

        processed_features = []
        if cat_inputs is not None and cat_processor is not None:
            cat_embedded = cat_processor(cat_inputs)
            processed_features.append(cat_embedded)

        if cont_inputs is not None and cont_linear is not None:
            cont_processed = cont_linear(cont_inputs)
            processed_features.append(cont_processed)
        elif cont_inputs is not None and cont_linear is None:
             # If continuous features exist but no linear layer (e.g., static)
             # we might need to handle them - perhaps pass through? Or error?
             # For now, assume if cont_dim > 0, cont_linear exists.
             # If this path is needed, add logic here.
             # Let's pass them through if no linear layer exists
             processed_features.append(cont_inputs)


        if not processed_features:
            # This case should ideally not happen if input_dim > 0
            raise ValueError(f"No features processed for {input_type_name} inputs.")

        # Concatenate processed categorical embeddings and continuous features
        if len(processed_features) == 1:
            return processed_features[0]
        else:
            # Concatenate along the feature dimension (last axis)
            return tf_concat(processed_features, axis=-1)

    def call(self, inputs, training=None):
        """Forward pass for TFT with categorical handling."""
        # Expect inputs as list/tuple: [static, dynamic, future]
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
             raise ValueError(
                 "TFT expects inputs as list/tuple: [static, dynamic, future]."
                 )
        static_inputs, dynamic_inputs, future_inputs = inputs

        # --- 1. Process Inputs (Categorical Embedding + Continuous Linear) ---
        static_processed = self._process_input_type(
            static_inputs, self.static_categorical_info,
            self.static_cat_processor, self.static_cont_linear, "Static"
        )
        dynamic_processed = self._process_input_type(
            dynamic_inputs, self.dynamic_categorical_info,
            self.dynamic_cat_processor, self.dynamic_cont_linear, "Dynamic"
        )
        future_processed = self._process_input_type(
            future_inputs, self.future_categorical_info,
            self.future_cat_processor, self.future_cont_linear, "Future"
        )
        # Note: Output dims are now potentially complex (sum of embed dims + cont dims or hidden_units)
        # VSN needs to handle this - let's assume VSN's `num_inputs` corresponds
        # to the *original* number of features and it handles the split internally.
        # REVISING: The VSN implementation shown expects input (B, T, NumVars, FeatPerVar=1)
        # or (B, NumVars, 1). The _process_input_type returns (B, T, TotalProcessedDim)
        # This requires rethinking VSN or the processing step.

        # --- REVISED PLAN 3 FLOW (Simpler VSN Input) ---
        # 1a. Split inputs into categorical and continuous parts
        static_cat, static_cont = self._split_inputs(
            static_inputs, self.static_categorical_info)
        dynamic_cat, dynamic_cont = self._split_inputs(
            dynamic_inputs, self.dynamic_categorical_info)
        future_cat, future_cont = self._split_inputs(
            future_inputs, self.future_categorical_info)

        # 1b. Process categorical inputs
        static_cat_embed = self.static_cat_processor(static_cat) if static_cat is not None else None
        dynamic_cat_embed = self.dynamic_cat_processor(dynamic_cat) if dynamic_cat is not None else None
        future_cat_embed = self.future_cat_processor(future_cat) if future_cat is not None else None

        # 1c. Combine processed categoricals and original continuous
        # Developer Comment: Ensure correct dimensions and concatenation strategy
        static_processed = self._combine_features(static_cat_embed, static_cont)
        dynamic_processed = self._combine_features(dynamic_cat_embed, dynamic_cont)
        future_processed = self._combine_features(future_cat_embed, future_cont)

        # --- 2. Variable Selection ---
        # VSNs now operate on the combined numerical+embedded categorical features
        # Assuming VSN handles input shape (B, [T,] TotalProcessedFeatures) and
        # selects based on internal logic using original num_inputs.
        # This might require VSN adaptation or careful input preparation.
        # Let's proceed assuming VSN works on the concatenated features for now.
        static_selected = self.static_vsn(static_processed, training=training)
        # ... (Context GRNs as before using static_selected) ...
        context_for_vsns = self.static_grn_for_vsns(static_selected, ...)
        context_for_enrichment = self.static_grn_for_enrichment(static_selected, ...)
        context_state_h = self.static_grn_for_state_h(static_selected, ...)
        context_state_c = self.static_grn_for_state_c(static_selected, ...)
        initial_state = [context_state_h, context_state_c]

        dynamic_selected = self.dynamic_vsn(dynamic_processed, training=training, context=context_for_vsns)
        future_selected = self.future_vsn(future_processed, training=training, context=context_for_vsns)

        # --- 3. Combine Temporal, Positional Encoding, LSTM, Enrichment, Attention, FF ---
        # (Rest of the flow remains largely the same as the previous revised version,
        # using dynamic_selected, future_selected, static contexts, etc.)
        temporal_features = tf_concat([dynamic_selected, future_selected], axis=1) # Check axis/alignment
        temporal_features_pos = self.positional_encoding(temporal_features)
        lstm_output = temporal_features_pos
        current_state = initial_state
        for i, layer in enumerate(self.lstm_layers):
            # ... (LSTM loop with initial state) ...
             if i == 0: lstm_output = layer(lstm_output, initial_state=current_state, training=training)
             else: lstm_output = layer(lstm_output, training=training)
        enriched_output = self.static_enrichment_grn(lstm_output, context=context_for_enrichment, training=training)
        attention_output = self.temporal_attention_layer(enriched_output, context_vector=context_for_vsns, training=training)
        final_temporal_repr = self.positionwise_grn(attention_output, training=training)

        # --- 4. Output Slice and Projection ---
        # (Output logic remains the same)
        output_features = final_temporal_repr[:, -self.forecast_horizon:, :]
        if self.quantiles:
            # ... (quantile output logic) ...
             quantile_outputs = [layer(output_features, training=training) for layer in self.output_layers]
             outputs = tf_stack(quantile_outputs, axis=2)
             if self.output_dim == 1: outputs = tf_squeeze(outputs, axis=-1)
        else:
            outputs = self.output_layer(output_features, training=training)

        return outputs

    # Helper methods for call (implement these)
    def _split_inputs(self, inputs, categorical_info):
         """Splits tensor into categorical and continuous parts based on info."""
         if not categorical_info:
             return None, inputs # All continuous
         if inputs is None:
             return None, None

         cat_indices = sorted(list(categorical_info.keys()))
         # XXX TODO 
         all_indices = tf_range(tf_shape(inputs)[-1])
         # Use tf.sets.difference or numpy equivalent logic
         # This needs careful implementation with tensors
         # For simplicity, assume indices are pre-calculated
         cont_indices = sorted(
             list(set(range(inputs.shape[-1])) - set(cat_indices))) # Use numpy for simplicity

         input_rank = tf_rank(inputs)
         cat_inputs , cont_inputs = None, None

         if input_rank == 2: # Static (B, F)
             if cat_indices: cat_inputs = tf_gather(inputs, cat_indices, axis=1)
             if cont_indices: cont_inputs = tf_gather(inputs, cont_indices, axis=1)
         elif input_rank == 3: # Temporal (B, T, F)
             if cat_indices: cat_inputs = tf_gather(inputs, cat_indices, axis=2)
             if cont_indices: cont_inputs = tf_gather(inputs, cont_indices, axis=2)
         else:
             raise ValueError(f"Unsupported input rank: {input_rank}")

         return cat_inputs, cont_inputs

    def _combine_features(self, cat_embed, cont_features):
         """Concatenates categorical embeddings and continuous features."""
         processed_features = []
         if cat_embed is not None:
             processed_features.append(cat_embed)
         if cont_features is not None:
             processed_features.append(cont_features)

         if not processed_features:
             return None # Or handle error
         if len(processed_features) == 1:
             return processed_features[0]
         else:
             return tf_concat(processed_features, axis=-1)

    # --- Keep compile, get_config, from_config methods ---
    # Need to update get_config/from_config for categorical_info dicts
    def compile(self, optimizer, loss=None, **kwargs):
        if self.quantiles is None:
            effective_loss = loss or 'mean_squared_error'
        else:
            effective_loss = loss or combined_quantile_loss(self.quantiles)
        super().compile(optimizer=optimizer, loss=effective_loss, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'dynamic_input_dim': self.dynamic_input_dim,
            'static_input_dim': self.static_input_dim,
            'future_input_dim': self.future_input_dim,
            'static_categorical_info': self.static_categorical_info, # Added
            'dynamic_categorical_info': self.dynamic_categorical_info,# Added
            'future_categorical_info': self.future_categorical_info, # Added
            'hidden_units': self.hidden_units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout_rate': self.recurrent_dropout_rate,
            'forecast_horizon': self.forecast_horizon,
            'quantiles': self.quantiles,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
            'num_lstm_layers': self.num_lstm_layers,
            'lstm_units': self.lstm_units, # Store original spec
            'output_dim': self.output_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
         # Need to handle deserialization of categorical info dicts if complex
        return cls(**config)
# -------------------------------------- TFT components -----------------------
# LSTM Encoder 
class _LSTMEncoder(Layer):
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
class _LSTMDecoder(Layer):
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
class _TemporalSelfAttention(Layer):
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
class _TemporalFusionDecoder(Layer):
    def __init__(self, hidden_units, num_heads, dropout_rate=0.0,
                 activation='elu', use_batch_norm=False):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        
        # Temporal Self-Attention layer
        self.attention = _TemporalSelfAttention(
            hidden_units=hidden_units, 
            num_heads=num_heads, 
            dropout_rate=dropout_rate
        )
        
        # Static enrichment via GRN
        self.static_enrichment = _GatedResidualNetwork(
            hidden_units=hidden_units, 
            dropout_rate=dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        
        # Final GRN after attention
        self.post_attention_grn = _GatedResidualNetwork(
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
class _GatedResidualNetwork(Layer):
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

class _VariableSelectionNetwork(Layer):
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
       

