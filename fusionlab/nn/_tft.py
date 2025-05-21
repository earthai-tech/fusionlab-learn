# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Temporal Fusion Transformer (TFT), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""
from numbers import Real, Integral  
import logging
from typing import Optional, List, Any, Union

from .._fusionlog import fusionlog
from ..api.docstring import DocstringComponents
from ..api.property import  NNLearner 
from ..core.checks import is_iterable
from ..core.diagnose_q import validate_quantiles
from ..core.handlers import param_deprecated_message 
from ..compat.sklearn import validate_params, Interval, StrOptions 
from ..decorators import Appender 
from ..utils.deps_utils import ensure_pkg

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from ._nn_docs import _shared_nn_params, _shared_docs  
 
if KERAS_BACKEND:
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
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    
    tf_reduce_sum = KERAS_DEPS.reduce_sum
    tf_stack = KERAS_DEPS.stack
    tf_expand_dims = KERAS_DEPS.expand_dims
    tf_tile = KERAS_DEPS.tile
    tf_range=KERAS_DEPS.range 
    tf_concat = KERAS_DEPS.concat
    tf_shape = KERAS_DEPS.shape
    tf_rank=KERAS_DEPS.rank
    
    tf_autograph=KERAS_DEPS.autograph
    tf_autograph.set_verbosity(0)
    
    from ._tensor_validation import validate_model_inputs, align_temporal_dimensions
    from .components import StaticEnrichmentLayer, PositionalEncoding
    from .components import GatedResidualNetwork, VariableSelectionNetwork 
    from .components import TemporalAttentionLayer
    
DEP_MSG = dependency_message('transformers.tft') 

__all__ = ["TemporalFusionTransformer", "DummyTFT"]

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_shared_nn_params), 
)


@Appender(_shared_docs['tft_math_doc'], join='\n', indents=0)
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'future_input_dim',
            'condition': lambda v: v is not None,
            'message': (
                "The 'future_input_dim' parameter is accepted by "
                "DummyTFT for API consistency with other TFT models, "
                "but DummyTFT does not utilize future covariates. "
                "This parameter will be ignored and effectively treated "
                "as None. If you need to use known future covariates, "
                "please consider using the standard "
                ":class:`~fusionlab.nn.transformers.TemporalFusionTransformer` "
                "(for flexible input handling) or "
                ":class:`~fusionlab.nn.transformers.TFT` (for stricter "
                "three-input structure)."
                ),
            'default': None 
        }
    ]
)
@register_keras_serializable(
    'fusionlab.nn.transformers',
    name="DummyTFT"
)
class DummyTFT(Model, NNLearner):
    """
    DummyTFT: Simplified TFT variant using only Static and Dynamic inputs.
    """
    @validate_params({
        "static_input_dim": [Interval(Integral, 0, None, closed='left')],
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', None],
        "activation": [StrOptions(
            {"elu", "relu", "tanh", "sigmoid", "linear", "gelu"})],
        "use_batch_norm": [bool],
        "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": ['array-like', Interval(
            Integral, 1, None, closed='left'), None],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
        },
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        future_input_dim: Any=None, 
        hidden_units: int = 32,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        forecast_horizon: int = 1,
        quantiles: Optional[List[float]] = None,
        activation: str = 'elu',
        use_batch_norm: bool = False,
        num_lstm_layers: int = 1,
        lstm_units: Optional[Union[int, List[int]]] = None,
        output_dim: int = 1,
        name: Optional[str] = None,
        **kwargs
    ):
        if name is None:
            name = "DummyTFT" # Default name
        super().__init__(name=name, **kwargs)

        # Initialize Logger
        if fusionlog:
            self.logger = fusionlog().get_fusionlab_logger(
                f"{__name__}.{self.__class__.__name__}"
            )
        else:
            self.logger = logging.getLogger(
                f"{__name__}.{self.__class__.__name__}"
            )
            self.logger.info(
                "fusionlog not found, using standard Python logging."
                )

        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim # Store as None
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.forecast_horizon = forecast_horizon
        self.activation_str = activation 
        self.use_batch_norm = use_batch_norm
        self.num_lstm_layers = num_lstm_layers
        
        self._lstm_units_config = lstm_units
        self.output_dim = output_dim # Store output_dim
        self.quantiles = quantiles 
        
        

        # Process quantiles
        if self.quantiles is None:
            # For point forecast, output layer produces output_dim features
            self.num_output_features = self.output_dim
        else:
            self.quantiles = validate_quantiles(quantiles) 
            # For quantile, each quantile predicts output_dim features
            self.num_output_features = len(self.quantiles) * self.output_dim

        self.logger.debug(
            f"Initializing {self.name} with output_dim={self.output_dim}"
            f", num_quantiles_for_output_layer_units="
            f"{self.num_output_features if self.quantiles else self.output_dim}"
            f"hidden_units={hidden_units}, "
            f"num_heads={num_heads}, "
            f"dropout_rate={dropout_rate}, "
            f"forecast_horizon={forecast_horizon}, "
            f"quantiles={quantiles}, "
            f"activation={activation}, "
            f"use_batch_norm={use_batch_norm}, "
            f"num_lstm_layers={num_lstm_layers}, "
            f"lstm_units={lstm_units}"
        )

        # --- Layer Definitions ---
        self.logger.debug(
            f"  Initializing internal layers for {self.name}...")

        # Variable Selection Networks
        self.static_var_sel = VariableSelectionNetwork(
            num_inputs=static_input_dim, units=hidden_units,
            dropout_rate=dropout_rate, activation=self.activation_str,
            use_batch_norm=use_batch_norm, name="static_vsn"
        )
        self.dynamic_var_sel = VariableSelectionNetwork(
            num_inputs=dynamic_input_dim, units=hidden_units,
            dropout_rate=dropout_rate, use_time_distributed=True,
            activation=self.activation_str, use_batch_norm=use_batch_norm,
            name="dynamic_vsn"
        )

        self.positional_encoding = PositionalEncoding(
            name="positional_encoding"
            )

        # Static Context GRNs
        self.static_context_grn_attn = GatedResidualNetwork(
            units=hidden_units, dropout_rate=dropout_rate,
            activation=self.activation_str,
            use_batch_norm=use_batch_norm,
            name="static_context_for_attention"
        )
        self.static_context_grn_enrich = GatedResidualNetwork(
            units=hidden_units, dropout_rate=dropout_rate,
            activation=self.activation_str,
            use_batch_norm=use_batch_norm, 
            name="static_context_for_enrichment"
        )

        # Determine LSTM units for state projection
        # Use units of the first LSTM layer if defined, else hidden_units
        _first_lstm_units = hidden_units # Default
        if isinstance(lstm_units, int):
            _first_lstm_units = lstm_units
        elif isinstance(lstm_units, list) and lstm_units:
            _first_lstm_units = lstm_units[0]

        self.static_to_lstm_state_h = Dense(
            _first_lstm_units, activation=self.activation_str,
            name="static_to_h"
            )
        self.static_to_lstm_state_c = Dense(
            _first_lstm_units, activation=self.activation_str,
            name="static_to_c"
            )

        # LSTM Encoder
        self.lstm_layers = []
        actual_lstm_units_list = []
        if lstm_units is None:
            actual_lstm_units_list = [hidden_units] * num_lstm_layers
        elif isinstance(lstm_units, int):
            actual_lstm_units_list = [lstm_units] * num_lstm_layers
        elif is_iterable(lstm_units): # Use your is_iterable
            lstm_units_list = list(lstm_units) # Ensure list
            if len(lstm_units_list) != num_lstm_layers:
                raise ValueError(
                    "Length of `lstm_units` list must match `num_lstm_layers`."
                )
            actual_lstm_units_list = lstm_units_list
        else: # Should be caught by @validate_params
            raise TypeError(
                "`lstm_units` must be int, list of int, or None."
                )

        for i in range(num_lstm_layers):
            self.lstm_layers.append(
                LSTM(
                    units=actual_lstm_units_list[i],
                    return_sequences=True,
                    dropout=dropout_rate,
                    name=f'lstm_encoder_layer_{i+1}'
                )
            )

        # Static Enrichment Layer
        self.static_enrichment = StaticEnrichmentLayer(
            units=hidden_units, activation=self.activation_str,
            use_batch_norm=use_batch_norm, name="static_enrichment"
        )

        # Temporal Attention Layer
        self.temporal_attention = TemporalAttentionLayer(
            units=hidden_units, num_heads=num_heads,
            dropout_rate=dropout_rate, activation=self.activation_str,
            use_batch_norm=use_batch_norm, name="temporal_self_attention"
        )

        # Position-wise Feedforward GRN
        self.positionwise_grn = GatedResidualNetwork(
            units=hidden_units, dropout_rate=dropout_rate,
            activation=self.activation_str,
            use_batch_norm=use_batch_norm, name="positionwise_ff_grn"
        )

        # Output Layer(s)
        if self.quantiles is not None:
            self.logger.debug(
                "  Initializing Quantile Output Projection Layers..."
                )
            # Each Dense layer predicts all output_dims for one quantile
            self.output_projection_layers = [
                TimeDistributed(
                    Dense(self.output_dim, name=f'dense_q_{q_idx}'),
                    name=f'output_projection_q_{q_idx}'
                    )
                for q_idx in range(len(self.quantiles))
            ]
        else: # Point forecast
            self.logger.debug(
                "  Initializing Point Output Projection Layer..."
                )
            self.output_projection_layer = TimeDistributed(
                Dense(self.output_dim, name='dense_point_output'),
                name='output_projection_point'
                )

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False, **kwargs):
        """Forward pass for DummyTFT (Static and Dynamic inputs only)."""
        self.logger.debug(
            f"DummyTFT '{self.name}': Entering call method."
            )

        # --- Input Validation ---
        # DummyTFT expects a list of 2: [static_raw, dynamic_raw]
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "DummyTFT expects inputs as a list/tuple of 2 elements: "
                "[static_inputs, dynamic_inputs]."
            )
        static_raw, dynamic_raw = inputs[0], inputs[1]
        # For validate_model_inputs, future_raw is None for DummyTFT
        future_raw = None

        if static_raw is None or dynamic_raw is None:
            # This check is crucial as both are required by __init__
            raise ValueError(
                "Static and Dynamic inputs must be provided (not None) "
                "for DummyTFT."
            )

        # Use validate_model_inputs.
        # It expects [S, D, F] and returns (S_p, D_p, F_p)
        static_input, dynamic_input, future_input_processed = \
            validate_model_inputs(
                inputs=[static_raw, dynamic_raw, future_raw],
                static_input_dim=self.static_input_dim,
                dynamic_input_dim=self.dynamic_input_dim,
                future_covariate_dim=None, # DummyTFT has no future_input_dim
                forecast_horizon=self.forecast_horizon,
                mode='soft', # Enforce dims for S and D
                model_name='tft_flex', # Treat as a strict variant
                verbose= 1 if self.logger.level <= logging.DEBUG else 0
        )
        # future_input_processed will be None here.

        if self.logger.level <= logging.DEBUG:
            self.logger.debug(
                f"  Inputs validated. Static: {static_input.shape}, "
                f"Dynamic: {dynamic_input.shape}"
            )

        # --- Static Pathway ---
        static_selected = self.static_var_sel(
            static_input, training=training
            )
        if hasattr(self.static_var_sel, 'variable_importances_'):
            self.static_variable_importances_ = \
                self.static_var_sel.variable_importances_
        if self.logger.level <= logging.DEBUG:
            self.logger.debug(
                f"  Static VSN output shape: {static_selected.shape}"
            )

        context_for_attention = self.static_context_grn_attn(
            static_selected, training=training
            )
        context_for_enrichment = self.static_context_grn_enrich(
            static_selected, training=training
            )
        initial_state_h = self.static_to_lstm_state_h(static_selected)
        initial_state_c = self.static_to_lstm_state_c(static_selected)
        lstm_initial_state = [initial_state_h, initial_state_c]

        # --- Temporal Pathway (Dynamic Inputs Only) ---
        dynamic_selected = self.dynamic_var_sel(
            dynamic_input, training=training
            )
        if hasattr(self.dynamic_var_sel, 'variable_importances_'):
            self.dynamic_variable_importances_ = \
                self.dynamic_var_sel.variable_importances_

        temporal_embeddings = self.positional_encoding(
            dynamic_selected, training=training
            )

        lstm_output = temporal_embeddings
        for i, layer in enumerate(self.lstm_layers):
            if self.logger.level <= logging.DEBUG: # Use standard logging levels
                self.logger.debug(
                    f"    LSTM layer {i+1} input: {lstm_output.shape}"
                    )
            if i == 0:
                lstm_output = layer(
                    lstm_output, initial_state=lstm_initial_state,
                    training=training
                    )
            else:
                lstm_output = layer(lstm_output, training=training)
            if self.logger.level <= logging.DEBUG:
                self.logger.debug(
                    f"    LSTM layer {i+1} output: {lstm_output.shape}"
                    )

        enriched_sequence = self.static_enrichment(
            lstm_output, context_vector=context_for_enrichment, # Pass context
            training=training
            )

        attention_output = self.temporal_attention(
            enriched_sequence, context_vector=context_for_attention,
            training=training
            )

        final_temporal_representation = self.positionwise_grn(
            attention_output, training=training
            )

        # Output Slice and Projection
        output_features_sliced = final_temporal_representation[
            :, -self.forecast_horizon:, :
            ]

        if self.quantiles is not None:
            quantile_outputs_list = [
                layer(output_features_sliced, training=training)
                for layer in self.output_projection_layers
            ]
            # Each element in list is (B, H, O).
            # If O=1, concat on last axis -> (B, H, Q).
            # If O>1, stack on new axis -> (B, H, Q, O).
            if self.output_dim == 1:
                outputs = tf_concat(quantile_outputs_list, axis=-1)
            else:
                outputs = tf_stack(quantile_outputs_list, axis=2)
        else: # Point forecast
            outputs = self.output_projection_layer(
                output_features_sliced, training=training
                ) # Shape (B, H, O)

        if self.logger.level <= logging.DEBUG:
            self.logger.debug(f"DummyTFT: Final output shape: {outputs.shape}")
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'static_input_dim': self.static_input_dim,
            'dynamic_input_dim': self.dynamic_input_dim,
            'future_input_dim': self.future_input_dim,
            'hidden_units': self.hidden_units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'forecast_horizon': self.forecast_horizon,
            'quantiles': self.quantiles, # Store original list or None
            'activation': self.activation_str,
            'use_batch_norm': self.use_batch_norm,
            'num_lstm_layers': self.num_lstm_layers,
            'lstm_units': self._lstm_units_config, # Store original
            'output_dim': self.output_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
@register_keras_serializable( 
   'fusionlab.nn.transformers', 
    name="TemporalFusionTransformer"
)
class TemporalFusionTransformer(Model, NNLearner):
    @validate_params({
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "static_input_dim" : [Interval(Integral, 0, None, closed='left'), None],
        "future_input_dim" : [Interval(Integral, 0, None, closed='left'), None],
        "hidden_units"     : [Interval(Integral, 1, None, closed='left'), None],
        "num_heads"        : [Interval(Integral, 1, None, closed='left')],
        "dropout_rate"     : [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles"        : ['array-like', None],
        "activation"       : [StrOptions({"elu", "relu", "tanh", "sigmoid",
                                          "linear", "gelu"})],
        "use_batch_norm"   : [bool],
        "num_lstm_layers"  : [Interval(Integral, 1, None, closed='left')],
        "lstm_units"       : ['array-like', Interval(Integral, 1, None, 
                                     closed='left'), None]
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        dynamic_input_dim,
        static_input_dim=None,
        future_input_dim=None,
        hidden_units=32,
        num_heads=4,
        dropout_rate=0.1,
        forecast_horizon=1,
        quantiles=None,
        activation='elu',
        use_batch_norm=False,
        num_lstm_layers=1,
        lstm_units=None,
        output_dim = 1, 
        **kw
    ):
        super().__init__(**kw)

        self.logger = fusionlog().get_fusionlab_logger(__name__)
        self.logger.debug(
            "Initializing NTemporalFusionTransformer with parameters: "
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
            f"num_lstm_layers={num_lstm_layers}, "
            f"lstm_units={lstm_units}"
        )

        self.static_input_dim  = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim  = future_input_dim
        self.hidden_units      = hidden_units
        self.num_heads         = num_heads
        self.dropout_rate      = dropout_rate
        self.forecast_horizon = forecast_horizon
        self.quantiles         = quantiles
        self.use_batch_norm    = use_batch_norm
        self.num_lstm_layers   = num_lstm_layers
        self.lstm_units        = lstm_units

        # Convert string activation to a Keras Activation layer 
        # for uniform usage across sub-layers.
        self.activation = activation

        # If quantiles are not provided, interpret as single output 
        # (deterministic). Otherwise, parse and store them.
        if quantiles is None:
            self.quantiles = None
            self.num_quantiles = 1
        else:
            self.quantiles = validate_quantiles(quantiles)
            self.num_quantiles = len(self.quantiles)

        # Initialize variable selection networks for static, dynamic, 
        # and optionally future inputs.
        self.logger.debug("Initializing Variable Selection Networks...")

        # For static inputs (metadata)
        self.static_var_sel = (
            VariableSelectionNetwork(
                num_inputs=static_input_dim,
                units=hidden_units,
                dropout_rate=dropout_rate,
                activation=self.activation,
                use_batch_norm=use_batch_norm
            ) if static_input_dim else None
        )

        # For dynamic (past) inputs
        self.dynamic_var_sel = VariableSelectionNetwork(
            num_inputs=dynamic_input_dim,
            units=hidden_units,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            activation=self.activation,
            use_batch_norm=use_batch_norm
        )

        # For future inputs (if needed)
        if self.future_input_dim is not None:
            self.logger.debug("Initializing Future Variable Selection Network...")
            self.future_var_sel = VariableSelectionNetwork(
                num_inputs=future_input_dim,
                units=hidden_units,
                dropout_rate=dropout_rate,
                use_time_distributed=True,
                activation=self.activation,
                use_batch_norm=use_batch_norm
            )
        else:
            self.future_var_sel = None

        # Positional Encoding handles the time step embedding 
        # for the dynamic/future sequences.
        self.logger.debug("Initializing Positional Encoding...")
        self.positional_encoding = PositionalEncoding()

        # Static Context GRNs are used to derive static context vectors
        # and enrichment vectors from static inputs (if present).
        self.logger.debug("Initializing Static Context GRNs...")
        self.static_context_grn = (
            GatedResidualNetwork(
                hidden_units,
                dropout_rate,
                activation=self.activation,
                use_batch_norm=use_batch_norm
            ) if static_input_dim else None
        )
        self.static_context_enrichment_grn = (
            GatedResidualNetwork(
                hidden_units,
                dropout_rate,
                activation=self.activation,
                use_batch_norm=use_batch_norm
            ) if static_input_dim else None
        )

        # LSTM Encoder: multi-layer LSTMs to encode the historical 
        # (plus possibly future) time series embeddings.
        self.logger.debug("Initializing LSTM Encoder Layers...")
        self.lstm_layers = []
        if self.lstm_units is not None:
            # Convert user-supplied LSTM units to a list if not already.
            self.lstm_units = is_iterable(self.lstm_units, transform=True)
        for i in range(num_lstm_layers):
            lstm_units_i = hidden_units
            if self.lstm_units is not None and i < len(self.lstm_units):
                lstm_units_i = self.lstm_units[i]
            self.lstm_layers.append(
                LSTM(
                    lstm_units_i,
                    return_sequences=True,
                    dropout=dropout_rate,
                    name=f'lstm_layer_{i+1}'
                )
            )

        # Static Enrichment: merges static context with the LSTM-encoded 
        # dynamic embeddings prior to attention.
        self.logger.debug("Initializing Static Enrichment Layer...")
        self.static_enrichment = (
            StaticEnrichmentLayer(
                hidden_units,
                activation=self.activation,
                use_batch_norm=use_batch_norm
            ) if static_input_dim else None
        )

        # Temporal Attention Layer for interpretability and weighting 
        # various time steps. 
        self.logger.debug("Initializing Temporal Attention Layer...")
        self.temporal_attention = TemporalAttentionLayer(
            hidden_units,
            num_heads,
            dropout_rate,
            activation=self.activation,
            use_batch_norm=use_batch_norm
        )

        # Position-wise Feedforward (GRN) for final transformation 
        # after attention.
        self.logger.debug("Initializing Position-wise Feedforward Network...")
        self.positionwise_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate,
            # use_time_distributed=True,
            activation=self.activation,
            use_batch_norm=use_batch_norm
        )

        # Output Layers for either multiple quantiles or a single 
        # deterministic point forecast.
        if self.quantiles is not None:
            self.logger.debug("Initializing Quantile Output Layers...")
            self.quantile_outputs = [
                TimeDistributed(Dense(1), name=f'quantile_output_{i+1}')
                for i in range(self.num_quantiles)
            ]
        else:
            self.logger.debug(
                "Initializing Single Output Layer for Point Predictions...")
            self.output_layer = TimeDistributed(Dense(1), name='output_layer')

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False, **kw):
        """
        The main forward pass for NTemporalFusionTransformer.

        1. Validate and unpack `inputs` using `validate_tft_inputs`.
        2. Apply variable selection to static, dynamic, and future inputs.
        3. Perform positional encoding on dynamic+future sequences.
        4. Compute static context vectors if static is present.
        5. Pass through LSTM encoders.
        6. Optionally enrich dynamic with static context.
        7. Temporal attention for interpretable weighting of time steps.
        8. Position-wise feedforward (GRN).
        9. Final slicing (forecast horizon) and output (quantiles or single).

        Parameters
        ----------
        inputs : tuple
            Should contain up to three elements:
            (dynamic_inputs, future_inputs, static_inputs)
            or fewer if not all are provided.
        training : bool, default=False
            Whether in training mode (affects dropout, BN, etc.).

        Returns
        -------
        tf.Tensor
            Final predicted sequences of shape 
            (batch_size, forecast_horizon, num_quantiles or 1).
        """
        self.logger.debug("Starting call method with inputs.")

        # Use the validation function to unify shapes and optionally
        # convert them to tf.float32. The function returns 
        # `validate_model_inputs` expects [S,D,F] and returns (S,D,F)
    
        static_inputs, dynamic_inputs, future_inputs = validate_model_inputs(
            inputs=inputs,
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            forecast_horizon=self.forecast_horizon,
            model_name='tft_flex',
            mode="soft", 
        )


        # 1. Apply Variable Selection on static (if present).
        self.logger.debug("Applying Variable Selection on Static Inputs (if any).")
        if self.static_input_dim and self.static_var_sel:
            static_embedding = self.static_var_sel(static_inputs, training=training)
            self.static_variable_importances_ = self.static_var_sel.variable_importances_
        else:
            static_embedding = None

        # 2. Variable Selection on dynamic (past) inputs.
        self.logger.debug("Applying Variable Selection on Dynamic Inputs...")
        dynamic_embedding = self.dynamic_var_sel(dynamic_inputs, training=training)
        self.dynamic_variable_importances_ = self.dynamic_var_sel.variable_importances_

        # 3. If future inputs exist, apply future var selection and 
        #    concatenate with dynamic.
        if self.future_input_dim and self.future_var_sel:
            self.logger.debug("Applying Variable Selection on Future Inputs...")
            fut_embed = self.future_var_sel(future_inputs, training=training)
            self.future_variable_importances_ = self.future_var_sel.variable_importances_
            
            # dynamic_input is the reference for T_past (lookback period).
    
            self.logger.debug("  Aligning temporal inputs for MultiModalEmbedding...")
            _, fut_embed = align_temporal_dimensions(
                tensor_ref=dynamic_embedding,       # Shape (B, T_past, D_dyn)
                tensor_to_align=fut_embed,   # Shape (B, T_future_total, D_fut)
                mode='slice_to_ref',            # Slice future if longer
                name="future_input_for_mme"
            )
            
            dynamic_embedding = tf_concat([dynamic_embedding, fut_embed], axis=1)

        # 4. Positional encoding for combined sequence embedding
        #    (dynamic + future).
        self.logger.debug("Applying Positional Encoding to dynamic/future Embedding...")
        dynamic_embedding = self.positional_encoding(
            dynamic_embedding, 
            training=training 
            )

        # 5. Compute static context if static_embedding is available.
        if static_embedding is not None:
            self.logger.debug("Generating Static Context Vector & Enrichment...")
            static_context_vector = self.static_context_grn(
                static_embedding, training=training
                )
            static_enrichment_vector = self.static_context_enrichment_grn(
                static_embedding, training=training
                )
        else:
            static_context_vector = None
            static_enrichment_vector = None

        # 6. Pass embeddings through multi-layer LSTM encoders.
        self.logger.debug("Passing through LSTM Encoder Layers...")
        x = dynamic_embedding
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)

        # 7. Static enrichment merges the static context with LSTM output 
        #    if static is present.
        if self.static_enrichment and static_enrichment_vector is not None:
            self.logger.debug("Applying Static Enrichment...")
            enriched_lstm_output = self.static_enrichment(
                x, static_enrichment_vector, training=training
            )
        else:
            enriched_lstm_output = x

        # 8. Temporal Attention for interpretable weighting of time steps, 
        #    possibly using static context as well.
        self.logger.debug("Applying Temporal Attention...")
        attention_output = self.temporal_attention(
            enriched_lstm_output,
            context_vector=static_context_vector,
            training=training
        )

        # 9. Position-wise feedforward transforms the attended output.
        self.logger.debug("Applying Position-wise Feedforward...")
        temporal_feature = self.positionwise_grn(
            attention_output, training=training
            )

        # 10. Slice the last `forecast_horizon` steps for final prediction
        #     and apply output projection (either quantiles or point).
        self.logger.debug("Generating Final Output...")
        decoder_steps = self.forecast_horizon
        outputs = temporal_feature[:, -decoder_steps:, :]

        if self.quantiles:
            self.logger.debug("Generating Quantile Outputs...")
            quantile_outs = []
            for i, qout in enumerate(self.quantile_outputs):
                out_i = qout(outputs)
                quantile_outs.append(out_i)
            final_output = Concatenate(
                axis=-1, name='final_quantile_output')(quantile_outs)
        else:
            self.logger.debug("Generating Point Predictions...")
            final_output = self.output_layer(outputs)

        self.logger.debug("Call method completed.")
        
        return final_output

    def get_config(self):
        """
        Return the model configuration for serialization. 
        Includes all hyperparameters that define the structure 
        of the NTemporalFusionTransformer.
        """
        config = super().get_config().copy()
        config.update({
            'dynamic_input_dim' : self.dynamic_input_dim,
            'static_input_dim'  : self.static_input_dim,
            'future_input_dim'  : self.future_input_dim,
            'hidden_units'      : self.hidden_units,
            'num_heads'         : self.num_heads,
            'dropout_rate'      : self.dropout_rate,
            'forecast_horizon' : self.forecast_horizon,
            'quantiles'         : self.quantiles,
            'activation'        : self.activation,
            'use_batch_norm'    : self.use_batch_norm,
            'num_lstm_layers'   : self.num_lstm_layers,
            'lstm_units'        : self.lstm_units,
        })
        self.logger.debug("Configuration for get_config has been updated.")
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreate NTemporalFusionTransformer instance from config.
        This classmethod is invoked by Keras to deserialize the model.
        """
        cls.logger = fusionlog().get_fusionlab_logger(__name__)
        cls.logger.debug("Creating NTemporalFusionTransformer instance from config.")
        return cls(**config)

DummyTFT.__doc__+="""\

The DummyTFT combines high-performance multi-horizon
forecasting with interpretable insights into temporal dynamics [1]_.
It integrates several advanced mechanisms, including:

- Variable Selection Networks (VSNs) for static and dynamic features.
- Gated Residual Networks (GRNs) for processing inputs.
- Static Enrichment Layer to incorporate static features into temporal
  processing.
- LSTM Encoder for capturing sequential dependencies.
- Temporal Attention Layer for focusing on important time steps.
- Position-wise Feedforward Layer.
- Final Output Layer for prediction.

Parameters
----------
static_input_dim : int
    The input dimension per static variable. Typically ``1`` for scalar
    features or higher for embeddings. This defines the number of features
    for each static variable. For example, if static variables are
    represented using embeddings of size 16, then ``static_input_dim``
    would be ``16``.

dynamic_input_dim : int
    The input dimension per dynamic variable. This defines the number of
    features for each dynamic variable at each time step. For instance, if
    dynamic variables are represented using embeddings or multiple features,
    specify the appropriate dimension.


{params.base.hidden_units} 
{params.base.num_heads}
{params.base.dropout_rate} 

forecast_horizon : int, optional
    The number of time steps to forecast. Default is ``1``. This parameter
    defines the number of future time steps the model will predict. For
    multi-step forecasting, set ``forecast_horizon`` to the desired number
    of future steps.

{params.base.quantiles} 
{params.base.activation} 
{params.base.use_batch_norm} 

num_lstm_layers : int, optional
    Number of LSTM layers in the encoder. Default is ``1``. Adding more
    layers can help the model capture more complex sequential patterns.
    Each additional layer processes the output of the previous LSTM layer.

lstm_units : list of int or None, optional
    List containing the number of units for each LSTM layer. If ``None``,
    all LSTM layers have ``hidden_units`` units. Default is ``None``.
    This parameter allows customizing the size of each LSTM layer. For
    example, to specify different units for each layer, provide a list like
    ``[64, 32]``.

Methods
-------
call(inputs, training=False)
    Forward pass of the model.

    Parameters
    ----------
    inputs : tuple of tensors
        A tuple containing ``(static_inputs, dynamic_inputs)``.

        - ``static_inputs``: Tensor of shape ``(batch_size, num_static_vars,
          static_input_dim)`` representing the static features.
        - ``dynamic_inputs``: Tensor of shape ``(batch_size, time_steps,
          num_dynamic_vars, dynamic_input_dim)`` representing the dynamic
          features.

    training : bool, optional
        Whether the model is in training mode. Default is ``False``.

    Returns
    -------
    Tensor
        The output predictions of the model. The shape depends on the
        ``forecast_horizon`` and whether ``quantiles`` are used.

get_config()
    Returns the configuration of the model for serialization.

from_config(config)
    Instantiates the model from a configuration dictionary.

""".format( params=_param_docs) 

TemporalFusionTransformer.__doc__="""\
TemporalFusionTransformer model implementation for multi-horizon 
forecasting, with optional static, past, and future inputs.

This class extends Keras `Model` and integrates with the gofast 
NNLearner interface. It supports dynamic (past) inputs, optional 
static inputs, and newly added optional future inputs 
(``future_input_dim``). By including the future covariates, the 
TemporalFusionTransformer can account for known future features 
(e.g., events, planned discount rates, etc.) in its predictions.

Parameters
----------
dynamic_input_dim: int
    Dimensionality of the dynamic (past) inputs. This is mandatory 
    for the TFT model.
static_input_dim : int, optional
    Dimensionality of static inputs. If not `None`, the call method
    will expect static inputs.
future_input_dim : int, optional
    Dimensionality of future (known) inputs. If not `None`, the call
    method will expect future inputs to handle exogenous covariates
    known in the future (e.g., events, planned promotions, etc.).
hidden_units : int, default=32
    Number of hidden units for the layers that do not have a distinct 
    specification (e.g., GRNs, variable selection networks).
num_heads : int, default=4
    Number of attention heads in the multi-head attention layer.
dropout_rate : float, default=0.1
    Dropout rate for various layers (GRNs, attention, etc.).
forecast_horizon : int, default=1
    Number of timesteps to forecast into the future.
quantiles : list of float, optional
    List of quantiles for probabilistic forecasting. If `None`, a 
    single deterministic output is produced.
activation : str, default='elu'
    Activation function. Must be one of ``{'elu', 'relu', 'tanh', 
    'sigmoid', 'linear', 'gelu'}``.
use_batch_norm : bool, default=False
    Whether to apply batch normalization in various sub-layers.
num_lstm_layers : int, default=1
    Number of LSTM layers in the encoder.
lstm_units : list of int or None, default=None
    If provided, each index corresponds to the number of LSTM 
    units for that layer. If `None`, uses ``hidden_units`` for 
    each layer.

Examples
--------
>>> from fusionlab.nn._tensor_validation import validate_tft_inputs
>>> from fusionlab.nn.tft import TemporalFusionTransformer
>>> model = TemporalFusionTransformer(
...     dynamic_input_dim=10,
...     static_input_dim=5,
...     future_input_dim=8,
...     hidden_units=32,
...     num_heads=4,
...     dropout_rate=0.1,
...     forecast_horizon=7,
...     quantiles=[0.1, 0.5, 0.9],
...     activation='elu',
...     use_batch_norm=True,
...     num_lstm_layers=2,
...     lstm_units=[64, 32]
... )

Notes
-----
The newly added ``future_input_dim`` allows the model to incorporate 
future covariates known at forecast time. In the ``call`` method, if 
``future_input_dim`` is not `None`, the model expects three inputs:
``(static_inputs, dynamic_inputs, future_inputs)``. Otherwise, it 
expects only ``(static_inputs, dynamic_inputs)``.

See Also
--------
VariableSelectionNetwork : For feature selection and embedding.
GatedResidualNetwork : A GRN used in various sub-layers.
LSTM : Keras LSTM layers for sequence processing.

References
----------
.. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2019).
       Temporal Fusion Transformers for Interpretable
       Multi-horizon Time Series Forecasting.
       https://arxiv.org/abs/1912.09363
"""