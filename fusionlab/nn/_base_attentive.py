# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Base class for advanced, attentive sequence-to-sequence models
like HALNet and PIHALNet and more. 
"""

from __future__ import annotations
import warnings 
from numbers import Integral, Real 
from typing import List, Optional, Union, Any, Dict

from .._fusionlog import fusionlog, OncePerMessageFilter
from ..api.docs import DocstringComponents, _halnet_core_params
from ..api.property import NNLearner
from ..compat.sklearn import validate_params, Interval, StrOptions
from ..utils.generic_utils import select_mode
from ..utils.deps_utils import ensure_pkg

from . import KERAS_BACKEND, KERAS_DEPS, dependency_message
from .comp_utils import resolve_attention_levels 

if KERAS_BACKEND:
    from .components import (
        Activation, 
        CrossAttention, 
        DynamicTimeWindow, 
        GatedResidualNetwork, 
        HierarchicalAttention,
        MemoryAugmentedAttention, 
        MultiDecoder, 
        MultiHeadAttention, 
        MultiResolutionAttentionFusion,
        MultiScaleLSTM, 
        PositionalEncoding, 
        QuantileDistributionModeling, 
        VariableSelectionNetwork, 
        aggregate_multiscale_on_3d, 
        aggregate_time_window_output
    )
    from .utils import set_default_params
    from ._tensor_validation import validate_model_inputs
  
    Add =KERAS_DEPS.Add 
    Dense= KERAS_DEPS.Dense 
    Tensor = KERAS_DEPS.Tensor
    Layer=KERAS_DEPS.Layer
    LayerNormalization=KERAS_DEPS.LayerNormalization
    LSTM=KERAS_DEPS.LSTM
    Model=KERAS_DEPS.Model
    tf_shape = KERAS_DEPS.shape
    tf_concat = KERAS_DEPS.concat
    tf_zeros = KERAS_DEPS.zeros
    tf_expand_dims = KERAS_DEPS.expand_dims
    tf_tile = KERAS_DEPS.tile
    tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor
    tf_assert_equal = KERAS_DEPS.debugging.assert_equal
else:
    class Model: pass
    class Layer: pass
    class _DummyRegister_keras_serializable: 
        pass 
    class KERAS_DEPS: 
        register_keras_serializable = _DummyRegister_keras_serializable
        
    Tensor = Any

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

DEP_MSG = dependency_message('nn.models')
_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_halnet_core_params), 
)

DEFAULT_ARCHITECTURE = {
    'encoder_type': 'hybrid',
    'decoder_attention_stack': ['cross', 'hierarchical', 'memory'],
    'feature_processing': 'vsn', 
}

@KERAS_DEPS.register_keras_serializable(
    'fusionlab.nn.models', name="BaseAttentive"
)
class BaseAttentive(Model, NNLearner):
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
            'array-like', Interval(Integral, 1, None, closed='left')
          ], 
        "lstm_units": [
            'array-like', Interval(Integral, 1, None, closed='left'), 
            None
        ], 
        "activation": [StrOptions(
            {"elu", "relu", "tanh", "sigmoid", "linear", "gelu"}),callable 
            ],
        "multi_scale_agg": [ 
            StrOptions({"last", "average",  "flatten", "auto", "sum", "concat"}),
            None
        ],
        "scales": ['array-like', StrOptions({"auto"}),  None],
        "use_residuals": [bool, Interval(Integral, 0, 1, closed="both")],
        "final_agg": [StrOptions({"last", "average",  "flatten"})],
        "mode": [
            StrOptions({'tft', 'pihal', 'tft_like', 'pihal_like',
                        "tft-like", "pihal-like"}), None
            ], 
        "objective": [
            StrOptions({'hybrid', 'transformer'}), None], 
        'use_vsn': [bool, int], 
        'use_batch_norm': [bool, int], 
        'vsn_units': [Interval(Integral, 0, None, closed="left"), None]
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_dim: int = 1,
        forecast_horizon: int = 1,
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
        use_batch_norm: bool=False, 
        apply_dtw : bool = True, 
        attention_levels : Optional [Union[str, List[str]]]=None,
        objective: str = 'hybrid',
        architecture_config: Optional[Dict] = None,
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
        self.use_batch_norm = use_batch_norm 
        self.vsn_units = (vsn_units if vsn_units is not None
                          else self.hidden_units)

        (self.quantiles, self.scales,
         self.lstm_return_sequences) = set_default_params(
            quantiles, scales, multi_scale_agg)
        self.lstm_return_sequences = True
        self.multi_scale_agg_mode = multi_scale_agg
        self.apply_dtw =apply_dtw 
        
        self.mode = mode 
        self._mode = select_mode(mode, default='pihal')
        
        # This single call handles all architectural logic.
        self.objective = objective 
        self.attention_levels = attention_levels 
        self.architecture_config = self._configure_architecture(
            objective=objective,
            use_vsn=use_vsn,
            attention_levels=attention_levels,
            architecture_config=architecture_config
        )
        # ---------------------------------------------------
        
        self._build_attentive_layers()
        
    def _configure_architecture(
        self,
        objective: Optional[str],
        use_vsn: bool,
        attention_levels: Optional[Union[str, List[str]]],
        architecture_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Initializes and validates the model's architectural configuration.
    
        This helper method centralizes the logic for setting up the
        model's internal architecture. It merges default settings with
        user-provided keyword arguments and the `architecture_config`
        dictionary, ensuring a consistent and valid final configuration.
    
        The order of precedence is:
        1. Default architecture settings.
        2. Explicit keyword arguments (`objective`, `use_vsn`, etc.).
        3. User-provided `architecture_config` dictionary (overrides others).
    
        Args:
            objective (str): The high-level objective ('hybrid' or 'transformer').
            use_vsn (bool): Whether to use Variable Selection Networks.
            attention_levels (Union[str, List[str]]): The desired attention layers.
            architecture_config (Dict, optional): A dictionary of specific
                architectural settings provided by the user.
    
        Returns:
            Dict[str, Any]: A finalized dictionary holding the complete
                            architectural configuration.
        """
        # Define the default architecture.
        final_config = {
            'encoder_type': 'hybrid',
            'decoder_attention_stack': ['cross', 'hierarchical', 'memory'],
            'feature_processing': 'vsn'
        }
    
        # 1. Apply settings from explicit keyword arguments.
        # The `objective` kwarg directly sets the `encoder_type`.
        
        final_config['encoder_type'] = select_mode(
            objective, default='hybrid', canonical=['hybrid', 'transformer']
        )
        # The `use_vsn` kwarg sets the default `feature_processing` method.
        if not use_vsn:
            final_config['feature_processing'] = 'dense'
            
        # The `attention_levels` kwarg sets the `decoder_attention_stack`.
    
        final_config['decoder_attention_stack'] = resolve_attention_levels(
            attention_levels
        )
        
        # 2. Merge and override with the user-provided dictionary.
        if architecture_config:
            user_config = architecture_config.copy()
            
            # Handle the deprecated 'objective' key for backward compatibility.
            if 'objective' in user_config:
                warnings.warn(
                    "The 'objective' key-role in `architecture_config` is"
                    " deprecated and will be rename in a future version."
                    " Please use 'encoder_type' instead.",
                    FutureWarning
                )
                # The new key takes precedence.
                user_config['encoder_type'] = user_config.pop('objective')
                
            final_config.update(user_config)
    
        # 3. Final validation and reconciliation.
        # Ensure `feature_processing` is consistent with `use_vsn`.
        if not use_vsn and final_config.get('feature_processing') == 'vsn':
            logger.info(
                "`use_vsn=False` was passed, but `architecture_config` specified"
                " `feature_processing='vsn'`. Reverting to 'dense'."
            )
            final_config['feature_processing'] = 'dense'
            
        return final_config

    def _build_attentive_layers(self):
        """
        Instantiates all shared layers for the attentive architecture.
        
        This method creates and initializes all necessary layers for the 
        model's encoder-decoder architecture, including Variable Selection 
        Networks (VSN), Gated Residual Networks (GRN), attention mechanisms, 
        and other core architectural components. The layers are instantiated 
        based on the specified configuration, including whether or not to use 
        VSN and the choice of encoder architecture (hybrid or transformer).
    
        Layers created by this method include:
        - VSN and GRN layers for static, dynamic, and future inputs.
        - Dense layers for non-VSN paths if VSN is not enabled.
        - MultiScaleLSTM and MultiHeadAttention for encoder processing, 
          depending on the chosen architecture.
        - PositionalEncoding, attention layers (cross, hierarchical, 
          memory-augmented), and fusion mechanisms for the decoder.
        - Residual connection layers for the decoder block, if enabled.
    
        Notes
        -----
        - If VSN is used, the method will create separate VSN and GRN layers 
          for static, dynamic, and future inputs. If not, it falls back to 
          non-VSN layers for processing.
        - The method sets up the attention mechanisms required for the 
          encoder-decoder interaction, including multi-head attention and 
          memory-augmented attention.
        - Multi-scale LSTM is used if the 'hybrid' architecture is chosen, 
          and transformer self-attention layers are used if the 'transformer' 
          architecture is selected.
        - The method ensures that the residual connections and normalization 
          layers are set up correctly if `use_residuals` is enabled.
    
    
        References
        ----------
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., 
          Gomez, A., Kaiser, Ł., Polosukhin, I. (2017). Attention is all 
          you need. *NeurIPS 2017*, 30, 6000-6010.
        - Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine 
          Translation by Jointly Learning to Align and Translate. *ICLR 2015*.
        """

        
        # VSN Layers
        if self.architecture_config.get('feature_processing') == 'vsn':
            if self.static_input_dim > 0:
                self.static_vsn = VariableSelectionNetwork(
                    num_inputs=self.static_input_dim,
                    units=self.vsn_units,
                    dropout_rate=self.dropout_rate,
                    name="static_vsn"
                )
                self.static_vsn_grn = GatedResidualNetwork(
                    units=self.hidden_units,
                    dropout_rate=self.dropout_rate,
                    name="static_vsn_grn"
                )
            else: 
                self.static_vsn = None
                self.static_vsn_grn = None
                
            if self.dynamic_input_dim > 0:
                self.dynamic_vsn = VariableSelectionNetwork(
                    num_inputs=self.dynamic_input_dim,
                    units=self.vsn_units,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True,
                    name="dynamic_vsn"
                )
                self.dynamic_vsn_grn = GatedResidualNetwork(
                    units=self.embed_dim,
                    dropout_rate=self.dropout_rate,
                    name="dynamic_vsn_grn"
                )
            else: 
                self.dynamic_vsn =None 
                self.dynamic_vsn_grn = None
                
            if self.future_input_dim > 0:
                self.future_vsn = VariableSelectionNetwork(
                    num_inputs=self.future_input_dim,
                    units=self.vsn_units,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True,
                    name="future_vsn"
                )
                self.future_vsn_grn = GatedResidualNetwork(
                    units=self.embed_dim,
                    dropout_rate=self.dropout_rate,
                    name="future_vsn_grn"
                )
            else: 
                self.future_vsn =None 
                self.future_vsn_grn =None
        else:
            # If not using VSNs, ensure all related attributes are None.
            self.static_vsn, self.static_vsn_grn = None, None
            self.dynamic_vsn, self.dynamic_vsn_grn = None, None
            self.future_vsn, self.future_vsn_grn = None, None

        # Shared & Non-VSN Path Layers
        # This GRN is used to process static features (if not using VSN)
        # and to refine the output of the cross-attention layer. Its
        # output dimension is set to `attention_units` for consistency
        # within the attention block.
        self.attention_processing_grn = GatedResidualNetwork(
            units=self.attention_units,
            dropout_rate=self.dropout_rate,
            activation=self.activation_fn_str,
            name="attention_processing_grn"
        )
        
        # This layer projects the combined decoder context into a
        # consistent feature space (`attention_units`) before it's used
        # in attention mechanisms and residual connections.
        self.decoder_input_projection = Dense(
            self.attention_units,
            activation=self.activation_fn_str,
            name="decoder_input_projection"
        )

        # These layers are only created if VSN is NOT used.
        if self.architecture_config.get('feature_processing') == 'dense': 
            if self.static_input_dim > 0:
                self.static_dense = Dense(
                    self.hidden_units, activation=self.activation_fn_str
                )
                # This GRN is specifically for the non-VSN static path. Its
                # dimensionality matches the static context (`hidden_units`).
                self.grn_static_non_vsn = GatedResidualNetwork(
                    units=self.hidden_units, 
                    dropout_rate=self.dropout_rate,
                    activation=self.activation_fn_str,
                    name="grn_static_non_vsn"
                )
            else:
                self.static_dense = None
                self.grn_static_non_vsn = None
        
            # Create dense layers for dynamic and future features
            # for non-VSN path
            self.dynamic_dense = Dense(self.embed_dim)
            self.future_dense = Dense(self.embed_dim)
        else: 
            self.static_dense =None 
            self.grn_static_non_vsn = None
            self.dynamic_dense =None
            self.future_dense = None


        # Encoder-specific Layers
        if self.architecture_config['encoder_type'] == 'hybrid':
            self.multi_scale_lstm = MultiScaleLSTM(
            lstm_units=self.lstm_units,
            scales=self.scales,
            return_sequences=True  # Critical for the encoder path
            )
            self.encoder_self_attention = None
            
        elif self.architecture_config['encoder_type'] == 'transformer':
            self.encoder_self_attention = [
                (MultiHeadAttention(
                    num_heads=self.num_heads, 
                    key_dim=self.attention_units),
                 LayerNormalization()) for _ in range(
                     self.num_encoder_layers)
            ]
            self.multi_scale_lstm = None

        # Core Architectural Layers
        # Create two separate instances of PositionalEncoding 
        self.encoder_positional_encoding = PositionalEncoding(
            name="encoder_pos_encoding"
            )
        self.decoder_positional_encoding = PositionalEncoding(
            name="decoder_pos_encoding"
            )

        self.hierarchical_attention = HierarchicalAttention(
            units=self.attention_units,
            num_heads=self.num_heads
        )
        self.cross_attention = CrossAttention(
            units=self.attention_units, 
            num_heads=self.num_heads
        )
        self.memory_augmented_attention = MemoryAugmentedAttention(
            units=self.attention_units,
            memory_size=self.memory_size,
            num_heads=self.num_heads
        )
        self.multi_resolution_attention_fusion = \
            MultiResolutionAttentionFusion(
                units=self.attention_units,
                num_heads=self.num_heads
            )
        self.dynamic_time_window = DynamicTimeWindow(
            max_window_size=self.max_window_size
        )
        self.multi_decoder = MultiDecoder(
            output_dim=self.output_dim,
            num_horizons=self.forecast_horizon
        )
        self.quantile_distribution_modeling = QuantileDistributionModeling(
            quantiles=self.quantiles,
            output_dim=self.output_dim
        )

        # --- 4. Layers for Residual Connections (Conditional) ---
        # Instantiate Add and LayerNormalization layers here to avoid
        # re-creation inside the `call` method, which is incompatible
        # with tf.function.
        if self.use_residuals:
            self.residual_dense = Dense(self.attention_units)
            # Layers for the first residual connection in the decoder
            self.decoder_add_norm = [Add(), LayerNormalization()]
            # Layers for the final residual connection
            self.final_add_norm = [Add(), LayerNormalization()]
        else:
            self.residual_dense = None
            self.decoder_add_norm = None
            self.final_add_norm = None

    def run_encoder_decoder_core(
        self,
        static_input: Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        training: bool
    ) -> Tensor:
        """
        Executes the data-driven pipeline with a selectable encoder 
        architecture, processing static, dynamic, and future inputs through 
        the encoder-decoder interaction. Attention mechanisms are applied in 
        the decoder block, with flexibility to select which types of attention 
        to use via the `att_levels` parameter.
        
        Parameters
        ----------
        static_input : Tensor
            The input tensor containing static features, which remain constant 
            over time (e.g., environmental data, geographical features).
            
        dynamic_input : Tensor
            The input tensor containing dynamic features, which vary over time 
            (e.g., sensor readings, time-series data).
        
        future_input : Tensor
            The input tensor representing future features, typically used for 
            forecasting or projection purposes.
            
        training : bool
            A flag indicating whether the model is in training mode. This flag 
            controls the use of training-specific operations, such as dropout 
            and batch normalization.
        
        Returns
        -------
        Tensor
            The final output tensor, which has undergone attention fusion and 
            time-based aggregation. This tensor is used for further tasks such 
            as classification, regression, or forecasting.
        
        Notes
        -----
        - The method processes static, dynamic, and future inputs through 
          separate paths before combining them for the encoder.
        - Attention mechanisms are applied in the decoder block. The 
          specific attention types and their order are controlled via the 
          `att_levels` parameter, which can include:
            - 'cross' for cross attention.
            - 'hierarchical' for hierarchical attention.
            - 'memory' for memory-augmented attention.
        - If multiple attention mechanisms are chosen, they are applied 
          sequentially.
        - The time dimension is collapsed in the final output, resulting 
          in a single vector per sample.
        
        References
        ----------
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., 
          Gomez, A., Kaiser, Ł., Polosukhin, I. (2017). Attention is all you 
          need. *NeurIPS 2017*, 30, 6000-6010.
          
        - Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine 
          Translation by Jointly Learning to Align and Translate. *ICLR 2015*.
        """

        time_steps = tf_shape(dynamic_input)[1]
        
        # 1. Initial Feature Processing
        static_context, dyn_proc, fut_proc = None, dynamic_input, future_input
        
        # 1. Initial Feature Processing
        if self.architecture_config.get('feature_processing') == 'vsn':
            if self.static_vsn is not None:
                vsn_static_out = self.static_vsn(
                    static_input, training=training)
                static_context = self.static_vsn_grn(
                    vsn_static_out, training=training)
            if self.dynamic_vsn is not None:
                dyn_context = self.dynamic_vsn(
                    dynamic_input, training=training 
                    )
                dyn_proc = self.dynamic_vsn_grn(
                    dyn_context, training=training
                )
            if self.future_vsn is not None:
                fut_context = self.future_vsn(
                    future_input, training=training 
                )
                fut_proc = self.future_vsn_grn(
                    fut_context,  training=training
                )
                
        else: # Non-VSN path
            if self.static_dense is not None:
                processed_static = self.static_dense(static_input)
                # Note: here the GRN's output dim might differ from the
                # VSN path. This is handled by the decoder_input_projection.
                static_context = self.grn_static_non_vsn(
                    processed_static, training=training) 
                
            dyn_proc = self.dynamic_dense(dynamic_input)
            fut_proc = self.future_dense(future_input)

        logger.debug(f"Shape after VSN/initial processing: "
                     f"Dynamic={getattr(dyn_proc, 'shape', 'N/A')}, "
                     f"Future={getattr(fut_proc, 'shape', 'N/A')}")
        
        # 2. Encoder Path
        encoder_input_parts = [dyn_proc]
        if self._mode == 'tft_like':
            # For TFT mode, slice historical part of future features
            # and add to the encoder input.
            fut_enc_proc = fut_proc[:, :time_steps, :]
            encoder_input_parts.append(fut_enc_proc)
        
        encoder_raw = tf_concat(encoder_input_parts, axis=-1)
        encoder_input = self.encoder_positional_encoding(encoder_raw)

        if self.architecture_config['encoder_type'] == 'hybrid':
            lstm_out = self.multi_scale_lstm(
            encoder_input, training=training 
            )
            encoder_sequences = aggregate_multiscale_on_3d(
                lstm_out, mode='concat')
            
        else: # transformer
            encoder_sequences = encoder_input
            for mha, norm in self.encoder_self_attention:
                attn_out = mha(encoder_sequences, encoder_sequences)
                encoder_sequences = norm(encoder_sequences + attn_out)
        
        if self.apply_dtw: 
            if self.dynamic_time_window is not None:
                encoder_sequences = self.dynamic_time_window(
                    encoder_sequences, training=training
                    )
        
        logger.debug(f"Encoder sequences shape: {encoder_sequences.shape}")
        
        # 3. Decoder Path
        if self._mode == 'tft_like':
            # For TFT mode, slice the forecast part of future features.
            fut_dec_proc = fut_proc[:, time_steps:, :]
        else: # For pihal_like mode, use the whole future tensor.
            fut_dec_proc = fut_proc
        
        decoder_parts = []
        if static_context is not None:
            static_expanded = tf_expand_dims(static_context, 1)
            static_expanded = tf_tile(
                static_expanded, [1, self.forecast_horizon, 1])
            decoder_parts.append(static_expanded)
        
        if self.future_input_dim > 0:
            future_with_pos = self.decoder_positional_encoding(
                fut_dec_proc)
            decoder_parts.append(future_with_pos)


        if not decoder_parts:
            batch_size = tf_shape(dynamic_input)[0]
            raw_decoder_input = tf_zeros(
                (batch_size, self.forecast_horizon, self.attention_units))
        else:
            raw_decoder_input = tf_concat(decoder_parts, axis=-1)
            
        # Project the raw decoder input to a consistent feature dimension.
        projected_decoder_input = self.decoder_input_projection(
            raw_decoder_input)
        logger.debug(f"Projected decoder input shape: "
                     f"{projected_decoder_input.shape}")

        # 4. Attention Fusion & Final Processing
        # --- 4. Attention-based Fusion (Encoder-Decoder Interaction) ---
        final_features = self.apply_attention_levels(
            projected_decoder_input, encoder_sequences, 
            training=training, 
            # att_levels= self.architecture_config['decoder_attention_stack'] 
        )
       
        logger.debug(f"Shape after final fusion: {final_features.shape}")
    
        # Collapse the time dimension to get a single vector per sample.
        return aggregate_time_window_output(final_features, self.final_agg)
    
    def apply_attention_levels(
        self,
        projected_decoder_input: Tensor,
        encoder_sequences: Tensor,
        training: bool,
        # att_levels: Union[str, List[str], int, None]
    ) -> Tensor:
        """
        Applies attention mechanisms in the order specified by `att_levels`,
        using the provided attention methods such as cross attention,
        hierarchical attention, and memory-augmented attention.
    
        Parameters
        ----------
        projected_decoder_input : Tensor
            The input tensor to be used in the attention mechanisms.
        
        encoder_sequences : Tensor
            The encoder output sequences used in attention.
        
        training : bool
            A flag indicating whether the model is in training mode.
        
        att_levels : str, list of str, int, or None
            Specifies the attention mechanisms to apply and the order:
            - If None or 'use_all' or '*', use all attention mechanisms.
            - If 'hier_att' or 'hierarchical_attention', apply
              hierarchical attention.
            - If 'memo_aug_att' or 'memory_augmented_attention', 
              apply memory-augmented attention.
            - If a list of strings, apply attention types in the provided order.
            - If an integer (1, 2, 3), map it to cross attention (1), 
              hierarchical attention (2),
              or memory-augmented attention (3).
    
        Returns
        -------
        Tensor
            The final output tensor after applying attention mechanisms in order.
    
        Notes
        -----
        The order of attention mechanisms is determined by the provided
        `att_levels` list.
        """
        
        # resolve attention levels 
        # self._attention_levels = self._attention_levels or resolve_attention_levels(
        #     att_levels or self.architecture_config['decoder_attention_stack']) 
        
        # Step 4: Attention Fusion (Encoder-Decoder Interaction)
        
        if 'cross' in self.architecture_config['decoder_attention_stack'] :
            cross_att_out = self.cross_attention(
                [projected_decoder_input, encoder_sequences], 
                training=training)
    
            att_proc = self.attention_processing_grn(
                cross_att_out, training=training)
    
            # Apply residual connection if enabled
            if self.use_residuals and self.decoder_add_norm is not None:
                context_att = self.decoder_add_norm[0](
                    [projected_decoder_input, att_proc])
                context_att = self.decoder_add_norm[1](
                    context_att)
            else:
                context_att = att_proc
    
        else:
            # If cross attention is not in the list, initialize context_att
            context_att = projected_decoder_input
    
        # Apply hierarchical attention if needed
        if 'hierarchical' in self.architecture_config['decoder_attention_stack']:
            hierarchical_att_output = self.hierarchical_attention(
                [context_att, context_att],
                training=training
            )
        else:
            # If no hierarchical attention, pass through
            hierarchical_att_output = context_att  
    
        # Apply memory-augmented attention if needed
        if 'memory' in self.architecture_config['decoder_attention_stack']:
            memory_attention_output = self.memory_augmented_attention(
                hierarchical_att_output, 
                training=training
            )
        else:
            # If no memory attention, pass through
            memory_attention_output = hierarchical_att_output  
    
        # Apply final fusion using multi-resolution attention fusion
        final_features = self.multi_resolution_attention_fusion(
            memory_attention_output, 
            training=training
        )
    
        # Apply final residual connection and normalization if enabled
        if self.use_residuals and self.final_add_norm is not None:
            # The residual_base must have the same dimension as final_features
            res_base = self.residual_dense(context_att)
            final_features = self.final_add_norm[0]([final_features, res_base])
            final_features = self.final_add_norm[1](final_features)
    
        return final_features


    def call(self, inputs, training=False):
        """
        Forward pass for the attentive model.
        
        This method processes the input data, validates the dimensions, 
        and then performs the forward pass through the encoder-decoder 
        network. The model applies attention mechanisms in the decoder 
        phase and performs quantile distribution modeling if enabled.
    
        Parameters
        ----------
        inputs : Tensor
            A tensor containing the input data. It includes the static, 
            dynamic, and future covariate features required for the model.
        
        training : bool, optional, default=False
            A flag indicating whether the model is in training mode. 
            This flag controls operations such as dropout and batch 
            normalization.
    
        Returns
        -------
        Tensor
            The final output tensor after passing through the model, 
            which may include quantile distribution modeling depending 
            on the configuration of the model.
        
        Notes
        -----
        - The method first validates the input dimensions for static, 
          dynamic, and future features using `validate_model_inputs`.
        - The model then asserts that the future input tensor has the 
          correct time span using `tf_assert_equal`.
        - The forward pass is completed by invoking the encoder-decoder 
          core method (`run_encoder_decoder_core`), followed by the 
          multi-decoder and quantile distribution modeling (if enabled).
    
        References
        ----------
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., 
          Gomez, A., Kaiser, Ł., Polosukhin, I. (2017). Attention is all 
          you need. *NeurIPS 2017*, 30, 6000-6010.
        - Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine 
          Translation by Jointly Learning to Align and Translate. 
          *ICLR 2015*.
        """
        # `validate_model_inputs` can provide a secondary, more detailed
        # check on the unpacked feature tensors.
        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=inputs,
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            #forecast_horizon=self.forecast_horizon,
            mode='strict',
            verbose=0 # Set to 1 for more detailed logging from validator
        )
        logger.debug(
            "Input shapes after validation:"
            f" S={getattr(static_p, 'shape', 'None')}, "
            f"D={getattr(dynamic_p, 'shape', 'None')},"
            f" F={getattr(future_p, 'shape', 'None')}"
        )
        
        # ***  Validate future_p shape based on mode ***
        if self.mode == 'tft_like':
            expected_future_span = self.max_window_size + self.forecast_horizon
        else:  # pihal_like
            expected_future_span = self.forecast_horizon

        actual_future_span = tf_shape(future_p)[1]
        expected_span_tensor = tf_convert_to_tensor(
            expected_future_span, dtype=actual_future_span.dtype)
        
        tf_assert_equal(
            actual_future_span, expected_span_tensor,
            message=(
                f"Incorrect 'future_features' tensor length for "
                f"mode='{self.mode}'. Expected time dimension of "
                f"{expected_future_span}, but got {actual_future_span}."
            )
        )
        
        final_features = self.run_encoder_decoder_core(
            static_input=static_p, 
            dynamic_input=dynamic_p, 
            future_input=future_p, 
            training=training
            )
        # Get mean predictions from the multi-horizon decoder (usefull for PDE) 
        self._decoded_outputs = self.multi_decoder(
            final_features, training=training
        )
        logger.debug(
            "Shape of decoded outputs (means):"
            f" {self._decoded_outputs.shape}")
        
        # Get final predictions (potentially with quantiles, for data loss)
        predictions_final_targets = self._decoded_outputs
        if self.quantiles is not None:
            predictions_final_targets = self.quantile_distribution_modeling(
                self._decoded_outputs, training=training
            )
       
        logger.debug(
            "Shape of final quantile outputs:"
            f" {predictions_final_targets.shape}"
        )
        return predictions_final_targets

    def get_config(self):
        """
        Returns the configuration of the model as a dictionary.
        
        This method retrieves the configuration of the model, 
        including all the hyperparameters and settings that define
        the model's behavior. The returned dictionary can be used for 
        saving, reproducing, or inspecting the model's configuration.
    
        The method overrides the default `get_config` method from 
        the parent class and includes specific attributes of the 
        `BaseAttentive` model, such as the input dimensions, architecture
        type, attention mechanisms, and regularization settings. The 
        configuration can be serialized and used to recreate the model 
        with the same parameters.
        """
        config = super().get_config()
        config.update({
            "static_input_dim": self.static_input_dim,
            "dynamic_input_dim": self.dynamic_input_dim,
            "future_input_dim": self.future_input_dim,
            "output_dim": self.output_dim,
            "forecast_horizon": self.forecast_horizon,
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
            "objective": self.objective, 
            "use_vsn": self.use_vsn,
            "vsn_units": self.vsn_units,
            "apply_dtw": self.apply_dtw, 
            "attention_levels": self.attention_levels, 
            "use_batch_norm": self.use_batch_norm, 
            "architecture_config": self.architecture_config,
            "name": self.name 
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Creates a model from its config.
        
        This method is the reverse of get_config, capable of handling
        the nested architecture_config dictionary.
        """
        # Separate architecture_config from the main config
        arch_config = config.pop("architecture_config", None)
        # Re-add it as a keyword argument for __init__
        return cls(**config, architecture_config=arch_config)
    
    def reconfigure(
        self,
        architecture_config: Dict[str, Any]
    ) -> "BaseAttentive":
        """Creates a new model instance with a modified architecture.
    
        This method takes the configuration of the current model, updates
        the architectural components with the provided dictionary, and
        returns a new, un-trained model instance with the specified
        changes.
    
        Parameters 
        ------------
        architecture_config (Dict[str, Any]):
            A dictionary with new architectural settings, such as
            {'encoder_type': 'transformer'}.
    
        Returns
        ----------
        BaseAttentive:
            A new model instance with the updated architecture.
        """
        # 1. Get the full configuration of the existing model
        config = self.get_config()
        
        # 2. Update the architecture configuration
        # get_config will have stored it as a nested dictionary
        config['architecture_config'].update(architecture_config)
        
        # 3. Create a new model from the modified config
        return self.__class__.from_config(config)
    

BaseAttentive.__doc__ = r"""
Base Attentive Model.

A foundational blueprint for building powerful, data-driven,
sequence-to-sequence time series forecasting models.

This class provides a sophisticated and highly configurable
encoder-decoder architecture. It is designed to process three
distinct types of inputs—static, dynamic past, and known future
features—and fuse them using a modular stack of attention
mechanisms. It serves as the core engine for models like ``HALNet``
and ``PIHALNet``.

A **data-driven** model architecture that can be used for both hybrid 
and transformer-based forecasting models. This model processes static, 
dynamic, and future input features through separate paths and applies 
multi-head attention mechanisms in the decoder block to produce forecasts. 
The model supports multi-horizon forecasting, uncertainty quantification 
using quantiles, and dynamic time warping (DTW) for time-series alignment.

The model offers flexibility through various options for configuration, 
residual connections, and feature selection mechanisms, making it suitable 
for both statistical and physics-informed settings.

The architecture can be configured to operate as a hybrid model,
combining the temporal feature extraction power of LSTMs with
attention, or as a pure transformer model.

See more in :ref:`User Guide <user_guide>`.

Parameters
----------
{params.base.static_input_dim}
{params.base.dynamic_input_dim}
{params.base.future_input_dim}

output_dim : int, default 1  
    Number of target variables produced at each forecast step. The model 
    outputs a tensor of shape :math:`(B, \, H, \, Q, \, \text{{output\_dim}})` 
    when *quantiles* are provided, or :math:`(B, \, H, \, \text{{output\_dim}})` 
    for point forecasts, where  

    .. math::  
       B = \text{{batch size}},\qquad  
       H = \text{{forecast horizon}},\qquad  
       Q = |\text{{quantiles}}|.  

forecast_horizon : int, default 1  
    Length of the prediction window into the future. The dynamic encoder 
    ingests *max_window_size* past steps and the decoder emits :math:`H` 
    steps ahead, where :math:`H=\text{{forecast_horizon}}`. Setting :math:`H > 1` 
    enables multi‑horizon sequence‑to‑sequence forecasts.  

mode : {{'pihal_like', 'tft_like'}}, default 'tft_like'  
    Controls how *future_features* are sliced and routed.  
    ``'pihal_like'`` expects ``future_input.shape[1] == forecast_horizon`` 
    and feeds the tensor only to the decoder.  
    ``'tft_like'`` expects ``time_steps + forecast_horizon`` rows, 
    sending the first *time_steps* rows to the encoder and the remaining 
    rows to the decoder, emulating the Temporal Fusion Transformer.
    
num_encoder_layers : int, default=2
    The number of self-attention blocks to stack in the encoder when
    using the `'transformer'` architecture.
    
quantiles : list[float] or None, default None  
    Optional quantile levels :math:`0 < q_1 < \dots < q_Q < 1`. When supplied, 
    a :class:`fusionlab.nn.components.QuantileDistributionModeling` head scales 
    the point forecast :math:`\hat{{y}}` into quantile estimates  

    .. math::  
       \hat{{y}}^{{(q)}} = \hat{{y}} + \sigma \,\Phi^{{-1}}(q),  

    where :math:`\sigma` is a learned spread parameter and :math:`\Phi^{{-1}}` 
    is the probit function. Omit or set to *None* to obtain deterministic forecasts.  
    
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

use_batch_norm : bool, default=False
    If ``True``, applies batch normalization.
    
apply_dtw : bool, default True  
    Whether to apply **Dynamic Time Warping (DTW)** for time-series alignment.  
    DTW is a technique used to align sequences that may be misaligned 
    in time. It is particularly useful when the time steps in the dynamic 
    and future features are not synchronized. Setting this to **True** 
    enables DTW, while setting it to **False** disables it.
    If ``True``, applies a `DynamicTimeWindow` layer to the encoder
    output, allowing the model to learn an optimal, data-dependent
    lookback window.
    
attention_levels : str or list[str], optional
    Legacy parameter. Controls the attention layers used in the
    decoder. It is recommended to use
    `architecture_config={{'decoder_attention_stack': [...]}}` instead.

objective : str, default 'hybrid' 
    Legacy parameter. Defines the underlying architecture of the model. 
    The configuration  can be either 'hybrid' 
    (combining LSTM and attention mechanisms) or 'transformer' 
    (using only transformer-based attention mechanisms).It is
    recommended to use `architecture_config={{'encoder_type': 'hybrid'}}`
    instead.
    
architecture_config : dict, optional
    A dictionary for fine-grained control over the model's internal
    architecture. This is the recommended way to configure the model.
    See the Notes section for details on keys like ``encoder_type``,
    ``decoder_attention_stack``, and ``feature_processing``.
    
name : str, default "BaseAttentiveModel"  
    Model identifier passed to :pyclass:`tf.keras.Model`. Appears in weight 
    filenames and TensorBoard scopes.  

**kwargs  
    Additional keyword arguments forwarded verbatim to the 
    :pyclass:`tf.keras.Model` constructor—e.g. ``dtype="float64"`` or 
    ``run_eagerly=True``.

Notes  
-----
- The composite latent size produced by the cross‑attention block is 
  :math:`d_\text{{model}} = \text{{attention\_units}}`. For stable training, 
  ensure :math:`d_\text{{model}}` is divisible by *num_heads*.
  
- The model configuration supports both hybrid and transformer-based designs. 
  The hybrid configuration combines LSTM with attention mechanisms, while 
  the transformer configuration exclusively uses self-attention mechanisms.
  
- The attention mechanism allows for both cross-attention (between encoder 
  and decoder) and self-attention within the decoder.


See Also  
--------
* :class:`fusionlab.nn.pinn.PIHALNet` – physics-informed extension.  
* :func:`fusionlab.utils.data_utils.widen_temporal_columns` – prepares 
  wide data frames for plotting forecasts.

**Smart Configuration**

The recommended way to define the model's structure is via the
``architecture_config`` dictionary. It provides clear, explicit
control over the most important architectural choices:

* **`encoder_type`**: Defines the encoder's core mechanism.
    * ``'hybrid'`` (default): Uses the ``MultiScaleLSTM`` for rich
      temporal feature extraction.
    * ``'transformer'``: Uses a pure self-attention stack, ideal for
      capturing very long-range dependencies.

* **`decoder_attention_stack`**: A ``list`` of strings that defines
    the sequence of attention layers in the decoder. The available
    layers are:
    * ``'cross'``: The crucial cross-attention between decoder
      queries and encoder memory.
    * ``'hierarchical'``: A self-attention layer that helps find
      structural patterns in the context.
    * ``'memory'``: A memory-augmented self-attention layer for
      long-term dependencies.
    * Example: ``['cross', 'hierarchical']`` creates a simpler decoder.

* **`feature_processing`**: Controls the initial feature embedding.
    * ``'vsn'`` (default): Uses ``VariableSelectionNetwork`` for
      learnable feature selection.
    * ``'dense'``: Uses standard ``Dense`` layers.

The legacy parameters (`objective`, `use_vsn`, `attention_levels`)
are maintained for backward compatibility but will be overridden by
any settings provided in ``architecture_config``.

Examples
--------
>>> from fusionlab.nn.models._base_attentive import BaseAttentive  
>>> model = BaseAttentive(  
...     static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,  
...     output_dim=2, forecast_horizon=24, quantiles=[0.1, 0.5, 0.9],  
...     scales=[1, 3], multi_scale_agg="concat", final_agg="last",  
...     attention_units=64, num_heads=8, dropout_rate=0.15,  
... )  
>>> x_static  = tf.random.normal([32, 4])              # B × S  
>>> x_dynamic = tf.random.normal([32, 10, 8])          # B × T × D  
>>> x_future  = tf.random.normal([32, 24, 6])          # B × H × F  
>>> y_hat = model( [x_static, x_dynamic,  x_future, ]
... )  
>>> y_hat.shape  
TensorShape([32, 24, 3, 2])  # B × H × Q × output_dim

>>> from fusionlab.nn.models import BaseAttentive
>>> import tensorflow as tf

>>> # Example using the recommended architecture_config
>>> transformer_config = {{
...     'encoder_type': 'transformer',
...     'decoder_attention_stack': ['cross', 'hierarchical'],
...     'feature_processing': 'dense'
... }}
>>> model = BaseAttentive(
...     static_input_dim=4,
...     dynamic_input_dim=8,
...     future_input_dim=6,
...     output_dim=2,
...     forecast_horizon=24,
...     max_window_size=10,
...     mode='tft_like',
...     quantiles=[0.1, 0.5, 0.9],
...     architecture_config=transformer_config
... )

>>> # Prepare dummy input data
>>> BATCH_SIZE = 32
>>> x_static  = tf.random.normal([BATCH_SIZE, 4])
>>> x_dynamic = tf.random.normal([BATCH_SIZE, 10, 8])
>>> x_future  = tf.random.normal([BATCH_SIZE, 10 + 24, 6])

>>> # Get model output
>>> y_hat = model([x_static, x_dynamic, x_future])
>>> y_hat.shape
TensorShape([32, 24, 3, 2])

See Also
--------
fusionlab.nn.pinn.PIHALNet
    A physics-informed extension of this architecture.
fusionlab.nn.components.MultiScaleLSTM
    The multi-resolution LSTM component used in the hybrid encoder.
fusionlab.nn.components.VariableSelectionNetwork
    The learnable feature-selection component.
fusionlab.nn.models.HALNet
    A direct, data-driven implementation of ``BaseAttentive``.

References  
----------
.. [1] Vaswani et al., “Attention Is All You Need,” *NeurIPS 2017*.  
.. [2] Lim et al., “Temporal Fusion Transformers for Interpretable  
       Multi‑Horizon Time Series Forecasting,” *IJCAI 2021*.  
""".format(params=_param_docs)
