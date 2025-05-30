# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Hybrid Attentive LSTM Network (HAL-Net), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""

from textwrap import dedent 
from numbers import Real, Integral  
from typing import List, Optional, Union, Dict, Any  
import numpy as np 

from .._fusionlog import fusionlog, OncePerMessageFilter
from ..api.docs import _shared_docs, doc 
from ..api.property import NNLearner 
from ..compat.sklearn import validate_params, Interval, StrOptions 
from ..core.handlers import param_deprecated_message 
from ..utils.deps_utils import ensure_pkg
# from ..decorators import Appender

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
 
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
    
    tf_autograph=KERAS_DEPS.autograph
    tf_autograph.set_verbosity(0)
    
    from ..compat.tf import optional_tf_function 
    from ._tensor_validation import validate_anomaly_scores 
    from ._tensor_validation import validate_model_inputs
    from ._tensor_validation import validate_anomaly_config 
    from ._tensor_validation import align_temporal_dimensions
    
    from .losses import ( 
        combined_quantile_loss, 
        combined_total_loss, 
        prediction_based_loss
    )
    from .utils import set_default_params
    from .components import (
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
            # VariableSelectionNetwork,
            PositionalEncoding, 
            aggregate_multiscale, 
            aggregate_time_window_output
        )
    
DEP_MSG = dependency_message('nn.transformers') 

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())


@register_keras_serializable('fusionlab.nn.transformers', name="HALNet")
@doc (
    key_improvements= dedent(_shared_docs['xtft_key_improvements']), 
    key_functions= dedent(_shared_docs['xtft_key_functions']), 
    methods= dedent( _shared_docs['xtft_methods']
    )
 )
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
    """ Hybrid Attentive LSTM Network (HAL-Net) """
    @validate_params({
        "static_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "future_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "embed_dim": [Interval(Integral, 1, None, closed='left')],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')], 
        "quantiles": ['array-like', StrOptions({'auto'}),  None],
        "max_window_size": [Interval(Integral, 1, None, closed='left')],
        "memory_size": [Interval(Integral, 1, None, closed='left')], 
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
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
        "use_batch_norm": [bool,  Interval(Integral, 0, 1, closed="both")],
        "use_residuals": [bool, Interval(Integral, 0, 1, closed="both")],
        "final_agg": [StrOptions({"last", "average",  "flatten"})],
        "anomaly_detection_strategy": [
            StrOptions({"prediction_based", "feature_based", "from_config"}), 
            None
        ],
        'anomaly_loss_weight': [Real]
      },
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        embed_dim: int = 32,
        forecast_horizon: int = 1,
        quantiles: Union[str, List[float], None] = None,
        max_window_size: int = 10,
        memory_size: int = 100,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        output_dim: int = 1, 
        attention_units: int = 32,
        hidden_units: int = 64,
        lstm_units: int = 64,
        scales: Union[str, List[int], None] = None,
        multi_scale_agg: Optional[str] = None, 
        activation: str = 'relu',
        use_residuals: bool = True,
        use_batch_norm: bool = False,
        final_agg: str = 'last',
        anomaly_config: Optional[Dict[str, Any]] = None,  
        anomaly_detection_strategy: Optional[str] = None,
        anomaly_loss_weight: float =.1, 
        **kw, 
    ):
        super().__init__(**kw)

        self.activation = Activation(activation).activation_str 
        
        logger.debug(
            "Initializing HALNet with parameters: "
            f"static_input_dim={static_input_dim}, "
            f"dynamic_input_dim={dynamic_input_dim}, "
            f"future_input_dim={future_input_dim}, "
            f"embed_dim={embed_dim}, "
            f"forecast_horizon={forecast_horizon}, "
            f"quantiles={quantiles}, "
            f"max_window_size={max_window_size},"
            f" memory_size={memory_size}, num_heads={num_heads}, "
            f"dropout_rate={dropout_rate}, output_dim={output_dim}, "
            f"attention_units={attention_units}, "
            f" hidden_units={hidden_units}, "
            f"lstm_units={lstm_units}, "
            f"scales={scales}, "
            f"activation={self.activation}, "
            f"use_residuals={use_residuals}, "
            f"use_batch_norm={use_batch_norm}, "
            f"final_agg={final_agg}"
        )
        # Handle default quantiles, scales and multi_scale_agg 
        quantiles, scales, return_sequences = set_default_params(
            quantiles, scales, multi_scale_agg ) 
        
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.embed_dim = embed_dim
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.max_window_size = max_window_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        self.attention_units = attention_units
        self.hidden_units = hidden_units
        self.lstm_units = lstm_units
        self.scales = scales
        self.multi_scale_agg = multi_scale_agg
        self.use_residuals = use_residuals
        self.use_batch_norm = use_batch_norm
        self.final_agg = final_agg
        self.anomaly_detection_strategy=anomaly_detection_strategy 
        self.anomaly_loss_weight=anomaly_loss_weight

        # Layers
        
        self.multi_modal_embedding = MultiModalEmbedding(embed_dim)
        
        # Add PositionalEncoding layer
        self.positional_encoding = PositionalEncoding()
        
        self.multi_scale_lstm = MultiScaleLSTM(
            lstm_units=self.lstm_units,
            scales=self.scales,
            return_sequences=return_sequences
        )

        self.hierarchical_attention = HierarchicalAttention(
            units=attention_units,
            num_heads=num_heads
        )
        self.cross_attention = CrossAttention(
            units=attention_units,
            num_heads=num_heads
        )
        self.memory_augmented_attention = MemoryAugmentedAttention(
            units=attention_units,
            memory_size=memory_size,
            num_heads=num_heads
        )
        self.multi_decoder = MultiDecoder(
            output_dim=output_dim,
            num_horizons=forecast_horizon
        )
        self.multi_resolution_attention_fusion = MultiResolutionAttentionFusion(
            units=attention_units,
            num_heads=num_heads
        )
        self.dynamic_time_window = DynamicTimeWindow(
            max_window_size=max_window_size
        )
        self.quantile_distribution_modeling = QuantileDistributionModeling(
            quantiles=self.quantiles,
            output_dim=output_dim
        )

        # Validate anomaly configuration
        self.anomaly_config, self.anomaly_detection_strategy,\
            self.anomaly_loss_weight =validate_anomaly_config(
                anomaly_config=anomaly_config,
                forecast_horizon= self.forecast_horizon, 
                strategy=anomaly_detection_strategy,
                default_anomaly_loss_weight=self.anomaly_loss_weight, 
                return_loss_weight=True
            )

        logger.debug(
            f"anomaly_config={self.anomaly_config.keys()}, "
            f"anomaly_detection_strategy={anomaly_detection_strategy}"
            f"anomaly_loss_weight={anomaly_loss_weight}"
        )
        # Initialize/Fetch anomaly scores 
        self.anomaly_scores = self.anomaly_config.get('anomaly_scores')
            
        # Anomaly scores handling
        self.anomaly_loss_layer = AnomalyLoss(
            weight=self.anomaly_loss_weight
        )

        # Initialize anomaly detection layers
        if self.anomaly_detection_strategy == 'feature_based':
            self._init_feature_based_components()
        # ---------------------------------------------------------------------
        # The MultiObjectiveLoss encapsulates both quantile and anomaly losses
        # to allow simultaneous training on multiple objectives. While this 
        # functionality can currently be bypassed, note that it may be removed 
        # in a future release. Users who rely on multi-objective training 
        # strategies should keep an eye on upcoming changes.
        # 
        # Here, we instantiate the MultiObjectiveLoss with an adaptive quantile 
        # loss function, which adjusts quantile estimates dynamically based on 
        # the provided quantiles, and an anomaly loss function that penalizes 
        # predictions deviating from expected anomaly patterns.
        # ---------------------------------------------------------------------
        self.multi_objective_loss = MultiObjectiveLoss(
            quantile_loss_fn=AdaptiveQuantileLoss(self.quantiles),
            anomaly_loss_fn=self.anomaly_loss_layer
        )
        # ---------------------------------------------------------------------
        self.learned_normalization = LearnedNormalization()
        self.static_dense = Dense(hidden_units, activation=self.activation)
        self.static_dropout = Dropout(dropout_rate)
        if self.use_batch_norm:
            self.static_batch_norm = LayerNormalization()
            
        # Initialize Gated Residual Networks (GRNs) for attention outputs
        self.grn_static = GatedResidualNetwork(
            units=hidden_units,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm
        )
    
        self.residual_dense = Dense(2 * embed_dim) if use_residuals else None

    def _init_feature_based_components(self):
        """
        Initializes architecture components for feature-based
        anomaly detection.
        
        Creates:
        1. Anomaly Attention: Multi-head attention layer to identify
           unusual patterns in feature relationships
        2. Anomaly Projection: Dense layer to project the anomaly 
           attention output to the desired dimension.
        3. Anomaly Scorer: Dense layer to convert the projected features
           outputs to anomaly scores
  
        Design Rationale:
        - key_dim aligns with hidden_units for dimension compatibility
        - Single attention head focuses on global anomaly patterns
        - Linear activation preserves relative magnitude of anomaly scores
        """
        self.anomaly_attention = MultiHeadAttention(
            num_heads=1, 
            key_dim=self.hidden_units,  
            name='anomaly_attention'
        )        
        # Projection layer to dynamically adjust the dimension.
        self.anomaly_projection = Dense(
            self.hidden_units, activation='linear', 
            name='anomaly_projection'
        )
        self.anomaly_scorer = Dense(
            1, activation='linear', 
            name='anomaly_scorer'
        )

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False, **kwargs):
        """
        Forward pass of the HALNet model.
    
        Parameters
        ----------
        inputs : tuple or list
            Input data containing three elements:
            1. Static features (batch_size, static_input_dim)
            2. Dynamic historical features (batch_size, time_steps, dynamic_input_dim)
            3. Future covariates (batch_size, horizon, future_input_dim)
        training : bool, optional
            Whether the model is in training mode, by default False
        **kwargs
            Additional keyword arguments
    
        Returns
        -------
        tf.Tensor
            Predictions tensor of shape:
            - (batch_size, horizon, len(quantiles)) if quantiles specified
            - (batch_size, horizon, output_dim) otherwise
    
        Raises
        ------
        ValueError
            If input validation fails through validate_xtft_inputs
    
        Notes
        -----
        - Handles three types of anomaly detection strategies:
          1. 'feature_based': Generates scores from attention mechanisms
          2. 'prediction_based': Handled in loss function
          3. 'from_config': Uses precomputed anomaly scores
        - Implements multi-scale temporal processing with:
          - Positional encoding
          - Hierarchical attention
          - Memory-augmented attention
          - Dynamic time windowing
        """
        static_input , dynamic_input, future_input = validate_model_inputs (
            inputs =inputs,
            static_input_dim=self.static_input_dim, 
            dynamic_input_dim= self.dynamic_input_dim, 
            future_covariate_dim= self.future_input_dim, 
            forecast_horizon=self.forecast_horizon, 
            mode='strict', # HALNet is generally strict
            model_name='xtft', # For specific validation logic if any
            verbose= 1 if logger.level <= 10 else 0 # DEBUG level
        )
        # Normalize and process static features
        normalized_static = self.learned_normalization(
            static_input, 
            training=training
        )
        logger.debug(
            f"Normalized Static Shape: {normalized_static.shape}"
        )
        # Apply -> GRN pipeline to cross attention
        static_features = self.static_dense(normalized_static)
        if self.use_batch_norm:
            static_features = self.static_batch_norm(
                static_features,
                training=training
            )
            logger.debug(
                "Static Features after BatchNorm Shape: "
                f"{static_features.shape}"
            )
    
        static_features = self.static_dropout(
            static_features,
            training=training
        )
        logger.debug(
            f"Static Features Shape: {static_features.shape}"
        )
        # XXX TODO # check apply --> GRN
        static_features = self.grn_static(
            static_features, 
            training=training
        ) 
   
        # --- Prepare inputs for MultiModalEmbedding ---
        # dynamic_input is the reference for T_past (lookback period).
        # future_input needs its time dimension aligned to dynamic_input's.
        logger.debug("  Aligning temporal inputs for MultiModalEmbedding...")
        _, future_input_for_embedding = align_temporal_dimensions(
            tensor_ref=dynamic_input,       # Shape (B, T_past, D_dyn)
            tensor_to_align=future_input,   # Shape (B, T_future_total, D_fut)
            mode='slice_to_ref',            # Slice future if longer
            name="future_input_for_mme"
        )
        # future_input_for_embedding now has shape (B, T_past, D_fut)
        logger.debug(
            f"    Dynamic for MME: {dynamic_input.shape}, "
            f"Future for MME: {future_input_for_embedding.shape}"
        )

        embeddings = self.multi_modal_embedding(
            [dynamic_input, future_input_for_embedding], # Pass ALIGNED inputs
            training=training
        )
        # Output of MultiModalEmbedding: (B, T_past, CombinedEmbedDim)
        logger.debug(
            f"  Embeddings shape after MultiModalEmbedding: {embeddings.shape}"
        )
    
        # Add positional encoding to embeddings
        # before attention mechanisms. 
        embeddings = self.positional_encoding(
            embeddings, 
            training=training 
        )  
        
        logger.debug(
            f"Embeddings with Positional Encoding Shape: {embeddings.shape}"
        )
    
        if self.use_residuals:
            embeddings = embeddings + self.residual_dense(embeddings)
            logger.debug(
                "Embeddings with Residuals Shape: "
                f"{embeddings.shape}"
            )
    
        # Multi-scale LSTM outputs
        lstm_output = self.multi_scale_lstm(
            dynamic_input,
            training=training
        )
        # If multi_scale_agg is None, lstm_output is (B, units * len(scales))
        # If multi_scale_agg is not None, lstm_output is a list of full 
        # sequences: [ (B, T', units), ... ]
        lstm_features = aggregate_multiscale(
            lstm_output, mode= self.multi_scale_agg 
        )
        # Since we are concatenating along the time dimension, we need 
        # all tensors to have the same shape along that dimension.
        time_steps = tf_shape(dynamic_input)[1]
        # Expand lstm_features to (B, 1, features)
        lstm_features = tf_expand_dims(lstm_features, axis=1)
        # Tile to match tf_time steps: (B, T, features)
        lstm_features = tf_tile(lstm_features, [1, time_steps, 1])

        logger.debug(
            f"LSTM Features Shape: {lstm_features.shape}"
        )
    
        # Attention mechanisms
       # For HierarchicalAttention, if it adds outputs,
       # inputs need same time dim.
        # We use dynamic_input (T_past) and the already sliced
        # future_input_for_embedding (T_past).
        logger.debug(
            "  Aligning temporal inputs for HierarchicalAttention..."
            )
        # No further alignment needed if using future_input_for_embedding
        # which is already aligned to dynamic_input's T_past.
        hierarchical_att = self.hierarchical_attention(
            [dynamic_input, future_input_for_embedding], # Both (B, T_past, Feats)
            training=training
        )
        logger.debug(
            f"Hierarchical Attention Shape: {hierarchical_att.shape}"
        )
    
        cross_attention_output = self.cross_attention(
            [dynamic_input, embeddings],
            training=training
        )
        logger.debug(
            f"Cross Attention Output Shape: {cross_attention_output.shape}"
        )
    
        memory_attention_output = self.memory_augmented_attention(
            hierarchical_att,
            training=training
        )
        logger.debug(
            "Memory Augmented Attention Output Shape: "
            f"{memory_attention_output.shape}"
        )
    
        # Combine all features
        time_steps = tf_shape(dynamic_input)[1]
        static_features_expanded = tf_tile(
            tf_expand_dims(static_features, axis=1),
            [1, time_steps, 1]
        )
        logger.debug(
            "Static Features Expanded Shape: "
            f"{static_features_expanded.shape}"
        )
    
        combined_features = Concatenate()([
            static_features_expanded,
            lstm_features,
            cross_attention_output, 
            hierarchical_att,
            memory_attention_output,
        ])
        logger.debug(
            f"Combined Features Shape: {combined_features.shape}"
        )
    
        attention_fusion_output = self.multi_resolution_attention_fusion(
            combined_features,
            training=training
        )
        logger.debug(
            "Attention Fusion Output Shape: "
            f"{attention_fusion_output.shape}"
        )
  
        time_window_output = self.dynamic_time_window(
            attention_fusion_output,
            training=training
        )
        logger.debug(
            f"Time Window Output Shape: {time_window_output.shape}"
        )
        # final_agg: last/average/flatten applied on time_window_output
        final_features = aggregate_time_window_output(
            time_window_output, self.final_agg
            )

        decoder_outputs = self.multi_decoder(
            final_features,
            training=training
        )
        logger.debug(
            f"Decoder Outputs Shape: {decoder_outputs.shape}"
        )
    
        predictions = self.quantile_distribution_modeling(
            decoder_outputs,
            training=training
        )
        # Anomaly detection branch
        if self.anomaly_detection_strategy == 'feature_based':
            # Compute anomaly scores from attention features
            attn_scores = self.anomaly_attention(
                query=attention_fusion_output,
                value=attention_fusion_output, 
                training=training
            )
            # Project the anomaly attention output 
            # to the desired dimension.
            projected_attn = self.anomaly_projection(
                attn_scores, training=training
                )
            # From config Anomaly score shape is (B, T, O) where= 
            # Batch size, T, is Time Steps and O is output dim. 
            # Compute anomaly scores using the projected output.
            self.anomaly_scores = self.anomaly_scorer(
                projected_attn, training=training
                )
  
        elif self.anomaly_detection_strategy == 'from_config':
            # Use anomaly_scores from anomaly_config
            # should give in 2D tensor (B, H)
            self.anomaly_scores = validate_anomaly_scores(
                self.anomaly_config, 
                self.forecast_horizon
            )
            logger.debug(
                "Using Anomaly Scores from anomaly_config"
                f" Shape: {self.anomaly_scores.shape}")
            
        # Handle anomaly loss
        if self.anomaly_scores is not None:
            # Use provided anomaly_scores from anomaly_config
            # Use default zeros placeholder for y_pred with
            # shape (B, T, O)
            # shape = tf_shape(self.anomaly_scores)
            # default_y_pred = tf_zeros(
            #     [shape[0], shape[1], shape[2]],
            #     dtype=self.anomaly_scores.dtype
            # )
            default_y_pred = tf_zeros_like(self.anomaly_scores)

            logger.debug(
                "Using Anomaly Scores from anomaly_config with"
                f" weight: {self.anomaly_loss_weight}.")
            
            # Define appropriate dimensions
            anomaly_loss = self.anomaly_loss_layer(
                self.anomaly_scores, 
                default_y_pred, 
            )
            self.add_loss(self.anomaly_loss_weight * anomaly_loss)
            logger.debug(
                f"Anomaly Loss Computed and Added: {anomaly_loss}")
        else:
            # Optionally, log a warning or set a default value.
            logger.warning(
                "Anomaly scores are None. Skipping anomaly loss."
            )
        
        logger.debug(
            f"Predictions Shape: {predictions.shape}"
        )
        
        # Explicitly squeeze ONLY if quantiles were 
        # requested AND output_dim is 1
        if self.quantiles is not None and self.output_dim == 1:
            # Check if the tensor actually has 4 dimensions before squeezing
            if len(predictions.shape) == 4:
                final_output = tf_squeeze(predictions, axis=-1)
                logger.debug( 
                    f"Squeezed final quantile output dim (O=1)."
                    f" New shape: {tf_shape(final_output)}"
                    )
            elif len(predictions.shape) == 3:
                 # Already has shape (B, H, Q), no squeeze needed
                 final_output = predictions
            else:
                 # Unexpected shape
                 logger.warning(f"Unexpected prediction shape before squeeze:"
                                f" {predictions.shape}. Returning as is.")
                 final_output = predictions

        elif self.quantiles is None:
             # Point forecast, ensure shape is (B, H, O)
             # (May not need specific handling 
             # if output_layer gives correct shape)
             final_output = predictions

        logger.debug(f"TFT '{self.name}': Final returned output shape:"
                     f" {tf_shape(final_output)}")
        
        return final_output

    def compile(self, optimizer, loss=None, **kws):
        """
        Compile the HALNet model, allowing an explicit user-specified loss
        to override the defaults.

        If the user provides a loss (loss=...), it is used regardless of
        quantiles or anomaly scores. Otherwise, the method uses the
        following logic:

        - If ``self.quantiles`` is None, defaults to "mse"(or the
          user-supplied ``loss``).
        - If ``self.quantiles`` is not None, uses a quantile-based loss.
          If ``anomaly_scores`` is present, a total loss is used that
          adds anomaly loss on top.
          
        See also:
        --------
        fusionlab.nn.losses.combined_quantile_loss
        fusionlab.nn.losses.combined_total_loss
        """
        # 1) If user explicitly provides a loss, respect that and skip defaults
        if loss is not None:
            super().compile(
                optimizer=optimizer,
                loss=loss,
                **kws
            )
            return
        
        # 2) Handle prediction-based strategy first
        if self.anomaly_detection_strategy == 'prediction_based':
            pred_loss_fn = prediction_based_loss(
                quantiles=self.quantiles,
                anomaly_loss_weight=self.anomaly_loss_weight
            )
            super().compile(
                optimizer=optimizer,
                loss=pred_loss_fn,
                **kws
            )
            return
    
        # 3) Otherwise, we handle the default logic
        if self.quantiles is None:
            # Deterministic scenario
            super().compile(
                optimizer=optimizer,
                loss="mean_squared_error",
                **kws
            )
            return
    
        # Probabilistic scenario with quantiles
        quantile_loss_fn = combined_quantile_loss(self.quantiles)
    
        # Handle from_config strategy
        if self.anomaly_detection_strategy == 'from_config':
            self.anomaly_scores = self.anomaly_config.get(
                "anomaly_scores")
            
            if self.anomaly_scores is not None:
                total_loss_fn = combined_total_loss(
                    quantiles=self.quantiles,
                    anomaly_layer=self.anomaly_loss_layer,
                    anomaly_scores=self.anomaly_scores
                )
                super().compile(
                    optimizer=optimizer,
                    loss=total_loss_fn,
                    **kws
                )
                return
        
        # Only quantile loss
        # Handles feature-based and other cases)
        super().compile(
            optimizer=optimizer,
            loss=quantile_loss_fn,
            **kws
        )

    @optional_tf_function
    def train_step(self, data):
        """
        Custom training step with anomaly detection strategy handling.
    
        Parameters
        ----------
        data : tuple/tf.data.Dataset
            Training data containing:
            - For prediction-based strategy: (x, y) pairs
            - Other strategies: Standard Keras-compatible format
    
        Returns
        -------
        dict
            Metric results dictionary
    
        Notes
        -----
        - Special handling for prediction-based anomaly detection:
          - Requires explicit (x, y) pairs
          - Validates y_true integrity
          - Falls back to standard training if data format invalid
        - For other strategies, uses native Keras training logic
    
        Raises
        ------
        Warning (logged)
            - For missing y_true in prediction-based mode
            - For invalid/nan values in y_true
    
        Example
        -------
        >>> model.compile(...)
        >>> model.fit(dataset, epochs=10)
        """
        # Handle prediction-based strategy
        if self.anomaly_detection_strategy == 'prediction_based':
            try:
                # Attempt to unpack (x, y) pair
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    x, y = data[0], data[1]
                else:
                    # For TF Dataset/other formats, try tensor split
                    x, y = tf_unstack(data, num=2, axis=0)
                    
            except (ValueError, tf_errors.InvalidArgumentError):
                logger.warning(
                    "Prediction-based strategy requires (x, y) data pairs. "
                    "Falling back to standard training step."
                )
                return super().train_step(data)
    
            # Verify y_true contains valid values
            if y.shape.ndims == 0 or tf_reduce_all(tf_is_nan(y)):
                logger.warning(
                    "Invalid y_true provided for prediction-based strategy. "
                    "Contains NaN values or incorrect shape."
                )
                return super().train_step(data)
    
            with tf_GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred)
    
            # Gradient updates
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            # Update metrics
            self.compiled_metrics.update_state(y, y_pred)
            return {m.name: m.result() for m in self.metrics}
    
        # Standard processing for other strategies
        return super().train_step(data)

    def get_config(self):
        """
        Get serialization configuration for model saving/loading.
    
        Returns
        -------
        dict
            Complete configuration dictionary containing:
            - Model architecture parameters
            - Anomaly detection configuration
            - Component hyperparameters
            - Training configuration
    
        Notes
        -----
        - Handles special cases for:
          - Quantile list serialization
          - Numpy array conversion for anomaly scores
          - Custom layer configurations
        - Logs configuration changes via model logger
    
        Example
        -------
        >>> config = model.get_config()
        >>> json.dump(config, open('model_config.json', 'w'))
        """
        # Retrieve the base configuration from the superclass.
        config = super().get_config().copy()
        # Update configuration with HALNet-specific parameters.
        config.update({
            'static_input_dim'  : int(self.static_input_dim),
            'dynamic_input_dim' : int(self.dynamic_input_dim),
            'future_input_dim'  : int(self.future_input_dim),
            'embed_dim'         : int(self.embed_dim),
            'forecast_horizon'  : int(self.forecast_horizon),
            'quantiles'         : (list(self.quantiles)
                                   if self.quantiles is not None 
                                   else None),
            'max_window_size'   : int(self.max_window_size),
            'memory_size'       : int(self.memory_size),
            'num_heads'         : int(self.num_heads),
            'dropout_rate'      : float(self.dropout_rate),
            'output_dim'        : int(self.output_dim),
            'attention_units'   : int(self.attention_units),
            'hidden_units'      : int(self.hidden_units),
            'lstm_units'        : (int(self.lstm_units)
                                   if self.lstm_units is not None 
                                   else None),
            'scales'            : (list(self.scales)
                                   if self.scales is not None 
                                   else None),
            'activation'        : self.activation,
            'use_residuals'     : bool(self.use_residuals),
            'use_batch_norm'    : bool(self.use_batch_norm),
            'final_agg'         : self.final_agg,
            'multi_scale_agg'   : (str(self.multi_scale_agg)
                                   if self.multi_scale_agg is not None 
                                   else None),
            'anomaly_config'    : {
                'anomaly_loss_weight': ( 
                    float(self.anomaly_loss_weight) if self.anomaly_loss_weight
                    is not None else 1.
                    )
            },
            'anomaly_loss_weight': self.anomaly_loss_weight, 
            'anomaly_detection_strategy': self.anomaly_detection_strategy, 
            
        })
    
        # Log that the configuration has been updated.
        logger.debug(
            "Configuration for HALNet has been updated in get_config."
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Reconstruct model instance from configuration dictionary.
    
        Parameters
        ----------
        config : dict
            Configuration dictionary generated by get_config()
    
        Returns
        -------
        HALNet
            Fully reconstructed model instance
    
        Notes
        -----
        - Handles special conversions:
          - Anomaly scores list -> numpy array
          - Quantile list restoration
          - Custom layer reconstruction
        - Maintains logger instance during reconstruction
    
        Example
        -------
        >>> loaded_model = HALNet.from_config(json.load(open('model_config.json')))
        """
        logger.debug("Creating HALNet instance from configuration.")
    
        # Convert anomaly_scores from list back to a NumPy array, if present.
        if config["anomaly_config"].get("anomaly_scores") is not None:
            config["anomaly_config"]["anomaly_scores"] = np.array(
                config["anomaly_config"]["anomaly_scores"], dtype=np.float32
            )
        # Return a new instance created using the updated configuration.
        return cls(**config)