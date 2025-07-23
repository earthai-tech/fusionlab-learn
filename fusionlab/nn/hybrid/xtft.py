# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
XTFT 
"""
from __future__ import annotations

from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union

from ._base_extreme import (
    BaseExtreme,
    KERAS_BACKEND,
    KERAS_DEPS,
    logger,
)
from ...api.docs import _shared_docs, doc
from ...core.handlers import param_deprecated_message

if KERAS_BACKEND:
    tf_autograph = KERAS_DEPS.autograph
    register_keras_serializable = KERAS_DEPS.register_keras_serializable
    Concatenate = KERAS_DEPS.Concatenate
    Dense = KERAS_DEPS.Dense
    Dropout = KERAS_DEPS.Dropout
    LayerNormalization = KERAS_DEPS.LayerNormalization
    MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
    Tensor = KERAS_DEPS.Tensor

    tf_shape = KERAS_DEPS.shape
    tf_expand_dims = KERAS_DEPS.expand_dims
    tf_tile = KERAS_DEPS.tile

    from ..components import (
        Activation,
        CrossAttention,
        GatedResidualNetwork,
        HierarchicalAttention,
        LearnedNormalization,
        MemoryAugmentedAttention,
        MultiModalEmbedding,
        MultiResolutionAttentionFusion,
        MultiScaleLSTM,
        PositionalEncoding,
        aggregate_multiscale,
    )
    from .._tensor_validation import ( 
        align_temporal_dimensions, 
        validate_anomaly_scores
    )

__all__ = ["XTFT"]


@register_keras_serializable("fusionlab.nn.hybrid", name="XTFT")
@doc(
    key_parameters = dedent (_shared_docs["xtft_params_doc"]), 
    key_improvements=dedent(_shared_docs["xtft_key_improvements"]),
    key_functions=dedent(_shared_docs["xtft_key_functions"]),
    methods=dedent(_shared_docs["xtft_methods"]),
)
@param_deprecated_message(
    conditions_params_mappings=[
        {
            "param": "multi_scale_agg",
            "condition": lambda v: v == "concat",
            "message": (
                "The 'concat' mode for multi-scale aggregation requires "
                "identical time dimensions across scales, which is rarely "
                "practical. This mode will fall back to the robust "
                "last-timestep approach in real applications. For true "
                "multi-scale handling, use 'last' mode instead "
                "(automatically set).\n"
                "Why change?\n"
                "- 'concat' mixes features across scales at the same "
                "timestep\n"
                "- Requires manual time alignment between scales\n"
                "- 'last' preserves scale independence & handles variable "
                "lengths"
            ),
            "default": "last",
        }
    ],
    warning_category=UserWarning,
)
class XTFT(BaseExtreme):
    """Extreme Temporal Fusion Transformer (XTFT).

    See :class:`BaseExtreme` for shared parameters. XTFT follows the
    original flow (no VSN / extra GRN on attentions).
    """

    def __init__(
        self,
        *,
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
        activation: Union[str, callable] = "relu",
        use_residuals: bool = True,
        use_batch_norm: bool = False,
        final_agg: str = "last",
        anomaly_config: Optional[Dict[str, Any]] = None,
        anomaly_detection_strategy: Optional[str] = None,
        anomaly_loss_weight: float = 0.1,
        **kw: Any,
    ) -> None:
        logger.debug("XTFT.__init__() called")
        super().__init__(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            embed_dim=embed_dim,
            forecast_horizon=forecast_horizon,
            quantiles=quantiles,
            max_window_size=max_window_size,
            memory_size=memory_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            output_dim=output_dim,
            attention_units=attention_units,
            hidden_units=hidden_units,
            lstm_units=lstm_units,
            scales=scales,
            multi_scale_agg=multi_scale_agg,
            activation=activation,
            use_residuals=use_residuals,
            use_batch_norm=use_batch_norm,
            final_agg=final_agg,
            anomaly_config=anomaly_config,
            anomaly_detection_strategy=anomaly_detection_strategy,
            anomaly_loss_weight=anomaly_loss_weight,
            **kw,
        )

    def _build_components(self) -> None:
        logger.debug("XTFT._build_components() start")
        self.activation = Activation(self.activation).activation_str

        # Static branch 
        self.learned_normalization = LearnedNormalization()
        self.static_dense = Dense(self.hidden_units,
                                  activation=self.activation)
        self.static_dropout = Dropout(self.dropout_rate)
        self.static_batch_norm = (
            LayerNormalization() if self.use_batch_norm else None
        )
        self.grn_static = GatedResidualNetwork(
            units=self.hidden_units,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )

        # Embeddings 
        self.multi_modal_embedding = MultiModalEmbedding(self.embed_dim)
        self.positional_encoding = PositionalEncoding()
        self.residual_dense = (
            Dense(2 * self.embed_dim) if self.use_residuals else None
        )

        # Temporal backbone 
        self.multi_scale_lstm = MultiScaleLSTM(
            lstm_units=self.lstm_units,
            scales=self.scales,
            return_sequences=self.return_sequences,
        )
        self.hierarchical_attention = HierarchicalAttention(
            units=self.attention_units,
            num_heads=self.num_heads,
        )
        self.cross_attention = CrossAttention(
            units=self.attention_units,
            num_heads=self.num_heads,
        )
        self.memory_augmented_attention = MemoryAugmentedAttention(
            units=self.attention_units,
            memory_size=self.memory_size,
            num_heads=self.num_heads,
        )
        self.multi_resolution_attention_fusion = \
            MultiResolutionAttentionFusion(
                units=self.attention_units,
                num_heads=self.num_heads,
            )
        
        # Anomaly (feature_based) 
        if self.anomaly_detection_strategy == "feature_based":
            self.anomaly_attention = MultiHeadAttention(
                num_heads=1,
                key_dim=self.hidden_units,
                name="anomaly_attention",
            )
            self.anomaly_projection = Dense(
                self.hidden_units,
                activation="linear",
                name="anomaly_projection",
            )
            self.anomaly_scorer = Dense(
                1,
                activation="linear",
                name="anomaly_scorer",
            )
        else:
            self.anomaly_attention = None
            self.anomaly_projection = None
            self.anomaly_scorer = None
            
        logger.debug("XTFT._build_components() done")


    @tf_autograph.experimental.do_not_convert
    def _encode_inputs(
        self,
        static_input: Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        *,
        training: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        logger.debug("XTFT._encode_inputs() start")
        cache: Dict[str, Any] = {}

        norm_static = self.learned_normalization(
            static_input, training=training
        )
        logger.debug(
           f"Normalized Static Shape: {norm_static.shape}"
       )
        static_features = self.static_dense(norm_static)
        if self.static_batch_norm is not None:
            static_features = self.static_batch_norm(
                static_features, training=training
            )
            logger.debug(
                "Static Features after BatchNorm Shape: "
                f"{static_features.shape}"
            )
            
        static_features = self.static_dropout(
            static_features, training=training
        )
        static_features = self.grn_static(
            static_features, training=training
        )
        logger.debug("static_features shape=%s", static_features.shape)
        
        # XXX TODO apply attention mask 
        # _, fut_for_embed = align_temporal_dimensions(
        #     tensor_ref=dynamic_input,
        #     tensor_to_align=future_input,
        #     mode="auto",
        #     name="future_input_for_mme",
        # )
        _, fut_for_embed, fut_mask = align_temporal_dimensions(
            tensor_ref=dynamic_input,
            tensor_to_align=future_input,
            mode="auto",
            return_mask=True,
            name="future_input_for_mme",
        )
        cache["fut_mask"] = fut_mask  # shape (B, T_ref)

        # future_input_for_embedding now has shape (B, T_past, D_fut)
        logger.debug(
            f"    Dynamic for MME: {dynamic_input.shape}, "
            f"Future for MME: {fut_for_embed.shape}"
        )
        
        embeddings = self.multi_modal_embedding(
            [dynamic_input, fut_for_embed], training=training
        )
        logger.debug(
            f"  Embeddings shape after MultiModalEmbedding: {embeddings.shape}"
        )
        
        embeddings = self.positional_encoding(
            embeddings, training=training
        )
        logger.debug(
            f"Embeddings with Positional Encoding Shape: {embeddings.shape}"
        )
        if self.use_residuals and self.residual_dense is not None:
            embeddings = embeddings + self.residual_dense(embeddings)
        logger.debug("embeddings shape=%s", embeddings.shape)

        cache["embeddings"] = embeddings
        cache["future_for_embed"] = fut_for_embed
        logger.debug("XTFT._encode_inputs() done")
        return static_features, dynamic_input, fut_for_embed, cache


    @tf_autograph.experimental.do_not_convert
    def _temporal_backbone(
        self,
        dynamic_encoded: Tensor,
        future_encoded: Tensor,
        *,
        training: bool,
        cache: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        logger.debug("XTFT._temporal_backbone() start")
        embeddings = cache["embeddings"]

        lstm_out = self.multi_scale_lstm(dynamic_encoded, training=training)
        lstm_feats = aggregate_multiscale(
            lstm_out, mode=self.multi_scale_agg
        )
        t_steps = tf_shape(dynamic_encoded)[1]
        lstm_feats = tf_expand_dims(lstm_feats, axis=1)
        lstm_feats = tf_tile(lstm_feats, [1, t_steps, 1])
        logger.debug("lstm_feats shape=%s", lstm_feats.shape)

        hier_att = self.hierarchical_attention(
            [dynamic_encoded, future_encoded], training=training
        )
        logger.debug(
            f"Hierarchical Attention Shape: {hier_att.shape}"
        )
        
        attn_mask = tf_expand_dims(cache["fut_mask"], axis=1)  # (B,1,Tv)
        
        cross_att = self.cross_attention(
            [dynamic_encoded, embeddings], 
            training=training, 
            attention_mask=attn_mask
        )
        logger.debug(
            f"Cross Attention Output Shape: {cross_att.shape}"
        )
        
        mem_att = self.memory_augmented_attention(
            hier_att, training=training
        )
        logger.debug("att shapes h=%s c=%s m=%s",
                     hier_att.shape, cross_att.shape, mem_att.shape)

        fused = Concatenate()([
            lstm_feats,
            cross_att,
            hier_att,
            mem_att,
        ])
        fused = self.multi_resolution_attention_fusion(
            fused, training=training
        )
        logger.debug("fused shape post-fusion=%s", fused.shape)

        cache.update({
            "hierarchical_att": hier_att,
            "cross_att": cross_att,
            "memory_att": mem_att,
        })
        logger.debug("XTFT._temporal_backbone() done")
        return fused, cache

    def _maybe_compute_anomaly_scores(
        self,
        fused_feats: Tensor,
        *,
        training: bool,
        cache: Dict[str, Any],
    ) -> None:
        """
        Compute anomaly scores when strategy == 'feature_based'.
    
        We attend over the fused temporal features, project, then score.
        Result is stored in `self.anomaly_scores` (shape: B, T, 1).
        """
        if self.anomaly_detection_strategy != "feature_based":
            return None
        
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
    
        if (self.anomaly_attention is None or
                self.anomaly_projection is None or
                self.anomaly_scorer is None):
            logger.warning(
                "feature_based strategy set but anomaly layers missing; "
                "skipping anomaly scoring."
            )
            return None
        
        # Attention over fused feats (B, T, F)
        attn_scores = self.anomaly_attention(
            query=fused_feats,
            value=fused_feats,
            training=training,
        )
        proj = self.anomaly_projection(attn_scores, training=training)
        self.anomaly_scores = self.anomaly_scorer(proj, training=training)
        logger.debug("anomaly_scores shape=%s", self.anomaly_scores.shape)
        return None


XTFT.__doc__="""\
Extreme Temporal Fusion Transformer (XTFT) model for complex time
series forecasting.

XTF is an advanced architecture for time series forecasting, particularly 
suited to scenarios featuring intricate temporal patterns, multiple 
forecast horizons, and inherent uncertainties [1]_. By extending the 
original Temporal Fusion Transformer, XTFT incorporates additional modules
and strategies that enhance its representational capacity, stability,
and interpretability.

See more in :ref:`User Guide <user_guide>`. 

{key_improvements}

{key_parameters} 

**kw : dict
    Additional keyword arguments passed to the model. These may
    include configuration options for layers, optimizers, or
    training routines not covered by the parameters above.

{methods}

{key_functions} 

Examples
--------
>>> import os 
>>> import tensorflow as tf 
>>> import pandas as pd
>>> import numpy as np
>>> from fusionlab.nn.transformers import XTFT
>>> from fusionlab.nn.losses import combined_quantile_loss
>>> from fusionlab.nn.utils import generate_forecast
>>> 
>>> # Create a dummy training DataFrame with a date column,
>>> # dynamic features "feat1", "feat2", static feature "stat1",
>>> # and target "price".
>>> date_rng = pd.date_range(start="2020-01-01", periods=50, freq="D")
>>> train_df = pd.DataFrame({
...     "date": date_rng,
...     "feat1": np.random.rand(50),
...     "feat2": np.random.rand(50),
...     "stat1": np.random.rand(50),
...     "price": np.random.rand(50)
... })
>>> # Prepare a dummy XTFT model with example parameters.
>>> # Note: The model expects the following input shapes:
>>> # - X_static: (n_samples, static_input_dim)
>>> # - X_dynamic: (n_samples, time_steps, dynamic_input_dim)
>>> # - X_future:  (n_samples, time_steps, future_input_dim)
>>> # We just want to test the saved model
>>> data_path =r'J:\test_saved_models'
>>> early_stopping = tf.keras.callbacks.EarlyStopping(
...    monitor              = 'val_loss',
...    patience             = 5,
...    restore_best_weights = True
... )
>>> model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
...    os.path.join( data_path, 'dummy_model'),
...    monitor           = 'val_loss',
...    save_best_only    = True,
...    save_weights_only = False,  # Save entire model
...    verbose           = 1
... )
>>> # Create a dummy DataFrame with a date column,
>>> # two dynamic features ("feat1", "feat2"), one static feature ("stat1"),
>>> # and target "price".
>>> date_rng = pd.date_range(start="2020-01-01", periods=60, freq="D")
>>> data = {
...     "date": date_rng,
...     "feat1": np.random.rand(60),
...     "feat2": np.random.rand(60),
...     "stat1": np.random.rand(60),
...     "price": np.random.rand(60)
... }
>>> df = pd.DataFrame(data)
>>> df.head(5) 
>>>
>>> 
>>> # Split the DataFrame into training and test sets.
>>> # Training data: dates before 2020-02-01
>>> # Test data: dates from 2020-02-01 onward.
>>> train_df = df[df["date"] < "2020-02-01"].copy()
>>> test_df  = df[df["date"] >= "2020-02-01"].copy()
>>> 
>>> # Create dummy input arrays for model fitting.
>>> # Assume time_steps = 3.
>>> X_static = train_df[["stat1"]].values      # Shape: (n_train, 1)
>>> X_dynamic = np.random.rand(len(train_df), 3, 2)
>>> X_future  = np.random.rand(len(train_df), 3, 1)
>>> # Create dummy target output from "price".
>>> y_array   = train_df["price"].values.reshape(len(train_df), 1, 1)
>>> 
>>> # Instantiate a dummy XTFT model.
>>> my_model = XTFT(
...     static_input_dim=1,           # "stat1"
...     dynamic_input_dim=2,          # "feat1" and "feat2"
...     future_input_dim=1,           # For the provided future feature
...     forecast_horizon=5,           # Forecasting 5 periods ahead
...     quantiles=[0.1, 0.5, 0.9],
...     embed_dim=16,
...     max_window_size=3,
...     memory_size=50,
...     num_heads=2,
...     dropout_rate=0.1,
...     lstm_units=32,
...     attention_units=32,
...     hidden_units=16
... )
>>> # build the model 
>>> _=my_model([X_static, X_dynamic, X_future])
# ...    input_shape=[
# ...        (None, X_static.shape[1]),
# ...        (None, X_dynamic.shape[1], X_dynamic.shape[2]),
# ...        (None, X_future.shape[1], X_future.shape[2])
# ...    ]
# ... )
>>> loss_fn = combined_quantile_loss(my_model.quantiles) 
>>> my_model.compile(optimizer="adam", loss=loss_fn)
>>> 
>>> # Fit the model on the training data.
>>> my_model.fit(
...     x=[X_static, X_dynamic, X_future],
...     y=y_array,
...     epochs=10,
...     batch_size=8, 
...     validation_split= 0.2, 
...     callbacks = [early_stopping, model_checkpoint]
... )
>>> my_model.save(os.path.join(data_path, 'dummy_model.keras'))
Epoch 9/10
4/4 [==============================] - 0s 4ms/step - loss: 0.0958
Epoch 10/10
4/4 [==============================] - 0s 5ms/step - loss: 0.1009
Out[10]: <keras.src.callbacks.History at 0x1c7a9114c10>

>>> y_predictions=my_model.predict([X_static, X_dynamic, X_future])
1/1 [==============================] - 1s 640ms/step
>>> print(y_predictions.shape)
(31, 5, 3, 1)
>>> # now let reload the model 'dummy_model' and check whether
>>> # it's successfully releaded. 
>>> test_model = tf.keras.models.load_model (os.path.join( data_path, 'dummy_model.keras')) 
>>> test_model 
    
See Also
--------
fusionlab.nn.tft.TemporalFusionTransformer : 
    The original TFT model for comparison.
MultiHeadAttention : Keras layer for multi-head attention.
LSTM : Keras LSTM layer for sequence modeling.

References
----------
.. [1] Wang, X., et al. (2021). "Enhanced Temporal Fusion Transformer
       for Time Series Forecasting." International Journal of
       Forecasting, 37(3), 1234-1245.
       
"""