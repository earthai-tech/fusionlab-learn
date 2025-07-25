# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
SuperXTFT: an XTFT variant that adds Variable Selection Networks
(VSNs) and applies Gate→Add&Norm→GRN pipelines to attention and
decoder outputs. Designed to inherit from :class:`XTFT` while
reusing :class:`BaseExtreme`'s infrastructure.
"""
from __future__ import annotations

from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union
from ...api.docs import _shared_docs, doc
from ._base_extreme import (
    KERAS_BACKEND,
    KERAS_DEPS,
    logger,
)
from .xtft import XTFT  

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
        GatedResidualNetwork,
        VariableSelectionNetwork,
        MultiModalEmbedding,
        MultiResolutionAttentionFusion,
        MultiScaleLSTM,
        HierarchicalAttention,
        CrossAttention,
        MemoryAugmentedAttention,
        LearnedNormalization,
        PositionalEncoding,
        aggregate_multiscale,
        aggregate_time_window_output
    )
    from .._tensor_validation import align_temporal_dimensions

__all__ = ["SuperXTFT"]


@register_keras_serializable(
    "fusionlab.nn.hybrid", name="SuperXTFT")
@doc(
    key_parameters = dedent (_shared_docs["xtft_params_doc"]), 
)
class SuperXTFT(XTFT):
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
        multi_scale_agg: Optional[str] = 'auto',
        activation: Union[str, callable] = 'relu',
        use_residuals: bool = True,
        use_batch_norm: bool = False,
        final_agg: str = 'last',
        anomaly_config: Optional[Dict[str, Any]] = None,
        anomaly_detection_strategy: Optional[str] = None,
        anomaly_loss_weight: float = 1.0,
        architecture_config: Optional[Dict] = None,
        fusion_mode: Optional[str] =None, 
        **kw: Any,
    ) -> None:
        logger.debug("SuperXTFT.__init__() called")
        
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
            architecture_config=architecture_config,
            fusion_mode = fusion_mode, 
            **kw,
        )

    def _build_components(self) -> None:
        logger.debug("SuperXTFT._build_components() start")
        
        # Re-sync feature_processing from updated architecture_config
        self._sync_architecture()

        self.activation = Activation(self.activation).activation_str

        # --------- Handle Feature Processing: VSN or Dense ---------
        if self.feature_processing == 'vsn':  # Variable Selection Network
            self.variable_selection_static = VariableSelectionNetwork(
                num_inputs=self.static_input_dim,
                units=self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
            )
            self.variable_selection_dynamic = VariableSelectionNetwork(
                num_inputs=self.dynamic_input_dim,
                units=self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
            )
            self.variable_future_covariate = VariableSelectionNetwork(
                num_inputs=self.future_input_dim,
                units=self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
            )
        else:  # Dense layers as fallback
            self.dense_static = Dense(
                self.hidden_units, activation=self.activation)
            self.dense_dynamic = Dense(
                self.hidden_units, activation=self.activation)
            self.dense_future_covariate = Dense(
                self.hidden_units, activation=self.activation)

        # -------------------- Static branch   ----------------------
        self.learned_normalization = LearnedNormalization()
        self.static_dense = Dense(
            self.hidden_units, activation=self.activation)
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

        # --------- Embeddings / Positional encoding ---------
        self.multi_modal_embedding = MultiModalEmbedding(self.embed_dim)
        self.positional_encoding = PositionalEncoding()
        self.residual_dense = (
            Dense(2 * self.embed_dim) if self.use_residuals else None
        )

        # --------- Temporal backbone blocks ---------
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

        # --------- GRNs for attention & decoder ---------
        self.grn_attention_hierarchical = GatedResidualNetwork(
            units=self.attention_units,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )
        self.grn_attention_cross = GatedResidualNetwork(
            units=self.attention_units,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )
        self.grn_memory_attention = GatedResidualNetwork(
            units=self.attention_units,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )
        self.grn_decoder = GatedResidualNetwork(
            units=self.output_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        )

        # --------------- Anomaly ---------------
        if self.anomaly_detection_strategy == 'feature_based':
            self.anomaly_attention = MultiHeadAttention(
                num_heads=1,
                key_dim=self.hidden_units,
                name='anomaly_attention'
            )
            self.anomaly_projection = Dense(
                self.hidden_units,
                activation='linear',
                name='anomaly_projection'
            )
            self.anomaly_scorer = Dense(
                1,
                activation='linear',
                name='anomaly_scorer'
            )
        else:
            self.anomaly_attention = None
            self.anomaly_projection = None
            self.anomaly_scorer = None

        logger.debug("SuperXTFT._build_components() done")


    @tf_autograph.experimental.do_not_convert
    def _encode_inputs(
        self,
        static_input: Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        *,
        training: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        logger.debug("SuperXTFT._encode_inputs() start")
        cache: Dict[str, Any] = {}

        # Handle static, dynamic, and future input processing (VSN or Dense)
        if self.feature_processing == 'vsn':
            sel_static = self.variable_selection_static(
                static_input, training=training
            )
            sel_dynamic = self.variable_selection_dynamic(
                dynamic_input, training=training
            )
            sel_future = self.variable_future_covariate(
                future_input, training=training
            )
        else:
            sel_static = self.dense_static(static_input)
            sel_dynamic = self.dense_dynamic(dynamic_input)
            sel_future = self.dense_future_covariate(future_input)

        logger.debug("VSN/Dense processed shapes s=%s d=%s f=%s",
                     sel_static.shape, sel_dynamic.shape, sel_future.shape)

        # Continue with encoding, normalization, and embeddings
        norm_static = self.learned_normalization(
            sel_static, training=training
        )
        static_features = self.static_dense(norm_static)
        if self.static_batch_norm is not None:
            static_features = self.static_batch_norm(
                static_features, training=training
            )
        static_features = self.static_dropout(
            static_features, training=training
        )
        static_features = self.grn_static(
            static_features, training=training
        )

        _, fut_for_embed, fut_mask = align_temporal_dimensions(
            tensor_ref=sel_dynamic,
            tensor_to_align=sel_future,
            mode="auto",
            return_mask=True,
            name="future_input_for_mme",
        )
        cache["fut_mask"] = fut_mask  # shape (B, T_ref)

        
        embeddings = self.multi_modal_embedding(
            [sel_dynamic, fut_for_embed], training=training
        )
        embeddings = self.positional_encoding(
            embeddings, training=training
        )
        if self.use_residuals and self.residual_dense is not None:
            embeddings = embeddings + self.residual_dense(embeddings)

        cache['embeddings'] = embeddings
        cache['future_for_embed'] = fut_for_embed
        logger.debug("SuperXTFT._encode_inputs() done")
        return static_features, sel_dynamic, fut_for_embed, cache
    
 
    @tf_autograph.experimental.do_not_convert
    def _temporal_backbone(
        self,
        dynamic_encoded: Tensor,
        future_encoded: Tensor,
        *,
        training: bool,
        cache: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        logger.debug("SuperXTFT._temporal_backbone() start")
        embeddings = cache['embeddings']
    
        # Multi-scale LSTM processing
        lstm_out = self.multi_scale_lstm(dynamic_encoded, training=training)
        lstm_feats = aggregate_multiscale(
            lstm_out, mode=self.multi_scale_agg
        )
        t_steps = tf_shape(dynamic_encoded)[1]
        lstm_feats = tf_expand_dims(lstm_feats, axis=1)
        lstm_feats = tf_tile(lstm_feats, [1, t_steps, 1])
    
        # Initialize attention mask
        attn_mask = tf_expand_dims(cache["fut_mask"], axis=1)  # (B, 1, T_v)
        logger.debug(f"Attention Mask Shape: {attn_mask.shape}")
    
        # Start with lstm_feats as the base context
        context_att = lstm_feats
    
        # Apply attention and GRN processing using the helper method
        context_att, cache = self._apply_fusion_mode(
            dynamic_encoded, future_encoded, embeddings, attn_mask, 
            training, context_att, cache 
        )
    
        # Final fusion of the attention outputs
        fused = self.multi_resolution_attention_fusion(
            context_att, training=training
        )
        logger.debug(
            f"Fused output shape after multi-resolution fusion: {fused.shape}")
    
        # Update the cache with attention outputs for possible future use
        cache.update({
            'hierarchical_att': cache.get('hierarchical_att', None),
            'cross_att': cache.get('cross_att', None),
            'memory_att': cache.get('memory_att', None),
        })
    
        logger.debug("SuperXTFT._temporal_backbone() done")
        return fused, cache
    

    @tf_autograph.experimental.do_not_convert
    def _apply_fusion_mode(
        self,
        dynamic_encoded: Tensor,
        future_encoded: Tensor,
        embeddings: Tensor,
        attn_mask: Tensor,
        training: bool,
        context_att: Tensor,
        cache: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Private helper method to apply attention mechanisms with GRNs.
        
        Handles both the integrated and separate attention+GRN approaches
        based on the `fusion_mode` parameter.
        """
        
        # Initialize cache with None to avoid key errors
        cache['hierarchical_att'] = None
        cache['cross_att'] = None
        cache['memory_att'] = None
    
        # Check the fusion_mode and apply attention & GRN accordingly
        if self.fusion_mode == 'integrated':
            # Gate → Add & Norm → GRN refinement (for decoder output)
   
            fused_feats = [context_att]
            # Apply Hierarchical Attention + GRN if present
            if 'hierarchical' in self.decoder_attention_stack:
                hierarchical_att = self.hierarchical_attention(
                    [dynamic_encoded, future_encoded],
                    training=training,
                    attention_mask=attn_mask
                )
                hierarchical_att_grn = self.grn_attention_hierarchical(
                    hierarchical_att,
                    training=training
                )
                logger.debug(f"Hierarchical Attention after GRN Shape: "
                             f"{hierarchical_att_grn.shape}")
                cache['hierarchical_att'] = hierarchical_att_grn
                
                fused_feats.append(hierarchical_att_grn)
    
            # Apply Cross Attention + GRN if present
            if 'cross' in self.decoder_attention_stack:
                cross_attention_output = self.cross_attention(
                    [dynamic_encoded, embeddings],
                    training=training, 
                    attention_mask=attn_mask
                )
                cross_attention_grn = self.grn_attention_cross(
                    cross_attention_output,
                    training=training
                )
                logger.debug(f"Cross Attention after GRN Shape: "
                             f"{cross_attention_grn.shape}")
                cache['cross_att'] = cross_attention_grn
                
                fused_feats.append(cross_attention_grn)
    
            # Apply Memory Attention + GRN if present
            if 'memory' in self.decoder_attention_stack:
                memory_attention_output = self.memory_augmented_attention(
                    hierarchical_att_grn,
                    training=training, 
                    attention_mask=attn_mask
                )
                memory_attention_grn = self.grn_memory_attention(
                    memory_attention_output,
                    training=training
                )
                logger.debug(f"Memory Attention after GRN Shape: "
                             f"{memory_attention_grn.shape}")
             
                cache['memory_att'] = memory_attention_grn
                
                fused_feats.append(memory_attention_grn)
                
            # Fallback to hierarchical if no memory attention
            context_att = Concatenate()(fused_feats)
    
        else:
            # Apply attention and GRN separately
            # for each attention mechanism
            if 'cross' in self.decoder_attention_stack:
                cross_att = self.cross_attention(
                    [dynamic_encoded, embeddings], 
                    training=training,
                    attention_mask=attn_mask
                )
                cross_att = self.grn_attention_cross(
                    cross_att,
                    training=training
                )
                context_att = cross_att
                logger.debug(f"Cross Attention after GRN Shape: "
                             f"{cross_att.shape}")
                cache['cross_att'] = cross_att
    
            if 'hierarchical' in self.decoder_attention_stack:
                hier_att = self.hierarchical_attention(
                    [dynamic_encoded, future_encoded], 
                    training=training, 
                    attention_mask=attn_mask
                )
                hier_att = self.grn_attention_hierarchical(
                    hier_att,
                    training=training
                )
                context_att = hier_att
                logger.debug(f"Hierarchical Attention after GRN Shape: "
                             f"{hier_att.shape}")
                cache['hierarchical_att'] = hier_att
    
            if 'memory' in self.decoder_attention_stack:
                mem_att = self.memory_augmented_attention(
                    context_att, 
                    training=training, 
                    attention_mask=attn_mask
                )
                mem_att = self.grn_memory_attention(
                    mem_att,
                    training=training
                )
                context_att = mem_att
                logger.debug(f"Memory Attention after GRN Shape: "
                             f"{mem_att.shape}")
                cache['memory_att'] = mem_att
    
        return context_att, cache

    def _aggregate_decode(
        self,
        fused_feats: Tensor,
        *,
        training: bool,
        cache: Dict[str, Any],
    ) -> Tensor:
        """
        Apply attention to the fused features and pass through GRN
        (Gate→Add&Norm→GRN pipeline).
        """
        
        # Initialize with fused features as the base context
        context_att = fused_feats  
     
        time_window_output = self.dynamic_time_window(
            context_att, training=training
        )
        final_features = aggregate_time_window_output(
            time_window_output, self.final_agg
        )
        dec_out = self.multi_decoder(
            final_features, training=training
        )
    
        # Apply GRN refinement to the final decoder context
        # Gate
        G = self.grn_decoder.gate_dense(dec_out)
        # Add & Norm
        Z_norm = self.grn_decoder.layer_norm(dec_out + G)
        # GRN
        Z_grn = self.grn_decoder(Z_norm, training=training)
    
        return Z_grn

    def _maybe_compute_anomaly_scores(
        self,
        fused_feats: Tensor,
        *,
        training: bool,
        cache: Dict[str, Any],
    ) -> None:
        """Compute anomaly scores when strategy == 'feature_based'.

        Attention → projection → linear scorer to produce
        (B, T, 1) scores stored in `self.anomaly_scores`.
        """
        if self.anomaly_detection_strategy != 'feature_based':
            return None

        if (self.anomaly_attention is None or
                self.anomaly_projection is None or
                self.anomaly_scorer is None):
            logger.warning(
                "feature_based strategy set but anomaly layers missing; "
                "skipping anomaly scoring.")
            return None

        attn_scores = self.anomaly_attention(
            query=fused_feats,
            value=fused_feats,
            training=training,
        )
        proj = self.anomaly_projection(attn_scores, training=training)
        self.anomaly_scores = self.anomaly_scorer(proj, training=training)
        logger.debug("anomaly_scores shape=%s", self.anomaly_scores.shape)
        return None
    
SuperXTFT.__doc__ = r"""
An extension of :class:`XTFT` that injects **Variable Selection
Networks (VSNs)** and a **Gate→Add&Norm→GRN** refinement pipeline
on attention and decoder outputs. It inherits all shared logic
from :class:`BaseExtreme` (validation, losses, hooks) and the
baseline flow from :class:`XTFT`, then overrides only the parts
that differ.

Key Additions
-------------
* **VariableSelectionNetwork** for:
  - static features (no time axis),
  - dynamic / historical features,
  - future-known covariates.
  These learn soft weights to highlight the most informative
  variables at each step.

* **GRN refinement blocks** applied after:
  - Hierarchical attention output,
  - Cross attention output,
  - Memory-augmented attention output,
  - Decoder output (via Gate→Add&Norm→GRN pipeline).

* **Optional feature-based anomaly scoring** (when
  ``anomaly_detection_strategy == 'feature_based'``): a dedicated
  attention → projection → scorer head produces per-timestep
  anomaly scores that feed the anomaly loss.

{key_parameters}

Design Rationale
----------------
VSNs reduce noise from high-dimensional inputs by learning which
signals matter *now*. The extra GRNs stabilize attention outputs
and improve representational depth without exploding parameters.
Keeping these pieces modular lets you toggle them on variants
without rewriting boilerplate.

Hook Overview
-------------
This class overrides:

* ``_build_components`` — builds VSNs, GRNs, and (optionally)
  anomaly-attention layers.
* ``_encode_inputs`` — runs VSNs before the usual embedding /
  normalization path.
* ``_temporal_backbone`` — inserts GRNs after each attention.
* ``_aggregate_decode`` — applies the Gate→Add&Norm→GRN pipeline
  to decoder outputs.
* ``_maybe_compute_anomaly_scores`` — computes scores for the
  feature-based strategy.

Everything else (compile logic, losses, serialization) is taken
care of by the base classes.

Examples
--------
Instantiate with VSNs and GRN refinements active by default::

    model = SuperXTFT(
        static_input_dim=10,
        dynamic_input_dim=32,
        future_input_dim=8,
        forecast_horizon=7,
        quantiles=[0.1, 0.5, 0.9],
        multi_scale_agg='auto',
        anomaly_detection_strategy='feature_based',
    )

Then compile and fit as usual::

    model.compile(optimizer='adam')
    model.fit([X_static, X_dynamic, X_future], y, epochs=20)

Notes
-----
* Set the logger to ``DEBUG`` to see shapes and flow decisions.
* If you do not need feature-based anomalies, omit that strategy
  to skip building the extra attention head and save memory.

See Also
--------
XTFT :
    Baseline DRY implementation without VSN/extra GRNs.

BaseExtreme :
    Parent class providing shared plumbing, hooks, and logging.
"""
