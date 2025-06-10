import numpy as np
import tensorflow as tf

# Assuming PIHALNet and other components are correctly imported
from fusionlab.nn.pinn.models import PIHALNet
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset, AUTOTUNE

# Toy configuration
batch_size = 4
time_steps = 5  # look-back window (max_window_size)
forecast_horizon =10  # forecast horizon (now > time_steps)
static_input_dim = 2
dynamic_input_dim = 3
future_input_dim = 1

fixed_model_params = {
    "static_input_dim": static_input_dim,
    "dynamic_input_dim": dynamic_input_dim,
    "future_input_dim": future_input_dim,
    "output_subsidence_dim": 1,
    "output_gwl_dim": 1,
    "forecast_horizon": forecast_horizon,
    "quantiles": None,
    "max_window_size": time_steps,
    "memory_size": 10,
    "scales": [1],
    "multi_scale_agg": "last",
    "final_agg": "last",
    "use_residuals": True,
    "use_batch_norm": False,
    "pde_mode": "consolidation",
    "pinn_coefficient_C": "learnable",
    "gw_flow_coeffs": None,
    "use_vsn": True
}

# Instantiate PIHALNet
model = PIHALNet(
    **fixed_model_params,
    embed_dim=16,
    hidden_units=16,
    lstm_units=16,
    attention_units=8,
    num_heads=2,
    dropout_rate=0.1,
    activation="relu"
)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
    loss={
        "subs_pred": MeanSquaredError(name="subs_data_loss"),
        "gwl_pred": MeanSquaredError(name="gwl_data_loss"),
    },
    metrics={
        "subs_pred": ["mae"],
        "gwl_pred": ["mae"],
    },
    loss_weights={"subs_pred": 1.0, "gwl_pred": 0.8},
    lambda_pde=0.1
)

# --- CORRECTED Data Generation ---

# 1. Generate features for the lookback window (time_steps)
static_features = np.random.rand(
    batch_size, static_input_dim
).astype("float32")
dynamic_features = np.random.rand(
    batch_size, time_steps, dynamic_input_dim
).astype("float32")

# 2. Generate features and coordinates for the forecast window (forecast_horizon)
# These are the coordinates that the PDE will be evaluated on.
future_t_coords = np.tile(
    np.arange(forecast_horizon, dtype="float32").reshape(1, forecast_horizon, 1),
    (batch_size, 1, 1)
)
future_x_coords = np.random.rand(
    batch_size, forecast_horizon, 1
).astype("float32")
future_y_coords = np.random.rand(
    batch_size, forecast_horizon, 1
).astype("float32")

# This is the CORRECT coordinates tensor for the model input dict
forecast_coords = np.concatenate(
    [future_t_coords, future_x_coords, future_y_coords], axis=-1
)

future_features = np.random.rand(
    batch_size, forecast_horizon, future_input_dim
).astype("float32")

# 3. Generate targets matching the forecast_horizon
subs_targets = np.random.rand(
    batch_size, forecast_horizon, 1
).astype("float32")
gwl_targets = np.random.rand(
    batch_size, forecast_horizon, 1
).astype("float32")

# 4. Package inputs exactly as PIHALNet expects
inputs = {
    # CRITICAL FIX: Pass the coordinates corresponding to the forecast horizon
    "coords": forecast_coords,
    "static_features": static_features,
    "dynamic_features": dynamic_features,
    "future_features": future_features,
}
targets = {
    "subs_pred": subs_targets,
    "gwl_pred": gwl_targets,
}

# Create the dataset
dataset = Dataset.from_tensor_slices((inputs, targets)).batch(batch_size)

print("Running model.fit() on corrected toy data for 1 epoch…")
# This should now run without shape errors
history = model.fit(dataset, epochs=50)

print("\nModel training step completed successfully.")
from fusionlab.nn.models.utils import plot_history_in 

pihalnet_metrics = {
    "Loss Components": ["total_loss", "data_loss", "physics_loss"],
    "Subsidence MAE": ["subs_pred_mae", "gwl_pred_mae"]
}
plot_history_in(
    history,
    metrics=pihalnet_metrics,
    layout='subplots',
    title='PIHALNet Training History'
)

# import numpy as np
# import tensorflow as tf

# from fusionlab.nn.pinn.models import PIHALNet
# from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
# from tensorflow.keras.optimizers import Adam
# from tensorflow.data import Dataset, AUTOTUNE

# # Toy configuration
# batch_size       = 4
# time_steps       = 5   # look-back window (max_window_size)
# forecast_horizon = 6   # forecast horizon
# static_input_dim = 2
# dynamic_input_dim= 3
# future_input_dim = 1

# fixed_model_params = {
#     "static_input_dim":      static_input_dim,
#     "dynamic_input_dim":     dynamic_input_dim,
#     "future_input_dim":      future_input_dim,
#     "output_subsidence_dim": 1,
#     "output_gwl_dim":        1,
#     "forecast_horizon":      forecast_horizon,
#     "quantiles":             None,
#     "max_window_size":       time_steps,
#     "memory_size":           10,
#     "scales":                [1],
#     "multi_scale_agg":       "last",
#     "final_agg":             "last",
#     "use_residuals":         True,
#     "use_batch_norm":        False,
#     "pde_mode":              "consolidation",
#     "pinn_coefficient_C":    "learnable",
#     "gw_flow_coeffs":        None,
#     "use_vsn":               True
# }

# # Instantiate PIHALNet
# model = PIHALNet(
#     **fixed_model_params,
#     embed_dim=16,
#     hidden_units=16,
#     lstm_units=16,
#     attention_units=8,
#     num_heads=2,
#     dropout_rate=0.1,
#     activation="relu"
# )

# model.compile(
#     optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
#     loss={
#         "subs_pred": MeanSquaredError(name="subs_data_loss"),
#         "gwl_pred":  MeanSquaredError(name="gwl_data_loss"),
#     },
#     metrics={
#         "subs_pred": [MeanAbsoluteError(name="subs_mae")],
#         "gwl_pred":  [MeanAbsoluteError(name="gwl_mae")],
#     },
#     loss_weights={"subs_pred": 1.0, "gwl_pred": 0.8},
#     lambda_pde=0.1
# )

# # Generate random toy data
# coords = np.random.rand(batch_size, time_steps, 3).astype("float32")
# static_features  = np.random.rand(batch_size, static_input_dim).astype("float32")
# dynamic_features = np.random.rand(batch_size, time_steps, dynamic_input_dim).astype("float32")

# # Crucial fix: time tensor for future must match forecast_horizon
# # For example, assume yearly steps [t0, t0+1, t0+2, …]
# # We simulate 'future_time' going from 0…H-1 for each example:
# future_time = np.tile(
#     np.arange(forecast_horizon, dtype="float32").reshape(1, forecast_horizon, 1),
#     (batch_size, 1, 1)
# )

# future_features = np.random.rand(batch_size, forecast_horizon, future_input_dim).astype("float32")

# # Targets (batch, forecast_horizon, 1)
# subs_targets = np.random.rand(batch_size, forecast_horizon, 1).astype("float32")
# gwl_targets = np.random.rand(batch_size, forecast_horizon, 1).astype("float32")

# # Package inputs exactly as PIHALNet expects:
# inputs = {
#     "coords":             coords,
#     "static_features":    static_features,
#     "dynamic_features":   dynamic_features,
#     "future_features":    future_features,
#     "time_steps":         future_time       # <— pass future_time here
# }
# targets = {
#     "subs_pred": subs_targets,
#     "gwl_pred":  gwl_targets,
# }

# dataset = Dataset.from_tensor_slices((inputs, targets))\
#                  .batch(batch_size)\
#                  .prefetch(AUTOTUNE)

# print("Running model.fit() on corrected toy data for 1 epoch …")
# model.fit(dataset, epochs=1)

    # def run_halnet_core1(
    #     self,
    #     static_input: Tensor,
    #     dynamic_input: Tensor,
    #     future_input: Tensor,
    #     training: bool
    # ) -> Tensor:
    #     """
    #     Executes the core data-driven feature extraction pipeline.
    
    #     This method processes static, dynamic, and future inputs through
    #     VSNs (if enabled), LSTMs, and various attention mechanisms to
    #     produce a final set of features ready for decoding. This revised
    #     version follows an encoder-decoder pattern to correctly handle
    #     inputs with different temporal lengths (`time_steps` and
    #     `forecast_horizon`).
    
    #     Args:
    #         static_input (tf.Tensor): Processed static features.
    #         dynamic_input (tf.Tensor): Processed dynamic historical features.
    #                                    Shape: (batch, time_steps, features).
    #         future_input (tf.Tensor): Processed known future features.
    #                                   Shape: (batch, forecast_horizon, features).
    #         training (bool): Python boolean indicating training mode.
    
    #     Returns:
    #         tf.Tensor: A feature tensor for the forecast horizon, ready for
    #                    the MultiDecoder. Shape: (batch, forecast_horizon, features).
    #     """
    #     # --- 1. Initial Feature Processing (VSN or simple Dense) ---
    #     # This part processes each input type to a common feature space.
    #     static_context = None
    #     if self.use_vsn and self.static_vsn is not None:
    #         vsn_static_out = self.static_vsn(
    #             static_input, training=training)
    #         static_context = self.static_vsn_grn(
    #             vsn_static_out, training=training
    #             )
    #     elif self.static_input_dim > 0 and self.grn_static is not None:
    #         processed_static = self.static_dense(static_input)
    #         static_context = self.grn_static(
    #             processed_static, training=training)
        
    #     dynamic_processed = dynamic_input
    #     if self.use_vsn and self.dynamic_vsn is not None:
    #         dynamic_processed = self.dynamic_vsn(
    #             dynamic_input, training=training)
    #         dynamic_processed = self.dynamic_vsn_grn(
    #             dynamic_processed, training=training)
    
    #     future_processed = future_input
    #     if self.use_vsn and self.future_vsn is not None:
    #         future_processed = self.future_vsn(
    #             future_input, training=training)
    #         future_processed = self.future_vsn_grn(
    #             future_processed, training=training)
        
    #     # --- 2. Encoder Path (Processes Past Data) ---
    #     # The encoder summarizes the `dynamic_input` from `time_steps`.
    #     encoder_input = self.positional_encoding(
    #         dynamic_processed, training=training)
        
    #     # Multi-scale LSTM processes the historical features.
    #     # It must return sequences for attention.
    #     lstm_output_raw = self.multi_scale_lstm(
    #         encoder_input, training=training
    #     )
        
    #     # If multi_scale_agg is None, lstm_output is (B, units * len(scales))
    #     # If multi_scale_agg is not None, lstm_output is a list of full 
    #     # sequences: [ (B, T', units), ... ]
    #     # lstm_features = aggregate_multiscale(
    #     #     lstm_output_raw, mode= self.multi_scale_agg_mode 
    #     # )
    #     # # Since we are concatenating along the time dimension, we need 
    #     # # all tensors to have the same shape along that dimension.
        
    #     # time_steps = tf_shape(dynamic_processed)[1]
    #     # # Expand lstm_features to (B, 1, features)
    #     # lstm_features = tf_expand_dims(lstm_features, axis=1)
    #     # # Tile to match tf_time steps: (B, T, features)
    #     # encoder_output = tf_tile(lstm_features, [1, time_steps, 1])
        
    #     # # `encoder_output` now holds the history summary.
    #     # # Shape: (batch, time_steps, lstm_units * num_scales)
        
    #     encoder_output = aggregate_multiscale_on_3d( # rename aggregate_multiscale
    #         lstm_output_raw, mode='concat' 
    #     )
        
    #     # Apply dynamic time window to the historical encoder output if desired.
    #     if self.dynamic_time_window is not None:
    #          encoder_output = self.dynamic_time_window(
    #              encoder_output, training=training)
             
    
    #     # --- 3. Decoder Path (Prepares Context for Forecasting) ---
    #     # The decoder's context is built from static and future inputs
    #     # over the `forecast_horizon`.
        
    #     # Tile static context to match the `forecast_horizon`.
    #     static_context_expanded = None
    #     if static_context is not None:
    #         static_context_expanded = tf_expand_dims(static_context, 1)
    #         static_context_expanded = tf_tile(
    #             static_context_expanded, [1, self.forecast_horizon, 1]
    #         )
    
    #     # Add positional encoding to the known future features.
    #     future_processed_with_pos = self.positional_encoding(
    #         future_processed, training=training)
    
    #     # Combine static and future features to form the initial decoder context.
    #     decoder_input_parts = []
    #     if static_context_expanded is not None:
    #         decoder_input_parts.append(static_context_expanded)
    #     if self.future_input_dim > 0:
    #         decoder_input_parts.append(future_processed_with_pos)
    
    #     # If no static or future inputs, create a zero-tensor as placeholder.
    #     if not decoder_input_parts:
    #         # Determine batch size from dynamic_input.
    #         batch_size = tf_shape(dynamic_input)[0]
    #         # Use `embed_dim` or another consistent feature dimension.
    #         decoder_input = tf_zeros(
    #             (batch_size, self.forecast_horizon, self.embed_dim)
    #         )
    #     else:
    #         decoder_input = tf_concat(decoder_input_parts, axis=-1)
    
    #     # --- 4. Attention-based Fusion (Encoder-Decoder Interaction) ---
    #     # The decoder context queries the encoder's historical summary.
    #     cross_attention_output = self.cross_attention(
    #         [decoder_input, encoder_output], # Query, Value( Key)
    #         training=training
    #     )
    
    #     # A GRN to process the output of cross-attention.
    #     cross_attention_processed = self.grn_static(cross_attention_output)
    
    #     # A residual connection for the decoder part.
    #     decoder_context_with_attention = Add()(
    #         [decoder_input, cross_attention_processed]
    #     )
    
    #     # Additional attention mechanisms can refine this context further.
    #     # HierarchicalAttention might now be used on the decoder context.
    #     hierarchical_att_output = self.hierarchical_attention(
    #         [decoder_context_with_attention, decoder_context_with_attention],
    #         training=training
    #     )
    
    #     memory_attention_output = self.memory_augmented_attention(
    #         hierarchical_att_output, training=training
    #     )
        
    #     # --- 5. Final Feature Combination for Decoding ---
    #     # The output of this stage should have shape 
    #     # (batch, forecast_horizon, features).
    #     # We aggregate all the refined context information.
    #     final_features = self.multi_resolution_attention_fusion(
    #          memory_attention_output, training=training
    #     )
        
    #     # Another residual connection to stabilize the final block.
    #     if self.use_residuals:
    #         final_features = Add()(
    #             [final_features, decoder_context_with_attention])
    #         final_features = LayerNormalization()(final_features)
        
    #     # `final_agg` can aggregate features over the time dimension if needed,
    #     # but for MultiDecoder, we typically want to preserve the forecast_horizon.
    #     # We will assume `final_agg` is 'last' or 'average' 
    #     # which is handled after the decoder.
    #     # The result here should be passed to the MultiDecoder.
        
    #     final_features_for_decode = aggregate_time_window_output(
    #         final_features, self.final_agg
    #     )
        
    #     return final_features_for_decode        
 
    # def run_halnet_core0(
    #     self,
    #     static_input: Tensor,
    #     dynamic_input: Tensor,
    #     future_input: Tensor,
    #     training: bool
    # ) -> Tensor:
    #     """
    #     Executes the core data-driven feature extraction pipeline.

    #     This method processes static, dynamic, and future inputs through
    #     VSNs (if enabled), LSTMs, and various attention mechanisms to
    #     produce a final set of features ready for decoding.

    #     Args:
    #         static_input (tf.Tensor): Processed static features.
    #         dynamic_input (tf.Tensor): Processed dynamic historical features.
    #         future_input (tf.Tensor): Processed known future features.
    #         training (bool): Python boolean indicating training mode.

    #     Returns:
    #         tf.Tensor: Features processed by the core, ready for the
    #                    MultiDecoder.
    #     """
    #     # --- 1. Initial Feature Processing (VSN or simple Dense) ---
        
    #     # A. Process Static Features
    #     static_features_grn_out = None
    #     if self.use_vsn and self.static_vsn is not None:
    #         # VSN path for static features
    #         static_context = self.static_vsn(static_input, training=training)
    #         static_features_grn_out = self.static_vsn_grn(
    #             static_context, training=training
    #         )
    #     elif self.static_input_dim > 0:
    #         # Non-VSN path for static features
    #         # self.static_dense_initial is created in _build_halnet_layers
    #         processed_static = self.static_dense_initial(static_input) 
    #         static_features_grn_out = self.grn_static(
    #             processed_static, training=training
    #         )
        
    #     # B. Process Dynamic and Future Time-Varying Features
    #     dynamic_processed = dynamic_input
    #     future_processed = future_input

    #     if self.use_vsn:
    #         if self.dynamic_vsn is not None:
    #             dynamic_processed = self.dynamic_vsn(
    #                 dynamic_input, training=training
    #             )
    #             dynamic_processed = self.dynamic_vsn_grn(
    #                 dynamic_processed, training=training
    #             )
    #         if self.future_vsn is not None:
    #             future_processed = self.future_vsn(
    #                 future_input, training=training
    #             )
    #             future_processed = self.future_vsn_grn(
    #                 future_processed, training=training
    #             )

    #     # --- 2. Temporal Feature Alignment & Embedding ---
    #     # Align future features to the same past time steps as dynamic features
    #     _, future_for_embedding = align_temporal_dimensions(
    #         tensor_ref=dynamic_processed,
    #         tensor_to_align=future_processed,
    #         mode='pad_to_ref',
    #         name="future_for_embedding"
    #     )
        
    #     # VSN path logic
    #     if self.multi_modal_embedding is None:
    #         # Use tf.cond to handle the condition check for symbolic tensors
    #         future_input_is_valid = tf_shape(future_for_embedding)[-1] > 0
    #         # Create a tensor to append based on whether future_for_embedding is valid
    #         # We must ensure the concatenated tensor has a consistent shape.
    #         inputs_to_concat = [dynamic_processed]
    #         inputs_to_concat.append(
    #             tf_cond(
    #                 future_input_is_valid,
    #                 lambda: future_for_embedding,  # If valid, use future_for_embedding
    #                 # If future features are absent, append zeros to keep shape consistent
    #                 # for the residual connection later.
    #                 lambda: tf_zeros_like(future_for_embedding)  # Otherwise, append zeros
    #             )
    #         )
    #         embeddings = Concatenate(axis=-1)(inputs_to_concat)
    #     else: 
    #         # Non-VSN path
    #         # Similar logic for MultiModalEmbedding if it accepts zero-dim tensors
    #         # or we ensure its inputs are consistent.
    #         # Assuming MME path works as intended for now.
    #         embeddings = self.multi_modal_embedding(
    #             [dynamic_processed, future_for_embedding], training=training
    #         )
    
    #     embeddings = self.positional_encoding(embeddings, training=training)
    
    #     if self.use_residuals and self.residual_dense is not None:
    #         # Now, `embeddings` will consistently have shape (..., 2 * embed_dim)
    #         # because we concatenate with zeros if future features are absent.
    #         # The Add() should no longer fail.
    #         embeddings = Add()([embeddings, self.residual_dense(embeddings)])
    

    #     # --- 3. LSTM and Attention Mechanisms ---
    #     lstm_output_raw = self.multi_scale_lstm(
    #         dynamic_processed, training=training # Use VSN-processed dynamic feats
    #     )
    #     lstm_features = aggregate_multiscale(
    #         lstm_output_raw, mode=self.multi_scale_agg_mode
    #     )
        
    #     time_steps_dyn = tf_shape(dynamic_processed)[1]
    #     lstm_features_tiled = tf_tile(
    #         tf_expand_dims(lstm_features, axis=1), [1, time_steps_dyn, 1]
    #     )
        
    #     # Hierarchical Attention inputs need to be of compatible dimension.
    #     # Assuming they are already at embed_dim after VSN-GRN path.
    #     hierarchical_att = self.hierarchical_attention(
    #        [dynamic_processed, future_for_embedding], training=training
    #     )
    #     cross_attention_output = self.cross_attention(
    #         [dynamic_processed, embeddings], training=training
    #     )
    #     memory_attention_output = self.memory_augmented_attention(
    #         hierarchical_att, training=training
    #     )
        
    #     # Tile static context to match temporal dimension for combination
    #     static_features_expanded = None
    #     if static_features_grn_out is not None:
    #         static_features_expanded = tf_tile(
    #             tf_expand_dims(static_features_grn_out, axis=1), 
    #             [1, time_steps_dyn, 1]
    #         )
        
    #     # --- 4. Feature Fusion and Final Processing ---
    #     features_to_combine = [
    #         lstm_features_tiled,
    #         cross_attention_output,
    #         hierarchical_att,
    #         memory_attention_output,
    #     ]
    #     if static_features_expanded is not None:
    #         features_to_combine.insert(0, static_features_expanded)
            
    #     combined_features = Concatenate(axis=-1)(features_to_combine)
        
    #     attention_fusion_output = self.multi_resolution_attention_fusion(
    #         combined_features, training=training
    #     )
    #     time_window_output = self.dynamic_time_window(
    #         attention_fusion_output, training=training
    #     )
    #     final_features_for_decode = aggregate_time_window_output(
    #         time_window_output, self.final_agg
    #     )
        
    #     return final_features_for_decode
    
    
# def _build_halnet_layers8(self):
#     """
#     Instantiates all layers for the core data-driven HALNet architecture.
#     This revised version ensures necessary layers like GRNs are always
#     available for the encoder-decoder architecture.
#     """
#     # --- VSN Layers (conditionally created) ---
#     if self.use_vsn:
#         if self.static_input_dim > 0:
#             self.static_vsn = VariableSelectionNetwork(
#                 num_inputs=self.static_input_dim,
#                 units=self.vsn_units, dropout_rate=self.dropout_rate,
#                 name="static_vsn"
#             )
#             self.static_vsn_grn = GatedResidualNetwork(
#                 units=self.hidden_units, dropout_rate=self.dropout_rate,
#                 name="static_vsn_grn"
#             )
#         else:
#             self.static_vsn, self.static_vsn_grn = None, None

#         if self.dynamic_input_dim > 0:
#             self.dynamic_vsn = VariableSelectionNetwork(
#                 num_inputs=self.dynamic_input_dim, units=self.vsn_units,
#                 dropout_rate=self.dropout_rate, use_time_distributed=True,
#                 name="dynamic_vsn"
#             )
#             self.dynamic_vsn_grn = GatedResidualNetwork(
#                 units=self.embed_dim, dropout_rate=self.dropout_rate,
#                 name="dynamic_vsn_grn"
#             )
#         else:
#             self.dynamic_vsn, self.dynamic_vsn_grn = None, None

#         if self.future_input_dim > 0:
#             self.future_vsn = VariableSelectionNetwork(
#                 num_inputs=self.future_input_dim, units=self.vsn_units,
#                 dropout_rate=self.dropout_rate, use_time_distributed=True,
#                 name="future_vsn"
#             )
#             self.future_vsn_grn = GatedResidualNetwork(
#                 units=self.embed_dim, dropout_rate=self.dropout_rate,
#                 name="future_vsn_grn"
#             )
#         else:
#             self.future_vsn, self.future_vsn_grn = None, None
#     else:
#         # If not using VSN, all VSN-related layers are None.
#         self.static_vsn, self.static_vsn_grn = None, None
#         self.dynamic_vsn, self.dynamic_vsn_grn = None, None
#         self.future_vsn, self.future_vsn_grn = None, None

#     # --- Shared & Non-VSN Path Layers ---
#     # Create a GRN that is ALWAYS available for processing static context
#     # or attention outputs.
#     self.static_processing_grn = GatedResidualNetwork(
#         units=self.attention_units, dropout_rate=self.dropout_rate,
#         activation=self.activation_fn_str, name="static_processing_grn"
#     )

#     # *** NEW: Add a projection layer for the decoder input ***
#     # This ensures a consistent dimension for residual connections.
#     self.decoder_input_projection = Dense(
#         self.attention_units, activation=self.activation_fn_str,
#         name="decoder_input_projection"
#     )
    
#     # Create the initial static dense layer only if VSN is NOT used.
#     if not self.use_vsn and self.static_input_dim > 0:
#         self.static_dense = Dense(
#             self.hidden_units, activation=self.activation_fn_str
#         )
#     else:
#         self.static_dense = None

#     # --- Core Architectural Layers (always created) ---
#     self.positional_encoding = PositionalEncoding()
    
#     # Ensure MultiScaleLSTM is configured to return sequences for the encoder.
#     self.multi_scale_lstm = MultiScaleLSTM(
#         lstm_units=self.lstm_units,
#         scales=self.scales,
#         return_sequences=True # This is critical for the encoder
#     )
#     self.hierarchical_attention = HierarchicalAttention(
#         units=self.attention_units,
#         num_heads=self.num_heads
#     )
#     self.cross_attention = CrossAttention(
#         units=self.attention_units, 
#         num_heads=self.num_heads
#     )
#     self.memory_augmented_attention = MemoryAugmentedAttention(
#         units=self.attention_units, 
#         memory_size=self.memory_size,
#         num_heads=self.num_heads
#     )
#     self.multi_resolution_attention_fusion = MultiResolutionAttentionFusion(
#         units=self.attention_units, 
#         num_heads=self.num_heads
#     )
#     self.dynamic_time_window = DynamicTimeWindow(
#         max_window_size=self.max_window_size
#     )
#     self.multi_decoder = MultiDecoder(
#         output_dim=self._combined_output_target_dim,
#         num_horizons=self.forecast_horizon
#     )
#     self.quantile_distribution_modeling = QuantileDistributionModeling(
#         quantiles=self.quantiles,
#         output_dim=self._combined_output_target_dim
#     )
#     # --- FIX: Instantiate layers for residual connections here ---
#     if self.use_residuals:
#         self.residual_dense = Dense(self.attention_units)
#         # Add layer for the first residual connection in the decoder
#         self.decoder_add_norm = [
#             Add(),
#             LayerNormalization()
#         ]
#         # Add layer for the final residual connection
#         self.final_add_norm = [
#             Add(),
#             LayerNormalization()
#         ]
#     else:
#         self.residual_dense = None
#         self.decoder_add_norm = None
#         self.final_add_norm = None
        

# def run_halnet_core8(
#     self,
#     static_input: Tensor,
#     dynamic_input: Tensor,
#     future_input: Tensor,
#     training: bool
# ) -> Tensor:
#     """
#     Executes the core data-driven feature extraction pipeline using a
#     robust encoder-decoder architecture.
#     """
#     # --- 1. Initial Feature Processing ---
#     static_context = None
#     if self.use_vsn and self.static_vsn is not None:
#         vsn_static_out = self.static_vsn(static_input, training=training)
#         static_context = self.static_vsn_grn(vsn_static_out, training=training)
#     elif self.static_dense is not None: # Non-VSN path
#         # Note: In this path, static_context has `hidden_units`.
#         # This context is combined with `future_processed` before projection.
    
#         processed_static = self.static_dense(static_input)
#         # Use the always-available GRN for processing
#         static_context = self.static_processing_grn(processed_static, training=training)

#     dynamic_processed = dynamic_input
#     if self.use_vsn and self.dynamic_vsn is not None:
#         dynamic_processed = self.dynamic_vsn(dynamic_input, training=training)
#         dynamic_processed = self.dynamic_vsn_grn(dynamic_processed, training=training)

#     future_processed = future_input
#     if self.use_vsn and self.future_vsn is not None:
#         future_processed = self.future_vsn(future_input, training=training)
#         future_processed = self.future_vsn_grn(future_processed, training=training)

#     # --- 2. Encoder Path (Processes Past Data) ---
#     encoder_input = self.positional_encoding(
#         dynamic_processed, training=training)
#     lstm_output_raw = self.multi_scale_lstm(encoder_input, training=training)
    
#     encoder_sequences = aggregate_multiscale_on_3d(
#         lstm_output_raw, mode='concat')
    
#     if self.dynamic_time_window is not None:
#         encoder_sequences = self.dynamic_time_window(
#             encoder_sequences, training=training)
    
#     # --- 3. Decoder Path (Prepares Context for Forecasting) ---
#     static_context_expanded = None
#     if static_context is not None:
#         static_context_expanded = tf_expand_dims(static_context, 1)
#         static_context_expanded = tf_tile(
#             static_context_expanded, [1, self.forecast_horizon, 1]
#         )

#     future_with_pos = self.positional_encoding(
#         future_processed, training=training)

#     decoder_input_parts = []
#     if static_context_expanded is not None:
#         decoder_input_parts.append(static_context_expanded)
#     if self.future_input_dim > 0:
#         decoder_input_parts.append(future_with_pos)

#     if not decoder_input_parts:
#         batch_size = tf_shape(dynamic_input)[0]
#         # Use a consistent dimension, e.g., hidden_units
#         raw_decoder_input = tf_zeros(
#             (batch_size, self.forecast_horizon, self.attention_units))
#     else:
#         raw_decoder_input = tf_concat(decoder_input_parts, axis=-1)

#     # *** FIX: Project the decoder input to a consistent dimension ***
#     projected_decoder_input = self.decoder_input_projection(raw_decoder_input)

#     # --- 4. Attention-based Fusion (Encoder-Decoder Interaction) ---
#     cross_attention_output = self.cross_attention( # Cross Attention use a list of two imputs 
#                                                   # Q, K (V) for value, the K is repeated. 
#                                                   # no need to add V values since K=V
#         #  [decoder_input, encoder_sequences, encoder_sequences], training=training    becomes                                       
#         [projected_decoder_input, encoder_sequences], training=training
#     )

#     # *** FIX: Use the shared, always-available GRN ***
#     cross_attention_processed = self.static_processing_grn(
#         cross_attention_output, training=training
#     )

#     # *** FIX: Perform residual connection with correctly shaped tensors ***
#     # decoder_context_with_attention = Add()(
#     #     [projected_decoder_input, cross_attention_processed]
#     # )
#     decoder_context_with_attention = self.decoder_add_norm[0](
#     [projected_decoder_input, cross_attention_processed]
#     )
#     decoder_context_with_attention = self.decoder_add_norm[1](
#     decoder_context_with_attention
#     )
#     # decoder_context_with_attention = LayerNormalization()(
#     #     decoder_context_with_attention
#     # )
    
#     hierarchical_att_output = self.hierarchical_attention(
#         [decoder_context_with_attention, decoder_context_with_attention], 
#         training=training
#     )
#     memory_attention_output = self.memory_augmented_attention(
#         hierarchical_att_output, training=training
#     )
    
#     # --- 5. Final Feature Combination for Decoding ---
#     final_features = self.multi_resolution_attention_fusion(
#          memory_attention_output, training=training
#     )
#     if self.use_residuals and self.residual_dense:
#         # Project the context to match the shape for residual connection
#         residual_base = self.residual_dense(decoder_context_with_attention)
#         # *** FIX: Use pre-instantiated layers for the final residual connection ***
#         final_features = self.final_add_norm[0]([final_features, residual_base])
#         final_features = self.final_add_norm[1](final_features)
    
#     final_features_for_decode = aggregate_time_window_output(
#         final_features, self.final_agg
#     )
    
#     return final_features_for_decode

# def _build_halnet_layers1(self):
#     """
#     Instantiates all layers for the core data-driven HALNet architecture,
#     optionally including VariableSelectionNetworks.
#     """
#     if self.use_vsn:
#         if self.static_input_dim > 0:
#             self.static_vsn = VariableSelectionNetwork(
#                 num_inputs=self.static_input_dim,
#                 units=self.vsn_units, 
#                 dropout_rate=self.dropout_rate,
#                 activation=self.activation_fn_str,
#                 use_batch_norm=self.use_batch_norm, 
#                 name="static_vsn"
#             )
#             # GRN after static VSN 
#             # (common in TFT to refine VSN output)
#             self.static_vsn_grn = GatedResidualNetwork(
#                 units=self.hidden_units, 
#                 dropout_rate=self.dropout_rate,
#                 activation=self.activation_fn_str,
#                 use_batch_norm=self.use_batch_norm,
#                 name="static_vsn_grn"
#             )
#         else:
#             self.static_vsn = None
#             self.static_vsn_grn = None

#         if self.dynamic_input_dim > 0:
#             self.dynamic_vsn = VariableSelectionNetwork(
#                 num_inputs=self.dynamic_input_dim,
#                 units=self.vsn_units,
#                 dropout_rate=self.dropout_rate,
#                 use_time_distributed=True,
#                 activation=self.activation_fn_str,
#                 use_batch_norm=self.use_batch_norm,
#                 name="dynamic_vsn"
#             )
#             # GRN for dynamic VSN output
#             # (optional, but good for consistency)
#             self.dynamic_vsn_grn = GatedResidualNetwork(
#                 units=self.embed_dim, 
#                 dropout_rate=self.dropout_rate,
#                 activation=self.activation_fn_str,
#                 use_batch_norm=self.use_batch_norm,
#                 name="dynamic_vsn_grn"
#             )

#         else: 
#             # Should not happen as dynamic_input_dim must be > 0
#             self.dynamic_vsn = None
#             self.dynamic_vsn_grn = None


#         if self.future_input_dim > 0:
#             self.future_vsn = VariableSelectionNetwork(
#                 num_inputs=self.future_input_dim,
#                 units=self.vsn_units,
#                 dropout_rate=self.dropout_rate,
#                 use_time_distributed=True,
#                 activation=self.activation_fn_str,
#                 use_batch_norm=self.use_batch_norm,
#                 name="future_vsn"
#             )
#             # GRN for future VSN output
#             self.future_vsn_grn = GatedResidualNetwork(
#                 units=self.embed_dim, # Target dim for MME/Attention
#                 dropout_rate=self.dropout_rate,
#                 activation=self.activation_fn_str,
#                 use_batch_norm=self.use_batch_norm,
#                 name="future_vsn_grn"
#             )
#         else:
#             self.future_vsn = None
#             self.future_vsn_grn = None
        
#                 # If VSNs are handling the embedding, we might not need MME.
#         # Its attribute can be set to None, and the `run_halnet_core` method
#         # will use the VSN outputs directly.
#         self.multi_modal_embedding = None
        
#         logger.info(
#             " VSN is enabled. MultiModalEmbedding"
#             " for dynamic/future inputs will be bypassed."
#             )
#     else: # If not using VSNs
#         self.static_vsn, self.static_vsn_grn = None, None
#         self.dynamic_vsn, self.dynamic_vsn_grn = None, None
#         self.future_vsn, self.future_vsn_grn = None, None
#         # If not using VSN, initial processing for static features
#         if self.static_input_dim > 0:
#              self.static_dense_initial = Dense( # Renamed to avoid clash
#                 self.hidden_units, activation=self.activation_fn_str
#             )
#         else:
#             self.static_dense_initial = None

#     # --- Subsequent Layers (Inputs to these might change based on VSN usage) ---
#     # MultiModalEmbedding now takes outputs of dynamic_vsn_grn and future_vsn_grn
#     # Or, if VSNs are not used, it takes raw (or simply Densed) dynamic/future inputs.
#     # The VSN outputs are already "embedded" to vsn_units or embed_dim.
#     # So, MultiModalEmbedding might become redundant or need to adapt.
#     # For TFT, VSN outputs (after GRN) directly feed into LSTM/attention.
#     # Let's assume VSN outputs (after GRN) are the new "embedded" inputs.
#     # So, we might not need self.multi_modal_embedding if VSNs are used
#     # and their output GRNs project to self.embed_dim.
#     if not self.use_vsn or (
#             self.dynamic_vsn_grn is None and self.future_vsn_grn is None) :
#         # If VSNs are not used for dynamic/future, or not fully configured,
#         # keep MME for raw inputs.
#         # If no VSN, we need MME for initial feature processing.
#        self.multi_modal_embedding = MultiModalEmbedding(self.embed_dim)
#        logger.info(
#            "VSN is disabled. MultiModalEmbedding"
#            " will be used for raw inputs.")
#     else:
#         # If VSNs are used and their GRNs output embed_dim, MME might be skipped
#         # for dynamic/future as they are already processed.
#         # Or MME could take these processed VSN outputs if they are of different dims.
#         # For simplicity, let's assume if VSNs are used, their outputs (after GRN)
#         # are what we use, and MME might be for other modalities if any.
#         # If dynamic_vsn_grn and future_vsn_grn output embed_dim, we can
#         # directly concatenate them if needed.
#         self.multi_modal_embedding = None # Or adapt its usage

#     self.positional_encoding = PositionalEncoding()
#     self.multi_scale_lstm = MultiScaleLSTM(
#         lstm_units=self.lstm_units,
#         scales=self.scales,
#         return_sequences=self.lstm_return_sequences
#     )
#     self.hierarchical_attention = HierarchicalAttention(
#         units=self.attention_units, num_heads=self.num_heads
#     )
#     self.cross_attention = CrossAttention(
#         units=self.attention_units, num_heads=self.num_heads
#     )
#     self.memory_augmented_attention = MemoryAugmentedAttention(
#         units=self.attention_units, memory_size=self.memory_size,
#         num_heads=self.num_heads
#     )
#     self.multi_resolution_attention_fusion = MultiResolutionAttentionFusion(
#         units=self.attention_units, num_heads=self.num_heads
#     )
#     self.dynamic_time_window = DynamicTimeWindow(
#         max_window_size=self.max_window_size
#     )
#     self.multi_decoder = MultiDecoder(
#         output_dim=self._combined_output_target_dim,
#         num_horizons=self.forecast_horizon
#     )
#     self.quantile_distribution_modeling = QuantileDistributionModeling(
#         quantiles=self.quantiles,
#         output_dim=self._combined_output_target_dim
#     )
    
#     # LearnedNormalization might be applied to VSN inputs or outputs,
#     # or raw inputs if VSN not used.
#     # We have option to apply to raw inputs before VSN if VSN is used.
#     # Or, it could be part of the VSN's internal GRN processing.
#     # For now, we keep it as a general layer.
#     self.learned_normalization = LearnedNormalization()

#     # Static processing if VSN is NOT used for static features
#     if not self.use_vsn or self.static_vsn is None:
#         self.static_dense = Dense( # This was self.static_dense_initial before
#             self.hidden_units, activation=self.activation_fn_str
#         )
#         self.static_dropout = Dropout(self.dropout_rate)
#         if self.use_batch_norm:
#             self.static_batch_norm = LayerNormalization()
#         else:
#             self.static_batch_norm = None
#         self.grn_static = GatedResidualNetwork( 
#             units=self.hidden_units, dropout_rate=self.dropout_rate,
#             activation=self.activation_fn_str,
#             use_batch_norm=self.use_batch_norm
#         )
#     else: # If static_vsn is used, these specific layers might not be needed
#           # as static_vsn_grn produces the final static context
#         self.static_dense = None
#         self.static_dropout = None
#         self.static_batch_norm = None
#         self.grn_static = None # static_vsn_grn takes this role
        

#     self.residual_dense = Dense(
#         2 * self.embed_dim # This was tied to MME output usually
#     ) if self.use_residuals else None


# # new revise 
#     def run_halnet_core_(
#         self,
#         static_input:Tensor,
#         dynamic_input: Tensor,
#         future_input: Tensor,
#         training: bool
#     ) -> Tensor:
#         """Executes data-driven pipeline using an encoder-decoder.

#         This revised method correctly separates the processing of past
#         (dynamic) and future inputs to handle cases where `time_steps`
#         and `forecast_horizon` are different.

#         Args:
#             static_input: Processed static features.
#             dynamic_input: Processed dynamic historical features.
#                            Shape: (batch, time_steps, features).
#             future_input: Processed known future features.
#                           Shape: (batch, forecast_horizon, features).
#             training: Python boolean indicating training mode.

#         Returns:
#             A feature tensor for the forecast horizon, ready for the
#             MultiDecoder. Shape is determined by `self.final_agg`.
#         """
#         # Get the length of the historical time window
#         #ctime_steps = tf_shape(dynamic_input)[1]

#         # --- 1. Initial Feature Processing ---
#         # Process static inputs first.
#         static_context = None
#         if self.use_vsn and self.static_vsn is not None:
#             static_out = self.static_vsn(static_input, training=training)
#             static_context = self.static_vsn_grn(
#                 static_out, training=training
#             )
#         elif self.static_dense is not None:
#             static_out = self.static_dense(static_input)
#             static_context = self.grn_static_non_vsn(
#                 static_out, training=training
#             )

#         logger.debug(
#             f"Static context shape: {getattr(static_context, 'shape', 'None')}"
#         )

#         # --- 2. Encoder Path (Processes Past Data) ---
#         # The encoder uses `dynamic_input` and the historical part of
#         # `future_input`.
        
#         if self.mode =='tft_like': 
#             # In models like TFT, "future" inputs are known for both the past
#             # and future. We slice the part corresponding to the encoder's timeline.
#             # This assumes `future_input` has length `time_steps + forecast_horizon`.
#             # If it only has length `forecast_horizon`, this logic needs adjustment.
#             # Let's assume for now `future_input` is just for the decoder.
#             # The encoder will only use `dynamic_input`.
#             future_for_encoder = None
#             time_steps = tf_shape(dynamic_input)[1]
#             # For TFT-style, slice the historical part of future inputs
#             future_for_encoder = future_input[:, :time_steps, :]
        
#             dyn_proc = self.dynamic_dense(
#                 dynamic_input) if not self.use_vsn else self.dynamic_vsn_grn(
#                     self.dynamic_vsn(dynamic_input))
            
#             encoder_input_parts = [dyn_proc]
#             if future_for_encoder is not None:
#                 fut_enc_proc = self.future_dense(
#                     future_for_encoder) if not self.use_vsn else self.future_vsn_grn(
#                         self.future_vsn(future_for_encoder))
                        
#                 encoder_input_parts.append(fut_enc_proc)
                
#             encoder_inputs = tf_concat(encoder_input_parts, axis=-1)
        
#             fut_proc = None
#             if self.future_input_dim > 0:
#                 # For TFT-style, slice the forecast part of future inputs
#                 fut_proc = future_input[:, time_steps:, :]
#             else:
#                 # For standard encoder-decoder, 
#                 # the entire future_input is for the decoder
#                 fut_proc = future_input
#         else: 
            
#             dyn_proc = dynamic_input
#             if self.use_vsn and self.dynamic_vsn is not None:
#                 dyn_proc = self.dynamic_vsn_grn(self.dynamic_vsn(
#                     dynamic_input, training=training), training=training)
          
#             fut_proc = future_input
#             # Process future features for the decoder.
#             if self.use_vsn and self.future_vsn is not None:
#                 fut_proc = self.future_vsn_grn(self.future_vsn(
#                     future_input, training=training), training=training)
                
#             encoder_inputs = dyn_proc # Self encoder-decoder architecture. 
           
#         logger.debug(f"Shape after VSN/initial processing: "
#                      f"Dynamic={getattr(dyn_proc, 'shape', 'N/A')}, "
#                      f"Future={getattr(fut_proc, 'shape', 'N/A')}")
        
#         encoder_input = self.positional_encoding(
#             encoder_inputs, training=training
#             )
        
#         lstm_out = self.multi_scale_lstm(
#             encoder_input, training=training
#         )
#         # `aggregate_multiscale` with 'concat' now correctly returns a 3D tensor
#         # by padding and concatenating features.
#         encoder_sequences = aggregate_multiscale_on_3d(
#             lstm_out, mode='concat'
#         )
        
#         if self.dynamic_time_window is not None:
#             encoder_sequences = self.dynamic_time_window(
#                 encoder_sequences, training=training)
            
#         logger.debug(
#             f"Encoder output sequence shape: {encoder_sequences.shape}"
#         )
#         # --- 3. Decoder Path (Prepares Context for Forecasting) ---
#         # The decoder uses `static_context` and `future_input` over the
#         # `forecast_horizon`.
#         # Combine static context with future features for the decoder.
#         static_expanded =None 
#         if static_context is not None:
#             static_expanded = tf_expand_dims(static_context, 1)
#             static_expanded = tf_tile(
#                 static_expanded, [1, self.forecast_horizon, 1]
#             )
            
#         future_with_pos = self.positional_encoding(
#             fut_proc, training=training
#         )
        
#         decoder_parts = []
#         if static_expanded is not None :
#             decoder_parts.append(static_expanded)
#         if self.future_input_dim > 0: 
#             decoder_parts.append(future_with_pos)
            
#         if not decoder_parts: 
#             batch_size = tf_shape(dynamic_input)[0]
#             raw_decoder_input = tf_zeros(
#                 (batch_size, self.forecast_horizon, self.attention_units))
#         else: 
#             raw_decoder_input = tf_concat(decoder_parts, axis=-1)
            
#         # Project decoder input to the right dimension for attention.
#         projected_decoder_input = self.decoder_input_projection(
#             raw_decoder_input
#         )
        
#         logger.debug(
#             f"Projected decoder input shape: {projected_decoder_input.shape}"
#         )

#         # --- 4. Attention-based Fusion ---
#         # The decoder context (query) attends to the encoder sequences (key/value).
#         cross_att_out = self.cross_attention(
#             [projected_decoder_input, encoder_sequences], 
#             training=training
#         )
        
#         att_proc = self.attention_processing_grn(
#             cross_att_out, training=training
#         )
#         # Process attention output and add residual connection.
#         if self.use_residuals and self.decoder_add_norm:
#             context_att = self.decoder_add_norm[0](
#                 [projected_decoder_input, att_proc]
#             )
#             context_att = self.decoder_add_norm[1](context_att)
#         else:
#             context_att = cross_att_out
            
#         # Apply further attention layers to refine the context.
#         hier_att_out = self.hierarchical_attention(
#             [context_att, context_att], training=training # Self-attention
#         )
#         mem_att_out = self.memory_augmented_attention(
#             hier_att_out, training=training
#         )
        
#         # --- 5. Final Combination and Aggregation ---
#         final_features = self.multi_resolution_attention_fusion(
#             mem_att_out, training=training
#         )

#         if self.use_residuals and self.final_add_norm:
#             res_base = self.residual_dense(context_att)
#             final_features = self.final_add_norm[0](
#                 [final_features, res_base]
#             )
#             final_features = self.final_add_norm[1](final_features)
        
#         logger.debug(
#             f"Shape after final fusion: {final_features.shape}"
#         )

#         # Collapse the time dimension to get a single vector for the decoder.
#         return aggregate_time_window_output(final_features, self.final_agg)


    
#     def call_(self, inputs: List[Optional[Tensor]], training: bool = False) -> Tensor:
#         """Forward pass for the HALNet model."""
#         static_p, dynamic_p, future_p = validate_model_inputs(
#             inputs=inputs, static_input_dim=self.static_input_dim,
#             dynamic_input_dim=self.dynamic_input_dim,
#             future_covariate_dim=self.future_input_dim,
#             mode='strict', 
#             model_name='xtft' # Re-use strict validation
#         )
        
#         final_features = self.run_halnet_core(
#             static_p, dynamic_p, future_p, training=training)
        
#         decoded_outputs = self.multi_decoder(final_features, training=training)
        
#         if self.quantiles is not None:
#             return self.quantile_distribution_modeling(decoded_outputs)
        
#         return decoded_outputs


# ===============================================

    # def _build_halnet_layers(self):
    #     """Instantiates all layers for the HALNet architecture."""
    #     # This is where all Keras layers are created to avoid issues
    #     # with tf.function recompilation.
    #     if self.use_vsn:
    #         if self.static_input_dim > 0:
    #             self.static_vsn = VariableSelectionNetwork(
    #                 num_inputs=self.static_input_dim,
    #                 units=self.vsn_units,
    #                 dropout_rate=self.dropout_rate, name="static_vsn")
    #             self.static_vsn_grn = GatedResidualNetwork(
    #                 units=self.hidden_units,
    #                 dropout_rate=self.dropout_rate, name="static_vsn_grn")
    #         else:
    #             self.static_vsn, self.static_vsn_grn = None, None

    #         if self.dynamic_input_dim > 0:
    #             self.dynamic_vsn = VariableSelectionNetwork(
    #                 num_inputs=self.dynamic_input_dim,
    #                 units=self.vsn_units, use_time_distributed=True,
    #                 dropout_rate=self.dropout_rate, name="dynamic_vsn")
    #             self.dynamic_vsn_grn = GatedResidualNetwork(
    #                 units=self.embed_dim,
    #                 dropout_rate=self.dropout_rate, 
    #                 name="dynamic_vsn_grn"
    #             )
    #         else:
    #             self.dynamic_vsn, self.dynamic_vsn_grn = None, None

    #         if self.future_input_dim > 0:
    #             self.future_vsn = VariableSelectionNetwork(
    #                 num_inputs=self.future_input_dim,
    #                 units=self.vsn_units, use_time_distributed=True,
    #                 dropout_rate=self.dropout_rate, name="future_vsn")
    #             self.future_vsn_grn = GatedResidualNetwork(
    #                 units=self.embed_dim,
    #                 dropout_rate=self.dropout_rate, name="future_vsn_grn")
    #         else:
    #             self.future_vsn, self.future_vsn_grn = None, None
    #     else:
    #         # If not using VSN, ensure all related attributes are None.
    #         self.static_vsn, self.static_vsn_grn = None, None
    #         self.dynamic_vsn, self.dynamic_vsn_grn = None, None
    #         self.future_vsn, self.future_vsn_grn = None, None

    #     # A GRN for processing attention outputs.
    #     self.attention_processing_grn = GatedResidualNetwork(
    #         units=self.attention_units,
    #         dropout_rate=self.dropout_rate,
    #         activation=self.activation_fn_str,
    #         name="attention_processing_grn")
            
    #     # A projection layer for the decoder input.
    #     self.decoder_input_projection = Dense(
    #         self.attention_units,
    #         activation=self.activation_fn_str,
    #         name="decoder_input_projection")

    #     if not self.use_vsn:
    #         # Create dense layers for non-VSN path.
    #         if self.static_input_dim > 0:
    #             self.static_dense = Dense(
    #                 self.hidden_units, activation=self.activation_fn_str)
    #             self.grn_static_non_vsn = GatedResidualNetwork(
    #                 units=self.hidden_units,
    #                 dropout_rate=self.dropout_rate,
    #                 activation=self.activation_fn_str,
    #                 name="grn_static_non_vsn")
    #         else:
    #             self.static_dense, self.grn_static_non_vsn = None, None
                
    #         # Create dense layers for dynamic and future features
    #         # for non-VSN path
    #         self.dynamic_dense = Dense(self.embed_dim)
    #         self.future_dense = Dense(self.embed_dim)
    #     else:
    #         self.static_dense, self.grn_static_non_vsn = None, None
    #         self.dynamic_dense, self.future_dense = None, None
            
    #     # --- Core Architectural Layers (Always Created) ---
    #     self.positional_encoding = PositionalEncoding()
    #     self.multi_scale_lstm = MultiScaleLSTM(
    #         lstm_units=self.lstm_units, scales=self.scales,
    #         return_sequences=True)
    #     self.cross_attention = CrossAttention(
    #         units=self.attention_units, num_heads=self.num_heads)
    #     self.hierarchical_attention = HierarchicalAttention(
    #         units=self.attention_units, num_heads=self.num_heads)
    #     self.memory_augmented_attention = MemoryAugmentedAttention(
    #         units=self.attention_units,
    #         memory_size=self.memory_size, num_heads=self.num_heads)
    #     self.multi_resolution_attention_fusion = \
    #         MultiResolutionAttentionFusion(
    #             units=self.attention_units, num_heads=self.num_heads)
        
    #     self.dynamic_time_window = DynamicTimeWindow(
    #         max_window_size=self.max_window_size
    #     )
        
    #     # Final output layers
    #     self.multi_decoder = MultiDecoder(
    #         output_dim=self.output_dim,
    #         num_horizons=self.forecast_horizon)
    #     self.quantile_distribution_modeling = QuantileDistributionModeling(
    #         quantiles=self.quantiles, output_dim=self.output_dim)

    #     # --- Layers for Residual Connections ---
    #     if self.use_residuals:
    #         self.decoder_add_norm = [Add(), LayerNormalization()]
    #         self.final_add_norm = [Add(), LayerNormalization()]
    #         self.residual_dense = Dense(self.attention_units)
    #     else:
    #         self.decoder_add_norm, self.final_add_norm, self.residual_dense = \
    #             None, None, None
  
    # def run_halnet_core(self, static_input, dynamic_input, future_input, training):
    #     """Executes data-driven pipeline using an encoder-decoder."""
        
    #     # Get the lookback window size from the dynamic input tensor.
    #     time_steps = tf_shape(dynamic_input)[1]

    #     # 1. Initial Feature Processing & Slicing based on mode
    #     static_context, dyn_proc, fut_enc_proc, fut_dec_proc = (
    #         None, dynamic_input, None, future_input
    #     )
    #     if self.use_vsn:
    #         if self.static_vsn:
    #             static_context = self.static_vsn_grn(self.static_vsn(
    #                 static_input, training=training), training=training)
    #         if self.dynamic_vsn:
    #             dyn_proc = self.dynamic_vsn_grn(self.dynamic_vsn(
    #                 dynamic_input, training=training), training=training)
    #         if self.future_vsn:
    #             # Process the entire future tensor first
    #             future_processed = self.future_vsn_grn(self.future_vsn(
    #                 future_input, training=training), training=training)
    #     else: # Non-VSN path
    #         if self.static_dense:
    #             static_context = self.grn_static_non_vsn(self.static_dense(
    #                 static_input), training=training)
    #         dyn_proc = self.dynamic_dense(dynamic_input)
    #         future_processed = self.future_dense(future_input)

    #     logger.debug("Static context shape: %s",
    #                  getattr(static_context, 'shape', 'N/A'))
        
    #     # Handle TFT-like input slicing
    #     if self.mode == 'tft_like':
    #         fut_enc_proc = future_processed[:, :time_steps, :]
    #         fut_dec_proc = future_processed[:, time_steps:, :]
    #     else: # For pihal_like, encoder does not use future inputs
    #         fut_enc_proc = None
    #         fut_dec_proc = future_processed
            
    #     # 2. Encoder Path (Processes Past Data)
    #     encoder_input_parts = [dyn_proc]
    #     if fut_enc_proc is not None:
    #         encoder_input_parts.append(fut_enc_proc)
    #     encoder_raw = tf_concat(encoder_input_parts, axis=-1)
    #     encoder_input = self.positional_encoding(encoder_raw, training=training)
    #     lstm_out = self.multi_scale_lstm(encoder_input, training=training)
    #     encoder_sequences = aggregate_multiscale_on_3d(lstm_out, mode='concat')
        
    #     # 3. Decoder Path (Prepares Context for Forecasting) 
    #     static_expanded = None
    #     if static_context is not None:
    #         static_expanded = tf_expand_dims(static_context, 1)
    #         static_expanded = tf_tile(
    #             static_expanded, [1, self.forecast_horizon, 1])
    #     future_with_pos = self.positional_encoding(fut_dec_proc, training=training)
        
    #     decoder_parts = [future_with_pos]
    #     if static_expanded is not None:
    #         decoder_parts.append(static_expanded)
    #     raw_decoder_input = tf_concat(decoder_parts, axis=-1)
    #     projected_decoder_input = self.decoder_input_projection(raw_decoder_input)

    #     logger.debug(
    #         "Projected decoder input shape: %s",
    #         projected_decoder_input.shape
    #     )
    #     # 4. Attention Fusion
    #     cross_att_out = self.cross_attention(
    #         [projected_decoder_input, encoder_sequences], training=training)
    #     att_proc = self.attention_processing_grn(cross_att_out, training=training)
        
    #     if self.use_residuals and self.decoder_add_norm:
    #         context_att = self.decoder_add_norm[0]([projected_decoder_input, att_proc])
    #         context_att = self.decoder_add_norm[1](context_att)
    #     else:
    #         context_att = att_proc
            
    #     # 5. Final Processing
    #     # Apply further attention layers to refine the context.
    #     hier_att_out = self.hierarchical_attention(
    #         [context_att, context_att], training=training # Self-attention
    #     )
    #     mem_att_out = self.memory_augmented_attention(
    #         hier_att_out, training=training
    #     )
        
    #     final_features = self.multi_resolution_attention_fusion(mem_att_out)
        
    #     if self.use_residuals and self.final_add_norm:
    #         res_base = self.residual_dense(context_att)
    #         final_features = self.final_add_norm[0]([final_features, res_base])
    #         final_features = self.final_add_norm[1](final_features)
        
    #     logger.debug("Shape after final fusion: %s", final_features.shape)
        
    #     return aggregate_time_window_output(final_features, self.final_agg)

    # def call(
    #         self, inputs: List[Optional[Tensor]], training: bool = False) -> Tensor:
    #     """Forward pass for the HALNet model."""
        
    #     expected_future_span = None
    #     if self.mode == 'tft_like':
    #         expected_future_span = self.max_window_size + self.forecast_horizon
    #     else: # pihal_like
    #         expected_future_span = self.forecast_horizon

    #     static_p, dynamic_p, future_p = validate_model_inputs(
    #         inputs=inputs, static_input_dim=self.static_input_dim,
    #         dynamic_input_dim=self.dynamic_input_dim,
    #         future_covariate_dim=self.future_input_dim,
    #         # Pass the expected span to the validator if it supports it
    #         # Or perform a check here.
    #         mode='strict', 
    #         model_name='xtft'
    #     )
    #     # Convert the Python int to a tensor so the comparison is graph‑safe
    #     # Check future_input shape based on mode
    #     actual_future_span = tf_shape(future_p)[1]
    #     expected_span_tensor = tf_convert_to_tensor(
    #         expected_future_span, dtype=actual_future_span.dtype
    #     )
    
    #     tf_assert_equal(   # raises InvalidArgumentError in graph mode
    #         actual_future_span,
    #         expected_span_tensor ,
    #         message=(
    #             f"For mode='{self.mode}', `future_input` time dimension "
    #             f"must be {expected_future_span} but is "
    #             f"{actual_future_span}."
    #         ),
    #     )

    #     final_features = self.run_halnet_core(
    #         static_p, dynamic_p, future_p, training=training)
    #     decoded_outputs = self.multi_decoder(
    #         final_features, training=training)
        
    #     if self.quantiles is not None:
    #         return self.quantile_distribution_modeling(decoded_outputs)
        
    #     return decoded_outputs
    
    # def call(
    #         self, inputs: List[Optional[Tensor]], training: bool = False) -> Tensor:
    #     """Forward pass for the HALNet model."""
        
    #     expected_future_span = None
    #     if self.mode == 'tft_like':
    #         expected_future_span = self.max_window_size + self.forecast_horizon
    #     else: # pihal_like
    #         expected_future_span = self.forecast_horizon

    #     static_p, dynamic_p, future_p = validate_model_inputs(
    #         inputs=inputs, static_input_dim=self.static_input_dim,
    #         dynamic_input_dim=self.dynamic_input_dim,
    #         future_covariate_dim=self.future_input_dim,
    #         # Pass the expected span to the validator if it supports it
    #         # Or perform a check here.
    #         mode='strict', 
    #         model_name='xtft'
    #     )
    #     # Convert the Python int to a tensor so the comparison is graph‑safe
    #     # Check future_input shape based on mode
    #     actual_future_span = tf_shape(future_p)[1]
    #     expected_span_tensor = tf_convert_to_tensor(
    #         expected_future_span, dtype=actual_future_span.dtype
    #     )
    
    #     tf_assert_equal(   # raises InvalidArgumentError in graph mode
    #         actual_future_span,
    #         expected_span_tensor ,
    #         message=(
    #             f"For mode='{self.mode}', `future_input` time dimension "
    #             f"must be {expected_future_span} but is "
    #             f"{actual_future_span}."
    #         ),
    #     )

    #     final_features = self.run_halnet_core(
    #         static_p, dynamic_p, future_p, training=training)
    #     decoded_outputs = self.multi_decoder(
    #         final_features, training=training)
        
    #     if self.quantiles is not None:
    #         return self.quantile_distribution_modeling(decoded_outputs)
        
    #     return decoded_outputs

    # def get_config(self):
    #     """Returns the configuration of the HALNet model."""
    #     config = super().get_config()
    #     config.update({
    #         'static_input_dim': self.static_input_dim,
    #         'dynamic_input_dim': self.dynamic_input_dim,
    #         'future_input_dim': self.future_input_dim,
    #         'output_dim': self.output_dim,
    #         'forecast_horizon': self.forecast_horizon,
    #         'quantiles': self.quantiles,
    #         'embed_dim': self.embed_dim,
    #         'hidden_units': self.hidden_units,
    #         'lstm_units': self.lstm_units,
    #         'attention_units': self.attention_units,
    #         'num_heads': self.num_heads,
    #         'dropout_rate': self.dropout_rate,
    #         'max_window_size': self.max_window_size,
    #         'memory_size': self.memory_size,
    #         'scales': self.scales,
    #         'multi_scale_agg': self.multi_scale_agg_mode,
    #         'final_agg': self.final_agg,
    #         'activation': self.activation_fn_str,
    #         'use_residuals': self.use_residuals,
    #         'use_vsn': self.use_vsn,
    #         'vsn_units': self.vsn_units,
    #         'mode': self.mode 
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config, custom_objects=None):
    #     return cls(**config)