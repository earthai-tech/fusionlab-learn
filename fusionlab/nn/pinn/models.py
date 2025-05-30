
# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""[docstring here ]]
"""

from textwrap import dedent # noqa 
from numbers import Real, Integral  
from typing import List, Optional, Union, Dict, Tuple  

from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.property import NNLearner 
from ...compat.sklearn import validate_params, Interval, StrOptions 
from ...utils.deps_utils import ensure_pkg

from .. import KERAS_DEPS, KERAS_BACKEND, dependency_message
 
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
    Variable =KERAS_DEPS.Variable 
    AddLayer =KERAS_DEPS.Add
    Constant =KERAS_DEPS.Constant 
    GradientTape =KERAS_DEPS.GradientTape 
    
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    
    tf_zeros_like= KERAS_DEPS.zeros_like
    tf_zeros =KERAS_DEPS.zeros
    tf_reduce_mean =KERAS_DEPS.reduce_mean
    tf_square =KERAS_DEPS.square
    tf_constant =KERAS_DEPS.constant 
    tf_log = KERAS_DEPS.log
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
    tf_log =KERAS_DEPS.log 
    tf_exp =KERAS_DEPS.exp 
    tf_rank =KERAS_DEPS.rank 
    
    tf_autograph=KERAS_DEPS.autograph
    tf_autograph.set_verbosity(0)
    
    from .._tensor_validation import validate_model_inputs
    from .._tensor_validation import align_temporal_dimensions
    

    from ..utils import set_default_params
    from ..components import (
            Activation, 
            CrossAttention,
            DynamicTimeWindow,
            GatedResidualNetwork,
            HierarchicalAttention,
            LearnedNormalization,
            MemoryAugmentedAttention,
            MultiDecoder,
            MultiModalEmbedding,
            MultiResolutionAttentionFusion,
            MultiScaleLSTM,
            QuantileDistributionModeling,
            VariableSelectionNetwork,
            PositionalEncoding, 
            aggregate_multiscale, 
            aggregate_time_window_output
        )
    from .op import process_pinn_inputs, compute_consolidation_residual 
    
    
DEP_MSG = dependency_message('nn.pinn.models') 

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

__all__ =["PIHALNet"] 

@register_keras_serializable('fusionlab.nn.pinn', name="PIHALNet")
class PIHALNet(Model, NNLearner):
    """
    Physics-Informed Hybrid Attentive LSTM Network (PIHALNet).

    This model integrates a data-driven forecasting architecture, based on
    LSTMs and multiple attention mechanisms, with physics-informed
    constraints derived from the governing equations of land subsidence.
    It is designed to produce physically consistent, multi-horizon
    probabilistic forecasts for both subsidence and groundwater levels,
    while also offering the capability to discover physical parameters
    from observational data.

    The architecture can operate in two modes for its physical
    coefficients:
    1.  **Parameter Specification:** Use predefined physical constants.
    2.  **Parameter Discovery:** Treat physical constants as trainable
        variables to be learned during training (default).
    """
    @validate_params({
        # --- Data Shape Parameters ---
        "static_input_dim": [Interval(Integral, 0, None, closed='left')],
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "future_input_dim": [Interval(Integral, 0, None, closed='left')],
        "output_subsidence_dim": [Interval(Integral, 1, None, closed='left')],
        "output_gwl_dim": [Interval(Integral, 1, None, closed='left')],
        
        # --- Core Architectural Parameters ---
        "embed_dim": [Interval(Integral, 1, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": [Interval(Integral, 1, None, closed='left'), None],
        "attention_units": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        
        # --- Forecasting & Component Parameters ---
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')], 
        "quantiles": ['array-like', StrOptions({'auto'}), None],
        "max_window_size": [Interval(Integral, 1, None, closed='left')],
        "memory_size": [Interval(Integral, 1, None, closed='left')], 
        "scales": ['array-like', StrOptions({'auto'}), None],
        "multi_scale_agg": [StrOptions({
            "last", "average", "flatten", "auto", "sum", "concat"}), None],
        "final_agg": [StrOptions({"last", "average", "flatten"})],
        # --- Behavior & Style Parameters ---
        "activation": [str, callable],
        "use_residuals": [bool],
        "use_batch_norm": [bool],

        # --- PINN-Specific Parameters ---
        "pinn_coefficient_C": [str, Real, None, StrOptions({"learnable"})],
        'use_vsn': [bool], 
        'vsn_units': [Interval(Integral, 0, None, closed="left"), None]
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)   
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_subsidence_dim: int = 1,
        output_gwl_dim: int = 1,
        embed_dim: int = 32,       # Used by MME, and as target for VSN outputs
        hidden_units: int = 64,    # General GRN units, can also be VSN output units
        lstm_units: int = 64,
        attention_units: int = 32,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        forecast_horizon: int = 1,
        quantiles: Optional[List[float]] = None,
        max_window_size: int = 10,
        memory_size: int = 100,
        scales: Optional[List[int]] = None,
        multi_scale_agg: str = 'last',
        final_agg: str = 'last',
        activation: str = 'relu',
        use_residuals: bool = True,
        use_batch_norm: bool = False,
        pinn_coefficient_C: Union[str, float, None] = 'learnable',
        # New VSN parameters
        use_vsn: bool = True,
        vsn_units: Optional[int] = None, # Units for GRNs within VSN, defaults to hidden_units
        name: str = "PIHALNet",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # --- Store all parameters ---
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.output_subsidence_dim = output_subsidence_dim
        self.output_gwl_dim = output_gwl_dim
        self._combined_output_target_dim = (
            output_subsidence_dim + output_gwl_dim
        )
        self.embed_dim = embed_dim # Target dim for MME
        self.hidden_units = hidden_units # General purpose GRN units
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.forecast_horizon = forecast_horizon
        self.max_window_size = max_window_size
        self.memory_size = memory_size
        self.final_agg = final_agg
        self.activation_fn_str = Activation(activation).activation_str
        self.use_residuals = use_residuals
        self.use_batch_norm = use_batch_norm
        self.pinn_coefficient_C_config = pinn_coefficient_C
        
        self.use_vsn = use_vsn
        # If vsn_units not specified, use hidden_units as a sensible default
        self.vsn_units = vsn_units if vsn_units is not None else self.hidden_units

        (self.quantiles, self.scales,
         self.lstm_return_sequences) = set_default_params(
            quantiles, scales, multi_scale_agg
        )
        self.multi_scale_agg_mode = multi_scale_agg

        self._build_halnet_layers() # This will now include VSNs
        self._build_pinn_components()
        
  
    def _build_halnet_layers(self):
        """
        Instantiates all layers for the core data-driven HALNet architecture,
        optionally including VariableSelectionNetworks.
        """
        # --- Variable Selection Networks (Applied first if use_vsn is True) ---
        if self.use_vsn:
            if self.static_input_dim > 0:
                self.static_vsn = VariableSelectionNetwork(
                    num_inputs=self.static_input_dim,
                    units=self.vsn_units, # Output dim for static features
                    dropout_rate=self.dropout_rate,
                    activation=self.activation_fn_str,
                    use_batch_norm=self.use_batch_norm, # VSN GRNs can use BN
                    name="static_vsn"
                )
                # GRN after static VSN (common in TFT to refine VSN output)
                self.static_vsn_grn = GatedResidualNetwork(
                    units=self.hidden_units, # Final static context dim
                    dropout_rate=self.dropout_rate,
                    activation=self.activation_fn_str,
                    use_batch_norm=self.use_batch_norm,
                    name="static_vsn_grn"
                )
            else:
                self.static_vsn = None
                self.static_vsn_grn = None

            if self.dynamic_input_dim > 0:
                self.dynamic_vsn = VariableSelectionNetwork(
                    num_inputs=self.dynamic_input_dim,
                    units=self.vsn_units, # Output dim per time step
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True,
                    activation=self.activation_fn_str,
                    use_batch_norm=self.use_batch_norm,
                    name="dynamic_vsn"
                )
                # GRN for dynamic VSN output (optional, but good for consistency)
                self.dynamic_vsn_grn = GatedResidualNetwork(
                    units=self.embed_dim, # Target dim for LSTM/Attention
                    dropout_rate=self.dropout_rate,
                    activation=self.activation_fn_str,
                    use_batch_norm=self.use_batch_norm,
                    name="dynamic_vsn_grn"
                )

            else: # Should not happen as dynamic_input_dim must be > 0
                self.dynamic_vsn = None
                self.dynamic_vsn_grn = None


            if self.future_input_dim > 0:
                self.future_vsn = VariableSelectionNetwork(
                    num_inputs=self.future_input_dim,
                    units=self.vsn_units,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True,
                    activation=self.activation_fn_str,
                    use_batch_norm=self.use_batch_norm,
                    name="future_vsn"
                )
                # GRN for future VSN output
                self.future_vsn_grn = GatedResidualNetwork(
                    units=self.embed_dim, # Target dim for MME/Attention
                    dropout_rate=self.dropout_rate,
                    activation=self.activation_fn_str,
                    use_batch_norm=self.use_batch_norm,
                    name="future_vsn_grn"
                )
            else:
                self.future_vsn = None
                self.future_vsn_grn = None
            
                    # If VSNs are handling the embedding, we might not need MME.
            # Its attribute can be set to None, and the `run_halnet_core` method
            # will use the VSN outputs directly.
            self.multi_modal_embedding = None
            
            logger.info(
                " VSN is enabled. MultiModalEmbedding"
                " for dynamic/future inputs will be bypassed."
                )
        else: # If not using VSNs
            self.static_vsn, self.static_vsn_grn = None, None
            self.dynamic_vsn, self.dynamic_vsn_grn = None, None
            self.future_vsn, self.future_vsn_grn = None, None
            # If not using VSN, initial processing for static features
            if self.static_input_dim > 0:
                 self.static_dense_initial = Dense( # Renamed to avoid clash
                    self.hidden_units, activation=self.activation_fn_str
                )
            else:
                self.static_dense_initial = None

        # --- Subsequent Layers (Inputs to these might change based on VSN usage) ---
        # MultiModalEmbedding now takes outputs of dynamic_vsn_grn and future_vsn_grn
        # Or, if VSNs are not used, it takes raw (or simply Densed) dynamic/future inputs.
        # The VSN outputs are already "embedded" to vsn_units or embed_dim.
        # So, MultiModalEmbedding might become redundant or need to adapt.
        # For TFT, VSN outputs (after GRN) directly feed into LSTM/attention.
        # Let's assume VSN outputs (after GRN) are the new "embedded" inputs.
        # So, we might not need self.multi_modal_embedding if VSNs are used
        # and their output GRNs project to self.embed_dim.
        if not self.use_vsn or (
                self.dynamic_vsn_grn is None and self.future_vsn_grn is None) :
            # If VSNs are not used for dynamic/future, or not fully configured,
            # keep MME for raw inputs.
            # If no VSN, we need MME for initial feature processing.
           self.multi_modal_embedding = MultiModalEmbedding(self.embed_dim)
           logger.info(
               "VSN is disabled. MultiModalEmbedding"
               " will be used for raw inputs.")
        else:
            # If VSNs are used and their GRNs output embed_dim, MME might be skipped
            # for dynamic/future as they are already processed.
            # Or MME could take these processed VSN outputs if they are of different dims.
            # For simplicity, let's assume if VSNs are used, their outputs (after GRN)
            # are what we use, and MME might be for other modalities if any.
            # If dynamic_vsn_grn and future_vsn_grn output embed_dim, we can
            # directly concatenate them if needed.
            self.multi_modal_embedding = None # Or adapt its usage

        self.positional_encoding = PositionalEncoding()
        self.multi_scale_lstm = MultiScaleLSTM(
            lstm_units=self.lstm_units,
            scales=self.scales,
            return_sequences=self.lstm_return_sequences
        )
        self.hierarchical_attention = HierarchicalAttention(
            units=self.attention_units, num_heads=self.num_heads
        )
        self.cross_attention = CrossAttention(
            units=self.attention_units, num_heads=self.num_heads
        )
        self.memory_augmented_attention = MemoryAugmentedAttention(
            units=self.attention_units, memory_size=self.memory_size,
            num_heads=self.num_heads
        )
        self.multi_resolution_attention_fusion = MultiResolutionAttentionFusion(
            units=self.attention_units, num_heads=self.num_heads
        )
        self.dynamic_time_window = DynamicTimeWindow(
            max_window_size=self.max_window_size
        )
        self.multi_decoder = MultiDecoder(
            output_dim=self._combined_output_target_dim,
            num_horizons=self.forecast_horizon
        )
        self.quantile_distribution_modeling = QuantileDistributionModeling(
            quantiles=self.quantiles,
            output_dim=self._combined_output_target_dim
        )
        
        # LearnedNormalization might be applied to VSN inputs or outputs,
        # or raw inputs if VSN not used.
        # Let's assume it's applied to raw inputs before VSN if VSN is used.
        # Or, it could be part of the VSN's internal GRN processing.
        # For now, keep it as a general layer.
        self.learned_normalization = LearnedNormalization()

        # Static processing if VSN is NOT used for static features
        if not self.use_vsn or self.static_vsn is None:
            self.static_dense = Dense( # This was self.static_dense_initial before
                self.hidden_units, activation=self.activation_fn_str
            )
            self.static_dropout = Dropout(self.dropout_rate)
            if self.use_batch_norm:
                self.static_batch_norm = LayerNormalization()
            else:
                self.static_batch_norm = None
            self.grn_static = GatedResidualNetwork( # This processes output of static_dense
                units=self.hidden_units, dropout_rate=self.dropout_rate,
                activation=self.activation_fn_str,
                use_batch_norm=self.use_batch_norm
            )
        else: # If static_vsn is used, these specific layers might not be needed
              # as static_vsn_grn produces the final static context
            self.static_dense = None
            self.static_dropout = None
            self.static_batch_norm = None
            self.grn_static = None # static_vsn_grn takes this role

        self.residual_dense = Dense(
            2 * self.embed_dim # This was tied to MME output usually
        ) if self.use_residuals else None

    
    def _build_pinn_components(self):
        """
        Instantiates components required for the physics-informed module.
        """
        # --- Learnable Physical Coefficient C ---
        if self.pinn_coefficient_C_config == 'learnable':
            # We learn log(C) and use exp(log(C)) to ensure C > 0
            self.log_C_coefficient = self.add_weight(
                name="log_pinn_coefficient_C",
                shape=(), # Scalar
                initializer=Constant(
                    tf_log(0.01) # Start with C=0.01
                ),
                trainable=True,
            )
            self._get_C = lambda: tf_exp(self.log_C_coefficient)

        elif isinstance(self.pinn_coefficient_C_config, (float, int)):
            # Use a fixed, non-trainable constant value
            self._get_C = lambda: tf_constant(
                self.pinn_coefficient_C_config, dtype=tf_float32
            )
        elif self.pinn_coefficient_C_config is None:
            # Physics is disabled, C is effectively 1 but will not be used
            # if lambda_pde is 0 in compile()
            self._get_C = lambda: tf_constant(1.0, dtype=tf_float32)
        else:
            raise ValueError(
                "pinn_coefficient_C must be 'learnable', a number, or None."
            )

    def get_pinn_coefficient_C(self) -> Tensor:
        """Returns the physical coefficient C."""
        return self._get_C()

    @tf_autograph.experimental.do_not_convert
    def call(
        self,
        inputs: Dict[str, Optional[Tensor]],
        training: bool = False
    ) -> Dict[str, Tensor]:
        """
        Forward pass for PIHALNet, computes predictions and PDE residual.
        
        This method orchestrates the data flow through the data-driven
        HALNet core and then computes the physics-based residual on the
        model's mean predictions.

        Args:
            inputs (Dict[str, tf.Tensor]): A dictionary of input tensors.
                It is processed by `process_pinn_inputs` and is expected
                to contain keys like 'coords', 'static_features', 
                'dynamic_features', and 'future_features'.
            training (bool): Python boolean indicating training mode.

        Returns:
            Dict[str, tf.Tensor]: A dictionary containing:
                - 'subs_pred': Subsidence predictions (potentially with quantiles).
                - 'gwl_pred': GWL predictions (potentially with quantiles).
                - 'pde_residual': The calculated physics residual tensor.
        """
        # --- 1. Process and Validate All Inputs ---
        # The `process_pinn_inputs` helper unpacks the input dict and
        # isolates the coordinate tensors for later use.
        logger.debug("PIHALNet call: Processing PINN inputs.")
        t, x, y, static_features, dynamic_features, future_features = \
            process_pinn_inputs(inputs, mode='as_dict')

        # `validate_model_inputs` can provide a secondary, more detailed
        # check on the unpacked feature tensors.
        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=[static_features, dynamic_features, future_features],
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            forecast_horizon=self.forecast_horizon,
            mode='strict',
            verbose=0 # Set to 1 for more detailed logging from validator
        )
        logger.debug(
            "Input shapes after validation:"
            f" S={getattr(static_p, 'shape', 'None')}, "
            f"D={getattr(dynamic_p, 'shape', 'None')},"
            f" F={getattr(future_p, 'shape', 'None')}"
        )

        # --- 2. Run Core Data-Driven Feature Extraction ---
        # This self-contained method performs the complex feature engineering
        # using LSTMs and attention mechanisms.
        logger.debug("Running HALNet core for feature extraction.")
        final_features_for_decode = self.run_halnet_core(
            static_input=static_p,
            dynamic_input=dynamic_p,
            future_input=future_p,
            training=training
        )
        logger.debug(
            f"Shape of features for decoder: {final_features_for_decode.shape}"
        )

        # --- 3. Generate Predictions ---
        # Get mean predictions (for PDE) from the multi-horizon decoder
        decoded_outputs = self.multi_decoder(
            final_features_for_decode, training=training
        )
        logger.debug(f"Shape of decoded outputs (means): {decoded_outputs.shape}")

        # Get final predictions (potentially with quantiles, for data loss)
        predictions_final_targets = decoded_outputs
        if self.quantiles is not None:
            predictions_final_targets = self.quantile_distribution_modeling(
                decoded_outputs, training=training
            )
        logger.debug(
            f"Shape of final quantile outputs: {predictions_final_targets.shape}"
        )

        # --- 4. Split and Organize Outputs ---
        # Use helper to separate subsidence and GWL predictions for both
        # data loss (quantiles) and physics loss (mean).
        (s_pred_final, gwl_pred_final, 
         s_pred_mean_for_pde, gwl_pred_mean_for_pde) = self.split_outputs(
             predictions_combined=predictions_final_targets,
             decoded_outputs_for_mean=decoded_outputs
         )

        # --- 5. Calculate Physics Residual ---
        # The PDE residual is calculated on the mean predictions using
        # finite differences, which is suitable for sequence outputs.
        # This does NOT require a GradientTape in the call method.
        logger.debug("Computing PDE residual from mean predictions.")
        if self.forecast_horizon > 1:
            pde_residual = compute_consolidation_residual(
                s_pred=s_pred_mean_for_pde,
                h_pred=gwl_pred_mean_for_pde,
                time_steps=t, # Assumes `t` holds the forecast time sequence
                C=self.get_pinn_coefficient_C()
            )
        else:
            # Cannot compute discrete time derivative with a single point
            logger.warning(
                "Forecast horizon is 1, cannot compute finite-difference "
                "based PDE residual. Returning zeros."
            )
            pde_residual = tf_zeros_like(s_pred_mean_for_pde)
        
        logger.debug(f"Shape of PDE residual: {pde_residual.shape}")
        
        # --- 6. Return All Components for Loss Calculation ---
        return {
            "subs_pred": s_pred_final,
            "gwl_pred": gwl_pred_final,
            "pde_residual": pde_residual,
        }
    
    def compile(
        self, 
        optimizer, 
        loss, 
        metrics=None, 
        loss_weights=None,
        lambda_pde=1.0, # Add PINN loss weight as a compile-time parameter
        **kwargs
    ):
        """
        Configures the model for training with a composite PINN loss.

        Args:
            optimizer: Keras optimizer instance.
            loss (Dict[str, any]): A dictionary mapping output names 
                ('subs_pred', 'gwl_pred') to Keras loss functions or
                their string identifiers (e.g., 'mse').
            metrics (Dict[str, any], optional): A dictionary mapping 
                output names to a list of metrics to track for each output.
            loss_weights (Dict[str, float], optional): A dictionary
                mapping output names to scalar coefficients to weight the
                data loss contributions. Defaults to 1.0 for each.
            lambda_pde (float, optional): The weight for the physics-based
                PDE residual loss. Defaults to 1.0.
            **kwargs: Additional arguments for `tf.keras.Model.compile`.
        """
        # Call the parent's compile method. It will handle the setup of
        # losses, metrics, and weights for our named outputs.
        super().compile(
            optimizer=optimizer, 
            loss=loss, 
            metrics=metrics, 
            loss_weights=loss_weights,
            **kwargs
        )
        # Store the PINN-specific loss weight
        self.lambda_pde = lambda_pde
        
    def train_step(self, data: Tuple[Dict, Dict]) -> Dict[str, Tensor]:
        """
        Custom training step to handle the composite PINN loss.
        """
        # Unpack the data. Keras provides it as a tuple of (inputs, targets).
        # We expect both to be dictionaries.
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError(
                "Expected data to be a tuple of (inputs_dict, targets_dict)."
            )
        inputs, targets = data

        # Open a GradientTape to record operations for automatic differentiation.
        with GradientTape() as tape:
            # 1. Forward pass to get model outputs
            # The `call` method returns a dict: {'subs_pred', 'gwl_pred', 'pde_residual'}
            outputs = self(inputs, training=True)

            # Structure predictions to match the 'loss' dictionary from compile()
            y_pred_for_loss = {
                'subs_pred': outputs['subs_pred'],
                'gwl_pred': outputs['gwl_pred']
            }

            # 2. Calculate Data Fidelity Loss (L_data)
            # Keras's self.compute_loss handles the dictionary of losses,
            # targets, predictions, and loss_weights automatically.
            data_loss = self.compute_loss(
                x=inputs, y=targets, y_pred=y_pred_for_loss
            )

            # 3. Calculate Physics Residual Loss (L_pde)
            # We penalize the mean of the squared residual to force it to zero.
            pde_residual = outputs['pde_residual']
            loss_pde = tf_reduce_mean(tf_square(pde_residual))

            # 4. Combine losses to form the Total Loss
            # This is the final scalar loss that will be differentiated.
            total_loss = data_loss + self.lambda_pde * loss_pde

        # 5. Compute and Apply Gradients
        # This includes the model's layers and the learnable `log_C_coefficient`.
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 6. Update and Return Metrics
        # Update the metrics passed in compile() (e.g., 'mae', 'rmse' for each output)
        self.compiled_metrics.update_state(targets, y_pred_for_loss)

        # Build a dictionary of results to be displayed by Keras.
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "total_loss": total_loss,      # The main loss we optimized
            "data_loss": data_loss,      # The part of the loss from data
            "physics_loss": loss_pde,    # The part of the loss from physics
        })
        return results


    def get_config(self):
        config = super().get_config()
        # Get all parameters from __init__ to serialize the model
        base_config = {
           'static_input_dim': self.static_input_dim,
           'dynamic_input_dim': self.dynamic_input_dim,
           'future_input_dim': self.future_input_dim,
           'output_subsidence_dim': self.output_subsidence_dim,
           'output_gwl_dim': self.output_gwl_dim,
           'embed_dim': self.embed_dim,
           'hidden_units': self.hidden_units,
           'lstm_units': self.lstm_units,
           'attention_units': self.attention_units,
           'num_heads': self.num_heads,
           'dropout_rate': self.dropout_rate,
           'forecast_horizon': self.forecast_horizon,
           'quantiles': self.quantiles,
           'max_window_size': self.max_window_size,
           'memory_size': self.memory_size,
           'scales': self.scales,
           'multi_scale_agg': self.multi_scale_agg_mode,
           'final_agg': self.final_agg,
           'activation': self.activation_fn_str,
           'use_residuals': self.use_residuals,
           'use_batch_norm': self.use_batch_norm,
           'pinn_coefficient_C': self.pinn_coefficient_C_config,
        }
        config.update(base_config)
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    def run_halnet_core(
        self,
        static_input: Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        training: bool
    ) -> Tensor:
        """
        Executes the core data-driven feature extraction pipeline.

        This method processes static, dynamic, and future inputs through
        VSNs (if enabled), LSTMs, and various attention mechanisms to
        produce a final set of features ready for decoding.

        Args:
            static_input (tf.Tensor): Processed static features.
            dynamic_input (tf.Tensor): Processed dynamic historical features.
            future_input (tf.Tensor): Processed known future features.
            training (bool): Python boolean indicating training mode.

        Returns:
            tf.Tensor: Features processed by the core, ready for the
                       MultiDecoder.
        """
        # --- 1. Initial Feature Processing (VSN or simple Dense) ---
        
        # A. Process Static Features
        static_features_grn_out = None
        if self.use_vsn and self.static_vsn is not None:
            # VSN path for static features
            static_context = self.static_vsn(static_input, training=training)
            static_features_grn_out = self.static_vsn_grn(
                static_context, training=training
            )
        elif self.static_input_dim > 0:
            # Non-VSN path for static features
            # self.static_dense_initial is created in _build_halnet_layers
            processed_static = self.static_dense_initial(static_input) 
            static_features_grn_out = self.grn_static(
                processed_static, training=training
            )
        
        # B. Process Dynamic and Future Time-Varying Features
        dynamic_processed = dynamic_input
        future_processed = future_input

        if self.use_vsn:
            if self.dynamic_vsn is not None:
                dynamic_processed = self.dynamic_vsn(
                    dynamic_input, training=training
                )
                dynamic_processed = self.dynamic_vsn_grn(
                    dynamic_processed, training=training
                )
            if self.future_vsn is not None:
                future_processed = self.future_vsn(
                    future_input, training=training
                )
                future_processed = self.future_vsn_grn(
                    future_processed, training=training
                )

        # --- 2. Temporal Feature Alignment & Embedding ---
        # Align future features to the same past time steps as dynamic features
        _, future_for_embedding = align_temporal_dimensions(
            tensor_ref=dynamic_processed,
            tensor_to_align=future_processed,
            mode='slice_to_ref',
            name="future_for_embedding"
        )
        
        # If VSNs are not used, we need MultiModalEmbedding.
        # If VSNs *are* used, they've already projected features to embed_dim,
        # so we just need to concatenate them.
        if self.multi_modal_embedding is not None:
            # Non-VSN path
            embeddings = self.multi_modal_embedding(
                [dynamic_processed, future_for_embedding], training=training
            )
        else:
            # VSN path: dynamic_processed & future_for_embedding are already at embed_dim
            embeddings = Concatenate(axis=-1)([
                dynamic_processed, future_for_embedding
            ])

        embeddings = self.positional_encoding(embeddings, training=training)
        
        if self.use_residuals and self.residual_dense is not None:
            embeddings = AddLayer()([embeddings, self.residual_dense(embeddings)])
            
        # --- 3. LSTM and Attention Mechanisms ---
        lstm_output_raw = self.multi_scale_lstm(
            dynamic_processed, training=training # Use VSN-processed dynamic feats
        )
        lstm_features = aggregate_multiscale(
            lstm_output_raw, mode=self.multi_scale_agg_mode
        )
        
        time_steps_dyn = tf_shape(dynamic_processed)[1]
        lstm_features_tiled = tf_tile(
            tf_expand_dims(lstm_features, axis=1), [1, time_steps_dyn, 1]
        )
        
        # Hierarchical Attention inputs need to be of compatible dimension.
        # Assuming they are already at embed_dim after VSN-GRN path.
        hierarchical_att = self.hierarchical_attention(
           [dynamic_processed, future_for_embedding], training=training
        )
        cross_attention_output = self.cross_attention(
            [dynamic_processed, embeddings], training=training
        )
        memory_attention_output = self.memory_augmented_attention(
            hierarchical_att, training=training
        )
        
        # Tile static context to match temporal dimension for combination
        static_features_expanded = None
        if static_features_grn_out is not None:
            static_features_expanded = tf_tile(
                tf_expand_dims(static_features_grn_out, axis=1), 
                [1, time_steps_dyn, 1]
            )
        
        # --- 4. Feature Fusion and Final Processing ---
        features_to_combine = [
            lstm_features_tiled,
            cross_attention_output,
            hierarchical_att,
            memory_attention_output,
        ]
        if static_features_expanded is not None:
            features_to_combine.insert(0, static_features_expanded)
            
        combined_features = Concatenate(axis=-1)(features_to_combine)
        
        attention_fusion_output = self.multi_resolution_attention_fusion(
            combined_features, training=training
        )
        time_window_output = self.dynamic_time_window(
            attention_fusion_output, training=training
        )
        final_features_for_decode = aggregate_time_window_output(
            time_window_output, self.final_agg
        )
        
        return final_features_for_decode

    def split_outputs(
        self, 
        predictions_combined: Tensor, 
        decoded_outputs_for_mean: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Splits combined model predictions into subsidence and GWL components.

        Also extracts mean predictions suitable for PDE residual calculation
        if quantiles are used.

        Args:
            predictions_combined (tf.Tensor): The final output tensor, 
                potentially with a quantile dimension.
            decoded_outputs_for_mean (tf.Tensor): The output from MultiDecoder,
                representing mean predictions before quantile distribution.
                Shape: (Batch, Horizon, CombinedTargetsDim).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing:
                - `s_pred_final`: Subsidence predictions for data loss.
                - `gwl_pred_final`: GWL predictions for data loss.
                - `s_pred_mean_for_pde`: Mean subsidence for physics loss.
                - `gwl_pred_mean_for_pde`: Mean GWL for physics loss.
        """
        # --- 1. Extract Mean Predictions (for PDE Loss) ---
        # These come from the decoder output *before* quantile distribution.
        # This provides a stable point forecast for derivative calculation.
        s_pred_mean_for_pde = decoded_outputs_for_mean[
            ..., :self.output_subsidence_dim
        ]
        gwl_pred_mean_for_pde = decoded_outputs_for_mean[
            ..., self.output_subsidence_dim:
        ]

        # --- 2. Extract Final Predictions (for Data Loss) ---
        # These may or may not include a quantile dimension.
        # We check the tensor's rank to decide how to slice.
        # Keras may return a known static rank during build time,
        # or we can use tf.rank for dynamic graph execution.
        if self.quantiles and hasattr(
                predictions_combined, 'shape') and len(
                    predictions_combined.shape) == 4:
            # Case: Quantiles are present.
            # Shape is (Batch, Horizon, NumQuantiles, CombinedOutputDim)
            s_pred_final = predictions_combined[
                ..., :self.output_subsidence_dim
            ]
            gwl_pred_final = predictions_combined[
                ..., self.output_subsidence_dim:
            ]
        elif ( 
                hasattr(predictions_combined, 'shape') 
                and len(predictions_combined.shape) == 3
            ):
            # Case: No quantiles. Shape is (Batch, Horizon, CombinedOutputDim)
            s_pred_final = predictions_combined[
                ..., :self.output_subsidence_dim
            ]
            gwl_pred_final = predictions_combined[
                ..., self.output_subsidence_dim:
            ]
        else:
            # This case handles dynamic shapes during graph execution
            # and acts as a fallback.
            if self.quantiles and tf_rank(predictions_combined) == 4:
                s_pred_final = predictions_combined[..., :self.output_subsidence_dim]
                gwl_pred_final = predictions_combined[..., self.output_subsidence_dim:]
                
            elif tf_rank(predictions_combined) == 3:
                 s_pred_final = predictions_combined[..., :self.output_subsidence_dim]
                 gwl_pred_final = predictions_combined[..., self.output_subsidence_dim:]
            else:
                # This case should ideally not be reached if QDM is consistent
                 raise ValueError(
                    f"Unexpected shape from QuantileDistributionModeling: "
                    f"Rank is {tf_rank(predictions_combined)}"
                )
            
        return (s_pred_final, gwl_pred_final,
                s_pred_mean_for_pde, gwl_pred_mean_for_pde)
    
    # def split_outputs(
    #     self, 
    #     predictions_combined: Tensor, 
    #     decoded_outputs_for_mean: Tensor
    #     ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    #     """
    #     Splits combined model predictions into subsidence and GWL components.
    
    #     Also extracts mean predictions suitable for PDE residual calculation
    #     if quantiles are used.
    
    #     Args:
    #         predictions_combined (tf.Tensor): The output from
    #             QuantileDistributionModeling. Shape can be:
    #             - (Batch, Horizon, CombinedTargetsDim) if no quantiles.
    #             - (Batch, Horizon, NumQuantiles, CombinedTargetsDim) if quantiles.
    #         decoded_outputs_for_mean (tf.Tensor): The output from MultiDecoder,
    #             representing mean predictions before quantile distribution.
    #             Shape: (Batch, Horizon, CombinedTargetsDim).
    
    #     Returns:
    #         Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    #             - subs_pred_quantiles_or_mean: Subsidence predictions.
    #             - gwl_pred_quantiles_or_mean: GWL predictions.
    #             - subs_pred_mean_for_pde: Mean subsidence for PDE.
    #             - gwl_pred_mean_for_pde: Mean GWL for PDE.
    #     """
    #     # --- Mean predictions for PDE (from MultiDecoder output) ---
    #     subs_pred_mean_for_pde = decoded_outputs_for_mean[
    #         ..., :self.output_subsidence_dim
    #     ]
    #     gwl_pred_mean_for_pde = decoded_outputs_for_mean[
    #         ..., self.output_subsidence_dim:
    #     ]
    
    #     # --- Final predictions for data loss (potentially with quantiles) ---
    #     if self.quantiles and len(tf_shape(predictions_combined)) == 4:
    #         # Quantiles are present, shape is (B, H, Q, Combined_O)
    #         subs_pred_quantiles_or_mean = predictions_combined[
    #             ..., :self.output_subsidence_dim
    #         ]
    #         gwl_pred_quantiles_or_mean = predictions_combined[
    #             ..., self.output_subsidence_dim:
    #         ]
    #     elif len(tf_shape(predictions_combined)) == 3:
    #         # No quantiles, or QDM already gave mean-like output
    #         # Shape is (B, H, Combined_O)
    #         subs_pred_quantiles_or_mean = predictions_combined[
    #             ..., :self.output_subsidence_dim
    #         ]
    #         gwl_pred_quantiles_or_mean = predictions_combined[
    #             ..., self.output_subsidence_dim:
    #         ]
    #     else:
    #         # This case should ideally not be reached if QDM is consistent
    #         raise ValueError(
    #             f"Unexpected shape from QuantileDistributionModeling: "
    #             f"{tf_shape(predictions_combined)}"
    #         )
            
    #     return (subs_pred_quantiles_or_mean, gwl_pred_quantiles_or_mean,
    #             subs_pred_mean_for_pde, gwl_pred_mean_for_pde)
    
    # def _run_halnet_core(
    #     self,
    #     static_input: Tensor,
    #     dynamic_input: Tensor,
    #     future_input: Tensor,
    #     training: bool
    # ) -> Tensor:
    #     """
    #     Executes the core data-driven feature extraction pipeline of HALNet.

    #     This method processes static, dynamic, and future inputs through
    #     embeddings, LSTMs, and various attention mechanisms to produce
    #     a final set of features ready for decoding into predictions.

    #     Args:
    #         static_input (tf.Tensor): Processed static features.
    #             Shape: (batch_size, static_feature_dim_processed) or
    #                    (batch_size, 0) if no static features.
    #         dynamic_input (tf.Tensor): Processed dynamic historical features.
    #             Shape: (batch_size, past_time_steps, dynamic_feature_dim)
    #         future_input (tf.Tensor): Processed known future features.
    #             Shape: (batch_size, future_time_span, future_feature_dim)
    #                    or (batch_size, future_time_span, 0) if no future feats.
    #         training (bool): Python boolean indicating training mode.

    #     Returns:
    #         tf.Tensor: Features processed by the HALNet core, ready for
    #                    the MultiDecoder. Shape: (batch_size, final_feature_dim)
    #     """
    #     # Static features processing
    #     # Initialize to a default that won't break concatenation if no static features
    #     batch_size_for_default = tf_shape(dynamic_input)[0]
    #     static_features_grn = tf_zeros( # KERAS_DEPS.zeros
    #         (batch_size_for_default, self.hidden_units), dtype=tf_float32
    #         ) 

    #     # Process static features only if they are actually present
    #     if self.static_input_dim > 0 and tf_shape(static_input)[-1] > 0:
    #         normalized_static = self.learned_normalization(
    #             static_input, training=training
    #         )
    #         processed_static_features = self.static_dense(normalized_static)
    #         if self.use_batch_norm and self.static_batch_norm is not None:
    #             processed_static_features = self.static_batch_norm(
    #                 processed_static_features, training=training
    #             )
    #         processed_static_features = self.static_dropout(
    #             processed_static_features, training=training
    #         )
    #         static_features_grn = self.grn_static(
    #             processed_static_features, training=training
    #         )

    #     # Align temporal inputs for MultiModalEmbedding
    #     # The 'year' and other coordinate features for PINN are part of these inputs
    #     _, future_input_for_embedding = align_temporal_dimensions(
    #         tensor_ref=dynamic_input,
    #         tensor_to_align=future_input,
    #         mode='slice_to_ref', # Or 'pad_to_ref' based on strategy
    #         name="future_input_for_mme"
    #     )
        
    #     # MultiModalEmbedding expects a list of tensors
    #     # Handle cases where future_input_for_embedding might have 0 features
    #     mme_inputs = [dynamic_input]
    #     if tf_shape(future_input_for_embedding)[-1] > 0:
    #         mme_inputs.append(future_input_for_embedding)
    #     elif len(mme_inputs) < len(self.multi_modal_embedding.dense_layers):
    #         # If future_input_for_embedding has 0 features but MME was built for 2 inputs,
    #         # we need to pass a correctly shaped zero tensor for the second modality.
    #         # This assumes MME was built based on initial non-zero future_input_dim.
    #          dummy_future_for_mme = tf_zeros_like(dynamic_input) # Match B, T
    #          dummy_future_for_mme = dummy_future_for_mme[
    #             ..., :0] # Make feature dim 0 if MME can handle it
    #                     # Or ensure MME is robust to this.
    #                     # A safer bet is to ensure MME is built with correct num inputs
    #                     # based on actual feature dimensions > 0.
    #          # For simplicity, if future_input_for_embedding has 0 features,
    #          # we might only pass dynamic_input if MME is built dynamically.
    #          # Assuming MME handles this by having been built with appropriate inputs.
    #          pass

    #     embeddings = self.multi_modal_embedding(
    #         mme_inputs, training=training
    #     )
    #     embeddings = self.positional_encoding(
    #         embeddings, training=training
    #     )
    #     if self.use_residuals and self.residual_dense is not None:
    #         embeddings = AddLayer()([
    #             embeddings, self.residual_dense(embeddings)
    #         ])
            
    #     lstm_output_raw = self.multi_scale_lstm(
    #         dynamic_input, training=training
    #     )
    #     lstm_features = aggregate_multiscale(
    #         lstm_output_raw, mode=self.multi_scale_agg_mode
    #     )
        
    #     time_steps_dyn = tf_shape(dynamic_input)[1]
    #     lstm_features_tiled = tf_tile(
    #         tf_expand_dims(lstm_features, axis=1), [1, time_steps_dyn, 1]
    #     )
        
    #     # For HierarchicalAttention, ensure inputs have compatible feature dimensions
    #     # or that HierarchicalAttention handles projection internally.
    #     # Assuming future_input_for_embedding is used.
    #     hierarchical_att_inputs = [dynamic_input]
    #     if tf_shape(future_input_for_embedding)[-1] > 0 :
    #         hierarchical_att_inputs.append(future_input_for_embedding)
    #     else: # If no future features, maybe repeat dynamic or use zeros
    #         hierarchical_att_inputs.append(tf_zeros_like(dynamic_input))


    #     hierarchical_att = self.hierarchical_attention(
    #        hierarchical_att_inputs , training=training
    #     )
    #     cross_attention_output = self.cross_attention(
    #         [dynamic_input, embeddings], training=training
    #     )
    #     memory_attention_output = self.memory_augmented_attention(
    #         hierarchical_att, training=training
    #     )
        
    #     static_features_expanded = tf_tile(
    #         tf_expand_dims(static_features_grn, axis=1), 
    #         [1, time_steps_dyn, 1]
    #     )
        
    #     combined_features = Concatenate(axis=-1)([
    #         static_features_expanded,
    #         lstm_features_tiled,
    #         cross_attention_output,
    #         hierarchical_att,
    #         memory_attention_output,
    #     ])
        
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