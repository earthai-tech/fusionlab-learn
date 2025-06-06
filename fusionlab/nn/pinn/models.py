# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Physics-Informed Hybrid Attentive LSTM Network (PIHALNet).
"""

from textwrap import dedent # noqa 
from numbers import Real, Integral  
from typing import List, Optional, Union, Dict, Tuple  

from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...params import LearnableC, FixedC, DisabledC 
from ...api.property import NNLearner 
from ...core.handlers import param_deprecated_message 
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
    tf_cond = KERAS_DEPS.cond
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
    from .._tensor_validation import check_inputs 
    

    from ..utils import set_default_params, squeeze_last_dim_if #noqa
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
    from .utils import process_pde_modes 
    
    
DEP_MSG = dependency_message('nn.pinn.models') 

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

__all__ =["PIHALNet"] 


@register_keras_serializable('fusionlab.nn.pinn', name="PIHALNet")
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'pde_mode', # The __init__ parameter name
            'condition': lambda p_value: (
                p_value == 'gw_flow' or
                p_value == 'both' or
                p_value == 'none' or
                (isinstance(p_value, list) and
                 any(mode in ['gw_flow', 'both', 'none'] for mode in p_value) and
                 not ('consolidation' in p_value and len(p_value) == 1) 
                )
            ),
            'message': (
                "Warning: The 'pde_mode' parameter received a value that "
                "includes options ('gw_flow', 'both', 'none') which are "
                "intended for future development or are not the primary "
                "focus of the current physics-informed implementation. "
                "This version of PIHALNet is optimized for and defaults to "
                "'consolidation' mode to ensure robust physical constraints "
                "based on Terzaghi's theory with finite differences. "
                "The model will proceed using 'consolidation' mode. Full "
                "support for other PDE modes and their specific derivative "
                "requirements will be explored in future releases."
            ),
            'default': 'consolidation', 
        }
    ],
    warning_category=UserWarning 
)

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
        "static_input_dim": [Interval(Integral, 0, None, closed='left')],
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "future_input_dim": [Interval(Integral, 0, None, closed='left')],
        "output_subsidence_dim": [Interval(Integral, 1, None, closed='left')],
        "output_gwl_dim": [Interval(Integral, 1, None, closed='left')],
        
        "embed_dim": [Interval(Integral, 1, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": [Interval(Integral, 1, None, closed='left'), None],
        "attention_units": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')], 
        "quantiles": ['array-like', StrOptions({'auto'}), None],
        "max_window_size": [Interval(Integral, 1, None, closed='left')],
        "memory_size": [Interval(Integral, 1, None, closed='left')], 
        "scales": ['array-like', StrOptions({'auto'}), None],
        "multi_scale_agg": [StrOptions({
            "last", "average", "flatten", "auto", "sum", "concat"}), None],
        "final_agg": [StrOptions({"last", "average", "flatten"})],

        "activation": [str, callable],
        "use_residuals": [bool],
        "use_batch_norm": [bool],
        
        "pde_mode": [
            StrOptions({'consolidation', 'gw_flow', 'both', 'none'}), 
            'array-like', None 
        ],
        "pinn_coefficient_C": [
            str, Real,None, StrOptions({"learnable"}) 
        ], 
        "gw_flow_coeffs": [dict, type(None)], 
        'use_vsn': [bool, int], 
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
        embed_dim: int = 32,
        hidden_units: int = 64,
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
        pde_mode: Union[str, List[str], None] = 'consolidation',
        pinn_coefficient_C: Union[
            LearnableC, FixedC, DisabledC, str, float, None
        ] = LearnableC(initial_value=0.01),
        gw_flow_coeffs: Optional[Dict[str, Union[str, float, None]]] = None,
        use_vsn: bool = True,
        vsn_units: Optional[int] = None,
        name: str = "PIHALNet",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.output_subsidence_dim = output_subsidence_dim
        self.output_gwl_dim = output_gwl_dim
        self._combined_output_target_dim = (
            output_subsidence_dim + output_gwl_dim
        )
        self.embed_dim = embed_dim 
        self.hidden_units = hidden_units 
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
        

        self.use_vsn = use_vsn
        self.vsn_units = vsn_units if vsn_units is not None else self.hidden_units

        (self.quantiles, self.scales,
         self.lstm_return_sequences) = set_default_params(
            quantiles, scales, multi_scale_agg
        )
        self.multi_scale_agg_mode = multi_scale_agg

        # --- Store PINN Configuration ---
        self.pde_modes_active = process_pde_modes(
            pde_mode, enforce_consolidation=True,
            solo_return=True,
        )

        # Normalize pinn_coefficient_C into one of our helper classes or legacy:
        self.pinn_coefficient_C_config = self._normalize_C_descriptor(
            pinn_coefficient_C)

        self.gw_flow_coeffs_config = gw_flow_coeffs if gw_flow_coeffs is not None else {}

        self._build_halnet_layers()
        self._build_pinn_components()

    def _normalize_C_descriptor(
        self,
        raw: Union[LearnableC, FixedC, DisabledC, str, float, None]
    ):
        """
        Internal helper: turn the user‐passed `raw` into exactly one of
        our three classes (LearnableC, FixedC, DisabledC).

        Raises
        ------
        ValueError
            If `raw` is an unrecognized string or negative float.
        TypeError
            If `raw` is not one of the expected types.
        """
        # 1) Already one of our classes?
        if isinstance(raw, (LearnableC, FixedC, DisabledC)):
            return raw

        # 2) If user passed a bare float or int: treat as FixedC(value=raw)
        if isinstance(raw, (float, int)):
            if raw < 0:
                raise ValueError(
                    "Numeric pinn_coefficient_C must"
                    f" be non‐negative, got {raw}"
                )
            # Nonzero means 'fixed to that value'
            return FixedC(value=float(raw))

        # 3) If user passed a string, allow the legacy
        # values "learnable" or "fixed"
        if isinstance(raw, str):
            low = raw.strip().lower()
            if low == "learnable":
                # Default initial value = 0.01 is built into LearnableC
                return LearnableC(initial_value=0.01)
            if low == "fixed":
                # Default fixed value = 1.0
                return FixedC(value=1.0)
            if low in ("none", "disabled", "off"):
                return DisabledC()

            raise ValueError(
                f"Unrecognized pinn_coefficient_C string: '{raw}'. "
                "Expected 'learnable', 'fixed', 'none', or use"
                " a LearnableC/FixedC/DisabledC instance."
            )

        # 4) If user passed None, treat as DisabledC()
        if raw is None:
            return DisabledC()

        raise TypeError(
            f"pinn_coefficient_C must be LearnableC, FixedC, DisabledC, "
            f"str, float, or None; got {type(raw).__name__}."
        )

    def _build_pinn_components(self):
        """
        Instantiates components required for the physics‐informed module.
        Specifically, sets up how we obtain C:
          - If LearnableC, create a trainable variable log_C.
          - If FixedC, store a lambda that returns a fixed tf.constant(value).
          - If DisabledC, store a lambda that returns tf.constant(1.0).
        """
        # Check which descriptor we have:
        desc = self.pinn_coefficient_C_config

        if isinstance(desc, LearnableC):
            # We learn log(C) so that C = exp(log_C) is always > 0
            self.log_C_coefficient = self.add_weight(
                name="log_pinn_coefficient_C",
                shape=(),  # scalar
                initializer=Constant(tf_log(desc.initial_value)),
                trainable=True
            )
            self._get_C = lambda: tf_exp(self.log_C_coefficient)

        elif isinstance(desc, FixedC):
            # Fixed value, non‐trainable
            val = desc.value
            self._get_C = lambda: tf_constant(val, dtype=tf_float32)

        elif isinstance(desc, DisabledC):
            # Physics disabled => C internally 1.0 but not used if lambda_pde==0
            # in compile()
            self._get_C = lambda: tf_constant(1.0, dtype=tf_float32)

        else:
            # Should never happen if _normalize_C_descriptor is correct
            raise RuntimeError(
                "Internal error: pinn_coefficient_C_config is not a recognized type."
            )

    def get_pinn_coefficient_C(self) -> Tensor:
        """Returns the physical coefficient C."""
        return self._get_C()
        
    def _build_halnet_layers(self):
        """
        Instantiates all layers for the core data-driven HALNet architecture,
        optionally including VariableSelectionNetworks.
        """
        if self.use_vsn:
            if self.static_input_dim > 0:
                self.static_vsn = VariableSelectionNetwork(
                    num_inputs=self.static_input_dim,
                    units=self.vsn_units, 
                    dropout_rate=self.dropout_rate,
                    activation=self.activation_fn_str,
                    use_batch_norm=self.use_batch_norm, 
                    name="static_vsn"
                )
                # GRN after static VSN 
                # (common in TFT to refine VSN output)
                self.static_vsn_grn = GatedResidualNetwork(
                    units=self.hidden_units, 
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
                    units=self.vsn_units,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True,
                    activation=self.activation_fn_str,
                    use_batch_norm=self.use_batch_norm,
                    name="dynamic_vsn"
                )
                # GRN for dynamic VSN output
                # (optional, but good for consistency)
                self.dynamic_vsn_grn = GatedResidualNetwork(
                    units=self.embed_dim, 
                    dropout_rate=self.dropout_rate,
                    activation=self.activation_fn_str,
                    use_batch_norm=self.use_batch_norm,
                    name="dynamic_vsn_grn"
                )

            else: 
                # Should not happen as dynamic_input_dim must be > 0
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
        # We have option to apply to raw inputs before VSN if VSN is used.
        # Or, it could be part of the VSN's internal GRN processing.
        # For now, we keep it as a general layer.
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
            self.grn_static = GatedResidualNetwork( 
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
        
        # basic tensors checks. 
        check_inputs(
            dynamic_inputs= dynamic_features, 
            static_inputs= static_features, 
            future_inputs= future_features, 
            dynamic_input_dim= self.dynamic_input_dim,
            static_input_dim = self.static_input_dim, 
            future_input_dim= self.future_input_dim,
            forecast_horizon= self.forecast_horizon 
        )
        # `validate_model_inputs` can provide a secondary, more detailed
        # check on the unpacked feature tensors.
        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=[static_features, dynamic_features, future_features],
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            # forecast_horizon=self.forecast_horizon,
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
   
        # XXX TODO: Implement GW flow
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
        lambda_pde=1.0, 
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
            # "loss": total_loss, # nust keep it for API consistency 
            "total_loss": total_loss,    # The main loss we optimized
            "data_loss": data_loss,      # The part of the loss from data
            "physics_loss": loss_pde,    # The part of the loss from physics
        })
        return results


    def get_config(self):
        config = super().get_config()
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
           'use_vsn': self.use_vsn, 
           'vsn_units': self.vsn_units,
           'name': self.name, 
           'pde_mode': self.pde_modes_active, 
           # 'pde_mode': self.pde_modes_active if len(
           #     self.pde_modes_active) > 1 else self.pde_modes_active[0], 
           'gw_flow_coeffs': self.gw_flow_coeffs_config,
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
            mode='pad_to_ref',
            name="future_for_embedding"
        )
        
        # VSN path logic
        if self.multi_modal_embedding is None:
            # Use tf.cond to handle the condition check for symbolic tensors
            future_input_is_valid = tf_shape(future_for_embedding)[-1] > 0
            # Create a tensor to append based on whether future_for_embedding is valid
            # We must ensure the concatenated tensor has a consistent shape.
            inputs_to_concat = [dynamic_processed]
            inputs_to_concat.append(
                tf_cond(
                    future_input_is_valid,
                    lambda: future_for_embedding,  # If valid, use future_for_embedding
                    # If future features are absent, append zeros to keep shape consistent
                    # for the residual connection later.
                    lambda: tf_zeros_like(future_for_embedding)  # Otherwise, append zeros
                )
            )
 
            embeddings = Concatenate(axis=-1)(inputs_to_concat)
        else: 
            # Non-VSN path
            # Similar logic for MultiModalEmbedding if it accepts zero-dim tensors
            # or we ensure its inputs are consistent.
            # Assuming MME path works as intended for now.
            embeddings = self.multi_modal_embedding(
                [dynamic_processed, future_for_embedding], training=training
            )
    
        embeddings = self.positional_encoding(embeddings, training=training)
    
        if self.use_residuals and self.residual_dense is not None:
            # Now, `embeddings` will consistently have shape (..., 2 * embed_dim)
            # because we concatenate with zeros if future features are absent.
            # The AddLayer() should no longer fail.
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
    
 
PIHALNet.__doc__+=r"""\n
The architecture can operate in two modes for its physical coefficient “C”:
1. **Parameter Specification:** Use a user-supplied constant value.
2. **Parameter Discovery:** Treat the coefficient as trainable (default),
   learning log(C) during training to ensure positivity.

PIHALNet’s total loss is a weighted sum of the data fidelity loss on
subsidence/GWL predictions and a physics residual loss (PDE loss).

Formulation
~~~~~~~~~~~~
Given inputs :math:`\mathbf{x}_{\text{static}}`, :math:`\mathbf{x}_{\text{dyn}}`,
and (optionally) :math:`\mathbf{x}_{\text{fut}}`, PIHALNet produces multi-horizon
predictions::
    
    :math:`\hat{s}[t+h],\; \hat{h}[t+h] \quad (h=1,\dots,H),`
    
for subsidence :math:`s` and GWL :math:`h`. The data loss
(:math:`L_{\text{data}}`) is::

    .. math::
        
       L_{\text{data}} = \sum_{h=1}^H 
       \bigl\{ \ell\bigl( \hat{s}[t+h], s[t+h]\bigr)
       + \ell\bigl( \hat{h}[t+h], h[t+h]\bigr) \bigr\},

where :math:`\ell` is typically MSE or MAE. The PDE residual loss
(:math:`L_{\text{pde}}`) for Terzaghi’s consolidation equation is::

    .. math::

       L_{\text{pde}} = 
       \frac{1}{H} \sum_{h=1}^H \bigl\| C \, \Delta s[t+h] 
       - \frac{\partial h}{\partial t}[t+h] \bigr\|^2,

computed via finite differences on the sequence of mean predictions.
The total loss is::

    .. math::

       L_{\text{total}} = L_{\text{data}} 
       + \lambda_{\text{pde}} \, L_{\text{pde}},

where :math:`\lambda_{\text{pde}}` is a user-defined weight.

Parameters
----------
static_input_dim : int
    Dimensionality of static (time-invariant) feature vector. Must be
    :math:`\geq 0`. If zero, static inputs are omitted.
dynamic_input_dim : int
    Dimensionality of the historical dynamic feature vector at each time
    step. Must be :math:`\geq 1`.
future_input_dim : int
    Dimensionality of known future covariates at each forecast step. Must
    be :math:`\geq 0`. If zero, no future covariates are used.
output_subsidence_dim : int, default 1
    Number of simultaneous subsidence targets (usually 1). Must be
    :math:`\geq 1`.
output_gwl_dim : int, default 1
    Number of simultaneous groundwater-level targets (usually 1). Must be
    :math:`\geq 1`.
embed_dim : int, default 32
    Size of the embedding after initial feature processing (VSN/GRN).  
    Controls hidden dimension for attention and LSTM inputs.
hidden_units : int, default 64
    Number of hidden units in the Gated Residual Networks (GRNs) and
    Dense layers. Must be :math:`\geq 1`.
lstm_units : int, default 64
    Number of units in each LSTM cell. Must be :math:`\geq 1`.  
    For multi-scale LSTM, this is the base size at each scale.
attention_units : int, default 32
    Number of units in multi-head attention and Hierarchical Attention
    layers. Must be :math:`\geq 1`.
num_heads : int, default 4
    Number of attention heads in all multi-head attention modules.
    Must be :math:`\geq 1`.
dropout_rate : float, default 0.1
    Dropout rate applied in various layers (VSN GRNs, attention heads,
    final MLP). Must be in :math:`[0.0, 1.0]`.
forecast_horizon : int, default 1
    Number of future time steps to predict. Must be :math:`\geq 1`.  
    Multi-horizon predictions are produced for 
    :math:`h=1,\dots,\text{forecast_horizon}`.
quantiles : list of float, optional
    If provided, PIHALNet will output quantile forecasts at each horizon.
    Each quantile dimension produces an additional branch. If *None*, only
    mean predictions are used for PDE residual (physics) loss.
max_window_size : int, default 10
    Maximum time-window length for DynamicTimeWindow layer. Must be
    :math:`\geq 1`. Controls the longest subsequence of historical dynamic
    features used at each decoding step.
memory_size : int, default 100
    Size of the memory bank for MemoryAugmentedAttention. Must be
    :math:`\geq 1`.
scales : list of int or “auto”, optional
    Scales used in MultiScaleLSTM. If “auto”, scales are chosen
    automatically based on forecast_horizon. Otherwise, each scale must
    divide the forecast_horizon. Example: :math:`[1, 3, 6]` for a 6-step
    horizon.
multi_scale_agg : {“last”, “average”, “flatten”, “auto”}, default “last”
    Aggregation method for multi-scale outputs:
    - “last”: take last time-step output from each scale.
    - “average”: average outputs over time.
    - “flatten”: concatenate outputs over time.
    - “auto”: choose “last” by default.
final_agg : {“last”, “average”, “flatten”}, default “last”
    Aggregation method after DynamicTimeWindow:
    - “last”: use final time-step.
    - “average”: average over windows.
    - “flatten”: flatten all window outputs.
activation : str or callable, default “relu”
    Activation function for all GRNs, Dense layers, and VSNs.  
    If a string, must be one of Keras built-ins (e.g. “relu”, “gelu”).
use_residuals : bool, default True
    If True, apply a residual connection via a Dense layer to embeddings.
use_batch_norm : bool, default False
    If True, apply LayerNormalization after each Dense/GRN block.
pde_mode : str or list of str or None, default “consolidation”
    Determines which PDE component(s) to include in the physics loss:
    - “consolidation”: solve Terzaghi’s consolidation only (1-D vertical).
    - “gw_flow”: solve coupled groundwater flow (reserved for future release).
    - “both”: include both consolidation and gw_flow (reserved).
    - “none”: disable physics loss entirely.
    If a list is provided, only those modes are active.  
    “consolidation” is enforced by default for this release.
pinn_coefficient_C : str, float, or None, default “learnable”
    Configuration for consolidation coefficient :math:`C`:
    - “learnable”: estimate :math:`\log(C)` as a trainable scalar.
    - float (:math:`>0`): use this fixed constant.
    - None: disable physics entirely (:math:`C=1` but unused).
gw_flow_coeffs : dict or None, default None
    Dictionary of groundwater-flow coefficients:
    - “K”: hydraulic conductivity (:math:`>0`), default “learnable”.
    - “Ss”: specific storage (:math:`>0`), default “learnable”.
    - “Q”: source/sink term, default 0.0.
    Only used if “gw_flow” is in pde_mode.
use_vsn : bool, default True
    If True, apply VariableSelectionNetwork blocks to static, dynamic,
    and future inputs. If False, skip VSN and use simple dense projections.
vsn_units : int or None, default None
    Output dimension of each VSN block. If None, defaults to hidden_units.
name : str, default “PIHALNet”
    Keras model name.
**kwargs
    Additional keyword arguments forwarded to tf.keras.Model initializer.

Attributes
----------
static_vsn : VariableSelectionNetwork or None
    VSN block for static features (if use_vsn=True and static_input_dim>0).
    Otherwise None.
dynamic_vsn : VariableSelectionNetwork or None
    VSN block for dynamic features (if use_vsn=True).
future_vsn : VariableSelectionNetwork or None
    VSN block for future features (if use_vsn=True and future_input_dim>0).
multi_modal_embedding : MultiModalEmbedding or None
    Fallback embedding layer if VSNs are disabled or incomplete.
multi_scale_lstm : MultiScaleLSTM
    LSTM module that operates at multiple temporal scales.
hierarchical_attention : HierarchicalAttention
    Performs hierarchical attention over dynamic/future features.
cross_attention : CrossAttention
    Cross-attends dynamic features with fused embeddings.
memory_augmented_attention : MemoryAugmentedAttention
    Attention mechanism with an external memory bank.
dynamic_time_window : DynamicTimeWindow
    Splits attention-fused features into overlapping windows.
multi_decoder : MultiDecoder
    Produces final multi-horizon mean forecasts for combined targets.
quantile_distribution_modeling : QuantileDistributionModeling or None
    If quantiles is not None, applies quantile modeling over decoder outputs.
positional_encoding : PositionalEncoding
    Adds positional embeddings to fused features.
learned_normalization : LearnedNormalization
    Optional normalization layer applied to raw inputs or VSN outputs.
log_C_coefficient : tf.Variable or None
    If pinn_coefficient_C == “learnable”, stores :math:`\log(C)`. Otherwise None.
log_K_gwflow_var : tf.Variable or None
    Trainable log(K) for groundwater flow, if enabled.
log_Ss_gwflow_var : tf.Variable or None
    Trainable log(Ss) for groundwater flow, if enabled.
Q_gwflow_var : tf.Variable or None
    Trainable or fixed Q term for groundwater flow, if enabled.

Methods
-------
call(inputs, training=False) -> dict[str, tf.Tensor]
    Forward pass computing predictions and PDE residual:
    1. Process inputs via process_pinn_inputs.
    2. Validate tensor shapes (via validate_model_inputs).
    3. Extract features with build_halnet_layers and attention/LSTM.
    4. Decode multi-horizon mean predictions via multi_decoder.
    5. If quantiles is provided, produce quantile outputs.
    6. Split outputs into subs_pred, gwl_pred, plus mean for PDE.
    7. Compute PDE residual via compute_consolidation_residual.
    Returns a dict containing:
      - “subs_pred”: subsidence forecasts (with quantiles if requested).
      - “gwl_pred”: GWL forecasts (with quantiles if requested).
      - “pde_residual”: tensor of physics residuals.

compile(optimizer, loss, metrics=None, loss_weights=None, lambda_pde=1.0, **kwargs)
    Configures the model for training with a composite PINN loss.
    Args:
      optimizer : Keras optimizer instance (e.g. Adam).
      loss : dict mapping output names (“subs_pred”, “gwl_pred”) to Keras loss
             functions or string identifiers (e.g. “mse”).
      metrics : dict mapping output names to lists of metrics to track.
      loss_weights : dict mapping output names to scalar weights for data loss.
      lambda_pde : float, weight for physics residual loss. Defaults to 1.0.
      **kwargs : Additional args for tf.keras.Model.compile.

train_step(data: tuple) -> dict[str, tf.Tensor]
    Custom training step to handle the composite PINN loss.
    1. Unpack (inputs_dict, targets_dict) from data.
    2. Forward pass with self(inputs, training=True).
    3. Extract subs_pred, gwl_pred for data loss.
    4. Compute data loss via self.compute_loss(x, y, y_pred).
    5. Compute physics loss: :math:`\mathrm{MSE}(\text{pde_residual})`.
    6. Total loss = data_loss + lambda_pde * physics_loss.
    7. Compute/apply gradients via optimizer.
    8. Update compiled metrics.
    Returns a dict of results:
      - “loss” (total PINN loss), “data_loss”, “physics_loss”,
        plus any compiled metrics (e.g. “subs_mae”, “gwl_mae”).

get_pinn_coefficient_C() -> tf.Tensor or None
    Returns the positive consolidation coefficient :math:`C`:
    - If “learnable”, returns :math:`\exp(\log_C_coefficient)`.
    - If fixed float was provided, returns that constant tensor.
    - If disabled, returns :math:`1.0` if only consolidation is active, else None.

get_K_gwflow(), get_Ss_gwflow(), get_Q_gwflow() -> tf.Tensor or None
    Return positive hydraulic conductivity :math:`K`, specific storage
    :math:`S_s`, and source/sink term :math:`Q` for groundwater flow PDE,
    if “gw_flow” mode is active. If not, return None.

get_config() -> dict
    Returns a dict of all initialization arguments (static, dynamic, future
    dims; PINN coefficients; architectural HPs). Enables model saving/loading
    via tf.keras.models.clone_model.

from_config(config: dict, custom_objects=None) -> PIHALNet
    Reconstructs a PIHALNet instance from get_config() output.

run_halnet_core(static_input, dynamic_input, future_input, training) -> tf.Tensor
    Executes the core data-driven feature pipeline:
    - Applies VSNs (if enabled) or Dense to each input block.
    - Aligns future features via align_temporal_dimensions.
    - Concatenates dynamic + future embeddings (or uses MME if no VSN).
    - Applies positional encoding and optional residual connection.
    - Runs MultiScaleLSTM, hierarchical/cross/memory-attention, and fusion.
    - Returns final features to feed into MultiDecoder.

split_outputs(predictions_combined, decoded_outputs_for_mean) -> tuple
    Separates combined predictions into subsidence and GWL components:
    - predictions_combined: may include a quantile dimension (Rank 4) or be
      Rank 3 if only mean forecasts. Splits along last axis into subsidence
      and GWL for data loss.
    - decoded_outputs_for_mean: always Rank 3 (Batch, Horizon, CombinedDim)
      before quantile modeling. Splits into s_pred_mean_for_pde and
      gwl_pred_mean_for_pde for physics residual calculation.

Notes
-----
- If quantiles is provided, final outputs shape is 
  (batch_size, horizon, num_quantiles, combined_output_dim).  
  Otherwise shape is (batch_size, horizon, combined_output_dim).
- Consolidation residual uses finite differences along horizon steps
  to approximate :math:`\partial h / \partial t` and :math:`\Delta s`.
- Groundwater-flow PDE is reserved for a future release (“gw_flow” mode).
- VariableSelectionNetworks (VSNs) refine feature selection. If
  use_vsn=False, simple Dense projections are used instead.
- MultiScaleLSTM can process temporal patterns at different resolutions.
  Scales must divide the forecast horizon if not “auto”.

Examples
--------
# 1) Instantiate PIHALNet for point forecasts (no quantiles):
>>> from fusionlab.nn.pinn.models import PIHALNet
>>> model = PIHALNet(
...     static_input_dim=5,
...     dynamic_input_dim=3,
...     future_input_dim=2,
...     output_subsidence_dim=1,
...     output_gwl_dim=1,
...     embed_dim=32,
...     hidden_units=64,
...     lstm_units=64,
...     attention_units=32,
...     num_heads=4,
...     dropout_rate=0.1,
...     forecast_horizon=1,
...     quantiles=None,
...     max_window_size=10,
...     memory_size=100,
...     scales='auto',
...     multi_scale_agg='last',
...     final_agg='last',
...     activation='relu',
...     use_residuals=True,
...     use_batch_norm=False,
...     pde_mode='consolidation',
...     pinn_coefficient_C='learnable',
...     gw_flow_coeffs=None,
...     use_vsn=True,
...     vsn_units=None,
... )
>>> model.compile(
...     optimizer='adam',
...     loss={'subs_pred': 'mse', 'gwl_pred': 'mse'},
...     metrics={'subs_pred': ['mae'], 'gwl_pred': ['mae']},
...     loss_weights={'subs_pred': 1.0, 'gwl_pred': 0.8},
...     lambda_pde=0.5,
... )
>>> import numpy as np
>>> inputs = {
...     'coords': np.zeros((2, 2), dtype='float32'),
...     'static_features': np.zeros((2, 5), dtype='float32'),
...     'dynamic_features': np.zeros((2, 1, 3), dtype='float32'),
...     'future_features': np.zeros((2, 1, 2), dtype='float32'),
... }
>>> targets = {
...     'subs_pred': np.zeros((2, 1, 1), dtype='float32'),
...     'gwl_pred': np.zeros((2, 1, 1), dtype='float32'),
... }
>>> outputs = model(inputs, training=False)
>>> print(outputs['subs_pred'].shape, outputs['gwl_pred'].shape)
(2, 1, 1) (2, 1, 1)

# 2) Instantiate with quantile forecasting (e.g., [0.1, 0.5, 0.9]):
>>> model_q = PIHALNet(
...     static_input_dim=5,
...     dynamic_input_dim=3,
...     future_input_dim=2,
...     output_subsidence_dim=1,
...     output_gwl_dim=1,
...     embed_dim=32,
...     hidden_units=64,
...     lstm_units=64,
...     attention_units=32,
...     num_heads=4,
...     dropout_rate=0.1,
...     forecast_horizon=3,
...     quantiles=[0.1, 0.5, 0.9],
...     max_window_size=10,
...     memory_size=100,
...     scales=[1, 3],
...     multi_scale_agg='average',
...     final_agg='flatten',
...     activation='gelu',
...     use_residuals=True,
...     use_batch_norm=True,
...     pde_mode='consolidation',
...     pinn_coefficient_C=0.02,  # fixed C = 0.02
...     gw_flow_coeffs={'K': 1e-4, 'Ss': 1e-5, 'Q': 0.0},
...     use_vsn=False,
...     vsn_units=None,
... )
>>> model_q.compile(
...     optimizer='adam',
...     loss={'subs_pred': 'mse', 'gwl_pred': 'mse'},
...     metrics={'subs_pred': ['mae'], 'gwl_pred': ['mae']},
...     loss_weights={'subs_pred': 1.0, 'gwl_pred': 0.8},
...     lambda_pde=1.0,
... )
>>> outputs_q = model_q(inputs, training=False)
>>> # Since quantiles=3, final outputs have shape (2, 3, 3, 2):
>>> print(outputs_q['subs_pred'].shape, outputs_q['gwl_pred'].shape)
(2, 3, 3, 1) (2, 3, 3, 1)

See Also
--------
fusionlab.nn.pinn.tuning.PIHALTuner
    Hyperparameter tuner specifically built for PIHALNet.
fusionlab.nn.pinn.utils.process_pinn_inputs
    Preprocessing of nested input dict to tensors.
fusionlab.nn.pinn._tensor_validation.validate_model_inputs
    Ensures dynamic/static/future tensors match declared dims.

References
----------
.. [1] Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
       *Physics-informed neural networks: A deep learning framework
       for solving forward and inverse problems involving nonlinear
       partial differential equations*. Journal of Computational Physics,
       378, 686–707.

.. [2] Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P.,
       Wang, S., & Yang, L. (2021). *Physics-informed machine learning*.
       Nature Reviews Physics, 3(6), 422–440.

.. [3] Heng, M.H., Chen, W., & Smith, E.C. (2022). *Joint modeling of
       land subsidence and groundwater levels with PINNs*. Environmental
       Modelling & Software, 150, 105347.
"""
