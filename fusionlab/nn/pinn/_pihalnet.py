# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Legacy Physics-Informed Hybrid Attentive LSTM Network (PiHALNet).
"""

from textwrap import dedent # noqa 
from numbers import Real, Integral  
from typing import List, Optional, Union, Dict, Tuple  
import warnings 

from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.property import NNLearner 
from ...core.handlers import param_deprecated_message 
from ...compat.sklearn import validate_params, Interval, StrOptions 
from ...params import LearnableC, FixedC, DisabledC 
from ...utils.deps_utils import ensure_pkg
from ...utils.generic_utils import select_mode 

from .. import KERAS_DEPS, KERAS_BACKEND, dependency_message
 
if KERAS_BACKEND: 
    LSTM = KERAS_DEPS.LSTM
    Dense = KERAS_DEPS.Dense
    LayerNormalization = KERAS_DEPS.LayerNormalization 
    Model= KERAS_DEPS.Model 
    Tensor=KERAS_DEPS.Tensor
    Variable =KERAS_DEPS.Variable 
    Add =KERAS_DEPS.Add
    Constant =KERAS_DEPS.Constant 
    GradientTape =KERAS_DEPS.GradientTape 
    
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    
    tf_zeros_like= KERAS_DEPS.zeros_like
    tf_zeros =KERAS_DEPS.zeros
    tf_reduce_mean =KERAS_DEPS.reduce_mean
    tf_square =KERAS_DEPS.square
    tf_constant =KERAS_DEPS.constant 
    tf_log = KERAS_DEPS.log
    tf_expand_dims = KERAS_DEPS.expand_dims
    tf_tile = KERAS_DEPS.tile
    tf_concat = KERAS_DEPS.concat
    tf_shape = KERAS_DEPS.shape
    tf_float32=KERAS_DEPS.float32
    tf_exp =KERAS_DEPS.exp 
    tf_rank =KERAS_DEPS.rank 
    tf_assert_equal = KERAS_DEPS.assert_equal 
    tf_convert_to_tensor =KERAS_DEPS.convert_to_tensor 
    
    tf_autograph=KERAS_DEPS.autograph
    tf_autograph.set_verbosity(0)
    
    from .._tensor_validation import validate_model_inputs
    from .._tensor_validation import check_inputs 
    
    from ..utils import set_default_params
    from ..components import (
            Activation, 
            CrossAttention,
            DynamicTimeWindow,
            GatedResidualNetwork,
            HierarchicalAttention,
            MemoryAugmentedAttention,
            MultiDecoder,
            MultiResolutionAttentionFusion,
            MultiScaleLSTM,
            QuantileDistributionModeling,
            VariableSelectionNetwork,
            PositionalEncoding, 
            aggregate_time_window_output, 
            aggregate_multiscale_on_3d
        )
    from .op import process_pinn_inputs, compute_consolidation_residual 
    from .utils import process_pde_modes 
    
    
DEP_MSG = dependency_message('nn.pinn.models') 

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

__all__ =["PiHALNet"] 


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
                "This version of PiHALNet is optimized for and defaults to "
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

class PiHALNet(Model, NNLearner):
    """
    Physics-Informed Hybrid Attentive LSTM Network (PiHALNet).

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
            str, Real, None, StrOptions({"learnable"}),
            LearnableC, FixedC, DisabledC
        ], 
        "mode": [
            StrOptions({'tft', 'pihal', 'tft_like', 'pihal_like',
                        "tft-like", "pihal-like"}), 
            None
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
        mode: Optional[str]=None, 
        name: str = "PiHALNet",
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
        
        self.mode = select_mode (mode ) # default to pihal-like 
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

        self.gw_flow_coeffs_config = ( 
            gw_flow_coeffs if gw_flow_coeffs is not None else {}
            )
        
        # ------------------------------------------------------------------
        # PIHALNet is designed for 1-D consolidation only.  If the user
        # supplies coefficients for the transient 2-D/3-D groundwater-flow
        # equation (K, Ss, Q) they probably meant to instantiate
        # TransFlowSubsNet, which couples consolidation and groundwater flow.
        # ------------------------------------------------------------------
        if self.gw_flow_coeffs_config:
            warnings.warn(
                "PIHALNet ignores groundwater-flow coefficients "
                "(K, Ss, Q).  The model minimises only the 1-D "
                "consolidation residual.  For joint head–subsidence "
                "modelling use `TransFlowSubsNet` instead.",
                UserWarning,
                stacklevel=2,
            )
     
        self._build_halnet_layers()
        self._build_pinn_components()
            
    def run_halnet_core(
        self,
        static_input: Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        training: bool
    ) -> Tensor:
        r"""
        Execute the core **encoder–decoder** data‑driven pipeline of
        :class:`~fusionlab.nn.pinn.PIHALNet`.
        
        Executes data-driven pipeline with flexible encoder-decoder logic.
        
        The method ingests *static*, *dynamic* (past), and *future*
        covariates, passes them through Variable Selection Networks
        (VSNs) or dense projections, and then processes them with a
        multi‑scale LSTM encoder and a hierarchical attention‑augmented
        decoder to produce a single latent vector per sample.  This
        vector is subsequently fed to the model’s task‑specific output
        head (not shown here).
        
        Parameters
        ----------
        static_input : Tensor
            Tensor of shape ``(B, S)`` containing time‑invariant
            features such as lithology or well depth, where *B* is the
            batch size and *S* is ``static_input_dim``.
        dynamic_input : Tensor
            Past time‑series of length :math:`T_\mathrm{past}` with
            shape ``(B, T_past, D_in)``.  Typical examples are
            historical groundwater levels or precipitation.
        future_input : Tensor
            Known future covariates of length
            :pyattr:`forecast_horizon` with shape
            ``(B, T_future, F_in)``.
        training : bool
            Flag forwarded to Keras layers to enable dropout and other
            training‑only behaviour.
        
        Returns
        -------
        Tensor
            A 2‑D tensor of shape ``(B, A)``, where *A* is
            ``attention_units``.  This latent representation encodes the
            fused historical context, static descriptors, and known
            future information.
        
        Notes
        -----
        * If :pyattr:`use_vsn` is *True*, each input type is first passed
          through a Variable Selection Network that outputs both
          feature‑wise importance weights and transformed features.
        * Duplicate temporal resolutions produced by
          :pyclass:`~fusionlab.layers.MultiScaleLSTM` are aggregated with
          :pyfunc:`fusionlab.ops.aggregate_multiscale_on_3d`.
        * Duplicate residual connections follow the original TFT design
          but employ GRN‑based :math:`\mathrm{Add}\!\!+\!\!
          \mathrm{LayerNorm}` blocks for improved stability.
 
        """
        time_steps = tf_shape(dynamic_input)[1]

        # 1. Initial Feature Processing
        static_context, dyn_proc, fut_proc = None, dynamic_input, future_input
        if self.use_vsn:
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
        
        logger.debug(f"Initial processed shapes: Dynamic={dyn_proc.shape}, "
                     f"Future={fut_proc.shape}")

        # 2. Encoder Path
        encoder_input_parts = [dyn_proc]
        if self.mode == 'tft_like':
            # For TFT mode, slice historical part of future features
            # and add to the encoder input.
            fut_enc_proc = fut_proc[:, :time_steps, :]
            encoder_input_parts.append(fut_enc_proc)
        
        encoder_raw = tf_concat(encoder_input_parts, axis=-1)
        encoder_input = self.encoder_positional_encoding(encoder_raw)
        lstm_out = self.multi_scale_lstm(
            encoder_input, training=training 
            )
        encoder_sequences = aggregate_multiscale_on_3d(
            lstm_out, mode='concat')
        
        if self.dynamic_time_window is not None:
            encoder_sequences = self.dynamic_time_window(
                encoder_sequences, training=training)
            
        logger.debug(f"Encoder sequences shape: {encoder_sequences.shape}")

        # 3. Decoder Path
        if self.mode == 'tft_like':
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
    
        # --- 4. Attention-based Fusion (Encoder-Decoder Interaction) ---
        cross_attention_output = self.cross_attention(
            [projected_decoder_input, encoder_sequences], training=training)
        
        cross_attention_processed = self.attention_processing_grn(
            cross_attention_output, training=training)
    
        # First residual connection within the decoder block.
        if self.use_residuals and self.decoder_add_norm is not None:
            decoder_context_with_attention = self.decoder_add_norm[0](
                [projected_decoder_input, cross_attention_processed])
            decoder_context_with_attention = self.decoder_add_norm[1](
                decoder_context_with_attention)
        else:
            # If no residual, the processed attention output becomes the context.
            decoder_context_with_attention = cross_attention_processed
        
        hierarchical_att_output = self.hierarchical_attention(
            [decoder_context_with_attention, decoder_context_with_attention],
            training=training
        )
        memory_attention_output = self.memory_augmented_attention(
            hierarchical_att_output, training=training
        )
        
        # --- 5. Final Feature Combination for Decoding ---
        final_features = self.multi_resolution_attention_fusion(
             memory_attention_output, training=training
        )
        
        if self.use_residuals and self.final_add_norm:
            # The `residual_base` must have the same dimension as `final_features`
            residual_base = self.residual_dense(
                decoder_context_with_attention)
            final_features = self.final_add_norm[0](
                [final_features, residual_base])
            final_features = self.final_add_norm[1](final_features)
        
        logger.debug(f"Shape after final fusion: {final_features.shape}")
    
        # Collapse the time dimension to get a single vector per sample.
        final_features_for_decode = aggregate_time_window_output(
            final_features, self.final_agg
        )
        logger.debug(f"Final features for decoder shape: "
                     f"{final_features_for_decode.shape}")
        
        return final_features_for_decode

    def split_outputs(
        self, 
        predictions_combined: Tensor, 
        decoded_outputs_for_mean: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        r"""
        Separate the **combined output tensor** into individual
        subsidence and groundwater‑level (GWL) components and return
        both the *final* and *mean* predictions needed for the two loss
        terms used in PIHALNet (data loss and physics/PDE loss).
        
        The method supports two output shapes:
        
        * **Quantile mode**  
          ``(B, H, Q, C)`` where *Q* is the number of quantiles and
          *C* = ``output_subsidence_dim + output_gwl_dim``.
        * **Deterministic mode**  
          ``(B, H, C)`` when quantiles are disabled.
        
        Parameters
        ----------
        predictions_combined : Tensor
            Network output after the
            :class:`~fusionlab.nn.pinn.QuantileDistributionModeling`
            stage.  Shape is ``(B, H, C)`` or ``(B, H, Q, C)``.
        decoded_outputs_for_mean : Tensor
            Decoder output *before* quantile distribution, used to
            compute the PDE residual.  Shape is ``(B, H, C)``.
        training : bool, optional
            *Inherited from the calling context.*  Present only in
            TensorFlow graph mode; not used explicitly here.
        
        Returns
        -------
        s_pred_final : Tensor
            Subsidence predictions ready for the data‑fidelity loss.
            Shape matches ``predictions_combined`` minus the *C* split.
        gwl_pred_final : Tensor
            GWL predictions ready for the data‑fidelity loss.
        s_pred_mean_for_pde : Tensor
            Mean (deterministic) subsidence predictions used when
            computing physics‑based derivatives.
        gwl_pred_mean_for_pde : Tensor
            Mean GWL predictions for the PDE residual term.
        
        Notes
        -----
        * Mean predictions are extracted *only* from
          ``decoded_outputs_for_mean`` because applying the quantile
          mapping first would break the differentiability required for
          spatial–temporal derivatives.
        * When TensorFlow executes in graph mode and the rank of
          *predictions_combined* is dynamic, the function falls back to
          :pyfunc:`tf.rank` for shape inspection.
        
        Examples
        --------
        >>> outputs = model(...)                       # forward pass
        >>> s_final, gwl_final, s_mean, gwl_mean = (
        ...     model.split_outputs(
        ...         predictions_combined=outputs["pred"],
        ...         decoded_outputs_for_mean=outputs["dec_mean"],
        ...     )
        ... )
        >>> s_final.shape
        TensorShape([32, 24, 3])          # e.g. B=32, H=24, Q=3
        >>> gwl_mean.shape
        TensorShape([32, 24, 1])          # deterministic mean
        
        See Also
        --------
        fusionlab.nn.pinn.QuantileDistributionModeling :
            Layer that adds the quantile dimension.
        fusionlab.nn.pinn.PIHALNet.run_halnet_core :
            Produces ``decoded_outputs_for_mean``.
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
    
    @tf_autograph.experimental.do_not_convert
    def call(
        self,
        inputs: Dict[str, Optional[Tensor]],
        training: bool = False
    ) -> Dict[str, Tensor]:
        r"""
        Forward pass for :class:`~fusionlab.nn.pinn.PIHALNet`.
        
        This method orchestrates the full **physics‑informed** workflow:
        
        1. Validate and split the input dictionary into static, dynamic,
           future, and coordinate tensors.
        2. Run the HALNet encoder–decoder core to obtain a latent
           representation.
        3. Produce mean forecasts with the multi‑horizon decoder.
        4. Optionally expand those means into quantile predictions.
        5. Separate combined outputs into subsidence and GWL streams for
           both data‑loss and physics‑loss branches.
        6. Compute the consolidation PDE residual on the mean series.
        7. Return a dictionary ready for the model’s composite loss.
        
        Parameters
        ----------
        inputs : dict[str, Tensor]
            Dictionary containing at least the following keys
            (created by :pyfunc:`fusionlab.nn.pinn.process_pinn_inputs`):
        
            * ``'coords'`` – tensor ``(B, H, 3)`` with
              *(t, x, y)* coordinates.
            * ``'static_features'`` – tensor ``(B, S)``.
            * ``'dynamic_features'`` – tensor ``(B, T_past, D)``.
            * ``'future_features'`` – tensor ``(B, H, F)``.
        
        training : bool, default=False
            Standard Keras flag indicating training or inference mode.
        
        Returns
        -------
        dict[str, Tensor]
            * ``'subs_pred'`` – subsidence predictions, shape
              ``(B, H, Q, O_s)`` or ``(B, H, O_s)``.
            * ``'gwl_pred'`` – GWL predictions, same layout for *O_g*.
            * ``'pde_residual'`` – physics residual,
              shape ``(B, H, 1)`` (all zeros if *H = 1*).
        
        Notes
        -----
        * Quantile outputs are produced only when the model’s
          ``quantiles`` attribute is not *None*.
        * The PDE residual is based on a discrete‑time consolidation
          equation evaluated with finite differences; therefore a
          forecast horizon > 1 is required.
        
        Examples
        --------
        >>> out = pihalnet(
        ...     {
        ...         "coords": coords,
        ...         "static_features": s,
        ...         "dynamic_features": d,
        ...         "future_features": f,
        ...     },
        ...     training=True,
        ... )
        >>> out["subs_pred"].shape
        TensorShape([32, 24, 3, 1])  # B=32, H=24, Q=3, O_s=1
        >>> out["pde_residual"].shape
        TensorShape([32, 24, 1])
        
        See Also
        --------
        fusionlab.nn.pinn.PIHALNet.run_halnet_core :
            Feature‑extraction backbone called internally.
        fusionlab.nn.pinn.PIHALNet.split_outputs :
            Helper that separates subsidence and GWL channels.
        """
        # 1. Unpack and Validate Inputs
        # - Process and Validate All Inputs ---
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
            forecast_horizon= self.forecast_horizon, 
            verbose=0 # Set to >0 for  detailed logging from checks
            
        )
        # `validate_model_inputs` can provide a secondary, more detailed
        # check on the unpacked feature tensors.
        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=[static_features, dynamic_features, future_features],
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
        # Let's verify the shape and slice if needed
        # (this is a patch, not the ideal fix)
        if t.shape[1] != self.forecast_horizon:
             # This indicates a data pipeline issue, but we can patch it here
             # by assuming the last `forecast_horizon` steps of `t` are the correct ones.
             # This assumption may be incorrect depending on your sequence prep.
             t_for_pde = t[:, -self.forecast_horizon:, :]
             # A better approach is to fix the data pipeline so `inputs['coords']`
             # has the shape (batch, forecast_horizon, 3) from the start.
        else:
            t_for_pde = t
         
        # The PDE residual is calculated on the mean predictions using
        # finite differences, which is suitable for sequence outputs.
        # This does NOT require a GradientTape in the call method.
        logger.debug("Computing PDE residual from mean predictions.")
        if self.forecast_horizon > 1:
            pde_residual = compute_consolidation_residual(
                s_pred=s_pred_mean_for_pde,
                h_pred=gwl_pred_mean_for_pde,
                time_steps=t_for_pde, #  `t` holds the forecast time sequence
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
        lambda_pde: float = 1.0,
        **kwargs,
    ):
        r"""
        Configure PIHALNet for training with a **composite PINN loss**.

        The total loss optimised during :pyfunc:`train_step`
        is

        .. math::

           L_\text{total} \;=\;
           \sum_i w_i\,L_{\text{data},i}
           + \lambda_\text{pde}\,L_\text{pde}

        where *i* indexes the outputs (subsidence, GWL).

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer | str
            Optimiser instance or Keras alias (e.g., ``"adam"``).
        loss : dict
            Mapping ``{"subs_pred": loss_fn, "gwl_pred": loss_fn}``.
            Each value can be a Keras loss object or a string such
            as ``"mse"``.
        metrics : dict, optional
            Mapping from output keys to a list of Keras metric objects
            (or their aliases) that will be tracked during training and
            evaluation.
        loss_weights : dict, optional
            Scalar weights :math:`w_i` for each *data* loss term.
            Defaults to 1 for every output.
        lambda_pde : float, default=1.0
            Weight applied to :math:`L_\text{pde}` (physics residual).
        **kwargs
            Additional keywords forwarded to
            :pyfunc:`tf.keras.Model.compile`.

        Notes
        -----
        * The physics‑residual term is added manually in
          :pyfunc:`train_step`; therefore ``lambda_pde`` is stored as an
          attribute rather than passed to ``loss_weights``.
        """
        # Call the parent's compile method. It will handle the setup of
        # losses, metrics, and weights for our named outputs.
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            **kwargs,
        )
        # Store the PINN-specific loss weight
        self.lambda_pde = lambda_pde


    def train_step(self, data: Tuple[Dict, Dict]) -> Dict[str, Tensor]:
        r"""
        Single optimisation step implementing the composite loss.

        The procedure is:

        1. Forward pass → ``self.call`` to obtain
           ``{"subs_pred", "gwl_pred", "pde_residual"}``.
        2. Compute data‑fidelity loss via
           :pyfunc:`tf.keras.Model.compute_loss`.
        3. Compute physics residual loss  
           :math:`L_\text{pde} = \langle r^2\rangle` where
           ``r = outputs["pde_residual"]``.
        4. Form  
           :math:`L_\text{total}=L_\text{data} +
           \lambda_\text{pde}\,L_\text{pde}` and back‑propagate.
        5. Update Keras metrics and return a results dictionary.

        Parameters
        ----------
        data : tuple(dict, dict)
            Tuple ``(inputs, targets)`` produced by the data pipeline.

        Returns
        -------
        dict[str, Tensor]
            Includes user‑defined metrics plus ``"total_loss"``,
            ``"data_loss"``, and ``"physics_loss"``.

        Raises
        ------
        ValueError
            If *data* is not a two‑element tuple of dictionaries.
        """
        # Unpack the data. Keras provides it as a tuple of (inputs, targets).
        # We expect both to be dictionaries.
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError(
                "Expected data to be a tuple of (inputs_dict, targets_dict)."
            )
        inputs, targets = data

        # Open a GradientTape to record operations 
        # for automatic differentiation.
        with GradientTape() as tape:
            # 1. Forward pass to get model outputs
            # The `call` method returns a dict:
                #   {'subs_pred', 'gwl_pred', 'pde_residual'}
            outputs = self(inputs, training=True)

            # Structure predictions to match the 
            # 'loss' dictionary from compile()
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
        # Update the metrics passed in compile() 
        # (e.g., 'mae', 'rmse' for each output)
        self.compiled_metrics.update_state(targets, y_pred_for_loss)

        # Build a dictionary of results to be displayed by Keras.
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "total_loss": total_loss,    # The main loss we optimized
            "data_loss": data_loss,      # The part of the loss from data
            "physics_loss": loss_pde,    # The part of the loss from physics
        })
        return results

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
        """Instantiates all layers for the HALNet architecture.
    
        This method sets up all necessary components, including conditional
        Variable Selection Networks (VSNs) and the core layers for the
        encoder-decoder structure like LSTMs and attention mechanisms.
        """
        # --- 1. Variable Selection Networks (Conditional) ---
        # These layers are created only if `self.use_vsn` is True.
        # They serve to select the most relevant input features and embed
        # them into a common feature space.
        if self.use_vsn:
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
                self.static_vsn, self.static_vsn_grn = None, None
    
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
                self.dynamic_vsn, self.dynamic_vsn_grn = None, None
    
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
                self.future_vsn, self.future_vsn_grn = None, None
        else:
            # If not using VSNs, ensure all related attributes are None.
            self.static_vsn, self.static_vsn_grn = None, None
            self.dynamic_vsn, self.dynamic_vsn_grn = None, None
            self.future_vsn, self.future_vsn_grn = None, None
    
        # --- 2. Shared & Non-VSN Path Layers ---
        # These layers are essential for processing attention outputs or
        # initial features if VSNs are disabled.
    
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
        if not self.use_vsn: 
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
            
     
        # --- 3. Core Architectural Layers ---
        # self.positional_encoding = PositionalEncoding()
        # *** FIX: Create two separate instances of PositionalEncoding ***
        self.encoder_positional_encoding = PositionalEncoding(
            name="encoder_pos_encoding")
        self.decoder_positional_encoding = PositionalEncoding(
            name="decoder_pos_encoding")
    
        self.multi_scale_lstm = MultiScaleLSTM(
            lstm_units=self.lstm_units,
            scales=self.scales,
            return_sequences=True  # Critical for the encoder path
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
            output_dim=self._combined_output_target_dim,
            num_horizons=self.forecast_horizon
        )
        self.quantile_distribution_modeling = QuantileDistributionModeling(
            quantiles=self.quantiles,
            output_dim=self._combined_output_target_dim
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
           'mode': self.mode, 
           'pde_mode': self.pde_modes_active, 
           'gw_flow_coeffs': self.gw_flow_coeffs_config,
        }
        config.update(base_config)
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)         
    
PiHALNet.__doc__+=r"""\n
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
>>> from fusionlab.nn.pinn.models import PiHALNet
>>> model = PiHALNet(
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
