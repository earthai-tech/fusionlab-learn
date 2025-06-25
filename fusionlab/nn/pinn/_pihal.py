# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Physics-Informed Hybrid Attentive LSTM Network (PIHALNet).
"""
from __future__ import annotations
from numbers import Integral, Real 
from typing import List, Optional, Union, Dict, Tuple 

from ..._fusionlog import fusionlog, OncePerMessageFilter 
from ...core.handlers import param_deprecated_message 
from ...compat.sklearn import validate_params, Interval, StrOptions 
from ...params import LearnableC, FixedC, DisabledC
from ...utils.deps_utils import ensure_pkg
from ...utils.generic_utils import rename_dict_keys 

from .. import KERAS_BACKEND, KERAS_DEPS,  dependency_message
from .._base_attentive import BaseAttentive 

if KERAS_BACKEND:
    from .._tensor_validation import check_inputs
    from .op import process_pinn_inputs, compute_consolidation_residual
    from .utils import process_pde_modes, _get_coords  
  
MeanSquaredError = KERAS_DEPS.Adam 
Adam =KERAS_DEPS.Adam 
Tensor = KERAS_DEPS.Tensor
Constant =KERAS_DEPS.Constant 
GradientTape = KERAS_DEPS.GradientTape

tf_exp = KERAS_DEPS.exp
tf_square = KERAS_DEPS.square
tf_reduce_mean = KERAS_DEPS.reduce_mean
tf_zeros_like = KERAS_DEPS.zeros_like
tf_rank =KERAS_DEPS.rank 
tf_exp =KERAS_DEPS.exp 
tf_constant =KERAS_DEPS.constant 
tf_log = KERAS_DEPS.log
tf_constant =KERAS_DEPS.constant
tf_float32=KERAS_DEPS.float32

tf_autograph=KERAS_DEPS.autograph
tf_autograph.set_verbosity(0)
 
register_keras_serializable =KERAS_DEPS.register_keras_serializable

DEP_MSG = dependency_message('nn.pinn.models') 

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

__all__ = ["PIHALNet"]

@register_keras_serializable(
    'fusionlab.nn.pinn', name="PIHALNet"
)
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'pde_mode', 
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
                "the primary focus of the current physics-informed implementation. "
                "This version of PIHALNet is optimized for and defaults to "
                "'consolidation' mode to ensure robust physical constraints "
                "based on Terzaghi's theory with finite differences. "
                "The model will proceed using 'consolidation' mode. Full "
                "support for other PDE modes and their specific derivative "
                "can be explored in 'fusionlab.nn.pinn.TransFlowSubsNet' instead."
            ),
            'default': 'consolidation', 
        }
    ],
    warning_category=UserWarning 
)

class PIHALNet(BaseAttentive):
    """
    Physics-Informed Hybrid Attentive-based LSTM Network (PIHALNet).

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
        'output_subsidence_dim': [
            Interval(Integral,1, None, closed="left")], 
        'output_gwl_dim': [
            Interval(Integral,1, None, closed="left"),], 
        "pde_mode": [
            StrOptions({'consolidation', 'gw_flow', 'both', 'none'}), 
            'array-like', None 
        ],
        "pinn_coefficient_C": [
            str, Real, None, StrOptions({"learnable"}),
            LearnableC, FixedC, DisabledC
        ], 
        "gw_flow_coeffs": [dict, None], 
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
        attention_levels:Optional[Union[str, List[str]]]=None, 
        architecture_config: Optional[Dict] = None,
        name: str = "PIHALNet",
        **kwargs
    ):
        self._combined_output_target_dim = (
            output_subsidence_dim + output_gwl_dim
        )
        if 'output_dim' in kwargs: 
            kwargs.pop ('output_dim')
            
        super().__init__(
           static_input_dim=static_input_dim, 
           dynamic_input_dim=dynamic_input_dim, 
           future_input_dim=future_input_dim, 
           output_dim= self._combined_output_target_dim,
           forecast_horizon=forecast_horizon,
           mode=mode, 
           quantiles=quantiles,
           embed_dim=embed_dim,
           hidden_units=hidden_units,
           lstm_units=lstm_units,
           attention_units=attention_units,
           num_heads=num_heads,
           dropout_rate=dropout_rate,
           max_window_size=max_window_size,
           memory_size=memory_size,
           scales=scales,
           multi_scale_agg=multi_scale_agg,
           final_agg=final_agg,
           activation=activation,
           use_residuals=use_residuals,
           use_vsn=use_vsn,
           use_batch_norm =use_batch_norm, 
           vsn_units=vsn_units,
           attention_levels =attention_levels,
           architecture_config=architecture_config,
           name=name,
            **kwargs
        )
        # Initialize only PINN-specific attributes
        self.output_subsidence_dim = output_subsidence_dim
        self.output_gwl_dim = output_gwl_dim
        
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

        
        # Build the physics-related components
        self._build_pinn_components()

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
        # The `process_pinn_inputs` helper unpacks the input dict and
        # isolates the coordinate tensors for later use.
        logger.debug("PIHALNet call: Processing PINN inputs.")
        t, x, y, static_features, dynamic_features, future_features = \
            process_pinn_inputs(inputs, mode='auto')
            
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
         
        logger.debug(
            "Input shapes after validation:"
            f" S={getattr(static_features, 'shape', 'None')}, "
            f"D={getattr(dynamic_features, 'shape', 'None')},"
            f" F={getattr(future_features, 'shape', 'None')}"
        )
        
        # 2. Get the data-driven predictions from the base model
        # The base call handles validation and the core encoder-decoder
        #  Get the data-driven predictions from the base model
        # The base call handles validation and the core encoder-decoder
        # outputs the decoder final predictions 
        predictions_final_targets = super().call(
            [static_features, dynamic_features, future_features], 
            training=training)
        
        # `base_outputs` contains the combined predictions.
        # We need both the final (potentially quantile) predictions for
        # data loss and the mean predictions for the PDE residual.
       
        logger.debug(
            f"Shape of decoded outputs (means):"
            f" {predictions_final_targets.shape}")

        # --- 4. Split and Organize Outputs ---
        # Use helper to separate subsidence and GWL predictions for both
        # data loss (quantiles) and physics loss (mean).
        (s_pred_final, gwl_pred_final, 
         s_pred_mean_for_pde, gwl_pred_mean_for_pde) = self.split_outputs(
             predictions_combined=predictions_final_targets,
             decoded_outputs_for_mean=self._decoded_outputs
         )
        # --- 5. Calculate Physics Residual ---
        # Let's verify the shape and slice if needed
        # (this is a patch, not the ideal fix)
        if t.shape[1] != self.forecast_horizon:
             # This indicates a data pipeline issue, but we can patch it here
             # by assuming the last `forecast_horizon` steps of `t`
             # are the correct ones.
             # This assumption may be incorrect depending on your sequence prep.
             t_for_pde = t[:, -self.forecast_horizon:, :]
             # A better approach is to fix the data pipeline so `inputs['coords']`
             # has the shape (batch, forecast_horizon, 3) from the start.
        else:
            t_for_pde = t
        
        # 3. Compute the physics residual 
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
  
        #  4. Return the combined dictionary for the loss function
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
        if isinstance (targets, dict): 
            # For consistency, map targets if users explicitely 
            # provide 'subsidence', and 'gwl' as keys .
            targets = rename_dict_keys(
                targets.copy(),
                param_to_rename={
                    "subsidence": "subs_pred", 
                    "gwl": "gwl_pred"
                }
        )

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

    def get_config(self):
        """Returns the full configuration of the PIHALNet model."""
        # Get the config from the base class first
        base_config = super().get_config()
        # Add PIHALNet-specific parameters
        base_config.update({
            "output_subsidence_dim": self.output_subsidence_dim,
            "output_gwl_dim": self.output_gwl_dim,
            "pde_mode": self.pde_modes_active,
            "pinn_coefficient_C": self.pinn_coefficient_C_config,
            "gw_flow_coeffs": self.gw_flow_coeffs_config, 
            "name": self.name # for consistency 
        })
        # The base config already contains all other necessary params
        return base_config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        Instantiates the class from a configuration dictionary.
    
        This method is a factory method for creating an instance of the class 
        from a dictionary containing the configuration. The `output_dim` is 
        removed from the configuration since it is computed dynamically based 
        on other parameters (e.g., `output_subsidence_dim + output_gwl_dim`), 
        ensuring that it is not redundant in the configuration.
    
        Parameters
        ----------
        config : dict
            A dictionary containing the configuration parameters for the class.
            The dictionary should contain all the required parameters except for 
            `output_dim`, which is computed internally.
        
        custom_objects : dict, optional
            A dictionary of custom objects that might be needed to interpret 
            the configuration, such as custom layers, activation functions, 
            or other components.
    
        Returns
        -------
        object
            An instance of the class, initialized with the parameters from 
            the `config` dictionary.
    
        Notes
        -----
        - The method removes the `'output_dim'` key from the configuration 
          because it is derived from the sum of `output_subsidence_dim` and 
          `output_gwl_dim`, and including it would result in redundancy.
        - This method is typically used for loading a model or initializing 
          an object from a saved configuration, which is common in model 
          serialization/deserialization.
        
        Example
        -------
        >>> config = {
        >>>     'output_subsidence_dim': 2,
        >>>     'output_gwl_dim': 3,
        >>>     'other_param': 42
        >>> }
        >>> model = MyModel.from_config(config)
        >>> print(model.output_dim)  # Will be computed as 5, not directly passed.
        """
        if 'output_dim' in config: 
            config.pop('output_dim')  # Remove 'output_dim' since it is computed
                                      # from output_subsidence_dim + output_gwl_dim.
        return cls(**config)
   

    
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
