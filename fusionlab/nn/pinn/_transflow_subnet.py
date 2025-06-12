# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations
from numbers import Integral, Real 
from typing import Optional, Union, Dict, List, Any, Tuple

from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...compat.sklearn import validate_params, Interval, StrOptions 
from ...utils.deps_utils import ensure_pkg 
from ...params import (
    LearnableK, LearnableSs, LearnableQ, LearnableC,
    FixedC, DisabledC, resolve_physical_param
)

from .. import KERAS_BACKEND, KERAS_DEPS, dependency_message 
from .._base_attentive import BaseAttentive


if KERAS_BACKEND:
    
    from .._tensor_validation import check_inputs, validate_model_inputs 
    from .op import process_pinn_inputs
    from .utils import process_pde_modes 
    from ..comp_utils import resolve_gw_coeffs, normalize_C_descriptor
    
    LSTM = KERAS_DEPS.LSTM
    Dense = KERAS_DEPS.Dense
    LayerNormalization = KERAS_DEPS.LayerNormalization 
    Model= KERAS_DEPS.Model 
    Tensor=KERAS_DEPS.Tensor
    Variable =KERAS_DEPS.Variable 
    Add =KERAS_DEPS.Add
    Constant =KERAS_DEPS.Constant 
    GradientTape =KERAS_DEPS.GradientTape 

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
    
    Model = KERAS_DEPS.Model
    Tensor = KERAS_DEPS.Tensor
    GradientTape = KERAS_DEPS.GradientTape
    
else:
    class Model: pass
    class Layer: pass
    class _DummyRegister_keras_serializable: 
        pass 
    class KERAS_DEPS: 
        register_keras_serializable = _DummyRegister_keras_serializable
        
    Tensor = Any
    
DEP_MSG = dependency_message('nn.pinn.models') 
logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

__all__ = ["TransFlowSubsNet"]

@KERAS_DEPS.register_keras_serializable(
    'fusionlab.nn.pinn', name="TransFlowSubsNet") 
class TransFlowSubsNet(BaseAttentive):
    """Transient Groundwater Flow-Driven Subsidence Network.

    This class inherits the data-driven architecture from
    BaseAttentive and adds a physics-informed module for coupled
    groundwater flow and consolidation processes.
    """
    @validate_params({
        'output_subsidence_dim': [Interval(Integral,1, None, closed="left")], 
        'output_gwl_dim': [Interval(Integral,1, None, closed="left"),], 
        "pde_mode": [
            StrOptions({'consolidation', 'gw_flow', 'both', 'none'}), 
            'array-like', None 
        ],
        "pinn_coefficient_C": [
            str, Real, None, StrOptions({"learnable"}),
            LearnableC, FixedC, DisabledC
        ], 
        "K": [Real, None, StrOptions({"learnable"}),LearnableK], 
        "Ss": [Real, None, StrOptions({"learnable"}),LearnableSs], 
        "Q": [Real, None, StrOptions({"learnable"}),LearnableQ,], 
        "gw_flow_coeffs": [dict, None], }
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)   
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_subsidence_dim: int = 1,
        output_gwl_dim: int = 1,
        pde_mode: Union[str, List[str]] = 'both',
        K: Union[float, LearnableK] = 1e-4,
        Ss: Union[float, LearnableSs] = 1e-5,
        Q: Union[float, LearnableQ] = 0.0,
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
        pinn_coefficient_C: Union[
            LearnableC, FixedC, DisabledC, str, float, None
        ] = LearnableC(initial_value=0.01),
        gw_flow_coeffs: Optional[Dict[str, Union[str, float, None]]] = None,
        use_vsn: bool = True,
        vsn_units: Optional[int] = None,
        mode: Optional[str]=None, 
        objective: Optional[str]=None, 
        attention_levels:Optional[Union[str, List[str]]]=None, 
        name: str = "TransFlowSubsNet",
        **kwargs
    ):
        # The total output dimension for the base data-driven model
        # is the sum of the subsidence and GWL dimensions.
        self._combined_output_dim = (
            output_subsidence_dim + output_gwl_dim
        )
        # Pass all shared architectural parameters to the parent class.
        super().__init__(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            output_dim=self._combined_output_dim,
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
            objective=objective, 
            name=name,
            **kwargs
        )
        # Initialize PINN-specific attributes.
        self.output_subsidence_dim = output_subsidence_dim
        self.output_gwl_dim = output_gwl_dim
        self.pde_modes_active = process_pde_modes(pde_mode)
        
        # Store configurations for serialization.
        # Normalize pinn_coefficient_C into one of our  legacy
        self.pinn_coefficient_C_config = normalize_C_descriptor(
            pinn_coefficient_C
        )
        # This shows how the helper simplifies the logic.
        K, Ss, Q = resolve_gw_coeffs(
            gw_flow_coeffs=gw_flow_coeffs,
            K=K, Ss=Ss, Q=Q, 
            param_status="learnable"
        )

        logger.info(f"Initialized with K={K}, Ss={Ss}, Q={Q}")
        
        self.K_config = K
        self.Ss_config = Ss
        self.Q_config = Q

        # Build the physics-related components.
        self._build_pinn_components()
        
    def _build_C_components(self):
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
                "Internal error: pinn_coefficient_C_config"
                " is not a recognized type."
            )

    def get_pinn_coefficient_C(self) -> Tensor:
        """Returns the physical coefficient C."""
        return self._get_C()

    def _resolve_pinn_components(self):
        """Instantiates trainable/fixed physical coefficients."""
        # Create tf.Variables as direct attributes for Keras to track.
        self.C = self.get_pinn_coefficient_C() 
        self.K = resolve_physical_param(
            self.K_config, name="param_K")
        self.Ss = resolve_physical_param(
            self.Ss_config, name="param_Ss")
        self.Q = resolve_physical_param(
            self.Q_config, name="param_Q")
        

    def _build_pinn_components(self):
        """
        Instantiates trainable/fixed physical coefficients using the
        correct Keras API to ensure they are tracked by the model.
        """
        # This method is already correct because it uses self.add_weight.
        # It creates `self.log_C_coefficient` if learnable.
        self._build_C_components()
        
        # This getter now correctly accesses the learnable variable.
        self.C = self.get_pinn_coefficient_C()
    
        # --- FIX: Create K, Ss, and Q using self.add_weight ---
        # This ensures they are registered as trainable variables.
    
        # Handle Hydraulic Conductivity (K)
        if isinstance(self.K_config, LearnableK):
            self.log_K = self.add_weight(
                name="log_param_K",
                shape=(),
                initializer=Constant(
                    tf_log(self.K_config.initial_value)
                ),
                trainable=True
            )
            self.K = tf_exp(self.log_K)
        else:  # Fixed value
            self.K = tf_constant(float(self.K_config), dtype=tf_float32)
    
        # Handle Specific Storage (Ss)
        if isinstance(self.Ss_config, LearnableSs):
            self.log_Ss = self.add_weight(
                name="log_param_Ss",
                shape=(),
                initializer=Constant(
                    tf_log(self.Ss_config.initial_value)
                ),
                trainable=True
            )
            self.Ss = tf_exp(self.log_Ss)
        else:  # Fixed value
            self.Ss = tf_constant(float(self.Ss_config), dtype=tf_float32)
    
        # Handle Source/Sink Term (Q)
        if isinstance(self.Q_config, LearnableQ):
            self.Q = self.add_weight(
                name="param_Q",
                shape=(),
                initializer=Constant(
                    self.Q_config.initial_value
                ),
                trainable=True
            )
        else:  # Fixed value
            self.Q = tf_constant(float(self.Q_config), dtype=tf_float32)
            
            
    # In your TransFlowSubsNet 
    def call(self, inputs: Dict[str, Optional[Tensor]], training: bool = False):
        """
        Orchestrates the forward pass for the PINN, combining the data-driven
        and physics-informed modules.
        """
        # --- 1. Unpack and Validate All Inputs ---
        # The `process_pinn_inputs` helper unpacks the input dict and
        # isolates the coordinate tensors for later use. It is assumed
        # that the `inputs` dict contains all necessary keys.
        logger.debug("TransFlowSubsNet call: Unpacking and validating inputs.")
        (t, x, y, static_features,
         dynamic_features, future_features) = process_pinn_inputs(
             inputs, mode='as_dict'
        )
        
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
        
        # The `validate_model_inputs` provides a detailed check on the
        # unpacked feature tensors to ensure they match model expectations.
        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=[static_features, dynamic_features, future_features],
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            mode='strict',
            verbose= 0 # 1 if logger.level <= 10 else 0
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
        # This method performs the complex feature engineering using LSTMs
        # and attention mechanisms from the BaseAttentive class.
        logger.debug("Running data-driven core for feature extraction.")
        final_features_for_decode = self.run_encoder_decoder_core(
            static_input=static_p,
            dynamic_input=dynamic_p,
            future_input=future_p,
            training=training
        )
        # --- 3. Generate Predictions ---
        # Get mean predictions (for PDE) from the multi-horizon decoder.
        decoded_means = self.multi_decoder(
            final_features_for_decode, training=training
        )
        # Get final predictions (potentially with quantiles) for data loss.
        final_predictions = decoded_means
        if self.quantiles is not None:
            final_predictions = self.quantile_distribution_modeling(
                decoded_means, training=training
            )
    
        # --- 4. Split and Organize Outputs ---
        # Separate the combined output tensor into subsidence and GWL streams.
        (s_pred_final, gwl_pred_final,
         s_pred_mean, gwl_pred_mean) = self.split_outputs(
             predictions_combined=final_predictions,
             decoded_outputs_for_mean=decoded_means
         )
    
        # --- 5. Calculate Physics Residuals ---
        logger.debug("Computing PDE residuals from mean predictions.")
        
        # *** FIX: Access coordinates explicitly by key, not by index. ***
        # The `coords` used for the PDE must be the same ones passed in the
        # main `inputs` dictionary.
        try: 
            coords_for_pde = inputs['coords']
        except : 
            # reconstruct input for coordinate 
            coords_for_pde = {'t': t, 'x': x, 'y': y}
    
        # Compute the residual for each active PDE mode.
        cons_res = self._compute_consolidation_residual(
            coords_for_pde, s_pred_mean, gwl_pred_mean
        )
        gw_res = self._compute_gw_flow_residual(
            coords_for_pde, gwl_pred_mean
        )
    
        # --- 6. Return All Components for the Loss Function ---
        return {
            "subs_pred": s_pred_final,
            "gwl_pred": gwl_pred_final,
            "consolidation_residual": cons_res,
            "gw_flow_residual": gw_res,
        }

    def train_step(self, data):
        """Custom training step for the composite PINN loss."""
        inputs, targets = data
        
        with GradientTape() as tape:
            # All operations within this block are recorded for differentiation.
            outputs = self(inputs, training=True)
            
            # --- 1. Data Fidelity Loss ---
            # Create a dictionary of predictions that matches the keys
            # expected by the loss functions provided in `compile`.
            y_pred_for_loss = {
                'subs_pred': outputs['subs_pred'],
                'gwl_pred': outputs['gwl_pred']
            }
            # Keras's internal `compute_loss` handles the dictionary,
            # loss weights, and regularization losses automatically.
            data_loss = self.compute_loss(
                y=targets,
                y_pred=y_pred_for_loss,
                regularization_losses=self.losses
            )
            
            # --- 2. Physics-Based Loss ---
            # Calculate the mean of the squared residuals to force them to zero.
            loss_cons = tf_reduce_mean(tf_square(
                outputs['consolidation_residual']))
            loss_gw = tf_reduce_mean(tf_square(
                outputs['gw_flow_residual']))
            
            # --- 3. Combine Losses ---
            # Create the final composite loss that will be differentiated.
            total_loss = (data_loss
                          + self.lambda_cons * loss_cons
                          + self.lambda_gw * loss_gw)

        # Compute gradients of the total loss with respect to all
        # trainable variables (network weights and physical parameters).
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        
        # Apply the gradients to update the variables.
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        # Update the metrics passed in `compile` (e.g., MAE).
        self.compiled_metrics.update_state(targets, y_pred_for_loss)
        
        # Return a dictionary of all computed values for logging.
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "total_loss": total_loss,
            "data_loss": data_loss,
            "consolidation_loss": loss_cons,
            "gw_flow_loss": loss_gw,
        })
        return results

    # In your TransFlowSubsNet class in fusionlab/nn/pinn/_geos.py
    def _compute_consolidation_residual(
        self,
        coords: Dict[str, Tensor],
        s_pred: Tensor,
        h_pred: Tensor
    ) -> Tensor:
        """Computes the consolidation PDE residual.

        This implementation uses a simplified form relating the rate of
        subsidence to the Laplacian of the hydraulic head, representing
        the coupling between the two processes.
        PDE form: ds/dt - C * (d²h/dx² + d²h/dy²) = 0
        """
        # Return zeros if this PDE mode is not active.
        if 'consolidation' not in self.pde_modes_active:
            return tf_zeros_like(s_pred)

        # A persistent tape is needed to compute multiple gradients,
        # especially second-order derivatives.
        with GradientTape(persistent=True) as tape:
            # Watch the coordinate and prediction tensors that need
            # to be differentiated.
            tape.watch([
                coords['t'], coords['x'], coords['y'], s_pred, h_pred
            ])

            # First-order derivative of subsidence w.r.t. time.
            ds_dt = tape.gradient(s_pred, coords['t'])

            # First-order spatial derivatives of head.
            dh_dx = tape.gradient(h_pred, coords['x'])
            dh_dy = tape.gradient(h_pred, coords['y'])

        # Second-order spatial derivatives of head.
        d2h_dx2 = tape.gradient(dh_dx, coords['x'])
        d2h_dy2 = tape.gradient(dh_dy, coords['y'])

        # Release the tape from memory once all gradients are computed.
        del tape

        # Validate that all necessary gradients were computed.
        if any(g is None for g in [ds_dt, d2h_dx2, d2h_dy2]):
            raise ValueError(
                "Failed to compute consolidation gradients. Ensure all "
                "input coordinates influence the predictions."
            )

        # Assemble the residual for the consolidation equation.
        laplacian_h = d2h_dx2 + d2h_dy2
        residual = ds_dt - (self.C * laplacian_h)
        return residual

    def _compute_gw_flow_residual(
        self,
        coords: Dict[str, Tensor],
        h_pred: Tensor
    ) -> Tensor:
        """Computes the transient groundwater flow PDE residual.

        This implements the 2D transient groundwater flow equation.
        PDE form: Ss * dh/dt - K * (d²h/dx² + d²h/dy²) - Q = 0
        """
        # Return zeros if this PDE mode is not active.
        if 'gw_flow' not in self.pde_modes_active:
            return tf_zeros_like(h_pred)

        # Use a persistent tape for calculating multiple derivatives.
        with GradientTape(persistent=True) as tape:
            tape.watch([
                coords['t'], coords['x'], coords['y'], h_pred
            ])

            # First-order derivatives.
            dh_dt = tape.gradient(h_pred, coords['t'])
            dh_dx = tape.gradient(h_pred, coords['x'])
            dh_dy = tape.gradient(h_pred, coords['y'])

        # Second-order spatial derivatives.
        d2h_dx2 = tape.gradient(dh_dx, coords['x'])
        d2h_dy2 = tape.gradient(dh_dy, coords['y'])
        
        # Release the tape.
        del tape

        # Validate gradients.
        if any(g is None for g in [dh_dt, d2h_dx2, d2h_dy2]):
            raise ValueError(
                "Failed to compute groundwater flow gradients. Ensure all "
                "input coordinates influence the head prediction."
            )

        # Assemble the residual for the groundwater flow equation.
        laplacian_h = d2h_dx2 + d2h_dy2
        residual = (self.Ss * dh_dt) - (self.K * laplacian_h) - self.Q
        return residual
    
    def split_outputs(
        self, 
        predictions_combined: Tensor, 
        decoded_outputs_for_mean: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        r"""
        Separate the **combined output tensor** into individual
        subsidence and groundwater‑level (GWL) components and return
        both the *final* and *mean* predictions needed for the two loss
        terms used in TransFlowSubsNet (data loss and physics/PDE loss).
        
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
        fusionlab.nn.pinn.PiHALNet.run_halnet_core :
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
    
    def _split_outputs(
        self,
        predictions_combined: Tensor,
        decoded_outputs_for_mean: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Splits combined model outputs into subsidence and GWL tensors.

        This helper function takes the stacked output tensors from the
        final layers and separates them into their respective parts for
        subsidence and groundwater level, for both the final (potentially
        quantile) predictions and the mean predictions.

        Args:
            predictions_combined: The final output tensor from the model,
                which may include a quantile dimension. Shape is either
                (B, H, O_s + O_g) or (B, H, O_s + O_g, Q).
            decoded_outputs_for_mean: The mean predictions from the
                decoder, used for PDE calculations. Shape is
                (B, H, O_s + O_g).

        Returns:
            A tuple containing four tensors:
            (s_pred_final, gwl_pred_final, s_pred_mean, gwl_pred_mean)
        """
        # The feature dimension is always the one before the quantiles,
        # or the last one if no quantiles are present.
        # feature_axis = -2 if self.quantiles is not None else -1

        # Split the final predictions (for data loss)
        s_pred_final = predictions_combined[
            ..., :self.output_subsidence_dim, :]
        gwl_pred_final = predictions_combined[
            ..., self.output_subsidence_dim:, :]
        
        # Split the mean predictions (for physics loss)
        s_pred_mean_for_pde = decoded_outputs_for_mean[
            ..., :self.output_subsidence_dim]
        gwl_pred_mean_for_pde = decoded_outputs_for_mean[
            ..., self.output_subsidence_dim:]

        return (s_pred_final, gwl_pred_final,
                s_pred_mean_for_pde, gwl_pred_mean_for_pde)
    
    def compile(
        self,
        lambda_cons: float = 1.0,
        lambda_gw: float = 1.0,
        **kwargs
    ):
        """Compiles the model with composite loss weights.

        This method extends the default Keras `compile` method to
        accept weights for the physics-based loss components.

        Args:
            lambda_cons (float, optional): The weight for the
                consolidation PDE residual loss. Defaults to 1.0.
            lambda_gw (float, optional): The weight for the
                groundwater flow PDE residual loss. Defaults to 1.0.
            **kwargs: Standard arguments for `tf.keras.Model.compile`,
                such as `optimizer`, `loss`, and `metrics`.
        """
        super().compile(**kwargs)
        # Store the weights for the physics loss components
        self.lambda_cons = lambda_cons
        self.lambda_gw = lambda_gw


    def get_config(self) -> dict:
        """Returns the full configuration of the model.

        This method serializes the model's configuration by
        combining the configuration of the parent `BaseAttentive`
        class with the parameters specific to this PINN subclass.

        Returns
        -------
        dict
            A dictionary containing all the necessary parameters to
            reconstruct the model.
        """
        # Get the configuration from the base class, which includes
        # all the data-driven architectural parameters.
        base_config = super().get_config()
        
        # Add the parameters that are unique to this subclass.
        pinn_config = {
            "output_subsidence_dim": self.output_subsidence_dim,
            "output_gwl_dim": self.output_gwl_dim,
            "pde_mode": self.pde_modes_active,
            "pinn_coefficient_C": self.pinn_coefficient_C_config,
            "gw_flow_coeffs": self.gw_flow_coeffs, 
            "K": self.K_config,
            "Ss": self.Ss_config,
            "Q": self.Q_config,
        }
        base_config.update(pinn_config)
        
        return base_config

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "TransFlowSubsNet":
        """Reconstructs a model instance from its configuration.

        Args:
            config (dict): Configuration dictionary from `get_config`.
            custom_objects (dict, optional): Unused here, as custom
                objects are expected to be registered with Keras.

        Returns
        -------
        TransFlowSubsNet
            A new instance of the model.
        """
        # The configuration dictionary contains all the arguments
        # needed by the __init__ method. Keras will handle the
        # deserialization of registered custom objects like LearnableC.
        if 'output_dim' in config: 
            config.pop('output_dim') 
            
        return cls(**config)