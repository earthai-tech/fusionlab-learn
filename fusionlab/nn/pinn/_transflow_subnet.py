# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations
from numbers import Integral, Real 
from typing import Optional, Union, Dict, List, Tuple

from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.docs import DocstringComponents, _halnet_core_params
from ...compat.sklearn import validate_params, Interval, StrOptions 
from ...utils.deps_utils import ensure_pkg 
from ...utils.generic_utils import rename_dict_keys 
from ...params import (
    LearnableK, LearnableSs, LearnableQ, LearnableC,
    FixedC, DisabledC
)

from .. import KERAS_BACKEND, KERAS_DEPS, dependency_message 
from .._base_attentive import BaseAttentive


if KERAS_BACKEND:
    from .._tensor_validation import check_inputs, validate_model_inputs 
    from .op import process_pinn_inputs
    from .utils import process_pde_modes, extract_txy_in, _get_coords 
    from ..comp_utils import resolve_gw_coeffs, normalize_C_descriptor
    
LSTM = KERAS_DEPS.LSTM
Dense = KERAS_DEPS.Dense
LayerNormalization = KERAS_DEPS.LayerNormalization 
Sequential =KERAS_DEPS.Sequential
InputLayer =KERAS_DEPS.InputLayer
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

register_keras_serializable = KERAS_DEPS.register_keras_serializable
deserialize_keras_object= KERAS_DEPS.deserialize_keras_object

tf_autograph=KERAS_DEPS.autograph
tf_autograph.set_verbosity(0)
  

DEP_MSG = dependency_message('nn.pinn.models') 
logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_halnet_core_params)
)


__all__ = ["TransFlowSubsNet"]

@register_keras_serializable(
    'fusionlab.nn.pinn', name="TransFlowSubsNet") 
class TransFlowSubsNet(BaseAttentive):
    @validate_params({
        'output_subsidence_dim': [Interval(Integral,1, None, closed="left")], 
        'output_gwl_dim': [Interval(Integral,1, None, closed="left"),], 
        "pde_mode": [
            StrOptions({'consolidation', 'gw_flow', 'both', 'none'}), 
            'array-like', None 
        ],
        "pinn_coefficient_C": [
            str, Real, None, StrOptions({"learnable", "fixed"}),
            LearnableC, FixedC, DisabledC
        ], 
        "K": [Real, None, StrOptions({"learnable", "fixed"}),LearnableK], 
        "Ss": [Real, None, StrOptions({"learnable", "fixed"}),LearnableSs], 
        "Q": [Real, None, StrOptions({"learnable", "fixed"}),LearnableQ], 
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
        pde_mode: Union[str, List[str]] = 'both',
        K: Union[str, float, LearnableK
                 ] = LearnableK(initial_value=1e-4), 
        Ss: Union[float, LearnableSs, str
                  ] = LearnableSs(initial_value =1e-5), 
        Q: Union[float, LearnableQ, str
                 ] = LearnableQ(initial_value =0.0), 
        pinn_coefficient_C: Union[
            LearnableC, FixedC, DisabledC, str, float, None
        ] = LearnableC(initial_value=0.01),
        gw_flow_coeffs: Optional[Dict[str, Union[str, float, None]]] = None,
        use_vsn: bool = True,
        vsn_units: Optional[int] = None,
        mode: Optional[str]=None, 
        objective: Optional[str]=None, 
        attention_levels:Optional[Union[str, List[str]]]=None, 
        architecture_config: Optional[Dict] = None,
        name: str = "TransFlowSubsNet",
        **kwargs
    ):
        
        # The total output dimension for the base data-driven model
        # is the sum of the subsidence and GWL dimensions.
        if 'output_dim' in kwargs: 
            kwargs.pop ('output_dim') # delegate it from combination 
            
        self._combined_output_dim = (
            output_subsidence_dim + output_gwl_dim
        )
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
            architecture_config=architecture_config, 
            name=name,
            **kwargs
        )
        
        self.output_subsidence_dim = output_subsidence_dim
        self.output_gwl_dim = output_gwl_dim
        self.pde_modes_active = process_pde_modes(pde_mode)
        
        self.pinn_coefficient_C =pinn_coefficient_C 
        self.pinn_coefficient_C_config = normalize_C_descriptor(
            pinn_coefficient_C
        )
  
        self.gw_flow_coeffs = gw_flow_coeffs 
        K, Ss, Q = resolve_gw_coeffs(
            gw_flow_coeffs=self.gw_flow_coeffs,
            K=K, Ss=Ss, Q=Q, 
        )

        logger.info(f"Initialized with K={K}, Ss={Ss}, Q={Q}")
        
        self.K_config = K
        self.Ss_config = Ss
        self.Q_config = Q
        
        self._init_coordinate_corrections()
        
        self._build_pinn_components()
        
    def _init_coordinate_corrections(
        self,
        gwl_units: Union [int,  None] = None,
        subs_units: Union [int, None] = None,
        hidden: Tuple[int, int] = (32, 16),
        act: str = "gelu",
    ) -> None:
        r"""
        Build two compact MLPs that transform coordinates into
        additive corrections for the model's head and subsidence
        predictions.
    
        Parameters
        ----------
        gwl_units : int | None, default ``self.output_gwl_dim``  
            Output width of the groundwater‐head branch.  Pass *None* to
            reuse ``self.output_gwl_dim``.
    
        subs_units : int | None, default ``self.output_subsidence_dim``  
            Output width of the subsidence branch.  Pass *None* to reuse
            ``self.output_subsidence_dim``.
    
        hidden : tuple[int, int], default ``(32, 16)``  
            Widths of the two hidden dense layers *shared by both* MLPs.
    
        act : str, default ``"gelu"``  
            Activation applied after each hidden layer.
    
        Notes
        -----
        Let the prediction network output  
    
        .. math::
           h^{\text{net}}(t,x,y),\qquad s^{\text{net}}(t,x,y).
    
        This helper adds two learnable corrections
    
        .. math::
           \Delta h = f_{\theta_h}(t,x,y), \qquad
           \Delta s = f_{\theta_s}(t,x,y),
    
        where :math:`f_{\theta_*}` are shallow MLPs.
        The final quantities used in the PDE terms become  
    
        .. math::
           h = h^{\text{net}} + \Delta h, \qquad
           s = s^{\text{net}} + \Delta s.
    
        The two branches are exposed on the instance as
        ``self.coord_mlp`` and ``self.subs_coord_mlp`` and are trained
        jointly with the rest of the network.
    
        The models are stored on the instance as  
        ``self.coord_mlp`` and ``self.subs_coord_mlp``.  They are **not**
        compiled; they inherit the parent model’s optimizer and are trained
        end-to-end inside :py:meth:`train_step`.
        """
        # fallback to instance-level defaults
        gwl_units = gwl_units or self.output_gwl_dim
        subs_units = subs_units or self.output_subsidence_dim
    
        def _branch(out_units: int, name: str) -> Sequential:
            return Sequential(
                [
                    InputLayer(input_shape=(None, 3)),
                    Dense(hidden[0], activation=act),
                    Dense(hidden[1], activation=act),
                    Dense(out_units),
                ],
                name=name,
            )
    
        # Ground-water head and subsidence correction heads
        self.coord_mlp = _branch(gwl_units, "coord_mlp")
        self.subs_coord_mlp = _branch(subs_units, "subs_coord_mlp")

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
            # Physics disabled => C internally 1.0 
            # but not used if lambda_cons==0 in compile
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
    
        # Create K, Ss, and Q using self.add_weight ---
    
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
            
    @tf_autograph.experimental.do_not_convert
    def call(
        self, inputs: Dict[str, Optional[Tensor]],
        training: bool = False
    ):
        r"""
        Single forward sweep mixing data and physics paths.
    
        The routine
    
        1. extracts :tmath:`t,x,y` and covariate tensors from *inputs*;
        2. runs sanity checks on dimensionality;
        3. feeds the validated features through the inherited
           encoder–decoder stack; and
        4. splits the decoder output into mean and final
           predictions that will later enter the data‐ and
           PDE‐loss terms.
    
        Returns
        -------
        dict
            ``{"subs_pred": s, "gwl_pred": h, 
            "subs_pred_mean": \bar s, "gwl_pred_mean": \bar h}``
            with shapes  
    
            .. math::
               s,\;h &\in \mathbb R^{B\times H\times d},\\
               \bar s,\;\bar h &\in \mathbb R^{B\times H\times d}.
    
            Here *B* is the batch size, *H* the forecast horizon and
            *d* each target's width.
    
        Notes
        -----
        * All coordinate tensors are **not** differentiated here;
          their gradients are taken in :py:meth:`train_step`.
        * The method remains side‐effect free: no weights are updated,
          no losses are added.  It purely produces tensors needed by the
          custom training loop that follows.
    
        """
        # --- 1. Unpack and Validate All Inputs ---
        # The `process_pinn_inputs` helper unpacks the input dict and
        # isolates the coordinate tensors for later use. It is assumed
        # that the `inputs` dict contains all necessary keys.
        logger.debug("TransFlowSubsNet call: Unpacking and validating inputs.")
        (t, x, y, static_features,
         dynamic_features, future_features) = process_pinn_inputs(
             inputs, mode='auto'
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
            verbose= 0 
        )
        
        # Validate future_p shape based on mode 
        if self._mode == 'tft_like':
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
 
        logger.debug("Computing PDE residuals from mean predictions.")
        
        # --- 5. Return All Predictions ---
        # The call method now returns all necessary prediction tensors.
        # The train_step will be responsible for using these to compute
        # the final composite loss.
        # Return All Components for the Loss Function ---
        return {
            "subs_pred": s_pred_final,
            "gwl_pred": gwl_pred_final,
            "subs_pred_mean": s_pred_mean,
            "gwl_pred_mean": gwl_pred_mean,
        }

    def compute_physics_loss(
        self, inputs: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes the physics-based loss terms for both consolidation
        and groundwater flow.
        """
        # This method uses its own tape to calculate the gradients
        # required for the PDE residuals.
        with GradientTape(persistent=True) as tape:
            # Watch the coordinate tensor and the model's predictions.
            coords = inputs['coords']
            tape.watch(coords)
            
            # Re-run the forward pass *within this tape's context*
            # to get predictions that are differentiable wrt coords.
            predictions = self(inputs, training=True)
            s_pred_mean = predictions['subs_pred_mean']
            h_pred_mean = predictions['gwl_pred_mean']
            
            # Explicitly watch the prediction tensors too
            tape.watch(s_pred_mean)
            tape.watch(h_pred_mean)

            # Unpack coordinates for differentiation
            t, x, y = coords[..., 0:1], coords[..., 1:2], coords[..., 2:3]

            # --- First-Order Derivatives ---
            ds_dt = tape.gradient(s_pred_mean, t)
            dh_dt = tape.gradient(h_pred_mean, t)
            dh_dx = tape.gradient(h_pred_mean, x)
            dh_dy = tape.gradient(h_pred_mean, y)

        # --- Second-Order Derivatives ---
        d2h_dx2 = tape.gradient(dh_dx, x)
        d2h_dy2 = tape.gradient(dh_dy, y)
        
        # Clean up the persistent tape
        del tape

        # Validate gradients
        if any(g is None for g in [ds_dt, dh_dt, d2h_dx2, d2h_dy2]):
             raise ValueError("Failed to compute one or more PDE gradients.")
             
        # Assemble residuals using stateless helpers
        cons_res = self._compute_consolidation_residual(ds_dt, d2h_dx2, d2h_dy2)
        gw_res = self._compute_gw_flow_residual(dh_dt, d2h_dx2, d2h_dy2)
        
        # Calculate the loss for each residual
        loss_cons = tf_reduce_mean(tf_square(cons_res))
        loss_gw = tf_reduce_mean(tf_square(gw_res))
        
        return loss_cons, loss_gw

    def train_step(self, data):
        """
        One optimization step uniting data and physics terms.
    
        Uses a single :class:`tf.GradientTape` to obtain all
        first- and second-order coordinate derivatives required
        by the groundwater-flow and consolidation PDEs.
    
        Data loss: any Keras loss supplied in :py:meth:`compile`.
    
        PDE losses:
        .. math::
           S_s \,\partial_t h
           \;-\; K (\partial_{xx} h + \partial_{yy} h) - Q &= 0 \\
           \partial_t s
           \;-\; C (\partial_{xx} h + \partial_{yy} h) &= 0
    
        The final objective is
        :math:`\mathcal L = L_\text{data}
        + \lambda_c L_\text{cons} + \lambda_g L_\text{gw}`.
    
        On return the method updates built-in metrics and
        provides a dict with total, data and PDE losses.
        """

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
        with GradientTape(persistent=True) as tape:
            # The tape must watch the input coordinates to compute
            # gradients of the model's output with respect to them.
            coords = _get_coords(inputs)
            # coords = inputs['coords']
            t, x, y = extract_txy_in(coords)
            tape.watch([t, x, y,])
 
            # --- FORWARD PASS & DATA LOSS ---
            outputs = self(inputs, training=True)
            
            # Same t,x,y that the tape watches → gradients exist
            coords_flat = tf_concat([t, x, y], axis=-1)          # [B,T,3]
            mlp_corr = self.coord_mlp(coords_flat, training=True)
            s_corr = self.subs_coord_mlp(coords_flat, training=True)
            
            # inject the correction into the head that feeds the PDE
            h_pred_mean = outputs["gwl_pred_mean"] + mlp_corr
            s_pred_mean  = outputs["subs_pred_mean"] + s_corr
            # save a copy for the data loss dictionary
            outputs["gwl_pred_mean"] = h_pred_mean
            outputs["subs_pred_mean"] = s_pred_mean
            
            y_pred_for_loss = {
                'subs_pred': outputs['subs_pred'],
                'gwl_pred': outputs['gwl_pred']
            }
            
            data_loss = self.compiled_loss(
                y_true=targets, y_pred=y_pred_for_loss,
                regularization_losses=self.losses
            )
    
            # --- PDE RESIDUAL CALCULATION ---
            s_pred_mean = outputs['subs_pred_mean']
            # h_pred_mean = outputs['gwl_pred_mean']
            
            tape.watch([s_pred_mean , h_pred_mean]) 
            # --- First-order derivatives ---
            ds_dt = tape.gradient(s_pred_mean, t)
            dh_dt = tape.gradient(h_pred_mean, t)
            dh_dx = tape.gradient(h_pred_mean, x)
            dh_dy = tape.gradient(h_pred_mean, y)
    
            # --- Second-order derivatives ---
            d2h_dx2 = tape.gradient(dh_dx, x)
            d2h_dy2 = tape.gradient(dh_dy, y)
    
            # Validate that all necessary gradients were computed.
            if any(g is None for g in [ds_dt, dh_dt, d2h_dx2, d2h_dy2]):
                raise ValueError(
                    "One or more PDE gradients are missing; "
                    "check t, x, y in the forward graph and "
                    "ensure they influence the model's predictions."
                )
    
            # Assemble residuals using the stateless helpers.
            cons_res = self._compute_consolidation_residual(
                ds_dt, d2h_dx2, d2h_dy2)
            gw_res = self._compute_gw_flow_residual(
                dh_dt, d2h_dx2, d2h_dy2)
        
            # --- COMPOSITE LOSS ---
            loss_cons = tf_reduce_mean(tf_square(cons_res))
            loss_gw = tf_reduce_mean(tf_square(gw_res))
            # physics_loss = (+ self.lambda_cons * loss_cons
            #               + self.lambda_gw * loss_gw)
            total_loss = (
                        data_loss 
                        + self.lambda_cons * loss_cons 
                        + self.lambda_gw * loss_gw
                    )
                        
        # --- APPLY GRADIENTS ---
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        del tape                              # free memory
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
    
        # --- METRICS & RETURN ---
        self.compiled_metrics.update_state(targets, y_pred_for_loss)
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "total_loss": total_loss,
            "data_loss": data_loss,
            "physics_loss": (
                self.lambda_cons * loss_cons + self.lambda_gw * loss_gw),    
            "consolidation_loss": loss_cons,
            "gw_flow_loss": loss_gw,
        })
        return results
    
    def _compute_consolidation_residual(
            self, ds_dt, d2h_dx2, d2h_dy2):
        """
        Residual of the consolidation balance.
    
        .. math::
           R_c = \partial_t s - C(\partial_{xx} h + \partial_{yy} h)
        """
        if 'consolidation' not in self.pde_modes_active:
            return tf_zeros_like(ds_dt)
        return ds_dt - self.C * (d2h_dx2 + d2h_dy2)
    
    def _compute_gw_flow_residual(
            self, dh_dt, d2h_dx2, d2h_dy2):
        """
        Residual of the transient groundwater flow.
    
        .. math::
           R_g = S_s \,\partial_t h
           - K(\partial_{xx} h + \partial_{yy} h) - Q
        """
        if 'gw_flow' not in self.pde_modes_active:
            return tf_zeros_like(dh_dt)
        return (self.Ss * dh_dt) - self.K * (d2h_dx2 + d2h_dy2) - self.Q

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
            "pinn_coefficient_C": self.pinn_coefficient_C ,
            "gw_flow_coeffs": self.gw_flow_coeffs, 
            "K": self.K_config,
            "Ss": self.Ss_config,
            "Q": self.Q_config,
        }
        base_config.update(pinn_config)
        
        return base_config

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
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
            
        # Re-create model and nested learnable parameters.
        if custom_objects is None:
            # ensure all helper classes are available to the deserializer
            custom_objects = {
                "LearnableK": LearnableK,
                "LearnableSs": LearnableSs,
                "LearnableQ": LearnableQ,
                "LearnableC": LearnableC,
                "FixedC":     FixedC,
                "DisabledC":  DisabledC,
            }
    
        # keys that may be nested serialisable dicts
        for key in ("K", "Ss", "Q", "pinn_coefficient_C"):
            obj = config.get(key)
            if isinstance(obj, dict) and "class_name" in obj:
                # turn the JSON blob back into a live object
                config[key] = deserialize_keras_object(obj, custom_objects)
    
        return cls(**config)

# ------------------------------------ docstring-------------------------------
TransFlowSubsNet.__doc__ = r"""
Transient Ground-Water–Driven Subsidence Network

TransFlowSubsNet fuses deep-learning encoder–decoder with two
physics losses so that the network **learns** a forecast **and**
honours the governing PDEs at once.

*   **Consolidation** loss forces surface settlement :math:`s`
    to balance the Laplacian of hydraulic head :math:`h`.
*   **Transient ground-water flow** loss constrains head
    to obey the diffusivity equation with source/sink term.

Both terms vanish when *pde_mode* switches them off.

See :ref:`User Guide <user_guide_transflowsubsnet>` for a walkthrough.

Parameters
----------
{params.base.static_input_dim}
{params.base.dynamic_input_dim}
{params.base.future_input_dim}

output_subsidence_dim : int, default 1  
    How many subsidence series are produced at each horizon
    step.  A multi-well scenario with *n* Digital Leveling
    benchmarks would use ``n``.

output_gwl_dim : int, default 1  
    How many head series are produced.  Use >1 for multi-aquifer
    or multi-well settings.

forecast_horizon : int, default 1  
    Horizon length :math:`H`.  The decoder emits :math:`H`
    steps; the physics terms are evaluated for every emitted
    step.

quantiles : list[float] | None, default None  
    Optional list of quantile levels; enables the
    Quantile-Distribution head.

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
{params.base.use_batch_norm}
{params.base.use_vsn}
{params.base.vsn_units}

pde_mode : {{'consolidation', 'gw_flow', 'both', 'none'}}, \
default ``'both'``  
    Select which PDE residuals participate in the loss:  
    ┌─────────────────┬────────────────────────────────────────┐  
    │ 'consolidation' │ only :math:`s`–balance term            │  
    │ 'gw_flow'       │ only flow equation for :math:`h`       │  
    │ 'both'          │ both residuals (recommended)           │  
    │ 'none'          │ pure data-driven; behaves like HAL-Net │  
    └─────────────────┴────────────────────────────────────────┘

K, Ss, Q : float | str | Learnable*, defaults 1e-4, 1e-5, 0.  
    Hydraulic conductivity :math:`K`, specific storage
    :math:`S_s`, and volumetric source/sink :math:`Q`.  
    Accepted forms:  
    * **float / int** → fixed numeric.  
    * ``'learnable'`` → wrap into the corresponding
      :class:`LearnableK` / *Ss* / *Q*.  Initial seed is taken
      from the numeric value given *in the same call* or falls
      back to 1e-4 / 1e-5 / 0.  
    * ``'fixed'`` → force numeric even if
      *param_status='learnable'* in
      :func:`resolve_gw_coeffs`.  
    * **Learnable* instance** → forwarded unchanged.

pinn_coefficient_C : float | str | LearnableC | FixedC | DisabledC, \
default ``LearnableC(0.01)``  
    Coefficient in the consolidation PDE:  

    .. math:: \partial_t s - C\,\nabla^2 h = 0  

    * ``float`` – fixed.  
    * ``'learnable'`` or :class:`LearnableC` – optimised in log-space
      to keep :math:`C>0`.  
    * :class:`DisabledC` – disables consolidation regardless of
      *pde_mode*.

gw_flow_coeffs : dict | None, default None  
    Convenience container overriding *K/Ss/Q* in one go, e.g. ::

        gw_flow_coeffs = {{'K': 'learnable',
                           'Ss': 1e-6,
                           'Q': 'fixed'}}

    Dict entries win over the individual keyword arguments.

mode : {{'pihal_like', 'tft_like'}}, default ``None``  
    Routing for *future_features*:  
    * **pihal_like** – decoder gets all :math:`H` rows, encoder none.  
    * **tft_like**  – first *max_window_size* rows to encoder,  
      next :math:`H` rows to decoder, matching the original
      Temporal Fusion Transformer.  ``None`` inherits
      BaseAttentive default ('tft_like').

objective : {{'hybrid', 'transformer'}}, default ``'hybrid'``  
    Selects the backbone architecture that processes dynamic-past  
    and (optionally) known-future covariates before the decoding stage.  

    * ``'hybrid'`` – **Multi-scale LSTM -> Transformer**.  
      The encoder first extracts multi-resolution temporal features  
      with a stack of LSTMs (one per *scale*), then refines these  
      features with hierarchical/cross attention blocks.  
      This configuration balances the strong sequence-memory capability  
      of recurrent networks with the global-context modelling power of  
      Transformers and is recommended for most tabular time-series data.  

    * ``'transformer'`` – **Pure Transformer**.  
      Bypasses the LSTM stack and feeds the embeddings directly into the  
      attention encoder, resulting in a lightweight, fully self-attention  
      model.  Choose this if your data exhibit long-range dependencies  
      for which an LSTM adds little benefit, or when you need faster  
      training/inference at the cost of some short-term pattern capture.  

    In future release: 
        
    Shortcut for common loss presets.  Should be recognised:  
    * ``'nse'`` – Nash–Sutcliffe model-efficiency score.  
    * ``'rmse'`` – root-mean-square error.  
    When *None* we will supply losses via :py:meth:`compile`.

attention_levels : str | list[str] | None  
    Which hierarchical attention outputs are returned when the
    model is called with ``training=False``.  Use ``'all'`` or a
    subset such as ``['scale', 'cross']`` for interpretability.

name : str, default ``"TransFlowSubsNet"``  
    Model scope as registered in Keras.

**kwargs  
    Forwarded verbatim to :class:`tf.keras.Model`.

Notes
-----
Physics loss is added **outside** the Keras loss container inside
``train_step``; compile with ``lambda_cons`` and ``lambda_gw`` to
scale them.  When any parameter is *learnable* its
:pyattr:`tf.Variable` automatically appears in ``model.trainable_variables``.

See Also
--------
fusionlab.nn.models.HALNet
    Purely data-driven encoder–decoder (no physics terms).

fusionlab.nn.pinn.models.PIHALNet
    Physics-informed HAL-Net that couples consolidation PDEs 
    and adds an anomaly module.

fusionlab.nn.pinn.models.PiTGWFlow
    Stand-alone PINN that solves 2-D / 3-D transient groundwater-flow
    equations without subsidence coupling.

Examples
--------
>>> model = TransFlowSubsNet(
...     static_input_dim=3, dynamic_input_dim=8, future_input_dim=4,
...     output_subsidence_dim=1, output_gwl_dim=1,
...     K='learnable', Ss=1e-5, Q='fixed',
...     pde_mode='both', scales=[1, 3], multi_scale_agg='concat'
... )
>>> batch = {{
...     "static_features":  tf.zeros([8, 3]),
...     "dynamic_features": tf.zeros([8, 12, 8]),
...     "future_features":  tf.zeros([8, 6, 4]),
...     "coords":           tf.zeros([8, 6, 3]),
... }}
>>> pred = model(batch, training=False)
>>> list(pred)
['subs_pred', 'gwl_pred', 'subs_pred_mean', 'gwl_pred_mean']
""".format (params =_param_docs)


