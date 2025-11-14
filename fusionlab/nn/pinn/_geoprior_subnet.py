# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations
from numbers import Integral, Real 
from typing import Optional, Union, Dict, List, Tuple, Any
from collections.abc import Mapping

import numpy as np 
from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.docs import DocstringComponents, _halnet_core_params
from ...compat.sklearn import validate_params, Interval, StrOptions 
from ...utils.deps_utils import ensure_pkg 
from ...utils.generic_utils import rename_dict_keys 
from ...params import (
    LearnableMV, LearnableKappa, FixedGammaW, FixedHRef,
    LearnableK, LearnableSs, LearnableQ, LearnableC,
    FixedC, DisabledC
)

from .. import KERAS_BACKEND, KERAS_DEPS, dependency_message 
from .._base_attentive import BaseAttentive

if KERAS_BACKEND:
    from .._tensor_validation import check_inputs, validate_model_inputs
    from .._utils import get_tensor_from 
    from .io import ( 
        _maybe_subsample, 
        default_meta_from_model, 
        save_physics_payload, 
        load_physics_payload, 
        gather_physics_payload
    ) 
    from .op import ( 
        process_pinn_inputs, default_scales, scale_residual, positive 
    )
    from .utils import  process_pde_modes, extract_txy_in, _get_coords
    from ..components import ( 
        aggregate_multiscale_on_3d, 
        aggregate_time_window_output 
    )
    
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
Mean =KERAS_DEPS.Mean 
Dataset = KERAS_DEPS.Dataset

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
tf_split = KERAS_DEPS.split 
tf_sqrt = KERAS_DEPS.sqrt 
tf_stack = KERAS_DEPS.stack

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


__all__ = ["GeoPriorSubsNet"]

@register_keras_serializable(
    'fusionlab.nn.pinn', name="GeoPriorSubsNet") 
class GeoPriorSubsNet(BaseAttentive):
    @validate_params({
        'output_subsidence_dim': [Interval(Integral,1, None, closed="left")], 
        'output_gwl_dim': [Interval(Integral,1, None, closed="left"),], 
        "pde_mode": [
            StrOptions({'consolidation', 'gw_flow', 'both', 'none', 'on', 'off'}), 
            'array-like', None 
        ],
        "mv": [LearnableMV, Real],
        "kappa": [LearnableKappa, Real],
        "gamma_w": [FixedGammaW, Real],
        "h_ref": [FixedHRef, Real], 
        "use_effective_h": [bool],
        "hd_factor": [Interval(Real, 0, 1, closed="right")], 
        "kappa_mode": [StrOptions({"bar", "kb"})]
        
    }
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
        mv: Union[LearnableMV, float] = LearnableMV(initial_value=1e-7),
        kappa: Union[LearnableKappa, float] = LearnableKappa(initial_value=1.0),
        gamma_w: Union[FixedGammaW, float] = FixedGammaW(value=9810.0),
        h_ref: Union[FixedHRef, float] = FixedHRef(value=0.0),
        use_effective_h: bool = False,
        hd_factor: float = 1.0 ,  # if Hd = Hd_factor * H
        kappa_mode: str = "bar",   # {"bar", "kb"}  # κ̄ vs κ_b
        use_vsn: bool = True,
        vsn_units: Optional[int] = None,
        mode: Optional[str]=None, 
        objective: Optional[str]=None, 
        attention_levels:Optional[Union[str, List[str]]]=None, 
        architecture_config: Optional[Dict] = None,
        scale_pde_residuals: bool = True,
        scaling_kwargs: Optional[Dict[str, Any]] = None,
        name: str = "GeoPriorSubsNet", 
        **kwargs
    ):
        
        self.output_subsidence_dim = output_subsidence_dim
        self.output_gwl_dim = output_gwl_dim
        self._data_output_dim = (
            self.output_subsidence_dim + self.output_gwl_dim
        )
        
        self.output_K_dim = 1      # K(x,y)
        self.output_Ss_dim = 1     # Ss(x,y)
        self.output_tau_dim = 1    # tau(x,y)
        self._phys_output_dim = (
            self.output_K_dim + self.output_Ss_dim + self.output_tau_dim
        )
        
        if 'output_dim' in kwargs: 
            kwargs.pop ('output_dim') 
            
        super().__init__(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            output_dim=self._data_output_dim, 
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
        
        self.pde_modes_active = process_pde_modes(pde_mode)
        self.scale_pde_residuals = bool(scale_pde_residuals)
        self.scaling_kwargs = dict(scaling_kwargs or {})
    
        # --- Process new scalar physics params ---
        if isinstance(mv, (int, float)):
            mv = LearnableMV(initial_value=float(mv))
        if isinstance(kappa, (int, float)):
            kappa = LearnableKappa(initial_value=float(kappa))
        if isinstance(gamma_w, (int, float)):
            gamma_w = FixedGammaW(value=float(gamma_w))
        if isinstance(h_ref, (int, float)):
            h_ref = FixedHRef(value=float(h_ref))

        self.mv_config = mv
        self.kappa_config = kappa
        self.gamma_w_config = gamma_w
        self.h_ref_config = h_ref
        
        self.use_effective_thickness = use_effective_h
        self.Hd_factor = hd_factor   # if Hd = Hd_factor * H
        self.kappa_mode = kappa_mode  # {"bar", "kb"}  # κ̄ vs κ_b

        
        logger.info(
            f"Initialized GeoPriorSubsNet with scalar physics params:"
            f" mv_trainable={mv.trainable},"
            f" kappa_trainable={kappa.trainable}"
        )
        
        self._init_coordinate_corrections()
        self._build_pinn_components()
        
        
    def _build_attentive_layers(self):
        """
        Overrides BaseAttentive to add the physics prediction head.
        """
        super()._build_attentive_layers()
        
        self.physics_mean_head = Dense(
            self._phys_output_dim, name="physics_mean_head"
        )
        # This attribute will store H_field during the forward pass
        self.H_field: Optional[Tensor] = None
        
        # Add validator means priors 
        self.eps_prior_metric = Mean(name="epsilon_prior")
        self.eps_cons_metric  = Mean(name="epsilon_cons")

    def _init_coordinate_corrections(
        self,
        gwl_units: Union [int,  None] = None,
        subs_units: Union [int, None] = None,
        hidden: Tuple[int, int] = (32, 16),
        act: str = "gelu",
    ) -> None:
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
    
        self.coord_mlp = _branch(gwl_units, "coord_mlp")
        self.subs_coord_mlp = _branch(subs_units, "subs_coord_mlp")
        
        self.K_coord_mlp = _branch(self.output_K_dim, "K_coord_mlp")
        self.Ss_coord_mlp = _branch(self.output_Ss_dim, "Ss_coord_mlp")
        self.tau_coord_mlp = _branch(self.output_tau_dim, "tau_coord_mlp")


    def _build_pinn_components(self):
        r"""
        Instantiates trainable/fixed scalar physical coefficients.
        """
        # Handle Compressibility (mv)
        if isinstance(self.mv_config, LearnableMV):
            self.log_mv = self.add_weight(
                name="log_param_mv", shape=(),
                initializer=Constant(
                    tf_log(self.mv_config.initial_value)
                ),
                trainable=self.mv_config.trainable
            )
            # self.mv = tf_exp(self.log_mv)
        else:
            self._mv_fixed = tf_constant(
                float(self.mv_config.initial_value), dtype=tf_float32
                )
            # self.mv = tf_constant(
            #     float(self.mv_config.initial_value), dtype=tf_float32
            # )

        # Handle Consistency Prior (kappa)
        if isinstance(self.kappa_config, LearnableKappa):
            self.log_kappa = self.add_weight(
                name="log_param_kappa", shape=(),
                initializer=Constant(
                    tf_log(self.kappa_config.initial_value)
                ),
                trainable=self.kappa_config.trainable
            )
            # self.kappa = tf_exp(self.log_kappa)
        else:
            self._kappa_fixed = tf_constant(
                float(self.kappa_config.initial_value), dtype=tf_float32
                )
            # self.kappa = tf_constant(
            #     float(self.kappa_config.initial_value), dtype=tf_float32
            # )
            
        self.gamma_w = tf_constant(
            float(self.gamma_w_config.value), dtype=tf_float32
        )
        self.h_ref = tf_constant(
            float(self.h_ref_config.value), dtype=tf_float32
        )
        
        # Placeholders for the predicted fields.
        self.K_field = None
        self.Ss_field = None
        self.tau_field = None

    def _mv_value(self):
        return tf_exp(self.log_mv) if hasattr(
            self, "log_mv") else self._mv_fixed
    
    def _kappa_value(self):
        return tf_exp(self.log_kappa) if hasattr(
            self, "log_kappa") else self._kappa_fixed

    @tf_autograph.experimental.do_not_convert
    def run_encoder_decoder_core(
        self,
        static_input: Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        coords_input: Tensor, 
        training: bool
    ) -> Tuple[Tensor, Tensor]: 
        """
        Executes the data-driven pipeline, injecting coordinates into
        the decoder path and returning separate 2D data and 3D physics
        features.
        
        This method is an OVERRIDE of BaseAttentive.run_encoder_decoder_core
        to allow `coords_input` (B, H, 3) to be concatenated with the
        decoder's features. This ensures that the physics head,
        which is attached to the decoder's output, has a
        computational path from (t, x, y) to (K, Ss, tau).
        
        Returns
        -------
        data_features_2d : tf.Tensor
            The 2D, time-aggregated features for the data decoder.
            Shape: (B, U)
        phys_features_raw_3d : tf.Tensor
            The 3D, time-distributed raw features *before* the physics head,
            ready for the physics head. Shape: (B, H, U)
        """

        time_steps = tf_shape(dynamic_input)[1]
        
        # 1. Initial Feature Processing (Copied from BaseAttentive)
        static_context, dyn_proc, fut_proc = None, dynamic_input, future_input
        
        if self.architecture_config.get('feature_processing') == 'vsn':
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
                static_context = self.grn_static_non_vsn(
                    processed_static, training=training) 
            if self.dynamic_dense: 
                dyn_proc = self.dynamic_dense(dynamic_input)
            if self.future_dense: 
                fut_proc = self.future_dense(future_input)

        logger.debug(f"Shape after VSN/initial processing: "
                     f"Dynamic={getattr(dyn_proc, 'shape', 'N/A')}, "
                     f"Future={getattr(fut_proc, 'shape', 'N/A')}")
        
        # 2. Encoder Path (Copied from BaseAttentive)
        encoder_input_parts = [dyn_proc]
        if self._mode == 'tft_like' and self.future_input_dim > 0:
            fut_enc_proc = fut_proc[:, :time_steps, :]
            encoder_input_parts.append(fut_enc_proc)
        
        encoder_raw = tf_concat(encoder_input_parts, axis=-1)
        encoder_input = self.encoder_positional_encoding(encoder_raw)

        if self.architecture_config['encoder_type'] == 'hybrid':
            lstm_out = self.multi_scale_lstm(
                encoder_input, training=training 
            )
            encoder_sequences = aggregate_multiscale_on_3d(
                lstm_out, mode='concat')
        else: # transformer
            encoder_sequences = encoder_input
            for mha, norm in self.encoder_self_attention:
                attn_out = mha(encoder_sequences, encoder_sequences)
                encoder_sequences = norm(encoder_sequences + attn_out)
        
        if self.apply_dtw: 
            if self.dynamic_time_window is not None:
                encoder_sequences = self.dynamic_time_window(
                    encoder_sequences, training=training
                )
        
        logger.debug(f"Encoder sequences shape: {encoder_sequences.shape}")
        
        # 3. Decoder Path (MODIFIED)
        if self._mode == 'tft_like' and self.future_input_dim > 0:
            fut_dec_proc = fut_proc[:, time_steps:, :]
        elif self.future_input_dim > 0: # pihal_like
            fut_dec_proc = fut_proc
        else:
            fut_dec_proc = None # No future features
        
        decoder_parts = []
        if static_context is not None:
            static_expanded = tf_expand_dims(static_context, 1)
            static_expanded = tf_tile(
                static_expanded, [1, self.forecast_horizon, 1])
            decoder_parts.append(static_expanded)
        
        if fut_dec_proc is not None:
            future_with_pos = self.decoder_positional_encoding(
                fut_dec_proc)
            decoder_parts.append(future_with_pos)

        # Inject the coordinates tensor (shape B, H, 3) here.
        if coords_input is None:
             raise ValueError("GeoPriorSubsNet.run_encoder_decoder_core "
                              "requires 'coords_input'.")
        decoder_parts.append(coords_input)

        if not decoder_parts:
            batch_size = tf_shape(dynamic_input)[0]
            raw_decoder_input = tf_zeros(
                (batch_size, self.forecast_horizon, self.attention_units))
        else:
            raw_decoder_input = tf_concat(decoder_parts, axis=-1)
            
        projected_decoder_input = self.decoder_input_projection(
            raw_decoder_input)
        logger.debug(f"Projected decoder input shape: "
                     f"{projected_decoder_input.shape}")

        # 4. Attention Fusion & Final Processing (Copied from BaseAttentive)
        # `final_features` is the 3D tensor (B, H, U)
        final_features = self.apply_attention_levels(
            projected_decoder_input, encoder_sequences, 
            training=training, 
        )
       
        logger.debug(f"Shape after final fusion: {final_features.shape}")
    
        # The 3D features (final_features) are the basis for the physics
        phys_features_raw_3d = final_features
        
        # Collapse time to get 2D features for data decoder
        data_features_2d = aggregate_time_window_output(
            final_features, self.final_agg
        )
        
        return data_features_2d, phys_features_raw_3d
        
    @tf_autograph.experimental.do_not_convert
    def call(
        self, inputs: Dict[str, Optional[Tensor]],
        training: bool = False
    ) -> Dict[str, Tensor]:
        r"""
    
        Single forward sweep separating data and physics paths.
        This override now also unpacks and stores the H_field.
        """
        # --- 1. Unpack and Validate All Inputs ---
        (t, x, y, H_field, static_features,
          dynamic_features, future_features) = process_pinn_inputs(
              inputs, mode='auto', model_name ="geoprior"
        )
        
        # *** Create coords tensor for injection ***
        # This tensor (B, H, 3) will be passed to the new core method
        coords_for_decoder = tf_concat([t, x, y], axis=-1) 
        
        
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
             
        # Store H_field for train_step
        self.H_field = H_field
        
        static_p, dynamic_p, future_p = validate_model_inputs(
              inputs=[static_features, dynamic_features, future_features],
              static_input_dim=self.static_input_dim,
              dynamic_input_dim=self.dynamic_input_dim,
              future_covariate_dim=self.future_input_dim,
              mode='strict',
              verbose= 0 
          )

        # --- 2. Run Core Encoder-Decoder ----
        # Pass coords_for_decoder and expect two outputs ***
        data_features_2d, phys_features_raw_3d = self.run_encoder_decoder_core(
            static_input=static_p,
            dynamic_input=dynamic_p,
            future_input=future_p,
            coords_input=coords_for_decoder, # Pass coords here
            training=training
        )
        
        # --- 3. Generate Predictions (Data Path) ---
        # Use the 2D aggregated features for the data decoder
        decoded_data_means = self.multi_decoder(
            data_features_2d, training=training
        )
        final_data_predictions = decoded_data_means
        if self.quantiles is not None:
            final_data_predictions = self.quantile_distribution_modeling(
                decoded_data_means, training=training
            )
            
        # --- 4. Generate Predictions (Physics Path) ---
        # The 3D physics features are now *returned directly* from the core
        # We apply the physics_mean_head to the 3D features
        decoded_physics_means_raw = self.physics_mean_head(
            phys_features_raw_3d, training=training
        )
        
        # --- 5. Return All Components ---
        return {
            "data_final": final_data_predictions,
            "data_mean": decoded_data_means,
            "phys_mean_raw": decoded_physics_means_raw,
        }


    def train_step(self, data):
        r"""

        One optimization step uniting data and new physics terms.
        
        Implements the loss function from the revised manuscript:
        L = L_data + L_gw + L_cons + L_prior + L_smooth
        """

        inputs, targets = data
        if isinstance (targets, dict): 
            targets = rename_dict_keys(
                targets.copy(),
                param_to_rename={
                    "subsidence": "subs_pred", 
                    "gwl": "gwl_pred"
                }
        )
        
        # Get H_field from inputs, it must be present
        H_field_input = get_tensor_from(
            inputs, 'H_field', 'soil_thickness', auto_convert=True
        )
        
        if H_field_input is None:
            raise ValueError(
                "Input dictionary must contain 'H_field' or 'soil_thickness'"
                " for train_step."
            )
        H_field = tf_convert_to_tensor(H_field_input, dtype=tf_float32)

        with GradientTape(persistent=True) as tape:
            # --- 1. Setup Coordinates ---
            coords = _get_coords(inputs)
            t, x, y = extract_txy_in(coords)
            tape.watch([t, x, y,])
 
            # --- 2. FORWARD PASS & DATA LOSS ---
            # This call() will also set `self.H_field` internally
            outputs_dict = self(inputs, training=True)

        
            # Split final (quantile) predictions for data loss
            (s_pred_final, gwl_pred_final) = self.split_data_predictions(
                outputs_dict["data_final"]
            )
            y_pred_for_loss = {
                'subs_pred': s_pred_final,
                'gwl_pred': gwl_pred_final
            }
            data_loss = self.compiled_loss(
                y_true=targets, y_pred=y_pred_for_loss,
                regularization_losses=self.losses
            )
    
            # --- 3. PDE PREPARATION ---
            # Split mean predictions and apply positivity
            (s_pred_mean, gwl_pred_mean,
             K_field_raw, Ss_field_raw, tau_field_raw) = self.split_physics_predictions(
                 outputs_dict
             )
        
            # Apply coordinate-based corrections
            coords_flat = tf_concat([t, x, y], axis=-1)
            # Apply coordinate-based corrections
            mlp_corr = self.coord_mlp(coords_flat, training=True)
            s_corr = self.subs_coord_mlp(coords_flat, training=True)
            
            h_pred_mean_corr = gwl_pred_mean + mlp_corr
            s_pred_mean_corr = s_pred_mean + s_corr
            
            # Apply corrections to K, Ss, tau
            K_corr = self.K_coord_mlp(coords_flat, training=True)
            Ss_corr = self.Ss_coord_mlp(coords_flat, training=True)
            tau_corr = self.tau_coord_mlp(coords_flat, training=True)
            
            # Add correction before positivity constraint
            K_field = positive(K_field_raw + K_corr)
            Ss_field = positive(Ss_field_raw + Ss_corr)
            tau_field = positive(tau_field_raw + tau_corr)

            
            # Watch all tensors needed for differentiation
            tape.watch([
                s_pred_mean_corr, h_pred_mean_corr,
                K_field, Ss_field, tau_field
            ])
            
            # --- 4. CALCULATE DERIVATIVES ---
            
            # Derivatives for R_cons
            ds_dt = tape.gradient(s_pred_mean_corr, t)

            # Derivatives for R_gw = Ss*dh/dt - div(K*grad(h)) - Q
            dh_dt = tape.gradient(h_pred_mean_corr, t)
            dh_dx = tape.gradient(h_pred_mean_corr, x)
            dh_dy = tape.gradient(h_pred_mean_corr, y)
            
            K_dh_dx = K_field * dh_dx
            K_dh_dy = K_field * dh_dy
            
            d_K_dh_dx_dx = tape.gradient(K_dh_dx, x)
            d_K_dh_dy_dy = tape.gradient(K_dh_dy, y)
            
            # Derivatives for L_smooth
            dK_dx = tape.gradient(K_field, x)
            dK_dy = tape.gradient(K_field, y)
            dSs_dx = tape.gradient(Ss_field, x)
            dSs_dy = tape.gradient(Ss_field, y)

            # Check for None gradients (critical)
            derivs = {
                "ds_dt": ds_dt, "dh_dt": dh_dt, "d_K_dh_dx_dx": d_K_dh_dx_dx,
                "d_K_dh_dy_dy": d_K_dh_dy_dy, "dK_dx": dK_dx, "dK_dy": dK_dy,
                "dSs_dx": dSs_dx, "dSs_dy": dSs_dy
            }
            if any(v is None for v in derivs.values()):
                none_keys = [k for k, v in derivs.items() if v is None]
                raise ValueError(
                    f"One or more PDE gradients are None: {none_keys}. "
                    "Check that (t, x, y) influence all 5 model outputs "
                    "(s, h, K, Ss, tau) and that coordinate corrections "
                    "are applied."
                )

            # --- 5. COMPUTE RESIDUALS (NEW PHYSICS) ---
            
            # R_gw = Ss*dh/dt - (div(K*grad(h))) - Q
            # Note: Q=0 in revised manuscript
            gw_res = self._compute_gw_flow_residual(
                dh_dt, d_K_dh_dx_dx, d_K_dh_dy_dy, Ss_field, Q=0.0
            )
            
            # R_cons = ds/dt - (s_eq - s) / tau
            cons_res = self._compute_consolidation_residual(
                ds_dt, s_pred_mean_corr, h_pred_mean_corr, H_field, tau_field
            )
            
            # L_prior = ||log(tau) - log(phys_tau)||^2
            prior_res = self._compute_consistency_prior(
                K_field, Ss_field, tau_field, H_field
            )
            
            # L_smooth = ||grad(K)||^2 + ||grad(Ss)||^2
            smooth_res = self._compute_smoothness_prior(
                dK_dx, dK_dy, dSs_dx, dSs_dy
            )
            
            mv_prior_res = self._compute_mv_prior(Ss_field)     
            loss_mv = tf_reduce_mean(tf_square(mv_prior_res))   
            
            if self._physics_off():
                # Force all physics terms to zero
                cons_res = tf_zeros_like(cons_res)
                gw_res   = tf_zeros_like(gw_res)
                prior_res   = tf_zeros_like(prior_res)
                smooth_res  = tf_zeros_like(smooth_res)
                loss_mv     = tf_zeros_like(loss_mv)
                
            # --- 6. SCALE RESIDUALS ---
            if (not self._physics_off()) and self.scale_pde_residuals:
                scales = self._compute_scales(
                    t, s_pred_mean_corr, h_pred_mean_corr,
                    K_field, Ss_field, Q=0.0
                )
                cons_res = scale_residual(cons_res, scales.get("cons_scale"))
                gw_res   = scale_residual(gw_res,   scales.get("gw_scale"))
                # Priors are often scaled differently or not at all.
                # We will not scale prior_res or smooth_res by default.
    
            # --- 7. COMPOSITE LOSS ---
            loss_cons = tf_reduce_mean(tf_square(cons_res))
            loss_gw = tf_reduce_mean(tf_square(gw_res))
            loss_prior = tf_reduce_mean(tf_square(prior_res))
            loss_smooth = tf_reduce_mean(smooth_res) # Already squared

            total_loss = (
                  data_loss 
                + self.lambda_cons * loss_cons 
                + self.lambda_gw * loss_gw
                + self.lambda_prior * loss_prior
                + self.lambda_smooth * loss_smooth
                + self.lambda_mv * loss_mv  
            )
                        
        # --- 8. APPLY GRADIENTS ---
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        del tape # Free memory
        self.optimizer.apply_gradients(
            self._scale_param_grads(grads, trainable_vars))
            
        # self.optimizer.apply_gradients(zip(grads, trainable_vars))
    
        # --- 9. METRICS & RETURN ---
        self.compiled_metrics.update_state(targets, y_pred_for_loss)
        results = {m.name: m.result() for m in self.metrics}
        
        physics_loss = (
            self.lambda_cons * loss_cons 
            + self.lambda_gw * loss_gw
            + self.lambda_prior * loss_prior
            + self.lambda_smooth * loss_smooth
            + self.lambda_mv * loss_mv
        )
        
        results.update({
            "total_loss": total_loss,
            "data_loss": data_loss,
            "physics_loss": physics_loss,    
            "consolidation_loss": loss_cons,
            "gw_flow_loss": loss_gw,
            "prior_loss": loss_prior,
            "smooth_loss": loss_smooth,
            "mv_prior_loss": loss_mv, 
        })
        return results

    def test_step(self, data):
        """
        Evaluation step that mirrors train_step's supervised mapping.
        Computes data losses/metrics 
        (and optionally logs physics terms without grads).
        """
        inputs, targets = data
        if isinstance(targets, dict):
            targets = rename_dict_keys(
                targets.copy(),
                param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"}
            )
    
        # Forward pass (no training)
        outputs = self(inputs, training=False)
    
        # Split final quantile predictions into the two supervised heads
        s_pred_final, gwl_pred_final = self.split_data_predictions(
            outputs["data_final"]
            )
        y_pred_for_eval = {
            "subs_pred": s_pred_final,   # shape typically (B, H, Q[, ...])
            "gwl_pred":  gwl_pred_final,
        }
    
        # Compute loss + update compiled metrics on
        # the SAME mapping used in training
        loss = self.compiled_loss(
            y_true=targets, y_pred=y_pred_for_eval, 
            regularization_losses=self.losses
        )
        self.compiled_metrics.update_state(targets, y_pred_for_eval)
    
        # results = {m.name: m.result() for m in self.metrics}
        # results.update({"loss": loss})
    
        # (Optional) quick physics diagnostics during eval — no gradients needed
        if not self._physics_off():
            phys = self.evaluate_physics(inputs)  # returns tensors
            self.eps_prior_metric.update_state(phys["epsilon_prior"])
            self.eps_cons_metric.update_state(phys["epsilon_cons"])
        else:
            # push zeros so logs stay stable
            self.eps_prior_metric.update_state(0.0)
            self.eps_cons_metric.update_state(0.0)

        # results.update({k: tf.cast(v, tf.float32) for k, v in phys.items()})
        results = {m.name: m.result() for m in self.metrics}
        
        # Package results
        results.update({
            "loss": loss,
            "epsilon_prior": self.eps_prior_metric.result(),
            "epsilon_cons":  self.eps_cons_metric.result(),
        })
        
    
        return results

    def current_mv(self):     
        return self._mv_value()
    
    def current_kappa(self):  
        return self._kappa_value()
    
    def _compute_mv_prior(self, Ss_field):
        # R_mv = log(Ss) - log(mv*gamma_w)
        # return tf_log(Ss_field) - (tf_log(self.mv) + tf_log(self.gamma_w))
        return tf_log(Ss_field) - (tf_log(self._mv_value()) + tf_log(self.gamma_w))

    def _compute_gw_flow_residual(
        self, dh_dt, d_K_dh_dx_dx, d_K_dh_dy_dy, Ss_field, Q=0.0
    ):
        r"""
        Computes groundwater residual: R_gw = Ss*dh/dt - div(K*grad(h)) - Q
        """
        if 'gw_flow' not in self.pde_modes_active:
            return tf_zeros_like(dh_dt)
        
        div_K_grad_h = d_K_dh_dx_dx + d_K_dh_dy_dy
        storage_term = Ss_field * dh_dt
        
        return storage_term - div_K_grad_h - Q

    def _compute_consolidation_residual(
        self, ds_dt, s_mean, h_mean, H_field, tau_field
    ):
        r"""
        Computes consolidation residual: R_cons = ds/dt - (s_eq - s) / tau
        """
        if 'consolidation' not in self.pde_modes_active:
            return tf_zeros_like(ds_dt)
        
        # s_eq(h) = m_v * gamma_w * Delta_h * H
        # Delta_h = h_ref - h
        delta_h = self.h_ref - h_mean
        # s_eq = self.mv * self.gamma_w * delta_h * H_field
        
        delta_h = self.h_ref - h_mean
        s_eq = self._mv_value() * self.gamma_w * delta_h * H_field
        
        relaxation_term = (s_eq - s_mean) / tau_field
        
        return ds_dt - relaxation_term

    def _compute_consistency_prior(
            self, K_field, Ss_field, tau_field, H_field
        ):
        r"""
        Computes consistency prior: R_prior = log(tau) - log(phys_tau)
        """
        pi_squared = tf_constant(np.pi**2, dtype=tf_float32)
        
        # log(tau_pred)
        log_tau_pred = tf_log(tau_field)
        
        # Add a small epsilon to H_field for numerical stability.
        # This prevents log(0) if H_field was scaled to 0.0.
        # safe_H_field = H_field + 1e-6
        safe_H_eff = ( 
            H_field * self.Hd_factor if self.use_effective_thickness else H_field 
            ) + 1e-6
        
        if self.kappa_mode == "bar":
            # log(tau_phys) = log( (kappa * H^2) / ( (pi^2 * K) / Ss ) )
            # log(tau_phys) = log(kappa) + 2*log(H) - (log(pi^2) + log(K) - log(Ss))
            log_tau_phys = (
                # tf_log(self.kappa) 
                tf_log(self._kappa_value())
                + 2 * tf_log(safe_H_eff)  # 
                - (tf_log(pi_squared) + tf_log(K_field) - tf_log(Ss_field))
            )
        else:  # "kb": kappa is κ_b; incorporate (Hd/H)^2 explicitly
            ratio = (safe_H_eff / (H_field + 1e-6))  # safe ratio
            log_tau_phys = (
                # tf_log(self.kappa) 
                tf_log(self._kappa_value())
                + 2*tf_log(ratio) 
                + 2*tf_log(H_field)
                - (tf_log(pi_squared) + tf_log(K_field) - tf_log(Ss_field))
            )
            
        return log_tau_pred - log_tau_phys

    def get_last_physics_fields(self):
        """
        Returns the most recent physics fields and H used by the model call.
        Shapes: (B, H, 1) each, matching the last forward pass.
        """
        return {
            "tau":  self.tau_field,
            "K":    self.K_field,
            "Ss":   self.Ss_field,
            "H_in": self.H_field,   # raw H passed in inputs
        }
    
    def _tau_phys_from_fields(self, K_field, Ss_field, H_field):
        """
        Physical time scale from fields, matching _compute_consistency_prior.
        Returns tau_phys in linear space (
            no logs), with the same K-mode logic.
        """
        pi_squared = tf_constant(np.pi**2, dtype=tf_float32)
        # same H logic as in _compute_consistency_prior
        safe_H_eff = (
            H_field * self.Hd_factor if self.use_effective_thickness else H_field
        ) + 1e-6
    
        if self.kappa_mode == "bar":
            # τ_phys = (κ * H_eff^2 * Ss) / (π^2 * K)
            tau_phys = (
                self._kappa_value() * (safe_H_eff**2) * Ss_field
                / (pi_squared * K_field)
            )
        else:  # "kb"
            ratio = safe_H_eff / (H_field + 1e-6)
            tau_phys = (
                self._kappa_value() * (ratio**2) * (
                    (H_field + 1e-6) **2) * Ss_field
                / (pi_squared * K_field)
            )
        return tau_phys, safe_H_eff  # return Hd actually used


    def _compute_smoothness_prior(
        self, dK_dx, dK_dy, dSs_dx, dSs_dy
    ):
        r"""
        Computes smoothness prior: R_smooth = ||grad(K)||^2 + ||grad(Ss)||^2
        """
        grad_K_squared = tf_square(dK_dx) + tf_square(dK_dy)
        grad_Ss_squared = tf_square(dSs_dx) + tf_square(dSs_dy)
        
        return grad_K_squared + grad_Ss_squared


    def _compute_scales(self, t, s_mean, h_mean, K_field, Ss_field, Q=0.0):
        r"""
        Build dimensionless scales for PDE residuals using default_scales().
        (Updated to accept K and Ss as fields)
        """
        dt_tensor = None
        if hasattr(t, "shape") and t.shape.rank is not None and t.shape.rank >= 2:
            if t.shape[1] and t.shape[1] > 1:
                dt_tensor = t[:, 1:, :] - t[:, :-1, :]
    
        if dt_tensor is None:
            dt_tensor = tf_zeros_like(s_mean[..., :1]) + 1.0
    
        # --- Pass the predicted K and Ss fields ---
        return default_scales(
            h=h_mean,
            s=s_mean,
            dt=dt_tensor,
            K=K_field,  # <-- Use predicted field
            Ss=Ss_field, # <-- Use predicted field
            Q=Q,         # <-- Use scalar Q (0.0)
            **self.scaling_kwargs
        )
    
    def _evaluate_physics_on_batch(
        self,
        inputs: Dict[str, Optional[Tensor]],
        return_maps: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Core implementation: physics diagnostics on a *single batch*.
        This is basically the old `evaluate_physics` body.
        """
        # --- Validate presence of H_field/soil_thickness and extract coords ---
        H_field_in = get_tensor_from(
            inputs, "H_field", "soil_thickness",
            auto_convert=True,
        )
        if H_field_in is None:
            raise ValueError(
                "evaluate_physics() requires 'H_field' "
                "(or 'soil_thickness') in `inputs`."
            )
        H_field = tf_convert_to_tensor(H_field_in, dtype=tf_float32)
    
        coords = _get_coords(inputs)
        t, x, y = extract_txy_in(coords)
    
        # We must compute model outputs under the tape so AD sees dependency on t.
        with GradientTape() as tape:
            tape.watch(t)
    
            # Forward pass (no training); also stores self.H_field internally.
            outputs = self(inputs, training=False)
    
            # Split means and positive physics fields.
            (
                s_mean,
                h_mean,
                K_field,
                Ss_field,
                tau_field,
            ) = self.split_physics_predictions(outputs)
    
            # Bind s_mean to t so the tape tracks s(t,x,y)
            s_bind = s_mean + 0.0 * t
    
        # AD: ds/dt at the same spatiotemporal points
        ds_dt = tape.gradient(s_bind, t)
        if ds_dt is None:
            raise ValueError(
                "Automatic differentiation returned None. "
                "Ensure (t,x,y) influence the subsidence head "
                "via the coordinate injection path."
            )
    
        # --- Residuals using the model's own helpers (exactly as in training) ---
        # Prior: R_prior = log(tau) - log( (kappa * H^2 Ss) / (π^2 K) )
        R_prior = self._compute_consistency_prior(
            K_field, Ss_field, tau_field, H_field
        )
    
        # Consolidation: R_cons = ∂s/∂t - (s_eq - s)/tau,
        # with s_eq = m_v γ_w (h_ref - h) H
        R_cons = self._compute_consolidation_residual(
            ds_dt, s_mean, h_mean, H_field, tau_field
        )
    
        # --- Unscaled RMS errors (paper definitions) ---
        eps_prior = tf_sqrt(tf_reduce_mean(tf_square(R_prior)))
        eps_cons = tf_sqrt(tf_reduce_mean(tf_square(R_cons)))
    
        out = {"epsilon_prior": eps_prior, "epsilon_cons": eps_cons}
        if return_maps:
            # Also return fields needed for Fig.4 payload
            tau_phys, Hd_eff = self._tau_phys_from_fields(
                K_field, Ss_field, H_field
            )
            out.update(
                {
                    "R_prior": R_prior,
                    "R_cons": R_cons,
                    "K": K_field,
                    "Ss": Ss_field,
                    "H": Hd_eff,  # effective H actually used
                    # numerically safe no-op to get a clean tensor
                    "tau": tf_exp(tf_log(tau_field)),
                    "tau_prior": tau_phys,
                }
            )
        return out
    

    def evaluate_physics(
        self,
        inputs: Union[Dict[str, Optional[Tensor]], "Dataset"],
        return_maps: bool = False,
        max_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        r"""
        Evaluate physics diagnostics.
    
        Parameters
        ----------
        inputs : dict or tf.data.Dataset
            - If a mapping (dict), it is treated as **one batch**, exactly
              as in the original implementation.  This is the path used by
              Keras' ``test_step``.
            - If a ``tf.data.Dataset``, the dataset is iterated batch by
              batch (up to ``max_batches``) and the scalar diagnostics are
              aggregated across batches.
    
        return_maps : bool, default=False
            Whether to also return residual maps and physical fields
            (``R_prior``, ``R_cons``, ``K``, ``Ss``, ``H``, ``tau``,
            ``tau_prior``).  When ``inputs`` is a dataset these maps are
            returned **for the last processed batch only**, to avoid
            storing huge tensors in memory.
    
        max_batches : int or None, default=None
            When ``inputs`` is a dataset, limit the number of batches
            that are processed.  If ``None``, the whole dataset is used.
            This is the main knob to avoid out-of-memory situations on
            very large splits.
    
        batch_size : int or None, default=None
            Convenience option when passing a dict of NumPy arrays:
            if not ``None``, a temporary ``tf.data.Dataset`` is built
            and processed using mini-batches of this size.
    
        Returns
        -------
        dict
            Always contains the scalar diagnostics
    
            ``{"epsilon_prior": ..., "epsilon_cons": ...}``
    
            and, if ``return_maps=True``, also residual/field maps.
    
        Notes
        -----
        * Keras' ``test_step`` still passes a **single-batch dict**,
          so behaviour during training/validation is unchanged.
        * For large evaluation splits you should pass a pre-batched
          dataset, for example::
    
              ds = tf.data.Dataset.from_tensor_slices((X_fore, y_fore_fmt))\
                                   .batch(256)
              phys = model.evaluate_physics(ds, max_batches=10)
    
          which only uses the first 10 batches instead of the full
          Zhongshan split.
        """

        # 1) Dataset path: iterate up to `max_batches` and aggregate scalars
        if isinstance(inputs, Dataset):
            eps_prior_vals = []
            eps_cons_vals = []
            last_maps = None
    
            for i, elem in enumerate(inputs):
                # elem may be (xb, yb) or xb
                xb = elem[0] if isinstance(elem, (tuple, list)) else elem
    
                out_b = self._evaluate_physics_on_batch(
                    xb, return_maps=return_maps
                )
                eps_prior_vals.append(out_b["epsilon_prior"])
                eps_cons_vals.append(out_b["epsilon_cons"])
    
                if return_maps:
                    # keep non-scalar maps only from the last batch
                    last_maps = {
                        k: v
                        for k, v in out_b.items()
                        if k not in ("epsilon_prior", "epsilon_cons")
                    }
    
                if max_batches is not None and (i + 1) >= max_batches:
                    break
    
            if not eps_prior_vals:
                raise ValueError("Empty dataset provided to evaluate_physics.")
    
            # eps_* values are already RMS per batch; we aggregate by mean
            eps_prior = tf_reduce_mean(tf_stack(eps_prior_vals))
            eps_cons = tf_reduce_mean(tf_stack(eps_cons_vals))
    
            out = {"epsilon_prior": eps_prior, "epsilon_cons": eps_cons}
            if return_maps and last_maps is not None:
                out.update(last_maps)
            return out
    
        # 2) Dict-of-NumPy arrays convenience path with `batch_size`
        if isinstance(inputs, Mapping) and batch_size is not None:
            # If at least one value is not a Tensor, assume NumPy-like.
            any_tensor = any(
                isinstance(v, Tensor) for v in inputs.values() if v is not None
            )
            if not any_tensor:
                ds = Dataset.from_tensor_slices(inputs).batch(batch_size)
                return self.evaluate_physics(
                    ds,
                    return_maps=return_maps,
                    max_batches=max_batches,
                )
    
        # 3) Backwards-compatible single-batch behaviour
        # Either a dict of Tensors (from test_step) or a small dict of NumPy arrays
        return self._evaluate_physics_on_batch(
            inputs, return_maps=return_maps
        )

    def export_physics_payload(
        self,
        dataset,
        max_batches=None,
        save_path=None,
        format="npz",
        overwrite=False,
        metadata=None,
        random_subsample=None,
        float_dtype=np.float32,
    ):
        """
        Gather physics payload from `dataset` and optionally persist to disk.
        """
        payload = gather_physics_payload(
            self, dataset, max_batches=max_batches, float_dtype=float_dtype
        )
        if random_subsample is not None:
            payload = _maybe_subsample(payload, random_subsample)
    
        if save_path is not None:
            meta = default_meta_from_model(self)
            if metadata:
                meta.update(metadata)
            save_physics_payload(
                payload, meta, save_path, format=format, overwrite=overwrite
            )
        return payload
    
    @staticmethod
    def load_physics_payload(path):
        """Load a previously saved physics payload + metadata."""
        return load_physics_payload(path)

    def split_data_predictions(
        self,
        data_tensor: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Splits the combined data tensor (s, h) into its components.
        (From Step 2)
        """
        s_pred = data_tensor[..., :self.output_subsidence_dim]
        gwl_pred = data_tensor[..., self.output_subsidence_dim:]
        
        return s_pred, gwl_pred

    def split_physics_predictions(
        self,
        outputs_dict: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Splits all mean predictions and applies positivity constraints.
        (From Step 2)
        """
        data_means_tensor = outputs_dict["data_mean"]
        phys_means_raw_tensor = outputs_dict["phys_mean_raw"]
        
        s_pred_mean = data_means_tensor[..., :self.output_subsidence_dim]
        gwl_pred_mean = data_means_tensor[..., self.output_subsidence_dim:]
        
        start_idx = 0
        end_idx = self.output_K_dim
        K_raw = phys_means_raw_tensor[..., start_idx:end_idx]
        
        start_idx = end_idx
        end_idx += self.output_Ss_dim
        Ss_raw = phys_means_raw_tensor[..., start_idx:end_idx]
        
        start_idx = end_idx
        tau_raw = phys_means_raw_tensor[..., start_idx:]
        
        # Apply positivity constraint
        K_field = positive(K_raw)
        Ss_field = positive(Ss_raw)
        tau_field = positive(tau_raw)
        
        # Store the fields from the latest call on the model instance
        self.K_field = K_field
        self.Ss_field = Ss_field
        self.tau_field = tau_field
        
        return s_pred_mean, gwl_pred_mean, K_field, Ss_field, tau_field

    def _scale_param_grads(self, grads, trainable_vars):
        """Scale grads for special scalar params (m_v, kappa)."""
        scaled = []
        mv_var    = getattr(self, "log_mv", None)
        kappa_var = getattr(self, "log_kappa", None)
    
        for g, v in zip(grads, trainable_vars):
            if g is None:
                continue
            if (mv_var is not None) and (v is mv_var):
                g = g * self._mv_lr_mult
            elif (kappa_var is not None) and (v is kappa_var):
                g = g * self._kappa_lr_mult
            scaled.append((g, v))
        return scaled

    def _physics_off(self) -> bool:
        return isinstance(self.pde_modes_active, (list, tuple)) \
               and ('none' in self.pde_modes_active)


    @property
    def mv_lr_mult(self) -> float: 
        return self._mv_lr_mult
    
    @property
    def kappa_lr_mult(self) -> float:
        return self._kappa_lr_mult

    def compile(
        self,
        lambda_cons: float = 1.0,
        lambda_gw: float = 1.0,
        lambda_prior: float = 1.0,
        lambda_smooth: float = 1.0,
        lambda_mv: float = 0.0,
        mv_lr_mult: float = 1.0,
        kappa_lr_mult: float = 1.0,
        **kwargs
    ):
        """
        Compile the model and set physics/data loss weights.
    
        Parameters
        ----------
        lambda_cons : float, default=1.0
            Weight for consolidation residual L_cons.
        lambda_gw : float, default=1.0
            Weight for groundwater-flow residual L_gw.
        lambda_prior : float, default=0.5
            Weight for geomechanical consistency prior L_prior
            that links (tau, K, S_s, H).
        lambda_smooth : float, default=1.0
            Weight for smoothness prior on (K, S_s).
        lambda_mv : float, default=0.0
            Weight for the storage identity penalty
            R_mv = log(S_s) - log(m_v * gamma_w).
            Set > 0 to give m_v a direct gradient.
        mv_lr_mult : float, default=1.0
            Learning-rate multiplier applied only to the gradient of
            the scalar parameter log(m_v).
        kappa_lr_mult : float, default=1.0
            Learning-rate multiplier applied only to the gradient of
            the scalar parameter log(κ).
    
        **kwargs
            Forwarded to `tf.keras.Model.compile`.
        """

        super().compile(**kwargs)
        
        self.lambda_cons = lambda_cons
        self.lambda_gw = lambda_gw
        
        if self._physics_off(): 
            self.lambda_prior=self.lambda_smooth=self.lambda_mv=0.0
        else:
            self.lambda_prior = lambda_prior
            self.lambda_smooth = lambda_smooth
            self.lambda_mv = lambda_mv
            
        self._mv_lr_mult = float(mv_lr_mult)
        self._kappa_lr_mult = float(kappa_lr_mult)
        

    def get_config(self) -> dict:
        """Returns the full configuration of the model."""
        base_config = super().get_config()
        
        pinn_config = {
            "output_subsidence_dim": self.output_subsidence_dim,
            "output_gwl_dim": self.output_gwl_dim,
            "pde_mode": self.pde_modes_active,
            "mv": self.mv_config,
            "kappa": self.kappa_config,
            "gamma_w": self.gamma_w_config,
            "h_ref": self.h_ref_config,
            "scale_pde_residuals": self.scale_pde_residuals,
            "scaling_kwargs": self.scaling_kwargs,
            "model_version": "3.0-GeoPrior",
        }
        base_config.update(pinn_config)
        base_config['output_dim'] = self._data_output_dim
        
        return base_config

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        """Reconstructs a model instance from its configuration."""
        if custom_objects is None:
            custom_objects = {}
        
        custom_objects.update({
            "LearnableMV": LearnableMV,
            "LearnableKappa": LearnableKappa,
            "FixedGammaW": FixedGammaW,
            "FixedHRef": FixedHRef,
            "LearnableK": LearnableK, "LearnableSs": LearnableSs,
            "LearnableQ": LearnableQ, "LearnableC": LearnableC,
            "FixedC": FixedC, "DisabledC": DisabledC,
        })

        for key in ("mv", "kappa", "gamma_w", "h_ref"):
            obj = config.get(key)
            if isinstance(obj, dict) and "class_name" in obj:
                config[key] = deserialize_keras_object(
                    obj, custom_objects=custom_objects
                )
        
        config.pop("K", None)
        config.pop("Ss", None)
        config.pop("Q", None)
        config.pop("pinn_coefficient_C", None)
        config.pop("gw_flow_coeffs", None)
        config.pop('output_dim', None) 
        config.pop('model_version', None)
    
        return cls(**config)

GeoPriorSubsNet.__doc__ = r""" 
Geomechanical Prior-Informed Subsidence Network (GeoPriorSubsNet)

This model fuses a deep encoder–decoder (from ``BaseAttentive``) with
a physically-sound geomechanical model, addressing key limitations of
simpler PINNs by learning a physically consistent system.

**Model Outputs:**

The network is multi-headed, predicting five distinct fields:
1.  **Subsidence :math:`s(x,y,t)`** (Data, with quantiles)
2.  **Hydraulic Head :math:`h(x,y,t)`** (Data, with quantiles)
3.  **Hydraulic Conductivity :math:`K(x,y)`** (Physics field, positive)
4.  **Specific Storage :math:`S_s(x,y)`** (Physics field, positive)
5.  **Relaxation Time :math:`\tau(x,y)` (Physics field, positive)

**Physics & Loss Function:**

The model is trained by minimizing a composite loss function that
ensures physical consistency, as defined in the revised manuscript:

.. math::
    
    \mathcal{{L}} = \mathcal{{L}}_{{\text{{data}}}} +
                  \lambda_{{gw}} \mathcal{{L}}_{{gw}} +
                  \lambda_{{cons}} \mathcal{{L}}_{{cons}} +
                  \lambda_{{prior}} \mathcal{{L}}_{{prior}} +
                  \lambda_{{smooth}} \mathcal{{L}}_{{smooth}}

Where:

* :math:`\mathcal{{L}}_{{\text{{data}}}}` (Data Fidelity):
  Pinball loss on the predicted quantiles of :math:`s` and :math:`h`.

* :math:`\mathcal{{L}}_{{gw}}` (Groundwater Flow):
  Residual of the 2D transient flow equation. Assumes :math:`Q=0`.
  
  .. math::
      R_{{gw}} = S_s \frac{{\partial h}}{{\partial t}} -
               \nabla \cdot (K \nabla h)

* :math:`\mathcal{{L}}_{{cons}}` (Reduced-Order Consolidation):
  Residual of the 1D relaxation model.
  
  .. math::
      R_{{cons}} = \frac{{\partial s}}{{\partial t}} -
                 \frac{{s_{{eq}}(h) - s}}{{\tau}}
                 
  where:
  .. math::
      s_{{eq}}(h) = m_v \gamma_w (h_{{ref}} - h) H

* :math:`\mathcal{{L}}_{{prior}}` (Geomechanical Consistency Prior):
  Enforces a physical link between the learned fields, addressing
  the non-uniqueness.
  
  .. math::
      R_{{prior}} = \log(\tau) - \log\left(
          \frac{{\bar{{\kappa}} H^2}}{{(\pi^2 K) / S_s}}
      \right)

* :math:`\mathcal{{L}}_{{smooth}}` (Smoothness Prior):
  A regularizer on the spatial gradients of the predicted fields.
  
  .. math::
      R_{{smooth}}= \|\nabla K\|^2_2 + \|\nabla S_s\|^2_2

See :ref:`User Guide <user_guide_geopriorsubsnet>` for a walkthrough.

Parameters
----------
{params.base.static_input_dim}
{params.base.dynamic_input_dim}
{params.base.future_input_dim}

output_subsidence_dim : int, default 1
    Number of subsidence series per horizon step. (Data output :math:`s`)
output_gwl_dim : int, default 1
    Number of hydraulic-head series. (Data output :math:`h`)

forecast_horizon : int, default 1
    Horizon length :math:`H`. The decoder emits :math:`H` steps
    for all data and physics outputs.

quantiles : list[float] | None, default None
    Optional list of quantile levels; enables the Quantile-Distribution
    head for :math:`s` and :math:`h`.

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


pde_mode : {{'consolidation', 'gw_flow', 'both', 'none'}}, default 'both'
    Select which physics residuals participate in the loss.
    (Priors :math:`\mathcal{{L}}_{{prior}}` and 
     :math:`\mathcal{{L}}_{{smooth}}`
    are always active if their lambda weights are > 0).

    ┌─────────────────┬───────────────────────────────────────────────┐
    │ 'consolidation' │ only the **consolidation** term               │
    │                 │ :math:`R_{{cons}} = \dot{{s}} - (s_{{eq}} - s)/\tau` │
    │ 'gw_flow'       │ only the **diffusivity** term                 │
    │                 │ :math:`R_{{gw}} = S_s \partial_t h - \nabla \cdot (K \nabla h)`
    │ 'both'          │ both residuals (recommended)                  │
    │ 'none'          │ pure data-driven (disables :math:`R_{{cons}}`, :math:`R_{{gw}}`) │
    └─────────────────┴───────────────────────────────────────────────┘

mv : float or LearnableMV, default ``LearnableMV(1e-7)``
    Scalar coefficient of volume compressibility :math:`m_v`[Pa⁻¹].
    Used to calculate :math:`s_{{eq}}`.
    Can be a fixed float or a :class:`LearnableMV` instance.

kappa : float or LearnableKappa, default ``LearnableKappa(1.0)``
    Scalar consistency prior parameter :math:`\bar{{\kappa}}` (unitless).
    Used in :math:`\mathcal{{L}}_{{prior}}` to link :math:`\tau, K, S_s, H`.
    Can be a fixed float or a :class:`LearnableKappa` instance.

gamma_w : float or FixedGammaW, default ``FixedGammaW(9810.0)``
    Scalar unit weight of water :math:`\gamma_w` [N m⁻³].
    Used to calculate :math:`s_{{eq}}`.
    Can be a fixed float or a :class:`FixedGammaW` instance.

h_ref : float or FixedHRef, default ``FixedHRef(0.0)``
    Scalar reference head :math:`h_{{ref}}` [m].
    Used to calculate drawdown :math:`\Delta h = h_{{ref}} - h`.
    Can be a fixed float or a :class:`FixedHRef` instance.

use_effective_h : bool, default False
    If ``True``, use an **effective drained thickness** :math:`H_d`
    in place of the physical thickness :math:`H` wherever thickness
    appears in the physics terms (e.g., in :math:`s_{{eq}}` and the
    :math:`\tau` prior). This is useful when only a fraction of the
    layer drains/responds over the forecast horizon.

hd_factor : float, default 1.0
    Multiplicative factor defining the effective thickness
    :math:`H_d = \text{{Hd\_factor}}\, H`. Only used when
    ``use_effective_h=True``. Typical values lie in :math:`(0, 1]`,
    where lower values model partially draining layers.

kappa_mode : {{'bar', 'kb'}}, default 'bar'
    Interpretation of :math:`\kappa` in the geomechanical-consistency
    prior.
    
    - ``'bar'`` – bulk calibration :math:`\bar{{\kappa}}` with thickness
      :math:`H_*` (which equals :math:`H_d` if
      ``use_effective_h=True``, else :math:`H`):
      
      .. math::
         R_{{\text{{prior}}}} = \log \tau - \log \left(
           \frac{{\bar{{\kappa}}\, H_*^{{2}}\, S_s}}{{\pi^{{2}}\, K}} \right)
    
    - ``'kb'`` – boundary factor :math:`\kappa_b` (same functional
      form, different physical meaning). Also pairs :math:`\kappa_b`
      with :math:`H_*` as above:
      
      .. math::
         R_{{\text{{prior}}}} = \log \tau - \log \left(
           \frac{{\kappa_b\, H_*^{{2}}\, S_s}}{{\pi^{{2}}\, K}} \right)

scale_pde_residuals : bool, default True
    If ``True``, non-dimensionalize physics residuals 
    (:math:`R_{{gw}}`, :math:`R_{{cons}}`)
    with simple data-driven scales so the loss terms are
    :math:`\mathcal{{O}}(1)`.
    (See :func:`fusionlab.nn.pinn.op.default_scales`).

scaling_kwargs : dict | None, default None
    Extra keyword arguments forwarded to :func:`default_scales`.

mode : {{'pihal_like', 'tft_like'}}, default ``None``
    Routing for *future_features*:
        
    * **pihal_like** – decoder gets all :math:`H` rows, encoder none.
    * **tft_like** – first *max_window_size* rows to encoder,
      next :math:`H` rows to decoder. ``None`` inherits
      BaseAttentive default ('tft_like').

objective : {{'hybrid', 'transformer'}}, default ``'hybrid'``
    Selects the backbone architecture. (See `BaseAttentive` docs).
    
attention_levels : str | list[str] | None
    Controls the attention layers used in the decoder.
    (See `BaseAttentive` docs).

name : str, default "GeoPriorSubsNet"
    Model scope as registered in Keras.

**kwargs
    Forwarded verbatim to :class:`tf.keras.Model`.

Notes
-----
* **Required Input:** The `inputs` dictionary for `call` and
    `train_step` **must** include an `H_field` (or `soil_thickness`)
    tensor of shape `(B, H, 1)` representing the soil thickness.
* **Loss Weights:** The composite loss is controlled by four
    weights passed to :meth:`compile`:
    `lambda_cons`, `lambda_gw`, `lambda_prior`, `lambda_smooth`.
* **Outputs:** The model's `call` method returns a dictionary with
    keys: `data_final` (for :math:`s, h` quantiles), `data_mean` (for $s, h$
    means), and `phys_mean_raw` (for raw :math:`K, S_s, \tau` logits).

See Also
--------
fusionlab.nn.models.BaseAttentive
    The data-driven backbone for this model.
fusionlab.nn.pinn.models.TransFlowSubsNet
    The previous, simpler PINN model with scalar physics parameters.
fusionlab.params.LearnableMV
    Parameter class for learnable :math:`m_v`.
fusionlab.params.LearnableKappa
    Parameter class for learnable :math:`\bar{{\kappa}}`.
fusionlab.nn.pinn.op.process_pinn_inputs
    Utility for unpacking input dictionaries.

Examples
--------
>>> import tensorflow as tf
>>> from fusionlab.nn.pinn import GeoPriorSubsNet
>>> from fusionlab.params import LearnableMV, LearnableKappa
>>>
>>> B, T, H = 8, 12, 6 # Batch, Timesteps, Horizon
>>>
>>> model = GeoPriorSubsNet(
...     static_input_dim=3, dynamic_input_dim=8, future_input_dim=4,
...     output_subsidence_dim=1, output_gwl_dim=1,
...     forecast_horizon=H, max_window_size=T,
...     mv=LearnableMV(1e-8),       # Pass new scalar params
...     kappa=LearnableKappa(1.2),  # (K, Ss, Q are removed)
...     pde_mode='both',
...     scale_pde_residuals=True
... )
>>>
>>> # Note: future_features length depends on 'mode'
>>> # Default mode 'tft_like' requires T + H = 12 + 6 = 18 steps
>>> batch = {{
...     "static_features":  tf.zeros([B, 3]),
...     "dynamic_features": tf.zeros([B, T, 8]),
...     "future_features":  tf.zeros([B, T + H, 4]),
...     "coords":           tf.zeros([B, H, 3]),
...     "H_field":          tf.ones([B, H, 1]) * 20.0 # Soil thickness
... }}
>>>
>>> # Compile with new loss weights
>>> model.compile(
...     optimizer='adam',
...     loss='mae', # Data loss (will be wrapped)
...     lambda_cons=1.0,
...     lambda_gw=1.0,
...     lambda_prior=0.5,
...     lambda_smooth=0.1
... )
>>>
>>> # Call returns a dictionary of all outputs
>>> out = model(batch, training=False)
>>> sorted(out.keys())
['data_final', 'data_mean', 'phys_mean_raw']
>>> out['data_mean'].shape
TensorShape([8, 6, 2])
>>> out['phys_mean_raw'].shape
TensorShape([8, 6, 3])
""".format(params=_param_docs)