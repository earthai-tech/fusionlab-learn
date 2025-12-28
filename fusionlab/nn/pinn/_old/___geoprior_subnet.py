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
    from .op import process_pinn_inputs
    from ._geoprior_maths import (
        compute_mv_prior,
        compute_gw_flow_residual,
        compute_smoothness_prior,
        get_log_bounds,
        compose_physics_fields,
        compute_bounds_residual,
        compute_scales,
        rate_to_per_second, 
        scale_residual, 
        integrate_consolidation_mean,
        compute_consolidation_step_residual,
        dt_to_seconds,
        to_rms
    )

    from ._geoprior_utils import (
        to_si_thickness,
        to_si_head,
        deg_to_m,
        coord_ranges,
        validate_scaling_kwargs,
        gwl_to_head_m,
        get_s_init_si, 
        get_h_ref_si,
        infer_dt_units_from_t,
        from_si_subsidence
    )

    from .utils import  process_pde_modes,  _get_coords
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
RandomNormal = KERAS_DEPS.RandomNormal

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
tf_maximum =KERAS_DEPS.maximum 
tf_debugging = KERAS_DEPS.debugging
tf_identity = KERAS_DEPS.identity 
tf_pow = KERAS_DEPS.pow 
tf_sigmoid = KERAS_DEPS.sigmoid 
tf_stop_gradient = KERAS_DEPS.stop_gradient
tf_cast = KERAS_DEPS.cast
tf_abs = KERAS_DEPS.abs
tf_broadcast_to = KERAS_DEPS.broadcast_to
tf_int32 = KERAS_DEPS.int32
tf_cond =KERAS_DEPS.cond 
tf_equal =KERAS_DEPS.equal 
tf_print =KERAS_DEPS.print 
tf_ones = KERAS_DEPS.ones 
tf_greater = KERAS_DEPS.greater 
tf_greater_equal = KERAS_DEPS.greater_equal
tf_reshape = KERAS_DEPS.reshape 
tf_nn =KERAS_DEPS.nn 

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
        "h_ref": [FixedHRef, Real, StrOptions({"auto", "fixed"}), None], 
        "use_effective_h": [bool],
        "hd_factor": [Interval(Real, 0, 1, closed="right")], 
        "kappa_mode": [StrOptions({"bar", "kb"})],
        "offset_mode": [StrOptions({"mul", "log10"})],
        "time_units": [str, None], 
        "bounds_mode": [StrOptions({"soft", "hard"}), None],
        "residual_method":[StrOptions({"exact", "euler"})]       
        
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
        h_ref: Union[FixedHRef, float, str, None] = FixedHRef(value=0.0, mode="auto"), 
        use_effective_h: bool = False,
        hd_factor: float = 1.0 ,  # if Hd = Hd_factor * H
        kappa_mode: str = "kb",   # {"bar", "kb"}  # κ̄ vs κ_b
        offset_mode: str = "mul",  # {"mul", "log10"}
        bounds_mode : str ="soft", 
        residual_method: str = 'exact', # {"exact", "euler"}
        time_units: Optional[str] = None,  
        use_vsn: bool = True,
        vsn_units: Optional[int] = None,
        mode: Optional[str]=None, 
        objective: Optional[str]=None, 
        attention_levels:Optional[Union[str, List[str]]]=None, 
        architecture_config: Optional[Dict] = None,
        scale_pde_residuals: bool = True,
        scaling_kwargs: Optional[Dict[str, Any]] = None,
        name: str = "GeoPriorSubsNet", 
        verbose: int = 0, 
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

        # Always include a forcing term Q(t,x,y) for gw_flow PDE
        self.output_Q_dim = 1
        self._phys_output_dim = (
            self.output_K_dim + self.output_Ss_dim + self.output_tau_dim 
            + self.output_Q_dim
        )

        if 'output_dim' in kwargs: 
            kwargs.pop ('output_dim') 
            
        self.scaling_kwargs = dict(scaling_kwargs or {})
        b = self.scaling_kwargs.get("bounds")
        if isinstance(b, Mapping) and not isinstance(b, dict):
            self.scaling_kwargs["bounds"] = dict(b)

        self.bounds_mode = bounds_mode or "soft"
        # -----------------------------------------------------------------
        # Resolve time_units (scaling_kwargs wins; else propagate init)
        # -----------------------------------------------------------------
 
        if ("time_units" not in self.scaling_kwargs): 
                self.scaling_kwargs["time_units"] = time_units 
        
        scale_tu = self.scaling_kwargs.get("time_units", None)
        if isinstance(scale_tu, str) and not scale_tu.strip():
            scale_tu = None
        
        # precedence: scaling wins; else init; then always store in scaling_kwargs
        self.time_units = scale_tu if scale_tu is not None else time_units
        self.scaling_kwargs["time_units"] = self.time_units

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
            verbose= verbose, 
            name=name,
            **kwargs
        )
        
        self.pde_modes_active = process_pde_modes(pde_mode)
        self.scale_pde_residuals = bool(scale_pde_residuals)

        # --- Process new scalar physics params ---
        if isinstance(mv, (int, float)):
            mv = LearnableMV(initial_value=float(mv))
        if isinstance(kappa, (int, float)):
            kappa = LearnableKappa(initial_value=float(kappa))
        if isinstance(gamma_w, (int, float)):
            gamma_w = FixedGammaW(value=float(gamma_w))
        if isinstance(h_ref, str):
            key = h_ref.strip().lower()
            if key in ("auto", "history", "last", "last_obs", "last_observed"):
                h_ref = FixedHRef(value=0.0, mode="auto")
            else:
                raise ValueError(
                    f"Unsupported h_ref={h_ref!r}. Use a float or 'auto'."
                )
        elif h_ref is None:
            h_ref = FixedHRef(value=0.0, mode="auto")
        elif isinstance(h_ref, (int, float)):
            # numeric => explicit fixed datum
            h_ref = FixedHRef(value=float(h_ref), mode="fixed")
        # else: assume it's already a FixedHRef-like objec
        self.h_ref_config = h_ref

        self.mv_config = mv
        self.kappa_config = kappa
        self.gamma_w_config = gamma_w

        
        self.use_effective_thickness = use_effective_h
        self.Hd_factor = hd_factor   # if Hd = Hd_factor * H
        self.kappa_mode = kappa_mode  # {"bar", "kb"}  # κ̄ vs κ_b

        # Sensible defaults before compile() is called
        self.lambda_cons = 1.0
        self.lambda_gw = 1.0
        self.lambda_prior = 1.0
        self.lambda_smooth = 1.0
        self.lambda_mv = 0.0
        self._mv_lr_mult = 1.0
        self._kappa_lr_mult = 1.0
        self.lambda_bounds = 0.0   

        # global scaling for *all* physics terms
        self.offset_mode = offset_mode
        self.residual_method = residual_method
        
        self._lambda_offset = self.add_weight(
            name="lambda_offset",
            shape=(),
            initializer=Constant(1.0),
            trainable=False,
            dtype=tf_float32,
        )
        self._gwl_dyn_index = None 

        logger.info(
            f"Initialized GeoPriorSubsNet with scalar physics params:"
            f" mv_trainable={mv.trainable},"
            f" kappa_trainable={kappa.trainable}"
        )
        
        self._init_coordinate_corrections()
        self._build_pinn_components()


    def _assert_dynamic_names_match_tensor(self, Xh):
        sk = self.scaling_kwargs or {}
        names = sk.get("dynamic_feature_names", None)
        if names is None:
            return
        n = len(list(names))
        # python-side check if possible, otherwise tf assertion
        tf_debugging.assert_equal(
            tf_shape(Xh)[-1], tf_constant(n, tf_int32),
            message=( 
                "dynamic_feature_names length"
                " != dynamic_features last dim"
                )
        )

    def _build_attentive_layers(self):
        super()._build_attentive_layers()
        self._build_physics_layers()
    
    def _build_physics_layers(self):
        logK_min, logK_max, logSs_min, logSs_max = get_log_bounds(
            self, as_tensor=False, verbose=self.verbose
        )
    
        # fallback if bounds missing (soft can survive; hard should not)
        if (logK_min is None) or (logSs_min is None):
            if self.bounds_mode == "hard":
                raise ValueError(
                    "bounds_mode='hard' requires bounds for"
                    " K and Ss in scaling_kwargs['bounds'] "
                    "(K_min/K_max/Ss_min/Ss_max or logK_*/logSs_*)."
                )
            logK0 = 0.0
            logSs0 = 0.0
        else:
            logK0  = 0.5 * (logK_min  + logK_max)
            logSs0 = 0.5 * (logSs_min + logSs_max)
    
        if self.bounds_mode == "hard":
            k_bias = 0.0
            ss_bias = 0.0
        else:
            k_bias = float(logK0)
            ss_bias = float(logSs0)
    
        # ------------------------------------------------------------
        # Q head is optional (v3.2): only create if output_Q_dim > 0
        # ------------------------------------------------------------
        if int(getattr(self, "output_Q_dim", 0) or 0) > 0:
            self.Q_head = Dense(
                self.output_Q_dim,  # usually 1
                name="Q_head",
                kernel_initializer="zeros",
                bias_initializer=Constant(0.0),
            )
        else:
            self.Q_head = None
    
        self.K_head = Dense(
            self.output_K_dim,  # usually 1
            name="K_head",
            kernel_initializer="zeros",
            bias_initializer=Constant(k_bias),
        )
        self.Ss_head = Dense(
            self.output_Ss_dim,  # usually 1
            name="Ss_head",
            kernel_initializer="zeros",
            bias_initializer=Constant(ss_bias),
        )
        self.tau_head = Dense(
            self.output_tau_dim,  # usually 1
            name="tau_head",
            kernel_initializer="zeros",
            bias_initializer=Constant(0.0),
        )
    
        self.H_field = None
        self.eps_prior_metric = Mean(name="epsilon_prior")
        self.eps_cons_metric = Mean(name="epsilon_cons")

    
    def _init_coordinate_corrections(
        self,
        gwl_units: Union[int, None] = None,
        subs_units: Union[int, None] = None,
        hidden: Tuple[int, int] = (32, 16),
        act: str = "gelu",
    ) -> None:

        gwl_units = gwl_units or self.output_gwl_dim
        subs_units = subs_units or self.output_subsidence_dim
    
        def _branch(out_units: int, name: str) -> Sequential:
            """
            Small helper to create a (t, x, y) -> field-correction MLP.
    
            Input shape is (None, 3), i.e. a per-time-step coordinate
            vector. Keras will treat the leading dimension as time/space
            when used in a time-distributed manner.
            """
            return Sequential([
                InputLayer(input_shape=(None, 3)),
                Dense(hidden[0], activation=act, name=f"{name}_dense1"),
                Dense(hidden[1], activation=act, name=f"{name}_dense2"),
                Dense(
                    out_units,
                    activation=None,
                    kernel_initializer=RandomNormal(stddev=1e-4),
                    bias_initializer="zeros",
                    name=f"{name}_out",
                ),
            ], name=name)
        
            
        # Coordinate-based correction for groundwater head
        self.coord_mlp = _branch(gwl_units, "coord_mlp")
    
        # Coordinate-based correction for subsidence
        self.subs_coord_mlp = _branch(subs_units, "subs_coord_mlp")
    
        # Coordinate-based corrections for physics fields K, Ss, tau
        self.K_coord_mlp = _branch(self.output_K_dim, "K_coord_mlp")
        self.Ss_coord_mlp = _branch(self.output_Ss_dim, "Ss_coord_mlp")
        self.tau_coord_mlp = _branch(self.output_tau_dim, "tau_coord_mlp")
        
    def _build_pinn_components(self):

        # Compressibility m_v
        if isinstance(self.mv_config, LearnableMV):
            # Trainable log-parameter for m_v
            self.log_mv = self.add_weight(
                name="log_param_mv",
                shape=(),
                initializer=Constant(
                    tf_log(self.mv_config.initial_value)
                ),
                trainable=self.mv_config.trainable,
            )
        else:
            # Fixed scalar m_v as a constant
            self._mv_fixed = tf_constant(
                float(self.mv_config.initial_value),
                dtype=tf_float32,
            )
    
    
        # Consistency factor κ
        if isinstance(self.kappa_config, LearnableKappa):
            # Trainable log-parameter for κ
            self.log_kappa = self.add_weight(
                name="log_param_kappa",
                shape=(),
                initializer=Constant(
                    tf_log(self.kappa_config.initial_value)
                ),
                trainable=self.kappa_config.trainable,
            )
        else:
            # Fixed scalar κ as a constant
            self._kappa_fixed = tf_constant(
                float(self.kappa_config.initial_value),
                dtype=tf_float32,
            )
    
        # Fixed physical constants (water unit weight, reference head)
        self.gamma_w = tf_cast(self.gamma_w_config.get_value(), tf_float32)

        self.h_ref_mode = getattr(self.h_ref_config, "mode", "fixed")

        self.h_ref = tf_constant(
            float(self.h_ref_config.value),  # always a float fallback
            dtype=tf_float32,
        )
    
        # Runtime placeholders for physics fields
        # These are filled in train_step / evaluate_physics()
        # *after* coordinate corrections and positivity have been applied,
        # and can be inspected via get_last_physics_fields().
        self.K_field = None
        self.Ss_field = None
        self.tau_field = None


    @tf_autograph.experimental.do_not_convert
    def run_encoder_decoder_core(
        self,
        static_input: Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        coords_input: Tensor,
        training: bool,
    ) -> Tuple[Tensor, Tensor]:
        
        # ------------------------------------------------------------------
        # 0. Basic time dimension inference
        # ------------------------------------------------------------------
        time_steps = tf_shape(dynamic_input)[1]
    
        # ------------------------------------------------------------------
        # 1. Initial feature processing (VSN or dense path)
        # ------------------------------------------------------------------
        static_context, dyn_proc, fut_proc = (
            None,
            dynamic_input,
            future_input,
        )
    
        if self.architecture_config.get("feature_processing") == "vsn":
            # Static VSN path
            if self.static_vsn is not None:
                vsn_static_out = self.static_vsn(
                    static_input,
                    training=training,
                )
                static_context = self.static_vsn_grn(
                    vsn_static_out,
                    training=training,
                )
    
            # Dynamic VSN path
            if self.dynamic_vsn is not None:
                dyn_context = self.dynamic_vsn(
                    dynamic_input,
                    training=training,
                )
                dyn_proc = self.dynamic_vsn_grn(
                    dyn_context,
                    training=training,
                )
    
            # Future VSN path
            if self.future_vsn is not None:
                fut_context = self.future_vsn(
                    future_input,
                    training=training,
                )
                fut_proc = self.future_vsn_grn(
                    fut_context,
                    training=training,
                )
        else:
            # Non-VSN dense preprocessing path
            if self.static_dense is not None:
                processed_static = self.static_dense(static_input)
                static_context = self.grn_static_non_vsn(
                    processed_static,
                    training=training,
                )
            if self.dynamic_dense is not None:
                dyn_proc = self.dynamic_dense(dynamic_input)
            if self.future_dense is not None:
                fut_proc = self.future_dense(future_input)
    
        logger.debug(
            "Shape after VSN/initial processing: "
            f"Dynamic={getattr(dyn_proc, 'shape', 'N/A')}, "
            f"Future={getattr(fut_proc, 'shape', 'N/A')}"
        )
    
        # ------------------------------------------------------------------
        # 2. Encoder path (hybrid LSTM/Transformer)
        # ------------------------------------------------------------------
        encoder_input_parts = [dyn_proc]
    
        if self._mode == "tft_like" and self.future_input_dim > 0:
            # For TFT-like mode, the first T steps of future covariates
            # are concatenated with dynamic features in the encoder.
            fut_enc_proc = fut_proc[:, :time_steps, :]
            encoder_input_parts.append(fut_enc_proc)
    
        encoder_raw = tf_concat(encoder_input_parts, axis=-1)
        encoder_input = self.encoder_positional_encoding(encoder_raw)
    
        if self.architecture_config["encoder_type"] == "hybrid":
            # Multi-scale LSTM encoder followed by multiscale aggregation
            lstm_out = self.multi_scale_lstm(
                encoder_input,
                training=training,
            )
            encoder_sequences = aggregate_multiscale_on_3d(
                lstm_out,
                mode="concat",
            )
        else:
            # Pure transformer encoder
            encoder_sequences = encoder_input
            for mha, norm in self.encoder_self_attention:
                attn_out = mha(
                    encoder_sequences,
                    encoder_sequences,
                )
                encoder_sequences = norm(
                    encoder_sequences + attn_out
                )
    
        # Optional dynamic time windowing (DTW)
        if self.apply_dtw and self.dynamic_time_window is not None:
            encoder_sequences = self.dynamic_time_window(
                encoder_sequences,
                training=training,
            )
    
        logger.debug(
            f"Encoder sequences shape: {encoder_sequences.shape}"
        )
    
        # ------------------------------------------------------------------
        # 3. Decoder path (modified to inject coords_input)
        # ------------------------------------------------------------------
        if self._mode == "tft_like" and self.future_input_dim > 0:
            # TFT-like: remaining steps go to decoder
            fut_dec_proc = fut_proc[:, time_steps:, :]
        elif self.future_input_dim > 0:
            # PIHAL-like: decoder sees all future covariates over horizon
            fut_dec_proc = fut_proc
        else:
            fut_dec_proc = None
    
        decoder_parts = []
    
        # Broadcast static context to all horizon steps
        if static_context is not None:
            static_expanded = tf_expand_dims(static_context, 1)
            static_expanded = tf_tile(
                static_expanded,
                [1, self.forecast_horizon, 1],
            )
            decoder_parts.append(static_expanded)
    
        # Decoder future features with positional encoding
        if fut_dec_proc is not None:
            future_with_pos = self.decoder_positional_encoding(
                fut_dec_proc
            )
            decoder_parts.append(future_with_pos)
    
        # Coordinate injection: this is the crucial (t, x, y) signal
        if coords_input is None:
            raise ValueError(
                "GeoPriorSubsNet.run_encoder_decoder_core requires "
                "'coords_input' (B, H, 3) to be provided."
            )
        decoder_parts.append(coords_input)
    
        # If everything is missing (very degenerate case), fall back to
        # a zero tensor so shapes remain valid.
        if not decoder_parts:
            batch_size = tf_shape(dynamic_input)[0]
            raw_decoder_input = tf_zeros(
                (batch_size, self.forecast_horizon, self.attention_units)
            )
        else:
            raw_decoder_input = tf_concat(decoder_parts, axis=-1)
    
        projected_decoder_input = self.decoder_input_projection(
            raw_decoder_input
        )
        logger.debug(
            "Projected decoder input shape: "
            f"{projected_decoder_input.shape}"
        )
    
        # ------------------------------------------------------------------
        # 4. Apply decoder attention levels and aggregate
        # ------------------------------------------------------------------
        # final_features is the 3D tensor (B, H, U) that both data and
        # physics paths will consume.
        final_features = self.apply_attention_levels(
            projected_decoder_input,
            encoder_sequences,
            training=training,
        )
    
        logger.debug(
            f"Shape after final fusion: {final_features.shape}"
        )
    
        # 3D features for physics head
        phys_features_raw_3d = final_features
    
        # Time-aggregated 2D features for data decoder
        data_features_2d = aggregate_time_window_output(
            final_features,
            self.final_agg,
        )
    
        return data_features_2d, phys_features_raw_3d

    @tf_autograph.experimental.do_not_convert
    def call(
        self,
        inputs: Dict[str, Optional[Tensor]],
        training: bool = False,
    ) -> Dict[str, Tensor]:
    
        # --------------------------------------------------------------
        # 1) Unpack standardized PINN inputs
        # --------------------------------------------------------------
        (
            t, x, y,
            H_field,
            static_features,
            dynamic_features,
            future_features,
        ) = process_pinn_inputs(inputs, mode="auto", model_name="geoprior")
    
        coords_for_decoder = tf_concat([t, x, y], axis=-1)  # (B,H,3)
        self.H_field = H_field
    
        check_inputs(
            dynamic_inputs=dynamic_features,
            static_inputs=static_features,
            future_inputs=future_features,
            dynamic_input_dim=self.dynamic_input_dim,
            static_input_dim=self.static_input_dim,
            future_input_dim=self.future_input_dim,
            forecast_horizon=self.forecast_horizon,
            verbose=0,
        )
    
        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=[static_features, dynamic_features, future_features],
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            mode="strict",
            verbose=0,
        )
    
        # --------------------------------------------------------------
        # 2) Shared core (encoder/decoder)
        # --------------------------------------------------------------
        data_features_2d, phys_features_raw_3d = self.run_encoder_decoder_core(
            static_input=static_p,
            dynamic_input=dynamic_p,
            future_input=future_p,
            coords_input=coords_for_decoder,
            training=training,
        )
    
        # --------------------------------------------------------------
        # 3) Data path: predict head (mean), subsidence residual optional
        # --------------------------------------------------------------
        mlp_corr = self.coord_mlp(coords_for_decoder, training=training)       # (B,H,gwl_dim)
        s_corr   = self.subs_coord_mlp(coords_for_decoder, training=training)  # (B,H,subs_dim)
    
        decoded_data_means_net = self.multi_decoder(
            data_features_2d,
            training=training,
        )
        decoded_data_means_net = decoded_data_means_net + tf_concat([s_corr, mlp_corr], axis=-1)
    
        subs_net = decoded_data_means_net[..., : self.output_subsidence_dim]   # (B,H,1) residual (optional)
        gwl_mean = decoded_data_means_net[..., self.output_subsidence_dim :]   # (B,H,1)
    
        # --------------------------------------------------------------
        # 4) Physics path: predict K,Ss,Δlogτ and Q
        # --------------------------------------------------------------
        K_raw   = self.K_head(phys_features_raw_3d, training=training)     # (B,H,1)
        Ss_raw  = self.Ss_head(phys_features_raw_3d, training=training)
        tau_raw = self.tau_head(phys_features_raw_3d, training=training)
        Q_raw   = self.Q_head(phys_features_raw_3d, training=training)    

        Q_raw = (      # (B,H,1)
            self.Q_head(phys_features_raw_3d, training=training) 
            if self.Q_head is not None else None
            )
        parts = [K_raw, Ss_raw, tau_raw]
        if Q_raw is not None:
            parts.append(Q_raw)
        phys_mean_raw = tf_concat(parts, axis=-1)

        # --------------------------------------------------------------
        # 5) OPTION-1: compute physics-driven mean subsidence in SI
        # --------------------------------------------------------------
        # -- Make base K,Ss,tau spatial-only (optional but recommended) --
        # (prevents time-varying K,Ss,tau through phys_features_raw_3d)
        K_base   = tf_broadcast_to(tf_reduce_mean(K_raw,   axis=1, keepdims=True), tf_shape(K_raw))
        Ss_base  = tf_broadcast_to(tf_reduce_mean(Ss_raw,  axis=1, keepdims=True), tf_shape(Ss_raw))
        tau_base = tf_broadcast_to(tf_reduce_mean(tau_raw, axis=1, keepdims=True), tf_shape(tau_raw))
    
        coords_flat = coords_for_decoder  # already (B,H,3)
        H_si = to_si_thickness(H_field, self.scaling_kwargs)
    
        (K_field, Ss_field, tau_field,
         tau_phys, Hd_eff,
         delta_log_tau, logK, logSs) = compose_physics_fields(
            self,
            coords_flat=coords_flat,
            H_si=H_si,
            K_base=K_base,
            Ss_base=Ss_base,
            tau_base=tau_base,
            training=training,
            verbose=0,
        )
    
        # predicted head -> SI head (meters) (handles "depth" vs "head")
        h_mean_si = to_si_head(gwl_mean, self.scaling_kwargs)
        h_mean_si = gwl_to_head_m(h_mean_si, self.scaling_kwargs, inputs=inputs)
    
        # refs / init
        h_ref_si  = get_h_ref_si(self, inputs, like=h_mean_si)
        # s_init_si = get_s_init_si(self, inputs, like=h_mean_si)
        
        # baseline cumulative at forecast start (SI)
        s0_cum_si = get_s_init_si(self, inputs, like=h_mean_si)
        
        # IMPORTANT: Voigt state here should be incremental relative to h_ref at t0
        s0_inc_si = tf_zeros_like(s0_cum_si)
        
        # dt in *time_units*
        dt_units = infer_dt_units_from_t(t, self.scaling_kwargs)
    
        # physics mean settlement path (exact-step)
        s_inc_si = integrate_consolidation_mean(
            h_mean_si=h_mean_si,
            Ss_field=Ss_field,
            H_field_si=H_si,
            tau_field=tau_field,
            h_ref_si=h_ref_si,
            s_init_si=s0_inc_si,     # <-- change
            dt=dt_units,
            time_units=self.time_units,
            method=self.residual_method,
            verbose=self.verbose,
        )  # (B,H,1) incremental compaction since t0

        # map to requested subsidence kind
        kind = str((self.scaling_kwargs or {}).get(
            "subsidence_kind", "cumulative")).strip().lower()
        
        if kind == "increment":
            ds0 = s_inc_si[:, :1, :]
            dsr = s_inc_si[:, 1:, :] - s_inc_si[:, :-1, :]
            subs_phys_si = tf_concat([ds0, dsr], axis=1)
        else:
            # cumulative output = baseline + increment
            subs_phys_si = s0_cum_si + s_inc_si
    
        # map physics mean to the training “subsidence_kind”
        subs_phys_model = from_si_subsidence(subs_phys_si, self.scaling_kwargs)
    
        # Option: allow a learned residual around physics mean
        allow_resid = bool((self.scaling_kwargs or {}).get("allow_subs_residual", False))
        subs_mean = subs_phys_model + subs_net if allow_resid else subs_phys_model
    
        decoded_data_means = tf_concat([subs_mean, gwl_mean], axis=-1)
        data_mean_raw = decoded_data_means
    
        # --------------------------------------------------------------
        # 6) Quantiles (unchanged): now centered on physics mean
        # --------------------------------------------------------------
        if self.quantiles is not None:
            data_final = self.quantile_distribution_modeling(
                decoded_data_means,
                training=training,
            )
        else:
            data_final = decoded_data_means
    
        return {
            "data_final": data_final,
            "data_mean_raw": data_mean_raw,   
            "phys_mean_raw": phys_mean_raw,
        }

    def train_step(self, data):
        """
        Custom Keras training step for GeoPriorSubsNet (v3.2 / Option-1).
    
        Option-1 recap
        --------------
        - The **mean subsidence** is physics-driven: it is computed from the
          predicted head via the consolidation integrator (exact-step).
        - The network may still output an optional **subsidence residual**
          around the physics mean (controlled in call()).
        - Physics fields (K, Ss, tau) are predicted as logits and then mapped
          to SI-consistent, bounded/positive fields via `compose_physics_fields`.
        - Groundwater PDE residual includes an optional learnable forcing Q(t,x,y).
    
        Important implementation detail
        -------------------------------
        All PDE derivatives are taken w.r.t. the **same coords tensor** that is fed
        into `call()` (we inject it into inputs_fwd["coords"]). Otherwise,
        `tape.gradient(..., coords)` can become None.
        """
        verbose = int(getattr(self, "verbose", 0))
    
        # ----------------------------------------------------------
        # 0) Unpack (inputs, targets) + normalize target keys
        # ----------------------------------------------------------
        inputs, targets = data
        if isinstance(targets, dict):
            targets = rename_dict_keys(
                targets.copy(),
                param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"},
            )
    
        # ----------------------------------------------------------
        # 1) Thickness field is required by physics (H_field / soil_thickness)
        # ----------------------------------------------------------
        H_in = get_tensor_from(inputs, "H_field", "soil_thickness", auto_convert=True)
        if H_in is None:
            raise ValueError(
                "train_step requires 'H_field' (or 'soil_thickness') in inputs."
            )
    
        H_field = tf_convert_to_tensor(H_in, dtype=tf_float32)
        self.H_field = H_field  # optional debug handle
    
        validate_scaling_kwargs(self.scaling_kwargs)
        H_si = to_si_thickness(H_field, self.scaling_kwargs)  # SI meters
    
        # ----------------------------------------------------------
        # 2) Coordinates: ensure (B,H,3) and force forward-pass to use them
        # ----------------------------------------------------------
        coords = tf_convert_to_tensor(_get_coords(inputs), tf_float32)
    
        # Accept (B,3) -> (B,1,3) edge cases
        if coords.shape.rank == 2:
            coords = tf_expand_dims(coords, axis=1)
    
        if coords.shape.rank != 3 or coords.shape[-1] != 3:
            raise ValueError(
                "coords must have shape (B,H,3) with last dim (t,x,y). "
                f"Got shape={coords.shape}."
            )
    
        inputs_fwd = dict(inputs)
        inputs_fwd["coords"] = coords  # crucial for AD connectivity
    
        t = coords[..., 0:1]  # (B,H,1)
    
        # ----------------------------------------------------------
        # 3) Forward + losses + physics derivatives
        # ----------------------------------------------------------
        with GradientTape(persistent=True) as tape:
            tape.watch(coords)
    
            # ----------------------
            # 3.1 Forward pass
            # ----------------------
            outputs = self(inputs_fwd, training=True)
    
            data_final = outputs.get("data_final", None)
            if data_final is None:
                raise ValueError("Model outputs missing 'data_final'.")
    
            # Quantile-aware tensors are OK here; we only slice the last dim
            s_pred_final, gwl_pred_final = self.split_data_predictions(data_final)
            y_pred = {"subs_pred": s_pred_final, "gwl_pred": gwl_pred_final}
    
            # Supervised data loss (pinball, MSE, etc. defined at compile-time)
            data_loss = self.compiled_loss(
                y_true=targets,
                y_pred=y_pred,
                regularization_losses=self.losses,
            )
    
            # ----------------------
            # 3.2 Mean head for physics
            # ----------------------
            # Prefer the explicit mean output produced by call() (Option-1).
            data_mean_raw = outputs.get("data_mean_raw", None)
    
            if data_mean_raw is not None:
                _, gwl_mean_raw = self.split_data_predictions(data_mean_raw)
            else:
                # Fallback: average over quantile axis if present
                gwl_mean_raw = gwl_pred_final
                r = tf_rank(gwl_mean_raw)
                # If (B,H,Q,1) or (B,H,Q), average over Q axis=2
                gwl_mean_raw = tf_cond(
                    tf_greater_equal(r, 3),
                    lambda: tf_reduce_mean(gwl_mean_raw, axis=2),
                    lambda: gwl_mean_raw,
                )
                if tf_rank(gwl_mean_raw) == 2:
                    gwl_mean_raw = gwl_mean_raw[:, :, None]
    
            # Convert predicted gwl/depth -> SI meters -> head meters (SI)
            gwl_si = to_si_head(gwl_mean_raw, self.scaling_kwargs)
            h_si = gwl_to_head_m(gwl_si, self.scaling_kwargs, inputs=inputs_fwd)  # (B,H,1)
    
            # ----------------------
            # 3.3 Physics logits: K, Ss, Δlogτ (+ optional Q)
            # ----------------------
            phys_mean_raw = outputs.get("phys_mean_raw", None)
            if phys_mean_raw is None:
                raise ValueError("Model outputs missing 'phys_mean_raw'.")
    
            # Expected layout: [K, Ss, delta_log_tau, (optional) Q]
            ( K_logits, Ss_logits, dlogtau_logits, Q_logits  
            )= self.split_physics_predictions(phys_mean_raw)


            # Option-1 recommendation: keep K,Ss,tau time-constant over the horizon
            sk = self.scaling_kwargs or {}
            freeze_fields = bool(sk.get("freeze_physics_fields_over_time", True))
            if freeze_fields:
                K_base = tf_broadcast_to(
                    tf_reduce_mean(K_logits, axis=1, keepdims=True),
                    tf_shape(K_logits),
                )
                Ss_base = tf_broadcast_to(
                    tf_reduce_mean(Ss_logits, axis=1, keepdims=True),
                    tf_shape(Ss_logits),
                )
                dlogtau_base = tf_broadcast_to(
                    tf_reduce_mean(dlogtau_logits, axis=1, keepdims=True),
                    tf_shape(dlogtau_logits),
                )
            else:
                K_base, Ss_base, dlogtau_base = K_logits, Ss_logits, dlogtau_logits
    
            # Map logits -> SI-consistent fields (positivity/bounds + tau_phys)
            (
                K_field, Ss_field, tau_field, tau_phys, Hd_eff,
                delta_log_tau, logK, logSs,
            ) = compose_physics_fields(
                self,
                coords_flat=coords,
                H_si=H_si,
                K_base=K_base,
                Ss_base=Ss_base,
                tau_base=dlogtau_base,  # interpreted as Δlogτ
                training=True,
                verbose=verbose,
            )
    
            # ------------------------------------------------------
            # 4) GW PDE derivatives via AD (w.r.t coords)
            # ------------------------------------------------------
            dh_dcoords = tape.gradient(h_si, coords)
            if dh_dcoords is None:
                raise ValueError(
                    "dh_dcoords is None: graph not connected to coords. "
                    "Ensure call() consumes inputs['coords'] (we inject it above)."
                )
    
            # Raw derivatives w.r.t (t,x,y) in *whatever coordinate units* you use
            dh_dt_raw = dh_dcoords[..., 0:1]
            dh_dx_raw = dh_dcoords[..., 1:2]
            dh_dy_raw = dh_dcoords[..., 2:3]
    
            # Divergence terms in raw coord units:
            #   ∂x(K ∂x h) + ∂y(K ∂y h)
            K_dh_dx = K_field * dh_dx_raw
            K_dh_dy = K_field * dh_dy_raw
    
            dKdhx_dcoords = tape.gradient(K_dh_dx, coords)
            dKdhy_dcoords = tape.gradient(K_dh_dy, coords)
            if (dKdhx_dcoords is None) or (dKdhy_dcoords is None):
                raise ValueError("Second-order PDE gradients are None.")
    
            d_K_dh_dx_dx_raw = dKdhx_dcoords[..., 1:2]  # ∂x(...)
            d_K_dh_dy_dy_raw = dKdhy_dcoords[..., 2:3]  # ∂y(...)
    
            # Smoothness prior needs ∂xK,∂yK,∂xSs,∂ySs
            dK_dcoords = tape.gradient(K_field, coords)
            dSs_dcoords = tape.gradient(Ss_field, coords)
            if (dK_dcoords is None) or (dSs_dcoords is None):
                raise ValueError("K/Ss spatial gradients are None.")
    
            dK_dx_raw = dK_dcoords[..., 1:2]
            dK_dy_raw = dK_dcoords[..., 2:3]
            dSs_dx_raw = dSs_dcoords[..., 1:2]
            dSs_dy_raw = dSs_dcoords[..., 2:3]
    
            # ------------------------------------------------------
            # 4.1 Chain-rule correction (normalized coords / degrees)
            # ------------------------------------------------------
            coords_norm = bool(sk.get("coords_normalized", False))
            coords_deg = bool(sk.get("coords_in_degrees", False))
    
            # Start from raw derivatives and progressively convert to SI:
            dh_dt = dh_dt_raw
            d_K_dh_dx_dx = d_K_dh_dx_dx_raw
            d_K_dh_dy_dy = d_K_dh_dy_dy_raw
            dK_dx, dK_dy, dSs_dx, dSs_dy = dK_dx_raw, dK_dy_raw, dSs_dx_raw, dSs_dy_raw
    
            # If coords are normalized, convert derivatives to per-(original coord unit)
            tR_tf = xR_tf = yR_tf = None
            if coords_norm:
                tR, xR, yR = coord_ranges(self.scaling_kwargs)
                if tR is None or xR is None or yR is None:
                    raise ValueError(
                        "coords_normalized=True but coord_ranges missing."
                    )
    
                tR_tf = tf_constant(float(tR), tf_float32)
                xR_tf = tf_constant(float(xR), tf_float32)
                yR_tf = tf_constant(float(yR), tf_float32)
    
                # First derivatives: /range
                dh_dt = dh_dt / tR_tf
    
                # Second derivatives: /range^2
                d_K_dh_dx_dx = d_K_dh_dx_dx / (xR_tf * xR_tf)
                d_K_dh_dy_dy = d_K_dh_dy_dy / (yR_tf * yR_tf)
    
                # Smoothness: /range
                dK_dx = dK_dx / xR_tf
                dK_dy = dK_dy / yR_tf
                dSs_dx = dSs_dx / xR_tf
                dSs_dy = dSs_dy / yR_tf
    
            # If coords are degrees, convert spatial derivatives to per-meter
            if coords_deg:
                deg2m_x = deg_to_m("x", self.scaling_kwargs)  # m/deg
                deg2m_y = deg_to_m("y", self.scaling_kwargs)
    
                d_K_dh_dx_dx = d_K_dh_dx_dx / (deg2m_x * deg2m_x)
                d_K_dh_dy_dy = d_K_dh_dy_dy / (deg2m_y * deg2m_y)
    
                dK_dx = dK_dx / deg2m_x
                dK_dy = dK_dy / deg2m_y
                dSs_dx = dSs_dx / deg2m_x
                dSs_dy = dSs_dy / deg2m_y
    
            # Time derivative must be per second for SI PDE
            dh_dt = rate_to_per_second(dh_dt, time_units=self.time_units)
    
            # ------------------------------------------------------
            # 5) Q logits -> SI (1/s)
            # ------------------------------------------------------
            Q_base = tf_cast(Q_logits, tf_float32)
    
            # If Q was produced w.r.t normalized time, de-normalize it first
            if coords_norm and bool(sk.get("Q_wrt_normalized_time", False)):
                if tR_tf is None:
                    raise ValueError(
                        "Q_wrt_normalized_time=True but coord_range_t missing."
                    )
                Q_base = Q_base / (tR_tf + tf_constant(1e-12, tf_float32))
    
            # Convert 1/time_units -> 1/s unless already SI
            if bool(sk.get("Q_in_per_second", False)) or bool(sk.get("Q_in_si", False)):
                Q_si = Q_base
            else:
                Q_si = rate_to_per_second(Q_base, time_units=self.time_units)
    
            # Ensure broadcastable with (B,H,1)
            # if tf_rank(Q_si) == 2:
            #     Q_si = Q_si[:, :, None]
            # Q_si = Q_si + tf_zeros_like(dh_dt)
            
            # Ensure (B,H,1) and broadcastable with dh_dt (graph-safe)
            r = tf_rank(Q_si)
            Q_si = tf_cond(
                tf_equal(r, 2),
                lambda: tf_reshape(Q_si, [tf_shape(Q_si)[0], tf_shape(Q_si)[1], 1]),
                lambda: Q_si,
            )
            
            # now broadcast with dh_dt
            Q_si = Q_si + tf_zeros_like(dh_dt)

            # ------------------------------------------------------
            # 6) Physics-driven mean settlement (Option-1)
            # ------------------------------------------------------
            dt_units = infer_dt_units_from_t(t, self.scaling_kwargs)  # (B,H,1) in self.time_units
            s_init_si = get_s_init_si(self, inputs_fwd, like=h_si[:, :1, :])   # (B,1,1)
            h_ref_si_11 = get_h_ref_si(self, inputs_fwd, like=h_si[:, :1, :])  # (B,1,1)
            h_ref_si = h_ref_si_11 + tf_zeros_like(h_si)                       # (B,H,1)
    
            # Mean cumulative settlement path (B,H,1) in SI meters
            s_bar = integrate_consolidation_mean(
                h_mean_si=h_si,
                Ss_field=Ss_field,
                H_field_si=H_si,
                tau_field=tau_field,
                h_ref_si=h_ref_si,
                s_init_si=s_init_si,
                dt=dt_units,
                time_units=self.time_units,
                method=self.residual_method,
                verbose=self.verbose, 
            )
    
            # One-step residual in meters, then normalize to m/s using dt_sec
            # Build state length H+1 for step residual:
            s_state = tf_concat([s_init_si, s_bar], axis=1)          # (B,H+1,1)
            # Align h_n with steps: use h_si for each step; pad last value for shape
            h_state = tf_concat([h_si, h_si[:, -1:, :]], axis=1)     # (B,H+1,1)
    
            cons_step_m = compute_consolidation_step_residual(
                s_state_si=s_state,
                h_mean_si=h_state,
                Ss_field=Ss_field,
                H_field_si=H_si,
                tau_field=tau_field,
                h_ref_si=h_ref_si,
                dt=dt_units,
                time_units=self.time_units,
                method="exact",
                verbose=self.verbose, 
            )  # (B,H,1)
    
            dt_sec = dt_to_seconds(dt_units, time_units=self.time_units)
            cons_res = cons_step_m / tf_maximum(dt_sec, tf_constant(1e-12, tf_float32))
    
            # ------------------------------------------------------
            # 7) Residuals + priors
            # ------------------------------------------------------
            gw_res = compute_gw_flow_residual(
                self,
                dh_dt=dh_dt,
                d_K_dh_dx_dx=d_K_dh_dx_dx,
                d_K_dh_dy_dy=d_K_dh_dy_dy,
                Ss_field=Ss_field,
                Q=Q_si,
                verbose=verbose,
            )
    
            prior_res = delta_log_tau  # tau consistency prior in log-space
    
            smooth_res = compute_smoothness_prior(
                dK_dx, dK_dy, dSs_dx, dSs_dy,
                K_field=K_field,
                Ss_field=Ss_field,
            )
    
            mv_prior_rms = compute_mv_prior(
                    self,
                    Ss_field,
                    reduction="domain_mean",
                    as_loss=True,
                    verbose=verbose,
                )
            loss_mv = to_rms(mv_prior_rms)
            
            R_H, R_K, R_Ss = compute_bounds_residual(
                self, K_field, Ss_field, H_si, verbose=verbose
            )
            bounds_res = tf_concat([R_H, R_K, R_Ss], axis=-1)
            loss_bounds = tf_reduce_mean(tf_square(bounds_res))
    
            # Physics-off shortcut (keeps returned keys stable)
            if self._physics_off():
                cons_res = tf_zeros_like(cons_res)
                gw_res = tf_zeros_like(gw_res)
                prior_res = tf_zeros_like(prior_res)
                smooth_res = tf_zeros_like(smooth_res)
                loss_mv = tf_zeros_like(mv_prior_rms)
                bounds_res = tf_zeros_like(bounds_res)
                loss_bounds = tf_zeros_like(loss_bounds)
    
            # ------------------------------------------------------
            # 8) Optional residual scaling (nondimensionalization)
            # ------------------------------------------------------
            if (not self._physics_off()) and bool(
                    getattr(self, "scale_pde_residuals", True)):
                scales = compute_scales(
                    self,
                    t=t,
                    s_mean=s_bar,   # physics mean settlement
                    h_mean=h_si,    # latent head
                    K_field=K_field,
                    Ss_field=Ss_field,
                    ds_dt=None,     # v3.2: no ds/dt autodiff
                    tau_field=tau_field,
                    H_field=H_si,
                    h_ref_si=h_ref_si_11,
                    Q=Q_si,
                    verbose=verbose,
                )
                cons_res = scale_residual(cons_res, scales.get("cons_scale"))
                gw_res = scale_residual(gw_res, scales.get("gw_scale"))
    
            # ------------------------------------------------------
            # 9) Physics loss + total loss
            # ------------------------------------------------------
            loss_cons = tf_reduce_mean(tf_square(cons_res))
            loss_gw = tf_reduce_mean(tf_square(gw_res))
            loss_prior = tf_reduce_mean(tf_square(prior_res))
            loss_smooth = tf_reduce_mean(smooth_res)
    
            physics_loss_raw = (
                self.lambda_cons * loss_cons
                + self.lambda_gw * loss_gw
                + self.lambda_prior * loss_prior
                + self.lambda_smooth * loss_smooth
                + self.lambda_mv * loss_mv
                + self.lambda_bounds * loss_bounds
            )
    
            phys_mult = self._physics_loss_multiplier()
            physics_loss_scaled = phys_mult * physics_loss_raw
            total_loss = data_loss + physics_loss_scaled
    
        # ----------------------------------------------------------
        # 10) Apply gradients
        # ----------------------------------------------------------
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        del tape
    
        self.optimizer.apply_gradients(
            self._scale_param_grads(grads, trainable_vars)
        )
    
        # ----------------------------------------------------------
        # 11) Update metrics + logs
        # ----------------------------------------------------------
        self.compiled_metrics.update_state(targets, y_pred)
    
        eps_prior = to_rms(prior_res)
        eps_cons = to_rms(cons_res)
        eps_gw = to_rms(gw_res)
    
        results = {
            m.name: m.result()
            for m in self.metrics
            if m.name not in ("epsilon_prior", "epsilon_cons", "epsilon_gw")
        }
        results.update(
            {
                "loss": total_loss,
                "total_loss": total_loss,
                "data_loss": data_loss,
    
                "physics_loss": physics_loss_raw,
                "physics_mult": phys_mult,
                "physics_loss_scaled": physics_loss_scaled,
                "lambda_offset": self._lambda_offset,
    
                "consolidation_loss": loss_cons,
                "gw_flow_loss": loss_gw,
                "prior_loss": loss_prior,
                "smooth_loss": loss_smooth,
                "mv_prior_loss": loss_mv,
                "bounds_loss": loss_bounds,
    
                "epsilon_prior": eps_prior,
                "epsilon_cons": eps_cons,
                "epsilon_gw": eps_gw,
            }
        )
        return results


    
    def test_step(self, data):
        """
        Validation step (v3.2).
    
        - Computes supervised loss/metrics from `data_final`.
        - Optionally adds physics loss (same weighting scheme as train_step).
        - Uses `_evaluate_physics_on_batch()` for physics diagnostics, which
          internally uses a GradientTape (but no optimizer update here).
        """
        inputs, targets = data
    
        if isinstance(targets, dict):
            targets = rename_dict_keys(
                targets.copy(),
                param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"},
            )
    
        # Forward pass (no optimizer / gradients applied)
        outputs = self(inputs, training=False)
    
        data_final = outputs.get("data_final", None)
        if data_final is None:
            raise ValueError("Model outputs missing 'data_final' in test_step().")
    
        s_pred_final, gwl_pred_final = self.split_data_predictions(data_final)
        y_pred_for_eval = {"subs_pred": s_pred_final, "gwl_pred": gwl_pred_final}
    
        data_loss = self.compiled_loss(
            y_true=targets,
            y_pred=y_pred_for_eval,
            regularization_losses=self.losses,
        )
        self.compiled_metrics.update_state(targets, y_pred_for_eval)
    
        # -------------------------------
        # Physics diagnostics / loss
        # -------------------------------
        if not self._physics_off():
            phys = self._evaluate_physics_on_batch(inputs, return_maps=False)
    
            # phys contains (scaled) component losses:
            #   loss_consolidation, loss_gw_flow, loss_prior, loss_smooth, loss_mv, loss_bounds
            physics_loss_raw = (
                self.lambda_cons * phys["loss_consolidation"]
                + self.lambda_gw * phys["loss_gw_flow"]
                + self.lambda_prior * phys["loss_prior"]
                + self.lambda_smooth * phys["loss_smooth"]
                + self.lambda_mv * phys["loss_mv"]
                + self.lambda_bounds * phys["loss_bounds"]
            )
    
            phys_mult = self._physics_loss_multiplier()
            physics_loss_scaled = phys_mult * physics_loss_raw
    
            total_loss = data_loss + physics_loss_scaled
    
            eps_prior = phys.get("eps_prior", tf_constant(0.0, tf_float32))
            eps_cons = phys.get("eps_cons", tf_constant(0.0, tf_float32))
        else:
            physics_loss_raw = tf_constant(0.0, tf_float32)
            phys_mult = tf_constant(1.0, tf_float32)
            physics_loss_scaled = tf_constant(0.0, tf_float32)
            total_loss = data_loss
            eps_prior = tf_constant(0.0, tf_float32)
            eps_cons = tf_constant(0.0, tf_float32)
    
        # Keep epsilon metrics well-defined if you track them
        if hasattr(self, "eps_prior_metric"):
            self.eps_prior_metric.update_state(eps_prior)
        if hasattr(self, "eps_cons_metric"):
            self.eps_cons_metric.update_state(eps_cons)
    
        results = {
            m.name: m.result()
            for m in self.metrics
            if m.name not in ("epsilon_prior", "epsilon_cons")
        }
    
        results.update(
            {
                "loss": total_loss,  # drives val_loss
                "total_loss": total_loss,
                "data_loss": data_loss,
    
                "physics_loss": physics_loss_raw,
                "physics_mult": phys_mult,
                "physics_loss_scaled": physics_loss_scaled,
                "lambda_offset": self._lambda_offset,
    
                # expose eps for dashboards
                "epsilon_prior": (
                    self.eps_prior_metric.result()
                    if hasattr(self, "eps_prior_metric")
                    else eps_prior
                ),
                "epsilon_cons": (
                    self.eps_cons_metric.result()
                    if hasattr(self, "eps_cons_metric")
                    else eps_cons
                ),
            }
        )
    
        # If you also want component losses in val logs, keep them when physics is on
        if (not self._physics_off()):
            results.update(
                {
                    "consolidation_loss": phys["loss_consolidation"],
                    "gw_flow_loss": phys["loss_gw_flow"],
                    "prior_loss": phys["loss_prior"],
                    "smooth_loss": phys["loss_smooth"],
                    "mv_prior_loss": phys["loss_mv"],
                    "bounds_loss": phys["loss_bounds"],
                }
            )
        else:
            results.update(
                {
                    "consolidation_loss": tf_constant(0.0, tf_float32),
                    "gw_flow_loss": tf_constant(0.0, tf_float32),
                    "prior_loss": tf_constant(0.0, tf_float32),
                    "smooth_loss": tf_constant(0.0, tf_float32),
                    "mv_prior_loss": tf_constant(0.0, tf_float32),
                    "bounds_loss": tf_constant(0.0, tf_float32),
                }
            )
    
        return results

        
    def _evaluate_physics_on_batch(
        self,
        inputs: Dict[str, Optional[Tensor]],
        return_maps: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Physics diagnostics on a single batch (v3.2 / Option-1).
    
        - Mean subsidence is physics-driven (stable exact-step integrator).
        - GW residual supports learnable forcing Q(t,x,y).
        - Residuals enforced in SI; chain-rule corrections applied when coords
          are normalized and/or in degrees.
        """
        eps = tf_constant(1e-12, dtype=tf_float32)
        sk = self.scaling_kwargs or {}
    
        # --------------------------------------------------------------
        # 0) Physics OFF shortcut
        # --------------------------------------------------------------
        if self._physics_off():
            z = tf_constant(0.0, dtype=tf_float32)
            out = dict(
                loss_physics=z,
                loss_consolidation=z,
                loss_gw_flow=z,
                loss_prior=z,
                loss_smooth=z,
                loss_mv=z,
                loss_bounds=z,
                epsilon_prior=z,
                epsilon_cons=z,
                epsilon_gw=z,
            )
            return out
    
        # --------------------------------------------------------------
        # 1) Validate thickness + coords
        # --------------------------------------------------------------
        H_field_in = get_tensor_from(
            inputs, "H_field", "soil_thickness", auto_convert=True
        )
        if H_field_in is None:
            raise ValueError(
                "_evaluate_physics_on_batch() requires 'H_field' "
                "(or 'soil_thickness') in `inputs`."
            )
    
        H_field = tf_convert_to_tensor(H_field_in, dtype=tf_float32)  # model scale
        H_si = to_si_thickness(H_field, sk)                           # SI meters
    
        coords = tf_convert_to_tensor(_get_coords(inputs), tf_float32)
    
        # accept (B,3) -> (B,H,3)
        if coords.shape.rank == 2:
            coords = tf_expand_dims(coords, axis=1)
            coords = tf_tile(coords, [1, self.forecast_horizon, 1])
    
        if coords.shape.rank != 3 or coords.shape[-1] != 3:
            raise ValueError(
                "coords must have shape (B,H,3) (t,x,y). "
                f"Got shape={coords.shape}."
            )
    
        # IMPORTANT: ensure forward uses the same coords tensor
        inputs_fwd = dict(inputs)
        inputs_fwd["coords"] = coords
    
        t = coords[..., 0:1]
    
        coords_norm = bool(sk.get("coords_normalized", False))
        coords_deg = bool(sk.get("coords_in_degrees", False))
    
        # --------------------------------------------------------------
        # 2) Forward + autodiff wrt coords
        # --------------------------------------------------------------
        with GradientTape(persistent=True) as tape:
            tape.watch(coords)
    
            outputs = self(inputs_fwd, training=False)
    
            data_mean_raw = outputs.get("data_mean_raw", None)
            if data_mean_raw is None:
                raise ValueError("Model outputs missing 'data_mean_raw'.")
    
            subs_mean_raw, gwl_mean_raw = self.split_data_predictions(data_mean_raw)
    
            # predicted gwl/depth -> SI head (meters)
            gwl_si = to_si_head(tf_cast(gwl_mean_raw, tf_float32), sk)
            h_si = gwl_to_head_m(gwl_si, sk, inputs=inputs_fwd)
    
            phys_mean_raw = outputs.get("phys_mean_raw", None)
            if phys_mean_raw is None:
                raise ValueError("Model outputs missing 'phys_mean_raw'.")
    
            # Option-1 phys head layout: [K, Ss, dlogtau, Q] (B,H,4)
            ( 
                K_logits, 
                Ss_logits, 
                tau_logits, 
                Q_logits 
            )= self.split_physics_predictions(phys_mean_raw)

            # recommended: keep K,Ss,tau spatial-only (average over time then broadcast)
            K_base = tf_broadcast_to(
                tf_reduce_mean(K_logits, axis=1, keepdims=True),
                tf_shape(K_logits),
            )
            Ss_base = tf_broadcast_to(
                tf_reduce_mean(Ss_logits, axis=1, keepdims=True),
                tf_shape(Ss_logits),
            )
            tau_base = tf_broadcast_to(
                tf_reduce_mean(tau_logits, axis=1, keepdims=True),
                tf_shape(tau_logits),
            )
    
            (
                K_field, Ss_field, tau_field, tau_phys, Hd_eff,
                delta_log_tau, logK, logSs,
            ) = compose_physics_fields(
                self,
                coords_flat=coords,
                H_si=H_si,
                K_base=K_base,
                Ss_base=Ss_base,
                tau_base=tau_base,
                training=False,
                verbose=0,
            )
    
            # --- 1st derivatives
            dh_dcoords = tape.gradient(h_si, coords)
            if dh_dcoords is None:
                raise ValueError(
                    "dh_dcoords is None. Ensure self() consumes inputs['coords']."
                )
    
            dh_dt = dh_dcoords[..., 0:1]
            dh_dx = dh_dcoords[..., 1:2]
            dh_dy = dh_dcoords[..., 2:3]
    
            # --- divergence of (K * grad h)
            K_dh_dx = K_field * dh_dx
            K_dh_dy = K_field * dh_dy
    
            dKdhx_dcoords = tape.gradient(K_dh_dx, coords)
            dKdhy_dcoords = tape.gradient(K_dh_dy, coords)
    
            if (dKdhx_dcoords is None) or (dKdhy_dcoords is None):
                raise ValueError("Second-order PDE gradients are None.")
    
            d_K_dh_dx_dx = dKdhx_dcoords[..., 1:2]
            d_K_dh_dy_dy = dKdhy_dcoords[..., 2:3]
    
            # smoothness grads
            dK_dcoords = tape.gradient(K_field, coords)
            dSs_dcoords = tape.gradient(Ss_field, coords)
            if (dK_dcoords is None) or (dSs_dcoords is None):
                raise ValueError("K/Ss spatial gradients are None.")
    
            dK_dx = dK_dcoords[..., 1:2]
            dK_dy = dK_dcoords[..., 2:3]
            dSs_dx = dSs_dcoords[..., 1:2]
            dSs_dy = dSs_dcoords[..., 2:3]
    
        del tape
    
        # --------------------------------------------------------------
        # 3) Chain-rule corrections
        # --------------------------------------------------------------
        if coords_norm:
            tR, xR, yR = coord_ranges(sk)
            if (tR is None) or (xR is None) or (yR is None):
                raise ValueError("coords_normalized=True but coord_ranges missing.")
            tR = tf_constant(float(tR), tf_float32)
            xR = tf_constant(float(xR), tf_float32)
            yR = tf_constant(float(yR), tf_float32)
    
            dh_dt = dh_dt / (tR + eps)
            dh_dx = dh_dx / (xR + eps)
            dh_dy = dh_dy / (yR + eps)
    
            d_K_dh_dx_dx = d_K_dh_dx_dx / (xR * xR + eps)
            d_K_dh_dy_dy = d_K_dh_dy_dy / (yR * yR + eps)
    
            dK_dx = dK_dx / (xR + eps)
            dK_dy = dK_dy / (yR + eps)
            dSs_dx = dSs_dx / (xR + eps)
            dSs_dy = dSs_dy / (yR + eps)
    
        if coords_deg:
            deg2m_x = deg_to_m("x", sk)
            deg2m_y = deg_to_m("y", sk)
    
            dh_dx = dh_dx / (deg2m_x + eps)
            dh_dy = dh_dy / (deg2m_y + eps)
    
            d_K_dh_dx_dx = d_K_dh_dx_dx / (deg2m_x * deg2m_x + eps)
            d_K_dh_dy_dy = d_K_dh_dy_dy / (deg2m_y * deg2m_y + eps)
    
            dK_dx = dK_dx / (deg2m_x + eps)
            dK_dy = dK_dy / (deg2m_y + eps)
            dSs_dx = dSs_dx / (deg2m_x + eps)
            dSs_dy = dSs_dy / (deg2m_y + eps)
    
        dh_dt = rate_to_per_second(dh_dt, time_units=self.time_units)
    
        # --------------------------------------------------------------
        # 4) Q -> SI (1/s)  (branch-free: Q_logits is always a tensor)
        # --------------------------------------------------------------
        Q_base = tf_cast(Q_logits, tf_float32)
        
        if coords_norm and bool(sk.get("Q_wrt_normalized_time", False)):
            tR, _, _ = coord_ranges(sk)
            if tR is None:
                raise ValueError(
                    "Q_wrt_normalized_time=True but coord_range_t missing."
                )
            Q_base = Q_base / (tf_constant(float(tR), tf_float32) + eps)
        
        if bool(sk.get("Q_in_per_second", False)) or bool(sk.get("Q_in_si", False)):
            Q_si = Q_base
        else:
            Q_si = rate_to_per_second(Q_base, time_units=self.time_units)
        
        # Ensure (B,H,1) and broadcastable with dh_dt
        r = tf_rank(Q_si)
        Q_si = tf_cond(
            tf_equal(r, 2),
            lambda: tf_reshape(Q_si, [tf_shape(Q_si)[0], tf_shape(Q_si)[1], 1]),
            lambda: Q_si,
        )
        # now broadcast with dh_dt
        Q_si = Q_si + tf_zeros_like(dh_dt)
        # --------------------------------------------------------------
        # 5) Physics-driven mean settlement + consolidation residual
        # --------------------------------------------------------------
        dt_units = infer_dt_units_from_t(t, sk)
        dt_sec = dt_to_seconds(dt_units, time_units=self.time_units)
    
        s_init_si = get_s_init_si(self, inputs_fwd, like=h_si[:, :1, :])     # (B,1,1)
        h_ref_si_11 = get_h_ref_si(self, inputs_fwd, like=h_si[:, :1, :])    # (B,1,1)
        h_ref_si = h_ref_si_11 + tf_zeros_like(h_si)                         # (B,H,1)
    
        s_bar = integrate_consolidation_mean(
            h_mean_si=h_si,
            Ss_field=Ss_field,
            H_field_si=H_si,
            tau_field=tau_field,
            h_ref_si=h_ref_si,
            s_init_si=s_init_si,
            dt=dt_units,
            time_units=self.time_units,
            method=self.residual_method,
            verbose= self.verbose, 
        )
    
        s_state = tf_concat([s_init_si, s_bar], axis=1)          # (B,H+1,1)
        h_state = tf_concat([h_ref_si_11, h_si], axis=1)         # (B,H+1,1)
    
        cons_step_m = compute_consolidation_step_residual(
            s_state_si=s_state,
            h_mean_si=h_state,
            Ss_field=Ss_field,
            H_field_si=H_si,
            tau_field=tau_field,
            h_ref_si=h_ref_si,
            dt=dt_units,
            time_units=self.time_units,
            method=self.residual_method,
            verbose= self.verbose, 
        )
    
        cons_res = cons_step_m / tf_maximum(dt_sec, eps)         # (B,H,1) in m/s
    
        # --------------------------------------------------------------
        # 6) GW residual + priors
        # --------------------------------------------------------------
        gw_res = compute_gw_flow_residual(
            self,
            dh_dt,
            d_K_dh_dx_dx,
            d_K_dh_dy_dy,
            Ss_field,
            Q=Q_si,
            verbose=0,
        )
    
        prior_res = delta_log_tau
    
        smooth_res = compute_smoothness_prior(
            dK_dx, dK_dy, dSs_dx, dSs_dy,
            K_field=K_field,
            Ss_field=Ss_field,
        )
    
        mv_res = compute_mv_prior(
            self, Ss_field, 
            reduction="domain_mean", 
            as_loss=False, 
            verbose=0
            )
        
        R_H, R_K, R_Ss = compute_bounds_residual(
            self, K_field, Ss_field, H_si, verbose=0
        )
        bounds_res = tf_concat([R_H, R_K, R_Ss], axis=-1)
    
        # --------------------------------------------------------------
        # 7) Optional nondimensionalization
        # --------------------------------------------------------------
        if bool(getattr(self, "scale_pde_residuals", True)):
            scales = compute_scales(
                self,
                t=t,
                s_mean=s_bar,
                h_mean=h_si,
                K_field=K_field,
                Ss_field=Ss_field,
                ds_dt=None,
                tau_field=tau_field,
                H_field=H_si,
                h_ref_si=h_ref_si_11,
                Q=Q_si,
                verbose=0,
            )
            cons_res_scaled = scale_residual(cons_res, scales.get("cons_scale", None))
            gw_res_scaled = scale_residual(gw_res, scales.get("gw_scale", None))
        else:
            cons_res_scaled = cons_res
            gw_res_scaled = gw_res
    
        # --------------------------------------------------------------
        # 8) Losses + epsilons
        # --------------------------------------------------------------
        loss_cons = tf_reduce_mean(tf_square(cons_res_scaled))
        loss_gw = tf_reduce_mean(tf_square(gw_res_scaled))
        loss_prior = tf_reduce_mean(tf_square(prior_res))
        loss_smooth = tf_reduce_mean(smooth_res)
        loss_mv = to_rms (mv_res)
        loss_bounds = tf_reduce_mean(tf_square(bounds_res))
    
        eps_prior = to_rms(prior_res)
        eps_cons = to_rms(cons_res)
        eps_gw = to_rms(gw_res)
    
        out = {
            "loss_physics": loss_cons + loss_gw,
            "loss_consolidation": loss_cons,
            "loss_gw_flow": loss_gw,
            "loss_prior": loss_prior,
            "loss_smooth": loss_smooth,
            "loss_mv": loss_mv,
            "loss_bounds": loss_bounds,
            "epsilon_prior": eps_prior,
            "epsilon_cons": eps_cons,
            "epsilon_gw": eps_gw,
        }
    
        if return_maps:
            out.update(
                {
                    "h_si": h_si,
                    "h_ref_si": h_ref_si,
                    "s_bar": s_bar,
                    "s_init_si": s_init_si,
                    "dt_units": dt_units,
                    "dt_sec": dt_sec,
                    "Q_si": Q_si,
                    "K_field": K_field,
                    "Ss_field": Ss_field,
                    "tau_field": tau_field,
                    "tau_phys": tau_phys,
                    "Hd_eff": Hd_eff,
                    "R_cons": cons_res, 
                    "R_gw": gw_res,
                    "R_prior": prior_res,
                    "R_smooth": smooth_res,
                    "R_mv": mv_res,
                    "R_bounds": bounds_res,
                    "R_cons_scaled": cons_res_scaled,
                    "R_gw_scaled": gw_res_scaled,
                    "subs_mean_raw": subs_mean_raw,
                    "gwl_mean_raw": gwl_mean_raw,
                    
                    "K": K_field,               # effective K (m/s)
                    "Ss": Ss_field,             # effective Ss (1/m)
                    "tau": tau_field,           # learned tau (s)
                    "tau_prior": tau_phys,      # closure tau (s)
                    "tau_closure": tau_phys,    # alias (clearer naming)
                    
                    "Hd": Hd_eff,  # effective drainage thickness (m)
                    "H": H_si,    # base thickness (m)
                    "H_field": H_si,  # legacy name used elsewhere
                    "cons_res_vals": cons_res,  # alias
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
        """
        Evaluate physics diagnostics.
    
        - If `inputs` is a Dataset, we aggregate ONLY scalar diagnostics
          (loss_* and epsilon_*) across batches.
        - If `return_maps=True`, we also return physics maps from the LAST
          processed batch (keys in MAP_KEYS).
        """
        MAP_KEYS = (
            "R_prior",
            "R_cons",
            "R_gw",
            "K",
            "Ss",
            "H_field",
            "Hd",
            "H",
            "tau",
            "tau_prior",
            "tau_closure",
        )
    
        SCALAR_PREFIXES = ("loss_", "epsilon_")
    
        if isinstance(inputs, Dataset):
            acc: Dict[str, List[Tensor]] = {}
            last_maps: Optional[Dict[str, Tensor]] = None
    
            for i, elem in enumerate(inputs):
                xb = elem[0] if isinstance(elem, (tuple, list)) else elem
                out_b = self._evaluate_physics_on_batch(
                    xb,
                    return_maps=return_maps,
                )
    
                # Accumulate ONLY scalar diagnostics across batches.
                for k, v in out_b.items():
                    if k.startswith(SCALAR_PREFIXES):
                        acc.setdefault(k, []).append(v)
    
                # Keep maps from the last batch only (if requested).
                if return_maps:
                    last_maps = {
                        k: out_b[k] 
                        for k in MAP_KEYS 
                        if k in out_b
                    }
    
                if max_batches is not None and (i + 1) >= max_batches:
                    break
    
            out = {k: tf_reduce_mean(tf_stack(vs)) for k, vs in acc.items()}
            if return_maps and last_maps is not None:
                out.update(last_maps)
            return out
    
        # If user passed numpy-like arrays (no tensors) 
        # + batch_size, wrap into Dataset.
        if isinstance(inputs, Mapping) and batch_size is not None:
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
    
        return self._evaluate_physics_on_batch(
            inputs,
            return_maps=return_maps,
        )

    def _physics_loss_multiplier(self) -> Tensor:
        # If physics is off, multiplier is irrelevant; keep neutral.
        if self._physics_off():
            return tf_constant(1.0, dtype=tf_float32)
    
        mode = self.offset_mode  # <-- correct name
    
        if mode == "mul":
            # optional safety (graph-safe):
            tf_debugging.assert_greater(
                self._lambda_offset,
                tf_constant(0.0, tf_float32),
                message="lambda_offset must be > 0 when offset_mode='mul'.",
            )
            return tf_identity(self._lambda_offset)
    
        if mode == "log10":
            return tf_pow(
                tf_constant(10.0, dtype=tf_float32),
                tf_identity(self._lambda_offset),
            )
    
        # This should be impossible due to validate_params
        raise ValueError(
            f"Invalid offset_mode={mode!r}. Expected 'mul' or 'log10'."
        )

    def _mv_value(self) -> Tensor:
        r"""
        Return the current value of :math:`m_v` in linear space.
    
        If :math:`m_v` is learnable, this is ``exp(log_mv)``; otherwise
        it is the fixed constant ``_mv_fixed``.
    
        Returns
        -------
        tf.Tensor
            Scalar tensor (0D) representing :math:`m_v > 0`.
        """
        return (
            tf_exp(self.log_mv)
            if hasattr(self, "log_mv")
            else self._mv_fixed
        )
    
    
    def _kappa_value(self) -> Tensor:
        r"""
        Return the current value of :math:`\kappa` in linear space.
    
        If :math:`\kappa` is learnable, this is ``exp(log_kappa)``;
        otherwise it is the fixed constant ``_kappa_fixed``.
    
        Returns
        -------
        tf.Tensor
            Scalar tensor (0D) representing :math:`\kappa > 0`.
        """
        return (
            tf_exp(self.log_kappa)
            if hasattr(self, "log_kappa")
            else self._kappa_fixed
        )

    def current_mv(self):
        r"""
        Return the current value of the compressibility :math:`m_v`.
    
        This is a thin convenience wrapper around :meth:`_mv_value`,
        which handles both the trainable (log-parameterized) and
        fixed-scalar cases.
    
        Returns
        -------
        tf.Tensor
            Scalar tensor representing :math:`m_v` in linear space.
        """
        return self._mv_value()
    
    
    def current_kappa(self):
        r"""
        Return the current value of the consistency coefficient
        :math:`\kappa`.
    
        This is a thin convenience wrapper around :meth:`_kappa_value`,
        which handles both the trainable (log-parameterized) and
        fixed-scalar cases.
    
        Returns
        -------
        tf.Tensor
            Scalar tensor representing :math:`\kappa` in linear space.
        """
        return self._kappa_value()


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


    def export_physics_payload(
        self,
        dataset,
        max_batches=None,
        save_path=None,
        format: str = "npz",
        overwrite: bool = False,
        metadata=None,
        random_subsample=None,
        float_dtype=np.float32,
        log_fn =None, 
        **tqdm_kws
    ):

        payload = gather_physics_payload(
            self,
            dataset,
            max_batches=max_batches,
            float_dtype=float_dtype,
            log_fn=log_fn, 
            **tqdm_kws
        )
    
        if random_subsample is not None:
            payload = _maybe_subsample(payload, random_subsample)
    
        if save_path is not None:
            meta = default_meta_from_model(self)
            if metadata:
                meta.update(metadata)
            save_physics_payload(
                payload,
                meta,
                save_path,
                format=format,
                overwrite=overwrite,
                log_fn=log_fn,
            )
    
        return payload
    
    
    @staticmethod
    def load_physics_payload(path):
        return load_physics_payload(path)
    
    def split_data_predictions(
        self,
        data_tensor: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        s_pred = data_tensor[..., : self.output_subsidence_dim]
        gwl_pred = data_tensor[..., self.output_subsidence_dim :]
    
        return s_pred, gwl_pred
    
    def split_physics_predictions(
        self,
        phys_means_raw_tensor: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Slice phys_mean_raw -> (K_logits, Ss_logits, dlogtau_logits, Q_logits)
    
        Notes
        -----
        - If self.output_Q_dim == 0, returns Q_logits as zeros with shape
          matching (B,H,1) so downstream code never needs branching.
        - If phys_means_raw_tensor does not contain Q (backward compatibility),
          returns zeros for Q as well.
        """
        start = 0
    
        K_logits = phys_means_raw_tensor[..., start:start + self.output_K_dim]
        start += self.output_K_dim
    
        Ss_logits = phys_means_raw_tensor[..., start:start + self.output_Ss_dim]
        start += self.output_Ss_dim
    
        dlogtau_logits = phys_means_raw_tensor[..., start:start + self.output_tau_dim]
        start += self.output_tau_dim
    
        # ---- Q: always return a tensor (B,H,1) ----
        q_dim = int(getattr(self, "output_Q_dim", 0) or 0)
    
        # If Q is disabled, force a zeros tensor shaped like (B,H,1)
        if q_dim <= 0:
            Q_logits = tf_zeros_like(K_logits[..., :1])
            return K_logits, Ss_logits, dlogtau_logits, Q_logits
    
        # If Q is enabled but phys_mean_raw doesn't have it, fallback to zeros
        end = start + q_dim
        n_phys = tf_shape(phys_means_raw_tensor)[-1]
        q_shape = tf_concat(
            [tf_shape(phys_means_raw_tensor)[:-1], tf_constant([q_dim], tf_int32)],
            axis=0,
        )
        Q_fallback = tf_zeros(q_shape, dtype=phys_means_raw_tensor.dtype)
    
        Q_logits = tf_cond(
            tf_greater_equal(n_phys, tf_constant(end, tf_int32)),
            lambda: phys_means_raw_tensor[..., start:end],
            lambda: Q_fallback,
        )
    
        # (Optional safety) if q_dim != 1 but you still want (B,H,1) everywhere:
        # Q_logits = Q_logits[..., :1]
    
        return K_logits, Ss_logits, dlogtau_logits, Q_logits

    def _scale_param_grads(self, grads, trainable_vars):

        scaled = []
        mv_var = getattr(self, "log_mv", None)
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
        r"""
        Return ``True`` if physics constraints are effectively disabled.
    
        Physics is considered "off" when ``pde_modes_active`` is a
        list/tuple containing the sentinel value ``"none"``. In that
        case:
    
        * PDE residuals are short-circuited to zero, and
        * physics loss weights are forced to zero in :meth:`compile`.
    
        Returns
        -------
        bool
            ``True`` if PDE constraints should not contribute to the
            loss; ``False`` otherwise.
        """
        return isinstance(self.pde_modes_active, (list, tuple)) and (
            "none" in self.pde_modes_active
        )

    @property
    def lambda_offset_value(self) -> float:
        """Current raw value stored in the TF weight ``_lambda_offset``."""
        try:
            return float(self._lambda_offset.numpy())
        except:
            return float(self._lambda_offset)

    @property
    def lambda_offset(self) -> float:
        return float(self._lambda_offset.numpy())
    
    @lambda_offset.setter
    def lambda_offset(self, value: float) -> None:
        self._lambda_offset.assign(float(value))

    @property
    def mv_lr_mult(self) -> float:
        r"""
        Learning-rate multiplier for :math:`m_v` (via ``log_mv``).
    
        This factor multiplies the gradient of the log-parameter
        ``log_mv`` inside :meth:`_scale_param_grads`, allowing
        :math:`m_v` to learn faster or slower than the rest of the
        network.
    
        Returns
        -------
        float
            Current value of the multiplier for ``log_mv``.
        """
        return self._mv_lr_mult
    
    
    @property
    def kappa_lr_mult(self) -> float:
        r"""
        Learning-rate multiplier for :math:`\kappa` (via ``log_kappa``).
    
        This factor multiplies the gradient of the log-parameter
        ``log_kappa`` inside :meth:`_scale_param_grads`, allowing
        :math:`\kappa` to learn at a different pace than the other
        parameters.
    
        Returns
        -------
        float
            Current value of the multiplier for ``log_kappa``.
        """
        return self._kappa_lr_mult

    def compile(
        self,
        lambda_cons: float = 1.0,
        lambda_gw: float = 1.0,
        lambda_prior: float = 1.0,
        lambda_smooth: float = 1.0,
        lambda_mv: float = 0.0,
        lambda_bounds: float = 0.0,
        lambda_offset: float = 1.0,
        mv_lr_mult: float = 1.0,
        kappa_lr_mult: float = 1.0,
        **kwargs,
    ):
        r"""
        Compile the model and configure data/physics loss weighting.
    
        This override extends :meth:`tf.keras.Model.compile` by exposing
        explicit weights for each physics term used in the PINN objective,
        plus a **global physics multiplier** (``lambda_offset``) applied
        uniformly to the *sum* of all physics losses.
    
        In this implementation, the physics objective is assembled as:
    
        .. math::
    
            L_\text{phys}
            = \lambda_\text{cons} L_\text{cons}
            + \lambda_\text{gw}   L_\text{gw}
            + \lambda_\text{prior} L_\text{prior}
            + \lambda_\text{smooth} L_\text{smooth}
            + \lambda_{m_v} L_{m_v}
            + \lambda_\text{bounds} L_\text{bounds},
    
        and the total training loss used by ``train_step`` is:
    
        .. math::
    
            L
            = L_\text{data}
            + \alpha(\text{offset\_mode}, \lambda_\text{offset})
              \, L_\text{phys},
    
        where :math:`\alpha(\cdot)` is computed by
        :meth:`_physics_loss_multiplier` according to ``self.offset_mode``:
    
        * ``offset_mode="mul"``:
          :math:`\alpha = \lambda_\text{offset}` (must be :math:`>0`)
        * ``offset_mode="log10"``:
          :math:`\alpha = 10^{\lambda_\text{offset}}`
          (e.g. ``0`` :math:`\to 1`, ``1`` :math:`\to 10`, ``-1`` :math:`\to 0.1`)
    
        ``lambda_offset`` is stored as a **non-trainable TF weight**
        ``self._lambda_offset`` (created with ``add_weight(trainable=False)``),
        making it safe to change during training via callbacks using:
    
        ``self.model._lambda_offset.assign(value)``.
    
        Parameters
        ----------
        lambda_cons : float, default=1.0
            Weight for the consolidation residual loss :math:`L_\text{cons}`
            associated with :math:`R_\text{cons}`.
    
        lambda_gw : float, default=1.0
            Weight for the groundwater-flow residual loss :math:`L_\text{gw}`
            associated with :math:`R_\text{gw}`.
    
        lambda_prior : float, default=1.0
            Weight for the consistency prior loss :math:`L_\text{prior}`
            linking :math:`\tau`, :math:`K`, :math:`S_s`, and :math:`H`.
    
        lambda_smooth : float, default=1.0
            Weight for the smoothness prior loss :math:`L_\text{smooth}`,
            based on :math:`\|\nabla K\|^2 + \|\nabla S_s\|^2`.
    
        lambda_mv : float, default=0.0
            Weight for the storage-identity penalty :math:`L_{m_v}` derived
            from:
    
            .. math::
    
                R_{m_v} = \log(S_s) - \log(m_v \gamma_w).
    
            This term provides a direct gradient signal for :math:`m_v`.
            Set this to a positive value to inform :math:`m_v` from the
            predicted :math:`S_s`.
    
        lambda_bounds : float, default=0.0
            Weight for the soft-constraints/bounds penalty
            :math:`L_\text{bounds}` (e.g., on :math:`H`, :math:`\log K`,
            :math:`\log S_s` as configured in ``scaling_kwargs``).
    
        lambda_offset : float, default=1.0
            Global physics scaling parameter stored in ``self._lambda_offset``.
            The effective multiplier applied to the total physics loss is
            determined by ``self.offset_mode`` (see above). Use this to
            globally raise/lower the influence of all physics terms without
            changing each individual ``lambda_*``.
    
            If you need *scheduling* (warmup/ramp), prefer a callback
            (e.g., ``LambdaOffsetScheduler``) that calls
            ``self.model._lambda_offset.assign(value)``.
    
        mv_lr_mult : float, default=1.0
            Learning-rate multiplier applied only to the gradient of
            ``log_mv`` (the log-parameter of :math:`m_v`).
    
        kappa_lr_mult : float, default=1.0
            Learning-rate multiplier applied only to the gradient of
            ``log_kappa`` (the log-parameter of :math:`\kappa`).
    
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`tf.keras.Model.compile`, such as ``optimizer``, ``loss``,
            ``metrics``, ``jit_compile``, etc.
    
        Notes
        -----
        * When :meth:`_physics_off` is ``True`` (e.g., PDE modes contain
          ``"none"``), the physics weights
          ``lambda_prior``, ``lambda_smooth``, ``lambda_mv`` and
          ``lambda_bounds`` are hard-set to ``0.0`` regardless of the values
          passed here, and ``self._lambda_offset`` is set to ``1.0`` (neutral).
        * ``lambda_offset`` is validated once per run in Python. For
          ``offset_mode="mul"``, it must be strictly positive.
        * The data loss weight is implicitly ``1.0`` and is defined by the
          loss passed in ``**kwargs``.
        """
        # Let the base class set up optimizer/loss/metrics first.
        super().compile(**kwargs)
    
        # Store core physics weights.
        self.lambda_cons = float(lambda_cons)
        self.lambda_gw = float(lambda_gw)
    
        if self._physics_off():
            # When physics is off, hard-disable these contributions.
            self.lambda_prior = 0.0
            self.lambda_smooth = 0.0
            self.lambda_mv = 0.0
            self.lambda_bounds = 0.0
    
            # Keep neutral; avoids any assertion trouble and keeps logs stable.
            self._lambda_offset.assign(1.0)
        else:
            self.lambda_prior = float(lambda_prior)
            self.lambda_smooth = float(lambda_smooth)
            self.lambda_mv = float(lambda_mv)
            self.lambda_bounds = float(lambda_bounds)
            if self.bounds_mode == "hard":
                self.lambda_bounds = 0.0
    
            # Validate once, in Python, per run.
            lo = float(lambda_offset)
            if self.offset_mode == "mul" and lo <= 0.0:
                raise ValueError(
                    "lambda_offset must be > 0 when offset_mode='mul'."
                )
            self._lambda_offset.assign(lo)
    
        # Per-parameter LR multipliers for log_mv and log_kappa.
        self._mv_lr_mult = float(mv_lr_mult)
        self._kappa_lr_mult = float(kappa_lr_mult)


    def get_config(self) -> dict:
        r"""
        Return the full serializable configuration of the model.
    
        This extends :meth:`BaseAttentive.get_config` with additional
        physics-related knobs so that :class:`GeoPriorSubsNet` can be
        saved and restored with :func:`tf.keras.models.clone_model`,
        :func:`tf.keras.models.model_from_json`, or the Keras SavedModel
        machinery.
    
        Returns
        -------
        dict
            A configuration dictionary suitable for reconstruction via
            :meth:`from_config`. It includes:
    
            * all keys from :meth:`BaseAttentive.get_config`,
            * PINN-specific entries:
    
              - ``"output_subsidence_dim"``,
              - ``"output_gwl_dim"``,
              - ``"pde_mode"`` (active PDE modes),
              - ``"mv"`` (possibly a serialized :class:`LearnableMV`),
              - ``"kappa"`` (possibly a serialized :class:`LearnableKappa`),
              - ``"gamma_w"`` and ``"h_ref"`` (fixed physics scalars),
              - ``"scale_pde_residuals"`` and ``"scaling_kwargs"``,
    
            * a lightweight ``"model_version"`` tag for manual inspection.
        """
        base_config = super().get_config()
    
        pinn_config = {
            "output_subsidence_dim": self.output_subsidence_dim,
            "output_gwl_dim": self.output_gwl_dim,
            # Note: pde_modes_active is typically a list; we store it as-is.
            "pde_mode": self.pde_modes_active,
            # These may be wrapper objects (LearnableMV, LearnableKappa,
            # FixedGammaW, FixedHRef) which Keras knows how to serialize.
            "mv": self.mv_config,
            "kappa": self.kappa_config,
            "gamma_w": self.gamma_w_config,
            "h_ref": self.h_ref_config,
            "scale_pde_residuals": self.scale_pde_residuals,
            "time_units": self.time_units,      
            "scaling_kwargs": self.scaling_kwargs,
            #  keep the effective thickness / κ-mode knobs
            "use_effective_h": self.use_effective_thickness,
            "hd_factor": self.Hd_factor,
            "offset_mode": self.offset_mode, 
            "kappa_mode": self.kappa_mode,
            "bounds_mode": self.bounds_mode,
            "residual_method": self.residual_method, 
            "verbose": self.verbose, 
            "model_version": "3.2-GeoPrior",
        }
    
        base_config.update(pinn_config)
        # For backward compatibility with BaseAttentive, keep the original
        # data output dimension visible.
        base_config["output_dim"] = self._data_output_dim
    
        return base_config


    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        r"""
        Reconstruct a :class:`GeoPriorSubsNet` instance from config.
    
        This is the inverse of :meth:`get_config` and is used by the
        Keras serialization utilities. It also handles two important
        concerns:
    
        * Re-hydrating physics parameter wrappers (e.g. ``LearnableMV``)
          from their serialized dict form.
        * Gracefully dropping legacy or intermediate keys that no longer
          belong in the public constructor (older SavedModels / configs).
    
        Parameters
        ----------
        config : dict
            Configuration dictionary produced by :meth:`get_config`,
            possibly extended by Keras when saving/loading. May contain
            serialized versions of ``mv``, ``kappa``, ``gamma_w``,
            ``h_ref`` as nested dicts with ``"class_name"`` and
            ``"config"`` fields.
    
        custom_objects : dict or None, default=None
            Optional mapping of string names to custom classes or
            functions. If ``None``, a fresh dictionary is created and
            populated with the physics parameter wrappers required to
            deserialize the config (e.g. :class:`LearnableMV`,
            :class:`LearnableKappa`, :class:`FixedGammaW`,
            :class:`FixedHRef`, etc.).
    
        Returns
        -------
        GeoPriorSubsNet
            A newly constructed model instance with weights uninitialized
            (to be loaded separately if needed).
    
        Notes
        -----
        * Any legacy keys that refer to older model variants
          (e.g. explicit ``"K"``, ``"Ss"`` or generic
          ``"pinn_coefficient_C"``) are removed to keep the constructor
          signature clean.
        * The ``"output_dim"`` key is also removed because
          :class:`GeoPriorSubsNet` derives it from
          ``output_subsidence_dim`` and ``output_gwl_dim``.
        """
        if custom_objects is None:
            custom_objects = {}
    
        # Register all physics parameter wrappers so that Keras'
        # deserialize_keras_object() can rebuild them from their
        # serialized representation.
        custom_objects.update(
            {
                "LearnableMV": LearnableMV,
                "LearnableKappa": LearnableKappa,
                "FixedGammaW": FixedGammaW,
                "FixedHRef": FixedHRef,
                "LearnableK": LearnableK,
                "LearnableSs": LearnableSs,
                "LearnableQ": LearnableQ,
                "LearnableC": LearnableC,
                "FixedC": FixedC,
                "DisabledC": DisabledC,
            }
        )
    
        # Re-hydrate scalar physics configs if they were serialized as
        # Keras objects (with {"class_name": ..., "config": ...}).
        for key in ("mv", "kappa", "gamma_w", "h_ref"):
            obj = config.get(key)
            if isinstance(obj, dict) and "class_name" in obj:
                config[key] = deserialize_keras_object(
                    obj,
                    custom_objects=custom_objects,
                )
    
        # Drop legacy / internal keys that should not be passed to
        # the public __init__. This keeps backwards compatibility with
        # older SavedModels while avoiding unexpected kwargs.
        config.pop("K", None)
        config.pop("Ss", None)
        config.pop("Q", None)
        config.pop("pinn_coefficient_C", None)
        config.pop("gw_flow_coeffs", None)
        config.pop("output_dim", None)
        config.pop("model_version", None)
    
        return cls(**config)

