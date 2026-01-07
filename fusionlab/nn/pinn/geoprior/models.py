# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...._fusionlog import OncePerMessageFilter, fusionlog
from ....api.docs import DocstringComponents, _halnet_core_params
from ....compat.sklearn import Interval, StrOptions, validate_params
from ....params import (
    DisabledC,
    FixedC,
    FixedGammaW,
    FixedHRef,
    LearnableC,
    LearnableK,
    LearnableKappa,
    LearnableMV,
    LearnableQ,
    LearnableSs,
)

from ... import KERAS_DEPS, dependency_message
from ..._base_attentive import BaseAttentive
from ..._tensor_validation import check_inputs, validate_model_inputs
from ...components import (
    aggregate_multiscale_on_3d,
    aggregate_time_window_output,
)
from ...custom_metrics import GeoPriorTrackers

from ..op import process_pinn_inputs
from ..utils import process_pde_modes

from .batch_io import _align_true_for_loss, _canonicalize_targets
from .debugs import (
    dbg_call_nonfinite,
    dbg_step0_inputs_targets,
    dbg_step10_grads,
    dbg_step9_losses,
    dbg_term_grads_finite,
)
from .losses import pack_step_results
from .maths import (
    _EPSILON,
    LogClipConstraint,
    compose_physics_fields,
    get_log_bounds,
    integrate_consolidation_mean,
    resolve_cons_drawdown_options,
    tf_print_nonfinite,
)
from .payloads import (
    _maybe_subsample,
    default_meta_from_model,
    gather_physics_payload,
    load_physics_payload,
    save_physics_payload,
)
from .stability import filter_nan_gradients
from .step_core import physics_core
from .utils import (
    canonicalize_scaling_kwargs,
    enforce_scaling_alias_consistency,
    from_si_subsidence,
    get_h_ref_si,
    get_s_init_si,
    get_sk,
    gwl_to_head_m,
    infer_dt_units_from_t,
    load_scaling_kwargs,
    policy_gate,
    to_si_head,
    to_si_thickness,
    validate_scaling_kwargs,
)

# ---------------------------------------------------------------------
# Keras deps aliases (keep short lines, stable across backends)
# ---------------------------------------------------------------------
K = KERAS_DEPS

LSTM = K.LSTM
Dense = K.Dense
LayerNormalization = K.LayerNormalization
Sequential = K.Sequential
InputLayer = K.InputLayer
Model = K.Model
Tensor = K.Tensor
Variable = K.Variable
Add = K.Add
Constant = K.Constant
GradientTape = K.GradientTape
Mean = K.Mean
Dataset = K.Dataset
RandomNormal = K.RandomNormal

tf_abs = K.abs
tf_add_n = K.add_n
tf_broadcast_to = K.broadcast_to
tf_cast = K.cast
tf_clip_by_global_norm = K.clip_by_global_norm
tf_clip_by_value = K.clip_by_value
tf_concat = K.concat
tf_cond = K.cond
tf_constant = K.constant
tf_convert_to_tensor = K.convert_to_tensor
tf_debugging = K.debugging
tf_equal = K.equal
tf_exp = K.exp
tf_expand_dims = K.expand_dims
tf_float32 = K.float32
tf_float64 = K.float64
tf_greater = K.greater
tf_greater_equal = K.greater_equal
tf_identity = K.identity
tf_int32 = K.int32
tf_log = K.log
tf_math = K.math
tf_maximum = K.maximum
tf_nn = K.nn
tf_ones = K.ones
tf_pow = K.pow
tf_print = K.print
tf_rank = K.rank
tf_reduce_all = K.reduce_all
tf_reduce_max = K.reduce_max
tf_reduce_mean = K.reduce_mean
tf_reduce_min = K.reduce_min
tf_reshape = K.reshape
tf_shape = K.shape 
tf_sigmoid = K.sigmoid
tf_split = K.split
tf_sqrt = K.sqrt
tf_square = K.square
tf_stack = K.stack
tf_stop_gradient = K.stop_gradient
tf_tile = K.tile
tf_where = K.where
tf_zeros = K.zeros
tf_zeros_like = K.zeros_like

register_keras_serializable = K.register_keras_serializable
deserialize_keras_object = K.deserialize_keras_object

# Optional: silence autograph verbosity in TF-backed runtimes.
tf_autograph = K.autograph
tf_autograph.set_verbosity(0)

# ---------------------------------------------------------------------
# Module logger + shared docs
# ---------------------------------------------------------------------
DEP_MSG = dependency_message("nn.pinn.geoprior.models")

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_halnet_core_params),
)


__all__ = ["GeoPriorSubsNet", "PoroElasticSubsNet"]

@register_keras_serializable(
    'fusionlab.nn.pinn', name="GeoPriorSubsNet") 
class GeoPriorSubsNet(BaseAttentive):
    
    OUTPUT_KEYS = ("subs_pred", "gwl_pred")
    
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
        
        self._output_keys = list(self.OUTPUT_KEYS)
        
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

        # --------------------------------------------------------------
        # Scaling kwargs: accept dict OR JSON string/path.
        # Then canonicalize + validate alias consistency.
        # --------------------------------------------------------------
        self.scaling_kwargs = load_scaling_kwargs(
            scaling_kwargs,
            copy=True,
        )

        # Ensure nested bounds is a plain dict if provided.
        b = self.scaling_kwargs.get("bounds", None)
        if isinstance(b, Mapping) and not isinstance(b, dict):
            self.scaling_kwargs["bounds"] = dict(b)
        
        self._track_aux_metrics = get_sk(
            self.scaling_kwargs, "track_aux_metrics", default=True
            )

        # Canonicalize missing canonical keys from aliases.
        self.scaling_kwargs = canonicalize_scaling_kwargs(
            self.scaling_kwargs,
        )

        # Enforce that canonical and alias keys do not disagree.
        enforce_scaling_alias_consistency(
            self.scaling_kwargs,
            where="validate",
        )

        # --------------------------------------------------------------
        # Resolve time_units (scaling wins; else __init__ arg).
        # Always store final value back to scaling_kwargs.
        # --------------------------------------------------------------
        scale_tu = get_sk(
            self.scaling_kwargs,
            "time_units",
            default=None,
        )
        if isinstance(scale_tu, str) and not scale_tu.strip():
            scale_tu = None

        self.time_units = (
            scale_tu if scale_tu is not None else time_units
        )
        self.scaling_kwargs["time_units"] = self.time_units

        # ------------------------------------------------------------------
        # Drainage mode (controls Hd_factor used in tau_phys prior)
        # ------------------------------------------------------------------
        self.use_effective_thickness = use_effective_h
        self.Hd_factor = hd_factor   # if Hd = Hd_factor * H
        
        drainage_mode = self.scaling_kwargs.get("drainage_mode", None)
        
        # Only auto-override if user didn't explicitly pass custom values
        if drainage_mode is not None and (
                use_effective_h is False and hd_factor == 1.0):
            dm = str(drainage_mode).strip().lower()
            self.use_effective_thickness = True
            self.Hd_factor = 0.5 if dm.startswith("double") else 1.0

        # Optional: run scaling sanity checks now (policy-aware).
        validate_scaling_kwargs(self.scaling_kwargs)

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
            mv = LearnableMV(initial_value=float(mv), trainable=False)
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
            
        self.h_ref_config = h_ref

        self.mv_config = mv
        self.kappa_config = kappa
        self.gamma_w_config = gamma_w

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
        self.lambda_q = 0.0

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
        
        self.add_on = None
        
        if self._track_aux_metrics:
            self.add_on = GeoPriorTrackers(
                quantiles=bool(self.quantiles),
                subs_key="subs_pred",
                gwl_key="gwl_pred",
                q_axis=2,
                n_q=3,
            )

            
        self._init_coordinate_corrections()
        self._build_pinn_components()
        
        self.output_names = list(self._output_keys)
        

    # @property
    # def output_names(self):
    #     # Used by Keras containers as “the output order”
    #     return list(self._output_keys)

    @property
    def _output_keys(self):
        return self.__output_keys

    @_output_keys.setter
    def _output_keys(self, v):
        self.__output_keys = list(v)
       
    def _order_by_output_keys(self, d: dict) -> OrderedDict:
        return OrderedDict(
            (k, d[k])
            for k in self._output_keys
            if (k in d and d[k] is not None)
        )

    @property
    def metrics(self):
        base = super().metrics
        extras = []
    
        for m in (
            getattr(self, "eps_prior_metric", None),
            getattr(self, "eps_cons_metric", None),
            getattr(self, "eps_gw_metric", None),
        ):
            if m is not None:
                extras.append(m)
    
        if getattr(self, "add_on", None) is not None:
            extras.extend(self.add_on.metrics)
    
        seen = set()
        out = []
        for m in list(base) + list(extras):
            if id(m) not in seen:
                out.append(m)
                seen.add(id(m))
        return out


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
        self.eps_gw_metric =  Mean(name="epsilon_gw")

    
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
        """
        Create scalar physics params + fixed constants.
    
        Notes
        -----
        - m_v is stored in log-space when learnable.
        - We use a NaN-safe clip constraint so a bad
          update cannot leave log_mv as NaN forever.
        """
    
        # -------------------------------------------------
        # Compressibility m_v
        # -------------------------------------------------
        mv0 = float(self.mv_config.initial_value)
    
        # Hard safety window for exp(log_mv) in float32.
        log_mv_min = tf_log(tf_constant(_EPSILON, tf_float32))
        log_mv_max = tf_log(tf_constant(1e-4, tf_float32))
    
        if isinstance(self.mv_config, LearnableMV):
            # Learnable scalar in log-space to enforce
            # positivity: mv = exp(log_mv).
            self.log_mv = self.add_weight(
                name="log_param_mv",
                shape=(),
                initializer=Constant(
                    tf_log(tf_constant(mv0, tf_float32)),
                ),
                trainable=bool(
                    getattr(self.mv_config, "trainable", False),
                ),
                constraint=LogClipConstraint(
                    min_value=log_mv_min,
                    max_value=log_mv_max,
                ),
            )
        else:
            # Fixed scalar (linear space).
            self._mv_fixed = tf_constant(mv0, dtype=tf_float32)
    
        # -------------------------------------------------
        # Consistency factor κ (log-space if learnable)
        # -------------------------------------------------
        self._kappa_fixed = tf_constant(
            float(self.kappa_config.initial_value),
            dtype=tf_float32,
        )
            
        if isinstance(self.kappa_config, LearnableKappa):
            self.log_kappa = self.add_weight(
                name="log_param_kappa",
                shape=(),
                initializer=Constant(
                    tf_log(self.kappa_config.initial_value),
                ),
                trainable=bool(
                    getattr(self.kappa_config, "trainable", False),
                ),
            )
    
        # -------------------------------------------------
        # Fixed physical constants
        # -------------------------------------------------
        self.gamma_w = tf_cast(
            self.gamma_w_config.get_value(),
            tf_float32,
        )
    
        self.h_ref_mode = getattr(
            self.h_ref_config,
            "mode",
            "fixed",
        )
    
        # Always store a numeric head datum.
        self.h_ref = tf_constant(
            float(self.h_ref_config.value),
            dtype=tf_float32,
        )
    
        # -------------------------------------------------
        # Runtime placeholders for last evaluated fields
        # -------------------------------------------------
        self.K_field = None
        self.Ss_field = None
        self.tau_field = None


   # @tf_autograph.experimental.do_not_convert
    def run_encoder_decoder_core(
        self,
        static_input: Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        coords_input: Tensor,
        training: bool,
    ) -> Tuple[Tensor, Tensor]:
        
        def _assert_finite(x: Tensor, tag: str) -> Tensor:
            tf_debugging.assert_all_finite(
                x,
                f"NaN/Inf at {tag}",
            )
            return x

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
        
        dynamic_input = tf_cast(dynamic_input, tf_float32)
        dynamic_input = _assert_finite(dynamic_input, "dynamic_input")
        
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
                dyn_context = _assert_finite(
                    dyn_context,
                    "dyn_context (dynamic_vsn)",
                )
                dyn_proc = self.dynamic_vsn_grn(
                    dyn_context,
                    training=training,
                )
                dyn_proc = _assert_finite(
                    dyn_proc,
                    "dyn_proc (dynamic_vsn_grn)",
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
                dyn_proc = _assert_finite(
                    dyn_proc,
                    "dyn_proc (dynamic_dense)",
                )
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
        
        # dyn_proc = _assert_finite(dyn_proc, "dyn_proc")
        if self.verbose >= 1: 
            fut_proc = _assert_finite(fut_proc, "fut_proc")
            
            encoder_raw = _assert_finite(
                encoder_raw,
                "encoder_raw",
            )
            encoder_input = _assert_finite(
                encoder_input,
                "encoder_input",
            )
        
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
                    training=training,
                )
                encoder_sequences = norm(
                    encoder_sequences + attn_out
                )
        
        if self.verbose >= 1: 
            encoder_sequences = _assert_finite(
                encoder_sequences,
                "encoder_sequences",
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
        
        if self.verbose >= 1:
            # After decoder projection
            projected_decoder_input = _assert_finite(
                projected_decoder_input,
                "projected_decoder_input",
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
        if self.verbose >= 1:
            # After apply_attention_levels
            final_features = _assert_finite(
                final_features,
                "final_features",
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
    
    def forward_with_aux(
        self,
        inputs: Dict[str, Optional[Tensor]],
        training: bool = False,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Returns (y_pred, aux) for diagnostics;"
        " does not affect Keras training."""
        
        return self._forward_all(
            inputs, training=training
        )
    
    # @tf_autograph.experimental.do_not_convert
    def call(
        self,
        inputs: Dict[str, Optional[Tensor]],
        training: bool = False,
    ) -> Dict[str, Tensor]:
        """Keras forward: return supervised outputs only."""
        y_pred, _aux = self._forward_all(
            inputs,
            training=training,
        )
        return y_pred

    def _forward_all(
        self,
        inputs: Dict[str, Optional[Tensor]],
        training: bool = False,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Forward: data + physics heads + aux."""
        sk = self.scaling_kwargs or {}
    
        # ==========================================================
        # 1) Standardized PINN unpack
        # ==========================================================
        # t,x,y: (B,H,1)
        # H_field: (B,1,1) or (B,H,1) broadcastable
        # static_features: (B,S)
        # dynamic_features: (B,H,D)
        # future_features: (B,H,F)
        (
            t,
            x,
            y,
            H_field,
            static_features,
            dynamic_features,
            future_features,
        ) = process_pinn_inputs(
            inputs,
            mode="auto",
            model_name="geoprior",
        )
    
        # coords_for_decoder: (B,H,3) with last dim [t,x,y]
        coords_for_decoder = tf_concat(
            [t, x, y],
            axis=-1,
        )
        tf_debugging.assert_shapes(
            [(coords_for_decoder, ("B", "H", 3))],
        )
    
        # Keep a handle (debug / external reads).
        self.H_field = H_field
    
        # Validate features vs model dims.
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
            inputs=[
                static_features,
                dynamic_features,
                future_features,
            ],
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            mode="strict",
            verbose=0,
        )
    
        # ==========================================================
        # 2) Shared encoder/decoder backbone
        # ==========================================================
        # data_feat_2d: (B,H,Cd)
        # phys_feat_raw_3d: (B,H,Cp)
        data_feat_2d, phys_feat_raw_3d = (
            self.run_encoder_decoder_core(
                static_input=static_p,
                dynamic_input=dynamic_p,
                future_input=future_p,
                coords_input=coords_for_decoder,
                training=training,
            )
        )
    
        # Fail-fast: physics features must be finite.
        tf_debugging.assert_all_finite(
            phys_feat_raw_3d,
            "phys_feat_raw_3d has NaN/Inf.",
        )
    
        if self.verbose > 1:
            if "tf_print_nonfinite" in globals():
                tf_print_nonfinite(
                    "call/phys_feat_raw_3d",
                    phys_feat_raw_3d,
                )
    
        # ==========================================================
        # 3) Data path (mean): gwl/head + optional subs residual
        # ==========================================================
        # gwl_corr: (B,H,output_gwl_dim)
        # subs_corr: (B,H,output_subsidence_dim)
        gwl_corr = self.coord_mlp(
            coords_for_decoder,
            training=training,
        )
        subs_corr = self.subs_coord_mlp(
            coords_for_decoder,
            training=training,
        )
    
        # decoded_means_net: (B,H,subs_dim+gwl_dim)
        decoded_means_net = self.multi_decoder(
            data_feat_2d,
            training=training,
        )
        decoded_means_net = decoded_means_net + tf_concat(
            [subs_corr, gwl_corr],
            axis=-1,
        )
    
        # subs_res_net: (B,H,subs_dim)
        # gwl_mean_net: (B,H,gwl_dim)
        subs_res_net = decoded_means_net[
            ...,
            : self.output_subsidence_dim,
        ]
        gwl_mean_net = decoded_means_net[
            ...,
            self.output_subsidence_dim :,
        ]
    
        # ==========================================================
        # 4) Physics heads: K, Ss, Δlogτ, optional Q
        # ==========================================================
        # Each head returns (B,H,1) by design.
        K_raw = self.K_head(
            phys_feat_raw_3d,
            training=training,
        )
        Ss_raw = self.Ss_head(
            phys_feat_raw_3d,
            training=training,
        )
        dlogtau_raw = self.tau_head(
            phys_feat_raw_3d,
            training=training,
        )
    
        Q_raw = None
        if self.Q_head is not None:
            Q_raw = self.Q_head(
                phys_feat_raw_3d,
                training=training,
            )
    
        parts = [K_raw, Ss_raw, dlogtau_raw]
        if Q_raw is not None:
            parts.append(Q_raw)
    
        # phys_mean_raw: (B,H,3) or (B,H,4)
        phys_mean_raw = tf_concat(
            parts,
            axis=-1,
        )
    
        # ==========================================================
        # 5) OPTION-1 mean subsidence: physics-driven in SI
        # ==========================================================
        # Freeze fields over time to avoid K/Ss/tau drifting
        # across horizons. Uses mean over H, then broadcast.
        freeze_fields = bool(
            get_sk(
                sk,
                "freeze_physics_fields_over_time",
                default=True,
            )
        )
    
        if freeze_fields:
            K_base = tf_broadcast_to(
                tf_reduce_mean(K_raw, axis=1, keepdims=True),
                tf_shape(K_raw),
            )
            Ss_base = tf_broadcast_to(
                tf_reduce_mean(Ss_raw, axis=1, keepdims=True),
                tf_shape(Ss_raw),
            )
            dlogtau_base = tf_broadcast_to(
                tf_reduce_mean(
                    dlogtau_raw,
                    axis=1,
                    keepdims=True,
                ),
                tf_shape(dlogtau_raw),
            )
        else:
            K_base = K_raw
            Ss_base = Ss_raw
            dlogtau_base = dlogtau_raw
    
        # H_si: (B,1,1) or (B,H,1) in meters.
        H_si = to_si_thickness(
            H_field,
            sk,
        )
        H_floor = float(
            get_sk(sk, "H_floor_si", default=1e-3)
        )
        H_si = tf_maximum(
            H_si,
            tf_constant(H_floor, tf_float32),
        )
    
        # K_field: (B,H,1) m/s
        # Ss_field: (B,H,1) 1/m
        # tau_field: (B,H,1) seconds
        (
            K_field,
            Ss_field,
            tau_field,
            _tau_phys,
            _Hd_eff,
            _delta_log_tau,
            _logK,
            _logSs,
            _log_tau,
            _log_tau_phys,
        ) = compose_physics_fields(
            self,
            coords_flat=coords_for_decoder,
            H_si=H_si,
            K_base=K_base,
            Ss_base=Ss_base,
            tau_base=dlogtau_base,
            training=training,
            verbose=0,
        )
    
        # ----------------------------------------------------------
        # 5.1) Convert gwl_mean -> head in SI meters
        # ----------------------------------------------------------
        # h_mean_si: (B,H,1)
        h_mean_si = to_si_head(
            gwl_mean_net,
            sk,
        )
        h_mean_si = gwl_to_head_m(
            h_mean_si,
            sk,
            inputs=inputs,
        )
    
        # ----------------------------------------------------------
        # 5.2) Base shapes at t0 (B,1,1)
        # ----------------------------------------------------------
        like_11 = h_mean_si[:, :1, :1]
    
        h_ref_si_11 = get_h_ref_si(
            self,
            inputs,
            like=like_11,
        )
        s0_cum_si_11 = get_s_init_si(
            self,
            inputs,
            like=like_11,
        )
    
        # ODE state is incremental: start at zero.
        s0_inc_si_11 = tf_zeros_like(s0_cum_si_11)
    
        # dt_units: (B,H,1) in model time_units.
        dt_units = infer_dt_units_from_t(
            t,
            sk,
        )
    
        # ----------------------------------------------------------
        # 5.3) Integrate consolidation mean (incremental)
        # ----------------------------------------------------------
        dd = resolve_cons_drawdown_options(sk)
    
        # s_inc_si: (B,H,1) incremental settlement since t0.
        s_inc_si = integrate_consolidation_mean(
            h_mean_si=h_mean_si,
            Ss_field=Ss_field,
            H_field_si=H_si,
            tau_field=tau_field,
            h_ref_si=h_ref_si_11,
            s_init_si=s0_inc_si_11,
            dt=dt_units,
            time_units=self.time_units,
            method=self.residual_method,
            relu_beta=dd["relu_beta"],
            drawdown_mode=dd["drawdown_mode"],
            drawdown_rule=dd["drawdown_rule"],
            stop_grad_ref=dd["stop_grad_ref"],
            drawdown_zero_at_origin=dd[
                "drawdown_zero_at_origin"
            ],
            drawdown_clip_max=dd["drawdown_clip_max"],
            verbose=self.verbose,
        )
    
        dbg_call_nonfinite(
            verbose=self.verbose,
            coords_for_decoder=coords_for_decoder,
            H_si=H_si,
            K_base=K_base,
            Ss_base=Ss_base,
            dlogtau_base=dlogtau_base,
            tau_field=tau_field,
        )
    
        # ----------------------------------------------------------
        # 5.4) Map to configured subsidence_kind
        # ----------------------------------------------------------
        kind = str(
            get_sk(sk, "subsidence_kind", default="cumulative")
        ).strip().lower()
    
        # subs_phys_si: (B,H,1) in meters.
        if kind == "increment":
            ds0 = s_inc_si[:, :1, :]
            dsr = s_inc_si[:, 1:, :] - s_inc_si[:, :-1, :]
            subs_phys_si = tf_concat(
                [ds0, dsr],
                axis=1,
            )
        else:
            subs_phys_si = s0_cum_si_11 + s_inc_si
    
        # Convert SI mean -> model space.
        subs_phys_model = from_si_subsidence(
            subs_phys_si,
            sk,
        )
    
        # Optional learned residual around physics mean.
        allow_resid = bool(
            get_sk(sk, "allow_subs_residual", default=False)
        )
        subs_gate = self._subs_resid_gate()
        if not allow_resid:
            subs_gate = tf_constant(0.0, tf_float32)
    
        # subs_mean: (B,H,subs_dim)
        subs_mean = subs_phys_model + subs_gate * subs_res_net
    
        # decoded_means: (B,H,subs_dim+gwl_dim)
        decoded_means = tf_concat(
            [subs_mean, gwl_mean_net],
            axis=-1,
        )
        data_mean_raw = decoded_means
    
        # ==========================================================
        # 6) Quantiles (centered on physics mean)
        # ==========================================================
        if self.quantiles is not None:
            data_final = self.quantile_distribution_modeling(
                decoded_means,
                training=training,
            )
        else:
            data_final = decoded_means
    
        # Split supervised heads.
        subs_pred, gwl_pred = self.split_data_predictions(
            data_final,
        )
    
        y_pred_raw = {
            "gwl_pred": gwl_pred,
            "subs_pred": subs_pred,
        }
        y_pred = self._order_by_output_keys(y_pred_raw)
    
        aux = {
            "data_final": data_final,
            "data_mean_raw": data_mean_raw,
            "phys_mean_raw": phys_mean_raw,
            "phys_features_raw_3d": phys_feat_raw_3d,
        }
        return y_pred, aux

    
    def train_step(self, data):
        """Custom training step (v3.2)."""
        # ------------------------------------------------------
        # 0) Unpack + canonicalize targets
        # ------------------------------------------------------
        inputs, targets = data
    
        targets = _canonicalize_targets(targets)
        targets = self._order_by_output_keys(targets)
        targets = {k: targets[k] for k in self.output_names}
    
        dbg_step0_inputs_targets(
            verbose=self.verbose,
            inputs=inputs,
            targets=targets,
        )
    
        sk = self.scaling_kwargs or {}
        debug_grads = bool(
            get_sk(
                sk,
                "debug_physics_grads",
                default=False,
            )
        )
        
        # ------------------------------------------------------
        # 1) Forward + physics inside a single outer tape
        #    (physics_core uses an inner tape for coord grads)
        # ------------------------------------------------------
        with GradientTape(persistent=True) as tape:
            out = physics_core(
                self,
                inputs=inputs,
                training=True,
                return_maps=False,
                for_train=True,
            )
    
            y_pred = out["y_pred"]
            # aux = out["aux"]
            phys = out["physics"]
            terms_scaled = out["terms_scaled"]
    
            # Keep only supervised outputs (stable ordering)
            y_pred = {k: y_pred[k] for k in self.output_names}
    
            # --------------------------------------------------
            # 2) Data loss (compiled)
            # --------------------------------------------------
            targets_aligned = {
                k: _align_true_for_loss(targets[k], y_pred[k])
                for k in self.output_names
            }
    
            yt_list = [targets_aligned[k] for k in self.output_names]
            yp_list = [y_pred[k] for k in self.output_names]
    
            data_loss = self.compiled_loss(
                yt_list,
                yp_list,
                regularization_losses=self.losses,
            )
    
            # --------------------------------------------------
            # 3) Total loss = data + physics
            # --------------------------------------------------
            if phys is None:
                phys_scaled = tf_constant(0.0, tf_float32)
            else:
                phys_scaled = phys["physics_loss_scaled"]
    
            total_loss = data_loss + phys_scaled
    
        dbg_step9_losses(
            verbose=self.verbose,
            data_loss=data_loss,
            physics_loss_scaled=phys_scaled,
            total_loss=total_loss,
        )
    
        # ------------------------------------------------------
        # 4) Grads + scaling + clip
        # ------------------------------------------------------
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
    
        scaled = self._scale_param_grads(grads, trainable_vars)
        scaled = filter_nan_gradients(scaled)
    
        pairs = [
            (g, v)
            for g, v in zip(scaled, trainable_vars)
            if g is not None
        ]
        if pairs:
            gs, vs = zip(*pairs)
            gs, _ = tf_clip_by_global_norm(list(gs), 1.0)
            gs = filter_nan_gradients(gs)
            self.optimizer.apply_gradients(zip(gs, vs))
    
        dbg_step10_grads(
            verbose=self.verbose,
            trainable_vars=trainable_vars,
            grads=grads,
        )
    
        dbg_term_grads_finite(
            verbose=self.verbose,
            debug_grads=debug_grads,
            trainable_vars=trainable_vars,
            data_loss=data_loss,
            terms_scaled=terms_scaled,
            tape=tape,
        )
    
        del tape
    
        # ------------------------------------------------------
        # 5) Add-on trackers
        # ------------------------------------------------------
        if self.add_on is not None:
            self.add_on.update_state(targets, y_pred)
    
        manual = None
        if self.add_on is not None:
            manual = self.add_on.as_dict
    
        # ------------------------------------------------------
        # 6) Return packed results (single path)
        # ------------------------------------------------------
        return pack_step_results(
            self,
            total_loss=total_loss,
            data_loss=data_loss,
            targets=targets,
            y_pred=y_pred,
            manual_trackers=manual,
            physics=phys,
        )


    def test_step(self, data):
        """
        Validation step (v3.2).
    
        - Computes supervised loss/metrics from `data_final`.
        - Optionally adds physics loss (same weighting scheme as train_step).
        - Uses `_evaluate_physics_on_batch()` for physics diagnostics, which
          internally uses a GradientTape (but no optimizer update here).
        """
        inputs, targets = data
        targets = self._order_by_output_keys(_canonicalize_targets(targets))

        # Forward pass (no optimizer / gradients applied)
        y_pred_for_eval = self(inputs, training=False)

        # safest: dict path (maps by output name)
        targets = {k: targets[k] for k in self.output_names}

        # Force plain python dicts (avoid wrapper weirdness)
        y_pred_for_eval = {k: y_pred_for_eval[k] for k in self.output_names}
        targets_aligned = {
            k: _align_true_for_loss(targets[k], y_pred_for_eval[k])
            for k in self.output_names
        }
        
        # Always call compiled_loss with ordered lists (stable)
        yt_list = [targets_aligned[k] for k in self.output_names]
        yp_list = [y_pred_for_eval[k] for k in self.output_names]
        
        data_loss = self.compiled_loss(
            yt_list, yp_list,
            regularization_losses=self.losses,
        )

        physics_bundle = None
        
        if self.add_on is not None:
            self.add_on.update_state(targets, y_pred_for_eval)

        
        if not self._physics_off():
            phys = self._evaluate_physics_on_batch(
                inputs,
                return_maps=False,
            )
            physics_bundle = phys
            total_loss = data_loss + phys["physics_loss_scaled"]
        else:
            total_loss = data_loss
        
        return pack_step_results(
            self,
            total_loss=total_loss,
            data_loss=data_loss,
            targets=targets,
            y_pred=y_pred_for_eval,
            manual_trackers=(
                self.add_on.as_dict
                if self.add_on is not None
                else None
            ),
            physics=physics_bundle,
        )


    def _evaluate_physics_on_batch(
        self,
        inputs: Dict[str, Optional[Tensor]],
        return_maps: bool = False,
    ) -> Dict[str, Tensor]:
        """Physics diagnostics on one batch."""
        out = physics_core(
            self,
            inputs=inputs,
            training=False,
            return_maps=return_maps,
            for_train=False,
        )
    
        packed = out["physics_packed"]
    
        if not return_maps:
            return packed
    
        maps: Dict[str, Tensor] = {}
    
        # dt in model.time_units
        if "dt_units" in out:
            maps["dt_units"] = out["dt_units"]
    
        # Core fields / residual maps (if available)
        if "R_prior" in out:
            maps["R_prior"] = out["R_prior"]
        if "R_cons" in out:
            maps["R_cons"] = out["R_cons"]
            maps["cons_res_vals"] = out["R_cons"]
        if "R_gw" in out:
            maps["R_gw"] = out["R_gw"]
    
        # Scaled residuals (helpful for debugging)
        if "R_cons_scaled" in out:
            maps["R_cons_scaled"] = out["R_cons_scaled"]
        if "R_gw_scaled" in out:
            maps["R_gw_scaled"] = out["R_gw_scaled"]
    
        # Learned fields (aliases kept for old callers)
        if "K_field" in out:
            maps["K_field"] = out["K_field"]
            maps["K"] = out["K_field"]
        if "Ss_field" in out:
            maps["Ss_field"] = out["Ss_field"]
            maps["Ss"] = out["Ss_field"]
    
        if "tau_field" in out:
            maps["tau_field"] = out["tau_field"]
            maps["tau"] = out["tau_field"]
    
        if "tau_phys" in out:
            maps["tau_phys"] = out["tau_phys"]
            maps["tau_prior"] = out["tau_phys"]
            maps["tau_closure"] = out["tau_phys"]
    
        if "Hd_eff" in out:
            maps["Hd_eff"] = out["Hd_eff"]
            maps["Hd"] = out["Hd_eff"]
    
        if "H_si" in out:
            maps["H_si"] = out["H_si"]
            maps["H"] = out["H_si"]
            maps["H_field"] = out["H_si"]
    
        if "Q_si" in out:
            maps["Q_si"] = out["Q_si"]
    
        # Optional extras
        if "R_smooth" in out:
            maps["R_smooth"] = out["R_smooth"]
        if "R_bounds" in out:
            maps["R_bounds"] = out["R_bounds"]
    
        merged = dict(packed)
        merged.update(maps)
        return merged

    def evaluate_physics(
        self,
        inputs: Union[Dict[str, Optional[Tensor]], "Dataset"],
        return_maps: bool = False,
        max_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Evaluate physics diagnostics (scalars + optional last maps)."""
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
    
        # ----------------------------------------------------------
        # Dataset path: aggregate scalars across batches.
        # If return_maps=True, keep maps from the last batch only.
        # ----------------------------------------------------------
        if isinstance(inputs, Dataset):
            acc: Dict[str, List[Tensor]] = {}
            last_maps: Optional[Dict[str, Tensor]] = None
    
            for i, elem in enumerate(inputs):
                xb = elem[0] if isinstance(elem, (tuple, list)) else elem
    
                out_b = self._evaluate_physics_on_batch(
                    xb,
                    return_maps=return_maps,
                )
    
                for k, v in out_b.items():
                    if k.startswith(SCALAR_PREFIXES):
                        acc.setdefault(k, []).append(v)
    
                if return_maps:
                    last_maps = {
                        k: out_b[k]
                        for k in MAP_KEYS
                        if k in out_b
                    }
    
                if max_batches is not None:
                    if (i + 1) >= max_batches:
                        break
    
            if not acc:
                return {}
    
            out = {
                k: tf_reduce_mean(tf_stack(vs))
                for k, vs in acc.items()
            }
            if return_maps and last_maps is not None:
                out.update(last_maps)
    
            return out
    
        # ----------------------------------------------------------
        # Mapping path: allow numpy-like arrays when batch_size is
        # provided, by wrapping into a Dataset.
        # ----------------------------------------------------------
        if isinstance(inputs, Mapping) and batch_size is not None:
            any_tensor = any(
                isinstance(v, Tensor)
                for v in inputs.values()
                if v is not None
            )
            if not any_tensor:
                ds = Dataset.from_tensor_slices(inputs)
                ds = ds.batch(batch_size)
                return self.evaluate_physics(
                    ds,
                    return_maps=return_maps,
                    max_batches=max_batches,
                )
    
        # ----------------------------------------------------------
        # Single-batch path: assume tensors already shaped.
        # ----------------------------------------------------------
        return self._evaluate_physics_on_batch(
            inputs,
            return_maps=return_maps,
        )

    def _physics_loss_multiplier(self) -> Tensor:
        """Physics multiplier from lambda_offset + offset_mode."""
        # If physics is off, multiplier is irrelevant.
        if self._physics_off():
            return tf_constant(1.0, dtype=tf_float32)
    
        mode = self.offset_mode
    
        if mode == "mul":
            tf_debugging.assert_greater(
                self._lambda_offset,
                tf_constant(0.0, tf_float32),
                message=(
                    "lambda_offset must be > 0 when "
                    "offset_mode='mul'."
                ),
            )
            return tf_identity(self._lambda_offset)
    
        if mode == "log10":
            return tf_pow(
                tf_constant(10.0, dtype=tf_float32),
                tf_identity(self._lambda_offset),
            )
    
        raise ValueError(
            f"Invalid offset_mode={mode!r}. "
            "Expected 'mul' or 'log10'."
        )
        
    # --------------------------------------------------------------
    # Training strategy gates (Q and subsidence residual)
    # --------------------------------------------------------------
    def _current_step_tensor(self) -> Tensor:
        """Graph-safe global step for warmup/ramp gates."""
        opt = getattr(self, "optimizer", None)
        it = getattr(opt, "iterations", None) if opt is not None else None
    
        # In inference/no-optimizer contexts: behave as "fully on".
        if it is None:
            return tf_constant(10**9, dtype=tf_int32)
    
        return tf_cast(it, tf_int32)
    
    
    def _q_gate(self) -> Tensor:
        """Gate for Q forcing (0..1)."""
        sk = self.scaling_kwargs or {}
    
        policy = str(sk.get("q_policy", "always_on"))
        warmup = int(sk.get("q_warmup_steps", 0) or 0)
        ramp = int(sk.get("q_ramp_steps", 0) or 0)
    
        return policy_gate(
            self._current_step_tensor(),
            policy,
            warmup_steps=warmup,
            ramp_steps=ramp,
            dtype=tf_float32,
        )
    
    
    def _subs_resid_gate(self) -> Tensor:
        """Gate for subsidence residual head (0..1)."""
        sk = self.scaling_kwargs or {}
    
        policy = str(sk.get("subs_resid_policy", "always_on"))
        warmup = int(sk.get("subs_resid_warmup_steps", 0) or 0)
        ramp = int(sk.get("subs_resid_ramp_steps", 0) or 0)
    
        return policy_gate(
            self._current_step_tensor(),
            policy,
            warmup_steps=warmup,
            ramp_steps=ramp,
            dtype=tf_float32,
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

        if hasattr(self, "log_mv"):
            # clip already enforced by constraint, but re-clip defensively
            log_mv = tf_cast(self.log_mv, tf_float32)
            log_mv = tf_where(tf_math.is_finite(log_mv), log_mv,
                              tf_log(tf_constant(1e-12, tf_float32)))
            return tf_exp(log_mv)
        
        return tf_cast(self._mv_fixed, tf_float32)

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
                scaled.append(None)
                continue
            mult = 1.0
            if mv_var is not None and v is mv_var:
                mult *= float(self._mv_lr_mult)
            if kappa_var is not None and v is kappa_var:
                mult *= float(self._kappa_lr_mult)
            scaled.append(g * tf_cast(mult, g.dtype))
            
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
        lambda_q: float = 0.0,
        lambda_offset: float = 1.0,
        mv_lr_mult: float = 1.0,
        kappa_lr_mult: float = 1.0,
        scale_mv_with_offset: bool = False,
        scale_q_with_offset: bool = True,
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
        self.lambda_q = float(lambda_q)
        
        self._scale_mv_with_offset = bool(scale_mv_with_offset)
        self._scale_q_with_offset = bool(scale_q_with_offset)
    
        if self._physics_off():
            # When physics is off, hard-disable these contributions.
            self.lambda_prior = 0.0
            self.lambda_smooth = 0.0
            self.lambda_mv = 0.0
            self.lambda_q = 0.0
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
            "scaling_kwargs": dict(self.scaling_kwargs or {}),
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

@register_keras_serializable("fusionlab.nn.pinn", name="PoroElasticSubsNet")
class PoroElasticSubsNet(GeoPriorSubsNet):
    """
    Poroelastic surrogate variant of GeoPriorSubsNet.

    Same architecture and outputs as GeoPriorSubsNet, but with:

    * default ``pde_mode='consolidation'`` (no groundwater-flow residual);
    * effective drained thickness enabled
      (``use_effective_h=True``, ``hd_factor < 1``);
    * stronger geomechanical consistency prior and soft bounds on
      (H, K, S_s) via larger default lambda weights.

    Intended as a physics-driven baseline for ablation / comparison.
    """

    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        # keep all public kwargs, but change some defaults:
        pde_mode: str = "consolidation",
        use_effective_h: bool = True,
        hd_factor: float = 0.6,
        kappa_mode: str = "bar",
        scale_pde_residuals: bool = True,
        scaling_kwargs: Optional[Dict[str, Any]] = None,
        name: str = "PoroElasticSubsNet",
        **kwargs,
    ):
        # ------------------------------------------------------------------
        # 1) Merge scaling_kwargs with default bounds, if not provided.
        # ------------------------------------------------------------------
        if scaling_kwargs is None:
            scaling_kwargs = {}

        bounds = dict(scaling_kwargs.get("bounds", {}) or {})

        # Only fill missing keys; do not overwrite user-provided ones.
        default_bounds = dict(
            H_min=5.0,
            H_max=80.0,
            logK_min=float(np.log(1e-8)),
            logK_max=float(np.log(1e-3)),
            logSs_min=float(np.log(1e-7)),
            logSs_max=float(np.log(1e-3)),
        )
        for k, v in default_bounds.items():
            bounds.setdefault(k, v)

        scaling_kwargs["bounds"] = bounds

        logger.info(
            "Initializing GeoPriorStrongPrior with "
            f"pde_mode={pde_mode}, use_effective_h={use_effective_h}, "
            f"hd_factor={hd_factor}, kappa_mode={kappa_mode}, "
            f"bounds={bounds}"
        )

        super().__init__(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            # pass through everything else, with updated defaults:
            pde_mode=pde_mode,
            use_effective_h=use_effective_h,
            hd_factor=hd_factor,
            kappa_mode=kappa_mode,
            scale_pde_residuals=scale_pde_residuals,
            scaling_kwargs=scaling_kwargs,
            name=name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Stronger default physics weights in compile()
    # ------------------------------------------------------------------
    def compile(
        self,
        lambda_cons: float = 1.0,
        lambda_gw: float = 0.0,   # gw_flow off by default for surrogate
        lambda_prior: float = 5.0,
        lambda_smooth: float = 1.0,
        lambda_mv: float = 0.1,
        lambda_bounds: float = 0.05,
        mv_lr_mult: float = 0.5,
        kappa_lr_mult: float = 0.5,
        **kwargs,
    ):
        """
        Compile with stronger defaults for the geomechanical prior.

        Compared to GeoPriorSubsNet, this variant:

        * sets ``lambda_gw=0.0`` (no groundwater-flow residual),
        * increases ``lambda_prior`` and ``lambda_bounds`` so that
          :math:`tau` is tightly tied to :math:`tau_phys`,
        * gives :math:`m_v` and :math:`kappa` a smaller LR multiplier
          so they move more conservatively.
        """
        logger.info(
            "Compiling PoroElasticSubsNet with "
            f"lambda_cons={lambda_cons}, lambda_gw={lambda_gw}, "
            f"lambda_prior={lambda_prior}, lambda_smooth={lambda_smooth}, "
            f"lambda_mv={lambda_mv}, lambda_bounds={lambda_bounds}"
        )
        return super().compile(
            lambda_cons=lambda_cons,
            lambda_gw=lambda_gw,
            lambda_prior=lambda_prior,
            lambda_smooth=lambda_smooth,
            lambda_mv=lambda_mv,
            lambda_bounds=lambda_bounds,
            mv_lr_mult=mv_lr_mult,
            kappa_lr_mult=kappa_lr_mult,
            **kwargs,
        )
