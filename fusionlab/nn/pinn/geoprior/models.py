# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

from ...._fusionlog import OncePerMessageFilter, fusionlog
from ....api.docs import DocstringComponents, _halnet_core_params
from ....compat.sklearn import Interval, StrOptions, validate_params
from ....compat.keras import compute_loss 
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
from .doc import GEOPRIOR_SUBSNET_DOC, POROELASTIC_SUBSNET_DOC
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
from .scaling import GeoPriorScalingConfig
from .stability import filter_nan_gradients
from .step_core import physics_core
from .utils import (
    from_si_subsidence,
    get_h_ref_si,
    get_s_init_si,
    get_sk,
    gwl_to_head_m,
    infer_dt_units_from_t,
    policy_gate,
    to_si_head,
    to_si_thickness,
)

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


# Module logger + shared docs
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

    @validate_params(
        {
            "output_subsidence_dim": [
                Interval(Integral, 1, None, closed="left"),
            ],
            "output_gwl_dim": [
                Interval(Integral, 1, None, closed="left"),
            ],
            "pde_mode": [
                StrOptions(
                    {
                        "consolidation",
                        "gw_flow",
                        "both",
                        "none",
                        "on",
                        "off",
                    }
                ),
                "array-like",
                None,
            ],
            "mv": [LearnableMV, Real],
            "kappa": [LearnableKappa, Real],
            "gamma_w": [FixedGammaW, Real],
            "h_ref": [
                FixedHRef,
                Real,
                StrOptions({"auto", "fixed"}),
                None,
            ],
            "use_effective_h": [bool],
            "hd_factor": [
                Interval(Real, 0, 1, closed="right"),
            ],
            "kappa_mode": [StrOptions({"bar", "kb"})],
            "offset_mode": [StrOptions({"mul", "log10"})],
            "time_units": [str, None],
            "bounds_mode": [StrOptions({"soft", "hard"}), None],
            "residual_method": [
                StrOptions({"exact", "euler"}),
            ],
            "scaling_kwargs": [
                Mapping,
                str,
                GeoPriorScalingConfig,
                None,
            ],
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
            
        self.bounds_mode = bounds_mode or "soft"

        # --------------------------------------------------------------
        # Scaling kwargs: accept None / Mapping / path / config.
        # Always resolve to a canonical, validated dict.
        # --------------------------------------------------------------
        self.scaling_cfg = GeoPriorScalingConfig.from_any(
            scaling_kwargs,
            copy=True,
        )
        
        # If user passed time_units but scaling has none,
        # inject it *before* resolve so derived fields match.
        if time_units is not None:
            tu0 = self.scaling_cfg.payload.get("time_units", None)
            if tu0 is None:
                self.scaling_cfg.payload["time_units"] = time_units
            elif isinstance(tu0, str) and not tu0.strip():
                self.scaling_cfg.payload["time_units"] = time_units
        
        try:
            self.scaling_kwargs = self.scaling_cfg.resolve()
        except Exception as err:
            logger.exception(
                "Scaling resolve failed (source=%r): %s",
                self.scaling_cfg.source, err
            )
            raise
        
        # Ensure nested bounds is a plain dict.
        b = self.scaling_kwargs.get("bounds", None)
        if isinstance(b, Mapping) and not isinstance(b, dict):
            self.scaling_kwargs["bounds"] = dict(b)
        
        # Resolve time_units from final scaling dict.
        self.time_units = self.scaling_kwargs.get(
            "time_units",
            None,
        )
        
        # If __init__ forces a bounds_mode and scaling is silent,
        # keep existing behavior (bounds_mode wins).
        if bounds_mode is None:
            bm0 = self.scaling_kwargs.get("bounds_mode", None)
            if bm0 is not None:
                self.bounds_mode = str(bm0)
        else:
            self.bounds_mode = bounds_mode or "soft"
        
        # Aux metrics flag (read from canonical scaling).
        self._track_aux_metrics = get_sk(
            self.scaling_kwargs,
            "track_aux_metrics",
            default=True,
        )

        # ------------------------------------------------------------------
        # Drainage mode (controls Hd_factor used in tau_phys prior)
        # ------------------------------------------------------------------
        self.use_effective_thickness = use_effective_h
        self.Hd_factor = hd_factor
        
        drainage_mode = self.scaling_kwargs.get(
            "drainage_mode",
            None,
        )
        
        if drainage_mode is not None and (
            use_effective_h is False and hd_factor == 1.0
        ):
            dm = str(drainage_mode).strip().lower()
            self.use_effective_thickness = True
            self.Hd_factor = 0.5 if dm.startswith(
                "double"
            ) else 1.0
        
        # mutate self.scaling_kwargs (time_units, drainage, etc)
        self.scaling_cfg = GeoPriorScalingConfig.from_any(
            self.scaling_kwargs,
            copy=True,
        )
        
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
        
        self.output_names = list(self._output_keys)
        
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
        

    def build(self, input_shape: Any) -> None:
        """
        Build the model's weights and sublayers.

        Keras may call `build()` (e.g. via `model.build()` or
        `model.summary()`) before the first forward pass.
        For subclassed models, we must ensure all sublayers
        are actually built, otherwise Keras can mark the layer
        as built while internal state remains unbuilt.
        
        How to use it 
        ---------------
        model.build(
            {
                "static_features": (None, S),
                "dynamic_features": (None, H, D),
                "future_features": (None, H, F),
                "coords": (None, H, 3),
                "H_field": (None, H, 1),
            }
        )
        model.summary()

        """
        if getattr(self, "built", False):
            return

        # -------------------------------------------------
        # 0) Ensure heads/layers exist (if lazily created)
        # -------------------------------------------------
        if not hasattr(self, "K_head"):
            # This also calls `_build_physics_layers()`.
            self._build_attentive_layers()

        # -------------------------------------------------
        # 1) Extract shapes (dict-input is the common case)
        # -------------------------------------------------
        shp = input_shape
        s_sh = None
        d_sh = None
        f_sh = None
        c_sh = None
        h_sh = None

        if isinstance(shp, Mapping):
            s_sh = shp.get("static_features", None)
            d_sh = shp.get("dynamic_features", None)
            f_sh = shp.get("future_features", None)
            c_sh = shp.get("coords", None)
            h_sh = (
                shp.get("H_field", None)
                or shp.get("soil_thickness", None)
            )
        elif isinstance(shp, (list, tuple)):
            # Best-effort positional fallback.
            if len(shp) >= 1:
                s_sh = shp[0]
            if len(shp) >= 2:
                d_sh = shp[1]
            if len(shp) >= 3:
                f_sh = shp[2]
            if len(shp) >= 4:
                c_sh = shp[3]
            if len(shp) >= 5:
                h_sh = shp[4]

        def _as_list(x: Any) -> List[int | None]:
            if x is None:
                return []
            if hasattr(x, "as_list"):
                return list(x.as_list())
            try:
                return list(x)
            except Exception:
                return []

        def _fix_shape(
            raw: Any,
            fallback: Tuple[int, ...],
        ) -> Tuple[int, ...]:
            sh = _as_list(raw)
            if not sh:
                sh = list(fallback)
            if len(sh) != len(fallback):
                sh = list(fallback)
            # Replace None with fallback dims.
            for i, dim in enumerate(sh):
                if dim is None:
                    sh[i] = fallback[i]
            # Force a concrete batch for dummy build.
            sh[0] = 1
            return tuple(int(v) for v in sh)

        # -------------------------------------------------
        # 2) Choose safe fallback dims
        # -------------------------------------------------
        H = int(getattr(self, "forecast_horizon", 1) or 1)
        H = max(H, 1)

        s_fb = (1, int(self.static_input_dim))
        d_fb = (1, H, int(self.dynamic_input_dim))
        f_fb = (1, H, int(self.future_input_dim))
        c_fb = (1, H, 3)
        h_fb = (1, H, 1)

        s_shape = _fix_shape(s_sh, s_fb)
        d_shape = _fix_shape(d_sh, d_fb)
        f_shape = _fix_shape(f_sh, f_fb)
        c_shape = _fix_shape(c_sh, c_fb)
        h_shape = _fix_shape(h_sh, h_fb)

        # -------------------------------------------------
        # 3) Dummy forward to force-build sublayers
        # -------------------------------------------------
        # Avoid surfacing non-critical scaling warnings
        # during `summary()` / `build()`.
    

        dummy_inputs = {
            "static_features": tf_zeros(s_shape, tf_float32),
            "dynamic_features": tf_zeros(d_shape, tf_float32),
            "future_features": tf_zeros(f_shape, tf_float32),
            "coords": tf_zeros(c_shape, tf_float32),
            "H_field": tf_zeros(h_shape, tf_float32),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _ = self.call(dummy_inputs, training=False)

        super().build(input_shape)
 

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
        r"""
        Return predictions and auxiliary tensors for diagnostics.
        
        This method is a thin, public wrapper around :meth:`_forward_all`
        that exposes both:
        
        * ``y_pred``: the supervised outputs (what :meth:`call` returns),
        * ``aux``: intermediate tensors useful for debugging, physics
          evaluation, and research diagnostics.
        
        Unlike :meth:`call`, this method is intended for inspection and
        tooling. It does not change Keras training behavior because it does
        not alter loss computation or variable updates; it simply returns
        additional tensors already produced by the internal forward path.
        
        Parameters
        ----------
        inputs : dict
            Dict-input batch compatible with GeoPrior PINN models.
        
            Typical entries include:
        
            * ``static_features`` : Tensor, shape ``(B, S)``
            * ``dynamic_features`` : Tensor, shape ``(B, H, D)``
            * ``future_features`` : Tensor, shape ``(B, H, F)``
            * ``coords`` : Tensor, shape ``(B, H, 3)`` with last axis
              ordered as (t, x, y)
            * ``H_field`` or ``soil_thickness`` : Tensor, thickness field
              broadcastable to ``(B, H, 1)``
        
            Notes
            -----
            The exact required keys depend on the model configuration and
            Stage-1 export. This wrapper delegates all parsing and
            validation to :meth:`_forward_all`.
        training : bool, default False
            Forward-pass training flag. When True, dropout, batch norm,
            and other training-time layers behave accordingly.
        
        Returns
        -------
        y_pred : dict of str to Tensor
            Supervised predictions in the same format as :meth:`call`.
            At minimum, keys include ``'subs_pred'`` and ``'gwl_pred'``.
        aux : dict of str to Tensor
            Auxiliary tensors for diagnostics. Typical keys include:
        
            * ``data_final``: final data head tensor used for supervised
              outputs (may include quantile axis).
            * ``data_mean_raw``: mean-path output before quantile modeling.
            * ``phys_mean_raw``: concatenated physics logits (K, Ss, dlogtau,
              optional Q).
            * ``phys_features_raw_3d``: physics feature tensor emitted by the
              shared encoder-decoder core.
        
        Notes
        -----
        This method is recommended for:
        
        * debugging NaN/Inf propagation (by inspecting ``aux``),
        * computing physics residuals outside ``train_step`` using the same
          forward tensors,
        * building evaluation utilities that need intermediate heads.
        
        Examples
        --------
        Run a forward pass and inspect physics logits:
        
        >>> y_pred, aux = model.forward_with_aux(batch, training=False)
        >>> aux["phys_mean_raw"].shape
        TensorShape([B, H, 4])
        
        See Also
        --------
        call
            Standard Keras forward that returns supervised outputs only.
        
        _forward_all
            Internal forward routine that returns both predictions and
            auxiliary tensors.
        
        References
        ----------
        .. [1] Abadi, M., et al. TensorFlow: Large-scale machine learning
           on heterogeneous systems. 2015.
        """

        return self._forward_all(
            inputs, training=training
        )
    
    def call(
        self,
        inputs: Dict[str, Optional[Tensor]],
        training: bool = False,
    ) -> Dict[str, Tensor]:
        r"""
        Keras forward method returning supervised outputs only.
        
        This method defines the standard inference and training forward
        behavior expected by ``tf.keras.Model``. It returns only the
        supervised output dictionary that participates in Keras loss
        computation and metric updates.
        
        Internally, :meth:`call` delegates to :meth:`_forward_all` and
        discards the auxiliary outputs to ensure a stable, minimal
        prediction contract.
        
        Parameters
        ----------
        inputs : dict
            Dict-input batch compatible with GeoPrior PINN models.
        
            Typical entries include:
        
            * ``static_features`` : Tensor, shape ``(B, S)``
            * ``dynamic_features`` : Tensor, shape ``(B, H, D)``
            * ``future_features`` : Tensor, shape ``(B, H, F)``
            * ``coords`` : Tensor, shape ``(B, H, 3)`` with last axis
              ordered as (t, x, y)
            * ``H_field`` or ``soil_thickness`` : Tensor, thickness field
        
            Notes
            -----
            All parsing, shape checks, and coordinate handling are performed
            by :meth:`_forward_all`.
        training : bool, default False
            Forward-pass training flag. When True, training-time behavior
            (dropout, batch norm, etc.) is enabled.
        
        Returns
        -------
        y_pred : dict of str to Tensor
            Supervised prediction dictionary. Keys are ordered by the model
            output contract (for example, ``('subs_pred', 'gwl_pred')``).
            Each tensor is typically shaped:
        
            * without quantiles: ``(B, H, 1)``
            * with quantiles: ``(B, H, Q, 1)`` or a model-defined quantile
              layout
        
        Notes
        -----
        Auxiliary tensors such as physics logits and intermediate features
        are intentionally excluded from the return value. Use
        :meth:`forward_with_aux` when diagnostics are required.
        
        Examples
        --------
        Standard inference call:
        
        >>> y = model(batch, training=False)
        >>> sorted(y.keys())
        ['gwl_pred', 'subs_pred']
        
        See Also
        --------
        forward_with_aux
            Forward wrapper returning both predictions and diagnostics.
        
        _forward_all
            Internal routine returning ``(y_pred, aux)``.
        
        References
        ----------
        .. [1] Chollet, F. et al. Keras. 2015.
        """

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
        r"""
        Run the internal forward pass producing data and physics heads.
        
        This method implements the complete forward computation used by
        GeoPrior-style PINN models. It returns:
        
        * ``y_pred``: supervised outputs for training and inference,
        * ``aux``: diagnostic tensors required by the physics pathway and
          debugging utilities.
        
        The forward computation couples a shared encoder-decoder backbone
        with two output branches:
        
        Data branch
            Produces groundwater-level (or depth) predictions and a
            subsidence prediction that is anchored to a physics-derived mean
            path with an optional learned residual.
        
        Physics branch
            Produces per-location physics logits for the learned fields,
            typically :math:`K`, :math:`S_s`, and :math:`tau`, and optionally
            a forcing term :math:`Q`.
        
        The returned auxiliary dictionary provides the raw tensors required
        by :func:`fusionlab.nn.pinn.geoprior.step_core.physics_core`, which
        computes PDE derivatives and residual losses.
        
        Parameters
        ----------
        inputs : dict
            Dict-input batch compatible with the GeoPrior PINN API.
        
            The internal unpack expects the following conceptual groups:
        
            coordinates
                * ``coords`` : Tensor with (t, x, y) coordinates.
                  Shape is typically ``(B, H, 3)``.
            thickness
                * ``H_field`` or ``soil_thickness`` : Tensor thickness field,
                  broadcastable to ``(B, H, 1)``.
            features
                * ``static_features`` : Tensor, shape ``(B, S)``
                * ``dynamic_features`` : Tensor, shape ``(B, H, D)``
                * ``future_features`` : Tensor, shape ``(B, H, F)``
        
            Notes
            -----
            Input extraction and validation are delegated to helper
            functions such as ``process_pinn_inputs`` and ``check_inputs``.
        training : bool, default False
            Forward-pass training flag controlling dropout, batch norm, and
            other training-time layers.
        
        Returns
        -------
        y_pred : dict of str to Tensor
            Supervised outputs dictionary containing:
        
            ``'subs_pred'``
                Subsidence predictions. If quantiles are enabled, this may
                include a quantile axis.
            ``'gwl_pred'``
                Groundwater level (or related) predictions, aligned to the
                dataset convention.
        
            Notes
            -----
            Output key ordering is normalized by
            ``self._order_by_output_keys`` to ensure stable contracts.
        aux : dict of str to Tensor
            Auxiliary tensors required for physics evaluation and
            diagnostics. Keys include:
        
            ``data_final`` : Tensor
                Final data head output used to form ``subs_pred`` and
                ``gwl_pred``. Includes quantile modeling if enabled.
            ``data_mean_raw`` : Tensor
                Mean-path output before quantile distribution modeling.
            ``phys_mean_raw`` : Tensor
                Concatenated physics logits, typically:
        
                * K logits
                * Ss logits
                * dlogtau logits (tau parameterization)
                * optional Q logits
        
                Shape is ``(B, H, 3)`` or ``(B, H, 4)``.
            ``phys_features_raw_3d`` : Tensor
                Physics feature tensor produced by the shared backbone.
        
        Notes
        -----
        Physics-driven subsidence mean (Option-1)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        This forward routine computes the subsidence mean path from a
        consolidation integrator driven by predicted head. Conceptually,
        an incremental settlement state :math:`s(t)` is evolved using a
        relaxation form:
        
        .. math::
        
           \partial_t s = \frac{s_{eq}(h) - s}{tau}
        
        where :math:`s_{eq}(h)` depends on drawdown derived from head.
        The model can optionally learn a residual around this mean to
        capture unmodeled effects.
        
        Freeze-over-horizon behavior
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        When enabled in ``scaling_kwargs``, physics logits are averaged over
        the horizon dimension and broadcast back across time. This prevents
        K/Ss/tau from drifting across forecast steps, which can improve
        stability and identifiability in short-horizon training.
        
        Quantile outputs
        ~~~~~~~~~~~~~~~~
        If ``self.quantiles`` is not None, the final supervised output is
        wrapped by a quantile-distribution module. The quantile head is
        centered on the physics-driven mean so that uncertainty is modeled
        around a physically consistent baseline.
        
        Examples
        --------
        Run full forward and access both supervised and physics heads:
        
        >>> y_pred, aux = model._forward_all(batch, training=False)
        >>> y_pred["subs_pred"].shape
        TensorShape([B, H, 1])
        >>> aux["phys_mean_raw"].shape
        TensorShape([B, H, 4])
        
        Use aux outputs in the shared physics core:
        
        >>> out = physics_core(
        ...     model=model,
        ...     inputs=batch,
        ...     training=False,
        ... )
        >>> float(out["physics"]["eps_prior"])
        0.0
        
        See Also
        --------
        forward_with_aux
            Public wrapper returning ``(y_pred, aux)`` for diagnostics.
        
        call
            Keras forward returning supervised outputs only.
        
        fusionlab.nn.pinn.geoprior.step_core.physics_core
            Shared physics pathway that consumes ``phys_mean_raw`` and
            computes PDE residuals and losses.
        
        fusionlab.nn.pinn.geoprior.maths.compose_physics_fields
            Map physics logits to bounded physical fields and priors.
        
        References
        ----------
        .. [1] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
           Physics-informed neural networks: A deep learning framework
           for solving forward and inverse problems involving nonlinear
           partial differential equations. Journal of Computational
           Physics, 2019.
        
        .. [2] Terzaghi, K. Theoretical Soil Mechanics. Wiley, 1943.
        """

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
        r"""
        Run one custom training step for GeoPrior-style PINN training.
        
        This method overrides the standard ``tf.keras.Model.train_step`` to
        train a hybrid, physics-informed model with dict inputs and
        multi-output supervision. The step integrates:
        
        * supervised data losses (from ``compile`` / ``compiled_loss``),
        * physics losses computed by :func:`physics_core`,
        * optional gradient scaling for selected parameters,
        * robust gradient sanitization and global-norm clipping,
        * optional auxiliary metric trackers.
        
        The overall objective optimized by this step is:
        
        .. math::
        
           L_{total} = L_{data} + L_{phys}
        
        where :math:`L_{data}` is the compiled supervised loss and
        :math:`L_{phys}` is the scaled physics loss returned by
        :func:`physics_core`.
        
        Parameters
        ----------
        data : tuple
            Keras batch payload as ``(inputs, targets)``.
        
            * ``inputs`` is a dict of tensors matching the GeoPrior input
              API (static, dynamic, future, coords, thickness, etc.).
            * ``targets`` is a dict (or dict-like) of supervised targets.
        
            Notes
            -----
            The method expects a dict-style multi-output target structure.
            Targets are canonicalized and reordered to match
            ``self.output_names``.
        
        Returns
        -------
        metrics : dict
            Dictionary of scalar tensors suitable for Keras logging.
            The exact keys are produced by :func:`pack_step_results` and
            typically include:
        
            * ``loss`` / ``total_loss``: total objective value.
            * per-output supervised losses and metrics (from
              ``self.compiled_loss`` and ``self.compiled_metrics``).
            * physics summary terms (e.g., ``physics_loss_scaled`` and
              selected components) when physics is enabled.
            * optional "manual" metrics from add-on trackers.
        
        Notes
        -----
        Step outline
        ~~~~~~~~~~~~
        This training step performs the following stages:
        
        0) Unpack and canonicalize targets
            Targets are normalized into a stable dict structure using
            ``_canonicalize_targets`` and reordered by
            ``self._order_by_output_keys``. Only keys in
            ``self.output_names`` are retained to guarantee consistent
            ordering for both loss computation and logging.
        
        1) Forward pass with physics precomputation
            The step calls :func:`physics_core` inside a single outer
            ``GradientTape``. The physics core performs its own inner tape
            to compute coordinate derivatives required by PDE residuals.
            The outer tape ensures gradients flow through both:
        
            * supervised data predictions, and
            * physics loss scalars produced by the physics pathway.
        
        2) Supervised data loss
            Targets are aligned to prediction shapes (including quantile
            layout when applicable) using ``_align_true_for_loss`` and then
            passed as lists to ``self.compiled_loss``. This allows Keras to
            apply:
        
            * per-output losses configured in ``compile``,
            * regularization losses in ``self.losses``,
            * sample weighting logic if configured.
        
        3) Total objective
            The physics loss contribution is taken from the physics bundle
            as ``physics_loss_scaled``. If physics is disabled (or gated off)
            the contribution is treated as zero.
        
        4) Gradients, scaling, and clipping
            Gradients of the total objective are computed w.r.t. all
            trainable variables. The step then:
        
            * applies optional parameter-specific gradient scaling via
              ``self._scale_param_grads`` (for example, to slow down
              ``m_v`` or ``kappa`` updates),
            * filters NaN/Inf gradients using ``filter_nan_gradients``,
            * applies global norm clipping (default clip value is 1.0),
            * applies gradients via ``self.optimizer.apply_gradients``.
        
            This sequence is intended to improve stability for stiff
            physics losses and mixed-scale parameters.
        
        5) Auxiliary trackers
            If the model is configured with add-on trackers (for example,
            quantile coverage/sharpness or other custom diagnostics),
            ``update_state`` is called on the supervised outputs.
        
        6) Packed return
            The step returns a single packed dictionary from
            :func:`pack_step_results` so both training logs and evaluation
            summaries remain consistent.
        
        Physics loss semantics
        ~~~~~~~~~~~~~~~~~~~~~~
        The physics contribution returned by :func:`physics_core` is already
        assembled with internal multipliers and (optionally) warmup/ramp
        gating. In other words, ``physics_loss_scaled`` is the quantity that
        should be added to the supervised loss.
        
        If you need raw components for debugging, enable physics debug
        options in ``scaling_kwargs`` (for example,
        ``debug_physics_grads=True``) and use the debug hooks called inside
        this step.
        
        Gradient sanity and debugging
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        This method provides multiple stability and debug mechanisms:
        
        * NaN/Inf gradient filtering before applying updates.
        * Global-norm clipping to limit catastrophic updates.
        * Optional per-term gradient checks via ``dbg_term_grads_finite``
          when ``scaling_kwargs['debug_physics_grads']`` is enabled.
        
        These are particularly useful when PDE residuals are large early in
        training or when coordinate scaling is misconfigured.
        
        Examples
        --------
        Typical usage: compile and fit normally, relying on this custom
        train step:
        
        >>> model.compile(
        ...     optimizer=tf.keras.optimizers.Adam(1e-3),
        ...     loss={"subs_pred": "mse", "gwl_pred": "mse"},
        ... )
        >>> history = model.fit(train_ds, validation_data=val_ds, epochs=5)
        
        Inspect returned metrics keys during training:
        
        >>> logs = model.train_step(next(iter(train_ds)))
        >>> sorted(list(logs))[:5]
        ['data_loss', 'loss', 'physics_loss_scaled', 'total_loss', ...]
        
        See Also
        --------
        fusionlab.nn.pinn.geoprior.step_core.physics_core
            Shared physics pathway used to compute PDE residuals and physics
            loss scalars consistently across train and eval.
        
        pack_step_results
            Pack supervised metrics, physics terms, and manual trackers into
            a stable Keras logging dictionary.
        
        filter_nan_gradients
            Sanitize gradient lists by removing NaN/Inf tensors.
        
        tf.clip_by_global_norm
            TensorFlow utility for global-norm gradient clipping.
        
        References
        ----------
        .. [1] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
           Physics-informed neural networks: A deep learning framework
           for solving forward and inverse problems involving nonlinear
           partial differential equations. Journal of Computational
           Physics, 2019.
        
        .. [2] Goodfellow, I., Bengio, Y., and Courville, A.
           Deep Learning. MIT Press, 2016.
        """

        # ------------------------------------------------------
        # 0) Unpack + canonicalize targets
        # ------------------------------------------------------
        inputs, targets = data
        
        # XXX NOTE:
        #   Historically we enforced:
        #       targets = {k: targets[k] for k in self.output_names}
        #   This is STRICT and will raise KeyError if any output head
        #   (e.g. "gwl_pred") is intentionally *not supervised* during
        #   warm-start transferability runs (stage5).
        #
        #   Warm-start may provide only {"subs_pred": ...} targets while
        #   the model still exposes both outputs in self.output_names.
        #   In that case, strict indexing crashes.
        #
        # FIX / FEATURE:
        #   Introduce an opt-in "allow_missing_targets" flag (store-backed
        #   via scaling_kwargs). When enabled, missing/None targets are
        #   replaced *for loss only* with stop_gradient(y_pred) so the
        #   corresponding head contributes ~0 supervised loss without
        #   crashing. Metrics/add-on trackers MUST NOT see placeholders.
        #
        #   - Strict mode (default): missing targets => raise KeyError
        #   - Warm mode: allow_missing_targets=True => warn once and continue
        #
        # TODO:
        #   Consider adding a stage5 (transferrability) manifest/audit line 
        #   that records which heads were supervised vs. unsupervised 
        #   during warm-start.
    
        targets = _canonicalize_targets(targets)
        targets = self._order_by_output_keys(targets)
        
        # targets = {k: targets[k] for k in self.output_names}
        
        # Keep output ordering stable but allow missing keys.
        # (Missing or None => unsupervised head for this step.)
        targets = {k: targets.get(k) for k in self.output_names}
        
        # "Real" targets are what metrics / add_on / logs should see.
        # We drop unsupervised heads to avoid fake metrics.
        targets_real = {
            k: v for k, v in targets.items()
            if v is not None
        }    
        
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
            # y_pred = {k: y_pred[k] for k in self.output_names}
            # Keep only declared outputs (stable ordering)
            y_pred = {k: y_pred[k] for k in self.output_names}
            
            # --------------------------------------------------
            # 2) Data loss (compiled)[old]
            # --------------------------------------------------
            # targets_aligned = {
            #     k: _align_true_for_loss(targets[k], y_pred[k])
            #     for k in self.output_names
            # }
    
            # yt_list = [targets_aligned[k] for k in self.output_names]
            # yp_list = [y_pred[k] for k in self.output_names]
    
            # data_loss = self.compiled_loss(
            #     yt_list,
            #     yp_list,
            #     regularization_losses=self.losses,
            # )
            
            # --------------------------------------------------
            # 2) Data loss (compiled) [new]
            # --------------------------------------------------
            # XXX: OLD (STRICT) - crashes if a head target is missing:
            # targets = {k: targets[k] for k in self.output_names}
            #
            # FIX: build "loss targets" that may include placeholders for
            # missing/None heads when allow_missing_targets=True.
            targets_loss = self._targets_for_loss(targets, y_pred)
        
            targets_aligned = {
                k: _align_true_for_loss(targets_loss[k], y_pred[k])
                for k in self.output_names
            }
        
            # XXX IMPORT NOTE:
                # This removes the deprecation warning because Keras 3 will use
                # compute_loss, while Keras 2 will still work via compiled_loss.
                
            # yt_list = [targets_aligned[k] for k in self.output_names]
            # yp_list = [y_pred[k] for k in self.output_names]
        
            # data_loss = self.compiled_loss(
            #     yt_list,
            #     yp_list,
            #     regularization_losses=self.losses,
            # )
            data_loss = compute_loss(
                self,
                x=inputs,
                y=targets_aligned,
                y_pred=y_pred,
                sample_weight=None,
                training=True,
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
        # if self.add_on is not None:
        #     self.add_on.update_state(targets, y_pred)
        
        #XXX IMPORTANT:
        #   Use targets_real (no placeholders) so metrics reflect only
        #   supervised heads. Otherwise we'd log misleadingly good stats.
        
        if self.add_on is not None:
            self.add_on.update_state(targets_real, y_pred)
        manual = None
        if self.add_on is not None:
            manual = self.add_on.as_dict
    
        # ------------------------------------------------------
        # 6) Return packed results (single path)
        # ------------------------------------------------------
        # IMPORTANT:
        #   pass targets_real to pack_step_results so compiled metric
        #   updater only sees supervised heads (and won't crash on None)
        
        return pack_step_results(
            self,
            total_loss=total_loss,
            data_loss=data_loss,
            # targets=targets,
            targets=targets_real,  # IMPORTANT
            y_pred=y_pred,
            manual_trackers=manual,
            physics=phys,
        )
    
    def _allow_missing_targets(self) -> bool:
        sk = getattr(self, "scaling_kwargs", None) or {}
        return bool(get_sk(
            sk,
            "allow_missing_targets",
            default=False,
        ))
    
    def _warn_missing_targets_once(self, missing) -> None:
        if getattr(self, "_warned_missing_targets", False):
            return
        self._warned_missing_targets = True
        logger.warning(
            "Missing targets for outputs: %s. "
            "Using stop_gradient(y_pred) as a "
            "loss-only placeholder (head not "
            "supervised).",
            ", ".join(missing),
        )
    
    def _targets_for_loss(self, targets, y_pred):
        missing = [
            k for k in self.output_names
            if (k not in targets) or (targets[k] is None)
        ]
        if not missing:
            return dict(targets)
    
        if not self._allow_missing_targets():
            raise KeyError(
                "Missing targets for outputs: "
                + ", ".join(missing)
            )
    
        self._warn_missing_targets_once(missing)
    
        t = dict(targets)
        for k in missing:
            t[k] = tf_stop_gradient(y_pred[k])
        return t

    def test_step(self, data):
        r"""
        Run one evaluation (validation/test) step for GeoPrior models.
        
        This method overrides the standard ``tf.keras.Model.test_step`` to
        evaluate GeoPrior-style PINN models with dict inputs and multi-output
        targets. It computes:
        
        * supervised validation loss and metrics via ``compiled_loss`` and
          compiled metrics,
        * optional physics diagnostics and physics loss via
          ``_evaluate_physics_on_batch`` (no optimizer updates),
        * optional add-on tracker metrics (for example, quantile coverage
          and sharpness),
        * a unified packed logging dictionary returned by
          :func:`pack_step_results`.
        
        Unlike :meth:`train_step`, this method does not apply gradients or
        update model parameters. It may still use a GradientTape internally
        for physics derivatives when physics is enabled, but no optimizer
        step occurs.
        
        Parameters
        ----------
        data : tuple
            Keras batch payload as ``(inputs, targets)``.
        
            * ``inputs`` is a dict of tensors matching the GeoPrior input
              API (static, dynamic, future, coords, thickness, etc.).
            * ``targets`` is a dict (or dict-like) of supervised targets.
        
            Notes
            -----
            Targets are canonicalized and reordered to match
            ``self.output_names`` for stable loss computation.
        
        Returns
        -------
        metrics : dict
            Dictionary of scalar tensors suitable for Keras validation
            logging. The exact keys depend on configured losses, metrics,
            and physics settings, and are produced by
            :func:`pack_step_results`.
        
            Typical keys include:
        
            * ``loss`` / ``total_loss``: total evaluation objective.
            * ``data_loss``: supervised loss only.
            * per-output losses/metrics from Keras compiled configuration.
            * physics summary terms (for example ``physics_loss_scaled``,
              epsilons) if physics is enabled.
            * custom tracker metrics if add-on trackers are enabled.
        
        Notes
        -----
        Step outline
        ~~~~~~~~~~~~
        This evaluation step follows a stable, dict-safe flow:
        
        1) Unpack and canonicalize targets
            Targets are normalized into a stable dict structure and
            reordered by output key contract.
        
        2) Forward pass (supervised only)
            The method calls :meth:`call` via ``self(inputs, training=False)``
            to obtain supervised predictions only. Aux tensors are not
            returned here by design.
        
        3) Supervised loss and metrics
            Targets are aligned to prediction shapes using
            ``_align_true_for_loss`` and passed to ``compiled_loss`` as
            ordered lists to ensure consistent behavior across Keras
            versions and dict wrappers.
        
        4) Add-on trackers (optional)
            If configured, add-on trackers are updated with targets and
            predictions. These trackers are purely diagnostic and do not
            affect loss values unless explicitly integrated elsewhere.
        
        5) Physics diagnostics (optional)
            If physics is enabled, the method calls
            ``_evaluate_physics_on_batch(inputs, return_maps=False)`` to
            compute physics residual summaries and a scaled physics loss.
        
            The total evaluation objective is then:
        
            .. math::
        
               L_{total} = L_{data} + L_{phys}
        
            where :math:`L_{phys}` is the physics loss scalar returned by
            the physics evaluator.
        
            Notes
            -----
            The physics evaluator may use internal autodiff to compute PDE
            derivatives for residual diagnostics, but gradients are not used
            to update parameters in ``test_step``.
        
        6) Packed return
            The method returns a single packed dictionary from
            :func:`pack_step_results` to keep training and validation logs
            consistent.
        
        When to use physics in validation
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Enabling physics during validation is useful to monitor:
        
        * PDE residual RMS values (epsilon metrics),
        * consistency priors (for example, time-scale prior),
        * bounds penalties and stability signals.
        
        If validation speed is a concern, physics can be disabled with the
        model physics switch (for example, ``_physics_off()`` returning
        True), in which case only supervised losses/metrics are computed.
        
        Examples
        --------
        Standard evaluation with physics enabled:
        
        >>> logs = model.test_step(next(iter(val_ds)))
        >>> float(logs["data_loss"])
        1.23
        >>> float(logs["physics_loss_scaled"])
        0.01
        
        Disable physics for faster validation (model-specific switch):
        
        >>> model._physics_off = lambda: True
        >>> logs = model.test_step(next(iter(val_ds)))
        >>> "physics_loss_scaled" in logs
        False  # depends on pack_step_results configuration
        
        See Also
        --------
        train_step
            Custom training step that computes physics loss and applies
            gradients.
        
        _evaluate_physics_on_batch
            Evaluation-only physics routine that computes residual
            diagnostics without applying optimizer updates.
        
        pack_step_results
            Pack supervised metrics, physics terms, and manual trackers into
            a stable Keras logging dictionary.
        
        References
        ----------
        .. [1] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
           Physics-informed neural networks: A deep learning framework
           for solving forward and inverse problems involving nonlinear
           partial differential equations. Journal of Computational
           Physics, 2019.
        
        .. [2] Chollet, F. et al. Keras. 2015.
        """

        # ------------------------------------------------------
        # 0) Unpack + canonicalize targets
        # ------------------------------------------------------
        inputs, targets = data
        targets = self._order_by_output_keys(_canonicalize_targets(targets))
    
        # ------------------------------------------------------
        # 1) Forward pass (eval mode; no optimizer updates)
        # ------------------------------------------------------
        y_pred_for_eval = self(inputs, training=False)
    
        # XXX NOTE (strict vs warm-start):
        #   OLD behavior enforced strict supervision for *all* heads:
        #
        #       targets = {k: targets[k] for k in self.output_names}
        #
        #   This crashes in transfer warm-start if a head (e.g. "gwl_pred")
        #   is intentionally not provided in the dataset targets.
        #
        #   New behavior:
        #   - Keep stable output ordering but allow missing keys.
        #   - Build two target views:
        #       * targets_real: used for metrics / add_on / logging
        #         (drop missing/None => avoids fake "perfect" metrics)
        #       * targets_loss: used for compiled_loss only
        #         (fill missing with stop_gradient(y_pred) if allowed)
        #
        #   Strict mode (default): missing => KeyError (debug-friendly)
        #   Warm mode: scaling_kwargs["allow_missing_targets"]=True
        #     => warn once and continue.
    
        # Keep output ordering stable but allow missing keys.
        targets = {k: targets.get(k) for k in self.output_names}
    
        # Force plain python dicts (avoid wrapper weirdness)
        y_pred_for_eval = {k: y_pred_for_eval[k] for k in self.output_names}
    
        # Real targets (metrics / add_on) => drop None (unsupervised heads)
        targets_real = {
            k: v for k, v in targets.items()
            if v is not None
        }
    
        # Loss targets => fill missing with stop_gradient if allowed
        targets_loss = self._targets_for_loss(targets, y_pred_for_eval)
    
        # ------------------------------------------------------
        # 2) Supervised loss (compiled) - always list-based
        # ------------------------------------------------------
        targets_aligned = {
            k: _align_true_for_loss(targets_loss[k], y_pred_for_eval[k])
            for k in self.output_names
        }
    
        # yt_list = [targets_aligned[k] for k in self.output_names]
        # yp_list = [y_pred_for_eval[k] for k in self.output_names]
    
        # data_loss = self.compiled_loss(
        #     yt_list,
        #     yp_list,
        #     regularization_losses=self.losses,
        # )
        data_loss = compute_loss(
            self,
            x=inputs,
            y=targets_aligned,
            y_pred=y_pred_for_eval,
            sample_weight=None,
            training=False,
            regularization_losses=self.losses,
        )

        # ------------------------------------------------------
        # 3) Optional add-on trackers (diagnostic only)
        # ------------------------------------------------------
        # XXX IMPORTANT: use targets_real (no placeholders) to avoid
        # misleading metrics for unsupervised heads.
        if self.add_on is not None:
            self.add_on.update_state(targets_real, y_pred_for_eval)
    
        # ------------------------------------------------------
        # 4) Optional physics diagnostics
        # ------------------------------------------------------
        physics_bundle = None
        if not self._physics_off():
            phys = self._evaluate_physics_on_batch(
                inputs,
                return_maps=False,
            )
            physics_bundle = phys
            total_loss = data_loss + phys["physics_loss_scaled"]
        else:
            total_loss = data_loss
    
        # ------------------------------------------------------
        # 5) Return packed results (stable logs)
        # ------------------------------------------------------
        # IMPORTANT: pass targets_real so compiled metric updater
        # only sees supervised heads (dict-safe across Keras 2/3).
        return pack_step_results(
            self,
            total_loss=total_loss,
            data_loss=data_loss,
            targets=targets_real,  # IMPORTANT
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
        r"""
        Compute physics diagnostics on a single batch.
        
        This is a small evaluation wrapper around :func:`physics_core`.
        It runs the physics pathway with ``training=False`` and returns a
        packed dictionary of physics scalars suitable for logging.
        
        If ``return_maps=True``, the returned dict is augmented with selected
        residual maps and learned field tensors (including legacy aliases)
        from the same batch.
        
        Parameters
        ----------
        inputs : dict
            Dict input batch following the GeoPrior PINN batch API.
        return_maps : bool, default False
            If True, include residual maps and learned fields from the batch.
        
        Returns
        -------
        out : dict
            Packed physics scalars, plus optional maps if requested.
        
        See Also
        --------
        evaluate_physics
            Aggregate physics diagnostics over a dataset or batch.
        
        fusionlab.nn.pinn.geoprior.step_core.physics_core
            Shared physics computation used for diagnostics and training.
        """

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
        r"""
        Evaluate physics diagnostics over a batch or a dataset.
        
        This method computes physics-only diagnostics for GeoPrior-style
        PINN models. It supports three input modes:
        
        1. Dataset mode
            If ``inputs`` is a ``tf.data.Dataset``, physics scalars are
            computed for each batch and aggregated (mean) across batches.
            If ``return_maps=True``, residual maps and learned fields are
            returned from the last processed batch only.
        
        2. Mapping mode with batching
            If ``inputs`` is a mapping (dict-like) and ``batch_size`` is
            provided, the mapping can be wrapped into a dataset and batched
            automatically (primarily for numpy-like arrays).
        
        3. Single-batch mode
            If ``inputs`` is a dict of tensors already shaped for one batch,
            physics diagnostics are computed once and returned.
        
        The returned values are intended for monitoring PDE consistency,
        prior adherence, and stability during training and validation.
        
        Parameters
        ----------
        inputs : dict or Dataset
            Input payload used for physics evaluation.
        
            * If a dict: it should follow the GeoPrior batch API and contain
              tensors (or array-like values if ``batch_size`` is provided).
            * If a Dataset: the dataset should yield either:
              - a dict of inputs, or
              - a tuple/list where the first element is the inputs dict
                (targets are ignored).
        
        return_maps : bool, default False
            If True, include residual maps and learned field tensors.
        
            Notes
            -----
            In Dataset mode, maps are not aggregated across batches. The
            method returns maps from the last processed batch only to keep
            memory usage bounded and avoid ambiguous aggregation semantics.
        max_batches : int or None, default None
            Maximum number of dataset batches to process. If None, iterate
            through the entire dataset.
        
            Notes
            -----
            This option is useful for quick diagnostics on large datasets.
        batch_size : int or None, default None
            If provided and ``inputs`` is a mapping of numpy-like arrays,
            wrap into a dataset and batch by this size before evaluation.
        
        Returns
        -------
        out : dict of str to Tensor
            Dictionary of physics diagnostics.
        
            Aggregated scalars (Dataset mode)
            --------------------------------
            Scalars are aggregated by mean across processed batches for keys
            whose names start with one of these prefixes:
        
            * ``'loss_'`` (physics loss components)
            * ``'epsilon_'`` (RMS-style residual diagnostics)
        
            Example aggregated outputs include:
        
            * ``loss_cons`` / ``loss_gw`` / ``loss_prior`` / ``loss_smooth``
            * ``loss_bounds`` / ``loss_mv`` / ``loss_q_reg``
            * ``epsilon_cons`` / ``epsilon_gw`` / ``epsilon_prior``
        
            Optional maps (when return_maps=True)
            -------------------------------------
            The method may include maps from the last processed batch,
            selected from:
        
            * residuals: ``R_prior``, ``R_cons``, ``R_gw``
            * learned fields: ``K``, ``Ss``, ``tau``
            * closure prior: ``tau_prior`` / ``tau_closure``
            * thickness: ``H_field`` / ``H``, drainage thickness ``Hd``
        
            Notes
            -----
            Map availability depends on the underlying physics computation
            and whether the batch contains required inputs (coords, thickness
            field, etc.).
        
        Raises
        ------
        ValueError
            If the underlying physics computation requires missing inputs
            (for example, thickness) or inputs have incompatible shapes.
        
        Notes
        -----
        What this method is for
        ~~~~~~~~~~~~~~~~~~~~~~~
        Use this method to evaluate physics consistency independently of the
        supervised data loss. Typical use cases include:
        
        * monitoring PDE residual RMS values during training,
        * diagnosing unit or coordinate convention mismatches,
        * validating bounds and prior strength before long training runs,
        * generating physics maps for qualitative inspection.
        
        What this method is not
        ~~~~~~~~~~~~~~~~~~~~~~~
        This method does not compute or aggregate supervised metrics. It is
        intentionally physics-focused and ignores targets even if they are
        present in dataset elements.
        
        Aggregation semantics
        ~~~~~~~~~~~~~~~~~~~~~
        In Dataset mode, only scalar keys (loss and epsilon prefixes) are
        aggregated across batches. Residual maps and learned fields are not
        aggregated because they are spatially structured tensors; returning
        the last batch maps is a predictable, bounded-memory behavior.
        
        Examples
        --------
        Evaluate physics scalars over a validation dataset:
        
        >>> phys = model.evaluate_physics(val_ds, max_batches=10)
        >>> float(phys["epsilon_prior"])
        0.01
        
        Evaluate physics and retrieve last-batch maps:
        
        >>> phys = model.evaluate_physics(val_ds, return_maps=True, max_batches=1)
        >>> phys["R_gw"].shape
        TensorShape([B, H, 1])
        
        Evaluate a single batch dictionary:
        
        >>> phys = model.evaluate_physics(batch_dict, return_maps=False)
        >>> sorted([k for k in phys if k.startswith("loss_")])[:3]
        ['loss_bounds', 'loss_cons', 'loss_gw']
        
        Wrap numpy-like arrays into batches (mapping mode):
        
        >>> phys = model.evaluate_physics(inputs_np, batch_size=256, max_batches=5)
        
        See Also
        --------
        _evaluate_physics_on_batch
            Per-batch physics diagnostics wrapper.
        
        fusionlab.nn.pinn.geoprior.step_core.physics_core
            Shared physics computation used for diagnostics and training.
        
        References
        ----------
        .. [1] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
           Physics-informed neural networks: A deep learning framework
           for solving forward and inverse problems involving nonlinear
           partial differential equations. Journal of Computational
           Physics, 2019.
        
        .. [2] Bear, J. Dynamics of Fluids in Porous Media. Dover
           Publications, 1988.
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

    def split_data_predictions(
        self,
        data_tensor: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Split a combined supervised output tensor into subsidence and GWL
        components.
        
        GeoPrior models often compute a single "data head" tensor whose
        last dimension concatenates multiple supervised targets:
        
        .. math::
        
           y = [s, g]
        
        where :math:`s` is subsidence and :math:`g` is groundwater level
        (or a GWL-like driver). This helper slices the last axis into:
        
        * subsidence prediction tensor ``s_pred``
        * groundwater-level prediction tensor ``gwl_pred``
        
        The slicing is controlled by the model attributes
        ``self.output_subsidence_dim`` and ``self.output_gwl_dim``.
        
        Parameters
        ----------
        data_tensor : Tensor
            Combined supervised output tensor with last axis size
            ``output_subsidence_dim + output_gwl_dim``.
        
            Typical shapes include:
        
            * ``(B, H, D)`` for point predictions, where
              ``D = subs_dim + gwl_dim``.
            * ``(B, H, Q, D)`` for quantile predictions. In this case, the
              slicing is still applied on the last dimension ``D``.
        
        Returns
        -------
        s_pred : Tensor
            Subsidence slice from ``data_tensor[..., :output_subsidence_dim]``.
        
        gwl_pred : Tensor
            GWL slice from ``data_tensor[..., output_subsidence_dim:]``.
        
        Notes
        -----
        - This method performs a pure tensor slice and does not apply any
          unit conversions. Unit handling is managed by scaling helpers
          elsewhere.
        - If quantiles are present, the Q axis is preserved and only the
          last axis is split.
        
        Examples
        --------
        Point outputs:
        
        >>> y = tf.zeros([8, 3, 2])  # subs_dim=1, gwl_dim=1
        >>> s_pred, gwl_pred = model.split_data_predictions(y)
        >>> s_pred.shape, gwl_pred.shape
        (TensorShape([8, 3, 1]), TensorShape([8, 3, 1]))
        
        Quantile outputs:
        
        >>> yq = tf.zeros([8, 3, 3, 2])  # (B,H,Q,D)
        >>> s_pred, gwl_pred = model.split_data_predictions(yq)
        >>> s_pred.shape, gwl_pred.shape
        (TensorShape([8, 3, 3, 1]), TensorShape([8, 3, 3, 1]))
        
        See Also
        --------
        split_physics_predictions
            Split the physics-head tensor into (K, Ss, dlogtau, Q) logits.
        """

        s_pred = data_tensor[..., : self.output_subsidence_dim]
        gwl_pred = data_tensor[..., self.output_subsidence_dim :]
    
        return s_pred, gwl_pred
    
    def split_physics_predictions(
        self,
        phys_means_raw_tensor: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Split the combined physics-head tensor into per-field logits.
        
        GeoPrior models predict a compact "physics head" tensor whose last
        dimension concatenates the raw logits for multiple physics fields.
        This helper slices that tensor into:
        
        * ``K_logits``       : hydraulic conductivity logits
        * ``Ss_logits``      : specific storage logits
        * ``dlogtau_logits`` : relaxation time offset logits
        * ``Q_logits``       : optional forcing / source-term logits
        
        The canonical ordering is:
        
        .. math::
        
           p = [K, S_s, dlogtau, Q]
        
        where each component is typically 1-dimensional, i.e. shape
        ``(B, H, 1)`` per component.
        
        Parameters
        ----------
        phys_means_raw_tensor : Tensor
            Combined physics-head tensor. Expected shape is typically:
        
            * ``(B, H, P)`` where ``P`` is the total physics output
              dimension.
            * Some callers may supply tensors with additional axes, but the
              slicing always occurs along the last axis.
        
        Returns
        -------
        K_logits : Tensor
            Slice corresponding to the conductivity logits. Shape is
            ``(..., output_K_dim)`` and usually ``(B, H, 1)``.
        
        Ss_logits : Tensor
            Slice corresponding to the storage logits. Shape is
            ``(..., output_Ss_dim)`` and usually ``(B, H, 1)``.
        
        dlogtau_logits : Tensor
            Slice corresponding to the relaxation-time offset logits.
            Shape is ``(..., output_tau_dim)`` and usually ``(B, H, 1)``.
        
        Q_logits : Tensor
            Slice corresponding to the forcing/source logits. Shape is
            ``(..., output_Q_dim)`` and usually ``(B, H, 1)``.
        
            If Q is disabled or missing from the input tensor, a zeros
            tensor with the appropriate broadcastable shape is returned.
        
        Notes
        -----
        Backward compatibility and "always return Q"
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        This helper is designed so downstream physics code never needs to
        branch on whether Q exists.
        
        - If ``self.output_Q_dim <= 0``, Q is treated as disabled and a
          zeros tensor shaped like ``K_logits[..., :1]`` is returned.
        - If Q is enabled but ``phys_means_raw_tensor`` does not contain
          enough channels to include Q (older checkpoints), Q is returned
          as zeros with the correct shape.
        
        This allows PDE residual code to accept a consistent signature
        regardless of whether Q is actually trained.
        
        Shape and dimension conventions
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The slice widths are controlled by model attributes:
        
        * ``output_K_dim``
        * ``output_Ss_dim``
        * ``output_tau_dim``
        * ``output_Q_dim`` (optional)
        
        If your model uses multi-dimensional physics heads, the returned
        tensors will preserve those widths accordingly.
        
        Examples
        --------
        Standard case with Q present:
        
        >>> p = tf.zeros([8, 3, 4])  # [K,Ss,dlogtau,Q]
        >>> K, Ss, dlogtau, Q = model.split_physics_predictions(p)
        >>> K.shape, Ss.shape, dlogtau.shape, Q.shape
        (TensorShape([8, 3, 1]), TensorShape([8, 3, 1]),
         TensorShape([8, 3, 1]), TensorShape([8, 3, 1]))
        
        Backward-compatible case (no Q channel in stored tensor):
        
        >>> p_old = tf.zeros([8, 3, 3])  # [K,Ss,dlogtau]
        >>> K, Ss, dlogtau, Q = model.split_physics_predictions(p_old)
        >>> Q.shape
        TensorShape([8, 3, 1])
        
        See Also
        --------
        compose_physics_fields
            Map raw logits into bounded SI-consistent physics fields.
        
        q_to_gw_source_term_si
            Convert Q logits to the SI source term used in the GW PDE.
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
        
        This override extends :meth:`tf.keras.Model.compile` with explicit
        weights for each physics term used by GeoPrior PINN training, plus a
        global physics multiplier (``lambda_offset``) that can be scheduled
        during training.
        
        The GeoPrior training objective (as used by :meth:`train_step`) is:
        
        .. math::
        
           L_{total} = L_{data} + alpha(\text{offset\_mode}, \lambda_{offset})
                       \, L_{phys}
        
        where the physics objective is assembled from multiple components:
        
        .. math::
        
           L_{phys} =
               \lambda_{cons}   L_{cons}
             + \lambda_{gw}     L_{gw}
             + \lambda_{prior}  L_{prior}
             + \lambda_{smooth} L_{smooth}
             + \lambda_{mv}     L_{mv}
             + \lambda_{bounds} L_{bounds}
             + \lambda_{q}      L_{q}
        
        Each component corresponds to a residual (or penalty) computed in the
        shared physics core and summarized as mean-square values. The global
        multiplier :math:`alpha` is determined by ``self.offset_mode``:
        
        * ``offset_mode='mul'``  : :math:`alpha = \lambda_{offset}`
        * ``offset_mode='log10'``: :math:`alpha = 10^{\lambda_{offset}}`
        
        The value of ``lambda_offset`` is stored in a non-trainable scalar
        weight ``self._lambda_offset`` (created via ``add_weight``), which
        makes it safe to update during training from callbacks.
        
        Parameters
        ----------
        lambda_cons : float, default 1.0
            Weight for the consolidation residual loss :math:`L_{cons}`.
        
            This term penalizes the (scaled) consolidation residual
            :math:`R_{cons}` derived from the settlement relaxation update,
            and is typically computed as:
        
            .. math::
        
               L_{cons} = E[ R_{cons}^2 ]
        
        lambda_gw : float, default 1.0
            Weight for the groundwater-flow residual loss :math:`L_{gw}`.
        
            This term penalizes the (scaled) groundwater PDE residual
            :math:`R_{gw}` of the form:
        
            .. math::
        
               R_{gw} = S_s \, \partial_t h - \nabla \cdot (K \nabla h) - Q
        
            and is typically computed as:
        
            .. math::
        
               L_{gw} = E[ R_{gw}^2 ]
        
        lambda_prior : float, default 1.0
            Weight for the consistency prior loss :math:`L_{prior}`.
        
            This term ties the learned relaxation time :math:`tau` to a
            closure-based timescale :math:`tau_{phys}` computed from the
            learned fields and thickness. In the current implementation the
            residual is commonly expressed in log space:
        
            .. math::
        
               R_{prior} = \log(\tau) - \log(\tau_{phys})
        
            and the loss is:
        
            .. math::
        
               L_{prior} = E[ R_{prior}^2 ]
        
        lambda_smooth : float, default 1.0
            Weight for the smoothness prior loss :math:`L_{smooth}`.
        
            This term penalizes spatial roughness in the learned hydraulic
            fields, typically via squared first derivatives:
        
            .. math::
        
               L_{smooth} = E[ (\partial_x K)^2 + (\partial_y K)^2
                               + (\partial_x S_s)^2 + (\partial_y S_s)^2 ]
        
            It stabilizes training and encourages spatially coherent fields.
        
        lambda_mv : float, default 0.0
            Weight for the ``m_v`` consistency prior :math:`L_{mv}`.
        
            This term is designed to provide a direct learning signal for
            :math:`m_v` by aligning :math:`S_s` with the expected relation
            with compressibility and water unit weight:
        
            .. math::
        
               S_s \approx m_v \, \gamma_w
        
            A common residual is constructed in log space for stability:
        
            .. math::
        
               R_{mv} = \log(S_s) - \log(m_v \gamma_w)
        
            and the loss is:
        
            .. math::
        
               L_{mv} = E[ \rho(R_{mv}) ]
        
            where :math:`rho` may be a robust penalty (for example, Huber)
            depending on ``scaling_kwargs`` configuration. When set to a
            positive value, this term can help constrain :math:`m_v` in
            underdetermined settings.
        
        lambda_bounds : float, default 0.0
            Weight for the bounds penalty :math:`L_{bounds}`.
        
            This term penalizes violations of configured parameter bounds
            (for example, thickness and log-parameter ranges) provided in
            ``scaling_kwargs['bounds']``. When ``bounds_mode='soft'``, the
            penalty is differentiable and contributes to the objective:
        
            .. math::
        
               L_{bounds} = E[ R_{bounds}^2 ]
        
            When ``bounds_mode='hard'``, parameters may be clipped or
            projected by the physics mapping, and this weight is typically
            forced to zero.
        
        lambda_q : float, default 0.0
            Weight for the forcing regularization term :math:`L_{q}`.
        
            This term discourages excessive forcing magnitude by penalizing
            the mean-square of the SI source term :math:`Q` used in the GW
            residual:
        
            .. math::
        
               L_{q} = E[ Q^2 ]
        
            It is useful when a learnable forcing head is enabled and you
            want it to remain near zero unless required by data.
        
        lambda_offset : float, default 1.0
            Global physics multiplier stored in ``self._lambda_offset``.
        
            The effective multiplier applied to :math:`L_{phys}` is:
        
            * for ``offset_mode='mul'``  : :math:`alpha = \lambda_{offset}`
            * for ``offset_mode='log10'``: :math:`alpha = 10^{\lambda_{offset}}`
        
            Notes
            -----
            ``self._lambda_offset`` is a non-trainable scalar weight so it
            can be updated safely during training, for example:
        
            ``model._lambda_offset.assign(new_value)``
        
        mv_lr_mult : float, default 1.0
            Learning-rate multiplier applied to the gradient updates of the
            ``m_v`` log-parameter. This affects only the parameter update
            scaling, not the loss definition.
        
        kappa_lr_mult : float, default 1.0
            Learning-rate multiplier applied to the gradient updates of the
            ``kappa`` log-parameter (the closure/unit-conversion factor used
            by the timescale prior). This affects only parameter update
            scaling, not the loss definition.
        
        scale_mv_with_offset : bool, default False
            If True, multiply the :math:`L_{mv}` contribution by the global
            physics multiplier :math:`alpha` in addition to ``lambda_mv``.
        
            Notes
            -----
            This is useful when :math:`L_{mv}` should follow the same warmup
            schedule as other physics terms. If False, :math:`L_{mv}` is
            weighted only by ``lambda_mv``.
        
        scale_q_with_offset : bool, default True
            If True, multiply the :math:`L_{q}` contribution by the global
            physics multiplier :math:`alpha` in addition to ``lambda_q``.
        
            Notes
            -----
            This is commonly enabled so forcing regularization ramps in
            together with other physics terms during physics warmup.
        
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`tf.keras.Model.compile`, such as ``optimizer``, ``loss``,
            ``metrics``, ``run_eagerly``, ``jit_compile``, and so on.
        
        Returns
        -------
        self : GeoPriorSubsNet
            Returns the compiled model instance.
        
        Notes
        -----
        Physics-off behavior
        ~~~~~~~~~~~~~~~~~~~~
        If the model physics is disabled (for example, by PDE mode settings
        or a physics switch), this method forces all physics weights to
        neutral values regardless of the inputs:
        
        * ``lambda_prior = 0.0``
        * ``lambda_smooth = 0.0``
        * ``lambda_mv = 0.0``
        * ``lambda_q = 0.0``
        * ``lambda_bounds = 0.0``
        * ``self._lambda_offset = 1.0``
        
        This ensures that :meth:`train_step` and :meth:`test_step` remain
        stable and that logs do not contain misleading physics terms.
        
        Validation of lambda_offset
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        For ``offset_mode='mul'``, ``lambda_offset`` must be strictly
        positive. For ``offset_mode='log10'``, any real value is allowed and
        acts as a log10-scale controller.
        
        Scheduling lambda_offset
        ~~~~~~~~~~~~~~~~~~~~~~~~
        A recommended pattern is to keep individual ``lambda_*`` values
        fixed and schedule ``lambda_offset`` (warmup/ramp) using a callback.
        Because ``self._lambda_offset`` is a non-trainable TF weight, it is
        safe to update at runtime.
        
        Examples
        --------
        Compile with physics enabled and a moderate prior:
        
        >>> model.compile(
        ...     optimizer=tf.keras.optimizers.Adam(1e-3),
        ...     loss={"subs_pred": "mse", "gwl_pred": "mse"},
        ...     lambda_cons=1.0,
        ...     lambda_gw=1.0,
        ...     lambda_prior=2.0,
        ...     lambda_smooth=0.1,
        ...     lambda_bounds=0.01,
        ...     lambda_offset=0.1,
        ... )
        
        Disable forcing penalty and use a stronger smoothness prior:
        
        >>> model.compile(
        ...     optimizer=tf.keras.optimizers.Adam(5e-4),
        ...     loss={"subs_pred": "mse", "gwl_pred": "mse"},
        ...     lambda_q=0.0,
        ...     lambda_smooth=1.0,
        ... )
        
        Use log10 scaling for the global physics multiplier:
        
        >>> model.offset_mode = "log10"
        >>> model.compile(
        ...     optimizer=tf.keras.optimizers.Adam(1e-3),
        ...     loss={"subs_pred": "mse", "gwl_pred": "mse"},
        ...     lambda_offset=-1.0,  # physics multiplier = 0.1
        ... )
        
        See Also
        --------
        train_step
            Uses the configured lambdas to assemble the total loss and
            apply gradients.
        
        _physics_loss_multiplier
            Computes the global physics multiplier from ``offset_mode`` and
            ``self._lambda_offset``.
        
        fusionlab.nn.pinn.geoprior.step_core.physics_core
            Computes per-batch physics residuals and loss terms.
        
        References
        ----------
        .. [1] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
           Physics-informed neural networks: A deep learning framework
           for solving forward and inverse problems involving nonlinear
           partial differential equations. Journal of Computational
           Physics, 2019.
        
        .. [2] Bear, J. Dynamics of Fluids in Porous Media. Dover
           Publications, 1988.
        
        .. [3] Terzaghi, K. Theoretical Soil Mechanics. Wiley, 1943.
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
        r"""
        Export physics diagnostics as a flat payload.
        
        This helper collects physics diagnostics from a trained
        GeoPrior-style model and optionally persists them to disk.
        
        Internally, it calls :func:`gather_physics_payload` to iterate
        over ``dataset`` and evaluate physics maps and scalar summaries
        via :meth:`GeoPriorSubsNet.evaluate_physics` with
        ``return_maps=True``. The per-batch tensors are flattened and
        concatenated into 1D arrays suitable for scatter plots,
        histograms, and reproducibility artifacts.
        
        Parameters
        ----------
        dataset : iterable
            Batched iterable (typically a ``tf.data.Dataset``) yielding
            either ``inputs`` or ``(inputs, targets)``. Targets, if
            present, are ignored. Each ``inputs`` must contain the
            tensors required by :meth:`evaluate_physics` (notably the
            coordinate tensor and thickness field, depending on the
            model configuration).
        
        max_batches : int or None, default None
            Maximum number of batches to process. If None, consumes the
            entire iterable.
        
        save_path : str or None, default None
            If provided, write the payload to this location using
            :func:`save_physics_payload`. If ``save_path`` is a
            directory, a default filename is used by the saver.
        
        format : {'npz', 'csv', 'parquet'}, default 'npz'
            Output format for persistence. ``'npz'`` writes a compressed
            NumPy archive and a JSON sidecar metadata file.
        
        overwrite : bool, default False
            If False and ``save_path`` already exists, raise an error.
        
        metadata : dict or None, default None
            Optional user metadata to merge into the auto-generated
            provenance returned by :func:`default_meta_from_model`.
            User keys override defaults on conflict.
        
        random_subsample : float or None, default None
            If provided, randomly subsample the flat payload after it is
            gathered. Must be in ``(0, 1]`` and is interpreted as the
            fraction of rows to keep. This is useful to reduce file size
            for large grids.
        
        float_dtype : numpy dtype, default numpy.float32
            Dtype used when casting flattened arrays. Using float32 keeps
            files compact and is typically sufficient for diagnostics.
        
        log_fn : callable or None, default None
            Optional logger used by the progress helper (for example,
            ``print``). If None, the progress helper may be silent.
        
        **tqdm_kws
            Extra keyword arguments forwarded to the progress helper used
            inside :func:`gather_physics_payload`.
        
        Returns
        -------
        payload : dict[str, numpy.ndarray]
            Flat diagnostics payload with 1D arrays. The exact keys are
            defined by :func:`gather_physics_payload`, but typically
            include:
        
            - ``tau`` : effective relaxation time (seconds)
            - ``tau_prior`` / ``tau_closure`` : closure timescale (seconds)
            - ``K`` : effective hydraulic conductivity (m/s)
            - ``Ss`` : effective specific storage (1/m)
            - ``Hd`` : effective drainage thickness (m)
            - ``cons_res_vals`` : consolidation residual values
            - ``log10_tau`` and ``log10_tau_prior``
            - ``metrics`` : nested dict with summary scalars
        
        Notes
        -----
        - This routine does not change units. Unit consistency is a
          responsibility of the model physics and its ``scaling_kwargs``.
        - If ``return_maps=True`` is used inside
          :meth:`evaluate_physics`, maps are collected per batch and then
          flattened here. When saving, the payload is stored exactly as
          returned by the model.
        - Random subsampling is performed *after* concatenation, so it
          samples rows uniformly across all processed batches.
        
        See Also
        --------
        gather_physics_payload
            Core collector that builds the flat arrays.
        save_physics_payload
            Persist payload + metadata to disk.
        default_meta_from_model
            Build lightweight provenance metadata from a model.
        GeoPriorSubsNet.evaluate_physics
            Compute physics scalars and (optionally) maps.
        
        Examples
        --------
        >>> # ds is a batched tf.data.Dataset yielding (inputs, targets)
        >>> payload = model.export_physics_payload(
        ...     ds, max_batches=20, random_subsample=0.25
        ... )
        >>> # Save to disk (creates a .meta.json sidecar for npz/csv/parquet)
        >>> _ = model.export_physics_payload(
        ...     ds,
        ...     max_batches=50,
        ...     save_path="physics_payload.npz",
        ...     format="npz",
        ...     overwrite=True,
        ... )
        
        References
        ----------
        .. [1] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
           Physics-informed neural networks: A deep learning framework
           for solving forward and inverse problems involving nonlinear
           partial differential equations. Journal of Computational
           Physics, 2019.
        """

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
        r"""
        Load a previously saved physics payload.
        
        This is a thin convenience wrapper around
        :func:`load_physics_payload` from the diagnostics payload module.
        It reads the data file and its optional JSON sidecar metadata.
        
        Parameters
        ----------
        path : str
            Path to a saved payload. Supported extensions depend on the
            underlying loader and typically include ``.npz``, ``.csv``,
            and ``.parquet``. For formats that support it, a sidecar
            metadata file is expected at ``path + '.meta.json'``.
        
        Returns
        -------
        (payload, meta) : tuple(dict, dict)
            payload : dict[str, numpy.ndarray]
                Dictionary of arrays loaded from disk. Backward- and
                forward-compatible aliases may be added by the loader
                (for example, ensuring both ``tau_prior`` and
                ``tau_closure`` are present).
            meta : dict
                Metadata dictionary loaded from the JSON sidecar if found,
                otherwise an empty dict.
        
        Notes
        -----
        - This method performs I/O only. It does not validate that the
          payload matches a particular model instance.
        - If you saved with ``format='npz'``, the payload is loaded using
          NumPy. For CSV/Parquet, the loader typically uses pandas.
        
        See Also
        --------
        load_physics_payload
            The underlying loader that performs format dispatch.
        GeoPriorSubsNet.export_physics_payload
            Export and optionally save a payload.
        
        Examples
        --------
        >>> payload, meta = GeoPriorSubsNet.load_physics_payload(
        ...     "physics_payload.npz"
        ... )
        >>> list(payload)[:5]
        ['tau', 'tau_prior', 'K', 'Ss', 'Hd']
        
        References
        ----------
        .. [1] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
           Physics-informed neural networks: A deep learning framework
           for solving forward and inverse problems involving nonlinear
           partial differential equations. Journal of Computational
           Physics, 2019.
        """

        return load_physics_payload(path)
    
    def get_config(self) -> dict:
        r"""
        Return a Keras-serializable configuration for model reconstruction.
        
        This method extends :meth:`tf.keras.Model.get_config` to ensure
        ``GeoPriorSubsNet`` can be saved and reloaded with
        :meth:`tf.keras.models.load_model` (or :func:`keras.models.load_model`)
        while preserving the model's physics options and scaling pipeline.
        
        The returned dictionary contains:
        
        * the base configuration from :class:`~fusionlab.nn.BaseAttentive`
          (via ``super().get_config()``),
        * the supervised output layout (``output_dim``),
        * the resolved scaling configuration serialized as a Keras object,
        * GeoPrior-specific physics constructor arguments and flags.
        
        The output is designed to be JSON-serializable by Keras. Objects
        that are not plain JSON (for example, ``GeoPriorScalingConfig`` and
        scalar wrappers such as ``LearnableMV``) are included as Keras
        serialized objects via :func:`keras.saving.serialize_keras_object`.
        
        Returns
        -------
        config : dict
            A configuration dictionary that can be passed to
            :meth:`from_config` to reconstruct the model.
        
        Notes
        -----
        - ``output_dim`` is kept for compatibility with the BaseAttentive
          constructor signature. It is not a user-facing argument for the
          GeoPrior model; it is derived from:
        
          .. math::
        
             output\_dim = output\_subsidence\_dim + output\_gwl\_dim
        
        - ``scaling_kwargs`` is stored as a serialized Keras object
          representing the validated scaling configuration. This preserves
          the exact conventions (units, coordinate normalization, bounds)
          used during training and is critical for consistent inference.
        
        - This config does not include runtime-only state such as optimizer
          variables or training metrics. Those are handled by standard Keras
          checkpointing mechanisms.
        
        Examples
        --------
        Serialize and reconstruct manually:
        
        >>> cfg = model.get_config()
        >>> model2 = model.__class__.from_config(cfg)
        
        Save and reload through Keras:
        
        >>> model.save("geoprior.keras")
        >>> model2 = keras.models.load_model(
        ...     "geoprior.keras",
        ...     custom_objects={"GeoPriorSubsNet": GeoPriorSubsNet},
        ... )
        
        See Also
        --------
        from_config
            Reconstruct a model instance from the serialized config.
        
        keras.saving.serialize_keras_object
            Keras helper used to serialize non-JSON config objects.
        
        References
        ----------
        .. [1] Keras Team. Keras serialization and saving API documentation.
        """

        cfg = super().get_config()
    
        # Keep BaseAttentive compatible output_dim.
        cfg["output_dim"] = self._data_output_dim
    
        # Store scaling as a Keras object so load_model()
        # reconstructs the exact scaling pipeline.
        cfg["scaling_kwargs"] = K.serialize_keras_object(
            self.scaling_cfg,
        )
    
        # Physics + PINN knobs (constructor args).
        cfg.update(
            {
                "output_subsidence_dim": (
                    self.output_subsidence_dim
                ),
                "output_gwl_dim": self.output_gwl_dim,
                "pde_mode": self.pde_modes_active,
                "mv": self.mv_config,
                "kappa": self.kappa_config,
                "gamma_w": self.gamma_w_config,
                "h_ref": self.h_ref_config,
                "scale_pde_residuals": (
                    self.scale_pde_residuals
                ),
                "time_units": self.time_units,
                "use_effective_h": (
                    self.use_effective_thickness
                ),
                "hd_factor": self.Hd_factor,
                "offset_mode": self.offset_mode,
                "kappa_mode": self.kappa_mode,
                "bounds_mode": self.bounds_mode,
                "residual_method": self.residual_method,
                "verbose": self.verbose,
                "model_version": "3.2-GeoPrior",
            }
        )
    
        return cfg
    
    @classmethod
    def from_config(
        cls,
        config: dict,
        custom_objects=None,
    ):
        r"""
        Rebuild a GeoPrior model instance from a serialized configuration.
        
        This classmethod reconstructs the model from a configuration
        dictionary produced by :meth:`get_config` and used by the Keras
        serialization stack.
        
        The method performs three reconstruction steps:
        
        1. Build a ``custom_objects`` registry that includes all GeoPrior
           wrappers and scaling configuration classes needed for safe
           deserialization.
        
        2. Rehydrate wrapper objects stored as Keras-serialized dicts
           (``{"class_name": ..., "config": ...}``) for keys such as
           ``mv``, ``kappa``, ``gamma_w``, and ``h_ref``.
        
        3. Rehydrate the scaling configuration stored under
           ``scaling_kwargs`` if present as a Keras object.
        
        Finally, the method removes legacy/internal keys that are not part of
        the current constructor signature and returns ``cls(**config)``.
        
        Parameters
        ----------
        config : dict
            Serialized configuration dictionary. Typically produced by
            :meth:`get_config` and passed by Keras during deserialization.
        
        custom_objects : dict or None, default None
            Optional mapping used by Keras to resolve custom layers, models,
            and config objects. If None, an internal registry is created and
            merged with any user-provided entries.
        
        Returns
        -------
        model : GeoPriorSubsNet
            A reconstructed model instance equivalent to the original model
            at save time (architecture and configuration). Weights are loaded
            by Keras separately when using :func:`keras.models.load_model`.
        
        Notes
        -----
        - This method is designed to be robust to older saved configs by
          explicitly dropping keys that were used by previous GeoPrior/PINN
          variants (for example, legacy groundwater coefficient keys and
          internal version markers).
        
        - The deserialization process relies on Keras helpers and the
          ``custom_objects`` registry. If you have custom subclasses or
          external layers referenced inside ``architecture_config``, you
          must provide them in ``custom_objects`` or register them with
          Keras before loading.
        
        - If scaling deserialization fails, the method raises the underlying
          exception because the scaling configuration is required for
          consistent unit handling and PDE residual computation.
        
        Examples
        --------
        Reconstruct from a saved config dictionary:
        
        >>> cfg = model.get_config()
        >>> model2 = GeoPriorSubsNet.from_config(
        ...     cfg,
        ...     custom_objects={"GeoPriorSubsNet": GeoPriorSubsNet},
        ... )
        
        Load a saved model with explicit custom_objects:
        
        >>> model2 = keras.models.load_model(
        ...     "geoprior.keras",
        ...     custom_objects={
        ...         "GeoPriorSubsNet": GeoPriorSubsNet,
        ...         "GeoPriorScalingConfig": GeoPriorScalingConfig,
        ...     },
        ... )
        
        See Also
        --------
        get_config
            Produce the configuration dictionary used for reconstruction.
        
        keras.saving.deserialize_keras_object
            Keras helper used to rehydrate serialized config objects.
        
        References
        ----------
        .. [1] Keras Team. Keras serialization and saving API documentation.
        """

        if custom_objects is None:
            custom_objects = {}
    
        # Register wrappers for deserialization safety.
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
                "GeoPriorScalingConfig": (
                    GeoPriorScalingConfig
                ),
            }
        )
    
        # Rehydrate scalar wrappers when saved as
        # {"class_name": ..., "config": ...}.
        for key in ("mv", "kappa", "gamma_w", "h_ref"):
            obj = config.get(key, None)
            if isinstance(obj, dict) and "class_name" in obj:
                config[key] = deserialize_keras_object(
                    obj,
                    custom_objects=custom_objects,
                )
    
        # Rehydrate scaling config if it is a Keras object.
        sk = config.get("scaling_kwargs", None)
        if isinstance(sk, dict) and "class_name" in sk:
            try:
                config["scaling_kwargs"] = (
                    deserialize_keras_object(
                        sk,
                        custom_objects=custom_objects,
                    )
                )
            except Exception as err:
                logger.exception(
                    f"Failed to deserialize scaling_kwargs: {err}"
                )
                raise
    
        # Drop legacy / internal keys not in __init__.
        config.pop("K", None)
        config.pop("Ss", None)
        config.pop("Q", None)
        config.pop("pinn_coefficient_C", None)
        config.pop("gw_flow_coeffs", None)
        config.pop("output_dim", None)
        config.pop("model_version", None)
    
        return cls(**config)

GeoPriorSubsNet.__doc__ = GEOPRIOR_SUBSNET_DOC

@register_keras_serializable(
    "fusionlab.nn.pinn.models.geoprior",
    name="PoroElasticSubsNet"
 )
class PoroElasticSubsNet(GeoPriorSubsNet):

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


PoroElasticSubsNet.__doc__ = POROELASTIC_SUBSNET_DOC
