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
        process_pinn_inputs, default_scales, scale_residual,
        rate_to_per_second, 
        
    )
    from .utils import  process_pde_modes,  _get_coords #, extract_txy_in,
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
        kappa_mode: str = "bar",   # {"bar", "kb"}  # κ̄ vs κ_b
        offset_mode: str = "mul",  # {"mul", "log10"}
        bounds_mode : str ="soft", 
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

    def _to_si_thickness(self, H_model: Tensor) -> Tensor:
        a, b = self._affine_from_cfg(
            scale_key="H_scale_si",
            bias_key="H_bias_si",
            meta_keys=("H_z_meta",),
            unit_key="thickness_unit_to_si",
        )
        return H_model * a + b
    
    def _deg_to_m(self, axis: str) -> Tensor:
        """
        If coords are lon/lat degrees, returns meters-per-degree factor.
    
        Parameters
        ----------
        axis : {"x", "y"}
            "x" for longitude, "y" for latitude.
    
        Returns
        -------
        Tensor
            meters-per-degree conversion factor as tf.float32.
    
        Raises
        ------
        ValueError
            If axis is not one of {"x","y"}, or if coords_in_degrees=True but
            the required deg_to_m_* key is missing/invalid.
        """
        if axis not in ("x", "y"):
            raise ValueError(
                f"_deg_to_m(axis): axis must be 'x' or 'y', got {axis!r}."
            )
    
        cfg = self.scaling_kwargs or {}
    
        # UTM / already-in-meters mode
        if not bool(cfg.get("coords_in_degrees", False)):
            return tf_constant(1.0, tf_float32)
    
        key = "deg_to_m_lon" if axis == "x" else "deg_to_m_lat"
        val = cfg.get(key, None)
    
        if val is None:
            raise ValueError(
                "coords_in_degrees=True but missing "
                f"scaling_kwargs[{key!r}]. "
                "Provide deg_to_m_lon/deg_to_m_lat"
                " (meters per degree) from Stage-1."
            )
    
        try:
            v = float(val)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid {key!r}={val!r}. Must be "
                "a finite float (meters per degree)."
            )
    
        # Basic sanity: must be positive and finite
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(
                f"Invalid {key!r}={v}. Must be positive"
                " and finite (meters per degree)."
            )
    
        return tf_constant(v, tf_float32)
    
    def _affine_from_cfg(
        self,
        *,
        scale_key: str,
        bias_key: str,
        meta_keys: Tuple[str, ...] = (),
        unit_key: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return (a,b) for y_si = y_model * a + b.
    
        Priority:
          1) explicit {scale_key,bias_key}
          2) meta dict(s): {"mu"/"mean", "sigma"/"std"} in SI units
          3) unit-only conversion (b=0)
        """
        cfg = self.scaling_kwargs or {}
    
        a = cfg.get(scale_key, None)
        b = cfg.get(bias_key, None)
    
        # 1) explicit affine
        if a is not None or b is not None:
            a = 1.0 if a is None else float(a)
            b = 0.0 if b is None else float(b)
            return tf_constant(a, tf_float32), tf_constant(b, tf_float32)
    
        # 2) meta fallback
        for mk in meta_keys:
            meta = cfg.get(mk, None)
            if isinstance(meta, dict):
                mu  = meta.get("mu",  meta.get("mean", None))
                sig = meta.get("sigma", meta.get("std", None))
                if mu is not None and sig is not None:
                    return tf_constant(
                        float(sig), tf_float32), tf_constant(float(mu), tf_float32)
    
        # 3) unit-only
        if unit_key is not None:
            u = float(cfg.get(unit_key, 1.0))
            return tf_constant(u, tf_float32), tf_constant(0.0, tf_float32)
    
        return tf_constant(1.0, tf_float32), tf_constant(0.0, tf_float32)
    
    def _resolve_gwl_dyn_index(self) -> int:
        sk = self.scaling_kwargs or {}
    
        # 1) Best: Stage-1 wrote it explicitly
        idx = sk.get("gwl_dyn_index", None)
        if idx is not None:
            return int(idx)
    
        # 2) Fallback: compute from names + gwl_col
        names = sk.get("dynamic_feature_names", None)
        gwl_col = sk.get("gwl_col", None)
    
        if names is not None and gwl_col is not None:
            names = list(names)
            if gwl_col in names:
                return int(names.index(gwl_col))
    
        raise ValueError(
            "Cannot resolve GWL/depth channel in inputs['dynamic_features']. "
            "Provide scaling_kwargs['gwl_dyn_index'] (recommended) or "
            "scaling_kwargs['dynamic_feature_names'] + scaling_kwargs['gwl_col']."
        )

    def _validate_scaling_kwargs(self):
        sk = self._scaling_kwargs or {}
    
        # Coords sanity
        if sk.get("coords_in_degrees", False):
            raise ValueError(
                "coords_in_degrees=True but you are feeding UTM meters. "
                "Set coords_in_degrees=False."
            )
    
        # Normalization sanity
        if sk.get("coords_normalized", False) and not sk.get("coord_ranges"):
            raise ValueError(
                "coords_normalized=True but coord_ranges is missing. "
                "Either provide coord_ranges or set coords_normalized=False."
            )
    
        # Time sanity
        if "time_units" not in sk:
            raise ValueError(
                "time_units missing in scaling_kwargs. "
                "Set time_units='year' if t is in years."
            )

    def _get_gwl_dyn_index(self) -> int:
        # cache after first resolve
        idx = getattr(self, "_gwl_dyn_index", None)
        if idx is None:
            idx = self._resolve_gwl_dyn_index()
            self._gwl_dyn_index = int(idx)
        return int(idx)
    
    def _slice_dynamic_channel(self, Xh, idx: int):
        # Xh: (B,T,F)
        idx_t = tf_cast(idx, tf_int32)
        F = tf_shape(Xh)[-1]
        tf_debugging.assert_less(
            idx_t, F,
            message="gwl_dyn_index out of range for inputs['dynamic_features']"
        )
        return Xh[:, :, idx_t:idx_t + 1]  # (B,T,1)
    
            
    def _to_si_head(self, h_model: Tensor) -> Tensor:
        a, b = self._affine_from_cfg(
            scale_key="head_scale_si",
            bias_key="head_bias_si",
            meta_keys=("head_z_meta", "gwl_z_meta"),     
            unit_key="head_unit_to_si",
        )
        return h_model * a + b
    
    def _gwl_to_head_m(self, v_m, inputs=None):
        sk = self.scaling_kwargs or {}
        kind = str(sk.get("gwl_kind", "head")).lower()
        sign = str(sk.get("gwl_sign", "down_positive")).lower()
        use_proxy = bool(sk.get("use_head_proxy", True))
        z_surf_col = sk.get("z_surf_col", None)
    
        # If it's already head (meters), done.
        if kind == "head":
            return v_m
    
        # Otherwise interpret v_m as depth below ground surface (meters)
        depth_m = v_m if sign == "down_positive" else -v_m
    
        # If z_surf is provided as an input, compute true head: h = z_surf - depth
        z_surf = None
        if (inputs is not None) and z_surf_col:
            z_surf = inputs.get(z_surf_col, None)
            if z_surf is not None:
                z_surf = tf_cast(z_surf, tf_float32)
    
        if z_surf is not None:
            return z_surf - depth_m
    
        # No z_surf: use proxy head (recommended)
        return -depth_m if use_proxy else depth_m


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
        
    def _get_h_ref_si(self, inputs, like):
        """
        Return reference head :math:`h_ref` in SI units (meters), broadcastable
        to the shape of `like` (typically ``(B, H, 1)``).
    
        This reference is used to define drawdown, e.g.
        :math:`\\Delta h = \\max(h_{ref} - h, 0)`.
    
        Priority
        --------
        1) Explicit tensor provided in `inputs`
           (keys: ``"h_ref_si"``, ``"head_ref_si"``, ``"h_ref"``, ``"head_ref"``).
    
        2) Automatic per-sample reference derived from the *last historical*
           groundwater observation in ``inputs["dynamic_features"]``
           (recommended; yields a rolling baseline at forecast start).
    
        3) Fixed scalar fallback ``self.h_ref`` (SI meters).
    
        Parameters
        ----------
        inputs : dict or None
            Batch inputs dictionary.
        like : Tensor
            Tensor whose shape should be matched via broadcasting
            (e.g., PDE head ``h_mean`` with shape ``(B, H, 1)``).
    
        Returns
        -------
        Tensor
            Reference head in meters (SI), broadcast to ``tf.shape(like)``.
        """
        # ------------------------------------------------------------------
        # 0) Optional mode gate: if explicitly configured as fixed, skip
        #    history-derived reference. (Keeps backward compatibility while
        #    enabling "auto" semantics via FixedHRef.mode.)
        # ------------------------------------------------------------------
        mode = getattr(getattr(self, "h_ref_config", None), "mode", "auto")
        mode = "fixed" if str(mode).strip().lower() == "fixed" else "auto"
    
        # ------------------------------------------------------------------
        # 1) Explicit per-batch reference provided by the caller
        #    Accept multiple key aliases; normalize to (B, 1, 1) so that
        #    broadcasting to (B, H, 1) is always safe.
        # ------------------------------------------------------------------
        for k in ("h_ref_si", "head_ref_si", "h_ref", "head_ref"):
            if inputs is not None and k in inputs and inputs[k] is not None:
                h_ref = tf_cast(inputs[k], tf_float32)
    
                # Normalize rank for robust broadcasting:
                #   (B,)   -> (B,1,1)
                #   (B,1)  -> (B,1,1)
                #   (B,1,1) or (B,H,1) -> keep
                r = tf_rank(h_ref)
                h_ref = tf_cond(
                    tf_equal(r, 1),
                    lambda: h_ref[:, None, None],
                    lambda: tf_cond(
                        tf_equal(r, 2),
                        lambda: h_ref[:, None, :],
                        lambda: h_ref,
                    ),
                )
                return h_ref + tf_zeros_like(like)  # -> shape(like)
    
        # ------------------------------------------------------------------
        # 2) Automatic reference from last historical GWL (preferred)
        #    Only attempted when mode != "fixed".
        #
        #    Expected:
        #      Xh = dynamic_features with shape (B, T, F)
        #      gwl_series = slice channel -> (B, T, 1)
        #      gwl_hist   = last timestep -> (B, 1, 1) keeps singleton time axis
        # ------------------------------------------------------------------
        if (
            mode != "fixed"
            and inputs is not None
            and "dynamic_features" in inputs
            and inputs["dynamic_features"] is not None
        ):
            Xh = tf_cast(inputs["dynamic_features"], tf_float32)  # (B,T,F)
            self._assert_dynamic_names_match_tensor(Xh)
    
            idx = self._get_gwl_dyn_index()
            gwl_series = self._slice_dynamic_channel(Xh, idx)      # (B,T,1)
    
            # Keep singleton time axis for safe broadcast against (B,H,1).
            gwl_hist = gwl_series[:, -1:, :]                       # (B,1,1)
    
            # Convert history GWL to SI meters using the same affine mapping
            # used for PDE outputs.
            gwl_hist_si = self._to_si_head(gwl_hist)               # (B,1,1)
    
            # Convert GWL->head in meters using the same logic as PDE:
            # - if GWL_KIND="depth_bgs" and USE_HEAD_PROXY=True -> head ≈ -depth
            # - if z_surf is available -> head = z_surf - depth
            h_ref_si = self._gwl_to_head_m(gwl_hist_si, inputs=inputs)  # (B,1,1)
    
            return h_ref_si + tf_zeros_like(like)                  # -> (B,H,1)
    
        # ------------------------------------------------------------------
        # 3) Final fallback: fixed scalar reference head (SI meters)
        #    Ensure shape (1,1,1) so broadcasting is clean.
        # ------------------------------------------------------------------
        h0 = tf_cast(getattr(self, "h_ref", 0.0), tf_float32)       # scalar
        h0 = h0[None, None, None]                                  # (1,1,1)
        return h0 + tf_zeros_like(like)                             # -> shape(like)
        
    def _to_si_subsidence(self, s_model: Tensor) -> Tensor:
        a, b = self._affine_from_cfg(
            scale_key="subs_scale_si",
            bias_key="subs_bias_si",
            meta_keys=("subs_z_meta",),
            unit_key="subs_unit_to_si",
        )
        return s_model * a + b
    

    def _coord_ranges(self):
        cfg = self.scaling_kwargs or {}
        if not bool(cfg.get("coords_normalized", False)):
            return None, None, None
    
        r = cfg.get("coord_ranges", {}) or {}
    
        def _get(name, *alts):
            v = r.get(name, None)
            if v is None:
                for a in alts:
                    v = cfg.get(a, None)
                    if v is not None:
                        break
            return (None if v is None else float(v))
    
        tR = _get("t", "t_range", "coord_range_t")
        xR = _get("x", "x_range", "coord_range_x")
        yR = _get("y", "y_range", "coord_range_y")
        return tR, xR, yR


    def _build_attentive_layers(self):
        super()._build_attentive_layers()
        self._build_physics_layers()
    
    def _build_physics_layers(self):
        logK_min, logK_max, logSs_min, logSs_max = self._get_log_bounds(
            as_tensor=False
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
            # sigmoid(0)=0.5 => midpoint in [min,max]
            k_bias = 0.0
            ss_bias = 0.0
        else:
            # soft/log-field mode: base outputs are logK/logSs directly
            k_bias = float(logK0)
            ss_bias = float(logSs0)
    
        self.K_head = Dense(
            1, name="K_head",
            kernel_initializer="zeros",
            bias_initializer=Constant(k_bias)
        )
        self.Ss_head = Dense(
            1, name="Ss_head",
            kernel_initializer="zeros",
            bias_initializer=Constant(ss_bias)
        )
        self.tau_head = Dense(
            1, name="tau_head",
            kernel_initializer="zeros",
            bias_initializer=Constant(0.0)
        )  # delta_log_tau=0 -> tau=tau_phys
    
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
        self.gamma_w = tf_constant(
            float(self.gamma_w_config.value),
            dtype=tf_float32,
        )
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
        
        # ------------------------------------------------------------------
        # 1. Unpack and validate all PINN inputs
        # ------------------------------------------------------------------
        # process_pinn_inputs standardizes the interface and returns
        # tensors with consistent shapes suitable for the encoder/decoder.
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
    
        # Build coordinate tensor (B, H, 3) used by the decoder and
        # physics head. This is the explicit (t,x,y) signal that the
        # PINN can differentiate with respect to.
        coords_for_decoder = tf_concat([t, x, y], axis=-1)
    
        # Lightweight shape checks to catch glaring mistakes early:
        #   - consistent feature dims,
        #   - correct forecast horizon length, etc.
        check_inputs(
            dynamic_inputs=dynamic_features,
            static_inputs=static_features,
            future_inputs=future_features,
            dynamic_input_dim=self.dynamic_input_dim,
            static_input_dim=self.static_input_dim,
            future_input_dim=self.future_input_dim,
            forecast_horizon=self.forecast_horizon,
            verbose=0,  # set >0 to debug shape mismatches
        )
        logger.debug(
            "Input shapes after validation:"
            f" S={getattr(static_features, 'shape', 'None')}, "
            f"D={getattr(dynamic_features, 'shape', 'None')}, "
            f"F={getattr(future_features, 'shape', 'None')}"
        )
    
        # Store H_field on the instance so that downstream methods
        # (e.g. evaluate_physics, diagnostic payload exports) can
        # access the thickness field used in this forward pass.
        self.H_field = H_field
    
        # Further strict validation for the encoder/decoder,
        # handling possible 0-width static/future features etc.
        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=[static_features, dynamic_features, future_features],
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            mode="strict",
            verbose=0,
        )
    
        # ------------------------------------------------------------------
        # 2. Run the shared encoder/decoder core
        # ------------------------------------------------------------------
        # The overridden core returns:
        #   - data_features_2d: (B, U_data)   for data decoder
        #   - phys_features_raw_3d: (B, H, U) for physics head
        data_features_2d, phys_features_raw_3d = self.run_encoder_decoder_core(
            static_input=static_p,
            dynamic_input=dynamic_p,
            future_input=future_p,
            coords_input=coords_for_decoder,
            training=training,
        )
    
        # ------------------------------------------------------------------
        # 3. Data path: decode subsidence and head
        # ------------------------------------------------------------------
        # Coordinate-based correction should affect data outputs too
        mlp_corr = self.coord_mlp(coords_for_decoder, training=training)        # (B,H,gwl_dim)
        s_corr   = self.subs_coord_mlp(coords_for_decoder, training=training)   # (B,H,subs_dim)

        decoded_data_means = self.multi_decoder(
            data_features_2d,
            training=training,
        )
        decoded_data_means = decoded_data_means + tf_concat(
            [s_corr, mlp_corr], axis=-1)
        final_data_predictions = decoded_data_means
        if self.quantiles is not None:
            # Expand means into full predictive distributions
            final_data_predictions = self.quantile_distribution_modeling(
                decoded_data_means,
                training=training,
            )
    
        # ------------------------------------------------------------------
        # 4. Physics path: decode K, Ss, tau (pre-positivity)
        # ------------------------------------------------------------------
        # decoded_physics_means_raw = self.physics_mean_head(
        #     phys_features_raw_3d,
        #     training=training,
        # )
        K_raw   = self.K_head(phys_features_raw_3d, training=training)
        Ss_raw  = self.Ss_head(phys_features_raw_3d, training=training)
        tau_raw = self.tau_head(phys_features_raw_3d, training=training)
        
        decoded_physics_means_raw = tf_concat([K_raw, Ss_raw, tau_raw], axis=-1)

        # ------------------------------------------------------------------
        # 5. Return all components for downstream use
        # ------------------------------------------------------------------
        # Note: positivity and splitting into individual fields is done
        # by split_physics_predictions(...) inside train_step /
        # evaluate_physics.
        return {
            "data_final": final_data_predictions,
            "data_mean": decoded_data_means,
            "phys_mean_raw": decoded_physics_means_raw,
        }
    
    def train_step(self, data):
        """
        Custom Keras training step for GeoPriorSubsNet.
    
        Key design choice (fix for "all-zero" GW derivatives):
        ------------------------------------------------------
        We differentiate PDE outputs w.r.t. the *coords tensor*
        itself (inputs["coords"]), not w.r.t. separately
        extracted (t, x, y) tensors. This guarantees that the
        derivative target is the same object the forward pass
        depends on, and avoids the "0.0 * t" binding hack that
        forces gradients to be exactly zero.
        """
        # --------------------------------------------------------------
        # 0) Unpack batch and normalize target dict keys
        # --------------------------------------------------------------
        inputs, targets = data
    
        if isinstance(targets, dict):
            targets = rename_dict_keys(
                targets.copy(),
                param_to_rename={
                    "subsidence": "subs_pred",
                    "gwl": "gwl_pred",
                },
            )
    
        # --------------------------------------------------------------
        # 1) Fetch H_field (required by consolidation + tau prior)
        # --------------------------------------------------------------
        H_field_in = get_tensor_from(
            inputs,
            "H_field",
            "soil_thickness",
            auto_convert=True,
        )
        if H_field_in is None:
            raise ValueError(
                "train_step requires 'H_field' (or "
                "'soil_thickness') in inputs."
            )
    
        H_field = tf_convert_to_tensor(
            H_field_in,
            dtype=tf_float32,
        )
        self.H_field = H_field
        H_si = self._to_si_thickness(H_field)
    
        self._validate_scaling_kwargs()
        # --------------------------------------------------------------
        # 2) GradientTape: forward pass + data loss + physics
        # --------------------------------------------------------------
        with GradientTape(persistent=True) as tape:
            # ----------------------------------------------------------
            # 2.1 Watch coords (the exact tensor the forward uses)
            # ----------------------------------------------------------
            coords = tf_convert_to_tensor(
                _get_coords(inputs),
                dtype=tf_float32,
            )
    
            # Support (B, 3) coords by expanding to (B, 1, 3)
            if coords.shape.rank == 2:
                coords = tf_expand_dims(coords, axis=1)
    
            tape.watch(coords)
    
            # # Convenience slices (NOT watched separately)
            t = coords[..., 0:1]
            # x = coords[..., 1:2]
            # y = coords[..., 2:3]
    
            # ----------------------------------------------------------
            # 2.2 Forward pass through the model
            # ----------------------------------------------------------
            outputs = self(inputs, training=True)
    
            # ----------------------------------------------------------
            # 2.3 Supervised data loss
            # ----------------------------------------------------------
            s_pred, gwl_pred = self.split_data_predictions(
                outputs["data_final"]
            )
            y_pred = {
                "subs_pred": s_pred,
                "gwl_pred": gwl_pred,
            }
    
            data_loss = self.compiled_loss(
                y_true=targets,
                y_pred=y_pred,
                regularization_losses=self.losses,
            )
    
            # ----------------------------------------------------------
            # 3) Physics fields (K, Ss, tau) + SI transforms
            # ----------------------------------------------------------
            (
                s_mean,
                gwl_mean,
                K_base,
                Ss_base,
                tau_base,
            ) = self.split_physics_predictions(outputs)
    
            # NOTE:
            # s_mean / gwl_mean already include coord corrections
            # because call() adds coord_mlp/subs_coord_mlp into
            # data_mean. Do NOT add them again here.
    
            v_m = self._to_si_head(gwl_mean)
            h_si = self._gwl_to_head_m(v_m, inputs=inputs)
            s_si = self._to_si_subsidence(s_mean)
    
            # Compose positive physics fields using coords
            (
                K_field,
                Ss_field,
                tau_field,
                tau_phys,
                Hd_eff,
                delta_log_tau,
                logK,
                logSs,
            ) = self._compose_physics_fields(
                coords_flat=coords,
                H_si=H_si,
                K_base=K_base,
                Ss_base=Ss_base,
                tau_base=tau_base,
                training=True,
            )
    
            # Save last fields for diagnostics / export
            self.K_field = K_field
            self.Ss_field = Ss_field
            self.tau_field = tau_field
    
            # ----------------------------------------------------------
            # 4) PDE derivatives via AD w.r.t coords (B,H,3)
            # ----------------------------------------------------------
            s_pde = s_si
            h_pde = h_si
    
            ds_dcoords = tape.gradient(s_pde, coords)
            dh_dcoords = tape.gradient(h_pde, coords)
    
            if (ds_dcoords is None) or (dh_dcoords is None):
                raise ValueError(
                    "PDE gradients are None. This means the "
                    "forward graph is not connected to "
                    "inputs['coords']."
                )
    
            ds_dt = ds_dcoords[..., 0:1]
            dh_dt = dh_dcoords[..., 0:1]
            dh_dx = dh_dcoords[..., 1:2]
            dh_dy = dh_dcoords[..., 2:3]
    
            # div(K * grad(h))
            K_dh_dx = K_field * dh_dx
            K_dh_dy = K_field * dh_dy
    
            dKdhx_dcoords = tape.gradient(K_dh_dx, coords)
            dKdhy_dcoords = tape.gradient(K_dh_dy, coords)
    
            if (dKdhx_dcoords is None) or (dKdhy_dcoords is None):
                raise ValueError(
                    "Second-order PDE gradients are None. "
                    "Check that K_field and h_pde depend on "
                    "coords in the forward graph."
                )
    
            d_K_dh_dx_dx = dKdhx_dcoords[..., 1:2]
            d_K_dh_dy_dy = dKdhy_dcoords[..., 2:3]
    
            # Smoothness grads (only x,y are used)
            dK_dcoords = tape.gradient(K_field, coords)
            dSs_dcoords = tape.gradient(Ss_field, coords)
    
            if (dK_dcoords is None) or (dSs_dcoords is None):
                raise ValueError(
                    "K/Ss spatial gradients are None. "
                    "Check K_field/Ss_field dependence on "
                    "coords (via coord injection or K_coord_mlp)."
                )
    
            dK_dx = dK_dcoords[..., 1:2]
            dK_dy = dK_dcoords[..., 2:3]
            dSs_dx = dSs_dcoords[..., 1:2]
            dSs_dy = dSs_dcoords[..., 2:3]
    
            # ----------------------------------------------------------
            # 4.1 Chain-rule correction for normalized/degree coords
            # ----------------------------------------------------------
            coords_norm = bool(
                self.scaling_kwargs.get("coords_normalized", False)
            )
            coords_deg = bool(
                self.scaling_kwargs.get("coords_in_degrees", False)
            )
    
            tR, xR, yR = self._coord_ranges()
            deg2m_x = self._deg_to_m("x")
            deg2m_y = self._deg_to_m("y")
    
            # --- time scaling: undo normalization then convert to seconds
            if coords_norm and tR:
                ds_dt = ds_dt / tR
                dh_dt = dh_dt / tR
    
            ds_dt = rate_to_per_second(ds_dt, self.time_units)
            dh_dt = rate_to_per_second(dh_dt, self.time_units)
    
            # --- spatial scaling
            if coords_norm:
                if xR:
                    xRt = xR * deg2m_x
                    dh_dx = dh_dx / xRt
                    d_K_dh_dx_dx = d_K_dh_dx_dx / (xRt * xRt)
                    dK_dx = dK_dx / xRt
                    dSs_dx = dSs_dx / xRt
                if yR:
                    yRt = yR * deg2m_y
                    dh_dy = dh_dy / yRt
                    d_K_dh_dy_dy = d_K_dh_dy_dy / (yRt * yRt)
                    dK_dy = dK_dy / yRt
                    dSs_dy = dSs_dy / yRt
    
            elif coords_deg:
                dh_dx = dh_dx / deg2m_x
                d_K_dh_dx_dx = d_K_dh_dx_dx / (deg2m_x * deg2m_x)
                dK_dx = dK_dx / deg2m_x
                dSs_dx = dSs_dx / deg2m_x
    
                dh_dy = dh_dy / deg2m_y
                d_K_dh_dy_dy = d_K_dh_dy_dy / (deg2m_y * deg2m_y)
                dK_dy = dK_dy / deg2m_y
                dSs_dy = dSs_dy / deg2m_y
    
            # ----------------------------------------------------------
            # 5) Residuals for each physics component
            # ----------------------------------------------------------
            gw_res = self._compute_gw_flow_residual(
                dh_dt,
                d_K_dh_dx_dx,
                d_K_dh_dy_dy,
                Ss_field,
                Q=0.0,
            )
    
            cons_res = self._compute_consolidation_residual(
                ds_dt,
                s_pde,
                h_pde,
                H_si,
                tau_field,
                Ss_field=Ss_field,
                inputs=inputs,
            )
    
            prior_res = delta_log_tau
    
            smooth_res = self._compute_smoothness_prior(
                dK_dx,
                dK_dy,
                dSs_dx,
                dSs_dy,
                K_field=K_field,
                Ss_field=Ss_field,
            )
    
            mv_prior_res = self._compute_mv_prior(Ss_field)
            loss_mv = tf_reduce_mean(tf_square(mv_prior_res))
    
            R_H, R_K, R_Ss = self._compute_bounds_residual(
                K_field,
                Ss_field,
                H_si,
            )
            bounds_res = tf_concat([R_H, R_K, R_Ss], axis=-1)
            loss_bounds = tf_reduce_mean(tf_square(bounds_res))
    
            if self._physics_off():
                cons_res = tf_zeros_like(cons_res)
                gw_res = tf_zeros_like(gw_res)
                prior_res = tf_zeros_like(prior_res)
                smooth_res = tf_zeros_like(smooth_res)
                loss_mv = tf_zeros_like(loss_mv)
                bounds_res = tf_zeros_like(bounds_res)
                loss_bounds = tf_zeros_like(loss_bounds)
    
            # ----------------------------------------------------------
            # 6) Residual scaling (dimensionless normalization)
            # ----------------------------------------------------------
            # (B,1,1) -> broadcasts to (B,H,1)
            h_ref_si = self._get_h_ref_si(inputs, like=h_pde)  
            if (not self._physics_off()) and self.scale_pde_residuals:
                scales = self._compute_scales(
                            t=t,
                            s_mean=s_pde,          # (B,H,1) in m
                            h_mean=h_pde,          # (B,H,1) in m
                            K_field=K_field,
                            Ss_field=Ss_field,
                            ds_dt=ds_dt,           # (B,H,1) in m/s  (already per-second)
                            tau_field=tau_field,   # (B,H,1) in s
                            H_field=H_si,          # (B,H,1) in m
                            h_ref_si=h_ref_si, 
                            Q=0.0,
                        )

                cons_res = scale_residual(
                    cons_res,
                    scales.get("cons_scale"),
                )
                gw_res = scale_residual(
                    gw_res,
                    scales.get("gw_scale"),
                )
    
            # ----------------------------------------------------------
            # 7) Composite physics loss and total loss
            # ----------------------------------------------------------
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
    
        # --------------------------------------------------------------
        # 8) Apply gradients
        # --------------------------------------------------------------
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        del tape
    
        self.optimizer.apply_gradients(
            self._scale_param_grads(grads, trainable_vars)
        )
    
        # --------------------------------------------------------------
        # 9) Update metrics and return logs
        # --------------------------------------------------------------
        self.compiled_metrics.update_state(targets, y_pred)
    
        results = {
            m.name: m.result()
            for m in self.metrics
            if m.name not in ("epsilon_prior", "epsilon_cons")
        }
    
        results.update(
            {
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
            }
        )
    
        results["loss"] = total_loss
        return results


    def test_step(self, data):
        
        # --- 0. Unpack and normalize targets 
        inputs, targets = data
    
        # For convenience, accept either {"subsidence", "gwl"} or
        # {"subs_pred", "gwl_pred"} as keys and normalize to the
        # supervised mapping expected by compiled_loss.
        if isinstance(targets, dict):
            targets = rename_dict_keys(
                targets.copy(),
                param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"},
            )
    
        # --- 1. Forward pass (no gradients) 
        # This returns the same output structure as during training:
        # {
        #   "data_final": ... (quantile-expanded),
        #   "data_mean":  ...,
        #   "phys_mean_raw": ...
        # }
        outputs = self(inputs, training=False)
    
        # --- 2. Split data predictions for supervised heads 
        # data_final typically has shape (B, H, Q, 2) or similar; we
        # only care about splitting along the last dimension into
        # subsidence and head.
        s_pred_final, gwl_pred_final = self.split_data_predictions(
            outputs["data_final"]
        )
        y_pred_for_eval = {
            "subs_pred": s_pred_final,  # shape ~ (B, H, Q[, ...])
            "gwl_pred": gwl_pred_final,
        }
    
        # --- 3. Data loss + metrics (same mapping as train_step) 
        data_loss  = self.compiled_loss(
            y_true=targets,
            y_pred=y_pred_for_eval,
            regularization_losses=self.losses,
        )
        self.compiled_metrics.update_state(targets, y_pred_for_eval)
    
        # --- 4. Optional physics diagnostics (no GradientTape here) ------
        # This section is deliberately lightweight and purely diagnostic:
        # it probes the current state of the physics without influencing
        # gradients.
        if not self._physics_off():
            # evaluate_physics() returns scalar epsilons and optionally
            # residual maps. Here we only need the scalars.
            # phys = self.evaluate_physics(inputs)  # returns tensors
            # self.eps_prior_metric.update_state(phys["epsilon_prior"])
            # self.eps_cons_metric.update_state(phys["epsilon_cons"])
            
            phys_pack = self.evaluate_physics(inputs, return_maps=False)
            total_loss = data_loss + phys_pack["physics_loss_scaled"]
        
        else:
            # Physics is disabled: push zeros to keep metrics well-defined
            # and prevent NaNs in logs.
            # self.eps_prior_metric.update_state(0.0)
            # self.eps_cons_metric.update_state(0.0)
            phys_pack = {
                "epsilon_prior": tf_constant(0.0, tf_float32),
                "epsilon_cons": tf_constant(0.0, tf_float32),
                "physics_loss_raw": tf_constant(0.0, tf_float32),
                "physics_mult": tf_constant(1.0, tf_float32),
                "physics_loss_scaled": tf_constant(0.0, tf_float32),
                "consolidation_loss": tf_constant(0.0, tf_float32),
                "gw_flow_loss": tf_constant(0.0, tf_float32),
                "prior_loss": tf_constant(0.0, tf_float32),
                "smooth_loss": tf_constant(0.0, tf_float32),
                "mv_prior_loss": tf_constant(0.0, tf_float32),
                "bounds_loss": tf_constant(0.0, tf_float32),
            }
            total_loss = data_loss

        # keep eps metrics well-defined
        self.eps_prior_metric.update_state(phys_pack["epsilon_prior"])
        self.eps_cons_metric.update_state(phys_pack["epsilon_cons"])
        
        # --- 5. Collect and return metrics 
        results = {
                m.name: m.result()
                for m in self.metrics
                if m.name not in ("epsilon_prior", "epsilon_cons")
            }
    
        results.update({
            # THIS drives val_loss
            "loss": total_loss,
        
            # extras for plotting
            "total_loss": total_loss,
            "data_loss": data_loss,
            "physics_loss": phys_pack["physics_loss_raw"],
            "physics_mult": phys_pack["physics_mult"],
            "physics_loss_scaled": phys_pack["physics_loss_scaled"],
            "lambda_offset": self._lambda_offset,
        
            "consolidation_loss": phys_pack["consolidation_loss"],
            "gw_flow_loss": phys_pack["gw_flow_loss"],
            "prior_loss": phys_pack["prior_loss"],
            "smooth_loss": phys_pack["smooth_loss"],
            "mv_prior_loss": phys_pack["mv_prior_loss"],
            "bounds_loss": phys_pack["bounds_loss"],
        
            "epsilon_prior": self.eps_prior_metric.result(),
            "epsilon_cons": self.eps_cons_metric.result(),
        })
        return results

    def _evaluate_physics_on_batch(
        self,
        inputs: Dict[str, Optional[Tensor]],
        return_maps: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Physics diagnostics on a single batch.
    
        IMPORTANT (same fix as train_step):
        -----------------------------------
        Differentiate w.r.t. the *coords tensor itself* (inputs["coords"]),
        not separately watched (t, x, y) tensors. This guarantees the AD
        graph is connected to what the forward pass actually used and avoids
        the `0.0 * t` binding trick (which forces gradients to exactly zero).
        """
        # --------------------------------------------------------------
        # 0) Validate required thickness input
        # --------------------------------------------------------------
        H_field_in = get_tensor_from(
            inputs,
            "H_field",
            "soil_thickness",
            auto_convert=True,
        )
        if H_field_in is None:
            raise ValueError(
                "evaluate_physics() requires 'H_field' "
                "(or 'soil_thickness') in inputs."
            )
    
        H_field = tf_convert_to_tensor(
            H_field_in,
            dtype=tf_float32,
        )
        H_si = self._to_si_thickness(H_field)
    
        # --------------------------------------------------------------
        # 1) Extract coords and prepare watched tensor
        # --------------------------------------------------------------
        coords = tf_convert_to_tensor(
            _get_coords(inputs),
            dtype=tf_float32,
        )
    
        # Support (B, 3) coords by expanding to (B, 1, 3)
        if coords.shape.rank == 2:
            coords = tf_expand_dims(coords, axis=1)
    
        # Convenience slices (NOT watched separately)
        t = coords[..., 0:1]
        # x = coords[..., 1:2]
        # y = coords[..., 2:3]
    
        # --------------------------------------------------------------
        # 2) Forward + AD derivatives
        # --------------------------------------------------------------
        with GradientTape(persistent=True) as tape:
            tape.watch(coords)
    
            # Forward pass (inference mode)
            outputs = self(inputs, training=False)
    
            (
                s_mean,
                h_mean,
                K_base,
                Ss_base,
                tau_base,
            ) = self.split_physics_predictions(outputs)
    
            # NOTE:
            # - data_mean already includes coord corrections
            #   because call() applies coord_mlp/subs_coord_mlp.
            # - Do NOT add coord corrections again here.
    
            v_m = self._to_si_head(h_mean)
            h_si = self._gwl_to_head_m(v_m, inputs=inputs)
            s_si = self._to_si_subsidence(s_mean)
    
            h_pde = h_si
            s_pde = s_si
    
            (
                K_field,
                Ss_field,
                tau_field,
                tau_phys,
                Hd_eff,
                delta_log_tau,
                logK,
                logSs,
            ) = self._compose_physics_fields(
                coords_flat=coords,
                H_si=H_si,
                K_base=K_base,
                Ss_base=Ss_base,
                tau_base=tau_base,
                training=False,
            )
    
            # Keep latest fields consistent with training path
            self.K_field = K_field
            self.Ss_field = Ss_field
            self.tau_field = tau_field
    
            # --------------------------
            # 2.1 First derivatives
            # --------------------------
            ds_dcoords = tape.gradient(s_pde, coords)
            dh_dcoords = tape.gradient(h_pde, coords)
    
            if (ds_dcoords is None) or (dh_dcoords is None):
                raise ValueError(
                    "evaluate_physics(): PDE gradients are None. "
                    "This indicates the forward graph is not "
                    "connected to inputs['coords']."
                )
    
            ds_dt = ds_dcoords[..., 0:1]
            dh_dt = dh_dcoords[..., 0:1]
            dh_dx = dh_dcoords[..., 1:2]
            dh_dy = dh_dcoords[..., 2:3]
    
            # --------------------------
            # 2.2 Second derivatives
            # --------------------------
            K_dh_dx = K_field * dh_dx
            K_dh_dy = K_field * dh_dy
    
            dKdhx_dcoords = tape.gradient(K_dh_dx, coords)
            dKdhy_dcoords = tape.gradient(K_dh_dy, coords)
    
            if (dKdhx_dcoords is None) or (dKdhy_dcoords is None):
                raise ValueError(
                    "evaluate_physics(): second-order grads are None. "
                    "Check that K_field and h_pde depend on coords."
                )
    
            d_K_dh_dx_dx = dKdhx_dcoords[..., 1:2]
            d_K_dh_dy_dy = dKdhy_dcoords[..., 2:3]
    
            # --------------------------
            # 2.3 Smoothness grads
            # --------------------------
            dK_dcoords = tape.gradient(K_field, coords)
            dSs_dcoords = tape.gradient(Ss_field, coords)
    
            if (dK_dcoords is None) or (dSs_dcoords is None):
                raise ValueError(
                    "evaluate_physics(): K/Ss grads are None. "
                    "Check K_field/Ss_field dependence on coords."
                )
    
            dK_dx = dK_dcoords[..., 1:2]
            dK_dy = dK_dcoords[..., 2:3]
            dSs_dx = dSs_dcoords[..., 1:2]
            dSs_dy = dSs_dcoords[..., 2:3]
    
            # ----------------------------------------------------------
            # 2.4 Chain-rule correction for normalized/degree coords
            # ----------------------------------------------------------
            coords_norm = bool(
                self.scaling_kwargs.get("coords_normalized", False)
            )
            coords_deg = bool(
                self.scaling_kwargs.get("coords_in_degrees", False)
            )
    
            tR, xR, yR = self._coord_ranges()
            deg2m_x = self._deg_to_m("x")
            deg2m_y = self._deg_to_m("y")
            
            # --- time ---
            if coords_norm and tR:
                ds_dt = ds_dt / tR
                dh_dt = dh_dt / tR
            
            ds_dt = rate_to_per_second(ds_dt, self.time_units)
            dh_dt = rate_to_per_second(dh_dt, self.time_units)

            # --- space ---
            if coords_norm:
                if xR:
                    xRt = xR * deg2m_x
                    dh_dx = dh_dx / xRt
                    d_K_dh_dx_dx = d_K_dh_dx_dx / (xRt * xRt)
                    dK_dx = dK_dx / xRt
                    dSs_dx = dSs_dx / xRt
                if yR:
                    yRt = yR * deg2m_y
                    dh_dy = dh_dy / yRt
                    d_K_dh_dy_dy = d_K_dh_dy_dy / (yRt * yRt)
                    dK_dy = dK_dy / yRt
                    dSs_dy = dSs_dy / yRt
    
            elif coords_deg:
                dh_dx = dh_dx / deg2m_x
                d_K_dh_dx_dx = d_K_dh_dx_dx / (deg2m_x * deg2m_x)
                dK_dx = dK_dx / deg2m_x
                dSs_dx = dSs_dx / deg2m_x
    
                dh_dy = dh_dy / deg2m_y
                d_K_dh_dy_dy = d_K_dh_dy_dy / (deg2m_y * deg2m_y)
                dK_dy = dK_dy / deg2m_y
                dSs_dy = dSs_dy / deg2m_y
    
        del tape
    
        # --------------------------------------------------------------
        # 3) Residuals (unscaled first)
        # --------------------------------------------------------------
        gw_res = self._compute_gw_flow_residual(
            dh_dt,
            d_K_dh_dx_dx,
            d_K_dh_dy_dy,
            Ss_field,
            Q=0.0,
        )
    
        cons_res = self._compute_consolidation_residual(
            ds_dt,
            s_pde,
            h_pde,
            H_si,
            tau_field,
            Ss_field=Ss_field,
            inputs=inputs,
        )
    
        smooth_res = self._compute_smoothness_prior(
            dK_dx,
            dK_dy,
            dSs_dx,
            dSs_dy,
            K_field=K_field,
            Ss_field=Ss_field,
        )
    
        mv_prior_res = self._compute_mv_prior(Ss_field)
        loss_mv = tf_reduce_mean(tf_square(mv_prior_res))
    
        R_H, R_K, R_Ss = self._compute_bounds_residual(
            K_field,
            Ss_field,
            H_si,
        )
    
        prior_res = delta_log_tau
    
        # Paper epsilons: RMS of *unscaled* residuals
        eps_prior = tf_sqrt(tf_reduce_mean(tf_square(prior_res)))
        eps_cons = tf_sqrt(tf_reduce_mean(tf_square(cons_res)))
        eps_gw    = tf_sqrt(tf_reduce_mean(tf_square(gw_res)))
    
        # --------------------------------------------------------------
        # 4) Physics-off shortcut
        # --------------------------------------------------------------
        if self._physics_off():
            z = tf_constant(0.0, tf_float32)
            return {
                "epsilon_prior": z,
                "epsilon_cons": z,
                "epsilon_gw": z,
                "consolidation_loss": z,
                "gw_flow_loss": z,
                "prior_loss": z,
                "smooth_loss": z,
                "mv_prior_loss": z,
                "bounds_loss": z,
                "physics_loss_raw": z,
                "physics_mult": tf_constant(1.0, tf_float32),
                "physics_loss_scaled": z,
            }
    
        # --------------------------------------------------------------
        # 5) Scaling (only cons/gw, same as train_step)
        # --------------------------------------------------------------
        cons_for_loss = cons_res
        gw_for_loss = gw_res
        
        # (B,1,1) -> broadcasts to (B,H,1)
        h_ref_si = self._get_h_ref_si(inputs, like=h_pde)  

        if self.scale_pde_residuals:
            scales = self._compute_scales(
                t=t,
                s_mean=s_pde,          # (B,H,1) in m
                h_mean=h_pde,          # (B,H,1) in m
                K_field=K_field,
                Ss_field=Ss_field,
                ds_dt=ds_dt,           # (B,H,1) in m/s  
                tau_field=tau_field,   # (B,H,1) in s
                H_field=H_si,          # (B,H,1) in m
                h_ref_si=h_ref_si,   
                Q=0.0,
            )
            
            cons_for_loss = scale_residual(
                cons_for_loss,
                scales.get("cons_scale"),
            )
            gw_for_loss = scale_residual(
                gw_for_loss,
                scales.get("gw_scale"),
            )
    
        # --------------------------------------------------------------
        # 6) Component losses (match train_step)
        # --------------------------------------------------------------
        loss_cons = tf_reduce_mean(tf_square(cons_for_loss))
        loss_gw = tf_reduce_mean(tf_square(gw_for_loss))
        loss_prior = tf_reduce_mean(tf_square(prior_res))
        loss_smooth = tf_reduce_mean(smooth_res)
    
        bounds_res = tf_concat([R_H, R_K, R_Ss], axis=-1)
        loss_bounds = tf_reduce_mean(tf_square(bounds_res))
    
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
    
        out: Dict[str, Tensor] = {
            "epsilon_prior": eps_prior,
            "epsilon_cons": eps_cons,
            "epsilon_gw": eps_gw,
            "consolidation_loss": loss_cons,
            "gw_flow_loss": loss_gw,
            "prior_loss": loss_prior,
            "smooth_loss": loss_smooth,
            "mv_prior_loss": loss_mv,
            "bounds_loss": loss_bounds,
            "physics_loss_raw": physics_loss_raw,
            "physics_mult": phys_mult,
            "physics_loss_scaled": physics_loss_scaled,
        }
    
        # --------------------------------------------------------------
        # 7) Optional maps for diagnostics export
        # --------------------------------------------------------------
        if return_maps:
            tau_prior, Hd_eff = self._tau_phys_from_fields(
                K_field,
                Ss_field,
                H_si,
            )
            out.update(
                {
                    "R_prior": prior_res,
                    "R_cons": cons_res,
                    "R_gw": gw_res,
                    "K": K_field,
                    "Ss": Ss_field,
                    "H_field": H_field,
                    "Hd": Hd_eff,
                    # Keep legacy key for older loaders
                    "H": Hd_eff,
                    "tau": tau_field,
                    "tau_prior": tau_prior,
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


        # # 1) Dataset path: iterate up to `max_batches` and aggregate scalars
        MAP_KEYS = (
            "R_prior","R_cons","K","Ss","H_field",
            "Hd","H","tau","tau_prior", "R_gw", 
        )
        
        if isinstance(inputs, Dataset):
            acc = {}          
            last_maps = None
        
            for i, elem in enumerate(inputs):
                xb = elem[0] if isinstance(elem, (tuple, list)) else elem
                out_b = self._evaluate_physics_on_batch(
                    xb, return_maps=return_maps
                )
        
                for k, v in out_b.items():
                    if (not return_maps) or (k not in MAP_KEYS):
                        acc.setdefault(k, []).append(v)
        
                if return_maps:
                    last_maps = {k: out_b[k] for k in MAP_KEYS if k in out_b}
        
                if max_batches is not None and (i + 1) >= max_batches:
                    break
        
            out = {k: tf_reduce_mean(tf_stack(vs)) for k, vs in acc.items()}
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

    def _compute_mv_prior(self, Ss_field):
        r"""
        Compute the storage identity prior
    
        .. math::
    
            R_{m_v}
            = \log S_s - \log (m_v \gamma_w),
    
        which encourages the learned specific storage :math:`S_s`
        to remain consistent with the scalar compressibility
        :math:`m_v` and the unit-weight of water :math:`\gamma_w`.
    
        **NaN safety**
    
        - ``S_s`` is clamped from below by a small :math:`\varepsilon`
          to prevent ``\log(0)`` and the resulting ``-inf`` / ``NaN``
          in the loss.
    
        Parameters
        ----------
        Ss_field : tf.Tensor
            Predicted specific storage field :math:`S_s(x,y)`,
            shape ``(B, H, 1)`` (already passed through a positivity
            function earlier in the pipeline).
    
        Returns
        -------
        tf.Tensor
            Residual :math:`R_{m_v}` with the same shape as
            ``Ss_field``.
        """
        eps = tf_constant(1e-6, dtype=tf_float32)
        Ss_safe = tf_maximum(Ss_field, eps)
    
        return tf_log(Ss_safe) - (
            tf_log(self._mv_value()) + tf_log(self.gamma_w)
        )


    def _compute_gw_flow_residual(
        self,
        dh_dt,
        d_K_dh_dx_dx,
        d_K_dh_dy_dy,
        Ss_field,
        Q: float = 0.0,
    ):

        if "gw_flow" not in self.pde_modes_active:
            return tf_zeros_like(dh_dt)
        
        Q = tf_convert_to_tensor(Q, dtype=tf_float32)
        Q = tf_broadcast_to(Q, tf_shape(dh_dt))

        div_K_grad_h = d_K_dh_dx_dx + d_K_dh_dy_dy
        storage_term = Ss_field * dh_dt

        return storage_term - div_K_grad_h - Q

    def _compute_consolidation_residual(
        self,
        ds_dt,
        s_mean,
        h_mean,
        H_field,
        tau_field,
        Ss_field=None,
        inputs=None,          # <--- add this
    ):
        if "consolidation" not in self.pde_modes_active:
            return tf_zeros_like(ds_dt)
    
        eps = tf_constant(1e-12, dtype=tf_float32)
        tau_safe = tf_maximum(tau_field, eps)
    
        # Δh = max(0, h_ref - h)
        h_ref_si = self._get_h_ref_si(inputs, like=h_mean)
        delta_h = tf_maximum(h_ref_si - h_mean, 0.0)
    
        # Ss effective
        if Ss_field is None:
            Ss_eff = self._mv_value() * self.gamma_w
        else:
            Ss_eff = Ss_field
    
        # s_eq ≈ Ss * Δh * H
        s_eq = Ss_eff * delta_h * H_field
    
        relaxation_term = (s_eq - s_mean) / tau_safe
        return ds_dt - relaxation_term

    def _compute_consistency_prior(self, K_field, Ss_field, tau_field, H_field):
        eps = tf_constant(1e-12, dtype=tf_float32)
    
        tau_safe = tf_maximum(tau_field, eps)
        tau_phys, _Hd = self._tau_phys_from_fields(K_field, Ss_field, H_field)
        tau_phys_safe = tf_maximum(tau_phys, eps)
    
        return tf_log(tau_safe) - tf_log(tau_phys_safe)

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
        # Use a tiny epsilon: avoid div-by-zero without clamping physics away
        epsK = tf_constant(1e-12, dtype=tf_float32)
        eps  = tf_constant(1e-12, dtype=tf_float32)
        pi_sq = tf_constant(np.pi**2, dtype=tf_float32)
    
        K_safe  = tf_maximum(K_field,  epsK)
        Ss_safe = tf_maximum(Ss_field, eps)
    
        H_safe = tf_maximum(H_field, eps)
    
        # Drainage path: Hd = eta * H (eta = Hd_factor), if enabled
        Hd = (H_safe * self.Hd_factor) if self.use_effective_thickness else H_safe
        Hd = tf_maximum(Hd, eps)
    
        ratio = Hd / H_safe  # Hd/H
    
        if self.kappa_mode == "bar":
            # kappa_bar := (Hd/H)^2 / kappa_b  (>= 0)
            # tau = kappa_bar * H^2 * Ss / (pi^2 * K)
            tau_phys = (
                self._kappa_value()
                * (H_safe ** 2)
                * Ss_safe
                / (pi_sq * K_safe)
            )
        else:  # "kb"
            # tau = (Hd/H)^2 * H^2 * Ss / (pi^2 * kappa_b * K)
            #     = Hd^2 * Ss / (pi^2 * kappa_b * K)
            tau_phys = (
                (ratio ** 2)
                * (H_safe ** 2)
                * Ss_safe
                / (pi_sq * self._kappa_value() * K_safe)
            )
    
        return tau_phys, Hd

    def _compute_smoothness_prior(
        self,
        dK_dx, dK_dy,
        dSs_dx, dSs_dy,
        K_field: Optional[Tensor] = None,
        Ss_field: Optional[Tensor] = None,
    ):
        eps = tf_constant(1e-12, dtype=tf_float32)
    
        # If fields are provided, smoothness is on log-fields (unit-consistent).
        if (K_field is not None) and (Ss_field is not None):
            dlogK_dx  = dK_dx  / (K_field  + eps)
            dlogK_dy  = dK_dy  / (K_field  + eps)
            dlogSs_dx = dSs_dx / (Ss_field + eps)
            dlogSs_dy = dSs_dy / (Ss_field + eps)
            return (
                tf_square(dlogK_dx) + tf_square(dlogK_dy)
                + tf_square(dlogSs_dx) + tf_square(dlogSs_dy)
            )
    
        # Fallback to old behavior (kept for backwards compatibility)
        grad_K_squared = tf_square(dK_dx) + tf_square(dK_dy)
        grad_Ss_squared = tf_square(dSs_dx) + tf_square(dSs_dy)
        return grad_K_squared + grad_Ss_squared

    def _get_log_bounds(self, as_tensor: bool = True, dtype=tf_float32):
        b = (getattr(self, "scaling_kwargs", {}) or {}).get("bounds", {}) or {}
    
        # accept either log* or linear bounds
        def _get_log_pair(
                log_min_key, log_max_key, 
                lin_min_key, lin_max_key
            ):
            log_min = b.get(log_min_key, None)
            log_max = b.get(log_max_key, None)
            if log_min is None or log_max is None:
                if (lin_min_key in b) and (lin_max_key in b):
                    vmin = float(b[lin_min_key])
                    vmax = float(b[lin_max_key])

                    # ------------------------------------------------------------------
                    # Backwards-compatibility heuristic:
                    # Some configs historically (incorrectly) provided *m_v* (Pa^-1)
                    # bounds under the name Ss_min/Ss_max. Since Ss ~= m_v * gamma_w
                    # (1/m), convert those values when they are clearly in the Pa^-1
                    # range (very small), and gamma_w is O(1e4).
                    # ------------------------------------------------------------------
                    if (lin_min_key == "Ss_min") and (lin_max_key == "Ss_max"):
                        try:
                            gw = getattr(self, "gamma_w", None)
                            gw = float(gw.numpy()) if hasattr(gw, "numpy") else float(gw)
                            mv0 = getattr(getattr(self, "mv_config", None), "initial_value", None)
                            mv0 = float(mv0) if mv0 is not None else None
                            ss_expected = (mv0 * gw) if mv0 is not None else None

                            looks_like_mv = (vmax <= 1e-5) and (gw > 1e3)
                            if looks_like_mv and (ss_expected is None or ss_expected > 1e-5):
                                vmin *= gw
                                vmax *= gw
                                fusionlog.warning(
                                    "Interpreting Ss_min/Ss_max as m_v bounds "
                                    "and converting via Ss = m_v * gamma_w."
                                )
                        except Exception:
                            pass

                    log_min = float(np.log(vmin))
                    log_max = float(np.log(vmax))
                else:
                    # no bounds provided
                    return None, None
            return float(log_min), float(log_max)
    
        logK_min, logK_max   = _get_log_pair(
            "logK_min",  "logK_max",  "K_min",  "K_max"
            )
        logSs_min, logSs_max = _get_log_pair(
            "logSs_min", "logSs_max", "Ss_min", "Ss_max"
        )
    
        if (logK_min is None) or (logSs_min is None):
            # caller can decide fallback/raise
            return (None, None, None, None)
    
        if not as_tensor:
            return logK_min, logK_max, logSs_min, logSs_max
    
        return (
            tf_constant(logK_min, dtype),
            tf_constant(logK_max, dtype),
            tf_constant(logSs_min, dtype),
            tf_constant(logSs_max, dtype),
        )

    def _bounded_exp(self, raw, log_min, log_max, eps=1e-12, return_log=False):
        z = tf_sigmoid(raw)  # (0,1) so we can enforce [min,max]
        logv = log_min + z * (log_max - log_min)
        out = tf_exp(logv) + tf_constant(eps, tf_float32)
        return (out, logv) if return_log else out
    

    def _compose_physics_fields(
        self,
        coords_flat,
        H_si,
        K_base,
        Ss_base,
        tau_base,
        training: bool = False,
        eps_KSs: float = 1e-12,
        eps_tau: float = 1e-6,
    ):
        # K_corr   = self.K_coord_mlp(coords_flat, training=training)
        # Ss_corr  = self.Ss_coord_mlp(coords_flat, training=training)
        # tau_corr = self.tau_coord_mlp(coords_flat, training=training)
        coords_xy0 = tf_concat(
            [tf_zeros_like(coords_flat[..., :1]), coords_flat[..., 1:]],
            axis=-1
        )
        
        K_corr   = self.K_coord_mlp(coords_xy0, training=training)
        Ss_corr  = self.Ss_coord_mlp(coords_xy0, training=training)
        tau_corr = self.tau_coord_mlp(coords_xy0, training=training)  # if you want tau(x,y) too

        rawK  = K_base  + K_corr
        rawSs = Ss_base + Ss_corr
    
        if getattr(self, "bounds_mode", "soft") == "hard":
            logK_min, logK_max, logSs_min, logSs_max = self._get_log_bounds(
                as_tensor=True, dtype=rawK.dtype)
            K_field,  logK  = self._bounded_exp(
                rawK,  logK_min,  logK_max,
                eps=eps_KSs, return_log=True
            )
            Ss_field, logSs = self._bounded_exp(
                rawSs, logSs_min, logSs_max,
               eps=eps_KSs, return_log=True
             )
        else:
            # PAPER-CONSISTENT: treat rawK/rawSs as log-fields,
            # softly bounded via L_bounds (but still center-init in-range)
            logK  = rawK
            logSs = rawSs
        
            logK_min, logK_max, logSs_min, logSs_max = self._get_log_bounds(
                as_tensor=True, dtype=rawK.dtype
            )

            K_field  = tf_exp(logK)  + tf_constant(eps_KSs, logK.dtype)
            Ss_field = tf_exp(logSs) +  tf_constant(eps_KSs, logSs.dtype)
    
        # tau = tau_phys(K,Ss,H) * exp(delta_log_tau)
        delta_log_tau = tau_base + tau_corr
        
        tau_phys, Hd_eff = self._tau_phys_from_fields(
            K_field, Ss_field, H_si
        )  # in `time_units` implied by K
        tau_field = tau_phys * tf_exp(
            delta_log_tau) + tf_constant(eps_tau, tau_phys.dtype)
    
        return ( 
            K_field, Ss_field, tau_field, 
            tau_phys, Hd_eff, delta_log_tau, 
            logK, logSs
        )

    def _compute_bounds_residual(
        self,
        K_field: Tensor,
        Ss_field: Tensor,
        H_field: Tensor,
    ) -> Tensor:
    
        dtype = K_field.dtype
        eps  = tf_constant(1e-12, dtype=dtype)   
        zero = tf_constant(0.0,  dtype=dtype)
    
        K_safe  = tf_maximum(K_field, eps)
        Ss_safe = tf_maximum(Ss_field, eps)
        H_safe  = tf_maximum(H_field, eps)
    
        bounds_cfg = (self.scaling_kwargs or {}).get("bounds", {}) or {}
    
        # --- H bounds (dimensionless) -------------------------------------
        H_min = bounds_cfg.get("H_min", None)
        H_max = bounds_cfg.get("H_max", None)
        if H_min is None or H_max is None:
            R_H = tf_zeros_like(H_safe)
        else:
            H_min_t = tf_constant(float(H_min), dtype=dtype)
            H_max_t = tf_constant(float(H_max), dtype=dtype)
            lower_H = tf_maximum(H_min_t - H_safe, zero)
            upper_H = tf_maximum(H_safe - H_max_t, zero)
            H_range = tf_maximum(H_max_t - H_min_t, eps)
            R_H = (lower_H + upper_H) / H_range
    
        # Helper: log-bound residual normalized by log-range
        def _log_bound_residual(val_safe, log_min, log_max):
            logv = tf_log(val_safe)
            log_min_t = tf_constant(float(log_min), dtype=dtype)
            log_max_t = tf_constant(float(log_max), dtype=dtype)
            lower = tf_maximum(log_min_t - logv, zero)
            upper = tf_maximum(logv - log_max_t, zero)
            log_range = tf_maximum(log_max_t - log_min_t, eps)
            return (lower + upper) / log_range  # dimensionless
    
        # --- log K bounds --------------------------------------------------
        logK_min = bounds_cfg.get("logK_min", None)
        logK_max = bounds_cfg.get("logK_max", None)
        if (logK_min is None or logK_max is None) and (
                bounds_cfg.get("K_min") is not None and bounds_cfg.get(
                    "K_max") is not None):
            logK_min = float(np.log(float(bounds_cfg["K_min"])))
            logK_max = float(np.log(float(bounds_cfg["K_max"])))
    
        R_K = tf_zeros_like(K_safe) if (
            logK_min is None or logK_max is None) else _log_bound_residual(
                K_safe, logK_min, logK_max)
    
        # --- log Ss bounds -------------------------------------------------
        logSs_min = bounds_cfg.get("logSs_min", None)
        logSs_max = bounds_cfg.get("logSs_max", None)
        if (logSs_min is None or logSs_max is None) and (
                bounds_cfg.get("Ss_min") is not None and bounds_cfg.get(
                    "Ss_max") is not None):
            logSs_min = float(np.log(float(bounds_cfg["Ss_min"])))
            logSs_max = float(np.log(float(bounds_cfg["Ss_max"])))
    
        R_Ss = tf_zeros_like(Ss_safe) if (
            logSs_min is None or logSs_max is None
            ) else _log_bound_residual(Ss_safe, logSs_min, logSs_max)
    
        return R_H , R_K , R_Ss

    def _compute_scales(
        self,
        t,
        s_mean,      # SI (m)
        h_mean,      # SI (m)
        K_field,
        Ss_field,
        ds_dt=None,        # SI (m/s)
        tau_field=None,    # SI (s)
        H_field=None,      # SI (m)
        h_ref_si=None,     # SI (m), per-batch/per-pixel ref head (preferred)
        Q: float = 0.0,
    ):
        # --- dt_tensor in time_units (e.g., years) ---
        dt_tensor = None
        if hasattr(t, "shape") and t.shape.rank is not None and t.shape.rank >= 2:
            if (t.shape[1] is not None) and (t.shape[1] > 1):
                dt_tensor = t[:, 1:, :] - t[:, :-1, :]
    
        if dt_tensor is None:
            if (s_mean.shape.rank is not None) and (
                s_mean.shape[1] is not None) and (s_mean.shape[1] > 1):
                dt_tensor = tf_zeros_like(s_mean[:, 1:, :]) + 1.0
            else:
                dt_tensor = tf_zeros_like(s_mean[..., :1]) + 1.0
    
        coords_norm = bool(self.scaling_kwargs.get("coords_normalized", False))
        tR, _, _ = self._coord_ranges()
        if coords_norm and tR:
            dt_tensor = dt_tensor * tR  # back to time_units (e.g., years)
    
        time_units = (
            self.scaling_kwargs.get(
                "time_units", self.scaling_kwargs.get(
                    "time_unit", None))
            or self.time_units
        )
    
        scales = default_scales(
            h=h_mean,
            s=s_mean,
            dt=dt_tensor,
            K=K_field,
            Ss=Ss_field,
            Q=Q,
            time_units=time_units,
        )
    
        # --- τ-aware cons scale (recommended) ---
        if (ds_dt is not None) and (tau_field is not None) and (H_field is not None):
            eps = tf_constant(1e-12, tf_float32)
    
            # Prefer per-batch ref head; fallback to scalar self.h_ref (SI meters)
            href = h_ref_si
            if href is None:
                href = tf_cast(self.h_ref, h_mean.dtype)
    
            # IMPORTANT: match consolidation residual convention (drawdown only)
            delta_h = tf_maximum(href - h_mean, 0.0)  # (B,H,1)
    
            # s_eq ≈ Ss * Δh * H
            s_eq = Ss_field * delta_h * H_field  # (m)
    
            relax = (s_eq - s_mean) / (tau_field + eps)  # (m/s)
    
            # stop gradients: scale should not become a learnable escape hatch
            term1 = tf_stop_gradient(tf_reduce_mean(tf_abs(ds_dt)) + eps)
            term2 = tf_stop_gradient(tf_reduce_mean(tf_abs(relax)) + eps)
            scales["cons_scale"] = term1 + term2
    
        return scales

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
        outputs_dict: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:



        data_means_tensor = outputs_dict["data_mean"]
        phys_means_raw_tensor = outputs_dict["phys_mean_raw"]
    
        # --- Split mean data predictions (s, h)
        s_pred_mean = data_means_tensor[..., : self.output_subsidence_dim]
        gwl_pred_mean = data_means_tensor[..., self.output_subsidence_dim :]
    
        # --- Slice physics logits: (K, Ss, delta_log_tau)
        start_idx = 0
        end_idx = self.output_K_dim
        K_logits = phys_means_raw_tensor[..., start_idx:end_idx]
    
        start_idx = end_idx
        end_idx += self.output_Ss_dim
        Ss_logits = phys_means_raw_tensor[..., start_idx:end_idx]
    
        start_idx = end_idx
        delta_log_tau_logits = phys_means_raw_tensor[..., start_idx:]
    
        # NOTE: no positivity transform here by design.
        return s_pred_mean, gwl_pred_mean, K_logits, Ss_logits, delta_log_tau_logits

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
            "model_version": "3.1-GeoPrior",
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

