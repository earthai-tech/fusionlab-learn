# -*- coding: utf-8 -*-
# _geoprior_utils.py
"""
GeoPrior small utilities (no derivatives here).
Short docs only; full docs later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .. import KERAS_DEPS


Tensor = KERAS_DEPS.Tensor

tf_float32 = KERAS_DEPS.float32
tf_int32 = KERAS_DEPS.int32

tf_cast = KERAS_DEPS.cast
tf_constant = KERAS_DEPS.constant
tf_debugging = KERAS_DEPS.debugging
tf_equal = KERAS_DEPS.equal
tf_maximum = KERAS_DEPS.maximum
tf_rank = KERAS_DEPS.rank
tf_cond = KERAS_DEPS.cond
tf_shape = KERAS_DEPS.shape
tf_zeros_like = KERAS_DEPS.zeros_like
tf_ones = KERAS_DEPS.ones 
tf_greater = KERAS_DEPS.greater 
tf_cond = KERAS_DEPS.cond 
tf_concat = KERAS_DEPS.concat 
tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor 
tf_ones_like = KERAS_DEPS.ones_like 
tf_less_equal =KERAS_DEPS.less_equal 


def affine_from_cfg(
    scaling_kwargs: Optional[Dict[str, Any]],
    *,
    scale_key: str,
    bias_key: str,
    meta_keys: Tuple[str, ...] = (),
    unit_key: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """Return (a,b) for y_si = y_model*a + b."""
    cfg = scaling_kwargs or {}

    a = cfg.get(scale_key, None)
    b = cfg.get(bias_key, None)

    if a is not None or b is not None:
        a = 1.0 if a is None else float(a)
        b = 0.0 if b is None else float(b)
        return tf_constant(a, tf_float32), tf_constant(b, tf_float32)

    for mk in meta_keys:
        meta = cfg.get(mk, None)
        if isinstance(meta, dict):
            mu = meta.get("mu", meta.get("mean", None))
            sig = meta.get("sigma", meta.get("std", None))
            if mu is not None and sig is not None:
                return (
                    tf_constant(float(sig), tf_float32),
                    tf_constant(float(mu), tf_float32),
                )

    if unit_key is not None:
        u = float(cfg.get(unit_key, 1.0))
        return tf_constant(u, tf_float32), tf_constant(0.0, tf_float32)

    return tf_constant(1.0, tf_float32), tf_constant(0.0, tf_float32)


def to_si_thickness(
    H_model: Tensor,
    scaling_kwargs: Optional[Dict[str, Any]],
) -> Tensor:
    """Convert thickness to SI."""
    a, b = affine_from_cfg(
        scaling_kwargs,
        scale_key="H_scale_si",
        bias_key="H_bias_si",
        meta_keys=("H_z_meta",),
        unit_key="thickness_unit_to_si",
    )
    return tf_cast(H_model, tf_float32) * a + b


def to_si_head(
    h_model: Tensor,
    scaling_kwargs: Optional[Dict[str, Any]],
) -> Tensor:
    """Convert head/depth to SI meters."""
    a, b = affine_from_cfg(
        scaling_kwargs,
        scale_key="head_scale_si",
        bias_key="head_bias_si",
        meta_keys=("head_z_meta", "gwl_z_meta"),
        unit_key="head_unit_to_si",
    )
    return tf_cast(h_model, tf_float32) * a + b


def to_si_subsidence(
    s_model: Tensor,
    scaling_kwargs: Optional[Dict[str, Any]],
) -> Tensor:
    """Convert subsidence to SI meters."""
    a, b = affine_from_cfg(
        scaling_kwargs,
        scale_key="subs_scale_si",
        bias_key="subs_bias_si",
        meta_keys=("subs_z_meta",),
        unit_key="subs_unit_to_si",
    )
    return tf_cast(s_model, tf_float32) * a + b

def from_si_subsidence(
    s_si: Tensor,
    scaling_kwargs: Optional[Dict[str, Any]],
) -> Tensor:
    """Inverse of to_si_subsidence: s_model = (s_si - b) / a."""
    a, b = affine_from_cfg(
        scaling_kwargs,
        scale_key="subs_scale_si",
        bias_key="subs_bias_si",
        meta_keys=("subs_z_meta",),
        unit_key="subs_unit_to_si",
    )
    eps = tf_constant(1e-12, tf_float32)
    return (tf_cast(s_si, tf_float32) - b) / (a + eps)

def deg_to_m(
    axis: str,
    scaling_kwargs: Optional[Dict[str, Any]],
) -> Tensor:
    """Meters per degree factor for lon/lat coords."""
    if axis not in ("x", "y"):
        raise ValueError(
            f"deg_to_m: axis must be 'x' or 'y', got {axis!r}."
        )

    cfg = scaling_kwargs or {}
    if not bool(cfg.get("coords_in_degrees", False)):
        return tf_constant(1.0, tf_float32)

    key = "deg_to_m_lon" if axis == "x" else "deg_to_m_lat"
    val = cfg.get(key, None)
    if val is None:
        raise ValueError(
            "coords_in_degrees=True but missing "
            f"scaling_kwargs[{key!r}]."
        )

    try:
        v = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid {key!r}={val!r}.") from e

    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"Invalid {key!r}={v}.")

    return tf_constant(v, tf_float32)


def coord_ranges(
    scaling_kwargs: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (tR,xR,yR) if coords_normalized."""
    cfg = scaling_kwargs or {}
    if not bool(cfg.get("coords_normalized", False)):
        return None, None, None

    r = cfg.get("coord_ranges", {}) or {}

    def get(name: str, *alts: str) -> Optional[float]:
        v = r.get(name, None)
        if v is None:
            for a in alts:
                v = cfg.get(a, None)
                if v is not None:
                    break
        return None if v is None else float(v)

    tR = get("t", "t_range", "coord_range_t")
    xR = get("x", "x_range", "coord_range_x")
    yR = get("y", "y_range", "coord_range_y")
    return tR, xR, yR


def validate_scaling_kwargs(
    scaling_kwargs: Optional[Dict[str, Any]],
) -> None:
    """Basic scaling sanity checks."""
    sk = scaling_kwargs or {}

    # --- lon/lat degrees mode ------------------------------------------
    # v3.2 supports coords in degrees by converting spatial derivatives
    # using stored meters-per-degree factors (deg_to_m_lon/lat).
    if bool(sk.get("coords_in_degrees", False)):
        for key in ("deg_to_m_lon", "deg_to_m_lat"):
            val = sk.get(key, None)
            if val is None:
                raise ValueError(
                    "coords_in_degrees=True but missing "
                    f"scaling_kwargs[{key!r}]."
                )
            try:
                v = float(val)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid {key!r}={val!r}.") from e
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError(f"Invalid {key!r}={v}.")

    if bool(sk.get("coords_normalized", False)) and not sk.get(
        "coord_ranges",
        None,
    ):
        raise ValueError(
            "coords_normalized=True but coord_ranges missing."
        )

    if "time_units" not in sk:
        raise ValueError("time_units missing in scaling_kwargs.")


def resolve_gwl_dyn_index(
    scaling_kwargs: Optional[Dict[str, Any]],
) -> int:
    """Resolve GWL channel index for dynamic_features."""
    sk = scaling_kwargs or {}

    idx = sk.get("gwl_dyn_index", None)
    if idx is not None:
        return int(idx)

    names = sk.get("dynamic_feature_names", None)
    gwl_col = sk.get("gwl_col", None)

    if names is not None and gwl_col is not None:
        names = list(names)
        if gwl_col in names:
            return int(names.index(gwl_col))

    raise ValueError(
        "Cannot resolve GWL channel. Provide gwl_dyn_index "
        "or dynamic_feature_names + gwl_col."
    )


def get_gwl_dyn_index_cached(model) -> int:
    """Cache gwl_dyn_index on model after first resolve."""
    idx = getattr(model, "gwl_dyn_index", None)
    if idx is None:
        idx = resolve_gwl_dyn_index(getattr(
            model,
            "scaling_kwargs",
            None,
        ))
        setattr(model, "gwl_dyn_index", int(idx))
    return int(idx)


def resolve_subs_dyn_index(
    scaling_kwargs: Optional[Dict[str, Any]],
) -> int:
    """Resolve subsidence channel index for dynamic_features.

    This is optional: v3.2 can use historical subsidence as a dynamic
    driver to provide a physics-friendly initial condition for the mean
    settlement path.
    """
    sk = scaling_kwargs or {}

    idx = sk.get("subs_dyn_index", None)
    if idx is not None:
        return int(idx)

    names = sk.get("dynamic_feature_names", None)
    subs_col = sk.get("subs_dyn_name", None) or sk.get("subs_col", None)
    if names is not None and subs_col is not None:
        names = list(names)
        if subs_col in names:
            return int(names.index(subs_col))

    raise ValueError(
        "Cannot resolve subsidence channel. Provide subs_dyn_index "
        "or dynamic_feature_names + subs_dyn_name."
    )


def get_subs_dyn_index_cached(model) -> int:
    """Cache subs_dyn_index on model after first resolve."""
    idx = getattr(model, "subs_dyn_index", None)
    if idx is None:
        idx = resolve_subs_dyn_index(getattr(model, "scaling_kwargs", None))
        setattr(model, "subs_dyn_index", int(idx))
    return int(idx)


def slice_dynamic_channel(Xh: Tensor, idx: int) -> Tensor:
    """Slice (B,T,F) -> (B,T,1) at idx."""
    idx_t = tf_cast(idx, tf_int32)
    F = tf_shape(Xh)[-1]
    tf_debugging.assert_less(
        idx_t,
        F,
        message="gwl_dyn_index out of range.",
    )
    return Xh[:, :, idx_t:idx_t + 1]


def assert_dynamic_names_match_tensor(
    Xh: Tensor,
    scaling_kwargs: Optional[Dict[str, Any]],
) -> None:
    """Check dynamic_feature_names length matches Xh."""
    sk = scaling_kwargs or {}
    names = sk.get("dynamic_feature_names", None)
    if names is None:
        return
    n = len(list(names))
    tf_debugging.assert_equal(
        tf_shape(Xh)[-1],
        tf_constant(n, tf_int32),
        message="dynamic_feature_names != Xh last dim",
    )

def gwl_to_head_m(
    v_m: Tensor,
    scaling_kwargs: Optional[Dict[str, Any]],
    *,
    inputs: Optional[Dict[str, Tensor]] = None,
) -> Tensor:
    """Convert depth-bgs to head if requested.

    If `gwl_kind` is missing, infer it from `gwl_col`: any name
    containing "depth" is treated as depth (down-positive by default).
    This keeps raw and *_z column variants consistent.
    """
    sk = scaling_kwargs or {}

    kind_raw = sk.get("gwl_kind", None)
    if kind_raw is None or str(kind_raw).strip() == "":
        gwl_col = str(sk.get("gwl_col", "")).lower()
        kind = "depth" if ("depth" in gwl_col) else "head"
    else:
        kind = str(kind_raw).lower()

    # Any non-head kind is treated as depth (backward-compatible with
    # "depth_bgs", etc.)
    if kind in ("head", "waterhead", "hydraulic_head"):
        return tf_cast(v_m, tf_float32)

    sign = str(sk.get("gwl_sign", "down_positive")).lower()
    proxy = bool(sk.get("use_head_proxy", True))
    z_surf_col = sk.get("z_surf_col", None)

    v_m = tf_cast(v_m, tf_float32)
    depth_m = v_m if sign == "down_positive" else -v_m

    z_surf = None

    if inputs is not None and z_surf_col:
        z_surf = inputs.get(z_surf_col, None)
        if z_surf is not None:
            z_surf = tf_cast(z_surf, tf_float32)

    if z_surf is None and inputs is not None:
        sf = inputs.get("static_features", None)
        if sf is not None:
            sf = tf_cast(sf, tf_float32)

            idx = sk.get("z_surf_static_index", None)
            if idx is None:
                names = sk.get("static_feature_names", None)
                if names is None:
                    names = sk.get("static_features_names", None)
                if names is not None and z_surf_col is not None:
                    names = list(names)
                    if z_surf_col in names:
                        idx = int(names.index(z_surf_col))

            if idx is not None:
                idx_i = int(idx)
                tf_debugging.assert_less(
                    tf_cast(idx_i, tf_int32),
                    tf_shape(sf)[-1],
                    message="z_surf_static_index out of range.",
                )

                r = getattr(sf.shape, "rank", None)
                if r == 2:
                    z_surf = sf[:, idx_i:idx_i + 1]
                elif r == 3:
                    z_surf = sf[:, :, idx_i:idx_i + 1]
                else:
                    rr = tf_rank(sf)
                    z_surf = tf_cond(
                        tf_equal(rr, 2),
                        lambda: sf[:, idx_i:idx_i + 1],
                        lambda: sf[:, :, idx_i:idx_i + 1],
                    )

    if z_surf is not None:
        r = tf_rank(z_surf)
        z_surf = tf_cond(
            tf_equal(r, 1),
            lambda: z_surf[:, None, None],
            lambda: tf_cond(
                tf_equal(r, 2),
                lambda: z_surf[:, None, :],
                lambda: z_surf,
            ),
        )
        z_surf = z_surf + tf_zeros_like(depth_m)
        return z_surf - depth_m

    return -depth_m if proxy else depth_m


def _reshape_to_b11(v: Tensor) -> Tensor:
    """Coerce a tensor to (B,1,1) if possible."""
    v = tf_cast(v, tf_float32)
    r = tf_rank(v)
    return tf_cond(
        tf_equal(r, 1),
        lambda: v[:, None, None],
        lambda: tf_cond(
            tf_equal(r, 2),
            lambda: v[:, None, :],
            lambda: v,
        ),
    )


def get_h_hist_si(
    model,
    inputs: Dict[str, Tensor],
    *,
    want_head: bool = True,
) -> Tensor:
    """Return head (or depth) history in SI meters.

    Parameters
    ----------
    model : object
        The model instance (provides ``scaling_kwargs`` and cached indices).
    inputs : dict
        Batch inputs; expects ``dynamic_features`` unless an explicit
        head history key is provided.
    want_head : bool, default=True
        If True, convert depth-bgs to hydraulic head when possible.

    Returns
    -------
    Tensor
        (B,T,1) tensor in SI meters.
    """
    sk = getattr(model, "scaling_kwargs", None)

    # Explicit override (useful for scenario-driven runs)
    for k in ("h_hist_si", "head_hist_si", "gwl_hist_si"):
        if k in inputs and inputs[k] is not None:
            v = tf_cast(inputs[k], tf_float32)
            # (B,T) -> (B,T,1)
            if tf_equal(tf_rank(v), 2):
                v = v[:, :, None]
            if want_head:
                v = gwl_to_head_m(v, sk, inputs=inputs)
            return v

    Xh = inputs.get("dynamic_features", None)
    if Xh is None:
        raise ValueError(
            "Cannot build head history: missing inputs['dynamic_features'] "
            "and no explicit head history key (h_hist_si/head_hist_si)."
        )

    Xh = tf_cast(Xh, tf_float32)
    assert_dynamic_names_match_tensor(Xh, sk)

    gwl_idx = get_gwl_dyn_index_cached(model)
    gwl = slice_dynamic_channel(Xh, gwl_idx)
    gwl_si = to_si_head(gwl, sk)

    return gwl_to_head_m(gwl_si, sk, inputs=inputs) if want_head else gwl_si


def get_s_init_si(
    model,
    inputs: Optional[Dict[str, Tensor]],
    like: Tensor,
) -> Tensor:
    """Return initial settlement (cumulative subsidence) in SI meters.

    Priority:
    1) explicit keys in inputs (s_init_si/subs_hist_last_si/...)
    2) last historical value from dynamic_features if subs_dyn_index exists
    3) zeros (broadcast)
    """
    sk = getattr(model, "scaling_kwargs", None)

    if inputs is not None:
        for k in (
            "s_init_si", "subs_init_si", "subs_hist_last_si",
            "s_ref_si", "subs_ref_si", "s_init", "subs_init",
        ):
            if k in inputs and inputs[k] is not None:
                return _reshape_to_b11(inputs[k]) + tf_zeros_like(like)

        Xh = inputs.get("dynamic_features", None)
        if Xh is not None:
            try:
                subs_idx = get_subs_dyn_index_cached(model)
            except Exception:
                subs_idx = None
            if subs_idx is not None:
                Xh = tf_cast(Xh, tf_float32)
                assert_dynamic_names_match_tensor(Xh, sk)
                s_hist = slice_dynamic_channel(Xh, int(subs_idx))
                s_last = s_hist[:, -1:, :]
                s_last_si = to_si_subsidence(s_last, sk)
                return s_last_si + tf_zeros_like(like)

    return tf_zeros_like(like)


def get_h_ref_si(
    model,
    inputs: Optional[Dict[str, Tensor]],
    like: Tensor,
) -> Tensor:
    """Return h_ref in SI meters, broadcast to like."""
    # sk = getattr(model, "scaling_kwargs", None)

    mode = getattr(getattr(model, "h_ref_config", None), "mode", "auto")
    mode = "fixed" if str(mode).lower().strip() == "fixed" else "auto"

    if inputs is not None:
        for k in ("h_ref_si", "head_ref_si", "h_ref", "head_ref"):
            if (k in inputs) and (inputs[k] is not None):
                h_ref = tf_cast(inputs[k], tf_float32)
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
                return h_ref + tf_zeros_like(like)

    if (
        mode != "fixed"
        and inputs is not None
        and "dynamic_features" in inputs
        and inputs["dynamic_features"] is not None
    ):
        h_hist = get_h_hist_si(model, inputs, want_head=True)
        return h_hist[:, -1:, :] + tf_zeros_like(like)

    h0 = tf_cast(getattr(model, "h_ref", 0.0), tf_float32)
    h0 = h0[None, None, None]
    return h0 + tf_zeros_like(like)


def infer_dt_units_from_t(
    t_BH1: Tensor,
    scaling_kwargs: Optional[Dict[str, Any]],
    *,
    eps: float = 1e-12,
) -> Tensor:
    """
    Infer per-step dt in *time_units* from time tensor t(B,H,1).

    Shapes
    ------
    t_BH1 : (B,H,1)
    returns: (B,H,1)

    Notes
    -----
    - dt uses diffs along H; first step uses the first diff.
    - If coords are normalized, dt is multiplied by the de-normalization
      time range tR (from coord_ranges()).
    - Output is clipped to >= eps.
    """
    
    sk = scaling_kwargs or {}
    t = tf_convert_to_tensor(t_BH1, dtype=tf_float32)

    # t shape: (B,H,1)
    H = tf_shape(t)[1]
    dt_default = tf_ones_like(t)  # (B,H,1), safe in-graph

    def _multi_step():
        diffs = t[:, 1:, :] - t[:, :-1, :]         # (B,H-1,1)
        dt_first = diffs[:, :1, :]                 # (B,1,1)
        dt = tf_concat([dt_first, diffs], axis=1)  # (B,H,1)
        
        # If coords were normalized, dt is still normalized -> scale back
        if bool(sk.get("coords_normalized", False)):
            tR, _, _ = coord_ranges(sk)
            if tR is None:
                raise ValueError("coords_normalized=True but coord_ranges missing.")
            dt = dt * tf_constant(float(tR), dtype=tf_float32)
        return dt

    # if H <= 1: ones; else: diffs
    dt = tf_cond(tf_less_equal(H, 1), lambda: dt_default, _multi_step)
    return dt

def _infer_dt_units_from_t(
    t_BH1: Tensor,
    scaling_kwargs: Optional[Dict[str, Any]],
    *,
    eps: float = 1e-12,
) -> Tensor:
    """
    Infer per-step dt in *time_units* from time tensor t(B,H,1).

    Shapes
    ------
    t_BH1 : (B,H,1)
    returns: (B,H,1)

    Notes
    -----
    - dt uses diffs along H; first step uses the first diff.
    - If coords are normalized, dt is multiplied by the de-normalization
      time range tR (from coord_ranges()).
    - Output is clipped to >= eps.
    """
    cfg = scaling_kwargs or {}
    t = tf_cast(t_BH1, tf_float32)

    B = tf_shape(t)[0]
    H = tf_shape(t)[1]
    eps_t = tf_constant(eps, tf_float32)

    # Default dt = 1 time-unit
    dt = tf_ones([B, 1, 1], dtype=tf_float32)

    def _dt_from_diffs() -> Tensor:
        diffs = t[:, 1:, :] - t[:, :-1, :]          # (B,H-1,1)
        dt_first = diffs[:, :1, :]                  # (B,1,1)
        return tf_concat([dt_first, diffs], axis=1) # (B,H,1)

    dt = tf_cond(tf_greater(H, 1), _dt_from_diffs, lambda: dt)

    # If coords were normalized, dt is still normalized -> scale back
    if bool(cfg.get("coords_normalized", False)):
        tR, _, _ = coord_ranges(cfg)
        if tR is None:
            raise ValueError(
                "coords_normalized=True but coord_range_t missing."
            )
        dt = dt * tf_constant(float(tR), tf_float32)

    return tf_maximum(dt, eps_t)

