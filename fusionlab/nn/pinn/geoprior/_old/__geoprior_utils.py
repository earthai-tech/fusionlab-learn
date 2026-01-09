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

    if sk.get("coords_in_degrees", False):
        raise ValueError(
            "coords_in_degrees=True but you feed UTM meters."
        )

    if sk.get("coords_normalized", False) and not sk.get(
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
    """Convert depth-bgs to head if requested."""
    sk = scaling_kwargs or {}

    kind = str(sk.get("gwl_kind", "head")).lower()
    sign = str(sk.get("gwl_sign", "down_positive")).lower()
    proxy = bool(sk.get("use_head_proxy", True))
    z_surf_col = sk.get("z_surf_col", None)

    if kind == "head":
        return v_m

    depth_m = v_m if sign == "down_positive" else -v_m

    z_surf = None
    if inputs is not None and z_surf_col:
        z_surf = inputs.get(z_surf_col, None)
        if z_surf is not None:
            z_surf = tf_cast(z_surf, tf_float32)

    if z_surf is not None:
        return z_surf - depth_m

    return -depth_m if proxy else depth_m


def get_h_ref_si(
    model,
    inputs: Optional[Dict[str, Tensor]],
    like: Tensor,
) -> Tensor:
    """Return h_ref in SI meters, broadcast to like."""
    sk = getattr(model, "scaling_kwargs", None)

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
        Xh = tf_cast(inputs["dynamic_features"], tf_float32)
        assert_dynamic_names_match_tensor(Xh, sk)

        idx = get_gwl_dyn_index_cached(model)
        gwl = slice_dynamic_channel(Xh, idx)
        gwl_hist = gwl[:, -1:, :]

        gwl_si = to_si_head(gwl_hist, sk)
        href = gwl_to_head_m(gwl_si, sk, inputs=inputs)
        return href + tf_zeros_like(like)

    h0 = tf_cast(getattr(model, "h_ref", 0.0), tf_float32)
    h0 = h0[None, None, None]
    return h0 + tf_zeros_like(like)


