# -*- coding: utf-8 -*-
# _geoprior_utils.py
"""
GeoPrior small utilities (no derivatives here).
Short docs only; full docs later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from collections.abc import Mapping
from  warnings import warn
import json
from pathlib import Path

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
tf_minimum = KERAS_DEPS.minimum
tf_greater_equal = KERAS_DEPS.greater_equal
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
tf_abs = KERAS_DEPS.abs 
tf_print = KERAS_DEPS.print 
tf_reduce_mean = KERAS_DEPS.reduce_mean 
tf_expand_dims = KERAS_DEPS.expand_dims 
tf_tile = KERAS_DEPS.tile 



_EPSILON = 1e-12
# ---------------------------------------------------------------------
# Scaling kwargs access helpers (alias-safe)
# ---------------------------------------------------------------------
_SK_ALIASES = {
    # common naming drift
    "time_units": (
        "time_unit",
    ),
    "cons_residual_units": (
        "cons_residual_unit",
    ),

    # policy drift
    "scaling_error_policy": (
        "error_policy",
        "scaling_policy",
    ),

    # coord drift
    "coords_normalized": (
        "coord_normalized",
        "coords_norm",
    ),
    "coords_in_degrees": (
        "coord_in_degrees",
        "coords_deg",
    ),
    "coord_order": (
        "coords_order",
    ),
    "coord_ranges": (
        "coord_range",
    ),

    # feature-name list drift
    "dynamic_feature_names": (
        "dynamic_features_names",
        "dyn_feature_names",
    ),
    "future_feature_names": (
        "future_features_names",
        "fut_feature_names",
    ),
    "static_feature_names": (
        "static_features_names",
        "stat_feature_names",
    ),

    # feature-channel naming drift
    "gwl_col": (
        "gwl_dyn_name",
        "gwl_dyn_col",
        "gwl_name",
    ),
    "subs_dyn_name": (
        "subs_col",
        "subs_dyn_col",
        "subsidence_dyn_name",
    ),

    # feature-channel index drift
    "gwl_dyn_index": (
        "gwl_index",
        "gwl_feature_index",
        "gwl_channel_index",
    ),
    "subs_dyn_index": (
        "subs_index",
        "subs_feature_index",
        "subs_channel_index",
    ),

    # z_surf drift
    "z_surf_col": (
        "z_surf_key",
        "z_surf_name",
    ),
    # bounds drift (often nested under scaling_kwargs['bounds'])
    "log_tau_min": (
        "logTau_min",
        "logtau_min",
    ),
    "log_tau_max": (
        "logTau_max",
        "logtau_max",
    ),
    "tau_min": (
        "Tau_min",
        "tauMin",
        "tau_min_sec",
        "tau_min_seconds",
    ),
    "tau_max": (
        "Tau_max",
        "tauMax",
        "tau_max_sec",
        "tau_max_seconds",
    ),
    "tau_min_units": (
        "tau_min_time_units",
        "tau_min_in_time_units",
    ),
    "tau_max_units": (
        "tau_max_time_units",
        "tau_max_in_time_units",
    ),
    "Q_length_in_si": ( 
        "Q_in_m_per_s", 
        )

}

_SK_ALIASES.update({
    "cons_drawdown_mode": (
        "drawdown_mode",
        "cons_delta_mode",
    ),
    "cons_drawdown_rule": (
        "drawdown_rule",
        "cons_delta_rule",
    ),
    "cons_stop_grad_ref": (
        "stop_grad_ref",
        "cons_stopgrad_ref",
    ),
    "cons_drawdown_zero_at_origin": (
        "drawdown_zero_at_origin",
        "cons_zero_at_origin",
    ),
    "cons_drawdown_clip_max": (
        "drawdown_clip_max",
        "cons_clip_max",
    ),
    "cons_relu_beta": (
        "relu_beta",
        "cons_beta",
    ),
})


# MV prior drift (mode/weight/warmup + loss knobs)
_SK_ALIASES.update({
    "mv_prior_mode": (
        "mv_mode",
        "mvprior_mode",
        "mv_prior_kind",
    ),
    "mv_weight": (
        "mv_prior_weight",
        "mvprior_weight",
        "mv_w",
    ),
    "mv_warmup_steps": (
        "mv_prior_warmup_steps",
        "mv_warmup_steps",
        "mv_warmup_iters",
        "mv_warmup_iterations",
    ),
    "mv_alpha_disp": (
        "mv_prior_alpha_disp",
        "mv_disp_alpha",
        "mv_alpha",
    ),
    "mv_huber_delta": (
        "mv_prior_huber_delta",
        "mv_delta",
        "mv_huber",
    ),
    "mv_prior_units": (
        "mv_units",
        "mv_gamma_units",
        "mv_gw_units",
    ),
})


def enforce_scaling_alias_consistency(
    scaling_kwargs: Optional[Dict[str, Any]],
    *,
    where: str = "validate",
) -> None:
    """
    Enforce that canonical keys and aliases agree.

    If both canonical and an alias exist and their
    values differ, apply the scaling error policy.
    """
    sk = scaling_kwargs or {}

    for key, aliases in _SK_ALIASES.items():
        if key not in sk:
            continue

        v0 = sk.get(key, None)
        if v0 is None:
            continue

        for a in aliases:
            if a not in sk:
                continue

            va = sk.get(a, None)
            if va is None:
                continue

            if va != v0:
                msg = (
                    "Conflicting scaling keys: "
                    f"{key!r}={v0!r} != {a!r}={va!r}."
                )
                _handle_scaling_issue(
                    sk,
                    msg,
                    where=where,
                )


def canonicalize_scaling_kwargs(
    scaling_kwargs: Optional[Dict[str, Any]],
    *,
    copy: bool = True,
) -> Dict[str, Any]:
    """
    Return a canonicalized scaling dict.

    - If a canonical key is missing, but one of its
      aliases exists, copy alias -> canonical.
    - Keeps existing canonical values unchanged.
    """
    sk0 = scaling_kwargs or {}
    sk = dict(sk0) if copy else sk0

    for key, aliases in _SK_ALIASES.items():
        if key in sk and sk.get(key, None) is not None:
            continue

        for a in aliases:
            if a in sk and sk.get(a, None) is not None:
                sk[key] = sk[a]
                break

    return sk

def load_scaling_kwargs(
    scaling_kwargs: Optional[Any],
    *,
    copy: bool = True,
) -> Dict[str, Any]:
    """
    Load scaling kwargs from a dict-like object or JSON.

    Supported inputs
    ----------------
    - dict / Mapping:
        Returned (copied by default).
    - str:
        * If it looks like JSON ("{...}" or "[...]"), parse as JSON.
        * Else treat as a filesystem path to a JSON file.
    - pathlib.Path:
        Treated as a filesystem path to a JSON file.
    - None:
        Returns {}.

    Parameters
    ----------
    scaling_kwargs : Any
        Scaling configuration input. Can be a dict, JSON string,
        path to JSON file, or None.
    copy : bool, default=True
        If True, returns a shallow copy of the dict.

    Returns
    -------
    dict
        Parsed scaling kwargs as a Python dict.

    Raises
    ------
    TypeError
        If the input type is unsupported.
    ValueError
        If JSON parsing fails or JSON does not decode to a dict.
    FileNotFoundError
        If a JSON path is given but does not exist.
    """
    if scaling_kwargs is None:
        return {}

    if isinstance(scaling_kwargs, Mapping):
        return dict(scaling_kwargs) if copy else scaling_kwargs

    if isinstance(scaling_kwargs, Path):
        path = scaling_kwargs
        text = path.read_text(encoding="utf-8")
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(
                "Scaling JSON must decode to an object/dict, "
                f"got {type(obj).__name__}."
            )
        return obj

    if isinstance(scaling_kwargs, str):
        s = scaling_kwargs.strip()

        # 1) Inline JSON object/array.
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(
                    "Invalid scaling_kwargs JSON string."
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    "Scaling JSON must decode to an object/dict, "
                    f"got {type(obj).__name__}."
                )
                
            return obj

        # 2) Treat as file path to JSON.
        path = Path(s).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Scaling kwargs JSON file not found: {str(path)!r}."
            )
        text = path.read_text(encoding="utf-8")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in scaling kwargs file: {str(path)!r}."
            ) from e
        if not isinstance(obj, dict):
            raise ValueError(
                "Scaling JSON file must decode to an object/dict, "
                f"got {type(obj).__name__}."
            )
        return obj

    try:
        obj = dict(scaling_kwargs)
    except Exception as e:
        raise TypeError(
            "scaling_kwargs must be a dict/Mapping, JSON string, "
            "Path, or a path string to a JSON file."
        ) from e

    return obj

def get_sk(
    scaling_kwargs,
    key: str,
    *aliases: str,
    default=None,
    required: bool = False,
    cast=None,
):
    """
    Fetch a key from `scaling_kwargs` with aliases + default.

    - Tries: key -> built-in aliases -> explicit aliases
    - Treats None and blank strings as "missing" and keeps searching.
    """
    sk = scaling_kwargs or {}
    if not isinstance(sk, Mapping):
        try:
            sk = dict(sk)
        except Exception:
            sk = {}

    cand = [key]
    cand.extend(_SK_ALIASES.get(key, ()))
    cand.extend([a for a in aliases if a])

    for k in cand:
        if k in sk:
            v = sk[k]
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            if cast is not None:
                try:
                    v = cast(v)
                except Exception as e:
                    raise ValueError(f"Invalid scaling_kwargs[{k!r}]={v!r}.") from e
            return v

    if required:
        alias_txt = ", ".join(repr(x) for x in cand[1:]) or "none"
        raise ValueError(
            f"Missing required scaling key {key!r} (aliases: {alias_txt})."
        )
    if cast is not None and default is not None:
        try:
            return cast(default)
        except Exception:
            return default
    return default


def _norm_policy(policy: Optional[str]) -> str:
    """
    Normalize scaling error policy.

    Allowed:
    - 'ignore'
    - 'warn'   (default)
    - 'raise'
    """
    p = (policy or "warn").strip().lower()
    if p not in ("ignore", "warn", "raise"):
        p = "warn"
    return p


def _handle_scaling_issue(
    scaling_kwargs: Optional[Dict[str, Any]],
    message: str,
    *,
    where: str = "validate",
) -> None:
    """
    Apply scaling error policy.

    Notes
    -----
    You asked for: even if policy is 'raise', runtime
    fallback paths should still fall back to zeros.
    So:
    - where='validate': obey ignore/warn/raise
    - where='runtime' : treat 'raise' as 'warn'
    """
    sk = scaling_kwargs or {}
    policy = _norm_policy(
        get_sk(sk, "scaling_error_policy", default="warn")
    )

    # Runtime must not crash; still fall back later.
    if where != "validate" and policy == "raise":
        policy = "warn"

    if policy == "ignore":
        return

    if policy == "warn":
        warn(
            message,
            category=RuntimeWarning,
            stacklevel=2,
        )
        return
    # validate + raise
    raise ValueError(message)

def validate_scaling_kwargs(
    scaling_kwargs: Optional[Dict[str, Any]],
) -> None:
    """
    Basic scaling sanity checks.

    This includes policy-controlled heuristic checks
    for common "silent fallback" cases.
    """
    sk = canonicalize_scaling_kwargs(scaling_kwargs)
    enforce_scaling_alias_consistency(sk, where="validate")

    # --------------------------------------------------
    # Degrees mode requires meters-per-degree factors.
    # --------------------------------------------------
    if bool(sk.get("coords_in_degrees", False)):
        for key in ("deg_to_m_lon", "deg_to_m_lat"):
            val = sk.get(key, None)
            if val is None:
                msg = (
                    "coords_in_degrees=True but missing "
                    f"scaling_kwargs[{key!r}]."
                )
                raise ValueError(msg)
            try:
                v = float(val)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid {key!r}={val!r}."
                ) from e
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError(f"Invalid {key!r}={v}.")

    # --------------------------------------------------
    # Normalized coords require coord_ranges.
    # --------------------------------------------------
    if bool(sk.get("coords_normalized", False)) and not sk.get(
        "coord_ranges",
        None,
    ):
        raise ValueError(
            "coords_normalized=True but coord_ranges missing."
        )

    # --------------------------------------------------
    # Require time units (alias-safe).
    # --------------------------------------------------
    if get_sk(sk, "time_units", default=None) is None:
        raise ValueError("time_units missing in scaling_kwargs.")

    # --------------------------------------------------
    # Heuristic checks (policy-controlled).
    # --------------------------------------------------
    names = sk.get("dynamic_feature_names", None)
    names = list(names) if names is not None else []

    # A) Subsidence init: detect cum subs channel.
    has_subs_cum = any(
        ("subs" in str(n).lower() and "cum" in str(n).lower())
        for n in names
    )

    subs_idx = sk.get("subs_dyn_index", None)
    subs_name = get_sk(sk, "subs_dyn_name", default=None)

    meta = sk.get("gwl_z_meta", {}) or {}
    cols = meta.get("cols", {}) or {}
    subs_meta = cols.get("subs_model", None)

    if has_subs_cum and subs_idx is None and subs_name is None:
        if subs_meta is None:
            msg = (
                "dynamic_feature_names contains a cumulative "
                "subsidence channel, but no subs_dyn_index/"
                "subs_dyn_name and no gwl_z_meta.cols.subs_model. "
                "Initial settlement will fall back to zeros."
            )
            _handle_scaling_issue(
                sk,
                msg,
                where="validate",
            )

    # B) Depth->head conversion needs z_surf when proxy=False.
    kind = str(sk.get("gwl_kind", "")).lower()
    proxy = bool(sk.get("use_head_proxy", True))

    if (not proxy) and (
        kind not in ("head", "waterhead", "hydraulic_head")
    ):
        z_col = sk.get("z_surf_col", None)
        z_col = z_col or meta.get("z_surf_col", None)

        z_static = cols.get("z_surf_static", None)
        z_idx = sk.get("z_surf_static_index", None)

        static_names = get_sk(
            sk,
            "static_feature_names",
            default=None,
        )

        # If you did not provide a way to locate z_surf
        # in static features, conversion may fallback.
        if (
            z_idx is None
            and static_names is None
            and z_col is not None
            and z_static is not None
            and z_col != z_static
        ):
            msg = (
                "use_head_proxy=False and gwl_kind is depth-like, "
                "but z_surf_col differs from gwl_z_meta.cols."
                "z_surf_static, and no static_feature_names/"
                "z_surf_static_index provided. Depth->head "
                "conversion may fall back to depth."
            )
            _handle_scaling_issue(
                sk,
                msg,
                where="validate",
            )


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
    eps = tf_constant(_EPSILON, tf_float32)
    return (tf_cast(s_si, tf_float32) - b) / (a + eps)

def deg_to_m(
    axis: str,
    scaling_kwargs: Optional[Dict[str, Any]],
) -> Tensor:
    """
    Meters per degree factor for lon/lat coords.

    If coords_in_degrees=True and deg_to_m_lon/lat are missing, we try
    to compute them from lat0_deg (recommended).
    """
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
        lat0 = cfg.get("lat0_deg", None)
        if lat0 is None:
            raise ValueError(
                "coords_in_degrees=True but missing deg_to_m_lon/deg_to_m_lat "
                "and lat0_deg (needed for lon scaling)."
            )
        lat0 = float(lat0)
        if axis == "x":
            v = 111320.0 * float(np.cos(np.deg2rad(lat0)))
        else:
            v = 110574.0
        return tf_constant(v, tf_float32)
    
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

def resolve_gwl_dyn_index(
    scaling_kwargs: Optional[Dict[str, Any]],
) -> int:
    """Resolve GWL channel index for dynamic_features."""
    sk = scaling_kwargs or {}

    idx = sk.get("gwl_dyn_index", None)
    if idx is not None:
        return int(idx)

    names = sk.get("dynamic_feature_names", None)
    gwl_col = get_sk(sk, "gwl_col", default=None)

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

def resolve_subs_dyn_index(scaling_kwargs):
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

    subs_col = get_sk(sk, "subs_dyn_name", default=None)

    # NEW: fallback to gwl_z_meta.cols.subs_model
    if subs_col is None:
        meta = sk.get("gwl_z_meta", {}) or {}
        cols = meta.get("cols", {}) or {}
        subs_col = cols.get("subs_model", None)

    if names is not None and subs_col is not None:
        names = list(names)
        if subs_col in names:
            return int(names.index(subs_col))

    raise ValueError(
        "Cannot resolve subsidence channel. Provide subs_dyn_index "
        "or dynamic_feature_names + subs_dyn_name (or gwl_z_meta.cols.subs_model)."
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
    """
    Convert depth-bgs to head if possible.

    Behavior
    --------
    - If gwl_kind is head-like: return v_m.
    - Otherwise treat as depth and try:
      head = z_surf - depth.
    - If z_surf is missing:
      * use_head_proxy=True  -> return -depth
      * use_head_proxy=False -> return depth
    """
    sk = scaling_kwargs or {}

    # --------------------------------------------------
    # 1) Decide whether v_m is head or depth.
    # --------------------------------------------------
    kind_raw = sk.get("gwl_kind", None)
    if kind_raw is None or str(kind_raw).strip() == "":
        gwl_col = str(get_sk(sk, "gwl_col", default=""))
        gwl_col = gwl_col.lower()
        kind = "depth" if ("depth" in gwl_col) else "head"
    else:
        kind = str(kind_raw).lower()

    if kind in ("head", "waterhead", "hydraulic_head"):
        return tf_cast(v_m, tf_float32)

    # --------------------------------------------------
    # 2) Depth convention + proxy behavior.
    # --------------------------------------------------
    sign = str(sk.get("gwl_sign", "down_positive")).lower()
    proxy = bool(sk.get("use_head_proxy", True))

    # --------------------------------------------------
    # 3) Collect possible z_surf keys.
    # Prefer SI/static key first when available.
    # --------------------------------------------------
    meta = sk.get("gwl_z_meta", {}) or {}
    cols = meta.get("cols", {}) or {}

    z_surf_col = sk.get("z_surf_col", None)
    z_surf_col = z_surf_col or meta.get("z_surf_col", None)

    z_surf_static = cols.get("z_surf_static", None)
    z_surf_raw = cols.get("z_surf_raw", None)

    z_surf_keys = [
        k
        for k in (z_surf_static, z_surf_col, z_surf_raw)
        if k
    ]

    # Dedupe while preserving order.
    seen = set()
    z_surf_keys = [
        k
        for k in z_surf_keys
        if not (k in seen or seen.add(k))
    ]

    # --------------------------------------------------
    # 4) Convert to positive-down depth.
    # --------------------------------------------------
    v_m = tf_cast(v_m, tf_float32)
    depth_m = v_m if sign == "down_positive" else -v_m

    # --------------------------------------------------
    # 5) Try direct inputs[z_surf_key] first.
    # --------------------------------------------------
    z_surf = None
    if inputs is not None:
        for k in z_surf_keys:
            z_surf = inputs.get(k, None)
            if z_surf is not None:
                z_surf = tf_cast(z_surf, tf_float32)
                break

    # --------------------------------------------------
    # 6) If missing, try static_features lookup.
    # --------------------------------------------------
    if z_surf is None and inputs is not None:
        sf = inputs.get("static_features", None)
        if sf is not None:
            sf = tf_cast(sf, tf_float32)

            idx = sk.get("z_surf_static_index", None)
            if idx is None:
                names = get_sk(
                    sk,
                    "static_feature_names",
                    default=None,
                )
                if names is not None:
                    names = list(names)
                    for k in z_surf_keys:
                        if k in names:
                            idx = int(names.index(k))
                            break

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
                    
    if z_surf is None:
        # if bool(sk.get("debug_units", False)):
        tf_print("[gwl_to_head_m] z_surf missing ->",
                 "use_head_proxy=", bool(sk.get("use_head_proxy", False)),
                 "returning depth-like quantity (NOT true head)")

    # --------------------------------------------------
    # 7) If we have z_surf: head = z_surf - depth.
    # --------------------------------------------------
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

        # Broadcast z_surf to match depth_m.
        z_surf = z_surf + tf_zeros_like(depth_m)
        return z_surf - depth_m

    # --------------------------------------------------
    # 8) Fallback: proxy head or keep depth.
    # --------------------------------------------------
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
            except Exception as e:
                _handle_scaling_issue(
                    getattr(model, "scaling_kwargs", None),
                    f"Could not resolve subsidence init channel ({e}). "
                    "Falling back to zeros for s_init_si.",
                    where="runtime",
                )
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
    dt = tf_abs(dt)
    dt_pos = tf_greater(dt, tf_constant(0.0, tf_float32))
    dt_pos_f = tf_cast(dt_pos, tf_float32)
    dt = dt * dt_pos_f + dt_default * (1.0 - dt_pos_f)
    
    dt_eps = float(get_sk(sk, "dt_min_units", default=1e-6))
    dt = tf_maximum(dt, tf_constant(dt_eps, tf_float32))


    return dt


# ==============================================================
# VERBOSE SHAPE DIAGNOSTICS (paste-only snippets)
# - Uses tf_shape() (works in graph) + a small helper.
# - Triggered when verbose > 3.
# ==============================================================

def _vshape(tag, x):
    """Print runtime shape + rank (graph-safe)."""
    if x is None:
        tf_print("[shape]", tag, "= None")
        return
    xr = tf_rank(x)
    xs = tf_shape(x)
    tf_print("[shape]", tag, "rank=", xr, "shape=", xs)

def _vshapes(title, items):
    """Convenience: print a block of shapes."""
    tf_print("\n==========", title, "==========")
    for tag, x in items:
        _vshape(tag, x)
    tf_print("================================\n")



# ---------------------------------------------------------------------
# Training strategy gates (Q and subsidence residual)
# ---------------------------------------------------------------------
def policy_gate(
    step: Tensor,
    policy: str,
    *,
    warmup_steps: int = 0,
    ramp_steps: int = 0,
    dtype: Any = tf_float32,
) -> Tensor:
    r"""Return a scalar gate in ``[0,1]`` based on a policy + step.

    Parameters
    ----------
    step : Tensor
        Global step counter (typically ``optimizer.iterations``).
    policy : {"always_on","always_off","warmup_off"}
        Gating behavior:
        - ``always_on``  : gate = 1
        - ``always_off`` : gate = 0
        - ``warmup_off`` : gate = 0 for ``step < warmup_steps``,
          then ramps to 1 over ``ramp_steps`` (linear) if ``ramp_steps>0``,
          otherwise becomes 1 immediately at ``warmup_steps``.
    warmup_steps : int, default=0
        Number of steps to keep the gate at 0 (only for ``warmup_off``).
    ramp_steps : int, default=0
        Number of steps for a linear ramp from 0->1 after warmup.
        If 0, the gate is a hard step.
    dtype : dtype, default=tf_float32
        Output dtype.
    """
    pol = (policy or "always_on").strip().lower()
    if pol in ("always_on", "on", "true", "1"):
        return tf_constant(1.0, dtype=dtype)
    if pol in ("always_off", "off", "false", "0"):
        return tf_constant(0.0, dtype=dtype)

    w = int(warmup_steps or 0)
    r = int(ramp_steps or 0)

    if w <= 0 and r <= 0:
        return tf_constant(1.0, dtype=dtype)

    step_i = tf_cast(step, tf_int32)

    if r <= 0:
        return tf_cast(tf_greater_equal(step_i, tf_constant(w, tf_int32)), dtype)

    step_f = tf_cast(step_i, dtype)
    w_f = tf_constant(float(w), dtype)
    r_f = tf_constant(float(r), dtype)
    frac = (step_f - w_f) / r_f
    frac = tf_maximum(tf_constant(0.0, dtype), frac)
    frac = tf_minimum(tf_constant(1.0, dtype), frac)
    return frac

# ---------------------------------------------------------------------
# Quantile helpers (shape-safe)
# ---------------------------------------------------------------------
def _q_index(quantiles, q=0.5):
    """Return the index of q in quantiles (python int), else None."""
    if quantiles is None:
        return None
    try:
        qs = [float(x) for x in list(quantiles)]
    except Exception:
        return None
    # exact match first
    for i, qi in enumerate(qs):
        if abs(qi - float(q)) < 1e-12:
            return i
    # fallback: closest
    qv = float(q)
    i0 = int(np.argmin([abs(qi - qv) for qi in qs]))
    return i0


def select_q(pred, quantiles=None, q=0.5, fallback="mean"):
    """
    Select q-quantile from pred.

    pred:
      - (B,H,Q,O) -> returns (B,H,O)
      - (B,H,Q,1) -> returns (B,H,1)
      - otherwise returned as-is.
    """
    r = getattr(pred, "shape", None)
    rank = pred.shape.rank if r is not None else None

    if rank != 4:
        return pred

    idx = _q_index(quantiles, q=q)
    if idx is not None:
        return pred[:, :, idx, :]

    # fallback if quantiles list is missing/odd
    if str(fallback).lower() == "mean":
        return tf_reduce_mean(pred, axis=2)
    # "middle" fallback
    qn = tf_shape(pred)[2]
    mid = qn // 2
    return pred[:, :, mid, :]


def tile_true_to_quantiles(y_true, y_pred):
    """
    Make y_true compatible with y_pred when y_pred has quantile axis.

    y_true: (B,H,O) and y_pred: (B,H,Q,O) -> return y_true_q: (B,H,Q,O)
    Else return y_true unchanged.
    """
    rt = y_true.shape.rank
    rp = y_pred.shape.rank
    if (rp == 4) and (rt == 3):
        y4 = tf_expand_dims(y_true, axis=2)  # (B,H,1,O)
        qn = tf_shape(y_pred)[2]
        return tf_tile(y4, [1, 1, qn, 1])
    return y_true

# def _align_true_for_loss(yt, yp):
#     # If y_true is already tiled (B,H,Q,O), reduce it to (B,H,O).
#     # Pinball will broadcast over Q internally.
#     if yt.shape.rank == 4:
#         return yt[:, :, 0, :]
#     return yt

def _align_true_for_loss(y_true, y_pred):
    """
    Align y_true to y_pred for quantile-aware losses/metrics.

    Never tiles y_true across Q. Only reduces y_true if it accidentally
    arrives with a quantile axis.
    """
    y_true = tf_convert_to_tensor(y_true)
    y_pred = tf_convert_to_tensor(y_pred)

    # (B,H) -> (B,H,1)
    if y_true.shape.rank == 2:
        y_true = tf_expand_dims(y_true, axis=-1)

    rt = y_true.shape.rank
    rp = y_pred.shape.rank

    # (B,H,Q,O) -> (B,H,O)
    if rt == 4:
        return y_true[:, :, 0, :]

    # (B,H,Q) -> (B,H,1)  (rare but happens if upstream tiled targets)
    if rt == 3 and rp in (3, 4):
        if rp == 4:
            q_dim = y_pred.shape[2]
            if (q_dim is None) or (y_true.shape[-1] == q_dim):
                return tf_expand_dims(y_true[:, :, 0], axis=-1)
        else:
            return tf_expand_dims(y_true[:, :, 0], axis=-1)

    return y_true

# ---------------------------------------------------------------------
# Derived SI conversion helpers (optional, but recommended)
# ---------------------------------------------------------------------
def finalize_scaling_kwargs(sk: Dict[str, Any]) -> Dict[str, Any]:
    """Add derived SI conversion constants to ``scaling_kwargs``.

    Adds (when possible):
    - ``seconds_per_time_unit``: float
    - ``coord_ranges_si``: dict with keys ``t`` (seconds), ``x``/``y`` (meters)
    - ``coord_inv_ranges_si``: inverse of the above (safe floor).

    Notes
    -----
    This helper is designed to be called *once* when assembling
    ``scaling_kwargs`` (e.g., in your stage2 script) so the model can
    reuse those constants without recomputing unit conversions in the
    hot training loop.
    """
    if sk is None:
        return sk

    sk = dict(sk)

    tu = str(get_sk(sk, "time_units", default="second")).strip().lower()
    time_unit_to_seconds = {
        "second": 1.0, "sec": 1.0, "s": 1.0,
        "minute": 60.0, "min": 60.0, "m": 60.0,
        "hour": 3600.0, "h": 3600.0,
        "day": 86400.0, "d": 86400.0,
        # Julian year (365.2425 days) to match prior_maths.py
        "year": 31556952.0, "yr": 31556952.0, "y": 31556952.0,
    }
    sec_u = float(time_unit_to_seconds.get(tu, 1.0))
    sk.setdefault("seconds_per_time_unit", sec_u)

    cr = get_sk(sk, "coord_ranges", default=None)
    if isinstance(cr, Mapping) and all(k in cr for k in ("t", "x", "y")):
        tR = float(cr.get("t", 1.0))
        xR = float(cr.get("x", 1.0))
        yR = float(cr.get("y", 1.0))

        # If coordinates are degrees, convert spans to meters.
        if bool(get_sk(sk, "coords_in_degrees", default=False)):
            deg_to_m_lon = get_sk(sk, "deg_to_m_lon", default=None)
            deg_to_m_lat = get_sk(sk, "deg_to_m_lat", default=None)
            if deg_to_m_lon is not None and deg_to_m_lat is not None:
                xR *= float(deg_to_m_lon)
                yR *= float(deg_to_m_lat)

        # Convert time span to seconds (important if coords_normalized=True).
        tR *= sec_u

        sk["coord_ranges_si"] = {"t": tR, "x": xR, "y": yR}
        eps = 1e-12
        sk["coord_inv_ranges_si"] = {
            "t": 1.0 / max(tR, eps),
            "x": 1.0 / max(xR, eps),
            "y": 1.0 / max(yR, eps),
        }

    return sk


def coord_ranges_si(
        sk: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return coordinate spans in SI (t in seconds; x/y in meters).

    If ``coord_ranges_si`` is present in ``sk``, it is used directly.
    Otherwise, this is computed from ``coord_ranges`` and ``time_units``
    (and degree-to-meter factors when applicable).
    """
    cr_si = get_sk(sk, "coord_ranges_si", default=None)
    if isinstance(cr_si, Mapping) and all(k in cr_si for k in ("t", "x", "y")):
        return float(cr_si["t"]), float(cr_si["x"]), float(cr_si["y"])

    sk2 = finalize_scaling_kwargs(sk)
    cr_si = get_sk(sk2, "coord_ranges_si", default=None)
    if isinstance(cr_si, Mapping) and all(k in cr_si for k in ("t", "x", "y")):
        return float(cr_si["t"]), float(cr_si["x"]), float(cr_si["y"])

    return None, None, None
