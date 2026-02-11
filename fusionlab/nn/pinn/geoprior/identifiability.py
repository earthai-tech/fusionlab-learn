# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
Identifiability scenarios for GeoPrior-style models.

Goal:
- break non-identifiability ridges by construction.

Option A:
- learn tau only
- derive K from tau via closure
- freeze (or fix) Ss and Hd
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ( 
    Any, 
    Dict, 
    Mapping, 
    Optional, 
    Iterable
)

import numpy as np

from ... import KERAS_DEPS

K = KERAS_DEPS

Tensor = K.Tensor

tf_cast = K.cast
tf_constant = K.constant
tf_maximum = K.maximum
tf_reduce_mean = K.reduce_mean
tf_square = K.square
tf_stop_gradient = K.stop_gradient

tf_math = K.math


_PI2 = float(np.pi**2)

_IDENTIFIABILITY_PROFILES = {
    "base": {
        "sk": {
            "freeze_physics_fields_over_time": True,
            "bounds_loss_kind": "barrier",
            "physics_warmup_steps": 500,
            "physics_ramp_steps": 500,
        },
        "compile": {
            "lambda_bounds": 1.0,
            "lambda_prior": 1.0,
            "lambda_q": 0.0,
            "lambda_mv": 0.0,
        },
        "locks": {},
    },
    "anchored": {
        "sk": {
            "freeze_physics_fields_over_time": True,
            "allow_subs_residual": False,
            "bounds_loss_kind": "barrier",
            "bounds_beta": 30.0,
            "bounds_guard": 5.0,
            "bounds_w": 2.0,
            "bounds_include_tau": True,
            "bounds_tau_w": 2.0,
            "stop_grad_ref": True,
            "drawdown_zero_at_origin": True,
            "physics_warmup_steps": 200,
            "physics_ramp_steps": 800,
        },
        "compile": {
            "lambda_bounds": 10.0,
            "lambda_prior": 10.0,
            "lambda_cons": 50.0,
            "lambda_q": 0.0,
            "lambda_mv": 0.0,
        },
        "locks": {},
    },
    "closure_locked": {
        "sk": {
            "freeze_physics_fields_over_time": True,
            "allow_subs_residual": False,
            "bounds_loss_kind": "barrier",
            "bounds_beta": 30.0,
            "bounds_w": 2.0,
            "bounds_include_tau": True,
            "bounds_tau_w": 3.0,
            "stop_grad_ref": True,
            "drawdown_zero_at_origin": True,
            "physics_warmup_steps": 0,
            "physics_ramp_steps": 500,
        },
        "compile": {
            "lambda_bounds": 10.0,
            "lambda_prior": 100.0,
            "lambda_cons": 50.0,
            "lambda_q": 0.0,
            "lambda_mv": 0.0,
        },
        "locks": {"tau_head": True},
    },
    "data_relaxed": {
        "sk": {
            "freeze_physics_fields_over_time": False,
            "allow_subs_residual": True,
            "bounds_loss_kind": "residual",
            "bounds_beta": 10.0,
            "bounds_w": 1.0,
            "bounds_include_tau": True,
            "bounds_tau_w": 1.0,
            "stop_grad_ref": False,
            "drawdown_zero_at_origin": False,
            "physics_warmup_steps": 1500,
            "physics_ramp_steps": 1500,
        },
        "compile": {
            "lambda_bounds": 1.0,
            "lambda_prior": 0.0,
            "lambda_cons": 1.0,
            "lambda_gw": 0.0,
            "lambda_q": 0.0,
            "lambda_mv": 0.0,
        },
        "locks": {},
    },
}



__all__ = [
    "init_identifiability",
    "apply_ident_locks",
    "resolve_compile_weights",
    "get_ident_profile",
    "ident_audit_dict",
]

_AUDIT_SK_KEYS = (
    "freeze_physics_fields_over_time",
    "allow_subs_residual",
    "bounds_loss_kind",
    "bounds_beta",
    "bounds_guard",
    "bounds_w",
    "bounds_include_tau",
    "bounds_tau_w",
    "stop_grad_ref",
    "drawdown_zero_at_origin",
    "physics_warmup_steps",
    "physics_ramp_steps",
)


def _pick_keys(
    d: Dict[str, Any],
    keys: Iterable[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        if k in d:
            out[k] = d.get(k)
    return out


def _head_state(model: Any, name: str) -> Dict[str, Any]:
    layer = getattr(model, name, None)
    if layer is None:
        return {"exists": False}
    return {
        "exists": True,
        "trainable": bool(getattr(layer, "trainable", True)),
        "name": getattr(layer, "name", name),
    }


def ident_audit_dict(
    model: Any,
    *,
    extra_sk_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Small, JSON-safe audit of identifiability configuration.

    Intended for experiment logs / manifests / eval JSON.
    """
    reg = getattr(model, "identifiability_regime", None)
    prof = getattr(model, "_ident_profile", None) or {}
    sk_prof = (prof.get("sk", None) or {})
    sk_now = getattr(model, "scaling_kwargs", None) or {}
    
    enabled = (reg is not None) and bool(prof)

    keys = list(_AUDIT_SK_KEYS)
    if extra_sk_keys is not None:
        for k in extra_sk_keys:
            if k not in keys:
                keys.append(str(k))

    locks = (prof.get("locks", None) or {})
    lock_keys = ("tau_head", "K_head", "Ss_head")

    lam = {
        "lambda_cons": float(getattr(model, "lambda_cons", 0.0)),
        "lambda_gw": float(getattr(model, "lambda_gw", 0.0)),
        "lambda_prior": float(getattr(model, "lambda_prior", 0.0)),
        "lambda_smooth": float(
            getattr(model, "lambda_smooth", 0.0)
        ),
        "lambda_bounds": float(
            getattr(model, "lambda_bounds", 0.0)
        ),
        "lambda_mv": float(getattr(model, "lambda_mv", 0.0)),
        "lambda_q": float(getattr(model, "lambda_q", 0.0)),
    }

    out = {
        "regime": reg,  # JSON-safe: None or str,
        "profile_keys": {
            "sk": sorted(list(sk_prof.keys())),
            "compile": sorted(
                list((prof.get("compile", {}) or {}).keys())
            ),
            "locks": sorted(list(locks.keys())),
        },
        "locks": {k: bool(locks.get(k, False)) for k in lock_keys},
        "heads": {
            "tau_head": _head_state(model, "tau_head"),
            "K_head": _head_state(model, "K_head"),
            "Ss_head": _head_state(model, "Ss_head"),
        },
        "lambdas": lam,
        "sk_from_profile": _pick_keys(sk_prof, keys),
        "sk_effective": _pick_keys(sk_now, keys),
        "bounds_loss": sk_now.get("bounds_loss", None),
        
        "enabled": bool(enabled),
    }
    
    return out

def _norm_regime(regime: str | None) -> str | None:
    if regime is None:
        return None
    s = str(regime).strip().lower()
    if not s:
        return None
    if s in {"off", "none", "disabled"}:
        return None
    return s

def get_ident_profile(
    regime: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    r = _norm_regime(regime)
    if r is None:
        return None, None

    prof = _IDENTIFIABILITY_PROFILES.get(r)
    if prof is None:
        return "base", _IDENTIFIABILITY_PROFILES["base"]
    return r, prof


def _ensure_bounds_loss_dict(sk: dict) -> dict:
    if sk.get("bounds_loss", None) is not None:
        return sk

    kind = sk.get("bounds_loss_kind", "barrier")
    sk["bounds_loss"] = {"kind": str(kind)}
    return sk


def init_identifiability(
    regime: str | None,
    scaling_kwargs: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None, dict]:
    """
    Apply identifiability profile to scaling kwargs.

    - does NOT override user-provided keys
    - ensures sk["bounds_loss"] exists (dict form)
    """
    r, prof = get_ident_profile(regime)

    sk0 = dict(scaling_kwargs or {})

    # Disabled: do not merge defaults.
    if r is None or prof is None:
        return None, None, sk0

    sk1 = _merge_defaults(sk0, prof.get("sk", {}))
    sk1 = _ensure_bounds_loss_dict(sk1)
    return r, prof, sk1

   
def apply_ident_locks(
    model: Any,
    profile: Mapping[str, Any] | None = None,
) -> None:
    prof = profile or getattr(model, "_ident_profile", None)
    if not prof:
        return

    locks = (prof.get("locks") or {})
    if not locks:
        return

    for key in ("tau_head", "K_head", "Ss_head"):
        if not locks.get(key, False):
            continue
        layer = getattr(model, key, None)
        if layer is not None:
            layer.trainable = False


def resolve_compile_weights(
    profile: Optional[Mapping[str, Any]],
    *,
    lambda_cons: Optional[float],
    lambda_gw: Optional[float],
    lambda_prior: Optional[float],
    lambda_smooth: Optional[float],
    lambda_mv: Optional[float],
    lambda_bounds: Optional[float],
    lambda_q: Optional[float],
) -> Dict[str, float]:
    d = (profile or {}).get("compile", {}) or {}

    def _pick(val: Optional[float], k: str, fb: float) -> float:
        if val is not None:
            return float(val)
        if k in d and d[k] is not None:
            return float(d[k])
        return float(fb)

    return {
        "lambda_cons": _pick(lambda_cons, "lambda_cons", 1.0),
        "lambda_gw": _pick(lambda_gw, "lambda_gw", 1.0),
        "lambda_prior": _pick(lambda_prior, "lambda_prior", 1.0),
        "lambda_smooth": _pick(lambda_smooth, "lambda_smooth", 1.0),
        "lambda_mv": _pick(lambda_mv, "lambda_mv", 0.0),
        "lambda_bounds": _pick(lambda_bounds, "lambda_bounds", 0.0),
        "lambda_q": _pick(lambda_q, "lambda_q", 0.0),
    }

def _merge_defaults(dst: dict, src: dict) -> dict:
    out = dict(dst)
    for k, v in (src or {}).items():
        if k not in out or out[k] is None:
            out[k] = v
    return out

def _stop(x: Tensor) -> Tensor:
    return tf_stop_gradient(x)


def _safe_log(x: Tensor, *, eps: float = 1e-12) -> Tensor:
    x = tf_maximum(x, tf_constant(eps, dtype=x.dtype))
    return tf_math.log(x)


def derive_K_from_tau_tf(
    tau_sec: Tensor,
    Ss: Tensor,
    Hd: Tensor,
    kappa_b: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """
    Closure:
      tau = Hd^2 * Ss / (pi^2 * kappa_b * K)
    => K = Hd^2 * Ss / (pi^2 * kappa_b * tau)
    """
    tau_sec = tf_maximum(
        tau_sec,
        tf_constant(eps, dtype=tau_sec.dtype),
    )
    kappa_b = tf_maximum(
        kappa_b,
        tf_constant(eps, dtype=kappa_b.dtype),
    )

    num = (Hd * Hd) * Ss
    pi2 = tf_constant(_PI2, dtype=num.dtype)
    den = pi2 * kappa_b * tau_sec
    return num / den


def derive_K_from_tau_np(
    tau_sec: Any,
    Ss: Any,
    Hd: Any,
    kappa_b: Any,
    *,
    eps: float = 1e-12,
) -> Any:
    tau_sec = np.maximum(np.asarray(tau_sec), eps)
    kappa_b = np.maximum(np.asarray(kappa_b), eps)
    num = (np.asarray(Hd) ** 2) * np.asarray(Ss)
    den = _PI2 * kappa_b * tau_sec
    return num / den


@dataclass(frozen=True)
class ScenarioOut:
    params: Dict[str, Tensor]
    extra_loss: Tensor
    diag: Dict[str, Tensor]


def scenario_tau_only_derive_K(
    params: Mapping[str, Tensor],
    *,
    kappa_b: Tensor,
    tau_key: str = "tau_sec",
    K_key: str = "K_mps",
    Ss_key: str = "Ss",
    Hd_key: str = "Hd",
    Ss_fixed: Optional[Tensor] = None,
    Hd_fixed: Optional[Tensor] = None,
    freeze_Ss: bool = True,
    freeze_Hd: bool = True,
    add_soft_anchor: bool = False,
    anchor_w: float = 0.0,
    Ss_prior: Optional[Tensor] = None,
    Hd_prior: Optional[Tensor] = None,
) -> ScenarioOut:
    """
    Option A:
    - keep tau free
    - freeze/fix Ss and Hd
    - derive K from tau by closure

    Returns:
      ScenarioOut(params, extra_loss, diag)
    """
    p: Dict[str, Tensor] = dict(params)

    tau = p[tau_key]
    Ss = p[Ss_key]
    Hd = p[Hd_key]

    if Ss_fixed is not None:
        Ss = Ss_fixed
    elif freeze_Ss:
        Ss = _stop(Ss)

    if Hd_fixed is not None:
        Hd = Hd_fixed
    elif freeze_Hd:
        Hd = _stop(Hd)

    p[Ss_key] = Ss
    p[Hd_key] = Hd

    K_val = derive_K_from_tau_tf(
        tau_sec=tau,
        Ss=Ss,
        Hd=Hd,
        kappa_b=kappa_b,
    )
    p[K_key] = K_val

    extra = tf_constant(0.0, dtype=tau.dtype)

    if add_soft_anchor and (anchor_w > 0.0):
        w = tf_cast(anchor_w, tau.dtype)

        if (Ss_prior is not None) and (not freeze_Ss):
            extra += w * tf_reduce_mean(
                tf_square(
                    _safe_log(Ss) - _safe_log(Ss_prior)
                )
            )

        if (Hd_prior is not None) and (not freeze_Hd):
            extra += w * tf_reduce_mean(
                tf_square(Hd - Hd_prior)
            )

    diag = {
        "scenario_tau_only": tf_constant(1.0, tau.dtype),
        "K_from_tau_mean": tf_reduce_mean(K_val),
    }
    return ScenarioOut(
        params=p,
        extra_loss=extra,
        diag=diag,
    )

