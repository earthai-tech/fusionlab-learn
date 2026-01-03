
# -*- coding: utf-8 -*-
# _geoprior_maths.py
"""
GeoPrior maths helpers (physics terms + scaling).
Short docs only; full docs later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Sequence, Union 

import numpy as np

from .. import KERAS_DEPS, dependency_message
# from ...utils.generic_utils import vlog 
from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.docs import DocstringComponents, _halnet_core_params
from ._geoprior_utils import coord_ranges, get_sk, get_h_ref_si

# ---------------------------------------------------------------------
# Keras deps aliases (keep short lines for linting)
# ---------------------------------------------------------------------
Tensor = KERAS_DEPS.Tensor
Dataset = KERAS_DEPS.Dataset
GradientTape = KERAS_DEPS.GradientTape
Constraint = KERAS_DEPS.Constraint

tf_float32 = KERAS_DEPS.float32
tf_int32 = KERAS_DEPS.int32
tf_abs = KERAS_DEPS.abs
tf_broadcast_to = KERAS_DEPS.broadcast_to
tf_cast = KERAS_DEPS.cast
tf_concat = KERAS_DEPS.concat
tf_cond = KERAS_DEPS.cond
tf_constant = KERAS_DEPS.constant
tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor
tf_debugging = KERAS_DEPS.debugging
tf_equal = KERAS_DEPS.equal
tf_exp = KERAS_DEPS.exp
tf_expand_dims = KERAS_DEPS.expand_dims
tf_identity = KERAS_DEPS.identity
tf_log = KERAS_DEPS.log
tf_maximum = KERAS_DEPS.maximum
tf_pow = KERAS_DEPS.pow
tf_rank = KERAS_DEPS.rank
tf_reduce_max = KERAS_DEPS.reduce_max
tf_reduce_mean = KERAS_DEPS.reduce_mean
tf_reshape = KERAS_DEPS.reshape
tf_sigmoid = KERAS_DEPS.sigmoid
tf_softplus = KERAS_DEPS.softplus
tf_shape = KERAS_DEPS.shape
tf_sqrt = KERAS_DEPS.sqrt
tf_square = KERAS_DEPS.square
tf_stack = KERAS_DEPS.stack
tf_stop_gradient = KERAS_DEPS.stop_gradient
tf_tile = KERAS_DEPS.tile
tf_transpose = KERAS_DEPS.transpose
tf_scan = KERAS_DEPS.scan
tf_zeros = KERAS_DEPS.zeros
tf_zeros_like = KERAS_DEPS.zeros_like
tf_print =KERAS_DEPS.print 
tf_reduce_min =KERAS_DEPS.reduce_min 
tf_argmin  = KERAS_DEPS.argmin 
tf_reduce_sum = KERAS_DEPS.reduce_sum 
tf_is_nan = KERAS_DEPS.is_nan 
tf_is_inf = KERAS_DEPS.is_inf 
tf_logical_or = KERAS_DEPS.logical_or 
tf_where = KERAS_DEPS.where 
tf_cumsum = KERAS_DEPS.cumsum 
tf_math = KERAS_DEPS.math 
tf_ones_like = KERAS_DEPS.ones_like 
tf_clip_by_value = KERAS_DEPS.clip_by_value
tf_gather = KERAS_DEPS.gather 
tf_minimum = KERAS_DEPS.minimum 
tf_switch_case= KERAS_DEPS.switch_case
tf_logical_and = KERAS_DEPS.logical_and 
tf_greater = KERAS_DEPS.greater 

register_keras_serializable = KERAS_DEPS.register_keras_serializable
deserialize_keras_object = KERAS_DEPS.deserialize_keras_object

tf_autograph = getattr(KERAS_DEPS, "autograph", None)
if tf_autograph is not None:
    tf_autograph.set_verbosity(0)

DEP_MSG = dependency_message("nn.pinn._geoprior_maths")

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_halnet_core_params)
)


_EPSILON = 1e-15

# ---------------------------------------------------------------------
# Time units + scaling
# ---------------------------------------------------------------------
TIME_UNIT_TO_SECONDS = {
    "unitless": 1.0,
    "step": 1.0,
    "index": 1.0,
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "day": 86400.0,
    "days": 86400.0,
    "week": 7.0 * 86400.0,
    "weeks": 7.0 * 86400.0,
    "year": 31556952.0,
    "years": 31556952.0,
    "yr": 31556952.0,
    "month": 31556952.0 / 12.0,
    "months": 31556952.0 / 12.0,
}


AxisLike = Optional[Union[int, Sequence[int]]]

class LogClipConstraint(Constraint):
    """
    NaN-safe clip constraint for log-params.

    Important: clip_by_value(NaN, a, b) -> NaN,
    so we must sanitize NaNs before clipping.
    """

    def __init__(self, min_value, max_value):
        self.min_value = tf_cast(min_value, tf_float32)
        self.max_value = tf_cast(max_value, tf_float32)

    def __call__(self, w):
        w = tf_cast(w, tf_float32)
        w = tf_where(
            tf_math.is_finite(w),
            w,
            self.min_value,
        )
        return tf_clip_by_value(
            w,
            self.min_value,
            self.max_value,
        )
        
def vprint(verbose: int, *args) -> None:
    """Verbose print (eager-friendly)."""
    if int(verbose) > 0:
        tf_print(*args)

def tf_print_nonfinite(tag: str, x: Tensor, summarize: int = 6) -> Tensor:
    """Print a compact report ONLY if x contains NaN/Inf (graph-safe)."""
    x = tf_convert_to_tensor(x, dtype=tf_float32)
    is_nan = tf_is_nan(x)
    is_inf = tf_is_inf(x)
    is_bad = tf_logical_or(is_nan, is_inf)
    n_nan = tf_reduce_sum(tf_cast(is_nan, tf_int32))
    n_inf = tf_reduce_sum(tf_cast(is_inf, tf_int32))
    n_bad = tf_reduce_sum(tf_cast(is_bad, tf_int32))

    def _do_print():
        # safe stats: replace bad values with 0 for min/max/mean
        x_safe = tf_where(is_bad, tf_zeros_like(x), x)
        tf_print(
            "[NONFINITE]", tag,
            "| shape=", tf_shape(x),
            "| n_bad=", n_bad, "n_nan=", n_nan, "n_inf=", n_inf,
            "| min=", tf_reduce_min(x_safe),
            "| max=", tf_reduce_max(x_safe),
            "| mean=", tf_reduce_mean(x_safe),
            summarize=summarize,
        )
        return tf_constant(0, tf_int32)

    return tf_cond(n_bad > 0, _do_print, lambda: tf_constant(0, tf_int32))

# ---------------------------------------------------------------------
# Q-kind support (gw forcing)
# ---------------------------------------------------------------------

def resolve_q_kind(sk: Optional[Dict[str, Any]]) -> str:
    """Normalize Q meaning for gw forcing."""
    if not sk:
        return "per_volume"

    v = get_sk(sk, "Q_kind", "q_kind", "gw_q_kind", default="per_volume")
    mode = str(v).strip().lower()

    if mode in ("pervol", "per_volume", "volumetric", "per_volume_rate"):
        return "per_volume"
    if mode in ("recharge", "recharge_rate", "infiltration", "r"):
        return "recharge_rate"
    if mode in ("head_rate", "dhdt", "head_forcing", "qh"):
        return "head_rate"

    return "per_volume"


def q_to_gw_source_term_si(
    model,
    Q_logits: Tensor,
    *,
    Ss_field: Optional[Tensor],
    H_field: Optional[Tensor],
    coords_normalized: bool,
    t_range_units: Optional[Tensor],
    time_units: Optional[str],
    scaling_kwargs: Optional[Dict[str, Any]],
    H_floor: float = 1e0, #1e-6,
    verbose: int = 0,
) -> Tensor:
    """
    Convert Q_logits into a GW PDE source term with SI units (1/s),
    compatible with:
        R_gw = Ss*dh_dt - div(K grad h) - Q_term

    Modes
    -----
    - per_volume: Q_logits is 1/time_unit (or already 1/s via flags)
    - recharge_rate: Q_logits is m/time_unit; Q_term = (R_per_s / H)
    - head_rate: Q_logits is m/time_unit; Q_term = Ss * qh_per_s
    """
    sk = scaling_kwargs or {}
    kind = resolve_q_kind(sk)

    Q_base = tf_cast(Q_logits, tf_float32)
    Q_base = _apply_q_normalized_time_rule(
        Q_base,
        sk=sk,
        coords_normalized=coords_normalized,
        t_range_units=t_range_units,
    )

    if kind == "per_volume":
        # Backward-compatible flags for volumetric Q:
        Q_in_per_second = bool(get_sk(sk, "Q_in_per_second", default=False))
        Q_in_si = bool(get_sk(sk, "Q_in_si", default=False))
        if Q_in_per_second or Q_in_si:
            Q_per_s = Q_base
        else:
            Q_per_s = rate_to_per_second(Q_base, time_units=time_units)

        vprint(verbose, "Q_kind=per_volume, Q_term(1/s)=", Q_per_s)
        return Q_per_s

    # For the other kinds, interpret Q as a LENGTH RATE (m/time)
    # Use a *separate* flag so we don't conflict with Q_in_si default=True.
    Q_len_in_si = bool(get_sk(sk, "Q_length_in_si", default=False))
    if Q_len_in_si:
        Q_m_per_s = Q_base
    else:
        Q_m_per_s = rate_to_per_second(Q_base, time_units=time_units)

    if kind == "recharge_rate":
        if H_field is None:
            raise ValueError("Q_kind='recharge_rate' requires H_field.")
        H_safe = tf_maximum(
            tf_cast(H_field, tf_float32), tf_constant(H_floor, tf_float32))
        Q_term = Q_m_per_s / H_safe
        vprint(verbose, "Q_kind=recharge_rate, Q_term(1/s)=", Q_term)
        return Q_term

    # kind == "head_rate"
    if Ss_field is None:
        # robust fallback consistent with your consolidation logic:
        Ss_eff = model._mv_value() * model.gamma_w
    else:
        Ss_eff = Ss_field

    Q_term = tf_cast(Ss_eff, tf_float32) * Q_m_per_s
    vprint(verbose, "Q_kind=head_rate, Q_term(1/s)=", Q_term)
    
    return Q_term


def _apply_q_normalized_time_rule(
    Q_base: Tensor,
    *,
    sk: Optional[Dict[str, Any]],
    coords_normalized: bool,
    t_range_units: Optional[Tensor],
) -> Tensor:
    """
    If Q was produced w.r.t normalized time, convert it back to per-time_unit
    by dividing by t_range_units.
    """
    if not sk:
        return Q_base

    Q_wrt_norm_t = bool(get_sk(sk, "Q_wrt_normalized_time", default=False))
    if coords_normalized and Q_wrt_norm_t:
        if t_range_units is None:
            tR, _, _ = coord_ranges(sk)
            if tR is None:
                raise ValueError(
                    "Q_wrt_normalized_time=True but coord_ranges['t'] missing."
                )
            t_range_units = tf_constant(float(tR), tf_float32)
        Q_base = Q_base / (t_range_units + tf_constant(_EPSILON, tf_float32))

    return Q_base

def q_to_per_second(
    Q_base: Tensor,
    *,
    scaling_kwargs: Optional[Dict[str, Any]],
    time_units: Optional[str],
    coords_normalized: bool,
    t_range_units: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> Tensor:
    """
    Normalize Q into 1/s.

    Assumed meaning (recommended default):
      Q_kind = "per_volume"  -> Q is already 1/time_unit or 1/s, representing
                               volumetric source/sink per unit volume.

    If coords_normalized and Q_wrt_normalized_time=True, we de-normalize
    by the time range first (same chain rule as dh/dt).
    """
    sk = scaling_kwargs or {}

    Q = tf_cast(Q_base, tf_float32)

    # If produced w.r.t normalized time, de-normalize by t_range (in time_units)
    if coords_normalized and bool(get_sk(sk, "Q_wrt_normalized_time", default=False)):
        if t_range_units is None:
            tR, _, _ = coord_ranges(sk)
            if tR is None:
                raise ValueError(
                    "Q_wrt_normalized_time=True but coord_ranges['t'] missing."
                )
            t_range_units = tf_constant(float(tR), tf_float32)
        Q = Q / (t_range_units + tf_constant(eps, tf_float32))

    # Interpretation:
    # - If Q_in_per_second=True: Q already 1/s
    # - Else: treat Q as 1/time_units and convert to 1/s
    if bool(get_sk(sk, "Q_in_per_second", default=False)):
        return Q

    # IMPORTANT: I recommend default=False here (safer).
    # Keep your current behavior if you must, but "Q_in_si" is ambiguous.
    if bool(get_sk(sk, "Q_in_si", default=False)):
        return Q

    return rate_to_per_second(Q, time_units=time_units)


def cons_step_to_cons_residual(
    cons_step_m: Tensor,
    *,
    dt_units: Tensor,
    scaling_kwargs: Optional[Dict[str, Any]],
    time_units: Optional[str],
    eps: float = 1e-12,
) -> Tensor:
    """
    Convert consolidation step residual (meters per step) into the chosen
    residual units:
      - "step"      -> meters
      - "time_unit" -> meters / time_unit
      - "second"    -> meters / second (SI rate)
    """
    sk = scaling_kwargs or {}
    mode = resolve_cons_units(sk)

    # dt safety (in time_units, e.g. years)
    dt_min = float(get_sk(sk, "dt_min_units", default=1e-6))
    dt_u = tf_maximum(
        tf_abs(tf_cast(dt_units, tf_float32)),
        tf_constant(dt_min, tf_float32),
    )

    if mode == "step":
        return cons_step_m

    if mode == "time_unit":
        return cons_step_m / dt_u

    # default: seconds
    dt_sec = dt_to_seconds(dt_u, time_units=time_units)
    dt_sec = tf_maximum(dt_sec, tf_constant(eps, tf_float32))
    return cons_step_m / dt_sec

# XXX 
# ---------------------------------------------------------------------
# Physics residuals / priors
# ---------------------------------------------------------------------

def _canon_mv_prior_mode(v) -> str:
    """
    Normalize mv-prior mode string to canonical labels.
    """
    if v is None:
        return "calibrate"

    s = str(v).strip().lower()
    s = s.replace("-", "_")
    
    # ---- explicit off/disable ----
    if s in (
            "off", 
            "none", 
            "disabled", 
            "disable", 
            "false", 
            "0"
        ):
        return "off"

    # Default / detach-style synonyms.
    if s in (
        "default",
        "detach",
        "stopgrad",
        "stop_grad",
        "stop_gradient",
        "calibrate",
        "calibrate_mv",
    ):
        return "calibrate"

    # Fully coupled (can be unstable).
    if s in (
        "field",
        "ss_field",
        "backprop",
        "coupled",
    ):
        return "field"

    # Prefer log-parameterization (safer anchoring).
    if s in (
        "logss",
        "log_ss",
        "logs",
    ):
        return "logss"

    # Unknown: keep user value (but non-empty).
    return s or "calibrate"


def _get_mv_prior_mode(model) -> str:
    """
    Resolve mv-prior mode from scaling kwargs (alias-safe).

    Notes
    -----
    We try top-level keys first, then `bounds` fallback.
    """
    sk = getattr(model, "scaling_kwargs", None) or {}

    # 1) Top-level scaling kwargs (alias-safe).
    v = get_sk(sk, "mv_prior_mode", default=None)

    # 2) Nested bounds fallback (common pattern in this codebase).
    if v is None:
        b = sk.get("bounds", None) or {}
        v = get_sk(b, "mv_prior_mode", default=None)

    return _canon_mv_prior_mode(v)

def _resolve_mv_prior_weight(
    model,
    *,
    weight=None,
    warmup_steps=None,
    step=None,
    dtype=tf_float32,
) -> Optional[Tensor]:
    """
    Resolve mv-prior weight with delay + warmup.

    Keys
    ----
    mv_schedule_unit: "epoch" or "step"
    mv_delay_epochs,  mv_warmup_epochs
    mv_delay_steps,   mv_warmup_steps
    mv_steps_per_epoch (epoch->step)
    """
    sk = getattr(model, "scaling_kwargs", None) or {}
    b = sk.get("bounds", None) or {}

    # ----------------------------
    # Base weight.
    # ----------------------------
    if weight is None:
        weight = get_sk(sk, "mv_weight", default=None)
    if weight is None:
        weight = get_sk(b, "mv_weight", default=None)
    if weight is None:
        return None

    w = tf_constant(float(weight), dtype)
    w = _finite_or_zero(w)

    # No step => constant weight.
    if step is None:
        return w

    # ----------------------------
    # Schedule unit.
    # ----------------------------
    unit = get_sk(sk, "mv_schedule_unit", default=None)
    if unit is None:
        unit = get_sk(b, "mv_schedule_unit", default=None)

    if unit is None:
        unit = "step" if warmup_steps is not None else "epoch"

    unit = str(unit).strip().lower()
    if unit not in ("epoch", "step"):
        unit = "step"

    # ----------------------------
    # Epoch params.
    # ----------------------------
    de = get_sk(sk, "mv_delay_epochs", default=None)
    if de is None:
        de = get_sk(b, "mv_delay_epochs", default=None)

    we = get_sk(sk, "mv_warmup_epochs", default=None)
    if we is None:
        we = get_sk(b, "mv_warmup_epochs", default=None)

    # ----------------------------
    # Step params.
    # ----------------------------
    ds = get_sk(sk, "mv_delay_steps", default=None)
    if ds is None:
        ds = get_sk(b, "mv_delay_steps", default=None)

    if warmup_steps is None:
        ws = get_sk(sk, "mv_warmup_steps", default=None)
        if ws is None:
            ws = get_sk(b, "mv_warmup_steps", default=None)
    else:
        ws = warmup_steps

    spe = get_sk(sk, "mv_steps_per_epoch", default=None)
    if spe is None:
        spe = get_sk(b, "mv_steps_per_epoch", default=None)

    # ----------------------------
    # Convert to ints.
    # ----------------------------
    def _to_int(v):
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    delay_s = _to_int(ds)
    warm_s = _to_int(ws)

    # Epoch -> step conversion.
    if unit == "epoch":
        spe_i = _to_int(spe)
        if spe_i is not None and spe_i > 0:
            if delay_s is None:
                de_i = _to_int(de) or 0
                delay_s = max(0, de_i) * spe_i
            if warm_s is None:
                we_i = _to_int(we) or 0
                warm_s = max(0, we_i) * spe_i

    if delay_s is None:
        delay_s = 0

    # ----------------------------
    # Ramp with delay + warmup.
    # ----------------------------
    s = tf_cast(step, dtype)
    s = _finite_or_zero(s)

    d = tf_constant(float(delay_s), dtype)
    d = _finite_or_zero(d)

    # Hard gate if warmup missing/0.
    if (warm_s is None) or (warm_s <= 0):
        one = tf_constant(1.0, dtype)
        zero = tf_constant(0.0, dtype)
        ramp = tf_where(s >= d, one, zero)
        return w * ramp

    wu = tf_constant(float(max(1, warm_s)), dtype)
    wu = _finite_or_zero(wu)

    ramp = tf_clip_by_value((s - d) / wu, 0.0, 1.0)
    ramp = _finite_or_zero(ramp)

    return w * ramp


def resolve_mv_gamma_log_target_from_logSs(
    model,
    logSs,
    *,
    eps=_EPSILON,
    verbose=0,
) -> Tensor:
    """
    Like resolve_mv_gamma_log_target(), but uses logSs.

    This is the preferred path for mode='logss' because it
    avoids the 1/Ss gradient amplification from log(Ss_field).
    """
    mv_units = _get_mv_prior_units(model)

    log_mv = _safe_log_mv(model, eps=eps)
    log_gw = _safe_log_gw(model, eps=eps)

    # Strict path: smooth and stable.
    if mv_units != "auto":
        log_target = log_mv + log_gw
        vprint(verbose, "mv_prior_units:", mv_units)
        vprint(verbose, "log_target(strict):", log_target)
        return log_target

    # Auto path: choose 1e3 convention by matching mean(logSs).
    logSs = tf_cast(logSs, tf_float32)

    eps_t = tf_constant(float(eps), tf_float32)
    log_eps = tf_log(eps_t)

    logSs = tf_where(tf_math.is_finite(logSs), logSs, log_eps)
    logSs_mean = tf_reduce_mean(logSs)

    log1000 = tf_log(tf_constant(1000.0, tf_float32))

    log_mv_c = tf_stack(
        [log_mv, log_mv - log1000, log_mv, log_mv - log1000],
    )
    log_gw_c = tf_stack(
        [log_gw, log_gw, log_gw + log1000, log_gw + log1000],
    )

    log_targets = log_mv_c + log_gw_c
    errs = tf_abs(logSs_mean - log_targets)

    idx = tf_cast(
        tf_argmin(tf_stop_gradient(errs), axis=0),
        tf_int32,
    )
    log_target = tf_gather(log_targets, idx)

    vprint(verbose, "mv_prior_units:", mv_units)
    vprint(verbose, "mv/gw idx:", idx)
    vprint(verbose, "log_target(auto):", log_target)

    return log_target

def _mv_prior_disabled_return(
    *,
    as_loss: bool,
    Ss_field: Optional[Tensor],
    logSs: Optional[Tensor],
    dtype=tf_float32,
) -> Tensor:
    """
    Return zeros for disabled mv-prior.

    - as_loss=True  -> scalar 0
    - as_loss=False -> zeros_like(logSs or Ss_field)
    """
    if bool(as_loss):
        return tf_constant(0.0, dtype)

    ref = logSs if (logSs is not None) else Ss_field
    if ref is None:
        return tf_constant(0.0, dtype)

    ref = tf_cast(ref, dtype)
    return tf_zeros_like(ref)


def _mv_prior_is_disabled(model, *, mode: str) -> bool:
    """
    True if mv-prior should be skipped.
    """
    if mode == "off":
        return True

    lam = float(getattr(model, "lambda_mv", 0.0))
    return lam <= 0.0

def compute_mv_prior(
    model,
    Ss_field: Optional[Tensor] = None,
    *,
    logSs: Optional[Tensor] = None,
    mode: Optional[str] = None,
    as_loss: bool = True,
    weight=None,
    warmup_steps=None,
    step=None,
    alpha_disp=0.1,
    delta=1.0,
    eps=_EPSILON,
    verbose=0,
):
    """
    MV prior with 3 modes (default: calibrate).

    Residual
    --------
    r = log(Ss) - log(m_v * gamma_w)

    Modes
    -----
    calibrate (default)
        Uses stop_gradient(Ss_field). This calibrates mv without
        reshaping Ss_field. Most stable for physics-driven mean
        subsidence.

    field
        Backprop through Ss_field. Can be unstable due to
        d log(Ss)/dSs = 1/Ss amplification.

    logss
        Backprop through logSs (recommended if you need stronger
        Ss anchoring). Pass logSs from compose_physics_fields().

    Returns
    -------
    If as_loss=True:
        Scalar loss (Huber on mean + dispersion term).
    If as_loss=False:
        Residual field r (same shape as logSs / derived logSs).
    """
    # ----------------------------------------------------------
    # 1) Resolve mode (alias-safe via scaling_kwargs).
    # ----------------------------------------------------------
    if mode is None:
        mode = _get_mv_prior_mode(model)
    mode = _canon_mv_prior_mode(mode)

    # ----------------------------
    # 1b) Off / disabled gate.
    # ----------------------------
    if _mv_prior_is_disabled(model, mode=mode):
        return _mv_prior_disabled_return(
            as_loss=as_loss,
            Ss_field=Ss_field,
            logSs=logSs,
        )
    
    # ----------------------------------------------------------
    # 2) Build log-space residual r.
    # ----------------------------------------------------------
    if mode == "logss":
        if logSs is None:
            raise ValueError(
                "mode='logss' requires `logSs` from "
                "compose_physics_fields().",
            )

        logSs_ = tf_cast(logSs, tf_float32)
        log_target = resolve_mv_gamma_log_target_from_logSs(
            model,
            logSs_,
            eps=eps,
            verbose=verbose,
        )
        r = logSs_ - log_target

    else:
        if Ss_field is None:
            raise ValueError(
                "compute_mv_prior requires Ss_field "
                "for mode != 'logss'.",
            )

        Ss_in = Ss_field

        # Default: detach Ss to avoid trunk destabilization.
        if mode == "calibrate":
            Ss_in = tf_stop_gradient(Ss_field)

        logSs_ = safe_log_pos(Ss_in, eps=eps)
        log_target = resolve_mv_gamma_log_target(
            model,
            Ss_in,
            eps=eps,
            verbose=verbose,
        )
        r = logSs_ - log_target

    # Return residual if requested (diagnostics use-case).
    if not bool(as_loss):
        return r

    # ----------------------------------------------------------
    # 3) Scalar loss: global mismatch + dispersion penalty.
    # ----------------------------------------------------------
    
    r_bar = tf_reduce_mean(r)
    loss_g = huber(r_bar, delta=delta)

    loss_d = tf_reduce_mean(huber(r - r_bar, delta=delta))
    a = tf_constant(float(alpha_disp), tf_float32)

    loss = loss_g + a * loss_d

    # ----------------------------------------------------------
    # 4) Optional independent weight + warmup ramp.
    # ----------------------------------------------------------
    w = _resolve_mv_prior_weight(
        model,
        weight=weight,
        warmup_steps=warmup_steps,
        step=step,
    )
    if w is not None:
        loss = loss * w

    return loss


def _get_mv_prior_units(model) -> str:
    """
    Get mv prior units mode from scaling kwargs.

    Expected values:
    - "auto"   : choose best 1e3 convention
    - "strict" : use log(mv) + log(gamma_w)
    """
    sk = getattr(model, "scaling_kwargs", None) or {}

    # Allow either top-level or nested placement.
    v = sk.get("mv_prior_units", None)

    if v is None:
        b = sk.get("bounds", None) or {}
        v = b.get("mv_prior_units", None)

    if v is None:
        return "strict"

    return str(v).strip().lower()

def _safe_log_mv(model, *, eps=_EPSILON) -> Tensor:
    """
    Return log(mv) safely.

    - If mv is learnable: use model.log_mv (log-space).
    - If mv is fixed: log(mv_fixed) in a safe way.
    - If missing/None: return log(eps).
    """
    eps_t = tf_constant(float(eps), tf_float32)
    log_eps = tf_log(eps_t)

    log_mv_raw = getattr(model, "log_mv", None)
    if log_mv_raw is not None:
        log_mv = tf_cast(log_mv_raw, tf_float32)
        return tf_where(
            tf_math.is_finite(log_mv),
            log_mv,
            log_eps,
        )

    mv = getattr(model, "_mv_fixed", None)
    if mv is None:
        return log_eps

    return safe_log_pos(mv, eps=eps)


def _safe_log_gw(model, *, eps=_EPSILON) -> Tensor:
    """
    Return log(gamma_w) safely.

    Uses a constant fallback if gamma_w is missing/None.
    """
    gw = getattr(model, "gamma_w", None)
    if gw is None:
        gw = tf_constant(9810.0, tf_float32)

    return safe_log_pos(gw, eps=eps)


def resolve_mv_gamma_log_target(
    model,
    Ss_field,
    *,
    eps=_EPSILON,
    verbose=0,
) -> Tensor:
    """
    Return log(mv * gamma_w) with configurable units.

    If mv_prior_units == "strict":
        log_target = log(mv) + log(gamma_w)

    If mv_prior_units == "auto":
        pick among 4 candidates that best matches
        mean(log(Ss_field)) in magnitude:
        - mv      vs mv/1000
        - gamma_w vs gamma_w*1000
    """
    mv_units = _get_mv_prior_units(model)

    log_mv = _safe_log_mv(model, eps=eps)
    log_gw = _safe_log_gw(model, eps=eps)

    # Strict path: no argmin, no discrete switches.
    if mv_units != "auto":
        log_target = log_mv + log_gw
        vprint(verbose, "mv_prior_units:", mv_units)
        vprint(verbose, "log_target(strict):", log_target)
        return log_target

    # Auto path: use Ss only for scale matching.
    logSs_mean = tf_reduce_mean(
        safe_log_pos(Ss_field, eps=eps),
    )

    log1000 = tf_log(tf_constant(1000.0, tf_float32))

    # 4 candidates (log-space).
    log_mv_c = tf_stack(
        [log_mv, log_mv - log1000, log_mv, log_mv - log1000],
    )
    log_gw_c = tf_stack(
        [log_gw, log_gw, log_gw + log1000, log_gw + log1000],
    )

    log_targets = log_mv_c + log_gw_c
    errs = tf_abs(logSs_mean - log_targets)

    # Discrete choice only; do not backprop it.
    idx = tf_cast(
        tf_argmin(tf_stop_gradient(errs), axis=0),
        tf_int32,
    )

    log_target = tf_gather(log_targets, idx)

    vprint(verbose, "mv_prior_units:", mv_units)
    vprint(verbose, "mv/gw idx:", idx)
    vprint(verbose, "log_target(auto):", log_target)

    return log_target

# -----------------------------
# Reusable numeric helpers
# -----------------------------
def safe_pos(x, *, eps=_EPSILON, dtype=tf_float32):
    """
    Force x to be finite and >= eps.

    Replaces NaN/Inf by eps, then floors.
    """
    eps_t = tf_constant(float(eps), dtype)
    x = tf_cast(x, dtype)
    x = tf_where(tf_math.is_finite(x), x, eps_t)
    x = tf_clip_by_value(x, eps_t, tf_constant(1e30, dtype))
    return tf_maximum(x, eps_t)


def safe_log_pos(x, *, eps=_EPSILON, dtype=tf_float32):
    """log(safe_pos(x))."""
    return tf_log(safe_pos(x, eps=eps, dtype=dtype))


def huber(x, *, delta=1.0):
    """
    Huber loss (elementwise).

    delta is treated as a scalar constant.
    """
    d = tf_constant(float(delta), x.dtype)
    ax = tf_abs(x)
    quad = tf_minimum(ax, d)
    lin = ax - quad
    return 0.5 * tf_square(quad) + d * lin


def compute_gw_flow_residual(
    model,
    dh_dt: Tensor,
    d_K_dh_dx_dx: Tensor,
    d_K_dh_dy_dy: Tensor,
    Ss_field: Tensor,
    *,
    Q: Optional[Tensor] = None,
    verbose: int = 0,
) -> Tensor:
    """Groundwater flow PDE residual (NaN/Inf-safe, broadcast-safe)."""
    if "gw_flow" not in model.pde_modes_active:
        return tf_zeros_like(dh_dt)

    # --- convert + sanitize core terms ---
    dh_dt = _finite_or_zero(tf_convert_to_tensor(dh_dt, dtype=tf_float32))
    d_K_dh_dx_dx = _finite_or_zero(tf_convert_to_tensor(d_K_dh_dx_dx, dtype=dh_dt.dtype))
    d_K_dh_dy_dy = _finite_or_zero(tf_convert_to_tensor(d_K_dh_dy_dy, dtype=dh_dt.dtype))
    Ss_field = _finite_or_zero(tf_convert_to_tensor(Ss_field, dtype=dh_dt.dtype))
    
    # --- Q: scalar / (H,) / (B,H) / (B,H,1) -> (B,H,1) ---
    if Q is None:
        Qv = tf_zeros_like(dh_dt)
    else:
        Qv = tf_convert_to_tensor(Q, dtype=dh_dt.dtype)
        Qv = ensure_3d(Qv)                         # scalar->(1,1,1), (H,)->(1,H,1), (B,H)->(B,H,1)
        Qv = tf_broadcast_to(Qv, tf_shape(dh_dt))  # now broadcast is valid
        Qv = _finite_or_zero(Qv)

    div_K_grad_h = d_K_dh_dx_dx + d_K_dh_dy_dy
    storage_term = Ss_field * dh_dt

    out = storage_term - div_K_grad_h - Qv
    out = _finite_or_zero(out)  # optional, but makes the "output finite" contract explicit

    if verbose > 6:
        vprint(verbose, "gw: dh_dt=", dh_dt)
        vprint(verbose, "gw: div=", div_K_grad_h)
        vprint(verbose, "gw: Q=", Qv)
        vprint(verbose, "gw: out=", out)
    
        tf_print( 
            "to_rms(Ss_field * dh_dt)=", 
            to_rms(Ss_field * dh_dt),
        
            'to_rms(div_K_grad_h)=',
            to_rms(div_K_grad_h),
            
            'to_rms(Qv)=',
            to_rms(Qv),
            
            'to_rms(out)=',
            to_rms(out),
            )

    return out


def compute_consolidation_residual(
    model,
    ds_dt: Tensor,
    s_state: Tensor,
    h_mean: Tensor,
    H_field: Tensor,
    tau_field: Tensor,
    *,
    Ss_field: Optional[Tensor] = None,
    inputs: Optional[Dict[str, Tensor]] = None,
    verbose: int = 0,
) -> Tensor:
    """Consolidation PDE residual (Voigt)."""
    
    if "consolidation" not in model.pde_modes_active:
        return tf_zeros_like(ds_dt)

    eps = tf_constant(_EPSILON, dtype=tf_float32)
    tau_safe = tf_maximum(tau_field, eps)

    h_ref_si = get_h_ref_si(model, inputs, like=h_mean)
    delta_h = tf_maximum(h_ref_si - h_mean, 0.0)

    if Ss_field is None:
        Ss_eff = model._mv_value() * model.gamma_w
        src = "mv*gw"
    else:
        Ss_eff = Ss_field
        src = "Ss_field"

    s_eq = Ss_eff * delta_h * H_field
    relaxation = (s_eq - s_state) / tau_safe
    out = ds_dt - relaxation

    vprint(verbose, "cons: h_ref=", h_ref_si)
    vprint(verbose, "cons: delta_h=", delta_h)
    vprint(verbose, "cons: Ss_eff(", src, ")=", Ss_eff)
    vprint(verbose, "cons: s_eq=", s_eq)
    vprint(verbose, "cons: s_state=", s_state)
    vprint(verbose, "cons: relax=", relaxation)
    vprint(verbose, "cons: out=", out)

    return out


def _positive_part(
    x: Tensor,
    *,
    mode: str = "smooth_relu",
    beta: float = 20.0,
    eps: float = _EPSILON,
    zero_at_origin: bool = False,
) -> Tensor:
    """Return the non-negative part of x, with selectable smoothness.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    mode : {'smooth_relu', 'relu', 'softplus', 'none'}
        - 'smooth_relu': softplus(beta*x)/beta  (smooth ReLU approx)
        - 'relu'       : max(x, 0)
        - 'softplus'   : softplus(x)            (always > 0)
        - 'none'       : x (no clamping)
    beta : float
        Curvature control for 'smooth_relu'. Larger -> closer to ReLU.
    eps : float
        Small additive floor after gating (usually 0).
    zero_at_origin : bool
        If True and mode == 'smooth_relu', shift so that output is
        (approximately) 0 at x=0:
            softplus(beta*x)/beta - log(2)/beta
        Note: this shifted version can become slightly negative for x<0.
        If you need strict non-negativity, keep this False.

    Returns
    -------
    Tensor
        Gated tensor.
    """
    mode = str(mode).strip().lower()

    x = tf_cast(x, tf_float32)
    x = _finite_or_zero(x)  

    if mode == "none":
        y = x

    elif mode == "relu":
        y = tf_maximum(x, tf_constant(eps, dtype=x.dtype))

    elif mode == "softplus":
        y = positive(x, eps=eps)

    elif mode == "smooth_relu":
        b = tf_constant(float(beta), dtype=x.dtype)
        y = tf_softplus(b * x) / b
        if bool(zero_at_origin):
            log2 = tf_constant(float(np.log(2.0)), dtype=x.dtype)
            y = y - (log2 / b)

    else:
        raise ValueError(
            "_positive_part: mode must be one of "
            "{'smooth_relu','relu','softplus','none'}."
        )

    if eps and float(eps) > 0.0:
        y = y + tf_constant(float(eps), dtype=y.dtype)

    return y

def equilibrium_compaction_si(
    *,
    h_mean_si: Tensor,
    h_ref_si: Tensor,
    Ss_field: Tensor,
    H_field_si: Tensor,
    drawdown_mode: str = "smooth_relu",
    drawdown_rule: str = "ref_minus_mean",
    relu_beta: float = 20.0,
    stop_grad_ref: bool = True,
    drawdown_zero_at_origin: bool = False,
    drawdown_clip_max: Optional[float] = None,
    eps: float = _EPSILON,
    verbose: int = 0,
) -> Tensor:
    """Compute equilibrium compaction s_eq (SI meters).

    Notes
    -----
    - The physically common convention for drawdown (head loss) is:
        delta_h = h_ref - h_mean
      which is `drawdown_rule='ref_minus_mean'`.

    - If you accidentally pass *depth* (down-positive) instead of head,
      drawdown often needs to be flipped:
        delta_h = h_mean - h_ref
      which is `drawdown_rule='mean_minus_ref'`.

    - `stop_grad_ref=True` is strongly recommended: it prevents the model
      from collapsing delta_h by moving the reference.

    Parameters
    ----------
    h_mean_si : Tensor
        Mean head (or depth, depending on your pipeline), SI meters.
        Shape (B,H,1) or broadcastable.
    h_ref_si : Tensor
        Reference head (or depth), SI meters, broadcastable to h_mean_si.
    Ss_field : Tensor
        Specific storage field in 1/m, broadcastable.
    H_field_si : Tensor
        Compressible thickness in meters, broadcastable.
    drawdown_mode : str
        Gating for positive drawdown: 'smooth_relu' (default), 'relu',
        'softplus', or 'none'.
    drawdown_rule : str
        'ref_minus_mean' (default) or 'mean_minus_ref'.
    relu_beta : float
        Smoothness parameter for 'smooth_relu'.
    stop_grad_ref : bool
        If True, uses stop_gradient on h_ref_si.
    drawdown_zero_at_origin : bool
        If True and drawdown_mode='smooth_relu', shift the smooth ReLU so
        it is ~0 at x=0 (see _positive_part doc).
    drawdown_clip_max : float, optional
        If provided, clips delta_h to [0, drawdown_clip_max] after gating.
    verbose : int
        Verbosity level.

    Returns
    -------
    Tensor
        Equilibrium compaction s_eq in meters, shape (B,H,1).
    """

    h_mean_si = _ensure_3d(tf_cast(h_mean_si, tf_float32))
    h_ref_si = _broadcast_like(_ensure_3d(tf_cast(h_ref_si, tf_float32)), h_mean_si)
    Ss_field = _broadcast_like(_ensure_3d(Ss_field), h_mean_si)
    H_field_si = _broadcast_like(_ensure_3d(H_field_si), h_mean_si)

    def _n_bad(x: Tensor) -> Tensor:
        return tf_reduce_sum(tf_cast(~tf_math.is_finite(x), tf_int32))

    # --- debug counts BEFORE sanitization
    vprint(
        verbose,
        "[equilibrium_compaction_si] nonfinite counts (pre):",
        "h_mean", _n_bad(h_mean_si),
        "h_ref", _n_bad(h_ref_si),
        "Ss", _n_bad(Ss_field),
        "H", _n_bad(H_field_si),
    )

    # --- sanitize ALL inputs (this is what makes your tests pass)
    h_mean_si = _finite_or_zero(h_mean_si)
    h_ref_si = _finite_or_zero(h_ref_si)
    Ss_field = _finite_or_zero(Ss_field)
    H_field_si = _finite_or_zero(H_field_si)

    # Optional: enforce non-negativity for physical fields
    # (keeps math stable if something goes negative)
    zero = tf_constant(0.0, dtype=tf_float32)
    Ss_field = tf_maximum(Ss_field, zero)
    H_field_si = tf_maximum(H_field_si, zero)

    # --- debug counts AFTER sanitization
    vprint(
        verbose,
        "[equilibrium_compaction_si] nonfinite counts (post):",
        "h_mean", _n_bad(h_mean_si),
        "h_ref", _n_bad(h_ref_si),
        "Ss", _n_bad(Ss_field),
        "H", _n_bad(H_field_si),
    )

    if bool(stop_grad_ref):
        h_ref_si = tf_stop_gradient(h_ref_si)

    vprint(
        verbose,
        "[equilibrium_compaction_si] shapes:",
        "h_mean", h_mean_si.shape,
        "h_ref", h_ref_si.shape,
        "Ss", Ss_field.shape,
        "H", H_field_si.shape,
        "| mode=", drawdown_mode,
        "| rule=", drawdown_rule,
        "| stop_grad_ref=", stop_grad_ref,
    )

    rule = str(drawdown_rule).strip().lower()
    if rule in {"ref_minus_mean", "ref-mean", "ref_mean"}:
        delta_raw = h_ref_si - h_mean_si
    elif rule in {"mean_minus_ref", "mean-ref", "mean_ref"}:
        delta_raw = h_mean_si - h_ref_si
    else:
        raise ValueError(
            "equilibrium_compaction_si: drawdown_rule must be "
            "'ref_minus_mean' or 'mean_minus_ref'."
        )

    delta_h = _positive_part(
        delta_raw,
        mode=drawdown_mode,
        beta=relu_beta,
        eps=eps,
        zero_at_origin=bool(drawdown_zero_at_origin),
    )

    if drawdown_clip_max is not None:
        mx = tf_constant(float(drawdown_clip_max), dtype=delta_h.dtype)
        delta_h = tf_clip_by_value(
            delta_h, tf_constant(eps, dtype=delta_h.dtype), mx)

    vprint(
        verbose,
        "[equilibrium_compaction_si] delta_h stats:",
        "min=", tf_reduce_min(delta_h),
        "max=", tf_reduce_max(delta_h),
        "mean=", tf_reduce_mean(delta_h),
    )

    s_eq = Ss_field * delta_h * H_field_si
    s_eq = _finite_or_zero(s_eq)  # extra safety net

    vprint(
        verbose,
        "[equilibrium_compaction_si] s_eq stats:",
        "min=", tf_reduce_min(s_eq),
        "max=", tf_reduce_max(s_eq),
        "mean=", tf_reduce_mean(s_eq),
        "| nonfinite=", _n_bad(s_eq),
    )
    return s_eq

def integrate_consolidation_mean(
    *,
    h_mean_si: Tensor,
    Ss_field: Tensor,
    H_field_si: Tensor,
    tau_field: Tensor,
    h_ref_si: Tensor,
    s_init_si: Tensor,
    dt: Optional[Tensor] = None,
    time_units: Optional[str] = "yr",
    method: str = "exact",
    eps_tau: float = 1e-12,
    relu_beta: float = 20.0,
    drawdown_mode: str = "smooth_relu",
    drawdown_rule: str = "ref_minus_mean",
    stop_grad_ref: bool = True,
    drawdown_zero_at_origin: bool = False,
    drawdown_clip_max: Optional[float] = None,
    verbose: int = 0,
) -> Tensor:

    """Integrate mean settlement \bar{s}(t) using a stable stepper.

    Parameters
    ----------
    h_mean_si : Tensor
        Mean head in SI meters, shape (B,H,1) (or (B,H)).
    Ss_field, H_field_si, tau_field : Tensor
        Effective fields in SI (Ss in 1/m, H in m, tau in seconds).
        May be broadcastable to (B,H,1).
    h_ref_si : Tensor
        Reference head for drawdown (broadcastable).
    s_init_si : Tensor
        Initial cumulative settlement (SI meters), shape (B,1,1) or (B,1).
    dt : Tensor, optional
        Time step in `time_units`, broadcastable to (B,H,1). If None,
        defaults to 1 per step.
    time_units : str, optional
        Units of `dt` (converted to seconds).
    method : {'exact','euler'}
        Stepping scheme.

    Returns
    -------
    s_bar_si : Tensor
        Mean cumulative settlement over the horizon, shape (B,H,1).
    """
    def _align_to_horizon(x: Tensor, *, name: str) -> Tensor:
        """Align x time-length to horizon H (or keep length 1)."""
        xt = _ensure_3d(tf_cast(x, tf_float32))
        tx = tf_shape(xt)[1]
    
        # If provided as state-length (H+1), slice to horizon H.
        xt = tf_cond(
            tf_equal(tx, H + 1),
            lambda: xt[:, :-1, :],
            lambda: xt,
        )
    
        # If provided as step-length (H-1), pad to horizon H by
        # repeating the first step (consistent with dt inference).
        tx2 = tf_shape(xt)[1]
    
        def _pad_prepend() -> Tensor:
            first = xt[:, :1, :]
            return tf_concat([first, xt], axis=1)
    
        xt = tf_cond(
            tf_logical_and(
                tf_greater(H, 1),
                tf_equal(tx2, H - 1),
            ),
            _pad_prepend,
            lambda: xt,
        )
    
        # Now must be length H or 1.
        tx3 = tf_shape(xt)[1]
        ok = tf_logical_or(tf_equal(tx3, H), tf_equal(tx3, 1))
        tf_debugging.assert_equal(
            ok,
            True,
            message=(
                f"{name} has incompatible time length; "
                "expected H, H-1, H+1, or 1."
            ),
        )
        return xt

    h_mean_si = _ensure_3d(tf_cast(h_mean_si, tf_float32))

    # ----------------------------------------------------------
    # Force a strict (B,H,1) shape (static last dim = 1).
    # This prevents tf.scan from widening shapes to (None,None).
    # ----------------------------------------------------------
    shp = tf_shape(h_mean_si)
    B = shp[0]
    H = shp[1]
    h_mean_si = tf_reshape(h_mean_si, [B, H, 1])

    vprint(
        verbose,
        "[integrate_consolidation_mean] B,H =",
        B, H,
        "| time_units=", time_units,
        "| method=", method,
    )

    # --- dt in seconds (BH1) -----------------------------------
    if dt is None:
        dt = tf_ones_like(h_mean_si)
        vprint(
            verbose,
            "[integrate_consolidation_mean] dt=None -> 1",
        )
    else:
        dt_in = _align_to_horizon(dt, name="dt")
        dt = _broadcast_like(dt_in, h_mean_si)
        
    dt = tf_reshape(dt, [B, H, 1])
    
    # sanitize dt before converting
    dt = _finite_or_zero(dt)
    # Optional: disallow negative time steps
    dt = tf_maximum(dt, tf_constant(0.0, dtype=dt.dtype))

    dt_sec = dt_to_seconds(dt, time_units=time_units)
    dt_sec = tf_reshape(dt_sec, [B, H, 1])
    
    # sanitize dt_sec too (unit conversion could create non-finite)
    dt_sec = _finite_or_zero(dt_sec)
    dt_sec = tf_maximum(dt_sec, tf_constant(0.0, dtype=dt_sec.dtype))

    vprint(
        verbose,
        "[integrate_consolidation_mean] dt_sec stats:",
        "min=", tf_reduce_min(dt_sec),
        "max=", tf_reduce_max(dt_sec),
        "mean=", tf_reduce_mean(dt_sec),
    )

    # --- tau (BH1) ---------------------------------------------
    tau_in = _align_to_horizon(tau_field, name="tau_field")
    tau = _broadcast_like(tau_in, h_mean_si)
    tau = tf_reshape(tau, [B, H, 1])

    tf_debugging.assert_equal(
        tf_shape(tau)[1], H,
        message=( 
            "integrate_consolidation_mean:"
            " tau horizon must match h_mean_si horizon"
            )
    )

    # sanitize tau BEFORE clamping
    tau = _finite_or_zero(tau)
    
    tau = tf_maximum(
        tau,
        tf_constant(eps_tau, dtype=tf_float32),
    )

    vprint(
        verbose,
        "[integrate_consolidation_mean] tau stats:",
        "min=", tf_reduce_min(tau),
        "max=", tf_reduce_max(tau),
        "mean=", tf_reduce_mean(tau),
    )

    # --- equilibrium compaction (BH1) --------------------------
    s_eq = equilibrium_compaction_si(
        h_mean_si=h_mean_si,
        h_ref_si=h_ref_si,
        Ss_field=Ss_field,
        H_field_si=H_field_si,
        # NEW forwarding:
        drawdown_mode=drawdown_mode,
        drawdown_rule=drawdown_rule,
        stop_grad_ref=stop_grad_ref,
        drawdown_zero_at_origin=drawdown_zero_at_origin,
        drawdown_clip_max=drawdown_clip_max,
        relu_beta=relu_beta,
        verbose=verbose,
    )
    s_eq = tf_reshape(s_eq, [B, H, 1])

    method = str(method).strip().lower()
    if method not in {"exact", "euler"}:
        raise ValueError(
            "integrate_consolidation_mean: "
            "method must be 'exact' or 'euler'."
        )

    # --- initializer (B,1) -------------------------------------
    # s0 = _ensure_3d(tf_cast(s_init_si, tf_float32))
    # s0 = s0[:, :1, :1]
    # s0 = tf_reshape(s0, [B, 1, 1])
    # s0 = _finite_or_zero(s0)
    
    # s0_2d = tf_reshape(s0[:, 0, :], [B, 1])
    
    s0 = _ensure_3d(tf_cast(s_init_si, tf_float32))
    s0 = s0[:, :1, :1]
    
    # broadcast to (B,1,1) using the same mechanism as dt/tau
    s0 = _broadcast_like(s0, h_mean_si[:, :1, :1])
    s0 = tf_reshape(s0, [B, 1, 1])
    
    s0 = _finite_or_zero(s0)
    s0_2d = tf_reshape(s0[:, 0, :], [B, 1])

    vprint(
        verbose,
        "[integrate_consolidation_mean] s_init stats:",
        "min=", tf_reduce_min(s0_2d),
        "max=", tf_reduce_max(s0_2d),
        "mean=", tf_reduce_mean(s0_2d),
    )

    if tf_transpose is None or tf_scan is None:
        raise RuntimeError(
            "TensorFlow ops 'transpose'/'scan' missing "
            "from KERAS_DEPS."
        )

    # time-major: (H,B,1)
    dt_tm = tf_transpose(dt_sec, [1, 0, 2])
    tau_tm = tf_transpose(tau, [1, 0, 2])
    seq_tm = tf_transpose(s_eq, [1, 0, 2])

    def step(
        prev: Tensor,
        elems: Tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        dt_i, tau_i, seq_i = elems

        # Force (B,1) each iteration (prevents widening).
        shp_prev = tf_shape(prev)
        dt_i = tf_reshape(dt_i, shp_prev)
        tau_i = tf_reshape(tau_i, shp_prev)
        seq_i = tf_reshape(seq_i, shp_prev)

        if method == "exact":
            a = tf_exp(
                -dt_i / (
                    tau_i
                    + tf_constant(_EPSILON, tau_i.dtype)
                )
            )
            nxt = prev * a + seq_i * (1.0 - a)
        else:
            nxt = prev + dt_i * (seq_i - prev) / (
                tau_i + tf_constant(_EPSILON, tau_i.dtype)
            )

        return tf_reshape(nxt, shp_prev)

    s_tm = tf_scan(
        fn=step,
        elems=(dt_tm, tau_tm, seq_tm),
        initializer=s0_2d,
    )

    s_bar = tf_transpose(s_tm, [1, 0, 2])
    s_bar = _finite_or_zero(s_bar)

    vprint(
        verbose,
        "[integrate_consolidation_mean] s_bar stats:",
        "min=", tf_reduce_min(s_bar),
        "max=", tf_reduce_max(s_bar),
        "mean=", tf_reduce_mean(s_bar),
    )
    return s_bar

def compute_consolidation_step_residual(
    *,
    s_state_si: Tensor,
    h_mean_si: Tensor,
    Ss_field: Tensor,
    H_field_si: Tensor,
    tau_field: Tensor,
    h_ref_si: Tensor,
    dt: Optional[Tensor] = None,
    time_units: Optional[str] = "yr",
    method: str = "exact",
    eps_tau: float = 1e-12,
    relu_beta: float = 20.0,
    drawdown_mode: str = "smooth_relu",
    drawdown_rule: str = "ref_minus_mean",
    stop_grad_ref: bool = True,
    drawdown_zero_at_origin: bool = False,
    drawdown_clip_max: Optional[float] = None,
    verbose: int = 0,
) -> Tensor:
    """
    One-step consolidation residual in SI space.

    This enforces the Voigt-style ODE using a stable stepper.

    Inputs
    ------
    s_state_si : (B,T,1)
        Settlement *state* (typically incremental).
    h_mean_si : (B,T,1)
        Head state aligned with s_state_si.
    Ss_field, H_field_si, tau_field, h_ref_si :
        Fields used to compute equilibrium settlement.

    Returns
    -------
    res : (B,T-1,1)
        Residual per step n -> n+1 in meters.
    """
    # ---------------------------------------------------------
    # 1) Normalize core tensors to (B,T,1) float32.
    # ---------------------------------------------------------
    s_state = _ensure_3d(tf_cast(s_state_si, tf_float32))
    h_state = _ensure_3d(tf_cast(h_mean_si, tf_float32))

    T_s = tf_shape(s_state)[1]
    T_h = tf_shape(h_state)[1]
    tf_debugging.assert_equal(
        T_s,
        T_h,
        message="s_state_si and h_mean_si must share T.",
    )

    vprint(
        verbose,
        "[compute_cons_step_res] T=",
        T_s,
        "| method=",
        method,
    )

    # ---------------------------------------------------------
    # 2) Build step-aligned sequences (length H = T-1).
    # ---------------------------------------------------------
    s_n = s_state[:, :-1, :]     # (B,H,1)
    s_np1 = s_state[:, 1:, :]   # (B,H,1)
    h_n = h_state[:, :-1, :]    # (B,H,1)

    H = tf_shape(s_n)[1]        # H = T-1

    # ---------------------------------------------------------
    # 3) Helper: align a time series to step length H.
    #    Accepts:
    #      - (B,T,1) -> slice to (B,H,1)
    #      - (B,H,1) -> keep
    #      - (B,1,1) -> broadcast later
    # ---------------------------------------------------------
    def _align_to_steps(x: Optional[Tensor], name: str) -> Optional[Tensor]:
        if x is None:
            return None

        xt = _ensure_3d(tf_cast(x, tf_float32))
        tx = tf_shape(xt)[1]

        # If provided at state length T, slice to steps.
        xt = tf_cond(
            tf_equal(tx, H + 1),
            lambda: xt[:, :-1, :],
            lambda: xt,
        )

        # After slicing, require time dim == H or 1.
        tx2 = tf_shape(xt)[1]
        ok = tf_logical_or(tf_equal(tx2, H), tf_equal(tx2, 1))
        tf_debugging.assert_equal(
            ok,
            True,
            message=(
                f"{name} time length must be H or 1 "
                "or T (then sliced)."
            ),
        )
        return xt

    # ---------------------------------------------------------
    # 4) dt handling: align then broadcast to (B,H,1).
    # ---------------------------------------------------------
    if dt is None:
        dt_steps = tf_ones_like(s_n)
        vprint(
            verbose,
            "[compute_cons_step_res] dt=None -> 1 per step",
        )
    else:
        dt_in = _align_to_steps(dt, "dt")
        dt_steps = _broadcast_like(dt_in, s_n)

    # Convert dt to seconds for the stepper.
    dt_sec = dt_to_seconds(dt_steps, time_units=time_units)

    # Optional safety: keep finite, non-negative dt.
    dt_sec = _finite_or_zero(dt_sec)
    dt_sec = tf_maximum(dt_sec, tf_constant(0.0, tf_float32))

    # ---------------------------------------------------------
    # 5) Align other time-series fields to step length.
    # ---------------------------------------------------------
    h_ref_n = _align_to_steps(h_ref_si, "h_ref_si")
    Ss_n = _align_to_steps(Ss_field, "Ss_field")
    Hf_n = _align_to_steps(H_field_si, "H_field_si")
    tau_n = _align_to_steps(tau_field, "tau_field")

    # Broadcast each aligned series to (B,H,1).
    h_ref_n = _broadcast_like(h_ref_n, s_n)
    Ss_n = _broadcast_like(Ss_n, s_n)
    Hf_n = _broadcast_like(Hf_n, s_n)
    tau = _broadcast_like(tau_n, s_n)

    # Clamp tau for numerical stability.
    tau = _finite_or_zero(tau)
    tau = tf_maximum(tau, tf_constant(eps_tau, tf_float32))

    # ---------------------------------------------------------
    # 6) Compute equilibrium settlement at step times.
    # ---------------------------------------------------------
    s_eq_n = equilibrium_compaction_si(
        h_mean_si=h_n,
        h_ref_si=h_ref_n,
        Ss_field=Ss_n,
        H_field_si=Hf_n,
        drawdown_mode=drawdown_mode,
        drawdown_rule=drawdown_rule,
        stop_grad_ref=stop_grad_ref,
        drawdown_zero_at_origin=drawdown_zero_at_origin,
        drawdown_clip_max=drawdown_clip_max,
        relu_beta=relu_beta,
        verbose=verbose,
    )

    # ---------------------------------------------------------
    # 7) Stable one-step prediction and residual.
    # ---------------------------------------------------------
    m = str(method).strip().lower()
    if m == "exact":
        a = tf_exp(-dt_sec / (tau + tf_constant(_EPSILON, tau.dtype)))
        pred = s_n * a + s_eq_n * (1.0 - a)
    elif m == "euler":
        pred = s_n + dt_sec * (s_eq_n - s_n) / (
            tau + tf_constant(_EPSILON, tau.dtype)
        )
    else:
        raise ValueError(
            "compute_consolidation_step_residual: "
            "method must be 'exact' or 'euler'."
        )

    res = s_np1 - pred
    res = _finite_or_zero(res)

    vprint(
        verbose,
        "[compute_cons_step_res] res stats:",
        "min=",
        tf_reduce_min(res),
        "max=",
        tf_reduce_max(res),
        "mean=",
        tf_reduce_mean(res),
    )
    return res


def tau_phys_from_fields(
    model,
    K_field: Tensor,
    Ss_field: Tensor,
    H_field: Tensor,
    *,
    eps: float = _EPSILON,
    verbose: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Compute tau_phys (s) and Hd (m)."""
    eps = float(eps)
    pi_sq = tf_constant(np.pi**2, dtype=tf_float32)

    K_safe = finite_floor(K_field, eps=eps)
    Ss_safe = finite_floor(Ss_field, eps=eps)
    H_safe = finite_floor(H_field, eps=eps)

    use_hd = bool(getattr(model, "use_effective_thickness", False))
    if use_hd:
        f = getattr(model, "Hd_factor", 1.0)
        f = tf_cast(f, H_safe.dtype)
        f = tf_where(tf_math.is_finite(f), f, tf_constant(1.0, H_safe.dtype))
        Hd = H_safe * f
    else:
        Hd = H_safe

    Hd = finite_floor(Hd, eps=eps)
    ratio = Hd / H_safe

    kappa = model._kappa_value()
    kappa = tf_cast(kappa, H_safe.dtype)
    kappa = tf_where(
        tf_math.is_finite(kappa),
        kappa,
        tf_constant(1.0, H_safe.dtype),
    )
    kappa = finite_floor(kappa, eps=eps)

    if str(getattr(model, "kappa_mode", "bar")) == "bar":
        tau_phys = (
            kappa * (H_safe ** 2) * Ss_safe / (pi_sq * K_safe)
        )
    else:
        tau_phys = (
            (ratio ** 2)
            * (H_safe ** 2) * Ss_safe
            / (pi_sq * kappa * K_safe)
        )

    tau_phys = finite_floor(tau_phys, eps=eps)

    vprint(verbose, "tau_phys: K=", K_safe)
    vprint(verbose, "tau_phys: Ss=", Ss_safe)
    vprint(verbose, "tau_phys: H=", H_safe)
    vprint(verbose, "tau_phys: Hd=", Hd)
    vprint(verbose, "tau_phys: out=", tau_phys)

    return tau_phys, Hd

def compute_consistency_prior(
    model,
    K_field: Tensor,
    Ss_field: Tensor,
    tau_field: Tensor,
    H_field: Tensor,
    *,
    verbose: int = 0,
) -> Tensor:
    """Consistency prior: log(tau)-log(tau_phys)."""
    eps = tf_constant(_EPSILON, dtype=tf_float32)

    tau_safe = tf_maximum(tau_field, eps)
    tau_phys, _ = tau_phys_from_fields(
        model,
        K_field,
        Ss_field,
        H_field,
        verbose=0,
    )
    tau_phys_safe = tf_maximum(tau_phys, eps)

    out = tf_log(tau_safe) - tf_log(tau_phys_safe)

    vprint(verbose, "cons_prior: tau=", tau_safe)
    vprint(verbose, "cons_prior: tau_phys=", tau_phys_safe)
    vprint(verbose, "cons_prior: out=", out)

    return out

def compute_smoothness_prior(
    dK_dx: Tensor,
    dK_dy: Tensor,
    dSs_dx: Tensor,
    dSs_dy: Tensor,
    *,
    K_field: Optional[Tensor] = None,
    Ss_field: Optional[Tensor] = None,
    already_log: bool = False,
    verbose: int = 0,
) -> Tensor:
    """
    Smoothness prior on spatial gradients.

    If already_log=True, inputs are d(logK)/dx, 
    d(logK)/dy, d(logSs)/dx, d(logSs)/dy.
    Otherwise, if K_field/Ss_field are provided,
    we convert via division (less stable).
    """
    eps = tf_constant(_EPSILON, dtype=tf_float32)

    if already_log:
        out = (
            tf_square(dK_dx) + tf_square(dK_dy)
            + tf_square(dSs_dx) + tf_square(dSs_dy)
        )
        vprint(verbose, "smooth(log-direct): out=", out)
        return out

    if (K_field is not None) and (Ss_field is not None):
        dlogK_dx  = dK_dx  / (K_field  + eps)
        dlogK_dy  = dK_dy  / (K_field  + eps)
        dlogSs_dx = dSs_dx / (Ss_field + eps)
        dlogSs_dy = dSs_dy / (Ss_field + eps)
        out = (
            tf_square(dlogK_dx) + tf_square(dlogK_dy)
            + tf_square(dlogSs_dx) + tf_square(dlogSs_dy)
        )
        vprint(verbose, "smooth(log-div): out=", out)
        return out

    out = ( 
        tf_square(dK_dx) 
        + tf_square(dK_dy) 
        + tf_square(dSs_dx) 
        + tf_square(dSs_dy)
    )
    
    vprint(verbose, "smooth(raw): out=", out)
    
    return out

# ---------------------------------------------------------------------
# Bounds + field composition
# ---------------------------------------------------------------------
def guarded_exp_from_bounds(
    raw_log,
    log_min,
    log_max,
    *,
    eps=0.0,
    guard=5.0,
    dtype=None,
    name="",
):
    """
    Safe exp() with a wide log-space guard-band around [log_min, log_max].

    - raw_log: unconstrained log-parameter (may drift during training)
    - log_min/log_max: physical bounds in log-space
    - guard: extra margin to avoid overflow; values outside are clipped only
             for *numerical safety*, not as a hard physical constraint.
    """
    if dtype is None:
        dtype = raw_log.dtype

    raw_log = tf_cast(raw_log, dtype)
    log_min = tf_cast(log_min, dtype)
    log_max = tf_cast(log_max, dtype)
    guard = tf_cast(tf_constant(guard), dtype)
    eps = tf_cast(tf_constant(eps), dtype)

    # replace NaN/Inf with 0 to avoid propagating non-finites
    raw_log = tf_where(
        tf_math.is_finite(raw_log), raw_log, 
        tf_zeros_like(raw_log)
    )

    # guard-band clip (prevents exp overflow)
    log_safe = tf_clip_by_value(
        raw_log, log_min - guard, log_max + guard
        )

    field = tf_exp(log_safe) + eps

    if name:
        tf_debugging.assert_all_finite(raw_log,  f"{name} raw_log non-finite")
        tf_debugging.assert_all_finite(field,    f"{name} field non-finite")

    return field, raw_log, log_safe

def get_log_bounds(
    model,
    *,
    as_tensor: bool = True,
    dtype=tf_float32,
    verbose: int = 0,
) -> Tuple[Any, Any, Any, Any]:
    """
    Return (logK_min, logK_max, logSs_min, logSs_max).

    Contract:
    - If bounds are present but invalid (NaN, inf, <=0, max<=min),
      raise ValueError (do NOT emit NaN logs).
    - If bounds are missing, return (None, None, None, None).
    """
    sk = getattr(model, "scaling_kwargs", None) or {}
    b = (sk.get("bounds", None) or {}) or {}

    def _as_float(v: Any) -> float:
        """Best-effort cast to float for config values."""
        if hasattr(v, "numpy"):
            v = v.numpy()
        return float(v)

    def _is_finite(v: float) -> bool:
        return bool(np.isfinite(v))

    def _validate_lin_pair(
        vmin: float,
        vmax: float,
        *,
        name_min: str,
        name_max: str,
    ) -> Tuple[float, float]:
        """Validate linear bounds are finite and positive."""
        if (not _is_finite(vmin)) or (not _is_finite(vmax)):
            raise ValueError(
                f"{name_min}/{name_max} must be finite. "
                f"Got vmin={vmin}, vmax={vmax}."
            )
        if (vmin <= 0.0) or (vmax <= 0.0):
            raise ValueError(
                f"{name_min}/{name_max} must be > 0. "
                f"Got vmin={vmin}, vmax={vmax}."
            )
        if vmax <= vmin:
            raise ValueError(
                f"{name_max} must be > {name_min}. "
                f"Got vmin={vmin}, vmax={vmax}."
            )
        return vmin, vmax

    def _validate_log_pair(
        lmin: float,
        lmax: float,
        *,
        name_min: str,
        name_max: str,
    ) -> Tuple[float, float]:
        """Validate log-bounds are finite and ordered."""
        if (not _is_finite(lmin)) or (not _is_finite(lmax)):
            raise ValueError(
                f"{name_min}/{name_max} must be finite. "
                f"Got lmin={lmin}, lmax={lmax}."
            )
        if lmax <= lmin:
            raise ValueError(
                f"{name_max} must be > {name_min}. "
                f"Got lmin={lmin}, lmax={lmax}."
            )
        return lmin, lmax

    def _maybe_convert_ss_from_mv(
        vmin: float,
        vmax: float,
    ) -> Tuple[float, float]:
        """
        Heuristic: if Ss bounds look like m_v, convert using gamma_w.

        This only runs for (Ss_min, Ss_max). If gamma_w is missing
        or non-finite, we skip conversion (and still validate).
        """
        try:
            gw = getattr(model, "gamma_w", None)
            if gw is None:
                return vmin, vmax

            gw_f = _as_float(gw)
            if (not _is_finite(gw_f)) or (gw_f <= 0.0):
                return vmin, vmax

            # mv_config.initial_value is only used as a sanity hint.
            mv0 = getattr(getattr(model, "mv_config", None),
                          "initial_value", None)
            mv0 = float(mv0) if mv0 is not None else None

            ss_exp = (mv0 * gw_f) if mv0 else None

            # "looks like mv" = very small upper bound,
            # and gamma_w looks like N/m^3.
            looks_mv = (vmax <= 1e-5) and (gw_f > 1e3)

            if looks_mv and (ss_exp is None or ss_exp > 1e-5):
                logger.warning(
                    "Ss_min/max look like m_v; convert via "
                    "Ss = m_v * gamma_w."
                )
                return vmin * gw_f, vmax * gw_f

        except:
            # Never crash: conversion is optional.
            return vmin, vmax

        return vmin, vmax

    def _get_pair(
        log_min_key: str,
        log_max_key: str,
        lin_min_key: str,
        lin_max_key: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Read either log-bounds or linear bounds and return log-bounds.

        Returns (None, None) if neither form exists.
        Raises ValueError on invalid values.
        """
        # 1) Prefer explicit log-bounds if provided.
        log_min = b.get(log_min_key, None)
        log_max = b.get(log_max_key, None)

        if (log_min is not None) and (log_max is not None):
            lmin = _as_float(log_min)
            lmax = _as_float(log_max)
            lmin, lmax = _validate_log_pair(
                lmin,
                lmax,
                name_min=log_min_key,
                name_max=log_max_key,
            )
            return lmin, lmax

        # 2) Otherwise, build from linear bounds if provided.
        if (lin_min_key not in b) or (lin_max_key not in b):
            return None, None

        vmin = _as_float(b[lin_min_key])
        vmax = _as_float(b[lin_max_key])

        # Optional: detect Ss_min/max passed as m_v.
        if (lin_min_key == "Ss_min") and (lin_max_key == "Ss_max"):
            vmin, vmax = _maybe_convert_ss_from_mv(vmin, vmax)

        vmin, vmax = _validate_lin_pair(
            vmin,
            vmax,
            name_min=lin_min_key,
            name_max=lin_max_key,
        )

        return float(np.log(vmin)), float(np.log(vmax))

    logK_min, logK_max = _get_pair(
        "logK_min",
        "logK_max",
        "K_min",
        "K_max",
    )
    logSs_min, logSs_max = _get_pair(
        "logSs_min",
        "logSs_max",
        "Ss_min",
        "Ss_max",
    )

    # If either set is missing, treat bounds as not configured.
    if (logK_min is None) or (logSs_min is None):
        return (None, None, None, None)

    if not as_tensor:
        return logK_min, logK_max, logSs_min, logSs_max

    out = (
        tf_constant(float(logK_min), dtype),
        tf_constant(float(logK_max), dtype),
        tf_constant(float(logSs_min), dtype),
        tf_constant(float(logSs_max), dtype),
    )

    vprint(verbose, "bounds: out=", out)
    return out


def get_log_tau_bounds(
    model,
    *,
    as_tensor: bool = True,
    dtype=tf_float32,
    verbose: int = 0,
) -> Tuple[Any, Any]:
    """
    Return (log_tau_min, log_tau_max) for consolidation timescale.

    Contract:
    - If user provides tau bounds but they are invalid (NaN/inf,
      <=0, max<=min), raise ValueError.
    - If tau bounds are missing, use robust defaults.
    - Returned units are log-seconds (tau is SI seconds).
    """
    sk = getattr(model, "scaling_kwargs", None) or {}
    bounds = (sk.get("bounds", None) or {}) or {}

    def _is_finite(v: float) -> bool:
        return bool(np.isfinite(v))

    def _need_raise(v: Optional[float]) -> bool:
        return (v is not None) and (not _is_finite(float(v)))

    # 1) Explicit log-bounds (already in log-seconds).
    log_min = get_sk(bounds, "log_tau_min",
                     default=None, cast=float)
    log_max = get_sk(bounds, "log_tau_max",
                     default=None, cast=float)

    if _need_raise(log_min) or _need_raise(log_max):
        raise ValueError(
            "log_tau_min/log_tau_max must be finite."
        )

    if (log_min is not None) and (log_max is not None):
        if float(log_max) <= float(log_min):
            raise ValueError(
                "log_tau_max must be > log_tau_min. "
                f"Got {log_min}, {log_max}."
            )
        if not as_tensor:
            return float(log_min), float(log_max)

        out = (
            tf_constant(float(log_min), dtype=dtype),
            tf_constant(float(log_max), dtype=dtype),
        )
        vprint(verbose, "tau_bounds(log-sec):", out)
        return out

    # 2) Linear tau bounds (seconds).
    tau_min = get_sk(bounds, "tau_min", default=None, cast=float)
    tau_max = get_sk(bounds, "tau_max", default=None, cast=float)

    if _need_raise(tau_min) or _need_raise(tau_max):
        raise ValueError("tau_min/tau_max must be finite.")

    # 2b) Linear tau bounds in "time_units".
    if (tau_min is None) or (tau_max is None):
        tau_min_u = get_sk(bounds, "tau_min_units",
                           default=None, cast=float)
        tau_max_u = get_sk(bounds, "tau_max_units",
                           default=None, cast=float)

        if _need_raise(tau_min_u) or _need_raise(tau_max_u):
            raise ValueError(
                "tau_min_units/tau_max_units must be finite."
            )

        if (tau_min_u is not None) and (tau_max_u is not None):
            tu = (
                get_sk(sk, "time_units", default=None)
                or getattr(model, "time_units", None)
                or "yr"
            )
            key = normalize_time_units(tu)
            sec_per = float(TIME_UNIT_TO_SECONDS.get(key, 1.0))
            tau_min = float(tau_min_u) * sec_per
            tau_max = float(tau_max_u) * sec_per

    # 2c) Defaults if still missing.
    if (tau_min is None) or (tau_max is None):
        sec_day = 86400.0
        sec_year = float(
            TIME_UNIT_TO_SECONDS.get("yr", 31556952.0),
        )
        tau_min = 7.0 * sec_day
        tau_max = 300.0 * sec_year
        logger.warning(
            "Tau bounds not found in scaling_kwargs['bounds']; "
            "using defaults: tau_min=7 days, "
            "tau_max=300 years (SI seconds)."
        )

    tau_min = float(tau_min)
    tau_max = float(tau_max)

    if (not _is_finite(tau_min)) or (not _is_finite(tau_max)):
        raise ValueError(
            f"tau_min/tau_max must be finite. "
            f"Got {tau_min}, {tau_max}."
        )
    if (tau_min <= 0.0) or (tau_max <= 0.0):
        raise ValueError(
            f"tau_min/tau_max must be > 0. "
            f"Got {tau_min}, {tau_max}."
        )
    if tau_max < tau_min:
        logger.warning("tau_max < tau_min; swapping tau bounds.")
        tau_min, tau_max = tau_max, tau_min

    log_min = float(np.log(tau_min))
    log_max = float(np.log(tau_max))

    if not as_tensor:
        return log_min, log_max

    out = (
        tf_constant(float(log_min), dtype=dtype),
        tf_constant(float(log_max), dtype=dtype),
    )
    vprint(verbose, "tau_bounds(log-sec):", out)
    return out


def bounded_exp(
    raw: Tensor,
    log_min: Tensor,
    log_max: Tensor,
    *,
    eps: float = _EPSILON,
    return_log: bool = False,
    verbose: int = 0,
):
    """
    Exp with hard log-bounds.

    Contract:
    - Never emit NaN/Inf due to NaN/Inf in `raw`.
    - (log_min, log_max) are assumed finite; if they are not,
      we fall back to safe constants to prevent NaNs.
    """
    eps_t = tf_constant(float(eps), tf_float32)
    log_eps = tf_log(eps_t)

    # Sanitize inputs to avoid NaN propagation.
    raw = tf_cast(raw, tf_float32)
    raw = tf_where(tf_math.is_finite(raw), raw, tf_zeros_like(raw))

    log_min = tf_cast(log_min, tf_float32)
    log_max = tf_cast(log_max, tf_float32)

    log_min = tf_where(
        tf_math.is_finite(log_min),
        log_min,
        log_eps,
    )
    log_max = tf_where(
        tf_math.is_finite(log_max),
        log_max,
        log_min + tf_constant(1.0, tf_float32),
    )

    # If user swapped bounds, repair silently (safe + monotone).
    log_lo = tf_minimum(log_min, log_max)
    log_hi = tf_maximum(log_min, log_max)

    # Map raw -> (0,1) then interpolate inside [log_lo, log_hi].
    z = tf_sigmoid(raw)
    logv = log_lo + z * (log_hi - log_lo)

    # Output is positive, with epsilon floor.
    out = tf_exp(logv) + eps_t

    vprint(verbose, "bounded_exp: logv=", logv)
    vprint(verbose, "bounded_exp: out=", out)

    if return_log:
        return out, logv
    return out


def finite_floor(x: Tensor, eps: float) -> Tensor:
    """
    Replace NaN/Inf by eps and floor to eps.

    Useful when you want "never NaN" behaviour, not strict errors.
    """
    x = tf_cast(x, tf_float32)
    eps_t = tf_constant(float(eps), tf_float32)
    x = tf_where(tf_math.is_finite(x), x, eps_t)
    return tf_maximum(x, eps_t)

def _finite_or_zero(x: Tensor) -> Tensor:
    x = tf_cast(x, tf_float32)
    return tf_where(tf_math.is_finite(x), x, tf_zeros_like(x))

def compose_physics_fields(
    model,
    *,
    coords_flat: Tensor,
    H_si: Tensor,
    K_base: Tensor,
    Ss_base: Tensor,
    tau_base: Tensor,
    training: bool = False,
    eps_KSs: float = _EPSILON,
    eps_tau: float = 1e-6,
    verbose: int = 0,
):
    """Compose K,Ss,tau fields with coord MLPs."""

    if verbose > 6:
        tf_print_nonfinite("compose/coords_flat", coords_flat)
        tf_print_nonfinite("compose/K_base", K_base)
        tf_print_nonfinite("compose/Ss_base", Ss_base)
        tf_print_nonfinite("compose/tau_base", tau_base)

    coords_xy0 = tf_concat(
        [tf_zeros_like(coords_flat[..., :1]), coords_flat[..., 1:]],
        axis=-1,
    )
    coords_xy0 = _finite_or_zero(coords_xy0)
    
    K_corr = _finite_or_zero(model.K_coord_mlp(coords_xy0, training=training))
    Ss_corr = _finite_or_zero(model.Ss_coord_mlp(coords_xy0, training=training))
    tau_corr = _finite_or_zero(model.tau_coord_mlp(coords_xy0, training=training))

    # K_corr = model.K_coord_mlp(coords_xy0, training=training)
    # Ss_corr = model.Ss_coord_mlp(coords_xy0, training=training)
    # tau_corr = model.tau_coord_mlp(coords_xy0, training=training)

    if verbose > 6:
        tf_print_nonfinite("compose/K_corr", K_corr)
        tf_print_nonfinite("compose/Ss_corr", Ss_corr)
        tf_print_nonfinite("compose/tau_corr", tau_corr)

    rawK = K_base + K_corr
    rawSs = Ss_base + Ss_corr

    bounds_mode = str(getattr(model, "bounds_mode", "soft")).strip().lower()
    
    logK_min, logK_max, logSs_min, logSs_max = get_log_bounds(
        model, as_tensor=True, dtype=rawK.dtype, verbose=verbose,
    )
    # ---- K, Ss  ----
    if bounds_mode == "hard":
        K_field, logK = bounded_exp(
            rawK, logK_min, logK_max, eps=eps_KSs,
            return_log=True, verbose=verbose,
        )
        Ss_field, logSs = bounded_exp(
            rawSs, logSs_min, logSs_max, eps=eps_KSs,
            return_log=True, verbose=verbose,
        )

    else:
        # Keep raw log-params (useful for priors/diagnostics),
        # but NEVER feed an unbounded log into exp() in float32.
        K_field,  logK,  _ = guarded_exp_from_bounds(
            rawK,  logK_min,  logK_max, eps=eps_KSs, 
            guard=5.0, name="K"
        )
        Ss_field, logSs, _ = guarded_exp_from_bounds(
            rawSs, logSs_min, logSs_max,eps=eps_KSs,
            guard=5.0, name="Ss"
        )
    
        # Optional: keep the asserts, but now they won't trip from exp overflow.
        tf_debugging.assert_all_finite(logK,    "rawK/logK non-finite")
        tf_debugging.assert_all_finite(logSs,   "rawSs/logSs non-finite")
        tf_debugging.assert_all_finite(K_field, "K_field non-finite")
        tf_debugging.assert_all_finite(Ss_field,"Ss_field non-finite")
    

    # ---- tau ( log-space composition + bounds) ----
    delta_log_tau = _finite_or_zero(tau_base + tau_corr)

    if verbose > 6:
        tf_print_nonfinite("compose/rawK", rawK)
        tf_print_nonfinite("compose/rawSs", rawSs)
        tf_print_nonfinite("compose/delta_log_tau", delta_log_tau)

    tau_phys, Hd_eff = tau_phys_from_fields(
        model, K_field, Ss_field, H_si, verbose=0,
    )

    # Safe log(tau_phys)
    tau_phys_safe = tf_maximum(
        tau_phys,
        tf_constant(eps_tau, tau_phys.dtype),
    )
    log_tau_phys = tf_math.log(tau_phys_safe)
    log_tau_total = log_tau_phys + delta_log_tau

    log_tau_min, log_tau_max = get_log_tau_bounds(
        model,
        as_tensor=True,
        dtype=log_tau_total.dtype,
        verbose=0,
    )

    if bounds_mode == "hard":
        # true hard bounds: clip in log-space (keeps tau_phys anchoring)
        log_tau = tf_clip_by_value(log_tau_total, log_tau_min, log_tau_max)
        tau_field = tf_exp(log_tau) + tf_constant(eps_tau, log_tau.dtype)
    else:
        # soft mode: keep log_tau for bounds penalty, but guard exp overflow
        log_tau = log_tau_total

        guard_lo = log_tau_min - tf_constant(10.0, log_tau.dtype)
        guard_hi = log_tau_max + tf_constant(10.0, log_tau.dtype)
        log_tau_safe = tf_clip_by_value(log_tau, guard_lo, guard_hi)

        tau_field = tf_exp(log_tau_safe) + tf_constant(eps_tau, log_tau.dtype)

    if verbose > 6:
        tf_print_nonfinite("compose/K_field", K_field)
        tf_print_nonfinite("compose/Ss_field", Ss_field)
        tf_print_nonfinite("compose/tau_phys", tau_phys)
        tf_print_nonfinite("compose/log_tau_phys", log_tau_phys)
        tf_print_nonfinite("compose/log_tau_total", log_tau_total)
        tf_print_nonfinite("compose/tau_field", tau_field)

    vprint(verbose, "fields: K=", K_field)
    vprint(verbose, "fields: Ss=", Ss_field)
    vprint(verbose, "fields: tau=", tau_field)
    vprint(verbose, "fields: tau_phys=", tau_phys)

    return (
        K_field,
        Ss_field,
        tau_field,
        tau_phys,
        Hd_eff,
        delta_log_tau,
        logK,
        logSs,
        log_tau,        # return log_tau for bounds penalty + diagnostics
        log_tau_phys,   # optional but very useful for priors/diagnostics
    )

def _log_bounds_residual(
    logv: Tensor,
    lo: Tensor,
    hi: Tensor,
    *,
    eps: float = 1e-12,
    name: str = "",
) -> Tensor:
    """
    Normalized bound violation in log-space.

    We compute a symmetric distance outside [lo, hi], then
    normalize by the range (hi - lo). This returns 0 inside
    bounds and >0 outside bounds.

    Notes
    -----
    - We sanitize non-finite logv to avoid NaN explosions.
    - lo/hi are assumed finite tensors (from helpers).
    """
    dtype = logv.dtype
    zero = tf_constant(0.0, dtype=dtype)
    eps_t = tf_constant(float(eps), dtype=dtype)

    # Sanitize inputs: never propagate NaN/Inf into loss.
    is_ok = tf_math.is_finite(logv)
    logv = tf_where(is_ok, logv, tf_zeros_like(logv))

    lo = tf_cast(lo, dtype)
    hi = tf_cast(hi, dtype)

    lower = tf_maximum(lo - logv, zero)
    upper = tf_maximum(logv - hi, zero)

    rng = tf_maximum(hi - lo, eps_t)
    res = (lower + upper) / rng

    # Optional debug checks (keep off by default).
    if name:
        msg = name + " bounds residual non-finite"
        tf_debugging.assert_all_finite(res, msg)

    return res


def compute_bounds_residual(
    model: Any,
    *,
    H_field: Tensor,
    logK: Optional[Tensor] = None,
    logSs: Optional[Tensor] = None,
    log_tau: Optional[Tensor] = None,
    K_field: Optional[Tensor] = None,
    Ss_field: Optional[Tensor] = None,
    tau_field: Optional[Tensor] = None,
    eps: float = _EPSILON, 
    verbose: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Bounds residuals for H, K, Ss, tau.

    Preferred (soft mode)
    ---------------------
    Pass raw logs:
      - logK, logSs, log_tau
    so penalties see the *true* violations, not guarded
    exp() outputs.

    Fallback
    --------
    If logs are missing, infer them from fields:
      - log(K_field), log(Ss_field), log(tau_field)
    This is less ideal in soft mode if fields were guarded.
    """
    dtype = H_field.dtype
    eps_t = tf_constant(eps, dtype=dtype)
    zero = tf_constant(0.0, dtype=dtype)

    # ------------------------------------------------------
    # H bounds (linear space, SI meters).
    # ------------------------------------------------------
    H_safe = tf_maximum(tf_cast(H_field, dtype), eps_t)

    sk = getattr(model, "scaling_kwargs", None) or {}
    b = (sk.get("bounds", None) or {}) or {}

    H_min = b.get("H_min", None)
    H_max = b.get("H_max", None)

    if (H_min is None) or (H_max is None):
        R_H = tf_zeros_like(H_safe)
    else:
        H_min_t = tf_constant(float(H_min), dtype=dtype)
        H_max_t = tf_constant(float(H_max), dtype=dtype)

        lo = tf_maximum(H_min_t - H_safe, zero)
        hi = tf_maximum(H_safe - H_max_t, zero)

        rng = tf_maximum(H_max_t - H_min_t, eps_t)
        R_H = (lo + hi) / rng

    # ------------------------------------------------------
    # K, Ss bounds (log-space).
    # Prefer raw logs if provided.
    # ------------------------------------------------------
    out = get_log_bounds(
        model,
        as_tensor=True,
        dtype=dtype,
        verbose=0,
    )
    logK_min, logK_max, logSs_min, logSs_max = out

    if logK_min is None:
        # Bounds not configured -> no penalty.
        R_K = tf_zeros_like(H_safe)
        R_Ss = tf_zeros_like(H_safe)
    else:
        # ---- K residual ----
        if logK is None:
            if K_field is None:
                R_K = tf_zeros_like(H_safe)
            else:
                K_safe = tf_maximum(tf_cast(K_field, dtype), eps_t)
                logK_hat = tf_math.log(K_safe)
                R_K = _log_bounds_residual(
                    logK_hat,
                    logK_min,
                    logK_max,
                    name="K",
                )
        else:
            R_K = _log_bounds_residual(
                tf_cast(logK, dtype),
                logK_min,
                logK_max,
                name="K",
            )

        # ---- Ss residual ----
        if logSs is None:
            if Ss_field is None:
                R_Ss = tf_zeros_like(H_safe)
            else:
                Ss_safe = tf_maximum(
                    tf_cast(Ss_field, dtype),
                    eps_t,
                )
                logSs_hat = tf_math.log(Ss_safe)
                R_Ss = _log_bounds_residual(
                    logSs_hat,
                    logSs_min,
                    logSs_max,
                    name="Ss",
                )
        else:
            R_Ss = _log_bounds_residual(
                tf_cast(logSs, dtype),
                logSs_min,
                logSs_max,
                name="Ss",
            )

    # ------------------------------------------------------
    # tau bounds (log-space, seconds).
    # Prefer raw log_tau if provided.
    # ------------------------------------------------------
    log_tau_min, log_tau_max = get_log_tau_bounds(
        model,
        as_tensor=True,
        dtype=dtype,
        verbose=0,
    )

    if log_tau is not None:
        R_tau = _log_bounds_residual(
            tf_cast(log_tau, dtype),
            log_tau_min,
            log_tau_max,
            name="tau",
        )
    elif tau_field is not None:
        tau_safe = tf_maximum(tf_cast(tau_field, dtype), eps_t)
        log_tau_hat = tf_math.log(tau_safe)
        R_tau = _log_bounds_residual(
            log_tau_hat,
            log_tau_min,
            log_tau_max,
            name="tau",
        )
    else:
        R_tau = tf_zeros_like(H_safe)

    if verbose > 6:
        vprint(verbose, "bounds: R_H=", R_H)
        vprint(verbose, "bounds: R_K=", R_K)
        vprint(verbose, "bounds: R_Ss=", R_Ss)
        vprint(verbose, "bounds: R_tau=", R_tau)

    return R_H, R_K, R_Ss, R_tau

def _compute_bounds_residual(
    model,
    K_field: Tensor,
    Ss_field: Tensor,
    H_field: Tensor,
    *,
    verbose: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Bounds residuals for H,K,Ss."""
    dtype = K_field.dtype
    eps = tf_constant(1e-12, dtype=dtype)
    zero = tf_constant(0.0, dtype=dtype)

    K_safe = tf_maximum(K_field, eps)
    Ss_safe = tf_maximum(Ss_field, eps)
    H_safe = tf_maximum(H_field, eps)
    
    bounds_cfg = (model.scaling_kwargs or {}).get(
        "bounds",
        {},
    ) or {}

    H_min = bounds_cfg.get("H_min", None)
    H_max = bounds_cfg.get("H_max", None)
    if (H_min is None) or (H_max is None):
        R_H = tf_zeros_like(H_safe)
    else:
        H_min_t = tf_constant(float(H_min), dtype=dtype)
        H_max_t = tf_constant(float(H_max), dtype=dtype)

        lower = tf_maximum(H_min_t - H_safe, zero)
        upper = tf_maximum(H_safe - H_max_t, zero)

        H_rng = tf_maximum(H_max_t - H_min_t, eps)
        R_H = (lower + upper) / H_rng

    def log_bound(val_safe, log_min, log_max):
        logv = tf_log(val_safe)
        lo = tf_constant(float(log_min), dtype=dtype)
        hi = tf_constant(float(log_max), dtype=dtype)

        lower = tf_maximum(lo - logv, zero)
        upper = tf_maximum(logv - hi, zero)

        rng = tf_maximum(hi - lo, eps)
        return (lower + upper) / rng

    logK_min = bounds_cfg.get("logK_min", None)
    logK_max = bounds_cfg.get("logK_max", None)
    if (logK_min is None or logK_max is None) and (
        bounds_cfg.get("K_min") is not None
        and bounds_cfg.get("K_max") is not None
    ):
        logK_min = float(np.log(float(bounds_cfg["K_min"])))
        logK_max = float(np.log(float(bounds_cfg["K_max"])))

    if (logK_min is None) or (logK_max is None):
        R_K = tf_zeros_like(K_safe)
    else:
        R_K = log_bound(K_safe, logK_min, logK_max)

    logSs_min = bounds_cfg.get("logSs_min", None)
    logSs_max = bounds_cfg.get("logSs_max", None)
    if (logSs_min is None or logSs_max is None) and (
        bounds_cfg.get("Ss_min") is not None
        and bounds_cfg.get("Ss_max") is not None
    ):
        logSs_min = float(np.log(float(bounds_cfg["Ss_min"])))
        logSs_max = float(np.log(float(bounds_cfg["Ss_max"])))

    if (logSs_min is None) or (logSs_max is None):
        R_Ss = tf_zeros_like(Ss_safe)
    else:
        R_Ss = log_bound(Ss_safe, logSs_min, logSs_max)
    
    if verbose > 6:
        vprint(verbose, "bounds: R_H=", R_H)
        vprint(verbose, "bounds: R_K=", R_K)
        vprint(verbose, "bounds: R_Ss=", R_Ss)

    return R_H, R_K, R_Ss


def guard_scale_with_residual(
    residual: Tensor,
    scale: Tensor,
    *,
    floor: float,
    eps: float = _EPSILON,
) -> Tensor:
    """Ensure `scale` is never tiny vs the actual residual."""
    dtype = residual.dtype
    eps_t = tf_constant(float(eps), dtype=dtype)
    floor_t = tf_constant(float(floor), dtype=dtype)

    r = tf_abs(_finite_or_zero(residual))
    r = tf_reshape(r, [-1])

    r_ref = tf_stop_gradient(tf_reduce_mean(r) + eps_t)
    r_max = tf_stop_gradient(tf_reduce_max(r) + eps_t)

    s = tf_cast(scale, dtype)
    s = tf_where(tf_math.is_finite(s), s, floor_t)

    # Guard: scale >= typical residual magnitude
    s = tf_maximum(s, r_ref)
    s = tf_maximum(s, tf_constant(0.1, dtype) * r_max)

    return tf_stop_gradient(tf_maximum(s, floor_t))

def scale_residual(
    residual: Tensor, 
    scale: Tensor, *, 
    floor: float = _EPSILON
 ) -> Tensor:
    s = tf_cast(scale, residual.dtype)
    f = tf_constant(float(floor), residual.dtype)

    # If scale is NaN/Inf -> replace with floor BEFORE max()
    s = tf_where(tf_math.is_finite(s), s, f)

    s = tf_maximum(s, f)
    s = tf_stop_gradient(s)
    return residual / (s + tf_constant(_EPSILON, residual.dtype))

def _cons_scale_core(
    *,
    s: Tensor,
    h: Tensor,
    Ss: Tensor,
    dt_ref_u: Tensor,
    dt_ref_s: Tensor,
    mode: str,
    time_units: str,
    tau: Tensor,
    Hf: Tensor,
    href: Tensor,
    use_relax: bool,
    floor: float,
) -> Tensor:
    """
    Consolidation scale.

    Output units match consolidation residual units:
    - "step": meters / step
    - "time_unit": meters / time_unit
    - "si": meters / second
    """
    eps = tf_constant(_EPSILON, tf_float32)
    floor_t = tf_constant(float(floor), tf_float32)

    # ------------------------------------------------------
    # Sanitize inputs (avoid NaN/Inf in reductions).
    # ------------------------------------------------------
    s = _finite_or_zero(s)
    h = _finite_or_zero(h)
    Ss = _finite_or_zero(Ss)

    dt_ref_u = finite_floor(dt_ref_u, _EPSILON)
    dt_ref_s = finite_floor(dt_ref_s, _EPSILON)

    # ------------------------------------------------------
    # ds statistics (meters). Must be graph-safe:
    # use tf_cond, not a Python `if` on tf.shape().
    # ------------------------------------------------------
    def _ds_stats() -> tuple[Tensor, Tensor]:
        ds = s[:, 1:, :] - s[:, :-1, :]
        ds = _finite_or_zero(ds)

        ds_abs = tf_abs(tf_reshape(ds, [-1]))

        ds_ref = tf_stop_gradient(tf_reduce_mean(ds_abs))
        ds_max = tf_stop_gradient(tf_reduce_max(ds_abs))

        return ds_ref, ds_max

    def _ds_stats_zero() -> tuple[Tensor, Tensor]:
        z = tf_constant(0.0, tf_float32)
        return z, z

    # Horizon length H = shape(s)[1]
    H_len = tf_shape(s)[1]
    has_ds = tf_greater(H_len, tf_constant(1, tf_int32))

    ds_ref, ds_max = tf_cond(has_ds, _ds_stats, _ds_stats_zero)

    # ------------------------------------------------------
    # Base scale from ds (step / rate).
    # ------------------------------------------------------
    if mode == "step":
        cons = tf_maximum(ds_ref, 0.1 * ds_max)

    elif mode == "time_unit":
        cons = tf_maximum(
            ds_ref / dt_ref_u,
            0.1 * (ds_max / dt_ref_u),
        )

    else:
        cons = tf_maximum(
            ds_ref / dt_ref_s,
            0.1 * (ds_max / dt_ref_s),
        )

    # ------------------------------------------------------
    # Optional equilibrium / relaxation term.
    # ------------------------------------------------------
    if use_relax:
        tau = finite_floor(tau, _EPSILON)
        Hf = tf_maximum(_finite_or_zero(Hf), 0.0)
        href = _finite_or_zero(href)

        # dh >= 0 (drawdown / head loss proxy)
        dh = tf_maximum(href - h, 0.0)

        # 1D equilibrium settlement (meters)
        s_eq = Ss * dh * Hf

        # Misfit to equilibrium (meters)
        eq_mis = tf_abs(_finite_or_zero(s_eq - s))
        eq_vec = tf_reshape(eq_mis, [-1])

        eq_ref = tf_stop_gradient(
            tf_reduce_mean(eq_vec) + eps
        )
        eq_max = tf_stop_gradient(
            tf_reduce_max(eq_vec) + eps
        )

        if mode == "step":
            # In step mode, keep as meters/step.
            cons = tf_maximum(cons, eq_ref)
            cons = tf_maximum(cons, 0.1 * eq_max)

        else:
            # Relaxation rate: meters/second.
            relax = tf_abs(eq_mis / (tau + eps))

            if mode == "time_unit":
                # Convert to meters/time_unit.
                sec_u = seconds_per_time_unit(
                    time_units,
                    dtype=tf_float32,
                )
                relax = relax * sec_u

            relax = _finite_or_zero(relax)
            r_vec = tf_reshape(relax, [-1])

            r_ref = tf_stop_gradient(
                tf_reduce_mean(r_vec) + eps
            )
            r_max = tf_stop_gradient(
                tf_reduce_max(r_vec) + eps
            )

            cons = tf_maximum(cons, r_ref)
            cons = tf_maximum(cons, 0.1 * r_max)

    # ------------------------------------------------------
    # Final floor and stop-gradient.
    # ------------------------------------------------------
    cons = tf_maximum(cons, floor_t)
    return tf_stop_gradient(cons)

def _gw_scale_core(
    *,
    h: Tensor,
    Ss: Tensor,
    dt_ref_s: Tensor,
    time_units: str,
    gw_units: str,
    dh_dt: Optional[Tensor],
    div_K_grad_h: Optional[Tensor],
    Q: Optional[Tensor],
    floor: float,
) -> Tensor:
    """
    Groundwater scale.

    Base SI terms are in 1/s:
    - storage: Ss * dh/dt
    - div:     div(K grad h)  (precomputed upstream)
    - forcing: Q             (precomputed upstream)

    Output units:
    - default: SI (1/s)
    - if gw_units == "time_unit": 1/time_unit
    """
    eps = tf_constant(_EPSILON, tf_float32)
    floor_t = tf_constant(float(floor), tf_float32)

    # ------------------------------------------------------
    # Sanitize inputs (avoid NaN/Inf in reductions).
    # ------------------------------------------------------
    h = _finite_or_zero(h)
    Ss = _finite_or_zero(Ss)
    dt_ref_s = finite_floor(dt_ref_s, _EPSILON)

    # ------------------------------------------------------
    # dh/dt reference (SI: m/s).
    # If dh_dt is provided, use it directly.
    # Otherwise estimate from consecutive steps in h.
    # Must be graph-safe: use tf_cond for shape checks.
    # ------------------------------------------------------
    def _dh_dt_from_h() -> tuple[Tensor, Tensor]:
        dh = h[:, 1:, :] - h[:, :-1, :]
        dh = _finite_or_zero(dh)

        dh_abs = tf_abs(tf_reshape(dh, [-1]))

        dh_ref = tf_stop_gradient(tf_reduce_mean(dh_abs))
        dh_max = tf_stop_gradient(tf_reduce_max(dh_abs))

        dh_dt_ref = dh_ref / dt_ref_s
        dh_dt_max = dh_max / dt_ref_s

        return dh_dt_ref, dh_dt_max

    def _dh_dt_zero() -> tuple[Tensor, Tensor]:
        z = tf_constant(0.0, tf_float32)
        return z, z

    if dh_dt is None:
        H_len = tf_shape(h)[1]
        has_dh = tf_greater(H_len, tf_constant(1, tf_int32))
        dh_dt_ref, dh_dt_max = tf_cond(
            has_dh,
            _dh_dt_from_h,
            _dh_dt_zero,
        )
    else:
        d = _finite_or_zero(dh_dt)
        d = tf_abs(tf_reshape(d, [-1]))

        dh_dt_ref = tf_stop_gradient(tf_reduce_mean(d))
        dh_dt_max = tf_stop_gradient(tf_reduce_max(d))

    # ------------------------------------------------------
    # Ss reference (1/m).
    # ------------------------------------------------------
    Ss_abs = tf_abs(tf_reshape(_finite_or_zero(Ss), [-1]))
    Ss_ref = tf_stop_gradient(tf_reduce_mean(Ss_abs))

    # Storage term scale (1/s).
    storage_ref = Ss_ref * dh_dt_ref
    storage_max = Ss_ref * dh_dt_max

    gw = tf_maximum(storage_ref, 0.1 * storage_max)

    # ------------------------------------------------------
    # Optional div term (already in compatible units).
    # ------------------------------------------------------gim
    if div_K_grad_h is not None:
        divv = _finite_or_zero(div_K_grad_h)
        divv = tf_abs(tf_reshape(divv, [-1]))

        div_ref = tf_stop_gradient(tf_reduce_mean(divv) + eps)
        div_max = tf_stop_gradient(tf_reduce_max(divv) + eps)

        gw = tf_maximum(gw, div_ref)
        gw = tf_maximum(gw, 0.1 * div_max)

    # ------------------------------------------------------
    # Optional forcing term Q (already in compatible units).
    # ------------------------------------------------------
    if Q is not None:
        QQ = _finite_or_zero(Q)
        QQ = tf_abs(tf_reshape(QQ, [-1]))

        Q_ref = tf_stop_gradient(tf_reduce_mean(QQ) + eps)
        Q_max = tf_stop_gradient(tf_reduce_max(QQ) + eps)

        gw = tf_maximum(gw, Q_ref)
        gw = tf_maximum(gw, 0.1 * Q_max)

    # Floor for numerical stability.
    gw = tf_maximum(gw, floor_t)

    # ------------------------------------------------------
    # Optional "per time_unit" conversion (non-SI).
    # ------------------------------------------------------
    if gw_units == "time_unit":
        sec_u = seconds_per_time_unit(
            time_units,
            dtype=tf_float32,
        )
        gw = gw * sec_u

    return tf_stop_gradient(gw)

# @optional_tf_function
def compute_scales(
    model,
    *,
    t: Tensor,
    s_mean: Tensor,
    h_mean: Tensor,
    K_field: Tensor,
    Ss_field: Tensor,
    tau_field: Optional[Tensor] = None,
    H_field: Optional[Tensor] = None,
    h_ref_si: Optional[Tensor] = None,
    Q: Optional[Tensor] = None,
    dt: Optional[Tensor] = None,
    time_units: Optional[str] = None,
    dh_dt: Optional[Tensor] = None,
    div_K_grad_h: Optional[Tensor] = None,
    verbose: int = 0,
) -> Dict[str, Tensor]:
    """
    Robust residual scales (v3.2).

    This wrapper is *not* tf.function-traced because it
    accepts a Python `model` object.
    """
    sk = getattr(model, "scaling_kwargs", None) or {}
    mode = resolve_cons_units(sk)
    gw_units = resolve_gw_units(sk)

    # --- Normalize ranks to (B,H,1).
    s = tf_cast(s_mean, tf_float32)
    h = tf_cast(h_mean, tf_float32)
    if s.shape.rank == 2:
        s = s[:, :, None]
    if h.shape.rank == 2:
        h = h[:, :, None]

    # --- Time units (consistent source of truth).
    if time_units is None:
        time_units = (
            get_sk(sk, "time_units", default=None)
            or getattr(model, "time_units", None)
            or "unitless"
        )

    
    def _diffs():
        return tt[:, 1:, :] - tt[:, :-1, :]
    
    def _ones():
        return tf_zeros_like(s[:, :1, :]) + 1.0
    
    # --- Build dt in *time_units*.
    if dt is None:
        tt = tf_cast(t, tf_float32)
        if tt.shape.rank == 2:
            tt = tt[:, :, None]
            
        H = tf_shape(tt)[1]

        # if (tt.shape.rank >= 2) and (tt.shape[1] > 1):
        #     dt_step = tt[:, 1:, :] - tt[:, :-1, :]
        # else:
        #     dt_step = tf_zeros_like(s[:, :1, :]) + 1.0
        
        dt_step = tf_cond(tf_greater(H, 1), _diffs, _ones)

        # De-normalize if coords were normalized.
        coords_norm = bool(sk.get("coords_normalized", False))
        tR, _, _ = coord_ranges(sk)
        if coords_norm and tR:
            dt_step = dt_step * tf_cast(float(tR),
                                        tf_float32)
    else:
        dt_step = tf_cast(dt, tf_float32)
        if dt_step.shape.rank == 2:
            dt_step = dt_step[:, :, None]

    # Sanitize dt before any conversion/reduction.
    dt_step = tf_abs(_finite_or_zero(dt_step))
    dt_step = finite_floor(dt_step, _EPSILON)

    dt_sec = dt_to_seconds(dt_step, time_units=time_units)
    dt_sec = tf_abs(_finite_or_zero(dt_sec))
    dt_sec = finite_floor(dt_sec, _EPSILON)

    # Scalar dt refs.
    dt_ref_u = tf_reduce_mean(tf_reshape(dt_step, [-1]))
    dt_ref_u = finite_floor(dt_ref_u, _EPSILON)

    dt_ref_s = tf_reduce_mean(tf_reshape(dt_sec, [-1]))
    dt_ref_s = finite_floor(dt_ref_s, _EPSILON)

    # Prefer a sane SI lower bound when dt is broken.
    sec_u = seconds_per_time_unit(
        time_units,
        dtype=tf_float32,
    )
    dt_ref_s = tf_maximum(dt_ref_s, sec_u)

    # --- h_ref broadcast (finite).
    if h_ref_si is None:
        h_ref_si = tf_cast(getattr(model, "h_ref", 0.0),
                           tf_float32)
    href = tf_convert_to_tensor(h_ref_si, tf_float32)
    href = tf_broadcast_to(href, tf_shape(h))
    href = _finite_or_zero(href)

    # --- Floors.
    # cons_floor_def = _EPSILON
    # if mode in ("step", "time_unit"):
    #     cons_floor_def = 1e-6
    
    cons_floor = resolve_auto_scale_floor("cons", sk)
    gw_floor   = resolve_auto_scale_floor("gw", sk)
    
    # cons_floor = float(
    #     get_sk(sk, "cons_scale_floor", default=cons_floor_def)
    # )
    # gw_floor = float(
    #     get_sk(sk, "gw_scale_floor", default=_EPSILON)
    # )

    # --- Optional tau/H (shape-safe).
    use_relax = (tau_field is not None) and (H_field is not None)

    if use_relax:
        tau = tf_cast(tau_field, tf_float32)
        Hf = tf_cast(H_field, tf_float32)
        if tau.shape.rank == 2:
            tau = tau[:, :, None]
        if Hf.shape.rank == 2:
            Hf = Hf[:, :, None]
        tau = tf_broadcast_to(tau, tf_shape(h))
        Hf = tf_broadcast_to(Hf, tf_shape(h))
    else:
        tau = tf_ones_like(h)
        Hf = tf_zeros_like(h)

    # --- Sanitize Ss once, then reuse.
    Ss = tf_cast(Ss_field, tf_float32)
    if Ss.shape.rank == 2:
        Ss = Ss[:, :, None]
    Ss = tf_broadcast_to(Ss, tf_shape(h))
    Ss = _finite_or_zero(Ss)

    cons_scale = _cons_scale_core(
        s=s,
        h=h,
        Ss=Ss,
        dt_ref_u=dt_ref_u,
        dt_ref_s=dt_ref_s,
        mode=mode,
        time_units=time_units,
        tau=tau,
        Hf=Hf,
        href=href,
        use_relax=use_relax,
        floor=cons_floor,
    )

    gw_scale = _gw_scale_core(
        h=h,
        Ss=Ss,
        dt_ref_s=dt_ref_s,
        time_units=time_units,
        gw_units=gw_units,
        dh_dt=dh_dt,
        div_K_grad_h=div_K_grad_h,
        Q=Q,
        floor=gw_floor,
    )

    if verbose > 0:
        _stats("cons_scale", cons_scale)
        _stats("gw_scale", gw_scale)

    return {"cons_scale": cons_scale, "gw_scale": gw_scale}


def resolve_auto_scale_floor(
    key: str,
    scaling_kwargs: dict[str, Any] | None,
    default_val: float | str = "auto",
) -> float:
    """
    Robustly determine a numerical stability floor for physics scales.

    If the user provides a float in scaling_kwargs, it is respected.
    If 'auto', we derive a safe floor based on float32 stability limits
    converted to the active unit system (SI, time_units, or steps).

    Baselines (SI):
      - cons (velocity): 1e-7 m/s  (~3 m/yr)
        High floor because velocity residuals are often noise-dominated.
      - gw (rate):       1e-9 1/s  (~0.03 /yr)
        Lower floor to capture subtler groundwater dynamics.
    """
    sk = scaling_kwargs or {}
    
    # 1. Check user override in config (e.g., "cons_scale_floor": 1e-12)
    #    We strip "auto" if it appears as a string literal.
    val = get_sk(sk, f"{key}_scale_floor", default=default_val)
    
    if isinstance(val, (float, int)) and not isinstance(val, bool):
        return float(val)
        
    if str(val).lower() != "auto":
        try:
            return float(val)
        except (ValueError, TypeError):
            pass # Fallthrough to auto logic

    # 2. "Auto" Logic: Derive based on Units
    time_units = get_sk(sk, "time_units", default="year")
    
    # Calculate conversion factor: 1 "time_unit" = X seconds
    try:
        sec_per_unit = float(seconds_per_time_unit(time_units))
    except Exception:
        sec_per_unit = 31556952.0 # Default to year if unknown
        
    # Define Safe SI Baselines (float32 stability thresholds)
    # m/s for cons, 1/s for gw
    SI_BASE_CONS = 1e-7 
    SI_BASE_GW   = 1e-9 

    if key == "cons":
        # Target units: "second", "time_unit", or "step"
        resid_units = str(get_sk(sk, "cons_residual_units", default="second")).lower()
        
        if "second" in resid_units:
            return SI_BASE_CONS
        elif "time" in resid_units:
            # Convert m/s -> m/year (or m/month, etc)
            # floor = (m/s) * (s/unit) = m/unit
            return SI_BASE_CONS * sec_per_unit
        else:
            # "step": treat roughly like SI (conservative)
            return SI_BASE_CONS

    elif key == "gw":
        # Target units: "second" or "time_unit"
        resid_units = str(get_sk(sk, "gw_residual_units", default="time_unit")).lower()
        
        if "second" in resid_units:
            return SI_BASE_GW
        elif "time" in resid_units:
            # Convert 1/s -> 1/year
            # floor = (1/s) * (s/unit) = 1/unit
            return SI_BASE_GW * sec_per_unit
            
    # Fallback safe default
    return 1e-7

def resolve_gw_units(sk):
    v = get_sk(sk, "gw_residual_units", default="time_unit")
    v = str(v).strip().lower()
    if v in ("sec", "second", "seconds", "s"):
        return "second"
    return "time_unit"

def resolve_cons_units(
    sk: Optional[Dict[str, Any]],
) -> str:
    """Normalize consolidation residual units."""
    if not sk:
        return "second"

    v = get_sk(sk, "cons_residual_units", default="second")
    mode = str(v).strip().lower()

    if mode in ("s", "sec", "secs", "seconds"):
        mode = "second"
    elif mode in ("tu", "time", "timeunit", "time_units"):
        mode = "time_unit"
    elif mode in ("step", "index", "unitless"):
        mode = "step"

    if mode not in ("step", "time_unit", "second"):
        mode = "second"

    return mode

# ---------------------------------------------------------------------
# Settlement-kind adaptation
# ---------------------------------------------------------------------
def settlement_state_for_pde(
    s_pred_si: Tensor,
    t: Tensor,
    *,
    scaling_kwargs: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Tensor]] = None,
    time_units: Optional[str] = None,
    baseline_keys: Sequence[str] = (
        "s0_si", "subs0_si", "s_ref_si", "subs_ref_si",
    ),
    dt: Optional[Tensor] = None,                 
    return_incremental: bool = True,            
    verbose: int = 0,
) -> Tensor:
    """Map model output to settlement state in meters."""
    sk = scaling_kwargs or {}
    kind = str(
        get_sk(sk, "subsidence_kind", default="cumulative")
        ).strip().lower()

    s = tf_cast(s_pred_si, tf_float32)
    if getattr(s, "shape", None) is not None and s.shape.rank == 2:
        s = s[:, :, None]

    # --- baseline s0 (SI meters) ---
    s0 = None
    if inputs is not None:
        for k in baseline_keys:
            if (k in inputs) and (inputs[k] is not None):
                s0 = tf_cast(inputs[k], tf_float32)
                r = tf_rank(s0)
                s0 = tf_cond(
                    tf_equal(r, 1),
                    lambda: s0[:, None, None],
                    lambda: tf_cond(
                        tf_equal(r, 2),
                        lambda: s0[:, :, None],
                        lambda: s0,
                    ),
                )
                break
    if s0 is None:
        s0 = tf_zeros_like(s[:, :1, :])


    vprint(verbose, "settlement_kind=", kind)
    vprint(verbose, "s_pred_si=", s)
    vprint(verbose, "s0=", s0)
    vprint(verbose, "time_units=", time_units)

    # -------------------------------------------------------------
    # Build cumulative series s_cum(t) first (same shape as s)
    # -------------------------------------------------------------
    if kind == "cumulative":
        s_cum = s  # may include baseline (as in call(): s0_cum + s_inc)

    elif kind == "increment":
        # s is Δs per step
        s_cum = s0 + tf_cumsum(s, axis=1)

    elif kind == "rate":
        # s is ds/dt (meters / time_unit)
        if dt is not None:
            dtt = tf_cast(dt, tf_float32)
            if getattr(dtt, "shape", None) is not None and dtt.shape.rank == 2:
                dtt = dtt[:, :, None]
            ds = s * dtt
        else:
            tt = tf_cast(t, tf_float32)
            if getattr(tt, "shape", None) is not None and tt.shape.rank == 2:
                tt = tt[:, :, None]
            dtn = tt[:, 1:, :] - tt[:, :-1, :]
            # fallback default for first step (kept for backward compat)
            dt0 = tf_zeros_like(tt[:, :1, :]) + 1.0
            ds = s * tf_concat([dt0, dtn], axis=1)
        s_cum = s0 + tf_cumsum(ds, axis=1)

        vprint(verbose, "t=", tt)
        vprint(verbose, "ds=", ds)

    else:
        raise ValueError(
            f"Unsupported subsidence_kind={kind!r}. "
            "Use one of {'cumulative','increment','rate'}."
        )

    # -------------------------------------------------------------
    # Return incremental ODE state if requested: s_inc(t)=s_cum(t)-s0
    # -------------------------------------------------------------
    if return_incremental:
        s0H = s0 + tf_zeros_like(s_cum)  # broadcast to (B,H,1)
        return s_cum - s0H

    return s_cum

def to_rms(
    x: Tensor,
    *,
    axis: AxisLike = None,
    keepdims: bool = False,
    eps: Optional[float] = None,
    ms_floor: float | None = None,
    rms_floor: Optional[float] = None,
    nan_policy: str = "propagate",
    dtype: Any = None,
) -> Tensor:
    """Root-mean-square (RMS) of a tensor.

    Notes
    -----
    - Default dtype is float32 (fast). Pass dtype=tf_float64
      for diagnostics when values are extremely small.
    - By default (eps=None and ms_floor=None), NO flooring is
      applied. Floors are opt-in.
    - nan_policy:
      - "propagate": NaN/Inf propagate naturally.
      - "raise": assert all finite before reduce.
      - "omit": ignore non-finite entries in RMS.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int | Sequence[int] | None
        Reduction axis/axes. If None, reduce all.
    keepdims : bool
        Keep reduced dimensions.
    eps : float | None
        Optional lower bound on the mean-square before sqrt.
        If None, disabled.
    ms_floor : float | None
        Alias for a mean-square floor (applied after eps).
        If None, disabled.
    rms_floor : float | None
        Optional lower bound on RMS after sqrt. If None, disabled.
    nan_policy : {"propagate","raise","omit"}
        NaN/Inf handling policy.
    dtype : tf.DType | None
        Compute dtype. If None, uses tf_float32.

    Returns
    -------
    rms : Tensor
        RMS value (scalar if axis=None; else reduced).
    """
    pol = str(nan_policy or "propagate").strip().lower()
    if pol not in ("propagate", "raise", "omit"):
        pol = "propagate"

    dt = tf_float32 if dtype is None else dtype
    x = tf_cast(x, dt)

    if pol == "raise":
        tf_debugging.assert_all_finite(
            x,
            "to_rms(): x has NaN/Inf",
        )
        ms = tf_reduce_mean(
            tf_square(x),
            axis=axis,
            keepdims=keepdims,
        )

    elif pol == "omit":
        finite = tf_math.is_finite(x)
        x0 = tf_where(finite, x, tf_zeros_like(x))

        num = tf_reduce_sum(
            tf_square(x0),
            axis=axis,
            keepdims=keepdims,
        )
        den = tf_reduce_sum(
            tf_cast(finite, dt),
            axis=axis,
            keepdims=keepdims,
        )
        den = tf_maximum(
            den,
            tf_constant(1.0, dt),
        )
        ms = num / den

    else:  # propagate
        ms = tf_reduce_mean(
            tf_square(x),
            axis=axis,
            keepdims=keepdims,
        )

    # mean-square floors (opt-in)
    if eps is not None and float(eps) > 0.0:
        ms = tf_maximum(
            ms,
            tf_constant(float(eps), dt),
        )

    if ms_floor is not None and float(ms_floor) > 0.0:
        ms = tf_maximum(
            ms,
            tf_constant(float(ms_floor), dt),
        )

    rms = tf_sqrt(ms)

    if rms_floor is not None and float(rms_floor) > 0.0:
        rms = tf_maximum(
            rms,
            tf_constant(float(rms_floor), dt),
        )

    return rms

def _as_bool(x: Any, default: bool = False) -> bool:
    """Parse bool-like values robustly (bool/int/str)."""
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(default)

def _cast_lower_str(v):
    return str(v).strip().lower()

def _cast_optional_float(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"", "none", "null"}:
            return None
    return float(v)

def resolve_cons_drawdown_options(
    scaling_kwargs,
    *,
    default_mode: str = "smooth_relu",
    default_rule: str = "ref_minus_mean",
    default_stop_grad_ref: bool = True,
    default_zero_at_origin: bool = False,
    default_clip_max: Optional[float] = None,
    default_relu_beta: float = 20.0,
) -> Dict[str, Any]:
    """Resolve consolidation drawdown options from scaling_kwargs.

    Supported keys (prefer the 'cons_*' names):
    - cons_drawdown_mode / drawdown_mode
    - cons_drawdown_rule / drawdown_rule
    - cons_stop_grad_ref / stop_grad_ref
    - cons_drawdown_zero_at_origin / drawdown_zero_at_origin
    - cons_drawdown_clip_max / drawdown_clip_max
    - cons_relu_beta / relu_beta

    Returns
    -------
    dict with keys:
      drawdown_mode, drawdown_rule, stop_grad_ref,
      drawdown_zero_at_origin, drawdown_clip_max, relu_beta
    """
    sk = scaling_kwargs or {}

    mode = get_sk(
        sk, "cons_drawdown_mode",
        default=default_mode,
        cast=_cast_lower_str,
    )
    rule = get_sk(
        sk, "cons_drawdown_rule",
        default=default_rule,
        cast=_cast_lower_str,
    )
    stopg = get_sk(
        sk, "cons_stop_grad_ref",
        default=default_stop_grad_ref,
        cast=lambda x: _as_bool(x, default_stop_grad_ref),
    )
    zero0 = get_sk(
        sk, "cons_drawdown_zero_at_origin",
        default=default_zero_at_origin,
        cast=lambda x: _as_bool(x, default_zero_at_origin),
    )
    clipm = get_sk(
        sk, "cons_drawdown_clip_max",
        default=default_clip_max,
        cast=_cast_optional_float,
    )
    beta = get_sk(
        sk, "cons_relu_beta",
        default=default_relu_beta,
        cast=float,
    )

    allowed_modes = {"smooth_relu", "relu", "softplus", "none"}
    allowed_rules = {"ref_minus_mean", "mean_minus_ref"}

    if mode not in allowed_modes:
        pol = _cast_lower_str(get_sk(sk, "scaling_error_policy", default="raise"))
        if pol == "raise":
            raise ValueError(
                f"drawdown_mode must be {sorted(allowed_modes)}; got {mode!r}")
        mode = default_mode

    if rule not in allowed_rules:
        pol = _cast_lower_str(get_sk(
            sk, "scaling_error_policy", default="raise"))
        if pol == "raise":
            raise ValueError(
                f"drawdown_rule must be {sorted(allowed_rules)}; got {rule!r}")
        rule = default_rule

    return {
        "drawdown_mode": mode,
        "drawdown_rule": rule,
        "stop_grad_ref": stopg,
        "drawdown_zero_at_origin": zero0,
        "drawdown_clip_max": clipm,
        "relu_beta": beta,
    }

# ---------------------------------------
# Helpers 
# ---------------------------------------

def normalize_time_units(u: Optional[str]) -> str:
    """Normalize time unit strings."""
    if u is None:
        return "unitless"

    s = str(u).strip().lower().replace(" ", "")
    if "/" in s:
        s = s.split("/", 1)[1]
    if s.startswith("1/"):
        s = s[2:]

    if s == "secs":
        s = "sec"
    elif s == "yrs":
        s = "yr"
    elif s == "mins":
        s = "min"
    elif s == "hrs":
        s = "hr"

    return s


def seconds_per_time_unit(
    time_units: Optional[str],
    *,
    dtype=tf_float32,
) -> Tensor:
    """Seconds-per-unit."""
    key = normalize_time_units(time_units)

    if key not in TIME_UNIT_TO_SECONDS:
        keys = sorted(TIME_UNIT_TO_SECONDS.keys())
        raise ValueError(
            f"Unsupported time_units={time_units!r}. "
            f"Supported: {keys}"
        )

    return tf_constant(float(TIME_UNIT_TO_SECONDS[key]), dtype=dtype)


# ---------------------------------------------------------------------
# v3.2 helpers: physics-driven mean settlement via stable stepping
# ---------------------------------------------------------------------

def ensure_3d(x: Tensor) -> Tensor:
    """
    Return a rank-3 tensor, preferring static rank when available.

    Rules
    -----
    r=0 -> (1,1,1)
    r=1 -> (1,N,1)
    r=2 -> (B,H,1)
    r=3 -> unchanged
    """
    x = tf_convert_to_tensor(x)
    r_static = x.shape.rank

    # --- Fast path: static rank known (works great with KerasTensors) ---
    if r_static is not None:
        if r_static == 0:
            # scalar -> (1,1,1)
            return tf_reshape(x, [1, 1, 1])
        if r_static == 1:
            # (B,) -> (B,1,1)
            n = tf_shape(x)[0]
            return tf_reshape(x, [1, n, 1])
        if r_static == 2:
            return tf_expand_dims(x, axis=-1)
        if r_static == 3:
            return x
        raise ValueError(f"_ensure_3d: rank {r_static} not supported")

    # --- Fallback: dynamic rank (only if static is unknown) ---
    r = tf_rank(x)

    def r0():
        return tf_reshape(x, [1, 1, 1])

    def r1():
        n = tf_shape(x)[0]
        return tf_reshape(x, [1, n, 1])

    def r2():
        # (B,H) -> (B,H,1)
        return tf_expand_dims(x, axis=-1)

    def r3():
        # already (B,H,1)
        return x

    x = tf_cond(tf_equal(r, 0), r0, lambda: x)
    x = tf_cond(tf_equal(tf_rank(x), 1), r1, lambda: x)
    x = tf_cond(tf_equal(tf_rank(x), 2), r2, lambda: x)
    tf_debugging.assert_equal(
        tf_rank(x), 3, 
        message="_ensure_3d must return rank-3"
    )
    return x

def _ensure_3d(x: Tensor) -> Tensor:
    """Ensure (B,T,1) shape."""
    if getattr(x, "shape", None) is not None and x.shape.rank == 2:
        return x[:, :, None]
    return x


def _broadcast_like(x: Optional[Tensor], like: Tensor) -> Tensor:
    """Convert and broadcast x to the shape of `like` (dtype preserved)."""
    if x is None:
        return tf_zeros_like(like)
    xt = tf_convert_to_tensor(x, dtype=like.dtype)
    return tf_broadcast_to(xt, tf_shape(like))

def dt_to_seconds(dt: Tensor, *, time_units: Optional[str]) -> Tensor:
    """dt(time_units) -> seconds."""
    dt = tf_convert_to_tensor(dt)
    dt = tf_cast(dt, tf_float32)
    dt = _finite_or_zero(dt)  # NaN/Inf -> 0
    dt = tf_maximum(dt, tf_constant(0.0, dt.dtype))  # no negative dt
    sec = seconds_per_time_unit(time_units, dtype=dt.dtype)
    return dt * sec

def rate_to_per_second(
    dz_dt: Tensor,
    *,
    time_units: Optional[str],
) -> Tensor:
    """d/d(time_units) -> d/ds."""
    sec = seconds_per_time_unit(time_units, dtype=dz_dt.dtype)
    return dz_dt / (sec + tf_constant(_EPSILON, dz_dt.dtype))


def smooth_relu(x: Tensor, *, beta: float = 20.0) -> Tensor:
    """Smooth approximation to relu(x) with controlled curvature."""
    b = tf_constant(float(beta), dtype=x.dtype)
    return tf_softplus(b * x) / b


def positive(x: Tensor, *, eps: float = _EPSILON) -> Tensor:
    """Softplus positivity."""
    return tf_softplus(x) + tf_constant(eps, x.dtype)

def _stats(name: str, x: Tensor) -> None:
    x = tf_cast(x, tf_float32)
    tf_print(
        name,
        "shape=", tf_shape(x),
        "min=", tf_reduce_min(x),
        "mean=", tf_reduce_mean(x),
        "max=", tf_reduce_max(x),
        summarize=8,
    )

def _frac_leq_zero(x: Tensor) -> Tensor:
    x = tf_cast(x, tf_float32)
    return tf_reduce_mean(tf_cast(x <= 0.0, tf_float32))

def _assert_grads_finite(
    grads: list[Tensor | None],
    vars_: list[Tensor],
) -> None:
    for g, v in zip(grads, vars_):
        if g is None:
            continue
        tf_debugging.assert_all_finite(
            g,
            f"NaN/Inf grad for {v.name}",
        )