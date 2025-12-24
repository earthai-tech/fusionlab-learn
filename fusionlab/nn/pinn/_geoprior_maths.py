
# -*- coding: utf-8 -*-
# _geoprior_maths.py
"""
GeoPrior maths helpers (physics terms + scaling).
Short docs only; full docs later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Sequence

import numpy as np

from .. import KERAS_DEPS, dependency_message
from ...compat.tf import optional_tf_function
from ...utils.generic_utils import vlog 
from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.docs import DocstringComponents, _halnet_core_params

# ---------------------------------------------------------------------
# Keras deps aliases (keep short lines for linting)
# ---------------------------------------------------------------------
Tensor = KERAS_DEPS.Tensor
Dataset = KERAS_DEPS.Dataset
GradientTape = KERAS_DEPS.GradientTape

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
register_keras_serializable = KERAS_DEPS.register_keras_serializable
deserialize_keras_object = KERAS_DEPS.deserialize_keras_object


tf_autograph = getattr(KERAS_DEPS, "autograph", None)
if tf_autograph is not None:
    tf_autograph.set_verbosity(0)


DEP_MSG = dependency_message("nn.pinn.models")

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_halnet_core_params)
)


SMALL = 1e-12


def vprint(verbose: int, *args) -> None:
    """Verbose print (eager-friendly)."""
    if int(verbose) > 0:
        print(*args)

# ---------------------------------------------------------------------
# Physics residuals / priors
# ---------------------------------------------------------------------
def resolve_mv_gamma_si(
    model,
    Ss_field: Tensor,
    *,
    eps: float = 1e-12,
    verbose: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Return (mv_si [Pa^-1], gamma_w_si [Pa/m]) with auto unit alignment.

    We test a small set of plausible conversions and choose the pair
    that best matches the domain-mean of log(Ss).

    Candidates:
      - (mv, gw)                      : assume both already SI
      - (mv/1000, gw)                 : mv provided in 1/kPa, gw in Pa/m
      - (mv, gw*1000)                 : mv in 1/Pa, gw provided in kPa/m
      - (mv/1000, gw*1000)            : mv in 1/kPa and gw in kPa/m

    Notes
    -----
    This avoids requiring `mv_units` / `gamma_w_units` in config and
    works even when people forget to set them.
    """
    eps_t = tf_constant(float(eps), tf_float32)

    Ss_safe = tf_maximum(tf_cast(Ss_field, tf_float32), eps_t)
    logSs_mean = tf_reduce_mean(tf_log(Ss_safe))  # scalar

    mv_raw = tf_maximum(tf_cast(model._mv_value(), tf_float32), eps_t)
    gw_raw = tf_maximum(tf_cast(getattr(model, "gamma_w", mv_raw * 0 + 9810.0),
                                tf_float32), eps_t)

    # --- candidates (all scalars) ------------------------------------
    mv_cands = tf_stack([
        mv_raw,
        mv_raw / tf_constant(1000.0, tf_float32),
        mv_raw,
        mv_raw / tf_constant(1000.0, tf_float32),
    ])
    gw_cands = tf_stack([
        gw_raw,
        gw_raw,
        gw_raw * tf_constant(1000.0, tf_float32),
        gw_raw * tf_constant(1000.0, tf_float32),
    ])

    log_targets = tf_log(mv_cands) + tf_log(gw_cands)     # (4,)
    errs = tf_abs(logSs_mean - log_targets)               # (4,)

    idx = tf_cast(tf_argmin(errs, axis=0), tf_int32)

    mv_sel = mv_cands[idx]
    gw_sel = gw_cands[idx]

    vprint(verbose, "resolve_mv_gamma_si:")
    vprint(verbose, "  mv_raw=", mv_raw, " gw_raw=", gw_raw)
    vprint(verbose, "  logSs_mean=", logSs_mean)
    vprint(verbose, "  log_targets=", log_targets, " errs=", errs, " idx=", idx)
    vprint(verbose, "  mv_sel=", mv_sel, " gw_sel=", gw_sel)

    return mv_sel, gw_sel

def compute_mv_prior(
    model,
    Ss_field: Tensor,
    *,
    reduction: str = "domain_mean",
    as_loss: bool = False,
    verbose: int = 0,
) -> Tensor:
    """Specific-storage identity prior in log-space.

    Encodes: Ss ≈ mv · gamma_w.

    Option A (default): compare scalar mv to the domain-mean of log(Ss),
    instead of penalizing every pixel.
    Option B: when `as_loss=True`, return RMS magnitude (log-units),
    which is more stable than MSE (squared log-units).
    """
    eps = tf_constant(1e-12, dtype=tf_float32)
    Ss_safe = tf_maximum(tf_cast(Ss_field, tf_float32), eps)

    mv, gw = resolve_mv_gamma_si(model, Ss_field, verbose=verbose)

    log_target = tf_log(mv) + tf_log(gw)

    red = str(reduction).strip().lower()
    if red in ("domain_mean", "mean", "global"):
        logSs = tf_reduce_mean(tf_log(Ss_safe))  # scalar
    elif red in ("field", "none", "per_pixel", "pixel"):
        logSs = tf_log(Ss_safe)                  # field
    else:
        raise ValueError(
            "compute_mv_prior(reduction=...) must be one of "
            "{'domain_mean', 'field'}. Got: %r" % reduction
        )

    res = logSs - log_target  # signed residual (scalar or field)

    if as_loss:
        # RMS in log-units (preferred scale for weighting).
        out = to_rms(res)
    else:
        out = res

    vprint(verbose, "mv_prior: reduction=", red)
    vprint(verbose, "mv_prior: mv=", mv, "gw=", gw)
    vprint(verbose, "mv_prior: log_target=", log_target)
    vprint(verbose, "mv_prior: out=", out)
    return out

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
    """Groundwater flow PDE residual."""
    if "gw_flow" not in model.pde_modes_active:
        return tf_zeros_like(dh_dt)

    # NOTE: v3.2: Q can be a learned forcing field (Tensor) or a scalar.
    if Q is None:
        Qv = tf_zeros_like(dh_dt)
    else:
        Qv = tf_convert_to_tensor(Q, dtype=dh_dt.dtype)
        # Broadcast scalars / rank-1/2 to match dh_dt.
        Qv = tf_broadcast_to(Qv, tf_shape(dh_dt))

    div_K_grad_h = d_K_dh_dx_dx + d_K_dh_dy_dy
    storage_term = Ss_field * dh_dt

    out = storage_term - div_K_grad_h - Qv

    vprint(verbose, "gw: dh_dt=", dh_dt)
    vprint(verbose, "gw: div=", div_K_grad_h)
    vprint(verbose, "gw: out=", out)

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
    
    from ._geoprior_utils import get_h_ref_si 
    
    if "consolidation" not in model.pde_modes_active:
        return tf_zeros_like(ds_dt)

    eps = tf_constant(1e-12, dtype=tf_float32)
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


# ---------------------------------------------------------------------
# v3.2 helpers: physics-driven mean settlement via stable stepping
# ---------------------------------------------------------------------
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

def equilibrium_compaction_si(
    *,
    h_mean_si: Tensor,
    h_ref_si: Tensor,
    Ss_field: Tensor,
    H_field_si: Tensor,
    use_relu: bool = True,
    verbose: int = 0,
) -> Tensor:
    """Compute equilibrium compaction s_eq in SI meters."""
    h_mean_si = _ensure_3d(tf_cast(h_mean_si, tf_float32))
    h_ref_si = _broadcast_like(_ensure_3d(tf_cast(h_ref_si, tf_float32)), h_mean_si)
    Ss_field = _broadcast_like(_ensure_3d(Ss_field), h_mean_si)
    H_field_si = _broadcast_like(_ensure_3d(H_field_si), h_mean_si)

    vprint(
        verbose,
        "[equilibrium_compaction_si] shapes:",
        "h_mean", h_mean_si.shape,
        "h_ref", h_ref_si.shape,
        "Ss", Ss_field.shape,
        "H", H_field_si.shape,
        "| use_relu=", use_relu,
    )

    if use_relu:
        delta_h = tf_maximum(h_ref_si - h_mean_si, 0.0)
    else:
        delta_h = h_ref_si - h_mean_si

    vprint(
        verbose,
        "[equilibrium_compaction_si] delta_h stats:",
        "min=", tf_reduce_min(delta_h),
        "max=", tf_reduce_max(delta_h),
        "mean=", tf_reduce_mean(delta_h),
    )

    s_eq = Ss_field * delta_h * H_field_si

    vprint(
        verbose,
        "[equilibrium_compaction_si] s_eq stats:",
        "min=", tf_reduce_min(s_eq),
        "max=", tf_reduce_max(s_eq),
        "mean=", tf_reduce_mean(s_eq),
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
    h_mean_si = _ensure_3d(tf_cast(h_mean_si, tf_float32))
    B = tf_shape(h_mean_si)[0]
    H = tf_shape(h_mean_si)[1]

    vprint(
        verbose,
        "[integrate_consolidation_mean] B,H =",
        B, H,
        "| time_units=", time_units,
        "| method=", method,
    )

    # --- dt in seconds (same shape as horizon) ---
    if dt is None:
        dt = tf_zeros_like(h_mean_si) + 1.0
        vprint(verbose, "[integrate_consolidation_mean] dt=None -> using 1.0 per step")
    else:
        dt = _broadcast_like(_ensure_3d(tf_cast(dt, tf_float32)), h_mean_si)

    dt_sec = dt_to_seconds(dt, time_units=time_units)

    vprint(
        verbose,
        "[integrate_consolidation_mean] dt_sec stats:",
        "min=", tf_reduce_min(dt_sec),
        "max=", tf_reduce_max(dt_sec),
        "mean=", tf_reduce_mean(dt_sec),
    )

    # --- fields, broadcast to horizon ---
    tau = _broadcast_like(_ensure_3d(tf_cast(tau_field, tf_float32)), h_mean_si)
    tau = tf_maximum(tau, tf_constant(eps_tau, dtype=tf_float32))

    vprint(
        verbose,
        "[integrate_consolidation_mean] tau stats:",
        "min=", tf_reduce_min(tau),
        "max=", tf_reduce_max(tau),
        "mean=", tf_reduce_mean(tau),
    )

    s_eq = equilibrium_compaction_si(
        h_mean_si=h_mean_si,
        h_ref_si=h_ref_si,
        Ss_field=Ss_field,
        H_field_si=H_field_si,
        use_relu=True,
        verbose=verbose,
    )

    method = str(method).strip().lower()
    if method not in {"exact", "euler"}:
        raise ValueError(
            "integrate_consolidation_mean: method must be 'exact' or 'euler'."
        )

    # --- scan over time axis (time-major) ---
    # s0 = _ensure_3d(tf_cast(s_init_si, tf_float32))
    # s0 = tf_broadcast_to(s0, [B, 1, 1])
    # s0_2d = s0[:, 0, :]  # (B,1)
    s0 = _ensure_3d(tf_cast(s_init_si, tf_float32))
    s0 = s0[:, :1, :]          # <-- safe even if already (B,1,1)
    s0 = tf_broadcast_to(s0, [B, 1, 1])
    s0_2d = s0[:, 0, :]  # (B,1)

    vprint(
        verbose,
        "[integrate_consolidation_mean] s_init stats:",
        "min=", tf_reduce_min(s0_2d),
        "max=", tf_reduce_max(s0_2d),
        "mean=", tf_reduce_mean(s0_2d),
    )

    if tf_transpose is None or tf_scan is None:
        raise RuntimeError(
            "TensorFlow ops 'transpose'/'scan' missing from KERAS_DEPS. "
            "Check backend initialization."
        )

    dt_tm = tf_transpose(dt_sec, [1, 0, 2])  # (H,B,1)
    tau_tm = tf_transpose(tau, [1, 0, 2])
    seq_tm = tf_transpose(s_eq, [1, 0, 2])

    def step(prev: Tensor, elems: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        dt_i, tau_i, seq_i = elems  # each (B,1)
        if method == "exact":
            a = tf_exp(-dt_i / (tau_i + tf_constant(SMALL, tau_i.dtype)))
            nxt = prev * a + seq_i * (1.0 - a)
        else:
            nxt = prev + dt_i * (seq_i - prev) / (
                tau_i + tf_constant(SMALL, tau_i.dtype)
            )
        return nxt

    s_tm = tf_scan(
        fn=step,
        elems=(dt_tm, tau_tm, seq_tm),
        initializer=s0_2d,
    )

    s_bar = tf_transpose(s_tm, [1, 0, 2])

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
    verbose: int = 0,
) -> Tensor:
    """One-step consolidation residual in SI space.

    This is useful when `s_state_si` is produced by a network and you want
    to *enforce* the consolidation ODE via a stable stepper.

    Returns a residual for each step n -> n+1: shape (B,T-1,1).
    """
    s_state_si = _ensure_3d(tf_cast(s_state_si, tf_float32))
    h_mean_si = _ensure_3d(tf_cast(h_mean_si, tf_float32))

    T = tf_shape(s_state_si)[1]
    vprint(verbose, "[compute_consolidation_step_residual] T =", T, "| method=", method)

    if dt is None:
        dt = tf_zeros_like(s_state_si[:, 1:, :]) + 1.0
        vprint(verbose, "[compute_consolidation_step_residual] dt=None -> using 1.0 per step")
    else:
        dt = _broadcast_like(
            _ensure_3d(tf_cast(dt, tf_float32)),
            s_state_si[:, 1:, :],
        )

    dt_sec = dt_to_seconds(dt, time_units=time_units)

    # Align to n=0..T-2
    s_n = s_state_si[:, :-1, :]
    s_np1 = s_state_si[:, 1:, :]
    h_n = h_mean_si[:, :-1, :]

    tau = _broadcast_like(_ensure_3d(tf_cast(tau_field, tf_float32)), s_n)
    tau = tf_maximum(tau, tf_constant(eps_tau, dtype=tf_float32))

    s_eq_n = equilibrium_compaction_si(
        h_mean_si=h_n,
        h_ref_si=h_ref_si,
        Ss_field=Ss_field,
        H_field_si=H_field_si,
        use_relu=True,
        verbose=verbose,
    )

    method = str(method).strip().lower()
    if method == "exact":
        a = tf_exp(-dt_sec / (tau + tf_constant(SMALL, tau.dtype)))
        pred = s_n * a + s_eq_n * (1.0 - a)
    elif method == "euler":
        pred = s_n + dt_sec * (s_eq_n - s_n) / (tau + tf_constant(SMALL, tau.dtype))
    else:
        raise ValueError(
            "compute_consolidation_step_residual: method must be 'exact' or 'euler'."
        )

    res = s_np1 - pred

    vprint(
        verbose,
        "[compute_consolidation_step_residual] residual stats:",
        "min=", tf_reduce_min(res),
        "max=", tf_reduce_max(res),
        "mean=", tf_reduce_mean(res),
    )
    return res


def tau_phys_from_fields(
    model,
    K_field: Tensor,
    Ss_field: Tensor,
    H_field: Tensor,
    *,
    verbose: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Compute tau_phys and Hd."""
    epsK = tf_constant(1e-12, dtype=tf_float32)
    eps = tf_constant(1e-12, dtype=tf_float32)
    pi_sq = tf_constant(np.pi**2, dtype=tf_float32)

    K_safe = tf_maximum(K_field, epsK)
    Ss_safe = tf_maximum(Ss_field, eps)
    H_safe = tf_maximum(H_field, eps)

    if bool(model.use_effective_thickness):
        Hd = H_safe * tf_cast(model.Hd_factor, H_safe.dtype)
    else:
        Hd = H_safe

    Hd = tf_maximum(Hd, eps)
    ratio = Hd / H_safe

    if str(model.kappa_mode) == "bar":
        tau_phys = (
            model._kappa_value()
            * (H_safe ** 2)
            * Ss_safe
            / (pi_sq * K_safe)
        )
    else:
        tau_phys = (
            (ratio ** 2)
            * (H_safe ** 2)
            * Ss_safe
            / (pi_sq * model._kappa_value() * K_safe)
        )

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
    eps = tf_constant(1e-12, dtype=tf_float32)

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
    verbose: int = 0,
) -> Tensor:
    """Smoothness prior on (log) fields."""
    eps = tf_constant(1e-12, dtype=tf_float32)

    if (K_field is not None) and (Ss_field is not None):
        dlogK_dx = dK_dx / (K_field + eps)
        dlogK_dy = dK_dy / (K_field + eps)
        dlogSs_dx = dSs_dx / (Ss_field + eps)
        dlogSs_dy = dSs_dy / (Ss_field + eps)

        out = (
            tf_square(dlogK_dx)
            + tf_square(dlogK_dy)
            + tf_square(dlogSs_dx)
            + tf_square(dlogSs_dy)
        )
        vprint(verbose, "smooth(log): out=", out)
        return out

    out = (
        tf_square(dK_dx)
        + tf_square(dK_dy)
        + tf_square(dSs_dx)
        + tf_square(dSs_dy)
    )
    vprint(verbose, "smooth: out=", out)
    return out


# ---------------------------------------------------------------------
# Bounds + field composition
# ---------------------------------------------------------------------
def get_log_bounds(
    model,
    *,
    as_tensor: bool = True,
    dtype=tf_float32,
    verbose: int = 0,
) -> Tuple[Any, Any, Any, Any]:
    """Get (logK_min,logK_max,logSs_min,logSs_max)."""
    b = (getattr(model, "scaling_kwargs", {}) or {}).get(
        "bounds",
        {},
    ) or {}

    def get_pair(
        log_min_key: str,
        log_max_key: str,
        lin_min_key: str,
        lin_max_key: str,
    ):
        log_min = b.get(log_min_key, None)
        log_max = b.get(log_max_key, None)

        if (log_min is None) or (log_max is None):
            if (lin_min_key in b) and (lin_max_key in b):
                vmin = float(b[lin_min_key])
                vmax = float(b[lin_max_key])

                if lin_min_key == "Ss_min" and lin_max_key == "Ss_max":
                    try:
                        gw = getattr(model, "gamma_w", None)
                        gw = float(gw.numpy()) if hasattr(
                            gw,
                            "numpy",
                        ) else float(gw)

                        mv0 = getattr(
                            getattr(model, "mv_config", None),
                            "initial_value",
                            None,
                        )
                        mv0 = float(mv0) if mv0 is not None else None
                        ss_exp = (mv0 * gw) if mv0 else None

                        looks_mv = (vmax <= 1e-5) and (gw > 1e3)
                        if looks_mv and (
                            ss_exp is None or ss_exp > 1e-5
                        ):
                            vmin *= gw
                            vmax *= gw
                            logger.warning(
                                "Ss_min/max look like m_v; "
                                "convert via Ss=m_v*gamma_w."
                            )
                    except Exception:
                        pass

                return float(np.log(vmin)), float(np.log(vmax))

            return None, None

        return float(log_min), float(log_max)

    logK_min, logK_max = get_pair(
        "logK_min",
        "logK_max",
        "K_min",
        "K_max",
    )
    logSs_min, logSs_max = get_pair(
        "logSs_min",
        "logSs_max",
        "Ss_min",
        "Ss_max",
    )

    if (logK_min is None) or (logSs_min is None):
        return (None, None, None, None)

    if not as_tensor:
        return logK_min, logK_max, logSs_min, logSs_max

    out = (
        tf_constant(logK_min, dtype),
        tf_constant(logK_max, dtype),
        tf_constant(logSs_min, dtype),
        tf_constant(logSs_max, dtype),
    )

    vprint(verbose, "bounds: out=", out)
    return out


def bounded_exp(
    raw: Tensor,
    log_min: Tensor,
    log_max: Tensor,
    *,
    eps: float = 1e-12,
    return_log: bool = False,
    verbose: int = 0,
):
    """Exp with hard log-bounds."""
    z = tf_sigmoid(raw)
    logv = log_min + z * (log_max - log_min)
    out = tf_exp(logv) + tf_constant(eps, tf_float32)

    vprint(verbose, "bounded_exp: logv=", logv)
    vprint(verbose, "bounded_exp: out=", out)

    if return_log:
        return out, logv
    return out


def compose_physics_fields(
    model,
    *,
    coords_flat: Tensor,
    H_si: Tensor,
    K_base: Tensor,
    Ss_base: Tensor,
    tau_base: Tensor,
    training: bool = False,
    eps_KSs: float = 1e-12,
    eps_tau: float = 1e-6,
    verbose: int = 0,
):
    """Compose K,Ss,tau fields with coord MLPs."""
    coords_xy0 = tf_concat(
        [tf_zeros_like(coords_flat[..., :1]),
         coords_flat[..., 1:]],
        axis=-1,
    )

    K_corr = model.K_coord_mlp(coords_xy0, training=training)
    Ss_corr = model.Ss_coord_mlp(coords_xy0, training=training)
    tau_corr = model.tau_coord_mlp(coords_xy0, training=training)

    rawK = K_base + K_corr
    rawSs = Ss_base + Ss_corr

    bounds_mode = getattr(model, "bounds_mode", "soft")
    if str(bounds_mode) == "hard":
        logK_min, logK_max, logSs_min, logSs_max = get_log_bounds(
            model,
            as_tensor=True,
            dtype=rawK.dtype,
            verbose=0,
        )

        K_field, logK = bounded_exp(
            rawK,
            logK_min,
            logK_max,
            eps=eps_KSs,
            return_log=True,
            verbose=0,
        )
        Ss_field, logSs = bounded_exp(
            rawSs,
            logSs_min,
            logSs_max,
            eps=eps_KSs,
            return_log=True,
            verbose=0,
        )
    else:
        logK = rawK
        logSs = rawSs

        K_field = tf_exp(logK) + tf_constant(eps_KSs, logK.dtype)
        Ss_field = tf_exp(logSs) + tf_constant(eps_KSs, logSs.dtype)

    delta_log_tau = tau_base + tau_corr
    tau_phys, Hd_eff = tau_phys_from_fields(
        model,
        K_field,
        Ss_field,
        H_si,
        verbose=0,
    )
    tau_field = (
        tau_phys * tf_exp(delta_log_tau)
        + tf_constant(eps_tau, tau_phys.dtype)
    )

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
    )


def compute_bounds_residual(
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

    vprint(verbose, "bounds: R_H=", R_H)
    vprint(verbose, "bounds: R_K=", R_K)
    vprint(verbose, "bounds: R_Ss=", R_Ss)

    return R_H, R_K, R_Ss


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


@optional_tf_function
def dt_to_seconds(dt: Tensor, *, time_units: Optional[str]) -> Tensor:
    """dt(time_units) -> seconds."""
    sec = seconds_per_time_unit(time_units, dtype=dt.dtype)
    return dt * sec


@optional_tf_function
def rate_to_per_second(
    dz_dt: Tensor,
    *,
    time_units: Optional[str],
) -> Tensor:
    """d/d(time_units) -> d/ds."""
    sec = seconds_per_time_unit(time_units, dtype=dz_dt.dtype)
    return dz_dt / (sec + tf_constant(SMALL, dz_dt.dtype))


def positive(x: Tensor, *, eps: float = SMALL) -> Tensor:
    """Softplus positivity."""
    return tf_softplus(x) + tf_constant(eps, x.dtype)


def default_scales(
    *,
    s: Tensor,
    h: Tensor,
    dt: Tensor,
    K: Optional[Tensor] = None,
    Ss: Optional[Tensor] = None,
    Q: Optional[Tensor] = None,
    time_units: Optional[str] = "yr",
    eps: float = 1e-12,
    min_cons_rate: float = 1e-3,
    min_gw_scale: float = 1e-12,
    head_scale_floor: float = 1.0,
    verbose: int = 0,
) -> Dict[str, Tensor]:
    """Default PDE scaling."""
    if time_units is None:
        time_units = "yr"

    def flat(x: Tensor) -> Tensor:
        return tf_reshape(x, [-1])

    if getattr(s, "shape", None) is not None and s.shape.rank == 2:
        s = s[:, :, None]
    if getattr(h, "shape", None) is not None and h.shape.rank == 2:
        h = h[:, :, None]

    dt_sec = dt_to_seconds(dt, time_units=time_units)
    dt_ref = tf_reduce_mean(tf_abs(flat(dt_sec)))

    floor = seconds_per_time_unit(time_units)
    dt_ref = tf_maximum(dt_ref, tf_cast(floor, dt_ref.dtype))

    ds = s[:, 1:, :] - s[:, :-1, :]
    ds_abs = tf_abs(flat(ds))
    s_abs = tf_abs(flat(s))

    ds_mean = tf_reduce_mean(ds_abs)
    s_mean = tf_reduce_mean(s_abs)

    T = tf_shape(s)[1]
    Tm1 = tf_maximum(T - 1, 1)
    Tm1_f = tf_cast(Tm1, dt_ref.dtype)
    dt_total = dt_ref * Tm1_f

    rate_inc = ds_mean / dt_ref
    rate_amp = s_mean / dt_total

    rate_inc_max = tf_reduce_max(ds_abs) / dt_ref
    rate_amp_max = tf_reduce_max(s_abs) / dt_total

    cons_scale = tf_maximum(
        tf_maximum(rate_inc, rate_amp),
        0.1 * tf_maximum(rate_inc_max, rate_amp_max),
    )

    cons_floor = rate_to_per_second(
        tf_cast(min_cons_rate, cons_scale.dtype),
        time_units=time_units,
    )
    cons_scale = tf_maximum(cons_scale, cons_floor)
    cons_scale = tf_maximum(
        cons_scale,
        tf_cast(eps, cons_scale.dtype),
    )

    dh = h[:, 1:, :] - h[:, :-1, :]
    dh_abs = tf_abs(flat(dh))
    h_abs = tf_abs(flat(h))

    dh_mean = tf_reduce_mean(dh_abs)
    h_mean = tf_reduce_mean(h_abs)

    dh_dt_inc = dh_mean / dt_ref
    dh_dt_amp = h_mean / dt_total
    dh_dt_ref = tf_maximum(dh_dt_inc, dh_dt_amp)

    head_scale = tf_maximum(
        h_mean,
        tf_cast(head_scale_floor, dh_dt_ref.dtype),
    )

    if Ss is not None:
        Ss_ref = tf_reduce_mean(tf_abs(flat(Ss)))
    else:
        Ss_ref = 1.0 / head_scale

    gw_scale_1 = Ss_ref * dh_dt_ref
    gw_scale_2 = dh_dt_ref / head_scale
    gw_scale = tf_maximum(gw_scale_1, gw_scale_2)
    gw_scale = tf_maximum(
        gw_scale,
        tf_cast(min_gw_scale, gw_scale.dtype),
    )
    gw_scale = tf_maximum(
        gw_scale,
        tf_cast(eps, gw_scale.dtype),
    )

    # v3.2: include forcing magnitude when Q is present so GW residual
    # doesn't collapse to ~1e-9 due to unit/scale mismatch.
    if Q is not None:
        Qv = tf_convert_to_tensor(Q, dtype=tf_float32)
        Q_ref = tf_reduce_mean(tf_abs(flat(Qv)))
        gw_scale = tf_maximum(
            gw_scale,
            tf_cast(Q_ref, gw_scale.dtype) + tf_constant(eps, gw_scale.dtype),
        )

    out = {"cons_scale": cons_scale, "gw_scale": gw_scale}

    vprint(verbose, "scales:", out)
    return out


@optional_tf_function
def scale_residual(residual: Tensor, scale: Tensor) -> Tensor:
    """Residual / scale with safety."""
    return residual / (
        tf_cast(scale, residual.dtype)
        + tf_constant(SMALL, residual.dtype)
    )


def compute_scales(
    model,
    *,
    t: Tensor,
    s_mean: Tensor,
    h_mean: Tensor,
    K_field: Tensor,
    Ss_field: Tensor,
    ds_dt: Optional[Tensor] = None,
    tau_field: Optional[Tensor] = None,
    H_field: Optional[Tensor] = None,
    h_ref_si: Optional[Tensor] = None,
    Q: Optional[Tensor] = None,
    verbose: int = 0,
) -> Dict[str, Tensor]:
    """Wrapper: dt handling + tau-aware cons_scale."""
    from ._geoprior_utils import coord_ranges
    
    dt_tensor = None
    if hasattr(t, "shape") and t.shape.rank is not None:
        if (t.shape.rank >= 2) and (t.shape[1] is not None):
            if t.shape[1] > 1:
                dt_tensor = t[:, 1:, :] - t[:, :-1, :]

    if dt_tensor is None:
        if (s_mean.shape.rank is not None) and (
            s_mean.shape[1] is not None
        ) and (s_mean.shape[1] > 1):
            dt_tensor = tf_zeros_like(s_mean[:, 1:, :]) + 1.0
        else:
            dt_tensor = tf_zeros_like(s_mean[..., :1]) + 1.0

    coords_norm = bool(model.scaling_kwargs.get(
        "coords_normalized",
        False,
    ))
    tR, _, _ = coord_ranges(model.scaling_kwargs)
    if coords_norm and tR:
        dt_tensor = dt_tensor * tf_cast(tR, dt_tensor.dtype)

    time_units = (
        model.scaling_kwargs.get("time_units", None)
        or model.scaling_kwargs.get("time_unit", None)
        or model.time_units
    )

    scales = default_scales(
        s=s_mean,
        h=h_mean,
        dt=dt_tensor,
        K=K_field,
        Ss=Ss_field,
        Q=Q,
        time_units=time_units,
        verbose=0,
    )

    if (
        ds_dt is not None
        and tau_field is not None
        and H_field is not None
    ):
        eps = tf_constant(1e-12, tf_float32)

        href = h_ref_si
        if href is None:
            href = tf_cast(model.h_ref, h_mean.dtype)

        delta_h = tf_maximum(href - h_mean, 0.0)
        s_eq = Ss_field * delta_h * H_field
        relax = (s_eq - s_mean) / (tau_field + eps)

        term1 = tf_stop_gradient(
            tf_reduce_mean(tf_abs(ds_dt)) + eps
        )
        term2 = tf_stop_gradient(
            tf_reduce_mean(tf_abs(relax)) + eps
        )
        scales["cons_scale"] = term1 + term2

    vprint(verbose, "compute_scales:", scales)
    return scales

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
        "s0_si",
        "subs0_si",
        "s_ref_si",
        "subs_ref_si",
    ),
    verbose: int = 0,
) -> Tensor:
    """Map model output to cumulative settlement state s(t) in meters."""
    sk = scaling_kwargs or {}
    kind = str(sk.get("subsidence_kind", "cumulative"))
    kind = kind.strip().lower()

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

    # -----------------------------------------------------------------
    # cases
    # -----------------------------------------------------------------
    if kind == "cumulative":
        return s

    if kind == "increment":
        # s is Δs per step (meters)
        out = s0 + KERAS_DEPS.cumsum(s, axis=1)
        vprint(verbose, "s_state(increment)=", out)
        return out

    if kind == "rate":
        # s is ds/dt in meters / time_unit
        tt = tf_cast(t, tf_float32)
        if getattr(tt, "shape", None) is not None and tt.shape.rank == 2:
            tt = tt[:, :, None]

        dt = tt[:, 1:, :] - tt[:, :-1, :]
        dt0 = tf_zeros_like(tt[:, :1, :]) + 1.0
        dt_step = tf_concat([dt0, dt], axis=1)

        ds = s * dt_step
        out = s0 + KERAS_DEPS.cumsum(ds, axis=1)

        vprint(verbose, "t=", tt)
        vprint(verbose, "dt_step=", dt_step)
        vprint(verbose, "ds=", ds)
        vprint(verbose, "s_state(rate)=", out)

        return out

    raise ValueError(
        "Unsupported subsidence_kind="
        f"{kind!r}. Use one of "
        "{'cumulative','increment','rate'}."
    )
    

def to_rms(
    x: Tensor,
    *,
    axis=None,
    keepdims: bool = False,
    eps: float = 0.0,
) -> Tensor:
    """Root-mean-square (RMS) of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int | Sequence[int] | None
        Reduction axis/axes. If None, reduce over all elements.
    keepdims : bool
        Keep reduced dimensions.
    eps : float
        Optional lower bound on the mean-square before sqrt
        (useful to avoid sqrt(0) in some diagnostics).

    Returns
    -------
    rms : Tensor
        RMS value (scalar if axis=None; else reduced tensor).
    """
    x = tf_cast(x, tf_float32)
    ms = tf_reduce_mean(tf_square(x), axis=axis, keepdims=keepdims)
    if eps and float(eps) > 0.0:
        ms = tf_maximum(ms, tf_constant(float(eps), tf_float32))
    return tf_sqrt(ms)

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