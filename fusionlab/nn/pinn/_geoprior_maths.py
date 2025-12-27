
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
from ...compat.tf import optional_tf_function
from ...utils.generic_utils import vlog 
from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.docs import DocstringComponents, _halnet_core_params
from ._geoprior_utils import coord_ranges, get_sk, get_h_ref_si

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
tf_reduce_sum = KERAS_DEPS.reduce_sum 
tf_is_nan = KERAS_DEPS.is_nan 
tf_is_inf = KERAS_DEPS.is_inf 
tf_logical_or = KERAS_DEPS.logical_or 
tf_where = KERAS_DEPS.where 
tf_cumsum = KERAS_DEPS.cumsum 
tf_math = KERAS_DEPS.math 
tf_ones_like = KERAS_DEPS.ones_like 
tf_clip_by_value = KERAS_DEPS.clip_by_value

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


_SMALL = 1e-12

AxisLike = Optional[Union[int, Sequence[int]]]

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
    H_floor: float = 1e-6,
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
        Q_in_si = bool(get_sk(sk, "Q_in_si", default=True))
        if Q_in_per_second or Q_in_si:
            Q_per_s = Q_base
        else:
            Q_per_s = rate_to_per_second(Q_base, time_units=time_units)

        vprint(verbose, "Q_kind=per_volume, Q_term(1/s)=", Q_per_s)
        return Q_per_s

    # For the other kinds, interpret Q as a LENGTH RATE (m/time)
    # Use a *separate* flag so we don't conflict with Q_in_si default=True.
    Q_len_in_si = bool(get_sk(sk, "Q_length_in_si", "Q_in_m_per_s", default=False))
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
        Q_base = Q_base / (t_range_units + tf_constant(_SMALL, tf_float32))

    return Q_base


@optional_tf_function
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
    """Integrate mean settlement using a stable stepper."""
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
        dt = _broadcast_like(
            _ensure_3d(tf_cast(dt, tf_float32)),
            h_mean_si,
        )
    dt = tf_reshape(dt, [B, H, 1])

    dt_sec = dt_to_seconds(dt, time_units=time_units)
    dt_sec = tf_reshape(dt_sec, [B, H, 1])

    vprint(
        verbose,
        "[integrate_consolidation_mean] dt_sec stats:",
        "min=", tf_reduce_min(dt_sec),
        "max=", tf_reduce_max(dt_sec),
        "mean=", tf_reduce_mean(dt_sec),
    )

    # --- tau (BH1) ---------------------------------------------
    tau = _broadcast_like(
        _ensure_3d(tf_cast(tau_field, tf_float32)),
        h_mean_si,
    )
    tau = tf_reshape(tau, [B, H, 1])
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
        use_relu=True,
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
    s0 = _ensure_3d(tf_cast(s_init_si, tf_float32))
    s0 = s0[:, :1, :1]
    s0 = tf_reshape(s0, [B, 1, 1])
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
                    + tf_constant(_SMALL, tau_i.dtype)
                )
            )
            nxt = prev * a + seq_i * (1.0 - a)
        else:
            nxt = prev + dt_i * (seq_i - prev) / (
                tau_i + tf_constant(_SMALL, tau_i.dtype)
            )

        return tf_reshape(nxt, shp_prev)

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


def _integrate_consolidation_mean(
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
            a = tf_exp(-dt_i / (tau_i + tf_constant(_SMALL, tau_i.dtype)))
            nxt = prev * a + seq_i * (1.0 - a)
        else:
            nxt = prev + dt_i * (seq_i - prev) / (
                tau_i + tf_constant(_SMALL, tau_i.dtype)
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
        a = tf_exp(-dt_sec / (tau + tf_constant(_SMALL, tau.dtype)))
        pred = s_n * a + s_eq_n * (1.0 - a)
    elif method == "euler":
        pred = s_n + dt_sec * (s_eq_n - s_n) / (tau + tf_constant(_SMALL, tau.dtype))
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
                
                    if vmin <= 0.0 or vmax <= 0.0:
                        raise ValueError(
                            f"{lin_min_key}/{lin_max_key} must be > 0. "
                            f"Got vmin={vmin}, vmax={vmax}."
                        )
                    if vmax <= vmin:
                        raise ValueError(
                            f"{lin_max_key} must be > {lin_min_key}. "
                            f"Got vmin={vmin}, vmax={vmax}."
                        )


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

def get_log_tau_bounds(
    model,
    *,
    as_tensor: bool = True,
    dtype=tf_float32,
    verbose: int = 0,
) -> Tuple[Any, Any]:
    """
    Get (log_tau_min, log_tau_max) for the consolidation timescale.

    Tau is in SI seconds (because tau_phys_from_fields returns seconds).
    Bounds are returned in log-seconds.

    Looks inside model.scaling_kwargs['bounds'] using get_sk() aliases.

    Defaults (safety net):
      tau_min = 7 days
      tau_max = 300 years
    """
    bounds = (getattr(model, "scaling_kwargs", {}) or {}).get("bounds", {}) or {}

    # 1) explicit log-bounds (log-seconds)
    log_min = get_sk(bounds, "log_tau_min", default=None, cast=float)
    log_max = get_sk(bounds, "log_tau_max", default=None, cast=float)

    # 2) build from linear bounds (seconds or time_units)
    if (log_min is None) or (log_max is None):
        tau_min = get_sk(bounds, "tau_min", default=None, cast=float)
        tau_max = get_sk(bounds, "tau_max", default=None, cast=float)

        # 2b) bounds in native time_units -> convert to seconds
        if (tau_min is None) or (tau_max is None):
            tau_min_u = get_sk(bounds, "tau_min_units", default=None, cast=float)
            tau_max_u = get_sk(bounds, "tau_max_units", default=None, cast=float)

            if (tau_min_u is not None) and (tau_max_u is not None):
                sk = getattr(model, "scaling_kwargs", None) or {}
                tu = (
                    get_sk(sk, "time_units", default=None)
                    or getattr(model, "time_units", None)
                    or "yr"
                )
                key = normalize_time_units(tu)
                sec_per = float(TIME_UNIT_TO_SECONDS.get(key, 1.0))
                tau_min = float(tau_min_u) * sec_per
                tau_max = float(tau_max_u) * sec_per

        # 2c) defaults
        if (tau_min is None) or (tau_max is None):
            sec_day = 86400.0
            sec_year = float(TIME_UNIT_TO_SECONDS.get("yr", 31556952.0))
            tau_min = 7.0 * sec_day
            tau_max = 300.0 * sec_year
            logger.warning(
                "Tau bounds not found in scaling_kwargs['bounds']; "
                "using defaults: tau_min=7 days, tau_max=300 years (SI seconds)."
            )

        # sanitize
        tau_min = max(float(tau_min), _SMALL)
        tau_max = max(float(tau_max), _SMALL)
        if tau_max < tau_min:
            logger.warning("tau_max < tau_min; swapping tau bounds.")
            tau_min, tau_max = tau_max, tau_min

        log_min = float(np.log(tau_min))
        log_max = float(np.log(tau_max))

    if not as_tensor:
        return float(log_min), float(log_max)

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

def finite_floor(x: Tensor, eps: float) -> Tensor:
    x = tf_cast(x, tf_float32)
    eps_t = tf_constant(float(eps), tf_float32)
    x = tf_where(tf_math.is_finite(x), x, eps_t)
    return tf_maximum(x, eps_t)

def compose_physics_fields(
    model,
    *,
    coords_flat: Tensor,
    H_si: Tensor,
    K_base: Tensor,
    Ss_base: Tensor,
    tau_base: Tensor,
    training: bool = False,
    eps_KSs: float = _SMALL,
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

    K_corr = model.K_coord_mlp(coords_xy0, training=training)
    Ss_corr = model.Ss_coord_mlp(coords_xy0, training=training)
    tau_corr = model.tau_coord_mlp(coords_xy0, training=training)

    if verbose > 6:
        tf_print_nonfinite("compose/K_corr", K_corr)
        tf_print_nonfinite("compose/Ss_corr", Ss_corr)
        tf_print_nonfinite("compose/tau_corr", tau_corr)

    rawK = K_base + K_corr
    rawSs = Ss_base + Ss_corr

    bounds_mode = str(getattr(model, "bounds_mode", "soft")).strip().lower()

    # ---- K, Ss  ----
    if bounds_mode == "hard":
        logK_min, logK_max, logSs_min, logSs_max = get_log_bounds(
            model, as_tensor=True, dtype=rawK.dtype, verbose=0,
        )
        K_field, logK = bounded_exp(
            rawK, logK_min, logK_max, eps=eps_KSs,
            return_log=True, verbose=0,
        )
        Ss_field, logSs = bounded_exp(
            rawSs, logSs_min, logSs_max, eps=eps_KSs,
            return_log=True, verbose=0,
        )
    else:
        logK = rawK
        logSs = rawSs
        K_field = tf_exp(logK) + tf_constant(eps_KSs, logK.dtype)
        Ss_field = tf_exp(logSs) + tf_constant(eps_KSs, logSs.dtype)

    # ---- tau ( log-space composition + bounds) ----
    delta_log_tau = tau_base + tau_corr

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
        log_tau,        # NEW: return log_tau for bounds penalty + diagnostics
        log_tau_phys,   # NEW: optional but very useful for priors/diagnostics
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
    return dz_dt / (sec + tf_constant(_SMALL, dz_dt.dtype))


def positive(x: Tensor, *, eps: float = _SMALL) -> Tensor:
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
def scale_residual(residual: Tensor, scale: Tensor, *, floor: float = _SMALL) -> Tensor:
    s = tf_cast(scale, residual.dtype)
    f = tf_constant(float(floor), residual.dtype)

    # If scale is NaN/Inf -> replace with floor BEFORE max()
    s = tf_where(tf_math.is_finite(s), s, f)

    s = tf_maximum(s, f)
    s = tf_stop_gradient(s)
    return residual / (s + tf_constant(_SMALL, residual.dtype))


@optional_tf_function
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
    # NEW: pass the SAME dt/time_units used to build cons_res
    dt: Optional[Tensor] = None,
    time_units: Optional[str] = None,
    # NEW: pass real GW terms if available
    dh_dt: Optional[Tensor] = None,
    div_K_grad_h: Optional[Tensor] = None,
    verbose: int = 0,
) -> Dict[str, Tensor]:
    """
    Robust residual scales for v3.2.

    - - cons_scale matches cons_res units.
    - gw_scale is based on magnitudes of storage/div/forcing terms (SI).
    """
    

    sk = model.scaling_kwargs or {}
    mode = resolve_cons_units(sk)

    # -----------------------------
    # 0) Normalize shapes
    # -----------------------------
    s = tf_cast(s_mean, tf_float32)
    h = tf_cast(h_mean, tf_float32)

    if s.shape.rank == 2:
        s = s[:, :, None]
    if h.shape.rank == 2:
        h = h[:, :, None]

    # -----------------------------
    # 1) Choose time_units CONSISTENTLY
    # -----------------------------
    if time_units is None:
        time_units = (
            get_sk(getattr(model, "scaling_kwargs", None), 
                   "time_units", default=None)
            or getattr(model, "time_units", None)
            or "unitless"
        )

    # -----------------------------
    # 2) Build dt (in same units as residual construction)
    # -----------------------------
    if dt is None:
        # infer from t if not provided
        tt = tf_cast(t, tf_float32)
        if tt.shape.rank == 2:
            tt = tt[:, :, None]

        if (tt.shape.rank >= 2) and (tt.shape[1] is not None) and (tt.shape[1] > 1):
            dt_step = tt[:, 1:, :] - tt[:, :-1, :]
        else:
            dt_step = tf_zeros_like(s[:, :1, :]) + 1.0

        # de-normalize if coords_normalized=True
        coords_norm = bool((model.scaling_kwargs or {}).get("coords_normalized", False))
        tR, _, _ = coord_ranges(model.scaling_kwargs or {})
        if coords_norm and tR:
            dt_step = dt_step * tf_cast(float(tR), dt_step.dtype)
    else:
        dt_step = tf_cast(dt, tf_float32)
        if dt_step.shape.rank == 2:
            dt_step = dt_step[:, :, None]

    # dt_step should be (B,H,1) OR (B,H-1,1). We'll need a per-step dt_sec.
    dt_sec = dt_to_seconds(dt_step, time_units=time_units)
    dt_sec = tf_maximum(dt_sec, tf_constant(_SMALL, tf_float32))

    # A single reference dt for scaling (mean over batch & time)
    dt_ref = tf_stop_gradient(tf_reduce_mean(tf_abs(tf_reshape(dt_sec, [-1]))))
    dt_ref = tf_maximum(dt_ref, _SMALL) 

    # -----------------------------
    # 3) Consolidation scale (rate) = max(|ds/dt|, |(s_eq - s)/tau|)
    # -----------------------------
    # Estimate ds/dt from s_mean
    # Use differences along time: ds has shape (B,H-1,1)
    eps = tf_constant(_SMALL, tf_float32)
    
    # dt in native time_units (scalar ref)
    dt_ref_u = tf_stop_gradient(
        tf_reduce_mean(
            tf_abs(tf_reshape(dt_step, [-1]))
        )
    )
    dt_ref_u = tf_maximum(dt_ref_u, eps)
    
    # dt in seconds (scalar ref)
    dt_ref = tf_stop_gradient(
        tf_reduce_mean(
            tf_abs(tf_reshape(dt_sec, [-1]))
        )
    )
    dt_ref = tf_maximum(
        dt_ref,
        seconds_per_time_unit(
            time_units,
            dtype=tf_float32,
        ),
    )
    
    # ds magnitude (meters)
    if tf_shape(s)[1] > 1:
        ds = s[:, 1:, :] - s[:, :-1, :]
        ds_abs = tf_abs(tf_reshape(ds, [-1]))
        ds_ref = tf_stop_gradient(tf_reduce_mean(ds_abs))
        ds_max = tf_stop_gradient(tf_reduce_max(ds_abs))
    else:
        ds_ref = tf_constant(0.0, tf_float32)
        ds_max = tf_constant(0.0, tf_float32)
    
    if mode == "step":
        cons_scale = tf_maximum(ds_ref, 0.1 * ds_max)
    
    elif mode == "time_unit":
        cons_scale = tf_maximum(
            ds_ref / dt_ref_u,
            0.1 * (ds_max / dt_ref_u),
        )
    
    else:
        cons_scale = tf_maximum(
            ds_ref / dt_ref,
            0.1 * (ds_max / dt_ref),
        )
    
    # Add equilibrium / relaxation magnitude if tau/H available
    if (tau_field is not None) and (H_field is not None):
        tau = tf_maximum(tf_cast(tau_field, tf_float32), eps)
    
        href = h_ref_si
        if href is None:
            href = tf_cast(getattr(model, "h_ref", 0.0), tf_float32)
        href = tf_broadcast_to(
            tf_convert_to_tensor(href, tf_float32),
            tf_shape(h),
        )
    
        delta_h = tf_maximum(href - h, 0.0)
        s_eq = (
            tf_cast(Ss_field, tf_float32)
            * delta_h
            * tf_cast(H_field, tf_float32)
        )
    
        eq_mis = tf_abs(s_eq - s)
        eq_ref = tf_stop_gradient(
            tf_reduce_mean(tf_reshape(eq_mis, [-1])) + eps
        )
        eq_max = tf_stop_gradient(
            tf_reduce_max(tf_reshape(eq_mis, [-1])) + eps
        )
    
        if mode == "step":
            cons_scale = tf_maximum(cons_scale, eq_ref)
            cons_scale = tf_maximum(cons_scale, 0.1 * eq_max)
    
        else:
            relax = tf_abs(eq_mis / (tau + eps))
    
            if mode == "time_unit":
                sec_u = seconds_per_time_unit(
                    time_units,
                    dtype=tf_float32,
                )
                relax = relax * sec_u
    
            r_ref = tf_stop_gradient(
                tf_reduce_mean(tf_reshape(relax, [-1])) + eps
            )
            r_max = tf_stop_gradient(
                tf_reduce_max(tf_reshape(relax, [-1])) + eps
            )
    
            cons_scale = tf_maximum(cons_scale, r_ref)
            cons_scale = tf_maximum(cons_scale, 0.1 * r_max)
    
    # Unit-aware floor
    floor_def = _SMALL
    if mode in ("step", "time_unit"):
        floor_def = 1e-6
    
    floor_val = float(get_sk(sk, "cons_scale_floor", default=floor_def))

    cons_scale = tf_maximum(
        cons_scale,
        tf_constant(floor_val, tf_float32),
    )
    
    cons_scale = tf_stop_gradient(cons_scale)


    # -----------------------------
    # 4) GW scale = max(|Ss dh/dt|, |div|, |Q|)
    # -----------------------------
    # dh/dt reference:
    if dh_dt is None:
        if tf_shape(h)[1] > 1:
            dh = h[:, 1:, :] - h[:, :-1, :]
            dh_abs = tf_abs(tf_reshape(dh, [-1]))
            dh_dt_ref = tf_stop_gradient(tf_reduce_mean(dh_abs) / dt_ref)
            dh_dt_max = tf_stop_gradient(tf_reduce_max(dh_abs) / dt_ref)
        else:
            dh_dt_ref = tf_constant(0.0, tf_float32)
            dh_dt_max = tf_constant(0.0, tf_float32)
    else:
        dh_dt_ref = tf_stop_gradient(tf_reduce_mean(tf_abs(tf_reshape(tf_cast(dh_dt, tf_float32), [-1]))))
        dh_dt_max = tf_stop_gradient(tf_reduce_max(tf_abs(tf_reshape(tf_cast(dh_dt, tf_float32), [-1]))))

    Ss_ref = tf_stop_gradient(tf_reduce_mean(tf_abs(tf_reshape(tf_cast(Ss_field, tf_float32), [-1]))))
    storage_ref = Ss_ref * dh_dt_ref
    storage_max = Ss_ref * dh_dt_max

    gw_scale = tf_maximum(storage_ref, tf_constant(0.1, tf_float32) * storage_max)

    if div_K_grad_h is not None:
        div_abs = tf_abs(tf_reshape(tf_cast(div_K_grad_h, tf_float32), [-1]))
        div_ref = tf_stop_gradient(tf_reduce_mean(div_abs) + tf_constant(1e-12, tf_float32))
        div_max = tf_stop_gradient(tf_reduce_max(div_abs) + tf_constant(1e-12, tf_float32))
        gw_scale = tf_maximum(gw_scale, div_ref)
        gw_scale = tf_maximum(gw_scale, tf_constant(0.1, tf_float32) * div_max)

    if Q is not None:
        Q_abs = tf_abs(tf_reshape(tf_cast(Q, tf_float32), [-1]))
        Q_ref = tf_stop_gradient(tf_reduce_mean(Q_abs) + tf_constant(1e-12, tf_float32))
        Q_max = tf_stop_gradient(tf_reduce_max(Q_abs) + tf_constant(1e-12, tf_float32))
        gw_scale = tf_maximum(gw_scale, Q_ref)
        gw_scale = tf_maximum(gw_scale, tf_constant(0.1, tf_float32) * Q_max)

    gw_floor = tf_constant(
        float(get_sk(
            getattr(model, "scaling_kwargs", None), "gw_scale_floor", default=1e-12)),
        tf_float32
    )

    gw_scale = tf_maximum(gw_scale, gw_floor)
    gw_scale = tf_stop_gradient(gw_scale)

    out = {"cons_scale": cons_scale, "gw_scale": gw_scale}

    if verbose > 0:
        vprint(
            verbose,
            "cons_scale:",
            "min=", tf_reduce_min(cons_scale),
            "mean=", tf_reduce_mean(cons_scale),
            "max=", tf_reduce_max(cons_scale),
        )
        vprint(
            verbose,
            "gw_scale:",
            "min=", tf_reduce_min(gw_scale),
            "mean=", tf_reduce_mean(gw_scale),
            "max=", tf_reduce_max(gw_scale),
        )
    vprint(verbose, "compute_scales(v3.2): time_units=", time_units)
    vprint(verbose, "compute_scales(v3.2): cons_scale=", 
           cons_scale, "gw_scale=", gw_scale)
    return out


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
    dt: Optional[Tensor] = None,                 # NEW
    return_incremental: bool = True,             # NEW
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
    dtype: Tensor = tf_float32,
) -> Tensor:
    """Root-mean-square (RMS) of a tensor.

    Notes
    -----
    - By default (eps=None), NO flooring is applied.
      This avoids "frozen" epsilons in logs.
    - Use eps (mean-square floor) only when you really
      need a nonzero lower bound (rare for metrics).
    - Use rms_floor if you specifically want an RMS
      lower bound (also opt-in).

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int | Sequence[int] | None
        Reduction axis/axes. If None, reduce all.
    keepdims : bool
        Keep reduced dimensions.
    eps : float | None
        Optional lower bound on the mean-square before
        sqrt (mean-square floor). If None, disabled.
    rms_floor : float | None
        Optional lower bound on the RMS after sqrt.
        If None, disabled.
    nan_policy : {"propagate", "raise", "omit"}
        - "propagate": NaN/Inf propagate naturally.
        - "raise": assert all finite before reduce.
        - "omit": ignore non-finite entries in RMS.
    dtype : tf.DType
        Compute dtype (casts x before reduction).

    Returns
    -------
    rms : Tensor
        RMS value (scalar if axis=None; else reduced).
    """
    x = tf_cast(x, dtype)

    if nan_policy == "raise":
        tf_debugging.assert_all_finite(
            x,
            "to_rms(): x has NaN/Inf",
        )
        ms = tf_reduce_mean(
            tf_square(x),
            axis=axis,
            keepdims=keepdims,
        )

    elif nan_policy == "omit":
        finite = tf_math.is_finite(x)
        x0 = tf_where(finite, x, tf_zeros_like(x))
        num = tf_reduce_sum(
            tf_square(x0),
            axis=axis,
            keepdims=keepdims,
        )
        den = tf_reduce_sum(
            tf_cast(finite, dtype),
            axis=axis,
            keepdims=keepdims,
        )
        den = tf_maximum(
            den,
            tf_constant(1.0, dtype),
        )
        ms = num / den

    else:  # "propagate"
        ms = tf_reduce_mean(
            tf_square(x),
            axis=axis,
            keepdims=keepdims,
        )

    if eps is not None and float(eps) > 0.0:
        ms = tf_maximum(
            ms,
            tf_constant(float(eps), dtype),
        )
        
    if ms_floor is not None:
        msf = tf_constant(float(ms_floor), dtype)
        ms = tf_maximum(ms, msf)
        
    rms = tf_sqrt(ms)

    if rms_floor is not None and float(rms_floor) > 0.0:
        rms = tf_maximum(
            rms,
            tf_constant(float(rms_floor), dtype),
        )

    return rms

# def to_rms(
#     x: Tensor,
#     *,
#     axis=None,
#     keepdims: bool = False,
#     eps: float = _SMALL,
# ) -> Tensor:
#     """Root-mean-square (RMS) of a tensor.

#     Parameters
#     ----------
#     x : Tensor
#         Input tensor.
#     axis : int | Sequence[int] | None
#         Reduction axis/axes. If None, reduce over all elements.
#     keepdims : bool
#         Keep reduced dimensions.
#     eps : float
#         Optional lower bound on the mean-square before sqrt
#         (useful to avoid sqrt(0) in some diagnostics).

#     Returns
#     -------
#     rms : Tensor
#         RMS value (scalar if axis=None; else reduced tensor).
#     """
#     x = tf_cast(x, tf_float32)
#     ms = tf_reduce_mean(tf_square(x), axis=axis, keepdims=keepdims)
#     if eps and float(eps) > 0.0:
#         ms = tf_maximum(ms, tf_constant(float(eps), tf_float32))
#     return tf_sqrt(ms)

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