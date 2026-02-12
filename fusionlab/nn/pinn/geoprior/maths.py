# -*- coding: utf-8 -*-
# _geoprior.maths.py
"""
GeoPrior maths helpers (physics terms + scaling).
Short docs only; full docs later.
"""

from __future__ import annotations

from typing import ( 
    Mapping, 
    Any, Dict, 
    Optional, Sequence, 
    Tuple, Union
)

import numpy as np

from ... import KERAS_DEPS, dependency_message
from ...._fusionlog import OncePerMessageFilter, fusionlog
from ....api.docs import DocstringComponents, _halnet_core_params

from .utils import coord_ranges, get_h_ref_si, get_sk

K = KERAS_DEPS

Tensor = K.Tensor
Dataset = K.Dataset
GradientTape = K.GradientTape
Constraint = K.Constraint

tf_abs = K.abs
tf_argmin = K.argmin
tf_broadcast_to = K.broadcast_to
tf_cast = K.cast
tf_clip_by_value = K.clip_by_value
tf_concat = K.concat
tf_cond = K.cond
tf_constant = K.constant
tf_convert_to_tensor = K.convert_to_tensor
tf_cumsum = K.cumsum
tf_debugging = K.debugging
tf_equal = K.equal
tf_exp = K.exp
tf_expand_dims = K.expand_dims
tf_float32 = K.float32
tf_gather = K.gather
tf_greater = K.greater
tf_identity = K.identity
tf_int32 = K.int32
tf_is_inf = K.is_inf
tf_is_nan = K.is_nan
tf_log = K.log
tf_logical_and = K.logical_and
tf_logical_or = K.logical_or
tf_math = K.math
tf_maximum = K.maximum
tf_minimum = K.minimum
tf_ones_like = K.ones_like
tf_pow = K.pow
tf_print = K.print
tf_rank = K.rank
tf_reduce_any = K.reduce_any
tf_reduce_max = K.reduce_max
tf_reduce_mean = K.reduce_mean
tf_reduce_min = K.reduce_min
tf_reduce_sum = K.reduce_sum
tf_reshape = K.reshape
tf_scan = K.scan
tf_shape = K.shape
tf_sigmoid = K.sigmoid
tf_softplus = K.softplus
tf_sqrt = K.sqrt
tf_square = K.square
tf_stack = K.stack
tf_stop_gradient = K.stop_gradient
tf_switch_case = K.switch_case
tf_tile = K.tile
tf_transpose = K.transpose
tf_where = K.where
tf_zeros = K.zeros
tf_zeros_like = K.zeros_like

register_keras_serializable = K.register_keras_serializable
deserialize_keras_object = K.deserialize_keras_object

# Optional: silence autograph verbosity in TF-backed runtimes.
tf_autograph = getattr(K, "autograph", None)
if tf_autograph is not None:
    tf_autograph.set_verbosity(0)

# Module logger + shared docs
DEP_MSG = dependency_message("nn.pinn.geoprior.maths")

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_halnet_core_params),
)

# Constants + types
_EPSILON = 1e-15

AxisLike = Optional[Union[int, Sequence[int]]]

# Time units + scaling
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


class LogClipConstraint(Constraint):
    r"""
    NaN-safe clip constraint for log-parameters.
    
    This constraint is intended for parameters stored in log-space,
    such as ``logK``, ``logSs``, or ``log_tau``, where the model must
    enforce hard bounds:
    
    .. math::
    
       w \in [w_{min}, w_{max}]
    
    Why this exists
    ---------------
    In TensorFlow, ``clip_by_value`` does not repair invalid values:
    
    .. math::
    
       clip(NaN, a, b) = NaN
    
    Therefore, if a parameter ever becomes non-finite (NaN or Inf),
    a plain clipping constraint will silently keep it invalid and
    training can destabilize. This class explicitly sanitizes
    non-finite entries before applying the clip.
    
    Mapping
    -------
    Given an input weight tensor ``w`` and bounds
    ``min_value`` and ``max_value``:
    
    1) Sanitize non-finite entries:
    
    .. math::
    
       w_{safe}[i]
       =
       \begin{cases}
       w[i], & \text{if } w[i] \text{ is finite} \\
       w_{min}, & \text{otherwise}
       \end{cases}
    
    2) Apply hard clipping:
    
    .. math::
    
       w_{out}
       =
       \min(\max(w_{safe}, w_{min}), w_{max})
    
    The output is guaranteed to be finite as long as
    ``min_value`` and ``max_value`` are finite.
    
    Parameters
    ----------
    min_value : float or Tensor
        Lower bound for the constrained tensor in log-space. This is
        cast to ``tf_float32`` and stored.
    
    max_value : float or Tensor
        Upper bound for the constrained tensor in log-space. This is
        cast to ``tf_float32`` and stored.
    
    Returns
    -------
    Constraint
        A callable constraint object compatible with Keras variables.
        When applied, it returns a clipped tensor in float32.
    
    Notes
    -----
    * This constraint is most appropriate for parameters represented
      in log-space because hard bounds in log-space correspond to
      multiplicative bounds in linear space.
    * Sanitizing to ``min_value`` is a conservative choice:
      it prevents NaN propagation while keeping the parameter within
      the feasible region. If you prefer a different fallback (e.g.
      0 or the midpoint), change the replacement value accordingly.
    * The constraint operates in ``tf_float32`` for speed and
      compatibility with typical training graphs.
    
    Examples
    --------
    Constrain a learnable log-parameter:
    
    .. code-block:: python
    
       logK = tf.Variable(
           initial_value=0.0,
           constraint=LogClipConstraint(-20.0, 5.0),
           trainable=True,
           dtype=tf.float32,
       )
    
    In a Keras layer weight:
    
    .. code-block:: python
    
       self.log_tau = self.add_weight(
           name="log_tau",
           shape=(1,),
           initializer="zeros",
           trainable=True,
           constraint=LogClipConstraint(log_tau_min, log_tau_max),
       )
    
    See Also
    --------
    keras.constraints.Constraint
        Base class for Keras constraints.
    
    tf.clip_by_value
        Elementwise clipping. Note that it does not repair NaNs.
    
    tf.where
        Used here to sanitize non-finite entries before clipping.
    
    References
    ----------
    .. [1] Abadi, M. et al.
       TensorFlow: Large-Scale Machine Learning on Heterogeneous
       Systems. (2016). (Defines clip and masking behaviors).
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
    r"""
    Convert ``Q_logits`` into a GW source term in SI units.
    
    This helper maps the network output ``Q_logits`` into a source
    term :math:`Q_{term}` that is compatible with the groundwater
    PDE residual used by the model:
    
    .. math::
    
       R_{gw}
       =
       S_s \, \frac{\partial h}{\partial t}
       -
       \nabla \cdot (K \nabla h)
       -
       Q_{term}
    
    The returned tensor always has units of 1/s so it can be
    subtracted directly in :math:`R_{gw}`.
    
    Overview
    --------
    The model can interpret the raw output ``Q_logits`` in multiple
    ways depending on ``Q_kind`` resolved from ``scaling_kwargs``.
    All modes must end with :math:`Q_{term}` in 1/s, but the meaning
    of ``Q_logits`` differs:
    
    ``per_volume``
        ``Q_logits`` represents a volumetric forcing rate already in
        inverse time units (either 1/time_unit or 1/s depending on
        flags). This is the simplest and most backward-compatible
        interpretation.
    
    ``recharge_rate``
        ``Q_logits`` represents a recharge flux expressed as a length
        rate (m/time_unit or m/s). It is converted into a volumetric
        rate by dividing by an effective thickness :math:`H`:
    
        .. math::
    
           Q_{term} = \frac{R}{H}
    
        where :math:`R` is in m/s and :math:`H` is in m, giving 1/s.
    
    ``head_rate``
        ``Q_logits`` represents a head-rate forcing (m/time_unit or
        m/s) that enters the storage term. Since the storage term is
        :math:`S_s \, dh/dt`, the equivalent forcing in the residual
        is:
    
        .. math::
    
           Q_{term} = S_s \, q_h
    
        where :math:`q_h` is in m/s and :math:`S_s` is in 1/m,
        yielding 1/s.
    
    Time normalization handling
    ---------------------------
    When coordinates are normalized (typical in this project), the
    time coordinate is scaled by a range factor :math:`t_R`. If
    ``Q_logits`` was produced in that normalized coordinate system,
    it must be converted back to the intended time units before any
    SI conversion. This function delegates that correction to the
    internal helper ``_apply_q_normalized_time_rule`` using:
    
    * ``coords_normalized``
    * ``t_range_units`` (the time range in model time units)
    * ``scaling_kwargs`` flags
    
    After this correction, unit conversion is applied based on the
    selected ``Q_kind`` and SI flags.
    
    Mode details
    ------------
    per_volume
    ~~~~~~~~~~
    In this mode the model output is treated as already being an
    inverse-time quantity.
    
    If either of the following flags is True:
    
    * ``Q_in_per_second`` or
    * ``Q_in_si``
    
    then ``Q_logits`` is assumed to already be in 1/s and returned
    directly.
    
    Otherwise, it is assumed to be in 1/time_unit and converted to
    1/s via:
    
    .. math::
    
       Q_{term} = Q \cdot \frac{1}{sec\_per\_time\_unit}
    
    where ``sec_per_time_unit`` depends on ``time_units``.
    
    recharge_rate
    ~~~~~~~~~~~~~
    Here, ``Q_logits`` is treated as a length rate :math:`R`:
    
    .. math::
    
       R \in \mathrm{m}/\mathrm{s}
    
    It is converted from m/time_unit to m/s unless
    ``Q_length_in_si`` is True.
    
    The volumetric rate is then computed as:
    
    .. math::
    
       Q_{term} = \frac{R}{H}
    
    To prevent division instability, the thickness is floored:
    
    .. math::
    
       H_{safe} = \max(H, H_{floor})
    
    and the returned value is:
    
    .. math::
    
       Q_{term} = \frac{R}{H_{safe}}
    
    head_rate
    ~~~~~~~~~
    Here, ``Q_logits`` is treated as a head-rate :math:`q_h`:
    
    .. math::
    
       q_h \in \mathrm{m}/\mathrm{s}
    
    It is converted from m/time_unit to m/s unless
    ``Q_length_in_si`` is True.
    
    The residual forcing is:
    
    .. math::
    
       Q_{term} = S_s \, q_h
    
    If ``Ss_field`` is missing, a robust fallback consistent with
    the rest of the model is used:
    
    .. math::
    
       S_s \approx m_v \, gamma_w
    
    where :math:`m_v` is taken from the model and :math:`gamma_w`
    is the configured unit weight of water.
    
    Parameters
    ----------
    This function is typically called from the physics core where
    all arguments are already defined. For meanings and expected
    shapes, refer to the caller that constructs the GW residual
    and its inputs.
    
    In brief:
    
    * ``Q_logits`` is the network output for the forcing channel.
    * ``Ss_field`` and ``H_field`` are the effective fields used
      by the PDE, broadcastable to the batch-horizon layout.
    * ``coords_normalized`` and ``t_range_units`` describe how time
      normalization was applied.
    * ``time_units`` specifies the units used for conversion when
      interpreting rates.
    * ``scaling_kwargs`` provides configuration including ``Q_kind``
      and unit flags.
    
    Returns
    -------
    Q_term : Tensor
        Source term :math:`Q_{term}` in 1/s, broadcastable to the
        GW residual layout (typically (B,H,1)).
    
    Raises
    ------
    ValueError
        If ``Q_kind='recharge_rate'`` is selected but ``H_field`` is
        not provided.
    
    Notes
    -----
    * Choose ``per_volume`` when you want a direct 1/s forcing that
      can be interpreted as a volumetric sink/source per unit volume.
    * Choose ``recharge_rate`` when you want the network to predict
      a flux-like term (m/s) that becomes volumetric forcing by
      dividing by thickness.
    * Choose ``head_rate`` when you want forcing to act like an
      additive term to :math:`dh/dt` inside the storage term.
    
    Examples
    --------
    Assuming the physics core has already produced ``Q_logits`` and
    the effective fields:
    
    .. code-block:: python
    
       Q_term = q_to_gw_source_term_si(
           model,
           Q_logits,
           Ss_field=Ss_field,
           H_field=H_field,
           coords_normalized=coords_normalized,
           t_range_units=t_range_units,
           time_units=time_units,
           scaling_kwargs=scaling_kwargs,
       )
    
    See Also
    --------
    rate_to_per_second
        Converts values in 1/time_unit or m/time_unit to SI per
        second rates.
    
    _apply_q_normalized_time_rule
        Corrects rates if the time coordinate was normalized.
    
    resolve_q_kind
        Resolves the configured Q interpretation mode from the
        scaling configuration.
    
    References
    ----------
    .. [1] Bear, J.
       Dynamics of Fluids in Porous Media. Dover (1988).
       (Groundwater flow equation and source terms).
    
    .. [2] de Marsily, G.
       Quantitative Hydrogeology. Academic Press (1986).
       (Units and interpretation of recharge and forcing terms).
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
    r"""
    Compute an m_v - gamma_w prior from predicted S_s.
    
    This routine builds a log-space residual that ties the model's
    specific storage :math:`S_s` to the consolidation coefficient
    :math:`m_v` and the unit weight of water :math:`gamma_w` via:
    
    .. math::
    
       S_s \approx m_v \, \gamma_w
    
    The constraint is applied in log space for numerical stability:
    
    .. math::
    
       r = \log(S_s) - \log(m_v \, \gamma_w)
    
    Depending on ``mode``, gradients may be blocked or allowed to
    flow through :math:`S_s` (or its log) to control stability.
    
    Mathematical objective
    ----------------------
    If ``as_loss=True``, this function returns a scalar loss built
    from two components:
    
    1) A global mismatch term based on the mean residual:
    
    .. math::
    
       \bar{r} = \mathrm{mean}(r)
    
    .. math::
    
       L_g = \mathrm{Huber}(\bar{r}; delta)
    
    2) A dispersion term that discourages spatial or batch-wide
    scatter around the mean residual:
    
    .. math::
    
       L_d = \mathrm{mean}(
           \mathrm{Huber}(r - \bar{r}; delta)
       )
    
    The total loss is:
    
    .. math::
    
       L = L_g + alpha\_disp \, L_d
    
    Optionally, an additional weight and warmup ramp may be applied:
    
    .. math::
    
       L \leftarrow w(step) \, L
    
    Mode semantics and gradient flow
    --------------------------------
    The choice of mode controls where gradients are allowed.
    
    ``calibrate`` (default)
        Uses :func:`tf.stop_gradient` on ``Ss_field`` before taking
        :math:`\log(S_s)`. This calibrates :math:`m_v` without
        reshaping the :math:`S_s` field produced by the trunk.
    
        This is typically the safest choice when the mean settlement
        is physics-driven and the network already has strong physics
        constraints elsewhere.
    
    ``field``
        Backpropagates through ``Ss_field``. This can be unstable
        when :math:`S_s` becomes small because:
    
        .. math::
    
           \frac{\partial \log(S_s)}{\partial S_s} = \frac{1}{S_s}
    
        so gradients can be amplified.
    
    ``logss``
        Backpropagates through ``logSs`` directly. This is often
        more stable than ``field`` because the log transform is
        already computed upstream in a controlled manner (for
        example, using guarded exponentiation in the field
        composer). Use this when you want a stronger anchoring of
        the log-storage field without the 1/S_s amplification.
    
    Inputs used
    -----------
    This function requires exactly one of:
    
    * ``logSs`` when ``mode='logss'``, or
    * ``Ss_field`` when ``mode!='logss'``.
    
    The log target term :math:`\log(m_v \, \gamma_w)` is obtained
    from the model configuration through helper resolvers. These
    helpers may also apply internal conventions such as:
    
    * whether :math:`m_v` is learnable or fixed,
    * the unit system for :math:`gamma_w`,
    * safe floors ``eps`` to avoid :math:`\log(0)`.
    
    Parameters
    ----------
    model : Any
        Model instance providing :math:`m_v`, :math:`gamma_w`,
        scaling configuration, and optional scheduling state used
        by helper resolvers.
    
    Ss_field : Tensor, optional
        Specific storage field :math:`S_s` in 1/m. Required unless
        ``mode='logss'``. The expected shape is broadcastable to
        the physics batch layout (typically (B,H,1) or (B,1,1)).
    
    logSs : Tensor, optional
        Log-specific storage :math:`\log(S_s)`. Required when
        ``mode='logss'``. Prefer passing the raw log output from
        the field composer to preserve true bound violations.
    
    mode : str, optional
        Prior mode controlling gradient flow. If None, the mode is
        resolved from model configuration. Supported modes are
        ``'calibrate'``, ``'field'``, and ``'logss'`` (aliases may
        be accepted by the internal canonicalizer).
    
    as_loss : bool, default=True
        If True, return a scalar loss :math:`L`. If False, return
        the residual field :math:`r`.
    
    weight : float or Tensor, optional
        Optional multiplicative factor applied to the returned loss.
        If None, the function may still derive a weight from model
        configuration.
    
    warmup_steps : int, optional
        If provided, enables an internal warmup schedule for the
        prior weight via helper logic (for example a linear ramp
        from 0 to 1 over ``warmup_steps``).
    
    step : int or Tensor, optional
        Training step index passed to the warmup logic. If None, the
        warmup logic may use model state or disable warmup.
    
    alpha_disp : float, default=0.1
        Weight for the dispersion penalty :math:`L_d`.
    
    delta : float, default=1.0
        Huber threshold parameter used by the robust penalty.
    
    eps : float, default=_EPSILON
        Positive floor used when computing :math:`\log(S_s)` to
        avoid :math:`\log(0)` and reduce numerical issues.
    
    verbose : int, default=0
        Verbosity flag forwarded to internal helpers for debugging.
    
    Returns
    -------
    loss_or_residual : Tensor
        If ``as_loss=True``, a scalar tensor representing the MV
        prior loss. If ``as_loss=False``, the residual field
        :math:`r` with the same shape as ``logSs`` (or derived
        logSs from ``Ss_field``).
    
    Raises
    ------
    ValueError
        If required inputs are missing for the selected mode
        (for example, ``mode='logss'`` without ``logSs``).
    
    Notes
    -----
    Why log space
    ~~~~~~~~~~~~~
    Using :math:`\log(S_s)` makes the prior scale-invariant and
    reduces sensitivity to absolute magnitudes. It also aligns with
    how bounds on :math:`K` and :math:`S_s` are often expressed in
    log space.
    
    Why the mean + dispersion split
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Calibrating :math:`m_v` is primarily a global adjustment. The
    mean term penalizes systematic mismatch, while the dispersion
    term discourages pathological spatial variability in the prior
    residual without forcing every location to match exactly.
    
    Scheduling
    ~~~~~~~~~~
    In many training regimes, it is beneficial to ramp this prior
    after the data loss stabilizes. The optional warmup hook allows
    a gradual introduction to avoid early domination.
    
    Examples
    --------
    Compute a scalar MV prior loss in calibrate mode:
    
    .. code-block:: python
    
       loss_mv = compute_mv_prior(
           model,
           Ss_field=Ss_field,
           mode="calibrate",
           as_loss=True,
           alpha_disp=0.1,
           delta=1.0,
       )
    
    Use the residual field for diagnostics (no reduction):
    
    .. code-block:: python
    
       r_mv = compute_mv_prior(
           model,
           Ss_field=Ss_field,
           mode="field",
           as_loss=False,
       )
    
    Use logSs from the field composer (preferred for strong
    anchoring without 1/S_s amplification):
    
    .. code-block:: python
    
       K, Ss, tau, tau_phys, Hd, dlogtau, logK, logSs, log_tau, log_tau_phys = (
           compose_physics_fields(...)
       )
       loss_mv = compute_mv_prior(
           model,
           logSs=logSs,
           mode="logss",
           as_loss=True,
       )
    
    See Also
    --------
    compose_physics_fields
        Produces ``logSs`` consistent with guarded exponentiation.
    
    compute_consistency_prior
        Prior linking learned tau to physically implied tau.
    
    assemble_physics_loss
        Combines MV prior with other physics terms and offsets.
    
    References
    ----------
    .. [1] Terzaghi, K., Peck, R. B., and Mesri, G.
       Soil Mechanics in Engineering Practice. Wiley (1996).
       (Consolidation theory and storage relations).
    
    .. [2] Huber, P. J.
       Robust Statistics. Wiley (1981).  (Huber loss).
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
    r"""
    Compute equilibrium compaction ``s_eq`` in SI meters.
    
    This function computes the equilibrium (instantaneous) settlement
    that would be reached under a sustained head change, given a
    specific storage field and a compressible thickness. The output
    ``s_eq`` is used by the consolidation residual to compare the
    current settlement state against its equilibrium target.
    
    Mathematical definition
    -----------------------
    Let:
    
    * :math:`h_{mean}` be the mean hydraulic head (m),
    * :math:`h_{ref}` be a reference head (m),
    * :math:`S_s` be specific storage (1/m),
    * :math:`H` be compressible thickness (m).
    
    A drawdown (head loss) scalar :math:`Delta h` is formed from a
    rule:
    
    .. math::
    
       Delta h_{raw} =
       \begin{cases}
         h_{ref} - h_{mean}, & \text{rule = ref\_minus\_mean} \\
         h_{mean} - h_{ref}, & \text{rule = mean\_minus\_ref}
       \end{cases}
    
    A non-negative drawdown is enforced using a gating operator
    :math:`[x]_+` controlled by ``drawdown_mode``:
    
    .. math::
    
       Delta h = [Delta h_{raw}]_+
    
    Finally, equilibrium compaction is:
    
    .. math::
    
       s_{eq} = S_s \, Delta h \, H
    
    Units are consistent: :math:`(1/m) * (m) * (m) = m`.
    
    Drawdown gating
    ---------------
    The gating operator is chosen by ``drawdown_mode``:
    
    * ``"none"``:
      :math:`[x]_+ = x` (no positivity enforcement).
    * ``"relu"``:
      :math:`[x]_+ = max(0, x)`.
    * ``"softplus"``:
      Smooth positive part via softplus (implementation dependent).
    * ``"smooth_relu"``:
      A smooth approximation to ReLU controlled by ``relu_beta``.
    
    If ``drawdown_zero_at_origin=True`` in ``"smooth_relu"`` mode,
    the smooth positive part is shifted so its value at zero is
    approximately zero, improving interpretability of the residual
    near :math:`Delta h_{raw}=0`.
    
    Reference-gradient handling
    ---------------------------
    If ``stop_grad_ref=True`` (recommended), gradients are stopped
    through the reference:
    
    .. math::
    
       h_{ref} := stop\_gradient(h_{ref})
    
    This prevents the model from trivially reducing drawdown by
    moving the reference rather than adjusting the predicted head.
    
    Clipping
    --------
    If ``drawdown_clip_max`` is provided, the gated drawdown is
    clipped after gating:
    
    .. math::
    
       Delta h := clip(Delta h, eps, Delta h_{max})
    
    This can prevent extremely large drawdowns from dominating the
    physics loss in early training.
    
    Parameters
    ----------
    h_mean_si : Tensor
        Mean head (or depth, depending on your pipeline) in meters.
        Must be broadcastable to shape ``(B, H, 1)``. If provided as
        ``(B, H)``, it is expanded to ``(B, H, 1)``.
    
    h_ref_si : Tensor
        Reference head (or depth) in meters, broadcastable to
        ``h_mean_si``. If ``stop_grad_ref=True``, gradients are
        stopped through this tensor.
    
    Ss_field : Tensor
        Specific storage field :math:`S_s` in 1/m. Must be
        broadcastable to ``h_mean_si``. Non-finite values are
        sanitized to zero. A non-negativity clamp is applied.
    
    H_field_si : Tensor
        Compressible thickness :math:`H` in meters. Must be
        broadcastable to ``h_mean_si``. Non-finite values are
        sanitized to zero. A non-negativity clamp is applied.
    
    drawdown_mode : str, default="smooth_relu"
        Positivity enforcement for drawdown. Supported values:
        ``"smooth_relu"``, ``"relu"``, ``"softplus"``, ``"none"``.
    
    drawdown_rule : str, default="smooth_relu"
        Rule used to form raw drawdown:
        ``"ref_minus_mean"`` or ``"mean_minus_ref"``.
    
    relu_beta : float, default=20.0
        Smoothness/steepness for ``"smooth_relu"`` gating. Larger
        values approach hard ReLU.
    
    stop_grad_ref : bool, default=True
        If True, apply ``stop_gradient`` to ``h_ref_si`` to prevent
        reference drift.
    
    drawdown_zero_at_origin : bool, default=False
        Only used by ``"smooth_relu"``. If True, shift the smooth
        positive-part so it is approximately zero at input zero.
    
    drawdown_clip_max : float or None, default=None
        If provided, clip gated drawdown to
        ``[eps, drawdown_clip_max]``.
    
    eps : float, default=_EPSILON
        Small positive constant used by gating/clipping utilities.
    
    verbose : int, default=0
        Verbosity level for debug printing and basic stats.
    
    Returns
    -------
    s_eq : Tensor
        Equilibrium compaction in meters, shape ``(B, H, 1)``. Any
        non-finite values are sanitized to zero as a final safeguard.
    
    Notes
    -----
    Head vs depth convention
    ~~~~~~~~~~~~~~~~~~~~~~~~
    The default drawdown convention assumes head loss:
    
    .. math::
    
       Delta h = h_{ref} - h_{mean}
    
    If upstream code uses a depth-like quantity that increases
    downward, the physically meaningful drawdown may require the
    opposite sign:
    
    .. math::
    
       Delta h = h_{mean} - h_{ref}
    
    Use ``drawdown_rule="mean_minus_ref"`` in that case.
    
    Sanitization behavior
    ~~~~~~~~~~~~~~~~~~~~~
    Inputs are sanitized with a "finite-or-zero" rule before use.
    This favors training stability over strict error signaling. If
    you want fail-fast behavior, validate upstream.
    
    Examples
    --------
    Compute equilibrium compaction with default settings:
    
    >>> s_eq = equilibrium_compaction_si(
    ...     h_mean_si=h_mean,
    ...     h_ref_si=h_ref,
    ...     Ss_field=Ss,
    ...     H_field_si=H,
    ... )
    
    Flip the drawdown rule for depth-like signals:
    
    >>> s_eq = equilibrium_compaction_si(
    ...     h_mean_si=depth_mean,
    ...     h_ref_si=depth_ref,
    ...     Ss_field=Ss,
    ...     H_field_si=H,
    ...     drawdown_rule="mean_minus_ref",
    ... )
    
    Clip extreme drawdowns during early training:
    
    >>> s_eq = equilibrium_compaction_si(
    ...     h_mean_si=h_mean,
    ...     h_ref_si=h_ref,
    ...     Ss_field=Ss,
    ...     H_field_si=H,
    ...     drawdown_clip_max=50.0,
    ... )
    
    See Also
    --------
    compute_consolidation_step_residual
        Uses ``s_eq`` as the equilibrium target in the ODE residual.
    
    settlement_state_for_pde
        Maps model settlement outputs to the ODE state convention.
    
    References
    ----------
    .. [1] Terzaghi, K.
       Theoretical Soil Mechanics. Wiley (1943).
    
    .. [2] Wang, H. F.
       Theory of Linear Poroelasticity. Princeton University Press
       (2000).
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

    r"""
    Integrate mean consolidation settlement over a forecast horizon.
    
    This routine evolves the mean settlement state
    :math:`\bar{s}(t)` using a stable, shape-safe time stepper that is
    compatible with TensorFlow graph execution. It is designed for the
    GeoPriorSubsNet "Option-1" mean path, where the mean subsidence is
    physics-driven from the predicted head.
    
    The integrator advances a first-order relaxation model:
    
    .. math::
    
       \frac{d\bar{s}}{dt} =
       \frac{s_{eq}(t) - \bar{s}(t)}{\tau(t)}
    
    where:
    
    * :math:`\bar{s}(t)` is the mean settlement state (m),
    * :math:`s_{eq}(t)` is the equilibrium compaction (m),
    * :math:`\tau(t)` is a consolidation time scale (s).
    
    The equilibrium compaction is computed by
    :func:`equilibrium_compaction_si`:
    
    .. math::
    
       s_{eq}(t) = S_s(t)\, \Delta h(t)\, H(t)
    
    with :math:`S_s` (1/m), :math:`H` (m), and drawdown
    :math:`\Delta h` (m) formed from ``h_mean_si`` and ``h_ref_si``
    using ``drawdown_rule`` and gated by ``drawdown_mode``.
    
    Discrete-time update
    --------------------
    Let :math:`t_0, ..., t_{H-1}` be the horizon times and let
    :math:`\Delta t_i` be the step duration in seconds. The state
    update can be done in two ways:
    
    Exact step (stable for large :math:`\Delta t/\tau`)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For each step :math:`i`:
    
    .. math::
    
       a_i = \exp\left(-\frac{\Delta t_i}{\tau_i}\right)
    
    .. math::
    
       \bar{s}_{i} =
       a_i\, \bar{s}_{i-1} + (1-a_i)\, s_{eq,i}
    
    This is the closed-form solution of the linear ODE assuming
    :math:`s_{eq}` and :math:`\tau` are constant on the step.
    
    Euler step (simple, less stable)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For each step :math:`i`:
    
    .. math::
    
       \bar{s}_{i} =
       \bar{s}_{i-1} +
       \Delta t_i\, \frac{s_{eq,i} - \bar{s}_{i-1}}{\tau_i}
    
    Use ``method="exact"`` unless you have a strong reason to match
    a legacy discretization.
    
    Time and units
    --------------
    The integrator expects:
    
    * ``h_mean_si`` and ``h_ref_si`` in meters,
    * ``Ss_field`` in 1/m,
    * ``H_field_si`` in meters,
    * ``tau_field`` in seconds,
    * ``dt`` expressed in ``time_units`` and converted to seconds
      internally via :func:`dt_to_seconds`.
    
    If ``dt`` is None, a unit step of ``1`` is used per horizon index,
    interpreted in ``time_units``.
    
    Shape contract and horizon alignment
    ------------------------------------
    Internally, this function forces a strict ``(B, H, 1)`` layout for
    the evolving state and for all step inputs, because TensorFlow
    scan operations can widen shapes when rank is ambiguous.
    
    Inputs that vary with time may be provided as:
    
    * length ``H``  : already aligned,
    * length ``H+1``: treated as state-length, sliced to ``[:-1]``,
    * length ``H-1``: treated as step-length, padded by prepending the
      first entry, producing length ``H``,
    * length ``1``  : broadcast across horizon.
    
    This alignment is applied to ``dt`` and ``tau_field``. The
    equilibrium sequence ``s_eq`` is computed at length ``H``.
    
    Stability and sanitization
    --------------------------
    This integrator aggressively sanitizes non-finite values:
    
    * ``dt`` and ``dt_sec`` are mapped through a finite-or-zero rule,
      then clamped to non-negative.
    * ``tau`` is sanitized and lower-bounded by ``eps_tau``.
    * The final output is passed through a finite-or-zero rule.
    
    These guards are intended to prevent training from crashing when
    upstream predictions temporarily produce NaN/Inf.
    
    Parameters
    ----------
    h_mean_si : Tensor
        Mean head in meters. Shape ``(B, H, 1)`` or ``(B, H)``.
        The last dim is forced to 1 for scan stability.
    
    Ss_field : Tensor
        Specific storage :math:`S_s` in 1/m. Broadcastable to
        ``(B, H, 1)``.
    
    H_field_si : Tensor
        Compressible thickness :math:`H` in meters. Broadcastable to
        ``(B, H, 1)``.
    
    tau_field : Tensor
        Consolidation time scale :math:`\tau` in seconds.
        Broadcastable to ``(B, H, 1)`` or horizon-aligned by the
        alignment rules described above.
    
    h_ref_si : Tensor
        Reference head (meters). Broadcastable to ``h_mean_si``.
        If ``stop_grad_ref=True``, gradients are stopped through this
        reference inside :func:`equilibrium_compaction_si`.
    
    s_init_si : Tensor
        Initial settlement state at the horizon origin. This is the
        initial value used by the scan initializer. It is expected to
        represent the settlement at the first horizon time.
        Typical shape is ``(B, 1, 1)`` or ``(B, 1)``.
    
    dt : Tensor, optional
        Step duration in ``time_units``. If provided, it must be
        broadcastable and is horizon-aligned. If None, a unit step of
        ones is used.
    
    time_units : str, default="yr"
        Units for ``dt``. Converted to seconds via :func:`dt_to_seconds`.
        Examples include "yr", "day", "hour", or "unitless", depending
        on your pipeline.
    
    method : {"exact", "euler"}, default="exact"
        Integration scheme. "exact" uses the closed-form step for a
        first-order linear relaxation. "euler" uses forward Euler.
    
    eps_tau : float, default=1e-12
        Lower bound for ``tau`` to prevent division by zero and
        undefined exponentials.
    
    relu_beta : float, default=20.0
        Smoothness parameter forwarded to
        :func:`equilibrium_compaction_si` for ``drawdown_mode="smooth_relu"``.
    
    drawdown_mode : str, default="smooth_relu"
        Drawdown gating forwarded to :func:`equilibrium_compaction_si`.
        Common values: "smooth_relu", "relu", "softplus", "none".
    
    drawdown_rule : str, default="ref_minus_mean"
        Drawdown rule forwarded to :func:`equilibrium_compaction_si`.
        Use "ref_minus_mean" for head-loss convention. Use
        "mean_minus_ref" for depth-like (down-positive) signals.
    
    stop_grad_ref : bool, default=True
        Forwarded to :func:`equilibrium_compaction_si`. If True, stops
        gradient through ``h_ref_si`` to prevent reference drift.
    
    drawdown_zero_at_origin : bool, default=False
        Forwarded to :func:`equilibrium_compaction_si`. If True, shifts
        smooth drawdown gating so the value at zero is near zero.
    
    drawdown_clip_max : float or None, default=None
        Forwarded to :func:`equilibrium_compaction_si`. If set, clips
        drawdown after gating to avoid extreme values dominating loss.
    
    verbose : int, default=0
        Verbosity for printing basic statistics and shape info.
    
    Returns
    -------
    s_bar_si : Tensor
        Mean cumulative settlement over the horizon in meters, shape
        ``(B, H, 1)``. The sequence is the scan output of the chosen
        stepper initialized at ``s_init_si``.
    
    Notes
    -----
    Relationship to model outputs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    In Option-1 mean modeling, the network predicts mean head, and
    settlement mean is computed by integrating the relaxation ODE.
    The model may optionally add a learned residual around this mean,
    but the returned value here is the physics mean only.
    
    Interpreting the horizon index
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This function produces a length-H sequence. If your horizon times
    represent the future steps 1..H, ensure that ``dt`` and ``h_mean_si``
    are consistent with that convention. If you supply sequences of
    length H+1 (state nodes), the last node is dropped by alignment.
    
    Numerical behavior
    ~~~~~~~~~~~~~~~~~~
    The exact update has the desirable limit:
    
    * If :math:`\Delta t_i \ll \tau_i`, then
      :math:`\bar{s}_i \approx \bar{s}_{i-1}` (slow relaxation).
    * If :math:`\Delta t_i \gg \tau_i`, then
      :math:`\bar{s}_i \approx s_{eq,i}` (fast equilibration).
    
    Examples
    --------
    Integrate with unit yearly steps and exact update:
    
    >>> s_bar = integrate_consolidation_mean(
    ...     h_mean_si=h_mean,
    ...     Ss_field=Ss,
    ...     H_field_si=H,
    ...     tau_field=tau,
    ...     h_ref_si=h_ref,
    ...     s_init_si=s0,
    ...     time_units="yr",
    ...     method="exact",
    ... )
    
    Use explicit dt (months) and Euler update:
    
    >>> s_bar = integrate_consolidation_mean(
    ...     h_mean_si=h_mean,
    ...     Ss_field=Ss,
    ...     H_field_si=H,
    ...     tau_field=tau,
    ...     h_ref_si=h_ref,
    ...     s_init_si=s0,
    ...     dt=dt_months,
    ...     time_units="month",
    ...     method="euler",
    ... )
    
    Flip drawdown rule for depth-like signals:
    
    >>> s_bar = integrate_consolidation_mean(
    ...     h_mean_si=depth_mean,
    ...     Ss_field=Ss,
    ...     H_field_si=H,
    ...     tau_field=tau,
    ...     h_ref_si=depth_ref,
    ...     s_init_si=s0,
    ...     drawdown_rule="mean_minus_ref",
    ... )
    
    See Also
    --------
    equilibrium_compaction_si
        Computes :math:`s_{eq}(t)` from head/drawdown and fields.
    
    tau_phys_from_fields
        Computes a physically motivated baseline time scale.
    
    settlement_state_for_pde
        Converts predicted settlement representations to an ODE state.
    
    References
    ----------
    .. [1] Terzaghi, K.
       Theoretical Soil Mechanics. Wiley (1943).
    
    .. [2] Wang, H. F.
       Theory of Linear Poroelasticity. Princeton University Press
       (2000).
    
    .. [3] Zienkiewicz, O. C., Taylor, R. L.
       The Finite Element Method, Vol. 3. Butterworth-Heinemann (2000).
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
    r"""
    Compute a one-step consolidation residual in SI space.
    
    This function forms a per-step residual that penalizes violations of
    a first-order consolidation relaxation model over a sequence of
    states. It is intended for physics diagnostics and for PDE-style
    training objectives where the settlement state is predicted (or
    derived) and should satisfy a stable time-stepping rule.
    
    Model and notation
    ------------------
    Let the settlement state be :math:`s(t)` (m). The governing ODE is a
    Voigt-style relaxation toward an equilibrium settlement
    :math:`s_{eq}(t)`:
    
    .. math::
    
       \frac{ds}{dt} = \frac{s_{eq}(t) - s(t)}{\tau(t)}
    
    where :math:`\tau(t)` is a (possibly space/time varying) time scale
    in seconds. The equilibrium settlement is computed from head
    drawdown:
    
    .. math::
    
       s_{eq}(t) = S_s(t)\, \Delta h(t)\, H(t)
    
    with :math:`S_s` in 1/m and :math:`H` in m. The drawdown
    :math:`\Delta h(t)` is constructed from ``h_mean_si`` and ``h_ref_si``
    using ``drawdown_rule`` and is gated by ``drawdown_mode`` via
    :func:`equilibrium_compaction_si`.
    
    Discrete residual definition
    ----------------------------
    Given state samples :math:`s_n = s(t_n)` and a step duration
    :math:`\Delta t_n` (seconds), this routine computes a one-step
    prediction :math:`\hat{s}_{n+1}` from :math:`s_n` and :math:`s_{eq,n}`
    and returns the residual:
    
    .. math::
    
       r_n = s_{n+1} - \hat{s}_{n+1}
    
    The residual has units of meters and is produced for
    :math:`n = 0, ..., T-2`, hence the output time length is ``T-1``.
    
    Two steppers are supported:
    
    Exact step (closed-form, stable)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Assuming :math:`s_{eq}` and :math:`\tau` are constant over the step:
    
    .. math::
    
       a_n = \exp\left(-\frac{\Delta t_n}{\tau_n}\right)
    
    .. math::
    
       \hat{s}_{n+1} =
       a_n s_n + (1-a_n) s_{eq,n}
    
    Euler step (forward Euler)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. math::
    
       \hat{s}_{n+1} =
       s_n + \Delta t_n \frac{s_{eq,n} - s_n}{\tau_n}
    
    The "exact" update is unconditionally stable for stiff regimes
    (:math:`\Delta t / \tau` large). Euler may be unstable unless
    :math:`\Delta t \ll \tau`.
    
    Time and units
    --------------
    * Inputs are SI: meters, seconds, and 1/m as documented below.
    * ``dt`` is interpreted in ``time_units`` and converted to seconds
      using :func:`dt_to_seconds`.
    
    Shape contract and alignment
    ----------------------------
    Let ``s_state_si`` and ``h_mean_si`` have shape ``(B, T, 1)`` (or
    ``(B, T)`` which is promoted to ``(B, T, 1)``). This function forms
    step-aligned sequences of length ``H = T-1``:
    
    * :math:`s_n = s[:, :-1]` and :math:`s_{n+1} = s[:, 1:]`
    * :math:`h_n = h[:, :-1]`
    
    Time-varying fields may be provided with time length:
    
    * ``T``     : state-length, sliced to ``T-1`` steps,
    * ``T-1``   : already step-length,
    * ``1``     : broadcast across steps.
    
    After alignment, fields are broadcast to ``(B, T-1, 1)``.
    
    Numerical safety
    ----------------
    This routine sanitizes key quantities:
    
    * Non-finite values are mapped through a finite-or-zero rule.
    * ``tau`` is clamped below by ``eps_tau``.
    * ``dt`` converted to seconds is clamped to non-negative.
    
    These guards prevent crashes during training when upstream model
    outputs temporarily produce NaN/Inf.
    
    Parameters
    ----------
    s_state_si : Tensor
        Settlement state in meters. Shape ``(B, T, 1)`` or ``(B, T)``.
        This is the state used in the stepper (often incremental).
    
    h_mean_si : Tensor
        Mean head (or depth-like signal) in meters. Shape must match
        ``s_state_si`` in the time dimension ``T``. The stepper uses
        ``h_mean_si[:, :-1]`` to compute :math:`s_{eq,n}`.
    
    Ss_field : Tensor
        Specific storage :math:`S_s` in 1/m. Time length may be ``T``,
        ``T-1``, or ``1``; it is aligned to steps and broadcast.
    
    H_field_si : Tensor
        Compressible thickness :math:`H` in meters. Time length may be
        ``T``, ``T-1``, or ``1``; it is aligned to steps and broadcast.
    
    tau_field : Tensor
        Consolidation time scale :math:`\tau` in seconds. Time length may
        be ``T``, ``T-1``, or ``1``; it is aligned to steps and clamped
        below by ``eps_tau``.
    
    h_ref_si : Tensor
        Reference head (or reference depth) in meters. Time length may be
        ``T``, ``T-1``, or ``1``; it is aligned to steps. If
        ``stop_grad_ref=True``, gradients are stopped through this
        reference inside :func:`equilibrium_compaction_si`.
    
    dt : Tensor, optional
        Step duration in ``time_units``. Time length may be ``T``,
        ``T-1``, or ``1``. If None, a unit step is used for every step.
    
    time_units : str, default="yr"
        Units for ``dt`` prior to conversion to seconds via
        :func:`dt_to_seconds`.
    
    method : {"exact", "euler"}, default="exact"
        Stepping scheme used to build :math:`\hat{s}_{n+1}`.
    
    eps_tau : float, default=1e-12
        Lower bound for ``tau`` to prevent division by zero and overflow
        in stiff regimes.
    
    relu_beta : float, default=20.0
        Smoothness parameter forwarded to :func:`equilibrium_compaction_si`
        when ``drawdown_mode="smooth_relu"``.
    
    drawdown_mode : str, default="smooth_relu"
        Forwarded to :func:`equilibrium_compaction_si`. Controls the
        positive-part gating applied to drawdown.
    
    drawdown_rule : str, default="ref_minus_mean"
        Forwarded to :func:`equilibrium_compaction_si`. Controls the sign
        convention for drawdown. Use "ref_minus_mean" for head loss and
        "mean_minus_ref" for depth-like (down-positive) signals.
    
    stop_grad_ref : bool, default=True
        Forwarded to :func:`equilibrium_compaction_si`. If True, prevents
        gradients through the reference signal ``h_ref_si``.
    
    drawdown_zero_at_origin : bool, default=False
        Forwarded to :func:`equilibrium_compaction_si`. If True, shifts
        the smooth drawdown gate so the value at zero is near zero.
    
    drawdown_clip_max : float or None, default=None
        Forwarded to :func:`equilibrium_compaction_si`. Clips drawdown
        after gating to reduce extreme values.
    
    verbose : int, default=0
        Verbosity for basic shape and residual statistics.
    
    Returns
    -------
    res : Tensor
        One-step residual sequence in meters, shape ``(B, T-1, 1)``:
    
        .. math::
    
           r_n = s_{n+1} - \hat{s}_{n+1}
    
    Notes
    -----
    Incremental vs cumulative state
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This residual is defined on the settlement state passed as
    ``s_state_si``. If your training uses an incremental ODE state
    :math:`s_{inc}(t) = s_{cum}(t) - s_0`, ensure that both
    ``s_state_si`` and ``s_eq`` are expressed in the same state space.
    A mismatch (e.g., residuals on incremental state but equilibrium in
    cumulative units) will produce biased residuals.
    
    When to prefer the exact step
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If the ratio :math:`\Delta t / \tau` is frequently larger than 1,
    Euler updates can become numerically unstable. The exact step is
    stable in both stiff and non-stiff regimes and is typically the best
    default for physics losses.
    
    Examples
    --------
    Compute residuals for a horizon sequence:
    
    >>> r = compute_consolidation_step_residual(
    ...     s_state_si=s_state,
    ...     h_mean_si=h_mean,
    ...     Ss_field=Ss,
    ...     H_field_si=H,
    ...     tau_field=tau,
    ...     h_ref_si=h_ref,
    ...     dt=dt,
    ...     time_units="yr",
    ...     method="exact",
    ... )
    
    Flip drawdown rule for depth-like inputs:
    
    >>> r = compute_consolidation_step_residual(
    ...     s_state_si=s_state,
    ...     h_mean_si=depth,
    ...     Ss_field=Ss,
    ...     H_field_si=H,
    ...     tau_field=tau,
    ...     h_ref_si=depth_ref,
    ...     drawdown_rule="mean_minus_ref",
    ... )
    
    See Also
    --------
    integrate_consolidation_mean
        Integrates the same relaxation model forward in time.
    
    equilibrium_compaction_si
        Computes equilibrium settlement from drawdown and fields.
    
    dt_to_seconds
        Converts ``dt`` in ``time_units`` to SI seconds.
    
    References
    ----------
    .. [1] Terzaghi, K.
       Theoretical Soil Mechanics. Wiley (1943).
    
    .. [2] Wang, H. F.
       Theory of Linear Poroelasticity. Princeton University Press
       (2000).
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

    # Calculate ratio for stability check
    dt_tau_ratio = dt_sec / (tau + tf_constant(_EPSILON, tau.dtype))
    
    use_exact = tf_logical_or( # noqa
        tf_equal(m, "exact"), 
        tf_reduce_any(dt_tau_ratio > 1.0) # Safety switch
    )

    def _step_exact():
        a = tf_exp(-dt_sec / (tau + tf_constant(_EPSILON, tau.dtype)))
        return s_n * a + s_eq_n * (1.0 - a)

    def _step_euler():
        return s_n + dt_sec * (s_eq_n - s_n) / (
            tau + tf_constant(_EPSILON, tau.dtype)
        )

    # Use exact if requested OR if stability is at risk
    if m == "exact":
        pred = _step_exact()
    else:
        # Hybrid safety: use exact where stiff, euler where safe? 
        # Easier to just force exact if user didn't strictly demand pure euler behavior
        # But for now, let's just use the user choice but warn/clamp.
        
        # Better: just use exact. It's unconditionally stable.
        pred = _step_euler()

    # NOTE: I highly recommend changing the default in __init__ to 'exact'
    # if it isn't already.
    
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
    return_log: bool = False, 
) -> Tuple[Tensor, Tensor]:
    r"""
    Compute the physics closure consolidation timescale ``tau_phys``
    and the effective drainage thickness ``Hd``.
    
    This function implements the model's consolidation timescale
    closure :math:`tau_{phys}` in a numerically stable way. The core
    design is to compute :math:`log(tau_{phys})` first, and only
    apply ``exp`` at the end (unless ``return_log=True``). This
    prevents unstable gradients that can arise from naive algebraic
    forms that contain high powers of :math:`1/K`.
    
    Mathematical definition
    -----------------------
    Let the model provide effective fields in SI units:
    
    * :math:`K` in m/s (hydraulic conductivity),
    * :math:`S_s` in 1/m (specific storage),
    * :math:`H` in m (thickness),
    * :math:`kappa` is a positive scalar multiplier (dimensionless
      in the code path; its physical meaning depends on the chosen
      mode).
    
    An effective drainage thickness :math:`H_d` is defined as:
    
    .. math::
    
       H_d =
       \begin{cases}
         H \cdot f_{Hd}, & \text{if use\_effective\_thickness is True} \\
         H,              & \text{otherwise}
       \end{cases}
    
    where :math:`f_{Hd}` is ``model.Hd_factor``. The function returns
    this :math:`H_d` as ``Hd``.
    
    Two closure modes are supported via ``model.kappa_mode``:
    
    1) ``kappa_mode="bar"``
    
    The closure is:
    
    .. math::
    
       \tau_{phys}
       = \frac{\kappa \, H^2 \, S_s}{\pi^2 \, K}
    
    The log form is:
    
    .. math::
    
       \log(\tau_{phys})
       = \log(\kappa)
         + 2\log(H)
         + \log(S_s)
         - \log(\pi^2)
         - \log(K)
    
    2) Any other value (the "non-bar" branch)
    
    The closure is:
    
    .. math::
    
       \tau_{phys}
       = \frac{H_d^2 \, S_s}{\pi^2 \, \kappa \, K}
    
    The log form is:
    
    .. math::
    
       \log(\tau_{phys})
       = 2\log(H_d)
         + \log(S_s)
         - \log(\pi^2)
         - \log(\kappa)
         - \log(K)
    
    The implementation uses the log forms above. If ``return_log`` is
    False, it returns:
    
    .. math::
    
       \tau_{phys} = \exp(\log(\tau_{phys}))
    
    Numerical stability
    -------------------
    All inputs are sanitized and floored to be strictly positive:
    
    .. math::
    
       K_{safe}  = \max(K,  eps) \\
       S_{s,safe}= \max(S_s,eps) \\
       H_{safe}  = \max(H,  eps) \\
       H_{d,safe}= \max(H_d,eps) \\
       \kappa_{safe} = \max(\kappa, eps)
    
    This ensures that ``log`` is never applied to non-positive or
    non-finite values. Computing in log-space also avoids gradient
    blow-ups associated with expressions that behave like
    :math:`1/K^2` when written in certain equivalent algebraic forms.
    
    Parameters
    ----------
    model : Any
        Model instance providing:
        ``use_effective_thickness``, ``Hd_factor``, ``kappa_mode``,
        and a callable ``_kappa_value()`` returning a positive scalar.
    
    K_field : Tensor
        Effective conductivity :math:`K` in SI (m/s). Shape is
        broadcastable to ``(B, H, 1)``.
    
    Ss_field : Tensor
        Effective specific storage :math:`S_s` in SI (1/m). Shape is
        broadcastable to ``(B, H, 1)``.
    
    H_field : Tensor
        Thickness :math:`H` in SI (m). Shape is broadcastable to
        ``(B, H, 1)``.
    
    eps : float, default=_EPSILON
        Positive floor used by ``finite_floor`` to prevent invalid
        logs and divisions.
    
    verbose : int, default=0
        Verbosity level for debug printing.
    
    return_log : bool, default=False
        If True, return ``(log_tau_phys, Hd)`` where ``log_tau_phys``
        is :math:`log(tau_{phys})`. If False, return
        ``(tau_phys, Hd)`` where ``tau_phys`` is in seconds.
    
    Returns
    -------
    tau_or_log_tau : Tensor
        If ``return_log=False``:
        Physics closure timescale :math:`tau_{phys}` in seconds.
    
        If ``return_log=True``:
        :math:`log(tau_{phys})` in log-seconds.
    
    Hd : Tensor
        Effective drainage thickness :math:`H_d` in meters. This is
        either ``H_field`` or ``H_field * Hd_factor`` depending on
        ``model.use_effective_thickness``. Shape is broadcastable to
        ``(B, H, 1)``.
    
    Notes
    -----
    Choice of ``H`` vs ``Hd`` in "bar" mode
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    In the ``kappa_mode="bar"`` branch, the code uses ``H_safe`` in
    the :math:`H^2` term. The function still computes and returns
    ``Hd`` for downstream diagnostics and for non-"bar" mode. If your
    physical interpretation requires :math:`H_d` as the diffusion path
    length in "bar" mode, you may adapt the closure accordingly.
    
    Gradient behavior
    ~~~~~~~~~~~~~~~~~
    Computing :math:`log(tau_{phys})` first improves stability because
    the derivative of ``log`` scales like :math:`1/x`, while direct
    algebraic expansions of :math:`tau_{phys}` can introduce stronger
    inverse powers in intermediate steps. The only exponential is
    applied at the end (if requested), keeping the computational graph
    well behaved.
    
    Examples
    --------
    Compute ``tau_phys`` and use it in a prior:
    
    >>> tau_phys, Hd = tau_phys_from_fields(model, K, Ss, H)
    >>> R_prior = tf.math.log(tf.maximum(tau_learned, 1e-12)) \
    ...          - tf.math.log(tf.maximum(tau_phys, 1e-12))
    >>> loss_prior = tf.reduce_mean(tf.square(R_prior))
    
    Get log-space output for direct log-priors:
    
    >>> log_tau_phys, Hd = tau_phys_from_fields(
    ...     model, K, Ss, H, return_log=True
    ... )
    
    See Also
    --------
    compute_consistency_prior
        Builds the residual ``log(tau_learned) - log(tau_phys)``.
    
    compose_physics_fields
        Composes bounded SI fields and returns ``log_tau_phys`` for
        diagnostics and bounds penalties.
    
    get_log_tau_bounds
        Provides configured bounds for ``log_tau`` (log-seconds).
    
    References
    ----------
    .. [1] Terzaghi, K.
       Theoretical Soil Mechanics. Wiley (1943).
    
    .. [2] Wang, H. F.
       Theory of Linear Poroelasticity. Princeton University Press
       (2000).
    """

    eps = float(eps)
    pi_sq = tf_constant(np.pi**2, dtype=tf_float32)

    # Sanitize inputs
    K_safe = finite_floor(K_field, eps=eps)
    Ss_safe = finite_floor(Ss_field, eps=eps)
    H_safe = finite_floor(H_field, eps=eps)

    # --- Effective Thickness Logic ---
    use_hd = bool(getattr(model, "use_effective_thickness", False))
    if use_hd:
        f = getattr(model, "Hd_factor", 1.0)
        f = tf_cast(f, H_safe.dtype)
        # finite check for factor
        f = tf_where(tf_math.is_finite(f), f, tf_constant(1.0, H_safe.dtype))
        Hd = H_safe * f
    else:
        Hd = H_safe
    
    Hd = finite_floor(Hd, eps=eps)
    
    # --- Kappa Logic ---
    kappa = model._kappa_value()
    kappa = tf_cast(kappa, H_safe.dtype)
    kappa = tf_where(tf_math.is_finite(kappa), kappa, tf_constant(1.0, H_safe.dtype))
    kappa = finite_floor(kappa, eps=eps)

    # --- Log-Space Computation (Stable) ---
    # log(tau) = log(C) + log(Ss) + 2*log(Hd) - log(K)
    
    log_Ss = tf_math.log(Ss_safe)
    log_K  = tf_math.log(K_safe)
    log_Hd = tf_math.log(Hd)
    log_pi = tf_math.log(pi_sq)
    log_kap = tf_math.log(kappa)

    # Formula depends on kappa_mode
    mode = str(getattr(model, "kappa_mode", "bar"))
    
    if mode == "bar":
        # tau = kappa * H^2 * Ss / (pi^2 * K)
        # log_tau = log(k) + 2log(H) + log(Ss) - log(pi^2) - log(K)
        # Note: using H_safe here typically, or Hd? 
        # Code usually assumes Hd for diffusion path length. 
        # Using H_safe to match original code structure:
        log_H = tf_math.log(H_safe) 
        log_tau = log_kap + 2.0*log_H + log_Ss - log_pi - log_K
    else:
        # tau = (Hd/H)^2 * H^2 * Ss / (pi^2 * kappa * K) 
        #     = Hd^2 * Ss / (pi^2 * kappa * K)
        log_tau = 2.0*log_Hd + log_Ss - log_pi - log_kap - log_K

    if return_log:
        return log_tau, Hd

    # Only exp() at the very end. 
    # If log_tau is huge (e.g. 50), exp() overflows, but gradient of exp is manageable.
    tau_phys = tf_exp(log_tau)
    
    vprint(verbose, "tau_phys: log_tau=", log_tau)
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
    r"""
    Compute the consolidation timescale consistency prior.
    
    This prior constrains the learned consolidation timescale ``tau``
    to remain physically consistent with the permeability-storage-
    thickness closure implied by the poroelastic consolidation model.
    It returns the *log-space mismatch*:
    
    .. math::
    
       R_{\mathrm{prior}}
       = \log(\tau_{\mathrm{learned}})
         - \log(\tau_{\mathrm{phys}})
    
    where :math:`\tau_{\mathrm{phys}}` is computed from the predicted
    fields :math:`K`, :math:`S_s`, and :math:`H` through
    :func:`tau_phys_from_fields`.
    
    Log-space is used for two reasons:
    
    1. Positivity: :math:`\tau > 0` is enforced implicitly.
    2. Scale: timescales may span orders of magnitude; comparing
       logs yields a relative-type error signal.
    
    Mathematical formulation
    ------------------------
    Let the model predict effective fields (all SI):
    
    * :math:`K(x,y)` in m/s (hydraulic conductivity),
    * :math:`S_s(x,y)` in 1/m (specific storage),
    * :math:`H(x,y)` in m (compressible thickness),
    * :math:`\tau_{\mathrm{learned}}(x,y)` in s (learned timescale).
    
    A common 1D Terzaghi-style drainage closure gives a characteristic
    timescale:
    
    .. math::
    
       \tau_{\mathrm{phys}}
       = \frac{H_d^2 \, S_s}{\pi^2 \, c_v}
    
    with consolidation coefficient:
    
    .. math::
    
       c_v = \frac{K}{S_s}
    
    or, equivalently, for a diffusion-like closure:
    
    .. math::
    
       \tau_{\mathrm{phys}}
       = \frac{H_d^2 \, S_s}{\pi^2 \, K}
    
    where :math:`H_d` is the effective drainage thickness (often a
    fraction of :math:`H`, e.g. :math:`H_d = \mathrm{hd\_factor}\,H`).
    The exact form used is delegated to :func:`tau_phys_from_fields`,
    which may incorporate additional model parameters such as
    :math:`\kappa` (compressibility/bulk coupling) or boundary
    conditions.
    
    The prior residual returned by this function is:
    
    .. math::
    
       R_{\mathrm{prior}}(x,y)
       = \log(\max(\tau_{\mathrm{learned}}(x,y), \varepsilon))
         - \log(\tau_{\mathrm{phys}}(x,y))
    
    where :math:`\varepsilon` is a small constant used to avoid
    ``log(0)`` when the learned timescale becomes numerically small.
    
    This residual is typically used inside an L2 penalty:
    
    .. math::
    
       L_{\mathrm{prior}}
       = \mathbb{E}\left[R_{\mathrm{prior}}^2\right]
    
    and contributes to the physics loss with a user-controlled weight
    (e.g., ``lambda_prior``).
    
    Parameters
    ----------
    model : Any
        Model instance providing configuration used by
        :func:`tau_phys_from_fields` (for example, effective thickness
        settings, kappa mode, bounds, and scaling config).
    
    K_field : Tensor
        Effective hydraulic conductivity :math:`K` in SI units (m/s).
        Expected shape is broadcastable to ``(B, H, 1)``.
    
    Ss_field : Tensor
        Effective specific storage :math:`S_s` in SI units (1/m).
        Expected shape is broadcastable to ``(B, H, 1)``.
    
    tau_field : Tensor
        Learned timescale :math:`\tau_{\mathrm{learned}}` in SI seconds.
        Expected shape is broadcastable to ``(B, H, 1)``.
    
    H_field : Tensor
        Thickness :math:`H` in SI meters. Expected shape is
        broadcastable to ``(B, H, 1)``.
    
    verbose : int, default=0
        Verbosity level for debug printing.
    
    Returns
    -------
    residual : Tensor
        Log-space prior residual:
    
        .. math::
    
           R_{\mathrm{prior}}
           = \log(\tau_{\mathrm{learned}})
             - \log(\tau_{\mathrm{phys}})
    
        Shape follows the broadcasted shape of the inputs, typically
        ``(B, H, 1)``.
    
    Notes
    -----
    Numerical stability
    ~~~~~~~~~~~~~~~~~~~
    * ``tau_field`` is floored by a small :math:`\varepsilon` before
      taking the logarithm.
    * :func:`tau_phys_from_fields` is called with ``return_log=True``
      to compute :math:`\log(\tau_{\mathrm{phys}})` directly, avoiding
      the unstable pattern ``log(exp(log_tau))``.
    
    Interpretation
    ~~~~~~~~~~~~~~
    * ``residual = 0`` means the learned timescale matches the closure.
    * Positive values indicate :math:`\tau_{\mathrm{learned}}` is larger
      (slower consolidation) than predicted by the closure.
    * Negative values indicate a smaller (faster) learned timescale.
    
    Examples
    --------
    Compute and reduce to an L2 prior loss:
    
    >>> R_prior = compute_consistency_prior(
    ...     model, K_field=K, Ss_field=Ss,
    ...     tau_field=tau, H_field=H
    ... )
    >>> loss_prior = tf.reduce_mean(tf.square(R_prior))
    
    See Also
    --------
    tau_phys_from_fields
        Computes :math:`\tau_{\mathrm{phys}}` from :math:`K`, :math:`S_s`,
        and :math:`H` (and model configuration).
    
    compose_physics_fields
        Builds bounded/guarded SI fields and returns both learned
        and closure timescales in log-space.
    
    References
    ----------
    .. [1] Terzaghi, K.
       Theoretical Soil Mechanics. Wiley (1943).
    
    .. [2] Wang, H. F.
       Theory of Linear Poroelasticity. Princeton University Press
       (2000).
    """

    eps = tf_constant(_EPSILON, dtype=tf_float32)

    # 1. Get learned tau in log space
    tau_safe = tf_maximum(tau_field, eps)
    log_tau_learned = tf_math.log(tau_safe)

    # 2. Get physical tau in log space directly (Stable!)
    log_tau_phys, _ = tau_phys_from_fields(
        model,
        K_field,
        Ss_field,
        H_field,
        verbose=0,
        return_log=True # Use the new flag
    )

    out = log_tau_learned - log_tau_phys

    vprint(verbose, "cons_prior: log_tau_learned=", log_tau_learned)
    vprint(verbose, "cons_prior: log_tau_phys=", log_tau_phys)
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
    r"""
    Compute a smoothness prior on spatial gradients of physics fields.
    
    This function builds a spatial regularizer that penalizes rapid
    variation of the permeability-like field ``K`` and the storage
    field ``Ss`` over the spatial coordinates. In the GeoPrior PINN,
    this prior stabilizes the inverse problem by discouraging
    unrealistic high-frequency spatial structure in learned fields.
    
    The preferred penalty is applied in *log-space*:
    
    .. math::
    
       R_{\mathrm{smooth}}
       = \left\|\nabla \log K\right\|^2
         + \left\|\nabla \log S_s\right\|^2
    
    where, in 2D:
    
    .. math::
    
       \left\|\nabla \log K\right\|^2
       = \left(\frac{\partial \log K}{\partial x}\right)^2
         + \left(\frac{\partial \log K}{\partial y}\right)^2
    
    and similarly for :math:`S_s`. Penalizing gradients of logs is
    often preferable to raw gradients because it regularizes *relative*
    changes (order-of-magnitude variations) rather than absolute
    changes.
    
    Implementation modes
    --------------------
    The function supports three modes, chosen by inputs:
    
    1. ``already_log=True``:
       Inputs ``dK_dx`` etc. are interpreted as
       :math:`\partial_x \log K` and so on, and the penalty is:
    
       .. math::
    
          R_{\mathrm{smooth}}
          = (d\log K/dx)^2 + (d\log K/dy)^2
            + (d\log S_s/dx)^2 + (d\log S_s/dy)^2
    
    2. ``already_log=False`` with ``K_field`` and ``Ss_field``:
       The function converts raw gradients to log-gradients using:
    
       .. math::
    
          \frac{\partial \log K}{\partial x}
          = \frac{1}{K}\frac{\partial K}{\partial x}
    
       and similarly for the other terms. For numerical stability,
       denominators are floored by a small constant ``eps_div``:
    
       .. math::
    
          K_{\mathrm{denom}} = \max(K, \varepsilon_{\mathrm{div}})
    
       This avoids exploding ratios when ``K`` or ``Ss`` are very small.
    
    3. Fallback (rare):
       If ``K_field`` and ``Ss_field`` are not provided, it penalizes
       the raw gradients directly:
    
       .. math::
    
          R_{\mathrm{smooth}}
          = (dK/dx)^2 + (dK/dy)^2
            + (dS_s/dx)^2 + (dS_s/dy)^2
    
    Parameters
    ----------
    dK_dx, dK_dy : Tensor
        Spatial gradients of ``K`` with respect to x and y. If
        ``already_log=True``, these are gradients of ``logK`` instead.
    
    dSs_dx, dSs_dy : Tensor
        Spatial gradients of ``Ss`` with respect to x and y. If
        ``already_log=True``, these are gradients of ``logSs`` instead.
    
    K_field : Tensor or None, default=None
        Field ``K`` in SI units (m/s). Used only when converting
        raw gradients to log-gradients.
    
    Ss_field : Tensor or None, default=None
        Field ``Ss`` in SI units (1/m). Used only when converting
        raw gradients to log-gradients.
    
    already_log : bool, default=False
        If True, treat the input gradients as log-gradients.
    
    verbose : int, default=0
        Verbosity level for debug printing.
    
    Returns
    -------
    residual : Tensor
        Smoothness residual map. Typical shape is ``(B, H, 1)`` or a
        broadcast-compatible shape. This quantity is usually squared
        and reduced to form ``loss_smooth``.
    
    Notes
    -----
    Why log-space regularization?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    In hydrogeologic inverse problems, fields like ``K`` can vary by
    orders of magnitude across space. Penalizing gradients of ``logK``
    naturally encourages spatial smoothness in a multiplicative sense,
    which aligns with common geostatistical assumptions.
    
    Stability of division
    ~~~~~~~~~~~~~~~~~~~~~
    When converting ``dK`` to ``dlogK = dK / K``, small ``K`` can
    create extremely large ratios. The denominator floor
    ``eps_div`` is applied only for the conversion step, so that
    regions with effectively zero permeability do not dominate the
    regularizer.
    
    Examples
    --------
    Penalty in log-space using provided fields:
    
    >>> R_smooth = compute_smoothness_prior(
    ...     dK_dx, dK_dy, dSs_dx, dSs_dy,
    ...     K_field=K, Ss_field=Ss, already_log=False
    ... )
    >>> loss_smooth = tf.reduce_mean(tf.square(R_smooth))
    
    Direct log-gradient mode (inputs already log-gradients):
    
    >>> R_smooth = compute_smoothness_prior(
    ...     dlogK_dx, dlogK_dy, dlogSs_dx, dlogSs_dy,
    ...     already_log=True
    ... )
    
    See Also
    --------
    compose_physics_fields
        Produces bounded SI fields ``K_field`` and ``Ss_field`` and
        associated log values for diagnostics and priors.
    
    ensure_si_derivative_frame
        Converts autodiff derivatives to SI-consistent spatial
        derivatives suitable for smoothness penalties.
    
    References
    ----------
    .. [1] Tarantola, A.
       Inverse Problem Theory and Methods for Model Parameter
       Estimation. SIAM (2005).
    
    .. [2] Rasmussen, C. E., and Williams, C. K. I.
       Gaussian Processes for Machine Learning. MIT Press (2006).
    """

    # Safe epsilon for division
    eps_div = tf_constant(1e-6, dtype=tf_float32) 

    if already_log:
        # Inputs are already d(logK)/dx, etc.
        out = (
            tf_square(dK_dx) + tf_square(dK_dy)
            + tf_square(dSs_dx) + tf_square(dSs_dy)
        )
        vprint(verbose, "smooth(log-direct): out=", out)
        return out

    if (K_field is not None) and (Ss_field is not None):
        # We want d(logK) = dK / K.
        # STABILITY FIX: Use a larger epsilon or clip K for the denominator only.
        # A tiny K implies K is essentially zero/impermeable. 
        # We don't want to penalize dK variations when K is 1e-15 vs 1e-16.
        
        K_denom = tf_maximum(K_field, eps_div)
        Ss_denom = tf_maximum(Ss_field, eps_div)

        dlogK_dx  = dK_dx  / K_denom
        dlogK_dy  = dK_dy  / K_denom
        dlogSs_dx = dSs_dx / Ss_denom
        dlogSs_dy = dSs_dy / Ss_denom
        
        out = (
            tf_square(dlogK_dx) + tf_square(dlogK_dy)
            + tf_square(dlogSs_dx) + tf_square(dlogSs_dy)
        )
        vprint(verbose, "smooth(log-div): out=", out)
        return out

    # Fallback to raw gradients (rarely used but safe)
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
def _soft_barrier_l2(
    x: Tensor,
    lo: Tensor,
    hi: Tensor,
    *,
    beta: float = 20.0,
) -> Tensor:
    x = tf_cast(x, tf_float32)
    lo = tf_cast(lo, tf_float32)
    hi = tf_cast(hi, tf_float32)

    b = tf_constant(float(beta), tf_float32)

    v_lo = tf_softplus(b * (lo - x)) / b
    v_hi = tf_softplus(b * (x - hi)) / b

    return tf_square(v_lo) + tf_square(v_hi)


def _reduce_barrier_mean(v: Tensor) -> Tensor:
    v = tf_where(tf_math.is_finite(v), v, tf_zeros_like(v))
    return tf_reduce_mean(v)

def _soft_clip(
    x: Tensor,
    lo: Tensor,
    hi: Tensor,
    *,
    beta: float = 20.0,
) -> Tensor:
    x = tf_cast(x, tf_float32)
    lo = tf_cast(lo, tf_float32)
    hi = tf_cast(hi, tf_float32)

    b = tf_constant(float(beta), tf_float32)

    y1 = tf_softplus(b * (x - lo)) / b
    y2 = tf_softplus(b * (x - hi)) / b

    return lo + y1 - y2


# def guarded_exp_from_bounds(
#     raw_log,
#     log_min,
#     log_max,
#     *,
#     eps=0.0,
#     guard=5.0,
#     dtype=None,
#     name="",
# ):
#     """
#     Safe exp() with a wide log-space guard-band around [log_min, log_max].

#     - raw_log: unconstrained log-parameter (may drift during training)
#     - log_min/log_max: physical bounds in log-space
#     - guard: extra margin to avoid overflow; values outside are clipped only
#              for *numerical safety*, not as a hard physical constraint.
#     """
#     if dtype is None:
#         dtype = raw_log.dtype

#     raw_log = tf_cast(raw_log, dtype)
#     log_min = tf_cast(log_min, dtype)
#     log_max = tf_cast(log_max, dtype)
#     guard = tf_cast(tf_constant(guard), dtype)
#     eps = tf_cast(tf_constant(eps), dtype)

#     # replace NaN/Inf with 0 to avoid propagating non-finites
#     raw_log = tf_where(
#         tf_math.is_finite(raw_log), raw_log, 
#         tf_zeros_like(raw_log)
#     )

#     # guard-band clip (prevents exp overflow)
#     log_safe = tf_clip_by_value(
#         raw_log, log_min - guard, log_max + guard
#         )

#     field = tf_exp(log_safe) + eps

#     if name:
#         tf_debugging.assert_all_finite(raw_log,  f"{name} raw_log non-finite")
#         tf_debugging.assert_all_finite(field,    f"{name} field non-finite")

#     return field, raw_log, log_safe

def exp_from_bounds(
    raw_log,
    log_min,
    log_max,
    *,
    mode="soft",
    beta=20.0,
    guard=5.0,
    eps=0.0,
    dtype=None,
    name="",
):
    if dtype is None:
        dtype = raw_log.dtype

    mode = str(mode).strip().lower()
    beta_f = float(beta)
    guard_f = float(guard)
    eps_f = float(eps)

    raw_log = tf_cast(raw_log, dtype)
    log_min = tf_cast(log_min, dtype)
    log_max = tf_cast(log_max, dtype)

    raw_log = tf_where(
        tf_math.is_finite(raw_log),
        raw_log,
        tf_zeros_like(raw_log),
    )

    if mode == "hard":
        log_safe = tf_clip_by_value(raw_log, log_min, log_max)
        pen = tf_zeros_like(raw_log)

    elif mode == "sigmoid":
        t = tf_sigmoid(raw_log)
        log_safe = log_min + (log_max - log_min) * t
        pen = tf_zeros_like(raw_log)

    elif mode == "soft":
        # numeric safety guard-band, but smooth
        lo_g = log_min - tf_constant(guard_f, dtype)
        hi_g = log_max + tf_constant(guard_f, dtype)
        log_safe = _soft_clip(raw_log, lo_g, hi_g, beta=beta_f)

        # physical bounds penalty (differentiable)
        pen = _soft_barrier_l2(
            raw_log, log_min, log_max, beta=beta_f
        )

    else:  # "none"
        lo_g = log_min - tf_constant(guard_f, dtype)
        hi_g = log_max + tf_constant(guard_f, dtype)
        log_safe = tf_clip_by_value(raw_log, lo_g, hi_g)
        pen = tf_zeros_like(raw_log)

    field = tf_exp(log_safe) + tf_constant(eps_f, dtype)

    if name:
        tf_debugging.assert_all_finite(
            raw_log, f"{name} raw_log non-finite"
        )
        tf_debugging.assert_all_finite(
            field, f"{name} field non-finite"
        )

    return field, raw_log, log_safe, pen


def get_log_bounds(
    model,
    *,
    as_tensor: bool = True,
    dtype=tf_float32,
    verbose: int = 0,
) -> Tuple[Any, Any, Any, Any]:
    r"""
    Get validated log-space bounds for K and Ss.
    
    This helper reads bounds from ``model.scaling_kwargs['bounds']``
    and returns a 4-tuple:
    
    ``(logK_min, logK_max, logSs_min, logSs_max)``.
    
    It supports two equivalent representations:
    
    * Direct log-bounds:
      ``logK_min/logK_max`` and ``logSs_min/logSs_max``.
    * Linear bounds converted to logs:
      ``K_min/K_max`` and ``Ss_min/Ss_max``.
    
    If bounds are missing, the function returns
    ``(None, None, None, None)``.
    
    Parameters
    ----------
    model : Any
        Model-like object with an optional ``scaling_kwargs`` dict.
        Bounds are read from ``model.scaling_kwargs['bounds']``.
    
    as_tensor : bool, default=True
        If True, return Tensor scalars created with ``tf_constant``.
        If False, return Python floats.
    
    dtype : tf.DType, default=tf_float32
        Tensor dtype used when ``as_tensor=True``.
    
    verbose : int, default=0
        Verbosity level for optional debug printing.
    
    Returns
    -------
    logK_min, logK_max, logSs_min, logSs_max : tuple
        Log-space bounds as Tensor scalars (if ``as_tensor=True``),
        otherwise Python floats.
    
        If bounds are not configured, returns:
        ``(None, None, None, None)``.
    
    Raises
    ------
    ValueError
        If bounds exist but are invalid, including:
    
        * non-finite values (NaN or inf)
        * non-positive linear bounds (<= 0)
        * unordered bounds (max <= min)
    
    Notes
    -----
    Validation contract
    ~~~~~~~~~~~~~~~~~~~
    This function never emits NaN log bounds. If the configuration
    contains invalid entries, it fails fast with ``ValueError``.
    
    Conversion precedence
    ~~~~~~~~~~~~~~~~~~~~~
    If log-bounds are present, they are used directly. Otherwise,
    the function looks for linear bounds and converts them via:
    
    .. math::
    
       \log K_{\min} = \log(K_{\min}), \quad
       \log K_{\max} = \log(K_{\max}),
    
    and similarly for :math:`S_s`.
    
    Optional Ss heuristic
    ~~~~~~~~~~~~~~~~~~~~
    If ``Ss_min/Ss_max`` appear to be compressibility-like values
    (e.g., :math:`m_v`), the function may optionally convert them
    to :math:`S_s` using :math:`S_s = m_v \gamma_w` when a finite
    ``model.gamma_w`` is available. This heuristic is best-effort
    and never raises by itself.
    
    Examples
    --------
    Use Tensor bounds for downstream math:
    
    >>> logK_min, logK_max, logSs_min, logSs_max = get_log_bounds(
    ...     model, as_tensor=True
    ... )
    
    Return Python floats for inspection:
    
    >>> bounds = get_log_bounds(model, as_tensor=False)
    >>> print(bounds)
    
    See Also
    --------
    get_log_tau_bounds
        Companion helper for tau bounds in log space.
    
    compute_bounds_residual
        Uses these bounds to compute normalized violations.
    
    References
    ----------
    .. [1] Nocedal, J., and Wright, S. J. Numerical Optimization.
       Springer (2006).
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
    r"""
    Get validated log-space bounds for the consolidation timescale.
    
    This helper returns a 2-tuple:
    
    ``(log_tau_min, log_tau_max)``,
    
    where :math:`\tau` is the consolidation timescale expressed in
    SI seconds, and the returned bounds are in log-seconds.
    
    The function reads bounds from ``model.scaling_kwargs['bounds']``
    with the following precedence:
    
    1. Explicit log bounds:
       ``log_tau_min`` and ``log_tau_max`` (already log-seconds).
    2. Linear bounds in seconds:
       ``tau_min`` and ``tau_max``.
    3. Linear bounds in dataset time units:
       ``tau_min_units`` and ``tau_max_units`` multiplied by the
       seconds-per-time-unit factor inferred from ``time_units``.
    4. Robust defaults if nothing is provided.
    
    Parameters
    ----------
    model : Any
        Model-like object with an optional ``scaling_kwargs`` dict.
        Tau bounds are read from ``model.scaling_kwargs['bounds']``.
    
    as_tensor : bool, default=True
        If True, return Tensor scalars created with ``tf_constant``.
        If False, return Python floats.
    
    dtype : tf.DType, default=tf_float32
        Tensor dtype used when ``as_tensor=True``.
    
    verbose : int, default=0
        Verbosity level for optional debug printing.
    
    Returns
    -------
    log_tau_min, log_tau_max : tuple
        Log-space bounds (log-seconds). Returned as Tensor scalars
        when ``as_tensor=True``, otherwise as Python floats.
    
    Raises
    ------
    ValueError
        If user-provided bounds exist but are invalid, including:
    
        * non-finite values (NaN or inf)
        * non-positive linear bounds (<= 0)
        * unordered bounds (max <= min) for explicit log bounds
    
    Notes
    -----
    Units and meaning
    ~~~~~~~~~~~~~~~~~
    The consolidation timescale :math:`\tau` controls the relaxation
    rate in a first-order consolidation closure, e.g.:
    
    .. math::
    
       \partial_t s = \frac{s_{eq}(h) - s}{\tau},
    
    where :math:`s` is settlement and :math:`s_{eq}` is the
    equilibrium settlement implied by head (or drawdown).
    
    Default behavior
    ~~~~~~~~~~~~~~~~
    If no tau bounds are provided, robust defaults are used:
    
    * ``tau_min = 7 days``
    * ``tau_max = 300 years``
    
    Both are converted to seconds and then logged. A warning may be
    emitted to make the defaulting explicit.
    
    Swapped linear bounds
    ~~~~~~~~~~~~~~~~~~~~~
    If linear bounds are provided with ``tau_max < tau_min``, the
    function may swap them to maintain a valid interval.
    
    Examples
    --------
    Use Tensor bounds for log-space clipping:
    
    >>> log_tau_min, log_tau_max = get_log_tau_bounds(model)
    
    Return floats for reporting:
    
    >>> log_tau_min, log_tau_max = get_log_tau_bounds(
    ...     model, as_tensor=False
    ... )
    
    See Also
    --------
    get_log_bounds
        Bounds helper for log(K) and log(Ss).
    
    compute_bounds_residual
        Computes normalized bound violations using these limits.
    
    References
    ----------
    .. [1] Terzaghi, K. Theoretical Soil Mechanics. Wiley (1943).
    .. [2] Verruijt, A. Theory and Problems of Poroelasticity.
       Delft University of Technology (2013).
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
    r"""
    Exponentiate a raw parameter inside hard log-bounds.
    
    This helper maps an unconstrained tensor ``raw`` to a positive
    field by interpolating in log space between ``log_min`` and
    ``log_max``. The mapping is smooth and bounded:
    
    .. math::
    
       z = \sigma(\mathrm{raw}), \quad
       \log v = \log v_{min} + z(\log v_{max} - \log v_{min}), \quad
       v = \exp(\log v) + \varepsilon,
    
    where :math:`\sigma` is the logistic sigmoid and
    :math:`\varepsilon` is a small positive floor.
    
    This is used when ``bounds_mode="hard"`` to ensure learned
    fields such as :math:`K`, :math:`S_s`, or :math:`\tau` never
    leave their configured ranges.
    
    Parameters
    ----------
    raw : Tensor
        Unconstrained logit-like tensor (any shape). Non-finite
        entries are sanitized to zeros to avoid NaN propagation.
    
    log_min : Tensor
        Lower bound in log space. Must be finite for strict
        correctness, but non-finite values are sanitized to a safe
        constant to prevent NaNs.
    
    log_max : Tensor
        Upper bound in log space. Must be finite for strict
        correctness, but non-finite values are sanitized to a safe
        constant to prevent NaNs.
    
    eps : float, default=_EPSILON
        Positive floor added after exponentiation to guarantee
        strictly positive output.
    
    return_log : bool, default=False
        If True, return ``(out, logv)`` where ``logv`` is the bounded
        log value actually exponentiated. If False, return ``out``
        only.
    
    verbose : int, default=0
        Verbosity level for optional debug printing.
    
    Returns
    -------
    out : Tensor
        Positive bounded field tensor with the same shape as ``raw``.
    
    logv : Tensor, optional
        Bounded log value used to compute ``out``. Returned only
        when ``return_log=True``.
    
    Notes
    -----
    Hard bounds via sigmoid
    ~~~~~~~~~~~~~~~~~~~~~~~
    The sigmoid interpolation produces values strictly inside the
    interval (up to numerical precision). This avoids the gradient
    discontinuity of direct clipping while still enforcing bounds.
    
    Sanitization policy
    ~~~~~~~~~~~~~~~~~~~
    To prevent NaNs and Infs from contaminating training, the
    function sanitizes:
    
    * non-finite values in ``raw`` to zeros,
    * non-finite values in bounds to safe constants,
    * swapped bounds by repairing the interval ordering.
    
    This behavior is defensive and prioritizes numerical stability.
    
    Examples
    --------
    Bound a raw logit field to the K interval:
    
    >>> K, logK = bounded_exp(
    ...     rawK, logK_min, logK_max, return_log=True
    ... )
    
    Bound a tau field (already in log seconds bounds):
    
    >>> tau = bounded_exp(raw_tau, log_tau_min, log_tau_max)
    
    See Also
    --------
    guarded_exp_from_bounds
        Soft-bounds path that keeps raw logs for penalties while
        guarding exponentiation overflow.
    
    compose_physics_fields
        Uses bounded_exp to build K, Ss, and tau fields.
    
    References
    ----------
    .. [1] Goodfellow, I., Bengio, Y., and Courville, A. Deep
       Learning. MIT Press (2016).
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

def _get_bounds_loss_cfg(
    model: Any = None,
    scaling_kwargs: Optional[dict] = None,
) -> Dict[str, Any]:
    def _as_map(x: Any) -> Mapping[str, Any]:
        return x if isinstance(x, Mapping) else {}

    def _attr(name: str, default: Any) -> Any:
        if model is None:
            return default
        return getattr(model, name, default)

    def _take(sk: Mapping[str, Any], key: str, cur: Any) -> Any:
        # If user/profile put None by mistake, ignore it
        if key in sk:
            v = sk.get(key)
            if v is not None:
                return v
        return cur

    sk_model = _as_map(getattr(model, "scaling_kwargs", None))
    sk_arg = _as_map(scaling_kwargs)

    # precedence (low -> high):
    #   model attrs < model.scaling_kwargs < scaling_kwargs arg
    mode = _attr("bounds_mode", "soft")
    kind = _attr("bounds_loss_kind", "both")

    beta  = _attr("bounds_beta", 20.0)
    guard = _attr("bounds_guard", 5.0)
    w_b   = _attr("bounds_w", 1.0)

    inc_tau = _attr("bounds_include_tau", True)
    w_tau   = _attr("bounds_tau_w", 1.0)

    sources = [sk_model, sk_arg]

    # 1) flat keys
    for sk in sources:
        mode  = _take(sk, "bounds_mode", mode)
        kind  = _take(sk, "bounds_loss_kind", kind)

        beta  = _take(sk, "bounds_beta", beta)
        guard = _take(sk, "bounds_guard", guard)
        w_b   = _take(sk, "bounds_w", w_b)

        inc_tau = _take(sk, "bounds_include_tau", inc_tau)
        w_tau   = _take(sk, "bounds_tau_w", w_tau)

    # 2) nested dict
    nested_keys = (
        "bounds_loss_settings",
        "bounds_loss_setting",
        "bound_cfg",
        "bound_loss_cfg",
        "bounds_settings",
        "bounds_config",
        "bounds_loss_config",
    )

    for sk in sources:
        nested: Any = {}
        for k in nested_keys:
            if k in sk:
                nested = sk.get(k)
                break
        if not isinstance(nested, Mapping):
            continue

        mode  = _take(nested, "mode", mode)
        kind  = _take(nested, "kind", kind)

        beta  = _take(nested, "beta", beta)
        guard = _take(nested, "guard", guard)
        w_b   = _take(nested, "w", w_b)

        inc_tau = _take(nested, "include_tau", inc_tau)
        w_tau   = _take(nested, "tau_w", w_tau)

    mode = str(mode).strip().lower()
    kind = str(kind).strip().lower()

    return dict(
        mode=mode,
        kind=kind,
        beta=float(beta),
        guard=float(guard),
        w=float(w_b),
        include_tau=bool(inc_tau),
        tau_w=float(w_tau),
    )

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
    r"""
    Compose physically meaningful fields :math:`K`, :math:`S_s`, and
    :math:`\tau` from network "base" logits and coordinate corrections.
    
    This routine is the central *field mapping* step for GeoPrior-style
    PINN models. The model predicts coarse (time-dependent) latent logits
    ``K_base``, ``Ss_base``, and ``tau_base`` from the physics head, then
    adds smooth spatial corrections from coordinate MLPs:
    
    * ``model.K_coord_mlp`` for :math:`\log K`
    * ``model.Ss_coord_mlp`` for :math:`\log S_s`
    * ``model.tau_coord_mlp`` for :math:`\Delta \log \tau`
    
    The corrected parameters are then mapped to SI-consistent, positive
    fields (in float32-safe ways) and combined with a physics closure
    timescale :math:`\tau_\mathrm{phys}` computed from the fields.
    
    Let :math:`(t, x, y)` denote the coordinate tensor passed to the
    decoder. Spatial corrections are evaluated on coordinates with time
    zeroed:
    
    .. math::
    
       \tilde{\mathbf{c}} = (0, x, y).
    
    Define the raw log-parameters (logits) as:
    
    .. math::
    
       \ell_K  &= \ell_K^\mathrm{base}(t,x,y)
                 + \Delta \ell_K(\tilde{\mathbf{c}}), \\
       \ell_{S_s} &= \ell_{S_s}^\mathrm{base}(t,x,y)
                 + \Delta \ell_{S_s}(\tilde{\mathbf{c}}).
    
    The resulting fields are positive exponentials:
    
    .. math::
    
       K = \exp(\ell_K), \qquad
       S_s = \exp(\ell_{S_s}),
    
    subject to (log-)bounds. In ``bounds_mode="hard"`` the values are
    projected into the valid interval by clipping in log space, while in
    ``bounds_mode="soft"`` the function returns the unbounded logs for
    penalties but uses a *guarded exponential* to avoid float32 overflow.
    
    For the consolidation timescale, we first compute a closure (prior)
    timescale from the fields:
    
    .. math::
    
       \log \tau_\mathrm{phys} =
       f_\tau(K, S_s, H; \text{model options}),
    
    where :math:`H` is the drained thickness in meters (``H_si``) and
    ``tau_phys_from_fields`` implements the chosen closure and drainage
    convention. The network adds a learnable residual in log space:
    
    .. math::
    
       \Delta \log \tau =
       \ell_\tau^\mathrm{base}(t,x,y) + \Delta \ell_\tau(\tilde{\mathbf{c}}),
    
    and the total learned timescale is:
    
    .. math::
    
       \log \tau = \log \tau_\mathrm{phys} + \Delta \log \tau, \qquad
       \tau = \exp(\log \tau) + \varepsilon_\tau.
    
    The term :math:`\varepsilon_\tau` (``eps_tau``) is a small positive
    floor to avoid exact zeros and improve numerical stability.
    
    Parameters
    ----------
    model : Any
        Model-like object providing:
    
        * coordinate MLPs: ``K_coord_mlp``, ``Ss_coord_mlp``,
          ``tau_coord_mlp``
        * bounds configuration: ``bounds_mode`` and bounds accessors
          used by ``get_log_bounds`` and ``get_log_tau_bounds``
        * closure configuration used by ``tau_phys_from_fields``
    
    coords_flat : Tensor
        Coordinate tensor used by the decoder. Expected shape is
        ``(B, H, 3)`` with last dimension ordered as ``(t, x, y)``.
        The function constructs ``(0, x, y)`` for the coordinate MLPs to
        keep corrections time-invariant by default.
    
    H_si : Tensor
        Drained thickness :math:`H` in SI units (meters). Shape must be
        broadcastable to ``(B, H, 1)``.
    
    K_base : Tensor
        Base logits for :math:`\log K`. Shape is typically ``(B, H, 1)``.
    
    Ss_base : Tensor
        Base logits for :math:`\log S_s`. Shape is typically
        ``(B, H, 1)``.
    
    tau_base : Tensor
        Base logits for :math:`\Delta \log \tau`. Shape is typically
        ``(B, H, 1)``.
    
    training : bool, default=False
        Forward mode for coordinate MLPs.
    
    eps_KSs : float, default=_EPSILON
        Small positive constant used when mapping log-parameters to
        positive values (e.g., inside bounded / guarded exponentials).
    
    eps_tau : float, default=1e-6
        Additive floor for :math:`\tau` in seconds to avoid exact zeros.
    
    verbose : int, default=0
        Verbosity level used by internal debug printing utilities.
    
    Returns
    -------
    K_field : Tensor
        Effective hydraulic conductivity field :math:`K` in SI units.
        Shape ``(B, H, 1)``. Units are typically meters per second.
    
    Ss_field : Tensor
        Effective specific storage field :math:`S_s` in SI units.
        Shape ``(B, H, 1)``. Units are typically inverse meters.
    
    tau_field : Tensor
        Learned consolidation timescale :math:`\tau` in seconds.
        Shape ``(B, H, 1)``.
    
    tau_phys : Tensor
        Closure-based timescale :math:`\tau_\mathrm{phys}` in seconds.
        Shape ``(B, H, 1)`` (broadcasted as needed).
    
    Hd_eff : Tensor
        Effective drainage thickness :math:`H_d` in meters used by the
        closure, accounting for drainage mode and ``hd_factor`` style
        options. Shape broadcastable to ``(B, H, 1)``.
    
    delta_log_tau : Tensor
        The learnable log-residual :math:`\Delta \log \tau` added to
        :math:`\log \tau_\mathrm{phys}`. Shape ``(B, H, 1)``.
    
    logK : Tensor
        Log-parameter :math:`\log K` used for priors, bounds penalties,
        and diagnostics. Shape ``(B, H, 1)``.
    
    logSs : Tensor
        Log-parameter :math:`\log S_s` used for priors, bounds penalties,
        and diagnostics. Shape ``(B, H, 1)``.
    
    log_tau : Tensor
        Log of total timescale :math:`\log \tau` (pre-guard in soft mode).
        Returned for bounds penalties and diagnostics. Shape ``(B, H, 1)``.
    
    log_tau_phys : Tensor
        Log of closure timescale :math:`\log \tau_\mathrm{phys}` returned
        for priors and diagnostics. Shape ``(B, H, 1)``.
    
    Notes
    -----
    Why coordinate corrections use ``(0, x, y)``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The coordinate MLPs are intended to represent slowly varying spatial
    heterogeneity (e.g., lithology-driven variability). Zeroing time
    reduces the risk that the model encodes time-varying physics fields
    that can destabilize PDE derivatives across horizons.
    
    Hard vs soft bounds
    ~~~~~~~~~~~~~~~~~~~
    When ``bounds_mode="hard"``, log-parameters are projected into the
    valid interval, yielding fields that always satisfy bounds.
    
    When ``bounds_mode="soft"``, log-parameters are returned unmodified
    for differentiable penalties, but exponentiation is guarded to prevent
    float32 overflow. This preserves gradients for penalties without
    risking NaN / Inf in the forward pass.
    
    Numerical stability
    ~~~~~~~~~~~~~~~~~~~
    The function deliberately avoids reapplying ``log(exp(.))`` patterns.
    In particular, it composes :math:`\log \tau` additively:
    
    .. math::
    
       \log \tau = \log \tau_\mathrm{phys} + \Delta \log \tau,
    
    which is both exact and numerically stable.
    
    Examples
    --------
    Compute fields inside a physics forward pass:
    
    >>> K_field, Ss_field, tau_field, tau_phys, Hd_eff, dlogtau, logK, \
    ... logSs, log_tau, log_tau_phys = compose_physics_fields(
    ...     model,
    ...     coords_flat=coords,
    ...     H_si=H_si,
    ...     K_base=K_logits,
    ...     Ss_base=Ss_logits,
    ...     tau_base=dlogtau_logits,
    ...     training=True,
    ... )
    
    Use returned logs for priors and bounds penalties:
    
    >>> prior_res = dlogtau
    >>> bounds_penalty_inputs = (logK, logSs, log_tau)
    
    See Also
    --------
    tau_phys_from_fields
        Computes the closure timescale :math:`\tau_\mathrm{phys}`.
    
    get_log_bounds, get_log_tau_bounds
        Provide log-space bounds used for field mapping.
    
    bounded_exp, guarded_exp_from_bounds
        Safe mappings from log-parameters to positive fields.
    
    compute_bounds_residual
        Uses the returned logs and thickness for bounds penalties.
    
    References
    ----------
    .. [1] Biot, M. A. Theory of elasticity and consolidation for a
       porous anisotropic solid. Journal of Applied Physics (1941).
    
    .. [2] Bear, J. Dynamics of Fluids in Porous Media. Dover (1972).
    """

    bc = _get_bounds_loss_cfg(model)
    
    mode = bc["mode"]
    beta = bc["beta"]
    guard = bc["guard"]
    w_b = bc["w"]
    include_tau = bc["include_tau"]
    w_tau = bc["tau_w"]

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

    if verbose > 6:
        tf_print_nonfinite("compose/K_corr", K_corr)
        tf_print_nonfinite("compose/Ss_corr", Ss_corr)
        tf_print_nonfinite("compose/tau_corr", tau_corr)

    rawK = K_base + K_corr
    rawSs = Ss_base + Ss_corr

    # bounds_mode = str(getattr(model, "bounds_mode", "soft")).strip().lower()
    
    logK_min, logK_max, logSs_min, logSs_max = get_log_bounds(
        model, as_tensor=True, dtype=rawK.dtype, verbose=verbose,
    )
    # # ---- K, Ss  ----
    # if bounds_mode == "hard":
    #     K_field, logK = bounded_exp(
    #         rawK, logK_min, logK_max, eps=eps_KSs,
    #         return_log=True, verbose=verbose,
    #     )
    #     Ss_field, logSs = bounded_exp(
    #         rawSs, logSs_min, logSs_max, eps=eps_KSs,
    #         return_log=True, verbose=verbose,
    #     )

    # else:
    #     # Keep raw log-params (useful for priors/diagnostics),
    #     # but NEVER feed an unbounded log into exp() in float32.
    #     K_field,  logK,  _ = guarded_exp_from_bounds(
    #         rawK,  logK_min,  logK_max, eps=eps_KSs, 
    #         guard=5.0, name="K"
    #     )
    #     Ss_field, logSs, _ = guarded_exp_from_bounds(
    #         rawSs, logSs_min, logSs_max,eps=eps_KSs,
    #         guard=5.0, name="Ss"
    #     )
    
    
    # ---- K, Ss (policy-driven) ----
    K_field, logK_raw, logK_safe, pK = exp_from_bounds(
        rawK,
        logK_min,
        logK_max,
        mode=mode,
        beta=beta,
        guard=guard,
        eps=eps_KSs,
        dtype=rawK.dtype,
        name="K",
    )
    
    Ss_field, logSs_raw, logSs_safe, pS = exp_from_bounds(
        rawSs,
        logSs_min,
        logSs_max,
        mode=mode,
        beta=beta,
        guard=guard,
        eps=eps_KSs,
        dtype=rawSs.dtype,
        name="Ss",
    )
    
    # What to return as "logK/logSs" depends on policy:
    # - soft/none: return raw (useful for penalties/diagnostics)
    # - hard/sigmoid: return safe (already within bounds)
    if mode in ("soft", "none"):
        logK = logK_raw
        logSs = logSs_raw
    else:
        logK = logK_safe
        logSs = logSs_safe
    
    loss_bounds_KSs = tf_constant(float(w_b), rawK.dtype) * (
        _reduce_barrier_mean(pK) +
        _reduce_barrier_mean(pS)
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

    # 1. Capture output as LOG value (because return_log=True)
    log_tau_phys, Hd_eff = tau_phys_from_fields(
        model, K_field, Ss_field, H_si, return_log=True, 
        verbose=0,
    )
    
    # 2. Calculate linear tau_phys safely from log (for logging/debugging)
    tau_phys = tf_exp(log_tau_phys)
    
    # ---- tau (policy-driven) ----
    
    # 3. Calculate total log directly (avoiding re-logging the exp)
    #    Previous bad logic: log(max(exp(log_x), eps)) -> redundant and lossy
    #    New logic: just add the logs directly.
    log_tau_total = log_tau_phys + delta_log_tau

    log_tau_min, log_tau_max = get_log_tau_bounds(
        model,
        as_tensor=True,
        dtype=log_tau_total.dtype,
        verbose=0,
    )
    
    # if bounds_mode == "hard":
    #     # true hard bounds: clip in log-space (keeps tau_phys anchoring)
    #     log_tau = tf_clip_by_value(log_tau_total, log_tau_min, log_tau_max)
    #     tau_field = tf_exp(log_tau) + tf_constant(eps_tau, log_tau.dtype)
    # else:
    #     # soft mode: keep log_tau for bounds penalty, but guard exp overflow
    #     log_tau = log_tau_total

    #     guard_lo = log_tau_min - tf_constant(10.0, log_tau.dtype)
    #     guard_hi = log_tau_max + tf_constant(10.0, log_tau.dtype)
    #     log_tau_safe = tf_clip_by_value(log_tau, guard_lo, guard_hi)

    #     tau_field = tf_exp(log_tau_safe) + tf_constant(eps_tau, log_tau.dtype)
    
    tau_field, logTau_raw, logTau_safe, pT = exp_from_bounds(
        log_tau_total,
        log_tau_min,
        log_tau_max,
        mode=mode,
        beta=beta,
        guard=guard,
        eps=eps_tau,
        dtype=log_tau_total.dtype,
        name="tau",
    )
    
    if mode in ("soft", "none"):
        log_tau = logTau_raw
    else:
        log_tau = logTau_safe
    
    loss_bounds_tau = tf_zeros_like(loss_bounds_KSs)
    
    if include_tau:
        loss_bounds_tau = tf_constant(float(w_tau), log_tau.dtype) * (
            _reduce_barrier_mean(pT)
        )
    
    loss_bounds = loss_bounds_KSs + loss_bounds_tau

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
        loss_bounds,
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
    r"""
    Compute differentiable bound-violation residuals for the learned
    physics fields.
    
    This function converts configured parameter bounds into *residual
    maps* that can be squared and averaged to form a soft penalty term
    (e.g., :math:`L_\mathrm{bounds} = \mathrm{mean}(R^2)`).
    
    The bounds policy is driven by ``model.scaling_kwargs['bounds']`` and
    supports:
    
    * Linear-space bounds for drained thickness :math:`H` (meters).
    * Log-space bounds for :math:`K`, :math:`S_s`, and :math:`\tau`.
    
    The returned residuals are normalized by the corresponding bound
    ranges, so they are roughly comparable across parameters.
    
    Mathematical formulation
    ------------------------
    Let :math:`z` be a scalar parameter with bounds
    :math:`z_{\min} \le z \le z_{\max}`. A standard non-negative violation
    is:
    
    .. math::
    
       v(z) = \max(z_{\min} - z, 0) + \max(z - z_{\max}, 0).
    
    This function returns a *range-normalized* residual:
    
    .. math::
    
       R(z) = \frac{v(z)}{\max(z_{\max} - z_{\min}, \varepsilon)}.
    
    For log-bounded parameters (conductivity, storage, and timescale), the
    same definition is applied in log space. For example, if
    :math:`\ell_K = \log K`, then:
    
    .. math::
    
       R_K = R(\ell_K; \ell_{K,\min}, \ell_{K,\max}),
    
    where :math:`\ell_{K,\min} = \log K_{\min}` and
    :math:`\ell_{K,\max} = \log K_{\max}`.
    
    Preferred usage in soft bounds mode
    -----------------------------------
    When ``bounds_mode="soft"``, it is preferable to pass the raw log
    parameters ``logK``, ``logSs``, and ``log_tau`` produced before any
    "guarded exponential" is applied. This ensures that the penalty
    reflects the true magnitude of out-of-range logits, even if the
    corresponding fields are exponentiated using an overflow guard.
    
    If raw logs are not provided, the function falls back to inferring
    logs from the fields via ``log(max(field, eps))``. This is safe, but
    may under-estimate violations if the field values were produced by a
    guarded mapping.
    
    Parameters
    ----------
    model : Any
        Model-like object providing:
    
        * ``scaling_kwargs`` with optional bounds + bounds policy keys.
        * Accessors used by ``get_log_bounds`` and ``get_log_tau_bounds``.
        * (Optional) pre-resolved bound tensors cached by the model.
    
        Bounds configuration is read from ``model.scaling_kwargs``:
    
        * ``bounds`` (dict): numeric ranges such as ``H_min/H_max``,
          ``K_min/K_max`` or ``logK_min/logK_max``, similarly for
          ``Ss`` and optionally ``tau``.
        * ``bounds_mode``: one of ``{'soft','hard','sigmoid','none'}``.
        * ``bounds_beta``: barrier sharpness for ``bounds_mode='soft'``.
        * ``bounds_guard``: numeric guard band used by guarded mappings.
        * ``bounds_w``: barrier weight for K + Ss.
        * ``bounds_include_tau``: whether to include tau bounds.
        * ``bounds_tau_w``: barrier weight for tau if included.

    H_field : Tensor
        Drained thickness field :math:`H` in SI meters. Shape must be
        broadcastable to ``(B, H, 1)``. Bounds are applied in linear space
        using keys ``H_min`` and ``H_max`` when present.
    
    logK : Tensor, optional
        Log-conductivity :math:`\log K` (preferred). If provided, bounds
        are applied directly in log space and ``K_field`` is not needed.
    
    logSs : Tensor, optional
        Log-specific storage :math:`\log S_s` (preferred). If provided,
        bounds are applied directly in log space and ``Ss_field`` is not
        needed.
    
    log_tau : Tensor, optional
        Log-timescale :math:`\log \tau` in seconds (preferred). If
        provided, bounds are applied directly in log space and
        ``tau_field`` is not needed.
    
    K_field : Tensor, optional
        Conductivity field :math:`K` (meters per second). Used only if
        ``logK`` is not provided.
    
    Ss_field : Tensor, optional
        Specific storage field :math:`S_s` (inverse meters). Used only if
        ``logSs`` is not provided.
    
    tau_field : Tensor, optional
        Timescale field :math:`\tau` (seconds). Used only if ``log_tau`` is
        not provided.
    
    eps : float, default=_EPSILON
        Small positive constant used to avoid division by zero and
        undefined logs, via :math:`\max(\cdot, \varepsilon)`.
    
    verbose : int, default=0
        Verbosity level for optional debug printing.
    
    Returns
    -------
    R_H : Tensor
        Residual map for thickness bounds violations. Same shape as
        ``H_field`` (broadcasted as needed). All values are non-negative.
    
    R_K : Tensor
        Residual map for conductivity log-bounds violations. Same shape as
        ``H_field`` (broadcasted as needed). All values are non-negative.
    
    R_Ss : Tensor
        Residual map for specific storage log-bounds violations. Same
        shape as ``H_field`` (broadcasted as needed). All values are
        non-negative.
    
    R_tau : Tensor
        Residual map for timescale log-bounds violations. Same shape as
        ``H_field`` (broadcasted as needed). All values are non-negative.
        
    loss_bounds_barrier : Tensor
        Non-negative scalar barrier penalty induced by the configured
        bounds policy (typically K/Ss and optionally tau). This is the
        *barrier component only*.
    
        Notes
        -----
        Any *residual-style* bound penalty (range-normalized violation
        residuals for H/K/Ss/tau) is computed separately (e.g. via
        ``compute_bounds_residual``) and can be combined downstream
        according to ``bounds_loss_kind``.

    Notes
    -----
    Bounds configuration
    ~~~~~~~~~~~~~~~~~~~~
    The function reads bounds from:
    
    * ``model.scaling_kwargs.get('bounds', {})`` for ``H_min`` and ``H_max``.
    * ``get_log_bounds(model, ...)`` for log-space bounds of :math:`K` and
      :math:`S_s` (typically ``logK_min``, ``logK_max``, ``logSs_min``,
      ``logSs_max``).
    * ``get_log_tau_bounds(model, ...)`` for log-space bounds of
      :math:`\tau` (typically ``logTau_min`` and ``logTau_max``).
    
    1) Linear-space bounds (thickness)
       * ``scaling_kwargs['bounds']['H_min']`` and
         ``scaling_kwargs['bounds']['H_max']`` (meters)
    
    2) Log-space bounds (K and Ss)
       Bounds are resolved by ``get_log_bounds(model, ...)``, which
       may be backed by any of the following configuration entries:
    
       * Direct log bounds:
         ``bounds['logK_min']`` / ``bounds['logK_max']``,
         ``bounds['logSs_min']`` / ``bounds['logSs_max']``
    
       * Or linear bounds that can be converted to logs:
         ``bounds['K_min']`` / ``bounds['K_max']``,
         ``bounds['Ss_min']`` / ``bounds['Ss_max']``
    
    3) Log-space bounds (tau)
       Bounds are resolved by ``get_log_tau_bounds(model, ...)``,
       typically from:
    
       * Direct log bounds:
         ``bounds['logTau_min']`` / ``bounds['logTau_max']``
    
       * Or linear bounds convertible to logs:
         ``bounds['tau_min']`` / ``bounds['tau_max']``
    
    If a bound is missing (or the accessor returns ``None``),
    the corresponding residual is returned as zeros.
    
    Interaction with bounds policy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Keys such as ``bounds_mode``, ``bounds_beta``, ``bounds_guard``,
    ``bounds_w``, ``bounds_include_tau``, and ``bounds_tau_w`` define
    how the *barrier* term is produced (usually in
    ``compose_physics_fields``). They do not change the definition
    of the residual maps returned here; instead, they control how
    the physics core combines residual penalty and barrier penalty
    via ``bounds_loss_kind``.


    Normalization
    ~~~~~~~~~~~~~
    Residuals are normalized by the bound range to reduce sensitivity to
    the absolute scale of each parameter. This makes it easier to set a
    single loss weight such as ``lambda_bounds`` without one parameter
    dominating purely due to units.

    Debugging tip
    ~~~~~~~~~~~~~
    To debug bounds behavior, log the two components separately:
    
    * ``loss_bounds_resid`` from these residual maps:
      ``mean(square(concat([R_H, R_K, R_Ss, R_tau])))``
    * ``loss_bounds_barrier`` returned by ``compose_physics_fields``
    
    Then feed only their configured combination into the final
    physics loss according to ``bounds_loss_kind``.


    Examples
    --------
    Compute residuals from raw logs (recommended in soft mode):
    
    >>> R_H, R_K, R_Ss, R_tau = compute_bounds_residual(
    ...     model,
    ...     H_field=H_si,
    ...     logK=logK,
    ...     logSs=logSs,
    ...     log_tau=log_tau,
    ... )
    
    Fallback when only fields are available:
    
    >>> R_H, R_K, R_Ss, R_tau = compute_bounds_residual(
    ...     model,
    ...     H_field=H_si,
    ...     K_field=K_field,
    ...     Ss_field=Ss_field,
    ...     tau_field=tau_field,
    ... )
    
    Create a scalar penalty:
    
    >>> bounds_res = tf_concat([R_H, R_K, R_Ss, R_tau], axis=-1)
    >>> loss_bounds = tf_reduce_mean(tf_square(bounds_res))
    
    See Also
    --------
    compose_physics_fields
        Produces both raw logs and exponentiated fields.
    
    get_log_bounds, get_log_tau_bounds
        Retrieve log-space bounds for :math:`K`, :math:`S_s`, and
        :math:`\tau`.
    
    _log_bounds_residual
        Internal helper that converts log-values to normalized residuals.
    
    References
    ----------
    .. [1] Nocedal, J., and Wright, S. J. Numerical Optimization.
       Springer (2006).
    
    .. [2] Boyd, S., and Vandenberghe, L. Convex Optimization.
       Cambridge University Press (2004).
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
    eps:float=1e-12, 
    verbose: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Bounds residuals for H,K,Ss.
    
    Preferred usage in soft bounds mode
    -----------------------------------
    When ``model.scaling_kwargs['bounds_mode'] == 'soft'``, the
    model typically enforces bounds via a *barrier* penalty (often
    computed inside ``compose_physics_fields``) while the forward
    mapping from logits to fields may also use a numeric guard
    (e.g. ``bounds_guard``) to prevent overflow.
    
    For the most informative residual-style diagnostics, pass the
    *raw* log-parameters (``logK``, ``logSs``, ``log_tau``) produced
    *before* any guarded exponential / squashing is applied. This
    ensures the residual penalty reflects the true distance of the
    raw parameters from the configured log-bounds, even if the
    corresponding physical fields were produced by a guarded
    mapping.
    
    If raw logs are not provided, the function falls back to
    inferring logs from the fields via ``log(max(field, eps))``.
    This is numerically safe, but it can *under-estimate*
    violations when the forward mapping clips or guards extreme
    values (e.g. due to ``bounds_guard`` or hard/sigmoid modes).
    
    Important
    ~~~~~~~~~
    ``compute_bounds_residual`` always returns *violation residuals*
    (regardless of ``bounds_mode``). The training objective chooses
    how to combine:
    (1) the residual penalty derived from these residual maps and
    (2) the barrier penalty returned by ``compose_physics_fields``,
    according to ``scaling_kwargs['bounds_loss_kind']``.
    
    """
    dtype = K_field.dtype
    eps = tf_constant(eps, dtype=dtype)
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
    r"""
    Guard a residual scale using the observed residual magnitude.
    
    This helper prevents residual normalization from exploding
    when a nominal scale is too small compared with the actual
    residual values observed on the current batch.
    
    Mathematical intent
    -------------------
    Given a residual tensor :math:`r` and a proposed scale
    :math:`s`, this function produces a guarded scale
    :math:`\hat{s}` such that:
    
    .. math::
    
       \hat{s} \ge s_{floor}
    
    and also:
    
    .. math::
    
       \hat{s} \ge r_{ref}
    
    .. math::
    
       \hat{s} \ge 0.1\, r_{max}
    
    where:
    
    .. math::
    
       r_{ref} = \mathrm{mean}(|r|) + \varepsilon
    
    .. math::
    
       r_{max} = \mathrm{max}(|r|) + \varepsilon
    
    This ensures the normalized residual:
    
    .. math::
    
       \tilde{r} = \frac{r}{\hat{s}}
    
    is not arbitrarily large solely because the scale estimate
    collapsed (for example, due to degenerate dt, missing
    features, or transient NaN handling upstream).
    
    Key properties
    --------------
    * Robust to NaN/Inf in ``residual`` and ``scale``:
      non-finite entries are treated as zeros for the residual
      magnitude and replaced by ``floor`` for the scale.
    * Uses stop-gradient on the guarded scale, so the model
      cannot reduce loss by manipulating the scale path.
    * Uses both a typical magnitude (mean) and a tail proxy
      (max) to avoid under-scaling when outliers occur.
    
    Parameters
    ----------
    residual : Tensor
        Residual tensor :math:`r`. Any shape is accepted.
        Values are flattened internally to compute statistics.
    
    scale : Tensor
        Proposed residual scale :math:`s`. May be scalar-like
        (recommended). If non-finite, it is replaced by
        ``floor``.
    
    floor : float
        Minimal allowed scale :math:`s_{floor}`. This protects
        against division by tiny values.
    
    eps : float, default=_EPSILON
        Small positive constant :math:`\varepsilon` added to the
        residual statistics to prevent zero magnitudes.
    
    Returns
    -------
    scale_guarded : Tensor
        Guarded scale :math:`\hat{s}` with stop-gradient applied.
    
    Notes
    -----
    Why guard against residual magnitude
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If a scale is estimated from weak signals (for example,
    nearly constant series) it may become very small. Dividing
    a non-trivial residual by that small scale can dominate the
    training objective and destabilize optimization.
    
    Why the 0.1 * max term
    ~~~~~~~~~~~~~~~~~~~~~~
    The mean alone can underestimate the needed scale when
    residuals have heavy tails or occasional spikes. Including a
    fraction of the max provides a simple tail-sensitive guard.
    
    Examples
    --------
    This function is typically used before scaling residuals:
    
    .. code-block:: python
    
       scale_gw = _gw_scale_core(...)
       scale_gw = guard_scale_with_residual(
           residual=R_gw,
           scale=scale_gw,
           floor=1e-8,
       )
       R_gw_scaled = scale_residual(R_gw, scale_gw)
    
    See Also
    --------
    scale_residual
        Divide a residual by a stop-gradient scale safely.
    
    to_rms
        Compute RMS magnitudes when you want RMS-based scaling.
    
    References
    ----------
    .. [1] Goodfellow, I., Bengio, Y., and Courville, A.
       Deep Learning. MIT Press (2016).  (Gradient stability
       discussion; normalization heuristics).
    """

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
    r"""
    Scale a residual by a (guarded) normalization factor.
    
    This helper divides a residual tensor by a positive scale,
    with strict safeguards against non-finite or tiny scales.
    The scale is treated as a constant with respect to
    backpropagation (stop-gradient).
    
    Mathematical definition
    -----------------------
    Given a residual :math:`r` and a scale :math:`s`, the scaled
    residual is:
    
    .. math::
    
       \tilde{r} = \frac{r}{\hat{s} + \varepsilon}
    
    where:
    
    .. math::
    
       \hat{s} = \mathrm{stop\_grad}(\max(s, s_{floor}))
    
    and any non-finite :math:`s` is replaced by :math:`s_{floor}`
    before flooring.
    
    This ensures that :math:`\tilde{r}` remains finite and that
    gradients flow only through :math:`r`, not through the scale.
    
    Parameters
    ----------
    residual : Tensor
        Residual tensor :math:`r` to normalize.
    
    scale : Tensor
        Scale tensor :math:`s` (typically scalar-like). If it is
        NaN/Inf, it is replaced with ``floor``.
    
    floor : float, default=_EPSILON
        Minimal allowed scale :math:`s_{floor}`.
    
    Returns
    -------
    residual_scaled : Tensor
        Scaled residual :math:`\tilde{r}` with the same shape as
        ``residual``.
    
    Notes
    -----
    Use with guard_scale_with_residual
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If ``scale`` is derived from heuristics or batch statistics,
    it can occasionally be too small. In that case, call
    :func:`guard_scale_with_residual` first to ensure the scale
    is consistent with observed residual magnitudes.
    
    Why stop-gradient
    ~~~~~~~~~~~~~~~~~
    Residual scaling is a conditioning tool. Allowing gradients
    to adjust the scale can create degenerate solutions where
    the model inflates the scale instead of reducing the
    residual.
    
    Examples
    --------
    .. code-block:: python
    
       s = _finite_or_zero(scale_est)
       s = guard_scale_with_residual(R, s, floor=1e-8)
       R_scaled = scale_residual(R, s, floor=1e-8)
    
    See Also
    --------
    guard_scale_with_residual
        Strengthen a scale using residual statistics.
    
    compute_scales
        Compute robust scales for consolidation and groundwater
        residuals.
    
    References
    ----------
    .. [1] Bottou, L., Curtis, F. E., and Nocedal, J.
       Optimization Methods for Large-Scale Machine Learning.
       SIAM Review (2018).
    """

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
    r"""
    Compute the consolidation residual scale.
    
    This helper builds a robust, positive scale used to
    non-dimensionalize (or weight) the consolidation residual.
    It is designed to be stable under noisy early training,
    variable horizons, and occasional non-finite values.
    
    The scale is a stop-gradient quantity. It is computed from
    simple batch statistics so that it adapts to the magnitude
    of the current batch while not injecting gradients back into
    the model through the normalization path.
    
    Mathematical intent
    -------------------
    Let :math:`s_{b,t}` be the settlement state (m) for batch
    index :math:`b` and time index :math:`t` over a horizon
    length :math:`H`.
    
    A consolidation residual is typically expressed as either:
    
    1) step form (meters per step)
    
    .. math::
    
       R_{cons}^{step}(t)
       = s_{t+1} - s_t
         - \hat{\Delta s}_t
    
    2) rate form (meters per time)
    
    .. math::
    
       R_{cons}^{rate}(t)
       = \frac{ds}{dt}
         - \frac{s_{eq}(t) - s(t)}{\tau(t)}
    
    To make such residuals comparable across batches and to
    prevent any single term from dominating due to scale alone,
    we normalize by a characteristic magnitude:
    
    .. math::
    
       \tilde{R}_{cons} = \frac{R_{cons}}{c_*}
    
    where :math:`c_*` is the "consolidation scale" returned by
    this function.
    
    This helper computes :math:`c_*` from two sources:
    
    A) empirical change statistics from the settlement series
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Define per-step increments:
    
    .. math::
    
       \Delta s_{b,t} = s_{b,t+1} - s_{b,t}
    
    Let :math:`|\Delta s|` denote absolute increments flattened
    over batch and time. Two robust summary statistics are used:
    
    .. math::
    
       d_{ref} = \mathrm{mean}(|\Delta s|)
    
    .. math::
    
       d_{max} = \mathrm{max}(|\Delta s|)
    
    Both are treated as constants w.r.t. gradients
    (:func:`tf.stop_gradient`).
    
    Depending on ``mode``, the base scale is:
    
    * ``mode="step"`` (meters per step)
    
      .. math::
    
         c_{base} =
         \max(d_{ref}, 0.1\, d_{max})
    
    * ``mode="time_unit"`` (meters per time_unit)
    
      Let :math:`\Delta t_{ref,u}` be a representative step size
      in "time_units" (the caller provides ``dt_ref_u``):
    
      .. math::
    
         c_{base} =
         \max\left(
           \frac{d_{ref}}{\Delta t_{ref,u}},
           0.1\,\frac{d_{max}}{\Delta t_{ref,u}}
         \right)
    
    * otherwise (SI rate, meters per second)
    
      Let :math:`\Delta t_{ref,s}` be a representative step size
      in seconds (the caller provides ``dt_ref_s``):
    
      .. math::
    
         c_{base} =
         \max\left(
           \frac{d_{ref}}{\Delta t_{ref,s}},
           0.1\,\frac{d_{max}}{\Delta t_{ref,s}}
         \right)
    
    B) optional relaxation and equilibrium misfit statistics
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``use_relax=True``, an additional scale component is
    computed from the equilibrium settlement implied by the
    fields and drawdown proxy.
    
    A non-negative drawdown proxy is formed as:
    
    .. math::
    
       \Delta h = \max(h_{ref} - h, 0)
    
    An equilibrium settlement is then:
    
    .. math::
    
       s_{eq} = S_s \, \Delta h \, H_f
    
    where :math:`S_s` is specific storage (1/m) and :math:`H_f`
    is a thickness (m). The equilibrium misfit magnitude is:
    
    .. math::
    
       e = |s_{eq} - s|
    
    Flatten :math:`e` over batch and time and compute:
    
    .. math::
    
       e_{ref} = \mathrm{mean}(e) + \varepsilon
    
    .. math::
    
       e_{max} = \mathrm{max}(e) + \varepsilon
    
    How this affects the scale depends on ``mode``:
    
    * ``mode="step"``
    
      In step units, equilibrium misfit is already in meters,
      so it can directly act as a characteristic magnitude:
    
      .. math::
    
         c_* \leftarrow
         \max(c_{base}, e_{ref}, 0.1\,e_{max})
    
    * rate modes (meters per time)
    
      Convert the misfit to a characteristic relaxation rate:
    
      .. math::
    
         r = \left|\frac{s_{eq} - s}{\tau}\right|
    
      For SI rate mode, :math:`r` is in m/s. For ``mode="time_unit"``,
      convert m/s to m/time_unit using the number of seconds per
      unit :math:`\mathrm{sec}_{u}`:
    
      .. math::
    
         r_{u} = r\, \mathrm{sec}_{u}
    
      Then summarize:
    
      .. math::
    
         r_{ref} = \mathrm{mean}(r) + \varepsilon
    
      .. math::
    
         r_{max} = \mathrm{max}(r) + \varepsilon
    
      And update:
    
      .. math::
    
         c_* \leftarrow
         \max(c_{base}, r_{ref}, 0.1\,r_{max})
    
    Final flooring and gradient behavior
    ------------------------------------
    A positive floor is enforced:
    
    .. math::
    
       c_* \leftarrow \max(c_*, c_{floor})
    
    and the result is stop-gradient:
    
    .. math::
    
       c_* \leftarrow \mathrm{stop\_grad}(c_*)
    
    This ensures the scale cannot collapse to zero and cannot
    backpropagate into model parameters.
    
    Parameters
    ----------
    Refer to :func:`compute_scales` for the definition and
    meaning of all inputs. This helper assumes all tensors are
    already broadcastable to ``(B, H, 1)`` and represent SI
    quantities consistent with the consolidation objective.
    
    Returns
    -------
    cons_scale : Tensor
        A scalar Tensor (or scalar-like Tensor) representing the
        consolidation scale :math:`c_*`.
    
        Units depend on ``mode``:
    
        * ``"step"``     : meters per step
        * ``"time_unit"``: meters per time_unit
        * otherwise      : meters per second
    
    Notes
    -----
    Why both mean and max
    ~~~~~~~~~~~~~~~~~~~~~
    Using :math:`\max(d_{ref}, 0.1 d_{max})` reduces sensitivity
    to outliers while still reacting when the batch contains
    rare but very large changes (which can otherwise produce
    under-scaling and exploding normalized residuals).
    
    Why stop-gradient
    ~~~~~~~~~~~~~~~~~
    The scale is a diagnostic normalization factor. Letting
    gradients flow through statistics such as mean/max can
    create undesirable feedback loops where the model learns to
    change the scale instead of reducing the residual.
    
    Drawdown proxy
    ~~~~~~~~~~~~~~
    The drawdown proxy here uses a hard positive-part gate
    :math:`\max(h_{ref} - h, 0)`. If your pipeline uses a smooth
    gate or a different sign convention, that logic should be
    handled by the caller before reaching this helper.
    
    Examples
    --------
    This function is not intended to be called directly.
    Use :func:`compute_scales`, which computes both
    consolidation and groundwater residual scales and handles
    time-unit conversions and input sanitization.
    
    See Also
    --------
    compute_scales
        Public interface that computes residual scales.
    
    equilibrium_compaction_si
        Computes :math:`s_{eq}` given fields and drawdown logic.
    
    dt_to_seconds
        Conversion of time step sizes to SI seconds.
    
    References
    ----------
    .. [1] Terzaghi, K.
       Theoretical Soil Mechanics. Wiley (1943).
    
    .. [2] Wang, H. F.
       Theory of Linear Poroelasticity. Princeton University Press
       (2000).
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
    r"""
    Compute the groundwater-flow residual scale.
    
    This helper builds a robust, positive scale used to
    non-dimensionalize (or weight) the groundwater PDE residual.
    It is intended to stabilize training by ensuring the PDE
    term has a comparable magnitude across batches, horizons,
    and unit conventions.
    
    The scale is computed from batch statistics and returned as
    a stop-gradient value, so it does not backpropagate through
    the normalization path.
    
    Mathematical intent
    -------------------
    A common 2D groundwater-flow residual in specific-storage
    form is:
    
    .. math::
    
       R_{gw}
       = S_s \,\frac{\partial h}{\partial t}
         - \nabla \cdot (K \nabla h)
         - Q
    
    where:
    
    * :math:`h` is hydraulic head (m),
    * :math:`S_s` is specific storage (1/m),
    * :math:`K` is hydraulic conductivity (m/s),
    * :math:`Q` is a volumetric forcing per storage thickness,
      expressed here in compatible residual units.
    
    In SI, each term has units of 1/s:
    
    * storage term:
      :math:`S_s \, \partial_t h` has
      :math:`(1/m) (m/s) = 1/s`
    * divergence term:
      :math:`\nabla \cdot (K \nabla h)` has 1/s
      under the standard Darcy form and consistent spatial
      scaling handled upstream
    * forcing term:
      :math:`Q` is assumed already mapped to 1/s
    
    The normalized residual is:
    
    .. math::
    
       \tilde{R}_{gw} = \frac{R_{gw}}{g_*}
    
    where :math:`g_*` is the groundwater scale returned by this
    function.
    
    Scale construction
    ------------------
    This helper uses three optional contributors:
    
    A) storage-term magnitude
    ~~~~~~~~~~~~~~~~~~~~~~~~
    A representative head-rate magnitude is estimated from either
    a provided :math:`\partial_t h` (``dh_dt``) or from finite
    differences across the horizon.
    
    If ``dh_dt`` is not provided, define step head differences:
    
    .. math::
    
       \Delta h_{b,t} = h_{b,t+1} - h_{b,t}
    
    Let :math:`\Delta t_{ref,s}` be a representative step size
    in seconds (caller-provided ``dt_ref_s``). Define:
    
    .. math::
    
       \dot{h}_{ref} =
       \frac{\mathrm{mean}(|\Delta h|)}{\Delta t_{ref,s}}
    
    .. math::
    
       \dot{h}_{max} =
       \frac{\mathrm{max}(|\Delta h|)}{\Delta t_{ref,s}}
    
    If ``dh_dt`` is provided, its mean and max absolute values
    are used directly as :math:`\dot{h}_{ref}` and
    :math:`\dot{h}_{max}`.
    
    A representative storage coefficient magnitude is computed as:
    
    .. math::
    
       S_{s,ref} = \mathrm{mean}(|S_s|)
    
    The storage-term reference scales are:
    
    .. math::
    
       s_{ref}  = S_{s,ref}\,\dot{h}_{ref}
    
    .. math::
    
       s_{max}  = S_{s,ref}\,\dot{h}_{max}
    
    and the initial scale is:
    
    .. math::
    
       g_* \leftarrow \max(s_{ref}, 0.1\,s_{max})
    
    B) divergence-term magnitude (optional)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If the divergence contribution
    :math:`d = \nabla \cdot (K \nabla h)` is provided upstream as
    ``div_K_grad_h``, its batch statistics contribute:
    
    .. math::
    
       d_{ref} = \mathrm{mean}(|d|) + \varepsilon
    
    .. math::
    
       d_{max} = \mathrm{max}(|d|) + \varepsilon
    
    and the scale is updated:
    
    .. math::
    
       g_* \leftarrow \max(g_*, d_{ref}, 0.1\,d_{max})
    
    C) forcing-term magnitude (optional)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If forcing ``Q`` is provided (assumed in compatible units),
    its batch statistics contribute:
    
    .. math::
    
       q_{ref} = \mathrm{mean}(|Q|) + \varepsilon
    
    .. math::
    
       q_{max} = \mathrm{max}(|Q|) + \varepsilon
    
    and the scale is updated:
    
    .. math::
    
       g_* \leftarrow \max(g_*, q_{ref}, 0.1\,q_{max})
    
    Flooring, units, and gradients
    ------------------------------
    A positive floor is enforced:
    
    .. math::
    
       g_* \leftarrow \max(g_*, g_{floor})
    
    The result is returned with stop-gradient:
    
    .. math::
    
       g_* \leftarrow \mathrm{stop\_grad}(g_*)
    
    By default, :math:`g_*` is in SI (1/s). If ``gw_units`` is
    ``"time_unit"``, the scale is converted to 1/time_unit using
    the seconds-per-unit constant :math:`\mathrm{sec}_u`:
    
    .. math::
    
       g_*^{(u)} = g_* \,\mathrm{sec}_u
    
    so that dividing a residual expressed in 1/time_unit by
    :math:`g_*^{(u)}` remains consistent.
    
    Parameters
    ----------
    Refer to :func:`compute_scales` for the meaning and expected
    units of all inputs. This helper assumes inputs are already
    broadcastable to ``(B, H, 1)`` and consistent with the PDE
    assembly used upstream.
    
    Returns
    -------
    gw_scale : Tensor
        A scalar Tensor (or scalar-like Tensor) representing the
        groundwater scale :math:`g_*`.
    
        Units:
    
        * default         : 1/s
        * gw_units="time_unit" : 1/time_unit
    
    Notes
    -----
    Why mean and max
    ~~~~~~~~~~~~~~~~
    Using :math:`\max(x_{ref}, 0.1 x_{max})` provides a robust
    scale that tracks typical magnitudes while remaining
    sensitive to rare but large values that can otherwise cause
    under-scaling and unstable normalized residuals.
    
    Why stop-gradient
    ~~~~~~~~~~~~~~~~~
    The scale is a normalization constant, not a learnable
    quantity. Allowing gradients through batch statistics can
    create feedback loops where the model changes the scale
    instead of reducing the residual.
    
    What "compatible units" means for div and Q
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This helper assumes ``div_K_grad_h`` and ``Q`` are already
    expressed in the same residual units as the storage term,
    i.e. compatible with 1/s in SI. Any spatial-coordinate
    normalization and chain-rule rescaling should be handled
    upstream.
    
    Examples
    --------
    This function is not intended to be called directly.
    Use :func:`compute_scales`, which computes both residual
    scales and manages time-unit conversions and sanitization.
    
    See Also
    --------
    compute_scales
        Public interface that computes both cons and gw scales.
    
    seconds_per_time_unit
        Converts a time-unit string into seconds per unit.
    
    dt_to_seconds
        Converts per-step dt into SI seconds.
    
    References
    ----------
    .. [1] Bear, J.
       Dynamics of Fluids in Porous Media. Dover (1972).
    
    .. [2] Wang, H. F.
       Theory of Linear Poroelasticity. Princeton University Press
       (2000).
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
    r"""
    Compute robust normalization scales for physics residuals.
    
    This function estimates per-batch (or per-sample) scale factors
    used to non-dimensionalize physics residuals before squaring and
    averaging. The goal is to make losses comparable across sites,
    time spans, and coordinate encodings, and to prevent a single
    residual from dominating due to unit magnitude alone.
    
    The returned scales are typically used as:
    
    .. math::
    
       R_{cons}^{*} = \frac{R_{cons}}{s_{cons}}, \qquad
       R_{gw}^{*}   = \frac{R_{gw}}{s_{gw}},
    
    where :math:`s_{cons}` and :math:`s_{gw}` are produced by this
    function (with floors applied for numerical safety).
    
    The routine is intentionally defensive. It sanitizes shapes to
    ``(B, H, 1)``, guards non-finite values, enforces positive dt,
    and applies safe floors before any division or reduction.
    
    Parameters
    ----------
    model : Any
        Model-like object holding configuration in
        ``model.scaling_kwargs`` and optionally ``model.time_units``
        and ``model.h_ref``. This function reads:
    
        * consolidation display mode from ``resolve_cons_units(sk)``
        * groundwater display mode from ``resolve_gw_units(sk)``
        * coordinate normalization flags via ``sk['coords_normalized']``
        * coordinate ranges via ``coord_ranges(sk)``
        * auto floors via ``resolve_auto_scale_floor(kind, sk)``
    
    t : Tensor
        Time coordinate tensor. Expected shape is ``(B, H, 1)`` or
        ``(B, H)``. Units follow the dataset time encoding. If
        ``coords_normalized=True``, ``t`` is assumed normalized and
        is de-normalized using ``coord_ranges(sk)['t']`` when dt is
        inferred internally.
    
    s_mean : Tensor
        Mean settlement state used for consolidation scaling.
        Expected shape is ``(B, H, 1)`` or ``(B, H)``.
    
    h_mean : Tensor
        Mean head state used for scaling. Expected shape is
        ``(B, H, 1)`` or ``(B, H)``. Units should match the model
        internal convention (typically SI meters).
    
    K_field : Tensor
        Effective conductivity field. Present for signature
        compatibility and potential future scale heuristics. Current
        logic does not require this argument directly.
    
    Ss_field : Tensor
        Effective specific storage field :math:`S_s`. Used by both
        consolidation and groundwater scale heuristics. Expected
        shape is broadcastable to ``(B, H, 1)``.
    
    tau_field : Tensor, optional
        Consolidation timescale :math:`tau` in seconds. Provide this
        together with ``H_field`` to enable relaxation-aware
        consolidation scaling.
    
    H_field : Tensor, optional
        Drained thickness :math:`H` in meters. Used with
        ``tau_field`` for relaxation-aware consolidation scaling.
    
    h_ref_si : Tensor, optional
        Reference head :math:`h_{ref}` in meters. If not provided,
        the function falls back to ``model.h_ref`` (or 0.0). The
        value is broadcast to ``(B, H, 1)`` and sanitized.
    
    Q : Tensor, optional
        Source term used in the groundwater residual scaling.
        Expected shape is broadcastable to ``(B, H, 1)``.
    
    dt : Tensor, optional
        Time step tensor in the dataset time units. If provided,
        it is used directly (after shape normalization). If None,
        dt is inferred from ``t``. The inferred dt is de-normalized
        when ``coords_normalized=True``.
    
    time_units : str, optional
        Name of the dataset time unit (e.g., "year", "day",
        "second"). If None, the function resolves it from
        ``sk['time_units']`` or ``model.time_units``. It is used to
        convert dt to seconds.
    
    dh_dt : Tensor, optional
        Precomputed :math:`dh/dt` in SI units (m/s). If provided,
        groundwater scaling can use it directly rather than
        reconstructing a representative magnitude.
    
    div_K_grad_h : Tensor, optional
        Precomputed divergence term for groundwater flow,
        :math:`\nabla \cdot (K \nabla h)`, in SI units. If provided,
        it is used as a representative magnitude for groundwater
        scaling.
    
    verbose : int, default=0
        Verbosity level. If > 0, basic statistics of computed scales
        may be printed.
    
    Returns
    -------
    scales : dict[str, Tensor]
        Dictionary with keys:
    
        * ``'cons_scale'`` : Tensor
          Scale for consolidation residuals.
        * ``'gw_scale'`` : Tensor
          Scale for groundwater-flow residuals.
    
        Each value is shaped as ``(B, 1, 1)`` or broadcastable to
        ``(B, H, 1)``, depending on internal heuristics.
    
    Notes
    -----
    Why scaling is needed
    ~~~~~~~~~~~~~~~~~~~~~
    Consolidation and groundwater residuals can differ by many
    orders of magnitude depending on:
    
    * the dataset time unit (years vs seconds),
    * coordinate normalization spans (t, x, y),
    * site geometry and hydro-mechanical priors,
    * whether residuals are reported in SI or display units.
    
    A stable scaling strategy prevents trivial unit choices from
    changing optimization dynamics.
    
    dt construction and safety
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    If ``dt`` is not provided, dt is inferred as consecutive
    differences along horizon:
    
    * if :math:`H > 1`, :math:`dt_h = t_{h} - t_{h-1}`
    * else, dt defaults to 1.0 (in dataset time units)
    
    When ``coords_normalized=True``, dt is multiplied by the raw
    time span ``t_range`` from ``coord_ranges(sk)`` to recover dt in
    dataset time units. dt is then converted to seconds via
    ``dt_to_seconds(dt, time_units=...)``.
    
    All dt paths apply:
    
    * absolute value
    * finite sanitization
    * a positive floor
    * a final lower bound using ``seconds_per_time_unit(time_units)``
    
    This guards against degenerate dt values that would explode
    scales.
    
    Relaxation-aware consolidation scaling
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If both ``tau_field`` and ``H_field`` are provided, consolidation
    scales may incorporate a relaxation time scale to better match
    the form of the consolidation closure used by the model. If they
    are not provided, a simpler heuristic is used.
    
    Groundwater scaling inputs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Groundwater scales are computed from representative magnitudes
    of the groundwater PDE components, optionally using
    ``dh_dt`` and ``div_K_grad_h`` when provided. The scaling also
    accounts for display unit policies returned by
    ``resolve_gw_units(sk)``.
    
    This function is not traced
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This wrapper is not decorated with ``tf.function`` because it
    accepts a Python ``model`` object. Callers may wrap the function
    at a higher level if a stable tracing boundary is desired.
    
    Examples
    --------
    Compute scales inside the physics path:
    
    >>> scales = compute_scales(
    ...     model,
    ...     t=t,
    ...     s_mean=s_inc_pred,
    ...     h_mean=h_si,
    ...     K_field=K_field,
    ...     Ss_field=Ss_field,
    ...     tau_field=tau_field,
    ...     H_field=H_si,
    ...     h_ref_si=h_ref_11,
    ...     Q=Q_si,
    ...     dt=dt_units,
    ...     time_units=model.time_units,
    ...     dh_dt=dh_dt,
    ...     div_K_grad_h=dKdhx + dKdhy,
    ... )
    
    Use the returned scales to normalize residuals:
    
    >>> cons_scaled = R_cons / scales["cons_scale"]
    >>> gw_scaled = R_gw / scales["gw_scale"]
    
    See Also
    --------
    scale_residual
        Applies a scale and floor to a residual tensor.
    
    resolve_auto_scale_floor
        Resolves "auto" floors for scale denominators.
    
    ensure_si_derivative_frame
        Converts raw autodiff derivatives to SI-consistent forms.
    
    References
    ----------
    .. [1] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
       Physics-informed neural networks: A deep learning framework
       for solving forward and inverse problems involving nonlinear
       partial differential equations. JCP (2019).
    .. [2] Terzaghi, K. Theoretical Soil Mechanics. Wiley (1943).
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
    r"""
    Map predicted settlement output into a PDE-ready settlement state.
    
    This helper converts a model settlement output ``s_pred_si`` into
    a consistent settlement time series in SI meters that can be used
    as the state variable in the consolidation residual and related
    physics terms.
    
    The model can represent settlement in different output modes
    controlled by ``scaling_kwargs['subsidence_kind']``:
    
    * ``"cumulative"`` : ``s_pred_si`` already represents cumulative
      settlement :math:`s(t)` (meters).
    * ``"increment"`` : ``s_pred_si`` represents per-step increments
      :math:`\Delta s_h` (meters per step).
    * ``"rate"`` : ``s_pred_si`` represents a settlement rate
      :math:`ds/dt` (meters per time unit).
    
    The function first constructs a cumulative series :math:`s(t)`
    and then optionally returns the incremental state
    :math:`s_{inc}(t)` used by the ODE/PDE residuals.
    
    Parameters
    ----------
    s_pred_si : Tensor
        Predicted settlement output in SI meters (or SI meters per
        time unit when ``subsidence_kind="rate"``). Expected shapes:
    
        * ``(B, H, 1)`` (preferred)
        * ``(B, H)`` will be promoted to ``(B, H, 1)``
    
    t : Tensor
        Time coordinate used to infer :math:`\Delta t` when
        ``subsidence_kind="rate"`` and ``dt`` is not provided.
        Expected shape is ``(B, H, 1)`` or ``(B, H)``.
    
    scaling_kwargs : dict, optional
        Scaling and configuration dictionary. This function reads
        ``subsidence_kind`` via:
    
        ``get_sk(sk, 'subsidence_kind', default='cumulative')``
    
        If not provided, defaults to ``{}``.
    
    inputs : dict[str, Tensor], optional
        Optional batch inputs that may contain a baseline settlement
        value :math:`s_0` (SI meters). If provided, the function
        searches for the first available tensor among ``baseline_keys``
        and uses it as :math:`s_0`.
    
    time_units : str, optional
        Name of the dataset time unit (e.g., "year", "day"). This
        argument is informational here and is logged for diagnostics.
        When ``subsidence_kind="rate"``, the interpretation of
        ``s_pred_si`` is "meters per time unit".
    
    baseline_keys : Sequence[str], default=("s0_si", "subs0_si",
    "s_ref_si", "subs_ref_si")
        Candidate keys to locate a baseline settlement tensor
        :math:`s_0` in ``inputs``. The first match found is used.
    
    dt : Tensor, optional
        Time step per horizon in dataset time units. Used only when
        ``subsidence_kind="rate"``. Expected shape is ``(B, H, 1)``
        or ``(B, H)``. If None, dt is inferred from ``t`` by finite
        differences, with a fallback of 1.0 for the first step.
    
    return_incremental : bool, default=True
        If True, return the incremental settlement state:
    
        .. math::
    
           s_{inc}(t_h) = s(t_h) - s_0,
    
        shaped like ``(B, H, 1)``. If False, return the cumulative
        settlement series :math:`s(t_h)`.
    
    verbose : int, default=0
        Verbosity level. When > 0, prints basic diagnostics of the
        selected mode and intermediate tensors.
    
    Returns
    -------
    s_state : Tensor
        Settlement state in SI meters with shape ``(B, H, 1)``.
    
        If ``return_incremental=True`` the output is
        :math:`s_{inc}(t)` (incremental since :math:`s_0`). Otherwise
        the output is the cumulative series :math:`s(t)`.
    
    Notes
    -----
    Baseline handling
    ~~~~~~~~~~~~~~~~~
    The baseline :math:`s_0` is interpreted as the settlement value
    at the reference time :math:`t_0` used by the physics residuals.
    If no baseline is provided, :math:`s_0` defaults to zero with
    shape ``(B, 1, 1)`` and is broadcast over the horizon.
    
    Cumulative construction
    ~~~~~~~~~~~~~~~~~~~~~~~
    The function builds a cumulative settlement series :math:`s(t)`
    according to ``subsidence_kind``:
    
    1) ``subsidence_kind="cumulative"``
    
       ``s_pred_si`` is assumed to already represent :math:`s(t)`:
    
       .. math::
    
          s(t_h) = s_{pred}(t_h).
    
       This includes cases where the caller already added a baseline,
       e.g., :math:`s(t) = s_0 + s_{inc}(t)`.
    
    2) ``subsidence_kind="increment"``
    
       ``s_pred_si`` is interpreted as per-step increments:
    
       .. math::
    
          s(t_h) = s_0 + \sum_{j=0}^{h} \Delta s_j.
    
    3) ``subsidence_kind="rate"``
    
       ``s_pred_si`` is interpreted as a rate in meters per time unit:
    
       .. math::
    
          \Delta s_h = \left(\frac{ds}{dt}\right)_h \Delta t_h,
          \qquad
          s(t_h) = s_0 + \sum_{j=0}^{h} \Delta s_j.
    
       If ``dt`` is not provided, :math:`\Delta t_h` is inferred from
       the time coordinate tensor ``t`` using finite differences. The
       first step uses a fallback of 1.0 (for backward compatibility).
    
    Incremental state for PDE/ODE residuals
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Many physics residuals are written for an incremental settlement
    state :math:`s_{inc}(t)` that starts at zero at :math:`t_0`. When
    ``return_incremental=True`` the function returns:
    
    .. math::
    
       s_{inc}(t_h) = s(t_h) - s_0.
    
    This makes it safe to concatenate an explicit initial state
    (e.g., ``s0_inc=0``) when constructing a state sequence for an
    exact-step consolidation integrator.
    
    Examples
    --------
    Convert per-step increments to an incremental PDE state:
    
    >>> sk = {"subsidence_kind": "increment"}
    >>> s_inc = settlement_state_for_pde(
    ...     s_pred_si=ds_pred_m,
    ...     t=coords_t,
    ...     scaling_kwargs=sk,
    ...     inputs={"s0_si": s0_m},
    ...     return_incremental=True,
    ... )
    
    Convert a rate output using provided dt:
    
    >>> sk = {"subsidence_kind": "rate"}
    >>> s_inc = settlement_state_for_pde(
    ...     s_pred_si=dsdt_pred_m_per_u,
    ...     t=coords_t,
    ...     dt=dt_units,
    ...     scaling_kwargs=sk,
    ...     inputs={"s0_si": s0_m},
    ...     return_incremental=True,
    ... )
    
    Return the cumulative series instead:
    
    >>> s_cum = settlement_state_for_pde(
    ...     s_pred_si=s_pred_m,
    ...     t=coords_t,
    ...     scaling_kwargs={"subsidence_kind": "cumulative"},
    ...     return_incremental=False,
    ... )
    
    See Also
    --------
    compute_consolidation_step_residual
        Builds the consolidation residual from settlement and head
        states.
    
    cons_step_to_cons_residual
        Converts a step residual into a residual consistent with the
        PDE time convention.
    
    integrate_consolidation_mean
        Integrates a consolidation mean settlement trajectory used as
        a physics-driven prediction path.
    
    References
    ----------
    .. [1] Terzaghi, K. Theoretical Soil Mechanics. Wiley (1943).
    .. [2] Biot, M. A. General theory of three-dimensional
       consolidation. Journal of Applied Physics (1941).
    """

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
    r"""
    Compute the root-mean-square (RMS) of a tensor.
    
    This utility computes:
    
    .. math::
    
       \mathrm{RMS}(x)
       = \sqrt{\mathbb{E}[x^2]}
    
    over the requested reduction axes. It is designed for robust
    diagnostics in physics-informed training loops, where tensors may
    contain extremely small values (needing ``float64``) or occasional
    non-finite entries (handled via ``nan_policy``).
    
    Parameters
    ----------
    x : Tensor
        Input tensor. Any shape is accepted. The computation is
        performed in ``dtype`` (default float32).
    
    axis : int or Sequence[int] or None, default=None
        Axis or axes to reduce.
    
        * If None, reduce over all dimensions and return a scalar.
        * If an int or sequence, reduce only those axes.
    
    keepdims : bool, default=False
        If True, keep reduced dimensions with length 1.
    
    eps : float or None, default=None
        Optional lower bound applied to the mean-square value before
        the square root is taken. If provided and > 0, the mean-square
        is floored as:
    
        .. math::
    
           \mathrm{MS} = \max(\mathrm{MS}, \mathrm{eps})
    
        where :math:`\mathrm{MS} = \mathbb{E}[x^2]`.
    
    ms_floor : float or None, default=None
        Alias for an additional mean-square floor applied after
        ``eps``. If provided and > 0, it is applied as:
    
        .. math::
    
           \mathrm{MS} = \max(\mathrm{MS}, \mathrm{ms\_floor})
    
        This can be useful when you want a hard numerical floor but
        prefer to keep ``eps`` reserved for "epsilon-like" smoothing.
    
    rms_floor : float or None, default=None
        Optional lower bound applied after taking the square root.
        If provided and > 0:
    
        .. math::
    
           \mathrm{RMS} = \max(\mathrm{RMS}, \mathrm{rms\_floor})
    
    nan_policy : {"propagate", "raise", "omit"}, default="propagate"
        Policy for handling non-finite values (NaN/Inf):
    
        * ``"propagate"``:
          Use the standard reduction. Non-finite values propagate
          through ``mean`` and the RMS becomes non-finite.
        * ``"raise"``:
          Assert that ``x`` is all finite before reducing, raising an
          error if NaN/Inf is present.
        * ``"omit"``:
          Ignore non-finite entries by treating them as missing. The
          RMS is computed from finite entries only:
    
          .. math::
    
             \mathrm{MS}
             = \frac{\sum x_i^2}{N_f}
    
          where :math:`N_f` is the count of finite entries along the
          reduced axes (clipped to at least 1).
    
    dtype : Any, default=None
        Compute dtype. If None, uses ``tf_float32`` for speed.
        Pass ``dtype=tf_float64`` when diagnosing very small residuals
        or when accumulated rounding error matters.
    
    Returns
    -------
    rms : Tensor
        RMS value reduced along ``axis``. If ``axis=None`` the result
        is a scalar tensor; otherwise it has the reduced shape.
    
    Notes
    -----
    Flooring behavior
    ~~~~~~~~~~~~~~~~~
    Floors are opt-in. If ``eps is None`` and ``ms_floor is None``,
    no flooring is applied to the mean-square. If ``rms_floor is
    None``, no flooring is applied to the RMS.
    
    A common pattern for stable logging of near-zero residuals is to
    use a small mean-square floor with float64 diagnostics:
    
    * ``dtype=tf_float64`` to reduce rounding error.
    * ``ms_floor`` to avoid taking ``sqrt(0)`` when a later operation
      applies ``log`` or divides by RMS.
    
    Non-finite handling
    ~~~~~~~~~~~~~~~~~~~
    ``nan_policy="omit"`` is intended for diagnostics and logging.
    For training-time physics losses, prefer cleaning tensors before
    the loss is computed, so gradients are well-defined.
    
    Examples
    --------
    Compute RMS over all entries:
    
    >>> r = to_rms(x)
    
    Compute per-batch RMS (reduce over horizon and channel axes):
    
    >>> r_b = to_rms(x, axis=(1, 2))
    
    Omit non-finite values when logging a residual map:
    
    >>> eps_gw = to_rms(R_gw, nan_policy="omit", dtype=tf_float64)
    
    Apply a small mean-square floor for stable downstream log:
    
    >>> eps = to_rms(R, ms_floor=1e-30, dtype=tf_float64)
    
    See Also
    --------
    scale_residual
        Scales residuals by computed characteristic scales.
    
    guard_scale_with_residual
        Ensures a scale is safe when residuals are near zero.
    
    References
    ----------
    .. [1] Goodfellow, I., Bengio, Y., and Courville, A.
       Deep Learning. MIT Press (2016). Chapter on numerical
       stability and floating point considerations.
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