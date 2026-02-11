# -*- coding: utf-8 -*-
# _geoprior_losses.py

"""
GeoPrior loss assembly and logging helpers.

This module centralizes:
- physics loss assembly (no double offset)
- return packaging for train/test/eval
"""

from __future__ import annotations

from typing import Any
 
from ....compat.keras_fit import (
    ensure_targets_for_outputs,
    update_compiled_metrics as _update_compiled_metrics,
    compiled_metrics_dict,
)
from ... import KERAS_DEPS
# from ..._shapes import _as_BHO

from .utils import get_sk

Tensor = KERAS_DEPS.Tensor
tf_float32 = KERAS_DEPS.float32
tf_constant = KERAS_DEPS.constant
tf_identity = KERAS_DEPS.identity
Tensor = KERAS_DEPS.Tensor
tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor
tf_expand_dims =KERAS_DEPS.expand_dims 
tf_squeeze = KERAS_DEPS.squeeze 
tf_print = KERAS_DEPS.print 
tf_shape = KERAS_DEPS.shape 


# ---------------------------------------------------------------------
# Small switches
# ---------------------------------------------------------------------
def should_log_physics(model: Any) -> bool:
    """
    Decide whether to expose physics keys in logs.

    If physics is off, logs are included only if
    scaling_kwargs["log_physics_when_off"] is True.
    """
    sk = getattr(model, "scaling_kwargs", None) or {}
    if not hasattr(model, "_physics_off"):
        return True
    if not model._physics_off():
        return True
    return bool(get_sk(
        sk,
        "log_physics_when_off",
        default=False,
    ))

# ---------------------------------------------------------------------
# Physics multiplier + loss assembly
# ---------------------------------------------------------------------

def assemble_physics_loss(
    model: Any,
    *,
    loss_cons: Tensor,
    loss_gw: Tensor,
    loss_prior: Tensor,
    loss_smooth: Tensor,
    loss_mv: Tensor,
    loss_q_reg: Tensor,
    loss_bounds: Tensor,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    r"""
    Assemble the full physics objective with an offset-aware multiplier.
    
    This helper combines individual physics loss components computed by
    the GeoPrior PINN core into:
    
    * an unscaled physics loss (for logging and diagnostics),
    * a scaled physics loss (the quantity added to the data loss),
    * the global physics multiplier used for scaling,
    * a dictionary of per-term scaled contributions that is consistent
      with the scaled physics loss.
    
    The function implements the GeoPrior weighting convention:
    
    1) Each component loss is first multiplied by its corresponding
       per-term weight stored on the model instance (``lambda_*``).
    
    2) A global physics multiplier ``phys_mult`` is computed by
       ``model._physics_loss_multiplier()``, which depends on
       ``model.offset_mode`` and the scalar state ``model._lambda_offset``.
    
    3) The multiplier is applied to PDE-style terms by default, while
       certain calibration/regularization terms can opt out depending on
       model flags (see Notes).
    
    Formally, define weighted terms:
    
    .. math::
    
       T_{cons}   = \lambda_{cons}   L_{cons}
       \\
       T_{gw}     = \lambda_{gw}     L_{gw}
       \\
       T_{prior}  = \lambda_{prior}  L_{prior}
       \\
       T_{smooth} = \lambda_{smooth} L_{smooth}
       \\
       T_{bounds} = \lambda_{bounds} L_{bounds}
       \\
       T_{mv}     = \lambda_{mv}     L_{mv}
       \\
       T_{q}      = \lambda_{q}      L_{q}
    
    Let the PDE core sum be:
    
    .. math::
    
       L_{core} = T_{cons} + T_{gw} + T_{prior} + T_{smooth} + T_{bounds}
    
    and the unscaled physics loss be:
    
    .. math::
    
       L_{phys,raw} = L_{core} + T_{mv} + T_{q}
    
    The scaled physics loss is:
    
    .. math::
    
       L_{phys,scaled} =
           phys\_mult \, L_{core}
           + s_{mv} \, T_{mv}
           + s_{q}  \, T_{q}
    
    where:
    
    * :math:`s_{mv} = phys\_mult` if ``model._scale_mv_with_offset`` is
      True, else :math:`s_{mv} = 1`.
    * :math:`s_{q}  = phys\_mult` if ``model._scale_q_with_offset`` is
      True, else :math:`s_{q} = 1`.
    
    Parameters
    ----------
    model : Any
        Model-like object providing the weighting attributes:
    
        * ``lambda_cons``, ``lambda_gw``, ``lambda_prior``,
          ``lambda_smooth``, ``lambda_bounds``, ``lambda_mv``,
          ``lambda_q``
        * ``_physics_loss_multiplier()`` method
        * optional flags ``_scale_mv_with_offset`` and
          ``_scale_q_with_offset``
    
    loss_cons : Tensor
        Consolidation loss :math:`L_{cons}` (typically mean-square of a
        scaled consolidation residual).
    
    loss_gw : Tensor
        Groundwater-flow PDE loss :math:`L_{gw}` (typically mean-square
        of a scaled groundwater residual).
    
    loss_prior : Tensor
        Timescale-consistency prior loss :math:`L_{prior}` (often
        mean-square of a log-timescale residual).
    
    loss_smooth : Tensor
        Smoothness prior loss :math:`L_{smooth}` (regularizes spatial
        gradients of learned fields).
    
    loss_mv : Tensor
        Storage identity / compressibility calibration loss
        :math:`L_{mv}`.
    
    loss_q_reg : Tensor
        Forcing regularization loss :math:`L_{q}` (typically
        mean-square of the SI forcing field :math:`Q`).
    
    loss_bounds : Tensor
        Soft-bounds penalty loss :math:`L_{bounds}` derived from bound
        residuals (if enabled).
    
    Returns
    -------
    physics_raw : Tensor
        Unscaled physics loss:
    
        .. math::
    
           L_{phys,raw} = L_{core} + T_{mv} + T_{q}
    
        Useful for diagnostics, independent of ``lambda_offset``.
    
    physics_scaled : Tensor
        Scaled physics loss, consistent with the global multiplier and
        the optional scaling rules for ``mv`` and ``q`` terms.
    
    phys_mult : Tensor
        The global physics multiplier returned by
        ``model._physics_loss_multiplier()``.
    
    terms_scaled : dict[str, Tensor]
        Per-term contributions consistent with ``physics_scaled``.
        Keys are:
    
        * ``'cons'``, ``'gw'``, ``'prior'``, ``'smooth'``, ``'bounds'``,
          ``'mv'``, ``'q'``.
    
    Notes
    -----
    Offset-aware scaling policy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    The global multiplier ``phys_mult`` is intended as a single knob to
    warm up or damp all PDE-style physics terms together. By default:
    
    * PDE-style terms (cons, gw, prior, smooth, bounds) are always scaled
      by ``phys_mult``.
    * The ``mv`` term is treated as a calibration loss and is not scaled
      by ``phys_mult`` unless ``model._scale_mv_with_offset`` is True.
    * The ``q`` regularization term is scaled by ``phys_mult`` only if
      ``model._scale_q_with_offset`` is True.
    
    This separation avoids unintended suppression of calibration signals
    when physics warmup is used.
    
    Logging and gradient debugging
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Returning both ``physics_raw`` and ``physics_scaled`` helps debug
    training stability:
    
    * ``physics_raw`` shows whether residual magnitudes are decreasing.
    * ``physics_scaled`` shows the effective contribution to the total
      optimization objective.
    
    Examples
    --------
    Assemble physics loss inside a training loop:
    
    >>> physics_raw, physics_scaled, phys_mult, terms = (
    ...     assemble_physics_loss(
    ...         model,
    ...         loss_cons=loss_cons,
    ...         loss_gw=loss_gw,
    ...         loss_prior=loss_prior,
    ...         loss_smooth=loss_smooth,
    ...         loss_mv=loss_mv,
    ...         loss_q_reg=loss_q_reg,
    ...         loss_bounds=loss_bounds,
    ...     )
    ... )
    >>> total_loss = data_loss + physics_scaled
    
    Inspect per-term contributions:
    
    >>> float(terms["prior"])
    0.0123
    
    See Also
    --------
    fusionlab.nn.pinn.geoprior.step_core.physics_core
        Produces the component losses used as inputs here.
    
    GeoPriorSubsNet.compile
        Configures the ``lambda_*`` weights and the offset multiplier.
    
    References
    ----------
    .. [1] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
       Physics-informed neural networks: A deep learning framework
       for solving forward and inverse problems involving nonlinear
       partial differential equations. Journal of Computational
       Physics, 2019.
    """

    # ----------------------------------------------------------
    # 1) Unscaled weighted terms.
    # ----------------------------------------------------------
    t_cons = model.lambda_cons * loss_cons
    t_gw = model.lambda_gw * loss_gw
    t_prior = model.lambda_prior * loss_prior
    t_smooth = model.lambda_smooth * loss_smooth
    t_bounds = model.lambda_bounds * loss_bounds
    t_mv = model.lambda_mv * loss_mv
    t_q = model.lambda_q * loss_q_reg

    core_raw = t_cons + t_gw + t_prior + t_smooth + t_bounds
    physics_raw = core_raw + t_mv + t_q

    # ----------------------------------------------------------
    # 2) Global multiplier (offset_mode aware).
    # ----------------------------------------------------------
    phys_mult = model._physics_loss_multiplier()

    scale_mv = bool(getattr(model, "_scale_mv_with_offset", False))
    scale_q = bool(getattr(model, "_scale_q_with_offset", False))

    core_scaled = phys_mult * core_raw
    mv_scaled = phys_mult * t_mv if scale_mv else t_mv
    q_scaled = phys_mult * t_q if scale_q else t_q

    physics_scaled = core_scaled + mv_scaled + q_scaled
    # ----------------------------------------------------------
    # 3) Per-term contributions consistent with physics_scaled.
    # ----------------------------------------------------------
    terms_scaled = {
        "cons":   phys_mult * t_cons,
        "gw":     phys_mult * t_gw,
        "prior":  phys_mult * t_prior,
        "smooth": phys_mult * t_smooth,
        "bounds": phys_mult * t_bounds,
        "mv": mv_scaled,
        "q": q_scaled,
    }
    return physics_raw, physics_scaled, phys_mult, terms_scaled

# ---------------------------------------------------------------------
# Physics bundles
# ---------------------------------------------------------------------
def zero_physics_bundle(
    model: Any,
    *,
    dtype: Any = tf_float32,
) -> dict[str, Tensor]:
    """
    Canonical zero physics bundle.

    This keeps dashboards stable when requested.
    """
    z = tf_constant(0.0, dtype=dtype)
    one = tf_constant(1.0, dtype=dtype)

    lam = getattr(model, "_lambda_offset", None)
    if lam is None:
        lam = one

    return {
        "physics_loss_raw": z,
        "physics_loss_scaled": z,
        "physics_mult": one,
        "lambda_offset": tf_identity(lam),

        "loss_consolidation": z,
        "loss_gw_flow": z,
        "loss_prior": z,
        "loss_smooth": z,
        "loss_mv": z,
        "loss_q_reg": z,
        "q_rms": z,
        "q_gate": z,
        "subs_resid_gate": z,

        "loss_bounds": z,

        "epsilon_prior": z,
        "epsilon_cons": z,
        "epsilon_gw": z,
        "epsilon_cons_raw": z,
        "epsilon_gw_raw": z,
    }


def build_physics_bundle(
    model: Any,
    *,
    physics_loss_raw: Tensor,
    physics_loss_scaled: Tensor,
    phys_mult: Tensor,
    loss_cons: Tensor,
    loss_gw: Tensor,
    loss_prior: Tensor,
    loss_smooth: Tensor,
    loss_mv: Tensor,
    loss_q_reg: Tensor,
    q_rms: Tensor,
    q_gate: Tensor,
    subs_resid_gate: Tensor,
    loss_bounds: Tensor,
    eps_prior: Tensor,
    eps_cons: Tensor,
    eps_gw: Tensor,
    eps_cons_raw: Tensor | None = None,
    eps_gw_raw: Tensor | None = None,
) -> dict[str, Tensor]:
    """
    Canonical physics bundle used by train/test/eval packers.
    """
    z = tf_constant(0.0, dtype=tf_float32)

    lam = getattr(model, "_lambda_offset", None)
    if lam is None:
        lam = tf_constant(1.0, tf_float32)

    return {
        "physics_loss_raw": physics_loss_raw,
        "physics_loss_scaled": physics_loss_scaled,
        "physics_mult": phys_mult,
        "lambda_offset": tf_identity(lam),

        "loss_consolidation": loss_cons,
        "loss_gw_flow": loss_gw,
        "loss_prior": loss_prior,
        "loss_smooth": loss_smooth,
        "loss_mv": loss_mv,
        
        "loss_q_reg": loss_q_reg,
        "q_rms": q_rms,
        "q_gate": q_gate,
        "subs_resid_gate": subs_resid_gate,

        "loss_bounds": loss_bounds,

        "epsilon_prior": eps_prior,
        "epsilon_cons": eps_cons,
        "epsilon_gw": eps_gw,
        "epsilon_cons_raw": (
            eps_cons_raw if eps_cons_raw is not None else z
        ),
        "epsilon_gw_raw": (
            eps_gw_raw if eps_gw_raw is not None else z
        ),
    }


# ---------------------------------------------------------------------
# Epsilon metric helpers
# ---------------------------------------------------------------------
def update_epsilon_metrics(
    model: Any,
    *,
    eps_prior: Tensor,
    eps_cons: Tensor,
    eps_gw: Tensor,
) -> None:
    """
    Update optional epsilon metrics if present.
    """
    m = getattr(model, "eps_prior_metric", None)
    if m is not None:
        m.update_state(eps_prior)

    m = getattr(model, "eps_cons_metric", None)
    if m is not None:
        m.update_state(eps_cons)

    m = getattr(model, "eps_gw_metric", None)
    if m is not None:
        m.update_state(eps_gw)


def _set_metric_results(m, fallback):
    try:
        # Keras 3 check: if not built, result() crashes.
        if hasattr(m, "built") and not m.built:
            return fallback
        return m.result()
    except Exception:
        return fallback
    
def epsilon_value_for_logs(
        model: Any, which: str, 
        fallback: Tensor
    ) -> Tensor:
    """
    Prefer tracked epsilon metric if it exists.
    """
    key = f"eps_{which}_metric"
    m = getattr(model, key, None)
    if m is not None:
        return _set_metric_results(m, fallback)
    return fallback


# ---------------------------------------------------------------------
# Train/Test step packer (no duplication)
# ---------------------------------------------------------------------
def _ordered_by_outputs(model, d):
    keys = getattr(model, "output_names", None) or getattr(
        model, "_output_keys", None)
    if not keys:
        keys = list(d.keys())
    return [d[k] for k in keys if k in d and d[k] is not None]

def _get_real_compile_metrics(model):
    # Keras 3: real container
    cm = getattr(model, "_compile_metrics", None)
    if cm is not None:
        return cm

    # Fallbacks (older / different builds)
    for name in ("_compiled_metrics", "_metrics_container"):
        cm = getattr(model, name, None)
        if cm is not None:
            return cm

    # Last resort: deprecated property (may be wrapper)
    return getattr(model, "compiled_metrics", None)


def update_compiled_metrics(model, targets, y_pred):
    r"""
    Update compiled Keras metrics for multi-output dict predictions.
    
    This helper updates the metric container created by
    :meth:`tf.keras.Model.compile` in a way that is robust across Keras 2
    and Keras 3 behavior when the model uses named outputs (dict-style)
    and the training loop uses a custom :meth:`train_step` /
    :meth:`test_step`.
    
    The function:
    
    1) Locates the "real" compiled metrics object for the model (if any)
       using an internal helper (``_get_real_compile_metrics``).
    
    2) Determines the ordered list of output keys from the model
       (preferably ``model.output_names`` and then ``model._output_keys``).
    
    3) Aligns the shapes of ground truth tensors to match prediction
       tensors (via ``_as_BHO``), so metrics always see consistent batch
       layout.
    
    4) Attempts to update metrics using the most stable calling pattern
       for the installed Keras version:
    
       * First try list-based update (``update_state(y_true_list,
         y_pred_list)``), which avoids dict key routing issues that can
         occur with certain Keras 2 configurations.
       * If that fails, fall back to dict-based update
         (``update_state(y_true_dict, y_pred_dict)``).
       * If that also fails, fall back to manually updating per-output
         metric objects by matching metric name prefixes.
    
    This helper is primarily used to keep metric reporting consistent
    when custom training logic bypasses the default Keras fit loop
    internals.
    
    Parameters
    ----------
    model : Any
        A Keras model instance (or model-like object) that has been
        compiled with ``metrics`` and possibly multi-output losses.
    
    targets : dict-like
        Ground truth outputs keyed by output name. Values can be tensors
        or tensor-like arrays.
    
    y_pred : dict-like
        Model predictions keyed by output name. Values are tensors.
    
    Returns
    -------
    None
        Updates the compiled metrics state in-place.
    
    Notes
    -----
    Why a custom updater is needed
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Keras multi-output metric routing depends on how metrics were compiled
    (list-based vs dict-based) and how outputs are named and returned. In
    custom :meth:`train_step` / :meth:`test_step`, you often compute losses
    manually and must also call metric updates manually to preserve the
    behavior of ``model.fit``.
    
    Compatibility behavior
    ~~~~~~~~~~~~~~~~~~~~~~~~
    - In some Keras 2 environments, calling ``compiled.update_state`` with
      dicts can fail or silently mis-route metrics when output names do not
      align with how the metric container was constructed. The list-first
      strategy is a defensive approach.
    - The final manual fallback updates metric objects directly by matching
      their name prefix (``<output_name>_``) and skipping loss-like metrics.
    
    Shape normalization
    ~~~~~~~~~~~~~~~~~~~~~
    The helper normalizes ground-truth shapes to match prediction shapes
    before updating metrics. This reduces common failures when targets are
    provided as ``(B,H)`` or ``(B,H,1)`` while predictions may be
    ``(B,H,Q,1)`` (quantiles) or similar.
    
    Examples
    --------
    Inside a custom test_step:
    
    >>> y_pred = model(inputs, training=False)
    >>> update_compiled_metrics(model, targets, y_pred)
    
    Inside a custom train_step:
    
    >>> with tf.GradientTape() as tape:
    ...     y_pred = model(inputs, training=True)
    ...     loss = model.compiled_loss(...)
    >>> update_compiled_metrics(model, targets, y_pred)
    
    See Also
    --------
    tf.keras.Model.compiled_metrics
        Standard entry point for metric containers in Keras.
    
    GeoPriorSubsNet.train_step
        Custom training loop that may use this helper to keep metrics
        consistent.
    
    References
    ----------
    .. [1] Keras Team. Keras fit/compile metrics routing documentation.
    """
    
    _update_compiled_metrics(
        model = model, 
        targets=targets, 
        y_pred=y_pred
    )
    
    # compiled = _get_real_compile_metrics(model)
    # if compiled is None:
    #     return

    # out_keys = list(
    #     getattr(model, "output_names", None)
    #     or getattr(model, "_output_keys", None)
    #     or []
    # )
    # if not out_keys:
    #     return
    # # XXX IMPORTANT: recheck to let the loss compiles with non multiple targts
    # # keys = [k for k in out_keys if (k in targets) and (k in y_pred)]
    # keys = [
    #     k for k in out_keys
    #     if (k in targets)
    #     and (targets[k] is not None)
    #     and (k in y_pred)
    # ]
    # if not keys:
    #     return

    # # Plain dicts + y_true normalized to BHO
    # t_norm = {k: _as_BHO(targets[k], y_pred=y_pred[k]) for k in keys}
    # p_norm = {k: y_pred[k] for k in keys}

    # # For keras 2.0 
    # yt_list = [t_norm[k] for k in keys]
    # yp_list = [p_norm[k] for k in keys]
    
    # # Try list path first (works with list-compiled metrics,
    # # avoids dict key weirdness) in keras 2
    # try:
    #     compiled.update_state(yt_list, yp_list)
    #     return
    # except:
    #     pass
    
    # # IMPORTANT: use dict path (per-output), never lists
    # try:
    #     compiled.update_state(t_norm, p_norm)
    #     return
    # except:
    #     # Safe fallback: update per-output metrics manually
    #     for out in keys:
    #         yt = t_norm[out]
    #         yp = p_norm[out]
    #         prefix = out + "_"
    #         for m in getattr(model, "metrics", []):
    #             name = getattr(m, "name", "") or ""
    #             if name.startswith(prefix) and "loss" not in name:
    #                 m.update_state(yt, yp)

def _needs_full_quantiles(metric_name: str) -> bool:
    n = metric_name.lower()
    return ("coverage" in n) or ("sharpness" in n)

def _metric_key_from_name(name: str):
    # Keras names: "subs_pred_mae", "subs_pred_coverage80", "gwl_pred_mse", ...
    # Skip Keras loss trackers
    if name in ("loss",) or name.endswith("_loss"):
        return None

    if name.startswith("subs_pred_"):
        return "subs_pred"
    if name.startswith("gwl_pred_"):
        return "gwl_pred"
    return None

# ---------------------------------------------------------------------
# Helper: Keras 3 Safe Result Getter
# ---------------------------------------------------------------------
def safe_metric_result(metric: Any, fallback: float = 0.0) -> Tensor:
    """
    Safely obtain a metric result (Keras 3-safe).

    In Keras 3, calling `metric.result()` may raise if the metric hasn't
    been built/updated yet. In that case we return `fallback`.

    Parameters
    ----------
    metric : Any
        A Keras metric instance (or a scalar/tensor-like).
    fallback : float, default=0.0
        Value returned if the metric is not ready or errors.

    Returns
    -------
    Tensor
        Metric result as a float32 tensor (or fallback).
    """
    if metric is None:
        return tf_constant(fallback, dtype=tf_float32)

    # Keras 3: many metrics expose `.built`; if False, result() may raise.
    if hasattr(metric, "built") and not getattr(metric, "built", True):
        return tf_constant(fallback, dtype=tf_float32)

    # Standard metric objects
    if hasattr(metric, "result"):
        try:
            return tf_convert_to_tensor(metric.result(), dtype=tf_float32)
        except Exception:
            return tf_constant(fallback, dtype=tf_float32)

    # Scalar / tensor-like fallback
    try:
        return tf_convert_to_tensor(metric, dtype=tf_float32)
    except Exception:
        return tf_constant(fallback, dtype=tf_float32)
    
def pack_step_results(
    model: Any,
    *,
    total_loss: Tensor,
    data_loss: Tensor,
    targets: Any,
    y_pred: Any,
    physics: dict[str, Tensor] | None = None,
    manual_trackers: dict | None = None,
) -> dict[str, Tensor]:
    r"""
    Canonical return dictionary for custom ``train_step`` / ``test_step``.
    
    This helper builds a stable logging payload for GeoPrior-style models
    that use a custom training loop. It combines:
    
    * supervised loss scalars (data and total),
    * compiled Keras metrics (if available),
    * optional manual trackers (e.g., add-on quantile trackers),
    * optional physics diagnostics (PINN losses and epsilons).
    
    The function is intentionally defensive across Keras versions:
    
    * It explicitly updates and reads compiled metrics using
      ``update_compiled_metrics`` and the underlying compile-metrics
      container, rather than relying on ``model.metrics`` alone.
    * It reserves the key ``"loss"`` as the authoritative scalar returned
      to Keras, while also including explicit ``"total_loss"`` and
      ``"data_loss"`` entries for clarity.
    
    Parameters
    ----------
    model : Any
        Model-like object that provides compiled metrics and configuration.
        Expected attributes and helpers include:
    
        * ``metrics`` (optional list of metric objects)
        * ``output_names`` or ``_output_keys`` (output ordering)
        * ``scaling_kwargs`` (optional dict)
        * functions used by this module such as
          ``should_log_physics``, ``zero_physics_bundle``,
          ``update_compiled_metrics``, ``safe_metric_result``,
          ``update_epsilon_metrics``, and ``epsilon_value_for_logs``.
    
    total_loss : Tensor
        The scalar loss used for optimization in the current step. This is
        returned as ``results["loss"]`` and ``results["total_loss"]``.
    
    data_loss : Tensor
        The supervised loss computed from the compiled loss function
        (i.e., the data term). Returned as ``results["data_loss"]``.
    
    targets : Any
        Ground-truth targets for the supervised outputs. Typically a dict
        keyed by output names (e.g., ``{"subs_pred": ..., "gwl_pred": ...}``)
        but may be any structure supported by ``update_compiled_metrics``.
    
    y_pred : Any
        Predicted outputs corresponding to ``targets``. Typically a dict
        keyed by output names.
    
    physics : dict[str, Tensor] or None, optional
        Physics bundle produced by ``physics_core`` (or an equivalent).
        If None and physics logging is enabled, a zero bundle is used.
    
    manual_trackers : dict or None, optional
        Optional additional trackers to log. Values may be metric objects
        with ``result()`` or raw scalars/tensors. This is typically used
        for add-on metrics that are not part of Keras compiled metrics.
    
    Returns
    -------
    results : dict[str, Tensor]
        A dictionary suitable for returning from ``train_step`` or
        ``test_step``. At minimum it contains:
    
        * ``loss``: total loss used by Keras progress reporting.
        * ``total_loss``: same as ``loss`` (explicit alias).
        * ``data_loss``: supervised/data loss term.
    
        If compiled metrics are available, additional keys are included
        (e.g., ``subs_pred_mae``, quantile coverage, etc.). If physics
        logging is enabled, physics diagnostics are appended (see Notes).
    
    Notes
    -----
    Metric collection strategy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Compiled metrics are updated via ``update_compiled_metrics`` and then
    read from the underlying compile-metrics object. This avoids common
    routing failures when using dict outputs in custom training loops.
    
    Reserved and excluded keys
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Certain names are reserved to prevent collisions with Keras internals
    and to ensure that the loss scalar remains authoritative. Some epsilon
    fields may also be excluded from the compiled-metric collection to
    avoid duplicate/conflicting reporting.
    
    Physics logging
    ~~~~~~~~~~~~~~~
    If physics logging is enabled (``should_log_physics(model)`` returns
    True), this helper adds a consistent set of physics metrics, typically:
    
    * physics losses (raw and scaled),
    * per-term losses (consolidation, gw flow, priors, bounds),
    * epsilon metrics (scaled and raw variants).
    
    If physics is disabled for the model and logging is enabled, a zero
    bundle is inserted to keep log schemas stable.
    
    Q and residual gates
    ~~~~~~~~~~~~~~~~~~~~
    When ``scaling_kwargs`` requests Q diagnostics
    (``log_q_diagnostics=True``), additional fields such as Q RMS and gate
    values may be included for debugging training schedules.
    
    Examples
    --------
    Inside a custom training step:
    
    >>> results = pack_step_results(
    ...     model,
    ...     total_loss=total_loss,
    ...     data_loss=data_loss,
    ...     targets=targets,
    ...     y_pred=y_pred,
    ...     manual_trackers=(model.add_on.as_dict if model.add_on else None),
    ...     physics=physics_bundle,
    ... )
    >>> return results
    
    Inside a custom test step:
    
    >>> return pack_step_results(
    ...     model,
    ...     total_loss=total_loss,
    ...     data_loss=data_loss,
    ...     targets=targets,
    ...     y_pred=y_pred,
    ...     physics=physics_bundle,
    ... )
    
    See Also
    --------
    update_compiled_metrics
        Compatibility helper to update metrics for multi-output dicts.
    
    assemble_physics_loss
        Builds the scaled physics objective used in ``total_loss``.
    
    physics_core
        Produces the physics bundle consumed by this packer.
    
    References
    ----------
    .. [1] Keras Team. Customizing ``fit`` with ``train_step`` and
       ``test_step``. Keras guides and API documentation.
    """

    
    RESERVED = {"loss", "total_loss", "data_loss" , "compile_metrics"}
    EXCLUDE = {"epsilon_prior", "epsilon_cons", "epsilon_gw"}

#     # ------------------------------------------------------------------
#     # 1) Collect logs (DO NOT rely on model.metrics only)
#     # ------------------------------------------------------------------
#     results: dict[str, Tensor] = {}
    
#     def _add_compiled_results():
#         cm = _get_real_compile_metrics(model)
#         if cm is None:
#             return
#         try:
#             # In Keras 3 CompileMetrics.result() returns a dict like:
#             # {'subs_pred_mae_q50': ..., 'subs_pred_coverage80': ..., ...}
#             d = cm.result()
#         except Exception:
#             return
#         if not isinstance(d, dict):
#             return
    
#         for k, v in d.items():
#             if (not k) or (k in RESERVED) or (k in EXCLUDE):
#                 continue
#             if k in results:
#                 continue
#             results[k] = tf_convert_to_tensor(v, dtype=tf_float32)  
            
#     # ------------------------------------------------------------------
#     # 0) Update compiled metrics (MANUAL UPDATE for Keras 3)
#     # ------------------------------------------------------------------
#     # We DO NOT use model.compiled_metrics.update_state(targets, y_pred)
#     # because it crashes with TypeError on dicts in Keras 3.
#     # 1. Update states (Builds the metrics)
#     update_compiled_metrics(model, targets=targets, y_pred = y_pred)
#     _add_compiled_results()

#     # ------------------------------------------------------------------
#     # Optional: log extra Q/subs-residual diagnostics
#     # ------------------------------------------------------------------
#     sk = getattr(model, "scaling_kwargs", None) or {}
#     log_q_diag = bool(get_sk(sk, "log_q_diagnostics", default=False))
    
#     def _add_metric_list(metrics):
#         for mm in metrics or []:
#             nm = getattr(mm, "name", "") or ""
#             if (not nm) or (nm in RESERVED) or (nm in EXCLUDE):
#                 continue
#             if nm in results:
#                 continue
    
#             # Keras 3: metric may exist but not yet built (no update_state called)
#             try:
#                 # If metric hasn't seen data, result() might fail or return 0
#                 results[nm] = mm.result()
#             except Exception:
#                 # never crash logging
#                 continue
    

#     # per-output loss trackers from compile(loss=...)
#     _add_metric_list(getattr(model, "metrics", []))

#     # Canonical loss fields (authoritative)
#     results["loss"] = total_loss
#     results["total_loss"] = total_loss
#     results["data_loss"] = data_loss

#     if manual_trackers:
#         for name, tracker in manual_trackers.items():
#             if name not in results:
#                 results[name] = safe_metric_result(tracker)
                
#     # ------------------------------------------------------------------
#     # 2) Physics logs (optional)
#     # ------------------------------------------------------------------
#     if not should_log_physics(model):
#         return results

#     if physics is None:
#         physics = zero_physics_bundle(model)

#     update_epsilon_metrics(
#         model,
#         eps_prior=physics["epsilon_prior"],
#         eps_cons=physics["epsilon_cons"],
#         eps_gw=physics["epsilon_gw"],
#     )

#     results.update({
#         "physics_loss": physics["physics_loss_raw"],
#         "physics_mult": physics["physics_mult"],
#         "physics_loss_scaled": physics["physics_loss_scaled"],
#         "lambda_offset": physics["lambda_offset"],

#         "consolidation_loss": physics["loss_consolidation"],
#         "gw_flow_loss": physics["loss_gw_flow"],
#         "prior_loss": physics["loss_prior"],
#         "smooth_loss": physics["loss_smooth"],
#         "mv_prior_loss": physics["loss_mv"],
#         "bounds_loss": physics["loss_bounds"],
#         "epsilon_prior": epsilon_value_for_logs(
#             model,
#             "prior",
#             physics["epsilon_prior"],
#         ),
#         "epsilon_cons": epsilon_value_for_logs(
#             model,
#             "cons",
#             physics["epsilon_cons"],
#         ),
#         "epsilon_gw": epsilon_value_for_logs(
#             model,
#             "gw",
#             physics["epsilon_gw"],
#         ),

#         "epsilon_cons_raw": physics["epsilon_cons_raw"],
#         "epsilon_gw_raw": physics["epsilon_gw_raw"],
#     })
    
#     if log_q_diag:
#         results.update({
#             "q_reg_loss": physics.get("loss_q_reg", tf_constant(0.0, tf_float32)),
#             "q_rms": physics.get("q_rms", tf_constant(0.0, tf_float32)),
#             "q_gate": physics.get("q_gate", tf_constant(0.0, tf_float32)),
#             "subs_resid_gate": physics.get("subs_resid_gate", tf_constant(0.0, tf_float32)),
#         })

#     return results
# def pack_step_results(
#     model: Any,
#     *,
#     total_loss: Tensor,
#     data_loss: Tensor,
#     targets: Any,
#     y_pred: Any,
#     physics: dict[str, Tensor] | None = None,
#     manual_trackers: dict | None = None,
# ) -> dict[str, Tensor]:

    # RESERVED = {"loss", "total_loss", "data_loss", "compile_metrics"}
    # EXCLUDE = {"epsilon_prior", "epsilon_cons", "epsilon_gw"}

    results: dict[str, Tensor] = {}

    # ----------------------------------------------------------
    # 0) Determine model output order (for multi-output).
    # ----------------------------------------------------------
    out_names = list(
        getattr(model, "output_names", None)
        or getattr(model, "_output_keys", None)
        or []
    )

    # ----------------------------------------------------------
    # 1) Ensure targets exist for every output.
    #
    # If a head is "loss-only" (no y_true provided), we fill
    # it as stop_gradient(y_pred) so:
    # - compiled multi-output loss dict doesn't crash
    # - no gradients flow for that head
    # ----------------------------------------------------------
    targets = ensure_targets_for_outputs(
        output_names=out_names,
        targets=targets,
        y_pred=y_pred,
        log_fn=getattr(model, "log_fn", None),
    )

    # ----------------------------------------------------------
    # 2) Update compiled metrics safely (Keras 2/3).
    #
    # This replaces any direct use of:
    #   model.compiled_metrics.update_state(...)
    # and any local routing logic.
    # ----------------------------------------------------------
    update_compiled_metrics(
        model,
        targets=targets,
        y_pred=y_pred,
    )

    # ----------------------------------------------------------
    # 3) Read compiled metrics results (Keras 2/3).
    # ----------------------------------------------------------
    cm = compiled_metrics_dict(model, dtype=tf_float32)
    for k, v in cm.items():
        if (not k) or (k in RESERVED) or (k in EXCLUDE):
            continue
        if k in results:
            continue
        results[k] = v

    # ----------------------------------------------------------
    # 4) Canonical loss fields (authoritative).
    # ----------------------------------------------------------
    results["loss"] = total_loss
    results["total_loss"] = total_loss
    results["data_loss"] = data_loss

    # ----------------------------------------------------------
    # 5) Optional: extra trackers not in compiled metrics.
    # ----------------------------------------------------------
    if manual_trackers:
        for name, tracker in manual_trackers.items():
            if name not in results:
                results[name] = safe_metric_result(tracker)

    # ----------------------------------------------------------
    # Optional: log extra Q/subs-residual diagnostics
    # (unchanged from your code)
    # ----------------------------------------------------------
    sk = getattr(model, "scaling_kwargs", None) or {}
    log_q_diag = bool(get_sk(
        sk,
        "log_q_diagnostics",
        default=False,
    ))

    # ----------------------------------------------------------
    # 6) Physics logs (unchanged from your code)
    # ----------------------------------------------------------
    if not should_log_physics(model):
        return results

    if physics is None:
        physics = zero_physics_bundle(model)

    update_epsilon_metrics(
        model,
        eps_prior=physics["epsilon_prior"],
        eps_cons=physics["epsilon_cons"],
        eps_gw=physics["epsilon_gw"],
    )

    results.update({
        "physics_loss": physics["physics_loss_raw"],
        "physics_mult": physics["physics_mult"],
        "physics_loss_scaled": physics["physics_loss_scaled"],
        "lambda_offset": physics["lambda_offset"],

        "consolidation_loss": physics["loss_consolidation"],
        "gw_flow_loss": physics["loss_gw_flow"],
        "prior_loss": physics["loss_prior"],
        "smooth_loss": physics["loss_smooth"],
        "mv_prior_loss": physics["loss_mv"],
        "bounds_loss": physics["loss_bounds"],

        "epsilon_prior": epsilon_value_for_logs(
            model,
            "prior",
            physics["epsilon_prior"],
        ),
        "epsilon_cons": epsilon_value_for_logs(
            model,
            "cons",
            physics["epsilon_cons"],
        ),
        "epsilon_gw": epsilon_value_for_logs(
            model,
            "gw",
            physics["epsilon_gw"],
        ),

        "epsilon_cons_raw": physics["epsilon_cons_raw"],
        "epsilon_gw_raw": physics["epsilon_gw_raw"],
    })

    if log_q_diag:
        results.update({
            "q_reg_loss": physics.get(
                "loss_q_reg",
                tf_constant(0.0, tf_float32),
            ),
            "q_rms": physics.get(
                "q_rms",
                tf_constant(0.0, tf_float32),
            ),
            "q_gate": physics.get(
                "q_gate",
                tf_constant(0.0, tf_float32),
            ),
            "subs_resid_gate": physics.get(
                "subs_resid_gate",
                tf_constant(0.0, tf_float32),
            ),
        })

    return results

# ---------------------------------------------------------------------
# Eval packer (for _evaluate_physics_on_batch)
# ---------------------------------------------------------------------
def pack_eval_physics(
    model: Any,
    *,
    physics: dict[str, Tensor] | None,
) -> dict[str, Tensor]:
    r"""
    Canonical physics bundle output for batch-level physics evaluation.
    
    This helper normalizes the output of physics diagnostics so that
    callers can rely on a stable schema regardless of whether physics is
    enabled for the model.
    
    Behavior:
    
    * If a physics bundle is provided, it is returned unchanged.
    * If physics is off and logging is enabled, a zero-valued physics
      bundle is returned (to keep downstream logging stable).
    * If physics is off and logging is disabled, an empty dict is
      returned.
    
    Parameters
    ----------
    model : Any
        Model-like object that controls whether physics logging is enabled.
        This function relies on ``should_log_physics(model)`` and
        ``zero_physics_bundle(model)`` which are expected to be available
        in the surrounding module.
    
    physics : dict[str, Tensor] or None
        Physics bundle produced by ``physics_core`` or a compatible
        routine. If None, behavior depends on whether physics logging is
        enabled.
    
    Returns
    -------
    out : dict[str, Tensor]
        Canonical physics dictionary.
    
        If physics is enabled (or logging when off), keys typically include
        (implementation dependent):
    
        * ``physics_loss_raw``
        * ``physics_loss_scaled``
        * ``physics_mult``
        * per-term losses and epsilon diagnostics
    
        If physics is off and logging is disabled, returns ``{}``.
    
    Notes
    -----
    Stable logging schema
    ~~~~~~~~~~~~~~~~~~~~
    Returning a zero bundle when physics is off is useful for dashboards
    and automated training loops where missing keys complicate aggregation.
    
    Examples
    --------
    Batch-level evaluation:
    
    >>> packed = pack_eval_physics(model, physics=physics_bundle)
    
    Physics-off scenario:
    
    >>> packed = pack_eval_physics(model, physics=None)
    >>> packed  # either {} or a zero bundle depending on settings
    
    See Also
    --------
    GeoPriorSubsNet.evaluate_physics
        Aggregates these batch outputs across datasets.
    
    physics_core
        Produces the physics bundle consumed by this helper.
    """

    if physics is None:
        if should_log_physics(model):
            return zero_physics_bundle(model)
        return {}

    return physics

