# -*- coding: utf-8 -*-
# _geoprior_losses.py

"""
GeoPrior loss assembly and logging helpers.

Centralizes:
- physics loss assembly (no double offset)
- return packaging for train/test/eval
- Keras 3 safe metric handling (metrics may be "not built" early)
"""

from __future__ import annotations

from typing import Any

from .. import KERAS_DEPS
from ._geoprior_utils import get_sk

Tensor = KERAS_DEPS.Tensor
tf_float32 = KERAS_DEPS.float32
tf_constant = KERAS_DEPS.constant
tf_identity = KERAS_DEPS.identity
tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor
tf_reduce_mean =KERAS_DEPS.reduce_mean 
tf_cast = KERAS_DEPS.cast 
tf_add_n = KERAS_DEPS.add_n 

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


# ---------------------------------------------------------------------
# Small switches
# ---------------------------------------------------------------------
def should_log_physics(model: Any) -> bool:
    """
    Decide whether to expose physics keys in logs.

    Behavior:
    - If model has no `_physics_off()` hook => True
    - If physics is ON => True
    - If physics is OFF => scaling_kwargs["log_physics_when_off"] controls
    """
    sk = getattr(model, "scaling_kwargs", None) or {}

    hook = getattr(model, "_physics_off", None)
    if hook is None:
        return True

    try:
        if not hook():
            return True
    except Exception:
        # If hook misbehaves, default to logging physics to avoid silent loss.
        return True

    return bool(get_sk(sk, "log_physics_when_off", default=False))


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
    """
    Assemble physics loss with an explicit offset scope.
    Returns raw/scaled totals, multiplier, and per-term contributions.
    """
    # 1) Unscaled weighted terms.
    t_cons = model.lambda_cons * loss_cons
    t_gw = model.lambda_gw * loss_gw
    t_prior = model.lambda_prior * loss_prior
    t_smooth = model.lambda_smooth * loss_smooth
    t_bounds = model.lambda_bounds * loss_bounds
    t_mv = model.lambda_mv * loss_mv
    t_q = model.lambda_q * loss_q_reg

    core_raw = t_cons + t_gw + t_prior + t_smooth + t_bounds
    physics_raw = core_raw + t_mv + t_q

    # 2) Global multiplier (offset_mode aware).
    phys_mult = model._physics_loss_multiplier()

    scale_mv = bool(getattr(model, "_scale_mv_with_offset", False))
    scale_q = bool(getattr(model, "_scale_q_with_offset", False))

    core_scaled = phys_mult * core_raw
    mv_scaled = phys_mult * t_mv if scale_mv else t_mv
    q_scaled = phys_mult * t_q if scale_q else t_q

    physics_scaled = core_scaled + mv_scaled + q_scaled

    # 3) Per-term contributions consistent with physics_scaled.
    terms_scaled = {
        "cons": phys_mult * t_cons,
        "gw": phys_mult * t_gw,
        "prior": phys_mult * t_prior,
        "smooth": phys_mult * t_smooth,
        "bounds": phys_mult * t_bounds,
        "mv": mv_scaled,
        "q": q_scaled,
    }
    return physics_raw, physics_scaled, phys_mult, terms_scaled


# ---------------------------------------------------------------------
# Physics bundles
# ---------------------------------------------------------------------
def zero_physics_bundle(model: Any, *, dtype: Any = tf_float32) -> dict[str, Tensor]:
    """Canonical zero physics bundle."""
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
    """Canonical physics bundle used by train/test/eval packers."""
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
        "epsilon_cons_raw": (eps_cons_raw if eps_cons_raw is not None else z),
        "epsilon_gw_raw": (eps_gw_raw if eps_gw_raw is not None else z),
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
    """Update optional epsilon metrics if present."""
    for name, val in [
        ("eps_prior_metric", eps_prior),
        ("eps_cons_metric", eps_cons),
        ("eps_gw_metric", eps_gw),
    ]:
        m = getattr(model, name, None)
        if m is not None:
            m.update_state(val)


def epsilon_value_for_logs(model: Any, which: str, fallback: Tensor) -> Tensor:
    """Prefer tracked epsilon metric if it exists."""
    key = f"eps_{which}_metric"
    m = getattr(model, key, None)
    if m is not None:
        return safe_metric_result(m, fallback=0.0)
    return fallback


# ---------------------------------------------------------------------
# Metric Update Logic (Keras 3-safe)
# ---------------------------------------------------------------------
def _infer_output_names(model: Any, targets: dict, y_pred: dict) -> list[str]:
    """
    Infer a stable output ordering for CompileMetrics.update_state.

    Priority:
    1) model._output_keys (your internal convention)
    2) model.output_names (if present)
    3) y_pred insertion order
    4) common canonical order if present
    """
    out_names = getattr(model, "_output_keys", None) or getattr(model, "output_names", None)

    if isinstance(out_names, str):
        out_names = [out_names]
    if not out_names:
        out_names = list(y_pred.keys())

    # canonical fallback ordering (only used if those keys exist)
    if not out_names:
        out_names = [ "subs_pred", "gwl_pred"]

    # Keep only names that exist in BOTH dicts and are not None.
    out_names = [
        n for n in out_names
        if (n in targets and n in y_pred and targets[n] is not None and y_pred[n] is not None)
    ]
    return out_names


def _strip_tiled_true(yt: Any) -> Any:
    """If y_true got tiled to (B,H,Q,O), strip to (B,H,O)."""
    if yt is None:
        return None
    if getattr(yt, "shape", None) is not None and yt.shape.rank == 4:
        return yt[:, :, 0, :]
    return yt


def update_compiled_metrics(model: Any, targets: Any, y_pred: Any, sample_weight: Any = None) -> None:
    """
    Manually update compiled metrics in a Keras 3-safe way.

    Notes
    -----
    - Expects dict targets/preds for multi-output models.
    - Uses positional structure ([y_true_i], [y_pred_i]) for Keras 3 stability.
    """
    if not isinstance(targets, dict) or not isinstance(y_pred, dict):
        return

    cm = getattr(model, "compiled_metrics", None)
    if cm is None:
        return

    # Strip tiled y_true if needed
    targets2 = dict(targets)
    for k in ( "subs_pred", "gwl_pred"):
        if k in targets2:
            targets2[k] = _strip_tiled_true(targets2[k])

    out_names = _infer_output_names(model, targets2, y_pred)
    if not out_names:
        return

    y_true_struct = [targets2[n] for n in out_names]
    y_pred_struct = [y_pred[n] for n in out_names]

    sw = sample_weight
    if isinstance(sample_weight, dict):
        sw = [sample_weight.get(n, None) for n in out_names]

    try:
        if sw is None:
            cm.update_state(y_true_struct, y_pred_struct)
        else:
            cm.update_state(y_true_struct, y_pred_struct, sample_weight=sw)
    except TypeError:
        # Some containers accept sample_weight as positional
        if sw is None:
            cm.update_state(y_true_struct, y_pred_struct)
        else:
            cm.update_state(y_true_struct, y_pred_struct, sw)


def _iter_leaf_metrics(model: Any):
    """Yield leaf metric objects from model.metrics and compiled_metrics.metrics."""
    seen = set()
    stack = []
    stack.extend(list(getattr(model, "metrics", []) or []))

    cm = getattr(model, "compiled_metrics", None)
    if cm is not None:
        stack.extend(list(getattr(cm, "metrics", []) or []))

    while stack:
        m = stack.pop(0)
        if m is None:
            continue
        mid = id(m)
        if mid in seen:
            continue
        seen.add(mid)

        children = getattr(m, "metrics", None)
        if children:
            stack.extend(list(children))
            continue

        yield m


def pack_step_results(
    model: Any,
    *,
    total_loss: Tensor,
    data_loss: Tensor,
    targets: Any,
    y_pred: Any,
    physics: dict[str, Tensor] | None = None,
    manual_trackers: dict | None = None,
    sample_weight: Any = None,
) -> dict[str, Tensor]:
    """
    Canonical return dict for train_step/test_step.

    - Updates compiled metrics manually (Keras 3 safe)
    - Collects metric results via safe_metric_result()
    - Optionally merges physics logs and q diagnostics
    """
    # 0) Update compiled metrics (manual: Keras 3)
    update_compiled_metrics(model, targets, y_pred, sample_weight=sample_weight)

    # 1) Base logs
    results: dict[str, Tensor] = {
        "loss": total_loss,
        "total_loss": total_loss,
        "data_loss": data_loss,
    }

    sk = getattr(model, "scaling_kwargs", None) or {}
    log_q_diag = bool(get_sk(sk, "log_q_diagnostics", default=False))

    RESERVED = {"loss", "total_loss", "data_loss"}
    EXCLUDE = {"epsilon_prior", "epsilon_cons", "epsilon_gw"}  # added later (physics block)

    for mm in _iter_leaf_metrics(model):
        nm = getattr(mm, "name", "") or ""
        if (not nm) or (nm in RESERVED) or (nm in EXCLUDE):
            continue
        if nm in results:
            continue
        results[nm] = safe_metric_result(mm)

    if manual_trackers:
        for name, tracker in manual_trackers.items():
            if name not in results:
                results[name] = safe_metric_result(tracker)

    # 2) Physics logs (optional)
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

        "epsilon_prior": epsilon_value_for_logs(model, "prior", physics["epsilon_prior"]),
        "epsilon_cons": epsilon_value_for_logs(model, "cons", physics["epsilon_cons"]),
        "epsilon_gw": epsilon_value_for_logs(model, "gw", physics["epsilon_gw"]),

        "epsilon_cons_raw": physics["epsilon_cons_raw"],
        "epsilon_gw_raw": physics["epsilon_gw_raw"],
    })

    if log_q_diag:
        results.update({
            "q_reg_loss": physics.get("loss_q_reg", tf_constant(0.0, tf_float32)),
            "q_rms": physics.get("q_rms", tf_constant(0.0, tf_float32)),
            "q_gate": physics.get("q_gate", tf_constant(0.0, tf_float32)),
            "subs_resid_gate": physics.get("subs_resid_gate", tf_constant(0.0, tf_float32)),
        })

    return results


# ---------------------------------------------------------------------
# Eval packer (for _evaluate_physics_on_batch)
# ---------------------------------------------------------------------
def pack_eval_physics(model: Any, *, physics: dict[str, Tensor] | None) -> dict[str, Tensor]:
    """Canonical output for evaluate_physics_on_batch."""
    if physics is None:
        return zero_physics_bundle(model) if should_log_physics(model) else {}
    return physics

def _safe_compiled_data_loss(
    self,
    targets: dict[str, Tensor],
    y_pred: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute supervised data loss robustly across Keras versions.

    In some Keras 3 / subclassed-model edge cases, `self.compiled_loss`
    may be initialized in "single-output" mode (loss path `()`), which
    raises a KeyError when `y_pred` is a dict. We fall back to a manual
    per-output loss aggregation that respects `loss_weights` and adds
    `regularization_losses`.
    """
    try:
        data_loss = self.compiled_loss(
            targets,
            y_pred,
            regularization_losses=self.losses,
        )
        return data_loss, {}
    except KeyError as e:
        # Only fall back for the known "loss path ()" mismatch.
        if "path: ()" not in str(e):
            raise
        return self._manual_data_loss(targets, y_pred)

def _manual_data_loss(
    self,
    targets: dict[str, Tensor],
    y_pred: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Manual supervised loss: sum_i w_i * loss_i(y_i, ŷ_i)."""
    from ._geoprior_losses import _strip_tiled_true

    loss_cfg = self.loss
    loss_w = getattr(self, "loss_weights", None)

    out_losses: dict[str, Tensor] = {}
    total = tf_constant(0.0, tf_float32)

    # Helper to safely read mappings.
    def _get(mapping, key, default=None):
        try:
            return mapping.get(key, default)
        except Exception:
            return default

    for i, name in enumerate(self.output_names):
        fn = _get(loss_cfg, name) if hasattr(loss_cfg, "get") else loss_cfg
        if fn is None:
            continue

        yt = _get(targets, name) if hasattr(targets, "get") else None
        yp = _get(y_pred, name) if hasattr(y_pred, "get") else None
        if yt is None or yp is None:
            continue

        # If y_true was tiled to match quantile dimensions, strip it.
        yt_use = _strip_tiled_true(yt)

        l = fn(yt_use, yp)
        l = tf_reduce_mean(l)

        w = 1.0
        if isinstance(loss_w, dict):
            w = float(loss_w.get(name, 1.0))
        elif isinstance(loss_w, (list, tuple)):
            w = float(loss_w[i]) if i < len(loss_w) else 1.0
        elif loss_w is not None:
            try:
                w = float(loss_w)
            except Exception:
                w = 1.0

        out_losses[f"{name}_loss"] = l
        total = total + tf_cast(w, tf_float32) * tf_cast(l, tf_float32)

    # Add regularization losses if any.
    if self.losses:
        total = total + tf_add_n([tf_cast(x, tf_float32) for x in self.losses])

    return total, out_losses