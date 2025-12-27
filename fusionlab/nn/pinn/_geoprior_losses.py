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

from .. import KERAS_DEPS
from ._geoprior_utils import get_sk


Tensor = KERAS_DEPS.Tensor
tf_float32 = KERAS_DEPS.float32
tf_constant = KERAS_DEPS.constant
tf_identity = KERAS_DEPS.identity


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
    loss_bounds: Tensor,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """
    Assemble physics loss in one canonical place.

    Returns
    -------
    physics_raw : Tensor
        Weighted sum of component losses (no offset).
    physics_scaled : Tensor
        physics_raw multiplied once by global multiplier.
    phys_mult : Tensor
        Effective global multiplier (offset_mode aware).
    terms_scaled : dict
        Per-term contributions (each includes phys_mult once).
    """
    physics_raw = (
        model.lambda_cons * loss_cons
        + model.lambda_gw * loss_gw
        + model.lambda_prior * loss_prior
        + model.lambda_smooth * loss_smooth
        + model.lambda_mv * loss_mv
        + model.lambda_bounds * loss_bounds
    )

    phys_mult = model._physics_loss_multiplier()
    physics_scaled = phys_mult * physics_raw
    
    # Optional: per-term scaled contributions for debug/logging
    terms_scaled = {
        "cons":   phys_mult * model.lambda_cons   * loss_cons,
        "gw":     phys_mult * model.lambda_gw     * loss_gw,
        "prior":  phys_mult * model.lambda_prior  * loss_prior,
        "smooth": phys_mult * model.lambda_smooth * loss_smooth,
        "mv":     phys_mult * model.lambda_mv     * loss_mv,
        "bounds": phys_mult * model.lambda_bounds * loss_bounds,
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


def epsilon_value_for_logs(
    model: Any,
    which: str,
    fallback: Tensor,
) -> Tensor:
    """
    Prefer tracked epsilon metric if it exists.
    """
    if which == "prior":
        m = getattr(model, "eps_prior_metric", None)
        if m is not None:
            return m.result()

    if which == "cons":
        m = getattr(model, "eps_cons_metric", None)
        if m is not None:
            return m.result()

    if which == "gw":
        m = getattr(model, "eps_gw_metric", None)
        if m is not None:
            return m.result()

    return fallback


# ---------------------------------------------------------------------
# Train/Test step packer (no duplication)
# ---------------------------------------------------------------------
def pack_step_results(
    model: Any,
    *,
    total_loss: Tensor,
    data_loss: Tensor,
    targets: Any,
    y_pred: Any,
    physics: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """
    Canonical return dict for train_step/test_step.

    - Always includes loss/total_loss/data_loss + compiled metrics.
    - Includes physics keys only if `should_log_physics(model)`.
    """
    model.compiled_metrics.update_state(targets, y_pred)

    results: dict[str, Tensor] = {
        m.name: m.result()
        for m in model.metrics
        if m.name
        not in ("epsilon_prior", "epsilon_cons", "epsilon_gw")
    }
    results.update({
        "loss": total_loss,
        "total_loss": total_loss,
        "data_loss": data_loss,
    })

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
    return results

# ---------------------------------------------------------------------
# Eval packer (for _evaluate_physics_on_batch)
# ---------------------------------------------------------------------
def pack_eval_physics(
    model: Any,
    *,
    physics: dict[str, Tensor] | None,
) -> dict[str, Tensor]:
    """
    Canonical output for evaluate_physics_on_batch.

    If physics is off:
    - return zeros if `log_physics_when_off` is True
    - else return an empty dict
    """
    if physics is None:
        if should_log_physics(model):
            return zero_physics_bundle(model)
        return {}

    return physics
