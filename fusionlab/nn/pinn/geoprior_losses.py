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
from ._geoprior_utils import get_sk, select_q  


Tensor = KERAS_DEPS.Tensor
tf_float32 = KERAS_DEPS.float32
tf_constant = KERAS_DEPS.constant
tf_identity = KERAS_DEPS.identity
Tensor = KERAS_DEPS.Tensor
tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor

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
    """
    Assemble physics loss with an explicit offset scope.

    Notes
    -----
    By default, the global physics multiplier `phys_mult`
    (derived from `lambda_offset`) scales PDE-style terms
    (cons, gw, prior, smooth, bounds). The mv term is treated
    as a calibration loss and is *not* scaled by `phys_mult`
    unless `model._scale_mv_with_offset` is True.
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

# def update_compiled_metrics(model, targets, y_pred):
#     """
#     Update compiled metrics without relying on Keras structure-matching.

#     Works even if model.call() returns a dict unrelated to (subs_pred, gwl_pred).
#     Assumes `targets` and `y_pred` are dicts with keys:
#       - "subs_pred"
#       - "gwl_pred"
#     """
#     if not isinstance(targets, dict) or not isinstance(y_pred, dict):
#         return

#     # metrics = getattr(model.compiled_metrics, "metrics", []) or []
#     # Keras 3: model.metrics contains all instantiated metrics
#     metrics = getattr(model, "metrics", []) or []
#     for m in metrics:
#         name = getattr(m, "name", "") or ""

#         if name.startswith("subs_pred"):
#             key = "subs_pred"
#         elif name.startswith("gwl_pred"):
#             key = "gwl_pred"
#         else:
#             continue

#         yt = targets.get(key, None)
#         yp = y_pred.get(key, None)
#         if yt is None or yp is None:
#             continue

#         try:
#             m.update_state(yt, yp)
#         except TypeError:
#             # defensive: some custom metrics might be single-arg
#             m.update_state(yt)

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

# def epsilon_value_for_logs(
#     model: Any,
#     which: str,
#     fallback: Tensor,
# ) -> Tensor:
#     """
#     Prefer tracked epsilon metric if it exists.
#     """
#     if which == "prior":
#         m = getattr(model, "eps_prior_metric", None)
#         if m is not None:
#             return _set_metric_results(m, fallback)

#     if which == "cons":
#         m = getattr(model, "eps_cons_metric", None)
#         if m is not None:
#             return _set_metric_results(m, fallback)

#     if which == "gw":
#         m = getattr(model, "eps_gw_metric", None)
#         if m is not None:
#             return _set_metric_results(m, fallback)

#     return fallback


# ---------------------------------------------------------------------
# Train/Test step packer (no duplication)
# ---------------------------------------------------------------------
def update_compiled_metrics(model, targets, y_pred):
    """
    Update compiled metrics explicitly for Keras 3 compatibility.
    Handles 'lazy building' by ensuring all relevant metrics see data.
    """
    if not isinstance(targets, dict) or not isinstance(y_pred, dict):
        return

    # Keras 3: model.metrics contains all instantiated metrics
    metrics = getattr(model, "metrics", []) or []
    
    for m in metrics:
        name = getattr(m, "name", "") or ""

        # Skip loss trackers
        if name == "loss" or name.endswith("_loss"):
            continue

        # --- MATCHING LOGIC ---
        # Keras 3 prefixes metric names with output names.
        # e.g. "subs_pred_coverage80", "gwl_pred_mse"
        key = None
        
        # Check explicit output names first
        if "subs_pred" in name:
            key = "subs_pred"
        elif "gwl_pred" in name:
            key = "gwl_pred"
        
        if key is None:
            # Fallback: if user didn't use prefixes but we have exact names
            if name in ("coverage80", "sharpness80", "mae", "mse"):
                # Ambiguous if we have multiple outputs. 
                # Usually we assume the first output or try both.
                # Here we skip ambiguous to avoid crashing.
                continue

        if key is not None:
            yt = targets.get(key, None)
            yp = y_pred.get(key, None)
            
            if yt is not None and yp is not None:
                try:
                    # This call BUILDS the metric if it wasn't built yet
                    m.update_state(yt, yp)
                except Exception:
                    pass
                
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
    """
    # ------------------------------------------------------------------
    # 0) Update compiled metrics (MANUAL UPDATE for Keras 3)
    # ------------------------------------------------------------------
    # We DO NOT use model.compiled_metrics.update_state(targets, y_pred)
    # because it crashes with TypeError on dicts in Keras 3.
    # 1. Update states (Builds the metrics)
    update_compiled_metrics(model, targets, y_pred)
    
    # ------------------------------------------------------------------
    # Optional: log extra Q/subs-residual diagnostics
    # ------------------------------------------------------------------
    sk = getattr(model, "scaling_kwargs", None) or {}
    log_q_diag = bool(get_sk(sk, "log_q_diagnostics", default=False))
    
    # ------------------------------------------------------------------
    # 1) Collect logs (DO NOT rely on model.metrics only)
    # ------------------------------------------------------------------
    results: dict[str, Tensor] = {}

    RESERVED = {"loss", "total_loss", "data_loss"}
    EXCLUDE = {"epsilon_prior", "epsilon_cons", "epsilon_gw"}

    def _add_metric_list(metrics):
        for mm in metrics or []:
            nm = getattr(mm, "name", "") or ""
            if (not nm) or (nm in RESERVED) or (nm in EXCLUDE):
                continue
            if nm in results:
                continue
    
            # Keras 3: metric may exist but not yet built (no update_state called)
            try:
                # If metric hasn't seen data, result() might fail or return 0
                results[nm] = mm.result()
            except Exception:
                # never crash logging
                continue
    

    # per-output loss trackers from compile(loss=...)
    _add_metric_list(getattr(model, "metrics", []))

    # Canonical loss fields (authoritative)
    results["loss"] = total_loss
    results["total_loss"] = total_loss
    results["data_loss"] = data_loss

    # ------------------------------------------------------------------
    # 2) Physics logs (optional)
    # ------------------------------------------------------------------
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
            "q_reg_loss": physics.get("loss_q_reg", tf_constant(0.0, tf_float32)),
            "q_rms": physics.get("q_rms", tf_constant(0.0, tf_float32)),
            "q_gate": physics.get("q_gate", tf_constant(0.0, tf_float32)),
            "subs_resid_gate": physics.get("subs_resid_gate", tf_constant(0.0, tf_float32)),
        })

    return results

# def pack_step_results(
#     model: Any,
#     *,
#     total_loss: Tensor,
#     data_loss: Tensor,
#     targets: Any,
#     y_pred: Any,
#     physics: dict[str, Tensor] | None = None,
# ) -> dict[str, Tensor]:
#     """
#     Canonical return dict for train_step/test_step.

#     - Always returns:
#         loss, total_loss, data_loss
#         + compiled loss trackers (e.g., subs_pred_loss, gwl_pred_loss)
#         + compiled metrics (e.g., mae/mse/coverage/sharpness)
#     - Adds physics keys only if `should_log_physics(model)`.
#     """
#     # ------------------------------------------------------------------
#     # 0) Update compiled metrics (quantile-safe)
#     # ------------------------------------------------------------------
    
#     # q = getattr(model, "quantiles", None)

#     # if isinstance(targets, dict) and isinstance(y_pred, dict):
#     #     # Lazy p50 cache (only computed for point-metrics)
#     #     y_pred_p50: dict[str, Tensor] = {}

#     #     def _get_p50(key: str) -> Tensor:
#     #         if key not in y_pred_p50:  # full (B,H,Q,1)
#     #             y_pred_p50[key] = select_q(
#     #                 y_pred[key],
#     #                 quantiles=q,
#     #                 q=0.5,
#     #                 fallback="mean",
#     #             )
#     #         return y_pred_p50[key] # p50 (B,H,1)

#     #     for m in getattr(model.compiled_metrics, "metrics", []):
#     #         mname = getattr(m, "name", "") or ""
#     #         key = _metric_key_from_name(mname)
#     #         if key is None:
#     #             continue

#     #         yt = targets.get(key, None)
#     #         yp_full = y_pred.get(key, None)
#     #         if (yt is None) or (yp_full is None):
#     #             continue

#     #         yp = yp_full if _needs_full_quantiles(mname) else _get_p50(key)
#     #         m.update_state(yt, yp)

#     # else:
#     #     # Fallback: if caller didn't use dict outputs, defer to Keras
#     #     try:
#     #        
#     #     except Exception:
#     #         pass
#     safe_update_compiled_metrics(model, targets, y_pred)
#     # ------------------------------------------------------------------
#     # Optional: log extra Q/subs-residual diagnostics
#     # ------------------------------------------------------------------
#     sk = getattr(model, "scaling_kwargs", None) or {}
#     log_q_diag = bool(get_sk(sk, "log_q_diagnostics", default=False))
    
#     # ------------------------------------------------------------------
#     # 1) Collect logs (DO NOT rely on model.metrics only)
#     # ------------------------------------------------------------------
#     results: dict[str, Tensor] = {}

#     RESERVED = {"loss", "total_loss", "data_loss"}
#     EXCLUDE = {"epsilon_prior", "epsilon_cons", "epsilon_gw"}

#     # def _add_metric_list(metrics):
#     #     for mm in metrics or []:
#     #         nm = getattr(mm, "name", "") or ""
#     #         if (not nm) or (nm in RESERVED) or (nm in EXCLUDE):
#     #             continue
#     #         if nm in results:
#     #             continue
#     #         results[nm] = mm.result()

#     def _add_metric_list(metrics):
#         for mm in metrics or []:
#             nm = getattr(mm, "name", "") or ""
#             if (not nm) or (nm in RESERVED) or (nm in EXCLUDE):
#                 continue
#             if nm in results:
#                 continue
    
#             # Keras 3: metric may exist but not yet built (no update_state called)
#             try:
#                 if getattr(mm, "built", True) is False:
#                     continue
#                 results[nm] = mm.result()
#             except Exception:
#                 # never crash logging
#                 continue
    

#     # per-output loss trackers from compile(loss=...)
#     _add_metric_list(getattr(model.compiled_loss, "metrics", []))
#     # compile(metrics=...)
#     _add_metric_list(getattr(model.compiled_metrics, "metrics", []))
#     # any other custom trackers attached to model (rare but possible)
#     # _add_metric_list(getattr(model, "metrics", []))

#     # Canonical loss fields (authoritative)
#     results["loss"] = total_loss
#     results["total_loss"] = total_loss
#     results["data_loss"] = data_loss

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

