# -*- coding: utf-8 -*-
# .py
"""Debug helpers for GeoPriorSubsNet.

Keep *all* verbosity + shape/unit printing here so
`_geoprior_subnet.py` stays clean.

All functions are safe to call inside `tf.function`:
they use `tf.print` and TensorFlow assertions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .. import KERAS_DEPS

from ._geoprior_utils import _vshape, _vshapes
from ._geoprior_maths import (
    tf_print_nonfinite,
    _assert_grads_finite,
    resolve_cons_units,
    to_rms, _stats, 
    seconds_per_time_unit,
    dt_to_seconds,

)

# ---------------------------------------------------------------------
# TF aliases (keep local + short)
# ---------------------------------------------------------------------
Tensor = KERAS_DEPS.Tensor

tf_abs = KERAS_DEPS.abs
tf_cast = KERAS_DEPS.cast
tf_cond = KERAS_DEPS.cond
tf_constant = KERAS_DEPS.constant
tf_debugging = KERAS_DEPS.debugging
tf_equal = KERAS_DEPS.equal
tf_float32 = KERAS_DEPS.float32
tf_greater = KERAS_DEPS.greater
tf_int32 = KERAS_DEPS.int32
tf_print = KERAS_DEPS.print
tf_rank = KERAS_DEPS.rank
tf_reduce_max = KERAS_DEPS.reduce_max
tf_reduce_mean = KERAS_DEPS.reduce_mean
tf_reduce_min = KERAS_DEPS.reduce_min
tf_shape = KERAS_DEPS.shape
tf_square = KERAS_DEPS.square
tf_where = KERAS_DEPS.where
tf_zeros_like = KERAS_DEPS.zeros_like
tf_math =KERAS_DEPS.math 
tf_sqrt = KERAS_DEPS.sqrt 
tf_maximum = KERAS_DEPS.maximum
tf_string = KERAS_DEPS.string
# ---------------------------------------------------------------------
# Small gates
# ---------------------------------------------------------------------
def dbg_on(verbose: int, level: int) -> bool:
    """Return True if verbose is strictly above level."""
    return int(verbose) > int(level)


def dbg_run_first_iter(
    *,
    verbose: int,
    level: int,
    iterations: Tensor,
    fn,
) -> None:
    """
    Run fn() only at optimizer.iterations == 0.

    Graph-safe: uses tf.cond and returns a dummy scalar.
    """
    if not dbg_on(verbose, level):
        return

    it = tf_cast(iterations, tf_int32)
    z = tf_constant(0, tf_int32)

    def _do():
        fn()
        return z

    tf_cond(tf_equal(it, 0), _do, lambda: z)

# ---------------------------------------------------------------------
# Basic stats helpers
# ---------------------------------------------------------------------
def dbg_stats(tag: str, x: Tensor) -> None:
    """Print min/max/mean (graph-safe)."""
    tf_print(
        tag,
        "min/max/mean=",
        tf_reduce_min(x),
        tf_reduce_max(x),
        tf_reduce_mean(x),
    )


def dbg_pde_divergence_maxabs(
    *,
    verbose: int,
    raw_dKdhx_dcoords: Tensor,
    raw_d_K_dh_dx_dx: Tensor,
    raw_d_K_dh_dy_dy: Tensor,
    d_K_dh_dx_dx: Optional[Tensor] = None,
    d_K_dh_dy_dy: Optional[Tensor] = None,
    level: int = 8,
    prefix: str = "pde/div",
) -> None:
    """
    Print max-abs diagnostics for the divergence terms, before and
    optionally after normalization/chain-rule correction.

    Usage
    -----
    # before normalization only:
    dbg_pde_divergence_maxabs(
        verbose=verbose,
        raw_dKdhx_dcoords=dKdhx_dcoords,
        raw_d_K_dh_dx_dx=d_K_dh_dx_dx_raw,
        raw_d_K_dh_dy_dy=d_K_dh_dy_dy_raw,
    )

    # after normalization too:
    dbg_pde_divergence_maxabs(
        verbose=verbose,
        raw_dKdhx_dcoords=dKdhx_dcoords,
        raw_d_K_dh_dx_dx=d_K_dh_dx_dx_raw,
        raw_d_K_dh_dy_dy=d_K_dh_dy_dy_raw,
        d_K_dh_dx_dx=d_K_dh_dx_dx,
        d_K_dh_dy_dy=d_K_dh_dy_dy,
    )
    """
    if not dbg_on(verbose, level):
        return

    tf_print(f"[{prefix}] before normalization:")
    tf_print(
        "max|dKdhx_dcoords|=",
        tf_reduce_max(tf_abs(raw_dKdhx_dcoords)),
    )
    tf_print(
        "max|d_K_dh_dx_dx_raw|=",
        tf_reduce_max(tf_abs(raw_d_K_dh_dx_dx)),
    )
    tf_print(
        "max|d_K_dh_dy_dy_raw|=",
        tf_reduce_max(tf_abs(raw_d_K_dh_dy_dy)),
    )

    if (d_K_dh_dx_dx is None) or (d_K_dh_dy_dy is None):
        return

    tf_print(f"[{prefix}] after normalization:")
    tf_print(
        "max|d_K_dh_dx_dx|=",
        tf_reduce_max(tf_abs(d_K_dh_dx_dx)),
    )
    tf_print(
        "max|d_K_dh_dy_dy|=",
        tf_reduce_max(tf_abs(d_K_dh_dy_dy)),
    )


def dbg_gw_units_and_sec_scale(
    *,
    verbose: int,
    gw_units: Any,
    gw_res_before: Tensor,
    gw_res_after: Tensor,
    level: int = 7,
    prefix: str = "gw/units",
) -> None:
    """
    Print GW residual diagnostics before/after applying sec_u scaling.

    Call this right after you do:
        gw_res_before = gw_res
        gw_res = gw_res * sec_u
        gw_res_after = gw_res

    Parameters
    ----------
    gw_units:
        Usually resolve_gw_units(sk). Keep it as Any so callers can
        pass python strings without TF ops.

    Notes
    -----
    We print RMS to catch accidental unit explosions.
    """
    if not dbg_on(verbose, level):
        return

    tf_print(
        f"[{prefix}] resolve_gw_units(sk)=",
        gw_units,
    )
    tf_print(
        f"[{prefix}] gw_res BEFORE sec_u: RMS=",
        to_rms(gw_res_before),
        "| max|.|=",
        tf_reduce_max(tf_abs(gw_res_before)),
    )
    tf_print(
        f"[{prefix}] gw_res AFTER  sec_u: RMS=",
        to_rms(gw_res_after),
        "| max|.|=",
        tf_reduce_max(tf_abs(gw_res_after)),
    )

    # Optional: if you want to catch NaNs early (cheap)
    tf_print_nonfinite(f"{prefix}/gw_res_before", gw_res_before)
    tf_print_nonfinite(f"{prefix}/gw_res_after", gw_res_after)


def dbg_mae(tag: str, y: Tensor, yhat: Tensor) -> None:
    """Print batch MAE for y vs yhat."""
    tf_print(tag, "mae(batch)=", tf_reduce_mean(tf_abs(y - yhat)))


def dbg_chk_finite(tag: str, x: Tensor) -> Tensor:
    """Assert finite, return x (small helper)."""
    tf_debugging.assert_all_finite(x, f"{tag} has NaN/Inf")
    return x


def _median_index(quantiles: Optional[Sequence[float]]) -> Optional[int]:
    if quantiles is None:
        return None
    qs = list(quantiles)
    try:
        return int(qs.index(0.5))
    except ValueError:
        return int(len(qs) // 2)


def _pick_q50(
    y_pred: Tensor,
    quantiles: Optional[Sequence[float]],
) -> Tensor:
    """Pick the median quantile if `y_pred` is quantile-aware.

    Uses static rank whenever possible (safe in `tf.function`).
    """
    q_idx = _median_index(quantiles)
    if q_idx is None:
        return y_pred

    r = y_pred.shape.rank
    # (B,H,Q,1) -> (B,H,1)
    if r == 4:
        return y_pred[:, :, q_idx, :]
    # (B,H,Q) -> (B,H,1)
    if r == 3:
        return y_pred[:, :, q_idx : q_idx + 1]

    # Unknown rank: keep original tensor (debug-only).
    return y_pred

    r = tf_rank(y_pred)
    # (B,H,Q,1) -> (B,H,1)
    if int(r) == 4:
        return y_pred[:, :, q_idx, :]
    # (B,H,Q) -> (B,H,1)
    if int(r) == 3:
        return y_pred[:, :, q_idx : q_idx + 1]
    return y_pred

# ---------------------------------------------------------------------
# Step 0/1/2: input packaging
# ---------------------------------------------------------------------
def dbg_step0_inputs_targets(
    *,
    verbose: int,
    inputs: Dict[str, Any],
    targets: Any,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("\n[train_step] verbose=", verbose)
    tf_print("[train_step] --- step 0/1 inputs+targets ---")
    tf_print("[train_step] inputs keys:", list(inputs.keys()))
    if isinstance(targets, dict):
        tf_print("[train_step] targets keys:", list(targets.keys()))

    _vshapes(
        "STEP 0/1 (raw tensors)",
        [
            ("inputs['H_field']", inputs.get("H_field", None)),
            ("inputs['soil_thickness']",
             inputs.get("soil_thickness", None)),
            ("inputs['coords']", inputs.get("coords", None)),
            ("inputs['static_features']",
             inputs.get("static_features", None)),
            ("inputs['dynamic_features']",
             inputs.get("dynamic_features", None)),
            ("inputs['future_features']",
             inputs.get("future_features", None)),
        ],
    )


def dbg_step1_thickness(
    *,
    verbose: int,
    H_field: Tensor,
    H_si: Tensor,
) -> None:
    if not dbg_on(verbose, 3):
        return

    tf_print("[train_step] --- step 1 thickness ---")
    _vshapes(
        "STEP 1 (thickness)",
        [
            ("H_field", H_field),
            ("H_si", H_si),
        ],
    )


def dbg_step2_coords_checks(
    *,
    verbose: int,
    coords: Tensor,
    inputs: Dict[str, Any],
) -> None:
    if not dbg_on(verbose, 6):
        return

    tf_debugging.assert_equal(
        tf_rank(coords),
        3,
        "coords must be rank-3 (B,H,3)",
    )
    tf_debugging.assert_equal(
        tf_shape(coords)[-1],
        3,
        "coords last dim must be 3",
    )

    tf_print_nonfinite("train_step/coords", coords)

    for k in (
        "static_features",
        "dynamic_features",
        "future_features",
        "H_field",
    ):
        v = inputs.get(k, None)
        if v is None:
            continue
        tf_print_nonfinite(
            f"train_step/{k}",
            tf_cast(v, tf_float32),
        )


# ---------------------------------------------------------------------
# Units / layout checks
# ---------------------------------------------------------------------
def dbg_units_once(
    *,
    verbose: int,
    iterations: Tensor,
    targets: Dict[str, Tensor],
    gwl_pred_final: Tensor,
    s_pred_final: Tensor,
    quantiles: Optional[Sequence[float]],
) -> None:
    if not dbg_on(verbose, 6):
        return

    it = tf_cast(iterations, tf_int32)

    def _dbg() -> Tensor:
        yt_g = tf_cast(targets["gwl_pred"], tf_float32)
        yt_s = tf_cast(targets["subs_pred"], tf_float32)

        yp_g = _pick_q50(tf_cast(gwl_pred_final, tf_float32),
                         quantiles)
        yp_s = _pick_q50(tf_cast(s_pred_final, tf_float32),
                         quantiles)

        tf_print("\n[train_step][units] iter=", it)

        tf_print("[shapes] yt_g=", tf_shape(yt_g),
                 "yp_g=", tf_shape(yp_g))
        tf_print("[shapes] yt_s=", tf_shape(yt_s),
                 "yp_s=", tf_shape(yp_s))

        tf_print("[gwl] y_true min/max/mean =",
                 tf_reduce_min(yt_g),
                 tf_reduce_max(yt_g),
                 tf_reduce_mean(yt_g))
        tf_print("[gwl] y_pred(p50) min/max/mean =",
                 tf_reduce_min(yp_g),
                 tf_reduce_max(yp_g),
                 tf_reduce_mean(yp_g))

        tf_print("[subs] y_true min/max/mean =",
                 tf_reduce_min(yt_s),
                 tf_reduce_max(yt_s),
                 tf_reduce_mean(yt_s))
        tf_print("[subs] y_pred(p50) min/max/mean =",
                 tf_reduce_min(yp_s),
                 tf_reduce_max(yp_s),
                 tf_reduce_mean(yp_s))

        tf_print("[gwl] mean(|y_true|) =",
                 tf_reduce_mean(tf_abs(yt_g)))
        tf_print("[gwl] mae(batch,p50) =",
                 tf_reduce_mean(tf_abs(yt_g - yp_g)))
        return tf_constant(0, tf_int32)

    _ = tf_cond(
        tf_equal(it, 0),
        _dbg,
        lambda: tf_constant(0, tf_int32),
    )


def dbg_assert_data_layout(
    *,
    verbose: int,
    data_final: Tensor,
    data_mean_raw: Optional[Tensor],
    quantiles: Optional[Sequence[float]],
) -> None:
    if not dbg_on(verbose, 10):
        return

    if quantiles is None:
        tf_debugging.assert_equal(
            tf_rank(data_final),
            3,
            "data_final must be (B,H,O)",
        )
    else:
        tf_debugging.assert_equal(
            tf_rank(data_final),
            4,
            "data_final must be (B,H,Q,O)",
        )
        tf_debugging.assert_equal(
            tf_shape(data_final)[2],
            len(list(quantiles)),
            "Q axis mismatch",
        )

    if data_mean_raw is not None:
        tf_debugging.assert_equal(
            tf_rank(data_mean_raw),
            3,
            "data_mean_raw must be (B,H,O)",
        )


# ---------------------------------------------------------------------
# Step 3: mean-head prep
# ---------------------------------------------------------------------
def dbg_step3_mean_head(
    *,
    verbose: int,
    gwl_mean_raw: Tensor,
    gwl_si: Tensor,
    h_si: Tensor,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step: mean-head prep ---")
    _vshapes(
        "STEP 3.2 (mean head)",
        [
            ("gwl_mean_raw", gwl_mean_raw),
            ("gwl_si", gwl_si),
            ("h_si", h_si),
        ],
    )


def dbg_step31_forward_outputs(
    *,
    verbose: int,
    data_final: Tensor,
    s_pred_final: Tensor,
    gwl_pred_final: Tensor,
    data_mean_raw: Optional[Tensor],
    phys_mean_raw: Tensor,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step: forward outputs ---")
    _vshape("data_final", data_final)
    _vshape("s_pred_final", s_pred_final)
    _vshape("gwl_pred_final", gwl_pred_final)
    if data_mean_raw is not None:
        _vshape("data_mean_raw", data_mean_raw)
    _vshape("phys_mean_raw", phys_mean_raw)


# ---------------------------------------------------------------------
# Step 3.3: physics fields
# ---------------------------------------------------------------------
def dbg_step33_physics_logits(
    *,
    verbose: int,
    K_logits: Tensor,
    Ss_logits: Tensor,
    dlogtau_logits: Tensor,
    Q_logits: Optional[Tensor],
    K_base: Tensor,
    Ss_base: Tensor,
    dlogtau_base: Tensor,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step: physics logits/base ---")
    _vshapes(
        "STEP 3.3 (physics logits)",
        [
            ("K_logits", K_logits),
            ("Ss_logits", Ss_logits),
            ("dlogtau_logits", dlogtau_logits),
            ("Q_logits", Q_logits),
            ("K_base", K_base),
            ("Ss_base", Ss_base),
            ("dlogtau_base", dlogtau_base),
        ],
    )


def dbg_step33_physics_fields(
    *,
    verbose: int,
    K_field: Tensor,
    Ss_field: Tensor,
    tau_field: Tensor,
    tau_phys: Tensor,
    Hd_eff: Tensor,
    delta_log_tau: Tensor,
    logK: Tensor,
    logSs: Tensor,
    log_tau: Tensor,
    log_tau_phys: Tensor,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step: physics fields (SI) ---")
    _vshapes(
        "STEP 3.3 (physics fields)",
        [
            ("K_field", K_field),
            ("Ss_field", Ss_field),
            ("tau_field", tau_field),
            ("tau_phys", tau_phys),
            ("Hd_eff", Hd_eff),
            ("delta_log_tau", delta_log_tau),
            ("logK", logK),
            ("logSs", logSs),
            ("log_tau", log_tau),
            ("log_tau_phys", log_tau_phys),
        ],
    )


# ---------------------------------------------------------------------
# Step 4: AD derivatives
# ---------------------------------------------------------------------
def dbg_step4_ad_raw(
    *,
    verbose: int,
    dh_dcoords: Tensor,
    dh_dt_raw: Tensor,
    dh_dx_raw: Tensor,
    dh_dy_raw: Tensor,
    K_dh_dx: Tensor,
    K_dh_dy: Tensor,
    dKdhx_dcoords: Tensor,
    dKdhy_dcoords: Tensor,
    d_K_dh_dx_dx_raw: Tensor,
    d_K_dh_dy_dy_raw: Tensor,
    dK_dcoords: Tensor,
    dSs_dcoords: Tensor,
    dK_dx_raw: Tensor,
    dK_dy_raw: Tensor,
    dSs_dx_raw: Tensor,
    dSs_dy_raw: Tensor,
) -> None:
    if not dbg_on(verbose, 7):
        return

    tf_print("[train_step] --- step: AD derivatives (raw) ---")
    _vshapes(
        "STEP 4 (AD raw grads)",
        [
            ("dh_dcoords", dh_dcoords),
            ("dh_dt_raw", dh_dt_raw),
            ("dh_dx_raw", dh_dx_raw),
            ("dh_dy_raw", dh_dy_raw),
            ("K_dh_dx", K_dh_dx),
            ("K_dh_dy", K_dh_dy),
            ("dKdhx_dcoords", dKdhx_dcoords),
            ("dKdhy_dcoords", dKdhy_dcoords),
            ("d_K_dh_dx_dx_raw", d_K_dh_dx_dx_raw),
            ("d_K_dh_dy_dy_raw", d_K_dh_dy_dy_raw),
            ("dK_dcoords", dK_dcoords),
            ("dSs_dcoords", dSs_dcoords),
            ("dK_dx_raw", dK_dx_raw),
            ("dK_dy_raw", dK_dy_raw),
            ("dSs_dx_raw", dSs_dx_raw),
            ("dSs_dy_raw", dSs_dy_raw),
        ],
    )


def dbg_step41_si_grads(
    *,
    verbose: int,
    dh_dt: Tensor,
    d_K_dh_dx_dx: Tensor,
    d_K_dh_dy_dy: Tensor,
    dK_dx: Tensor,
    dK_dy: Tensor,
    dSs_dx: Tensor,
    dSs_dy: Tensor,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step: chain-rule corrected ---")
    _vshapes(
        "STEP 4.1 (SI grads)",
        [
            ("dh_dt", dh_dt),
            ("d_K_dh_dx_dx", d_K_dh_dx_dx),
            ("d_K_dh_dy_dy", d_K_dh_dy_dy),
            ("dK_dx", dK_dx),
            ("dK_dy", dK_dy),
            ("dSs_dx", dSs_dx),
            ("dSs_dy", dSs_dy),
        ],
    )


# ---------------------------------------------------------------------
# Step 5: Q forcing
# ---------------------------------------------------------------------
def dbg_step5_q_source(
    *,
    verbose: int,
    Q_si: Tensor,
    dh_dt: Tensor,
) -> None:
    if not dbg_on(verbose, 3):
        return

    tf_print("[train_step] --- step: Q source term ---")
    _vshapes(
        "STEP 5 (Q)",
        [
            ("Q_si", Q_si),
            ("dh_dt", dh_dt),
        ],
    )


# ---------------------------------------------------------------------
# Step 6: consolidation
# ---------------------------------------------------------------------
def dbg_cons_units_rms(
    *,
    verbose: int,
    sk: Dict[str, Any],
    cons_res: Tensor,
) -> None:
    if not dbg_on(verbose, 6):
        return

    mode = resolve_cons_units(sk)
    tf_print("cons_units=", mode, "rms=", to_rms(cons_res))


def dbg_step6_consolidation(
    *,
    verbose: int,
    allow_resid: bool,
    cons_active: bool,
    s_mean_raw: Tensor,
    s_pred_si: Tensor,
    dt_units: Tensor,
    s0_cum_11: Tensor,
    s_inc_pred: Tensor,
    s_state: Tensor,
    h_ref_si_11: Tensor,
    h_state: Tensor,
    cons_step_m: Tensor,
    cons_res: Tensor,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step: consolidation branch ---")
    tf_print("[train_step] allow_resid=", allow_resid,
             "| cons_active=", cons_active)

    _vshapes(
        "STEP 6 (consolidation state build)",
        [
            ("s_mean_raw", s_mean_raw),
            ("s_pred_si", s_pred_si),
            ("dt_units", dt_units),
            ("s0_cum_11", s0_cum_11),
            ("s_inc_pred", s_inc_pred),
            ("s_state", s_state),
            ("h_ref_si_11", h_ref_si_11),
            ("h_state", h_state),
            ("cons_step_m", cons_step_m),
            ("cons_res", cons_res),
        ],
    )


# ---------------------------------------------------------------------
# Step 7: residuals + priors
# ---------------------------------------------------------------------
def dbg_step7_residuals(
    *,
    verbose: int,
    gw_res: Tensor,
    prior_res: Tensor,
    smooth_res: Tensor,
    loss_mv: Tensor,
    bounds_res: Tensor,
    loss_bounds: Tensor,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step: residuals + priors ---")
    _vshapes(
        "STEP 7 (residual tensors)",
        [
            ("gw_res", gw_res),
            ("prior_res", prior_res),
            ("smooth_res", smooth_res),
            ("loss_mv", loss_mv),
            ("bounds_res", bounds_res),
            ("loss_bounds", loss_bounds),
        ],
    )


# ---------------------------------------------------------------------
# Step 8: scaling / finiteness checks
# ---------------------------------------------------------------------
def dbg_step8_scaling(
    *,
    verbose: int,
    cons_res_raw: Tensor,
    gw_res_raw: Tensor,
    cons_res: Tensor,
    gw_res: Tensor,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step 8 scaling ---")
    _vshapes(
        "STEP 8 (before/after scaling)",
        [
            ("cons_res_raw", cons_res_raw),
            ("gw_res_raw", gw_res_raw),
            ("cons_res", cons_res),
            ("gw_res", gw_res),
        ],
    )


def dbg_chk_scales(
    *,
    verbose: int,
    level: int,
    scales: Dict[str, Tensor],
) -> None:
    if not dbg_on(verbose, level):
        return

    for k in ("cons_scale", "gw_scale"):
        v = scales.get(k, None)
        if v is None:
            continue
        tf_debugging.assert_all_finite(
            v,
            f"{k} has NaN/Inf",
        )


def dbg_chk_core_finite(
    *,
    verbose: int,
    level: int,
    cons_res: Tensor,
    gw_res: Tensor,
    tau_field: Tensor,
    K_field: Tensor,
    Ss_field: Tensor,
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_debugging.assert_all_finite(
        cons_res,
        "cons_res has NaN/Inf",
    )
    tf_debugging.assert_all_finite(
        gw_res,
        "gw_res has NaN/Inf",
    )
    tf_debugging.assert_all_finite(
        tau_field,
        "tau_field has NaN/Inf",
    )
    tf_debugging.assert_all_finite(
        K_field,
        "K_field has NaN/Inf",
    )
    tf_debugging.assert_all_finite(
        Ss_field,
        "Ss_field has NaN/Inf",
    )


# ---------------------------------------------------------------------
# Step 9/10: loss + grads
# ---------------------------------------------------------------------
def dbg_step9_losses(
    *,
    verbose: int,
    data_loss: Tensor,
    loss_cons: Tensor,
    loss_gw: Tensor,
    loss_prior: Tensor,
    loss_smooth: Tensor,
    physics_loss_raw: Tensor,
    physics_loss_scaled: Tensor,
    total_loss: Tensor,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step 9 losses ---")
    _vshapes(
        "STEP 9 (loss scalars)",
        [
            ("data_loss", data_loss),
            ("loss_cons", loss_cons),
            ("loss_gw", loss_gw),
            ("loss_prior", loss_prior),
            ("loss_smooth", loss_smooth),
            ("physics_loss_raw", physics_loss_raw),
            ("physics_loss_scaled", physics_loss_scaled),
            ("total_loss", total_loss),
        ],
    )


def dbg_step10_grads(
    *,
    verbose: int,
    trainable_vars: Sequence[Tensor],
    grads: Sequence[Optional[Tensor]],
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step 10 gradients ---")
    tf_print("[train_step] n_trainable_vars =",
             tf_constant(len(trainable_vars), tf_int32))

    n_show = min(12, len(trainable_vars))
    for v, g in list(zip(trainable_vars, grads))[:n_show]:
        if g is None:
            tf_print("[grad] NONE | var:", v.name,
                     "| var_shape:", tf_shape(v))
        else:
            tf_print("[grad]", v.name,
                     "| var_shape:", tf_shape(v),
                     "| grad_shape:", tf_shape(g))


def dbg_term_grads_finite(
    *,
    verbose: int,
    debug_grads: bool,
    trainable_vars: Sequence[Tensor],
    data_loss: Tensor,
    terms_scaled: Dict[str, Tensor],
    tape: Any,
) -> None:
    if not debug_grads:
        return
    if not dbg_on(verbose, 1):
        # debug_grads is an explicit toggle, but keep a
        # tiny verbosity gate to avoid surprises.
        pass

    # Check each term independently.
    for name, lt in {"data": data_loss, **terms_scaled}.items():
        g = tape.gradient(lt, trainable_vars)
        _assert_grads_finite(g, trainable_vars)


def dbg_done_apply_gradients(
    *,
    verbose: int,
) -> None:
    if not dbg_on(verbose, 10):
        return
    tf_print("[train_step] --- apply_gradients done ---\n")

def dbg_select_q(
    y: Tensor,
    quantiles: Optional[Sequence[float]],
    *,
    q: float = 0.5,
) -> Tensor:
    """
    Select a quantile slice if y is (B,H,Q,1)/(B,H,Q).

    If quantiles is None, returns y as-is.
    """
    if quantiles is None:
        return y

    qs = np.asarray(list(quantiles), dtype=float)
    idx = int(np.argmin(np.abs(qs - float(q))))

    r = int(y.shape.rank or 0)
    if r >= 3:
        # (B,H,Q,1) or (B,H,Q)
        yq = y[..., idx : idx + 1, ...]
        # If last dim is 1, squeeze Q only.
        # Keep (B,H,1) convention when possible.
        if yq.shape.rank == 4:
            return yq[:, :, 0, :]
        return yq
    return y


def dbg_step5_q(
    *,
    verbose: int,
    Q_si: Tensor,
    dh_dt: Tensor,
) -> None:
    """Print Q source term block."""
    if not dbg_on(verbose, 3):
        return

    tf_print("[train_step] --- Q source term ---")
    _vshapes(
        "STEP 5 (Q)",
        [
            ("Q_si", Q_si),
            ("dh_dt", dh_dt),
        ],
    )



def _dbg_stats_line(tag: str, x: Tensor) -> None:
    """Print min/mean/max + rms for a tensor."""
    x = tf_cast(x, tf_float32)
    tf_print(
        "[stats]",
        tag,
        "min/mean/max=",
        tf_reduce_min(x),
        tf_reduce_mean(x),
        tf_reduce_max(x),
        "| rms=",
        tf_sqrt(tf_reduce_mean(tf_square(x))),
    )


def dbg_step8_residual_scale_stats(
    *,
    verbose: int,
    level: int = 3,
    cons_res_raw: Tensor,
    cons_scale: Tensor,
    gw_res_raw: Tensor,
    gw_scale: Tensor,
) -> None:
    """
    Debug block for residuals + scaling stats.

    Replaces:
        _stats("cons_res_raw", cons_res)
        _stats("cons_scale", scales["cons_scale"])
        _stats("gw_res_raw", gw_res)
        _stats("gw_scale", scales["gw_scale"])
    """
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- residual/scale stats ---")
    _dbg_stats_line("cons_res_raw", cons_res_raw)
    _dbg_stats_line("cons_scale", cons_scale)
    _dbg_stats_line("gw_res_raw", gw_res_raw)
    _dbg_stats_line("gw_scale", gw_scale)


def dbg_dt_debug(
    *,
    verbose: int,
    level: int = 6,
    time_units: str,
    dt_units: Tensor,
    t: Tensor,
) -> None:
    """
    Debug dt conversion and t-grid sanity.

    Replaces the "dt debug" block.
    """
    if not dbg_on(verbose, level):
        return

    sec_u = seconds_per_time_unit(
        time_units,
        dtype=tf_float32,
    )
    dt_sec = dt_to_seconds(
        dt_units,
        time_units=time_units,
    )

    tf_print("[dt debug] time_units =", time_units)
    tf_print("[dt debug] sec_u =", sec_u)

    tf_print(
        "[dt debug] dt_units min/mean/max =",
        tf_reduce_min(dt_units),
        tf_reduce_mean(dt_units),
        tf_reduce_max(dt_units),
    )

    tf_print(
        "[dt debug] dt_sec   min/mean/max =",
        tf_reduce_min(dt_sec),
        tf_reduce_mean(dt_sec),
        tf_reduce_max(dt_sec),
    )

    tf_print(
        "[dt debug] dt_sec / dt_units (mean) =",
        tf_reduce_mean(
            dt_sec / tf_maximum(dt_units, tf_constant(1e-12, tf_float32))
        ),
    )

    # show the first sample time grid (B=0)
    tf_print("[t debug] t[0,:,0] =", t[0, :, 0])


def dbg_call_nonfinite(
    *,
    verbose: int,
    level: int = 6,
    coords_for_decoder: Tensor,
    H_si: Tensor,
    K_base: Tensor,
    Ss_base: Tensor,
    dlogtau_base: Tensor,
    tau_field: Tensor,
) -> None:
    """
    Debug non-finite checks for call() internal tensors.

    Replaces:
        tf_print_nonfinite("call/coords_for_decoder", coords_for_decoder)
        ...
    """
    if not dbg_on(verbose, level):
        return

    tf_print_nonfinite("call/coords_for_decoder", coords_for_decoder)
    tf_print_nonfinite("call/H_si", H_si)
    tf_print_nonfinite("call/K_base", K_base)
    tf_print_nonfinite("call/Ss_base", Ss_base)
    tf_print_nonfinite("call/tau_base", dlogtau_base)
    tf_print_nonfinite(
        "call/tau_field(pre-integrator)",
        tau_field,
    )

def dbg_step3_residual_scales(
    *,
    verbose: int,
    cons_res: Tensor,
    gw_res: Tensor,
    scales: dict,
    level: int = 3,
) -> None:
    """Print raw residual stats + scaling factors."""
    if not dbg_on(verbose, level):
        return

    cons_scale = scales.get("cons_scale", None)
    gw_scale = scales.get("gw_scale", None)

    _stats("cons_res_raw", cons_res)
    if cons_scale is not None:
        _stats("cons_scale", cons_scale)

    _stats("gw_res_raw", gw_res)
    if gw_scale is not None:
        _stats("gw_scale", gw_scale)


def dbg_dt_diag(
    *,
    verbose: int,
    time_units: str,
    dt_units: Tensor,
    t: Tensor,
    level: int = 7,
) -> None:
    """Print dt consistency checks in time_units and seconds."""
    if not dbg_on(verbose, level):
        return

    sec_u = seconds_per_time_unit(
        time_units,
        dtype=tf_float32,
    )
    dt_sec = dt_to_seconds(
        dt_units,
        time_units=time_units,
    )

    tf_print("[dt debug] time_units =", time_units)
    tf_print("[dt debug] sec_u =", sec_u)

    tf_print(
        "[dt debug] dt_units min/mean/max =",
        tf_reduce_min(dt_units),
        tf_reduce_mean(dt_units),
        tf_reduce_max(dt_units),
    )

    tf_print(
        "[dt debug] dt_sec   min/mean/max =",
        tf_reduce_min(dt_sec),
        tf_reduce_mean(dt_sec),
        tf_reduce_max(dt_sec),
    )

    tf_print(
        "[dt debug] dt_sec / dt_units (mean) =",
        tf_reduce_mean(
            dt_sec / tf_maximum(dt_units, 1e-12),
        ),
    )

    tf_print("[t debug] t[0,:,0] =", t[0, :, 0])


def dbg_call_nonfinite_diag(
    *,
    verbose: int,
    coords_for_decoder: Tensor,
    H_si: Tensor,
    K_base: Tensor,
    Ss_base: Tensor,
    dlogtau_base: Tensor,
    tau_field: Tensor,
    level: int = 7,
) -> None:
    """Print non-finite diagnostics inside call()."""
    if not dbg_on(verbose, level):
        return

    tf_print_nonfinite(
        "call/coords_for_decoder",
        coords_for_decoder,
    )
    tf_print_nonfinite("call/H_si", H_si)
    tf_print_nonfinite("call/K_base", K_base)
    tf_print_nonfinite("call/Ss_base", Ss_base)
    tf_print_nonfinite(
        "call/tau_base",
        dlogtau_base,
    )
    tf_print_nonfinite(
        "call/tau_field(pre-integrator)",
        tau_field,
    )

def dbg_gw_grad_flux_rms(
    *,
    verbose: int,
    dh_dx_raw: Tensor,
    dh_dy_raw: Tensor,
    K_field: Tensor,
    level: int = 8,
    prefix: str = "gw/gradflux",
) -> None:
    """
    Print RMS diagnostics for spatial head gradients and Darcy-like
    flux terms K*∂h/∂x, K*∂h/∂y (raw coord units).

    Replaces:
        tf_print("to_rms(dh_dx)=", to_rms(dh_dx_raw))
        tf_print("to_rms(dh_dy)=", to_rms(dh_dy_raw))
        tf_print("to_rms(K_field * dh_dx)=", to_rms(K_field * dh_dx_raw))
        tf_print("to_rms(K_field * dh_dy)=", to_rms(K_field * dh_dy_raw))
    """
    if not dbg_on(verbose, level):
        return

    tf_print(f"[{prefix}] rms(dh_dx_raw)=", to_rms(dh_dx_raw))
    tf_print(f"[{prefix}] rms(dh_dy_raw)=", to_rms(dh_dy_raw))

    fx = K_field * dh_dx_raw
    fy = K_field * dh_dy_raw

    tf_print(f"[{prefix}] rms(K*dh_dx_raw)=", to_rms(fx))
    tf_print(f"[{prefix}] rms(K*dh_dy_raw)=", to_rms(fy))
