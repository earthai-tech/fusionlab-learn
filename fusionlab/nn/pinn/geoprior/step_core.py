# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations

from typing import Any, Dict, Optional

from ... import KERAS_DEPS
from ..._utils import get_tensor_from

from .batch_io import _get_coords
from .debugs import ( 
    dbg_step9_losses, 
    dbg_step2_coords_checks,
    dbg_step33_physics_fields, 
    dbg_step33_physics_logits
   )
from .derivatives import (
    compute_head_pde_derivatives_raw,
    ensure_si_derivative_frame,
)
from .losses import (
    assemble_physics_loss,
    build_physics_bundle,
    pack_eval_physics,
)
from .maths import (
    compose_physics_fields,
    compute_bounds_residual,
    compute_consolidation_step_residual,
    compute_gw_flow_residual,
    compute_mv_prior,
    compute_scales,
    compute_smoothness_prior,
    cons_step_to_cons_residual,
    guard_scale_with_residual,
    q_to_gw_source_term_si,
    resolve_auto_scale_floor,
    resolve_cons_drawdown_options,
    resolve_gw_units,
    scale_residual,
    seconds_per_time_unit,
    settlement_state_for_pde,
    to_rms,
)
from .stability import (
    clamp_physics_logits,
    compute_physics_warmup_gate,
    sanitize_scales,
)
from .utils import (
    get_h_ref_si,
    get_s_init_si,
    get_sk,
    gwl_to_head_m,
    infer_dt_units_from_t,
    to_si_head,
    to_si_subsidence,
    to_si_thickness,
    validate_scaling_kwargs,
)

K = KERAS_DEPS

Tensor = K.Tensor
GradientTape = K.GradientTape

tf_broadcast_to = K.broadcast_to
tf_cast = K.cast
tf_concat = K.concat
tf_cond = K.cond
tf_constant = K.constant
tf_convert_to_tensor = K.convert_to_tensor
tf_equal = K.equal
tf_expand_dims = K.expand_dims
tf_float32 = K.float32
tf_float64 = K.float64
tf_greater_equal = K.greater_equal
tf_int32 = K.int32
tf_maximum = K.maximum
tf_rank = K.rank
tf_reduce_mean = K.reduce_mean
tf_reshape = K.reshape
tf_shape = K.shape
tf_square = K.square
tf_stop_gradient = K.stop_gradient
tf_tile = K.tile
tf_zeros_like = K.zeros_like


def _mean_if_quantiles(x: Tensor) -> Tensor:
    """Mean over Q axis if present; ensure (B,H,1)."""
    r = tf_rank(x)
    x = tf_cond(
        tf_greater_equal(r, 3),
        lambda: tf_reduce_mean(x, axis=2),
        lambda: x,
    )
    r2 = tf_rank(x)
    x = tf_cond(
        tf_equal(r2, 2),
        lambda: tf_expand_dims(x, axis=-1),
        lambda: x,
    )
    return x


def _ensure_bh1(x: Tensor, like: Tensor) -> Tensor:
    """Force (B,H,1) and broadcast to `like`."""
    r = tf_rank(x)
    x = tf_cond(
        tf_equal(r, 2),
        lambda: tf_reshape(
            x,
            [tf_shape(x)[0], tf_shape(x)[1], 1],
        ),
        lambda: x,
    )
    return x + tf_zeros_like(like)


def _coords_to_bh3(model: Any, coords: Tensor) -> Tensor:
    """Ensure coords is (B,H,3)."""
    if coords.shape.rank == 2:
        coords = tf_expand_dims(coords, axis=1)
        H = int(getattr(model, "forecast_horizon", 1))
        coords = tf_tile(coords, [1, H, 1])
    return coords


def _physics_is_on(model: Any) -> bool:
    """True if physics terms are enabled."""
    if hasattr(model, "_physics_off"):
        return not bool(model._physics_off())
    return True


def physics_core(
    model: Any,
    inputs: Dict[str, Optional[Tensor]],
    training: bool,
    return_maps: bool = False,
    *,
    for_train: bool = False,
) -> Dict[str, Any]:
    """Shared physics path for train/eval."""
    sk = getattr(model, "scaling_kwargs", None) or {}
    validate_scaling_kwargs(sk)

    verbose = getattr(model, "verbose", 0) if for_train else 0
    time_units = getattr(model, "time_units", None)

    # ----------------------------------------------------------
    # 1) Prepare H_si + coords + dt
    # ----------------------------------------------------------
    H_in = get_tensor_from(
        inputs,
        "H_field",
        "soil_thickness",
        auto_convert=True,
    )
    if H_in is None:
        raise ValueError(
            "physics_core() requires 'H_field' "
            "(or 'soil_thickness') in inputs."
        )

    H_field = tf_convert_to_tensor(H_in, dtype=tf_float32)
    H_si = to_si_thickness(H_field, sk)

    H_floor = float(get_sk(sk, "H_floor_si", default=1e-3))
    H_si = tf_maximum(H_si, tf_constant(H_floor, tf_float32))

    coords = tf_convert_to_tensor(_get_coords(inputs), tf_float32)
    coords = _coords_to_bh3(model, coords)

    if coords.shape.rank != 3 or coords.shape[-1] != 3:
        raise ValueError(
            "coords must be (B,H,3) with (t,x,y)."
        )

    inputs_fwd = dict(inputs)
    inputs_fwd["coords"] = coords

    t = coords[..., 0:1]
    dt_units = infer_dt_units_from_t(t, sk)

    coords_norm = bool(get_sk(sk, "coords_normalized", False))
    coords_deg = bool(get_sk(sk, "coords_in_degrees", False))

    dbg_step2_coords_checks(
        verbose=verbose,
        coords=coords,
        inputs=inputs,
    )
        
    # ----------------------------------------------------------
    # 2) Physics OFF shortcut
    # ----------------------------------------------------------
    if not _physics_is_on(model):
        y_pred, aux = model._forward_all(
            inputs_fwd,
            training=training,
        )
        return {
            "y_pred": y_pred,
            "aux": aux,
            "physics": None,
            "physics_packed": pack_eval_physics(
                model,
                physics=None,
            ),
            "terms_scaled": {},
            "dt_units": dt_units,
        }

    # ----------------------------------------------------------
    # 3) Forward + AD derivatives (raw coord units)
    # ----------------------------------------------------------
    with GradientTape(persistent=True) as tape:
        tape.watch(coords)

        # dbg_step3_forward(...)
        y_pred, aux = model._forward_all(
            inputs_fwd,
            training=training,
        )

        data_mean_raw = aux.get("data_mean_raw", None)
        if data_mean_raw is not None:
            subs_m, gwl_m = model.split_data_predictions(
                data_mean_raw,
            )
        else:
            subs_m = _mean_if_quantiles(y_pred["subs_pred"])
            gwl_m = _mean_if_quantiles(y_pred["gwl_pred"])

        subs_mean_raw = _mean_if_quantiles(subs_m)
        gwl_mean_raw = _mean_if_quantiles(gwl_m)

        gwl_si = to_si_head(tf_cast(gwl_mean_raw, tf_float32), sk)
        h_si = gwl_to_head_m(gwl_si, sk, inputs=inputs_fwd)

        phys_mean_raw = aux.get("phys_mean_raw", None)
        if phys_mean_raw is None:
            raise ValueError("Missing 'phys_mean_raw'.")

        parts = model.split_physics_predictions(phys_mean_raw)
        K_l, Ss_l, dlt_l, Q_l = parts

        K_l, Ss_l, dlt_l, Q_l = clamp_physics_logits(
            K_l,
            Ss_l,
            dlt_l,
            Q_l,
        )

        freeze = bool(get_sk(
            sk,
            "freeze_physics_fields_over_time",
            default=True,
        ))
        if freeze:
            K_b = tf_broadcast_to(
                tf_reduce_mean(K_l, axis=1, keepdims=True),
                tf_shape(K_l),
            )
            Ss_b = tf_broadcast_to(
                tf_reduce_mean(Ss_l, axis=1, keepdims=True),
                tf_shape(Ss_l),
            )
            tau_b = tf_broadcast_to(
                tf_reduce_mean(dlt_l, axis=1, keepdims=True),
                tf_shape(dlt_l),
            )
        else:
            K_b, Ss_b, tau_b = K_l, Ss_l, dlt_l

        (
            K_field,
            Ss_field,
            tau_field,
            tau_phys,
            Hd_eff,
            dlogtau,
            logK,
            logSs,
            log_tau,
            log_tau_phys,
        ) = compose_physics_fields(
            model,
            coords_flat=coords,
            H_si=H_si,
            K_base=K_b,
            Ss_base=Ss_b,
            tau_base=tau_b,
            training=training,
            verbose=verbose,
        )
 
        dbg_step33_physics_logits(
            verbose=verbose,
            K_logits=K_l,
            Ss_logits=Ss_l,
            dlogtau_logits=dlt_l,
            Q_logits=Q_l,
            K_base=K_b,
            Ss_base=Ss_b,
            dlogtau_base=tau_b,
        )

        dbg_step33_physics_fields(
            verbose=verbose,
            K_field=K_field,
            Ss_field=Ss_field,
            tau_field=tau_field,
            tau_phys=tau_phys,
            Hd_eff=Hd_eff,
            delta_log_tau=dlogtau,
            logK=logK,
            logSs=logSs,
            log_tau=log_tau,
            log_tau_phys=log_tau_phys,
        )
            
        deriv_raw = compute_head_pde_derivatives_raw(
            tape,
            coords,
            h_si,
            K_field,
            Ss_field,
        )

    del tape

    # ----------------------------------------------------------
    # 4) Chain-rule conversion to SI
    # ----------------------------------------------------------
    # dbg_step5_chain_rule(...)
    deriv_si, dmeta = ensure_si_derivative_frame(
        dh_dt_raw=deriv_raw["dh_dt_raw"],
        d_K_dh_dx_dx_raw=deriv_raw["d_K_dh_dx_dx_raw"],
        d_K_dh_dy_dy_raw=deriv_raw["d_K_dh_dy_dy_raw"],
        dK_dx_raw=deriv_raw["dK_dx_raw"],
        dK_dy_raw=deriv_raw["dK_dy_raw"],
        dSs_dx_raw=deriv_raw["dSs_dx_raw"],
        dSs_dy_raw=deriv_raw["dSs_dy_raw"],
        scaling_kwargs=sk,
        time_units=time_units,
        coords_normalized=coords_norm,
        coords_in_degrees=coords_deg,
    )

    dh_dt = deriv_si["dh_dt"]
    dKdhx = deriv_si["d_K_dh_dx_dx"]
    dKdhy = deriv_si["d_K_dh_dy_dy"]
    dK_dx = deriv_si["dK_dx"]
    dK_dy = deriv_si["dK_dy"]
    dSs_dx = deriv_si["dSs_dx"]
    dSs_dy = deriv_si["dSs_dy"]

    tR_tf = dmeta.get("t_range_units_tf", None)

    # ----------------------------------------------------------
    # 5) Q in SI + gate
    # ----------------------------------------------------------
    # dbg_step6_q(...)
    if Q_l is None:
        Q_si = tf_zeros_like(dh_dt)
    else:
        Q_si = q_to_gw_source_term_si(
            model,
            Q_l,
            Ss_field=Ss_field,
            H_field=H_si,
            coords_normalized=coords_norm,
            t_range_units=tR_tf,
            time_units=time_units,
            scaling_kwargs=sk,
            verbose=verbose,
        )
        Q_si = _ensure_bh1(Q_si, like=dh_dt)

    if hasattr(model, "_q_gate"):
        q_gate = model._q_gate()
    else:
        q_gate = tf_constant(1.0, tf_float32)

    Q_si = Q_si * q_gate
    loss_q_reg = tf_reduce_mean(tf_square(Q_si))
    q_rms = to_rms(Q_si)

    if hasattr(model, "_subs_resid_gate"):
        subs_gate = model._subs_resid_gate()
    else:
        subs_gate = tf_constant(0.0, tf_float32)

    # ----------------------------------------------------------
    # 6) Consolidation residual
    # ----------------------------------------------------------
    # dbg_step7_consolidation(...)
    allow_resid = bool(get_sk(sk, "allow_subs_residual", False))
    cons_active = (
        hasattr(model, "pde_modes_active")
        and "consolidation" in getattr(
            model,
            "pde_modes_active",
            (),
        )
    )

    like_11 = h_si[:, :1, :1]
    h_ref_11 = get_h_ref_si(model, inputs_fwd, like=like_11)
    h_ref = h_ref_11 + tf_zeros_like(h_si)

    s_inc_pred = tf_zeros_like(h_si)

    if (not cons_active) or (cons_active and not allow_resid):
        cons_res = tf_zeros_like(h_si)
    else:
        s_pred_si = to_si_subsidence(
            tf_cast(subs_mean_raw, tf_float32),
            sk,
        )

        s0_cum_11 = get_s_init_si(model, inputs_fwd, like=like_11)

        pde_inputs = dict(inputs_fwd)
        pde_inputs["s0_si"] = s0_cum_11

        s_inc_pred = settlement_state_for_pde(
            s_pred_si,
            t,
            scaling_kwargs=sk,
            inputs=pde_inputs,
            time_units=time_units,
            dt=dt_units,
            return_incremental=True,
        )

        s0_inc_11 = tf_zeros_like(s0_cum_11)
        s_state = tf_concat([s0_inc_11, s_inc_pred], axis=1)
        h_state = tf_concat([h_ref_11, h_si], axis=1)

        dd = resolve_cons_drawdown_options(sk)

        cons_step = compute_consolidation_step_residual(
            s_state_si=s_state,
            h_mean_si=h_state,
            Ss_field=Ss_field,
            H_field_si=H_si,
            tau_field=tau_field,
            h_ref_si=h_ref,
            dt=dt_units,
            time_units=time_units,
            method="exact",
            relu_beta=dd["relu_beta"],
            drawdown_mode=dd["drawdown_mode"],
            drawdown_rule=dd["drawdown_rule"],
            stop_grad_ref=dd["stop_grad_ref"],
            drawdown_zero_at_origin=dd[
                "drawdown_zero_at_origin"
            ],
            drawdown_clip_max=dd["drawdown_clip_max"],
            verbose=verbose,
        )

        cons_res = cons_step_to_cons_residual(
            cons_step,
            dt_units=dt_units,
            scaling_kwargs=sk,
            time_units=time_units,
        )

    # ----------------------------------------------------------
    # 7) GW residual + priors
    # ----------------------------------------------------------
    # dbg_step8_residuals(...)
    gw_res = compute_gw_flow_residual(
        model,
        dh_dt=dh_dt,
        d_K_dh_dx_dx=dKdhx,
        d_K_dh_dy_dy=dKdhy,
        Ss_field=Ss_field,
        Q=Q_si,
        verbose=verbose,
    )

    prior_res = dlogtau
    smooth_res = compute_smoothness_prior(
        dK_dx,
        dK_dy,
        dSs_dx,
        dSs_dy,
        K_field=K_field,
        Ss_field=Ss_field,
    )

    step = getattr(getattr(model, "optimizer", None), "iterations", None)
    if step is None:
        step = tf_constant(0, tf_int32)

    loss_mv = compute_mv_prior(
        model,
        Ss_field,
        logSs=logSs,
        as_loss=True,
        step=step,
        alpha_disp=float(get_sk(sk, "mv_alpha_disp", 0.1)),
        delta=float(get_sk(sk, "mv_huber_delta", 1.0)),
        verbose=verbose,
    )

    R_H, R_K, R_Ss, R_tau = compute_bounds_residual(
        model,
        H_field=H_si,
        logK=logK,
        logSs=logSs,
        log_tau=log_tau,
        verbose=verbose,
    )
    bounds_res = tf_concat([R_H, R_K, R_Ss, R_tau], axis=-1)
    loss_bounds = tf_reduce_mean(tf_square(bounds_res))

    # ----------------------------------------------------------
    # 8) GW display units (raw diagnostics only)
    # ----------------------------------------------------------
    gw_units = resolve_gw_units(sk)
    gw_res_si = gw_res
    gw_res_disp = gw_res_si

    if gw_units == "time_unit":
        sec_u = seconds_per_time_unit(
            time_units,
            dtype=tf_float32,
        )
        gw_res_disp = gw_res_si * sec_u

    cons_res_raw = cons_res
    gw_res_raw = gw_res_disp

    # ----------------------------------------------------------
    # 9) Optional nondimensionalization
    # ----------------------------------------------------------
    cons_scaled = cons_res
    gw_scaled = gw_res_si
    scales: Optional[Dict[str, Tensor]] = None

    if bool(getattr(model, "scale_pde_residuals", False)):
        cons_floor = resolve_auto_scale_floor("cons", sk)
        gw_floor = resolve_auto_scale_floor("gw", sk)

        div_term = dKdhx + dKdhy
        s_for_scales = (
            tf_stop_gradient(s_inc_pred)
            if (cons_active and allow_resid)
            else tf_zeros_like(h_si)
        )

        scales = compute_scales(
            model,
            t=t,
            dt=dt_units,
            time_units=time_units,
            s_mean=s_for_scales,
            h_mean=h_si,
            K_field=K_field,
            Ss_field=Ss_field,
            tau_field=tau_field,
            H_field=H_si,
            h_ref_si=h_ref_11,
            Q=Q_si,
            dh_dt=dh_dt,
            div_K_grad_h=div_term,
            verbose=verbose,
        )
        scales = sanitize_scales(scales)
        scales = {k: tf_stop_gradient(v) for k, v in scales.items()}

        cons_s = guard_scale_with_residual(
            residual=cons_res,
            scale=scales["cons_scale"],
            floor=cons_floor,
        )
        gw_s = guard_scale_with_residual(
            residual=gw_res_si,
            scale=scales["gw_scale"],
            floor=gw_floor,
        )

        cons_scaled = scale_residual(
            cons_res,
            cons_s,
            floor=cons_floor,
        )
        gw_scaled = scale_residual(
            gw_res_si,
            gw_s,
            floor=gw_floor,
        )

    # ----------------------------------------------------------
    # 10) Losses + epsilons
    # ----------------------------------------------------------
    loss_cons = tf_reduce_mean(tf_square(cons_scaled))
    loss_gw = tf_reduce_mean(tf_square(gw_scaled))
    loss_prior = tf_reduce_mean(tf_square(prior_res))
    loss_smooth = tf_reduce_mean(tf_square(smooth_res))

    eps_prior = to_rms(prior_res)
    eps_cons_raw = to_rms(cons_res_raw)
    eps_gw_raw = to_rms(gw_res_raw, dtype=tf_float64)
    eps_cons = to_rms(cons_scaled)
    eps_gw = to_rms(gw_scaled, dtype=tf_float64)

    (
        phys_raw,
        phys_scaled,
        phys_mult,
        terms_scaled,
    ) = assemble_physics_loss(
        model,
        loss_cons=loss_cons,
        loss_gw=loss_gw,
        loss_prior=loss_prior,
        loss_smooth=loss_smooth,
        loss_mv=loss_mv,
        loss_q_reg=loss_q_reg,
        loss_bounds=loss_bounds,
    )

    if for_train:
        w = int(get_sk(sk, "physics_warmup_steps", default=500))
        r = int(get_sk(sk, "physics_ramp_steps", default=500))

        gate = compute_physics_warmup_gate(
            step,
            warmup_steps=w,
            ramp_steps=r,
        )
        phys_scaled = phys_scaled * gate
        terms_scaled = {k: v * gate for k, v in terms_scaled.items()}

    physics = build_physics_bundle(
        model,
        physics_loss_raw=phys_raw,
        physics_loss_scaled=phys_scaled,
        phys_mult=phys_mult,
        loss_cons=loss_cons,
        loss_gw=loss_gw,
        loss_prior=loss_prior,
        loss_smooth=loss_smooth,
        loss_mv=loss_mv,
        loss_q_reg=loss_q_reg,
        q_rms=q_rms,
        q_gate=q_gate,
        subs_resid_gate=subs_gate,
        loss_bounds=loss_bounds,
        eps_prior=eps_prior,
        eps_cons=eps_cons,
        eps_gw=eps_gw,
        eps_cons_raw=eps_cons_raw,
        eps_gw_raw=eps_gw_raw,
    )

    out: Dict[str, Any] = {
        "y_pred": y_pred,
        "aux": aux,
        "physics": physics,
        "physics_packed": pack_eval_physics(
            model,
            physics=physics,
        ),
        "terms_scaled": terms_scaled,
        "dt_units": dt_units,
        "scales": scales,
    }

    if return_maps:
        out.update(
            {
                "Q_si": Q_si,
                "K_field": K_field,
                "Ss_field": Ss_field,
                "tau_field": tau_field,
                "tau_phys": tau_phys,
                "Hd_eff": Hd_eff,
                "H_si": H_si,
                "R_cons": cons_res,
                "R_gw": gw_res_si,
                "R_prior": prior_res,
                "R_smooth": smooth_res,
                "R_bounds": bounds_res,
                "R_cons_scaled": cons_scaled,
                "R_gw_scaled": gw_scaled,
                "gw_res_display": gw_res_disp,
                "h_si": h_si,
                "h_ref_si_11": h_ref_11,
            }
        )

        # add legacies names. 
        out.update(
            {
                "K": K_field,               # effective K (m/s)
                "Ss": Ss_field,             # effective Ss (1/m)
                "tau": tau_field,           # learned tau (s)
                "tau_prior": tau_phys,      # closure tau (s)
                "tau_closure": tau_phys,    # alias (clearer naming)
                "Hd": Hd_eff,               # effective drainage thickness (m)
                "H": H_si,                  # base thickness (m)
                "H_field": H_si,            # legacy name used elsewhere
                "cons_res_vals": cons_res,  # alias
            }
        )
        
    dbg_step9_losses(
        verbose=verbose,
        loss_cons=loss_cons,
        loss_gw=loss_gw,
        loss_prior=loss_prior,
        loss_smooth=loss_smooth,
    )

    return out
