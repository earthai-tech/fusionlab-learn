# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""SM3 synthetic identifiability (GeoPrior v3.2).

Example:
  python nat/com/sm3_synth_identifiability_v32.py \
    --outdir results/sm3_synth \
    --n-realizations 30 \
    --n-years 20 \
    --time-steps 5 \
    --epochs 50 \
    --noise-std 0.02 \
    --load-type step \
    --seed 123
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from fusionlab.nn.losses import make_weighted_pinball
from fusionlab.nn.pinn.geoprior.payloads import (
    identifiability_diagnostics_from_payload,
    load_physics_payload,
    summarise_effective_params,
)
from fusionlab.nn.pinn.models import GeoPriorSubsNet
from fusionlab.params import (
    FixedGammaW,
    FixedHRef,
    LearnableKappa,
    LearnableMV,
)

from fusionlab.nn.keras_metrics import (
    Coverage80,
    MAEQ50,
    MSEQ50,
    Sharpness80,
)
from fusionlab.utils.scale_metrics import resolve_noise_std

SEC_PER_YEAR = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class LithoPrior:
    name: str
    K_prior: float
    Ss_prior: float
    Hmin: float
    Hmax: float


def litho_priors() -> List[LithoPrior]:
    return [
        LithoPrior(
            "Fine",
            K_prior=1e-8,
            Ss_prior=5e-5,
            Hmin=5.0,
            Hmax=40.0,
        ),
        LithoPrior(
            "Mixed",
            K_prior=3e-7,
            Ss_prior=2e-5,
            Hmin=5.0,
            Hmax=50.0,
        ),
        LithoPrior(
            "Coarse",
            K_prior=2e-6,
            Ss_prior=8e-6,
            Hmin=5.0,
            Hmax=60.0,
        ),
        LithoPrior(
            "Rock",
            K_prior=1e-7,
            Ss_prior=2e-6,
            Hmin=3.0,
            Hmax=30.0,
        ),
    ]


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[i] = 1.0
    return v


def build_load(
    years: np.ndarray,
    kind: str,
    step_year: int,
    amplitude: float,
    ramp_years: int,
) -> np.ndarray:
    dh = np.zeros_like(years, dtype=float)

    if kind == "step":
        dh[step_year:] = amplitude
        return dh

    if kind != "ramp":
        raise ValueError(
            f"Unknown load type: {kind!r}"
        )

    for i in range(len(years)):
        if i < step_year:
            dh[i] = 0.0
            continue

        if i < step_year + ramp_years:
            denom = max(1, ramp_years)
            frac = (i - step_year + 1) / denom
            dh[i] = amplitude * frac
            continue

        dh[i] = amplitude

    return dh


def settlement_from_tau(
    years: np.ndarray,
    dh: np.ndarray,
    tau_years: float,
    alpha: float,
) -> np.ndarray:
    """Simple settlement response with timescale tau."""
    tau = max(float(tau_years), 1e-6)

    # p is "pressure" proxy: drawdown is -dh.
    p = -np.asarray(dh, dtype=float)

    dp = np.zeros_like(p)
    dp[0] = p[0]
    dp[1:] = p[1:] - p[:-1]

    s = np.zeros_like(p)
    for t in range(len(years)):
        acc = 0.0
        for k in range(t + 1):
            dt = float(years[t] - years[k])
            if dt < 0.0:
                continue
            U = 1.0 - math.exp(-dt / tau)
            acc += dp[k] * U
        s[t] = alpha * acc
    return s

def make_one_step_windows(
    years: np.ndarray,
    dh: np.ndarray,
    s_cum: np.ndarray,
    static_vec: np.ndarray,
    time_steps: int,
    H_field_value: float,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    float,
]:
    years = np.asarray(years, dtype=float)
    dh = np.asarray(dh, dtype=float)
    s_cum = np.asarray(s_cum, dtype=float)

    T = int(time_steps)
    H = 1
    B = len(years) - T
    if B <= 0:
        raise ValueError(
            "Not enough samples for time_steps."
        )

    # Depth-to-water z_GWL (m), positive down.
    z_gwl = -dh

    t0 = float(years[0])
    t_range = float(years[-1] - t0)
    t_range = max(t_range, 1.0)

    X: Dict[str, np.ndarray] = {}

    X["static_features"] = np.repeat(
        static_vec[None, :],
        B,
        axis=0,
    ).astype(np.float32)

    X["dynamic_features"] = np.zeros(
        (B, T, 1),
        np.float32,
    )

    X["future_features"] = np.zeros(
        (B, H, 1),
        np.float32,
    )

    X["coords"] = np.zeros(
        (B, H, 3),
        np.float32,
    )

    X["H_field"] = np.full(
        (B, H, 1),
        float(H_field_value),
        np.float32,
    )

    y = {
        "subs_pred": np.zeros((B, H, 1), np.float32),
        "gwl_pred": np.zeros((B, H, 1), np.float32),
    }

    for i in range(B):
        j = i + T

        X["dynamic_features"][i, :, 0] = z_gwl[i : i + T]
        # X["dynamic_features"][i, :, 1] = s_cum[i : i + T]
        
        y["subs_pred"][i, 0, 0] = s_cum[j]
        # after: z_surf = 0 -> head = -depth
        y["gwl_pred"][i, 0, 0] = -z_gwl[j]
        # y["gwl_pred"][i, 0, 0] = z_gwl[j]

        # normalized coords
        X["coords"][i, 0, 0] = (years[j] - t0) / t_range
        X["coords"][i, 0, 1] = 0.0
        X["coords"][i, 0, 2] = 0.0

    return X, y, t_range

def tf_dataset(
    X: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    *,
    batch: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    xb = {k: tf.convert_to_tensor(v) for k, v in X.items()}
    yb = {k: tf.convert_to_tensor(v) for k, v in y.items()}

    ds = tf.data.Dataset.from_tensor_slices((xb, yb))
    if shuffle:
        n = len(next(iter(X.values())))
        ds = ds.shuffle(
            buffer_size=n,
            seed=int(seed),
            reshuffle_each_iteration=True,
        )
    ds = ds.batch(int(batch))
    return ds.prefetch(tf.data.AUTOTUNE)


def split_tail(
    X: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    val_tail: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    B = len(next(iter(X.values())))
    if not (0 < val_tail < B):
        raise ValueError(
            "val_tail must be in (0, B)."
        )

    cut = B - int(val_tail)

    Xtr = {k: v[:cut] for k, v in X.items()}
    ytr = {k: v[:cut] for k, v in y.items()}
    Xva = {k: v[cut:] for k, v in X.items()}
    yva = {k: v[cut:] for k, v in y.items()}

    return Xtr, ytr, Xva, yva


def _rms(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(a * a)))


def _infer_payload_time_units(meta: Dict[str, Any]) -> str:
    units = meta.get("units") or {}
    tau_u = str(units.get("tau", "")).strip().lower()

    if tau_u in {"s", "sec", "second", "seconds"}:
        return "sec"

    if tau_u in {"y", "yr", "year", "years"}:
        return "year"

    pu = str(meta.get("payload_time_units", ""))
    pu = pu.strip().lower()

    if pu.startswith("y"):
        return "year"

    if pu.startswith("s"):
        return "sec"

    return "sec"


def convert_payload_time_units(
    payload: Dict[str, np.ndarray],
    *,
    from_units: str,
    to_units: str,
    sec_per_year: float = SEC_PER_YEAR,
) -> Dict[str, np.ndarray]:
    fu = str(from_units).strip().lower()
    tu = str(to_units).strip().lower()

    fu = "sec" if fu.startswith("s") else "year"
    tu = "sec" if tu.startswith("s") else "year"

    out = dict(payload)
    if fu == tu:
        return out

    spy = float(sec_per_year)

    def _as(a):
        return np.asarray(a, dtype=float)

    if fu == "sec" and tu == "year":
        for k in ("tau", "tau_prior", "tau_closure"):
            if k in out:
                out[k] = _as(out[k]) / spy
        if "K" in out:
            out["K"] = _as(out["K"]) * spy
        if "cons_res_vals" in out:
            out["cons_res_vals"] = (
                _as(out["cons_res_vals"]) * spy
            )

    if fu == "year" and tu == "sec":
        for k in ("tau", "tau_prior", "tau_closure"):
            if k in out:
                out[k] = _as(out[k]) * spy
        if "K" in out:
            out["K"] = _as(out["K"]) / spy
        if "cons_res_vals" in out:
            out["cons_res_vals"] = (
                _as(out["cons_res_vals"]) / spy
            )

    return out


def _make_scaling_kwargs(
    *,
    t_range: float,
    logK_min: float,
    logK_max: float,
    logSs_min: float,
    logSs_max: float,
    z_surf_static_index: int
) -> Dict[str, Any]:
    # Key fix vs old SM3:
    # - gwl_kind must be "depth_*" (or gwl_col has
    #   "depth") otherwise gwl_to_head_m assumes head.
    # - use_head_proxy=True yields head = -depth when
    #   z_surf is missing (SM3 synthetic case).
    return dict(
        subs_scale_si=1.0,
        subs_bias_si=0.0,
        head_scale_si=1.0,
        head_bias_si=0.0,
        time_units="year",
        seconds_per_time_unit=float(SEC_PER_YEAR),
        coords_normalized=True,
        coord_order=["t", "x", "y"],
        coord_ranges=dict(
            t=float(t_range),
            x=1.0,
            y=1.0,
        ),
        coords_in_degrees=False,
        
        allow_subs_residual=True,
        
        physics_warmup_steps=0, # 10,
        physics_ramp_steps=0, #20,

        # subs_dyn_index=1,
        # subs_dyn_name="subs_model",
        # input is depth (down+), model converts to head using z_surf
        gwl_dyn_index=0,
        gwl_col="GWL_depth_bgs_m",
        gwl_kind="depth_bgs",
        gwl_sign="down_positive",

        # target is head (up+)
        gwl_target_kind="head",
        gwl_target_sign="up_positive",

        # provide a datum so we don't use "head proxy"
        z_surf_m=0.0,
        z_surf=0.0,
        
        z_surf_static_index=int(z_surf_static_index),

        # disable proxy (even if ignored, harmless)
        use_head_proxy=False,

        dynamic_feature_names=["GWL_depth_bgs_m"],
        # dynamic_feature_names=[
        #     "GWL_depth_bgs_m",
        #     "subs_model",
        # ],
        bounds=dict(
            H_min=3.0,
            H_max=80.0,
            logK_min=float(logK_min),
            logK_max=float(logK_max),
            logSs_min=float(logSs_min),
            logSs_max=float(logSs_max),
        ),
    )

def train_one_pixel(
    Xtr: Dict[str, np.ndarray],
    ytr: Dict[str, np.ndarray],
    Xva: Dict[str, np.ndarray],
    yva: Dict[str, np.ndarray],
    *,
    outdir: str,
    seed: int,
    epochs: int,
    batch: int,
    lr: float,
    kappa_b: float,
    gamma_w: float,
    hd_factor: float,
    t_range_years: float,
    n_lith: int, 
    lambda_offset: float = 0, 
) -> Tuple[GeoPriorSubsNet, tf.data.Dataset]:

    tf.keras.utils.set_random_seed(int(seed))

    s_dim = int(Xtr["static_features"].shape[-1])
    d_dim = int(Xtr["dynamic_features"].shape[-1])
    f_dim = int(Xtr["future_features"].shape[-1])

    sk = _make_scaling_kwargs(
        t_range=float(t_range_years),
        logK_min=np.log(1e-14),
        logK_max=np.log(1e-3),
        logSs_min=np.log(1e-8),
        logSs_max=np.log(1e-3),
        z_surf_static_index = n_lith + 2, 
    )

    model = GeoPriorSubsNet(
        static_input_dim=s_dim,
        dynamic_input_dim=d_dim,
        future_input_dim=f_dim,
        output_subsidence_dim=1,
        output_gwl_dim=1,
        forecast_horizon=1,
        quantiles=[0.1, 0.5, 0.9],
        embed_dim=32,
        hidden_units=64,
        lstm_units=64,
        attention_units=32,
        num_heads=2,
        dropout_rate=0.10,
        max_window_size=int(
            Xtr["dynamic_features"].shape[1]
        ),
        attention_levels=["cross"],
        mv=LearnableMV(initial_value=1e-7),
        kappa=LearnableKappa(
            initial_value=float(kappa_b),
            trainable=False,
        ),
        gamma_w=FixedGammaW(value=float(gamma_w)),
        h_ref=FixedHRef(value=0.0),
        kappa_mode="kb",
        use_effective_h=True,
        hd_factor=float(hd_factor),

        # keep False for "raw eps" clarity.
        # you can turn True later once you
        # export cons_res_scaled in payloads.
        scale_pde_residuals=True,

        pde_mode="consolidation",
        offset_mode="log10",
        scaling_kwargs=sk,
    )
    print("logK bounds used:", model.scaling_kwargs["bounds"]["logK_min"],
                             model.scaling_kwargs["bounds"]["logK_max"])
    print("K bounds used:", np.exp(model.scaling_kwargs["bounds"]["logK_min"]),
                           np.exp(model.scaling_kwargs["bounds"]["logK_max"]))

    ds_tr = tf_dataset(
        Xtr,
        ytr,
        batch=batch,
        shuffle=True,
        seed=seed,
    )
    ds_va = tf_dataset(
        Xva,
        yva,
        batch=batch,
        shuffle=False,
        seed=seed,
    )

    for xb, _ in ds_tr.take(1):
        _ = model(xb)
        break

    QUANTILES = [0.1, 0.5, 0.9]
    SUBS_WEIGHTS = {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}

    loss_dict = {
        "subs_pred": make_weighted_pinball(
            QUANTILES, SUBS_WEIGHTS
        ),
        "gwl_pred": tf.keras.losses.MSE,
    }

    # # Metrics (meters).
    metrics_arg = {
        "subs_pred": [
            MAEQ50(name="mae_q50"),
            MSEQ50(name="mse_q50"),
            Coverage80(name="coverage80"),
            Sharpness80(name="sharpness80"),
        ],
        "gwl_pred": [
            MAEQ50(name="mae_q50"),
            MSEQ50(name="mse_q50"),
        ],
    }
    physics_loss_weights = {
        "lambda_cons": 100, #1.0,
        "lambda_gw": 0.0,
        "lambda_prior":1.0, # 0.3,
        "lambda_smooth": 0.0,
        "lambda_bounds": 1.0, #0.0,
        "lambda_mv": 0.0,
        "mv_lr_mult": 1.0,
        "kappa_lr_mult": 0.0,
        "lambda_offset": float(lambda_offset),
        # IMPORTANT: penalize quantile crossing
        "lambda_q": 0.1,
    }

    loss_weights_dict = {"subs_pred": 1.0, "gwl_pred": 1.0}

    out_names = (
        list(getattr(model, "output_names", []))
        or ["subs_pred", "gwl_pred"]
    )

    import keras
    IS_KERAS2 = keras.__version__.startswith("2.")

    if IS_KERAS2:
        loss_arg = [loss_dict[k] for k in out_names]
        lossw_arg = [
            loss_weights_dict.get(k, 1.0) for k in out_names
        ]
        metrics_compile = [
            metrics_arg.get(k, []) for k in out_names
        ]
    else:
        loss_arg = loss_dict
        lossw_arg = loss_weights_dict
        metrics_compile = metrics_arg

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(lr),
            clipnorm=1.0,
        ),
        loss=loss_arg,
        loss_weights=lossw_arg,
        metrics=metrics_compile,
        **physics_loss_weights,
    )

    os.makedirs(outdir, exist_ok=True)
    ckpt = os.path.join(outdir, "best.keras")

    cbs = [
        tf.keras.callbacks.EarlyStopping(
            "val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt,
            "val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=int(epochs),
        verbose=1,
        callbacks=cbs,
    )
    return model, ds_va


def flatten_diag(diag: Dict[str, Any]) -> Dict[str, float]:
    row: Dict[str, float] = {}

    tau_rel = diag.get("tau_rel_error", {})
    row["tau_rel_q50"] = float(tau_rel.get("q50", np.nan))
    row["tau_rel_q90"] = float(tau_rel.get("q90", np.nan))
    row["tau_rel_q95"] = float(tau_rel.get("q95", np.nan))

    clo = diag.get("closure_log_resid", {})
    row["closure_log_resid_mean"] = float(
        clo.get("mean", np.nan)
    )
    row["closure_log_resid_q95"] = float(
        clo.get("q95", np.nan)
    )

    offs = diag.get("offsets", {})
    for block in ("vs_true", "vs_prior"):
        blk = offs.get(block, {})
        for key in ("delta_K", "delta_Ss", "delta_Hd"):
            d = blk.get(key, {})
            f1 = f"{block}_{key}_q50"
            f2 = f"{block}_{key}_q95"
            row[f1] = float(d.get("q50", np.nan))
            row[f2] = float(d.get("q95", np.nan))

    return row


def run_one_realisation(
    *,
    r: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    priors: List[LithoPrior],
    years: np.ndarray,
    outdir: str,
) -> Dict[str, Any]:
    n_lith = len(priors)

    run_dir = os.path.join(outdir, f"real_{r:03d}")
    os.makedirs(run_dir, exist_ok=True)

    lith_idx = int(rng.integers(0, n_lith))
    lp = priors[lith_idx]

    H_phys = float(rng.uniform(lp.Hmin, lp.Hmax))
    H_eff = float(min(H_phys, args.thickness_cap))
    Hd_prior = float(args.hd_factor * H_eff)

    Ss_prior = float(lp.Ss_prior)

    logtau_p = float(
        rng.uniform(
            np.log10(args.tau_min),
            np.log10(args.tau_max),
        )
    )
    tau_prior_year = float(10.0**logtau_p)
    tau_prior_sec = tau_prior_year * SEC_PER_YEAR

    numer = (Hd_prior**2) * Ss_prior
    denom = (np.pi**2) * float(args.kappa_b) * tau_prior_sec
    K_prior_mps = float(numer / denom)

    dlogSs = float(
        rng.normal(0.0, float(args.Ss_spread_dex))
    )
    Ss_true = float(Ss_prior * (10.0**dlogSs))

    logtau_t = float(
        logtau_p
        + rng.normal(0.0, float(args.tau_spread_dex))
    )
    tau_true_year = float(10.0**logtau_t)
    tau_true_year = float(
        np.clip(tau_true_year, args.tau_min, args.tau_max)
    )
    tau_true_sec = tau_true_year * SEC_PER_YEAR

    Hd_true = float(Hd_prior)

    numer_t = (Hd_true**2) * Ss_true
    denom_t = (np.pi**2) * float(args.kappa_b) * tau_true_sec
    K_true_mps = float(numer_t / denom_t)

    amp = float(rng.uniform(-15.0, -5.0))
    step_hi = max(3, args.n_years // 3)
    step_year = int(rng.integers(2, step_hi))
    ramp_years = max(3, args.n_years // 4)

    dh = build_load(
        years=years,
        kind=args.load_type,
        step_year=step_year,
        amplitude=amp,
        ramp_years=ramp_years,
    )

    unit_scale = float(args.alpha)
    alpha_eff = unit_scale * Ss_true * H_eff

    s_cum = settlement_from_tau(
        years,
        dh,
        tau_true_year,
        alpha=alpha_eff,
    )

    y_inc = np.zeros_like(s_cum)
    y_inc[0] = s_cum[0]
    y_inc[1:] = s_cum[1:] - s_cum[:-1]


    # AUTO noise from deterministic increments
    noise_std = resolve_noise_std(
        y_inc,
        noise_std=args.noise_std,
        noise_frac=getattr(args, "noise_frac", 0.10),
        percentile=95.0,
        min_std=0.0,
    )

    y_inc = y_inc + rng.normal(
        0.0,
        float(noise_std),
        size=y_inc.shape,
    )

    s_obs = np.cumsum(y_inc)

    is_capped = float(H_phys > args.thickness_cap)
    # static_vec = np.concatenate(
    #     [
    #         one_hot(lith_idx, n_lith),
    #         np.array([H_eff, is_capped], np.float32),
    #     ],
    #     axis=0,
    # ).astype(np.float32)
    z_surf = 0.0  # constant datum for SM3
    static_vec = np.concatenate(
        [one_hot(lith_idx, n_lith),
         np.array([H_eff, is_capped, z_surf], np.float32)],
        axis=0,
    ).astype(np.float32)

    X, y, t_range_years = make_one_step_windows(
        years=years,
        dh=dh,
        s_cum=s_obs,
        static_vec=static_vec,
        time_steps=args.time_steps,
        H_field_value=H_eff,
    )
    
    Xtr, ytr, Xva, yva = split_tail(X, y, args.val_tail)

    model, ds_va = train_one_pixel(
        Xtr,
        ytr,
        Xva,
        yva,
        outdir=run_dir,
        seed=int(args.seed + r),
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        kappa_b=args.kappa_b,
        gamma_w=args.gamma_w,
        hd_factor=args.hd_factor,
        t_range_years=float(t_range_years),
        n_lith = n_lith, 
        lambda_offset= float(args.lambda_offset),
        
    )

    npz_path = os.path.join(run_dir, "phys_payload_val.npz")


    meta = {
        "synthetic": True,
        "realisation": int(r),
        "report_time_units": "year",
        "sec_per_year": float(SEC_PER_YEAR),
        "kappa_b": float(args.kappa_b),
        "tau_true_sec": float(tau_true_sec),
        "tau_true_year": float(tau_true_year),
        "K_true_mps": float(K_true_mps),
        "Ss_true": float(Ss_true),
        "Hd_true": float(Hd_true),
        "tau_prior_sec": float(tau_prior_sec),
        "tau_prior_year": float(tau_prior_year),
        "K_prior_mps": float(K_prior_mps),
        "Ss_prior": float(Ss_prior),
        "Hd_prior": float(Hd_prior),
        "H_eff": float(H_eff),
        "H_phys": float(H_phys),
        "load_type": str(args.load_type),
        "amp_dh": float(amp),
        "step_year": int(step_year),
        "noise_std": float(noise_std),
        
        "payload_time_units": "sec",
        "units": {
            "tau": "s",
            "K": "m/s",
            "cons_res_vals": "m/s",
        },
    }

    meta["z_surf_m"] = 0.0
    meta["units"]["head"] = "m"

    model.export_physics_payload(
        ds_va,
        save_path=npz_path,
        format="npz",
        overwrite=True,
        metadata=meta,
    )

    payload, meta = load_physics_payload(npz_path)

    # Payload is SI by contract:
    # tau: s, K: m/s, cons_res_vals: m/s
    eff = summarise_effective_params(payload)
    
    tau_est_sec = float(eff["tau"])
    K_est_mps = float(eff["K"])
    
    tau_est_year = tau_est_sec / SEC_PER_YEAR
    K_est_m_per_year = K_est_mps * SEC_PER_YEAR
    
    eps_cons_raw_mps = _rms(payload["cons_res_vals"])
    eps_cons_raw_m_per_year = eps_cons_raw_mps * SEC_PER_YEAR
    
    eps_cons_scaled = float("nan")
    if "cons_res_scaled" in payload:
        eps_cons_scaled = _rms(payload["cons_res_scaled"])
    
    # Diagnostics block (must be in SAME units)
    diag = identifiability_diagnostics_from_payload(
        payload,
        tau_true=float(meta["tau_true_sec"]),
        K_true=float(meta["K_true_mps"]),
        Ss_true=float(meta["Ss_true"]),
        Hd_true=float(meta["Hd_true"]),
        K_prior=float(meta["K_prior_mps"]),
        Ss_prior=float(meta["Ss_prior"]),
        Hd_prior=float(meta["Hd_prior"]),
    )
    
    # pull payload metrics from meta sidecar
    pm = (meta.get("payload_metrics") or {})
    eps_prior_rms = pm.get("eps_prior_rms", np.nan)
    r2_logtau = pm.get("r2_logtau", np.nan)
    clos_rms = pm.get("closure_consistency_rms", np.nan)

    diag_path = os.path.join(
        run_dir,
        "sm3_identifiability_diag.json",
    )
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    dlog10_tau_med = float(
        np.log10(max(tau_est_sec, 1e-12))
        - np.log10(max(meta["tau_true_sec"], 1e-12))
    )
    
    row = {
        "realisation": int(r),
        "lith_idx": int(lith_idx),
    
        "tau_true_sec": float(meta["tau_true_sec"]),
        "tau_prior_sec": float(meta["tau_prior_sec"]),
        "tau_est_med_sec": tau_est_sec,
    
        "tau_true_year": float(meta["tau_true_year"]),
        "tau_prior_year": float(meta["tau_prior_year"]),
        "tau_est_med_year": tau_est_year,
    
        "K_true_mps": float(meta["K_true_mps"]),
        "K_prior_mps": float(meta["K_prior_mps"]),
        "K_est_med_mps": K_est_mps,
        "K_est_med_m_per_year": K_est_m_per_year,
    
        "Ss_true": float(meta["Ss_true"]),
        "Ss_prior": float(meta["Ss_prior"]),
        "Ss_est_med": float(eff["Ss"]),
    
        "Hd_true": float(meta["Hd_true"]),
        "Hd_prior": float(meta["Hd_prior"]),
        "Hd_est_med": float(eff["Hd"]),
    
        # PHYSICAL epsilon (report)
        "eps_cons_raw_rms_mps": float(eps_cons_raw_mps),
        "eps_cons_raw_rms_m_per_year": float(
            eps_cons_raw_m_per_year
        ),
    
        # TRAINING epsilon (unitless, optional)
        "eps_cons_scaled_rms": float(eps_cons_scaled),
    
        # closure diagnostics (dimensionless)
        "eps_prior_rms": float(eps_prior_rms),
        "r2_logtau": float(r2_logtau),
        "closure_consistency_rms": float(clos_rms),
    
        "dlog10_tau_med": dlog10_tau_med,
        "kappa_b": float(args.kappa_b),
    }
    
    row.update(flatten_diag(diag))
    offs = diag.get("offsets", {}).get("vs_true", {})

    def _q50(name: str) -> float:
        d = offs.get(name, {})
        return float(d.get("q50", np.nan))
    
    dK = _q50("delta_K")
    dS = _q50("delta_Ss")
    dH = _q50("delta_Hd")
    
    row["dlogK_q50"] = dK
    row["dlogSs_q50"] = dS
    row["dlogHd_q50"] = dH
    
    row["ridge_resid_q50"] = float(
        abs(dK - (dS + 2.0 * dH))
    )


    return row

def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    priors = litho_priors()
    years = np.arange(int(args.n_years), dtype=float)

    rows: List[Dict[str, Any]] = []

    for r in range(int(args.n_realizations)):
        print("=" * 62)
        print(
            f"Realisation {r+1:03d}/{args.n_realizations:03d}"
        )
        print("=" * 62)

        row = run_one_realisation(
            r=r,
            args=args,
            rng=rng,
            priors=priors,
            years=years,
            outdir=args.outdir,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    
    x = np.log10(df.tau_est_med_sec.to_numpy())
    y = np.log10(df.tau_true_sec.to_numpy())
    
    if x.size < 2:
        r2 = np.nan
    else:
        r = np.corrcoef(x, y)[0, 1]
        r2 = r * r

    print("r2 = np.corrcoef(np.log10(df.tau_est_med_sec), np.log10(df.tau_true_sec))[0,1]**2")
    print("r2 (computed manually):", r2 )
    
    runs_csv = os.path.join(
        args.outdir,
        "sm3_synth_runs.csv",
    )
    df.to_csv(runs_csv, index=False)

    metrics = []
    for c in df.columns:
        if c in ("realisation", "lith_idx"):
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        metrics.append(c)

    summ_rows: List[Dict[str, Any]] = []
    for c in metrics:
        x = df[c].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            mean = float(np.mean(x))
            std = float(np.std(x))
            p05 = float(np.quantile(x, 0.05))
            p50 = float(np.quantile(x, 0.50))
            p95 = float(np.quantile(x, 0.95))
        else:
            mean = float("nan")
            std = float("nan")
            p05 = float("nan")
            p50 = float("nan")
            p95 = float("nan")

        summ_rows.append(
            dict(
                metric=c,
                mean=mean,
                std=std,
                p05=p05,
                p50=p50,
                p95=p95,
            )
        )

    df_sum = pd.DataFrame(summ_rows)
    sum_csv = os.path.join(
        args.outdir,
        "sm3_synth_summary.csv",
    )
    df_sum.to_csv(sum_csv, index=False)

    print("[OK] wrote:", runs_csv)
    print("[OK] wrote:", sum_csv)

    return df


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--outdir", required=True)

    ap.add_argument("--n-realizations", type=int, default=30)
    ap.add_argument("--n-years", type=int, default=20)
    ap.add_argument("--time-steps", type=int, default=5)
    
    ap.add_argument("--lambda-offset", type=float, default=0.0)

    ap.add_argument("--val-tail", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--noise-std", type=float, default=None)
    ap.add_argument("--noise-frac", type=float, default=0.10)

    ap.add_argument(
        "--load-type",
        choices=("step", "ramp"),
        default="step",
    )

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--hd-factor", type=float, default=0.6)
    ap.add_argument(
        "--thickness-cap",
        type=float,
        default=30.0,
    )
    ap.add_argument("--kappa-b", type=float, default=1.0)
    ap.add_argument("--gamma-w", type=float, default=9810.0)

    ap.add_argument("--tau-min", type=float, default=0.3)
    ap.add_argument("--tau-max", type=float, default=10.0)

    ap.add_argument(
        "--tau-spread-dex",
        type=float,
        default=0.3,
    )
    ap.add_argument(
        "--Ss-spread-dex",
        type=float,
        default=0.4,
    )

    args = ap.parse_args()

    ok = args.tau_min > 0.0 and args.tau_max > args.tau_min
    if not ok:
        raise ValueError(
            "--tau-min must be >0 and <tau-max."
        )

    B = int(args.n_years) - int(args.time_steps)
    if not (0 < int(args.val_tail) < B):
        raise ValueError(
            "--val-tail must be in (0, B)."
        )

    _ = run_experiment(args)


if __name__ == "__main__":
    main()
