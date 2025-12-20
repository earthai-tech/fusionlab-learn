# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

# $ python nat.com/sm3_synthetic_identifiability.py 
# --outdir results/sm3_synth --n-realizations 30 --n-years 20 
# --time-steps 5 --epochs 50 --noise-std 0.02 
# --load-type step --seed 123

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple 
import numpy as np
import pandas as pd
import tensorflow as tf

#  I/O + SM3 diagnostics 
from fusionlab.nn.pinn.io import (
    load_physics_payload,
    identifiability_diagnostics_from_payload,
    summarise_effective_params,
)

# model
from fusionlab.nn.pinn.models import GeoPriorSubsNet
from fusionlab.nn.losses import make_weighted_pinball
from fusionlab.params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef

SEC_PER_YEAR = 365.25 * 24 * 3600.0

# ---------------------------------------------------------------------
# Synthetic priors (keep simple & controlled)
# ---------------------------------------------------------------------
@dataclass
class LithoPrior:
    name: str
    K_prior: float
    Ss_prior: float
    Hmin: float
    Hmax: float


def litho_priors() -> List[LithoPrior]:
    return [
        LithoPrior("Fine",   K_prior=1e-8, Ss_prior=5e-5, Hmin=5.0,  Hmax=40.0),
        LithoPrior("Mixed",  K_prior=3e-7, Ss_prior=2e-5, Hmin=5.0,  Hmax=50.0),
        LithoPrior("Coarse", K_prior=2e-6, Ss_prior=8e-6, Hmin=5.0,  Hmax=60.0),
        LithoPrior("Rock",   K_prior=1e-7, Ss_prior=2e-6, Hmin=3.0,  Hmax=30.0),
    ]


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[i] = 1.0
    return v


# ---------------------------------------------------------------------

def tau_from_closure(Hd: float, Ss: float, K: float, kappa_b: float,
                     eps: float = 1e-12) -> float:
    Hd = max(float(Hd), eps)
    Ss = max(float(Ss), eps)
    K  = max(float(K), eps)
    kb = max(float(kappa_b), eps)
    return float((Hd * Hd) * Ss / (np.pi**2 * kb * K))


def build_load(years: np.ndarray, kind: str, step_year: int,
               amplitude: float, ramp_years: int) -> np.ndarray:
    dh = np.zeros_like(years, dtype=float)
    if kind == "step":
        dh[step_year:] = amplitude
        return dh
    if kind == "ramp":
        for i in range(len(years)):
            if i < step_year:
                dh[i] = 0.0
            elif i < step_year + ramp_years:
                frac = (i - step_year + 1) / max(1, ramp_years)
                dh[i] = amplitude * frac
            else:
                dh[i] = amplitude
        return dh
    raise ValueError(f"Unknown load type: {kind!r}")


def settlement_from_tau(years: np.ndarray, dh: np.ndarray,
                        tau_years: float, alpha: float) -> np.ndarray:
    """
    Simple 1D convolution-style settlement:
    each increment in -dh triggers an exponential response with timescale tau.
    """
    tau = max(float(tau_years), 1e-6)
    p = -np.asarray(dh, dtype=float)
    dp = np.zeros_like(p)
    dp[0] = p[0]
    dp[1:] = p[1:] - p[:-1]

    s = np.zeros_like(p)
    for t in range(len(years)):
        acc = 0.0
        for k in range(t + 1):
            dt = float(years[t] - years[k])
            U = 1.0 - math.exp(-dt / tau) if dt >= 0 else 0.0
            acc += dp[k] * U
        s[t] = alpha * acc
    return s

def make_one_step_windows(
    years: np.ndarray,
    dh: np.ndarray,
    s_cum: np.ndarray,          # cumulative subsidence observation (in mm here)
    static_vec: np.ndarray,
    time_steps: int,
    H_field_value: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    1-step-ahead windows (H=1), single-pixel synthetic.

    Conventions (aligned with GeoPrior):
    - We store groundwater as depth-to-water z_GWL (m, positive downward).
      Here dh is a synthetic "head change" (negative drawdown), so:
          z_GWL = -dh   (positive downward)
      With z_surf = 0, this implies head h = z_surf - z_GWL = -z_GWL = dh.

    Shapes:
    - dynamic_features: (B, T, 1) = past z_GWL history
    - future_features : (B, T+H, 1) = dummy zeros, but length T+H to satisfy
      any TFT-like "known future" slicing logic (even if unused).
    - coords          : (B, H, 3) = forecast coords at time j (x=y=0 here)
    - H_field         : (B, H, 1) = static thickness tiled across horizon

    Targets:
    - subs_pred: cumulative subsidence s(t=j) (mm in this synthetic generator)
    - gwl_pred : z_GWL(t=j) (m, positive downward)
    """
    years = np.asarray(years, dtype=float)
    dh    = np.asarray(dh, dtype=float)
    s_cum = np.asarray(s_cum, dtype=float)

    T = int(time_steps)
    H = 1
    B = len(years) - T
    if B <= 0:
        raise ValueError("Not enough samples for given time_steps.")

    # Convert synthetic head-change series to GeoPrior-style depth-to-water
    z_gwl = -dh  # (m) positive downward

    X = {
        "static_features": np.repeat(static_vec[None, :], B, axis=0).astype(np.float32),
        "dynamic_features": np.zeros((B, T, 1), dtype=np.float32),
        # IMPORTANT for TFT-like internals: future length should be auto T+H
        "future_features": np.zeros((B, H, 1), dtype=np.float32),

        "coords": np.zeros((B, H, 3), dtype=np.float32),
        "H_field": np.full((B, H, 1), float(H_field_value), dtype=np.float32),
    }

    y = {
        "subs_pred": np.zeros((B, H, 1), dtype=np.float32),
        "gwl_pred":  np.zeros((B, H, 1), dtype=np.float32),
    }

    for i in range(B):
        j = i + T  # forecast index

        # history window [i, i+T)
        X["dynamic_features"][i, :, 0] = z_gwl[i:i + T]

        # forecast targets at time j
        y["subs_pred"][i, 0, 0] = s_cum[j]
        y["gwl_pred"][i, 0, 0]  = z_gwl[j]

        # coords at forecast time (x=y=0 for single pixel)
        X["coords"][i, 0, 0] = years[j]  # t

    return X, y

def tf_dataset(X: Dict[str, np.ndarray], y: Dict[str, np.ndarray],
               batch: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    xb = {k: tf.convert_to_tensor(v, tf.float32) for k, v in X.items()}
    yb = {k: tf.convert_to_tensor(v, tf.float32) for k, v in y.items()}
    ds = tf.data.Dataset.from_tensor_slices((xb, yb))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(next(iter(X.values()))),
                        seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def split_tail(X, y, val_tail: int):
    B = len(next(iter(X.values())))
    if not (0 < val_tail < B):
        raise ValueError("val_tail must be in (0, B).")
    cut = B - val_tail
    Xtr = {k: v[:cut] for k, v in X.items()}
    ytr = {k: v[:cut] for k, v in y.items()}
    Xva = {k: v[cut:] for k, v in X.items()}
    yva = {k: v[cut:] for k, v in y.items()}
    return Xtr, ytr, Xva, yva


def train_one_pixel(
    Xtr, ytr, Xva, yva,
    outdir: str,
    seed: int,
    epochs: int,
    batch: int,
    lr: float,
    kappa_b: float,
    gamma_w: float,
    hd_factor: float,
):
    tf.keras.utils.set_random_seed(seed)

    s_dim = Xtr["static_features"].shape[-1]
    d_dim = Xtr["dynamic_features"].shape[-1]
    f_dim = Xtr["future_features"].shape[-1]

    model = GeoPriorSubsNet(
        static_input_dim=int(s_dim),
        dynamic_input_dim=int(d_dim),
        future_input_dim=int(f_dim),
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
        max_window_size=int(Xtr["dynamic_features"].shape[1]),
        attention_levels=["cross"],

        mv=LearnableMV(initial_value=1e-7),
        kappa=LearnableKappa(initial_value=float(kappa_b), trainable=False),
        gamma_w=FixedGammaW(value=float(gamma_w)),
        h_ref=FixedHRef(value=0.0),
        kappa_mode="kb",
        use_effective_h=True,
        hd_factor=float(hd_factor),
        scale_pde_residuals=False,
        
        pde_mode="consolidation",
        offset_mode="log10",  # optional but good for branch coverage
    
        # IMPORTANT: make units/bounds path non-trivial
        scaling_kwargs=dict(
            # subsidence target is in mm in  SM3 (alpha=1000) -> convert to meters
            subs_scale_si=1e-3,
            subs_bias_si=1.0,
    
            # head is conceptually meters in  synthetic setup
            head_scale_si=1.0,
            head_bias_si=0.0,
    
            # force the per-second conversion branch
            time_units="year",
            
            # --- CRITICAL: tell the model which dynamic channel is GWL/depth ---
            gwl_dyn_index=0,       # <-- because dynamic_features is (B,T,1)
        
            # (optional but robust / self-documenting)
            dynamic_feature_names=["z_GWL"],
            gwl_col="z_GWL",

            # bounds used by _compute_bounds_residual()
            bounds=dict(
                H_min=3.0,
                H_max=80.0,
                logK_min=float(np.log(1e-14)),  # m/s
                logK_max=float(np.log(1e-3)),
                logSs_min=float(np.log(1e-8)),  # 1/m (typical-ish broad range)
                logSs_max=float(np.log(1e-3)),
            ),
        ),
    )



    ds_tr = tf_dataset(Xtr, ytr, batch, True, seed)
    ds_va = tf_dataset(Xva, yva, batch, False, seed)

    # Build once
    for xb, _ in ds_tr.take(1):
        _ = model(xb)
        break

    loss_sub = make_weighted_pinball([0.1, 0.5, 0.9], {0.1: 3.0, 0.5: 1.0, 0.9: 3.0})
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr), clipnorm=1.0),
        loss={"subs_pred": loss_sub, "gwl_pred": tf.keras.losses.MSE},
        loss_weights={"subs_pred": 1.0, "gwl_pred": 1.0},
    
        # Make consolidation physics actually constrain tau
        lambda_cons=1.0,
    
        # No groundwater PDE in 1-pixel SM3
        lambda_gw=0.0,
    
        # Prior helps stabilize closure but don’t over-lock it
        lambda_prior=0.3,
    
        # No spatial fields => smoothness is meaningless here
        lambda_smooth=0.0,
    
        # Optional (small); turn off initially if you want cleaner tau learning
        lambda_mv=0.0,
    
        # Bounds can dominate early in 1-pixel: keep off initially
        lambda_bounds=0.0,
    
        # keep branch coverage if you want, but neutral is fine
        lambda_offset=0.0,
        mv_lr_mult=1.0,
        kappa_lr_mult=0.0,
    )

    os.makedirs(outdir, exist_ok=True)
    ckpt = os.path.join(outdir, "best.keras")
    cbs = [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=10,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(ckpt, "val_loss",
                                           save_best_only=True, verbose=1),
    ]
    model.fit(ds_tr, validation_data=ds_va, epochs=int(epochs), verbose=1,
              callbacks=cbs)
    return model, ds_va


def flatten_diag(diag: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert your SM3 diagnostic dict to a single flat row.
    We use the keys produced by quantiles=(0.5,0.75,0.9,0.95) -> q50,q75,q90,q95.
    """
    row = {}
    row["tau_rel_q50"] = diag["tau_rel_error"]["q50"]
    row["tau_rel_q90"] = diag["tau_rel_error"]["q90"]
    row["tau_rel_q95"] = diag["tau_rel_error"]["q95"]

    row["closure_log_resid_mean"] = diag["closure_log_resid"]["mean"]
    row["closure_log_resid_q95"]  = diag["closure_log_resid"]["q95"]

    for block in ("vs_true", "vs_prior"):
        for key in ("delta_K", "delta_Ss", "delta_Hd"):
            d = diag["offsets"][block][key]
            row[f"{block}_{key}_q50"] = d["q50"]
            row[f"{block}_{key}_q95"] = d["q95"]
    return row

def convert_payload_time_units(
    payload: Dict[str, np.ndarray],
    *,
    from_units: str = "sec",
    to_units: str = "sec",
    sec_per_year: float = SEC_PER_YEAR,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Convert a physics payload between:
      - "sec"  : tau in seconds, K in m/s
      - "year" : tau in years,   K in m/year

    Notes
    -----
    We convert only the *time-dependent physical fields*:
      - tau, tau_prior, tau_closure (if present)
      - K
    Other fields (Ss, Hd, etc.) are left unchanged.
    """
    fu = (from_units or "sec").strip().lower()
    tu = (to_units or "sec").strip().lower()

    if fu not in ("sec", "s", "second", "seconds", "year", "years"):
        raise ValueError(f"Unsupported from_units={from_units!r}")
    if tu not in ("sec", "s", "second", "seconds", "year", "years"):
        raise ValueError(f"Unsupported to_units={to_units!r}")

    fu = "sec" if fu.startswith("s") else "year"
    tu = "sec" if tu.startswith("s") else "year"

    out = dict(payload)
    if fu == tu:
        # Still refresh log fields for safety/consistency.
        if "log10_tau" in out and "tau" in out:
            out["log10_tau"] = np.log10(np.clip(np.asarray(out["tau"], float), eps, None))
        if "log10_tau_prior" in out and "tau_prior" in out:
            out["log10_tau_prior"] = np.log10(np.clip(np.asarray(
                out["tau_prior"], float), eps, None))
        if "log10_tau_closure" in out and "tau_closure" in out:
            out["log10_tau_closure"] = np.log10(np.clip(np.asarray(
                out["tau_closure"], float), eps, None))
        return out

    # --- Conversion factors ---
    # sec -> year : tau /= spy, K *= spy
    # year -> sec : tau *= spy, K /= spy
    spy = float(sec_per_year)

    def _as_float(a):
        return np.asarray(a, float)

    if fu == "sec" and tu == "year":
        if "tau" in out:
            out["tau"] = _as_float(out["tau"]) / spy
        if "tau_prior" in out:
            out["tau_prior"] = _as_float(out["tau_prior"]) / spy
        if "tau_closure" in out:
            out["tau_closure"] = _as_float(out["tau_closure"]) / spy
        if "K" in out:
            out["K"] = _as_float(out["K"]) * spy

    elif fu == "year" and tu == "sec":
        if "tau" in out:
            out["tau"] = _as_float(out["tau"]) * spy
        if "tau_prior" in out:
            out["tau_prior"] = _as_float(out["tau_prior"]) * spy
        if "tau_closure" in out:
            out["tau_closure"] = _as_float(out["tau_closure"]) * spy
        if "K" in out:
            out["K"] = _as_float(out["K"]) / spy

    # Refresh log fields if they exist (useful for plots)
    if "log10_tau" in out and "tau" in out:
        out["log10_tau"] = np.log10(np.clip(
            np.asarray(out["tau"], float), eps, None))
    if "log10_tau_prior" in out and "tau_prior" in out:
        out["log10_tau_prior"] = np.log10(np.clip(
            np.asarray(out["tau_prior"], float), eps, None))
    if "log10_tau_closure" in out and "tau_closure" in out:
        # IMPORTANT: closure must track tau_closure (not tau_prior)
        out["log10_tau_closure"] = np.log10(np.clip(
            np.asarray(out["tau_closure"], float), eps, None))

    return out


def _infer_payload_time_units(meta: Dict[str, Any]) -> str:
    """
    Infer payload units robustly from metadata.

    Priority:
      1) meta["units"]["tau"] if present
      2) meta["payload_time_units"] if you write it
      3) meta["time_units"] fallback
    """
    units = meta.get("units") or {}
    tau_u = str(units.get("tau", "")).lower()
    if "year" in tau_u:
        return "year"
    if "sec" in tau_u or tau_u in ("s", "second", "seconds"):
        return "sec"

    pu = str(meta.get("payload_time_units", "")).strip().lower()
    if pu.startswith("y"):
        return "year"
    if pu.startswith("s"):
        return "sec"

    return str(meta.get("time_units", "year")).strip().lower()


def run_one_realisation(
    *,
    r: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    priors: List[LithoPrior],
    years: np.ndarray,
    outdir: str,
) -> Dict[str, Any]:
    """
    One SM3 synthetic run:
      - sample truth/prior
      - simulate series
      - train 1-pixel GeoPriorSubsNet
      - export payload
      - load payload + run unit-safe diagnostics
      - return a flat row dict for the global CSV
    """
    n_lith = len(priors)

    run_dir = os.path.join(outdir, f"real_{r:03d}")
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Sample lithology + thickness -> H_eff -> Hd_prior
    # ------------------------------------------------------------
    lith_idx = int(rng.integers(0, n_lith))
    lp = priors[lith_idx]

    H_phys = float(rng.uniform(lp.Hmin, lp.Hmax))
    H_eff = float(min(H_phys, args.thickness_cap))
    Hd_prior = float(args.hd_factor * H_eff)

    # ------------------------------------------------------------
    # 2) Ss_prior and tau_prior (years)
    # ------------------------------------------------------------
    Ss_prior = float(lp.Ss_prior)

    logtau_p = float(rng.uniform(np.log10(args.tau_min), np.log10(args.tau_max)))
    tau_prior_year = float(10.0 ** logtau_p)
    tau_prior_sec = tau_prior_year * SEC_PER_YEAR

    # ------------------------------------------------------------
    # 3) K_prior from closure (SI: m/s, consistent with tau in seconds)
    # ------------------------------------------------------------
    K_prior_mps = float(
        (Hd_prior ** 2) * Ss_prior
        / (np.pi ** 2 * float(args.kappa_b) * tau_prior_sec)
    )

    # ------------------------------------------------------------
    # 4) Sample truth (Ss_true, tau_true), set Hd_true, derive K_true
    # ------------------------------------------------------------
    dlogSs = float(rng.normal(0.0, float(args.Ss_spread_dex)))
    Ss_true = float(Ss_prior * (10.0 ** dlogSs))

    logtau_t = float(logtau_p + rng.normal(0.0, float(args.tau_spread_dex)))
    tau_true_year = float(np.clip(10.0 ** logtau_t, args.tau_min, args.tau_max))
    tau_true_sec = tau_true_year * SEC_PER_YEAR

    Hd_true = float(Hd_prior)
    K_true_mps = float(
        (Hd_true ** 2) * Ss_true
        / (np.pi ** 2 * float(args.kappa_b) * tau_true_sec)
    )

    # ------------------------------------------------------------
    # 5) Drawdown Δh(t) and settlement with tau_true
    # ------------------------------------------------------------
    amp = float(rng.uniform(-15.0, -5.0))
    step_year = int(rng.integers(2, max(3, args.n_years // 3)))

    dh = build_load(
        years=years,
        kind=args.load_type,
        step_year=step_year,
        amplitude=amp,
        ramp_years=max(3, args.n_years // 4),
    )

    # alpha_eff controls output magnitude (mm if alpha=1000)
    unit_scale = float(args.alpha)
    alpha_eff = unit_scale * Ss_true * H_eff

    s_cum = settlement_from_tau(years, dh, tau_true_year, alpha=alpha_eff)

    y_inc = np.zeros_like(s_cum)
    y_inc[0] = s_cum[0]
    y_inc[1:] = s_cum[1:] - s_cum[:-1]
    y_inc = y_inc + rng.normal(0.0, args.noise_std, size=y_inc.shape)
    # cumulative observation derived from (noisy) increments
    s_obs = np.cumsum(y_inc)

    # ------------------------------------------------------------
    # 6) Build windows + split
    # ------------------------------------------------------------
    is_capped = float(H_phys > args.thickness_cap)
    static_vec = np.concatenate(
        [one_hot(lith_idx, n_lith), np.array([H_eff, is_capped], np.float32)],
        axis=0
    ).astype(np.float32)

    X, y = make_one_step_windows(
        years=years,
        dh=dh,
        s_cum=s_obs,     # cumulative observations
        static_vec=static_vec,
        time_steps=args.time_steps,
        H_field_value=H_eff,
    )


    Xtr, ytr, Xva, yva = split_tail(X, y, args.val_tail)

    # ------------------------------------------------------------
    # 7) Train
    # ------------------------------------------------------------
    model, ds_va = train_one_pixel(
        Xtr, ytr, Xva, yva,
        outdir=run_dir,
        seed=int(args.seed + r),
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        kappa_b=args.kappa_b,
        gamma_w=args.gamma_w,
        hd_factor=args.hd_factor,
    )

    # ------------------------------------------------------------
    # 8) Export physics payload (+ truth/prior metadata)
    # ------------------------------------------------------------
    npz_path = os.path.join(run_dir, "phys_payload_val.npz")

    # IMPORTANT:
    # Do NOT claim payload is "sec" if model exports "year".
    # We keep report_time_units="year" for SM3.
    model.export_physics_payload(
        ds_va,
        save_path=npz_path,
        format="npz",
        overwrite=True,
        metadata={
            "synthetic": True,
            "realisation": int(r),
            
            "payload_time_units": "sec",
            # "units": {
            #     "tau": "sec",
            #     "K": "m/s",
            # },

            "report_time_units": "year",
            "sec_per_year": float(SEC_PER_YEAR),
            "kappa_b": float(args.kappa_b),

            # truth (SI + year)
            "tau_true_sec": float(tau_true_sec),
            "tau_true_year": float(tau_true_year),
            "K_true_mps": float(K_true_mps),
            "Ss_true": float(Ss_true),
            "Hd_true": float(Hd_true),

            # priors (SI + year)
            "tau_prior_sec": float(tau_prior_sec),
            "tau_prior_year": float(tau_prior_year),
            "K_prior_mps": float(K_prior_mps),
            "Ss_prior": float(Ss_prior),
            "Hd_prior": float(Hd_prior),

            # thickness bookkeeping
            "H_eff": float(H_eff),
            "H_phys": float(H_phys),

            # load/noise
            "load_type": str(args.load_type),
            "amp_dh": float(amp),
            "step_year": int(step_year),
            "noise_std": float(args.noise_std),
        },
    )

    # ------------------------------------------------------------
    # 9) Load payload + run unit-safe SM3 diagnostics
    # ------------------------------------------------------------
    payload_raw, meta = load_physics_payload(npz_path)

    payload_units = _infer_payload_time_units(meta)     # now correctly "sec"
    report_units  = meta.get("report_time_units", payload_units)
    spy = float(meta.get("sec_per_year", SEC_PER_YEAR))

    # Convert payload into report units for diagnostics
    payload_u = convert_payload_time_units(
        payload_raw, from_units=payload_units, 
        to_units=report_units, sec_per_year=spy
    )

    # Convert truth/prior to the SAME report units
    if report_units == "year":
        tau_true_u = float(meta["tau_true_year"])
        tau_prior_u = float(meta["tau_prior_year"])
        K_true_u = float(meta["K_true_mps"]) * spy      # m/year
        K_prior_u = float(meta["K_prior_mps"]) * spy    # m/year
    elif report_units == "sec":
        tau_true_u = float(meta["tau_true_sec"])
        tau_prior_u = float(meta["tau_prior_sec"])
        K_true_u = float(meta["K_true_mps"])            # m/s
        K_prior_u = float(meta["K_prior_mps"])          # m/s
    else:
        raise ValueError(f"Unsupported report_units={report_units!r}")

    diag = identifiability_diagnostics_from_payload(
        payload_u,
        tau_true=tau_true_u,
        K_true=K_true_u,
        Ss_true=float(meta["Ss_true"]),
        Hd_true=float(meta["Hd_true"]),
        K_prior=K_prior_u,
        Ss_prior=float(meta["Ss_prior"]),
        Hd_prior=float(meta["Hd_prior"]),
    )

    with open(os.path.join(run_dir, "sm3_identifiability_diag.json"), "w",
              encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    # For debugging: always keep both year and sec summaries (consistent)
    payload_year = convert_payload_time_units(
        payload_raw, from_units=payload_units, to_units="year", 
        sec_per_year=spy)
    payload_sec = convert_payload_time_units(
        payload_raw, from_units=payload_units, to_units="sec", 
        sec_per_year=spy)

    eff_year = summarise_effective_params(payload_year)
    eff_sec = summarise_effective_params(payload_sec)

    # ------------------------------------------------------------
    # 10) Build one CSV row
    # ------------------------------------------------------------
    row = {
        "realisation": int(r),
        "lith_idx": int(lith_idx),

        "tau_true_year": float(tau_true_year),
        "tau_prior_year": float(tau_prior_year),
        "tau_true_sec": float(tau_true_sec),
        "tau_prior_sec": float(tau_prior_sec),

        "K_true_mps": float(K_true_mps),
        "K_prior_mps": float(K_prior_mps),

        "Ss_true": float(Ss_true),
        "Ss_prior": float(Ss_prior),
        "Hd_true": float(Hd_true),
        "Hd_prior": float(Hd_prior),

        # estimates (always provide both unit systems)
        "tau_est_med_year": float(eff_year["tau"]),
        "K_est_med_m_per_year": float(eff_year["K"]),
        "Ss_est_med": float(eff_year["Ss"]),
        "Hd_est_med": float(eff_year["Hd"]),

        "tau_est_med_sec": float(eff_sec["tau"]),
        "K_est_med_mps": float(eff_sec["K"]),

        "kappa_b": float(args.kappa_b),
    }

    row.update(flatten_diag(diag))
    return row


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    """
    Driver for the full SM3 experiment:
      - loops over realisations
      - collects rows
      - writes runs CSV + summary CSV
      - returns the runs dataframe (useful for interactive debugging)
    """
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    priors = litho_priors()
    years = np.arange(args.n_years, dtype=float)

    rows: List[Dict[str, Any]] = []

    for r in range(args.n_realizations):
        print("=" * 72)
        print(f"Realisation {r+1:03d}/{args.n_realizations:03d}")
        print("=" * 72)

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

    runs_csv = os.path.join(args.outdir, "sm3_synth_runs.csv")
    df.to_csv(runs_csv, index=False)

    # Summary stats over numeric columns
    metrics = [
        c for c in df.columns
        if c not in ("realisation", "lith_idx")
        and np.issubdtype(df[c].dtype, np.number)
    ]

    summ_rows: List[Dict[str, Any]] = []
    for c in metrics:
        x = df[c].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        summ_rows.append({
            "metric": c,
            "mean": float(np.mean(x)) if x.size else float("nan"),
            "std":  float(np.std(x))  if x.size else float("nan"),
            "p05":  float(np.quantile(x, 0.05)) if x.size else float("nan"),
            "p50":  float(np.quantile(x, 0.50)) if x.size else float("nan"),
            "p95":  float(np.quantile(x, 0.95)) if x.size else float("nan"),
        })

    df_sum = pd.DataFrame(summ_rows)
    sum_csv = os.path.join(args.outdir, "sm3_synth_summary.csv")
    df_sum.to_csv(sum_csv, index=False)

    print("[OK] wrote:", runs_csv)
    print("[OK] wrote:", sum_csv)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()

    # Output / experiment size
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-realizations", type=int, default=30)
    ap.add_argument("--n-years", type=int, default=20)
    ap.add_argument("--time-steps", type=int, default=5)
    ap.add_argument("--val-tail", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)

    # Synthetic data controls
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--load-type", choices=("step", "ramp"), default="step")
    ap.add_argument("--alpha", type=float, default=1000.0)

    # GeoPrior physics knobs
    ap.add_argument("--hd-factor", type=float, default=0.6)
    ap.add_argument("--thickness-cap", type=float, default=30.0)
    ap.add_argument("--kappa-b", type=float, default=1.0)
    ap.add_argument("--gamma-w", type=float, default=9810.0)

    # Timescale sampling
    ap.add_argument("--tau-min", type=float, default=0.3)
    ap.add_argument("--tau-max", type=float, default=10.0)
    ap.add_argument("--tau-spread-dex", type=float, default=0.3)
    ap.add_argument("--Ss-spread-dex", type=float, default=0.4)

    args = ap.parse_args()
    if not (args.tau_min > 0.0 and args.tau_max > args.tau_min):
        raise ValueError("--tau-min must be > 0 and < --tau-max.")
    if not (0 < args.val_tail < (args.n_years - args.time_steps)):
        raise ValueError("--val-tail must be in (0, B) with B=n_years-time_steps.")

    _ = run_experiment(args)


if __name__ == "__main__":
    main()
