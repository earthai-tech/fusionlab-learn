# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

# $ python nat.com/sm3_synthetic_identifiability.py 
# --outdir results/sm3_synth --n-realizations 50 --n-years 20 
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
    y_inc: np.ndarray,
    static_vec: np.ndarray,
    time_steps: int,
    H_field_value: float,   
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

    """
    1-step-ahead windows so we don't need future drivers.
    Inputs match the model/evaluate_physics convention:
      static_features, dynamic_features, future_features, coords
    """
    years = np.asarray(years, dtype=float)
    dh = np.asarray(dh, dtype=float)
    y_inc = np.asarray(y_inc, dtype=float)

    T = int(time_steps)
    H = 1
    B = len(years) - T
    if B <= 0:
        raise ValueError("Not enough samples for given time_steps.")

    X = {
        "static_features": np.repeat(static_vec[None, :], B, axis=0).astype(np.float32),
        "dynamic_features": np.zeros((B, T, 1), dtype=np.float32),
        "future_features": np.zeros((B, H, 1), dtype=np.float32),
        "coords": np.zeros((B, H, 3), dtype=np.float32),          # was (B, T+H, 3)
        "H_field": np.full((B, H, 1), float(H_field_value), dtype=np.float32),  
    }

    y = {
        "subs_pred": np.zeros((B, H, 1), dtype=np.float32),
        "gwl_pred":  np.zeros((B, H, 1), dtype=np.float32),
    }

    for i in range(B):
        X["dynamic_features"][i, :, 0] = dh[i:i + T]
        j = i + T
        y["subs_pred"][i, 0, 0] = y_inc[j]
        X["coords"][i, 0, 0] = years[j]   # horizon time (H=1)
        # x,y stay 0


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
        # kappa=LearnableKappa(initial_value=float(kappa_b)),
        kappa=LearnableKappa(initial_value=float(kappa_b), trainable=False),
        gamma_w=FixedGammaW(value=float(gamma_w)),
        h_ref=FixedHRef(value=0.0),
        kappa_mode="kb",
        use_effective_h=True,
        hd_factor=float(hd_factor),
        scale_pde_residuals=True,
        

        pde_mode="consolidation",
        offset_mode="log10",  # optional but good for branch coverage
    
        # IMPORTANT: make units/bounds path non-trivial
        scaling_kwargs=dict(
            # subsidence target is in mm in  SM3 (alpha=1000) -> convert to meters
            subs_scale_si=1e-3,
            subs_bias_si=0.0,
    
            # head is conceptually meters in  synthetic setup
            head_scale_si=1.0,
            head_bias_si=0.0,
    
            # force the per-second conversion branch
            time_units="year",
    
            # bounds used by _compute_bounds_residual()
            bounds=dict(
                H_min=3.0,
                H_max=80.0,
                logK_min=float(np.log(1e-10)),  # m/s
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
        loss_weights={"subs_pred": 1.0, "gwl_pred": 0.0},
    
        lambda_cons=0.1,
        lambda_gw=0.0,
        lambda_prior=0.1,
        lambda_smooth=0.01,
        lambda_mv=0.01,
    
        lambda_bounds=0.05,      
        # lambda_offset=1.0,       
        lambda_offset=0.0,    # 10^0 = 1 (neutral), but exercises log10 branch
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

def main():
    """
    Run the Synthetic Identifiability Experiment (Supplementary Methods 3).

    Key design (fixed):
    - Avoid degenerate/clamped tau by sampling tau_prior in a resolvable range.
    - Enforce *truth consistency* by back-computing K from the closure:
          tau = Hd^2 * Ss / (pi^2 * kappa_b * K)
      so the parameters recorded as "true" are exactly those that generated
      the synthetic series.
    - If the current GeoPrior physics head does not learn Hd independently,
      set Hd_true = Hd_prior (otherwise the experiment is unfair by design).

    Workflow per realisation r:
      1) Sample lithology and thickness H_phys -> H_eff -> Hd_prior.
      2) Take Ss_prior from lithology; sample tau_prior (log-uniform).
      3) Compute K_prior from (Hd_prior, Ss_prior, tau_prior).
      4) Sample Ss_true (log-offset around Ss_prior), sample tau_true
         (log-offset around tau_prior), clip tau_true to [tau_min, tau_max],
         set Hd_true = Hd_prior, then compute K_true from closure.
      5) Generate drawdown history Δh(t) and simulate settlement with tau_true.
      6) Build 1-step windows, train a 1-pixel GeoPriorSubsNet.
      7) Export physics payload, run SM3 diagnostics, save per-run JSON + CSV.
    """
    ap = argparse.ArgumentParser()

    # --------------------------
    # Output / experiment size
    # --------------------------
    ap.add_argument("--outdir", required=True,
                    help="Root output folder. A subfolder real_XXX is created per run.")
    ap.add_argument("--n-realizations", type=int, default=30,
                    help="Number of independent synthetic runs (different seeds/loads/noise).")
    ap.add_argument("--n-years", type=int, default=20,
                    help="Length of synthetic annual time series.")
    ap.add_argument("--time-steps", type=int, default=5,
                    help="Encoder window length T (years).")
    ap.add_argument("--val-tail", type=int, default=5,
                    help="Hold-out tail length (years) for validation/evaluation.")
    ap.add_argument("--seed", type=int, default=123,
                    help="Base RNG seed (each realisation adds +r).")

    # --------------------------
    # Training hyperparameters
    # --------------------------
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)

    # --------------------------
    # Synthetic data controls
    # --------------------------
    ap.add_argument("--noise-std", type=float, default=0.02,
                    help="Std dev of additive Gaussian noise on increments (same unit as y_inc).")
    ap.add_argument("--load-type", choices=("step", "ramp"), default="step",
                    help="Drawdown history type Δh(t).")
    # ap.add_argument("--alpha", type=float, default=1.0,
    #                 help="Settlement scaling in settlement_from_tau (dimensionless).")
    ap.add_argument("--alpha", type=float, default=1000.0,
                    help="Unit scale applied to Ss*Δh*H (e.g., 1000 for mm).")
    # If you set alpha=1000 (mm), keep noise_std like 0.02–0.2 (mm).
    # If you keep alpha=1 (meters), use noise_std like 2e-5–2e-4 (m) (i.e., 0.02–0.2 mm).

    # --------------------------
    # GeoPrior physics knobs
    # --------------------------
    ap.add_argument("--hd-factor", type=float, default=0.6,
                    help="Hd_prior = hd_factor * H_eff (effective drainage path fraction).")
    ap.add_argument("--thickness-cap", type=float, default=30.0,
                    help="Cap for H_eff = min(H_phys, thickness_cap).")
    ap.add_argument("--kappa-b", type=float, default=1.0,
                    help="Anisotropy/leakage factor κ_b (k_v = κ_b K).")
    ap.add_argument("--gamma-w", type=float, default=9810.0,
                    help="Unit weight of water (kept for model signature; not used in synthetic forward).")

    # --------------------------
    # Timescale sampling (core SM3 fix)
    # --------------------------
    ap.add_argument("--tau-min", type=float, default=0.3,
                    help="Minimum sampled timescale (years).")
    ap.add_argument("--tau-max", type=float, default=10.0,
                    help="Maximum sampled timescale (years).")
    ap.add_argument("--tau-spread-dex", type=float, default=0.3,
                    help="Std dev (dex) for log10(tau_true) around log10(tau_prior).")

    # Optional: truth spread for Ss (dex via log10)
    ap.add_argument("--Ss-spread-dex", type=float, default=0.4,
                    help="Std dev (dex) for log10(Ss_true) around log10(Ss_prior).")

    args = ap.parse_args()

    if not (args.tau_min > 0.0 and args.tau_max > args.tau_min):
        raise ValueError("--tau-min must be > 0 and < --tau-max.")

    os.makedirs(args.outdir, exist_ok=True)

    # RNG (reproducible)
    rng = np.random.default_rng(args.seed)

    # Lithology priors: we keep Ss_prior and thickness ranges from here.
    # NOTE: lp.K_prior (m/s) is no longer used to *drive* tau; K is derived
    # from sampled tau to ensure timescale diversity and consistency.
    priors = litho_priors()
    n_lith = len(priors)

    # Annual times in years
    years = np.arange(args.n_years, dtype=float)

    rows: List[Dict[str, Any]] = []

    for r in range(args.n_realizations):
        run_dir = os.path.join(args.outdir, f"real_{r:03d}")
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
        # 2) Set Ss_prior (internal unit) and sample tau_prior (years)
        # ------------------------------------------------------------
        Ss_prior = float(lp.Ss_prior)

        logtau_p = float(rng.uniform(np.log10(args.tau_min), np.log10(args.tau_max)))
        
        # tau_prior is sampled in YEARS (for the simulator)
        tau_prior_year = float(10.0 ** logtau_p)
        
        # SI version used by the model when time_units="year"
        tau_prior_sec = tau_prior_year * SEC_PER_YEAR

        # ------------------------------------------------------------
        # 3) Compute K_prior from closure (SI: m/s)
        #    tau = Hd^2 * Ss / (pi^2 * kappa_b * K)
        # -> K = Hd^2 * Ss / (pi^2 * kappa_b * tau)
        # ------------------------------------------------------------
        
        # K in SI (m/s), consistent with tau in seconds
        K_prior_mps = float(
            (Hd_prior ** 2) * Ss_prior
            / (np.pi ** 2 * float(args.kappa_b) * tau_prior_sec)
        )

        # ------------------------------------------------------------
        # 4) Sample truth: Ss_true, tau_true; set Hd_true; derive K_true
        # ------------------------------------------------------------
        dlogSs = float(rng.normal(0.0, float(args.Ss_spread_dex)))
        Ss_true = float(Ss_prior * (10.0 ** dlogSs))

        logtau_t = float(logtau_p + rng.normal(0.0, float(args.tau_spread_dex)))
        tau_true = float(10.0 ** logtau_t)
        
        tau_true_year = float(np.clip(tau_true, args.tau_min, args.tau_max))
        tau_true_sec  = tau_true_year * SEC_PER_YEAR
        
        Hd_true = float(Hd_prior)
        
        K_true_mps = float(
            (Hd_true ** 2) * Ss_true
            / (np.pi ** 2 * float(args.kappa_b) * tau_true_sec)
        )

        # ------------------------------------------------------------
        # 5) Synthetic drawdown Δh(t) and settlement with tau_true
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
        # Δh is in meters, Ss_true is your effective storage scale,
        # H_eff is thickness (m). unit_scale turns meters -> mm if desired.
        unit_scale = float(args.alpha)  # interpret --alpha as "unit scale"
        alpha_eff = unit_scale * Ss_true * H_eff
        
        s_cum = settlement_from_tau(years, dh, tau_true_year, alpha=alpha_eff)

        y_inc = np.zeros_like(s_cum)
        y_inc[0] = s_cum[0]
        y_inc[1:] = s_cum[1:] - s_cum[:-1]
        y_inc = y_inc + rng.normal(0.0, args.noise_std, size=y_inc.shape)

        # ------------------------------------------------------------
        # 6) Build windows + split
        # ------------------------------------------------------------
        # static_vec = np.concatenate(
        #     [one_hot(lith_idx, n_lith), np.array([H_eff], np.float32)],
        #     axis=0
        # ).astype(np.float32)
        is_capped = float(H_phys > args.thickness_cap)
        static_vec = np.concatenate(
            [one_hot(lith_idx, n_lith),
             np.array([H_eff, is_capped], np.float32)],
            axis=0
        ).astype(np.float32)

        X, y = make_one_step_windows(
            years=years,
            dh=dh,
            y_inc=y_inc,
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
        # 8) Export physics payload + metadata
        # ------------------------------------------------------------
        npz_path = os.path.join(run_dir, "phys_payload_val.npz")

        model.export_physics_payload(
            ds_va,
            save_path=npz_path,
            format="npz",
            overwrite=True,
            metadata={
                "synthetic": True,
                "realisation": int(r),
            
                # --- ground truth (SI for units-aware physics) ---
                "tau_true_sec": float(tau_true_sec),     # seconds
                "K_true_mps": float(K_true_mps),         # m/s
                "Ss_true": float(Ss_true),
                "Hd_true": float(Hd_true),               # m
            
                # --- priors (SI) ---
                "tau_prior_sec": float(tau_prior_sec),   # seconds
                "K_prior_mps": float(K_prior_mps),       # m/s
                "Ss_prior": float(Ss_prior),
                "Hd_prior": float(Hd_prior),
            
                # optional: keep the “human readable” year versions too
                "tau_true_year": float(tau_true_year),
                "tau_prior_year": float(tau_prior_year),
            
                # thickness bookkeeping
                "H_eff": float(H_eff),
                "H_phys": float(H_phys),
            
                # load / noise settings
                "load_type": str(args.load_type),
                "amp_dh": float(amp),
                "step_year": int(step_year),
                "noise_std": float(args.noise_std),
            }, 

        )

        # ------------------------------------------------------------
        # 9) Load payload and compute SM3 diagnostics
        # ------------------------------------------------------------
        payload, meta = load_physics_payload(npz_path)

        diag = identifiability_diagnostics_from_payload(
            payload,
            tau_true=tau_true_sec,
            K_true=K_true_mps, Ss_true=Ss_true, Hd_true=Hd_true,
            K_prior=K_prior_mps, Ss_prior=Ss_prior, Hd_prior=Hd_prior,
        )

        with open(os.path.join(run_dir, "sm3_identifiability_diag.json"),
                  "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)

        eff = summarise_effective_params(payload)

        # ------------------------------------------------------------
        # 10) Row for global CSV
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
        
            "tau_est_med": float(eff["tau"]),        # should now be seconds if model is consistent
            "K_est_med": float(eff["K"]),            # should now be m/s
            "Ss_est_med": float(eff["Ss"]),
            "Hd_est_med": float(eff["Hd"]),
        }
        row["kappa_b"] = float(args.kappa_b)
        
        row.update(flatten_diag(diag))
        rows.append(row)

    # --------------------------
    # Save per-run table
    # --------------------------
    df = pd.DataFrame(rows)
    runs_csv = os.path.join(args.outdir, "sm3_synth_runs.csv")
    df.to_csv(runs_csv, index=False)

    # --------------------------
    # Save summary table
    # --------------------------
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




if __name__ == "__main__":
    main()
