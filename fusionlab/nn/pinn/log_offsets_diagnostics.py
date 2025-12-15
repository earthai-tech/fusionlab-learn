# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>


from __future__ import annotations

import os
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _pick_first(d: Dict[str, Any], *candidates: str) -> Optional[str]:
    """Return first key present in dict from a list of candidates."""
    for k in candidates:
        if k in d:
            return k
    return None


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load a .npz file into a dict of arrays."""
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _compute_offsets_table(
    payload: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Build a tidy table with log-offsets from the physics payload.

    This is intentionally defensive: it supports several possible
    naming conventions and only computes offsets for pairs that exist.

    Expected (but not strictly required) keys:
        - K_prior, K_eff
        - Ss_prior, Ss_eff
        - Hd_prior, Hd_eff
        - tau_prior, tau_eff

    Returns
    -------
    df : pandas.DataFrame
        Columns may include:
        ['delta_logK', 'delta_logSs', 'delta_logHd',
         'delta_log_tau', 'log_tau_prior', 'log_tau_eff',
         'depth', 'depth_index'].
    """
    keys = set(payload.keys())

    # Try to find depth / vertical coordinate (optional but nice)
    depth_key = _pick_first(
        payload,
        "depth",
        "z",
        "coord_z",
        "z_mid",
        "cell_center_z",
        "z_index",
    )

    # Utility to grab an array, or None if missing
    def arr(key_candidates: List[str]) -> Optional[np.ndarray]:
        k = _pick_first(payload, *key_candidates)
        return None if k is None else np.asarray(payload[k])

    # Candidate names for each physical field
    K_prior = arr(["K_prior", "k_prior", "K_lith_prior"])
    K_eff = arr(["K_eff", "K_effective", "K_geo", "K_posterior"])
    Ss_prior = arr(["Ss_prior", "Ss_lith_prior", "Ss0"])
    Ss_eff = arr(["Ss_eff", "Ss_effective", "Ss_geo", "Ss_posterior"])
    Hd_prior = arr(["Hd_prior", "H_d_prior", "Hd_lith_prior"])
    Hd_eff = arr(["Hd_eff", "H_d_effective", "Hd_geo", "Hd_posterior"])

    tau_prior = arr(["tau_prior", "timescale_prior", "tau0"])
    tau_eff = arr(["tau_eff", "tau_effective", "tau_geoprior"])

    depth = None
    if depth_key is not None:
        depth = np.asarray(payload[depth_key])

    # Determine base size and flatten
    base_size = None

    def _check_and_flatten(x: np.ndarray, name: str) -> np.ndarray:
        nonlocal base_size
        x_flat = x.ravel()
        if base_size is None:
            base_size = x_flat.size
        elif x_flat.size != base_size:
            raise ValueError(
                f"Inconsistent size for {name}: {x_flat.size} vs {base_size}"
            )
        return x_flat

    cols: Dict[str, np.ndarray] = {}

    if (K_prior is not None) and (K_eff is not None):
        delta_logK = np.log10(K_eff) - np.log10(K_prior)
        cols["delta_logK"] = _check_and_flatten(delta_logK, "delta_logK")

    if (Ss_prior is not None) and (Ss_eff is not None):
        delta_logSs = np.log10(Ss_eff) - np.log10(Ss_prior)
        cols["delta_logSs"] = _check_and_flatten(delta_logSs, "delta_logSs")

    if (Hd_prior is not None) and (Hd_eff is not None):
        # Hd is already in linear space; we use log10 for consistency
        delta_logHd = np.log10(Hd_eff) - np.log10(Hd_prior)
        cols["delta_logHd"] = _check_and_flatten(delta_logHd, "delta_logHd")

    if (tau_prior is not None) and (tau_eff is not None):
        log_tau_prior = np.log10(tau_prior)
        log_tau_eff = np.log10(tau_eff)
        delta_log_tau = log_tau_eff - log_tau_prior

        cols["log_tau_prior"] = _check_and_flatten(
            log_tau_prior, "log_tau_prior"
        )
        cols["log_tau_eff"] = _check_and_flatten(
            log_tau_eff, "log_tau_eff"
        )
        cols["delta_log_tau"] = _check_and_flatten(
            delta_log_tau, "delta_log_tau"
        )

    if not cols:
        raise RuntimeError(
            "No matching K/Ss/Hd/tau pairs found in physics payload; "
            f"available keys: {sorted(keys)}"
        )

    # Optional depth
    if depth is not None:
        depth_flat = depth.ravel()
        if depth_flat.size == base_size:
            cols["depth"] = depth_flat
        else:
            # Fall back to an index if depth has different topology
            cols["depth"] = np.linspace(0.0, 1.0, base_size)

    # Always include an index
    cols["depth_index"] = np.arange(base_size, dtype=int)

    df = pd.DataFrame(cols)
    return df


def _summarize_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise offsets: mean, std, p05, p50, p95 for each δ-column.
    """
    delta_cols = [
        c
        for c in df.columns
        if c.startswith("delta_log") or c == "delta_log_tau"
    ]
    if not delta_cols:
        raise RuntimeError("No δ-columns found in DataFrame.")

    desc = df[delta_cols].describe(
        percentiles=[0.05, 0.5, 0.95]
    ).T
    desc = desc.rename(
        columns={
            "mean": "mean",
            "std": "std",
            "5%": "p05",
            "50%": "p50",
            "95%": "p95",
        }
    )
    # Keep only the relevant columns
    cols = ["mean", "std", "p05", "p50", "p95"]
    desc = desc[cols]
    desc.index.name = "metric"
    return desc


def _make_plots(
    df: pd.DataFrame,
    outdir: str,
    prefix: str = "sm3_offsets",
) -> List[str]:
    """
    Produce simple histograms and τ–δτ scatter plot.
    Returns a list of saved file paths.
    """
    saved: List[str] = []
    _ensure_dir(outdir)

    # Histograms
    for col in ["delta_logK", "delta_logSs", "delta_logHd", "delta_log_tau"]:
        if col not in df.columns:
            continue

        plt.figure()
        df[col].dropna().hist(bins=50)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        fname = os.path.join(outdir, f"{prefix}_{col}_hist.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        saved.append(fname)

    # τ–δτ scatter (prior vs offset)
    if {"log_tau_prior", "delta_log_tau"}.issubset(df.columns):
        plt.figure()
        plt.scatter(
            df["log_tau_prior"],
            df["delta_log_tau"],
            s=4,
            alpha=0.3,
        )
        plt.axhline(0.0, linestyle="--")
        plt.xlabel(r"$\log_{10}\tau_{\mathrm{prior}}$")
        plt.ylabel(r"$\delta_{\tau} = \log_{10}\tau_{\mathrm{eff}} - "
                   r"\log_{10}\tau_{\mathrm{prior}}$")
        plt.title("Timescale prior vs log-offset (SM3)")
        fname = os.path.join(outdir, f"{prefix}_tau_scatter.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        saved.append(fname)

    return saved


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def run_sm3_offsets_from_payload(
    physics_npz_path: str,
    outdir: Optional[str] = None,
    city: Optional[str] = None,
    model_name: str = "GeoPriorSubsNet",
) -> Dict[str, Any]:
    """
    High-level driver: compute SM3 diagnostics from a physics payload.

    Parameters
    ----------
    physics_npz_path : str
        Path to ``*_phys_payload_run_val.npz`` as written by
        :meth:`GeoPriorSubsNet.export_physics_payload`.
    outdir : str, optional
        Directory where CSVs and plots are written. If ``None``,
        defaults to the directory of ``physics_npz_path``.
    city : str, optional
        City name for filenames.
    model_name : str, default "GeoPriorSubsNet"
        Model name for filenames.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'raw_csv'
        - 'summary_csv'
        - 'plots' (list of paths)
    """
    physics_npz_path = os.path.abspath(physics_npz_path)
    if outdir is None:
        outdir = os.path.dirname(physics_npz_path)
    _ensure_dir(outdir)

    tag = []
    if city:
        tag.append(city.lower())
    if model_name:
        tag.append(model_name.lower())
    tag_str = "_".join(tag) if tag else "geoprior"

    print(f"[SM3] Loading physics payload: {physics_npz_path}")
    payload = _load_npz(physics_npz_path)
    print(f"[SM3] Keys in physics payload: {sorted(payload.keys())}")

    df = _compute_offsets_table(payload)
    print(f"[SM3] Built offsets table with shape {df.shape}.")

    raw_csv = os.path.join(outdir, f"{tag_str}_sm3_offsets_raw.csv")
    df.to_csv(raw_csv, index=False)
    print(f"[SM3] Saved raw offsets CSV -> {raw_csv}")

    summary_df = _summarize_offsets(df)
    summary_csv = os.path.join(outdir, f"{tag_str}_sm3_offsets_summary.csv")
    summary_df.to_csv(summary_csv)
    print(f"[SM3] Saved summary offsets CSV -> {summary_csv}")

    plots = _make_plots(df, outdir=outdir, prefix=f"{tag_str}_sm3")
    for p in plots:
        print(f"[SM3] Saved plot -> {p}")

    return {
        "raw_csv": raw_csv,
        "summary_csv": summary_csv,
        "plots": plots,
    }

