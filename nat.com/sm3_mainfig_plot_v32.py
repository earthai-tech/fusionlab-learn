# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D


SEC_PER_YEAR = 365.25 * 24.0 * 3600.0


def set_figure_style(
    *,
    fontsize: int = 8,
    dpi: int = 600,
) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _require_cols(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    ctx: str = "",
) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        m = ", ".join(miss)
        raise KeyError(f"Missing cols ({ctx}): {m}")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _linfit_stats(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    slope, inter = np.polyfit(x[m], y[m], 1)
    yhat = inter + slope * x[m]
    ss_res = float(np.nansum((y[m] - yhat) ** 2))
    ss_tot = float(np.nansum((y[m] - np.nanmean(y[m])) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return float(slope), float(inter), float(r2)


def plot_sm3_identifiability_v32(
    csv_path: str,
    outpath: str,
    *,
    show_prior: bool = True,
    tau_units: str = "year",
    metric: str = "ridge_resid",
) -> None:
    """
    metric:
      - "ridge_resid" (recommended for identifiability)
      - "eps_prior_rms"
      - "closure_consistency_rms"
    """
    df = pd.read_csv(csv_path)

    tau_units = str(tau_units).strip().lower()
    if tau_units not in ("year", "sec"):
        raise ValueError("tau_units must be year/sec.")

    base = [
        "lith_idx",
        "kappa_b",
        "tau_true_sec",
        "tau_prior_sec",
        "tau_true_year",
        "tau_prior_year",
        "tau_est_med_sec",
        "tau_est_med_year",
        "K_true_mps",
        "K_prior_mps",
        "K_est_med_mps",
        "K_est_med_m_per_year",
        "Ss_true",
        "Ss_prior",
        "Ss_est_med",
        "Hd_true",
        "Hd_prior",
        "Hd_est_med",
        "vs_true_delta_K_q50",
        "vs_true_delta_Ss_q50",
        "vs_true_delta_Hd_q50",
        "eps_prior_rms",
        "closure_consistency_rms",
    ]
    _require_cols(df, base, ctx="v3.2")

    if metric not in (
        "ridge_resid",
        "eps_prior_rms",
        "closure_consistency_rms",
    ):
        raise ValueError("metric not supported.")

    ln10 = float(np.log(10.0))
    eps = 1e-12

    # -----------------
    # (a) tau recovery
    # -----------------
    if tau_units == "year":
        t_true = np.clip(df["tau_true_year"], eps, None)
        t_est = np.clip(df["tau_est_med_year"], eps, None)
        t_pri = np.clip(df["tau_prior_year"], eps, None)
        xlab = r"$\log_{10}\,\tau_{\mathrm{true}}$ (yr)"
        ylab = r"$\log_{10}\,\hat{\tau}$ (yr)"
    else:
        t_true = np.clip(df["tau_true_sec"], eps, None)
        t_est = np.clip(df["tau_est_med_sec"], eps, None)
        t_pri = np.clip(df["tau_prior_sec"], eps, None)
        xlab = r"$\log_{10}\,\tau_{\mathrm{true}}$ (s)"
        ylab = r"$\log_{10}\,\hat{\tau}$ (s)"

    x_tau = np.log10(t_true.to_numpy(float))
    y_tau = np.log10(t_est.to_numpy(float))

    # closure tau from inferred K,Ss,Hd
    kappa = np.clip(df["kappa_b"].to_numpy(float), eps, None)
    K_est = np.clip(df["K_est_med_mps"].to_numpy(float), eps, None)
    Ss_est = np.clip(df["Ss_est_med"].to_numpy(float), eps, None)
    Hd_est = np.clip(df["Hd_est_med"].to_numpy(float), eps, None)

    tau_cl = (Hd_est**2) * Ss_est / (np.pi**2 * kappa * K_est)
    tau_cl = np.clip(tau_cl, eps, None)

    if tau_units == "year":
        y_cl = np.log10(tau_cl / SEC_PER_YEAR)
        y_pr = np.log10(t_pri.to_numpy(float))
    else:
        y_cl = np.log10(tau_cl)
        y_pr = np.log10(t_pri.to_numpy(float))

    mae_tau = float(np.nanmean(np.abs(y_tau - x_tau)))
    slope_tau, _, r2_tau = _linfit_stats(x_tau, y_tau)

    # -----------------
    # (b) K recovery (m/yr)
    # -----------------
    K_true = np.clip(df["K_true_mps"].to_numpy(float), eps, None)
    x_K = np.log10(K_true * SEC_PER_YEAR)
    y_K = np.log10(
        np.clip(
            df["K_est_med_m_per_year"].to_numpy(float),
            eps,
            None,
        )
    )
    mae_K = float(np.nanmean(np.abs(y_K - x_K)))
    slope_K, _, r2_K = _linfit_stats(x_K, y_K)

    # -----------------
    # (c) ridge plot (degeneracy direction)
    # offsets are ln => convert to log10
    # -----------------
    dK = df["vs_true_delta_K_q50"].to_numpy(float) / ln10
    dSs = df["vs_true_delta_Ss_q50"].to_numpy(float) / ln10
    dHd = df["vs_true_delta_Hd_q50"].to_numpy(float) / ln10

    ridge_x = dSs + 2.0 * dHd
    ridge_y = dK
    ridge_abs = np.abs(ridge_y - ridge_x)

    rho_ridge = _spearman(ridge_x, ridge_y)
    ridge_q50 = float(np.nanquantile(ridge_abs, 0.50))
    ridge_q95 = float(np.nanquantile(ridge_abs, 0.95))

    # -----------------
    # (d) error vs identifiability metric
    # -----------------
    err_tau = np.abs(y_tau - x_tau)

    if metric == "ridge_resid":
        y_m = np.log10(np.clip(ridge_abs, eps, None))
        ylab_m = (
            r"$\log_{10}\,"
            r"|\delta_K-(\delta_{S_s}+2\delta_{H_d})|$"
        )
    else:
        m_raw = np.clip(df[metric].to_numpy(float), eps, None)
        y_m = np.log10(m_raw)
        ylab_m = rf"$\log_{{10}}\,{metric}$"

    rho_id = _spearman(err_tau, y_m)

    # lithology styling
    lith = df["lith_idx"].to_numpy(int)
    lith_names = {0: "Fine", 1: "Mixed", 2: "Coarse", 3: "Rock"}
    markers = {0: "o", 1: "s", 2: "^", 3: "D"}

    set_figure_style(fontsize=8, dpi=600)

    def _beautify(ax) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.6)

    def _panel(ax, lab: str) -> None:
        ax.text(
            -0.14,
            1.05,
            lab,
            transform=ax.transAxes,
            fontweight="bold",
            va="bottom",
        )

    fig = plt.figure(figsize=(7.2, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # (a)
    axA = fig.add_subplot(gs[0, 0])
    _beautify(axA)
    _panel(axA, "a")

    for i in sorted(np.unique(lith)):
        m = lith == i
        axA.scatter(
            x_tau[m],
            y_tau[m],
            s=22,
            marker=markers.get(i, "o"),
            alpha=0.9,
            label=lith_names.get(i, f"L{i}"),
            rasterized=True,
        )

    axA.scatter(
        x_tau,
        y_cl,
        s=22,
        facecolors="none",
        edgecolors="k",
        linewidths=0.7,
        label=r"$\tau_{cl}$",
        rasterized=True,
    )

    if show_prior:
        axA.scatter(
            x_tau,
            y_pr,
            s=24,
            marker="^",
            facecolors="none",
            edgecolors="0.4",
            linewidths=0.7,
            label="prior",
            rasterized=True,
        )

    lo = float(np.nanmin(np.r_[x_tau, y_tau, y_cl]))
    hi = float(np.nanmax(np.r_[x_tau, y_tau, y_cl]))
    pad = 0.06 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad
    axA.plot([lo, hi], [lo, hi], "--", linewidth=0.9)
    axA.set_xlim(lo, hi)
    axA.set_ylim(lo, hi)
    axA.set_title("Timescale recovery", pad=2)
    axA.set_xlabel(xlab)
    axA.set_ylabel(ylab)

    axA.text(
        0.03,
        0.97,
        (
            f"MAE = {mae_tau:.2f} (log10)\n"
            f"$R^2$ = {r2_tau:.3f}\n"
            f"slope = {slope_tau:.2f}"
        ),
        transform=axA.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.85,
            linewidth=0.0,
        ),
    )

    # (b)
    axB = fig.add_subplot(gs[0, 1])
    _beautify(axB)
    _panel(axB, "b")

    for i in sorted(np.unique(lith)):
        m = lith == i
        axB.scatter(
            x_K[m],
            y_K[m],
            s=22,
            marker=markers.get(i, "o"),
            alpha=0.9,
            rasterized=True,
        )

    lo = float(np.nanmin(np.r_[x_K, y_K]))
    hi = float(np.nanmax(np.r_[x_K, y_K]))
    pad = 0.06 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad
    axB.plot([lo, hi], [lo, hi], "--", linewidth=0.9)
    axB.set_xlim(lo, hi)
    axB.set_ylim(lo, hi)

    axB.set_title("Permeability recovery", pad=2)
    axB.set_xlabel(r"$\log_{10}\,K_{true}$ (m/yr)")
    axB.set_ylabel(r"$\log_{10}\,\hat{K}$ (m/yr)")

    axB.text(
        0.03,
        0.97,
        (
            f"MAE = {mae_K:.2f} (log10)\n"
            f"$R^2$ = {r2_K:.3f}\n"
            f"slope = {slope_K:.2f}"
        ),
        transform=axB.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.85,
            linewidth=0.0,
        ),
    )

    # (c) ridge degeneracy plot
    axC = fig.add_subplot(gs[1, 0])
    _beautify(axC)
    _panel(axC, "c")

    for i in sorted(np.unique(lith)):
        m = lith == i
        axC.scatter(
            ridge_x[m],
            ridge_y[m],
            s=22,
            marker=markers.get(i, "o"),
            alpha=0.9,
            rasterized=True,
        )

    lo = float(np.nanmin(np.r_[ridge_x, ridge_y]))
    hi = float(np.nanmax(np.r_[ridge_x, ridge_y]))
    pad = 0.10 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad
    axC.plot([lo, hi], [lo, hi], "--", linewidth=0.9)
    axC.axvline(0.0, linewidth=0.8)
    axC.axhline(0.0, linewidth=0.8)

    axC.set_xlim(lo, hi)
    axC.set_ylim(lo, hi)

    axC.set_title("Degeneracy ridge check", pad=2)
    axC.set_xlabel(r"$\delta_{S_s}+2\,\delta_{H_d}$ (log$_{10}$)")
    axC.set_ylabel(r"$\delta_K$ (log$_{10}$)")

    axC.text(
        0.03,
        0.97,
        (
            f"Spearman r = {rho_ridge:.2f}\n"
            f"|ridge| q50 = {ridge_q50:.2f}\n"
            f"|ridge| q95 = {ridge_q95:.2f}"
        ),
        transform=axC.transAxes,
        va="top",
        ha="left",
    )

    # (d)
    axD = fig.add_subplot(gs[1, 1])
    _beautify(axD)
    _panel(axD, "d")

    for i in sorted(np.unique(lith)):
        m = lith == i
        axD.scatter(
            err_tau[m],
            y_m[m],
            s=22,
            marker=markers.get(i, "o"),
            alpha=0.9,
            rasterized=True,
        )

    axD.set_title("Error vs identifiability metric", pad=2)
    axD.set_xlabel(r"$|\Delta\log_{10}\tau|$")
    axD.set_ylabel(ylab_m)
    axD.text(
        0.03,
        0.97,
        f"Spearman r = {rho_id:.2f}",
        transform=axD.transAxes,
        va="top",
        ha="left",
    )

    # Legend (top, shared)
    handles: List = []
    for i in sorted(np.unique(lith)):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=markers.get(i, "o"),
                linestyle="none",
                markersize=5,
                label=lith_names.get(i, f"L{i}"),
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="k",
            label=r"$\tau_{cl}$",
        )
    )
    if show_prior:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="none",
                markersize=5,
                markerfacecolor="none",
                markeredgecolor="0.4",
                label="prior",
            )
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 6),
        frameon=False,
    )

    outdir = os.path.dirname(outpath) or "."
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    base = r"results/sm3_synth"
    csv = os.path.join(base, "sm3_synth_runs.csv")

    plot_sm3_identifiability_v32(
        csv,
        "figs/sm3_ident_v32_year.png",
        tau_units="year",
        metric="ridge_resid",
    )
