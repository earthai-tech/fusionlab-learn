# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = ", ".join(missing)
        raise KeyError(
            f"Missing columns ({ctx}): {msg}"
        )


def _save_legend_only(
    handles: List,
    labels: List[str],
    legend_path: str,
    *,
    ncol: int = 2,
) -> None:
    figL = plt.figure(
        figsize=(3.2, 0.8),
        constrained_layout=True,
    )
    figL.legend(
        handles,
        labels,
        loc="center",
        ncol=int(ncol),
        frameon=False,
    )
    outdir = os.path.dirname(legend_path) or "."
    os.makedirs(outdir, exist_ok=True)
    figL.savefig(legend_path, bbox_inches="tight")
    plt.close(figL)


def plot_sm3_suppfig_v32(
    csv_path: str,
    outpath: str,
    *,
    legend_mode: str = "inside",
    legend_outpath: Optional[str] = None,
    tau_units: str = "year",
    kappa_b: float = 1.0,
    show_prior: bool = False,
) -> None:
    """
    v3.2 plotter for SM3 synthetic identifiability.

    legend_mode:
      - "inside"   : legends inside axes
      - "none"     : no legends drawn
      - "separate" : save legend-only fig

    tau_units:
      - "year": use tau_est_med_year, K_est_med_m_per_year
      - "sec" : use tau_est_med_sec,  K_est_med_mps
    """
    df = pd.read_csv(csv_path)

    tau_units = (tau_units or "year").strip().lower()
    if tau_units not in ("year", "sec"):
        raise ValueError(
            "tau_units must be one of {'year','sec'}"
        )

    if legend_mode not in ("inside", "none", "separate"):
        raise ValueError(
            "legend_mode must be inside/none/separate"
        )

    base_cols = [
        "tau_true_year",
        "tau_prior_year",
        "tau_true_sec",
        "tau_prior_sec",
        "Ss_est_med",
        "Hd_est_med",
        "kappa_b",
        "vs_true_delta_K_q50",
        "vs_true_delta_Ss_q50",
        "vs_true_delta_Hd_q50",
        "vs_prior_delta_K_q50",
        "vs_prior_delta_Ss_q50",
        "vs_prior_delta_Hd_q50",
    ]
    _require_cols(df, base_cols, ctx="base")

    if tau_units == "year":
        _require_cols(
            df,
            [
                "tau_est_med_year",
                "K_est_med_m_per_year",
            ],
            ctx="tau_units=year",
        )
    else:
        _require_cols(
            df,
            [
                "tau_est_med_sec",
                "K_est_med_mps",
            ],
            ctx="tau_units=sec",
        )

    LN10 = float(np.log(10.0))
    EPS = 1e-12

    tau_true_year = np.clip(
        df["tau_true_year"].to_numpy(float),
        EPS,
        None,
    )
    tau_prior_year = np.clip(
        df["tau_prior_year"].to_numpy(float),
        EPS,
        None,
    )
    tau_true_sec = np.clip(
        df["tau_true_sec"].to_numpy(float),
        EPS,
        None,
    )
    tau_prior_sec = np.clip(
        df["tau_prior_sec"].to_numpy(float),
        EPS,
        None,
    )

    if tau_units == "year":
        tau_true = tau_true_year
        tau_prior = tau_prior_year

        tau_est = np.clip(
            df["tau_est_med_year"].to_numpy(float),
            EPS,
            None,
        )
        K_est = np.clip(
            df["K_est_med_m_per_year"].to_numpy(float),
            EPS,
            None,
        )

        xlab = (
            r"$\log_{10}\,\tau_{\mathrm{true}}"
            r"\ (\mathrm{yr})$"
        )
        ylab = (
            r"$\log_{10}\,\hat{\tau}"
            r"\ (\mathrm{yr})$"
        )

        cc_fallback = "eps_cons_raw_rms_m_per_year"
        cc_lab_fb = (
            r"$\log_{10}\,\varepsilon_{\mathrm{cons}}"
            r"\ (\mathrm{m/yr})$"
        )
    else:
        tau_true = tau_true_sec
        tau_prior = tau_prior_sec

        tau_est = np.clip(
            df["tau_est_med_sec"].to_numpy(float),
            EPS,
            None,
        )
        K_est = np.clip(
            df["K_est_med_mps"].to_numpy(float),
            EPS,
            None,
        )

        xlab = (
            r"$\log_{10}\,\tau_{\mathrm{true}}"
            r"\ (\mathrm{s})$"
        )
        ylab = (
            r"$\log_{10}\,\hat{\tau}"
            r"\ (\mathrm{s})$"
        )

        cc_fallback = "eps_cons_raw_rms_mps"
        cc_lab_fb = (
            r"$\log_{10}\,\varepsilon_{\mathrm{cons}}"
            r"\ (\mathrm{m/s})$"
        )

    Ss_est = np.clip(
        df["Ss_est_med"].to_numpy(float),
        EPS,
        None,
    )
    Hd_est = np.clip(
        df["Hd_est_med"].to_numpy(float),
        EPS,
        None,
    )

    if "kappa_b" in df.columns:
        kappa = df["kappa_b"].to_numpy(float)
    else:
        kappa = float(kappa_b)

    kappa = np.clip(kappa, EPS, None)

    tau_cl = (Hd_est**2) * Ss_est / (
        (np.pi**2) * kappa * K_est
    )
    tau_cl = np.clip(tau_cl, EPS, None)

    x = np.log10(np.clip(tau_true, EPS, None))
    y = np.log10(np.clip(tau_est, EPS, None))
    ycl = np.log10(np.clip(tau_cl, EPS, None))
    ypr = np.log10(np.clip(tau_prior, EPS, None))

    def ln_to_log10(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        return a / LN10

    set_figure_style(fontsize=8, dpi=600)

    def _beautify(ax) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.6)
        ax.grid(False)

    def _panel_label(ax, s: str) -> None:
        ax.text(
            -0.18,
            1.06,
            s,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    def _robust_limits(
        a: np.ndarray,
        b: np.ndarray,
        *,
        q: Tuple[int, int] = (1, 99),
        pad: float = 0.04,
    ) -> Tuple[float, float]:
        lo = np.nanpercentile(np.r_[a, b], q[0])
        hi = np.nanpercentile(np.r_[a, b], q[1])

        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.nanmin(np.r_[a, b]))
            hi = float(np.nanmax(np.r_[a, b]))

        if lo >= hi:
            lo = float(np.nanmin(np.r_[a, b]))
            hi = float(np.nanmax(np.r_[a, b]))

        span = float(hi - lo)
        return (
            float(lo - pad * span),
            float(hi + pad * span),
        )

    fig = plt.figure(
        figsize=(7.1, 2.55),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(1, 3)

    # =========================
    # (a) log10 tau recovery
    # =========================
    axA = fig.add_subplot(gs[0, 0])
    _beautify(axA)
    _panel_label(axA, "a")

    n = x.size
    rng = np.random.default_rng(42)

    if n > 4000:
        keep = rng.choice(
            np.arange(n),
            size=min(1200, n),
            replace=False,
        )
    else:
        keep = np.arange(n)

    if n > 4000:
        hb = axA.hexbin(
            x,
            y,
            gridsize=45,
            mincnt=1,
            bins="log",
        )
        cb = fig.colorbar(
            hb,
            ax=axA,
            fraction=0.05,
            pad=0.02,
        )
        cb.set_label(r"$\log_{10}(\mathrm{count})$")
    else:
        axA.scatter(
            x,
            y,
            s=12,
            alpha=0.85,
            color="#303030",
            rasterized=True,
        )

    axA.scatter(
        x[keep],
        ycl[keep],
        s=10 if n > 4000 else 12,
        alpha=0.7,
        facecolors="none",
        edgecolors="k",
        linewidths=0.6 if n <= 4000 else 0.5,
        rasterized=True,
    )

    if show_prior:
        axA.scatter(
            x[keep],
            ypr[keep],
            s=10 if n > 4000 else 12,
            alpha=0.7,
            marker="^",
            facecolors="none",
            edgecolors="#666666",
            linewidths=0.5,
            rasterized=True,
        )

    if show_prior:
        y_all = np.r_[y, ycl, ypr]
    else:
        y_all = np.r_[y, ycl]

    lo, hi = _robust_limits(
        x,
        y_all,
        q=(1, 99),
        pad=0.05,
    )

    axA.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=0.8,
        color="#444444",
    )
    axA.set_xlim(lo, hi)
    axA.set_ylim(lo, hi)
    axA.set_aspect("equal", adjustable="box")

    mfit = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[mfit], y[mfit], 1)

    yhat = intercept + slope * x[mfit]
    ss_res = np.nansum((y[mfit] - yhat) ** 2)
    ss_tot = np.nansum(
        (y[mfit] - np.nanmean(y[mfit])) ** 2
    )
    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan")

    mae = float(np.nanmean(np.abs(y - x)))
    if np.isfinite(slope) and abs(slope - 1.0) < 0.05:
        cfit = float(10.0 ** intercept)
        fit_line = (
            r"$\hat{\tau} \approx "
            f"{cfit:.1e}"
            r"\,\tau_{true}$"
        )
    else:
        fit_line = f"$y={intercept:.2f}+{slope:.2f}x$"

    axA.set_title("Timescale recovery", pad=2)
    axA.set_xlabel(xlab)
    axA.set_ylabel(ylab)

    axA.text(
        0.03,
        0.97,
        (
            fr"MAE = {mae:.2f} (log$_{{10}}$)"
            "\n"
            fr"$R^2 = {r2:.4f}$"
            "\n"
            f"{fit_line}"
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

    handles_A = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4,
            markerfacecolor="#303030",
            markeredgecolor="#303030",
            label=r"$\hat{\tau}$ (est.)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor="k",
            label=r"$\tau_{\mathrm{cl}}$ (closure)",
        ),
    ]

    if show_prior:
        handles_A.append(
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="none",
                markersize=4,
                markerfacecolor="none",
                markeredgecolor="#666666",
                label=r"$\tau_{\mathrm{prior}}$",
            )
        )

    if legend_mode == "inside":
        axA.legend(
            handles=handles_A,
            frameon=False,
            loc="lower right",
            borderaxespad=0.2,
            handletextpad=0.4,
            labelspacing=0.2,
        )

    axA.plot(
        [lo, hi],
        [intercept + slope * lo, intercept + slope * hi],
        linewidth=0.8,
        color="#888888",
    )

    # ===========================================
    # (b) marginal deviations (paired boxplots)
    # ===========================================
    axB = fig.add_subplot(gs[0, 1])
    _beautify(axB)
    _panel_label(axB, "b")

    true_cols = [
        "vs_true_delta_K_q50",
        "vs_true_delta_Ss_q50",
        "vs_true_delta_Hd_q50",
    ]
    prior_cols = [
        "vs_prior_delta_K_q50",
        "vs_prior_delta_Ss_q50",
        "vs_prior_delta_Hd_q50",
    ]

    data_true = [
        ln_to_log10(df[c].to_numpy()) for c in true_cols
    ]
    data_prior = [
        ln_to_log10(df[c].to_numpy()) for c in prior_cols
    ]

    base = np.array([1.0, 2.0, 3.0])
    pos_true = base - 0.18
    pos_prior = base + 0.18

    bp1 = axB.boxplot(
        data_true,
        positions=pos_true,
        widths=0.28,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(linewidth=0.9),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        boxprops=dict(linewidth=0.8),
    )
    bp2 = axB.boxplot(
        data_prior,
        positions=pos_prior,
        widths=0.28,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(linewidth=0.9),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        boxprops=dict(linewidth=0.8),
    )

    for b0 in bp1["boxes"]:
        b0.set_facecolor("#BDBDBD")
        b0.set_alpha(0.55)

    for b0 in bp2["boxes"]:
        b0.set_facecolor("none")
        b0.set_edgecolor("#303030")

    axB.axhline(0.0, linewidth=0.8, color="#444444")
    axB.set_title("Log-offsets (median)", pad=2)
    axB.set_ylabel(r"$\delta$ (log$_{10}$ offset)")
    axB.set_xticks(base)
    axB.set_xticklabels([r"$K$", r"$S_s$", r"$H_d$"])

    handles_B = [
        Patch(
            facecolor="#BDBDBD",
            edgecolor="#303030",
            alpha=0.55,
            label="vs true",
        ),
        Patch(
            facecolor="none",
            edgecolor="#303030",
            label="vs prior",
        ),
    ]

    if legend_mode == "inside":
        axB.legend(
            handles=handles_B,
            frameon=False,
            loc="lower right",
            borderaxespad=0.2,
            handlelength=1.2,
            handletextpad=0.5,
            labelspacing=0.2,
        )

    legend_handles = handles_A + handles_B
    legend_labels = [h.get_label() for h in legend_handles]

    if legend_mode == "separate":
        if not legend_outpath:
            raise ValueError(
                "legend_outpath required for separate mode"
            )
        _save_legend_only(
            legend_handles,
            legend_labels,
            legend_outpath,
            ncol=2,
        )

    # =========================
    # (c) trade-off map
    # =========================
    axC = fig.add_subplot(gs[0, 2])
    _beautify(axC)
    _panel_label(axC, "c")

    dx = ln_to_log10(df["vs_true_delta_K_q50"].to_numpy())
    dy = ln_to_log10(df["vs_true_delta_Ss_q50"].to_numpy())

    # Color choice priority (v3.2):
    # 1) closure_log_resid_mean (from diagnostics; ln)
    # 2) closure_consistency_rms
    # 3) eps_prior_rms
    # 4) eps_cons_raw_rms_* fallback (physical)
    if "closure_log_resid_mean" in df.columns:
        cc = ln_to_log10(
            df["closure_log_resid_mean"].to_numpy()
        )
        cbar_lab = (
            r"$\varepsilon_{\mathrm{cl}}$"
            r" (mean, log$_{10}$)"
        )
    elif "closure_consistency_rms" in df.columns:
        raw = df["closure_consistency_rms"].to_numpy(float)
        raw = np.clip(raw, EPS, None)
        cc = np.log10(raw)
        cbar_lab = (
            r"$\log_{10}\,"
            r"\varepsilon_{\mathrm{cl,rms}}$"
        )
    elif "eps_prior_rms" in df.columns:
        raw = df["eps_prior_rms"].to_numpy(float)
        raw = np.clip(raw, EPS, None)
        cc = np.log10(raw)
        cbar_lab = (
            r"$\log_{10}\,"
            r"\varepsilon_{\mathrm{prior}}$"
        )
    else:
        _require_cols(df, [cc_fallback], ctx="panel(c)")
        raw = df[cc_fallback].to_numpy(float)
        raw = np.clip(raw, EPS, None)
        cc = np.log10(raw)
        cbar_lab = cc_lab_fb

    m = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(cc)
    dx, dy, cc = dx[m], dy[m], cc[m]

    vlim = np.nanpercentile(np.abs(cc), 99)
    if not np.isfinite(vlim) or vlim <= 0:
        vlim = np.nanmax(np.abs(cc))

    if dx.size > 6000:
        hb = axC.hexbin(
            dx,
            dy,
            C=cc,
            reduce_C_function=np.nanmean,
            gridsize=45,
            mincnt=1,
            vmin=-vlim,
            vmax=vlim,
        )
        cb = fig.colorbar(
            hb,
            ax=axC,
            fraction=0.05,
            pad=0.02,
        )
    else:
        sc = axC.scatter(
            dx,
            dy,
            c=cc,
            s=12,
            alpha=0.85,
            rasterized=True,
            vmin=-vlim,
            vmax=vlim,
        )
        cb = fig.colorbar(
            sc,
            ax=axC,
            fraction=0.05,
            pad=0.02,
        )

    cb.set_label(cbar_lab)

    axC.set_aspect("equal", adjustable="box")
    axC.axvline(0.0, linewidth=0.8, color="#444444")
    axC.axhline(0.0, linewidth=0.8, color="#444444")

    limx = np.nanpercentile(np.abs(dx), 99)
    limy = np.nanpercentile(np.abs(dy), 99)
    lim = max(limx, limy)

    if np.isfinite(lim) and lim > 0:
        axC.set_xlim(-lim, lim)
        axC.set_ylim(-lim, lim)

    axC.set_title(r"$K$–$S_s$ trade-off", pad=2)
    axC.set_xlabel(r"$\delta_K^{true}$ (log$_{10}$)")
    axC.set_ylabel(r"$\delta_{S_s}^{true}$ (log$_{10}$)")

    outdir = os.path.dirname(outpath) or "."
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    path = r"F:\repositories\fusionlab-learn\results\sm3_synth"
    csv = os.path.join(path, "sm3_synth_runs.csv")

    plot_sm3_suppfig_v32(
        csv,
        "figs/fig_year_v32.png",
        tau_units="year",
    )
    plot_sm3_suppfig_v32(
        csv,
        "figs/fig_sec_v32.png",
        tau_units="sec",
    )
    plot_sm3_suppfig_v32(
        csv,
        "figs/fig_year_show_prior_v32.png",
        tau_units="year",
        show_prior=True,
    )
