# geoprior/ui/tools/identifiability_viz.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def make_payload_timescale_figure(
    payload: Dict[str, np.ndarray],
    *,
    report_units: str,
    kappa_b: float,
    tau_true: Optional[float] = None,
) -> Figure:
    """
    Unit-safe preview:
      - show tau(est) and tau_closure
      - if tau_true is provided, plot vs tau_true
    """
    eps = 1e-12
    kappa = max(float(kappa_b), eps)

    tau = np.asarray(payload.get("tau", []), float)
    K = np.asarray(payload.get("K", []), float)
    Ss = np.asarray(payload.get("Ss", []), float)
    Hd = np.asarray(payload.get("Hd", []), float)

    tau = np.clip(tau, eps, None)
    K = np.clip(K, eps, None)
    Ss = np.clip(Ss, eps, None)
    Hd = np.clip(Hd, eps, None)

    tau_cl = (Hd**2) * Ss / (np.pi**2 * kappa * K)
    tau_cl = np.clip(tau_cl, eps, None)

    fig = Figure(figsize=(5.2, 3.4))
    ax = fig.add_subplot(111)

    y = np.log10(tau)
    ycl = np.log10(tau_cl)

    if tau_true is not None:
        x0 = np.log10(max(float(tau_true), eps))
        ax.axhline(
            x0,
            lw=0.9,
            ls="--",
            alpha=0.5,
            label="log10(tau_true)",
        )
        ax.scatter(
            np.full_like(y, x0),
            y,
            s=12,
            alpha=0.85,
            label="log10(tau_est)",
        )
        ax.scatter(
            np.full_like(ycl, x0),
            ycl,
            s=12,
            alpha=0.85,
            facecolors="none",
            edgecolors="k",
            label="log10(tau_cl)",
        )
        ax.set_xlabel("log10(tau_true)")
        ax.set_ylabel("log10(tau)")
    else:
        ax.scatter(
            np.log10(tau_cl),
            np.log10(tau),
            s=12,
            alpha=0.85,
        )
        ax.set_xlabel("log10(tau_cl)")
        ax.set_ylabel("log10(tau_est)")
        ax.set_title("Self-consistency (no SM3 truth)")

    u = "yr" if str(report_units).startswith("y") else "s"
    ax.set_title(f"Timescale preview ({u})")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def make_sm3_suppfig_figure(
    df,
    *,
    tau_units: str = "year",
    kappa_b: float = 1.0,
    show_prior: bool = False,
    legend_mode: str = "inside",
    dpi: int = 600,
    fontsize: int = 8,
) -> Tuple[Figure, Optional[Figure]]:
    """
    Port of the 3-panel SM3 suppfig (script version).

    Returns
    -------
    fig : Figure
        Main 3-panel figure.
    legend_fig : Figure | None
        Legend-only figure when legend_mode="separate".
    """
    tau_units = (tau_units or "year").strip().lower()
    if tau_units not in ("year", "sec"):
        raise ValueError("tau_units must be 'year' or 'sec'")

    legend_mode = (legend_mode or "inside").strip().lower()
    if legend_mode not in ("inside", "none", "separate"):
        raise ValueError(
            "legend_mode must be inside/none/separate"
        )

    LN10 = np.log(10.0)
    EPS = 1e-12

    def ln_to_log10(a):
        return np.asarray(a, float) / LN10

    def beautify(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.6)
        ax.grid(False)

    def panel_label(ax, s):
        ax.text(
            -0.18,
            1.06,
            s,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=fontsize + 1,
            fontweight="bold",
        )

    def robust_limits(a, b, q=(1, 99), pad=0.04):
        v = np.r_[np.asarray(a, float), np.asarray(b, float)]
        lo = np.nanpercentile(v, q[0])
        hi = np.nanpercentile(v, q[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = np.nanmin(v)
            hi = np.nanmax(v)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = -1.0, 1.0
        span = hi - lo
        return lo - pad * span, hi + pad * span

    # ---- true/prior tau available in CSV
    t_ty = np.clip(df["tau_true_year"].to_numpy(float), EPS, None)
    t_py = np.clip(df["tau_prior_year"].to_numpy(float), EPS, None)
    t_ts = np.clip(df["tau_true_sec"].to_numpy(float), EPS, None)
    t_ps = np.clip(df["tau_prior_sec"].to_numpy(float), EPS, None)

    if tau_units == "year":
        tau_true = t_ty
        tau_prior = t_py
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
        xlab = r"$\log_{10}\,\tau_{\mathrm{true}}\ (\mathrm{yr})$"
        ylab = r"$\log_{10}\,\hat{\tau}\ (\mathrm{yr})$"
    else:
        tau_true = t_ts
        tau_prior = t_ps
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
        xlab = r"$\log_{10}\,\tau_{\mathrm{true}}\ (\mathrm{s})$"
        ylab = r"$\log_{10}\,\hat{\tau}\ (\mathrm{s})$"

    Ss_est = np.clip(df["Ss_est_med"].to_numpy(float), EPS, None)
    Hd_est = np.clip(df["Hd_est_med"].to_numpy(float), EPS, None)

    if "kappa_b" in df.columns:
        kappa = np.clip(df["kappa_b"].to_numpy(float), EPS, None)
    else:
        kappa = max(float(kappa_b), EPS)

    tau_cl = (Hd_est**2) * Ss_est / (np.pi**2 * kappa * K_est)
    tau_cl = np.clip(tau_cl, EPS, None)

    x = np.log10(np.clip(tau_true, EPS, None))
    y = np.log10(np.clip(tau_est, EPS, None))
    ycl = np.log10(np.clip(tau_cl, EPS, None))
    ypr = np.log10(np.clip(tau_prior, EPS, None))

    # ---- local style (avoid global rcParams)
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["axes.titlesize"] = fontsize
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["legend.fontsize"] = fontsize
    mpl.rcParams["axes.linewidth"] = 0.6
    mpl.rcParams["xtick.major.width"] = 0.6
    mpl.rcParams["ytick.major.width"] = 0.6
    mpl.rcParams["lines.linewidth"] = 1.0

    fig = Figure(figsize=(7.1, 2.55), constrained_layout=True)
    fig.set_dpi(dpi)
    gs = fig.add_gridspec(1, 3)

    # =========================
    # (a) log10 tau recovery
    # =========================
    axA = fig.add_subplot(gs[0, 0])
    beautify(axA)
    panel_label(axA, "a")

    n = int(x.size)
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
        cb = fig.colorbar(hb, ax=axA, fraction=0.05, pad=0.02)
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

    y_all = np.r_[y, ycl, ypr] if show_prior else np.r_[y, ycl]
    lo, hi = robust_limits(x, y_all, q=(1, 99), pad=0.05)

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
    if int(mfit.sum()) >= 2:
        b, a = np.polyfit(x[mfit], y[mfit], 1)
        yhat = a + b * x[mfit]
        ss_res = np.nansum((y[mfit] - yhat) ** 2)
        ss_tot = np.nansum(
            (y[mfit] - np.nanmean(y[mfit])) ** 2
        )
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        a, b, r2 = np.nan, np.nan, np.nan

    mae = np.nanmean(np.abs(y - x))

    if np.isfinite(b) and abs(b - 1.0) < 0.05:
        cfit = 10.0 ** a
        fit_line = (
            r"$\hat{\tau} \approx "
            f"{cfit:.1e}"
            r"\,\tau_{true}$"
        )
    else:
        fit_line = f"$y={a:.2f}+{b:.2f}x$"

    axA.set_title("Timescale recovery", pad=2)
    axA.set_xlabel(xlab)
    axA.set_ylabel(ylab)

    axA.text(
        0.03,
        0.97,
        fr"MAE = {mae:.2f} (log$_{{10}}$)"
        "\n"
        fr"$R^2 = {r2:.4f}$"
        "\n"
        f"{fit_line}",
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

    if np.isfinite(a) and np.isfinite(b):
        axA.plot(
            [lo, hi],
            [a + b * lo, a + b * hi],
            linewidth=0.8,
            color="#888888",
        )

    # ===========================================
    # (b) marginal deviations (paired boxplots)
    # ===========================================
    axB = fig.add_subplot(gs[0, 1])
    beautify(axB)
    panel_label(axB, "b")

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

    data_true = [ln_to_log10(df[c].to_numpy()) for c in true_cols]
    data_prior = [ln_to_log10(df[c].to_numpy()) for c in prior_cols]

    base = np.array([1, 2, 3], dtype=float)
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

    # =========================
    # (c) trade-off map
    # =========================
    axC = fig.add_subplot(gs[0, 2])
    beautify(axC)
    panel_label(axC, "c")

    dx = ln_to_log10(df["vs_true_delta_K_q50"].to_numpy())
    dy = ln_to_log10(df["vs_true_delta_Ss_q50"].to_numpy())
    cc = ln_to_log10(df["closure_log_resid_mean"].to_numpy())

    m = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(cc)
    dx, dy, cc = dx[m], dy[m], cc[m]

    vlim = np.nanpercentile(np.abs(cc), 99)
    if not np.isfinite(vlim) or vlim <= 0:
        vlim = np.nanmax(np.abs(cc)) if cc.size else 1.0

    if int(dx.size) > 6000:
        hb2 = axC.hexbin(
            dx,
            dy,
            C=cc,
            reduce_C_function=np.nanmean,
            gridsize=45,
            mincnt=1,
            vmin=-vlim,
            vmax=vlim,
        )
        cb2 = fig.colorbar(hb2, ax=axC, fraction=0.05, pad=0.02)
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
        cb2 = fig.colorbar(sc, ax=axC, fraction=0.05, pad=0.02)

    axC.set_aspect("equal", adjustable="box")
    cb2.set_label(r"$\varepsilon_{\mathrm{cl}}$ (mean, log$_{10}$)")
    axC.axvline(0.0, linewidth=0.8, color="#444444")
    axC.axhline(0.0, linewidth=0.8, color="#444444")

    limx = np.nanpercentile(np.abs(dx), 99) if dx.size else 1.0
    limy = np.nanpercentile(np.abs(dy), 99) if dy.size else 1.0
    lim = max(limx, limy)
    if np.isfinite(lim) and lim > 0:
        axC.set_xlim(-lim, lim)
        axC.set_ylim(-lim, lim)

    axC.set_title(r"$K$–$S_s$ trade-off", pad=2)
    axC.set_xlabel(r"$\delta_K^{true}$ (log$_{10}$)")
    axC.set_ylabel(r"$\delta_{S_s}^{true}$ (log$_{10}$)")

    # ---- legend-only optional
    legend_fig = None
    if legend_mode == "separate":
        legend_fig = make_sm3_legend_figure(
            handles_A + handles_B
        )

    return fig, legend_fig


def make_sm3_legend_figure(
    handles,
    *,
    ncol: int = 2,
) -> Figure:
    """
    Legend-only figure for separate export.
    """
    figL = Figure(figsize=(3.2, 0.8), constrained_layout=True)
    labels = [h.get_label() for h in handles]
    figL.legend(handles, labels, loc="center", ncol=ncol, frameon=False)
    return figL
