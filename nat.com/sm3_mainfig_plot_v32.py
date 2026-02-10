# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os
from typing import Iterable, List, Tuple, Callable

import warnings

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
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan"), float("nan"), float("nan")

    xx = x[m]
    yy = y[m]

    # Degenerate x => slope undefined
    if float(np.nanstd(xx)) == 0.0:
        return float("nan"), float("nan"), float("nan")

    # NumPy compatibility:
    # - some builds have np.RankWarning
    # - others expose it via numpy.polynomial.polyutils
    try:
        rank_warn = np.RankWarning
    except Exception:
        try:
            from numpy.polynomial.polyutils import (
                RankWarning as rank_warn,
            )
        except Exception:
            rank_warn = Warning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rank_warn)
        try:
            slope, inter = np.polyfit(xx, yy, 1)
        except Exception:
            return float("nan"), float("nan"), float("nan")

    yhat = inter + slope * xx
    ss_res = float(np.nansum((yy - yhat) ** 2))
    ss_tot = float(np.nansum((yy - np.nanmean(yy)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    return float(slope), float(inter), float(r2)


def _mask_pair(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _boot_ci(
    a: np.ndarray,
    *,
    ci: float = 0.95,
) -> tuple[float, float]:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")

    alpha = (1.0 - float(ci)) / 2.0
    lo = float(np.quantile(a, alpha))
    hi = float(np.quantile(a, 1.0 - alpha))
    return lo, hi


def _bootstrap_pair(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    x, y = _mask_pair(x, y)
    n = int(x.size)
    if n < 2:
        return np.full((int(n_boot),), np.nan)

    rng = np.random.default_rng(int(seed))
    out = np.empty((int(n_boot),), float)

    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        out[b] = float(fn(x[idx], y[idx]))

    return out
def _fmt_ci(
    est: float,
    lo: float,
    hi: float,
    *,
    nd: int = 2,
) -> str:
    """
    Format an estimate with an optional CI.

    If CI bounds are not finite, returns only the estimate.
    """
    ok = (
        np.isfinite(est)
        and np.isfinite(lo)
        and np.isfinite(hi)
    )
    if not ok:
        return f"{est:.{nd}f}"
    return f"{est:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]"
def plot_sm3_identifiability_v32(
    csv_path: str,
    outpath: str,
    *,
    show_prior: bool = True,
    show_legend: bool = True,
    show_stats_text: bool = True,
    tau_units: str = "year",
    metric: str = "ridge_resid",
    k_from_tau: bool | None = None,
    k_cl_source: str = "prior",  # prior|true|est
    only_identify: str | None = None,
    nx_min: int | None = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    boot_seed: int = 0,
    show_ci: bool = True,
) -> None:
    """
    Plot identifiability diagnostics (v3.2).

    Notes
    -----
    - If k_from_tau is None:
        * default False when identify includes "both"
        * else default True
    - Bootstrap CIs are computed over rows. At small n,
      intervals will be wide / unstable.
    - show_stats_text controls the metric annotation
      blocks inside panels (a-d).
    - show_legend controls the shared legend at the top.
    """
    df = pd.read_csv(csv_path)

    # -------------------------
    # Filters (strict)
    # -------------------------
    if only_identify is not None:
        if "identify" not in df.columns:
            raise KeyError(
                "only_identify set but missing "
                "'identify' column."
            )
        want = str(only_identify).strip().lower()
        got = (
            df["identify"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df = df.loc[got == want].copy()

    if nx_min is not None:
        if "nx" not in df.columns:
            raise KeyError(
                "nx_min set but missing 'nx' column."
            )
        nxv = df["nx"].astype(int)
        df = df.loc[nxv >= int(nx_min)].copy()

    if df.empty:
        raise ValueError("No rows left after filtering.")

    # -------------------------
    # Decide k_from_tau default
    # -------------------------
    if k_from_tau is None:
        if "identify" in df.columns:
            modes = (
                df["identify"]
                .astype(str)
                .str.strip()
                .str.lower()
                .unique()
                .tolist()
            )
            k_from_tau = "both" not in set(modes)
        else:
            k_from_tau = True

    src = str(k_cl_source).strip().lower()
    if src not in ("prior", "true", "est"):
        raise ValueError(
            "k_cl_source must be prior|true|est."
        )

    tau_units = str(tau_units).strip().lower()
    if tau_units not in ("year", "sec"):
        raise ValueError("tau_units must be year/sec.")

    if metric not in (
        "ridge_resid",
        "eps_prior_rms",
        "closure_consistency_rms",
    ):
        raise ValueError("metric not supported.")

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

    ln10 = float(np.log(10.0))
    eps = 1e-12

    # -------------------------
    # Metric helpers (for CI)
    # -------------------------
    def _mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.nanmean(np.abs(a - b)))

    def _r2(a: np.ndarray, b: np.ndarray) -> float:
        # Avoid corrcoef warnings when std == 0
        a, b = _mask_pair(a, b)
        if a.size < 2:
            return float("nan")
        if float(np.nanstd(a)) == 0.0:
            return float("nan")
        if float(np.nanstd(b)) == 0.0:
            return float("nan")

        with np.errstate(invalid="ignore", divide="ignore"):
            r = float(np.corrcoef(a, b)[0, 1])

        if not np.isfinite(r):
            return float("nan")
        return r * r

    def _rho(a: np.ndarray, b: np.ndarray) -> float:
        return float(_spearman(a, b))

    def _slope(a: np.ndarray, b: np.ndarray) -> float:
        s, _, _ = _linfit_stats(a, b)
        return float(s)

    def _stat_ci_text(
        x: np.ndarray,
        y: np.ndarray,
        fn: Callable[[np.ndarray, np.ndarray], float],
        *,
        nd: int,
        seed_off: int,
        est: float | None = None,
    ) -> tuple[float, tuple[float, float], str]:
        if est is None:
            est_v = float(fn(x, y))
        else:
            est_v = float(est)

        lo = float("nan")
        hi = float("nan")

        if show_ci:
            bt = _bootstrap_pair(
                x,
                y,
                fn,
                n_boot=int(n_boot),
                seed=int(boot_seed + seed_off),
            )
            lo, hi = _boot_ci(bt, ci=float(ci))

        txt = _fmt_ci(est_v, lo, hi, nd=int(nd))
        return est_v, (lo, hi), txt

    # -----------------
    # (a) tau recovery
    # -----------------
    if tau_units == "year":
        t_true = np.clip(df["tau_true_year"], eps, None)
        t_est = np.clip(df["tau_est_med_year"], eps, None)
        t_pri = np.clip(df["tau_prior_year"], eps, None)
        xlab = (
            r"$\log_{10}\,\tau_{\mathrm{true}}$ "
            r"(yr)"
        )
        ylab = (
            r"$\log_{10}\,\hat{\tau}$ "
            r"(yr)"
        )
    else:
        t_true = np.clip(df["tau_true_sec"], eps, None)
        t_est = np.clip(df["tau_est_med_sec"], eps, None)
        t_pri = np.clip(df["tau_prior_sec"], eps, None)
        xlab = (
            r"$\log_{10}\,\tau_{\mathrm{true}}$ "
            r"(s)"
        )
        ylab = (
            r"$\log_{10}\,\hat{\tau}$ "
            r"(s)"
        )

    x_tau = np.log10(t_true.to_numpy(float))
    y_tau = np.log10(t_est.to_numpy(float))

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

    _, _, mae_txt = _stat_ci_text(
        x_tau,
        y_tau,
        _mae,
        nd=2,
        seed_off=11,
        est=mae_tau,
    )
    _, _, r2_txt = _stat_ci_text(
        x_tau,
        y_tau,
        _r2,
        nd=3,
        seed_off=12,
        est=r2_tau,
    )
    _, _, sl_txt = _stat_ci_text(
        x_tau,
        y_tau,
        _slope,
        nd=2,
        seed_off=13,
        est=slope_tau,
    )

    # -----------------
    # (b) K recovery (m/yr)
    # -----------------
    K_true = np.clip(df["K_true_mps"].to_numpy(float), eps, None)
    x_K = np.log10(K_true * SEC_PER_YEAR)

    if bool(k_from_tau):
        tau_est = np.clip(
            df["tau_est_med_sec"].to_numpy(float),
            eps,
            None,
        )

        if src == "true":
            Ss_use = np.clip(
                df["Ss_true"].to_numpy(float),
                eps,
                None,
            )
            Hd_use = np.clip(
                df["Hd_true"].to_numpy(float),
                eps,
                None,
            )
        elif src == "est":
            Ss_use = np.clip(
                df["Ss_est_med"].to_numpy(float),
                eps,
                None,
            )
            Hd_use = np.clip(
                df["Hd_est_med"].to_numpy(float),
                eps,
                None,
            )
        else:
            Ss_use = np.clip(
                df["Ss_prior"].to_numpy(float),
                eps,
                None,
            )
            Hd_use = np.clip(
                df["Hd_prior"].to_numpy(float),
                eps,
                None,
            )

        kappa = np.clip(
            df["kappa_b"].to_numpy(float),
            eps,
            None,
        )

        K_cl_mps = (Hd_use**2) * Ss_use / (
            np.pi**2 * kappa * tau_est
        )
        y_K = np.log10(
            np.clip(K_cl_mps * SEC_PER_YEAR, eps, None)
        )
        k_title = "Permeability from closure"
        ylabK = (
            r"$\log_{10}\,K_{cl}(\hat{\tau})$ "
            r"(m/yr)"
        )
    else:
        y_K = np.log10(
            np.clip(
                df["K_est_med_m_per_year"].to_numpy(float),
                eps,
                None,
            )
        )
        k_title = "Permeability recovery"
        ylabK = r"$\log_{10}\,\hat{K}$ (m/yr)"

    mae_K = float(np.nanmean(np.abs(y_K - x_K)))
    slope_K, _, r2_K = _linfit_stats(x_K, y_K)

    _, _, maeK_txt = _stat_ci_text(
        x_K,
        y_K,
        _mae,
        nd=2,
        seed_off=31,
        est=mae_K,
    )
    _, _, r2K_txt = _stat_ci_text(
        x_K,
        y_K,
        _r2,
        nd=3,
        seed_off=32,
        est=r2_K,
    )
    _, _, slK_txt = _stat_ci_text(
        x_K,
        y_K,
        _slope,
        nd=2,
        seed_off=33,
        est=slope_K,
    )

    # -----------------
    # (c) ridge plot
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

    _, _, rhoR_txt = _stat_ci_text(
        ridge_x,
        ridge_y,
        _rho,
        nd=2,
        seed_off=41,
        est=rho_ridge,
    )

    # -----------------
    # (d) err vs metric
    # -----------------
    err_tau = np.abs(y_tau - x_tau)

    if metric == "ridge_resid":
        y_m = np.log10(np.clip(ridge_abs, eps, None))
        ylab_m = (
            r"$\log_{10}\,|\delta_K-"
            r"(\delta_{S_s}+2\delta_{H_d})|$"
        )
    else:
        m_raw = np.clip(df[metric].to_numpy(float), eps, None)
        y_m = np.log10(m_raw)
        ylab_m = rf"$\log_{{10}}\,{metric}$"

    rho_id = _spearman(err_tau, y_m)

    _, _, rhoI_txt = _stat_ci_text(
        err_tau,
        y_m,
        _rho,
        nd=2,
        seed_off=42,
        est=rho_id,
    )

    # -----------------
    # Styling + plotting
    # -----------------
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

    nx_note = ""
    if "nx" in df.columns:
        nxu = sorted(set(df["nx"].astype(int).tolist()))
        nx_note = f" (nx={nxu})"

    fig = plt.figure(
        figsize=(7.2, 4.2),
        constrained_layout=True,
    )
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

    loA = float(np.nanmin(np.r_[x_tau, y_tau, y_cl]))
    hiA = float(np.nanmax(np.r_[x_tau, y_tau, y_cl]))
    padA = 0.06 * (hiA - loA + 1e-9)
    loA -= padA
    hiA += padA
    axA.plot([loA, hiA], [loA, hiA], "--", linewidth=0.9)
    axA.set_xlim(loA, hiA)
    axA.set_ylim(loA, hiA)
    axA.set_title("Timescale recovery" + nx_note, pad=2)
    axA.set_xlabel(xlab)
    axA.set_ylabel(ylab)

    if show_stats_text:
        axA.text(
            0.03,
            0.97,
            (
                f"MAE = {mae_txt} (log10)\n"
                f"$R^2$ = {r2_txt}\n"
                f"slope = {sl_txt}"
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

    loB = float(np.nanmin(np.r_[x_K, y_K]))
    hiB = float(np.nanmax(np.r_[x_K, y_K]))
    padB = 0.06 * (hiB - loB + 1e-9)
    loB -= padB
    hiB += padB
    axB.plot([loB, hiB], [loB, hiB], "--", linewidth=0.9)
    axB.set_xlim(loB, hiB)
    axB.set_ylim(loB, hiB)

    axB.set_title(k_title + nx_note, pad=2)
    axB.set_xlabel(r"$\log_{10}\,K_{true}$ (m/yr)")
    axB.set_ylabel(ylabK)

    if show_stats_text:
        axB.text(
            0.03,
            0.97,
            (
                f"MAE = {maeK_txt} (log10)\n"
                f"$R^2$ = {r2K_txt}\n"
                f"slope = {slK_txt}"
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

    # (c)
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

    loC = float(np.nanmin(np.r_[ridge_x, ridge_y]))
    hiC = float(np.nanmax(np.r_[ridge_x, ridge_y]))
    padC = 0.10 * (hiC - loC + 1e-9)
    loC -= padC
    hiC += padC

    axC.plot([loC, hiC], [loC, hiC], "--", linewidth=0.9)
    axC.axvline(0.0, linewidth=0.8)
    axC.axhline(0.0, linewidth=0.8)
    axC.set_xlim(loC, hiC)
    axC.set_ylim(loC, hiC)

    axC.set_title("Degeneracy ridge check", pad=2)
    axC.set_xlabel(
        r"$\delta_{S_s}+2\,\delta_{H_d}$ (log$_{10}$)"
    )
    axC.set_ylabel(r"$\delta_K$ (log$_{10}$)")

    if show_stats_text:
        axC.text(
            0.03,
            0.97,
            (
                f"Spearman r = {rhoR_txt}\n"
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

    if show_stats_text:
        axD.text(
            0.03,
            0.97,
            f"Spearman r = {rhoI_txt}",
            transform=axD.transAxes,
            va="top",
            ha="left",
        )

    # Legend (shared)
    if show_legend:
        handles: List[Line2D] = []
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
    base = r"results/sm3_synth_1d"
    csv = os.path.join(base, "sm3_synth_runs.csv")
    # plot_sm3_identifiability_v32(
    #     csv,
    #     "figs/sm3_tau.png",
    #     only_identify="tau",
    #     k_from_tau=True,
    # )

    # plot_sm3_identifiability_v32(
    #     csv,
    #     "figs/sm3_ident_v32_year.png",
    #     tau_units="year",
    #     metric="ridge_resid",
    #     k_from_tau=True ,
    #     k_cl_source="est", #"est",   # <<< THIS makes panel (b) use Ss_est_med/Hd_est_med
    # )
    plot_sm3_identifiability_v32(
        csv,
        "figs/sm3_both_clean_from_tau.png",
        only_identify="both",
        nx_min=5,
        k_from_tau=True,
        show_stats_text=True,
        show_legend=True,
    )
