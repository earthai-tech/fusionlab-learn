# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# scripts/plot_sm3_identifiability.py
#
r"""
Supplementary Methods 3 • Synthetic identifiability (SM3)

Panels
------
(a) Timescale recovery:
    log10(tau_true) vs log10(tau_est)
    + optional tau_cl overlay (closure-implied)
    + optional tau_prior marker.

(b) Permeability recovery:
    log10(K_true) vs log10(K_est)  OR
    log10(K_true) vs log10(K_cl(tau_est)).

(c) Degeneracy ridge check:
    delta_K vs (delta_Ss + 2 delta_Hd).

(d) Error vs identifiability metric:
    |Δ log10 tau| vs chosen identifiability metric.

Inputs
------
CSV file produced by SM3 synth runs, typically:
  results/sm3_synth_1d/sm3_synth_runs.csv

Outputs
-------
- Figure: scripts/figs/<out>.png and <out>.svg
- Summary table: scripts/out/<out_csv>
- Summary json: scripts/out/<out_json>

API conventions
---------------
- Style via scripts.utils.set_paper_style()
- Figure output via scripts.utils.resolve_fig_out()
- Tables/JSON via scripts.utils.resolve_out_out()
- Common plot text toggles via scripts.utils.add_plot_text_args()

Notes
-----
- Bootstrap CIs are computed row-wise.
- If --k-from-tau=auto:
    * defaults to false when identify includes "both"
    * else defaults to true.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import config as cfg
from . import utils


# -------------------------------------------------------------------
# Small numeric helpers
# -------------------------------------------------------------------
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


def _mask_pair(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _mask_pair(x, y)
    if x.size < 3:
        return float("nan")

    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()

    if float(np.nanstd(rx)) == 0.0:
        return float("nan")
    if float(np.nanstd(ry)) == 0.0:
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
        return (float("nan"), float("nan"), float("nan"))

    xx = x[m]
    yy = y[m]

    if float(np.nanstd(xx)) == 0.0:
        return (float("nan"), float("nan"), float("nan"))

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
            return (float("nan"), float("nan"), float("nan"))

    yhat = inter + slope * xx
    ss_res = float(np.nansum((yy - yhat) ** 2))
    ss_tot = float(np.nansum((yy - np.nanmean(yy)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    return (float(slope), float(inter), float(r2))


def _bootstrap_pair(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int,
    seed: int,
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


def _boot_ci(
    a: np.ndarray,
    *,
    ci: float,
) -> Tuple[float, float]:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float("nan"), float("nan"))

    alpha = (1.0 - float(ci)) / 2.0
    lo = float(np.quantile(a, alpha))
    hi = float(np.quantile(a, 1.0 - alpha))
    return (lo, hi)


def _fmt_ci(
    est: float,
    lo: float,
    hi: float,
    *,
    nd: int,
) -> str:
    ok = (
        np.isfinite(est)
        and np.isfinite(lo)
        and np.isfinite(hi)
    )
    if not ok:
        return f"{est:.{nd}f}"
    return f"{est:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]"


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot-sm3-identifiability",
        description="SM3: Synthetic identifiability figure.",
    )

    p.add_argument(
        "--csv",
        type=str,
        default="results/sm3_synth_1d/sm3_synth_runs.csv",
        help="Input CSV (sm3_synth_runs.csv).",
    )

    p.add_argument(
        "--tau-units",
        type=str,
        choices=["year", "sec"],
        default="year",
        help="Units for tau panels (year/sec).",
    )

    p.add_argument(
        "--metric",
        type=str,
        choices=[
            "ridge_resid",
            "eps_prior_rms",
            "closure_consistency_rms",
        ],
        default="ridge_resid",
        help="Metric used in panel (d).",
    )

    p.add_argument(
        "--k-from-tau",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Panel (b): derive K from tau (auto/true/false).",
    )

    p.add_argument(
        "--k-cl-source",
        type=str,
        default="prior",
        choices=["prior", "true", "est"],
        help="If K is derived from tau, which Ss/Hd to use.",
    )

    p.add_argument(
        "--only-identify",
        type=str,
        default=None,
        help="Filter identify column (e.g., tau/both).",
    )

    p.add_argument(
        "--nx-min",
        type=int,
        default=None,
        help="Filter rows with nx >= nx-min.",
    )

    p.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap replicates for CI.",
    )
    p.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="CI level (e.g., 0.95).",
    )
    p.add_argument(
        "--boot-seed",
        type=int,
        default=0,
        help="Bootstrap RNG seed base.",
    )

    p.add_argument(
        "--show-ci",
        type=str,
        default="true",
        help="Show CI in stats annotations (true/false).",
    )
    p.add_argument(
        "--show-stats",
        type=str,
        default="true",
        help="Show stats annotation blocks (true/false).",
    )
    p.add_argument(
        "--show-prior",
        type=str,
        default="true",
        help="Overlay prior points in panel (a).",
    )
    p.add_argument(
        "--show-tau-cl",
        type=str,
        default="true",
        help="Overlay closure tau_cl in panel (a).",
    )

    p.add_argument(
        "--font",
        type=int,
        default=cfg.PAPER_FONT,
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=cfg.PAPER_DPI,
    )

    p.add_argument(
        "--out-csv",
        type=str,
        default="sm3_identifiability_summary.csv",
        help="Summary table path (scripts/out if rel).",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default="sm3_identifiability_summary.json",
        help="Summary json path (scripts/out if rel).",
    )

    utils.add_plot_text_args(
        p,
        default_out="sm3-identifiability",
    )

    return p.parse_args(argv)


# -------------------------------------------------------------------
# Core
# -------------------------------------------------------------------
def plot_sm3_identifiability(
    df: pd.DataFrame,
    *,
    tau_units: str,
    metric: str,
    k_from_tau: bool,
    k_cl_source: str,
    show_prior: bool,
    show_tau_cl: bool,
    show_legend: bool,
    show_labels: bool,
    show_ticklabels: bool,
    show_panel_titles: bool,
    show_title: bool,
    title: Optional[str],
    show_stats: bool,
    show_ci: bool,
    n_boot: int,
    ci: float,
    boot_seed: int,
    out: str,
    out_csv: str,
    out_json: str,
    dpi: int,
) -> None:
    utils.ensure_script_dirs()
    utils.set_paper_style()

    base_cols = [
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
    _require_cols(df, base_cols, ctx="SM3")

    eps = 1e-12
    sec_per_year = float(cfg.SECONDS_PER_YEAR)
    ln10 = float(np.log(10.0))

    # -------------------------
    # helpers for stats + CI
    # -------------------------
    def _mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.nanmean(np.abs(a - b)))

    def _r2(a: np.ndarray, b: np.ndarray) -> float:
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

    def _slope(a: np.ndarray, b: np.ndarray) -> float:
        s, _, _ = _linfit_stats(a, b)
        return float(s)

    def _rho(a: np.ndarray, b: np.ndarray) -> float:
        return float(_spearman(a, b))

    def _stat_ci(
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        fn: Callable[[np.ndarray, np.ndarray], float],
        *,
        nd: int,
        seed_off: int,
        est: Optional[float] = None,
    ) -> Tuple[dict, str]:
        est_v = float(fn(x, y)) if est is None else float(est)

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

        rec = {
            "name": name,
            "estimate": est_v,
            "ci_lo": lo,
            "ci_hi": hi,
            "nd": int(nd),
        }
        txt = _fmt_ci(est_v, lo, hi, nd=int(nd))
        return rec, txt

    # -------------------------
    # Panel (a): tau recovery
    # -------------------------
    if tau_units == "year":
        t_true = np.clip(df["tau_true_year"], eps, None)
        t_est = np.clip(df["tau_est_med_year"], eps, None)
        t_pri = np.clip(df["tau_prior_year"], eps, None)

        xlab_a = r"$\log_{10}\,\tau_{\mathrm{true}}$ (yr)"
        ylab_a = r"$\log_{10}\,\hat{\tau}$ (yr)"
        y_pr = np.log10(t_pri.to_numpy(float))
    else:
        t_true = np.clip(df["tau_true_sec"], eps, None)
        t_est = np.clip(df["tau_est_med_sec"], eps, None)
        t_pri = np.clip(df["tau_prior_sec"], eps, None)

        xlab_a = r"$\log_{10}\,\tau_{\mathrm{true}}$ (s)"
        ylab_a = r"$\log_{10}\,\hat{\tau}$ (s)"
        y_pr = np.log10(t_pri.to_numpy(float))

    x_tau = np.log10(t_true.to_numpy(float))
    y_tau = np.log10(t_est.to_numpy(float))

    kappa = np.clip(df["kappa_b"].to_numpy(float), eps, None)
    K_est = np.clip(df["K_est_med_mps"].to_numpy(float), eps, None)
    Ss_est = np.clip(df["Ss_est_med"].to_numpy(float), eps, None)
    Hd_est = np.clip(df["Hd_est_med"].to_numpy(float), eps, None)

    tau_cl = (Hd_est**2) * Ss_est / (np.pi**2 * kappa * K_est)
    tau_cl = np.clip(tau_cl, eps, None)

    if tau_units == "year":
        y_cl = np.log10(tau_cl / sec_per_year)
    else:
        y_cl = np.log10(tau_cl)

    mae_tau = float(np.nanmean(np.abs(y_tau - x_tau)))
    slope_tau, _, r2_tau = _linfit_stats(x_tau, y_tau)

    s_mae, mae_txt = _stat_ci(
        "tau_mae_log10",
        x_tau,
        y_tau,
        _mae,
        nd=2,
        seed_off=11,
        est=mae_tau,
    )
    s_r2, r2_txt = _stat_ci(
        "tau_r2",
        x_tau,
        y_tau,
        _r2,
        nd=3,
        seed_off=12,
        est=r2_tau,
    )
    s_sl, sl_txt = _stat_ci(
        "tau_slope",
        x_tau,
        y_tau,
        _slope,
        nd=2,
        seed_off=13,
        est=slope_tau,
    )

    # -------------------------
    # Panel (b): K recovery
    # -------------------------
    K_true = np.clip(df["K_true_mps"].to_numpy(float), eps, None)
    x_K = np.log10(K_true * sec_per_year)

    if k_from_tau:
        tau_est_sec = np.clip(
            df["tau_est_med_sec"].to_numpy(float),
            eps,
            None,
        )

        src = str(k_cl_source).strip().lower()
        if src == "true":
            Ss_use = np.clip(df["Ss_true"].to_numpy(float), eps, None)
            Hd_use = np.clip(df["Hd_true"].to_numpy(float), eps, None)
        elif src == "est":
            Ss_use = np.clip(df["Ss_est_med"].to_numpy(float), eps, None)
            Hd_use = np.clip(df["Hd_est_med"].to_numpy(float), eps, None)
        else:
            Ss_use = np.clip(df["Ss_prior"].to_numpy(float), eps, None)
            Hd_use = np.clip(df["Hd_prior"].to_numpy(float), eps, None)

        K_cl_mps = (Hd_use**2) * Ss_use / (
            np.pi**2 * kappa * tau_est_sec
        )
        y_K = np.log10(np.clip(K_cl_mps * sec_per_year, eps, None))

        title_b = "Permeability from closure"
        ylab_b = r"$\log_{10}\,K_{cl}(\hat{\tau})$ (m/yr)"
    else:
        y_K = np.log10(
            np.clip(
                df["K_est_med_m_per_year"].to_numpy(float),
                eps,
                None,
            )
        )
        title_b = "Permeability recovery"
        ylab_b = r"$\log_{10}\,\hat{K}$ (m/yr)"

    mae_K = float(np.nanmean(np.abs(y_K - x_K)))
    slope_K, _, r2_K = _linfit_stats(x_K, y_K)

    s_maeK, maeK_txt = _stat_ci(
        "K_mae_log10",
        x_K,
        y_K,
        _mae,
        nd=2,
        seed_off=31,
        est=mae_K,
    )
    s_r2K, r2K_txt = _stat_ci(
        "K_r2",
        x_K,
        y_K,
        _r2,
        nd=3,
        seed_off=32,
        est=r2_K,
    )
    s_slK, slK_txt = _stat_ci(
        "K_slope",
        x_K,
        y_K,
        _slope,
        nd=2,
        seed_off=33,
        est=slope_K,
    )

    # -------------------------
    # Panel (c): ridge check
    # -------------------------
    dK = df["vs_true_delta_K_q50"].to_numpy(float) / ln10
    dSs = df["vs_true_delta_Ss_q50"].to_numpy(float) / ln10
    dHd = df["vs_true_delta_Hd_q50"].to_numpy(float) / ln10

    ridge_x = dSs + 2.0 * dHd
    ridge_y = dK
    ridge_abs = np.abs(ridge_y - ridge_x)

    rho_ridge = _spearman(ridge_x, ridge_y)
    ridge_q50 = float(np.nanquantile(ridge_abs, 0.50))
    ridge_q95 = float(np.nanquantile(ridge_abs, 0.95))

    s_rhoR, rhoR_txt = _stat_ci(
        "ridge_spearman",
        ridge_x,
        ridge_y,
        _rho,
        nd=2,
        seed_off=41,
        est=rho_ridge,
    )

    # -------------------------
    # Panel (d): err vs metric
    # -------------------------
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

    s_rhoI, rhoI_txt = _stat_ci(
        "err_vs_metric_spearman",
        err_tau,
        y_m,
        _rho,
        nd=2,
        seed_off=42,
        est=rho_id,
    )

    # -------------------------
    # Summary outputs (out/)
    # -------------------------
    stats_rows: List[dict] = [
        s_mae, s_r2, s_sl,
        s_maeK, s_r2K, s_slK,
        s_rhoR, s_rhoI,
    ]

    meta = {
        "n_rows": int(len(df)),
        "tau_units": str(tau_units),
        "metric": str(metric),
        "k_from_tau": bool(k_from_tau),
        "k_cl_source": str(k_cl_source),
        "ci": float(ci),
        "n_boot": int(n_boot),
        "boot_seed": int(boot_seed),
        "ridge_q50": float(ridge_q50),
        "ridge_q95": float(ridge_q95),
    }

    out_csv_p = utils.resolve_out_out(out_csv)
    out_json_p = utils.resolve_out_out(out_json)

    pd.DataFrame(stats_rows).to_csv(out_csv_p, index=False)
    out_json_p.write_text(
        json.dumps(
            {"meta": meta, "stats": stats_rows},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # -------------------------
    # Plot styling + plotting
    # -------------------------
    lith = df["lith_idx"].to_numpy(int)
    lith_names = {0: "Fine", 1: "Mixed", 2: "Coarse", 3: "Rock"}
    markers = {0: "o", 1: "s", 2: "^", 3: "D"}

    def _beautify(ax: plt.Axes) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.6)

    def _panel(ax: plt.Axes, lab: str) -> None:
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

    if show_tau_cl:
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

    if show_panel_titles:
        axA.set_title("Timescale recovery" + nx_note, pad=2)

    if show_labels:
        axA.set_xlabel(xlab_a)
        axA.set_ylabel(ylab_a)
    else:
        axA.set_xlabel("")
        axA.set_ylabel("")

    if not show_ticklabels:
        axA.set_xticklabels([])
        axA.set_yticklabels([])

    if show_stats:
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

    if show_panel_titles:
        axB.set_title(title_b + nx_note, pad=2)

    if show_labels:
        axB.set_xlabel(r"$\log_{10}\,K_{true}$ (m/yr)")
        axB.set_ylabel(ylab_b)
    else:
        axB.set_xlabel("")
        axB.set_ylabel("")

    if not show_ticklabels:
        axB.set_xticklabels([])
        axB.set_yticklabels([])

    if show_stats:
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

    if show_panel_titles:
        axC.set_title("Degeneracy ridge check", pad=2)

    if show_labels:
        axC.set_xlabel(
            r"$\delta_{S_s}+2\,\delta_{H_d}$ (log$_{10}$)"
        )
        axC.set_ylabel(r"$\delta_K$ (log$_{10}$)")
    else:
        axC.set_xlabel("")
        axC.set_ylabel("")

    if not show_ticklabels:
        axC.set_xticklabels([])
        axC.set_yticklabels([])

    if show_stats:
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

    if show_panel_titles:
        axD.set_title("Error vs identifiability metric", pad=2)

    if show_labels:
        axD.set_xlabel(r"$|\Delta\log_{10}\tau|$")
        axD.set_ylabel(ylab_m)
    else:
        axD.set_xlabel("")
        axD.set_ylabel("")

    if not show_ticklabels:
        axD.set_xticklabels([])
        axD.set_yticklabels([])

    if show_stats:
        axD.text(
            0.03,
            0.97,
            f"Spearman r = {rhoI_txt}",
            transform=axD.transAxes,
            va="top",
            ha="left",
        )

    # shared legend
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

        if show_tau_cl:
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

    if show_title:
        ttl = utils.resolve_title(
            default="SM3 • Synthetic identifiability",
            title=title,
        )
        fig.suptitle(ttl, fontsize=11, fontweight="bold")

    # fig_p = utils.resolve_fig_out(out)
    # if fig_p.suffix:
    #     fig_p = fig_p.with_suffix("")

    # fig.savefig(str(fig_p) + ".png", dpi=int(dpi),
    #             bbox_inches="tight")
    # fig.savefig(str(fig_p) + ".svg", bbox_inches="tight")
    # plt.close(fig)
    utils.save_figure(fig, out, dpi= int(dpi))

    # print(f"[OK] figs -> {fig_p}.png/.svg")
    # print(f"[OK] table -> {out_csv_p}")
    # print(f"[OK] json  -> {out_json_p}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def plot_sm3_identifiability_main(
    argv: Optional[List[str]] = None,
) -> None:
    args = _parse_args(argv)

    utils.set_paper_style(
        fontsize=int(args.font),
        dpi=int(args.dpi),
    )

    df = pd.read_csv(Path(args.csv).expanduser())

    # strict filtering
    if args.only_identify is not None:
        if "identify" not in df.columns:
            raise KeyError(
                "--only-identify set but missing identify column."
            )
        want = str(args.only_identify).strip().lower()
        got = df["identify"].astype(str).str.strip().str.lower()
        df = df.loc[got == want].copy()

    if args.nx_min is not None:
        if "nx" not in df.columns:
            raise KeyError("--nx-min set but missing nx.")
        nxv = df["nx"].astype(int)
        df = df.loc[nxv >= int(args.nx_min)].copy()

    if df.empty:
        raise ValueError("No rows left after filtering.")

    # decide k_from_tau
    if args.k_from_tau == "auto":
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
    else:
        k_from_tau = utils.str_to_bool(args.k_from_tau)

    show_legend = utils.str_to_bool(args.show_legend, default=True)
    show_labels = utils.str_to_bool(args.show_labels, default=True)
    show_ticks = utils.str_to_bool(
        args.show_ticklabels,
        default=True,
    )
    show_title = utils.str_to_bool(args.show_title, default=True)
    show_pan_t = utils.str_to_bool(
        args.show_panel_titles,
        default=True,
    )

    show_ci = utils.str_to_bool(args.show_ci, default=True)
    show_stats = utils.str_to_bool(args.show_stats, default=True)
    show_prior = utils.str_to_bool(args.show_prior, default=True)
    show_tau_cl = utils.str_to_bool(args.show_tau_cl, default=True)

    plot_sm3_identifiability(
        df,
        tau_units=str(args.tau_units),
        metric=str(args.metric),
        k_from_tau=bool(k_from_tau),
        k_cl_source=str(args.k_cl_source),
        show_prior=show_prior,
        show_tau_cl=show_tau_cl,
        show_legend=show_legend,
        show_labels=show_labels,
        show_ticklabels=show_ticks,
        show_panel_titles=show_pan_t,
        show_title=show_title,
        title=args.title,
        show_stats=show_stats,
        show_ci=show_ci,
        n_boot=int(args.n_boot),
        ci=float(args.ci),
        boot_seed=int(args.boot_seed),
        out=str(args.out),
        out_csv=str(args.out_csv),
        out_json=str(args.out_json),
        dpi=int(args.dpi),
    )


def main(argv: Optional[List[str]] = None) -> None:
    plot_sm3_identifiability_main(argv)


if __name__ == "__main__":
    main()
