# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# scripts/plot_uncertainty_extras.py
#
# Supplementary Fig. S5 — Expanded uncertainty diagnostics.
#
# Outputs (new scripts layout)
# ---------------------------
# Figures:
#   scripts/figs/<out>.png
#   scripts/figs/<out>.pdf
#
# Tables:
#   scripts/out/<tables-stem>.csv
#   scripts/out/<tables-stem>.tex
#
# Inputs (per city)
# -----------------
# - Calibrated forecast CSV (val/test) with columns:
#     forecast_step,
#     subsidence_q10, subsidence_q50, subsidence_q90,
#     subsidence_actual
#
# - Optional GeoPrior phys JSON:
#     interval_calibration.factors_per_horizon
#
# Discovery (recommended)
# -----------------------
# Provide --ns-src / --zh-src and optionally --split.
# The script will auto-locate calibrated CSV + phys JSON.

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config as cfg
from . import utils


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------
def _smart_glob(path_or_glob: Optional[str]) -> Optional[Path]:
    """
    Resolve a path or glob.

    If wildcards are present, we pick the most recently
    modified match to reduce ambiguity when multiple runs
    exist under results/.
    """
    if not path_or_glob:
        return None

    s = str(path_or_glob).strip()
    if not s:
        return None

    if any(ch in s for ch in "*?[]"):
        cands = glob.glob(s, recursive=True)
        if not cands:
            return None
        cands.sort(
            key=lambda p: Path(p).stat().st_mtime,
            reverse=True,
        )
        return Path(cands[0]).expanduser()

    p = Path(s).expanduser()
    return p if p.exists() else None


def _extract_factors_per_horizon(
    meta: Optional[Dict[str, Any]],
) -> Optional[List[float]]:
    """
    Robustly extract horizon scale factors from GeoPrior JSON.

    Expected:
      interval_calibration.factors_per_horizon

    Accepts:
      - list[float]
      - dict like {"H1": 1.1, "H2": 0.9, ...}
    """
    if not meta:
        return None

    ic = meta.get("interval_calibration") or {}
    fac = ic.get("factors_per_horizon", None)
    if fac is None:
        return None

    if isinstance(fac, list):
        out = []
        for v in fac:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out or None

    if isinstance(fac, dict):
        items: List[Tuple[int, float]] = []
        for k, v in fac.items():
            ks = str(k).strip().lower()
            if ks.startswith("h"):
                ks = ks[1:]
            try:
                hi = int(ks)
                fv = float(v)
            except Exception:
                continue
            items.append((hi, fv))

        if not items:
            return None

        items.sort(key=lambda t: t[0])
        return [fv for _, fv in items]

    return None


# -------------------------------------------------------------------
# Core uncertainty helpers
# -------------------------------------------------------------------
def _coverage_sharpness_by_horizon(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizons: List[int] = []
    cov: List[float] = []
    shp: List[float] = []

    for h, g in df.groupby("forecast_step", sort=True):
        horizons.append(int(h))

        q10 = g["subsidence_q10"].to_numpy()
        q90 = g["subsidence_q90"].to_numpy()
        y = g["subsidence_actual"].to_numpy()

        cov.append(float(np.mean((y >= q10) & (y <= q90))))
        shp.append(float(np.mean(q90 - q10)))

    H = np.asarray(horizons, dtype=int)
    idx = np.argsort(H)

    return (
        H[idx],
        np.asarray(cov, dtype=float)[idx],
        np.asarray(shp, dtype=float)[idx],
    )


def _bootstrap_ci(
    x: np.ndarray,
    func,
    *,
    B: int,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    rng = rng or np.random.default_rng(42)
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        return (np.nan, np.nan)

    stats = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        idx = rng.integers(0, n, n)
        stats[b] = func(x[idx])

    lo, hi = np.quantile(
        stats,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return (float(lo), float(hi))


def _empirical_cdf_ks(u: np.ndarray) -> Tuple[float, float]:
    """
    KS statistic vs Uniform(0,1), with asymptotic p-value.

    p ≈ 2 Σ (-1)^{k-1} exp(-2 k² n D²), k=1..100
    """
    u = np.sort(np.asarray(u, dtype=float).ravel())
    n = u.size
    if n == 0:
        return (np.nan, np.nan)

    ecdf = np.arange(1, n + 1, dtype=float) / n
    diff1 = ecdf - u
    diff2 = u - (np.arange(n, dtype=float) / n)
    D = float(np.max(np.maximum(diff1, diff2)))

    s = 0.0
    for k in range(1, 101):
        s += ((-1) ** (k - 1)) * np.exp(
            -2.0 * (k * k) * n * (D * D)
        )

    p = max(0.0, min(1.0, 2.0 * s))
    return (D, p)


def _pit_from_three_quantiles(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Approximate PIT using a 3-point piecewise-linear CDF.

    CDF anchors:
      (q10 -> 0.1), (q50 -> 0.5), (q90 -> 0.9)
    with linear extrapolation and clamped tails.
    """
    q10 = np.asarray(q10, dtype=float).ravel()
    q50 = np.asarray(q50, dtype=float).ravel()
    q90 = np.asarray(q90, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    eps = 1e-8
    m1 = (0.5 - 0.1) / np.maximum(q50 - q10, eps)
    m2 = (0.9 - 0.5) / np.maximum(q90 - q50, eps)

    pit = np.empty_like(y, dtype=float)
    for i in range(y.size):
        if y[i] <= q50[i]:
            p = 0.1 + m1[i] * (y[i] - q10[i])
        else:
            p = 0.5 + m2[i] * (y[i] - q50[i])

        pit[i] = min(max(float(p), 0.0), 1.0)

    return pit


def _reliability_points(
    df: pd.DataFrame,
    taus: List[float],
    qcmap: Dict[float, str],
) -> List[Tuple[float, float]]:
    y = df["subsidence_actual"].to_numpy()
    pts: List[Tuple[float, float]] = []

    for t in taus:
        q = df[qcmap[t]].to_numpy()
        pts.append((t, float(np.mean(y <= q))))

    return pts


def _quantile_residuals(
    df: pd.DataFrame,
    taus: List[float],
    qcmap: Dict[float, str],
) -> Dict[float, np.ndarray]:
    out: Dict[float, np.ndarray] = {}

    for t in taus:
        res: List[float] = []
        for _, g in df.groupby("forecast_step", sort=True):
            emp = np.mean(
                g["subsidence_actual"].to_numpy()
                <= g[qcmap[t]].to_numpy()
            )
            res.append(float(emp - t))
        out[t] = np.asarray(res, dtype=float)

    return out


# -------------------------------------------------------------------
# Per-city summary
# -------------------------------------------------------------------
def summarize_city(
    df: pd.DataFrame,
    *,
    city: str,
    B: int,
    taus: Tuple[float, float, float] = (0.1, 0.5, 0.9),
) -> Dict[str, Any]:
    qc = {
        0.1: "subsidence_q10",
        0.5: "subsidence_q50",
        0.9: "subsidence_q90",
    }

    H, cov, shp = _coverage_sharpness_by_horizon(df)

    cov_ci: List[Tuple[float, float]] = []
    shp_ci: List[Tuple[float, float]] = []

    rng = np.random.default_rng(42)

    for h in H:
        g = df[df["forecast_step"] == h]
        q10 = g["subsidence_q10"].to_numpy()
        q90 = g["subsidence_q90"].to_numpy()
        y = g["subsidence_actual"].to_numpy()

        cov_vec = ((y >= q10) & (y <= q90)).astype(float)
        width = (q90 - q10).astype(float)

        cov_ci.append(
            _bootstrap_ci(
                cov_vec,
                np.mean,
                B=B,
                rng=rng,
            )
        )
        shp_ci.append(
            _bootstrap_ci(
                width,
                np.mean,
                B=B,
                rng=rng,
            )
        )

    rel = {
        int(h): _reliability_points(
            df[df["forecast_step"] == h],
            list(taus),
            qc,
        )
        for h in H
    }

    pit = _pit_from_three_quantiles(
        df["subsidence_q10"],
        df["subsidence_q50"],
        df["subsidence_q90"],
        df["subsidence_actual"],
    )
    ksD, ksp = _empirical_cdf_ks(pit)

    qres = _quantile_residuals(df, list(taus), qc)

    return {
        "city": city,
        "horizons": H,
        "coverage80": cov,
        "coverage80_ci": np.asarray(cov_ci, dtype=float),
        "sharpness80": shp,
        "sharpness80_ci": np.asarray(shp_ci, dtype=float),
        "reliability": rel,
        "pit": pit,
        "ks_D": ksD,
        "ks_p": ksp,
        "qresiduals": qres,
    }


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------
def plot_s5(
    *,
    nansha: Optional[Dict[str, Any]],
    zhongshan: Optional[Dict[str, Any]],
    factors_n: Optional[List[float]],
    factors_z: Optional[List[float]],
    out_stem: Path,
    dpi: int,
    show_legends: bool,
    show_labels: bool,
    show_ticklabels: bool,
    show_title: bool,
    show_panel_titles: bool,
    title: Optional[str],
) -> None:
    utils.set_paper_style(dpi=int(dpi))

    horizons: List[np.ndarray] = []
    if nansha is not None:
        horizons.append(nansha["horizons"])
    if zhongshan is not None:
        horizons.append(zhongshan["horizons"])

    if not horizons:
        raise SystemExit("No city data to plot.")

    H = np.unique(np.concatenate(horizons))
    nH = int(len(H))

    fig = plt.figure(figsize=(10.2, 8.6), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=max(nH, 3),
        height_ratios=[1.2, 1.0, 1.0, 1.0],
        hspace=0.65,
        wspace=0.35,
    )

    taus = [0.1, 0.5, 0.9]

    # A) Reliability small multiples
    for i, h in enumerate(H):
        ax = fig.add_subplot(gs[0, i])

        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            "--",
            color="#888888",
            lw=1.0,
            label="Ideal",
        )

        def _one(city: Optional[Dict[str, Any]], name: str):
            if city is None:
                return
            pts = city["reliability"].get(int(h), [])
            if not pts:
                return
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs,
                ys,
                marker="o",
                lw=1.0,
                color=cfg.CITY_COLORS[name],
                label=name,
            )

        _one(nansha, "Nansha")
        _one(zhongshan, "Zhongshan")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        if show_ticklabels:
            ax.set_xticks(taus)
            ax.set_yticks([0.0, 0.5, 1.0])
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if show_panel_titles:
            ax.set_title(
                f"Reliability • H{int(h)}",
                loc="left",
                fontweight="bold",
            )

        if i == 0 and show_labels:
            ax.set_ylabel("Empirical quantile")

        if i == nH - 1 and show_legends:
            ax.legend(
                frameon=False,
                loc="lower right",
            )

    # B) PIT histograms (2 panels)
    ax_pit_n = fig.add_subplot(gs[1, 0])
    if nansha is not None:
        ax_pit_n.hist(
            nansha["pit"],
            bins=20,
            range=(0.0, 1.0),
            color=cfg.CITY_COLORS["Nansha"],
            alpha=0.9,
            edgecolor="white",
        )
        if show_panel_titles:
            ax_pit_n.set_title(
                "PIT • Nansha "
                f"(KS={nansha['ks_D']:.2f}, "
                f"p≈{nansha['ks_p']:.2f})",
                loc="left",
                fontweight="bold",
            )
    else:
        ax_pit_n.set_axis_off()

    ax_pit_z = fig.add_subplot(gs[1, 1])
    if zhongshan is not None:
        ax_pit_z.hist(
            zhongshan["pit"],
            bins=20,
            range=(0.0, 1.0),
            color=cfg.CITY_COLORS["Zhongshan"],
            alpha=0.9,
            edgecolor="white",
        )
        if show_panel_titles:
            ax_pit_z.set_title(
                "PIT • Zhongshan "
                f"(KS={zhongshan['ks_D']:.2f}, "
                f"p≈{zhongshan['ks_p']:.2f})",
                loc="left",
                fontweight="bold",
            )
    else:
        ax_pit_z.set_axis_off()

    for ax in (ax_pit_n, ax_pit_z):
        if not show_ticklabels:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlim(0.0, 1.0)

        if show_labels:
            ax.set_xlabel("PIT")
            ax.set_ylabel("Count")
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")

    # C) Coverage vs horizon
    ax_cov = fig.add_subplot(gs[1, max(2, nH - 1) :])
    for city, name in ((nansha, "Nansha"), (zhongshan, "Zhongshan")):
        if city is None:
            continue
        Hc = city["horizons"]
        cov = city["coverage80"]
        ci = city["coverage80_ci"]

        ax_cov.plot(
            Hc,
            cov,
            marker="o",
            color=cfg.CITY_COLORS[name],
            label=name,
        )
        ax_cov.fill_between(
            Hc,
            ci[:, 0],
            ci[:, 1],
            color=cfg.CITY_COLORS[name],
            alpha=0.15,
        )

    ax_cov.axhline(
        0.8,
        ls="--",
        lw=1.0,
        color="#888888",
        label="Target 80%",
    )

    if show_panel_titles:
        ax_cov.set_title(
            "Coverage (80%) vs horizon",
            loc="left",
            fontweight="bold",
        )

    if show_labels:
        ax_cov.set_xlabel("Horizon (years)")
        ax_cov.set_ylabel("Coverage (prop.)")
    else:
        ax_cov.set_xlabel("")
        ax_cov.set_ylabel("")

    if not show_ticklabels:
        ax_cov.set_xticks([])
        ax_cov.set_yticks([])
    else:
        ax_cov.set_ylim(0.0, 1.0)

    if show_legends:
        ax_cov.legend(frameon=False, loc="lower right")

    # D) Sharpness vs horizon
    ax_shp = fig.add_subplot(gs[2, 0:2])
    for city, name in ((nansha, "Nansha"), (zhongshan, "Zhongshan")):
        if city is None:
            continue
        Hc = city["horizons"]
        shp = city["sharpness80"]
        ci = city["sharpness80_ci"]

        ax_shp.plot(
            Hc,
            shp,
            marker="o",
            color=cfg.CITY_COLORS[name],
            label=name,
        )
        ax_shp.fill_between(
            Hc,
            ci[:, 0],
            ci[:, 1],
            color=cfg.CITY_COLORS[name],
            alpha=0.15,
        )

    if show_panel_titles:
        ax_shp.set_title(
            "Sharpness (80% width) vs horizon",
            loc="left",
            fontweight="bold",
        )

    if show_labels:
        ax_shp.set_xlabel("Horizon (years)")
        ax_shp.set_ylabel("Width (units)")
    else:
        ax_shp.set_xlabel("")
        ax_shp.set_ylabel("")

    if not show_ticklabels:
        ax_shp.set_xticks([])
        ax_shp.set_yticks([])

    if show_legends:
        ax_shp.legend(frameon=False, loc="upper left")

    # E) Calibration factors (optional)
    ax_fac = fig.add_subplot(gs[2, max(2, nH - 1) :])
    have_fac = (factors_n is not None) or (factors_z is not None)

    if have_fac:
        x = np.arange(nH, dtype=float)
        w = 0.35

        if factors_n is not None:
            ax_fac.bar(
                x - w / 2.0,
                factors_n[:nH],
                width=w,
                color=cfg.CITY_COLORS["Nansha"],
                edgecolor="white",
                label="Nansha",
            )
        if factors_z is not None:
            ax_fac.bar(
                x + w / 2.0,
                factors_z[:nH],
                width=w,
                color=cfg.CITY_COLORS["Zhongshan"],
                edgecolor="white",
                label="Zhongshan",
            )

        if show_ticklabels:
            ax_fac.set_xticks(x)
            ax_fac.set_xticklabels([f"H{int(h)}" for h in H])
        else:
            ax_fac.set_xticks([])
            ax_fac.set_yticks([])

        if show_panel_titles:
            ax_fac.set_title(
                "Calibration factors (per horizon)",
                loc="left",
                fontweight="bold",
            )

        if show_labels:
            ax_fac.set_ylabel("Scale factor")
        else:
            ax_fac.set_ylabel("")

        if show_legends:
            ax_fac.legend(frameon=False)
    else:
        ax_fac.text(
            0.5,
            0.5,
            "No calibration factors found",
            ha="center",
            va="center",
            transform=ax_fac.transAxes,
        )
        ax_fac.set_axis_off()

    # F) Quantile residuals vs horizon
    ax_qr = fig.add_subplot(gs[3, :])
    for t, style in zip([0.1, 0.5, 0.9], ["--", "-", "--"]):
        if nansha is not None:
            ax_qr.plot(
                nansha["horizons"],
                nansha["qresiduals"][t],
                linestyle=style,
                marker="o",
                color=cfg.CITY_COLORS["Nansha"],
                label=f"Nansha Δ({t:.1f})",
            )
        if zhongshan is not None:
            ax_qr.plot(
                zhongshan["horizons"],
                zhongshan["qresiduals"][t],
                linestyle=style,
                marker="s",
                color=cfg.CITY_COLORS["Zhongshan"],
                label=f"Zhongshan Δ({t:.1f})",
            )

    ax_qr.axhline(0.0, color="#888888", lw=1.0)

    if show_panel_titles:
        ax_qr.set_title(
            "Quantile residuals by horizon",
            loc="left",
            fontweight="bold",
        )

    if show_labels:
        ax_qr.set_xlabel("Horizon (years)")
        ax_qr.set_ylabel("Residual")
    else:
        ax_qr.set_xlabel("")
        ax_qr.set_ylabel("")

    if not show_ticklabels:
        ax_qr.set_xticks([])
        ax_qr.set_yticks([])

    if show_legends:
        ax_qr.legend(frameon=False, ncol=3, loc="upper center")

    if show_title:
        fig.suptitle(
            utils.resolve_title(
                default=(
                    "Supplementary Fig. S5 • "
                    "Expanded uncertainty diagnostics"
                ),
                title=title,
            ),
            y=0.995,
            fontweight="bold",
        )
    
    utils.save_figure(fig, str(out_stem), dpi=int(dpi))
    # fig.savefig(str(out_stem) + ".png", bbox_inches="tight")
    # fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------
def write_tables(
    *,
    nansha: Optional[Dict[str, Any]],
    zhongshan: Optional[Dict[str, Any]],
    out_csv: Path,
    out_tex: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    for city in (nansha, zhongshan):
        if city is None:
            continue
        for i, h in enumerate(city["horizons"]):
            rows.append(
                {
                    "city": city["city"],
                    "horizon": int(h),
                    "coverage80": float(city["coverage80"][i]),
                    "coverage80_lo": float(
                        city["coverage80_ci"][i, 0]
                    ),
                    "coverage80_hi": float(
                        city["coverage80_ci"][i, 1]
                    ),
                    "sharpness80": float(city["sharpness80"][i]),
                    "sharpness80_lo": float(
                        city["sharpness80_ci"][i, 0]
                    ),
                    "sharpness80_hi": float(
                        city["sharpness80_ci"][i, 1]
                    ),
                    "ks_D": float(city["ks_D"]),
                    "ks_p": float(city["ks_p"]),
                }
            )

    df = pd.DataFrame(rows)
    utils.ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)

    utils.ensure_dir(out_tex.parent)
    with out_tex.open("w", encoding="utf-8") as f:
        f.write(
            "\\begin{table}[t]\\centering\\small\n"
        )
        f.write(
            "\\begin{tabular}{l r r r r r r r r r}"
            "\\toprule\n"
        )
        f.write(
            "City & H & Cov$_{80}$ & CI$_{lo}$ & "
            "CI$_{hi}$ & Sharp$_{80}$ & CI$_{lo}$ & "
            "CI$_{hi}$ & KS $D$ & KS $p$ \\\\\n"
        )
        f.write("\\midrule\n")

        for _, r in df.iterrows():
            f.write(
                f"{r['city']} & {int(r['horizon'])} & "
                f"{r['coverage80']:.3f} & "
                f"{r['coverage80_lo']:.3f} & "
                f"{r['coverage80_hi']:.3f} & "
                f"{r['sharpness80']:.3f} & "
                f"{r['sharpness80_lo']:.3f} & "
                f"{r['sharpness80_hi']:.3f} & "
                f"{r['ks_D']:.3f} & "
                f"{r['ks_p']:.3f} \\\\\n"
            )

        f.write("\\bottomrule\\end{tabular}\n")
        f.write(
            "\\caption{Expanded uncertainty diagnostics "
            "supporting Fig. 5.}\n"
        )
        f.write("\\label{tab:s5_uncertainty}\\end{table}\n")


# -------------------------------------------------------------------
# Artifact resolution (new structure)
# -------------------------------------------------------------------
def _pick_forecast_from_src(
    src: Any,
    *,
    split: str,
) -> Optional[Path]:
    art = utils.detect_artifacts(src)

    if split == "val":
        return art.forecast_val_csv
    if split == "test":
        return art.forecast_test_csv

    if art.forecast_test_csv is not None:
        return art.forecast_test_csv
    return art.forecast_val_csv


def _pick_phys_json_from_src(src: Any) -> Optional[Path]:
    # Prefer interpretable JSON when available.
    p = utils.find_phys_json(src)
    if p is not None:
        return p
    art = utils.detect_artifacts(src)
    return art.phys_json


# -------------------------------------------------------------------
# CLI (new API)
# -------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot_uncertainty_extras",
        description=(
            "Build Supplementary Fig. S5 + reliability "
            "table from calibrated forecasts."
        ),
    )

    utils.add_city_flags(p, default_both=True)

    p.add_argument(
        "--ns-src",
        type=str,
        default=None,
        help="Nansha run dir (auto-discover inputs).",
    )
    p.add_argument(
        "--zh-src",
        type=str,
        default=None,
        help="Zhongshan run dir (auto-discover inputs).",
    )

    p.add_argument(
        "--ns-forecast",
        type=str,
        default=None,
        help="Override Nansha calibrated forecast CSV.",
    )
    p.add_argument(
        "--zh-forecast",
        type=str,
        default=None,
        help="Override Zhongshan calibrated forecast CSV.",
    )

    p.add_argument(
        "--ns-phys-json",
        type=str,
        default=None,
        help="Override Nansha GeoPrior phys JSON.",
    )
    p.add_argument(
        "--zh-phys-json",
        type=str,
        default=None,
        help="Override Zhongshan GeoPrior phys JSON.",
    )

    p.add_argument(
        "--split",
        type=str,
        choices=["auto", "val", "test"],
        default="auto",
        help="When using --*-src, pick val/test files.",
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap draws for CIs.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=cfg.PAPER_DPI,
        help="Figure dpi.",
    )

    # Text toggles (Nature workflow)
    utils.add_plot_text_args(
        p,
        default_out="supp-fig-s5-uncertainty-extras",
    )

    # Backward-friendly switch
    p.add_argument(
        "--no-legends",
        action="store_true",
        help="Alias for --show-legend=false",
    )

    p.add_argument(
        "--tables-stem",
        type=str,
        default="supp-table-s5-reliability",
        help="Output stem for CSV/TEX (scripts/out/).",
    )

    return p.parse_args(argv)


def supp_figS5_uncertainty_extras_main(
    argv: List[str] | None = None,
) -> None:
    args = parse_args(argv)
    utils.ensure_script_dirs()

    cities = utils.resolve_cities(args)
    if not cities:
        cities = ["Nansha", "Zhongshan"]

    want_ns = "Nansha" in cities
    want_zh = "Zhongshan" in cities

    show_leg = utils.str_to_bool(args.show_legend, default=True)
    show_lbl = utils.str_to_bool(args.show_labels, default=True)
    show_tkl = utils.str_to_bool(args.show_ticklabels, default=True)
    show_tit = utils.str_to_bool(args.show_title, default=True)
    show_pt = utils.str_to_bool(
        args.show_panel_titles,
        default=True,
    )

    if args.no_legends:
        show_leg = False

    # Resolve Nansha inputs
    df_n = None
    fac_n = None
    if want_ns:
        fp = _smart_glob(args.ns_forecast)
        jp = _smart_glob(args.ns_phys_json)

        if fp is None and args.ns_src:
            fp = _pick_forecast_from_src(args.ns_src, split=args.split)
        if jp is None and args.ns_src:
            jp = _pick_phys_json_from_src(args.ns_src)

        if fp is not None:
            df_n = utils.load_forecast_csv(fp)

        jmeta = utils.safe_load_json(jp)
        fac_n = _extract_factors_per_horizon(jmeta)

    # Resolve Zhongshan inputs
    df_z = None
    fac_z = None
    if want_zh:
        fp = _smart_glob(args.zh_forecast)
        jp = _smart_glob(args.zh_phys_json)

        if fp is None and args.zh_src:
            fp = _pick_forecast_from_src(args.zh_src, split=args.split)
        if jp is None and args.zh_src:
            jp = _pick_phys_json_from_src(args.zh_src)

        if fp is not None:
            df_z = utils.load_forecast_csv(fp)

        jmeta = utils.safe_load_json(jp)
        fac_z = _extract_factors_per_horizon(jmeta)

    nansha = (
        summarize_city(df_n, city="Nansha", B=args.bootstrap)
        if df_n is not None
        else None
    )
    zhongshan = (
        summarize_city(df_z, city="Zhongshan", B=args.bootstrap)
        if df_z is not None
        else None
    )

    # Outputs
    fig_stem = utils.resolve_fig_out(args.out)
    if fig_stem.suffix:
        fig_stem = fig_stem.with_suffix("")

    tab_stem = utils.resolve_out_out(args.tables_stem)
    if tab_stem.suffix:
        tab_stem = tab_stem.with_suffix("")

    plot_s5(
        nansha=nansha,
        zhongshan=zhongshan,
        factors_n=fac_n,
        factors_z=fac_z,
        out_stem=fig_stem,
        dpi=int(args.dpi),
        show_legends=bool(show_leg),
        show_labels=bool(show_lbl),
        show_ticklabels=bool(show_tkl),
        show_title=bool(show_tit),
        show_panel_titles=bool(show_pt),
        title=args.title,
    )

    write_tables(
        nansha=nansha,
        zhongshan=zhongshan,
        out_csv=Path(str(tab_stem) + ".csv"),
        out_tex=Path(str(tab_stem) + ".tex"),
    )

    print(f"[OK] fig  : {fig_stem}.png/.pdf")
    print(f"[OK] table: {tab_stem}.csv/.tex")


def main(argv: List[str] | None = None) -> None:
    supp_figS5_uncertainty_extras_main(argv)


if __name__ == "__main__":
    main()
    
# Example runs

# Auto-discover from run folders:

# python -m scripts supp-fig-s5-uncertainty-extras \
#   --ns-src results/nansha_stage2_run \
#   --zh-src results/zhongshan_stage2_run \
#   --split auto


# Override CSVs directly:

# python -m scripts supp-fig-s5-uncertainty-extras \
#   --ns-forecast "results/**/nansha_*calibrated.csv" \
#   --zh-forecast "results/**/zhongshan_*calibrated.csv" \
#   --no-legends