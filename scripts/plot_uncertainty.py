# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# scripts/plot_uncertainty.py
#
# Figure 5 — Uncertainty calibration (reliability) + radial
# uncertainty summary.
#
# Outputs:
#   - scripts/figs/<out>.png + <out>.svg
#   - scripts/out/<out-csv>
#

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from . import config as cfg
from . import utils


_QMAP = {
    "subsidence_q10": 0.10,
    "subsidence_q50": 0.50,
    "subsidence_q90": 0.90,
}


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def quantile_reliability(
    df: pd.DataFrame,
    *,
    by_horizon: bool,
    ycol: str = "subsidence_actual",
) -> pd.DataFrame:
    """
    Empirical Pr(Y <= q_pred) versus nominal q.

    If by_horizon=True, returns:
      forecast_step, nominal, empirical
    else:
      nominal, empirical
    """
    rows: List[Dict[str, Any]] = []

    if by_horizon:
        for h, g in df.groupby("forecast_step"):
            for qc, q in _QMAP.items():
                emp = float(
                    np.mean(g[ycol].values <= g[qc].values)
                )
                rows.append(
                    {
                        "forecast_step": int(h),
                        "nominal": float(q),
                        "empirical": float(emp),
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["forecast_step", "nominal"])

    for qc, q in _QMAP.items():
        emp = float(
            np.mean(df[ycol].values <= df[qc].values)
        )
        rows.append({"nominal": float(q), "empirical": float(emp)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("nominal")


def interval_stats(
    df: pd.DataFrame,
    *,
    by_horizon: bool,
    lower: str = "subsidence_q10",
    upper: str = "subsidence_q90",
    ycol: str = "subsidence_actual",
) -> pd.DataFrame:
    """
    Coverage + sharpness for the 80% interval.

    If by_horizon=True:
      forecast_step, coverage80, sharpness80
    else:
      one-row DataFrame with coverage80, sharpness80
    """

    def _one(sub: pd.DataFrame) -> Dict[str, float]:
        l = sub[lower].values
        u = sub[upper].values
        y = sub[ycol].values
        cov = float(np.mean((y >= l) & (y <= u)))
        shp = float(np.mean(u - l))
        return {"coverage80": cov, "sharpness80": shp}

    if by_horizon:
        rows: List[Dict[str, Any]] = []
        for h, g in df.groupby("forecast_step"):
            rows.append({"forecast_step": int(h), **_one(g)})
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values("forecast_step")

    return pd.DataFrame([_one(df)])


def per_horizon_from_phys(
    meta: Dict[str, Any],
) -> pd.DataFrame:
    """
    Extract per-horizon coverage/sharpness from GeoPrior JSON.

    Expected:
      per_horizon:
        coverage80:  {"H1":..., "H2":...}
        sharpness80: {"H1":..., "H2":...}

    Returns empty DataFrame if missing.
    """
    if not meta:
        return pd.DataFrame()

    ph = meta.get("per_horizon", {}) or {}
    cov = ph.get("coverage80") or ph.get("coverage")
    shp = ph.get("sharpness80") or ph.get("sharpness")

    if not isinstance(cov, dict) or not isinstance(shp, dict):
        return pd.DataFrame()

    def _h(x: Any) -> int:
        s = str(x).strip()
        if s.upper().startswith("H"):
            s = s[1:]
        return int(float(s))

    rows: List[Dict[str, Any]] = []
    keys = sorted(cov.keys(), key=_h)
    for k in keys:
        if k not in shp:
            continue
        rows.append(
            {
                "forecast_step": _h(k),
                "coverage80": float(cov[k]),
                "sharpness80": float(shp[k]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("forecast_step")


def _pivot_emp(
    rel: pd.DataFrame,
    *,
    by_horizon: bool,
) -> pd.DataFrame:
    """
    Convert (nominal, empirical) rows to wide emp_q10/50/90.
    """
    if rel.empty:
        return pd.DataFrame()

    r = rel.copy()
    r["qkey"] = r["nominal"].map(
        {
            0.10: "emp_q10",
            0.50: "emp_q50",
            0.90: "emp_q90",
        }
    )

    if by_horizon:
        wide = r.pivot_table(
            index="forecast_step",
            columns="qkey",
            values="empirical",
            aggfunc="mean",
        )
        wide = wide.reset_index()
        return wide

    wide = r.pivot_table(
        index=None,
        columns="qkey",
        values="empirical",
        aggfunc="mean",
    )
    return wide.reset_index(drop=True)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def _axes_cleanup(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_reliability(
    ax: plt.Axes,
    rel: pd.DataFrame,
    *,
    city: str,
    color: str,
    show_point_values: bool,
    show_legend: bool,
    show_labels: bool,
    show_ticklabels: bool,
    show_panel_titles: bool,
    title_txt: str,
) -> None:
    x = np.array([0.0, 1.0])
    ax.plot(
        x,
        x,
        ls="--",
        lw=1.0,
        color="#777777",
        zorder=1,
    )

    rr = rel.sort_values("nominal")
    ax.plot(
        rr["nominal"],
        rr["empirical"],
        marker="o",
        lw=1.5,
        color=color,
        label=city,
        zorder=2,
    )

    if show_point_values:
        for a, b in rr[["nominal", "empirical"]].values:
            ax.text(
                float(a),
                float(b),
                f"{float(b):.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    if show_labels:
        ax.set_xlabel("Nominal quantile")
        ax.set_ylabel("Empirical probability")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if not show_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if show_panel_titles:
        ax.set_title(
            title_txt,
            loc="left",
            pad=6,
            fontweight="bold",
        )

    if show_legend:
        ax.legend(frameon=False, loc="lower right")

    _axes_cleanup(ax)

def plot_horizon_row(
    fig: plt.Figure,
    gs_row: List[Any],
    rel_items: List[Tuple[str, pd.DataFrame, str]],
    *,
    horizons: List[int],
    show_mini_titles: bool,
    show_mini_legend: bool,
    show_labels: bool,
    show_ticklabels: bool,
) -> None:
    """
    Row of mini reliability panels (H1..Hk).

    Each mini panel shows:
      - y=x diagonal
      - one curve per city in rel_items

    rel_items is:
      [(city_name, rel_by_h_df, color), ...]
    where rel_by_h_df has:
      forecast_step, nominal, empirical
    """
    if not horizons:
        return

    for i, h in enumerate(horizons):
        ax = fig.add_subplot(gs_row[i])

        x = np.array([0.0, 1.0])
        ax.plot(
            x,
            x,
            ls="--",
            lw=0.9,
            color="#999999",
            zorder=1,
        )

        for name, rel_h, color in rel_items:
            if rel_h.empty:
                continue

            sub = rel_h.loc[
                rel_h["forecast_step"] == int(h)
            ].sort_values("nominal")

            if sub.empty:
                continue

            ax.plot(
                sub["nominal"],
                sub["empirical"],
                marker="o",
                lw=1.2,
                color=color,
                label=name,
                zorder=2,
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        if show_mini_titles:
            ax.set_title(f"H{int(h)}", pad=4)

        if show_labels:
            ax.set_xlabel("Nominal")
            if i == 0:
                ax.set_ylabel("Empirical")
        else:
            ax.set_xlabel("")
            if i == 0:
                ax.set_ylabel("")

        if not show_ticklabels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            if i != 0:
                ax.set_yticklabels([])

        if i == 0 and show_mini_legend:
            ax.legend(frameon=False, loc="lower right")

        _axes_cleanup(ax)


def plot_radial(
    ax: plt.Axes,
    stats_h: pd.DataFrame,
    *,
    city: str,
    color: str,
    title_mode: str,
    show_ticklabels: bool,
    show_panel_titles: bool,
) -> None:
    if stats_h.empty:
        ax.set_axis_off()
        return

    steps = stats_h["forecast_step"].astype(int).values
    shp = stats_h["sharpness80"].values
    cov = stats_h["coverage80"].values

    n = max(1, len(steps))
    thetas = 2.0 * np.pi * np.arange(n) / float(n)

    edge_cols = [
        "#D62728" if float(c) < 0.8 else "#1F78B4"
        for c in cov
    ]

    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_facecolor("none")
    ax.grid(True, alpha=0.3)

    th_closed = np.r_[thetas, thetas[0]]
    shp_closed = np.r_[shp, shp[0]]

    ax.fill(
        th_closed,
        shp_closed,
        color=color,
        alpha=0.08,
        linewidth=0.0,
    )
    ax.plot(th_closed, shp_closed, lw=1.1, color=color)
    ax.scatter(
        thetas,
        shp,
        s=32,
        c=color,
        edgecolors=edge_cols,
        linewidths=1.2,
        zorder=3,
    )

    if len(shp) and np.any(np.isfinite(shp)):
        rmax = float(np.nanpercentile(shp, 95))
        rmax = max(1e-6, rmax)
        ax.set_rlim(0.0, rmax)
        ax.set_yticks(np.linspace(rmax / 4.0, rmax, 3))
        ax.set_yticklabels([])

    ax.set_xticks(thetas)
    if show_ticklabels:
        ax.set_xticklabels([f"H{h}" for h in steps])
    else:
        ax.set_xticklabels([])

    if not show_panel_titles:
        return

    if title_mode == "full":
        ttl = (
            f"{city}: sharpness by horizon\n"
            "(edge = coverage error vs 0.8)"
        )
    elif title_mode == "city":
        ttl = city
    else:
        ttl = ""

    if ttl:
        ax.set_title(ttl, va="bottom", fontsize=9)


def _annot_interval_calib(
    ax: plt.Axes,
    meta: Dict[str, Any],
    *,
    enabled: bool,
) -> None:
    if not enabled or not meta:
        return

    ical = meta.get("interval_calibration", {}) or {}
    if not ical:
        return

    uc = ical.get("coverage80_uncalibrated")
    ac = ical.get("coverage80_calibrated")
    us = ical.get("sharpness80_uncalibrated")
    cs = ical.get("sharpness80_calibrated")

    if uc is not None and ac is not None:
        ax.text(
            0.02,
            0.06,
            f"Cov80: {float(uc):.2f} → {float(ac):.2f}",
            transform=ax.transAxes,
            fontsize=8,
        )
    if us is not None and cs is not None:
        ax.text(
            0.02,
            0.01,
            f"Sharp80: {float(us):.3g} → {float(cs):.3g}",
            transform=ax.transAxes,
            fontsize=8,
        )


# ---------------------------------------------------------------------
# IO resolution
# ---------------------------------------------------------------------
def _pick_split_path(
    art: utils.Artifacts,
    split: str,
) -> Tuple[Optional[Path], str]:
    if split == "val":
        return (art.forecast_val_csv, "val")
    if split == "test":
        return (art.forecast_test_csv, "test")

    # auto: prefer test if present, else val
    if art.forecast_test_csv is not None:
        return (art.forecast_test_csv, "test")
    return (art.forecast_val_csv, "val")


def _resolve_city_inputs(
    *,
    city: str,
    src: Optional[str],
    forecast: Optional[str],
    phys_json: Optional[str],
    split: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], str, str]:
    """
    Returns:
      (forecast_df, phys_meta_mm, split_label, src_note)
    """
    if forecast:
        df = utils.load_forecast_csv(forecast)
        meta = utils.safe_load_json(phys_json)
        meta = utils.phys_json_to_mm(meta)
        return (df, meta, split, "manual-forecast")

    if not src:
        raise ValueError(
            f"{city}: provide --{city[:2].lower()}-src "
            f"or --{city[:2].lower()}-forecast"
        )

    art = utils.detect_artifacts(src)
    f_p, split_lab = _pick_split_path(art, split)

    if f_p is None:
        raise FileNotFoundError(
            f"{city}: no forecast CSV found under {src}"
        )

    df = utils.load_forecast_csv(f_p)

    pj = phys_json
    if pj is None:
        pj2 = utils.find_phys_json(src)
        pj = str(pj2) if pj2 else None

    meta = utils.safe_load_json(pj)
    meta = utils.phys_json_to_mm(meta)

    return (df, meta, split_lab, str(f_p))


def _build_metrics_table(
    *,
    city: str,
    split: str,
    rel_all: pd.DataFrame,
    rel_h: pd.DataFrame,
    int_all: pd.DataFrame,
    int_h: pd.DataFrame,
    int_source: str,
) -> pd.DataFrame:
    r0 = _pivot_emp(rel_all, by_horizon=False)
    rh = _pivot_emp(rel_h, by_horizon=True)

    if int_all.empty:
        cov0 = shp0 = float("nan")
    else:
        cov0 = float(int_all.iloc[0]["coverage80"])
        shp0 = float(int_all.iloc[0]["sharpness80"])

    base = {
        "city": city,
        "split": split,
        "interval_source": int_source,
    }

    rows: List[Dict[str, Any]] = []

    # overall row (forecast_step=0)
    rows.append(
        {
            **base,
            "forecast_step": 0,
            "coverage80": cov0,
            "sharpness80": shp0,
            "emp_q10": float(r0.get("emp_q10", np.nan)),
            "emp_q50": float(r0.get("emp_q50", np.nan)),
            "emp_q90": float(r0.get("emp_q90", np.nan)),
        }
    )

    # per-horizon rows
    if not rh.empty:
        for _, rr in rh.iterrows():
            h = int(rr["forecast_step"])
            sub = int_h.loc[int_h["forecast_step"] == h]
            if sub.empty:
                cov = shp = float("nan")
            else:
                cov = float(sub.iloc[0]["coverage80"])
                shp = float(sub.iloc[0]["sharpness80"])

            rows.append(
                {
                    **base,
                    "forecast_step": h,
                    "coverage80": cov,
                    "sharpness80": shp,
                    "emp_q10": float(rr.get("emp_q10", np.nan)),
                    "emp_q50": float(rr.get("emp_q50", np.nan)),
                    "emp_q90": float(rr.get("emp_q90", np.nan)),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plot (Figure 5)
# ---------------------------------------------------------------------
def plot_fig5_uncertainty(
    *,
    ns_df: Optional[pd.DataFrame],
    zh_df: Optional[pd.DataFrame],
    ns_meta: Optional[Dict[str, Any]],
    zh_meta: Optional[Dict[str, Any]],
    split_ns: str,
    split_zh: str,
    out: str,
    out_csv: str,
    dpi: int,
    font: int,
    show_legend: bool,
    show_labels: bool,
    show_ticklabels: bool,
    show_title: bool,
    show_panel_titles: bool,
    show_point_values: bool,
    show_mini_titles: bool,
    show_mini_legend: bool,
    show_json_notes: bool,
    radial_title: str,
    title: Optional[str],
) -> None:
    """
    Fig. 5:
      - Top row: per-city reliability (overall)
      - Mid row: per-horizon mini reliability (all cities)
      - Bot row: per-city radial sharpness (edge cov err)

    Supports 1-city and 2-city layouts.
    """
    cities: List[Dict[str, Any]] = []

    if ns_df is not None:
        cities.append(
            {
                "name": "Nansha",
                "df": ns_df,
                "meta": ns_meta or {},
                "split": split_ns,
                "color": cfg.CITY_COLORS["Nansha"],
            }
        )

    if zh_df is not None:
        cities.append(
            {
                "name": "Zhongshan",
                "df": zh_df,
                "meta": zh_meta or {},
                "split": split_zh,
                "color": cfg.CITY_COLORS["Zhongshan"],
            }
        )

    if not cities:
        raise ValueError("No city data provided.")

    utils.ensure_script_dirs()
    utils.set_paper_style(
        fontsize=int(font),
        dpi=int(dpi),
    )

    # -----------------------------
    # Compute metrics per city
    # -----------------------------
    for c in cities:
        df = c["df"]
        meta = c["meta"]

        c["rel_all"] = quantile_reliability(
            df,
            by_horizon=False,
        )
        c["rel_h"] = quantile_reliability(
            df,
            by_horizon=True,
        )

        st_h = per_horizon_from_phys(meta)
        c["int_source"] = (
            "phys_json" if not st_h.empty else "csv"
        )

        if st_h.empty:
            st_h = interval_stats(df, by_horizon=True)

        c["int_h"] = st_h
        c["int_all"] = interval_stats(df, by_horizon=False)

    # -----------------------------
    # Horizons (union)
    # -----------------------------
    horizons_set = set()
    for c in cities:
        rh = c["rel_h"]
        if rh.empty:
            continue
        horizons_set |= set(rh["forecast_step"].astype(int))

    horizons = sorted(horizons_set) if horizons_set else [1]

    # -----------------------------
    # Figure sizing
    # -----------------------------
    n_h = max(1, len(horizons))
    n_c = max(1, len(cities))
    ncols = max(2, n_h, n_c)

    if ncols <= 7:
        fig_w = 9.0
    else:
        fig_w = min(13.0, 1.25 * float(ncols))

    fig = plt.figure(figsize=(fig_w, 8.2))

    # -----------------------------
    # Layout:
    # outer grid = 3 rows
    # each row is a subgrid
    # -----------------------------
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[2.2, 1.4, 2.2],
        hspace=0.55,
    )

    gs_top = gs[0].subgridspec(
        1,
        n_c,
        wspace=0.30,
    )
    gs_mid = gs[1].subgridspec(
        1,
        n_h,
        wspace=0.15,
    )
    gs_bot = gs[2].subgridspec(
        1,
        n_c,
        wspace=0.35,
    )

    # -----------------------------
    # Top row: reliability per city
    # -----------------------------
    for i, c in enumerate(cities):
        ax = fig.add_subplot(gs_top[0, i])

        t = (
            f"Reliability • {c['name']} "
            f"({c['split']})"
        )

        plot_reliability(
            ax,
            c["rel_all"],
            city=c["name"],
            color=c["color"],
            show_point_values=show_point_values,
            show_legend=show_legend,
            show_labels=show_labels,
            show_ticklabels=show_ticklabels,
            show_panel_titles=show_panel_titles,
            title_txt=t,
        )

        _annot_interval_calib(
            ax,
            c["meta"],
            enabled=show_json_notes,
        )

    # -----------------------------
    # Middle row: mini horizon row
    # -----------------------------
    rel_items: List[Tuple[str, pd.DataFrame, str]] = []
    for c in cities:
        rel_items.append(
            (c["name"], c["rel_h"], c["color"])
        )

    gs_row = [gs_mid[0, i] for i in range(n_h)]
    plot_horizon_row(
        fig,
        gs_row,
        rel_items,
        horizons=horizons,
        show_mini_titles=show_mini_titles,
        show_mini_legend=(
            show_legend and show_mini_legend
        ),
        show_labels=show_labels,
        show_ticklabels=show_ticklabels,
    )

    # -----------------------------
    # Bottom row: radial per city
    # -----------------------------
    for i, c in enumerate(cities):
        ax = fig.add_subplot(
            gs_bot[0, i],
            projection="polar",
        )

        plot_radial(
            ax,
            c["int_h"],
            city=c["name"],
            color=c["color"],
            title_mode=radial_title,
            show_ticklabels=show_ticklabels,
            show_panel_titles=show_panel_titles,
        )

    # -----------------------------
    # Title
    # -----------------------------
    if show_title:
        ttl = utils.resolve_title(
            default="Fig. 5 — Uncertainty calibration",
            title=title,
        )
        fig.suptitle(ttl, x=0.02, ha="left")

    # -----------------------------
    # Save figure
    # -----------------------------
    fig_p = utils.resolve_fig_out(out)
    if fig_p.suffix:
        fig_p = fig_p.with_suffix("")

    fig.savefig(
        str(fig_p) + ".png",
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        str(fig_p) + ".svg",
        bbox_inches="tight",
    )
    plt.close(fig)

    # -----------------------------
    # Export metrics table
    # -----------------------------
    tabs: List[pd.DataFrame] = []
    for c in cities:
        tabs.append(
            _build_metrics_table(
                city=c["name"],
                split=c["split"],
                rel_all=c["rel_all"],
                rel_h=c["rel_h"],
                int_all=c["int_all"],
                int_h=c["int_h"],
                int_source=c["int_source"],
            )
        )

    tbl = pd.concat(tabs, ignore_index=True)
    out_csv_p = utils.resolve_out_out(out_csv)
    tbl.to_csv(out_csv_p, index=False)

    print(f"[OK] wrote {fig_p}.png/.svg")
    print(f"[OK] wrote {out_csv_p}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _add_plot_fig5_args(ap: argparse.ArgumentParser) -> None:
    utils.add_city_flags(ap, default_both=True)

    ap.add_argument(
        "--ns-src",
        type=str,
        default=None,
        help="Nansha results dir (auto-detect artifacts).",
    )
    ap.add_argument(
        "--zh-src",
        type=str,
        default=None,
        help="Zhongshan results dir (auto-detect artifacts).",
    )

    ap.add_argument(
        "--ns-forecast",
        type=str,
        default=None,
        help="Override Nansha calibrated forecast CSV.",
    )
    ap.add_argument(
        "--zh-forecast",
        type=str,
        default=None,
        help="Override Zhongshan calibrated forecast CSV.",
    )

    ap.add_argument(
        "--ns-phys-json",
        type=str,
        default=None,
        help="Override GeoPrior phys JSON (Nansha).",
    )
    ap.add_argument(
        "--zh-phys-json",
        type=str,
        default=None,
        help="Override GeoPrior phys JSON (Zhongshan).",
    )

    ap.add_argument(
        "--split",
        type=str,
        choices=["auto", "val", "test"],
        default="auto",
        help="Which calibrated CSV to use (default auto).",
    )

    ap.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)
    ap.add_argument("--font", type=int, default=cfg.PAPER_FONT)

    ap.add_argument(
        "--out-csv",
        type=str,
        default="ext-table-fig5-uncertainty.csv",
    )

    ap.add_argument(
        "--radial-title",
        type=str,
        choices=["full", "city", "none"],
        default="full",
    )

    ap.add_argument(
        "--show-point-values",
        type=str,
        default="true",
        help="Show values next to reliability points.",
    )
    ap.add_argument(
        "--show-mini-titles",
        type=str,
        default="true",
        help="Show H1..Hk in mini panels.",
    )
    ap.add_argument(
        "--show-mini-legend",
        type=str,
        default="true",
        help="Show legend in first mini panel.",
    )
    ap.add_argument(
        "--show-json-notes",
        type=str,
        default="true",
        help="Show Cov/Sharp notes if JSON has them.",
    )

    utils.add_plot_text_args(
        ap,
        default_out="fig5-uncertainty",
    )


def plot_fig5_uncertainty_main(
    argv: Optional[List[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        prog="plot-uncertainty",
        description="Fig. 5 uncertainty calibration + radial.",
    )
    _add_plot_fig5_args(ap)
    args = ap.parse_args(argv)

    # Text toggles (common, consistent across scripts)
    show_legend = utils.str_to_bool(args.show_legend, default=True)
    show_labels = utils.str_to_bool(args.show_labels, default=True)
    show_ticks = utils.str_to_bool(
        args.show_ticklabels,
        default=True,
    )
    show_title = utils.str_to_bool(args.show_title, default=True)
    show_pt = utils.str_to_bool(
        args.show_panel_titles,
        default=True,
    )

    # Fig5-specific toggles
    show_pv = utils.str_to_bool(
        args.show_point_values,
        default=True,
    )
    show_mt = utils.str_to_bool(
        args.show_mini_titles,
        default=True,
    )
    show_ml = utils.str_to_bool(
        args.show_mini_legend,
        default=True,
    )
    show_jn = utils.str_to_bool(
        args.show_json_notes,
        default=True,
    )

    cities = utils.resolve_cities(args)
    if not cities:
        cities = ["Nansha", "Zhongshan"]

    # # For this figure we keep both city panels; if user
    # # selects only one, we still require the other inputs
    # # unless you later decide to make a 1-city layout.
    # if "Nansha" not in cities or "Zhongshan" not in cities:
    #     raise ValueError(
    #         "Fig. 5 layout expects both cities. "
    #         "Use default cities (ns,zh)."
    #     )
    
    want_ns = "Nansha" in cities
    want_zh = "Zhongshan" in cities

    if not want_ns and not want_zh:
        want_ns = True
        want_zh = True

    ns_df = ns_meta = None
    zh_df = zh_meta = None
    ns_split = "auto"
    zh_split = "auto"

    if want_ns:
        ns_df, ns_meta, ns_split, _ = _resolve_city_inputs(
            city="Nansha",
            src=args.ns_src,
            forecast=args.ns_forecast,
            phys_json=args.ns_phys_json,
            split=args.split,
        )

    if want_zh:
        zh_df, zh_meta, zh_split, _ = _resolve_city_inputs(
            city="Zhongshan",
            src=args.zh_src,
            forecast=args.zh_forecast,
            phys_json=args.zh_phys_json,
            split=args.split,
        )

    plot_fig5_uncertainty(
        ns_df=ns_df,
        zh_df=zh_df,
        ns_meta=ns_meta,
        zh_meta=zh_meta,
        split_ns=ns_split,
        split_zh=zh_split,
        out=args.out,
        out_csv=args.out_csv,
        dpi=int(args.dpi),
        font=int(args.font),
        show_legend=show_legend,
        show_labels=show_labels,
        show_ticklabels=show_ticks,
        show_title=show_title,
        show_panel_titles=show_pt,
        show_point_values=show_pv,
        show_mini_titles=show_mt,
        show_mini_legend=show_ml,
        show_json_notes=show_jn,
        radial_title=args.radial_title,
        title=args.title,
    )


def main(argv: Optional[List[str]] = None) -> None:
    plot_fig5_uncertainty_main(argv)


if __name__ == "__main__":
    main()
    
# Example runs (auto-pick test if present, else val):

# python -m scripts.plot_uncertainty \
#   --ns-src results/nansha_stage2_run \
#   --zh-src results/zhongshan_stage2_run \
#   --split auto


# Force test:

# python -m scripts.plot_uncertainty \
#   --ns-src results/nansha_stage2_run \
#   --zh-src results/zhongshan_stage2_run \
#   --split test