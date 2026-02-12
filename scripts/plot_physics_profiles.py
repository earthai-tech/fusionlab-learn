# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# nat.com/figs/make_appendix_figA1_phys_profiles.py
#
r"""
Appendix Fig. A1 • 1D physics sensitivity profiles

Complements:
- Supp. Fig. S6 (ablations: skill + coverage),
- Supp. Fig. S7 (2D residual heatmaps),

by providing 1D "profiles" (physics-on only):

For each city (rows) and each metric (metric_prior, metric_cons),
we draw two line profiles:

- metric vs λ_prior  (mean over λ_cons, pde_mode='both')
- metric vs λ_cons   (mean over λ_prior, pde_mode='both')

Layout
------
2 rows × 4 columns:

Row 1: City A (default: Nansha)
  Col 1: metric_prior vs λ_prior
  Col 2: metric_prior vs λ_cons
  Col 3: metric_cons  vs λ_prior
  Col 4: metric_cons  vs λ_cons

Row 2: City B (default: Zhongshan)
  same four panels.

Data source
-----------
We scan for JSONL under:
  <root>/**/ablation_records/ablation_record*.jsonl

Outputs
-------
- Figure: <out>.png and <out>.pdf
- Tidy table copy:
    appendix_table_A1_phys_profiles_tidy.csv
  written next to the figure.

API conventions
---------------
- Style via scripts.utils.set_paper_style()
- Output path via scripts.utils.resolve_fig_out()
- JSONL discovery via cfg.PATTERNS["ablation_record_jsonl"] +
  scripts.utils.find_all()
- main(argv) wrapper calls a *_main(argv) function.

Linting / format
----------------
- black + ruff, line length <= 62
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts import config as cfg
from scripts import utils


# Metrics where "lower is better" (best point highlight).
_LOWER_IS_BETTER = {
    "mae",
    "mse",
    "sharpness80",
    "epsilon_prior",
    "epsilon_cons",
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot-physics-profiles",
        description="Appendix A1: 1D physics profiles",
    )

    p.add_argument(
        "--root",
        type=str,
        default="results",
        help=(
            "Root to scan for "
            "**/ablation_records/ablation_record*.jsonl"
        ),
    )

    # Backward compat: explicit out dir override.
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output dir override.",
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
        "--city-a",
        type=str,
        default="Nansha",
        help="City A name as recorded in JSONL.",
    )
    p.add_argument(
        "--city-b",
        type=str,
        default="Zhongshan",
        help="City B name as recorded in JSONL.",
    )

    p.add_argument(
        "--metric-prior",
        type=str,
        default="epsilon_prior",
        choices=[
            "epsilon_prior",
            "coverage80",
            "sharpness80",
            "r2",
            "mae",
            "mse",
        ],
        help="Metric used for prior profiles.",
    )
    p.add_argument(
        "--metric-cons",
        type=str,
        default="epsilon_cons",
        choices=[
            "epsilon_cons",
            "coverage80",
            "sharpness80",
            "r2",
            "mae",
            "mse",
        ],
        help="Metric used for cons profiles.",
    )

    utils.add_plot_text_args(
        p,
        default_out="appendix_fig_A1_phys_profiles",
    )

    # Optional highlight of best point per panel.
    p.add_argument(
        "--show-best",
        type=str,
        default="true",
        help="Mark best point (true/false).",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------
def _read_records(root: Path) -> pd.DataFrame:
    rows: List[dict] = []

    files = utils.find_all(
        root,
        cfg.PATTERNS.get("ablation_record_jsonl", ()),
    )

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        rec = json.loads(s)
                    except Exception:
                        continue
                    if isinstance(rec, dict):
                        rec["_src"] = str(fp)
                        rows.append(rec)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["lambda_prior", "lambda_cons"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _city_mask(df: pd.DataFrame, city: str) -> pd.Series:
    return (
        df["city"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(str(city).strip().lower())
    )


def _metric_label(name: str) -> str:
    k = str(name).strip()

    # Prefer central physics labels when available.
    if k in cfg.PHYS_LABELS:
        return cfg.PHYS_LABELS[k]

    kl = k.lower()
    if kl == "coverage80":
        return "Coverage (80%)"
    if kl == "sharpness80":
        return "Sharpness (80%)"
    if kl == "r2":
        return "R\u00b2"
    if kl == "mae":
        return "MAE"
    if kl == "mse":
        return "MSE"

    return k


def _axes_cleanup(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _only_physics_on(df: pd.DataFrame) -> pd.DataFrame:
    # Some JSONLs may omit pde_mode. In that case we
    # keep all rows (treat as "physics on").
    if "pde_mode" not in df.columns:
        return df
    return df.loc[df["pde_mode"].astype(str).eq("both")]


def _profile_over(
    df: pd.DataFrame,
    *,
    city: str,
    metric: str,
    axis: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if axis not in ("lambda_prior", "lambda_cons"):
        raise ValueError(f"bad axis: {axis!r}")

    sub = df.loc[_city_mask(df, city)].copy()
    sub = _only_physics_on(sub)

    if sub.empty or metric not in sub.columns:
        return np.array([]), np.array([])

    g = (
        sub.groupby(axis, dropna=True)[metric]
        .mean()
        .reset_index()
        .dropna(subset=[axis])
        .sort_values(axis)
    )
    if g.empty:
        return np.array([]), np.array([])

    x = g[axis].to_numpy(dtype=float)
    y = pd.to_numeric(
        g[metric],
        errors="coerce",
    ).to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _best_idx(y: np.ndarray, *, metric: str) -> Optional[int]:
    if y.size == 0:
        return None
    if not np.isfinite(y).any():
        return None
    try:
        if metric.lower() in _LOWER_IS_BETTER:
            return int(np.nanargmin(y))
        return int(np.nanargmax(y))
    except Exception:
        return None


def _plot_profile_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    city: str,
    metric: str,
    axis: str,
    color: str,
    show_best: bool,
    show_labels: bool,
    show_ticks: bool,
    show_title: bool,
) -> None:
    x, y = _profile_over(
        df,
        city=city,
        metric=metric,
        axis=axis,
    )
    if x.size == 0:
        ax.set_axis_off()
        return

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=1.0,
        color=color,
    )

    if show_best:
        idx = _best_idx(y, metric=metric)
        if idx is not None:
            ax.scatter(
                [x[idx]],
                [y[idx]],
                s=35,
                facecolors="none",
                edgecolors="black",
                linewidths=1.0,
            )

    if show_ticks:
        ax.tick_params(axis="both", which="both")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if show_labels:
        xl = (
            r"$\lambda_{\mathrm{prior}}$"
            if axis == "lambda_prior"
            else r"$\lambda_{\mathrm{cons}}$"
        )
        ax.set_xlabel(xl)
        ax.set_ylabel(_metric_label(metric))
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if show_title:
        which = (
            r"vs $\lambda_{\mathrm{prior}}$"
            if axis == "lambda_prior"
            else r"vs $\lambda_{\mathrm{cons}}$"
        )
        ax.set_title(
            f"{city} — {_metric_label(metric)} {which}",
            loc="left",
            pad=4,
            fontweight="bold",
        )

    _axes_cleanup(ax)


def _resolve_out(
    *,
    out: str,
    out_dir: Optional[str],
) -> Path:
    if out_dir:
        base = Path(out_dir).expanduser()
        return (base / Path(out).expanduser()).resolve()
    return utils.resolve_fig_out(out)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def figA1_phys_profiles_main(
    argv: List[str] | None = None,
) -> None:
    args = _parse_args(argv)

    utils.set_paper_style(
        fontsize=int(args.font),
        dpi=int(args.dpi),
    )

    show_labels = utils.str_to_bool(
        args.show_labels,
        default=True,
    )
    show_ticks = utils.str_to_bool(
        args.show_ticklabels,
        default=True,
    )
    show_title = utils.str_to_bool(
        args.show_title,
        default=True,
    )
    show_pan_t = utils.str_to_bool(
        args.show_panel_titles,
        default=True,
    )
    show_best = utils.str_to_bool(
        args.show_best,
        default=True,
    )

    root = utils.as_path(args.root)
    df = _read_records(root)

    if df.empty:
        raise SystemExit(
            "No ablation_record*.jsonl found under:\n"
            f"  {root.resolve()}\n"
            "Run ablations with the logger enabled first."
        )

    city_a = utils.canonical_city(args.city_a)
    city_b = utils.canonical_city(args.city_b)

    out = _resolve_out(out=args.out, out_dir=args.out_dir)
    utils.ensure_dir(out.parent)

    tidy_csv = (
        out.parent / "appendix_table_A1_phys_profiles_tidy.csv"
    )
    df.to_csv(tidy_csv, index=False)
    print(f"[OK] table -> {tidy_csv}")

    ca = cfg.CITY_COLORS.get(city_a, "#1F78B4")
    cb = cfg.CITY_COLORS.get(city_b, "#E31A1C")

    fig = plt.figure(figsize=(10.0, 6.5))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        left=0.06,
        right=0.98,
        top=0.92,
        bottom=0.10,
        hspace=0.35,
        wspace=0.35,
    )

    # Row 1: city A
    _plot_profile_panel(
        fig.add_subplot(gs[0, 0]),
        df,
        city=city_a,
        metric=args.metric_prior,
        axis="lambda_prior",
        color=ca,
        show_best=show_best,
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )
    _plot_profile_panel(
        fig.add_subplot(gs[0, 1]),
        df,
        city=city_a,
        metric=args.metric_prior,
        axis="lambda_cons",
        color=ca,
        show_best=show_best,
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )
    _plot_profile_panel(
        fig.add_subplot(gs[0, 2]),
        df,
        city=city_a,
        metric=args.metric_cons,
        axis="lambda_prior",
        color=ca,
        show_best=show_best,
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )
    _plot_profile_panel(
        fig.add_subplot(gs[0, 3]),
        df,
        city=city_a,
        metric=args.metric_cons,
        axis="lambda_cons",
        color=ca,
        show_best=show_best,
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )

    # Row 2: city B
    _plot_profile_panel(
        fig.add_subplot(gs[1, 0]),
        df,
        city=city_b,
        metric=args.metric_prior,
        axis="lambda_prior",
        color=cb,
        show_best=show_best,
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )
    _plot_profile_panel(
        fig.add_subplot(gs[1, 1]),
        df,
        city=city_b,
        metric=args.metric_prior,
        axis="lambda_cons",
        color=cb,
        show_best=show_best,
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )
    _plot_profile_panel(
        fig.add_subplot(gs[1, 2]),
        df,
        city=city_b,
        metric=args.metric_cons,
        axis="lambda_prior",
        color=cb,
        show_best=show_best,
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )
    _plot_profile_panel(
        fig.add_subplot(gs[1, 3]),
        df,
        city=city_b,
        metric=args.metric_cons,
        axis="lambda_cons",
        color=cb,
        show_best=show_best,
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )

    if show_title:
        default = (
            "Appendix Fig. A1 • 1D physics sensitivity profiles"
        )
        ttl = utils.resolve_title(
            default=default,
            title=args.title,
        )
        fig.suptitle(
            ttl,
            fontsize=11,
            fontweight="bold",
        )

    png = out.with_suffix(".png")
    pdf = out.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[OK] figs -> {png} | {pdf}")


def main(argv: List[str] | None = None) -> None:
    figA1_phys_profiles_main(argv)


if __name__ == "__main__":
    main()
