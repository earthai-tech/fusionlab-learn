# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# nat.com/figs/make_supp_figS6_ablations.py
#
r"""
Supplement S6 • Extended ablations & sensitivities

This figure supports the main ablation figure (Fig. 2) by showing how
predictive performance and uncertainty/physics metrics vary across
physics hyper-parameters.

Layout
------
Top row (bar strips)
  Per-city bar strips of a chosen "skill" metric (default: r2) versus
  lambda_prior, with separate bars for pde_mode in {"none","both"}.

Bottom row (heatmaps)
  Per-city heatmaps over (lambda_cons, lambda_prior) of a chosen metric
  (default: coverage80). Best cell is marked (min/max depending on
  metric). A shared colourbar can be shown/hidden.

Data source
-----------
We scan for JSONL under:
  <root>/**/ablation_records/ablation_record*.jsonl

Each line is one JSON record (dict), typically containing:
  city, pde_mode, lambda_cons, lambda_prior, r2, coverage80, ...

Outputs
-------
- Figure: <out>.png and <out>.pdf
- Tidy table copy: tableS6_ablations_tidy.csv (next to figure)

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
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from . import utils


# Metrics where "lower is better" (best cell = min).
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
        prog="plot-ablations-sensitivity",
        description="Supplement S6: Ablations & sensitivities",
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

    # Backward compat: explicit output directory override.
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

    # Which metrics to display
    p.add_argument(
        "--bar-metric",
        type=str,
        default="r2",
        choices=[
            "r2",
            "mae",
            "mse",
            "coverage80",
            "sharpness80",
            "epsilon_prior",
            "epsilon_cons",
        ],
        help="Metric for top-row bar plots.",
    )
    p.add_argument(
        "--heatmap-metric",
        type=str,
        default="coverage80",
        choices=[
            "r2",
            "mae",
            "mse",
            "coverage80",
            "sharpness80",
            "epsilon_prior",
            "epsilon_cons",
        ],
        help="Metric for bottom-row heatmaps.",
    )

    # Paper editing toggles via our standard args
    utils.add_plot_text_args(
        p,
        default_out="supp_fig_S6_ablations",
    )

    # Script-specific toggles
    p.add_argument(
        "--mute-values",
        type=str,
        default="false",
        help="Annotate values above bars (true/false).",
    )
    p.add_argument(
        "--no-colorbar",
        type=str,
        default="false",
        help="Show shared heatmap colourbar (true/false).",
    )

    # City labels kept explicit for consistency with old S6.
    p.add_argument(
        "--city-a",
        type=str,
        default="Nansha",
        help="First city name as recorded in JSONL.",
    )
    p.add_argument(
        "--city-b",
        type=str,
        default="Zhongshan",
        help="Second city name as recorded in JSONL.",
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


def _present_modes(series: Iterable[str]) -> List[str]:
    # Keep stable order for visual consistency.
    seen = set(str(x) for x in series)
    out: List[str] = []
    for m in ("none", "both"):
        if m in seen:
            out.append(m)
    return out


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


def _best_ij(
    arr: np.ndarray,
    *,
    metric: str,
) -> Optional[Tuple[int, int]]:
    try:
        a = np.asarray(arr, dtype=float)
        if not np.isfinite(a).any():
            return None

        if metric.lower() in _LOWER_IS_BETTER:
            idx = int(np.nanargmin(a))
        else:
            idx = int(np.nanargmax(a))

        i, j = np.unravel_index(idx, a.shape)
        return (int(i), int(j))
    except Exception:
        return None


def _bars_by_lambda(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    city: str,
    metric: str,
    color: str,
    annotate: bool,
    show_legend: bool,
    show_labels: bool,
    show_ticks: bool,
    show_title: bool,
) -> None:
    sub = df[_city_mask(df, city)].copy()
    if sub.empty or metric not in sub.columns:
        ax.set_axis_off()
        return

    keep = sub[sub["pde_mode"].isin(["none", "both"])].copy()
    if keep.empty:
        ax.set_axis_off()
        return

    grp = (
        keep.groupby(
            ["pde_mode", "lambda_prior"],
            dropna=False,
        )[metric]
        .mean()
        .reset_index()
    )
    if grp.empty:
        ax.set_axis_off()
        return

    xs = sorted(grp["lambda_prior"].dropna().unique())
    if not xs:
        ax.set_axis_off()
        return

    modes = _present_modes(grp["pde_mode"])
    if not modes:
        ax.set_axis_off()
        return

    base = np.arange(len(xs), dtype=float)
    width = 0.8 / max(len(modes), 1)
    offset0 = (len(modes) - 1) / 2.0

    for i, m in enumerate(modes):
        y = [
            grp.loc[
                (grp["pde_mode"] == m)
                & (grp["lambda_prior"] == x),
                metric,
            ].mean()
            for x in xs
        ]

        xloc = base + (i - offset0) * width
        ax.bar(
            xloc,
            y,
            width=width,
            label=m,
            color=color,
            alpha=0.35 if m == "none" else 0.95,
            edgecolor="white",
        )

        if annotate:
            for xi, yi in zip(xloc, y):
                if pd.notna(yi):
                    ax.text(
                        xi,
                        yi,
                        f"{yi:.3g}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    ax.set_xticks(base)

    if show_ticks:
        ax.set_xticklabels(
            [f"{x:.2g}" for x in xs],
        )
    else:
        ax.set_xticklabels([])

    if show_labels:
        ax.set_xlabel(r"$\lambda_{\mathrm{prior}}$")
        ax.set_ylabel(_metric_label(metric))
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if show_title:
        ax.set_title(
            f"{city} — {_metric_label(metric)}",
            loc="left",
            pad=6,
            fontweight="bold",
        )

    if show_legend and len(modes) > 1:
        ax.legend(title="pde_mode", frameon=False)

    _axes_cleanup(ax)


def _heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    city: str,
    metric: str,
    cmap: str,
    show_labels: bool,
    show_ticks: bool,
    show_title: bool,
) -> Optional[plt.AxesImage]:
    sub = df[
        _city_mask(df, city)
        & df.get("pde_mode", "both").eq("both")
    ].copy()

    if sub.empty or metric not in sub.columns:
        ax.set_axis_off()
        return None

    piv = sub.pivot_table(
        index="lambda_cons",
        columns="lambda_prior",
        values=metric,
        aggfunc="mean",
    )
    if piv.empty:
        ax.set_axis_off()
        return None

    piv = piv.sort_index().sort_index(axis=1)

    im = ax.imshow(
        piv.values,
        aspect="auto",
        cmap=cmap,
    )

    ax.set_xticks(range(piv.shape[1]))
    ax.set_yticks(range(piv.shape[0]))

    if show_ticks:
        ax.set_xticklabels(
            [f"{c:.2g}" for c in piv.columns],
            rotation=0,
        )
        ax.set_yticklabels(
            [f"{r:.2g}" for r in piv.index],
        )
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if show_labels:
        ax.set_xlabel(r"$\lambda_{\mathrm{prior}}$")
        ax.set_ylabel(r"$\lambda_{\mathrm{cons}}$")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if show_title:
        ax.set_title(
            f"{city} — {_metric_label(metric)} (physics on)",
            loc="left",
            pad=6,
            fontweight="bold",
        )

    ij = _best_ij(piv.values, metric=metric)
    if ij is not None:
        i, j = ij
        ax.scatter(
            [j],
            [i],
            marker="o",
            s=50,
            facecolors="none",
            edgecolors="white",
            linewidths=1.5,
        )

    _axes_cleanup(ax)
    return im


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
def figS6_ablations_main(
    argv: List[str] | None = None,
) -> None:
    args = _parse_args(argv)

    utils.set_paper_style(
        fontsize=int(args.font),
        dpi=int(args.dpi),
    )

    show_legend = utils.str_to_bool(
        args.show_legend,
        default=True,
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

    annotate = not utils.str_to_bool(
        args.mute_values,
        default=False,
    )
    show_cbar = not utils.str_to_bool(
        args.no_colorbar,
        default=False,
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

    tidy_csv = out.parent / "tableS6_ablations_tidy.csv"
    df.to_csv(tidy_csv, index=False)
    print(f"[OK] table -> {tidy_csv}")

    fig = plt.figure(figsize=(8.6, 7.0))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        left=0.07,
        right=0.90 if (show_legend and show_cbar) else 0.98,
        top=0.94,
        bottom=0.10,
        hspace=0.32,
        wspace=0.30,
    )

    # Top row: bars (skill metric vs lambda_prior)
    ax11 = fig.add_subplot(gs[0, 0])
    _bars_by_lambda(
        ax11,
        df,
        city=city_a,
        metric=args.bar_metric,
        color=cfg.CITY_COLORS.get(city_a, "#1F78B4"),
        annotate=annotate,
        show_legend=show_legend,
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )

    ax12 = fig.add_subplot(gs[0, 1])
    _bars_by_lambda(
        ax12,
        df,
        city=city_b,
        metric=args.bar_metric,
        color=cfg.CITY_COLORS.get(city_b, "#E31A1C"),
        annotate=annotate,
        show_legend=show_legend,
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )

    # Bottom row: heatmaps (metric vs lambda_cons/lambda_prior)
    ax21 = fig.add_subplot(gs[1, 0])
    im_a = _heatmap(
        ax21,
        df,
        city=city_a,
        metric=args.heatmap_metric,
        cmap="viridis",
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )

    ax22 = fig.add_subplot(gs[1, 1])
    im_b = _heatmap(
        ax22,
        df,
        city=city_b,
        metric=args.heatmap_metric,
        cmap="viridis",
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )

    # Shared colourbar (optional)
    if show_legend and show_cbar:
        cax = fig.add_axes([0.92, 0.12, 0.015, 0.30])
        im_for = im_a or im_b
        if im_for is not None:
            fig.colorbar(
                im_for,
                cax=cax,
                orientation="vertical",
                label=_metric_label(args.heatmap_metric),
            )
        else:
            cax.set_axis_off()

    if show_title:
        default = (
            "Supplement S6 • Extended ablations & sensitivities"
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
    figS6_ablations_main(argv)


if __name__ == "__main__":
    main()
