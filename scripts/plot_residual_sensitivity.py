# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# nat.com/figs/make_supp_figS7_physics_sensitivity.py
#
"""
Supplement S7 • Physics sensitivity (ε_prior, ε_cons)

Visualises how physics residual metrics vary across physics weights:

- Row 1:  metric_prior (default ε_prior)
          over (λ_cons, λ_prior) for each city.
- Row 2:  metric_cons  (default ε_cons)
          over (λ_cons, λ_prior) for each city.

Data source
-----------
We scan for JSONL under:
  <root>/**/ablation_records/ablation_record*.jsonl

Each line is one JSON record, typically with:
  city, pde_mode, lambda_cons, lambda_prior,
  epsilon_prior, epsilon_cons, ...

Outputs
-------
- Figure: <out>.png and <out>.pdf
- Tidy table copy: tableS7_physics_tidy.csv
  written next to the figure outputs.

API conventions
---------------
- Style via scripts.utils.set_paper_style()
- Output path via scripts.utils.resolve_fig_out()
- main(argv) wrapper calls a *_main(argv) function.

Linting / format
----------------
- black + ruff, line length <= 62
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import  List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts import config as cfg
from scripts import utils


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
def _parse_args(
    argv: List[str] | None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot-residual-sensitivity", 
        description="Supplement S7: Physics sensitivity heatmaps",
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

    # Keep backward compat: allow an explicit out dir.
    # If omitted, we use scripts/figs/ via resolve_fig_out().
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
        help="City name as recorded in JSONL.",
    )
    p.add_argument(
        "--city-b",
        type=str,
        default="Zhongshan",
        help="City name as recorded in JSONL.",
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
    )

    utils.add_plot_text_args(
        p,
        default_out="supp_fig_S7_physics_sensitivity",
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

    # Prefer your central physics labels when available.
    if k in cfg.PHYS_LABELS:
        return cfg.PHYS_LABELS[k]

    # Then provide a few common paper labels.
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
            f"{city} — {_metric_label(metric)}",
            loc="left",
            pad=6,
            fontweight="bold",
        )

    ij = _best_ij(
        piv.values,
        metric=metric,
    )
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
    # If user supplies --out-dir, we respect it literally.
    if out_dir:
        base = Path(out_dir).expanduser()
        return (base / Path(out).expanduser()).resolve()

    # Otherwise: standard project behavior.
    return utils.resolve_fig_out(out)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def figS7_physics_sensitivity_main(
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

    # Save tidy copy next to the figure.
    tidy_csv = out.parent / "tableS7_physics_tidy.csv"
    df.to_csv(tidy_csv, index=False)
    print(f"[OK] table -> {tidy_csv}")

    fig = plt.figure(figsize=(8.6, 7.0))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        left=0.07,
        right=0.90 if show_legend else 0.98,
        top=0.94,
        bottom=0.10,
        hspace=0.32,
        wspace=0.30,
    )

    # Row 1 (metric_prior)
    ax11 = fig.add_subplot(gs[0, 0])
    im_a1 = _heatmap(
        ax11,
        df,
        city=city_a,
        metric=args.metric_prior,
        cmap="magma",
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )

    ax12 = fig.add_subplot(gs[0, 1])
    im_b1 = _heatmap(
        ax12,
        df,
        city=city_b,
        metric=args.metric_prior,
        cmap="magma",
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )

    # Row 2 (metric_cons)
    ax21 = fig.add_subplot(gs[1, 0])
    im_a2 = _heatmap(
        ax21,
        df,
        city=city_a,
        metric=args.metric_cons,
        cmap="magma",
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )

    ax22 = fig.add_subplot(gs[1, 1])
    im_b2 = _heatmap(
        ax22,
        df,
        city=city_b,
        metric=args.metric_cons,
        cmap="magma",
        show_labels=show_labels,
        show_ticks=show_ticks,
        show_title=show_pan_t,
    )

    # Shared colorbars (one per row)
    if show_legend:
        cax1 = fig.add_axes([0.92, 0.56, 0.015, 0.30])
        row1 = im_a1 or im_b1
        if row1 is not None:
            fig.colorbar(
                row1,
                cax=cax1,
                orientation="vertical",
                label=_metric_label(args.metric_prior),
            )
        else:
            cax1.set_axis_off()

        cax2 = fig.add_axes([0.92, 0.12, 0.015, 0.30])
        row2 = im_a2 or im_b2
        if row2 is not None:
            fig.colorbar(
                row2,
                cax=cax2,
                orientation="vertical",
                label=_metric_label(args.metric_cons),
            )
        else:
            cax2.set_axis_off()

    if show_title:
        default = (
            "Supplement S7 • Physics residual sensitivity\n"
            r"($\epsilon_{\mathrm{prior}}$ and "
            r"$\epsilon_{\mathrm{cons}}$ vs. "
            r"$\lambda_{\mathrm{prior}}, "
            r"\lambda_{\mathrm{cons}}$)"
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
    figS7_physics_sensitivity_main(argv)


if __name__ == "__main__":
    main()
