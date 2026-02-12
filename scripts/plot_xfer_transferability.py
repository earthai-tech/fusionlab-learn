# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# Program: plot-xfer-transferability
#
# scripts/plot_xfer_transferability.py
#
"""
Supplementary Figure — Cross-city transferability (v3.2)

Reads xfer_results.csv produced by nat.com/xfer_matrix(v3.2).py.

Default figure:
- Column 1: MAE bars vs calibration (per direction row)
- Column 2: MSE bars vs calibration (per direction row)
- Column 3: Coverage vs sharpness scatter (per direction row)

R² is optional:
- Use --metric-bottom r2

Run examples
-----------
# Auto-detect latest xfer_results.csv under src
python scripts/plot_xfer_transferability.py \
  --src results/xfer/nansha__zhongshan \
  --split val \
  --rescale-mode strict

# Explicit CSV
python scripts/plot_xfer_transferability.py \
  --xfer-csv results/xfer/nansha__zhongshan/ \
  20260126-164613/xfer_results.csv \
  --strategies baseline xfer warm \
  --calib-modes none source target \
  --metric-bottom r2

Notes
-----
- baseline rows are A_to_A / B_to_B.
  For cross-city rows:
    A_to_B uses baseline from B_to_B (target-only ref)
    B_to_A uses baseline from A_to_A (target-only ref)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from . import config as cfg
from . import utils as u

# ---------------------------------------------------------------------
# Plot encodings
# ---------------------------------------------------------------------

@dataclass
class TextFlags:
    show_legend: bool
    show_labels: bool
    show_ticklabels: bool
    show_title: bool
    show_panel_titles: bool
    title: Optional[str]


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def _find_xfer_csv(src: Any) -> Path:
    pats = (
        "xfer_results.csv",
        "*xfer_results*.csv",
    )
    p = u.find_latest(src, pats, must_exist=True)
    if p is None:
        raise FileNotFoundError(str(src))
    return p


def _to_num(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "strategy": ("strategy",),
        "rescale_mode": ("rescale_mode", "rescale_m"),
        "direction": ("direction",),
        "source_city": ("source_city", "source_cit"),
        "target_city": ("target_city", "target_cit"),
        "split": ("split",),
        "calibration": ("calibration", "calibratio"),
        "overall_mae": ("overall_mae",),
        "overall_mse": ("overall_mse",),
        "overall_r2": ("overall_r2",),
        "coverage80": ("coverage80", "coverage8"),
        "sharpness80": ("sharpness80", "sharpness"),
    }
    u.ensure_columns(df, aliases=aliases)

    for c in (
        "strategy",
        "rescale_mode",
        "direction",
        "split",
        "calibration",
        "source_city",
        "target_city",
    ):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    for c in ("strategy", "rescale_mode", "direction", "split"):
        if c in df.columns:
            df[c] = df[c].str.lower()

    for c in ("calibration",):
        if c in df.columns:
            df[c] = df[c].str.lower()

    for c in ("source_city", "target_city"):
        if c in df.columns:
            df[c] = df[c].str.lower()

    for c in (
        "overall_mae",
        "overall_mse",
        "overall_r2",
        "coverage80",
        "sharpness80",
    ):
        _to_num(df, c)

    return df


def _subset(
    df: pd.DataFrame,
    *,
    direction: str,
    strategy: str,
    split: str,
    calib: str,
    rescale_mode: Optional[str],
    baseline_rescale: str,
) -> pd.DataFrame:
    d = str(direction).lower()
    s = str(strategy).lower()
    sp = str(split).lower()
    cm = str(calib).lower()

    use_dir = d
    use_rm = rescale_mode

    if s == "baseline":
        use_dir = cfg._BASELINE_MAP.get(d, d).lower()
        use_rm = baseline_rescale

    m = df["direction"].eq(use_dir)
    m &= df["strategy"].eq(s)
    m &= df["split"].eq(sp)
    m &= df["calibration"].eq(cm)

    if use_rm is not None:
        m &= df["rescale_mode"].eq(str(use_rm).lower())

    return df.loc[m].copy()


def _reduce_vals(
    vals: Iterable[float],
    *,
    how: str,
    better: str,
    cov_target: float,
) -> float:
    a = np.asarray(list(vals), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")

    how = str(how).lower()

    if how == "mean":
        return float(a.mean())
    if how == "median":
        return float(np.median(a))

    # best
    if better == "max":
        return float(a.max())

    # special: coverage -> closest to target
    if better == "cov":
        j = int(np.argmin(np.abs(a - float(cov_target))))
        return float(a[j])

    return float(a.min())


def _pick_metric(
    df: pd.DataFrame,
    *,
    direction: str,
    strategy: str,
    split: str,
    calib: str,
    rescale_mode: Optional[str],
    baseline_rescale: str,
    col: str,
    reduce: str,
    better: str,
    cov_target: float,
) -> float:
    sub = _subset(
        df,
        direction=direction,
        strategy=strategy,
        split=split,
        calib=calib,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
    )
    if col not in sub.columns:
        return float("nan")

    return _reduce_vals(
        sub[col].values,
        how=reduce,
        better=better,
        cov_target=cov_target,
    )


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def _dir_label(df: pd.DataFrame, direction: str) -> str:
    d = str(direction).lower()
    sub = df[df["direction"].eq(d)]
    if sub.empty:
        return direction

    r0 = sub.iloc[0]
    sc = u.canonical_city(str(r0.get("source_city", "")))
    tc = u.canonical_city(str(r0.get("target_city", "")))

    return f"{sc} \u2192 {tc}"


def _dir_target_city(df: pd.DataFrame, direction: str) -> str:
    d = str(direction).lower()
    sub = df[df["direction"].eq(d)]
    if sub.empty:
        return ""
    return str(sub.iloc[0].get("target_city", "") or "")


def _bar_panel(
    ax: Any,
    df: pd.DataFrame,
    *,
    direction: str,
    split: str,
    strategies: List[str],
    calib_modes: List[str],
    rescale_mode: Optional[str],
    baseline_rescale: str,
    metric_key: str,
    reduce: str,
    cov_target: float,
    text: TextFlags,
) -> None:
    met = cfg._METRIC_DEF[str(metric_key)]
    col, ylab, better = met

    tgt = _dir_target_city(df, direction)
    tgt = u.canonical_city(tgt)
    base_c = cfg.CITY_COLORS.get(tgt, "#777777")

    n_cal = len(calib_modes)
    n_str = len(strategies)
    x0 = np.arange(n_cal, dtype=float)

    group_w = 0.78
    bw = group_w / max(1, n_str)

    for j, strat in enumerate(strategies):
        for i, cm in enumerate(calib_modes):
            v = _pick_metric(
                df,
                direction=direction,
                strategy=strat,
                split=split,
                calib=cm,
                rescale_mode=rescale_mode,
                baseline_rescale=baseline_rescale,
                col=col,
                reduce=reduce,
                better=better,
                cov_target=cov_target,
            )
            if not np.isfinite(v):
                continue

            off = -0.5 * group_w + (j + 0.5) * bw
            xx = x0[i] + off

            ax.bar(
                xx,
                v,
                width=bw,
                color=base_c,
                alpha=0.85,
                hatch=cfg._STRAT_HATCH.get(strat, ""),
                edgecolor=cfg._STRAT_EDGE.get(strat, "#111111"),
                linewidth=0.6,
            )

    ax.set_xticks(x0)
    ax.set_xticklabels([cfg._CAL_LABEL.get(c, c) for c in calib_modes])

    if text.show_labels:
        ax.set_ylabel(ylab)
    else:
        ax.set_ylabel("")

    if not text.show_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if metric_key == "r2":
        ax.axhline(0.0, linestyle="--", linewidth=0.6)

    ax.grid(False)


def _scatter_panel(
    ax: Any,
    df: pd.DataFrame,
    *,
    direction: str,
    split: str,
    strategies: List[str],
    calib_modes: List[str],
    rescale_mode: Optional[str],
    baseline_rescale: str,
    cov_target: float,
    text: TextFlags,
) -> None:
    tgt = _dir_target_city(df, direction)
    tgt = u.canonical_city(tgt)
    face_c = cfg.CITY_COLORS.get(tgt, "#777777")

    for strat in strategies:
        for cm in calib_modes:
            sub = _subset(
                df,
                direction=direction,
                strategy=strat,
                split=split,
                calib=cm,
                rescale_mode=rescale_mode,
                baseline_rescale=baseline_rescale,
            )
            if sub.empty:
                continue

            cov = sub["coverage80"].values
            shp = sub["sharpness80"].values

            x = _reduce_vals(
                shp,
                how="best",
                better="min",
                cov_target=cov_target,
            )
            y = _reduce_vals(
                cov,
                how="best",
                better="cov",
                cov_target=cov_target,
            )

            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            ax.scatter(
                x,
                y,
                s=42,
                marker=cfg._CAL_MARKER.get(cm, "o"),
                facecolors=face_c,
                edgecolors=cfg._STRAT_EDGE.get(strat, "#111111"),
                linewidths=0.8,
                alpha=0.90,
            )

    ax.axhline(
        float(cov_target),
        linestyle="--",
        linewidth=0.6,
    )
    ax.set_ylim(0.0, 1.0)

    if text.show_labels:
        ax.set_xlabel("Sharpness80 (mm)")
        ax.set_ylabel("Coverage80")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if not text.show_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.grid(False)


def _add_legends(
    ax: Any,
    *,
    strategies: List[str],
    calib_modes: List[str],
) -> None:
    cal_h = []
    for c in calib_modes:
        cal_h.append(
            Line2D(
                [],
                [],
                marker=cfg._CAL_MARKER.get(c, "o"),
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor="#111111",
                markersize=5,
                label=cfg._CAL_LABEL.get(c, c),
            )
        )

    strat_h = []
    for s in strategies:
        strat_h.append(
            Line2D(
                [],
                [],
                marker="s",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=cfg._STRAT_EDGE.get(s, "#111111"),
                markersize=6,
                label=cfg._STRAT_LABEL.get(s, s),
            )
        )

    leg1 = ax.legend(
        handles=cal_h,
        frameon=False,
        loc="lower right",
        title="Calibration",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=strat_h,
        frameon=False,
        loc="lower left",
        title="Strategy",
    )


# ---------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------
def render(
    df: pd.DataFrame,
    *,
    split: str,
    strategies: List[str],
    calib_modes: List[str],
    rescale_mode: Optional[str],
    baseline_rescale: str,
    metric_top: str,
    metric_bottom: str,
    reduce: str,
    cov_target: float,
    out: Path,
    text: TextFlags,
) -> Tuple[Path, Path]:
    u.set_paper_style()

    dirs = ("A_to_B", "B_to_A")

    fig = plt.figure(figsize=(8.1, 4.3))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=(1.0, 1.0, 1.15),
        height_ratios=(1.0, 1.0),
        wspace=0.45,
        hspace=0.55,
    )

    ax = {}
    for r, d in enumerate(dirs):
        ax[(r, 0)] = fig.add_subplot(gs[r, 0])
        ax[(r, 1)] = fig.add_subplot(gs[r, 1])
        ax[(r, 2)] = fig.add_subplot(gs[r, 2])

        _bar_panel(
            ax[(r, 0)],
            df,
            direction=d,
            split=split,
            strategies=strategies,
            calib_modes=calib_modes,
            rescale_mode=rescale_mode,
            baseline_rescale=baseline_rescale,
            metric_key=metric_top,
            reduce=reduce,
            cov_target=cov_target,
            text=text,
        )
        _bar_panel(
            ax[(r, 1)],
            df,
            direction=d,
            split=split,
            strategies=strategies,
            calib_modes=calib_modes,
            rescale_mode=rescale_mode,
            baseline_rescale=baseline_rescale,
            metric_key=metric_bottom,
            reduce=reduce,
            cov_target=cov_target,
            text=text,
        )
        _scatter_panel(
            ax[(r, 2)],
            df,
            direction=d,
            split=split,
            strategies=strategies,
            calib_modes=calib_modes,
            rescale_mode=rescale_mode,
            baseline_rescale=baseline_rescale,
            cov_target=cov_target,
            text=text,
        )

        if text.show_panel_titles:
            row_t = _dir_label(df, d)
            ax[(r, 0)].set_title(row_t)
            ax[(r, 1)].set_title(row_t)
            ax[(r, 2)].set_title(row_t)

    if text.show_legend:
        _add_legends(
            ax[(1, 2)],
            strategies=strategies,
            calib_modes=calib_modes,
        )

    if text.show_title:
        t0 = "Cross-city transferability"
        if rescale_mode is not None:
            t0 = f"{t0} (rescale={rescale_mode})"
        t = u.resolve_title(default=t0, title=text.title)
        fig.suptitle(t, x=0.02, y=0.99, ha="left")

    stem = out
    if stem.suffix:
        stem = stem.with_suffix("")
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")

    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    return png, svg


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> Any:
    ap = u.argparse.ArgumentParser(  # type: ignore[attr-defined]
        prog="plot-xfer-transferability",
        description=(
            "Plot cross-city transferability from xfer_results.csv"
        ),
    )

    ap.add_argument(
        "--src",
        type=str,
        default="results/xfer",
        help="Folder to search for latest xfer_results.csv",
    )
    ap.add_argument(
        "--xfer-csv",
        type=str,
        default=None,
        help="Explicit xfer_results.csv path",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("val", "test"),
        help="Which split to plot",
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=list(cfg._STRAT_DEFAULT),
        help="Strategies to include (baseline xfer warm)",
    )
    ap.add_argument(
        "--calib-modes",
        nargs="+",
        default=list(cfg._CAL_ORDER),
        help="Calibration modes (none source target)",
    )
    ap.add_argument(
        "--rescale-mode",
        type=str,
        default="strict",
        choices=("strict", "as_is"),
        help="Rescale mode for xfer/warm rows",
    )
    ap.add_argument(
        "--baseline-rescale",
        type=str,
        default="as_is",
        choices=("strict", "as_is"),
        help="Rescale mode used to fetch baseline rows",
    )
    ap.add_argument(
        "--metric-top",
        type=str,
        default="mae",
        choices=tuple(cfg._METRIC_DEF.keys()),
        help="Top metric (default mae)",
    )
    ap.add_argument(
        "--metric-bottom",
        type=str,
        default="mse",
        choices=tuple(cfg._METRIC_DEF.keys()),
        help="Bottom metric (default mse, or r2)",
    )
    ap.add_argument(
        "--reduce",
        type=str,
        default="best",
        choices=("best", "mean", "median"),
        help="How to reduce duplicates per cell",
    )
    ap.add_argument(
        "--cov-target",
        type=float,
        default=0.80,
        help="Target coverage reference line",
    )

    u.add_plot_text_args(
        ap,
        default_out="xfer_transferability",
    )

    return ap.parse_args(argv)


def _text_flags(args: Any) -> TextFlags:
    return TextFlags(
        show_legend=u.str_to_bool(args.show_legend, default=True),
        show_labels=u.str_to_bool(args.show_labels, default=True),
        show_ticklabels=u.str_to_bool(
            args.show_ticklabels,
            default=True,
        ),
        show_title=u.str_to_bool(args.show_title, default=True),
        show_panel_titles=u.str_to_bool(
            args.show_panel_titles,
            default=True,
        ),
        title=args.title,
    )


def figSx_xfer_transferability_main(
    argv: List[str] | None = None,
) -> None:
    u.ensure_script_dirs()

    args = parse_args(argv)
    text = _text_flags(args)

    if args.xfer_csv:
        csv_p = Path(args.xfer_csv).expanduser()
        if csv_p.is_dir():
            csv_p = _find_xfer_csv(csv_p)
    else:
        csv_p = _find_xfer_csv(args.src)

    df = pd.read_csv(csv_p)
    df = _canon_cols(df)

    split = str(args.split).lower()
    df = df[df["split"].eq(split)].copy()
    if df.empty:
        raise SystemExit("No rows after split filter.")

    strategies = [str(s).lower() for s in args.strategies]
    calib = [str(c).lower() for c in args.calib_modes]

    rm = str(args.rescale_mode).lower()
    brm = str(args.baseline_rescale).lower()

    out = u.resolve_fig_out(args.out)

    png, svg = render(
        df,
        split=split,
        strategies=strategies,
        calib_modes=calib,
        rescale_mode=rm,
        baseline_rescale=brm,
        metric_top=str(args.metric_top).lower(),
        metric_bottom=str(args.metric_bottom).lower(),
        reduce=str(args.reduce).lower(),
        cov_target=float(args.cov_target),
        out=out,
        text=text,
    )

    print(f"[OK] Wrote {png}")
    print(f"[OK] Wrote {svg}")


def main(argv: List[str] | None = None) -> None:
    figSx_xfer_transferability_main(argv)


if __name__ == "__main__":
    main()
