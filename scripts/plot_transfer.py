# scripts/plot_transfer.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Supplementary Figure Sx — Cross-city transferability of GeoPriorSubsNet
---------------------------------------------------------------------

Uses xfer_results.csv produced by nat.com/xfer_matrix.py to show:

(a) Cross-city performance vs calibration mode:
    - Overall MAE (mm)
    - Overall R²

(b,c) Coverage–sharpness tradeoff for the 80% interval, split
      by direction:
    - x-axis: sharpness80 (mm)
    - y-axis: coverage80 (fraction)

Panels summarise:
- Nansha → Zhongshan (A_to_B)
- Zhongshan → Nansha (B_to_A)

Run (module form required)
-------------------------
python -m scripts.plot_transfer \
  --src results/xfer/nansha__zhongshan \
  --split val

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
from matplotlib.patches import Patch

from . import config as cfg
from . import utils as u


_DIR_CANON = {
    "a_to_b": "A_to_B",
    "b_to_a": "B_to_A",
    "a_to_a": "A_to_A",
    "b_to_b": "B_to_B",
}


_STRAT_MS = {
    "baseline": 38,
    "xfer": 50,
    "warm": 62,
}


@dataclass
class TextFlags:
    show_legend: bool
    show_labels: bool
    show_ticklabels: bool
    show_title: bool
    show_panel_titles: bool
    title: Optional[str]


def _canon_dir(x: Any) -> str:
    s = str(x).strip()
    k = s#.lower()
    return _DIR_CANON.get(k, s)


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
        "overall_r2": ("overall_r2",),
        "coverage80": ("coverage80", "coverage8"),
        "sharpness80": ("sharpness80", "sharpness"),
    }
    u.ensure_columns(df, aliases=aliases)

    for c in (
        "strategy",
        "rescale_mode",
        "split",
        "calibration",
        "source_city",
        "target_city",
    ):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    if "direction" in df.columns:
        df["direction"] = df["direction"].map(_canon_dir)

    for c in (
        "overall_mae",
        "overall_r2",
        "coverage80",
        "sharpness80",
    ):
        _to_num(df, c)

    return df


def _find_xfer_csv(src: Any) -> Path:
    pats = None
    if hasattr(cfg, "PATTERNS"):
        pats = (cfg.PATTERNS.get("xfer_results_csv") or None)
    pats = pats or ("xfer_results.csv", "*xfer_results*.csv")

    p = u.find_latest(src, pats, must_exist=True)
    if p is None:
        raise FileNotFoundError(str(src))
    return Path(p)


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
    d = _canon_dir(direction)
    s = str(strategy).lower()
    sp = str(split).lower()
    cm = str(calib).lower()

    use_dir = d
    use_rm = rescale_mode

    if s == "baseline":
        use_dir = cfg._BASELINE_MAP.get(d, d)
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

    if better == "max":
        return float(a.max())

    if better == "cov":
        j = int(np.argmin(np.abs(a - float(cov_target))))
        return float(a[j])

    return float(a.min())


def _pick(
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


def _dir_target_city(df: pd.DataFrame, direction: str) -> str:
    d = _canon_dir(direction)
    sub = df[df["direction"].eq(d)]
    if sub.empty:
        return ""
    return str(sub.iloc[0].get("target_city", "") or "")


def _dir_label(df: pd.DataFrame, direction: str) -> str:
    d = _canon_dir(direction)
    sub = df[df["direction"].eq(d)]
    if sub.empty:
        return d

    r0 = sub.iloc[0]
    sc = u.canonical_city(str(r0.get("source_city", "")))
    tc = u.canonical_city(str(r0.get("target_city", "")))

    return f"{sc} \u2192 {tc}"


def _plot_bars(
    ax: Any,
    df: pd.DataFrame,
    *,
    metric_key: str,
    split: str,
    directions: List[str],
    strategies: List[str],
    calib_modes: List[str],
    rescale_mode: Optional[str],
    baseline_rescale: str,
    reduce: str,
    cov_target: float,
    text: TextFlags,
) -> None:
    col, ylab, better = cfg._METRIC_DEF[str(metric_key)]

    n_cal = len(calib_modes)
    n_dir = len(directions)
    n_str = len(strategies)

    x0 = np.arange(n_cal, dtype=float)

    group_w = 0.84
    dir_w = group_w / max(1, n_dir)
    bw = dir_w / max(1, n_str)

    for di, d in enumerate(directions):
        tgt = _dir_target_city(df, d)
        tgt = u.canonical_city(tgt).lower()
        face = cfg.CITY_COLORS.get(tgt, "#777777")

        for si, s in enumerate(strategies):
            for i, cm in enumerate(calib_modes):
                v = _pick(
                    df,
                    direction=d,
                    strategy=s,
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

                off_d = -0.5 * group_w + (di + 0.5) * dir_w
                off_s = -0.5 * dir_w + (si + 0.5) * bw
                xx = x0[i] + off_d + off_s

                ax.bar(
                    xx,
                    v,
                    width=bw,
                    color=face,
                    alpha=0.86,
                    hatch=cfg._STRAT_HATCH.get(s, ""),
                    edgecolor=cfg._STRAT_EDGE.get(s, "#111111"),
                    linewidth=0.6,
                )

    ax.set_xticks(x0)
    ax.set_xticklabels(
        [cfg._CAL_LABEL.get(c, c) for c in calib_modes],
        rotation=15,
    )

    if text.show_labels:
        ax.set_ylabel(ylab)
    else:
        ax.set_ylabel("")

    if not text.show_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if metric_key == "r2":
        ax.axhline(0.0, linestyle="--", linewidth=0.7)

    ax.grid(False)


def _plot_cov_sharp(
    ax: Any,
    df: pd.DataFrame,
    *,
    direction: str,
    split: str,
    strategies: List[str],
    calib_modes: List[str],
    rescale_mode: Optional[str],
    baseline_rescale: str,
    reduce: str,
    cov_target: float,
    text: TextFlags,
) -> Tuple[List[float], List[float]]:
    tgt = _dir_target_city(df, direction)
    tgt = u.canonical_city(tgt).lower()
    face = cfg.CITY_COLORS.get(tgt, "#777777")

    all_cov: List[float] = []
    all_shp: List[float] = []

    for s in strategies:
        for cm in calib_modes:
            sub = _subset(
                df,
                direction=direction,
                strategy=s,
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
                how=reduce,
                better="min",
                cov_target=cov_target,
            )
            y = _reduce_vals(
                cov,
                how=reduce,
                better="cov",
                cov_target=cov_target,
            )

            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            ax.scatter(
                x,
                y,
                s=float(_STRAT_MS.get(s, 50)),
                marker=cfg._CAL_MARKER.get(cm, "o"),
                facecolors=face,
                edgecolors=cfg._STRAT_EDGE.get(s, "#111111"),
                linewidths=0.8,
                alpha=0.92,
            )

            all_cov.append(float(y))
            all_shp.append(float(x))

    ax.axhline(float(cov_target), linestyle="--", linewidth=0.7)

    if text.show_labels:
        ax.set_xlabel("Sharpness80 (mm)")
        ax.set_ylabel("Coverage80")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if not text.show_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.set_ylim(0.0, 1.0)
    ax.grid(False)

    return all_cov, all_shp


def _add_legends(
    ax: Any,
    df: pd.DataFrame,
    *,
    directions: List[str],
    strategies: List[str],
    calib_modes: List[str],
) -> None:
    dir_handles: List[Any] = []
    for d in directions:
        lab = _dir_label(df, d)
        tgt = _dir_target_city(df, d)
        tgt = u.canonical_city(tgt).lower()
        face = cfg.CITY_COLORS.get(tgt, "#777777")

        dir_handles.append(
            Patch(
                facecolor=face,
                edgecolor="#111111",
                linewidth=0.6,
                label=lab,
            )
        )

    strat_handles: List[Any] = []
    for s in strategies:
        strat_handles.append(
            Patch(
                facecolor="white",
                edgecolor=cfg._STRAT_EDGE.get(s, "#111111"),
                hatch=cfg._STRAT_HATCH.get(s, ""),
                linewidth=0.8,
                label=cfg._STRAT_LABEL.get(s, s),
            )
        )

    cal_handles: List[Any] = []
    for c in calib_modes:
        cal_handles.append(
            Line2D(
                [],
                [],
                marker=cfg._CAL_MARKER.get(c, "o"),
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="#111111",
                markersize=5,
                label=cfg._CAL_LABEL.get(c, c),
            )
        )

    leg1 = ax.legend(
        handles=dir_handles,
        frameon=False,
        loc="upper left",
        title="Direction",
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=strat_handles,
        frameon=False,
        loc="upper right",
        title="Strategy",
    )
    ax.add_artist(leg2)

    ax.legend(
        handles=cal_handles,
        frameon=False,
        loc="lower right",
        title="Calibration",
    )


def render(
    df: pd.DataFrame,
    *,
    split: str,
    strategies: List[str],
    calib_modes: List[str],
    rescale_mode: Optional[str],
    baseline_rescale: str,
    reduce: str,
    cov_target: float,
    out: Path,
    text: TextFlags,
) -> Tuple[Path, Path]:
    u.set_paper_style()

    directions = ["A_to_B", "B_to_A"]

    fig = plt.figure(figsize=(7.0, 4.0))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=(1.18, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.40,
        hspace=0.55,
    )

    ax_mae = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[1, 0])
    ax_ab = fig.add_subplot(gs[0, 1])
    ax_ba = fig.add_subplot(gs[1, 1])

    _plot_bars(
        ax_mae,
        df,
        metric_key="mae",
        split=split,
        directions=directions,
        strategies=strategies,
        calib_modes=calib_modes,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
        reduce=reduce,
        cov_target=cov_target,
        text=text,
    )

    _plot_bars(
        ax_r2,
        df,
        metric_key="r2",
        split=split,
        directions=directions,
        strategies=strategies,
        calib_modes=calib_modes,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
        reduce=reduce,
        cov_target=cov_target,
        text=text,
    )

    cov_ab, shp_ab = _plot_cov_sharp(
        ax_ab,
        df,
        direction="A_to_B",
        split=split,
        strategies=strategies,
        calib_modes=calib_modes,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
        reduce=reduce,
        cov_target=cov_target,
        text=text,
    )

    cov_ba, shp_ba = _plot_cov_sharp(
        ax_ba,
        df,
        direction="B_to_A",
        split=split,
        strategies=strategies,
        calib_modes=calib_modes,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
        reduce=reduce,
        cov_target=cov_target,
        text=text,
    )

    if text.show_panel_titles:
        ax_mae.set_title("(a) MAE vs calibration")
        ax_r2.set_title("(a) $R^2$ vs calibration")
        ax_ab.set_title(
            "(b) Coverage–sharpness: " + _dir_label(df, "A_to_B")
        )
        ax_ba.set_title(
            "(c) Coverage–sharpness: " + _dir_label(df, "B_to_A")
        )

    all_cov = np.asarray(cov_ab + cov_ba, dtype=float)
    all_shp = np.asarray(shp_ab + shp_ba, dtype=float)

    m = np.isfinite(all_cov) & np.isfinite(all_shp)
    if m.any():
        ymin = max(0.0, float(all_cov[m].min()) - 0.05)
        ymax = min(1.0, float(all_cov[m].max()) + 0.05)

        xmin = float(all_shp[m].min())
        xmax = float(all_shp[m].max())

        pad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
        xmin -= pad
        xmax += pad

        ax_ab.set_ylim(ymin, ymax)
        ax_ba.set_ylim(ymin, ymax)
        ax_ab.set_xlim(xmin, xmax)
        ax_ba.set_xlim(xmin, xmax)

    if text.show_legend:
        _add_legends(
            ax_ba,
            df,
            directions=directions,
            strategies=strategies,
            calib_modes=calib_modes,
        )

    if text.show_title:
        t0 = (
            "Supplementary Figure Sx — Cross-city transferability"
        )
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


def parse_args(argv: List[str] | None = None) -> Any:
    ap = u.argparse.ArgumentParser(  # type: ignore[attr-defined]
        prog="plot-transfer",
        description=(
            "Supplementary Figure Sx — transferability from xfer CSV"
        ),
    )

    ap.add_argument(
        "--src",
        type=str,
        default="results/xfer",
        help="Folder to search latest xfer_results.csv",
    )
    ap.add_argument(
        "--xfer-csv",
        type=str,
        default=None,
        help="Explicit xfer_results.csv path (file or dir)",
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
        help="Strategies (baseline xfer warm)",
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
        help="Rescale mode for xfer/warm",
    )
    ap.add_argument(
        "--baseline-rescale",
        type=str,
        default="as_is",
        choices=("strict", "as_is"),
        help="Rescale mode for baseline fetch",
    )
    ap.add_argument(
        "--reduce",
        type=str,
        default="best",
        choices=("best", "mean", "median"),
        help="Reduce duplicates per cell",
    )
    ap.add_argument(
        "--cov-target",
        type=float,
        default=0.80,
        help="Target coverage reference",
    )

    u.add_plot_text_args(
        ap,
        default_out="figureS_xfer_transferability",
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
        p = Path(args.xfer_csv).expanduser()
        if p.is_dir():
            csv_p = _find_xfer_csv(p)
        else:
            csv_p = p
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

    out = u.resolve_fig_out(args.out)

    png, svg = render(
        df,
        split=split,
        strategies=strategies,
        calib_modes=calib,
        rescale_mode=str(args.rescale_mode).lower(),
        baseline_rescale=str(args.baseline_rescale).lower(),
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
