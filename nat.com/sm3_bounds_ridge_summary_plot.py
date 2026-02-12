
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
sm3_bounds_ridge_summary_plot.py

Summarize "clipped to bounds" vs "ridge non-identifiability"
from SM3 synthetic runs, and export paper-style figures.

This script mirrors the style used in sm3_mainfig_plot_v32.py:
- font size 8
- 600 dpi
- thin axes, no top/right spines
- panel labels (a–d)

Inputs
------
CSV produced by the synthetic identifiability pipeline, e.g.
  sm3_synth_runs.csv

Outputs
-------
- Figure (PNG/PDF)
- JSON summary (counts + fractions)
- CSV category table (optional)

Definitions (explicit)
----------------------
Bounds are inferred from the observed extrema in the runs file:
  K_min/max = min/max(K_est_med_mps)
  tau_min/max = min/max(tau_est_med_sec)
  Hd_min/max = min/max(Hd_est_med)

A run is "clipped" if an estimate is close to a bound.
Two clipping modes are reported:
  1) "primary": K at max OR tau at min OR Hd at max
  2) "any-side": K at min/max OR tau at min/max OR Hd at min/max

"Strong ridge" is ridge_resid_q50 > ridge_thr (default 2.0 decades).

Example
-------
python sm3_bounds_ridge_summary_plot.py \
  --csv sm3_synth_runs.csv \
  --out figs/sm3_clip_vs_ridge.png \
  --ridge_thr 2
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def set_figure_style(*, fontsize: int = 8, dpi: int = 600) -> None:
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


def _require(df: pd.DataFrame, cols, ctx: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing cols ({ctx}): {', '.join(miss)}")


def _isclose(a: np.ndarray, b: float, *, rtol: float) -> np.ndarray:
    # Keep it simple and robust for floats from CSV.
    a = np.asarray(a, float)
    return np.isclose(a, float(b), rtol=float(rtol), atol=0.0)


@dataclass(frozen=True)
class BoundInfo:
    K_min: float
    K_max: float
    tau_min: float
    tau_max: float
    Hd_min: float
    Hd_max: float


def infer_bounds(df: pd.DataFrame) -> BoundInfo:
    K = df["K_est_med_mps"].to_numpy(float)
    tau = df["tau_est_med_sec"].to_numpy(float)
    Hd = df["Hd_est_med"].to_numpy(float)

    return BoundInfo(
        K_min=float(np.nanmin(K)),
        K_max=float(np.nanmax(K)),
        tau_min=float(np.nanmin(tau)),
        tau_max=float(np.nanmax(tau)),
        Hd_min=float(np.nanmin(Hd)),
        Hd_max=float(np.nanmax(Hd)),
    )


def compute_flags(
    df: pd.DataFrame,
    bounds: BoundInfo,
    *,
    rtol: float,
    ridge_thr: float,
) -> Dict[str, np.ndarray]:
    K = df["K_est_med_mps"].to_numpy(float)
    tau = df["tau_est_med_sec"].to_numpy(float)
    Hd = df["Hd_est_med"].to_numpy(float)
    ridge = df["ridge_resid_q50"].to_numpy(float)

    K_hi = _isclose(K, bounds.K_max, rtol=rtol)
    K_lo = _isclose(K, bounds.K_min, rtol=rtol)
    tau_lo = _isclose(tau, bounds.tau_min, rtol=rtol)
    tau_hi = _isclose(tau, bounds.tau_max, rtol=rtol)
    Hd_hi = _isclose(Hd, bounds.Hd_max, rtol=rtol)
    Hd_lo = _isclose(Hd, bounds.Hd_min, rtol=rtol)

    clipped_primary = K_hi | tau_lo | Hd_hi
    clipped_any = K_hi | K_lo | tau_lo | tau_hi | Hd_hi | Hd_lo
    ridge_strong = np.asarray(ridge, float) > float(ridge_thr)

    return {
        "K_clip_hi": K_hi,
        "K_clip_lo": K_lo,
        "tau_clip_lo": tau_lo,
        "tau_clip_hi": tau_hi,
        "Hd_clip_hi": Hd_hi,
        "Hd_clip_lo": Hd_lo,
        "clipped_primary": clipped_primary,
        "clipped_any": clipped_any,
        "ridge_strong": ridge_strong,
        "ridge_resid_q50": np.asarray(ridge, float),
    }


def summarize_counts(
    flags: Dict[str, np.ndarray],
    *,
    use: str,
) -> Dict[str, float]:
    if use not in ("primary", "any"):
        raise ValueError("use must be 'primary' or 'any'")

    clipped = flags["clipped_primary"] if use == "primary" else flags["clipped_any"]
    ridge = flags["ridge_strong"]
    n = int(clipped.size)

    both = int((clipped & ridge).sum())
    clip_only = int((clipped & ~ridge).sum())
    ridge_only = int((~clipped & ridge).sum())
    neither = int((~clipped & ~ridge).sum())

    def frac(k: int) -> float:
        return float(k) / float(n) if n else float("nan")

    out = {
        "n": n,
        "clipped": int(clipped.sum()),
        "ridge_strong": int(ridge.sum()),
        "both_clipped_and_ridge": both,
        "clipped_only": clip_only,
        "ridge_only": ridge_only,
        "neither": neither,
        "clipped_frac": frac(int(clipped.sum())),
        "ridge_strong_frac": frac(int(ridge.sum())),
        "both_frac": frac(both),
        "clipped_only_frac": frac(clip_only),
        "ridge_only_frac": frac(ridge_only),
        "neither_frac": frac(neither),
    }
    return out

def _maybe_panel(ax, lab: str, *, show: bool) -> None:
    if not show:
        return
    _panel(ax, lab)


def _maybe_title(ax, title: str, *, show: bool) -> None:
    if not show:
        return
    ax.set_title(title, pad=2)


def _maybe_xlabel(ax, lab: str, *, show: bool) -> None:
    if not show:
        return
    ax.set_xlabel(lab)


def _maybe_ylabel(ax, lab: str, *, show: bool) -> None:
    if not show:
        return
    ax.set_ylabel(lab)


def _maybe_legend(ax, *, show: bool, **kw) -> None:
    if not show:
        return
    ax.legend(**kw)


def _add_bool_arg(
    ap: argparse.ArgumentParser,
    name: str,
    *,
    default: bool = True,
    help: str,
) -> None:
    # Gives: --show-legend / --no-show-legend
    act = getattr(argparse, "BooleanOptionalAction", None)
    if act is None:
        # Fallback (older Python): add explicit --no-*
        dest = name.replace("-", "_")
        ap.add_argument(
            f"--{name}",
            dest=dest,
            action="store_true",
            default=default,
            help=help,
        )
        ap.add_argument(
            f"--no-{name}",
            dest=dest,
            action="store_false",
        )
        return

    ap.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=act,
        default=default,
        help=help,
    )
    
def plot_summary(
    df: pd.DataFrame,
    flags: Dict[str, np.ndarray],
    bounds: BoundInfo,
    *,
    outpath: str,
    ridge_thr: float,
    use: str,
    show_legend: bool,
    show_labels: bool,
    show_titles: bool,
    show_panels: bool,
) -> None:
    set_figure_style(fontsize=8, dpi=600)

    clipped = (
        flags["clipped_primary"]
        if use == "primary"
        else flags["clipped_any"]
    )
    ridge = flags["ridge_strong"]
    ridge_resid = flags["ridge_resid_q50"]
    n = int(len(df))

    both = clipped & ridge
    clip_only = clipped & ~ridge
    ridge_only = ~clipped & ridge
    neither = ~clipped & ~ridge

    counts = np.array(
        [
            int(both.sum()),
            int(clip_only.sum()),
            int(ridge_only.sum()),
            int(neither.sum()),
        ],
        dtype=int,
    )
    cats = [
        "Clipped+Ridge",
        "Clipped only",
        "Ridge only",
        "Neither",
    ]
    if n:
        fracs = counts / float(n)
    else:
        fracs = np.full_like(counts, np.nan, float)

    lith = None
    if "lith_idx" in df.columns:
        lith = df["lith_idx"].to_numpy(int)

    lith_names = {
        0: "Fine",
        1: "Mixed",
        2: "Coarse",
        3: "Rock",
    }
    lith_order = [0, 1, 2, 3] if lith is not None else []

    fig = plt.figure(
        figsize=(7.2, 4.2),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(2, 2)

    # (a)
    axA = fig.add_subplot(gs[0, 0])
    _beautify(axA)
    _maybe_panel(axA, "a", show=show_panels)

    bound_labels = [
        "K@max",
        "K@min",
        "τ@min",
        "τ@max",
        "Hd@max",
        "Hd@min",
    ]
    bound_counts = np.array(
        [
            int(flags["K_clip_hi"].sum()),
            int(flags["K_clip_lo"].sum()),
            int(flags["tau_clip_lo"].sum()),
            int(flags["tau_clip_hi"].sum()),
            int(flags["Hd_clip_hi"].sum()),
            int(flags["Hd_clip_lo"].sum()),
        ],
        dtype=int,
    )
    x = np.arange(bound_counts.size)
    axA.bar(x, bound_counts)
    axA.set_xticks(x)
    axA.set_xticklabels(
        bound_labels,
        rotation=30,
        ha="right",
    )
    _maybe_ylabel(axA, "Count", show=show_labels)
    _maybe_title(
        axA,
        "Bound hits inferred from runs",
        show=show_titles,
    )

    for i, c in enumerate(bound_counts.tolist()):
        if n:
            pct = 100.0 * float(c) / float(n)
        else:
            pct = float("nan")
        axA.text(
            i,
            c + 0.5,
            f"{c}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
        )

    if show_labels:
        axA.text(
            0.02,
            0.02,
            (
                f"K_max={bounds.K_max:.3e} m/s\n"
                f"τ_min={bounds.tau_min:.2f} s\n"
                f"Hd_max={bounds.Hd_max:g}"
            ),
            transform=axA.transAxes,
            va="bottom",
            ha="left",
        )

    # (b)
    axB = fig.add_subplot(gs[0, 1])
    _beautify(axB)
    _maybe_panel(axB, "b", show=show_panels)

    rr = np.asarray(ridge_resid, float)
    rr = rr[np.isfinite(rr)]
    axB.hist(rr, bins=18)
    axB.axvline(
        float(ridge_thr),
        linestyle="--",
        linewidth=0.9,
    )
    _maybe_xlabel(
        axB,
        "ridge_resid_q50 (decades)",
        show=show_labels,
    )
    _maybe_ylabel(axB, "Count", show=show_labels)
    _maybe_title(
        axB,
        "Ridge non-identifiability (distribution)",
        show=show_titles,
    )

    if rr.size:
        frac_ridge = float((rr > float(ridge_thr)).sum())
        frac_ridge = frac_ridge / float(rr.size)
    else:
        frac_ridge = float("nan")

    if show_labels:
        axB.text(
            0.03,
            0.97,
            (
                f"Strong ridge (> {ridge_thr:g}) "
                f"= {100*frac_ridge:.1f}%"
            ),
            transform=axB.transAxes,
            va="top",
            ha="left",
        )

    # (c)
    axC = fig.add_subplot(gs[1, 0])
    _beautify(axC)
    _maybe_panel(axC, "c", show=show_panels)

    mat = np.array(
        [
            [
                int((~clipped & ~ridge).sum()),
                int((~clipped & ridge).sum()),
            ],
            [
                int((clipped & ~ridge).sum()),
                int((clipped & ridge).sum()),
            ],
        ],
        dtype=int,
    )
    axC.imshow(mat, aspect="auto")
    axC.set_xticks([0, 1])
    axC.set_yticks([0, 1])
    axC.set_xticklabels(["No ridge", "Strong ridge"])
    axC.set_yticklabels(["Not clipped", "Clipped"])
    _maybe_title(
        axC,
        f"Clipping vs ridge ({use})",
        show=show_titles,
    )

    for (i, j), v in np.ndenumerate(mat):
        if n:
            p = 100.0 * float(v) / float(n)
        else:
            p = float("nan")
        axC.text(
            j,
            i,
            f"{v}\n({p:.1f}%)",
            ha="center",
            va="center",
        )

    # (d)
    axD = fig.add_subplot(gs[1, 1])
    _beautify(axD)
    _maybe_panel(axD, "d", show=show_panels)

    if lith is None:
        axD.bar(np.arange(4), fracs)
        axD.set_xticks(np.arange(4))
        axD.set_xticklabels(
            cats,
            rotation=25,
            ha="right",
        )
        axD.set_ylim(0, 1)
        _maybe_ylabel(axD, "Fraction", show=show_labels)
        _maybe_title(
            axD,
            "Category fractions (overall)",
            show=show_titles,
        )
    else:
        lith_labels = [
            lith_names.get(i, f"L{i}")
            for i in lith_order
        ]
        x = np.arange(len(lith_order))
        bottoms = np.zeros_like(x, float)

        masks = [both, clip_only, ridge_only, neither]
        for cat, m in zip(cats, masks):
            vals = []
            for li in lith_order:
                mm = (lith == li)
                denom = float(mm.sum())
                if denom:
                    vals.append(float((m & mm).sum()) / denom)
                else:
                    vals.append(0.0)
            vals = np.asarray(vals, float)
            axD.bar(
                x,
                vals,
                bottom=bottoms,
                label=cat,
            )
            bottoms = bottoms + vals

        axD.set_xticks(x)
        axD.set_xticklabels(lith_labels)
        axD.set_ylim(0, 1)
        _maybe_ylabel(
            axD,
            "Fraction within lithology",
            show=show_labels,
        )
        _maybe_title(
            axD,
            "Failure modes by lithology",
            show=show_titles,
        )
        _maybe_legend(
            axD,
            show=show_legend,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )

    outdir = os.path.dirname(outpath) or "."
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument(
        "--use",
        default="any",
        choices=["any", "primary"],
    )
    ap.add_argument("--ridge-thr", type=float, default=2.0)
    ap.add_argument("--rtol", type=float, default=1e-6)

    _add_bool_arg(
        ap,
        "show-legend",
        default=True,
        help="Show legends (default: on).",
    )
    _add_bool_arg(
        ap,
        "show-labels",
        default=True,
        help="Show axis labels (default: on).",
    )
    _add_bool_arg(
        ap,
        "show-titles",
        default=True,
        help="Show panel titles (default: on).",
    )
    _add_bool_arg(
        ap,
        "show-panels",
        default=True,
        help="Show panel letters a–d (default: on).",
    )

    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    need = [
        "K_est_med_mps",
        "tau_est_med_sec",
        "Hd_est_med",
        "ridge_resid_q50",
    ]
    _require(df, need, ctx="summary")

    bounds = infer_bounds(df)
    flags = compute_flags(
        df,
        bounds,
        rtol=float(args.rtol),
        ridge_thr=float(args.ridge_thr),
    )

    summ_primary = summarize_counts(flags, use="primary")
    summ_any = summarize_counts(flags, use="any")

    payload = {
        "csv": os.path.abspath(args.csv),
        "bounds_inferred": {
            "K_min": bounds.K_min,
            "K_max": bounds.K_max,
            "tau_min": bounds.tau_min,
            "tau_max": bounds.tau_max,
            "Hd_min": bounds.Hd_min,
            "Hd_max": bounds.Hd_max,
        },
        "ridge_thr": float(args.ridge_thr),
        "rtol": float(args.rtol),
        "summary_primary": summ_primary,
        "summary_any": summ_any,
    }

    if args.out_json is None:
        base, _ = os.path.splitext(args.out)
        args.out_json = base + ".json"

    outj_dir = os.path.dirname(args.out_json) or "."
    os.makedirs(outj_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_summary(
        df,
        flags,
        bounds,
        outpath=args.out,
        ridge_thr=float(args.ridge_thr),
        use=str(args.use),
        show_legend=bool(args.show_legend),
        show_labels=bool(args.show_labels),
        show_titles=bool(args.show_titles),
        show_panels=bool(args.show_panels),
    )



if __name__ == "__main__":
    main()

# python sm3_bounds_ridge_summary_plot.py \
#   --csv sm3_synth_runs.csv \
#   --out sm3_clip_vs_ridge.png

# python sm3_bounds_ridge_summary_plot.py \
#   --csv sm3_synth_runs.csv \
#   --ou.png \
#   --no-show-legend \
#   --no-show-titles

