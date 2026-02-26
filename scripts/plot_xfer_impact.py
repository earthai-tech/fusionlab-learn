# scripts/plot_xfer_impact.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Supplementary Figure Sy — Impactful transferability
--------------------------------------------------

This figure extends ``plot_transfer.py`` by adding
decision-maker oriented diagnostics:

Panels
------
(a) Retention vs target baseline (overall):
    - R² retention   = R² / R²_baseline
    - MAE retention  = MAE_baseline / MAE

(b) Horizon retention (H1–H3) for R².

(c) Coverage–sharpness tradeoff (@80) per direction.

(d) Threshold risk skill (optional if eval CSVs exist):
    - Reliability diagram for exceedance
    - Brier score summary

(e) Hotspot stability (optional):
    - Jaccard@K overlap of top-K hotspots
    - Spearman rank correlation on overlap
    - Optional error bars (mean ± std)
    - Optional time-series small multiples

Inputs
------
Requires ``xfer_results.csv`` (from nat.com/xfer_matrix.py).

For panel (d), it also uses ``xfer_results.json``
if available, to locate per-job ``csv_eval`` paths.

Run (module form required)
-------------------------
python -m scripts plot-xfer-impact \
  --src results/xfer/nansha__zhongshan \
  --split val \
  --calib source

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config as cfg
from . import utils as u

from fusionlab.utils.transfer import xfer_risk


_DIR_CANON = {
    "a_to_b": "A_to_B",
    "b_to_a": "B_to_A",
    "a_to_a": "A_to_A",
    "b_to_b": "B_to_B",
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
    k = s.lower()
    return _DIR_CANON.get(k, s)


def _to_num(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def _metrics_unit(df: pd.DataFrame) -> str:
    for c in ("subsidence_unit", "metrics_unit", "metric_unit"):
        if c in df.columns:
            s = df[c].dropna()
            if not s.empty:
                u0 = str(s.iloc[0]).strip().lower()
                if u0.startswith("m"):
                    return "m"
                if u0.startswith("mm"):
                    return "mm"
    return "mm"

def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "strategy": ("strategy",),
        "rescale_mode": ("rescale_mode", "rescale_m"),
        "direction": ("direction",),
        "source_city": ("source_city",),
        "target_city": ("target_city",),
        "split": ("split",),
        "calibration": ("calibration",),
        "overall_mae": ("overall_mae",),
        "overall_r2": ("overall_r2",),
        "coverage80": ("coverage80",),
        "sharpness80": ("sharpness80",),
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
            df[c] = (
                df[c].astype(str)
                .str.strip()
                .str.lower()
            )

    if "direction" in df.columns:
        df["direction"] = df["direction"].map(_canon_dir)

    for c in (
        "overall_mae",
        "overall_r2",
        "coverage80",
        "sharpness80",
    ):
        _to_num(df, c)

    for c in df.columns:
        if str(c).startswith("per_horizon_"):
            _to_num(df, c)

    return df


def _find_xfer_csv(src: Any) -> Path:
    pats = None
    if hasattr(cfg, "PATTERNS"):
        pats = cfg.PATTERNS.get("xfer_results_csv")

    pats = pats or ("xfer_results.csv", "*xfer_results*.csv")
    p = u.find_latest(src, pats, must_exist=True)
    if p is None:
        raise FileNotFoundError(str(src))
    return Path(p)


def _find_xfer_json(csv_p: Path) -> Optional[Path]:
    root = csv_p.parent
    p = u.find_latest(root, ("xfer_results.json",))
    return None if p is None else Path(p)



def _is_pair_dir(p: Path) -> bool:
    """
    True if p looks like a cityA__cityB folder containing
    timestamp subfolders with xfer_results.csv inside.
    """
    if not p.exists() or not p.is_dir():
        return False
    direct = (p / "xfer_results.csv").exists()
    if direct:
        return False
    for d in p.iterdir():
        if d.is_dir() and (d / "xfer_results.csv").exists():
            return True
    return False


def _row_matches(
    df: pd.DataFrame,
    *,
    strategy: str,
    split: str,
    calib: str,
    rescale_mode: Optional[str],
) -> bool:
    if df is None or df.empty:
        return False
    m = df["strategy"].eq(str(strategy).lower())
    m &= df["split"].eq(str(split).lower())
    m &= df["calibration"].eq(str(calib).lower())
    if rescale_mode is not None and "rescale_mode" in df.columns:
        m &= df["rescale_mode"].eq(str(rescale_mode).lower())
    return bool(m.any())


def _pick_latest_csvs_by_strategy(
    pair_dir: Path,
    *,
    strategies: List[str],
    split: str,
    calib: str,
    rescale_mode: str,
    baseline_rescale: str,
) -> Dict[str, Path]:
    """
    From pair_dir (cityA__cityB), pick the latest run folder
    per strategy, based on the content of xfer_results.csv.
    """
    pats = ("xfer_results.csv", "*xfer_results*.csv")
    cands = u.find_all(pair_dir, pats, must_exist=True)

    want = [str(s).lower() for s in strategies]
    picked: Dict[str, Path] = {}

    def _rm(s: str) -> str:
        return baseline_rescale if s == "baseline" else rescale_mode

    # pass 1: require rescale match
    for fp in cands:
        if len(picked) == len(want):
            break
        try:
            d0 = _canon_cols(pd.read_csv(fp))
        except Exception:
            continue
        for s in want:
            if s in picked:
                continue
            if _row_matches(
                d0,
                strategy=s,
                split=split,
                calib=calib,
                rescale_mode=_rm(s),
            ):
                picked[s] = Path(fp)

    # pass 2: fallback (ignore rescale) if still missing
    if len(picked) < len(want):
        for fp in cands:
            if len(picked) == len(want):
                break
            try:
                d0 = _canon_cols(pd.read_csv(fp))
            except Exception:
                continue
            for s in want:
                if s in picked:
                    continue
                if _row_matches(
                    d0,
                    strategy=s,
                    split=split,
                    calib=calib,
                    rescale_mode=None,
                ):
                    picked[s] = Path(fp)

    return picked


def _load_multi_jobs(
    pair_dir: Path,
    *,
    strategies: List[str],
    split: str,
    calib: str,
    rescale_mode: str,
    baseline_rescale: str,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Path]]:
    picked = _pick_latest_csvs_by_strategy(
        pair_dir,
        strategies=strategies,
        split=split,
        calib=calib,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
    )

    dfs: List[pd.DataFrame] = []
    rows: List[Dict[str, Any]] = []
    seen_eval: set = set()

    for s, fp in picked.items():
        try:
            d0 = pd.read_csv(fp)
            dfs.append(d0)
        except Exception:
            pass

        jp = fp.parent / "xfer_results.json"
        if not jp.exists():
            jp2 = _find_xfer_json(fp)
            jp = jp2 if jp2 is not None else jp
        if jp.exists():
            for r in _load_json_rows(jp):
                key = str(r.get("csv_eval", "")) or str(r.get("csv_future", ""))
                if key and key in seen_eval:
                    continue
                if key:
                    seen_eval.add(key)
                rows.append(r)

    if not dfs:
        raise SystemExit(f"No usable xfer_results.csv under {pair_dir}")

    return pd.concat(dfs, axis=0, ignore_index=True), rows, picked


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

    if use_rm is not None and "rescale_mode" in df.columns:
        m &= df["rescale_mode"].eq(str(use_rm).lower())

    out = df.loc[m].copy()
    if out.empty and s == "baseline":
        m2 = df["direction"].eq(d)
        m2 &= df["strategy"].eq(s)
        m2 &= df["split"].eq(sp)
        m2 &= df["calibration"].eq(cm)
        if use_rm is not None and (
            "rescale_mode" in df.columns
        ):
            m2 &= df["rescale_mode"].eq(str(use_rm).lower())
        out = df.loc[m2].copy()

    return out


def _reduce_vals(vals: np.ndarray) -> float:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.median(a))


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
    if sub.empty or col not in sub.columns:
        return float("nan")
    return _reduce_vals(sub[col].values)

def _tgt_color(df: pd.DataFrame, direction: str) -> str:
    d = _canon_dir(direction)
    sub = df[df["direction"].eq(d)]
    if sub.empty:
        return "#777777"
    tc = str(sub.iloc[0].get("target_city", "") or "")
    tc = u.canonical_city(tc)  # keep canonical casing
    return cfg.CITY_COLORS.get(
        tc,
        cfg.CITY_COLORS.get(tc.title(), cfg.CITY_COLORS.get(tc.lower(), "#777777")),
    )

def _dir_label(df: pd.DataFrame, direction: str) -> str:
    d = _canon_dir(direction)
    sub = df[df["direction"].eq(d)]
    if sub.empty:
        return d
    r0 = sub.iloc[0]
    sc = u.canonical_city(str(r0.get("source_city", "")))
    tc = u.canonical_city(str(r0.get("target_city", "")))
    return f"{sc} \u2192 {tc}"


def _clean_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_retention(
    ax: Any,
    df: pd.DataFrame,
    *,
    split: str,
    calib: str,
    directions: List[str],
    strategies: List[str],
    rescale_mode: Optional[str],
    baseline_rescale: str,
    metric: str,
    text: TextFlags,
) -> None:
    x0 = np.arange(len(strategies), dtype=float)
    group_w = 0.78
    dir_w = group_w / max(1, len(directions))
    bw = dir_w * 0.92

    for di, d in enumerate(directions):
        face = _tgt_color(df, d)
        for si, s in enumerate(strategies):
            v = _pick_metric(
                df,
                direction=d,
                strategy=s,
                split=split,
                calib=calib,
                rescale_mode=rescale_mode,
                baseline_rescale=baseline_rescale,
                col=(
                    "overall_r2"
                    if metric == "r2"
                    else "overall_mae"
                ),
            )
            b = _pick_metric(
                df,
                direction=d,
                strategy="baseline",
                split=split,
                calib=calib,
                rescale_mode=rescale_mode,
                baseline_rescale=baseline_rescale,
                col=(
                    "overall_r2"
                    if metric == "r2"
                    else "overall_mae"
                ),
            )
            if not (np.isfinite(v) and np.isfinite(b)):
                continue

            if metric == "r2":
                y = v / b if b != 0 else float("nan")
            else:
                y = b / v if v != 0 else float("nan")

            if not np.isfinite(y):
                continue

            off = -0.5 * group_w + (di + 0.5) * dir_w
            xx = x0[si] + off

            ax.bar(
                xx,
                y,
                width=bw,
                color=face,
                alpha=0.86,
                hatch=cfg._STRAT_HATCH.get(s, ""),
                edgecolor=cfg._STRAT_EDGE.get(s, "#111111"),
                linewidth=0.6,
            )

    ax.axhline(1.0, linestyle="--", linewidth=0.7)

    ax.set_xticks(x0)
    ax.set_xticklabels(
        [cfg._STRAT_LABEL.get(s, s) for s in strategies],
        rotation=12,
        ha="right",
    )

    if text.show_labels:
        if metric == "r2":
            ax.set_ylabel("R² retention (× baseline)")
        else:
            ax.set_ylabel("MAE retention (× baseline)")
    else:
        ax.set_ylabel("")

    if not text.show_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    _clean_axes(ax)
    ax.grid(False)

def _pick_horizons(
    df: pd.DataFrame,
    *,
    metric: str,
    max_n: int = 3,
) -> list[str]:
    pref = f"per_horizon_{metric}."
    hs: list[str] = []

    for c in df.columns:
        c = str(c)
        if not c.startswith(pref):
            continue
        h = c.split(pref, 1)[1]
        if re.fullmatch(r"H\d+", h):
            hs.append(h)

    hs = sorted(set(hs), key=lambda s: int(s[1:]))
    return hs[:max_n] if hs else [f"H{i}" for i in range(1, max_n + 1)]

def _plot_horizon_ret(
    ax: Any,
    df: pd.DataFrame,
    *,
    direction: str,
    split: str,
    calib: str,
    strategies: List[str],
    rescale_mode: Optional[str],
    baseline_rescale: str,
    text: TextFlags,
) -> None:
    hs = _pick_horizons(df, metric="r2", max_n=3)
    x = np.arange(len(hs), dtype=float)

    # baseline per-horizon values
    base: list[float] = []
    for h in hs:
        col = f"per_horizon_r2.{h}"
        b = _pick_metric(
            df,
            direction=direction,
            strategy="baseline",
            split=split,
            calib=calib,
            rescale_mode=rescale_mode,
            baseline_rescale=baseline_rescale,
            col=col,
        )
        base.append(b)

    base_arr = np.asarray(base, dtype=float)
    face = _tgt_color(df, direction)

    for s in strategies:
        if str(s).lower() == "baseline":
            continue

        ys: list[float] = []
        for i, h in enumerate(hs):
            col = f"per_horizon_r2.{h}"
            v = _pick_metric(
                df,
                direction=direction,
                strategy=s,
                split=split,
                calib=calib,
                rescale_mode=rescale_mode,
                baseline_rescale=baseline_rescale,
                col=col,
            )
            b = base_arr[i]
            ok = np.isfinite(v) and np.isfinite(b) and b != 0.0
            ys.append(v / b if ok else float("nan"))

        yarr = np.asarray(ys, dtype=float)
        m = np.isfinite(yarr)
        if not m.any():
            continue

        ax.plot(
            x[m],
            yarr[m],
            color=face,
            linestyle=cfg._STRAT_LINESTYLE.get(s, "-"),
            marker=cfg._STRAT_MARKER.get(s, "o"),
            markersize=3.2,
            linewidth=1.0,
        )

    ax.axhline(1.0, linestyle="--", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(hs)  # shows H2/H3/H4 if that’s what exists
    ax.set_ylim(0.0, 1.25)

    if text.show_labels:
        ax.set_ylabel("R² retention")
        ax.set_xlabel("Horizon")
    else:
        ax.set_ylabel("")
        ax.set_xlabel("")

    if not text.show_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    _clean_axes(ax)
    ax.grid(False)

def _plot_cov_sharp(
    ax: Any,
    df: pd.DataFrame,
    *,
    direction: str,
    split: str,
    calib: str,
    strategies: List[str],
    rescale_mode: Optional[str],
    baseline_rescale: str,
    cov_target: float,
    text: TextFlags,
) -> None:
    face = _tgt_color(df, direction)

    for s in strategies:
        sub = _subset(
            df,
            direction=direction,
            strategy=s,
            split=split,
            calib=calib,
            rescale_mode=rescale_mode,
            baseline_rescale=baseline_rescale,
        )
        if sub.empty:
            continue

        x = _reduce_vals(sub["sharpness80"].values)
        y = _reduce_vals(sub["coverage80"].values)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        ax.scatter(
            x,
            y,
            s=55.0,
            facecolors=face,
            edgecolors=cfg._STRAT_EDGE.get(s, "#111111"),
            linewidths=0.9,
            alpha=0.92,
            marker=cfg._STRAT_MARKER.get(s, "o"),
        )

    ax.axhline(
        float(cov_target),
        linestyle="--",
        linewidth=0.7,
    )
    ax.set_ylim(0.0, 1.0)
    
    unit = _metrics_unit(df)
    
    if text.show_labels:
        ax.set_xlabel(f"Sharpness80 ({unit})")
        ax.set_ylabel("Coverage80")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if not text.show_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    _clean_axes(ax)
    ax.grid(False)


def _load_json_rows(p: Path) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _norm_rel_path(s: str) -> Path:
    # ss = str(s).replace("\\\\", "/").strip()
    # return Path(ss)
    ss = str(s).strip().replace("\\", "/")
    p = Path(ss)
    if p.is_absolute() or p.exists():
        return p
    # try repo root (scripts/..)
    repo = cfg.SCRIPTS_DIR.parent
    p2 = repo / p
    if p2.exists():
        return p2
    return p

def _risk_tables(
    rows: List[Dict[str, Any]],
    *,
    split: str,
    calib: str,
    threshold: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.DataFrame]]:
    if xfer_risk is None:
        return pd.DataFrame(), {}

    items: List[Dict[str, Any]] = []
    rel: Dict[Tuple[str, str], List[pd.DataFrame]] = {}

    for r in rows:
        if str(r.get("split", "")).lower() != split:
            continue
        if str(r.get("calibration", "")).lower() != calib:
            continue

        strat = str(r.get("strategy", "")).lower()
        direc = _canon_dir(r.get("direction", ""))
        pth = _norm_rel_path(str(r.get("csv_eval", "")))
        if not pth.exists():
            continue

        try:
            df = pd.read_csv(pth)
        except Exception:
            continue

        need = [
            "subsidence_actual",
            "subsidence_q10",
            "subsidence_q50",
            "subsidence_q90",
        ]
        if any(c not in df.columns for c in need):
            continue

        q10 = df["subsidence_q10"].to_numpy(float)
        q50 = df["subsidence_q50"].to_numpy(float)
        q90 = df["subsidence_q90"].to_numpy(float)
        ya = df["subsidence_actual"].to_numpy(float)

        pr = xfer_risk.exceed_prob_from_quantiles(
            q10,
            q50,
            q90,
            threshold=float(threshold),
        )
        yy = (ya >= float(threshold)).astype(float)

        bs = xfer_risk.brier_score(pr, yy)
        items.append(
            {
                "strategy": strat,
                "direction": direc,
                "brier": bs,
                "n": int(np.isfinite(pr).sum()),
            }
        )

        key = (direc, strat)
        rel_df = xfer_risk.reliability_bins(
            pr,
            yy,
            n_bins=10,
        )
        rel.setdefault(key, []).append(rel_df)

    if not items:
        return pd.DataFrame(), {}

    tab = pd.DataFrame(items)

    rel_out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for k, parts in rel.items():
        if not parts:
            continue
        dd = pd.concat(parts, axis=0, ignore_index=True)
        g = dd.groupby(
            ["bin_lo", "bin_hi"],
            as_index=False,
        )
        rel_out[k] = g.agg(
            {"p_mean": "mean", "y_rate": "mean", "n": "sum"}
        )

    return tab, rel_out


def _plot_risk(
    ax_rel: Any,
    ax_bs: Any,
    *,
    tab: pd.DataFrame,
    rel: Dict[Tuple[str, str], pd.DataFrame],
    directions: List[str],
    strategies: List[str],
    dir_colors: Dict[str, str],
    text: TextFlags,
) -> None:
    if tab.empty:
        ax_rel.text(
            0.5,
            0.5,
            "No eval CSVs for risk panel",
            ha="center",
            va="center",
        )
        ax_rel.set_axis_off()
        ax_bs.set_axis_off()
        return

    ax_rel.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=0.7,
    )

    for d in directions:
        for s in strategies:
            key = (_canon_dir(d), str(s).lower())
            rr = rel.get(key)
            if rr is None or rr.empty:
                continue
            color = dir_colors.get(key[0], "#777777")
            ax_rel.plot(
                rr["p_mean"].to_numpy(float),
                rr["y_rate"].to_numpy(float),
                color=color,
                linestyle=cfg._STRAT_LINESTYLE.get(key[1], "-"),
                marker=cfg._STRAT_MARKER.get(key[1], "o"),
                markersize=2.8,
                linewidth=1.0,
            )

    ax_rel.set_xlim(0.0, 1.0)
    ax_rel.set_ylim(0.0, 1.0)

    if text.show_labels:
        ax_rel.set_xlabel("Predicted prob")
        ax_rel.set_ylabel("Observed freq")
    else:
        ax_rel.set_xlabel("")
        ax_rel.set_ylabel("")

    _clean_axes(ax_rel)
    ax_rel.grid(False)

    g = tab.groupby(["direction", "strategy"], as_index=False)
    bs = g["brier"].mean()

    x0 = np.arange(len(strategies), dtype=float)
    group_w = 0.78
    dir_w = group_w / max(1, len(directions))
    bw = dir_w * 0.92

    for di, d in enumerate(directions):
        face = dir_colors.get(_canon_dir(d), "#777777")
        for si, s in enumerate(strategies):
            m = bs["direction"].eq(_canon_dir(d))
            m &= bs["strategy"].eq(str(s).lower())
            if not m.any():
                continue
            v = float(bs.loc[m, "brier"].iloc[0])
            off = -0.5 * group_w + (di + 0.5) * dir_w
            xx = x0[si] + off
            ax_bs.bar(
                xx,
                v,
                width=bw,
                color=face,
                alpha=0.86,
                hatch=cfg._STRAT_HATCH.get(s, ""),
                edgecolor=cfg._STRAT_EDGE.get(s, "#111111"),
                linewidth=0.6,
            )

    ax_bs.set_xticks(x0)
    ax_bs.set_xticklabels(
        [cfg._STRAT_LABEL.get(s, s) for s in strategies],
        rotation=12,
        ha="right",
    )

    if text.show_labels:
        ax_bs.set_ylabel("Brier (↓)")
    else:
        ax_bs.set_ylabel("")

    if not text.show_ticklabels:
        ax_bs.set_xticklabels([])
        ax_bs.set_yticklabels([])

    _clean_axes(ax_bs)
    ax_bs.grid(False)



def _hotspot_tables(
    rows: List[Dict[str, Any]],
    *,
    split: str,
    calib: str,
    strategies: List[str],
    directions: List[str],
    ref_strategy: str,
    score: str,
    horizon: str,
    threshold: float,
    k: int,
) -> pd.DataFrame:
    if xfer_risk is None:
        return pd.DataFrame()

    dfs: Dict[Tuple[str, str], pd.DataFrame] = {}

    for r in rows:
        if str(r.get("split", "")).lower() != split:
            continue
        if str(r.get("calibration", "")).lower() != calib:
            continue

        strat = str(r.get("strategy", "")).lower()
        direc = _canon_dir(r.get("direction", ""))
        pth = _norm_rel_path(str(r.get("csv_eval", "")))
        if not pth.exists():
            continue

        try:
            df = pd.read_csv(pth)
        except Exception:
            continue

        need = ["coord_t", "coord_x", "coord_y"]
        if any(c not in df.columns for c in need):
            continue

        try:
            rk = xfer_risk.prepare_hotspot_ranks(
                df,
                score=str(score).lower(),
                threshold=float(threshold),
                horizon=horizon,
            )
        except Exception:
            continue

        dfs[(direc, strat)] = rk

    ref = str(ref_strategy).lower()

    out: List[Dict[str, Any]] = []

    for d in directions:
        dd = _canon_dir(d)
        ref_rk = dfs.get((dd, ref))
        if ref_rk is None or ref_rk.empty:
            continue

        for s in strategies:
            ss = str(s).lower()
            if ss == ref:
                continue

            oth = dfs.get((dd, ss))
            if oth is None or oth.empty:
                continue

            st = xfer_risk.hotspot_stability(
                ref_rk,
                oth,
                k=int(k),
                time_col="coord_t",
                x_col="coord_x",
                y_col="coord_y",
                rank_col="rank",
            )

            if st.empty:
                continue

            out.append(
                {
                    "direction": dd,
                    "strategy": ss,
                    "jaccard": float(st["jaccard"].mean()),
                    "spearman": float(st["spearman"].mean()),
                    "n_years": int(len(st)),
                    "n_common": float(st["n_common"].mean()),
                }
            )

    return pd.DataFrame(out)


def _hotspot_series(
    rows: List[Dict[str, Any]],
    *,
    split: str,
    calib: str,
    strategies: List[str],
    directions: List[str],
    ref_strategy: str,
    score: str,
    horizon: str,
    threshold: float,
    k: int,
) -> pd.DataFrame:
    if xfer_risk is None:
        return pd.DataFrame()

    dfs: Dict[Tuple[str, str], pd.DataFrame] = {}

    for r in rows:
        if str(r.get("split", "")).lower() != split:
            continue
        if str(r.get("calibration", "")).lower() != calib:
            continue

        strat = str(r.get("strategy", "")).lower()
        direc = _canon_dir(r.get("direction", ""))
        pth = _norm_rel_path(str(r.get("csv_eval", "")))
        if not pth.exists():
            continue

        try:
            df = pd.read_csv(pth)
        except Exception:
            continue

        need = ["coord_t", "coord_x", "coord_y"]
        if any(c not in df.columns for c in need):
            continue

        try:
            rk = xfer_risk.prepare_hotspot_ranks(
                df,
                score=str(score).lower(),
                threshold=float(threshold),
                horizon=horizon,
            )
        except Exception:
            continue

        dfs[(direc, strat)] = rk

    ref = str(ref_strategy).lower()
    out: List[pd.DataFrame] = []

    for d in directions:
        dd = _canon_dir(d)
        ref_rk = dfs.get((dd, ref))
        if ref_rk is None or ref_rk.empty:
            continue

        for s in strategies:
            ss = str(s).lower()
            if ss == ref:
                continue

            oth = dfs.get((dd, ss))
            if oth is None or oth.empty:
                continue

            st = xfer_risk.hotspot_stability(
                ref_rk,
                oth,
                k=int(k),
                time_col="coord_t",
                x_col="coord_x",
                y_col="coord_y",
                rank_col="rank",
            )

            if st.empty:
                continue

            st = st.assign(direction=dd, strategy=ss)
            out.append(st)

    if not out:
        return pd.DataFrame()

    return pd.concat(out, axis=0, ignore_index=True)


def _plot_hotspots_bar(
    ax_j: Any,
    ax_s: Any,
    *,
    tab: pd.DataFrame,
    directions: List[str],
    strategies: List[str],
    ref_strategy: str,
    dir_colors: Dict[str, str],
    k: int,
    errorbars: bool,
    text: TextFlags,
) -> None:
    if tab.empty:
        ax_j.text(
            0.5,
            0.5,
            "No eval CSVs for hotspots",
            ha="center",
            va="center",
        )
        ax_j.set_axis_off()
        ax_s.set_axis_off()
        return

    ref = str(ref_strategy).lower()
    comps = [
        str(s).lower() for s in strategies
        if str(s).lower() != ref
    ]

    x0 = np.arange(len(comps), dtype=float)
    group_w = 0.78
    dir_w = group_w / max(1, len(directions))
    bw = dir_w * 0.92

    for di, d in enumerate(directions):
        dd = _canon_dir(d)
        face = dir_colors.get(dd, "#777777")
        for si, s in enumerate(comps):
            m = tab["direction"].eq(dd)
            m &= tab["strategy"].eq(s)
            if not m.any():
                continue

            jv = float(tab.loc[m, "jaccard_mean"].iloc[0])
            sv = float(tab.loc[m, "spearman_mean"].iloc[0])
            je = float(tab.loc[m, "jaccard_std"].iloc[0])
            se = float(tab.loc[m, "spearman_std"].iloc[0])

            off = -0.5 * group_w + (di + 0.5) * dir_w
            xx = x0[si] + off

            ec = cfg._STRAT_EDGE.get(s, "#111111")
            kw = dict(
                width=bw,
                color=face,
                alpha=0.86,
                hatch=cfg._STRAT_HATCH.get(s, ""),
                edgecolor=ec,
                linewidth=0.6,
            )
            if errorbars:
                ax_j.bar(xx, jv, yerr=je, capsize=2, **kw)
                ax_s.bar(xx, sv, yerr=se, capsize=2, **kw)
            else:
                ax_j.bar(xx, jv, **kw)
                ax_s.bar(xx, sv, **kw)

    ax_j.set_xticks(x0)
    ax_s.set_xticks(x0)

    labs = [cfg._STRAT_LABEL.get(s, s) for s in comps]
    ax_j.set_xticklabels(labs, rotation=12, ha="right")
    ax_s.set_xticklabels(labs, rotation=12, ha="right")

    ax_j.set_ylim(0.0, 1.0)
    ax_s.set_ylim(-1.0, 1.0)

    if text.show_labels:
        ax_j.set_ylabel(f"Jaccard@{int(k)}")
        ax_s.set_ylabel(f"Spearman@{int(k)}")
    else:
        ax_j.set_ylabel("")
        ax_s.set_ylabel("")

    if not text.show_ticklabels:
        ax_j.set_xticklabels([])
        ax_j.set_yticklabels([])
        ax_s.set_xticklabels([])
        ax_s.set_yticklabels([])

    _clean_axes(ax_j)
    _clean_axes(ax_s)
    ax_j.grid(False)
    ax_s.grid(False)


def _plot_hotspots_timeline(
    ax_j_ab: Any,
    ax_j_ba: Any,
    ax_s_ab: Any,
    ax_s_ba: Any,
    *,
    series: pd.DataFrame,
    strategies: List[str],
    ref_strategy: str,
    k: int,
    text: TextFlags,
) -> None:
    if series is None or series.empty:
        for ax in (ax_j_ab, ax_j_ba, ax_s_ab, ax_s_ba):
            ax.text(
                0.5,
                0.5,
                "No eval CSVs for hotspots",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
        return

    ref = str(ref_strategy).lower()
    comps = [
        str(s).lower() for s in strategies
        if str(s).lower() != ref
    ]

    ls = {"xfer": "--", "warm": "-", "baseline": "-"}
    mk = {"xfer": "o", "warm": "s", "baseline": "o"}

    def _one(
        ax: Any,
        *,
        direction: str,
        col: str,
        ylim: Tuple[float, float],
    ) -> None:
        dd = _canon_dir(direction)
        for s in comps:
            sub = series[series["direction"].eq(dd)]
            sub = sub[sub["strategy"].eq(s)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("coord_t")
            x = sub["coord_t"].to_numpy(float)
            y = sub[col].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if not m.any():
                continue
            ax.plot(
                x[m],
                y[m],
                linestyle=ls.get(s, "-"),
                marker=mk.get(s, "o"),
                markersize=2.6,
                linewidth=1.0,
                label=cfg._STRAT_LABEL.get(s, s),
            )

        ax.set_ylim(*ylim)
        _clean_axes(ax)
        ax.grid(False)

    _one(
        ax_j_ab,
        direction="A_to_B",
        col="jaccard",
        ylim=(0.0, 1.0),
    )
    _one(
        ax_j_ba,
        direction="B_to_A",
        col="jaccard",
        ylim=(0.0, 1.0),
    )
    _one(
        ax_s_ab,
        direction="A_to_B",
        col="spearman",
        ylim=(-1.0, 1.0),
    )
    _one(
        ax_s_ba,
        direction="B_to_A",
        col="spearman",
        ylim=(-1.0, 1.0),
    )

    if text.show_labels:
        ax_j_ab.set_ylabel(f"Jaccard@{int(k)}")
        ax_s_ab.set_ylabel(f"Spearman@{int(k)}")
        ax_s_ab.set_xlabel("Year")
        ax_s_ba.set_xlabel("Year")

    if not text.show_ticklabels:
        for ax in (ax_j_ab, ax_j_ba, ax_s_ab, ax_s_ba):
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if text.show_legend:
        ax_j_ba.legend(frameon=False, loc="best")


def _add_legends_fig(
    fig: Any,
    df: pd.DataFrame,
    *,
    directions: List[str],
    strategies: List[str],
    y: float,
) -> None:
    dir_handles: List[Any] = []
    for d in directions:
        lab = _dir_label(df, d)
        face = _tgt_color(df, d)
        dir_handles.append(
            Patch(
                facecolor=face,
                edgecolor="#111111",
                linewidth=0.6,
                label=lab,
            )
        )

    strat_handles = []
    for s in strategies:
        strat_handles.append(
            Line2D(
                [0], [0],
                color="#111111",
                linestyle=cfg._STRAT_LINESTYLE.get(s, "-"),
                marker=cfg._STRAT_MARKER.get(s, "o"),
                linewidth=1.0,
                markersize=4.0,
                label=cfg._STRAT_LABEL.get(s, s),
            )
        )
        
    fig.legend(
        handles=dir_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, y),
        title="Direction",
        borderaxespad=0.0,
        handlelength=1.2,
        labelspacing=0.25,
    )
    fig.legend(
        handles=strat_handles,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, y),
        title="Strategy",
        borderaxespad=0.0,
        handlelength=1.2,
        labelspacing=0.25,
    )

def render(
    df: pd.DataFrame,
    *,
    split: str,
    calib: str,
    strategies: List[str],
    directions: List[str],
    rescale_mode: Optional[str],
    baseline_rescale: str,
    cov_target: float,
    threshold: float,
    xfer_rows: List[Dict[str, Any]],
    add_hotspots: bool,
    hotspot_k: int,
    hotspot_score: str,
    hotspot_horizon: str,
    hotspot_ref: str,
    hotspot_style: str,
    hotspot_errorbars: bool,
    out: Path,
    text: TextFlags,
) -> Tuple[Path, Path]:
    u.set_paper_style()

    hs = str(hotspot_style).lower().strip()
    if add_hotspots and hs == "timeline":
        h = 7.6
    elif add_hotspots:
        h = 6.2
    else:
        h = 4.4
    r = 3 if add_hotspots else 2
    fig = plt.figure(figsize=(7.4, h))
    gs = GridSpec(
        r,
        2,
        figure=fig,
        wspace=0.35,
        hspace=0.55,
    )

    gs_a = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0, 0], wspace=0.55
    )
        
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])

    gs_b = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0, 1], wspace=0.55
    )
    
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1])

    gs_c = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1, 0], wspace=0.55
    )
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])

    gs_d = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1, 1], wspace=0.55
    )
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])

    ax_e1 = None
    ax_e2 = None
    ax_e3 = None
    ax_e4 = None
    if add_hotspots:
        if hs == "timeline":
            gs_e = GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=gs[2, :],
                wspace=0.35,
                hspace=0.45,
            )
            ax_e1 = fig.add_subplot(gs_e[0, 0])
            ax_e2 = fig.add_subplot(gs_e[0, 1])
            ax_e3 = fig.add_subplot(gs_e[1, 0])
            ax_e4 = fig.add_subplot(gs_e[1, 1])
        else:
            gs_e = GridSpecFromSubplotSpec(
                1, 2, subplot_spec=gs[2, :]
            )
            ax_e1 = fig.add_subplot(gs_e[0, 0])
            ax_e2 = fig.add_subplot(gs_e[0, 1])

    _plot_retention(
        ax_a1,
        df,
        split=split,
        calib=calib,
        directions=directions,
        strategies=strategies,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
        metric="r2",
        text=text,
    )
    _plot_retention(
        ax_a2,
        df,
        split=split,
        calib=calib,
        directions=directions,
        strategies=strategies,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
        metric="mae",
        text=text,
    )

    _plot_horizon_ret(
        ax_b1,
        df,
        direction="A_to_B",
        split=split,
        calib=calib,
        strategies=strategies,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
        text=text,
    )
    _plot_horizon_ret(
        ax_b2,
        df,
        direction="B_to_A",
        split=split,
        calib=calib,
        strategies=strategies,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
        text=text,
    )

    _plot_cov_sharp(
        ax_c1,
        df,
        direction="A_to_B",
        split=split,
        calib=calib,
        strategies=strategies,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
        cov_target=cov_target,
        text=text,
    )
    _plot_cov_sharp(
        ax_c2,
        df,
        direction="B_to_A",
        split=split,
        calib=calib,
        strategies=strategies,
        rescale_mode=rescale_mode,
        baseline_rescale=baseline_rescale,
        cov_target=cov_target,
        text=text,
    )

    tab, rel = _risk_tables(
        xfer_rows,
        split=split,
        calib=calib,
        threshold=float(threshold),
    )

    dir_colors = {
        _canon_dir(d): _tgt_color(df, d)
        for d in directions
    }
    _plot_risk(
        ax_d1,
        ax_d2,
        tab=tab,
        rel=rel,
        directions=directions,
        strategies=strategies,
        dir_colors=dir_colors,
        text=text,
    )

    if add_hotspots and ax_e1 is not None:
        hs_series = _hotspot_series(
            xfer_rows,
            split=split,
            calib=calib,
            strategies=strategies,
            directions=directions,
            ref_strategy=hotspot_ref,
            score=hotspot_score,
            horizon=hotspot_horizon,
            threshold=float(threshold),
            k=int(hotspot_k),
        )

        if hs == "timeline" and ax_e3 is not None:
            _plot_hotspots_timeline(
                ax_e1,
                ax_e2,
                ax_e3,
                ax_e4,
                series=hs_series,
                strategies=strategies,
                ref_strategy=hotspot_ref,
                k=int(hotspot_k),
                text=text,
            )
        else:
            if xfer_risk is None:
                hs_tab = pd.DataFrame()
            else:
                hs_tab = xfer_risk.stability_group_summary(
                    hs_series,
                    group_cols=("direction", "strategy"),
                    time_col="coord_t",
                    ddof=1,
                )
            _plot_hotspots_bar(
                ax_e1,
                ax_e2,
                tab=hs_tab,
                directions=directions,
                strategies=strategies,
                ref_strategy=hotspot_ref,
                dir_colors=dir_colors,
                k=int(hotspot_k),
                errorbars=bool(hotspot_errorbars),
                text=text,
            )

    if text.show_panel_titles:
        ax_a1.set_title("(a) Retention: R²")
        ax_a2.set_title("(a) Retention: MAE")

        ax_b1.set_title(
            "(b) Horizon:\n" + _dir_label(df, "A_to_B"),
            pad=2,
        )
        ax_b2.set_title(
            "(b) Horizon:\n" + _dir_label(df, "B_to_A"),
            pad=2,
        )
        ax_c1.set_title(
            "(c) Cov–sharp:\n" + _dir_label(df, "A_to_B"),
            pad=2,
        )
        ax_c2.set_title(
            "(c) Cov–sharp:\n" + _dir_label(df, "B_to_A"),
            pad=2,
        )

        unit = _metrics_unit(df)
        ax_d1.set_title(f"(d) Reliability @ {threshold:g} {unit}")
        ax_d2.set_title("(d) Brier")
        if add_hotspots and ax_e1 is not None:
            k0 = int(hotspot_k)
            if hs == "timeline" and ax_e3 is not None:
                ax_e1.set_title(
                    f"(e) J@{k0}: "
                    + _dir_label(df, "A_to_B")
                )
                ax_e2.set_title(
                    f"(e) J@{k0}: "
                    + _dir_label(df, "B_to_A")
                )
                ax_e3.set_title(
                    f"(e) ρ@{k0}: "
                    + _dir_label(df, "A_to_B")
                )
                ax_e4.set_title(
                    f"(e) ρ@{k0}: "
                    + _dir_label(df, "B_to_A")
                )
            else:
                ax_e1.set_title(f"(e) Hotspots J@{k0}")
                ax_e2.set_title(f"(e) Hotspots ρ@{k0}")

    has_leg = bool(text.show_legend)
    has_ttl = bool(text.show_title)
    
    if has_leg:
        y_leg = 0.995 if not has_ttl else 0.965
        _add_legends_fig(
            fig,
            df,
            directions=directions,
            strategies=strategies,
            y=y_leg,
        )
    
    if has_ttl:
        t0 = "Supplementary Figure Sy — Transfer impact"
        t = u.resolve_title(default=t0, title=text.title)
        y_ttl = 0.94 if has_leg else 0.99
        fig.suptitle(t, x=0.02, y=y_ttl, ha="left")
    
    # reserve headroom for (legend + title)
    if has_leg and has_ttl:
        fig.subplots_adjust(top=0.86)
    elif has_leg:
        fig.subplots_adjust(top=0.90)
    elif has_ttl:
        fig.subplots_adjust(top=0.93)
        
    top = 0.86 if (has_leg and has_ttl) else (
        0.90 if has_leg else (0.93 if has_ttl else 0.98))
    # fig.tight_layout(rect=[0.0, 0.0, 1.0, top])

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
    ap = argparse.ArgumentParser(
        prog="plot-xfer-impact",
        description=(
            "Supplementary Sy — transfer impact plots."
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
        help="Explicit xfer_results.csv (file or dir)",
    )
    ap.add_argument(
        "--xfer-json",
        type=str,
        default=None,
        help="Explicit xfer_results.json (optional)",
    )

    ap.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("val", "test"),
        help="Which split to plot",
    )
    ap.add_argument(
        "--calib",
        type=str,
        default="source",
        choices=("none", "source", "target"),
        help="Calibration mode for panels",
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=["baseline", "xfer", "warm"],
        help="Strategies (baseline xfer warm)",
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
        "--cov-target",
        type=float,
        default=0.80,
        help="Coverage reference line",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Exceedance threshold for risk panel",
    )


    ap.add_argument(
        "--add-hotspots",
        type=str,
        default="false",
        help="Add hotspot stability panels (true/false).",
    )
    ap.add_argument(
        "--hotspot-k",
        type=int,
        default=100,
        help="Top-K hotspots for stability.",
    )
    ap.add_argument(
        "--hotspot-score",
        type=str,
        default="q50",
        choices=("q50", "exceed"),
        help="Hotspot score: q50 or exceed.",
    )
    ap.add_argument(
        "--hotspot-horizon",
        type=str,
        default="H3",
        choices=("H1", "H2", "H3", "all"),
        help="Which horizon to rank (H1/H2/H3/all).",
    )
    ap.add_argument(
        "--hotspot-ref",
        type=str,
        default="baseline",
        help="Reference strategy for stability.",
    )

    ap.add_argument(
        "--hotspot-style",
        type=str,
        default="bar",
        choices=("bar", "timeline"),
        help="Hotspot panel: bar or timeline.",
    )
    ap.add_argument(
        "--hotspot-errorbars",
        type=str,
        default="false",
        help="Show mean±std bars (true/false).",
    )

    u.add_plot_text_args(
        ap,
        default_out="figureS_xfer_impact",
    )

    return ap.parse_args(argv)


def _text_flags(args: Any) -> TextFlags:
    return TextFlags(
        show_legend=u.str_to_bool(
            args.show_legend, default=True
        ),
        show_labels=u.str_to_bool(
            args.show_labels, default=True
        ),
        show_ticklabels=u.str_to_bool(
            args.show_ticklabels,
            default=True,
        ),
        show_title=u.str_to_bool(
            args.show_title, default=True
        ),
        show_panel_titles=u.str_to_bool(
            args.show_panel_titles,
            default=True,
        ),
        title=args.title,
    )


def figSx_xfer_impact_main(
    argv: List[str] | None = None,
) -> None:
    u.ensure_script_dirs()

    args = parse_args(argv)
    text = _text_flags(args)

    split = str(args.split).lower()

    calib = str(args.calib).lower()

    strategies = [str(s).lower() for s in args.strategies]
    directions = ["A_to_B", "B_to_A"]

    rows: List[Dict[str, Any]] = []

    rm = str(args.rescale_mode).lower()
    brm = str(args.baseline_rescale).lower()

    rows: List[Dict[str, Any]] = []
    picked: Dict[str, Path] = {}

    # --- Resolve CSV(s) ---
    if args.xfer_csv:
        p = Path(args.xfer_csv).expanduser()
    else:
        p = Path(args.src).expanduser()

    if p.is_dir() and _is_pair_dir(p):
        df_raw, rows_raw, picked = _load_multi_jobs(
            p,
            strategies=strategies,
            split=split,
            calib=calib,
            rescale_mode=rm,
            baseline_rescale=brm,
        )
        df = _canon_cols(df_raw)
        # only use merged JSON rows if user didn't force one
        if not args.xfer_json:
            rows = rows_raw
        if picked:
            for s, fp in picked.items():
                print(f"[pick] {s}: {fp.parent.name}")
    else:
        csv_p = _find_xfer_csv(p if p.exists() else args.src)
        df = _canon_cols(pd.read_csv(csv_p))

        if args.xfer_json:
            jp = Path(args.xfer_json).expanduser()
            if jp.is_dir():
                jp = jp / "xfer_results.json"
            if jp.exists():
                rows = _load_json_rows(jp)
        else:
            jp2 = _find_xfer_json(csv_p)
            if jp2 is not None and jp2.exists():
                rows = _load_json_rows(jp2)

    # --- Apply filters after loading ---
    df = df[df["split"].eq(split)].copy()
    if df.empty:
        raise SystemExit("No rows after split filter.")
    df = df[df["calibration"].eq(calib)].copy()
    if df.empty:
        raise SystemExit("No rows after calib filter.")
    
    out = u.resolve_fig_out(args.out)

    png, svg = render(
        df,
        split=split,
        calib=calib,
        strategies=strategies,
        directions=directions,
        rescale_mode=rm,
        baseline_rescale=brm,
        cov_target=float(args.cov_target),
        threshold=float(args.threshold),
        xfer_rows=rows,
        add_hotspots=u.str_to_bool(
            args.add_hotspots, default=False
        ),
        hotspot_k=int(args.hotspot_k),
        hotspot_score=str(args.hotspot_score).lower(),
        hotspot_horizon=str(args.hotspot_horizon),
        hotspot_ref=str(args.hotspot_ref).lower(),
        hotspot_style=str(args.hotspot_style).lower(),
        hotspot_errorbars=u.str_to_bool(
            args.hotspot_errorbars,
            default=False,
        ),
        out=out,
        text=text,
    )

    print(f"[OK] Wrote {png}")
    print(f"[OK] Wrote {svg}")


def main(argv: List[str] | None = None) -> None:
    figSx_xfer_impact_main(argv)


if __name__ == "__main__":
    main()
