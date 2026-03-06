# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
#
"""
Supplement S7 • Physics sensitivity (ε_prior, ε_cons)

Robust plotting over (λ_cons, λ_prior) with:
- precedence for ablation_record.updated*.jsonl
- fallback to legacy ablation_record*.jsonl
- explicit --input support: CSV / JSON / JSONL
- model/city/pde_mode filters
- render styles via --render:
    heatmap | tricontour | pcolormesh

Outputs
-------
- Figure: <out>.png and <out>.pdf
- Used tidy table: tableS7_physics_used.csv
  written next to the figure outputs.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from scripts import config as cfg
from scripts import utils


_LOWER_IS_BETTER = {
    "mae",
    "mse",
    "rmse",
    "sharpness80",
    "epsilon_prior",
    "epsilon_cons",
    "pss",
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args(
    argv: List[str] | None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot-physics-sensitivity",
        description="Supplement S7: Physics sensitivity",
    )

    p.add_argument(
        "--results-root",
        "--root",
        dest="root",
        type=str,
        default="results",
        help="Root to scan for ablation records.",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        action="append",
        help=(
            "Explicit ablation table/file (repeatable). "
            "Supports .jsonl / .json / .csv."
        ),
    )
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

    utils.add_city_flags(p, default_both=True)

    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma list of model names to keep.",
    )
    p.add_argument(
        "--pde-modes",
        type=str,
        default="on,both,consolidation",
        help="Comma list of pde_mode to keep or 'all'.",
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
            "rmse",
            "pss",
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
            "rmse",
            "pss",
        ],
    )

    utils.add_render_args(p, default="heatmap")

    utils.add_plot_text_args(
        p,
        default_out="supp_fig_S7_physics_sensitivity",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------
# I/O loaders
# ---------------------------------------------------------------------
def _fit_plane_grad(
    sub: pd.DataFrame,
    metric: str,
    *,
    min_n: int,
) -> Optional[Tuple[float, float]]:
    """
    Fit z = a*x + b*y + c and return (a, b).
    """
    cols = ["lambda_prior", "lambda_cons", metric]
    d = sub[cols].copy().dropna()
    if len(d) < int(min_n):
        return None

    x = d["lambda_prior"].astype(float).values
    y = d["lambda_cons"].astype(float).values
    z = d[metric].astype(float).values

    if not np.isfinite(z).any():
        return None

    A = np.column_stack([x, y, np.ones_like(x)])
    try:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    except Exception:
        return None

    a = float(coef[0])
    b = float(coef[1])

    if not np.isfinite(a) or not np.isfinite(b):
        return None

    if abs(a) + abs(b) < 1e-12:
        return None

    return (a, b)


def _add_trend_arrow(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    metric: str,
    lower_is_better: bool,
    arrow_len: float,
    arrow_pos: str,
    min_n: int,
) -> None:
    """
    Draw a subtle arrow indicating improvement direction.
    """
    g = _fit_plane_grad(sub, metric, min_n=min_n)
    if g is None:
        return

    gx, gy = g

    # Improvement direction in data space.
    if lower_is_better:
        gx, gy = (-gx, -gy)

    # Use a data-space step around center.
    x = sub["lambda_prior"].astype(float).values
    y = sub["lambda_cons"].astype(float).values
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return

    x0 = float(np.nanmean(x))
    y0 = float(np.nanmean(y))

    xr = float(np.nanmax(x) - np.nanmin(x))
    yr = float(np.nanmax(y) - np.nanmin(y))
    step = 0.25 * max(xr, yr, 1e-9)

    v = np.array([gx, gy], dtype=float)
    nv = float(np.linalg.norm(v))
    if not np.isfinite(nv) or nv <= 0.0:
        return
    v = v / nv

    p0 = np.array([x0, y0])
    p1 = p0 + v * step

    # Convert direction to axes-fraction.
    t_data = ax.transData
    t_ax = ax.transAxes.inverted()

    q0 = t_ax.transform(t_data.transform(p0))
    q1 = t_ax.transform(t_data.transform(p1))
    d = q1 - q0

    nd = float(np.linalg.norm(d))
    if not np.isfinite(nd) or nd <= 0.0:
        return
    d = d / nd

    # Parse anchor position.
    try:
        px, py = [float(v) for v in arrow_pos.split(",")]
    except Exception:
        px, py = (0.78, 0.14)

    dx = float(d[0]) * float(arrow_len)
    dy = float(d[1]) * float(arrow_len)

    ax.annotate(
        "",
        xy=(px + dx, py + dy),
        xytext=(px, py),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.2,
            color="white",
            shrinkA=0.0,
            shrinkB=0.0,
        ),
        zorder=5,
    )
    
def _read_jsonl(fp: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with fp.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if isinstance(rec, dict):
                rows.append(rec)
    return rows


def _load_one_input(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()

    if suf == ".csv":
        df = pd.read_csv(path)
        df["_src"] = str(path)
        return df

    if suf == ".jsonl":
        rows = _read_jsonl(path)
        df = pd.DataFrame(rows)
        df["_src"] = str(path)
        return df

    if suf == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            obj = None

        if isinstance(obj, list):
            df = pd.DataFrame(obj)
            df["_src"] = str(path)
            return df

        if isinstance(obj, dict):
            df = pd.DataFrame([obj])
            df["_src"] = str(path)
            return df

    return pd.DataFrame([])


def _scan_records(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    files = utils.find_all(
        root,
        cfg.PATTERNS.get("ablation_record_jsonl", ()),
    )

    for fp in files:
        df = _load_one_input(fp)
        if not df.empty:
            rows.append(df.to_dict("records"))

    if not rows:
        return pd.DataFrame([])

    flat: List[Dict[str, Any]] = []
    for block in rows:
        flat.extend(block)

    return pd.DataFrame(flat)


def _load_records(args: argparse.Namespace) -> pd.DataFrame:
    if args.input:
        dfs: List[pd.DataFrame] = []
        for s in args.input:
            p = utils.as_path(s)
            if p.exists():
                dfs.append(_load_one_input(p))

        if not dfs:
            return pd.DataFrame([])
        return pd.concat(dfs, ignore_index=True)

    root = utils.as_path(args.root)
    return _scan_records(root)


# ---------------------------------------------------------------------
# Canonicalization + robustness
# ---------------------------------------------------------------------
def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    aliases = {
        "timestamp": ("timestamp", "ts"),
        "city": ("city", "City"),
        "model": ("model", "Model"),
        "pde_mode": ("pde_mode", "pde"),
        "lambda_cons": ("lambda_cons", "lambda_c"),
        "lambda_prior": ("lambda_prior", "lambda_p"),
        "epsilon_prior": ("epsilon_prior", "epsilon_p"),
        "epsilon_cons": ("epsilon_cons", "epsilon_c"),
        "epsilon_gw": ("epsilon_gw", "epsilon_g"),
        "coverage80": ("coverage80", "coverage8"),
        "sharpness80": ("sharpness80", "sharpness"),
    }

    utils.ensure_columns(df, aliases=aliases)

    if "city" in df.columns:
        df["city"] = df["city"].astype(str).map(utils.canonical_city)

    num_cols = [
        "lambda_cons",
        "lambda_prior",
        "epsilon_prior",
        "epsilon_cons",
        "epsilon_gw",
        "r2",
        "mae",
        "mse",
        "rmse",
        "coverage80",
        "sharpness80",
        "pss",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # rmse fallback
    if "rmse" in df.columns and "mse" in df.columns:
        m = df["rmse"].isna() & df["mse"].notna()
        df.loc[m, "rmse"] = np.sqrt(df.loc[m, "mse"])

    return df


def _needs_si_to_mm(row: pd.Series) -> bool:
    u = row.get("units", None)
    if isinstance(u, dict):
        uu = str(u.get("subs_metrics_unit", "")).lower()
        if uu == "mm":
            return False
        if uu in {"m", "meter", "metre"}:
            return True

    mae = row.get("mae", np.nan)
    mse = row.get("mse", np.nan)
    shp = row.get("sharpness80", np.nan)

    # Safe heuristic for legacy SI:
    # MAE ~ 0.0x, MSE ~ 0.000x, sharpness ~ 0.0x
    ok = (
        np.isfinite(mae)
        and np.isfinite(mse)
        and np.isfinite(shp)
        and 0.0 < float(mae) < 1.0
        and 0.0 < float(mse) < 1.0
        and 0.0 < float(shp) < 1.0
    )
    return bool(ok)


def _harmonize_units(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "mae" not in df.columns or "mse" not in df.columns:
        return df

    mask = df.apply(_needs_si_to_mm, axis=1)
    if not mask.any():
        return df

    # m -> mm for distance-like
    for c in ["mae", "rmse", "sharpness80"]:
        if c in df.columns:
            df.loc[mask, c] = df.loc[mask, c] * 1000.0

    # m^2 -> mm^2
    if "mse" in df.columns:
        df.loc[mask, "mse"] = df.loc[mask, "mse"] * 1e6

    return df


def _record_score(row: pd.Series) -> int:
    s = 0
    src = str(row.get("_src", "")).lower()
    if "updated" in src:
        s += 100

    u = row.get("units", None)
    if isinstance(u, dict):
        uu = str(u.get("subs_metrics_unit", "")).lower()
        if uu == "mm":
            s += 50

    if isinstance(row.get("legacy", None), dict):
        s += 10
    if isinstance(row.get("metrics", None), dict):
        s += 10

    for k in ["rmse", "pss", "epsilon_prior", "epsilon_cons"]:
        v = row.get(k, np.nan)
        if np.isfinite(v):
            s += 1

    return int(s)


def _dedupe_prefer_best(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        return df

    df = df.copy()
    if "city" not in df.columns:
        df["city"] = ""

    df["_score"] = df.apply(_record_score, axis=1)

    df = df.sort_values(
        by=["timestamp", "city", "_score"],
        ascending=[True, True, False],
    )

    df = df.drop_duplicates(
        subset=["timestamp", "city"],
        keep="first",
    ).copy()

    df = df.drop(columns=["_score"], errors="ignore")
    return df


def _filter_df(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, List[str]]:
    if df.empty:
        return df, []

    cities = utils.resolve_cities(args)
    if cities and "city" in df.columns:
        df = df[df["city"].isin(cities)].copy()

    raw_models = str(args.models or "").strip()
    if raw_models:
        keep = [m.strip() for m in raw_models.split(",") if m.strip()]
        if keep and "model" in df.columns:
            df = df[df["model"].astype(str).isin(keep)].copy()

    pm = str(args.pde_modes or "").strip().lower()
    if pm and pm != "all" and "pde_mode" in df.columns:
        keep = [x.strip() for x in pm.split(",") if x.strip()]
        df = df[df["pde_mode"].astype(str).str.lower().isin(keep)]

    return df, cities


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def _metric_label(metric: str) -> str:
    k = str(metric).strip()
    if k in cfg.PHYS_LABELS:
        return cfg.PHYS_LABELS[k]
    meta = cfg.PLOT_METRIC_META.get(k, None)
    if isinstance(meta, dict):
        yl = str(meta.get("ylabel", k))
        return yl.format(unit=str(meta.get("unit", "")))
    return k


def _lower_is_better(metric: str) -> bool:
    return str(metric).lower() in _LOWER_IS_BETTER


def _parse_clip(x: str) -> Tuple[float, float]:
    s = str(x or "").strip()
    if not s:
        return (2.0, 98.0)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        return (2.0, 98.0)
    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except Exception:
        return (2.0, 98.0)
    lo = max(0.0, min(100.0, lo))
    hi = max(0.0, min(100.0, hi))
    if hi <= lo:
        return (2.0, 98.0)
    return (lo, hi)


def _row_norm(
    df: pd.DataFrame,
    metric: str,
    *,
    clip: str,
) -> Tuple[float, float]:
    if metric not in df.columns:
        return (np.nan, np.nan)

    vals = pd.to_numeric(df[metric], errors="coerce").values
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)

    lo, hi = _parse_clip(clip)
    vmin = float(np.percentile(vals, lo))
    vmax = float(np.percentile(vals, hi))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (np.nan, np.nan)

    if vmax <= vmin:
        vmax = vmin + 1e-12

    return (vmin, vmax)


def _best_point(
    df: pd.DataFrame,
    metric: str,
) -> Optional[Tuple[float, float]]:
    if df.empty or metric not in df.columns:
        return None

    sub = df[
        ["lambda_prior", "lambda_cons", metric]
    ].copy()

    sub = sub.dropna()
    if sub.empty:
        return None

    if _lower_is_better(metric):
        j = int(np.nanargmin(sub[metric].values))
    else:
        j = int(np.nanargmax(sub[metric].values))

    x = float(sub.iloc[j]["lambda_prior"])
    y = float(sub.iloc[j]["lambda_cons"])
    return (x, y)


def _panel_label(ax: plt.Axes, lab: str) -> None:
    ax.text(
        0.02,
        0.98,
        lab,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )


def _axes_cleanup(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _pivot_grid(
    df: pd.DataFrame,
    *,
    metric: str,
    agg: str,
) -> pd.DataFrame:
    f = np.nanmean if agg == "mean" else np.nanmedian
    piv = df.pivot_table(
        index="lambda_cons",
        columns="lambda_prior",
        values=metric,
        aggfunc=f,
    )
    piv = piv.sort_index().sort_index(axis=1)
    return piv


def _render_heatmap(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    metric: str,
    cmap: str,
    vmin: float,
    vmax: float,
    agg: str,
    show_ticks: bool,
    show_labels: bool,
    show_points: bool,
) -> Optional[Any]:
    piv = _pivot_grid(sub, metric=metric, agg=agg)
    if piv.empty:
        return None

    im = ax.imshow(
        piv.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(range(piv.shape[1]))
    ax.set_yticks(range(piv.shape[0]))

    if show_ticks:
        ax.set_xticklabels(
            [f"{c:.2g}" for c in piv.columns],
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

    if show_points:
        # overlay sampled cells (grid index coords)
        xs = []
        ys = []
        for yi, lc in enumerate(piv.index):
            for xi, lp in enumerate(piv.columns):
                v = piv.loc[lc, lp]
                if np.isfinite(v):
                    xs.append(xi)
                    ys.append(yi)
        ax.scatter(
            xs,
            ys,
            s=10,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
        )

    return im


def _render_tricontour(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    metric: str,
    cmap: str,
    vmin: float,
    vmax: float,
    levels: int,
    show_ticks: bool,
    show_labels: bool,
    show_points: bool,
) -> Optional[Any]:
    import matplotlib.tri as mtri

    d = sub[
        ["lambda_prior", "lambda_cons", metric]
    ].copy()
    d = d.dropna()
    if len(d) < 3:
        return None

    x = d["lambda_prior"].astype(float).values
    y = d["lambda_cons"].astype(float).values
    z = d[metric].astype(float).values

    # de-dup identical (x,y) to avoid tri issues
    key = pd.Series(list(zip(x, y)))
    keep = ~key.duplicated()
    x = x[keep.values]
    y = y[keep.values]
    z = z[keep.values]

    if len(z) < 3:
        return None

    tri = mtri.Triangulation(x, y)

    levs = int(levels)
    if levs < 4:
        levs = 4

    im = ax.tricontourf(
        tri,
        z,
        levels=levs,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.tricontour(
        tri,
        z,
        levels=levs,
        colors="white",
        linewidths=0.3,
        alpha=0.5,
    )

    if show_points:
        ax.scatter(
            x,
            y,
            s=14,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
        )

    if show_labels:
        ax.set_xlabel(r"$\lambda_{\mathrm{prior}}$")
        ax.set_ylabel(r"$\lambda_{\mathrm{cons}}$")

    if not show_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return im


def _render_pcolormesh(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    metric: str,
    cmap: str,
    vmin: float,
    vmax: float,
    agg: str,
    show_ticks: bool,
    show_labels: bool,
    show_points: bool,
) -> Optional[Any]:
    piv = _pivot_grid(sub, metric=metric, agg=agg)
    if piv.empty:
        return None

    xs = piv.columns.astype(float).values
    ys = piv.index.astype(float).values
    z = piv.values

    # build cell edges (midpoints)
    def _edges(v: np.ndarray) -> np.ndarray:
        if len(v) == 1:
            dv = 1.0
            return np.array([v[0] - dv, v[0] + dv])
        mid = (v[1:] + v[:-1]) / 2.0
        e0 = v[0] - (mid[0] - v[0])
        e1 = v[-1] + (v[-1] - mid[-1])
        return np.concatenate([[e0], mid, [e1]])

    xe = _edges(xs)
    ye = _edges(ys)

    im = ax.pcolormesh(
        xe,
        ye,
        z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    if show_points:
        xx, yy = np.meshgrid(xs, ys)
        m = np.isfinite(z)
        ax.scatter(
            xx[m],
            yy[m],
            s=14,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
        )

    if show_labels:
        ax.set_xlabel(r"$\lambda_{\mathrm{prior}}$")
        ax.set_ylabel(r"$\lambda_{\mathrm{cons}}$")

    if not show_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return im


def _plot_cell(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    city: str,
    metric: str,
    render: str,
    cmap: str,
    vmin: float,
    vmax: float,
    levels: int,
    agg: str,
    show_ticks: bool,
    show_labels: bool,
    show_points: bool,
    trend_arrow: bool, 
    trend_arrow_len:float,
    trend_arrow_pos:str,
    trend_arrow_min_n:int,
) -> Optional[Any]:
    sub = df[df["city"].astype(str) == str(city)].copy()

    need = ["lambda_cons", "lambda_prior", metric]
    for c in need:
        if c not in sub.columns:
            ax.set_axis_off()
            return None

    sub = sub.dropna(subset=["lambda_cons", "lambda_prior"])
    if sub.empty:
        ax.set_axis_off()
        return None

    if render == "tricontour":
        im = _render_tricontour(
            ax,
            sub,
            metric=metric,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            show_ticks=show_ticks,
            show_labels=show_labels,
            show_points=show_points,
        )
    elif render == "pcolormesh":
        im = _render_pcolormesh(
            ax,
            sub,
            metric=metric,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            agg=agg,
            show_ticks=show_ticks,
            show_labels=show_labels,
            show_points=show_points,
        )
    else:
        im = _render_heatmap(
            ax,
            sub,
            metric=metric,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            agg=agg,
            show_ticks=show_ticks,
            show_labels=show_labels,
            show_points=show_points,
        )

    if im is None:
        ax.set_axis_off()
        return None

    # best marker (in parameter space)
    bp = _best_point(sub, metric)
    if bp is not None:
        x, y = bp
        if render == "heatmap":
            # map (x,y) to pivot indices for heatmap
            piv = _pivot_grid(sub, metric=metric, agg=agg)
            if (
                x in piv.columns.astype(float).values
                and y in piv.index.astype(float).values
            ):
                j = int(np.where(piv.columns == x)[0][0])
                i = int(np.where(piv.index == y)[0][0])
                ax.scatter(
                    [j],
                    [i],
                    s=50,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=1.5,
                )
        else:
            ax.scatter(
                [x],
                [y],
                s=60,
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
            )

    if trend_arrow:
        _add_trend_arrow(
            ax,
            sub,
            metric=metric,
            lower_is_better=_lower_is_better(metric),
            arrow_len=float(trend_arrow_len),
            arrow_pos=str(trend_arrow_pos),
            min_n=int(trend_arrow_min_n),
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
def plot_physics_sensitivity_main(
    argv: List[str] | None = None,
) -> None:
    args = _parse_args(argv)

    utils.set_paper_style(
        fontsize=int(args.font),
        dpi=int(args.dpi),
    )
    trend_arrow = utils.str_to_bool(
        args.trend_arrow,
        default=False,
    )
    
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

    show_points = utils.str_to_bool(
        args.show_points,
        default=True,
    )

    df = _load_records(args)
    if df.empty:
        raise SystemExit("No ablation data found.")

    df = _canon_cols(df)
    df = _harmonize_units(df)
    df = _dedupe_prefer_best(df)

    df, cities = _filter_df(df, args)
    if df.empty:
        raise SystemExit("No rows after filtering.")

    if not cities:
        cities = sorted(df["city"].dropna().unique().tolist())

    out = _resolve_out(out=args.out, out_dir=args.out_dir)
    utils.ensure_dir(out.parent)

    used_csv = out.parent / "tableS7_physics_used.csv"
    df.to_csv(used_csv, index=False)
    print(f"[OK] table -> {used_csv}")

    # row-wise normalization (shared across cities per row)
    vmin1, vmax1 = _row_norm(df, args.metric_prior, clip=args.clip)
    vmin2, vmax2 = _row_norm(df, args.metric_cons, clip=args.clip)

    # Nature-style: best should pop visually
    if _lower_is_better(args.metric_prior):
        cmap1 = "magma_r"
    else:
        cmap1 = "magma"

    if _lower_is_better(args.metric_cons):
        cmap2 = "magma_r"
    else:
        cmap2 = "magma"

    ncols = max(1, len(cities))
    fig = plt.figure(figsize=(4.2 * ncols, 7.0))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=ncols,
        left=0.08,
        right=0.90 if show_legend else 0.98,
        top=0.94,
        bottom=0.10,
        hspace=0.35,
        wspace=0.30,
    )

    ims1: List[Any] = []
    ims2: List[Any] = []

    # Row 1
    for j, city in enumerate(cities):
        ax = fig.add_subplot(gs[0, j])
        im = _plot_cell(
            ax,
            df,
            city=city,
            metric=args.metric_prior,
            render=str(args.render),
            cmap=cmap1,
            vmin=vmin1,
            vmax=vmax1,
            levels=int(args.levels),
            agg=str(args.agg),
            show_ticks=show_ticks,
            show_labels=show_labels,
            show_points=show_points,
            trend_arrow=trend_arrow,
            trend_arrow_len=args.trend_arrow_len,
            trend_arrow_pos=args.trend_arrow_pos,
            trend_arrow_min_n=args.trend_arrow_min_n,
        )
        if show_pan_t:
            ax.set_title(
                f"{city} — {_metric_label(args.metric_prior)}",
                loc="left",
                pad=6,
                fontweight="bold",
            )
        if j == 0:
            _panel_label(ax, "a")
        elif j == 1:
            _panel_label(ax, "b")
        if im is not None:
            ims1.append(im)

    # Row 2
    for j, city in enumerate(cities):
        ax = fig.add_subplot(gs[1, j])
        im = _plot_cell(
            ax,
            df,
            city=city,
            metric=args.metric_cons,
            render=str(args.render),
            cmap=cmap2,
            vmin=vmin2,
            vmax=vmax2,
            levels=int(args.levels),
            agg=str(args.agg),
            show_ticks=show_ticks,
            show_labels=show_labels,
            show_points=show_points,
            trend_arrow=trend_arrow,
            trend_arrow_len=args.trend_arrow_len,
            trend_arrow_pos=args.trend_arrow_pos,
            trend_arrow_min_n=args.trend_arrow_min_n,
        )
        if show_pan_t:
            ax.set_title(
                f"{city} — {_metric_label(args.metric_cons)}",
                loc="left",
                pad=6,
                fontweight="bold",
            )
        if j == 0:
            _panel_label(ax, "c")
        elif j == 1:
            _panel_label(ax, "d")
        if im is not None:
            ims2.append(im)

    # Shared colorbars (one per row)
    if show_legend:
        cax1 = fig.add_axes([0.92, 0.56, 0.015, 0.30])
        if ims1:
            fig.colorbar(
                ims1[0],
                cax=cax1,
                orientation="vertical",
                label=_metric_label(args.metric_prior),
            )
        else:
            cax1.set_axis_off()

        cax2 = fig.add_axes([0.92, 0.12, 0.015, 0.30])
        if ims2:
            fig.colorbar(
                ims2[0],
                cax=cax2,
                orientation="vertical",
                label=_metric_label(args.metric_cons),
            )
        else:
            cax2.set_axis_off()

    if show_title:
        default = (
            "Supplement S7 • Physics sensitivity\n"
            r"($\epsilon_{\mathrm{prior}}$ and "
            r"$\epsilon_{\mathrm{cons}}$ vs. "
            r"$\lambda_{\mathrm{prior}}, "
            r"\lambda_{\mathrm{cons}}$)"
        )
        ttl = utils.resolve_title(default=default, title=args.title)
        fig.suptitle(ttl, fontsize=11, fontweight="bold")

    png = out.with_suffix(".png")
    pdf = out.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print(f"[OK] figs -> {png} | {pdf}")


def main(argv: List[str] | None = None) -> None:
    plot_physics_sensitivity_main(argv)


if __name__ == "__main__":
    main()