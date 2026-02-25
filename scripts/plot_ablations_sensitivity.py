# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# scripts/plot_ablations_sensitivity.py
#
r"""
Supplement S6 • Extended ablations & sensitivities

Updated v3.2 behavior:
- Prefer ablation_record.updated*.jsonl when present.
- Dedupe (timestamp, city): keep best record (updated wins).
- Normalize pde_mode to buckets: none vs both
  (off/none -> none, else -> both).
- Optional explicit --input (csv/json/jsonl) and --models filtering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from . import utils

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
        help="Root to scan for ablation_record*.jsonl",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        action="append",
        help=(
            "Explicit ablation file/table (repeatable). "
            "Supports .jsonl/.json/.csv."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output dir override.",
    )

    p.add_argument("--font", type=int, default=cfg.PAPER_FONT)
    p.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)

    p.add_argument(
        "--bar-metric",
        type=str,
        default="mae",
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
        default="sharpness80",
        choices=[
            "mae",
            "mse",
            "r2",
            "coverage80",
            "sharpness80",
            "epsilon_prior",
            "epsilon_cons",
        ],
        help="Metric for bottom-row heatmaps.",
    )

    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma list of model names to keep.",
    )

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
        help="Hide shared heatmap colorbar (true/false).",
    )

    p.add_argument(
        "--cities",
        type=str,
        default="",
        help="Comma list of cities (overrides city-a/b).",
    )
    p.add_argument(
        "--city-a",
        type=str,
        default="Nansha",
        help="First city name.",
    )
    p.add_argument(
        "--city-b",
        type=str,
        default="Zhongshan",
        help="Second city name.",
    )

    utils.add_plot_text_args(
        p,
        default_out="supp_fig_S6_ablations",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------
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
        if "_src" not in df.columns:
            df["_src"] = str(path)
        return df

    if suf == ".jsonl":
        rows = _read_jsonl(path)
        df = pd.DataFrame(rows)
        if "_src" not in df.columns:
            df["_src"] = str(path)
        return df

    if suf == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            obj = None

        if isinstance(obj, list):
            df = pd.DataFrame(obj)
            if "_src" not in df.columns:
                df["_src"] = str(path)
            return df

        if isinstance(obj, dict):
            df = pd.DataFrame([obj])
            if "_src" not in df.columns:
                df["_src"] = str(path)
            return df

    return pd.DataFrame([])


def _scan_records(root: Path) -> pd.DataFrame:
    files = utils.find_all(
        root,
        cfg.PATTERNS.get("ablation_record_jsonl", ()),
    )

    blocks: List[pd.DataFrame] = []
    for fp in files:
        df = _load_one_input(fp)
        if not df.empty:
            blocks.append(df)

    if not blocks:
        return pd.DataFrame([])
    return pd.concat(blocks, ignore_index=True)


def _load_records(args: argparse.Namespace) -> pd.DataFrame:
    if args.input:
        blocks: List[pd.DataFrame] = []
        for s in args.input:
            p = utils.as_path(s)
            if p.exists():
                df = _load_one_input(p)
                if not df.empty:
                    blocks.append(df)
        if not blocks:
            return pd.DataFrame([])
        return pd.concat(blocks, ignore_index=True)

    root = utils.as_path(args.root)
    return _scan_records(root)


# ---------------------------------------------------------------------
# Canonicalization / precedence
# ---------------------------------------------------------------------
def _canon_pde_mode(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"none", "off", "no", "0", "false"}:
        return "none"
    if s in {"both", "on", "1", "true"}:
        return "both"
    # treat any physics-on mode as "both" bucket
    if s in {"consolidation", "gw_flow"}:
        return "both"
    return s


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    aliases = {
        "timestamp": ("timestamp", "ts"),
        "city": ("city", "City"),
        "model": ("model", "Model"),
        "pde_mode": ("pde_mode", "pde"),
        "lambda_prior": ("lambda_prior", "lambda_p"),
        "lambda_cons": ("lambda_cons", "lambda_c"),
        "coverage80": ("coverage80", "coverage8"),
        "sharpness80": ("sharpness80", "sharpness"),
    }
    utils.ensure_columns(df, aliases=aliases)

    if "city" in df.columns:
        df["city"] = df["city"].astype(str).map(
            utils.canonical_city
        )

    if "pde_mode" in df.columns:
        df["pde_mode"] = df["pde_mode"].map(_canon_pde_mode)
    else:
        df["pde_mode"] = "both"

    # stable bucket used by S6 layout:
    #   - "none": physics off
    #   - "both": any physics on mode
    df["pde_bucket"] = np.where(
        df["pde_mode"].astype(str).eq("none"),
        "none",
        "both",
    )

    num_cols = [
        "lambda_prior",
        "lambda_cons",
        "r2",
        "mae",
        "mse",
        "coverage80",
        "sharpness80",
        "epsilon_prior",
        "epsilon_cons",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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

    for c in ["mae", "sharpness80"]:
        if c in df.columns:
            df.loc[mask, c] = df.loc[mask, c] * 1000.0

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

    return int(s)


def _dedupe_prefer_best(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        return df
    if "city" not in df.columns:
        df["city"] = ""

    df = df.copy()
    df["_score"] = df.apply(_record_score, axis=1)

    df = df.sort_values(
        by=["timestamp", "city", "_score"],
        ascending=[True, True, False],
    )
    df = df.drop_duplicates(
        subset=["timestamp", "city"],
        keep="first",
    ).copy()

    return df.drop(columns=["_score"], errors="ignore")


def _filter_models(df: pd.DataFrame, models: str) -> pd.DataFrame:
    s = str(models or "").strip()
    if not s or "model" not in df.columns:
        return df

    keep = [m.strip() for m in s.split(",") if m.strip()]
    if not keep:
        return df

    return df.loc[df["model"].astype(str).isin(keep)].copy()


def _resolve_cities(args: argparse.Namespace) -> List[str]:
    raw = str(args.cities or "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        out = [utils.canonical_city(p) for p in parts]
        return out

    a = utils.canonical_city(args.city_a)
    b = utils.canonical_city(args.city_b)
    return [a, b]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _metric_label(name: str) -> str:
    k = str(name).strip()
    if k in cfg.PHYS_LABELS:
        return cfg.PHYS_LABELS[k]

    meta = cfg.PLOT_METRIC_META.get(k, None)
    if isinstance(meta, dict):
        yl = str(meta.get("ylabel", k))
        unit = str(meta.get("unit", "") or "")
        return yl.format(unit=unit)

    kl = k.lower()
    if kl == "coverage80":
        return "Coverage (80%)"
    if kl == "sharpness80":
        return "Sharpness (80%)"
    if kl == "r2":
        return "R\u00b2"
    return k


def _metric_fmt(metric: str) -> str:
    meta = cfg.PLOT_METRIC_META.get(str(metric), None)
    if isinstance(meta, dict):
        return str(meta.get("fmt", "{:.3g}"))
    return "{:.3g}"


def _axes_cleanup(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _best_ij(arr: np.ndarray, *, metric: str) -> Optional[Tuple[int, int]]:
    try:
        a = np.asarray(arr, dtype=float)
        if not np.isfinite(a).any():
            return None

        if str(metric).lower() in _LOWER_IS_BETTER:
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
    sub = df.loc[df["city"].astype(str).eq(str(city))].copy()
    if sub.empty or metric not in sub.columns:
        ax.set_axis_off()
        return

    keep = sub.loc[sub["pde_bucket"].isin(["none", "both"])].copy()
    if keep.empty:
        ax.set_axis_off()
        return

    grp = (
        keep.groupby(["pde_bucket", "lambda_prior"], dropna=False)[metric]
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

    modes = ["none", "both"]
    base = np.arange(len(xs), dtype=float)
    width = 0.8 / 2.0
    offset0 = 0.5

    fmt = _metric_fmt(metric)

    for i, m in enumerate(modes):
        y = [
            grp.loc[
                (grp["pde_bucket"] == m)
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
                        fmt.format(float(yi)),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    ax.set_xticks(base)

    if show_ticks:
        ax.set_xticklabels([f"{x:.2g}" for x in xs])
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

    if show_legend:
        ax.legend(title="physics", frameon=False)

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
    sub = df.loc[df["city"].astype(str).eq(str(city))].copy()
    sub = sub.loc[sub["pde_bucket"].eq("both")].copy()

    if sub.empty or metric not in sub.columns:
        ax.set_axis_off()
        return None

    sub = sub.dropna(subset=["lambda_cons", "lambda_prior"])
    if sub.empty:
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

    im = ax.imshow(piv.values, aspect="auto", cmap=cmap)

    ax.set_xticks(range(piv.shape[1]))
    ax.set_yticks(range(piv.shape[0]))

    if show_ticks:
        ax.set_xticklabels([f"{c:.2g}" for c in piv.columns])
        ax.set_yticklabels([f"{r:.2g}" for r in piv.index])
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


def _resolve_out(*, out: str, out_dir: Optional[str]) -> Path:
    if out_dir:
        base = Path(out_dir).expanduser()
        return (base / Path(out).expanduser()).resolve()
    return utils.resolve_fig_out(out)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def plot_ablations_sensivity_main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)

    utils.set_paper_style(fontsize=int(args.font), dpi=int(args.dpi))

    show_legend = utils.str_to_bool(args.show_legend, default=True)
    show_labels = utils.str_to_bool(args.show_labels, default=True)
    show_ticks = utils.str_to_bool(args.show_ticklabels, default=True)
    show_title = utils.str_to_bool(args.show_title, default=True)
    show_pan_t = utils.str_to_bool(
        args.show_panel_titles,
        default=True,
    )

    annotate = not utils.str_to_bool(args.mute_values, default=False)
    show_cbar = not utils.str_to_bool(args.no_colorbar, default=False)

    df = _load_records(args)
    if df.empty:
        root = utils.as_path(args.root)
        raise SystemExit(
            "No ablation_record*.jsonl found under:\n"
            f"  {root.resolve()}\n"
            "Run ablations with the logger enabled first."
        )

    df = _canon_cols(df)
    df = _harmonize_units(df)
    df = _dedupe_prefer_best(df)
    df = _filter_models(df, args.models)

    cities = _resolve_cities(args)
    if not cities:
        raise SystemExit("No cities selected.")

    out = _resolve_out(out=args.out, out_dir=args.out_dir)
    utils.ensure_dir(out.parent)

    tidy_csv = out.parent / "tableS6_ablations_used.csv"
    df.to_csv(tidy_csv, index=False)
    print(f"[OK] table -> {tidy_csv}")

    if len(cities) == 1:
        cities = [cities[0], cities[0]]

    city_a, city_b = cities[0], cities[1]

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

    # Top row: bars (metric vs lambda_prior)
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

    # Bottom row: heatmaps (physics-on bucket)
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
        default = "Supplement S6 • Extended ablations & sensitivities"
        ttl = utils.resolve_title(default=default, title=args.title)
        fig.suptitle(ttl, fontsize=11, fontweight="bold")

    png = out.with_suffix(".png")
    pdf = out.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[OK] figs -> {png} | {pdf}")


def main(argv: List[str] | None = None) -> None:
    plot_ablations_sensivity_main(argv)


if __name__ == "__main__":
    main()