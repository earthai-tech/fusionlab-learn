# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# nat.com/summarize_hotspots.py
#
r"""
Summarise hotspot point clouds:
- subsidence level (mm/yr): `value`
- anomaly / delta metric (mm/yr): `metric_value`

Input
-----
Hotspot CSV produced by the Fig.6 spatial script
(e.g., make_figure6_spatial_forecasts.py) with columns:

  city, panel, kind, year, coord_x, coord_y, value,
  hotspot_mode, hotspot_quantile, metric_value,
  baseline_value, threshold

Only these are required:
  city, year, kind, value, metric_value

Output
------
A tidy summary grouped by (city, year, kind) with:

  n_hotspots
  value_min, value_mean, value_max
  metric_min, metric_mean, metric_max
  baseline_min, baseline_mean, baseline_max   (if present)
  threshold_min, threshold_max                (if present)

API conventions
---------------
- Use scripts.utils.resolve_out_out() for tables/artifacts.
- Use cfg.OUT_DIR as default output location.
- Expose a stable main(argv) wrapper.
- CLI has a program name (hyphenated).

Examples
--------
Write to scripts/out/ by default:
  python nat.com/summarize_hotspots.py \
    --hotspot-csv results/figs/fig6_hotspot_points.csv

Explicit output (relative -> scripts/out/):
  python nat.com/summarize_hotspots.py \
    --hotspot-csv results/figs/fig6_hotspot_points.csv \
    --out fig6_hotspot_summary.csv
"""

from __future__ import annotations

import argparse
# from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# from . import config as cfg
from . import utils


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="summarize-hotspots",
        description="Summarise hotspot clouds (stats by city/year/kind).",
    )

    p.add_argument(
        "--hotspot-csv",
        type=str,
        required=True,
        help="Hotspot CSV (points) from Fig.6 script.",
    )

    # API convention:
    # - tables go under scripts/out/ by default.
    # - if --out is relative, it resolves under cfg.OUT_DIR.
    p.add_argument(
        "--out",
        "-o",
        type=str,
        default="fig6_hotspot_summary.csv",
        help="Output CSV name/path (relative -> scripts/out/).",
    )

    # Optional: also print the pretty table
    p.add_argument(
        "--quiet",
        type=str,
        default="false",
        help="Do not print table to console (true/false).",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, req: List[str]) -> None:
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _to_numeric_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _summary_per_group(g: pd.DataFrame) -> pd.Series:
    """
    Compute summary stats for one (city, year, kind) group.
    """
    out: Dict[str, Any] = {
        "n_hotspots": int(len(g)),
        # Hotspot subsidence (mm/yr)
        "value_min": float(np.nanmin(g["value"].to_numpy())),
        "value_max": float(np.nanmax(g["value"].to_numpy())),
        "value_mean": float(np.nanmean(g["value"].to_numpy())),
        # Hotspot anomaly metric (mm/yr)
        "metric_min": float(np.nanmin(g["metric_value"].to_numpy())),
        "metric_max": float(np.nanmax(g["metric_value"].to_numpy())),
        "metric_mean": float(np.nanmean(g["metric_value"].to_numpy())),
    }

    if "baseline_value" in g.columns:
        b = g["baseline_value"].to_numpy(dtype=float)
        out.update(
            {
                "baseline_min": float(np.nanmin(b)),
                "baseline_max": float(np.nanmax(b)),
                "baseline_mean": float(np.nanmean(b)),
            }
        )

    if "threshold" in g.columns:
        # thresholds are typically constant per group,
        # but we keep min/max for robustness.
        t = pd.to_numeric(g["threshold"], errors="coerce")
        out.update(
            {
                "threshold_min": float(np.nanmin(t.to_numpy())),
                "threshold_max": float(np.nanmax(t.to_numpy())),
            }
        )

    return pd.Series(out)


def summarize_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize hotspot points into a group-level table.

    Groups:
      (city, year, kind)

    Returns:
      summary DataFrame with stable column ordering.
    """
    _require_columns(
        df,
        ["city", "year", "kind", "value", "metric_value"],
    )

    # Normalize city names to canonical form.
    df = df.copy()
    df["city"] = df["city"].astype(str).map(utils.canonical_city)

    # Types
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    _to_numeric_inplace(
        df,
        ["value", "metric_value", "baseline_value", "threshold"],
    )

    grouped = df.groupby(["city", "year", "kind"], dropna=False)
    out = grouped.apply(_summary_per_group).reset_index()

    # Stable column ordering.
    base = [
        "city",
        "year",
        "kind",
        "n_hotspots",
        "value_min",
        "value_mean",
        "value_max",
        "metric_min",
        "metric_mean",
        "metric_max",
    ]

    extra = [c for c in out.columns if c not in base]
    return out[base + extra]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def summarize_hotspots_main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)

    src = utils.as_path(args.hotspot_csv)
    if not src.exists():
        raise FileNotFoundError(str(src))

    df = pd.read_csv(src)
    summary = summarize_hotspots(df)

    # Default: write to scripts/out/
    out_path = utils.resolve_out_out(args.out)
    utils.ensure_dir(out_path.parent)

    summary.to_csv(out_path, index=False)

    quiet = utils.str_to_bool(args.quiet, default=False)
    if not quiet:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(summary.to_string(index=False))
        print(f"\n[OK] summary -> {out_path}")


def main(argv: List[str] | None = None) -> None:
    summarize_hotspots_main(argv)


if __name__ == "__main__":
    main()
