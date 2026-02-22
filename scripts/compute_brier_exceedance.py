# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# nat.com/compute_brier_exceedance.py
#
r"""
Compute Brier scores for subsidence exceedance events.

Event
-----
    y = 1{ subsidence_actual >= T }

Probability approximation
-------------------------
We approximate P(subsidence >= T) from three quantiles
(q10,q50,q90) by piecewise-linear interpolation of the CDF
at:
    (q10,0.1), (q50,0.5), (q90,0.9)
then:
    p = 1 - F(T)

Inputs
------
A calibrated forecast CSV that contains:
    coord_t,
    subsidence_actual,
    subsidence_q10, subsidence_q50, subsidence_q90

Hold-out note
-------------
Your "TestSet" calibrated CSV can also contain
subsidence_actual (e.g. 2020-2022 hold-out). So we support
three discovery modes:

    --source auto  (default): use TestSet if present, else
                              fall back to Validation.
    --source test           : require TestSet.
    --source val            : require Validation.

Outputs
-------
A tidy CSV under scripts/out/ by default:
    city, years, threshold_mm_per_yr,
    brier_score, n_samples, src_csv

Examples
--------
Auto-discover (test-first):
  python nat.com/compute_brier_exceedance.py \
    --root results \
    --source auto \
    --thresholds 30,50 \
    --years 2020,2021,2022

Force Validation:
  python nat.com/compute_brier_exceedance.py \
    --root results \
    --source val \
    --out brier_scores_val.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts import config as cfg
from scripts import utils


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="brier-exceedance",
        description="Brier scores for exceedance events.",
    )

    p.add_argument(
        "--root",
        type=str,
        default="results",
        help="Root dir to scan for forecast CSVs.",
    )
    p.add_argument(
        "--ns-csv",
        type=str,
        default=None,
        help="Explicit Nansha CSV (overrides scan).",
    )
    p.add_argument(
        "--zh-csv",
        type=str,
        default=None,
        help="Explicit Zhongshan CSV (overrides scan).",
    )

    p.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "test", "val"],
        help="Which split to use (default: auto).",
    )

    p.add_argument(
        "--thresholds",
        type=str,
        default="30,50",
        help="Comma list of thresholds (mm/yr).",
    )
    p.add_argument(
        "--years",
        type=str,
        default="2020,2021,2022",
        help="Comma list of years, or 'all'.",
    )

    p.add_argument(
        "--out",
        "-o",
        type=str,
        default="brier_scores_validation.csv",
        help="Output CSV (relative -> scripts/out/).",
    )
    p.add_argument(
        "--quiet",
        type=str,
        default="false",
        help="Do not print table (true/false).",
    )

    return p.parse_args(argv)


def _parse_float_list(s: str) -> List[float]:
    parts = [p.strip() for p in str(s).split(",")]
    out: List[float] = []
    for p in parts:
        if p:
            out.append(float(p))
    return out


def _parse_years(s: str) -> Optional[List[int]]:
    s = str(s or "").strip().lower()
    if not s or s == "all":
        return None
    parts = [p.strip() for p in s.split(",")]
    out: List[int] = []
    for p in parts:
        if p:
            out.append(int(p))
    return out


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------
def _pick_city_csv(
    root: Path,
    *,
    city_key: str,
    patterns: Sequence[str],
    must_exist: bool = True,
) -> Optional[Path]:
    """
    Pick newest match under root for a given city.

    Notes:
    - utils.find_all() returns newest-first.
    - We prioritize filename containing the city key.
    """
    cands = utils.find_all(root, patterns, must_exist=False)
    ck = str(city_key).strip().lower()

    def _has_city(fp: Path) -> bool:
        return ck in fp.name.lower()

    city_hits = [p for p in cands if _has_city(p)]
    if city_hits:
        return city_hits[0]

    if cands:
        # fallback to newest overall candidate
        return cands[0]

    if must_exist:
        raise FileNotFoundError(
            f"No match under {root} for {patterns}"
        )
    return None


def _discover_inputs(
    root: Path,
    *,
    source: str,
) -> List[Tuple[str, Path]]:
    """
    Discover (city, csv_path) with a test-first policy.

    source:
      - "auto": test if present else val
      - "test": require test
      - "val" : require val
    """
    if source not in {"auto", "test", "val"}:
        raise ValueError(f"Bad source={source!r}")

    test_pats = cfg.PATTERNS.get("forecast_test_csv", ())
    val_pats = cfg.PATTERNS.get("forecast_val_csv", ())

    def _need(pats: Sequence[str]) -> Sequence[str]:
        if not pats:
            raise KeyError("Missing PATTERNS entry.")
        return pats

    items: List[Tuple[str, Path]] = []

    for city, key in [("Nansha", "nansha"),
                      ("Zhongshan", "zhongshan")]:
        if source == "val":
            p = _pick_city_csv(
                root,
                city_key=key,
                patterns=_need(val_pats),
                must_exist=True,
            )
            items.append((city, p))  # type: ignore[arg-type]
            continue

        # source == "test" or "auto" (test-first)
        p_test = _pick_city_csv(
            root,
            city_key=key,
            patterns=_need(test_pats),
            must_exist=(source == "test"),
        )

        if p_test is not None:
            items.append((city, p_test))
            continue

        # auto fallback: validation
        p_val = _pick_city_csv(
            root,
            city_key=key,
            patterns=_need(val_pats),
            must_exist=True,
        )
        items.append((city, p_val))  # type: ignore[arg-type]

    return items


# ---------------------------------------------------------------------
# Schema + probability model
# ---------------------------------------------------------------------
_ALIASES: Dict[str, Tuple[str, ...]] = {
    "coord_t": ("coord_t", "year", "t"),
    "subsidence_actual": ("subsidence_actual", "actual"),
    "subsidence_q10": ("subsidence_q10", "q10", "p10"),
    "subsidence_q50": ("subsidence_q50", "q50", "p50"),
    "subsidence_q90": ("subsidence_q90", "q90", "p90"),
}


def _ensure_needed_cols(df: pd.DataFrame) -> None:
    utils.ensure_columns(df, aliases=_ALIASES)

    req = [
        "coord_t",
        "subsidence_actual",
        "subsidence_q10",
        "subsidence_q50",
        "subsidence_q90",
    ]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")


def exceed_prob_from_quantiles(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """
    Vectorized piecewise-linear CDF interpolation.

    Returns p = P(s >= threshold) for each row.
    """
    qs = np.stack([q10, q50, q90], axis=1).astype(float)
    ps0 = np.array([0.1, 0.5, 0.9], dtype=float)

    ok = np.isfinite(qs).all(axis=1)
    out = np.full(qs.shape[0], np.nan, dtype=float)
    if not np.any(ok):
        return out

    qs_ok = qs[ok]
    order = np.argsort(qs_ok, axis=1)
    qs_s = np.take_along_axis(qs_ok, order, axis=1)
    ps_s = np.take(ps0, order)

    T = float(threshold)
    F = np.empty(qs_s.shape[0], dtype=float)

    lo = T <= qs_s[:, 0]
    hi = T >= qs_s[:, 2]
    mid = ~(lo | hi)

    F[lo] = ps_s[lo, 0]
    F[hi] = ps_s[hi, 2]

    if np.any(mid):
        q1 = qs_s[mid, 1]
        seg = (T > q1).astype(int)

        ii = np.arange(seg.size)
        x0 = qs_s[mid][ii, seg]
        x1 = qs_s[mid][ii, seg + 1]
        p0 = ps_s[mid][ii, seg]
        p1 = ps_s[mid][ii, seg + 1]

        den = x1 - x0
        same = den == 0.0

        Fm = np.empty(seg.size, dtype=float)
        Fm[same] = p1[same]
        Fm[~same] = p0[~same] + (
            (T - x0[~same]) / den[~same]
        ) * (p1[~same] - p0[~same])

        F[mid] = Fm

    out[ok] = 1.0 - F
    return out


# ---------------------------------------------------------------------
# Brier computation
# ---------------------------------------------------------------------
def compute_brier_scores(
    df: pd.DataFrame,
    *,
    thresholds: Sequence[float],
    years: Optional[Sequence[int]],
) -> List[Dict[str, object]]:
    """
    Compute Brier scores for multiple thresholds.

    Returns rows with:
      threshold_mm_per_yr, brier_score, n_samples
    """
    df = df.copy()
    _ensure_needed_cols(df)

    df["year"] = pd.to_numeric(df["coord_t"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    if years is not None:
        df = df[df["year"].isin(list(years))].copy()

    for c in [
        "subsidence_actual",
        "subsidence_q10",
        "subsidence_q50",
        "subsidence_q90",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            "subsidence_actual",
            "subsidence_q10",
            "subsidence_q50",
            "subsidence_q90",
        ]
    ).copy()

    q10 = df["subsidence_q10"].to_numpy(dtype=float)
    q50 = df["subsidence_q50"].to_numpy(dtype=float)
    q90 = df["subsidence_q90"].to_numpy(dtype=float)
    s = df["subsidence_actual"].to_numpy(dtype=float)

    rows: List[Dict[str, object]] = []

    for T in thresholds:
        p = exceed_prob_from_quantiles(
            q10,
            q50,
            q90,
            threshold=float(T),
        )
        y = (s >= float(T)).astype(float)

        mask = np.isfinite(p) & np.isfinite(y)
        n = int(mask.sum())
        if n:
            bs = float(np.mean((p[mask] - y[mask]) ** 2))
        else:
            bs = np.nan

        rows.append(
            {
                "threshold_mm_per_yr": float(T),
                "brier_score": bs,
                "n_samples": n,
            }
        )

    return rows


# ---------------------------------------------------------------------
# Main (API)
# ---------------------------------------------------------------------
def brier_exceedance_main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)

    thresholds = _parse_float_list(args.thresholds)
    years = _parse_years(args.years)

    root = utils.as_path(args.root)

    # Explicit inputs override scanning.
    ns = utils.as_path(args.ns_csv) if args.ns_csv else None
    zh = utils.as_path(args.zh_csv) if args.zh_csv else None

    if ns is not None and zh is not None:
        inputs = [("Nansha", ns), ("Zhongshan", zh)]
    else:
        inputs = _discover_inputs(root, source=args.source)

    results: List[Dict[str, object]] = []

    for city, path in inputs:
        if path is None or not path.exists():
            raise FileNotFoundError(str(path) if path else city)

        df = pd.read_csv(path)

        rows = compute_brier_scores(
            df,
            thresholds=thresholds,
            years=years,
        )

        for r in rows:
            r["city"] = utils.canonical_city(city)
            if years is None:
                r["years"] = "all"
            else:
                r["years"] = ",".join(str(y) for y in years)
            r["src_csv"] = str(path)
            r["source"] = str(args.source)
            results.append(r)

    out_df = pd.DataFrame(results)[
        [
            "city",
            "source",
            "years",
            "threshold_mm_per_yr",
            "brier_score",
            "n_samples",
            "src_csv",
        ]
    ]

    out_path = utils.resolve_out_out(args.out)
    utils.ensure_dir(out_path.parent)
    out_df.to_csv(out_path, index=False)

    quiet = utils.str_to_bool(args.quiet, default=False)
    if not quiet:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(out_df.to_string(index=False))
        print(f"\n[OK] brier -> {out_path}")


def main(argv: List[str] | None = None) -> None:
    brier_exceedance_main(argv)


if __name__ == "__main__":
    main()
