# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""fusionlab.utils.transfer.xfer_metrics

Metric helpers for transferability plots.

This module focuses on:
- interval metrics (coverage, sharpness)
- point metrics (MAE/MSE/RMSE/R2)
- Pareto frontier extraction (coverage vs sharpness)
- reshaping xfer_results.csv horizon columns
- retention computation vs a reference baseline

All functions are independent from plotting libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd


# ----------------------------------------------------------
# Basic scores
# ----------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return float("nan")
    return float(np.mean((y_true[m] - y_pred[m]) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    v = mse(y_true, y_pred)
    return float(np.sqrt(v))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return float("nan")

    yt = y_true[m]
    yp = y_pred[m]

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


# ----------------------------------------------------------
# Interval metrics
# ----------------------------------------------------------

def interval_width(
    df: pd.DataFrame,
    *,
    lo: str = "subsidence_q10",
    hi: str = "subsidence_q90",
) -> pd.Series:
    """Interval width (sharpness proxy)."""
    a = pd.to_numeric(df[hi], errors="coerce")
    b = pd.to_numeric(df[lo], errors="coerce")
    return a - b


def interval_coverage(
    df: pd.DataFrame,
    *,
    y: str = "subsidence_actual",
    lo: str = "subsidence_q10",
    hi: str = "subsidence_q90",
) -> float:
    """Fraction of actuals that fall inside [lo, hi]."""
    yy = pd.to_numeric(df[y], errors="coerce")
    lo_v = pd.to_numeric(df[lo], errors="coerce")
    hi_v = pd.to_numeric(df[hi], errors="coerce")

    m = yy.notna() & lo_v.notna() & hi_v.notna()
    if not bool(m.any()):
        return float("nan")

    inside = (yy[m] >= lo_v[m]) & (yy[m] <= hi_v[m])
    return float(inside.mean())


def interval_sharpness(
    df: pd.DataFrame,
    *,
    lo: str = "subsidence_q10",
    hi: str = "subsidence_q90",
) -> float:
    """Mean interval width (mm)."""
    w = interval_width(df, lo=lo, hi=hi)
    w = pd.to_numeric(w, errors="coerce")
    if not bool(w.notna().any()):
        return float("nan")
    return float(w.dropna().mean())


@dataclass(frozen=True)
class EvalSummary:
    """Compact summary of eval csv metrics."""

    mae: float
    mse: float
    rmse: float
    r2: float
    coverage80: float
    sharpness80: float
    n: int


def summarize_eval_df(
    df: pd.DataFrame,
    *,
    y: str = "subsidence_actual",
    pred: str = "subsidence_q50",
    lo: str = "subsidence_q10",
    hi: str = "subsidence_q90",
) -> EvalSummary:
    """Compute point + interval metrics from one eval df."""
    yy = pd.to_numeric(df[y], errors="coerce").to_numpy()
    pp = pd.to_numeric(df[pred], errors="coerce").to_numpy()

    m = np.isfinite(yy) & np.isfinite(pp)
    n = int(np.sum(m))

    out_mae = mae(yy, pp)
    out_mse = mse(yy, pp)
    out_rmse = rmse(yy, pp)
    out_r2 = r2_score(yy, pp)

    cov = interval_coverage(df, y=y, lo=lo, hi=hi)
    shp = interval_sharpness(df, lo=lo, hi=hi)

    return EvalSummary(
        mae=out_mae,
        mse=out_mse,
        rmse=out_rmse,
        r2=out_r2,
        coverage80=cov,
        sharpness80=shp,
        n=n,
    )


def summarize_by(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    y: str = "subsidence_actual",
    pred: str = "subsidence_q50",
    lo: str = "subsidence_q10",
    hi: str = "subsidence_q90",
) -> pd.DataFrame:
    """Groupwise summary for eval csv tables."""
    rows: List[Dict[str, object]] = []

    for keys, g in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        s = summarize_eval_df(g, y=y, pred=pred, lo=lo, hi=hi)
        row: Dict[str, object] = {}
        for c, v in zip(group_cols, keys):
            row[c] = v
        row.update(
            {
                "mae": s.mae,
                "mse": s.mse,
                "rmse": s.rmse,
                "r2": s.r2,
                "coverage80": s.coverage80,
                "sharpness80": s.sharpness80,
                "n": s.n,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------
# Pareto frontier
# ----------------------------------------------------------

def pareto_frontier_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> np.ndarray:
    """Return a boolean mask for non-dominated points.

    A point i is dominated if there exists j such that:
      - x_j <= x_i (if minimize_x) or x_j >= x_i
      - y_j >= y_i (if maximize_y) or y_j <= y_i
      - and at least one strict.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)

    m = np.zeros_like(ok, dtype=bool)
    idx = np.where(ok)[0]

    if idx.size == 0:
        return m

    xv = x[idx]
    yv = y[idx]

    # Normalize to: minimize both.
    if not minimize_x:
        xv = -xv
    if maximize_y:
        yv = -yv

    keep = np.ones(idx.size, dtype=bool)

    for i in range(idx.size):
        if not keep[i]:
            continue
        dx = xv <= xv[i]
        dy = yv <= yv[i]
        strict = (xv < xv[i]) | (yv < yv[i])
        dom = dx & dy & strict
        if np.any(dom):
            keep[i] = False

    m[idx[keep]] = True
    return m


def mark_pareto_frontier(
    df: pd.DataFrame,
    *,
    x: str = "sharpness80",
    y: str = "coverage80",
    out_col: str = "is_pareto",
) -> pd.DataFrame:
    """Add a boolean column marking Pareto-optimal rows."""
    out = df.copy()
    mask = pareto_frontier_mask(
        pd.to_numeric(out[x], errors="coerce").to_numpy(),
        pd.to_numeric(out[y], errors="coerce").to_numpy(),
        minimize_x=True,
        maximize_y=True,
    )
    out[out_col] = mask
    return out


# ----------------------------------------------------------
# xfer_results.csv reshaping helpers
# ----------------------------------------------------------

def melt_horizon_cols(
    df: pd.DataFrame,
    *,
    prefix: str,
    id_cols: Sequence[str],
    horizon_name: str = "horizon",
    value_name: str = "value",
) -> pd.DataFrame:
    """Melt wide horizon cols like per_horizon_r2.H1."""
    cols = [
        c for c in df.columns
        if str(c).startswith(prefix)
    ]
    if not cols:
        cols_out = list(id_cols)
        cols_out += [horizon_name, value_name]
        return pd.DataFrame(columns=cols_out)

    long = df.melt(
        id_vars=list(id_cols),
        value_vars=cols,
        var_name="_var",
        value_name=value_name,
    )
    long[horizon_name] = long["_var"].astype(str).str.split(
        ".", n=1, expand=True
    )[1]
    long = long.drop(columns=["_var"])
    return long


def compute_retention(
    df: pd.DataFrame,
    *,
    ref: pd.DataFrame,
    join_cols: Sequence[str],
    metric_cols: Sequence[str],
    mode: str = "ratio",
    suffix_ref: str = "_ref",
) -> pd.DataFrame:
    """Compute retention vs a reference table.

    Parameters
    ----------
    df:
        Candidate rows (e.g. xfer, warm).
    ref:
        Reference rows (e.g. target-trained baseline).
    join_cols:
        Columns used to align df and ref.
    metric_cols:
        Metric columns to compare.
    mode:
        - "ratio": df / ref
        - "delta": df - ref

    Returns
    -------
    pd.DataFrame
        Joined table with new ``ret_<metric>`` columns.
    """
    if mode not in {"ratio", "delta"}:
        raise ValueError("mode must be 'ratio' or 'delta'.")

    left = df.copy()
    right = ref.copy()

    keep = list(join_cols) + list(metric_cols)
    right = right[keep].copy()

    right = right.rename(
        columns={c: f"{c}{suffix_ref}" for c in metric_cols}
    )

    out = left.merge(right, on=list(join_cols), how="left")

    for c in metric_cols:
        a = pd.to_numeric(out[c], errors="coerce")
        key = f"{c}{suffix_ref}"
        b = pd.to_numeric(out[key], errors="coerce")
        if mode == "ratio":
            out[f"ret_{c}"] = a / b
        else:
            out[f"ret_{c}"] = a - b

    return out
