# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""fusionlab.utils.transfer.xfer_risk

Risk-oriented helpers for transferability plots.

Key idea
--------
Decision makers often act on thresholds:
  "Will subsidence exceed T mm/yr?"

With quantiles (q10, q50, q90), we approximate the CDF
piecewise-linearly at:
  (q10,0.1), (q50,0.5), (q90,0.9)
then return exceedance probability:
  p = 1 - F(T)

This is the same logic used in your paper scripts, but
exposed here as reusable utilities.
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
# Exceedance probability (from quantiles)
# ----------------------------------------------------------

def exceed_prob_from_quantiles(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Approximate P(s >= threshold) for each row.

    Notes
    -----
    - Quantiles can be non-monotone due to noise.
      We sort them row-wise before interpolation.
    - Tail behavior is "flat":
        T <= qmin -> F(T)=pmin
        T >= qmax -> F(T)=pmax
      so the exceedance is clamped to [0.1,0.9].

    Parameters
    ----------
    q10, q50, q90:
        Arrays of quantiles.
    threshold:
        Event threshold (same unit as quantiles).

    Returns
    -------
    np.ndarray
        Exceedance probabilities.
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


# ----------------------------------------------------------
# Brier + reliability
# ----------------------------------------------------------

def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    if not np.any(m):
        return float("nan")
    return float(np.mean((p[m] - y[m]) ** 2))


@dataclass(frozen=True)
class BrierRow:
    threshold: float
    brier: float
    n: int


def brier_from_eval_df(
    df: pd.DataFrame,
    *,
    threshold: float,
    y: str = "subsidence_actual",
    q10: str = "subsidence_q10",
    q50: str = "subsidence_q50",
    q90: str = "subsidence_q90",
) -> BrierRow:
    """Compute Brier score from an eval csv table."""
    yy = pd.to_numeric(df[y], errors="coerce").to_numpy(float)
    q10v = pd.to_numeric(
        df[q10], errors="coerce"
    ).to_numpy(float)
    q50v = pd.to_numeric(
        df[q50], errors="coerce"
    ).to_numpy(float)
    q90v = pd.to_numeric(
        df[q90], errors="coerce"
    ).to_numpy(float)

    p = exceed_prob_from_quantiles(
        q10v,
        q50v,
        q90v,
        threshold=float(threshold),
    )
    yy_bin = (yy >= float(threshold)).astype(float)

    m = np.isfinite(p) & np.isfinite(yy_bin)
    n = int(np.sum(m))
    return BrierRow(
        threshold=float(threshold),
        brier=brier_score(p, yy_bin),
        n=n,
    )


def reliability_bins(
    p: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute reliability table for a diagram.

    Returns columns:
      bin_lo, bin_hi, p_mean, y_rate, n
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(p) & np.isfinite(y)
    p = p[m]
    y = y[m]

    if p.size == 0:
        return pd.DataFrame(
            columns=[
                "bin_lo",
                "bin_hi",
                "p_mean",
                "y_rate",
                "n",
            ]
        )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(
        np.digitize(p, edges) - 1,
        0,
        n_bins - 1,
    )

    rows: List[Dict[str, float]] = []

    for b in range(n_bins):
        mm = idx == b
        n = int(np.sum(mm))
        if n == 0:
            continue
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "p_mean": float(np.mean(p[mm])),
                "y_rate": float(np.mean(y[mm])),
                "n": float(n),
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------
# Hotspot / top-K overlap
# ----------------------------------------------------------

def rank_locations(
    df: pd.DataFrame,
    *,
    value_col: str = "subsidence_q50",
    x_col: str = "coord_x",
    y_col: str = "coord_y",
    agg: str = "mean",
) -> pd.DataFrame:
    """Rank locations by an aggregated risk value."""
    val = pd.to_numeric(df[value_col], errors="coerce")

    g = df.assign(_v=val).groupby(
        [x_col, y_col], dropna=False
    )
    if agg == "mean":
        s = g["_v"].mean()
    elif agg == "median":
        s = g["_v"].median()
    else:
        raise ValueError("agg must be mean or median")

    out = s.reset_index().rename(columns={"_v": "value"})
    out = out.sort_values("value", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def topk_set(
    ranked: pd.DataFrame,
    *,
    k: int,
    x_col: str = "coord_x",
    y_col: str = "coord_y",
) -> set:
    k = int(k)
    if k <= 0:
        return set()
    sub = ranked.head(k)
    return set(zip(sub[x_col].tolist(), sub[y_col].tolist()))


def jaccard_topk(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    inter = len(a & b)
    uni = len(a | b)
    if uni == 0:
        return float("nan")
    return float(inter / uni)


def spearman_rank_corr(
    r1: pd.DataFrame,
    r2: pd.DataFrame,
    *,
    x_col: str = "coord_x",
    y_col: str = "coord_y",
    rank_col: str = "rank",
) -> float:
    """Spearman correlation between ranks (overlap only)."""
    a = r1[[x_col, y_col, rank_col]].copy()
    b = r2[[x_col, y_col, rank_col]].copy()

    m = a.merge(b, on=[x_col, y_col], how="inner")
    if len(m) < 3:
        return float("nan")

    r1v = m[f"{rank_col}_x"].to_numpy(float)
    r2v = m[f"{rank_col}_y"].to_numpy(float)

    r1c = r1v - np.mean(r1v)
    r2c = r2v - np.mean(r2v)

    den = float(np.sqrt(np.sum(r1c**2) * np.sum(r2c**2)))
    if den == 0.0:
        return float("nan")
    return float(np.sum(r1c * r2c) / den)



def _parse_horizon(h: object) -> Optional[int]:
    if h is None:
        return None
    s = str(h).strip().lower()
    if s in {"all", "none", ""}:
        return None
    if s.startswith("h") and len(s) > 1:
        s = s[1:]
    try:
        i = int(float(s))
    except Exception:
        return None
    return i if i > 0 else None


def prepare_hotspot_ranks(
    df: pd.DataFrame,
    *,
    score: str = "q50",
    threshold: float = 50.0,
    horizon: object = "H3",
    time_col: str = "coord_t",
    step_col: str = "forecast_step",
    x_col: str = "coord_x",
    y_col: str = "coord_y",
    agg: str = "median",
) -> pd.DataFrame:
    """Create per-time ranked hotspots table.

    Returns columns:
      time_col, x_col, y_col, value, rank

    Notes
    -----
    - score='q50' ranks by subsidence_q50.
    - score='exceed' ranks by P(s>=threshold).
    - horizon filters forecast_step (H1->1, ...).
    - Aggregation is applied over duplicates
      within each (time,x,y).
    """
    h = _parse_horizon(horizon)

    d = df.copy()

    if time_col not in d.columns:
        raise KeyError(time_col)
    if x_col not in d.columns:
        raise KeyError(x_col)
    if y_col not in d.columns:
        raise KeyError(y_col)

    if h is not None and step_col in d.columns:
        fs = pd.to_numeric(
            d[step_col], errors="coerce"
        )
        d = d.loc[fs.eq(float(h))].copy()

    if score.strip().lower() == "exceed":
        need = [
            "subsidence_q10",
            "subsidence_q50",
            "subsidence_q90",
        ]
        miss = [c for c in need if c not in d.columns]
        if miss:
            raise KeyError(str(miss))

        q10 = pd.to_numeric(
            d["subsidence_q10"], errors="coerce"
        ).to_numpy(float)
        q50 = pd.to_numeric(
            d["subsidence_q50"], errors="coerce"
        ).to_numpy(float)
        q90 = pd.to_numeric(
            d["subsidence_q90"], errors="coerce"
        ).to_numpy(float)

        v = exceed_prob_from_quantiles(
            q10,
            q50,
            q90,
            threshold=float(threshold),
        )
        d = d.assign(_v=v)
    else:
        v = pd.to_numeric(
            d["subsidence_q50"],
            errors="coerce",
        )
        d = d.assign(_v=v)

    d[time_col] = pd.to_numeric(
        d[time_col], errors="coerce"
    )

    g = d.groupby(
        [time_col, x_col, y_col],
        dropna=False,
    )["_v"]

    if agg == "mean":
        s = g.mean()
    elif agg == "median":
        s = g.median()
    else:
        raise ValueError("agg must be mean/median")

    out = s.reset_index().rename(columns={"_v": "value"})

    out = out.sort_values(
        [time_col, "value"],
        ascending=[True, False],
    )

    r = out.groupby(time_col)["value"].rank(
        method="first",
        ascending=False,
    )

    out["rank"] = r.astype(int)
    return out[[time_col, x_col, y_col, "value", "rank"]]


def hotspot_stability(
    ref: pd.DataFrame,
    other: pd.DataFrame,
    *,
    k: int = 100,
    time_col: str = "coord_t",
    x_col: str = "coord_x",
    y_col: str = "coord_y",
    rank_col: str = "rank",
) -> pd.DataFrame:
    """Compute top-K stability over time.

    Returns per-time rows:
      time, jaccard, spearman, n_common
    """
    k = int(k)
    if k <= 0:
        return pd.DataFrame(
            columns=[
                time_col,
                "jaccard",
                "spearman",
                "n_common",
            ]
        )

    t1 = set(ref[time_col].dropna().unique())
    t2 = set(other[time_col].dropna().unique())
    ts = sorted(t1 & t2)

    rows: List[Dict[str, float]] = []

    for t in ts:
        a = ref.loc[ref[time_col].eq(t)].copy()
        b = other.loc[other[time_col].eq(t)].copy()

        a = a.sort_values(rank_col).head(k)
        b = b.sort_values(rank_col).head(k)

        sa = set(zip(a[x_col], a[y_col]))
        sb = set(zip(b[x_col], b[y_col]))

        jac = jaccard_topk(sa, sb)
        spr = spearman_rank_corr(
            a,
            b,
            x_col=x_col,
            y_col=y_col,
            rank_col=rank_col,
        )

        rows.append(
            {
                time_col: float(t),
                "jaccard": float(jac),
                "spearman": float(spr),
                "n_common": float(len(sa & sb)),
            }
        )

    return pd.DataFrame(rows)



def stability_group_summary(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = (
        "direction",
        "strategy",
    ),
    time_col: str = "coord_t",
    ddof: int = 1,
) -> pd.DataFrame:
    """Summarize stability series (mean ± std).

    Parameters
    ----------
    df:
        Stability series with group identifiers.
    group_cols:
        Columns defining each group.
    time_col:
        Time column name.
    ddof:
        Delta degrees of freedom for std.

    Returns
    -------
    pd.DataFrame
        Columns:
          <group_cols>,
          jaccard_mean, jaccard_std,
          spearman_mean, spearman_std,
          n_years
    """
    if df is None or df.empty:
        cols = list(group_cols) + [
            "jaccard_mean",
            "jaccard_std",
            "spearman_mean",
            "spearman_std",
            "n_years",
        ]
        return pd.DataFrame(columns=cols)

    use = df.copy()
    for c in ("jaccard", "spearman"):
        if c in use.columns:
            use[c] = pd.to_numeric(
                use[c], errors="coerce"
            )

    if time_col in use.columns:
        use[time_col] = pd.to_numeric(
            use[time_col], errors="coerce"
        )

    g = use.groupby(list(group_cols), dropna=False)

    def _std(s: pd.Series) -> float:
        a = s.to_numpy(float)
        a = a[np.isfinite(a)]
        if a.size <= 1:
            return 0.0
        return float(np.std(a, ddof=int(ddof)))

    out = g.agg(
        jaccard_mean=("jaccard", "mean"),
        spearman_mean=("spearman", "mean"),
        n_years=(time_col, "nunique")
        if time_col in use.columns
        else ("jaccard", "size"),
    ).reset_index()

    out["jaccard_std"] = g["jaccard"].apply(_std).values
    out["spearman_std"] = g["spearman"].apply(_std).values

    return out
