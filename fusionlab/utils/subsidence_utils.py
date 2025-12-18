
from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd


def detect_subsidence_mode(
    df: pd.DataFrame,
    *,
    rate_col: str = "subsidence",
    cum_col: str = "subsidence_cum",
    time_col: str = "year",
    group_cols: Iterable[str] = ("longitude", "latitude", "city"),
    tol_rel: float = 0.05,
    min_points: int = 3,
    max_groups: int = 200,
    random_state: int = 42,
) -> Dict:
    """
    Infer whether subsidence columns represent 'rate', 'cumulative',
    or an inconsistent pair.

    Strategy
    --------
    If both columns exist, check whether:
        Δcum(t_i) ≈ rate(t_i) * Δt_i
    per group (lon/lat[/city]) over i>=1. This is baseline-invariant.

    Returns
    -------
    dict with keys:
        mode: 'cumulative', 'rate', 'pair-consistent', 'unknown'
        details: diagnostics (errors, counts, etc.)
    """
    cols = set(df.columns)
    has_rate = rate_col in cols
    has_cum = cum_col in cols

    if has_cum and not has_rate:
        return {"mode": "cumulative", "details": {"reason": "cum_col present only"}}
    if has_rate and not has_cum:
        return {"mode": "rate", "details": {"reason": "rate_col present only"}}
    if not (has_rate and has_cum):
        return {"mode": "unknown", "details": {"reason": "neither column present"}}

    gcols = [c for c in group_cols if c in cols]
    if not gcols:
        return {
            "mode": "unknown",
            "details": {"reason": "no valid group_cols found in df"},
        }

    # sample groups for speed
    groups = df.groupby(gcols, sort=False)
    keys = list(groups.groups.keys())
    if not keys:
        return {"mode": "unknown", "details": {"reason": "no groups"}}

    rng = np.random.default_rng(random_state)
    if len(keys) > max_groups:
        keys = [keys[i] for i in rng.choice(len(keys), size=max_groups, replace=False)]

    rel_errs = []
    used_groups = 0
    used_points = 0

    for k in keys:
        g = groups.get_group(k).sort_values(time_col)
        if len(g) < min_points:
            continue

        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(float)
        r = pd.to_numeric(g[rate_col], errors="coerce").to_numpy(float)
        c = pd.to_numeric(g[cum_col], errors="coerce").to_numpy(float)

        m = np.isfinite(t) & np.isfinite(r) & np.isfinite(c)
        t, r, c = t[m], r[m], c[m]
        if t.size < min_points:
            continue

        dt = np.diff(t)
        dc = np.diff(c)
        r_next = r[1:]

        m2 = np.isfinite(dt) & np.isfinite(dc) & np.isfinite(r_next) & (dt != 0)
        if m2.sum() < (min_points - 1):
            continue

        dt, dc, r_next = dt[m2], dc[m2], r_next[m2]

        pred_dc = r_next * dt
        denom = np.maximum(np.mean(np.abs(pred_dc)), 1e-12)
        rel = np.mean(np.abs(dc - pred_dc)) / denom

        rel_errs.append(rel)
        used_groups += 1
        used_points += int(m2.sum())

    if used_groups == 0:
        return {"mode": "unknown", "details": {"reason": "insufficient data per group"}}

    med = float(np.median(rel_errs))
    out = {
        "median_rel_error": med,
        "mean_rel_error": float(np.mean(rel_errs)),
        "n_groups": used_groups,
        "n_points": used_points,
        "tol_rel": tol_rel,
    }

    if med <= tol_rel:
        return {"mode": "pair-consistent", "details": out}
    return {"mode": "unknown", "details": out}

def rate_to_cumulative(
    df: pd.DataFrame,
    *,
    rate_col: str = "subsidence",
    cum_col: str = "subsidence_cum",
    time_col: str = "year",
    group_cols: Iterable[str] = ("longitude", "latitude", "city"),
    initial: str = "first_equals_rate_dt",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Build cumulative displacement from a rate series.

    Assumption
    ----------
    rate(t_i) represents the rate over the interval (t_{i-1}, t_i].
    Then:
        cum(t_i) = cum(t_{i-1}) + rate(t_i) * dt_i

    Parameters
    ----------
    initial:
        - 'zero': cum at first time is 0
        - 'first_equals_rate_dt': cum(t0) = rate(t0) * dt_ref
          where dt_ref is median dt in that group (fallback 1).

    Returns
    -------
    DataFrame with cum_col added/overwritten.
    """
    out = df if inplace else df.copy()
    cols = set(out.columns)
    gcols = [c for c in group_cols if c in cols]
    if not gcols:
        raise ValueError("No valid group_cols found in df.")

    out[cum_col] = np.nan

    for _, gidx in out.groupby(gcols, sort=False).groups.items():
        g = out.loc[gidx].sort_values(time_col)
        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(float)
        r = pd.to_numeric(g[rate_col], errors="coerce").to_numpy(float)

        m = np.isfinite(t) & np.isfinite(r)
        t, r = t[m], r[m]
        if t.size == 0:
            continue

        dt = np.diff(t)
        dt_ref = float(np.median(dt[np.isfinite(dt) & (dt != 0)])) if t.size > 1 else 1.0
        if not np.isfinite(dt_ref) or dt_ref == 0:
            dt_ref = 1.0

        c = np.zeros_like(r, dtype=float)
        if initial == "zero":
            c[0] = 0.0
        elif initial == "first_equals_rate_dt":
            c[0] = r[0] * dt_ref
        else:
            raise ValueError("initial must be 'zero' or 'first_equals_rate_dt'.")

        if r.size > 1:
            dt_i = np.diff(t)
            dt_i = np.where((~np.isfinite(dt_i)) | (dt_i == 0), dt_ref, dt_i)
            c[1:] = c[0] + np.cumsum(r[1:] * dt_i)

        # write back aligned to original rows in this group
        out.loc[g.index[m], cum_col] = c

    return out

def cumulative_to_rate(
    df: pd.DataFrame,
    *,
    cum_col: str = "subsidence_cum",
    rate_col: str = "subsidence",
    time_col: str = "year",
    group_cols: Iterable[str] = ("longitude", "latitude", "city"),
    first: str = "cum_over_dtref",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Recover a rate series from cumulative displacement.

    rate(t_i) = (cum(t_i) - cum(t_{i-1})) / dt_i  for i>=1

    first:
        - 'nan': first rate is NaN
        - 'cum_over_dtref': rate(t0) = cum(t0)/dt_ref (dt_ref median dt)

    Returns
    -------
    DataFrame with rate_col added/overwritten.
    """
    out = df if inplace else df.copy()
    cols = set(out.columns)
    gcols = [c for c in group_cols if c in cols]
    if not gcols:
        raise ValueError("No valid group_cols found in df.")

    out[rate_col] = np.nan

    for _, gidx in out.groupby(gcols, sort=False).groups.items():
        g = out.loc[gidx].sort_values(time_col)
        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(float)
        c = pd.to_numeric(g[cum_col], errors="coerce").to_numpy(float)

        m = np.isfinite(t) & np.isfinite(c)
        t, c = t[m], c[m]
        if t.size == 0:
            continue

        dt = np.diff(t)
        dt_ref = float(np.median(dt[np.isfinite(dt) & (dt != 0)])) if t.size > 1 else 1.0
        if not np.isfinite(dt_ref) or dt_ref == 0:
            dt_ref = 1.0

        r = np.zeros_like(c, dtype=float)

        if first == "nan":
            r[0] = np.nan
        elif first == "cum_over_dtref":
            r[0] = c[0] / dt_ref
        else:
            raise ValueError("first must be 'nan' or 'cum_over_dtref'.")

        if c.size > 1:
            dt_i = np.diff(t)
            dt_i = np.where((~np.isfinite(dt_i)) | (dt_i == 0), dt_ref, dt_i)
            r[1:] = np.diff(c) / dt_i

        out.loc[g.index[m], rate_col] = r

    return out
