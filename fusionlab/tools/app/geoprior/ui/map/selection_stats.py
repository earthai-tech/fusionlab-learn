# geoprior/ui/map/selection_stats.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
selection_stats

Pure (non-UI) helpers for "Selection" analytics.

The UI tab can pass:
- a single selected point (x, y), or
- a group of points [(x, y), ...]
where (x, y) are the dataset's X/Y columns.

The helpers load only needed columns from CSV in chunks,
match points using a quantized tolerance grid, then compute
summary series for subsidence-style variables.

Notes
-----
We assume the dataset has:
- time column: t
- value column: z
- optional quantile columns qXX (e.g., q05, q50, q95)

The matching strategy is designed for gridded / repeated
coordinates. It is robust to small float noise via tol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelSummary:
    n_points: int
    t_min: Optional[float]
    t_max: Optional[float]
    z_min: Optional[float]
    z_max: Optional[float]
    z_mean: Optional[float]


def _quantize(a: np.ndarray, tol: float) -> np.ndarray:
    if tol <= 0.0:
        return a.astype("float64")
    return np.round(a / float(tol)).astype("int64")


def _prep_sel(
    pts: Sequence[tuple[float, float]],
    *,
    tol: float,
) -> pd.DataFrame:
    if not pts:
        return pd.DataFrame(columns=["x", "y", "qx", "qy"])
    x = np.array([p[0] for p in pts], dtype="float64")
    y = np.array([p[1] for p in pts], dtype="float64")
    qx = _quantize(x, tol)
    qy = _quantize(y, tol)
    df = pd.DataFrame({"x": x, "y": y, "qx": qx, "qy": qy})
    return df.drop_duplicates(subset=["qx", "qy"])


def load_series_for_points(
    *,
    path: Path,
    x: str,
    y: str,
    t: str,
    z: str,
    q_cols: Sequence[str],
    pts: Sequence[tuple[float, float]],
    tol: float = 1e-6,
    chunksize: int = 200_000,
    max_rows: int = 2_000_000,
) -> pd.DataFrame:
    """
    Load all (t, z, q*) rows matching selected points.

    Returns
    -------
    DataFrame with columns:
    - x, y, t, z, (q*)
    - _pid : stable point id string "qx:qy"
    """
    if not path.exists():
        return pd.DataFrame()

    if not pts:
        return pd.DataFrame()

    sel = _prep_sel(pts, tol=tol)
    if sel.empty:
        return pd.DataFrame()

    use = [c for c in (x, y, t, z) if c]
    use = list(dict.fromkeys(use))
    for qc in q_cols or []:
        if qc and qc not in use:
            use.append(str(qc))

    parts: list[pd.DataFrame] = []
    n_out = 0

    try:
        it = pd.read_csv(
            path,
            usecols=use,
            chunksize=int(chunksize),
        )
    except Exception:
        return pd.DataFrame()

    for ch in it:
        if x not in ch.columns or y not in ch.columns:
            continue

        # coarse prune: bbox around selection
        xmin = float(sel["x"].min()) - float(tol)
        xmax = float(sel["x"].max()) + float(tol)
        ymin = float(sel["y"].min()) - float(tol)
        ymax = float(sel["y"].max()) + float(tol)

        m0 = (
            (ch[x] >= xmin)
            & (ch[x] <= xmax)
            & (ch[y] >= ymin)
            & (ch[y] <= ymax)
        )
        if not bool(m0.any()):
            continue

        sub = ch.loc[m0].copy()
        sub["_qx"] = _quantize(sub[x].to_numpy(), tol)
        sub["_qy"] = _quantize(sub[y].to_numpy(), tol)

        # inner merge on quantized coords
        hit = sub.merge(
            sel[["qx", "qy"]],
            left_on=["_qx", "_qy"],
            right_on=["qx", "qy"],
            how="inner",
        )
        if hit.empty:
            continue

        hit["_pid"] = (
            hit["_qx"].astype(str) + ":" + hit["_qy"].astype(str)
        )
        cols_out = [c for c in use if c in hit.columns]
        cols_out += ["_pid"]
        parts.append(hit[cols_out])
        n_out += len(hit)
        if n_out >= int(max_rows):
            break

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


def summarize_series(
    df: pd.DataFrame,
    *,
    t: str,
    z: str,
) -> SelSummary:
    if df is None or df.empty:
        return SelSummary(0, None, None, None, None, None)

    tt = df[t] if t in df.columns else None
    zz = df[z] if z in df.columns else None

    t_min = float(tt.min()) if tt is not None else None
    t_max = float(tt.max()) if tt is not None else None

    z_min = float(zz.min()) if zz is not None else None
    z_max = float(zz.max()) if zz is not None else None
    z_mean = float(zz.mean()) if zz is not None else None

    n_pts = 0
    if "_pid" in df.columns:
        n_pts = int(df["_pid"].nunique())
    return SelSummary(n_pts, t_min, t_max, z_min, z_max, z_mean)


def pick_mid_col(
    z: str,
    q_cols: Sequence[tuple[float, str]],
) -> str:
    """Prefer q50 if present, else z."""
    for p, c in q_cols or []:
        if abs(float(p) - 0.50) < 1e-9:
            return str(c)
    return str(z)


def band_cols(
    q_cols: Sequence[tuple[float, str]],
    *,
    band: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Map band label -> (lo, hi) columns.

    band options:
    - "80"  => q10..q90
    - "50"  => q25..q75
    - "90"  => q05..q95
    """
    want = {
        "80": (0.10, 0.90),
        "50": (0.25, 0.75),
        "90": (0.05, 0.95),
    }.get(str(band), (0.10, 0.90))

    lo = hi = None
    for p, c in q_cols or []:
        if abs(float(p) - want[0]) < 1e-9:
            lo = str(c)
        if abs(float(p) - want[1]) < 1e-9:
            hi = str(c)
    return lo, hi


def group_trend(
    df: pd.DataFrame,
    *,
    t: str,
    mid: str,
    agg: str = "median",
) -> pd.DataFrame:
    """
    Aggregate selected points over time.

    Returns DataFrame with columns:
    - t
    - mid (aggregate)
    - p10, p90 across points (optional)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if t not in df.columns or mid not in df.columns:
        return pd.DataFrame()

    g = df.groupby([t], as_index=False)
    if str(agg) == "mean":
        out = g[mid].mean().rename(columns={mid: "mid"})
    else:
        out = g[mid].median().rename(columns={mid: "mid"})

    # point-spread envelope (10-90) across points
    def _p(x: pd.Series, q: float) -> float:
        try:
            return float(x.quantile(q))
        except Exception:
            return float("nan")

    p10 = g[mid].apply(lambda s: _p(s, 0.10))
    p90 = g[mid].apply(lambda s: _p(s, 0.90))
    out["p10"] = p10.values
    out["p90"] = p90.values
    out = out.sort_values(by=t, kind="mergesort")
    return out


def exceed_prob_from_quantiles(
    row: pd.Series,
    *,
    thr: float,
    q_cols: Sequence[tuple[float, str]],
) -> float:
    """
    Approx P(Z > thr) using piecewise-linear CDF from quantiles.
    """
    qs: list[tuple[float, float]] = []
    for p, c in q_cols or []:
        if c in row.index:
            v = row[c]
            if pd.isna(v):
                continue
            qs.append((float(p), float(v)))

    if len(qs) < 2:
        return float("nan")

    qs.sort(key=lambda it: it[1])  # sort by value
    if thr <= qs[0][1]:
        return 1.0
    if thr >= qs[-1][1]:
        return 0.0

    # find bracket
    for (p0, v0), (p1, v1) in zip(qs, qs[1:]):
        if v0 <= thr <= v1:
            if abs(v1 - v0) < 1e-12:
                cdf = p1
            else:
                a = (thr - v0) / (v1 - v0)
                cdf = p0 + a * (p1 - p0)
            return float(1.0 - cdf)
    return float("nan")

def load_series_for_ids(
    *,
    path: Path,
    id_col: str,
    ids: Sequence[int],
    t: str,
    z: str,
    q_cols: Sequence[str],
    chunksize: int = 200_000,
    max_rows: int = 2_000_000,
) -> pd.DataFrame:
    """
    Load all (t, z, q*) rows matching selected point ids.

    Returns
    -------
    DataFrame with columns:
    - id_col, t, z, (q*)
    - _pid : stable point id string (id)
    """
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    ids = [int(i) for i in (ids or [])]
    if not ids:
        return pd.DataFrame()

    use = [c for c in (id_col, t, z) if c]
    use = list(dict.fromkeys(use))
    for qc in q_cols or []:
        qc = str(qc)
        if qc and qc not in use:
            use.append(qc)

    parts: list[pd.DataFrame] = []
    n_out = 0

    try:
        it = pd.read_csv(
            p,
            usecols=use,
            chunksize=int(chunksize),
        )
    except Exception:
        return pd.DataFrame()

    want = set(ids)

    for ch in it:
        if id_col not in ch.columns:
            continue

        try:
            sid = pd.to_numeric(
                ch[id_col],
                errors="coerce",
            ).astype("Int64")
        except Exception:
            continue

        m = sid.isin(want)
        if not bool(m.any()):
            continue

        sub = ch.loc[m].copy()
        sub[id_col] = (
            pd.to_numeric(sub[id_col], errors="coerce")
            .astype("Int64")
        )
        sub["_pid"] = sub[id_col].astype(str)

        parts.append(sub)
        n_out += int(sub.shape[0])
        if n_out >= int(max_rows):
            break

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out
