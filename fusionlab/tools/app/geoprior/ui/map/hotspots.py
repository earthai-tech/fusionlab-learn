# geoprior/ui/map/hotspots.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.hotspots

Hotspot detection utilities for MapTab.

Hotspots are "attention points" computed from
(lon, lat, value) samples.

We keep deps light: pandas + numpy only.

Outputs are small payload dicts that can be
rendered as a separate layer (rings, labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HotspotCfg:
    method: str = "grid"       # "grid" | "quantile"
    metric: str = "value"      # "value" | "abs" | "high" | "low"
    quantile: float = 0.98
    thr_mode: str = "quantile"    # "quantile" | "absolute"
    abs_thr: Optional[float] = None
    
    # optional time aggregation
    time_col: str = ""            # name of time column if present
    time_agg: str = "current"     # "current" | "mean" | "max" | "trend"
    time_window: int = 0          # 0 => use all available
    max_n: int = 8
    min_sep_km: float = 2.0
    cell_km: float = 1.0
    min_pts: int = 20


@dataclass(frozen=True)
class Hotspot:
    lon: float
    lat: float
    v: float
    score: float
    n: int
    rank: int
    sev: str
    label: str


def build_points(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    v: str,
    t: str = "",
) -> pd.DataFrame:

    """
    Normalize dataframe into columns: lon, lat, v.

    This is the canonical schema used by the map
    and hotspot computations.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["lon", "lat", "v"])

    if x not in df.columns or y not in df.columns:
        return pd.DataFrame(columns=["lon", "lat", "v"])

    if v not in df.columns:
        return pd.DataFrame(columns=["lon", "lat", "v"])

    cols = [x, y, v]
    has_t = bool(t and (t in df.columns))
    if has_t:
        cols.append(t)

    out = df[cols].copy()
    if has_t:
        out.columns = ["lon", "lat", "v", "t"]
    else:
        out.columns = ["lon", "lat", "v"]

    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["v"] = pd.to_numeric(out["v"], errors="coerce")

    if "t" in out.columns:
        # Accept numeric, datetime, or sortable categories
        try:
            tt = pd.to_datetime(out["t"], errors="coerce")
            if tt.notna().any():
                out["t"] = tt
            else:
                out["t"] = pd.to_numeric(out["t"], errors="coerce")
        except Exception:
            pass

    out = out.dropna(subset=["lon", "lat"])
    return out


def clip_lonlat(pts: pd.DataFrame) -> pd.DataFrame:
    if pts is None or pts.empty:
        return pts

    ok_lon = pts["lon"].between(-180.0, 180.0)
    ok_lat = pts["lat"].between(-90.0, 90.0)
    return pts[ok_lon & ok_lat]

def aggregate_points_over_time(
    pts: pd.DataFrame,
    *,
    time_col: str,
    time_agg: str = "current",
    time_window: int = 0,
) -> pd.DataFrame:
    """
    Aggregate (lon,lat,v[,t]) into (lon,lat,v) using a time aggregation.

    - current: keep latest time slice (requires time_col)
    - mean: mean over all/window
    - max: max over all/window
    - trend: slope over time (linear fit) per location/grid key

    Notes
    -----
    - If time_col missing or invalid, returns pts unchanged.
    - time_window > 0 => use only last N unique times.
    """
    if pts is None or pts.empty:
        return pts
    if not time_col or time_col not in pts.columns:
        return pts

    p = pts.copy()
    if p[time_col].isna().all():
        return pts

    # Normalize and sort time
    t = p[time_col]
    if np.issubdtype(t.dtype, np.datetime64):
        p = p.sort_values(time_col)
        uniq = pd.Index(p[time_col].dropna().unique()).sort_values()
    else:
        # numeric or other sortable
        p = p.sort_values(time_col)
        uniq = pd.Index(p[time_col].dropna().unique())

    if len(uniq) == 0:
        return pts

    if int(time_window) > 0 and len(uniq) > int(time_window):
        keep_times = set(uniq[-int(time_window):])
        p = p[p[time_col].isin(keep_times)].copy()
        if p.empty:
            return pts

    agg = str(time_agg or "current").strip().lower()

    if agg == "current":
        t_last = uniq[-1]
        return p[p[time_col] == t_last][["lon", "lat", "v"]].copy()

    # group by exact lon/lat (assumes already normalized)
    g = p.groupby(["lon", "lat"], sort=False)

    if agg == "mean":
        return g.agg(v=("v", "mean")).reset_index()

    if agg == "max":
        return g.agg(v=("v", "max")).reset_index()

    if agg == "trend":
        # slope of v vs time (per location)
        # time -> numeric axis
        if np.issubdtype(p[time_col].dtype, np.datetime64):
            x = p[time_col].astype("int64") / 1e9
        else:
            x = pd.to_numeric(p[time_col], errors="coerce")

        p2 = p.copy()
        p2["_tx"] = x

        def _slope(df: pd.DataFrame) -> float:
            xx = df["_tx"].to_numpy(dtype=float)
            yy = df["v"].to_numpy(dtype=float)
            m = np.isfinite(xx) & np.isfinite(yy)
            xx = xx[m]
            yy = yy[m]
            if xx.size < 2:
                return np.nan
            # OLS slope
            vx = xx - xx.mean()
            vy = yy - yy.mean()
            den = float((vx * vx).sum())
            if den <= 0:
                return np.nan
            return float((vx * vy).sum() / den)

        out = p2.groupby(["lon", "lat"], sort=False).apply(_slope)
        out = out.reset_index()
        out.columns = ["lon", "lat", "v"]
        return out

    # fallback
    return pts

def compute_hotspots(
    pts: pd.DataFrame,
    *,
    cfg: HotspotCfg,
    coord_mode: str = "lonlat",
) -> List[Hotspot]:
    """
    Compute hotspot list from pts(lon,lat,v).

    coord_mode:
      - "lonlat": haversine for separation
      - other: euclid in coord units (approx)
    """
    if pts is None or pts.empty:
        return []

    p = pts.copy()

    if str(coord_mode).lower() == "lonlat":
        p = clip_lonlat(p)

    if p.empty:
        return []

    # Optional: time aggregation (if pts contains cfg.time_col)
    if getattr(cfg, "time_col", ""):
        p = aggregate_points_over_time(
            p,
            time_col=str(cfg.time_col),
            time_agg=str(getattr(cfg, "time_agg", "current")),
            time_window=int(getattr(cfg, "time_window", 0) or 0),
        )
        if p is None or p.empty:
            return []

    score = _score_series(p["v"], cfg.metric)
    p["_score"] = score

    # Threshold filter (quantile or absolute) happens inside the method.
    if cfg.method == "quantile":
        c = _hotspots_quantile(p, cfg)
    else:
        c = _hotspots_grid(p, cfg)

    if c.empty:
        return []

    c = c.sort_values("_score", ascending=False)

    keep = _apply_min_sep(
        c,
        max_n=int(cfg.max_n),
        min_sep_km=float(cfg.min_sep_km),
        coord_mode=str(coord_mode),
    )
    if keep.empty:
        return []

    hs = _to_hotspots(keep)
    return hs


def hotspots_payload(hs: Iterable[Hotspot]) -> List[Dict[str, object]]:
    """
    Convert hotspots to JSON-friendly payload.
    """
    out: List[Dict[str, object]] = []
    for h in hs or []:
        out.append(
            {
                "lon": float(h.lon),
                "lat": float(h.lat),
                "v": float(h.v),
                "score": float(h.score),
                "n": int(h.n),
                "rank": int(h.rank),
                "sev": str(h.sev),
                "label": str(h.label),
            }
        )
    return out


# -------------------------------------------------
# Internals
# -------------------------------------------------
def _score_series(s: pd.Series, metric: str) -> pd.Series:
    m = str(metric or "value").strip().lower()

    x = pd.to_numeric(s, errors="coerce")
    if m == "abs":
        return x.abs()
    if m == "low":
        return (-x)
    if m == "high":
        return x
    return x


def _hotspots_quantile(p: pd.DataFrame, cfg: HotspotCfg) -> pd.DataFrame:
    s = p["_score"].dropna()
    if s.empty:
        return pd.DataFrame()

    thr_mode = str(getattr(cfg, "thr_mode", "quantile") or "quantile").lower()

    if thr_mode == "absolute":
        thr = getattr(cfg, "abs_thr", None)
        if thr is None:
            return pd.DataFrame()
        thr = float(thr)
    else:
        q = float(getattr(cfg, "quantile", 0.98) or 0.98)
        q = max(0.0, min(1.0, q))
        thr = float(s.quantile(q))

    c = p[p["_score"] >= thr].copy()
    c["_n"] = 1
    return c[["lon", "lat", "v", "_score", "_n"]]


def _hotspots_grid(p: pd.DataFrame, cfg: HotspotCfg) -> pd.DataFrame:
    if float(cfg.cell_km) <= 0.0:
        return _hotspots_quantile(p, cfg)

    lon = p["lon"].to_numpy(dtype=float)
    lat = p["lat"].to_numpy(dtype=float)

    lat0 = float(np.nanmedian(lat))
    cell_km = float(cfg.cell_km)

    dy = cell_km / 111.0
    dx = cell_km / max(1e-9, 111.0 * math.cos(
        math.radians(lat0)
    ))

    lon0 = float(np.nanmin(lon))
    lat0m = float(np.nanmin(lat))

    gx = np.floor((lon - lon0) / dx).astype("int64")
    gy = np.floor((lat - lat0m) / dy).astype("int64")

    p2 = p.copy()
    p2["_gx"] = gx
    p2["_gy"] = gy
    p2["_cell"] = (p2["_gy"] * 10_000_000) + p2["_gx"]

    g = p2.groupby("_cell", sort=False)

    out = g.agg(
        lon=("lon", "mean"),
        lat=("lat", "mean"),
        v=("v", "mean"),
        _score=("_score", "mean"),
        _n=("v", "size"),
    ).reset_index(drop=True)

    out = out[out["_n"] >= int(cfg.min_pts)]

    # Apply threshold mode on aggregated scores
    thr_mode = str(getattr(cfg, "thr_mode", "quantile") or "quantile").lower()
    if not out.empty:
        if thr_mode == "absolute":
            thr = getattr(cfg, "abs_thr", None)
            if thr is None:
                return pd.DataFrame()
            thr = float(thr)
            out = out[out["_score"] >= thr]
        else:
            q = float(getattr(cfg, "quantile", 0.98) or 0.98)
            q = max(0.0, min(1.0, q))
            thr = float(out["_score"].dropna().quantile(q)) \
                if out["_score"].notna().any() else np.nan
            if np.isfinite(thr):
                out = out[out["_score"] >= thr]
            else:
                return pd.DataFrame()

    return out

def _apply_min_sep(
    c: pd.DataFrame,
    *,
    max_n: int,
    min_sep_km: float,
    coord_mode: str,
) -> pd.DataFrame:
    if c is None or c.empty:
        return pd.DataFrame()

    max_n = max(1, int(max_n))
    min_sep_km = max(0.0, float(min_sep_km))

    keep: List[int] = []
    pts = c[["lon", "lat"]].to_numpy(dtype=float)

    for i in range(len(c)):
        if len(keep) >= max_n:
            break

        if not keep:
            keep.append(i)
            continue

        ok = True
        for j in keep:
            d = _dist_km(
                pts[i, 0],
                pts[i, 1],
                pts[j, 0],
                pts[j, 1],
                coord_mode,
            )
            if d < min_sep_km:
                ok = False
                break
        if ok:
            keep.append(i)

    return c.iloc[keep].copy()


def _dist_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    coord_mode: str,
) -> float:
    if str(coord_mode).lower() != "lonlat":
        dx = float(lon1 - lon2)
        dy = float(lat1 - lat2)
        return float(math.sqrt(dx * dx + dy * dy) / 1000.0)

    return _haversine_km(lon1, lat1, lon2, lat2)


def _haversine_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    r = 6371.0
    p = math.pi / 180.0

    a1 = float(lat1) * p
    a2 = float(lat2) * p
    dlat = (float(lat2) - float(lat1)) * p
    dlon = (float(lon2) - float(lon1)) * p

    s1 = math.sin(dlat / 2.0)
    s2 = math.sin(dlon / 2.0)

    a = (s1 * s1) + math.cos(a1) * math.cos(a2) * (s2 * s2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(r * c)


def _severity(rank: int) -> str:
    if rank <= 1:
        return "critical"
    if rank <= 3:
        return "high"
    if rank <= 6:
        return "medium"
    return "low"


def _to_hotspots(c: pd.DataFrame) -> List[Hotspot]:
    out: List[Hotspot] = []

    cols = ["lon", "lat", "v", "_score", "_n"]
    for cc in cols:
        if cc not in c.columns:
            return out

    it = c[cols].itertuples(
        index=False,
        name=None,
    )

    for k, (lon, lat, v, sc, n) in enumerate(
        it,
        start=1,
    ):
        try:
            lon = float(lon)
            lat = float(lat)
            v = float(v)
            sc = float(sc)
            n = int(n)
        except Exception:
            continue

        sev = _severity(k)
        lab = f"{sev} · #{k} · n={n}"

        out.append(
            Hotspot(
                lon=lon,
                lat=lat,
                v=v,
                score=sc,
                n=n,
                rank=k,
                sev=sev,
                label=lab,
            )
        )

    return out

