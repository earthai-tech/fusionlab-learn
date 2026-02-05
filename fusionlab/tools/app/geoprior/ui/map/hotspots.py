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
from typing import ( 
    Dict, 
    Iterable, 
    List, 
    Optional, 
    Callable, 
    Any
)
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

def _dist_km_vec(
    lon1: float,
    lat1: float,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    coord_mode: str,
) -> np.ndarray:
    if str(coord_mode).lower() != "lonlat":
        dx = lons - float(lon1)
        dy = lats - float(lat1)
        return np.sqrt(dx * dx + dy * dy) / 1000.0

    return _haversine_km_vec(lon1, lat1, lons, lats)


def _haversine_km_vec(
    lon1: float,
    lat1: float,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    r = 6371.0
    p = math.pi / 180.0

    a1 = float(lat1) * p
    a2 = lats * p
    dlat = (lats - float(lat1)) * p
    dlon = (lons - float(lon1)) * p

    s1 = np.sin(dlat / 2.0)
    s2 = np.sin(dlon / 2.0)

    a = (s1 * s1) + np.cos(a1) * np.cos(a2) * (s2 * s2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def _to_time_axis(s: pd.Series) -> np.ndarray:
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if dt.notna().any():
            x = dt.astype("int64").to_numpy(dtype=float)
            return x / 1e9
    except Exception:
        pass

    x2 = pd.to_numeric(s, errors="coerce")
    return x2.to_numpy(dtype=float)


def _fit_slope(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[
    Optional[float],
    Optional[float],
    Optional[float],
]:
    if x.size < 2:
        return None, None, None

    xx = x.astype(float)
    yy = y.astype(float)

    m = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[m]
    yy = yy[m]
    if xx.size < 2:
        return None, None, None

    vx = xx - float(xx.mean())
    vy = yy - float(yy.mean())

    den = float(np.sum(vx * vx))
    if den <= 0.0:
        return None, None, None

    slope = float(np.sum(vx * vy) / den)

    i0 = int(np.argmin(xx))
    i1 = int(np.argmax(xx))
    dv = float(yy[i1] - yy[i0])

    yhat = (slope * vx) + float(yy.mean())
    ssr = float(np.sum((yy - yhat) ** 2))
    sst = float(np.sum((yy - float(yy.mean())) ** 2))

    if sst <= 0.0:
        r2 = None
    else:
        r2 = float(1.0 - (ssr / sst))

    return slope, dv, r2

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

def hotspot_extremes(
    hs: Iterable[Dict[str, object]],
    pts: pd.DataFrame,
    *,
    v_col: str = "v",
    t_col: str = "t",
    lon_col: str = "lon",
    lat_col: str = "lat",
    radius_km: float = 1.0,
    min_n: int = 10,
    coord_mode: str = "lonlat",
) -> List[Dict[str, object]]:
    """
    Decorate hotspot payload with local extremes.

    Adds keys:
      - n_local
      - v_min, v_max, v_mean, v_std
      - t_min, t_max (if t_col exists)
    """
    if not hs:
        return []

    if pts is None or pts.empty:
        return [dict(h) for h in hs]

    need = [lon_col, lat_col, v_col]
    for c in need:
        if c not in pts.columns:
            return [dict(h) for h in hs]

    cols = [lon_col, lat_col, v_col]
    has_t = bool(t_col and (t_col in pts.columns))
    if has_t:
        cols.append(t_col)

    p = pts[cols].copy()
    p[lon_col] = pd.to_numeric(p[lon_col], errors="coerce")
    p[lat_col] = pd.to_numeric(p[lat_col], errors="coerce")
    p[v_col] = pd.to_numeric(p[v_col], errors="coerce")
    p = p.dropna(subset=[lon_col, lat_col, v_col])

    if p.empty:
        return [dict(h) for h in hs]

    lons = p[lon_col].to_numpy(dtype=float)
    lats = p[lat_col].to_numpy(dtype=float)
    vals = p[v_col].to_numpy(dtype=float)

    tt = None
    if has_t:
        tt = _to_time_axis(p[t_col])

    r = max(0.0, float(radius_km))
    k = max(1, int(min_n))

    out: List[Dict[str, object]] = []
    for h in hs:
        hh = dict(h)
        try:
            lon0 = float(hh.get("lon"))  # type: ignore[arg-type]
            lat0 = float(hh.get("lat"))  # type: ignore[arg-type]
        except Exception:
            out.append(hh)
            continue

        d = _dist_km_vec(
            lon0,
            lat0,
            lons,
            lats,
            coord_mode=str(coord_mode),
        )

        if r > 0.0:
            idx = np.where(d <= r)[0]
        else:
            idx = np.arange(d.size, dtype=int)

        if idx.size == 0:
            idx = np.argsort(d)[:k]

        vv = vals[idx]
        vv = vv[np.isfinite(vv)]
        if vv.size == 0:
            out.append(hh)
            continue

        hh["n_local"] = int(vv.size)
        hh["v_min"] = float(np.min(vv))
        hh["v_max"] = float(np.max(vv))
        hh["v_mean"] = float(np.mean(vv))
        hh["v_std"] = float(np.std(vv))

        if tt is not None:
            t0 = tt[idx]
            t0 = t0[np.isfinite(t0)]
            if t0.size > 0:
                hh["t_min"] = float(np.min(t0))
                hh["t_max"] = float(np.max(t0))

        out.append(hh)

    return out


def hotspot_trends(
    hs: Iterable[Dict[str, object]],
    pts: pd.DataFrame,
    *,
    v_col: str = "v",
    t_col: str = "t",
    lon_col: str = "lon",
    lat_col: str = "lat",
    radius_km: float = 1.0,
    min_n: int = 10,
    coord_mode: str = "lonlat",
) -> List[Dict[str, object]]:
    """
    Decorate hotspot payload with local time trends.

    Uses points within radius_km around each hotspot,
    groups by time, and fits slope(v) vs time.

    Adds keys:
      - trend_slope
      - trend_dv
      - trend_r2
      - trend_n_t
      - trend_n
      - trend_dir  (up|down|flat)
    """
    if not hs:
        return []

    if pts is None or pts.empty:
        return [dict(h) for h in hs]

    need = [lon_col, lat_col, v_col, t_col]
    for c in need:
        if c not in pts.columns:
            return [dict(h) for h in hs]

    p = pts[[lon_col, lat_col, v_col, t_col]].copy()
    p[lon_col] = pd.to_numeric(p[lon_col], errors="coerce")
    p[lat_col] = pd.to_numeric(p[lat_col], errors="coerce")
    p[v_col] = pd.to_numeric(p[v_col], errors="coerce")
    p = p.dropna(subset=[lon_col, lat_col, v_col, t_col])
    if p.empty:
        return [dict(h) for h in hs]

    lons = p[lon_col].to_numpy(dtype=float)
    lats = p[lat_col].to_numpy(dtype=float)
    vals = p[v_col].to_numpy(dtype=float)
    tx = _to_time_axis(p[t_col])

    r = max(0.0, float(radius_km))
    k = max(1, int(min_n))

    out: List[Dict[str, object]] = []
    for h in hs:
        hh = dict(h)
        try:
            lon0 = float(hh.get("lon"))  # type: ignore[arg-type]
            lat0 = float(hh.get("lat"))  # type: ignore[arg-type]
        except Exception:
            out.append(hh)
            continue

        d = _dist_km_vec(
            lon0,
            lat0,
            lons,
            lats,
            coord_mode=str(coord_mode),
        )

        if r > 0.0:
            idx = np.where(d <= r)[0]
        else:
            idx = np.arange(d.size, dtype=int)

        if idx.size == 0:
            idx = np.argsort(d)[:k]

        tt = tx[idx]
        vv = vals[idx]

        m2 = np.isfinite(tt) & np.isfinite(vv)
        tt = tt[m2]
        vv = vv[m2]
        if tt.size < 2:
            out.append(hh)
            continue

        df0 = pd.DataFrame({"t": tt, "v": vv})
        g = df0.groupby("t", sort=True)["v"].mean()

        x = g.index.to_numpy(dtype=float)
        y = g.to_numpy(dtype=float)

        slope, dv, r2 = _fit_slope(x, y)

        hh["trend_slope"] = (
            None if slope is None else float(slope)
        )
        hh["trend_dv"] = None if dv is None else float(dv)
        hh["trend_r2"] = None if r2 is None else float(r2)
        hh["trend_n_t"] = int(x.size)
        hh["trend_n"] = int(tt.size)

        if slope is None or slope == 0.0:
            hh["trend_dir"] = "flat"
        elif slope > 0.0:
            hh["trend_dir"] = "up"
        else:
            hh["trend_dir"] = "down"

        out.append(hh)

    return out

def decorate_hotspots(
    hs_payload: Iterable[Dict[str, object]],
    df_frame: Optional[pd.DataFrame],
    df_all: Optional[pd.DataFrame],
    *,
    add_extremes: bool = True,
    add_trends: bool = False,
    v_col: str = "v",
    t_col: str = "t",
    lon_col: str = "lon",
    lat_col: str = "lat",
    extremes_radius_km: float = 1.0,
    trends_radius_km: float = 1.0,
    min_n: int = 10,
    coord_mode: str = "lonlat",
) -> List[Dict[str, object]]:
    """
    Apply common hotspot payload decorators.

    Parameters
    ----------
    hs_payload
        JSON-friendly hotspot payload (dicts with lon/lat).
    df_frame
        Current frame points dataframe (lon/lat/v[/t]).
        Used for local extremes.
    df_all
        Full history points dataframe (lon/lat/v/t).
        Used for trends (and extremes fallback).
    add_extremes
        If True, add local v_min/v_max/v_mean/v_std/n_local.
    add_trends
        If True, add local trend_slope/trend_dv/trend_r2/...
    v_col, t_col, lon_col, lat_col
        Column names for points frames. Defaults match
        the normalized controller schema (lon,lat,v,t).
    extremes_radius_km, trends_radius_km
        Neighborhood radius for extremes/trends.
    min_n
        Fallback minimum number of nearest points.
    coord_mode
        "lonlat" uses haversine km; otherwise euclid/1000.

    Returns
    -------
    list of dict
        Decorated hotspot payload.
    """
    base = [dict(h) for h in (hs_payload or [])]
    if not base:
        return []

    out = base

    if add_extremes:
        src = df_frame
        if src is None or src.empty:
            src = df_all

        if src is not None and not src.empty:
            out = hotspot_extremes(
                out,
                src,
                v_col=v_col,
                t_col=t_col,
                lon_col=lon_col,
                lat_col=lat_col,
                radius_km=float(extremes_radius_km),
                min_n=int(min_n),
                coord_mode=str(coord_mode),
            )

    if add_trends:
        src2 = df_all
        if src2 is not None and not src2.empty:
            out = hotspot_trends(
                out,
                src2,
                v_col=v_col,
                t_col=t_col,
                lon_col=lon_col,
                lat_col=lat_col,
                radius_km=float(trends_radius_km),
                min_n=int(min_n),
                coord_mode=str(coord_mode),
            )

    return list(out or [])

def decorate_hotspots_from_store(
    get: Callable[[str, Any], Any],
    hs_payload: Iterable[Dict[str, object]],
    df_frame: Optional[pd.DataFrame],
    df_all: Optional[pd.DataFrame],
    *,
    base_key: str = "map.view.hotspots.",
    coord_mode: str = "lonlat",
    v_col: str = "v",
    t_col: str = "t",
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> List[Dict[str, object]]:
    """
    Decorate hotspots using map.view.hotspots.* keys.

    Reads (with safe fallbacks):
      - map.view.hotspots.decorate_extremes        (bool)
      - map.view.hotspots.decorate_trends          (bool)
      - map.view.hotspots.extremes_radius_km       (float)
      - map.view.hotspots.trends_radius_km         (float)
      - map.view.hotspots.decorate_min_n           (int)

    Also accepts common aliases (in case your UI used
    different names while iterating).
    """
    def _g(name: str, default: Any) -> Any:
        try:
            return get(base_key + name, default)
        except Exception:
            return default

    # toggles (aliases supported)
    add_ext = bool(
        _g(
            "decorate_extremes",
            _g("extremes", True),
        )
    )
    add_tr = bool(
        _g(
            "decorate_trends",
            _g("trends", False),
        )
    )

    # radii (aliases supported)
    ext_r = _g("extremes_radius_km", None)
    if ext_r is None:
        ext_r = _g("decorate_radius_km", 1.0)
    try:
        ext_r = float(ext_r)
    except Exception:
        ext_r = 1.0

    tr_r = _g("trends_radius_km", None)
    if tr_r is None:
        tr_r = _g("decorate_trends_radius_km", ext_r)
    try:
        tr_r = float(tr_r)
    except Exception:
        tr_r = float(ext_r)

    # min_n (aliases supported)
    mn = _g("decorate_min_n", None)
    if mn is None:
        mn = _g("min_n", 10)
    try:
        mn = int(mn)
    except Exception:
        mn = 10

    return decorate_hotspots(
        hs_payload,
        df_frame,
        df_all,
        add_extremes=bool(add_ext),
        add_trends=bool(add_tr),
        v_col=str(v_col),
        t_col=str(t_col),
        lon_col=str(lon_col),
        lat_col=str(lat_col),
        extremes_radius_km=float(ext_r),
        trends_radius_km=float(tr_r),
        min_n=int(mn),
        coord_mode=str(coord_mode),
    )
