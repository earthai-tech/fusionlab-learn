# geoprior/ui/xfer/map/interaction_extras.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Interaction extras:

- Δ-hotspots (hotspots on delta grid)
- Overlap intensity (|Δ| × coverage)
- Buffered intersection (grow inter cells by k)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math 
import numpy as np
import pandas as pd

from ...map.hotspots import HotspotCfg, compute_hotspots


@dataclass(frozen=True)
class IntExtrasCfg:
    # grid used by delta/intensity/buffer
    cell_km: float = 2.0
    agg: str = "mean"
    delta: str = "a_minus_b"

    # delta hotspots
    hot_enable: bool = False
    hot_topn: int = 8
    hot_metric: str = "abs"
    hot_quantile: float = 0.98
    hot_min_sep_km: float = 2.0

    # overlap intensity
    intens_enable: bool = False

    # buffered intersection
    buf_enable: bool = False
    buf_k: int = 1


def compute_interaction_extras(
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    *,
    cfg: IntExtrasCfg,
    coord_mode: str = "lonlat",
    radius: int = 7,
    opacity: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Return extra layer payloads.

    Each payload is:
      {id,name,df,opts,legend?}
    df schema: lon,lat,v[,sid,tip]
    """
    a0 = _prep(a_df)
    b0 = _prep(b_df)
    if a0.empty or b0.empty:
        return []

    cell_km = float(cfg.cell_km or 2.0)
    cell_km = max(0.2, min(20.0, cell_km))

    a_g = _grid_agg(a0, cell_km=cell_km, agg=cfg.agg)
    b_g = _grid_agg(b0, cell_km=cell_km, agg=cfg.agg)
    if a_g.empty or b_g.empty:
        return []

    j = _join(a_g, b_g)

    out: List[Dict[str, Any]] = []

    if bool(cfg.intens_enable):
        p = _layer_intensity(
            j,
            cell_km=cell_km,
            delta=str(cfg.delta or "a_minus_b"),
            radius=radius,
            opacity=opacity,
        )
        if p is not None:
            out.append(p)

    if bool(cfg.hot_enable):
        p = _layer_delta_hotspots(
            j,
            cell_km=cell_km,
            delta=str(cfg.delta or "a_minus_b"),
            coord_mode=str(coord_mode or "lonlat"),
            topn=int(cfg.hot_topn or 8),
            metric=str(cfg.hot_metric or "abs"),
            quant=float(cfg.hot_quantile or 0.98),
            min_sep=float(cfg.hot_min_sep_km or 2.0),
            radius=radius,
            opacity=opacity,
        )
        if p is not None:
            out.append(p)

    if bool(cfg.buf_enable):
        p = _layer_buf_intersection(
            j,
            cell_km=cell_km,
            k=int(cfg.buf_k or 1),
            radius=radius,
            opacity=opacity,
        )
        if p is not None:
            out.append(p)

    return out


# -------------------------
# Join utilities
# -------------------------
def _prep(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["lon", "lat", "v"])
    out = df.copy()
    for c in ("lon", "lat", "v"):
        out[c] = pd.to_numeric(out.get(c), errors="coerce")
    out = out.dropna(subset=["lon", "lat", "v"])
    return out


def _grid_agg(
    df: pd.DataFrame,
    *,
    cell_km: float,
    agg: str,
) -> pd.DataFrame:
    lon = df["lon"].to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    v = df["v"].to_numpy(dtype=float)

    lat0 = float(np.nanmedian(lat))
    dy = float(cell_km) / 111.0
    dx = float(cell_km) / max(
        1e-9,
        111.0 * np.cos(np.deg2rad(lat0)),
    )

    lon_min = float(np.nanmin(lon))
    lat_min = float(np.nanmin(lat))

    gx = np.floor((lon - lon_min) / dx).astype("int64")
    gy = np.floor((lat - lat_min) / dy).astype("int64")

    p = pd.DataFrame(
        {
            "gx": gx,
            "gy": gy,
            "lon": lon,
            "lat": lat,
            "v": v,
        }
    )

    a = str(agg or "mean").strip().lower()
    if a == "median":
        f = "median"
    elif a == "max":
        f = "max"
    else:
        f = "mean"

    g = p.groupby(["gx", "gy"], sort=False)
    out = g.agg(
        lon=("lon", "mean"),
        lat=("lat", "mean"),
        v=("v", f),
        n=("v", "count"),
    ).reset_index()

    return out


def _join(a_g: pd.DataFrame, b_g: pd.DataFrame) -> pd.DataFrame:
    aa = a_g.rename(
        columns={
            "lon": "lon_a",
            "lat": "lat_a",
            "v": "v_a",
            "n": "n_a",
        }
    )
    bb = b_g.rename(
        columns={
            "lon": "lon_b",
            "lat": "lat_b",
            "v": "v_b",
            "n": "n_b",
        }
    )
    return pd.merge(aa, bb, on=["gx", "gy"], how="outer")


def _pick_lonlat(j: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    lon = j["lon_a"].where(j["lon_a"].notna(), j["lon_b"])
    lat = j["lat_a"].where(j["lat_a"].notna(), j["lat_b"])
    return lon, lat


def _delta_vals(j: pd.DataFrame, delta: str) -> pd.Series:
    d = str(delta or "a_minus_b").strip().lower()
    if d in ("b_minus_a", "b-a"):
        return j["v_b"] - j["v_a"]
    return j["v_a"] - j["v_b"]


# -------------------------
# Extra layers
# -------------------------
def _layer_delta_hotspots(
    j: pd.DataFrame,
    *,
    cell_km: float,
    delta: str,
    coord_mode: str,
    topn: int,
    metric: str,
    quant: float,
    min_sep: float,
    radius: int,
    opacity: float,
) -> Optional[Dict[str, Any]]:
    m = j["v_a"].notna() & j["v_b"].notna()
    if not bool(m.any()):
        return None

    jj = j[m].copy()
    lon = (jj["lon_a"] + jj["lon_b"]) / 2.0
    lat = (jj["lat_a"] + jj["lat_b"]) / 2.0

    dv = _delta_vals(jj, delta)
    pts = pd.DataFrame({"lon": lon, "lat": lat, "v": dv})
    pts = pts.dropna(subset=["lon", "lat", "v"])
    if pts.empty:
        return None

    q = max(0.50, min(0.999, float(quant)))
    cfg = HotspotCfg(
        method="grid",
        metric=str(metric or "abs"),
        quantile=q,
        max_n=max(1, int(topn)),
        min_sep_km=max(0.0, float(min_sep)),
        cell_km=max(0.2, float(cell_km)),
        min_pts=5,
    )

    hs = compute_hotspots(
        pts,
        cfg=cfg,
        coord_mode=str(coord_mode or "lonlat"),
    )
    if not hs:
        return None

    rows: List[Dict[str, Any]] = []
    for h in hs:
        tip = (
            "Δ-hotspot\n"
            f"rank={int(h.rank)}\n"
            f"Δ={float(h.v):.3g}\n"
            f"score={float(h.score):.3g}\n"
            f"n={int(h.n)}"
        )
        rows.append(
            {
                "lon": float(h.lon),
                "lat": float(h.lat),
                "v": float(h.v),
                "sid": int(h.rank),
                "tip": tip,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    opts = {
        "stroke": "#111827",
        "fillMode": "fixed",
        "fillColor": "#EF4444",
        "shape": "diamond",
        "radius": int(max(3, min(18, radius + 2))),
        "opacity": float(opacity),
        "pulse": True,
        "enableTooltip": True,
    }

    return {
        "id": "I_DHOT",
        "name": "Δ hotspots",
        "df": df,
        "opts": opts,
    }


def _layer_intensity(
    j: pd.DataFrame,
    *,
    cell_km: float,
    delta: str,
    radius: int,
    opacity: float,
) -> Optional[Dict[str, Any]]:
    m = j["v_a"].notna() & j["v_b"].notna()
    if not bool(m.any()):
        return None

    jj = j[m].copy()
    lon = (jj["lon_a"] + jj["lon_b"]) / 2.0
    lat = (jj["lat_a"] + jj["lat_b"]) / 2.0

    dv = _delta_vals(jj, delta).to_numpy(dtype=float)
    ad = np.abs(dv)

    na = pd.to_numeric(jj.get("n_a"), errors="coerce").to_numpy()
    nb = pd.to_numeric(jj.get("n_b"), errors="coerce").to_numpy()

    if np.isfinite(na).any() and np.isfinite(nb).any():
        mn = np.minimum(na, nb)
        sm = na + nb
        w = (2.0 * mn) / (sm + 1e-12)
        w = np.clip(w, 0.0, 1.0)
    else:
        w = np.ones_like(ad)

    inten = ad * w

    tip: List[str] = []
    for k in range(len(inten)):
        tip.append(
            "Overlap intensity\n"
            f"cell={cell_km:.2f} km\n"
            f"|Δ|={ad[k]:.3g}\n"
            f"w={w[k]:.3g}\n"
            f"I={inten[k]:.3g}"
        )

    df = pd.DataFrame(
        {
            "lon": lon.to_numpy(dtype=float),
            "lat": lat.to_numpy(dtype=float),
            "v": inten,
            "tip": tip,
        }
    )
    df = df.dropna(subset=["lon", "lat", "v"])
    if df.empty:
        return None

    x = pd.to_numeric(df["v"], errors="coerce").to_numpy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None

    vmin = float(np.min(x))
    vmax = float(np.max(x))

    opts = {
        "stroke": "#111827",
        "fillMode": "value",
        "shape": "square",
        "radius": int(max(3, min(18, radius))),
        "opacity": float(opacity),
        "pulse": False,
        "enableTooltip": True,
        "vmin": vmin,
        "vmax": vmax,
    }

    leg = {"title": "Overlap intensity", "vmin": vmin, "vmax": vmax}

    return {
        "id": "I_INTENS",
        "name": "Overlap intensity",
        "df": df,
        "opts": opts,
        "legend": leg,
    }


def _layer_buf_intersection(
    j: pd.DataFrame,
    *,
    cell_km: float,
    k: int,
    radius: int,
    opacity: float,
) -> Optional[Dict[str, Any]]:
    m = j["v_a"].notna() & j["v_b"].notna()
    if not bool(m.any()):
        return None

    kk = max(0, int(k))
    if kk == 0:
        return None
    
    jj = j.copy()
    jj = jj[jj["gx"].notna() & jj["gy"].notna()].copy()
    
    jj["gx"] = jj["gx"].astype("int64")
    jj["gy"] = jj["gy"].astype("int64")
    jj["key"] = (jj["gx"] * 10_000_000) + jj["gy"]
    
    keyset = set(jj["key"].to_numpy(dtype="int64"))
    
    m2 = jj["v_a"].notna() & jj["v_b"].notna()
    base = jj.loc[m2, ["gx", "gy"]].to_numpy(dtype="int64")

    if base.size == 0:
        return None

    tgt: set[int] = set()
    for gx, gy in base:
        for dx in range(-kk, kk + 1):
            for dy in range(-kk, kk + 1):
                key = int((gx + dx) * 10_000_000 + (gy + dy))
                if key in keyset:
                    tgt.add(key)

    if not tgt:
        return None

    keep = jj[jj["key"].isin(list(tgt))].copy()
    lon, lat = _pick_lonlat(keep)

    tip: List[str] = []
    for _, r in keep.iterrows():
        a = r.get("v_a", np.nan)
        b = r.get("v_b", np.nan)
        tip.append(
            "Buffered intersection\n"
            f"k={kk}\n"
            f"A={_fmt(a)}  B={_fmt(b)}"
        )

    df = pd.DataFrame(
        {
            "lon": lon,
            "lat": lat,
            "v": 1.0,
            "tip": tip,
        }
    )
    df = df.dropna(subset=["lon", "lat"])
    if df.empty:
        return None

    opts = {
        "stroke": "#111827",
        "fillMode": "fixed",
        "fillColor": "#7B2CBF",
        "shape": "square",
        "radius": int(max(3, min(18, radius))),
        "opacity": float(opacity),
        "pulse": False,
        "enableTooltip": True,
    }

    return {
        "id": "I_BUFINT",
        "name": "Buffered intersection",
        "df": df,
        "opts": opts,
    }


def _fmt(v: object) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.3g}"



def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2.0) ** 2
    )
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def build_hotspot_links(
    hs_a: pd.DataFrame,
    hs_b: pd.DataFrame,
    *,
    mode: str = "nearest",
    k: int = 1,
    max_links: int = 12,
    show_dist: bool = True,
) -> List[List[Any]]:
    """
    Return JS rows:
      [lat1,lon1,lat2,lon2,dist_km,tip,label]
    Expects cols: lon,lat,sid (optional), tip (optional)
    """
    if hs_a is None or hs_b is None:
        return []
    if getattr(hs_a, "empty", True) or getattr(hs_b, "empty", True):
        return []

    a = hs_a.copy()
    b = hs_b.copy()

    for df in (a, b):
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    a = a.dropna(subset=["lon", "lat"])
    b = b.dropna(subset=["lon", "lat"])
    if a.empty or b.empty:
        return []

    mode = str(mode or "nearest").strip().lower()
    k = max(1, int(k or 1))
    max_links = max(1, int(max_links or 12))

    pairs: List[tuple[int, int, float]] = []

    if mode == "rank":
        n = min(len(a), len(b), max_links)
        for i in range(n):
            d = _haversine_km(
                float(a.iloc[i]["lat"]),
                float(a.iloc[i]["lon"]),
                float(b.iloc[i]["lat"]),
                float(b.iloc[i]["lon"]),
            )
            pairs.append((i, i, d))
    else:
        cand: List[tuple[int, int, float]] = []
        for ia in range(len(a)):
            la = float(a.iloc[ia]["lat"])
            loa = float(a.iloc[ia]["lon"])
            for ib in range(len(b)):
                lb = float(b.iloc[ib]["lat"])
                lob = float(b.iloc[ib]["lon"])
                cand.append((ia, ib, _haversine_km(la, loa, lb, lob)))
        cand.sort(key=lambda x: x[2])

        if mode == "nearest":
            used_a: set[int] = set()
            used_b: set[int] = set()
            for ia, ib, d in cand:
                if ia in used_a or ib in used_b:
                    continue
                pairs.append((ia, ib, d))
                used_a.add(ia)
                used_b.add(ib)
                if len(pairs) >= max_links:
                    break
        else:
            # knn
            per_a: dict[int, int] = {}
            for ia, ib, d in cand:
                c = per_a.get(ia, 0)
                if c >= k:
                    continue
                pairs.append((ia, ib, d))
                per_a[ia] = c + 1
                if len(pairs) >= max_links:
                    break

    out: List[List[Any]] = []
    for ia, ib, d in pairs[:max_links]:
        ra = a.iloc[ia]
        rb = b.iloc[ib]
        label = f"{d:.2f} km" if show_dist else ""
        tip = (
            "Hotspot link\n"
            f"A#{ra.get('sid','?')} → "
            f"B#{rb.get('sid','?')}\n"
            f"d={d:.2f} km"
        )
        out.append(
            [
                float(ra["lat"]),
                float(ra["lon"]),
                float(rb["lat"]),
                float(rb["lon"]),
                float(d),
                str(tip),
                str(label),
            ]
        )
    return out


def build_radar_centers(
    hs_a: pd.DataFrame,
    hs_b: pd.DataFrame,
    *,
    target: str,
    overlay: str,
    order: str,
) -> List[List[Any]]:
    """
    Return JS rows:
      [lat,lon,rank,score,tip]
    """
    target = str(target or "overlay").strip().lower()
    overlay = str(overlay or "both").strip().lower()
    order = str(order or "score").strip().lower()

    use_a = target in ("a",) or (target == "overlay" and overlay in ("a",))
    use_b = target in ("b",) or (target == "overlay" and overlay in ("b",))
    use_both = target in ("both",) or (target == "overlay" and overlay == "both")

    frames: List[pd.DataFrame] = []
    if use_both or use_a:
        if hs_a is not None and not hs_a.empty:
            frames.append(hs_a.copy())
    if use_both or use_b:
        if hs_b is not None and not hs_b.empty:
            frames.append(hs_b.copy())

    if not frames:
        return []

    df = pd.concat(frames, axis=0, ignore_index=True)
    df["score"] = pd.to_numeric(df.get("score", df.get("v")), errors="coerce")
    df["sid"] = pd.to_numeric(df.get("sid", 0), errors="coerce").fillna(0)

    if order in ("rank",):
        df = df.sort_values("sid", ascending=True)
    else:
        # score/abs
        sc = df["score"].to_numpy(dtype=float)
        df = df.assign(_k=np.abs(sc)).sort_values("_k", ascending=False)

    out: List[List[Any]] = []
    rk = 1
    for r in df.itertuples(index=False):
        lat = float(getattr(r, "lat"))
        lon = float(getattr(r, "lon"))
        score = float(getattr(r, "score", 0.0))
        sid = int(getattr(r, "sid", rk))
        tip = getattr(r, "tip", None)
        if not tip:
            tip = f"Hotspot\nrank={sid}\nscore={score:.3g}"
        out.append([lat, lon, rk, score, str(tip)])
        rk += 1

    return out

