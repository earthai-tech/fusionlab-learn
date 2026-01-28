# geoprior/ui/xfer/map/interactions.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.map.interactions

Compute derived "interaction" layers from City A/B
point sets:

- A-only mask
- B-only mask
- Union mask
- Intersection mask
- Delta layer: Δ(A - B) or Δ(B - A)

Design
------
We avoid heavy GIS deps by using a grid join
in lon/lat space:

1) project lon/lat degrees -> approx meters
2) bin into cell_km x cell_km grid
3) aggregate per cell for A and B
4) outer-join cells and derive layers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InteractionCfg:
    mode: str = "none"
    cell_km: float = 2.0
    agg: str = "mean"
    delta: str = "a_minus_b"


@dataclass(frozen=True)
class LayerSpec:
    layer_id: str
    name: str
    df: pd.DataFrame
    opts: Dict[str, Any]
    legend: Optional[Dict[str, Any]] = None


def compute_interaction_layers(
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    *,
    cfg: InteractionCfg,
    radius: int = 6,
    opacity: float = 0.85,
) -> List[LayerSpec]:
    """
    Build interaction layer specs.

    Inputs must have columns: lon, lat, v.
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

    mode = str(cfg.mode or "none").strip().lower()
    if mode in ("none", "", "off"):
        return []

    out: List[LayerSpec] = []

    if mode in ("zones", "partition"):
        out.extend(
            _zones_layers(
                j,
                cell_km=cell_km,
                radius=radius,
                opacity=opacity,
            )
        )
        return out

    if mode == "a_only":
        df = _mask_a_only(j, cell_km=cell_km)
        return [
            _mask_spec(
                "I_AONLY",
                "A-only",
                df,
                fill="#2E3191",
                radius=radius,
                opacity=opacity,
            )
        ]

    if mode == "b_only":
        df = _mask_b_only(j, cell_km=cell_km)
        return [
            _mask_spec(
                "I_BONLY",
                "B-only",
                df,
                fill="#F28620",
                radius=radius,
                opacity=opacity,
            )
        ]

    if mode == "union":
        df = _mask_union(j, cell_km=cell_km)
        return [
            _mask_spec(
                "I_UNION",
                "Union",
                df,
                fill="#4B5563",
                radius=radius,
                opacity=opacity,
            )
        ]

    if mode in ("intersection", "inter"):
        df = _mask_inter(j, cell_km=cell_km)
        return [
            _mask_spec(
                "I_INTER",
                "Intersection",
                df,
                fill="#7B2CBF",
                radius=radius,
                opacity=opacity,
            )
        ]

    if mode == "delta":
        df, vmin, vmax, title = _delta(
            j,
            cell_km=cell_km,
            delta=str(cfg.delta or "a_minus_b"),
        )
        return [
            _delta_spec(
                df,
                vmin=vmin,
                vmax=vmax,
                title=title,
                radius=radius,
                opacity=opacity,
            )
        ]

    return []


# -------------------------
# Join + aggregation
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
    if df is None or df.empty:
        return pd.DataFrame()

    lon = df["lon"].to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    v = df["v"].to_numpy(dtype=float)

    m_per_deg = 111_320.0
    latr = np.deg2rad(lat)
    x_m = lon * m_per_deg * np.cos(latr)
    y_m = lat * m_per_deg

    cell_m = float(cell_km) * 1000.0
    gx = np.floor(x_m / cell_m).astype(int)
    gy = np.floor(y_m / cell_m).astype(int)

    tmp = pd.DataFrame(
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

    g = tmp.groupby(["gx", "gy"], sort=False)
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
    j = pd.merge(
        aa,
        bb,
        on=["gx", "gy"],
        how="outer",
    )
    return j


# -------------------------
# Masks
# -------------------------
def _pick_lonlat(j: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    lon = j["lon_a"].where(j["lon_a"].notna(), j["lon_b"])
    lat = j["lat_a"].where(j["lat_a"].notna(), j["lat_b"])
    return lon, lat


def _mk_sid(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["sid"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _mask_a_only(j: pd.DataFrame, *, cell_km: float) -> pd.DataFrame:
    m = j["v_a"].notna() & j["v_b"].isna()
    if not bool(m.any()):
        return pd.DataFrame(columns=["lon", "lat", "v", "tip"])
    lon, lat = _pick_lonlat(j[m])
    tip = _tip_mask("A-only", cell_km, j[m])
    df = pd.DataFrame({"lon": lon, "lat": lat, "v": 1.0, "tip": tip})
    return _mk_sid(df)


def _mask_b_only(j: pd.DataFrame, *, cell_km: float) -> pd.DataFrame:
    m = j["v_b"].notna() & j["v_a"].isna()
    if not bool(m.any()):
        return pd.DataFrame(columns=["lon", "lat", "v", "tip"])
    lon, lat = _pick_lonlat(j[m])
    tip = _tip_mask("B-only", cell_km, j[m])
    df = pd.DataFrame({"lon": lon, "lat": lat, "v": 1.0, "tip": tip})
    return _mk_sid(df)


def _mask_union(j: pd.DataFrame, *, cell_km: float) -> pd.DataFrame:
    m = j["v_a"].notna() | j["v_b"].notna()
    if not bool(m.any()):
        return pd.DataFrame(columns=["lon", "lat", "v", "tip"])
    lon, lat = _pick_lonlat(j[m])
    tip = _tip_mask("Union", cell_km, j[m])
    df = pd.DataFrame({"lon": lon, "lat": lat, "v": 1.0, "tip": tip})
    return _mk_sid(df)


def _mask_inter(j: pd.DataFrame, *, cell_km: float) -> pd.DataFrame:
    m = j["v_a"].notna() & j["v_b"].notna()
    if not bool(m.any()):
        return pd.DataFrame(columns=["lon", "lat", "v", "tip"])
    lon, lat = _pick_lonlat(j[m])
    tip = _tip_mask("Intersection", cell_km, j[m])
    df = pd.DataFrame({"lon": lon, "lat": lat, "v": 1.0, "tip": tip})
    return _mk_sid(df)


def _tip_mask(
    name: str,
    cell_km: float,
    jj: pd.DataFrame,
) -> List[str]:
    out: List[str] = []
    for _, r in jj.iterrows():
        a = r.get("v_a", np.nan)
        b = r.get("v_b", np.nan)
        txt = (
            f"{name}\n"
            f"cell={cell_km:.2f} km\n"
            f"A={_fmt(a)}  B={_fmt(b)}"
        )
        out.append(txt)
    return out


# -------------------------
# Delta
# -------------------------
def _delta(
    j: pd.DataFrame,
    *,
    cell_km: float,
    delta: str,
) -> Tuple[pd.DataFrame, float, float, str]:
    m = j["v_a"].notna() & j["v_b"].notna()
    if not bool(m.any()):
        df = pd.DataFrame(columns=["lon", "lat", "v", "tip"])
        return df, 0.0, 1.0, "Δ"

    jj = j[m].copy()

    lon = (jj["lon_a"] + jj["lon_b"]) / 2.0
    lat = (jj["lat_a"] + jj["lat_b"]) / 2.0

    d = str(delta or "a_minus_b").strip().lower()
    if d in ("b_minus_a", "b-a"):
        vv = jj["v_b"] - jj["v_a"]
        title = "Δ(B − A)"
    else:
        vv = jj["v_a"] - jj["v_b"]
        title = "Δ(A − B)"

    tip: List[str] = []
    for _, r in jj.iterrows():
        a = r.get("v_a", np.nan)
        b = r.get("v_b", np.nan)
        dv = (b - a) if title == "Δ(B − A)" else (a - b)
        txt = (
            f"{title}\n"
            f"cell={cell_km:.2f} km\n"
            f"A={_fmt(a)}  B={_fmt(b)}\n"
            f"Δ={_fmt(dv)}"
        )
        tip.append(txt)

    df = pd.DataFrame({"lon": lon, "lat": lat, "v": vv, "tip": tip})
    df = _mk_sid(df)

    x = pd.to_numeric(df["v"], errors="coerce").to_numpy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return df, 0.0, 1.0, title

    return df, float(x.min()), float(x.max()), title


def _fmt(v: object) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.3g}"


# -------------------------
# Specs
# -------------------------
def _mask_spec(
    lid: str,
    name: str,
    df: pd.DataFrame,
    *,
    fill: str,
    radius: int,
    opacity: float,
) -> LayerSpec:
    opts = {
        "stroke": "#111827",
        "fillMode": "fixed",
        "fillColor": str(fill),
        "shape": "square",
        "radius": int(max(3, min(14, radius))),
        "opacity": float(opacity),
        "pulse": False,
        "enableTooltip": True,
    }
    return LayerSpec(lid, name, df, opts, None)


def _delta_spec(
    df: pd.DataFrame,
    *,
    vmin: float,
    vmax: float,
    title: str,
    radius: int,
    opacity: float,
) -> LayerSpec:
    opts = {
        "stroke": "#111827",
        "fillMode": "value",
        "shape": "diamond",
        "radius": int(max(3, min(16, radius + 1))),
        "opacity": float(opacity),
        "pulse": False,
        "enableTooltip": True,
        "vmin": float(vmin),
        "vmax": float(vmax),
    }
    leg = {"title": title, "vmin": vmin, "vmax": vmax}
    return LayerSpec("I_DELTA", title, df, opts, leg)


def _zones_layers(
    j: pd.DataFrame,
    *,
    cell_km: float,
    radius: int,
    opacity: float,
) -> List[LayerSpec]:
    out: List[LayerSpec] = []

    a = _mask_a_only(j, cell_km=cell_km)
    b = _mask_b_only(j, cell_km=cell_km)
    i = _mask_inter(j, cell_km=cell_km)

    if not a.empty:
        out.append(
            _mask_spec(
                "I_AONLY",
                "A-only",
                a,
                fill="#2E3191",
                radius=radius,
                opacity=opacity,
            )
        )
    if not b.empty:
        out.append(
            _mask_spec(
                "I_BONLY",
                "B-only",
                b,
                fill="#F28620",
                radius=radius,
                opacity=opacity,
            )
        )
    if not i.empty:
        out.append(
            _mask_spec(
                "I_INTER",
                "Intersection",
                i,
                fill="#7B2CBF",
                radius=radius,
                opacity=opacity,
            )
        )

    return out
