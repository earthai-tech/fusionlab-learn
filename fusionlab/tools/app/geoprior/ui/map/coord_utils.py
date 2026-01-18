# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.coord_utils

Coordinate helpers for MapTab.

Leaflet markers need lon/lat degrees
(WGS84 / EPSG:4326).

If the data is UTM or any projected
EPSG, we convert to lon/lat first.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore


WGS84_EPSG = 4326

def parse_epsg(v: object) -> Optional[int]:
    if v is None:
        return None
    try:
        x = int(str(v).strip())
    except Exception:
        return None
    return x if x > 0 else None


@lru_cache(maxsize=32)
def _tr(src: int, dst: int) -> "Transformer":
    if Transformer is None:
        raise RuntimeError(
            "pyproj is required for EPSG/UTM reprojection."
        )
    return Transformer.from_crs(
        f"EPSG:{int(src)}",
        f"EPSG:{int(dst)}",
        always_xy=True,
    )


def ensure_lonlat(
    pts: pd.DataFrame,
    *,
    mode: str,
    utm_epsg: Optional[int] = None,
    src_epsg: Optional[int] = None,
    dst_epsg: int = WGS84_EPSG,
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Ensure pts has lon/lat in degrees.

    pts must have columns: lon, lat
    (even if they are x/y in meters).

    Returns
    -------
    (out, ok, msg)
    """
    if pts is None or pts.empty:
        return pts, True, ""

    m = str(mode or "lonlat").strip().lower()

    if m == "lonlat":
        out = _clip_lonlat(pts)
        ok = not out.empty
        msg = "" if ok else "No valid lon/lat points."
        return out, ok, msg

    if m == "utm":
        src = utm_epsg
    else:
        src = src_epsg

    src = parse_epsg(src)
    if src is None:
        return (
            pd.DataFrame(columns=pts.columns),
            False,
            "Missing source EPSG for coord reprojection.",
        )

    try:
        out = _reproject_xy(
            pts,
            src_epsg=int(src),
            dst_epsg=int(dst_epsg),
        )
    except Exception as e:
        return (
            pd.DataFrame(columns=pts.columns),
            False,
            f"Reprojection failed: {e}",
        )

    out = _clip_lonlat(out)
    ok = not out.empty
    msg = "" if ok else "No valid lon/lat after reprojection."
    return out, ok, msg


def _reproject_xy(
    pts: pd.DataFrame,
    *,
    src_epsg: int,
    dst_epsg: int,
) -> pd.DataFrame:
    out = pts.copy()

    x = pd.to_numeric(out["lon"], errors="coerce")
    y = pd.to_numeric(out["lat"], errors="coerce")

    ok = x.notna() & y.notna()
    if not bool(ok.any()):
        return pd.DataFrame(columns=out.columns)

    tr = _tr(int(src_epsg), int(dst_epsg))

    xx = x.to_numpy(dtype=float)
    yy = y.to_numpy(dtype=float)

    lon, lat = tr.transform(xx, yy)

    out["lon"] = lon
    out["lat"] = lat
    return out


def _clip_lonlat(pts: pd.DataFrame) -> pd.DataFrame:
    out = pts.copy()
    lon = pd.to_numeric(out["lon"], errors="coerce")
    lat = pd.to_numeric(out["lat"], errors="coerce")

    ok = (
        lon.between(-180.0, 180.0)
        & lat.between(-90.0, 90.0)
    )
    out = out[ok]
    return out


def df_to_lonlat(
    df: pd.DataFrame,
    *,
    x: str = "lon",
    y: str = "lat",
    mode: str = "lonlat",
    utm_epsg: Optional[int] = None,
    src_epsg: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert df[x], df[y] -> lon/lat degrees.

    Returns a new df with columns "lon","lat" replaced.
    If conversion is unsupported, returns empty df.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["lon", "lat", "v"])

    if x not in df.columns or y not in df.columns:
        return pd.DataFrame(columns=["lon", "lat", "v"])

    out = df.copy()
    xx = pd.to_numeric(out[x], errors="coerce").to_numpy(
        dtype=float
    )
    yy = pd.to_numeric(out[y], errors="coerce").to_numpy(
        dtype=float
    )

    lon, lat, ok = to_lonlat(
        xx,
        yy,
        mode=str(mode or "lonlat"),
        utm_epsg=utm_epsg,
        src_epsg=src_epsg,
    )
    if not ok:
        return pd.DataFrame(columns=["lon", "lat", "v"])

    out["lon"] = lon
    out["lat"] = lat

    out = out.dropna(subset=["lon", "lat"])

    ok_lon = out["lon"].between(-180.0, 180.0)
    ok_lat = out["lat"].between(-90.0, 90.0)
    out = out[ok_lon & ok_lat]

    return out


def to_lonlat(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    utm_epsg: Optional[int],
    src_epsg: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Convert arrays (x,y) into (lon,lat) degrees.
    """
    m = str(mode or "lonlat").strip().lower()

    if _looks_like_lonlat(x, y):
        return x, y, True

    if m == "lonlat":
        return x, y, True

    if m == "utm":
        if not utm_epsg:
            return x * np.nan, y * np.nan, False
        z, south = _utm_zone_hemi_from_epsg(utm_epsg)
        if z <= 0:
            return x * np.nan, y * np.nan, False
        lon, lat = utm_to_lonlat(x, y, z, south=south)
        return lon, lat, True

    if m == "epsg":
        if not src_epsg:
            return x * np.nan, y * np.nan, False
        if int(src_epsg) == 4326:
            return x, y, True
        z, south = _utm_zone_hemi_from_epsg(int(src_epsg))
        if z > 0:
            lon, lat = utm_to_lonlat(x, y, z, south=south)
            return lon, lat, True
        return x * np.nan, y * np.nan, False

    return x * np.nan, y * np.nan, False


def _looks_like_lonlat(x: np.ndarray, y: np.ndarray) -> bool:
    if x.size == 0 or y.size == 0:
        return False
    xx = x[np.isfinite(x)]
    yy = y[np.isfinite(y)]
    if xx.size == 0 or yy.size == 0:
        return False
    return (
        float(np.nanmax(np.abs(xx))) <= 180.0
        and float(np.nanmax(np.abs(yy))) <= 90.0
    )


def _utm_zone_hemi_from_epsg(epsg: int) -> Tuple[int, bool]:
    """
    EPSG 32601..32660 => north
    EPSG 32701..32760 => south
    """
    try:
        e = int(epsg)
    except Exception:
        return 0, False

    if 32601 <= e <= 32660:
        return e - 32600, False
    if 32701 <= e <= 32760:
        return e - 32700, True
    return 0, False


def utm_to_lonlat(
    e: np.ndarray,
    n: np.ndarray,
    zone: int,
    *,
    south: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized UTM -> lon/lat (WGS84).
    """
    a = 6378137.0
    ecc = 0.00669438
    k0 = 0.9996

    x = e.astype(float) - 500000.0
    y = n.astype(float)
    if south:
        y = y - 10000000.0

    z = int(zone)
    lon0 = (z - 1) * 6.0 - 180.0 + 3.0

    ecc_p = ecc / (1.0 - ecc)
    m = y / k0

    mu = m / (
        a
        * (
            1.0
            - ecc / 4.0
            - 3.0 * ecc * ecc / 64.0
            - 5.0 * ecc * ecc * ecc / 256.0
        )
    )

    e1 = (1.0 - np.sqrt(1.0 - ecc)) / (
        1.0 + np.sqrt(1.0 - ecc)
    )

    j1 = 3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0
    j2 = 21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0
    j3 = 151.0 * e1**3 / 96.0
    j4 = 1097.0 * e1**4 / 512.0

    fp = (
        mu
        + j1 * np.sin(2.0 * mu)
        + j2 * np.sin(4.0 * mu)
        + j3 * np.sin(6.0 * mu)
        + j4 * np.sin(8.0 * mu)
    )

    sinf = np.sin(fp)
    cosf = np.cos(fp)

    t1 = np.tan(fp) ** 2
    c1 = ecc_p * cosf**2

    r1 = a * (1.0 - ecc) / (1.0 - ecc * sinf**2) ** 1.5
    n1 = a / np.sqrt(1.0 - ecc * sinf**2)

    d = x / (n1 * k0)

    q1 = n1 * np.tan(fp) / r1

    q2 = (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2
           - 9.0 * ecc_p) * d**4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1**2
           - 252.0 * ecc_p - 3.0 * c1**2) * d**6 / 720.0
    )

    lat = fp - q1 * q2

    q3 = (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2
           + 8.0 * ecc_p + 24.0 * t1**2) * d**5 / 120.0
    )

    lon = np.deg2rad(lon0) + q3 / cosf

    return np.rad2deg(lon), np.rad2deg(lat)
