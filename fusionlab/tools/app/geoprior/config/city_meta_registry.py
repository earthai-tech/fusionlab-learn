# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.config.cities.meta

Tiny, file-driven city metadata registry.

Goal
----
Provide optional city metadata (lat/lon + EPSG hints) that
the UI can load at startup so Map/Xfer can:
- auto-fill xfer.city_{a,b}_{lat,lon}
- auto-guess UTM EPSG if needed
- avoid asking users to hardcode city settings

This file stays dependency-light (pure python).
Line length target: <= 62 chars (black config).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


__all__ = [
    "CityMeta",
    "CityMetaIndex",
    "slug_city",
    "load_city_meta_file",
    "load_city_meta_merged",
    "install_city_meta",
    "lookup_city_meta",
    "apply_city_meta_to_store",
]


_SLUG_RE = re.compile(r"[^a-z0-9]+", re.I)


def slug_city(name: str) -> str:
    """Normalize a city key for stable matching."""
    s = str(name or "").strip().lower()
    s = _SLUG_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _norm_alias(s: str) -> str:
    return slug_city(s)


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


@dataclass(frozen=True)
class CityMeta:
    key: str
    name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    utm_epsg: Optional[int] = None
    src_epsg: Optional[int] = None
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CityMetaIndex:
    """Index with alias lookup."""
    cities: Dict[str, CityMeta]
    alias_to_key: Dict[str, str]

    def keys(self) -> List[str]:
        return list(self.cities.keys())

    def get(self, key_or_alias: str) -> Optional[CityMeta]:
        k = slug_city(key_or_alias)
        if k in self.cities:
            return self.cities[k]
        k2 = self.alias_to_key.get(k)
        if k2:
            return self.cities.get(k2)
        return None


def _parse_payload(obj: Any) -> List[Dict[str, Any]]:
    """Accept either {cities:[...]} or a list."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        if "cities" in obj and isinstance(obj["cities"], list):
            return list(obj["cities"])
        if "cities" in obj and isinstance(obj["cities"], dict):
            out: List[Dict[str, Any]] = []
            for k, v in obj["cities"].items():
                if isinstance(v, dict):
                    vv = dict(v)
                    vv.setdefault("id", k)
                    out.append(vv)
            return out
        if "items" in obj and isinstance(obj["items"], list):
            return list(obj["items"])
        return []
    if isinstance(obj, list):
        return list(obj)
    return []


def load_city_meta_file(path: Path) -> CityMetaIndex:
    """
    Load one JSON file into an index.

    JSON formats supported
    ----------------------
    A) Dict container:
        {
          "version": 1,
          "cities": [
            {"id": "nansha", "name": "...",
             "lat": 22.6, "lon": 113.6,
             "utm_epsg": 32649,
             "aliases": ["Nansha", "南沙"]}
          ]
        }

    B) List:
        [
          {"id": "...", "name": "...", ...},
          ...
        ]

    Notes
    -----
    - Unknown fields are ignored.
    - id/key/name are normalized into `key`.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return CityMetaIndex(cities={}, alias_to_key={})

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return CityMetaIndex(cities={}, alias_to_key={})

    items = _parse_payload(data)

    cities: Dict[str, CityMeta] = {}
    alias_to_key: Dict[str, str] = {}

    for row in items:
        if not isinstance(row, dict):
            continue

        raw_id = (
            row.get("id")
            or row.get("key")
            or row.get("slug")
            or row.get("name")
            or ""
        )
        name = str(row.get("name") or raw_id or "").strip()
        key = slug_city(raw_id)

        lat = _as_float(row.get("lat"))
        lon = _as_float(row.get("lon"))

        utm_epsg = _as_int(row.get("utm_epsg"))
        src_epsg = _as_int(row.get("src_epsg") or row.get("epsg"))

        aliases = row.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []

        als: List[str] = []
        for a in aliases:
            s = str(a or "").strip()
            if s:
                als.append(s)

        meta = CityMeta(
            key=key,
            name=name or key,
            lat=lat,
            lon=lon,
            utm_epsg=utm_epsg,
            src_epsg=src_epsg,
            aliases=tuple(als),
        )
        cities[key] = meta

        # alias index
        alias_to_key[_norm_alias(key)] = key
        if name:
            alias_to_key[_norm_alias(name)] = key
        for a in als:
            alias_to_key[_norm_alias(a)] = key

    return CityMetaIndex(
        cities=cities,
        alias_to_key=alias_to_key,
    )


def load_city_meta_merged(
    paths: Iterable[Path],
) -> CityMetaIndex:
    """Merge multiple files, later files override."""
    merged: Dict[str, CityMeta] = {}
    alias_to_key: Dict[str, str] = {}

    for p in (paths or []):
        idx = load_city_meta_file(Path(p))
        for k, v in idx.cities.items():
            merged[k] = v

        # rebuild aliases at end for consistency
    for k, v in merged.items():
        alias_to_key[_norm_alias(k)] = k
        alias_to_key[_norm_alias(v.name)] = k
        for a in v.aliases:
            alias_to_key[_norm_alias(a)] = k

    return CityMetaIndex(
        cities=merged,
        alias_to_key=alias_to_key,
    )


def install_city_meta(
    store: Any,
    *,
    paths: Iterable[Path],
    store_key: str = "cities.meta",
) -> CityMetaIndex:
    """
    Load meta JSON files and store the index payload.

    This is safe to call during app startup.

    The store payload is a dict:
      {
        "cities": {key: {...}},
        "aliases": {alias: key},
      }
    """
    idx = load_city_meta_merged(paths)

    payload: Dict[str, Any] = {
        "cities": {
            k: {
                "key": v.key,
                "name": v.name,
                "lat": v.lat,
                "lon": v.lon,
                "utm_epsg": v.utm_epsg,
                "src_epsg": v.src_epsg,
                "aliases": list(v.aliases),
            }
            for k, v in idx.cities.items()
        },
        "aliases": dict(idx.alias_to_key),
    }

    try:
        store.set(store_key, payload)
    except Exception:
        # Store may not exist in unit tests.
        pass

    return idx


def lookup_city_meta(
    store: Any,
    city: str,
    *,
    store_key: str = "cities.meta",
) -> Optional[Dict[str, Any]]:
    """Find city meta in store by key or alias."""
    payload = None
    try:
        payload = store.get(store_key, None)
    except Exception:
        payload = None

    if not isinstance(payload, dict):
        return None

    cities = payload.get("cities") or {}
    aliases = payload.get("aliases") or {}
    if not isinstance(cities, dict):
        return None
    if not isinstance(aliases, dict):
        aliases = {}

    k = slug_city(city)
    if k in cities and isinstance(cities[k], dict):
        return cities[k]

    k2 = aliases.get(k)
    if k2 and (k2 in cities) and isinstance(cities[k2], dict):
        return cities[k2]

    return None


def apply_city_meta_to_store(
    store: Any,
    *,
    side: str,
    city: str,
    force: bool = False,
) -> bool:
    """
    Apply lat/lon hints to xfer.city_{a,b}_{lat,lon}.

    Parameters
    ----------
    side : {"a","b"}
        Which city slot to update.
    city : str
        City name chosen by the user.
    force : bool
        Override existing lat/lon values.

    Returns
    -------
    ok : bool
        True if any value was written.
    """
    sd = str(side or "").strip().lower()
    if sd not in ("a", "b"):
        return False

    meta = lookup_city_meta(store, city)
    if not meta:
        return False

    lat = meta.get("lat")
    lon = meta.get("lon")

    k_lat = f"xfer.city_{sd}_lat"
    k_lon = f"xfer.city_{sd}_lon"

    changed = False
    try:
        cur_lat = store.get(k_lat, None)
        cur_lon = store.get(k_lon, None)
    except Exception:
        cur_lat, cur_lon = None, None

    if force or (cur_lat is None and lat is not None):
        try:
            store.set(k_lat, lat)
            changed = True
        except Exception:
            pass

    if force or (cur_lon is None and lon is not None):
        try:
            store.set(k_lon, lon)
            changed = True
        except Exception:
            pass

    return changed
