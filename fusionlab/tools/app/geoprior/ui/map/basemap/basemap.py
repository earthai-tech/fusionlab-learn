# geoprior/ui/map/basemap/basemap.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
geoprior.ui.map.basemap.basemap

Central basemap registry.

Goals
-----
- Provide engine-specific basemap choices (UX differs).
- Resolve (engine, provider, style) -> spec dict.
- Fallback to a safe default when unsupported.

Engines
-------
- leaflet  : raster tiles via L.tileLayer(url, ...).
- maplibre : raster tiles via addSource(type=raster).
- google   : mapTypeId + optional styles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    "BasemapChoice",
    "engine_providers",
    "engine_styles",
    "resolve_basemap",
    "default_basemap",
]


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BasemapChoice:
    """A selectable basemap option for UI."""

    key: str
    label: str


# ---------------------------------------------------------------------
# Keys + labels (stable identifiers)
# ---------------------------------------------------------------------

_ENGINE_KEYS = ("leaflet", "maplibre", "google")

_PROVIDER_LABEL: Dict[str, str] = {
    "osm": "OSM",
    "terrain": "Terrain",
    "satellite": "Satellite",
    "hybrid": "Hybrid",
}

_STYLE_LABEL: Dict[str, str] = {
    "light": "Light",
    "dark": "Dark",
    "gray": "Gray",
}


def _norm(x: Any) -> str:
    return str(x or "").strip().lower()


def _eng(engine: str) -> str:
    e = _norm(engine)
    if e not in _ENGINE_KEYS:
        return "leaflet"
    return e


# ---------------------------------------------------------------------
# Raster tile URLs (Leaflet uses {s} and {r}; MapLibre uses arrays)
# ---------------------------------------------------------------------

_OSM_LF = (
    "https://{s}.tile.openstreetmap.org/"
    "{z}/{x}/{y}.png"
)
_OSM_ML = (
    "https://a.tile.openstreetmap.org/"
    "{z}/{x}/{y}.png",
    "https://b.tile.openstreetmap.org/"
    "{z}/{x}/{y}.png",
    "https://c.tile.openstreetmap.org/"
    "{z}/{x}/{y}.png",
)

_CARTO_DARK_LF = (
    "https://{s}.basemaps.cartocdn.com/"
    "dark_all/{z}/{x}/{y}{r}.png"
)
_CARTO_DARK_ML = (
    "https://a.basemaps.cartocdn.com/"
    "dark_all/{z}/{x}/{y}.png",
    "https://b.basemaps.cartocdn.com/"
    "dark_all/{z}/{x}/{y}.png",
    "https://c.basemaps.cartocdn.com/"
    "dark_all/{z}/{x}/{y}.png",
    "https://d.basemaps.cartocdn.com/"
    "dark_all/{z}/{x}/{y}.png",
)

_CARTO_LIGHT_LF = (
    "https://{s}.basemaps.cartocdn.com/"
    "light_all/{z}/{x}/{y}{r}.png"
)
_CARTO_LIGHT_ML = (
    "https://a.basemaps.cartocdn.com/"
    "light_all/{z}/{x}/{y}.png",
    "https://b.basemaps.cartocdn.com/"
    "light_all/{z}/{x}/{y}.png",
    "https://c.basemaps.cartocdn.com/"
    "light_all/{z}/{x}/{y}.png",
    "https://d.basemaps.cartocdn.com/"
    "light_all/{z}/{x}/{y}.png",
)

_TOPO_LF = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
_TOPO_ML = (
    "https://a.tile.opentopomap.org/"
    "{z}/{x}/{y}.png",
    "https://b.tile.opentopomap.org/"
    "{z}/{x}/{y}.png",
    "https://c.tile.opentopomap.org/"
    "{z}/{x}/{y}.png",
)

_ESRI_IMG = (
    "https://server.arcgisonline.com/"
    "ArcGIS/rest/services/World_Imagery/"
    "MapServer/tile/{z}/{y}/{x}"
)


# ---------------------------------------------------------------------
# Registry: (provider, style) -> spec per engine
# ---------------------------------------------------------------------
# Spec fields:
# - kind: "raster" | "google"
# - tiles: str or tuple[str,...]
# - attribution: str
# - subdomains: optional (leaflet)
# - retina: optional (leaflet)
# - google_type: optional (google)
# - google_styles: optional (google)
# ---------------------------------------------------------------------

_LEAFLET: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("osm", "light"): {
        "kind": "raster",
        "tiles": _OSM_LF,
        "subdomains": "abc",
        "retina": False,
        "attribution": "© OpenStreetMap",
    },
    ("osm", "dark"): {
        "kind": "raster",
        "tiles": _CARTO_DARK_LF,
        "subdomains": "abcd",
        "retina": True,
        "attribution": "© OpenStreetMap © CARTO",
    },
    ("osm", "gray"): {
        "kind": "raster",
        "tiles": _CARTO_LIGHT_LF,
        "subdomains": "abcd",
        "retina": True,
        "attribution": "© OpenStreetMap © CARTO",
    },
    ("terrain", "light"): {
        "kind": "raster",
        "tiles": _TOPO_LF,
        "subdomains": "abc",
        "retina": False,
        "attribution": "© OpenStreetMap © OpenTopoMap",
    },
    ("satellite", "light"): {
        "kind": "raster",
        "tiles": _ESRI_IMG,
        "subdomains": "",
        "retina": False,
        "attribution": "Tiles © Esri",
    },
}

_MAPLIBRE: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("osm", "light"): {
        "kind": "raster",
        "tiles": _OSM_ML,
        "attribution": "© OpenStreetMap",
    },
    ("osm", "dark"): {
        "kind": "raster",
        "tiles": _CARTO_DARK_ML,
        "attribution": "© OpenStreetMap © CARTO",
    },
    ("osm", "gray"): {
        "kind": "raster",
        "tiles": _CARTO_LIGHT_ML,
        "attribution": "© OpenStreetMap © CARTO",
    },
    ("terrain", "light"): {
        "kind": "raster",
        "tiles": _TOPO_ML,
        "attribution": "© OpenStreetMap © OpenTopoMap",
    },
    ("satellite", "light"): {
        "kind": "raster",
        "tiles": (_ESRI_IMG,),
        "attribution": "Tiles © Esri",
    },
}

# Minimal Google styles (optional; can expand later).
_GOOGLE_DARK = [
    {"elementType": "geometry", "stylers": [{"color": "#242f3e"}]},
    {"elementType": "labels.text.fill",
     "stylers": [{"color": "#746855"}]},
    {"elementType": "labels.text.stroke",
     "stylers": [{"color": "#242f3e"}]},
]
_GOOGLE_GRAY = [
    {"stylers": [{"saturation": -100}]},
]

_GOOGLE: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("osm", "light"): {
        "kind": "google",
        "google_type": "roadmap",
        "google_styles": None,
    },
    ("osm", "dark"): {
        "kind": "google",
        "google_type": "roadmap",
        "google_styles": _GOOGLE_DARK,
    },
    ("osm", "gray"): {
        "kind": "google",
        "google_type": "roadmap",
        "google_styles": _GOOGLE_GRAY,
    },
    ("terrain", "light"): {
        "kind": "google",
        "google_type": "terrain",
        "google_styles": None,
    },
    ("terrain", "dark"): {
        "kind": "google",
        "google_type": "terrain",
        "google_styles": _GOOGLE_DARK,
    },
    ("terrain", "gray"): {
        "kind": "google",
        "google_type": "terrain",
        "google_styles": _GOOGLE_GRAY,
    },
    ("satellite", "light"): {
        "kind": "google",
        "google_type": "satellite",
        "google_styles": None,
    },
    ("hybrid", "light"): {
        "kind": "google",
        "google_type": "hybrid",
        "google_styles": None,
    },
}


def default_basemap(engine: str) -> Tuple[str, str]:
    """
    Return safe (provider, style) default for an engine.
    """
    e = _eng(engine)
    if e == "google":
        return ("osm", "light")
    return ("osm", "light")


def engine_providers(engine: str) -> List[BasemapChoice]:
    """
    Provider choices for the given engine (UX differs).
    """
    e = _eng(engine)

    if e == "google":
        keys = ["osm", "terrain", "satellite", "hybrid"]
    else:
        keys = ["osm", "terrain", "satellite"]

    out: List[BasemapChoice] = []
    for k in keys:
        out.append(
            BasemapChoice(
                key=k,
                label=_PROVIDER_LABEL.get(k, k),
            )
        )
    return out


def engine_styles(engine: str, provider: str) -> List[BasemapChoice]:
    """
    Style choices for the given engine + provider.

    Leaflet/MapLibre:
      - osm: light/dark/gray
      - terrain/satellite: light only

    Google:
      - roadmap/terrain: light/dark/gray
      - satellite/hybrid: light only
    """
    e = _eng(engine)
    p = _norm(provider) or "osm"

    if e in ("leaflet", "maplibre"):
        if p == "osm":
            keys = ["light", "dark", "gray"]
        else:
            keys = ["light"]
    else:
        if p in ("osm", "terrain"):
            keys = ["light", "dark", "gray"]
        else:
            keys = ["light"]

    out: List[BasemapChoice] = []
    for k in keys:
        out.append(
            BasemapChoice(
                key=k,
                label=_STYLE_LABEL.get(k, k),
            )
        )
    return out


def resolve_basemap(
    engine: str,
    provider: str,
    style: str,
) -> Dict[str, Any]:
    """
    Resolve (engine, provider, style) into an engine spec.

    Always returns a valid spec by falling back to defaults.
    """
    e = _eng(engine)
    p = _norm(provider) or "osm"
    s = _norm(style) or "light"

    reg: Dict[Tuple[str, str], Dict[str, Any]]
    if e == "leaflet":
        reg = _LEAFLET
    elif e == "maplibre":
        reg = _MAPLIBRE
    else:
        reg = _GOOGLE

    spec = reg.get((p, s))
    if spec is None:
        # fallback: provider default style
        s0 = default_basemap(e)[1]
        spec = reg.get((p, s0))

    if spec is None:
        # fallback: engine default provider/style
        p0, s0 = default_basemap(e)
        spec = reg.get((p0, s0))

    if spec is None:
        # last resort: hard default
        spec = {"kind": "raster"}

    out = dict(spec)
    out["engine"] = e
    out["provider"] = p
    out["style"] = s
    return out
