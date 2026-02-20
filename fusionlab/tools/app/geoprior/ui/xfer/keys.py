# geoprior/ui/xfer/keys.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.keys

Central place for xfer.* + xfer.map.* keys and defaults.
"""

from __future__ import annotations

from typing import Dict, Set

# -------------------------
# Existing tab keys
# -------------------------
K_VIEW_MODE = "xfer.view_mode"

# -------------------------
# Map page keys (new)
# -------------------------
K_MAP_SPLIT = "xfer.map.split"
K_MAP_VALUE = "xfer.map.value"
K_MAP_OVERLAY = "xfer.map.overlay"
K_MAP_SHARED = "xfer.map.shared_scale"

K_MAP_TIME_MODE = "xfer.map.time_mode"
K_MAP_STEP = "xfer.map.step"
K_MAP_YEAR = "xfer.map.year"

K_MAP_MAX_POINTS = "xfer.map.max_points"
K_MAP_RADIUS = "xfer.map.radius"
K_MAP_OPACITY = "xfer.map.opacity"
K_MAP_CLIP_PCT = "xfer.map.clip_pct"

K_MAP_POINTS_MODE = "xfer.map.points_mode"
K_MAP_MARKER_SHAPE = "xfer.map.marker_shape"
K_MAP_MARKER_SIZE = "xfer.map.marker_size"

K_MAP_HOTSPOT_TOPN = "xfer.map.hotspot.top_n"
K_MAP_HOTSPOT_MIN_SEP_KM = "xfer.map.hotspot.min_sep_km"
K_MAP_HOTSPOT_METRIC = "xfer.map.hotspot.metric"
K_MAP_HOTSPOT_QUANTILE = "xfer.map.hotspot.quantile"

K_MAP_ANIM_PULSE = "xfer.map.anim.pulse"
K_MAP_ANIM_PLAY_MS = "xfer.map.anim.play_ms"

K_MAP_COORD_MODE = "xfer.map.coord_mode"
K_MAP_UTM_EPSG = "xfer.map.utm_epsg"
K_MAP_SRC_EPSG = "xfer.map.src_epsg"

# City A dataset selection
K_MAP_A_JOB_KIND = "xfer.map.a.job_kind"
K_MAP_A_JOB_ID = "xfer.map.a.job_id"
K_MAP_A_FILE = "xfer.map.a.file"

# City B dataset selection
K_MAP_B_JOB_KIND = "xfer.map.b.job_kind"
K_MAP_B_JOB_ID = "xfer.map.b.job_id"
K_MAP_B_FILE = "xfer.map.b.file"
K_MAP_EXPANDED = "xfer.map.expanded"

# Map insight (transferability badge)
K_MAP_INSIGHT = "xfer.map.insight"

K_MAP_INTERACTION = "xfer.map.interaction"
K_MAP_INT_CELL_KM = "xfer.map.int.cell_km"
K_MAP_INT_AGG = "xfer.map.int.agg"
K_MAP_INT_DELTA = "xfer.map.int.delta"

# extras toggles

K_MAP_INT_HOT_ENABLE = "xfer.map.int.hot.enable"
K_MAP_INT_HOT_TOPN = "xfer.map.int.hot.topn"
K_MAP_INT_HOT_METRIC = "xfer.map.int.hot.metric"
K_MAP_INT_HOT_Q = "xfer.map.int.hot.q"
K_MAP_INT_HOT_SEP = "xfer.map.int.hot.sep_km"

K_MAP_INT_INTENS_ENABLE = "xfer.map.int.intens.enable"

K_MAP_INT_BUF_ENABLE = "xfer.map.int.buf.enable"
K_MAP_INT_BUF_K = "xfer.map.int.buf.k"

K_MAP_INTERP_HTML = "xfer.map.interp_html"
K_MAP_INTERP_TIP = "xfer.map.interp_tip"

# xfer/map/keys.py (additions)

K_MAP_RADAR_ENABLE = "xfer.map.radar.enable"
K_MAP_RADAR_TARGET = "xfer.map.radar.target"        # overlay|both|a|b
K_MAP_RADAR_ORDER = "xfer.map.radar.order"          # score|abs|rank
K_MAP_RADAR_DWELL_MS = "xfer.map.radar.dwell_ms"    # int
K_MAP_RADAR_RADIUS_KM = "xfer.map.radar.radius_km"  # float
K_MAP_RADAR_RINGS = "xfer.map.radar.rings"          # int

K_MAP_LINKS_ENABLE = "xfer.map.links.enable"
K_MAP_LINKS_MODE = "xfer.map.links.mode"            # nearest|rank|knn
K_MAP_LINKS_K = "xfer.map.links.k"                  # int
K_MAP_LINKS_MAX = "xfer.map.links.max"              # int
K_MAP_LINKS_SHOW_DIST = "xfer.map.links.show_dist"  # bool

# Basemap / provider
K_MAP_BASEMAP = "xfer.map.basemap"

# Specific EPSG overrides for A and B
K_MAP_A_EPSG = "xfer.map.a.epsg"
K_MAP_B_EPSG = "xfer.map.b.epsg"

# NEW: Path to loaded meta file
K_CITIES_META_PATH = "cities.meta.path"

BASEMAP_CHOICES = (
    ("osm", "OpenStreetMap"),
    ("esri_sat", "Esri Satellite"),
    ("esri_topo", "Esri Topo"),
    ("esri_terrain", "Esri Terrain"),
    ("opentopo", "OpenTopoMap"),
    ("carto_light", "Carto Light"),
    ("carto_dark", "Carto Dark"),
)

# Add just below BASEMAP_CHOICES

BASEMAP_ICON_ALIASES: Dict[str, str] = {
    "osm": "bm_osm.svg",
    "esri_sat": "bm_sat.svg",
    "esri_topo": "bm_topo.svg",
    "esri_terrain": "bm_terrain.svg",
    "opentopo": "bm_opentopo.svg",
    "carto_light": "bm_carto_light.svg",
    "carto_dark": "bm_carto_dark.svg",
}


def basemap_icon_name(bid: str) -> str:
    # Falls back to a convention if you later rename icons
    # as bm_<bid>.svg
    return BASEMAP_ICON_ALIASES.get(
        bid,
        f"bm_{bid}.svg",
    )

DEFAULTS: Dict[str, object] = {
    K_VIEW_MODE: "map",
    K_MAP_SPLIT: "val",
    K_MAP_VALUE: "auto",
    K_MAP_OVERLAY: "both",
    K_MAP_SHARED: True,
    K_MAP_TIME_MODE: "forecast_step",
    K_MAP_STEP: 1,
    K_MAP_YEAR: None,
    K_MAP_MAX_POINTS: 15000,
    K_MAP_RADIUS: 6,
    K_MAP_OPACITY: 0.90,
    K_MAP_CLIP_PCT: 0.0,
    K_MAP_POINTS_MODE: "all",        # all|hotspots|hotspots_plus
    K_MAP_MARKER_SHAPE: "auto",      # auto|circle|triangle|diamond|square
    K_MAP_MARKER_SIZE: 6,
    
    K_MAP_HOTSPOT_TOPN: 8,
    K_MAP_HOTSPOT_MIN_SEP_KM: 2.0,
    K_MAP_HOTSPOT_METRIC: "abs",     # value|abs|high|low
    K_MAP_HOTSPOT_QUANTILE: 0.98,
    K_MAP_EXPANDED: False,
    
    K_MAP_ANIM_PULSE: True,
    K_MAP_ANIM_PLAY_MS: 320,
    K_MAP_COORD_MODE: "auto",
    K_MAP_UTM_EPSG: None,
    K_MAP_SRC_EPSG: None,
    K_MAP_A_JOB_KIND: None,
    K_MAP_A_JOB_ID: None,
    K_MAP_A_FILE: "",
    K_MAP_B_JOB_KIND: None,
    K_MAP_B_JOB_ID: None,
    K_MAP_B_FILE: "",
    K_MAP_INSIGHT: False,
    
    K_MAP_INTERACTION: "none",
    K_MAP_INT_CELL_KM: 2.0,
    K_MAP_INT_AGG: "mean",
    K_MAP_INT_DELTA: "a_minus_b",
    
    K_MAP_INTERP_HTML: "",
    K_MAP_INTERP_TIP: "",
    
    K_MAP_INT_HOT_ENABLE: False,
    K_MAP_INT_HOT_TOPN: 8,
    K_MAP_INT_HOT_METRIC: "abs",
    K_MAP_INT_HOT_Q: 0.98,
    K_MAP_INT_HOT_SEP: 2.0,
    K_MAP_INT_INTENS_ENABLE: False,
    K_MAP_INT_BUF_ENABLE: False,
    K_MAP_INT_BUF_K: 1,
    
    K_MAP_RADAR_ENABLE: False,
    K_MAP_RADAR_TARGET: "overlay",   # overlay|both|a|b
    K_MAP_RADAR_ORDER: "score",      # score|abs|rank
    K_MAP_RADAR_DWELL_MS: 520,
    K_MAP_RADAR_RADIUS_KM: 8.0,
    K_MAP_RADAR_RINGS: 3,

    K_MAP_LINKS_ENABLE: False,
    K_MAP_LINKS_MODE: "nearest",     # nearest|rank|knn
    K_MAP_LINKS_K: 1,
    K_MAP_LINKS_MAX: 12,
    K_MAP_LINKS_SHOW_DIST: True,
    
    # Basemap / provider
    K_MAP_BASEMAP: "osm",
    
}


def map_keys() -> Set[str]:
    return {
        K_MAP_SPLIT,
        K_MAP_VALUE,
        K_MAP_OVERLAY,
        K_MAP_SHARED,
        K_MAP_TIME_MODE,
        K_MAP_STEP,
        K_MAP_YEAR,
        K_MAP_MAX_POINTS,
        K_MAP_RADIUS,
        K_MAP_OPACITY,
        K_MAP_CLIP_PCT,
        K_MAP_POINTS_MODE,
        K_MAP_MARKER_SHAPE,
        K_MAP_MARKER_SIZE,
        K_MAP_HOTSPOT_TOPN,
        K_MAP_HOTSPOT_MIN_SEP_KM,
        K_MAP_HOTSPOT_METRIC,
        K_MAP_HOTSPOT_QUANTILE,
        K_MAP_ANIM_PULSE,
        K_MAP_ANIM_PLAY_MS,
        K_MAP_COORD_MODE,
        K_MAP_UTM_EPSG,
        K_MAP_SRC_EPSG,
        K_MAP_A_JOB_KIND,
        K_MAP_A_JOB_ID,
        K_MAP_A_FILE,
        K_MAP_B_JOB_KIND,
        K_MAP_B_JOB_ID,
        K_MAP_B_FILE,
        K_MAP_INSIGHT,
        K_MAP_INTERACTION,
        K_MAP_INT_CELL_KM,
        K_MAP_INT_AGG,
        K_MAP_INT_DELTA,
        K_MAP_INT_HOT_ENABLE,
        K_MAP_INT_HOT_TOPN,
        K_MAP_INT_HOT_METRIC,
        K_MAP_INT_HOT_Q,
        K_MAP_INT_HOT_SEP,
        K_MAP_INT_INTENS_ENABLE,
        K_MAP_INT_BUF_ENABLE,
        K_MAP_INT_BUF_K,
        
        K_MAP_RADAR_ENABLE,
        K_MAP_RADAR_TARGET,
        K_MAP_RADAR_ORDER,
        K_MAP_RADAR_DWELL_MS,
        K_MAP_RADAR_RADIUS_KM,
        K_MAP_RADAR_RINGS,
        K_MAP_LINKS_ENABLE,
        K_MAP_LINKS_MODE,
        K_MAP_LINKS_K,
        K_MAP_LINKS_MAX,
        K_MAP_LINKS_SHOW_DIST,
        
        K_MAP_BASEMAP,
        
        K_MAP_A_EPSG,
        K_MAP_B_EPSG,
        K_CITIES_META_PATH,
    }
