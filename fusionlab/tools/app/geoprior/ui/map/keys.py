# geoprior/ui/map/keys.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
geoprior.ui.map.keys

Central store keys for map UI.
Keeps view_panel/controller in sync.
"""

from __future__ import annotations
from typing import Dict, List
from ..view.keys import (
    DEFAULTS as _VIEW_RENDER_DEFAULTS,
    K_PLOT_KIND as _VK_PLOT_KIND,
    K_PLOT_CMAP as _VK_PLOT_CMAP,
    K_PLOT_OPACITY as _VK_PLOT_OPACITY,
    K_HEX_GRIDSIZE as _VK_HEX_GRIDSIZE,
    K_HEX_METRIC as _VK_HEX_METRIC,
    K_FILTER_ENABLE as _VK_FILTER_ENABLE,
    K_FILTER_V_MIN as _VK_FILTER_V_MIN,
    K_FILTER_V_MAX as _VK_FILTER_V_MAX,
    K_SPACE_MODE as _VK_SPACE_MODE,
    K_SPACE_RADIUS as _VK_SPACE_RADIUS,
    K_CONTOUR_BANDWIDTH as _VK_CONTOUR_BANDWIDTH,
    K_CONTOUR_STEPS as _VK_CONTOUR_STEPS,
    K_CONTOUR_FILLED as _VK_CONTOUR_FILLED,
    K_CONTOUR_LABELS as _VK_CONTOUR_LABELS,
)
# -------------------------
# Engine key candidates
# -------------------------
ENGINE_KEYS: List[str] = [
    "map.engine",
    "map.engine.active",
    "map.canvas.engine",
    "map.view.engine",
]

def get_engine(store, default: str = "leaflet") -> str:
    """
    Best-effort engine lookup from store.
    """
    for k in ENGINE_KEYS:
        v = store.get(k, None)
        if v is None:
            continue
        s = str(v).strip().lower()
        if s:
            return s
    return str(default).strip().lower()

MAP_VIEW_PREFIX = "map."

_VIEW_KEY_ALIASES: Dict[str, str] = {
    # keep your existing map keys for these:
    _VK_PLOT_CMAP: "map.view.colormap",
    _VK_PLOT_OPACITY: "map.view.marker_opacity",
}

def map_view_key(k: str) -> str:
    """
    Map a canonical 'view.*' key into map namespace.

    Uses aliases for legacy map keys where needed.
    """
    if k.startswith("map."):
        return k
    ali = _VIEW_KEY_ALIASES.get(k)
    if ali:
        return ali
    return MAP_VIEW_PREFIX + k

# --- exported map-namespaced constants ---
K_PLOT_KIND = map_view_key(_VK_PLOT_KIND)
K_PLOT_CMAP = map_view_key(_VK_PLOT_CMAP)
K_PLOT_OPACITY = map_view_key(_VK_PLOT_OPACITY)

K_HEX_GRIDSIZE = map_view_key(_VK_HEX_GRIDSIZE)
K_HEX_METRIC = map_view_key(_VK_HEX_METRIC)

K_FILTER_ENABLE = map_view_key(_VK_FILTER_ENABLE)
K_FILTER_V_MIN = map_view_key(_VK_FILTER_V_MIN)
K_FILTER_V_MAX = map_view_key(_VK_FILTER_V_MAX)

K_SPACE_MODE = map_view_key(_VK_SPACE_MODE)
K_SPACE_RADIUS = map_view_key(_VK_SPACE_RADIUS)

K_CONTOUR_BANDWIDTH = map_view_key(_VK_CONTOUR_BANDWIDTH)
K_CONTOUR_STEPS = map_view_key(_VK_CONTOUR_STEPS)
K_CONTOUR_FILLED = map_view_key(_VK_CONTOUR_FILLED)
K_CONTOUR_LABELS = map_view_key(_VK_CONTOUR_LABELS)


MAP_UTM_EPSG="map.utm_epsg"
MAP_COORD_EPSG="map.coord_epsg"
MAP_SRC_EPSG="map.src_epsg"

# -------------------------
# Global Keys: Map Core
# -------------------------
MAP_ENGINE = "map.engine"
MAP_COORD_MODE = "map.coord_mode"
MAP_GOOGLE_API_KEY = "map.google_api_key"
MAP_X_COL = "map.x_col"
MAP_Y_COL = "map.y_col"
MAP_Z_COL = "map.z_col"
MAP_FOCUS_MODE = "map.focus_mode"
MAP_SHOW_ANALYTICS = "map.show_analytics"
MAP_DATA_SOURCE = "map.data_source"
MAP_MANUAL_FILES = "map.manual_files"
MAP_SELECTED_FILES = "map.selected_files"
MAP_ACTIVE_FILE = "map.active_file"
MAP_TIME_COL = "map.time_col"
MAP_STEP_COL = "map.step_col"
MAP_VALUE_COL = "map.value_col"
MAP_VALUE_UNIT = "map.value_unit"
MAP_TIME_VALUE = "map.time_value"
MAP_ID_COL = "map.id_col"
MAP_OBS_COL = "map.obs_col"
MAP_CLICK_SAMPLE_IDX = "map.click_sample_idx"
MAP_BOOKMARKS = "map.bookmarks"
MAP_MEASURE_MODE = "map.measure_mode"

# -------------------------
# Derived Keys: Controller
# -------------------------
# Produced by ui/map/controller.py
# Consumed by analytics_panel.py (and optionally MapTab)

MAP_DF_ALL = "map.df_all"
MAP_DF_FRAME = "map.df_frame"
MAP_DF_POINTS = "map.df_points"

# -------------------------
# Global Keys: Map View (Visuals)
# -------------------------
MAP_VIEW_BASEMAP = "map.view.basemap"
MAP_VIEW_SHOW_GRID = "map.view.show_grid"
MAP_VIEW_SHOW_COLORBAR = "map.view.show_colorbar"
MAP_VIEW_BASEMAP_STYLE = "map.view.basemap_style"
MAP_VIEW_TILES_OPACITY = "map.view.tiles_opacity"
MAP_VIEW_COLORMAP = "map.view.colormap"
MAP_VIEW_CMAP_INVERT = "map.view.cmap_invert"
MAP_VIEW_AUTOSCALE = "map.view.autoscale"
MAP_VIEW_VMIN = "map.view.vmin"
MAP_VIEW_VMAX = "map.view.vmax"
MAP_VIEW_MARKER_SIZE = "map.view.marker_size"
MAP_VIEW_MARKER_OPACITY = "map.view.marker_opacity"
MAP_VIEW_CLIP_MODE = "map.view.clip_mode"
MAP_VIEW_SHOW_LEGEND = "map.view.show_legend"
MAP_VIEW_LEGEND_POS = "map.view.legend_pos"

MAP_ANALYTICS_HEIGHT="map.analytics.height"
MAP_ANALYTICS_PINNED ="map.analytics.pinned"
# -------------------------
# Global Keys: Tooltab
# -------------------------
MAP_TOOLTAB_PINNED = "map.tooltab.pinned"

# -------------------------
# Global Keys: Sampling
# -------------------------
MAP_SAMPLING_MODE = "map.sampling.mode"
MAP_SAMPLING_METHOD = "map.sampling.method"
MAP_SAMPLING_MAX_POINTS = "map.sampling.max_points"
MAP_SAMPLING_SEED = "map.sampling.seed"
MAP_SAMPLING_CELL_KM = "map.sampling.cell_km"
MAP_SAMPLING_MAX_PER_CELL = "map.sampling.max_per_cell"
MAP_SAMPLING_APPLY_HOTSPOTS = "map.sampling.apply_hotspots"

# -------------------------
# Global Keys: Hotspots
# -------------------------
MAP_VIEW_HOTSPOTS_ENABLED = "map.view.hotspots.enabled"
MAP_VIEW_HOTSPOTS_MODE = "map.view.hotspots.mode"
MAP_VIEW_HOTSPOTS_METHOD = "map.view.hotspots.method"
MAP_VIEW_HOTSPOTS_METRIC = "map.view.hotspots.metric"
MAP_VIEW_HOTSPOTS_TIME_AGG = "map.view.hotspots.time_agg"
MAP_VIEW_HOTSPOTS_THR_MODE = "map.view.hotspots.thr_mode"
MAP_VIEW_HOTSPOTS_QUANTILE = "map.view.hotspots.quantile"
MAP_VIEW_HOTSPOTS_ABS_THR = "map.view.hotspots.abs_thr"
MAP_VIEW_HOTSPOTS_CELL_KM = "map.view.hotspots.cell_km"
MAP_VIEW_HOTSPOTS_MIN_PTS = "map.view.hotspots.min_pts"
MAP_VIEW_HOTSPOTS_MAX_N = "map.view.hotspots.max_n"
MAP_VIEW_HOTSPOTS_MIN_SEP_KM = "map.view.hotspots.min_sep_km"
MAP_VIEW_HOTSPOTS_STYLE = "map.view.hotspots.style"
MAP_VIEW_HOTSPOTS_PULSE = "map.view.hotspots.pulse"
MAP_VIEW_HOTSPOTS_PULSE_SPEED = "map.view.hotspots.pulse_speed"
MAP_VIEW_HOTSPOTS_RING_KM = "map.view.hotspots.ring_km"
MAP_VIEW_HOTSPOTS_LABELS = "map.view.hotspots.labels"

# -------------------------
# Global Keys: Interpretation
# -------------------------
MAP_VIEW_INTERP_ENABLED = "map.view.interp.enabled"
MAP_VIEW_INTERP_SCHEME = "map.view.interp.scheme"
MAP_VIEW_INTERP_CALLOUTS = "map.view.interp.callouts"
MAP_VIEW_INTERP_CALLOUT_LEVEL = "map.view.interp.callout_level"
MAP_VIEW_INTERP_CALLOUT_ACTIONS = "map.view.interp.callout_actions"
MAP_VIEW_INTERP_ACTION_PACK = "map.view.interp.action_pack"
MAP_VIEW_INTERP_ACTION_INTENSITY = "map.view.interp.action_intensity"
MAP_VIEW_INTERP_TONE = "map.view.interp.tone"
MAP_VIEW_INTERP_SUMMARY = "map.view.interp.summary"


# Propagation / Simulation Options
K_PROP_ENABLED = "map.view.prop.enabled"        # bool
K_PROP_YEARS = "map.view.prop.years_extra"      # int (e.g. 5 future years)
K_PROP_SPEED = "map.view.prop.speed"            # int (ms per frame)
K_PROP_MODE = "map.view.prop.mode"              # absolute | differential | risk_mask
K_PROP_VECTORS = "map.view.prop.show_vectors"   # bool (show expansion arrows)
K_PROP_LOOP = "map.view.prop.loop"              # bool

# Extra derived (canonical full-history points)
MAP_DF_ALL_POINTS = "map.df_all_points"

# Propagation derived (controller writes these)
MAP_PROP_BUILD_ID = "map.prop.build_id"   # int nonce
MAP_PROP_YEAR = "map.prop.year"          # int
MAP_PROP_TIMELINE = "map.prop.timeline"  # list[int]

MAP_PROP_DF_ALL = "map.prop.df_all"
MAP_PROP_DF_FRAME = "map.prop.df_frame"
MAP_PROP_DF_POINTS = "map.prop.df_points"

MAP_PROP_VMIN = "map.prop.vmin"
MAP_PROP_VMAX = "map.prop.vmax"
MAP_PROP_VECTORS = "map.prop.vectors"

# Legend policy (view key)
K_PROP_LEGEND = "map.view.prop.legend"   # global|frame|fixed


# --- Alert System Keys ---
K_ALERT_ENABLED = "map.view.alerts.enabled"
K_ALERT_TRIGGER = "map.view.alerts.trigger"

# -------------------------
# Global Keys: Plot Types & Rendering
# -------------------------
MAP_VIEW_PLOT_KIND = "map.view.plot.kind"

# Hexbin
MAP_VIEW_HEX_GRIDSIZE = "map.view.hex.gridsize"
MAP_VIEW_HEX_METRIC = "map.view.hex.metric"

# Contour / KDE
MAP_VIEW_CONTOUR_BANDWIDTH = "map.view.contour.bandwidth"
MAP_VIEW_CONTOUR_STEPS = "map.view.contour.steps"
MAP_VIEW_CONTOUR_FILLED = "map.view.contour.filled"
MAP_VIEW_CONTOUR_LABELS = "map.view.contour.labels"

# Data Value Filter
MAP_VIEW_FILTER_ENABLE = "map.view.filter.enable"
MAP_VIEW_FILTER_VMIN = "map.view.filter.v_min"
MAP_VIEW_FILTER_VMAX = "map.view.filter.v_max"

# -------------------------
# Global Keys: Selection
# -------------------------
# Store-backed selection mode for the map.
#
# We keep this in the store (GUI-only) so the controller
# and panels can react without tight coupling.
MAP_SELECT_MODE = "map.select.mode"      # off|point|group
MAP_SELECT_IDS = "map.select.ids"        # list[int]
MAP_SELECT_OPEN = "map.select.open"      # bool
MAP_SELECT_PINNED = "map.select.pinned"  # bool

# # -------------------------
# Map Core Defaults
# -------------------------
_MAP_DEFAULTS = {
    MAP_ENGINE: "leaflet",
    MAP_COORD_MODE: "lonlat",
    MAP_GOOGLE_API_KEY: "",
    MAP_X_COL: "",
    MAP_Y_COL: "",
    MAP_Z_COL: "",
    MAP_FOCUS_MODE: False,
    MAP_SHOW_ANALYTICS: False,
    MAP_DATA_SOURCE: "auto",
    MAP_MANUAL_FILES: [],
    MAP_SELECTED_FILES: [],
    MAP_ACTIVE_FILE: "",
    MAP_TIME_COL: "",
    MAP_STEP_COL: "",
    MAP_VALUE_COL: "",
    MAP_VALUE_UNIT: "",
    MAP_TIME_VALUE: "",
    MAP_ID_COL: "sample_idx",
    MAP_OBS_COL: "",
    MAP_CLICK_SAMPLE_IDX: None,
    MAP_VIEW_BASEMAP: "streets",
    MAP_VIEW_SHOW_GRID: False,
    MAP_VIEW_SHOW_COLORBAR: True,
    MAP_BOOKMARKS: [],
    MAP_MEASURE_MODE: "off",
    
    # Derived (controller-populated)
    MAP_DF_ALL: None,
    MAP_DF_FRAME: None,
    MAP_DF_POINTS: None,

    MAP_VIEW_BASEMAP_STYLE: "light",
    MAP_VIEW_TILES_OPACITY: 1.0,
    MAP_VIEW_COLORMAP: "viridis",
    MAP_VIEW_CMAP_INVERT: False,
    MAP_VIEW_AUTOSCALE: True,
    MAP_VIEW_VMIN: 0.0,
    MAP_VIEW_VMAX: 1.0,
    MAP_VIEW_MARKER_SIZE: 6,
    MAP_VIEW_MARKER_OPACITY: 0.9,

    # Hover toolbar
    MAP_TOOLTAB_PINNED: False,
    
    # Sampling
    MAP_SAMPLING_MODE: "auto",
    MAP_SAMPLING_METHOD: "grid",
    MAP_SAMPLING_MAX_POINTS: 80000,
    MAP_SAMPLING_SEED: 0,
    MAP_SAMPLING_CELL_KM: 1.0,
    MAP_SAMPLING_MAX_PER_CELL: 50,
    MAP_SAMPLING_APPLY_HOTSPOTS: True,
    
    # Selection (point / group)
    MAP_SELECT_MODE: "off",
    MAP_SELECT_IDS: [],
    MAP_SELECT_OPEN: False,
    MAP_SELECT_PINNED: False,
    
    K_PROP_LEGEND: "global",
    
    MAP_DF_ALL_POINTS: None,
    MAP_PROP_BUILD_ID: 0,
    MAP_PROP_YEAR: None,
    MAP_PROP_TIMELINE: [],
    MAP_PROP_DF_ALL: None,
    MAP_PROP_DF_FRAME: None,
    MAP_PROP_DF_POINTS: None,
    MAP_PROP_VMIN: None,
    MAP_PROP_VMAX: None,
    MAP_PROP_VECTORS: [],


}

# -------------------------
# View keys + defaults
# -------------------------
VIEW_DEFAULTS: Dict[str, object] = {
    MAP_VIEW_BASEMAP: "osm",
    MAP_VIEW_BASEMAP_STYLE: "light",
    MAP_VIEW_TILES_OPACITY: 1.0,
    MAP_VIEW_COLORMAP: "viridis",
    MAP_VIEW_CMAP_INVERT: False,
    MAP_VIEW_AUTOSCALE: True,
    MAP_VIEW_VMIN: 0.0,
    MAP_VIEW_VMAX: 1.0,
    MAP_VIEW_CLIP_MODE: "none",
    MAP_VIEW_MARKER_SIZE: 6,
    MAP_VIEW_MARKER_OPACITY: 0.85,
    MAP_VIEW_SHOW_COLORBAR: True,
    MAP_VIEW_SHOW_LEGEND: False,
    MAP_VIEW_LEGEND_POS: "br",

    # Hotspots (attention layer)
    MAP_VIEW_HOTSPOTS_ENABLED: False,
    MAP_VIEW_HOTSPOTS_MODE: "auto",          # auto|manual|merge
    MAP_VIEW_HOTSPOTS_METHOD: "grid",        # grid|quantile|cluster
    MAP_VIEW_HOTSPOTS_METRIC: "high",        # value|abs|high|low
    MAP_VIEW_HOTSPOTS_TIME_AGG: "current",   # current|mean|max|trend
    MAP_VIEW_HOTSPOTS_THR_MODE: "quantile",  # quantile|absolute
    MAP_VIEW_HOTSPOTS_QUANTILE: 0.98,
    MAP_VIEW_HOTSPOTS_ABS_THR: 0.0,
    MAP_VIEW_HOTSPOTS_CELL_KM: 1.0,
    MAP_VIEW_HOTSPOTS_MIN_PTS: 20,
    MAP_VIEW_HOTSPOTS_MAX_N: 8,
    MAP_VIEW_HOTSPOTS_MIN_SEP_KM: 2.0,
    
    MAP_VIEW_HOTSPOTS_STYLE: "pulse",        # pulse|glow
    MAP_VIEW_HOTSPOTS_PULSE: True,
    MAP_VIEW_HOTSPOTS_PULSE_SPEED: 1.0,
    MAP_VIEW_HOTSPOTS_RING_KM: 0.8,
    MAP_VIEW_HOTSPOTS_LABELS: True,
    
    # Interpretation (policy-ready, store-backed)
    MAP_VIEW_INTERP_ENABLED: False,
    MAP_VIEW_INTERP_SCHEME: "subsidence",
    MAP_VIEW_INTERP_CALLOUTS: True,
    MAP_VIEW_INTERP_CALLOUT_LEVEL: "standard",
    MAP_VIEW_INTERP_CALLOUT_ACTIONS: True,
    MAP_VIEW_INTERP_ACTION_PACK: "balanced",
    MAP_VIEW_INTERP_ACTION_INTENSITY: "balanced",
    MAP_VIEW_INTERP_TONE: "municipal",
    MAP_VIEW_INTERP_SUMMARY: "",
    
    # Propagation / Animation
    K_PROP_ENABLED: False,
    K_PROP_YEARS: 5,
    K_PROP_SPEED: 800,  # 800ms per year
    K_PROP_MODE: "absolute",
    K_PROP_VECTORS: True,
    K_PROP_LOOP: False,
    
    # Alerts
    K_ALERT_ENABLED: False,
    K_ALERT_TRIGGER: "Severity: Critical",
    
    # Render Defaults Merge
    **{
        map_view_key(k): v
        for k, v in _VIEW_RENDER_DEFAULTS.items()
    },
}


VIEW_KEYS: List[str] = list(VIEW_DEFAULTS.keys())
