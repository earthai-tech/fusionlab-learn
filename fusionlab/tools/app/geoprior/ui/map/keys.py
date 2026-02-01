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

# Propagation / Simulation Options
K_PROP_ENABLED = "map.view.prop.enabled"        # bool
K_PROP_YEARS = "map.view.prop.years_extra"      # int (e.g. 5 future years)
K_PROP_SPEED = "map.view.prop.speed"            # int (ms per frame)
K_PROP_MODE = "map.view.prop.mode"              # absolute | differential | risk_mask
K_PROP_VECTORS = "map.view.prop.show_vectors"   # bool (show expansion arrows)
K_PROP_LOOP = "map.view.prop.loop"              # bool

# --- Alert System Keys ---
K_ALERT_ENABLED = "map.view.alerts.enabled"
K_ALERT_TRIGGER = "map.view.alerts.trigger"

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

_MAP_DEFAULTS = {
    "map.engine": "leaflet",
    "map.coord_mode": "lonlat",
    "map.google_api_key": "",
    "map.x_col": "",
    "map.y_col": "",
    "map.z_col": "",
    "map.focus_mode": False,
    "map.show_analytics": False,
    "map.data_source": "auto",
    "map.manual_files": [],
    "map.selected_files": [],
    "map.active_file": "",
    "map.time_col": "",
    "map.step_col": "",
    "map.value_col": "",
    "map.time_value": "",
    "map.view.basemap": "streets",
    "map.view.show_grid": False,
    "map.view.show_colorbar": True,
    "map.bookmarks": [],
    "map.measure_mode": "off",
    "map.view.basemap_style": "light",
    "map.view.tiles_opacity": 1.0,
    "map.view.colormap": "viridis",
    "map.view.cmap_invert": False,
    "map.view.autoscale": True,
    "map.view.vmin": 0.0,
    "map.view.vmax": 1.0,
    "map.view.marker_size": 6,
    "map.view.marker_opacity": 0.9,
    
    "map.sampling.mode": "auto",
    "map.sampling.method": "grid",
    "map.sampling.max_points": 80000,
    "map.sampling.seed": 0,
    "map.sampling.cell_km": 1.0,
    "map.sampling.max_per_cell": 50,
    "map.sampling.apply_hotspots": True,
}

# -------------------------
# View keys + defaults
# -------------------------

VIEW_DEFAULTS: Dict[str, object] = {
    "map.view.basemap": "osm",
    "map.view.basemap_style": "light",
    "map.view.tiles_opacity": 1.0,
    "map.view.colormap": "viridis",
    "map.view.cmap_invert": False,
    "map.view.autoscale": True,
    "map.view.vmin": 0.0,
    "map.view.vmax": 1.0,
    "map.view.clip_mode": "none",
    "map.view.marker_size": 6,
    "map.view.marker_opacity": 0.85,
    "map.view.show_colorbar": True,
    "map.view.show_legend": False,
    "map.view.legend_pos": "br",
    # Hotspots (attention layer)
    "map.view.hotspots.enabled": False,
    "map.view.hotspots.mode": "auto",          # auto|manual|merge
    "map.view.hotspots.method": "grid",        # grid|quantile|cluster
    "map.view.hotspots.metric": "high",        # value|abs|high|low
    "map.view.hotspots.time_agg": "current",   # current|mean|max|trend
    "map.view.hotspots.thr_mode": "quantile",  # quantile|absolute
    "map.view.hotspots.quantile": 0.98,
    "map.view.hotspots.abs_thr": 0.0,
    "map.view.hotspots.cell_km": 1.0,
    "map.view.hotspots.min_pts": 20,
    "map.view.hotspots.max_n": 8,
    "map.view.hotspots.min_sep_km": 2.0,
    
    "map.view.hotspots.style": "pulse",        # pulse|glow
    "map.view.hotspots.pulse": True,
    "map.view.hotspots.pulse_speed": 1.0,
    "map.view.hotspots.ring_km": 0.8,
    "map.view.hotspots.labels": True,
    
    # Interpretation (policy-ready, store-backed)
    "map.view.interp.enabled": False,
    "map.view.interp.scheme": "subsidence",
    "map.view.interp.callouts": True,
    "map.view.interp.callout_level": "standard",
    "map.view.interp.callout_actions": True,
    "map.view.interp.action_pack": "balanced",
    "map.view.interp.action_intensity": "balanced",
    "map.view.interp.tone": "municipal",
    "map.view.interp.summary": "",
    
    K_PROP_ENABLED: False,
    K_PROP_YEARS: 5,
    K_PROP_SPEED: 800,  # 800ms per year
    K_PROP_MODE: "absolute",
    K_PROP_VECTORS: True,
    K_PROP_LOOP: False,
    
    K_ALERT_ENABLED: False,
    K_ALERT_TRIGGER: "Severity: Critical",
    
    **{
        map_view_key(k): v
        for k, v in _VIEW_RENDER_DEFAULTS.items()
    },

}


VIEW_KEYS: List[str] = list(VIEW_DEFAULTS.keys())
