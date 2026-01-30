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

from typing import Dict, List, Optional


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

}


VIEW_KEYS: List[str] = list(VIEW_DEFAULTS.keys())
