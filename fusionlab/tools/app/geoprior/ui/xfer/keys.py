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
    }
