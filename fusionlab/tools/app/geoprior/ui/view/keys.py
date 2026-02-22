# geoprior/ui/view/keys.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.view.keys

Configuration keys for the Visualization Engine.
"""

# Plot Types
K_PLOT_KIND = "view.plot.kind"          # scatter | hexbin | heatmap
K_PLOT_CMAP = "view.plot.cmap"          # viridis | plasma | seismic | etc.
K_PLOT_OPACITY = "view.plot.opacity"    # 0.0 to 1.0 (float)

# Hexbin / Grid Options
K_HEX_GRIDSIZE = "view.hex.gridsize"    # int (e.g. 20, 50)
K_HEX_METRIC = "view.hex.metric"        # mean | max | min | count

# Data Filtering (Robustness)
K_FILTER_ENABLE = "view.filter.enable"
K_FILTER_V_MIN = "view.filter.v_min"    # float (clip below)
K_FILTER_V_MAX = "view.filter.v_max"    # float (clip above)

# Spatial Filters
K_SPACE_MODE = "view.space.mode"        # all | hotspots_only | hotspots_proximity
K_SPACE_RADIUS = "view.space.radius"    # km (for proximity)


# Contour Options
K_CONTOUR_BANDWIDTH = "view.contour.bandwidth"  # Smoothing (km)
K_CONTOUR_STEPS = "view.contour.steps"          # Number of levels
K_CONTOUR_FILLED = "view.contour.filled"        # Bool: fill or lines
K_CONTOUR_LABELS = "view.contour.labels"  
K_CONTOUR_METRIC = "view.contour.metric"

DEFAULTS = {
    K_PLOT_KIND: "scatter",
    K_PLOT_CMAP: "viridis",
    K_PLOT_OPACITY: 0.85,
    K_HEX_GRIDSIZE: 30,
    K_HEX_METRIC: "mean",
    K_FILTER_ENABLE: False,
    K_FILTER_V_MIN: -50.0,
    K_FILTER_V_MAX: 50.0,
    K_SPACE_MODE: "all",
    K_SPACE_RADIUS: 2.0,
    K_CONTOUR_BANDWIDTH: 15.0,
    K_CONTOUR_STEPS: 10,
    K_CONTOUR_FILLED: True,
    K_CONTOUR_LABELS: False,
    
}
