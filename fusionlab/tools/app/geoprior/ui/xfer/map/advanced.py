# geoprior/ui/xfer/map/advanced.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.xfer.map.advanced

Advanced controls for the Xfer Map mode.

This panel is store-backed only:
- It writes xfer.map.* keys
- XferMapController reacts and re-renders

It is meant to live inside the floating MapToolDock.
"""

from __future__ import annotations

from typing import Optional, Set

from PyQt5.QtCore import Qt, QSize, QSignalBlocker
from PyQt5.QtWidgets import (
    QCheckBox, 
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QTextBrowser, 
    QFileDialog, 
)

from ....config.store import GeoConfigStore
from ....config.city_meta_registry import install_city_meta
from ..keys import (
    K_MAP_COORD_MODE,
    K_MAP_HOTSPOT_METRIC,
    K_MAP_HOTSPOT_MIN_SEP_KM,
    K_MAP_HOTSPOT_QUANTILE,
    K_MAP_HOT_RINGS_ENABLE,
    K_MAP_HOT_RINGS_RADIUS_KM,
    K_MAP_HOT_RINGS_COUNT,
    K_MAP_MAX_POINTS,
    K_MAP_OPACITY,
    K_MAP_POINTS_MODE,
    K_MAP_UTM_EPSG,
    K_MAP_RADAR_ENABLE,
    K_MAP_RADAR_TARGET,
    K_MAP_RADAR_ORDER,
    K_MAP_RADAR_DWELL_MS,
    K_MAP_RADAR_RADIUS_KM,
    K_MAP_RADAR_RINGS,
    K_MAP_BASEMAP,
    K_MAP_LINKS_ENABLE,
    K_MAP_LINKS_MODE,
    K_MAP_LINKS_K,
    K_MAP_LINKS_MAX,
    K_MAP_LINKS_SHOW_DIST,
    K_MAP_INTERP_HTML,
    K_MAP_INTERP_TIP,
    K_MAP_A_EPSG, 
    K_MAP_B_EPSG,
    K_CITIES_META_PATH, 
    BASEMAP_CHOICES, 
    basemap_icon_name, 
    K_MAP_ANIM_PULSE,
)
from ...view.keys import (
    K_PLOT_KIND, 
    K_HEX_GRIDSIZE,
    K_HEX_METRIC,
    K_CONTOUR_BANDWIDTH, 
    K_CONTOUR_STEPS, 
    K_CONTOUR_FILLED, 
    K_CONTOUR_LABELS,
    K_FILTER_ENABLE, 
    K_FILTER_V_MIN, 
    K_FILTER_V_MAX, 
    K_SPACE_MODE, 
    K_CONTOUR_METRIC
)
from ...icon_utils import try_icon
from .interactions_ui import XferMapInteractionsBlock
from .interpretation import map_help_html


__all__ = ["XferMapAdvancedPanel"]


class _Fold(QFrame):
    def __init__(
        self,
        title: str,
        hint: str,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setObjectName("xferMapAdvSection")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.btn = QToolButton(self)
        self.btn.setIconSize(QSize(14, 14))
        self.btn.setObjectName("xferMapAdvHeader")
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setChecked(False)
        self.btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn.setAutoRaise(True)

        self.hint = QLabel(hint, self)
        self.hint.setObjectName("xferMapAdvHint")
        self.hint.setWordWrap(True)

        hdr = QVBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(2)
        hdr.addWidget(self.btn, 0)
        hdr.addWidget(self.hint, 0)

        root.addLayout(hdr, 0)

        self.body = QFrame(self)
        self.body.setFrameShape(QFrame.NoFrame)
        self.body.setObjectName("xferMapAdvBody")

        self.body_l = QGridLayout(self.body)
        self.body_l.setContentsMargins(10, 6, 10, 6)
        self.body_l.setHorizontalSpacing(10)
        self.body_l.setVerticalSpacing(8)

        root.addWidget(self.body, 0)

        self.btn.setArrowType(Qt.RightArrow)
        # drive both: body visibility + arrow direction
        self.btn.toggled.connect(self._on_toggled)
        
        # start collapsed
        self.body.setVisible(False)

    def add_row(self, r: int, lbl: str, w: QWidget) -> None:
        lab = QLabel(lbl, self)
        lab.setObjectName("xferMapAdvLabel")
        self.body_l.addWidget(lab, r, 0, 1, 1)
        self.body_l.addWidget(w, r, 1, 1, 1)
        
    def _on_toggled(self, expanded: bool) -> None:
        self.body.setVisible(bool(expanded))
        self.btn.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )


class XferMapAdvancedPanel(QWidget):
    """
    Advanced map controls (View / Hotspots / Interactions / Interpretation).

    Store contract:
    - writes keys under xfer.map.*
    - controller listens and refreshes map
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store

        self.setObjectName("xferMapAdvancedPanel")
        self._build_ui()
        self._wire()
        self._apply_from_store(set())

        try:
            self._s.config_changed.connect(self._on_store_changed)
        except Exception:
            pass

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # -------------------------
        # 1) View
        # -------------------------
        self.sec_view = _Fold(
            "View",
            "Change draw density, opacity and coord reprojection.",
            parent=self,
        )

        self.sp_max_pts = QSpinBox(self)
        self.sp_max_pts.setRange(1000, 500000)
        self.sp_max_pts.setSingleStep(1000)

        self.sp_opacity = QDoubleSpinBox(self)
        self.sp_opacity.setDecimals(2)
        self.sp_opacity.setRange(0.05, 1.0)
        self.sp_opacity.setSingleStep(0.05)

        # 1. Global / Fallback controls (Existing)
        self.cmb_coord = QComboBox(self)
        self.cmb_coord.addItems(["Auto", "Lon/Lat", "UTM", "EPSG"])
        
        self.sp_utm = QSpinBox(self)
        self.sp_utm.setRange(0, 999999)
        self.sp_utm.setSpecialValueText("Auto")
        self.sp_utm.setToolTip("Global Default EPSG (if A/B not set)")

        # 2. NEW: Specific Overrides
        self.sp_epsg_a = QSpinBox(self)
        self.sp_epsg_a.setRange(0, 999999)
        self.sp_epsg_a.setSpecialValueText("Auto / Meta")
        self.sp_epsg_a.setToolTip("Override EPSG for City A (Input)")

        self.sp_epsg_b = QSpinBox(self)
        self.sp_epsg_b.setRange(0, 999999)
        self.sp_epsg_b.setSpecialValueText("Auto / Meta")
        self.sp_epsg_b.setToolTip("Override EPSG for City B (Target)")

        self.cmb_basemap = QComboBox(self.sec_view)
        self.cmb_basemap.setIconSize(QSize(18, 18))
        for bid, label in BASEMAP_CHOICES:
            ico = try_icon(basemap_icon_name(bid))
            self.cmb_basemap.addItem(ico, label, bid)
            
        # self.cmb_basemap.addItem("OpenStreetMap", "osm")
        # self.cmb_basemap.addItem("Esri Satellite", "esri_sat")
        # self.cmb_basemap.addItem("Esri Topo", "esri_topo")
        # self.cmb_basemap.addItem("Esri Terrain", "esri_terrain")
        # self.cmb_basemap.addItem("OpenTopoMap", "opentopo")
        # self.cmb_basemap.addItem("Carto Light", "carto_light")
        # self.cmb_basemap.addItem("Carto Dark", "carto_dark")

        # 3. NEW: Load Meta JSON
        self.btn_load_meta = QToolButton(self)
        self.btn_load_meta.setText("Load cities.meta.json...")
        self.btn_load_meta.setToolTip(
            "Load a registry file to auto-detect EPSGs by city name"
            )
        self.btn_load_meta.clicked.connect(self._on_load_meta_click)

        # Add to layout
        self.sec_view.add_row(0, "Max points", self.sp_max_pts)
        self.sec_view.add_row(1, "Opacity", self.sp_opacity)
        # self.sec_view.add_row(2, "Coords Mode", self.cmb_coord)
        # self.sec_view.add_row(3, "Default EPSG", self.sp_utm)
        self.sec_view.add_row(2, "Basemap", self.cmb_basemap)
        self.sec_view.add_row(3, "Coords Mode", self.cmb_coord)
        self.sec_view.add_row(4, "Default EPSG", self.sp_utm)

        # Separator or label for specifics
        lbl_spec = QLabel("<b>Specific Overrides</b>", self)
        # self.sec_view.body_l.addWidget(lbl_spec, 5, 0, 1, 2)
        
        # self.sec_view.add_row(6, "City A EPSG", self.sp_epsg_a)
        # self.sec_view.add_row(7, "City B EPSG", self.sp_epsg_b)
        # self.sec_view.body_l.addWidget(self.btn_load_meta, 8, 0, 1, 2)
        self.sec_view.body_l.addWidget(lbl_spec, 6, 0, 1, 2)
        self.sec_view.add_row(7, "City A EPSG", self.sp_epsg_a)
        self.sec_view.add_row(8, "City B EPSG", self.sp_epsg_b)
        self.sec_view.body_l.addWidget(self.btn_load_meta, 9, 0, 1, 2)

        root.addWidget(self.sec_view, 0)

        # -------------------------
        # 1b) Visualization 
        # -------------------------
        self.sec_viz = _Fold(
            "Visualization", 
            "Plot type (Scatter, Contour, Hexbin) and style.",
            parent=self
        )

        self.cmb_plot_kind = QComboBox(self)
        self.cmb_plot_kind.addItems(["scatter", "contour", "hexbin"])

        # Contour Specifics
        self.sp_cont_bw = QDoubleSpinBox(self)
        self.sp_cont_bw.setRange(1.0, 200.0)
        self.sp_cont_bw.setSingleStep(2.0)
        self.sp_cont_bw.setSuffix(" px")
        self.sp_cont_bw.setToolTip("Bandwidth (smoothing radius)")

        self.sp_cont_steps = QSpinBox(self)
        self.sp_cont_steps.setRange(2, 50)
        self.sp_cont_steps.setToolTip("Number of contour levels")
        
        # Contour Metric
        self.cmb_cont_metric = QComboBox(self)
        self.cmb_cont_metric.addItems(["value", "density"])
        self.cmb_cont_metric.setToolTip(
            "value: value-weighted density\n"
            "density: pure point density"
        )

        self.chk_cont_fill = QCheckBox("Filled contours", self)
        self.chk_cont_lbl  = QCheckBox("Annotate", self)

        # Hexbin Specifics
        self.sp_hex_grid = QSpinBox(self)
        self.sp_hex_grid.setRange(5, 200)
        self.sp_hex_grid.setSuffix(" px")
        self.sp_hex_grid.setToolTip("Hexagon radius/size")
        
        self.cmb_hex_metric = QComboBox(self) #
        self.cmb_hex_metric.addItems(["mean", "max", "min", "count"])
        # Layout
        self.sec_viz.add_row(0, "Plot Type", self.cmb_plot_kind)
        
        # Conditional rows (labels are stored to toggle visibility)
        self.lbl_cont_bw = QLabel("Smoothing", self)
        self.sec_viz.body_l.addWidget(self.lbl_cont_bw, 1, 0)
        self.sec_viz.body_l.addWidget(self.sp_cont_bw, 1, 1)

        self.lbl_cont_steps = QLabel("Levels", self)
        self.sec_viz.body_l.addWidget(self.lbl_cont_steps, 2, 0)
        self.sec_viz.body_l.addWidget(self.sp_cont_steps, 2, 1)
        
        # row: contour metric (row 3)
        self.lbl_cont_metric = QLabel("Contour metric", self)
        self.sec_viz.body_l.addWidget(self.lbl_cont_metric, 3, 0)
        self.sec_viz.body_l.addWidget(self.cmb_cont_metric, 3, 1)

        # move fill/labels to row 4
        self.sec_viz.body_l.addWidget(self.chk_cont_fill, 4, 0)
        self.sec_viz.body_l.addWidget(self.chk_cont_lbl, 4, 1)

        # move hex rows down to 5/6
        self.lbl_hex_grid = QLabel("Grid Size", self)
        self.sec_viz.body_l.addWidget(self.lbl_hex_grid, 5, 0)
        self.sec_viz.body_l.addWidget(self.sp_hex_grid, 5, 1)

        self.lbl_hex_mt = QLabel("Metric", self)
        self.sec_viz.body_l.addWidget(self.lbl_hex_mt, 6, 0)
        self.sec_viz.body_l.addWidget(self.cmb_hex_metric, 6, 1)

        root.addWidget(self.sec_viz, 0)
        
        # -------------------------
        # 1c) Data Filters (NEW SECTION)
        # -------------------------
        self.sec_filt = _Fold(
            "Data Filters", "Clip values and spatial range", 
            parent=self
            )
        
        self.chk_filt_en = QCheckBox("Enable Value Filter", self)
        
        self.sp_vmin = QDoubleSpinBox(self)
        self.sp_vmin.setRange(-9999, 9999)
        self.sp_vmin.setPrefix("Min: ")
        
        self.sp_vmax = QDoubleSpinBox(self)
        self.sp_vmax.setRange(-9999, 9999)
        self.sp_vmax.setPrefix("Max: ")

        self.cmb_space = QComboBox(self)
        self.cmb_space.addItems(
            ["all", "hotspots_only", "hotspots_proximity"]
            )

        self.sec_filt.body_l.addWidget(self.chk_filt_en, 0, 0, 1, 2)
        self.sec_filt.add_row(1, "Min Value", self.sp_vmin)
        self.sec_filt.add_row(2, "Max Value", self.sp_vmax)
        self.sec_filt.add_row(3, "Spatial Mode", self.cmb_space)
        

        root.addWidget(self.sec_filt, 0)
        # -------------------------
        # 2) Hotspots
        # -------------------------
        self.sec_hot = _Fold(
            "Hotspots",
            "Tune hotspot extraction (Top-N is in toolbar).",
            parent=self,
        )

        self.cmb_pts_mode = QComboBox(self)
        self.cmb_pts_mode.addItems(
            [
                "All points",
                "Hotspots only",
                "Hotspots + base",
            ]
        )

        self.sp_sep = QDoubleSpinBox(self)
        self.sp_sep.setDecimals(1)
        self.sp_sep.setRange(0.0, 50.0)
        self.sp_sep.setSingleStep(0.5)

        self.cmb_metric = QComboBox(self)
        self.cmb_metric.addItems(
            [
                "abs",
                "high",
                "low",
                "pos",
                "neg",
            ]
        )

        self.sp_q = QDoubleSpinBox(self)
        self.sp_q.setDecimals(3)
        self.sp_q.setRange(0.50, 0.999)
        self.sp_q.setSingleStep(0.005)

        self.sec_hot.add_row(0, "Points", self.cmb_pts_mode)
        self.sec_hot.add_row(1, "Min sep (km)", self.sp_sep)
        self.sec_hot.add_row(2, "Metric", self.cmb_metric)
        self.sec_hot.add_row(3, "Quantile", self.sp_q)
        
        self.btn_hot_pro = QToolButton(self)
        self.btn_hot_pro.setAutoRaise(True)
        self.btn_hot_pro.setText("Apply pro preset")
        self.btn_hot_pro.setToolTip(
            "1-click hotspot view: pulse + rings + "
            "radar + links."
        )
        self.btn_hot_pro.setIcon(
            try_icon(
                "sparkles.svg",
                fallback=self.style().standardIcon(
                    QStyle.SP_ArrowUp
                ),
            )
        )
        self.sec_hot.body_l.addWidget(
            self.btn_hot_pro, 4, 0, 1, 2
        )

        # ---- NEW: Hotspot analytics (Radar + Arrows) ----
        self.lbl_hot_ana = QLabel("<b>Hotspot analytics</b>", self)
        self.lbl_hot_ana.setObjectName("xferMapAdvSectionTitle")
        self.sec_hot.body_l.addWidget(self.lbl_hot_ana, 5, 0, 1, 2)

        # Radar
        self.chk_radar = QCheckBox("Radar sweep hotspots", self)
        self.cmb_radar_target = QComboBox(self)
        self.cmb_radar_target.addItems(
            ["overlay", "both", "a", "b"]
        )
        self.cmb_radar_order = QComboBox(self)
        self.cmb_radar_order.addItems(
            ["score", "abs", "rank"]
        )
        self.sp_radar_ms = QSpinBox(self)
        self.sp_radar_ms.setRange(120, 5000)
        self.sp_radar_ms.setSingleStep(80)

        self.sp_radar_rkm = QDoubleSpinBox(self)
        self.sp_radar_rkm.setDecimals(1)
        self.sp_radar_rkm.setRange(0.5, 50.0)
        self.sp_radar_rkm.setSingleStep(0.5)

        self.sp_radar_rings = QSpinBox(self)
        self.sp_radar_rings.setRange(1, 8)
        self.sp_radar_rings.setSingleStep(1)

        self.sec_hot.add_row(6, "Radar", self.chk_radar)
        self.sec_hot.add_row(7, "Radar target", self.cmb_radar_target)
        self.sec_hot.add_row(8, "Radar order", self.cmb_radar_order)
        self.sec_hot.add_row(9, "Radar dwell (ms)", self.sp_radar_ms)
        self.sec_hot.add_row(10, "Radar radius (km)", self.sp_radar_rkm)
        self.sec_hot.add_row(11, "Radar rings", self.sp_radar_rings)

        # Links (arrows)
        self.chk_links = QCheckBox("Arrows between hotspots", self)
        self.cmb_links_mode = QComboBox(self)
        self.cmb_links_mode.addItems(["nearest", "rank", "knn"])

        self.sp_links_k = QSpinBox(self)
        self.sp_links_k.setRange(1, 6)
        self.sp_links_k.setSingleStep(1)

        self.sp_links_max = QSpinBox(self)
        self.sp_links_max.setRange(1, 80)
        self.sp_links_max.setSingleStep(1)

        self.chk_links_dist = QCheckBox("Show distance labels", self)

        self.sec_hot.add_row(12, "Links", self.chk_links)
        self.sec_hot.add_row(13, "Link mode", self.cmb_links_mode)
        self.sec_hot.add_row(14, "k (knn)", self.sp_links_k)
        self.sec_hot.add_row(15, "Max links", self.sp_links_max)
        self.sec_hot.add_row(16, "Distance", self.chk_links_dist)
        
        self.chk_hot_rings = QCheckBox(
            "Impact rings around hotspots", self
        )
        self.sp_hot_ring_km = QDoubleSpinBox(self)
        self.sp_hot_ring_km.setDecimals(1)
        self.sp_hot_ring_km.setRange(0.5, 80.0)
        self.sp_hot_ring_km.setSingleStep(0.5)

        self.sp_hot_ring_n = QSpinBox(self)
        self.sp_hot_ring_n.setRange(1, 8)
        self.sp_hot_ring_n.setSingleStep(1)

        self.sec_hot.add_row(17, "Rings", self.chk_hot_rings)
        self.sec_hot.add_row(
            18, "Ring radius (km)", self.sp_hot_ring_km
        )
        self.sec_hot.add_row(
            19, "Ring count", self.sp_hot_ring_n
        )

        root.addWidget(self.sec_hot, 0)

        # -------------------------
        # 3) Interactions
        # -------------------------
        self.sec_int = _Fold(
            "Interactions",
            "Compare cities and derived layers.",
            parent=self,
        )
        
        self.int_block = XferMapInteractionsBlock(
            store=self._s,
            parent=self.sec_int.body,
        )
        self.sec_int.body_l.addWidget(self.int_block, 0, 0, 1, 2)
        root.addWidget(self.sec_int, 0)

        # -------------------------
        # 4) Interpretation
        # -------------------------
        self.sec_interp = _Fold(
            "Interpretation",
            "Guidance + recommended actions for transfers.",
            parent=self,
        )

        self.txt_interp = QTextBrowser(self)
        self.txt_interp.setObjectName("xferMapInterpDoc")
        self.txt_interp.setOpenExternalLinks(False)
        self.txt_interp.setFrameShape(QFrame.NoFrame)
        self.txt_interp.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self.txt_interp.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.txt_interp.setHtml(map_help_html())

        # Make it feel “expanded” when opened.
        self.txt_interp.setMinimumHeight(220)

        self.sec_interp.body_l.addWidget(
            self.txt_interp,
            0,
            0,
            1,
            2,
        )

        # Tip (short, action-oriented)
        self.lbl_interp_tip = QLabel(self)
        self.lbl_interp_tip.setObjectName("xferMapInterpTip")
        self.lbl_interp_tip.setWordWrap(True)
        self.lbl_interp_tip.setText(
            "Tip: start with Shared scale for fair A↔B "
            "comparison, then switch to Auto to inspect "
            "within-city patterns. Use Interactions→Δ "
            "to see A−B shifts; enable Extras "
            "(Δ-hotspots / overlap intensity / buffered "
            "intersection) to judge domain shift. "
            "Turn on Radar/Arrows to scan hotspot "
            "rank/score and verify spatial alignment "
            "(Coords + EPSG)."
        )

        self.sec_interp.body_l.addWidget(
            self.lbl_interp_tip,
            1,
            0,
            1,
            2,
        )

        root.addWidget(self.sec_interp, 0)


        root.addStretch(1)

    # -------------------------
    # Store sync
    # -------------------------
    def _wire(self) -> None:
        self.sp_max_pts.valueChanged.connect(self._push_view)
        self.sp_opacity.valueChanged.connect(self._push_view)
        self.cmb_basemap.currentIndexChanged.connect(
           self._push_view
        )
        self.cmb_coord.currentIndexChanged.connect(
             self._push_view
        )
         
        self.cmb_coord.currentIndexChanged.connect(
            self._push_view
        )
        self.sp_utm.valueChanged.connect(self._push_view)
        # self.sp_src.valueChanged.connect(self._push_view)

        self.cmb_pts_mode.currentIndexChanged.connect(
            self._push_hot
        )
        self.sp_sep.valueChanged.connect(self._push_hot)
        self.cmb_metric.currentIndexChanged.connect(
            self._push_hot
        )
        self.sp_q.valueChanged.connect(self._push_hot)

        self.btn_hot_pro.clicked.connect(
            self._apply_hot_pro_preset
        )

        # NEW: hotspot analytics -> store
        self.chk_radar.toggled.connect(self._push_hot_ana)
        self.cmb_radar_target.currentIndexChanged.connect(
            self._push_hot_ana
        )
        self.cmb_radar_order.currentIndexChanged.connect(
            self._push_hot_ana
        )
        self.sp_radar_ms.valueChanged.connect(self._push_hot_ana)
        self.sp_radar_rkm.valueChanged.connect(self._push_hot_ana)
        self.sp_radar_rings.valueChanged.connect(self._push_hot_ana)

        self.chk_links.toggled.connect(self._push_hot_ana)
        self.cmb_links_mode.currentIndexChanged.connect(
            self._push_hot_ana
        )
        self.sp_links_k.valueChanged.connect(self._push_hot_ana)
        self.sp_links_max.valueChanged.connect(self._push_hot_ana)
        self.chk_links_dist.toggled.connect(self._push_hot_ana)
        
        self.chk_hot_rings.toggled.connect(self._push_hot_ana)
        self.sp_hot_ring_km.valueChanged.connect(self._push_hot_ana)
        self.sp_hot_ring_n.valueChanged.connect(self._push_hot_ana)

        self.sp_epsg_a.valueChanged.connect(self._push_view)
        self.sp_epsg_b.valueChanged.connect(self._push_view)
        
        # Visualization
        self.cmb_plot_kind.currentIndexChanged.connect(self._push_viz)
        self.sp_cont_bw.valueChanged.connect(self._push_viz)
        self.sp_cont_steps.valueChanged.connect(self._push_viz)
        self.chk_cont_fill.toggled.connect(self._push_viz)
        self.sp_hex_grid.valueChanged.connect(self._push_viz)
        self.chk_cont_lbl.toggled.connect(self._push_viz)
        self.cmb_hex_metric.currentIndexChanged.connect(self._push_viz)
        
        # Filters
        self.chk_filt_en.toggled.connect(self._push_filt)
        self.sp_vmin.valueChanged.connect(self._push_filt)
        self.sp_vmax.valueChanged.connect(self._push_filt)
        self.cmb_space.currentIndexChanged.connect(self._push_filt)
        self.cmb_cont_metric.currentIndexChanged.connect(
            self._push_viz
        )

        
    def _on_load_meta_click(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load City Metadata", "", "JSON Files (*.json)"
            )
            if path:
                try:
                    # Install into store immediately
                    install_city_meta(self._s, paths=[path])
                    self._s.set(K_CITIES_META_PATH, path)
                    # Force refresh of controller
                    self._s.set("xfer.map.meta_reloaded", True) 
                except Exception as e:
                    print(f"Error loading meta: {e}")
                
    def _on_store_changed(self, keys: object) -> None:
        ch = self._as_set(keys)
        if not ch:
            return
        self._apply_from_store(ch)

    @staticmethod
    def _as_set(keys: object) -> Set[str]:
        try:
            return set(keys or [])
        except Exception:
            return set()

    def _apply_from_store(self, ch: Set[str]) -> None:
        s = self._s

        b_max = QSignalBlocker(self.sp_max_pts)
        b_op = QSignalBlocker(self.sp_opacity)
        b_cm = QSignalBlocker(self.cmb_coord)
        b_utm = QSignalBlocker(self.sp_utm)
        b_bm = QSignalBlocker(self.cmb_basemap)
        # b_src = QSignalBlocker(self.sp_src)
        b_pm = QSignalBlocker(self.cmb_pts_mode)
        b_sep = QSignalBlocker(self.sp_sep)
        b_met = QSignalBlocker(self.cmb_metric)
        b_q = QSignalBlocker(self.sp_q)
        b_radar = QSignalBlocker(self.chk_radar)
        b_rtar = QSignalBlocker(self.cmb_radar_target)
        b_rord = QSignalBlocker(self.cmb_radar_order)
        b_rms = QSignalBlocker(self.sp_radar_ms)
        b_rr = QSignalBlocker(self.sp_radar_rkm)
        b_rng = QSignalBlocker(self.sp_radar_rings)
        b_links = QSignalBlocker(self.chk_links)
        b_lm = QSignalBlocker(self.cmb_links_mode)
        b_lk = QSignalBlocker(self.sp_links_k)
        b_lmax = QSignalBlocker(self.sp_links_max)
        b_ld = QSignalBlocker(self.chk_links_dist)

        b_pk = QSignalBlocker(self.cmb_plot_kind)
        b_cbw = QSignalBlocker(self.sp_cont_bw)
        b_cst = QSignalBlocker(self.sp_cont_steps)
        b_cfil = QSignalBlocker(self.chk_cont_fill)
        b_hgd = QSignalBlocker(self.sp_hex_grid)
        b_cmet = QSignalBlocker(self.cmb_cont_metric)
        
        _ = (
            b_max,
            b_op,
            b_cm,
            b_utm,
            b_pk, 
            b_cbw, 
            b_cst, 
            b_cfil, 
            b_hgd, 
            
            b_cmet,
            b_pm,
            b_bm,
            b_sep,
            b_met,
            b_q,
            b_radar, 
            b_rtar,
            b_rord, 
            b_rms, 
            b_rr, 
            b_rng, 
            b_lm, 
            b_links, 
            b_lk, 
            b_lmax, 
            b_ld
            
        )

        if not ch or K_MAP_MAX_POINTS in ch:
            v = int(s.get(K_MAP_MAX_POINTS, 20000) or 20000)
            self.sp_max_pts.setValue(max(1000, v))

        if not ch or K_MAP_OPACITY in ch:
            v = float(s.get(K_MAP_OPACITY, 0.90) or 0.90)
            self.sp_opacity.setValue(max(0.05, min(1.0, v)))

        if not ch or K_MAP_COORD_MODE in ch:
            cm = str(s.get(K_MAP_COORD_MODE, "auto") or "auto")
            cm = cm.strip().lower()
            
            idx = 0  # auto
            if cm in ("lonlat", "degrees"):
                idx = 1
            elif cm == "utm":
                idx = 2
            elif cm == "epsg":
                idx = 3
            
            self.cmb_coord.setCurrentIndex(idx)

        if not ch or K_MAP_UTM_EPSG in ch:
            v = int(s.get(K_MAP_UTM_EPSG, 0) or 0)
            self.sp_utm.setValue(max(0, v))

        # if not ch or K_MAP_SRC_EPSG in ch:
        #     v = int(s.get(K_MAP_SRC_EPSG, 0) or 0)
        #     self.sp_src.setValue(max(0, v))

        if not ch or K_MAP_POINTS_MODE in ch:
            pm = str(s.get(K_MAP_POINTS_MODE, "all") or "")
            pm = pm.strip().lower()
            idx = 0
            if pm == "hotspots":
                idx = 1
            elif pm == "hotspots_plus":
                idx = 2
            self.cmb_pts_mode.setCurrentIndex(idx)

        if not ch or K_MAP_HOTSPOT_MIN_SEP_KM in ch:
            v = float(
                s.get(K_MAP_HOTSPOT_MIN_SEP_KM, 2.0) or 2.0
            )
            v = max(0.0, min(50.0, v))
            self.sp_sep.setValue(v)

        if not ch or K_MAP_HOTSPOT_METRIC in ch:
            m = str(
                s.get(K_MAP_HOTSPOT_METRIC, "abs") or "abs"
            )
            i = max(0, self.cmb_metric.findText(m))
            self.cmb_metric.setCurrentIndex(i)

        if not ch or K_MAP_HOTSPOT_QUANTILE in ch:
            q = float(
                s.get(K_MAP_HOTSPOT_QUANTILE, 0.98) or 0.98
            )
            q = max(0.50, min(0.999, q))
            self.sp_q.setValue(q)

        # ---- NEW: hotspot analytics ----
        if not ch or K_MAP_RADAR_ENABLE in ch:
            self.chk_radar.setChecked(
                bool(s.get(K_MAP_RADAR_ENABLE, False))
            )
        if not ch or K_MAP_RADAR_TARGET in ch:
            v = str(s.get(K_MAP_RADAR_TARGET, "overlay") or "overlay")
            i = max(0, self.cmb_radar_target.findText(v))
            self.cmb_radar_target.setCurrentIndex(i)
        if not ch or K_MAP_RADAR_ORDER in ch:
            v = str(s.get(K_MAP_RADAR_ORDER, "score") or "score")
            i = max(0, self.cmb_radar_order.findText(v))
            self.cmb_radar_order.setCurrentIndex(i)
        if not ch or K_MAP_RADAR_DWELL_MS in ch:
            v = int(s.get(K_MAP_RADAR_DWELL_MS, 520) or 520)
            self.sp_radar_ms.setValue(max(120, min(5000, v)))
        if not ch or K_MAP_RADAR_RADIUS_KM in ch:
            v = float(s.get(K_MAP_RADAR_RADIUS_KM, 8.0) or 8.0)
            self.sp_radar_rkm.setValue(max(0.5, min(50.0, v)))
        if not ch or K_MAP_RADAR_RINGS in ch:
            v = int(s.get(K_MAP_RADAR_RINGS, 3) or 3)
            self.sp_radar_rings.setValue(max(1, min(8, v)))

        if not ch or K_MAP_LINKS_ENABLE in ch:
            self.chk_links.setChecked(
                bool(s.get(K_MAP_LINKS_ENABLE, False))
            )
        if not ch or K_MAP_LINKS_MODE in ch:
            v = str(s.get(K_MAP_LINKS_MODE, "nearest") or "nearest")
            i = max(0, self.cmb_links_mode.findText(v))
            self.cmb_links_mode.setCurrentIndex(i)
        if not ch or K_MAP_LINKS_K in ch:
            v = int(s.get(K_MAP_LINKS_K, 1) or 1)
            self.sp_links_k.setValue(max(1, min(6, v)))
        if not ch or K_MAP_LINKS_MAX in ch:
            v = int(s.get(K_MAP_LINKS_MAX, 12) or 12)
            self.sp_links_max.setValue(max(1, min(80, v)))
        if not ch or K_MAP_LINKS_SHOW_DIST in ch:
            self.chk_links_dist.setChecked(
                bool(s.get(K_MAP_LINKS_SHOW_DIST, True))
            )
        if not ch or K_MAP_INTERP_HTML in ch:
            html = str(s.get(K_MAP_INTERP_HTML, "") or "")
            if html.strip():
                self.txt_interp.setHtml(html)
            else:
                self.txt_interp.setHtml(map_help_html())

        if not ch or K_MAP_INTERP_TIP in ch:
            tip = str(s.get(K_MAP_INTERP_TIP, "") or "")
            if tip.strip():
                self.lbl_interp_tip.setText(tip)
            else:
                self.lbl_interp_tip.setText(
                    "Tip: enable Radar/Arrows to inspect Δ-hotspots."
                )
        if not ch or K_MAP_INTERP_TIP in ch:
            tip = str(self._s.get(K_MAP_INTERP_TIP, "") or "")
            if tip.strip():
                self.lbl_interp_tip.setText(tip)
                
        if not ch or K_MAP_A_EPSG in ch:
            v = int(self._s.get(K_MAP_A_EPSG, 0) or 0)
            self.sp_epsg_a.setValue(v)
            
        if not ch or K_MAP_B_EPSG in ch:
            v = int(self._s.get(K_MAP_B_EPSG, 0) or 0)
            self.sp_epsg_b.setValue(v)
            
        if not ch or K_PLOT_KIND in ch:
            kind = str(s.get(K_PLOT_KIND, "scatter") or "scatter")
            idx = self.cmb_plot_kind.findText(kind)
            if idx >= 0: self.cmb_plot_kind.setCurrentIndex(idx)

        if not ch or K_CONTOUR_BANDWIDTH in ch:
            v = float(s.get(K_CONTOUR_BANDWIDTH, 15.0) or 15.0)
            self.sp_cont_bw.setValue(v)

        if not ch or K_CONTOUR_STEPS in ch:
            v = int(s.get(K_CONTOUR_STEPS, 10) or 10)
            self.sp_cont_steps.setValue(v)

        if not ch or K_CONTOUR_FILLED in ch:
            self.chk_cont_fill.setChecked(
                bool(s.get(K_CONTOUR_FILLED, True))
            )

        if not ch or K_HEX_GRIDSIZE in ch:
            v = int(s.get(K_HEX_GRIDSIZE, 30) or 30)
            self.sp_hex_grid.setValue(v)

        if not ch or K_FILTER_ENABLE in ch:
            self.chk_filt_en.setChecked(bool(s.get(K_FILTER_ENABLE, False)))
            
        if not ch or K_CONTOUR_METRIC in ch:
            m = str(s.get(K_CONTOUR_METRIC, "value") or "value")
            i = self.cmb_cont_metric.findText(m)
            if i < 0:
                i = 0
            self.cmb_cont_metric.setCurrentIndex(i)

        if not ch or K_MAP_BASEMAP in ch:
            bid = str(s.get(K_MAP_BASEMAP, "osm") or "osm")
            for i in range(self.cmb_basemap.count()):
                if self.cmb_basemap.itemData(i) == bid:
                    self.cmb_basemap.setCurrentIndex(i)
                    break

        self._enable_viz_opts()
        self._enable_hot_ana()
        
        cm_i = self.cmb_coord.currentIndex()
        self.sp_utm.setEnabled(cm_i in (0, 2))  # auto or utm
        # self.sp_src.setEnabled(cm_i in (0, 3))  # auto or epsg
        
    # -------------------------
    # Push -> store
    # -------------------------
    def _push_viz(self) -> None:
        s = self._s
        with s.batch():
            s.set(K_PLOT_KIND, self.cmb_plot_kind.currentText())
            s.set(K_CONTOUR_BANDWIDTH, self.sp_cont_bw.value())
            s.set(K_CONTOUR_STEPS, self.sp_cont_steps.value())
            s.set(K_CONTOUR_FILLED, self.chk_cont_fill.isChecked())
            s.set(K_CONTOUR_LABELS, self.chk_cont_lbl.isChecked())
            s.set(K_HEX_GRIDSIZE, self.sp_hex_grid.value())
            s.set(K_HEX_METRIC, self.cmb_hex_metric.currentText())
            s.set(K_CONTOUR_METRIC, self.cmb_cont_metric.currentText())

        self._enable_viz_opts()
        
    def _push_filt(self) -> None:
        s = self._s
        with s.batch():
            s.set(K_FILTER_ENABLE, self.chk_filt_en.isChecked())
            s.set(K_FILTER_V_MIN, self.sp_vmin.value())
            s.set(K_FILTER_V_MAX, self.sp_vmax.value())
            s.set(K_SPACE_MODE, self.cmb_space.currentText())
            
    def _enable_viz_opts(self) -> None:
        kind = self.cmb_plot_kind.currentText()
        is_cont = (kind == "contour")
        is_hex = (kind == "hexbin")

        # Toggle Contour Widgets
        self.lbl_cont_bw.setVisible(is_cont)
        self.sp_cont_bw.setVisible(is_cont)
        self.lbl_cont_steps.setVisible(is_cont)
        self.sp_cont_steps.setVisible(is_cont)
        self.chk_cont_fill.setVisible(is_cont)
        self.chk_cont_lbl.setVisible(is_cont)
        
        # Toggle Hex Widgets
        self.lbl_hex_grid.setVisible(is_hex)
        self.sp_hex_grid.setVisible(is_hex)
        self.lbl_hex_mt.setVisible(is_hex)
        self.cmb_hex_metric.setVisible(is_hex)
        self.lbl_cont_metric.setVisible(is_cont)
        self.cmb_cont_metric.setVisible(is_cont)

        
    def _push_view(self) -> None:
        s = self._s
        cm_i = int(self.cmb_coord.currentIndex())
        
        cm = "auto"
        if cm_i == 1:
            cm = "lonlat"
        elif cm_i == 2:
            cm = "utm"
        elif cm_i == 3:
            cm = "epsg"
        
        with s.batch():
            s.set(K_MAP_MAX_POINTS, int(self.sp_max_pts.value()))
            s.set(K_MAP_OPACITY, float(self.sp_opacity.value()))
            s.set(
                K_MAP_BASEMAP,
                str(self.cmb_basemap.currentData() or "osm"),
            )
            s.set(K_MAP_COORD_MODE, cm)
            s.set(K_MAP_UTM_EPSG, int(self.sp_utm.value()))
            # s.set(K_MAP_SRC_EPSG, int(self.sp_src.value()))
            s.set(K_MAP_A_EPSG, int(self.sp_epsg_a.value()))
            s.set(K_MAP_B_EPSG, int(self.sp_epsg_b.value()))


    def _push_hot(self) -> None:
        s = self._s
        idx = int(self.cmb_pts_mode.currentIndex())
        pm = "all"
        if idx == 1:
            pm = "hotspots"
        elif idx == 2:
            pm = "hotspots_plus"

        with s.batch():
            s.set(K_MAP_POINTS_MODE, pm)
            s.set(
                K_MAP_HOTSPOT_MIN_SEP_KM,
                float(self.sp_sep.value()),
            )
            s.set(
                K_MAP_HOTSPOT_METRIC,
                str(self.cmb_metric.currentText()),
            )
            s.set(
                K_MAP_HOTSPOT_QUANTILE,
                float(self.sp_q.value()),
            )

    def _push_hot_ana(self) -> None:
        s = self._s
        with s.batch():
            s.set(
                K_MAP_RADAR_ENABLE,
                bool(self.chk_radar.isChecked()),
            )
            s.set(
                K_MAP_RADAR_TARGET,
                str(self.cmb_radar_target.currentText()),
            )
            s.set(
                K_MAP_RADAR_ORDER,
                str(self.cmb_radar_order.currentText()),
            )
            s.set(
                K_MAP_RADAR_DWELL_MS,
                int(self.sp_radar_ms.value()),
            )
            s.set(
                K_MAP_RADAR_RADIUS_KM,
                float(self.sp_radar_rkm.value()),
            )
            s.set(
                K_MAP_RADAR_RINGS,
                int(self.sp_radar_rings.value()),
            )

            s.set(
                K_MAP_LINKS_ENABLE,
                bool(self.chk_links.isChecked()),
            )
            s.set(
                K_MAP_LINKS_MODE,
                str(self.cmb_links_mode.currentText()),
            )
            s.set(
                K_MAP_LINKS_K,
                int(self.sp_links_k.value()),
            )
            s.set(
                K_MAP_LINKS_MAX,
                int(self.sp_links_max.value()),
            )
            s.set(
                K_MAP_LINKS_SHOW_DIST,
                bool(self.chk_links_dist.isChecked()),
            )
            s.set(K_MAP_HOT_RINGS_ENABLE, bool(self.chk_hot_rings.isChecked()))
            s.set(K_MAP_HOT_RINGS_RADIUS_KM, float(self.sp_hot_ring_km.value()))
            s.set(K_MAP_HOT_RINGS_COUNT, int(self.sp_hot_ring_n.value()))

        self._enable_hot_ana()

    def _enable_hot_ana(self) -> None:
        r = bool(self.chk_radar.isChecked())
        for w in (
            self.cmb_radar_target,
            self.cmb_radar_order,
            self.sp_radar_ms,
            self.sp_radar_rkm,
            self.sp_radar_rings,
        ):
            w.setEnabled(r)

        l = bool(self.chk_links.isChecked())
        self.cmb_links_mode.setEnabled(l)
        self.sp_links_max.setEnabled(l)
        self.chk_links_dist.setEnabled(l)
        self.sp_links_k.setEnabled(l and self.cmb_links_mode.currentText() == "knn")
        hr = bool(self.chk_hot_rings.isChecked())
        self.sp_hot_ring_km.setEnabled(hr)
        self.sp_hot_ring_n.setEnabled(hr)
        
    def _apply_hot_pro_preset(self) -> None:
        s = self._s
        with s.batch():
            s.set(K_MAP_POINTS_MODE, "hotspots_plus")
    
            s.set(K_MAP_ANIM_PULSE, True)
            s.set(K_MAP_HOT_RINGS_ENABLE, True)
            s.set(K_MAP_HOT_RINGS_RADIUS_KM, 4.0)
            s.set(K_MAP_HOT_RINGS_COUNT, 3)
    
            s.set(K_MAP_RADAR_ENABLE, True)
            s.set(K_MAP_RADAR_TARGET, "overlay")
            s.set(K_MAP_RADAR_ORDER, "score")
            s.set(K_MAP_RADAR_DWELL_MS, 620)
            s.set(K_MAP_RADAR_RADIUS_KM, 8.0)
            s.set(K_MAP_RADAR_RINGS, 3)
    
            s.set(K_MAP_LINKS_ENABLE, True)
            s.set(K_MAP_LINKS_MODE, "nearest")
            s.set(K_MAP_LINKS_K, 1)
            s.set(K_MAP_LINKS_MAX, 24)
            s.set(K_MAP_LINKS_SHOW_DIST, False)