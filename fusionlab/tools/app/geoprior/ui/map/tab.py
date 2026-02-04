# geoprior/ui/map/tab.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map_tab

MapTab updates:
- pass store into Data panel
- persist map.* defaults for A-panel state
- store selected files via map.selected_files
"""

from __future__ import annotations

from typing import Optional, Sequence
from pathlib import Path

import pandas as pd

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QSplitter,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QInputDialog,
)

from ...config.store import GeoConfigStore
from ..view.factory import ViewFactory

from .analytics_panel import CollapsibleAnalyticsPanel
from .controller import MapController
from .tooltab import MapToolTab
from .canvas import ForecastMapView
from .data_panel import AutoHideDataPanel
from .head import MapHeadBar
from .view_panel import AutoHideViewPanel
from .utils import parse_city_dir  
from .hotspots import (
    HotspotCfg,
    build_points,
    compute_hotspots,
    hotspots_payload,
)
from .coord_utils import ( 
    ensure_lonlat,
    parse_epsg, 
    df_to_lonlat, 
    lonlat_to_xy
)
from .interpretation import (
    cfg_from_get,
    apply_interp,
    rows_for_export,
    geojson_from_rows,
    dump_geojson,
    policy_brief_md,
)
from .keys import ( 
    _MAP_DEFAULTS,     
    K_PROP_ENABLED, 
    K_ALERT_TRIGGER,
    K_PROP_VECTORS,
    K_ALERT_ENABLED,
    MAP_DF_ALL,
    MAP_DF_POINTS,
    MAP_X_COL, 
    MAP_Y_COL, 
    MAP_Z_COL, 
    MAP_VIEW_HOTSPOTS_ENABLED, 
    MAP_TIME_COL, 
    MAP_VALUE_COL, 
    MAP_COORD_MODE, 
    MAP_UTM_EPSG, 
    MAP_SHOW_ANALYTICS, 
    MAP_FOCUS_MODE,
    MAP_BOOKMARKS, 
    MAP_STEP_COL,
    MAP_SELECT_MODE,
    MAP_SELECT_IDS,
    MAP_SELECT_OPEN,
    MAP_SELECT_PINNED,
    MAP_SAMPLING_APPLY_HOTSPOTS, 
    MAP_SAMPLING_CELL_KM, 
    MAP_SAMPLING_MAX_PER_CELL, 
    MAP_SAMPLING_MAX_POINTS,
    MAP_SAMPLING_MODE,
    MAP_SAMPLING_SEED, 
    MAP_SAMPLING_METHOD, 
    MAP_VIEW_BASEMAP, 
    MAP_VIEW_BASEMAP_STYLE, 
    MAP_VIEW_SHOW_COLORBAR, 
    MAP_MEASURE_MODE, 
    MAP_VIEW_TILES_OPACITY,
    MAP_VIEW_COLORMAP,
    MAP_VIEW_CMAP_INVERT,
    MAP_VIEW_AUTOSCALE,
    MAP_VIEW_VMIN,
    MAP_VIEW_VMAX,
    MAP_VIEW_MARKER_OPACITY,
    MAP_VIEW_SHOW_GRID,
    MAP_CLICK_SAMPLE_IDX, 
    MAP_ANALYTICS_HEIGHT, 
    MAP_ANALYTICS_PINNED, 
    MAP_VIEW_MARKER_SIZE, 
    MAP_SELECTED_FILES, 
    MAP_ACTIVE_FILE, 
    MAP_TIME_VALUE, 
    MAP_SRC_EPSG, 
    MAP_COORD_EPSG,
    MAP_ENGINE,
    MAP_GOOGLE_API_KEY,
    map_view_key, 
    
)
from .sampling import (
    cfg_from_get as samp_cfg_from_get,
    sample_points,
)

from .prop_panel import PropagationPanel
from .prop_utils import ( 
    compute_propagation_vectors, 
    detect_time_col, 
    process_simulation, 
)
from .overlay_dock import SideDrawer, FloatingDockWindow
from .selection_panel import SelectionPanel
from .selection_router import SelectionRouter

class _OverlayHost(QWidget):
    def __init__(
        self,
        *,
        layout_cb,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._layout_cb = layout_cb

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        cb = self._layout_cb
        if cb is not None:
            cb()

class MapTab(QWidget):
    
    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self._auto_fit = True
        
        self._vf = ViewFactory(
            store=self.store,
            key_fn=map_view_key,
            radius_key=
            MAP_VIEW_MARKER_SIZE,
        )

        self._last_hs_payload = []
        self._last_hs_ctx = {}
        self._sim_df = pd.DataFrame() # Store for simulated data
        
        self._sim_time_col = ""
        self._sim_value_col = ""

        self._engine_applied = ""
        self._google_key_applied = ""


        self.head = MapHeadBar(parent=self)
        
        self.nav_a = AutoHideDataPanel(
            store=self.store,
            parent=self,
            embedded=True,
        )
        
        self.canvas = ForecastMapView(parent=self)
        
        self.nav_c = AutoHideViewPanel(
            store=self.store,
            parent=self,
            embedded=True,
        )
        
        self.panel_d = CollapsibleAnalyticsPanel(
            store=self.store,
            parent=self,
        )
        
        self._split_vert = QSplitter(Qt.Vertical, self)
        
        self._dock_a = SideDrawer(
            title="Data",
            side="left",
            width=380,
            parent=self,
        )
        self._dock_c = SideDrawer(
            title="View",
            side="right",
            width=380,
            parent=self,
        )
        
        self._win_a = FloatingDockWindow(
            title="Data panel",
            side="left",
            parent=self,
        )
        self._win_c = FloatingDockWindow(
            title="View panel",
            side="right",
            parent=self,
        )
        self._win_d = FloatingDockWindow(
            title="Analytics",
            side="panel",
            parent=self,
        )
        
        self._tabs_d = None

        self._overlay_host = _OverlayHost(
            layout_cb=self._layout_overlays,
            parent=self,
        )
        
        olay = QVBoxLayout(self._overlay_host)
        olay.setContentsMargins(0, 0, 0, 0)
        olay.setSpacing(0)
        olay.addWidget(self.canvas, 1)
        
        # Overlays live ABOVE the canvas (not in a stack).
        for w in (self._dock_a, self._dock_c, self.panel_d):
            w.setParent(self._overlay_host)
            w.raise_()
            
        # Selection drawer (right-side overlay)
        self.sel_panel = SelectionPanel(
            store=self.store,
            parent=self._overlay_host,
        )
        self.sel_panel.setVisible(False)
        self.sel_panel.raise_()

        # Top-center hover toolbar (hotzone + bar).
        self.tooltab = MapToolTab(
            parent=self._overlay_host,
            store=self.store,
        )
        self.tooltab.raise_()
        
        self._dock_a.set_panel(self.nav_a)
        self._dock_c.set_panel(self.nav_c)
        
        # Important: make the overlay widgets only as wide as
        # the drawer, so they don't cover the whole map.
        self._dock_a.setFixedWidth(380)
        self._dock_c.setFixedWidth(380)
        
        # Start analytics hidden (same intent as before).
        self.panel_d.setVisible(False)

        # self._sizes_vert: Optional[list[int]] = None
        self._hover_lock = False

        self._build_ui()
        self._apply_defaults()
        self._connect_signals()
        
        # ---------------------------------------------
        # PATCH: Map brain (controller)
        # ---------------------------------------------
        self.controller = MapController(
            store=self.store,
            canvas=self.canvas,
            view_factory=self._vf,
            parent=self,
        )
        self.sel_router = SelectionRouter(
            store=self.store,
            canvas=self.canvas,
            controller=self.controller,
            parent=self,
        )


    def set_available_columns(self, cols: Sequence[str]) -> None:
        self.head.set_available_columns(cols)


    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)
    
        lay.addWidget(self.head)
        
        center_container = QWidget()
        center_lay = QVBoxLayout(center_container)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(0)
        
        center_lay.addWidget(self._overlay_host, 1)
        
        self.prop_panel = PropagationPanel(
            store=self.store,
            parent=center_container,
        )
        self.prop_panel.setVisible(False)
        center_lay.addWidget(self.prop_panel, 0)
        
        lay.addWidget(center_container, 1)
        
        # analytics starts hidden
        self._set_analytics_visible(False)

        self._set_analytics_visible(False)
        self._split_vert.setSizes([1, 0])

    def _layout_overlays(self) -> None:
        host = self._overlay_host
        if host is None:
            return
    
        r = host.rect()
        w = r.width()
        h = r.height()
    
        # Left dock (Data)
        if self._dock_a.isVisible():
            dw = self._dock_a.drawer.width()
            self._dock_a.setGeometry(0, 0, dw, h)
            self._dock_a.raise_()
    
        # Right dock (View)
        if self._dock_c.isVisible():
            dw = self._dock_c.drawer.width()
            x0 = max(0, w - dw)
            self._dock_c.setGeometry(x0, 0, dw, h)
            self._dock_c.raise_()

        # Selection drawer (right overlay)
        sp = getattr(self, "sel_panel", None)
        if sp is not None and sp.isVisible():
            pw = 420
            ph = min(360, max(220, h - 80))

            right_pad = 0
            if self._dock_c.isVisible():
                right_pad = self._dock_c.drawer.width()

            x0 = max(0, w - right_pad - pw - 10)
            y0 = 60
            sp.setGeometry(x0, y0, pw, ph)
            sp.raise_()

        # Bottom dock (Analytics)
        if self.panel_d.isVisible():
            ph = int(
                self.store.get(
                    MAP_ANALYTICS_HEIGHT,
                    280,
                ) or 280
            )
            ph = max(220, min(ph, h - 60))
            self.panel_d.setGeometry(0, h - ph, w, ph)
            self.panel_d.raise_()

        # Hover tooltab (top-center).
        if getattr(self, "tooltab", None) is not None:
            self.tooltab.relayout(w, h)

    def _apply_defaults(self) -> None:
        cfg = self.store.cfg
        if not self.store.get(MAP_UTM_EPSG, None):
            self.store.set(
                MAP_UTM_EPSG,
                int(getattr(cfg, "utm_epsg", 32631)),
            )
        if not self.store.get(MAP_COORD_EPSG, None):
            self.store.set(
                MAP_COORD_EPSG,
                int(getattr(cfg, "coord_src_epsg", 4326)),
            )
        x0 = cfg.lon_col
        y0 = cfg.lat_col
        z0 = cfg.subs_col

        with self.store.batch():
            for k, v in _MAP_DEFAULTS.items():
                cur = self.store.get(k, None)
                if cur is None or cur == "":
                    self.store.set(k, v)

            if not self.store.get(MAP_X_COL, ""):
                self.store.set(MAP_X_COL, x0)
            if not self.store.get(MAP_Y_COL, ""):
                self.store.set(MAP_Y_COL, y0)
            if not self.store.get(MAP_Z_COL, ""):
                self.store.set(MAP_Z_COL, z0)

        self._sync_from_store()

    def _connect_signals(self) -> None:
        self.store.config_replaced.connect(self._on_cfg_replaced)
        self.store.config_changed.connect(self._on_store_changed)

        self.nav_a.selection_changed.connect(
            self._on_files_selected,
        )

        self.head.engine_changed.connect(self._on_engine_changed)
        self.head.coord_mode_changed.connect(self._on_coord_changed)
        self.head.focus_toggled.connect(self._on_focus_toggled)
        self.head.analytics_toggled.connect(
            self._on_analytics_toggled,
        )

        self.head.x_changed.connect(
            lambda v: self.store.set(MAP_X_COL, v),
        )
        self.head.y_changed.connect(
            lambda v: self.store.set(MAP_Y_COL, v),
        )
        self.head.z_changed.connect(
            lambda v: self.store.set(MAP_Z_COL, v),
        )

        self.canvas.request_focus_mode.connect(
            self._on_focus_toggled,
        )
        
        self.head.data_toggled.connect(self._on_data_toggled)
        self.head.view_toggled.connect(self._on_view_toggled)

        if getattr(self, "tooltab", None) is not None:
            self.tooltab.triggered.connect(
                self._on_tooltab_triggered,
            )
        
        self._dock_a.request_pin.connect(self._pin_data)
        self._dock_a.request_close.connect(self._close_data)
        self._win_a.request_unpin.connect(self._unpin_data)
        
        self._dock_c.request_pin.connect(self._pin_view)
        self._dock_c.request_close.connect(self._close_view)
        self._win_c.request_unpin.connect(self._unpin_view)

        self.nav_c.export_requested.connect(
            self._on_export_requested
        )

        self.nav_a.columns_changed.connect(
            self._on_map_cols,
        )
        
        self.panel_d.request_close.connect(
            self._close_analytics
        )
        self.panel_d.request_pin.connect(
            self._pin_analytics
        )
        self._win_d.request_unpin.connect(
            self._unpin_analytics
        )

        self.nav_a.active_changed.connect(
            self._on_active_file,
        )
        self.nav_c.changed.connect(self._on_view_changed)
        self.head.fit_clicked.connect(self.canvas.fit_points)
        self.head.swap_xy_clicked.connect(
            self._on_swap_xy,
        )
        self.head.reset_mapping_clicked.connect(
            self._on_reset_xyz,
        )
        self.canvas.point_clicked.connect(
            self._on_map_point_clicked
        )
        self.canvas.point_clicked_id.connect(
            self._on_map_point_clicked_id
        )

        self.head.basemap_changed.connect(
            self._on_basemap_changed
        )
        self.head.grid_toggled.connect(
            self._on_grid_toggled
        )
        self.head.legend_toggled.connect(
            self._on_legend_toggled
        )
        self.head.export_requested.connect(
            self._on_export_requested
        )
        self.head.bookmark_requested.connect(
            self._on_bookmark_requested
        )
        self.head.measure_mode_changed.connect(
            self._on_measure_mode_changed
        )
        
        self.head.epsg_changed.connect(
            self._on_coord_epsg_changed,
        )

        # in _connect_signals()
        self.head.clear_map_requested.connect(
            self._on_clear_map_requested
        )

        # Propagation connections
        self.prop_panel.simulation_requested.connect(
            self._on_run_simulation
            )
        self.prop_panel.frame_changed.connect(
            self._on_sim_frame_changed
        )
        
        # Listen to store to show/hide the panel
        self.store.config_changed.connect(
            self._check_prop_visibility
        )
        self.sel_panel.request_close.connect(
            self._on_sel_close
        )
        self.sel_panel.request_pin.connect(
            lambda _on: self._layout_overlays()
        )
        # ---------------------------------------------
        # PATCH: repaint map when controller updates
        # ---------------------------------------------
        self.store.config_changed.connect(
            self._on_map_derived_changed
        )

        # Connect AlertGroup signals
        ag = self.nav_c.alert_group
        ag.focus_requested.connect(self._on_focus_critical)
        ag.report_requested.connect(self._on_issue_warning)
        
    def _on_sel_close(self) -> None:
        with self.store.batch():
            self.store.set(MAP_SELECT_OPEN, False)
            self.store.set(MAP_SELECT_PINNED, False)
        self._layout_overlays()

    def _on_tooltab_triggered(
        self,
        key: str,
        state: bool,
    ) -> None:
        k = str(key or "").strip().lower()

        if k == "data":
            self._on_data_toggled(bool(state))
        elif k == "view":
            self._on_view_toggled(bool(state))
        elif k == "focus":
            self._on_focus_toggled(bool(state))
        elif k == "analytics":
            self._on_analytics_toggled(bool(state))
        elif k == "fit":
            self.canvas.fit_points()
        elif k == "clear":
            self._on_clear_map_requested()
        elif k == "reset_xyz":
            self._on_reset_xyz()
        elif k == "select_point":
            self._set_select_mode(
                "point" if bool(state) else "off"
            )
        elif k == "select_group":
            self._set_select_mode(
                "group" if bool(state) else "off"
            )
        elif k == "clear_selection":
            self._clear_selection()
            
    def _set_select_mode(self, mode: str) -> None:
        m = str(mode or "off").strip().lower()
        if m not in ("off", "point", "group"):
            m = "off"

        prev = str(
            self.store.get(MAP_SELECT_MODE, "off")
            or "off"
        ).strip().lower()

        with self.store.batch():
            if prev != m:
                self.store.set(MAP_SELECT_IDS, [])
            self.store.set(MAP_SELECT_MODE, m)
            self.store.set(MAP_SELECT_OPEN, m != "off")

        if getattr(self, "tooltab", None) is None:
            return
        
        self.tooltab.set_checked("select_point", m == "point")
        self.tooltab.set_checked("select_group", m == "group")
        
        try:
            self.canvas.set_select_mode(m)
        except Exception:
            pass

    def _clear_selection(self) -> None:
        with self.store.batch():
            self.store.set(MAP_SELECT_IDS, [])
            self.store.set(MAP_SELECT_MODE, "off")
            self.store.set(MAP_SELECT_OPEN, False)

        if getattr(self, "tooltab", None) is None:
            return
        self.tooltab.set_checked("select_point", False)
        self.tooltab.set_checked("select_group", False)
        try:
            self.canvas.set_select_mode("off")
        except Exception:
            pass

    def _on_data_toggled(self, on: bool) -> None:
        if self._win_a.isVisible():
            self._win_a.raise_()
            self._win_a.activateWindow()
            self.head.set_data_checked(False)
            if getattr(self, "tooltab", None) is not None:
                self.tooltab.set_checked("data", False)
            return
        self._dock_a.set_open(bool(on))
        self._layout_overlays()

        is_open = bool(self._dock_a.isVisible())
        self.head.set_data_checked(is_open)
        if getattr(self, "tooltab", None) is not None:
            self.tooltab.set_checked("data", is_open)
        
    def _on_view_toggled(self, on: bool) -> None:
        if self._win_c.isVisible():
            self._win_c.raise_()
            self._win_c.activateWindow()
            self.head.set_view_checked(False)
            if getattr(self, "tooltab", None) is not None:
                self.tooltab.set_checked("view", False)
            return
        self._dock_c.set_open(bool(on))
        self._layout_overlays()

        is_open = bool(self._dock_c.isVisible())
        self.head.set_view_checked(is_open)
        if getattr(self, "tooltab", None) is not None:
            self.tooltab.set_checked("view", is_open)
    
    def _close_data(self) -> None:
        self._dock_a.set_open(False)
        self.head.set_data_checked(False)
        if getattr(self, "tooltab", None) is not None:
            self.tooltab.set_checked("data", False)
        self._layout_overlays()
    
    def _close_view(self) -> None:
        self._dock_c.set_open(False)
        self.head.set_view_checked(False)
        if getattr(self, "tooltab", None) is not None:
            self.tooltab.set_checked("view", False)
        self._layout_overlays()
    
    def _pin_data(self) -> None:
        w = self._dock_a.take_panel()
        if w is None:
            return
        self._dock_a.set_open(False)
        self.head.set_data_checked(False)
        if getattr(self, "tooltab", None) is not None:
            self.tooltab.set_checked("data", False)
        self._win_a.set_panel(w)
        self._win_a.show()
        self._win_a.raise_()
        self._win_a.activateWindow()
    
    def _unpin_data(self) -> None:
        w = self._win_a.take_panel()
        self._win_a.hide()
        if w is None:
            return
        self._dock_a.set_panel(w)
    
    def _pin_view(self) -> None:
        w = self._dock_c.take_panel()
        if w is None:
            return
        self._dock_c.set_open(False)
        self.head.set_view_checked(False)
        if getattr(self, "tooltab", None) is not None:
            self.tooltab.set_checked("view", False)
        self._win_c.set_panel(w)
        self._win_c.show()
        self._win_c.raise_()
        self._win_c.activateWindow()
    
    def _unpin_view(self) -> None:
        w = self._win_c.take_panel()
        self._win_c.hide()
        if w is None:
            return
        self._dock_c.set_panel(w)

    def _check_prop_visibility(self, keys):
        if K_PROP_ENABLED in keys:
            enabled = bool(self.store.get(K_PROP_ENABLED, False))
            self.prop_panel.setVisible(enabled)
            # If disabled, restore original data view
            if not enabled:
                self._refresh_points() 
                self.canvas.set_vectors([])
                
    def _close_analytics(self) -> None:
        if self._win_d.isVisible():
            self._unpin_analytics()
    
        self.store.set(MAP_SHOW_ANALYTICS, False)
    
    
    def _pin_analytics(self) -> None:
        w = self.panel_d.take_tabs()
        if w is None:
            return
    
        self._tabs_d = w
        self.store.set(MAP_ANALYTICS_PINNED, True)
        self.store.set(MAP_SHOW_ANALYTICS, True)
    
        self._set_analytics_visible(False)
    
        self._win_d.set_panel(w)
        self._win_d.show()
        self._win_d.raise_()
        self._win_d.activateWindow()
    
    
    def _unpin_analytics(self) -> None:
        w = self._win_d.take_panel()
        self._win_d.hide()
    
        self.store.set(MAP_ANALYTICS_PINNED, False)
    
        if w is None:
            self._tabs_d = None
            return
    
        self._tabs_d = None
        self.panel_d.set_tabs(w)
    
        show_d = bool(self.store.get(MAP_SHOW_ANALYTICS, False))
        self._set_analytics_visible(show_d)

    def _on_run_simulation(self, years_to_add: int):
        """
        1. Get current active dataframe (clean)
        2. Run extrapolation
        3. Update UI timeline
        """
        df_clean = self._get_current_clean_df() 
        if df_clean.empty:
            return

        self.head.set_active_dataset(
            "Running simulation...", tooltip="Please wait"
            )
        QTimer.singleShot(100, lambda: self._process_simulation(
            df_clean, years_to_add))

    def _check_alerts(self) -> None:
        # 1. Check if enabled
        if not self.store.get(K_ALERT_ENABLED, False):
            return

        # 2. Get Hotspots
        hs = self._last_hs_payload
        if not hs:
            self.nav_c.alert_group.set_status(False)
            return

        # 3. Analyze Severity
        n_crit = sum(1 for h in hs if h.get("sev") == "critical")
        n_high = sum(1 for h in hs if h.get("sev") == "high")
        
        # 4. Check Rule
        trig = str(self.store.get(
            K_ALERT_TRIGGER,
            "Severity: Critical",
        ))
        
        warn = False
        count = 0
        
        if trig == "Severity: Critical":
            warn = (n_crit > 0)
            count = n_crit
        elif trig == "Severity: High+":
            warn = ((n_crit + n_high) > 0)
            count = n_crit + n_high
        elif trig == "Severity: Any":
            warn = (len(hs) > 0)
            count = len(hs)
        elif trig == "Manual Threshold":
            # Assume hotspots list already filtered by abs_thr
            warn = (len(hs) > 0)
            count = len(hs)

        # 5. Update UI
        self.nav_c.alert_group.set_status(warn, count)
        
    def _process_simulation(self, df, years):
        sim, ys, t_used, v_used = process_simulation(
            df,
            years_to_add=int(years),
            time_col=str(self.store.get(
                MAP_TIME_COL, ""
            )),
            value_col=str(self.store.get(
                MAP_VALUE_COL, ""
            )),
        )
        self._sim_df = sim
        self._sim_time_col = t_used
        self._sim_value_col = v_used
    
        if ys:
            self.prop_panel.set_timeline(ys)
    
        self.head.set_active_dataset(
            f"Simulated (+{years} yrs)",
            tooltip="Simulation active",
        )

    def _on_sim_frame_changed(self, year: int) -> None:
        if self._sim_df.empty:
            return
    
        t_col = str(self._sim_time_col or "").strip()
        if not t_col:
            t_col = str(
                self.store.get(MAP_TIME_COL, "")
            ).strip()
    
        if (not t_col) or (t_col not in self._sim_df.columns):
            return
    
        frame = self._sim_df[self._sim_df[t_col] == year]
        if frame.empty:
            return
    
        self._render_frame_points(frame)
    
        if bool(self.store.get(K_PROP_VECTORS, True)):
            v_col = str(self._sim_value_col or "").strip()
            vecs = compute_propagation_vectors(
                frame,
                value_col=v_col,
            )
            self.canvas.set_vectors(vecs)
        else:
            self.canvas.set_vectors([])


    def _get_current_clean_df(self) -> pd.DataFrame:
        """Helper to get current data without UI rendering side effects"""
        # 1. Get Active File
        p = str(self.store.get(MAP_ACTIVE_FILE, "")).strip()
        if not p:
            return pd.DataFrame()
        fp = Path(p)
        if not fp.exists():
            return pd.DataFrame()

        # 2. Get Column Mapping
        x = str(self.store.get(MAP_X_COL, "")).strip()
        y = str(self.store.get(MAP_Y_COL, "")).strip()
        z0 = str(self.store.get(MAP_Z_COL, "")).strip()
        v0 = str(self.store.get(MAP_VALUE_COL, "")).strip()
        v = v0 or z0
        t = str(self.store.get(MAP_TIME_COL, "")).strip()
        
        if not t:
            t = detect_time_col(fp)

        if not x or not y or not v:
            return pd.DataFrame()

        # 3. Read Data (Full history, no time filtering)
        use = [c for c in (x, y, v, t) if c]
        use = list(dict.fromkeys(use))

        try:
            df = pd.read_csv(fp, usecols=use)
        except Exception:
            try:
                df = pd.read_csv(fp)
            except Exception:
                return pd.DataFrame()

        # 4. Standardize to canonical columns [lon, lat, v, t]
        # build_points handles numeric conversion and renaming
        out = build_points(df, x=x, y=y, v=v, t=t)
        if out.empty:
            return pd.DataFrame()

        # 5. Ensure Coordinates are WGS84 (lon/lat)
        # This matches the logic in _refresh_points to ensure visual consistency
        mode = str(self.store.get(MAP_COORD_MODE, "lonlat")).strip().lower()
        
        utm_epsg = parse_epsg(self.store.get(MAP_UTM_EPSG, None))
        if utm_epsg is None:
            utm_epsg = parse_epsg(self.store.get("utm_epsg", None))

        src_epsg = parse_epsg(self.store.get(MAP_COORD_EPSG, None))
        if src_epsg is None:
            src_epsg = parse_epsg(self.store.get(MAP_SRC_EPSG, None))
        if src_epsg is None:
            src_epsg = parse_epsg(self.store.get("coord_src_epsg", None))

        try:
            out, ok, _ = ensure_lonlat(
                out,
                mode=mode,
                utm_epsg=utm_epsg,
                src_epsg=src_epsg,
            )
            if not ok:
                # Fallback to simple conversion if ensure_lonlat strict checks fail
                raise ValueError("Strict reprojection failed")
        except Exception:
            out = df_to_lonlat(
                out,
                x="lon",
                y="lat",
                mode=mode,
                utm_epsg=utm_epsg,
                src_epsg=src_epsg,
            )

        if out.empty:
            return pd.DataFrame()

        # 6. Alias columns for Propagation Engine
        # The propagation utils might query 'coord_t' or 'subsidence' based on config.
        # We ensure those columns exist by mirroring the standardized 't' and 'v'.
        if t and "t" in out.columns:
            out[t] = out["t"]
        
        if v and "v" in out.columns:
            out[v] = out["v"]
            
        # Map standardized coords back to config keys if needed for ID grouping
        if x and "lon" in out.columns:
            out[x] = out["lon"]
        if y and "lat" in out.columns:
            out[y] = out["lat"]

        return out

    def _render_frame_points(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
    
        snap = self._view_snapshot()
    
        # rad = int(snap.get("marker_size") or 6)
        # opa = float(snap.get("marker_opacity") or 0.9)
        leg = bool(snap.get("show_colorbar", True))
    
        auto = bool(snap.get("autoscale", True))
        vmin = None if auto else snap.get("vmin", None)
        vmax = None if auto else snap.get("vmax", None)
    
        cmap = str(snap.get("colormap", "viridis"))
        inv = bool(snap.get("cmap_invert", False))
    
        label = str(self._sim_value_col or "").strip()
        if not label:
            label = str(
                self.store.get(MAP_VALUE_COL, "")
            ).strip()
        if not label:
            label = str(self.store.get(MAP_Z_COL, "")).strip()
    
        # pts = df.to_dict("records")
    
        # Build minimal payload for Leaflet:
        # JS expects {lat, lon, v}. Anything else can break JSON
        # (e.g. coord_t can contain NaT).
        vcol = str(self._sim_value_col or "").strip()
        if not vcol:
            vcol = str(self.store.get(MAP_VALUE_COL, "")).strip()
        if not vcol:
            vcol = str(self.store.get(MAP_Z_COL, "")).strip()
        
        if ("lat" not in df.columns) or ("lon" not in df.columns):
            # Do not send projected coords to Leaflet.
            # Avoid crashing and avoid plotting nonsense.
            print("[map] missing lon/lat columns. skip.")
            return
        
        if vcol not in df.columns:
            print(f"[map] missing value col: {vcol!r}. skip.")
            return
        
        cols = ["lat", "lon", vcol]
        if "sample_idx" in df.columns:
            cols.append("sample_idx")
        d0 = df[cols].copy()
        d0 = d0.rename(columns={vcol: "v"})
        
        # force numeric + drop NaN/NaT
        d0["lat"] = pd.to_numeric(d0["lat"], errors="coerce")
        d0["lon"] = pd.to_numeric(d0["lon"], errors="coerce")
        d0["v"] = pd.to_numeric(d0["v"], errors="coerce")
        d0 = d0.dropna(subset=["lat", "lon", "v"])
        
        payload = self._vf.build_layer(d0, "sim")

        try:
            if (
                payload is not None
                and getattr(payload, "kind", "") in ("points", "scatter")
                and isinstance(getattr(payload, "data", None), list)
                and "sample_idx" in d0.columns
            ):
                sids = pd.to_numeric(
                    d0["sample_idx"],
                    errors="coerce",
                )
                data = payload.data
                for i, pt in enumerate(data):
                    if i >= len(sids):
                        break
                    sid = sids.iloc[i]
                    if pd.isna(sid):
                        continue
                    try:
                        pt["sid"] = int(sid)
                    except Exception:
                        continue
        except Exception:
            pass

        if payload is None:
            return
        
        self.canvas.set_layer(
            payload.kind,
            payload.data,
            opts=payload.opts,
            vmin=vmin,
            vmax=vmax,
            label=label,
            show_legend=leg,
            cmap=cmap,
            invert=inv,
        )

    def _on_clear_map_requested(self) -> None:
        try:
            self.canvas.clear_points()
        except Exception:
            pass
        try:
            self.canvas.clear_hotspots()
        except Exception:
            pass
    
        self._last_hs_payload = []
        self._last_hs_ctx = {}
        
        self._win_d.hide()
        with self.store.batch():
            self.store.set(MAP_FOCUS_MODE, False)
            self.store.set(MAP_SHOW_ANALYTICS, False)
    
            self.store.set(MAP_ACTIVE_FILE, "")
            self.store.set(MAP_SELECTED_FILES, [])
    
            self.store.set(MAP_TIME_COL, "")
            self.store.set(MAP_TIME_VALUE, "")
            self.store.set(MAP_STEP_COL, "")
            self.store.set(MAP_VALUE_COL, "")
    
            self.store.set(MAP_VIEW_HOTSPOTS_ENABLED, False)
            self.store.set(MAP_ANALYTICS_PINNED, False)
            self.store.set(MAP_SHOW_ANALYTICS, False)
            
            self.store.set(MAP_SELECT_IDS, [])
            self.store.set(MAP_SELECT_MODE, "off")
            self.store.set(MAP_SELECT_OPEN, False)
            self.store.set(MAP_SELECT_PINNED, False)


    def _on_coord_epsg_changed(self, epsg: int) -> None:
        try:
            v = int(epsg)
        except Exception:
            return
        if v <= 0:
            return
        self.store.set(MAP_COORD_EPSG, v)

    def _on_basemap_changed(self, key: str) -> None:
        k = str(key or "").strip().lower()
    
        # Treat style selections coming from HeadBar as style changes
        if k in {"light", "dark", "gray"}:
            self.store.set(MAP_VIEW_BASEMAP_STYLE, k)
            return
    
        # Normalize provider aliases to ViewPanel/Canvas keys
        alias = {
            "streets": "osm",
            "sat": "satellite",
        }
        self.store.set(MAP_VIEW_BASEMAP, alias.get(k, k))


    def _on_grid_toggled(self, on: bool) -> None:
        self.store.set(MAP_VIEW_SHOW_GRID, bool(on))

    def _on_legend_toggled(self, on: bool) -> None:
        self.store.set(
            MAP_VIEW_SHOW_COLORBAR,
            bool(on),
        )

    def _on_measure_mode_changed(self, mode: str) -> None:
        self.store.set(MAP_MEASURE_MODE, str(mode))
     
    def _on_export_requested(self, kind: str) -> None:
        kind = str(kind or "").strip().lower()

        if kind == "png":
            self._export_png_snapshot()
            return

        if kind == "hotspots_csv":
            self._export_hotspots_csv()
            return

        if kind == "hotspots_geojson":
            self._export_hotspots_geojson()
            return

        if kind == "policy_brief":
            self._export_policy_brief()
            return

    def _export_hotspots_csv(self) -> None:
        if not self._last_hs_payload:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export hotspots CSV",
            "hotspots.csv",
            "CSV (*.csv)",
        )
        if not path:
            return

        rows = rows_for_export(
            self._last_hs_payload,
            basis=str(self._last_hs_ctx.get("basis", "")),
            metric=str(self._last_hs_ctx.get("metric", "")),
        )
        df = pd.DataFrame(rows)
        df.to_csv(str(path), index=False)

    def _export_hotspots_geojson(self) -> None:
        if not self._last_hs_payload:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export hotspots GeoJSON",
            "hotspots.geojson",
            "GeoJSON (*.geojson *.json)",
        )
        if not path:
            return

        rows = rows_for_export(
            self._last_hs_payload,
            basis=str(self._last_hs_ctx.get("basis", "")),
            metric=str(self._last_hs_ctx.get("metric", "")),
        )
        gj = geojson_from_rows(rows)
        txt = dump_geojson(gj)

        with open(str(path), "w", encoding="utf-8") as f:
            f.write(txt)

    def _export_policy_brief(self) -> None:
        if not self._last_hs_payload:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export policy brief",
            "policy_brief.md",
            "Markdown (*.md)",
        )
        if not path:
            return

        icfg = cfg_from_get(self.store.get)

        rows = rows_for_export(
            self._last_hs_payload,
            basis=str(self._last_hs_ctx.get("basis", "")),
            metric=str(self._last_hs_ctx.get("metric", "")),
        )

        md = policy_brief_md(
            rows,
            cfg=icfg,
            ctx=self._last_hs_ctx,
        )

        with open(str(path), "w", encoding="utf-8") as f:
            f.write(md)

    def _on_bookmark_requested(self, req: str) -> None:
        req = str(req or "").strip()
        if req == "add":
            self._bookmark_add()
            return
        if req == "clear":
            self.store.set(MAP_BOOKMARKS, [])
            return
        if req.startswith("goto:"):
            name = req.split(":", 1)[1].strip()
            self._bookmark_goto(name)

    def _bookmark_add(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Add bookmark",
            "Name",
        )
        if not ok:
            return
        name = str(name or "").strip()
        if not name:
            return

        snap = {}
        try:
            full = dict(self.nav_c.snapshot() or {})
            for k, v in full.items():
                kk = str(k)
                if kk.startswith("map.view."):
                    snap[kk.split("map.view.", 1)[1]] = v
        except Exception:
            snap = dict(self._view_snapshot() or {})

        bms = list(self.store.get(MAP_BOOKMARKS, []) or [])
        bms.append({"name": name, "view": snap})
        self.store.set(MAP_BOOKMARKS, bms)

    def _bookmark_goto(self, name: str) -> None:
        bms = list(self.store.get(MAP_BOOKMARKS, []) or [])
        view = None
        for it in bms:
            if str(it.get("name", "")) == str(name):
                view = it.get("view", None)
                break
        if not isinstance(view, dict):
            return

        with self.store.batch():
            for k, v in view.items():
                self.store.set("map.view." + str(k), v)

    def _on_map_point_clicked(self, x: float, y: float) -> None:
        if str(
            self.store.get(MAP_SELECT_MODE, "off")
        ).strip().lower() != "off":
            return

        if bool(self.store.get(MAP_FOCUS_MODE, False)):
            self.store.set(MAP_FOCUS_MODE, False)
    
        if not bool(self.store.get(MAP_SHOW_ANALYTICS, False)):
            self.store.set(MAP_SHOW_ANALYTICS, True)
    
        tabs = self._tabs_d
        if tabs is None:
            tabs = getattr(self.panel_d, "tabs", None)
        if tabs is None:
            return
    
        try:
            # JS emits lon/lat (EPSG:4326). Analytics expects data X/Y.
            mode = str(self.store.get(
                MAP_COORD_MODE, "lonlat")).strip().lower()
            utm_epsg = parse_epsg(self.store.get(
                MAP_UTM_EPSG, None))
            src_epsg = parse_epsg(self.store.get(
                MAP_COORD_EPSG, None))
    
            x_data, y_data, _ok, _msg = lonlat_to_xy(
                float(x),
                float(y),
                mode=mode,
                utm_epsg=utm_epsg,
                src_epsg=src_epsg,
            )
    
            try:
                tabs.setCurrentWidget(self.panel_d.selection)
                sel = getattr(self.panel_d, "selection", None)
                if sel is not None and hasattr(sel, "set_xy"):
                    sel.set_xy(float(x_data), float(y_data))
            except Exception:
                pass

            try:
                insp = getattr(self.panel_d, "inspector", None)
                if insp is not None and hasattr(insp, "set_xy"):
                    insp.set_xy(float(x_data), float(y_data))
            except Exception:
                pass
        except Exception:
            pass


    def _on_map_point_clicked_id(self, sid: int) -> None:
        if str(
            self.store.get(MAP_SELECT_MODE, "off")
        ).strip().lower() != "off":
            return

        if bool(self.store.get(MAP_FOCUS_MODE, False)):
            self.store.set(MAP_FOCUS_MODE, False)

        if not bool(self.store.get(MAP_SHOW_ANALYTICS, False)):
            self.store.set(MAP_SHOW_ANALYTICS, True)

        self.store.set(MAP_CLICK_SAMPLE_IDX, int(sid))

        tabs = self._tabs_d
        if tabs is None:
            tabs = getattr(self.panel_d, "tabs", None)
        if tabs is None:
            return

        try:
            # Prefer Selection tab for time-series plot.
            try:
                tabs.setCurrentWidget(self.panel_d.selection)
                sel = getattr(self.panel_d, "selection", None)
                if sel is not None and hasattr(sel, "set_id"):
                    sel.set_id(int(sid))
            except Exception:
                pass

            # Keep Inspector in sync (optional).
            try:
                insp = getattr(self.panel_d, "inspector", None)
                if insp is not None and hasattr(insp, "set_id"):
                    insp.set_id(int(sid))
            except Exception:
                pass
        except Exception:
            return

    def _on_swap_xy(self) -> None:
        x = str(self.store.get(MAP_X_COL, ""))
        y = str(self.store.get(MAP_Y_COL, ""))

        with self.store.batch():
            self.store.set(MAP_X_COL, y)
            self.store.set(MAP_Y_COL, x)

    def _on_reset_xyz(self) -> None:
        cfg = self.store.cfg
        x0 = str(getattr(cfg, "lon_col", "") or "")
        y0 = str(getattr(cfg, "lat_col", "") or "")
        z0 = str(getattr(cfg, "subs_col", "") or "")

        with self.store.batch():
            self.store.set(MAP_X_COL, x0)
            self.store.set(MAP_Y_COL, y0)
            self.store.set(MAP_Z_COL, z0)

    def _on_view_changed(self, _snap) -> None:
        self._apply_view_to_canvas()
        self._refresh_points()

    def _on_map_derived_changed(self, keys) -> None:
        ks = set(keys or [])
        if not ks:
            return

        if MAP_DF_POINTS in ks:
            self._refresh_points()

    def _on_map_cols(self, cols) -> None:
        cols = list(cols or [])
        self.head.set_available_columns(cols)
    
        x = str(self.store.get(MAP_X_COL, "")).strip()
        y = str(self.store.get(MAP_Y_COL, "")).strip()
        z = str(self.store.get(MAP_Z_COL, "")).strip()
    
        if x and x not in cols:
            self.store.set(MAP_X_COL, "")
        if y and y not in cols:
            self.store.set(MAP_Y_COL, "")
        if z and z not in cols:
            self.store.set(MAP_Z_COL, "")
    
    def _on_active_file(self, path: str) -> None:
        self._auto_fit = True
        
        self.store.set(MAP_ACTIVE_FILE, str(path))
        city, model, stage = self._infer_city_from_path(path)
        self.head.set_city_badge(city=city, model=model, stage=stage)
        try:
            p = Path(path)
            name = p.name
        except Exception:
            name = str(path)
        self.head.set_active_dataset(
            name,
            tooltip=str(path),
        )
        
    def _refresh_points(self) -> None:
        # ---------------------------------------------
        # Helpers
        # ---------------------------------------------
        def _clear_all() -> None:
            self.canvas.clear_points()
            try:
                self.canvas.clear_hotspots()
            except Exception:
                pass

        # ---------------------------------------------
        # PATCH: Use controller-derived frames
        # ---------------------------------------------
        out = self.store.get(MAP_DF_POINTS, None)
        if (not isinstance(out, pd.DataFrame)) or out.empty:
            _clear_all()
            return

        # label uses current value mapping (for legend text)
        z0 = str(self.store.get(MAP_Z_COL, "")).strip()
        v0 = str(self.store.get(MAP_VALUE_COL, "")).strip()
        vlab = v0 or z0 or "value"

        snap = self._view_snapshot()
        leg = bool(snap.get("show_colorbar", True))

        auto = bool(snap.get("autoscale", True))
        vmin = None if auto else snap.get("vmin", None)
        vmax = None if auto else snap.get("vmax", None)

        cmap = str(snap.get("colormap", "viridis"))
        inv = bool(snap.get("cmap_invert", False))

        payload = self._vf.build_layer(out, "main")
        if payload is None:
            _clear_all()
            return

        # ensure sid is present in JS payload for click sync
        try:
            if (
                getattr(payload, "kind", "") in ("points", "scatter")
                and isinstance(getattr(payload, "data", None), list)
                and "sample_idx" in out.columns
            ):
                sids = pd.to_numeric(
                    out["sample_idx"],
                    errors="coerce",
                )
                data = payload.data
                for i, pt in enumerate(data):
                    if i >= len(sids):
                        break
                    sid = sids.iloc[i]
                    if pd.isna(sid):
                        continue
                    try:
                        pt["sid"] = int(sid)
                    except Exception:
                        continue
        except Exception:
            pass

        self.canvas.set_layer(
            payload.kind,
            payload.data,
            opts=payload.opts,
            vmin=vmin,
            vmax=vmax,
            label=vlab,
            show_legend=leg,
            cmap=cmap,
            invert=inv,
        )
        
        
        # ---------------------------------------------
        # Hotspots (reuse store df_all when needed)
        # ---------------------------------------------
        k_en = MAP_VIEW_HOTSPOTS_ENABLED
        hot_on = bool(self.store.get(k_en, False))

        if not hot_on:
            try:
                self.canvas.clear_hotspots()
            except Exception:
                pass
            self._last_hs_payload = []
            self._last_hs_ctx = {}
        else:
            df_all = self.store.get(MAP_DF_ALL, None)
            if not isinstance(df_all, pd.DataFrame):
                df_all = pd.DataFrame()

            t = str(self.store.get(MAP_TIME_COL, "")).strip()
            x = str(self.store.get(MAP_X_COL, "")).strip()
            y = str(self.store.get(MAP_Y_COL, "")).strip()
            vcol = str(self.store.get(MAP_VALUE_COL, "")).strip()
            if not vcol:
                vcol = str(self.store.get(MAP_Z_COL, "")).strip()

            k = "map.view.hotspots."

            hs_mode = str(self.store.get(k + "mode", "auto")).lower()
            method = str(self.store.get(k + "method", "grid")).lower()
            metric = str(self.store.get(k + "metric", "high")).lower()

            thr_mode = str(
                self.store.get(k + "thr_mode", "quantile")
            ).lower()

            q = float(self.store.get(k + "quantile", 0.98) or 0.98)
            q = max(0.0, min(1.0, q))

            abs_thr = self.store.get(k + "abs_thr", None)

            time_agg = str(
                self.store.get(k + "time_agg", "current")
            ).lower()

            time_win = int(self.store.get(k + "time_window", 0) or 0)

            cell_km = float(self.store.get(k + "cell_km", 1.0) or 1.0)
            min_pts = int(self.store.get(k + "min_pts", 20) or 20)
            max_n = int(self.store.get(k + "max_n", 8) or 8)

            min_sep = float(
                self.store.get(k + "min_sep_km", 2.0) or 2.0
            )

            if method not in ("grid", "quantile"):
                method = "grid"

            hs_payload = []

            if hs_mode in ("auto", "merge"):
                # current slice: already lon/lat/v
                pts_for_hs = out[["lon", "lat", "v"]].copy()

                # optional full-history aggregation
                if (
                    time_agg != "current"
                    and (not df_all.empty)
                    and x
                    and y
                    and vcol
                    and (x in df_all.columns)
                    and (y in df_all.columns)
                    and (vcol in df_all.columns)
                ):
                    try:
                        pts_for_hs = build_points(
                            df_all,
                            x=x,
                            y=y,
                            v=vcol,
                            t=t,
                        )
                    except TypeError:
                        pts_for_hs = build_points(
                            df_all,
                            x=x,
                            y=y,
                            v=vcol,
                        )

                    # ensure lon/lat
                    mode = str(
                        self.store.get(MAP_COORD_MODE, "lonlat")
                    ).strip().lower()

                    utm_epsg = parse_epsg(self.store.get(MAP_UTM_EPSG, None))
                    src_epsg = parse_epsg(self.store.get(MAP_COORD_EPSG, None))

                    try:
                        pts_for_hs, ok2, _m2 = ensure_lonlat(
                            pts_for_hs,
                            mode=mode,
                            utm_epsg=utm_epsg,
                            src_epsg=src_epsg,
                        )
                    except Exception:
                        pts_for_hs = df_to_lonlat(
                            pts_for_hs,
                            x="lon",
                            y="lat",
                            mode=mode,
                            utm_epsg=utm_epsg,
                            src_epsg=src_epsg,
                        )
                        ok2 = not pts_for_hs.empty

                    if (not ok2) or pts_for_hs.empty:
                        pts_for_hs = pd.DataFrame(
                            columns=["lon", "lat", "v"]
                        )

                if thr_mode == "absolute" and abs_thr is None:
                    hs_payload = []
                elif not pts_for_hs.empty:
                    t_col = "t" if (time_agg != "current" and t) else ""
                    scfg = samp_cfg_from_get(self.store.get)
                    if scfg.apply_hotspots:
                        pts_for_hs = sample_points(
                            pts_for_hs,
                            scfg,
                            lon="lon",
                            lat="lat",
                        )

                    cfg = HotspotCfg(
                        method=method,
                        metric=metric,
                        quantile=q,
                        thr_mode=thr_mode,
                        abs_thr=None
                        if abs_thr is None
                        else float(abs_thr),
                        max_n=max_n,
                        min_sep_km=min_sep,
                        cell_km=cell_km,
                        min_pts=min_pts,
                        time_col=t_col,
                        time_agg=time_agg,
                        time_window=time_win,
                    )

                    hs = compute_hotspots(
                        pts_for_hs,
                        cfg=cfg,
                        coord_mode="lonlat",
                    )
                    hs_payload = hotspots_payload(hs)

            style = str(self.store.get(k + "style", "pulse") or "pulse")
            pulse = bool(self.store.get(k + "pulse", True))

            spd = float(
                self.store.get(k + "pulse_speed", 1.0) or 1.0
            )
            spd = max(0.2, min(3.0, spd))

            ring_km = float(self.store.get(k + "ring_km", 0.8) or 0.8)
            ring_km = max(0.01, ring_km)

            icfg = cfg_from_get(self.store.get)

            hs_payload = apply_interp(
                hs_payload,
                cfg=icfg,
                basis=time_agg,
                metric=metric,
            )

            labels = bool(self.store.get(k + "labels", True))
            if icfg.enabled and icfg.callouts:
                labels = True

            self._last_hs_payload = list(hs_payload or [])
            self._last_hs_ctx = {
                "basis": time_agg,
                "metric": metric,
            }

            try:
                self.canvas.set_hotspots(
                    hs_payload,
                    show=True,
                    style=style,
                    pulse=pulse,
                    pulse_speed=spd,
                    ring_km=ring_km,
                    labels=labels,
                )
            except Exception:
                pass

        # Auto-fit once per dataset (unchanged behavior)
        if self._auto_fit:
            self.canvas.fit_points()
            self._auto_fit = False

        self._check_alerts()

    def _freeze_hover(self, ms: int = 250) -> None:
        if bool(self.store.get(MAP_FOCUS_MODE, False)):
            return
    
        self._hover_lock = True
        self.nav_a.set_hover_enabled(False)
        self.nav_c.set_hover_enabled(False)
    
        # If the View drawer is open (and not pinned),
        # close it to avoid “stuck overlay” during source switch.
        if self._dock_c.is_open() and (not self._win_c.isVisible()):
            self._dock_c.set_open(False)
            self.head.set_view_checked(False)
    
        QTimer.singleShot(ms, self._unfreeze_hover)

    def _unfreeze_hover(self) -> None:
        self._hover_lock = False

    def _view_snapshot(self) -> dict:
        keys = [
            MAP_VIEW_BASEMAP,
            MAP_VIEW_BASEMAP_STYLE,
            MAP_VIEW_TILES_OPACITY,
            MAP_VIEW_COLORMAP,
            MAP_VIEW_CMAP_INVERT,
            MAP_VIEW_AUTOSCALE,
            MAP_VIEW_VMIN,
            MAP_VIEW_VMAX,
            MAP_VIEW_MARKER_SIZE,
            MAP_VIEW_MARKER_OPACITY,
            MAP_VIEW_SHOW_COLORBAR,
            MAP_VIEW_SHOW_GRID,
        ]
        out = {}
        for k in keys:
            out[k.split("map.view.", 1)[1]] = self.store.get(k)
        return out

    def _export_png_snapshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save snapshot",
            "map.png",
            "PNG (*.png)",
        )
        if not path:
            return
        pm = self.canvas.grab()
        pm.save(str(path), "PNG")

    def _apply_view_to_canvas(self) -> None:
        snap = self._view_snapshot()
        self.canvas.apply_view(snap)


    def _on_files_selected(self, paths) -> None:
        # Store already updated in panel; keep hook.
        paths = list(paths or [])
        self.head.set_dataset_count(len(paths))


    def _on_cfg_replaced(self, _cfg) -> None:
        self._apply_defaults()

    def _on_store_changed(self, keys) -> None:
        keys = set(keys or [])
        sel_keys = {
            MAP_SELECT_MODE,
            MAP_SELECT_OPEN,
            MAP_SELECT_PINNED,
        }
        if keys & sel_keys:
            self._sync_selection_ui()
            self._layout_overlays()
            return

        if not keys:
            return
    
        if "map.data_source" in keys:
            self._freeze_hover(ms=250)
    
        ui_keys = {
            MAP_ENGINE,
            MAP_GOOGLE_API_KEY,
            MAP_COORD_MODE,
            MAP_X_COL,
            MAP_Y_COL,
            MAP_Z_COL,
            MAP_FOCUS_MODE,
            MAP_SHOW_ANALYTICS,
            MAP_ACTIVE_FILE,
            MAP_VIEW_BASEMAP,
            MAP_VIEW_SHOW_GRID,
            MAP_VIEW_SHOW_COLORBAR,
            MAP_BOOKMARKS,
            MAP_MEASURE_MODE,
            MAP_VALUE_COL,
            MAP_TIME_COL,
            MAP_TIME_VALUE,
            MAP_SAMPLING_MODE,
            MAP_SAMPLING_METHOD,
            MAP_SAMPLING_MAX_POINTS,
            MAP_SAMPLING_SEED,
            MAP_SAMPLING_CELL_KM,
            MAP_SAMPLING_MAX_PER_CELL,
            MAP_SAMPLING_APPLY_HOTSPOTS,
            MAP_ANALYTICS_PINNED

        }
       
        view_keys = [
            k for k in keys
            if k.startswith("map.view.")
        ]
    
        if keys & ui_keys:
            self._sync_from_store()
            return
    
        if view_keys:
            self._apply_view_to_canvas()

            needs_pts = any(
                k.startswith("map.view.hotspots.")
                or k.startswith("map.view.interp.")
                or k.startswith("map.view.plot.")
                or k.startswith("map.view.hex.")
                or k.startswith("map.view.contour.")
                or k.startswith("map.view.filter.")
                or k.startswith("map.view.space.")
                for k in view_keys
            )
            if needs_pts:
                self._refresh_points()
            return
        
    def _sync_selection_ui(self) -> None:
        m = str(
            self.store.get(MAP_SELECT_MODE, "off")
            or "off"
        ).strip().lower()

        if m not in ("off", "point", "group"):
            m = "off"

        if getattr(self, "tooltab", None) is not None:
            self.tooltab.set_checked("select_point", m == "point")
            self.tooltab.set_checked("select_group", m == "group")

        try:
            self.canvas.set_select_mode(m)
        except Exception:
            pass

    def _on_engine_changed(self, engine: str) -> None:
        self.store.set(MAP_ENGINE, str(engine))

    def _on_coord_changed(self, mode: str) -> None:
        self.store.set(MAP_COORD_MODE, str(mode))

    def _on_focus_toggled(self, enabled: bool) -> None:
        self.store.set(MAP_FOCUS_MODE, bool(enabled))
    
    def _on_analytics_toggled(self, enabled: bool) -> None:
        enabled = bool(enabled)
    
        pinned = bool(
            self.store.get(MAP_ANALYTICS_PINNED, False)
        )
    
        # If pinned, the toggle controls the floating window.
        if pinned:
            if enabled:
                self._win_d.show()
                self._win_d.raise_()
                self._win_d.activateWindow()
                self.store.set(MAP_SHOW_ANALYTICS, True)
            else:
                self._close_analytics()
            return
    
        # Normal (embedded) mode.
        self.store.set(MAP_SHOW_ANALYTICS, enabled)
    
    
    def _sync_from_store(self) -> None:
        # --------------------------------------------------
        # Engine
        # --------------------------------------------------
        engine = str(
            self.store.get(MAP_ENGINE, "leaflet")
        ).strip()
    
        gkey = str(
            self.store.get(MAP_GOOGLE_API_KEY, "") or ""
        ).strip()
    
        if (
            engine != self._engine_applied
            or gkey != self._google_key_applied
        ):
            self.canvas.set_engine(engine, google_key=gkey)
            self._engine_applied = engine
            self._google_key_applied = gkey
    
        # --------------------------------------------------
        # Header state
        # --------------------------------------------------
        coord = str(
            self.store.get(MAP_COORD_MODE, "lonlat")
        )
    
        x = str(self.store.get(MAP_X_COL, ""))
        y = str(self.store.get(MAP_Y_COL, ""))
        z = str(self.store.get(MAP_Z_COL, ""))
    
        focus = bool(
            self.store.get(MAP_FOCUS_MODE, False)
        )
        show_d = bool(
            self.store.get(MAP_SHOW_ANALYTICS, False)
        )
        pinned = bool(
            self.store.get(MAP_ANALYTICS_PINNED, False)
        )
    
        epsg = int(
            self.store.get(
                MAP_COORD_EPSG,
                getattr(
                    self.store.cfg,
                    "coord_src_epsg",
                    4326,
                ),
            )
        )
    
        self.head.set_epsg(epsg)
        self.head.set_engine(engine)
        self.head.set_coord_mode(coord)
        self.head.set_xyz(x=x, y=y, z=z)
        self.head.set_focus_checked(focus)

        if getattr(self, "tooltab", None) is not None:
            self.tooltab.set_checked("focus", focus)
    
        # Analytics checkbox should appear ON if
        # either embedded is on OR pinned exists.
        self.head.set_analytics_checked(show_d or pinned)

        if getattr(self, "tooltab", None) is not None:
            self.tooltab.set_checked(
                "analytics",
                bool(show_d or pinned),
            )
    
        self.canvas.set_focus_checked(focus)
    
        # --------------------------------------------------
        # Focus mode controls overall layout.
        # --------------------------------------------------
        self._set_focus_mode(focus)
    
        # --------------------------------------------------
        # Analytics presentation (pinned vs embedded)
        # (Focus mode always hides analytics)
        # --------------------------------------------------
        if focus:
            self._set_analytics_visible(False)
            if pinned:
                self._win_d.hide()
        else:
            if pinned:
                self._set_analytics_visible(False)
                if not self._win_d.isVisible():
                    self._win_d.show()
            else:
                self._win_d.hide()
                self._set_analytics_visible(show_d)
    
        # --------------------------------------------------
        # Apply view + points
        # --------------------------------------------------
        self._apply_view_to_canvas()
        self._refresh_points()
    
        # --------------------------------------------------
        # Other header toggles
        # --------------------------------------------------
        bm = str(self.store.get(MAP_VIEW_BASEMAP, "streets"))
        gd = bool(self.store.get(MAP_VIEW_SHOW_GRID, False))
        lg = bool(
            self.store.get(
                MAP_VIEW_SHOW_COLORBAR,
                True,
            )
        )
        mm = str(self.store.get(MAP_MEASURE_MODE, "off"))
    
        self.head.set_basemap(bm)
        self.head.set_grid_checked(gd)
        self.head.set_legend_checked(lg)
        self.head.set_measure_mode(mm)
    
        bms = list(self.store.get(MAP_BOOKMARKS, []) or [])
        names = []
        for it in bms:
            names.append(str(it.get("name", "")))
        names = [n for n in names if n.strip()]
        self.head.set_bookmarks(names)
    

    
    def _set_focus_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
    
        if enabled:
            self._dock_a.set_open(False)
            self._dock_c.set_open(False)
            self._win_a.hide()
            self._win_c.hide()
    
            self.head.set_data_checked(False)
            self.head.set_view_checked(False)
    
            self._set_analytics_visible(False)
    
        self._layout_overlays()


    def _set_analytics_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible:
            self.panel_d.expand()
        else:
            self.panel_d.collapse()
    
        self._layout_overlays()

                
    def _infer_city_from_path(self, p: str) -> tuple[str, str, str]:
        try:
            pp = Path(p).resolve()
        except Exception:
            return "", "", ""
        for parent in pp.parents:
            parsed = parse_city_dir(parent.name)
            if parsed:
                return parsed  # (city, model, stage)
        return "", "", ""

    def _on_focus_critical(self) -> None:
        """Filter view to only critical points."""
        hs = self._last_hs_payload
        if not hs:
            return

        # 1. Identify Critical Hotspots
        crit = [
            h for h in hs
            if h.get("sev", "high") == "critical"
        ]
        
        # Fallback to high if no critical
        if not crit:
            crit = [
                h for h in hs
                if h.get("sev") == "high"
            ]
        
        if not crit:
            return

        # 2. Calculate Bounding Box 
        try:
            lats = [float(c["lat"]) for c in crit]
            lons = [float(c["lon"]) for c in crit]
            
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)

            # 3. Call map fitBounds (Safe new method)
            self.canvas.fit_bounds(
                min_lon, min_lat, max_lon, max_lat
            )
        except (ValueError, KeyError):
            pass
        
    def _on_issue_warning(self) -> None:
        """Trigger the policy brief export."""
        self._on_export_requested("policy_brief")
