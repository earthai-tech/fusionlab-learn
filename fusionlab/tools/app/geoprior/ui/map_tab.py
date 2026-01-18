# geoprior/ui/map_tab.py
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

from ..config.store import GeoConfigStore
from .map.analytics_panel import CollapsibleAnalyticsPanel
from .map.canvas import ForecastMapView
from .map.data_panel import AutoHideDataPanel
from .map.head import MapHeadBar
from .map.view_panel import AutoHideViewPanel
from .map.utils import parse_city_dir  
from .map.hotspots import (
    HotspotCfg,
    build_points,
    compute_hotspots,
    hotspots_payload,
)
from .map.coord_utils import ( 
    ensure_lonlat,
    parse_epsg, 
    df_to_lonlat
)
from .map.interpretation import (
    cfg_from_get,
    apply_interp,
    rows_for_export,
    geojson_from_rows,
    dump_geojson,
    policy_brief_md,
)

_MAP_DEFAULTS = {
    "map.engine": "leaflet",
    "map.coord_mode": "lonlat",
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
}


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

        self._last_hs_payload = []
        self._last_hs_ctx = {}

        self.head = MapHeadBar(parent=self)
        self.nav_a = AutoHideDataPanel(
            store=self.store,
            parent=self,
        )
        self.canvas = ForecastMapView(parent=self)
        self.nav_c = AutoHideViewPanel(
            store=self.store,
            parent=self,
        )
        # in MapTab.__init__:
        self.panel_d = CollapsibleAnalyticsPanel(
            store=self.store,
            parent=self,
        )

        self._split_main = QSplitter(Qt.Horizontal, self)
        self._split_vert = QSplitter(Qt.Vertical, self)

        self._sizes_main: Optional[list[int]] = None
        self._sizes_vert: Optional[list[int]] = None
        self._hover_lock = False

        self._build_ui()
        self._apply_defaults()
        self._connect_signals()

    def set_available_columns(self, cols: Sequence[str]) -> None:
        self.head.set_available_columns(cols)

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        lay.addWidget(self.head)

        self._split_main.setChildrenCollapsible(False)
        self._split_main.addWidget(self.nav_a)
        self._split_main.addWidget(self.canvas)
        self._split_main.addWidget(self.nav_c)

        self._split_main.setStretchFactor(0, 0)
        self._split_main.setStretchFactor(1, 1)
        self._split_main.setStretchFactor(2, 0)
        self._split_main.setHandleWidth(2)

        self._split_vert.setChildrenCollapsible(False)
        self._split_vert.addWidget(self._split_main)
        self._split_vert.addWidget(self.panel_d)

        lay.addWidget(self._split_vert, 1)

        self._set_analytics_visible(False)
        self._split_main.setSizes([320, 1, 320])
        self._split_vert.setSizes([1, 0])

    def _apply_defaults(self) -> None:
        cfg = self.store.cfg
        if not self.store.get("map.utm_epsg", None):
            self.store.set(
                "map.utm_epsg",
                int(getattr(cfg, "utm_epsg", 32631)),
            )
        if not self.store.get("map.coord_epsg", None):
            self.store.set(
                "map.coord_epsg",
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

            if not self.store.get("map.x_col", ""):
                self.store.set("map.x_col", x0)
            if not self.store.get("map.y_col", ""):
                self.store.set("map.y_col", y0)
            if not self.store.get("map.z_col", ""):
                self.store.set("map.z_col", z0)

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
            lambda v: self.store.set("map.x_col", v),
        )
        self.head.y_changed.connect(
            lambda v: self.store.set("map.y_col", v),
        )
        self.head.z_changed.connect(
            lambda v: self.store.set("map.z_col", v),
        )

        self.canvas.request_focus_mode.connect(
            self._on_focus_toggled,
        )
        
        self.nav_a.width_changed.connect(
            self._on_left_width_changed,
        )
        self.nav_c.width_changed.connect(
            self._on_right_width_changed,
        )
        
        self.nav_a.pinned_changed.connect(
            self._on_left_pinned,
        )
        self.nav_c.pinned_changed.connect(
            self._on_right_pinned,
        )
        self.nav_c.export_requested.connect(
            self._on_export_requested
        )

        self.nav_a.columns_changed.connect(
            self._on_map_cols,
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
        self.canvas.point_clicked.connect(self._on_map_point_clicked)

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

    def _on_coord_epsg_changed(self, epsg: int) -> None:
        try:
            v = int(epsg)
        except Exception:
            return
        if v <= 0:
            return
        self.store.set("map.coord_epsg", v)

    def _on_basemap_changed(self, key: str) -> None:
        self.store.set("map.view.basemap", str(key))

    def _on_grid_toggled(self, on: bool) -> None:
        self.store.set("map.view.show_grid", bool(on))

    def _on_legend_toggled(self, on: bool) -> None:
        self.store.set(
            "map.view.show_colorbar",
            bool(on),
        )

    def _on_measure_mode_changed(self, mode: str) -> None:
        self.store.set("map.measure_mode", str(mode))
     
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
            self.store.set("map.bookmarks", [])
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

        bms = list(self.store.get("map.bookmarks", []) or [])
        bms.append({"name": name, "view": snap})
        self.store.set("map.bookmarks", bms)

    def _bookmark_goto(self, name: str) -> None:
        bms = list(self.store.get("map.bookmarks", []) or [])
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
        # If focus mode hides analytics, exit focus
        if bool(self.store.get("map.focus_mode", False)):
            self.store.set("map.focus_mode", False)
    
        # Ensure analytics is visible and select Inspector tab
        if not bool(self.store.get("map.show_analytics", False)):
            self.store.set("map.show_analytics", True)
            self._set_analytics_visible(True)
    
        try:
            self.panel_d.tabs.setCurrentWidget(self.panel_d.inspector)
            self.panel_d.inspector.set_xy(float(x), float(y))
        except Exception:
            pass


    def _on_swap_xy(self) -> None:
        x = str(self.store.get("map.x_col", ""))
        y = str(self.store.get("map.y_col", ""))

        with self.store.batch():
            self.store.set("map.x_col", y)
            self.store.set("map.y_col", x)

    def _on_reset_xyz(self) -> None:
        cfg = self.store.cfg
        x0 = str(getattr(cfg, "lon_col", "") or "")
        y0 = str(getattr(cfg, "lat_col", "") or "")
        z0 = str(getattr(cfg, "subs_col", "") or "")

        with self.store.batch():
            self.store.set("map.x_col", x0)
            self.store.set("map.y_col", y0)
            self.store.set("map.z_col", z0)

    def _on_view_changed(self, _snap) -> None:
        self._apply_view_to_canvas()
        self._refresh_points()

    def _on_map_cols(self, cols) -> None:
        cols = list(cols or [])
        self.head.set_available_columns(cols)
    
        x = str(self.store.get("map.x_col", "")).strip()
        y = str(self.store.get("map.y_col", "")).strip()
        z = str(self.store.get("map.z_col", "")).strip()
    
        if x and x not in cols:
            self.store.set("map.x_col", "")
        if y and y not in cols:
            self.store.set("map.y_col", "")
        if z and z not in cols:
            self.store.set("map.z_col", "")
    
    def _on_active_file(self, path: str) -> None:
        self._auto_fit = True
        
        self.store.set("map.active_file", str(path))
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

    def _on_left_pinned(self, pinned: bool) -> None:
        if self._hover_lock:
            return

        if bool(self.store.get("map.focus_mode", False)):
            return
    
        if pinned:
            self._apply_split_width(
                left=self.nav_a.expanded_width(),
                right=None,
            )
            return

        self._apply_split_width(
            left=self.nav_a.handle_width(),
            right=None,
        )

    
    def _on_right_pinned(self, pinned: bool) -> None:
        if self._hover_lock:
            return
        
        if bool(self.store.get("map.focus_mode", False)):
            return
    
        if pinned:
          self._apply_split_width(
              left=None,
              right=self.nav_c.expanded_width(),
          )
          return

        # self._apply_split_width(
        #     left=None,
        #     right=int(self.nav_c._handle_w),  # or 22
        # )
        
        self._apply_split_width(
            left=None, 
            right=self.nav_c.handle_width(),
        )

    def _on_left_width_changed(self, w: int) -> None:
        if self._hover_lock:
            return
        if bool(self.store.get("map.focus_mode", False)):
            return
        self._apply_split_width(left=int(w), right=None)
    
    def _on_right_width_changed(self, w: int) -> None:
        if self._hover_lock:
            return
        if bool(self.store.get("map.focus_mode", False)):
            return
        self._apply_split_width(left=None, right=int(w))

    def _refresh_points(self) -> None:
        # --------------------------------------------------
        # Helpers
        # --------------------------------------------------
        def _clear_all() -> None:
            self.canvas.clear_points()
            try:
                self.canvas.clear_hotspots()
            except Exception:
                pass
    
        # --------------------------------------------------
        # Active dataset
        # --------------------------------------------------
        p = str(self.store.get("map.active_file", "")).strip()
        if not p:
            _clear_all()
            return
    
        fp = Path(p)
        if not fp.exists():
            _clear_all()
            return
    
        # --------------------------------------------------
        # Column mapping (X/Y + value)
        # --------------------------------------------------
        x = str(self.store.get("map.x_col", "")).strip()
        y = str(self.store.get("map.y_col", "")).strip()
    
        z0 = str(self.store.get("map.z_col", "")).strip()
        v0 = str(self.store.get("map.value_col", "")).strip()
        v = v0 or z0
    
        if not x or not y or not v:
            _clear_all()
            return
    
        # --------------------------------------------------
        # Optional time filtering (viewer selection)
        # --------------------------------------------------
        t = str(self.store.get("map.time_col", "")).strip()
        tv = str(self.store.get("map.time_value", "")).strip()
    
        use = [c for c in (x, y, v, t) if c]
        use = list(dict.fromkeys(use))
    
        # --------------------------------------------------
        # Read CSV (prefer minimal columns; fallback full)
        # --------------------------------------------------
        try:
            df = pd.read_csv(fp, usecols=use)
        except Exception:
            try:
                df = pd.read_csv(fp)
            except Exception:
                _clear_all()
                return
    
        df_all = df
    
        # --------------------------------------------------
        # Viewer slice by time value (df_view)
        # --------------------------------------------------
        df_view = df_all
        if tv and t and t in df_all.columns:
            s = df_all[t]
            if pd.api.types.is_numeric_dtype(s):
                try:
                    df_view = df_all[s == float(tv)]
                except Exception:
                    df_view = df_all
            else:
                df_view = df_all[s.astype(str) == str(tv)]
    
        # --------------------------------------------------
        # Validate required columns exist
        # --------------------------------------------------
        if x not in df_all.columns or y not in df_all.columns:
            _clear_all()
            return
    
        if v not in df_all.columns:
            _clear_all()
            return
    
        # --------------------------------------------------
        # Normalize to canonical schema: lon/lat/v
        # --------------------------------------------------
        out = build_points(df_view, x=x, y=y, v=v)
    
        # --------------------------------------------------
        # Coords: ensure lon/lat degrees for Leaflet
        # --------------------------------------------------
        mode = str(
            self.store.get("map.coord_mode", "lonlat")
        ).strip().lower()
    
        utm_epsg = parse_epsg(
            self.store.get("map.utm_epsg", None)
        )
        if utm_epsg is None:
            utm_epsg = parse_epsg(
                self.store.get("utm_epsg", None)
            )
    
        src_epsg = parse_epsg(
            self.store.get("map.coord_epsg", None)
        )
        if src_epsg is None:
            src_epsg = parse_epsg(
                self.store.get("map.src_epsg", None)
            )
        if src_epsg is None:
            src_epsg = parse_epsg(
                self.store.get("coord_src_epsg", None)
            )
    
        try:
            out, ok, msg = ensure_lonlat(
                out,
                mode=mode,
                utm_epsg=utm_epsg,
                src_epsg=src_epsg,
            )
            # If there’s a problem, display the error message
            if not ok:
                # self.lb_status.setText(f"Error: {msg}")
                # self.lb_status.setProperty("state", "warn")
                # self.lb_status.setVisible(True)
                return
            
        except Exception:
            out = df_to_lonlat(
                out,
                x="lon",
                y="lat",
                mode=mode,
                utm_epsg=utm_epsg,
                src_epsg=src_epsg,
            )
            ok = not out.empty
    
        if (not ok) or out.empty:
            # self.lb_status.setText("Error: Invalid coordinates")
            # self.lb_status.setProperty("state", "warn")
            # self.lb_status.setVisible(True)
            _clear_all()
            return
    
        # --------------------------------------------------
        # Cap points for performance
        # --------------------------------------------------
        max_pts = 80000
        if len(out) > max_pts:
            out = out.sample(n=max_pts, random_state=0)
    
        # --------------------------------------------------
        # Render points (style from view panel)
        # --------------------------------------------------
        snap = self._view_snapshot()
    
        rad = int(snap.get("marker_size") or 6)
        opa = float(snap.get("marker_opacity") or 0.9)
        leg = bool(snap.get("show_colorbar", True))
    
        auto = bool(snap.get("autoscale", True))
        vmin = None if auto else snap.get("vmin", None)
        vmax = None if auto else snap.get("vmax", None)
    
        cmap = str(snap.get("colormap", "viridis"))
        inv = bool(snap.get("cmap_invert", False))
    
        pts = out.to_dict("records")
    
        self.canvas.set_points(
            pts,
            vmin=vmin,
            vmax=vmax,
            radius=rad,
            opacity=opa,
            label=v,
            show_legend=leg,
            cmap=cmap,
            invert=inv,
        )
    
        # --------------------------------------------------
        # Hotspots (attention layer)
        # --------------------------------------------------
        k_en = "map.view.hotspots.enabled"
        hot_on = bool(self.store.get(k_en, False))
    
        if not hot_on:
            try:
                self.canvas.clear_hotspots()
            except Exception:
                pass
            self._last_hs_payload = []
            self._last_hs_ctx = {}
        else:
            k = "map.view.hotspots."
    
            hs_mode = str(self.store.get(k + "mode", "auto"))
            hs_mode = hs_mode.lower()
    
            method = str(self.store.get(k + "method", "grid"))
            method = method.lower()
    
            metric = str(self.store.get(k + "metric", "high"))
            metric = metric.lower()
    
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
                pts_for_hs = out[["lon", "lat", "v"]].copy()
    
                if time_agg != "current" and t and (t in df_all.columns):
                    try:
                        pts_for_hs = build_points(
                            df_all,
                            x=x,
                            y=y,
                            v=v,
                            t=t,
                        )
                    except TypeError:
                        pts_for_hs = build_points(
                            df_all,
                            x=x,
                            y=y,
                            v=v,
                        )
    
                    # Ensure lon/lat for full-data hotspots too
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

            labels = bool(
                self.store.get(k + "labels", True)
            )
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

        # --------------------------------------------------
        # Auto-fit (once per new active dataset)
        # --------------------------------------------------
        if self._auto_fit:
            self.canvas.fit_points()
            self._auto_fit = False


    
    def _freeze_hover(self, ms: int = 250) -> None:
        if bool(self.store.get("map.focus_mode", False)):
            return
    
        self._hover_lock = True
        self.nav_a.set_hover_enabled(False)
        self.nav_c.set_hover_enabled(False)
    
        # Optional: immediately collapse C if it is not pinned.
        if not self.nav_c.is_pinned():
            self.nav_c.collapse(immediate=True, force=True)
            self._apply_split_width(
                left=None, right=self.nav_c.handle_width())
    
        QTimer.singleShot(ms, self._unfreeze_hover)
    
    def _unfreeze_hover(self) -> None:
        self._hover_lock = False
        if bool(self.store.get("map.focus_mode", False)):
            return
        self.nav_a.set_hover_enabled(True)
        self.nav_c.set_hover_enabled(True)

    def _view_snapshot(self) -> dict:
        keys = [
            "map.view.basemap",
            "map.view.basemap_style",
            "map.view.tiles_opacity",
            "map.view.colormap",
            "map.view.cmap_invert",
            "map.view.autoscale",
            "map.view.vmin",
            "map.view.vmax",
            "map.view.marker_size",
            "map.view.marker_opacity",
            "map.view.show_colorbar",
            "map.view.show_grid",
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

    def _apply_split_width(
        self,
        *,
        left: Optional[int],
        right: Optional[int],
    ) -> None:
        sizes = list(self._split_main.sizes())
        if len(sizes) != 3:
            return
    
        total = int(sum(sizes)) or 1
    
        l = int(left) if left is not None else int(sizes[0])
        r = int(right) if right is not None else int(sizes[2])
    
        l = max(0, l)
        r = max(0, r)
    
        c = total - l - r
        if c < 1:
            c = 1
            # If overflow, shrink the changing side.
            if left is not None:
                l = max(0, total - r - c)
            if right is not None:
                r = max(0, total - l - c)
    
        self._split_main.setSizes([l, c, r])

    def _on_files_selected(self, paths) -> None:
        # Store already updated in panel; keep hook.
        paths = list(paths or [])
        self.head.set_dataset_count(len(paths))


    def _on_cfg_replaced(self, _cfg) -> None:
        self._apply_defaults()

    def _on_store_changed(self, keys) -> None:
        keys = set(keys or [])
        if not keys:
            return
    
        if "map.data_source" in keys:
            self._freeze_hover(ms=250)
    
        ui_keys = {
            "map.engine",
            "map.coord_mode",
            "map.x_col",
            "map.y_col",
            "map.z_col",
            "map.focus_mode",
            "map.show_analytics",
            "map.active_file",
            "map.view.basemap",
            "map.view.show_grid",
            "map.view.show_colorbar",
            "map.bookmarks",
            "map.measure_mode",
            "map.value_col",
            "map.time_col",
            "map.time_value",
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
                for k in view_keys
            )
            if needs_pts:
                self._refresh_points()
            return

    def _on_engine_changed(self, engine: str) -> None:
        self.store.set("map.engine", str(engine))

    def _on_coord_changed(self, mode: str) -> None:
        self.store.set("map.coord_mode", str(mode))

    def _on_focus_toggled(self, enabled: bool) -> None:
        self.store.set("map.focus_mode", bool(enabled))

    def _on_analytics_toggled(self, enabled: bool) -> None:
        self.store.set("map.show_analytics", bool(enabled))

    def _sync_from_store(self) -> None:
        engine = str(self.store.get("map.engine", "leaflet"))
        coord = str(self.store.get("map.coord_mode", "lonlat"))

        x = str(self.store.get("map.x_col", ""))
        y = str(self.store.get("map.y_col", ""))
        z = str(self.store.get("map.z_col", ""))

        focus = bool(self.store.get("map.focus_mode", False))
        show_d = bool(self.store.get("map.show_analytics", False))

        epsg = int(
            self.store.get(
                "map.coord_epsg",
                getattr(self.store.cfg, "coord_src_epsg", 4326),
            )
        )
        self.head.set_epsg(epsg)
        self.head.set_engine(engine)
        self.head.set_coord_mode(coord)
        self.head.set_xyz(x=x, y=y, z=z)
        self.head.set_focus_checked(focus)
        self.head.set_analytics_checked(show_d)

        self.canvas.set_focus_checked(focus)

        self._set_focus_mode(focus)
        if not focus:
            self._set_analytics_visible(show_d)
        
        self._apply_view_to_canvas()
        self._refresh_points()
        
        bm = str(self.store.get("map.view.basemap", "streets"))
        gd = bool(self.store.get("map.view.show_grid", False))
        lg = bool(self.store.get(
            "map.view.show_colorbar",
            True,
        ))
        mm = str(self.store.get("map.measure_mode", "off"))
        
        self.head.set_basemap(bm)
        self.head.set_grid_checked(gd)
        self.head.set_legend_checked(lg)
        self.head.set_measure_mode(mm)
        
        bms = list(self.store.get("map.bookmarks", []) or [])
        names = []
        for it in bms:
            names.append(str(it.get("name", "")))
        names = [x for x in names if x.strip()]
        self.head.set_bookmarks(names)



    def _set_focus_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)

        if enabled:
            if self._sizes_main is None:
                self._sizes_main = self._split_main.sizes()
            if self._sizes_vert is None:
                self._sizes_vert = self._split_vert.sizes()

            self.nav_a.setVisible(False)
            self.nav_c.setVisible(False)
            self.panel_d.setVisible(False)

            self.nav_a.set_hover_enabled(False)
            self.nav_c.set_hover_enabled(False)

            self._split_main.setSizes([0, 1, 0])
            self._split_vert.setSizes([1, 0])
            return

        self.nav_a.setVisible(True)
        self.nav_c.setVisible(True)

        self.nav_a.set_hover_enabled(True)
        self.nav_c.set_hover_enabled(True)

        if self._sizes_main:
            self._split_main.setSizes(self._sizes_main)
        else:
            self._split_main.setSizes([320, 1, 320])

        show_d = bool(self.store.get("map.show_analytics", False))
        self._set_analytics_visible(show_d)

        if self._sizes_vert and show_d:
            self._split_vert.setSizes(self._sizes_vert)
        elif show_d:
            self._split_vert.setSizes([750, 250])
        else:
            self._split_vert.setSizes([1, 0])

        self._sizes_main = None
        self._sizes_vert = None

    def _set_analytics_visible(self, visible: bool) -> None:
        visible = bool(visible)
        self.panel_d.setVisible(visible)

        if visible:
            self.panel_d.expand()
            self._split_vert.setSizes([750, 250])
        else:
            self.panel_d.collapse()
            self._split_vert.setSizes([1, 0])
            
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
