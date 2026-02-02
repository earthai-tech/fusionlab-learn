# geoprior/ui/map/canvas.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.canvas

ForecastMapView (B) with QWebEngine + Leaflet base.

Features
--------
- Safe JS API defined before Leaflet init
- Pending JS queue until loadFinished
- Overlay controls: + / - / Fit / Focus
- Placeholder fallback when PyQtWebEngine is missing
"""

from __future__ import annotations

from pathlib import Path
import json
import math 

from typing import Any, Dict, List, Optional, Sequence

from PyQt5.QtCore import (
    Qt,
    QUrl,
    QTimer,
    pyqtSignal,
    QObject,
    pyqtSlot,
)
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from .basemap.basemap import resolve_basemap
from .engines.leaflet_html import _leaflet_html
from .engines.maplibre_html import maplibre_html
from .engines.google_html import google_html

try:
    from PyQt5.QtWebEngineWidgets import (
        QWebEngineSettings,
        QWebEngineView,
        QWebEnginePage,
    )
except Exception:  # pragma: no cover
    QWebEngineView = None  # type: ignore

try:
    from PyQt5.QtWebChannel import QWebChannel
except Exception:  # pragma: no cover
    QWebChannel = None  # type: ignore

_ASSET_ROOT = (
    Path(__file__).resolve().parent / "assets"
)

class _LogPage(QWebEnginePage):
    def javaScriptConsoleMessage(
        self,
        level: QWebEnginePage.JavaScriptConsoleMessageLevel,
        message: str,
        line: int,
        source_id: str,
    ) -> None:
        try:
            print(
                f"[js:{int(level)}] "
                f"{source_id}:{line} - {message}"
            )
        except Exception:
            pass

class _GeoPriorBridge(QObject):
    point_clicked = pyqtSignal(float, float)

    @pyqtSlot(float, float)
    def pointClicked(self, x: float, y: float) -> None:
        # x,y here are "data coords" (for lon/lat: x=lon, y=lat)
        try:
            self.point_clicked.emit(float(x), float(y))
        except Exception:
            pass

class ForecastMapView(QFrame):
    request_focus_mode = pyqtSignal(bool)
    point_clicked = pyqtSignal(float, float)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ForecastMapView")
        self.setFrameShape(QFrame.StyledPanel)

        self._ready = False
        self._page_loaded = False
        self._pending_js: List[str] = []

        self._last_points: List[Dict[str, Any]] = []
        self._last_label: str = "Z"
        self._last_vmin: Optional[float] = None
        self._last_vmax: Optional[float] = None

        self._last_layer_kind = "points"
        self._last_layer_opts: Dict[str, Any] = {}


        self._view: Dict[str, Any] = {}
        self._last_hotspots: List[Dict[str, Any]] = []
        self._hot_opts: Dict[str, Any] = {
            "show": True,
            "pulse": True,
        }

        # Engine management
        self._engine_req = "leaflet"
        self._engine_active = "leaflet"
        self._google_key = ""

        self._probe_tries = 0
        self._probe_max = 10

        self._web: Optional[QWebEngineView]
        self._placeholder: Optional[QLabel]

        self._build_ui()

        if self._web is not None:
            self._web.loadFinished.connect(self._on_loaded)

        self._load_map()

    # -----------------------------
    # Engine API
    # -----------------------------
    def _leaflet_index_url(self) -> Optional[QUrl]:
        p = _ASSET_ROOT / "leaflet" / "index.html"
        try:
            if p.exists():
                return QUrl.fromLocalFile(str(p))
        except Exception:
            pass
        return None

    def set_vectors(self, vectors: List[Dict[str, float]]) -> None:
        """
        Draw direction arrows.
        vectors = [{lat, lon, angle, mag}, ...]
        """
        js_data = json.dumps(vectors)
        js = (
            "if (window.__GeoPriorMap && window.__GeoPriorMap.setVectors) {"
            f"  window.__GeoPriorMap.setVectors({js_data});"
            "}"
        )
        self._run_js(js)
        
    def set_engine(
        self,
        engine: str,
        *,
        google_key: str = "",
    ) -> None:
        eng = str(engine or "leaflet").strip().lower()
        if eng not in ("leaflet", "maplibre", "google"):
            eng = "leaflet"

        self._engine_req = eng

        # key = str(google_key or "").strip()
        # if key:
        #     self._google_key = key
   
        self._google_key = str(google_key or "").strip()

        self._load_map()

    # -----------------------------
    # WebEngine lifecycle
    # -----------------------------
    def _load_map(self) -> None:
        if self._web is None:
            return

        self._ready = False
        self._page_loaded = False

        self._pending_js.clear()

        st = self._web.settings()
        st.setAttribute(
            QWebEngineSettings.LocalContentCanAccessRemoteUrls,
            True,
        )
        st.setAttribute(
            QWebEngineSettings.LocalContentCanAccessFileUrls,
            True,
        )
        st.setAttribute(
            QWebEngineSettings.WebGLEnabled,
            True,
        )

        html, active = self._engine_html()
        self._engine_active = active
        
        # todo : to use consol output later
        print(
            f"[map] request={self._engine_req} "
            f"active={self._engine_active}"
        )

        # WebChannel must exist before JS tries to connect
        self._bridge = None
        self._channel = None
        if QWebChannel is not None:
            self._bridge = _GeoPriorBridge(self)
            self._channel = QWebChannel(self._web.page())
            self._channel.registerObject("bridge", self._bridge)
            self._web.page().setWebChannel(self._channel)
            self._bridge.point_clicked.connect(self.point_clicked)

        self._web.setHtml(
            html,
            QUrl("https://geoprior.local/"),
        )
        
        # XXX TO OPTIMIZE: We comment this until everything refactored in 
        # using assets/, common/ , layers/with js module are stables. 
        # -----------------------------------------------
        # # Load engine page.
        # # Leaflet: prefer local index.html so relative
        # # assets (style.css, layers/*.js) work.
        # if self._engine_active == "leaflet":
        #     url = self._leaflet_index_url()
        #     if url is not None:
        #         self._web.setUrl(url)
        #         return

        # # Fallback: inline HTML mode (also used by
        # # google/maplibre engines for now).
        # self._web.setHtml(
        #     html,
        #     QUrl("https://geoprior.local/"),
        # )
        # # -------------------------------------------------------
        
        
    def _engine_html(self) -> tuple[str, str]:
        req = self._engine_req

        if req == "maplibre":
            return maplibre_html(), "maplibre"

        if req == "google":
            if not self._google_key:
                return _leaflet_html(), "leaflet"
            return google_html(self._google_key), "google"

        return _leaflet_html(), "leaflet"


    def _on_loaded(self, ok: bool) -> None:
        self._page_loaded = bool(ok)
    
        if not self._page_loaded:
            if self._engine_active != "leaflet":
                self._fallback_leaflet()
            return
    
        self._probe_tries = 0
        self._schedule_probe()
    
    
    def _schedule_probe(self) -> None:
        delay = 0
        if self._engine_active == "google":
            delay = 250
        elif self._engine_active == "maplibre":
            delay = 250
    
        QTimer.singleShot(delay, self._probe_engine)
    
    
    def _probe_engine(self) -> None:
        if self._web is None:
            return
        if not self._page_loaded:
            return
    
        eng = self._engine_active
    
        if eng == "leaflet":
            js = "typeof L !== 'undefined'"
        elif eng == "maplibre":
            js = (
                "window.__GeoPriorMap ? {"
                "ready: !!window.__GeoPriorMap.__ready,"
                "failed: !!window.__GeoPriorMap.__failed,"
                "err: window.__GeoPriorMap.__err || ''"
                "} : null"
            )
        else:
            js = (
                "typeof google !== 'undefined' && "
                "typeof google.maps !== 'undefined'"
            )
    
        self._web.page().runJavaScript(js, self._on_probe_result)
    
    
    def _on_probe_result(self, ok: Any) -> None:
        eng = self._engine_active
        is_ok = False
        failed = False
        err = ""
    
        if eng == "maplibre" and isinstance(ok, dict):
            is_ok = bool(ok.get("ready"))
            failed = bool(ok.get("failed"))
            err = str(ok.get("err") or "")
        else:
            is_ok = bool(ok)
    
        print(
            f"[map] probe eng={eng} ok={ok} "
            f"tries={self._probe_tries} "
            f"failed={failed} err={err!r}"
        )
        
        if is_ok:
            self._finalize_ready()
            return
    
        if failed and eng == "maplibre":
            print("[map] maplibre failed:", err)
            self._fallback_leaflet()
            return
    
        if eng in ("google", "maplibre"):
            if self._probe_tries < self._probe_max:
                self._probe_tries += 1
                QTimer.singleShot(300, self._probe_engine)
                return
    
        if eng != "leaflet":
            self._fallback_leaflet()
    
    
    def _finalize_ready(self) -> None:
        if self._web is None:
            return
    
        self._ready = True
    
        if self._pending_js:
            q = list(self._pending_js)
            self._pending_js.clear()
            for js in q:
                self._web.page().runJavaScript(str(js))
    
        try:
            self.apply_view(self._view)
        except Exception:
            pass

    def _fallback_leaflet(self) -> None:
        self._engine_req = "leaflet"
        self._engine_active = "leaflet"
        self._load_map()


    # -----------------------------
    # Public API
    # -----------------------------
    def set_focus_checked(self, checked: bool) -> None:
        self.btn_focus.blockSignals(True)
        self.btn_focus.setChecked(bool(checked))
        self.btn_focus.blockSignals(False)

    # def clear_points(self) -> None:
    #     self._run_js(
    #         "if (window.__GeoPriorMap) {"
    #         "  window.__GeoPriorMap.clearPoints();"
    #         "}",
    #     )
    def clear_points(self) -> None:
        self._last_points = []
        self._last_layer_kind = "points"
        self._last_layer_opts = {}

        self._run_js(
            "if(window.__GeoPriorMap) "
            "window.__GeoPriorMap.clearPoints();"
        )

    def fit_points(self) -> None:
        self._run_js(
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.fitPoints();"
            "}",
        )

    def fit_bounds(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> None:
        """
        Zoom map to specific bounding box (New).
        Used by Focus Critical alert.
        """
        try:
            # Leaflet expects [[lat1, lon1], [lat2, lon2]]
            c1 = [float(min_lat), float(min_lon)]
            c2 = [float(max_lat), float(max_lon)]
        except (ValueError, TypeError):
            return

        js_coords = json.dumps([c1, c2])
        js = (
            "if (window.__GeoPriorMap && "
            "    window.__GeoPriorMap.fitBounds) {"
            f"  window.__GeoPriorMap.fitBounds({js_coords});"
            "}"
        )
        self._run_js(js)

    def clear_hotspots(self) -> None:
        self._last_hotspots = []
        self._run_js(
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.clearHotspots();"
            "}",
        )

    def show_hotspots(self, on: bool) -> None:
        self._hot_opts["show"] = bool(on)
        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.showHotspots("
            f"{json.dumps(bool(on))}"
            ");"
            "}"
        )
        self._run_js(js)

    def set_hotspots(
        self,
        hotspots: Sequence[Dict[str, Any]],
        *,
        show: bool = True,
        style: str = "pulse",          # "pulse" | "glow"
        pulse: bool = True,
        pulse_speed: float = 1.0,      # 0.2..3.0
        ring_km: float = 0.8,          # base radius in km
        labels: bool = True,
    ) -> None:
        hs = list(hotspots or [])
        opts: Dict[str, Any] = {
            "show": bool(show),
            "style": str(style or "pulse"),
            "pulse": bool(pulse),
            "pulseSpeed": float(pulse_speed),
            "ringKm": float(ring_km),
            "labels": bool(labels),
        }
    
        # Cache for view re-apply / reload safety.
        self._last_hotspots = hs
        self._hot_opts = dict(opts)
    
        js_h = json.dumps(hs, separators=(",", ":"))
        js_o = json.dumps(opts, separators=(",", ":"))
    
        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.setHotspots("
            f"{js_h}, {js_o}"
            ");"
            "}"
        )
        self._run_js(js)

    def _norm_points(
        self,
        points: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for p in points or []:
            if not isinstance(p, dict):
                continue
    
            try:
                lat = float(p.get("lat"))
                lon = float(p.get("lon"))
                v = float(p.get("v"))
            except Exception:
                continue
    
            if not math.isfinite(lat):
                continue
            if not math.isfinite(lon):
                continue
            if not math.isfinite(v):
                continue
    
            out.append({"lat": lat, "lon": lon, "v": v})
    
        return out

    def set_layer(
        self,
        kind: str,
        points: Sequence[Dict[str, Any]],
        *,
        opts: Optional[Dict[str, Any]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        label: str = "Z",
        show_legend: bool = True,
        cmap: str = "viridis",
        invert: bool = False,
    ) -> None:
        k = str(kind or "points").strip().lower()
        o = dict(opts or {})

        self._last_layer_kind = k
        self._last_layer_opts = dict(o or {})
        self._last_label = str(label or "Z")
        self._last_vmin = vmin
        self._last_vmax = vmax

        pts = self._norm_points(points)
        self._last_points = list(pts)

        if k in ("points", "scatter"):
            self.set_points(
                pts,
                vmin=vmin,
                vmax=vmax,
                radius=int(o.get("radius", 6) or 6),
                opacity=float(o.get("opacity", 0.9) or 0.9),
                label=label,
                show_legend=show_legend,
                cmap=cmap,
                invert=invert,
            )
            return

        js_pts = json.dumps(pts, separators=(",", ":"))

        o.update(
            {
                "vmin": vmin,
                "vmax": vmax,
                "label": str(label or "Z"),
                "showLegend": bool(show_legend),
                "cmap": str(cmap or "viridis"),
                "invert": bool(invert),
            }
        )
        js_opt = json.dumps(o, separators=(",", ":"))

        fn = "setPoints"
        if k in ("hexbin_source", "hexbin"):
            fn = "setHexbin"
        elif k in (
            "contour_source",
            "contour",
            "contours",
        ):
            fn = "setContours"

        js = (
            "if (window.__GeoPriorMap) {"
            f"  if (window.__GeoPriorMap.{fn}) {{"
            f"    window.__GeoPriorMap.{fn}({js_pts}, {js_opt});"
            "  } else {"
            "    window.__GeoPriorMap.setPoints("
            f"{js_pts}, {js_opt}"
            "    );"
            "  }"
            "}"
        )
        self._run_js(js)

    def set_points(
        self,
        points: Sequence[Dict[str, Any]],
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        radius: int = 6,
        opacity: float = 0.9,
        label: str = "Z",
        show_legend: bool = True,
        cmap: str = "viridis",
        invert: bool = False,
    ) -> None:

        self._last_layer_kind = "points"
        self._last_layer_opts = {
            "radius": int(radius),
            "opacity": float(opacity),
        }

        # self._last_points = list(points or [])
        self._last_label = str(label or "Z")
        self._last_vmin = vmin
        self._last_vmax = vmax
        
        # Keep last view hints (optional)
        self._view["radius"] = int(radius)
        self._view["opacity"] = float(opacity)
        self._view["showLegend"] = bool(show_legend)

        pts = self._norm_points(points)
        self._last_points = list(pts)
        
        # pts = list(points or [])
        opts: Dict[str, Any] = {
            "vmin": vmin,
            "vmax": vmax,
            "radius": int(radius),
            "opacity": float(opacity),
            "label": str(label or "Z"),
            "showLegend": bool(show_legend),
            "cmap": str(cmap or "viridis"),
            "invert": bool(invert),
        }

        js_pts = json.dumps(pts, separators=(",", ":"))
        js_opt = json.dumps(opts, separators=(",", ":"))

        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.setPoints("
            f"{js_pts}, {js_opt}"
            ");"
            "}"
        )
        self._run_js(js)

    def apply_view(self, view: Dict[str, Any]) -> None:
        v = dict(view or {})
        self._view.update(v)
        
        base = str(v.get("basemap", "osm"))
        style = str(v.get("basemap_style", "light"))
        top = float(v.get("tiles_opacity", 1.0))
        
        # if not self._last_points:
        #     return
        # Apply basemap even if there are no points yet.
        # Points/hotspots redraw only if we have payload.

        radius = int(v.get("marker_size", 6))
        opacity = float(v.get("marker_opacity", 0.9))
    
        show_leg = bool(v.get("show_colorbar", True))
    
        autoscale = bool(v.get("autoscale", True))
        vmin = None if autoscale else v.get("vmin", None)
        vmax = None if autoscale else v.get("vmax", None)
    
        cmap = str(v.get("colormap", "viridis"))
        inv = bool(v.get("cmap_invert", False))
    
        self._set_basemap(base, style, top)
        if self._last_points:
            self._redraw_layer(
                radius=radius,
                opacity=opacity,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                invert=inv,
                show_legend=show_leg,
            )

        # Re-apply hotspots if present (optional).
        if self._last_hotspots:
            try:
                self.set_hotspots(
                    self._last_hotspots,
                    show=bool(self._hot_opts.get("show", True)),
                    style=str(self._hot_opts.get("style", "pulse")),
                    pulse=bool(self._hot_opts.get("pulse", True)),
                    pulse_speed=float(
                        self._hot_opts.get("pulseSpeed", 1.0)
                    ),
                    ring_km=float(self._hot_opts.get("ringKm", 0.8)),
                    labels=bool(self._hot_opts.get("labels", True)),
                )
            except Exception:
                pass
            
    def _redraw_layer(
        self,
        *,
        radius: int,
        opacity: float,
        vmin: Optional[float],
        vmax: Optional[float],
        cmap: str,
        invert: bool,
        show_legend: bool,
    ) -> None:
        if not self._last_points:
            return

        o = dict(self._last_layer_opts or {})
        o["radius"] = int(radius)
        o["opacity"] = float(opacity)

        self.set_layer(
            self._last_layer_kind,
            self._last_points,
            opts=o,
            vmin=vmin,
            vmax=vmax,
            label=self._last_label,
            show_legend=show_legend,
            cmap=cmap,
            invert=invert,
        )

    def _redraw_points(
        self,
        *,
        radius: int,
        opacity: float,
        vmin: Optional[float],
        vmax: Optional[float],
        cmap: str,
        invert: bool,
        show_legend: bool,
    ) -> None:
        # self.set_points(
        #     self._last_points,
        #     vmin=vmin,
        #     vmax=vmax,
        #     radius=radius,
        #     opacity=opacity,
        #     label=self._last_label,
        #     show_legend=show_legend,
        #     cmap=cmap,
        #     invert=invert,
        # )
        self._redraw_layer(
            radius=radius,
            opacity=opacity,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            invert=invert,
            show_legend=show_legend,
        )

    def _set_basemap(
        self, provider: str, 
        style: str, 
        opacity: float
        ) -> None:
        #     spec = resolve_basemap(
        #         self._engine_active,
        #         provider,
        #         style,
        #     )
        
        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.setBasemap("
            f"{json.dumps(provider)},"
            f"{json.dumps(style)},"
            f"{json.dumps(float(opacity))}"
            ");"
            "}"
        )
        self._run_js(js)

    
    # def _set_basemap(self, provider: str, style: str, opacity: float) -> None:
    #     spec = resolve_basemap(
    #         self._engine_active,
    #         provider,
    #         style,
    #     )
    #     spec["opacity"] = float(opacity)
    
    #     js = (
    #         "if (window.__GeoPriorMap) {"
    #         "  window.__GeoPriorMap.setBasemap("
    #         f"{json.dumps(spec)}"
    #         ");"
    #         "}"
    #     )
    #     self._run_js(js)

    # -----------------------------
    # UI
    # -----------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._container = QWidget(self)
        stack = QStackedLayout(self._container)
        stack.setStackingMode(QStackedLayout.StackAll)

        if QWebEngineView is None:
            self._web = None
            self._placeholder = QLabel(
                "Interactive map requires PyQtWebEngine.\n"
                "Install: pip install PyQtWebEngine",
                self._container,
            )
            self._placeholder.setAlignment(Qt.AlignCenter)
            stack.addWidget(self._placeholder)
        else:
            self._placeholder = None
            # self._web = QWebEngineView(self._container)
            # stack.addWidget(self._web)
            self._web = QWebEngineView(self._container)
            try:
                self._web.setPage(_LogPage(self._web))
            except Exception:
                pass
            stack.addWidget(self._web)

        self._overlay = self._make_overlay(self._container)
        stack.addWidget(self._overlay)

        root.addWidget(self._container, 1)

    def _make_overlay(self, parent: QWidget) -> QWidget:
        w = QWidget(parent)
        w.setAttribute(Qt.WA_TranslucentBackground, True)

        row = QHBoxLayout(w)
        row.setContentsMargins(10, 10, 10, 10)
        row.setSpacing(6)

        self.btn_plus = QToolButton(w)
        self.btn_plus.setText("+")

        self.btn_minus = QToolButton(w)
        self.btn_minus.setText("-")

        self.btn_fit = QToolButton(w)
        self.btn_fit.setText("Fit")

        self.btn_focus = QToolButton(w)
        self.btn_focus.setText("Focus")
        self.btn_focus.setCheckable(True)
        self.btn_focus.toggled.connect(self.request_focus_mode)

        for b in (
            self.btn_plus,
            self.btn_minus,
            self.btn_fit,
            self.btn_focus,
        ):
            b.setFixedHeight(26)

        self.btn_plus.clicked.connect(self._on_zoom_in)
        self.btn_minus.clicked.connect(self._on_zoom_out)
        self.btn_fit.clicked.connect(self.fit_points)

        row.addStretch(1)
        row.addWidget(self.btn_plus)
        row.addWidget(self.btn_minus)
        row.addWidget(self.btn_fit)
        row.addWidget(self.btn_focus)

        return w

    def _run_js(self, js: str) -> None:
        if self._web is None:
            return
        if not self._ready:
            self._pending_js.append(str(js))
            return
        self._web.page().runJavaScript(str(js))

    # -----------------------------
    # Overlay actions
    # -----------------------------
    def _on_zoom_in(self) -> None:
        self._run_js(
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.zoomIn();"
            "}",
        )

    def _on_zoom_out(self) -> None:
        self._run_js(
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.zoomOut();"
            "}",
        )
