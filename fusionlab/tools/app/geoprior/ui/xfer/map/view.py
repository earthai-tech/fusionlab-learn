# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>


"""
geoprior.ui.xfer.map.view

Leaflet host for Xfer map layers.

Implements the MapApi contract (see ui/xfer/types.py):
- set_centroids(...)
- set_layer(...)
- clear_layer(...)
- clear_layers()
- fit_layers(...)
- set_legend(...)
- clear()

Notes
-----
- Uses Leaflet CDN; requires internet access.
- If PyQtWebEngine is missing, we show a placeholder
  and all API calls become safe no-ops.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from PyQt5.QtCore import QUrl, Qt, QEvent, QTimer
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt5.QtWebEngineWidgets import (
        QWebEngineSettings,
        QWebEngineView  # noqa: F401
    )
except Exception:  # pragma: no cover
    QWebEngineView = None  # type: ignore

from ..types import MapPoint

# Imported from the centralized location as requested
from ...map.engines.xfer_leaflet_html import _LEAFLET_HTML


def _as_js_str(text: str) -> str:
    s = (text or "").replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    s = s.replace("\n", "\\n").replace("\r", "\\r")
    return s


def _json_min(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


class MapView(QWidget):
    """
    Leaflet host implementing MapApi.

    This widget is intentionally minimal:
    - only the web map + safe API calls
    - toolbar/buttons live in map/page.py
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._ready = False
        self._pending_js: list[str] = []
        self._overlays: list[QWidget] = []

        if QWebEngineView is None:
            self._web = None
            self._fallback = QLabel(
                "Interactive map requires PyQtWebEngine.\n"
                "Install: pip install PyQtWebEngine"
            )
            self._fallback.setAlignment(Qt.AlignCenter)
            self._fallback.setFrameShape(QFrame.StyledPanel)
        else:
            self._fallback = None
            self._web = QWebEngineView(self)
            st = self._web.settings()
            st.setAttribute(
                QWebEngineSettings
                .LocalContentCanAccessRemoteUrls,
                True,
            )
            self._web.setHtml(
                _LEAFLET_HTML,
                QUrl("https://geoprior.local/"),
            )
            self._web.loadFinished.connect(self._on_loaded)
            self._web.installEventFilter(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if self._web is not None:
            root.addWidget(self._web, 1)
            QTimer.singleShot(0, self._sync_overlays)
        else:
            root.addWidget(self._fallback, 1)

    def eventFilter(self, obj, ev):
        if obj is self._web and ev.type() in (
            QEvent.Resize,
            QEvent.Show,
        ):
            self._sync_overlays()
        return super().eventFilter(obj, ev)

    def _sync_overlays(self) -> None:
        if self._web is None:
            return
        r = self._web.rect()
        for w in self._overlays:
            w.setGeometry(r)
            w.raise_()

    def add_overlay(self, w: QWidget) -> None:
        if self._web is None:
            return
        w.setParent(self._web)
        w.setGeometry(self._web.rect())
        w.show()
        w.raise_()
        self._overlays.append(w)
    # -------------------------
    # Internal
    # -------------------------
    def _on_loaded(self, ok: bool) -> None:
        self._ready = bool(ok)
        if not self._ready:
            self._pending_js.clear()
            return

        if self._pending_js and self._web is not None:
            for js in self._pending_js:
                self._web.page().runJavaScript(js)
        self._pending_js.clear()

    def _run_js(self, js: str) -> None:
        if self._web is None:
            return
        if not self._ready:
            self._pending_js.append(js)
            return
        self._web.page().runJavaScript(js)

    # -------------------------
    # MapApi
    # -------------------------
    def clear(self) -> None:
        self.clear_layers()
        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.clearCentroids();"
            "}"
        )
        self._run_js(js)

    def clear_centroids(self) -> None:
        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.clearCentroids();"
            "}"
        )
        self._run_js(js)

    # -------------------------
    #  Links (polylines with arrows)
    # -------------------------
    def set_links(
        self,
        layer_id: str,
        name: str,
        links: Iterable[list],
        opts: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = _json_min(list(links))
        o = _json_min(opts or {})
        lid = _as_js_str(layer_id)
        nm = _as_js_str(name)
        js = (
            "if (window.__GeoPriorMap) {"
            f"  window.__GeoPriorMap.setLinks("
            f"'{lid}','{nm}',{payload},{o}"
            ");}"
        )
        self._run_js(js)

    def clear_links(self, layer_id: str) -> None:
        lid = _as_js_str(layer_id)
        js = (
            "if (window.__GeoPriorMap) {"
            f"  window.__GeoPriorMap.clearLinks('{lid}');"
            "}"
        )
        self._run_js(js)

    def set_radar(
        self,
        layer_id: str,
        centers: Iterable[list],
        opts: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = _json_min(list(centers))
        o = _json_min(opts or {})
        lid = _as_js_str(layer_id)
        js = (
            "if (window.__GeoPriorMap) {"
            f"  window.__GeoPriorMap.setRadar("
            f"'{lid}',{payload},{o}"
            ");}"
        )
        self._run_js(js)

    def clear_radar(self, layer_id: str) -> None:
        lid = _as_js_str(layer_id)
        js = (
            "if (window.__GeoPriorMap) {"
            f"  window.__GeoPriorMap.clearRadar('{lid}');"
            "}"
        )
        self._run_js(js)

    def fit_to_cities(self) -> None:
        # JS fitLayers already includes city markers.
        self.fit_layers([])

    def set_centroids(
        self,
        src_name: str,
        src_lat: float,
        src_lon: float,
        tgt_name: str,
        tgt_lat: float,
        tgt_lon: float,
    ) -> None:
        s = _as_js_str(src_name)
        t = _as_js_str(tgt_name)
        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.setCentroids("
            f"'{s}', {src_lat}, {src_lon}, "
            f"'{t}', {tgt_lat}, {tgt_lon}"
            ");"
            "}"
        )
        self._run_js(js)

    def set_layer(
        self,
        layer_id: str,
        name: str,
        points: Iterable[MapPoint] | None = None,
        opts: Optional[Dict[str, Any]] = None,
        # New argument to support Factory outputs (Scatter/Hex/Contour)
        payload: Optional[Any] = None,
    ) -> None:
        """
        Push a layer to the map.

        Supports both legacy 'points' list and new 'payload' object
        (from ViewFactory) containing kind, data, and opts.
        """
        # 1. Normalize Input -> p_dict structure
        p_dict: Dict[str, Any] = {}

        if payload is not None:
            # Use the factory payload directly
            p_dict = {
                "kind": getattr(payload, "kind", "points"),
                "data": getattr(payload, "data", []),
                "opts": getattr(payload, "opts", {}),
            }
        else:
            # Fallback: legacy points list -> kind="points" (scatter)
            # We compact MapPoint objects into arrays for JS speed
            data = []
            if points:
                for p in points:
                    data.append([p.lat, p.lon, p.v, p.sid, p.tip])

            p_dict = {
                "kind": "points",
                "data": data,
                "opts": opts or {},
            }

        # 2. Serialize the whole package
        json_str = _json_min(p_dict)
        lid = _as_js_str(layer_id)
        nm = _as_js_str(name)

        # 3. Call centralized JS API
        # window.__GeoPriorMap.setLayer handles routing to LayerFactory
        js = (
            "if (window.__GeoPriorMap) {"
            f"  window.__GeoPriorMap.setLayer('{lid}', '{nm}', {json_str});"
            "}"
        )
        self._run_js(js)

    def clear_layer(self, layer_id: str) -> None:
        lid = _as_js_str(layer_id)
        js = (
            "if (window.__GeoPriorMap) {"
            f"  window.__GeoPriorMap.clearLayer('{lid}');"
            "}"
        )
        self._run_js(js)

    def clear_layers(self) -> None:
        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.clearLayers();"
            "}"
        )
        self._run_js(js)

    def set_basemap(self, basemap: str) -> None:
        bid = _as_js_str(str(basemap or "osm"))
        js = (
            "if (window.__GeoPriorMap) {"
            f"  window.__GeoPriorMap.setBasemap('{bid}');"
            "}"
        )
        self._run_js(js)

    def fit_layers(
        self,
        layer_ids: Optional[Iterable[str]] = None,
    ) -> None:
        if layer_ids is None:
            payload = "null"
        else:
            ids = [str(x) for x in layer_ids]
            payload = _json_min(ids)

        js = (
            "if (window.__GeoPriorMap) {"
            f"  window.__GeoPriorMap.fitLayers({payload});"
            "}"
        )
        self._run_js(js)

    def set_legend(self, opts: Dict[str, Any]) -> None:
        o = _json_min(opts or {})
        js = (
            "if (window.__GeoPriorMap) {"
            f"  window.__GeoPriorMap.setLegend({o});"
            "}"
        )
        self._run_js(js)