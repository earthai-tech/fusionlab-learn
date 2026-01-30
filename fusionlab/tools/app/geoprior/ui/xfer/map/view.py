# geoprior/ui/xfer/map/view.py
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

from PyQt5.QtCore import QUrl, Qt
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

from .engines.leaflet_html import _LEAFLET_HTML

def _as_js_str(text: str) -> str:
    s = (text or "").replace("\\", "\\\\")
    s = s.replace("'", "\\'")
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
        # QtWebEngine is much more reliable when its chain
        # contains native widgets.
        # self.setAttribute(Qt.WA_NativeWindow, True)

        self._ready = False
        self._pending_js: list[str] = []

        if QWebEngineView is None:
            self._web = None
            self._web.setAttribute(Qt.WA_NativeWindow, True)
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

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if self._web is not None:
            root.addWidget(self._web, 1)
        else:
            root.addWidget(self._fallback, 1)

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
        points: Iterable[MapPoint],
        opts: Optional[Dict[str, Any]] = None,
    ) -> None:
        pts = []
        for p in points:
            # compact representation for JS
            pts.append([p.lat, p.lon, p.v, p.sid, p.tip])
        payload = _json_min(pts)
        o = _json_min(opts or {})
        lid = _as_js_str(layer_id)
        nm = _as_js_str(name)

        js = (
            "if (window.__GeoPriorMap) {"
            f"  window.__GeoPriorMap.setLayer("
            f"'{lid}', '{nm}', {payload}, {o}"
            ");"
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

    def fit_layers(
        self,
        layer_ids: Optional[Iterable[str]] = None,
    ) -> None:
        ids = None
        if layer_ids is not None:
            ids = [str(x) for x in layer_ids]
        payload = _json_min(ids) if ids else "null"

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
