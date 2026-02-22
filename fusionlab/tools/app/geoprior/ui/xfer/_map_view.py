# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.map_view

Leaflet-based interactive map for the Transfer tab.

- Pan / zoom (mouse + touchpad).
- Two city markers (source / target).
- Fit-to-cities helper.

If PyQtWebEngine is missing, a lightweight fallback
placeholder is shown instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt5.QtWebEngineWidgets import (
        QWebEngineView,
        QWebEngineSettings
    )
except Exception:  # pragma: no cover
    QWebEngineView = None  # type: ignore


@dataclass
class CityPoint:
    name: str
    lat: float
    lon: float


def _fmt_ll(lat: float, lon: float) -> str:
    return f"{lat:.5f}, {lon:.5f}"


def _as_js_str(text: str) -> str:
    s = (text or "").replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    return s


_LEAFLET_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { height: 100%; margin: 0; }
  #map { height: 100%; width: 100%; }
</style>
</head>
<body>
<div id="map"></div>
<script>
(function () {
  // Always define API first (prevents "undefined" errors)
  window.__GeoPriorMap = {
    setCities: function(){},
    clearCities: function(){},
    fitCities: function(){}
  };

  // If Leaflet failed (VPN/firewall), stop safely
  if (typeof L === 'undefined') {
    console.log("Leaflet not available (L undefined).");
    return;
  }

  const map = L.map('map', { zoomControl: true }).setView([0, 0], 2);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap'
  }).addTo(map);

  let srcMarker = null;
  let tgtMarker = null;
  let linkLine = null;

  function clearCities() {
    if (srcMarker) map.removeLayer(srcMarker);
    if (tgtMarker) map.removeLayer(tgtMarker);
    if (linkLine) map.removeLayer(linkLine);
    srcMarker = null;
    tgtMarker = null;
    linkLine = null;
  }

  function fitCities() {
    const pts = [];
    if (srcMarker) pts.push(srcMarker.getLatLng());
    if (tgtMarker) pts.push(tgtMarker.getLatLng());
    if (!pts.length) return;
    map.fitBounds(L.latLngBounds(pts).pad(0.3));
  }

  function setCities(srcName, srcLat, srcLon, tgtName, tgtLat, tgtLon) {
    clearCities();

    const src = [srcLat, srcLon];
    const tgt = [tgtLat, tgtLon];

    srcMarker = L.circleMarker(src, {
      radius: 7,
      color: '#2E3191',
      fillColor: '#2E3191',
      fillOpacity: 0.9
    }).addTo(map).bindPopup(srcName);

    tgtMarker = L.circleMarker(tgt, {
      radius: 7,
      color: '#F28620',
      fillColor: '#F28620',
      fillOpacity: 0.9
    }).addTo(map).bindPopup(tgtName);

    linkLine = L.polyline([src, tgt], {
      color: '#3399ff',
      weight: 2,
      opacity: 0.8
    }).addTo(map);

    fitCities();
  }

  // Attach real functions
  window.__GeoPriorMap.setCities = setCities;
  window.__GeoPriorMap.clearCities = clearCities;
  window.__GeoPriorMap.fitCities = fitCities;
})();
</script>
</body>
</html>
"""



class MapView(QWidget):
    """
    Interactive map widget.

    request_open_options
        Emitted when user clicks "Edit coords…".
    """

    request_open_options = pyqtSignal()
    request_fit = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._ready = False
        self._pending: Optional[
            Tuple[CityPoint, CityPoint]
        ] = None

        self._title = QLabel("Map")
        self._hint = QLabel(
            "Tip: set coordinates in Options "
            "to show cities."
        )
        self._hint.setWordWrap(True)

        self.btn_fit = QPushButton("Fit to cities")
        self.btn_fit.clicked.connect(self.request_fit.emit)

        self.btn_options = QPushButton(
            "Edit coordinates…"
        )
        self.btn_options.clicked.connect(
            self.request_open_options.emit
        )

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self.btn_fit)
        top.addWidget(self.btn_options)

        self._status = QLabel("")
        self._status.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        if QWebEngineView is None:
            self._web = None
            self._placeholder = QLabel(
                "Interactive map requires "
                "PyQtWebEngine.\n"
                "Install: pip install "
                "PyQtWebEngine"
            )
            self._placeholder.setAlignment(
                Qt.AlignCenter
            )
            self._placeholder.setFrameShape(
                QFrame.StyledPanel
            )
        else:
            self._placeholder = None
            self._web = QWebEngineView(self)
            # self._web.setHtml(_LEAFLET_HTML)
            st = self._web.settings()
            st.setAttribute(
                QWebEngineSettings.LocalContentCanAccessRemoteUrls,
                True,
            )
            self._web.setHtml(
                    _LEAFLET_HTML,
                    QUrl("https://geoprior.local/"),
                )
            self._web.loadFinished.connect(
                self._on_loaded
            )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        root.addLayout(top)
        root.addWidget(self._status)

        if self._web is not None:
            root.addWidget(self._web, 1)
        else:
            root.addWidget(self._placeholder, 1)

        root.addWidget(self._hint)

    def _on_loaded(self, ok: bool) -> None:
        self._ready = bool(ok)
        if not self._ready:
            self._status.setText(
                "Map failed to load "
                "(no internet?)."
            )
            return

        if self._pending is not None:
            a, b = self._pending
            self._pending = None
            self.set_cities(a, b)

    def clear(self) -> None:
        if self._web is None:
            self._status.setText("")
            return

        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.clearCities();"
            "}"
        )

        self._web.page().runJavaScript(js)
        self._status.setText("")
        self._pending = None

    def fit_to_cities(self) -> None:
        if self._web is None:
            return
        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.fitCities();"
            "}"
        )

        self._web.page().runJavaScript(js)

    def set_cities(
        self,
        src: CityPoint,
        tgt: CityPoint,
    ) -> None:
        if self._web is None:
            self._status.setText(
                f"{src.name}: "
                f"{_fmt_ll(src.lat, src.lon)}; "
                f"{tgt.name}: "
                f"{_fmt_ll(tgt.lat, tgt.lon)}"
            )
            return

        if not self._ready:
            self._pending = (src, tgt)
            return

        s_name = _as_js_str(src.name)
        t_name = _as_js_str(tgt.name)

        js = (
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.setCities("
            f"'{s_name}', {src.lat}, {src.lon}, "
            f"'{t_name}', {tgt.lat}, {tgt.lon}"
            ");"
            "} else {"
            "  console.log('GeoPriorMap not ready');"
            "}"
        )

        self._web.page().runJavaScript(js)

        self._status.setText(
            f"{src.name}: "
            f"{_fmt_ll(src.lat, src.lon)}; "
            f"{tgt.name}: "
            f"{_fmt_ll(tgt.lat, tgt.lon)}"
        )
