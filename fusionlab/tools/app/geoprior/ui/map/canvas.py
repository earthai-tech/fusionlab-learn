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

import json
from typing import Any, Dict, List, Optional, Sequence

from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt5.QtWebEngineWidgets import (
        QWebEngineSettings,
        QWebEngineView,
    )
except Exception:  # pragma: no cover
    QWebEngineView = None  # type: ignore

try:
    from PyQt5.QtWebChannel import QWebChannel
except Exception:  # pragma: no cover
    QWebChannel = None  # type: ignore


def _leaflet_html() -> str:
    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8"/>',
        '<meta name="viewport"',
        'content="width=device-width, initial-scale=1.0"/>',
        '<link rel="stylesheet"',
        'href="https://unpkg.com/leaflet@1.9.4/'
        'dist/leaflet.css"/>',
        "<script",
        'src="https://unpkg.com/leaflet@1.9.4/'
        'dist/leaflet.js"></script>',
        '<script src="qrc:///qtwebchannel/'
        'qwebchannel.js"></script>',
        "<style>",
        "  html, body { height: 100%; margin: 0; }",
        "  #map { height: 100%; width: 100%; }",
        "</style>",
        "</head>",
        "<body>",
        '<div id="map"></div>',
        "<script>",
        "(function () {",
        "  // Define API first (safe if Leaflet fails)",
        "  window.__GeoPriorMap = {",
        "    setPoints: function(){},",
        "    clearPoints: function(){},",
        "    fitPoints: function(){},",
        "    zoomIn: function(){},",
        "    zoomOut: function(){},",
        "    setLegend: function(){},",
        "    setBasemap: function(){},",
        "    setHotspots: function(){},",
        "    clearHotspots: function(){},",
        "    showHotspots: function(){}",
        "  };",
        "",
        "  if (typeof L === 'undefined') {",
        "    console.log('Leaflet not available.');",
        "    return;",
        "  }",
        "",
        "  const map = L.map('map', {",
        "    zoomControl: true",
        "  }).setView([0, 0], 2);",
        "",
        "  const osmUrl = (",
        "    'https://{s}.tile.openstreetmap.org/' +",
        "    '{z}/{x}/{y}.png'",
        "  );",
        "  const osmAtt = '© OpenStreetMap';",
        "",
        "  // Track tile layer so we can swap basemaps",
        "  let tileLayer = L.tileLayer(osmUrl, {",
        "    maxZoom: 19,",
        "    attribution: osmAtt",
        "  }).addTo(map);",
        "",
        "  const layer = L.layerGroup().addTo(map);",
        "  let legendCtl = null;",
        "",
        "  // Qt WebChannel bridge (JS -> Python)",
        "  let bridge = null;",
        "",
        "  // Hotspots layer (separate from points)",
        "  const hotLayer = L.layerGroup().addTo(map);",
        "  let hotOn = true;",
        "  let hotTimers = [];",
        "",
        "  function _stopHotTimers() {",
        "    for (let i = 0; i < hotTimers.length; i++) {",
        "      try { clearInterval(hotTimers[i]); }",
        "      catch (e) {}",
        "    }",
        "    hotTimers = [];",
        "  }",
        "",
        "  function clearHotspots() {",
        "    _stopHotTimers();",
        "    hotLayer.clearLayers();",
        "  }",
        "",
        "  function _sevColor(sev) {",
        "    const s = (sev || 'high').toLowerCase();",
        "    if (s === 'critical') return '#d7263d';",
        "    if (s === 'high') return '#f18f01';",
        "    if (s === 'medium') return '#3f88c5';",
        "    return '#6c757d';",
        "  }",
        "",
        "  function _sevRadius(sev) {",
        "    const s = (sev || 'high').toLowerCase();",
        "    if (s === 'critical') return 18;",
        "    if (s === 'high') return 14;",
        "    if (s === 'medium') return 12;",
        "    return 10;",
        "  }",
        "",
        "  function _pulseRing(ring, r0m, speed) {",
        "    let r = r0m;",
        "    let a = 0.9;",
        "",
        "    let sp = Number(speed);",
        "    if (!isFinite(sp) || sp <= 0) sp = 1.0;",
        "",
        "    // Faster speed -> smaller dt",
        "    let dt = 80 / sp;",
        "    dt = Math.max(25, Math.min(200, dt));",
        "",
        "    const t = setInterval(function () {",
        "      r += (r0m * 0.06);",
        "      a -= 0.04;",
        "",
        "      if (a <= 0.05) {",
        "        r = r0m;",
        "        a = 0.9;",
        "      }",
        "",
        "      try {",
        "        ring.setRadius(r);",
        "        ring.setStyle({ opacity: a });",
        "      } catch (e) {}",
        "    }, dt);",
        "",
        "    hotTimers.push(t);",
        "  }",
        "",
        "  function showHotspots(on) {",
        "    hotOn = !!on;",
        "    if (hotOn) map.addLayer(hotLayer);",
        "    else map.removeLayer(hotLayer);",
        "  }",
        "",
        "  function _emitPointClicked(x, y) {",
        "    if (bridge && bridge.pointClicked) {",
        "      bridge.pointClicked(x, y);",
        "    }",
        "  }",
        "",
        "  if (typeof QWebChannel !== 'undefined' &&",
        "      typeof qt !== 'undefined' &&",
        "      qt.webChannelTransport) {",
        "    new QWebChannel(",
        "      qt.webChannelTransport,",
        "      function (channel) {",
        "        bridge = channel.objects.bridge || null;",
        "      }",
        "    );",
        "  }",
        "",
        "  // Optional: clicking map background also selects",
        "  map.on('click', function (e) {",
        "    if (!e || !e.latlng) return;",
        "    _emitPointClicked(e.latlng.lng, e.latlng.lat);",
        "  });",

        "  function clearPoints() {",
        "    layer.clearLayers();",
        "  }",
        "",
        "  function _clamp(x, a, b) {",
        "    return Math.max(a, Math.min(b, x));",
        "  }",
        "",
        "  function _color(t, cmap, inv) {",
        "    let tt = t;",
        "    if (inv) tt = 1 - tt;",
        "    // v0: keep HSL fallback (cmap reserved)",
        "    const h = 240 * (1 - tt);",
        "    return 'hsl(' + h + ',80%,45%)';",
        "  }",
        "",
        "  function setBasemap(provider, style, opacity) {",
        "    const p = (provider || 'osm').toLowerCase();",
        "    const s = (style || 'light').toLowerCase();",
        "",
        "    let url = osmUrl;",
        "    let att = osmAtt;",
        "",
        "    // Keep simple for now; extend later.",
        "    // Example: if you want a dark style,",
        "    // swap to Carto dark tiles, etc.",
        "    if (p === 'osm' && s === 'dark') {",
        "      url = (",
        "        'https://{s}.basemaps.cartocdn.com/' +",
        "        'dark_all/{z}/{x}/{y}{r}.png'",
        "      );",
        "      att = '© OpenStreetMap © CARTO';",
        "    }",
        "",
        "    const op = (opacity != null) ? opacity : 1.0;",
        "",
        "    if (tileLayer) {",
        "      map.removeLayer(tileLayer);",
        "      tileLayer = null;",
        "    }",
        "",
        "    tileLayer = L.tileLayer(url, {",
        "      maxZoom: 19,",
        "      attribution: att,",
        "      opacity: op",
        "    }).addTo(map);",
        "  }",
        "",
        "  function setLegend(vmin, vmax, label, cmap, inv) {",
        "    if (legendCtl) {",
        "      map.removeControl(legendCtl);",
        "      legendCtl = null;",
        "    }",
        "",
        "    const cm = cmap || 'viridis';",
        "    const iv = !!inv;",
        "",
        "    legendCtl = L.control({ position: 'bottomright' });",
        "    legendCtl.onAdd = function() {",
        "      const div = L.DomUtil.create('div');",
        "      div.style.background = 'rgba(255,255,255,0.85)';",
        "      div.style.padding = '8px';",
        "      div.style.borderRadius = '10px';",
        "      div.style.fontFamily = 'sans-serif';",
        "      div.style.fontSize = '12px';",
        "",
        "      const title = (label || 'Z');",
        "      const g = (",
        "        'linear-gradient(to top,' +",
        "        _color(0, cm, iv) + ',' +",
        "        _color(1, cm, iv) + ')'",
        "      );",
        "",
        "      div.innerHTML = (",
        "        '<div style=\"font-weight:600;\">' +",
        "        title + '</div>' +",
        "        '<div style=\"display:flex;gap:8px;' +",
        "        'align-items:center;\">' +",
        "        '<div style=\"width:12px;height:90px;' +",
        "        'background:' + g + ';border-radius:6px;\">' +",
        "        '</div>' +",
        "        '<div>' +",
        "        '<div>' + Number(vmax).toFixed(3) + '</div>' +",
        "        '<div style=\"height:62px;\"></div>' +",
        "        '<div>' + Number(vmin).toFixed(3) + '</div>' +",
        "        '</div></div>'",
        "      );",
        "",
        "      return div;",
        "    };",
        "    legendCtl.addTo(map);",
        "  }",
        "",
        "  function setPoints(points, opts) {",
        "    clearPoints();",
        "    const p = points || [];",
        "    const o = opts || {};",
        "",
        "    const r = (o.radius != null) ? o.radius : 6;",
        "    const op = (o.opacity != null) ? o.opacity : 0.9;",
        "    const vmin = (o.vmin != null) ? o.vmin : null;",
        "    const vmax = (o.vmax != null) ? o.vmax : null;",
        "",
        "    const cmap = o.cmap || 'viridis';",
        "    const inv = !!o.invert;",
        "",
        "    let lo = Infinity;",
        "    let hi = -Infinity;",
        "",
        "    for (let i = 0; i < p.length; i++) {",
        "      const v = p[i].v;",
        "      if (v == null || !isFinite(v)) continue;",
        "      lo = Math.min(lo, v);",
        "      hi = Math.max(hi, v);",
        "    }",
        "",
        "    const a = (vmin != null) ? vmin : lo;",
        "    const b = (vmax != null) ? vmax : hi;",
        "",
        "    for (let i = 0; i < p.length; i++) {",
        "      const pt = p[i];",
        "      const lat = pt.lat;",
        "      const lon = pt.lon;",
        "      const v = pt.v;",
        "      if (!isFinite(lat) || !isFinite(lon)) continue;",
        "",
        "      let t = 0.5;",
        "      if (v != null && isFinite(v) && b > a) {",
        "        t = (v - a) / (b - a);",
        "      }",
        "      t = _clamp(t, 0, 1);",
        "      const col = _color(t, cmap, inv);",
        "",
        "      const m = L.circleMarker([lat, lon], {",
        "        radius: r,",
        "        color: col,",
        "        fillColor: col,",
        "        fillOpacity: op,",
        "        weight: 1",
        "      }).addTo(layer);",
        "",
        "      // Emit (x,y) as (lon,lat) to match X/Y columns",
        "      m.on('click', function () {",
        "        _emitPointClicked(lon, lat);",
        "      });",
        "    }",
        "",
        "    if (o.showLegend) {",
        "      setLegend(a, b, o.label || 'Z', cmap, inv);",
        "    }",
        "  }",
        "",
        "  function setHotspots(hs, opts) {",
        "    clearHotspots();",
        "",
        "    const h = hs || [];",
        "    const o = opts || {};",
        "",
        "    const want = (o.show != null) ? !!o.show : true;",
        "    showHotspots(want);",
        "",
        "    const style = String(o.style || 'pulse').toLowerCase();",
        "    const pulse = (o.pulse != null) ? !!o.pulse : true;",
        "    const labels = (o.labels != null) ? !!o.labels : true;",
        "",
        "    let baseKm = Number(o.ringKm);",
        "    if (!isFinite(baseKm) || baseKm <= 0) baseKm = 0.8;",
        "",
        "    let sp = Number(o.pulseSpeed);",
        "    if (!isFinite(sp) || sp <= 0) sp = 1.0;",
        "",
        "    function _sevMul(sev) {",
        "      const s = (sev || 'high').toLowerCase();",
        "      if (s === 'critical') return 1.6;",
        "      if (s === 'high') return 1.25;",
        "      if (s === 'medium') return 1.0;",
        "      return 0.8;",
        "    }",
        "",
        "    for (let i = 0; i < h.length; i++) {",
        "      const pt = h[i] || {};",
        "      const lat = pt.lat;",
        "      const lon = pt.lon;",
        "      if (!isFinite(lat) || !isFinite(lon)) continue;",
        "",
        "      const sev = pt.sev || 'high';",
        "      const col = _sevColor(sev);",
        "      const mul = _sevMul(sev);",
        "",
        "      const tip = pt.label || ('Hotspot #' + (i + 1));",
        "",
        "      const core = L.circleMarker([lat, lon], {",
        "        radius: 5,",
        "        color: col,",
        "        fillColor: col,",
        "        fillOpacity: 0.95,",
        "        weight: 1",
        "      }).addTo(hotLayer);",
        "",
        "      if (labels) core.bindTooltip(tip);",
        "      core.on('click', function () {",
        "        _emitPointClicked(lon, lat);",
        "      });",
        "",
        "      // Style layer",
        "      if (style === 'glow') {",
        "        const glow = L.circle([lat, lon], {",
        "          radius: (baseKm * 1000.0 * mul),",
        "          color: col,",
        "          opacity: 0.55,",
        "          fillColor: col,",
        "          fillOpacity: 0.12,",
        "          weight: 1",
        "        }).addTo(hotLayer);",
        "        if (labels) glow.bindTooltip(tip);",
        "        glow.on('click', function () {",
        "          _emitPointClicked(lon, lat);",
        "        });",
        "        continue;",
        "      }",
        "",
        "      // Pulse ring (meters-based)",
        "      const r0m = (baseKm * 1000.0 * mul);",
        "      const ring = L.circle([lat, lon], {",
        "        radius: r0m,",
        "        color: col,",
        "        opacity: 0.9,",
        "        fillOpacity: 0.0,",
        "        weight: 2",
        "      }).addTo(hotLayer);",
        "",
        "      if (labels) ring.bindTooltip(tip);",
        "      ring.on('click', function () {",
        "        _emitPointClicked(lon, lat);",
        "      });",
        "",
        "      if (pulse) _pulseRing(ring, r0m, sp);",
        "    }",
        "  }",
        "",
        "  function fitPoints() {",
        "    const pts = [];",
        "",
        "    layer.eachLayer(function (m) {",
        "      if (m.getLatLng) pts.push(m.getLatLng());",
        "    });",
        "",
        "    hotLayer.eachLayer(function (m) {",
        "      if (m.getLatLng) pts.push(m.getLatLng());",
        "      if (m.getBounds) {",
        "        try {",
        "          const b = m.getBounds();",
        "          pts.push(b.getNorthWest());",
        "          pts.push(b.getSouthEast());",
        "        } catch (e) {}",
        "      }",
        "    });",
        "",
        "    if (!pts.length) return;",
        "    map.fitBounds(L.latLngBounds(pts).pad(0.2));",
        "  }",
        "",
        "  function zoomIn() { map.zoomIn(); }",
        "  function zoomOut() { map.zoomOut(); }",
        "",
        "  // Attach real functions",
        "  window.__GeoPriorMap.setPoints = setPoints;",
        "  window.__GeoPriorMap.clearPoints = clearPoints;",
        "  window.__GeoPriorMap.fitPoints = fitPoints;",
        "  window.__GeoPriorMap.zoomIn = zoomIn;",
        "  window.__GeoPriorMap.zoomOut = zoomOut;",
        "  window.__GeoPriorMap.setLegend = setLegend;",
        "  window.__GeoPriorMap.setBasemap = setBasemap;",
        "  window.__GeoPriorMap.setHotspots = setHotspots;",
        "  window.__GeoPriorMap.clearHotspots = clearHotspots;",
        "  window.__GeoPriorMap.showHotspots = showHotspots;",
        "})();",
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(lines)

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
    """
    Map canvas widget with overlay controls.
    """

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
        self._pending_js: List[str] = []
        
        self._last_points: List[Dict[str, Any]] = []
        self._last_label: str = "Z"
        self._last_vmin: Optional[float] = None
        self._last_vmax: Optional[float] = None
        
        self._view: Dict[str, Any] = {}
        self._last_hotspots: List[Dict[str, Any]] = []
        self._hot_opts: Dict[str, Any] = {
            "show": True,
            "pulse": True,
        }

        self._web: Optional[QWebEngineView]
        self._placeholder: Optional[QLabel]

        self._build_ui()
        self._load_map()

    # -----------------------------
    # Public API
    # -----------------------------
    def set_focus_checked(self, checked: bool) -> None:
        self.btn_focus.blockSignals(True)
        self.btn_focus.setChecked(bool(checked))
        self.btn_focus.blockSignals(False)

    def clear_points(self) -> None:
        self._run_js(
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.clearPoints();"
            "}",
        )

    def fit_points(self) -> None:
        self._run_js(
            "if (window.__GeoPriorMap) {"
            "  window.__GeoPriorMap.fitPoints();"
            "}",
        )
    
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

        self._last_points = list(points or [])
        self._last_label = str(label or "Z")
        self._last_vmin = vmin
        self._last_vmax = vmax
        
        # Keep last view hints (optional)
        self._view["radius"] = int(radius)
        self._view["opacity"] = float(opacity)
        self._view["showLegend"] = bool(show_legend)


        pts = list(points or [])
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
            self._redraw_points(
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
        self.set_points(
            self._last_points,
            vmin=vmin,
            vmax=vmax,
            radius=radius,
            opacity=opacity,
            label=self._last_label,
            show_legend=show_legend,
            cmap=cmap,
            invert=invert,
        )
    
    def _set_basemap(
        self,
        provider: str,
        style: str,
        opacity: float,
    ) -> None:
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
            self._web = QWebEngineView(self._container)
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

    # -----------------------------
    # WebEngine lifecycle
    # -----------------------------
    def _load_map(self) -> None:
        if self._web is None:
            return

        st = self._web.settings()
        st.setAttribute(
            QWebEngineSettings.LocalContentCanAccessRemoteUrls,
            True,
        )

        html = _leaflet_html()
        self._web.setHtml(
            html,
            QUrl("https://geoprior.local/"),
        )
        
        # WebChannel: JS -> Qt (marker click / map click)
        self._bridge = None
        self._channel = None
        if QWebChannel is not None:
            self._bridge = _GeoPriorBridge(self)
            self._channel = QWebChannel(self._web.page())
            self._channel.registerObject("bridge", self._bridge)
            self._web.page().setWebChannel(self._channel)
            self._bridge.point_clicked.connect(self.point_clicked)

        self._web.loadFinished.connect(self._on_loaded)

    def _on_loaded(self, ok: bool) -> None:
        self._ready = bool(ok)
        if not self._ready:
            return
        if self._pending_js:
            q = list(self._pending_js)
            self._pending_js.clear()
            for js in q:
                self._web.page().runJavaScript(js)

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
