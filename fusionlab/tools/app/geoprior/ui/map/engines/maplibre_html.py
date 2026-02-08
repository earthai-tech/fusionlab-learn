# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
MapLibre HTML template for GeoPrior map canvas.

This template exposes:
  window.__GeoPriorMap.{setPoints,clearPoints,fitPoints,
  zoomIn,zoomOut,setLegend,setBasemap,setHotspots,
  clearHotspots,showHotspots}

It also wires Qt WebChannel:
  bridge.pointClicked(lon, lat)
"""

from __future__ import annotations

from typing import List


__all__ = ["maplibre_html"]


def maplibre_html() -> str:
    lines: List[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8"/>',
        '<meta name="viewport" '
        'content="width=device-width, initial-scale=1.0"/>',
        '<link rel="stylesheet" '
        'href="https://unpkg.com/maplibre-gl@3.6.2/'
        'dist/maplibre-gl.css"/>',
        "<script "
        'src="https://unpkg.com/maplibre-gl@3.6.2/'
        'dist/maplibre-gl.js"></script>',
        '<script src="qrc:///qtwebchannel/qwebchannel.js"></script>',
        "<style>",
        "  html, body { height: 100%; margin: 0; }",
        "  #map { height: 100%; width: 100%; position: relative; }",
        "  .gp-legend {",
        "    position: absolute;",
        "    right: 12px;",
        "    bottom: 12px;",
        "    background: rgba(255,255,255,0.85);",
        "    padding: 8px;",
        "    border-radius: 10px;",
        "    font-family: sans-serif;",
        "    font-size: 12px;",
        "    pointer-events: none;",
        "  }",
        "  .gp-sel-rect {",
        "    position: absolute;",
        "    border: 2px solid rgba(0,200,83,0.95);",
        "    background: rgba(0,200,83,0.12);",
        "    pointer-events: none;",
        "    z-index: 9999;",
        "  }",
        "  .gp-sel-hint {",
        "    border: 1px dashed rgba(0,200,83,0.55);",
        "    background: rgba(0,200,83,0.05);",
        "    opacity: 0.85;",
        "  }",
        "  .prop-arrow {",
        "    width: 26px;",
        "    height: 26px;",
        "    display: flex;",
        "    align-items: center;",
        "    justify-content: center;",
        "    font-size: 22px;",
        "    color: #d7263d;",
        "    text-shadow: 0 1px 2px rgba(0,0,0,0.35);",
        "    pointer-events: none;",
        "    transform-origin: 50% 50%;",
        "  }",
        "  /* Hotspot interpretation labels */",
        "  .gp-hot-label {",
        "    background: rgba(20,20,20,0.78);",
        "    color: #fff;",
        "    padding: 6px 8px;",
        "    border-radius: 10px;",
        "    font-family: sans-serif;",
        "    font-size: 12px;",
        "    max-width: 260px;",
        "    line-height: 1.15;",
        "    box-shadow: 0 6px 18px rgba(0,0,0,0.25);",
        "    pointer-events: none;",
        "    border: 1px solid rgba(255,255,255,0.12);",
        "  }",
        "  .gp-hot-label .t { font-weight: 700; margin-bottom: 2px; }",
        "  .gp-hot-label .b { opacity: 0.92; }",
        "  .gp-hot-label.critical { background: rgba(215,38,61,0.88); }",
        "  .gp-hot-label.high { background: rgba(241,143,1,0.88); }",
        "  .gp-hot-label.medium { background: rgba(63,136,197,0.88); }",
        "  .gp-hot-label.low { background: rgba(108,117,125,0.82); }",
        "  .gp-hot-popup .maplibregl-popup-content {",
        "    padding: 0;",
        "    background: transparent;",
        "    box-shadow: none;",
        "  }",
        "  .gp-hot-popup .maplibregl-popup-tip { display: none; }",
        "</style>",
        "</head>",
        "<body>",
        '<div id="map"></div>',
        "<script>",
        "(function () {",
        "  // Cache for reload safety (define BEFORE API)",
        "  let lastPoints = [];",
        "  let lastPointOpts = {};",
        "  let lastMainKind = 'points';",
        "  let lastHotspots = [];",
        "  let lastHotOpts = {};",
        "  let lastLegend = null;",
        "  let lastBasemap = { p: 'osm', s: 'light', o: 1.0 };",
        "  let lastSelectMode = 'off';",
        "  let lastVectors = [];",
        "  let lastVecOpts = {};",
        "",
        "  // Define API first (contract is ready immediately)",
        "  window.__GeoPriorMap = {",
        "    __engine: 'maplibre',",
        "    __ready: true,",
        "    __loaded: false,",
        "    __failed: false,",
        "    __err: '',",
        "    __debug: function () {",
        "      return {",
        "        engine: String(this.__engine || ''),",
        "        ready: !!this.__ready,",
        "        loaded: !!this.__loaded,",
        "        failed: !!this.__failed,",
        "        err: String(this.__err || ''),",
        "      };",
        "    },",
        "    setPoints: function (p, o) {",
        "      lastMainKind = 'points';",
        "      lastPoints = p || [];",
        "      lastPointOpts = o || {};",
        "    },",
        "    setHexbin: function (p, o) {",
        "      lastMainKind = 'hexbin_source';",
        "      lastPoints = p || [];",
        "      lastPointOpts = o || {};",
        "    },",
        "    setContours: function (p, o) {",
        "      lastMainKind = 'contour_source';",
        "      lastPoints = p || [];",
        "      lastPointOpts = o || {};",
        "    },",
        "    clearPoints: function () {",
        "      lastPoints = [];",
        "      lastPointOpts = {};",
        "      lastMainKind = 'points';",
        "    },",
        "    fitPoints: function () {},",
        "    zoomIn: function () {},",
        "    zoomOut: function () {},",
        "    setLegend: function (vmin, vmax, label, cmap, inv) {",
        "      lastLegend = {",
        "        vmin: Number(vmin),",
        "        vmax: Number(vmax),",
        "        label: String(label || 'Z'),",
        "        cmap: String(cmap || 'viridis'),",
        "        inv: !!inv,",
        "      };",
        "    },",
        "    setBasemap: function (provider, style, opacity) {",
        "      lastBasemap = {",
        "        p: String(provider || 'osm'),",
        "        s: String(style || 'light'),",
        "        o: (opacity != null) ? Number(opacity) : 1.0,",
        "      };",
        "    },",
        "    setHotspots: function (hs, o) {",
        "      lastHotspots = hs || [];",
        "      lastHotOpts = o || {};",
        "    },",
        "    clearHotspots: function () {",
        "      lastHotspots = [];",
        "      lastHotOpts = {};",
        "    },",
        "    showHotspots: function () {},",
        "    setVectors: function (v, o) {",
        "      lastVectors = v || [];",
        "      lastVecOpts = o || {};",
        "    },",
        "    setSelectMode: function (m) {",
        "      lastSelectMode = String(m || 'off');",
        "    },",
        "  };",
        "",
        "  if (typeof maplibregl === 'undefined') {",
        "    window.__GeoPriorMap.__failed = true;",
        "    window.__GeoPriorMap.__err = 'maplibregl undefined';",
        "    console.log('MapLibre not available.');",
        "    return;",
        "  }",
        "",
        "  // Qt WebChannel bridge (JS -> Python)",
        "  let bridge = null;",
        "  function _emitPointClicked(x, y) {",
        "    if (bridge && bridge.pointClicked) {",
        "      bridge.pointClicked(x, y);",
        "    }",
        "  }",
        "  function _emitPointClickedId(sid) {",
        "    if (bridge && bridge.pointClickedId) {",
        "      bridge.pointClickedId(sid);",
        "    }",
        "  }",
        "  function _emitGroupBBox(a, b, c, d) {",
        "    if (bridge && bridge.groupSelectedBBox) {",
        "      bridge.groupSelectedBBox(a, b, c, d);",
        "    }",
        "  }",
        "",
        "  let selectMode = 'off';",
        "  let _suppressClickUntil = 0;",
        "  function _setSelectMode(m) {",
        "    const s = String(m || 'off').toLowerCase();",
        "    if (s === 'point' || s === 'group') selectMode = s;",
        "    else selectMode = 'off';",
        "    try {",
        "      const host = (map && map.getContainer) ? map.getContainer() : null;",
        "      if (host) {",
        "        if (selectMode === 'group') host.style.cursor = 'crosshair';",
        "        else if (selectMode === 'point') host.style.cursor = 'pointer';",
        "        else host.style.cursor = '';",
        "      }",
        "    } catch (e) {}",
        "    try {",
        "      if (selectMode === 'group') {",
        "        const r = _host.getBoundingClientRect();",
        "        const p = _lastMouse || { x: r.width * 0.5, y: r.height * 0.5 };",
        "        _hintShowAt(p);",
        "      } else {",
        "        _hintHide();",
        "      }",
        "    } catch (e) {}",
        "  }",
        "",
        "  window.__GeoPriorMap.setSelectMode = function (m) {",
        "    _setSelectMode(m);",
        "  };",
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
        "  // Map + layers",
        "  let _gpLoaded = false;",
        "",        
        "  const GP_EMPTY_STYLE = {",
        "    version: 8,",
        "    sources: {},",
        "    layers: [",
        "      {",
        "        id: 'gp_bg',",
        "        type: 'background',",
        "        paint: { 'background-color': '#111' }",
        "      }",
        "    ]",
        "  };",
        "  let gpActive = { kind: 'raster', style: '', tiles: '' };",
        "",
        "  let map = null;",
        "  try {",
        "    map = new maplibregl.Map({",
        "      container: 'map',",
        "      style: GP_EMPTY_STYLE,",
        "      center: [0, 0],",
        "      zoom: 2,",
        "      attributionControl: true",
        "    });",
        "  } catch (e) {",
        "    window.__GeoPriorMap.__failed = true;",
        "    window.__GeoPriorMap.__err = String(e);",
        "    console.log('MapLibre init failed: ' + String(e));",
        "    return;",
        "  }",
        "",
        "  map.on('error', function (e) {",
        "    // Non-fatal errors (tiles/styles) can trigger here; don't kill engine",
        "    if (e && e.error) {",
        "      window.__GeoPriorMap.__err = String(e.error);",
        "    } else {",
        "      window.__GeoPriorMap.__err = String(e || '');",
        "    }",
        "  });",
        "",
        "  map.addControl(",
        "    new maplibregl.NavigationControl({ showCompass: false }),",
        "    'top-right'",
        "  );",
        "",
        "  // Basemap switching:",
        "  // - osm/light uses a vector style JSON (true MapLibre feel)",
        "  // - others use raster tiles via gp_basemap on an empty style",
        "",
        "  function _sameBasemap(a, b) {",
        "    if (!a || !b) return false;",
        "    if (a.kind !== b.kind) return false;",
        "    if (a.kind === 'style') {",
        "      return String(a.style || '') === String(b.style || '');",
        "    }",
        "    if (a.kind === 'raster') {",
        "      return String(a.tiles || '') === String(b.tiles || '');",
        "    }",
        "    return false;",
        "  }",
        "",
        "  function _removeRasterBasemap() {",
        "    if (map.getLayer('gp_basemap')) {",
        "      try { map.removeLayer('gp_basemap'); } catch (e) {}",
        "    }",
        "    if (map.getSource('gp_basemap')) {",
        "      try { map.removeSource('gp_basemap'); } catch (e) {}",
        "    }",
        "  }",
        "",        
        "  function _basemapBeforeId() {",
        "    const ids = [",
        "      'gp_points_layer',",
        "      'gp_hex_fill',",
        "      'gp_contour_layer',",
        "      'gp_hot_ring_layer',",
        "      'gp_hot_core_layer'",
        "    ];",
        "    for (let i = 0; i < ids.length; i++) {",
        "      try {",
        "        if (map.getLayer(ids[i])) return ids[i];",
        "      } catch (e) {}",
        "    }",
        "    return null;",
        "  }",
        "",
        "  function _applyRasterBasemap(spec, opacity) {",
        "    const op = (opacity != null) ? Number(opacity) : 1.0;",
        "    const o = (isFinite(op)) ? op : 1.0;",
        "",
        "    _removeRasterBasemap();",
        "",
        "    map.addSource('gp_basemap', {",
        "      type: 'raster',",
        "      tiles: spec.tiles,",
        "      tileSize: 256,",
        "      attribution: spec.att",
        "    });",
        "",
        "    const layer = {",
        "      id: 'gp_basemap',",
        "      type: 'raster',",
        "      source: 'gp_basemap',",
        "      paint: { 'raster-opacity': o }",
        "    };",
        "",
        "    const beforeId = _basemapBeforeId();",
        "    try {",
        "      if (beforeId) map.addLayer(layer, beforeId);",
        "      else map.addLayer(layer);",
        "    } catch (e) {",
        "      try { map.addLayer(layer); } catch (e2) {}",
        "    }",
        "  }",
        "",
        "  function _installOverlays() {",
        "    try { _ensurePointLayer(); } catch (e) {}",
        "    try { _ensureHotLayers(); } catch (e) {}",
        "    try { showHotspots(hotOn); } catch (e) {}",
        "",
        "    // replay cached state",
        "    try {",
        "      if (lastPoints && lastPoints.length) {",
        "        if (lastMainKind === 'hexbin_source') {",
        "          setHexbin(lastPoints, lastPointOpts || {});",
        "        } else if (lastMainKind === 'contour_source') {",
        "          setContours(lastPoints, lastPointOpts || {});",
        "        } else {",
        "          setPoints(lastPoints, lastPointOpts || {});",
        "        }",
        "      }",
        "    } catch (e) {}",
        "    try {",
        "      if (lastHotspots && lastHotspots.length) {",
        "        setHotspots(lastHotspots, lastHotOpts || {});",
        "      }",
        "    } catch (e) {}",
        "    try {",
        "      if (lastVectors && lastVectors.length) {",
        "        setVectors(lastVectors, lastVecOpts || {});",
        "      }",
        "    } catch (e) {}",
        "    try {",
        "      if (lastLegend) {",
        "        setLegend(",
        "          lastLegend.vmin,",
        "          lastLegend.vmax,",
        "          lastLegend.label,",
        "          lastLegend.cmap,",
        "          lastLegend.inv",
        "        );",
        "      }",
        "    } catch (e) {}",
        "  }",
        "",        "  function _basemapSpec(provider, style) {",
        "    const p = String(provider || 'osm').toLowerCase();",
        "    const s = String(style || 'light').toLowerCase();",
        "",
        "    // Raster defaults",
        "    let tiles = [",
        "      'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png',",
        "      'https://b.tile.openstreetmap.org/{z}/{x}/{y}.png',",
        "      'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png'",
        "    ];",
        "    let att = '© OpenStreetMap';",
        "",
        "    if (p === 'osm' && s === 'dark') {",
        "      tiles = [",
        "        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',",
        "        'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',",
        "        'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',",
        "        'https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'",
        "      ];",
        "      att = '© OpenStreetMap © CARTO';",
        "    } else if (p === 'osm' && s === 'gray') {",
        "      tiles = [",
        "        'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',",
        "        'https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',",
        "        'https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',",
        "        'https://d.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'",
        "      ];",
        "      att = '© OpenStreetMap © CARTO';",
        "    } else if (p === 'terrain') {",
        "      tiles = [",
        "        'https://a.tile.opentopomap.org/{z}/{x}/{y}.png',",
        "        'https://b.tile.opentopomap.org/{z}/{x}/{y}.png',",
        "        'https://c.tile.opentopomap.org/{z}/{x}/{y}.png'",
        "      ];",
        "      att = '© OpenStreetMap © OpenTopoMap';",
        "    } else if (p === 'satellite') {",
        "      tiles = [",
        "        'https://server.arcgisonline.com/' +",
        "        'ArcGIS/rest/services/World_Imagery/' +",
        "        'MapServer/tile/{z}/{y}/{x}'",
        "      ];",
        "      att = 'Tiles © Esri';",
        "    }",
        "",
        "    return { kind: 'raster', tiles: tiles, att: att };",
        "  }",
        "", 
        "  function setBasemap(provider, style, opacity) {",
        "    lastBasemap = {",
        "      p: String(provider || 'osm'),",
        "      s: String(style || 'light'),",
        "      o: (opacity != null) ? Number(opacity) : 1.0",
        "    };",
        "    if (!_gpLoaded) return;",
        "",
        "    const spec = _basemapSpec(provider, style);",
        "    _applyRasterBasemap(spec, opacity);",
        "  }",
        "  // Points: GPU-friendly circle layer via GeoJSON",
        "  function _ensurePointLayer() {",
        "    if (!map.getSource('gp_points')) {",
        "      map.addSource('gp_points', {",
        "        type: 'geojson',",
        "        data: { type: 'FeatureCollection', features: [] }",
        "      });",
        "    }",
        "    if (!map.getLayer('gp_points_layer')) {",
        "      map.addLayer({",
        "        id: 'gp_points_layer',",
        "        type: 'circle',",
        "        source: 'gp_points',",
        "        paint: {",
        "          'circle-radius': 6,",
        "          'circle-color': ['get', 'col'],",
        "          'circle-opacity': 0.9,",
        "          'circle-stroke-color': ['get', 'col'],",
        "          'circle-stroke-width': 1",
        "        }",
        "      });",
        "",
        "      map.on('click', 'gp_points_layer', function (e) {",
        "        try {",
        "          const f = e.features && e.features[0];",
        "          if (!f) return;",
        "          const g = f.geometry;",
        "          if (!g || g.type !== 'Point') return;",
        "          const c = g.coordinates || [];",
        "          const props = f.properties || {};",
        "          const sid = props.sid;",
        "          if (sid != null && isFinite(sid)) {",
        "            _emitPointClickedId(Number(sid));",
        "          } else {",
        "            _emitPointClicked(Number(c[0]), Number(c[1]));",
        "          }",
        "        } catch (err) {}",
        "      });",
        "    }",
        "  }",
        "",



                r"""
          // Hotspots layers (core + ring + hover interpretation)
          // Fix: circle-radius is PIXELS in MapLibre, so we convert km->px per zoom.
          let hotOn = true;
          let hotTimer = null;

          // Stateful ring objects for pulse & zoom scaling
          // { g, lat, km0, r0, r, a, col, is_new }
          let hotRings = [];

          // Hover interpretation popup
          let hotTextOn = true;
          let hotTextMinZoom = 6.0;
          let hotHoverInstalled = false;
          let hotPopup = null;

          function _stopHotTimer() {
            if (hotTimer != null) {
              try { clearInterval(hotTimer); } catch (e) {}
            }
            hotTimer = null;
          }

          function _esc(s) {
            const t = String(s == null ? "" : s);
            return t
              .replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#39;");
          }

          function _sevColor(sev) {
            const s = (sev || "high").toLowerCase();
            if (s === "critical") return "#d7263d";
            if (s === "high") return "#f18f01";
            if (s === "medium") return "#3f88c5";
            return "#6c757d";
          }

          function _sevClass(sev) {
            const s = (sev || "high").toLowerCase();
            if (s === "critical") return "critical";
            if (s === "high") return "high";
            if (s === "medium") return "medium";
            return "low";
          }

          function _kmToPx(km, lat) {
            const z = map.getZoom();
            const c = Math.cos((Number(lat) * Math.PI) / 180.0);
            const mpp = (156543.03392 * c) / Math.pow(2, z);
            const m = Number(km) * 1000.0;
            if (!isFinite(mpp) || mpp <= 0) return 24;
            const px = m / mpp;
            // clamp for usability (avoid gigantic rings)
            return Math.max(6, Math.min(220, px));
          }

          function _hotPopupRemove() {
            if (hotPopup) {
              try { hotPopup.remove(); } catch (e) {}
            }
            hotPopup = null;
          }

          function _hotCursorRestore() {
            try {
              const cv = map.getCanvas();
              if (!cv) return;
              if (selectMode === "group") cv.style.cursor = "crosshair";
              else if (selectMode === "point") cv.style.cursor = "pointer";
              else cv.style.cursor = "";
            } catch (e) {}
          }

          function _hotHtmlFromProps(props) {
            const sev = String((props && props.sev) || "high");
            const cls = _sevClass(sev);

            const title = String((props && props.ttl) || "").trim();
            const body = String((props && props.body) || "").trim();

            if (!title && !body) return "";

            let html = '<div class="gp-hot-label ' + cls + '">';
            if (title) {
              html += '<div class="t">' + _esc(title) + "</div>";
            }
            if (body) {
              html += (
                '<div class="b" style="white-space:pre-line;">' +
                _esc(body) +
                "</div>"
              );
            }
            html += "</div>";
            return html;
          }

          function _installHotHover() {
            if (hotHoverInstalled) return;
            hotHoverInstalled = true;

            map.on("mousemove", "gp_hot_core_layer", function (e) {
              try {
                if (!hotOn || !hotTextOn) {
                  _hotPopupRemove();
                  return;
                }
                if (map.getZoom() < hotTextMinZoom) {
                  _hotPopupRemove();
                  return;
                }

                const f = e.features && e.features[0];
                if (!f) {
                  _hotPopupRemove();
                  return;
                }

                const props = f.properties || {};
                const html = _hotHtmlFromProps(props);
                if (!html) {
                  _hotPopupRemove();
                  return;
                }

                // Cursor: do not override group mode crosshair
                try {
                  const cv = map.getCanvas();
                  if (cv && selectMode !== "group") {
                    cv.style.cursor = "pointer";
                  }
                } catch (e2) {}

                const g = f.geometry || null;
                const c = (g && g.coordinates) ? g.coordinates : null;
                const ll = c ? c : [e.lngLat.lng, e.lngLat.lat];

                if (!hotPopup) {
                  hotPopup = new maplibregl.Popup({
                    closeButton: false,
                    closeOnClick: false,
                    className: "gp-hot-popup",
                    offset: [0, -12]
                  });
                }

                hotPopup.setLngLat(ll).setHTML(html).addTo(map);
              } catch (err) {}
            });

            map.on("mouseleave", "gp_hot_core_layer", function () {
              _hotPopupRemove();
              _hotCursorRestore();
            });
          }

          function _ensureHotLayers() {
            if (!map.getSource("gp_hot_core")) {
              map.addSource("gp_hot_core", {
                type: "geojson",
                data: { type: "FeatureCollection", features: [] }
              });
            }
            if (!map.getSource("gp_hot_ring")) {
              map.addSource("gp_hot_ring", {
                type: "geojson",
                data: { type: "FeatureCollection", features: [] }
              });
            }

            if (!map.getLayer("gp_hot_ring_layer")) {
              map.addLayer({
                id: "gp_hot_ring_layer",
                type: "circle",
                source: "gp_hot_ring",
                paint: {
                  "circle-radius": ["get", "r"],            // <-- r is px now
                  "circle-color": ["get", "col"],
                  "circle-opacity": 0.0,
                  "circle-stroke-color": ["get", "col"],
                  "circle-stroke-opacity": ["get", "a"],
                  "circle-stroke-width": ["case",
                    ["==", ["get", "is_new"], 1], 3, 2
                  ]
                }
              });
            }

            if (!map.getLayer("gp_hot_core_layer")) {
              map.addLayer({
                id: "gp_hot_core_layer",
                type: "circle",
                source: "gp_hot_core",
                paint: {
                  "circle-radius": ["case",
                    ["==", ["get", "is_new"], 1], 7, 5
                  ],
                  "circle-color": ["get", "col"],
                  "circle-opacity": 0.95,
                  "circle-stroke-color": ["get", "col"],
                  "circle-stroke-width": 1
                }
              });

              map.on("click", "gp_hot_core_layer", function (e) {
                try {
                  const f = e.features && e.features[0];
                  if (!f) return;
                  const g = f.geometry;
                  if (!g || g.type !== "Point") return;
                  const c = g.coordinates || [];
                  const props = f.properties || {};
                  const sid = props.sid;
                  if (sid != null && isFinite(sid)) {
                    _emitPointClickedId(Number(sid));
                  } else {
                    _emitPointClicked(Number(c[0]), Number(c[1]));
                  }
                } catch (err) {}
              });

              // Hover interpretation (Leaflet-like tooltip)
              try { _installHotHover(); } catch (e) {}
            }
          }

          function showHotspots(on) {
            hotOn = !!on;
            const vis = hotOn ? "visible" : "none";
            try {
              if (map && map.getLayer && map.getLayer("gp_hot_core_layer")) {
                map.setLayoutProperty("gp_hot_core_layer", "visibility", vis);
              }
              if (map && map.getLayer && map.getLayer("gp_hot_ring_layer")) {
                map.setLayoutProperty("gp_hot_ring_layer", "visibility", vis);
              }
            } catch (e) {}

            // Keep pulse in sync with visibility
            if (!hotOn) {
              _stopHotTimer();
              _hotPopupRemove();
            } else {
              const o = lastHotOpts || {};
              const pulseOn = (o.pulse != null) ? !!o.pulse : false;
              const sp = (o.pulseSpeed != null) ? Number(o.pulseSpeed) : 1.0;
              if (pulseOn) _startPulse(sp);
              else _stopHotTimer();
            }
          }

          function _clearHotSources() {
            _stopHotTimer();
            _hotPopupRemove();
            hotRings = [];

            try {
              const s1 = map.getSource("gp_hot_core");
              if (s1) s1.setData({ type: "FeatureCollection", features: [] });
            } catch (e) {}
            try {
              const s2 = map.getSource("gp_hot_ring");
              if (s2) s2.setData({ type: "FeatureCollection", features: [] });
            } catch (e) {}
          }

          function clearHotspots() {
            lastHotspots = [];
            lastHotOpts = {};
            _clearHotSources();
          }

          function _pushHotRings() {
            const ringFeats = [];
            for (let i = 0; i < hotRings.length; i++) {
              const rr = hotRings[i];
              ringFeats.push({
                type: "Feature",
                geometry: rr.g,
                properties: {
                  col: rr.col,
                  r: rr.r,
                  a: rr.a,
                  is_new: rr.is_new ? 1 : 0
                }
              });
            }
            try {
              const s2 = map.getSource("gp_hot_ring");
              if (s2) s2.setData({ type: "FeatureCollection", features: ringFeats });
            } catch (e) {}
          }

          function _refreshHotR0FromZoom() {
            // Recompute base pixel radii so the ring represents constant km
            for (let i = 0; i < hotRings.length; i++) {
              const rr = hotRings[i];
              const basePx = _kmToPx(rr.km0, rr.lat);
              rr.r0 = basePx;
              rr.r = basePx;
            }
            _pushHotRings();
          }

          function _startPulse(speed) {
            _stopHotTimer();

            let sp = Number(speed);
            if (!isFinite(sp) || sp <= 0) sp = 1.0;

            let dt = 80 / sp;
            dt = Math.max(25, Math.min(200, dt));

            hotTimer = setInterval(function () {
              for (let i = 0; i < hotRings.length; i++) {
                const rr = hotRings[i];

                // keep km meaning even while pulsing
                const basePx = _kmToPx(rr.km0, rr.lat);
                rr.r0 = basePx;

                const isNew = (rr.is_new === 1 || rr.is_new === true);
                const k = isNew ? 0.10 : 0.06;
                const da = isNew ? 0.03 : 0.04;
                const a0 = isNew ? 0.95 : 0.90;

                rr.r += (rr.r0 * k);
                rr.a -= da;

                if (rr.a <= 0.05) {
                  rr.r = rr.r0;
                  rr.a = a0;
                }
              }
              _pushHotRings();
            }, dt);
          }

          function _normHotText(pt) {
            const rawTitle = String(pt.title || pt.label || "").trim();
            let rawBody = String(pt.text || pt.note || pt.msg || "").trim();
            const rawFull = String(pt.label_full || pt.full || pt.popup || "").trim();

            if (!rawBody && rawFull) rawBody = rawFull;

            let title = rawTitle;
            let body = rawBody;

            if (!title && body) {
              const lines = body
                .split(/\r?\n/)
                .map((s) => String(s || "").trim())
                .filter((s) => s.length > 0);
              if (lines.length) {
                title = lines[0];
                body = lines.slice(1).join("\n");
              }
            }

            if (title && body && title === body) body = "";
            return { t: title, b: body };
          }

          function setHotspots(hs, opts) {
            const h = hs || [];
            const o = opts || {};
            lastHotspots = h;
            lastHotOpts = o;

            _clearHotSources();
            if (!_gpLoaded) return;

            const want = (o.show != null) ? !!o.show : true;
            showHotspots(want);
            if (!want) return;

            const style = String(o.style || "pulse").toLowerCase();
            const pulse = (o.pulse != null) ? !!o.pulse : true;

            let baseKm = Number(o.ringKm);
            if (!isFinite(baseKm) || baseKm <= 0) baseKm = 0.8;

            let sp = Number(o.pulseSpeed);
            if (!isFinite(sp) || sp <= 0) sp = 1.0;

            // Interpretation (hover) settings
            hotTextOn = (o.showText != null) ? !!o.showText : true;
            hotTextMinZoom = (o.textMinZoom != null)
              ? Number(o.textMinZoom) : 6.0;

            function _sevMul(sev) {
              const s = (sev || "high").toLowerCase();
              if (s === "critical") return 1.6;
              if (s === "high") return 1.25;
              if (s === "medium") return 1.0;
              return 0.8;
            }

            const coreFeats = [];
            hotRings = [];

            // Build sources + ring states
            for (let i = 0; i < h.length; i++) {
              const pt = h[i] || {};
              const lat = Number(pt.lat);
              const lon = Number(pt.lon);
              if (!isFinite(lat) || !isFinite(lon)) continue;

              const sid =
                (pt.sid != null && isFinite(pt.sid))
                  ? Number(pt.sid)
                  : null;

              const sev = pt.sev || "high";
              const col = _sevColor(sev);
              const isNew = !!pt.is_new;
              const mul = _sevMul(sev) * (isNew ? 1.15 : 1.0);

              const tb = _normHotText(pt);

              // core feature (store text so hover can render it)
              coreFeats.push({
                type: "Feature",
                geometry: { type: "Point", coordinates: [lon, lat] },
                properties: {
                  col: col,
                  sid: sid,
                  is_new: isNew ? 1 : 0,
                  sev: String(sev || "high"),
                  ttl: String(tb.t || ""),
                  body: String(tb.b || "")
                }
              });

              // ring state (km -> px)
              const km0 = baseKm * mul;
              const r0px = _kmToPx(km0, lat);

              // glow style = static ring (no pulse), pulse style = animated ring
              const a0 = isNew ? 0.95 : 0.90;
              hotRings.push({
                g: { type: "Point", coordinates: [lon, lat] },
                lat: lat,
                km0: km0,
                r0: r0px,
                r: r0px,
                a: (style === "glow") ? (isNew ? 0.75 : 0.55) : a0,
                col: col,
                is_new: isNew ? 1 : 0
              });
            }

            // Commit core
            try {
              const s1 = map.getSource("gp_hot_core");
              if (s1) s1.setData({
                type: "FeatureCollection",
                features: coreFeats
              });
            } catch (e) {}

            // Commit rings (initial)
            _pushHotRings();

            // Pulse (optional)
            if (style !== "glow" && pulse) {
              _startPulse(sp);
            }
          }

          // Keep ring meaning stable on zoom (km -> px rescale)
          map.on("zoomend", function () {
            try {
              if (hotOn && hotRings && hotRings.length) {
                if (hotTimer == null) _refreshHotR0FromZoom();
                if (hotTimer != null) _refreshHotR0FromZoom();
              }
            } catch (e) {}

            // Hide tooltip at low zoom
            try {
              if (map.getZoom() < hotTextMinZoom) {
                _hotPopupRemove();
              }
            } catch (e) {}
          });
        """,

        
        
        
 
        "  function clearPoints() {",
        "    lastPoints = [];",
        "    lastPointOpts = {};",
        "    lastMainKind = 'points';",
        "",
        "    // points",
        "    try {",
        "      const s = map.getSource('gp_points');",
        "      if (s) s.setData({",
        "        type: 'FeatureCollection', features: []",
        "      });",
        "    } catch (e) {}",
        "",
        "    // hexbin",
        "    try { if (map.getLayer('gp_hex_fill'))",
        "      map.removeLayer('gp_hex_fill'); } catch (e) {}",
        "    try { if (map.getLayer('gp_hex_line'))",
        "      map.removeLayer('gp_hex_line'); } catch (e) {}",
        "    try { if (map.getSource('gp_hex'))",
        "      map.removeSource('gp_hex'); } catch (e) {}",
        "",
        "    // contours (image overlay)",
        "    try { if (map.getLayer('gp_contour_layer'))",
        "      map.removeLayer('gp_contour_layer'); } catch (e) {}",
        "    try { if (map.getSource('gp_contour_img'))",
        "      map.removeSource('gp_contour_img'); } catch (e) {}",
        "  }",
        "",
        "  function _clamp(x, a, b) {",
        "    return Math.max(a, Math.min(b, x));",
        "  }",
        "",
                r"""
          // -------------------------------------------------
          // Visualization strategies: hexbin + contours
          // -------------------------------------------------
        
          function _agg(vals, metric) {
            const m = String(metric || "mean").toLowerCase();
            if (!vals || !vals.length) return null;
            if (m === "count") return vals.length;
            if (m === "max") return Math.max.apply(null, vals);
            if (m === "min") return Math.min.apply(null, vals);
            if (m === "sum") {
              let s = 0;
              for (let i = 0; i < vals.length; i++) s += vals[i];
              return s;
            }
            let s = 0;
            for (let i = 0; i < vals.length; i++) s += vals[i];
            return s / vals.length;
          }
        
          function _ensureHexLayer() {
            if (!map.getSource("gp_hex")) {
              map.addSource("gp_hex", {
                type: "geojson",
                data: { type: "FeatureCollection", features: [] }
              });
            }
            if (!map.getLayer("gp_hex_fill")) {
              map.addLayer({
                id: "gp_hex_fill",
                type: "fill",
                source: "gp_hex",
                paint: {
                  "fill-color": ["get", "col"],
                  "fill-opacity": ["get", "a"]
                }
              });
            }
            if (!map.getLayer("gp_hex_line")) {
              map.addLayer({
                id: "gp_hex_line",
                type: "line",
                source: "gp_hex",
                paint: {
                  "line-color": ["get", "col"],
                  "line-opacity": ["get", "a"],
                  "line-width": 1
                }
              });
            }
          }
        
          function _renderHexbin() {
            const pts = lastPoints || [];
            const o = lastPointOpts || {};
        
            if (!_gpLoaded) return;
        
            // clear other overlays but keep caches
            try { if (map.getLayer("gp_contour_layer"))
              map.removeLayer("gp_contour_layer"); } catch (e) {}
            try { if (map.getSource("gp_contour_img"))
              map.removeSource("gp_contour_img"); } catch (e) {}
        
            _ensureHexLayer();
        
            const gs = Math.max(5, Number(o.gridsize || 30));
            const metric = String(o.metric || "mean");
            const op = (o.opacity != null) ? Number(o.opacity) : 0.85;
        
            if (!pts.length) {
              try {
                const s = map.getSource("gp_hex");
                if (s) s.setData({ type: "FeatureCollection", features: [] });
              } catch (e) {}
              return;
            }
        
            const bins = {};
            const sqrt3 = Math.sqrt(3);
        
            for (let i = 0; i < pts.length; i++) {
              const p = pts[i] || {};
              const lat = Number(p.lat);
              const lon = Number(p.lon);
              const v = Number(p.v);
              if (!isFinite(lat) || !isFinite(lon) || !isFinite(v)) continue;
        
              const q = map.project([lon, lat]); // pixel coords
              const x = q.x;
              const y = q.y;
        
              const col = Math.round(x / (1.5 * gs));
              const row = Math.round(
                (y - (col & 1) * (sqrt3 * 0.5 * gs)) / (sqrt3 * gs)
              );
              const key = col + "," + row;
        
              if (!bins[key]) bins[key] = { col: col, row: row, vs: [] };
              bins[key].vs.push(v);
            }
        
            let lo = Infinity;
            let hi = -Infinity;
        
            for (const k in bins) {
              if (!bins.hasOwnProperty(k)) continue;
              const it = bins[k];
              const vv = _agg(it.vs, metric);
              if (vv == null || !isFinite(vv)) continue;
              it.v = vv;
              lo = Math.min(lo, vv);
              hi = Math.max(hi, vv);
            }
        
            const a = (o.vmin != null) ? Number(o.vmin) : lo;
            const b = (o.vmax != null) ? Number(o.vmax) : hi;
        
            const feats = [];
        
            for (const k in bins) {
              if (!bins.hasOwnProperty(k)) continue;
              const it = bins[k];
              if (it.v == null || !isFinite(it.v)) continue;
        
              const cx = it.col * 1.5 * gs;
              const cy = it.row * (sqrt3 * gs) +
                (it.col & 1) * (sqrt3 * 0.5 * gs);
        
              const t = _clamp((it.v - a) / ((b - a) || 1), 0, 1);
              const col = _color(t, o.cmap, !!o.invert);
        
              const ring = [];
              for (let j = 0; j < 6; j++) {
                const ang = (Math.PI / 3) * j;
                const px = cx + gs * Math.cos(ang);
                const py = cy + gs * Math.sin(ang);
                const ll = map.unproject([px, py]);
                ring.push([ll.lng, ll.lat]);
              }
              ring.push(ring[0]); // close
        
              feats.push({
                type: "Feature",
                geometry: { type: "Polygon", coordinates: [ring] },
                properties: { col: col, a: op }
              });
            }
        
            try {
              const s = map.getSource("gp_hex");
              if (s) s.setData({ type: "FeatureCollection", features: feats });
            } catch (e) {}
        
            if (o.showLegend) {
              setLegend(a, b, o.label || "Z", o.cmap, !!o.invert);
              lastLegend = {
                vmin: a, vmax: b, label: o.label || "Z",
                cmap: (o.cmap || "viridis"), inv: !!o.invert
              };
            }
          }
        
          function setHexbin(points, opts) {
            lastMainKind = "hexbin_source";
            lastPoints = points || [];
            lastPointOpts = opts || {};
            if (!_gpLoaded) return;
            _renderHexbin();
          }
        
          function _renderContours() {
            const pts = lastPoints || [];
            const o = lastPointOpts || {};
        
            if (!_gpLoaded) return;
        
            // clear hexbin
            try { if (map.getLayer("gp_hex_fill"))
              map.removeLayer("gp_hex_fill"); } catch (e) {}
            try { if (map.getLayer("gp_hex_line"))
              map.removeLayer("gp_hex_line"); } catch (e) {}
            try { if (map.getSource("gp_hex"))
              map.removeSource("gp_hex"); } catch (e) {}
        
            const op = (o.opacity != null) ? Number(o.opacity) : 0.7;
            const bw = Math.max(4, Number(o.bandwidth || 15));
            const step = Math.max(6, Math.min(80, bw));
        
            const canvas = document.createElement("canvas");
            const w = map.getCanvas().width;
            const h = map.getCanvas().height;
            if (!w || !h || !pts.length) return;
        
            canvas.width = w;
            canvas.height = h;
        
            const ctx = canvas.getContext("2d");
            if (!ctx) return;
        
            const proj = [];
            let lo = Infinity;
            let hi = -Infinity;
        
            for (let i = 0; i < pts.length; i++) {
              const p = pts[i] || {};
              const lat = Number(p.lat);
              const lon = Number(p.lon);
              const v = Number(p.v);
              if (!isFinite(lat) || !isFinite(lon) || !isFinite(v)) continue;
              const q = map.project([lon, lat]);
              proj.push({ x: q.x, y: q.y, v: v });
              lo = Math.min(lo, v);
              hi = Math.max(hi, v);
            }
        
            const a = (o.vmin != null) ? Number(o.vmin) : lo;
            const b = (o.vmax != null) ? Number(o.vmax) : hi;
        
            const sig2 = 2 * step * step;
            ctx.globalAlpha = 1.0;
        
            for (let y = 0; y < h; y += step) {
              for (let x = 0; x < w; x += step) {
                let num = 0;
                let den = 0;
        
                for (let i = 0; i < proj.length; i++) {
                  const p = proj[i];
                  const dx = x - p.x;
                  const dy = y - p.y;
                  const d2 = dx * dx + dy * dy;
                  const w0 = Math.exp(-d2 / sig2);
                  num += w0 * p.v;
                  den += w0;
                }
        
                if (den <= 0) continue;
        
                const v = num / den;
                const t = _clamp((v - a) / ((b - a) || 1), 0, 1);
        
                ctx.fillStyle = _color(t, o.cmap, !!o.invert);
                ctx.fillRect(x, y, step, step);
              }
            }
        
            const url = canvas.toDataURL("image/png");
        
            // MapLibre image source expects corners (NW, NE, SE, SW)
            const bb = map.getBounds();
            const nw = bb.getNorthWest();
            const ne = bb.getNorthEast();
            const se = bb.getSouthEast();
            const sw = bb.getSouthWest();
        
            try { if (map.getLayer("gp_contour_layer"))
              map.removeLayer("gp_contour_layer"); } catch (e) {}
            try { if (map.getSource("gp_contour_img"))
              map.removeSource("gp_contour_img"); } catch (e) {}
        
            map.addSource("gp_contour_img", {
              type: "image",
              url: url,
              coordinates: [
                [nw.lng, nw.lat],
                [ne.lng, ne.lat],
                [se.lng, se.lat],
                [sw.lng, sw.lat]
              ]
            });
        
            map.addLayer({
              id: "gp_contour_layer",
              type: "raster",
              source: "gp_contour_img",
              paint: { "raster-opacity": op }
            });
        
            if (o.showLegend) {
              setLegend(a, b, o.label || "Z", o.cmap, !!o.invert);
              lastLegend = {
                vmin: a, vmax: b, label: o.label || "Z",
                cmap: (o.cmap || "viridis"), inv: !!o.invert
              };
            }
          }
        
          function setContours(points, opts) {
            lastMainKind = "contour_source";
            lastPoints = points || [];
            lastPointOpts = opts || {};
            if (!_gpLoaded) return;
            _renderContours();
          }
        
          map.on("moveend", function () {
            if (lastMainKind === "hexbin_source") _renderHexbin();
            if (lastMainKind === "contour_source") _renderContours();
          });
        
          map.on("zoomend", function () {
            if (lastMainKind === "hexbin_source") _renderHexbin();
            if (lastMainKind === "contour_source") _renderContours();
          });
        """,
        "  function _clamp01(x) {",
        "    return Math.max(0, Math.min(1, x));",
        "  }",
        "  function _lerp(a, b, t) { return a + (b - a) * t; }",
        "",
        "  function _parseCmap(cmap, inv) {",
        "    const s = String(cmap || '').toLowerCase();",
        "    if (s.endsWith('_r')) {",
        "      return { cm: s.slice(0, -2) || 'viridis', inv: !inv };",
        "    }",
        "    return { cm: (s || 'viridis'), inv: !!inv };",
        "  }",
        "",
        "  function _getPalette(cm) {",
        "    const k = String(cm || 'viridis').toLowerCase();",
        "    if (k === 'viridis') return [",
        "      [68,1,84],[72,40,120],[62,74,137],[49,104,142],",
        "      [38,130,142],[31,158,137],[53,183,121],",
        "      [110,206,88],[181,222,43],[253,231,37]",
        "    ];",
        "    if (k === 'plasma') return [",
        "      [13,8,135],[84,2,163],[139,10,165],[185,50,137],",
        "      [219,92,104],[244,136,73],[254,188,43],[239,248,33]",
        "    ];",
        "    if (k === 'inferno') return [",
        "      [0,0,4],[31,12,72],[85,15,109],[136,34,106],",
        "      [186,54,85],[227,89,51],[249,140,10],",
        "      [252,195,32],[252,255,164]",
        "    ];",
        "    if (k === 'magma') return [",
        "      [0,0,4],[28,16,68],[79,18,123],[129,37,129],",
        "      [181,54,122],[229,80,100],[251,135,97],",
        "      [254,194,135],[252,253,191]",
        "    ];",
        "    if (k === 'cividis') return [",
        "      [0,32,76],[0,68,115],[40,102,116],[90,129,99],",
        "      [144,155,71],[199,183,39],[253,215,6]",
        "    ];",
        "    if (k === 'turbo') return [",
        "      [48,18,59],[60,74,160],[34,144,201],[45,202,104],",
        "      [189,223,38],[250,186,22],[245,95,37],[180,4,38]",
        "    ];",
        "    if (k === 'jet') return [",
        "      [0,0,128],[0,0,255],[0,255,255],[255,255,0],",
        "      [255,0,0],[128,0,0]",
        "    ];",
        "    if (k === 'gray' || k === 'greys') return [[0,0,0],[255,255,255]];",
        "    return null;",
        "  }",
        "",
        "  function _color(t, cmap, inv, pal) {",
        "    let tt = _clamp01(t);",
        "    const pr = _parseCmap(cmap, !!inv);",
        "    const cm = pr.cm;",
        "    const iv = pr.inv;",
        "    if (iv) tt = 1 - tt;",
        "",
        "    const p = (Array.isArray(pal) && pal.length > 1)",
        "      ? pal : _getPalette(cm);",
        "",
        "    if (Array.isArray(p) && p.length > 1) {",
        "      const n = p.length;",
        "      const x = tt * (n - 1);",
        "      const i = Math.floor(x);",
        "      const f = x - i;",
        "      const a = p[i];",
        "      const b = p[Math.min(i + 1, n - 1)];",
        "      const r = Math.round(_lerp(a[0], b[0], f));",
        "      const g = Math.round(_lerp(a[1], b[1], f));",
        "      const bb = Math.round(_lerp(a[2], b[2], f));",
        "      return 'rgb(' + r + ',' + g + ',' + bb + ')';",
        "    }",
        "",
        "    const h = (1 - tt) * 240;",
        "    return 'hsl(' + h + ', 90%, 50%)';",
        "  }",
        "",
        "  function _applyLegendPos(el, pos) {",
        "    const k = String(pos || 'br').toLowerCase();",
        "    el.style.left = ''; el.style.right = '';",
        "    el.style.top = ''; el.style.bottom = '';",
        "    if (k === 'tr') { el.style.right = '12px'; el.style.top = '12px'; }",
        "    else if (k === 'tl') { el.style.left = '12px'; el.style.top = '12px'; }",
        "    else if (k === 'bl') { el.style.left = '12px'; el.style.bottom = '12px'; }",
        "    else { el.style.right = '12px'; el.style.bottom = '12px'; }",
        "  }",
        "",
        "  function _gradCss(cmap, inv, pal, ori) {",
        "    const pr = _parseCmap(cmap, !!inv);",
        "    const cm = pr.cm;",
        "    const iv = pr.inv;",
        "    const dir = (String(ori || 'vertical').toLowerCase()",
        "      === 'horizontal') ? 'to right' : 'to top';",
        "",
        "    const p = (Array.isArray(pal) && pal.length > 1)",
        "      ? pal : _getPalette(cm);",
        "",
        "    if (Array.isArray(p) && p.length > 1) {",
        "      const n = p.length;",
        "      const step = Math.max(1, Math.floor(n / 10));",
        "      const parts = [];",
        "      for (let i = 0; i < n; i += step) {",
        "        const c = p[i];",
        "        const pp = Math.round((i / (n - 1)) * 100);",
        "        parts.push(",
        "          'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ') ' + pp + '%'",
        "        );",
        "      }",
        "      return 'linear-gradient(' + dir + ',' + parts.join(',') + ')';",
        "    }",
        "",
        "    return 'linear-gradient(' + dir + ',' +",
        "      _color(0, cm, iv, null) + ',' +",
        "      _color(1, cm, iv, null) + ')';",
        "  }",
        "",
        "  let legendEl = null;",
        "  function setLegend(vmin, vmax, label, cmap, inv, pal, pos, ori) {",
        "    if (legendEl && legendEl.parentNode) {",
        "      try { legendEl.parentNode.removeChild(legendEl); }",
        "      catch (e) {}",
        "    }",
        "",
        "    const cm = (cmap || 'viridis');",
        "    const iv = !!inv;",
        "    const title = (label || 'Z');",
        "    const pp = String(pos || 'br').toLowerCase();",
        "    const oo = (String(ori || 'vertical').toLowerCase() === 'horizontal')",
        "      ? 'horizontal' : 'vertical';",
        "",
        "    const g = _gradCss(cm, iv, pal, oo);",
        "",
        "    const div = document.createElement('div');",
        "    div.className = 'gp-legend';",
        "    _applyLegendPos(div, pp);",
        "",
        "    if (oo === 'horizontal') {",
        "      div.innerHTML = (",
        "        '<div style=\"font-weight:600;\">' + title + '</div>' +",
        "        '<div style=\"width:140px;\">' +",
        "        '<div style=\"width:140px;height:12px;background:' + g +",
        "        ';border-radius:6px;\"></div>' +",
        "        '<div style=\"display:flex;justify-content:space-between;\">' +",
        "        '<span>' + Number(vmin).toFixed(3) + '</span>' +",
        "        '<span>' + Number(vmax).toFixed(3) + '</span>' +",
        "        '</div></div>'",
        "      );",
        "    } else {",
        "      div.innerHTML = (",
        "        '<div style=\"font-weight:600;\">' + title + '</div>' +",
        "        '<div style=\"display:flex;gap:8px;align-items:center;\">' +",
        "        '<div style=\"width:12px;height:90px;background:' + g +",
        "        ';border-radius:6px;\"></div>' +",
        "        '<div>' +",
        "        '<div>' + Number(vmax).toFixed(3) + '</div>' +",
        "        '<div style=\"height:62px;\"></div>' +",
        "        '<div>' + Number(vmin).toFixed(3) + '</div>' +",
        "        '</div></div>'",
        "      );",
        "    }",
        "",
        "    document.body.appendChild(div);",
        "    legendEl = div;",
        "  }",
        "",
        "  function setPoints(points, opts) {",
        "    const p = points || [];",
        "    const o = opts || {};",
        "",
        "    lastPoints = p;",
        "    lastPointOpts = o;",
        "    lastMainKind = 'points';",
        "    if (!_gpLoaded) return;",
        "",
        "    _ensurePointLayer();",
        "",
        "    const r = (o.radius != null) ? Number(o.radius) : 6;",
        "    const op = (o.opacity != null) ? Number(o.opacity) : 0.9;",
        "    const vmin = (o.vmin != null) ? Number(o.vmin) : null;",
        "    const vmax = (o.vmax != null) ? Number(o.vmax) : null;",
        "    const cmap = o.cmap || 'viridis';",
        "    const inv = !!o.invert;",
        "    const pal = o.palette || null;",
        "    const pos = o.legendPos || 'br';",
        "    const ori = o.legendOrient || 'vertical';",
        "",
        "    let lo = Infinity;",
        "    let hi = -Infinity;",
        "    for (let i = 0; i < p.length; i++) {",
        "      const vv = p[i].v;",
        "      if (vv == null || !isFinite(vv)) continue;",
        "      lo = Math.min(lo, vv);",
        "      hi = Math.max(hi, vv);",
        "    }",
        "",
        "    const a = (vmin != null) ? vmin : lo;",
        "    const b = (vmax != null) ? vmax : hi;",
        "",
        "    const feats = [];",
        "    for (let i = 0; i < p.length; i++) {",
        "      const pt = p[i];",
        "      const lat = Number(pt.lat);",
        "      const lon = Number(pt.lon);",
        "      const vv = pt.v;",
        "      if (!isFinite(lat) || !isFinite(lon)) continue;",
        "",
        "      let t = 0.5;",
        "      if (vv != null && isFinite(vv) && b > a) {",
        "        t = (vv - a) / (b - a);",
        "      }",
        "      t = _clamp(t, 0, 1);",
        "      const col = _color(t, cmap, inv, pal);",
        "",
        "      feats.push({",
        "        type: 'Feature',",
        "        geometry: { type: 'Point', coordinates: [lon, lat] },",
        "        properties: { col: col, sid: (pt.sid != null ? Number(pt.sid) : null) }",
        "      });",
        "    }",
        "",
        "    try {",
        "      const s = map.getSource('gp_points');",
        "      if (s) s.setData({",
        "        type: 'FeatureCollection',",
        "        features: feats",
        "      });",
        "    } catch (e) {}",
        "",
        "    try {",
        "      map.setPaintProperty(",
        "        'gp_points_layer', 'circle-radius', r",
        "      );",
        "      map.setPaintProperty(",
        "        'gp_points_layer', 'circle-opacity', op",
        "      );",
        "    } catch (e) {}",
        "",
        "    if (o.showLegend) {",
        "      setLegend(a, b, o.label || 'Z', cmap, inv, pal, pos, ori);",
        "      lastLegend = {",
        "        vmin: a, vmax: b, label: o.label || 'Z',",
        "        cmap: cmap, inv: inv",
        "      };",
        "    }",
        "  }",
        "",
        "  function fitPoints() {",
        "    const pts = [];",
        "    for (let i = 0; i < lastPoints.length; i++) {",
        "      const pt = lastPoints[i];",
        "      const lat = Number(pt.lat);",
        "      const lon = Number(pt.lon);",
        "      if (!isFinite(lat) || !isFinite(lon)) continue;",
        "      pts.push([lon, lat]);",
        "    }",
        "    for (let i = 0; i < lastHotspots.length; i++) {",
        "      const ht = lastHotspots[i] || {};",
        "      const lat = Number(ht.lat);",
        "      const lon = Number(ht.lon);",
        "      if (!isFinite(lat) || !isFinite(lon)) continue;",
        "      pts.push([lon, lat]);",
        "    }",
        "    if (!pts.length) return;",
        "",
        "    let minX = pts[0][0], maxX = pts[0][0];",
        "    let minY = pts[0][1], maxY = pts[0][1];",
        "    for (let i = 1; i < pts.length; i++) {",
        "      minX = Math.min(minX, pts[i][0]);",
        "      maxX = Math.max(maxX, pts[i][0]);",
        "      minY = Math.min(minY, pts[i][1]);",
        "      maxY = Math.max(maxY, pts[i][1]);",
        "    }",
        "    map.fitBounds([[minX, minY], [maxX, maxY]], {",
        "      padding: 40",
        "    });",
        "  }",
        "",
        "  function _asLonLatPair(p) {",
        "    if (!p || p.length < 2) return null;",
        "    const a = Number(p[0]);",
        "    const b = Number(p[1]);",
        "    if (!isFinite(a) || !isFinite(b)) return null;",
        "    // Accept [lon,lat] or [lat,lon] (auto-detect)",
        "    if (Math.abs(a) > 90 && Math.abs(b) <= 90) return [a, b];",
        "    if (Math.abs(b) > 90 && Math.abs(a) <= 90) return [b, a];",
        "    return [a, b];",
        "  }",
        "",
        "  function fitBounds(coords) {",
        "    if (!map) return;",
        "    if (!coords || coords.length < 2) return;",
        "    const p0 = _asLonLatPair(coords[0]);",
        "    const p1 = _asLonLatPair(coords[1]);",
        "    if (!p0 || !p1) return;",
        "    const minLon = Math.min(p0[0], p1[0]);",
        "    const minLat = Math.min(p0[1], p1[1]);",
        "    const maxLon = Math.max(p0[0], p1[0]);",
        "    const maxLat = Math.max(p0[1], p1[1]);",
        "    try {",
        "      map.fitBounds([[minLon, minLat], [maxLon, maxLat]], {",
        "        padding: 40",
        "      });",
        "    } catch (e) {}",
        "  }",
        "",
        "  function zoomIn() { map.zoomIn(); }",
        "  function zoomOut() { map.zoomOut(); }",
        "",
        "  let _vecMarkers = [];",
        "  function _clearVecMarkers() {",
        "    for (let i = 0; i < _vecMarkers.length; i++) {",
        "      try { _vecMarkers[i].remove(); } catch (e) {}",
        "    }",
        "    _vecMarkers = [];",
        "  }",
        "",
        "  function setVectors(vecs, opts) {",
        "    lastVectors = vecs || [];",
        "    lastVecOpts = opts || {};",
        "    if (!_gpLoaded) return;",
        "    _clearVecMarkers();",
        "",
        "    const v = lastVectors || [];",
        "    if (!v.length) return;",
        "",
        "    let maxMag = 0;",
        "    for (let i = 0; i < v.length; i++) {",
        "      const m = Number((v[i] || {}).mag);",
        "      if (isFinite(m) && m > maxMag) maxMag = m;",
        "    }",
        "    if (!(maxMag > 0)) maxMag = 1;",
        "",
        "    for (let i = 0; i < v.length; i++) {",
        "      const d = v[i] || {};",
        "      const lat = Number(d.lat);",
        "      const lon = Number(d.lon);",
        "      if (!isFinite(lat) || !isFinite(lon)) continue;",
        "",
        "      const a0 = Number(d.angle);",
        "      const a = isFinite(a0) ? a0 : 0;",
        "      const m0 = Number(d.mag);",
        "      const mag = isFinite(m0) ? Math.max(0, m0) : 0;",
        "      const t = Math.min(1, mag / maxMag);",
        "      const sc = 0.7 + 0.9 * t;",
        "      const op = 0.35 + 0.65 * t;",
        "      const col = (d.color ? String(d.color) : '');",
        "",
        "      const el = document.createElement('div');",
        "      el.className = 'prop-arrow';",
        "      el.textContent = '➤';",
        "      el.style.transform = 'rotate(' + a + 'deg) scale(' + sc + ')';",
        "      el.style.opacity = String(op);",
        "      if (col) el.style.color = col;",
        "      el.style.pointerEvents = 'none';",
        "",
        "      const mk = new maplibregl.Marker({",
        "        element: el,",
        "        anchor: 'center'",
        "      }).setLngLat([lon, lat]).addTo(map);",
        "",
        "      _vecMarkers.push(mk);",
        "    }",
        "  }",
        "",
        "  // Map background click -> point selection (only in point mode)",
        "  map.on('click', function (e) {",
        "    if (!e || !e.lngLat) return;",
        "    const now = Date.now();",
        "    if (now < _suppressClickUntil) return;",
        "    if (selectMode !== 'point') return;",
        "    _emitPointClicked(e.lngLat.lng, e.lngLat.lat);",
        "  });",
        "",
        "  // Shift+drag rectangle -> bbox selection (group mode)",
        "  let _boxOn = false;",
        "  let _box0 = null;",
        "  let _boxEl = null;",
        "  let _hintEl = null;",
        "  let _lastMouse = null;",
        "",
        "  function _boxClear() {",
        "    if (_boxEl && _boxEl.parentNode) {",
        "      try { _boxEl.parentNode.removeChild(_boxEl); }",
        "      catch (e) {}",
        "    }",
        "    _boxEl = null;",
        "    _boxOn = false;",
        "    _box0 = null;",
        "  }",
        "",
        "  function _boxSet(p0, p1) {",
        "    if (!_boxEl) return;",
        "    const x0 = Math.min(p0.x, p1.x);",
        "    const y0 = Math.min(p0.y, p1.y);",
        "    const x1 = Math.max(p0.x, p1.x);",
        "    const y1 = Math.max(p0.y, p1.y);",
        "    _boxEl.style.left = x0 + 'px';",
        "    _boxEl.style.top = y0 + 'px';",
        "    _boxEl.style.width = (x1 - x0) + 'px';",
        "    _boxEl.style.height = (y1 - y0) + 'px';",
        "  }",
        "",
        "  function _hintEnsure() {",
        "    if (_hintEl) return;",
        "    _hintEl = document.createElement('div');",
        "    _hintEl.className = 'gp-sel-rect gp-sel-hint';",
        "    _hintEl.style.width = '28px';",
        "    _hintEl.style.height = '20px';",
        "    _hintEl.style.display = 'none';",
        "    _host.appendChild(_hintEl);",
        "  }",
        "",
        "  function _hintHide() {",
        "    if (_hintEl) _hintEl.style.display = 'none';",
        "  }",
        "",
        "  function _hintShowAt(p) {",
        "    if (selectMode !== 'group' || !p) return;",
        "    _hintEnsure();",
        "    _hintEl.style.display = 'block';",
        "    _hintEl.style.left = (p.x - 14) + 'px';",
        "    _hintEl.style.top = (p.y - 10) + 'px';",
        "  }",
        "",
        "  const _host = map.getContainer();",
        "  _host.addEventListener('mousedown', function (ev) {",
        "    if (!ev) return;",
        "    if (selectMode !== 'group') return;",
        "    if (ev.button != null && ev.button !== 0) return;",
        "    const r = _host.getBoundingClientRect();",
        "    const p = { x: ev.clientX - r.left, y: ev.clientY - r.top };",
        "    _lastMouse = p;",
        "    _hintHide();",
        "    _boxOn = true;",
        "    _box0 = p;",
        "    try { map.dragPan.disable(); } catch (e) {}",
        "    try { map.boxZoom.disable(); } catch (e) {}",
        "    _boxEl = document.createElement('div');",
        "    _boxEl.className = 'gp-sel-rect';",
        "    _host.appendChild(_boxEl);",
        "    _boxSet(_box0, p);",
        "    ev.preventDefault();",
        "    ev.stopPropagation();",
        "  }, true);",
        "",
        "  _host.addEventListener('mousemove', function (ev) {",
        "    if (!ev) return;",
        "    const r = _host.getBoundingClientRect();",
        "    const p = { x: ev.clientX - r.left, y: ev.clientY - r.top };",
        "    _lastMouse = p;",
        "    if (selectMode === 'group' && !_boxOn) {",
        "      _hintShowAt(p);",
        "      return;",
        "    }",
        "    if (!_boxOn || !_box0) return;",
        "    _boxSet(_box0, p);",
        "    ev.preventDefault();",
        "  }, true);",
        "",
        "  _host.addEventListener('mouseleave', function () {",
        "    _hintHide();",
        "  }, true);",
        "",
        "  _host.addEventListener('mouseup', function (ev) {",
        "    if (!_boxOn || !_box0) return;",
        "    const r = _host.getBoundingClientRect();",
        "    const p1 = { x: ev.clientX - r.left, y: ev.clientY - r.top };",
        "    const x0 = Math.min(_box0.x, p1.x);",
        "    const y0 = Math.min(_box0.y, p1.y);",
        "    const x1 = Math.max(_box0.x, p1.x);",
        "    const y1 = Math.max(_box0.y, p1.y);",
        "    _boxClear();",
        "    try { map.dragPan.enable(); } catch (e) {}",
        "    try { map.boxZoom.enable(); } catch (e) {}",
        "    _lastMouse = p1;",
        "    if (selectMode === 'group') _hintShowAt(p1);",
        "    _suppressClickUntil = Date.now() + 250;",
        "    if (Math.abs(x1 - x0) < 4 || Math.abs(y1 - y0) < 4) return;",
        "    const a = map.unproject([x0, y0]);",
        "    const b = map.unproject([x1, y1]);",
        "    const minLon = Math.min(a.lng, b.lng);",
        "    const maxLon = Math.max(a.lng, b.lng);",
        "    const minLat = Math.min(a.lat, b.lat);",
        "    const maxLat = Math.max(a.lat, b.lat);",
        "    _emitGroupBBox(minLon, minLat, maxLon, maxLat);",
        "    ev.preventDefault();",
        "    ev.stopPropagation();",
        "  }, true);",
        "  // Install layers on load",
        "  map.on('load', function () {",
        "    _gpLoaded = true;",
        "    window.__GeoPriorMap.__loaded = true;",
        "",
        "    // Install raster basemap first (style is always GP_EMPTY_STYLE)",
        "    try {",
        "      const spec = _basemapSpec(lastBasemap.p, lastBasemap.s);",
        "      _applyRasterBasemap(spec, lastBasemap.o);",
        "    } catch (e) {}",
        "",
        "    try {",
        "      _installOverlays();",
        "    } catch (e) {}",
        "",
        "  });",
        "",
        "  // Attach real functions",
        "  window.__GeoPriorMap.setPoints = setPoints;",
        "  window.__GeoPriorMap.setHexbin = setHexbin;",
        "  window.__GeoPriorMap.setContours = setContours;",
        "  window.__GeoPriorMap.clearPoints = clearPoints;",
        "  window.__GeoPriorMap.fitPoints = fitPoints;",
        "  window.__GeoPriorMap.fitBounds = fitBounds;",
        "  window.__GeoPriorMap.zoomIn = zoomIn;",
        "  window.__GeoPriorMap.zoomOut = zoomOut;",
        "  window.__GeoPriorMap.setLegend = setLegend;",
        "  window.__GeoPriorMap.setBasemap = setBasemap;",
        "  window.__GeoPriorMap.setVectors = setVectors;",
        "  window.__GeoPriorMap.setHotspots = function (hs, o) {",
        "    lastHotspots = hs || [];",
        "    lastHotOpts = o || {};",
        "    setHotspots(hs, o);",
        "  };",
        "  window.__GeoPriorMap.clearHotspots = clearHotspots;",
        "  window.__GeoPriorMap.showHotspots = showHotspots;",
        "})();",
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(lines)
