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
        "  #map { height: 100%; width: 100%; }",
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
        "</style>",
        "</head>",
        "<body>",
        '<div id="map"></div>',
        "<script>",
        "(function () {",
        "  // Define API first (safe if engine fails)",
        "  window.__GeoPriorMap = {",
        "    __engine: 'maplibre',",
        "    __ready: false,",
        "    __failed: false,",
        "    __err: '',",
        "    __debug: function () {",
        "      return {",
        "        engine: String(this.__engine || ''),",
        "        ready: !!this.__ready,",
        "        failed: !!this.__failed,",
        "        err: String(this.__err || ''),",
        "      };",
        "    },",
        "    setPoints: function(){},",
        "    setHexbin: function(){},",
        "    setContours: function(){},",
        "    clearPoints: function(){},",
        "    fitPoints: function(){},",
        "    zoomIn: function(){},",
        "    zoomOut: function(){},",
        "    setLegend: function(){},",
        "    setBasemap: function(){},",
        "    setHotspots: function(){},",
        "    clearHotspots: function(){},",
        "    showHotspots: function(){},",
        "    setVectors: function(){}",
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
        "  // Cache for reload safety",
        "  let lastPoints = [];",
        "  let lastPointOpts = {};",
        "  let lastMainKind = 'points';",
        "  let lastHotspots = [];",
        "  let lastHotOpts = {};",
        "  let lastLegend = null;",
        "",
        "  // Map + layers",
        "  let _gpLoaded = false;",
        "  let lastBasemap = { p: 'osm', s: 'light', o: 1.0 };",
        "",
        "  const GP_EMPTY_STYLE = { version: 8, sources: {}, layers: [] };",
        "  const GP_OSM_VECTOR = 'https://demotiles.maplibre.org/style.json';",
        "  let gpActive = { kind: 'style', style: GP_OSM_VECTOR, tiles: '' };",
        "",
        "  let map = null;",
        "  try {",
        "    map = new maplibregl.Map({",
        "      container: 'map',",
        "      style: GP_OSM_VECTOR,",
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
        "    window.__GeoPriorMap.__failed = true;",
        "    if (e && e.error) {",
        "      window.__GeoPriorMap.__err = String(e.error);",
        "    } else {",
        "      window.__GeoPriorMap.__err = String(e);",
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
        "    const layers = (map.getStyle().layers || []);",
        "    if (layers.length) map.addLayer(layer, layers[0].id);",
        "    else map.addLayer(layer);",
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
        "",
        "  function _basemapSpec(provider, style) {",
        "    const p = String(provider || 'osm').toLowerCase();",
        "    const s = String(style || 'light').toLowerCase();",
        "",
        "    // Vector style for osm/light",
        "    if (p === 'osm' && s === 'light') {",
        "      return { kind: 'style', style: GP_OSM_VECTOR };",
        "    }",
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
        "",
        "    // Vector style mode",
        "    if (spec.kind === 'style' && spec.style) {",
        "      const want = { kind: 'style', style: spec.style, tiles: '' };",
        "      if (_sameBasemap(gpActive, want)) return;",
        "      gpActive = want;",
        "",
        "      window.__GeoPriorMap.__ready = false;",
        "      _removeRasterBasemap();",
        "      map.setStyle(spec.style);",
        "      map.once('style.load', function () {",
        "        _gpLoaded = true;",
        "        _installOverlays();",
        "        window.__GeoPriorMap.__ready = true;",
        "      });",
        "      return;",
        "    }",
        "",
        "    // Raster mode (use an empty style so satellite/terrain are clean)",
        "    const rasterKey = (spec.tiles || []).join('|');",
        "    const wantR = { kind: 'raster', style: '', tiles: rasterKey };",
        "",
        "    window.__GeoPriorMap.__ready = false;",
        "",
        "    // If we are not already in raster mode, reset style first",
        "    if (gpActive.kind !== 'raster') {",
        "      gpActive = wantR;",
        "      map.setStyle(GP_EMPTY_STYLE);",
        "      map.once('style.load', function () {",
        "        _gpLoaded = true;",
        "        _applyRasterBasemap(spec, opacity);",
        "        _installOverlays();",
        "        window.__GeoPriorMap.__ready = true;",
        "      });",
        "      return;",
        "    }",
        "",
        "    // Already raster mode: just swap tiles + opacity",
        "    gpActive = wantR;",
        "    _applyRasterBasemap(spec, opacity);",
        "    _installOverlays();",
        "    window.__GeoPriorMap.__ready = true;",
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
        "          _emitPointClicked(Number(c[0]), Number(c[1]));",
        "        } catch (err) {}",
        "      });",
        "    }",
        "  }",
        "",
        "  // Hotspots layers",
        "  let hotOn = true;",
        "  let hotTimer = null;",
        "",
        "  function _stopHotTimer() {",
        "    if (hotTimer != null) {",
        "      try { clearInterval(hotTimer); } catch (e) {}",
        "    }",
        "    hotTimer = null;",
        "  }",
        "",
        "  function _ensureHotLayers() {",
        "    if (!map.getSource('gp_hot_core')) {",
        "      map.addSource('gp_hot_core', {",
        "        type: 'geojson',",
        "        data: { type: 'FeatureCollection', features: [] }",
        "      });",
        "    }",
        "    if (!map.getSource('gp_hot_ring')) {",
        "      map.addSource('gp_hot_ring', {",
        "        type: 'geojson',",
        "        data: { type: 'FeatureCollection', features: [] }",
        "      });",
        "    }",
        "    if (!map.getLayer('gp_hot_ring_layer')) {",
        "      map.addLayer({",
        "        id: 'gp_hot_ring_layer',",
        "        type: 'circle',",
        "        source: 'gp_hot_ring',",
        "        paint: {",
        "          'circle-radius': ['get', 'r'],",
        "          'circle-color': ['get', 'col'],",
        "          'circle-opacity': 0.0,",
        "          'circle-stroke-color': ['get', 'col'],",
        "          'circle-stroke-opacity': ['get', 'a'],",
        "          'circle-stroke-width': 2",
        "        }",
        "      });",
        "    }",
        "    if (!map.getLayer('gp_hot_core_layer')) {",
        "      map.addLayer({",
        "        id: 'gp_hot_core_layer',",
        "        type: 'circle',",
        "        source: 'gp_hot_core',",
        "        paint: {",
        "          'circle-radius': 5,",
        "          'circle-color': ['get', 'col'],",
        "          'circle-opacity': 0.95,",
        "          'circle-stroke-color': ['get', 'col'],",
        "          'circle-stroke-width': 1",
        "        }",
        "      });",
        "",
        "      map.on('click', 'gp_hot_core_layer', function (e) {",
        "        try {",
        "          const f = e.features && e.features[0];",
        "          if (!f) return;",
        "          const g = f.geometry;",
        "          if (!g || g.type !== 'Point') return;",
        "          const c = g.coordinates || [];",
        "          _emitPointClicked(Number(c[0]), Number(c[1]));",
        "        } catch (err) {}",
        "      });",
        "    }",
        "  }",
        "",
        "  function showHotspots(on) {",
        "    hotOn = !!on;",
        "    const v = hotOn ? 'visible' : 'none';",
        "    if (map.getLayer('gp_hot_core_layer')) {",
        "      map.setLayoutProperty('gp_hot_core_layer',",
        "        'visibility', v);",
        "    }",
        "    if (map.getLayer('gp_hot_ring_layer')) {",
        "      map.setLayoutProperty('gp_hot_ring_layer',",
        "        'visibility', v);",
        "    }",
        "  }",
        "",
        "  function clearHotspots() {",
        "    _stopHotTimer();",
        "    lastHotspots = [];",
        "    lastHotOpts = {};",
        "    try {",
        "      const s1 = map.getSource('gp_hot_core');",
        "      if (s1) s1.setData({",
        "        type: 'FeatureCollection', features: []",
        "      });",
        "    } catch (e) {}",
        "    try {",
        "      const s2 = map.getSource('gp_hot_ring');",
        "      if (s2) s2.setData({",
        "        type: 'FeatureCollection', features: []",
        "      });",
        "    } catch (e) {}",
        "  }",
        "",
        "  function _kmToPx(km, lat) {",
        "    const z = map.getZoom();",
        "    const c = Math.cos((lat * Math.PI) / 180.0);",
        "    const mpp = (156543.03392 * c) / Math.pow(2, z);",
        "    const m = Number(km) * 1000.0;",
        "    if (!isFinite(mpp) || mpp <= 0) return 24;",
        "    const px = m / mpp;",
        "    return Math.max(6, Math.min(220, px));",
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
        "  function _pulseRings(rings, speed) {",
        "    _stopHotTimer();",
        "    let sp = Number(speed);",
        "    if (!isFinite(sp) || sp <= 0) sp = 1.0;",
        "    let dt = 80 / sp;",
        "    dt = Math.max(25, Math.min(200, dt));",
        "",
        "    hotTimer = setInterval(function () {",
        "      for (let i = 0; i < rings.length; i++) {",
        "        const r = rings[i];",
        "        r.r += (r.r0 * 0.06);",
        "        r.a -= 0.04;",
        "        if (r.a <= 0.05) {",
        "          r.r = r.r0;",
        "          r.a = 0.9;",
        "        }",
        "      }",
        "      // push update",
        "      const feats = [];",
        "      for (let i = 0; i < rings.length; i++) {",
        "        const rr = rings[i];",
        "        feats.push({",
        "          type: 'Feature',",
        "          geometry: rr.g,",
        "          properties: { col: rr.col, r: rr.r, a: rr.a }",
        "        });",
        "      }",
        "      try {",
        "        const s2 = map.getSource('gp_hot_ring');",
        "        if (s2) s2.setData({",
        "          type: 'FeatureCollection',",
        "          features: feats",
        "        });",
        "      } catch (e) {}",
        "    }, dt);",
        "  }",
        "",
        "  function setHotspots(hs, opts) {",
        "    clearHotspots();",
        "",
        "    const h = hs || [];",
        "    const o = opts || {};",
        "    if (!_gpLoaded) {",
        "      lastHotspots = h;",
        "      lastHotOpts = o;",
        "      return;",
        "    }",
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
        "    // build cores + rings",
        "    const cores = [];",
        "    const rings = [];",
        "",
        "    for (let i = 0; i < h.length; i++) {",
        "      const pt = h[i] || {};",
        "      const lat = Number(pt.lat);",
        "      const lon = Number(pt.lon);",
        "      if (!isFinite(lat) || !isFinite(lon)) continue;",
        "",
        "      const sev = pt.sev || 'high';",
        "      const col = _sevColor(sev);",
        "",
        "      const g = {",
        "        type: 'Point',",
        "        coordinates: [lon, lat]",
        "      };",
        "",
        "      cores.push({",
        "        type: 'Feature',",
        "        geometry: g,",
        "        properties: { col: col }",
        "      });",
        "",
        "      const r0 = _kmToPx(baseKm, lat);",
        "      const ring = {",
        "        g: g,",
        "        col: col,",
        "        r0: r0,",
        "        r: r0,",
        "        a: (style === 'glow') ? 0.55 : 0.9",
        "      };",
        "      rings.push(ring);",
        "    }",
        "",
        "    // cores update",
        "    try {",
        "      const s1 = map.getSource('gp_hot_core');",
        "      if (s1) s1.setData({",
        "        type: 'FeatureCollection',",
        "        features: cores",
        "      });",
        "    } catch (e) {}",
        "",
        "    // rings update once (and maybe animate)",
        "    const ringFeats = [];",
        "    for (let i = 0; i < rings.length; i++) {",
        "      const rr = rings[i];",
        "      ringFeats.push({",
        "        type: 'Feature',",
        "        geometry: rr.g,",
        "        properties: { col: rr.col, r: rr.r, a: rr.a }",
        "      });",
        "    }",
        "    try {",
        "      const s2 = map.getSource('gp_hot_ring');",
        "      if (s2) s2.setData({",
        "        type: 'FeatureCollection',",
        "        features: ringFeats",
        "      });",
        "    } catch (e) {}",
        "",
        "    if (style !== 'glow' && pulse) {",
        "      _pulseRings(rings, sp);",
        "    }",
        "",
        "    // tooltips (optional; minimal)",
        "    if (labels) {",
        "      // MapLibre raster style + basic UI: skip for now",
        "      // (kept to align with Leaflet API).",
        "    }",
        "  }",
        "",
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
                
        
        
        "  function _color(t, cmap, inv) {",
        "    let tt = t;",
        "    if (inv) tt = 1 - tt;",
        "    const h = 240 * (1 - tt);",
        "    return 'hsl(' + h + ',80%,45%)';",
        "  }",
        "",
        "  let legendEl = null;",
        "  function setLegend(vmin, vmax, label, cmap, inv) {",
        "    if (legendEl && legendEl.parentNode) {",
        "      try { legendEl.parentNode.removeChild(legendEl); }",
        "      catch (e) {}",
        "    }",
        "",
        "    const cm = cmap || 'viridis';",
        "    const iv = !!inv;",
        "    const title = (label || 'Z');",
        "    const g = (",
        "      'linear-gradient(to top,' +",
        "      _color(0, cm, iv) + ',' +",
        "      _color(1, cm, iv) + ')'",
        "    );",
        "",
        "    const div = document.createElement('div');",
        "    div.className = 'gp-legend';",
        "    div.innerHTML = (",
        "      '<div style=\"font-weight:600;\">' + title + '</div>' +",
        "      '<div style=\"display:flex;gap:8px;align-items:center;\">' +",
        "      '<div style=\"width:12px;height:90px;background:' + g +",
        "      ';border-radius:6px;\"></div>' +",
        "      '<div>' +",
        "      '<div>' + Number(vmax).toFixed(3) + '</div>' +",
        "      '<div style=\"height:62px;\"></div>' +",
        "      '<div>' + Number(vmin).toFixed(3) + '</div>' +",
        "      '</div></div>'",
        "    );",
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
        "      const col = _color(t, cmap, inv);",
        "",
        "      feats.push({",
        "        type: 'Feature',",
        "        geometry: { type: 'Point', coordinates: [lon, lat] },",
        "        properties: { col: col }",
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
        "      setLegend(a, b, o.label || 'Z', cmap, inv);",
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
        "  function zoomIn() { map.zoomIn(); }",
        "  function zoomOut() { map.zoomOut(); }",
        "",
        "  // Map click background also selects",
        "  map.on('click', function (e) {",
        "    if (!e || !e.lngLat) return;",
        "    _emitPointClicked(e.lngLat.lng, e.lngLat.lat);",
        "  });",
        "",
        "  // Install layers on load",
        "  map.on('load', function () {",
        "    _gpLoaded = true;",
        "    _ensurePointLayer();",
        "    _ensureHotLayers();",
        "    setBasemap(lastBasemap.p, lastBasemap.s, lastBasemap.o);",
        "    window.__GeoPriorMap.__ready = true;",
        "",
        "    // Replay cached state (if any)",
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
        "  });",
        "",
        "  // Attach real functions",
        "  window.__GeoPriorMap.setPoints = setPoints;",
        "  window.__GeoPriorMap.setHexbin = setHexbin;",
        "  window.__GeoPriorMap.setContours = setContours;",
        "  window.__GeoPriorMap.clearPoints = clearPoints;",
        "  window.__GeoPriorMap.fitPoints = fitPoints;",
        "  window.__GeoPriorMap.zoomIn = zoomIn;",
        "  window.__GeoPriorMap.zoomOut = zoomOut;",
        "  window.__GeoPriorMap.setLegend = setLegend;",
        "  window.__GeoPriorMap.setBasemap = setBasemap;",
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
