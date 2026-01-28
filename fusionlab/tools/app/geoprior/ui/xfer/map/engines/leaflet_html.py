
from __future__ import annotations 


_LEAFLET_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport"
      content="width=device-width,initial-scale=1.0"/>
<link rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script
  src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
</script>

<style>
  html, body { height:100%; margin:0; }
  #map { height:100%; width:100%; }

  .gp-legend {
    background: rgba(255,255,255,0.92);
    padding: 8px 10px;
    border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.12);
    font-family: sans-serif;
    font-size: 12px;
    line-height: 1.3;
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
  }

  .gp-legend .row {
    display:flex;
    align-items:center;
    gap:8px;
    margin-top:6px;
  }

  .gp-legend .bar {
    width: 120px;
    height: 10px;
    border-radius: 8px;
    background: linear-gradient(
      90deg,
      #2c7bb6,
      #abd9e9,
      #ffffbf,
      #fdae61,
      #d7191c
    );
    border: 1px solid rgba(0,0,0,0.10);
  }

  .gp-legend .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo,
                 Monaco, Consolas, "Liberation Mono",
                 "Courier New", monospace;
    opacity: 0.85;
  }
  .gp-mkr {
    position: relative; /* needed for ::after pulse */
    width: calc(var(--gp-r) * 2);
    height: calc(var(--gp-r) * 2);
    background: var(--gp-fill);
    opacity: var(--gp-op);
    border: 2px solid var(--gp-stroke);
    border-radius: 999px;
    box-sizing: border-box;
  }

  .gp-mkr.triangle {
    border-radius: 4px;
    clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
  }

  .gp-mkr.diamond {
    border-radius: 4px;
    clip-path: polygon(50% 0%, 0% 50%, 50% 100%, 100% 50%);
  }

  .gp-mkr.square {
    border-radius: 3px;
  }

  @keyframes gpPulse {
    0%   { transform: scale(0.95); opacity: 0.65; }
    60%  { transform: scale(1.25); opacity: 0.25; }
    100% { transform: scale(1.45); opacity: 0.00; }
  }

  .gp-pulse::after {
    content: "";
    position: absolute;
    left: 50%;
    top: 50%;
    width: calc(var(--gp-r) * 2);
    height: calc(var(--gp-r) * 2);
    transform: translate(-50%, -50%);
    border-radius: 999px;
    border: 2px solid var(--gp-stroke);
    animation: gpPulse 1.2s ease-out infinite;
    opacity: 0.7;
    pointer-events: none;
  }

</style>
</head>

<body>
<div id="map"></div>

<script>
(function () {
  // Public API (always defined).
  window.__GeoPriorMap = {
    setCentroids: function(){},
    clearCentroids: function(){},
    setLayer: function(){},
    clearLayer: function(){},
    clearLayers: function(){},
    fitLayers: function(){},
    setLegend: function(){}
  };

  if (typeof L === 'undefined') {
    console.log("Leaflet not available.");
    return;
  }

  const map = L.map('map', { zoomControl:true })
               .setView([0,0], 2);
  
  map.attributionControl.setPrefix('');

  L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { maxZoom:19, attribution:'© OpenStreetMap' }
  ).addTo(map);

  const canvas = L.canvas({ padding:0.3 });

  // Layer registry
  const layers = {};        // id -> L.LayerGroup
  const overlays = {};      // name -> L.LayerGroup
  let layerCtl = null;

  // Centroids + link
  let srcMarker = null;
  let tgtMarker = null;
  let linkLine = null;

  function _ensureLayerCtl() {
    if (layerCtl) return;
    layerCtl = L.control.layers(
      null,
      overlays,
      { collapsed:true }
    ).addTo(map);
  }

  function _rebuildLayerCtl() {
    if (!layerCtl) {
      _ensureLayerCtl();
      return;
    }
    layerCtl.remove();
    layerCtl = null;
    _ensureLayerCtl();
  }

  function _clamp01(x) {
    if (x < 0) return 0;
    if (x > 1) return 1;
    return x;
  }

  // Simple ramp for value fill
  function _ramp(t) {
    // blue -> cyan -> yellow -> orange -> red
    t = _clamp01(t);
    const stops = [
      [44,123,182],
      [171,217,233],
      [255,255,191],
      [253,174,97],
      [215,25,28]
    ];
    const n = stops.length - 1;
    const x = t * n;
    const i = Math.floor(x);
    const f = x - i;
    const a = stops[Math.max(0, Math.min(n, i))];
    const b = stops[Math.max(0, Math.min(n, i+1))];
    const r = Math.round(a[0] + (b[0]-a[0]) * f);
    const g = Math.round(a[1] + (b[1]-a[1]) * f);
    const bl = Math.round(a[2] + (b[2]-a[2]) * f);
    return `rgb(${r},${g},${bl})`;
  }

  function clearCentroids() {
    if (srcMarker) map.removeLayer(srcMarker);
    if (tgtMarker) map.removeLayer(tgtMarker);
    if (linkLine) map.removeLayer(linkLine);
    srcMarker = null;
    tgtMarker = null;
    linkLine = null;
  }

  function setCentroids(
    srcName, srcLat, srcLon,
    tgtName, tgtLat, tgtLon
  ) {
    clearCentroids();

    const src = [srcLat, srcLon];
    const tgt = [tgtLat, tgtLon];

    srcMarker = L.circleMarker(src, {
      radius: 7,
      color: '#2E3191',
      fillColor: '#2E3191',
      fillOpacity: 0.90
    }).addTo(map).bindPopup(srcName);

    tgtMarker = L.circleMarker(tgt, {
      radius: 7,
      color: '#F28620',
      fillColor: '#F28620',
      fillOpacity: 0.90
    }).addTo(map).bindPopup(tgtName);

    linkLine = L.polyline([src, tgt], {
      color: '#3399ff',
      weight: 2,
      opacity: 0.80
    }).addTo(map);
  }

  function clearLayer(id) {
    const g = layers[id];
    if (!g) return;
    map.removeLayer(g);
    delete overlays[g.__name || id];
    delete layers[id];
    _rebuildLayerCtl();
  }

  function clearLayers() {
    const ids = Object.keys(layers);
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const g = layers[id];
      if (g) map.removeLayer(g);
      if (g) delete overlays[g.__name || id];
      delete layers[id];
    }
    _rebuildLayerCtl();
  }

  function setLayer(id, name, points, opts) {
    clearLayer(id);

    const o = opts || {};
    const stroke = o.strokeColor || o.stroke || '#2E3191';
    const radius = o.radius || 6;
    const opacity = (o.opacity != null) ? o.opacity : 0.90;

    const fillMode = o.fillMode || o.fill_mode || 'value';
    const fillColor = o.fillColor || o.fill || stroke;

    const enableTip = (o.enableTooltip === true);

    const shape = (o.shape || 'circle');  // auto|circle|triangle|diamond|square
    const pulse = (o.pulse === true);

    const vmin = (o.vmin != null) ? o.vmin : null;
    const vmax = (o.vmax != null) ? o.vmax : null;

    const g = L.layerGroup();
    g.__name = name || id;

    const tooManyForHtml = (points.length > 2500);
    const useHtml = (
      !tooManyForHtml &&
      (shape !== 'circle' || pulse)
    );

    function _tooltipText(rawTip, sid, v) {
      let ttxt = '';

      if (rawTip != null && String(rawTip).length) {
        ttxt = String(rawTip);
      } else {
        if (sid != null) ttxt += `#${sid} `;
        if (v != null && !Number.isNaN(v)) ttxt += `v=${v}`;
      }

      // Multi-line tips (from Python) -> HTML breaks for Leaflet tooltips
      if (ttxt.indexOf('\n') >= 0) {
        ttxt = ttxt.replace(/\n/g, '<br/>');
      }
      return ttxt;
    }

    function _mkCircle(lat, lon, fc, v, sid, rawTip) {
      const m = L.circleMarker([lat, lon], {
        radius: radius,
        color: stroke,
        weight: 2,
        opacity: opacity,
        fillColor: fc,
        fillOpacity: opacity,
        renderer: canvas
      });

      const ttxt = _tooltipText(rawTip, sid, v);
      if (enableTip && ttxt) m.bindTooltip(ttxt);

      m.addTo(g);
    }

    function _mkHtml(lat, lon, fc, v, sid, rawTip) {
      const r = Math.max(2, radius);
      const cls = ['gp-mkr', String(shape)];
      if (pulse) cls.push('gp-pulse');

      const html = `<div class="${cls.join(' ')}"
        style="--gp-r:${r}px;--gp-fill:${fc};
               --gp-stroke:${stroke};--gp-op:${opacity};">
      </div>`;

      const ic = L.divIcon({
        className: '',
        html: html,
        iconSize: [2*r, 2*r],
        iconAnchor: [r, r],
      });

      const m = L.marker([lat, lon], { icon: ic });

      const ttxt = _tooltipText(rawTip, sid, v);
      if (enableTip && ttxt) m.bindTooltip(ttxt);

      m.addTo(g);
    }

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const lat = Array.isArray(p) ? p[0] : p.lat;
      const lon = Array.isArray(p) ? p[1] : p.lon;
      const v   = Array.isArray(p) ? p[2] : p.v;
      const sid = Array.isArray(p) ? p[3] : p.sid;
      const tip = Array.isArray(p) ? p[4] : p.tip;

      let fc = fillColor;
      if (fillMode === 'value' && vmin != null && vmax != null) {
        const t = (v - vmin) / (vmax - vmin + 1e-12);
        fc = _ramp(t);
      }

      if (useHtml) _mkHtml(lat, lon, fc, v, sid, tip);
      else _mkCircle(lat, lon, fc, v, sid, tip);
    }

    g.addTo(map);
    layers[id] = g;
    overlays[g.__name] = g;
    _ensureLayerCtl();
    _rebuildLayerCtl();
  }

  function fitLayers(layerIds) {
    const ids = layerIds && layerIds.length ? layerIds : null;
    const b = [];

    if (srcMarker) b.push(srcMarker.getLatLng());
    if (tgtMarker) b.push(tgtMarker.getLatLng());

    const useIds = ids ? ids : Object.keys(layers);
    useIds.forEach((id) => {
      const g = layers[id];
      if (!g) return;
      try {
        const bb = g.getBounds();
        if (bb.isValid()) {
          b.push(bb.getNorthEast());
          b.push(bb.getSouthWest());
        }
      } catch (e) {}
    });

    if (!b.length) return;
    map.fitBounds(L.latLngBounds(b).pad(0.20));
  }

  // Legend
  let legend = null;
  function _ensureLegend() {
    if (legend) return;
    legend = L.control({ position:'bottomright' });
    legend.onAdd = function () {
      const div = L.DomUtil.create('div', 'gp-legend');
      div.innerHTML = '';
      return div;
    };
    legend.addTo(map);
  }

  function setLegend(opts) {
    _ensureLegend();
    const o = opts || {};
    const title = o.title || 'Value';
    const unit  = o.unit || '';
    const vmin  = (o.vmin != null) ? o.vmin : '';
    const vmax  = (o.vmax != null) ? o.vmax : '';
    const mode  = o.mode || '';

    const el = legend.getContainer();
    el.innerHTML = `
      <div><b>${title}</b></div>
      <div class="mono">${mode}</div>
      <div class="row">
        <div class="mono">${vmin}</div>
        <div class="bar"></div>
        <div class="mono">${vmax}</div>
      </div>
      <div class="mono">${unit}</div>
    `;
  }

  // Attach
  window.__GeoPriorMap.setCentroids = setCentroids;
  window.__GeoPriorMap.clearCentroids = clearCentroids;

  window.__GeoPriorMap.setLayer = setLayer;
  window.__GeoPriorMap.clearLayer = clearLayer;
  window.__GeoPriorMap.clearLayers = clearLayers;
  window.__GeoPriorMap.fitLayers = fitLayers;
  window.__GeoPriorMap.setLegend = setLegend;
})();
</script>
</body>
</html>
"""