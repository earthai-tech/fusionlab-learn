
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
  
  .gp-link-label {
    background: rgba(17,24,39,0.85);
    color: #fff;
    border: 0;
    border-radius: 8px;
    padding: 2px 6px;
    font: 12px ui-monospace, monospace;
  }

  .gp-arrow {
    font-size: 16px;
    line-height: 16px;
    color: var(--gp-ac);
    transform: rotate(var(--gp-rot));
    transform-origin: 50% 50%;
    text-shadow: 0 2px 6px rgba(0,0,0,0.25);
    user-select: none;
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
    setLegend: function(){},
    setLinks: function(){},
    clearLinks: function(){},
    setRadar: function(){},
    clearRadar: function(){}
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
  const linkLayers = {};   // id -> L.LayerGroup
  const radarLayers = {};  // id -> L.LayerGroup
  const radarTimers = {};  // id -> timer id


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
    // clear normal point layers
    const ids = Object.keys(layers);
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const g = layers[id];
      if (g) map.removeLayer(g);
      if (g) delete overlays[g.__name || id];
      delete layers[id];
    }

    // clear link layers
    const lids = Object.keys(linkLayers);
    for (let i = 0; i < lids.length; i++) {
      const id = lids[i];
      const g = linkLayers[id];
      if (g) map.removeLayer(g);
      if (g) delete overlays[g.__name || id];
      delete linkLayers[id];
    }

    // clear radar layers + timers
    const rids = Object.keys(radarLayers);
    for (let i = 0; i < rids.length; i++) {
      const id = rids[i];

      if (radarTimers[id]) {
        clearInterval(radarTimers[id]);
        delete radarTimers[id];
      }

      const g = radarLayers[id];
      if (g) map.removeLayer(g);
      if (g) delete overlays[g.__name || id];
      delete radarLayers[id];
    }

    _rebuildLayerCtl();
  }
  
  // -------------------------
  // Links (arrows + distance labels)
  // -------------------------
  function _arrowIcon(angleDeg, color) {
    const html = `
      <div class="gp-arrow"
        style="--gp-rot:${angleDeg}deg;--gp-ac:${color};">
        ▶
      </div>`;
    return L.divIcon({
      className: '',
      html: html,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    });
  }

  function _angleDeg(a, b) {
    const z = map.getZoom();
    const p1 = map.project(a, z);
    const p2 = map.project(b, z);
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    return Math.atan2(dy, dx) * 180 / Math.PI;
  }

  function clearLinks(id) {
    const g = linkLayers[id];
    if (!g) return;
    map.removeLayer(g);
    delete overlays[g.__name || id];
    delete linkLayers[id];
    _rebuildLayerCtl();
  }

  function setLinks(id, name, links, opts) {
    clearLinks(id);

    const o = opts || {};
    const color = o.color || '#111827';

    const g = L.layerGroup();
    g.__name = name || id;

    for (let i = 0; i < links.length; i++) {
      const p = links[i];
      const a = L.latLng(p[0], p[1]);
      const b = L.latLng(p[2], p[3]);
      const tip = p[5] || '';
      let label = p[6] || '';
      if (!label && typeof p[4] === 'number') {
        label = `${p[4].toFixed(2)} km`;
      }

      const line = L.polyline([a, b], {
        color: color,
        weight: 2,
        opacity: 0.85
      }).addTo(g);

      if (tip) {
        line.bindTooltip(
          String(tip).replace(/\n/g, '<br/>')
        );
      }

      if (o.arrow !== false) {
        const ang = _angleDeg(a, b);
        const m = L.marker(b, {
          icon: _arrowIcon(ang, color),
          interactive: false,
        }).addTo(g);

        // store endpoints so we can recompute angle on zoom/move
        m.__gp_link_a = a;
        m.__gp_link_b = b;
        m.__gp_link_color = color;
      }

      if (o.label === true && label) {
        const mid = L.latLng(
          (a.lat + b.lat) / 2.0,
          (a.lng + b.lng) / 2.0
        );
        const lm = L.circleMarker(mid, {
          radius: 1,
          opacity: 0,
          fillOpacity: 0
        }).addTo(g);

        lm.bindTooltip(String(label), {
          permanent: true,
          direction: 'center',
          className: 'gp-link-label',
          opacity: 0.95
        }).openTooltip();
      }
    }

    g.addTo(map);
    linkLayers[id] = g;
    overlays[g.__name] = g;
    _ensureLayerCtl();
    _rebuildLayerCtl();
  }

  map.on('zoomend moveend', function () {
    const ids = Object.keys(linkLayers);
    for (let k = 0; k < ids.length; k++) {
      const g = linkLayers[ids[k]];
      if (!g) continue;
      g.eachLayer(function (ly) {
        if (!ly || !ly.setIcon) return;
        if (!ly.__gp_link_a || !ly.__gp_link_b) return;

        const ang = _angleDeg(ly.__gp_link_a, ly.__gp_link_b);
        const col = ly.__gp_link_color || '#111827';
        ly.setIcon(_arrowIcon(ang, col));
      });
    }
  });

  // -------------------------
  // Radar sweep (rings + wedge sweep)
  // -------------------------
  function clearRadar(id) {
    if (radarTimers[id]) {
      clearInterval(radarTimers[id]);
      delete radarTimers[id];
    }
    const g = radarLayers[id];
    if (!g) return;
    map.removeLayer(g);
    delete overlays[g.__name || id];
    delete radarLayers[id];
    _rebuildLayerCtl();
  }

  function _offsetLatLng(lat, lon, dxm, dym) {
    const dlat = dym / 111320.0;
    const dlon = dxm / (
      111320.0 * Math.max(1e-9, Math.cos(lat * Math.PI / 180))
    );
    return [lat + dlat, lon + dlon];
  }

  function _wedge(center, rM, angDeg, widthDeg) {
    const lat = center.lat;
    const lon = center.lng;

    const a0 = (angDeg - widthDeg / 2) * Math.PI / 180;
    const a1 = (angDeg + widthDeg / 2) * Math.PI / 180;

    const p0 = _offsetLatLng(
      lat,
      lon,
      rM * Math.cos(a0),
      rM * Math.sin(a0)
    );
    const p1 = _offsetLatLng(
      lat,
      lon,
      rM * Math.cos(a1),
      rM * Math.sin(a1)
    );

    return [[lat, lon], p0, p1];
  }

  function setRadar(id, centers, opts) {
    clearRadar(id);

    const o = opts || {};
    const dwellMs = o.dwellMs || 520;
    const rKm = o.radiusKm || 8.0;
    const rings = Math.max(1, Math.min(8, o.rings || 3));

    if (!centers || !centers.length) return;

    const g = L.layerGroup();
    g.__name = 'Radar';

    let idx = 0;
    let baseT = Date.now();

    let ringLayersLocal = [];
    let wedgePoly = null;

    function _setCenter(c) {
      for (let i = 0; i < ringLayersLocal.length; i++) {
        g.removeLayer(ringLayersLocal[i]);
      }
      ringLayersLocal = [];

      const lat = c[0], lon = c[1];
      const center = L.latLng(lat, lon);

      const rM = (rKm * 1000.0);
      for (let k = 1; k <= rings; k++) {
        const rr = (rM * k) / rings;
        const cr = L.circle(center, {
          radius: rr,
          color: '#00C853',
          weight: 1,
          opacity: 0.35,
          fillOpacity: 0
        });
        cr.addTo(g);
        ringLayersLocal.push(cr);
      }

      if (wedgePoly) g.removeLayer(wedgePoly);
      wedgePoly = L.polygon(_wedge(center, rM, 0, 26), {
        color: '#00C853',
        weight: 1,
        opacity: 0.30,
        fillColor: '#00C853',
        fillOpacity: 0.12
      }).addTo(g);
    }

    _setCenter(centers[idx]);

    g.addTo(map);
    radarLayers[id] = g;
    overlays[g.__name] = g;
    _ensureLayerCtl();
    _rebuildLayerCtl();

    radarTimers[id] = setInterval(function () {
      const now = Date.now();
      let dt = now - baseT;

      if (dt >= dwellMs) {
        idx = (idx + 1) % centers.length;
        baseT = now;
        dt = 0;
        _setCenter(centers[idx]);
      }

      const c = centers[idx];
      const center = L.latLng(c[0], c[1]);
      const rM = (rKm * 1000.0);

      const ang = (dt / dwellMs) * 360.0;
      const pts = _wedge(center, rM, ang, 26);
      if (wedgePoly) wedgePoly.setLatLngs(pts);
    }, 40);
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
  window.__GeoPriorMap.setLinks = setLinks;
  window.__GeoPriorMap.clearLinks = clearLinks;

  window.__GeoPriorMap.setRadar = setRadar;
  window.__GeoPriorMap.clearRadar = clearRadar;

})();
</script>
</body>
</html>
"""