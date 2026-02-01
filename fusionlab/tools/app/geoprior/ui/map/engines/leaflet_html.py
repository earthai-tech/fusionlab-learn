
from __future__ import annotations

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
        "  .prop-arrow {",
        "    font-size: 16px;",
        "    color: #d7263d;",
        "    text-align: center;",
        "    line-height: 16px;",
        "    font-weight: bold;",
        "    text-shadow: 0 0 2px white;",
        "  }",
        "</style>",
        "</head>",
        "<body>",
        '<div id="map"></div>',
        "<script>",
        "(function () {",
        "  // Define API first (safe if Leaflet fails)",
        "  window.__GeoPriorMap = {",
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
        "  if (typeof L === 'undefined') {",
        "    console.log('Leaflet not available.');",
        "    return;",
        "  }",
        "",
        "  const map = L.map('map', {",
        "    zoomControl: true",
        "  }).setView([0, 0], 2);",
        "",
        "  // Remove only the Leaflet prefix, keep provider attribution",
        "  map.attributionControl.setPrefix('');",
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
        "  let lastMainKind = 'points';",
        "  let lastMainPoints = [];",
        "  let lastMainOpts = {};",
        "  let contourOverlay = null;",
        "  // Qt WebChannel bridge (JS -> Python)",
        "  let bridge = null;",
        "",
        "  // Hotspots layer (separate from points)",
        "  const hotLayer = L.layerGroup().addTo(map);",
        "  let hotOn = true;",
        "  let hotTimers = [];",
        "  const vectorLayer = L.layerGroup().addTo(map);",
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
        "  function setVectors(vecs) {",
        "    vectorLayer.clearLayers();",
        "    const v = vecs || [];",
        "    for (let i = 0; i < v.length; i++) {",
        "      const d = v[i];",
        "      if (!isFinite(d.lat) || !isFinite(d.lon)) continue;",
        "",
        "      const icon = L.divIcon({",
        "        className: 'prop-arrow-container',",
        "        html: '<div class=\"prop-arrow\" style=\"transform: rotate(' + (d.angle - 90) + 'deg);\">➤</div>',",
        "        iconSize: [20, 20],",
        "        iconAnchor: [10, 10]",
        "      });",
        "",
        "      L.marker([d.lat, d.lon], { icon: icon, interactive: false }).addTo(vectorLayer);",
        "    }",
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
        "    if (contourOverlay) {",
        "      try { map.removeLayer(contourOverlay); }",
        "      catch (e) {}",
        "      contourOverlay = null;",
        "    }",
        "    lastMainKind = 'points';",
        "    lastMainPoints = [];",
        "    lastMainOpts = {};",
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
        "  function _normBasemapArgs(provider, style, opacity) {",
        "    let p0 = provider;",
        "    let s0 = style;",
        "    let o0 = opacity;",
        "    // Allow passing a single object: {p/s/o} or {provider/style/opacity}",
        "    if (p0 && (typeof p0 === 'object')) {",
        "      o0 = (p0.o != null) ? p0.o : ((p0.opacity != null) ? p0.opacity : o0);",
        "      s0 = (p0.s != null) ? p0.s : ((p0.style != null) ? p0.style : s0);",
        "      p0 = (p0.p != null) ? p0.p : ((p0.provider != null) ? p0.provider : p0.key);",
        "    }",
        "    const p = String(p0 || 'osm').toLowerCase();",
        "    const s = String(s0 || 'light').toLowerCase();",
        "    const o = (o0 != null) ? Number(o0) : 1.0;",
        "    return { p: p, s: s, o: o };",
        "  }",
        "",
        "  function setBasemap(provider, style, opacity) {",
        "    const a = _normBasemapArgs(provider, style, opacity);",
        "    const p = a.p;",
        "    const s = a.s;",
        "",
        "    let url = osmUrl;",
        "    let att = osmAtt;",
        "    let sub = 'abc';",
        "    let maxZ = 19;",
        "",
        "    if (p === 'osm' && s === 'dark') {",
        "      url = (",
        "        'https://{s}.basemaps.cartocdn.com/' +",
        "        'dark_all/{z}/{x}/{y}{r}.png'",
        "      );",
        "      att = '© OpenStreetMap © CARTO';",
        "      sub = 'abcd';",
        "      maxZ = 19;",
        "    } else if (p === 'osm' && s === 'gray') {",
        "      url = (",
        "        'https://{s}.basemaps.cartocdn.com/' +",
        "        'light_all/{z}/{x}/{y}{r}.png'",
        "      );",
        "      att = '© OpenStreetMap © CARTO';",
        "      sub = 'abcd';",
        "      maxZ = 19;",
        "    } else if (p === 'terrain') {",
        "      url = 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png';",
        "      att = '© OpenStreetMap © OpenTopoMap';",
        "      sub = 'abc';",
        "      maxZ = 17;",
        "    } else if (p === 'satellite') {",
        "      url = (",
        "        'https://server.arcgisonline.com/' +",
        "        'ArcGIS/rest/services/World_Imagery/' +",
        "        'MapServer/tile/{z}/{y}/{x}'",
        "      );",
        "      att = 'Tiles © Esri';",
        "      sub = '';",
        "      maxZ = 19;",
        "    }",
        "",
        "    const op = (isFinite(a.o)) ? a.o : 1.0;",
        "",
        "    if (tileLayer) {",
        "      map.removeLayer(tileLayer);",
        "      tileLayer = null;",
        "    }",
        "",
        "    tileLayer = L.tileLayer(url, {",
        "      maxZoom: maxZ,",
        "      attribution: att,",
        "      opacity: op,",
        "      subdomains: sub",
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
        "      if (m.getBounds) {",
        "        try {",
        "          const b = m.getBounds();",
        "          pts.push(b.getNorthWest());",
        "          pts.push(b.getSouthEast());",
        "        } catch (e) {}",
        "      }",
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
        "  function fitBounds(coords) {",
        "    // coords = [[lat1, lon1], [lat2, lon2]]",
        "    if (!coords || coords.length !== 2) return;",
        "    try {",
        "      map.fitBounds(coords, { padding: [50, 50] });",
        "    } catch (e) {",
        "      console.log('fitBounds error:', e);",
        "    }",
        "  }",
        "",
                r"""
          // -------------------------------------------------
          // Visualization strategies (hexbin / contours)
          // -------------------------------------------------
        
          function _agg(vals, metric) {
            const m = (metric || "mean").toLowerCase();
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
        
          function _renderHexbin() {
            const pts = lastMainPoints || [];
            const o = lastMainOpts || {};
        
            clearPoints();
        
            const gs = Math.max(5, Number(o.gridsize || 30));
            const metric = String(o.metric || "mean");
            const op = (o.opacity != null) ? Number(o.opacity) : 0.85;
        
            if (!pts.length) return;
        
            const bins = {};
            const sqrt3 = Math.sqrt(3);
        
            for (let i = 0; i < pts.length; i++) {
              const p = pts[i];
              const q = map.latLngToContainerPoint([p.lat, p.lon]);
              const x = q.x;
              const y = q.y;
        
              const col = Math.round(x / (1.5 * gs));
              const row = Math.round(
                (y - (col & 1) * (sqrt3 * 0.5 * gs)) / (sqrt3 * gs)
              );
              const key = col + "," + row;
              if (!bins[key]) bins[key] = {col: col, row: row, vs: []};
              bins[key].vs.push(Number(p.v));
            }
        
            let lo = Infinity;
            let hi = -Infinity;
            for (const k in bins) {
              if (!bins.hasOwnProperty(k)) continue;
              const it = bins[k];
              const v = _agg(it.vs, metric);
              if (v == null) continue;
              it.v = v;
              lo = Math.min(lo, v);
              hi = Math.max(hi, v);
            }
        
            const a = (o.vmin != null) ? Number(o.vmin) : lo;
            const b = (o.vmax != null) ? Number(o.vmax) : hi;
        
            for (const k in bins) {
              if (!bins.hasOwnProperty(k)) continue;
              const it = bins[k];
              if (it.v == null) continue;
        
              const cx = it.col * 1.5 * gs;
              const cy = it.row * (sqrt3 * gs) + (it.col & 1)
                * (sqrt3 * 0.5 * gs);
        
              const t = _clamp((it.v - a) / ((b - a) || 1), 0, 1);
              const col = _color(t, o.cmap, !!o.invert);
        
              const latlngs = [];
              for (let j = 0; j < 6; j++) {
                const ang = (Math.PI / 3) * j;
                const px = cx + gs * Math.cos(ang);
                const py = cy + gs * Math.sin(ang);
                const ll = map.containerPointToLatLng(L.point(px, py));
                latlngs.push(ll);
              }
        
              L.polygon(latlngs, {
                color: col,
                weight: 1,
                fillColor: col,
                fillOpacity: op,
                interactive: false,
              }).addTo(layer);
            }
        
            if (o.showLegend) {
              setLegend(a, b, o.label || "Z", o.cmap, !!o.invert);
            }
          }
        
          function setHexbin(points, opts) {
            lastMainKind = "hexbin_source";
            lastMainPoints = points || [];
            lastMainOpts = opts || {};
            _renderHexbin();
          }
        
          function _renderContours() {
            const pts = lastMainPoints || [];
            const o = lastMainOpts || {};
        
            clearPoints();
        
            const op = (o.opacity != null) ? Number(o.opacity) : 0.7;
            const bw = Math.max(4, Number(o.bandwidth || 15));
            const step = Math.max(6, Math.min(80, bw));
            const size = map.getSize();
            const w = size.x;
            const h = size.y;
        
            if (!pts.length || !w || !h) return;
        
            const proj = [];
            let lo = Infinity;
            let hi = -Infinity;
            for (let i = 0; i < pts.length; i++) {
              const p = pts[i];
              const q = map.latLngToContainerPoint([p.lat, p.lon]);
              const v = Number(p.v);
              proj.push({x: q.x, y: q.y, v: v});
              lo = Math.min(lo, v);
              hi = Math.max(hi, v);
            }
        
            const a = (o.vmin != null) ? Number(o.vmin) : lo;
            const b = (o.vmax != null) ? Number(o.vmax) : hi;
        
            const cvs = document.createElement("canvas");
            cvs.width = w;
            cvs.height = h;
        
            const ctx = cvs.getContext("2d");
            if (!ctx) return;
        
            ctx.globalAlpha = op;
        
            const sig2 = 2 * step * step;
        
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
        
            const url = cvs.toDataURL("image/png");
            const bounds = map.getBounds();
            contourOverlay = L.imageOverlay(url, bounds, {
              opacity: 1.0,
              interactive: false,
            }).addTo(map);
        
            if (o.showLegend) {
              setLegend(a, b, o.label || "Z", o.cmap, !!o.invert);
            }
          }
        
          function setContours(points, opts) {
            lastMainKind = "contour_source";
            lastMainPoints = points || [];
            lastMainOpts = opts || {};
            _renderContours();
          }
        
          map.on("zoomend moveend", function () {
            if (lastMainKind === "hexbin_source") _renderHexbin();
            if (lastMainKind === "contour_source") _renderContours();
          });
        """,
        
        "  function zoomIn() { map.zoomIn(); }",
        "  function zoomOut() { map.zoomOut(); }",
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
        "  window.__GeoPriorMap.setHotspots = setHotspots;",
        "  window.__GeoPriorMap.clearHotspots = clearHotspots;",
        "  window.__GeoPriorMap.showHotspots = showHotspots;",
        "  window.__GeoPriorMap.setVectors = setVectors;",
        "})();",
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(lines)

# function setLayer(name, kind, data, opts) {
#   if (kind === "points") return drawPoints(name, data, opts);
#   if (kind === "hexbin_source") return drawHexbin(name, data, opts);
#   if (kind === "contour_source") return drawContour(name, data, opts);
# }
