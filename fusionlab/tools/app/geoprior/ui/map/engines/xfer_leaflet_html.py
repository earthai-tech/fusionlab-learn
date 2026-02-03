# geoprior/ui/map/engines/leaflet_html.py
# -*- coding: utf-8 -*-

"""
Shared Leaflet HTML Template.
Centralized D3 (Contours) + Leaflet (Scatter/Radar/Links).
"""

from __future__ import annotations


_LEAFLET_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>

<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://d3js.org/d3-hexbin.v0.2.min.js"></script>

<style>
  html, body { height:100%; margin:0; }
  #map { height:100%; width:100%; }

  /* --- Legend --- */
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
  .gp-legend .row { display:flex; align-items:center; gap:8px; margin-top:6px; }
  .gp-legend .bar {
    width: 120px; height: 10px; border-radius: 8px;
    background: linear-gradient(90deg, #2c7bb6, #abd9e9, #ffffbf, #fdae61, #d7191c);
    border: 1px solid rgba(0,0,0,0.10);
  }
  .gp-legend .mono { font-family: monospace; opacity: 0.85; }

  /* --- Markers (Scatter) --- */
  .gp-mkr {
    position: relative; width: calc(var(--gp-r) * 2); height: calc(var(--gp-r) * 2);
    background: var(--gp-fill); opacity: var(--gp-op);
    border: 2px solid var(--gp-stroke); border-radius: 999px; box-sizing: border-box;
  }
  .gp-mkr.triangle { border-radius: 4px; clip-path: polygon(50% 0%, 0% 100%, 100% 100%); }
  .gp-mkr.diamond  { border-radius: 4px; clip-path: polygon(50% 0%, 0% 50%, 50% 100%, 100% 50%); }
  .gp-mkr.square   { border-radius: 3px; }

  @keyframes gpPulse {
    0%   { transform: scale(0.95); opacity: 0.65; }
    100% { transform: scale(1.45); opacity: 0.00; }
  }
  .gp-pulse::after {
    content: ""; position: absolute; left: 50%; top: 50%;
    width: calc(var(--gp-r) * 2); height: calc(var(--gp-r) * 2);
    transform: translate(-50%, -50%); border-radius: 999px;
    border: 2px solid var(--gp-stroke); animation: gpPulse 1.2s ease-out infinite;
    pointer-events: none;
  }
  
  /* --- Links --- */
  .gp-link-label {
    background: rgba(17,24,39,0.85); color: #fff; border: 0; border-radius: 8px;
    padding: 2px 6px; font: 12px monospace;
  }
  .gp-arrow {
    font-size: 16px; color: var(--gp-ac); transform: rotate(var(--gp-rot));
    transform-origin: 50% 50%; text-shadow: 0 2px 6px rgba(0,0,0,0.25);
  }
  
  /* --- Contour Labels --- */
  .gp-cont-label {
    font-family: sans-serif; font-weight: 600;
    pointer-events: none; text-shadow: 0 0 3px white;
  }
</style>
</head>
<body>
<div id="map"></div>

<script>
(function () {
  // --- 1. Map Initialization ---
  if (typeof L === 'undefined') return;

  const map = L.map('map', { zoomControl:true }).setView([0,0], 2);
  map.attributionControl.setPrefix('');
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom:19, attribution:'&copy; OpenStreetMap'
  }).addTo(map);

  const canvas = L.canvas({ padding:0.3 });

  // Global registries
  const layers = {};      
  const overlays = {};    
  let layerCtl = null;
  const linkLayers = {};
  const radarLayers = {};
  const radarTimers = {};
  let srcMarker = null; let tgtMarker = null; let linkLine = null;

  // --- 2. Color Utilities ---
  function _ramp(t) {
    if(t<0)t=0; if(t>1)t=1;
    const stops = [[44,123,182], [171,217,233], [255,255,191], [253,174,97], [215,25,28]];
    const n = stops.length - 1;
    const x = t * n;
    const i = Math.floor(x);
    const f = x - i;
    const a = stops[Math.max(0, Math.min(n, i))];
    const b = stops[Math.max(0, Math.min(n, i+1))];
    return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;
  }
  // --- 2b. Point / sampling helpers (used by hexbin) ---
  function _ptLat(d) { return Array.isArray(d) ? d[0] : d.lat; }
  function _ptLon(d) { return Array.isArray(d) ? d[1] : d.lon; }
  function _ptVal(d) { return Array.isArray(d) ? d[2] : d.v; }

  function _isFiniteNum(x) {
    const n = Number(x);
    return (Number.isFinite ? Number.isFinite(n) : isFinite(n));
  }

  function _colorFromDomain(v, vmin, vmax) {
    const vv = Number(v);
    const a = Number(vmin);
    const b = Number(vmax);
    if (!isFinite(vv) || !isFinite(a) || !isFinite(b) || Math.abs(b - a) < 1e-12) {
      return _ramp(0.5);
    }
    return _ramp((vv - a) / (b - a));
  }

  function _sampleEven(arr, maxN) {
    if (!arr) return [];
    const n = arr.length;
    const m = Math.max(1, parseInt(maxN || n, 10));
    if (n <= m) return arr;
    const step = Math.ceil(n / m);
    const out = [];
    for (let i = 0; i < n; i += step) out.push(arr[i]);
    return out;
  }

  // --- 3. THE LAYER FACTORY ---
  const LayerFactory = {
    render: function(id, kind, data, opts) {
      this.clear(id);
      let layer = null;
      if (kind === 'points' || kind === 'scatter') {
        layer = this._buildScatter(data, opts);
      } else if (kind === 'contour_source') {
        layer = this._buildContour(data, opts);
      } else if (kind === 'hexbin_source') {
        layer = this._buildHexbin(data, opts);
      }

      if (layer) {
        layer.addTo(map);
        layer.__name = opts.name || id; 
        layers[id] = layer;
        overlays[layer.__name] = layer;
        _rebuildLayerCtl();
      }
    },

    clear: function(id) {
      if (layers[id]) {
        map.removeLayer(layers[id]);
        const name = layers[id].__name;
        if(name) delete overlays[name];
        delete layers[id];
        _rebuildLayerCtl();
      }
    },

    // --- A. Scatter ---
    _buildScatter: function(data, opts) {
      const g = L.layerGroup();
      const o = opts || {};
      
      const radius = o.radius || 6;
      const opacity = o.opacity || 0.9;
      const stroke = o.stroke || '#2E3191';
      const shape = o.shape || 'circle';
      const pulse = (o.pulse === true);
      const vmin = o.vmin, vmax = o.vmax;
      const useHtml = (data.length < 2500 && (shape !== 'circle' || pulse));

      data.forEach(d => {
        let fc = o.fill || stroke;
        const val = (Array.isArray(d) ? d[2] : d.v);
        if (val != null && vmin != null && vmax != null) {
           const t = (val - vmin) / (vmax - vmin + 1e-12);
           fc = _ramp(t);
        }

        const lat = Array.isArray(d) ? d[0] : d.lat;
        const lon = Array.isArray(d) ? d[1] : d.lon;
        const tipTxt = d.tip ? d.tip : (d.v ? `v=${d.v}` : '');

        if (useHtml) {
           const cls = `gp-mkr ${shape} ${pulse?'gp-pulse':''}`;
           const html = `<div class="${cls}" style="--gp-r:${radius}px;--gp-fill:${fc};--gp-stroke:${stroke};--gp-op:${opacity};"></div>`;
           const icon = L.divIcon({ html: html, iconSize: [2*radius, 2*radius], iconAnchor: [radius, radius], className: '' });
           const m = L.marker([lat, lon], { icon: icon }).addTo(g);
           if(tipTxt) m.bindTooltip(tipTxt);
        } else {
           const m = L.circleMarker([lat, lon], {
             radius: radius, color: stroke, weight: 1, opacity: opacity,
             fillColor: fc, fillOpacity: opacity, renderer: canvas
           }).addTo(g);
           if(tipTxt) m.bindTooltip(tipTxt);
        }
      });
      return g;
    },

    // --- B. Contour (D3) ---
    _buildContour: function(data, opts) {
      const ContourLayer = L.Layer.extend({
        onAdd: function(map) {
          this._map = map;
          this._svg = L.svg(); 
          map.addLayer(this._svg); 
          this._root = d3.select(this._svg._rootGroup).append("g");
          this._update();
          map.on('zoomend', this._update, this);
        },
        onRemove: function(map) {
          this._root.remove();
          map.removeLayer(this._svg);
          map.off('zoomend', this._update, this);
        },
        _update: function() {
          const sel = this._root;
          sel.selectAll("*").remove();

          // 1. Project points
          const pts = data.map(d => {
            const p = map.latLngToLayerPoint(L.latLng(d.lat, d.lon));
            return { x: p.x, y: p.y, v: d.v };
          });

          const w = map.getSize().x;
          const h = map.getSize().y;
          
          // 2. Compute Density
          // FIX: Use Math.abs(d.v) because subsidence is negative.
          // contourDensity requires positive weights to generate standard "heat" contours.
          const metric =
            String(opts.metric || "value").toLowerCase();

          const wfun = (metric === "density")
            ? (d => 1.0)
            : (d => Math.abs(d.v));

          const contour = d3.contourDensity()
            .x(d => d.x).y(d => d.y)
            .weight(wfun)
            .size([w + 200, h + 200])
            .bandwidth(opts.bandwidth || 20)
            .thresholds(opts.steps || 15);

          const geoData = contour(pts);

          // 3. Draw
          const color = d3.scaleSequential(d3.interpolateViridis)
            .domain(d3.extent(geoData, d => d.value));

          // Style options
          const isFilled = (opts.filled !== false);
          const doLabels = (opts.labels === true);
          const pathGen = d3.geoPath(); // Create generator once

          // Draw Paths
          sel.selectAll("path")
            .data(geoData)
            .enter().append("path")
            .attr("d", pathGen)
            .attr("fill", d => isFilled ? color(d.value) : "none")
            .attr("stroke", d => isFilled ? "none" : color(d.value))
            .attr("stroke-width", isFilled ? 0 : 1.5)
            .attr("opacity", opts.opacity || 0.6);

          // Draw Labels (if enabled)
          if (doLabels) {
             sel.selectAll("text")
                .data(geoData)
                .enter().append("text")
                .attr("transform", d => {
                    const c = pathGen.centroid(d);
                    return `translate(${c[0]},${c[1]})`;
                })
                .text(d => d.value.toFixed(1))
                .attr("class", "gp-cont-label")
                .attr("fill", "#333")
                .attr("font-size", "10px");
          }
        }
      });
      return new ContourLayer();
    },

    // --- C. Hexbin (D3-hexbin) ---
    _buildHexbin: function(data, opts) {
      const o = opts || {};
      const pad = 80;

      if (!d3.hexbin) {
        // Fail safe: if CDN blocked, fall back to scatter.
        return this._buildScatter(data, opts);
      }

      function _agg(bin, metric) {
        const m = String(metric || "mean").toLowerCase();

        if (m === "count") return bin.length;

        let sum = 0.0;
        let mn = Infinity;
        let mx = -Infinity;

        for (let i = 0; i < bin.length; i++) {
          const v = Number(bin[i].v);
          if (!isFinite(v)) continue;
          sum += v;
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }

        if (m === "max") return mx;
        if (m === "min") return mn;

        // default mean
        return (bin.length > 0) ? (sum / bin.length) : NaN;
      }

      const HexLayer = L.Layer.extend({
        onAdd: function(map) {
          this._map = map;
          this._svg = L.svg();
          map.addLayer(this._svg);
          this._root = d3.select(this._svg._rootGroup)
            .append("g")
            .attr("class", "gp-hexbin");

          this._update();
          map.on("zoomend moveend resize", this._update, this);
        },

        onRemove: function(map) {
          map.off("zoomend moveend resize", this._update, this);
          this._root.remove();
          map.removeLayer(this._svg);
        },

        _update: function() {
          const sel = this._root;
          sel.selectAll("*").remove();

          if (!data || !data.length) return;

          const w = map.getSize().x;
          const h = map.getSize().y;

          sel.attr("transform", `translate(${-pad},${-pad})`);

          let pts = [];
          for (let i = 0; i < data.length; i++) {
            const d = data[i];
            const lat = Number(_ptLat(d));
            const lon = Number(_ptLon(d));
            const vv  = Number(_ptVal(d));
            if (!isFinite(lat) || !isFinite(lon) || !isFinite(vv)) continue;

            const p = map.latLngToLayerPoint(L.latLng(lat, lon));
            pts.push({ x: p.x + pad, y: p.y + pad, v: vv });
          }

          const maxPts = parseInt(o.max_pts || 4000, 10);
          pts = _sampleEven(pts, Math.max(300, maxPts));
          if (!pts.length) return;

          const ww = w + 2 * pad;
          const hh = h + 2 * pad;

          const r = Math.max(4, parseInt(o.gridsize || 30, 10));
          const metric = o.metric || "mean";
          const opacity = (o.opacity != null) ? Number(o.opacity) : 0.7;

          const hex = d3.hexbin()
            .x(d => d.x)
            .y(d => d.y)
            .radius(r)
            .extent([[0, 0], [ww, hh]]);

          const bins = hex(pts);
          if (!bins.length) return;

          // Compute bin values
          for (let i = 0; i < bins.length; i++) {
            bins[i].gp_v = _agg(bins[i], metric);
          }

          // Use shared min/max if provided; else from bins.
          let vmin = Infinity, vmax = -Infinity;
          if (_isFiniteNum(o.vmin) && _isFiniteNum(o.vmax)) {
            vmin = Number(o.vmin);
            vmax = Number(o.vmax);
          } else {
            for (let i = 0; i < bins.length; i++) {
              const v = bins[i].gp_v;
              if (!isFinite(v)) continue;
              if (v < vmin) vmin = v;
              if (v > vmax) vmax = v;
            }
          }
          if (!isFinite(vmin) || !isFinite(vmax)) return;

          const stroke = o.stroke || "#111827";

          sel.selectAll("path")
            .data(bins)
            .enter().append("path")
            .attr("d", hex.hexagon())
            .attr("transform", d => `translate(${d.x},${d.y})`)
            .attr("fill", d => _colorFromDomain(d.gp_v, vmin, vmax))
            .attr("stroke", stroke)
            .attr("stroke-width", 1)
            .attr("opacity", opacity)
            .append("title")
            .text(d => {
              const vv = d.gp_v;
              const lab = (metric === "count")
                ? `count=${d.length}`
                : `v=${vv.toFixed(3)}`;
              return lab;
            });
        }
      });

      return new HexLayer();
    },
  };

  // --- 4. Auxiliaries ---

  function _ensureLayerCtl() {
    if (layerCtl) return;
    layerCtl = L.control.layers(null, overlays, { collapsed:true }).addTo(map);
  }
  function _rebuildLayerCtl() {
    if (layerCtl) layerCtl.remove();
    layerCtl = null;
    _ensureLayerCtl();
  }

  // Legend
  let legend = null;
  function setLegend(opts) {
    if(!legend) {
        legend = L.control({ position:'bottomright' });
        legend.onAdd = () => L.DomUtil.create('div', 'gp-legend');
        legend.addTo(map);
    }
    const o = opts || {};
    legend.getContainer().innerHTML = `
      <div><b>${o.title || 'Value'}</b></div>
      <div class="mono">${o.mode || ''}</div>
      <div class="row">
        <div class="mono">${o.vmin != null ? o.vmin : ''}</div>
        <div class="bar"></div>
        <div class="mono">${o.vmax != null ? o.vmax : ''}</div>
      </div>
      <div class="mono">${o.unit || ''}</div>
    `;
  }

  // Legacy (Centroids, Links, Radar, Fit)
  function clearCentroids() {
    if (srcMarker) map.removeLayer(srcMarker);
    if (tgtMarker) map.removeLayer(tgtMarker);
    if (linkLine) map.removeLayer(linkLine);
    srcMarker=null; tgtMarker=null; linkLine=null;
  }
  function setCentroids(sn, sla, slo, tn, tla, tlo) {
    clearCentroids();
    const s=[sla,slo], t=[tla,tlo];
    srcMarker = L.circleMarker(s, {radius:7,color:'#2E3191',fillColor:'#2E3191',fillOpacity:0.9}).addTo(map).bindPopup(sn);
    tgtMarker = L.circleMarker(t, {radius:7,color:'#F28620',fillColor:'#F28620',fillOpacity:0.9}).addTo(map).bindPopup(tn);
    linkLine = L.polyline([s,t], {color:'#3399ff',weight:2,opacity:0.8}).addTo(map);
  }
  function clearLayers() {
    Object.keys(layers).forEach(k => LayerFactory.clear(k));
    Object.keys(linkLayers).forEach(k => clearLinks(k));
    Object.keys(radarLayers).forEach(k => clearRadar(k));
  }
  function fitLayers(ids) {
    const use = (ids && ids.length) ? ids : Object.keys(layers);
    const b = [];
    if(srcMarker) b.push(srcMarker.getLatLng());
    if(tgtMarker) b.push(tgtMarker.getLatLng());
    use.forEach(id => {
       const g = layers[id];
       if(g) { try{ const bb=g.getBounds(); if(bb.isValid()) {b.push(bb.getNorthEast()); b.push(bb.getSouthWest());} }catch(e){} }
    });
    if(!b.length) return;
    map.fitBounds(L.latLngBounds(b).pad(0.2));
  }

  // Links
  function _arrowIcon(ang, col) {
    const html = `<div class="gp-arrow" style="--gp-rot:${ang}deg;--gp-ac:${col};">▶</div>`;
    return L.divIcon({className:'', html:html, iconSize:[20,20], iconAnchor:[10,10]});
  }
  function _angleDeg(a, b) {
    const z = map.getZoom();
    const p1 = map.project(a, z);
    const p2 = map.project(b, z);
    return Math.atan2(p2.y - p1.y, p2.x - p1.x) * 180 / Math.PI;
  }
  function clearLinks(id) {
    const g = linkLayers[id];
    if (!g) return;
    map.removeLayer(g); delete overlays[g.__name||id]; delete linkLayers[id]; _rebuildLayerCtl();
  }
  function setLinks(id, name, links, opts) {
    clearLinks(id);
    const o = opts || {};
    const color = o.color || '#111827';
    const g = L.layerGroup(); g.__name = name || id;
    for (let i=0; i<links.length; i++) {
      const p = links[i];
      const a = L.latLng(p[0],p[1]), b = L.latLng(p[2],p[3]);
      const tip = p[5]||'', label = p[6]||((typeof p[4]==='number')?`${p[4].toFixed(2)} km`:'');
      const line = L.polyline([a,b], {color:color, weight:2, opacity:0.85}).addTo(g);
      if(tip) line.bindTooltip(String(tip));
      if(o.arrow!==false) {
        const ang = _angleDeg(a,b);
        const m = L.marker(b, {icon:_arrowIcon(ang,color), interactive:false}).addTo(g);
        m.__gp_link_a = a; m.__gp_link_b = b; m.__gp_link_color = color;
      }
      if(o.label===true && label) {
         const mid = L.latLng((a.lat+b.lat)/2, (a.lng+b.lng)/2);
         L.circleMarker(mid, {radius:1, opacity:0}).addTo(g).bindTooltip(String(label), {permanent:true, direction:'center', className:'gp-link-label'});
      }
    }
    g.addTo(map); linkLayers[id]=g; overlays[g.__name]=g; _rebuildLayerCtl();
  }
  map.on('zoomend moveend', function(){
    Object.keys(linkLayers).forEach(k=>{
       const g=linkLayers[k];
       if(g) g.eachLayer(l=>{ if(l.setIcon && l.__gp_link_a) l.setIcon(_arrowIcon(_angleDeg(l.__gp_link_a,l.__gp_link_b), l.__gp_link_color)); });
    });
  });

  // Radar
  function clearRadar(id) {
    if(radarTimers[id]) { clearInterval(radarTimers[id]); delete radarTimers[id]; }
    const g = radarLayers[id];
    if(g) { map.removeLayer(g); delete overlays[g.__name||id]; delete radarLayers[id]; }
    _rebuildLayerCtl();
  }
  function _offset(lat,lon,dx,dy) {
    const dlat = dy/111320; 
    const dlon = dx/(111320*Math.max(1e-9,Math.cos(lat*Math.PI/180)));
    return [lat+dlat, lon+dlon];
  }
  function _wedge(c,rM,ang,w) {
    const a0=(ang-w/2)*Math.PI/180, a1=(ang+w/2)*Math.PI/180;
    return [[c.lat,c.lng], _offset(c.lat,c.lng,rM*Math.cos(a0),rM*Math.sin(a0)), _offset(c.lat,c.lng,rM*Math.cos(a1),rM*Math.sin(a1))];
  }
  function setRadar(id, centers, opts) {
    clearRadar(id);
    const o=opts||{};
    const dwell=o.dwellMs||520, rM=(o.radiusKm||8.0)*1000, rings=o.rings||3;
    if(!centers||!centers.length) return;
    const g=L.layerGroup(); g.__name='Radar';
    let idx=0, baseT=Date.now(), ringL=[], wedge=null;
    function _draw(c) {
       ringL.forEach(l=>g.removeLayer(l)); ringL=[];
       const center=L.latLng(c[0],c[1]);
       for(let k=1; k<=rings; k++) {
          const cr=L.circle(center, {radius:rM*k/rings, color:'#00C853', weight:1, opacity:0.35, fillOpacity:0}).addTo(g);
          ringL.push(cr);
       }
       if(wedge) g.removeLayer(wedge);
       wedge=L.polygon(_wedge(center,rM,0,26), {color:'#00C853', weight:1, opacity:0.3, fillColor:'#00C853', fillOpacity:0.12}).addTo(g);
    }
    _draw(centers[0]);
    g.addTo(map); radarLayers[id]=g; overlays[g.__name]=g; _rebuildLayerCtl();
    radarTimers[id] = setInterval(()=>{
       const now=Date.now(), dt=now-baseT;
       if(dt>=dwell) { idx=(idx+1)%centers.length; baseT=now; _draw(centers[idx]); }
       else if(wedge) {
          const c=centers[idx];
          wedge.setLatLngs(_wedge(L.latLng(c[0],c[1]), rM, dt/dwell*360, 26));
       }
    }, 40);
  }

  // --- 5. Public API ---
  window.__GeoPriorMap = {
    setLayer: function(id, name, payload) {
       const kind = payload.kind || 'points';
       const data = payload.data || [];
       const opts = payload.opts || {};
       opts.name = name;
       LayerFactory.render(id, kind, data, opts);
    },
    clearLayer: function(id) { LayerFactory.clear(id); },
    clearLayers: clearLayers,
    setCentroids: setCentroids,
    clearCentroids: clearCentroids,
    setLinks: setLinks,
    clearLinks: clearLinks,
    setRadar: setRadar,
    clearRadar: clearRadar,
    fitLayers: fitLayers,
    setLegend: setLegend
  };

})();
</script>
</body>
</html>
"""