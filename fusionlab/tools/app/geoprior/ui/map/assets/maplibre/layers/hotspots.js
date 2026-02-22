/* geoprior/ui/map/assets/maplibre/layers/hotspots.js */
/* hotspots.js
 *
 * Extracted from maplibre_html.py (hotspots layers + pulse rings).
 *
 * Requires:
 *   gp.map
 *   gp._emitPointClicked(lon,lat)
 *
 * Shares:
 *   gp.__state.{lastHotspots,lastHotOpts,hotOn,_gpLoaded}
 */

(function () {
  "use strict";

  const gp = window.__GeoPriorMap || (window.__GeoPriorMap = {});
  const S = gp.__state || (gp.__state = {});

  if (!S.lastHotspots) S.lastHotspots = [];
  if (!S.lastHotOpts) S.lastHotOpts = {};
  if (S.hotOn == null) S.hotOn = true;

  let hotTimer = null;

  function _stopHotTimer() {
    if (hotTimer != null) {
      try { clearInterval(hotTimer); } catch (e) {}
    }
    hotTimer = null;
  }

  function _ensureHotLayers() {
    const map = gp.map;
    if (!map) return;

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
          "circle-radius": ["get", "r"],
          "circle-color": ["get", "col"],
          "circle-opacity": 0.0,
          "circle-stroke-color": ["get", "col"],
          "circle-stroke-opacity": ["get", "a"],
          "circle-stroke-width": 2
        }
      });
    }

    if (!map.getLayer("gp_hot_core_layer")) {
      map.addLayer({
        id: "gp_hot_core_layer",
        type: "circle",
        source: "gp_hot_core",
        paint: {
          "circle-radius": 5,
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
          if (gp._emitPointClicked) {
            gp._emitPointClicked(Number(c[0]), Number(c[1]));
          }
        } catch (err) {}
      });
    }
  }

  function showHotspots(on) {
    const map = gp.map;
    if (!map) return;

    S.hotOn = !!on;
    const v = S.hotOn ? "visible" : "none";

    if (map.getLayer("gp_hot_core_layer")) {
      map.setLayoutProperty("gp_hot_core_layer", "visibility", v);
    }
    if (map.getLayer("gp_hot_ring_layer")) {
      map.setLayoutProperty("gp_hot_ring_layer", "visibility", v);
    }
  }

  function clearHotspots() {
    const map = gp.map;

    _stopHotTimer();
    S.lastHotspots = [];
    S.lastHotOpts = {};

    try {
      const s1 = map && map.getSource("gp_hot_core");
      if (s1) {
        s1.setData({ type: "FeatureCollection", features: [] });
      }
    } catch (e) {}

    try {
      const s2 = map && map.getSource("gp_hot_ring");
      if (s2) {
        s2.setData({ type: "FeatureCollection", features: [] });
      }
    } catch (e) {}
  }

  function _kmToPx(km, lat) {
    const map = gp.map;
    if (!map) return 24;

    const z = map.getZoom();
    const c = Math.cos((lat * Math.PI) / 180.0);
    const mpp = (156543.03392 * c) / Math.pow(2, z);
    const m = Number(km) * 1000.0;
    if (!isFinite(mpp) || mpp <= 0) return 24;
    const px = m / mpp;
    return Math.max(6, Math.min(220, px));
  }

  function _sevColor(sev) {
    const s = (sev || "high").toLowerCase();
    if (s === "critical") return "#d7263d";
    if (s === "high") return "#f18f01";
    if (s === "medium") return "#3f88c5";
    return "#6c757d";
  }

  function _pulseRings(rings, speed) {
    const map = gp.map;
    if (!map) return;

    _stopHotTimer();

    let sp = Number(speed);
    if (!isFinite(sp) || sp <= 0) sp = 1.0;

    let dt = 80 / sp;
    dt = Math.max(25, Math.min(200, dt));

    hotTimer = setInterval(function () {
      for (let i = 0; i < rings.length; i++) {
        const r = rings[i];
        r.r += (r.r0 * 0.06);
        r.a -= 0.04;
        if (r.a <= 0.05) {
          r.r = r.r0;
          r.a = 0.9;
        }
      }

      // push update
      const feats = [];
      for (let i = 0; i < rings.length; i++) {
        const rr = rings[i];
        feats.push({
          type: "Feature",
          geometry: rr.g,
          properties: { col: rr.col, r: rr.r, a: rr.a }
        });
      }

      try {
        const s2 = map.getSource("gp_hot_ring");
        if (s2) {
          s2.setData({
            type: "FeatureCollection",
            features: feats
          });
        }
      } catch (e) {}
    }, dt);
  }

  function _setHotspotsInternal(hs, opts) {
    const map = gp.map;

    clearHotspots();

    const h = hs || [];
    const o = opts || {};

    if (!S._gpLoaded || !map) {
      S.lastHotspots = h;
      S.lastHotOpts = o;
      return;
    }

    const want = (o.show != null) ? !!o.show : true;
    showHotspots(want);

    const style = String(o.style || "pulse").toLowerCase();
    const pulse = (o.pulse != null) ? !!o.pulse : true;
    const labels = (o.labels != null) ? !!o.labels : true;

    let baseKm = Number(o.ringKm);
    if (!isFinite(baseKm) || baseKm <= 0) baseKm = 0.8;

    let sp = Number(o.pulseSpeed);
    if (!isFinite(sp) || sp <= 0) sp = 1.0;

    // build cores + rings
    const cores = [];
    const rings = [];

    for (let i = 0; i < h.length; i++) {
      const pt = h[i] || {};
      const lat = Number(pt.lat);
      const lon = Number(pt.lon);
      if (!isFinite(lat) || !isFinite(lon)) continue;

      const sev = pt.sev || "high";
      const col = _sevColor(sev);

      const g = { type: "Point", coordinates: [lon, lat] };

      cores.push({
        type: "Feature",
        geometry: g,
        properties: { col: col }
      });

      const r0 = _kmToPx(baseKm, lat);
      const ring = {
        g: g,
        col: col,
        r0: r0,
        r: r0,
        a: (style === "glow") ? 0.55 : 0.9
      };
      rings.push(ring);
    }

    // cores update
    try {
      const s1 = map.getSource("gp_hot_core");
      if (s1) {
        s1.setData({
          type: "FeatureCollection",
          features: cores
        });
      }
    } catch (e) {}

    // rings update once (and maybe animate)
    const ringFeats = [];
    for (let i = 0; i < rings.length; i++) {
      const rr = rings[i];
      ringFeats.push({
        type: "Feature",
        geometry: rr.g,
        properties: { col: rr.col, r: rr.r, a: rr.a }
      });
    }

    try {
      const s2 = map.getSource("gp_hot_ring");
      if (s2) {
        s2.setData({
          type: "FeatureCollection",
          features: ringFeats
        });
      }
    } catch (e) {}

    if (style !== "glow" && pulse) {
      _pulseRings(rings, sp);
    }

    // tooltips (optional; minimal)
    if (labels) {
      // MapLibre raster style + basic UI: skip for now
      // (kept to align with Leaflet API).
    }
  }

  // Public API wrapper (kept exactly like your end-of-file attach logic)
  function setHotspots(hs, o) {
    S.lastHotspots = hs || [];
    S.lastHotOpts = o || {};
    _setHotspotsInternal(hs, o);
  }

  // Export
  gp._ensureHotLayers = _ensureHotLayers;
  gp.showHotspots = showHotspots;
  gp.clearHotspots = clearHotspots;
  gp.setHotspots = setHotspots;
})();

