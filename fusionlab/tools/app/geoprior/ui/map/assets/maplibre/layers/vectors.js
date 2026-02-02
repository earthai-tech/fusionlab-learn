/* geoprior/ui/map/assets/maplibre/layers/vectors.js
 *
 * Vector overlay for MapLibre.
 *
 * Exposes:
 *   window.__GeoPriorMap.setVectors(vectors, opts)
 *
 * Input vectors supported (per item):
 *   - { lon, lat, lon2, lat2 }  (explicit end)
 *   - { lon, lat, dx, dy }      (components)
 *   - { lon, lat, u, v }        (components)
 *
 * opts:
 *   - units: "deg" | "px" | "m" | "km"   (default: "deg")
 *   - color: string (default: "#111")
 *   - opacity: number (default: 0.9)
 *   - width: number (default: 2)
 *   - arrowheads: bool (default: true)
 *   - headPx: number (default: 10)
 *   - headAngleDeg: number (default: 25)
 */

(function () {
  const api = window.__GeoPriorMap || (window.__GeoPriorMap = {});
  const S = api.__state || (api.__state = {});

  function _map() {
    return S.map || null;
  }

  function _isReady() {
    return !!S.__gpLoaded && !!_map();
  }

  function _degPerMeterLat() {
    return 1.0 / 111320.0;
  }

  function _degPerMeterLon(lat) {
    const c = Math.cos((lat * Math.PI) / 180.0);
    const m = 111320.0 * Math.max(0.1, c);
    return 1.0 / m;
  }

  function _ensureVectorLayers() {
    const map = _map();
    if (!map) return;

    if (!map.getSource("gp_vectors")) {
      map.addSource("gp_vectors", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }

    if (!map.getLayer("gp_vectors_line")) {
      map.addLayer({
        id: "gp_vectors_line",
        type: "line",
        source: "gp_vectors",
        filter: ["==", ["get", "kind"], "shaft"],
        paint: {
          "line-color": ["get", "col"],
          "line-opacity": ["get", "a"],
          "line-width": ["get", "w"],
        },
      });
    }

    if (!map.getLayer("gp_vectors_head")) {
      map.addLayer({
        id: "gp_vectors_head",
        type: "line",
        source: "gp_vectors",
        filter: ["==", ["get", "kind"], "head"],
        paint: {
          "line-color": ["get", "col"],
          "line-opacity": ["get", "a"],
          "line-width": ["get", "w"],
        },
      });
    }
  }

  function _clearVectors() {
    const map = _map();
    if (!map) return;

    try {
      const s = map.getSource("gp_vectors");
      if (s) {
        s.setData({ type: "FeatureCollection", features: [] });
        return;
      }
    } catch (e) {}

    try {
      if (map.getLayer("gp_vectors_line")) map.removeLayer("gp_vectors_line");
    } catch (e) {}
    try {
      if (map.getLayer("gp_vectors_head")) map.removeLayer("gp_vectors_head");
    } catch (e) {}
    try {
      if (map.getSource("gp_vectors")) map.removeSource("gp_vectors");
    } catch (e) {}
  }

  function _vecEnd(lon, lat, item, opts) {
    const map = _map();
    if (!map) return null;

    const lon2 = item.lon2 != null ? Number(item.lon2) : null;
    const lat2 = item.lat2 != null ? Number(item.lat2) : null;
    if (isFinite(lon2) && isFinite(lat2)) {
      return [lon2, lat2];
    }

    // accept dx/dy or u/v
    let dx = item.dx != null ? Number(item.dx) : null;
    let dy = item.dy != null ? Number(item.dy) : null;
    if (!isFinite(dx) || !isFinite(dy)) {
      dx = item.u != null ? Number(item.u) : null;
      dy = item.v != null ? Number(item.v) : null;
    }
    if (!isFinite(dx) || !isFinite(dy)) return null;

    const units = String(opts.units || "deg").toLowerCase();

    // Pixel-based vectors: compute end by project/unproject
    if (units === "px" || units === "pixel" || units === "pixels") {
      const p = map.project([lon, lat]);
      const p2 = { x: p.x + dx, y: p.y + dy };
      const ll = map.unproject([p2.x, p2.y]);
      return [ll.lng, ll.lat];
    }

    // Meters / kilometers (approx)
    if (units === "m" || units === "meter" || units === "meters") {
      const dlon = dx * _degPerMeterLon(lat);
      const dlat = dy * _degPerMeterLat();
      return [lon + dlon, lat + dlat];
    }
    if (units === "km" || units === "kilometer" || units === "kilometers") {
      const dlon = (dx * 1000.0) * _degPerMeterLon(lat);
      const dlat = (dy * 1000.0) * _degPerMeterLat();
      return [lon + dlon, lat + dlat];
    }

    // Degrees default
    return [lon + dx, lat + dy];
  }

  function _addArrowHeads(features, lon1, lat1, lon2, lat2, col, a, w, opts) {
    const map = _map();
    if (!map) return;

    const want = opts.arrowheads != null ? !!opts.arrowheads : true;
    if (!want) return;

    const headPx = isFinite(Number(opts.headPx)) ? Number(opts.headPx) : 10;
    const angDeg = isFinite(Number(opts.headAngleDeg))
      ? Number(opts.headAngleDeg)
      : 25;
    const spread = (angDeg * Math.PI) / 180.0;

    const p1 = map.project([lon1, lat1]);
    const p2 = map.project([lon2, lat2]);
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;

    if (!isFinite(dx) || !isFinite(dy)) return;

    const base = Math.atan2(dy, dx);
    const a1 = base + Math.PI - spread;
    const a2 = base + Math.PI + spread;

    const w1 = { x: p2.x + headPx * Math.cos(a1), y: p2.y + headPx * Math.sin(a1) };
    const w2 = { x: p2.x + headPx * Math.cos(a2), y: p2.y + headPx * Math.sin(a2) };

    const ll1 = map.unproject([w1.x, w1.y]);
    const ll2 = map.unproject([w2.x, w2.y]);

    features.push({
      type: "Feature",
      geometry: {
        type: "LineString",
        coordinates: [
          [lon2, lat2],
          [ll1.lng, ll1.lat],
        ],
      },
      properties: { kind: "head", col: col, a: a, w: w },
    });

    features.push({
      type: "Feature",
      geometry: {
        type: "LineString",
        coordinates: [
          [lon2, lat2],
          [ll2.lng, ll2.lat],
        ],
      },
      properties: { kind: "head", col: col, a: a, w: w },
    });
  }

  function _renderVectors() {
    const map = _map();
    if (!_isReady()) return;

    const vecs = S.lastVectors || [];
    const o = S.lastVectorOpts || {};

    _ensureVectorLayers();

    const col0 = String(o.color || "#111");
    const a0 = o.opacity != null ? Number(o.opacity) : 0.9;
    const w0 = o.width != null ? Number(o.width) : 2;

    const feats = [];

    for (let i = 0; i < vecs.length; i++) {
      const it = vecs[i] || {};
      const lat = Number(it.lat);
      const lon = Number(it.lon);
      if (!isFinite(lat) || !isFinite(lon)) continue;

      const end = _vecEnd(lon, lat, it, o);
      if (!end) continue;

      const lon2 = Number(end[0]);
      const lat2 = Number(end[1]);
      if (!isFinite(lon2) || !isFinite(lat2)) continue;

      const col = it.col != null ? String(it.col) : col0;
      const a = it.a != null ? Number(it.a) : a0;
      const w = it.w != null ? Number(it.w) : w0;

      feats.push({
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: [
            [lon, lat],
            [lon2, lat2],
          ],
        },
        properties: { kind: "shaft", col: col, a: a, w: w },
      });

      _addArrowHeads(feats, lon, lat, lon2, lat2, col, a, w, o);
    }

    try {
      const s = map.getSource("gp_vectors");
      if (s) s.setData({ type: "FeatureCollection", features: feats });
    } catch (e) {}
  }

  function setVectors(vectors, opts) {
    S.lastVectors = vectors || [];
    S.lastVectorOpts = opts || {};
    if (!_isReady()) return;
    _renderVectors();
  }

  function clearVectors() {
    S.lastVectors = [];
    S.lastVectorOpts = {};
    _clearVectors();
  }

  function _installHooks() {
    const map = _map();
    if (!map || S.__vectorsHooksInstalled) return;
    S.__vectorsHooksInstalled = true;

    // Re-install after style resets and keep arrowheads sized in px
    map.on("style.load", function () {
      try {
        if (S.lastVectors && S.lastVectors.length) _renderVectors();
      } catch (e) {}
    });

    map.on("zoomend", function () {
      const o = S.lastVectorOpts || {};
      const units = String(o.units || "deg").toLowerCase();
      const wantHeads = o.arrowheads != null ? !!o.arrowheads : true;
      if (wantHeads || units === "px" || units === "pixel" || units === "pixels") {
        try {
          if (S.lastVectors && S.lastVectors.length) _renderVectors();
        } catch (e) {}
      }
    });
  }

  api.setVectors = setVectors;
  api.clearVectors = clearVectors;

  _installHooks();
})();
