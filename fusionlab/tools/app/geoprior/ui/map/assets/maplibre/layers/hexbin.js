/* geoprior/ui/map/assets/maplibre/layers/hexbin.js
 *
 * Hexbin visualization strategy for MapLibre.
 *
 * Exposes:
 *   window.__GeoPriorMap.setHexbin(points, opts)
 *
 * Uses shared state:
 *   api.__state.map
 *   api.__state.__gpLoaded
 *   api.__state.lastPoints
 *   api.__state.lastPointOpts
 *   api.__state.lastMainKind
 *   api.__state.lastLegend
 */

(function () {
  const api = window.__GeoPriorMap || (window.__GeoPriorMap = {});
  const S = api.__state || (api.__state = {});

  function _clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }

  function _color(t, cmap, inv) {
    // Prefer shared colormap if you have one in common/colormap.js
    if (typeof api.__color === "function") {
      return api.__color(t, cmap, inv);
    }
    let tt = t;
    if (inv) tt = 1 - tt;
    const h = 240 * (1 - tt);
    return "hsl(" + h + ",80%,45%)";
  }

  function _map() {
    return S.map || null;
  }

  function _isReady() {
    return !!S.__gpLoaded && !!_map();
  }

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
    const map = _map();
    if (!map) return;

    if (!map.getSource("gp_hex")) {
      map.addSource("gp_hex", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }

    if (!map.getLayer("gp_hex_fill")) {
      map.addLayer({
        id: "gp_hex_fill",
        type: "fill",
        source: "gp_hex",
        paint: {
          "fill-color": ["get", "col"],
          "fill-opacity": ["get", "a"],
        },
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
          "line-width": 1,
        },
      });
    }
  }

  function _clearContoursOverlay() {
    const map = _map();
    if (!map) return;
    try {
      if (map.getLayer("gp_contour_layer")) {
        map.removeLayer("gp_contour_layer");
      }
    } catch (e) {}
    try {
      if (map.getSource("gp_contour_img")) {
        map.removeSource("gp_contour_img");
      }
    } catch (e) {}
  }

  function _renderHexbin() {
    const map = _map();
    const pts = S.lastPoints || [];
    const o = S.lastPointOpts || {};

    if (!_isReady()) return;

    // clear other overlays but keep caches
    _clearContoursOverlay();

    _ensureHexLayer();

    const gs = Math.max(5, Number(o.gridsize || 30));
    const metric = String(o.metric || "mean");
    const op = o.opacity != null ? Number(o.opacity) : 0.85;

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
        (y - (col & 1) * (sqrt3 * 0.5 * gs)) / (sqrt3 * gs),
      );
      const key = col + "," + row;

      if (!bins[key]) bins[key] = { col: col, row: row, vs: [] };
      bins[key].vs.push(v);
    }

    let lo = Infinity;
    let hi = -Infinity;

    for (const k in bins) {
      if (!Object.prototype.hasOwnProperty.call(bins, k)) continue;
      const it = bins[k];
      const vv = _agg(it.vs, metric);
      if (vv == null || !isFinite(vv)) continue;
      it.v = vv;
      lo = Math.min(lo, vv);
      hi = Math.max(hi, vv);
    }

    const a = o.vmin != null ? Number(o.vmin) : lo;
    const b = o.vmax != null ? Number(o.vmax) : hi;

    const feats = [];

    for (const k in bins) {
      if (!Object.prototype.hasOwnProperty.call(bins, k)) continue;
      const it = bins[k];
      if (it.v == null || !isFinite(it.v)) continue;

      const cx = it.col * 1.5 * gs;
      const cy =
        it.row * (sqrt3 * gs) + (it.col & 1) * (sqrt3 * 0.5 * gs);

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
        properties: { col: col, a: op },
      });
    }

    try {
      const s = map.getSource("gp_hex");
      if (s) s.setData({ type: "FeatureCollection", features: feats });
    } catch (e) {}

    if (o.showLegend && typeof api.setLegend === "function") {
      api.setLegend(a, b, o.label || "Z", o.cmap, !!o.invert);
      S.lastLegend = {
        vmin: a,
        vmax: b,
        label: o.label || "Z",
        cmap: o.cmap || "viridis",
        inv: !!o.invert,
      };
    }
  }

  function setHexbin(points, opts) {
    S.lastMainKind = "hexbin_source";
    S.lastPoints = points || [];
    S.lastPointOpts = opts || {};
    if (!_isReady()) return;
    _renderHexbin();
  }

  function clearHexbin() {
    const map = _map();
    if (!map) return;
    try {
      if (map.getLayer("gp_hex_fill")) map.removeLayer("gp_hex_fill");
    } catch (e) {}
    try {
      if (map.getLayer("gp_hex_line")) map.removeLayer("gp_hex_line");
    } catch (e) {}
    try {
      if (map.getSource("gp_hex")) map.removeSource("gp_hex");
    } catch (e) {}
  }

  // Install move/zoom re-render hooks once
  function _installHooks() {
    const map = _map();
    if (!map || S.__hexHooksInstalled) return;
    S.__hexHooksInstalled = true;

    map.on("moveend", function () {
      if (S.lastMainKind === "hexbin_source") _renderHexbin();
    });

    map.on("zoomend", function () {
      if (S.lastMainKind === "hexbin_source") _renderHexbin();
    });

    // Also recover after style changes (setStyle wipes sources/layers)
    map.on("style.load", function () {
      if (S.lastMainKind === "hexbin_source") {
        try {
          _renderHexbin();
        } catch (e) {}
      }
    });
  }

  // Patch clearPoints defensively (if points.js already clears, this is safe)
  const _prevClear = api.clearPoints;
  api.clearPoints = function () {
    try {
      if (typeof _prevClear === "function") _prevClear();
    } catch (e) {}
    try {
      clearHexbin();
    } catch (e) {}
  };

  // Export
  api.setHexbin = setHexbin;
  api.__clearHexbin = clearHexbin;

  // Attempt hook install now (and later if map appears)
  _installHooks();
})();
