/* geoprior/ui/map/assets/maplibre/layers/contours.js
 *
 * Contours visualization strategy for MapLibre
 * (raster image overlay generated from point interpolation).
 *
 * Exposes:
 *   window.__GeoPriorMap.setContours(points, opts)
 */

(function () {
  const api = window.__GeoPriorMap || (window.__GeoPriorMap = {});
  const S = api.__state || (api.__state = {});

  function _clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }

  function _color(t, cmap, inv) {
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

  function _clearHexbinOverlay() {
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

  function _renderContours() {
    const map = _map();
    const pts = S.lastPoints || [];
    const o = S.lastPointOpts || {};

    if (!_isReady()) return;

    // clear hexbin
    _clearHexbinOverlay();

    const op = o.opacity != null ? Number(o.opacity) : 0.7;
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

    const a = o.vmin != null ? Number(o.vmin) : lo;
    const b = o.vmax != null ? Number(o.vmax) : hi;

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

    _clearContoursOverlay();

    map.addSource("gp_contour_img", {
      type: "image",
      url: url,
      coordinates: [
        [nw.lng, nw.lat],
        [ne.lng, ne.lat],
        [se.lng, se.lat],
        [sw.lng, sw.lat],
      ],
    });

    map.addLayer({
      id: "gp_contour_layer",
      type: "raster",
      source: "gp_contour_img",
      paint: { "raster-opacity": op },
    });

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

  function setContours(points, opts) {
    S.lastMainKind = "contour_source";
    S.lastPoints = points || [];
    S.lastPointOpts = opts || {};
    if (!_isReady()) return;
    _renderContours();
  }

  function clearContours() {
    _clearContoursOverlay();
  }

  function _installHooks() {
    const map = _map();
    if (!map || S.__contourHooksInstalled) return;
    S.__contourHooksInstalled = true;

    map.on("moveend", function () {
      if (S.lastMainKind === "contour_source") _renderContours();
    });

    map.on("zoomend", function () {
      if (S.lastMainKind === "contour_source") _renderContours();
    });

    // recover after setStyle
    map.on("style.load", function () {
      if (S.lastMainKind === "contour_source") {
        try {
          _renderContours();
        } catch (e) {}
      }
    });
  }

  // Patch clearPoints defensively
  const _prevClear = api.clearPoints;
  api.clearPoints = function () {
    try {
      if (typeof _prevClear === "function") _prevClear();
    } catch (e) {}
    try {
      clearContours();
    } catch (e) {}
  };

  api.setContours = setContours;
  api.__clearContours = clearContours;

  _installHooks();
})();
