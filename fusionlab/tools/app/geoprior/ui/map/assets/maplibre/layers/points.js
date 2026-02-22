/* geoprior/ui/map/assets/maplibre/layers/points.js */
/* points.js
 *
 * Extracted from maplibre_html.py (points + legend + fitPoints + clearPoints).
 *
 * Requires:
 *   gp.map
 *   gp._emitPointClicked(lon,lat)
 *
 * Shares:
 *   gp.__state.{lastPoints,lastPointOpts,lastMainKind,lastLegend,lastHotspots}
 */

(function () {
  "use strict";

  const gp = window.__GeoPriorMap || (window.__GeoPriorMap = {});
  const S = gp.__state || (gp.__state = {});

  if (!S.lastPoints) S.lastPoints = [];
  if (!S.lastPointOpts) S.lastPointOpts = {};
  if (!S.lastMainKind) S.lastMainKind = "points";
  if (S.lastLegend == null) S.lastLegend = null;

  function _clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }

  function _color(t, cmap, inv) {
    let tt = t;
    if (inv) tt = 1 - tt;
    const h = 240 * (1 - tt);
    return "hsl(" + h + ",80%,45%)";
  }

  let legendEl = null;

  function setLegend(vmin, vmax, label, cmap, inv) {
    if (legendEl && legendEl.parentNode) {
      try { legendEl.parentNode.removeChild(legendEl); }
      catch (e) {}
    }

    const cm = cmap || "viridis";
    const iv = !!inv;
    const title = (label || "Z");
    const g = (
      "linear-gradient(to top," +
      _color(0, cm, iv) + "," +
      _color(1, cm, iv) + ")"
    );

    const div = document.createElement("div");
    div.className = "gp-legend";
    div.innerHTML = (
      "<div style=\"font-weight:600;\">" + title + "</div>" +
      "<div style=\"display:flex;gap:8px;align-items:center;\">" +
      "<div style=\"width:12px;height:90px;background:" + g +
      ";border-radius:6px;\"></div>" +
      "<div>" +
      "<div>" + Number(vmax).toFixed(3) + "</div>" +
      "<div style=\"height:62px;\"></div>" +
      "<div>" + Number(vmin).toFixed(3) + "</div>" +
      "</div></div>"
    );

    document.body.appendChild(div);
    legendEl = div;
  }

  // Points: GPU-friendly circle layer via GeoJSON
  function _ensurePointLayer() {
    const map = gp.map;
    if (!map) return;

    if (!map.getSource("gp_points")) {
      map.addSource("gp_points", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] }
      });
    }

    if (!map.getLayer("gp_points_layer")) {
      map.addLayer({
        id: "gp_points_layer",
        type: "circle",
        source: "gp_points",
        paint: {
          "circle-radius": 6,
          "circle-color": ["get", "col"],
          "circle-opacity": 0.9,
          "circle-stroke-color": ["get", "col"],
          "circle-stroke-width": 1
        }
      });

      map.on("click", "gp_points_layer", function (e) {
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

  function setPoints(points, opts) {
    const map = gp.map;
    const p = points || [];
    const o = opts || {};

    S.lastPoints = p;
    S.lastPointOpts = o;
    S.lastMainKind = "points";
    if (!S._gpLoaded || !map) return;

    _ensurePointLayer();

    const r = (o.radius != null) ? Number(o.radius) : 6;
    const op = (o.opacity != null) ? Number(o.opacity) : 0.9;
    const vmin = (o.vmin != null) ? Number(o.vmin) : null;
    const vmax = (o.vmax != null) ? Number(o.vmax) : null;
    const cmap = o.cmap || "viridis";
    const inv = !!o.invert;

    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < p.length; i++) {
      const vv = p[i].v;
      if (vv == null || !isFinite(vv)) continue;
      lo = Math.min(lo, vv);
      hi = Math.max(hi, vv);
    }

    const a = (vmin != null) ? vmin : lo;
    const b = (vmax != null) ? vmax : hi;

    const feats = [];
    for (let i = 0; i < p.length; i++) {
      const pt = p[i];
      const lat = Number(pt.lat);
      const lon = Number(pt.lon);
      const vv = pt.v;
      if (!isFinite(lat) || !isFinite(lon)) continue;

      let t = 0.5;
      if (vv != null && isFinite(vv) && b > a) {
        t = (vv - a) / (b - a);
      }
      t = _clamp(t, 0, 1);
      const col = _color(t, cmap, inv);

      feats.push({
        type: "Feature",
        geometry: { type: "Point", coordinates: [lon, lat] },
        properties: { col: col }
      });
    }

    try {
      const src = map.getSource("gp_points");
      if (src) {
        src.setData({
          type: "FeatureCollection",
          features: feats
        });
      }
    } catch (e) {}

    try {
      map.setPaintProperty("gp_points_layer", "circle-radius", r);
      map.setPaintProperty("gp_points_layer", "circle-opacity", op);
    } catch (e) {}

    if (o.showLegend) {
      setLegend(a, b, o.label || "Z", cmap, inv);
      S.lastLegend = {
        vmin: a, vmax: b, label: o.label || "Z",
        cmap: cmap, inv: inv
      };
    }
  }

  function clearPoints() {
    const map = gp.map;

    S.lastPoints = [];
    S.lastPointOpts = {};
    S.lastMainKind = "points";

    // points
    try {
      const src = map && map.getSource("gp_points");
      if (src) {
        src.setData({
          type: "FeatureCollection",
          features: []
        });
      }
    } catch (e) {}

    // hexbin (kept exactly as original clearPoints)
    try {
      if (map && map.getLayer("gp_hex_fill")) map.removeLayer("gp_hex_fill");
    } catch (e) {}
    try {
      if (map && map.getLayer("gp_hex_line")) map.removeLayer("gp_hex_line");
    } catch (e) {}
    try {
      if (map && map.getSource("gp_hex")) map.removeSource("gp_hex");
    } catch (e) {}

    // contours (image overlay)
    try {
      if (map && map.getLayer("gp_contour_layer")) {
        map.removeLayer("gp_contour_layer");
      }
    } catch (e) {}
    try {
      if (map && map.getSource("gp_contour_img")) {
        map.removeSource("gp_contour_img");
      }
    } catch (e) {}
  }

  function fitPoints() {
    const map = gp.map;
    if (!map) return;

    const pts = [];

    for (let i = 0; i < (S.lastPoints || []).length; i++) {
      const pt = S.lastPoints[i];
      const lat = Number(pt.lat);
      const lon = Number(pt.lon);
      if (!isFinite(lat) || !isFinite(lon)) continue;
      pts.push([lon, lat]);
    }

    for (let i = 0; i < (S.lastHotspots || []).length; i++) {
      const ht = S.lastHotspots[i] || {};
      const lat = Number(ht.lat);
      const lon = Number(ht.lon);
      if (!isFinite(lat) || !isFinite(lon)) continue;
      pts.push([lon, lat]);
    }

    if (!pts.length) return;

    let minX = pts[0][0], maxX = pts[0][0];
    let minY = pts[0][1], maxY = pts[0][1];

    for (let i = 1; i < pts.length; i++) {
      minX = Math.min(minX, pts[i][0]);
      maxX = Math.max(maxX, pts[i][0]);
      minY = Math.min(minY, pts[i][1]);
      maxY = Math.max(maxY, pts[i][1]);
    }

    map.fitBounds([[minX, minY], [maxX, maxY]], { padding: 40 });
  }

  // Export (public + shared helpers for later hexbin/contours)
  gp._clamp = gp._clamp || _clamp;
  gp._color = gp._color || _color;

  gp.setLegend = setLegend;
  gp._ensurePointLayer = _ensurePointLayer;

  gp.setPoints = setPoints;
  gp.clearPoints = clearPoints;
  gp.fitPoints = fitPoints;
})();
