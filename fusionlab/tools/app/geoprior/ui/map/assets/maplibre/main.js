/* geoprior/ui/map/assets/maplibre/main.js
 * MapLibre engine bootstrap for GeoPrior.
 *
 * Responsibilities:
 * - create map instance + base event wiring
 * - keep caches (points, hotspots, legend, basemap, kind)
 * - implement window.__GeoPriorMap methods by delegating
 *   to extracted layer modules under:
 *     window.__GeoPriorLayers.maplibre.<layer>
 *
 * Depends on:
 * - ../common/api_contract.js  (defines window.__GeoPriorMap stubs)
 * - ../common/bridge_qt.js     (defines window._emitPointClicked)
 * - ../common/colormap.js      (defines window._color)
 */

(function () {
  "use strict";

  var gp = window.__GeoPriorMap || {};
  gp.__engine = "maplibre";
  gp.__ready = false;
  gp.__failed = false;
  gp.__err = "";

  function _showFail(msg) {
    try {
      var box = document.getElementById("gp-fail");
      var lbl = document.getElementById("gp-fail-msg");
      if (lbl) lbl.textContent = String(msg || "");
      if (box) box.style.display = "block";
    } catch (e) {}
  }

  function _fail(msg) {
    gp.__failed = true;
    gp.__ready = false;
    gp.__err = String(msg || "");
    try {
      console.log("[maplibre] failed:", gp.__err);
    } catch (e) {}
    _showFail(gp.__err);
  }

  function _clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }

  function _getLayersRoot() {
    var r = window.__GeoPriorLayers;
    if (r && r.maplibre) return r.maplibre;
    return null;
  }

  function _layerObj(name) {
    var root = _getLayersRoot();
    if (!root) return null;
    // allow either root[name] or root.layers[name]
    if (root[name]) return root[name];
    if (root.layers && root.layers[name]) return root.layers[name];
    return null;
  }

  function _call(layerName, method, args) {
    var obj = _layerObj(layerName);
    if (!obj) return false;

    var fn = obj[method];
    if (typeof fn !== "function") return false;

    try {
      fn.apply(null, args || []);
      return true;
    } catch (e) {
      console.log("[maplibre] layer error:", layerName, method, e);
      return false;
    }
  }

  // ------------------------------------------------------------------
  // State (cached for reload safety + fitPoints)
  // ------------------------------------------------------------------

  var state = {
    map: null,
    loaded: false,

    lastPoints: [],
    lastPointOpts: {},
    lastMainKind: "points", // points|hexbin|contours

    lastHotspots: [],
    lastHotOpts: {},

    lastLegend: null,

    basemap: { p: "osm", s: "light", o: 1.0 },
  };

  // ------------------------------------------------------------------
  // Map creation
  // ------------------------------------------------------------------

  if (typeof maplibregl === "undefined") {
    _fail("maplibregl undefined");
    window.__GeoPriorMap = gp;
    return;
  }

  var map = null;
  try {
    map = new maplibregl.Map({
      container: "map",
      style: "https://demotiles.maplibre.org/style.json",
      center: [0, 0],
      zoom: 2,
      attributionControl: true,
    });
  } catch (e) {
    _fail(String(e));
    window.__GeoPriorMap = gp;
    return;
  }

  state.map = map;

  map.on("error", function (e) {
    try {
      if (e && e.error) _fail(String(e.error));
      else _fail(String(e));
    } catch (err) {
      _fail("unknown map error");
    }
  });

  try {
    map.addControl(
      new maplibregl.NavigationControl({ showCompass: false }),
      "top-right"
    );
  } catch (e) {}

  // Background click emits coordinate (helps selection UX)
  map.on("click", function (e) {
    if (!e || !e.lngLat) return;
    if (typeof window._emitPointClicked === "function") {
      window._emitPointClicked(e.lngLat.lng, e.lngLat.lat);
    }
  });

  // ------------------------------------------------------------------
  // Legend (engine-level, used by layers via gp.setLegend)
  // ------------------------------------------------------------------

  var legendEl = null;

  function _legendGradient(cmap, inv) {
    var cm = cmap || "viridis";
    var iv = !!inv;
    var c0 = window._color ? window._color(0, cm, iv) : "#000";
    var c1 = window._color ? window._color(1, cm, iv) : "#fff";
    return "linear-gradient(to top," + c0 + "," + c1 + ")";
  }

  function setLegend(vmin, vmax, label, cmap, inv) {
    try {
      if (legendEl && legendEl.parentNode) {
        legendEl.parentNode.removeChild(legendEl);
      }
    } catch (e) {}
    legendEl = null;

    var div = document.createElement("div");
    div.className = "gp-legend";

    var title = String(label || "Z");
    var g = _legendGradient(cmap, inv);

    div.innerHTML =
      '<div class="gp-legend__title">' +
      title +
      "</div>" +
      '<div class="gp-legend__row">' +
      '<div class="gp-legend__bar" style="background:' +
      g +
      ';"></div>' +
      '<div class="gp-legend__labels">' +
      "<div>" +
      Number(vmax).toFixed(3) +
      "</div>" +
      '<div style="height:62px;"></div>' +
      "<div>" +
      Number(vmin).toFixed(3) +
      "</div>" +
      "</div>" +
      "</div>";

    document.body.appendChild(div);
    legendEl = div;

    state.lastLegend = {
      vmin: Number(vmin),
      vmax: Number(vmax),
      label: title,
      cmap: String(cmap || "viridis"),
      inv: !!inv,
    };
  }

  // ------------------------------------------------------------------
  // Overlays install + replay
  // ------------------------------------------------------------------

  function _ensureCoreLayers() {
    // points + hotspots are the "always available" layers
    _call("points", "ensure", [map, state]);
    _call("hotspots", "ensure", [map, state]);
  }

  function _replayCached() {
    // main visualization
    try {
      if (state.lastPoints && state.lastPoints.length) {
        if (state.lastMainKind === "hexbin") {
          gp.setHexbin(state.lastPoints, state.lastPointOpts, true);
        } else if (state.lastMainKind === "contours") {
          gp.setContours(state.lastPoints, state.lastPointOpts, true);
        } else {
          gp.setPoints(state.lastPoints, state.lastPointOpts);
        }
      }
    } catch (e) {}

    // hotspots
    try {
      if (state.lastHotspots && state.lastHotspots.length) {
        gp.setHotspots(state.lastHotspots, state.lastHotOpts);
      }
    } catch (e) {}

    // vectors (if you cache them in your vectors layer, it can replay)
    try {
      _call("vectors", "replay", [map, state]);
    } catch (e) {}

    // legend
    try {
      if (state.lastLegend) {
        setLegend(
          state.lastLegend.vmin,
          state.lastLegend.vmax,
          state.lastLegend.label,
          state.lastLegend.cmap,
          state.lastLegend.inv
        );
      }
    } catch (e) {}
  }

  // ------------------------------------------------------------------
  // API methods (window.__GeoPriorMap)
  // ------------------------------------------------------------------

  function clearPoints() {
    state.lastPoints = [];
    state.lastPointOpts = {};
    state.lastMainKind = "points";

    _call("points", "clear", [map, state]);
    _call("hexbin", "clear", [map, state]);
    _call("contours", "clear", [map, state]);

    // also remove legend
    try {
      if (legendEl && legendEl.parentNode) {
        legendEl.parentNode.removeChild(legendEl);
      }
    } catch (e) {}
    legendEl = null;
    state.lastLegend = null;
  }

  function setPoints(points, opts) {
    state.lastPoints = points || [];
    state.lastPointOpts = opts || {};
    state.lastMainKind = "points";

    // let the extracted layer handle coloring + styling
    if (_call("points", "set", [map, state, state.lastPoints, state.lastPointOpts])) {
      return;
    }

    // fallback: some extractions named it setPoints(...)
    _call("points", "setPoints", [map, state, state.lastPoints, state.lastPointOpts]);
  }

  function setHexbin(points, opts, rerender) {
    state.lastPoints = points || state.lastPoints || [];
    state.lastPointOpts = opts || state.lastPointOpts || {};
    state.lastMainKind = "hexbin";

    // prefer render() when triggered by move/zoom
    if (rerender) {
      if (_call("hexbin", "render", [map, state])) return;
    }

    if (_call("hexbin", "set", [map, state, state.lastPoints, state.lastPointOpts])) {
      return;
    }
    _call("hexbin", "setHexbin", [map, state, state.lastPoints, state.lastPointOpts]);
  }

  function setContours(points, opts, rerender) {
    state.lastPoints = points || state.lastPoints || [];
    state.lastPointOpts = opts || state.lastPointOpts || {};
    state.lastMainKind = "contours";

    if (rerender) {
      if (_call("contours", "render", [map, state])) return;
    }

    if (
      _call("contours", "set", [map, state, state.lastPoints, state.lastPointOpts])
    ) {
      return;
    }
    _call("contours", "setContours", [
      map,
      state,
      state.lastPoints,
      state.lastPointOpts,
    ]);
  }

  function setHotspots(hs, opts) {
    state.lastHotspots = hs || [];
    state.lastHotOpts = opts || {};
    if (_call("hotspots", "set", [map, state, state.lastHotspots, state.lastHotOpts])) {
      return;
    }
    _call("hotspots", "setHotspots", [
      map,
      state,
      state.lastHotspots,
      state.lastHotOpts,
    ]);
  }

  function clearHotspots() {
    state.lastHotspots = [];
    state.lastHotOpts = {};
    _call("hotspots", "clear", [map, state]);
    _call("hotspots", "clearHotspots", [map, state]);
  }

  function showHotspots(on) {
    _call("hotspots", "show", [map, state, !!on]);
    _call("hotspots", "showHotspots", [map, state, !!on]);
  }

  function setVectors(vectors, opts) {
    _call("vectors", "set", [map, state, vectors || [], opts || {}]);
    _call("vectors", "setVectors", [map, state, vectors || [], opts || {}]);
  }

  function fitBounds(bounds, padding) {
    if (!bounds) return;

    var pad = Number(padding);
    if (!isFinite(pad)) pad = 40;

    // Accept either:
    // - [[minLon,minLat],[maxLon,maxLat]]
    // - [minLon,minLat,maxLon,maxLat]
    var b = bounds;
    if (Array.isArray(b) && b.length === 4) {
      b = [
        [Number(b[0]), Number(b[1])],
        [Number(b[2]), Number(b[3])],
      ];
    }

    try {
      map.fitBounds(b, { padding: pad });
    } catch (e) {}
  }

  function fitPoints() {
    var pts = [];

    for (var i = 0; i < (state.lastPoints || []).length; i++) {
      var p = state.lastPoints[i] || {};
      var lon = Number(p.lon);
      var lat = Number(p.lat);
      if (!isFinite(lon) || !isFinite(lat)) continue;
      pts.push([lon, lat]);
    }

    for (var j = 0; j < (state.lastHotspots || []).length; j++) {
      var h = state.lastHotspots[j] || {};
      var hlon = Number(h.lon);
      var hlat = Number(h.lat);
      if (!isFinite(hlon) || !isFinite(hlat)) continue;
      pts.push([hlon, hlat]);
    }

    if (!pts.length) return;

    var minX = pts[0][0],
      maxX = pts[0][0];
    var minY = pts[0][1],
      maxY = pts[0][1];

    for (var k = 1; k < pts.length; k++) {
      minX = Math.min(minX, pts[k][0]);
      maxX = Math.max(maxX, pts[k][0]);
      minY = Math.min(minY, pts[k][1]);
      maxY = Math.max(maxY, pts[k][1]);
    }

    fitBounds([[minX, minY], [maxX, maxY]], 40);
  }

  function zoomIn() {
    try {
      map.zoomIn();
    } catch (e) {}
  }

  function zoomOut() {
    try {
      map.zoomOut();
    } catch (e) {}
  }

  function setBasemap(provider, style, opacity) {
    state.basemap = {
      p: String(provider || "osm"),
      s: String(style || "light"),
      o: opacity != null ? Number(opacity) : 1.0,
    };

    // Delegate to basemap layer if available.
    // It should handle style swaps and call our overlays install.
    if (
      _call("basemap", "set", [
        map,
        state,
        state.basemap.p,
        state.basemap.s,
        state.basemap.o,
        _ensureCoreLayers,
        _replayCached,
      ])
    ) {
      return;
    }

    // fallback if basemap layer uses setBasemap signature
    _call("basemap", "setBasemap", [
      map,
      state,
      state.basemap.p,
      state.basemap.s,
      state.basemap.o,
      _ensureCoreLayers,
      _replayCached,
    ]);
  }

  // Re-render strategies that depend on viewport
  map.on("moveend", function () {
    if (!state.loaded) return;
    if (state.lastMainKind === "hexbin") {
      setHexbin(state.lastPoints, state.lastPointOpts, true);
    } else if (state.lastMainKind === "contours") {
      setContours(state.lastPoints, state.lastPointOpts, true);
    }
  });

  map.on("zoomend", function () {
    if (!state.loaded) return;
    if (state.lastMainKind === "hexbin") {
      setHexbin(state.lastPoints, state.lastPointOpts, true);
    } else if (state.lastMainKind === "contours") {
      setContours(state.lastPoints, state.lastPointOpts, true);
    }
  });

  // ------------------------------------------------------------------
  // Map load: install + replay
  // ------------------------------------------------------------------

  map.on("load", function () {
    state.loaded = true;

    try {
      _ensureCoreLayers();
    } catch (e) {}

    // Basemap layer can do special style/raster modes.
    // If absent, keep default style already set.
    try {
      setBasemap(state.basemap.p, state.basemap.s, state.basemap.o);
    } catch (e) {}

    gp.__ready = true;

    // Replay cached overlays if any (safe no-op at start)
    try {
      _replayCached();
    } catch (e) {}
  });

  // ------------------------------------------------------------------
  // Attach API
  // ------------------------------------------------------------------

  gp.setLegend = setLegend;

  gp.setPoints = setPoints;
  gp.setHexbin = setHexbin;
  gp.setContours = setContours;

  gp.clearPoints = clearPoints;
  gp.fitPoints = fitPoints;
  gp.fitBounds = fitBounds;

  gp.zoomIn = zoomIn;
  gp.zoomOut = zoomOut;

  gp.setBasemap = setBasemap;

  gp.setHotspots = setHotspots;
  gp.clearHotspots = clearHotspots;
  gp.showHotspots = showHotspots;

  gp.setVectors = setVectors;

  window.__GeoPriorMap = gp;
})();
