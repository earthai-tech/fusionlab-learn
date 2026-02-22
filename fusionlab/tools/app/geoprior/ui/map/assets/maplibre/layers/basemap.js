/* geoprior/ui/map/assets/maplibre/layers/basemap.js 
 *
 * Extracted from maplibre_html.py (basemap switching section).
 *
 * Requires:
 *   gp.map (maplibregl.Map)
 *   gp.__state._gpLoaded (bool)
 *
 * Calls (if provided):
 *   gp._ensurePointLayer()
 *   gp._ensureHotLayers()
 *   gp.showHotspots(on)
 *   gp.setPoints(...)
 *   gp.setHexbin(...)
 *   gp.setContours(...)
 *   gp.setHotspots(...)
 *   gp.setLegend(...)
 */

(function () {
  "use strict";

  const gp = window.__GeoPriorMap || (window.__GeoPriorMap = {});
  const S = gp.__state || (gp.__state = {});

  // Defaults / shared state
  if (!S.lastBasemap) S.lastBasemap = { p: "osm", s: "light", o: 1.0 };
  if (S._gpLoaded == null) S._gpLoaded = false;
  if (S.hotOn == null) S.hotOn = true;

  // MapLibre basemap constants (as in original)
  const GP_EMPTY_STYLE = { version: 8, sources: {}, layers: [] };
  const GP_OSM_VECTOR = "https://demotiles.maplibre.org/style.json";

  if (!S.gpActive) {
    S.gpActive = { kind: "style", style: GP_OSM_VECTOR, tiles: "" };
  }

  function _sameBasemap(a, b) {
    if (!a || !b) return false;
    if (a.kind !== b.kind) return false;
    if (a.kind === "style") {
      return String(a.style || "") === String(b.style || "");
    }
    if (a.kind === "raster") {
      return String(a.tiles || "") === String(b.tiles || "");
    }
    return false;
  }

  function _removeRasterBasemap() {
    const map = gp.map;
    if (!map) return;

    if (map.getLayer("gp_basemap")) {
      try { map.removeLayer("gp_basemap"); } catch (e) {}
    }
    if (map.getSource("gp_basemap")) {
      try { map.removeSource("gp_basemap"); } catch (e) {}
    }
  }

  function _applyRasterBasemap(spec, opacity) {
    const map = gp.map;
    if (!map) return;

    const op = (opacity != null) ? Number(opacity) : 1.0;
    const o = (isFinite(op)) ? op : 1.0;

    _removeRasterBasemap();

    map.addSource("gp_basemap", {
      type: "raster",
      tiles: spec.tiles,
      tileSize: 256,
      attribution: spec.att
    });

    const layer = {
      id: "gp_basemap",
      type: "raster",
      source: "gp_basemap",
      paint: { "raster-opacity": o }
    };

    const layers = (map.getStyle().layers || []);
    if (layers.length) map.addLayer(layer, layers[0].id);
    else map.addLayer(layer);
  }

  function _installOverlays() {
    try { gp._ensurePointLayer && gp._ensurePointLayer(); } catch (e) {}
    try { gp._ensureHotLayers && gp._ensureHotLayers(); } catch (e) {}
    try { gp.showHotspots && gp.showHotspots(S.hotOn); } catch (e) {}

    // replay cached state
    try {
      if (S.lastPoints && S.lastPoints.length) {
        if (S.lastMainKind === "hexbin_source") {
          gp.setHexbin && gp.setHexbin(S.lastPoints, S.lastPointOpts || {});
        } else if (S.lastMainKind === "contour_source") {
          gp.setContours && gp.setContours(S.lastPoints, S.lastPointOpts || {});
        } else {
          gp.setPoints && gp.setPoints(S.lastPoints, S.lastPointOpts || {});
        }
      }
    } catch (e) {}

    try {
      if (S.lastHotspots && S.lastHotspots.length) {
        gp.setHotspots && gp.setHotspots(S.lastHotspots, S.lastHotOpts || {});
      }
    } catch (e) {}

    try {
      if (S.lastLegend && gp.setLegend) {
        gp.setLegend(
          S.lastLegend.vmin,
          S.lastLegend.vmax,
          S.lastLegend.label,
          S.lastLegend.cmap,
          S.lastLegend.inv
        );
      }
    } catch (e) {}
  }

  function _basemapSpec(provider, style) {
    const p = String(provider || "osm").toLowerCase();
    const s = String(style || "light").toLowerCase();

    // Vector style for osm/light
    if (p === "osm" && s === "light") {
      return { kind: "style", style: GP_OSM_VECTOR };
    }

    // Raster defaults
    let tiles = [
      "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
      "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
      "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"
    ];
    let att = "© OpenStreetMap";

    if (p === "osm" && s === "dark") {
      tiles = [
        "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
      ];
      att = "© OpenStreetMap © CARTO";
    } else if (p === "osm" && s === "gray") {
      tiles = [
        "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "https://d.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
      ];
      att = "© OpenStreetMap © CARTO";
    } else if (p === "terrain") {
      tiles = [
        "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        "https://b.tile.opentopomap.org/{z}/{x}/{y}.png",
        "https://c.tile.opentopomap.org/{z}/{x}/{y}.png"
      ];
      att = "© OpenStreetMap © OpenTopoMap";
    } else if (p === "satellite") {
      tiles = [
        "https://server.arcgisonline.com/" +
        "ArcGIS/rest/services/World_Imagery/" +
        "MapServer/tile/{z}/{y}/{x}"
      ];
      att = "Tiles © Esri";
    }

    return { kind: "raster", tiles: tiles, att: att };
  }

  function setBasemap(provider, style, opacity) {
    const map = gp.map;

    S.lastBasemap = {
      p: String(provider || "osm"),
      s: String(style || "light"),
      o: (opacity != null) ? Number(opacity) : 1.0
    };

    if (!S._gpLoaded || !map) return;

    const spec = _basemapSpec(provider, style);

    // Vector style mode
    if (spec.kind === "style" && spec.style) {
      const want = { kind: "style", style: spec.style, tiles: "" };
      if (_sameBasemap(S.gpActive, want)) return;
      S.gpActive = want;

      gp.__ready = false;
      _removeRasterBasemap();
      map.setStyle(spec.style);

      map.once("style.load", function () {
        S._gpLoaded = true;
        _installOverlays();
        gp.__ready = true;
      });
      return;
    }

    // Raster mode (use an empty style so satellite/terrain are clean)
    const rasterKey = (spec.tiles || []).join("|");
    const wantR = { kind: "raster", style: "", tiles: rasterKey };

    gp.__ready = false;

    // If we are not already in raster mode, reset style first
    if (!S.gpActive || S.gpActive.kind !== "raster") {
      S.gpActive = wantR;
      map.setStyle(GP_EMPTY_STYLE);

      map.once("style.load", function () {
        S._gpLoaded = true;
        _applyRasterBasemap(spec, opacity);
        _installOverlays();
        gp.__ready = true;
      });
      return;
    }

    // Already raster mode: just swap tiles + opacity
    S.gpActive = wantR;
    _applyRasterBasemap(spec, opacity);
    _installOverlays();
    gp.__ready = true;
  }

  // Export
  gp.GP_EMPTY_STYLE = GP_EMPTY_STYLE;
  gp.GP_OSM_VECTOR = GP_OSM_VECTOR;
  gp._installOverlays = _installOverlays;
  gp._basemapSpec = _basemapSpec;
  gp.setBasemap = setBasemap;
})();
