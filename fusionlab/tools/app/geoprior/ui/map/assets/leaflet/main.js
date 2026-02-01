/* geoprior/ui/map/assets/leaflet/main.js */
(function () {
  var gp = window.__GeoPriorMap || {};
  gp.__engine = "leaflet";
  gp.__ready = false;
  gp.__failed = false;
  gp.__err = "";
  window.__GeoPriorMap = gp;

  // Leaflet must exist
  if (typeof L === "undefined") {
    gp.__failed = true;
    gp.__err = "Leaflet not available.";
    console.log(gp.__err);
    return;
  }

  // Create map
  var map = L.map("map", {
    zoomControl: true,
  }).setView([0, 0], 2);

  // Build ctx for layer modules
  var ctx = {
    L: L,
    map: map,
  };

  // Init layer modules (safe guards)
  var NS = window.__GeoPriorLeaflet || {};

  function _safeInit(mod) {
    try {
      if (mod && mod.init) mod.init(ctx);
    } catch (e) {
      console.log("layer init error:", e);
    }
  }

  _safeInit(NS.basemap);
  _safeInit(NS.hotspots);
  _safeInit(NS.vectors);
  _safeInit(NS.points);
  _safeInit(NS.hexbin);
  _safeInit(NS.contours);

  // Background click selects (lon,lat) -> Python
  map.on("click", function (e) {
    if (!e || !e.latlng) return;
    if (window._emitPointClicked) {
      window._emitPointClicked(e.latlng.lng, e.latlng.lat);
    }
  });

  // Keep your old fitBounds behavior
  function fitBounds(coords) {
    // coords = [[lat1, lon1], [lat2, lon2]]
    if (!coords || coords.length !== 2) return;
    try {
      map.fitBounds(coords, { padding: [50, 50] });
    } catch (e) {
      console.log("fitBounds error:", e);
    }
  }

  function zoomIn() {
    map.zoomIn();
  }

  function zoomOut() {
    map.zoomOut();
  }

  // Wire functions onto the shared contract (exact names)
  gp.setPoints = ctx.setPoints || gp.setPoints;
  gp.clearPoints = ctx.clearPoints || gp.clearPoints;
  gp.fitPoints = ctx.fitPoints || gp.fitPoints;
  gp.fitBounds = fitBounds;

  gp.zoomIn = zoomIn;
  gp.zoomOut = zoomOut;

  gp.setLegend = ctx.setLegend || gp.setLegend;
  gp.setBasemap = ctx.setBasemap || gp.setBasemap;

  gp.setHotspots = ctx.setHotspots || gp.setHotspots;
  gp.clearHotspots = ctx.clearHotspots || gp.clearHotspots;
  gp.showHotspots = ctx.showHotspots || gp.showHotspots;

  gp.setVectors = ctx.setVectors || gp.setVectors;

  // Optional: future router (kept as stub for now)
  // gp.setLayer = gp.setLayer || function(){};

  gp.__ready = true;
})();
