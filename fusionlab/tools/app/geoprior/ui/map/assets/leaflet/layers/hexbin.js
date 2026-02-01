/* geoprior/ui/map/assets/leaflet/layers/hexbin.js
 * Stub for now: Leaflet version currently has no hexbin renderer.
 * Later we will implement using polygons or a canvas layer.
 */
(function () {
  window.__GeoPriorLeaflet = window.__GeoPriorLeaflet || {};
  const NS = window.__GeoPriorLeaflet;

  function init(ctx) {
    const L = ctx.L;
    const map = ctx.map;

    const hexLayer =
      ctx.hexLayer || L.layerGroup().addTo(map);
    ctx.hexLayer = hexLayer;

    function clearHexbin() {
      hexLayer.clearLayers();
    }

    function setHexbin(/* data, opts */) {
      // no-op for now (keeps API calls safe)
      clearHexbin();
      // later: draw polygons/cells + legend integration
    }

    ctx.clearHexbin = clearHexbin;
    ctx.setHexbin = setHexbin;
  }

  NS.hexbin = { init: init };
})();
