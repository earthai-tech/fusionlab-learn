/* geoprior/ui/map/assets/leaflet/layers/contours.js
 * Stub for now: Leaflet version currently has no contour renderer.
 * Later we will implement using isolines -> GeoJSON -> L.geoJSON.
 */
(function () {
  window.__GeoPriorLeaflet = window.__GeoPriorLeaflet || {};
  const NS = window.__GeoPriorLeaflet;

  function init(ctx) {
    const L = ctx.L;
    const map = ctx.map;

    const contourLayer =
      ctx.contourLayer || L.layerGroup().addTo(map);
    ctx.contourLayer = contourLayer;

    function clearContours() {
      contourLayer.clearLayers();
    }

    function setContours(/* data, opts */) {
      clearContours();
      // later: draw isolines/isobands
    }

    ctx.clearContours = clearContours;
    ctx.setContours = setContours;
  }

  NS.contours = { init: init };
})();
