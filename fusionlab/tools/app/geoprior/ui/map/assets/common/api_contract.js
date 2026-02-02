/* geoprior/ui/map/assets/common/api_contract.js
 * Defines a stable window.__GeoPriorMap contract (stubs first).
 * Must be loaded early (before engine main.js calls it).
 */
(function () {
  function _noop() {}

  var gp = window.__GeoPriorMap || {};

  // Metadata (helps probing / fallback logic)
  if (gp.__engine == null) gp.__engine = "unknown";
  if (gp.__ready == null) gp.__ready = false;
  if (gp.__failed == null) gp.__failed = false;
  if (gp.__err == null) gp.__err = "";

  gp.__debug =
    gp.__debug ||
    function () {
      return {
        engine: String(gp.__engine || ""),
        ready: !!gp.__ready,
        failed: !!gp.__failed,
        err: String(gp.__err || ""),
      };
    };

  // Core API (stubs)
  if (!gp.setPoints) gp.setPoints = _noop;
  if (!gp.setHexbin) gp.setHexbin = _noop;
  if (!gp.setContours) gp.setContours = _noop;

  if (!gp.clearPoints) gp.clearPoints = _noop;
  if (!gp.fitPoints) gp.fitPoints = _noop;
  if (!gp.fitBounds) gp.fitBounds = _noop;

  if (!gp.zoomIn) gp.zoomIn = _noop;
  if (!gp.zoomOut) gp.zoomOut = _noop;

  if (!gp.setLegend) gp.setLegend = _noop;
  if (!gp.setBasemap) gp.setBasemap = _noop;

  if (!gp.setHotspots) gp.setHotspots = _noop;
  if (!gp.clearHotspots) gp.clearHotspots = _noop;
  if (!gp.showHotspots) gp.showHotspots = _noop;

  if (!gp.setVectors) gp.setVectors = _noop;

  // Optional future expansion (safe stubs now)
  if (!gp.setLayer) gp.setLayer = _noop;
  if (!gp.clearLayer) gp.clearLayer = _noop;
  if (!gp.clearLayers) gp.clearLayers = _noop;

  window.__GeoPriorMap = gp;
})();
