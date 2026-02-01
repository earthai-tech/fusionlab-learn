/* geoprior/ui/map/assets/common/bridge_qt.js
 * Qt WebChannel bridge helper.
 * Exposes:
 *   window.__GeoPriorQt.bridge
 *   window._emitPointClicked(lon, lat)
 */
(function () {
  if (!window.__GeoPriorQt) {
    window.__GeoPriorQt = {
      bridge: null,
    };
  }

  function _initBridge() {
    try {
      if (
        typeof QWebChannel !== "undefined" &&
        typeof qt !== "undefined" &&
        qt.webChannelTransport
      ) {
        new QWebChannel(qt.webChannelTransport, function (channel) {
          window.__GeoPriorQt.bridge =
            (channel && channel.objects && channel.objects.bridge) || null;
        });
      }
    } catch (e) {
      // keep silent; bridge is optional
    }
  }

  // Keep the same function name your current code uses
  window._emitPointClicked = function (lon, lat) {
    var b = window.__GeoPriorQt.bridge;
    if (b && b.pointClicked) {
      try {
        b.pointClicked(lon, lat);
      } catch (e) {}
    }
  };

  _initBridge();
})();
