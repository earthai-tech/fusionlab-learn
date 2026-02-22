/* geoprior/ui/map/assets/common/colormap.js
 * Colormap helper.
 * Exposes:
 *   window._color(t, cmap, inv)
 */
(function () {
  window._color =
    window._color ||
    function (t, cmap, inv) {
      var tt = t;
      if (inv) tt = 1 - tt;

      // v0: keep HSL fallback (cmap reserved)
      var h = 240 * (1 - tt);
      return "hsl(" + h + ",80%,45%)";
    };
})();
