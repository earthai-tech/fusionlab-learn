/* geoprior/ui/map/assets/common/utils.js
 * Small helpers shared by engines.
 * Exposes:
 *   window._clamp(x,a,b)
 *   window._normBasemapArgs(provider, style, opacity)
 */
(function () {
  window._clamp =
    window._clamp ||
    function (x, a, b) {
      return Math.max(a, Math.min(b, x));
    };

  window._normBasemapArgs =
    window._normBasemapArgs ||
    function (provider, style, opacity) {
      var p0 = provider;
      var s0 = style;
      var o0 = opacity;

      // Allow passing a single object: {p/s/o} or
      // {provider/style/opacity} or {key}
      if (p0 && typeof p0 === "object") {
        o0 =
          p0.o != null
            ? p0.o
            : p0.opacity != null
            ? p0.opacity
            : o0;
        s0 =
          p0.s != null
            ? p0.s
            : p0.style != null
            ? p0.style
            : s0;
        p0 =
          p0.p != null
            ? p0.p
            : p0.provider != null
            ? p0.provider
            : p0.key;
      }

      var p = String(p0 || "osm").toLowerCase();
      var s = String(s0 || "light").toLowerCase();
      var o = o0 != null ? Number(o0) : 1.0;

      return { p: p, s: s, o: o };
    };
})();
