/* geoprior/ui/map/assets/leaflet/layers/contours.js
 * Leaflet contours renderer (simple kernel-smoothed heat overlay).
 */
(function () {
  window.__GeoPriorLeaflet = window.__GeoPriorLeaflet || {};
  var NS = window.__GeoPriorLeaflet;

  function init(ctx) {
    var L = ctx.L;
    var map = ctx.map;

    // Shared main layer used by points/hexbin.
    var layer =
      ctx.mainLayer ||
      ctx.layer ||
      L.layerGroup().addTo(map);

    ctx.mainLayer = layer;
    ctx.layer = layer;

    ctx.__main =
      ctx.__main || {
        kind: "points",
        points: [],
        opts: {},
      };

    function _clamp(x, a, b) {
      return Math.max(a, Math.min(b, x));
    }

    function _color(t, cmap, inv) {
      var C = window.__GeoPriorCmap || {};
      var fn =
        C.color ||
        C.getColor ||
        null;

      if (typeof fn === "function") {
        try {
          return fn(t, cmap, inv);
        } catch (e) {}
      }

      var tt = inv ? 1 - t : t;
      var h = 240 * (1 - tt);
      return "hsl(" + h + ",80%,45%)";
    }

    function _clearMainVisuals() {
      layer.clearLayers();

      if (ctx.contourOverlay) {
        try {
          map.removeLayer(ctx.contourOverlay);
        } catch (e) {}
        ctx.contourOverlay = null;
      }
    }

    function _renderContours() {
      var pts = ctx.__main.points || [];
      var o = ctx.__main.opts || {};

      _clearMainVisuals();

      var op =
        o.opacity != null
          ? Number(o.opacity)
          : 0.7;

      // Accept "bandwidth" or "steps" (future UI keys)
      var bwRaw =
        o.bandwidth != null
          ? Number(o.bandwidth)
          : Number(o.steps || 15);

      var bw = Math.max(4, bwRaw);

      // Step in pixels (keep old behavior)
      var step = Math.max(6, Math.min(80, bw));

      var size = map.getSize();
      var w = size.x;
      var h = size.y;

      if (!pts.length || !w || !h) return;

      var proj = [];
      var lo = Infinity;
      var hi = -Infinity;

      for (var i = 0; i < pts.length; i++) {
        var p = pts[i];
        var q = map.latLngToContainerPoint([p.lat, p.lon]);
        var v = Number(p.v);
        proj.push({ x: q.x, y: q.y, v: v });
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }

      var a =
        o.vmin != null
          ? Number(o.vmin)
          : lo;

      var b =
        o.vmax != null
          ? Number(o.vmax)
          : hi;

      var cvs = document.createElement("canvas");
      cvs.width = w;
      cvs.height = h;

      var g = cvs.getContext("2d");
      if (!g) return;

      g.globalAlpha = op;

      var sig2 = 2 * step * step;

      for (var yy = 0; yy < h; yy += step) {
        for (var xx = 0; xx < w; xx += step) {
          var num = 0;
          var den = 0;

          for (var k = 0; k < proj.length; k++) {
            var pp = proj[k];
            var dx = xx - pp.x;
            var dy = yy - pp.y;
            var d2 = dx * dx + dy * dy;
            var w0 = Math.exp(-d2 / sig2);
            num += w0 * pp.v;
            den += w0;
          }

          if (den <= 0) continue;

          var vv = num / den;
          var t = _clamp((vv - a) / ((b - a) || 1), 0, 1);
          g.fillStyle = _color(t, o.cmap, !!o.invert);
          g.fillRect(xx, yy, step, step);
        }
      }

      var url = cvs.toDataURL("image/png");
      var bounds = map.getBounds();

      ctx.contourOverlay = L.imageOverlay(url, bounds, {
        opacity: 1.0,
        interactive: false,
      }).addTo(map);

      if (o.showLegend && typeof ctx.setLegend === "function") {
        ctx.setLegend(
          a,
          b,
          o.label || "Z",
          o.cmap,
          !!o.invert,
        );
      }
    }

    function setContours(points, opts) {
      ctx.__main.kind = "contour_source";
      ctx.__main.points = points || [];
      ctx.__main.opts = opts || {};
      _renderContours();
    }

    function clearContours() {
      if (ctx.__main.kind === "contour_source") {
        ctx.__main.kind = "points";
        ctx.__main.points = [];
        ctx.__main.opts = {};
      }
      _clearMainVisuals();
    }

    ctx.setContours = setContours;
    ctx.clearContours = clearContours;
    ctx._renderContours = _renderContours;

    // NOTE: The shared zoom/move re-render hook is
    // installed by hexbin.js (guarded by ctx.__mainHooked).
  }

  NS.contours = { init: init };
})();
