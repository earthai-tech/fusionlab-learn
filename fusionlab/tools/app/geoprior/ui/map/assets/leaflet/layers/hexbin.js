/* geoprior/ui/map/assets/leaflet/layers/hexbin.js
 * Leaflet hexbin renderer (polygons in screen space).
 */
(function () {
  window.__GeoPriorLeaflet = window.__GeoPriorLeaflet || {};
  var NS = window.__GeoPriorLeaflet;

  function init(ctx) {
    var L = ctx.L;
    var map = ctx.map;

    // Shared "main layer" used by points/hexbin polygons.
    var layer =
      ctx.mainLayer ||
      ctx.layer ||
      L.layerGroup().addTo(map);

    ctx.mainLayer = layer;
    ctx.layer = layer;

    // Shared main-state for re-render on move/zoom.
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
      // Prefer shared colormap module if present.
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

      // Fallback: HSL gradient (same as old code).
      var tt = inv ? 1 - t : t;
      var h = 240 * (1 - tt);
      return "hsl(" + h + ",80%,45%)";
    }

    function _agg(vals, metric) {
      var m = String(metric || "mean").toLowerCase();
      if (!vals || !vals.length) return null;

      if (m === "count") return vals.length;
      if (m === "max") return Math.max.apply(null, vals);
      if (m === "min") return Math.min.apply(null, vals);

      if (m === "sum") {
        var ss = 0;
        for (var i = 0; i < vals.length; i++) ss += vals[i];
        return ss;
      }

      // mean
      var s = 0;
      for (var j = 0; j < vals.length; j++) s += vals[j];
      return s / vals.length;
    }

    function _clearMainVisuals() {
      layer.clearLayers();

      // If contours overlay exists, remove it too.
      if (ctx.contourOverlay) {
        try {
          map.removeLayer(ctx.contourOverlay);
        } catch (e) {}
        ctx.contourOverlay = null;
      }
    }

    function _renderHexbin() {
      var pts = ctx.__main.points || [];
      var o = ctx.__main.opts || {};

      _clearMainVisuals();

      var gs = Math.max(5, Number(o.gridsize || 30));
      var metric = String(o.metric || "mean");
      var op =
        o.opacity != null
          ? Number(o.opacity)
          : 0.85;

      if (!pts.length) return;

      var bins = {};
      var sqrt3 = Math.sqrt(3);

      for (var i = 0; i < pts.length; i++) {
        var p = pts[i];
        var q = map.latLngToContainerPoint([p.lat, p.lon]);
        var x = q.x;
        var y = q.y;

        var col = Math.round(x / (1.5 * gs));
        var row = Math.round(
          (y - (col & 1) * (sqrt3 * 0.5 * gs)) /
            (sqrt3 * gs),
        );

        var key = col + "," + row;
        if (!bins[key]) {
          bins[key] = { col: col, row: row, vs: [] };
        }
        bins[key].vs.push(Number(p.v));
      }

      var lo = Infinity;
      var hi = -Infinity;

      for (var k in bins) {
        if (!bins.hasOwnProperty(k)) continue;
        var it = bins[k];
        var vv = _agg(it.vs, metric);
        if (vv == null) continue;
        it.v = vv;
        lo = Math.min(lo, vv);
        hi = Math.max(hi, vv);
      }

      var a =
        o.vmin != null
          ? Number(o.vmin)
          : lo;

      var b =
        o.vmax != null
          ? Number(o.vmax)
          : hi;

      for (var k2 in bins) {
        if (!bins.hasOwnProperty(k2)) continue;
        var it2 = bins[k2];
        if (it2.v == null) continue;

        var cx = it2.col * 1.5 * gs;
        var cy =
          it2.row * (sqrt3 * gs) +
          (it2.col & 1) * (sqrt3 * 0.5 * gs);

        var t = _clamp(
          (it2.v - a) / ((b - a) || 1),
          0,
          1,
        );

        var col2 = _color(
          t,
          o.cmap,
          !!o.invert,
        );

        var latlngs = [];
        for (var j2 = 0; j2 < 6; j2++) {
          var ang = (Math.PI / 3) * j2;
          var px = cx + gs * Math.cos(ang);
          var py = cy + gs * Math.sin(ang);
          var ll = map.containerPointToLatLng(
            L.point(px, py),
          );
          latlngs.push(ll);
        }

        L.polygon(latlngs, {
          color: col2,
          weight: 1,
          fillColor: col2,
          fillOpacity: op,
          interactive: false,
        }).addTo(layer);
      }

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

    function setHexbin(points, opts) {
      ctx.__main.kind = "hexbin_source";
      ctx.__main.points = points || [];
      ctx.__main.opts = opts || {};
      _renderHexbin();
    }

    function clearHexbin() {
      if (ctx.__main.kind === "hexbin_source") {
        ctx.__main.kind = "points";
        ctx.__main.points = [];
        ctx.__main.opts = {};
      }
      _clearMainVisuals();
    }

    // Expose for main.js wiring + re-render hook
    ctx.setHexbin = setHexbin;
    ctx.clearHexbin = clearHexbin;
    ctx._renderHexbin = _renderHexbin;

    // Attach one shared re-render hook.
    if (!ctx.__mainHooked) {
      ctx.__mainHooked = true;
      map.on("zoomend moveend", function () {
        if (!ctx.__main) return;
        if (ctx.__main.kind === "hexbin_source") {
          if (ctx._renderHexbin) ctx._renderHexbin();
        }
        if (ctx.__main.kind === "contour_source") {
          if (ctx._renderContours) ctx._renderContours();
        }
      });
    }
  }

  NS.hexbin = { init: init };
})();
