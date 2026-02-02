/* geoprior/ui/map/assets/leaflet/layers/points.js */
(function () {
  window.__GeoPriorLeaflet = window.__GeoPriorLeaflet || {};
  const NS = window.__GeoPriorLeaflet;

  function init(ctx) {
    const L = ctx.L;
    const map = ctx.map;

    // -------------------------------------------------
    // Shared "main" layer for points/hexbin/contours.
    // Keep ctx.pointsLayer as an alias for compatibility.
    // -------------------------------------------------
    const layer =
      ctx.mainLayer ||
      ctx.layer ||
      ctx.pointsLayer ||
      L.layerGroup().addTo(map);

    ctx.mainLayer = layer;
    ctx.layer = layer;
    ctx.pointsLayer = layer;

    // Shared main-state (used by hexbin/contours rerender)
    ctx.__main =
      ctx.__main || {
        kind: "points",
        points: [],
        opts: {},
      };

    // Legend control (match your old `let legendCtl = null`)
    let legendCtl = null;

    function _clamp(x, a, b) {
      if (window._clamp) return window._clamp(x, a, b);
      return Math.max(a, Math.min(b, x));
    }

    function _color(t, cmap, inv) {
      // Prefer common colormap module if present
      const C = window.__GeoPriorCmap || {};
      const fn = C.color || C.getColor || null;

      if (typeof fn === "function") {
        try {
          return fn(t, cmap, inv);
        } catch (e) {}
      }

      // Fallback to legacy globals if you kept them
      if (window._color) {
        try {
          return window._color(t, cmap, inv);
        } catch (e) {}
      }

      // Final fallback: HSL gradient (like old inline code)
      const tt = inv ? 1 - t : t;
      const h = 240 * (1 - tt);
      return "hsl(" + h + ",80%,45%)";
    }

    function clearPoints() {
      layer.clearLayers();

      // IMPORTANT: remove contour overlay if it exists
      if (ctx.contourOverlay) {
        try {
          map.removeLayer(ctx.contourOverlay);
        } catch (e) {}
        ctx.contourOverlay = null;
      }

      // Reset main mode so zoom/move hooks don't rerender old modes
      ctx.__main.kind = "points";
      ctx.__main.points = [];
      ctx.__main.opts = {};
    }

    function setLegend(vmin, vmax, label, cmap, inv) {
      if (legendCtl) {
        try {
          map.removeControl(legendCtl);
        } catch (e) {}
        legendCtl = null;
      }

      const cm = cmap || "viridis";
      const iv = !!inv;

      legendCtl = L.control({ position: "bottomright" });
      legendCtl.onAdd = function () {
        const div = L.DomUtil.create("div");
        div.style.background = "rgba(255,255,255,0.85)";
        div.style.padding = "8px";
        div.style.borderRadius = "10px";
        div.style.fontFamily = "sans-serif";
        div.style.fontSize = "12px";

        const title = label || "Z";
        const g =
          "linear-gradient(to top," +
          _color(0, cm, iv) +
          "," +
          _color(1, cm, iv) +
          ")";

        div.innerHTML =
          '<div style="font-weight:600;">' +
          title +
          "</div>" +
          '<div style="display:flex;gap:8px;align-items:center;">' +
          '<div style="width:12px;height:90px;background:' +
          g +
          ';border-radius:6px;"></div>' +
          "<div>" +
          "<div>" +
          Number(vmax).toFixed(3) +
          "</div>" +
          '<div style="height:62px;"></div>' +
          "<div>" +
          Number(vmin).toFixed(3) +
          "</div>" +
          "</div></div>";

        return div;
      };
      legendCtl.addTo(map);
    }

    function setPoints(points, opts) {
      // Record main mode (so moving/zooming won't rerender hex/contours)
      ctx.__main.kind = "points";
      ctx.__main.points = points || [];
      ctx.__main.opts = opts || {};

      clearPoints();

      const p = points || [];
      const o = opts || {};

      const r = o.radius != null ? o.radius : 6;
      const op = o.opacity != null ? o.opacity : 0.9;
      const vmin = o.vmin != null ? o.vmin : null;
      const vmax = o.vmax != null ? o.vmax : null;

      const cmap = o.cmap || "viridis";
      const inv = !!o.invert;

      let lo = Infinity;
      let hi = -Infinity;

      for (let i = 0; i < p.length; i++) {
        const v = p[i].v;
        if (v == null || !isFinite(v)) continue;
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }

      const a = vmin != null ? vmin : lo;
      const b = vmax != null ? vmax : hi;

      for (let i = 0; i < p.length; i++) {
        const pt = p[i];
        const lat = pt.lat;
        const lon = pt.lon;
        const v = pt.v;

        if (!isFinite(lat) || !isFinite(lon)) continue;

        let t = 0.5;
        if (v != null && isFinite(v) && b > a) {
          t = (v - a) / (b - a);
        }
        t = _clamp(t, 0, 1);

        const col = _color(t, cmap, inv);

        const m = L.circleMarker([lat, lon], {
          radius: r,
          color: col,
          fillColor: col,
          fillOpacity: op,
          weight: 1,
        }).addTo(layer);

        // Emit (x,y) as (lon,lat) to match X/Y columns
        m.on("click", function () {
          if (window._emitPointClicked) {
            window._emitPointClicked(lon, lat);
          }
        });
      }

      if (o.showLegend) {
        setLegend(a, b, o.label || "Z", cmap, inv);
      }
    }

    function fitPoints() {
      const pts = [];

      layer.eachLayer(function (m) {
        if (m.getLatLng) pts.push(m.getLatLng());
        if (m.getBounds) {
          try {
            const b = m.getBounds();
            pts.push(b.getNorthWest());
            pts.push(b.getSouthEast());
          } catch (e) {}
        }
      });

      // include hotspots if present
      const hotLayer = ctx.hotLayer;
      if (hotLayer) {
        hotLayer.eachLayer(function (m) {
          if (m.getLatLng) pts.push(m.getLatLng());
          if (m.getBounds) {
            try {
              const b = m.getBounds();
              pts.push(b.getNorthWest());
              pts.push(b.getSouthEast());
            } catch (e) {}
          }
        });
      }

      // include contour overlay bounds if present
      if (ctx.contourOverlay && ctx.contourOverlay.getBounds) {
        try {
          const b = ctx.contourOverlay.getBounds();
          pts.push(b.getNorthWest());
          pts.push(b.getSouthEast());
        } catch (e) {}
      }

      if (!pts.length) return;
      map.fitBounds(L.latLngBounds(pts).pad(0.2));
    }

    // Expose to ctx
    ctx.clearPoints = clearPoints;
    ctx.setLegend = setLegend;
    ctx.setPoints = setPoints;
    ctx.fitPoints = fitPoints;

    // Optional: allow main.js to clear legend without points
    ctx.__legend = {
      clear: function () {
        if (legendCtl) {
          try {
            map.removeControl(legendCtl);
          } catch (e) {}
          legendCtl = null;
        }
      },
    };
  }

  NS.points = { init: init };
})();
