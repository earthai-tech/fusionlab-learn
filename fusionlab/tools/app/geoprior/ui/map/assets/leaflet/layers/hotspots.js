/* geoprior/ui/map/assets/leaflet/layers/hotspots.js */
(function () {
  window.__GeoPriorLeaflet = window.__GeoPriorLeaflet || {};
  const NS = window.__GeoPriorLeaflet;

  function init(ctx) {
    const L = ctx.L;
    const map = ctx.map;

    const hotLayer =
      ctx.hotLayer || L.layerGroup().addTo(map);
    ctx.hotLayer = hotLayer;

    let hotOn = true;
    let hotTimers = [];

    function _stopHotTimers() {
      for (let i = 0; i < hotTimers.length; i++) {
        try {
          clearInterval(hotTimers[i]);
        } catch (e) {}
      }
      hotTimers = [];
    }

    function clearHotspots() {
      _stopHotTimers();
      hotLayer.clearLayers();
    }

    function _sevColor(sev) {
      const s = (sev || "high").toLowerCase();
      if (s === "critical") return "#d7263d";
      if (s === "high") return "#f18f01";
      if (s === "medium") return "#3f88c5";
      return "#6c757d";
    }

    // Not currently used by your renderer, but extracted as-is
    function _sevRadius(sev) {
      const s = (sev || "high").toLowerCase();
      if (s === "critical") return 18;
      if (s === "high") return 14;
      if (s === "medium") return 12;
      return 10;
    }

    function _pulseRing(ring, r0m, speed) {
      let r = r0m;
      let a = 0.9;

      let sp = Number(speed);
      if (!isFinite(sp) || sp <= 0) sp = 1.0;

      // Faster speed -> smaller dt
      let dt = 80 / sp;
      dt = Math.max(25, Math.min(200, dt));

      const t = setInterval(function () {
        r += r0m * 0.06;
        a -= 0.04;

        if (a <= 0.05) {
          r = r0m;
          a = 0.9;
        }

        try {
          ring.setRadius(r);
          ring.setStyle({ opacity: a });
        } catch (e) {}
      }, dt);

      hotTimers.push(t);
    }

    function showHotspots(on) {
      hotOn = !!on;
      if (hotOn) map.addLayer(hotLayer);
      else map.removeLayer(hotLayer);
    }

    function setHotspots(hs, opts) {
      clearHotspots();

      const h = hs || [];
      const o = opts || {};

      const want = o.show != null ? !!o.show : true;
      showHotspots(want);

      const style = String(o.style || "pulse").toLowerCase();
      const pulse = o.pulse != null ? !!o.pulse : true;
      const labels = o.labels != null ? !!o.labels : true;

      let baseKm = Number(o.ringKm);
      if (!isFinite(baseKm) || baseKm <= 0) baseKm = 0.8;

      let sp = Number(o.pulseSpeed);
      if (!isFinite(sp) || sp <= 0) sp = 1.0;

      function _sevMul(sev) {
        const s = (sev || "high").toLowerCase();
        if (s === "critical") return 1.6;
        if (s === "high") return 1.25;
        if (s === "medium") return 1.0;
        return 0.8;
      }

      for (let i = 0; i < h.length; i++) {
        const pt = h[i] || {};
        const lat = pt.lat;
        const lon = pt.lon;
        if (!isFinite(lat) || !isFinite(lon)) continue;

        const sev = pt.sev || "high";
        const col = _sevColor(sev);
        const mul = _sevMul(sev);

        const tip = pt.label || "Hotspot #" + (i + 1);

        const core = L.circleMarker([lat, lon], {
          radius: 5,
          color: col,
          fillColor: col,
          fillOpacity: 0.95,
          weight: 1,
        }).addTo(hotLayer);

        if (labels) core.bindTooltip(tip);
        core.on("click", function () {
          if (window._emitPointClicked) {
            window._emitPointClicked(lon, lat);
          }
        });

        // Style layer
        if (style === "glow") {
          const glow = L.circle([lat, lon], {
            radius: baseKm * 1000.0 * mul,
            color: col,
            opacity: 0.55,
            fillColor: col,
            fillOpacity: 0.12,
            weight: 1,
          }).addTo(hotLayer);

          if (labels) glow.bindTooltip(tip);
          glow.on("click", function () {
            if (window._emitPointClicked) {
              window._emitPointClicked(lon, lat);
            }
          });
          continue;
        }

        // Pulse ring (meters-based)
        const r0m = baseKm * 1000.0 * mul;
        const ring = L.circle([lat, lon], {
          radius: r0m,
          color: col,
          opacity: 0.9,
          fillOpacity: 0.0,
          weight: 2,
        }).addTo(hotLayer);

        if (labels) ring.bindTooltip(tip);
        ring.on("click", function () {
          if (window._emitPointClicked) {
            window._emitPointClicked(lon, lat);
          }
        });

        if (pulse) _pulseRing(ring, r0m, sp);
      }
    }

    // Expose to ctx
    ctx.clearHotspots = clearHotspots;
    ctx.showHotspots = showHotspots;
    ctx.setHotspots = setHotspots;

    // Keep helpers accessible (optional)
    ctx.__hot = {
      sevColor: _sevColor,
      sevRadius: _sevRadius,
    };
  }

  NS.hotspots = { init: init };
})();
