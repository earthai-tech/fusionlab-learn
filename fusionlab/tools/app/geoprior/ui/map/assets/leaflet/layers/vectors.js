/* geoprior/ui/map/assets/leaflet/layers/vectors.js */
(function () {
  window.__GeoPriorLeaflet = window.__GeoPriorLeaflet || {};
  const NS = window.__GeoPriorLeaflet;

  function init(ctx) {
    const L = ctx.L;
    const map = ctx.map;

    const vectorLayer =
      ctx.vectorLayer || L.layerGroup().addTo(map);
    ctx.vectorLayer = vectorLayer;

    function setVectors(vecs) {
      vectorLayer.clearLayers();

      const v = vecs || [];
      for (let i = 0; i < v.length; i++) {
        const d = v[i];
        if (!isFinite(d.lat) || !isFinite(d.lon)) continue;

        const icon = L.divIcon({
          className: "prop-arrow-container",
          html:
            '<div class="prop-arrow" style="transform: rotate(' +
            (d.angle - 90) +
            'deg);">➤</div>',
          iconSize: [20, 20],
          iconAnchor: [10, 10],
        });

        L.marker([d.lat, d.lon], {
          icon: icon,
          interactive: false,
        }).addTo(vectorLayer);
      }
    }

    ctx.setVectors = setVectors;
  }

  NS.vectors = { init: init };
})();
