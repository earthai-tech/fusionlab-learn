/* geoprior/ui/map/assets/leaflet/layers/basemap.js */
(function () {
  window.__GeoPriorLeaflet = window.__GeoPriorLeaflet || {};
  const NS = window.__GeoPriorLeaflet;

  function init(ctx) {
    const L = ctx.L;
    const map = ctx.map;

    // Match current behavior
    if (map && map.attributionControl) {
      map.attributionControl.setPrefix("");
    }

    const osmUrl =
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
    const osmAtt = "© OpenStreetMap";

    // Track tile layer so we can swap basemaps
    let tileLayer = L.tileLayer(osmUrl, {
      maxZoom: 19,
      attribution: osmAtt,
    }).addTo(map);

    function setBasemap(provider, style, opacity) {
      const a = window._normBasemapArgs(provider, style, opacity);
      const p = a.p;
      const s = a.s;

      let url = osmUrl;
      let att = osmAtt;
      let sub = "abc";
      let maxZ = 19;

      if (p === "osm" && s === "dark") {
        url =
          "https://{s}.basemaps.cartocdn.com/" +
          "dark_all/{z}/{x}/{y}{r}.png";
        att = "© OpenStreetMap © CARTO";
        sub = "abcd";
        maxZ = 19;
      } else if (p === "osm" && s === "gray") {
        url =
          "https://{s}.basemaps.cartocdn.com/" +
          "light_all/{z}/{x}/{y}{r}.png";
        att = "© OpenStreetMap © CARTO";
        sub = "abcd";
        maxZ = 19;
      } else if (p === "terrain") {
        url =
          "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png";
        att = "© OpenStreetMap © OpenTopoMap";
        sub = "abc";
        maxZ = 17;
      } else if (p === "satellite") {
        url =
          "https://server.arcgisonline.com/" +
          "ArcGIS/rest/services/World_Imagery/" +
          "MapServer/tile/{z}/{y}/{x}";
        att = "Tiles © Esri";
        sub = "";
        maxZ = 19;
      }

      const op = isFinite(a.o) ? a.o : 1.0;

      if (tileLayer) {
        try {
          map.removeLayer(tileLayer);
        } catch (e) {}
        tileLayer = null;
      }

      tileLayer = L.tileLayer(url, {
        maxZoom: maxZ,
        attribution: att,
        opacity: op,
        subdomains: sub,
      }).addTo(map);
    }

    // Expose to ctx
    ctx.setBasemap = setBasemap;

    // Keep refs (optional debugging)
    ctx.__basemap = {
      osmUrl: osmUrl,
      osmAtt: osmAtt,
      getTileLayer: function () {
        return tileLayer;
      },
    };
  }

  NS.basemap = { init: init };
})();
