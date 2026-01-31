# geoprior/ui/view/factory.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import pandas as pd

from ...config.store import GeoConfigStore
from .keys import (
    K_PLOT_KIND,
    K_PLOT_OPACITY,
    K_HEX_GRIDSIZE,
    K_HEX_METRIC,
    K_FILTER_ENABLE,
    K_FILTER_V_MIN,
    K_FILTER_V_MAX,
    K_SPACE_MODE,
    K_CONTOUR_BANDWIDTH,
    K_CONTOUR_STEPS,
    K_CONTOUR_FILLED,
    K_CONTOUR_LABELS
)


class RenderPayload:
    """
    Standard output payload for the Map Controller.

    Decouples the 'how' (factory logic) from the 'what'
    (controller data).
    """

    def __init__(
        self,
        kind: str,
        data: Any,
        opts: Dict[str, Any],
    ) -> None:
        self.kind = kind
        self.data = data
        self.opts = opts


class ViewFactory:
    """
    Transform raw DataFrames into robust visual layers.

    Handles:
    - Filtering (Range & Spatial)
    - Binning (Hexbin prep)
    - Style prep (Opacity, Color)
    """

    def __init__(self, store: GeoConfigStore) -> None:
        self._s = store

    def build_layer(
        self,
        df: pd.DataFrame,
        layer_name: str,
    ) -> Optional[RenderPayload]:
        """
        Main entry point. Converts DF -> RenderPayload.
        """
        if df is None or df.empty:
            return None

        # 1. Robust Filtering (Value Range & Space)
        df_clean = self._apply_filters(df)
        if df_clean.empty:
            return None

        # 2. Determine Plot Type
        kind = str(
            self._s.get(K_PLOT_KIND, "scatter") or "scatter"
        )
        opacity = float(
            self._s.get(K_PLOT_OPACITY, 0.85) or 0.85
        )

        # 3. Generate Payload
        if kind == "hexbin":
            return self._build_hexbin(df_clean, opacity)
        
        if kind == "contour":  
            return self._build_contour(df_clean, opacity)
        
        # Default: Scatter
        return self._build_scatter(df_clean, opacity)

    # ---------------------------------------------------------
    # Filtering Logic
    # ---------------------------------------------------------
    def _apply_filters(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        out = df.copy()

        # A) Value Range Filtering
        if self._s.get(K_FILTER_ENABLE, False):
            vmin = float(
                self._s.get(K_FILTER_V_MIN, -9999)
            )
            vmax = float(
                self._s.get(K_FILTER_V_MAX, 9999)
            )
            # Filter rows where 'v' is within range
            if "v" in out.columns:
                out = out[
                    (out["v"] >= vmin) & (out["v"] <= vmax)
                ]

        # B) Spatial / Hotspot Filtering
        mode = str(
            self._s.get(K_SPACE_MODE, "all") or "all"
        )
        if mode == "hotspots_only":
            # 'sid' > 0 usually implies a ranked hotspot
            if "sid" in out.columns:
                out = out[out["sid"] > 0]

        return out

    # ---------------------------------------------------------
    # Renderers
    # ---------------------------------------------------------
    def _build_scatter(
        self,
        df: pd.DataFrame,
        opacity: float,
    ) -> RenderPayload:
        """
        Convert DataFrame to a list of dicts for Leaflet.

        Preserves 'sid' (rank) and 'tip' (tooltip) logic
        from the original controller implementation.
        """
        # Ensure numeric types for core coords
        lon = pd.to_numeric(df["lon"], errors="coerce")
        lat = pd.to_numeric(df["lat"], errors="coerce")
        val = pd.to_numeric(df["v"], errors="coerce")

        # Drop invalid rows immediately
        ok = lon.notna() & lat.notna() & val.notna()
        if not bool(ok.any()):
            return RenderPayload(
                "points",
                [],
                {"opacity": opacity},
            )

        # Identify optional columns
        has_sid = "sid" in df.columns
        has_tip = "tip" in df.columns

        # Select valid subset to iterate
        cols = ["lat", "lon", "v"]
        if has_sid:
            cols.append("sid")
        if has_tip:
            cols.append("tip")

        dd = df.loc[ok, cols]
        points: List[Dict[str, Any]] = []

        # Iterate efficiently using plain tuples
        for row in dd.itertuples(index=False, name=None):
            # core fields are always first 3
            la = float(row[0])
            lo = float(row[1])
            vv = float(row[2])

            sid = 0
            tip = ""

            # Extract optional fields based on presence
            if has_sid and has_tip:
                sid = int(row[3])
                tip = str(row[4] or "")
            elif has_sid:
                sid = int(row[3])
            elif has_tip:
                tip = str(row[3] or "")

            points.append(
                {
                    "lat": la,
                    "lon": lo,
                    "v": vv,
                    "sid": sid,
                    "tip": tip,
                }
            )

        return RenderPayload(
            kind="points",
            data=points,
            opts={
                "opacity": opacity,
                "radius": 6,
            },
        )

    def _build_hexbin(
        self,
        df: pd.DataFrame,
        opacity: float,
    ) -> RenderPayload:
        """
        Prepare data for hexbin aggregation.

        Currently returns points marked as 'hexbin_source'.
        The MapView (JS) handles the aggregation.
        """
        # Reuse scatter logic to get clean points list
        payload = self._build_scatter(df, opacity)

        gridsize = int(
            self._s.get(K_HEX_GRIDSIZE, 30) or 30
        )
        metric = str(
            self._s.get(K_HEX_METRIC, "mean") or "mean"
        )

        return RenderPayload(
            kind="hexbin_source",
            data=payload.data,
            opts={
                "opacity": opacity,
                "gridsize": gridsize,
                "metric": metric,
            },
        )


    # ---------------------------------------------------------
    # Contour Renderer
    # ---------------------------------------------------------
    def _build_contour(
        self,
        df: pd.DataFrame,
        opacity: float,
    ) -> RenderPayload:
        """
        Prepare data for density/value contouring.

        Returns points as 'contour_source'. The MapView (JS)
        should compute density/IDW and draw isobands.
        """
        # Reuse scatter to get clean, strictly typed points
        payload = self._build_scatter(df, opacity)

        # Fetch contour-specific settings
        bandwidth = float(
            self._s.get(K_CONTOUR_BANDWIDTH, 15.0) or 15.0
        )
        steps = int(
            self._s.get(K_CONTOUR_STEPS, 10) or 10
        )
        filled = bool(
            self._s.get(K_CONTOUR_FILLED, True)
        )
        labels = bool(self._s.get(K_CONTOUR_LABELS, False))

        return RenderPayload(
            kind="contour_source",
            data=payload.data,
            opts={
                "opacity": opacity,
                "bandwidth": bandwidth,
                "steps": steps,
                "filled": filled,
                "labels":labels,
                # Pass 'metric' if you want to switch between
                # pure density (heatmap) vs value interpolation
                "metric": "value", 
            },
        )
   