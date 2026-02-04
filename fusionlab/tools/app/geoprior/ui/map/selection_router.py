# geoprior/ui/map/selection_router.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import pandas as pd
from PyQt5.QtCore import QObject


from .keys import (
    MAP_DF_POINTS,
    MAP_SELECT_IDS,
    MAP_SELECT_MODE,
    MAP_SELECT_OPEN,
)


class SelectionRouter(QObject):
    """
    Small glue: canvas events -> store selection keys.

    Later we will also:
    - ask brain cache for time-series slices
    - request highlight on the map canvas
    """

    def __init__(
        self,
        *,
        store: Any,
        canvas: Any,
        controller: Optional[Any] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._c = canvas
        self._brain = controller

        self._connect_canvas()

    # -------------------------
    # Wiring
    # -------------------------
    def _connect_canvas(self) -> None:
        if self._c is None:
            return

        try:
            self._c.point_clicked_id.connect(
                self._on_point_id
            )
        except Exception:
            pass

        try:
            self._c.point_clicked.connect(
                self._on_point_lonlat
            )
        except Exception:
            pass

        # We will add this signal on the bridge later.
        # Connect only if it exists.
        sig = getattr(self._c, "group_bbox", None)
        if sig is not None:
            try:
                sig.connect(self._on_group_bbox)
            except Exception:
                pass
    # ------------------------- # Group selection (bbox)
    def _on_point_lonlat(self, lon: float, lat: float) -> None:
        sid = self._nearest_id(float(lon), float(lat))
        if sid is None:
            return
        self._on_point_id(int(sid))

    def _nearest_id(self, lon: float, lat: float) -> Optional[int]:
        df = self._df_points()
        if df is None or df.empty:
            return None

        if "lon" not in df.columns:
            return None
        if "lat" not in df.columns:
            return None
        if "sample_idx" not in df.columns:
            return None

        try:
            x = pd.to_numeric(df["lon"], errors="coerce").to_numpy()
            y = pd.to_numeric(df["lat"], errors="coerce").to_numpy()
            sid = pd.to_numeric(
                df["sample_idx"],
                errors="coerce",
            ).to_numpy()
        except Exception:
            return None

        m = ~(pd.isna(x) | pd.isna(y) | pd.isna(sid))
        if not bool(m.any()):
            return None

        dx = x[m] - float(lon)
        dy = y[m] - float(lat)
        d2 = dx * dx + dy * dy

        try:
            j = int(d2.argmin())
            out = sid[m][j]
            if pd.isna(out):
                return None
            return int(out)
        except Exception:
            return None

    # -------------------------
    # Public helpers
    # -------------------------
    def clear(self) -> None:
        with self._s.batch():
            self._s.set(MAP_SELECT_IDS, [])
            self._s.set(MAP_SELECT_MODE, "off")
            self._s.set(MAP_SELECT_OPEN, False)

    # -------------------------
    # Point selection
    # -------------------------
    def _on_point_id(self, sid: int) -> None:
        try:
            i = int(sid)
        except Exception:
            return

        with self._s.batch():
            self._s.set(MAP_SELECT_MODE, "point")
            self._s.set(MAP_SELECT_IDS, [i])
            self._s.set(MAP_SELECT_OPEN, True)

    # -------------------------
    # Group selection (bbox)
    # -------------------------
    def _on_group_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> None:
        ids = self._ids_from_bbox(
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        )
        if not ids:
            self.clear()
            return

        with self._s.batch():
            self._s.set(MAP_SELECT_MODE, "group")
            self._s.set(MAP_SELECT_IDS, ids)
            self._s.set(MAP_SELECT_OPEN, True)

    def _ids_from_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> List[int]:
        df = self._df_points()
        if df is None or df.empty:
            return []

        a = float(min(min_lon, max_lon))
        b = float(max(min_lon, max_lon))
        c = float(min(min_lat, max_lat))
        d = float(max(min_lat, max_lat))

        if "lon" not in df.columns:
            return []
        if "lat" not in df.columns:
            return []
        if "sample_idx" not in df.columns:
            return []

        m = (
            (df["lon"] >= a)
            & (df["lon"] <= b)
            & (df["lat"] >= c)
            & (df["lat"] <= d)
        )
        out = df.loc[m, "sample_idx"]
        try:
            return [int(x) for x in out.tolist()]
        except Exception:
            return []

    def _df_points(self) -> Optional[pd.DataFrame]:
        if self._brain is not None:
            try:
                b = self._brain.get_bundle()
                df = getattr(b, "df_points", None)
                if df is not None:
                    return df
            except Exception:
                pass

        try:
            return self._s.get(MAP_DF_POINTS, None)
        except Exception:
            return None
