# geoprior/ui/map/selection_router.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
from PyQt5.QtCore import QObject

from .keys import (
    MAP_DF_POINTS,
    MAP_SELECT_IDS,
    MAP_SELECT_MODE,
    MAP_SELECT_OPEN,
    K_PROP_ENABLED, 
    MAP_PROP_DF_POINTS
)


class SelectionRouter(QObject):
    """
    Canvas events -> store selection keys.

    IMPORTANT
    ---------
    This router must NOT change MAP_SELECT_MODE.

    MAP_SELECT_MODE is owned by the UI toolbar
    (MapTab._set_select_mode). Otherwise you get
    the exact bug you reported: deselect -> click
    -> router forces mode back to 'point'.
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
            self._c.point_clicked.connect(
                self._on_point_lonlat
            )
        except Exception:
            pass

        try:
            self._c.point_clicked_id.connect(
                self._on_point_id
            )
        except Exception:
            pass

        sig = getattr(self._c, "group_bbox", None)
        if sig is not None:
            try:
                sig.connect(self._on_group_bbox)
            except Exception:
                pass

    def _mode(self) -> str:
        if self._s is None:
            return "off"
        m = self._s.get(MAP_SELECT_MODE, "off")
        return str(m or "off").strip().lower()

    def clear(self) -> None:
        """
        Clear selection ids + close the drawer.

        IMPORTANT:
        Do NOT change MAP_SELECT_MODE here.
        Mode is owned by the toolbar.
        """
        if self._s is None:
            return
        with self._s.batch():
            self._s.set(MAP_SELECT_IDS, [])
            self._s.set(MAP_SELECT_OPEN, False)

    def _on_point_lonlat(self, lon: float, lat: float) -> None:
        # Only act if user is in point mode.
        if self._mode() != "point":
            return

        # if self.controller is None:
        #     return
        # b = self.controller.get_bundle()
        if self._brain is None:
            return
        b = self._brain.get_bundle()
        
        if b is None or b.df_points is None:
            return

        df = b.df_points
        if df.empty:
            return

        try:
            dx = df["lon"].to_numpy(dtype="float64") - float(lon)
            dy = df["lat"].to_numpy(dtype="float64") - float(lat)
            j = int((dx * dx + dy * dy).argmin())
            sid = int(df.iloc[j]["sample_idx"])
        except Exception:
            return

        self._set_ids("point", [sid])

    def _on_point_id(self, sid: int) -> None:
        # Only act if user is in point mode.
        if self._mode() != "point":
            return
        try:
            self._set_ids("point", [int(sid)])
        except Exception:
            return

    def _on_group_bbox(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> None:
        # Only act if user is in group mode.
        if self._mode() != "group":
            return

        ids = self._ids_from_bbox(
            float(xmin), float(ymin), float(xmax), float(ymax)
        )
        if not ids:
            self.clear()
            return

        self._set_ids("group", ids)

    def _set_ids(self, _mode: str, ids) -> None:
        """
        Update ids + open drawer.

        IMPORTANT:
        Do NOT write MAP_SELECT_MODE here.
        """
        if self._s is None:
            return
        with self._s.batch():
            self._s.set(MAP_SELECT_IDS, list(ids or []))
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
        # Prefer what is currently displayed on the map.
        try:
            if bool(self._s.get(K_PROP_ENABLED, False)):
                dfp = self._s.get(MAP_PROP_DF_POINTS, None)
                if isinstance(dfp, pd.DataFrame):
                    return dfp
        except :
            pass
    
        # Then fall back to controller bundle.
        if self._brain is not None:
            try:
                b = self._brain.get_bundle()
                df = getattr(b, "df_points", None)
                if isinstance(df, pd.DataFrame):
                    return df
            except :
                pass
    
        # Finally fall back to store base points.
        try:
            return self._s.get(MAP_DF_POINTS, None)
        except :
            return None
