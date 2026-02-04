# geoprior/ui/map/controller.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
geoprior.ui.map.controller

MapController is the "brain" of the Map tab.

Goal
----
All panels consume the same normalized dataset:

- Data panel selects files
- Header chooses X/Y/Z + coord mode + EPSG
- Controller loads CSV once, normalizes coordinates to
  lon/lat (WGS84), ensures a stable sample id, applies
  the same spatial sampling, then stores the results
  in the GeoConfigStore for reuse by:
    * map canvas renderer
    * analytics panel
    * future panels

Store contract (derived keys)
-----------------------------
- map.df_all      : full dataframe (all time steps)
- map.df_frame    : viewer frame (time slice) sampled
- map.df_points   : lon/lat/v/sid dataframe sampled
- map.id_col      : forced to "sample_idx" (created if
                    missing)

Notes
-----
- UI-light: watches store changes and builds derived data.
- Debounced: many store updates → one rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from PyQt5.QtCore import QObject, QTimer

from ...config.store import GeoConfigStore
from ..view.factory import ViewFactory
from .coord_utils import ensure_lonlat, parse_epsg, df_to_lonlat
from .sampling import cfg_from_get as samp_cfg_from_get
from .sampling import sample_points
from .keys import (
    MAP_ACTIVE_FILE,
    MAP_COORD_MODE,
    MAP_X_COL,
    MAP_Y_COL,
    MAP_Z_COL,
    MAP_VALUE_COL,
    MAP_TIME_COL,
    MAP_STEP_COL,
    MAP_TIME_VALUE,
    MAP_ID_COL,
    MAP_CLICK_SAMPLE_IDX,
    MAP_SAMPLING_MODE,
    MAP_SAMPLING_METHOD,
    MAP_SAMPLING_MAX_POINTS,
    MAP_SAMPLING_SEED,
    MAP_SAMPLING_CELL_KM,
    MAP_SAMPLING_MAX_PER_CELL,
    MAP_SAMPLING_APPLY_HOTSPOTS,
    MAP_DF_ALL,
    MAP_DF_FRAME,
    MAP_DF_POINTS,
    MAP_UTM_EPSG, 
    MAP_COORD_EPSG, 
    MAP_SRC_EPSG, 
)


@dataclass
class MapDataBundle:
    df_all: Optional[pd.DataFrame] = None
    df_frame: Optional[pd.DataFrame] = None
    df_points: Optional[pd.DataFrame] = None
    ok: bool = False
    err: str = ""


class MapController(QObject):
    """
    Store-driven controller that builds normalized map data.

    Parameters
    ----------
    store:
        GeoConfigStore single source of truth.
    canvas:
        Optional map canvas. If provided, click signals
        can be handled here (store updates).
    view_factory:
        Optional ViewFactory (future extension).
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        canvas: Any = None,
        view_factory: Optional[ViewFactory] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.canvas = canvas
        self.vf = view_factory

        self._bundle = MapDataBundle()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._rebuild)

        self._ignore_keys = {
            MAP_DF_ALL,
            MAP_DF_FRAME,
            MAP_DF_POINTS,
        }

        self._data_keys = {
            MAP_ACTIVE_FILE,
            MAP_X_COL,
            MAP_Y_COL,
            MAP_Z_COL,
            MAP_VALUE_COL,
            MAP_TIME_COL,
            MAP_STEP_COL,
            MAP_TIME_VALUE,
            MAP_COORD_MODE,
            MAP_UTM_EPSG, 
            MAP_COORD_EPSG, 
            MAP_SRC_EPSG, 
            MAP_SAMPLING_MODE,
            MAP_SAMPLING_METHOD,
            MAP_SAMPLING_MAX_POINTS,
            MAP_SAMPLING_SEED,
            MAP_SAMPLING_CELL_KM,
            MAP_SAMPLING_MAX_PER_CELL,
            MAP_SAMPLING_APPLY_HOTSPOTS,
        }

        self.store.config_changed.connect(
            self._on_store_changed
        )

        if self.canvas is not None:
            try:
                self.canvas.point_clicked_id.connect(
                    self._on_click_id
                )
            except Exception:
                pass

        self.request_rebuild()

    # -------------------------
    # Public
    # -------------------------
    def request_rebuild(self) -> None:
        self._timer.start()

    def get_bundle(self) -> MapDataBundle:
        return self._bundle

    # -------------------------
    # Store events
    # -------------------------
    def _on_store_changed(self, keys) -> None:
        ks = set(keys or [])
        if not ks:
            return
        if ks.issubset(self._ignore_keys):
            return
        if MAP_CLICK_SAMPLE_IDX in ks:
            return
        if ks.intersection(self._data_keys):
            self.request_rebuild()

    # -------------------------
    # Click handling
    # -------------------------
    def _on_click_id(self, sid: int) -> None:
        try:
            self.store.set(
                MAP_CLICK_SAMPLE_IDX,
                int(sid),
            )
        except Exception:
            return

    # -------------------------
    # Core build
    # -------------------------
    def _rebuild(self) -> None:
        b = self._build_bundle()
        self._bundle = b

        with self.store.batch():
            self.store.set(MAP_DF_ALL, b.df_all)
            self.store.set(MAP_DF_FRAME, b.df_frame)
            self.store.set(MAP_DF_POINTS, b.df_points)

            cur_id = str(
                self.store.get(MAP_ID_COL, "") or ""
            )
            if not cur_id.strip():
                self.store.set(MAP_ID_COL, "sample_idx")

    def _build_bundle(self) -> MapDataBundle:
        b = MapDataBundle()

        path = self._active_path()
        if path is None:
            b.err = "no active file"
            return b

        x = self._get_key(MAP_X_COL)
        y = self._get_key(MAP_Y_COL)
        z = self._get_value_col()
        t = self._get_key(MAP_TIME_COL)
        step = self._get_key(MAP_STEP_COL)
        tv = self._get_key(MAP_TIME_VALUE)

        if not (x and y and z):
            b.err = "mapping incomplete (x/y/z)"
            return b

        df_all = self._load_csv(
            path=path,
            x=x,
            y=y,
            z=z,
            t=t,
            step=step,
        )
        if df_all is None or df_all.empty:
            b.err = "failed to load csv"
            return b

        df_all = self._ensure_sample_idx(
            df_all,
            x=x,
            y=y,
        )

        df_view = self._slice_time(
            df_all,
            t=t,
            tv=tv,
        )

        pts = self._frame_to_lonlat(
            df_view,
            x=x,
            y=y,
            z=z,
        )
        if pts is None or pts.empty:
            b.df_all = df_all
            b.err = "invalid coordinates"
            return b

        scfg = samp_cfg_from_get(self.store.get)
        pts_s = sample_points(
            pts,
            scfg,
            lon="lon",
            lat="lat",
        )
        if pts_s is None or pts_s.empty:
            b.df_all = df_all
            b.err = "sampling removed all points"
            return b

        try:
            df_frame = df_view.loc[pts_s.index].copy()
        except Exception:
            df_frame = df_view.copy()

        b.df_all = df_all
        b.df_frame = df_frame
        b.df_points = pts_s
        b.ok = True
        return b

    # -------------------------
    # Helpers
    # -------------------------
    def _get_key(self, k: str) -> str:
        return str(self.store.get(k, "") or "").strip()

    def _get_value_col(self) -> str:
        z = self._get_key(MAP_VALUE_COL)
        if z:
            return z
        return self._get_key(MAP_Z_COL)

    def _active_path(self) -> Optional[Path]:
        p = self._get_key(MAP_ACTIVE_FILE)
        if not p:
            return None
        try:
            fp = Path(p)
        except Exception:
            return None
        if not fp.exists():
            return None
        return fp

    def _load_csv(
        self,
        *,
        path: Path,
        x: str,
        y: str,
        z: str,
        t: str,
        step: str,
    ) -> Optional[pd.DataFrame]:
        try:
            h = pd.read_csv(path, nrows=0)
            cols = list(h.columns)
        except Exception:
            cols = []

        use: list[str] = []
        for c in (x, y, z, t, step):
            if c and c not in use:
                use.append(c)

        for c in (
            "sample_idx",
            "sid",
            "site_id",
            "point_id",
            "id",
        ):
            if c in cols and c not in use:
                use.append(c)

        for c in cols:
            cl = str(c).lower()
            if cl.startswith("q") and cl[1:].isdigit():
                use.append(c)
                continue
            if any(
                k in cl
                for k in (
                    "obs",
                    "truth",
                    "actual",
                    "target",
                )
            ):
                use.append(c)

        use = list(dict.fromkeys(use))

        try:
            return pd.read_csv(path, usecols=use)
        except Exception:
            try:
                return pd.read_csv(path)
            except Exception:
                return None

    def _slice_time(
        self,
        df: pd.DataFrame,
        *,
        t: str,
        tv: str,
    ) -> pd.DataFrame:
        if not (t and tv and t in df.columns):
            return df

        s = df[t]
        if pd.api.types.is_numeric_dtype(s):
            try:
                v = float(tv)
                return df[s == v]
            except Exception:
                return df

        try:
            return df[s.astype(str) == str(tv)]
        except Exception:
            return df

    def _ensure_sample_idx(
        self,
        df: pd.DataFrame,
        *,
        x: str,
        y: str,
    ) -> pd.DataFrame:
        if "sample_idx" in df.columns:
            return self._coerce_int_id(df, "sample_idx")

        for cand in ("sid", "site_id", "point_id", "id"):
            if cand in df.columns:
                out = df.copy()
                out["sample_idx"] = pd.factorize(
                    out[cand].astype(str),
                    sort=True,
                )[0].astype("int64")
                return out

        return self._make_id_from_xy(df, x=x, y=y)

    def _coerce_int_id(
        self,
        df: pd.DataFrame,
        col: str,
    ) -> pd.DataFrame:
        out = df.copy()
        try:
            s = pd.to_numeric(out[col], errors="coerce")
            s = s.fillna(-1).astype("int64")
            out[col] = s
        except Exception:
            pass
        return out

    def _make_id_from_xy(
        self,
        df: pd.DataFrame,
        *,
        x: str,
        y: str,
    ) -> pd.DataFrame:
        out = df.copy()

        mode = str(
            self.store.get(
                MAP_COORD_MODE,
                "lonlat",
            ) or "lonlat"
        ).strip().lower()

        dec = 6
        if mode in ("utm", "xy", "epsg", "projected"):
            dec = 2

        try:
            rx = pd.to_numeric(
                out[x], errors="coerce"
            ).round(dec)
            ry = pd.to_numeric(
                out[y], errors="coerce"
            ).round(dec)
        except Exception:
            rx = out[x]
            ry = out[y]

        key = rx.astype(str) + "|" + ry.astype(str)

        out["sample_idx"] = pd.factorize(
            key,
            sort=True,
        )[0].astype("int64")

        return out

    def _frame_to_lonlat(
        self,
        df: pd.DataFrame,
        *,
        x: str,
        y: str,
        z: str,
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        if x not in df.columns or y not in df.columns:
            return None
        if z not in df.columns:
            return None

        d = pd.DataFrame(
            {
                "lon": df[x],
                "lat": df[y],
                "v": df[z],
                "sample_idx": df.get(
                    "sample_idx", None
                ),
            },
            index=df.index,
        )

        d["lon"] = pd.to_numeric(d["lon"], errors="coerce")
        d["lat"] = pd.to_numeric(d["lat"], errors="coerce")
        d["v"] = pd.to_numeric(d["v"], errors="coerce")
        d = d.dropna(subset=["lon", "lat", "v"])
        if d.empty:
            return None

        mode = str(
            self.store.get(
                MAP_COORD_MODE,
                "lonlat",
            ) or "lonlat"
        ).strip().lower()

        utm_epsg = parse_epsg(
            self.store.get(MAP_UTM_EPSG)
        )
        src_epsg = parse_epsg(
            self.store.get(MAP_COORD_EPSG)
        )
        if src_epsg is None:
            src_epsg = parse_epsg(
                self.store.get(MAP_SRC_EPSG)
            )

        try:
            out, ok, _msg = ensure_lonlat(
                d,
                mode=mode,
                utm_epsg=utm_epsg,
                src_epsg=src_epsg,
            )
        except Exception:
            out = df_to_lonlat(
                d,
                x="lon",
                y="lat",
                mode=mode,
                utm_epsg=utm_epsg,
                src_epsg=src_epsg,
            )
            ok = out is not None and (not out.empty)
        
        if (not ok) or out is None or out.empty:
            return None
        
        return out

