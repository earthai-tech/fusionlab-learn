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
from .prop_cache import PropagationScenarioCache
from .hotspots import build_points
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
    
    MAP_DF_ALL_POINTS,
    MAP_PROP_BUILD_ID,
    MAP_PROP_YEAR,
    MAP_PROP_TIMELINE,
    MAP_PROP_DF_ALL,
    MAP_PROP_DF_FRAME,
    MAP_PROP_DF_POINTS,
    MAP_PROP_VMIN,
    MAP_PROP_VMAX,
    MAP_PROP_VECTORS,
    K_PROP_ENABLED,
    K_PROP_YEARS,
    K_PROP_MODE,
    K_PROP_VECTORS,
    K_PROP_LEGEND,
    MAP_VIEW_PLOT_KIND,
)


@dataclass
class MapDataBundle:
    df_all: Optional[pd.DataFrame] = None
    df_frame: Optional[pd.DataFrame] = None
    df_points: Optional[pd.DataFrame] = None
    
    # optional: canonical lon/lat/v/t for full history
    df_all_points: Optional[pd.DataFrame] = None
    
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
        self._syncing = False
        
        self._prop_cache = PropagationScenarioCache()
        self._prop_sig = None


        self._bundle = MapDataBundle()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._rebuild)

        self._ignore_keys = {
            MAP_DF_ALL,
            MAP_DF_FRAME,
            MAP_DF_POINTS,
            
            MAP_DF_ALL_POINTS,
            MAP_PROP_TIMELINE,
            MAP_PROP_DF_ALL,
            MAP_PROP_DF_FRAME,
            MAP_PROP_DF_POINTS,
            MAP_PROP_VECTORS,
            MAP_PROP_VMIN,
            MAP_PROP_VMAX,
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
        
        self._prop_keys = {
            K_PROP_ENABLED,
            K_PROP_YEARS,
            K_PROP_MODE,
            K_PROP_LEGEND,
            K_PROP_VECTORS,
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

        # --- NEW: sync view "look" <-> propagation enabled ---
        self._sync_view_and_prop(ks)

        # Prop: timeline frame switch
        if MAP_PROP_YEAR in ks:
            self._update_prop_frame()
            return

        # Prop: explicit rebuild request (must FORCE)
        if ks == {MAP_PROP_BUILD_ID}:
            self._rebuild_prop_scenario(force=True)
            return

        # Prop: frame-only toggles (no rebuild)
        if ks.intersection({K_PROP_VECTORS, K_PROP_LEGEND}):
            if bool(self.store.get(K_PROP_ENABLED, False)):
                self._update_prop_frame()
            else:
                self._clear_prop_store()
            return

        # Prop: settings that require scenario rebuild
        if ks.intersection({K_PROP_ENABLED, K_PROP_YEARS, K_PROP_MODE}):
            if bool(self.store.get(K_PROP_ENABLED, False)):
                self._rebuild_prop_scenario(force=True)
            else:
                self._clear_prop_store()
            return

        if ks.intersection(self._data_keys):
            self.request_rebuild()

    def _sync_view_and_prop(self, ks: set) -> None:
        if self._syncing:
            return
    
        if (
            (MAP_VIEW_PLOT_KIND not in ks)
            and (K_PROP_ENABLED not in ks)
        ):
            return
    
        self._syncing = True
        
        try:
            kind = str(
                self.store.get(MAP_VIEW_PLOT_KIND, "")
                or ""
            ).strip().lower()
    
            prop_on = bool(
                self.store.get(K_PROP_ENABLED, False)
            )
    
            kind_changed = MAP_VIEW_PLOT_KIND in ks
            en_changed = K_PROP_ENABLED in ks
    
            # 1) User toggled propagation enable -> update view,
            #    but DO NOT apply "look => enable" back.
            if en_changed and not kind_changed:
                if prop_on and kind != "look":
                    self.store.set(MAP_VIEW_PLOT_KIND, "look")
                elif (not prop_on) and kind == "look":
                    self.store.set(MAP_VIEW_PLOT_KIND, "points")
                return
    
            # 2) User changed view kind -> update propagation.
            if kind_changed:
                if kind == "look" and not prop_on:
                    with self.store.batch():
                        self.store.set(K_PROP_ENABLED, True)
    
                        if self.store.get(K_PROP_YEARS, None) is None:
                            self.store.set(K_PROP_YEARS, 5)
    
                        bid = int(
                            self.store.get(MAP_PROP_BUILD_ID, 0)
                            or 0
                        )
                        self.store.set(MAP_PROP_BUILD_ID, bid + 1)
                    return
    
                if kind != "look" and prop_on:
                    self.store.set(K_PROP_ENABLED, False)
                return
    
        finally:
            self._syncing = False

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
            # Keep propagation scenario consistent with base data
            if bool(self.store.get(K_PROP_ENABLED, False)):
                self._rebuild_prop_scenario()

            cur_id = str(
                self.store.get(MAP_ID_COL, "") or ""
            )
            if not cur_id.strip():
                self.store.set(MAP_ID_COL, "sample_idx")

    def _build_bundle(self) -> MapDataBundle:
        b = MapDataBundle()

        path = self._active_path()
        
        # after: path = self._active_path()
        
        try:
            cols = list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            cols = []
        
        t = self._get_key(MAP_TIME_COL)
        if not t:
            for c in ("coord_t", "t", "year", "date"):
                if c in cols:
                    self.store.set(MAP_TIME_COL, c)
                    t = c
                    break
        
        step = self._get_key(MAP_STEP_COL)
        if not step:
            for c in ("forecast_step", "step", "horizon"):
                if c in cols:
                    self.store.set(MAP_STEP_COL, c)
                    step = c
                    break

        if path is None:
            b.err = "no active file"
            return b

        x = self._get_key(MAP_X_COL)
        y = self._get_key(MAP_Y_COL)
        z = self._get_value_col()
        step = self._get_key(MAP_STEP_COL)
        tv = self._get_key(MAP_TIME_VALUE)
        epsg = self._get_key (MAP_COORD_EPSG)

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
        
        df_all_points = self._df_all_to_points(
            df_all, x, y, z, t, epsg
            )
        
        b.df_all_points = df_all_points
        self.store.set(MAP_DF_ALL_POINTS, b.df_all_points)

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

    def _df_all_to_points(
        self,
        df_all: pd.DataFrame,
        x: str,
        y: str,
        v: str,
        t: str,
        _epsg: str = "",
    ) -> pd.DataFrame:
        """
        Build canonical lon/lat/v/t (+sample_idx) for full history.

        This is the stable "points history" used by propagation,
        hotspot trend decorators, and any future time analytics.
        """
        if df_all is None or df_all.empty:
            return pd.DataFrame(
                columns=["lon", "lat", "v", "t", "sample_idx"]
            )
        
        t_col = str(t or "").strip()
        if (not t_col) or (t_col not in df_all.columns):
            for c in ("coord_t", "t", "year", "date"):
                if c in df_all.columns:
                    t_col = c
                    break
        try:
            pts = build_points(
                df_all,
                x=str(x),
                y=str(y),
                v=str(v),
                t=str(t or ""),
            )
        except TypeError:
            pts = build_points(
                df_all,
                x=str(x),
                y=str(y),
                v=str(v),
            )

        if pts is None or pts.empty:
            return pd.DataFrame(
                columns=["lon", "lat", "v", "t", "sample_idx"]
            )

        # attach stable id if available
        if "sample_idx" in df_all.columns:
            try:
                pts["sample_idx"] = df_all.loc[
                    pts.index, "sample_idx"
                ]
            except Exception:
                pass

        mode = str(
            self.store.get(MAP_COORD_MODE, "lonlat")
            or "lonlat"
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
                pts,
                mode=mode,
                utm_epsg=utm_epsg,
                src_epsg=src_epsg,
            )
        except Exception:
            out = df_to_lonlat(
                pts,
                x="lon",
                y="lat",
                mode=mode,
                utm_epsg=utm_epsg,
                src_epsg=src_epsg,
            )
            ok = out is not None and (not out.empty)

        if (not ok) or out is None or out.empty:
            return pd.DataFrame(
                columns=["lon", "lat", "v", "t", "sample_idx"]
            )

        # Ensure canonical cols exist
        if "t" not in out.columns:
            out["t"] = pd.NA
        if "sample_idx" not in out.columns:
            out["sample_idx"] = pd.NA

        return out

    def _clear_prop_store(self) -> None:
        with self.store.batch():
            self.store.set(MAP_PROP_TIMELINE, [])
            self.store.set(MAP_PROP_DF_ALL, None)
            self.store.set(MAP_PROP_DF_FRAME, None)
            self.store.set(MAP_PROP_DF_POINTS, None)
            self.store.set(MAP_PROP_VECTORS, [])
            self.store.set(MAP_PROP_VMIN, None)
            self.store.set(MAP_PROP_VMAX, None)

        self._prop_sig = None
        try:
            self._prop_cache = PropagationScenarioCache()
        except Exception:
            pass

    def _rebuild_prop_scenario(self, *, force: bool = False) -> None:
        """
        Build (or reuse) the propagation scenario cache from
        MAP_DF_ALL_POINTS (canonical lon/lat/v/t history).
        """
        if not bool(self.store.get(K_PROP_ENABLED, False)):
            self._clear_prop_store()
            return

        base = self.store.get(MAP_DF_ALL_POINTS, None)
        if (not isinstance(base, pd.DataFrame)) or base.empty:
            # Try build from current bundle as fallback
            b = getattr(self, "_bundle", None)
            df_all = getattr(b, "df_all", None)
            if not isinstance(df_all, pd.DataFrame):
                self._clear_prop_store()
                return

            x = self._get_key(MAP_X_COL)
            y = self._get_key(MAP_Y_COL)
            v = self._get_value_col()
            t = self._get_key(MAP_TIME_COL)

            base = self._df_all_to_points(
                df_all,
                x,
                y,
                v,
                t,
            )

        if (not isinstance(base, pd.DataFrame)) or base.empty:
            self._clear_prop_store()
            return

        years = int(self.store.get(K_PROP_YEARS, 5) or 5)
        if years < 0:
            years = 0
            
        mode = str(
            self.store.get(K_PROP_MODE, "linear") or "linear"
        ).strip().lower()
        leg = str(
            self.store.get(K_PROP_LEGEND, "global") or "global"
        ).strip().lower()

        sig = (
            str(self.store.get(MAP_ACTIVE_FILE, "") or ""),
            int(years),
            str(mode),
            str(leg),
        )

        # Avoid rebuilding if nothing material changed.
        if (not force) and (self._prop_sig == sig):
            self._update_prop_frame()
            return

        try:
            self._prop_cache.build(
                base,
                years_to_add=int(years),
                time_col="t",
                value_col="v",
                mode=str(mode),
                legend_policy=str(leg),
            )
        except Exception:
            self._clear_prop_store()
            return

        self._prop_sig = sig

        # timeline
        tl = []
        try:
            tlf = getattr(self._prop_cache, "timeline", None)
            if callable(tlf):
                tl = list(tlf() or [])
            else:
                tl = list(getattr(self._prop_cache, "timeline", []) or [])
        except Exception:
            tl = []

        # scenario all-df (optional)
        df_prop_all = None
        try:
            af = getattr(self._prop_cache, "df_all", None)
            if callable(af):
                df_prop_all = af()
            elif isinstance(af, pd.DataFrame):
                df_prop_all = af
        except Exception:
            df_prop_all = None

        # global vmin/vmax (best-effort)
        vmin = None
        vmax = None
        try:
            rr = getattr(self._prop_cache, "legend_range", None)
            if callable(rr):
                vmin, vmax = rr(policy="global")
        except Exception:
            vmin, vmax = None, None

        if (
            (vmin is None or vmax is None)
            and isinstance(df_prop_all, pd.DataFrame)
            and (not df_prop_all.empty)
            and ("v" in df_prop_all.columns)
        ):
            s = pd.to_numeric(
                df_prop_all["v"], errors="coerce"
            )
            s = s[s.notna()]
            if len(s) > 0:
                vmin = float(s.min())
                vmax = float(s.max())

        # choose initial year: prefer current time slice
        y0 = self.store.get(MAP_PROP_YEAR, None)
        try:
            y0 = int(y0)
        except Exception:
            y0 = None

        if tl and (y0 not in set(tl)):
            tv = self.store.get(MAP_TIME_VALUE, None)
            y1 = None
            try:
                y1 = int(float(tv))
            except Exception:
                y1 = None
            if y1 in set(tl):
                y0 = int(y1)
            else:
                y0 = int(tl[0])

        with self.store.batch():
            self.store.set(MAP_PROP_TIMELINE, tl)

            if isinstance(df_prop_all, pd.DataFrame):
                self.store.set(MAP_PROP_DF_ALL, df_prop_all)
            else:
                self.store.set(MAP_PROP_DF_ALL, None)

            self.store.set(MAP_PROP_VMIN, vmin)
            self.store.set(MAP_PROP_VMAX, vmax)

            if y0 is not None:
                self.store.set(MAP_PROP_YEAR, int(y0))

        self._update_prop_frame()

    def _update_prop_frame(self) -> None:
        """
        Push the current propagation year frame into the store:
          - MAP_PROP_DF_FRAME
          - MAP_PROP_DF_POINTS (sampled)
          - MAP_PROP_VECTORS (optional)
          - MAP_PROP_VMIN/VMAX (if per-frame legend policy)
        """
        if not bool(self.store.get(K_PROP_ENABLED, False)):
            return

        year = self.store.get(MAP_PROP_YEAR, None)
        try:
            year = int(year)
        except Exception:
            return

        try:
            df_y = self._prop_cache.frame(int(year))
        except Exception:
            df_y = None

        if (not isinstance(df_y, pd.DataFrame)) or df_y.empty:
            with self.store.batch():
                self.store.set(MAP_PROP_DF_FRAME, None)
                self.store.set(MAP_PROP_DF_POINTS, None)
                self.store.set(MAP_PROP_VECTORS, [])
            return

        # Ensure canonical cols exist
        need = {"lon", "lat", "v"}
        if not need.issubset(set(df_y.columns)):
            with self.store.batch():
                self.store.set(MAP_PROP_DF_FRAME, None)
                self.store.set(MAP_PROP_DF_POINTS, None)
                self.store.set(MAP_PROP_VECTORS, [])
            return

        scfg = samp_cfg_from_get(self.store.get)

        pts = df_y.copy()
        pts["lon"] = pd.to_numeric(pts["lon"], errors="coerce")
        pts["lat"] = pd.to_numeric(pts["lat"], errors="coerce")
        pts["v"] = pd.to_numeric(pts["v"], errors="coerce")
        pts = pts.dropna(subset=["lon", "lat", "v"])

        if pts.empty:
            with self.store.batch():
                self.store.set(MAP_PROP_DF_FRAME, None)
                self.store.set(MAP_PROP_DF_POINTS, None)
                self.store.set(MAP_PROP_VECTORS, [])
            return

        pts_s = sample_points(
            pts,
            scfg,
            lon="lon",
            lat="lat",
        )

        # vectors (optional, cache-provided)
        vec = []
        if bool(self.store.get(K_PROP_VECTORS, True)):
            try:
                vf = getattr(self._prop_cache, "vectors", None)
                if callable(vf):
                    vec = list(vf(int(year)) or [])
            except Exception:
                vec = []

        # per-frame legend range (optional)
        leg = str(
            self.store.get(K_PROP_LEGEND, "global") or "global"
        ).strip().lower()

        vmin = None
        vmax = None
        if leg in ("frame", "per_frame", "year"):
            try:
                rr = getattr(self._prop_cache, "legend_range", None)
                if callable(rr):
                    vmin, vmax = rr(
                        year=int(year),
                        policy="frame",
                    )
            except Exception:
                vmin, vmax = None, None

            if vmin is None or vmax is None:
                s = pd.to_numeric(
                    pts_s["v"], errors="coerce"
                )
                s = s[s.notna()]
                if len(s) > 0:
                    vmin = float(s.min())
                    vmax = float(s.max())

        with self.store.batch():
            self.store.set(MAP_PROP_DF_FRAME, pts_s)
            self.store.set(MAP_PROP_DF_POINTS, pts_s)
            self.store.set(MAP_PROP_VECTORS, vec)

            if leg in ("frame", "per_frame", "year"):
                self.store.set(MAP_PROP_VMIN, vmin)
                self.store.set(MAP_PROP_VMAX, vmax)
