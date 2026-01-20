# geoprior/ui/xfer/map/controller.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.map.controller

Controller for the Xfer map page.
- store-driven
- uses scan_results_root + ensure_lonlat
- loads CSVs, slices by step/year, computes value
- pushes layers to MapApi (Leaflet)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from PyQt5.QtCore import QObject

from ....config.store import GeoConfigStore
from ...map.utils import scan_results_root
from ...map.coord_utils import ensure_lonlat
from ...map.hotspots import HotspotCfg, compute_hotspots
from ..insights import build_xfer_badges
from ..types import MapApi, MapPoint
from .toolbar import DatasetChoice, XferMapToolbar
from ..keys import (
    DEFAULTS,
    map_keys,
    K_MAP_SPLIT,
    K_MAP_VALUE,
    K_MAP_OVERLAY,
    K_MAP_SHARED,
    K_MAP_TIME_MODE,
    K_MAP_STEP,
    K_MAP_MAX_POINTS,
    K_MAP_COORD_MODE,
    K_MAP_UTM_EPSG,
    K_MAP_SRC_EPSG,
    K_MAP_A_JOB_KIND,
    K_MAP_A_JOB_ID,
    K_MAP_A_FILE,
    K_MAP_B_JOB_KIND,
    K_MAP_B_JOB_ID,
    K_MAP_B_FILE,
    K_MAP_OPACITY,
    K_MAP_POINTS_MODE,
    K_MAP_MARKER_SHAPE,
    K_MAP_MARKER_SIZE,
    K_MAP_HOTSPOT_TOPN,
    K_MAP_HOTSPOT_MIN_SEP_KM,
    K_MAP_HOTSPOT_METRIC,
    K_MAP_HOTSPOT_QUANTILE,
    K_MAP_ANIM_PULSE,
    K_MAP_ANIM_PLAY_MS,
    K_MAP_INSIGHT

)



@dataclass(frozen=True)
class JobKey:
    kind: str
    job_id: str


class XferMapController(QObject):
    """
    Wires store + toolbar + map view.

    Uses:
    - scan_results_root(results_root)
    - ensure_lonlat(...) for UTM/EPSG->WGS84
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        toolbar: XferMapToolbar,
        view: MapApi,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._tb = toolbar
        self._v = view

        self._root: Optional[Path] = None
        self._index: Optional[List[Any]] = None
        self._city_map: Dict[str, Any] = {}

        self._df_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

        self._connect()
        self.refresh()

    # -------------------------
    # Public
    # -------------------------
    def refresh(self) -> None:
        self._ensure_defaults()
        self._refresh_index(force=False)
        self._sync_toolbar_all()
        self._render_from_store()

    # -------------------------
    # Wiring
    # -------------------------
    def _connect(self) -> None:
        self._s.config_changed.connect(self._on_store_changed)
        self._tb.changed.connect(self._on_toolbar_changed)
        self._tb.request_refresh.connect(self.refresh)
        self._tb.request_fit.connect(self._on_fit)

    def _on_fit(self) -> None:
        ids = self._active_layer_ids()
        try:
            self._v.fit_layers(ids)
        except Exception:
            return

    def _on_store_changed(self, keys: object) -> None:
        ch = self._as_set(keys)
        if not ch:
            return
    
        if "results_root" in ch:
            self._refresh_index(force=True)
            self._sync_toolbar_all()
            self._render_from_store()
            return
    
        if ch & {"xfer.city_a", "xfer.city_b"}:
            self._sync_toolbar_all()
            self._render_from_store()
            return
    
        if ch & self._map_keys():
            self._sync_toolbar_from_store(ch)
            self._render_from_store()


    def _on_toolbar_changed(self) -> None:
        st = self._tb.get_ui_state()
        s = self._s
    
        a_job = st.get("job_a")
        b_job = st.get("job_b")
    
        a_jk = self._as_jobkey(a_job)
        b_jk = self._as_jobkey(b_job)
    
        with s.batch():
            # Cities are already stored globally in xfer tab
            # as xfer.city_a / xfer.city_b (no new keys here).
            s.set("xfer.city_a", st.get("city_a") or "")
            s.set("xfer.city_b", st.get("city_b") or "")

            # A selection
            s.set(K_MAP_A_JOB_KIND, a_jk.kind if a_jk else None)
            s.set(K_MAP_A_JOB_ID, a_jk.job_id if a_jk else None)
            s.set(K_MAP_A_FILE, st.get("file_a") or "")
    
            # B selection
            s.set(K_MAP_B_JOB_KIND, b_jk.kind if b_jk else None)
            s.set(K_MAP_B_JOB_ID, b_jk.job_id if b_jk else None)
            s.set(K_MAP_B_FILE, st.get("file_b") or "")
    
            # Map knobs
            s.set(K_MAP_SPLIT, st.get("split"))
            s.set(K_MAP_VALUE, st.get("value"))
            s.set(K_MAP_OVERLAY, st.get("overlay"))
            s.set(K_MAP_SHARED, bool(st.get("shared")))
            s.set(K_MAP_TIME_MODE, st.get("time_mode"))
            s.set(K_MAP_STEP, int(st.get("step") or 1))
            # View options
            s.set(
                K_MAP_POINTS_MODE,
                st.get("points_mode") or "all",
            )
            s.set(
                K_MAP_MARKER_SHAPE,
                st.get("marker_shape") or "auto",
            )
            s.set(
                K_MAP_MARKER_SIZE,
                int(st.get("marker_size") or 6),
            )
            s.set(
                K_MAP_HOTSPOT_TOPN,
                int(st.get("hotspot_top_n") or 8),
            )
            s.set(
                K_MAP_ANIM_PULSE,
                bool(st.get("pulse")),
            )
            s.set(
                K_MAP_ANIM_PLAY_MS,
                int(st.get("play_ms") or 320),
            )
            s.set(K_MAP_INSIGHT, bool(st.get("insight")))
        try:
            self._tb.set_play_ms(
                int(st.get("play_ms") or 320)
            )
        except Exception:
            pass

    # -------------------------
    # Index
    # -------------------------
    def _refresh_index(self, *, force: bool) -> None:
        root = self._s.get("results_root", "") or ""
        rp = Path(str(root)).expanduser()

        if not root or not rp.exists():
            self._root = None
            self._index = None
            self._city_map = {}
            self._tb.set_city_choices(a=[], b=[])
            self._tb.set_job_choices(a=[], b=[])
            self._tb.set_file_choices(a=[], b=[])
            return

        if (not force) and (self._root == rp) and self._index:
            return

        self._root = rp
        self._index = scan_results_root(rp) or []
        self._city_map = {c.city: c for c in self._index}

    # -------------------------
    # Toolbar sync
    # -------------------------
    def _sync_toolbar_all(self) -> None:
        self._sync_city_choices()
        self._sync_job_choices()
        self._sync_file_choices()
        self._sync_value_choices()
        self._sync_toolbar_from_store(set())

    def _sync_toolbar_from_store(self, _keys: Set[str]) -> None:
        s = self._s
    
        # Cities are owned by the main xfer tab
        a_city = str(s.get("xfer.city_a", "") or "")
        b_city = str(s.get("xfer.city_b", "") or "")
    
        if a_city:
            self._tb.set_current_data(self._tb.cmb_city_a, a_city)
        if b_city:
            self._tb.set_current_data(self._tb.cmb_city_b, b_city)
    
        # Jobs + files depend on chosen cities
        self._sync_job_choices()
        self._sync_file_choices()
    
        # Map knobs
        split = str(s.get(K_MAP_SPLIT, "val") or "val")
        value = str(s.get(K_MAP_VALUE, "auto") or "auto")
        overlay = str(s.get(K_MAP_OVERLAY, "both") or "both")
        time_mode = str(
            s.get(K_MAP_TIME_MODE, "forecast_step") or "forecast_step"
        )
        shared = bool(s.get(K_MAP_SHARED, True))
        
        self._tb.set_split(split)
        self._tb.set_value(value)
        self._tb.set_overlay(overlay)
        self._tb.set_time_mode(time_mode)
        self._tb.set_shared(shared)

        # File selection from store
        a_file = str(s.get(K_MAP_A_FILE, "") or "")
        b_file = str(s.get(K_MAP_B_FILE, "") or "")
    
        if a_file:
            self._tb.set_current_data(self._tb.cmb_file_a, a_file)
        if b_file:
            self._tb.set_current_data(self._tb.cmb_file_b, b_file)
    
        # Step range (updated again after load)
        step = int(s.get(K_MAP_STEP, 1) or 1)
        self._tb.set_time_range(step_min=1, step_max=1, step=step)

        pts_mode = str(
            s.get(K_MAP_POINTS_MODE, "all") or "all"
        )
        mk_shape = str(
            s.get(K_MAP_MARKER_SHAPE, "auto") or "auto"
        )
        mk_size = int(
            s.get(K_MAP_MARKER_SIZE, 6) or 6
        )
        topn = int(
            s.get(K_MAP_HOTSPOT_TOPN, 8) or 8
        )
        pulse = bool(
            s.get(K_MAP_ANIM_PULSE, True)
        )
        play_ms = int(
            s.get(K_MAP_ANIM_PLAY_MS, 320) or 320
        )

        self._tb.set_points_mode(pts_mode)
        self._tb.set_marker_shape(mk_shape)
        self._tb.set_marker_size(mk_size)
        self._tb.set_hotspot_topn(topn)
        self._tb.set_pulse(pulse)
        self._tb.set_play_ms(play_ms)
        self._tb.set_insight(bool(s.get(K_MAP_INSIGHT, False)))


    def _sync_city_choices(self) -> None:
        cities = sorted(self._city_map.keys())
        items = [DatasetChoice(c, c) for c in cities]
        self._tb.set_city_choices(a=items, b=items)

    def _sync_job_choices(self) -> None:
        
        a_city = str(self._tb.cmb_city_a.currentData() or "")
        b_city = str(self._tb.cmb_city_b.currentData() or "")

        a = self._job_items(a_city)
        b = self._job_items(b_city)

        self._tb.set_job_choices(a=a, b=b)
    
        a_kind = self._s.get(K_MAP_A_JOB_KIND, None)
        a_id = self._s.get(K_MAP_A_JOB_ID, None)
        b_kind = self._s.get(K_MAP_B_JOB_KIND, None)
        b_id = self._s.get(K_MAP_B_JOB_ID, None)
    
        if a and not (a_kind and a_id):
            jk = self._as_jobkey(a[0].data)
            if jk:
                self._s.set(K_MAP_A_JOB_KIND, jk.kind)
                self._s.set(K_MAP_A_JOB_ID, jk.job_id)
                self._s.set(K_MAP_A_FILE, "")
    
        if b and not (b_kind and b_id):
            jk = self._as_jobkey(b[0].data)
            if jk:
                self._s.set(K_MAP_B_JOB_KIND, jk.kind)
                self._s.set(K_MAP_B_JOB_ID, jk.job_id)
                self._s.set(K_MAP_B_FILE, "")
            
    def _sync_file_choices(self) -> None:
        a_city = str(self._tb.cmb_city_a.currentData() or "")
        b_city = str(self._tb.cmb_city_b.currentData() or "")
        
        a_kind = self._s.get(K_MAP_A_JOB_KIND, None)
        a_id = self._s.get(K_MAP_A_JOB_ID, None)
        b_kind = self._s.get(K_MAP_B_JOB_KIND, None)
        b_id = self._s.get(K_MAP_B_JOB_ID, None)
        
        a_job = JobKey(str(a_kind), str(a_id)) if a_kind and a_id else None
        b_job = JobKey(str(b_kind), str(b_id)) if b_kind and b_id else None


        a = self._file_items(a_city, a_job)
        b = self._file_items(b_city, b_job)

        self._tb.set_file_choices(a=a, b=b)

        if a and not str(self._s.get(K_MAP_A_FILE, "") or ""):
            self._s.set(K_MAP_A_FILE, a[0].data)
        if b and not str(self._s.get(K_MAP_B_FILE, "") or ""):
            self._s.set(K_MAP_B_FILE, b[0].data)

    def _sync_value_choices(self) -> None:
        items = [
            DatasetChoice("AUTO", "auto"),
            DatasetChoice("Median (q50)", "subsidence_q50"),
            DatasetChoice("Prediction", "subsidence_pred"),
            DatasetChoice("Spread (q90-q10)", "spread"),
            DatasetChoice("q10", "subsidence_q10"),
            DatasetChoice("q90", "subsidence_q90"),
        ]
        self._tb.set_value_choices(items)

    def _job_items(self, city: str) -> List[DatasetChoice]:
        c = self._city_map.get(city, None)
        if c is None:
            return []

        jobs = list(getattr(c, "jobs", []) or [])
        jobs = sorted(jobs, key=lambda j: j.job_id, reverse=True)

        out: List[DatasetChoice] = []
        for j in jobs:
            txt = f"{j.kind} {j.job_id}"
            out.append(DatasetChoice(txt, JobKey(j.kind, j.job_id)))
        return out

    def _file_items(
        self,
        city: str,
        job: Any,
    ) -> List[DatasetChoice]:
        c = self._city_map.get(city, None)
        if c is None or job is None:
            return []
    
        jk = self._as_jobkey(job)
        if jk is None:
            return []
    
        j = self._find_job(c, jk)
        if j is None:
            return []
    
        files = list(getattr(j, "files", ()) or ())
        out: List[DatasetChoice] = []
    
        for f in files:
            p = Path(str(f.path)).expanduser()
            txt = getattr(f, "display", p.name)
            out.append(DatasetChoice(txt, str(p)))
    
        return out

    def _render_insight(
        self,
        *,
        pts_out: Dict[str, pd.DataFrame],
        overlay: str,
    ) -> None:
        show = bool(self._s.get(K_MAP_INSIGHT, False))
        if not show:
            self._v.clear_layer("XFER_AB")
            self._v.clear_layer("XFER_BA")
            return
    
        root = self._s.get("results_root", "") or ""
        rp = Path(str(root)).expanduser()
        if not rp.exists():
            return
    
        a_city = str(self._s.get("xfer.city_a", "") or "")
        b_city = str(self._s.get("xfer.city_b", "") or "")
        if not a_city or not b_city:
            return
    
        split = str(self._s.get(K_MAP_SPLIT, "val") or "val")
    
        a_df = pts_out.get("A", None)
        b_df = pts_out.get("B", None)
        if a_df is None or b_df is None:
            return
    
        a_ctr = self._centroid(a_df)
        b_ctr = self._centroid(b_df)
        if a_ctr is None or b_ctr is None:
            return
    
        want_ab = overlay in ("a", "both")
        want_ba = overlay in ("b", "both")
    
        layers = build_xfer_badges(
            results_root=rp,
            city_a=a_city,
            city_b=b_city,
            split=split,
            a_lat=a_ctr[0],
            a_lon=a_ctr[1],
            b_lat=b_ctr[0],
            b_lon=b_ctr[1],
            want_ab=want_ab,
            want_ba=want_ba,
        )
    
        if not want_ab:
            self._v.clear_layer("XFER_AB")
        if not want_ba:
            self._v.clear_layer("XFER_BA")
    
        for lid, name, pts, opts in layers:
            self._v.set_layer(lid, name, pts, opts)

    def _find_job(self, city_obj: Any, jk: JobKey) -> Any:
        for j in (getattr(city_obj, "jobs", []) or []):
            if (j.kind == jk.kind) and (j.job_id == jk.job_id):
                return j
        return None

    def _as_jobkey(self, v: Any) -> Optional[JobKey]:
        if isinstance(v, JobKey):
            return v
        if isinstance(v, dict):
            k = v.get("kind", None)
            i = v.get("job_id", None)
            if k and i:
                return JobKey(str(k), str(i))
        return None

    # -------------------------
    # Rendering
    # -------------------------
    def _render_from_store(self) -> None:
        self._clear_map_layers()

        overlay = str(self._s.get(K_MAP_OVERLAY, "both") or "both")
        shared = bool(self._s.get(K_MAP_SHARED, True))
        mode = str(self._s.get(K_MAP_TIME_MODE, "forecast_step") or "forecast_step")
        step = int(self._s.get(K_MAP_STEP, 1) or 1)
        value = str(self._s.get(K_MAP_VALUE, "auto") or "auto")
        
        a_file = str(self._s.get(K_MAP_A_FILE, "") or "")
        b_file = str(self._s.get(K_MAP_B_FILE, "") or "")

        layers: List[Tuple[str, str, str]] = []
        if overlay in ("a", "both") and a_file:
            layers.append(("A", "City A", str(a_file)))
        if overlay in ("b", "both") and b_file:
            layers.append(("B", "City B", str(b_file)))

        pts_out: Dict[str, pd.DataFrame] = {}
        unit = ""

        # First pass: load all dfs
        for lid, name, fp in layers:
            df, ok, msg, u, step_max = self._load_points(
                fp=fp, value=value, time_mode=mode, step=step
            )
            if not ok:
                self._set_status(msg, ok=False)
                continue
            unit = unit or (u or "")
            pts_out[lid] = df
            self._update_step_range(step, step_max)
        
        if not pts_out:
            self._set_status("No map dataset to plot.", ok=False)
            return
        
        vmin = vmax = None
        if shared:
            vmin, vmax = self._shared_minmax(pts_out)
            self._apply_legend(vmin, vmax, unit)
        
        # Second pass: render layers with consistent scale
        pts_mode = str(
            self._s.get(K_MAP_POINTS_MODE, "all") or "all"
        )
        mk_shape = str(
            self._s.get(K_MAP_MARKER_SHAPE, "auto") or "auto"
        )
        mk_size = int(
            self._s.get(K_MAP_MARKER_SIZE, 6) or 6
        )
        opacity = float(
            self._s.get(K_MAP_OPACITY, 0.90) or 0.90
        )
        pulse = bool(
            self._s.get(K_MAP_ANIM_PULSE, True)
        )

        # Second pass: render layers (+ hotspots)
        for lid, df in pts_out.items():
            base_name = "City A" if lid == "A" else "City B"

            if pts_mode in ("all", "hotspots_plus"):
                try:
                    self._v.set_layer(
                        layer_id=lid,
                        name=base_name,
                        points=self._as_points(df),
                        opts=self._layer_opts(
                            lid,
                            vmin=vmin,
                            vmax=vmax,
                            shape=self._shape_for(
                                mk_shape,
                                is_hot=False,
                            ),
                            radius=self._radius_for(
                                mk_size,
                                is_hot=False,
                            ),
                            opacity=opacity,
                            pulse=False,
                            enable_tip=False,
                        ),
                    )
                except Exception:
                    pass

            if pts_mode in ("hotspots", "hotspots_plus"):
                hp = self._hotspot_points(df)
                if hp:
                    try:
                        self._v.set_layer(
                            layer_id=f"{lid}_hot",
                            name=f"{base_name} hotspots",
                            points=hp,
                            opts=self._layer_opts(
                                lid,
                                vmin=vmin,
                                vmax=vmax,
                                shape=self._shape_for(
                                    mk_shape,
                                    is_hot=True,
                                ),
                                radius=self._radius_for(
                                    mk_size,
                                    is_hot=True,
                                ),
                                opacity=opacity,
                                pulse=pulse,
                                enable_tip=True,
                            ),
                        )
                    except Exception:
                        pass
                    
        self._render_insight(
            pts_out=pts_out,
            overlay=overlay,
        )

    def _load_points(
        self,
        *,
        fp: str,
        value: str,
        time_mode: str,
        step: int,
    ) -> Tuple[pd.DataFrame, bool, str, str, int]:
        p = Path(str(fp)).expanduser()
        if not p.exists():
            return (
                pd.DataFrame(),
                False,
                f"Missing file: {p.name}",
                "",
                1,
            )

        df = self._read_cached(p)
        if df is None or df.empty:
            return (
                pd.DataFrame(),
                False,
                f"Empty CSV: {p.name}",
                "",
                1,
            )

        step_max = self._infer_step_max(df, time_mode)

        sl, label = self._slice_time(df, time_mode, step)
        if sl.empty:
            return (
                pd.DataFrame(),
                False,
                f"No rows for {label} in {p.name}",
                "",
                step_max,
            )

        v = self._compute_value(sl, value)
        out = pd.DataFrame(
            {
                "lon": sl.get("coord_x", np.nan),
                "lat": sl.get("coord_y", np.nan),
                "v": v,
            }
        )

        max_pts = int(self._s.get(K_MAP_MAX_POINTS, 20000) or 20000)
        out = self._downsample(out, max_pts)

        cm = self._coord_mode()
        ue = self._epsg(self._s.get(K_MAP_UTM_EPSG, None))
        se = self._epsg(self._s.get(K_MAP_SRC_EPSG, None))


        out, ok, msg = ensure_lonlat(
            out,
            mode=cm,
            utm_epsg=ue,
            src_epsg=se,
        )
        if not ok:
            return (
                pd.DataFrame(),
                False,
                msg,
                "",
                step_max,
            )

        unit = ""
        if "subsidence_unit" in df.columns:
            try:
                unit = str(df["subsidence_unit"].iloc[0])
            except Exception:
                unit = ""

        return out, True, "", unit, step_max

    def _read_cached(self, p: Path) -> Optional[pd.DataFrame]:
        try:
            mt = float(p.stat().st_mtime)
        except Exception:
            mt = -1.0

        key = str(p)
        hit = self._df_cache.get(key, None)
        if hit is not None and hit[0] == mt:
            return hit[1]

        try:
            df = pd.read_csv(p)
        except Exception:
            return None

        self._df_cache[key] = (mt, df)
        return df

    def _slice_time(
        self,
        df: pd.DataFrame,
        mode: str,
        step: int,
    ) -> Tuple[pd.DataFrame, str]:
        m = str(mode or "forecast_step")
        st = int(step or 1)

        if m == "year" and "coord_t" in df.columns:
            yrs = sorted(pd.unique(df["coord_t"]))
            if not yrs:
                return df.iloc[0:0], "year"
            idx = max(0, min(st - 1, len(yrs) - 1))
            y = yrs[idx]
            return df[df["coord_t"] == y], f"year={y}"

        if "forecast_step" in df.columns:
            return df[df["forecast_step"] == st], f"step={st}"

        return df, "all"

    def _infer_step_max(self, df: pd.DataFrame, mode: str) -> int:
        m = str(mode or "forecast_step")
        if m == "year" and "coord_t" in df.columns:
            yrs = pd.unique(df["coord_t"])
            return int(max(1, len(yrs)))
        if "forecast_step" in df.columns:
            try:
                return int(pd.to_numeric(df["forecast_step"]).max())
            except Exception:
                return 1
        return 1

    def _compute_value(self, df: pd.DataFrame, value: str) -> pd.Series:
        v = str(value or "auto")

        if v == "auto":
            for c in ("subsidence_q50", "subsidence_pred"):
                if c in df.columns:
                    return pd.to_numeric(df[c], errors="coerce")
            for c in df.columns:
                if str(c).startswith("subsidence_"):
                    return pd.to_numeric(df[c], errors="coerce")
            return pd.Series(np.nan, index=df.index)

        if v == "spread":
            if ("subsidence_q90" in df.columns) and (
                "subsidence_q10" in df.columns
            ):
                hi = pd.to_numeric(df["subsidence_q90"], errors="coerce")
                lo = pd.to_numeric(df["subsidence_q10"], errors="coerce")
                return hi - lo
            return pd.Series(np.nan, index=df.index)

        if v in df.columns:
            return pd.to_numeric(df[v], errors="coerce")

        return pd.Series(np.nan, index=df.index)

    def _downsample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if int(n) <= 0 or len(df) <= int(n):
            return df
        return df.sample(int(n), random_state=0)

    def _as_points(self, df: pd.DataFrame) -> List[MapPoint]:
        lon = pd.to_numeric(df["lon"], errors="coerce")
        lat = pd.to_numeric(df["lat"], errors="coerce")
        val = pd.to_numeric(df["v"], errors="coerce")
    
        ok = lon.notna() & lat.notna() & val.notna()
        if not bool(ok.any()):
            return []
    
        out: List[MapPoint] = []
        dd = df.loc[ok, ["lat", "lon", "v"]]
        for la, lo, vv in dd.to_numpy(dtype=float):
            out.append(MapPoint(lat=float(la), lon=float(lo), v=float(vv)))
        return out

    def _layer_opts(
        self,
        lid: str,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        shape: str = "circle",
        radius: int = 6,
        opacity: float = 0.90,
        pulse: bool = False,
        enable_tip: bool = False,
    ) -> Dict[str, Any]:
        stroke = "#2E3191" if lid == "A" else "#F28620"

        d: Dict[str, Any] = {
            "stroke": stroke,
            "fillMode": "value",
            "shape": str(shape or "circle"),
            "radius": int(radius or 6),
            "opacity": float(opacity or 0.90),
            "pulse": bool(pulse),
            "enableTooltip": bool(enable_tip),
        }

        if vmin is not None and vmax is not None:
            d["vmin"] = float(vmin)
            d["vmax"] = float(vmax)

        return d

    def _shared_minmax(self, dd: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
        vv: List[np.ndarray] = []
        for df in dd.values():
            x = pd.to_numeric(df["v"], errors="coerce").to_numpy()
            x = x[np.isfinite(x)]
            if x.size:
                vv.append(x)

        if not vv:
            return 0.0, 1.0

        allv = np.concatenate(vv, axis=0)
        return float(np.min(allv)), float(np.max(allv))

    def _apply_legend(self, vmin: float, vmax: float, unit: str) -> None:
        try:
            self._v.set_legend(
                {"vmin": vmin, "vmax": vmax, "unit": unit}
            )
        except Exception:
            return

    def _update_step_range(self, step: int, step_max: int) -> None:
        st = int(step or 1)
        mx = int(step_max or 1)
        mx = max(1, mx)
        st = max(1, min(st, mx))

        self._tb.set_time_range(step_min=1, step_max=mx, step=st)

        if st != int(self._s.get(K_MAP_STEP, 1) or 1):
            self._s.set(K_MAP_STEP, st)

    def _coord_mode(self) -> str:
        cm = str(self._s.get(K_MAP_COORD_MODE, "auto") or "auto")
        cm = cm.strip().lower()
        if cm in ("auto", "lonlat", "degrees"):
            return "lonlat"
        if cm == "utm":
            return "utm"
        if cm == "epsg":
            return "epsg"
        return "lonlat"


    def _epsg(self, v: Any) -> Optional[int]:
        try:
            x = int(str(v).strip())
        except Exception:
            return None
        return x if x > 0 else None
    
    def _shape_for(self, shape: str, *, is_hot: bool) -> str:
        sh = str(shape or "auto").strip().lower()
        if sh and sh != "auto":
            return sh
        return "triangle" if bool(is_hot) else "circle"

    def _radius_for(self, px: int, *, is_hot: bool) -> int:
        r = int(px or 6)
        r = max(2, min(r, 18))
        if not bool(is_hot):
            return r
        return max(r + 2, int(r * 1.4))

    def _hotspot_points(self, df: pd.DataFrame) -> List[MapPoint]:
        if df is None or df.empty:
            return []

        topn = int(
            self._s.get(K_MAP_HOTSPOT_TOPN, 8) or 8
        )
        min_sep = float(
            self._s.get(K_MAP_HOTSPOT_MIN_SEP_KM, 2.0) or 2.0
        )
        metric = str(
            self._s.get(K_MAP_HOTSPOT_METRIC, "abs") or "abs"
        )
        q = float(
            self._s.get(K_MAP_HOTSPOT_QUANTILE, 0.98) or 0.98
        )

        pts = df[["lon", "lat", "v"]].copy()
        pts["lon"] = pd.to_numeric(
            pts["lon"], errors="coerce"
        )
        pts["lat"] = pd.to_numeric(
            pts["lat"], errors="coerce"
        )
        pts["v"] = pd.to_numeric(
            pts["v"], errors="coerce"
        )
        pts = pts.dropna(subset=["lon", "lat", "v"])
        if pts.empty:
            return []

        cfg = HotspotCfg(
            method="grid",
            metric=metric,
            quantile=q,
            max_n=topn,
            min_sep_km=min_sep,
        )

        hs = compute_hotspots(
            pts,
            cfg=cfg,
            coord_mode=self._coord_mode(),
        )
        if not hs:
            return []

        out: List[MapPoint] = []
        for h in hs:
            out.append(
                MapPoint(
                    lat=float(h.lat),
                    lon=float(h.lon),
                    v=float(h.v),
                    sid=int(h.rank),
                )
            )
        return out

    def _clear_map_layers(self) -> None:
        try:
            self._v.clear_layers()
            return
        except Exception:
            pass
        try:
            self._v.clear()
        except Exception:
            return

    def _set_status(self, msg: str, *, ok: bool) -> None:
        # Optional:  MapView can expose set_status()
        if hasattr(self._v, "set_status"):
            try:
                self._v.set_status(str(msg or ""), bool(ok))
                return
            except Exception:
                return

    def _active_layer_ids(self) -> List[str]:
        overlay = str(
            self._s.get(K_MAP_OVERLAY, "both") or "both"
        )
        pts_mode = str(
            self._s.get(K_MAP_POINTS_MODE, "all") or "all"
        )

        ids: List[str] = []

        def _add(city: str) -> None:
            if pts_mode in ("all", "hotspots_plus"):
                ids.append(city)
            if pts_mode in ("hotspots", "hotspots_plus"):
                ids.append(f"{city}_hot")

        if overlay in ("a", "both"):
            _add("A")
        if overlay in ("b", "both"):
            _add("B")

        return ids

    def _ensure_defaults(self) -> None:
        s = self._s
        with s.batch():
            for k, v in DEFAULTS.items():
                if s.get(k, None) is None:
                    s.set(k, v)

    def _map_keys(self) -> Set[str]:
        # canonical xfer.map.* keys
        return set(map_keys())

    def _as_set(self, keys: object) -> Set[str]:
        try:
            return set(keys or [])
        except Exception:
            return set()
        
    def _centroid(
        self,
        df: pd.DataFrame,
    ) -> Optional[Tuple[float, float]]:
        lat = pd.to_numeric(df["lat"], errors="coerce")
        lon = pd.to_numeric(df["lon"], errors="coerce")
        ok = lat.notna() & lon.notna()
        if not bool(ok.any()):
            return None
        return float(lat[ok].mean()), float(lon[ok].mean())
