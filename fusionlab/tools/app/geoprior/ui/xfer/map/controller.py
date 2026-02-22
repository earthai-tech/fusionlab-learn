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

import re
import numpy as np
import pandas as pd

from PyQt5.QtCore import QObject

from ....config.store import GeoConfigStore
from ...map.coord_utils import ensure_lonlat
from ...map.hotspots import (
    HotspotCfg,
    compute_hotspots,
    hotspots_points_with_pulse,
)
from ...view.factory import ViewFactory

from ...view.keys import (
    K_PLOT_KIND, 
    K_HEX_GRIDSIZE,
    K_HEX_METRIC,
    K_CONTOUR_BANDWIDTH, 
    K_CONTOUR_STEPS,
    K_CONTOUR_FILLED, 
    K_CONTOUR_LABELS,
    K_FILTER_ENABLE, 
    K_FILTER_V_MIN, 
    K_FILTER_V_MAX, 
    K_SPACE_MODE, 
    K_CONTOUR_METRIC, 
    
)
from ..insights import build_xfer_badges
from ..types import MapApi, MapPoint
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
    # K_MAP_PLAY,
    K_MAP_STEP_MAX,
    K_MAP_STEP_MODE,
    K_MAP_YEAR0,
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
    K_MAP_BASEMAP,    
    K_MAP_POINTS_MODE,
    K_MAP_MARKER_SHAPE,
    K_MAP_MARKER_SIZE,
    K_MAP_HOTSPOT_TOPN,
    K_MAP_HOTSPOT_MIN_SEP_KM,
    K_MAP_HOTSPOT_METRIC,
    K_MAP_HOTSPOT_QUANTILE,
    K_MAP_HOT_RINGS_ENABLE,
    K_MAP_HOT_RINGS_RADIUS_KM,
    K_MAP_HOT_RINGS_COUNT,
    K_MAP_ANIM_PULSE,
    K_MAP_ANIM_PLAY_MS,
    K_MAP_INSIGHT, 
    K_MAP_INTERACTION,
    K_MAP_INT_CELL_KM,
    K_MAP_INT_AGG,
    K_MAP_INT_DELTA,
    K_MAP_INT_HOT_ENABLE, 
    K_MAP_INT_INTENS_ENABLE, 
    K_MAP_INT_BUF_ENABLE, 
    K_MAP_INT_HOT_TOPN,
    K_MAP_INT_HOT_METRIC,
    K_MAP_INT_HOT_Q,
    K_MAP_INT_HOT_SEP,
    K_MAP_INT_BUF_K,
    K_MAP_INTERP_HTML,
    K_MAP_INTERP_TIP,
    K_MAP_RADAR_ENABLE,
    K_MAP_RADAR_TARGET,
    K_MAP_RADAR_ORDER,
    K_MAP_RADAR_DWELL_MS,
    K_MAP_RADAR_RADIUS_KM,
    K_MAP_RADAR_RINGS,
    K_MAP_LINKS_ENABLE,
    K_MAP_LINKS_MODE,
    K_MAP_LINKS_K,
    K_MAP_LINKS_MAX,
    K_MAP_LINKS_SHOW_DIST,
    K_MAP_A_EPSG, 
    K_MAP_B_EPSG, 

)
from .interpretation import interpret_transfer
from .interpretation import render_html, render_tip
from .interactions import (
    InteractionCfg,
    compute_interaction_layers,
)
from .interaction_extras import (
    IntExtrasCfg,
    compute_interaction_extras,
    build_hotspot_links,
    build_radar_centers,
)
from .toolbar import DatasetChoice, XferMapToolbar
from ..utils import (
    parse_xfer_csv_name,
    scan_xfer_results_root as scan_results_root,
    select_xfer_csv,
    decode_job_id
)

_SLUG_RE = re.compile(r"[^0-9a-zA-Z]+")

@dataclass(frozen=True)
class JobKey:
    kind: str
    job_id: str


def _as_int(v: object) -> Optional[int]:
    if v is None:
        return None
    try:
        x = int(str(v).strip())
    except Exception:
        return None
    return x if x > 0 else None


def _slug_city(name: str) -> str:
    s = str(name or "").strip().lower()
    s = _SLUG_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _epsg_from_city_meta(
    meta: object,
    city: str,
) -> Optional[int]:
    if not isinstance(meta, dict):
        return None

    c = str(city or "").strip()
    if not c:
        return None

    def _pick(v: object) -> Optional[int]:
        if isinstance(v, dict):
            for kk in (
                "utm_epsg",
                "src_epsg",
                "epsg",
                "crs_epsg",
                "epsg_code",
            ):
                e = _as_int(v.get(kk, None))
                if e:
                    return e
            return None
        return _as_int(v)

    cities = meta.get("cities", None)
    if isinstance(cities, dict):
        aliases = meta.get("aliases", None)
        aliases = aliases if isinstance(aliases, dict) else {}

        k = _slug_city(c)
        v = cities.get(k, None)
        if v is None:
            k2 = aliases.get(k, None)
            if k2:
                v = cities.get(k2, None)

        e = _pick(v)
        if e:
            return e

    for k in (c, c.lower(), c.upper()):
        e = _pick(meta.get(k, None))
        if e:
            return e

    return None


class XferMapController(QObject):
    """
    Wires store + toolbar + map view.
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

        # self._view_factory = ViewFactory(
        #     store,
        #     key_fn=map_view_key,
        #     radius_key="map.view.marker_size"
        # )
        self._view_factory = ViewFactory(
            store,
            # xfer uses view.* keys directly
            key_fn=None,
            # let factory read xfer slider size
            radius_key=K_MAP_MARKER_SIZE,
        )

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
        # ids = self._active_layer_ids()
        try:
            # self._v.fit_layers(ids)
            # Pro UX: fit what the user is currently seeing
            # (checked/visible in the Leaflet layer control).
            self._v.fit_layers(["visible"])
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
            self._maybe_autofill_epsg()
            self._sync_toolbar_all()
            self._render_from_store()
            return
        
        # FIX 1: Listen to both map keys AND view keys
        if (ch & self._map_keys()) or (ch & self._view_keys()):
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
            s.set("xfer.city_a", st.get("city_a") or "")
            s.set("xfer.city_b", st.get("city_b") or "")

            s.set(K_MAP_A_JOB_KIND, a_jk.kind if a_jk else None)
            s.set(K_MAP_A_JOB_ID, a_jk.job_id if a_jk else None)
            s.set(K_MAP_A_FILE, st.get("file_a") or "")
            
            s.set(K_MAP_B_JOB_KIND, b_jk.kind if b_jk else None)
            s.set(K_MAP_B_JOB_ID, b_jk.job_id if b_jk else None)
            s.set(K_MAP_B_FILE, st.get("file_b") or "")
            
            s.set(K_MAP_SPLIT, st.get("split"))
            s.set(K_MAP_VALUE, st.get("value"))
            s.set(K_MAP_OVERLAY, st.get("overlay"))
            s.set(K_MAP_SHARED, bool(st.get("shared")))
            s.set(K_MAP_TIME_MODE, st.get("time_mode"))
            s.set(K_MAP_STEP, int(st.get("step") or 1))
            
            s.set(K_MAP_POINTS_MODE, st.get("points_mode") or "all")
            s.set(K_MAP_MARKER_SHAPE, st.get("marker_shape") or "auto")
            s.set(K_MAP_MARKER_SIZE, int(st.get("marker_size") or 6))
            s.set(K_MAP_HOTSPOT_TOPN, int(st.get("hotspot_top_n") or 8))
            s.set(K_MAP_ANIM_PULSE, bool(st.get("pulse")))
            s.set(K_MAP_ANIM_PLAY_MS, int(st.get("play_ms") or 320))
            s.set(K_MAP_INSIGHT, bool(st.get("insight")))
        
        self._maybe_autofill_epsg(
            city_a=st.get("city_a"),
            city_b=st.get("city_b"),
        )
        try:
            self._tb.set_play_ms(int(st.get("play_ms") or 320))
        except Exception:
            pass

    # -------------------------
    # Index & Toolbar Sync
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

    def _sync_toolbar_all(self) -> None:
        self._sync_city_choices()
        self._sync_job_choices()
        self._sync_file_choices()
        self._sync_value_choices()
        self._sync_toolbar_from_store(set())

    def _sync_toolbar_from_store(self, _keys: Set[str]) -> None:
        s = self._s
        a_city = str(s.get("xfer.city_a", "") or "")
        b_city = str(s.get("xfer.city_b", "") or "")
        
        if a_city:
            self._tb.set_current_data(self._tb.cmb_city_a, a_city)
        if b_city:
            self._tb.set_current_data(self._tb.cmb_city_b, b_city)
        
        self._sync_job_choices()
        self._sync_file_choices()
        
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

        a_file = str(s.get(K_MAP_A_FILE, "") or "")
        b_file = str(s.get(K_MAP_B_FILE, "") or "")
        
        if a_file:
            self._tb.set_current_data(self._tb.cmb_file_a, a_file)
        if b_file:
            self._tb.set_current_data(self._tb.cmb_file_b, b_file)
        
        step = int(s.get(K_MAP_STEP, 1) or 1)
        self._tb.set_time_range(step_min=1, step_max=1, step=step)

        pts_mode = str(s.get(K_MAP_POINTS_MODE, "all") or "all")
        mk_shape = str(s.get(K_MAP_MARKER_SHAPE, "auto") or "auto")
        mk_size = int(s.get(K_MAP_MARKER_SIZE, 6) or 6)
        topn = int(s.get(K_MAP_HOTSPOT_TOPN, 8) or 8)
        pulse = bool(s.get(K_MAP_ANIM_PULSE, True))
        play_ms = int(s.get(K_MAP_ANIM_PLAY_MS, 320) or 320)

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

        a_fb = str(a[0].data) if a else ""
        b_fb = str(b[0].data) if b else ""

        self._auto_pick_xfer_file(
            target=a_city, other=b_city,
            job_id=str(a_id) if a_id else None,
            file_key=K_MAP_A_FILE, fallback=a_fb,
        )
        self._auto_pick_xfer_file(
            target=b_city, other=a_city,
            job_id=str(b_id) if b_id else None,
            file_key=K_MAP_B_FILE, fallback=b_fb,
        )

    def _auto_pick_xfer_file(
        self,
        *,
        target: str,
        other: str,
        job_id: Optional[str],
        file_key: str,
        fallback: str,
    ) -> None:
        root = self._s.get("results_root", "") or ""
        rp = Path(str(root)).expanduser()
        if not rp.exists():
            return

        tgt = str(target or "").strip()
        src = str(other or "").strip()
        if not tgt or not src:
            if not str(self._s.get(file_key, "") or ""):
                if fallback:
                    self._s.set(file_key, fallback)
            return

        split = str(self._s.get(K_MAP_SPLIT, "val") or "val")
        cur = str(self._s.get(file_key, "") or "")
        
        try:
            if cur and not Path(cur).expanduser().exists():
                cur = ""
        except Exception:
            cur = ""
            
        want_kind = self._want_xfer_kind(cur)
        meta = None
        if cur:
            try:
                meta = parse_xfer_csv_name(Path(cur).name)
            except Exception:
                meta = None

        pref_strategy = meta.strategy if meta else None
        pref_calib = meta.calibration if meta else None
        pref_rescale = meta.rescale_mode if meta else None

        if cur and self._xfer_file_ok(
            cur, src=src, tgt=tgt, split=split, kind=want_kind
        ):
            return

        hit = select_xfer_csv(
            rp,
            city_a=tgt, city_b=src, target_city=tgt,
            kind=want_kind, strategy=pref_strategy,
            calibration=pref_calib, rescale_mode=pref_rescale,
            split=split, job_id=job_id,
        )
        if hit is not None:
            self._s.set(file_key, str(hit))
            return

        if (not cur) and fallback:
            self._s.set(file_key, fallback)

    def _want_xfer_kind(self, fp: str) -> str:
        p = Path(str(fp or "")).expanduser()
        meta = parse_xfer_csv_name(p.name)
        if meta is not None:
            return str(meta.kind or "eval")
        if "future" in str(fp).lower():
            return "future"
        return "eval"

    def _xfer_file_ok(
        self, fp: str, *, src: str, tgt: str, split: str, kind: str
    ) -> bool:
        p = Path(str(fp or "")).expanduser()
        if not p.exists():
            return False
        meta = parse_xfer_csv_name(p.name)
        if meta is None:
            return False
        if meta.src_city.lower() != str(src).lower():
            return False
        if meta.tgt_city.lower() != str(tgt).lower():
            return False
        sp = str(split or "").strip().lower()
        if sp and meta.split != sp:
            return False
        kd = str(kind or "eval").strip().lower()
        if kd and meta.kind != kd:
            return False
        return True

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
        def _sort_key(j: Any) -> str:
            _, jname = decode_job_id(getattr(j, "job_id", ""))
            return jname or str(getattr(j, "job_id", ""))
        
        jobs = sorted(jobs, key=_sort_key, reverse=True)
        out: List[DatasetChoice] = []
        for j in jobs:
            kind = str(getattr(j, "kind", "") or "")
            jid = str(getattr(j, "job_id", "") or "")
            label = f"{kind} {jid}"
            if kind.strip().lower() == "xfer":
                _, jname = decode_job_id(jid)
                label = jname or jid
            out.append(
                DatasetChoice(label, JobKey(j.kind, j.job_id))
            )
        return out

    def _file_items(self, city: str, job: Any) -> List[DatasetChoice]:
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
        self, *, pts_out: Dict[str, pd.DataFrame], overlay: str
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
            results_root=rp, city_a=a_city, city_b=b_city,
            split=split, a_lat=a_ctr[0], a_lon=a_ctr[1],
            b_lat=b_ctr[0], b_lon=b_ctr[1],
            want_ab=want_ab, want_ba=want_ba,
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
        try:
            bm = str(
                self._s.get(K_MAP_BASEMAP, "osm") or "osm"
            )
            self._v.set_basemap(bm)
        except Exception:
            pass

        self._clear_map_layers()

        overlay = str(self._s.get(K_MAP_OVERLAY, "both") or "both")
        mode = str(
            self._s.get(K_MAP_TIME_MODE, "forecast_step") or "forecast_step"
        )
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
        ok_any = False
        last_err = ""
        step_max_seen = 1
        year0_seen = None
        
        for lid, name, fp in layers:
            df, ok, msg, u, step_max, year0 = self._load_points(
                fp=fp, lid=lid, value=value,
                time_mode=mode, step=step,
            )
            try:
                step_max_seen = max(step_max_seen, int(step_max or 1))
            except Exception:
                pass
            
            if year0 is not None:
                if year0_seen is None or year0 < year0_seen:
                    year0_seen = int(year0)
 
            if not ok:
                last_err = msg or last_err
                continue
            
            ok_any = True
            unit = unit or (u or "")
            pts_out[lid] = df
        
        # self._update_step_range(step, step_max_seen)
        self._update_step_range(
            step,
            step_max_seen,
            time_mode=mode,
            year0=year0_seen,
        )
        
        if not ok_any:
            if last_err:
                self._set_status(last_err, ok=False)
            else:
                self._set_status("No map dataset to plot.", ok=False)
            return
        
        if not pts_out:
            self._set_status("No map dataset to plot.", ok=False)
            return

        shared = bool(self._s.get(K_MAP_SHARED, True))
        vmin = vmax = None
        if shared:
            vmin, vmax = self._shared_minmax(pts_out)
            self._apply_legend(vmin, vmax, unit)
        
        # --- RENDER MAIN LAYERS via ViewFactory ---
        pts_mode = str(self._s.get(K_MAP_POINTS_MODE, "all") or "all")
        opacity = float(self._s.get(K_MAP_OPACITY, 0.90) or 0.90)
        
        for lid, df in pts_out.items():
            base_name = "City A" if lid == "A" else "City B"

            # 1. Main Visualization (Scatter/Hex/Contour)
            if pts_mode in ("all", "hotspots_plus"):
                payload = self._view_factory.build_layer(df, lid)
                if payload:
                    # Sync shared min/max
                    if shared and vmin is not None:
                        payload.opts["vmin"] = float(vmin)
                        payload.opts["vmax"] = float(vmax)
                    
                    # FIX 2: Force opacity from the main map slider
                    payload.opts["opacity"] = opacity
                    
                    try:
                        self._v.set_layer(
                            layer_id=lid,
                            name=base_name,
                            payload=payload,
                        )
                    except Exception:
                        pass

            # 2. Hotspots Overlay
            if pts_mode in ("hotspots", "hotspots_plus"):
                hp = self._hotspot_points(df, name=base_name)
                if hp:
                    try:
                        mk_shape = str(
                            self._s.get(K_MAP_MARKER_SHAPE, "auto")
                        )
                        mk_size = int(
                            self._s.get(K_MAP_MARKER_SIZE, 6) or 6
                        )
                        pulse_on = bool(self._s.get(K_MAP_ANIM_PULSE, True))
                        rings_on = bool(self._s.get(K_MAP_HOT_RINGS_ENABLE, False))
                        ring_km = float(self._s.get(K_MAP_HOT_RINGS_RADIUS_KM, 0.0) or 0.0)
                        ring_n = int(self._s.get(K_MAP_HOT_RINGS_COUNT, 0) or 0)
                        opts=self._layer_opts(
                            lid,
                            vmin=vmin,
                            vmax=vmax,
                            shape=self._shape_for(mk_shape, is_hot=True),
                            radius=self._radius_for(mk_size, is_hot=True),
                            opacity=opacity,
                            pulse=pulse_on,
                            force_html=True,
                            rings=rings_on,
                            ring_radius_km=ring_km,
                            ring_count=ring_n,
                            enable_tip=True,
                        )
                        self._v.set_layer(
                            layer_id=f"{lid}_hot",
                            name=f"{base_name} hotspots",
                            points=hp,
                            opts=opts
                        )
                    except:
                        pass
                    
        self._render_insight(pts_out=pts_out, overlay=overlay)
        self._render_interactions(pts_out=pts_out, unit=unit)
        map_sig = self._render_interaction_extras(
            pts_out=pts_out, unit=unit
        )

        hs_a = self._hotspot_df(pts_out.get("A"), name="City A")
        hs_b = self._hotspot_df(pts_out.get("B"), name="City B")

        self._render_hotspot_analytics(
            hs_a=hs_a, hs_b=hs_b, overlay=overlay
        )
        self._push_interpretation(map_sig=map_sig)

    def _load_points(
        self, *, fp: str, lid: str, value: str,
        time_mode: str, step: int,
    ) -> Tuple[pd.DataFrame, bool, str, str, int, Optional[int]]:
        p = Path(str(fp)).expanduser()
        if not p.exists():
            # return (pd.DataFrame(), False, f"Missing file: {p.name}", "", 1)
            return (
                pd.DataFrame(),
                False,
                f"Missing file: {p.name}",
                "",
                1,
                None,
            )
        
        df = self._read_cached(p)
        if df is None or df.empty:
            # return (pd.DataFrame(), False, f"Empty CSV: {p.name}", "", 1)
            return (
                pd.DataFrame(),
                False,
                f"Empty CSV: {p.name}",
                "",
                1,
                None,
            )

        step_max = self._infer_step_max(df, time_mode)
        year0 = self._infer_year0(df, time_mode)
        sl, label = self._slice_time(df, time_mode, step)
        if sl.empty:
            return (
                pd.DataFrame(), False,
                f"No rows for {label} in {p.name}",
                "",
                step_max,
                year0,
             )
        
        v = self._compute_value(sl, value)
        out = pd.DataFrame({
            "lon": sl.get("coord_x", np.nan),
            "lat": sl.get("coord_y", np.nan),
            "v": v,
        })

        max_pts = int(self._s.get(K_MAP_MAX_POINTS, 20000) or 20000)
        out = self._downsample(out, max_pts)

        city_key = "xfer.city_a" if lid == "A" else "xfer.city_b"
        city_name = str(self._s.get(city_key, "") or "")

        epsg_override_key = K_MAP_A_EPSG if lid == "A" else K_MAP_B_EPSG
        manual_specific = self._epsg(self._s.get(epsg_override_key, None))
        
        meta_specific = None
        if not manual_specific:
            meta_specific = self._city_epsg(city_name)
            
        global_default = self._epsg(self._s.get(K_MAP_UTM_EPSG, None))
        if not global_default:
             global_default = self._epsg(self._s.get("utm_epsg", None))

        target_epsg = manual_specific or meta_specific or global_default
        cm = self._coord_mode()
        
        out, ok, msg = ensure_lonlat(
            out, mode=cm,
            utm_epsg=target_epsg,
            src_epsg=target_epsg,
        )
        if not ok:
            hint = f" (Target EPSG: {target_epsg or 'None'})"
            return (pd.DataFrame(), False, msg + hint, "", step_max)

        if out.empty:
            return (
                pd.DataFrame(), False,
                "No mappable points after CRS conversion. "
                "Check Coords + EPSG (UTM/src).",
                # "", step_max,
                "",
                step_max,
                year0,
            )

        unit = ""
        if "subsidence_unit" in df.columns:
            try:
                unit = str(df["subsidence_unit"].iloc[0])
            except Exception:
                unit = ""

        return out, True, "", unit, step_max, year0

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
        self, df: pd.DataFrame, mode: str, step: int
    ) -> Tuple[pd.DataFrame, str]:
        m = str(mode or "forecast_step")
        st = int(step or 1)
        # if m == "year" and "coord_t" in df.columns:
        #     yrs = sorted(pd.unique(df["coord_t"]))
        #     if not yrs:
        #         return df.iloc[0:0], "year"
        #     idx = max(0, min(st - 1, len(yrs) - 1))
        #     y = yrs[idx]
        #     return df[df["coord_t"] == y], f"year={y}"
        if m == "year" and "coord_t" in df.columns:
            s = pd.to_numeric(df["coord_t"], errors="coerce")
            s = s[np.isfinite(s)]
            if s.empty:
                return df.iloc[0:0], "year"

            yrs = sorted(pd.unique(s.astype(int)))
            if not yrs:
                return df.iloc[0:0], "year"

            idx = max(0, min(st - 1, len(yrs) - 1))
            y = int(yrs[idx])

            sel = pd.to_numeric(df["coord_t"], errors="coerce")
            sel = sel.astype("Int64")
            return df.loc[sel == y], f"year={y}"
        
        if "forecast_step" in df.columns:
            return df[df["forecast_step"] == st], f"step={st}"
        return df, "all"

    def _infer_step_max(self, df: pd.DataFrame, mode: str) -> int:
        m = str(mode or "forecast_step")
        if m == "year" and "coord_t" in df.columns:
            # yrs = pd.unique(df["coord_t"])
            # return int(max(1, len(yrs)))
            s = pd.to_numeric(df["coord_t"], errors="coerce")
            s = s[np.isfinite(s)]
            if s.empty:
                return 1
            yrs = pd.unique(s.astype(int))
            
            return int(max(1, len(yrs)))

        if "forecast_step" in df.columns:
            try:
                return int(pd.to_numeric(df["forecast_step"]).max())
            except Exception:
                return 1
        return 1
    
    def _infer_year0(
        self,
        df: pd.DataFrame,
        mode: str,
    ) -> Optional[int]:
        m = str(mode or "forecast_step").strip().lower()
        if m != "year":
            return None
        if "coord_t" not in df.columns:
            return None

        s = df["coord_t"]
        if s is None or s.empty:
            return None


        try:
            yy = pd.to_numeric(s, errors="coerce")
            yy = yy[np.isfinite(yy)]
            if yy.size:
                # y0 = int(np.min(yy))
                # return y0 if y0 > 0 else None
                y_min = int(np.min(yy))
                y_max = int(np.max(yy))

                # If it looks like a YEAR column, return it directly.
                # (Avoid pandas treating 2020 as ns since epoch => 1970.)
                if 1500 <= y_min <= 2500 and 1500 <= y_max <= 2500:
                    return int(y_min)
                
        except Exception:
            # return None
            pass

        # Fallback: parse as datetime, but only after coercing to string
        # so "2020" becomes year 2020, not ns since epoch.
        try:
            ss = s.astype(str).str.strip()
            dt = pd.to_datetime(ss, errors="coerce")
            if dt.notna().any():
                y0 = int(dt.min().year)
                return y0 if y0 > 0 else None
        except Exception:
            pass
 
        return None

    def _compute_value(
        self, df: pd.DataFrame, value: str
    ) -> pd.Series:
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
        
        has_sid = "sid" in df.columns
        has_tip = "tip" in df.columns
        
        out: List[MapPoint] = []
        cols = ["lat", "lon", "v"]
        if has_sid: cols.append("sid")
        if has_tip: cols.append("tip")
        
        dd = df.loc[ok, cols]
        for row in dd.itertuples(index=False, name=None):
            la = float(row[0])
            lo = float(row[1])
            vv = float(row[2])
            sid = 0
            tip = None
            if has_sid and has_tip:
                sid = int(row[3])
                tip = row[4]
            elif has_sid:
                sid = int(row[3])
            elif has_tip:
                tip = row[3]
            
            out.append(MapPoint(
                lat=la, lon=lo, v=vv, sid=sid, tip=tip
            ))
        return out

    def _layer_opts(
        self, lid: str, *, 
        vmin: Optional[float] = None,
        vmax: Optional[float] = None, 
        shape: str = "circle",
        radius: int = 6, 
        opacity: float = 0.90,
        pulse: bool = False,
        enable_tip: bool = False,
        force_html: bool = False,
        rings: bool = False,
        ring_radius_km: float = 0.0,
        ring_count: int = 0,
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
            "forceHtml": bool(force_html),
            "rings": bool(rings),
            "ringRadiusKm": float(ring_radius_km or 0.0),
            "ringCount": int(ring_count or 0),
        }
        if vmin is not None and vmax is not None:
            d["vmin"] = float(vmin)
            d["vmax"] = float(vmax)
        return d

    def _shared_minmax(
        self, dd: Dict[str, pd.DataFrame]
    ) -> Tuple[float, float]:
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

    def _update_step_range(
        self,
        step: int,
        step_max: int,
        *,
        time_mode: str,
        year0: Optional[int],
    ) -> None:
        st = int(step or 1)
        mx = max(1, int(step_max or 1))
        st = max(1, min(st, mx))

        self._tb.set_time_range(
            step_min=1,
            step_max=mx,
            step=st,
        )

        with self._s.batch():
            self._s.set(K_MAP_STEP_MAX, mx)

            tm = str(time_mode or "").strip().lower()
            if tm == "year":
                self._s.set(K_MAP_STEP_MODE, "year")
                if year0 is not None:
                    self._s.set(K_MAP_YEAR0, int(year0))
            else:
                self._s.set(K_MAP_STEP_MODE, "step")

            if st != int(self._s.get(K_MAP_STEP, 1) or 1):
                self._s.set(K_MAP_STEP, st)

    def _coord_mode(self) -> str:
        cm = str(self._s.get(K_MAP_COORD_MODE, "auto") or "auto")
        cm = cm.strip().lower()
        if cm in ("auto", "detect"): return "auto"
        if cm in ("lonlat", "degrees"): return "lonlat"
        if cm == "utm": return "utm"
        if cm == "epsg": return "epsg"
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


    def _hotspot_points(
        self,
        df: pd.DataFrame,
        *,
        name: str = "",
    ) -> List[list]:
        if df is None or df.empty:
            return []
    
        topn = int(self._s.get(K_MAP_HOTSPOT_TOPN, 8) or 8)
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
        for c in ("lon", "lat", "v"):
            pts[c] = pd.to_numeric(pts[c], errors="coerce")
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
    
        pref = str(name or "").strip()
        if pref:
            pref = f"{pref} hotspot"
    
        return hotspots_points_with_pulse(
            hs,
            tip_prefix=pref,
        )
    
    def _hotspot_df(
        self, df: pd.DataFrame, *, name: str
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["lon", "lat", "v", "sid", "score", "tip"]
            )
        
        topn = int(self._s.get(K_MAP_HOTSPOT_TOPN, 8) or 8)
        min_sep = float(
            self._s.get(K_MAP_HOTSPOT_MIN_SEP_KM, 2.0) or 2.0
        )
        metric = str(self._s.get(K_MAP_HOTSPOT_METRIC, "abs") or "abs")
        q = float(self._s.get(K_MAP_HOTSPOT_QUANTILE, 0.98) or 0.98)
        
        pts = df[["lon", "lat", "v"]].copy()
        for c in ("lon", "lat", "v"):
            pts[c] = pd.to_numeric(pts[c], errors="coerce")
        pts = pts.dropna(subset=["lon", "lat", "v"])
        if pts.empty:
            return pd.DataFrame(
                columns=["lon", "lat", "v", "sid", "score", "tip"]
            )
        
        cfg = HotspotCfg(
            method="grid", metric=metric, quantile=q,
            max_n=topn, min_sep_km=min_sep,
        )
        hs = compute_hotspots(
            pts, cfg=cfg, coord_mode=self._coord_mode()
        )
        if not hs:
            return pd.DataFrame(
                columns=["lon", "lat", "v", "sid", "score", "tip"]
            )
        
        rows: List[Dict[str, Any]] = []
        for h in hs:
            sc = float(getattr(h, "score", abs(float(h.v))))
            tip = (
                f"{name} hotspot\n"
                f"rank={int(h.rank)}\n"
                f"v={float(h.v):.3g}\n"
                f"score={sc:.3g}"
            )
            rows.append({
                "lon": float(h.lon), "lat": float(h.lat),
                "v": float(h.v), "sid": int(h.rank),
                "score": sc, "tip": tip,
            })
        return pd.DataFrame(rows)
    
    def _render_hotspot_analytics(
        self, *, hs_a: pd.DataFrame, hs_b: pd.DataFrame, overlay: str
    ) -> None:
        s = self._s
        v = self._v
        
        radar_on = bool(s.get(K_MAP_RADAR_ENABLE, False))
        if radar_on and hasattr(v, "set_radar"):
            dwell = int(s.get(K_MAP_RADAR_DWELL_MS, 520) or 520)
            rkm = float(s.get(K_MAP_RADAR_RADIUS_KM, 8.0) or 8.0)
            rings = int(s.get(K_MAP_RADAR_RINGS, 3) or 3)
            target = str(s.get(K_MAP_RADAR_TARGET, "overlay") or "overlay")
            order = str(s.get(K_MAP_RADAR_ORDER, "score") or "score")
            
            centers = build_radar_centers(
                hs_a, hs_b, target=target,
                overlay=overlay, order=order,
            )
            v.set_radar(
                "R_HOT", centers,
                opts={"dwellMs": dwell, "radiusKm": rkm, "rings": rings},
            )
        else:
            if hasattr(v, "clear_radar"):
                v.clear_radar("R_HOT")
        
        links_on = bool(s.get(K_MAP_LINKS_ENABLE, False))
        if links_on and hasattr(v, "set_links"):
            mode = str(s.get(K_MAP_LINKS_MODE, "nearest") or "nearest")
            k = int(s.get(K_MAP_LINKS_K, 1) or 1)
            mx = int(s.get(K_MAP_LINKS_MAX, 12) or 12)
            show_dist = bool(s.get(K_MAP_LINKS_SHOW_DIST, True))
            
            links = build_hotspot_links(
                hs_a, hs_b, mode=mode, k=k,
                max_links=mx, show_dist=show_dist,
            )
            v.set_links(
                "L_HOT", "Hotspot links", links,
                opts={"arrow": True, "label": show_dist},
            )
        else:
            if hasattr(v, "clear_links"):
                v.clear_links("L_HOT")

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
        
        if overlay in ("a", "both"): _add("A")
        if overlay in ("b", "both"): _add("B")
            
        mode = str(
            self._s.get(K_MAP_INTERACTION, "none") or "none"
        ).strip().lower()
        
        if mode in ("zones", "partition"):
            ids.extend(["I_AONLY", "I_BONLY", "I_INTER"])
        elif mode == "a_only":
            ids.append("I_AONLY")
        elif mode == "b_only":
            ids.append("I_BONLY")
        elif mode == "union":
            ids.append("I_UNION")
        elif mode in ("intersection", "inter"):
            ids.append("I_INTER")
        elif mode == "delta":
            ids.append("I_DELTA")
            
        if bool(self._s.get(K_MAP_INT_HOT_ENABLE, False)):
            ids.append("I_DHOT")
        if bool(self._s.get(K_MAP_INT_INTENS_ENABLE, False)):
            ids.append("I_INTENS")
        if bool(self._s.get(K_MAP_INT_BUF_ENABLE, False)):
            ids.append("I_BUFINT")
        
        return ids

    def _ensure_defaults(self) -> None:
        s = self._s
        with s.batch():
            for k, v in DEFAULTS.items():
                if s.get(k, None) is None:
                    s.set(k, v)
            m = str(s.get(K_CONTOUR_METRIC, "") or "").strip()
            if not m:
                s.set(K_CONTOUR_METRIC, "value")
    
    def _map_keys(self) -> Set[str]:
        return set(map_keys())

    #  Helper for new keys ---
    def _view_keys(self) -> Set[str]:

        return {
                # Plot Type
                K_PLOT_KIND, 
                # Hexbin
                K_HEX_GRIDSIZE, 
                K_HEX_METRIC,
                # Contour
                K_CONTOUR_BANDWIDTH,
                K_CONTOUR_STEPS, 
                K_CONTOUR_FILLED, 
                K_CONTOUR_LABELS,
                # Filters (Fixes "filters not working")
                K_FILTER_ENABLE,
                K_FILTER_V_MIN,
                K_FILTER_V_MAX, 
                K_SPACE_MODE, 
                K_CONTOUR_METRIC
            }

    def _as_set(self, keys: object) -> Set[str]:
        try:
            return set(keys or [])
        except Exception:
            return set()
        
    def _centroid(
        self, df: pd.DataFrame
    ) -> Optional[Tuple[float, float]]:
        lat = pd.to_numeric(df["lat"], errors="coerce")
        lon = pd.to_numeric(df["lon"], errors="coerce")
        ok = lat.notna() & lon.notna()
        if not bool(ok.any()):
            return None
        return float(lat[ok].mean()), float(lon[ok].mean())

    def _render_interactions(
        self, *, pts_out: Dict[str, pd.DataFrame], unit: str
    ) -> None:
        for lid in (
            "I_AONLY", "I_BONLY", "I_UNION", "I_INTER", "I_DELTA"
        ):
            try:
                self._v.clear_layer(lid)
            except Exception:
                pass
        
        a_df = pts_out.get("A", None)
        b_df = pts_out.get("B", None)
        if a_df is None or b_df is None:
            return
        
        mode = str(self._s.get(K_MAP_INTERACTION, "none") or "none")
        cell_km = float(self._s.get(K_MAP_INT_CELL_KM, 2.0) or 2.0)
        agg = str(self._s.get(K_MAP_INT_AGG, "mean") or "mean")
        delt = str(self._s.get(K_MAP_INT_DELTA, "a_minus_b") or "a_minus_b")
        
        if mode.strip().lower() in ("none", "", "off"):
            return
        
        mk_size = int(self._s.get(K_MAP_MARKER_SIZE, 6) or 6)
        opacity = float(self._s.get(K_MAP_OPACITY, 0.90) or 0.90)
        
        cfg = InteractionCfg(
            mode=mode, cell_km=cell_km, agg=agg, delta=delt
        )
        layers = compute_interaction_layers(
            a_df, b_df, cfg=cfg, radius=mk_size, opacity=opacity
        )
        if not layers:
            return
        
        for sp in layers:
            if sp.legend is not None:
                leg = dict(sp.legend)
                leg["unit"] = unit
                try:
                    self._v.set_legend(leg)
                except Exception:
                    pass
        
        for sp in layers:
            if sp.df is None or sp.df.empty:
                continue
            try:
                self._v.set_layer(
                    layer_id=sp.layer_id, name=sp.name,
                    points=self._as_points(sp.df), opts=sp.opts,
                )
            except Exception:
                pass
            
    def _render_interaction_extras(
        self, *, pts_out: Dict[str, pd.DataFrame], unit: str
    ) -> Optional[Dict[str, Any]]:
        for lid in ("I_DHOT", "I_INTENS", "I_BUFINT"):
            try:
                self._v.clear_layer(lid)
            except Exception:
                pass

        a_df = pts_out.get("A")
        b_df = pts_out.get("B")
        if a_df is None or b_df is None:
            return None

        cfg = IntExtrasCfg(
            cell_km=float(self._s.get(K_MAP_INT_CELL_KM, 2.0) or 2.0),
            agg=str(self._s.get(K_MAP_INT_AGG, "mean") or "mean"),
            delta=str(
                self._s.get(K_MAP_INT_DELTA, "a_minus_b") or "a_minus_b"
            ),
            hot_enable=bool(self._s.get(K_MAP_INT_HOT_ENABLE, False)),
            hot_topn=int(self._s.get(K_MAP_INT_HOT_TOPN, 8) or 8),
            hot_metric=str(self._s.get(K_MAP_INT_HOT_METRIC, "abs") or "abs"),
            hot_quantile=float(self._s.get(K_MAP_INT_HOT_Q, 0.98) or 0.98),
            hot_min_sep_km=float(self._s.get(K_MAP_INT_HOT_SEP, 2.0) or 2.0),
            intens_enable=bool(self._s.get(K_MAP_INT_INTENS_ENABLE, False)),
            buf_enable=bool(self._s.get(K_MAP_INT_BUF_ENABLE, False)),
            buf_k=int(self._s.get(K_MAP_INT_BUF_K, 1) or 1),
        )

        mk = int(self._s.get(K_MAP_MARKER_SIZE, 6) or 6)
        op = float(self._s.get(K_MAP_OPACITY, 0.90) or 0.90)

        layers = compute_interaction_extras(
            a_df, b_df, cfg=cfg, coord_mode=self._coord_mode(),
            radius=mk, opacity=op,
        )
        map_sig = self._summarize_map_sig(layers)

        for p in layers:
            leg = p.get("legend")
            if isinstance(leg, dict):
                leg = dict(leg)
                leg["unit"] = unit
                try:
                    self._v.set_legend(leg)
                except Exception:
                    pass

            df = p.get("df")
            if df is None or getattr(df, "empty", True):
                continue
            try:
                self._v.set_layer(
                    layer_id=str(p.get("id")),
                    name=str(p.get("name")),
                    points=self._as_points(df),
                    opts=dict(p.get("opts") or {}),
                )
            except Exception:
                pass

        return map_sig

    def _summarize_map_sig(
        self, layers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        sig: Dict[str, Any] = {}
        for p in layers or []:
            lid = str(p.get("id") or "")
            df = p.get("df", None)
            if df is None or getattr(df, "empty", True):
                continue

            if lid == "I_DHOT":
                sig["dhot_n"] = int(len(df))
                if "v" in df.columns:
                    vv = pd.to_numeric(df["v"], errors="coerce").to_numpy()
                    vv = vv[np.isfinite(vv)]
                    if vv.size:
                        sig["dhot_abs_max"] = float(np.max(np.abs(vv)))

            if lid == "I_INTENS":
                if "v" in df.columns:
                    vv = pd.to_numeric(df["v"], errors="coerce").to_numpy()
                    vv = vv[np.isfinite(vv)]
                    if vv.size:
                        sig["overlap_mean"] = float(np.mean(vv))

            if lid == "I_BUFINT":
                sig["buf_n"] = int(len(df))
        return sig
    
    def _push_interpretation(
        self, *, map_sig: Optional[Dict[str, Any]]
    ) -> None:
        split = str(self._s.get(K_MAP_SPLIT, "val") or "val")
        shared = bool(self._s.get(K_MAP_SHARED, True))
        ov = str(self._s.get(K_MAP_OVERLAY, "both") or "both")

        strat = "map"
        if ov in ("a", "both"):
            ak = self._s.get(K_MAP_A_JOB_KIND, None)
            if ak:
                strat = str(ak).strip().lower()
        if strat == "map" and ov in ("b", "both"):
            bk = self._s.get(K_MAP_B_JOB_KIND, None)
            if bk:
                strat = str(bk).strip().lower()

        run = {
            "strategy": strat, "split": split,
            "rescale_mode": "shared" if shared else "auto",
            "calibration": "na",
        }
        p = interpret_transfer(run, map_sig=map_sig)
        html = render_html(p)
        tip = render_tip(p)

        with self._s.batch():
            self._s.set(K_MAP_INTERP_HTML, html)
            self._s.set(K_MAP_INTERP_TIP, tip)

    def _maybe_autofill_epsg(
        self, *, city_a: Optional[str] = None,
        city_b: Optional[str] = None,
    ) -> None:
        if self._has_epsg():
            return
        s = self._s
        a = str(city_a or s.get("xfer.city_a", "") or "").strip()
        b = str(city_b or s.get("xfer.city_b", "") or "").strip()
        if not a and not b:
            return
        
        ea = self._city_epsg(a)
        eb = self._city_epsg(b)
        pick = self._pick_pair_epsg(ea, eb)
        if pick is None:
            return
        
        with s.batch():
            cm = str(s.get(K_MAP_COORD_MODE, "") or "")
            if not cm:
                s.set(K_MAP_COORD_MODE, "auto")
            
            if 32601 <= int(pick) <= 32760:
                s.set(K_MAP_UTM_EPSG, int(pick))
            else:
                s.set(K_MAP_SRC_EPSG, int(pick))

    def _has_epsg(self) -> bool:
        s = self._s
        ue = _as_int(s.get(K_MAP_UTM_EPSG, None))
        se = _as_int(s.get(K_MAP_SRC_EPSG, None))
        if not ue:
            ue = _as_int(s.get("utm_epsg", None))
        if not se:
            se = _as_int(s.get("coord_src_epsg", None))
        return bool(ue or se)

    def _pick_pair_epsg(
        self, a: Optional[int], b: Optional[int]
    ) -> Optional[int]:
        if a and b:
            return int(a) if int(a) == int(b) else None
        if a: return int(a)
        if b: return int(b)
        return None

    def _city_epsg(self, city: str) -> Optional[int]:
        c = str(city or "").strip()
        if not c: return None
        s = self._s
        meta = s.get("cities.meta", None)
        e = _epsg_from_city_meta(meta, c)
        if e: return int(e)
        for k in self._city_epsg_keys(c):
            e2 = _as_int(s.get(k, None))
            if e2: return int(e2)
        return None

    def _city_epsg_keys(self, city: str) -> List[str]:
        c = str(city or "").strip()
        if not c: return []
        cc = c.lower()
        out: List[str] = []
        for nm in (c, cc):
            out.extend([
                f"city.{nm}.utm_epsg", f"city.{nm}.epsg",
                f"geo.city.{nm}.utm_epsg", f"geo.city.{nm}.epsg",
                f"cities.{nm}.utm_epsg", f"cities.{nm}.epsg",
                f"city_meta.{nm}.utm_epsg", f"city_meta.{nm}.epsg",
            ])
        return out

