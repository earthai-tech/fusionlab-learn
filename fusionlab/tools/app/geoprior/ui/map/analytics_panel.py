# geoprior/ui/map/analytics_panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.analytics_panel

Bottom analytics panel (D), collapsible.

Modern skeleton
---------------
- Tabs: Spatial | Sharpness | Reliability | Inspector
- Each tab: controls + matplotlib canvas
- Store-driven refresh (map.* keys)

Notes
-----
This is a skeleton with real, useful plots.
We will deepen each tab progressively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal, QSignalBlocker
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QDoubleSpinBox,
    QCheckBox,
    QStyle,
    QSizePolicy, 
    QScrollArea
)

from ...config.store import GeoConfigStore
from ..icon_utils import try_icon
from .plot_utils import (
    HAS_MPL,
    MplPlot,
    pick_obs_col,
    safe_quantile_cols,
)
from .selection_stats import (
    band_cols,
    exceed_prob_from_quantiles,
    group_trend,
    load_series_for_points,
    pick_mid_col,
    summarize_series,
)
from .hotspots import build_points
from .coord_utils import (
    ensure_lonlat,
    df_to_lonlat,
    parse_epsg,
)
from .sampling import (
    cfg_from_get as samp_cfg_from_get,
    sample_points,
)
from .keys import (
    MAP_ACTIVE_FILE,
    MAP_TIME_COL,
    MAP_X_COL,
    MAP_Y_COL,
    MAP_Z_COL,
    MAP_OBS_COL,
    MAP_ID_COL,
    MAP_VALUE_COL,
    MAP_VALUE_COL,
    MAP_VALUE_UNIT,
    MAP_TIME_VALUE,
    MAP_STEP_COL,
    MAP_CLICK_SAMPLE_IDX,
    MAP_DF_ALL,
    MAP_DF_FRAME,
    MAP_DF_POINTS,
    MAP_SAMPLING_CELL_KM,
    MAP_SAMPLING_MODE,
    MAP_SAMPLING_METHOD,
    MAP_SAMPLING_MAX_POINTS,
    MAP_SAMPLING_SEED,
    MAP_SAMPLING_MAX_PER_CELL,
    MAP_SAMPLING_APPLY_HOTSPOTS,
)


def pick_id_col(cols: Sequence[str]) -> str:
    cand = [str(c) for c in (cols or [])]
    for name in (
        "sample_idx",
        "sid",
        "point_id",
        "site_id",
        "id",
    ):
        if name in cand:
            return name
    return ""

def _strip_q_suffix(name: str) -> str:
    n = str(name or "").strip()
    for suf in ("_q", "_p"):
        if suf in n:
            left, right = n.rsplit(suf, 1)
            if right.isdigit():
                return left
    return n


def _human_label(name: str) -> str:
    n = _strip_q_suffix(name)
    n = n.replace(".", " ").replace("_", " ").strip()
    while "  " in n:
        n = n.replace("  ", " ")
    if not n:
        return ""
    parts = [p for p in n.split(" ") if p]
    return " ".join(
        p[:1].upper() + p[1:]
        for p in parts
    )


def _label_with_unit(name: str, unit: str) -> str:
    base = _human_label(name)
    u = str(unit or "").strip()
    return f"{base} ({u})" if u else base


def _q_tag(col: str) -> str:
    c = str(col or "").strip()
    for suf in ("_q", "_p"):
        if suf in c:
            _l, right = c.rsplit(suf, 1)
            if right.isdigit():
                tag = suf[1:]
                return f"{tag}{right.zfill(2)}"
    return c

@dataclass
class MapAnaCtx:
    path: Optional[Path] = None
    df: Optional[pd.DataFrame] = None
    df_all: Optional[pd.DataFrame] = None
    df_points: Optional[pd.DataFrame] = None

    x: str = ""
    y: str = ""
    z: str = ""
    t: str = ""
    step: str = ""

    id_col: str = ""

    time_value: str = ""
    q_cols: list[tuple[float, str]] = field(default_factory=list)
    obs_col: str = ""


class CollapsibleAnalyticsPanel(QFrame):
    """
    Bottom panel container for analytics tabs.
    """
    request_close = pyqtSignal()
    request_pin = pyqtSignal()

    def __init__(
        self,
        *,
        store: Optional[GeoConfigStore] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("MapAnalyticsPanel")
        self.setFrameShape(QFrame.StyledPanel)

        self.store = store
        self._expanded = True

        self._ctx = MapAnaCtx(q_cols=[])

        self._sel = SelectionTab(
            store=self.store,
            parent=self,
        )

        self._spatial = SpatialTab(parent=self)
        self._sharp = SharpnessTab(parent=self)
        self._rely = ReliabilityTab(
            store=self.store,
            parent=self,
        )
        self._insp = InspectorTab(parent=self)
        
        # provide store access for unit labels
        self._spatial.store = self.store
        self._sharp.store = self.store
        self._insp.store = self.store
        
        self._build_ui()
        self._build_tabs()

        if self.store is not None:
            self.store.config_changed.connect(
                self._on_store_changed,
            )
            self.refresh()

    # -------------------------
    # Public API
    # -------------------------
    def expand(self) -> None:
        self._expanded = True
        self.setVisible(True)
        self.refresh()


    def collapse(self) -> None:
        self._expanded = False
        self.setVisible(False)

    def is_expanded(self) -> bool:
        return bool(self._expanded)

    def refresh(self) -> None:
        """
        Pull from store and update all tabs.
        """
        store = self.store
        if store is None:
            return
    
        def _sget(key: str) -> str:
            return str(store.get(key, "") or "").strip()
    
        p_raw = _sget(MAP_ACTIVE_FILE)
        path = Path(p_raw) if p_raw else None
    
        x = _sget(MAP_X_COL)
        y = _sget(MAP_Y_COL)
    
        z = _sget(MAP_VALUE_COL)
        if not z:
            z = _sget(MAP_Z_COL)
    
        t = _sget(MAP_TIME_COL)
        step = _sget(MAP_STEP_COL)
        tv = _sget(MAP_TIME_VALUE)
    
        df_all = store.get(MAP_DF_ALL, None)
        df_frame = store.get(MAP_DF_FRAME, None)
        df_points = store.get(MAP_DF_POINTS, None)
    
        hcols: list[str] = []
        if path is not None and path.exists():
            try:
                hcols = list(
                    pd.read_csv(
                        path,
                        nrows=0,
                    ).columns
                )
            except Exception:  # noqa: BLE001
                hcols = []
    
        id_col = pick_id_col(hcols)
        q_cols_hdr = safe_quantile_cols(hcols)
        obs_auto = pick_obs_col(hcols)
    
        obs_over = _sget(MAP_OBS_COL)
        obs_pick = obs_auto
        if obs_over and obs_over in hcols:
            obs_pick = obs_over
    
        extra: list[str] = []
        if id_col:
            extra.append(id_col)
        if obs_pick:
            extra.append(obs_pick)
    
        for _q, c in (q_cols_hdr or []):
            if c:
                extra.append(str(c))
    
        if (
            isinstance(df_frame, pd.DataFrame)
            and not df_frame.empty
        ):
            df = df_frame
        else:
            df = self._load_frame(
                path=path,
                x=x,
                y=y,
                z=z,
                t=t,
                step=step,
                time_value=tv,
                extra_cols=extra,
            )
    
        cols = list(df.columns) if df is not None else []
        q_cols = safe_quantile_cols(cols) or q_cols_hdr
        q_cols = [
            (float(q), str(c))
            for (q, c) in (q_cols or [])
            if str(c) in cols
        ]
    
        id_col2 = id_col if (id_col in cols) else ""
        obs2 = obs_pick if (obs_pick in cols) else ""
    
        self._ctx = MapAnaCtx(
            path=path,
            df=df,
            df_all=df_all
            if isinstance(df_all, pd.DataFrame)
            else None,
            df_points=df_points
            if isinstance(df_points, pd.DataFrame)
            else None,
            x=x,
            y=y,
            z=z,
            t=t,
            step=step,
            time_value=tv,
            q_cols=q_cols,
            obs_col=obs2,
            id_col=id_col2,
        )
    
        tabs = (
            self._sel,
            self._spatial,
            self._sharp,
            self._rely,
            self._insp,
        )
    
        for tab in tabs:
            try:
                setattr(tab, "store", self.store)
                tab.set_context(self._ctx)
            except Exception:
                pass

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
    
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)
    
        self.lb_title = QLabel("Analytics", self)
        self.lb_title.setObjectName("gpDockTitle")
        hdr.addWidget(self.lb_title, 0)
    
        hdr.addStretch(1)
    
        self.btn_pin = QToolButton(self)
        self.btn_pin.setObjectName("miniAction")
        self.btn_pin.setAutoRaise(True)
        self.btn_pin.setToolTip("Pin (pop out)")

        ico = try_icon("pin.svg")
        if ico is not None and (not ico.isNull()):
            self.btn_pin.setIcon(ico)
        else:
            self.btn_pin.setIcon(
                self.style().standardIcon(QStyle.SP_TitleBarMaxButton)
            )

        self.btn_pin.clicked.connect(
            self.request_pin.emit
        )
        hdr.addWidget(self.btn_pin, 0)
    
        self.btn_close = QToolButton(self)
        self.btn_close.setObjectName("miniAction")
        self.btn_close.setAutoRaise(True)
        self.btn_close.setToolTip("Close analytics")
        self.btn_close.setIcon(
            self.style().standardIcon(
                QStyle.SP_TitleBarCloseButton
            )
        )
        self.btn_close.clicked.connect(
            self.request_close.emit
        )
        hdr.addWidget(self.btn_close, 0)
    
        root.addLayout(hdr, 0)
    
        self.body = QWidget(self)
        self.body.setObjectName("gpDockBody")
        body_l = QVBoxLayout(self.body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(0)
    
        self.tabs = QTabWidget(self.body)
        self.tabs.setObjectName("gpAnalyticsTabs")
 
        body_l.addWidget(self.tabs, 1)
        root.addWidget(self.body, 1)

    
    def _build_tabs(self) -> None:
        tabs = self.tabs
        if tabs is None:
            return
    
        with QSignalBlocker(tabs):
            tabs.clear()
    
            tabs.addTab(self._sel, "Selection")
            self.selection = self._sel  
    
            tabs.addTab(self._spatial, "Spatial")
            tabs.addTab(self._sharp, "Sharpness")
            tabs.addTab(self._rely, "Reliability")
    
            self.inspector = self._insp
            tabs.addTab(self.inspector, "Inspector")
    
        
        # ensure we don't accumulate duplicate connections
        try:
            tabs.currentChanged.disconnect(self._on_tab_changed)
        except Exception:
            pass
    
        tabs.currentChanged.connect(self._on_tab_changed)
    
    
    def _on_tab_changed(self, _i: int) -> None:
        # Prefer sender() because tabs may be temporarily moved/reparented.
        s = self.sender()
        tabs = s if isinstance(s, QTabWidget) else self.tabs
        if tabs is None:
            return
    
        try:
            w = tabs.currentWidget()
        except Exception:
            return
    
        if w is None:
            return
    
        self.refresh()
    
    def take_tabs(self) -> Optional[QTabWidget]:
        tabs = self.tabs
        if tabs is None:
            return None
    
        # block signals to prevent queued currentChanged during detach
        blocker = QSignalBlocker(tabs)
    
        try:
            tabs.currentChanged.disconnect(self._on_tab_changed)
        except Exception:
            pass
    
        lay = self.body.layout()
        if lay is not None and lay.count():
            lay.takeAt(0)
    
        tabs.setParent(None)
    
        # release blocker explicitly (nice style)
        del blocker
        return tabs
    
    
    def set_tabs(self, w: QTabWidget) -> None:
        if w is None:
            return
    
        self.tabs = w
        blocker = QSignalBlocker(w)
    
        w.setParent(self.body)
    
        lay = self.body.layout()
        if lay is not None:
            lay.addWidget(w, 1)
    
        # reconnect cleanly
        try:
            w.currentChanged.disconnect(self._on_tab_changed)
        except Exception:
            pass
    
        w.currentChanged.connect(self._on_tab_changed)
    
        del blocker
    
        # optional: refresh current tab once after reattach
        try:
            self._on_tab_changed(int(w.currentIndex()))
        except Exception:
            pass


    # -------------------------
    # Store + IO
    # -------------------------
    def _on_store_changed(self, keys) -> None:
        ks = set(keys or [])
        
        if MAP_CLICK_SAMPLE_IDX in ks:
            sid = self.store.get(MAP_CLICK_SAMPLE_IDX, None)
            if sid is not None:
                try:
                    self._sel.set_id(int(sid))
                except Exception:
                    pass
                try:
                    self._insp.set_id(int(sid))
                except Exception:
                    pass
            return

        want = {
            MAP_ACTIVE_FILE, 
            MAP_X_COL,
            MAP_Y_COL,
            MAP_Z_COL,
            MAP_VALUE_COL,
            MAP_TIME_COL,
            MAP_STEP_COL,
            MAP_TIME_VALUE,
            MAP_SAMPLING_CELL_KM,
            MAP_SAMPLING_MODE, 
            MAP_SAMPLING_METHOD, 
            MAP_SAMPLING_MAX_POINTS, 
            MAP_SAMPLING_SEED, 
            MAP_SAMPLING_MAX_PER_CELL, 
            MAP_SAMPLING_APPLY_HOTSPOTS, 
            MAP_DF_ALL, 
            MAP_DF_FRAME, 
            MAP_DF_POINTS,
            MAP_ID_COL, 
            MAP_OBS_COL
        }
        if ks.intersection(want):
            self.refresh()

    def _load_frame(
        self,
        *,
        path: Optional[Path],
        x: str,
        y: str,
        z: str,
        t: str,
        step: str,
        time_value: str,
        extra_cols: Sequence[str] = (),
    ) -> Optional[pd.DataFrame]:
        if path is None:
            return None
        if not path.exists():
            return None

        use: list[str] = []
        for c in (x, y, z, t, step):
            if c and c not in use:
                use.append(c)
        for c in (extra_cols or ()):  # extras: q/obs/id
            c = str(c or "").strip()
            if c and c not in use:
                use.append(c)

        # If we have no mapping yet, load a small header.
        if not use:
            try:
                df0 = pd.read_csv(path, nrows=200)
            except Exception:
                return None
            return df0

        try:
            df = pd.read_csv(path, usecols=use)
        except Exception:
            try:
                df = pd.read_csv(path)
            except Exception:
                return None

        if time_value and t and t in df.columns:
            try:
                tv = int(time_value)
                df = df[df[t] == tv]
            except Exception:
                pass

        # -----------------------------------------
        # Shared sampling (same subset as MapTab)
        # -----------------------------------------
        if self.store is not None and x and y and z:
            if x in df.columns and y in df.columns and z in df.columns:
                pts = build_points(df, x=x, y=y, v=z)

                mode = str(self.store.get(
                    "map.coord_mode",
                    "lonlat",
                )).strip().lower()

                utm_epsg = parse_epsg(
                    self.store.get("map.utm_epsg", None)
                )
                src_epsg = parse_epsg(
                    self.store.get("map.coord_epsg", None)
                )

                try:
                    pts, ok, _msg = ensure_lonlat(
                        pts,
                        mode=mode,
                        utm_epsg=utm_epsg,
                        src_epsg=src_epsg,
                    )
                except Exception:
                    pts = df_to_lonlat(
                        pts,
                        x="lon",
                        y="lat",
                        mode=mode,
                        utm_epsg=utm_epsg,
                        src_epsg=src_epsg,
                    )
                    ok = not pts.empty

                if ok and (not pts.empty):
                    scfg = samp_cfg_from_get(self.store.get)
                    pts_s = sample_points(
                        pts,
                        scfg,
                        lon="lon",
                        lat="lat",
                    )
                    if pts_s is not None and (not pts_s.empty):
                        df = df.loc[pts_s.index]

        return df


# =====================================================
# Tabs
# =====================================================

class _TabBase(QWidget):
    _PLOT_MIN_W = 520
    _PLOT_MIN_H = 260
    _CTRL_MIN_W = 240
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ctx = MapAnaCtx(q_cols=[])
        self.store: Optional[GeoConfigStore] = None

    def set_context(self, ctx: MapAnaCtx) -> None:
        self._ctx = ctx
        self.refresh()

    def refresh(self) -> None:
        return
    
    def _value_unit(self) -> str:
        s = self.store
        if s is None:
            return ""
        return str(
            s.get(MAP_VALUE_UNIT, "")
            or ""
        ).strip()
    
    
    def _value_label(self, name: str) -> str:
        return _label_with_unit(
            name,
            self._value_unit(),
        )

    def _install_scroll(self, ctrl: QWidget, plot: QWidget) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sa = QScrollArea(self)
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sa.setFrameShape(QFrame.NoFrame)

        host = QWidget()
        h = QHBoxLayout(host)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)

        ctrl.setParent(host)
        plot.setParent(host)

        ctrl.setMinimumWidth(max(ctrl.sizeHint().width(), self._CTRL_MIN_W))
        plot.setMinimumSize(self._PLOT_MIN_W, self._PLOT_MIN_H)

        h.addWidget(ctrl, 0)
        h.addWidget(plot, 1)

        h.setSizeConstraint(QHBoxLayout.SetMinimumSize)

        sa.setWidget(host)
        outer.addWidget(sa, 1)
        
    def _save_plot_png(
        self,
        plot: MplPlot,
        *,
        default: str,
        title: str = "Save plot",
    ) -> None:
        p, _ = QFileDialog.getSaveFileName(
            self,
            title,
            default,
            "PNG (*.png)",
        )
        if not p:
            return
        plot.save_png(p)
        

class SelectionTab(_TabBase):
    """
    Selection tab (default):
    - shows stats for selected point(s)
    - smart plots (trend / fan / risk)
    """

    def __init__(
        self,
        *,
        store: Optional[GeoConfigStore] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.store = store

        self._sel: list[tuple[float, float]] = []
        self._sel_id: Optional[int] = None
        self._tol_auto = True
        
        self.lb_info = QLabel(
            "Click a point on the map to inspect.",
            self,
        )
        self.lb_info.setWordWrap(True)
        self.lb_info.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.cmb_view = QComboBox(self)
        self.cmb_view.addItems(
            [
                "Trend",
                "Fan (single point)",
                "Δ rate",
                "Distribution (frame)",
                "Exceedance risk",
            ]
        )

        self.cmb_agg = QComboBox(self)
        self.cmb_agg.addItems(["median", "mean"])
        self.cmb_band = QComboBox(self)
        self.cmb_band.addItems(
            ["80", "50", "90"]
        )

        self.sp_tol = QDoubleSpinBox(self)
        self.sp_tol.setRange(0.0, 1e6)
        self.sp_tol.setDecimals(9)
        self.sp_tol.setSingleStep(1e-6)
        self.sp_tol.setValue(1e-6)
        
        mode = "lonlat"
        if self.store is not None:
            mode = str(self.store.get(
                "map.coord_mode", "lonlat"
            ) or "lonlat").strip().lower()
        
        self.sp_tol.setValue(0.5 if mode == "utm" else 1e-6)


        self.ed_thr = QLineEdit(self)
        self.ed_thr.setPlaceholderText(
            "threshold (e.g., -50)"
        )

        self.btn_clear = QToolButton(self)
        self.btn_clear.setText("✕")
        self.btn_clear.setToolTip("Clear selection")
        self.btn_clear.setAutoRaise(True)

        self.plot = MplPlot(
            title="Selection analytics",
            parent=self,
        )
        self.plot.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self._build_ui()
        self._apply_default_tol()
        self._connect()
        
        if self.store is not None:
            self.store.config_changed.connect(
                self._on_store_changed
            )

        
    def _build_ui(self) -> None:
        box = QGroupBox("Selection", self)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)
    
        v.addWidget(self.lb_info, 0)
    
        r0 = QHBoxLayout()
        r0.setContentsMargins(0, 0, 0, 0)
        r0.setSpacing(6)
        r0.addWidget(QLabel("View:", box), 0)
        r0.addWidget(self.cmb_view, 1)
        r0.addWidget(self.btn_clear, 0)
        v.addLayout(r0)
    
        v.addWidget(QLabel("Agg:", box), 0)
        v.addWidget(self.cmb_agg, 0)
    
        v.addWidget(QLabel("Band:", box), 0)
        v.addWidget(self.cmb_band, 0)
    
        v.addWidget(QLabel("Match tol:", box), 0)
        v.addWidget(self.sp_tol, 0)
    
        v.addWidget(QLabel("Risk thr:", box), 0)
        v.addWidget(self.ed_thr, 0)
    
        v.addStretch(1)
    
        # Scroll wrapper (controls + plot)
        self._install_scroll(box, self.plot)

    def _connect(self) -> None:
        self.cmb_view.currentIndexChanged.connect(
            self.refresh
        )
        self.cmb_agg.currentIndexChanged.connect(
            self.refresh
        )
        self.cmb_band.currentIndexChanged.connect(
            self.refresh
        )
        self.sp_tol.valueChanged.connect(self._on_tol_changed)
        self.sp_tol.valueChanged.connect(self.refresh)

        self.ed_thr.editingFinished.connect(
            self.refresh
        )
        self.btn_clear.clicked.connect(
            self.clear_selection
        )

        if self.plot.actions is not None:
            self.plot.actions.btn_refresh.clicked.connect(
                self.refresh
            )
            self.plot.actions.btn_clear.clicked.connect(
                lambda: self.plot.show_empty(
                    title="No data to plot",
                    message="Plot cleared.",
                    hint="Click ↻ to refresh.",
                )
            )
            self.plot.actions.btn_save.clicked.connect(
                lambda: self._save_plot_png(
                    self.plot,
                    default="selection.png",
                )
            )

    # -------------------------
    # Public API
    # -------------------------
    def _on_store_changed(self, keys) -> None:
        if not self._tol_auto:
            return
        if "map.coord_mode" in set(keys or []):
            self._apply_default_tol()

    def _apply_default_tol(self) -> None:
        mode = "lonlat"
        if self.store is not None:
            mode = str(self.store.get(
                "map.coord_mode", "lonlat"
            ) or "lonlat").strip().lower()
    
        is_metric = mode in ("utm", "xy", "projected")
        self.sp_tol.setValue(0.5 if is_metric else 1e-6)
    
    def _on_tol_changed(self, _v: float) -> None:
        self._tol_auto = False


    def clear_selection(self) -> None:
        self._sel = []
        self._sel_id = None
        self.lb_info.setText(
            "Click a point on the map to inspect."
        )
        self.refresh()

    def set_id(self, sid: int) -> None:
        try:
            self._sel_id = int(sid)
        except Exception:
            return
        self._sel = []
        self.refresh()

    def set_xy(self, x: float, y: float) -> None:
        """Set selection to a single point."""
        self._sel_id = None
        try:
            self._sel = [(float(x), float(y))]
        except Exception:
            return
        self.refresh()

    def set_points(
        self,
        pts: Sequence[tuple[float, float]],
    ) -> None:
        out: list[tuple[float, float]] = []
        for p in pts or []:
            try:
                out.append((float(p[0]), float(p[1])))
            except Exception:
                continue
        self._sel = out
        self.refresh()


    def _load_id_series(
        self,
        *,
        path: Path,
        id_col: str,
        sid: int,
        t: str,
        z: str,
        step: str,
        q_cols: Sequence[str],
    ) -> Optional[pd.DataFrame]:
        
        use: list[str] = [id_col, t, z]
        
        df_all = getattr(self._ctx, "df_all", None)
        if isinstance(df_all, pd.DataFrame) and (not df_all.empty):
            if id_col in df_all.columns:
                cols = [c for c in use if c in df_all.columns]
                m = df_all[id_col] == int(sid)
                sub = df_all.loc[m, cols]
                if len(sub):
                    return sub.reset_index(drop=True)

        if step:
            use.append(step)
        for c in q_cols or []:
            if c and c not in use:
                use.append(str(c))
        try:
            chunks = pd.read_csv(
                path,
                usecols=use,
                chunksize=200_000,
            )
        except Exception:
            return None

        keep: list[pd.DataFrame] = []
        for ch in chunks:
            if id_col not in ch.columns:
                continue
            try:
                m = ch[id_col] == sid
            except Exception:
                m = False
            part = ch.loc[m]
            if not part.empty:
                keep.append(part)
        if not keep:
            return None
        try:
            return pd.concat(keep, ignore_index=True)
        except Exception:
            return None


    # -------------------------
    # Plot
    # -------------------------
    def refresh(self) -> None:
        if not HAS_MPL:
            return
        if not self.plot.is_ready():
            return

        ctx = self._ctx
        path = ctx.path

        if path is None or not path.exists():
            _show_empty(
                self.plot,
                title="No dataset selected",
                message=(
                    "Select an active file to inspect selection."
                ),
                hint="Map → Data panel → choose an active dataset.",
            )
            return

        x = ctx.x
        y = ctx.y
        t = ctx.t
        z = ctx.z
        # step = ctx.step
        qs = ctx.q_cols or []

        if not (x and y and t and z):
            _show_empty(
                self.plot,
                title="Mapping incomplete",
                message=(
                    "Selection needs X/Y + Time col + Z."
                ),
                hint="Pick X/Y/Z in the header and set Time col.",
            )
            return

        if not self._sel and self._sel_id is None:
            _show_empty(
                self.plot,
                title="No selection",
                message="Click a point or select a group.",
                hint="Tip: Shift+click can be added later for multi-select.",
            )
            return

        tol = float(self.sp_tol.value())
        q_names = [c for _, c in qs]

        df = None
        if (
            self._sel_id is not None
            and ctx.id_col
            and path is not None
            and path.exists()
        ):
            df = self._load_id_series(
                path=path,
                id_col=ctx.id_col,
                sid=int(self._sel_id),
                t=t,
                z=z,
                step=ctx.step,
                q_cols=q_names,
            )
        else:
            df = load_series_for_points(
                path=path,
                x=x,
                y=y,
                t=t,
                z=z,
                q_cols=q_names,
                pts=self._sel,
                tol=tol,
            )

        if df is None or df.empty:
            _show_empty(
                self.plot,
                title="No match",
                message="Selection did not match any rows.",
                hint="Try increasing Match tol.",
            )
            return

        s = summarize_series(df, t=t, z=z)
        self.lb_info.setText(
            f"{s.n_points} point(s) | "
            f"t=[{s.t_min}, {s.t_max}] | "
            f"z_mean={s.z_mean}"
        )

        view = str(self.cmb_view.currentText() or "Trend")
        agg = str(self.cmb_agg.currentText() or "median")
        band = str(self.cmb_band.currentText() or "80")
        mid = pick_mid_col(z, qs)
        lo, hi = band_cols(qs, band=band)

        self.plot.clear_axes()
        ax = self.plot.ax
        assert ax is not None

        # sort by time
        try:
            df = df.sort_values(by=t, kind="mergesort")
        except Exception:
            pass

        if view.startswith("Fan") and s.n_points == 1:
            self._plot_fan(
                ax=ax,
                df=df,
                t=t,
                mid=mid,
                lo=lo,
                hi=hi,
            )
        elif view.startswith("Δ"):
            self._plot_rate(
                ax=ax,
                df=df,
                t=t,
                mid=mid,
                agg=agg,
            )
        elif view.startswith("Distribution"):
            self._plot_dist_frame(
                ax=ax,
                df_frame=ctx.df,
                x=x,
                y=y,
                z=mid,
                tol=tol,
            )
        elif view.startswith("Exceedance"):
            self._plot_risk(
                ax=ax,
                df=df,
                t=t,
                qs=qs,
                agg=agg,
            )
        else:
            self._plot_trend(
                ax=ax,
                df=df,
                t=t,
                mid=mid,
                lo=lo,
                hi=hi,
                agg=agg,
            )
            
        ax.set_xlabel(str(t))
        ax.set_ylabel(self._value_label(z))
        ax.grid(True, alpha=0.25)
        
        self.plot.draw()

    def _plot_fan(
        self,
        *,
        ax,
        df: pd.DataFrame,
        t: str,
        mid: str,
        lo: Optional[str],
        hi: Optional[str],
    ) -> None:
        x = df[t].to_numpy()
        if lo and hi and lo in df.columns and hi in df.columns:
            ax.fill_between(
                x,
                df[lo].to_numpy(),
                df[hi].to_numpy(),
                alpha=0.25,
                label=f"{_q_tag(lo)}..{_q_tag(hi)}",
            )
        if mid in df.columns:
            ax.plot(
                x,
                df[mid].to_numpy(),
                linewidth=2.0,
                label=_q_tag(mid),
            )
        ax.set_title("Quantile fan (single point)")
        ax.legend(loc="best")

    def _plot_trend(
        self,
        *,
        ax,
        df: pd.DataFrame,
        t: str,
        mid: str,
        lo: Optional[str],
        hi: Optional[str],
        agg: str,
    ) -> None:
        s = summarize_series(df, t=t, z=mid)
        if s.n_points <= 1:
            self._plot_fan(
                ax=ax,
                df=df,
                t=t,
                mid=mid,
                lo=lo,
                hi=hi,
            )
            ax.set_title("Trend (single point)")
            return

        g = group_trend(df, t=t, mid=mid, agg=agg)
        if g.empty:
            return

        ax.plot(
            g[t].to_numpy(),
            g["mid"].to_numpy(),
            linewidth=2.0,
            label=f"group {agg}",
        )
        if "p10" in g.columns and "p90" in g.columns:
            ax.fill_between(
                g[t].to_numpy(),
                g["p10"].to_numpy(),
                g["p90"].to_numpy(),
                alpha=0.25,
                label="p10..p90 (across points)",
            )

        ax.set_title("Group trend")
        ax.legend(loc="best")

    def _plot_rate(
        self,
        *,
        ax,
        df: pd.DataFrame,
        t: str,
        mid: str,
        agg: str,
    ) -> None:
        s = summarize_series(df, t=t, z=mid)
        if s.n_points <= 1:
            g = df[[t, mid]].copy()
            g = g.sort_values(by=t)
            g["rate"] = g[mid].diff()
            ax.plot(
                g[t].to_numpy(),
                g["rate"].to_numpy(),
                linewidth=2.0,
            )
            ax.set_title("Δ per step (single point)")
            return

        g = group_trend(df, t=t, mid=mid, agg=agg)
        if g.empty:
            return
        g["rate"] = g["mid"].diff()
        ax.plot(
            g[t].to_numpy(),
            g["rate"].to_numpy(),
            linewidth=2.0,
        )
        ax.set_title("Δ per step (group)")

    def _plot_dist_frame(
        self,
        *,
        ax,
        df_frame: Optional[pd.DataFrame],
        x: str,
        y: str,
        z: str,
        tol: float,
    ) -> None:
        if df_frame is None or df_frame.empty:
            _show_empty(
                self.plot,
                title="No frame data",
                message=(
                    "Distribution needs a loaded frame slice."
                ),
                hint="Set a time value / step in Map controls.",
            )
            return

        if not self._sel:
            return

        # exact / tol match inside current frame
        xs = [p[0] for p in self._sel]
        ys = [p[1] for p in self._sel]
        xmin, xmax = min(xs) - tol, max(xs) + tol
        ymin, ymax = min(ys) - tol, max(ys) + tol

        m0 = (
            (df_frame[x] >= xmin)
            & (df_frame[x] <= xmax)
            & (df_frame[y] >= ymin)
            & (df_frame[y] <= ymax)
        )
        sub = df_frame.loc[m0]
        if sub.empty or z not in sub.columns:
            _show_empty(
                self.plot,
                title="No matches in frame",
                message="Selection not found in current frame.",
                hint="Try increasing Match tol.",
            )
            return

        vals = sub[z].dropna().to_numpy()
        if len(vals) < 2:
            _show_empty(
                self.plot,
                title="Too few points",
                message="Need >=2 values for distribution.",
                hint="Select a group or increase sampling.",
            )
            return

        ax.hist(vals, bins=20, alpha=0.8)
        ax.set_title("Distribution (current frame)")

    def _plot_risk(
        self,
        *,
        ax,
        df: pd.DataFrame,
        t: str,
        qs: list[tuple[float, str]],
        agg: str,
    ) -> None:
        if not qs:
            _show_empty(
                self.plot,
                title="No quantiles found",
                message="Risk view needs qXX columns.",
                hint="Use a calibrated forecast file with quantiles.",
            )
            return

        try:
            thr = float(self.ed_thr.text().strip())
        except Exception:
            thr = 0.0

        df = df.copy()
        df["p_exceed"] = df.apply(
            lambda r: exceed_prob_from_quantiles(
                r,
                thr=thr,
                q_cols=qs,
            ),
            axis=1,
        )

        if "_pid" in df.columns:
            g = df.groupby(t)["p_exceed"]
            if agg == "mean":
                out = g.mean()
            else:
                out = g.median()
        else:
            out = df.set_index(t)["p_exceed"]

        out = out.sort_index()
        ax.plot(
            out.index.to_numpy(),
            out.to_numpy(),
            linewidth=2.0,
        )
        ax.set_title(f"P(Z > {thr}) vs time")
        

class SpatialTab(_TabBase):
    """
    Spatial view (2D scatter or 3D scatter).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.cmb_mode = QComboBox(self)
        self.cmb_mode.addItems(["2D", "3D"])

        self.sp_max = QSpinBox(self)
        self.sp_max.setRange(2000, 500000)
        self.sp_max.setValue(80000)

        self.plot = MplPlot(
            title="Spatial scatter",
            parent=self,
        )
        self.plot.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self._build_ui()
    
    def _build_ui(self) -> None:
        ctrl = self._controls(self)
    
        # Scroll wrapper (controls + plot)
        self._install_scroll(ctrl, self.plot)
    
        self.cmb_mode.currentIndexChanged.connect(self.refresh)
        self.sp_max.valueChanged.connect(self.refresh)
    
        if self.plot.actions is not None:
            self.plot.actions.btn_clear.clicked.connect(
                lambda: self.plot.show_empty(
                    title="No data to plot",
                    message="Plot cleared.",
                    hint="Click ↻ to refresh.",
                )
            )
            self.plot.actions.btn_refresh.clicked.connect(self.refresh)
            self.plot.actions.btn_save.clicked.connect(
                lambda: self._save_plot_png(
                    self.plot,
                    default="spatial.png",
                )
            )


    def _controls(self, parent: QWidget) -> QWidget:
        box = QGroupBox("View", parent)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        v.addWidget(QLabel("Mode:", box), 0)
        v.addWidget(self.cmb_mode, 0)

        v.addWidget(QLabel("Max points:", box), 0)
        v.addWidget(self.sp_max, 0)

        v.addStretch(1)
        return box

    def refresh(self) -> None:
        if not HAS_MPL:
            return
        if not self.plot.is_ready():
            return

        df = self._ctx.df
        x = self._ctx.x
        y = self._ctx.y
        z = self._ctx.z

        if df is None or not len(df):
            _show_empty(
                self.plot,
                title="No data to plot",
                message="Select an active dataset to enable spatial analytics.",
                hint="Go to Map → Data panel and choose an active file.",
            )
            return

        if not (x and y and z):
            _show_empty(
                self.plot,
                title="Mapping incomplete",
                message="Select X, Y and Z columns to render the spatial scatter.",
                hint="Use the pickers in the map header (X / Y / Z).",
            )
            return
    
        if x not in df.columns or y not in df.columns or z not in df.columns:
            _show_empty(
                self.plot,
                title="Columns not found",
                message="One or more selected columns do not exist in the dataset.",
                hint="Check X/Y/Z selection and reload columns if needed.",
            )
            return
        
        # Guard: X and Y must differ (otherwise d[x] becomes 2D after selection)
        if x == y:
            _show_empty(
                self.plot,
                title="Invalid mapping",
                message="X and Y cannot be the same column.",
                hint="Pick two different columns for X and Y in the map header.",
            )
            return
        
        # Build a guaranteed-1D frame (handles duplicate column labels safely)
        sx = _pick_1d(df, x)
        sy = _pick_1d(df, y)
        sz = _pick_1d(df, z)
        
        d = pd.concat([sx, sy, sz], axis=1)
        d.columns = [x, y, z]
        d = d.dropna()
        
        nmax = int(self.sp_max.value())
        if len(d) > nmax:
            d = d.sample(n=nmax, random_state=0)
        
        # ---- plotting ----
        self.plot.clear_axes()
        assert self.plot.fig is not None
        
        mode = str(self.cmb_mode.currentText()).lower()
        
        if mode == "3d":
            self.plot.fig.clf()
            ax = self.plot.fig.add_subplot(111, projection="3d")
            ax.scatter(
                d[x].to_numpy(),
                d[y].to_numpy(),
                d[z].to_numpy(),
                s=8,
                alpha=0.85,
            )
            ax.set_xlabel(_human_label(x))
            ax.set_ylabel(_human_label(y))
            ax.set_zlabel(self._value_label(z))
            self.plot.ax = ax
            self.plot.draw()
            return
        
        ax = self.plot.ax
        assert ax is not None
        
        sc = ax.scatter(
            d[x].to_numpy(),
            d[y].to_numpy(),
            c=d[z].to_numpy(),
            s=10,
            alpha=0.85,
        )
        
        zlab = self._value_label(z)
        ax.set_xlabel(_human_label(x))
        ax.set_ylabel(_human_label(y))
        ax.set_title(f"Colored by {zlab}")
        
        cb =self.plot.fig.colorbar(
            sc, ax=ax, 
            fraction=0.03, 
            pad=0.02
        )
        cb.set_label(zlab)
        self.plot.draw()

class SharpnessTab(_TabBase):
    """
    Uncertainty width vs time (if quantiles exist).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.cmb_int = QComboBox(self)
        self.cmb_int.addItems([
            "50% (q25-q75)",
            "80% (q10-q90)",
            "90% (q05-q95)",
        ])

        self.plot = MplPlot(
            title="Interval width (sharpness)",
            parent=self,
        )
        self.plot.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self._build_ui()

    def _build_ui(self) -> None:
        ctrl = self._controls(self)
    
        # Scroll wrapper (controls + plot)
        self._install_scroll(ctrl, self.plot)
    
        self.cmb_int.currentIndexChanged.connect(self.refresh)
    
        if self.plot.actions is not None:
            self.plot.actions.btn_clear.clicked.connect(
                lambda: self.plot.show_empty(
                    title="No data to plot",
                    message="Plot cleared.",
                    hint="Click ↻ to refresh.",
                )
            )
            self.plot.actions.btn_refresh.clicked.connect(self.refresh)


    def _controls(self, parent: QWidget) -> QWidget:
        box = QGroupBox("Sharpness", parent)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        v.addWidget(QLabel("Interval:", box), 0)
        v.addWidget(self.cmb_int, 0)

        v.addStretch(1)
        return box

    def refresh(self) -> None:
        if not HAS_MPL:
            return
        if not self.plot.is_ready():
            return

        df = self._ctx.df
        qs = self._ctx.q_cols or []
        # step = self._ctx.step
        t = self._ctx.t

        if df is None or not len(df):
            _show_empty(
                self.plot,
                title="No data to plot",
                message="Select an active dataset to compute sharpness.",
            )
            return
    
        if not qs:
            _show_empty(
                self.plot,
                title="No quantiles found",
                message="Sharpness requires quantile columns (qXX).",
                hint="Load a probabilistic output file (q10/q50/q90...).",
            )
            return
    
        if not t or t not in df.columns:
            _show_empty(
                self.plot,
                title="Time column missing",
                message="Pick a Time column to group interval width over time.",
                hint="Set Map → Time col, then pick a time value if needed.",
            )
            return
        self.plot.clear_axes()
        ax = self.plot.ax
        assert ax is not None
        lo_q, hi_q = self._pick_interval(qs)
        
        if not (lo_q and hi_q):
            _show_empty(
                self.plot,
                title="Interval unavailable",
                message="Requested interval columns were not found.",
            )
            return
        
        d = df[[t, lo_q, hi_q]].dropna()
        
        if not len(d):
            _show_empty(
                self.plot,
                title="No rows after filtering",
                message="Nothing matched the current filters.",
            )
            return

        # Ensure that hi_q and lo_q are single columns
        # inside SharpnessTab.refresh()
        d0 = df[[t, lo_q, hi_q]].copy()
        d0 = d0.dropna(subset=[t])
        
        lo = _pick_1d(d0, lo_q)
        hi = _pick_1d(d0, hi_q)
        
        lo = pd.to_numeric(lo, errors="coerce")
        hi = pd.to_numeric(hi, errors="coerce")
        
        w = (hi - lo).abs()
        
        d = d0[[t]].copy()
        d["w"] = w
        d = d.dropna(subset=["w"])

        g = d.groupby(t)["w"].median()

        ax.plot(g.index.values, g.values)
        ax.set_xlabel(t)
        u = self._value_unit()
        yl = "Median width"
        if u:
            yl = f"{yl} ({u})"
        
        ax.set_ylabel(yl)
        ax.set_title(f"{_q_tag(lo_q)}..{_q_tag(hi_q)}")

        self.plot.draw()

    def _pick_interval(
        self,
        qs: list[tuple[float, str]],
    ) -> tuple[str, str]:
        want = str(self.cmb_int.currentText())

        mapping = {
            "50%": (0.25, 0.75),
            "80%": (0.10, 0.90),
            "90%": (0.05, 0.95),
        }
        a = b = None
        for k, (qa, qb) in mapping.items():
            if k in want:
                a, b = qa, qb

        if a is None or b is None:
            return "", ""

        lo = self._find_q(qs, a)
        hi = self._find_q(qs, b)
        return lo, hi

    def _find_q(
        self,
        qs: list[tuple[float, str]],
        q: float,
    ) -> str:
        best = ""
        err = 1e9
        for qq, col in qs:
            e = abs(float(qq) - float(q))
            if e < err:
                err = e
                best = col
        return best


class ReliabilityTab(_TabBase):
    """
    PIT + coverage if observation exists.
    """

    def __init__(
        self,
        *,
        store: Optional[GeoConfigStore] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.store = store

        self.cmb_obs = QComboBox(self)
        self.cmb_obs.addItem("(auto)")

        self.plot = MplPlot(
            title="Reliability / PIT",
            parent=self,
        )
        self.plot.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self._build_ui()

        self.cmb_obs.currentIndexChanged.connect(
            self._on_obs_changed,
        )

    def _build_ui(self) -> None:
        box = QGroupBox("Inputs", self)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)
    
        v.addWidget(QLabel("Obs column:", box), 0)
        v.addWidget(self.cmb_obs, 0)
        v.addStretch(1)
    
        # Scroll wrapper (controls + plot)
        self._install_scroll(box, self.plot)
    
        if self.plot.actions is not None:
            self.plot.actions.btn_refresh.clicked.connect(self.refresh)
            self.plot.actions.btn_clear.clicked.connect(
                lambda: self.plot.show_empty(
                    title="No data to plot",
                    message="Plot cleared.",
                    hint="Click ↻ to refresh.",
                )
            )



    def _on_obs_changed(self, _i: int) -> None:
        if self.store is None:
            return
        name = str(self.cmb_obs.currentText() or "").strip()
        if name == "(auto)":
            name = ""
        self.store.set(MAP_OBS_COL, name)
        self.refresh()

    def _refresh_obs_choices(
        self,
        *,
        cols: Sequence[str],
        obs_auto: str,
    ) -> None:
        cands: list[str] = []
        for c in cols or []:
            cl = str(c).lower()
            if any(k in cl for k in ("obs", "truth", "actual", "target")):
                cands.append(str(c))
        if obs_auto and obs_auto not in cands:
            cands.insert(0, obs_auto)

        cur = str(
            self.store.get(MAP_OBS_COL, "") if self.store else ""
        ).strip()
        want = cur or obs_auto or ""

        with QSignalBlocker(self.cmb_obs):
            self.cmb_obs.clear()
            self.cmb_obs.addItem("(auto)")
            for c in cands:
                self.cmb_obs.addItem(c)

            if want:
                j = self.cmb_obs.findText(want)
                if j >= 0:
                    self.cmb_obs.setCurrentIndex(int(j))


    def refresh(self) -> None:
        if not HAS_MPL:
            return
        if not self.plot.is_ready():
            return

        df = self._ctx.df
        qs = self._ctx.q_cols or []
        obs_auto = self._ctx.obs_col or ""
        obs_override = str(
            self.store.get(MAP_OBS_COL, "")
            if self.store else ""
        ).strip()
        obs = obs_override or obs_auto

        self._refresh_obs_choices(
            cols=list(df.columns) if df is not None else [],
            obs_auto=obs,
        )


        if df is None or not len(df):
            _show_empty(
                self.plot,
                title="No data to plot",
                message="Select an active dataset to compute reliability.",
            )
            return
    
        if not obs or obs not in df.columns:
            _show_empty(
                self.plot,
                title="No observation column",
                message="Reliability needs an observation/truth column.",
                hint="Provide a column like obs / truth / target (depending on your export).",
            )
            return
    
        if not qs:
            _show_empty(
                self.plot,
                title="No quantiles found",
                message="PIT/reliability requires quantile columns (qXX).",
            )
            return
        
        self.plot.clear_axes()
        ax = self.plot.ax
        assert ax is not None
        
        # PIT: fraction of quantiles below obs
        # (simple discrete PIT)
        cols = [c for _, c in qs if c in df.columns]
        d = df[[obs] + cols].dropna()
        if not len(d):
          _show_empty(
              self.plot,
              title="No valid rows",
              message="No rows had both obs and quantiles available.",
          )
          return

        y = d[obs].values
        qmat = d[cols].values
        pit = (qmat < y[:, None]).mean(axis=1)

        ax.hist(pit, bins=20)
        ax.set_title("PIT histogram")
        ax.set_xlabel("PIT")
        ax.set_ylabel("Count")

        self.plot.draw()


class InspectorTab(_TabBase):
    """
    Inspector v1:
    - Group by (X, Y)
    - Plot time series for one point
    - Draw fan (quantile bands) if available
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._pts: list[tuple[float, float]] = []
        self._ids: list[int] = []
        self._active_sid: Optional[int] = None
        self._pts_key: tuple[str, str, str] = ("", "", "")

        self.sp_pt = QSpinBox(self)
        self.sp_pt.setRange(0, 0)

        self.lb_xy = QLabel("-", self)
        self.lb_xy.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.btn_rebuild = QToolButton(self)
        self.btn_rebuild.setText("↻")
        self.btn_rebuild.setToolTip("Rebuild point list")
        self.btn_rebuild.setAutoRaise(True)

        self.sp_max = QSpinBox(self)
        self.sp_max.setRange(1000, 2000000)
        self.sp_max.setValue(80000)

        self.sp_tol = QDoubleSpinBox(self)
        self.sp_tol.setRange(0.0, 1e6)
        self.sp_tol.setDecimals(9)
        self.sp_tol.setSingleStep(1e-6)
        self.sp_tol.setValue(1e-6)
        
        p = self.parent()
        s = getattr(p, "store", None)
        m = "lonlat" if s is None else s.get(
            "map.coord_mode", "lonlat"
        )
        m = str(m or "lonlat").strip().lower()
        if m in ("utm", "epsg") or (
                m == "auto"
                and s is not None
                and (
                    parse_epsg(s.get("map.src_epsg"))
                    or parse_epsg(s.get("map.utm_epsg"))
                )
            ):
            self.sp_tol.setDecimals(3)
            self.sp_tol.setSingleStep(0.5)
            self.sp_tol.setValue(1.0)

        self.chk_fan = QCheckBox("Fan", self)
        self.chk_fan.setChecked(True)

        self.cmb_band = QComboBox(self)
        self.cmb_band.addItems([
            "80% (q10-q90)",
            "50% (q25-q75)",
            "90% (q05-q95)",
        ])

        self.chk_mid = QCheckBox("Median line", self)
        self.chk_mid.setChecked(True)

        self.plot = MplPlot(
            title="Point time series",
            parent=self,
        )
        self.plot.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self._build_ui()
        self._connect()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        box = QGroupBox("Point", self)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)
    
        r0 = QHBoxLayout()
        r0.setContentsMargins(0, 0, 0, 0)
        r0.setSpacing(6)
        r0.addWidget(QLabel("Index:", box), 0)
        r0.addWidget(self.sp_pt, 1)
        r0.addWidget(self.btn_rebuild, 0)
        v.addLayout(r0)
    
        v.addWidget(QLabel("X/Y:", box), 0)
        v.addWidget(self.lb_xy, 0)
    
        v.addWidget(QLabel("Max points:", box), 0)
        v.addWidget(self.sp_max, 0)
    
        v.addWidget(QLabel("Match tol:", box), 0)
        v.addWidget(self.sp_tol, 0)
    
        v.addSpacing(8)
        v.addWidget(self.chk_fan, 0)
        v.addWidget(self.cmb_band, 0)
        v.addWidget(self.chk_mid, 0)
    
        v.addStretch(1)
    
        # Scroll wrapper (controls + plot)
        self._install_scroll(box, self.plot)


    def _connect(self) -> None:
        self.sp_pt.valueChanged.connect(self.refresh)
        self.sp_max.valueChanged.connect(self._rebuild_pts)
        self.sp_tol.valueChanged.connect(self.refresh)
        self.chk_fan.toggled.connect(self.refresh)
        self.cmb_band.currentIndexChanged.connect(self.refresh)
        self.chk_mid.toggled.connect(self.refresh)
        self.btn_rebuild.clicked.connect(self._rebuild_pts)

        if self.plot.actions is not None:
            self.plot.actions.btn_refresh.clicked.connect(
                self.refresh,
            )
            self.plot.actions.btn_clear.clicked.connect(
                    lambda: self.plot.show_empty(
                        title="No data to plot",
                        message="Plot cleared.",
                        hint="Click ↻ to refresh.",
                    )
                )
                

    # -------------------------
    # Context
    # -------------------------
    def set_context(self, ctx: MapAnaCtx) -> None:
        self._ctx = ctx
        self._ensure_pts()
        self.refresh()
        
    def set_id(self, sid: int) -> None:
        try:
            self._active_sid = int(sid)
        except Exception:
            self._active_sid = None
    
        if self._active_sid is not None and self._ids:
            try:
                i = self._ids.index(self._active_sid)
                self.sp_pt.setValue(int(i))
            except Exception:
                pass
        self.refresh()
        
    # def set_id(self, sid: int) -> None:
    #     """
    #     Jump to a point by its stable id (sample_idx).
    #     """
    #     try:
    #         si = int(sid)
    #     except Exception:
    #         return

    #     self._active_sid = si

    #     if self._ids:
    #         try:
    #             j = self._ids.index(si)
    #             self.sp_pt.setValue(int(j))
    #             return
    #         except Exception:
    #             pass

    #     self.refresh()
        
    def _ensure_pts(self) -> None:
        """
        Rebuild point list when:
        - file changed
        - x/y mapping changed
        """
        p = str(self._ctx.path or "")
        x = str(self._ctx.x or "")
        y = str(self._ctx.y or "")
        key = (p, x, y)

        if key != self._pts_key:
            self._pts_key = key
            self._rebuild_pts()

    def _rebuild_pts(self) -> None:
        self._pts = []
        self.lb_xy.setText("-")
        self.sp_pt.setRange(0, 0)
        self.sp_pt.setValue(0)

        path = self._ctx.path
        x = self._ctx.x
        y = self._ctx.y

        if path is None or not path.exists():
            return
        if not (x and y):
            return

        id_col = str(self._ctx.id_col or "").strip()
        
        # ---------------------------------------------
        # PATCH: Prefer in-memory frame (ctx.df)
        # ---------------------------------------------
        df = getattr(self._ctx, "df", None)
        if isinstance(df, pd.DataFrame) and (not df.empty):
            lim = int(self.sp_max.value())

            # Prefer a stable id column if available
            cid = str(id_col or "").strip()
            if not cid:
                if "sample_idx" in df.columns:
                    cid = "sample_idx"
                elif "sid" in df.columns:
                    cid = "sid"

            if cid and (cid in df.columns):
                sub = df[[cid, x, y]].copy()
                sub = sub.dropna(subset=[cid, x, y])

                # Keep one row per id (stable list)
                try:
                    sub[cid] = pd.to_numeric(
                        sub[cid], errors="coerce"
                    )
                except Exception:
                    pass

                sub = sub.dropna(subset=[cid])
                sub = sub.drop_duplicates(subset=[cid])

                if lim > 0 and len(sub) > lim:
                    sub = sub.iloc[:lim]

                try:
                    xs = pd.to_numeric(
                        sub[x], errors="coerce"
                    )
                    ys = pd.to_numeric(
                        sub[y], errors="coerce"
                    )
                    ok = xs.notna() & ys.notna()
                    sub = sub.loc[ok]
                    pts = list(
                        zip(
                            xs.loc[ok].astype(float),
                            ys.loc[ok].astype(float),
                        )
                    )
                except Exception:
                    pts = list(zip(sub[x], sub[y]))

                ids: list[int] = []
                for v in sub[cid].tolist():
                    try:
                        if pd.isna(v):
                            continue
                        ids.append(int(v))
                    except Exception:
                        continue

                self._pts = pts
                self._ids = ids

            else:
                # Fallback: unique (x,y)
                sub = df[[x, y]].copy()
                sub = sub.dropna(subset=[x, y])
                sub = sub.drop_duplicates(subset=[x, y])

                if lim > 0 and len(sub) > lim:
                    sub = sub.iloc[:lim]

                try:
                    xs = pd.to_numeric(
                        sub[x], errors="coerce"
                    )
                    ys = pd.to_numeric(
                        sub[y], errors="coerce"
                    )
                    ok = xs.notna() & ys.notna()
                    sub = sub.loc[ok]
                    self._pts = list(
                        zip(
                            xs.loc[ok].astype(float),
                            ys.loc[ok].astype(float),
                        )
                    )
                except Exception:
                    self._pts = list(zip(sub[x], sub[y]))

                self._ids = []

            if not self._pts:
                return

            # reuse the same "range + active sid + label"
            self.sp_pt.setRange(0, len(self._pts) - 1)

            j = 0
            if self._active_sid is not None and self._ids:
                try:
                    j = self._ids.index(int(self._active_sid))
                except Exception:
                    j = 0

            self.sp_pt.setValue(int(j))
            self._update_xy_label()
            return

        
        if id_col:
            pts, ids = self._scan_unique_sid_xy(
                path=path,
                x=x,
                y=y,
                id_col=id_col,
                limit=int(self.sp_max.value()),
            )
            self._pts = pts
            self._ids = ids
        else:
            self._ids = []
            self._pts = self._scan_unique_xy(
                path=path,
                x=x,
                y=y,
                limit=int(self.sp_max.value()),
            )

        if not self._pts:
            return

        self.sp_pt.setRange(0, len(self._pts) - 1)

        j = 0
        if self._active_sid is not None and self._ids:
            try:
                j = self._ids.index(int(self._active_sid))
            except Exception:
                j = 0

        self.sp_pt.setValue(int(j))
        self._update_xy_label()

    def _update_xy_label(self) -> None:
        if not self._pts:
            self.lb_xy.setText("-")
            return

        i = int(self.sp_pt.value())
        i = max(0, min(i, len(self._pts) - 1))
        x0, y0 = self._pts[i]
        sid = None
        if i < len(self._ids):
            sid = self._ids[i]

        if sid is None:
            self.lb_xy.setText(f"{x0:.6f}, {y0:.6f}")
        else:
            self.lb_xy.setText(
                f"sid={sid} | {x0:.6f}, {y0:.6f}"
            )

    # -------------------------
    # Data access
    # -------------------------
    def _scan_unique_xy(
        self,
        *,
        path: Path,
        x: str,
        y: str,
        limit: int,
    ) -> list[tuple[float, float]]:
        """
        Chunked unique scan for (x,y).

        Stops when reaching `limit`.
        """
        out: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()

        use = [x, y]
        try:
            chunks = pd.read_csv(
                path,
                usecols=use,
                chunksize=200_000,
            )
        except Exception:
            return out

        for ch in chunks:
            if x not in ch.columns or y not in ch.columns:
                continue

            d = ch[[x, y]].dropna()
            if d.empty:
                continue

            d = d.drop_duplicates()
            for xx, yy in d.itertuples(index=False, name=None):
                try:
                    key = (float(xx), float(yy))
                except Exception:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
                if len(out) >= int(limit):
                    return out

        return out

    def _scan_unique_sid_xy(
        self,
        *,
        path: Path,
        x: str,
        y: str,
        id_col: str,
        limit: int = 2000,
    ) -> tuple[list[tuple[float, float]], list[int]]:
        """
        Chunked unique scan for (sample_idx, x, y).

        Returns:
            (points, ids) where points[i] corresponds to ids[i].
        """
        pts: list[tuple[float, float]] = []
        ids: list[int] = []
        seen: set[int] = set()

        use = [id_col, x, y]
        try:
            chunks = pd.read_csv(
                path,
                usecols=use,
                chunksize=200_000,
            )
        except Exception:
            return pts, ids

        for ch in chunks:
            if (
                id_col not in ch.columns
                or x not in ch.columns
                or y not in ch.columns
            ):
                continue

            ch = ch[[id_col, x, y]].dropna()
            if ch.empty:
                continue

            ch = ch.drop_duplicates(subset=[id_col])
            for sid, xx, yy in ch.itertuples(index=False, name=None):
                try:
                    si = int(sid)
                except Exception:
                    continue
                if si in seen:
                    continue
                try:
                    pts.append((float(xx), float(yy)))
                    ids.append(si)
                    seen.add(si)
                except Exception:
                    continue

                if len(ids) >= int(limit):
                    return pts, ids

        return pts, ids

    def _load_id_series(

        self,
        *,
        path: Path,
        id_col: str,
        sid: int,
        t: str,
        z: str,
        step: str,
        qs: list[tuple[float, str]],
    ) -> Optional[pd.DataFrame]:
        use: list[str] = [id_col, t, z]
        if step:
            use.append(step)
        for _q, c in (qs or []):
            if c and c not in use:
                use.append(str(c))
        try:
            chunks = pd.read_csv(
                path,
                usecols=use,
                chunksize=200_000,
            )
        except Exception:
            return None

        keep: list[pd.DataFrame] = []
        for ch in chunks:
            if id_col not in ch.columns:
                continue
            try:
                m = ch[id_col] == sid
            except Exception:
                m = False
            part = ch.loc[m]
            if not part.empty:
                keep.append(part)

        if not keep:
            return None
        try:
            return pd.concat(keep, ignore_index=True)
        except Exception:
            return None


    def _load_point_series(
        self,
        *,
        path: Path,
        x: str,
        y: str,
        t: str,
        z: str,
        qs: list[tuple[float, str]],
        x0: float,
        y0: float,
        tol: float,
    ) -> Optional[pd.DataFrame]:
        """
        Chunked filter for a single (x0,y0).
        """
        cols: list[str] = []
        for c in (x, y, t, z):
            if c and c not in cols:
                cols.append(c)

        for _, qc in (qs or []):
            if qc and qc not in cols:
                cols.append(qc)

        try:
            it = pd.read_csv(
                path,
                usecols=cols,
                chunksize=200000,
            )
        except Exception:
            return None

        parts: list[pd.DataFrame] = []
        for ch in it:
            if x not in ch.columns or y not in ch.columns:
                continue

            dx = (ch[x] - float(x0)).abs()
            dy = (ch[y] - float(y0)).abs()
            m = (dx <= float(tol)) & (dy <= float(tol))

            sub = ch.loc[m]
            if len(sub):
                parts.append(sub)

        if not parts:
            return None

        df = pd.concat(parts, axis=0, ignore_index=True)
        return df

    # -------------------------
    # Plot
    # -------------------------
    def refresh(self) -> None:
        if not HAS_MPL:
            return
        if not self.plot.is_ready():
            return

        self._update_xy_label()

        # df0 = self._ctx.df
        path = self._ctx.path

        x = self._ctx.x
        y = self._ctx.y
        t = self._ctx.t
        z = self._ctx.z
        qs = self._ctx.q_cols or []
        step = self._ctx.step

        if path is None or not path.exists():
            _show_empty(
                self.plot,
                title="No dataset selected",
                message="Select an active file to inspect point time series.",
                hint="Map → Data panel → choose an active dataset.",
            )
            return
        
        if not (x and y and t and z):
            _show_empty(
                self.plot,
                title="Mapping incomplete",
                message="Inspector needs X/Y + Time col + Z to plot a point series.",
                hint="Pick columns in the header (X/Y/Z) and set Time col.",
            )
            return
        
        if not self._pts:
            _show_empty(
                self.plot,
                title="No points found",
                message="Could not build the (X,Y) point list from this file.",
                hint="Try increasing Max points or verify X/Y columns.",
            )
            return

        self.plot.clear_axes()
        ax = self.plot.ax
        assert ax is not None
        
        i = int(self.sp_pt.value())
        i = max(0, min(i, len(self._pts) - 1))
        x0, y0 = self._pts[i]

        tol = float(self.sp_tol.value())

        df = None
        if self._ids and i < len(self._ids) and self._ctx.id_col:
            sid = self._ids[i]
            df = self._load_id_series(
                path=path,
                id_col=self._ctx.id_col,
                sid=int(sid),
                t=t,
                z=z,
                step=step,
                qs=qs,
            )

        if df is None or not len(df):
            df = self._load_point_series(
                path=path,
                x=x,
                y=y,
                t=t,
                z=z,
                qs=qs,
                x0=x0,
                y0=y0,
                tol=tol,
            )

        if df is None or not len(df):
            _show_empty(
                self.plot,
                title="No rows for this point",
                message="No rows matched the selected (X,Y) within tolerance.",
                hint="Increase Match tol or click ↻ to rebuild points.",
            )
            return

        if t not in df.columns or z not in df.columns:
            self.plot.draw()
            return

        d = df.dropna(subset=[t, z]).copy()
        if not len(d):
            self.plot.draw()
            return

        # Sort by time
        try:
            d[t] = d[t].astype(float)
        except Exception:
            pass
        d = d.sort_values(t)

        xs = d[t].values
        yz = d[z].values

        ax.plot(xs, yz, linewidth=1.5)
        ax.set_xlabel(_human_label(t))
        ax.set_ylabel(self._value_label(z))

        sid = None
        if self._ids and i < len(self._ids):
            sid = self._ids[i]

        if sid is None:
            title = f"({x0:.4f}, {y0:.4f})"
        else:
            title = f"sid={sid} | ({x0:.4f}, {y0:.4f})"
        ax.set_title(title)

        if self.chk_fan.isChecked():
            self._draw_fan(ax=ax, d=d, t=t, qs=qs)

        self.plot.draw()

    def _draw_fan(
        self,
        *,
        ax,
        d: pd.DataFrame,
        t: str,
        qs: list[tuple[float, str]],
    ) -> None:
        if not qs:
            return

        lo, hi = self._pick_band(qs)
        if not lo or not hi:
            return
        if lo not in d.columns or hi not in d.columns:
            return

        dd = d.dropna(subset=[t, lo, hi])
        if not len(dd):
            return

        x = dd[t].values
        y0 = dd[lo].values
        y1 = dd[hi].values

        ax.fill_between(
            x,
            y0,
            y1,
            alpha=0.25,
        )

        if self.chk_mid.isChecked():
            mid = self._find_mid(qs)
            if mid and mid in dd.columns:
                ym = dd[mid].values
                ax.plot(x, ym, linewidth=1.2)

    def _pick_band(
        self,
        qs: list[tuple[float, str]],
    ) -> tuple[str, str]:
        s = str(self.cmb_band.currentText())

        mapping = {
            "80%": (0.10, 0.90),
            "50%": (0.25, 0.75),
            "90%": (0.05, 0.95),
        }
        qa = qb = None
        for k, (a, b) in mapping.items():
            if k in s:
                qa, qb = a, b

        if qa is None or qb is None:
            return "", ""

        lo = self._find_q(qs, qa)
        hi = self._find_q(qs, qb)
        return lo, hi

    def _find_mid(
        self,
        qs: list[tuple[float, str]],
    ) -> str:
        return self._find_q(qs, 0.50)

    def _find_q(
        self,
        qs: list[tuple[float, str]],
        q: float,
    ) -> str:
        best = ""
        err = 1e9
        for qq, col in qs:
            e = abs(float(qq) - float(q))
            if e < err:
                err = e
                best = col
        return best


    def set_xy(self, x: float, y: float) -> None:
        """
        Jump to nearest point in the current _pts list.
        """
        if not self._pts:
            return
    
        try:
            x0 = float(x)
            y0 = float(y)
        except Exception:
            return
    
        best_i = 0
        best_d = float("inf")
    
        for i, (xx, yy) in enumerate(self._pts):
            try:
                dx = float(xx) - x0
                dy = float(yy) - y0
            except Exception:
                continue
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_i = i
    
        # triggers refresh via valueChanged
        self.sp_pt.setValue(int(best_i))
        
    def current_xy(self) -> Optional[tuple[float, float]]:
        if not self._pts:
            return None
    
        i = int(self.sp_pt.value())
        i = max(0, min(i, len(self._pts) - 1))
        return self._pts[i]

def _show_empty(
    plot: MplPlot,
    *,
    title: str,
    message: str,
    hint: str = "",
) -> None:
    """
    Render a clean empty-state (no axes/ticks) inside the plot area.
    """
    if not HAS_MPL:
        return
    if not plot.is_ready():
        return

    plot.clear_axes()
    ax = plot.ax
    if ax is None:
        return

    ax.set_axis_off()

    ax.text(
        0.5, 0.58, title,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=11,
        fontweight="bold",
    )

    body = message if not hint else f"{message}\n\n{hint}"
    ax.text(
        0.5, 0.40, body,
        transform=ax.transAxes,
        ha="center", va="center",
        wrap=True,
    )
    plot.draw()
    
def _pick_1d(df: pd.DataFrame, key: str) -> pd.Series:
    """
    Return a 1D Series for column key.

    If df has duplicate column labels, df.loc[:, key] returns a DataFrame;
    we keep the first occurrence to stay 1D.
    """
    col = df.loc[:, key]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col
