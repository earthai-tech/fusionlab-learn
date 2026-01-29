# geoprior/ui/xfer/map/advanced.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.xfer.map.advanced

Advanced controls for the Xfer Map mode.

This panel is store-backed only:
- It writes xfer.map.* keys
- XferMapController reacts and re-renders

It is meant to live inside the floating MapToolDock.
"""

from __future__ import annotations

from typing import Optional, Set

from PyQt5.QtCore import Qt, QSize, QSignalBlocker
from PyQt5.QtWidgets import (
    QCheckBox, 
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QTextBrowser
)

from ....config.store import GeoConfigStore
from ..keys import (
    K_MAP_COORD_MODE,
    K_MAP_HOTSPOT_METRIC,
    K_MAP_HOTSPOT_MIN_SEP_KM,
    K_MAP_HOTSPOT_QUANTILE,
    K_MAP_MAX_POINTS,
    K_MAP_OPACITY,
    K_MAP_POINTS_MODE,
    K_MAP_SRC_EPSG,
    K_MAP_UTM_EPSG,
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
    K_MAP_INTERP_HTML,
    K_MAP_INTERP_TIP,
    K_MAP_INTERP_TIP
)

from .interactions_ui import XferMapInteractionsBlock
from .interpretation import map_help_html


__all__ = ["XferMapAdvancedPanel"]


class _Fold(QFrame):
    def __init__(
        self,
        title: str,
        hint: str,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setObjectName("xferMapAdvSection")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.btn = QToolButton(self)
        self.btn.setIconSize(QSize(14, 14))
        self.btn.setObjectName("xferMapAdvHeader")
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setChecked(False)
        self.btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn.setAutoRaise(True)

        self.hint = QLabel(hint, self)
        self.hint.setObjectName("xferMapAdvHint")
        self.hint.setWordWrap(True)

        hdr = QVBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(2)
        hdr.addWidget(self.btn, 0)
        hdr.addWidget(self.hint, 0)

        root.addLayout(hdr, 0)

        self.body = QFrame(self)
        self.body.setFrameShape(QFrame.NoFrame)
        self.body.setObjectName("xferMapAdvBody")

        self.body_l = QGridLayout(self.body)
        self.body_l.setContentsMargins(10, 6, 10, 6)
        self.body_l.setHorizontalSpacing(10)
        self.body_l.setVerticalSpacing(8)

        root.addWidget(self.body, 0)

        self.btn.setArrowType(Qt.RightArrow)
        # drive both: body visibility + arrow direction
        self.btn.toggled.connect(self._on_toggled)
        
        # start collapsed
        self.body.setVisible(False)

    def add_row(self, r: int, lbl: str, w: QWidget) -> None:
        lab = QLabel(lbl, self)
        lab.setObjectName("xferMapAdvLabel")
        self.body_l.addWidget(lab, r, 0, 1, 1)
        self.body_l.addWidget(w, r, 1, 1, 1)
        
    def _on_toggled(self, expanded: bool) -> None:
        self.body.setVisible(bool(expanded))
        self.btn.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )


class XferMapAdvancedPanel(QWidget):
    """
    Advanced map controls (View / Hotspots / Interactions / Interpretation).

    Store contract:
    - writes keys under xfer.map.*
    - controller listens and refreshes map
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store

        self.setObjectName("xferMapAdvancedPanel")
        self._build_ui()
        self._wire()
        self._apply_from_store(set())

        try:
            self._s.config_changed.connect(self._on_store_changed)
        except Exception:
            pass

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # -------------------------
        # 1) View
        # -------------------------
        self.sec_view = _Fold(
            "View",
            "Change draw density, opacity and coord reprojection.",
            parent=self,
        )

        self.sp_max_pts = QSpinBox(self)
        self.sp_max_pts.setRange(1000, 500000)
        self.sp_max_pts.setSingleStep(1000)

        self.sp_opacity = QDoubleSpinBox(self)
        self.sp_opacity.setDecimals(2)
        self.sp_opacity.setRange(0.05, 1.0)
        self.sp_opacity.setSingleStep(0.05)

        self.cmb_coord = QComboBox(self)
        self.cmb_coord.addItems(
            [
                "Lon/Lat degrees",
                "UTM (EPSG:326xx/327xx)",
                "Projected (EPSG)",
            ]
        )

        self.sp_utm = QSpinBox(self)
        self.sp_utm.setRange(1, 999999)
        self.sp_utm.setSingleStep(1)

        self.sp_src = QSpinBox(self)
        self.sp_src.setRange(1, 999999)
        self.sp_src.setSingleStep(1)

        self.sec_view.add_row(
            0,
            "Max points",
            self.sp_max_pts,
        )
        self.sec_view.add_row(1, "Opacity", self.sp_opacity)
        self.sec_view.add_row(2, "Coords", self.cmb_coord)
        self.sec_view.add_row(3, "UTM EPSG", self.sp_utm)
        self.sec_view.add_row(4, "SRC EPSG", self.sp_src)

        root.addWidget(self.sec_view, 0)

        # -------------------------
        # 2) Hotspots
        # -------------------------
        self.sec_hot = _Fold(
            "Hotspots",
            "Tune hotspot extraction (Top-N is in toolbar).",
            parent=self,
        )

        self.cmb_pts_mode = QComboBox(self)
        self.cmb_pts_mode.addItems(
            [
                "All points",
                "Hotspots only",
                "Hotspots + base",
            ]
        )

        self.sp_sep = QDoubleSpinBox(self)
        self.sp_sep.setDecimals(1)
        self.sp_sep.setRange(0.0, 50.0)
        self.sp_sep.setSingleStep(0.5)

        self.cmb_metric = QComboBox(self)
        self.cmb_metric.addItems(
            [
                "abs",
                "high",
                "low",
                "pos",
                "neg",
            ]
        )

        self.sp_q = QDoubleSpinBox(self)
        self.sp_q.setDecimals(3)
        self.sp_q.setRange(0.50, 0.999)
        self.sp_q.setSingleStep(0.005)

        self.sec_hot.add_row(0, "Points", self.cmb_pts_mode)
        self.sec_hot.add_row(1, "Min sep (km)", self.sp_sep)
        self.sec_hot.add_row(2, "Metric", self.cmb_metric)
        self.sec_hot.add_row(3, "Quantile", self.sp_q)

        # ---- NEW: Hotspot analytics (Radar + Arrows) ----
        self.lbl_hot_ana = QLabel("<b>Hotspot analytics</b>", self)
        self.lbl_hot_ana.setObjectName("xferMapAdvSectionTitle")
        self.sec_hot.body_l.addWidget(self.lbl_hot_ana, 4, 0, 1, 2)

        # Radar
        self.chk_radar = QCheckBox("Radar sweep hotspots", self)
        self.cmb_radar_target = QComboBox(self)
        self.cmb_radar_target.addItems(
            ["overlay", "both", "a", "b"]
        )
        self.cmb_radar_order = QComboBox(self)
        self.cmb_radar_order.addItems(
            ["score", "abs", "rank"]
        )
        self.sp_radar_ms = QSpinBox(self)
        self.sp_radar_ms.setRange(120, 5000)
        self.sp_radar_ms.setSingleStep(80)

        self.sp_radar_rkm = QDoubleSpinBox(self)
        self.sp_radar_rkm.setDecimals(1)
        self.sp_radar_rkm.setRange(0.5, 50.0)
        self.sp_radar_rkm.setSingleStep(0.5)

        self.sp_radar_rings = QSpinBox(self)
        self.sp_radar_rings.setRange(1, 8)
        self.sp_radar_rings.setSingleStep(1)

        self.sec_hot.add_row(5, "Radar", self.chk_radar)
        self.sec_hot.add_row(6, "Radar target", self.cmb_radar_target)
        self.sec_hot.add_row(7, "Radar order", self.cmb_radar_order)
        self.sec_hot.add_row(8, "Radar dwell (ms)", self.sp_radar_ms)
        self.sec_hot.add_row(9, "Radar radius (km)", self.sp_radar_rkm)
        self.sec_hot.add_row(10, "Radar rings", self.sp_radar_rings)

        # Links (arrows)
        self.chk_links = QCheckBox("Arrows between hotspots", self)
        self.cmb_links_mode = QComboBox(self)
        self.cmb_links_mode.addItems(["nearest", "rank", "knn"])

        self.sp_links_k = QSpinBox(self)
        self.sp_links_k.setRange(1, 6)
        self.sp_links_k.setSingleStep(1)

        self.sp_links_max = QSpinBox(self)
        self.sp_links_max.setRange(1, 80)
        self.sp_links_max.setSingleStep(1)

        self.chk_links_dist = QCheckBox("Show distance labels", self)

        self.sec_hot.add_row(11, "Links", self.chk_links)
        self.sec_hot.add_row(12, "Link mode", self.cmb_links_mode)
        self.sec_hot.add_row(13, "k (knn)", self.sp_links_k)
        self.sec_hot.add_row(14, "Max links", self.sp_links_max)
        self.sec_hot.add_row(15, "Distance", self.chk_links_dist)
        

        root.addWidget(self.sec_hot, 0)

        # -------------------------
        # 3) Interactions
        # -------------------------
        self.sec_int = _Fold(
            "Interactions",
            "Compare cities and derived layers.",
            parent=self,
        )
        
        self.int_block = XferMapInteractionsBlock(
            store=self._s,
            parent=self.sec_int.body,
        )
        self.sec_int.body_l.addWidget(self.int_block, 0, 0, 1, 2)
        root.addWidget(self.sec_int, 0)

        # -------------------------
        # 4) Interpretation
        # -------------------------
        self.sec_interp = _Fold(
            "Interpretation",
            "Guidance + recommended actions for transfers.",
            parent=self,
        )

        self.txt_interp = QTextBrowser(self)
        self.txt_interp.setObjectName("xferMapInterpDoc")
        self.txt_interp.setOpenExternalLinks(False)
        self.txt_interp.setFrameShape(QFrame.NoFrame)
        self.txt_interp.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self.txt_interp.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.txt_interp.setHtml(map_help_html())

        # Make it feel “expanded” when opened.
        self.txt_interp.setMinimumHeight(220)

        self.sec_interp.body_l.addWidget(
            self.txt_interp,
            0,
            0,
            1,
            2,
        )

        # Tip (short, action-oriented)
        # Placed LAST so it reflects the newest features
        # (Interactions/Extras + Radar/Links) without
        # cluttering the main help HTML.
        self.lbl_interp_tip = QLabel(self)
        self.lbl_interp_tip.setObjectName("xferMapInterpTip")
        self.lbl_interp_tip.setWordWrap(True)
        self.lbl_interp_tip.setText(
            "Tip: start with Shared scale for fair A↔B "
            "comparison, then switch to Auto to inspect "
            "within-city patterns. Use Interactions→Δ "
            "to see A−B shifts; enable Extras "
            "(Δ-hotspots / overlap intensity / buffered "
            "intersection) to judge domain shift. "
            "Turn on Radar/Arrows to scan hotspot "
            "rank/score and verify spatial alignment "
            "(Coords + EPSG)."
        )

        self.sec_interp.body_l.addWidget(
            self.lbl_interp_tip,
            1,
            0,
            1,
            2,
        )

        root.addWidget(self.sec_interp, 0)


        root.addStretch(1)

    # -------------------------
    # Store sync
    # -------------------------
    def _wire(self) -> None:
        self.sp_max_pts.valueChanged.connect(self._push_view)
        self.sp_opacity.valueChanged.connect(self._push_view)
        self.cmb_coord.currentIndexChanged.connect(
            self._push_view
        )
        self.sp_utm.valueChanged.connect(self._push_view)
        self.sp_src.valueChanged.connect(self._push_view)

        self.cmb_pts_mode.currentIndexChanged.connect(
            self._push_hot
        )
        self.sp_sep.valueChanged.connect(self._push_hot)
        self.cmb_metric.currentIndexChanged.connect(
            self._push_hot
        )
        self.sp_q.valueChanged.connect(self._push_hot)

        # NEW: hotspot analytics -> store
        self.chk_radar.toggled.connect(self._push_hot_ana)
        self.cmb_radar_target.currentIndexChanged.connect(
            self._push_hot_ana
        )
        self.cmb_radar_order.currentIndexChanged.connect(
            self._push_hot_ana
        )
        self.sp_radar_ms.valueChanged.connect(self._push_hot_ana)
        self.sp_radar_rkm.valueChanged.connect(self._push_hot_ana)
        self.sp_radar_rings.valueChanged.connect(self._push_hot_ana)

        self.chk_links.toggled.connect(self._push_hot_ana)
        self.cmb_links_mode.currentIndexChanged.connect(
            self._push_hot_ana
        )
        self.sp_links_k.valueChanged.connect(self._push_hot_ana)
        self.sp_links_max.valueChanged.connect(self._push_hot_ana)
        self.chk_links_dist.toggled.connect(self._push_hot_ana)

    def _on_store_changed(self, keys: object) -> None:
        ch = self._as_set(keys)
        if not ch:
            return
        self._apply_from_store(ch)

    @staticmethod
    def _as_set(keys: object) -> Set[str]:
        try:
            return set(keys or [])
        except Exception:
            return set()

    def _apply_from_store(self, ch: Set[str]) -> None:
        s = self._s

        b_max = QSignalBlocker(self.sp_max_pts)
        b_op = QSignalBlocker(self.sp_opacity)
        b_cm = QSignalBlocker(self.cmb_coord)
        b_utm = QSignalBlocker(self.sp_utm)
        b_src = QSignalBlocker(self.sp_src)
        b_pm = QSignalBlocker(self.cmb_pts_mode)
        b_sep = QSignalBlocker(self.sp_sep)
        b_met = QSignalBlocker(self.cmb_metric)
        b_q = QSignalBlocker(self.sp_q)
        b_radar = QSignalBlocker(self.chk_radar)
        b_rtar = QSignalBlocker(self.cmb_radar_target)
        b_rord = QSignalBlocker(self.cmb_radar_order)
        b_rms = QSignalBlocker(self.sp_radar_ms)
        b_rr = QSignalBlocker(self.sp_radar_rkm)
        b_rng = QSignalBlocker(self.sp_radar_rings)
        b_links = QSignalBlocker(self.chk_links)
        b_lm = QSignalBlocker(self.cmb_links_mode)
        b_lk = QSignalBlocker(self.sp_links_k)
        b_lmax = QSignalBlocker(self.sp_links_max)
        b_ld = QSignalBlocker(self.chk_links_dist)

        _ = (
            b_max,
            b_op,
            b_cm,
            b_utm,
            b_src,
            b_pm,
            b_sep,
            b_met,
            b_q,
            b_radar, 
            b_rtar,
            b_rord, 
            b_rms, 
            b_rr, 
            b_rng, 
            b_lm, 
            b_links, 
            b_lk, 
            b_lmax, 
            b_ld
            
        )

        if not ch or K_MAP_MAX_POINTS in ch:
            v = int(s.get(K_MAP_MAX_POINTS, 20000) or 20000)
            self.sp_max_pts.setValue(max(1000, v))

        if not ch or K_MAP_OPACITY in ch:
            v = float(s.get(K_MAP_OPACITY, 0.90) or 0.90)
            self.sp_opacity.setValue(max(0.05, min(1.0, v)))

        if not ch or K_MAP_COORD_MODE in ch:
            cm = str(s.get(K_MAP_COORD_MODE, "auto") or "")
            cm = cm.strip().lower()
            idx = 0
            if cm == "utm":
                idx = 1
            elif cm == "epsg":
                idx = 2
            self.cmb_coord.setCurrentIndex(idx)

        if not ch or K_MAP_UTM_EPSG in ch:
            v = int(s.get(K_MAP_UTM_EPSG, 32600) or 32600)
            self.sp_utm.setValue(max(1, v))

        if not ch or K_MAP_SRC_EPSG in ch:
            v = int(s.get(K_MAP_SRC_EPSG, 4326) or 4326)
            self.sp_src.setValue(max(1, v))

        if not ch or K_MAP_POINTS_MODE in ch:
            pm = str(s.get(K_MAP_POINTS_MODE, "all") or "")
            pm = pm.strip().lower()
            idx = 0
            if pm == "hotspots":
                idx = 1
            elif pm == "hotspots_plus":
                idx = 2
            self.cmb_pts_mode.setCurrentIndex(idx)

        if not ch or K_MAP_HOTSPOT_MIN_SEP_KM in ch:
            v = float(
                s.get(K_MAP_HOTSPOT_MIN_SEP_KM, 2.0) or 2.0
            )
            v = max(0.0, min(50.0, v))
            self.sp_sep.setValue(v)

        if not ch or K_MAP_HOTSPOT_METRIC in ch:
            m = str(
                s.get(K_MAP_HOTSPOT_METRIC, "abs") or "abs"
            )
            i = max(0, self.cmb_metric.findText(m))
            self.cmb_metric.setCurrentIndex(i)

        if not ch or K_MAP_HOTSPOT_QUANTILE in ch:
            q = float(
                s.get(K_MAP_HOTSPOT_QUANTILE, 0.98) or 0.98
            )
            q = max(0.50, min(0.999, q))
            self.sp_q.setValue(q)

        # ---- NEW: hotspot analytics ----
        if not ch or K_MAP_RADAR_ENABLE in ch:
            self.chk_radar.setChecked(
                bool(s.get(K_MAP_RADAR_ENABLE, False))
            )
        if not ch or K_MAP_RADAR_TARGET in ch:
            v = str(s.get(K_MAP_RADAR_TARGET, "overlay") or "overlay")
            i = max(0, self.cmb_radar_target.findText(v))
            self.cmb_radar_target.setCurrentIndex(i)
        if not ch or K_MAP_RADAR_ORDER in ch:
            v = str(s.get(K_MAP_RADAR_ORDER, "score") or "score")
            i = max(0, self.cmb_radar_order.findText(v))
            self.cmb_radar_order.setCurrentIndex(i)
        if not ch or K_MAP_RADAR_DWELL_MS in ch:
            v = int(s.get(K_MAP_RADAR_DWELL_MS, 520) or 520)
            self.sp_radar_ms.setValue(max(120, min(5000, v)))
        if not ch or K_MAP_RADAR_RADIUS_KM in ch:
            v = float(s.get(K_MAP_RADAR_RADIUS_KM, 8.0) or 8.0)
            self.sp_radar_rkm.setValue(max(0.5, min(50.0, v)))
        if not ch or K_MAP_RADAR_RINGS in ch:
            v = int(s.get(K_MAP_RADAR_RINGS, 3) or 3)
            self.sp_radar_rings.setValue(max(1, min(8, v)))

        if not ch or K_MAP_LINKS_ENABLE in ch:
            self.chk_links.setChecked(
                bool(s.get(K_MAP_LINKS_ENABLE, False))
            )
        if not ch or K_MAP_LINKS_MODE in ch:
            v = str(s.get(K_MAP_LINKS_MODE, "nearest") or "nearest")
            i = max(0, self.cmb_links_mode.findText(v))
            self.cmb_links_mode.setCurrentIndex(i)
        if not ch or K_MAP_LINKS_K in ch:
            v = int(s.get(K_MAP_LINKS_K, 1) or 1)
            self.sp_links_k.setValue(max(1, min(6, v)))
        if not ch or K_MAP_LINKS_MAX in ch:
            v = int(s.get(K_MAP_LINKS_MAX, 12) or 12)
            self.sp_links_max.setValue(max(1, min(80, v)))
        if not ch or K_MAP_LINKS_SHOW_DIST in ch:
            self.chk_links_dist.setChecked(
                bool(s.get(K_MAP_LINKS_SHOW_DIST, True))
            )
        if not ch or K_MAP_INTERP_HTML in ch:
            html = str(s.get(K_MAP_INTERP_HTML, "") or "")
            if html.strip():
                self.txt_interp.setHtml(html)
            else:
                self.txt_interp.setHtml(map_help_html())

        if not ch or K_MAP_INTERP_TIP in ch:
            tip = str(s.get(K_MAP_INTERP_TIP, "") or "")
            if tip.strip():
                self.lbl_interp_tip.setText(tip)
            else:
                self.lbl_interp_tip.setText(
                    "Tip: enable Radar/Arrows to inspect Δ-hotspots."
                )
        if not ch or K_MAP_INTERP_TIP in ch:
            tip = str(self._s.get(K_MAP_INTERP_TIP, "") or "")
            if tip.strip():
                self.lbl_interp_tip.setText(tip)

        self._enable_hot_ana()
        
        cm = self.cmb_coord.currentIndex()
        self.sp_utm.setEnabled(cm == 1)
        self.sp_src.setEnabled(cm == 2)

    # -------------------------
    # Push -> store
    # -------------------------
    def _push_view(self) -> None:
        s = self._s
        cm_i = int(self.cmb_coord.currentIndex())
        cm = "lonlat"
        if cm_i == 1:
            cm = "utm"
        elif cm_i == 2:
            cm = "epsg"

        with s.batch():
            s.set(K_MAP_MAX_POINTS, int(self.sp_max_pts.value()))
            s.set(K_MAP_OPACITY, float(self.sp_opacity.value()))
            s.set(K_MAP_COORD_MODE, cm)
            s.set(K_MAP_UTM_EPSG, int(self.sp_utm.value()))
            s.set(K_MAP_SRC_EPSG, int(self.sp_src.value()))

    def _push_hot(self) -> None:
        s = self._s
        idx = int(self.cmb_pts_mode.currentIndex())
        pm = "all"
        if idx == 1:
            pm = "hotspots"
        elif idx == 2:
            pm = "hotspots_plus"

        with s.batch():
            s.set(K_MAP_POINTS_MODE, pm)
            s.set(
                K_MAP_HOTSPOT_MIN_SEP_KM,
                float(self.sp_sep.value()),
            )
            s.set(
                K_MAP_HOTSPOT_METRIC,
                str(self.cmb_metric.currentText()),
            )
            s.set(
                K_MAP_HOTSPOT_QUANTILE,
                float(self.sp_q.value()),
            )

    def _push_hot_ana(self) -> None:
        s = self._s
        with s.batch():
            s.set(
                K_MAP_RADAR_ENABLE,
                bool(self.chk_radar.isChecked()),
            )
            s.set(
                K_MAP_RADAR_TARGET,
                str(self.cmb_radar_target.currentText()),
            )
            s.set(
                K_MAP_RADAR_ORDER,
                str(self.cmb_radar_order.currentText()),
            )
            s.set(
                K_MAP_RADAR_DWELL_MS,
                int(self.sp_radar_ms.value()),
            )
            s.set(
                K_MAP_RADAR_RADIUS_KM,
                float(self.sp_radar_rkm.value()),
            )
            s.set(
                K_MAP_RADAR_RINGS,
                int(self.sp_radar_rings.value()),
            )

            s.set(
                K_MAP_LINKS_ENABLE,
                bool(self.chk_links.isChecked()),
            )
            s.set(
                K_MAP_LINKS_MODE,
                str(self.cmb_links_mode.currentText()),
            )
            s.set(
                K_MAP_LINKS_K,
                int(self.sp_links_k.value()),
            )
            s.set(
                K_MAP_LINKS_MAX,
                int(self.sp_links_max.value()),
            )
            s.set(
                K_MAP_LINKS_SHOW_DIST,
                bool(self.chk_links_dist.isChecked()),
            )

        self._enable_hot_ana()

    def _enable_hot_ana(self) -> None:
        r = bool(self.chk_radar.isChecked())
        for w in (
            self.cmb_radar_target,
            self.cmb_radar_order,
            self.sp_radar_ms,
            self.sp_radar_rkm,
            self.sp_radar_rings,
        ):
            w.setEnabled(r)

        l = bool(self.chk_links.isChecked())
        self.cmb_links_mode.setEnabled(l)
        self.sp_links_max.setEnabled(l)
        self.chk_links_dist.setEnabled(l)
        self.sp_links_k.setEnabled(l and self.cmb_links_mode.currentText() == "knn")