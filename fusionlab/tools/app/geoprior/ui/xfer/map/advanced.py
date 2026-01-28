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

from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
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
)
from .interactions_ui import XferMapInteractionsBlock


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
        self.btn.setObjectName("xferMapAdvHeader")
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setChecked(False)
        self.btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
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

        self.btn.toggled.connect(self.body.setVisible)
        self.body.setVisible(False)

    def add_row(self, r: int, lbl: str, w: QWidget) -> None:
        lab = QLabel(lbl, self)
        lab.setObjectName("xferMapAdvLabel")
        self.body_l.addWidget(lab, r, 0, 1, 1)
        self.body_l.addWidget(w, r, 1, 1, 1)


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
        txt = (
            "<b>How to read the map</b><br>"
            "• Color/size follows the selected value.<br>"
            "• Hotspots highlight spatial extremes<br>"
            "  after downsampling + quantile filter.<br>"
            "• If your CSV uses UTM/projected<br>"
            "  coordinates, set Coords + EPSG<br>"
            "  so points land correctly."
        )

        self.sec_interp = _Fold(
            "Interpretation",
            "Guidance for subsidence transferability maps.",
            parent=self,
        )
        self.lbl_interp = QLabel(txt, self)
        self.lbl_interp.setWordWrap(True)
        self.sec_interp.body_l.addWidget(
            self.lbl_interp,
            0,
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
