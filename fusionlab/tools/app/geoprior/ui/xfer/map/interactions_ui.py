# geoprior/ui/xfer/map/interactions_ui.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Optional, Set

from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ....config.store import GeoConfigStore
from ..keys import (
    K_MAP_INSIGHT,
    K_MAP_SHARED,
    K_MAP_INTERACTION,
    K_MAP_INT_CELL_KM,
    K_MAP_INT_AGG,
    K_MAP_INT_DELTA,
    K_MAP_INT_HOT_ENABLE,
    K_MAP_INT_HOT_TOPN,
    K_MAP_INT_HOT_METRIC,
    K_MAP_INT_HOT_Q,
    K_MAP_INT_HOT_SEP,
    K_MAP_INT_INTENS_ENABLE,
    K_MAP_INT_BUF_ENABLE,
    K_MAP_INT_BUF_K,
)


__all__ = ["XferMapInteractionsBlock"]


class XferMapInteractionsBlock(QWidget):
    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store

        self.setObjectName("xferMapInteractionsBlock")
        self._build_ui()
        self._wire()

        self._apply_from_store(set())

        try:
            self._s.config_changed.connect(self._on_store)
        except Exception:
            pass

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # Top toggles (existing ones you had)
        self.chk_shared = QCheckBox("Shared scale", self)
        self.chk_badges = QCheckBox("Show badges", self)

        root.addWidget(self.chk_shared, 0)
        root.addWidget(self.chk_badges, 0)

        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("xferMapAdvDivider")
        root.addWidget(line, 0)

        # Base interaction mode + grid knobs
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        self.cmb_mode = QComboBox(self)
        self.cmb_mode.addItem("None", "none")
        self.cmb_mode.addItem("Zones", "zones")
        self.cmb_mode.addItem("A-only", "a_only")
        self.cmb_mode.addItem("B-only", "b_only")
        self.cmb_mode.addItem("Union", "union")
        self.cmb_mode.addItem("Intersection", "intersection")
        self.cmb_mode.addItem("Delta (A−B)", "delta")

        self.sp_cell = QDoubleSpinBox(self)
        self.sp_cell.setDecimals(2)
        self.sp_cell.setRange(0.20, 20.0)
        self.sp_cell.setSingleStep(0.25)

        self.cmb_agg = QComboBox(self)
        self.cmb_agg.addItem("mean", "mean")
        self.cmb_agg.addItem("median", "median")
        self.cmb_agg.addItem("max", "max")

        self.cmb_delta = QComboBox(self)
        self.cmb_delta.addItem("A − B", "a_minus_b")
        self.cmb_delta.addItem("B − A", "b_minus_a")

        form.addRow(QLabel("Mode:"), self.cmb_mode)
        form.addRow(QLabel("Cell (km):"), self.sp_cell)
        form.addRow(QLabel("Agg:"), self.cmb_agg)
        form.addRow(QLabel("Delta:"), self.cmb_delta)

        root.addLayout(form, 0)

        # Extras
        lbl = QLabel("<b>Extras</b>", self)
        lbl.setObjectName("xferMapAdvSectionTitle")
        root.addWidget(lbl, 0)

        self.chk_dhot = QCheckBox("Δ hotspots", self)
        self.chk_intens = QCheckBox("Overlap intensity", self)
        self.chk_buf = QCheckBox("Buffered intersection", self)

        root.addWidget(self.chk_dhot, 0)
        root.addWidget(self.chk_intens, 0)
        root.addWidget(self.chk_buf, 0)

        extras = QFormLayout()
        extras.setLabelAlignment(Qt.AlignLeft)
        extras.setHorizontalSpacing(10)
        extras.setVerticalSpacing(6)

        self.sp_dhot_n = QSpinBox(self)
        self.sp_dhot_n.setRange(1, 50)
        self.sp_dhot_n.setSingleStep(1)

        self.cmb_dhot_metric = QComboBox(self)
        self.cmb_dhot_metric.addItem("abs", "abs")
        self.cmb_dhot_metric.addItem("high", "high")
        self.cmb_dhot_metric.addItem("low", "low")
        self.cmb_dhot_metric.addItem("pos", "pos")
        self.cmb_dhot_metric.addItem("neg", "neg")

        self.sp_dhot_q = QDoubleSpinBox(self)
        self.sp_dhot_q.setDecimals(3)
        self.sp_dhot_q.setRange(0.50, 0.999)
        self.sp_dhot_q.setSingleStep(0.005)

        self.sp_dhot_sep = QDoubleSpinBox(self)
        self.sp_dhot_sep.setDecimals(1)
        self.sp_dhot_sep.setRange(0.0, 50.0)
        self.sp_dhot_sep.setSingleStep(0.5)

        self.sp_buf_k = QSpinBox(self)
        self.sp_buf_k.setRange(1, 10)
        self.sp_buf_k.setSingleStep(1)

        extras.addRow(QLabel("Δ TopN:"), self.sp_dhot_n)
        extras.addRow(QLabel("Δ Metric:"), self.cmb_dhot_metric)
        extras.addRow(QLabel("Δ Quantile:"), self.sp_dhot_q)
        extras.addRow(QLabel("Δ Sep (km):"), self.sp_dhot_sep)
        extras.addRow(QLabel("Buffer k:"), self.sp_buf_k)

        root.addLayout(extras, 0)
        root.addStretch(1)

    # -------------------------
    # Wiring
    # -------------------------
    def _wire(self) -> None:
        self.chk_shared.toggled.connect(self._push_base)
        self.chk_badges.toggled.connect(self._push_base)

        self.cmb_mode.currentIndexChanged.connect(
            self._push_interaction
        )
        self.sp_cell.valueChanged.connect(self._push_interaction)
        self.cmb_agg.currentIndexChanged.connect(
            self._push_interaction
        )
        self.cmb_delta.currentIndexChanged.connect(
            self._push_interaction
        )

        self.chk_dhot.toggled.connect(self._push_extras)
        self.chk_intens.toggled.connect(self._push_extras)
        self.chk_buf.toggled.connect(self._push_extras)

        self.sp_dhot_n.valueChanged.connect(self._push_extras)
        self.cmb_dhot_metric.currentIndexChanged.connect(
            self._push_extras
        )
        self.sp_dhot_q.valueChanged.connect(self._push_extras)
        self.sp_dhot_sep.valueChanged.connect(self._push_extras)
        self.sp_buf_k.valueChanged.connect(self._push_extras)

    def _on_store(self, keys: object) -> None:
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

        _ = (
            QSignalBlocker(self.chk_shared),
            QSignalBlocker(self.chk_badges),
            QSignalBlocker(self.cmb_mode),
            QSignalBlocker(self.sp_cell),
            QSignalBlocker(self.cmb_agg),
            QSignalBlocker(self.cmb_delta),
            QSignalBlocker(self.chk_dhot),
            QSignalBlocker(self.chk_intens),
            QSignalBlocker(self.chk_buf),
            QSignalBlocker(self.sp_dhot_n),
            QSignalBlocker(self.cmb_dhot_metric),
            QSignalBlocker(self.sp_dhot_q),
            QSignalBlocker(self.sp_dhot_sep),
            QSignalBlocker(self.sp_buf_k),
        )

        if not ch or K_MAP_SHARED in ch:
            self.chk_shared.setChecked(
                bool(s.get(K_MAP_SHARED, True))
            )

        if not ch or K_MAP_INSIGHT in ch:
            self.chk_badges.setChecked(
                bool(s.get(K_MAP_INSIGHT, False))
            )

        if not ch or K_MAP_INTERACTION in ch:
            mode = str(s.get(K_MAP_INTERACTION, "none") or "none")
            i = self._find_data(self.cmb_mode, mode)
            self.cmb_mode.setCurrentIndex(i)

        if not ch or K_MAP_INT_CELL_KM in ch:
            v = float(s.get(K_MAP_INT_CELL_KM, 2.0) or 2.0)
            self.sp_cell.setValue(max(0.2, min(20.0, v)))

        if not ch or K_MAP_INT_AGG in ch:
            a = str(s.get(K_MAP_INT_AGG, "mean") or "mean")
            i = self._find_data(self.cmb_agg, a)
            self.cmb_agg.setCurrentIndex(i)

        if not ch or K_MAP_INT_DELTA in ch:
            d = str(
                s.get(K_MAP_INT_DELTA, "a_minus_b") or "a_minus_b"
            )
            i = self._find_data(self.cmb_delta, d)
            self.cmb_delta.setCurrentIndex(i)

        if not ch or K_MAP_INT_HOT_ENABLE in ch:
            self.chk_dhot.setChecked(
                bool(s.get(K_MAP_INT_HOT_ENABLE, False))
            )

        if not ch or K_MAP_INT_INTENS_ENABLE in ch:
            self.chk_intens.setChecked(
                bool(s.get(K_MAP_INT_INTENS_ENABLE, False))
            )

        if not ch or K_MAP_INT_BUF_ENABLE in ch:
            self.chk_buf.setChecked(
                bool(s.get(K_MAP_INT_BUF_ENABLE, False))
            )

        if not ch or K_MAP_INT_HOT_TOPN in ch:
            self.sp_dhot_n.setValue(
                int(s.get(K_MAP_INT_HOT_TOPN, 8) or 8)
            )

        if not ch or K_MAP_INT_HOT_METRIC in ch:
            m = str(s.get(K_MAP_INT_HOT_METRIC, "abs") or "abs")
            i = self._find_data(self.cmb_dhot_metric, m)
            self.cmb_dhot_metric.setCurrentIndex(i)

        if not ch or K_MAP_INT_HOT_Q in ch:
            q = float(s.get(K_MAP_INT_HOT_Q, 0.98) or 0.98)
            self.sp_dhot_q.setValue(max(0.5, min(0.999, q)))

        if not ch or K_MAP_INT_HOT_SEP in ch:
            v = float(s.get(K_MAP_INT_HOT_SEP, 2.0) or 2.0)
            self.sp_dhot_sep.setValue(max(0.0, min(50.0, v)))

        if not ch or K_MAP_INT_BUF_K in ch:
            k = int(s.get(K_MAP_INT_BUF_K, 1) or 1)
            self.sp_buf_k.setValue(max(1, min(10, k)))

        self._enable_children()

    @staticmethod
    def _find_data(cb: QComboBox, value: str) -> int:
        for i in range(cb.count()):
            if str(cb.itemData(i)) == str(value):
                return i
        return 0

    def _enable_children(self) -> None:
        en = bool(self.chk_dhot.isChecked())
        self.sp_dhot_n.setEnabled(en)
        self.cmb_dhot_metric.setEnabled(en)
        self.sp_dhot_q.setEnabled(en)
        self.sp_dhot_sep.setEnabled(en)

        self.sp_buf_k.setEnabled(bool(self.chk_buf.isChecked()))

    # -------------------------
    # Store push
    # -------------------------
    def _push_base(self) -> None:
        s = self._s
        with s.batch():
            s.set(K_MAP_SHARED, bool(self.chk_shared.isChecked()))
            s.set(K_MAP_INSIGHT, bool(self.chk_badges.isChecked()))

    def _push_interaction(self) -> None:
        s = self._s
        mode = str(self.cmb_mode.currentData() or "none")
        agg = str(self.cmb_agg.currentData() or "mean")
        delt = str(self.cmb_delta.currentData() or "a_minus_b")

        with s.batch():
            s.set(K_MAP_INTERACTION, mode)
            s.set(K_MAP_INT_CELL_KM, float(self.sp_cell.value()))
            s.set(K_MAP_INT_AGG, agg)
            s.set(K_MAP_INT_DELTA, delt)

    def _push_extras(self) -> None:
        s = self._s
        with s.batch():
            s.set(
                K_MAP_INT_HOT_ENABLE,
                bool(self.chk_dhot.isChecked()),
            )
            s.set(
                K_MAP_INT_INTENS_ENABLE,
                bool(self.chk_intens.isChecked()),
            )
            s.set(
                K_MAP_INT_BUF_ENABLE,
                bool(self.chk_buf.isChecked()),
            )
            s.set(
                K_MAP_INT_HOT_TOPN,
                int(self.sp_dhot_n.value()),
            )
            s.set(
                K_MAP_INT_HOT_METRIC,
                str(self.cmb_dhot_metric.currentData() or "abs"),
            )
            s.set(
                K_MAP_INT_HOT_Q,
                float(self.sp_dhot_q.value()),
            )
            s.set(
                K_MAP_INT_HOT_SEP,
                float(self.sp_dhot_sep.value()),
            )
            s.set(
                K_MAP_INT_BUF_K,
                int(self.sp_buf_k.value()),
            )

        self._enable_children()
