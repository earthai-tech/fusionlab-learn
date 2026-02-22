# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.run.run_center

Center panel (RUN mode) for XferTab.

Cards:
- Cities & splits
- Outputs & alignments
- Strategy & warm-start
- Results & view
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)

from ....config.store import GeoConfigStore
from ...common.card_registry import CardRegistry

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]


def _blocked(w: QWidget) -> QSignalBlocker:
    return QSignalBlocker(w)


def _parse_float_list(txt: str) -> Optional[List[float]]:
    raw = (txt or "").strip()
    if not raw:
        return None
    out: List[float] = []
    for b in [x.strip() for x in raw.split(",")]:
        if not b:
            continue
        try:
            out.append(float(b))
        except Exception:
            return None
    return out or None


class OutputsAlignmentsSection(QWidget):
    """
    Outputs & alignment (store-backed).
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._build_ui()
        self._connect()
        self._sync_from_store(set())
        self._s.config_changed.connect(self._on_store)

    def get_quantiles(self) -> Optional[List[float]]:
        return _parse_float_list(self.ed_q.text())

    def _build_ui(self) -> None:
        root = QGridLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setHorizontalSpacing(10)
        root.setVerticalSpacing(10)
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 1)

        self.ed_q = QLineEdit(self)
        self.ed_q.setPlaceholderText("0.1,0.5,0.9")

        self.lbl_q = QLabel("AUTO", self)
        self.lbl_q.setObjectName("xferAdvChip")
        self.lbl_q.setAlignment(Qt.AlignCenter)

        self.btn_q_clear = QToolButton(self)
        self.btn_q_clear.setObjectName("miniAction")
        self.btn_q_clear.setIcon(
            self.style().standardIcon(
                QStyle.SP_DialogCloseButton
            )
        )

        qrow = QWidget(self)
        ql = QHBoxLayout(qrow)
        ql.setContentsMargins(0, 0, 0, 0)
        ql.setSpacing(6)
        ql.addWidget(self.ed_q, 1)
        ql.addWidget(self.lbl_q, 0)
        ql.addWidget(self.btn_q_clear, 0)

        root.addWidget(QLabel("Quantiles:"), 0, 0)
        root.addWidget(qrow, 0, 1)

        self.chk_json = QCheckBox("Write JSON", self)
        self.chk_csv = QCheckBox("Write CSV", self)

        io = QWidget(self)
        iol = QHBoxLayout(io)
        iol.setContentsMargins(0, 0, 0, 0)
        iol.setSpacing(10)
        iol.addWidget(self.chk_json)
        iol.addWidget(self.chk_csv)
        iol.addStretch(1)

        root.addWidget(QLabel("Formats:"), 1, 0)
        root.addWidget(io, 1, 1)

        self.chk_prefer = QCheckBox(
            "Prefer tuned calibrator",
            self,
        )
        root.addWidget(self.chk_prefer, 2, 0, 1, 2)

        self.cmb_align = QComboBox(self)
        self.cmb_align.addItem(
            "Align by name (pad)",
            "align_by_name_pad",
        )
        self.cmb_align.addItem("Strict", "strict")

        root.addWidget(QLabel("Align:"), 3, 0)
        root.addWidget(self.cmb_align, 3, 1)

        self.cmb_dyn = QComboBox(self)
        self._fill_opt_bool(self.cmb_dyn)

        root.addWidget(QLabel("Reorder dynamic:"), 4, 0)
        root.addWidget(self.cmb_dyn, 4, 1)

        self.cmb_fut = QComboBox(self)
        self._fill_opt_bool(self.cmb_fut)

        root.addWidget(QLabel("Reorder future:"), 5, 0)
        root.addWidget(self.cmb_fut, 5, 1)

        self.sp_cov = QDoubleSpinBox(self)
        self.sp_cov.setRange(0.10, 0.99)
        self.sp_cov.setSingleStep(0.01)
        self.sp_cov.setDecimals(2)

        root.addWidget(QLabel("Interval target:"), 6, 0)
        root.addWidget(self.sp_cov, 6, 1)

        self.cmb_ep = QComboBox(self)
        self.cmb_ep.addItem("serve", "serve")
        self.cmb_ep.addItem("export", "export")

        root.addWidget(QLabel("Endpoint:"), 7, 0)
        root.addWidget(self.cmb_ep, 7, 1)

        self.chk_phys = QCheckBox(
            "Export physics payload",
            self,
        )
        self.chk_phys_csv = QCheckBox(
            "Export physical params CSV",
            self,
        )
        self.chk_eval_f = QCheckBox(
            "Write future eval CSV",
            self,
        )

        ex = QGridLayout()
        ex.setContentsMargins(0, 0, 0, 0)
        ex.setHorizontalSpacing(10)
        ex.setVerticalSpacing(8)

        ex.addWidget(self.chk_phys, 0, 0)
        ex.addWidget(self.chk_phys_csv, 0, 1)
        ex.addWidget(self.chk_eval_f, 1, 0, 1, 2)

        exw = QWidget(self)
        exw.setLayout(ex)

        root.addWidget(QLabel("Exports:"), 8, 0)
        root.addWidget(exw, 8, 1)

        self._update_q_chip()

    def _fill_opt_bool(self, cmb: QComboBox) -> None:
        cmb.clear()
        cmb.addItem("Auto", None)
        cmb.addItem("Allow", True)
        cmb.addItem("Block", False)

    def _connect(self) -> None:
        self.btn_q_clear.clicked.connect(
            lambda: self.ed_q.setText("")
        )
        self.btn_q_clear.clicked.connect(self._push)

        self.ed_q.editingFinished.connect(self._push)
        self.ed_q.editingFinished.connect(self._update_q_chip)

        for cb in (
            self.chk_json,
            self.chk_csv,
            self.chk_prefer,
            self.chk_phys,
            self.chk_phys_csv,
            self.chk_eval_f,
        ):
            cb.toggled.connect(self._push)

        self.cmb_align.currentIndexChanged.connect(self._push)
        self.cmb_dyn.currentIndexChanged.connect(self._push)
        self.cmb_fut.currentIndexChanged.connect(self._push)
        self.sp_cov.valueChanged.connect(self._push)
        self.cmb_ep.currentIndexChanged.connect(self._push)

    def _on_store(self, keys: object) -> None:
        try:
            ch = set(keys or [])
        except Exception:
            ch = set()
        if not ch:
            return
        want = {
            "xfer.quantiles_override",
            "xfer.write_json",
            "xfer.write_csv",
            "xfer.prefer_tuned",
            "xfer.align_policy",
            "xfer.allow_reorder_dynamic",
            "xfer.allow_reorder_future",
            "xfer.interval_target",
            "xfer.load_endpoint",
            "xfer.export_physics_payload",
            "xfer.export_physical_parameters_csv",
            "xfer.write_eval_future_csv",
        }
        if ch & want:
            self._sync_from_store(ch)

    def _sync_from_store(self, keys: set[str]) -> None:
        s = self._s

        q = s.get("xfer.quantiles_override", None)
        txt = ""
        if q:
            try:
                txt = ",".join(str(x) for x in q)
            except Exception:
                txt = ""
        with _blocked(self.ed_q):
            self.ed_q.setText(txt)

        with _blocked(self.chk_json):
            self.chk_json.setChecked(
                bool(s.get("xfer.write_json", True))
            )
        with _blocked(self.chk_csv):
            self.chk_csv.setChecked(
                bool(s.get("xfer.write_csv", True))
            )
        with _blocked(self.chk_prefer):
            self.chk_prefer.setChecked(
                bool(s.get("xfer.prefer_tuned", True))
            )

        self._set_combo(self.cmb_align, s.get(
            "xfer.align_policy",
            "align_by_name_pad",
        ))
        self._set_combo(self.cmb_dyn, s.get(
            "xfer.allow_reorder_dynamic",
            None,
        ))
        self._set_combo(self.cmb_fut, s.get(
            "xfer.allow_reorder_future",
            None,
        ))

        with _blocked(self.sp_cov):
            self.sp_cov.setValue(
                float(s.get("xfer.interval_target", 0.80))
            )

        self._set_combo(self.cmb_ep, s.get(
            "xfer.load_endpoint",
            "serve",
        ))

        with _blocked(self.chk_phys):
            self.chk_phys.setChecked(bool(
                s.get("xfer.export_physics_payload", True)
            ))
        with _blocked(self.chk_phys_csv):
            self.chk_phys_csv.setChecked(bool(
                s.get(
                    "xfer.export_physical_parameters_csv",
                    True,
                )
            ))
        with _blocked(self.chk_eval_f):
            self.chk_eval_f.setChecked(bool(
                s.get("xfer.write_eval_future_csv", True)
            ))

        self._update_q_chip()

    def _set_combo(self, cmb: QComboBox, v: Any) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == v:
                with _blocked(cmb):
                    cmb.setCurrentIndex(i)
                return

    def _q_status(self) -> Tuple[Optional[List[float]], bool, str]:
        raw = (self.ed_q.text() or "").strip()
        if not raw:
            return None, True, "AUTO"
        q = _parse_float_list(raw)
        if q is None:
            return None, False, "INVALID"
        return q, True, "OK"

    def _update_q_chip(self) -> None:
        _q, ok, txt = self._q_status()
        self.lbl_q.setText(txt)
        self.lbl_q.setProperty("ok", bool(ok))
        self.lbl_q.style().unpolish(self.lbl_q)
        self.lbl_q.style().polish(self.lbl_q)

    def _push(self) -> None:
        s = self._s

        q, ok, _t = self._q_status()
        raw = (self.ed_q.text() or "").strip()
        if ok or (not raw):
            s.set("xfer.quantiles_override", q)

        s.set("xfer.write_json", bool(self.chk_json.isChecked()))
        s.set("xfer.write_csv", bool(self.chk_csv.isChecked()))
        s.set(
            "xfer.prefer_tuned",
            bool(self.chk_prefer.isChecked()),
        )
        s.set(
            "xfer.align_policy",
            str(self.cmb_align.currentData()),
        )
        s.set(
            "xfer.allow_reorder_dynamic",
            self.cmb_dyn.currentData(),
        )
        s.set(
            "xfer.allow_reorder_future",
            self.cmb_fut.currentData(),
        )
        s.set("xfer.interval_target", float(self.sp_cov.value()))
        s.set(
            "xfer.load_endpoint",
            str(self.cmb_ep.currentData()),
        )
        s.set(
            "xfer.export_physics_payload",
            bool(self.chk_phys.isChecked()),
        )
        s.set(
            "xfer.export_physical_parameters_csv",
            bool(self.chk_phys_csv.isChecked()),
        )
        s.set(
            "xfer.write_eval_future_csv",
            bool(self.chk_eval_f.isChecked()),
        )


class StrategyWarmStartSection(QWidget):
    """
    Strategy & warm-start (store-backed).
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._build_ui()
        self._connect()
        self._sync_from_store(set())
        self._s.config_changed.connect(self._on_store)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        self.chk_base = QCheckBox(
            "In-domain baselines (A→A, B→B)",
            self,
        )
        self.chk_xfer = QCheckBox(
            "Zero-shot transfer (A→B, B→A)",
            self,
        )
        self.chk_warm = QCheckBox(
            "Warm-start fine-tune",
            self,
        )

        root.addWidget(self.chk_base)
        root.addWidget(self.chk_xfer)
        root.addWidget(self.chk_warm)

        rm = QHBoxLayout()
        rm.setContentsMargins(0, 0, 0, 0)
        rm.setSpacing(10)

        self.chk_as_is = QCheckBox("as-is", self)
        self.chk_strict = QCheckBox("strict", self)

        rm.addWidget(QLabel("Rescale variants:"))
        rm.addWidget(self.chk_as_is)
        rm.addWidget(self.chk_strict)
        rm.addStretch(1)

        root.addLayout(rm)

        self.warm_box = QGroupBox("Warm-start settings", self)
        warm = QGridLayout(self.warm_box)
        warm.setContentsMargins(8, 8, 8, 8)
        warm.setHorizontalSpacing(10)
        warm.setVerticalSpacing(8)

        self.cmb_split = QComboBox(self.warm_box)
        self.cmb_split.addItem("train", "train")
        self.cmb_split.addItem("val", "val")
        self.cmb_split.addItem("test", "test")

        self.sp_samples = QSpinBox(self.warm_box)
        self.sp_samples.setRange(1, 1_000_000)

        self.sp_frac = QDoubleSpinBox(self.warm_box)
        self.sp_frac.setRange(0.0, 1.0)
        self.sp_frac.setDecimals(2)
        self.sp_frac.setSingleStep(0.05)

        self.sp_epochs = QSpinBox(self.warm_box)
        self.sp_epochs.setRange(1, 10_000)

        self.sp_lr = QDoubleSpinBox(self.warm_box)
        self.sp_lr.setRange(1e-8, 1.0)
        self.sp_lr.setDecimals(8)
        self.sp_lr.setSingleStep(1e-4)

        self.sp_seed = QSpinBox(self.warm_box)
        self.sp_seed.setRange(0, 2_147_483_647)

        warm.addWidget(QLabel("Warm split:"), 0, 0)
        warm.addWidget(self.cmb_split, 0, 1)
        warm.addWidget(QLabel("Warm epochs:"), 0, 2)
        warm.addWidget(self.sp_epochs, 0, 3)

        warm.addWidget(QLabel("Warm samples:"), 1, 0)
        warm.addWidget(self.sp_samples, 1, 1)
        warm.addWidget(QLabel("Warm lr:"), 1, 2)
        warm.addWidget(self.sp_lr, 1, 3)

        warm.addWidget(QLabel("Warm frac:"), 2, 0)
        warm.addWidget(self.sp_frac, 2, 1)
        warm.addWidget(QLabel("Warm seed:"), 2, 2)
        warm.addWidget(self.sp_seed, 2, 3)

        warm.setColumnStretch(1, 1)
        warm.setColumnStretch(3, 1)

        root.addWidget(self.warm_box)

        self._update_warm_enabled()

    def _connect(self) -> None:
        for cb in (self.chk_base, self.chk_xfer, self.chk_warm):
            cb.toggled.connect(self._push)

        for cb in (self.chk_as_is, self.chk_strict):
            cb.toggled.connect(self._push)

        self.chk_warm.toggled.connect(
            lambda _v: self._update_warm_enabled()
        )

        self.cmb_split.currentIndexChanged.connect(self._push)
        self.sp_samples.valueChanged.connect(self._push)
        self.sp_frac.valueChanged.connect(self._push)
        self.sp_epochs.valueChanged.connect(self._push)
        self.sp_lr.valueChanged.connect(self._push)
        self.sp_seed.valueChanged.connect(self._push)

    def _on_store(self, keys: object) -> None:
        try:
            ch = set(keys or [])
        except Exception:
            ch = set()
        if not ch:
            return
        want = {
            "xfer.strategies",
            "xfer.rescale_modes",
            "xfer.warm_split",
            "xfer.warm_samples",
            "xfer.warm_frac",
            "xfer.warm_epochs",
            "xfer.warm_lr",
            "xfer.warm_seed",
        }
        if ch & want:
            self._sync_from_store(ch)

    def _sync_from_store(self, keys: set[str]) -> None:
        s = self._s

        self._sync_strats(s.get("xfer.strategies", None))
        self._sync_modes(s.get("xfer.rescale_modes", None))

        self._set_combo(self.cmb_split, s.get(
            "xfer.warm_split",
            "train",
        ))

        with _blocked(self.sp_samples):
            self.sp_samples.setValue(int(
                s.get("xfer.warm_samples", 20000)
            ))
        with _blocked(self.sp_frac):
            self.sp_frac.setValue(float(
                s.get("xfer.warm_frac", 0.0)
            ))
        with _blocked(self.sp_epochs):
            self.sp_epochs.setValue(int(
                s.get("xfer.warm_epochs", 3)
            ))
        with _blocked(self.sp_lr):
            self.sp_lr.setValue(float(
                s.get("xfer.warm_lr", 1e-4)
            ))
        with _blocked(self.sp_seed):
            self.sp_seed.setValue(int(
                s.get("xfer.warm_seed", 0)
            ))

        self._update_warm_enabled()

    def _set_combo(self, cmb: QComboBox, v: Any) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == v:
                with _blocked(cmb):
                    cmb.setCurrentIndex(i)
                return

    def _sync_strats(self, v: Any) -> None:
        ss = set(v or [])
        with _blocked(self.chk_base):
            self.chk_base.setChecked("baseline" in ss)
        with _blocked(self.chk_xfer):
            self.chk_xfer.setChecked("xfer" in ss)
        with _blocked(self.chk_warm):
            self.chk_warm.setChecked("warm" in ss)

    def _sync_modes(self, v: Any) -> None:
        mm = set(v or [])
        with _blocked(self.chk_as_is):
            self.chk_as_is.setChecked("as_is" in mm)
        with _blocked(self.chk_strict):
            self.chk_strict.setChecked("strict" in mm)

    def _get_strats(self) -> Optional[List[str]]:
        out: List[str] = []
        if self.chk_base.isChecked():
            out.append("baseline")
        if self.chk_xfer.isChecked():
            out.append("xfer")
        if self.chk_warm.isChecked():
            out.append("warm")
        return out or None

    def _get_modes(self) -> Optional[List[str]]:
        out: List[str] = []
        if self.chk_as_is.isChecked():
            out.append("as_is")
        if self.chk_strict.isChecked():
            out.append("strict")
        return out or None

    def _update_warm_enabled(self) -> None:
        on = bool(self.chk_warm.isChecked())
        self.warm_box.setEnabled(on)

    def _push(self) -> None:
        s = self._s
        s.set("xfer.strategies", self._get_strats())
        s.set("xfer.rescale_modes", self._get_modes())

        s.set("xfer.warm_split", str(self.cmb_split.currentData()))
        s.set("xfer.warm_samples", int(self.sp_samples.value()))
        s.set("xfer.warm_frac", float(self.sp_frac.value()))
        s.set("xfer.warm_epochs", int(self.sp_epochs.value()))
        s.set("xfer.warm_lr", float(self.sp_lr.value()))
        s.set("xfer.warm_seed", int(self.sp_seed.value()))


class XferRunCenter(QWidget):
    """
    RUN center column: scrollable cards + store bindings.
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._make_card = make_card

        self._build_ui()
        self._connect()
        self._sync_from_store(set())
        self._refresh_summaries()

        self._s.config_changed.connect(self._on_store)

    # -------------------------------------------------
    # Navigator jump API
    # -------------------------------------------------
    def goto_card(self, key: str) -> None:
        if hasattr(self, "registry"):
            self.registry.goto(key)
            
    # -------------------------
    # Navigation API
    # -------------------------
    def goto(self, key: str) -> None:
        if hasattr(self, "registry"):
            self.registry.goto(str(key))

    # compat name used in older code
    def scroll_to(self, key: str) -> None:
        self.goto(key)
    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
    
        self.center_scroll = QScrollArea(self)
        self.center_scroll.setWidgetResizable(True)
        self.center_scroll.setFrameShape(QFrame.NoFrame)
        self.center_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
    
        self._content = QWidget(self.center_scroll)
        self._content.setObjectName("xferRunCenterContent")
    
        lay = QVBoxLayout(self._content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
    
        # =================================================
        # Card 1: Cities & splits (expanded by default)
        # =================================================
        c1, b1 = self._make_card("Cities & splits")
        self.card_cities = c1
    
        sum_row = QWidget(c1)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)
    
        self.lbl_sum_cities = QLabel("cities, splits, batch", sum_row)
        self.lbl_sum_cities.setObjectName("sumLine")
        self.lbl_sum_cities.setWordWrap(True)
    
        self._cities_body = QWidget(c1)
        btn = self._mk_disclosure(
            summary=self.lbl_sum_cities,
            body=self._cities_body,
            expanded=True,
        )
    
        sum_l.addWidget(self.lbl_sum_cities, 1)
        sum_l.addWidget(btn, 0)
        b1.addWidget(sum_row, 0)
    
        cities_l = QVBoxLayout(self._cities_body)
        cities_l.setContentsMargins(0, 4, 0, 0)
        cities_l.setSpacing(8)
        self._build_cities_box(cities_l)
    
        b1.addWidget(self._cities_body, 0)
        lay.addWidget(c1, 0)
    
        # =================================================
        # Card 2: Outputs & alignments
        # =================================================
        c2, b2 = self._make_card("Outputs & alignments")
        self.card_outputs = c2
    
        sum_row = QWidget(c2)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)
    
        self.lbl_sum_outputs = QLabel("q, cov, align, formats", sum_row)
        self.lbl_sum_outputs.setObjectName("sumLine")
        self.lbl_sum_outputs.setWordWrap(True)
    
        self._out_body = QWidget(c2)
        btn = self._mk_disclosure(
            summary=self.lbl_sum_outputs,
            body=self._out_body,
            expanded=False,
        )
    
        sum_l.addWidget(self.lbl_sum_outputs, 1)
        sum_l.addWidget(btn, 0)
        b2.addWidget(sum_row, 0)
    
        out_l = QVBoxLayout(self._out_body)
        out_l.setContentsMargins(0, 4, 0, 0)
        out_l.setSpacing(8)
    
        self.sec_outputs = OutputsAlignmentsSection(store=self._s)
        out_l.addWidget(self.sec_outputs, 0)
    
        b2.addWidget(self._out_body, 0)
        lay.addWidget(c2, 0)
    
        # =================================================
        # Card 3: Strategy & warm-start
        # =================================================
        c3, b3 = self._make_card("Strategy & warm-start")
        self.card_strategy = c3
    
        sum_row = QWidget(c3)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)
    
        self.lbl_sum_strategy = QLabel("base, xfer, warm", sum_row)
        self.lbl_sum_strategy.setObjectName("sumLine")
        self.lbl_sum_strategy.setWordWrap(True)
    
        self._str_body = QWidget(c3)
        btn = self._mk_disclosure(
            summary=self.lbl_sum_strategy,
            body=self._str_body,
            expanded=False,
        )
    
        sum_l.addWidget(self.lbl_sum_strategy, 1)
        sum_l.addWidget(btn, 0)
        b3.addWidget(sum_row, 0)
    
        str_l = QVBoxLayout(self._str_body)
        str_l.setContentsMargins(0, 4, 0, 0)
        str_l.setSpacing(8)
    
        self.sec_strategy = StrategyWarmStartSection(store=self._s)
        str_l.addWidget(self.sec_strategy, 0)
    
        b3.addWidget(self._str_body, 0)
        lay.addWidget(c3, 0)
    
        # =================================================
        # Card 4: Results & view
        # =================================================
        c4, b4 = self._make_card("Results & view")
        self.card_results = c4
    
        sum_row = QWidget(c4)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)
    
        self.lbl_sum_results = QLabel("root, view, last run", sum_row)
        self.lbl_sum_results.setObjectName("sumLine")
        self.lbl_sum_results.setWordWrap(True)
    
        self._res_body = QWidget(c4)
        btn = self._mk_disclosure(
            summary=self.lbl_sum_results,
            body=self._res_body,
            expanded=False,
        )
    
        sum_l.addWidget(self.lbl_sum_results, 1)
        sum_l.addWidget(btn, 0)
        b4.addWidget(sum_row, 0)
    
        res_l = QVBoxLayout(self._res_body)
        res_l.setContentsMargins(0, 4, 0, 0)
        res_l.setSpacing(8)
        self._build_results_box(res_l)
    
        b4.addWidget(self._res_body, 0)
        lay.addWidget(c4, 0)
    
        lay.addStretch(1)
    
        self.center_scroll.setWidget(self._content)
        root.addWidget(self.center_scroll, 1)
    
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
    
        self.registry = CardRegistry(scroll=self.center_scroll)
        self.registry.register("cities", self.card_cities)
        self.registry.register("outputs", self.card_outputs)
        self.registry.register("strategy", self.card_strategy)
        self.registry.register("results", self.card_results)


    def _mk_disclosure(
        self,
        *,
        summary: QLabel,
        body: QWidget,
        expanded: bool = False,
    ) -> QToolButton:
        btn = QToolButton(self)
        btn.setObjectName("disclosure")
        btn.setCheckable(True)
        btn.setChecked(bool(expanded))
        btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        btn.setText("Edit")
        btn.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        body.setObjectName("drawer")
        body.setVisible(bool(expanded))
    
        def _toggle(on: bool) -> None:
            body.setVisible(bool(on))
            btn.setArrowType(
                Qt.DownArrow if on else Qt.RightArrow
            )
            p = body.parentWidget()
            if p is not None:
                p.updateGeometry()
    
        btn.toggled.connect(_toggle)
    
        summary.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )
        return btn


    def _refresh_summaries(self) -> None:
        s = self._s
    
        try:
            a = self.ed_city_a.text().strip()
            b = self.ed_city_b.text().strip()
    
            sp = ",".join(self._get_splits()) or "—"
            cal = ",".join(self._get_calib()) or "—"
            bsz = int(self.sp_batch.value())
            rs = "rescale✓" if self.chk_rescale.isChecked() else "rescale—"
    
            a_s = "A=✓" if a else "A=—"
            b_s = "B=✓" if b else "B=—"
    
            self.lbl_sum_cities.setText(
                f"{a_s}  •  {b_s}  •  split={sp}  •  "
                f"cal={cal}  •  batch={bsz}  •  {rs}"
            )
        except Exception:
            pass
    
        try:
            q = s.get("xfer.quantiles_override", None)
            q_s = "q=AUTO" if not q else "q=✓"
    
            cov = float(s.get("xfer.interval_target", 0.80))
            cov_s = f"cov={cov:.2f}"
    
            ap = str(s.get(
                "xfer.align_policy",
                "align_by_name_pad",
            ))
            ap_s = "align=pad" if "pad" in ap else "align=strict"
    
            js = bool(s.get("xfer.write_json", True))
            cs = bool(s.get("xfer.write_csv", True))
            io_s = ("json" if js else "—") + "/" + ("csv" if cs else "—")
    
            self.lbl_sum_outputs.setText(
                f"{q_s}  •  {cov_s}  •  {ap_s}  •  {io_s}"
            )
        except Exception:
            pass
    
        try:
            st = set(s.get("xfer.strategies", []) or [])
            base = "base✓" if "baseline" in st else "base—"
            xfer = "xfer✓" if "xfer" in st else "xfer—"
            warm = "warm✓" if "warm" in st else "warm—"
    
            self.lbl_sum_strategy.setText(
                f"{base}  •  {xfer}  •  {warm}"
            )
        except Exception:
            pass
    
        try:
            root = str(s.get("results_root", "") or "").strip()
            root_s = "root=✓" if root else "root=—"
    
            vk = str(s.get("xfer.view_kind", "calib_panel") or "")
            vk_s = "view=calib" if "calib" in vk else "view=summary"
    
            vs = str(s.get("xfer.view_split", "val") or "val")
            vs_s = f"split={vs}"
    
            last = str(self.lbl_last_out.text() or "").strip()
            has = bool(last) and ("No transfer run yet" not in last)
            run_s = "run=✓" if has else "run=—"
    
            self.lbl_sum_results.setText(
                f"{root_s}  •  {vk_s}  •  {vs_s}  •  {run_s}"
            )
        except Exception:
            pass

    # -------------------------------------------------
    # Public
    # -------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "city_a": self.ed_city_a.text().strip(),
            "city_b": self.ed_city_b.text().strip(),
            "results_root": (
                self.ed_results_root.text().strip() or None
            ),
            "splits": self._get_splits(),
            "calib_modes": self._get_calib(),
            "batch_size": int(self.sp_batch.value()),
            "rescale_to_source": bool(
                self.chk_rescale.isChecked()
            ),
            "view_kind": str(
                self.cmb_view_kind.currentData() or "calib_panel"
            ),
            "view_split": str(
                self.cmb_view_split.currentData() or "val"
            ),
        }

    def set_last_output(self, out_dir: Optional[str]) -> None:
        txt = out_dir or "No transfer run yet."
        self.lbl_last_out.setText(txt)
        self.btn_make_view.setVisible(bool(out_dir))

    def set_has_result(self, has: bool) -> None:
        self.btn_make_view.setVisible(bool(has))
        self.btn_make_view.setEnabled(bool(has))

    def _build_cities_box(self, box: QVBoxLayout) -> None:
        self.ed_city_a = QLineEdit(self)
        self.ed_city_a.setPlaceholderText("nansha")

        self.ed_city_b = QLineEdit(self)
        self.ed_city_b.setPlaceholderText("zhongshan")

        self.chk_split_train = QCheckBox("train", self)
        self.chk_split_val = QCheckBox("val", self)
        self.chk_split_test = QCheckBox("test", self)

        self.chk_cal_none = QCheckBox("none", self)
        self.chk_cal_source = QCheckBox("source", self)
        self.chk_cal_target = QCheckBox("target", self)

        self.sp_batch = QSpinBox(self)
        self.sp_batch.setRange(1, 2048)

        self.chk_rescale = QCheckBox(
            "Rescale target city to source domain",
            self,
        )

        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(10)

        r = 0
        g.addWidget(QLabel("City A (source):"), r, 0)
        g.addWidget(self.ed_city_a, r, 1)
        r += 1

        g.addWidget(QLabel("City B (target):"), r, 0)
        g.addWidget(self.ed_city_b, r, 1)
        r += 1

        g.addWidget(QLabel("Splits:"), r, 0)
        srow = QHBoxLayout()
        srow.addWidget(self.chk_split_train)
        srow.addWidget(self.chk_split_val)
        srow.addWidget(self.chk_split_test)
        srow.addStretch(1)
        g.addLayout(srow, r, 1)
        r += 1

        g.addWidget(QLabel("Calibration:"), r, 0)
        crow = QHBoxLayout()
        crow.addWidget(self.chk_cal_none)
        crow.addWidget(self.chk_cal_source)
        crow.addWidget(self.chk_cal_target)
        crow.addStretch(1)
        g.addLayout(crow, r, 1)
        r += 1

        g.addWidget(QLabel("Batch size:"), r, 0)
        g.addWidget(self.sp_batch, r, 1)
        r += 1

        g.addWidget(self.chk_rescale, r, 0, 1, 2)

        box.addLayout(g)

    def _build_results_box(self, box: QVBoxLayout) -> None:
        self.ed_results_root = QLineEdit(self)
        self.btn_browse_root = QPushButton("Browse…", self)

        self.lbl_last_out = QLabel("No transfer run yet.", self)
        self.lbl_last_out.setObjectName("xferLastOutLabel")

        self.cmb_view_kind = QComboBox(self)
        self.cmb_view_kind.addItem(
            "Calibration vs error (scatter panel)",
            "calib_panel",
        )
        self.cmb_view_kind.addItem(
            "Per-horizon MAE + cov/sharp (summary)",
            "summary_panel",
        )

        self.cmb_view_split = QComboBox(self)
        self.cmb_view_split.addItem("Validation (val)", "val")
        self.cmb_view_split.addItem("Test (test)", "test")

        self.btn_make_view = QPushButton(
            "Make view figure…",
            self,
        )
        self.btn_make_view.setVisible(False)

        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(10)

        r = 0
        g.addWidget(QLabel("Results root:"), r, 0)
        g.addWidget(self.ed_results_root, r, 1)
        g.addWidget(self.btn_browse_root, r, 2)
        r += 1

        g.addWidget(QLabel("Last output folder:"), r, 0)
        g.addWidget(self.lbl_last_out, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("View type:"), r, 0)
        g.addWidget(self.cmb_view_kind, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("View split:"), r, 0)
        g.addWidget(self.cmb_view_split, r, 1, 1, 2)
        r += 1

        g.addWidget(self.btn_make_view, r, 0, 1, 3)

        box.addLayout(g)

    # -------------------------------------------------
    # Wiring
    # -------------------------------------------------
    def _connect(self) -> None:
        for w in (
            self.ed_city_a,
            self.ed_city_b,
            self.ed_results_root,
        ):
            w.editingFinished.connect(self._push)

        for cb in (
            self.chk_split_train,
            self.chk_split_val,
            self.chk_split_test,
            self.chk_cal_none,
            self.chk_cal_source,
            self.chk_cal_target,
            self.chk_rescale,
        ):
            cb.toggled.connect(self._push)

        self.sp_batch.valueChanged.connect(self._push)
        self.cmb_view_kind.currentIndexChanged.connect(self._push)
        self.cmb_view_split.currentIndexChanged.connect(self._push)

        self.btn_browse_root.clicked.connect(self._browse_root)

    def _browse_root(self) -> None:
        # keep stub here; you can wire QFileDialog
        # from your host/controller (preferred)
        pass

    # -------------------------------------------------
    # Store
    # -------------------------------------------------
    def _on_store(self, keys: object) -> None:
        try:
            ch = set(keys or [])
        except Exception:
            ch = set()
        if not ch:
            return

        want = {
            "results_root",
            "xfer.city_a",
            "xfer.city_b",
            "xfer.splits",
            "xfer.calib_modes",
            "xfer.batch_size",
            "xfer.rescale_to_source",
            "xfer.view_kind",
            "xfer.view_split",
        }
        if ch & want:
            self._sync_from_store(ch)

        self._refresh_summaries()


    def _sync_from_store(self, keys: set[str]) -> None:
        s = self._s

        if "results_root" in keys or not keys:
            v = s.get("results_root", "")
            with _blocked(self.ed_results_root):
                self.ed_results_root.setText(str(v or ""))

        if "xfer.city_a" in keys or not keys:
            v = s.get("xfer.city_a", "")
            with _blocked(self.ed_city_a):
                self.ed_city_a.setText(str(v or ""))

        if "xfer.city_b" in keys or not keys:
            v = s.get("xfer.city_b", "")
            with _blocked(self.ed_city_b):
                self.ed_city_b.setText(str(v or ""))

        if "xfer.batch_size" in keys or not keys:
            v = s.get("xfer.batch_size", None)
            if v is not None:
                with _blocked(self.sp_batch):
                    self.sp_batch.setValue(int(v))

        if "xfer.rescale_to_source" in keys or not keys:
            v = bool(s.get("xfer.rescale_to_source", False))
            with _blocked(self.chk_rescale):
                self.chk_rescale.setChecked(v)

        if "xfer.splits" in keys or not keys:
            self._sync_splits(s.get(
                "xfer.splits",
                ("val", "test"),
            ))

        if "xfer.calib_modes" in keys or not keys:
            self._sync_calib(s.get(
                "xfer.calib_modes",
                ("none", "source", "target"),
            ))

        if "xfer.view_kind" in keys or not keys:
            self._set_combo(self.cmb_view_kind, s.get(
                "xfer.view_kind",
                "calib_panel",
            ))

        if "xfer.view_split" in keys or not keys:
            self._set_combo(self.cmb_view_split, s.get(
                "xfer.view_split",
                "val",
            ))

    def _push(self) -> None:
        s = self._s
        s.set("results_root", self.ed_results_root.text().strip())
        s.set("xfer.city_a", self.ed_city_a.text().strip())
        s.set("xfer.city_b", self.ed_city_b.text().strip())
        s.set("xfer.splits", self._get_splits())
        s.set("xfer.calib_modes", self._get_calib())
        s.set("xfer.batch_size", int(self.sp_batch.value()))
        s.set(
            "xfer.rescale_to_source",
            bool(self.chk_rescale.isChecked()),
        )
        s.set(
            "xfer.view_kind",
            str(self.cmb_view_kind.currentData()),
        )
        s.set(
            "xfer.view_split",
            str(self.cmb_view_split.currentData()),
        )

        self._refresh_summaries()

    # -------------------------------------------------
    # Small helpers
    # -------------------------------------------------
    def _get_splits(self) -> List[str]:
        out: List[str] = []
        if self.chk_split_train.isChecked():
            out.append("train")
        if self.chk_split_val.isChecked():
            out.append("val")
        if self.chk_split_test.isChecked():
            out.append("test")
        return out

    def _sync_splits(self, splits: Any) -> None:
        ss = set(splits or [])
        with _blocked(self.chk_split_train):
            self.chk_split_train.setChecked("train" in ss)
        with _blocked(self.chk_split_val):
            self.chk_split_val.setChecked("val" in ss)
        with _blocked(self.chk_split_test):
            self.chk_split_test.setChecked("test" in ss)

    def _get_calib(self) -> List[str]:
        out: List[str] = []
        if self.chk_cal_none.isChecked():
            out.append("none")
        if self.chk_cal_source.isChecked():
            out.append("source")
        if self.chk_cal_target.isChecked():
            out.append("target")
        return out

    def _sync_calib(self, modes: Any) -> None:
        mm = set(modes or [])
        with _blocked(self.chk_cal_none):
            self.chk_cal_none.setChecked("none" in mm)
        with _blocked(self.chk_cal_source):
            self.chk_cal_source.setChecked("source" in mm)
        with _blocked(self.chk_cal_target):
            self.chk_cal_target.setChecked("target" in mm)

    def _set_combo(self, cmb: QComboBox, v: Any) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == v:
                with _blocked(cmb):
                    cmb.setCurrentIndex(i)
                return
