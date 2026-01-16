# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.prob

Probabilistic outputs (pinball quantiles, weights,
uncertainty, calibration).

Modern UX:
- Two-column layout: quick controls + preview.
- Badges: quantiles count, calibration mode, weights status.
- "Configure…" opens ProbConfigDialog.
- "Copy" copies a compact JSON snippet.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....config.store import GeoConfigStore
from ....dialogs.prob_dialog import ProbConfigDialog
from ..bindings import Binder
from .base import CardBase 

class _Expander(QWidget):
    def __init__(
        self,
        title: str,
        *,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.btn = QToolButton(self)
        self.btn.setCheckable(True)
        self.btn.setChecked(False)
        self.btn.setText(str(title))
        self.btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn.setArrowType(Qt.RightArrow)

        self.body = QWidget(self)
        self.body.setVisible(False)

        self.body_l = QGridLayout(self.body)
        self.body_l.setContentsMargins(8, 6, 8, 6)
        self.body_l.setHorizontalSpacing(10)
        self.body_l.setVerticalSpacing(6)

        self.btn.toggled.connect(self._toggle)

        root.addWidget(self.btn, 0)
        root.addWidget(self.body, 0)

    def _toggle(self, on: bool) -> None:
        self.body.setVisible(bool(on))
        self.btn.setArrowType(
            Qt.DownArrow if on else Qt.RightArrow
        )


class ProbabilisticOutputsCard(CardBase):
    """Probabilistic outputs + uncertainty controls."""

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="prob",
            title="Probabilistic outputs",
            subtitle=(
                "Quantiles, pinball weights, intervals, "
                "crossing constraints, and calibration."
            ),
            parent=parent,
        )

        self.store = store
        self.binder = binder

        self._rows: List[QWidget] = []

        self._build()
        self._wire()
        self.refresh()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build(self) -> None:
        btn_cfg = self.add_action(
            text="Configure…",
            tip="Edit quantiles, weights, and calibration",
            icon=QStyle.SP_FileDialogDetailedView,
        )
        btn_cfg.clicked.connect(self._open_dialog)

        btn_copy = self.add_action(
            text="Copy",
            tip="Copy probabilistic settings as JSON",
            icon=QStyle.SP_DialogSaveButton,
        )
        btn_copy.clicked.connect(self._copy_json)

        body = self.body_layout()

        grid = QWidget(self)
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

        self.grp_quick = self._build_quick(grid)
        self.grp_prev = self._build_preview(grid)

        g.addWidget(self.grp_quick, 0, 0)
        g.addWidget(self.grp_prev, 0, 1)

        body.addWidget(grid, 0)

    def _build_quick(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Quick controls", parent)

        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

        r = 0

        # Interval level
        self.sp_interval = QDoubleSpinBox(box)
        self.sp_interval.setDecimals(3)
        self.sp_interval.setRange(0.001, 0.999)
        self.sp_interval.setSingleStep(0.05)
        self.sp_interval.setToolTip(
            "Interval level in (0, 1), e.g. 0.80.",
        )
        self.binder.bind_double_spin_box(
            "interval_level",
            self.sp_interval,
        )

        lay.addWidget(QLabel("Interval level:", box), r, 0)
        lay.addWidget(self.sp_interval, r, 1)
        r += 1

        # Calibration mode
        self.cmb_cal = QComboBox(box)
        self.cmb_cal.addItems(
            ["none", "temperature", "isotonic", "conformal"]
        )
        self.binder.bind_combo(
            "calibration_mode",
            self.cmb_cal,
            items=[
                ("none", "none"),
                ("temperature", "temperature"),
                ("isotonic", "isotonic"),
                ("conformal", "conformal"),
            ],
            editable=False,
            use_item_data=True,
        )

        lay.addWidget(QLabel("Calibration:", box), r, 0)
        lay.addWidget(self.cmb_cal, r, 1)
        r += 1

        # Calibration temperature
        self.sp_temp = QDoubleSpinBox(box)
        self.sp_temp.setDecimals(6)
        self.sp_temp.setRange(1e-6, 1e6)
        self.sp_temp.setSingleStep(0.1)
        self.sp_temp.setToolTip(
            "Temperature scaling parameter (> 0).",
        )
        self.binder.bind_double_spin_box(
            "calibration_temperature",
            self.sp_temp,
        )

        lay.addWidget(QLabel("Temperature:", box), r, 0)
        lay.addWidget(self.sp_temp, r, 1)
        r += 1

        # Advanced: crossing controls
        self.exp = _Expander("Crossing constraint", parent=box)

        self.sp_cross_pen = QDoubleSpinBox(box)
        self.sp_cross_pen.setDecimals(6)
        self.sp_cross_pen.setRange(0.0, 1e9)
        self.sp_cross_pen.setSingleStep(0.1)
        self.sp_cross_pen.setToolTip(
            "Penalty strength for quantile crossing.",
        )
        self.binder.bind_double_spin_box(
            "crossing_penalty",
            self.sp_cross_pen,
        )

        self.sp_cross_m = QDoubleSpinBox(box)
        self.sp_cross_m.setDecimals(6)
        self.sp_cross_m.setRange(0.0, 1e9)
        self.sp_cross_m.setSingleStep(0.01)
        self.sp_cross_m.setToolTip(
            "Optional margin for crossing constraint.",
        )
        self.binder.bind_double_spin_box(
            "crossing_margin",
            self.sp_cross_m,
        )

        self.exp.body_l.addWidget(
            QLabel("Penalty:", box),
            0,
            0,
        )
        self.exp.body_l.addWidget(self.sp_cross_pen, 0, 1)

        self.exp.body_l.addWidget(
            QLabel("Margin:", box),
            1,
            0,
        )
        self.exp.body_l.addWidget(self.sp_cross_m, 1, 1)

        lay.addWidget(self.exp, r, 0, 1, 2)

        return box

    def _build_preview(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Preview", parent)

        root = QVBoxLayout(box)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.lbl_note = QLabel("", box)
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setObjectName("muted")
        root.addWidget(self.lbl_note, 0)

        self.tbl = QWidget(box)
        self.t = QGridLayout(self.tbl)
        self.t.setContentsMargins(0, 0, 0, 0)
        self.t.setHorizontalSpacing(10)
        self.t.setVerticalSpacing(6)
        self.t.setColumnStretch(0, 0)
        self.t.setColumnStretch(1, 1)

        root.addWidget(self.tbl, 0)
        root.addStretch(1)

        return box

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.store.config_changed.connect(
            self._on_cfg_changed,
        )
        self.store.config_replaced.connect(
            self._on_cfg_replaced,
        )
        self.cmb_cal.currentTextChanged.connect(
            self._sync_cal_temp_enabled,
        )

    def _on_cfg_changed(self, _keys: object) -> None:
        self.refresh()

    def _on_cfg_replaced(self, _cfg: object) -> None:
        self.refresh()

    # -----------------------------------------------------------------
    # Refresh / badges / preview
    # -----------------------------------------------------------------
    def refresh(self) -> None:
        cfg = self.store.cfg

        q = list(getattr(cfg, "quantiles", []) or [])
        cal = str(getattr(cfg, "calibration_mode", "none") or "none")

        self.badge(
            "q",
            text=f"{len(q)} quantiles",
            accent="ok" if q else "warn",
            tip="Pinball quantile count",
        )
        self.badge(
            "cal",
            text=f"Cal: {cal}",
            accent="ok" if cal == "none" else "",
            tip="Calibration mode",
        )

        ok_w = self._weights_ok(q)
        self.badge(
            "w",
            text="Weights OK" if ok_w else "Weights mismatch",
            accent="ok" if ok_w else "warn",
            tip="Weights keys match quantiles",
        )

        self._sync_cal_temp_enabled()
        self._render_preview()

    def _sync_cal_temp_enabled(self, *_a: object) -> None:
        mode = str(self.cmb_cal.currentText() or "none")
        self.sp_temp.setEnabled(mode == "temperature")

    def _render_preview(self) -> None:
        cfg = self.store.cfg

        q = list(getattr(cfg, "quantiles", []) or [])
        subs = dict(getattr(cfg, "subs_weights", {}) or {})
        gwl = dict(getattr(cfg, "gwl_weights", {}) or {})

        # Note
        cal = str(getattr(cfg, "calibration_mode", "none") or "none")
        interval = float(getattr(cfg, "interval_level", 0.8) or 0.8)

        self.lbl_note.setText(
            "Pinball weights are applied per-quantile. "
            f"Interval={interval:.3f}, calibration={cal}."
        )

        # Clear
        self._clear_grid(self.t)
        self._rows.clear()

        # Header row
        self.t.addWidget(self._h("Quantile"), 0, 0)
        self.t.addWidget(self._h("Subs / GWL weights"), 0, 1)

        if not q:
            self.t.addWidget(
                QLabel("No quantiles configured.", self.tbl),
                1,
                0,
                1,
                2,
            )
            return

        rr = 1
        for qq in q:
            w = QWidget(self.tbl)
            wl = QHBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(8)

            s = subs.get(qq, None)
            g = gwl.get(qq, None)

            wl.addWidget(
                QLabel(self._fmt(qq), w),
                0,
            )
            wl.addStretch(1)

            chip = QLabel(self._chip_text(s, g), w)
            chip.setObjectName("setupBadge")
            wl.addWidget(chip, 0)

            self.t.addWidget(w, rr, 0, 1, 2)
            self._rows.append(w)
            rr += 1

    def _h(self, text: str) -> QLabel:
        lab = QLabel(str(text), self.tbl)
        lab.setStyleSheet("font-weight: 700;")
        return lab

    def _chip_text(self, s: Any, g: Any) -> str:
        ss = "?" if s is None else self._fmt(s)
        gg = "?" if g is None else self._fmt(g)
        return f"subs={ss}  gwl={gg}"

    def _weights_ok(self, q: List[Any]) -> bool:
        cfg = self.store.cfg
        subs = dict(getattr(cfg, "subs_weights", {}) or {})
        gwl = dict(getattr(cfg, "gwl_weights", {}) or {})

        qs = {self._q_sig(x) for x in q}
        ss = {self._q_sig(x) for x in subs.keys()}
        gs = {self._q_sig(x) for x in gwl.keys()}

        return bool(qs) and qs == ss and qs == gs

    def _q_sig(self, x: Any) -> str:
        try:
            v = float(x)
        except Exception:
            return str(x)
        return f"{v:.6f}".rstrip("0").rstrip(".")

    def _fmt(self, x: Any) -> str:
        try:
            v = float(x)
        except Exception:
            return str(x)
        return f"{v:.6f}".rstrip("0").rstrip(".")

    def _clear_grid(self, lay: QGridLayout) -> None:
        while lay.count():
            it = lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------
    def _open_dialog(self) -> None:
        dlg = ProbConfigDialog(store=self.store, parent=self)
        dlg.exec_()

    def _copy_json(self) -> None:
        cfg = self.store.cfg

        payload: Dict[str, Any] = {
            "quantiles": list(getattr(cfg, "quantiles", []) or []),
            "subs_weights": dict(
                getattr(cfg, "subs_weights", {}) or {}
            ),
            "gwl_weights": dict(
                getattr(cfg, "gwl_weights", {}) or {}
            ),
            "interval_level": float(
                getattr(cfg, "interval_level", 0.8) or 0.8
            ),
            "crossing_penalty": float(
                getattr(cfg, "crossing_penalty", 0.0) or 0.0
            ),
            "crossing_margin": float(
                getattr(cfg, "crossing_margin", 0.0) or 0.0
            ),
            "calibration_mode": str(
                getattr(cfg, "calibration_mode", "none")
                or "none"
            ),
            "calibration_temperature": float(
                getattr(cfg, "calibration_temperature", 1.0)
                or 1.0
            ),
        }

        try:
            txt = json.dumps(payload, indent=2)
            QApplication.clipboard().setText(txt)
            self.store.error_raised.emit(
                "Probabilistic settings copied to clipboard."
            )
        except Exception as exc:
            self.store.error_raised.emit(str(exc))
