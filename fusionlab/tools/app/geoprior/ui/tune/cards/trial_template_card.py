# geoprior/ui/tune/cards/trial_template_card.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Trial template card (Tune tab).

Train-like behavior:
- Uses make_card() for consistent styling.
- Header row: summary + Edit (disclosure)
- Body: expands inside the SAME card.

Controls (store-backed)
-----------------------
- epochs           (FieldKey("epochs"))
- batch_size       (FieldKey("batch_size"))
- learning_rate    (FieldKey("learning_rate"))
- tuner_max_trials (FieldKey("tuner_max_trials"))
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....config.prior_schema import FieldKey
from ....config.store import GeoConfigStore
from ...icon_utils import try_icon

__all__ = ["TrialTemplateCard"]

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]

_EPOCHS_FK = FieldKey("epochs")
_BATCH_FK = FieldKey("batch_size")
_LR_FK = FieldKey("learning_rate")
_TRIALS_FK = FieldKey("tuner_max_trials")


class TrialTemplateCard(QWidget):
    """
    Trial template card (expand/collapse in-place).

    Signals
    -------
    changed:
        Emitted after applying edits to the store.

    edit_toggled(bool):
        Emitted when user expands/collapses the card.
    """

    changed = pyqtSignal()
    edit_toggled = pyqtSignal(bool)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: MakeCardFn,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._make_card = make_card
        self._writing = False
        self._expanded = False

        self._build_ui()
        self._wire()
        self.refresh_from_store()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._frame, body = self._make_card("Trial template")
        root.addWidget(self._frame)

        # Summary + Edit (same row)
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        self.lbl_sum = QLabel("—", self._frame)
        self.lbl_sum.setObjectName("sumLine")
        self.lbl_sum.setWordWrap(True)

        self.btn_edit = QToolButton(self._frame)
        self.btn_edit.setObjectName("disclosure")
        self.btn_edit.setCursor(Qt.PointingHandCursor)
        self.btn_edit.setAutoRaise(True)
        self.btn_edit.setCheckable(True)
        self.btn_edit.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn_edit.setText("Edit")
        self._set_edit_icon(expanded=False)

        hdr.addWidget(self.lbl_sum, 1)
        hdr.addWidget(self.btn_edit, 0)
        body.addLayout(hdr)

        self.lbl_hint = QLabel(
            "Controls applied to each trial run "
            "(epochs, batch, LR, etc.).",
            self._frame,
        )
        self.lbl_hint.setObjectName("helpText")
        self.lbl_hint.setWordWrap(True)
        body.addWidget(self.lbl_hint)

        # Drawer
        self.details = QWidget(self._frame)
        self.details.setObjectName("drawer")
        self.details.setVisible(False)

        dlay = QVBoxLayout(self.details)
        dlay.setContentsMargins(0, 6, 0, 0)
        dlay.setSpacing(10)

        # Top actions
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        self.btn_apply = QPushButton("Apply", self.details)
        self.btn_apply.setObjectName("miniAction")
        self.btn_apply.setEnabled(False)

        self.btn_full = QPushButton(
            "Open full editor…", self.details
        )
        self.btn_full.setObjectName("miniAction")
        self.btn_full.setEnabled(False)  # future hook

        top.addStretch(1)
        top.addWidget(self.btn_apply, 0)
        top.addWidget(self.btn_full, 0)
        dlay.addLayout(top)

        # Form grid
        g = QGridLayout()
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)
        g.setColumnStretch(1, 1)

        self.sp_epochs = QSpinBox(self.details)
        self.sp_epochs.setRange(1, 100_000)
        self.sp_epochs.setMinimumHeight(26)

        self.sp_batch = QSpinBox(self.details)
        self.sp_batch.setRange(1, 8192)
        self.sp_batch.setMinimumHeight(26)

        self.sp_lr = QDoubleSpinBox(self.details)
        self.sp_lr.setDecimals(10)
        self.sp_lr.setRange(0.0, 10.0)
        self.sp_lr.setSingleStep(1e-4)
        self.sp_lr.setMinimumHeight(26)

        self.sp_trials = QSpinBox(self.details)
        self.sp_trials.setRange(1, 10_000)
        self.sp_trials.setMinimumHeight(26)

        r = 0
        g.addWidget(self._lbl("Epochs:"), r, 0)
        g.addWidget(self.sp_epochs, r, 1)
        r += 1

        g.addWidget(self._lbl("Batch size:"), r, 0)
        g.addWidget(self.sp_batch, r, 1)
        r += 1

        g.addWidget(self._lbl("Learning rate:"), r, 0)
        g.addWidget(self.sp_lr, r, 1)
        r += 1

        g.addWidget(self._lbl("Max trials:"), r, 0)
        g.addWidget(self.sp_trials, r, 1)

        dlay.addLayout(g)
        body.addWidget(self.details)

    def _lbl(self, text: str) -> QLabel:
        lab = QLabel(text, self.details)
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return lab

    # -----------------------------------------------------------------
    # Edit toggle helpers (shared convention)
    # -----------------------------------------------------------------
    def _set_edit_icon(self, *, expanded: bool) -> None:
        name = "chev_down.svg" if expanded else "chev_right.svg"
        ic = try_icon(name)
        if ic is not None:
            self.btn_edit.setIcon(ic)
        self.btn_edit.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )

    def _on_toggle(self, on: bool) -> None:
        self._expanded = bool(on)
        self.details.setVisible(self._expanded)
        self._set_edit_icon(expanded=self._expanded)
        self.edit_toggled.emit(bool(on))

        if self._expanded:
            self.btn_apply.setEnabled(False)

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.btn_edit.toggled.connect(self._on_toggle)
        self.btn_apply.clicked.connect(self._apply)

        self.sp_epochs.valueChanged.connect(lambda *_: self._on_dirty())
        self.sp_batch.valueChanged.connect(lambda *_: self._on_dirty())
        self.sp_lr.valueChanged.connect(lambda *_: self._on_dirty())
        self.sp_trials.valueChanged.connect(lambda *_: self._on_dirty())

    def _on_dirty(self) -> None:
        if self._writing:
            return
        if not self._expanded:
            return
        self.btn_apply.setEnabled(True)

    # -----------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------
    def refresh_from_store(self) -> None:
        self._writing = True
        try:
            ep = self._get_int(_EPOCHS_FK, 50)
            bs = self._get_int(_BATCH_FK, 32)
            lr = self._get_float(_LR_FK, 1e-4)
            tr = self._get_int(_TRIALS_FK, 20)

            with QSignalBlocker(self.sp_epochs):
                self.sp_epochs.setValue(int(ep))
            with QSignalBlocker(self.sp_batch):
                self.sp_batch.setValue(int(bs))
            with QSignalBlocker(self.sp_lr):
                self.sp_lr.setValue(float(lr))
            with QSignalBlocker(self.sp_trials):
                self.sp_trials.setValue(int(tr))

            self._update_summary()
            self.btn_apply.setEnabled(False)
        finally:
            self._writing = False

    # -----------------------------------------------------------------
    # Apply -> store
    # -----------------------------------------------------------------
    def _apply(self) -> None:
        if self._writing:
            return

        ep = int(self.sp_epochs.value())
        bs = int(self.sp_batch.value())
        lr = float(self.sp_lr.value())
        tr = int(self.sp_trials.value())

        self._writing = True
        try:
            with self._store.batch():
                self._set_fk(_EPOCHS_FK, ep)
                self._set_fk(_BATCH_FK, bs)
                self._set_fk(_LR_FK, lr)
                self._set_fk(_TRIALS_FK, tr)
        finally:
            self._writing = False

        self._update_summary()
        self.btn_apply.setEnabled(False)
        self.changed.emit()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    def _update_summary(self) -> None:
        ep = self._get_int(_EPOCHS_FK, 50)
        bs = self._get_int(_BATCH_FK, 32)
        lr = self._get_float(_LR_FK, 1e-4)
        tr = self._get_int(_TRIALS_FK, 20)

        self.lbl_sum.setText(
            f"epochs={int(ep)} batch={int(bs)} "
            f"lr={self._fmt_lr(lr)} trials={int(tr)}"
        )

    @staticmethod
    def _fmt_lr(v: float) -> str:
        try:
            if v == 0.0:
                return "0"
            if v < 1e-3 or v >= 1.0:
                return f"{v:.1e}"
            return f"{v:.6f}".rstrip("0").rstrip(".")
        except Exception:
            return str(v)

    # -----------------------------------------------------------------
    # Store helpers
    # -----------------------------------------------------------------
    def _get_int(self, fk: FieldKey, default: int) -> int:
        try:
            v = self._store.get_value(fk, default=default)
            return int(v)
        except Exception:
            return int(default)

    def _get_float(self, fk: FieldKey, default: float) -> float:
        try:
            v = self._store.get_value(fk, default=default)
            return float(v)
        except Exception:
            return float(default)

    def _set_fk(self, fk: FieldKey, value: Any) -> None:
        try:
            self._store.set_value_by_key(fk, value)
        except Exception:
            pass
