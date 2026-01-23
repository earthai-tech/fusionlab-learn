# geoprior/ui/tune/cards/advanced_card.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Advanced settings card (Tune tab).

This is the "hub" for heavier dialogs and safety actions:
- Model params dialog
- Scalars & losses dialog
- Architecture HP dialog
- Physics HP dialog
- Tune options dialog hub
- Export preferences + export action
- Reset search space (same intent as Head reset)

UI-only: emits signals. The parent (center_panel/tab)
decides which dialogs to open and when to refresh.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....config.store import GeoConfigStore
from ...icon_utils import try_icon

__all__ = ["TuneAdvancedCard"]

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]


class TuneAdvancedCard(QWidget):
    """
    Advanced settings hub card.

    Uses shared make_card() factory for Train-like styling.

    Signals are emitted on click. Connect them in the
    center panel or in tune/tab.py.
    """

    # Dialog hubs
    model_params_clicked = pyqtSignal()
    scalars_losses_clicked = pyqtSignal()
    arch_hp_clicked = pyqtSignal()
    phys_hp_clicked = pyqtSignal()
    tune_options_clicked = pyqtSignal()
    export_clicked = pyqtSignal()

    # Safety action (must reset store space)
    reset_space_clicked = pyqtSignal()

    # Expand/collapse convention
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
        self._expanded = False

        self._build_ui()
        self._wire()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._frame, body = self._make_card("Advanced")
        root.addWidget(self._frame)

        # Summary + Edit (same row)
        sum_row = QWidget(self._frame)
        sum_l = QHBoxLayout(sum_row)
        sum_l.setContentsMargins(0, 0, 0, 0)
        sum_l.setSpacing(8)

        self.lbl_sum = QLabel(
            "dialogs: model/HP/scalars · export · reset",
            self._frame,
        )
        self.lbl_sum.setObjectName("sumLine")
        self.lbl_sum.setWordWrap(True)

        self.btn_edit = QToolButton(self._frame)
        self.btn_edit.setObjectName("disclosure")
        self.btn_edit.setCursor(Qt.PointingHandCursor)
        self.btn_edit.setAutoRaise(True)
        self.btn_edit.setCheckable(True)
        self.btn_edit.setChecked(False)
        self.btn_edit.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_edit.setText("Edit")
        self._set_edit_icon(expanded=False)

        sum_l.addWidget(self.lbl_sum, 1)
        sum_l.addWidget(self.btn_edit, 0)
        body.addWidget(sum_row)

        self.lbl_help = QLabel(
            "Open advanced editors and recover from mistakes "
            "by resetting the search space.",
            self._frame,
        )
        self.lbl_help.setObjectName("helpText")
        self.lbl_help.setWordWrap(True)
        body.addWidget(self.lbl_help)

        # Details area (collapsed by default)
        self.details = QFrame(self._frame)
        self.details.setObjectName("drawer")
        self.details.setFrameShape(QFrame.NoFrame)
        self.details.setVisible(False)

        dlay = QVBoxLayout(self.details)
        dlay.setContentsMargins(0, 6, 0, 0)
        dlay.setSpacing(10)

        # -------------------------------------------------
        # Dialog launcher grid (2x2)
        # -------------------------------------------------
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self.btn_model = self._mk_btn("Model params…", "model.svg")
        self.btn_arch = self._mk_btn("Architecture HP…", "layers.svg")
        self.btn_scalars = self._mk_btn("Scalars && losses…", "sigma.svg")
        self.btn_phys = self._mk_btn("Physics HP…", "physics.svg")

        grid.addWidget(self.btn_model, 0, 0)
        grid.addWidget(self.btn_arch, 0, 1)
        grid.addWidget(self.btn_scalars, 1, 0)
        grid.addWidget(self.btn_phys, 1, 1)

        dlay.addLayout(grid)

        # -------------------------------------------------
        # Options / export row
        # -------------------------------------------------
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self.btn_options = self._mk_btn("Advanced options…", "settings.svg")
        self.btn_export = self._mk_btn("Export…", "export.svg")

        row.addWidget(self.btn_options, 1)
        row.addWidget(self.btn_export, 0)
        dlay.addLayout(row)

        # -------------------------------------------------
        # Safety: reset space
        # -------------------------------------------------
        self.btn_reset = QPushButton("Reset space to defaults", self.details)
        self.btn_reset.setObjectName("miniAction")
        self._set_btn_icon(self.btn_reset, "reset.svg")
        self.btn_reset.setMinimumHeight(30)
        dlay.addWidget(self.btn_reset)

        body.addWidget(self.details)

    def _mk_btn(self, text: str, icon: str) -> QPushButton:
        b = QPushButton(text, self.details)
        b.setObjectName("miniAction")
        b.setMinimumHeight(30)
        self._set_btn_icon(b, icon)
        return b

    def _set_btn_icon(self, btn: QPushButton, name: str) -> None:
        ic = try_icon(name)
        if ic is not None:
            btn.setIcon(ic)

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.btn_edit.toggled.connect(self._on_toggle)

        self.btn_model.clicked.connect(self.model_params_clicked.emit)
        self.btn_scalars.clicked.connect(self.scalars_losses_clicked.emit)
        self.btn_arch.clicked.connect(self.arch_hp_clicked.emit)
        self.btn_phys.clicked.connect(self.phys_hp_clicked.emit)
        self.btn_options.clicked.connect(self.tune_options_clicked.emit)
        self.btn_export.clicked.connect(self.export_clicked.emit)
        self.btn_reset.clicked.connect(self.reset_space_clicked.emit)

    # -----------------------------------------------------------------
    # Public expand/collapse API
    # -----------------------------------------------------------------
    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, on: bool) -> None:
        with QSignalBlocker(self.btn_edit):
            self.btn_edit.setChecked(bool(on))
        self._on_toggle(bool(on))

    def is_expanded(self) -> bool:
        return bool(self._expanded)

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
