# geoprior/ui/inference/navigator.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


__all__ = ["InferenceNavigator", "NavSection"]


@dataclass(frozen=True)
class NavSection:
    key: str
    title: str


_SECTIONS = [
    NavSection("artifacts", "Artifacts"),
    NavSection("uncertainty", "Uncertainty"),
    NavSection("calibration", "Calibration"),
    NavSection("outputs", "Outputs"),
    NavSection("advanced", "Advanced"),
]


class InferenceNavigator(QWidget):
    """
    Left navigator [A] with checklist chips.

    Emits section_requested(key) for center focus.
    """

    section_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._chips: Dict[str, QLabel] = {}
        self._rows: Dict[str, QPushButton] = {}

        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        title = QLabel("<b>Setup checklist</b>", self)

        sc = QScrollArea(self)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)

        inner = QWidget(self)
        v = QVBoxLayout(inner)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        for sec in _SECTIONS:
            row = QWidget(self)
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)

            btn = QPushButton(sec.title, self)
            btn.setObjectName("infNavRow")
            btn.clicked.connect(
                lambda _=False, k=sec.key: self.section_requested.emit(k)
            )

            chip = QLabel("—", self)
            chip.setObjectName("infNavChip")
            chip.setMinimumWidth(42)
            chip.setAlignment(0x84)  # Qt.AlignCenter

            h.addWidget(btn, stretch=1)
            h.addWidget(chip, stretch=0)

            self._rows[sec.key] = btn
            self._chips[sec.key] = chip

            v.addWidget(row)

        v.addStretch(1)
        sc.setWidget(inner)

        root.addWidget(title)
        root.addWidget(sc, stretch=1)

    # ----------------------------------------------------------
    # Public
    # ----------------------------------------------------------
    def set_readiness(self, mapping: Dict[str, str]) -> None:
        """
        Set chips by key.

        mapping values: "ok", "warn", "err", "off"
        """
        for key, val in mapping.items():
            chip = self._chips.get(key)
            if not chip:
                continue
            chip.setText(val.upper())

            # Styling left to styles.py / qss patches.
            chip.setProperty("chipState", val)
            chip.style().unpolish(chip)
            chip.style().polish(chip)
