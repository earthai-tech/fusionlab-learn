# geoprior/ui/inference/details.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..icon_utils import try_icon


__all__ = ["InferenceDetailsPanel"]


class InferenceDetailsPanel(QWidget):
    """
    Left extras [E] under navigator.

    Suggested:
    - last outputs shortcuts
    - tips/help
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._lbl = QLabel("<b>Details</b>", self)
        self._out = QLabel("Last outputs: —", self)
        self._out.setWordWrap(True)

        self._btn_open = QToolButton(self)
        self._btn_open.setIcon(try_icon("folder"))
        self._btn_open.setToolTip("Open last outputs")

        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        card = QFrame(self)
        card.setFrameShape(QFrame.StyledPanel)
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        v.addWidget(self._lbl)
        v.addWidget(self._out)
        v.addWidget(self._btn_open)

        v.addStretch(1)
        root.addWidget(card, stretch=1)

    def set_last_outputs(self, path: str) -> None:
        p = path.strip() if path else "—"
        self._out.setText(f"Last outputs: {p}")
