# geoprior/ui/inference/head.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

from ...config.store import GeoConfigStore
from ..icon_utils import try_icon


__all__ = ["InferenceHeadBar"]


class InferenceHeadBar(QWidget):
    """
    Head row [B]: global actions + summary.

    Keep it compact. No cross into left column.
    """

    refresh_clicked = pyqtSignal()
    copy_plan_clicked = pyqtSignal()
    open_outputs_clicked = pyqtSignal()
    help_clicked = pyqtSignal()

    def __init__(
        self,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store

        self._summary = QLabel("", self)

        self._btn_refresh = QToolButton(self)
        self._btn_copy = QToolButton(self)
        self._btn_open = QToolButton(self)
        self._btn_help = QToolButton(self)

        self._init_ui()
        self._wire()

    def _init_ui(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 0, 6, 0)
        row.setSpacing(6)

        self._summary.setText(
            "Inference: select artifacts, review plan, run."
        )
        self._summary.setWordWrap(False)

        self._btn_refresh.setIcon(try_icon("refresh"))
        self._btn_refresh.setToolTip("Refresh preview")

        self._btn_copy.setIcon(try_icon("copy"))
        self._btn_copy.setToolTip("Copy run plan")

        self._btn_open.setIcon(try_icon("folder"))
        self._btn_open.setToolTip("Open last outputs")

        self._btn_help.setIcon(try_icon("help"))
        self._btn_help.setToolTip("Help")

        row.addWidget(self._summary, stretch=1)
        row.addWidget(self._btn_refresh)
        row.addWidget(self._btn_copy)
        row.addWidget(self._btn_open)
        row.addWidget(self._btn_help)

    def _wire(self) -> None:
        self._btn_refresh.clicked.connect(self.refresh_clicked)
        self._btn_copy.clicked.connect(self.copy_plan_clicked)
        self._btn_open.clicked.connect(self.open_outputs_clicked)
        self._btn_help.clicked.connect(self.help_clicked)

    # ----------------------------------------------------------
    # Public
    # ----------------------------------------------------------
    def set_summary(self, text: str) -> None:
        self._summary.setText(text)
