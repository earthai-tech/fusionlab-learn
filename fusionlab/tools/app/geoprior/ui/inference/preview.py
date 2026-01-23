# geoprior/ui/inference/preview.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


__all__ = ["InferencePreviewPanel"]


class InferencePreviewPanel(QWidget):
    """
    Right preview [D]: readiness + resolved plan.

    This panel should own:
    - compute warnings
    - compute plan text
    - emit readiness_changed for nav chips
    """

    readiness_changed = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._title = QLabel("<b>Run preview</b>", self)
        self._hint = QLabel("", self)
        self._hint.setWordWrap(True)

        self._tree = QTreeWidget(self)
        self._tree.setHeaderHidden(True)

        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        card = QFrame(self)
        card.setObjectName("infPreview")
        card.setFrameShape(QFrame.StyledPanel)

        v = QVBoxLayout(card)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        self._hint.setText(
            "Resolved plan + readiness will appear here."
        )

        v.addWidget(self._title)
        v.addWidget(self._hint)
        v.addWidget(self._tree, stretch=1)

        root.addWidget(card, stretch=1)

    # ----------------------------------------------------------
    # Public
    # ----------------------------------------------------------
    def refresh(self) -> None:
        """
        Recompute preview.

        TODO: controller can inject a callable or
        we can pull data from store + center panel
        via signals later.
        """
        self._tree.clear()

        # Placeholder tree content
        sec = self._section("Readiness")
        self._kv(sec, "Status", "TODO")
        self._kv(sec, "Warnings", "0")

        self._section("Inputs")
        self._section("Options")
        self._section("Outputs")

        # Placeholder chips
        chips: Dict[str, str] = {
            "artifacts": "warn",
            "uncertainty": "ok",
            "calibration": "ok",
            "outputs": "ok",
            "advanced": "ok",
        }
        self.readiness_changed.emit(chips)

    # ----------------------------------------------------------
    # Tree helpers
    # ----------------------------------------------------------
    def _section(self, title: str) -> QTreeWidgetItem:
        it = QTreeWidgetItem([title])
        self._tree.addTopLevelItem(it)
        it.setExpanded(True)
        return it

    def _kv(
        self,
        parent: QTreeWidgetItem,
        k: str,
        v: str,
    ) -> None:
        child = QTreeWidgetItem([f"{k}: {v}"])
        parent.addChild(child)
