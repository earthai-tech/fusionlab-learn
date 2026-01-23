# geoprior/ui/inference/center_panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...config.store import GeoConfigStore


__all__ = ["InferenceCenterPanel"]


class InferenceCenterPanel(QWidget):
    """
    Center cards [C] with Edit disclosures.

    - store-backed settings should bind here.
    - runtime-only paths can live in local fields
      and be pulled by controller on run.
    """

    changed = pyqtSignal()

    browse_model_clicked = pyqtSignal()
    browse_manifest_clicked = pyqtSignal()
    browse_npz_clicked = pyqtSignal()

    def __init__(
        self,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store

        self._cards: Dict[str, QWidget] = {}

        self._init_ui()
        self._wire()

    # ==========================================================
    # UI
    # ==========================================================
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Cards (placeholders)
        self._cards["artifacts"] = self._mk_artifacts_card()
        self._cards["uncertainty"] = self._mk_simple_card(
            "Uncertainty",
            "interval, mode, temperature",
        )
        self._cards["calibration"] = self._mk_simple_card(
            "Calibration",
            "source calibrator, fit-on-val, file",
        )
        self._cards["outputs"] = self._mk_simple_card(
            "Outputs",
            "include GWL, plots, output dir",
        )
        self._cards["advanced"] = self._mk_simple_card(
            "Advanced",
            "batch size, misc toggles",
        )

        for k in ["artifacts", "uncertainty",
                  "calibration", "outputs", "advanced"]:
            root.addWidget(self._cards[k])

        root.addStretch(1)

    def _wire(self) -> None:
        # Card widgets will connect to changed.
        pass

    # ==========================================================
    # Card builders
    # ==========================================================
    def _mk_card_shell(
        self,
        title: str,
        summary: str,
    ) -> QWidget:
        box = QFrame(self)
        box.setObjectName("infCard")
        box.setFrameShape(QFrame.StyledPanel)

        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)

        lbl = QLabel(f"<b>{title}</b>", self)
        sm = QLabel(summary, self)
        sm.setObjectName("infCardSummary")

        btn = QToolButton(self)
        btn.setText("Edit")
        btn.setObjectName("infDisclosure")

        top.addWidget(lbl, stretch=1)
        top.addWidget(btn, stretch=0)

        v.addLayout(top)
        v.addWidget(sm)

        # Content placeholder inserted by caller.
        return box

    def _mk_simple_card(
        self,
        title: str,
        summary: str,
    ) -> QWidget:
        box = self._mk_card_shell(title, summary)
        # Insert a placeholder row to prevent empty look
        ph = QLabel("TODO: content", self)
        ph.setObjectName("infTodo")
        box.layout().addWidget(ph)
        return box

    def _mk_artifacts_card(self) -> QWidget:
        box = self._mk_card_shell(
            "Artifacts",
            "model, manifest, dataset/custom inputs",
        )

        # Minimal fields now; we will fill progressively.
        row = QWidget(self)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        self.ed_model = QLineEdit(self)
        self.ed_model.setPlaceholderText("Select .keras model...")

        btn_model = QPushButton("Browse...", self)
        btn_model.clicked.connect(self.browse_model_clicked)

        h.addWidget(self.ed_model, stretch=1)
        h.addWidget(btn_model, stretch=0)

        box.layout().addWidget(row)
        return box

    # ==========================================================
    # Navigator focus hook
    # ==========================================================
    def focus_section(self, key: str) -> None:
        w = self._cards.get(key)
        if not w:
            return
        w.setFocus()
        # Optional: scroll-to logic if wrapped later.

    # ==========================================================
    # Public getters used by controller (runtime state)
    # ==========================================================
    def model_path(self) -> str:
        return self.ed_model.text().strip()
