# geoprior/ui/tune/details.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ...config.store import GeoConfigStore
from ...device_options import runtime_summary_text

from .plan import _get_fk, space_stats


__all__ = ["TuneDetailsCard"]


class TuneDetailsCard(QFrame):
    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store

        self.setObjectName("trainNavCard")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()
        self.refresh_from_store()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        t1 = QLabel("Computer details")
        t1.setObjectName("trainNavTitle")
        root.addWidget(t1)

        self.lbl_compute = QLabel("")
        self.lbl_compute.setObjectName("runComputeText")
        self.lbl_compute.setWordWrap(True)
        self.lbl_compute.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        root.addWidget(self.lbl_compute)

        t2 = QLabel("Search-space stats")
        t2.setObjectName("trainNavTitle")
        root.addWidget(t2)

        self.lbl_space = QLabel("")
        self.lbl_space.setWordWrap(True)
        root.addWidget(self.lbl_space)

        root.addStretch(1)

    def refresh_from_store(self) -> None:
        self.lbl_compute.setText(
            runtime_summary_text(self._store)
        )

        space = _get_fk(self._store, "tuner_search_space", {})
        a, t = space_stats(space)
        self.lbl_space.setText(f"Active dims: {a}/{t}")
