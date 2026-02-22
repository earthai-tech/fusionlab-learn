# geoprior/ui/tune/details.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
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

        # ---------------------------------------------------------
        # Single scroll area for ALL content (compute + stats)
        # ---------------------------------------------------------
        sc = QScrollArea(self)
        sc.setObjectName("trainCompScroll")  # reuse existing QSS
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidgetResizable(True)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        body = QWidget(sc)
        body.setObjectName("trainCompBody")
        sc.setWidget(body)

        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        # =========================================================
        # Computer details
        # =========================================================
        t1 = QLabel("Computer details", body)
        t1.setObjectName("trainNavTitle")
        lay.addWidget(t1, 0)

        self.lbl_compute = QLabel("", body)
        self.lbl_compute.setObjectName("runComputeText")
        self.lbl_compute.setWordWrap(True)
        self.lbl_compute.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        self.lbl_compute.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )
        lay.addWidget(self.lbl_compute, 0)

        notes = QLabel(
            "• Backend/Device info updates when runtime changes.\n"
            "• GPU visibility depends on the selected backend.\n"
            "• Use presets to keep tuning runs reproducible.",
            body,
        )
        notes.setWordWrap(True)
        notes.setObjectName("sumLine")
        lay.addWidget(notes, 0)

        # =========================================================
        # Search-space stats
        # =========================================================
        t2 = QLabel("Search-space stats", body)
        t2.setObjectName("trainNavTitle")
        lay.addWidget(t2, 0)

        self.lbl_space = QLabel("", body)
        self.lbl_space.setWordWrap(True)
        lay.addWidget(self.lbl_space, 0)

        lay.addStretch(1)

        sc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(sc, 1)

    def refresh_from_store(self) -> None:
        self.lbl_compute.setText(
            runtime_summary_text(self._store)
        )

        space = _get_fk(self._store, "tuner_search_space", {})
        a, t = space_stats(space)
        self.lbl_space.setText(f"Active dims: {a}/{t}")
