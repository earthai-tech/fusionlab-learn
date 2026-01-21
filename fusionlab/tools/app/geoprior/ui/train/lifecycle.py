# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
    QSizePolicy,
)

from ...config.store import GeoConfigStore


__all__ = ["TrainingLifecycle"]


_LIFE_KEY = "train.lifecycle"
_BASE_KEY = "train.base_model_path"


class TrainingLifecycle(QWidget):
    """
    Training lifecycle controls.

    Store keys (GUI-only by default):
    - "train.lifecycle": "new" | "resume" | "finetune"
    - "train.base_model_path": str

    Notes
    -----
    We use GeoConfigStore.set/get string keys so we can
    store these as UI-only keys (store._extra) today,
    and later promote them to GeoPriorConfig fields
    without changing the UI code.
    """

    changed = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store

        self._build_ui()
        self._wire()
        self.refresh_from_store()

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("New training", "new")
        self.cmb_mode.addItem("Resume", "resume")
        self.cmb_mode.addItem("Fine-tune", "finetune")

        self.ed_base = QLineEdit()
        self.ed_base.setReadOnly(True)
        self.ed_base.setPlaceholderText(
            "Select a model/weights file..."
        )

        self.btn_pick = QPushButton("Select base model…")
        self.btn_pick.setAutoDefault(False)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setAutoDefault(False)

        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)

        g.addWidget(QLabel("Lifecycle:"), 0, 0)
        g.addWidget(self.cmb_mode, 0, 1, 1, 2)

        g.addWidget(QLabel("Base model:"), 1, 0)
        g.addWidget(self.ed_base, 1, 1, 1, 2)

        g.addWidget(self.btn_pick, 2, 1)
        g.addWidget(self.btn_clear, 2, 2)

        g.setColumnStretch(0, 0)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(2, 0)

    def _wire(self) -> None:
        self.cmb_mode.currentIndexChanged.connect(
            self._on_mode_changed
        )
        self.btn_pick.clicked.connect(self._on_pick)
        self.btn_clear.clicked.connect(self._on_clear)

    # -------------------------------------------------
    # Store IO
    # -------------------------------------------------
    def _get_mode(self) -> str:
        v = self._store.get(_LIFE_KEY, "new")
        v = str(v or "new").strip().lower()
        if v not in {"new", "resume", "finetune"}:
            return "new"
        return v

    def _set_mode(self, mode: str) -> None:
        self._store.set(_LIFE_KEY, str(mode))

    def _get_base(self) -> str:
        return str(self._store.get(_BASE_KEY, "") or "")

    def _set_base(self, path: str) -> None:
        self._store.set(_BASE_KEY, str(path or ""))

    # -------------------------------------------------
    # Behavior
    # -------------------------------------------------
    def _sync_enabled(self) -> None:
        mode = self._get_mode()
        is_ft = mode == "finetune"

        self.ed_base.setEnabled(is_ft)
        self.btn_pick.setEnabled(is_ft)
        self.btn_clear.setEnabled(is_ft)

        if not is_ft:
            # Keep UI clean and avoid accidental use.
            self.ed_base.setText("")
            self._set_base("")

    def _on_mode_changed(self) -> None:
        mode = str(self.cmb_mode.currentData() or "new")
        self._set_mode(mode)
        self._sync_enabled()
        self.changed.emit()

    def _on_pick(self) -> None:
        start = self._get_base().strip()
        start_dir = str(Path(start).parent) if start else ""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select base model",
            start_dir,
            "Models (*.keras *.h5 *.ckpt *.pt);;"
            "All files (*)",
        )
        if not path:
            return

        self.ed_base.setText(path)
        self._set_base(path)
        self.changed.emit()

    def _on_clear(self) -> None:
        self.ed_base.setText("")
        self._set_base("")
        self.changed.emit()

    # -------------------------------------------------
    # Public
    # -------------------------------------------------
    def refresh_from_store(self) -> None:
        mode = self._get_mode()
        idx = self.cmb_mode.findData(mode)
        if idx < 0:
            idx = 0

        self.cmb_mode.setCurrentIndex(idx)
        self.ed_base.setText(self._get_base())
        self._sync_enabled()
