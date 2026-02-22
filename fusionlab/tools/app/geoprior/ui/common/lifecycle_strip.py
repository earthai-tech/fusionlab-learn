# geoprior/ui/common/lifecycle_strip.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QStyle,
    QToolButton,
    QWidget,
)

from ..icon_utils import try_icon
from ...config.store import GeoConfigStore


__all__ = ["LifecycleStrip"]


class LifecycleStrip(QFrame):
    """
    Compact lifecycle row used by Train/Tune head bars.

    This is UI-only (store string keys), so Train/Tune can
    share the exact behavior without duplicating logic.
    """

    changed = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        life_key: str,
        base_key: str,
        base_modes: Optional[Set[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._life_key = str(life_key)
        self._base_key = str(base_key)
        self._base_modes = set(base_modes or {"finetune"})

        self.setObjectName("lifecycleStrip")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()
        self._wire()
        self.refresh_from_store()

    def _std_icon(self, sp: QStyle.StandardPixmap):
        return self.style().standardIcon(sp)

    def _set_icon(
        self,
        btn: QToolButton,
        name: str,
        fallback: QStyle.StandardPixmap,
    ) -> None:
        ic = try_icon(name)
        if ic is None:
            ic = self._std_icon(fallback)
        btn.setIcon(ic)

    def _build_ui(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        row.addWidget(QLabel("Lifecycle:"), 0)

        self.cmb_mode = QComboBox(self)
        self.cmb_mode.addItem("New training", "new")
        self.cmb_mode.addItem("Resume", "resume")
        self.cmb_mode.addItem("Fine-tune", "finetune")
        self.cmb_mode.setMinimumWidth(150)
        row.addWidget(self.cmb_mode, 0)

        self._base_group = QWidget(self)
        g = QHBoxLayout(self._base_group)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(6)

        self.lbl_base = QLabel("Base:")
        self.ed_base = QLineEdit(self)
        self.ed_base.setReadOnly(True)
        self.ed_base.setPlaceholderText(
            "Select a model/weights file..."
        )

        self.btn_pick = QToolButton(self)
        self.btn_pick.setAutoRaise(True)
        self.btn_pick.setToolTip("Select base model")
        self._set_icon(
            self.btn_pick,
            "folder_open.svg",
            QStyle.SP_DialogOpenButton,
        )

        self.btn_clear = QToolButton(self)
        self.btn_clear.setAutoRaise(True)
        self.btn_clear.setToolTip("Clear base model")
        self._set_icon(
            self.btn_clear,
            "close.svg",
            QStyle.SP_DialogCloseButton,
        )

        g.addWidget(self.lbl_base, 0)
        g.addWidget(self.ed_base, 1)
        g.addWidget(self.btn_pick, 0)
        g.addWidget(self.btn_clear, 0)

        row.addWidget(self._base_group, 1)

    def _wire(self) -> None:
        self.cmb_mode.currentIndexChanged.connect(
            self._on_mode_changed
        )
        self.btn_pick.clicked.connect(self._on_pick)
        self.btn_clear.clicked.connect(self._on_clear)

    def _get_mode(self) -> str:
        v = self._store.get(self._life_key, "new")
        v = str(v or "new").strip().lower()
        if v not in {"new", "resume", "finetune"}:
            return "new"
        return v

    def _set_mode(self, mode: str) -> None:
        self._store.set(self._life_key, str(mode))

    def _get_base(self) -> str:
        return str(self._store.get(self._base_key, "") or "")

    def _set_base(self, path: str) -> None:
        self._store.set(self._base_key, str(path or ""))

    def _sync_enabled(self) -> None:
        mode = self._get_mode()
        show = mode in self._base_modes

        self._base_group.setVisible(show)

        if not show:
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

    def refresh_from_store(self) -> None:
        mode = self._get_mode()
        idx = self.cmb_mode.findData(mode)
        if idx < 0:
            idx = 0

        with QSignalBlocker(self.cmb_mode):
            self.cmb_mode.setCurrentIndex(idx)

        self.ed_base.setText(self._get_base())
        self._sync_enabled()
