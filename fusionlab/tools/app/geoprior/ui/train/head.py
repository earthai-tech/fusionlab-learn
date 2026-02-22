# ui/train/head.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QStyle,
    QToolButton,
    QWidget,
)

from ..icon_utils import try_icon
from ..common.lifecycle_strip import LifecycleStrip
from ...config.store import GeoConfigStore


__all__ = ["TrainHeadBar"]


_PRESET_KEY = "train.preset_name"
_SEARCH_KEY = "train.head.search"
_LIFE_KEY = "train.lifecycle"
_BASE_KEY = "train.base_model_path"

class TrainHeadBar(QFrame):
    """
    Train head command bar.

    This bar is meant to stay pinned (non-scroll).
    It does not duplicate Navigator actions.

    Signals
    -------
    - reset_requested: user wants reset to defaults/custom
    - copy_plan_requested: user wants run plan copied
    - config_clicked: open config dialog
    - filter_clicked: optional filter UX hook
    - search_changed: live search text for jumping cards
    - lifecycle_changed: lifecycle/base model changed
    """

    toast = pyqtSignal(str)

    reset_requested = pyqtSignal()
    copy_plan_requested = pyqtSignal()
    config_clicked = pyqtSignal()
    filter_clicked = pyqtSignal()
    search_changed = pyqtSignal(str)
    lifecycle_changed = pyqtSignal()
    preset_changed = pyqtSignal(str)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._plan_text = ""

        self.setObjectName("trainHead")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()
        self._wire()
        self.refresh_from_store()

    # -------------------------------------------------
    # Icons
    # -------------------------------------------------
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

    def _mk_icon_btn(
        self,
        tip: str,
        icon_name: str,
        fallback: QStyle.StandardPixmap,
    ) -> QToolButton:
        b = QToolButton(self)
        b.setAutoRaise(True)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        b.setToolTip(tip)
        self._set_icon(b, icon_name, fallback)
        return b

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)
    
        self.lifecycle = LifecycleStrip(
            store=self._store,
            life_key="train.lifecycle",
            base_key="train.base_model_path",
        )
        row.addWidget(self.lifecycle, 1)
    
        row.addWidget(QLabel("Preset:"), 0)
    
        self.cmb_preset = QComboBox(self)
        self.cmb_preset.setMinimumWidth(160)
        row.addWidget(self.cmb_preset, 0)
    
        self.btn_reset = self._mk_icon_btn(
            "Reset to Custom/defaults",
            "reset.svg",
            QStyle.SP_BrowserReload,
        )
        self.btn_copy = self._mk_icon_btn(
            "Copy run plan",
            "copy.svg",
            QStyle.SP_DialogSaveButton,
        )
        self.btn_cfg = self._mk_icon_btn(
            "Open config",
            "settings.svg",
            QStyle.SP_FileDialogDetailedView,
        )
    
        row.addWidget(self.btn_reset, 0)
        row.addWidget(self.btn_copy, 0)
        row.addWidget(self.btn_cfg, 0)
    
        row.addStretch(1)
    
        self.btn_filter = self._mk_icon_btn(
            "Filter / search",
            "filter2.svg",
            QStyle.SP_FileDialogContentsView,
        )
        self.ed_search = QLineEdit(self)
        self.ed_search.setPlaceholderText("Search settings...")
        self.ed_search.setMaximumWidth(260)
    
        row.addWidget(self.btn_filter, 0)
        row.addWidget(self.ed_search, 0)

    def _wire(self) -> None:
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_copy.clicked.connect(self._on_copy)
        self.btn_cfg.clicked.connect(self.config_clicked.emit)
        self.btn_filter.clicked.connect(self._on_filter)
    
        self.ed_search.textChanged.connect(self._on_search)
    
        self.cmb_preset.currentIndexChanged.connect(
            lambda _=0: self._on_preset_changed()
        )
    
        self.lifecycle.changed.connect(
            self.lifecycle_changed.emit
        )

        
    def set_presets(self, names: list) -> None:
        cur = str(self._store.get(_PRESET_KEY, "Custom") or "")
        cur = cur.strip() or "Custom"
        with QSignalBlocker(self.cmb_preset):
            self.cmb_preset.clear()
            for n in names or ["Custom"]:
                self.cmb_preset.addItem(str(n))
            idx = self.cmb_preset.findText(cur)
            if idx < 0:
                idx = 0
            self.cmb_preset.setCurrentIndex(idx)
    
    def _on_preset_changed(self) -> None:
        nm = str(self.cmb_preset.currentText() or "")
        nm = nm.strip() or "Custom"
        self._store.set(_PRESET_KEY, nm)
        self.preset_changed.emit(nm)
    
    def refresh_from_store(self) -> None:
        nm = str(self._store.get(_PRESET_KEY, "Custom") or "")
        nm = nm.strip() or "Custom"
        idx = self.cmb_preset.findText(nm)
        if idx < 0:
            idx = 0
        with QSignalBlocker(self.cmb_preset):
            self.cmb_preset.setCurrentIndex(idx)
    
        s = str(self._store.get(_SEARCH_KEY, "") or "")
        if self.ed_search.text() != s:
            self.ed_search.setText(s)
    
        self.lifecycle.refresh_from_store()

    # -------------------------------------------------
    # Behavior
    # -------------------------------------------------
    def _on_reset(self) -> None:
        self.reset_requested.emit()

    def _on_copy(self) -> None:
        txt = (self._plan_text or "").strip()
        if not txt:
            self.copy_plan_requested.emit()
            return

        QApplication.clipboard().setText(txt)
        self.toast.emit("Run plan copied.")

    def _on_filter(self) -> None:
        self.ed_search.setFocus(Qt.OtherFocusReason)
        self.ed_search.selectAll()
        self.filter_clicked.emit()

    def _on_search(self, text: str) -> None:
        self._store.set(_SEARCH_KEY, str(text or ""))
        self.search_changed.emit(text)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def set_preset_name(self, name: str) -> None:
        nm = str(name or "Custom").strip() or "Custom"
        self._store.set(_PRESET_KEY, nm)
    
        idx = self.cmb_preset.findText(nm)
        if idx < 0:
            idx = 0
        with QSignalBlocker(self.cmb_preset):
            self.cmb_preset.setCurrentIndex(idx)

    def set_plan_text(self, text: str) -> None:
        self._plan_text = str(text or "")


