# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.header

Setup header bar.

Responsibilities
---------------
- Provide compact actions (icons + tooltips).
- Show dirty/override count.
- Provide "Lock for run" toggle.
- Provide search box.
- Offer a "More" menu with useful actions.

This widget is UI-only:
it emits signals and the parent handles logic.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class SetupHeader(QWidget):
    """Modern header bar for Setup."""

    request_load = pyqtSignal()
    request_save = pyqtSignal()
    request_save_as = pyqtSignal()

    request_reset = pyqtSignal()
    request_apply = pyqtSignal()
    request_diff = pyqtSignal()

    request_export_json = pyqtSignal()
    request_import_json = pyqtSignal()

    request_show_snapshot = pyqtSignal()
    request_show_overrides = pyqtSignal()

    request_copy_snapshot = pyqtSignal()
    request_copy_overrides = pyqtSignal()

    request_help = pyqtSignal()

    lock_changed = pyqtSignal(bool)
    search_changed = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._dirty_count = 0
        self._locked = False

        self._ctx_full = ""
        self._title_full = "Experiment setup"

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        bar = QFrame(self)
        bar.setObjectName("setupHeaderBar")
        bar.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)

        # Left: title + context (elided)
        left = QWidget(bar)
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(0)

        self.lbl_title = QLabel(self._title_full, left)
        self.lbl_title.setObjectName("setupHeaderTitle")

        self.lbl_ctx = QLabel("", left)
        self.lbl_ctx.setObjectName("setupHeaderContext")
        self.lbl_ctx.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        left_l.addWidget(self.lbl_title)
        left_l.addWidget(self.lbl_ctx)

        lay.addWidget(left, 0)

        # Middle: mini actions
        self.btn_load = self._mini_btn(
            bar,
            icon=QStyle.SP_DialogOpenButton,
            text="Load",
            tip="Load JSON snapshot",
        )
        self.btn_save = self._mini_btn(
            bar,
            icon=QStyle.SP_DialogSaveButton,
            text="Save",
            tip="Save JSON snapshot",
        )
        self.btn_apply = self._mini_btn(
            bar,
            icon=QStyle.SP_DialogApplyButton,
            text="Apply",
            tip="Broadcast refresh",
        )
        self.btn_diff = self._mini_btn(
            bar,
            icon=QStyle.SP_FileDialogDetailedView,
            text="Diff",
            tip="Show overrides",
        )
        self.btn_reset = self._mini_btn(
            bar,
            icon=QStyle.SP_BrowserReload,
            text="Reset",
            tip="Reset to defaults",
        )

        lay.addWidget(self.btn_load, 0)
        lay.addWidget(self.btn_save, 0)
        lay.addWidget(self.btn_apply, 0)
        lay.addWidget(self.btn_diff, 0)
        lay.addWidget(self.btn_reset, 0)

        # Dirty badge
        self.lbl_dirty = QLabel("0", bar)
        self.lbl_dirty.setObjectName("setupDirtyBadge")
        self.lbl_dirty.setToolTip("Override count")
        self._sync_dirty_badge()

        lay.addWidget(self.lbl_dirty, 0)

        # Lock toggle (icon + text)
        self._ico_locked = self.style().standardIcon(
            QStyle.SP_DialogCloseButton
        )
        self._ico_unlocked = self.style().standardIcon(
            QStyle.SP_DialogOpenButton
        )

        self.btn_lock = QToolButton(bar)
        self.btn_lock.setObjectName("miniAction")
        self.btn_lock.setCheckable(True)
        self.btn_lock.setCursor(Qt.PointingHandCursor)
        self.btn_lock.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )

        self._sync_lock_btn()

        lay.addWidget(self.btn_lock, 0)

        lay.addStretch(1)

        # Search
        self.search = QLineEdit(bar)
        self.search.setObjectName("setupHeaderSearch")
        self.search.setClearButtonEnabled(True)
        self.search.setPlaceholderText("Search…")

        lay.addWidget(self.search, 1)

        # More menu
        self.btn_more = QToolButton(bar)
        self.btn_more.setObjectName("miniAction")
        self.btn_more.setText("More")
        self.btn_more.setCursor(Qt.PointingHandCursor)
        self.btn_more.setPopupMode(
            QToolButton.InstantPopup
        )

        self._menu = QMenu(self.btn_more)
        self._build_menu(self._menu)
        self.btn_more.setMenu(self._menu)

        lay.addWidget(self.btn_more, 0)

        root.addWidget(bar, 0)

        self._wire()

        # Local styles (safe defaults)
        self.setObjectName("setupHeader")
        self.setStyleSheet(
            "\n".join(
                [
                    "QFrame#setupHeaderBar {",
                    "  border: 1px solid",
                    "    rgba(0,0,0,0.10);",
                    "  border-radius: 12px;",
                    "  background: rgba(0,0,0,0.02);",
                    "}",
                    "QLabel#setupHeaderTitle {",
                    "  font-weight: 700;",
                    "}",
                    "QLabel#setupHeaderContext {",
                    "  opacity: 0.82;",
                    "  font-size: 11px;",
                    "}",
                    "QLabel#setupDirtyBadge {",
                    "  padding: 2px 8px;",
                    "  border-radius: 10px;",
                    "  font-weight: 700;",
                    "  border: 1px solid",
                    "    rgba(0,0,0,0.15);",
                    "}",
                ]
            )
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def set_context(
        self,
        *,
        city: Optional[str] = None,
        model: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> None:
        parts = []
        if city:
            parts.append(str(city))
        if model:
            parts.append(str(model))
        if stage:
            parts.append(str(stage))

        self._ctx_full = "  •  ".join(parts)
        self._apply_elide()

    def set_dirty_count(self, count: int) -> None:
        self._dirty_count = int(max(0, count))
        self._sync_dirty_badge()

    def set_locked(self, locked: bool) -> None:
        self._locked = bool(locked)

        old = self.btn_lock.blockSignals(True)
        self.btn_lock.setChecked(self._locked)
        self.btn_lock.blockSignals(old)

        self._sync_lock_btn()

    def set_search_text(
        self,
        text: str,
        *,
        quiet: bool = False,
    ) -> None:
        if quiet:
            old = self.search.blockSignals(True)
            self.search.setText(str(text or ""))
            self.search.blockSignals(old)
            return
        self.search.setText(str(text or ""))

    # -----------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------
    def resizeEvent(self, ev) -> None:  # type: ignore[override]
        super().resizeEvent(ev)
        self._apply_elide()

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _apply_elide(self) -> None:
        txt = str(self._ctx_full or "")
        if not txt:
            self.lbl_ctx.setText("")
            return

        fm = QFontMetrics(self.lbl_ctx.font())
        w = max(10, int(self.lbl_ctx.width()))
        self.lbl_ctx.setText(
            fm.elidedText(txt, Qt.ElideRight, w)
        )

    def _sync_lock_btn(self) -> None:
        if self._locked:
            self.btn_lock.setIcon(self._ico_locked)
            self.btn_lock.setText("Locked")
            self.btn_lock.setToolTip(
                "Locked (click to unlock edits)"
            )
        else:
            self.btn_lock.setIcon(self._ico_unlocked)
            self.btn_lock.setText("Lock")
            self.btn_lock.setToolTip(
                "Lock config for run"
            )

    def _mini_btn(
        self,
        parent: QWidget,
        *,
        icon: QStyle.StandardPixmap,
        text: str,
        tip: str,
    ) -> QToolButton:
        b = QToolButton(parent)
        b.setObjectName("miniAction")
        b.setText(str(text))
        b.setToolTip(str(tip))
        b.setCursor(Qt.PointingHandCursor)
        b.setIcon(self.style().standardIcon(icon))
        b.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        return b

    def _build_menu(self, menu: QMenu) -> None:
        act = menu.addAction("Save as…")
        act.triggered.connect(self.request_save_as)

        menu.addSeparator()

        act = menu.addAction("Export JSON…")
        act.triggered.connect(self.request_export_json)

        act = menu.addAction("Import JSON…")
        act.triggered.connect(self.request_import_json)

        menu.addSeparator()

        act = menu.addAction("Show snapshot JSON")
        act.triggered.connect(self.request_show_snapshot)

        act = menu.addAction("Show overrides JSON")
        act.triggered.connect(self.request_show_overrides)

        menu.addSeparator()

        act = menu.addAction("Copy snapshot JSON")
        act.triggered.connect(self.request_copy_snapshot)

        act = menu.addAction("Copy overrides JSON")
        act.triggered.connect(self.request_copy_overrides)

        menu.addSeparator()

        act = menu.addAction("Help")
        act.triggered.connect(self.request_help)

    def _wire(self) -> None:
        self.btn_load.clicked.connect(self.request_load)
        self.btn_save.clicked.connect(self.request_save)
        self.btn_reset.clicked.connect(self.request_reset)
        self.btn_apply.clicked.connect(self.request_apply)
        self.btn_diff.clicked.connect(self.request_diff)

        self.btn_lock.toggled.connect(self._on_lock)

        self.search.textChanged.connect(
            self.search_changed
        )

    def _on_lock(self, checked: bool) -> None:
        self._locked = bool(checked)
        self._sync_lock_btn()
        self.lock_changed.emit(self._locked)

    def _sync_dirty_badge(self) -> None:
        n = int(self._dirty_count)
        self.lbl_dirty.setText(str(n))

        dirty = bool(n > 0)
        self.lbl_dirty.setProperty("dirty", dirty)

        # Force re-style in Qt
        self.lbl_dirty.style().unpolish(self.lbl_dirty)
        self.lbl_dirty.style().polish(self.lbl_dirty)
        self.lbl_dirty.update()
