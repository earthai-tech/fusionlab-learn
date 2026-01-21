# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
ConsoleActions: IDE-like toolbar for ConsoleDock.

Buttons:
- Clear, Copy, Save
- Wrap lines (toggle)
- Follow tail (toggle)
- Find: prev / next + Enter key
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLineEdit,
    QToolButton,
)


class ConsoleActions(QWidget):
    clear_clicked = pyqtSignal()
    copy_clicked = pyqtSignal()
    save_clicked = pyqtSignal()

    wrap_toggled = pyqtSignal(bool)
    follow_toggled = pyqtSignal(bool)

    find_next = pyqtSignal(str)
    find_prev = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("consoleActions")

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)

        self.btn_clear = self._btn("Clear", "Clear log")
        self.btn_clear.clicked.connect(self.clear_clicked)
        row.addWidget(self.btn_clear, 0)

        self.btn_copy = self._btn("Copy", "Copy log")
        self.btn_copy.clicked.connect(self.copy_clicked)
        row.addWidget(self.btn_copy, 0)

        self.btn_save = self._btn("Save", "Save log to disk")
        self.btn_save.clicked.connect(self.save_clicked)
        row.addWidget(self.btn_save, 0)

        self.btn_wrap = self._btn("Wrap", "Wrap lines")
        self.btn_wrap.setCheckable(True)
        self.btn_wrap.toggled.connect(self.wrap_toggled)
        row.addWidget(self.btn_wrap, 0)

        self.btn_follow = self._btn("Follow", "Auto scroll")
        self.btn_follow.setCheckable(True)
        self.btn_follow.setChecked(True)
        self.btn_follow.toggled.connect(self.follow_toggled)
        row.addWidget(self.btn_follow, 0)

        row.addSpacing(10)

        self.edt_find = QLineEdit(self)
        self.edt_find.setObjectName("consoleFind")
        self.edt_find.setPlaceholderText("Find…")
        self.edt_find.returnPressed.connect(self._enter_find)
        row.addWidget(self.edt_find, 1)

        self.btn_prev = self._btn("↑", "Find previous")
        self.btn_prev.clicked.connect(self._do_prev)
        row.addWidget(self.btn_prev, 0)

        self.btn_next = self._btn("↓", "Find next")
        self.btn_next.clicked.connect(self._do_next)
        row.addWidget(self.btn_next, 0)

    def _btn(self, text: str, tip: str) -> QToolButton:
        b = QToolButton(self)
        b.setObjectName("miniAction")
        b.setText(text)
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        return b

    def _enter_find(self) -> None:
        self.find_next.emit(self.edt_find.text().strip())

    def _do_next(self) -> None:
        self.find_next.emit(self.edt_find.text().strip())

    def _do_prev(self) -> None:
        self.find_prev.emit(self.edt_find.text().strip())

    def set_states(
        self,
        *,
        wrap: bool,
        follow: bool,
        can_save: bool,
    ) -> None:
        with QSignalBlocker(self.btn_wrap):
            self.btn_wrap.setChecked(bool(wrap))

        with QSignalBlocker(self.btn_follow):
            self.btn_follow.setChecked(bool(follow))

        self.btn_save.setEnabled(bool(can_save))
