# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
ConsoleActions: modern toolbar for ConsoleDock.

- Icon-first buttons
- Visual grouping with separators
- Overflow "More" menu
- Pause UI (buffer logs, stop appending) + pending badge
- Find: prev/next + match counter
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import ( 
    Qt, 
    QSignalBlocker, 
    pyqtSignal, 
    QEvent
)
from PyQt5.QtWidgets import (
    QAction,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QStyle,
    QToolButton,
    QWidget,
)


class ConsoleActions(QWidget):
    clear_clicked = pyqtSignal()
    copy_clicked = pyqtSignal()
    save_clicked = pyqtSignal()

    wrap_toggled = pyqtSignal(bool)
    follow_toggled = pyqtSignal(bool)
    pause_toggled = pyqtSignal(bool)

    find_next = pyqtSignal(str)
    find_prev = pyqtSignal(str)

    open_out_clicked = pyqtSignal()
    copy_out_clicked = pyqtSignal()
    ui_limit_requested = pyqtSignal(int)
    find_opts_changed = pyqtSignal(bool, bool, bool)
    timestamps_toggled = pyqtSignal(bool)
    
    find_live = pyqtSignal(str)
    find_cleared = pyqtSignal()
    find_opts_changed = pyqtSignal(bool, bool, bool)
    highlight_toggled = pyqtSignal(bool)
    
    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("consoleActions")

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)

        # --------------------
        # Group A: log ops
        # --------------------
        self.btn_clear = self._ibtn(
            self.style().standardIcon(
                QStyle.SP_DialogResetButton
            ),
            "Clear log",
        )
        self.btn_clear.clicked.connect(self.clear_clicked)
        row.addWidget(self.btn_clear, 0)

        self.btn_copy = self._tbtn("⧉", "Copy log")
        self.btn_copy.clicked.connect(self.copy_clicked)
        row.addWidget(self.btn_copy, 0)

        self.btn_save = self._ibtn(
            self.style().standardIcon(
                QStyle.SP_DialogSaveButton
            ),
            "Save log to disk",
        )
        self.btn_save.clicked.connect(self.save_clicked)
        row.addWidget(self.btn_save, 0)

        self.sep_a = self._sep()
        row.addWidget(self.sep_a, 0)

        # --------------------
        # Group B: view
        # --------------------
        self.btn_wrap = self._tbtn("↩", "Wrap lines")
        self.btn_wrap.setCheckable(True)
        self.btn_wrap.toggled.connect(self.wrap_toggled)
        row.addWidget(self.btn_wrap, 0)

        self.btn_follow = self._tbtn("⇣", "Auto scroll")
        self.btn_follow.setCheckable(True)
        self.btn_follow.setChecked(True)
        self.btn_follow.toggled.connect(self.follow_toggled)
        row.addWidget(self.btn_follow, 0)

        # Pause (single button)
        self.btn_pause = self._tbtn("⏸", "Pause UI updates")
        self.btn_pause.setObjectName("consolePause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._on_pause)
        row.addWidget(self.btn_pause, 0)

        # Pending badge (only visible when paused + pending>0)
        self.lbl_pending = QLabel("", self)
        self.lbl_pending.setObjectName("consolePending")
        self.lbl_pending.setVisible(False)
        self.lbl_pending.setAlignment(
            Qt.AlignCenter | Qt.AlignVCenter
        )
        row.addWidget(self.lbl_pending, 0)

        self.sep_b = self._sep()
        row.addWidget(self.sep_b, 0)

        # Find options
        self.btn_case = self._tbtn("Aa", "Case sensitive")
        self.btn_case.setCheckable(True)
        self.btn_case.toggled.connect(self._emit_find_opts)
        row.addWidget(self.btn_case, 0)

        self.btn_regex = self._tbtn(".*", "Regex search")
        self.btn_regex.setCheckable(True)
        self.btn_regex.toggled.connect(self._emit_find_opts)
        row.addWidget(self.btn_regex, 0)

        self.btn_word = self._tbtn("W", "Whole word")
        self.btn_word.setCheckable(True)
        self.btn_word.toggled.connect(self._emit_find_opts)
        row.addWidget(self.btn_word, 0)

        self.btn_hl = self._tbtn("✦", "Highlight matches")
        self.btn_hl.setCheckable(True)
        self.btn_hl.setChecked(True)
        self.btn_hl.toggled.connect(self.highlight_toggled)
        row.addWidget(self.btn_hl, 0)

        # --------------------
        # Group C: search
        # --------------------
        self.edt_find = QLineEdit(self)
        self.edt_find.setObjectName("consoleFind")
        self.edt_find.setPlaceholderText("Find…")
        self.edt_find.returnPressed.connect(self._enter_find)
        self.edt_find.textChanged.connect(self._on_find_live)
        self.edt_find.installEventFilter(self)
        
        row.addWidget(self.edt_find, 1)

        self.btn_prev = self._tbtn("↑", "Find previous")
        self.btn_prev.clicked.connect(self._do_prev)
        row.addWidget(self.btn_prev, 0)

        self.btn_next = self._tbtn("↓", "Find next")
        self.btn_next.clicked.connect(self._do_next)
        row.addWidget(self.btn_next, 0)

        self.lbl_match = QLabel("", self)
        self.lbl_match.setObjectName("consoleMatch")
        self.lbl_match.setAlignment(
            Qt.AlignCenter | Qt.AlignVCenter
        )
        row.addWidget(self.lbl_match, 0)

        # --------------------
        # Overflow: More
        # --------------------
        self.btn_more = self._tbtn("⋯", "More actions")
        self.btn_more.setObjectName("consoleMore")
        self.btn_more.setPopupMode(
            QToolButton.InstantPopup
        )
        self._menu = self._build_menu()
        self.btn_more.setMenu(self._menu)
        row.addWidget(self.btn_more, 0)

    # -------------------------
    # Public UI setters
    # -------------------------
    def eventFilter(self, obj, ev) -> bool:
        if obj is self.edt_find:
            if ev.type() == QEvent.KeyPress:
                if ev.key() == Qt.Key_Escape:
                    self._clear_find()
                    return True
        return super().eventFilter(obj, ev)

    def _clear_find(self) -> None:
        self.edt_find.setText("")
        self.set_match(0, 0)
        self.find_cleared.emit()

    def _on_find_live(self, txt: str) -> None:
        self.find_live.emit(str(txt).strip())

    def _emit_find_opts(self, *_: object) -> None:
        self.find_opts_changed.emit(
            bool(self.btn_case.isChecked()),
            bool(self.btn_regex.isChecked()),
            bool(self.btn_word.isChecked()),
        )

    def set_pending(self, n: int) -> None:
        n = int(n)
        if n <= 0:
            self.lbl_pending.setVisible(False)
            self.lbl_pending.setText("")
            self.lbl_pending.setToolTip("")
            return
    
        self.lbl_pending.setVisible(True)
        self.lbl_pending.setText(f"Paused ({n:,})")
        self.lbl_pending.setToolTip(
            f"{n:,} buffered line(s) while paused"
        )


    def set_match(self, cur: int, total: int) -> None:
        if total <= 0 or cur <= 0:
            self.lbl_match.setText("")
            self.lbl_match.setToolTip("")
            return
        self.lbl_match.setText(f"{cur:d}/{total:d}")
        self.lbl_match.setToolTip("Match index / total")

    def set_states(
        self,
        *,
        wrap: bool,
        follow: bool,
        paused: bool,
        pending: int,
        can_save: bool,
    ) -> None:
        with QSignalBlocker(self.btn_wrap):
            self.btn_wrap.setChecked(bool(wrap))

        with QSignalBlocker(self.btn_follow):
            self.btn_follow.setChecked(bool(follow))

        with QSignalBlocker(self.btn_pause):
            self.btn_pause.setChecked(bool(paused))
            self.btn_pause.setText("▶" if paused else "⏸")
            self.btn_pause.setToolTip(
                "Resume UI updates" if paused else "Pause UI updates"
            )

        self.set_pending(pending if paused else 0)
        self.btn_save.setEnabled(bool(can_save))

    def set_compact(self, compact: bool) -> None:
        """Show minimal toolbar when docked."""
        c = bool(compact)
        full = not c

        hide = [
            self.btn_clear,
            self.btn_copy,
            self.btn_save,
            getattr(self, "sep_a", None),
            self.btn_wrap,
            self.btn_follow,
            self.btn_case,
            self.btn_regex,
            self.btn_word,
            self.btn_hl,
            getattr(self, "sep_b", None),
            self.btn_prev,
            self.btn_next,
            self.lbl_match,
        ]

        for w in hide:
            if w is not None:
                w.setVisible(full)

        if c:
            self.edt_find.setMaximumWidth(180)
        else:
            self.edt_find.setMaximumWidth(16777215)


    # -------------------------
    # Internals
    # -------------------------
    def _build_menu(self) -> QMenu:
        m = QMenu(self)

        a_clear = QAction("Clear log", m)
        a_clear.triggered.connect(self.clear_clicked)
        m.addAction(a_clear)

        a_copy = QAction("Copy log", m)
        a_copy.triggered.connect(self.copy_clicked)
        m.addAction(a_copy)

        a_save = QAction("Save log", m)
        a_save.triggered.connect(self.save_clicked)
        m.addAction(a_save)

        m.addSeparator()

        a_open = QAction("Open output folder", m)
        a_open.triggered.connect(self.open_out_clicked)
        m.addAction(a_open)

        a_copy = QAction("Copy output path", m)
        a_copy.triggered.connect(self.copy_out_clicked)
        m.addAction(a_copy)

        m.addSeparator()

        a_ts = QAction("Show timestamps", m)
        a_ts.setCheckable(True)
        a_ts.setChecked(True)
        a_ts.toggled.connect(self.timestamps_toggled)
        m.addAction(a_ts)

        m.addSeparator()

        sub = m.addMenu("Visible lines")
        for n in (200, 500, 800, 1500):
            a = QAction(f"{n}", sub)
            a.triggered.connect(
                lambda _=False, x=n: self.ui_limit_requested.emit(x)
            )
            sub.addAction(a)

        return m

    def _sep(self) -> QFrame:
        f = QFrame(self)
        f.setObjectName("consoleSep")
        f.setFrameShape(QFrame.VLine)
        f.setFrameShadow(QFrame.Plain)
        return f

    def _ibtn(self, icon, tip: str) -> QToolButton:
        b = QToolButton(self)
        b.setObjectName("miniAction")
        b.setIcon(icon)
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        return b

    def _tbtn(self, text: str, tip: str) -> QToolButton:
        b = QToolButton(self)
        b.setObjectName("miniAction")
        b.setText(text)
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        return b

    def _on_pause(self, on: bool) -> None:
        self.pause_toggled.emit(bool(on))

    def _enter_find(self) -> None:
        self.find_next.emit(self.edt_find.text().strip())

    def _do_next(self) -> None:
        self.find_next.emit(self.edt_find.text().strip())

    def _do_prev(self) -> None:
        self.find_prev.emit(self.edt_find.text().strip())
