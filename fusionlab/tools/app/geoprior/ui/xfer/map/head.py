# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.map.head

Map head card for Xfer MAP mode.

This head *hosts the existing XferMapToolbar* (from
XferMapPage) so you do not duplicate controls.

Signals are forwarded:
- expand_toggled -> from toolbar.request_expand
- open_run_clicked -> from toolbar.request_open_options
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .toolbar import XferMapToolbar


__all__ = ["XferMapHeadBar"]


class XferMapHeadBar(QFrame):
    """
    Map head “card” that hosts the existing toolbar.

    Public API
    ----------
    - set_toolbar(toolbar)
    - take_toolbar() -> toolbar (or None)
    """

    expand_toggled = pyqtSignal(bool)
    open_run_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._toolbar: Optional[XferMapToolbar] = None

        self.setObjectName("mapHeadCard")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        # Top row: title + small pill
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(10)

        self.lbl_pill = QLabel("MAP", self)
        self.lbl_pill.setObjectName("mapHeadPill")

        self.lbl_title = QLabel("Transfer map", self)
        self.lbl_title.setObjectName("mapHeadKey")

        self.lbl_hint = QLabel(
            "Explore transferability metrics and overlays.",
            self,
        )
        self.lbl_hint.setObjectName("mapHeadDataset")
        self.lbl_hint.setWordWrap(True)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)
        title_col.addWidget(self.lbl_title)
        title_col.addWidget(self.lbl_hint)

        top.addWidget(self.lbl_pill, 0)
        top.addLayout(title_col, 1)
        
        self.btn_go_run = QToolButton(self)
        self.btn_go_run.setObjectName("miniAction")
        self.btn_go_run.setAutoRaise(True)
        self.btn_go_run.setToolTip("Go to Run mode")
        self.btn_go_run.setIcon(
            self.style().standardIcon(QStyle.SP_ArrowLeft)
        )
        self.btn_go_run.clicked.connect(
            self.open_run_clicked.emit
        )
        
        top.addWidget(self.btn_go_run, 0, Qt.AlignTop)

        root.addLayout(top)

        # Toolbar host (the existing XferMapToolbar lives here)
        self._host = QFrame(self)
        self._host.setObjectName("mapHeadGroup")
        self._host.setFrameShape(QFrame.NoFrame)
        self._host.setAttribute(Qt.WA_StyledBackground, True)

        self._host_l = QVBoxLayout(self._host)
        self._host_l.setContentsMargins(8, 8, 8, 8)
        self._host_l.setSpacing(0)

        root.addWidget(self._host)

    # -------------------------------------------------
    # Public
    # -------------------------------------------------
    def set_toolbar(self, tb: XferMapToolbar) -> None:
        """
        Reparent toolbar into this head host and wire signals.
        """
        if tb is None:
            return

        if self._toolbar is tb:
            return

        # Remove old toolbar if any
        old = self._toolbar
        if old is not None:
            self._host_l.removeWidget(old)
            old.setParent(None)

        self._toolbar = tb
        tb.setParent(self._host)
        self._host_l.addWidget(tb)

        # Forward key signals (disconnect-safe)
        try:
            tb.request_expand.disconnect()
        except Exception:
            pass
        try:
            tb.request_open_options.disconnect()
        except Exception:
            pass

        # Ensure menu label matches current head mode (MAP)
        if hasattr(tb, "set_mode"):
            tb.set_mode("map")
        
        tb.request_expand.connect(self.expand_toggled.emit)
        
        # legacy + new switch signal
        tb.request_open_options.connect(self.open_run_clicked.emit)
        
        if hasattr(tb, "request_mode_switch"):
            tb.request_mode_switch.connect(self._on_tb_mode_switch)

    def _on_tb_mode_switch(self, mode: str) -> None:
        m = str(mode or "").strip().lower()
        if m == "run":
            self.open_run_clicked.emit()

    def take_toolbar(self) -> Optional[XferMapToolbar]:
        """
        Detach and return the toolbar (caller can reparent).
        """
        tb = self._toolbar
        if tb is None:
            return None

        self._host_l.removeWidget(tb)
        tb.setParent(None)
        self._toolbar = None
        return tb
