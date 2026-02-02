# geoprior/ui/map/overlay_dock.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
Generic overlay dock primitives (no QDockWidget).

Pattern
-------
- A SideDrawer overlays the map (left/right).
- A FloatingDockWindow is a tool window (pin/pop).
- The SAME panel widget is moved between them by
  take_panel()/set_panel() (reparenting).

This matches the Xfer advanced overlay behaviour.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
)

__all__ = [
    "SideDrawer",
    "FloatingDockWindow",
]


class SideDrawer(QWidget):
    """Side overlay drawer that sits above a host."""

    request_pin = pyqtSignal()
    request_close = pyqtSignal()

    def __init__(
        self,
        *,
        title: str,
        side: str = "right",
        width: int = 380,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._side = str(side or "right").strip().lower()
        if self._side not in ("left", "right"):
            self._side = "right"

        self.setObjectName("gpDockOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if self._side == "right":
            root.addStretch(1)

        self.drawer = QFrame(self)
        self.drawer.setObjectName("gpDockDrawer")
        self.drawer.setFrameShape(QFrame.NoFrame)
        self.drawer.setFixedWidth(int(width) or 380)

        dlay = QVBoxLayout(self.drawer)
        dlay.setContentsMargins(10, 10, 10, 10)
        dlay.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        self.lb_title = QLabel(str(title or ""), self.drawer)
        self.lb_title.setObjectName("gpDockTitle")
        hdr.addWidget(self.lb_title, 0)

        hdr.addStretch(1)

        self.btn_pin = QToolButton(self.drawer)
        self.btn_pin.setObjectName("miniAction")
        self.btn_pin.setAutoRaise(True)
        self.btn_pin.setToolTip("Pin (pop out)")
        self.btn_pin.setIcon(
            self.style().standardIcon(
                QStyle.SP_TitleBarMaxButton
            )
        )
        self.btn_pin.clicked.connect(self.request_pin.emit)
        hdr.addWidget(self.btn_pin, 0)

        self.btn_close = QToolButton(self.drawer)
        self.btn_close.setObjectName("miniAction")
        self.btn_close.setAutoRaise(True)
        self.btn_close.setToolTip("Close")
        self.btn_close.setIcon(
            self.style().standardIcon(
                QStyle.SP_TitleBarCloseButton
            )
        )
        self.btn_close.clicked.connect(
            self.request_close.emit
        )
        hdr.addWidget(self.btn_close, 0)

        dlay.addLayout(hdr, 0)

        self._panel_wrap = QWidget(self.drawer)
        self._panel_lay = QVBoxLayout(self._panel_wrap)
        self._panel_lay.setContentsMargins(0, 0, 0, 0)
        self._panel_lay.setSpacing(0)
        dlay.addWidget(self._panel_wrap, 1)

        root.addWidget(self.drawer, 0)

        if self._side == "left":
            root.addStretch(1)

        self._panel: Optional[QWidget] = None
        self.set_open(False)

    def set_open(self, on: bool) -> None:
        self.setVisible(bool(on))
        if bool(on):
            self.raise_()
            self.drawer.raise_()

    def is_open(self) -> bool:
        return bool(self.isVisible())

    def take_panel(self) -> Optional[QWidget]:
        if self._panel is None:
            return None

        w = self._panel
        self._panel = None

        self._panel_lay.takeAt(0)
        w.setParent(None)
        return w

    def set_panel(self, w: QWidget) -> None:
        if w is None:
            return

        if self._panel is not None:
            old = self.take_panel()
            if old is not None:
                old.deleteLater()

        self._panel = w
        self._panel_lay.addWidget(w, 1)


class FloatingDockWindow(QDialog):
    """Floating tool window that receives a docked panel."""

    request_unpin = pyqtSignal()

    def __init__(
        self,
        *,
        title: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("gpDockWindow")
        self.setWindowFlags(
            Qt.Tool
            | Qt.WindowCloseButtonHint
            | Qt.WindowTitleHint
        )
        self.setWindowTitle(str(title or "Tools"))

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        self.lb_title = QLabel(str(title or ""), self)
        self.lb_title.setObjectName("gpDockTitle")
        hdr.addWidget(self.lb_title, 0)

        hdr.addStretch(1)

        self.btn_unpin = QToolButton(self)
        self.btn_unpin.setObjectName("miniAction")
        self.btn_unpin.setAutoRaise(True)
        self.btn_unpin.setToolTip("Return to drawer")
        self.btn_unpin.setIcon(
            self.style().standardIcon(
                QStyle.SP_TitleBarNormalButton
            )
        )
        self.btn_unpin.clicked.connect(
            self.request_unpin.emit
        )
        hdr.addWidget(self.btn_unpin, 0)

        root.addLayout(hdr, 0)

        self._panel_wrap = QWidget(self)
        self._panel_lay = QVBoxLayout(self._panel_wrap)
        self._panel_lay.setContentsMargins(0, 0, 0, 0)
        self._panel_lay.setSpacing(0)
        root.addWidget(self._panel_wrap, 1)

        self._panel: Optional[QWidget] = None

    def set_panel(self, w: QWidget) -> None:
        if w is None:
            return

        if self._panel is not None:
            old = self.take_panel()
            if old is not None:
                old.deleteLater()

        self._panel = w
        self._panel_lay.addWidget(w, 1)

    def take_panel(self) -> Optional[QWidget]:
        if self._panel is None:
            return None

        w = self._panel
        self._panel = None

        self._panel_lay.takeAt(0)
        w.setParent(None)
        return w

    def closeEvent(self, ev) -> None:
        self.request_unpin.emit()
        super().closeEvent(ev)
