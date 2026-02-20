# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QRegion
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
)

from ....config.store import GeoConfigStore
from .advanced import XferMapAdvancedPanel


__all__ = [
    "XferMapAdvDrawer",
    "XferMapAdvWindow",
]


class XferMapAdvDrawer(QWidget):
    """
    Right overlay drawer that sits above the map.

    Owns ONE instance of XferMapAdvancedPanel.
    Can hand it off to XferMapAdvWindow (pin).
    """

    request_pin = pyqtSignal()
    request_close = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("xferAdvOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addStretch(1)

        self.drawer = QFrame(self)
        self.drawer.setObjectName("xferAdvDrawer")
        self.drawer.setFrameShape(QFrame.NoFrame)
        self.drawer.setFixedWidth(380)

        dlay = QVBoxLayout(self.drawer)
        dlay.setContentsMargins(10, 10, 10, 10)
        dlay.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        title = QLabel("Advanced", self.drawer)
        title.setObjectName("xferAdvTitle")
        hdr.addWidget(title, 0)

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
        self.btn_close.clicked.connect(self.request_close.emit)
        hdr.addWidget(self.btn_close, 0)

        dlay.addLayout(hdr, 0)

        self.scroll = QScrollArea(self.drawer)
        self.scroll.setObjectName("xferAdvScroll")
        self.scroll.setWidgetResizable(True)

        self.panel = XferMapAdvancedPanel(
            store=store,
            parent=self.scroll,
        )
        self.scroll.setWidget(self.panel)
        dlay.addWidget(self.scroll, 1)

        root.addWidget(self.drawer, 0)

        self.setVisible(False)

    def set_open(self, on: bool) -> None:
        self.setVisible(bool(on))
        if bool(on):
            self.raise_()
            self.drawer.raise_()
        QTimer.singleShot(0, self._update_mask)

    def _update_mask(self) -> None:
        if not self.isVisible():
            self.clearMask()
            return
        r = self.drawer.geometry()
        self.setMask(QRegion(r))

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        QTimer.singleShot(0, self._update_mask)

    def is_open(self) -> bool:
        return bool(self.isVisible())

    def take_panel(self) -> Optional[QWidget]:
        w = self.scroll.widget()
        if w is None:
            return None
        self.scroll.takeWidget()
        w.setParent(None)
        return w

    def set_panel(self, w: QWidget) -> None:
        if w is None:
            return
        self.scroll.setWidget(w)


class XferMapAdvWindow(QDialog):
    """
    Floating tool window for advanced panel.
    Receives the SAME panel instance via reparent.
    """

    request_unpin = pyqtSignal()

    def __init__(
        self,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("xferAdvWindow")
        self.setWindowFlags(
            Qt.Tool
            | Qt.WindowCloseButtonHint
            | Qt.WindowTitleHint
        )
        self.setWindowTitle("Advanced map tools")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        title = QLabel("Advanced", self)
        title.setObjectName("xferAdvTitle")
        hdr.addWidget(title, 0)

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

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        root.addWidget(self.scroll, 1)

    def set_panel(self, w: QWidget) -> None:
        if w is None:
            return
        self.scroll.setWidget(w)

    def take_panel(self) -> Optional[QWidget]:
        w = self.scroll.widget()
        if w is None:
            return None
        self.scroll.takeWidget()
        w.setParent(None)
        return w

    def closeEvent(self, ev) -> None:
        self.request_unpin.emit()
        super().closeEvent(ev)
