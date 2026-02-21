# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QEvent, QPoint
from PyQt5.QtGui import QRegion, QColor
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
    QGraphicsDropShadowEffect,
    QSizeGrip,
)

from ....config.store import GeoConfigStore
from ...icon_utils import try_icon
from .advanced import XferMapAdvancedPanel


__all__ = [
    "XferMapAdvDrawer",
    "XferMapAdvWindow",
]

K_ADV_WIN_GEOM = "xfer.map.adv.win_geom"

def _std(st: QStyle, sp: QStyle.StandardPixmap):
    return st.standardIcon(sp)

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

        ico_pop = try_icon(
            "popout.svg",
            fallback=self.style().standardIcon(
                QStyle.SP_TitleBarMaxButton
            ),
        )
        
        self.btn_pin = QToolButton(self.drawer)
        self.btn_pin.setObjectName("miniAction")
        self.btn_pin.setAutoRaise(True)
        self.btn_pin.setToolTip("Pin (pop out)")
        self.btn_pin.setIcon(ico_pop)
        self.btn_pin.clicked.connect(self.request_pin.emit)
        hdr.addWidget(self.btn_pin, 0)
        
        # Close
        ico_close = try_icon(
            "close.svg",
            fallback=self.style().standardIcon(
                QStyle.SP_TitleBarCloseButton
            ),
        )
        
        self.btn_close = QToolButton(self.drawer)
        self.btn_close.setObjectName("miniAction")
        self.btn_close.setAutoRaise(True)
        self.btn_close.setToolTip("Close")
        self.btn_close.setIcon(ico_close)
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
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._s = store
        self._geom_key = K_ADV_WIN_GEOM
        self._drag_off: Optional[QPoint] = None

        self.setObjectName("xferAdvWindow")

        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
        )

        self.setWindowTitle("Advanced map tools")

        self.setAttribute(Qt.WA_StyledBackground, True)
        
        # XXX OPTIMIZE: OPTIMIZE 
        # If you a “black rectangle” appears on some Linux compositors, 
        # comment only this line in both windows
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.setMinimumSize(300, 340)
        self.resize(360, 420)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_geom)

        self._restore_geom()
        self._build_ui()

    def _build_ui(self) -> None:
        st = self.style()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(0)

        self._dock = QFrame(self)
        self._dock.setObjectName("xferControlsDock")
        self._dock.setAttribute(Qt.WA_StyledBackground, True)
        self._dock.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self._dock, 1)

        self._apply_shadow(self._dock)

        root = QVBoxLayout(self._dock)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.hdr_bar = QFrame(self._dock)
        self.hdr_bar.setObjectName("xferControlsHdrBar")
        self.hdr_bar.setAttribute(Qt.WA_StyledBackground, True)
        self.hdr_bar.installEventFilter(self)
        root.addWidget(self.hdr_bar, 0)

        hdr = QHBoxLayout(self.hdr_bar)
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        win_title = QLabel("Advanced map tools", self._dock)
        win_title.setObjectName("xferWinTitle")
        hdr.addWidget(win_title, 0)

        hdr.addStretch(1)

        ico_popin = try_icon(
            "popin.svg",
            fallback=st.standardIcon(
                QStyle.SP_TitleBarNormalButton
            ),
        )

        self.btn_unpin = QToolButton(self._dock)
        self.btn_unpin.setObjectName("miniAction")
        self.btn_unpin.setAutoRaise(True)
        self.btn_unpin.setToolTip("Return to drawer")
        self.btn_unpin.setIcon(ico_popin)
        self.btn_unpin.clicked.connect(
            self.request_unpin.emit
        )
        hdr.addWidget(self.btn_unpin, 0)

        ico_close = try_icon(
            "close.svg",
            fallback=st.standardIcon(
                QStyle.SP_TitleBarCloseButton
            ),
        )

        self.btn_close = QToolButton(self._dock)
        self.btn_close.setObjectName("miniAction")
        self.btn_close.setAutoRaise(True)
        self.btn_close.setToolTip("Close")
        self.btn_close.setIcon(ico_close)
        self.btn_close.clicked.connect(self.close)
        hdr.addWidget(self.btn_close, 0)

        self.scroll = QScrollArea(self._dock)
        self.scroll.setObjectName("xferControlsScroll")
        self.scroll.setWidgetResizable(True)
        root.addWidget(self.scroll, 1)

        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        foot.addStretch(1)

        grip = QSizeGrip(self._dock)
        grip.setObjectName("xferWinGrip")
        foot.addWidget(grip, 0)

        root.addLayout(foot, 0)

    def _apply_shadow(self, w: QWidget) -> None:
        eff = QGraphicsDropShadowEffect(w)
        eff.setBlurRadius(28)
        eff.setOffset(0, 10)
        eff.setColor(QColor(0, 0, 0, 90))
        w.setGraphicsEffect(eff)

    def eventFilter(self, obj, ev) -> bool:
        if obj is self.hdr_bar:
            if (
                ev.type() == QEvent.MouseButtonPress
                and ev.button() == Qt.LeftButton
            ):
                self._drag_off = (
                    ev.globalPos()
                    - self.frameGeometry().topLeft()
                )
                return True

            if (
                ev.type() == QEvent.MouseMove
                and self._drag_off is not None
                and (ev.buttons() & Qt.LeftButton)
            ):
                self.move(ev.globalPos() - self._drag_off)
                return True

            if ev.type() == QEvent.MouseButtonRelease:
                self._drag_off = None
                return True

        return super().eventFilter(obj, ev)

    # -------------------------------------------------
    # Geometry persistence (store)
    # -------------------------------------------------
    def _restore_geom(self) -> None:
        g = self._s.get(self._geom_key, None)
        if not isinstance(g, (tuple, list)):
            return
        if len(g) != 4:
            return
        try:
            x, y, w, h = [int(v) for v in g]
        except Exception:
            return

        w = max(int(self.minimumWidth()), w)
        h = max(int(self.minimumHeight()), h)
        self.setGeometry(int(x), int(y), int(w), int(h))

    def _save_geom(self) -> None:
        r = self.geometry()
        g = (
            int(r.x()),
            int(r.y()),
            int(r.width()),
            int(r.height()),
        )
        self._s.set(self._geom_key, g)

    def moveEvent(self, ev) -> None:
        super().moveEvent(ev)
        self._save_timer.start(250)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._save_timer.start(250)

    def hideEvent(self, ev) -> None:
        self._save_geom()
        super().hideEvent(ev)

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
        self._save_geom()
        self.request_unpin.emit()
        super().closeEvent(ev)
