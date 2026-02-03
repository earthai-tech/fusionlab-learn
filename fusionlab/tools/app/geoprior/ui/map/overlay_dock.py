# geoprior/ui/map/overlay_dock.py
# Drop-in replacement: uses try_icon() with SVGs + fallbacks.
# Supports side="left"|"right"|"panel" to pick dock icons.

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QGuiApplication, QIcon
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
    QApplication,
    QSizePolicy
)

from ..icon_utils import try_icon


__all__ = [
    "SideDrawer",
    "FloatingDockWindow",
]


def _icon_or_std(
    w: QWidget,
    svg_name: str,
    std: QStyle.StandardPixmap,
) -> QIcon:
    ico = try_icon(svg_name)
    if ico is not None and (not ico.isNull()):
        return ico
    return w.style().standardIcon(std)


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
            _icon_or_std(
                self.btn_pin,
                "pin.svg",
                QStyle.SP_TitleBarMaxButton,
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

        self._panel_wrap = QWidget(self.drawer)
        self._panel_wrap.setObjectName("gpDockBody")
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
        side: str = "right",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._side = str(side or "right").strip().lower()
        if self._side not in ("left", "right", "panel"):
            self._side = "right"

        self._auto_sized = False

        self.setObjectName("gpDockWindow")
        self.setWindowFlags(
            Qt.Tool
            | Qt.WindowCloseButtonHint
            | Qt.WindowTitleHint
        )
        self.setWindowTitle(str(title or "Tools"))
        self.setSizeGripEnabled(True)

        win_ico = try_icon("dock-panel.svg")
        if win_ico is not None and (not win_ico.isNull()):
            self.setWindowIcon(win_ico)

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

        if self._side == "left":
            svg = "dock-left.svg"
        elif self._side == "panel":
            svg = "dock-panel.svg"
        else:
            svg = "dock-right.svg"

        self.btn_unpin = QToolButton(self)
        self.btn_unpin.setObjectName("miniAction")
        self.btn_unpin.setAutoRaise(True)
        self.btn_unpin.setToolTip("Return to dock")
        self.btn_unpin.setIcon(
            _icon_or_std(
                self.btn_unpin,
                svg,
                QStyle.SP_TitleBarNormalButton,
            )
        )
        self.btn_unpin.clicked.connect(self.request_unpin.emit)
        hdr.addWidget(self.btn_unpin, 0)

        root.addLayout(hdr, 0)

        self._panel_wrap = QWidget(self)
        self._panel_wrap.setObjectName("gpDockBody")
        self._panel_wrap.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self._panel_lay = QVBoxLayout(self._panel_wrap)
        self._panel_lay.setContentsMargins(0, 0, 0, 0)
        self._panel_lay.setSpacing(0)

        root.addWidget(self._panel_wrap, 1)

        self._panel: Optional[QWidget] = None

    def _auto_size_from_panel(self, w: QWidget) -> None:
        if self._auto_sized:
            return

        scr = self.screen()
        if scr is None:
            scr = QGuiApplication.primaryScreen()
        if scr is None:
            return

        geo = scr.availableGeometry()

        hint = w.sizeHint()
        minh = w.minimumSizeHint()
        want = hint.expandedTo(minh)

        # For "panel" (analytics), prefer wide default
        # without hard-coded pixels: use screen fractions.
        if self._side == "panel":
            want = want.expandedTo(
                QSize(
                    int(geo.width() * 0.60), # 0.78
                    int(geo.height() * 0.52), # 0.62
                )
            )

        # Keep inside the available screen
        want = QSize(
            min(want.width(), int(geo.width() * 0.95)),
            min(want.height(), int(geo.height() * 0.95)),
        )

        # Respect panel's minimum sizing
        self.setMinimumSize(minh)
        self.resize(want)

        self._auto_sized = True

    def set_panel(self, w: QWidget) -> None:
        if w is None:
            return

        if self._panel is not None:
            old = self.take_panel()
            if old is not None:
                old.deleteLater()

        self._panel = w

        w.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self._panel_lay.addWidget(w, 1)

        self._auto_size_from_panel(w)

    def take_panel(self) -> Optional[QWidget]:
        if self._panel is None:
            return None

        w = self._panel
        self._panel = None

        self._panel_lay.takeAt(0)
        w.setParent(None)
        return w

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        if self._panel is not None:
            self._auto_size_from_panel(self._panel)

    def closeEvent(self, ev) -> None:
        self.request_unpin.emit()
        super().closeEvent(ev)
