# geoprior/ui/map/overlay_dock.py
# Drop-in replacement: uses try_icon() with SVGs + fallbacks.
# Supports side="left"|"right"|"panel" to pick dock icons.

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QSize, QEvent, QPoint, pyqtSignal
from PyQt5.QtGui import QGuiApplication, QIcon, QColor
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
    QSizePolicy, 
    QGraphicsDropShadowEffect,
    QSizeGrip,
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
        self.btn_pin.setToolTip("Undock (pop out)")
        self.btn_pin.setIcon(
            _icon_or_std(
                self.btn_pin,
                "popout.svg",
                QStyle.SP_TitleBarMaxButton,
            )
        )
        self.btn_pin.clicked.connect(self._on_pin_clicked)
        hdr.addWidget(self.btn_pin, 0)

        self.btn_close = QToolButton(self.drawer)
        self.btn_close.setObjectName("miniAction")
        self.btn_close.setAutoRaise(True)
        self.btn_close.setToolTip("Close")
        self.btn_close.setIcon(
            _icon_or_std(
                self.btn_close,
                "close.svg",
                QStyle.SP_TitleBarCloseButton,
            )
        )
        self.btn_close.clicked.connect(self._on_close_clicked)
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

        # stable "intended open" state (not tied to visibility)
        self._open = False

        # captured state at pin time (for parent to restore)
        self._pre_pin_open: Optional[bool] = None

        self.set_open(False)
        

    def _on_pin_clicked(self) -> None:
        # capture current intended state BEFORE any changes
        self._pre_pin_open = bool(self._open)
        self.request_pin.emit()

    def _on_close_clicked(self) -> None:
        # closing should update intended state too
        self.set_open(False)
        self.request_close.emit()

    def set_open(self, on: bool) -> None:
        on = bool(on)
        self._open = on
        self.setVisible(on)
        if on:
            self.raise_()
            self.drawer.raise_()

    def is_open(self) -> bool:
        # do NOT rely on isVisible(), it can flicker during reparent
        return bool(self._open)
    
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
    request_close = pyqtSignal()

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
        self._panel: Optional[QWidget] = None
        self._drag_off: Optional[QPoint] = None

        self.setObjectName("gpDockWindow")

        # Match Xfer: frameless + card inside.
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground, True)

        # If you see a black rectangle on some Linux compositors,
        # comment ONLY this line (same as Xfer windows).
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Keep outer window transparent; the inner card is styled.
        self.setStyleSheet(
            "QDialog#gpDockWindow{background:transparent;border:none;}"
        )

        self.setWindowTitle(str(title or "Tools"))
        self.setMinimumSize(280, 240)
        self.resize(380, 460)

        self._build_ui(title=title)

        win_ico = try_icon("dock-panel.svg")
        if win_ico is not None and (not win_ico.isNull()):
            self.setWindowIcon(win_ico)

    def _build_ui(self, *, title: str) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)

        # Inner “card”
        self._dock = QFrame(self)
        self._dock.setObjectName("gpDockDrawer")
        self._dock.setAttribute(Qt.WA_StyledBackground, True)
        self._dock.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self._dock, 1)

        self._apply_shadow(self._dock)

        root = QVBoxLayout(self._dock)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Draggable header bar
        self.hdr_bar = QFrame(self._dock)
        self.hdr_bar.setObjectName("gpDockHdrBar")
        self.hdr_bar.setAttribute(Qt.WA_StyledBackground, True)
        self.hdr_bar.installEventFilter(self)
        root.addWidget(self.hdr_bar, 0)

        hdr = QHBoxLayout(self.hdr_bar)
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        self.lb_title = QLabel(str(title or ""), self._dock)
        self.lb_title.setObjectName("gpDockTitle")
        hdr.addWidget(self.lb_title, 1)

        # if self._side == "left":
        #     svg = "dock-left.svg"
        # elif self._side == "panel":
        #     svg = "dock-panel.svg"
        # else:
        #     svg = "dock-right.svg"

        self.btn_unpin = QToolButton(self._dock)
        self.btn_unpin.setObjectName("miniAction")
        self.btn_unpin.setAutoRaise(True)
        # self.btn_unpin.setToolTip("Return to dock")
        # self.btn_unpin.setIcon(
        #     _icon_or_std(
        #         self.btn_unpin,
        #         svg,
        #         QStyle.SP_TitleBarNormalButton,
        #     )
        # )
        self.btn_unpin.setToolTip("Dock")
        self.btn_unpin.setIcon(
            _icon_or_std(
                self.btn_unpin,
                "popin.svg",
                QStyle.SP_TitleBarNormalButton,
            )
        )

        self.btn_unpin.clicked.connect(self.request_unpin.emit)
        hdr.addWidget(self.btn_unpin, 0)

        self.btn_close = QToolButton(self._dock)
        self.btn_close.setObjectName("miniAction")
        self.btn_close.setAutoRaise(True)
        self.btn_close.setToolTip("Close")
        self.btn_close.setIcon(
            _icon_or_std(
                self.btn_close,
                "close.svg",
                QStyle.SP_TitleBarCloseButton,
            )
        )
        self.btn_close.clicked.connect(self._on_close_clicked)
        hdr.addWidget(self.btn_close, 0)

        # Body (where the panel gets inserted)
        self._panel_wrap = QWidget(self._dock)
        self._panel_wrap.setObjectName("gpDockBody")
        self._panel_wrap.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self._panel_lay = QVBoxLayout(self._panel_wrap)
        self._panel_lay.setContentsMargins(0, 0, 0, 0)
        self._panel_lay.setSpacing(0)

        root.addWidget(self._panel_wrap, 1)

        # Bottom-right resize grip (like Xfer)
        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        foot.addStretch(1)

        grip = QSizeGrip(self._dock)
        grip.setObjectName("gpDockWinGrip")
        foot.addWidget(grip, 0)

        root.addLayout(foot, 0)

    def _on_close_clicked(self) -> None:
        self.request_close.emit()
        self.close()
    
    def _apply_shadow(self, w: QWidget) -> None:
        eff = QGraphicsDropShadowEffect(w)
        eff.setBlurRadius(28)
        eff.setOffset(0, 10)
        eff.setColor(QColor(0, 0, 0, 90))
        w.setGraphicsEffect(eff)

    def eventFilter(self, obj, ev) -> bool:
        if obj is getattr(self, "hdr_bar", None):
            et = ev.type()

            if (
                et == QEvent.MouseButtonPress
                and ev.button() == Qt.LeftButton
            ):
                child = self.hdr_bar.childAt(ev.pos())
                if child in (self.btn_unpin, self.btn_close):
                    return False
                self._drag_off = (
                    ev.globalPos()
                    - self.frameGeometry().topLeft()
                )
                return True

            if (
                et == QEvent.MouseMove
                and self._drag_off is not None
                and (ev.buttons() & Qt.LeftButton)
            ):
                self.move(ev.globalPos() - self._drag_off)
                return True

            if et == QEvent.MouseButtonRelease:
                self._drag_off = None
                return True

        return super().eventFilter(obj, ev)

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

        if self._side == "panel":
            want = want.expandedTo(
                QSize(
                    int(geo.width() * 0.60),
                    int(geo.height() * 0.52),
                )
            )

        want = QSize(
            min(want.width(), int(geo.width() * 0.95)),
            min(want.height(), int(geo.height() * 0.95)),
        )

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
        # Close means close (NOT attach).
        self.request_close.emit()
        super().closeEvent(ev)