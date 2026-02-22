# geoprior/ui/map/docks.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
from typing import Dict, Optional

from functools import partial

from PyQt5.QtCore import (
    Qt,
    QSignalBlocker,
    pyqtSignal,
)

from PyQt5.QtWidgets import (
    QAction,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QTabWidget,
    QToolButton,
    QWidget,
)


_DOCK_STATE_KEY = "map.docks.state"


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


class DockHeader(QWidget):
    search_changed = pyqtSignal(str)
    preset_selected = pyqtSignal(str)
    float_toggled = pyqtSignal(bool)
    close_clicked = pyqtSignal()

    def __init__(
        self,
        title: str,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._title = str(title or "")
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(6)

        self.lb = QLabel(self._title, self)
        self.lb.setObjectName("dockTitle")

        self.search = QLineEdit(self)
        self.search.setPlaceholderText(
            "Search setting..."
        )
        self.search.textChanged.connect(
            self.search_changed.emit
        )

        self.btn_preset = QToolButton(self)
        self.btn_preset.setText("⋯")
        self.btn_preset.setToolTip("Presets")
        self.btn_preset.setPopupMode(
            QToolButton.InstantPopup
        )

        menu = QMenu(self.btn_preset)
        for name in ("Minimal", "Standard", "Debug"):
            act = QAction(name, menu)
            act.triggered.connect(
                lambda _=False, n=name:
                self.preset_selected.emit(n)
            )
            menu.addAction(act)

        self.btn_preset.setMenu(menu)

        self.btn_float = QToolButton(self)
        self.btn_float.setText("⧉")
        self.btn_float.setToolTip("Float / Dock")
        self.btn_float.setCheckable(True)
        self.btn_float.toggled.connect(
            self.float_toggled.emit
        )

        self.btn_close = QToolButton(self)
        self.btn_close.setText("✕")
        self.btn_close.setToolTip("Close")
        self.btn_close.clicked.connect(
            self.close_clicked.emit
        )

        lay.addWidget(self.lb, 0)
        lay.addWidget(self.search, 1)
        lay.addWidget(self.btn_preset, 0)
        lay.addWidget(self.btn_float, 0)
        lay.addWidget(self.btn_close, 0)


class ToolDock(QDockWidget):
    def __init__(
        self,
        key: str,
        title: str,
        tool: QWidget,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(title, parent)

        self.key = str(key or "")
        self.tool = tool

        self.setObjectName(f"dock_{self.key}")
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        self.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )

        self.header = DockHeader(title, parent=self)
        self.setTitleBarWidget(self.header)
        self.setWidget(tool)

        self.header.float_toggled.connect(
            self.setFloating
        )
        self.header.close_clicked.connect(
            self.hide
        )

        self.topLevelChanged.connect(
            self._sync_float_btn
        )

    def _sync_float_btn(self, on: bool) -> None:
        b = self.header.btn_float
        with QSignalBlocker(b):
            b.setChecked(bool(on))


class DockHost(QMainWindow):
    def __init__(
        self,
        *,
        store,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._s = store
        self._docks: Dict[str, ToolDock] = {}

        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.AnimatedDocks
            | QMainWindow.AllowTabbedDocks
        )

        self.setTabPosition(
            Qt.LeftDockWidgetArea,
            QTabWidget.North,
        )
        self.setTabPosition(
            Qt.RightDockWidgetArea,
            QTabWidget.North,
        )

    def _on_layout_changed(self, *_a) -> None:
        self.save_layout()

    def add_tool(
        self,
        *,
        key: str,
        title: str,
        tool: QWidget,
        area: Qt.DockWidgetArea,
    ) -> ToolDock:
        dock = ToolDock(
            key=key,
            title=title,
            tool=tool,
            parent=self,
        )

        self._docks[key] = dock
        self.addDockWidget(area, dock)

        dock.header.preset_selected.connect(
            self.apply_preset
        )
        dock.header.search_changed.connect(
            partial(self._route_search, key)
        )

        dock.topLevelChanged.connect(
            self._on_layout_changed
        )
        dock.visibilityChanged.connect(
            self._on_layout_changed
        )

        return dock

    def restore_layout(self) -> bool:
        raw = self._s.get(_DOCK_STATE_KEY, "")
        if not raw:
            return False
        try:
            self.restoreState(_b64d(raw))
        except Exception:
            return False
        return True


    def toggle_tool(self, key: str) -> None:
        d = self._docks.get(key)
        if d is None:
            return
    
        if d.isVisible():
            d.hide()
            return
    
        d.show()
        d.raise_()


    def apply_preset(self, name: str) -> None:
        n = str(name or "").lower()

        if n == "minimal":
            self._set_vis(data=True, view=False, ana=False)
            return

        if n == "debug":
            self._set_vis(data=True, view=True, ana=True)
            return

        self._set_vis(data=True, view=True, ana=False)

    def _set_vis(
        self,
        *,
        data: bool,
        view: bool,
        ana: bool,
    ) -> None:
        self._set_one("data", data)
        self._set_one("view", view)
        self._set_one("analytics", ana)

    def _set_one(self, key: str, vis: bool) -> None:
        d = self._docks.get(key)
        if d is not None:
            d.setVisible(bool(vis))

    def _route_search(self, key: str, text: str) -> None:
        d = self._docks.get(key)
        if d is None:
            return

        tool = d.tool
        fn = getattr(tool, "apply_search", None)
        if callable(fn):
            fn(str(text or ""))

    def save_layout(self) -> None:
        try:
            raw = _b64e(self.saveState())
            self._s.set(_DOCK_STATE_KEY, raw)
        except Exception:
            return
