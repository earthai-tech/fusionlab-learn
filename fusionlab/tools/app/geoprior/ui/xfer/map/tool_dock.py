# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.map.tool_dock

Dockable container for the existing XferMapToolbar.

Rules:
- Docked: show only "basic/init" section.
- Floating: reveal "advanced" section (collapsible).
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
)
from ....config.store import GeoConfigStore
from .advanced import XferMapAdvancedPanel
from .toolbar import XferMapToolbar


__all__ = ["XferMapToolDock"]

class _DockTitleBar(QWidget):
    def __init__(
        self,
        dock: QDockWidget,
        title: str,
    ) -> None:
        super().__init__(dock)
        self._dock = dock
        self.setObjectName("dockTitleBar")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 8, 6)
        lay.setSpacing(8)

        self.lbl = QLabel(title, self)
        self.lbl.setObjectName("dockTitle")
        lay.addWidget(self.lbl, 0)

        self.chip = QLabel("Docked", self)
        self.chip.setObjectName("dockChip")
        self.chip.setProperty("kind", "info")
        lay.addWidget(self.chip, 0)

        lay.addStretch(1)

        self.btn_float = QToolButton(self)
        self.btn_float.setObjectName("dockBtn")
        self.btn_float.setAutoRaise(True)
        self.btn_float.clicked.connect(self._toggle_float)
        lay.addWidget(self.btn_float, 0)

        self.btn_close = QToolButton(self)
        self.btn_close.setObjectName("dockBtn")
        self.btn_close.setAutoRaise(True)
        self.btn_close.clicked.connect(self._close)
        lay.addWidget(self.btn_close, 0)

        self._sync_icons()

    def set_state(self, floating: bool) -> None:
        if floating:
            self.chip.setText("Floating")
            self.chip.setProperty("kind", "ok")
        else:
            self.chip.setText("Docked")
            self.chip.setProperty("kind", "info")
        self._sync_icons()
        self._repolish(self.chip)

    def _toggle_float(self) -> None:
        self._dock.setFloating(not self._dock.isFloating())

    def _close(self) -> None:
        self._dock.setVisible(False)

    def _sync_icons(self) -> None:
        st = self.style()
        if self._dock.isFloating():
            ico = st.standardIcon(QStyle.SP_TitleBarNormalButton)
        else:
            ico = st.standardIcon(QStyle.SP_TitleBarMaxButton)
        self.btn_float.setIcon(ico)
        self.btn_close.setIcon(
            st.standardIcon(QStyle.SP_TitleBarCloseButton)
        )

    @staticmethod
    def _repolish(w: QWidget) -> None:
        s = w.style()
        s.unpolish(w)
        s.polish(w)
        w.update()

class _DockBody(QWidget):
    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:

        super().__init__(parent)
        self.setObjectName("dockBody")
        self._s = store
        self._adv_panel: Optional[QWidget] = None
        

        self._tb: Optional[XferMapToolbar] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # -------------------------
        # Basic / Init (always)
        # -------------------------
        self.basic = QFrame(self)
        self.basic.setObjectName("mapToolDockBasic")
        self.basic.setFrameShape(QFrame.NoFrame)

        self._basic_l = QVBoxLayout(self.basic)
        self._basic_l.setContentsMargins(10, 8, 10, 8)
        self._basic_l.setSpacing(0)

        root.addWidget(self.basic, 0)

        # -------------------------
        # Advanced (floating only)
        # -------------------------
        self.adv = QFrame(self)
        self.adv.setObjectName("mapToolDockAdv")
        self.adv.setFrameShape(QFrame.NoFrame)

        adv_l = QVBoxLayout(self.adv)
        adv_l.setContentsMargins(10, 6, 10, 8)
        adv_l.setSpacing(6)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        self.btn_adv = QToolButton(self.adv)
        self.btn_adv.setObjectName("miniAction")
        self.btn_adv.setText("Advanced")
        self.btn_adv.setCheckable(True)
        self.btn_adv.setChecked(False)
        self.btn_adv.setAutoRaise(True)

        hdr.addWidget(self.btn_adv, 0)
        hdr.addWidget(QLabel("", self.adv), 1)

        adv_l.addLayout(hdr)

        self.adv_body = QFrame(self.adv)
        self.adv_body.setObjectName("mapToolDockAdvBody")
        self.adv_body.setFrameShape(QFrame.NoFrame)

        body_l = QVBoxLayout(self.adv_body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(6)

        self._adv_panel = XferMapAdvancedPanel(
            store=self._s,
            parent=self.adv_body,
        )
        body_l.addWidget(self._adv_panel)


        adv_l.addWidget(self.adv_body)

        self.btn_adv.toggled.connect(self.adv_body.setVisible)
        self.adv_body.setVisible(False)

        root.addWidget(self.adv, 0)

        # Hidden by default (dock mode)
        self.adv.setVisible(False)

    def set_toolbar(self, tb: XferMapToolbar) -> None:
        if tb is None:
            return
        if self._tb is tb:
            return

        old = self._tb
        if old is not None:
            self._basic_l.removeWidget(old)
            old.setParent(None)

        self._tb = tb
        tb.setParent(self.basic)
        tb.setVisible(True)
        self._basic_l.addWidget(tb)

    def toolbar(self) -> Optional[XferMapToolbar]:
        return self._tb

    def set_advanced_available(self, on: bool) -> None:
        on = bool(on)
        if not on:
            self.btn_adv.setChecked(False)
            self.adv_body.setVisible(False)
        self.adv.setVisible(on)


class XferMapToolDock(QDockWidget):
    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:

        super().__init__("Map tools", parent)
        
        self.setAllowedAreas(
            Qt.TopDockWidgetArea
            | Qt.LeftDockWidgetArea
            | Qt.RightDockWidgetArea
        )

        self.setObjectName("mapToolDock")
        self.setProperty("gpDock", True)

        self.setAllowedAreas(Qt.TopDockWidgetArea)

        self.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        self._title = _DockTitleBar(self, "Map tools")
        self.setTitleBarWidget(self._title)
        
        self._body = _DockBody(store=store, parent=self)
        self.setWidget(self._body)

        self.topLevelChanged.connect(self._on_top_level)
        self._update_docked_height()

    def set_toolbar(self, tb: Optional[XferMapToolbar]) -> None:
        self._body.set_toolbar(tb)
        self._update_docked_height()

    def _on_top_level(self, floating: bool) -> None:
        floating = bool(floating)
        self._body.set_advanced_available(floating)
        self._title.set_state(floating)
        self._update_docked_height()

    def _update_docked_height(self) -> None:
        if not self.isVisible() and not self.isFloating():
            self.setMaximumHeight(16777215)
            return

        tbw = self.titleBarWidget()
        title_h = tbw.sizeHint().height() if tbw else 28
    
        basic_h = self._body.basic.sizeHint().height()
        tb = self._body.toolbar()
        if tb is not None:
            basic_h = max(basic_h, tb.sizeHint().height() + 16)
    
        h = int(max(60, title_h + basic_h + 12))
        self.setMinimumHeight(h)
    
        if self.isFloating():
            self.setMaximumHeight(16777215)
        else:
            # If hints aren't ready yet, don't pin to "title-only" height.
            if basic_h < 40:
                self.setMaximumHeight(16777215)
            else:
                self.setMaximumHeight(h)


