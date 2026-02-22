# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import (
    Qt,
    QSize,
    QTimer,
    pyqtSignal,
    QPropertyAnimation,
    QEasingCurve,
)
from PyQt5.QtGui import QRegion
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from ....config.store import GeoConfigStore
from ...icon_utils import try_icon
from ..keys import (
    BASEMAP_CHOICES,
    K_MAP_BASEMAP,
    basemap_icon_name
)

K_BM_QUICK_OPEN = "xfer.map.basemap.quick_open"
K_BM_QUICK_PIN = "xfer.map.basemap.quick_pinned"
K_BM_QUICK_ANCH = "xfer.map.basemap.quick_anchor"


class XferBasemapQuickOverlay(QWidget):
    request_open_advanced = pyqtSignal()

    def __init__(
        self,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store

        self._dock = QFrame(self)
        self._dock.setObjectName("xferBasemapDock")
        
        self._setup_fade()

        self._open = bool(self._s.get(K_BM_QUICK_OPEN, False))
        self._pin = bool(self._s.get(K_BM_QUICK_PIN, False))
        self._anch = str(self._s.get(K_BM_QUICK_ANCH, "bl"))

        self._build_ui()
        self._wire()

        self._sync_from_store()
        QTimer.singleShot(0, self._reposition)
        QTimer.singleShot(0, self._schedule_idle_fade)


    def _build_ui(self) -> None:
        st = self.style()
        ico_layers = try_icon(
            "layers.svg",
            fallback=st.standardIcon(
                QStyle.SP_FileDialogListView
            ),
        )
        ico_pin = try_icon(
            "pin.svg",
            fallback=st.standardIcon(
                QStyle.SP_DialogYesButton
            ),
        )
        ico_gear = try_icon(
            "gear.svg",
            fallback=st.standardIcon(
                QStyle.SP_FileDialogDetailedView
            ),
        )
    
        # ensure QSS background renders
        self._dock.setAttribute(
            Qt.WA_StyledBackground, True
        )
    
        root = QVBoxLayout(self._dock)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
    
        head = QHBoxLayout()
        head.setSpacing(6)
    
        self.btn_toggle = QToolButton(self._dock)
        self.btn_toggle.setObjectName(
            "xferBasemapBtnToggle"
        )
        self.btn_toggle.setIcon(ico_layers)
        self.btn_toggle.setAutoRaise(True)
        self.btn_toggle.setToolTip("Basemap")
    
        self.btn_pin = QToolButton(self._dock)
        self.btn_pin.setObjectName(
            "xferBasemapBtnPin"
        )
        self.btn_pin.setIcon(ico_pin)
        self.btn_pin.setCheckable(True)
        self.btn_pin.setAutoRaise(True)
        self.btn_pin.setToolTip("Pin / unpin")
    
        self.btn_more = QToolButton(self._dock)
        self.btn_more.setObjectName(
            "xferBasemapBtnMore"
        )
        self.btn_more.setIcon(ico_gear)
        self.btn_more.setAutoRaise(True)
        self.btn_more.setToolTip("Advanced view…")
    
        head.addWidget(self.btn_toggle)
        head.addWidget(self.btn_pin)
        head.addWidget(self.btn_more)
        head.addStretch(1)
        root.addLayout(head)
    
        self._list = QWidget(self._dock)
        self._list.setObjectName("xferBasemapList")
    
        lst = QVBoxLayout(self._list)
        lst.setContentsMargins(0, 0, 0, 0)
        lst.setSpacing(2)
    
        self._bg = QButtonGroup(self._dock)
        self._bg.setExclusive(True)
    
        # for bid, label in BASEMAP_CHOICES:
        #     b = QToolButton(self._list)
        #     b.setText(label)
        #     b.setCheckable(True)
        #     b.setToolButtonStyle(
        #         Qt.ToolButtonTextBesideIcon
        #     )
        #     b.setAutoRaise(True)
        #     b.setProperty("bm_id", bid)
    
        #     # QSS hook for list items
        #     b.setProperty("bmItem", True)
    
        #     self._bg.addButton(b)
        #     lst.addWidget(b)

        for bid, label in BASEMAP_CHOICES:
            b = QToolButton(self._list)
            b.setText(label)
        
            b.setIcon(try_icon(basemap_icon_name(bid)))
            b.setIconSize(QSize(18, 18))
            b.setToolTip(label)
        
            b.setCheckable(True)
            b.setToolButtonStyle(
                Qt.ToolButtonTextBesideIcon
            )
            b.setAutoRaise(True)
            b.setProperty("bm_id", bid)
            b.setProperty("bmItem", True)
        
            self._bg.addButton(b)
            lst.addWidget(b)
    
        root.addWidget(self._list)
    
        self._apply_open(self._open or self._pin)

    def _wire(self) -> None:
        self.btn_toggle.clicked.connect(self._on_toggle)
        self.btn_pin.toggled.connect(self._on_pin)
        self.btn_more.clicked.connect(self.request_open_advanced)

        self._bg.buttonClicked.connect(self._on_pick)
        self._s.config_changed.connect(self._on_store)

    def _setup_fade(self) -> None:
        # Tunables (feel free to adjust)
        self._idle_opacity = 0.30
        self._idle_delay_ms = 2200
        self._fade_ms = 700
        self._hovering = False

        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._fade_to_idle)

        self._op_eff = QGraphicsOpacityEffect(self._dock)
        self._dock.setGraphicsEffect(self._op_eff)
        self._op_eff.setOpacity(1.0)

        self._anim = QPropertyAnimation(
            self._op_eff,
            b"opacity",
            self,
        )
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.setDuration(self._fade_ms)

    def _set_opacity(self, a: float, *, animate: bool = True) -> None:
        a = float(max(0.05, min(1.0, a)))

        if not animate:
            self._anim.stop()
            self._op_eff.setOpacity(a)
            return

        self._anim.stop()
        self._anim.setStartValue(float(self._op_eff.opacity()))
        self._anim.setEndValue(a)
        self._anim.start()

    def _schedule_idle_fade(self) -> None:
        # Never fade while expanded or pinned
        if self._pin or self._open or self._hovering:
            self._idle_timer.stop()
            self._set_opacity(1.0, animate=False)
            return

        self._idle_timer.start(self._idle_delay_ms)

    def _fade_to_idle(self) -> None:
        if self._pin or self._open or self._hovering:
            return
        self._set_opacity(self._idle_opacity, animate=True)

    def enterEvent(self, ev) -> None:
        super().enterEvent(ev)
        self._hovering = True
        self._idle_timer.stop()
        self._set_opacity(1.0, animate=True)

    def leaveEvent(self, ev) -> None:
        super().leaveEvent(ev)
        self._hovering = False
        self._schedule_idle_fade()
        
    def _on_toggle(self) -> None:
        if self._pin:
            self._open = True
        else:
            self._open = not self._open
        self._s.set(K_BM_QUICK_OPEN, self._open)
        self._apply_open(self._open)
        self._reposition()

    def _on_pin(self, on: bool) -> None:
        self._pin = bool(on)
        self._s.set(K_BM_QUICK_PIN, self._pin)
        if self._pin:
            self._open = True
            self._s.set(K_BM_QUICK_OPEN, True)
        self._apply_open(self._open or self._pin)
        self._reposition()

    def _on_pick(self, btn: QToolButton) -> None:
        bid = str(btn.property("bm_id") or "")
        if bid:
            self._s.set(K_MAP_BASEMAP, bid)

        if not self._pin:
            self._open = False
            self._s.set(K_BM_QUICK_OPEN, False)
            self._apply_open(False)
            self._reposition()

    def _on_store(self, keys: object) -> None:
        self._sync_from_store()

    # def _sync_from_store(self) -> None:
    #     cur = str(self._s.get(K_MAP_BASEMAP, "osm") or "osm")
    #     for b in self._bg.buttons():
    #         bid = str(b.property("bm_id") or "")
    #         b.setChecked(bid == cur)

    #     self.btn_pin.setChecked(bool(self._s.get(K_BM_QUICK_PIN, False)))

    def _sync_from_store(self) -> None:
        cur = str(self._s.get(K_MAP_BASEMAP, "osm") or "osm")
    
        for b in self._bg.buttons():
            bid = str(b.property("bm_id") or "")
            b.setChecked(bid == cur)
    
        # sync flags from store
        self._open = bool(self._s.get(K_BM_QUICK_OPEN, False))
        self._pin = bool(self._s.get(K_BM_QUICK_PIN, False))
        self._anch = str(self._s.get(K_BM_QUICK_ANCH, "bl") or "bl")
    
        # avoid feedback loops
        self.btn_pin.blockSignals(True)
        self.btn_pin.setChecked(self._pin)
        self.btn_pin.blockSignals(False)
    
        self._apply_open(self._open or self._pin)
        self._reposition()
    
    # def _apply_open(self, on: bool) -> None:
    #     self._list.setVisible(bool(on))

    def _apply_open(self, on: bool) -> None:
        self._list.setVisible(bool(on))

        # When list is open (or pinned), keep fully visible.
        if bool(on) or self._pin:
            self._idle_timer.stop()
            self._set_opacity(1.0, animate=True)
        else:
            self._schedule_idle_fade()
            
    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._reposition()

    def _reposition(self) -> None:
        if self.width() < 2 or self.height() < 2:
            self.clearMask()
            return
        self._dock.adjustSize()

        m = 12
        w = self.width()
        h = self.height()
        dw = self._dock.width()
        dh = self._dock.height()

        anch = str(self._s.get(K_BM_QUICK_ANCH, "bl") or "bl")
        self._anch = anch

        if anch == "br":
            x = max(m, w - dw - m)
        else:
            x = m
        y = max(m, h - dh - m)

        self._dock.move(x, y)
        self._dock.raise_()
        self.setMask(QRegion(self._dock.geometry()))