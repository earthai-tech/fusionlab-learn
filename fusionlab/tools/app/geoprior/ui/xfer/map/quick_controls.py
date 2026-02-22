# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""geoprior.ui.xfer.map.quick_controls

Floating overlays for the Xfer map page:

- XferMapQuickBarOverlay: always-visible 1-row shortcuts.
- XferMapControlsDrawer: a floating drawer hosting the full
  XferMapToolbar.

Both widgets:
- sit above the Leaflet canvas (MapView.add_overlay)
- use QRegion masking so only their geometry blocks
  mouse events (map remains interactive elsewhere)
- rely on GeoConfigStore for state

This module is UI-only.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import (
    Qt,
    QEvent,
    QTimer,
    pyqtSignal,
    QPropertyAnimation,
    QEasingCurve,
    QPoint,
    QSize
)
from PyQt5.QtGui import QRegion, QColor
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QGraphicsDropShadowEffect,
    QSizeGrip,
)

from ....config.store import GeoConfigStore
from ...icon_utils import try_icon
from ..keys import (
    K_MAP_OVERLAY,
    K_MAP_SPLIT,
    K_MAP_STEP,
    K_MAP_PLAY,
    K_MAP_STEP_MAX,
    K_MAP_STEP_MODE,
    K_MAP_YEAR0,
)

__all__ = [
    "K_CTL_OPEN",
    "K_CTL_PIN",
    "K_CTL_ANCH",
    "K_CTL_WIN_GEOM",
    "K_QB_PIN",
    "K_QB_ANCH",
    "XferMapQuickBarOverlay",
    "XferMapControlsDrawer",
    "XferMapControlsWindow",

]


# Drawer state keys (GUI-only)
K_CTL_OPEN = "xfer.map.controls.open"
K_CTL_PIN = "xfer.map.controls.pinned"
K_CTL_ANCH = "xfer.map.controls.anchor"  # tl|tc|tr

# Quickbar state keys (GUI-only)
K_QB_PIN = "xfer.map.quickbar.pinned"
K_QB_ANCH = "xfer.map.quickbar.anchor"  # tl|tc|tr
K_QB_BOTTOM = "xfer.map.quickbar.bottom"

K_CTL_FREE = "xfer.map.controls.free"
K_CTL_X = "xfer.map.controls.x"
K_CTL_Y = "xfer.map.controls.y"

K_QB_SAFE_LEFT = "xfer.map.quickbar.safe_left"
K_CTL_WIN_GEOM = "xfer.map.controls.win_geom"

def _std(st: QStyle, sp: QStyle.StandardPixmap):
    return st.standardIcon(sp)


class XferMapQuickBarOverlay(QWidget):
    """Always-visible 1-row shortcut bar."""

    request_open_advanced = pyqtSignal()
    request_fit = pyqtSignal()

    def __init__(
        self,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        # self._drag_on = False
        # self._drag_off = QPoint(0, 0)

        self._dock = QFrame(self)
        self._dock.setObjectName("xferQuickBarDock")
        self._dock.setAttribute(
            Qt.WA_StyledBackground, True
        )

        self._hovering = False
        self._idle_opacity = 0.35
        self._idle_delay_ms = 1800
        self._fade_ms = 650

        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(
            self._fade_to_idle
        )

        self._op_eff = QGraphicsOpacityEffect(
            self._dock
        )
        self._dock.setGraphicsEffect(self._op_eff)
        self._op_eff.setOpacity(1.0)

        self._anim = QPropertyAnimation(
            self._op_eff,
            b"opacity",
            self,
        )
        self._anim.setDuration(self._fade_ms)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._play_timer = QTimer(self)
        self._play_timer.setSingleShot(False)
        self._play_timer.setInterval(850)
        self._play_timer.timeout.connect(self._tick_play)

        self._build_ui()
        self._wire()
        self._sync_from_store()
        QTimer.singleShot(0, self._reposition)
        QTimer.singleShot(0, self._schedule_idle_fade)

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        st = self.style()

        ico_ctl = try_icon(
            "controls.svg",
            fallback=_std(
                st,
                QStyle.SP_FileDialogDetailedView,
            ),
        )
        ico_gear = try_icon(
            "gear.svg",
            fallback=_std(
                st,
                QStyle.SP_FileDialogDetailedView,
            ),
        )
        ico_fit = try_icon(
            "fit.svg",
            fallback=_std(
                st,
                QStyle.SP_DialogResetButton,
            ),
        )
        ico_pin = try_icon(
            "pin.svg",
            fallback=_std(
                st,
                QStyle.SP_DialogYesButton,
            ),
        )
        ico_prev = try_icon(
            "prev.svg",
            fallback=_std(st, QStyle.SP_ArrowBack),
        )
        ico_next = try_icon(
            "next.svg",
            fallback=_std(
                st,
                QStyle.SP_ArrowForward,
            ),
        )

        ico_play = try_icon(
            "play.svg",
            fallback=try_icon(
                "play-pause.svg",
                fallback=_std(st, QStyle.SP_MediaPlay),
            ),
        )
        ico_pause = try_icon(
            "pause.svg",
            fallback=try_icon(
                "play-pause.svg",
                fallback=_std(st, QStyle.SP_MediaPause),
            ),
        )

        self._ico_play = ico_play
        self._ico_pause = ico_pause

        root = QHBoxLayout(self._dock)
        root.setContentsMargins(10, 5, 10, 5)
        root.setSpacing(6)

        self.btn_pin = QToolButton(self._dock)
        self.btn_pin.setObjectName("xferQuickPin")
        self.btn_pin.setIcon(ico_pin)
        self.btn_pin.setCheckable(True)
        self.btn_pin.setAutoRaise(True)
        self.btn_pin.setToolTip("Pin quick bar")
        root.addWidget(self.btn_pin, 0)

        self.btn_cities = QToolButton(self._dock)
        self.btn_cities.setObjectName("xferQuickCities")
        self.btn_cities.setAutoRaise(True)
        self.btn_cities.setToolTip("Open controls")
        root.addWidget(self.btn_cities, 0)

        root.addSpacing(6)

        self._seg_split = QButtonGroup(self)
        self._seg_split.setExclusive(True)
        
        self.btn_val = self._mk_seg_btn("Val", "l")
        self.btn_test = self._mk_seg_btn("Test", "r")
        
        self._seg_split.addButton(self.btn_val)
        self._seg_split.addButton(self.btn_test)
        
        self.btn_val.setProperty("segValue", "val")
        self.btn_test.setProperty("segValue", "test")
        
        root.addWidget(self._mk_seg_group(
            self.btn_val,
            self.btn_test,
        ), 0)
        
        root.addSpacing(6)

        self._seg_ovl = QButtonGroup(self)
        self._seg_ovl.setExclusive(True)
        
        self.btn_ov_a = self._mk_seg_btn("A", "l")
        self.btn_ov_ab = self._mk_seg_btn("A+B", "m")
        self.btn_ov_b = self._mk_seg_btn("B", "r")
        
        for b, v in (
            (self.btn_ov_a, "a"),
            (self.btn_ov_ab, "both"),
            (self.btn_ov_b, "b"),
        ):
            self._seg_ovl.addButton(b)
            b.setProperty("segValue", v)
        
        root.addWidget(self._mk_seg_group(
            self.btn_ov_a,
            self.btn_ov_ab,
            self.btn_ov_b,
        ), 0)

        root.addStretch(1)

        self.btn_prev = QToolButton(self._dock)
        self.btn_prev.setObjectName("xferQuickNav")
        self.btn_prev.setIcon(ico_prev)
        self.btn_prev.setAutoRaise(True)
        self.btn_prev.setToolTip("Previous step")
        root.addWidget(self.btn_prev, 0)

        self.btn_play = QToolButton(self._dock)
        self.btn_play.setObjectName("xferQuickPlay")
        self.btn_play.setCheckable(True)
        self.btn_play.setAutoRaise(True)
        self.btn_play.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn_play.setIcon(self._ico_play)
        self.btn_play.setText("Step 1")
        self.btn_play.setToolTip("Play")
        root.addWidget(self.btn_play, 0)

        self.btn_next = QToolButton(self._dock)
        self.btn_next.setObjectName("xferQuickNav")
        self.btn_next.setIcon(ico_next)
        self.btn_next.setAutoRaise(True)
        self.btn_next.setToolTip("Next step")
        root.addWidget(self.btn_next, 0)

        root.addSpacing(8)

        self.btn_fit = QToolButton(self._dock)
        self.btn_fit.setObjectName("xferQuickAction")
        self.btn_fit.setIcon(ico_fit)
        self.btn_fit.setAutoRaise(True)
        self.btn_fit.setToolTip("Fit layers")
        root.addWidget(self.btn_fit, 0)

        self.btn_adv = QToolButton(self._dock)
        self.btn_adv.setObjectName("xferQuickAction")
        self.btn_adv.setIcon(ico_gear)
        self.btn_adv.setAutoRaise(True)
        self.btn_adv.setToolTip("Advanced")
        root.addWidget(self.btn_adv, 0)

        self.btn_controls = QToolButton(self._dock)
        self.btn_controls.setObjectName("xferQuickAction")
        self.btn_controls.setIcon(ico_ctl)
        self.btn_controls.setAutoRaise(True)
        self.btn_controls.setToolTip("Controls")
        root.addWidget(self.btn_controls, 0)
        
        sz = QSize(14, 14)
        for b in (
            self.btn_pin,
            self.btn_prev,
            self.btn_next,
            self.btn_play,
            self.btn_fit,
            self.btn_adv,
            self.btn_controls,
        ):
            b.setIconSize(sz)

    def _mk_seg_btn(
        self,
        text: str,
        pos: str,
    ) -> QToolButton:
        b = QToolButton(self._dock)
        b.setText(str(text))
        b.setCheckable(True)
        b.setAutoRaise(True)
        b.setToolButtonStyle(Qt.ToolButtonTextOnly)
        b.setProperty("seg", True)
        b.setProperty("segPos", str(pos))
        return b

    def _mk_seg_group(
        self,
        *btns: QToolButton,
    ) -> QWidget:
        w = QWidget(self._dock)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        for b in btns:
            lay.addWidget(b)
        return w

    def _wire(self) -> None:
        self._s.config_changed.connect(self._on_store)

        self.btn_controls.clicked.connect(
            self._toggle_controls
        )
        self.btn_cities.clicked.connect(
            self._toggle_controls
        )
        self.btn_adv.clicked.connect(
            self.request_open_advanced.emit
        )
        self.btn_fit.clicked.connect(self.request_fit.emit)

        self.btn_pin.toggled.connect(self._on_pin)
        self._seg_split.buttonClicked.connect(
            self._on_split
        )
        self._seg_ovl.buttonClicked.connect(
            self._on_overlay
        )

        self.btn_prev.clicked.connect(
            lambda: self._nudge_step(-1)
        )
        self.btn_next.clicked.connect(
            lambda: self._nudge_step(+1)
        )
        self.btn_play.toggled.connect(
            self._on_play
        )

    # -------------------------------------------------
    # Actions
    # -------------------------------------------------
    def _toggle_controls(self) -> None:
        on = bool(self._s.get(K_CTL_OPEN, False))
        self._s.set(K_CTL_OPEN, not on)

    def _on_pin(self, on: bool) -> None:
        self._s.set(K_QB_PIN, bool(on))
        self._schedule_idle_fade()

    def _on_split(self, b: QToolButton) -> None:
        v = str(b.property("segValue") or "val")
        self._s.set(K_MAP_SPLIT, v)

    def _on_overlay(self, b: QToolButton) -> None:
        v = str(b.property("segValue") or "both")
        self._s.set(K_MAP_OVERLAY, v)

    def _nudge_step(self, d: int) -> None:
        cur = int(self._s.get(K_MAP_STEP, 1) or 1)
        # nxt = max(1, cur + int(d))
        mx = int(self._s.get(K_MAP_STEP_MAX, 0) or 0)
        nxt = int(cur) + int(d)

        if mx > 0:
            if nxt < 1:
                nxt = int(mx)
            if nxt > int(mx):
                nxt = 1
        else:
            nxt = max(1, nxt)

        self._s.set(K_MAP_STEP, nxt)
 
    def _on_play(self, on: bool) -> None:
        self._s.set(K_MAP_PLAY, bool(on))

    def _tick_play(self) -> None:
        if not bool(self._s.get(K_MAP_PLAY, False)):
            if self._play_timer.isActive():
                self._play_timer.stop()
            return
        self._nudge_step(+1)


    # -------------------------------------------------
    # Store sync
    # -------------------------------------------------
    def _on_store(self, _keys: object) -> None:
        self._sync_from_store()

    def _sync_from_store(self) -> None:
        a = str(self._s.get("xfer.city_a", "A") or "A")
        b = str(self._s.get("xfer.city_b", "B") or "B")
        self.btn_cities.setText(f"A:{a}  →  B:{b}")

        split = str(self._s.get(K_MAP_SPLIT, "val") or "val")
        self._set_seg(self._seg_split, split)

        ovl = str(
            self._s.get(K_MAP_OVERLAY, "both")
            or "both"
        )
        self._set_seg(self._seg_ovl, ovl)

        st = int(self._s.get(K_MAP_STEP, 1) or 1)

        mode = str(
            self._s.get(K_MAP_STEP_MODE, "step")
            or "step"
        ).strip().lower()

        if mode == "year":
            y0 = int(self._s.get(K_MAP_YEAR0, 0) or 0)
            if y0 <= 0:
                label = f"Year {st}"
            else:
                label = f"Year {y0 + st - 1}"
        else:
            label = f"Step {st}"

        playing = bool(self._s.get(K_MAP_PLAY, False))

        self.btn_play.blockSignals(True)
        try:
            self.btn_play.setChecked(playing)
        finally:
            self.btn_play.blockSignals(False)

        if playing:
            self.btn_play.setIcon(self._ico_pause)
            self.btn_play.setToolTip("Pause")
            if not self._play_timer.isActive():
                self._play_timer.start()
        else:
            self.btn_play.setIcon(self._ico_play)
            self.btn_play.setToolTip("Play")
            if self._play_timer.isActive():
                self._play_timer.stop()

        self.btn_play.setText(label)

        pin = bool(self._s.get(K_QB_PIN, False))
        self.btn_pin.blockSignals(True)
        try:
            self.btn_pin.setChecked(pin)
        finally:
            self.btn_pin.blockSignals(False)

        QTimer.singleShot(0, self._reposition)
        self._schedule_idle_fade()

    def _set_seg(self, grp: QButtonGroup, v: str) -> None:
        vv = str(v or "").strip().lower()
        for b in grp.buttons():
            bv = str(b.property("segValue") or "")
            b.setChecked(bv == vv)

    # -------------------------------------------------
    # Fade + geometry
    # -------------------------------------------------
    def _set_opacity(self, a: float, *, anim: bool) -> None:
        a = float(max(0.05, min(1.0, a)))
        if not anim:
            self._anim.stop()
            self._op_eff.setOpacity(a)
            return
        self._anim.stop()
        self._anim.setStartValue(
            float(self._op_eff.opacity())
        )
        self._anim.setEndValue(a)
        self._anim.start()

    def _schedule_idle_fade(self) -> None:
        pin = bool(self._s.get(K_QB_PIN, False))
        if pin or self._hovering:
            self._idle_timer.stop()
            self._set_opacity(1.0, anim=False)
            return
        self._idle_timer.start(self._idle_delay_ms)

    def _fade_to_idle(self) -> None:
        pin = bool(self._s.get(K_QB_PIN, False))
        if pin or self._hovering:
            return
        self._set_opacity(self._idle_opacity, anim=True)

    def enterEvent(self, ev) -> None:
        super().enterEvent(ev)
        self._hovering = True
        self._idle_timer.stop()
        self._set_opacity(1.0, anim=True)

    def leaveEvent(self, ev) -> None:
        super().leaveEvent(ev)
        self._hovering = False
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
        dw = self._dock.width()
        anch = str(self._s.get(K_QB_ANCH, "tl") or "tl")
        
        safe = int(self._s.get(K_QB_SAFE_LEFT, 60))
        
        if anch == "tr":
            x = max(m, w - dw - m)
        elif anch == "tc":
            x = max(m, (w - dw) // 2)
        else:
            x = m + safe
        
        # clamp inside viewport
        x = max(m, min(int(x), w - dw - m))
        
        y = m
        self._dock.move(int(x), int(y))

        gap = 10
        bottom = int(y) + int(self._dock.height()) + gap
        
        old = self._s.get(K_QB_BOTTOM, None)
        if old != bottom:
            self._s.set(K_QB_BOTTOM, bottom)

        self._dock.raise_()
        self.setMask(QRegion(self._dock.geometry()))


class XferMapControlsDrawer(QWidget):
    """Floating drawer hosting the full XferMapToolbar."""

    request_close = pyqtSignal()
    request_detach = pyqtSignal()


    def __init__(
        self,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        
        self._drag_on = False
        self._drag_off = QPoint(0, 0)

        self.setObjectName("xferControlsOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._dock = QFrame(self)
        self._dock.setObjectName("xferControlsDock")
        self._dock.setAttribute(
            Qt.WA_StyledBackground, True
        )

        self._build_ui()
        self._wire()

        self.setVisible(False)

    def set_toolbar(self, w: QWidget) -> None:
        if w is None:
            return
        self.scroll.setWidget(w)

    def is_open(self) -> bool:
        return bool(self.isVisible())

    def set_open(self, on: bool) -> None:
        self.setVisible(bool(on))
        if bool(on):
            self.raise_()
            self._dock.raise_()
        QTimer.singleShot(0, self._update_mask)
        QTimer.singleShot(0, self._reposition)

    def eventFilter(self, obj, ev):
        if obj is self.hdr_bar:
            t = ev.type()
    
            if t == QEvent.MouseButtonPress:
                if ev.button() == Qt.LeftButton:
                    self._drag_on = True
                    dock_tl = self._dock.mapToGlobal(
                        QPoint(0, 0)
                    )
                    self._drag_off = (
                        ev.globalPos() - dock_tl
                    )
                    return True
    
            if t == QEvent.MouseMove and self._drag_on:
                parent_tl = self.mapToGlobal(
                    QPoint(0, 0)
                )
                new_tl = ev.globalPos() - self._drag_off
                pos = new_tl - parent_tl
    
                x = int(pos.x())
                y = int(pos.y())
    
                m = 12
                dw = self._dock.width()
                dh = self._dock.height()
                x = max(m, min(x, self.width() - dw - m))
                y = max(m, min(y, self.height() - dh - m))
    
                self._dock.move(x, y)
                self._update_mask()
    
                # persist free position
                self._s.set(K_CTL_FREE, True)
                self._s.set(K_CTL_X, x)
                self._s.set(K_CTL_Y, y)
                return True
    
            if t == QEvent.MouseButtonRelease:
                if ev.button() == Qt.LeftButton:
                    self._drag_on = False
                    return True
    
            if t == QEvent.MouseButtonDblClick:
                self._s.set(K_CTL_FREE, False)
                return True
    
        return super().eventFilter(obj, ev)


    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        st = self.style()

        ico_pin = try_icon(
            "pin.svg",
            fallback=_std(
                st,
                QStyle.SP_DialogYesButton,
            ),
        )
        ico_close = _std(
            st,
            QStyle.SP_TitleBarCloseButton,
        )

        root = QVBoxLayout(self._dock)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- header bar (drag handle) ---
        self.hdr_bar = QFrame(self._dock)
        self.hdr_bar.setObjectName("xferControlsHdrBar")
        self.hdr_bar.setCursor(Qt.SizeAllCursor)
        
        hdr = QHBoxLayout(self.hdr_bar)
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)
        
        title = QLabel("Controls", self._dock)
        title.setObjectName("xferControlsTitle")
        hdr.addWidget(title, 0)
        
        hdr.addStretch(1)
        
        ico_pop = try_icon(
            "popout.svg",
            fallback=_std(st, QStyle.SP_TitleBarMaxButton),
        )
        
        self.btn_pop = QToolButton(self._dock)
        self.btn_pop.setObjectName("miniAction")
        self.btn_pop.setAutoRaise(True)
        self.btn_pop.setIcon(ico_pop)
        self.btn_pop.setToolTip("Pop out")
        hdr.addWidget(self.btn_pop, 0)
        
        self.btn_pin = QToolButton(self._dock)
        self.btn_pin.setObjectName("miniAction")
        self.btn_pin.setAutoRaise(True)
        self.btn_pin.setIcon(ico_pin)
        self.btn_pin.setCheckable(True)
        self.btn_pin.setToolTip("Pin")
        hdr.addWidget(self.btn_pin, 0)
        
        self.btn_close = QToolButton(self._dock)
        self.btn_close.setObjectName("miniAction")
        self.btn_close.setAutoRaise(True)
        self.btn_close.setIcon(ico_close)
        self.btn_close.setToolTip("Close")
        hdr.addWidget(self.btn_close, 0)
        
        root.addWidget(self.hdr_bar, 0)
        
        # must install after hdr_bar exists
        self.hdr_bar.installEventFilter(self)

        self.scroll = QScrollArea(self._dock)
        self.scroll.setObjectName("xferControlsScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self.scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        root.addWidget(self.scroll, 1)

    def _wire(self) -> None:
        self._s.config_changed.connect(self._on_store)
        self.btn_close.clicked.connect(self._on_close)
        self.btn_pin.toggled.connect(self._on_pin)
        self.btn_pop.clicked.connect(self.request_detach.emit)


    # -------------------------------------------------
    # Store sync
    # -------------------------------------------------
    def _on_store(self, _keys: object) -> None:
        on = bool(self._s.get(K_CTL_OPEN, False))
        pin = bool(self._s.get(K_CTL_PIN, False))

        self.btn_pin.blockSignals(True)
        try:
            self.btn_pin.setChecked(pin)
        finally:
            self.btn_pin.blockSignals(False)

        if on != self.is_open():
            self.set_open(on)
        else:
            QTimer.singleShot(0, self._reposition)

    def _on_pin(self, on: bool) -> None:
        self._s.set(K_CTL_PIN, bool(on))

    def _on_close(self) -> None:
        self._s.set(K_CTL_OPEN, False)
        self.request_close.emit()

    # -------------------------------------------------
    # Mask + geometry
    # -------------------------------------------------
    def _update_mask(self) -> None:
        if not self.isVisible():
            self.clearMask()
            return
        r = self._dock.geometry()
        self.setMask(QRegion(r))

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        QTimer.singleShot(0, self._update_mask)
        QTimer.singleShot(0, self._reposition)

    def _reposition(self) -> None:
        if not self.isVisible():
            return
        if self.width() < 2 or self.height() < 2:
            return

        m = 12
        w = self.width()
        h = self.height()
        anch = str(self._s.get(K_CTL_ANCH, "tl") or "tl")

        # Size: cap to viewport
        self._dock.adjustSize()
        dw = min(self._dock.width(), max(320, w - 2 * m))
        self._dock.setFixedWidth(int(dw))

        # Height: keep it compact; scroll handles rest
        max_h = max(220, int(h * 0.42))
        self._dock.setMaximumHeight(max_h)

        dw = self._dock.width()
        dh = self._dock.height()
        
        free = bool(self._s.get(K_CTL_FREE, False))
        if free:
            x = int(self._s.get(K_CTL_X, m) or m)
            y = int(self._s.get(K_CTL_Y, m) or m)
        
            x = max(m, min(x, w - dw - m))
            y = max(m, min(y, h - dh - m))
        
            self._dock.move(int(x), int(y))
            self._dock.raise_()
            self._update_mask()
            return

        if anch == "tr":
            x = max(m, w - dw - m)
        elif anch == "tc":
            x = max(m, (w - dw) // 2)
        else:
            x = m

        qb_bottom = int(self._s.get(K_QB_BOTTOM, m + 46))
        gap = 12
        y = qb_bottom + gap
        
        # Keep inside viewport
        y = max(m, min(y, h - dh - m))

        self._dock.move(int(x), int(y))
        self._dock.raise_()
        self._update_mask()

class XferMapControlsWindow(QDialog):
    """
    Floating tool window for map controls.
    Receives the SAME toolbar instance via reparent.
    """

    request_attach = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._s = store
        self._geom_key = K_CTL_WIN_GEOM
        self._drag_off: Optional[QPoint] = None

        self.setObjectName("xferControlsWindow")

        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
        )

        self.setWindowTitle("Map controls")

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.setMinimumSize(520, 300)
        self.resize(760, 420)

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

        win_title = QLabel("Map controls", self._dock)
        win_title.setObjectName("xferWinTitle")
        hdr.addWidget(win_title, 0)

        hdr.addStretch(1)

        ico_attach = try_icon(
            "popin.svg",
            fallback=_std(
                st,
                QStyle.SP_TitleBarNormalButton,
            ),
        )

        self.btn_attach = QToolButton(self._dock)
        self.btn_attach.setObjectName("miniAction")
        self.btn_attach.setAutoRaise(True)
        self.btn_attach.setToolTip("Return to map")
        self.btn_attach.setIcon(ico_attach)
        self.btn_attach.clicked.connect(
            self.request_attach.emit
        )
        hdr.addWidget(self.btn_attach, 0)

        ico_close = try_icon(
            "close.svg",
            fallback=_std(
                st,
                QStyle.SP_TitleBarCloseButton,
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

    def set_toolbar(self, w: QWidget) -> None:
        if w is None:
            return
        self.scroll.setWidget(w)

    def take_toolbar(self) -> Optional[QWidget]:
        w = self.scroll.widget()
        if w is None:
            return None
        self.scroll.takeWidget()
        w.setParent(None)
        return w

    def closeEvent(self, ev) -> None:
        self._save_geom()
        self.request_attach.emit()
        super().closeEvent(ev)