# geoprior/ui/map/tooltab.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from PyQt5.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QToolButton,
    QWidget,
    QStyle,
)

from ..icon_utils import try_icon


@dataclass(frozen=True)
class ToolSpec:
    key: str
    tooltip: str
    icon_name: str = ""
    fallback_sp: Optional[int] = None
    checkable: bool = False


class MapToolTab(QWidget):
    """
    Top-center hover tooltab for the map.

    - Hover hotzone triggers reveal
    - Slides/fades like modern tooltip bars
    - Pin keeps visible
    """

    triggered = pyqtSignal(str, bool)

    def __init__(
        self,
        parent: QWidget,
        *,
        store=None,
        pinned_key: str = "map.tooltab.pinned",
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._pinned_key = str(pinned_key)

        self._pinned = False
        self._hover = False

        self._btns: Dict[str, QToolButton] = {}

        self._x = 0
        self._y_show = 10
        self._y_hide = -80

        self.setObjectName("mapToolTab")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        self._layout = lay

        eff = QGraphicsOpacityEffect(self)
        eff.setOpacity(0.0)
        self.setGraphicsEffect(eff)
        self._opacity = eff

        self._anim_pos = QPropertyAnimation(self, b"pos")
        self._anim_pos.setDuration(160)
        self._anim_pos.setEasingCurve(QEasingCurve.OutCubic)

        self._anim_op = QPropertyAnimation(eff, b"opacity")
        self._anim_op.setDuration(160)
        self._anim_op.setEasingCurve(QEasingCurve.OutCubic)
        self._anim_op.finished.connect(self._on_anim_done)

        self.hotzone = QWidget(parent)
        self.hotzone.setObjectName("mapToolTabHot")
        self.hotzone.setMouseTracking(True)

        self.hotzone.installEventFilter(self)
        self.installEventFilter(self)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(220)
        self._hide_timer.timeout.connect(self._hide)

        self._build_default_tools()

        if self._s is not None:
            self._pinned = bool(
                self._s.get(self._pinned_key, False)
            )

        self.set_checked("pin", self._pinned)
        self._apply_mode(initial=True)

    # -------------------------
    # Layout / API
    # -------------------------

    def relayout(self, w: int, h: int) -> None:
        bw = max(10, int(self.sizeHint().width()))
        bh = max(10, int(self.sizeHint().height()))

        self._x = max(0, (int(w) - bw) // 2)
        self._y_show = 10
        self._y_hide = -bh - 8

        hz_h = 12
        self.hotzone.setGeometry(self._x, 0, bw, hz_h)
        self.hotzone.raise_()

        if self._pinned or self._hover:
            self.setGeometry(self._x, self._y_show, bw, bh)
        else:
            self.setGeometry(self._x, self._y_hide, bw, bh)
        self.raise_()

    def add_separator(self) -> None:
        sep = QFrame(self)
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setFixedHeight(22)
        self._layout.addWidget(sep, 0)

    def add_tool(
        self,
        spec: ToolSpec,
        *,
        on_trigger: Optional[Callable[[bool], None]] = None,
    ) -> QToolButton:
        key = str(spec.key).strip()

        btn = QToolButton(self)
        btn.setObjectName("miniAction")
        btn.setProperty("role", "mapHead")
        btn.setAutoRaise(True)
        btn.setCheckable(bool(spec.checkable))
        btn.setToolTip(str(spec.tooltip))
        btn.setFixedSize(30, 30)

        ico = self._resolve_icon(spec)
        if not ico.isNull():
            btn.setIcon(ico)

        self._layout.addWidget(btn, 0)
        self._btns[key] = btn

        def _emit(state: bool = True) -> None:
            self.triggered.emit(key, bool(state))
            if on_trigger is not None:
                try:
                    on_trigger(bool(state))
                except Exception:
                    pass

        if btn.isCheckable():
            btn.toggled.connect(_emit)
        else:
            btn.clicked.connect(lambda: _emit(True))

        return btn

    def set_checked(self, key: str, checked: bool) -> None:
        btn = self._btns.get(str(key))
        if btn is None:
            return
        was = btn.blockSignals(True)
        btn.setChecked(bool(checked))
        btn.blockSignals(was)

    # -------------------------
    # Default tools
    # -------------------------

    def _build_default_tools(self) -> None:
        self.add_tool(
            ToolSpec(
                key="pin",
                tooltip="Pin / unpin toolbar",
                icon_name="pin.svg",
                fallback_sp=QStyle.SP_DialogYesButton,
                checkable=True,
            ),
            on_trigger=self._on_pin,
        )
        self.add_separator()

        self.add_tool(
            ToolSpec(
                key="data",
                tooltip="Toggle Data panel",
                icon_name="data-panel.svg",
                fallback_sp=QStyle.SP_FileDialogContentsView,
                checkable=True,
            )
        )
        self.add_tool(
            ToolSpec(
                key="view",
                tooltip="Toggle View panel",
                icon_name="view-panel.svg",
                fallback_sp=QStyle.SP_FileDialogDetailedView,
                checkable=True,
            )
        )
        self.add_tool(
            ToolSpec(
                key="focus",
                tooltip="Focus mode",
                icon_name="focus.svg",
                fallback_sp=QStyle.SP_DialogApplyButton,
                checkable=True,
            )
        )
        self.add_tool(
            ToolSpec(
                key="analytics",
                tooltip="Toggle Analytics",
                icon_name="analytics.svg",
                fallback_sp=QStyle.SP_ComputerIcon,
                checkable=True,
            )
        )

        self.add_separator()
    
        self.add_tool(
            ToolSpec(
                key="select_point",
                tooltip="Select a point",
                icon_name="select-point.svg",
                fallback_sp=getattr(
                    QStyle,
                    "SP_ArrowCursor",
                    QStyle.SP_ArrowUp,
                ),
                checkable=True,
            )
        )
        self.add_tool(
            ToolSpec(
                key="select_group",
                tooltip="Select a group",
                icon_name="select-group.svg",
                fallback_sp=QStyle.SP_FileDialogListView,
                checkable=True,
            )
        )
        self.add_tool(
            ToolSpec(
                key="clear_selection",
                tooltip="Clear selection",
                icon_name="clear-selection.svg",
                fallback_sp=getattr(
                    QStyle,
                    "SP_DialogResetButton",
                    QStyle.SP_BrowserReload,
                ),
                checkable=False,
            )
        )

        self.add_separator()

        self.add_tool(
            ToolSpec(
                key="fit",
                tooltip="Zoom to points",
                icon_name="map_icon.svg",
                fallback_sp=QStyle.SP_ArrowUp,
            )
        )
        self.add_tool(
            ToolSpec(
                key="clear",
                tooltip="Clear map",
                icon_name="clear-map.svg",
                fallback_sp=getattr(
                    QStyle,
                    "SP_TrashIcon",
                    QStyle.SP_DialogDiscardButton,
                ),
            )
        )
        self.add_tool(
            ToolSpec(
                key="reset_xyz",
                tooltip="Reset X/Y/Z mapping",
                icon_name="reset.svg",
                fallback_sp=QStyle.SP_BrowserReload,
            )
        )

    def _resolve_icon(self, spec: ToolSpec) -> QIcon:
        fb = QIcon()
        if spec.fallback_sp is not None:
            fb = self.style().standardIcon(int(spec.fallback_sp))
        return try_icon(spec.icon_name, fallback=fb, size=18)

    # -------------------------
    # Hover logic + animation
    # -------------------------
    def eventFilter(self, obj, ev) -> bool:
        t = ev.type()
        if obj in (self, self.hotzone):
            if t == QEvent.Enter:
                self._hover = True
                self._hide_timer.stop()
                self._show()
            elif t == QEvent.Leave:
                self._hover = False
                if not self._pinned:
                    self._hide_timer.start()
        return super().eventFilter(obj, ev)

    def _on_pin(self, pinned: bool) -> None:
        self._pinned = bool(pinned)
        if self._s is not None:
            self._s.set(self._pinned_key, self._pinned)
        self._apply_mode()

    def _apply_mode(self, *, initial: bool = False) -> None:
        if self._pinned:
            self._show(immediate=initial)
        else:
            self._hide(immediate=initial)

    def _show(self, *, immediate: bool = False) -> None:
        self.show()
        self.raise_()
        if immediate:
            self._opacity.setOpacity(1.0)
            self.move(self._x, self._y_show)
            return

        self._anim_pos.stop()
        self._anim_op.stop()

        self._anim_pos.setStartValue(self.pos())
        self._anim_pos.setEndValue(QPoint(self._x, self._y_show))
        self._anim_pos.start()

        self._anim_op.setStartValue(float(self._opacity.opacity()))
        self._anim_op.setEndValue(1.0)
        self._anim_op.start()

    def _hide(self, *, immediate: bool = False) -> None:
        if self._pinned:
            return
        if immediate:
            self._opacity.setOpacity(0.0)
            self.move(self._x, self._y_hide)
            self.hide()
            return

        self._anim_pos.stop()
        self._anim_op.stop()

        self._anim_pos.setStartValue(self.pos())
        self._anim_pos.setEndValue(QPoint(self._x, self._y_hide))
        self._anim_pos.start()

        self._anim_op.setStartValue(float(self._opacity.opacity()))
        self._anim_op.setEndValue(0.0)
        self._anim_op.start()

    def _on_anim_done(self) -> None:
        if self._pinned or self._hover:
            return
        if float(self._opacity.opacity()) <= 0.01:
            self.hide()
