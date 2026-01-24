# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.head

Head bars for Xfer tab.

- XferRunHeadBar: run mode top bar (optional, later)
- XferMapHeadBar: map mode top bar (mode + expand + map opts)
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..icon_utils import try_icon


__all__ = ["XferMapHeadBar"]


class XferMapHeadBar(QFrame):
    """
    Map-mode head bar.

    Signals
    -------
    - mode_changed(str): "run" or "map"
    - expand_toggled(bool): map takes full area
    - action_requested(str): generic actions ("reset_view", ...)
    """

    mode_changed = pyqtSignal(str)
    expand_toggled = pyqtSignal(bool)
    action_requested = pyqtSignal(str)

    def __init__(
        self,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("xferMapHead")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()

    # -------------------------------------------------
    # Icons
    # -------------------------------------------------
    def _std_icon(self, sp: QStyle.StandardPixmap):
        return self.style().standardIcon(sp)

    def _set_icon(
        self,
        btn: QToolButton,
        name: str,
        fallback: QStyle.StandardPixmap,
    ) -> None:
        ic = try_icon(name)
        if ic is None:
            ic = self._std_icon(fallback)
        btn.setIcon(ic)

    def _mk_btn(
        self,
        *,
        tip: str,
        icon_name: str,
        fallback: QStyle.StandardPixmap,
        checkable: bool = False,
    ) -> QToolButton:
        b = QToolButton(self)
        b.setObjectName("miniAction")
        b.setAutoRaise(True)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        b.setToolTip(tip)
        b.setCheckable(bool(checkable))
        self._set_icon(b, icon_name, fallback)
        return b

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        # root = QHBoxLayout(self)
        # root.setContentsMargins(8, 6, 8, 6)
        # root.setSpacing(10)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)
    
        top = QWidget(self)
        row = QHBoxLayout(top)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        
        # Left: title + subtitle
        tcol = QVBoxLayout()
        tcol.setContentsMargins(0, 0, 0, 0)
        tcol.setSpacing(0)

        self.lbl_title = QLabel("Transfer map")
        self.lbl_title.setObjectName("headTitle")

        self.lbl_sub = QLabel(
            "Compare A→B / B→A results and inspect per-split metrics."
        )
        self.lbl_sub.setObjectName("setupCardSubtitle")
        self.lbl_sub.setWordWrap(True)

        tcol.addWidget(self.lbl_title)
        tcol.addWidget(self.lbl_sub)

        root.addLayout(tcol, 1)

        # Middle: mode toggle (Run / Map)
        self.btn_mode_run = QToolButton(self)
        self.btn_mode_run.setText("Run")
        self.btn_mode_run.setCheckable(True)
        self.btn_mode_run.setObjectName("chipBtn")

        self.btn_mode_map = QToolButton(self)
        self.btn_mode_map.setText("Map")
        self.btn_mode_map.setCheckable(True)
        self.btn_mode_map.setObjectName("chipBtn")

        self._mode_grp = QButtonGroup(self)
        self._mode_grp.setExclusive(True)
        self._mode_grp.addButton(self.btn_mode_run)
        self._mode_grp.addButton(self.btn_mode_map)

        self.btn_mode_map.setChecked(True)

        root.addWidget(self.btn_mode_run, 0)
        root.addWidget(self.btn_mode_map, 0)

        # Right: map actions (expand is the key one)
        self.btn_expand = self._mk_btn(
            tip="Expand map (hide panels)",
            icon_name="expand.svg",
            fallback=QStyle.SP_TitleBarMaxButton,
            checkable=True,
        )
        self.btn_reset = self._mk_btn(
            tip="Reset view",
            icon_name="refresh.svg",
            fallback=QStyle.SP_BrowserReload,
            checkable=False,
        )

        root.addWidget(self.btn_expand, 0)
        root.addWidget(self.btn_reset, 0)
        

        self._tb_host = QWidget(self)
        self._tb_lay = QVBoxLayout(self._tb_host)
        self._tb_lay.setContentsMargins(0, 0, 0, 0)
        self._tb_lay.setSpacing(0)
        root.addWidget(self._tb_host)
    
        # Wire
        self.btn_mode_run.toggled.connect(self._on_mode_run)
        self.btn_mode_map.toggled.connect(self._on_mode_map)
        self.btn_expand.toggled.connect(self.expand_toggled.emit)
        self.btn_reset.clicked.connect(
            lambda: self.action_requested.emit("reset_view")
        )

        self._toolbar = None

    def set_toolbar(self, tb: QWidget) -> None:
        if getattr(self, "_toolbar", None) is tb:
            return
    
        old = getattr(self, "_toolbar", None)
        if old is not None:
            self._tb_lay.removeWidget(old)
            old.setParent(None)
    
        self._toolbar = tb
        tb.setParent(self._tb_host)
        self._tb_lay.addWidget(tb)
    
        # Forward toolbar signals if present
        if hasattr(tb, "request_expand"):
            tb.request_expand.connect(self.expand_toggled.emit)
        if hasattr(tb, "request_open_options"):
            tb.request_open_options.connect(
                lambda: self.mode_changed.emit("run")
            )
    
    def take_toolbar(self) -> Optional[QWidget]:
        tb = getattr(self, "_toolbar", None)
        if tb is None:
            return None
    
        try:
            if hasattr(tb, "request_expand"):
                tb.request_expand.disconnect(
                    self.expand_toggled.emit
                )
        except Exception:
            pass
    
        self._tb_lay.removeWidget(tb)
        tb.setParent(None)
        self._toolbar = None
        return tb

    # -------------------------------------------------
    # Mode logic
    # -------------------------------------------------
    def _on_mode_run(self, on: bool) -> None:
        if on:
            self.mode_changed.emit("run")

    def _on_mode_map(self, on: bool) -> None:
        if on:
            self.mode_changed.emit("map")

    # -------------------------------------------------
    # Public helpers
    # -------------------------------------------------
    def set_mode(self, mode: str) -> None:
        m = (mode or "").strip().lower()
        if m == "run":
            self.btn_mode_run.setChecked(True)
            return
        self.btn_mode_map.setChecked(True)

    def set_expanded(self, on: bool) -> None:
        self.btn_expand.setChecked(bool(on))

    def set_subtitle(self, text: str) -> None:
        self.lbl_sub.setText(str(text or ""))

    def set_title(self, text: str) -> None:
        self.lbl_title.setText(str(text or ""))
