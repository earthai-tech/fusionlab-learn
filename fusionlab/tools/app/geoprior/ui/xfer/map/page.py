# geoprior/ui/xfer/map/page.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QStackedLayout,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from ....config.store import GeoConfigStore
from .adv_overlay import (
    XferMapAdvDrawer,
    XferMapAdvWindow,
)
from .controller import XferMapController
from .head import XferMapHeadBar
from .toolbar import XferMapToolbar
from .view import MapView
from .basemap_overlay import XferBasemapQuickOverlay


class XferMapPage(QWidget):
    """
    Map page (Strategy-1):
      [ map head ]
      [ init toolbar ]
      [ overlay host: map view + adv drawer overlay ]
      + optional floating adv window (pin)
    """

    request_open_options = pyqtSignal()
    request_expand = pyqtSignal(bool)
    request_mode_switch = pyqtSignal(str)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._s = store

        # -------------------------
        # Head (always visible)
        # -------------------------
        self.head = XferMapHeadBar(parent=self)
        self.head.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        # -------------------------
        # Init toolbar (inline)
        # -------------------------
        self._toolbar = XferMapToolbar(parent=self)

        # -------------------------
        # Overlay host: view + drawer
        # -------------------------
        self._host = QWidget(self)
        self._host.setObjectName("xferMapOverlayHost")
        self._host.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self._host_stack = QStackedLayout(self._host)
        self._host_stack.setContentsMargins(0, 0, 0, 0)
        self._host_stack.setStackingMode(
            QStackedLayout.StackAll
        )

        self._view = MapView(parent=self._host)

        # self._adv = XferMapAdvDrawer(
        #     store=self._s,
        #     parent=self._host,
        # )
        # self._adv.set_open(False)

        self._host_stack.addWidget(self._view)

        self._bm = XferBasemapQuickOverlay(self._s)
        self._view.add_overlay(self._bm)

        self._adv = XferMapAdvDrawer(store=self._s)
        self._adv.set_open(False)
        self._view.add_overlay(self._adv)
        
        # self._host_stack.addWidget(self._bm)
        # self._host_stack.addWidget(self._adv)

        # Floating window for "Pin"
        self._adv_win = XferMapAdvWindow(parent=self)

        # -------------------------
        # Controller wires tb <-> view
        # -------------------------
        self._ctl = XferMapController(
            store=self._s,
            toolbar=self._toolbar,
            view=self._view,
            parent=self,
        )

        self._build_ui()
        self._wire()

    # -------------------------
    # Public API
    # -------------------------
    @property
    def toolbar(self) -> XferMapToolbar:
        return self._toolbar

    @property
    def view(self) -> MapView:
        return self._view

    def refresh(self) -> None:
        self._ctl.refresh()

    def set_expanded(self, on: bool) -> None:
        on = bool(on)
        self.head.setVisible(not on)
        if hasattr(self._toolbar, "set_expanded"):
            self._toolbar.set_expanded(on)

    def set_centroids(
        self,
        src_name: str,
        src_lat: float,
        src_lon: float,
        tgt_name: str,
        tgt_lat: float,
        tgt_lon: float,
    ) -> None:
        if hasattr(self._view, "set_centroids"):
            self._view.set_centroids(
                src_name,
                src_lat,
                src_lon,
                tgt_name,
                tgt_lat,
                tgt_lon,
            )

    def clear_centroids(self) -> None:
        if hasattr(self._view, "clear_centroids"):
            self._view.clear_centroids()

    # -------------------------
    # UI + wiring
    # -------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.addWidget(self.head, 0)
        root.addWidget(self._toolbar, 0)
        root.addWidget(self._host, 1)

    def _wire(self) -> None:
        self.head.open_run_clicked.connect(
            self.request_open_options.emit
        )

        self._toolbar.request_open_options.connect(
            self.request_open_options.emit
        )
        self._toolbar.request_expand.connect(
            self.request_expand.emit
        )

        if hasattr(self._toolbar, "request_mode_switch"):
            self._toolbar.request_mode_switch.connect(
                self.request_mode_switch.emit
            )

        # Toolbar -> toggle advanced drawer
        if hasattr(self._toolbar, "request_toggle_advanced"):
            self._toolbar.request_toggle_advanced.connect(
                self._on_adv_toggle
            )

        # Drawer controls
        self._adv.request_close.connect(self._close_adv)
        self._adv.request_pin.connect(self._pin_adv)

        self._bm.request_open_advanced.connect(
            self._on_adv_toggle
        )
        # Floating window controls
        self._adv_win.request_unpin.connect(self._unpin_adv)

    # -------------------------
    # Advanced drawer handlers
    # -------------------------
    # def _on_adv_toggle(self, on: bool) -> None:
    #     if self._adv_win.isVisible():
    #         self._adv_win.raise_()
    #         self._adv_win.activateWindow()
    #         return
    #     self._adv.set_open(bool(on))

    def _on_adv_toggle(self, on: Optional[bool] = None) -> None:
        # If advanced is pinned in a floating window,
        # clicking the icon just brings it to front.
        if self._adv_win.isVisible():
            self._adv_win.raise_()
            self._adv_win.activateWindow()
            return

        # If called without args (basemap gear),
        # toggle based on current drawer state.
        if on is None:
            on = not self._adv.is_open()
        else:
            # If some callers always emit True, treat
            # "True while already open" as toggle-close.
            if bool(on) and self._adv.is_open():
                on = False

        on = bool(on)
        self._adv.set_open(on)

        # Keep toolbar icon state consistent (if supported)
        if hasattr(self._toolbar, "set_advanced_open"):
            self._toolbar.set_advanced_open(on)
            
    def _close_adv(self) -> None:
        self._adv.set_open(False)
        if hasattr(self._toolbar, "set_advanced_open"):
            self._toolbar.set_advanced_open(False)

    def _pin_adv(self) -> None:
        w = self._adv.take_panel()
        if w is None:
            return

        self._adv.set_open(False)
        if hasattr(self._toolbar, "set_advanced_open"):
            self._toolbar.set_advanced_open(False)

        self._adv_win.set_panel(w)
        self._adv_win.show()
        self._adv_win.raise_()
        self._adv_win.activateWindow()

    def _unpin_adv(self) -> None:
        w = self._adv_win.take_panel()
        if w is None:
            self._adv_win.hide()
            return

        self._adv.set_panel(w)
        self._adv_win.hide()
