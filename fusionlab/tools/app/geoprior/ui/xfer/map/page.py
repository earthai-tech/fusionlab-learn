# geoprior/ui/xfer/map/page.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.map.page

Composable page:
[ toolbar row ]
[ big map view ]
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ....config.store import GeoConfigStore
from .toolbar import XferMapToolbar
from .controller import XferMapController
from .view import MapView

class XferMapPage(QWidget):
    """
    Map page for transferability.
    """

    request_open_options = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store

        self._toolbar = XferMapToolbar(parent=self)
        self._map = MapView(parent=self)

        # Defaults are handled by the controller (_ensure_defaults).
        self._ctl = XferMapController(
            store=self._store,
            toolbar=self._toolbar,
            view=self._map,
            parent=self,
        )

        self._build_ui()
        self._connect()

    @property
    def toolbar(self) -> XferMapToolbar:
        return self._toolbar

    @property
    def view(self) -> MapView:
        return self._map

    def refresh(self) -> None:
        self._ctl.refresh()

    # --- lightweight helpers for XferTab (centroids only) ---
    def set_centroids(
        self,
        src_name: str,
        src_lat: float,
        src_lon: float,
        tgt_name: str,
        tgt_lat: float,
        tgt_lon: float,
    ) -> None:
        self._map.set_centroids(
            src_name,
            src_lat,
            src_lon,
            tgt_name,
            tgt_lat,
            tgt_lon,
        )

    def clear_centroids(self) -> None:
        if hasattr(self._map, "clear_centroids"):
            self._map.clear_centroids()

    # ----------------------------
    # Internals
    # ----------------------------
    def _build_ui(self) -> None:
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(8)
        self._root.addWidget(self._toolbar)
        self._root.addWidget(self._map, 1)

    def _connect(self) -> None:
        self._toolbar.request_open_options.connect(
            self.request_open_options
        )
        
    def take_toolbar(self) -> None:
        if self._toolbar.parent() is not self:
            return
        self._root.removeWidget(self._toolbar)
        self._toolbar.setParent(None)

    def restore_toolbar(self) -> None:
        if self._toolbar.parent() is self:
            return
        self._toolbar.setParent(self)
        self._root.insertWidget(0, self._toolbar)
