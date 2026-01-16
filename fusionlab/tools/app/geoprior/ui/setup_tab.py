# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ..config.store import GeoConfigStore
from .setup.panel import ConfigCenterPanel

class SetupTab(QWidget):
    """
    Thin wrapper for consistency with DataTab, etc.

    This is a "shadow" of ConfigCenterPanel:
    SetupTab owns layout; panel owns logic.
    """

    def __init__(
        self,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.panel = ConfigCenterPanel(
            store=store,
            parent=self,
        )
        lay.addWidget(self.panel, 1)

    def set_dataset_columns(self, cols: List[str]) -> None:
        self.panel.set_dataset_columns(cols)
