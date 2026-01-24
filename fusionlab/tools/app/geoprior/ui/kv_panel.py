# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy


class KeyValuePanel(QFrame):
    """
    Simple 2-column key/value grid.
    Styled via objectNames:
      - QFrame#kvPanel
      - QLabel#kvKey
      - QLabel#kvVal
      - QLabel#kvVal[path="true"] (optional)
    """
    def __init__(
        self,
        parent=None,
        *,
        max_rows: Optional[int] = None,
        compact: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("kvPanel")
        self._max_rows = max_rows
        self._compact = compact

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(6 if compact else 10)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self._cells: List[Tuple[QLabel, QLabel]] = []

    def clear_rows(self) -> None:
        for k, v in self._cells:
            self._grid.removeWidget(k)
            self._grid.removeWidget(v)
            k.deleteLater()
            v.deleteLater()
        self._cells.clear()

    def set_rows(self, rows: List[Tuple[str, str]]) -> None:
        self.clear_rows()
        if not rows:
            rows = [("—", "—")]

        if self._max_rows is not None:
            rows = rows[: self._max_rows]

        for r, (key, val) in enumerate(rows):
            lk = QLabel(str(key), self)
            lk.setObjectName("kvKey")
            lk.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            lv = QLabel(str(val), self)
            lv.setObjectName("kvVal")
            lv.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lv.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lv.setWordWrap(True)

            # Tag path-like values so stylesheet can give them a pill background if desired
            if isinstance(val, str) and ("/" in val or "\\" in val):
                lv.setProperty("path", "true")

            self._grid.addWidget(lk, r, 0, 1, 1)
            self._grid.addWidget(lv, r, 1, 1, 1)

            self._grid.setColumnStretch(0, 0)
            self._grid.setColumnStretch(1, 1)

            self._cells.append((lk, lv))
