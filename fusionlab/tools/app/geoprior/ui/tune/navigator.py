# geoprior/ui/tune/navigator.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional

from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


__all__ = ["TuneNavigator"]


class _NavRow(QWidget):
    def __init__(
        self,
        text: str,
        key: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.key = str(key)

        self.setObjectName("navRow")
        self.setProperty("selected", False)
        self.setAttribute(Qt.WA_StyledBackground, True)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        self.lbl = QLabel(text)
        self.lbl.setObjectName("navText")

        self.chip = QLabel("OK")
        self.chip.setObjectName("navChip")
        self.chip.setProperty("status", "ok")

        lay.addWidget(self.lbl, 1)
        lay.addWidget(self.chip, 0)

        # Filter state (style-safe; does nothing unless QSS uses it)
        self.setProperty("matched", False)

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)

    def set_chip(self, status: str, text: str) -> None:
        self.chip.setProperty("status", str(status))
        self.chip.setText(str(text))
        self.chip.style().unpolish(self.chip)
        self.chip.style().polish(self.chip)

    def set_matched(self, on: bool) -> None:
        self.setProperty("matched", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)


class TuneNavigator(QFrame):
    """
    Left navigator:
    - setup checklist buttons
    - section list with chips
    - apply_filter(q, on): optional highlight hook
    """

    section_changed = pyqtSignal(str)

    features_clicked = pyqtSignal()
    arch_clicked = pyqtSignal()
    prob_clicked = pyqtSignal()
    physics_clicked = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        # Keep Train styling names on purpose
        self.setObjectName("trainNavCard")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._rows: Dict[str, _NavRow] = {}
        self._labels: Dict[str, str] = {}

        self._build_ui()
        self._wire()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        title = QLabel("Setup checklist")
        title.setObjectName("trainNavTitle")
        root.addWidget(title)

        self.btn_features = QPushButton("Feature")
        self.btn_arch = QPushButton("Arch")
        self.btn_prob = QPushButton("Prob")
        self.btn_phys = QPushButton("Physics")

        for b in (
            self.btn_features,
            self.btn_arch,
            self.btn_prob,
            self.btn_phys,
        ):
            b.setObjectName("miniAction")
            b.setMinimumHeight(30)

        acts = QWidget(self)
        g = QGridLayout(acts)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(8)

        g.addWidget(self.btn_features, 0, 0)
        g.addWidget(self.btn_arch, 0, 1)
        g.addWidget(self.btn_prob, 1, 0)
        g.addWidget(self.btn_phys, 1, 1)

        root.addWidget(acts)

        self.list = QListWidget(self)
        self.list.setObjectName("trainNavList")
        self.list.setSpacing(2)
        self.list.setUniformItemSizes(True)
        root.addWidget(self.list, 0)

        self.add_item("Search space", "space")
        self.add_item("Physics switches", "physics")
        self.add_item("Algorithm", "algo")
        self.add_item("Trial template", "trial")
        self.add_item("Compute", "compute")
        self.add_item("Advanced", "adv")

        self.list.setCurrentRow(0)
        self._sync_selected(0)

    def _wire(self) -> None:
        self.btn_features.clicked.connect(
            self.features_clicked.emit
        )
        self.btn_arch.clicked.connect(self.arch_clicked.emit)
        self.btn_prob.clicked.connect(self.prob_clicked.emit)
        self.btn_phys.clicked.connect(self.physics_clicked.emit)

        self.list.currentRowChanged.connect(self._on_row)

    def add_item(self, text: str, key: str) -> None:
        it = QListWidgetItem(self.list)
        it.setText("")
        it.setData(Qt.UserRole, str(key))
        

        row = _NavRow(text, key, parent=self.list)
        
        h = row.sizeHint().height()
        it.setSizeHint(QSize(0, h))
        # it.setSizeHint(row.sizeHint())

        self.list.addItem(it)
        self.list.setItemWidget(it, row)

        k = str(key)
        self._rows[k] = row
        self._labels[k] = str(text)

    def _sync_selected(self, row: int) -> None:
        for i in range(self.list.count()):
            it = self.list.item(i)
            k = str(it.data(Qt.UserRole) or "")
            w = self._rows.get(k)
            if w is None:
                continue
            w.set_selected(i == row)

    def _on_row(self, row: int) -> None:
        if row < 0:
            return
        self._sync_selected(row)

        it = self.list.item(row)
        key = str(it.data(Qt.UserRole) or "")
        if key:
            self.section_changed.emit(key)

    def set_chip(self, key: str, status: str, text: str) -> None:
        w = self._rows.get(str(key))
        if w is None:
            return
        w.set_chip(status, text)

    # -------------------------------------------------
    # Filter API (required by tab.py)
    # -------------------------------------------------
    def apply_filter(self, query: str, enabled: bool) -> None:
        """
        Highlight matches in-place (never hides items).

        This is intentionally style-neutral:
        it only sets a 'matched' property on rows.
        """
        q = str(query or "").strip().lower()
        on = bool(enabled)

        for k, row in self._rows.items():
            if (not on) or (not q):
                row.set_matched(False)
                continue

            label = self._labels.get(k, "")
            chip_txt = str(row.chip.text() or "")
            chip_stat = str(row.chip.property("status") or "")

            hay = f"{k} {label} {chip_txt} {chip_stat}".lower()
            row.set_matched(q in hay)
