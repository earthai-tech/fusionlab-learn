# geoprior/ui/preprocess/navigator.py
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
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..icon_utils import try_icon
from ..kv_panel import KeyValuePanel

__all__ = ["PreprocessNavigator"]


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
        self.setProperty("matched", False)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.setMinimumHeight(34)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        self.lbl = QLabel(text, self)
        self.lbl.setObjectName("navText")

        self.chip = QLabel("—", self)
        self.chip.setObjectName("navChip")
        self.chip.setProperty("status", "off")
        self.chip.setMinimumWidth(34)
        self.chip.setMinimumHeight(18)
        self.chip.setAlignment(Qt.AlignCenter)
        self.chip.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )

        lay.addWidget(self.lbl, 1)
        lay.addWidget(self.chip, 0)

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_chip(self, status: str, text: str) -> None:
        self.chip.setProperty("status", str(status))
        self.chip.setText(str(text))
        self.chip.style().unpolish(self.chip)
        self.chip.style().polish(self.chip)
        self.chip.update()

    def set_matched(self, on: bool) -> None:
        self.setProperty("matched", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class PreprocessNavigator(QWidget):
    """
    Left navigator for Preprocess tab:
    - top actions grid
    - chip-list sections
    - details scroll (computer/context/snapshot)
    """

    section_changed = pyqtSignal(str)

    readiness_clicked = pyqtSignal()
    inspect_clicked = pyqtSignal()
    history_clicked = pyqtSignal()
    artifacts_clicked = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._rows: Dict[str, _NavRow] = {}
        self._labels: Dict[str, str] = {}

        self._build_ui()
        self._wire()

    # -------------------------------------------------
    # QFrame card builder (matches TRAIN_NAV_* QSS)
    # -------------------------------------------------
    def _make_nav_card(
        self,
        title: str,
    ) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(self)
        card.setObjectName("trainNavCard")
        card.setFrameShape(QFrame.NoFrame)
        card.setAttribute(Qt.WA_StyledBackground, True)

        card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )

        outer = QVBoxLayout(card)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        lbl = QLabel(title, card)
        lbl.setObjectName("trainNavTitle")
        outer.addWidget(lbl, 0)

        body = QWidget(card)
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(8)

        outer.addWidget(body, 1)
        return card, body_lay

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # -------------------------
        # Card: Setup checklist
        # -------------------------
        nav_card, nav_box = self._make_nav_card(
            "Setup checklist"
        )

        self.btn_readiness = self._mini_btn(
            "Readiness",
            "readiness.svg",
            QStyle.SP_DialogYesButton,
        )
        self.btn_inspect = self._mini_btn(
            "Inspect",
            "inspect.svg",
            QStyle.SP_FileDialogContentsView,
        )
        self.btn_history = self._mini_btn(
            "Run history",
            "history.svg",
            QStyle.SP_BrowserReload,
        )
        self.btn_artifacts = self._mini_btn(
            "Artifacts",
            "artifacts.svg",
            QStyle.SP_DirIcon,
        )

        acts = QWidget(nav_card)
        g = QGridLayout(acts)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(8)

        g.addWidget(self.btn_readiness, 0, 0)
        g.addWidget(self.btn_inspect, 0, 1)
        g.addWidget(self.btn_history, 1, 0)
        g.addWidget(self.btn_artifacts, 1, 1)

        nav_box.addWidget(acts, 0)
        nav_box.addSpacing(6)

        self.list = QListWidget(nav_card)
        self.list.setObjectName("trainNavList")
        self.list.setSpacing(2)
        self.list.setUniformItemSizes(True)
        nav_box.addWidget(self.list, 1)

        self.add_item("Inputs", "inputs")
        self.add_item("Stage-1 policy", "policy")
        self.add_item("Stage-1 status", "status")
        self.add_item("Build plan", "plan")

        self.list.setCurrentRow(0)
        self._sync_selected(0)

        outer.addWidget(nav_card, 0)

        # -------------------------
        # Card: Details
        # -------------------------
        info_card, info_box = self._make_nav_card("Details")

        sc = QScrollArea(info_card)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        sc.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )

        sc.setObjectName("trainCompScroll")
        sc.viewport().setAutoFillBackground(False)

        body = QWidget(sc)
        body.setObjectName("trainCompBody")
        body.setAttribute(Qt.WA_StyledBackground, True)
        body.setAutoFillBackground(False)
        sc.setWidget(body)

        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        t1 = QLabel("Computer", body)
        t1.setObjectName("trainNavTitle")
        lay.addWidget(t1, 0)

        self.comp_panel = KeyValuePanel(
            self,
            max_rows=12,
            compact=True,
        )
        lay.addWidget(self.comp_panel, 0)

        t2 = QLabel("Context", body)
        t2.setObjectName("trainNavTitle")
        lay.addWidget(t2, 0)

        self.ctx_panel = KeyValuePanel(
            self,
            max_rows=8,
            compact=True,
        )
        lay.addWidget(self.ctx_panel, 0)

        t3 = QLabel("Stage-1 snapshot", body)
        t3.setObjectName("trainNavTitle")
        lay.addWidget(t3, 0)

        self.stage1_panel = KeyValuePanel(
            self,
            max_rows=10,
            compact=True,
        )
        lay.addWidget(self.stage1_panel, 0)

        notes = QLabel(
            "• Status & snapshot update automatically.\n"
            "• Use Inspect → Readiness for full checks.\n"
            "• Stage-1 outputs live under the city root.",
            body,
        )
        notes.setWordWrap(True)
        notes.setObjectName("sumLine")
        lay.addWidget(notes, 0)

        lay.addStretch(1)

        sc.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        info_box.addWidget(sc, 1)

        outer.addWidget(info_card, 1)

        # keep the splitter from collapsing too far
        self.setMinimumWidth(260)
        self.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding,
        )

    def _mini_btn(
        self,
        text: str,
        svg_name: str,
        fallback_std: QStyle.StandardPixmap,
    ) -> QPushButton:
        b = QPushButton(text, self)

        ico = try_icon(svg_name) if svg_name else None
        if (ico is None) or ico.isNull():
            ico = self.style().standardIcon(fallback_std)

        b.setIcon(ico)
        b.setIconSize(QSize(16, 16))
        b.setCursor(Qt.PointingHandCursor)
        b.setObjectName("miniAction")
        b.setMinimumHeight(30)
        b.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        return b

    # -------------------------------------------------
    # Wiring
    # -------------------------------------------------
    def _wire(self) -> None:
        self.btn_readiness.clicked.connect(
            self.readiness_clicked.emit
        )
        self.btn_inspect.clicked.connect(
            self.inspect_clicked.emit
        )
        self.btn_history.clicked.connect(
            self.history_clicked.emit
        )
        self.btn_artifacts.clicked.connect(
            self.artifacts_clicked.emit
        )
        self.list.currentRowChanged.connect(self._on_row)

    # -------------------------------------------------
    # List API
    # -------------------------------------------------
    def add_item(self, text: str, key: str) -> None:
        it = QListWidgetItem(self.list)
        it.setText("")
        it.setData(Qt.UserRole, str(key))

        row = _NavRow(text, key, parent=self.list)
        it.setSizeHint(QSize(0, row.sizeHint().height()))

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

    def apply_filter(self, query: str, enabled: bool) -> None:
        q = str(query or "").strip().lower()
        on = bool(enabled)

        for k, row in self._rows.items():
            if (not on) or (not q):
                row.set_matched(False)
                continue

            label = self._labels.get(k, "")
            chip_txt = str(row.chip.text() or "")
            chip_stat = str(row.chip.property("status") or "")

            hay = f"{k} {label} {chip_txt} {chip_stat}"
            row.set_matched(q in hay.lower())
