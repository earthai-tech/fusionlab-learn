# geoprior/ui/xfer/run/navigator.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.run.navigator

Left navigator (A+E) for Xfer RUN mode.

Re-styled to match Inference navigator:
- trainNavCard / trainNavTitle
- trainNavList + navRow/navText/navChip
- sumLine + runComputeText
- Computer details uses a scroll area (transparent)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
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
    QVBoxLayout,
    QWidget,
)

from ....config.store import GeoConfigStore
from ....device_options import runtime_summary_text
from .status import compute_xfer_nav


__all__ = ["XferNavigator"]


@dataclass(frozen=True)
class NavItemSpec:
    key: str
    title: str


def _btn(b: QPushButton) -> None:
    b.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    b.setMinimumHeight(34)
    b.setCursor(Qt.PointingHandCursor)  # type: ignore


class _NavRow(QWidget):
    """
    Inference/Train styled row.

    Uses selectors:
    - QWidget#navRow
    - QLabel#navText
    - QLabel#navChip[status="ok|warn|err|off"]
    """

    def __init__(
        self,
        *,
        title: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("navRow")
        self.setProperty("selected", False)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._text = QLabel(title, self)
        self._text.setObjectName("navText")
        self._text.setWordWrap(False)

        self._chip = QLabel("—", self)
        self._chip.setObjectName("navChip")
        self._chip.setAlignment(Qt.AlignCenter)
        self._chip.setProperty("status", "off")
        self._chip.setMinimumWidth(34)
        self._chip.setMinimumHeight(18)
        self._chip.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)
        lay.addWidget(self._text, 1)
        lay.addWidget(self._chip, 0)

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)

    def set_chip(self, *, status: str, text: str) -> None:
        self._chip.setText(str(text))
        self._chip.setProperty("status", str(status))
        self._chip.style().unpolish(self._chip)
        self._chip.style().polish(self._chip)


class XferNavigator(QWidget):
    """
    Xfer RUN navigator styled like Inference navigator.

    Cards
    -----
    1) Tips
    2) Setup checklist
    3) Computer details (scroll)
    """

    clicked = pyqtSignal(str)
    item_selected = pyqtSignal(str)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
        make_card=None,  # kept for backward compat, unused
    ) -> None:
        super().__init__(parent)
        self._s = store

        self._items: List[NavItemSpec] = [
            NavItemSpec("cities", "Cities & splits"),
            NavItemSpec("outputs", "Outputs & alignments"),
            NavItemSpec("strategy", "Strategy & warm-start"),
            NavItemSpec("results", "Results & view"),
        ]
        self._rows: Dict[str, _NavRow] = {}

        self._build_ui()
        self.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding,
        )
        self.setMinimumWidth(260)

        self._wire()
        self.refresh_compute()
        self.refresh()
        self._s.config_changed.connect(lambda _k: self.refresh())

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def refresh(self) -> None:
        st = compute_xfer_nav(self._s)
    
        for spec in self._items:
            row = self._rows.get(spec.key)
            if row is None:
                continue
    
            chip = st.get(spec.key, {"status": "off", "text": "—"})
            row.set_chip(
                status=str(chip.get("status", "off")),
                text=str(chip.get("text", "—")),
            )
    
        self._fix_nav_list_height()

    def set_status(self, key: str, state: str) -> None:
        row = self._rows.get(str(key))
        if row is None:
            return

        st = (state or "").strip().lower()
        if st not in ("ok", "warn", "err"):
            st = "off"

        if st == "ok":
            txt = "OK"
        elif st == "warn":
            txt = "Fix"
        elif st == "err":
            txt = "Err"
        else:
            txt = "—"

        row.set_chip(status=st, text=txt)

    def refresh_compute(self) -> None:
        try:
            txt = runtime_summary_text()
        except Exception:
            try:
                txt = runtime_summary_text(self._s)
            except Exception:
                txt = "Runtime summary unavailable."
        self.lbl_compute.setText(str(txt))

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # =========================================================
        # Card 0: Tips
        # =========================================================
        tips = QFrame(self)
        tips.setObjectName("trainNavCard")
        tips.setFrameShape(QFrame.NoFrame)

        tips_l = QVBoxLayout(tips)
        tips_l.setContentsMargins(10, 10, 10, 10)
        tips_l.setSpacing(10)

        tips_title = QLabel("Tips", tips)
        tips_title.setObjectName("trainNavTitle")
        tips_l.addWidget(tips_title, 0)

        self.btn_open_run = QPushButton("Open run", tips)
        self.btn_open_out = QPushButton("Open output", tips)
        self.btn_eval_csv = QPushButton("Eval CSV", tips)
        self.btn_summary = QPushButton("Summary JSON", tips)

        for b in (
            self.btn_open_run,
            self.btn_open_out,
            self.btn_eval_csv,
            self.btn_summary,
        ):
            _btn(b)
            b.setObjectName("miniAction")

        acts = QWidget(tips)
        g = QGridLayout(acts)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(8)

        g.addWidget(self.btn_open_run, 0, 0)
        g.addWidget(self.btn_open_out, 0, 1)
        g.addWidget(self.btn_eval_csv, 1, 0)
        g.addWidget(self.btn_summary, 1, 1)

        tips_l.addWidget(acts, 0)

        hint = QLabel("Shortcuts update after a run.", tips)
        hint.setObjectName("sumLine")
        hint.setWordWrap(True)
        tips_l.addWidget(hint, 0)

        root.addWidget(tips, 0)

        # =========================================================
        # Card 1: Setup checklist (NAV)
        # =========================================================
        nav = QFrame(self)
        nav.setObjectName("trainNavCard")
        nav.setFrameShape(QFrame.NoFrame)

        nav_l = QVBoxLayout(nav)
        nav_l.setContentsMargins(10, 10, 10, 10)
        nav_l.setSpacing(10)

        nav_title = QLabel("Setup checklist", nav)
        nav_title.setObjectName("trainNavTitle")
        nav_l.addWidget(nav_title, 0)

        self.list = QListWidget(nav)
        self.list.setObjectName("trainNavList")
        self.list.setSpacing(2)
        self.list.setUniformItemSizes(True)
        self.list.setVerticalScrollMode(
            QListWidget.ScrollPerPixel
        )
        self.list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        for spec in self._items:
            row = _NavRow(title=spec.title, parent=self.list)
            self._rows[spec.key] = row

            it = QListWidgetItem(self.list)
            it.setData(Qt.UserRole, spec.key)
            it.setSizeHint(row.sizeHint())
            self.list.addItem(it)
            self.list.setItemWidget(it, row)

        self.list.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )
        self._fix_nav_list_height()

        nav_l.addWidget(self.list, 0)
        root.addWidget(nav, 0)

        # =========================================================
        # Card 2: Computer details (fills remaining space)
        # =========================================================
        comp = QFrame(self)
        comp.setObjectName("trainNavCard")
        comp.setFrameShape(QFrame.NoFrame)

        comp_l = QVBoxLayout(comp)
        comp_l.setContentsMargins(10, 10, 10, 10)
        comp_l.setSpacing(10)

        comp_title = QLabel("Computer details", comp)
        comp_title.setObjectName("trainNavTitle")
        comp_l.addWidget(comp_title, 0)

        sc = QScrollArea(comp)
        sc.setObjectName("inferCompScroll")
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidgetResizable(True)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        sc.setStyleSheet(
            "QScrollArea { background: transparent; }"
        )
        sc.viewport().setAutoFillBackground(False)
        sc.setAttribute(Qt.WA_StyledBackground, True)

        body = QWidget(sc)
        body.setObjectName("inferCompBody")
        body.setAttribute(Qt.WA_StyledBackground, True)
        body.setAutoFillBackground(False)
        sc.setWidget(body)

        body_l = QVBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(8)

        self.lbl_compute = QLabel("", body)
        self.lbl_compute.setObjectName("runComputeText")
        self.lbl_compute.setWordWrap(True)
        self.lbl_compute.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        self.lbl_compute.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )
        body_l.addWidget(self.lbl_compute, 0)

        notes = QLabel(
            "• Ensure A/B outputs are produced by the same "
            "pipeline version.\n"
            "• Keep splits aligned (val/test) before view.\n"
            "• Result viewers use the latest run directory.",
            body,
        )
        notes.setWordWrap(True)
        notes.setObjectName("sumLine")
        body_l.addWidget(notes, 0)

        body_l.addStretch(1)

        sc.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        comp_l.addWidget(sc, 1)
        root.addWidget(comp, 1)

        # Default selection
        if self.list.count() > 0:
            self.list.setCurrentRow(0)
            self._sync_selected_row()

        self.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding,
        )

    def _fix_nav_list_height(self) -> None:
        if self.list is None:
            return
        n = self.list.count()
        if n <= 0:
            return

        cap = 6
        show_n = min(n, cap)

        h = 0
        for i in range(show_n):
            it = self.list.item(i)
            if it is None:
                continue
            idx = self.list.indexFromItem(it)
            h += self.list.sizeHintForIndex(idx).height()

        h += 12
        self.list.setFixedHeight(max(120, h))

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.list.currentItemChanged.connect(self._on_select)
        self._s.config_changed.connect(
            lambda _k: self.refresh_compute()
        )

        # NOTE: controller should connect these buttons
        # where needed (Open run, Open output, etc.)

    def _on_select(
        self,
        cur: Optional[QListWidgetItem],
        _prev: Optional[QListWidgetItem],
    ) -> None:
        if cur is None:
            return
        key = str(cur.data(Qt.UserRole) or "").strip()
        if not key:
            return

        self.list.scrollToItem(cur)
        self._sync_selected_row()
        self.clicked.emit(key)
        self.item_selected.emit(key)

    def _sync_selected_row(self) -> None:
        cur = self.list.currentItem()
        cur_key = ""
        if cur is not None:
            cur_key = str(cur.data(Qt.UserRole) or "")

        for k, row in self._rows.items():
            row.set_selected(k == cur_key)


# # -*- coding: utf-8 -*-
# # License: BSD-3-Clause
# # Author: LKouadio <etanoyau@gmail.com>

# """
# geoprior.ui.xfer.run.navigator

# Left navigator (A+E) for Xfer RUN mode.

# - Tips card
# - Setup checklist card (clickable)
# - Computer details card (optional)
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Callable, Dict, Optional, Tuple

# from PyQt5.QtCore import Qt, QSize, pyqtSignal
# from PyQt5.QtWidgets import (
#     QHBoxLayout,
#     QLabel,
#     QListWidget,
#     QListWidgetItem,
#     QPushButton,
#     QSizePolicy,
#     QVBoxLayout,
#     QWidget,
# )

# from ....config.store import GeoConfigStore
# from ....device_options import runtime_summary_text

# MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]


# @dataclass(frozen=True)
# class NavItem:
#     key: str
#     title: str


# class XferNavigator(QWidget):
#     """
#     Left navigator column (RUN mode).

#     Exposes:
#     - clicked(key): user wants to jump to section
#     - set_status(key, state): update checklist chips
#     """

#     clicked = pyqtSignal(str)
#     # compat: older code expects item_selected
#     item_selected = pyqtSignal(str)
    
#     def __init__(
#         self,
#         *,
#         store: GeoConfigStore,
#         make_card: MakeCardFn,
#         parent: Optional[QWidget] = None,
#     ) -> None:
#         super().__init__(parent)
#         self._s = store
#         self._make_card = make_card

#         self._chips: Dict[str, QLabel] = {}

#         self._build_ui()

#     # -------------------------------------------------
#     # Public
#     # -------------------------------------------------
#     def set_status(self, key: str, state: str) -> None:
#         chip = self._chips.get(key)
#         if chip is None:
#             return

#         st = (state or "").strip().lower()
#         text = st.upper() if st else "—"
#         chip.setText(text)

#         chip.setProperty("ok", st == "ok")
#         chip.setProperty("warn", st == "warn")
#         chip.setProperty("err", st == "err")

#         chip.style().unpolish(chip)
#         chip.style().polish(chip)

#     # -------------------------------------------------
#     # UI
#     # -------------------------------------------------
#     def _build_ui(self) -> None:
#         root = QVBoxLayout(self)
#         root.setContentsMargins(0, 0, 0, 0)
#         root.setSpacing(10)

#         tips_card, tips_box = self._make_card("Tips")
#         self._build_tips(tips_box)
#         root.addWidget(tips_card)

#         chk_card, chk_box = self._make_card("Setup checklist")
#         self._build_checklist(chk_box)
#         root.addWidget(chk_card)

#         sys_card, sys_box = self._make_card("Computer details")
#         self._build_computer(sys_box)
#         root.addWidget(sys_card)

#         root.addStretch(1)

#         self.setSizePolicy(
#             QSizePolicy.Preferred,
#             QSizePolicy.Expanding,
#         )

#     def _build_tips(self, box: QVBoxLayout) -> None:
#         row1 = QHBoxLayout()
#         row1.setContentsMargins(0, 0, 0, 0)
#         row1.setSpacing(8)

#         self.btn_open_run = QPushButton("Open run")
#         self.btn_open_out = QPushButton("Open output")

#         row1.addWidget(self.btn_open_run)
#         row1.addWidget(self.btn_open_out)
#         row1.addStretch(1)

#         row2 = QHBoxLayout()
#         row2.setContentsMargins(0, 0, 0, 0)
#         row2.setSpacing(8)

#         self.btn_eval_csv = QPushButton("Eval CSV")
#         self.btn_summary = QPushButton("Summary JSON")

#         row2.addWidget(self.btn_eval_csv)
#         row2.addWidget(self.btn_summary)
#         row2.addStretch(1)

#         hint = QLabel("Shortcuts update after a run.")
#         hint.setObjectName("setupCardSubtitle")
#         hint.setWordWrap(True)

#         box.addLayout(row1)
#         box.addLayout(row2)
#         box.addWidget(hint)

#     def _build_checklist(self, box: QVBoxLayout) -> None:
#         self.lst = QListWidget(self)
#         self.lst.setObjectName("xferChecklist")
#         self.lst.setSelectionMode(self.lst.NoSelection)

#         items = [
#             NavItem("cities", "Cities & splits"),
#             NavItem("outputs", "Outputs & alignments"),
#             NavItem("strategy", "Strategy & warm-start"),
#             NavItem("results", "Results & view"),
#         ]

#         for it in items:
#             qit = QListWidgetItem(self.lst)
#             qit.setSizeHint(QSize(10, 36))  # type: ignore[attr-defined]
#             w = self._make_nav_row(it.title, it.key)
#             self.lst.addItem(qit)
#             self.lst.setItemWidget(qit, w)

#         self.lst.itemClicked.connect(self._on_item_clicked)

#         box.addWidget(self.lst)

#     def _make_nav_row(self, title: str, key: str) -> QWidget:
#         w = QWidget(self)
#         lay = QHBoxLayout(w)
#         lay.setContentsMargins(8, 4, 8, 4)
#         lay.setSpacing(8)

#         lbl = QLabel(title)
#         lbl.setObjectName("xferNavItem")
#         lbl.setToolTip(title)

#         chip = QLabel("—")
#         chip.setObjectName("xferNavChip")
#         chip.setMinimumWidth(44)
#         chip.setAlignment(Qt.AlignCenter)

#         self._chips[key] = chip

#         lay.addWidget(lbl, 1)
#         lay.addWidget(chip, 0)

#         w.setProperty("xfer_nav_key", key)
#         return w

#     def _build_computer(self, box: QVBoxLayout) -> None:
#         txt = self._compute_runtime_text()
#         self.lbl_runtime = QLabel(txt)
#         self.lbl_runtime.setObjectName("xferRuntimeText")
#         self.lbl_runtime.setWordWrap(True)
#         box.addWidget(self.lbl_runtime)

#     def _compute_runtime_text(self) -> str:
#         return str(runtime_summary_text(self._s))

#     # -------------------------------------------------
#     # Events
#     # -------------------------------------------------
#     def _on_item_clicked(self, it: QListWidgetItem) -> None:
#         w = self.lst.itemWidget(it)
#         if w is None:
#             return
#         key = str(w.property("xfer_nav_key") or "")
#         if not key:
#             return 
        
#         self.clicked.emit(key)
#         self.item_selected.emit(key)
  
        