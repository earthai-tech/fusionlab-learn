# geoprior/ui/inference/navigator.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QHBoxLayout, 
    QScrollArea
)

from ...config.store import GeoConfigStore
from ...device_options import runtime_summary_text

from .runtime_snapshot import InferRuntimeSnapshot
from .status import compute_infer_nav


__all__ = ["InferenceNavigator"]


@dataclass(frozen=True)
class NavItemSpec:
    key: str
    title: str


def _btn(b: QPushButton) -> None:
    b.setSizePolicy(
        QSizePolicy.Minimum,
        QSizePolicy.Fixed,
    )
    b.setMinimumHeight(34)
    # Optional: keep it slim like Train
    # b.setMaximumWidth(320)
    b.setCursor(Qt.PointingHandCursor)  # type: ignore


class _NavRow(QWidget):
    """
    Train-styled nav row.

    Matches TRAIN_NAV_ROW selectors:
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
        # Train CSS sets min-height in stylesheet; keep a safe fallback:
        self._chip.setMinimumHeight(18)
        self._chip.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )

        lay = QHBoxLayout(self)
        # same values you already had; Train row look depends on this
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


class InferenceNavigator(QWidget):
    """
    Left [A]: Inference navigator reorganized like Train.

    Order
    -----
    1) Tips card (top): 4 shortcut buttons + hint
    2) Setup checklist card: nav list (Train styling)
    3) Computer details card: runtime + small inference notes
    """

    section_selected = pyqtSignal(str)

    # Tips shortcuts (controller can wire to Details panel)
    open_run_clicked = pyqtSignal()
    open_eval_clicked = pyqtSignal()
    open_future_clicked = pyqtSignal()
    open_json_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._snapshot: Optional[InferRuntimeSnapshot] = None

        self._items: List[NavItemSpec] = [
            NavItemSpec("artifacts", "Artifacts"),
            NavItemSpec("uncertainty", "Uncertainty"),
            NavItemSpec("calibration", "Calibration"),
            NavItemSpec("outputs", "Outputs"),
            NavItemSpec("advanced", "Advanced"),
        ]
        self._rows: Dict[str, _NavRow] = {}

        self._build_ui()
        self._wire()
        self.refresh()
        self.refresh_compute()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def refresh(self) -> None:
        st = compute_infer_nav(self._store)

        for spec in self._items:
            row = self._rows.get(spec.key)
            if row is None:
                continue
            chip = st.get(spec.key, {"status": "off", "text": "—"})
            row.set_chip(
                status=str(chip.get("status", "off")),
                text=str(chip.get("text", "—")),
            )

        if self._snapshot is not None:
            self._apply_runtime_overlay(self._snapshot)

        self._fix_nav_list_height()

    def set_runtime_snapshot(self, snap: InferRuntimeSnapshot) -> None:
        self._snapshot = snap
        self._apply_runtime_overlay(snap)
        self._fix_nav_list_height()

    def refresh_compute(self) -> None:
        try:
            txt = runtime_summary_text()
        except Exception:
            txt = "Runtime summary unavailable."
        self.lbl_compute.setText(txt)

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # =========================================================
        # Card 0: Tips (TOP)
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
        self.btn_open_eval = QPushButton("Eval CSV", tips)
        self.btn_open_future = QPushButton("Future CSV", tips)
        self.btn_open_json = QPushButton("Summary JSON", tips)

        for b in (
            self.btn_open_run,
            self.btn_open_eval,
            self.btn_open_future,
            self.btn_open_json,
        ):
            _btn(b)
            b.setObjectName("miniAction")

        acts = QWidget(tips)
        acts_l = QGridLayout(acts)
        acts_l.setContentsMargins(0, 0, 0, 0)
        acts_l.setHorizontalSpacing(8)
        acts_l.setVerticalSpacing(8)

        acts_l.addWidget(self.btn_open_run, 0, 0)
        acts_l.addWidget(self.btn_open_eval, 0, 1)
        acts_l.addWidget(self.btn_open_future, 1, 0)
        acts_l.addWidget(self.btn_open_json, 1, 1)

        tips_l.addWidget(acts, 0)

        tips_hint = QLabel("Shortcuts update after a run.", tips)
        tips_hint.setObjectName("sumLine")
        tips_hint.setWordWrap(True)
        tips_l.addWidget(tips_hint, 0)

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
        self.list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        for spec in self._items:
            row = _NavRow(title=spec.title, parent=self.list)
            self._rows[spec.key] = row

            it = QListWidgetItem(self.list)
            it.setData(Qt.UserRole, spec.key)
            it.setSizeHint(row.sizeHint())

            self.list.addItem(it)
            self.list.setItemWidget(it, row)

        # keep checklist compact (no big empty area)
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
        
        # --- Scroll area for body (so long text won't cover nav) ---
        sc = QScrollArea(comp)

        body = QWidget(sc)
        sc.setWidget(body)
        sc.setObjectName("inferCompScroll")
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidgetResizable(True)
        # # sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # ensure the scroll area and its viewport don't paint white
        sc.setStyleSheet("QScrollArea { background: transparent; }")
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
        self.lbl_compute.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_compute.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )
        body_l.addWidget(self.lbl_compute, 0)
        
        notes = QLabel(
            "• Manifest can be empty if auto-resolve is enabled.\n"
            "• Custom NPZ requires inputs; targets are optional.\n"
            "• Outputs are written under the model run directory.",
            body,
        )
        notes.setWordWrap(True)
        notes.setObjectName("sumLine")
        body_l.addWidget(notes, 0)
        
        body_l.addStretch(1)
        
        # Let scroll area take remaining height of card
        sc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        comp_l.addWidget(sc, 1)
        
        root.addWidget(comp, 1)


        # Default selection
        if self.list.count() > 0:
            self.list.setCurrentRow(0)
            self._sync_selected_row()

    def _fix_nav_list_height(self) -> None:
        """
        Train-like: keep the nav list compact (fit visible items).
        """
        if self.list is None:
            return
        n = self.list.count()
        if n <= 0:
            return

        # Use a safe cap so the left column does not balloon.
        cap = 6
        show_n = min(n, cap)

        h = 0
        for i in range(show_n):
            it = self.list.item(i)
            if it is not None:
                h += self.list.sizeHintForIndex(self.list.indexFromItem(it)).height()

        # add padding + frame
        h += 12
        self.list.setFixedHeight(max(120, h))

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.list.currentItemChanged.connect(self._on_select)
        self._store.config_changed.connect(lambda _k: self.refresh())

        self.btn_open_run.clicked.connect(self.open_run_clicked.emit)
        self.btn_open_eval.clicked.connect(self.open_eval_clicked.emit)
        self.btn_open_future.clicked.connect(self.open_future_clicked.emit)
        self.btn_open_json.clicked.connect(self.open_json_clicked.emit)

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
        self.section_selected.emit(key)

    def _sync_selected_row(self) -> None:
        cur = self.list.currentItem()
        cur_key = ""
        if cur is not None:
            cur_key = str(cur.data(Qt.UserRole) or "")

        for k, row in self._rows.items():
            row.set_selected(k == cur_key)

    # -----------------------------------------------------------------
    # Runtime overlay (preview warnings)
    # -----------------------------------------------------------------
    def _apply_runtime_overlay(self, snap: InferRuntimeSnapshot) -> None:
        warn = list(getattr(snap, "warnings", []) or [])

        fatal = False
        for w in warn:
            lw = str(w).lower()
            if "missing model" in lw:
                fatal = True
            if "model path does not exist" in lw:
                fatal = True
            if "custom inputs" in lw and "required" in lw:
                fatal = True
            if "inputs npz does not exist" in lw:
                fatal = True

        row = self._rows.get("artifacts")
        if row is None:
            return

        if fatal:
            row.set_chip(status="err", text="Err")
            return

        if warn:
            row.set_chip(status="warn", text="Fix")
            return
        
    def set_shortcuts_enabled(
        self,
        *,
        run_ok: bool,
        eval_ok: bool,
        future_ok: bool,
        json_ok: bool,
    ) -> None:
        self.btn_open_run.setEnabled(bool(run_ok))
        self.btn_open_eval.setEnabled(bool(eval_ok))
        self.btn_open_future.setEnabled(bool(future_ok))
        self.btn_open_json.setEnabled(bool(json_ok))
