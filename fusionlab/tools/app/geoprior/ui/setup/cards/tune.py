# geoprior/ui/setup/cards/tune.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.tune

Tuning card (v3.2)

Goals
-----
- Modern, compact tuning surface.
- Expose tuning budget + search algo meta.
- Provide "hp_*" dialog entry points:
  - Architecture HP
  - Physics HP
  - Search algorithm
  - Export preferences
- Provide a readable search-space preview
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ....config.geoprior_config import default_tuner_search_space
from ....config.store import GeoConfigStore
from ....dialogs.hp_arch_dialog import ArchHPDialog
from ....dialogs.hp_export_dialog import ExportDialog
from ....dialogs.hp_phys_dialog import PhysHPDialog
from ....dialogs.hp_search_dialog import SearchAlgoDialog
from ..bindings import Binder
from .base import CardBase


class TuningCard(CardBase):
    """Tuning card (budget, algo, search space, export)."""

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="tuning",
            title="Tuning",
            subtitle=(
                "Define tuning budget, search method, and the "
                "hyperparameter search space."
            ),
            parent=parent,
        )
        self.store = store
        self.binder = binder

        self._build_ui()
        self._wire_store()
        self.refresh()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        lay = self.body_layout()

        top = QWidget(self)
        top_l = QGridLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setHorizontalSpacing(12)
        top_l.setVerticalSpacing(10)
        top_l.setColumnStretch(0, 1)
        top_l.setColumnStretch(1, 1)

        # -----------------------------
        # Left: budget + quick controls
        # -----------------------------
        box_budget = self._subpanel(
            "Budget",
            icon=QStyle.SP_ComputerIcon,
        )
        b_l = box_budget.layout()

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        self.sp_trials = QSpinBox(self)
        self.sp_trials.setRange(1, 999999)
        self.sp_trials.setSingleStep(1)
        self.binder.bind_spin_box(
            "tuner_max_trials",
            self.sp_trials,
        )

        grid.addWidget(self._lbl("Max trials:"), 0, 0)
        grid.addWidget(self.sp_trials, 0, 1)

        self.lbl_space_stats = QLabel("", self)
        self.lbl_space_stats.setWordWrap(True)
        self.lbl_space_stats.setStyleSheet(
            "color: rgba(30,30,30,0.72);",
        )

        grid.addWidget(self._lbl("Space:"), 1, 0)
        grid.addWidget(self.lbl_space_stats, 1, 1)

        b_l.addLayout(grid)

        btn_row = QWidget(self)
        btn_l = QHBoxLayout(btn_row)
        btn_l.setContentsMargins(0, 0, 0, 0)
        btn_l.setSpacing(8)

        self.btn_arch_hp = self._btn(
            "Architecture HP…",
            icon=QStyle.SP_FileDialogDetailedView,
        )
        self.btn_phys_hp = self._btn(
            "Physics HP…",
            icon=QStyle.SP_FileDialogDetailedView,
        )
        self.btn_reset_space = self._btn(
            "Reset space",
            icon=QStyle.SP_BrowserReload,
        )

        btn_l.addWidget(self.btn_arch_hp)
        btn_l.addWidget(self.btn_phys_hp)
        btn_l.addStretch(1)
        btn_l.addWidget(self.btn_reset_space)

        b_l.addWidget(btn_row)

        # -----------------------------
        # Right: search algo + export
        # -----------------------------
        box_algo = self._subpanel(
            "Search",
            icon=QStyle.SP_FileDialogContentsView,
        )
        a_l = box_algo.layout()

        self.lbl_algo = QLabel("", self)
        self.lbl_algo.setWordWrap(True)

        self.lbl_objective = QLabel("", self)
        self.lbl_objective.setWordWrap(True)
        self.lbl_objective.setStyleSheet(
            "color: rgba(30,30,30,0.72);",
        )

        row = QWidget(self)
        row_l = QGridLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setHorizontalSpacing(10)
        row_l.setVerticalSpacing(8)
        row_l.setColumnStretch(1, 1)

        row_l.addWidget(self._lbl("Algorithm:"), 0, 0)
        row_l.addWidget(self.lbl_algo, 0, 1)
        row_l.addWidget(self._lbl("Objective:"), 1, 0)
        row_l.addWidget(self.lbl_objective, 1, 1)

        a_l.addWidget(row)

        row2 = QWidget(self)
        row2_l = QHBoxLayout(row2)
        row2_l.setContentsMargins(0, 0, 0, 0)
        row2_l.setSpacing(8)

        self.btn_search = self._btn(
            "Edit search…",
            icon=QStyle.SP_FileDialogDetailedView,
        )
        self.btn_export = self._btn(
            "Export…",
            icon=QStyle.SP_DialogSaveButton,
        )
        self.btn_copy = self._btn(
            "Copy space",
            icon=QStyle.SP_DialogOpenButton,
        )

        row2_l.addWidget(self.btn_search)
        row2_l.addWidget(self.btn_export)
        row2_l.addStretch(1)
        row2_l.addWidget(self.btn_copy)

        a_l.addWidget(row2)

        top_l.addWidget(box_budget, 0, 0)
        top_l.addWidget(box_algo, 0, 1)

        lay.addWidget(top)

        # -----------------------------
        # Preview (collapsible)
        # -----------------------------
        self.exp_btn, self.exp_body = self._make_expander(
            "Preview search space",
        )
        lay.addWidget(self.exp_btn, 0)
        lay.addWidget(self.exp_body, 0)

        body_l = QVBoxLayout(self.exp_body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(8)

        self.ed_preview = QPlainTextEdit(self)
        self.ed_preview.setReadOnly(True)
        self.ed_preview.setMinimumHeight(140)
        self.ed_preview.setPlaceholderText(
            "Search space preview…",
        )
        body_l.addWidget(self.ed_preview)

        # -----------------------------
        # Header actions (mini)
        # -----------------------------
        act_search = self.add_action(
            text="Search",
            tip="Edit tuning search settings",
            icon=QStyle.SP_FileDialogDetailedView,
        )
        act_space = self.add_action(
            text="Space",
            tip="Edit tuner search space",
            icon=QStyle.SP_FileDialogListView,
        )
        act_export = self.add_action(
            text="Export",
            tip="Export preferences",
            icon=QStyle.SP_DialogSaveButton,
        )

        act_search.clicked.connect(self._open_search_dialog)
        act_space.clicked.connect(self._open_arch_dialog)
        act_export.clicked.connect(self._open_export_dialog)

        # Wiring
        self.btn_arch_hp.clicked.connect(self._open_arch_dialog)
        self.btn_phys_hp.clicked.connect(self._open_phys_dialog)
        self.btn_search.clicked.connect(self._open_search_dialog)
        self.btn_export.clicked.connect(self._open_export_dialog)
        self.btn_copy.clicked.connect(self._copy_space)
        self.btn_reset_space.clicked.connect(self._reset_space)

    def _subpanel(
        self,
        title: str,
        *,
        icon: QStyle.StandardPixmap,
    ) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("card")

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        head = QWidget(frame)
        head_l = QHBoxLayout(head)
        head_l.setContentsMargins(0, 0, 0, 0)
        head_l.setSpacing(8)

        ico = QLabel(head)
        ico.setPixmap(
            self.style()
            .standardIcon(icon)
            .pixmap(16, 16)
        )

        ttl = QLabel(str(title), head)
        ttl.setObjectName("cardTitle")

        head_l.addWidget(ico, 0)
        head_l.addWidget(ttl, 0)
        head_l.addStretch(1)

        outer.addWidget(head, 0)

        body = QWidget(frame)
        body_l = QVBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(8)
        outer.addWidget(body, 1)

        return frame

    def _lbl(self, text: str) -> QLabel:
        lab = QLabel(text, self)
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return lab

    def _btn(
        self,
        text: str,
        *,
        icon: QStyle.StandardPixmap,
    ) -> QPushButton:
        btn = QPushButton(str(text), self)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setIcon(self.style().standardIcon(icon))
        btn.setObjectName("miniAction")
        return btn

    def _make_expander(
        self,
        title: str,
    ) -> Tuple[QToolButton, QWidget]:
        btn = QToolButton(self)
        btn.setText(str(title))
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.setArrowType(Qt.RightArrow)
        btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon,
        )

        body = QWidget(self)
        body.setVisible(False)

        def _toggle(on: bool) -> None:
            body.setVisible(bool(on))
            btn.setArrowType(
                Qt.DownArrow if on else Qt.RightArrow
            )

        btn.toggled.connect(_toggle)
        return btn, body

    # -----------------------------------------------------------------
    # Store wiring
    # -----------------------------------------------------------------
    def _wire_store(self) -> None:
        self.store.config_changed.connect(self._on_changed)
        self.store.config_replaced.connect(self._on_replaced)

    def _on_changed(self, _keys: object) -> None:
        self.refresh()

    def _on_replaced(self, _cfg: object) -> None:
        self.refresh()

    # -----------------------------------------------------------------
    # Refresh
    # -----------------------------------------------------------------
    def refresh(self) -> None:
        cfg = self.store.cfg
        meta = self._get_meta()

        algo = str(meta.get("tuner_algo", "bayesian") or "bayesian")
        obj = str(
            meta.get("tuner_objective", "val_loss") or "val_loss"
        )
        direction = str(
            meta.get("tuner_direction", "min") or "min"
        )
        seed = meta.get("tuner_seed", None)

        trials = int(getattr(cfg, "tuner_max_trials", 0) or 0)

        space = getattr(cfg, "tuner_search_space", None)
        if not isinstance(space, dict):
            space = {}

        space_keys = list(space.keys())
        n_keys = len(space_keys)

        n_arch = self._count_keys(space_keys, ArchHPDialog._ARCH_KEYS)
        n_phys = self._count_keys(space_keys, PhysHPDialog._PHYS_KEYS)
        n_other = max(0, n_keys - n_arch - n_phys)

        self.lbl_algo.setText(algo)
        seed_txt = "" if seed is None else f", seed={seed}"
        self.lbl_objective.setText(
            f"{obj} ({direction}){seed_txt}",
        )

        self.lbl_space_stats.setText(
            f"{n_keys} keys  •  arch {n_arch}  •  "
            f"phys {n_phys}  •  other {n_other}"
        )

        self.badge(
            "trials",
            text=f"{trials} trials",
        )
        self.badge(
            "algo",
            text=str(algo),
        )
        self.badge(
            "space",
            text=f"{n_keys} keys",
        )

        warn = self._space_warn(space)
        if warn:
            self.badge(
                "note",
                text="check",
                accent="warn",
                tip=warn,
            )
        else:
            self.badge(
                "note",
                text="ready",
                accent="ok",
                tip="Tuning settings look consistent.",
            )

        self.ed_preview.setPlainText(
            self._format_space(space),
        )

    @staticmethod
    def _count_keys(
        keys: Sequence[str],
        group: Sequence[str],
    ) -> int:
        s = set(str(k) for k in keys)
        g = set(str(k) for k in group)
        return int(len(s.intersection(g)))

    @staticmethod
    def _space_warn(space: Dict[str, Any]) -> str:
        pde = space.get("pde_mode", None)
        if isinstance(pde, (list, tuple)) and "off" in pde:
            return (
                "pde_mode includes 'off' in tuning space. "
                "If you want full physics, keep pde_mode=['both']."
            )
        return ""

    # -----------------------------------------------------------------
    # Dialogs / actions
    # -----------------------------------------------------------------
    def _open_search_dialog(self) -> None:
        dlg = SearchAlgoDialog(store=self.store, parent=self)
        if dlg.exec_():
            self.refresh()

    def _open_arch_dialog(self) -> None:
        dlg = ArchHPDialog(store=self.store, parent=self)
        if dlg.exec_():
            self.refresh()

    def _open_phys_dialog(self) -> None:
        dlg = PhysHPDialog(store=self.store, parent=self)
        if dlg.exec_():
            self.refresh()

    def _open_export_dialog(self) -> None:
        dlg = ExportDialog(store=self.store, parent=self)
        if dlg.exec_():
            self.refresh()

    def _copy_space(self) -> None:
        cfg = self.store.cfg
        space = getattr(cfg, "tuner_search_space", None)
        if not isinstance(space, dict):
            space = {}

        try:
            txt = json.dumps(space, indent=2)
        except Exception:
            txt = str(space)

        QApplication.clipboard().setText(txt)
        QMessageBox.information(
            self,
            "Copied",
            "Tuner search space copied to clipboard.",
        )

    def _reset_space(self) -> None:
        msg = (
            "Reset tuner_search_space to defaults?\n\n"
            "This will overwrite the current search space."
        )
        ans = QMessageBox.question(
            self,
            "Reset search space",
            msg,
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return

        defaults = default_tuner_search_space()
        with self.store.batch():
            self.store.set("tuner_search_space", dict(defaults))

        self.refresh()

    # -----------------------------------------------------------------
    # Meta + formatting
    # -----------------------------------------------------------------
    def _get_meta(self) -> Dict[str, Any]:
        try:
            cur = getattr(self.store.cfg, "_meta", None)
        except Exception:
            cur = None
        if not isinstance(cur, dict):
            return {}
        return dict(cur)

    @staticmethod
    def _fmt_val(v: Any) -> str:
        if isinstance(v, dict):
            typ = str(v.get("type", "") or "")
            if typ == "float":
                mn = v.get("min_value", None)
                mx = v.get("max_value", None)
                st = v.get("step", None)
                samp = v.get("sampling", None)
                bits = []
                if mn is not None and mx is not None:
                    bits.append(f"{mn}..{mx}")
                if st is not None:
                    bits.append(f"step={st}")
                if samp:
                    bits.append(str(samp))
                core = ", ".join(bits) if bits else "spec"
                return f"float({core})"
            if typ == "bool":
                return "bool(tune)"
            return f"spec({typ or 'dict'})"

        if isinstance(v, (list, tuple)):
            if len(v) <= 6:
                return "[" + ", ".join(map(str, v)) + "]"
            return f"[{len(v)} choices]"

        return str(v)

    def _format_space(self, space: Dict[str, Any]) -> str:
        keys = sorted(str(k) for k in space.keys())

        arch = []
        phys = []
        other = []

        arch_set = set(str(k) for k in ArchHPDialog._ARCH_KEYS)
        phys_set = set(str(k) for k in PhysHPDialog._PHYS_KEYS)

        for k in keys:
            v = space.get(k, None)
            line = f"- {k}: {self._fmt_val(v)}"
            if k in arch_set:
                arch.append(line)
            elif k in phys_set:
                phys.append(line)
            else:
                other.append(line)

        out = []
        out.append("[Architecture]")
        out.extend(arch or ["- (none)"])
        out.append("")
        out.append("[Physics]")
        out.extend(phys or ["- (none)"])
        out.append("")
        out.append("[Other]")
        out.extend(other or ["- (none)"])

        return "\n".join(out)
