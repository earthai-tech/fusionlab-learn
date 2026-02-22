# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.censoring

Censoring & H-field (modern UX).

Goals
-----
- Avoid raw JSON typing as the primary workflow.
- Provide a rule table driven by dataset columns.
- Keep full-fidelity dicts (preserve extra keys).
- Offer an optional "Raw JSON" advanced editor + preview.

Store keys
----------
- censoring_specs: List[Dict[str, Any]]
- use_effective_h_field: bool
- include_censor_flags_as_dynamic: bool
- include_censor_flags_as_future: bool
"""

from __future__ import annotations

import json
import pandas as pd
from copy import deepcopy
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
)

from ....dialogs.feature_dialog import FeatureConfigDialog
from ....config.store import GeoConfigStore
from ..bindings import Binder
from .base import CardBase

_DIR_ITEMS = [
    ("right", "right"),
    ("left", "left"),
]

_MODE_ITEMS = [
    ("clip", "clip"),
    ("cap_minus_eps", "cap_minus_eps"),
    ("nan_if_censored", "nan_if_censored"),
]


class _Expander(QWidget):
    def __init__(
        self,
        title: str,
        *,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.btn = QToolButton(self)
        self.btn.setCheckable(True)
        self.btn.setChecked(False)
        self.btn.setText(str(title))
        self.btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.btn.setArrowType(Qt.RightArrow)

        self.body = QWidget(self)
        self.body.setVisible(False)

        self.body_l = QVBoxLayout(self.body)
        self.body_l.setContentsMargins(8, 6, 8, 6)
        self.body_l.setSpacing(6)

        self.btn.toggled.connect(self._toggle)

        root.addWidget(self.btn, 0)
        root.addWidget(self.body, 0)

    def _toggle(self, on: bool) -> None:
        self.body.setVisible(bool(on))
        self.btn.setArrowType(
            Qt.DownArrow if on else Qt.RightArrow
        )

class _ColumnPicker(QDialog):
    def __init__(
        self,
        columns: List[str],
        *,
        title: str,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(title)
        self.resize(420, 360)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.ed = QLineEdit(self)
        self.ed.setPlaceholderText("Search…")
        self.ed.setClearButtonEnabled(True)

        self.list = QListWidget(self)
        self.list.setAlternatingRowColors(True)

        self._items: List[str] = list(columns or [])
        for c in self._items:
            QListWidgetItem(str(c), self.list)

        root.addWidget(self.ed, 0)
        root.addWidget(self.list, 1)

        self.ed.textChanged.connect(self._filter)
        self.list.itemDoubleClicked.connect(
            lambda _it: self.accept(),
        )

    def _filter(self, text: str) -> None:
        q = (text or "").strip().lower()
        for i in range(self.list.count()):
            it = self.list.item(i)
            t = (it.text() or "").lower()
            it.setHidden(bool(q) and q not in t)

    def selected(self) -> str:
        it = self.list.currentItem()
        if it is None:
            return ""
        return str(it.text() or "")

class CensoringCard(CardBase):
    """Censoring & H-field (store-driven)."""

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="censoring",
            title="Censoring & H-field",
            subtitle=(
                "Define censor rules for H-field and "
                "optional derived columns (flags/effective)."
            ),
            parent=parent,
        )

        self.store = store
        self.binder = binder

        self._cols: List[str] = []
        self._specs: List[Dict[str, Any]] = []
        self._updating = False

        self._build()
        self._wire()
        self.refresh()

    def _pick_column(self, row: int) -> None:
        dlg = _ColumnPicker(
            self._cols,
            title="Pick column",
            parent=self,
        )
        if dlg.exec_() != dlg.Accepted:
            return

        col = dlg.selected().strip()
        if not col:
            return

        cmb = self._col_combo(row)
        if cmb is None:
            return

        with QSignalBlocker(cmb):
            self._set_combo_text(cmb, col)

        self._commit_row(row)


    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------
    def set_dataset_columns(self, columns: List[str]) -> None:
        cols = [str(c) for c in (columns or [])]
        self._cols = cols
        self._refresh_col_widgets()

    # ---------------------------------------------------------------
    # UI
    # ---------------------------------------------------------------
    def _build(self) -> None:
        body = self.body_layout()

        grid = QWidget(self)
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)
        g.setColumnStretch(0, 2)
        g.setColumnStretch(1, 1)

        self.grp_rules = self._build_rules(grid)
        self.grp_preview = self._build_preview(grid)

        g.addWidget(self.grp_rules, 0, 0)
        g.addWidget(self.grp_preview, 0, 1)

        body.addWidget(grid, 0)

    def _build_rules(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Rules", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # Flags row (compact)
        flags = QWidget(box)
        fl = QHBoxLayout(flags)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(10)

        self.chk_eff = QCheckBox("Use effective H-field", box)
        self.chk_dyn = QCheckBox("Flags as dynamic", box)
        self.chk_fut = QCheckBox("Flags as future", box)

        self.binder.bind_checkbox(
            "use_effective_h_field",
            self.chk_eff,
        )
        self.binder.bind_checkbox(
            "include_censor_flags_as_dynamic",
            self.chk_dyn,
        )
        self.binder.bind_checkbox(
            "include_censor_flags_as_future",
            self.chk_fut,
        )

        fl.addWidget(self.chk_eff, 0)
        fl.addWidget(self.chk_dyn, 0)
        fl.addWidget(self.chk_fut, 0)
        fl.addStretch(1)

        lay.addWidget(flags, 0)

        # Toolbar
        bar = QWidget(box)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)

        self.btn_add = QPushButton("Add rule", box)
        self.btn_dup = QPushButton("Duplicate", box)
        self.btn_clear = QPushButton("Clear", box)

        self.btn_add.setCursor(Qt.PointingHandCursor)
        self.btn_dup.setCursor(Qt.PointingHandCursor)
        self.btn_clear.setCursor(Qt.PointingHandCursor)

        bl.addWidget(self.btn_add, 0)
        bl.addWidget(self.btn_dup, 0)
        bl.addWidget(self.btn_clear, 0)
        bl.addStretch(1)

        lay.addWidget(bar, 0)
        self.btn_import = QPushButton(
            "Import…",
            box,
        )
        self.btn_import.setCursor(
            Qt.PointingHandCursor
        )
        
        bl.addWidget(self.btn_import, 0)


        # Table
        self.tbl = QTableWidget(box)
        self.tbl.setColumnCount(6)
        self.tbl.setHorizontalHeaderLabels(
            [
                "Column",
                "Cap",
                "Direction",
                "Mode",
                "Flag thr",
                "",
            ]
        )
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setShowGrid(False)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setSelectionBehavior(
            self.tbl.SelectRows
        )
        self.tbl.setSelectionMode(
            self.tbl.SingleSelection
        )
        self.tbl.setEditTriggers(
            self.tbl.NoEditTriggers
        )
        self.tbl.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.tbl.setColumnWidth(0, 180)
        self.tbl.setColumnWidth(1, 90)
        self.tbl.setColumnWidth(2, 90)
        self.tbl.setColumnWidth(3, 140)
        self.tbl.setColumnWidth(4, 80)
        self.tbl.setColumnWidth(5, 34)

        lay.addWidget(self.tbl, 1)

        hint = QLabel(
            "Tip: prefer selecting columns from the list to "
            "avoid typos. Advanced JSON keeps extra keys "
            "like suffixes and imputation rules.",
            box,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "color: rgba(30,30,30,0.70);"
        )
        lay.addWidget(hint, 0)

        # Advanced raw JSON
        self.adv = _Expander("Advanced: raw JSON", parent=box)
        lay.addWidget(self.adv, 0)

        self.ed_json = QPlainTextEdit(self.adv.body)
        self.ed_json.setMinimumHeight(140)
        self.ed_json.setPlaceholderText(
            "List of dicts (JSON). Example:\n"
            "[\n"
            "  {\n"
            '    "col": "soil_thickness",\n'
            '    "cap": 30.0,\n'
            '    "direction": "right",\n'
            '    "eff_mode": "clip",\n'
            '    "flag_threshold": 0.5\n'
            "  }\n"
            "]"
        )

        json_bar = QWidget(self.adv.body)
        jbl = QHBoxLayout(json_bar)
        jbl.setContentsMargins(0, 0, 0, 0)
        jbl.setSpacing(6)

        self.btn_apply_json = QPushButton(
            "Apply JSON",
            self.adv.body,
        )
        self.btn_copy_json = QPushButton(
            "Copy",
            self.adv.body,
        )
        self.btn_apply_json.setCursor(
            Qt.PointingHandCursor
        )
        self.btn_copy_json.setCursor(
            Qt.PointingHandCursor
        )

        jbl.addStretch(1)
        jbl.addWidget(self.btn_copy_json, 0)
        jbl.addWidget(self.btn_apply_json, 0)

        self.adv.body_l.addWidget(self.ed_json, 1)
        self.adv.body_l.addWidget(json_bar, 0)

        return box

    def _build_preview(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Preview", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.lbl_note = QLabel("", box)
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setStyleSheet(
            "color: rgba(30,30,30,0.72);"
        )
        lay.addWidget(self.lbl_note, 0)

        self.row_eff = self._kv_row(box, "Effective col")
        self.row_flag = self._kv_row(box, "Flag col")
        self.row_n = self._kv_row(box, "Rules")

        lay.addWidget(self.row_eff["w"], 0)
        lay.addWidget(self.row_flag["w"], 0)
        lay.addWidget(self.row_n["w"], 0)

        self.v_eff = self.row_eff["v"]
        self.v_flag = self.row_flag["v"]
        self.v_n = self.row_n["v"]

        self.badge(
            "status",
            text="OK",
            accent="ok",
            tip="Censoring setup",
        )

        return box

    def _kv_row(self, parent: QWidget, key: str) -> dict:
        w = QWidget(parent)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        k = QLabel(f"{key}:", w)
        k.setStyleSheet(
            "color: rgba(30,30,30,0.70);"
        )

        v = QLabel("—", w)
        v.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        v.setStyleSheet(
            "\n".join(
                [
                    "font-weight: 600;",
                    "padding: 3px 10px;",
                    "border-radius: 12px;",
                    "border: 1px solid",
                    "  rgba(0,0,0,0.12);",
                    "background: rgba(0,0,0,0.03);",
                ]
            )
        )

        lay.addWidget(k, 0)
        lay.addWidget(v, 1)
        return {"w": w, "v": v}

    # ---------------------------------------------------------------
    # Wiring
    # ---------------------------------------------------------------
    def _wire(self) -> None:
        self.btn_add.clicked.connect(self._on_add)
        self.btn_dup.clicked.connect(self._on_dup)
        self.btn_clear.clicked.connect(self._on_clear)

        self.btn_copy_json.clicked.connect(
            self._on_copy_json,
        )
        self.btn_apply_json.clicked.connect(
            self._on_apply_json,
        )

        self.chk_eff.toggled.connect(self._sync_preview)
        self.chk_dyn.toggled.connect(self._sync_preview)
        self.chk_fut.toggled.connect(self._sync_preview)

        self.store.config_changed.connect(
            lambda _k: self.refresh(),
        )
        self.store.config_replaced.connect(
            lambda _cfg: self.refresh(),
        )
        self.btn_import.clicked.connect(
            self._import_from_feature_dialog,
        )


    # ---------------------------------------------------------------
    # Data <-> UI
    # ---------------------------------------------------------------
    def _import_from_feature_dialog(self) -> None:
        base_cfg = getattr(self.store.cfg, "_base_cfg", {})
        base_cfg = base_cfg or {}

        overrides = self.store.snapshot_overrides()

        df = None
        try:
            

            if self._cols:
                df = pd.DataFrame(columns=self._cols)
        except Exception:
            df = None

        csv_path = ""
        try:
            p = getattr(self.store.cfg, "dataset_path", "")
            if p:
                csv_path = str(p)
        except Exception:
            csv_path = ""

        dlg = FeatureConfigDialog(
            csv_path=csv_path,
            base_cfg=base_cfg,
            current_overrides=overrides,
            parent=self,
            df=df,
        )

        if dlg.exec_() != dlg.Accepted:
            return

        ov = dlg.get_overrides() or {}
        patch: Dict[str, Any] = {}

        # If dialog provides censoring specs,
        # map them directly.
        if "CENSORING_SPECS" in ov:
            patch["censoring_specs"] = ov["CENSORING_SPECS"]

        # Flags options (if exposed by dialog)
        if "INCLUDE_CENSOR_FLAGS_AS_FUTURE" in ov:
            patch["include_censor_flags_as_future"] = (
                ov["INCLUDE_CENSOR_FLAGS_AS_FUTURE"]
            )

        if patch:
            self.store.patch(patch)

    def refresh(self) -> None:
        self._updating = True
        try:
            cfg = self.store.cfg
            specs = getattr(cfg, "censoring_specs", None)
            self._specs = self._norm_specs(specs)

            self._rebuild_table()
            self._sync_preview()
            self._sync_json_editor()
        finally:
            self._updating = False

    def _norm_specs(self, specs: Any) -> List[Dict[str, Any]]:
        if not specs:
            return []
        if isinstance(specs, dict):
            return [dict(specs)]
        out: List[Dict[str, Any]] = []
        for x in list(specs):
            if isinstance(x, dict):
                out.append(dict(x))
        return out

    def _rebuild_table(self) -> None:
        with QSignalBlocker(self.tbl):
            self.tbl.setRowCount(0)

            for spec in self._specs:
                self._append_row(spec)
            
            for rr in range(self.tbl.rowCount()):
                self._apply_row_validation(rr)

            self.tbl.resizeRowsToContents()
            self._refresh_col_widgets()

        n = len(self._specs)
        self.btn_dup.setEnabled(n > 0)
        self.btn_clear.setEnabled(n > 0)

    def _append_row(self, spec: Dict[str, Any]) -> None:
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)

        # Column
        cmb_col = QComboBox(self.tbl)
        cmb_col.setEditable(True)
        cmb_col.setInsertPolicy(
            QComboBox.NoInsert
        )
        cmb_col.setMinimumWidth(160)
        self._fill_columns_combo(cmb_col)

        col = str(spec.get("col", "") or "").strip()
        self._set_combo_text(cmb_col, col)
        
        col_cell = QWidget(self.tbl)
        col_l = QHBoxLayout(col_cell)
        col_l.setContentsMargins(0, 0, 0, 0)
        col_l.setSpacing(6)
        
        btn_pick = QToolButton(col_cell)
        btn_pick.setText("…")
        btn_pick.setToolTip("Pick column…")
        btn_pick.setCursor(Qt.PointingHandCursor)
        btn_pick.setAutoRaise(True)
        btn_pick.setFixedWidth(22)
        
        col_l.addWidget(cmb_col, 1)
        col_l.addWidget(btn_pick, 0)
        
        btn_pick.clicked.connect(
            lambda _=False, row=r: self._pick_column(row),
        )


        # Cap
        sp_cap = QDoubleSpinBox(self.tbl)
        sp_cap.setDecimals(4)
        sp_cap.setRange(0.0, 1e12)
        sp_cap.setSingleStep(0.5)
        sp_cap.setValue(
            float(spec.get("cap", 0.0) or 0.0)
        )

        # Direction
        cmb_dir = QComboBox(self.tbl)
        for t, v in _DIR_ITEMS:
            cmb_dir.addItem(str(t), str(v))
        self._set_combo_data(
            cmb_dir,
            str(spec.get("direction", "right")),
        )

        # Mode
        cmb_mode = QComboBox(self.tbl)
        for t, v in _MODE_ITEMS:
            cmb_mode.addItem(str(t), str(v))
        self._set_combo_data(
            cmb_mode,
            str(spec.get("eff_mode", "clip")),
        )

        # Flag threshold
        sp_thr = QDoubleSpinBox(self.tbl)
        sp_thr.setDecimals(3)
        sp_thr.setRange(0.0, 1.0)
        sp_thr.setSingleStep(0.05)
        sp_thr.setValue(
            float(
                spec.get("flag_threshold", 0.5)
                or 0.0
            )
        )

        # Remove button
        btn_rm = QToolButton(self.tbl)
        btn_rm.setCursor(Qt.PointingHandCursor)
        btn_rm.setToolTip("Remove rule")
        btn_rm.setText("✕")
        btn_rm.setAutoRaise(True)

        # Place widgets
        self.tbl.setCellWidget(r, 0, col_cell)
        self.tbl.setCellWidget(r, 1, sp_cap)
        self.tbl.setCellWidget(r, 2, cmb_dir)
        self.tbl.setCellWidget(r, 3, cmb_mode)
        self.tbl.setCellWidget(r, 4, sp_thr)
        self.tbl.setCellWidget(r, 5, btn_rm)
        self._apply_row_validation(r)


        # Keep a stable height
        it = QTableWidgetItem("")
        it.setFlags(Qt.ItemIsEnabled)
        self.tbl.setItem(r, 5, it)

        # Wire row edits -> store
        cmb_col.currentIndexChanged.connect(
            lambda _i, row=r: self._commit_row(row),
        )
        if cmb_col.lineEdit() is not None:
            cmb_col.lineEdit().editingFinished.connect(
                lambda row=r: self._commit_row(row),
            )

        sp_cap.editingFinished.connect(
            lambda row=r: self._commit_row(row),
        )
        cmb_dir.currentIndexChanged.connect(
            lambda _i, row=r: self._commit_row(row),
        )
        cmb_mode.currentIndexChanged.connect(
            lambda _i, row=r: self._commit_row(row),
        )
        sp_thr.editingFinished.connect(
            lambda row=r: self._commit_row(row),
        )

        btn_rm.clicked.connect(
            lambda _=False, row=r: self._on_remove(row),
        )

    def _refresh_col_widgets(self) -> None:
        for r in range(self.tbl.rowCount()):
            cmb = self._col_combo(r)
            if cmb is None:
                continue

            cur = str(cmb.currentText() or "").strip()
            with QSignalBlocker(cmb):
                self._fill_columns_combo(cmb)
                self._set_combo_text(cmb, cur)

            self._apply_row_validation(r)

    def _fill_columns_combo(self, cmb: QComboBox) -> None:
        cur = str(cmb.currentText() or "").strip()
        cmb.clear()
        cmb.addItem("— Select —", "")
        for c in (self._cols or []):
            cmb.addItem(str(c), str(c))
        if cur:
            self._set_combo_text(cmb, cur)

    def _set_combo_text(self, cmb: QComboBox, text: str) -> None:
        t = str(text or "").strip()
        if not t:
            cmb.setCurrentIndex(0)
            return
        idx = cmb.findText(t)
        if idx >= 0:
            cmb.setCurrentIndex(idx)
            return
        if cmb.isEditable():
            cmb.setEditText(t)

    def _set_combo_data(self, cmb: QComboBox, data: str) -> None:
        d = str(data or "").strip()
        for i in range(cmb.count()):
            if str(cmb.itemData(i) or "") == d:
                cmb.setCurrentIndex(i)
                return
        cmb.setCurrentIndex(0)

    def _col_combo(self, row: int) -> Optional[QComboBox]:
        cell = self.tbl.cellWidget(row, 0)
        if isinstance(cell, QWidget):
            cmb = cell.findChild(QComboBox)
            if isinstance(cmb, QComboBox):
                return cmb
        return None

    def _row_is_valid(self, row: int) -> bool:
        cmb = self._col_combo(row)
        if cmb is None:
            return False

        col = str(cmb.currentText() or "").strip()
        if not col:
            return False

        if self._cols and col not in self._cols:
            return False

        return True


    def _apply_row_validation(self, row: int) -> None:
        ok = self._row_is_valid(row)
        cmb = self._col_combo(row)

        if cmb is not None:
            if ok:
                cmb.setStyleSheet("")
                cmb.setToolTip("")
            else:
                cmb.setStyleSheet(
                    "\n".join(
                        [
                            "QComboBox {",
                            "  border: 1px solid",
                            "    rgba(210,0,0,0.55);",
                            "  border-radius: 6px;",
                            "}",
                        ]
                    )
                )
                cmb.setToolTip(
                    "Pick a valid dataset column."
                )

    def _commit_row(self, row: int) -> None:
        if self._updating:
            return
        if row < 0 or row >= self.tbl.rowCount():
            return
        if row >= len(self._specs):
            return
    
        spec = dict(self._specs[row])
    
        sp_cap = self.tbl.cellWidget(row, 1)
        cmb_dir = self.tbl.cellWidget(row, 2)
        cmb_mode = self.tbl.cellWidget(row, 3)
        sp_thr = self.tbl.cellWidget(row, 4)
    
        cmb_col = self._col_combo(row)
        if isinstance(cmb_col, QComboBox):
            col = str(cmb_col.currentText() or "").strip()
            spec["col"] = col
    
        if isinstance(sp_cap, QDoubleSpinBox):
            spec["cap"] = float(sp_cap.value())
    
        if isinstance(cmb_dir, QComboBox):
            v = cmb_dir.currentData()
            spec["direction"] = str(v or "right")
    
        if isinstance(cmb_mode, QComboBox):
            v = cmb_mode.currentData()
            spec["eff_mode"] = str(v or "clip")
    
        if isinstance(sp_thr, QDoubleSpinBox):
            spec["flag_threshold"] = float(sp_thr.value())
    
        self._specs[row] = spec
        self._apply_row_validation(row)
        self._push_specs()

    def _push_specs(self) -> None:
        if self._updating:
            return
        cur = getattr(self.store.cfg, "censoring_specs", [])
        if isinstance(cur, dict):
            cur = [cur]
        new = deepcopy(self._specs)

        if list(cur or []) == list(new or []):
            self._sync_preview()
            self._sync_json_editor()
            return

        try:
            self.store.patch({"censoring_specs": new})
        except Exception:
            return

        self._sync_preview()
        self._sync_json_editor()


    def _sync_preview(self) -> None:
        cfg = self.store.cfg
        specs = getattr(cfg, "censoring_specs", None)
        specs_n = self._norm_specs(specs)
        any_bad = False
        for r in range(self.tbl.rowCount()):
            if not self._row_is_valid(r):
                any_bad = True
                break

        n = len(specs_n)
        self.v_n.setText(str(n))

        if n == 0:
            self.v_eff.setText("—")
            self.v_flag.setText("—")
            msg = (
                "No censor rules yet. Add one rule to "
                "define the H-field (and optional flags)."
            )
            self.badge(
                "status",
                text="Missing",
                accent="warn",
                tip=msg,
            )
            self.lbl_note.setText(msg)
            return

        first = specs_n[0]
        col = str(first.get("col", "") or "").strip()

        eff_suffix = str(first.get("eff_suffix", "_eff"))
        flag_suffix = str(
            first.get("flag_suffix", "_censored")
        )

        use_eff = bool(
            getattr(cfg, "use_effective_h_field", True)
        )

        eff_col = "—"
        if use_eff and col:
            eff_col = f"{col}{eff_suffix}"

        flag_col = "—"
        if col:
            flag_col = f"{col}{flag_suffix}"

        self.v_eff.setText(eff_col)
        self.v_flag.setText(flag_col)

        msg = (
            "Censoring generates a flag column "
            f"({flag_suffix}) and optionally an "
            f"effective column ({eff_suffix})."
        )
        self.lbl_note.setText(msg)

        ok = bool(col)
        if any_bad:
            self.badge(
                "status",
                text="Check",
                accent="warn",
                tip="Some rules have invalid columns.",
            )
        else:
            self.badge(
                "status",
                text="OK" if ok else "Check",
                accent="ok" if ok else "warn",
                tip=(
                    "Set the rule column (H-field) to enable"
                    " effective/flag derived columns."
                ),
            )


    def _sync_json_editor(self) -> None:
        cfg = self.store.cfg
        specs = getattr(cfg, "censoring_specs", [])
        try:
            txt = json.dumps(specs, indent=2)
        except Exception:
            txt = str(specs)

        with QSignalBlocker(self.ed_json):
            self.ed_json.setPlainText(txt)

    # ---------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------
    def _template_spec(self) -> Dict[str, Any]:
        cfg = self.store.cfg
        specs = getattr(cfg, "censoring_specs", None)
        items = self._norm_specs(specs)

        if items:
            base = dict(items[0])
        else:
            base = {}

        base.setdefault("col", "")
        base.setdefault("cap", 30.0)
        base.setdefault("direction", "right")
        base.setdefault("eff_mode", "clip")
        base.setdefault("flag_threshold", 0.5)

        base.setdefault("flag_suffix", "_censored")
        base.setdefault("eff_suffix", "_eff")

        return base

    def _on_add(self) -> None:
        self._specs = deepcopy(self._specs)
        self._specs.append(self._template_spec())
        self._rebuild_table()
        self._push_specs()

    def _on_dup(self) -> None:
        if not self._specs:
            return
        row = int(self.tbl.currentRow())
        if row < 0 or row >= len(self._specs):
            row = len(self._specs) - 1

        self._specs = deepcopy(self._specs)
        self._specs.insert(row + 1, dict(self._specs[row]))
        self._rebuild_table()
        self._push_specs()

    def _on_clear(self) -> None:
        self._specs = []
        self._rebuild_table()
        self._push_specs()

    def _on_remove(self, row: int) -> None:
        if row < 0 or row >= len(self._specs):
            return
        self._specs = deepcopy(self._specs)
        self._specs.pop(row)
        self._rebuild_table()
        self._push_specs()

    def _on_copy_json(self) -> None:
        txt = str(self.ed_json.toPlainText() or "").strip()
        if not txt:
            return
        try:
            QApplication.clipboard().setText(txt)
        except Exception:
            return

    def _on_apply_json(self) -> None:
        txt = str(self.ed_json.toPlainText() or "").strip()
        if not txt:
            return
        try:
            payload = json.loads(txt)
        except Exception:
            return

        if not isinstance(payload, list):
            return

        specs: List[Dict[str, Any]] = []
        for x in payload:
            if isinstance(x, dict):
                specs.append(dict(x))

        try:
            self.store.patch({"censoring_specs": specs})
        except Exception:
            return
