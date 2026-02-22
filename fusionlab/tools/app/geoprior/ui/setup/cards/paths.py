# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.paths

Project & paths card.

Modern UX:
- Project paths picker rows (Browse/Open/Copy).
- Audit stages as presets + checkboxes.
- Raw audit string stays available in Advanced.
- Stage-1 reuse policy in an expander.

The card writes to GeoConfigStore via Binder / store.patch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QFileDialog,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from .base import CardBase
from ..bindings import Binder
from ....config.store import GeoConfigStore


_AUD_KEYS = ("stage1", "stage2", "stage3", "stage4")


def _as_str_path(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, Path):
        return str(v)
    return str(v)


def _norm_audit_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace(" ", "")
    return t


def _parse_audit(text: str) -> Tuple[bool, Set[str]]:
    t = _norm_audit_text(text)
    if not t:
        return (False, set())
    if t == "*":
        return (True, set(_AUD_KEYS))

    toks = [x for x in t.split(",") if x]
    out: Set[str] = set()
    for x in toks:
        if x in _AUD_KEYS:
            out.add(x)
    return (False, out)


def _encode_audit(all_on: bool, sel: Set[str]) -> str:
    if all_on:
        return "*"
    if not sel:
        return ""
    ordered = [k for k in _AUD_KEYS if k in sel]
    return ",".join(ordered)


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

        self.body_l = QGridLayout(self.body)
        self.body_l.setContentsMargins(8, 6, 8, 6)
        self.body_l.setHorizontalSpacing(10)
        self.body_l.setVerticalSpacing(6)

        self.btn.toggled.connect(self._toggle)

        root.addWidget(self.btn, 0)
        root.addWidget(self.body, 0)

    def _toggle(self, on: bool) -> None:
        self.body.setVisible(bool(on))
        self.btn.setArrowType(
            Qt.DownArrow if on else Qt.RightArrow
        )


class ProjectPathsCard(CardBase):
    """Project & paths (store-driven via Binder)."""

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="paths",
            title="Project & paths",
            subtitle=(
                "Select city, dataset and results folder. "
                "Configure audit and preprocess reuse."
            ),
            parent=parent,
        )

        self.store = store
        self.binder = binder

        self._audit_boxes: Dict[str, QCheckBox] = {}
        self._audit_block = False

        self._build()
        self._wire()
        self.refresh()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build(self) -> None:
        body = self.body_layout()

        grid = QWidget(self)
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)

        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

        self.grp_proj = self._build_project_group(grid)
        self.grp_audit = self._build_audit_group(grid)

        g.addWidget(self.grp_proj, 0, 0)
        g.addWidget(self.grp_audit, 0, 1)

        body.addWidget(grid, 0)

    def _build_project_group(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Project", parent)
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)

        lay.setColumnStretch(1, 1)

        # City
        self.ed_city = QLineEdit(box)
        self.ed_city.setPlaceholderText("e.g. nansha")

        lay.addWidget(QLabel("City:", box), 0, 0)
        lay.addWidget(self.ed_city, 0, 1)

        # Dataset row
        self.ed_dataset = QLineEdit(box)
        self.ed_dataset.setReadOnly(True)

        self.btn_ds_browse = self._mini_btn(
            box,
            text="Browse",
            icon=QStyle.SP_DirOpenIcon,
            tip="Select dataset file",
        )
        self.btn_ds_open = self._mini_btn(
            box,
            text="Open",
            icon=QStyle.SP_DialogOpenButton,
            tip="Open dataset in OS",
        )
        self.btn_ds_copy = self._mini_btn(
            box,
            text="Copy",
            icon=QStyle.SP_DialogSaveButton,
            tip="Copy dataset path",
        )

        row_ds = self._path_row(
            box,
            self.ed_dataset,
            self.btn_ds_browse,
            self.btn_ds_open,
            self.btn_ds_copy,
        )

        lay.addWidget(QLabel("Dataset:", box), 1, 0)
        lay.addWidget(row_ds, 1, 1)

        # Results row
        self.ed_results = QLineEdit(box)
        self.ed_results.setReadOnly(False)

        self.btn_res_browse = self._mini_btn(
            box,
            text="Browse",
            icon=QStyle.SP_DirOpenIcon,
            tip="Select results folder",
        )
        self.btn_res_open = self._mini_btn(
            box,
            text="Open",
            icon=QStyle.SP_DialogOpenButton,
            tip="Open results folder",
        )
        self.btn_res_copy = self._mini_btn(
            box,
            text="Copy",
            icon=QStyle.SP_DialogSaveButton,
            tip="Copy results path",
        )

        row_res = self._path_row(
            box,
            self.ed_results,
            self.btn_res_browse,
            self.btn_res_open,
            self.btn_res_copy,
        )

        lay.addWidget(QLabel("Results:", box), 2, 0)
        lay.addWidget(row_res, 2, 1)

        # Status line
        self.lbl_paths_hint = QLabel("", box)
        self.lbl_paths_hint.setObjectName("pathsHint")
        self.lbl_paths_hint.setWordWrap(True)

        lay.addWidget(self.lbl_paths_hint, 3, 0, 1, 2)

        box.setStyleSheet(
            "\n".join(
                [
                    "QLabel#pathsHint {",
                    "  color: rgba(30,30,30,0.72);",
                    "  font-size: 11px;",
                    "}",
                ]
            )
        )

        # Bindings
        self.binder.bind_line_edit("city", self.ed_city)

        self.binder.bind_line_edit(
            "dataset_path",
            self.ed_dataset,
            from_store=_as_str_path,
            on="editingFinished",
        )

        self.binder.bind_line_edit(
            "results_root",
            self.ed_results,
            from_store=_as_str_path,
            on="editingFinished",
        )

        return box

    def _build_audit_group(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Auditing & preprocess", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # -------------------------
        # Audit stages (preset)
        # -------------------------
        row0 = QWidget(box)
        r0 = QHBoxLayout(row0)
        r0.setContentsMargins(0, 0, 0, 0)
        r0.setSpacing(8)

        r0.addWidget(QLabel("Audit:", box), 0)

        self.cmb_audit = QComboBox(box)
        self.cmb_audit.addItem("Off", "off")
        self.cmb_audit.addItem("Preprocess", "stage1")
        self.cmb_audit.addItem("Train + tune", "stage2_3")
        self.cmb_audit.addItem("All", "all")
        self.cmb_audit.addItem("Custom", "custom")

        r0.addWidget(self.cmb_audit, 1)

        self.btn_audit_reset = self._mini_btn(
            box,
            text="All",
            icon=QStyle.SP_BrowserReload,
            tip="Set audit to all stages",
        )
        r0.addWidget(self.btn_audit_reset, 0)

        lay.addWidget(row0, 0)

        # Checkboxes grid
        grid = QWidget(box)
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(6)

        self._audit_boxes["stage1"] = QCheckBox(
            "Stage-1 (preprocess)",
            box,
        )
        self._audit_boxes["stage2"] = QCheckBox(
            "Stage-2 (train)",
            box,
        )
        self._audit_boxes["stage3"] = QCheckBox(
            "Stage-3 (tune)",
            box,
        )
        self._audit_boxes["stage4"] = QCheckBox(
            "Stage-4 (infer)",
            box,
        )

        g.addWidget(self._audit_boxes["stage1"], 0, 0)
        g.addWidget(self._audit_boxes["stage2"], 0, 1)
        g.addWidget(self._audit_boxes["stage3"], 1, 0)
        g.addWidget(self._audit_boxes["stage4"], 1, 1)

        lay.addWidget(grid, 0)

        self.lbl_audit_hint = QLabel("", box)
        self.lbl_audit_hint.setWordWrap(True)
        self.lbl_audit_hint.setObjectName("auditHint")

        lay.addWidget(self.lbl_audit_hint, 0)

        # -------------------------
        # Advanced: raw string
        # -------------------------
        self.exp_audit = _Expander(
            "Advanced audit string",
            parent=box,
        )

        self.ed_audit = QLineEdit(self.exp_audit.body)
        self.ed_audit.setPlaceholderText(
            "* or stage1,stage2,stage3"
        )

        self.exp_audit.body_l.addWidget(
            QLabel("audit_stages:", self.exp_audit.body),
            0,
            0,
        )
        self.exp_audit.body_l.addWidget(self.ed_audit, 0, 1)

        self.binder.bind_line_edit(
            "audit_stages",
            self.ed_audit,
            on="editingFinished",
        )

        lay.addWidget(self.exp_audit, 0)

        # -------------------------
        # Stage-1 reuse policy
        # -------------------------
        self.exp_reuse = _Expander(
            "Stage-1 reuse policy",
            parent=box,
        )

        self.chk_clean = QCheckBox(
            "Clean Stage-1 dir",
            self.exp_reuse.body,
        )
        self.chk_reuse = QCheckBox(
            "Auto reuse if match",
            self.exp_reuse.body,
        )
        self.chk_force = QCheckBox(
            "Force rebuild if mismatch",
            self.exp_reuse.body,
        )

        self.exp_reuse.body_l.addWidget(self.chk_clean, 0, 0)
        self.exp_reuse.body_l.addWidget(self.chk_reuse, 1, 0)
        self.exp_reuse.body_l.addWidget(self.chk_force, 2, 0)

        self.binder.bind_checkbox(
            "clean_stage1_dir",
            self.chk_clean,
        )
        self.binder.bind_checkbox(
            "stage1_auto_reuse_if_match",
            self.chk_reuse,
        )
        self.binder.bind_checkbox(
            "stage1_force_rebuild_if_mismatch",
            self.chk_force,
        )

        lay.addWidget(self.exp_reuse, 0)

        # Hint styling
        box.setStyleSheet(
            "\n".join(
                [
                    "QLabel#auditHint {",
                    "  color: rgba(30,30,30,0.72);",
                    "  font-size: 11px;",
                    "}",
                ]
            )
        )

        return box

    def _path_row(
        self,
        parent: QWidget,
        ed: QLineEdit,
        b1: QToolButton,
        b2: QToolButton,
        b3: QToolButton,
    ) -> QWidget:
        w = QWidget(parent)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(ed, 1)
        lay.addWidget(b1, 0)
        lay.addWidget(b2, 0)
        lay.addWidget(b3, 0)
        return w

    def _mini_btn(
        self,
        parent: QWidget,
        *,
        text: str,
        icon: QStyle.StandardPixmap,
        tip: str,
    ) -> QToolButton:
        btn = QToolButton(parent)
        btn.setObjectName("miniAction")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setText(str(text))
        btn.setToolTip(str(tip))
        btn.setIcon(self.style().standardIcon(icon))
        btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        return btn

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.btn_ds_browse.clicked.connect(self._browse_ds)
        self.btn_ds_open.clicked.connect(self._open_ds)
        self.btn_ds_copy.clicked.connect(self._copy_ds)

        self.btn_res_browse.clicked.connect(self._browse_res)
        self.btn_res_open.clicked.connect(self._open_res)
        self.btn_res_copy.clicked.connect(self._copy_res)

        self.cmb_audit.currentIndexChanged.connect(
            self._on_audit_preset,
        )
        self.btn_audit_reset.clicked.connect(
            self._audit_all,
        )

        for k, cb in self._audit_boxes.items():
            cb.toggled.connect(
                lambda _on, kk=k: self._on_audit_box(kk),
            )

        self.ed_audit.editingFinished.connect(
            self._on_audit_text,
        )

        self.store.config_changed.connect(
            lambda _k: self.refresh(),
        )
        self.store.config_replaced.connect(
            lambda _cfg: self.refresh(),
        )

    # -----------------------------------------------------------------
    # Refresh
    # -----------------------------------------------------------------
    def refresh(self) -> None:
        self._sync_path_hint()
        self._sync_audit_from_store()

        ok = self._paths_ok()
        self.badge(
            "paths",
            text="Ready" if ok else "Missing paths",
            accent="ok" if ok else "warn",
            tip="Dataset and results root status",
        )

    def _paths_ok(self) -> bool:
        cfg = self.store.cfg
        ds = _as_str_path(cfg.dataset_path).strip()
        rr = _as_str_path(cfg.results_root).strip()
        return bool(ds) and bool(rr)

    def _sync_path_hint(self) -> None:
        cfg = self.store.cfg
        ds = _as_str_path(cfg.dataset_path).strip()
        rr = _as_str_path(cfg.results_root).strip()

        msgs = []
        if not ds:
            msgs.append("Dataset not set.")
        if not rr:
            msgs.append("Results root not set.")
        if not msgs:
            msgs.append("Paths look good.")

        self.lbl_paths_hint.setText("  ".join(msgs))

    # -----------------------------------------------------------------
    # Browse/Open/Copy
    # -----------------------------------------------------------------
    def _browse_ds(self) -> None:
        path, _flt = QFileDialog.getOpenFileName(
            self,
            "Select dataset file",
            "",
            "CSV (*.csv);;All files (*.*)",
        )
        if not path:
            return
        self.store.patch({"dataset_path": path})

    def _browse_res(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select results root",
            "",
        )
        if not path:
            return
        self.store.patch({"results_root": path})

    def _open_path(self, p: str) -> None:
        s = (p or "").strip()
        if not s:
            return
        url = QUrl.fromLocalFile(s)
        QDesktopServices.openUrl(url)

    def _open_ds(self) -> None:
        self._open_path(self.ed_dataset.text())

    def _open_res(self) -> None:
        self._open_path(self.ed_results.text())

    def _copy_text(self, text: str) -> None:
        try:
            QApplication.clipboard().setText(str(text))
        except Exception as exc:
            self.store.error_raised.emit(str(exc))

    def _copy_ds(self) -> None:
        self._copy_text(self.ed_dataset.text())

    def _copy_res(self) -> None:
        self._copy_text(self.ed_results.text())

    # -----------------------------------------------------------------
    # Audit logic
    # -----------------------------------------------------------------
    def _audit_all(self) -> None:
        self._set_audit_boxes(set(_AUD_KEYS))
        self._apply_audit_boxes()

    def _on_audit_preset(self) -> None:
        if self._audit_block:
            return

        mode = str(self.cmb_audit.currentData() or "")
        if mode == "off":
            self._set_audit_boxes(set())
        elif mode == "stage1":
            self._set_audit_boxes({"stage1"})
        elif mode == "stage2_3":
            self._set_audit_boxes({"stage2", "stage3"})
        elif mode == "all":
            self._set_audit_boxes(set(_AUD_KEYS))
        elif mode == "custom":
            # keep current boxes
            pass

        if mode != "custom":
            self._apply_audit_boxes()

    def _on_audit_box(self, _key: str) -> None:
        if self._audit_block:
            return
        self._set_preset_from_boxes()
        self._apply_audit_boxes()

    def _on_audit_text(self) -> None:
        if self._audit_block:
            return
        # Binder has already patched store. Sync boxes.
        self._sync_audit_from_store()

    def _sync_audit_from_store(self) -> None:
        cfg = self.store.cfg
        raw = str(getattr(cfg, "audit_stages", "") or "")
        all_on, sel = _parse_audit(raw)

        if all_on:
            sel = set(_AUD_KEYS)

        self._audit_block = True
        try:
            self._set_audit_boxes(sel)
            self._set_preset_from_boxes(raw_text=raw)
            self._sync_audit_hint(raw, sel)
        finally:
            self._audit_block = False

    def _set_audit_boxes(self, sel: Set[str]) -> None:
        for k, cb in self._audit_boxes.items():
            cb.setChecked(bool(k in sel))

    def _apply_audit_boxes(self) -> None:
        sel = {k for k, cb in self._audit_boxes.items()
               if cb.isChecked()}

        all_on = bool(sel == set(_AUD_KEYS))
        raw = _encode_audit(all_on, sel)

        self._audit_block = True
        try:
            self.store.patch({"audit_stages": raw})
            self.ed_audit.setText(raw)
            self._sync_audit_hint(raw, sel)
        finally:
            self._audit_block = False

    def _set_preset_from_boxes(
        self,
        *,
        raw_text: str = "",
    ) -> None:
        sel = {k for k, cb in self._audit_boxes.items()
               if cb.isChecked()}

        if raw_text.strip() == "*":
            preset = "all"
        elif not sel:
            preset = "off"
        elif sel == {"stage1"}:
            preset = "stage1"
        elif sel == {"stage2", "stage3"}:
            preset = "stage2_3"
        elif sel == set(_AUD_KEYS):
            preset = "all"
        else:
            preset = "custom"

        idx = self.cmb_audit.findData(preset)
        if idx >= 0:
            self.cmb_audit.setCurrentIndex(idx)

    def _sync_audit_hint(self, raw: str, sel: Set[str]) -> None:
        if raw.strip() == "*":
            msg = "Auditing: all stages"
        elif not sel:
            msg = "Auditing: off"
        else:
            msg = f"Auditing: {raw}"

        self.lbl_audit_hint.setText(msg)
