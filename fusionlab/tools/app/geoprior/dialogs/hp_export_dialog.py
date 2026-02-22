# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
hp_export_dialog.py

Store-driven dialog for "Export..." preferences.

This dialog stores UI-only export preferences in cfg._meta:
- tuner_export_kind
- tuner_export_format
- tuner_export_path
- tuner_export_pretty
- tuner_export_include_defaults
- tuner_export_include_overrides

Writes are applied only on OK.
Cancel does not touch the store.

Actual exporting is performed elsewhere (button handler),
using these saved preferences.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..config.store import GeoConfigStore


_ALLOWED_KIND = (
    "search_space",
    "config_snapshot",
    "config_overrides",
)
_ALLOWED_FMT = ("json", "yaml")


class ExportDialog(QDialog):
    """
    Edit export preferences (UI-only) via the store.

    Notes
    -----
    - Values are stored under cfg._meta.
    - OK writes to store, Cancel does nothing.
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QDialog] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self.setWindowTitle("Export")

        meta = self._get_meta()

        kind0 = str(meta.get("tuner_export_kind", "search_space") or "")
        fmt0 = str(meta.get("tuner_export_format", "json") or "json")
        path0 = str(meta.get("tuner_export_path", "") or "")

        pretty0 = bool(meta.get("tuner_export_pretty", True))
        inc_def0 = bool(meta.get("tuner_export_include_defaults", False))
        inc_ov0 = bool(meta.get("tuner_export_include_overrides", True))

        if kind0 not in _ALLOWED_KIND:
            kind0 = "search_space"
        if fmt0 not in _ALLOWED_FMT:
            fmt0 = "json"

        self._init = {
            "tuner_export_kind": kind0,
            "tuner_export_format": fmt0,
            "tuner_export_path": path0,
            "tuner_export_pretty": pretty0,
            "tuner_export_include_defaults": inc_def0,
            "tuner_export_include_overrides": inc_ov0,
        }

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        card, grid = self._make_card("Export preferences")

        self.cb_kind = QComboBox()
        self.cb_kind.addItems(list(_ALLOWED_KIND))
        self.cb_kind.setCurrentText(kind0)

        self.cb_fmt = QComboBox()
        self.cb_fmt.addItems(list(_ALLOWED_FMT))
        self.cb_fmt.setCurrentText(fmt0)

        self.le_path = QLineEdit()
        self.le_path.setText(path0)
        self.le_path.setPlaceholderText(
            "Choose output file path...",
        )

        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)

        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        path_row.addWidget(self.le_path, 1)
        path_row.addWidget(btn_browse)

        self.ck_pretty = QCheckBox("Pretty (indent)")
        self.ck_pretty.setChecked(bool(pretty0))

        self.ck_inc_defaults = QCheckBox("Include defaults")
        self.ck_inc_defaults.setChecked(bool(inc_def0))

        self.ck_inc_overrides = QCheckBox("Include overrides")
        self.ck_inc_overrides.setChecked(bool(inc_ov0))

        r = 0
        grid.addWidget(self._lbl("What:"), r, 0)
        grid.addWidget(self.cb_kind, r, 1)
        r += 1

        grid.addWidget(self._lbl("Format:"), r, 0)
        grid.addWidget(self.cb_fmt, r, 1)
        r += 1

        grid.addWidget(self._lbl("Output file:"), r, 0)
        grid.addLayout(path_row, r, 1)
        r += 1

        grid.addWidget(self._lbl("Options:"), r, 0)

        opt_col = QVBoxLayout()
        opt_col.setSpacing(6)
        opt_col.addWidget(self.ck_pretty)
        opt_col.addWidget(self.ck_inc_defaults)
        opt_col.addWidget(self.ck_inc_overrides)

        grid.addLayout(opt_col, r, 1)

        layout.addWidget(card)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        ok_btn = btns.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setDefault(True)

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _make_card(
        self,
        title: str,
    ) -> Tuple[QFrame, QGridLayout]:
        frame = QFrame()
        frame.setObjectName("card")

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        ttl = QLabel(title)
        ttl.setObjectName("cardTitle")
        outer.addWidget(ttl)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        outer.addLayout(grid)

        return frame, grid

    def _lbl(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return lab

    def _browse(self) -> None:
        fmt = str(self.cb_fmt.currentText() or "json").strip().lower()
        ext = "json" if fmt == "json" else "yaml"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to file",
            "",
            f"{ext.upper()} Files (*.{ext});;All Files (*)",
        )
        if path:
            self.le_path.setText(path)

    # -----------------------------------------------------------------
    # Store helpers
    # -----------------------------------------------------------------
    def _get_meta(self) -> Dict[str, Any]:
        try:
            cur = getattr(self._store.cfg, "_meta", None)
        except Exception:
            cur = None
        if not isinstance(cur, dict):
            return {}
        return dict(cur)

    def _patch_meta(self, patch: Dict[str, Any]) -> None:
        if not patch:
            return
        with self._store.batch():
            self._store.merge_dict_field(
                "_meta",
                dict(patch),
                replace=False,
            )

    # -----------------------------------------------------------------
    # OK
    # -----------------------------------------------------------------
    def _on_accept(self) -> None:
        kind = str(self.cb_kind.currentText() or "search_space")
        kind = kind.strip()

        fmt = str(self.cb_fmt.currentText() or "json")
        fmt = fmt.strip().lower()

        path = str(self.le_path.text() or "").strip()

        pretty = bool(self.ck_pretty.isChecked())
        inc_def = bool(self.ck_inc_defaults.isChecked())
        inc_ov = bool(self.ck_inc_overrides.isChecked())

        patch: Dict[str, Any] = {}

        def ch(k: str, v: Any) -> None:
            if self._init.get(k) != v:
                patch[k] = v

        ch("tuner_export_kind", kind)
        ch("tuner_export_format", fmt)
        ch("tuner_export_path", path)
        ch("tuner_export_pretty", pretty)
        ch("tuner_export_include_defaults", inc_def)
        ch("tuner_export_include_overrides", inc_ov)

        if patch:
            self._patch_meta(patch)

        self.accept()

    # -----------------------------------------------------------------
    # Public entry
    # -----------------------------------------------------------------
    @classmethod
    def edit(
        cls,
        *,
        store: GeoConfigStore,
        parent: Optional[QDialog] = None,
    ) -> bool:
        dlg = cls(store=store, parent=parent)
        return dlg.exec_() == QDialog.Accepted

    @staticmethod
    def snapshot(store: GeoConfigStore) -> Dict[str, Any]:
        keys = (
            "tuner_export_kind",
            "tuner_export_format",
            "tuner_export_path",
            "tuner_export_pretty",
            "tuner_export_include_defaults",
            "tuner_export_include_overrides",
        )

        try:
            meta = getattr(store.cfg, "_meta", None)
        except Exception:
            meta = None

        if not isinstance(meta, dict):
            meta = {}

        snap: Dict[str, Any] = {}
        for k in keys:
            snap[k] = meta.get(k)
        return snap
