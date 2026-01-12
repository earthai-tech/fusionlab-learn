# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

# physics_dialog.py
#
# Modern Physics dialog driven by prior_schema + GeoConfigStore.
# - No hardcoded parameter names
# - Left navigation + searchable settings pages
# - Live binding to store (OK keeps, Cancel reverts)
# - Dict fields (physics_bounds[K_min]) behave like normal fields

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QSizePolicy,
)

from ..config.geoprior_config import GeoPriorConfig
from ..config.store import GeoConfigStore
from ..config.prior_schema import (
    FieldKey,
    FieldSpec,
    PHYSICS_GROUPS,
    PHYSICS_GROUP_TITLES,
    PHYSICS_SCHEMA,
)

from ..styles import (
    FLAB_STYLE_SHEET,
    PALETTE,
    PRIMARY,
    SECONDARY,
)

_DLG_QSS = (
    FLAB_STYLE_SHEET
    + f"""
QDialog {{
  background: {PALETTE.get("light_bg", "#ffffff")};
}}

QFrame#card {{
  background: #ffffff;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
}}

QLabel#title {{
  font-weight: 700;
  color: {PALETTE.get("light_text_title", PRIMARY)};
}}

QLabel#muted {{
  color: {PALETTE.get("light_text_muted", "#5A6B7B")};
}}

QListWidget {{
  background: #ffffff;
  border: 1px solid rgba(0, 0, 0, 0.10);
  border-radius: 12px;
  padding: 6px;
}}

QListWidget::item {{
  padding: 9px 10px;
  border-radius: 10px;
}}

QListWidget::item:hover {{
  background: rgba(0, 0, 0, 0.04);
}}

QListWidget::item:selected {{
  background: {SECONDARY};
  color: #ffffff;
  font-weight: 700;
}}

QListWidget::item:selected:hover {{
  background: {SECONDARY};
}}

QFrame#fieldTile {{
  background: rgba(0, 0, 0, 0.02);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: 10px;
}}

QLabel#fieldLabel {{
  font-weight: 600;
}}

"""
)



class _JsonEditorDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        *,
        title: str,
        initial_obj: Any,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        self._obj = initial_obj

        lay = QVBoxLayout(self)

        info = QLabel(
            "Edit JSON. Leave empty to clear (None)."
        )
        info.setObjectName("muted")
        info.setWordWrap(True)
        lay.addWidget(info)

        self.ed = QPlainTextEdit()
        self.ed.setMinimumHeight(260)

        if initial_obj is None:
            self.ed.setPlainText("")
        else:
            try:
                txt = json.dumps(
                    initial_obj,
                    indent=2,
                    sort_keys=True,
                )
            except Exception:
                txt = str(initial_obj)
            self.ed.setPlainText(txt)

        lay.addWidget(self.ed)

        self.err = QLabel("")
        self.err.setObjectName("muted")
        self.err.setWordWrap(True)
        lay.addWidget(self.err)

        box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        box.accepted.connect(self._on_ok)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

        self.resize(560, 420)

    def value(self) -> Any:
        return self._obj

    def _on_ok(self) -> None:
        raw = self.ed.toPlainText().strip()
        if not raw:
            self._obj = None
            self.accept()
            return

        try:
            self._obj = json.loads(raw)
            self.accept()
        except Exception as exc:
            self.err.setText(
                f"Invalid JSON: {exc}"
            )


class PhysicsConfigDialog(QDialog):
    """
    Modern Physics editor bound to GeoConfigStore + prior_schema.

    Behavior
    --------
    - UI edits write immediately to store
    - OK keeps changes
    - Cancel reverts to the opening snapshot
    """

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        store: GeoConfigStore,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Physics & Units")
        self.setModal(True)
        self.setStyleSheet(_DLG_QSS)

        self._store = store
        self._default_cfg = GeoPriorConfig()
        self._initial = self._capture_initial()

        self._rows: Dict[FieldKey, QWidget] = {}
        self._widgets: Dict[FieldKey, QWidget] = {}
        self._specs: Dict[FieldKey, FieldSpec] = {}

        self._scaling_preview: Optional[
            QPlainTextEdit
        ] = None

        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(8)
        
        header = self._build_header()
        header.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Fixed,
        )
        
        body = self._build_body()
        footer = self._build_footer_widget()
        
        main.addWidget(header)
        main.addWidget(body, 1)
        main.addWidget(footer)
        
        self._connected = False
        self._connect_store_signals()

        self.resize(980, 620)

    def _connect_store_signals(self) -> None:
        if self._connected:
            return

        self._store.config_changed.connect(
            self._on_cfg_changed
        )
        self._store.dirty_changed.connect(
            self._on_dirty_changed
        )
        self._connected = True

        try:
            n = self._store.overrides_count()
        except Exception:
            n = 0
        self._on_dirty_changed(int(n))

    def _disconnect_store_signals(self) -> None:
        if not self._connected:
            return

        try:
            self._store.config_changed.disconnect(
                self._on_cfg_changed
            )
        except TypeError:
            pass

        try:
            self._store.dirty_changed.disconnect(
                self._on_dirty_changed
            )
        except TypeError:
            pass

        self._connected = False

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    def _build_header(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(6)
        lay.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Physics & Units")
        title.setObjectName("title")

        subtitle = QLabel(
            "Edit physics residuals, bounds, units,"
            " Q forcing, and training strategy."
        )
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)

        lay.addWidget(title)
        lay.addWidget(subtitle)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText(
            "Search settings..."
        )
        self.ed_search.textChanged.connect(
            self._apply_filter
        )

        self.cb_advanced = QCheckBox("Show advanced")
        self.cb_advanced.setChecked(False)
        self.cb_advanced.toggled.connect(
            self._apply_filter
        )

        self.lbl_dirty = QLabel("")
        self.lbl_dirty.setObjectName("muted")

        row.addWidget(self.ed_search, 1)
        row.addWidget(self.cb_advanced)
        row.addWidget(self.lbl_dirty)

        lay.addLayout(row)
        return w

    def _build_body(self) -> QWidget:
        split = QSplitter(Qt.Horizontal)

        # Left nav
        self.nav = QListWidget()
        self.nav.setMinimumWidth(260)
        self.nav.setMaximumWidth(340)

        # Right pages
        self.pages = QStackedWidget()

        group_ids = list(PHYSICS_GROUP_TITLES.keys())
        for gid in group_ids:
            title = PHYSICS_GROUP_TITLES.get(gid, gid)
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, gid)
            self.nav.addItem(item)

            page = self._build_group_page(gid)
            self.pages.addWidget(page)

        self.nav.currentRowChanged.connect(
            self.pages.setCurrentIndex
        )
        self.nav.setCurrentRow(0)

        split.addWidget(self.nav)
        split.addWidget(self.pages)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        return split

    def _build_footer_widget(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
    
        self.btn_reset_group = QPushButton("Reset group")
        self.btn_reset_group.setObjectName("reset")
        self.btn_reset_group.clicked.connect(
            self._reset_current_group
        )
    
        self.btn_reset_all = QPushButton("Reset all")
        self.btn_reset_all.setObjectName("reset")
        self.btn_reset_all.clicked.connect(self._reset_all)
    
        row.addWidget(self.btn_reset_group)
        row.addWidget(self.btn_reset_all)
        row.addStretch(1)
    
        box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
    
        ok_btn = box.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setObjectName("runButton")
    
        cancel_btn = box.button(QDialogButtonBox.Cancel)
        if cancel_btn is not None:
            cancel_btn.setObjectName("reset")
    
        row.addWidget(box)
        return w
    
    def _build_group_page(self, group_id: str) -> QWidget:
        outer = QWidget()
        out_lay = QVBoxLayout(outer)
        out_lay.setContentsMargins(0, 0, 0, 0)
    
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
    
        content = QWidget()
        content_lay = QVBoxLayout(content)
        content_lay.setContentsMargins(0, 0, 0, 0)
        content_lay.setSpacing(10)
    
        card = QFrame()
        card.setObjectName("card")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(12, 12, 12, 12)
        card_lay.setSpacing(10)
    
        title = QLabel(
            PHYSICS_GROUP_TITLES.get(group_id, group_id)
        )
        title.setObjectName("title")
        card_lay.addWidget(title)
    
        keys = list(PHYSICS_GROUPS.get(group_id, []))
        cols = self._grid_cols_for_group(keys)
    
        grid_box = QWidget()
        grid = QGridLayout(grid_box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
    
        r = 0
        c = 0
    
        for key in keys:
            spec = PHYSICS_SCHEMA.get(key)
            if spec is None:
                continue
    
            self._specs[key] = spec
    
            tile = self._make_tile(spec)
            self._rows[key] = tile
    
            if self._should_span_full(spec):
                grid.addWidget(tile, r, 0, 1, cols)
                r += 1
                c = 0
                continue
    
            grid.addWidget(tile, r, c, 1, 1)
            c += 1
            if c >= cols:
                r += 1
                c = 0
    
        if group_id == "physics.diagnostics":
            prev = self._make_scaling_preview()
            grid.addWidget(prev, r, 0, 1, cols)
    
        card_lay.addWidget(grid_box)
    
        content_lay.addWidget(card)
        scroll.setWidget(content)
        out_lay.addWidget(scroll)
        return outer


    # ---------------------------------------------------------
    # Row factory
    # ---------------------------------------------------------
    def _make_tile(self, spec: FieldSpec) -> QWidget:
        tile = QFrame()
        tile.setObjectName("fieldTile")
    
        val = self._store.get_value(spec.key)
        kind = self._effective_kind(spec, val)
    
        if kind == "bool":
            lay = QHBoxLayout(tile)
            lay.setContentsMargins(10, 8, 10, 8)
            lay.setSpacing(10)
    
            lbl = QLabel(spec.label)
            lbl.setObjectName("fieldLabel")
            lbl.setToolTip(spec.tooltip or "")
            lbl.setWordWrap(True)
    
            cb = self._make_editor(spec)
            cb.setToolTip(spec.tooltip or "")
    
            lay.addWidget(lbl, 1)
            lay.addWidget(cb, 0, Qt.AlignRight)
    
            self._widgets[spec.key] = cb
            self._refresh_widget(spec.key)
            return tile
    
        lay = QVBoxLayout(tile)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
    
        lbl = QLabel(spec.label)
        lbl.setObjectName("fieldLabel")
        lbl.setToolTip(spec.tooltip or "")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
    
        row = QHBoxLayout()
        row.setSpacing(8)
    
        editor = self._make_editor(spec)
        editor.setToolTip(spec.tooltip or "")
    
        self._tune_editor_width(spec, editor)
    
        row.addWidget(editor, 1)
    
        if spec.kind == "path":
            btns = self._make_path_buttons(spec)
            row.addWidget(btns)
    
        if spec.kind == "json":
            btn = QPushButton("Edit…")
            btn.setObjectName("reset")
            btn.clicked.connect(
                lambda _, k=spec.key: self._edit_json(k)
            )
            row.addWidget(btn)
    
        lay.addLayout(row)
    
        self._widgets[spec.key] = editor
        self._refresh_widget(spec.key)
        return tile

    def _tune_editor_width(
        self,
        spec: FieldSpec,
        editor: QWidget,
    ) -> None:
        val = self._store.get_value(spec.key)
        kind = self._effective_kind(spec, val)
    
        if isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            editor.setMaximumWidth(220)
            return
    
        if isinstance(editor, QComboBox):
            editor.setMaximumWidth(260)
            return
    
        if isinstance(editor, QLineEdit):
            if spec.kind == "path":
                return
    
            label = (spec.label or "").lower()
            if "col" in label or "mode" in label:
                editor.setMaximumWidth(280)
                return


    def _make_editor(self, spec: FieldSpec) -> QWidget:
        key = spec.key
        val = self._store.get_value(key)

        kind = self._effective_kind(spec, val)

        if kind == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(val))
            cb.toggled.connect(
                lambda v, k=key: self._store.set_value_by_key(
                    k,
                    bool(v),
                )
            )
            return cb

        if kind == "choice":
            cbx = QComboBox()
            if spec.choices:
                cbx.addItems(list(spec.choices))
            cbx.setEditable(bool(spec.editable))

            cur = "" if val is None else str(val)
            idx = cbx.findText(cur)
            if idx >= 0:
                cbx.setCurrentIndex(idx)
            elif cur and cbx.isEditable():
                cbx.setEditText(cur)

            cbx.currentTextChanged.connect(
                lambda t, k=key: self._store.set_value_by_key(
                    k,
                    str(t).strip(),
                )
            )
            return cbx

        if kind == "int":
            if val is None:
                ed = QLineEdit()
                ed.setPlaceholderText("auto")
                ed.editingFinished.connect(
                    lambda k=key, w=ed: self._set_opt_int(k, w)
                )
                return ed

            sb = QSpinBox()
            sb.setRange(-1_000_000_000, 1_000_000_000)
            sb.setValue(int(val))
            sb.setAlignment(Qt.AlignRight)
            sb.valueChanged.connect(
                lambda v, k=key: self._store.set_value_by_key(
                    k,
                    int(v),
                )
            )
            return sb

        if kind == "float":
            if val is None:
                ed = QLineEdit()
                ed.setPlaceholderText("auto")
                ed.editingFinished.connect(
                    lambda k=key, w=ed: self._set_opt_float(
                        k, w
                    )
                )
                return ed

            sb = QDoubleSpinBox()
            sb.setRange(-1e15, 1e15)
            sb.setDecimals(self._float_decimals(val))
            sb.setSingleStep(self._float_step(val))
            sb.setValue(float(val))
            sb.setAlignment(Qt.AlignRight)
            sb.valueChanged.connect(
                lambda v, k=key: self._store.set_value_by_key(
                    k,
                    float(v),
                )
            )
            return sb

        if kind == "json":
            # Show a compact read-only hint; edit via button.
            ed = QLineEdit()
            ed.setReadOnly(True)
            if val is None:
                ed.setText("")
            else:
                ed.setText("JSON set")
            return ed

        # text / path
        ed = QLineEdit()
        ed.setText("" if val is None else str(val))
        ed.editingFinished.connect(
            lambda k=key, w=ed: self._store.set_value_by_key(
                k,
                w.text().strip() or None,
            )
        )
        return ed

    # ---------------------------------------------------------
    # Special widgets
    # ---------------------------------------------------------
    def _make_path_buttons(self, spec: FieldSpec) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        btn_browse = QPushButton("Browse…")
        btn_browse.setObjectName("ghost")
        btn_browse.clicked.connect(
            lambda: self._browse_path(spec.key)
        )

        btn_open = QPushButton("Open")
        btn_open.setObjectName("ghost")
        btn_open.clicked.connect(
            lambda: self._open_path(spec.key)
        )

        btn_clear = QPushButton("Clear")
        btn_clear.setObjectName("ghost")
        btn_clear.clicked.connect(
            lambda: self._store.set_value_by_key(
                spec.key, None
            )
        )

        lay.addWidget(btn_browse)
        lay.addWidget(btn_open)
        lay.addWidget(btn_clear)

        return box

    def _make_scaling_preview(self) -> QWidget:
        box = QFrame()
        box.setObjectName("card")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        title = QLabel("Scaling kwargs preview")
        title.setObjectName("title")
        lay.addWidget(title)

        self._scaling_preview = QPlainTextEdit()
        self._scaling_preview.setReadOnly(True)
        self._scaling_preview.setMinimumHeight(180)
        lay.addWidget(self._scaling_preview)

        self._refresh_scaling_preview()
        return box

    # ---------------------------------------------------------
    # Filtering
    # ---------------------------------------------------------
    def _apply_filter(self) -> None:
        q = self.ed_search.text().strip().lower()
        show_adv = bool(self.cb_advanced.isChecked())

        for key, row in self._rows.items():
            spec = self._specs.get(key)
            if spec is None:
                row.setVisible(True)
                continue

            if (not show_adv) and spec.advanced:
                row.setVisible(False)
                continue

            if not q:
                row.setVisible(True)
                continue

            hay = " ".join([
                spec.label.lower(),
                spec.key.name.lower(),
                str(spec.key.subkey or "").lower(),
            ])
            row.setVisible(q in hay)

    # ---------------------------------------------------------
    # Store sync
    # ---------------------------------------------------------
    def _on_cfg_changed(self, keys: object) -> None:
        try:
            changed = set(keys or [])
        except Exception:
            changed = set()

        # Update widgets whose top-level field changed.
        for fkey in self._widgets:
            if fkey.name in changed:
                self._refresh_widget(fkey)

        if "scaling_kwargs_json_path" in changed:
            self._refresh_scaling_preview()

    def _on_dirty_changed(self, n: int) -> None:
        self.lbl_dirty.setText(f"Overrides: {int(n)}")

    def _refresh_widget(self, fkey: FieldKey) -> None:
        w = self._widgets.get(fkey)
        spec = self._specs.get(fkey)
        if w is None or spec is None:
            return

        val = self._store.get_value(fkey)

        kind = self._effective_kind(spec, val)

        # bool
        if isinstance(w, QCheckBox):
            w.blockSignals(True)
            w.setChecked(bool(val))
            w.blockSignals(False)
            return

        # choice
        if isinstance(w, QComboBox):
            w.blockSignals(True)
            cur = "" if val is None else str(val)
            idx = w.findText(cur)
            if idx >= 0:
                w.setCurrentIndex(idx)
            elif w.isEditable():
                w.setEditText(cur)
            w.blockSignals(False)
            return

        # int
        if isinstance(w, QSpinBox):
            w.blockSignals(True)
            w.setValue(int(val or 0))
            w.blockSignals(False)
            return

        # float
        if isinstance(w, QDoubleSpinBox):
            w.blockSignals(True)
            w.setDecimals(self._float_decimals(val))
            w.setSingleStep(self._float_step(val))
            w.setValue(float(val or 0.0))
            w.blockSignals(False)
            return

        # json read-only hint
        if kind == "json" and isinstance(w, QLineEdit):
            w.blockSignals(True)
            w.setText("" if val is None else "JSON set")
            w.blockSignals(False)
            return

        # text/path
        if isinstance(w, QLineEdit):
            w.blockSignals(True)
            w.setText("" if val is None else str(val))
            w.blockSignals(False)
            return

    # ---------------------------------------------------------
    # OK / Cancel semantics
    # ---------------------------------------------------------
    def accept(self) -> None:
        self._disconnect_store_signals()
        super().accept()

    def reject(self) -> None:
        self._restore_initial()
        self._disconnect_store_signals()
        super().reject()


    # ---------------------------------------------------------
    # Reset helpers
    # ---------------------------------------------------------
    def _reset_current_group(self) -> None:
        row = self.nav.currentRow()
        if row < 0:
            return

        item = self.nav.item(row)
        gid = str(item.data(Qt.UserRole))

        keys = PHYSICS_GROUPS.get(gid, [])
        self._reset_keys_to_defaults(keys)

    def _reset_all(self) -> None:
        keys = list(self._widgets.keys())
        self._reset_keys_to_defaults(keys)

    def _reset_keys_to_defaults(
        self,
        keys: Any,
    ) -> None:
        with self._store.batch():
            for key in keys:
                self._set_default_value(key)

    def _set_default_value(self, key: FieldKey) -> None:
        if not key.is_dict_item():
            val = getattr(self._default_cfg, key.name)
            self._store.set_value_by_key(key, val)
            return

        base = getattr(self._default_cfg, key.name)
        if isinstance(base, dict):
            v = base.get(key.subkey)
            self._store.set_value_by_key(key, v)
            return

    # ---------------------------------------------------------
    # Initial snapshot
    # ---------------------------------------------------------
    def _capture_initial(self) -> Dict[FieldKey, Any]:
        snap: Dict[FieldKey, Any] = {}
        for key in PHYSICS_SCHEMA.keys():
            snap[key] = self._store.get_value(key)
        return snap

    def _restore_initial(self) -> None:
        with self._store.batch():
            for key, v in self._initial.items():
                self._store.set_value_by_key(
                    key,
                    v,
                    strict_subkey=False,
                )

    # ---------------------------------------------------------
    # Path actions
    # ---------------------------------------------------------
    def _browse_path(self, key: FieldKey) -> None:
        cur = self._store.get_value(key)
        start = "" if cur is None else str(cur)

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select file",
            start,
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return

        self._store.set_value_by_key(key, path)
        self._refresh_scaling_preview()

    def _open_path(self, key: FieldKey) -> None:
        path = self._store.get_value(key)
        if not path:
            return
        p = str(path)
        if not os.path.exists(p):
            QMessageBox.warning(
                self,
                "File not found",
                "The selected file does not exist.",
            )
            return
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(p)
        )

    # ---------------------------------------------------------
    # JSON editor
    # ---------------------------------------------------------
    def _edit_json(self, key: FieldKey) -> None:
        spec = PHYSICS_SCHEMA.get(key)
        if spec is None:
            return

        cur = self._store.get_value(key)
        dlg = _JsonEditorDialog(
            self,
            title=spec.label,
            initial_obj=cur,
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        self._store.set_value_by_key(
            key,
            dlg.value(),
        )

    # ---------------------------------------------------------
    # Scaling preview
    # ---------------------------------------------------------
    def _refresh_scaling_preview(self) -> None:
        if self._scaling_preview is None:
            return

        key = FieldKey("scaling_kwargs_json_path")
        path = self._store.get_value(key)

        if not path:
            self._scaling_preview.setPlainText("")
            return

        p = str(path)
        if not os.path.exists(p):
            self._scaling_preview.setPlainText(
                "File not found."
            )
            return

        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            txt = json.dumps(
                obj,
                indent=2,
                sort_keys=True,
            )
            self._scaling_preview.setPlainText(txt)
        except Exception as exc:
            self._scaling_preview.setPlainText(
                f"Failed to parse JSON:\n{exc}"
            )

    # ---------------------------------------------------------
    # Parsers & heuristics
    # ---------------------------------------------------------
    def _effective_kind(
        self,
        spec: FieldSpec,
        val: Any,
    ) -> str:
        # Respect schema, but avoid obvious mismatches.
        if spec.kind in ("path", "json", "choice", "bool"):
            return spec.kind

        if isinstance(val, bool):
            return "bool"

        if isinstance(val, int) and not isinstance(val, bool):
            return "int"

        if isinstance(val, float):
            return "float"

        if val is None:
            return spec.kind

        # Some fields are typed as str but look numeric.
        if isinstance(val, str) and spec.kind == "float":
            return "text"

        return spec.kind

    @staticmethod
    def _grid_cols_for_group(keys: list) -> int:
        n = len(keys)
        if n >= 14:
            return 3
        return 2
    
    
    @staticmethod
    def _should_span_full(spec: FieldSpec) -> bool:
        if spec.kind in ("path", "json"):
            return True
    
        label = (spec.label or "").lower()
        if "path" in label or "file" in label:
            return True
    
        return False


    @staticmethod
    def _float_decimals(v: Any) -> int:
        try:
            x = abs(float(v))
        except Exception:
            return 6

        if x == 0.0:
            return 6
        if x < 1e-4:
            return 12
        if x < 1e-2:
            return 10
        return 6

    @staticmethod
    def _float_step(v: Any) -> float:
        try:
            x = abs(float(v))
        except Exception:
            return 0.1

        if x == 0.0:
            return 0.1
        if x < 1e-6:
            return 1e-9
        if x < 1e-3:
            return 1e-6
        if x < 1.0:
            return 1e-3
        return 0.1


    @staticmethod
    def _snapshot_store(
        store: GeoConfigStore,
    ) -> Dict[FieldKey, Any]:
        snap: Dict[FieldKey, Any] = {}
        for key in PHYSICS_SCHEMA.keys():
            snap[key] = store.get_value(key)
        return snap

    @staticmethod
    def _diff_snapshots(
        before: Dict[FieldKey, Any],
        after: Dict[FieldKey, Any],
    ) -> Dict[str, Any]:
        patch: Dict[str, Any] = {}

        for key, new_v in after.items():
            old_v = before.get(key)
            if old_v == new_v:
                continue

            if key.is_dict_item():
                grp = patch.setdefault(key.name, {})
                grp[str(key.subkey)] = new_v
            else:
                patch[key.name] = new_v

        return patch


    def _set_opt_int(self, key: FieldKey, ed: QLineEdit) -> None:
        t = ed.text().strip()
        if not t:
            self._store.set_value_by_key(key, None)
            return
        try:
            self._store.set_value_by_key(key, int(t))
        except Exception:
            self._emit_bad_value("int", ed)

    def _set_opt_float(self, key: FieldKey, ed: QLineEdit) -> None:
        t = ed.text().strip()
        if not t:
            self._store.set_value_by_key(key, None)
            return
        try:
            self._store.set_value_by_key(key, float(t))
        except Exception:
            self._emit_bad_value("float", ed)

    def _emit_bad_value(self, kind: str, ed: QLineEdit) -> None:
        QMessageBox.warning(
            self,
            "Invalid value",
            f"Expected a {kind}.",
        )
        self._refresh_widget(
            self._find_key_for_widget(ed)
        )

    def _find_key_for_widget(self, w: QWidget) -> FieldKey:
        for k, ww in self._widgets.items():
            if ww is w:
                return k
        return FieldKey("city")

    def _compute_patch(self) -> Dict[str, Any]:
        patch: Dict[str, Any] = {}

        for key, old in self._initial.items():
            new = self._store.get_value(key)

            if new == old:
                continue

            if key.is_dict_item():
                grp = patch.get(key.name)
                if grp is None:
                    grp = {}
                    patch[key.name] = grp

                grp[str(key.subkey)] = new
                continue

            patch[key.name] = new

        return patch

    def patch(self) -> Dict[str, Any]:
        return self._compute_patch()

    @classmethod
    def edit(
        cls,
        parent: QWidget,
        *,
        store: GeoConfigStore,
    ) -> Optional[Dict[str, Any]]:
        dlg = cls(parent, store=store)

        if dlg.exec_() != QDialog.Accepted:
            return None

        return dlg.patch()


def edit_physics_cfg(
    parent: QWidget,
    store: GeoConfigStore,
) -> bool:
    patch = PhysicsConfigDialog.edit(
        parent,
        store=store,
    )
    return patch is not None

