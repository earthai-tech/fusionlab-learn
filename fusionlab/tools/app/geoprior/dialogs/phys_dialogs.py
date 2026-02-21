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

from PyQt5.QtCore import Qt, QUrl,QEvent, pyqtSignal
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
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QTreeWidgetItem, 
    QTreeWidget, 
    QTextBrowser
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
from ..config.helps import help_text, first_line
from ..ui.delegates.nav_badge_delegate import (
    NavCountBadgeDelegate,
)


class _Badge(QLabel):
    def __init__(self, text: str, *, accent: bool = False) -> None:
        super().__init__(text)
        self.setObjectName("badge")
        self.setProperty("accent", bool(accent))
        self.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )


class _InspectorPanel(QFrame):
    reset_clicked = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("inspector")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        self.lbl_title = QLabel("Inspector")
        self.lbl_title.setObjectName("title")

        self.lbl_key = QLabel("")
        self.lbl_key.setObjectName("muted")
        self.lbl_key.setWordWrap(True)

        # self.lbl_help = QLabel("Select a setting to see details.")
        # self.lbl_help.setWordWrap(True)
        self.ed_help = QTextBrowser()
        self.ed_help.setFrameShape(QFrame.NoFrame)
        self.ed_help.setOpenExternalLinks(False)
        self.ed_help.setMinimumHeight(90)

        self.lbl_state = QLabel("")
        self.lbl_state.setObjectName("muted")
        self.lbl_state.setWordWrap(True)

        self.btn_reset = QPushButton("Reset this setting")
        self.btn_reset.setObjectName("reset")
        # self.btn_reset.setObjectName("reset")
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self._on_reset)

        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_key)
        # lay.addWidget(self.lbl_help, 1)
        lay.addWidget(self.ed_help, 1)
        lay.addWidget(self.lbl_state)
        lay.addWidget(self.btn_reset)

        self._cur_key: Optional[FieldKey] = None

    def set_field(
        self,
        *,
        spec: FieldSpec,
        current: Any,
        default: Any,
        overridden: bool,
        changed: bool,
    ) -> None:
        self._cur_key = spec.key
        self.btn_reset.setEnabled(True)

        key_txt = spec.key.name
        if spec.key.subkey:
            key_txt += f"[{spec.key.subkey}]"

        self.lbl_title.setText(spec.label)
        self.lbl_key.setText(key_txt)

        help_txt = (spec.tooltip or "").strip()
        if not help_txt:
            help_txt = help_text(
                spec.key.name,
                spec.key.subkey,
            )
        if not help_txt:
            help_txt = "No help available for this setting."

        # self.lbl_help.setText(help_txt)
        self.ed_help.setPlainText(help_txt)

        st = [
            f"Current: {current!r}",
            f"Default: {default!r}",
            f"Base override: {'yes' if overridden else 'no'}",
            f"Changed in this dialog: {'yes' if changed else 'no'}",
        ]
        self.lbl_state.setText(" | ".join(st))

    def clear(self) -> None:
        self._cur_key = None
        self.lbl_title.setText("Inspector")
        self.lbl_key.setText("")
        self.ed_help.setPlainText("Select a setting to see details.")
        self.lbl_state.setText("")
        self.btn_reset.setEnabled(False)
    

    def _on_reset(self) -> None:
        if self._cur_key is None:
            return
        self.reset_clicked.emit(self._cur_key)


class _ChangesDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        *,
        items: list,  # List[Tuple[FieldSpec, Any, Any]]
        on_revert: Any,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Changes")
        self.setModal(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        title = QLabel("Review changes")
        title.setObjectName("title")
        lay.addWidget(title)

        info = QLabel("Revert individual settings or close to keep.")
        info.setObjectName("muted")
        info.setWordWrap(True)
        lay.addWidget(info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)

        if not items:
            empty = QLabel("No changes.")
            empty.setObjectName("muted")
            bl.addWidget(empty)
        else:
            for spec, old, new in items:
                row = QFrame()
                row.setObjectName("card")
                rl = QHBoxLayout(row)
                rl.setContentsMargins(10, 8, 10, 8)
                rl.setSpacing(10)

                left = QVBoxLayout()
                lbl = QLabel(spec.label)
                lbl.setObjectName("fieldLabel")

                hint = QLabel(f"{old!r} -> {new!r}")
                hint.setObjectName("muted")
                hint.setWordWrap(True)

                left.addWidget(lbl)
                left.addWidget(hint)

                btn = QPushButton("Revert")
                btn.setObjectName("reset")
                btn.clicked.connect(
                    lambda _, k=spec.key: on_revert(k)
                )

                rl.addLayout(left, 1)
                rl.addWidget(btn, 0, Qt.AlignRight)
                bl.addWidget(row)

        bl.addStretch(1)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)

        box = QDialogButtonBox(QDialogButtonBox.Close)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

        self.resize(720, 520)


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
    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        store: GeoConfigStore,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Physics & Units")
        self.setModal(True)
        self.setObjectName("physicsDialog")

        self._store = store
        self._default_cfg = GeoPriorConfig()
        self._initial = self._capture_initial()

        self._rows: Dict[FieldKey, QWidget] = {}
        self._widgets: Dict[FieldKey, QWidget] = {}
        self._specs: Dict[FieldKey, FieldSpec] = {}
        self._row_badges: Dict[FieldKey, Dict[str, QLabel]] = {}

        self._scaling_preview: Optional[QPlainTextEdit] = None
        self._nav_item_by_gid: Dict[str, QTreeWidgetItem] = {}
        self._gid_base_title: Dict[str, str] = {}

        self._current_key: Optional[FieldKey] = None

        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(8)

        header = self._build_header()
        header.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        body = self._build_body()
        footer = self._build_footer_widget()

        main.addWidget(header)
        main.addWidget(body, 1)
        main.addWidget(footer)

        self._connected = False
        self._connect_store_signals()

        self.resize(800, 500) #1180, 700

    # ---------------------------
    # Header
    # ---------------------------
    def _build_header(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Physics & Units")
        title.setObjectName("title")

        subtitle = QLabel(
            "Configure physics residuals, bounds, units, "
            "Q forcing, and training strategy."
        )
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)

        lay.addWidget(title)
        lay.addWidget(subtitle)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText(
            "Search settings (label, key, subkey)..."
        )
        self.ed_search.textChanged.connect(self._apply_filter)

        self.cb_advanced = QCheckBox("Show advanced")
        self.cb_advanced.setChecked(False)
        self.cb_advanced.toggled.connect(self._apply_filter)

        self.btn_checks = QPushButton("Run checks")
        # self.btn_checks.setObjectName("ghost")
        self.btn_checks.clicked.connect(self._run_sanity_checks)

        self.btn_changes = QPushButton("Changes")
        # self.btn_changes.setObjectName("ghost")
        self.btn_changes.clicked.connect(self._open_changes)

        self.lbl_dirty = QLabel("")
        self.lbl_dirty.setObjectName("muted")

        row.addWidget(self.ed_search, 1)
        row.addWidget(self.cb_advanced)
        row.addWidget(self.btn_checks)
        row.addWidget(self.btn_changes)
        row.addWidget(self.lbl_dirty)

        lay.addLayout(row)
        return w

    # ---------------------------
    # Body (3-panel split)
    # ---------------------------
    def _build_body(self) -> QWidget:
        outer = QSplitter(Qt.Horizontal)
    
        # Build pages first so _on_nav_changed has _page_gid ready.
        self.pages = QStackedWidget()
        self._page_gid = []
    
        self._add_page(
            "physics.overview",
            self._build_overview(),
        )
    
        for gid in list(PHYSICS_GROUP_TITLES.keys()):
            self._add_page(gid, self._build_group_page(gid))
    
        # Now build nav (safe: pages + _page_gid exist).
        self.nav = self._build_nav_tree()
        self.nav.setMinimumWidth(260)
        self.nav.setMaximumWidth(360)
    
        self.inspector = _InspectorPanel()
        self.inspector.setMinimumWidth(320)
        self.inspector.reset_clicked.connect(
            self._reset_single_key
        )
    
        mid = QSplitter(Qt.Horizontal)
        mid.addWidget(self.pages)
        mid.addWidget(self.inspector)
        mid.setStretchFactor(0, 1)
        mid.setStretchFactor(1, 0)
    
        # outer.addWidget(self.nav)
        nav_wrap = QFrame()
        nav_wrap.setMinimumWidth(260)
        nav_wrap.setMaximumWidth(340)
        
        nav_wrap.setObjectName("physicsNavWrap")
        vl = QVBoxLayout(nav_wrap)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(6)
        
        lbl = QLabel("Sections")
        lbl.setObjectName("physicsNavTitle")
        
        vl.addWidget(lbl)
        vl.addWidget(self.nav, 1)
        
        outer.addWidget(nav_wrap)
        
        outer.addWidget(mid)
        outer.setStretchFactor(0, 0)
        outer.setStretchFactor(1, 1)
    
        return outer

    def _build_nav_tree(self) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setObjectName("physicsNav")
        tree.setItemDelegate(
            NavCountBadgeDelegate(tree, primary="#2E3191")
        )
        tree.setHeaderHidden(True)
        tree.setMouseTracking(True)
        
        root_over = QTreeWidgetItem(["Quick setup"])
        root_over.setData(0, Qt.UserRole, "physics.overview")
        tree.addTopLevelItem(root_over)
        self._register_nav_item("physics.overview", root_over)
    
        root_phys = QTreeWidgetItem(["Physics"])
        root_phys.setFlags(
            root_phys.flags() & ~Qt.ItemIsSelectable
        )
        tree.addTopLevelItem(root_phys)
    
        root_diag = QTreeWidgetItem(["Diagnostics & safeguards"])
        root_diag.setData(0, Qt.UserRole, "physics.diagnostics")
        tree.addTopLevelItem(root_diag)
        self._register_nav_item("physics.diagnostics", root_diag)
    
        for gid in list(PHYSICS_GROUP_TITLES.keys()):
            if gid == "physics.diagnostics":
                continue
            title = PHYSICS_GROUP_TITLES.get(gid, gid)
            item = QTreeWidgetItem([title])
            item.setData(0, Qt.UserRole, gid)
            root_phys.addChild(item)
            self._register_nav_item(gid, item)
    
        root_phys.setExpanded(True)
    
        # set default selection BEFORE connecting (avoid early emit)
        tree.setCurrentItem(root_over)
        tree.currentItemChanged.connect(self._on_nav_changed)
    
        tree.setMouseTracking(True)
        tree.viewport().setMouseTracking(True)
        tree.setIndentation(14)
        tree.setUniformRowHeights(True)
    
        return tree

    def _register_nav_item(self, gid: str, item: QTreeWidgetItem) -> None:
        self._nav_item_by_gid[gid] = item
        self._gid_base_title[gid] = item.text(0)

    def _add_page(self, gid: str, w: QWidget) -> None:
        self.pages.addWidget(w)
        self._page_gid.append(gid)
        
    def _on_nav_changed(
        self,
        cur: Optional[QTreeWidgetItem],
        prev: Optional[QTreeWidgetItem],
    ) -> None:
        if cur is None:
            return
    
        gid = cur.data(0, Qt.UserRole)
        if gid not in self._page_gid:
            return
    
        self.pages.setCurrentIndex(self._page_gid.index(gid))
    
        self._current_key = None
        insp = getattr(self, "inspector", None)
        if insp is not None:
            insp.clear()

    # ---------------------------
    # Pages
    # ---------------------------
    def _build_overview(self) -> QWidget:
        outer = QWidget()
        lay = QVBoxLayout(outer)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        card = QFrame()
        card.setObjectName("card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.setSpacing(8)

        t = QLabel("Quick setup")
        t.setObjectName("title")

        hint = QLabel(
            "Start here: pick strategy, units, and safeguards. "
            "Use 'Run checks' before training."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)

        cl.addWidget(t)
        cl.addWidget(hint)

        keys = [
            FieldKey("pde_mode"),
            FieldKey("training_strategy"),
            FieldKey("eval_json_units_mode"),
            FieldKey("eval_json_units_scope"),
            FieldKey("physics_baseline_mode"),
            FieldKey("scaling_error_policy"),
            FieldKey("scaling_kwargs_json_path"),
            FieldKey("track_aux_metrics"),
            FieldKey("debug_physics_grads"),
        ]

        for k in keys:
            spec = PHYSICS_SCHEMA.get(k)
            if spec is None:
                continue
            self._specs[k] = spec
            cl.addWidget(self._make_setting_row(spec))

        lay.addWidget(card)
        lay.addStretch(1)
        return outer

    def _build_group_page(self, group_id: str) -> QWidget:
        outer = QWidget()
        out_lay = QVBoxLayout(outer)
        out_lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(10)

        card = QFrame()
        card.setObjectName("card")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(12, 12, 12, 12)
        card_lay.setSpacing(10)

        title = QLabel(PHYSICS_GROUP_TITLES.get(group_id, group_id))
        title.setObjectName("title")
        card_lay.addWidget(title)

        for key in list(PHYSICS_GROUPS.get(group_id, [])):
            spec = PHYSICS_SCHEMA.get(key)
            if spec is None:
                continue
            self._specs[key] = spec
            card_lay.addWidget(self._make_setting_row(spec))

        if group_id == "physics.diagnostics":
            card_lay.addWidget(self._make_scaling_preview())

        card_lay.addStretch(1)
        cl.addWidget(card)

        scroll.setWidget(content)
        out_lay.addWidget(scroll)
        return outer

    # ---------------------------
    # Row widget
    # ---------------------------
    def _make_setting_row(self, spec: FieldSpec) -> QWidget:
        row = QFrame()
        row.setObjectName("settingRow")
        row.setProperty("sel", False)

        hl = QHBoxLayout(row)
        hl.setContentsMargins(12, 10, 12, 10)
        hl.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(4)

        lbl = QLabel(spec.label)
        lbl.setObjectName("fieldLabel")
        lbl.setWordWrap(True)
        
        raw = (spec.tooltip or "").strip()
        if not raw:
            raw = help_text(
                spec.key.name,
                spec.key.subkey,
            )
        
        hint_txt = first_line(raw) if raw else " "
        hint = QLabel(hint_txt)
        hint.setProperty("role", "hint")
        hint.setObjectName("hint")
        hint.setWordWrap(True)

        left.addWidget(lbl)
        left.addWidget(hint)

        editor = self._make_editor(spec)
        self._tune_editor_width(spec, editor)

        b_adv = _Badge("Advanced")
        b_adv.setVisible(bool(spec.advanced))

        b_chg = _Badge("Changed", accent=True)
        b_chg.setVisible(False)

        b_ovr = _Badge("Override")
        b_ovr.setVisible(False)

        badges = QHBoxLayout()
        badges.setSpacing(6)
        badges.addWidget(b_ovr)
        badges.addWidget(b_chg)
        badges.addWidget(b_adv)

        btn_reset = QPushButton("↺")
        btn_reset.setToolTip("Reset to default")
        btn_reset.setObjectName("ghost")
        btn_reset.setMaximumWidth(34)
        btn_reset.clicked.connect(
            lambda _, k=spec.key: self._reset_single_key(k)
        )

        right = QHBoxLayout()
        right.setSpacing(8)
        right.addWidget(editor, 0)
        if spec.kind == "path":
            right.addWidget(self._make_path_buttons(spec), 0)
        
        if spec.kind == "json":
            btn = QPushButton("Edit…")
            btn.setObjectName("ghost")
            btn.clicked.connect(
                lambda _, k=spec.key: self._edit_json(k)
            )
            right.addWidget(btn, 0)


        right.addLayout(badges, 0)
        right.addWidget(btn_reset, 0)

        hl.addLayout(left, 1)
        hl.addLayout(right, 0)

        self._rows[spec.key] = row
        self._widgets[spec.key] = editor
        self._row_badges[spec.key] = {
            "changed": b_chg,
            "override": b_ovr,
            "advanced": b_adv,
        }

        self._refresh_widget(spec.key)
        self._update_row_badges(spec.key)

        for w in (row, lbl, hint, editor):
            w.installEventFilter(self)
        
        # for combo boxes, focus may go to internal widgets
        if isinstance(editor, QComboBox):
            le = editor.lineEdit()
            if le is not None:
                le.installEventFilter(self)
        
        for child in editor.findChildren(QWidget):
            child.installEventFilter(self)

        return row
    
    def _default_value(self, key: FieldKey) -> Any:
        if not key.is_dict_item():
            return getattr(self._default_cfg, key.name)
    
        base = getattr(self._default_cfg, key.name, None)
        if isinstance(base, dict):
            return base.get(key.subkey)
        return None
    
    
    def _nat_key_for_field(self, field_name: str) -> str:
        # Most fields match by uppercasing.
        # Keep a tiny exceptions map when needed.
        overrides = {
            "bounds_mode": "PHYSICS_BOUNDS_MODE",
        }
        return overrides.get(field_name, field_name.upper())
    
    
    def _base_value(self, key: FieldKey) -> Any:
        cfg = self._store.cfg
        base = getattr(cfg, "_base_cfg", None) or {}
    
        nat_key = self._nat_key_for_field(key.name)
        if nat_key not in base:
            return self._default_value(key)
    
        v = base.get(nat_key)
    
        if key.is_dict_item() and isinstance(v, dict):
            return v.get(key.subkey)
    
        return v
    
    
    def _is_overridden(self, key: FieldKey) -> bool:
        cur = self._store.get_value(key)
        base = self._base_value(key)
        return cur != base
    
    
    def _update_row_badges(self, key: FieldKey) -> None:
        badges = self._row_badges.get(key)
        if not badges:
            return
    
        cur = self._store.get_value(key)
        old = self._initial.get(key)
        changed = (cur != old)
    
        overridden = self._is_overridden(key)
    
        b_chg = badges.get("changed")
        if b_chg is not None:
            b_chg.setVisible(bool(changed))
    
        b_ovr = badges.get("override")
        if b_ovr is not None:
            b_ovr.setVisible(bool(overridden))
    
    
    def _reset_single_key(self, key: FieldKey) -> None:
        with self._store.batch():
            self._set_default_value(key)
    
        self._update_row_badges(key)
    
        if self._current_key == key:
            self._select_key(key)

    def eventFilter(self, obj: Any, ev: Any) -> bool:
        t = ev.type()
        if t in (QEvent.FocusIn, QEvent.MouseButtonPress):
            key = self._key_for_object(obj)
            if key is not None:
                self._select_key(key)
        return super().eventFilter(obj, ev)


    def _select_key(self, key: FieldKey) -> None:
        self._current_key = key

        for k, r in self._rows.items():
            r.setProperty("sel", k == key)
            r.style().unpolish(r)
            r.style().polish(r)

        spec = self._specs.get(key)
        if spec is None:
            self.inspector.clear()
            return

        cur = self._store.get_value(key)
        dft = self._default_value(key)
        overridden = self._is_overridden(key)
        changed = (cur != self._initial.get(key))

        self.inspector.set_field(
            spec=spec,
            current=cur,
            default=dft,
            overridden=overridden,
            changed=changed,
        )

    # ---------------------------
    # Filter + nav counts
    # ---------------------------
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

            hay = " ".join(
                [
                    spec.label.lower(),
                    spec.key.name.lower(),
                    str(spec.key.subkey or "").lower(),
                ]
            )
            row.setVisible(q in hay)

        self._update_nav_counts()

    def _update_nav_counts(self) -> None:
        show_adv = bool(self.cb_advanced.isChecked())
        q = self.ed_search.text().strip().lower()

        def ok(k: FieldKey) -> bool:
            spec = self._specs.get(k)
            if spec is None:
                return True
            if (not show_adv) and spec.advanced:
                return False
            if not q:
                return True
            hay = " ".join(
                [
                    spec.label.lower(),
                    spec.key.name.lower(),
                    str(spec.key.subkey or "").lower(),
                ]
            )
            return q in hay

        it = self._nav_item_by_gid.get("physics.overview")
        if it is not None:
            it.setText(0, self._gid_base_title["physics.overview"])

        for gid, keys in PHYSICS_GROUPS.items():
            item = self._nav_item_by_gid.get(gid)
            if item is None:
                continue
            n = sum(1 for k in keys if ok(k))
            base = self._gid_base_title.get(gid, item.text(0))
            item.setText(0, f"{base}  ({n})" if q else base)

    # ---------------------------
    # Changes + checks
    # ---------------------------
    def _open_changes(self) -> None:
        items = []
        for key, old in self._initial.items():
            new = self._store.get_value(key)
            if new == old:
                continue
            spec = self._specs.get(key)
            if spec is None:
                continue
            items.append((spec, old, new))

        def revert(k: FieldKey) -> None:
            self._store.set_value_by_key(
                k,
                self._initial.get(k),
                strict_subkey=False,
            )

        _ChangesDialog(self, items=items, on_revert=revert).exec_()

    def _run_sanity_checks(self) -> None:
        issues = []

        p = self._store.get_value(FieldKey("scaling_kwargs_json_path"))
        if p:
            pp = str(p)
            if not os.path.exists(pp):
                issues.append("Scaling file: not found.")
            else:
                try:
                    with open(pp, "r", encoding="utf-8") as f:
                        json.load(f)
                except Exception as exc:
                    issues.append(f"Scaling file: invalid JSON ({exc}).")

        for prefix in ("K", "Ss", "tau", "H"):
            kmin = FieldKey("physics_bounds", f"{prefix}_min")
            kmax = FieldKey("physics_bounds", f"{prefix}_max")
            vmin = self._store.get_value(kmin)
            vmax = self._store.get_value(kmax)
            try:
                if vmin is not None and vmax is not None:
                    if float(vmin) > float(vmax):
                        issues.append(
                            f"Bounds: {prefix}_min > {prefix}_max."
                        )
            except Exception:
                issues.append(f"Bounds: invalid {prefix} values.")

        if not issues:
            QMessageBox.information(self, "Checks", "No issues detected.")
        else:
            QMessageBox.warning(
                self,
                "Checks",
                "\n".join(f"- {x}" for x in issues),
            )

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
    
        btn_cancel = box.button(QDialogButtonBox.Cancel)
        if btn_cancel is not None:
            pass
        
        self._btn_ok = box.button(QDialogButtonBox.Ok)
        if self._btn_ok is not None:
            self._btn_ok.setObjectName("runButton")
            self._btn_ok.setEnabled(False)
            self._btn_ok.setDefault(True)
        
            self._btn_ok.style().unpolish(self._btn_ok)
            self._btn_ok.style().polish(self._btn_ok)
            self._btn_ok.update()

       
        row.addWidget(box)
        return w

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
        ed.setCursorPosition(0)
        ed.deselect()
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
    # Store sync
    # ---------------------------------------------------------
    def _on_cfg_changed(self, keys: object) -> None:
        try:
            changed = set(keys or [])
        except Exception:
            changed = set()
    
        for fkey in self._widgets:
            if fkey.name in changed:
                self._refresh_widget(fkey)
                self._update_row_badges(fkey)
    
        if "scaling_kwargs_json_path" in changed:
            self._refresh_scaling_preview()
        
        self._btn_ok.setEnabled(bool(self._compute_patch()))


    def _on_dirty_changed(self, n: int) -> None:
        self.lbl_dirty.setText(f"Overrides: {int(n)}")
        self._btn_ok.setEnabled(bool(self._compute_patch()))

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
            w.setCursorPosition(0)
            w.deselect()
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
        it = self.nav.currentItem()
        if it is None:
            return
    
        gid = it.data(0, Qt.UserRole)
        gid = str(gid) if gid else ""
        if not gid:
            return
    
        if gid == "physics.overview":
            keys = [
                FieldKey("pde_mode"),
                FieldKey("training_strategy"),
                FieldKey("eval_json_units_mode"),
                FieldKey("eval_json_units_scope"),
                FieldKey("physics_baseline_mode"),
                FieldKey("scaling_error_policy"),
                FieldKey("scaling_kwargs_json_path"),
                FieldKey("track_aux_metrics"),
                FieldKey("debug_physics_grads"),
            ]
            self._reset_keys_to_defaults(keys)
            return
    
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
    
    def _key_for_object(self, obj: Any) -> Optional[FieldKey]:
        if isinstance(obj, QWidget):
            for k, row in self._rows.items():
                try:
                    if obj is row or row.isAncestorOf(obj):
                        return k
                except Exception:
                    continue
    
        for k, w in self._widgets.items():
            if obj is w:
                return k
            if isinstance(obj, QWidget) and isinstance(w, QWidget):
                try:
                    if w.isAncestorOf(obj):
                        return k
                except Exception:
                    pass
    
        return None

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

