# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.feature_registry

Feature registry (modern UX, typo-safe).

Key idea:
- Users select only from dataset columns (no free typing).
- Store remains the single source of truth.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QStyle,
)

from ....dialogs.feature_dialog import FeatureConfigDialog
from ....config.store import GeoConfigStore
from ..bindings import Binder
from .base import CardBase

# ---------------------------------------------------------------------
# Picker dialog (search + multi-select)
# ---------------------------------------------------------------------
class _ColumnPickDialog(QDialog):
    def __init__(
        self,
        columns: Sequence[str],
        *,
        title: str,
        preselect: Optional[Sequence[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(str(title))
        self.resize(520, 420)

        self._cols = [str(c) for c in (columns or [])]

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search columns…")
        self.search.setClearButtonEnabled(True)

        self.list = QListWidget(self)
        self.list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.list.setAlternatingRowColors(True)

        for c in self._cols:
            it = QListWidgetItem(c, self.list)
            it.setData(Qt.UserRole, c)
            
        # Preselect current values (for "Pick / Replace" UX)
        sel = set(str(x) for x in (preselect or []))
        if sel:
            for i in range(self.list.count()):
                it = self.list.item(i)
                if it and it.text() in sel:
                    it.setSelected(True)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel,
            parent=self,
        )

        root.addWidget(self.search, 0)
        root.addWidget(self.list, 1)
        root.addWidget(btns, 0)

        self.search.textChanged.connect(self._apply)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        self._apply("")

    def selected_columns(self) -> List[str]:
        out: List[str] = []
        for it in self.list.selectedItems():
            v = it.data(Qt.UserRole)
            if v is None:
                continue
            out.append(str(v))
        return out

    def _apply(self, text: str) -> None:
        q = (text or "").strip().lower()
        for i in range(self.list.count()):
            it = self.list.item(i)
            name = (it.text() or "").lower()
            it.setHidden(bool(q) and q not in name)


# ---------------------------------------------------------------------
# Column multi-select editor (safe registry)
# ---------------------------------------------------------------------
class _ColumnMulti(QWidget):
    changed = pyqtSignal()

    def __init__(
        self,
        title: str,
        *,
        parent: QWidget,
        allow_reorder: bool = True,
    ) -> None:
        super().__init__(parent)

        self._available: List[str] = []
        self._title = str(title)

        self._updating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        head = QWidget(self)
        h = QHBoxLayout(head)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        self.lbl = QLabel(self._title, head)
        self.lbl.setObjectName("featSectionTitle")
        
        self.badge = QLabel("0", head)
        self.badge.setObjectName("featCount")
        
        self.miss_chip = QLabel("", head)
        self.miss_chip.setObjectName("featMissingChip")
        self.miss_chip.setVisible(False)

        
        def _mini_btn(
            *,
            text: str = "",
            icon: Optional[QStyle.StandardPixmap] = None,
            tip: str = "",
        ) -> QToolButton:
            b = QToolButton(head)
            b.setCursor(Qt.PointingHandCursor)
            b.setAutoRaise(True)
            b.setObjectName("miniAction")  # <- uses your existing style
            b.setFixedHeight(24)
        
            if icon is not None:
                b.setIcon(self.style().standardIcon(icon))
                b.setIconSize(QSize(14, 14))
                b.setToolButtonStyle(Qt.ToolButtonIconOnly)
                b.setFixedWidth(30)
            else:
                b.setText(text)
                b.setToolButtonStyle(Qt.ToolButtonTextOnly)
                b.setFixedWidth(30)
        
            if tip:
                b.setToolTip(tip)
            return b
        
        # + / − as “icon-like” glyphs, trash icon for clear
        self.btn_add = _mini_btn(text="+", tip="Add columns…")
        self.btn_rm = _mini_btn(text="-", tip="Remove selected")
        self.btn_clear = _mini_btn(
            icon=QStyle.SP_TrashIcon,
            tip="Clear all",
        )
        
        self.btn_pick = _mini_btn(
            icon=QStyle.SP_FileDialogDetailedView,
            tip="Pick columns… (replace selection)",
        )
        self.btn_paste = _mini_btn(
            icon=QStyle.SP_DialogOpenButton,
            tip="Paste from clipboard (add valid columns only)",
        )

        # Layout: title left, actions grouped right (no big empty gap)
        h.addWidget(self.lbl, 0)
        h.addStretch(1)
        h.addWidget(self.miss_chip, 0)
        h.addWidget(self.badge, 0)
        h.addWidget(self.btn_pick, 0)
        h.addWidget(self.btn_paste, 0)
        h.addWidget(self.btn_add, 0)
        h.addWidget(self.btn_rm, 0)
        h.addWidget(self.btn_clear, 0)


        self.list = QListWidget(self)
        self.list.setObjectName("featList")
        self.list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.list.setAlternatingRowColors(True)

        if allow_reorder:
            self.list.setDragEnabled(True)
            self.list.setAcceptDrops(True)
            self.list.setDropIndicatorShown(True)
            self.list.setDragDropMode(
                QAbstractItemView.InternalMove
            )

        root.addWidget(head, 0)
        root.addWidget(self.list, 1)

        self.btn_add.clicked.connect(self._on_add)
        self.btn_rm.clicked.connect(self._on_remove)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_pick.clicked.connect(self._on_pick_replace)
        self.btn_paste.clicked.connect(self._on_paste_add)


        m = self.list.model()
        if m is not None:
            m.rowsMoved.connect(lambda *_: self._emit())
            m.rowsInserted.connect(lambda *_: self._emit())
            m.rowsRemoved.connect(lambda *_: self._emit())
            
        self.list.itemSelectionChanged.connect(self._sync_actions)
        self._sync_actions()

    def set_available(self, cols: Sequence[str]) -> None:
        self._available = [str(c) for c in (cols or [])]

    def set_values(self, cols: Sequence[str]) -> None:
        cur = [str(c) for c in (cols or [])]
    
        self._updating = True
        try:
            self.list.clear()
            for c in cur:
                it = QListWidgetItem(c, self.list)
                it.setData(Qt.UserRole, c)
        finally:
            self._updating = False
    
        self._refresh_badge()
        self._sync_actions()
        self._refresh_missing_chip()
    
    
    def _emit(self) -> None:
        if self._updating:
            return
        self._refresh_badge()
        self._sync_actions()
        self._refresh_missing_chip()
        self.changed.emit()


    def values(self) -> List[str]:
        out: List[str] = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            v = it.data(Qt.UserRole)
            if v is None:
                continue
            out.append(str(v))
        return out

    def missing(self) -> List[str]:
        have = set(self._available)
        return [c for c in self.values() if c not in have]
    
    def _refresh_missing_chip(self) -> None:
        # Only meaningful when dataset columns exist
        if not self._available:
            self.miss_chip.setVisible(False)
            return
    
        miss = self.missing()
        if not miss:
            self.miss_chip.setVisible(False)
            return
    
        self.miss_chip.setText(f"!{len(miss)}")
        self.miss_chip.setToolTip("Missing in dataset:\n" + "\n".join(miss))
        self.miss_chip.setVisible(True)
    
    
    def _on_pick_replace(self) -> None:
        """Open picker and replace current selection (still typo-safe)."""
        if not self._available:
            return
    
        dlg = _ColumnPickDialog(
            self._available,
            title=f"Select {self._title}",
            preselect=self.values(),
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
    
        picked = dlg.selected_columns()
        if picked is None:
            return
    
        # Replace selection
        self.set_values(picked)
        self._emit()
    
    
    def _on_paste_add(self) -> None:
        """
        Paste from clipboard and add valid columns only.
        Accepts newline/comma/semicolon/space separated lists.
        """
        if not self._available:
            return
    
        raw = (QApplication.clipboard().text() or "").strip()
        if not raw:
            return
    
        tokens = [t for t in re.split(r"[\s,;]+", raw) if t.strip()]
        if not tokens:
            return
    
        # Case-insensitive mapping to canonical column names
        lut = {c.lower(): c for c in self._available}
        have = set(self.values())
    
        add: List[str] = []
        ignored: List[str] = []
    
        for t in tokens:
            key = t.strip()
            if not key:
                continue
            canon = lut.get(key.lower())
            if canon is None:
                ignored.append(key)
                continue
            if canon in have or canon in add:
                continue
            add.append(canon)
    
        if not add and ignored:
            self.btn_paste.setToolTip(
                "Paste from clipboard (add valid columns only)\n"
                "Nothing added. Unknown columns:\n" + "\n".join(ignored[:12])
                + ("" if len(ignored) <= 12 else "\n…")
            )
            return
    
        for c in add:
            it = QListWidgetItem(str(c), self.list)
            it.setData(Qt.UserRole, str(c))
    
        if ignored:
            self.btn_paste.setToolTip(
                "Paste from clipboard (add valid columns only)\n"
                f"Added: {len(add)} • Ignored unknown: {len(ignored)}"
            )
        else:
            self.btn_paste.setToolTip(
                "Paste from clipboard (add valid columns only)\n"
                f"Added: {len(add)}"
            )
    
        self._emit()

    # ----------------------------
    # Actions
    # ----------------------------
    def _on_add(self) -> None:
        have = set(self.values())
        pool = [c for c in self._available if c not in have]

        dlg = _ColumnPickDialog(
            pool,
            title=f"Add to {self._title}",
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        add = dlg.selected_columns()
        if not add:
            return

        for c in add:
            it = QListWidgetItem(str(c), self.list)
            it.setData(Qt.UserRole, str(c))

        self._refresh_badge()
        self._emit()

    def _on_remove(self) -> None:
        for it in list(self.list.selectedItems()):
            row = self.list.row(it)
            self.list.takeItem(row)
        self._refresh_badge()
        self._emit()

    def _on_clear(self) -> None:
        self.list.clear()
        self._refresh_badge()
        self._emit()

    def _refresh_badge(self) -> None:
        self.badge.setText(str(self.list.count()))

    def _sync_actions(self) -> None:
        has_cols = bool(self._available)
        has_any = self.list.count() > 0
        has_sel = bool(self.list.selectedItems())
    
        self.btn_add.setEnabled(has_cols)
        self.btn_rm.setEnabled(has_sel)
        self.btn_clear.setEnabled(has_any)

# ---------------------------------------------------------------------
# List-of-lists editor (optional numeric registry)
# ---------------------------------------------------------------------
class _GroupsEditor(QWidget):
    changed = pyqtSignal()

    def __init__(self, *, parent: QWidget) -> None:
        super().__init__(parent)

        self._available: List[str] = []
        self._groups: List[List[str]] = []

        root = QGridLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setHorizontalSpacing(10)
        root.setVerticalSpacing(6)

        self.grp_list = QListWidget(self)
        self.grp_list.setObjectName("featList")
        self.grp_list.setSelectionMode(
            QAbstractItemView.SingleSelection
        )

        btns = QWidget(self)
        b = QHBoxLayout(btns)
        b.setContentsMargins(0, 0, 0, 0)
        b.setSpacing(6)

        self.btn_add = QToolButton(btns)
        self.btn_add.setObjectName("miniAction")
        self.btn_add.setText("Add group")
        self.btn_add.setCursor(Qt.PointingHandCursor)
        self.btn_add.setIcon(
            self.style().standardIcon(QStyle.SP_DialogYesButton)
        )
        self.btn_add.setToolTip("Add group")
        
        self.btn_rm = QToolButton(btns)
        self.btn_rm.setObjectName("miniAction")
        self.btn_rm.setText("Remove")
        self.btn_rm.setCursor(Qt.PointingHandCursor)
        self.btn_rm.setIcon(
            self.style().standardIcon(QStyle.SP_TrashIcon)
        )
        self.btn_rm.setToolTip("Remove group")
        
        b.addWidget(self.btn_add, 0)
        b.addWidget(self.btn_rm, 0)
        b.addStretch(1)

        self.editor = _ColumnMulti(
            "Group columns",
            parent=self,
            allow_reorder=True,
        )

        root.addWidget(QLabel("Groups", self), 0, 0)
        root.addWidget(QLabel("Selected columns", self), 0, 1)
        root.addWidget(self.grp_list, 1, 0)
        root.addWidget(self.editor, 1, 1)
        root.addWidget(btns, 2, 0, 1, 2)

        self.btn_add.clicked.connect(self._add_group)
        self.btn_rm.clicked.connect(self._rm_group)
        self.grp_list.currentRowChanged.connect(
            self._on_group_changed
        )
        self.editor.changed.connect(self._on_editor_changed)

    def set_available(self, cols: Sequence[str]) -> None:
        self._available = [str(c) for c in (cols or [])]
        self.editor.set_available(self._available)
        # self._sync_actions()
        # self._refresh_missing_chip()


    def set_groups(self, groups: Any) -> None:
        self._groups = []
        if isinstance(groups, list):
            for g in groups:
                if isinstance(g, list):
                    self._groups.append([str(x) for x in g])
                else:
                    self._groups.append([str(g)])

        self.grp_list.clear()
        for i in range(len(self._groups)):
            QListWidgetItem(f"Group {i + 1}", self.grp_list)

        if self.grp_list.count() > 0:
            self.grp_list.setCurrentRow(0)
        else:
            self.editor.set_values([])

    def groups(self) -> List[List[str]]:
        return [list(g) for g in self._groups]

    def missing(self) -> List[str]:
        have = set(self._available)
        out: List[str] = []
        for g in self._groups:
            for c in g:
                if c not in have:
                    out.append(c)
        return out

    def _add_group(self) -> None:
        self._groups.append([])
        QListWidgetItem(
            f"Group {len(self._groups)}",
            self.grp_list,
        )
        self.grp_list.setCurrentRow(self.grp_list.count() - 1)
        self.changed.emit()

    def _rm_group(self) -> None:
        row = self.grp_list.currentRow()
        if row < 0 or row >= len(self._groups):
            return
        self._groups.pop(row)
        self.grp_list.takeItem(row)

        if self.grp_list.count() > 0:
            new_row = min(row, self.grp_list.count() - 1)
            self.grp_list.setCurrentRow(max(new_row, 0))

        else:
            self.editor.set_values([])

        self.changed.emit()

    def _on_group_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._groups):
            self.editor.set_values([])
            return
        self.editor.set_values(self._groups[row])

    def _on_editor_changed(self) -> None:
        row = self.grp_list.currentRow()
        if row < 0 or row >= len(self._groups):
            return
        self._groups[row] = self.editor.values()
        self.changed.emit()


# ---------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------
class FeatureRegistryCard(CardBase):
    """
    Feature registry (store-driven, typo-safe).

    Provide columns via:
        card.set_dataset_columns(columns)
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="features",
            title="Feature registry",
            subtitle=(
                "Select feature columns from the dataset "
                "to avoid typos and ensure consistency."
            ),
            parent=parent,
        )
        self.store = store
        self.binder = binder
        self._cols: List[str] = []
        self._syncing = False

        self._build()
        self._wire()
        self.refresh()

    # ----------------------------
    # Public API
    # ----------------------------
    def set_dataset_columns(self, cols: Sequence[str]) -> None:
        self._cols = [str(c) for c in (cols or [])]
        self._push_available()
        self._update_status()

    def refresh(self) -> None:
        cfg = self.store.cfg
    
        self._syncing = True
        try:
            self.ed_dyn.set_values(
                getattr(cfg, "dynamic_driver_features", []) or []
            )
            self.ed_sta.set_values(
                getattr(cfg, "static_driver_features", []) or []
            )
            self.ed_fut.set_values(
                getattr(cfg, "future_driver_features", []) or []
            )
    
            self.ed_opt_cat.set_values(
                getattr(cfg, "optional_categorical_features", [])
                or []
            )
            self.ed_norm.set_values(
                getattr(cfg, "already_normalized_features", [])
                or []
            )
    
            self.ed_opt_num.set_groups(
                getattr(cfg, "optional_numeric_features", []) or []
            )
        finally:
            self._syncing = False
    
        self._update_status()


    # ----------------------------
    # UI
    # ----------------------------
    def _build(self) -> None:
        body = self.body_layout()

        top = QWidget(self)
        t = QHBoxLayout(top)
        t.setContentsMargins(0, 0, 0, 0)
        t.setSpacing(8)

        self.lbl_status = QLabel("No dataset columns", top)
        self.lbl_status.setObjectName("featStatus")
        self.lbl_status.setWordWrap(True)

        self.btn_cfg = QPushButton("Open Feature Config…", top)
        self.btn_cfg.setCursor(Qt.PointingHandCursor)

        self.btn_clear = QPushButton("Clear selections", top)
        self.btn_clear.setCursor(Qt.PointingHandCursor)

        t.addWidget(self.lbl_status, 1)
        t.addWidget(self.btn_cfg, 0)
        t.addWidget(self.btn_clear, 0)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        left = self._build_drivers(splitter)
        right = self._build_optional(splitter)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 420])


        body.addWidget(top, 0)
        body.addWidget(splitter, 1)

    def _build_drivers(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Drivers", parent)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
    
        tabs = QTabWidget(box)
        tabs.setDocumentMode(True)
        tabs.setUsesScrollButtons(True)
        tabs.tabBar().setExpanding(False)
    
        self.ed_dyn = _ColumnMulti("Dynamic", parent=box)
        self.ed_sta = _ColumnMulti("Static", parent=box)
        self.ed_fut = _ColumnMulti("Future", parent=box)
    
        tabs.addTab(self.ed_dyn, "Dynamic")
        tabs.addTab(self.ed_sta, "Static")
        tabs.addTab(self.ed_fut, "Future")
    
        hint = QLabel(
            "Tip: Dynamic = time-varying drivers • Static = spatial constants "
            "• Future = known-ahead covariates.",
            box,
        )
        hint.setWordWrap(True)
        hint.setObjectName("featHint")
    
        lay.addWidget(tabs, 1)
        lay.addWidget(hint, 0)
    
        return box

    def _build_optional(self, parent: QWidget) -> QGroupBox:
       box = QGroupBox("Stage-1 registries", parent)
       lay = QVBoxLayout(box)
       lay.setContentsMargins(10, 10, 10, 10)
       lay.setSpacing(8)
   
       tabs = QTabWidget(box)
       tabs.setDocumentMode(True)
       tabs.setUsesScrollButtons(True)
       tabs.tabBar().setExpanding(False)
   
       self.ed_opt_num = _GroupsEditor(parent=box)
   
       self.ed_opt_cat = _ColumnMulti(
           "Optional categorical",
           parent=box,
           allow_reorder=True,
       )
       self.ed_norm = _ColumnMulti(
           "Already normalized",
           parent=box,
           allow_reorder=True,
       )
   
       tabs.addTab(self.ed_opt_num, "Numeric groups")
       tabs.addTab(self.ed_opt_cat, "Categorical")
       tabs.addTab(self.ed_norm, "Normalized")
   
       lay.addWidget(tabs, 1)
       return box

    # ----------------------------
    # Wiring
    # ----------------------------
    def _wire(self) -> None:
        self.store.config_changed.connect(
            lambda _k: self.refresh()
        )
        self.store.config_replaced.connect(
            lambda _c: self.refresh()
        )

        self.ed_dyn.changed.connect(
            lambda: self._commit_list(
                "dynamic_driver_features",
                self.ed_dyn.values(),
            )
        )
        self.ed_sta.changed.connect(
            lambda: self._commit_list(
                "static_driver_features",
                self.ed_sta.values(),
            )
        )
        self.ed_fut.changed.connect(
            lambda: self._commit_list(
                "future_driver_features",
                self.ed_fut.values(),
            )
        )

        self.ed_opt_cat.changed.connect(
            lambda: self._commit_list(
                "optional_categorical_features",
                self.ed_opt_cat.values(),
            )
        )
        self.ed_norm.changed.connect(
            lambda: self._commit_list(
                "already_normalized_features",
                self.ed_norm.values(),
            )
        )

        self.ed_opt_num.changed.connect(self._commit_groups)

        self.btn_cfg.clicked.connect(self._open_dialog)
        self.btn_clear.clicked.connect(self._clear_all)

    def _commit_list(self, key: str, v: List[str]) -> None:
        if self._syncing:
            return
        self.store.patch({str(key): list(v or [])})
        self._update_status()
    
    
    def _commit_groups(self) -> None:
        if self._syncing:
            return
        self.store.patch(
            {"optional_numeric_features": self.ed_opt_num.groups()}
        )
        self._update_status()


    def _clear_all(self) -> None:
        self.store.patch(
            {
                "dynamic_driver_features": [],
                "static_driver_features": [],
                "future_driver_features": [],
                "optional_numeric_features": [],
                "optional_categorical_features": [],
                "already_normalized_features": [],
            }
        )

    # ----------------------------
    # FeatureConfigDialog bridge
    # ----------------------------
    def _open_dialog(self) -> None:
        base_cfg = getattr(self.store.cfg, "_base_cfg", {}) or {}
        overrides = self.store.snapshot_overrides()

        df = None
        try:
            import pandas as pd

            if self._cols:
                df = pd.DataFrame(columns=self._cols)
        except Exception:
            df = None

        csv = getattr(self.store.cfg, "dataset_path", None)
        csv_path = Path(str(csv)) if csv else Path(".")

        dlg = FeatureConfigDialog(
            csv_path=csv_path,
            base_cfg=base_cfg,
            current_overrides=overrides,
            df=df,
            parent=self,
        )

        if dlg.exec_() != QDialog.Accepted:
            return

        ov = dlg.get_overrides() or {}
        patch: Dict[str, Any] = {}

        mp = {
            "DYNAMIC_DRIVER_FEATURES": "dynamic_driver_features",
            "STATIC_DRIVER_FEATURES": "static_driver_features",
            "FUTURE_DRIVER_FEATURES": "future_driver_features",
            "OPTIONAL_NUMERIC_FEATURES_REGISTRY": (
                "optional_numeric_features"
            ),
            "OPTIONAL_CATEGORICAL_FEATURES_REGISTRY": (
                "optional_categorical_features"
            ),
            "ALREADY_NORMALIZED_FEATURES": (
                "already_normalized_features"
            ),
            "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC": (
                "include_censor_flags_as_dynamic"
            ),
            "INCLUDE_CENSOR_FLAGS_AS_FUTURE": (
                "include_censor_flags_as_future"
            ),
            "TIME_COL": "time_col",
            "LON_COL": "lon_col",
            "LAT_COL": "lat_col",
            "SUBSIDENCE_COL": "subs_col",
            "GWL_COL": "gwl_col",
            "H_FIELD_COL_NAME": "h_field_col",
            "CENSORING_SPECS": "censoring_specs",
            "USE_EFFECTIVE_H_FIELD": "use_effective_h_field",
        }

        for k, v in ov.items():
            if k in mp:
                patch[mp[k]] = v

        if patch:
            self.store.patch(patch)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _push_available(self) -> None:
        self.ed_dyn.set_available(self._cols)
        self.ed_sta.set_available(self._cols)
        self.ed_fut.set_available(self._cols)

        self.ed_opt_cat.set_available(self._cols)
        self.ed_norm.set_available(self._cols)
        self.ed_opt_num.set_available(self._cols)

    def _update_status(self) -> None:
        n = len(self._cols)
        if n <= 0:
            self.lbl_status.setText(
                "No dataset columns yet. "
                "Load a dataset to enable pickers."
            )
            return

        missing: List[str] = []
        missing += self.ed_dyn.missing()
        missing += self.ed_sta.missing()
        missing += self.ed_fut.missing()
        missing += self.ed_opt_cat.missing()
        missing += self.ed_norm.missing()
        missing += self.ed_opt_num.missing()

        missing = sorted(set(missing))

        if not missing:
            self.lbl_status.setText(
                f"Dataset columns: {n}. "
                "All selected features are valid."
            )
            return

        short = ", ".join(missing[:6])
        more = "" if len(missing) <= 6 else " …"
        self.lbl_status.setText(
            "Some selected features are not in the dataset: "
            f"{short}{more}"
        )
        self.lbl_status.setToolTip("\n".join(missing))
