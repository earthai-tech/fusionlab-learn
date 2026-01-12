# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTableView,
    QStackedWidget,
    QFrame,
    QSizePolicy,
    QSpinBox,
    QSplitter,
)

from ..dialogs.csv_dialog import _PandasModel


class DataTab(QWidget):
    """
    Data tab with a left dataset library + right preview/editor actions.
    """

    # Existing requests (handled by MainWindow)
    # Compat + new name (emit both)
    request_open = pyqtSignal()
    request_open_new = pyqtSignal()

    request_edit = pyqtSignal()
    request_save = pyqtSignal()
    request_save_as = pyqtSignal()
    request_reload = pyqtSignal()

    request_load_saved = pyqtSignal(str)
    request_duplicate_saved = pyqtSignal(str)
    
    dataset_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._datasets_root: Optional[Path] = None

        self._csv_path: Optional[Path] = None
        self._df: Optional[pd.DataFrame] = None
        self._city: str = ""
        self._dirty: bool = False

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        split = QSplitter(Qt.Horizontal, self)
        root.addWidget(split, 1)

        # ------------------------------------------------------------
        # Left: Dataset library
        # ------------------------------------------------------------
        left = QFrame()
        left.setFrameShape(QFrame.StyledPanel)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(10, 10, 10, 10)
        left_lay.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Datasets"), 0)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_refresh.clicked.connect(self.refresh_library)
        title_row.addWidget(self.btn_refresh, 0)

        title_row.addStretch(1)
        left_lay.addLayout(title_row)

        self.edt_lib_search = QLineEdit()
        self.edt_lib_search.setPlaceholderText("Search…")
        self.edt_lib_search.textChanged.connect(self.refresh_library)
        left_lay.addWidget(self.edt_lib_search)

        self.list_datasets = QListWidget()
        self.list_datasets.itemDoubleClicked.connect(
            self._on_library_item_activated
        )
        left_lay.addWidget(self.list_datasets, 1)

        lib_btns = QHBoxLayout()
        self.btn_use_selected = QPushButton("Load selected")
        self.btn_use_selected.setCursor(Qt.PointingHandCursor)
        self.btn_use_selected.clicked.connect(self._load_selected_clicked)
        lib_btns.addWidget(self.btn_use_selected)

        self.btn_duplicate = QPushButton("Duplicate")
        self.btn_duplicate.setCursor(Qt.PointingHandCursor)
        self.btn_duplicate.clicked.connect(self._duplicate_selected_clicked)
        lib_btns.addWidget(self.btn_duplicate)

        left_lay.addLayout(lib_btns)

        hint = QLabel("Tip: double-click to load")
        hint.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hint.setStyleSheet("font-size:11px; opacity:0.8;")
        left_lay.addWidget(hint)

        split.addWidget(left)

        # ------------------------------------------------------------
        # Right: stack (empty vs loaded)
        # ------------------------------------------------------------
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        self._stack = QStackedWidget()
        right_lay.addWidget(self._stack, 1)

        self._build_empty_page()
        self._build_loaded_page()
        self._show_empty()

        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        split.setSizes([280, 720])

    # -----------------------------------------------------------------
    # Public API (MainWindow calls these)
    # -----------------------------------------------------------------
    def set_datasets_root(self, root: str | Path) -> None:
        self._datasets_root = Path(root)
        self.refresh_library()
        
    def set_dataset(
        self,
        csv_path: Optional[str | Path],
        df: Optional[pd.DataFrame],
        *,
        city: str = "",
        dirty: bool = False,
    ) -> None:
        self._csv_path = Path(csv_path) if csv_path else None
        self._df = df
        self._city = city or ""
        self._dirty = bool(dirty)
    
        if self._df is None:
            self._show_empty()
            self.dataset_changed.emit([])
            return
    
        self._show_loaded()
        self._refresh_loaded_labels()
        self._refresh_preview()
    
        # highlight in library
        if self._csv_path is not None:
            self.refresh_library(select_path=self._csv_path)
    
        try:
            cols = [str(c) for c in self._df.columns]
        except Exception:
            cols = []
        self.dataset_changed.emit(cols)
        

    def refresh_library(self, select_path: Optional[str | Path] = None) -> None:
        root = self._datasets_root
        self.list_datasets.clear()

        if root is None or not root.exists():
            return

        term = self.edt_lib_search.text().strip().lower()
        paths = list(root.glob("*.csv"))

        # Sort newest first
        try:
            paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            paths.sort()

        selected_norm = None
        if select_path:
            try:
                selected_norm = str(Path(select_path).resolve())
            except Exception:
                selected_norm = str(select_path)

        for p in paths:
            name = p.stem
            if term and term not in name.lower():
                continue

            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, str(p))

            self.list_datasets.addItem(item)

            if self._csv_path is not None:
                try:
                    cur = str(self._csv_path.resolve())
                except Exception:
                    cur = str(self._csv_path)
                if str(p) == str(self._csv_path) or cur == str(p):
                    item.setSelected(True)

            if selected_norm:
                try:
                    if str(p.resolve()) == selected_norm:
                        item.setSelected(True)
                        self.list_datasets.scrollToItem(item)
                except Exception:
                    pass

    def set_dirty(self, dirty: bool) -> None:
        self._dirty = bool(dirty)
        self._refresh_loaded_labels()

    def current_csv_path(self) -> Optional[Path]:
        return self._csv_path

    def current_df(self) -> Optional[pd.DataFrame]:
        return self._df

    # -----------------------------------------------------------------
    # Empty / loaded pages
    # -----------------------------------------------------------------
    def _build_empty_page(self) -> None:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(12)

        lay.addStretch(2)

        title = QLabel("No dataset loaded")
        title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        title.setStyleSheet("font-size:18px; font-weight:600;")
        lay.addWidget(title)

        msg = QLabel(
            "Load a dataset to preview, edit and save it.\n"
            "Or pick one from the library on the left."
        )
        msg.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        msg.setWordWrap(True)
        lay.addWidget(msg)

        btn = QPushButton("Load new dataset…")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedWidth(240)
        btn.clicked.connect(self._emit_open)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

        lay.addStretch(3)

        self._stack.addWidget(page)
        self._empty_page = page

    def _build_loaded_page(self) -> None:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # Top action row
        top = QHBoxLayout()
        top.setSpacing(8)

        self.btn_load = QPushButton("Load…")
        self.btn_edit = QPushButton("Edit…")
        self.btn_save = QPushButton("Save")
        self.btn_save_as = QPushButton("Save as…")
        self.btn_reload = QPushButton("Reload")

        for b in (
            self.btn_load,
            self.btn_edit,
            self.btn_save,
            self.btn_save_as,
            self.btn_reload,
        ):
            b.setCursor(Qt.PointingHandCursor)

        self.btn_load.clicked.connect(self._emit_open)
        self.btn_edit.clicked.connect(self.request_edit.emit)
        self.btn_save.clicked.connect(self.request_save.emit)
        self.btn_save_as.clicked.connect(self.request_save_as.emit)
        self.btn_reload.clicked.connect(self.request_reload.emit)

        top.addWidget(self.btn_load)
        top.addWidget(self.btn_edit)
        top.addWidget(self.btn_save)
        top.addWidget(self.btn_save_as)
        top.addWidget(self.btn_reload)
        top.addStretch(1)

        self.lbl_state = QLabel("")
        self.lbl_state.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.lbl_state)

        outer.addLayout(top)

        # Info + preview
        info = QHBoxLayout()
        self.lbl_city = QLabel("City: -")
        self.lbl_path = QLabel("Path: -")
        self.lbl_shape = QLabel("Rows/Cols: -")

        for w in (self.lbl_city, self.lbl_path, self.lbl_shape):
            w.setWordWrap(True)
            info.addWidget(w, 1)

        outer.addLayout(info)

        preview_row = QHBoxLayout()
        self.edt_col_filter = QLineEdit()
        self.edt_col_filter.setPlaceholderText("Filter columns…")
        self.edt_col_filter.textChanged.connect(self._refresh_preview)

        self.spin_preview = QSpinBox()
        self.spin_preview.setMinimum(50)
        self.spin_preview.setMaximum(50_000)
        self.spin_preview.setSingleStep(50)
        self.spin_preview.setValue(500)
        self.spin_preview.valueChanged.connect(self._refresh_preview)

        preview_row.addWidget(self.edt_col_filter, 1)
        preview_row.addWidget(QLabel("Rows:"), 0)
        preview_row.addWidget(self.spin_preview, 0)

        outer.addLayout(preview_row)

        self.table = QTableView()
        self.table.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QTableView.NoEditTriggers)
        outer.addWidget(self.table, 1)

        self._stack.addWidget(page)
        self._loaded_page = page

    # -----------------------------------------------------------------
    # Library interactions
    # -----------------------------------------------------------------
    def _emit_open(self) -> None:
        # Old app.py expects request_open
        self.request_open.emit()
        # New name (if you later switch app.py)
        self.request_open_new.emit()

    def _selected_library_path(self) -> Optional[str]:
        item = self.list_datasets.currentItem()
        if item is None:
            return None
        val = item.data(Qt.UserRole)
        return str(val) if val else None

    def _on_library_item_activated(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.UserRole)
        if p:
            self.request_load_saved.emit(str(p))

    def _load_selected_clicked(self) -> None:
        p = self._selected_library_path()
        if p:
            self.request_load_saved.emit(p)

    def _duplicate_selected_clicked(self) -> None:
        p = self._selected_library_path()
        if p:
            self.request_duplicate_saved.emit(p)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _show_empty(self) -> None:
        self._stack.setCurrentWidget(self._empty_page)

    def _show_loaded(self) -> None:
        self._stack.setCurrentWidget(self._loaded_page)

    def _refresh_loaded_labels(self) -> None:
        df = self._df
        if df is None:
            return

        self.lbl_city.setText(f"City: {self._city or '-'}")
        self.lbl_path.setText(
            f"Path: {str(self._csv_path) if self._csv_path else '-'}"
        )
        self.lbl_shape.setText(
            f"Rows/Cols: {len(df)} / {df.shape[1]}"
        )

        self.lbl_state.setText(
            "Unsaved changes" if self._dirty else "Saved"
        )
        self.btn_save.setEnabled(self._dirty)

    def _refresh_preview(self) -> None:
        if self._df is None:
            return

        df = self._df
        term = self.edt_col_filter.text().strip().lower()

        if term:
            cols = [c for c in df.columns if term in str(c).lower()]
            if cols:
                df = df.loc[:, cols]

        n = int(self.spin_preview.value())
        view_df = df.head(n)

        model = _PandasModel(view_df, parent=self)
        self.table.setModel(model)
        self.table.resizeColumnsToContents()
