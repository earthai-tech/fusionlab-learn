# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
App dialog
"""

from __future__ import annotations

import pandas as pd

from PyQt5.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
)
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QDialog,
    QTableView,
    QMessageBox,
    QAction,
    QToolBar,
    QInputDialog,
    QStyle,          
)

# SECONDARY_COLOR = "#3399ff"
from ..styles import SECONDARY_TBLUE 


__all__ = ["CsvEditDialog", "_PandasModel"]


class CsvEditDialog(QDialog):
    """A dialog window for previewing and editing a CSV/DataFrame.

    Lightweight, non-destructive editor:
    - delete row(s) / col(s)
    - rename column
    - sort ascending / descending
    - drop rows with NA in selected column(s)
    - 
    selected column to numeric
    - reset index

    Edits are applied to an internal DataFrame copy and only
    committed to the caller when "Save / Apply" is clicked.
    """

    def __init__(
        self,
        source: str | pd.DataFrame,
        parent=None,
        *,
        preview_rows: int = 200,
    ):
        super().__init__(parent)
        self.setWindowTitle("Preview & Editing Data")
        self.resize(820, 400)

        # --- load full DataFrame -------------------------------------
        try:
            if isinstance(source, pd.DataFrame):
                # In-memory dataset (already loaded via read_data)
                self._df_full = source.copy()
            else:
                # Backwards-compatible: path to CSV on disk
                csv_path = str(source)
                self._df_full = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Data error", str(e))
            self._df_full = pd.DataFrame()
            self.reject()
            return

        # slice for the *table* only (view)
        self._view_rows = min(preview_rows, len(self._df_full))
        self._df_view = self._df_full.head(self._view_rows)

        vbox = QVBoxLayout(self)

        # ----------------- toolbar + actions ----------------------
        tb = QToolBar()
        tb.setMovable(False)

        # Standard icon helper (same idea as in main window)
        def _std_icon(sp: QStyle.StandardPixmap):
            return self.style().standardIcon(sp)

        # Base actions
        act_del_row = QAction(
            _std_icon(QStyle.SP_TrashIcon),
            "Delete row(s)",
            self,
        )
        act_del_col = QAction(
            _std_icon(QStyle.SP_TrashIcon),
            "Delete col(s)",
            self,
        )
        act_rename = QAction(
            _std_icon(QStyle.SP_FileDialogNewFolder),
            "Rename column",
            self,
        )

        # New actions with icons
        act_sort_asc = QAction(
            _std_icon(QStyle.SP_ArrowUp),
            "Sort ↑ (selected column)",
            self,
        )
        act_sort_desc = QAction(
            _std_icon(QStyle.SP_ArrowDown),
            "Sort ↓ (selected column)",
            self,
        )
        act_drop_na = QAction(
            _std_icon(QStyle.SP_DialogDiscardButton),
            "Drop rows with NA (selected col(s))",
            self,
        )
        act_to_numeric = QAction(
            _std_icon(QStyle.SP_DialogApplyButton),
            "Convert to numeric (selected col)",
            self,
        )
        act_reset_index = QAction(
            _std_icon(QStyle.SP_BrowserReload),
            "Reset index",
            self,
        )

        tb.addAction(act_del_row)
        tb.addAction(act_del_col)
        tb.addAction(act_rename)
        tb.addSeparator()
        tb.addAction(act_sort_asc)
        tb.addAction(act_sort_desc)
        tb.addSeparator()
        tb.addAction(act_drop_na)
        tb.addAction(act_to_numeric)
        tb.addSeparator()
        tb.addAction(act_reset_index)

        # Hover styling using SECONDARY_COLOR from styles.py
        tb.setStyleSheet(
            f"""
            QToolBar {{
                border: 0px;
                spacing: 4px;
            }}
            QToolButton {{
                padding: 3px 8px;
                border-radius: 4px;
            }}
            QToolButton:hover {{
                background-color: {SECONDARY_TBLUE};
            }}
            """
        )

        vbox.addWidget(tb)

        # ----------------- table view -----------------------------
        self.table = QTableView()
        self.model = _PandasModel(self._df_view)
        self.table.setModel(self.model)
        self.table.setSelectionMode(QTableView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        vbox.addWidget(self.table, 1)

        # ----------------- buttons -------------------------------
        hbtn = QHBoxLayout()
        hbtn.addStretch(1)
        btn_save = QPushButton("Save / Apply")
        btn_cancel = QPushButton("Cancel")
        hbtn.addWidget(btn_save)
        hbtn.addWidget(btn_cancel)
        vbox.addLayout(hbtn)

        # ----------------- signals -------------------------------
        act_del_row.triggered.connect(self._delete_rows)
        act_del_col.triggered.connect(self._delete_cols)
        act_rename.triggered.connect(self._rename_col)

        act_sort_asc.triggered.connect(
            lambda: self._sort_by_selected(ascending=True)
        )
        act_sort_desc.triggered.connect(
            lambda: self._sort_by_selected(ascending=False)
        )
        act_drop_na.triggered.connect(self._drop_na_in_selected_cols)
        act_to_numeric.triggered.connect(
            self._convert_selected_to_numeric
        )
        act_reset_index.triggered.connect(self._reset_index)

        btn_save.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    # ------------------------------------------------------------------
    # Core edit operations
    # ------------------------------------------------------------------
    def _selected_rows(self) -> set[int]:
        return {
            ix.row()
            for ix in self.table.selectionModel().selectedIndexes()
        }

    def _selected_cols(self) -> set[int]:
        return {
            ix.column()
            for ix in self.table.selectionModel().selectedIndexes()
        }

    def _delete_rows(self):
        rows = self._selected_rows()
        if not rows:
            return
        # translate view-index → full-df index
        full_idx = self._df_full.index[list(rows)]
        self._df_full.drop(index=full_idx, inplace=True)
        self._refresh_view()

    def _delete_cols(self):
        cols = self._selected_cols()
        if not cols:
            return
        cols_to_drop = self._df_full.columns[list(cols)]
        self._df_full.drop(columns=cols_to_drop, inplace=True)
        self._refresh_view()

    def _rename_col(self):
        cols = self._selected_cols()
        if len(cols) != 1:
            QMessageBox.information(
                self,
                "Rename column",
                "Select exactly one column.",
            )
            return
        col_ix = cols.pop()
        old = self._df_full.columns[col_ix]
        new, ok = QInputDialog.getText(
            self,
            "Rename column",
            f"New name for “{old}”:",
        )
        if ok and new:
            self._df_full.rename(columns={old: new}, inplace=True)
            self._refresh_view()

    # ------------------------------------------------------------------
    # New helpers
    # ------------------------------------------------------------------
    def _sort_by_selected(self, *, ascending: bool = True) -> None:
        """Sort the full DataFrame by a single selected column."""
        cols = self._selected_cols()
        if len(cols) != 1:
            QMessageBox.information(
                self,
                "Sort",
                "Select exactly one column to sort by.",
            )
            return
        col_ix = next(iter(cols))
        col_name = self._df_full.columns[col_ix]
        # Stable sort to preserve relative order of equal rows
        self._df_full.sort_values(
            by=col_name,
            ascending=ascending,
            inplace=True,
            kind="mergesort",
        )
        self._refresh_view()

    def _drop_na_in_selected_cols(self) -> None:
        """Drop rows with NA in any of the selected columns."""
        cols = self._selected_cols()
        if not cols:
            QMessageBox.information(
                self,
                "Drop NA",
                "Select at least one column.",
            )
            return
        col_names = list(self._df_full.columns[list(cols)])
        before = len(self._df_full)
        self._df_full.dropna(subset=col_names, inplace=True)
        after = len(self._df_full)
        dropped = before - after
        self._refresh_view()

        if dropped > 0:
            QMessageBox.information(
                self,
                "Drop NA",
                f"Dropped {dropped} row(s) with missing values "
                f"in selected column(s).",
            )

    def _convert_selected_to_numeric(self) -> None:
        """Convert a single selected column to numeric (coerce errors)."""
        cols = self._selected_cols()
        if len(cols) != 1:
            QMessageBox.information(
                self,
                "Convert to numeric",
                "Select exactly one column.",
            )
            return
        col_ix = cols.pop()
        col_name = self._df_full.columns[col_ix]

        reply = QMessageBox.question(
            self,
            "Convert to numeric",
            f"Convert column “{col_name}” to numeric?\n"
            "Non-numeric values will be set to NaN.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        self._df_full[col_name] = pd.to_numeric(
            self._df_full[col_name],
            errors="coerce",
        )
        self._refresh_view()

    def _reset_index(self) -> None:
        """Reset index after deletions / sorting."""
        self._df_full.reset_index(drop=True, inplace=True)
        self._refresh_view()

    # ------------------------------------------------------------------
    def _refresh_view(self):
        """Re-slice top N rows *after* edits and refresh model."""
        self._df_view = self._df_full.head(self._view_rows)
        self.model.beginResetModel()
        self.model._df = self._df_view
        self.model.endResetModel()

    # -public API
    def edited_dataframe(self) -> pd.DataFrame:
        """
        Return the **full** (possibly edited) DataFrame.
        Caller should copy if it wants to keep a private version.
        """
        return self._df_full.copy()


class _PandasModel(QAbstractTableModel):
    """A Qt Table Model for exposing a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    # -- basic shape -------
    def rowCount(self, _=QModelIndex()):
        return len(self._df)

    def columnCount(self, _=QModelIndex()):
        return self._df.shape[1]

    # -- data ↔ Qt -
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return str(self._df.iat[index.row(), index.column()])
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            self._df.iat[index.row(), index.column()] = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def flags(self, index):
        return (
            Qt.ItemIsSelectable
            | Qt.ItemIsEnabled
            | Qt.ItemIsEditable
        )

    # -- header labels
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return (
            self._df.columns[section]
            if orientation == Qt.Horizontal
            else str(section)
        )

    # exposed to outside world
    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df
