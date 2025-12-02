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
)


__all__= ["CsvEditDialog", "_PandasModel"]

class CsvEditDialog(QDialog):
    """A dialog window for previewing and editing a CSV file.

    This class provides a lightweight, non-destructive editing
    environment for a loaded dataset. It displays a preview of the
    data in a table and offers tools to delete rows, delete columns,
    and rename columns.

    Changes are made to an internal copy of the DataFrame and are only
    finalized and returned if the user clicks "Save / Apply".

    Parameters
    ----------
    source : str or pandas.DataFrame
        Either a path to a CSV file *or* an in-memory DataFrame.
        When a DataFrame is provided, it is copied internally and
        no file I/O is performed.
    parent : QWidget, optional
        The parent widget for this dialog, by default None.
    preview_rows : int, default=200
        The maximum number of rows to display in the table view. This
        is a performance optimization to prevent lag with very large
        files, while still allowing edits on the complete, underlying
        DataFrame.

    Methods
    -------
    edited_dataframe()
        Returns the full DataFrame with all user edits applied.

    See Also
    --------
    PyQt5.QtWidgets.QDialog : The base class for dialog windows.
    _PandasModel : The data model used to populate the table view.
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
        self._df_view   = self._df_full.head(self._view_rows)

        vbox = QVBoxLayout(self)

        # toolbar
        tb = QToolBar()
        act_del_row = QAction("Delete row(s)", self)
        act_del_col = QAction("Delete col(s)", self)
        act_rename  = QAction("Rename column", self)
        tb.addAction(act_del_row)
        tb.addAction(act_del_col)
        tb.addAction(act_rename)
        vbox.addWidget(tb)

        # table
        self.table = QTableView()
        self.model = _PandasModel(self._df_view)
        self.table.setModel(self.model)
        self.table.setSelectionMode(QTableView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        vbox.addWidget(self.table, 1)

        # buttons
        hbtn = QHBoxLayout()
        hbtn.addStretch(1)
        btn_save = QPushButton("Save / Apply")
        btn_cancel = QPushButton("Cancel")
        hbtn.addWidget(btn_save)
        hbtn.addWidget(btn_cancel)
        vbox.addLayout(hbtn)

        # signals
        act_del_row.triggered.connect(self._delete_rows)
        act_del_col.triggered.connect(self._delete_cols)
        act_rename.triggered.connect(self._rename_col)
        btn_save.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def _delete_rows(self):
        rows = {ix.row() for ix in self.table.selectionModel().selectedIndexes()}
        if not rows: return
        # translate view-index → full-df index
        full_idx = self._df_full.index[list(rows)]
        self._df_full.drop(index=full_idx, inplace=True)
        self._refresh_view()

    def _delete_cols(self):
        cols = {ix.column() for ix in self.table.selectionModel().selectedIndexes()}
        if not cols: return
        cols_to_drop = self._df_full.columns[list(cols)]
        self._df_full.drop(columns=cols_to_drop, inplace=True)
        self._refresh_view()

    def _rename_col(self):
        cols = {ix.column() for ix in self.table.selectionModel().selectedIndexes()}
        if len(cols) != 1:
            QMessageBox.information(self, "Rename column",
                                     "Select exactly one column.")
            return
        col_ix = cols.pop()
        old = self._df_full.columns[col_ix]
        new, ok = QInputDialog.getText(self, "Rename column",
                                       f"New name for “{old}”:")
        if ok and new:
            self._df_full.rename(columns={old: new}, inplace=True)
            self._refresh_view()

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
    """A Qt Table Model for exposing a pandas DataFrame.

    This class acts as a bridge between the data model (a pandas
    DataFrame) and the view component (a QTableView). It provides
    the necessary interface that allows Qt to read, display, and
    modify the data from the DataFrame in a table widget.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be displayed and managed by the model.
    parent : QObject, optional
        The parent Qt object for this model, by default None.

    Attributes
    ----------
    _df : pandas.DataFrame
        The internal reference to the DataFrame being managed.

    See Also
    --------
    PyQt5.QtCore.QAbstractTableModel : The base class for custom table models.
    """

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df                

    # -- basic shape -------
    def rowCount   (self, _=QModelIndex()): return len(self._df)
    def columnCount(self, _=QModelIndex()): return self._df.shape[1]

    # -- data ↔ Qt -
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid(): return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return str(self._df.iat[index.row(), index.column()])
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            self._df.iat[index.row(), index.column()] = value
            self.dataChanged.emit(index, index, [role]); return True
        return False

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    # -- header labels 
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return (self._df.columns[section] if orientation == Qt.Horizontal
                else str(section))

    # helper for column rename
    def rename_column(self, col_ix: int, new_name: str):
        self.beginResetModel()
        self._df.columns = (
            list(self._df.columns[:col_ix]) + [new_name] +
            list(self._df.columns[col_ix + 1 :])
        )
        self.endResetModel()

    # exposed to outside world
    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df