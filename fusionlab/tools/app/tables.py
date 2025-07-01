# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

""" Gui tables."""
from __future__ import annotations 
import pandas as pd 

from PyQt5.QtCore    import ( 
    Qt, 
    QAbstractTableModel, 
    QModelIndex, 
)

__all__ = ["_PandasModel"]

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