# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Dialog to choose a dataset CSV when multiple GUI-managed datasets
# are available for a given city.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialogButtonBox,
    QWidget,
    QMessageBox,
)

from ..services.dataset_resolver import (
    DatasetCandidate,
    collect_datasets_for_city,
)


class DatasetChoiceDialog(QDialog):
    """
    Simple, modern dialog to let the user choose a dataset CSV.

    It shows one row per DatasetCandidate with:
        - City
        - Dataset filename
        - Location (results root / default root)
        - Last modified timestamp
    """

    def __init__(
        self,
        parent: Optional[QWidget],
        city: str,
        candidates: List[DatasetCandidate],
    ) -> None:
        super().__init__(parent)

        self._city = city
        self._candidates = candidates

        self.setWindowTitle("Select dataset for Stage-1")
        self.setModal(True)
        self.setMinimumWidth(680)

        layout = QVBoxLayout(self)

        title = QLabel(
            f"<b>Multiple datasets found for city:</b> {city}"
        )
        title.setTextFormat(Qt.RichText)
        layout.addWidget(title)

        subtitle = QLabel(
            "Please choose which dataset you want to use for "
            "Stage-1 preprocessing."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        table = QTableWidget(len(candidates), 4, self)
        table.setHorizontalHeaderLabels(
            ["City", "Dataset file", "Location", "Last modified"]
        )
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.verticalHeader().setVisible(False)

        # Make the filename column stretch, others fit contents
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        for row, cand in enumerate(candidates):
            try:
                ts = cand.mtime
                mtime = datetime.fromtimestamp(ts).strftime(
                    "%Y-%m-%d %H:%M"
                ) if ts > 0 else ""
            except Exception:
                mtime = ""

            items = [
                QTableWidgetItem(cand.city),
                QTableWidgetItem(cand.path.name),
                QTableWidgetItem(cand.pretty_root()),
                QTableWidgetItem(mtime),
            ]
            for col, it in enumerate(items):
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                table.setItem(row, col, it)

        # Pre-select newest (row 0, candidates already sorted by mtime)
        if candidates:
            table.selectRow(0)

        table.cellDoubleClicked.connect(self._accept_from_double_click)

        layout.addWidget(table)
        self._table = table

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _accept_from_double_click(self, row: int, col: int) -> None:
        self._table.selectRow(row)
        self.accept()

    def selected_candidate(self) -> Optional[DatasetCandidate]:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._candidates):
            return None
        return self._candidates[row]


# ----------------------------------------------------------------------
# High-level helper following your 4-step logic
# ----------------------------------------------------------------------
def choose_dataset_for_city(
    parent: QWidget,
    city: str,
    results_root: Path,
) -> Optional[str]:
    """
    Decide which dataset CSV to use for Stage-1.

    Logic
    -----
    1. Look under ``results_root/_datasets``:
       - If exactly one match → use it (no dialog).
       - If > 1 → show a choice dialog.

    2. If ``_datasets`` does not exist or has no match in results_root,
       look under ``Path.home() / ".fusionlab_runs" / "_datasets"``:
       - If exactly one match → use it (no dialog).
       - If > 1 → show a choice dialog.

    3. If no datasets are found in either place → show a warning and
       return ``None``.

    Returns
    -------
    str or None
        Absolute path to the chosen CSV, or ``None`` if the user
        cancelled or nothing was found.
    """
    results_root = Path(results_root)
    default_root = Path.home() / ".fusionlab_runs"

    # 1) Primary: current GUI results root
    primary = collect_datasets_for_city(
        root=results_root,
        city=city,
        root_kind="results_root",
    )

    if primary:
        if len(primary) == 1:
            # Single dataset → auto-pick
            return str(primary[0].path)

        dlg = DatasetChoiceDialog(parent, city, primary)
        if dlg.exec_() == QDialog.Accepted:
            cand = dlg.selected_candidate()
            if cand is not None:
                return str(cand.path)
        # Cancel → None
        return None

    # 2) Fallback: default runs root (~/.fusionlab_runs)
    secondary = collect_datasets_for_city(
        root=default_root,
        city=city,
        root_kind="default_root",
    )

    if secondary:
        if len(secondary) == 1:
            return str(secondary[0].path)

        dlg = DatasetChoiceDialog(parent, city, secondary)
        if dlg.exec_() == QDialog.Accepted:
            cand = dlg.selected_candidate()
            if cand is not None:
                return str(cand.path)
        return None

    # 3) Nothing anywhere → ask user to pick a file manually
    QMessageBox.warning(
        parent,
        "No datasets found",
        (
            "No saved datasets were found for this city in either:\n"
            f"  • {results_root / '_datasets'}\n"
            f"  • {default_root / '_datasets'}\n\n"
            "Please open a dataset from the Train tab first, or "
            "manually select a training CSV."
        ),
    )
    return None
