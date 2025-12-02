# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Helpers for opening arbitrary datasets (CSV, parquet, Excel, …)
in the GeoPrior GUI.

Workflow:
    1. Ask user to pick a file.
    2. Load with fusionlab.core.io.read_data (robust multi-format).
    3. Show CsvEditDialog on the resulting DataFrame.
    4. Save the edited data as a canonical CSV under gui_runs_root/_datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from PyQt5.QtWidgets import (
    QWidget,
    QFileDialog,
    QMessageBox,
    QDialog,
)

from ....core.io import read_data
from .csv_dialog import CsvEditDialog

# Single filter including common formats handled by read_data
DATASET_FILTER = (
    "Data files (*.csv *.parq *.parquet *.xlsx *.xls "
    "*.json *.feather *.pkl *.txt);;"
    "All files (*.*)"
)


def _infer_city_name(path: Path) -> str:
    stem = path.stem.strip()
    if not stem:
        return "geoprior_city"
    return stem.replace(" ", "_")


def open_dataset_with_editor(
    parent: QWidget,
    *,
    gui_runs_root: Path,
    initial_dir: str | Path = "",
) -> Tuple[Optional[Path], Optional[pd.DataFrame], Optional[str]]:
    """
    High-level helper used by GeoPriorForecaster.

    Returns
    -------
    csv_path : Path or None
        Path to the *saved* CSV file (under gui_runs_root/_datasets),
        or None if the user cancelled or an error occurred.
    df : DataFrame or None
        The edited DataFrame (copy of what was saved).
    city_name : str or None
        Inferred city/dataset name from the original filename.
    """
    # 1) Choose a file
    path_str, _ = QFileDialog.getOpenFileName(
        parent,
        "Open dataset",
        str(initial_dir),
        DATASET_FILTER,
    )
    if not path_str:
        return None, None, None

    src_path = Path(path_str)

    # 2) Load with fusionlab.core.io.read_data
    try:
        df = read_data(str(src_path), sanitize=True)
    except Exception as exc:
        QMessageBox.critical(
            parent,
            "Dataset error",
            f"Could not read dataset:\n{src_path}\n\n{exc}",
        )
        return None, None, None

    if df.empty:
        reply = QMessageBox.question(
            parent,
            "Empty dataset?",
            "The selected file appears empty after loading.\n"
            "Do you still want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return None, None, None

    # 3) Open edit dialog backed by the DataFrame
    dlg = CsvEditDialog(df, parent)
    if dlg.exec_() != QDialog.Accepted:
        return None, None, None

    edited = dlg.edited_dataframe()

    # 4) Save as canonical CSV under gui_runs_root/_datasets
    dataset_dir = gui_runs_root / "_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    city_name = _infer_city_name(src_path)
    target_csv = dataset_dir / f"{city_name}.csv"

    # Avoid overwriting silently: add suffix if needed
    if target_csv.exists():
        i = 1
        while True:
            candidate = dataset_dir / f"{city_name}_{i}.csv"
            if not candidate.exists():
                target_csv = candidate
                break
            i += 1

    try:
        edited.to_csv(target_csv, index=False)
    except Exception as exc:
        QMessageBox.critical(
            parent,
            "Save error",
            f"Could not save edited dataset to CSV:\n"
            f"{target_csv}\n\n{exc}",
        )
        return None, None, None

    return target_csv, edited, city_name
