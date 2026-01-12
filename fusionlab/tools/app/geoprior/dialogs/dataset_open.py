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
    QApplication,
)

from .....core.io import read_data
from .pop_progress import PopProgressDialog  
from .csv_dialog import CsvEditDialog

# Single filter including common formats handled by read_data
DATASET_FILTER = (
    "Data files (*.csv *.parq *.parquet *.xlsx *.xls "
    "*.json *.feather *.pkl *.txt);;"
    "All files (*.*)"
)

def save_dataframe_to_csv(
    parent: QWidget,
    df: pd.DataFrame,
    target_csv: Path,
    *,
    chunk_size: int = 50_000,
) -> None:
    """
    Public wrapper around the chunked CSV save helper.
    """
    _save_csv_with_progress(
        parent,
        df,
        target_csv,
        chunk_size=chunk_size,
    )

def _load_dataset_with_progress(
    parent: QWidget,
    src_path: Path,
) -> Optional[pd.DataFrame]:
    """
    Load a dataset using read_data with a nice popup progress dialog.

    Returns a DataFrame or None if the user cancels.
    """
    dlg = PopProgressDialog(
        parent,
        title="Loading dataset",
        text=f"Reading data from: {src_path.name}",
        minimum=0,
        maximum=100,
        cancelable=True,
    )
    dlg.show()
    QApplication.processEvents()

    progress_cb = dlg.as_fraction_callback()

    try:
        df = read_data(
            str(src_path),
            sanitize=True,
            logger=None,           # or pass a GUI logger if you want
            progress_hook=progress_cb,
        )
    except Exception:
        dlg.finish()
        raise
    else:
        dlg.finish()

    if dlg.was_canceled():
        # Note: cancellation currently doesn't interrupt read_data,
        # but this ensures the GUI treats the operation as aborted.
        return None

    return df

def _run_with_busy_progress(
    parent: QWidget,
    title: str,
    label: str,
    func,
    *args,
    **kwargs,
):
    """
    Run a blocking function while showing an *indeterminate*
    progress dialog (busy mode).

    Returns the function's result, or re-raises any exception.
    """
    dlg = PopProgressDialog(
        parent,
        title=title,
        text=label,
        minimum=0,
        maximum=0,          # 0..0 = indeterminate style
        cancelable=False,   # no cancel for "busy" helper
    )
    dlg.show()
    QApplication.processEvents()

    try:
        result = func(*args, **kwargs)
    finally:
        dlg.finish()
        QApplication.processEvents()

    return result


def _save_csv_with_progress(
    parent: QWidget,
    df: pd.DataFrame,
    target_csv: Path,
    *,
    chunk_size: int = 50_000,
) -> None:
    """
    Save a DataFrame to CSV in chunks, with a PopProgressDialog
    showing real row-level progress.

    Raises RuntimeError if the user cancels, or any I/O exception.
    """
    total_rows = len(df)
    if total_rows == 0:
        # Trivial case
        df.to_csv(target_csv, index=False)
        return

    dlg = PopProgressDialog(
        parent,
        title="Saving dataset",
        text="Saving edited dataset to CSV…",
        minimum=0,
        maximum=total_rows,
        cancelable=True,
    )
    dlg.show()
    QApplication.processEvents()

    # Ensure we don't leave a half-written file if cancelled/error
    if target_csv.exists():
        target_csv.unlink()

    written = 0
    first_chunk = True

    try:
        with target_csv.open("w", encoding="utf-8", newline="") as f:
            for start in range(0, total_rows, chunk_size):
                if dlg.was_canceled():
                    raise RuntimeError("Save cancelled by user.")

                end = min(start + chunk_size, total_rows)
                chunk = df.iloc[start:end]

                # Write header only once
                chunk.to_csv(
                    f,
                    index=False,
                    header=first_chunk,
                )
                first_chunk = False

                written = end
                dlg.set_value(
                    written,
                    message=(
                        f"Saving edited dataset… "
                        f"{written:,} / {total_rows:,} rows"
                    ),
                )
    except Exception:
        # If something goes wrong, remove partial file
        if target_csv.exists():
            target_csv.unlink()
        raise
    finally:
        dlg.finish()
        QApplication.processEvents()


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

    # 2) Load with fusionlab.core.io.read_data, with progress popup.
    try:
        df = _load_dataset_with_progress(parent, src_path)
    except Exception as exc:
        QMessageBox.critical(
            parent,
            "Dataset error",
            f"Could not read dataset:\n{src_path}\n\n{exc}",
        )
        return None, None, None

    if df is None:
        # User cancelled during load
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
        _save_csv_with_progress(parent, edited, target_csv)
    except RuntimeError as exc:
        # User cancelled save
        QMessageBox.information(
            parent,
            "Save cancelled",
            str(exc),
        )
        return None, None, None
    except Exception as exc:
        QMessageBox.critical(
            parent,
            "Save error",
            f"Could not save edited dataset to CSV:\n"
            f"{target_csv}\n\n{exc}",
        )
        return None, None, None

    return target_csv, edited, city_name
