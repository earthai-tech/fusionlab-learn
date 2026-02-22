# geoprior/ui/dataset_open.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Helpers for opening arbitrary datasets (CSV, parquet, Excel,
...) in the GeoPrior GUI.

Workflow:
    1. Ask user to pick a file.
    2. Load with fusionlab.core.io.read_data.
    3. Show CsvEditDialog for editing.
    4. Save edited data as canonical CSV under:
       gui_runs_root/_datasets/<city>.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMessageBox,
    QWidget,
)

from .....core.io import read_data
from .csv_dialog import CsvEditDialog
from .pop_progress import PopProgressDialog


# Single filter including common formats handled by read_data
DATASET_FILTER = (
    "Data files ("
    "*.csv *.parq *.parquet *.xlsx *.xls "
    "*.json *.feather *.pkl *.txt);;"
    "All files (*.*)"
)

_INVALID_SEP_RE = re.compile(r"[_\s/\\]+")
_ALLOWED_RE = re.compile(r"[^a-z0-9\-]+")
_DASH_RE = re.compile(r"-{2,}")


def save_dataframe_to_csv(
    parent: QWidget,
    df: pd.DataFrame,
    target_csv: Path,
    *,
    chunk_size: int = 50_000,
) -> None:
    """Public wrapper around the chunked CSV save helper."""
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
    Load a dataset using read_data with a popup progress.

    Returns a DataFrame or None if user cancels.
    """
    dlg = PopProgressDialog(
        parent,
        title="Loading dataset",
        text=(
            "Reading data from:\n"
            f"{src_path.name}"
        ),
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
            logger=None,
            progress_hook=progress_cb,
        )
    except Exception:
        dlg.finish()
        raise
    else:
        dlg.finish()

    if dlg.was_canceled():
        return None

    return df


def _save_csv_with_progress(
    parent: QWidget,
    df: pd.DataFrame,
    target_csv: Path,
    *,
    chunk_size: int = 50_000,
) -> None:
    """
    Save a DataFrame to CSV in chunks, with a progress
    dialog showing row-level progress.

    Raises RuntimeError if user cancels.
    """
    total_rows = len(df)
    if total_rows == 0:
        df.to_csv(target_csv, index=False)
        return

    dlg = PopProgressDialog(
        parent,
        title="Saving dataset",
        text="Saving edited dataset to CSV...",
        minimum=0,
        maximum=total_rows,
        cancelable=True,
    )
    dlg.show()
    QApplication.processEvents()

    if target_csv.exists():
        target_csv.unlink()

    written = 0
    first_chunk = True

    try:
        with target_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            for start in range(0, total_rows, chunk_size):
                if dlg.was_canceled():
                    raise RuntimeError(
                        "Save cancelled by user."
                    )

                end = min(start + chunk_size, total_rows)
                chunk = df.iloc[start:end]

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
                        "Saving edited dataset...\n"
                        f"{written:,} / {total_rows:,} rows"
                    ),
                )
    except Exception:
        if target_csv.exists():
            target_csv.unlink()
        raise
    finally:
        dlg.finish()
        QApplication.processEvents()


def _default_normalize_city(raw: str) -> str:
    """
    Fallback normalizer if caller does not provide one.

    Keeps city safe for folder naming:
    - lowercase
    - spaces/_/slashes -> "-"
    - allow [a-z0-9-] only
    """
    s = (raw or "").strip().lower()
    if not s:
        return ""

    s = _INVALID_SEP_RE.sub("-", s)
    s = _ALLOWED_RE.sub("", s)
    s = _DASH_RE.sub("-", s)
    s = s.strip("-")

    return s


def _resolve_city_key(
    *,
    src_path: Path,
    city_hint: Optional[str],
    normalize_city: Optional[Callable[[str], str]],
) -> Tuple[str, str]:
    """
    Resolve canonical city key.

    Returns
    -------
    city_key, message
    """
    raw = (city_hint or "").strip()
    if not raw:
        raw = src_path.stem.strip()

    if not raw:
        raw = "geoprior_city"

    norm = normalize_city or _default_normalize_city
    key = norm(raw)

    if not key:
        return "geoprior_city", "City fallback applied."

    if key != raw.lower():
        msg = f"City normalized to '{key}'."
        return key, msg

    return key, ""


def open_dataset_with_editor(
    parent: QWidget,
    *,
    gui_runs_root: Path,
    initial_dir: str | Path = "",
    city_hint: Optional[str] = None,
    normalize_city: Optional[
        Callable[[str], str]
    ] = None,
    city_message_hook: Optional[
        Callable[[str], None]
    ] = None,
) -> Tuple[
    Optional[Path],
    Optional[pd.DataFrame],
    Optional[str],
]:
    """
    High-level helper used by GeoPriorForecaster.

    Returns
    -------
    csv_path:
        Saved CSV path under gui_runs_root/_datasets,
        or None if cancelled/error.
    df:
        Edited DataFrame (copy of what was saved),
        or None.
    city_key:
        Canonical city key used for naming, or None.
    """
    path_str, _ = QFileDialog.getOpenFileName(
        parent,
        "Open dataset",
        str(initial_dir),
        DATASET_FILTER,
    )
    if not path_str:
        return None, None, None

    src_path = Path(path_str)

    try:
        df = _load_dataset_with_progress(parent, src_path)
    except Exception as exc:
        QMessageBox.critical(
            parent,
            "Dataset error",
            (
                "Could not read dataset:\n"
                f"{src_path}\n\n{exc}"
            ),
        )
        return None, None, None

    if df is None:
        return None, None, None

    if df.empty:
        reply = QMessageBox.question(
            parent,
            "Empty dataset?",
            (
                "The selected file appears empty.\n"
                "Do you still want to continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return None, None, None

    dlg = CsvEditDialog(df, parent)
    if dlg.exec_() != QDialog.Accepted:
        return None, None, None

    edited = dlg.edited_dataframe()

    dataset_dir = gui_runs_root / "_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    city_key, city_msg = _resolve_city_key(
        src_path=src_path,
        city_hint=city_hint,
        normalize_city=normalize_city,
    )
    if city_msg and city_message_hook is not None:
        city_message_hook(city_msg)

    target_csv = dataset_dir / f"{city_key}.csv"

    if target_csv.exists():
        i = 1
        while True:
            cand = dataset_dir / f"{city_key}_{i}.csv"
            if not cand.exists():
                target_csv = cand
                break
            i += 1

    try:
        _save_csv_with_progress(parent, edited, target_csv)
    except RuntimeError as exc:
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
            (
                "Could not save edited dataset:\n"
                f"{target_csv}\n\n{exc}"
            ),
        )
        return None, None, None

    return target_csv, edited, city_key
