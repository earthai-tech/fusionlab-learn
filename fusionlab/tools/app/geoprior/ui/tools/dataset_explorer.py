# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Dataset explorer tool for the Tools tab.

Reads the *active* dataset from the main GeoPrior GUI (if available),
or lets the user pick a dataset under ``<results_root>/_datasets``.

Shows:
- basic shape (rows / columns);
- year range (based on GeoPriorConfig.time_col if present);
- per-column missing-value counts and percentages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QPlainTextEdit,  
)

from ...styles import SECONDARY_TBLUE
from ...dialogs import choose_dataset_for_city  


class DatasetExplorerTool(QWidget):
    """
    Lightweight, read-only dataset inspector.

    Parameters
    ----------
    app_ctx : object, optional
        Reference to the main :class:`GeoPriorForecaster` window.
        Used to read:
        - ``csv_path`` and ``_edited_df`` (active dataset);
        - ``geo_cfg`` (column names, time_col, etc.);
        - ``gui_runs_root`` and ``city_edit`` (for _datasets lookup).
    parent : QWidget, optional
        Standard Qt parent.
    """

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_ctx = app_ctx
        self._df: Optional[pd.DataFrame] = None
        self._csv_path: Optional[Path] = None

        self._init_ui()
        self._refresh_from_app()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- Header row: status + buttons --------------------------------
        header = QHBoxLayout()
        header.setSpacing(8)

        self.lbl_status = QLabel("No dataset loaded.")
        self.lbl_status.setStyleSheet("font-weight: 600;")
        header.addWidget(self.lbl_status, 1)

        self.btn_refresh = QPushButton("Refresh from GUI")
        self.btn_pick = QPushButton("Load from _datasets…")

        self.btn_refresh.setToolTip(
            "Use the dataset currently selected in the main toolbar "
            "(Open dataset…)."
        )
        self.btn_pick.setToolTip(
            "Pick a dataset for the current city from the "
            "<results_root>/_datasets folder."
        )

        header.addWidget(self.btn_refresh)
        header.addWidget(self.btn_pick)

        layout.addLayout(header)

        # --- Summary row --------------------------------------------------
        summary = QVBoxLayout()
        summary.setSpacing(2)

        self.lbl_shape = QLabel("Rows: –    Columns: –")
        self.lbl_year_range = QLabel("Year range: –")
        self.lbl_key_cols = QLabel("Key columns: –")

        for lab in (self.lbl_shape, self.lbl_year_range, self.lbl_key_cols):
            lab.setStyleSheet("color: #444444;")
            summary.addWidget(lab)

        layout.addLayout(summary)

        # --- Missing-values table ----------------------------------------
        self.tbl_missing = QTableWidget()
        self.tbl_missing.setColumnCount(3)
        self.tbl_missing.setHorizontalHeaderLabels(
            ["Column", "Missing", "Missing %"]
        )
        self.tbl_missing.horizontalHeader().setStretchLastSection(True)
        self.tbl_missing.verticalHeader().setVisible(False)
        self.tbl_missing.setAlternatingRowColors(True)
        layout.addWidget(self.tbl_missing, 2)

        # --- Column preview panel -------------------------------------
        self.preview_label = QLabel(
            "Column preview (select a row above)."
        )
        self.preview_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.preview_label)

        self.preview_edit = QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setMinimumHeight(120)
        layout.addWidget(self.preview_edit, 1)

        layout.addStretch(1)

        # --- connections ----------------------------------------------
        self.btn_refresh.clicked.connect(self._refresh_from_app)
        self.btn_pick.clicked.connect(self._choose_from_datasets)
        self.tbl_missing.itemSelectionChanged.connect(
            self._update_column_preview
        )

        # slight accent for header buttons
        self.btn_refresh.setStyleSheet(
            f"QPushButton:hover {{ background-color: {SECONDARY_TBLUE}; }}"
        )
        self.btn_pick.setStyleSheet(
            f"QPushButton:hover {{ background-color: {SECONDARY_TBLUE}; }}"
        )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _resolve_from_app(self) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
        """
        Try to obtain the active dataset from the main GUI.

        Priority:
        1. ``_edited_df`` (in-memory, after Open dataset…).
        2. ``csv_path`` (on-disk CSV).
        """
        ctx = self._app_ctx
        if ctx is None:
            return None, None

        # 1) In-memory edited DataFrame, if any
        edited = getattr(ctx, "_edited_df", None)
        csv_path = getattr(ctx, "csv_path", None)

        if isinstance(edited, pd.DataFrame) and not edited.empty:
            return edited.copy(), Path(csv_path) if csv_path else None

        # 2) On-disk CSV path
        if csv_path is not None:
            csv_path = Path(csv_path)
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    return df, csv_path
                except Exception:
                    # Let caller handle the error if needed
                    return None, csv_path

        return None, None

    def _refresh_from_app(self) -> None:
        """Refresh dataset view using the current main GUI state."""
        df, path = self._resolve_from_app()
        if df is None:
            self._set_empty(
                "No active dataset found. Use “Open dataset…” first, "
                "or load from _datasets."
            )
            return

        self._set_dataframe(df, path)

    def _choose_from_datasets(self) -> None:
        """
        Let the user pick a dataset from <results_root>/_datasets
        for the current city.
        """
        ctx = self._app_ctx
        if ctx is None:
            QMessageBox.information(
                self,
                "No context",
                "This tool needs the main GeoPrior window.",
            )
            return

        city = ""
        if hasattr(ctx, "city_edit"):
            city = ctx.city_edit.text().strip()

        if not city:
            QMessageBox.information(
                self,
                "City required",
                "Please enter a city/dataset name in the main toolbar "
                "before picking from _datasets.",
            )
            return

        results_root = getattr(ctx, "gui_runs_root", None) or getattr(
            ctx, "results_root", None
        )
        if not results_root:
            QMessageBox.warning(
                self,
                "Results root not set",
                "Results root is not configured yet.",
            )
            return

        csv_path_str = choose_dataset_for_city(
            parent=self,
            city=city,
            results_root=Path(results_root),
        )
        if not csv_path_str:
            # User cancelled or nothing found
            return

        csv_path = Path(csv_path_str)
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Dataset error",
                f"Could not read dataset:\n{csv_path}\n\n{exc}",
            )
            return

        self._set_dataframe(df, csv_path)

    # ------------------------------------------------------------------
    # View update
    # ------------------------------------------------------------------
    def _set_empty(self, message: str) -> None:
        self._df = None
        self._csv_path = None

        self.lbl_status.setText(message)
        self.lbl_shape.setText("Rows: –    Columns: –")
        self.lbl_year_range.setText("Year range: –")
        self.lbl_key_cols.setText("Key columns: –")

        self.tbl_missing.clearContents()
        self.tbl_missing.setRowCount(0)

    def _update_column_preview(self) -> None:
        """Update preview when a column row is selected."""
        if self._df is None or self._df.empty:
            self.preview_label.setText(
                "Column preview (no dataset loaded)."
            )
            self.preview_edit.clear()
            return

        selected = self.tbl_missing.selectedItems()
        if not selected:
            self.preview_label.setText(
                "Column preview (select a row above)."
            )
            self.preview_edit.clear()
            return

        # Take the row of the first selected cell
        row = selected[0].row()
        item = self.tbl_missing.item(row, 0)
        if item is None:
            return

        col_name = item.text()
        if col_name not in self._df.columns:
            self.preview_label.setText(
                f"Column preview (column '{col_name}' not in DataFrame)."
            )
            self.preview_edit.clear()
            return

        s = self._df[col_name]
        non_null = int(s.notna().sum())
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))

        lines = []
        lines.append(f"Column: {col_name}")
        lines.append(f"dtype: {s.dtype}")
        lines.append(
            f"Non-null: {non_null:,}   Missing: {missing:,}   "
            f"Unique: {unique:,}"
        )

        # Basic stats if numeric
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            lines.append("")
            lines.append("Numeric summary:")
            for key in ("mean", "std", "min", "25%", "50%", "75%", "max"):
                if key in desc:
                    lines.append(f"  {key}: {desc[key]:.4g}")

        # Head preview
        head = s.head(10)
        lines.append("")
        lines.append("Head(10):")
        for idx, val in head.items():
            lines.append(f"  {idx}: {repr(val)}")

        self.preview_label.setText(f"Column preview – {col_name}")
        self.preview_edit.setPlainText("\n".join(lines))

    def _set_dataframe(self, df: pd.DataFrame, path: Optional[Path]) -> None:
        self._df = df
        self._csv_path = path

        rows, cols = df.shape
        if path is not None:
            self.lbl_status.setText(f"Dataset: {path.name} ({rows:,} × {cols})")
        else:
            self.lbl_status.setText(f"In-memory dataset ({rows:,} × {cols})")

        self.lbl_shape.setText(f"Rows: {rows:,}    Columns: {cols}")

        # --- year range from GeoPriorConfig.time_col ------------------
        ctx = self._app_ctx
        time_col = None
        if ctx is not None and hasattr(ctx, "geo_cfg"):
            time_col = getattr(ctx.geo_cfg, "time_col", None)

        if time_col and time_col in df.columns:
            series = pd.to_numeric(df[time_col], errors="coerce").dropna()
            if len(series):
                y_min, y_max = int(series.min()), int(series.max())
                self.lbl_year_range.setText(
                    f"Year range ({time_col}): {y_min} – {y_max}"
                )
            else:
                self.lbl_year_range.setText(
                    f"Year range ({time_col}): no valid values."
                )
        elif "year" in df.columns:
            series = pd.to_numeric(df["year"], errors="coerce").dropna()
            if len(series):
                y_min, y_max = int(series.min()), int(series.max())
                self.lbl_year_range.setText(f"Year range (year): {y_min} – {y_max}")
            else:
                self.lbl_year_range.setText("Year range: no valid values.")
        else:
            self.lbl_year_range.setText("Year range: n/a (no year column).")

        # --- key columns presence summary -----------------------------
        key_desc = []
        if ctx is not None and hasattr(ctx, "geo_cfg"):
            cfg = ctx.geo_cfg
            for attr, label in [
                ("time_col", "Time"),
                ("lon_col", "Lon"),
                ("lat_col", "Lat"),
                ("subs_col", "Subsidence"),
                ("gwl_col", "GWL"),
                ("h_field_col", "H field"),
            ]:
                col_name = getattr(cfg, attr, None)
                if col_name and col_name in df.columns:
                    key_desc.append(f"{label}: “{col_name}”")

        if key_desc:
            self.lbl_key_cols.setText(
                "Key columns (found): " + ", ".join(key_desc)
            )
        else:
            self.lbl_key_cols.setText(
                "Key columns: none of the configured columns were found."
            )

        # --- missing-values table -------------------------------------
        if rows == 0:
            self.tbl_missing.clearContents()
            self.tbl_missing.setRowCount(0)
            return

        missing_counts = df.isna().sum()
        self.tbl_missing.setRowCount(len(missing_counts))

        for i, (col, miss) in enumerate(missing_counts.items()):
            item_col = QTableWidgetItem(str(col))
            item_miss = QTableWidgetItem(f"{int(miss):,}")
            pct = float(miss) / float(rows) * 100.0
            item_pct = QTableWidgetItem(f"{pct:.1f} %")

            item_miss.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item_pct.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            # light highlight if > 50% missing
            if pct >= 50.0:
                for it in (item_col, item_miss, item_pct):
                    it.setBackground(Qt.yellow)

            self.tbl_missing.setItem(i, 0, item_col)
            self.tbl_missing.setItem(i, 1, item_miss)
            self.tbl_missing.setItem(i, 2, item_pct)

        self.tbl_missing.resizeColumnsToContents()
