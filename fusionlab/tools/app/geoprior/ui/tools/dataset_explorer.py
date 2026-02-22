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
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHeaderView,
    QLineEdit,
    QSpinBox,
    QSplitter,
    QStyle,
    QToolButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar, 
    QSizePolicy, 
    QGridLayout

)


from ...styles import PALETTE
from ...dialogs import choose_dataset_for_city  

class _NumItem(QTableWidgetItem):
    """
    Table item that sorts using a numeric payload.

    Notes
    -----
    - The numeric value is stored in Qt.UserRole.
    - Text is for display only.
    """

    def __init__(self, text: str, value: float) -> None:
        super().__init__(text)
        self.setData(Qt.UserRole, float(value))

    def __lt__(self, other: "QTableWidgetItem") -> bool:
        try:
            a = float(self.data(Qt.UserRole))
            b = float(other.data(Qt.UserRole))
            return a < b
        except Exception:
            return super().__lt__(other)

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

    def _make_pct_bar(self, pct: float) -> QProgressBar:
        bar = QProgressBar()
        bar.setObjectName("dsxPctBar")

        # Use 0..1000 for 0.1% precision.
        bar.setRange(0, 1000)
        bar.setValue(int(round(pct * 10.0)))

        # Tiny + clean (value in tooltip).
        bar.setTextVisible(False)
        bar.setToolTip(f"Missing: {pct:.1f} %")

        bar.setFixedHeight(14)
        bar.setMinimumWidth(96)

        return bar

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """
        Used by ToolPageFrame refresh button.
        """
        self._refresh_from_app()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        top = QFrame(self)
        top.setObjectName("dsxTop")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(8, 8, 8, 8)
        tl.setSpacing(8)

        self.lbl_status = QLabel("No dataset loaded.", top)
        self.lbl_status.setObjectName("dsxStatusChip")
        self.lbl_status.setTextFormat(Qt.RichText)
        self.lbl_status.setMinimumWidth(0)
        self.lbl_status.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        tl.addWidget(self.lbl_status, 1)

        self.btn_use_active = QToolButton(top)
        self.btn_use_active.setObjectName("miniAction")
        self.btn_use_active.setToolTip(
            "Use dataset currently selected in "
            "the main GUI (Open dataset…)."
        )
        self.btn_use_active.setIcon(
            self.style().standardIcon(
                QStyle.SP_BrowserReload
            )
        )
        self.btn_use_active.setText("Use active")
        self.btn_use_active.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )

        self.btn_pick = QToolButton(top)
        self.btn_pick.setObjectName("miniAction")
        self.btn_pick.setToolTip(
            "Pick a saved dataset for the current city "
            "from <results_root>/_datasets."
        )
        self.btn_pick.setIcon(
            self.style().standardIcon(
                QStyle.SP_DialogOpenButton
            )
        )
        self.btn_pick.setText("Browse saved")
        self.btn_pick.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )

        tl.addWidget(self.btn_use_active)
        tl.addWidget(self.btn_pick)
        layout.addWidget(top)

        summary = QFrame(self)
        summary.setObjectName("dsxSummary")
        sl = QGridLayout(summary)
        sl.setContentsMargins(8, 8, 8, 8)
        sl.setHorizontalSpacing(8)
        sl.setVerticalSpacing(6)
        
        self.lbl_shape = QLabel("Rows: –   Cols: –", summary)
        self.lbl_shape.setObjectName("dsxChip")
        self.lbl_shape.setTextFormat(Qt.RichText)
        
        self.lbl_year_range = QLabel("Year: –", summary)
        self.lbl_year_range.setObjectName("dsxChip")
        self.lbl_year_range.setTextFormat(Qt.RichText)
        
        self.lbl_key_cols = QLabel("Key cols: –", summary)
        self.lbl_key_cols.setObjectName("dsxChip")
        self.lbl_key_cols.setTextFormat(Qt.RichText)
        self.lbl_key_cols.setWordWrap(True)
        self.lbl_key_cols.setMinimumWidth(0)
        self.lbl_key_cols.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )

        sl.addWidget(self.lbl_shape, 0, 0)
        sl.addWidget(self.lbl_year_range, 0, 1)
        sl.addWidget(self.lbl_key_cols, 1, 0, 1, 2)
        layout.addWidget(summary)

        filt = QFrame(self)
        filt.setObjectName("dsxFilterRow")
        fl = QHBoxLayout(filt)
        fl.setContentsMargins(8, 0, 8, 0)
        fl.setSpacing(8)

        self.ed_filter = QLineEdit(filt)
        self.ed_filter.setObjectName("dsxFilter")
        self.ed_filter.setPlaceholderText(
            "Filter columns… (e.g., gwl)"
        )

        self.chk_missing = QCheckBox("Missing only", filt)
        self.chk_missing.setChecked(False)

        self.sp_min_pct = QSpinBox(filt)
        self.sp_min_pct.setRange(0, 100)
        self.sp_min_pct.setValue(0)
        self.sp_min_pct.setSuffix(" %")
        self.sp_min_pct.setToolTip(
            "Hide columns below this missing %."
        )

        fl.addWidget(self.ed_filter, 1)
        fl.addWidget(self.chk_missing)
        fl.addWidget(self.sp_min_pct)
        layout.addWidget(filt)

        split = QSplitter(Qt.Vertical, self)
        split.setObjectName("dsxSplit")
        layout.addWidget(split, 1)

        self.tbl_missing = QTableWidget()
        self.tbl_missing.setObjectName("dsxMissingTable")
        self.tbl_missing.setColumnCount(3)
        self.tbl_missing.setHorizontalHeaderLabels(
            ["Column", "Missing", "Missing %"]
        )
        self.tbl_missing.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.tbl_missing.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self.tbl_missing.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.tbl_missing.setAlternatingRowColors(True)
        self.tbl_missing.setSortingEnabled(True)
        self.tbl_missing.verticalHeader().setVisible(False)

        hh = self.tbl_missing.horizontalHeader()
        hh.setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )
        hh.setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )
        hh.setSectionResizeMode(
            2,
            QHeaderView.ResizeToContents,
        )

        split.addWidget(self.tbl_missing)

        prev = QFrame(self)
        prev.setObjectName("dsxPreviewCard")
        pl = QVBoxLayout(prev)
        pl.setContentsMargins(8, 8, 8, 8)
        pl.setSpacing(6)

        self.preview_label = QLabel(
            "Column preview (select a row above).",
            prev,
        )
        self.preview_label.setObjectName("dsxPreviewTitle")
        pl.addWidget(self.preview_label)

        self.preview_edit = QPlainTextEdit(prev)
        self.preview_edit.setObjectName("dsxPreview")
        self.preview_edit.setReadOnly(True)
        pl.addWidget(self.preview_edit, 1)

        split.addWidget(prev)
        split.setSizes([520, 240])

        self.btn_use_active.clicked.connect(
            self._refresh_from_app
        )
        self.btn_pick.clicked.connect(
            self._choose_from_datasets
        )
        self.tbl_missing.itemSelectionChanged.connect(
            self._update_column_preview
        )
        self.ed_filter.textChanged.connect(
            self._apply_table_filter
        )
        self.chk_missing.toggled.connect(
            lambda _v: self._apply_table_filter()
        )
        self.sp_min_pct.valueChanged.connect(
            lambda _v: self._apply_table_filter()
        )
        
    def _apply_table_filter(self) -> None:
        key = (self.ed_filter.text() or "").strip().lower()
        only_miss = bool(self.chk_missing.isChecked())
        min_pct = int(self.sp_min_pct.value())

        for r in range(self.tbl_missing.rowCount()):
            it = self.tbl_missing.item(r, 0)
            it_pct = self.tbl_missing.item(r, 2)

            name = it.text().lower() if it else ""
            pct = 0.0
            if it_pct is not None:
                try:
                    pct = float(
                        it_pct.data(Qt.UserRole) or 0.0
                    )
                except Exception:
                    pct = 0.0

            ok = True
            if key and key not in name:
                ok = False
            if only_miss and pct <= 0.0:
                ok = False
            if pct < float(min_pct):
                ok = False

            self.tbl_missing.setRowHidden(r, not ok)

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
        
    def _span(self, txt: str, color: str) -> str:
        return (
            f'<span style="color:{color};'
            f'font-weight:700;">{txt}</span>'
        )
    
    def _muted(self, txt: str) -> str:
        return (
            f'<span style="color:{PALETTE["light_text_muted"]};'
            f'font-weight:600;">{txt}</span>'
        )

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
            name = path.name
            title = self._span(name, PALETTE["primary"])
            dims = self._span(f"{rows:,} × {cols}", PALETTE["secondary"])
            full = f"Dataset: {name} ({rows:,} × {cols})"
            self.lbl_status.setText(
                f"{self._muted('Dataset:')} {title} "
                f"({dims})"
            )
            self.lbl_status.setToolTip(full)
        else:
            dims = self._span(f"{rows:,} × {cols}", PALETTE["secondary"])
            full = f"In-memory dataset ({rows:,} × {cols})"
            self.lbl_status.setText(
                f"{self._muted('In-memory dataset')} ({dims})"
            )
            self.lbl_status.setToolTip(full)

        r = self._span(f"{rows:,}", PALETTE["primary"])
        c = self._span(str(cols), PALETTE["secondary"])
        self.lbl_shape.setText(
            f"{self._muted('Rows:')} {r}   "
            f"{self._muted('Columns:')} {c}"
        )

        # --- year range from GeoPriorConfig.time_col ------------------
        def _set_year_text(label: str) -> None:
            self.lbl_year_range.setText(label)
            self.lbl_year_range.setToolTip(
                self.lbl_year_range.text()
            )
        
        ctx = self._app_ctx
        time_col = None
        if ctx is not None and hasattr(ctx, "geo_cfg"):
            time_col = getattr(ctx.geo_cfg, "time_col", None)

        if time_col and time_col in df.columns:
            series = pd.to_numeric(df[time_col], errors="coerce").dropna()
            if len(series):
                y_min, y_max = int(series.min()), int(series.max())
                ymin = self._span(str(y_min), PALETTE["primary"])
                ymax = self._span(str(y_max), PALETTE["primary"])
                _set_year_text(
                    f"{self._muted('Year range:')} {ymin} – {ymax}"
                )
            else:
               # when invalid:
                warn = self._span("no valid values", PALETTE["secondary"])
                _set_year_text(
                    f"{self._muted('Year range:')} {warn}"
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
        tags = []
        full_parts = []
        
        if ctx is not None and hasattr(ctx, "geo_cfg"):
            cfg = ctx.geo_cfg
            for attr, label in [
                ("time_col", "Time"),
                ("lon_col", "Lon"),
                ("lat_col", "Lat"),
                ("subs_col", "Subs"),
                ("gwl_col", "GWL"),
                ("h_field_col", "H"),
            ]:
                col = getattr(cfg, attr, None)
                if col and col in df.columns:
                    tags.append(self._span(col, PALETTE["primary"]))
                    full_parts.append(f"{label}: {col}")
        
        if tags:
            short = "  •  ".join(tags)
            self.lbl_key_cols.setText(
                f"{self._muted('Key cols:')} {short}"
            )
            self.lbl_key_cols.setToolTip(
                "Key columns (found): " + ", ".join(full_parts)
            )
        else:
            msg = self._span("none found", PALETTE["secondary"])
            self.lbl_key_cols.setText(
                f"{self._muted('Key cols:')} {msg}"
            )
            self.lbl_key_cols.setToolTip(
                "No configured key columns were found."
            )

        # --- missing-values table -------------------------------------
        if rows == 0:
            self.tbl_missing.clearContents()
            self.tbl_missing.setRowCount(0)
            return

        missing_counts = df.isna().sum()
        self.tbl_missing.setRowCount(len(missing_counts))

        missing_counts = df.isna().sum()
        self.tbl_missing.setRowCount(len(missing_counts))

        for i, (col, miss) in enumerate(missing_counts.items()):
            pct = float(miss) / float(rows) * 100.0

            item_col = QTableWidgetItem(str(col))

            item_miss = _NumItem(
                text=f"{int(miss):,}",
                value=float(miss),
            )
            item_miss.setTextAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )

            # Keep an item for sorting/filtering...
            item_pct = _NumItem(
                text=f"{pct:.1f} %",
                value=float(pct),
            )
            item_pct.setTextAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )

            self.tbl_missing.setItem(i, 0, item_col)
            self.tbl_missing.setItem(i, 1, item_miss)
            self.tbl_missing.setItem(i, 2, item_pct)

            # ...but show a mini progress bar in the cell.
            bar = self._make_pct_bar(pct)
            self.tbl_missing.setCellWidget(i, 2, bar)

            # Optional: nicer row height for the bar.
            self.tbl_missing.setRowHeight(i, 26)

        self._apply_table_filter()

