# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Feature inspector tool.

Summarise all columns of the active dataset together with their roles
(from GeoPriorConfig) and basic statistics.

- One row per column.
- Shows inferred kind (numeric / categorical / datetime / bool).
- Shows roles: time index, targets, drivers, physics, optional features.
- Shows non-null count, missing %, and min / max for numeric columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional,  Any

import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
    QSplitter,
    QGroupBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ...styles import SECONDARY_TBLUE


class FeatureInspectorTool(QWidget):
    """
    Feature inspector panel for the Tools tab.

    It expects an application context ``app_ctx`` that exposes:

    - ``geo_cfg`` : GeoPriorConfig-like object
    - ``_edited_df`` or ``csv_path`` : active dataset

    The tool is read-only; edits still happen via CsvEditDialog /
    Dataset explorer.
    """

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._app_ctx = app_ctx
        self._cfg = getattr(app_ctx, "geo_cfg", None) if app_ctx else None
        self._df: Optional[pd.DataFrame] = None
        self._csv_path: Optional[Path] = None

        self._role_by_col: dict[str, list[str]] = {}

        self._fig: Figure | None = None
        self._canvas: FigureCanvas | None = None
        self._plot_label: QLabel | None = None

        self._init_ui()
        self._refresh_from_app()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header row: title + summary + refresh button
        header = QHBoxLayout()
        header.setSpacing(8)

        title_lbl = QLabel("<b>Feature inspector</b>", self)

        self._summary_lbl = QLabel(
            "No active dataset. Use “Open dataset…” on the main toolbar.",
            self,
        )
        self._summary_lbl.setWordWrap(True)

        btn_refresh = QPushButton("Refresh from GUI", self)
        btn_refresh.clicked.connect(self._refresh_from_app)
        # btn_refresh.setStyleSheet(
        #     f"""
        #     QPushButton {{
        #         padding: 4px 10px;
        #         border-radius: 4px;
        #         background-color: {SECONDARY_TBLUE};
        #         color: white;
        #     }}
        #     QPushButton:hover {{
        #         opacity: 0.9;
        #     }}
        #     """
        # )
        btn_refresh.setStyleSheet(
            f"""
            QToolButton {{
                padding: 4px 10px;
                border-radius: 4px;
                color: white;
            }}
            QToolButton:hover {{
                background-color: {SECONDARY_TBLUE};
                opacity: 0.9;
            }}
            """
        )
        header.addWidget(title_lbl)
        header.addSpacing(12)
        header.addWidget(self._summary_lbl, stretch=1)
        header.addWidget(btn_refresh)

        # ----------------- splitter: table (top) + plot (bottom) -----
        splitter = QSplitter(Qt.Vertical, self)

        # Table of features (top)
        self._table = QTableWidget(self)
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels(
            [
                "Feature",
                "Role(s)",
                "Kind",
                "dtype",
                "Non-null",
                "Missing %",
                "Min",
                "Max",
            ]
        )
        self._table.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)

        splitter.addWidget(self._table)

        # Plot / preview (bottom)
        plot_group = QGroupBox("Quick visual preview", self)
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.setSpacing(4)

        self._plot_label = QLabel(
            "Select a feature row above to see its distribution.",
            plot_group,
        )
        self._plot_label.setWordWrap(True)

        self._fig = Figure(figsize=(4, 2))
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        plot_layout.addWidget(self._plot_label)
        plot_layout.addWidget(self._canvas, stretch=1)

        splitter.addWidget(plot_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addLayout(header)
        layout.addWidget(splitter, stretch=1)

        # When selection changes → update plot
        self._table.itemSelectionChanged.connect(
            self._on_table_selection_changed
        )

    # ------------------------------------------------------------------
    # Data resolution helpers
    # ------------------------------------------------------------------
    def _resolve_dataframe(self) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
        """
        Try to obtain the active dataset from the main app.

        Priority:
        1) ``_edited_df`` attribute (what CsvEditDialog saved).
        2) ``csv_path`` on disk (re-read).
        """
        ctx = self._app_ctx
        if ctx is None:
            return None, None

        df = getattr(ctx, "_edited_df", None)
        csv_path = getattr(ctx, "csv_path", None)

        if isinstance(df, pd.DataFrame) and not df.empty:
            p = Path(csv_path) if csv_path else None
            return df.copy(), p

        if csv_path:
            p = Path(csv_path)
            if p.exists():
                try:
                    df2 = pd.read_csv(p)
                    return df2, p
                except Exception:
                    # If parsing fails, fall through
                    return None, p

        return None, None

    def _refresh_from_app(self) -> None:
        """
        Reload dataset + config from the application context.
        """
        df, path = self._resolve_dataframe()
        self._df = df
        self._csv_path = path

        if df is None or df.empty:
            msg = (
                "No active dataset. Use “Open dataset…” on the main "
                "toolbar, then come back here."
            )
            self._summary_lbl.setText(msg)
            self._populate_empty()
            return

        n_rows, n_cols = df.shape
        city = getattr(self._app_ctx.geo_cfg, "city", "") if (
            self._app_ctx and getattr(self._app_ctx, "geo_cfg", None)
        ) else ""
        city_txt = f"City: {city}" if city else "City: (not set)"

        src_txt = (
            f"Source: {self._csv_path.name}"
            if self._csv_path is not None
            else "Source: in-memory dataset"
        )

        self._summary_lbl.setText(
            f"{city_txt} — {src_txt} — shape: {n_rows:,} × {n_cols}"
        )

        self._populate_from_dataframe(df)

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------
    def _populate_empty(self) -> None:
        self._table.setRowCount(0)

    @staticmethod
    def _flatten_names(obj: Any) -> set[str]:
        """
        Flatten nested lists / tuples / strings into a set of names.
        """
        out: set[str] = set()

        def _walk(x: Any) -> None:
            if x is None:
                return
            if isinstance(x, (list, tuple, set)):
                for y in x:
                    _walk(y)
            else:
                out.add(str(x))

        _walk(obj)
        return out

    def _populate_from_dataframe(self, df: pd.DataFrame) -> None:
        cfg = getattr(self._app_ctx, "geo_cfg", None)

        # Pre-compute role sets from config (if available)
        if cfg is not None:
            time_col = getattr(cfg, "time_col", None)
            lon_col = getattr(cfg, "lon_col", None)
            lat_col = getattr(cfg, "lat_col", None)
            subs_col = getattr(cfg, "subs_col", None)
            gwl_col = getattr(cfg, "gwl_col", None)
            h_field_col = getattr(cfg, "h_field_col", None)

            dyn_drivers = set(getattr(cfg, "dynamic_driver_features", []))
            stat_drivers = set(getattr(cfg, "static_driver_features", []))
            fut_drivers = set(getattr(cfg, "future_driver_features", []))

            opt_num = self._flatten_names(
                getattr(cfg, "optional_numeric_features", [])
            )
            opt_cat = self._flatten_names(
                getattr(cfg, "optional_categorical_features", [])
            )
        else:
            time_col = lon_col = lat_col = subs_col = gwl_col = h_field_col = None
            dyn_drivers = stat_drivers = fut_drivers = set()
            opt_num = opt_cat = set()

        self._role_by_col.clear()
        self._table.setRowCount(len(df.columns))
        
        for row, col_name in enumerate(df.columns):
            s = df[col_name]
            n_total = len(s)
            non_null = int(s.notna().sum())
            missing = n_total - non_null
            missing_pct = (missing / n_total * 100.0) if n_total else 0.0

            # Kind / dtype
            kind = self._infer_kind(s)
            dtype_str = str(s.dtype)

            # Roles
            roles = []
            if col_name == time_col:
                roles.append("time index")
            if col_name == lon_col:
                roles.append("spatial: lon")
            if col_name == lat_col:
                roles.append("spatial: lat")
            if col_name == subs_col:
                roles.append("target: subsidence")
            if col_name == gwl_col:
                roles.append("target: GWL")
            if col_name == h_field_col:
                roles.append("physics: h_field")
            if col_name in dyn_drivers:
                roles.append("driver: dynamic")
            if col_name in stat_drivers:
                roles.append("driver: static")
            if col_name in fut_drivers:
                roles.append("driver: future")
            if col_name in opt_num:
                roles.append("optional numeric")
            if col_name in opt_cat:
                roles.append("optional categorical")
            if not roles:
                roles.append("other")

            roles_str = ", ".join(roles)
            
            # store roles for plotting logic
            self._role_by_col[str(col_name)] = roles
            
            # Numeric stats
            if is_numeric_dtype(s):
                if non_null > 0:
                    min_val = f"{s.min():.4g}"
                    max_val = f"{s.max():.4g}"
                else:
                    min_val = max_val = "—"
            else:
                min_val = max_val = "—"

            # Fill table
            items = [
                QTableWidgetItem(str(col_name)),
                QTableWidgetItem(roles_str),
                QTableWidgetItem(kind),
                QTableWidgetItem(dtype_str),
                QTableWidgetItem(f"{non_null:,}"),
                QTableWidgetItem(f"{missing_pct:5.1f}"),
                QTableWidgetItem(min_val),
                QTableWidgetItem(max_val),
            ]

            # Highlight targets / physics a bit
            if any(r.startswith("target") or r.startswith("physics") for r in roles):
                for it in items[:2]:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)

            for col_ix, it in enumerate(items):
                # Right-align numeric-ish columns
                if col_ix in (4, 5, 6, 7):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._table.setItem(row, col_ix, it)

        self._table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------
    def _on_table_selection_changed(self) -> None:
        if self._df is None or self._df.empty:
            self._clear_plot("No dataset loaded.")
            return

        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._clear_plot("Select a feature row above.")
            return

        row = rows[0].row()
        item = self._table.item(row, 0)
        if item is None:
            self._clear_plot("Select a feature row above.")
            return

        col_name = item.text()
        if col_name not in self._df.columns:
            self._clear_plot("Column not found in DataFrame.")
            return

        series = self._df[col_name]
        roles = self._role_by_col.get(col_name, [])
        self._update_plot_for_series(col_name, series, roles)

    def _clear_plot(self, message: str) -> None:
        if self._fig is None or self._canvas is None:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        if self._plot_label is not None:
            self._plot_label.setText(message)
        self._canvas.draw_idle()

    def _update_plot_for_series(
        self,
        col_name: str,
        s: pd.Series,
        roles: list[str],
    ) -> None:
        if self._fig is None or self._canvas is None:
            return

        cfg = getattr(self._app_ctx, "geo_cfg", None)
        time_col = getattr(cfg, "time_col", None) if cfg else None

        self._fig.clear()
        ax = self._fig.add_subplot(111)

        if is_numeric_dtype(s):
            # If dynamic driver and time column exists → mean over time
            is_dynamic = any(r.startswith("driver: dynamic") for r in roles)
            if (
                is_dynamic
                and time_col
                and time_col in self._df.columns  # type: ignore[union-attr]
            ):
                df = self._df  # type: ignore[assignment]
                grp = (
                    df[[time_col, col_name]]
                    .dropna()
                    .groupby(time_col)[col_name]
                    .mean()
                )
                if not grp.empty:
                    ax.plot(grp.index, grp.values, marker="o")
                    ax.set_xlabel(time_col)
                    ax.set_ylabel(col_name)
                    ax.set_title(f"{col_name} – mean over {time_col}")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No non-null data for this feature.",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
            else:
                clean = s.dropna()
                if not clean.empty:
                    ax.hist(clean.values, bins=30)
                    ax.set_xlabel(col_name)
                    ax.set_ylabel("Count")
                    ax.set_title(f"{col_name} – histogram")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No non-null data for this feature.",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()

        elif is_datetime64_any_dtype(s):
            clean = pd.to_datetime(s, errors="coerce").dropna()
            counts = clean.value_counts().sort_index()
            if not counts.empty:
                ax.plot(counts.index, counts.values, marker=".")
                ax.set_xlabel(col_name)
                ax.set_ylabel("Count")
                ax.set_title(f"{col_name} – datetime counts")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid datetime values.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
        else:
            # Categorical / text: top-15 value counts
            counts = s.astype("string").value_counts().head(15)
            if not counts.empty:
                ax.bar(range(len(counts)), counts.values)
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels(
                    list(counts.index),
                    rotation=45,
                    ha="right",
                )
                ax.set_ylabel("Count")
                ax.set_title(f"{col_name} – top categories")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No non-null values.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()

        if self._plot_label is not None:
            self._plot_label.setText(f"Visual preview for: {col_name}")
        self._canvas.draw_idle()

    @staticmethod
    def _infer_kind(s: pd.Series) -> str:
        """Rough semantic kind for a column."""
        if is_bool_dtype(s):
            return "bool"
        if is_numeric_dtype(s):
            return "numeric"
        if is_datetime64_any_dtype(s):
            return "datetime"
        if s.dtype.name == "category":
            return "categorical"
        if s.dtype == object:
            return "text"
        return str(s.dtype)
