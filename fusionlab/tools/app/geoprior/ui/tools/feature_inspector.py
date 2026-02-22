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
    QGroupBox,
    QAbstractItemView,
    QFormLayout,
    QFrame,
    QHeaderView,
    QLineEdit,
    QProgressBar,
    QToolButton,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ...styles import SECONDARY_TBLUE

class _NumItem(QTableWidgetItem):
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

        header = QHBoxLayout()
        header.setSpacing(8)

        title_lbl = QLabel("<b>Feature inspector</b>", self)

        self._summary_lbl = QLabel(
            "No active dataset. Use “Open dataset…” on the main "
            "toolbar.",
            self,
        )
        self._summary_lbl.setWordWrap(True)

        self._btn_refresh = QToolButton(self)
        self._btn_refresh.setObjectName("miniAction")
        self._btn_refresh.setText("Refresh from GUI")
        self._btn_refresh.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self._btn_refresh.clicked.connect(self._refresh_from_app)

        header.addWidget(title_lbl)
        header.addSpacing(12)
        header.addWidget(self._summary_lbl, 1)
        header.addWidget(self._btn_refresh)

        splitter = QSplitter(Qt.Vertical, self)

        top_split = QSplitter(Qt.Horizontal, self)

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
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )

        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )
        hh.setSectionResizeMode(
            1,
            QHeaderView.Stretch,
        )
        hh.setSectionResizeMode(
            2,
            QHeaderView.ResizeToContents,
        )
        hh.setSectionResizeMode(
            3,
            QHeaderView.ResizeToContents,
        )
        hh.setSectionResizeMode(
            4,
            QHeaderView.ResizeToContents,
        )
        hh.setSectionResizeMode(
            5,
            QHeaderView.ResizeToContents,
        )
        hh.setSectionResizeMode(
            6,
            QHeaderView.ResizeToContents,
        )
        hh.setSectionResizeMode(
            7,
            QHeaderView.ResizeToContents,
        )

        self._fx_card = self._build_feature_card()

        top_split.addWidget(self._table)
        top_split.addWidget(self._fx_card)
        top_split.setSizes([760, 280])

        splitter.addWidget(top_split)

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

        plot_layout.addWidget(self._plot_label)
        plot_layout.addWidget(self._canvas, 1)

        splitter.addWidget(plot_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addLayout(header)
        layout.addWidget(splitter, 1)

        self._table.itemSelectionChanged.connect(
            self._on_table_selection_changed
        )
        
    def _make_pct_bar(self, pct: float) -> QProgressBar:
        bar = QProgressBar(self)
        bar.setObjectName("fxPctBar")
        bar.setRange(0, 1000)
        bar.setValue(int(round(pct * 10.0)))
        bar.setTextVisible(False)
        bar.setFixedHeight(14)
        bar.setMinimumWidth(140)
        bar.setToolTip(f"Missing: {pct:.1f} %")
        return bar

    def _build_feature_card(self) -> QFrame:
        card = QFrame(self)
        card.setObjectName("fxCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self._fx_title = QLabel("No selection", card)
        self._fx_title.setObjectName("fxTitle")

        self._fx_meta = QLabel("Select a row in the table.", card)
        self._fx_meta.setWordWrap(True)

        self._fx_roles = QLabel("", card)
        self._fx_roles.setWordWrap(True)
        self._fx_roles.setObjectName("fxRoles")

        self._fx_missing_lbl = QLabel("Missingness", card)
        self._fx_missing = self._make_pct_bar(0.0)

        self._fx_counts = QLabel("", card)
        self._fx_counts.setWordWrap(True)

        self._fx_range = QLabel("", card)
        self._fx_range.setWordWrap(True)

        self._fx_corr = QLabel("", card)
        self._fx_corr.setWordWrap(True)

        self._fx_note = QLabel("", card)
        self._fx_note.setWordWrap(True)
        self._fx_note.setObjectName("fxNote")

        lay.addWidget(self._fx_title)
        lay.addWidget(self._fx_meta)
        lay.addWidget(self._fx_roles)
        lay.addWidget(self._fx_missing_lbl)
        lay.addWidget(self._fx_missing)
        lay.addWidget(self._fx_counts)
        lay.addWidget(self._fx_range)
        lay.addWidget(self._fx_corr)
        lay.addWidget(self._fx_note)
        lay.addStretch(1)
        
        self._fx_top_lbl = QLabel("Top values", card)
        self._fx_top_lbl.setObjectName("fxMiniTitle")

        self._fx_top_tbl = QTableWidget(card)
        self._fx_top_tbl.setObjectName("fxMiniTable")
        self._fx_top_tbl.setColumnCount(2)
        self._fx_top_tbl.setHorizontalHeaderLabels(
            ["Value", "Count"]
        )
        self._fx_top_tbl.verticalHeader().setVisible(False)
        self._fx_top_tbl.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self._fx_top_tbl.setSelectionMode(
            QAbstractItemView.NoSelection
        )
        self._fx_top_tbl.setSortingEnabled(False)
        self._fx_top_tbl.setAlternatingRowColors(True)
        self._fx_top_tbl.setMaximumHeight(180)

        hh = self._fx_top_tbl.horizontalHeader()
        hh.setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )
        hh.setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )

        lay.addWidget(self._fx_top_lbl)
        lay.addWidget(self._fx_top_tbl)

        return card

    # ------------------------------------------------------------------
    # Data resolution helpers
    # ------------------------------------------------------------------
    def _mini_clear(self) -> None:
        self._fx_top_tbl.setRowCount(0)

    def _mini_set_rows(
        self,
        title: str,
        rows: list[tuple[str, str]],
    ) -> None:
        self._fx_top_lbl.setText(title)
        self._fx_top_tbl.setRowCount(len(rows))

        for r, (a, b) in enumerate(rows):
            it0 = QTableWidgetItem(a)
            it1 = QTableWidgetItem(b)
            it1.setTextAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )
            self._fx_top_tbl.setItem(r, 0, it0)
            self._fx_top_tbl.setItem(r, 1, it1)

        self._fx_top_tbl.setRowHeight(0, 22)
        for r in range(self._fx_top_tbl.rowCount()):
            self._fx_top_tbl.setRowHeight(r, 22)
            
    def _update_mini_stats(self, s: pd.Series) -> None:
        if s is None or s.empty:
            self._mini_set_rows("Stats", [])
            return

        clean = s.dropna()
        if clean.empty:
            self._mini_set_rows("Stats", [])
            return

        n = int(clean.shape[0])
        if n > 200_000:
            try:
                clean = clean.sample(
                    200_000,
                    random_state=0,
                )
            except Exception:
                clean = clean.iloc[:200_000]

        if is_bool_dtype(clean):
            vc = clean.value_counts().head(8)
            rows = []
            for k, v in vc.items():
                rows.append((str(k), f"{int(v):,}"))
            self._mini_set_rows("Top values", rows)
            return

        if is_datetime64_any_dtype(clean):
            dt = pd.to_datetime(clean, errors="coerce").dropna()
            if dt.empty:
                self._mini_set_rows("Datetime", [])
                return

            top = dt.value_counts().head(8)
            rows = [
                ("min", str(dt.min())),
                ("max", str(dt.max())),
            ]
            for k, v in top.items():
                rows.append((str(k), f"{int(v):,}"))
            self._mini_set_rows("Datetime", rows)
            return

        if is_numeric_dtype(clean):
            x = pd.to_numeric(clean, errors="coerce").dropna()
            if x.empty:
                self._mini_set_rows("Numeric stats", [])
                return

            d = x.describe()
            keys = ["mean", "std", "min", "25%"]
            keys += ["50%", "75%", "max"]

            rows: list[tuple[str, str]] = []
            for k in keys:
                if k in d.index:
                    rows.append((k, f"{float(d[k]):.4g}"))
            self._mini_set_rows("Numeric stats", rows)
            return

        vc = clean.astype("string").value_counts().head(8)
        rows = []
        for k, v in vc.items():
            txt = str(k)
            if len(txt) > 28:
                txt = txt[:25] + "…"
            rows.append((txt, f"{int(v):,}"))
        self._mini_set_rows("Top values", rows)

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
            # items = [
            #     QTableWidgetItem(str(col_name)),
            #     QTableWidgetItem(roles_str),
            #     QTableWidgetItem(kind),
            #     QTableWidgetItem(dtype_str),
            #     QTableWidgetItem(f"{non_null:,}"),
            #     QTableWidgetItem(f"{missing_pct:5.1f}"),
            #     QTableWidgetItem(min_val),
            #     QTableWidgetItem(max_val),
            # ]
            it_name = QTableWidgetItem(str(col_name))
            it_roles = QTableWidgetItem(roles_str)
            it_kind = QTableWidgetItem(kind)
            it_dtype = QTableWidgetItem(dtype_str)

            it_non = _NumItem(
                text=f"{non_null:,}",
                value=float(non_null),
            )
            it_non.setTextAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )

            it_miss = _NumItem(
                text=f"{missing_pct:5.1f}",
                value=float(missing_pct),
            )
            it_miss.setTextAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )

            it_min = QTableWidgetItem(min_val)
            it_max = QTableWidgetItem(max_val)

            for it in (it_min, it_max):
                it.setTextAlignment(
                    Qt.AlignRight | Qt.AlignVCenter
                )

            self._table.setItem(row, 0, it_name)
            self._table.setItem(row, 1, it_roles)
            self._table.setItem(row, 2, it_kind)
            self._table.setItem(row, 3, it_dtype)
            self._table.setItem(row, 4, it_non)
            self._table.setItem(row, 5, it_miss)
            self._table.setItem(row, 6, it_min)
            self._table.setItem(row, 7, it_max)

            bar = self._make_pct_bar(float(missing_pct))
            self._table.setCellWidget(row, 5, bar)
            self._table.setRowHeight(row, 26)

            # Highlight targets / physics a bit
            # if any(r.startswith("target") or r.startswith("physics") for r in roles):
            #     for it in items[:2]:
            #         f = it.font()
            #         f.setBold(True)
            #         it.setFont(f)

            # for col_ix, it in enumerate(items):
            #     # Right-align numeric-ish columns
            #     if col_ix in (4, 5, 6, 7):
            #         it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            #     self._table.setItem(row, col_ix, it)

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
        self._update_feature_card(col_name)
        self._update_plot_for_series(col_name, series, roles)
        
    def _update_feature_card(self, col_name: str) -> None:
        if self._df is None or self._df.empty:
            self._fx_title.setText("No dataset")
            return

        if col_name not in self._df.columns:
            self._fx_title.setText("Invalid selection")
            return

        s = self._df[col_name]
        roles = self._role_by_col.get(col_name, [])

        n = int(s.shape[0])
        non = int(s.notna().sum())
        miss = n - non
        pct = (float(miss) / float(n) * 100.0) if n else 0.0
        uniq = int(s.nunique(dropna=True))

        self._fx_title.setText(col_name)
        self._fx_meta.setText(
            f"Kind: {self._infer_kind(s)}  •  dtype: {s.dtype}"
        )
        self._fx_roles.setText("Roles: " + ", ".join(roles))

        self._fx_missing.setValue(int(round(pct * 10.0)))
        self._fx_missing.setToolTip(f"Missing: {pct:.1f} %")

        self._fx_counts.setText(
            f"Rows: {n:,}  •  Non-null: {non:,}  •  "
            f"Missing: {miss:,}  •  Unique: {uniq:,}"
        )

        if is_numeric_dtype(s) and non > 0:
            mn = float(pd.to_numeric(s, errors="coerce").min())
            mx = float(pd.to_numeric(s, errors="coerce").max())
            self._fx_range.setText(f"Range: {mn:.4g} → {mx:.4g}")
        else:
            self._fx_range.setText("Range: —")

        self._fx_corr.setText(self._corr_text(col_name, s))

        note = self._quality_note(pct, uniq, non)
        self._fx_note.setText(note)
        self._update_mini_stats(s)

    def _corr_text(self, col: str, s: pd.Series) -> str:
        if self._df is None:
            return ""

        if not is_numeric_dtype(s):
            return ""

        cfg = getattr(self._app_ctx, "geo_cfg", None)
        if cfg is None:
            return ""

        targets = []
        subs = getattr(cfg, "subs_col", None)
        gwl = getattr(cfg, "gwl_col", None)

        for t in (subs, gwl):
            if t and t in self._df.columns:
                targets.append(str(t))

        if not targets:
            return ""

        out = []
        for t in targets:
            try:
                a = pd.to_numeric(self._df[col], errors="coerce")
                b = pd.to_numeric(self._df[t], errors="coerce")
                cc = a.corr(b)
                if pd.notna(cc):
                    out.append(f"corr({t})={cc:.3f}")
            except Exception:
                continue

        if not out:
            return ""

        return "Target links: " + "  •  ".join(out)

    @staticmethod
    def _quality_note(pct: float, uniq: int, non: int) -> str:
        warns = []
        if pct >= 50.0:
            warns.append("High missingness (≥50%).")
        if non > 0 and uniq <= 1:
            warns.append("Constant feature (no variance).")
        if not warns:
            return ""
        return "Notes: " + " ".join(warns)

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
