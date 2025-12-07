# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Config inspector & diff tool for the GeoPrior Tools tab.
#
# This panel lets the user compare the **current** GUI configuration
# (GeoPriorConfig.as_dict()) with a configuration loaded from disk
# (for example a manifest/config JSON saved by a past training / tuning
# run). It highlights which keys changed and provides a tiny visual
# diff for numeric settings.

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Dict
import json
import numbers

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
    QSplitter,
    QGroupBox,
    QMessageBox,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ...styles import SECONDARY_TBLUE

 
class ConfigDiffTool(QWidget):
    """
    Inspect / diff GeoPrior configuration.

    Left:
        - Buttons to reload the current GUI config and to load a config
          JSON from disk.
        - Summary line with which file is loaded.

    Centre:
        - Table with one row per key, showing current GUI value vs.
          loaded config value and a status (same / changed / only in GUI
          / only in file).

    Bottom:
        - Small plot that, for numeric values, shows a pair of bars
          (current vs. loaded). For non-numeric values, a textual
          explanation is shown.
    """

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._app_ctx = app_ctx
        self._cfg = getattr(app_ctx, "geo_cfg", None) if app_ctx else None

        self._current_cfg: Dict[str, Any] = {}
        self._loaded_cfg: Dict[str, Any] = {}
        self._loaded_path: Optional[Path] = None

        # Cache for table → underlying values
        self._row_key_map: Dict[int, str] = {}

        self._fig: Optional[Figure] = None
        self._canvas: Optional[FigureCanvas] = None

        self._init_ui()
        self._reload_from_gui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header: title + buttons + summary line
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        title_lbl = QLabel("<b>Config inspector & diff</b>", self)

        self._summary_lbl = QLabel(
            "No config file loaded yet. Use “Load config from JSON…”.",
            self,
        )
        self._summary_lbl.setWordWrap(True)

        btn_reload = QPushButton("Reload from GUI", self)
        btn_reload.clicked.connect(self._reload_from_gui)

        btn_load = QPushButton("Load config from JSON…", self)
        btn_load.clicked.connect(self._load_config_from_file)

        for btn in (btn_reload, btn_load):
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    padding: 4px 10px;
                    border-radius: 4px;
                    background-color: {SECONDARY_TBLUE};
                    color: white;
                }}
                QPushButton:hover {{
                    opacity: 0.9;
                }}
                """
            )

        header_row.addWidget(title_lbl)
        header_row.addSpacing(12)
        header_row.addWidget(self._summary_lbl, stretch=1)
        header_row.addWidget(btn_reload)
        header_row.addWidget(btn_load)

        # Splitter: diff table (top) + visual diff (bottom)
        splitter = QSplitter(Qt.Vertical, self)

        # --- Diff table ------------------------------------------------
        table_group = QGroupBox("Key-level differences", self)
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(6, 6, 6, 6)
        table_layout.setSpacing(4)

        self._table = QTableWidget(self)
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["Key", "Current (GUI)", "Loaded config", "Status"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)
        self._table.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        info_lbl = QLabel(
            "The table shows the union of keys from the current GUI "
            "config and the loaded JSON file.",
            table_group,
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: palette(mid); font-size: 9pt;")

        table_layout.addWidget(info_lbl)
        table_layout.addWidget(self._table, stretch=1)

        splitter.addWidget(table_group)

        # --- Visual diff panel ----------------------------------------
        plot_group = QGroupBox("Visual diff (selected key)", self)
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.setSpacing(4)

        self._plot_lbl = QLabel(
            "Select a row above to see a small visual comparison.",
            plot_group,
        )
        self._plot_lbl.setWordWrap(True)

        self._fig = Figure(figsize=(4, 2))
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        plot_layout.addWidget(self._plot_lbl)
        plot_layout.addWidget(self._canvas, stretch=1)

        splitter.addWidget(plot_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addLayout(header_row)
        layout.addWidget(splitter, stretch=1)

        # Selection → update visual diff
        self._table.itemSelectionChanged.connect(
            self._on_table_selection_changed
        )

    # ------------------------------------------------------------------
    # Config loading / diffing
    # ------------------------------------------------------------------
    def _reload_from_gui(self) -> None:
        """Rebuild the current config dict from GeoPriorConfig."""
        if self._cfg is None or not hasattr(self._cfg, "as_dict"):
            self._current_cfg = {}
        else:
            try:
                self._current_cfg = dict(self._cfg.as_dict())
            except Exception:
                self._current_cfg = {}

        if not self._current_cfg:
            self._summary_lbl.setText(
                "No GeoPriorConfig found on the GUI. "
                "Config diff will only show values from the loaded JSON."
            )
        else:
            self._summary_lbl.setText(
                "Current GUI config loaded. "
                "Pick a JSON file to compare against."
            )

        self._rebuild_table()

    def _load_config_from_file(self) -> None:
        """Open a JSON config/manifest file and extract the config dict."""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open config or manifest JSON",
            "",
            "JSON files (*.json);;All files (*.*)",
        )
        if not path_str:
            return

        p = Path(path_str)
        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Config error",
                f"Could not parse JSON file:\n{p}\n\n{exc}",
            )
            return

        # If the file has a top-level "config" key, use that; otherwise
        # assume the whole dict is the config.
        if isinstance(payload, dict) and "config" in payload:
            cfg = payload.get("config") or {}
        else:
            cfg = payload

        if not isinstance(cfg, dict):
            QMessageBox.critical(
                self,
                "Config error",
                "JSON file does not seem to contain a configuration "
                "mapping at top level or under 'config'.",
            )
            return

        self._loaded_cfg = cfg
        self._loaded_path = p

        src = f"{p.name}"
        if isinstance(payload, dict) and "city" in payload:
            src = f"{payload.get('city')} – {p.name}"

        if self._current_cfg:
            self._summary_lbl.setText(
                f"Comparing GUI config with: {src}"
            )
        else:
            self._summary_lbl.setText(
                f"Showing keys from loaded config file: {src}"
            )

        self._rebuild_table()

    def _rebuild_table(self) -> None:
        """Repopulate the diff table from _current_cfg and _loaded_cfg."""
        self._table.setRowCount(0)
        self._row_key_map.clear()

        all_keys = sorted(
            set(self._current_cfg.keys()) | set(self._loaded_cfg.keys())
        )

        self._table.setRowCount(len(all_keys))

        for row, key in enumerate(all_keys):
            cur = self._current_cfg.get(key, "<missing>")
            old = self._loaded_cfg.get(key, "<missing>")

            if key in self._current_cfg and key in self._loaded_cfg:
                status = (
                    "same"
                    if self._values_equal(cur, old)
                    else "changed"
                )
            elif key in self._current_cfg:
                status = "only in GUI"
            else:
                status = "only in file"

            cur_txt = self._format_value(cur)
            old_txt = self._format_value(old)

            items = [
                QTableWidgetItem(str(key)),
                QTableWidgetItem(cur_txt),
                QTableWidgetItem(old_txt),
                QTableWidgetItem(status),
            ]

            # Colour-code status
            if status == "changed":
                for it in items:
                    it.setBackground(Qt.yellow)
            elif status.startswith("only in"):
                for it in items:
                    it.setBackground(Qt.lightGray)

            items[1].setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            items[2].setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            items[3].setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)

            for col_ix, it in enumerate(items):
                self._table.setItem(row, col_ix, it)

            self._row_key_map[row] = key

        self._table.resizeColumnsToContents()
        self._clear_plot(
            "Select a row above to see a small visual comparison."
        )

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        """Looser equality that understands floats vs ints etc."""
        if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            return float(a) == float(b)
        return a == b

    @staticmethod
    def _format_value(v: Any, max_len: int = 80) -> str:
        if isinstance(v, str):
            txt = v
        elif isinstance(v, numbers.Number):
            txt = f"{v:.6g}"
        else:
            txt = repr(v)

        if len(txt) > max_len:
            return txt[: max_len - 3] + "..."
        return txt

    # ------------------------------------------------------------------
    # Visual diff
    # ------------------------------------------------------------------
    def _on_table_selection_changed(self) -> None:
        if not self._row_key_map:
            self._clear_plot("No keys to show yet.")
            return

        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._clear_plot(
                "Select a row above to see a visual comparison."
            )
            return

        row = rows[0].row()
        key = self._row_key_map.get(row)
        if key is None:
            self._clear_plot("Internal error: unknown row.")
            return

        cur = self._current_cfg.get(key, None)
        old = self._loaded_cfg.get(key, None)
        self._update_visual_for_key(key, cur, old)

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
        self._plot_lbl.setText(message)
        self._canvas.draw_idle()

    def _update_visual_for_key(
        self,
        key: str,
        cur: Any,
        old: Any,
    ) -> None:
        if self._fig is None or self._canvas is None:
            return

        self._fig.clear()
        ax = self._fig.add_subplot(111)

        # Only show bars for numeric/bool values present on both sides.
        if isinstance(cur, numbers.Number) and isinstance(old, numbers.Number):
            vals = [float(old), float(cur)]
            labels = ["Loaded", "GUI"]

            ax.bar([0, 1], vals)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(labels)
            ax.set_ylabel("Value")
            ax.set_title(key)

            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.4g}", ha="center", va="bottom")

            msg = (
                f"Numeric comparison for {key!r}: "
                f"loaded={vals[0]:.6g}, GUI={vals[1]:.6g}."
            )
            self._plot_lbl.setText(msg)
        else:
            # Non-numeric: just show text explaining values
            ax.set_axis_off()
            cur_txt = self._format_value(cur)
            old_txt = self._format_value(old)

            msg = (
                f"Non-numeric comparison for {key!r}.\n\n"
                f"Loaded: {old_txt}\n"
                f"GUI:    {cur_txt}"
            )
            ax.text(
                0.01,
                0.9,
                msg,
                ha="left",
                va="top",
                transform=ax.transAxes,
                wrap=True,
            )
            self._plot_lbl.setText(
                f"Textual comparison for {key!r} (non-numeric)."
            )

        self._canvas.draw_idle()
