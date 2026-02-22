# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Metrics dashboard tool for GeoPrior GUI.

This tool visualises evaluation metrics for GeoPriorSubsNet runs
(train / tune / inference) under the current ``results_root``.

For each run it tries to load:
- ``geoprior_eval_phys_*.json`` (core point + interval metrics);
- ``eval_diagnostics.json`` (per-year diagnostics);
- ``*_geopriorsubsnet_physical_parameters.csv`` (physics params).

Two Matplotlib figures are shown:

1. Core metrics:
   - Overall MAE / R² / coverage / PSS vs year
     (from ``eval_diagnostics.json`` when present).
   - Per-horizon MAE / R² from ``per_horizon`` in
     ``geoprior_eval_phys_*.json``.

2. PIT / reliability:
   - Placeholder by default. Once PIT histograms and reliability
     curves are saved per run (e.g. as JSON files), they can be
     parsed and plotted here via :func:`_load_pit_and_reliab`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QPlainTextEdit,
    QSplitter,
    QMessageBox,
    QFileDialog, 
    QDialog,
    QTabWidget,
    QLineEdit,
    QToolButton,
    QStyle,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

@dataclass
class RunEntry:
    city: str
    kind: str         # "train", "tuning", "inference"
    run_id: str       # e.g. "train_20251110-122128"
    path: Path

class _Chip(QLabel):
    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setObjectName("chip")
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setMinimumHeight(22)
        self.setStyleSheet(
            "padding:2px 10px;"
            "border-radius:11px;"
            "background: palette(midlight);"
            "color: palette(text);"
        )

# ----------------------------------------------------------------------
# Main tool
# ----------------------------------------------------------------------

class MetricsDashboardTool(QWidget):
    """
    Metrics dashboard for the Tools tab.

    Parameters
    ----------
    app_ctx : object, optional
        Reference to the main GeoPrior GUI. Used for:
        - ``gui_runs_root`` / ``results_root`` (results root directory);
        - ``city_edit`` (to pre-select city when present).
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

        self._results_root: Path = self._detect_results_root()
        self._runs: List[RunEntry] = []

        # remember last-selected run + metrics for pop-out windows
        self._last_entry: Optional[RunEntry] = None
        self._last_eval_diag: Optional[Dict[str, Any]] = None
        self._last_phys_eval: Optional[Dict[str, Any]] = None
        self._last_pit_data: Optional[Dict[str, Any]] = None
        self._last_reliab_data: Optional[Dict[str, Any]] = None
        
        self._init_ui()
        self._scan_and_populate()

    # ------------------------------------------------------------------
    # Results root detection
    # ------------------------------------------------------------------
    def _detect_results_root(self) -> Path:
        ctx = self._app_ctx
        if ctx is not None:
            # Prefer main ctx.results_root, then fallback to gui_runs_root
            root = getattr(ctx, "results_root", None) or getattr(
                ctx, "gui_runs_root", None
            )
            if root:
                return Path(root)

        # Fallback: ./results
        return Path(os.getcwd()) / "results"
    
    def _subs_scale_to_mm(
        self,
        phys_eval: Optional[Dict[str, Any]],
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """
        Return scale factor to convert phys metrics to mm.
    
        Returns
        -------
        scale : float
            Multiply phys subsidence metrics by this to get mm.
        dst_unit : str or None
            "mm" when we know we are in mm after conversion, else None.
        src_unit : str or None
            Original unit string from JSON, else None.
        """
        if not isinstance(phys_eval, dict):
            return 1.0, None, None
    
        units = phys_eval.get("units", {}) or {}
        src = str(units.get("subs_metrics_unit", "")).strip().lower()
        if not src:
            return 1.0, None, None
    
        if src == "mm":
            return 1.0, "mm", "mm"
    
        if src == "m":
            return 1.0e3, "mm", "m"
    
        # Unknown: do not force any unit label.
        return 1.0, None, src


    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        self._build_ui()
        self._connect_ui()
    
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
    
        # -------------------------------------------------
        # Header row: title + chips + actions
        # -------------------------------------------------
        hdr = QHBoxLayout()
        hdr.setSpacing(8)
    
        title = QLabel("Metrics dashboard", self)
        title.setStyleSheet("font-weight: 700; font-size: 12pt;")
    
        self._chip_group = _Chip("Diagnostics & Plots", self)
        self._chip_status = _Chip("Idle", self)
    
        hdr.addWidget(title)
        hdr.addWidget(self._chip_group)
        hdr.addStretch(1)
        hdr.addWidget(self._chip_status)
    
        self._btn_plot_tools = QToolButton(self)
        self._btn_plot_tools.setCheckable(True)
        self._btn_plot_tools.setAutoRaise(True)
        self._btn_plot_tools.setToolTip("Show / hide plot tools")
        self._btn_plot_tools.setIcon(
            self.style().standardIcon(QStyle.SP_TitleBarMenuButton)
        )
    
        self._btn_open_root = QToolButton(self)
        self._btn_open_root.setAutoRaise(True)
        self._btn_open_root.setToolTip("Open results root in explorer")
        self._btn_open_root.setIcon(
            self.style().standardIcon(QStyle.SP_DirOpenIcon)
        )
    
        hdr.addWidget(self._btn_plot_tools)
        hdr.addWidget(self._btn_open_root)
    
        root.addLayout(hdr)
    
        # -------------------------------------------------
        # Root bar: path + buttons
        # -------------------------------------------------
        top = QHBoxLayout()
        top.setSpacing(8)
    
        self.lbl_root = QLabel(f"Results root: {self._results_root}", self)
        self.lbl_root.setTextInteractionFlags(Qt.TextSelectableByMouse)
    
        self.btn_change_root = QPushButton("Change root…", self)
        self.btn_rescan = QPushButton("Rescan runs", self)
    
        top.addWidget(self.lbl_root, 1)
        top.addWidget(self.btn_change_root)
        top.addWidget(self.btn_rescan)
    
        root.addLayout(top)
    
        # -------------------------------------------------
        # Middle: runs + summary (splitter)
        # -------------------------------------------------
        mid = QSplitter(Qt.Horizontal, self)
        mid.setChildrenCollapsible(False)
    
        # ---- Runs panel
        runs_panel = QWidget(self)
        runs_lay = QVBoxLayout(runs_panel)
        runs_lay.setContentsMargins(0, 0, 0, 0)
        runs_lay.setSpacing(6)
    
        filt_row = QHBoxLayout()
        filt_row.setSpacing(6)
    
        self.edit_filter = QLineEdit(self)
        self.edit_filter.setPlaceholderText(
            "Filter city / kind / run id..."
        )
    
        self.btn_clear_filter = QToolButton(self)
        self.btn_clear_filter.setAutoRaise(True)
        self.btn_clear_filter.setToolTip("Clear filter")
        self.btn_clear_filter.setIcon(
            self.style().standardIcon(QStyle.SP_LineEditClearButton)
        )
    
        filt_row.addWidget(self.edit_filter, 1)
        filt_row.addWidget(self.btn_clear_filter)
    
        runs_lay.addLayout(filt_row)
    
        self.tree_runs = QTreeWidget(self)
        self.tree_runs.setHeaderLabels(["City", "Kind", "Run id"])
        self.tree_runs.setAlternatingRowColors(True)
        self.tree_runs.setRootIsDecorated(True)
        self.tree_runs.setUniformRowHeights(True)
    
        self.tree_runs.setColumnWidth(0, 140)
        self.tree_runs.setColumnWidth(1, 90)
    
        runs_lay.addWidget(self.tree_runs, 1)
    
        mid.addWidget(runs_panel)
    
        # ---- Summary panel
        summary_panel = QWidget(self)
        sum_lay = QVBoxLayout(summary_panel)
        sum_lay.setContentsMargins(0, 0, 0, 0)
        sum_lay.setSpacing(6)
    
        chips = QHBoxLayout()
        chips.setSpacing(6)
        self._chip_run = _Chip("No run", self)
        self._chip_units = _Chip("Units: —", self)
        chips.addWidget(self._chip_run)
        chips.addWidget(self._chip_units)
        chips.addStretch(1)
        sum_lay.addLayout(chips)
    
        self.summary_edit = QPlainTextEdit(self)
        self.summary_edit.setReadOnly(True)
        self.summary_edit.setMinimumWidth(300)
        sum_lay.addWidget(self.summary_edit, 1)
    
        mid.addWidget(summary_panel)
        mid.setStretchFactor(0, 3)
        mid.setStretchFactor(1, 2)
    
        root.addWidget(mid, 1)
    
        # -------------------------------------------------
        # Bottom: plots as tabs (cleaner + modern)
        # -------------------------------------------------
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
    
        # --- Core tab
        core_tab = QWidget(self)
        core_v = QVBoxLayout(core_tab)
        core_v.setContentsMargins(6, 6, 6, 6)
        core_v.setSpacing(6)
    
        core_actions = QHBoxLayout()
        core_actions.setSpacing(6)
    
        self.btn_pop_core = QPushButton("Pop out…", self)
        self.btn_pop_core.setMaximumWidth(110)
    
        core_actions.addStretch(1)
        core_actions.addWidget(self.btn_pop_core)
        core_v.addLayout(core_actions)
    
        self.fig_core = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas_core = FigureCanvas(self.fig_core)
        self.toolbar_core = NavigationToolbar(self.canvas_core, self)
        self.toolbar_core.setVisible(False)
        self.toolbar_core.setMovable(False)
        self.toolbar_core.setFloatable(False)
    
        core_v.addWidget(self.toolbar_core)
        core_v.addWidget(self.canvas_core, 1)
    
        # --- PIT tab
        pit_tab = QWidget(self)
        pit_v = QVBoxLayout(pit_tab)
        pit_v.setContentsMargins(6, 6, 6, 6)
        pit_v.setSpacing(6)
    
        pit_actions = QHBoxLayout()
        pit_actions.setSpacing(6)
    
        self.btn_pop_pit = QPushButton("Pop out…", self)
        self.btn_pop_pit.setMaximumWidth(110)
    
        pit_actions.addStretch(1)
        pit_actions.addWidget(self.btn_pop_pit)
        pit_v.addLayout(pit_actions)
    
        self.fig_pit = Figure(figsize=(5, 2.5), tight_layout=True)
        self.canvas_pit = FigureCanvas(self.fig_pit)
        self.toolbar_pit = NavigationToolbar(self.canvas_pit, self)
        self.toolbar_pit.setVisible(False)
        self.toolbar_pit.setMovable(False)
        self.toolbar_pit.setFloatable(False)
    
        pit_v.addWidget(self.toolbar_pit)
        pit_v.addWidget(self.canvas_pit, 1)
    
        self.tabs.addTab(core_tab, "Core metrics")
        self.tabs.addTab(pit_tab, "PIT / reliability")
        self.tabs.currentChanged.connect(self._sync_plot_toolbar)
    
        root.addWidget(self.tabs, 2)

    def _connect_ui(self) -> None:
        self.btn_change_root.clicked.connect(self._on_change_root)
        self.btn_rescan.clicked.connect(self._scan_and_populate)
    
        self.tree_runs.currentItemChanged.connect(self._on_run_selected)
    
        self.btn_pop_core.clicked.connect(self._show_core_popup)
        self.btn_pop_pit.clicked.connect(self._show_pit_popup)
    
        self._btn_plot_tools.toggled.connect(
            self._on_plot_tools_toggled
        )
        self._btn_open_root.clicked.connect(self._on_open_root)
    
        self.edit_filter.textChanged.connect(self._apply_filter)
        self.btn_clear_filter.clicked.connect(
            lambda: self.edit_filter.setText("")
        )
        
    def _sync_plot_toolbar(self, _i: int = 0) -> None:
        on = bool(self._btn_plot_tools.isChecked())
    
        self.toolbar_core.setVisible(
            on and self.tabs.currentIndex() == 0
        )
        self.toolbar_pit.setVisible(
            on and self.tabs.currentIndex() == 1
        )

    # ------------------------------------------------------------------
    # Run discovery
    # ------------------------------------------------------------------
    def _on_plot_tools_toggled(self, _on: bool) -> None:
        self._sync_plot_toolbar()

    def _on_open_root(self) -> None:
        p = str(self._results_root)
        QDesktopServices.openUrl(QUrl.fromLocalFile(p))
    
    def _apply_filter(self) -> None:
        q = self.edit_filter.text().strip().lower()
    
        for i in range(self.tree_runs.topLevelItemCount()):
            city_item = self.tree_runs.topLevelItem(i)
            any_child = False
    
            for j in range(city_item.childCount()):
                it = city_item.child(j)
                entry = it.data(0, Qt.UserRole)
    
                if entry is None:
                    it.setHidden(False)
                    any_child = True
                    continue
    
                text = (
                    f"{entry.city} {entry.kind} {entry.run_id}"
                ).lower()
    
                hit = (q in text) if q else True
                it.setHidden(not hit)
                if hit:
                    any_child = True
    
            city_item.setHidden(not any_child)

    def _scan_and_populate(self) -> None:
        """
        Scan ``results_root`` for runs and populate the QTreeWidget.
        """
        self._runs = []
        self.tree_runs.clear()

        root = self._results_root
        if not root.is_dir():
            self.summary_edit.setPlainText(
                f"[Info] Results root does not exist:\n{root}"
            )
            return

        # Per-city Stage-1 folders:
        #   <city>_GeoPriorSubsNet_stage1/
        for city_dir in sorted(root.glob("*_GeoPriorSubsNet_stage1")):
            if not city_dir.is_dir():
                continue

            city_name = city_dir.name.replace(
                "_GeoPriorSubsNet_stage1", ""
            )
            city_item = QTreeWidgetItem([city_name, "", ""])
            city_item.setFirstColumnSpanned(True)
            city_item.setFlags(Qt.ItemIsEnabled)
            self.tree_runs.addTopLevelItem(city_item)

            # TRAIN runs: train_YYYYMMDD-HHMMSS
            for run_dir in sorted(
                city_dir.glob("train_*"), key=lambda p: p.name
            ):
                if not run_dir.is_dir():
                    continue
                run_id = run_dir.name
                entry = RunEntry(
                    city=city_name,
                    kind="train",
                    run_id=run_id,
                    path=run_dir,
                )
                self._runs.append(entry)

                item = QTreeWidgetItem(
                    [city_name, "train", run_id]
                )
                item.setData(0, Qt.UserRole, entry)
                city_item.addChild(item)

            # TUNING runs: tuning/run_*
            tuning_root = city_dir / "tuning"
            if tuning_root.is_dir():
                for run_dir in sorted(
                    tuning_root.glob("run_*"), key=lambda p: p.name
                ):
                    if not run_dir.is_dir():
                        continue
                    run_id = run_dir.name
                    entry = RunEntry(
                        city=city_name,
                        kind="tuning",
                        run_id=run_id,
                        path=run_dir,
                    )
                    self._runs.append(entry)
                    item = QTreeWidgetItem(
                        [city_name, "tuning", run_id]
                    )
                    item.setData(0, Qt.UserRole, entry)
                    city_item.addChild(item)

            # INFERENCE runs: inference/<stamp>/
            inf_root = city_dir / "inference"
            if inf_root.is_dir():
                for run_dir in sorted(
                    inf_root.iterdir(), key=lambda p: p.name
                ):
                    if not run_dir.is_dir():
                        continue
                    run_id = run_dir.name
                    entry = RunEntry(
                        city=city_name,
                        kind="inference",
                        run_id=run_id,
                        path=run_dir,
                    )
                    self._runs.append(entry)
                    item = QTreeWidgetItem(
                        [city_name, "inference", run_id]
                    )
                    item.setData(0, Qt.UserRole, entry)
                    city_item.addChild(item)

        self.tree_runs.expandAll()

        if not self._runs:
            self._chip_status.setText("No runs")
            self.summary_edit.setPlainText(
                f"[Info] No runs found under:\n{root}"
            )
        else:
            self._chip_status.setText("Ready")
            self.summary_edit.setPlainText(
                "[Info] Select a run on the left to view metrics."
            )

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------

    def _on_run_selected(
        self,
        current: Optional[QTreeWidgetItem],
        previous: Optional[QTreeWidgetItem],
    ) -> None:
        if current is None:
            return

        entry = current.data(0, Qt.UserRole)
        if not isinstance(entry, RunEntry):
            # likely a city header
            return

        self._load_and_render_run(entry)
        
    def _on_change_root(self) -> None:
        """
        Let the user browse for a new results root and refresh the run list.
        """
        start_dir = str(self._results_root)
        path = QFileDialog.getExistingDirectory(
            self,
            "Select results root",
            start_dir,
        )
        if not path:
            return

        self._results_root = Path(path)
        self.lbl_root.setText(f"Results root: {self._results_root}")
        self._scan_and_populate()

    # ------------------------------------------------------------------
    # Run loading & plotting
    # ------------------------------------------------------------------

    def _load_and_render_run(self, entry: RunEntry) -> None:
        run_dir = entry.path

        try:
            eval_diag = self._load_eval_diagnostics(run_dir)
            phys_eval = self._load_geoprior_eval(run_dir)
            phys_params = self._load_phys_params(run_dir)
            pit_data, reliab_data = self._load_pit_and_reliab(run_dir)
        except Exception as exc:  # pragma: no cover
            QMessageBox.warning(
                self,
                "Metrics error",
                f"Failed to load metrics for run:\n{run_dir}\n\n{exc}",
            )
            return

        # remember for pop-out windows
        self._last_entry = entry
        self._last_eval_diag = eval_diag
        self._last_phys_eval = phys_eval
        self._last_pit_data = pit_data
        self._last_reliab_data = reliab_data
        
        self._chip_run.setText(
            f"{entry.city} / {entry.kind} / {entry.run_id}"
        )
        
        scale, dst_unit, src_unit = self._subs_scale_to_mm(phys_eval)
        
        if src_unit == "m" and dst_unit == "mm":
            phys_txt = "m → mm"
        elif src_unit == "mm":
            phys_txt = "mm"
        elif src_unit:
            phys_txt = src_unit
        else:
            phys_txt = "—"
        
        self._chip_units.setText(
            f"Units: diag=mm, phys={phys_txt}"
        )
        
        self._chip_status.setText("OK")

        self._update_summary(entry, eval_diag, phys_eval, phys_params)
        self._update_core_plot(eval_diag, phys_eval)
        self._update_pit_plot(pit_data, reliab_data)


    # ---- loaders -----------------------------------------------------

    def _load_eval_diagnostics(
        self, run_dir: Path
    ) -> Optional[Dict[str, Any]]:
        path = run_dir / "eval_diagnostics.json"
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_geoprior_eval(
        self, run_dir: Path
    ) -> Optional[Dict[str, Any]]:
        # pick latest geoprior_eval_phys_*.json (by name)
        cand = sorted(
            run_dir.glob("geoprior_eval_phys_*.json"),
            key=lambda p: p.name,
        )
        if not cand:
            return None
        path = cand[-1]
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_phys_params(
        self, run_dir: Path
    ) -> Optional[pd.DataFrame]:
        cand = sorted(
            run_dir.glob("*_geopriorsubsnet_physical_parameters.csv"),
            key=lambda p: p.name,
        )
        if not cand:
            return None
        path = cand[-1]
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _load_pit_and_reliab(
        self, run_dir: Path
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Placeholder loader for PIT / reliability diagnostics.

        Once you start saving those per run, adapt this function to
        match whatever filenames and JSON structure you choose.

        Example (what *could* be used)::

            pit_path = run_dir / "pit_histogram.json"  # {"bins": [...], "counts": [...]}
            rel_path = run_dir / "reliability_curve.json"  # {"forecast_prob": [...], "empirical_freq": [...]}

        For now, we simply look for those filenames and ignore if
        missing.
        """
        pit_path = run_dir / "pit_histogram.json"
        rel_path = run_dir / "reliability_curve.json"

        pit = None
        reliab = None

        if pit_path.is_file():
            with pit_path.open("r", encoding="utf-8") as f:
                pit = json.load(f)
        if rel_path.is_file():
            with rel_path.open("r", encoding="utf-8") as f:
                reliab = json.load(f)

        return pit, reliab

    # ---- summary -----------------------------------------------------

    def _update_summary(
        self,
        entry: RunEntry,
        eval_diag: Optional[Dict[str, Any]],
        phys_eval: Optional[Dict[str, Any]],
        phys_params: Optional[pd.DataFrame],
    ) -> None:
        lines: List[str] = []
        lines.append(
            f"Run: {entry.city} / {entry.kind} / {entry.run_id}"
        )
        lines.append(f"Path: {entry.path}")
        lines.append("")

        if phys_eval is not None:
            pm = phys_eval.get("point_metrics", {})
            me = phys_eval.get("metrics_evaluate", {})
            ic = phys_eval.get("interval_calibration", {})
            pdg = phys_eval.get("physics_diagnostics", {})

            lines.append("Point metrics:")
            scale, yunit, _src = self._subs_scale_to_mm(phys_eval)
            
            if "mae" in pm:
                v = float(pm["mae"]) * scale if yunit == "mm" else pm["mae"]
                suf = f" {yunit}" if yunit else ""
                lines.append(f"  mae: {v:.4g}{suf}")
            
            if "mse" in pm:
                if yunit == "mm":
                    v = float(pm["mse"]) * (scale ** 2)
                    suf = f" {yunit}^2"
                else:
                    v = pm["mse"]
                    suf = ""
                lines.append(f"  mse: {v:.4g}{suf}")
            
            if "r2" in pm:
                lines.append(f"  r2: {float(pm['r2']):.4g}")

            if "subs_pred_mae" in me or "subs_pred_mse" in me:
                lines.append("")
                lines.append("Subsidence (evaluate):")
                if "subs_pred_mae" in me:
                    lines.append(
                        f"  mae: {me['subs_pred_mae']:.4g}"
                    )
                if "subs_pred_mse" in me:
                    lines.append(
                        f"  mse: {me['subs_pred_mse']:.4g}"
                    )

            if ic:
                lines.append("")
                lines.append("Interval calibration (80%):")
                cov = ic.get("coverage80_calibrated")
                sh = ic.get("sharpness80_calibrated")
                if cov is not None:
                    lines.append(f"  coverage80_calibrated: {cov:.4g}")
                if sh is not None:
                    lines.append(f"  sharpness80_calibrated: {sh:.4g}")

            if pdg:
                lines.append("")
                lines.append("Physics diagnostics:")
                for key, val in pdg.items():
                    lines.append(f"  {key}: {val:.4g}")

        if eval_diag is not None:
            lines.append("")
            lines.append("Eval diagnostics per year:")
            years = sorted(eval_diag.keys(), key=float)
            for y in years:
                d = eval_diag[y]
                lines.append(
                    f"  {int(float(y))}: "
                    f"MAE={d.get('overall_mae', float('nan')):.3g}, "
                    f"R²={d.get('overall_r2', float('nan')):.3g}, "
                    f"cov80={d.get('coverage80', float('nan')):.3g}, "
                    f"PSS={d.get('pss', float('nan')):.3g}"
                )

        if phys_params is not None and not phys_params.empty:
            lines.append("")
            lines.append("Physical parameters:")
            for _, row in phys_params.iterrows():
                lines.append(
                    f"  {row.get('Parameter')}: {row.get('Value')}"
                )

        if not lines:
            lines = [
                "No metrics could be loaded for this run.",
                "Check that JSON and CSV diagnostics are present.",
            ]

        self.summary_edit.setPlainText("\n".join(lines))

    # ---- plotting: core metrics --------------------------------------
    
    def _show_core_popup(self) -> None:
        if self._last_entry is None:
            QMessageBox.information(
                self,
                "No run selected",
                "Please select a run in the tree first.",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(
            f"Core metrics – {self._last_entry.city} / "
            f"{self._last_entry.kind} / {self._last_entry.run_id}"
        )
        v = QVBoxLayout(dlg)

        canvas = FigureCanvas(Figure(figsize=(7, 4), tight_layout=True))
        toolbar = NavigationToolbar(canvas, dlg)
        v.addWidget(toolbar)
        v.addWidget(canvas)

        fig = canvas.figure
        self._render_core_plot(
            fig,
            canvas,
            self._last_eval_diag,
            self._last_phys_eval,
        )

        dlg.resize(900, 600)
        dlg.exec_()

    def _show_pit_popup(self) -> None:
        if self._last_entry is None:
            QMessageBox.information(
                self,
                "No run selected",
                "Please select a run in the tree first.",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(
            f"PIT / reliability – {self._last_entry.city} / "
            f"{self._last_entry.kind} / {self._last_entry.run_id}"
        )
        v = QVBoxLayout(dlg)

        canvas = FigureCanvas(Figure(figsize=(7, 4), tight_layout=True))
        toolbar = NavigationToolbar(canvas, dlg)
        v.addWidget(toolbar)
        v.addWidget(canvas)

        fig = canvas.figure
        self._render_pit_plot(
            fig,
            canvas,
            self._last_pit_data,
            self._last_reliab_data,
        )

        dlg.resize(900, 600)
        dlg.exec_()

    def _update_core_plot(
        self,
        eval_diag: Optional[Dict[str, Any]],
        phys_eval: Optional[Dict[str, Any]],
    ) -> None:
        self._render_core_plot(
            self.fig_core, self.canvas_core, eval_diag, phys_eval
        )

    def _render_core_plot(
        self,
        fig: Figure,
        canvas: FigureCanvas,
        eval_diag: Optional[Dict[str, Any]],
        phys_eval: Optional[Dict[str, Any]],
    ) -> None:
        fig.clear()

        if eval_diag is None and phys_eval is None:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No eval_diagnostics / geoprior_eval_phys JSON\n"
                "found for this run.",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_axis_off()
            canvas.draw()
            return

        axs = fig.subplots(2, 2, sharex=False)
        axs = axs.ravel()

        # --- panel 1: overall MAE vs year -----------------------------
        if eval_diag is not None:
            years = sorted(eval_diag.keys(), key=float)
            xs = [int(float(y)) for y in years]
            mae = [
                eval_diag[y].get("overall_mae", float("nan"))
                for y in years
            ]
            r2 = [
                eval_diag[y].get("overall_r2", float("nan"))
                for y in years
            ]

            ax = axs[0]
            ax.plot(xs, mae, marker="o", linewidth=1.8)
            ax.set_title("Overall MAE vs year")
            ax.set_ylabel("MAE (mm)")
            ax.grid(True, alpha=0.3)

            ax2 = ax.twinx()
            ax2.plot(xs, r2, marker="s", linestyle="--", linewidth=1.3)
            ax2.set_ylabel("R²")
        else:
            ax = axs[0]
            ax.text(
                0.5,
                0.5,
                "No eval_diagnostics.json",
                ha="center",
                va="center",
            )
            ax.set_axis_off()

        # --- panel 2: coverage80 & PSS vs year ------------------------
        if eval_diag is not None:
            years = sorted(eval_diag.keys(), key=float)
            xs = [int(float(y)) for y in years]
            cov = [
                eval_diag[y].get("coverage80", float("nan"))
                for y in years
            ]
            pss = [
                eval_diag[y].get("pss", float("nan"))
                for y in years
            ]

            ax = axs[1]
            ax.plot(xs, cov, marker="o", linewidth=1.5)
            ax.axhline(0.8, color="gray", linestyle=":", linewidth=1)
            ax.set_ylim(0, 1)
            ax.set_title("Coverage80 & PSS")
            ax.set_ylabel("Coverage80")

            ax2 = ax.twinx()
            ax2.plot(xs, pss, marker="^", linestyle="--", linewidth=1.3)
            ax2.set_ylabel("PSS")
            ax.grid(True, alpha=0.3)
        else:
            axs[1].set_axis_off()

        # --- panel 3: per-horizon MAE --------------------------------
        if phys_eval is not None:
            ph = phys_eval.get("per_horizon", {}) or {}
            ph_mae = ph.get("mae", {}) or {}
        
            if ph_mae:
                scale, yunit, _src = self._subs_scale_to_mm(phys_eval)
                ylabel = f"MAE ({yunit})" if yunit else "MAE"
        
                hs = sorted(ph_mae.keys())
                xs = list(range(1, len(hs) + 1))
                ys = [float(ph_mae[h]) * scale for h in hs]
        
                ax = axs[2]
                ax.plot(xs, ys, marker="o", linewidth=1.8)
                ax.set_title("Per-horizon MAE")
                ax.set_xlabel("Horizon")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
            else:
                axs[2].set_axis_off()
        else:
            axs[2].set_axis_off()

        # --- panel 4: per-horizon R² ----------------------------------
        if phys_eval is not None:
            ph = phys_eval.get("per_horizon", {})
            ph_r2 = ph.get("r2", {})
            if ph_r2:
                hs = sorted(ph_r2.keys())
                xs = list(range(1, len(hs) + 1))
                ys = [ph_r2[h] for h in hs]

                ax = axs[3]
                ax.plot(xs, ys, marker="o", linewidth=1.8)
                ax.set_title("Per-horizon R²")
                ax.set_xlabel("Horizon")
                ax.set_ylabel("R²")
                ax.grid(True, alpha=0.3)
            else:
                axs[3].set_axis_off()
        else:
            axs[3].set_axis_off()

        fig.tight_layout()
        canvas.draw()


    # ---- plotting: PIT / reliability --------------------------------

    def _update_pit_plot(
        self,
        pit_data: Optional[Dict[str, Any]],
        reliab_data: Optional[Dict[str, Any]],
    ) -> None:
        self._render_pit_plot(
            self.fig_pit, self.canvas_pit, pit_data, reliab_data
        )

    def _render_pit_plot(
        self,
        fig: Figure,
        canvas: FigureCanvas,
        pit_data: Optional[Dict[str, Any]],
        reliab_data: Optional[Dict[str, Any]],
    ) -> None:
        fig.clear()
        ax = fig.add_subplot(111)

        if pit_data is None and reliab_data is None:
            ax.text(
                0.5,
                0.5,
                "No PIT / reliability diagnostics found.\n"
                "Once you save e.g. 'pit_histogram.json' and "
                "'reliability_curve.json'\nper run, this panel "
                "will show them.",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_axis_off()
            canvas.draw()
            return

        if pit_data is not None and "bins" in pit_data \
                and "counts" in pit_data:
            bins = pit_data["bins"]
            counts = pit_data["counts"]
            ax.bar(
                bins,
                counts,
                width=1.0 / max(len(bins), 10),
                alpha=0.5,
                label="PIT histogram",
            )

        if reliab_data is not None and "forecast_prob" in reliab_data \
                and "empirical_freq" in reliab_data:
            fp = reliab_data["forecast_prob"]
            ef = reliab_data["empirical_freq"]
            ax.plot(
                fp,
                ef,
                marker="o",
                linewidth=1.5,
                label="Reliability curve",
            )
            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                linewidth=1,
                label="Perfect",
            )

        ax.set_xlabel("Probability / PIT")
        ax.set_ylabel("Frequency")
        ax.set_title("PIT & reliability diagnostics")
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend(loc="best")

        fig.tight_layout()
        canvas.draw()
