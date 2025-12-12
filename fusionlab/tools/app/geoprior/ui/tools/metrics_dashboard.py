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
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QGroupBox,
    QPlainTextEdit,
    QSplitter,
    QMessageBox,
    QFileDialog, 
    QDialog,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from ...styles import SECONDARY_TBLUE  


@dataclass
class RunEntry:
    city: str
    kind: str         # "train", "tuning", "inference"
    run_id: str       # e.g. "train_20251110-122128"
    path: Path

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

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(6)

        # ------- Top: root + rescan -----------------------------------
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.lbl_root = QLabel(f"Results root: {self._results_root}")
        self.lbl_root.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.btn_change_root = QPushButton("Change root…")
        self.btn_change_root.setStyleSheet(
            f"QPushButton:hover {{ background-color: {SECONDARY_TBLUE}; }}"
        )

        self.btn_rescan = QPushButton("Rescan runs")
        self.btn_rescan.setStyleSheet(
            f"QPushButton:hover {{ background-color: {SECONDARY_TBLUE}; }}"
        )

        top_row.addWidget(self.lbl_root, 1)
        top_row.addWidget(self.btn_change_root)
        top_row.addWidget(self.btn_rescan)
        main.addLayout(top_row)

        # ------- Middle: run list + summary ---------------------------
        mid = QHBoxLayout()
        mid.setSpacing(8)

        # Left: tree of runs
        runs_group = QGroupBox("Available runs")
        runs_layout = QVBoxLayout(runs_group)
        runs_layout.setContentsMargins(6, 6, 6, 6)
        runs_layout.setSpacing(4)

        self.tree_runs = QTreeWidget()
        self.tree_runs.setHeaderLabels(["City", "Kind", "Run id"])
        self.tree_runs.setColumnWidth(0, 130)
        self.tree_runs.setColumnWidth(1, 80)
        self.tree_runs.setAlternatingRowColors(True)
        runs_layout.addWidget(self.tree_runs)

        mid.addWidget(runs_group, 2)

        # Right: summary
        summary_group = QGroupBox("Run summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setContentsMargins(6, 6, 6, 6)
        summary_layout.setSpacing(4)

        self.summary_edit = QPlainTextEdit()
        self.summary_edit.setReadOnly(True)
        self.summary_edit.setMinimumWidth(260)
        summary_layout.addWidget(self.summary_edit)

        mid.addWidget(summary_group, 1)

        main.addLayout(mid, stretch=1)

        # ------- Bottom: plots (splitter) -----------------------------
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        # -- Core metrics figure
        core_group = QGroupBox("Core metrics (overall + per horizon)")
        core_v = QVBoxLayout(core_group)
        core_v.setContentsMargins(6, 6, 6, 6)
        core_v.setSpacing(4)

        # --- pop-out button for core metrics -------------------------
        core_btn_row = QHBoxLayout()
        core_btn_row.addStretch(1)
        self.btn_pop_core = QPushButton("Pop out…")
        self.btn_pop_core.setToolTip(
            "Open core metrics in a larger window."
        )
        self.btn_pop_core.setMaximumWidth(110)
        core_btn_row.addWidget(self.btn_pop_core)
        core_v.addLayout(core_btn_row)

        self.fig_core = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas_core = FigureCanvas(self.fig_core)
        self.toolbar_core = NavigationToolbar(self.canvas_core, self)

        core_v.addWidget(self.toolbar_core)
        core_v.addWidget(self.canvas_core)

        splitter.addWidget(core_group)

        # -- PIT / reliability figure
        pit_group = QGroupBox("PIT / reliability diagnostics")
        pit_v = QVBoxLayout(pit_group)
        pit_v.setContentsMargins(6, 6, 6, 6)
        pit_v.setSpacing(4)

        # --- pop-out button for PIT / reliability --------------------
        pit_btn_row = QHBoxLayout()
        pit_btn_row.addStretch(1)
        self.btn_pop_pit = QPushButton("Pop out…")
        self.btn_pop_pit.setToolTip(
            "Open PIT / reliability diagnostics in a larger window."
        )
        self.btn_pop_pit.setMaximumWidth(110)
        pit_btn_row.addWidget(self.btn_pop_pit)
        pit_v.addLayout(pit_btn_row)

        self.fig_pit = Figure(figsize=(5, 2.5), tight_layout=True)
        self.canvas_pit = FigureCanvas(self.fig_pit)
        self.toolbar_pit = NavigationToolbar(self.canvas_pit, self)

        pit_v.addWidget(self.toolbar_pit)
        pit_v.addWidget(self.canvas_pit)

        splitter.addWidget(pit_group)

        # preferable default: give a bit more height to first figure
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main.addWidget(splitter, stretch=2)

        # ------- Connections ------------------------------------------
        self.btn_change_root.clicked.connect(self._on_change_root)
        self.btn_rescan.clicked.connect(self._scan_and_populate)
        self.tree_runs.currentItemChanged.connect(
            self._on_run_selected
        )
        self.btn_pop_core.clicked.connect(self._show_core_popup)
        self.btn_pop_pit.clicked.connect(self._show_pit_popup)

    # ------------------------------------------------------------------
    # Run discovery
    # ------------------------------------------------------------------

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
            self.summary_edit.setPlainText(
                f"[Info] No runs found under:\n{root}"
            )
        else:
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
            for key in ("mae", "mse", "r2"):
                if key in pm:
                    lines.append(f"  {key}: {pm[key]:.4g}")

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
            ax.set_ylabel("MAE")
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
            ph = phys_eval.get("per_horizon", {})
            ph_mae = ph.get("mae", {})
            if ph_mae:
                hs = sorted(ph_mae.keys())
                xs = list(range(1, len(hs) + 1))
                ys = [ph_mae[h] for h in hs]

                ax = axs[2]
                ax.plot(xs, ys, marker="o", linewidth=1.8)
                ax.set_title("Per-horizon MAE")
                ax.set_xlabel("Horizon")
                ax.set_ylabel("MAE")
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
