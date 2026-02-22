# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Physics diagnostics tool for GeoPrior GUI.

This panel focuses on physics-related diagnostics produced by
GeoPriorSubsNet, reading (per run):

- ``geoprior_eval_phys_*.json``:
    * ``metrics_evaluate.{epsilon_prior, epsilon_cons, ...}``
    * ``physics_diagnostics.{epsilon_prior, epsilon_cons}``
    * ``interval_calibration.{coverage80_*_phys, sharpness80_*_phys}``
    * ``censor_stratified.{mae_censored, mae_uncensored, ...}``.

The UI mirrors the Metrics dashboard:
- results_root selector + rescan;
- tree of runs on the left;
- textual summary;
- a Matplotlib figure with small bar plots for the physics terms,
  plus a "Pop out…" button to view the figure in a larger dialog.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class PhysicsDiagnosticsTool(QWidget):
    """
    Physics diagnostics panel for the Tools tab.

    Parameters
    ----------
    app_ctx : object, optional
        Reference to the main GeoPrior GUI. Used for:
        - ``results_root`` / ``gui_runs_root`` (results root directory).
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

        # cache for "pop out" figure
        self._last_entry: Optional[RunEntry] = None
        self._last_phys_eval: Optional[Dict[str, Any]] = None

        self._init_ui()
        self._scan_and_populate()

    # ------------------------------------------------------------------
    # Results root detection
    # ------------------------------------------------------------------
    def _detect_results_root(self) -> Path:
        ctx = self._app_ctx
        if ctx is not None:
            root = getattr(ctx, "results_root", None) or getattr(
                ctx, "gui_runs_root", None
            )
            if root:
                return Path(root)

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
        summary_group = QGroupBox("Physics summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setContentsMargins(6, 6, 6, 6)
        summary_layout.setSpacing(4)

        self.summary_edit = QPlainTextEdit()
        self.summary_edit.setReadOnly(True)
        self.summary_edit.setMinimumWidth(260)
        summary_layout.addWidget(self.summary_edit)

        mid.addWidget(summary_group, 1)
        main.addLayout(mid, stretch=1)

        # ------- Bottom: physics figure -------------------------------
        phys_group = QGroupBox("Physics diagnostics")
        phys_layout = QVBoxLayout(phys_group)
        phys_layout.setContentsMargins(6, 6, 6, 6)
        phys_layout.setSpacing(4)

        # tiny row for "Pop out…" button
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_popout = QPushButton("Pop out…")
        self.btn_popout.setToolTip("Open physics diagnostics in a "
                                   "separate window.")
        self.btn_popout.setStyleSheet(
            f"QPushButton:hover {{ background-color: {SECONDARY_TBLUE}; }}"
        )
        btn_row.addWidget(self.btn_popout)
        phys_layout.addLayout(btn_row)

        self.fig_phys = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas_phys = FigureCanvas(self.fig_phys)
        self.toolbar_phys = NavigationToolbar(self.canvas_phys, self)

        phys_layout.addWidget(self.toolbar_phys)
        phys_layout.addWidget(self.canvas_phys)

        main.addWidget(phys_group, stretch=2)

        # ------- Connections ------------------------------------------
        self.btn_change_root.clicked.connect(self._on_change_root)
        self.btn_rescan.clicked.connect(self._scan_and_populate)
        self.tree_runs.currentItemChanged.connect(self._on_run_selected)
        self.btn_popout.clicked.connect(self._on_popout_clicked)

    # ------------------------------------------------------------------
    # Change root
    # ------------------------------------------------------------------
    def _on_change_root(self) -> None:
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
    # Run discovery
    # ------------------------------------------------------------------
    def _scan_and_populate(self) -> None:
        self._runs = []
        self.tree_runs.clear()

        root = self._results_root
        if not root.is_dir():
            self.summary_edit.setPlainText(
                f"[Info] Results root does not exist:\n{root}"
            )
            self.fig_phys.clear()
            self.canvas_phys.draw()
            return

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

            # TRAIN runs
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
                item = QTreeWidgetItem([city_name, "train, ", run_id])
                item.setText(1, "train")
                item.setData(0, Qt.UserRole, entry)
                city_item.addChild(item)

            # TUNING runs
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
                    item = QTreeWidgetItem([city_name, "tuning", run_id])
                    item.setData(0, Qt.UserRole, entry)
                    city_item.addChild(item)

            # INFERENCE runs
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
                    item = QTreeWidgetItem([city_name, "inference", run_id])
                    item.setData(0, Qt.UserRole, entry)
                    city_item.addChild(item)

        self.tree_runs.expandAll()

        if not self._runs:
            self.summary_edit.setPlainText(
                f"[Info] No runs found under:\n{root}"
            )
        else:
            self.summary_edit.setPlainText(
                "[Info] Select a run on the left to view physics "
                "diagnostics."
            )

        # clear figure when rescanning
        self.fig_phys.clear()
        self.canvas_phys.draw()
        self._last_entry = None
        self._last_phys_eval = None

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
            return

        self._load_and_render_run(entry)

    # ------------------------------------------------------------------
    # Run loading & plotting
    # ------------------------------------------------------------------
    def _load_and_render_run(self, entry: RunEntry) -> None:
        run_dir = entry.path

        try:
            phys_eval = self._load_geoprior_eval(run_dir)
        except Exception as exc:  # pragma: no cover
            QMessageBox.warning(
                self,
                "Physics diagnostics error",
                f"Failed to load physics metrics for run:\n"
                f"{run_dir}\n\n{exc}",
            )
            return

        self._last_entry = entry
        self._last_phys_eval = phys_eval

        self._update_summary(entry, phys_eval)
        self._update_phys_plot(phys_eval)

    # ---- loaders -----------------------------------------------------
    def _load_geoprior_eval(
        self, run_dir: Path
    ) -> Optional[Dict[str, Any]]:
        cand = sorted(
            run_dir.glob("geoprior_eval_phys_*.json"),
            key=lambda p: p.name,
        )
        if not cand:
            return None
        path = cand[-1]
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ---- summary -----------------------------------------------------
    def _update_summary(
        self,
        entry: RunEntry,
        phys_eval: Optional[Dict[str, Any]],
    ) -> None:
        lines: List[str] = []
        lines.append(
            f"Run: {entry.city} / {entry.kind} / {entry.run_id}"
        )
        lines.append(f"Path: {entry.path}")
        lines.append("")

        if phys_eval is None:
            lines.append("No geoprior_eval_phys_*.json found "
                         "for this run.")
            self.summary_edit.setPlainText("\n".join(lines))
            return

        me = phys_eval.get("metrics_evaluate", {})
        ic = phys_eval.get("interval_calibration", {})
        pdg = phys_eval.get("physics_diagnostics", {})
        cs = phys_eval.get("censor_stratified", {})

        lines.append("Physics residuals:")
        eps_p = me.get("epsilon_prior", pdg.get("epsilon_prior"))
        eps_c = me.get("epsilon_cons", pdg.get("epsilon_cons"))
        if eps_p is not None:
            lines.append(f"  epsilon_prior: {eps_p:.4g}")
        if eps_c is not None:
            lines.append(f"  epsilon_cons:  {eps_c:.4g}")

        if ic:
            lines.append("")
            lines.append("Interval calibration (physics):")
            tgt = ic.get("target", 0.8)
            lines.append(f"  target: {tgt:.3g}")
            for key in (
                "coverage80_uncalibrated_phys",
                "coverage80_calibrated_phys",
                "sharpness80_uncalibrated_phys",
                "sharpness80_calibrated_phys",
            ):
                if key in ic:
                    lines.append(f"  {key}: {ic[key]:.4g}")

        if cs:
            lines.append("")
            lines.append("Censor-stratified metrics:")
            flag = cs.get("flag_name", "?")
            thr = cs.get("threshold", None)
            lines.append(f"  flag: {flag}")
            if thr is not None:
                lines.append(f"  threshold: {thr}")
            if "mae_censored" in cs:
                lines.append(f"  mae_censored: {cs['mae_censored']:.4g}")
            if "mae_uncensored" in cs:
                lines.append(
                    f"  mae_uncensored: {cs['mae_uncensored']:.4g}"
                )

        self.summary_edit.setPlainText("\n".join(lines))

    # ---- plotting helpers --------------------------------------------
    def _plot_phys_into_figure(
        self,
        fig: Figure,
        phys_eval: Optional[Dict[str, Any]],
    ) -> None:
        fig.clear()

        if phys_eval is None:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No geoprior_eval_phys JSON found\nfor this run.",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_axis_off()
            return

        me = phys_eval.get("metrics_evaluate", {})
        ic = phys_eval.get("interval_calibration", {})
        pdg = phys_eval.get("physics_diagnostics", {})
        cs = phys_eval.get("censor_stratified", {})

        axs = fig.subplots(2, 2)
        axs = axs.ravel()

        # Panel 1: epsilons (residuals)
        eps_p = me.get("epsilon_prior", pdg.get("epsilon_prior"))
        eps_c = me.get("epsilon_cons", pdg.get("epsilon_cons"))
        if eps_p is not None or eps_c is not None:
            ax = axs[0]
            labels: List[str] = []
            vals: List[float] = []
            if eps_p is not None:
                labels.append("ε_prior")
                vals.append(eps_p)
            if eps_c is not None:
                labels.append("ε_cons")
                vals.append(eps_c)
            xs = range(len(vals))
            ax.bar(list(xs), vals, tick_label=labels)
            ax.set_title("Physics residuals")
            ax.set_ylabel("Value")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            axs[0].set_axis_off()

        # Panel 2: coverage80_phys
        cov_u = ic.get("coverage80_uncalibrated_phys")
        cov_c = ic.get("coverage80_calibrated_phys")
        tgt = ic.get("target", 0.8)
        if cov_u is not None or cov_c is not None:
            ax = axs[1]
            labels = []
            vals = []
            if cov_u is not None:
                labels.append("uncalib")
                vals.append(cov_u)
            if cov_c is not None:
                labels.append("calib")
                vals.append(cov_c)
            xs = range(len(vals))
            ax.bar(list(xs), vals, tick_label=labels)
            ax.axhline(tgt, linestyle=":", linewidth=1)
            ax.set_ylim(0.0, 1.0)
            ax.set_title("Coverage80 (phys)")
            ax.set_ylabel("Coverage")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            axs[1].set_axis_off()

        # Panel 3: sharpness80_phys
        sh_u = ic.get("sharpness80_uncalibrated_phys")
        sh_c = ic.get("sharpness80_calibrated_phys")
        if sh_u is not None or sh_c is not None:
            ax = axs[2]
            labels = []
            vals = []
            if sh_u is not None:
                labels.append("uncalib")
                vals.append(sh_u)
            if sh_c is not None:
                labels.append("calib")
                vals.append(sh_c)
            xs = range(len(vals))
            ax.bar(list(xs), vals, tick_label=labels)
            ax.set_title("Sharpness80 (phys)")
            ax.set_ylabel("Sharpness")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            axs[2].set_axis_off()

        # Panel 4: censor-stratified MAE
        mae_c = cs.get("mae_censored") if cs else None
        mae_u = cs.get("mae_uncensored") if cs else None
        if mae_c is not None or mae_u is not None:
            ax = axs[3]
            labels = []
            vals = []
            if mae_c is not None:
                labels.append("censored")
                vals.append(mae_c)
            if mae_u is not None:
                labels.append("uncensored")
                vals.append(mae_u)
            xs = range(len(vals))
            ax.bar(list(xs), vals, tick_label=labels)
            flag = cs.get("flag_name", "?") if cs else "?"
            ax.set_title(f"MAE by censor flag ({flag})")
            ax.set_ylabel("MAE")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            axs[3].set_axis_off()

        fig.tight_layout()

    def _update_phys_plot(
        self,
        phys_eval: Optional[Dict[str, Any]],
    ) -> None:
        self._plot_phys_into_figure(self.fig_phys, phys_eval)
        self.canvas_phys.draw()

    # ------------------------------------------------------------------
    # Pop-out handling
    # ------------------------------------------------------------------
    def _on_popout_clicked(self) -> None:
        if self._last_phys_eval is None:
            QMessageBox.information(
                self,
                "No data",
                "Select a run first to populate physics diagnostics.",
            )
            return

        title = "Physics diagnostics"
        if self._last_entry is not None:
            e = self._last_entry
            title = f"Physics diagnostics – {e.city}/{e.kind}/{e.run_id}"

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        v = QVBoxLayout(dlg)

        fig = Figure(figsize=(7, 4), tight_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dlg)

        v.addWidget(toolbar)
        v.addWidget(canvas)

        self._plot_phys_into_figure(fig, self._last_phys_eval)
        canvas.draw()

        dlg.resize(900, 600)
        dlg.exec_()
