# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Run comparison tool for GeoPrior GUI.

This tool lets you select multiple GeoPriorSubsNet runs (train/tuning/
inference) and compare their core metrics and high-level configuration
side by side.

It expects the standard Stage-1 layout, e.g.:

<results_root>/
    nansha_GeoPriorSubsNet_stage1/
        train_YYYYMMDD-HHMMSS/
            geoprior_eval_phys_*.json
            eval_diagnostics.json
            *training_summary.json (optional)
        tuning/
            run_YYYYMMDD-HHMMSS/
                geoprior_eval_phys_*.json
                eval_diagnostics.json
                *tuning_summary.json (optional)
        inference/
            YYYYMMDD-HHMMSS/
                geoprior_eval_phys_*.json
                eval_diagnostics.json
                *inference_summary.json (optional)
        artifacts/
            manifest.json  (fallback config if per-run summaries missing)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QMessageBox,
    QFileDialog,   
)

from ...styles import SECONDARY_TBLUE  # same pattern as BuildNPZTool


# ----------------------------------------------------------------------
# Small helper dataclass for a discovered run
# ----------------------------------------------------------------------


@dataclass
class RunEntry:
    city: str
    kind: str         # "train", "tuning", "inference"
    run_id: str       # e.g. "train_20251110-122128"
    path: Path


# ----------------------------------------------------------------------
# Main tool
# ----------------------------------------------------------------------


class RunComparisonTool(QWidget):
    """
    Run comparison tool for the Tools tab.

    Parameters
    ----------
    app_ctx : object, optional
        Reference to the main GeoPrior GUI. Used for:
        - ``gui_runs_root`` / ``results_root`` (results root directory);
        - ``city_edit`` (to pre-select city when present, later if needed).
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

        self._init_ui()
        self._scan_and_populate()

    # ------------------------------------------------------------------
    # Results root detection
    # ------------------------------------------------------------------
    # You have the option to let user browse to select the results root,
    # or by default use the ctx results_root. When the user changes the
    # results root, we refresh the run list accordingly.
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


        # ------- Middle: run list + comparison area -------------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # -- Left: runs tree (multi-select) ----------------------------
        runs_group = QGroupBox("Available runs")
        runs_layout = QVBoxLayout(runs_group)
        runs_layout.setContentsMargins(6, 6, 6, 6)
        runs_layout.setSpacing(4)

        self.tree_runs = QTreeWidget()
        self.tree_runs.setHeaderLabels(["City", "Kind", "Run id"])
        self.tree_runs.setColumnWidth(0, 130)
        self.tree_runs.setColumnWidth(1, 80)
        self.tree_runs.setAlternatingRowColors(True)
        self.tree_runs.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )

        runs_layout.addWidget(self.tree_runs)
        splitter.addWidget(runs_group)

        # -- Right: metrics table + config diff ------------------------
        right_group = QGroupBox("Run comparison")
        right_layout = QVBoxLayout(right_group)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(4)

        # Metrics table
        metrics_box = QGroupBox("Metrics (selected runs)")
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(4, 4, 4, 4)

        self.table_metrics = QTableWidget()
        self.table_metrics.setColumnCount(0)
        self.table_metrics.setRowCount(0)
        self.table_metrics.setAlternatingRowColors(True)
        self.table_metrics.horizontalHeader().setStretchLastSection(True)

        metrics_layout.addWidget(self.table_metrics)
        right_layout.addWidget(metrics_box, 3)

        # Config diff
        cfg_box = QGroupBox("Config differences (high-level)")
        cfg_layout = QVBoxLayout(cfg_box)
        cfg_layout.setContentsMargins(4, 4, 4, 4)

        self.cfg_edit = QPlainTextEdit()
        self.cfg_edit.setReadOnly(True)
        self.cfg_edit.setMinimumHeight(140)

        cfg_layout.addWidget(self.cfg_edit)
        right_layout.addWidget(cfg_box, 2)

        splitter.addWidget(right_group)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)

        main.addWidget(splitter, stretch=1)

        # ------- Connections ------------------------------------------
        self.btn_change_root.clicked.connect(self._on_change_root)
        self.btn_rescan.clicked.connect(self._scan_and_populate)
        self.tree_runs.itemSelectionChanged.connect(
            self._on_selection_changed
        )


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
            self.cfg_edit.setPlainText(
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
            self.cfg_edit.setPlainText(
                f"[Info] No runs found under:\n{root}"
            )
        else:
            self.cfg_edit.setPlainText(
                "[Info] Select two or more runs on the left to "
                "compare metrics and configs."
            )

        # clear metrics table
        self._clear_metrics_table()

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------
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

    def _on_selection_changed(self) -> None:
        items = self.tree_runs.selectedItems()
        entries: List[RunEntry] = []
        for it in items:
            entry = it.data(0, Qt.UserRole)
            if isinstance(entry, RunEntry):
                entries.append(entry)

        self._update_comparison(entries)

    # ------------------------------------------------------------------
    # Comparison core
    # ------------------------------------------------------------------

    def _update_comparison(self, entries: List[RunEntry]) -> None:
        if not entries:
            self._clear_metrics_table()
            self.cfg_edit.setPlainText(
                "No run selected. Select one or more runs to compare."
            )
            return

        metrics_rows: List[Dict[str, Any]] = []
        cfg_rows: List[Dict[str, Any]] = []

        for entry in entries:
            run_dir = entry.path
            try:
                eval_diag = self._load_eval_diagnostics(run_dir)
                phys_eval = self._load_geoprior_eval(run_dir)
                cfg = self._load_run_config(entry)
            except Exception as exc:  # pragma: no cover
                QMessageBox.warning(
                    self,
                    "Run load error",
                    f"Failed to load data for run:\n{run_dir}\n\n{exc}",
                )
                continue

            metrics_rows.append(
                self._build_metrics_row(entry, phys_eval, eval_diag)
            )
            cfg_rows.append(cfg or {})

        self._update_metrics_table(metrics_rows)
        self._update_config_diff(entries, cfg_rows)

    # ------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------

    def _clear_metrics_table(self) -> None:
        self.table_metrics.setRowCount(0)
        self.table_metrics.setColumnCount(0)
        self.table_metrics.setHorizontalHeaderLabels([])

    def _update_metrics_table(
        self, rows: List[Dict[str, Any]]
    ) -> None:
        self._clear_metrics_table()
        if not rows:
            return

        # Column spec: (key, label)
        columns = [
            ("city", "City"),
            ("kind", "Kind"),
            ("run_id", "Run ID"),
            ("mae", "MAE"),
            ("r2", "R²"),
            ("subs_mae", "Subs MAE"),
            ("subs_mse", "Subs MSE"),
            ("cov80_cal", "Cov80 cal"),
            ("sharp80_cal", "Sharp80 cal"),
            ("pss_last", "PSS (last yr)"),
            ("last_year", "Last year"),
            ("horizon", "H"),
            ("eps_prior", "ε_prior"),
            ("eps_cons", "ε_cons"),
        ]

        self.table_metrics.setColumnCount(len(columns))
        self.table_metrics.setRowCount(len(rows))
        self.table_metrics.setHorizontalHeaderLabels(
            [lbl for _, lbl in columns]
        )

        numeric_keys = {
            "mae",
            "r2",
            "subs_mae",
            "subs_mse",
            "cov80_cal",
            "sharp80_cal",
            "pss_last",
            "horizon",
            "eps_prior",
            "eps_cons",
        }

        for i, row in enumerate(rows):
            for j, (key, _label) in enumerate(columns):
                val = row.get(key, None)
                if val is None:
                    text = "–"
                elif isinstance(val, float):
                    text = f"{val:.3g}"
                else:
                    text = str(val)

                item = QTableWidgetItem(text)
                if key in numeric_keys:
                    item.setTextAlignment(
                        Qt.AlignRight | Qt.AlignVCenter
                    )
                self.table_metrics.setItem(i, j, item)

        self.table_metrics.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Config diff
    # ------------------------------------------------------------------

    def _update_config_diff(
        self,
        entries: List[RunEntry],
        cfg_rows: List[Dict[str, Any]],
    ) -> None:
        if not cfg_rows or all(not c for c in cfg_rows):
            self.cfg_edit.setPlainText(
                "No config information found for selected runs.\n"
                "Make sure manifest / summary JSON files are present."
            )
            return

        # High-level keys of interest
        keys = [
            "CITY_NAME",
            "MODEL_NAME",
            "TRAIN_END_YEAR",
            "FORECAST_START_YEAR",
            "FORECAST_HORIZON_YEARS",
            "TIME_STEPS",
            "MODE",
            "TIME_COL",
            "LON_COL",
            "LAT_COL",
            "SUBSIDENCE_COL",
            "GWL_COL",
            "H_FIELD_COL_NAME",
            "STATIC_DRIVER_FEATURES",
            "DYNAMIC_DRIVER_FEATURES",
            "FUTURE_DRIVER_FEATURES",
        ]

        lines: List[str] = []
        lines.append("Config comparison (selected runs):")
        lines.append("")

        for key in keys:
            vals: List[str] = []
            for cfg in cfg_rows:
                v = cfg.get(key, "–")
                if isinstance(v, (list, tuple)):
                    v = ", ".join(str(x) for x in v)
                vals.append(str(v))

            unique_vals = set(vals)
            if len(unique_vals) <= 1:
                # same across runs
                lines.append(
                    f"- {key}: {vals[0]}  (same across runs)"
                )
            else:
                lines.append(f"- {key}:")
                for entry, v in zip(entries, vals):
                    lines.append(
                        f"    {entry.city}/{entry.kind}/"
                        f"{entry.run_id}: {v}"
                    )
            lines.append("")

        self.cfg_edit.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # JSON loaders / helpers
    # ------------------------------------------------------------------

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

    def _load_run_config(self, entry: RunEntry) -> Dict[str, Any]:
        """
        Heuristic loader for a config dict for a given run.

        Priority:
        1. Per-run *summary JSON in run_dir with a 'config' key.
        2. Per-run manifest*.json with optional 'config' key.
        3. Stage-1 artifacts/manifest.json for that city.
        """
        run_dir = entry.path

        # Per-run summaries / manifests in run_dir itself
        cands = []
        cands += list(run_dir.glob("*training_summary*.json"))
        cands += list(run_dir.glob("*tuning_summary*.json"))
        cands += list(run_dir.glob("*inference_summary*.json"))
        cands += list(run_dir.glob("manifest*.json"))

        for path in cands:
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue

            if isinstance(payload, dict) and "config" in payload:
                return dict(payload["config"])
            if isinstance(payload, dict):
                return dict(payload)

        # Fallback: city-level Stage-1 manifest
        # For train_*: run_dir.parent is <city>_GeoPriorSubsNet_stage1
        # For tuning/inference: run_dir.parent.parent is that city dir.
        candidates_city_roots = [
            run_dir.parent,
            run_dir.parent.parent if run_dir.parent is not None else None,
        ]
        for city_root in candidates_city_roots:
            if not city_root or not city_root.is_dir():
                continue

            path = city_root / "artifacts" / "manifest.json"
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    continue

                if isinstance(payload, dict) and "config" in payload:
                    return dict(payload["config"])
                if isinstance(payload, dict):
                    return dict(payload)

        return {}

    def _build_metrics_row(
        self,
        entry: RunEntry,
        phys_eval: Optional[Dict[str, Any]],
        eval_diag: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "city": entry.city,
            "kind": entry.kind,
            "run_id": entry.run_id,
        }

        if phys_eval is not None:
            pm = phys_eval.get("point_metrics", {})
            me = phys_eval.get("metrics_evaluate", {})
            ic = phys_eval.get("interval_calibration", {})
            pdg = phys_eval.get("physics_diagnostics", {})

            row["mae"] = pm.get("mae")
            row["mse"] = pm.get("mse")
            row["r2"] = pm.get("r2")

            row["subs_mae"] = me.get("subs_pred_mae")
            row["subs_mse"] = me.get("subs_pred_mse")

            row["cov80_cal"] = ic.get("coverage80_calibrated")
            row["sharp80_cal"] = ic.get("sharpness80_calibrated")

            row["horizon"] = phys_eval.get("horizon")
            row["eps_prior"] = pdg.get("epsilon_prior")
            row["eps_cons"] = pdg.get("epsilon_cons")

        if eval_diag is not None:
            years = sorted(eval_diag.keys(), key=float)
            if years:
                last_y = years[-1]
                d = eval_diag[last_y]
                row["pss_last"] = d.get("pss")
                try:
                    row["last_year"] = int(float(last_y))
                except Exception:
                    row["last_year"] = last_y

        return row
