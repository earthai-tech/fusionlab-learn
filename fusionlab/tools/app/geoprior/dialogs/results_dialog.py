# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Nice dialog to inspect GeoPrior evaluation JSON files.

Usage
-----
from fusionlab.tools.app.geoprior.results_dialog import GeoPriorResultsDialog

# From a training / tuning finished slot:
GeoPriorResultsDialog.show_for_result(self, result, title="Training metrics")
"""

from __future__ import annotations

import os
import json
import glob
import shutil
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QFont, QDesktopServices
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTabWidget,
    QWidget,
    QFormLayout,
    QLabel,
    QPushButton,
    QDialogButtonBox,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QFileDialog,
    QMessageBox,
)


class GeoPriorResultsDialog(QDialog):
    """
    Dialog to visualise metrics from a GeoPrior evaluation JSON.
    """

    # ------------------------------------------------------------------
    # Convenience entry point for GUI slots
    # ------------------------------------------------------------------
    @classmethod
    def show_for_result(
        cls,
        parent: QWidget,
        result: Dict[str, Any],
        *,
        title: Optional[str] = None,
    ) -> None:
        """
        Open a metrics dialog for a finished job.

        Parameters
        ----------
        parent : QWidget
            Parent window (usually the main GUI).
        result : dict
            Result dict emitted by TrainingThread / TuningThread.
            Expected keys:
              - 'metrics_json' or 'metrics_json_path' (optional)
              - 'run_output_path' or 'run_dir' for fallback search.
        title : str, optional
            Window title override.
        """
        # 1) Direct path from job result (preferred)
        metrics_path = result.get("metrics_json") or result.get(
            "metrics_json_path"
        )

        # 2) Fallback: search under run directory for the latest JSON
        if not metrics_path or not os.path.exists(metrics_path):
            run_dir = result.get("run_output_path") or result.get("run_dir")
            if run_dir and os.path.isdir(run_dir):
                candidates: list[str] = []
                for pattern in (
                    "geoprior_eval_phys_*.json",
                    "geoprior_eval_phys_tuned_*.json",
                ):
                    candidates.extend(
                        glob.glob(os.path.join(run_dir, pattern))
                    )
                if candidates:
                    candidates.sort(
                        key=os.path.getmtime, reverse=True
                    )
                    metrics_path = candidates[0]

        if not metrics_path or not os.path.exists(metrics_path):
            QMessageBox.information(
                parent,
                "No evaluation metrics",
                "This run does not have an evaluation JSON yet.\n"
                "Enable evaluation in the Train/Tune tab and try again.",
            )
            return

        dlg = cls(metrics_path, parent=parent, title=title)
        dlg.exec_()

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        json_path: str,
        parent: Optional[QWidget] = None,
        *,
        title: Optional[str] = None,
        city: Optional[str]=None, 
    ) -> None:
        super().__init__(parent)

        self._json_path = os.path.abspath(json_path)
        self._data: Dict[str, Any] = self._load_json(self._json_path)
        self._city: Optional[str] = city  
        self.setWindowTitle(title or "GeoPrior evaluation metrics")
        self.setModal(True)
        self.resize(720, 520)

        main_layout = QVBoxLayout(self)

        # Header
        header = QLabel(self._build_header_text())
        header.setWordWrap(True)
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bold_font = QFont()
        bold_font.setPointSize(bold_font.pointSize() + 1)
        bold_font.setBold(True)
        header.setFont(bold_font)
        main_layout.addWidget(header)

        # Tabs
        tabs = QTabWidget(self)
        tabs.addTab(self._build_summary_tab(), "Summary")
        tabs.addTab(self._build_metrics_tab(), "Metrics")
        tabs.addTab(self._build_per_horizon_tab(), "Per horizon")
        tabs.addTab(self._build_json_tab(), "Raw JSON")
        main_layout.addWidget(tabs, 1)

        # Buttons: Close / Save / Open folder
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_save = QPushButton("Save as…")
        btn_open = QPushButton("Open folder")
        btn_box.addButton(btn_save, QDialogButtonBox.ActionRole)
        btn_box.addButton(btn_open, QDialogButtonBox.ActionRole)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)

        btn_save.clicked.connect(self._on_save_as)
        btn_open.clicked.connect(self._on_open_folder)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_json(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(
                self,
                "Failed to load metrics",
                f"Could not read metrics JSON:\n{path}\n\n{exc}",
            )
            return {}

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    def _build_header_text(self) -> str:
        meta = self._data
        ts = meta.get("timestamp", "unknown time")
        city = meta.get("city", self._city) or meta.get(
            "dataset_name", "unknown city")
        horizon = meta.get("horizon")
        q = meta.get("quantiles")
        q_str = (
            ", ".join(str(v) for v in q)
            if isinstance(q, (list, tuple))
            else "n/a"
        )
        return (
            f"GeoPrior evaluation for {city} — {ts}  |  "
            f"H={horizon}  |  q={q_str}"
        )

    def _build_summary_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QFormLayout(w)
        layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        meta = self._data
        point = meta.get("point_metrics", {})
        interval = meta.get("interval_calibration", {})
        phys = meta.get("physics_diagnostics", {})

        layout.addRow(
            "Timestamp:",
            QLabel(str(meta.get("timestamp", "n/a"))),
        )
        layout.addRow(
            "City / dataset:",
            QLabel(str(meta.get("city", self._city) or meta.get(
                "dataset_name", "n/a"))),
        )
        layout.addRow(
            "Horizon:",
            QLabel(str(meta.get("horizon", "n/a"))),
        )
        layout.addRow(
            "Batch size:",
            QLabel(str(meta.get("batch_size", "n/a"))),
        )

        # Global point metrics
        if point:
            layout.addRow(
                "R² (overall):", QLabel(self._fmt(point.get("r2")))
            )
            layout.addRow(
                "MAE (overall):", QLabel(self._fmt(point.get("mae")))
            )
            layout.addRow(
                "MSE (overall):", QLabel(self._fmt(point.get("mse")))
            )

        # Interval metrics
        if interval:
            layout.addRow(
                "Coverage80 (uncal.):",
                QLabel(self._fmt(interval.get("coverage80_uncalibrated"))),
            )
            layout.addRow(
                "Coverage80 (cal.):",
                QLabel(self._fmt(interval.get("coverage80_calibrated"))),
            )
            layout.addRow(
                "Sharpness80 (uncal.):",
                QLabel(self._fmt(interval.get("sharpness80_uncalibrated"))),
            )
            layout.addRow(
                "Sharpness80 (cal.):",
                QLabel(self._fmt(interval.get("sharpness80_calibrated"))),
            )

        # Physics diagnostics
        if phys:
            layout.addRow(
                "ε_prior:",
                QLabel(self._fmt(phys.get("epsilon_prior"))),
            )
            layout.addRow(
                "ε_cons:",
                QLabel(self._fmt(phys.get("epsilon_cons"))),
            )

        return w

    def _build_metrics_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)

        rows = []

        metrics_eval = self._data.get("metrics_evaluate", {})
        point = self._data.get("point_metrics", {})
        phys = self._data.get("physics_diagnostics", {})

        for k, v in metrics_eval.items():
            rows.append((f"evaluate.{k}", v))
        for k, v in point.items():
            rows.append((f"point.{k}", v))
        for k, v in phys.items():
            rows.append((f"physics.{k}", v))

        table = QTableWidget(len(rows), 2, w)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)

        for r, (name, value) in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(name))
            table.setItem(r, 1, QTableWidgetItem(self._fmt(value)))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        return w

    def _build_per_horizon_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)

        per_h = self._data.get("per_horizon")
        if not per_h:
            layout.addWidget(QLabel("No per-horizon metrics available."))
            return w

        mae = per_h.get("mae", {})
        r2 = per_h.get("r2", {})

        horizons = sorted(set(list(mae.keys()) + list(r2.keys())))

        table = QTableWidget(len(horizons), 3, w)
        table.setHorizontalHeaderLabels(["Horizon", "MAE", "R²"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)

        for row, h in enumerate(horizons):
            table.setItem(row, 0, QTableWidgetItem(h))
            table.setItem(row, 1, QTableWidgetItem(self._fmt(mae.get(h))))
            table.setItem(row, 2, QTableWidgetItem(self._fmt(r2.get(h))))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        return w

    def _build_json_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)

        editor = QPlainTextEdit(w)
        editor.setReadOnly(True)
        editor.setPlainText(json.dumps(self._data, indent=2, sort_keys=True))
        layout.addWidget(editor)

        return w

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _on_save_as(self) -> None:
        default_name = os.path.basename(self._json_path)
        start_dir = os.path.dirname(self._json_path)
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Save metrics JSON as…",
            os.path.join(start_dir, default_name),
            "JSON files (*.json);;All files (*)",
        )
        if not target:
            return

        try:
            shutil.copy2(self._json_path, target)
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(
                self,
                "Save failed",
                f"Could not copy metrics JSON to:\n{target}\n\n{exc}",
            )

    def _on_open_folder(self) -> None:
        folder = os.path.dirname(self._json_path)
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    # ------------------------------------------------------------------
    # Small utils
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            # 4 significant digits
            return f"{value:.4g}"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        return str(value)
