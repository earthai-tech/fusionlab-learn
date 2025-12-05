# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QDialogButtonBox,
    QGroupBox,
    QPushButton,
    QWidget,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
)

from ..config import GeoPriorConfig, Stage1Summary
from ..device_options import DeviceOptions, DeviceOptionsWidget
from ..jobs import (
    latest_jobs_for_root,
    discover_tune_jobs_for_root,
    TrainJobSpec,
)
from .stage1_dialogs import Stage1DetailsDialog

# ----------------------------------------------------------------------
# Lightweight "what to tune" spec
# ----------------------------------------------------------------------
@dataclass
class TuneJobSpec:
    """
    Describe which city / Stage-1 setup should be used for tuning.

    Attributes
    ----------
    stage1 : Stage1Summary
        Stage-1 summary (city, manifest, shapes, etc.).
    """

    stage1: Stage1Summary


# ----------------------------------------------------------------------
# QuickTuneDialog – simple picker by city
# ----------------------------------------------------------------------
class QuickTuneDialog(QDialog):
    """
    Simple dialog that lists existing cities (Stage-1 runs) and lets the
    user pick one for a new tuning run.

    Similar to :class:`QuickTrainDialog`, but there is no 'scratch vs
    reuse' choice for Stage-1: tuning always reuses the existing
    Stage-1 artifacts.
    """

    def __init__(self, results_root: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select city for tuning")
        self._results_root = Path(results_root)

        # We reuse TrainJobSpec so discovery logic stays in jobs.py
        self._jobs: List[TrainJobSpec] = list(
            latest_jobs_for_root(self._results_root)
        )

        # Map city -> latest TuneRunInfo
        self._tune_infos = discover_tune_jobs_for_root(self._results_root)

        vbox = QVBoxLayout(self)
        if not self._jobs:
            vbox.addWidget(
                QLabel(
                    "No Stage-1 cities found in:\n"
                    f"{self._results_root}"
                )
            )
            btns = QDialogButtonBox(QDialogButtonBox.Close)
            btns.rejected.connect(self.reject)
            vbox.addWidget(btns)
            self._selected: Optional[TuneJobSpec] = None
            return

        label = QLabel(
            "Select which city you want to tune.\n"
            "Each entry corresponds to a Stage-1 run."
        )
        label.setWordWrap(True)
        vbox.addWidget(label)

        self._list = QListWidget()

        for job in self._jobs:
            summary = job.stage1_summary
            # In practice this should never be None for tuning,
            # but guard anyway.
            if summary is None:
                text = f"{job.city} — (no Stage-1 manifest found)"
                self._list.addItem(QListWidgetItem(text))
                continue

            city = summary.city
            txt = (
                f"{city}  —  {summary.timestamp}  "
                f"(T={summary.time_steps}, H={summary.horizon_years})"
            )

            tinfo = self._tune_infos.get(city)
            if tinfo is not None:
                ts = tinfo.summary.get("timestamp") if tinfo.summary else None
                if ts:
                    txt += f" — last tuning: {ts}"
                else:
                    txt += " — tuned (summary missing)"

            self._list.addItem(QListWidgetItem(txt))

        vbox.addWidget(self._list)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        vbox.addWidget(btns)

        self._selected: Optional[TuneJobSpec] = None

    def _on_accept(self) -> None:
        idx = self._list.currentRow()
        if idx < 0:
            self.reject()
            return

        job = self._jobs[idx]
        summary = job.stage1_summary
        if summary is None:
            # Should not happen for valid Stage-1 runs; just bail.
            self.reject()
            return

        self._selected = TuneJobSpec(stage1=summary)
        self.accept()

    @classmethod
    def run(
        cls, results_root: Path, parent=None
    ) -> Tuple[bool, Optional[TuneJobSpec]]:
        dlg = cls(results_root=results_root, parent=parent)
        ok = dlg.exec_() == QDialog.Accepted
        return ok, dlg._selected


# ----------------------------------------------------------------------
# TuneOptionsDialog – advanced view / root chooser
# ----------------------------------------------------------------------
class TuneOptionsDialog(QDialog):
    """
    Advanced options for the 'Tune' tab.

    Responsibilities
    ----------------
    - Let user change the results root (folder where Stage-1 city
      directories live).
    - List detected Stage-1 cities and show for each:
        [Details]  [Run tuning]
    - Details: open :class:`Stage1DetailsDialog` on the manifest for
      that city.
    - Run tuning: queue a :class:`TuneJobSpec` for that city.

    Notes
    -----
    Unlike training, this dialog never cleans Stage-1 directories and
    there is no 'scratch vs reuse' choice: tuning always reuses the
    existing Stage-1 artifacts.
    """

    def __init__(
        self,
        cfg: GeoPriorConfig,
        gui_runs_root: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tune options")
        self._cfg = cfg
        self._gui_runs_root = Path(gui_runs_root)
        self._queued_job: Optional[TuneJobSpec] = None

        vbox = QVBoxLayout(self)

        # --- Results root selector ---
        root_box = QGroupBox("Results root")
        root_layout = QHBoxLayout(root_box)
        self._le_root = QLineEdit(str(self._gui_runs_root))
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._on_browse_root)
        root_layout.addWidget(self._le_root, stretch=1)
        root_layout.addWidget(btn_browse)
        vbox.addWidget(root_box)

        # --- Processor / devices ----------------------------------------
        base_cfg = getattr(self._cfg, "_base_cfg", {}) or {}
        dev_opts = DeviceOptions.from_cfg(base_cfg)

        self.device_widget = DeviceOptionsWidget(
            initial=dev_opts,
            parent=self,
        )
        vbox.addWidget(self.device_widget)
        
        # --- Stage-1 / city list ---
        self._city_box = QGroupBox("Available Stage-1 cities")
        self._city_layout = QVBoxLayout(self._city_box)
        vbox.addWidget(self._city_box)

        self._populate_cities()

        # --- Buttons ---
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        vbox.addWidget(btns)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    @property
    def new_root(self) -> Path:
        return Path(self._le_root.text()).expanduser()

    @property
    def queued_job(self) -> Optional[TuneJobSpec]:
        return self._queued_job

    def get_device_overrides(self) -> dict:
        """Return NAT-style device overrides from the devices widget."""
        if not hasattr(self, "device_widget"):
            return {}
        return self.device_widget.to_cfg_overrides()

    def _on_browse_root(self) -> None:
        new_dir = QFileDialog.getExistingDirectory(
            self,
            "Select results root for GeoPrior runs",
            str(self.new_root),
        )
        if new_dir:
            self._le_root.setText(new_dir)
            # repopulate list for new root
            self._clear_cities()
            self._populate_cities()

    def _clear_cities(self) -> None:
        while self._city_layout.count():
            item = self._city_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _populate_cities(self) -> None:
        root = self.new_root

        # Again we reuse TrainJobSpec discovery:
        jobs: List[TrainJobSpec] = list(latest_jobs_for_root(root))
        tune_infos = discover_tune_jobs_for_root(root)

        if not jobs:
            label = QLabel(
                "No Stage-1 city runs found yet in:\n"
                f"{root}"
            )
            label.setWordWrap(True)
            self._city_layout.addWidget(label)
            return

        for job in jobs:
            summary = job.stage1_summary
            if summary is None:
                # Should not happen, but skip just in case.
                continue

            city = summary.city

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            # Left: city label + short info
            txt = (
                f"{city} — Stage-1 @ {summary.timestamp} "
                f"(T={summary.time_steps}, "
                f"H={summary.horizon_years})"
            )

            tinfo = tune_infos.get(city)
            if tinfo is not None:
                ts = tinfo.summary.get("timestamp") if tinfo.summary else None
                if ts:
                    txt += f" — last tuning: {ts}"
                else:
                    txt += " — tuned (summary missing)"

            label = QLabel(txt)
            row_layout.addWidget(label, stretch=1)

            # Middle: Details button (Stage-1 manifest)
            btn_details = QPushButton("Details…")
            btn_details.clicked.connect(
                # Capture the specific summary for this row
                lambda _, s=summary: self._show_stage1_details(s)
            )
            row_layout.addWidget(btn_details)

            # Right: Run tuning button
            btn_run = QPushButton("Run")
            btn_run.clicked.connect(
                lambda _, s=summary: self._run_tuning_for(s)
            )
            row_layout.addWidget(btn_run)

            self._city_layout.addWidget(row_widget)

        self._city_layout.addStretch(1)

    def _show_stage1_details(self, summary: Stage1Summary) -> None:
        dlg = Stage1DetailsDialog(summary=summary, parent=self)
        dlg.exec_()

    def _run_tuning_for(self, summary: Stage1Summary) -> None:
        self._queued_job = TuneJobSpec(stage1=summary)
        self.accept()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    @classmethod
    def run(
        cls,
        cfg: GeoPriorConfig,
        gui_runs_root: Path,
        parent=None,
    ) -> Tuple[bool, Path, Optional[TuneJobSpec], dict]:
        dlg = cls(cfg=cfg, gui_runs_root=gui_runs_root, parent=parent)
        ok = dlg.exec_() == QDialog.Accepted
        dev_overrides: dict = {}
        if ok:
            dev_overrides = dlg.get_device_overrides()
        return ok, dlg.new_root, dlg.queued_job, dev_overrides
