# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QDialogButtonBox,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QFrame,
    QCheckBox, 
    QFileDialog
)

from .jobs import TrainJobSpec, latest_jobs_for_root
from .stage1_dialog import Stage1DetailsDialog
from .geoprior_config import GeoPriorConfig


class TrainOptionsDialog(QDialog):
    """
    Advanced Training options:

    - Stage-1 behaviour (clean / reuse).
    - Base results root (~/.fusionlab_runs).
    - List of cities detected under that root, with
      'Details…' and 'Reuse & train' actions.
    """

    def __init__(
        self,
        parent,
        geo_cfg: GeoPriorConfig,
        results_root: Path,
    ) -> None:
        super().__init__(parent)
        self.geo_cfg = geo_cfg
        self.results_root = Path(results_root)
        self.selected_job: Optional[TrainJobSpec] = None
        self.jobs: List[TrainJobSpec] = []

        self.setWindowTitle("Training options")
        self.setModal(True)
        self.setMinimumWidth(640)

        layout = QVBoxLayout(self)

        # ---- Section 1: Stage-1 behaviour --------------------------------
        grp_stage1 = QGroupBox("Stage-1 behaviour", self)
        g1 = QVBoxLayout(grp_stage1)

        self.chk_clean = QCheckBox(
            "Clean Stage-1 run dir before running Stage-1",
            grp_stage1,
        )
        self.chk_clean.setChecked(
            bool(getattr(self.geo_cfg, "clean_stage1_dir", False))
        )
        g1.addWidget(self.chk_clean)

        grp_stage1.setLayout(g1)
        layout.addWidget(grp_stage1)

        # ---- Section 2: Results root -------------------------------------
        grp_root = QGroupBox("Results root", self)
        g2 = QVBoxLayout(grp_root)

        lbl_info = QLabel(
            "Base folder where Stage-1, training, tuning, "
            "inference and GUI logs are stored. GUI runs stay "
            "separate from CLI runs."
        )
        lbl_info.setWordWrap(True)
        g2.addWidget(lbl_info)

        row = QHBoxLayout()
        self.path_edit = QLineEdit(str(self.results_root))
        self.path_edit.setReadOnly(True)
        btn_browse = QPushButton("Browse…")
        row.addWidget(self.path_edit, 1)
        row.addWidget(btn_browse)
        g2.addLayout(row)

        btn_reset = QPushButton(
            "Reset to default (~/.fusionlab_runs)"
        )
        g2.addWidget(btn_reset)

        grp_root.setLayout(g2)
        layout.addWidget(grp_root)

        # ---- Section 3: Existing cities / jobs ---------------------------
        grp_jobs = QGroupBox(
            "Existing cities detected in this results root",
            self,
        )
        self.jobs_layout = QVBoxLayout(grp_jobs)
        grp_jobs.setLayout(self.jobs_layout)
        layout.addWidget(grp_jobs)

        # ---- Dialog buttons ----------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        layout.addWidget(buttons)

        # ---- Wire local handlers -----------------------------------------
        btn_browse.clicked.connect(self._on_browse_root)
        btn_reset.clicked.connect(self._on_reset_root)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        # Initial job discovery
        self._refresh_jobs()

    # -- public helper ----------------------------------------------------
    @classmethod
    def run(
        cls,
        parent,
        geo_cfg: GeoPriorConfig,
        results_root: Path,
    ) -> tuple[bool, Path, Optional[TrainJobSpec]]:
        """
        Open the dialog and return (accepted, new_root, job).
        """
        dlg = cls(parent, geo_cfg, results_root)
        ok = dlg.exec_() == QDialog.Accepted
        return ok, dlg.results_root, dlg.selected_job

    # -- internals --------------------------------------------------------
    def _on_browse_root(self) -> None:
        

        base = str(self.results_root)
        path = QFileDialog.getExistingDirectory(
            self,
            "Select base results directory",
            base,
        )
        if path:
            self.results_root = Path(path)
            self.path_edit.setText(path)
            self._refresh_jobs()

    def _on_reset_root(self) -> None:
        default_root = Path.home() / ".fusionlab_runs"
        self.results_root = default_root
        self.path_edit.setText(str(default_root))
        self._refresh_jobs()

    def _on_accept(self) -> None:
        # Apply stage-1 cleaning flag back into config
        self.geo_cfg.clean_stage1_dir = self.chk_clean.isChecked()

        # Ensure directory exists
        try:
            self.results_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Results root",
                f"Could not create directory:\n{exc}",
            )
            return

        self.accept()

    def _refresh_jobs(self) -> None:
        # clear layout
        while self.jobs_layout.count():
            item = self.jobs_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        # Discover jobs using the Stage-1 config view
        try:
            stage1_cfg = self.geo_cfg.to_stage1_config()
            self.jobs = latest_jobs_for_root(
                self.results_root,
                current_cfg=stage1_cfg,
            )
        except Exception:
            self.jobs = []

        if not self.jobs:
            lbl = QLabel(
                "No Stage-1 runs detected yet in this root."
            )
            lbl.setWordWrap(True)
            self.jobs_layout.addWidget(lbl)
            self.jobs_layout.addStretch(1)
            return

        for job in self.jobs:
            self._add_job_card(job)

        self.jobs_layout.addStretch(1)

    def _add_job_card(self, job: TrainJobSpec) -> None:
        s = job.stage1_summary

        frame = QFrame(self)
        frame.setFrameShape(QFrame.StyledPanel)
        vbox = QVBoxLayout(frame)

        title = QLabel(f"{s.city} — {s.timestamp}")
        title.setObjectName("cityJobTitle")
        vbox.addWidget(title)

        meta = QLabel(
            f"T={s.time_steps}, H={s.horizon_years}, "
            f"train/val={s.n_train}/{s.n_val}"
        )
        meta.setStyleSheet("font-size: 9pt;")
        vbox.addWidget(meta)

        status_bits = []
        status_bits.append("complete" if s.is_complete else "incomplete")
        status_bits.append(
            "config OK" if s.config_match else "config mismatch"
        )
        if s.diff_fields:
            status_bits.append(
                "diff: " + ", ".join(s.diff_fields)
            )

        status = QLabel(" • ".join(status_bits))
        status.setStyleSheet("font-size: 8pt; color: gray;")
        vbox.addWidget(status)

        row = QHBoxLayout()
        btn_details = QPushButton("Details…", frame)
        btn_run = QPushButton("Reuse & train", frame)
        row.addWidget(btn_details)
        row.addStretch(1)
        row.addWidget(btn_run)
        vbox.addLayout(row)

        btn_details.clicked.connect(
            lambda _=False, summ=s: Stage1DetailsDialog(
                summ, parent=self
            ).exec_()
        )
        btn_run.clicked.connect(
            lambda _=False, j=job: self._on_run_job_clicked(j)
        )

        self.jobs_layout.addWidget(frame)

    def _on_run_job_clicked(self, job: TrainJobSpec) -> None:
        """
        User clicked 'Reuse & train' for a city.
        We just store the job and accept the dialog;
        the main GUI will trigger the workflow.
        """
        self.selected_job = job
        self.accept()


# ----------------------------------------------------------------------
# QuickTrainDialog
# ----------------------------------------------------------------------

class QuickTrainDialog(QDialog):
    """
    Minimal dialog listing available jobs so the user can quickly
    select a city and reuse its Stage-1 run.
    """

    def __init__(
        self,
        jobs: List[TrainJobSpec],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.jobs = jobs
        self.selected_job: Optional[TrainJobSpec] = None

        self.setWindowTitle("Select city to train")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        lbl = QLabel(
            "Existing Stage-1 runs were detected in the current "
            "results root. Choose a city to train using the "
            "latest Stage-1 artifacts."
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self.list_widget = QListWidget(self)
        for job in self.jobs:
            s = job.stage1_summary
            text = (
                f"{s.city} — {s.timestamp}  "
                f"(T={s.time_steps}, H={s.horizon_years}, "
                f"train/val={s.n_train}/{s.n_val})"
            )
            if not s.config_match:
                text += " [config mismatch]"
            self.list_widget.addItem(text)

        if self.jobs:
            self.list_widget.setCurrentRow(0)

        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        idx = self.list_widget.currentRow()
        if idx < 0:
            QMessageBox.warning(
                self,
                "Run training",
                "Please select a city to train.",
            )
            return
        self.selected_job = self.jobs[idx]
        self.accept()

    @classmethod
    def choose_job(
        cls,
        parent,
        jobs: List[TrainJobSpec],
    ) -> Optional[TrainJobSpec]:
        if not jobs:
            return None
        dlg = cls(jobs=jobs, parent=parent)
        if dlg.exec_() == QDialog.Accepted:
            return dlg.selected_job
        return None

