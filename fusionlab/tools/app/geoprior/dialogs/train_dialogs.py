# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt

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

from ..config import GeoPriorConfig
from ..device_options import DeviceOptions, DeviceOptionsWidget
from ..jobs import TrainJobSpec, latest_jobs_for_root
from .stage1_dialogs import Stage1DetailsDialog

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

        # --- Dangerous cleanup option ----------------------------------
        self.chk_clean = QCheckBox(
            "Clean Stage-1 run dir before running Stage-1",
            grp_stage1,
        )
        self.chk_clean.setChecked(
            bool(getattr(self.geo_cfg, "clean_stage1_dir", False))
        )
        self.chk_clean.toggled.connect(self._on_clean_toggled)
        g1.addWidget(self.chk_clean)

        # Subtle but always-visible warning (danger zone feel)
        warn_lbl = QLabel(
            "<b>Danger zone:</b> cleaning removes the whole "
            "<code>&lt;city&gt;_GeoPriorSubsNet_stage1/</code> folder "
            "for that city (artifacts, train_*, tuning/, inference/)."
        )
        warn_lbl.setTextFormat(Qt.RichText)
        warn_lbl.setWordWrap(True)
        warn_lbl.setStyleSheet("color: #b03030; font-size: 9pt;")
        g1.addWidget(warn_lbl)

        # --- Smart Stage-1 behaviour flags ------------------------------
        self.chk_auto_reuse = QCheckBox(
            "Auto-reuse latest complete Stage-1 when config matches "
            "(no prompt)",
            grp_stage1,
        )
        self.chk_auto_reuse.setChecked(
            bool(
                getattr(
                    self.geo_cfg,
                    "stage1_auto_reuse_if_match",
                    True,
                )
            )
        )
        g1.addWidget(self.chk_auto_reuse)

        self.chk_force_rebuild_mismatch = QCheckBox(
            "Force Stage-1 rebuild when no compatible run exists "
            "(config mismatch)",
            grp_stage1,
        )
        self.chk_force_rebuild_mismatch.setChecked(
            bool(
                getattr(
                    self.geo_cfg,
                    "stage1_force_rebuild_if_mismatch",
                    True,
                )
            )
        )
        g1.addWidget(self.chk_force_rebuild_mismatch)

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
        
        # ---- Section 3: Processor / devices ------------------------------
        base_cfg = getattr(self.geo_cfg, "_base_cfg", {}) or {}
        dev_opts = DeviceOptions.from_cfg(base_cfg)

        self.device_widget = DeviceOptionsWidget(
            initial=dev_opts,
            parent=self,
        )
        layout.addWidget(self.device_widget)
        
        # ---- Section 4: Existing cities / jobs ---------------------------
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

    @classmethod
    def run(
        cls,
        parent,
        geo_cfg: GeoPriorConfig,
        results_root: Path,
    ) -> tuple[bool, Path, Optional[TrainJobSpec], dict]:
        """
        Open the dialog and return
        (accepted, new_root, job, device_overrides).
        """
        dlg = cls(parent, geo_cfg, results_root)
        ok = dlg.exec_() == QDialog.Accepted
        dev_overrides: dict = {}
        if ok:
            dev_overrides = dlg.get_device_overrides()
        return ok, dlg.results_root, dlg.selected_job, dev_overrides

    # -- internals --------------------------------------------------------
    def _on_clean_toggled(self, checked: bool) -> None:
        """
        Guard rail for the 'Clean Stage-1' option.

        If the user enables it, show a destructive-action warning
        summarising what will be deleted and give them a chance
        to cancel.
        """
        if not checked:
            # User unticked it – nothing to do.
            return

        # Collect existing Stage-1 run dirs under this results root,
        # so the dialog can show concrete examples.
        run_dirs = []
        for job in getattr(self, "jobs", []) or []:
            s = job.stage1_summary
            if s is None:
                continue
            # Stage1Summary.run_dir is the <city>_GeoPriorSubsNet_stage1 dir
            run_dirs.append(str(s.run_dir))

        if run_dirs:
            max_show = 4
            listed = "".join(f"<br/>• {path}" for path in run_dirs[:max_show])
            if len(run_dirs) > max_show:
                listed += (
                    f"<br/>… ({len(run_dirs) - max_show} more "
                    "Stage-1 directories)"
                )
            extra_html = (
                "<p>Existing Stage-1 directories detected in this "
                f"results root include:{listed}</p>"
            )
        else:
            extra_html = (
                "<p>Currently no Stage-1 manifests were detected in this "
                "results root, but any existing "
                "<code>&lt;city&gt;_GeoPriorSubsNet_stage1/</code> "
                "directory for the next city you train will still be "
                "removed.</p>"
            )

        msg_html = (
            "<p><span style='color:#b03030; font-weight:bold;'>"
            "Danger zone – Stage-1 cleanup</span></p>"
            "<p>You are about to enable "
            "<b>cleaning of the Stage-1 run directory</b>.</p>"
            "<p>For the next training run, this will recursively delete "
            "the entire "
            "<code>&lt;city&gt;_GeoPriorSubsNet_stage1/</code> folder "
            "for that city, including:</p>"
            "<ul>"
            "<li><code>artifacts/</code> "
            "(NPZ sequences, <code>manifest.json</code>)</li>"
            "<li><code>train_*/</code> (all past training runs)</li>"
            "<li><code>tuning/</code> (all tuning runs)</li>"
            "<li><code>inference/</code> "
            "(diagnostic results and summaries)</li>"
            "</ul>"
            f"{extra_html}"
            "<p><b>This cannot be undone.</b> Use this only if you "
            "explicitly want to discard <b>all</b> previous results "
            "for that city.</p>"
            "<p>Do you still want to keep "
            "<b>'Clean Stage-1 run dir'</b> enabled?</p>"
        )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Confirm Stage-1 cleanup")
        box.setTextFormat(Qt.RichText)
        box.setText(msg_html)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)

        reply = box.exec_()

        if reply != QMessageBox.Yes:
            # User backed out: revert the checkbox without retriggering us.
            self.chk_clean.blockSignals(True)
            self.chk_clean.setChecked(False)
            self.chk_clean.blockSignals(False)

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
        # Apply Stage-1 behaviour flags back into config
        self.geo_cfg.clean_stage1_dir = self.chk_clean.isChecked()
        self.geo_cfg.stage1_auto_reuse_if_match = (
            self.chk_auto_reuse.isChecked()
        )
        self.geo_cfg.stage1_force_rebuild_if_mismatch = (
            self.chk_force_rebuild_mismatch.isChecked()
        )

        # Ensure directory exists (unchanged)
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

    def get_device_overrides(self) -> dict:
        """
        NAT-style overrides based on the Processor / devices block.

        Returns
        -------
        dict
            Something like:
            {
                "TF_DEVICE_MODE": "gpu",
                "TF_INTRA_THREADS": 8,
                "TF_INTER_THREADS": 4,
                "TF_GPU_ALLOW_GROWTH": True,
                "TF_GPU_MEMORY_LIMIT_MB": 12000,
            }
        """
        if not hasattr(self, "device_widget"):
            return {}
        return self.device_widget.to_cfg_overrides()


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

