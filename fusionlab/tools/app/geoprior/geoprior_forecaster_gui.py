# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
GeoPrior subsidence GUI.

Current features
----------------
- City / dataset field (optionally populated from a CSV picker).
- Single "Train" button that runs Stage-1 then Stage-2 training
  in sequence.
- Log panel + progress bar.
- A QTabWidget with 4 tabs:
  [Train, Tune, Inference, Transferability].

Only the Train tab is functionally wired for now. Other tabs are
simple placeholders and will be populated step by step.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QMessageBox,
    QTabWidget,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QGridLayout,
    QFileDialog,
    QDialog,
)


# from ....utils.nat_utils import get_default_runs_root
from ..qt_utils import auto_set_ui_fonts, auto_resize_window
from .threads import Stage1Thread, TrainingThread
from .geoprior_config import GeoPriorConfig
from .feature_dialog import FeatureConfigDialog
from .dialog import CsvEditDialog
from .styles import TAB_STYLES, LOG_STYLES, FLAB_STYLE_SHEET


PRIMARY = "#2E3191"


class GeoPriorForecaster(QMainWindow):
    """GUI wrapper for GeoPrior subsidence runs."""

    log_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(float)

    def __init__(self, *, theme: str = "fusionlab") -> None:
        super().__init__()
        self.theme = theme or "fusionlab"

        self.stage1_thread: Stage1Thread | None = None
        self.train_thread: TrainingThread | None = None

        # Selected CSV (optional) + config overrides passed to threads
        self.csv_path: Path | None = None
        self._cfg_overrides: Dict[str, Any] = {}

        # Central config object (defaults loaded from nat.com/config.py)
        self.geo_cfg = GeoPriorConfig.from_nat_config()
        
        # Dedicated root so GUI runs don't mix with CLI results
        # self.gui_runs_root = Path(get_default_runs_root())
        # --- NEW: dedicated runs root for GUI ---
        self.gui_runs_root = Path.home() / ".fusionlab_runs"
        self.gui_runs_root.mkdir(parents=True, exist_ok=True)
        
        self._build_ui()
        self._connect_signals()
        self._set_window_props()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _set_window_props(self) -> None:
        self.setWindowTitle("Fusionlab – GeoPrior Forecaster")
        auto_resize_window(self, base_size=(820, 520))

        ico_dir = Path(__file__).parent
        ico = ico_dir / "fusionlab_learn_logo.ico"
        if ico.exists():
            self.setWindowIcon(QIcon(str(ico)))

        # Apply Fusionlab stylesheet (tabs / cards / log)
        self.setStyleSheet(FLAB_STYLE_SHEET + TAB_STYLES + LOG_STYLES)

    def _make_card(
        self,
        title: str,
    ) -> tuple[QTabWidget, QVBoxLayout]:
        """
        Create a framed 'card' with a bold title and
        return (frame, inner_layout).
        """
        frame = QWidget()
        frame.setObjectName("card")
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(8, 6, 8, 8)

        lbl = QLabel(title)
        lbl.setObjectName("cardTitle")
        vbox.addWidget(lbl)

        return frame, vbox

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)

        # --- Top row: [Select CSV…] [City / Dataset] [Train] [Quit] ---
        top = QHBoxLayout()

        self.select_csv_btn = QPushButton("Select CSV…")
        top.addWidget(self.select_csv_btn)

        top.addWidget(QLabel("City / Dataset:"))

        self.city_edit = QLineEdit()
        self.city_edit.setPlaceholderText("nansha")
        top.addWidget(self.city_edit, 1)

        self.train_btn = QPushButton("Train")
        top.addWidget(self.train_btn)

        self.quit_btn = QPushButton("Quit")
        top.addWidget(self.quit_btn)

        layout.addLayout(top)

        # --- Tabs row: Train / Tune / Inference / Transferability ---
        self.tabs = QTabWidget()
        self._init_tabs()
        layout.addWidget(self.tabs, 1)

        # --- Status line ---
        self.status_label = QLabel("? Idle")
        self.status_label.setStyleSheet(f"color:{PRIMARY};")
        layout.addWidget(self.status_label)

        # --- Log widget ---
        self.log_widget = QPlainTextEdit()
        self.log_widget.setObjectName("logWidget")
        self.log_widget.setReadOnly(True)
        self.log_widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.log_widget.setMinimumHeight(200)
        layout.addWidget(self.log_widget, 1)

        # --- Progress row ---
        prog = QHBoxLayout()

        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.progress_label.setSizePolicy(
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )
        prog.addWidget(self.progress_label, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(18)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        prog.addWidget(self.progress_bar, 1)

        self.percent_label = QLabel("0 %")
        self.percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.percent_label.setSizePolicy(
            QSizePolicy.Minimum,
            QSizePolicy.Fixed,
        )
        prog.addWidget(self.percent_label, 0)

        layout.addLayout(prog)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    def _init_tabs(self) -> None:
        """Create tabs. Only Train is functional for now."""
        cfg = self.geo_cfg

        # ==================================================
        # Train tab
        # ==================================================
        train_tab = QWidget()
        t_layout = QVBoxLayout(train_tab)
        t_layout.setContentsMargins(6, 6, 6, 6)
        t_layout.setSpacing(8)

        info = QLabel(
            "Train tab – runs Stage-1 (prepare) and Stage-2 "
            "(GeoPrior training) for the selected city."
        )
        info.setWordWrap(True)
        t_layout.addWidget(info)

        # --- three cards in a horizontal row ---
        cards_row = QHBoxLayout()
        cards_row.setSpacing(10)

        # 1) Temporal window card
        temp_card, temp_box = self._make_card("Temporal window")

        self.sp_train_end = QSpinBox()
        self.sp_train_end.setRange(2000, 2100)
        self.sp_train_end.setValue(cfg.train_end_year)

        self.sp_forecast_start = QSpinBox()
        self.sp_forecast_start.setRange(2000, 2100)
        self.sp_forecast_start.setValue(cfg.forecast_start_year)

        self.sp_forecast_horizon = QSpinBox()
        self.sp_forecast_horizon.setRange(1, 50)
        self.sp_forecast_horizon.setValue(cfg.forecast_horizon_years)

        self.sp_time_steps = QSpinBox()
        self.sp_time_steps.setRange(1, 50)
        self.sp_time_steps.setValue(cfg.time_steps)

        grid_t = QGridLayout()
        grid_t.addWidget(QLabel("Train end year:"), 0, 0)
        grid_t.addWidget(self.sp_train_end, 0, 1)
        grid_t.addWidget(QLabel("Forecast start year:"), 1, 0)
        grid_t.addWidget(self.sp_forecast_start, 1, 1)
        grid_t.addWidget(QLabel("Forecast horizon (years):"), 2, 0)
        grid_t.addWidget(self.sp_forecast_horizon, 2, 1)
        grid_t.addWidget(QLabel("Time steps (look-back):"), 3, 0)
        grid_t.addWidget(self.sp_time_steps, 3, 1)
        temp_box.addLayout(grid_t)

        cards_row.addWidget(temp_card, 1)

        # 2) Training card
        train_card, train_box = self._make_card("Training")

        self.sp_epochs = QSpinBox()
        self.sp_epochs.setRange(1, 5000)
        self.sp_epochs.setValue(cfg.epochs)

        self.sp_batch_size = QSpinBox()
        self.sp_batch_size.setRange(1, 1024)
        self.sp_batch_size.setValue(cfg.batch_size)

        self.sp_lr = QDoubleSpinBox()
        self.sp_lr.setDecimals(6)
        self.sp_lr.setRange(1e-6, 1e-2)
        self.sp_lr.setSingleStep(1e-5)
        self.sp_lr.setValue(cfg.learning_rate)

        grid_tr = QGridLayout()
        grid_tr.addWidget(QLabel("Epochs:"), 0, 0)
        grid_tr.addWidget(self.sp_epochs, 0, 1)
        grid_tr.addWidget(QLabel("Batch size:"), 1, 0)
        grid_tr.addWidget(self.sp_batch_size, 1, 1)
        grid_tr.addWidget(QLabel("Learning rate:"), 2, 0)
        grid_tr.addWidget(self.sp_lr, 2, 1)
        train_box.addLayout(grid_tr)

        cards_row.addWidget(train_card, 1)

        # 3) Physics weights card
        phys_card, phys_box = self._make_card("Physics weights")

        self.cmb_pde_mode = QComboBox()
        self.cmb_pde_mode.addItems(
            ["off", "both", "consolidation", "gw_flow"]
        )
        idx = self.cmb_pde_mode.findText(cfg.pde_mode)
        if idx < 0:
            idx = 0
        self.cmb_pde_mode.setCurrentIndex(idx)

        def make_lambda_spin(val: float) -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setDecimals(4)
            sb.setRange(0.0, 10.0)
            sb.setSingleStep(0.005)
            sb.setValue(val)
            return sb

        self.sb_lcons = make_lambda_spin(cfg.lambda_cons)
        self.sb_lgw = make_lambda_spin(cfg.lambda_gw)
        self.sb_lprior = make_lambda_spin(cfg.lambda_prior)
        self.sb_lsmooth = make_lambda_spin(cfg.lambda_smooth)
        self.sb_lmv = make_lambda_spin(cfg.lambda_mv)

        grid_p = QGridLayout()
        grid_p.addWidget(QLabel("PDE mode:"), 0, 0)
        grid_p.addWidget(self.cmb_pde_mode, 0, 1)
        grid_p.addWidget(QLabel("λ consolidation:"), 1, 0)
        grid_p.addWidget(self.sb_lcons, 1, 1)
        grid_p.addWidget(QLabel("λ GW flow:"), 2, 0)
        grid_p.addWidget(self.sb_lgw, 2, 1)
        grid_p.addWidget(QLabel("λ prior:"), 3, 0)
        grid_p.addWidget(self.sb_lprior, 3, 1)
        grid_p.addWidget(QLabel("λ smooth:"), 4, 0)
        grid_p.addWidget(self.sb_lsmooth, 4, 1)
        grid_p.addWidget(QLabel("λ mᵥ:"), 5, 0)
        grid_p.addWidget(self.sb_lmv, 5, 1)
        phys_box.addLayout(grid_p)

        cards_row.addWidget(phys_card, 1)

        t_layout.addLayout(cards_row)

        # --- bottom flags ---
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        self.chk_clean_stage1 = QCheckBox("Clean Stage-1 run dir")
        self.chk_clean_stage1.setChecked(cfg.clean_stage1_dir)
        bottom_row.addWidget(self.chk_clean_stage1)

        self.chk_eval_training = QCheckBox(
            "Evaluate training metrics",
        )
        self.chk_eval_training.setChecked(
            cfg.evaluate_training,
        )
        bottom_row.addWidget(self.chk_eval_training)

        self.btn_features = QPushButton("Feature config…")
        bottom_row.addWidget(self.btn_features)

        bottom_row.addStretch(1)

        bottom_row.addStretch(1)
        t_layout.addLayout(bottom_row)
        t_layout.addStretch(1)

        # ==================================================
        # Placeholder tabs
        # ==================================================
        tune_tab = QWidget()
        u_layout = QVBoxLayout(tune_tab)
        u_label = QLabel(
            "Tune tab – hyperparameter search.\n"
            "To be implemented."
        )
        u_label.setWordWrap(True)
        u_layout.addWidget(u_label)
        u_layout.addStretch(1)

        infer_tab = QWidget()
        i_layout = QVBoxLayout(infer_tab)
        i_label = QLabel(
            "Inference tab – evaluation and forecasting.\n"
            "To be implemented."
        )
        i_label.setWordWrap(True)
        i_layout.addWidget(i_label)
        i_layout.addStretch(1)

        xfer_tab = QWidget()
        x_layout = QVBoxLayout(xfer_tab)
        x_label = QLabel(
            "Transferability tab – cross-city transfer matrix.\n"
            "To be implemented."
        )
        x_label.setWordWrap(True)
        x_layout.addWidget(x_label)
        x_layout.addStretch(1)

        self.tabs.addTab(train_tab, "Train")
        self.tabs.addTab(tune_tab, "Tune")
        self.tabs.addTab(infer_tab, "Inference")
        self.tabs.addTab(xfer_tab, "Transferability")

    # ------------------------------------------------------------------
    # Config synchronisation
    # ------------------------------------------------------------------
    def _sync_config_from_ui(self) -> None:
        """Copy widgets' values into self.geo_cfg."""
        cfg = self.geo_cfg
        cfg.train_end_year = self.sp_train_end.value()
        cfg.forecast_start_year = self.sp_forecast_start.value()
        cfg.forecast_horizon_years = self.sp_forecast_horizon.value()
        cfg.time_steps = self.sp_time_steps.value()

        cfg.epochs = self.sp_epochs.value()
        cfg.batch_size = self.sp_batch_size.value()
        cfg.learning_rate = self.sp_lr.value()

        cfg.pde_mode = self.cmb_pde_mode.currentText()
        cfg.lambda_cons = self.sb_lcons.value()
        cfg.lambda_gw = self.sb_lgw.value()
        cfg.lambda_prior = self.sb_lprior.value()
        cfg.lambda_smooth = self.sb_lsmooth.value()
        cfg.lambda_mv = self.sb_lmv.value()

        cfg.clean_stage1_dir = self.chk_clean_stage1.isChecked()
        cfg.evaluate_training = self.chk_eval_training.isChecked()

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.select_csv_btn.clicked.connect(self._on_select_csv)
        self.train_btn.clicked.connect(self._on_train_clicked)
        self.quit_btn.clicked.connect(self.close)

        self.log_updated.connect(self._append_log)
        self.status_updated.connect(self.status_label.setText)
        self.progress_updated.connect(self._update_progress)
        
        self.btn_features.clicked.connect(
            self._on_feature_config,
        )

    # ------------------------------------------------------------------
    # Logging / progress helpers
    # ------------------------------------------------------------------
    @pyqtSlot(str)
    def _append_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_widget.appendPlainText(f"[{ts}] {msg}")

    @pyqtSlot(float)
    def _update_progress(self, value: float) -> None:
        if value is None:
            value = 0.0
        value = max(0.0, min(1.0, float(value)))
        pct = int(round(100.0 * value))
        self.progress_bar.setValue(pct)
        self.percent_label.setText(f"{pct:3d} %")

    @pyqtSlot(float, str)
    def _on_thread_progress(self, value: float, message: str) -> None:
        self._update_progress(value)
        if message:
            self.progress_label.setText(message)
            
    @pyqtSlot()
    def _on_feature_config(self) -> None:
        if self.csv_path is None:
            QMessageBox.information(
                self,
                "CSV required",
                "Please select a CSV file first so "
                "that columns can be listed in the "
                "feature dialog.",
            )
            return

        base_cfg = self.geo_cfg._base_cfg or {}

        dlg = FeatureConfigDialog(
            csv_path=self.csv_path,
            base_cfg=base_cfg,
            current_overrides=self.geo_cfg.feature_overrides,
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        self.geo_cfg.feature_overrides = dlg.get_overrides()

        changed = ", ".join(
            sorted(self.geo_cfg.feature_overrides.keys()),
        )
        if not changed:
            changed = "none"

        self.log_updated.emit(
            "Feature configuration updated "
            f"(keys: {changed}).",
        )

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    @pyqtSlot()
    def _on_select_csv(self) -> None:
        """
        Open a CSV file, optionally edit it, and update
        City/Dataset + internal csv_path.
        """
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV file",
            "",
            "CSV Files (*.csv)",
        )
        if not path_str:
            return

        try:
            dlg = CsvEditDialog(path_str, self)
            if dlg.exec_() != QDialog.Accepted:
                return

            self.csv_path = Path(path_str)

            # Auto-populate City/Dataset if empty
            if not self.city_edit.text().strip():
                self.city_edit.setText(self.csv_path.stem)

            self.log_updated.emit(f"CSV selected: {self.csv_path}")
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(
                self,
                "CSV error",
                f"Failed to open CSV:\n{exc}",
            )
            self.csv_path = None

    @pyqtSlot()
    def _on_train_clicked(self) -> None:
        if self.stage1_thread or self.train_thread:
            QMessageBox.information(
                self,
                "Busy",
                "A workflow is already running.",
            )
            return

        city = self.city_edit.text().strip() or "nansha"

        # 1) Sync widgets -> GeoPriorConfig
        self._sync_config_from_ui()

        # 2) Convert to NAT-style overrides
        cfg_overrides = self.geo_cfg.to_cfg_overrides()

        # 2b) Force GUI runs under <repo_root>/.fusionlab_runs
        if getattr(self, "gui_runs_root", None) is not None:
            # Only set if not already forced by the caller
            cfg_overrides.setdefault(
                "BASE_OUTPUT_DIR",
                str(self.gui_runs_root),
            )

        # 3) Inject city / dataset information
        cfg_overrides["CITY_NAME"] = city
        if self.csv_path is not None:
            cfg_overrides["DATA_DIR"] = str(self.csv_path.parent)
            cfg_overrides["BIG_FN"] = self.csv_path.name

        self._cfg_overrides = cfg_overrides
        
        self.log_updated.emit(
            f"GUI runs root: {self.gui_runs_root}"
        )
        self.log_updated.emit(
            f"Start GeoPrior workflow for {city!r}."
        )
        self.status_updated.emit(f"Stage-1: preparing city={city}.")
        self._update_progress(0.0)
        self.train_btn.setEnabled(False)

        self._start_stage1(city)

    # ------------------------------------------------------------------
    # Thread orchestration
    # ------------------------------------------------------------------
    def _start_stage1(self, city: str) -> None:
        th = Stage1Thread(
            city=city,
            cfg_overrides=self._cfg_overrides,
            clean_run_dir=self.geo_cfg.clean_stage1_dir,
            parent=self,
        )
        self.stage1_thread = th

        th.log_updated.connect(self.log_updated.emit)
        th.status_updated.connect(self.status_updated.emit)
        th.progress_changed.connect(self._on_thread_progress)
        th.results_ready.connect(self._on_stage1_finished)
        th.error_occurred.connect(self._on_worker_error)

        th.start()

    @pyqtSlot(dict)
    def _on_stage1_finished(self, result: Dict[str, Any]) -> None:
        self.stage1_thread = None

        if not result:
            self.log_updated.emit(
                "Stage-1 finished with an empty result dict."
            )
            self.status_updated.emit("Stage-1 failed. See log.")
            self.train_btn.setEnabled(True)
            return

        manifest_path = result.get("manifest_path")
        if not manifest_path:
            self.log_updated.emit(
                "Stage-1 did not return a manifest_path."
            )
            self.status_updated.emit(
                "Cannot start training – no manifest."
            )
            self.train_btn.setEnabled(True)
            return

        self.log_updated.emit(
            "Stage-1 done. Manifest:\n"
            f"  {manifest_path}"
        )
        self.status_updated.emit("Stage-2: training GeoPrior model.")
        self._start_training(manifest_path)

    def _start_training(self, manifest_path: str) -> None:
        th = TrainingThread(
            manifest_path=manifest_path,
            cfg_overrides=self._cfg_overrides,
            evaluate_training=self.geo_cfg.evaluate_training,
            parent=self,
        )
        self.train_thread = th

        th.log_updated.connect(self.log_updated.emit)
        th.status_updated.connect(self.status_updated.emit)
        th.progress_changed.connect(self._on_thread_progress)
        th.training_finished.connect(self._on_training_finished)
        th.error_occurred.connect(self._on_worker_error)

        th.start()

    @pyqtSlot(dict)
    def _on_training_finished(self, result: Dict[str, Any]) -> None:
        self.train_thread = None

        if not result:
            self.log_updated.emit(
                "Training finished with an empty result dict."
            )
            self.status_updated.emit("Training failed. See log.")
            self.train_btn.setEnabled(True)
            return

        out_dir = result.get("run_output_path")
        ckpt = result.get("final_checkpoint")

        if out_dir:
            self.log_updated.emit(
                "Training artifacts in:\n"
                f"  {out_dir}"
            )
        if ckpt:
            self.log_updated.emit(
                "Final checkpoint:\n"
                f"  {ckpt}"
            )

        self.log_updated.emit("Training completed successfully.")
        self.status_updated.emit("Idle – training complete.")
        self._update_progress(1.0)
        self.train_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_worker_error(self, message: str) -> None:
        self.log_updated.emit(f"[ERROR] {message}")
        QMessageBox.critical(self, "Error", message)
        self.train_btn.setEnabled(True)
        self.stage1_thread = None
        self.train_thread = None
        self._update_progress(0.0)
        self.progress_label.setText("")


# ----------------------------------------------------------------------
# Entry point helper
# ----------------------------------------------------------------------
def launch_geoprior_gui(theme: str = "fusionlab") -> None:
    app = QApplication(sys.argv)
    auto_set_ui_fonts(app)

    gui = GeoPriorForecaster(theme=theme)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    launch_geoprior_gui()
