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

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize, QPoint
from PyQt5.QtGui import (
    QIcon,
    QCloseEvent,
    QPixmap,
    QPainter,
    QColor,
    QPen,
)
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
    QToolButton, 
)

from ..smart_stage1 import find_stage1_for_city
from ....utils.nat_utils import get_default_runs_root
from ..ux_utils import (
    auto_set_ui_fonts,
    auto_resize_window,
    enable_qt_crash_handler,
)
from ..ux_splash import LoadingSplash
from ..view_signals import VIS_SIGNALS 
from ..gui_popups import ImagePreviewDialog 

from .threads import (
    Stage1Thread,
    TrainingThread,
    TuningThread,
    InferenceThread,
    XferMatrixThread,
    XferViewThread
)
from .geoprior_config import GeoPriorConfig, default_tuner_search_space
from .feature_dialog import FeatureConfigDialog
from .architecture_dialog import ArchitectureConfigDialog   
from .prob_dialog import ProbConfigDialog 
from .xfer_dialog import XferAdvancedDialog, XferResultsDialog 
from .xfer_view import latest_xfer_csv, latest_xfer_json
from .inference_dialogs import InferenceOptionsDialog 
from .stage1_dialogs import Stage1ChoiceDialog

from .results_dialog import GeoPriorResultsDialog
from .csv_dialog import CsvEditDialog
from .styles import (
    TAB_STYLES,
    LOG_STYLES,
    FLAB_STYLE_SHEET,
    PRIMARY,
    MODE_DRY_COLOR,
    MODE_TRAIN_COLOR,
    MODE_TUNE_COLOR,
    MODE_INFER_COLOR,
    MODE_XFER_COLOR,
    MODE_RESULTS_COLOR, 
    RUN_BUTTON_IDLE,
)
from .jobs import TrainJobSpec, latest_jobs_for_root
from .train_dialogs import (
    TrainOptionsDialog,
    QuickTrainDialog,
)
from .tune_dialogs import (
    TuneOptionsDialog,
    QuickTuneDialog,
    TuneJobSpec,
)
from .results_tab import ResultsDownloadTab

from .manager import LogManager
from .components import RangeListEditor
from .scalars_loss_dialog import ScalarsLossDialog
from .view_utils import _notify_gui_xfer_view

class GeoPriorForecaster(QMainWindow):
    """GUI wrapper for GeoPrior subsidence runs."""

    log_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(float)

    def __init__(
        self,
        *,
        theme: str = "fusionlab",
        splash: LoadingSplash | None = None,
    ) -> None:
        super().__init__()
        self.theme = theme or "fusionlab"

        if splash:
            splash.set_progress(20, "Loading defaults…")

        self.stage1_thread: Stage1Thread | None = None
        self.train_thread: TrainingThread | None = None
        self.tuning_thread: TuningThread | None = None  
        self.inference_thread: InferenceThread | None = None
        self.xfer_thread: XferMatrixThread | None = None 
        self.xfer_view_thread: XferViewThread | None = None    
        
        self._xfer_last_result: Dict[str, Any] | None = None 
        
        # job selected from the options/quick dialogs
        self._queued_train_job: TrainJobSpec |None = None
        self._queued_tune_job: TuneJobSpec | None = None    

        # --- global run/stop state ---
        self._active_job_kind: str | None = None  # "train", "tune", "infer", "xfer"
        
        # Selected CSV (optional) + config overrides passed to threads
        self.csv_path: Path | None = None
        self._cfg_overrides: Dict[str, Any] = {}
        
        self._device_cfg_overrides: Dict[str, Any] = {}

        # Central config object (defaults loaded from nat.com/config.py)
        self.geo_cfg = GeoPriorConfig.from_nat_config()
        
        if splash:
            splash.set_progress(40, "Building UI…")
    
        self._stage1_manifest_hint: Path | None = None 
        # Dialog for scalar HPs / loss weights (used from Tune tab)
        self.scalars_dialog = ScalarsLossDialog(self)
        
        # Dedicated root so GUI runs don't mix with CLI results
        # self.gui_runs_root = Path(get_default_runs_root())
        # --- NEW: dedicated runs root for GUI ---
        self.gui_runs_root = Path.home() / ".fusionlab_runs"
        self.gui_runs_root.mkdir(parents=True, exist_ok=True)
        
        # user-overrideable base results root (defaults to gui_runs_root)
        self.results_root = self.gui_runs_root
        
        self._train_help_text = (
            "Train tab – runs Stage-1 (prepare) and Stage-2 "
            "(GeoPrior training) for the selected city."
        )
        self._train_tip_shown: bool = False  
        
        # Help text for Tune tab – will be shown as tooltip/notification
        self._tune_help_text = (
            "Tune tab – runs Stage-2 hyperparameter search for the "
            "selected city using existing Stage-1 sequences.\n"
            "Note: Stage-1 must already exist for this city; run the "
            "Train tab first if necessary."
        )
        self._tune_tip_shown: bool = False  # one-shot notification flag
        self._tune_help_text = (
            "Tune tab – runs Stage-2 hyperparameter search for the "
            "selected city using existing Stage-1 sequences.\n"
            "Note: Stage-1 must already exist for this city; run the "
            "Train tab first if necessary."
        )
        self._tune_tip_shown: bool = False  # one-shot notification flag
        
        # NEW: Inference tab helper
        self._infer_help_text = (
            "Inference tab – evaluate a trained/tuned GeoPriorSubsNet on\n "
            "train/val/test splits or run future forecasts based on\n "
            "Stage-1 future NPZ artifacts."
        )
        self._infer_tip_shown: bool = False       

        # Transferability tab helpe
        self._xfer_help_text = (
            "Transferability tab – run cross-city transfer matrix (A↔B)\n "
            "and build view figures from xfer_results.* files."
        )
        self._xfer_tip_shown: bool = False

        self._results_help_text = (
            "Results tab – browse and download Stage-1 artifacts,\n"
            "train/tune/inference runs, and transferability outputs\n"
            "as ZIP archives."
        )
        self._results_tip_shown: bool = False


        # Advanced options state for XFER
        self._xfer_quantiles_override = None
        self._xfer_write_json = True
        self._xfer_write_csv = True

        self._build_ui()
        
        if splash:
            splash.set_progress(70, "Preparing log manager…")
            
        # Log manager: central buffer + on-disk dumping
        self.log_mgr = LogManager(
            self.log_widget,
            mode="collapse",
            log_dir_name="_log",
        )
        if splash:
            splash.set_progress(85, "Connecting signals…")
            
        self._connect_signals()
        self._set_window_props()
        
        if splash:
            splash.set_progress(95, "Finalising…")
            
        self._preview_windows: list[QDialog] = []        # ← keep refs
        VIS_SIGNALS.figure_saved.connect(self._show_image_popup)

    def _show_image_popup(self, png_path: str) -> None:
        dlg = ImagePreviewDialog(png_path, parent=self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)             # frees memory on close
        dlg.show()                                        # modeless – *no* exec_()
        self._preview_windows.append(dlg)                 # keep it alive
        
    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _set_window_props(self) -> None:
        self.setWindowTitle("Fusionlab – GeoPrior-3.0 Forecaster")
    
        # Read window sizing from GeoPriorConfig (with safe fallbacks)
        cfg = self.geo_cfg
    
        base_w = getattr(cfg, "ui_base_width", 820)
        base_h = getattr(cfg, "ui_base_height", 520)
        min_w = getattr(cfg, "ui_min_width", 800)
        min_h = getattr(cfg, "ui_min_height", 600)
        max_ratio = float(getattr(cfg, "ui_max_ratio", 0.90))
    
        auto_resize_window(
            self,
            base_size=(base_w, base_h),
            min_size=(min_w, min_h),
            max_ratio=max_ratio,
        )
    
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

    # --------------------------------------------------------------
    # Run button helpers
    # --------------------------------------------------------------
    def _make_play_icon(
        self,
        diameter: int = 26,
        *,
        hollow: bool = False,
    ) -> QIcon:
        """
        Play icon used for run buttons.
    
        When hollow=True, the triangle is outlined instead of filled
        to visually indicate 'dry-run / preview'.
        """
        pix = QPixmap(diameter, diameter)
        pix.fill(Qt.transparent)
    
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
    
        color = QColor(RUN_BUTTON_IDLE)  # base green
    
        # Circle outline
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        radius = diameter // 2 - 2
        center = QPoint(diameter // 2, diameter // 2)
        painter.drawEllipse(center, radius, radius)
    
        # Triangle (play) inside the circle
        tri_w = diameter // 3
        tri_h = diameter // 2
        x0 = diameter // 2 - tri_w // 3
        y0 = diameter // 2 - tri_h // 2
    
        p1 = QPoint(x0, y0)
        p2 = QPoint(x0, y0 + tri_h)
        p3 = QPoint(x0 + tri_w, diameter // 2)
    
        if hollow:
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.NoBrush)
        else:
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
    
        painter.drawPolygon(p1, p2, p3)
        painter.end()
    
        return QIcon(pix)

    def _make_run_button(self, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("runButton")
        btn.setToolTip(tooltip)
        btn.setIcon(self._make_play_icon())
        btn.setIconSize(QSize(26, 26))
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        btn.setAutoRaise(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedSize(40, 40)        # match the CSS min/max
        return btn


    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)

        # --- Top row: [Select CSV…] [City / Dataset] [Dry run] [Mode] [Quit] ---
        top = QHBoxLayout()

        self.select_csv_btn = QPushButton("Select CSV…")
        top.addWidget(self.select_csv_btn)

        top.addWidget(QLabel("City / Dataset:"))

        self.city_edit = QLineEdit()
        self.city_edit.setPlaceholderText("e.g. nansha")
        top.addWidget(self.city_edit, 1)

        # stretch between dataset and right-side controls
        top.addStretch(1)

        # Dry-run checkbox (logic will be added later)
        self.chk_dry_run = QCheckBox("Dry run")
        self.chk_dry_run.setToolTip(
            "Prepare configuration and log actions\n"
            " without actually running Stage-1 / Stage-2."
        )
        top.addWidget(self.chk_dry_run)
        
        # Global Stop button – only visible while a job is running
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setVisible(False)
        self.btn_stop.setEnabled(False)
        top.addWidget(self.btn_stop)
        
        # Mode indicator button (updated when tab changes)
        self.mode_btn = QPushButton("Mode: Train")
        self.mode_btn.setEnabled(False)
        self.mode_btn.setFlat(True)
        top.addWidget(self.mode_btn)

        # self.quit_btn = QPushButton("Quit")
        # top.addWidget(self.quit_btn)

        self.quit_btn = QPushButton("Quit")
        self.quit_btn.setObjectName("quitButton")
        self.quit_btn.setToolTip("Quit GeoPrior Forecaster")

        # Local styling: red accent, clearer affordance
        self.quit_btn.setStyleSheet(
            """
            QPushButton#quitButton {
                background-color: #cc3333;
                color: white;
                font-weight: 600;
                border-radius: 8px;
                padding: 4px 14px;
            }
            QPushButton#quitButton:hover {
                background-color: #e55353;
            }
            QPushButton#quitButton:disabled {
                background-color: #888888;
                color: #dddddd;
            }
            """
        )
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
        
        # initialise global running state (no job at startup)
        self._update_global_running_state()
        
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
        
        # --- three cards in a horizontal row ---

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
        
        # Make the temporal rows expand to fill the card height
        for r in range(4):
            grid_t.setRowStretch(r, 1)

        # (optional) make the spin boxes a bit taller so the values
        # are easier to read
        for sb in (
            self.sp_train_end,
            self.sp_forecast_start,
            self.sp_forecast_horizon,
            self.sp_time_steps,
        ):
            sb.setMinimumHeight(26)
            
        temp_box.addLayout(grid_t)

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

        # 3) Physics weights card
        phys_card, phys_box = self._make_card("Physics weights")

        self.cmb_pde_mode = QComboBox()
        self.cmb_pde_mode.addItems(
            ["both", "consolidation", "gw_flow", "off"]
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
        
        # --- Cards + buttons as a grid so Physics can span two rows ---
        cards_grid = QGridLayout()
        cards_grid.setSpacing(10)

        # Make the whole grid top-aligned in its parent
        cards_grid.setAlignment(Qt.AlignTop)
        
        # Row 0: Temporal, Training, Physics (top part)
        cards_grid.addWidget(temp_card,  0, 0, 1, 1, Qt.AlignTop)
        cards_grid.addWidget(train_card, 0, 1, 1, 1, Qt.AlignTop)

        # Physics card spans two rows (rows 0 and 1) in the right column
        cards_grid.addWidget(phys_card, 0, 2, 2, 1, Qt.AlignTop) # rowSpan=2, colSpan=1
        
        # --- Model / config buttons row (only under Temporal + Training) ---
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(12)

        self.btn_features = QPushButton("Feature config…")
        buttons_row.addWidget(self.btn_features)

        self.btn_arch = QPushButton("Architecture…")
        buttons_row.addWidget(self.btn_arch)

        self.btn_prob = QPushButton("Probabilistic…")
        buttons_row.addWidget(self.btn_prob)

        buttons_row.addStretch(1)

        # Put the buttons row in a small container so it can go into the grid
        buttons_container = QWidget()
        buttons_container.setLayout(buttons_row)

        # Row 1, columns 0–1 (under Temporal + Training only)
        cards_grid.addWidget(buttons_container, 1, 0, 1, 2)

        # Add the whole grid to the Train tab layout
        t_layout.addLayout(cards_grid)


        # --- bottom flags ---
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        # Options… button takes the place of the visible clean checkbox
        self.btn_train_options = QPushButton("Advanced options…")
        bottom_row.addWidget(self.btn_train_options)
       
        self.chk_eval_training = QCheckBox(
            "Evaluate training metrics",
        )
        self.chk_eval_training.setChecked(
            cfg.evaluate_training,
        )
        bottom_row.addWidget(self.chk_eval_training)
        
        self.chk_build_future = QCheckBox(
            "Build future NPZ",
        )
        self.chk_build_future.setChecked(
            cfg.build_future_npz,
        )
        bottom_row.addWidget(self.chk_build_future)

        # Hidden checkbox used only as a backing store
        # (not shown directly on the Train tab anymore)
        self.chk_clean_stage1 = QCheckBox("Clean Stage-1 run dir")
        self.chk_clean_stage1.setChecked(cfg.clean_stage1_dir)
        self.chk_clean_stage1.setVisible(False)

        # push Run button to the far right
        bottom_row.addStretch(1)

        self.train_btn = self._make_run_button("Run training")
        bottom_row.addWidget(self.train_btn)

        
        t_layout.addLayout(bottom_row)
        t_layout.addStretch(1)
        
        # ==================================================
        # Tune tab – hyperparameter search
        # ==================================================
        tune_tab = QWidget()
        u_layout = QVBoxLayout(tune_tab)
        u_layout.setContentsMargins(6, 6, 6, 6)
        u_layout.setSpacing(8)

        # --- two cards in a horizontal row ---
        tune_row = QHBoxLayout()
        tune_row.setSpacing(10)

        # ----------------- 1) Architecture card -----------------
        arch_card, arch_box = self._make_card("Architecture search")

        self.hp_embed_dim = QLineEdit()
        self.hp_hidden_units = QLineEdit()
        self.hp_lstm_units = QLineEdit()
        self.hp_attention_units = QLineEdit()
        self.hp_num_heads = QLineEdit()
        self.hp_vsn_units = QLineEdit()

        self.hp_dropout = RangeListEditor(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=3,
            show_sampling=False,
        )
        self.hp_dropout.set_defaults(0.05, 0.20, sampling=None)

        grid_a = QGridLayout()
        row = 0

        # left + right columns -> larger line edits
        grid_a.addWidget(QLabel("Embedding dim (comma-sep):"), row, 0)
        grid_a.addWidget(self.hp_embed_dim, row, 1)
        grid_a.addWidget(QLabel("LSTM units:"), row, 2)
        grid_a.addWidget(self.hp_lstm_units, row, 3)
        row += 1

        grid_a.addWidget(QLabel("Hidden units:"), row, 0)
        grid_a.addWidget(self.hp_hidden_units, row, 1)
        grid_a.addWidget(QLabel("Attention units:"), row, 2)
        grid_a.addWidget(self.hp_attention_units, row, 3)
        row += 1

        grid_a.addWidget(QLabel("Attention heads:"), row, 0)
        grid_a.addWidget(self.hp_num_heads, row, 1)
        grid_a.addWidget(QLabel("VSN units:"), row, 2)
        grid_a.addWidget(self.hp_vsn_units, row, 3)
        row += 1

        grid_a.addWidget(QLabel("Dropout:"), row, 0)
        grid_a.addWidget(self.hp_dropout, row, 1, 1, 3)

        arch_box.addLayout(grid_a)
        tune_row.addWidget(arch_card, 1)

        # ----------------- 2) Physics card -----------------
        phys_card, phys_box = self._make_card("Physics switches")

        self.hp_pde_mode = QLineEdit()
        self.hp_scale_pde_bool = QCheckBox(
            "Tune 'scale PDE residuals' as boolean HP"
        )
        self.hp_kappa_mode = QLineEdit()

        self.hp_hd = RangeListEditor(
            min_allowed=0.0,
            max_allowed=2.0,
            decimals=3,
            show_sampling=False,
        )
        self.hp_hd.set_defaults(0.50, 0.70, sampling=None)

        grid_ph = QGridLayout()
        r = 0
        grid_ph.addWidget(QLabel("PDE modes:"), r, 0)
        grid_ph.addWidget(self.hp_pde_mode, r, 1); r += 1

        grid_ph.addWidget(QLabel("κ mode (bar/kb):"), r, 0)
        grid_ph.addWidget(self.hp_kappa_mode, r, 1); r += 1

        grid_ph.addWidget(self.hp_scale_pde_bool, r, 0, 1, 2); r += 1

        grid_ph.addWidget(QLabel("HD factor:"), r, 0)
        grid_ph.addWidget(self.hp_hd, r, 1)
        phys_box.addLayout(grid_ph)

        tune_row.addWidget(phys_card, 1)

        u_layout.addLayout(tune_row)

        # --- Bottom row: options + Scalars popup + Run button ---
        # Row 1: Evaluate checkbox + Max trials + Scalars button
        opts_row = QHBoxLayout()
        opts_row.setSpacing(16)
        
        self.chk_eval_tuned = QCheckBox("Evaluate tuned model")
        self.chk_eval_tuned.setChecked(False)
        opts_row.addWidget(self.chk_eval_tuned)
        
        # NEW: Max trials (defaults to 20)
        opts_row.addSpacing(8)
        opts_row.addWidget(QLabel("Max trials:"))
        self.spin_max_trials = QSpinBox()
        self.spin_max_trials.setRange(1, 999)
        self.spin_max_trials.setValue(20)  # default
        opts_row.addWidget(self.spin_max_trials)
        
        self.btn_scalars = QPushButton("Scalars & losses…")
        opts_row.addWidget(self.btn_scalars)
        
        opts_row.addStretch(1)   # push them to the left
        u_layout.addLayout(opts_row)


        # Row 2: Advanced options + Run tuning
        ops_run_row = QHBoxLayout()
        self.btn_tune_options = QPushButton("Advanced options…")
        self.btn_run_tune = self._make_run_button("Run tuning")

        ops_run_row.addWidget(self.btn_tune_options)
        ops_run_row.addStretch(1)
        ops_run_row.addWidget(self.btn_run_tune)
        u_layout.addLayout(ops_run_row)
        
 
        # ==================================================
        # Inference tab – evaluation & forecasting
        # ==================================================
        infer_tab = QWidget()
        i_layout = QVBoxLayout(infer_tab)
        i_layout.setContentsMargins(6, 6, 6, 6)
        i_layout.setSpacing(8)

        inf_row = QHBoxLayout()
        inf_row.setSpacing(10)

        # 1) Model & dataset card
        model_card, model_box = self._make_card("Model & dataset")

        self.inf_model_edit = QLineEdit()
        self.inf_model_edit.setPlaceholderText("Select .keras model…")
        self.inf_model_btn = QPushButton("Browse…")

        self.inf_manifest_edit = QLineEdit()
        self.inf_manifest_edit.setPlaceholderText(
            "Stage-1 manifest (auto if empty)"
        )
        self.inf_manifest_btn = QPushButton("Browse…")

        self.cmb_inf_dataset = QComboBox()
        self.cmb_inf_dataset.addItem("Validation (val)", "val")
        self.cmb_inf_dataset.addItem("Test (test)", "test")
        self.cmb_inf_dataset.addItem("Train (train)", "train")
        self.cmb_inf_dataset.addItem("Custom NPZ", "custom")

        self.chk_inf_use_future = QCheckBox(
            "Use Stage-1 future NPZ (forecast mode)"
        )

        self.inf_inputs_edit = QLineEdit()
        self.inf_inputs_edit.setPlaceholderText("Custom inputs .npz")
        self.inf_inputs_btn = QPushButton("Inputs…")

        self.inf_targets_edit = QLineEdit()
        self.inf_targets_edit.setPlaceholderText(
            "Optional targets .npz (for metrics)"
        )
        self.inf_targets_btn = QPushButton("Targets…")

        self.sp_inf_batch = QSpinBox()
        self.sp_inf_batch.setRange(1, 2048)
        self.sp_inf_batch.setValue(32)

        grid_inf1 = QGridLayout()
        r = 0
        grid_inf1.addWidget(QLabel("Model file:"), r, 0)
        grid_inf1.addWidget(self.inf_model_edit, r, 1)
        grid_inf1.addWidget(self.inf_model_btn, r, 2)
        r += 1

        grid_inf1.addWidget(QLabel("Stage-1 manifest:"), r, 0)
        grid_inf1.addWidget(self.inf_manifest_edit, r, 1)
        grid_inf1.addWidget(self.inf_manifest_btn, r, 2)
        r += 1

        grid_inf1.addWidget(QLabel("Dataset:"), r, 0)
        grid_inf1.addWidget(self.cmb_inf_dataset, r, 1, 1, 2)
        r += 1

        # checkbox + batch size on the same row
        npz_row = QHBoxLayout()
        npz_row.addWidget(self.chk_inf_use_future)
        npz_row.addSpacing(16)
        npz_row.addWidget(QLabel("Batch size:"))
        npz_row.addWidget(self.sp_inf_batch)
        npz_row.addStretch(1)
        grid_inf1.addLayout(npz_row, r, 0, 1, 3)
        r += 1

        grid_inf1.addWidget(QLabel("Custom inputs:"), r, 0)
        grid_inf1.addWidget(self.inf_inputs_edit, r, 1)
        grid_inf1.addWidget(self.inf_inputs_btn, r, 2)
        r += 1

        grid_inf1.addWidget(QLabel("Custom targets:"), r, 0)
        grid_inf1.addWidget(self.inf_targets_edit, r, 1)
        grid_inf1.addWidget(self.inf_targets_btn, r, 2)
        r += 1

        model_box.addLayout(grid_inf1)
        inf_row.addWidget(model_card, 1)

        # 2) Calibration & outputs card
        calib_card, calib_box = self._make_card("Calibration & outputs")

        self.chk_inf_use_source_calib = QCheckBox(
            "Use source calibrator (interval_factors_80.npy)"
        )
        self.chk_inf_fit_calib = QCheckBox(
            "Fit calibrator on validation split"
        )

        self.inf_calib_edit = QLineEdit()
        self.inf_calib_edit.setPlaceholderText(
            "Optional explicit calibrator .npy"
        )
        self.inf_calib_btn = QPushButton("Browse…")

        self.sp_inf_cov = QDoubleSpinBox()
        self.sp_inf_cov.setDecimals(3)
        self.sp_inf_cov.setRange(0.50, 0.99)
        self.sp_inf_cov.setSingleStep(0.01)
        self.sp_inf_cov.setValue(0.80)

        self.chk_inf_include_gwl = QCheckBox(
            "Include GWL columns in CSV"
        )
        self.chk_inf_include_gwl.setChecked(False)

        self.chk_inf_plots = QCheckBox("Generate plots")
        self.chk_inf_plots.setChecked(True)

        grid_inf2 = QGridLayout()
        c = 0
        grid_inf2.addWidget(self.chk_inf_use_source_calib, c, 0, 1, 3)
        c += 1
        grid_inf2.addWidget(self.chk_inf_fit_calib, c, 0, 1, 3)
        c += 1

        grid_inf2.addWidget(QLabel("Calibrator file:"), c, 0)
        grid_inf2.addWidget(self.inf_calib_edit, c, 1)
        grid_inf2.addWidget(self.inf_calib_btn, c, 2)
        c += 1

        grid_inf2.addWidget(QLabel("Target coverage:"), c, 0)
        grid_inf2.addWidget(self.sp_inf_cov, c, 1, 1, 2)
        c += 1

        grid_inf2.addWidget(self.chk_inf_include_gwl, c, 0, 1, 3)
        c += 1
        grid_inf2.addWidget(self.chk_inf_plots, c, 0, 1, 3)
        c += 1

        calib_box.addLayout(grid_inf2)
        inf_row.addWidget(calib_card, 1)

        i_layout.addLayout(inf_row)

        # --- Bottom row: Advanced options + Run inference -------------

        # Bottom row: advanced options + run button
        inf_run_row = QHBoxLayout()
        self.btn_inf_options = QPushButton("Advanced options…")
        inf_run_row.addWidget(self.btn_inf_options)

        inf_run_row.addStretch(1)

        self.btn_run_infer = self._make_run_button("Run inference")
        inf_run_row.addWidget(self.btn_run_infer)
        i_layout.addLayout(inf_run_row)

        i_layout.addStretch(1)

        # ==================================================
        # Transferability tab – cross-city transfer matrix
        # ==================================================
        xfer_tab = QWidget()
        x_layout = QVBoxLayout(xfer_tab)
        x_layout.setContentsMargins(6, 6, 6, 6)
        x_layout.setSpacing(8)
        
        row = QHBoxLayout()
        row.setSpacing(10)
        row.setAlignment(Qt.AlignTop)
        
        # ----- Cities & splits card -----
        cities_card, cities_box = self._make_card("Cities & splits")
        
        self.xfer_city_a = QLineEdit()
        self.xfer_city_a.setPlaceholderText("nansha")
        self.xfer_city_b = QLineEdit()
        self.xfer_city_b.setPlaceholderText("zhongshan")
        
        self.chk_xfer_split_train = QCheckBox("train")
        self.chk_xfer_split_val = QCheckBox("val")
        self.chk_xfer_split_test = QCheckBox("test")
        self.chk_xfer_split_val.setChecked(True)
        self.chk_xfer_split_test.setChecked(True)
        
        self.chk_xfer_cal_none = QCheckBox("none")
        self.chk_xfer_cal_source = QCheckBox("source")
        self.chk_xfer_cal_target = QCheckBox("target")
        for cb in (
            self.chk_xfer_cal_none,
            self.chk_xfer_cal_source,
            self.chk_xfer_cal_target,
        ):
            cb.setChecked(True)
        
        self.sp_xfer_batch = QSpinBox()
        self.sp_xfer_batch.setRange(1, 2048)
        self.sp_xfer_batch.setValue(32)
        
        self.chk_xfer_rescale = QCheckBox(
            "Rescale target city to source domain"
        )
        
        grid_cs = QGridLayout()
        r = 0
        grid_cs.addWidget(QLabel("City A (source):"), r, 0)
        grid_cs.addWidget(self.xfer_city_a, r, 1); r += 1
        grid_cs.addWidget(QLabel("City B (target):"), r, 0)
        grid_cs.addWidget(self.xfer_city_b, r, 1); r += 1
        
        grid_cs.addWidget(QLabel("Splits:"), r, 0)
        splits_row = QHBoxLayout()
        splits_row.addWidget(self.chk_xfer_split_train)
        splits_row.addWidget(self.chk_xfer_split_val)
        splits_row.addWidget(self.chk_xfer_split_test)
        splits_row.addStretch(1)
        grid_cs.addLayout(splits_row, r, 1); r += 1
        
        grid_cs.addWidget(QLabel("Calibration modes:"), r, 0)
        cal_row = QHBoxLayout()
        cal_row.addWidget(self.chk_xfer_cal_none)
        cal_row.addWidget(self.chk_xfer_cal_source)
        cal_row.addWidget(self.chk_xfer_cal_target)
        cal_row.addStretch(1)
        grid_cs.addLayout(cal_row, r, 1); r += 1
        
        grid_cs.addWidget(QLabel("Batch size:"), r, 0)
        grid_cs.addWidget(self.sp_xfer_batch, r, 1); r += 1
        
        grid_cs.addWidget(self.chk_xfer_rescale, r, 0, 1, 2); r += 1
        
        cities_box.addLayout(grid_cs)
        row.addWidget(cities_card, 1, Qt.AlignTop)
        
        # ----- Results & view card -----
        res_card, res_box = self._make_card("Results & view")

        self.xfer_results_root = QLineEdit(str(self.gui_runs_root))
        self.xfer_results_root_btn = QPushButton("Browse…")
        
        self.lbl_xfer_last_out = QLabel("No transfer run yet.")
        self.lbl_xfer_last_out.setObjectName("xferLastOutLabel")
        
        self.cmb_xfer_view = QComboBox()
        self.cmb_xfer_view.addItem(
            "Calibration vs error (scatter panel)",
            "calib_panel",
        )
        self.cmb_xfer_view.addItem(
            "Per-horizon MAE + cov/sharp (summary)",
            "summary_panel",
        )
        
        self.cmb_xfer_view_split = QComboBox()
        self.cmb_xfer_view_split.addItem("Validation (val)", "val")
        self.cmb_xfer_view_split.addItem("Test (test)", "test")
        
        self.btn_xfer_view = QPushButton("Make view figure…")
        self.btn_xfer_view.setVisible(False)   # appears after first run
        
        grid_res = QGridLayout()
        c = 0
        grid_res.addWidget(QLabel("Results root:"), c, 0)
        grid_res.addWidget(self.xfer_results_root, c, 1)
        grid_res.addWidget(self.xfer_results_root_btn, c, 2); c += 1
        
        grid_res.addWidget(QLabel("Last output folder:"), c, 0)
        grid_res.addWidget(self.lbl_xfer_last_out, c, 1, 1, 2); c += 1
        
        grid_res.addWidget(QLabel("View type:"), c, 0)
        grid_res.addWidget(self.cmb_xfer_view, c, 1, 1, 2); c += 1
        
        grid_res.addWidget(QLabel("View split:"), c, 0)
        grid_res.addWidget(self.cmb_xfer_view_split, c, 1, 1, 2); c += 1
        
        grid_res.addWidget(self.btn_xfer_view, c, 0, 1, 3); c += 1
        
        res_box.addLayout(grid_res)
        row.addWidget(res_card, 1, Qt.AlignTop)
        
        x_layout.addLayout(row)
        
        # Bottom row: advanced button + run button
        bottom = QHBoxLayout()
        self.btn_xfer_advanced = QPushButton("Advanced options…")
        bottom.addWidget(self.btn_xfer_advanced)
        bottom.addStretch(1)
        self.btn_run_xfer = self._make_run_button("Run transfer matrix")
        bottom.addWidget(self.btn_run_xfer)
        x_layout.addLayout(bottom)
        
        x_layout.addStretch(1)
        # ----------------------
 
        self.train_tab = train_tab
        self.tune_tab = tune_tab
        self.infer_tab = infer_tab
        self.xfer_tab = xfer_tab

        # NEW: Results tab – browse & download artifacts/runs
        results_tab = ResultsDownloadTab(
            results_root=self.gui_runs_root,
            get_results_root=lambda: self.gui_runs_root,
            parent=self,
        )
        self.results_tab = results_tab

        self.tabs.addTab(train_tab, "Train")
        self.tabs.addTab(tune_tab, "Tune")
        self.tabs.addTab(infer_tab, "Inference")
        self.tabs.addTab(xfer_tab, "Transferability")
        self.tabs.addTab(results_tab, "Results")

        # Tab indices (used by mode indicator)
        self._train_tab_index = self.tabs.indexOf(self.train_tab)
        self._tune_tab_index = self.tabs.indexOf(self.tune_tab)
        self._infer_tab_index = self.tabs.indexOf(self.infer_tab)
        self._xfer_tab_index = self.tabs.indexOf(self.xfer_tab)
        self._results_tab_index = self.tabs.indexOf(self.results_tab)


        # Initialise Mode button with current tab name
        self._update_mode_button(self.tabs.currentIndex())


        # After building Tune widgets, sync from current config
        self._load_tuner_space_into_ui()
        
        # After building Inference widgets, set initial enable/disable state
        self._update_infer_widgets_state()  
        
        # Try to auto-discover the latest transfer run under current results root
        try:
            self._discover_last_xfer_for_root()
        except Exception as exc:
            # Robust fallback: do not crash GUI if something is odd on disk
            self.log_updated.emit(
                f"[WARN] Could not auto-discover transfer results: {exc}"
            )
            self._xfer_last_result = {}
            self._update_xfer_view_state()

    def _discover_last_xfer_for_root(self) -> None:
        """
        Best-effort discovery of the latest transferability run
        under the current xfer results root.

        Keeps `_xfer_last_result` consistent with whatever is
        already on disk.
        """

        root_text = self.xfer_results_root.text().strip()
        if not root_text:
            # Nothing to search
            self._xfer_last_result = {}
            self.lbl_xfer_last_out.setText("No transfer run yet.")
            self._update_xfer_view_state()
            return

        results_root = Path(root_text)

        # latest_xfer_* return *strings*, so pass a string
        csv_path = latest_xfer_csv(str(results_root))
        json_path = latest_xfer_json(str(results_root))

        if not csv_path and not json_path:
            # No xfer/*/*/xfer_results.* found for this root
            self._xfer_last_result = {}
            self.lbl_xfer_last_out.setText("No transfer run yet.")
            self._update_xfer_view_state()
            return

        # Use whichever artifact we found to infer the run folder
        best_path = csv_path or json_path
        run_dir = Path(best_path).parent   # <-- wrap in Path

        self._xfer_last_result = {
            "out_dir": str(run_dir),
            "csv_path": str(csv_path) if csv_path else None,
            "json_path": str(json_path) if json_path else None,
        }

        self.lbl_xfer_last_out.setText(str(run_dir))
        self._update_xfer_view_state()

        # optional debug log
        self.log_updated.emit(
            "Detected latest transferability run under results root:\n"
            f"  {run_dir}"
        )

    def _is_dry_mode(self) -> bool:
        chk = getattr(self, "chk_dry_run", None)
        return bool(chk is not None and chk.isChecked())


    # ------------------------------------------------
    #   Update buttons 
    # ------------------------------------------------
    def _update_mode_button(self, index: int) -> None:
        """
        Update the top 'Mode' indicator according to the current tab
        and the Dry-run state. The mode tooltip carries the detailed
        help text for the active mode.
        """
        if not hasattr(self, "mode_btn"):
            return
        if not hasattr(self, "tabs") or self.tabs is None:
            return

        # Dry-run has priority over tab mode
        is_dry = (
            hasattr(self, "chk_dry_run")
            and self.chk_dry_run.isChecked()
        )

        if is_dry:
            mode_label = "DRY RUN"
            color = MODE_DRY_COLOR
            tooltip = (
                "Dry-run mode – prepare configuration and log actions\n"
                "without actually running Stage-1 / Stage-2 / Stage-3."
            )
        else:
            # Default: choose by current tab
            if index == getattr(self, "_train_tab_index", -1):
                mode_label = "TRAIN"
                color = MODE_TRAIN_COLOR
                tooltip = getattr(
                    self, "_train_help_text",
                    "Train tab – Stage-1 + Stage-2 GeoPrior training.",
                )
            elif index == getattr(self, "_tune_tab_index", -1):
                mode_label = "TUNING"
                color = MODE_TUNE_COLOR
                tooltip = getattr(
                    self, "_tune_help_text",
                    "Tune tab – Stage-2 hyperparameter search.",
                )
            elif index == getattr(self, "_infer_tab_index", -1):
                mode_label = "INFER"
                color = MODE_INFER_COLOR
                tooltip = getattr(
                    self, "_infer_help_text",
                    "Inference tab – evaluation and future forecasts.",
                )
            elif index == getattr(self, "_xfer_tab_index", -1):
                mode_label = "TRANSFER"
                color = MODE_XFER_COLOR
                tooltip = getattr(
                    self, "_xfer_help_text",
                    "Transferability tab – cross-city transfer matrix.",
                )
            elif index == getattr(self, "_results_tab_index", -1):
                mode_label = "RESULTS"
                color = MODE_RESULTS_COLOR
                tooltip = getattr(
                    self, "_results_help_text",
                    "Results tab – browse and download artifacts.",
                )
            else:
                # Fallback: just use tab text
                mode_label = self.tabs.tabText(index) or "–"
                color = MODE_TRAIN_COLOR
                tooltip = f"Mode: {mode_label}"

        # Text + tooltip
        self.mode_btn.setText(f"Mode: {mode_label}")
        self.mode_btn.setToolTip(tooltip)

        # Per-mode background; keep it non-clickable but styled
        self.mode_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 10px;
                padding: 2px 12px;
                font-weight: 600;
            }}
            QPushButton:disabled {{
                background-color: {color};
                color: white;
            }}
            """
        )

    def _update_xfer_view_state(self) -> None:
        has_result = bool(self._xfer_last_result)
        self.btn_xfer_view.setVisible(has_result)
        self.btn_xfer_view.setEnabled(has_result)

    def _update_infer_widgets_state(self) -> None:
        """
        Enable/disable custom NPZ fields depending on dataset mode
        and 'use future NPZ' switch.
        """
        if not hasattr(self, "cmb_inf_dataset"):
            return

        dataset_key = self.cmb_inf_dataset.currentData()
        use_future = self.chk_inf_use_future.isChecked()

        is_custom_npz = (dataset_key == "custom") and not use_future

        for w in (
            self.inf_inputs_edit,
            self.inf_inputs_btn,
            self.inf_targets_edit,
            self.inf_targets_btn,
        ):
            w.setEnabled(is_custom_npz)

        # When using future NPZ, dataset choice is not relevant
        self.cmb_inf_dataset.setEnabled(not use_future)
        
    # ------------------------------------------------
    #   Global run / stop helpers
    # ------------------------------------------------
    def _any_job_running(self) -> bool:
        """Return True if any long-running workflow thread is active."""
        return any(
            th is not None and th.isRunning()
            for th in (
                self.stage1_thread,
                self.train_thread,
                self.tuning_thread,
                self.inference_thread,
                self.xfer_thread,
            )
        )

    def _update_global_running_state(self) -> None:
        any_running = self._any_job_running()
    
        self.btn_stop.setEnabled(any_running)
    
        is_dry = self._is_dry_mode()
    
        # --- Train button ---------------------------------------------------
        if hasattr(self, "train_btn"):
            if self._active_job_kind == "train":
                tip = (
                    "Running training…"
                    if not is_dry
                    else "Dry-run – computing planned training steps…"
                )
            else:
                tip = (
                    "Run training"
                    if not is_dry
                    else "Run dry (show planned training workflow)"
                )
            self.train_btn.setToolTip(tip)
    
        # --- Tune button ----------------------------------------------------
        if hasattr(self, "btn_run_tune"):
            if self._active_job_kind == "tune":
                tip = (
                    "Running tuning…"
                    if not is_dry
                    else "Dry-run – computing planned tuning workflow…"
                )
            else:
                tip = (
                    "Run tuning"
                    if not is_dry
                    else "Run dry (show planned tuning workflow)"
                )
            self.btn_run_tune.setToolTip(tip)
    
        # --- Inference button -----------------------------------------------
        if hasattr(self, "btn_run_infer"):
            if self._active_job_kind == "infer":
                tip = (
                    "Running inference…"
                    if not is_dry
                    else "Dry-run – computing planned inference workflow…"
                )
            else:
                tip = (
                    "Run inference"
                    if not is_dry
                    else "Run dry (show planned inference workflow)"
                )
            self.btn_run_infer.setToolTip(tip)
    
        # --- Xfer (transferability) button ----------------------------------
        if hasattr(self, "btn_run_xfer"):
            if self._active_job_kind == "xfer":
                tip = (
                    "Running transfer matrix…"
                    if not is_dry
                    else "Dry-run – computing planned transfer workflow…"
                )
            else:
                tip = (
                    "Run transfer matrix"
                    if not is_dry
                    else "Run dry (show planned transfer matrix workflow)"
                )
            self.btn_run_xfer.setToolTip(tip)
        
        # --- Update run-button icons for dry vs normal mode ---
        try:
            icon_normal = self._make_play_icon(hollow=False)
            icon_dry = self._make_play_icon(hollow=True)
        except Exception:
            # Very defensive: never fail just for an icon
            return
        
        dry_icon = icon_dry if is_dry else icon_normal
        
        # Apply to all run buttons we have
        self.train_btn.setIcon(dry_icon)
        self.btn_run_tune.setIcon(dry_icon)
        self.btn_run_infer.setIcon(dry_icon)
        self.btn_run_xfer.setIcon(dry_icon)

    # --------------------------------------------------------------
    # Tuner search space <-> UI
    # --------------------------------------------------------------

    def _parse_int_list(
        self,
        text: str,
        fallback: list[int],
    ) -> list[int]:
        parts = []
        for tok in text.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                parts.append(int(tok))
            except Exception:
                pass
        return parts or list(fallback)

    def _load_tuner_space_into_ui(self) -> None:
        """Populate Tune tab widgets from self.geo_cfg.tuner_search_space."""
        space = self.geo_cfg.tuner_search_space or default_tuner_search_space()
        defaults = default_tuner_search_space()

        def _get(name: str):
            return space.get(name, defaults.get(name))

        # --- Architecture lists ---
        self.hp_embed_dim.setText(
            ", ".join(str(v) for v in _get("embed_dim"))
        )
        self.hp_hidden_units.setText(
            ", ".join(str(v) for v in _get("hidden_units"))
        )
        self.hp_lstm_units.setText(
            ", ".join(str(v) for v in _get("lstm_units"))
        )
        self.hp_attention_units.setText(
            ", ".join(str(v) for v in _get("attention_units"))
        )
        self.hp_num_heads.setText(
            ", ".join(str(v) for v in _get("num_heads"))
        )
        self.hp_vsn_units.setText(
            ", ".join(str(v) for v in _get("vsn_units"))
        )

        # Dropout: range or list
        self.hp_dropout.from_search_space_value(
            _get("dropout_rate"),
            defaults["dropout_rate"],
        )

        # --- Physics ---
        self.hp_pde_mode.setText(
            ", ".join(str(v) for v in _get("pde_mode"))
        )
        self.hp_kappa_mode.setText(
            ", ".join(str(v) for v in _get("kappa_mode"))
        )

        sc_pde = _get("scale_pde_residuals")
        self.hp_scale_pde_bool.setChecked(bool(sc_pde))

        self.hp_hd.from_search_space_value(
            _get("hd_factor"),
            defaults["hd_factor"],
        )

        # --- Scalars & loss weights (dialog only) ---
        self.scalars_dialog.load_from_space(space, defaults)


    def _build_tuner_space_from_ui(self) -> Dict[str, Any]:
        """Collect hyperparameters from Tune tab into a TUNER_SEARCH_SPACE dict."""
        defaults = default_tuner_search_space()
        space: Dict[str, Any] = {}

        # --- Architecture (lists) ---
        space["embed_dim"] = self._parse_int_list(
            self.hp_embed_dim.text(), defaults["embed_dim"]
        )
        space["hidden_units"] = self._parse_int_list(
            self.hp_hidden_units.text(), defaults["hidden_units"]
        )
        space["lstm_units"] = self._parse_int_list(
            self.hp_lstm_units.text(), defaults["lstm_units"]
        )
        space["attention_units"] = self._parse_int_list(
            self.hp_attention_units.text(), defaults["attention_units"]
        )
        space["num_heads"] = self._parse_int_list(
            self.hp_num_heads.text(), defaults["num_heads"]
        )
        space["vsn_units"] = self._parse_int_list(
            self.hp_vsn_units.text(), defaults["vsn_units"]
        )

        # Range / list editor
        space["dropout_rate"] = self.hp_dropout.to_search_space_value()

        # --- Physics ---
        def _parse_str_list(text: str, fallback: list[str]) -> list[str]:
            vals: list[str] = []
            for tok in text.replace(";", ",").split(","):
                t = tok.strip()
                if t:
                    vals.append(t)
            return vals or list(fallback)

        space["pde_mode"] = _parse_str_list(
            self.hp_pde_mode.text(), defaults["pde_mode"]
        )
        space["kappa_mode"] = _parse_str_list(
            self.hp_kappa_mode.text(), defaults["kappa_mode"]
        )

        if self.hp_scale_pde_bool.isChecked():
            space["scale_pde_residuals"] = {"type": "bool"}

        space["hd_factor"] = self.hp_hd.to_search_space_value()

        # --- Scalars & loss weights (from dialog) ---
        space.update(self.scalars_dialog.to_search_space_fragment())

        return space


    # ------------------------------------------------------------------
    # Config synchronisation
    # ------------------------------------------------------------------

    def _sync_config_from_ui(self) -> None:
        """Copy widgets' values into self.geo_cfg."""
        cfg = self.geo_cfg
        cfg.train_end_year = self.sp_train_end.value()
        cfg.forecast_start_year = self.sp_forecast_start.value()
        cfg.forecast_horizon_years = (
            self.sp_forecast_horizon.value()
        )
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

        cfg.clean_stage1_dir = (
            self.chk_clean_stage1.isChecked()
        )
        cfg.build_future_npz = (               
            self.chk_build_future.isChecked()
        )
        cfg.evaluate_training = (
            self.chk_eval_training.isChecked()
        )
        
    def _get_experiment_name(self, default: str) -> str:
        """
        Return the experiment name from an optional GUI 
        field if it exists,
        otherwise fall back to `default`.
        """
        widget = getattr(self, "edit_experiment_name", None)
        if widget is not None:
            try:
                name = widget.text().strip()
            except Exception:
                name = ""
            if name:
                return name
        return default

    # ------------------------------------------------------------------
    # Smart Stage-1 helpers
    # ------------------------------------------------------------------
    def _smart_stage1_handshake(
        self,
        city: str,
        csv_path: str,
    ) -> bool:
        """
        Decide whether we should run Stage-1, or reuse an existing run.

        Returns
        -------
        need_stage1 : bool
            True  -> run Stage-1 now.
            False -> either we jumped straight to training or user cancelled.
        """

        # 0) Forced rebuild via Training options
        if bool(getattr(self.geo_cfg, "clean_stage1_dir", False)):
            self.log_updated.emit(
                "[SmartStage1] 'Clean Stage-1 run dir' is enabled → "
                "forcing Stage-1 rebuild."
            )
            self._stage1_manifest_hint = None
            return True

        # 1) Build current Stage-1 config snapshot via nat.com config
        try:
            current_cfg = self.geo_cfg.to_stage1_config()

        except Exception as e:
            self.log_updated.emit(
                "[SmartStage1] Failed to build current Stage-1 config "
                f"({e}) → falling back to full Stage-1."
            )
            self._stage1_manifest_hint = None
            return True

        # 2) Discover Stage-1 manifests under the GUI results root
        results_root = (
            getattr(self, "gui_runs_root", None)
            or Path(get_default_runs_root())
        )

        runs_for_city, all_runs = find_stage1_for_city(
            city=city,
            results_root=Path(results_root),
            current_cfg=current_cfg,
        )

        if not runs_for_city:
            self.log_updated.emit(
                "[SmartStage1] No previous Stage-1 manifest found for this "
                "city → running Stage-1."
            )
            self._stage1_manifest_hint = None
            return True

        # ... after runs_for_city / all_runs have been computed ...
        if not all_runs:
            # unchanged: no Stage-1 at all => build
            self.log_updated.emit(
                "[Stage-1] No existing Stage-1 runs in this root – "
                "building Stage-1 from scratch."
            )
            return True

        # Smart options from config
        auto_reuse = bool(
            getattr(self.geo_cfg, "stage1_auto_reuse_if_match", True)
        )
        force_rebuild_mismatch = bool(
            getattr(self.geo_cfg, "stage1_force_rebuild_if_mismatch", True)
        )

        # 3a) Auto-reuse path: latest complete + config_match
        if auto_reuse:
            best = next(
                (
                    r
                    for r in runs_for_city
                    if r.is_complete and r.config_match
                ),
                None,
            )
            if best is not None:
                self.log_updated.emit(
                    "[Stage-1] Auto-reusing complete Stage-1 run "
                    f"for city '{city}' @ {best.timestamp} "
                    "(config matches current GUI settings)."
                )
                self._stage1_manifest_hint = best.manifest_path
                return False  # skip Stage-1 build

        # 3b) Force rebuild when nothing matches the current config
        any_match = any(r.config_match for r in runs_for_city)
        if force_rebuild_mismatch and not any_match:
            self.log_updated.emit(
                "[Stage-1] Existing Stage-1 runs found but none "
                "match the current GUI config – forcing Stage-1 "
                "rebuild for city '{city}'."
            )
            self._stage1_manifest_hint = None
            return True
        
        # 3c) Fallback: original behaviour with Stage1ChoiceDialog 
        decision, selected  = Stage1ChoiceDialog.ask(
            parent=self,
            city=city,
            runs_for_city=runs_for_city,
            all_runs=all_runs,
            clean_stage1=self.geo_cfg.clean_stage1_dir,
        )
        
        if decision == "cancel":
            self.log_updated.emit(
                "[SmartStage1] Training cancelled by user."
            )
            self._stage1_manifest_hint = None
            return False

        if decision == "rebuild":
            self.log_updated.emit(
                "[SmartStage1] User requested Stage-1 rebuild."
            )
            self._stage1_manifest_hint = None
            return True

        if decision == "reuse" and selected is not None:
            # Safety: only reuse fully compatible + complete runs
            if (not selected.is_complete) or (not selected.config_match):
                QMessageBox.warning(
                    self,
                    "Invalid Stage-1 selection",
                    "The selected Stage-1 run is incomplete or incompatible "
                    "with the current configuration.\n\n"
                    "Stage-1 will be rebuilt.",
                )
                self._stage1_manifest_hint = None
                return True

            self._stage1_manifest_hint = selected.manifest_path

            diff_msg = ""
            if selected.diff_fields:
                diff_msg = " (diff: " + ", ".join(selected.diff_fields) + ")"

            self.log_updated.emit(
                "[SmartStage1] Reusing Stage-1 run" + diff_msg + ":\n"
                f"  City        : {selected.city}\n"
                f"  Run dir     : {selected.run_dir}\n"
                f"  T, H        : {selected.time_steps}, "
                f"{selected.horizon_years}\n"
                f"  train / val : {selected.n_train} / {selected.n_val}"
            )
            return False

        # Fallback: rebuild
        self._stage1_manifest_hint = None
        return True
    
    @pyqtSlot(bool)
    def _on_dry_run_toggled(self, checked: bool) -> None:
        # Update the "Mode: Normal / Dry run" badge
        self._update_mode_button(self.tabs.currentIndex())
    
        # Update tooltips / labels on run buttons
        self._update_global_running_state()
    
        # Optional: tiny UX message
        if checked:
            self._append_status(
                "Dry-run mode enabled – Run buttons will only "
                "log the planned workflow (no training/tuning/inference)."
            )
        else:
            self._append_status(
                "Dry-run mode disabled – Run buttons now execute "
                "training/tuning/inference normally."
            )

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.select_csv_btn.clicked.connect(
            self._on_select_csv,
        )
        self.train_btn.clicked.connect(
            self._on_train_clicked,
        )
        self.quit_btn.clicked.connect(self._on_quit_clicked)

        self.btn_stop.clicked.connect(self._on_stop_clicked)
        

        self.chk_dry_run.toggled.connect(self._on_dry_run_toggled)
        self.log_updated.connect(self._append_log)
        self.status_updated.connect(
            self.status_label.setText,
        )
        self.progress_updated.connect(
            self._update_progress,
        )

        self.btn_features.clicked.connect(
            self._on_feature_config,
        )
        self.btn_arch.clicked.connect(          
            self._on_arch_config,
        )
        self.btn_prob.clicked.connect(          
            self._on_prob_config,
        )
    
        self.btn_tune_options.clicked.connect(
            self._on_tune_options_clicked,
        )

        self.btn_run_tune.clicked.connect(
            self._on_tune_clicked,
        )

        self.btn_scalars.clicked.connect(
            self._on_scalars_config,
        )
        
        self.btn_train_options.clicked.connect(
            self._on_train_options_clicked
        )

        # --- Inference tab ---
        # --- Inference tab ---
        self.btn_inf_options.clicked.connect(
            self._on_infer_options_clicked
        )
        self.btn_run_infer.clicked.connect(self._on_infer_clicked)
        self.inf_model_btn.clicked.connect(self._on_browse_model)
        self.inf_manifest_btn.clicked.connect(self._on_browse_manifest)
        self.inf_inputs_btn.clicked.connect(self._on_browse_inputs_npz)
        self.inf_targets_btn.clicked.connect(self._on_browse_targets_npz)
        self.inf_calib_btn.clicked.connect(self._on_browse_calibrator)
 
        self.cmb_inf_dataset.currentIndexChanged.connect(
            self._update_infer_widgets_state
        )
        self.chk_inf_use_future.toggled.connect(
            self._update_infer_widgets_state
        )

        # --- Transferability tab ---
        self.btn_run_xfer.clicked.connect(self._on_xfer_clicked)
        self.btn_xfer_advanced.clicked.connect(self._on_xfer_advanced)
        self.xfer_results_root_btn.clicked.connect(self._on_browse_xfer_root)
        self.btn_xfer_view.clicked.connect(self._on_xfer_view_clicked)

        self.tabs.currentChanged.connect(self._on_tab_changed)


    # ------------------------------------------------------------------
    # Logging / progress helpers
    # ------------------------------------------------------------------
    
    @pyqtSlot(str)
    def _append_log(self, msg: str) -> None:
        # Delegate to LogManager (adds timestamp, collapse, trimming)
        if hasattr(self, "log_mgr"):
            self.log_mgr.append(msg)
        else:
            # Fallback if log_mgr is not available for any reason
            ts = time.strftime("%H:%M:%S")
            self.log_widget.appendPlainText(f"[{ts}] {msg}")

    def log(self, msg: str) -> None:
        """
        Convenience helper: send a message to the GUI log.

        This simply emits the ``log_updated`` signal, which is
        connected to ``_append_log``.
        """
        self.log_updated.emit(msg)

    def _append_status(self, msg: str, error: bool = False) -> None:
        """
        Convenience helper: update status line *and* log.

        Parameters
        ----------
        msg : str
            Message to show.
        error : bool, optional
            If True, prefix with '[ERROR]' in the log/status.
        """
        if error:
            full = f"[ERROR] {msg}"
        else:
            full = msg

        # Update status via signal (label is connected to status_updated)
        self.status_updated.emit(full)

        # Also push to log
        self.log_updated.emit(full)

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
    def _on_quit_clicked(self) -> None:
        """
        Handle Quit button: warn if workflows are running.
        """
        if self._any_job_running():
            msg = (
                "One or more workflows are still running "
                "(Stage-1, training, tuning, inference or transfer matrix).\n\n"
                "If you quit now, they will be interrupted.\n\n"
                "Do you really want to quit?"
            )
            reply = QMessageBox.question(
                self,
                "Quit GeoPrior Forecaster?",
                msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # Best UX when idle: no extra dialog, just quit.
        self.close()

    def _save_gui_log_for_result(self, result: dict) -> None:
        """
        Persist the current GUI log cache into the run's directory.

        Looks for common keys used by the backend helpers:
        - 'run_dir' (training, tuning, inference, stage-1)
        - 'run_output_path' (legacy)
        - 'out_dir' (xfer matrix)
        """
        if not hasattr(self, "log_mgr"):
            return

        run_dir = (
            result.get("run_dir")
            or result.get("run_output_path")
            or result.get("out_dir")
        )
        if not run_dir:
            return

        try:
            log_path = self.log_mgr.save_cache(run_dir)
            self.log_updated.emit(
                f"GUI log saved to:\n  {log_path}"
            )
        except Exception as exc:
            self.log_updated.emit(
                f"[Warn] Could not save GUI log file: {exc}"
            )


    @pyqtSlot()
    def _on_tune_options_clicked(self) -> None:
        """
        Open the Advanced options dialog for tuning.

        Behaviour
        ---------
        - If Stage-1 / training / tuning threads are running, block.
        - Otherwise, open :class:`TuneOptionsDialog`.
        - If user picks a city and clicks 'Run', queue the corresponding
          tuning job and delegate to :meth:`_on_tune_clicked`.
        """
        # Do not allow changing options while long-running jobs are active
        if self.stage1_thread and self.stage1_thread.isRunning():
            QMessageBox.warning(
                self,
                "Stage-1 running",
                "Please wait for Stage-1 to finish before "
                "changing tuning options.",
            )
            return

        if self.train_thread and self.train_thread.isRunning():
            QMessageBox.warning(
                self,
                "Training running",
                "Please wait for training to finish before "
                "changing tuning options.",
            )
            return

        if self.tuning_thread and self.tuning_thread.isRunning():
            QMessageBox.warning(
                self,
                "Tuning running",
                "Hyperparameter search is already running.",
            )
            return

        # Open the advanced tune options dialog
        ok, new_root, job, dev_overrides = TuneOptionsDialog.run(
            cfg=self.geo_cfg,
            gui_runs_root=self.gui_runs_root,
            parent=self,
        )
        self._device_cfg_overrides = dev_overrides or {} 
        
        if not ok:
            return

        # Update GUI-level runs root used by all GUI runs
        self.gui_runs_root = new_root
        self.results_root = new_root  # keep in sync

        # keep Transferability tab in sync with the new root
        if hasattr(self, "xfer_results_root"):
            self.xfer_results_root.setText(str(self.gui_runs_root))
            try:
                self._discover_last_xfer_for_root()
            except Exception as exc:  # defensive
                self.log_updated.emit(
                    f"[WARN] Could not refresh transfer results "
                    f"for new root: {exc}"
                )

        # Keep GeoPriorConfig in sync if it exposes such a field
        if hasattr(self.geo_cfg, "results_root"):
            self.geo_cfg.results_root = new_root

        if job is not None:
            # Make the GUI city match the chosen Stage-1 summary
            self.city_edit.setText(job.stage1.city)

            # Mirror into config if the fields exist
            if hasattr(self.geo_cfg, "CITY_NAME"):
                self.geo_cfg.CITY_NAME = job.stage1.city
            if hasattr(self.geo_cfg, "city_label"):
                self.geo_cfg.city_label = job.stage1.city

            # Cache the job so `_on_tune_clicked` can reuse it
            self._queued_tune_job = job
            # Now delegate to the main tuning entry point
            self._on_tune_clicked()

    @pyqtSlot()
    def _on_infer_options_clicked(self) -> None:
        """
        Open the advanced inference dialog and, if a workflow/model is
        selected, pre-fill the Inference tab with the chosen model and
        Stage-1 manifest.

        Expected API of InferenceOptionsDialog.run:
            accepted, selection = InferenceOptionsDialog.run(
                parent=self,
                results_root=self.gui_runs_root,
                geo_cfg=self.geo_cfg,
            )

        where `selection` exposes at least:
            - selection.city
            - selection.kind          # "train" or "tune"
            - selection.run_dir
            - selection.model_path
            - selection.manifest_path (may be None)
        """
        # Avoid changing configuration while inference is running
        if self.inference_thread and self.inference_thread.isRunning():
            QMessageBox.information(
                self,
                "Inference running",
                "Please wait for the current inference to finish "
                "before changing advanced options.",
            )
            return

        try:
             accepted, new_root, choice = InferenceOptionsDialog.run(
                 parent=self,
                 geo_cfg=self.geo_cfg,
                 results_root=self.gui_runs_root,
             )
        except Exception as exc:  # defensive: don't crash the GUI
            self._append_status(
                f"[Inference options] Failed to open dialog: {exc}",
                error=True,
             )
            return
         
        if not accepted or choice is None:
            # User cancelled or closed the dialog
            return
        # Keep all workflows (train / tune / infer / xfer) in sync
        self.gui_runs_root = new_root
        self.results_root = new_root
        
        # propagate the new root to the Transferability tab
        if hasattr(self, "xfer_results_root"):
            self.xfer_results_root.setText(str(self.gui_runs_root))
            try:
                self._discover_last_xfer_for_root()
            except Exception as exc:  # defensive
                self.log_updated.emit(
                    f"[WARN] Could not refresh transfer results "
                    f"for new root: {exc}"
                )
        
        self._append_status(
            f"[Inference options] Results root set to: {new_root}"
        )
        
        # Pre-fill model + manifest in the Inference tab
        self.inf_model_edit.setText(str(choice.model_path))
        self.inf_manifest_edit.setText(str(choice.stage1_manifest_path))

        # Optional: bring the Inference tab to the front
        try:
            index = self.tabs.indexOf(self.infer_tab)
            if index >= 0:
                self.tabs.setCurrentIndex(index)
        except Exception:
            # If infer_tab isn't stored as an attribute, just ignore
            pass
        
        # Optional: log what we picked
        # kind = getattr(choice, "kind", "?")
        self.log_updated.emit(
            f"Selected inference model: city={choice.city}, "
            f"run_type={choice.run_type}, run={choice.run_name}, "
            f"model={choice.model_path.name}"
        )

        # Make sure user sees they are in the Inference tab
        if hasattr(self, "_infer_tab_index"):
            self.tabs.setCurrentIndex(self._infer_tab_index)

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
        
    @pyqtSlot()
    def _on_arch_config(self) -> None:
        base_cfg = self.geo_cfg._base_cfg or {}

        dlg = ArchitectureConfigDialog(
            base_cfg=base_cfg,
            current_overrides=self.geo_cfg.arch_overrides,
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        self.geo_cfg.arch_overrides = (
            dlg.get_overrides()
        )

        changed = ", ".join(
            sorted(self.geo_cfg.arch_overrides.keys()),
        )
        if not changed:
            changed = "none"

        self.log_updated.emit(
            "Architecture configuration updated "
            f"(keys: {changed}).",
        )

    @pyqtSlot()
    def _on_prob_config(self) -> None:
        base_cfg = self.geo_cfg._base_cfg or {}

        dlg = ProbConfigDialog(
            base_cfg=base_cfg,
            current_overrides=self.geo_cfg.prob_overrides,
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        self.geo_cfg.prob_overrides = (
            dlg.get_overrides()
        )

        changed = ", ".join(
            sorted(self.geo_cfg.prob_overrides.keys()),
        )
        if not changed:
            changed = "none"

        self.log_updated.emit(
            "Probabilistic configuration updated "
            f"(keys: {changed}).",
        )
      
    @pyqtSlot(int)
    def _on_tab_changed(self, index: int) -> None:
        """
        Update the Mode indicator and toggle the log visibility
        when the active tab changes.
        """
        # Update the top 'Mode: ...' indicator
        self._update_mode_button(index)

        # Show log for Train/Tune/Inference/Transfer, hide for Results
        if hasattr(self, "log_widget"):
            if index == getattr(self, "_results_tab_index", -1):
                self.log_widget.setVisible(False)
            else:
                self.log_widget.setVisible(True)


    @pyqtSlot()
    def _on_browse_xfer_root(self) -> None:
        root = QFileDialog.getExistingDirectory(
            self,
            "Select results root (Stage-1/2/xfer)",
            self.xfer_results_root.text() or str(self.gui_runs_root),
        )
        if not root:
            return

        # Normalise to Path and keep the global GUI runs root in sync
        root_path = Path(root)
        self.gui_runs_root = root_path
        self.results_root = root_path

        if hasattr(self.geo_cfg, "results_root"):
            self.geo_cfg.results_root = root_path

        # Update the Transferability line edit
        self.xfer_results_root.setText(str(root_path))

        self.log_updated.emit(
            f"Transferability results root changed to: {root_path}"
        )

        # Re-scan that root for the latest xfer_results.* under xfer/*/*
        self._discover_last_xfer_for_root()


    @pyqtSlot()
    def _on_scalars_config(self) -> None:
        """
        Open the Scalars & loss weights dialog.

        The dialog is pre-populated from the current tuner search space
        (or defaults). When the user clicks OK, values stay in the
        dialog instance and will be picked up by _build_tuner_space_from_ui.
        """
        space = self.geo_cfg.tuner_search_space or default_tuner_search_space()
        defaults = default_tuner_search_space()
        self.scalars_dialog.load_from_space(space, defaults)

        if self.scalars_dialog.exec_() == QDialog.Accepted:
            self.log_updated.emit(
                "Scalars & loss weights updated for tuning."
            )
    
    @pyqtSlot()
    def _on_train_options_clicked(self) -> None:
        """
        Open the advanced train options dialog.
    
        - Lets the user change the results root.
        - Shows existing Stage-1 runs (cities) in that root.
        - Optionally queues a TrainJobSpec when the user clicks
          \"Run now\" on a city card.
        """
        # Do not allow changing options mid-run
        if self.stage1_thread and self.stage1_thread.isRunning():
            QMessageBox.warning(
                self,
                "Stage-1 running",
                "Please wait for Stage-1 to finish before changing options.",
            )
            return
    
        if self.train_thread and self.train_thread.isRunning():
            QMessageBox.warning(
                self,
                "Training running",
                "Please wait for training to finish before changing options.",
            )
            return
    
        accepted, new_root, queued_job, dev_overrides = TrainOptionsDialog.run(
            self,
            geo_cfg=self.geo_cfg,
            results_root=self.gui_runs_root,
        )
        # Store device overrides globally (
        # used later when building cfg_overrides)
        self._device_cfg_overrides = dev_overrides or {}

        if not accepted:
            return
    
        # Update the root used by the GUI for all future runs
        self.gui_runs_root = new_root
        self.results_root = new_root  # keep them in sync for now
        
        # keep the Transferability tab in sync
        if hasattr(self, "xfer_results_root"):
            self.xfer_results_root.setText(str(self.gui_runs_root))
            try:
                self._discover_last_xfer_for_root()
            except Exception as exc:  # defensive, no GUI crash
                self.log_updated.emit(
                    f"[WARN] Could not refresh transfer results "
                    f"for new root: {exc}"
                )
        
        self._append_status(f"[Options] Results root set to: {new_root}")
    
        # If the user clicked \"Run\" on a city card, we receive a TrainJobSpec
        self._queued_train_job = queued_job
    
        if queued_job is not None:
            # Reuse the central entry point so all logic stays in one place
            self._on_train_clicked()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select trained/tuned model",
            str(self.gui_runs_root),
            "Keras models (*.keras *.h5);;All files (*)",
        )
        if path:
            self.inf_model_edit.setText(path)

    @pyqtSlot()
    def _on_browse_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Stage-1 manifest.json",
            str(self.gui_runs_root),
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.inf_manifest_edit.setText(path)

    @pyqtSlot()
    def _on_browse_inputs_npz(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select custom inputs NPZ",
            str(self.gui_runs_root),
            "NumPy archives (*.npz);;All files (*)",
        )
        if path:
            self.inf_inputs_edit.setText(path)

    @pyqtSlot()
    def _on_browse_targets_npz(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select custom targets NPZ (optional)",
            str(self.gui_runs_root),
            "NumPy archives (*.npz);;All files (*)",
        )
        if path:
            self.inf_targets_edit.setText(path)

    @pyqtSlot()
    def _on_browse_calibrator(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select calibrator .npy (optional)",
            str(self.gui_runs_root),
            "NumPy arrays (*.npy);;All files (*)",
        )
        if path:
            self.inf_calib_edit.setText(path)

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

    def _infer_city_name_from_csv(self, csv_path: str) -> str:
        """
        Infer a city/dataset name from the CSV filename.

        Example
        -------
        nansha.csv -> "nansha"
        my city.csv -> "my_city"
        """
        stem = Path(csv_path).stem.strip()
        if not stem:
            return "geoprior_city"

        # Normalise a bit for directory-friendly names
        stem = stem.replace(" ", "_")
        return stem

    def _run_train_dry_preview(self) -> None:
        # Validate basic inputs like the real run
        csv_path_str = str(self.csv_path) if self.csv_path is not None else ""
        if not csv_path_str:
            QMessageBox.warning(
                self,
                "No training CSV",
                "Dry-run: please choose a training CSV file first.",
            )
            return
    
        city = self.city_edit.text().strip()
        if not city and self.csv_path is not None:
            city = self.csv_path.stem
    
        if not city:
            QMessageBox.warning(
                self,
                "Missing city name",
                "Dry-run: please provide a city/dataset name.",
            )
            return
    
        # Sync GUI → config so logs match a real run
        self._sync_config_from_ui()
        cfg = self.geo_cfg
        cfg.TRAIN_CSV_PATH = csv_path_str
        if not getattr(cfg, "CITY_NAME", ""):
            cfg.CITY_NAME = self._infer_city_name_from_csv(csv_path_str)
    
        default_name = cfg.CITY_NAME or "geoprior_run"
        cfg.EXPERIMENT_NAME = self._get_experiment_name(default_name)
    
        # You can still call ensure_valid to reveal config errors
        try:
            cfg.ensure_valid()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Invalid configuration (dry-run)",
                f"The training configuration is not valid:\n\n{exc}",
            )
            return
    
        # Optionally: decide whether Stage-1 would be rebuilt or reused
        # without actually starting threads. For now we just mention the
        # Stage-1 policy flags.
        self._append_status(
            "[Dry-run / Train] Planned workflow:\n"
            f"  City           : {cfg.CITY_NAME}\n"
            f"  CSV            : {csv_path_str}\n"
            f"  Results root   : {self.gui_runs_root}\n"
            f"  Time steps     : {cfg.time_steps}\n"
            f"  Horizon (years): {cfg.forecast_horizon_years}\n"
            f"  Epochs / batch : {cfg.epochs} / {cfg.batch_size}\n"
            f"  PDE mode       : {cfg.pde_mode}\n"
            f"  Clean Stage-1  : {cfg.clean_stage1_dir}\n"
            f"  Build future   : {cfg.build_future_npz}\n"
            f"  experiment    : {cfg.EXPERIMENT_NAME}"
        )
        # --- NEW PART: ask Smart Stage-1 what it would do ---
        need_stage1 = self._smart_stage1_handshake(
            city=city,
            csv_path=csv_path_str,
        )
        manifest_hint = getattr(self, "_stage1_manifest_hint", None)

        if not need_stage1:
            # Either: reuse existing Stage-1, or user canceled
            if manifest_hint is None:
                # Handshake already logged "cancel" – just echo dry-run result.
                self._append_status(
                    "[Dry-run] Result: user would cancel at the "
                    "Stage-1 handshake → no training would run."
                )
                return

            # Reuse existing Stage-1
            self._append_status(
                "[Dry-run] Result: Stage-1 would be reused.\n"
                f"  manifest: {manifest_hint}"
            )
        else:
            # Stage-1 must run (rebuild/new)
            if manifest_hint:
                self._append_status(
                    "[Dry-run] Result: Stage-1 would be rebuilt, "
                    "ignoring existing manifest:\n"
                    f"  existing manifest: {manifest_hint}"
                )
            else:
                self._append_status(
                    "[Dry-run] Result: no compatible Stage-1 found → "
                    "Stage-1 would be run from scratch."
                )

        # Finally, what about Stage-2 training?
        self._append_status(
            "[Dry-run] After Stage-1, a TrainingThread would be started "
            "with the above configuration (but in dry mode, nothing is run)."
        )
        
    @pyqtSlot()
    def _on_train_clicked(self) -> None:
        """
        Main entry point when the user clicks \"Run training\".
    
        Behaviour:
    
        1. If a TrainJobSpec was queued from the Options dialog,
           run that job.
        2. Otherwise, if existing Stage-1 runs are found under the
           current root, show a QuickTrainDialog to let the user
           pick one (shortcut behaviour).
        3. If no jobs exist, fall back to the original smart
           Stage-1 handshake: create Stage-1 if needed, then train.
        """
        # Prevent concurrent runs
        if self.stage1_thread and self.stage1_thread.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "Stage-1 is already running.",
            )
            return
    
        if self.train_thread and self.train_thread.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "Training is already running.",
            )
            return
        
        # Dry-run short-circuit
        if self._is_dry_mode():
            self._run_train_dry_preview()
            return
    
        # 0. If options dialog has queued a specific job, honour it
        if self._queued_train_job is not None:
            job = self._queued_train_job
            self._queued_train_job = None
            self._run_job(job)
            return
    
        # 1. Try quick shortcut: reuse an existing Stage-1 run
        try:
            # Build minimal Stage-1 config snapshot to filter compatible runs
            stage1_cfg = self.geo_cfg.to_stage1_config()
            jobs = latest_jobs_for_root(
                results_root=self.gui_runs_root,
                current_cfg=stage1_cfg,
            )
            quick_job = QuickTrainDialog.choose_job(
                parent=self,
                jobs=jobs,
            )
        except Exception as exc:  # defensive
            self._append_status(
                f"[QuickTrain] Failed to list jobs: {exc}",
                error=True,
            )
            quick_job = None
        
        if quick_job is not None:
            self._run_job(quick_job)
            return

        # 2. No existing jobs: fall back to the original behaviour

        csv_path_str = str(self.csv_path) if self.csv_path is not None else ""
        if not csv_path_str:
            QMessageBox.warning(
                self,
                "No training CSV",
                "Please choose a training CSV file first.",
            )
            return
        
        city = self.city_edit.text().strip()
        if not city and self.csv_path is not None:
            city = self.csv_path.stem
        
        if not city:
            QMessageBox.warning(
                self,
                "Missing city name",
                "Please provide a city/dataset name.",
            )
            return

        cfg = self.geo_cfg
        cfg.TRAIN_CSV_PATH = csv_path_str
    
        # Infer city name from CSV if the user didn't type one
        if not cfg.CITY_NAME:
            cfg.CITY_NAME = self._infer_city_name_from_csv(csv_path_str)
    
        default_name = cfg.CITY_NAME or "geoprior_run"
        cfg.EXPERIMENT_NAME = self._get_experiment_name(default_name)

        try:
            cfg.ensure_valid()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Invalid configuration",
                f"The training configuration is not valid:\n\n{exc}",
            )
            return
    
        # Smart Stage-1 handshake: reuse if matching manifest exists,
        # otherwise build Stage-1 from scratch.
        stage1_cfg, stage1_only = self._smart_stage1_handshake(cfg)
    
        # self.progress.reset()
        self._update_progress(0.0)
        self.status_updated.emit("Stage-1: preparing sequences…")
        self.btn_train.setEnabled(False)
        self.btn_train_options.setEnabled(False)
    
        if stage1_only:
            stage1_thread = Stage1Thread(
                csv_path=csv_path_str,
                cfg_overrides=stage1_cfg,
                parent=self,
            )
            stage1_thread.stage1Finished.connect(
                lambda summary: self._on_stage1_finished(cfg, summary)
            )
            stage1_thread.errorOccurred.connect(self._on_worker_error)
            self.stage1_thread = stage1_thread
            stage1_thread.start()
        else:
            # stage1_cfg is a Stage1Summary in this branch
            summary = stage1_cfg  # type: ignore[assignment]
            self._on_stage1_finished(cfg, summary)
  

    @pyqtSlot()
    def _on_tune_clicked(self) -> None:
        """
        Run Stage-2 hyperparameter tuning for the selected city,
        using TuningThread (non-blocking).

        Priority:
        1) If `_queued_tune_job` is set by the Tune options dialog,
           use its Stage-1 summary.
        2) Otherwise, if no city is typed, open QuickTuneDialog to let
           the user select a Stage-1 city.
        3) Fallback: use the city from the City/Dataset field.
        """
        # Prevent concurrent tuning runs
        if self.tuning_thread and self.tuning_thread.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "Tuning is already running.",
            )
            return

        if hasattr(self, "log_mgr"):
            self.log_mgr.clear()

        # --- dry-run short-circuit ---
        if self._is_dry_mode():
            self._run_tune_dry_preview()
            return
        
        job: TuneJobSpec | None = None
        manifest_path: str | None = None

        # 0) Highest priority: job queued from TuneOptionsDialog
        if self._queued_tune_job is not None:
            job = self._queued_tune_job
            self._queued_tune_job = None

        city_text = self.city_edit.text().strip()

        # 1) If no queued job and no city typed, try QuickTuneDialog
        if job is None and not city_text:
            try:
                ok, quick_job = QuickTuneDialog.run(
                    results_root=self.gui_runs_root,
                    parent=self,
                )
            except Exception as exc:  # defensive
                self._append_status(
                    f"[QuickTune] Failed to list Stage-1 runs: {exc}",
                    error=True,
                )
                ok, quick_job = False, None

            if ok and quick_job is not None:
                job = quick_job

        # 2) If a job was selected (from advanced options or quick dialog),
        #    derive city and manifest from it.
        if job is not None:
            city = job.stage1.city
            manifest_path = str(job.stage1.manifest_path)
            self.city_edit.setText(city)
        else:
            city = city_text

        if not city:
            QMessageBox.warning(
                self,
                "Missing city",
                "Please provide a city/dataset name, or pick one "
                "from the tuning options.",
            )
            return

        # 3) Sync base training config (PDE weights, LR, etc.) from Train tab
        self._sync_config_from_ui()

        # 4) Update tuner search space from Tune tab widgets
        self.geo_cfg.tuner_search_space = self._build_tuner_space_from_ui()

        # 5) Build NAT-style overrides, including TUNER_SEARCH_SPACE
        cfg_overrides = self.geo_cfg.to_cfg_overrides()

        # Inject device overrides chosen in the options dialogs
        if getattr(self, "_device_cfg_overrides", None):
            cfg_overrides.update(self._device_cfg_overrides)

        # 5a) Inject desired max_trials into NAT config
        # run_tuning() will read `TUNER_MAX_TRIALS` from cfg_hp
        # and default to 20 if missing.
        if hasattr(self, "spin_max_trials"):
            cfg_overrides["TUNER_MAX_TRIALS"] = int(
                self.spin_max_trials.value()
            )

        # 5b) Force GUI runs under ~/.fusionlab_runs
        if getattr(self, "gui_runs_root", None) is not None:
            cfg_overrides.setdefault(
                "BASE_OUTPUT_DIR",
                str(self.gui_runs_root),
            )

        # 5c) Inject desired city (Stage-1 for this city must exist)
        cfg_overrides["CITY_NAME"] = city
        self._cfg_overrides = cfg_overrides

        self.log_updated.emit(
            f"Start GeoPrior tuning for city={city!r}."
        )
        self.status_updated.emit(
            f"Stage-2: tuning GeoPrior model for city={city}."
        )
        self._update_progress(0.0)

        eval_tuned = self.chk_eval_tuned.isChecked()

        # 6) Create and start TuningThread
        th = TuningThread(
            manifest_path=manifest_path,  # None -> auto-discover; else fixed
            cfg_overrides=self._cfg_overrides,
            evaluate_tuned=eval_tuned,
            parent=self,
        )
        self.tuning_thread = th

        # Wire signals
        th.log_updated.connect(self.log_updated.emit)
        th.status_updated.connect(self.status_updated.emit)
        th.progress_changed.connect(self._on_thread_progress)
        th.tuning_finished.connect(self._on_tuning_finished)
        th.error_occurred.connect(self._on_worker_error)

        # Disable Tune button while job is active
        self.btn_run_tune.setEnabled(False)

        self._active_job_kind = "tune"
        self._update_global_running_state()   # sets "Running tuning…"

        th.start()
        # after thread is running, ensure Stop button is shown/updated
        self._update_global_running_state()

    def _run_tune_dry_preview(self) -> None:
        """
        Simulate what a tuning run would do, without starting threads.
        """
        if not self.log_mgr:
            return

        self.log_mgr.clear()
        self._append_status(
            "[Dry-run] Previewing GeoPrior tuning workflow ..."
        )

        # 1. Resolve the TuneJobSpec the same way as in _on_tune_clicked
        job = self._queued_tune_job
        self._queued_tune_job = None

        if job is None:
            # Try the quick dialog to let the user pick a Stage-2 run
            ok, quick_job = QuickTuneDialog.run(
                parent=self,
                results_root=self.gui_runs_root,
            )
            if not ok or quick_job is None:
                self._append_status(
                    "[Dry-run] No tuning job selected → nothing would run."
                )
                return
            
            job = quick_job

        stage1 = job.stage1  # depending on your actual API this might be
                             # job.stage1_summary or similar
        city = stage1.city
        manifest_path = getattr(stage1, "manifest_path", None)

        # 2. Sync config / search space as usual
        self._sync_config_from_ui()
        cfg = self.geo_cfg

        search_space = default_tuner_search_space()

        max_trials = getattr(cfg, "TUNER_MAX_TRIALS", 20)

        self._append_status(
            "[Dry-run] Tuning plan:\n"
            f"  city          : {city}\n"
            f"  results_root  : {self.gui_runs_root}\n"
            f"  stage1_root   : {stage1.manifest_path}\n"
            f"  manifest      : {manifest_path}\n"
            f"  max_trials    : {max_trials}\n"
            f"  search keys   : {sorted(search_space.keys())}"
        )

        self._append_status(
            "[Dry-run] A TuningThread would be created with the above "
            "job spec, but in dry mode nothing is started."
        )

    @pyqtSlot()
    def _on_infer_clicked(self) -> None:
        """
        Run Stage-3 inference using InferenceThread (non-blocking).
        """
        if self.inference_thread is not None:
            QMessageBox.information(
                self,
                "Busy",
                "Inference is already running.",
            )
            return
        if hasattr(self, "log_mgr"):
            self.log_mgr.clear()
            
        model_path = self.inf_model_edit.text().strip()
        if not model_path:
            QMessageBox.warning(
                self,
                "Model required",
                "Please select a trained/tuned .keras model first.",
            )
            return

        dataset_key = self.cmb_inf_dataset.currentData() or "test"
        use_future = self.chk_inf_use_future.isChecked()

        manifest_path = self.inf_manifest_edit.text().strip() or None

        inputs_npz: str | None = None
        targets_npz: str | None = None
        if dataset_key == "custom" and not use_future:
            inputs_npz = self.inf_inputs_edit.text().strip() or None
            targets_npz = self.inf_targets_edit.text().strip() or None
            if not inputs_npz:
                QMessageBox.warning(
                    self,
                    "Inputs NPZ required",
                    "For 'Custom NPZ', please select an inputs .npz file.",
                )
                return

        use_source_calibrator = self.chk_inf_use_source_calib.isChecked()
        fit_calibrator = self.chk_inf_fit_calib.isChecked()
        calibrator_path = self.inf_calib_edit.text().strip() or None

        cov_target = float(self.sp_inf_cov.value())
        include_gwl = self.chk_inf_include_gwl.isChecked()
        batch_size = int(self.sp_inf_batch.value())
        make_plots = self.chk_inf_plots.isChecked()

        self.log_updated.emit(
            f"Start inference: model={model_path!r}, "
            f"dataset={dataset_key!r}, use_future={use_future}."
        )
        self.status_updated.emit("Stage-3: running inference.")
        self._update_progress(0.0)

        th = InferenceThread(
            model_path=model_path,
            dataset=dataset_key,
            use_stage1_future_npz=use_future,
            manifest_path=manifest_path,
            stage1_dir=None,
            inputs_npz=inputs_npz,
            targets_npz=targets_npz,
            use_source_calibrator=use_source_calibrator,
            calibrator_path=calibrator_path,
            fit_calibrator=fit_calibrator,
            cov_target=cov_target,
            include_gwl=include_gwl,
            batch_size=batch_size,
            make_plots=make_plots,
            cfg_overrides=None,
            parent=self,
        )
        self.inference_thread = th

        th.log_updated.connect(self.log_updated.emit)
        th.status_updated.connect(self.status_updated.emit)
        th.progress_changed.connect(self._on_thread_progress)
        th.error_occurred.connect(self._on_worker_error)
        th.inference_finished.connect(self._on_inference_finished)

        self.btn_run_infer.setEnabled(False)
        
        self._active_job_kind = "infer"
        self._update_global_running_state()

        th.start()
        self._update_global_running_state()
        
    @pyqtSlot()
    def _on_xfer_advanced(self) -> None:
        dlg = XferAdvancedDialog(
            parent=self,
            quantiles=self._xfer_quantiles_override,
            write_json=self._xfer_write_json,
            write_csv=self._xfer_write_csv,
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        self._xfer_quantiles_override = dlg.get_quantiles()
        self._xfer_write_json = dlg.write_json()
        self._xfer_write_csv = dlg.write_csv()

        self.log_updated.emit(
            "Transferability advanced options updated."
        )

    @pyqtSlot()
    def _on_xfer_clicked(self) -> None:
        """
        Run cross-city transfer matrix using XferMatrixThread.
        """
        if self.xfer_thread is not None:
            QMessageBox.information(
                self,
                "Busy",
                "Transferability is already running.",
            )
            return
        
        if hasattr(self, "log_mgr"):
            self.log_mgr.clear()
            
        city_a = self.xfer_city_a.text().strip()
        city_b = self.xfer_city_b.text().strip() 

        if not city_a or not city_b:
            QMessageBox.warning(
                self,
                "Cities required",
                "Please fill both City A and City B.",
            )
            return

        # Collect splits
        splits: list[str] = []
        if self.chk_xfer_split_train.isChecked():
            splits.append("train")
        if self.chk_xfer_split_val.isChecked():
            splits.append("val")
        if self.chk_xfer_split_test.isChecked():
            splits.append("test")
        if not splits:
            QMessageBox.warning(
                self,
                "Splits required",
                "Please select at least one split.",
            )
            return

        # Collect calibration modes
        calib_modes: list[str] = []
        if self.chk_xfer_cal_none.isChecked():
            calib_modes.append("none")
        if self.chk_xfer_cal_source.isChecked():
            calib_modes.append("source")
        if self.chk_xfer_cal_target.isChecked():
            calib_modes.append("target")
        if not calib_modes:
            QMessageBox.warning(
                self,
                "Calibration modes required",
                "Please select at least one calibration mode.",
            )
            return

        results_root = (
            self.xfer_results_root.text().strip()
            or str(self.gui_runs_root)
        )
        rescale = self.chk_xfer_rescale.isChecked()
        batch_size = int(self.sp_xfer_batch.value())
        quantiles_override = self._xfer_quantiles_override
        write_json = self._xfer_write_json
        write_csv = self._xfer_write_csv

        self.log_updated.emit(
            "Start cross-city transfer matrix: "
            f"{city_a!r} ↔ {city_b!r}; "
            f"splits={splits}, calib={calib_modes}, "
            f"rescale_to_source={rescale}."
        )
        self.status_updated.emit(
            f"XFER: running transfer matrix for {city_a} and {city_b}."
        )
        self._update_progress(0.0)

        th = XferMatrixThread(
            city_a=city_a,
            city_b=city_b,
            results_dir=results_root,
            splits=splits,
            calib_modes=calib_modes,
            rescale_to_source=rescale,
            batch_size=batch_size,
            quantiles_override=quantiles_override,
            out_dir=None,
            write_json=write_json,
            write_csv=write_csv,
            parent=self,
        )
        self.xfer_thread = th

        th.log_updated.connect(self.log_updated.emit)
        th.status_updated.connect(self.status_updated.emit)
        th.progress_changed.connect(self._on_thread_progress)
        th.error_occurred.connect(self._on_worker_error)
        th.xfer_finished.connect(self._on_xfer_finished)

        self.btn_run_xfer.setEnabled(False)
        
        self._active_job_kind = "xfer"
        self._update_global_running_state()

        th.start()
        self._update_global_running_state()
        
        
    @pyqtSlot(dict)
    def _on_xfer_finished(self, result: Dict[str, Any]) -> None:
        """
        Handle completion of XferMatrixThread.
        """
        self.xfer_thread = None
        self.btn_run_xfer.setEnabled(True)

        if not result:
            self.log_updated.emit(
                "Transfer matrix finished with an empty result dict."
            )
            self.status_updated.emit(
                "Transferability failed. See log."
            )
            self._update_progress(0.0)
            return

        out_dir = result.get("out_dir")
        json_path = result.get("json_path")
        csv_path = result.get("csv_path")

        # remember result for the view thread
        self._xfer_last_result = result
        if out_dir:
            self.log_updated.emit(
                "Transferability artifacts in:\n"
                f"  {out_dir}"
            )
            self.lbl_xfer_last_out.setText(out_dir)

        if json_path:
            self.log_updated.emit(
                "Transfer results JSON:\n"
                f"  {json_path}"
            )
        if csv_path:
            self.log_updated.emit(
                "Transfer results CSV:\n"
                f"  {csv_path}"
            )


        self.log_updated.emit(
            "Transfer matrix completed successfully."
        )
        
        # show pretty summary box for the chosen split (default 'val')
        try:
            view_split = (
                self.cmb_xfer_view_split.currentData() or "val"
            )
        except Exception:
            view_split = "val"

        XferResultsDialog.show_for_xfer_result(
            parent=self,
            result=result,
            split=view_split,
            title="Cross-city transfer summary",
        )

        # Let _update_xfer_view_state decide visibility
        self._update_xfer_view_state()
        
        # Save GUI log in xfer out_dir
        self._save_gui_log_for_result(result)
        
        self.status_updated.emit("Idle – transferability complete.")
        self._update_progress(1.0)
        
        self._active_job_kind = None
        self._update_global_running_state()

    @pyqtSlot()
    def _on_xfer_view_clicked(self) -> None:
        if self.xfer_view_thread is not None:
            QMessageBox.information(
                self,
                "Busy",
                "Transferability view is already running.",
            )
            return
    
        if not self._xfer_last_result:
            QMessageBox.warning(
                self,
                "No results",
                "No transfer results found. Run the transfer matrix first.",
            )
            return
    
        view_kind = self.cmb_xfer_view.currentData() or "calib_panel"
        view_split = self.cmb_xfer_view_split.currentData() or "val"
        results_root = (
            self.xfer_results_root.text().strip()
            or str(self.gui_runs_root)
        )
    
        r = self._xfer_last_result or {}
        out_dir = r.get("out_dir")
        csv_path = r.get("csv_path")
        json_path = r.get("json_path")
        
        if not (csv_path or json_path or out_dir):
            QMessageBox.warning(
                self,
                "No artifacts",
                "Could not find xfer_results.* in the last output folder.",
            )
            return

        self.log_updated.emit(
            f"Build transferability view ({view_kind}) "
            f"from {csv_path or json_path or 'latest under results root'}."
        )
        self.status_updated.emit("Rendering transferability view…")
        self._update_progress(0.0)
    
        th = XferViewThread(
            view_kind=view_kind,
            results_root=results_root,
            xfer_out_dir=out_dir,
            xfer_csv=csv_path,
            xfer_json=json_path,
            split=view_split,
            prefer_split=None,
            prefer_calibration=None,
            show_overall=True,
            dpi=300,
            fontsize=12,
            parent=self,
        )
        self.xfer_view_thread = th
    
        th.log_updated.connect(self.log_updated.emit)
        th.status_updated.connect(self.status_updated.emit)
        th.progress_changed.connect(self._on_thread_progress)
        th.error_occurred.connect(self._on_worker_error)
        th.xfer_view_finished.connect(self._on_xfer_view_finished)
    
        self.btn_xfer_view.setEnabled(False)
        th.start()
        
    @pyqtSlot(dict)
    def _on_xfer_view_finished(self, result: Dict[str, Any]) -> None:
        self.xfer_view_thread = None
        self.btn_xfer_view.setEnabled(True)
    
        if not result:
            self.log_updated.emit(
                "Transferability view finished with an empty result dict."
            )
            self.status_updated.emit("View failed. See log.")
            self._update_progress(0.0)
            return
        
        err = result.get("error")
        if err:
            self.log(f"[Xfer view] Error: {err}")
            return
        
        png = result.get("png_path")
        if png:
            self.log_updated.emit(
                "Transferability figure saved:\n"
                f"  {png}"
            )
        for key in ("svg_path", "pdf_path", "table_csv", "table_tex"):
            p = result.get(key)
            if p:
                self.log_updated.emit(f"{key}: {p}")
    
        view_kind = result.get("view_kind", "view")
        split = result.get("split")
        calib = result.get("calibration")
        bits = [view_kind]
        if split:
            bits.append(f"split={split}")
        if calib:
            bits.append(f"calib={calib}")
        self.log_updated.emit("View summary: " + ", ".join(bits))
        
        src = result.get("source_city")
        tgt = result.get("target_city")
        
        self.log(
            "[Xfer view] Transferability figure saved:\n"
            f"  kind   : {view_kind}\n"
            f"  source : {src}\n"
            f"  target : {tgt}\n"
            f"  file   : {png}"
         )

        self.status_updated.emit("Idle – transferability view ready.")
        # auto-preview in GUI (reuses ImagePreviewDialog machinery)
        _notify_gui_xfer_view(result)
        
        self._update_progress(1.0)

    @pyqtSlot()
    def _on_stop_clicked(self) -> None:
        """
        Best-effort global stop for any running workflow.

        It simply calls requestInterruption() on all workflow threads.
        The actual jobs periodically check _stop_check() and exit
        cleanly when the flag is set.
        """
        if not self._any_job_running():
            return

        self.log_updated.emit(
            "Stop requested – attempting to cancel running job(s)…"
        )
        self.status_updated.emit("Stopping… please wait for clean shutdown.")

        for label, th in (
            ("Stage-1", self.stage1_thread),
            ("Training", self.train_thread),
            ("Tuning", self.tuning_thread),
            ("Inference", self.inference_thread),
            ("Transfer matrix", self.xfer_thread),
        ):
            if th is not None and th.isRunning():
                try:
                    th.requestInterruption()
                    self.log_updated.emit(f"  stop requested for {label}.")
                except Exception as exc:
                    self.log_updated.emit(
                        f"  could not interrupt {label}: {exc}"
                    )

        # prevent repeated clicks; will be re-enabled on next run
        self.btn_stop.setEnabled(False)

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
        # Now a job is actually running → show Stop, update labels
        self._update_global_running_state()
       
    def _run_job(self, job: TrainJobSpec) -> None:
        """
        Execute a training job described by TrainJobSpec.
    
        Modes:
        - \"reuse\": skip Stage-1, go straight to training using
          the existing Stage-1 summary.
        - \"rebuild\" / \"scratch\": run Stage-1 first, then training.
          (We treat \"scratch\" as \"rebuild\" here.)
        """
        # Basic config from GUI
        cfg = self.geo_cfg
        # Lock the city to the Stage-1 city of the job
        cfg.CITY_NAME = job.stage1_summary.city

        # --- Only require CSV if we will (re)run Stage-1 -----------------------
        csv_path_str = ""
        if job.mode in {"rebuild", "scratch"}:
            if self.csv_path is not None:
                csv_path_str = str(self.csv_path)
    
            if not csv_path_str:
                QMessageBox.warning(
                    self,
                    "Missing CSV",
                    "This job needs to rebuild Stage-1.\n"
                    "Please choose an input CSV file first.",
                )
                return
    
            cfg.TRAIN_CSV_PATH = csv_path_str
    
        default_name = (
            f"{job.stage1_summary.city}_train_{job.stage1_summary.timestamp}"
        )
        cfg.EXPERIMENT_NAME = self._get_experiment_name(default_name)
        
        try:
            cfg.ensure_valid()
        except Exception as exc:  # pragma: no cover - pure GUI validation
            QMessageBox.critical(
                self,
                "Invalid configuration",
                f"The training configuration is not valid:\n\n{exc}",
            )
            return
    
        # Shared UI setup
        # self.progress.reset()
        self._update_progress(0.0)
        self.btn_train.setEnabled(False)
        self.btn_train_options.setEnabled(False)
    
        self._append_status(
            "[Job] Starting training "
            f"(city={job.stage1_summary.city}, mode={job.mode}, "
            f"root={job.stage1_root})"
        )
    
        # --- Case 1: reuse existing Stage-1, training only -------------------
        if job.mode == "reuse":
            self.status_updated.emit(
                "Stage-2: training GeoPrior model (reusing Stage-1)…"
            )
            # Bypass Stage-1 thread and jump directly to Stage-2
            self._on_stage1_finished(cfg, job.stage1_summary)
            return
    
        # --- Case 2: rebuild/scratch: run Stage-1, then training -------------
        stage1_cfg = cfg.to_stage1_config()
        # Ensure Stage-1 outputs go under the GUI runs root
        stage1_cfg["BASE_OUTPUT_DIR"] = str(self.gui_runs_root)
    
        # Optional hint to the Stage-1 script that this is a rebuild
        if job.mode in {"rebuild", "scratch"}:
            stage1_cfg["FORCE_REBUILD"] = True
    
        self.status_updated.emit("Stage-1: preparing sequences…")
    
        stage1_thread = Stage1Thread(
            csv_path=csv_path_str,
            cfg_overrides=stage1_cfg,
            parent=self,
        )
        stage1_thread.stage1Finished.connect(
            lambda summary: self._on_stage1_finished(cfg, summary)
        )
        stage1_thread.errorOccurred.connect(self._on_worker_error)
    
        self.stage1_thread = stage1_thread
        stage1_thread.start()


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
        device_overrides = getattr(
            self, "_device_cfg_overrides", {}) or {}
        
        cfg_overrides =self._cfg_overrides.update(device_overrides)
        
        th = TrainingThread(
            manifest_path=manifest_path,
            cfg_overrides=cfg_overrides,
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
        self._update_global_running_state()

    @pyqtSlot(dict)
    def _on_training_finished(self, result: Dict[str, Any]) -> None:
        self.train_thread = None

        if not result:
            self.log_updated.emit(
                "Training finished with an empty result dict."
            )
            self.status_updated.emit("Training failed. See log.")
            self.train_btn.setEnabled(True)
            self._active_job_kind = None
            self._update_global_running_state()
            
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
        metrics_json = result.get(
            "metrics_json") or result.get("metrics_json_path")
        if metrics_json or result.get(
                "run_output_path") or result.get("run_dir"):
            try:
                GeoPriorResultsDialog.show_for_result(
                    self,
                    result,
                    title="Training evaluation metrics",
                )
            except Exception as e:
                self._append_log(
                    f"[Warn] Could not open metrics dialog: {e}")

        # Persist GUI log into this training run directory
        self._save_gui_log_for_result(result)
        
        self.log_updated.emit("Training completed successfully.")
        self.status_updated.emit("Idle – training complete.")
        self._update_progress(1.0)
        
        self.train_btn.setEnabled(True)
        self._active_job_kind = None
        self._update_global_running_state()

        
    @pyqtSlot(str)
    def _on_worker_error(self, message: str) -> None:
        self.log_updated.emit(f"[ERROR] {message}")
        QMessageBox.critical(self, "Error", message)

        # Reset all threads/buttons that might be active
        self.train_btn.setEnabled(True)
        
        self.btn_run_tune.setEnabled(True)
        if hasattr(self, "btn_run_infer"):
            self.btn_run_infer.setEnabled(True)
        if hasattr(self, "btn_run_xfer"):
            self.btn_run_xfer.setEnabled(True)
        if hasattr(self, "btn_xfer_view"):
            self.btn_xfer_view.setEnabled(True)

        self.stage1_thread = None
        self.train_thread = None
        self.tuning_thread = None
        self.inference_thread = None
        self.xfer_thread = None
        self.xfer_view_thread = None
        
        self._active_job_kind = None
        self._update_global_running_state()
        
        self._update_progress(0.0)
        self.progress_label.setText("")


    @pyqtSlot(dict)
    def _on_tuning_finished(self, result: Dict[str, Any]) -> None:
        """Handle completion of TuningThread."""
        self.tuning_thread = None
        self.btn_run_tune.setEnabled(True)

        if not result:
            self.log_updated.emit(
                "Tuning finished with an empty result dict."
            )
            self.status_updated.emit("Tuning failed. See log.")
            self._update_progress(0.0)
            return

        run_dir = result.get("run_dir")
        best_hps_path = result.get("best_hps_path")
        best_model_path = result.get("best_model_path")
        best_weights_path = result.get("best_weights_path")

        if run_dir:
            self.log_updated.emit(
                "Tuning artifacts in:\n"
                f"  {run_dir}"
            )
        if best_hps_path:
            self.log_updated.emit(
                "Best hyperparameters JSON:\n"
                f"  {best_hps_path}"
            )
        if best_model_path:
            self.log_updated.emit(
                "Best tuned model:\n"
                f"  {best_model_path}"
            )
        if best_weights_path:
            self.log_updated.emit(
                "Best tuned weights:\n"
                f"  {best_weights_path}"
            )
        metrics_json = result.get(
            "metrics_json") or result.get(
                "metrics_json_path")
        if metrics_json or result.get(
                "run_output_path") or result.get("run_dir"):
            try:
                GeoPriorResultsDialog.show_for_result(
                    self,
                    result,
                    title="Tuned model evaluation",
                )
            except Exception as e:
                self._append_log(
                    f"[Warn] Could not open metrics dialog: {e}"
                )
        # Save GUI log in tuning run_dir
        self._save_gui_log_for_result(result)
        
        self.log_updated.emit("Tuning completed successfully.")
        self.status_updated.emit("Idle – tuning complete.")
        self._update_progress(1.0)
        
        self._active_job_kind = None
        self._update_global_running_state()
        
    @pyqtSlot(dict)
    def _on_inference_finished(self, result: Dict[str, Any]) -> None:
        """
        Handle completion of InferenceThread.
        """
        self.inference_thread = None
        self.btn_run_infer.setEnabled(True)

        if not result:
            self.log_updated.emit(
                "Inference finished with an empty result dict."
            )
            self.status_updated.emit("Inference failed. See log.")
            self._update_progress(0.0)
            return

        run_dir = result.get("run_dir")
        csv_eval = result.get("csv_eval_path")
        csv_future = result.get("csv_future_path")
        summary_json = result.get("inference_summary_json")

        coverage80 = result.get("coverage80")
        sharpness80 = result.get("sharpness80")
        point_phys = result.get("point_metrics_phys") or {}
        r2_phys = None
        if isinstance(point_phys, dict):
            r2_phys = point_phys.get("r2")

        if run_dir:
            self.log_updated.emit(
                "Inference artifacts in:\n"
                f"  {run_dir}"
            )
        if csv_eval:
            self.log_updated.emit(
                "Evaluation CSV:\n"
                f"  {csv_eval}"
            )
        if csv_future:
            self.log_updated.emit(
                "Future forecast CSV:\n"
                f"  {csv_future}"
            )
        if summary_json:
            self.log_updated.emit(
                "Inference summary JSON:\n"
                f"  {summary_json}"
            )

        metrics_bits = []
        try:
            if coverage80 is not None:
                metrics_bits.append(f"cov80={float(coverage80):.3f}")
        except Exception:
            pass
        try:
            if sharpness80 is not None:
                metrics_bits.append(f"sharp80={float(sharpness80):.2f}")
        except Exception:
            pass
        try:
            if r2_phys is not None:
                metrics_bits.append(f"R²_phys={float(r2_phys):.3f}")
        except Exception:
            pass

        if metrics_bits:
            self.log_updated.emit(
                "Inference summary metrics: " + ", ".join(metrics_bits)
            )
        # Save GUI log next to inference outputs
        self._save_gui_log_for_result(result)
        
        self.log_updated.emit("Inference completed successfully.")
        self.status_updated.emit("Idle – inference complete.")
        self._update_progress(1.0)
        
        self._active_job_kind = None
        self._update_global_running_state()
        
    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Intercept window-close to warn if workflows are active.
        """
        if self._any_job_running():
            msg = (
                "One or more workflows are still running "
                "(Stage-1, training, tuning, inference or transfer matrix).\n\n"
                "If you quit now, they will be interrupted.\n\n"
                "Do you really want to quit?"
            )
            reply = QMessageBox.question(
                self,
                "Quit GeoPrior Forecaster?",
                msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()


# ----------------------------------------------------------------------
# Entry point helper
# ----------------------------------------------------------------------

def launch_geoprior_gui(theme: str = "fusionlab") -> None:
    
    app = QApplication(sys.argv)
    cfg = GeoPriorConfig.from_nat_config()
    auto_set_ui_fonts(app)
    
    enable_qt_crash_handler(app, keep_gui_alive=False)  # nice tracebacks if something dies
    # --- create splash with your logo ---
    logo_path = Path(__file__).with_name("geoprior_splash.png")
    splash = LoadingSplash(logo_path)
    splash.show()
    app.processEvents()

    splash.set_progress(10, "Loading configuration…")

    scale = float(getattr(cfg, "ui_font_scale", 1.0))
    if scale != 1.0:
        f = app.font()
        f.setPointSizeF(max(6.0, f.pointSizeF() * scale))
        app.setFont(f)
    
    # pass splash into the main window so it can update during _build_ui()
    gui = GeoPriorForecaster(theme=theme, splash=splash)  

    splash.set_progress(100, "Ready")
    splash.finish(gui)   # hides splash when gui is shown

    gui.show()
    sys.exit(app.exec_())



if __name__ == "__main__":
    # High-DPI attributes should be set before the QApplication exists
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    launch_geoprior_gui()
