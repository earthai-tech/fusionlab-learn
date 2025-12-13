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

from PyQt5.QtCore import ( 
    Qt, 
    pyqtSignal, 
    pyqtSlot, 
    QSize, 
    QPoint, 
    QTimer, 
)
from PyQt5.QtGui import (
    QCloseEvent,
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QIcon, 
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
    QStyle
)


from .workflows.base import RunEnv, GUIHooks
from .workflows.train import TrainController, TrainGuiState
from .services.stage1_service import Stage1Service
from .workflows.tune import TuneController, TuneGuiState
from .workflows.inference import (
    InferenceController,
    InferenceGuiState,
    InferencePlan,
)

from .workflows.transfer import (
    TransferController,
    TransferGuiState,
    TransferPlan,
)
from .ui.mode_manager import ModeManager
from .services.results_service import ResultsService
from .ui.file_browse import FileBrowseHelper
from .ui.menu_manager import MenuManager
from .ui.splash import LoadingSplash
from .ui.tools_tab import ToolsTab

     
from ..ux_utils import (
    auto_set_ui_fonts,
    auto_resize_window,
    enable_qt_crash_handler,
)



from .utils.view_signals import VIS_SIGNALS 
from .utils.gui_popups import ImagePreviewDialog 

from .threads import (
    Stage1Thread,
    TrainingThread,
    TuningThread,
    InferenceThread,
    XferMatrixThread,
    XferViewThread
)


from .config import ( 
    GeoPriorConfig,
    default_tuner_search_space,
    find_stage1_for_city
)

from .dialogs import ( 
    FeatureConfigDialog,
    ArchitectureConfigDialog , 
    ProbConfigDialog ,
    XferAdvancedDialog, 
    XferResultsDialog, 
    InferenceOptionsDialog, 
    GeoPriorResultsDialog,
    PhysicsConfigDialog,
    PHYSICS_DEFAULTS,
    ScalarsLossDialog,
    ModelParamsDialog,
    TrainOptionsDialog, 
    QuickTrainDialog,
    TuneOptionsDialog, 
    QuickTuneDialog,
    Stage1ChoiceDialog, 
    TuneJobSpec,
    open_dataset_with_editor,
    choose_dataset_for_city
)

from .about import show_about_dialog, DOCS_URL

from .utils.view_utils import _notify_gui_xfer_view
from .utils.clock_timer import RunClockTimer

from .styles import (
    TAB_STYLES,
    LOG_STYLES,
    FLAB_STYLE_SHEET,
    PRIMARY,
    RUN_BUTTON_IDLE,
)

from .jobs import TrainJobSpec, latest_jobs_for_root
from .results_tab import ResultsDownloadTab
from .ui.log_manager import LogManager
from .utils.components import RangeListEditor


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
        self._splash = splash

        # 1) Core state / config / help texts
        self._update_splash(10, "Initialising core state…")
        self._init_core_state()
        self._init_help_texts()

        # 2) Dialogs + runs root
        self._update_splash(20, "Loading defaults…")
        self._init_dialogs_and_paths()

        # 3) Build UI skeleton (widgets, tabs, log, progress bar)
        self._update_splash(40, "Building UI…")
        self._build_ui()

        # 4) Workflow controllers + managers (ModeManager, FileBrowseHelper)
        self._update_splash(55, "Initialising workflows…")
        self._init_workflows()

        # 5) Menus
        self._update_splash(65, "Building menu bar…")
        self._build_menu_bar()

        # 6) Log manager
        self._update_splash(75, "Preparing log manager…")
        self._init_log_manager()

        # 7) Signals, window props, misc hooks
        self._update_splash(85, "Connecting signals…")
        self._connect_signals()

        self._update_splash(92, "Applying window settings…")
        self._set_window_props()

        self._update_splash(98, "Finalising…")
        self._post_init_misc()

    # ------------------------------------------------------------------
    # Splash helper
    # ------------------------------------------------------------------
    def _update_splash(self, value: int, msg: str) -> None:
        """Safe wrapper to update the loading splash, if present."""
        if self._splash is not None:
            self._splash.set_progress(value, msg)

    def _init_core_state(self) -> None:
        """Initialise core attributes that don't touch Qt widgets."""
        # Threads
        self.stage1_thread: Stage1Thread | None = None
        self.train_thread: TrainingThread | None = None
        self.tuning_thread: TuningThread | None = None
        self.inference_thread: InferenceThread | None = None
        self.xfer_thread: XferMatrixThread | None = None
        self.xfer_view_thread: XferViewThread | None = None

        # Last transferability result
        self._xfer_last_result: Dict[str, Any] | None = None

        # Jobs queued from options/quick dialogs
        self._queued_train_job: TrainJobSpec | None = None
        self._queued_tune_job: TuneJobSpec | None = None

        # Global run/stop state
        self._active_job_kind: str | None = None  # "train", "tune", "infer", "xfer"

        # Config overrides and device overrides
        self.csv_path: Path | None = None
        self._cfg_overrides: Dict[str, Any] = {}
        self._device_cfg_overrides: Dict[str, Any] = {}

        # Central config object (defaults loaded from nat.com/config.py)
        self.geo_cfg = GeoPriorConfig.from_defaults()

        # Stage-1 manifest hint (used by train/infer flows)
        self._stage1_manifest_hint: Path | None = None

        # XFER advanced options
        self._xfer_quantiles_override = None
        self._xfer_write_json = True
        self._xfer_write_csv = True

        # DataFrame edited in the GUI (e.g. open dataset)
        self._edited_df = None  # : pd.DataFrame | None
        
        self._feature_tip_shown: bool = False
        
        # Map lowercased city name -> manifest path (string)
        self._preferred_stage1_by_city: Dict[str, str] = {}


    def _init_help_texts(self) -> None:
        """Initialise per-tab help texts and one-shot flags."""
        self._train_help_text = ModeManager.DEFAULT_HELP_TEXTS["train"]
        self._train_tip_shown = False

        self._tune_help_text = ModeManager.DEFAULT_HELP_TEXTS["tune"]
        self._tune_tip_shown = False

        self._infer_help_text = ModeManager.DEFAULT_HELP_TEXTS["infer"]
        self._infer_tip_shown = False

        self._xfer_help_text = ModeManager.DEFAULT_HELP_TEXTS["xfer"]
        self._xfer_tip_shown = False

        self._results_help_text = ModeManager.DEFAULT_HELP_TEXTS["results"]
        self._results_tip_shown = False


    def _init_dialogs_and_paths(self) -> None:
        """Initialise dialogs and main results root paths."""
        # Dialog for scalar HPs / loss weights (used from Tune tab)
        self.scalars_dialog = ScalarsLossDialog(self)

        # Dialog for model-level HPs (memory, scales, attention, etc.)
        self.model_params_dialog = ModelParamsDialog(self)

        # Dedicated root so GUI runs don't mix with CLI results
        self.gui_runs_root = Path.home() / ".fusionlab_runs"
        self.gui_runs_root.mkdir(parents=True, exist_ok=True)

        # User-overrideable base results root (defaults to gui_runs_root)
        self.results_root = self.gui_runs_root

    def _init_log_manager(self) -> None:
        """Create the central LogManager once the log widget exists."""
        self.log_mgr = LogManager(
            self.log_widget,
            mode="collapse",
            log_dir_name="_log",
        )

    def _post_init_misc(self) -> None:
        """Final small initialisations that depend on everything else."""
        # Keep references to non-modal preview dialogs
        self._preview_windows: list[QDialog] = []

        VIS_SIGNALS.figure_saved.connect(self._show_image_popup)

    def _show_image_popup(self, png_path: str) -> None:
        dlg = ImagePreviewDialog(png_path, parent=self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)             # frees memory on close
        dlg.show()                                        # modeless – *no* exec_()
        self._preview_windows.append(dlg)                 # keep it alive


    def _init_workflows(self) -> None:
        """
        Initialise shared RunEnv/GUIHooks, Stage1Service and TrainController.

        This is called once from __init__ after geo_cfg and
        gui_runs_root are set.
        """
        # 1) Main workflow environment (shared for Train/Tune/... later)
        # --- Shared RunEnv and GUIHooks for all workflows -------------------
        self._run_env = RunEnv(
            gui_runs_root=Path(self.gui_runs_root),
            geo_cfg=self.geo_cfg,
            device_overrides=dict(getattr(self, "_device_cfg_overrides", {}) or {}),
            dry_mode=False,
        )
        
        self._gui_hooks = GUIHooks(
            log=self._append_status,             # or self.log_updated.emit
            status=self.status_updated.emit,
            update_progress=self._update_progress,
            ask_yes_no=self._ask_yes_no,         # small wrappers around QMessageBox
            warn=self._warn_dialog,
            error=self._error_dialog,
        )
        # 3) Stage-1 specific env + hooks for Stage1Service
        self._stage1_service = Stage1Service(
            env=self._run_env,
            hooks=self._gui_hooks,
            find_stage1_for_city=find_stage1_for_city,
            chooser=self._choose_stage1_run,
        )
        # 4) Training controller (real + dry)
        self.train_controller = TrainController(
            env=self._run_env,
            hooks=self._gui_hooks,
            stage1_svc=self._stage1_service,
        )
        
        self.tune_controller = TuneController(
            env=self._run_env,
            hooks=self._gui_hooks,
        )
        
        self.infer_controller = InferenceController(
            env=self._run_env,
            hooks=self._gui_hooks,
        )
        self.transfer_controller = TransferController(
            self._run_env, self._gui_hooks
             )
        
        # --------------------------------------------------------------
        # Mode / Stop manager
        # --------------------------------------------------------------
        self.mode_mgr = ModeManager(
            mode_btn=self.mode_btn,
            btn_stop=self.btn_stop,
            make_play_icon=self._make_play_icon,
            stop_pulse_timer=self._stop_pulse_timer,
            parent=self,
        )

        self.mode_mgr.set_run_buttons(
            train_btn=self.train_btn,
            tune_btn=self.btn_run_tune,
            infer_btn=self.btn_run_infer,
            xfer_btn=self.btn_run_xfer,
        )

        # Initial mode + running state (no job running at startup)
        self.mode_mgr.update_running_state(any_running=False)
        self._update_mode_button(self.tabs.currentIndex())

        # ------------------------------------------------------------------
        # File browser helper
        # ------------------------------------------------------------------
        self.file_browse = FileBrowseHelper(
            parent=self,
            root_getter=lambda: self.gui_runs_root,
        )

        # ----------------------------------------------------------
        # Results helper service
        # ----------------------------------------------------------
        self.results_svc = ResultsService(
            log=self.log_updated.emit  
        )
        
    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------
    def _build_menu_bar(self) -> None:
        """
        Delegate menu creation to MenuManager.

        Keeping this method as a thin wrapper avoids touching
        call sites that expect `_build_menu_bar` to exist.
        """
        self.menu_mgr = MenuManager(
            window=self,
            mode_mgr=self.mode_mgr,
            docs_url=DOCS_URL,
        )
        self.menu_mgr.build()
  
    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _set_window_props(self) -> None:
        self.setWindowTitle("GeoPrior-3.0 Forecaster")
    
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
        ico = ico_dir / "geoprior_logo.ico"
        if ico.exists():
            self.setWindowIcon(QIcon(str(ico)))
    
        # Apply Fusionlab stylesheet (tabs / cards / log)
        self.setStyleSheet(FLAB_STYLE_SHEET + TAB_STYLES + LOG_STYLES)
        
    def _build_ui(self) -> None: 
        root = QWidget(self)
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        
        self._update_splash(42, "Building top toolbar…") 
        
        # --- Top row: [Select CSV…] [City / Dataset] [Dry run] [Mode] [Quit] ---
        top = QHBoxLayout()
    
        
        self.select_csv_btn = QPushButton("Open dataset…")
        top.addWidget(self.select_csv_btn)

        label = QLabel("City / Dataset:")
        # label.setStyleSheet("font-weight: 600;")
        top.addWidget(label)
       
        self.city_edit = QLineEdit()
        self.city_edit.setPlaceholderText("e.g. nansha")
        top.addWidget(self.city_edit, 1)
        
        self.city_edit.setStyleSheet(
            """
            QLineEdit#cityDatasetEdit {
                background-color: #fff9f0;
                border: 1px solid #f0b96a;
                border-radius: 8px;
                padding: 3px 8px;
                font-weight: 600;
            }
            QLineEdit#cityDatasetEdit:focus {
                border: 1px solid #e8902f;
                background-color: #fff3e0;
            }
            """
        )

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
        self.btn_stop.setObjectName("stopButton")
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        top.addWidget(self.btn_stop)
        
        # --- NEW: base + pulse styles for the Stop button ---
        self._stop_base_style = """
        QPushButton#stopButton {
            background-color: #d9534f;   /* red */
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 4px 16px;
        }
        QPushButton#stopButton:hover {
            background-color: #c9302c;
        }
        QPushButton#stopButton:disabled {
            background-color: #a0a0a0;
            color: #eeeeee;
        }
        """
        
        # Slightly lighter / stronger red for the "pulse" frame
        self._stop_pulse_style = """
        QPushButton#stopButton {
            background-color: #ff6f6f;
            color: white;
            font-weight: 700;
            border-radius: 10px;
            padding: 4px 16px;
        }
        """
        
        # Apply the base style once
        self.btn_stop.setStyleSheet(self._stop_base_style)
        
        # Timer driving the pulse effect
        self._stop_pulse_timer = QTimer(self)
        self._stop_pulse_timer.setInterval(450)  # ms between frames
        self._stop_pulse_state = False

        # Mode indicator button (updated when tab changes)
        self.mode_btn = QPushButton("Mode: Train")
        self.mode_btn.setEnabled(False)
        self.mode_btn.setFlat(True)
        top.addWidget(self.mode_btn)
        
        layout.addLayout(top)
        # --- Tabs row: Train / Tune / Inference / Transferability ---
        self._update_splash(45, "Building tabs…")
        
        self.tabs = QTabWidget()
        self._init_tabs()
        layout.addWidget(self.tabs, 1)

        # --- Status line ---
        status_row = QHBoxLayout()

        self.status_label = QLabel("? Idle")
        self.status_label.setStyleSheet(f"color:{PRIMARY};")
        status_row.addWidget(self.status_label, 1)  # takes the left side
        
        status_row.addStretch(1)  # push timer fully to the right
        
        # digital run timer (black background, green digits)
        self.run_timer = RunClockTimer(self)
        self.run_timer.reset()
        self.run_timer.stop()
        self.run_timer.setVisible(False)
        status_row.addWidget(self.run_timer, 0)
        
        layout.addLayout(status_row)
        
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
        
        self.physics_btn = QPushButton("Physics config…")
        buttons_row.addWidget(self.physics_btn)
         
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
        
        # 
        self.btn_model_params = QPushButton("Model params…")
        opts_row.addWidget(self.btn_model_params)
        
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

        # Results tab – browse & download artifacts/runs
        results_tab = ResultsDownloadTab(
            results_root=self.gui_runs_root,
            get_results_root=lambda: self.gui_runs_root,
            parent=self,
        )
        self.results_tab = results_tab

        self._train_tab_index = self.tabs.addTab(
            self.train_tab,
            self._workflow_icon("train.svg", QStyle.SP_ComputerIcon),
            "Train",
        )

        self._tune_tab_index = self.tabs.addTab(
            self.tune_tab,
            self._workflow_icon("tune.svg", QStyle.SP_FileDialogDetailedView),
            "Tune",
        )

        self._infer_tab_index = self.tabs.addTab(
            self.infer_tab,
            self._workflow_icon("inference.svg", QStyle.SP_FileDialogListView),
            "Inference",
        )

        self._xfer_tab_index = self.tabs.addTab(
            self.xfer_tab,
            self._workflow_icon("transfer.svg", QStyle.SP_ArrowRight),
            "Transfer",
        )

        self._results_tab_index = self.tabs.addTab(
            self.results_tab,
            self._workflow_icon("results.svg", QStyle.SP_DirHomeIcon),
            "Results",
        )
        # Tools tab – utilities (data, manifests, diagnostics, environment)
        self.tools_tab = ToolsTab(
            app_ctx=self,   
            geo_cfg=self.geo_cfg,
            gui_runs_root=self.gui_runs_root,
            parent=self,
        )

        self._tools_tab_index = self.tabs.addTab(
            self.tools_tab,
            self._workflow_icon("tools.svg", QStyle.SP_FileDialogInfoView),
            "Tools",
        )

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

    # ------------------------------------------------------------------
    # Small QMessageBox / dialog helpers for GUIHooks
    # ------------------------------------------------------------------
    def _warn_dialog(self, title: str, msg: str) -> None:
        QMessageBox.warning(self, title, msg)

    def _error_dialog(self, title: str, msg: str) -> None:
        QMessageBox.critical(self, title, msg)

    def _ask_yes_no(self, title: str, question: str) -> bool:
        reply = QMessageBox.question(
            self,
            title,
            question,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _std_icon(self, sp: QStyle.StandardPixmap) -> QIcon:
        """
        Convenience wrapper around style().standardIcon(), so that we
        can keep icon usage consistent and later swap to custom icons
        in a single place if needed.
        """
        return self.style().standardIcon(sp)
    
    def _icon(self, name: str) -> QIcon:
        """Load a named icon from our app's icon bundle."""
        base = Path(__file__).resolve().parent / "icons"
        path = base / name
        if path.exists():
            return QIcon(str(path))
        # Fallback to a neutral standard icon or empty
        return QIcon()
    
    def _workflow_icon(
        self,
        svg_name: str,
        fallback: QStyle.StandardPixmap,
    ) -> QIcon:
        """
        Prefer a custom SVG icon from geoprior/icons/, with a Qt
        standard pixmap as fallback.
        """
        icon = self._icon(svg_name)
        if not icon.isNull():
            return icon
        return self._std_icon(fallback)

    def _choose_stage1_run(
        self,
        city: str,
        runs_for_city,
        all_runs,
        clean_stage1: bool,
    ):
        """
        Thin wrapper around Stage1ChoiceDialog.ask so Stage1Service
        doesn't depend on Qt directly.
        """
        return Stage1ChoiceDialog.ask(
            parent=self,
            city=city,
            runs_for_city=runs_for_city,
            all_runs=all_runs,
            clean_stage1=clean_stage1,
        )
    def _on_show_about(self) -> None:
        """Small wrapper so MenuManager can call About without extra imports."""
        show_about_dialog(self)

    # --------------------------------------------------------------
    # Run timer helpers
    # --------------------------------------------------------------
    def _start_run_timer(self) -> None:
        """
        Restart the digital run timer when a new job starts.

        Ensures the widget is visible, fully opaque and counting
        from zero.
        """
        timer = getattr(self, "run_timer", None)
        if timer is None:
            return

        timer.setVisible(True)
        # full opacity, cancel any pending dim
        timer.cancel_hibernate()   
        timer.restart()

    def _stop_run_timer(self) -> None:
        """
        Stop (but do not hide) the run timer when a job finishes
        or is aborted, then gently dim it after a delay.
        """
        timer = getattr(self, "run_timer", None)
        if timer is None:
            return

        timer.stop()
        # Keep the last elapsed value visible, then fade after 60 s
        timer.schedule_hibernate(timeout_ms=60_000)

    def _reset_run_timer(self) -> None:
        """
        Reset the run timer when a job finishes or is aborted,
        then gently dim it after a delay.
        """
        timer = getattr(self, "run_timer", None)
        if timer is None:
            return

        timer.reset()
        timer.schedule_hibernate(timeout_ms=60_000)


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

    def _discover_last_xfer_for_root(self) -> None:
        """
        Best-effort discovery of the latest transferability run
        under the current xfer results root.
    
        Keeps `_xfer_last_result` consistent with whatever is
        already on disk, but delegates the actual search to
        ResultsService.
        """
        root_text = self.xfer_results_root.text().strip()
        if not root_text:
            # Nothing to search
            self._xfer_last_result = {}
            self.lbl_xfer_last_out.setText("No transfer run yet.")
            self._update_xfer_view_state()
            return
    
        result = self.results_svc.discover_last_xfer(Path(root_text))
    
        # Store for later (view button, dialogs, etc.)
        self._xfer_last_result = result or {}
    
        if not result:
            self.lbl_xfer_last_out.setText("No transfer run yet.")
        else:
            self.lbl_xfer_last_out.setText(result["out_dir"])
    
        self._update_xfer_view_state()

    # ------------------------------------------------------------------
    # Logging / progress helpers
    # ------------------------------------------------------------------

    def set_console_visible(self, visible: bool) -> None:
        """
        Show or hide the bottom log console.

        Tools / tabs that do not need the shared log (for example the
        Tools → Dataset explorer) can call this helper so the central
        workspace uses the full height.
        """
        if hasattr(self, "log_widget") and self.log_widget is not None:
            self.log_widget.setVisible(visible)


    def _is_dry_mode(self) -> bool:
        chk = getattr(self, "chk_dry_run", None)
        return bool(chk is not None and chk.isChecked())

    # ------------------------------------------------
    #   Update buttons 
    # ------------------------------------------------
    def _update_mode_button(self, index: int) -> None:
        """
        Thin wrapper: delegate mode badge update to ModeManager.
        """
        if not hasattr(self, "tabs") or self.tabs is None:
            return

        help_texts = {
            "train": (
                getattr(self, "_train_tab_index", -1),
                getattr(
                    self,
                    "_train_help_text",
                    ModeManager.DEFAULT_HELP_TEXTS["train"],
                ),
            ),
            "tune": (
                getattr(self, "_tune_tab_index", -1),
                getattr(
                    self,
                    "_tune_help_text",
                    ModeManager.DEFAULT_HELP_TEXTS["tune"],
                ),
            ),
            "infer": (
                getattr(self, "_infer_tab_index", -1),
                getattr(
                    self,
                    "_infer_help_text",
                    ModeManager.DEFAULT_HELP_TEXTS["infer"],
                ),
            ),
            "xfer": (
                getattr(self, "_xfer_tab_index", -1),
                getattr(
                    self,
                    "_xfer_help_text",
                    ModeManager.DEFAULT_HELP_TEXTS["xfer"],
                ),
            ),
            "results": (
                getattr(self, "_results_tab_index", -1),
                getattr(
                    self,
                    "_results_help_text",
                    ModeManager.DEFAULT_HELP_TEXTS["results"],
                ),
            ),
        }

        self.mode_mgr.update_for_tab(index, self.tabs, help_texts)

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
        """
        Delegate all mode / stop / run-button styling to ModeManager.
        """
        any_running = self._any_job_running()
        self.mode_mgr.set_active_job_kind(self._active_job_kind)
        self.mode_mgr.set_dry_mode(self._is_dry_mode())
        self.mode_mgr.update_running_state(any_running)


    
    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.select_csv_btn.clicked.connect(
            self._on_open_dataset,
        )
        self.train_btn.clicked.connect(
            self._on_train_clicked,
        )

        # Connect Stop button to the manager, and manager → "real" stop logic
        self.btn_stop.clicked.connect(self.mode_mgr.on_stop_clicked)
        self.mode_mgr.stop_requested.connect(self._on_stop_requested)

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
        self.physics_btn.clicked.connect(
            self._on_physics_config_clicked
        )
    
        self.btn_tune_options.clicked.connect(
            self._on_tune_options_clicked,
        )

        self.btn_run_tune.clicked.connect(
            self._on_tune_clicked,
        )

        self.btn_model_params.clicked.connect(
            self._on_model_params_config,
        )
        
        self.btn_scalars.clicked.connect(
            self._on_scalars_config,
        )
        
        self.btn_train_options.clicked.connect(
            self._on_train_options_clicked
        )
    
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

    @pyqtSlot(bool)
    def _on_dry_run_toggled(self, checked: bool) -> None:
        # Update ModeManager's state
        self.mode_mgr.set_dry_mode(checked)
    
        # Refresh mode badge + buttons for current tab
        current_index = self.tabs.currentIndex()
        self._update_mode_button(current_index)
        self._update_global_running_state()
    
        # Optional UX message
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
    
                
    def _on_tab_changed(self, index: int) -> None:
        """
        Update the Mode indicator and toggle the log visibility
        when the active tab changes.
        """
        self._update_mode_button(index)

        if not hasattr(self, "log_widget"):
            return

        results_idx = getattr(self, "_results_tab_index", -1)
        tools_idx   = getattr(self, "_tools_tab_index", -1)

        # Train / Tune / Inference / Transfer → console visible
        # Results → console hidden
        # Tools   → hidden by default; individual tools can override.
        if index == results_idx:
            self.set_console_visible(False)
        elif index == tools_idx:
            self.set_console_visible(False)   # default for Tools
        else:
            self.set_console_visible(True)
         
    # ------------------------------------------------------------------
    # Logging / progress helpers
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_stop_requested(self) -> None:
        """
        Actual cancellation logic for running threads.
    
        UI aspects (button disable, pulse, etc.) are handled
        by ModeManager.
        """
        if not self._any_job_running():
            return
    
        self.log_updated.emit(
            "Stop requested – attempting to cancel running job(s)…"
        )
        self.status_updated.emit(
            "Stopping… please wait for clean shutdown."
        )
    
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
    
        # Optional: stop the run-duration timer here, like before
        self._stop_run_timer()

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


    def _save_gui_log_for_result(self, result: dict) -> None:
        """
        Forward GUI log persistence to ResultsService.
    
        This keeps all the 'where do I save logs?' logic in one place.
        """
        log_mgr = getattr(self, "log_mgr", None)
        self.results_svc.save_gui_log(log_mgr, result)

    
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

        # This is the missing piece:
        cfg.tuner_search_space = self._build_tuner_space_from_ui()
        
    # ------------------------------------------------------------
    #     configure dialog boxes 
    # ------------------------------------------------------------

    # Physics scalar config bridge  (GeoPriorConfig ↔ PhysicsConfigDialog)

    def _physics_cfg_from_geo_cfg(self) -> dict:
        """
        Build a NAT-style physics dict from self.geo_cfg.

        Keys match PHYSICS_DEFAULTS / NAT config:
        MV_LR_MULT, KAPPA_LR_MULT, GEOPRIOR_INIT_MV, ...
        """
        cfg = self.geo_cfg

        def get(name: str, attr: str):
            # Fall back to PHYSICS_DEFAULTS if the dataclass
            # does not carry the attribute for some reason.
            return getattr(cfg, attr, PHYSICS_DEFAULTS[name])

        return {
            "MV_LR_MULT":          float(
                get("MV_LR_MULT", "mv_lr_mult")),
            "KAPPA_LR_MULT":       float(
                get("KAPPA_LR_MULT", "kappa_lr_mult")),
            "GEOPRIOR_INIT_MV":    float(
                get("GEOPRIOR_INIT_MV", "geoprior_init_mv")),
            "GEOPRIOR_INIT_KAPPA": float(
                get("GEOPRIOR_INIT_KAPPA", "geoprior_init_kappa")),
            "GEOPRIOR_GAMMA_W":    float(
                get("GEOPRIOR_GAMMA_W", "geoprior_gamma_w")),
            "GEOPRIOR_H_REF":      float(
                get("GEOPRIOR_H_REF", "geoprior_h_ref")),
            "GEOPRIOR_KAPPA_MODE": str(
                get("GEOPRIOR_KAPPA_MODE", "geoprior_kappa_mode")).lower(),
            "GEOPRIOR_HD_FACTOR":  float(
                get("GEOPRIOR_HD_FACTOR", "geoprior_hd_factor")),
        }

    def _apply_physics_cfg_to_geo_cfg(self, phys_cfg: dict) -> None:
        """
        Take a NAT-style physics dict (as returned by the dialog)
        and write it back into self.geo_cfg.
        """
        cfg = self.geo_cfg

        if "MV_LR_MULT" in phys_cfg:
            cfg.mv_lr_mult = float(phys_cfg["MV_LR_MULT"])

        if "KAPPA_LR_MULT" in phys_cfg:
            cfg.kappa_lr_mult = float(phys_cfg["KAPPA_LR_MULT"])

        if "GEOPRIOR_INIT_MV" in phys_cfg:
            cfg.geoprior_init_mv = float(phys_cfg["GEOPRIOR_INIT_MV"])

        if "GEOPRIOR_INIT_KAPPA" in phys_cfg:
            cfg.geoprior_init_kappa = float(phys_cfg["GEOPRIOR_INIT_KAPPA"])

        if "GEOPRIOR_GAMMA_W" in phys_cfg:
            cfg.geoprior_gamma_w = float(phys_cfg["GEOPRIOR_GAMMA_W"])

        if "GEOPRIOR_H_REF" in phys_cfg:
            cfg.geoprior_h_ref = float(phys_cfg["GEOPRIOR_H_REF"])

        if "GEOPRIOR_KAPPA_MODE" in phys_cfg:
            cfg.geoprior_kappa_mode = str(
                phys_cfg["GEOPRIOR_KAPPA_MODE"]
            ).lower()

        if "GEOPRIOR_HD_FACTOR" in phys_cfg:
            cfg.geoprior_hd_factor = float(
                phys_cfg["GEOPRIOR_HD_FACTOR"])

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

    @pyqtSlot()
    def _on_feature_config(self) -> None:
        if self.csv_path is None and self._edited_df is None:
            QMessageBox.information(
                self,
                "Dataset required",
                "Please open a dataset first so that columns "
                "can be listed.",
            )
            return
    
        base_cfg = self.geo_cfg._base_cfg or {}
    
        dlg = FeatureConfigDialog(
            csv_path=self.csv_path,
            base_cfg=base_cfg,
            current_overrides=self.geo_cfg.feature_overrides,
            parent=self,
            df=self._edited_df,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
    
        # Store overrides back on the config
        self.geo_cfg.feature_overrides = dlg.get_overrides()
        overs = self.geo_cfg.feature_overrides or {}
    
        # ------------------------------------------------------------------
        # Core driver groups
        # ------------------------------------------------------------------
        dyn_list = overs.get("DYNAMIC_DRIVER_FEATURES", []) or []
        stat_list = overs.get("STATIC_DRIVER_FEATURES", []) or []
        fut_list = overs.get("FUTURE_DRIVER_FEATURES", []) or []
    
        def _list_or(text_list, empty: str) -> str:
            return ", ".join(text_list) if text_list else empty
    
        dyn = _list_or(dyn_list, "auto-detect from dataset")
        stat = _list_or(stat_list, "none")
        fut = _list_or(fut_list, "none")
    
        hcol = overs.get("H_FIELD_COL_NAME") or "none"
    
        # Core spatio-temporal / targets (fall back to base_cfg defaults)
        time_col = overs.get("TIME_COL") or base_cfg.get("TIME_COL", "year")
        lon_col = overs.get("LON_COL") or base_cfg.get("LON_COL", "longitude")
        lat_col = overs.get("LAT_COL") or base_cfg.get("LAT_COL", "latitude")
        subs_col = overs.get("SUBSIDENCE_COL") or base_cfg.get(
            "SUBSIDENCE_COL", "subsidence"
        )
        gwl_col = overs.get("GWL_COL") or base_cfg.get(
            "GWL_COL", "GWL_depth_bgs_z"
        )
    
        use_eff = overs.get(
            "USE_EFFECTIVE_H_FIELD",
            base_cfg.get("USE_EFFECTIVE_H_FIELD", True),
        )
        flags_dyn = overs.get(
            "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC",
            base_cfg.get("INCLUDE_CENSOR_FLAGS_AS_DYNAMIC", True),
        )
    
        msg_lines = [
            "Feature configuration updated:",
            (
                "  • Time / Lon / Lat / Subs / GWL: "
                f"{time_col} / {lon_col} / {lat_col} / {subs_col} / {gwl_col}"
            ),
            f"  • Dynamic driver features: {dyn}",
            f"  • Static features: {stat}",
            f"  • Future drivers: {fut}",
            f"  • H-field column: {hcol}",
            f"  • Use effective H-field: {use_eff}",
            f"  • Flags as dynamic drivers: {flags_dyn}",
        ]
    
        # ------------------------------------------------------------------
        # Registries & already-normalised
        # ------------------------------------------------------------------
        num_reg = overs.get("OPTIONAL_NUMERIC_FEATURES_REGISTRY", []) or []
        cat_reg = overs.get("OPTIONAL_CATEGORICAL_FEATURES_REGISTRY", []) or []
        norm_feats = overs.get("ALREADY_NORMALIZED_FEATURES", []) or []
    
        if num_reg:
            msg_lines.append(
                f"  • Numeric registry groups: {len(num_reg)}"
            )
        if cat_reg:
            msg_lines.append(
                f"  • Categorical registry groups: {len(cat_reg)}"
            )
        if norm_feats:
            msg_lines.append(
                "  • Already-normalized features: "
                + ", ".join(norm_feats)
            )
    
        self.log_updated.emit("\n".join(msg_lines))

        
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


    @pyqtSlot()
    def _on_physics_config_clicked(self) -> None:
        """
        Open the PhysicsConfigDialog.

        Values are initialised from self.geo_cfg and written
        back to self.geo_cfg if the user clicks OK.
        """
        # Make sure temporal + lambda widgets are synced into geo_cfg
        self._sync_config_from_ui()

        # Current scalar physics values as NAT-style dict
        initial = self._physics_cfg_from_geo_cfg()

        updated = PhysicsConfigDialog.edit_physics(
            parent=self,
            cfg=initial,
        )
        if updated is None:
            # User cancelled
            return

        # Push the new values into GeoPriorConfig
        self._apply_physics_cfg_to_geo_cfg(updated)

        # Optional: sanity check + log
        try:
            self.geo_cfg.ensure_valid()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Physics configuration",
                f"Updated physics values may be inconsistent:\n\n{exc}",
            )
        else:
            self.log_updated.emit(
                "[Physics] Scalar physics parameters updated."
            )
            

    # ------------------------------------------------------------------
    # Preferred Stage-1 manifest API (used by Stage1ManagerTool)
    # ------------------------------------------------------------------
    def set_preferred_stage1_manifest(
        self,
        city: str,
        manifest_path: str,
    ) -> None:
        """
        Remember a preferred Stage-1 manifest for a given city.

        Called by the Tools → Stage-1 manager when the user clicks
        \"Use for this city in GUI\".
        """
        city_slug = (city or "").strip().lower()
        if not city_slug:
            return

        p = Path(manifest_path).expanduser()
        if not p.is_file():
            # Do not crash if file disappeared; just log a short warning.
            try:
                self.log_updated.emit(
                    "[Stage-1 manager] Cannot set preferred manifest for "
                    f"'{city_slug}': file does not exist:\n  {p}"
                )
            except Exception:
                pass
            return

        self._preferred_stage1_by_city[city_slug] = str(p)

        # Optional but convenient: keep City field in sync with the choice
        try:
            if hasattr(self, "city_edit"):
                self.city_edit.setText(city)
        except Exception:
            pass

        try:
            self.log_updated.emit(
                "[Stage-1 manager] Preferred Stage-1 manifest for "
                f"city '{city_slug}' set to:\n  {p}"
            )
        except Exception:
            pass

    def _get_preferred_stage1_manifest_for_city(
        self,
        city: str,
    ) -> str | None:
        """
        Return a valid preferred manifest for ``city``, or ``None``.

        Also auto-cleans the registry if the file no longer exists.
        """
        if not city:
            return None

        key = city.strip().lower()
        path = self._preferred_stage1_by_city.get(key)
        if not path:
            return None

        p = Path(path)
        if not p.is_file():
            # Manifest disappeared → drop mapping and warn once.
            self._preferred_stage1_by_city.pop(key, None)
            try:
                self.log_updated.emit(
                    "[Stage-1 manager] Preferred manifest for "
                    f"'{key}' no longer exists → entry cleared."
                )
            except Exception:
                pass
            return None

        return str(p)

    def _resolve_stage1_manifest_for_current_city(self) -> str | None:
        """
        Convenience wrapper: infer the active city from GeoPriorConfig
        and return a preferred Stage-1 manifest if one was selected.

        Fallback: also try the city line-edit if present.
        """
        city = None

        # 1) GeoPriorConfig (canonical)
        if getattr(self, "geo_cfg", None) is not None:
            city = getattr(self.geo_cfg, "city", None)

        # 2) City line-edit on the toolbar, if any
        if not city and hasattr(self, "city_lineedit"):
            try:
                txt = self.city_lineedit.text().strip()
                if txt:
                    city = txt
            except Exception:
                pass

        return self._get_preferred_stage1_manifest_for_city(city)

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
        # --- Model-level params (dialog only) ---
        self.model_params_dialog.load_from_space(space, defaults)
        
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

        # --- Model-level params (from dialog) ---
        space.update(self.model_params_dialog.to_search_space_fragment())
        # --- Scalars & loss weights (from dialog) ---
        space.update(self.scalars_dialog.to_search_space_fragment())

        return space

    @pyqtSlot()
    def _on_model_params_config(self) -> None:
        """
        Open the Model params dialog.
    
        The dialog is pre-populated from the current tuner search space
        (or defaults). When the user clicks OK, the values are written
        back into GeoPriorConfig.tuner_search_space so they persist
        across dialog openings.
        """
        space = self.geo_cfg.tuner_search_space or default_tuner_search_space()
        defaults = default_tuner_search_space()
        self.model_params_dialog.load_from_space(space, defaults)
    
        if self.model_params_dialog.exec_() == QDialog.Accepted:
            # 1) get fragment from dialog
            frag = self.model_params_dialog.to_search_space_fragment()
    
            # 2) merge into current tuner space (copy to avoid in-place surprises)
            current = dict(self.geo_cfg.tuner_search_space
                           or default_tuner_search_space())
            current.update(frag)
            self.geo_cfg.tuner_search_space = current
    
            # 3) (optional) if you want Tune tab widgets to reflect any
            #    model-level changes immediately, call:
            # self._load_tuner_space_into_ui()
    
            self.log_updated.emit("Model-level tuning parameters updated.")
    

    @pyqtSlot()
    def _on_scalars_config(self) -> None:
        """
        Open the Scalars & loss weights dialog.
        
        The dialog is pre-populated from the current tuner search space
        (or defaults). When the user clicks OK, the values are written
        back into GeoPriorConfig.tuner_search_space so they persist
        across dialog openings and sessions.
        """
        space = self.geo_cfg.tuner_search_space or default_tuner_search_space()
        defaults = default_tuner_search_space()
        self.scalars_dialog.load_from_space(space, defaults)
    
        if self.scalars_dialog.exec_() == QDialog.Accepted:
            frag = self.scalars_dialog.to_search_space_fragment()
            current = dict(self.geo_cfg.tuner_search_space
                           or default_tuner_search_space())
            current.update(frag)
            self.geo_cfg.tuner_search_space = current
    
            # Optionally refresh Tune tab UI:
            self._load_tuner_space_into_ui()
    
            self.log_updated.emit("Scalars & loss weights updated for tuning.")
    
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
        
    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_browse_model(self) -> None:
        self.file_browse.browse_model(self.inf_model_edit)
    
    @pyqtSlot()
    def _on_browse_manifest(self) -> None:
        self.file_browse.browse_manifest(self.inf_manifest_edit)
    
    @pyqtSlot()
    def _on_browse_inputs_npz(self) -> None:
        self.file_browse.browse_inputs_npz(self.inf_inputs_edit)
    
    @pyqtSlot()
    def _on_browse_targets_npz(self) -> None:
        self.file_browse.browse_targets_npz(self.inf_targets_edit)
    
    @pyqtSlot()
    def _on_browse_calibrator(self) -> None:
        self.file_browse.browse_calibrator(self.inf_calib_edit)
        
    @pyqtSlot()
    def _on_open_dataset(self) -> None:
        """
        Open a dataset (CSV/parquet/xlsx/…), let the user edit it,
        then save as a canonical CSV and update City/Dataset.
        """
        csv_path, self._edited_df, city_name = open_dataset_with_editor(
            self,
            gui_runs_root=self.gui_runs_root,
            initial_dir="",
        )
        if csv_path is None:
            return  # user cancelled or error

        self.csv_path = csv_path

        # Auto-fill City/Dataset if empty
        if not self.city_edit.text().strip() and city_name:
            self.city_edit.setText(city_name)

        self.log_updated.emit(f"Dataset selected: {self.csv_path}")

        # Friendly hint: check the Feature config dialog at least once
        if not getattr(self, "_feature_tip_shown", False):
            msg = QMessageBox(self)
            msg.setWindowTitle("Check feature configuration")
            msg.setIcon(QMessageBox.Information)
            msg.setTextFormat(Qt.RichText)
            msg.setText(
                "Dataset selected.<br><br>"
                "Please open the <b>“Feature config…”</b> box to "
                "verify the dynamic features and H-field column before "
                "running the PINN."
            )
            msg.exec_()
            self._feature_tip_shown = True

    @pyqtSlot()
    def _on_train_clicked(self) -> None:
        """
        Main entry point when the user clicks "Run training".
    
        Behaviour:
    
        1. If a TrainJobSpec was queued from the Options dialog, run that job.
        2. Otherwise, if existing Stage-1 runs are found, offer QuickTrain.
        3. If no jobs exist, delegate planning to TrainController, which
           will decide Stage-1 reuse vs. rebuild.
        """
        # --------------------------------------------------------------
        # Concurrency guard
        # --------------------------------------------------------------
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
    
        # Push current GUI values into self.geo_cfg
        self._sync_config_from_ui()
    
        # Keep shared RunEnv in sync
        self._run_env.geo_cfg = self.geo_cfg
        self._run_env.gui_runs_root = Path(self.gui_runs_root)
        self._run_env.device_overrides = (
            getattr(self, "_device_cfg_overrides", {}) or {}
        )
        self._run_env.dry_mode = self._is_dry_mode()
    
        # --------------------------------------------------------------
        # Dry-run short-circuit: use TrainController.dry_preview
        # --------------------------------------------------------------
        if self._is_dry_mode():
            csv_path = getattr(self, "csv_path", None)
            city_text = self.city_edit.text().strip()
    
            exp_name = None
            if hasattr(self, "exp_name_edit"):
                exp_name = self.exp_name_edit.text().strip() or None
    
            gui_state = TrainGuiState(
                city_text=city_text,
                csv_path=csv_path,
                experiment_name=exp_name,
            )
            self.train_controller.dry_preview(gui_state)
            return
        # --------------------------------------------------------------
        # Preferred Stage-1 manifest (from Tools → Stage-1 manager)
        # --------------------------------------------------------------
        preferred_manifest = None
        try:
            city_text = self.city_edit.text().strip()
        except Exception:
            city_text = ""
    
        if city_text:
            preferred_manifest = self._get_preferred_stage1_manifest_for_city(
                city_text
            )

        # --------------------------------------------------------------
        # 0. If options dialog queued a specific job, honour it
        # --------------------------------------------------------------
        if self._queued_train_job is not None:
            job = self._queued_train_job
            self._queued_train_job = None
            self._run_job(job)
            return

        # --------------------------------------------------------------
        # 1. Try QuickTrain shortcut: reuse an existing Stage-1 run
        #    (respect preferred manifest if one was set).
        # --------------------------------------------------------------
        try:
            stage1_cfg = self.geo_cfg.to_stage1_config()
            jobs = latest_jobs_for_root(
                results_root=self.gui_runs_root,
                current_cfg=stage1_cfg,
            )

            quick_job = None

            # 1.a If a preferred manifest exists for this city, try to
            #     find a matching job and reuse it directly.
            if preferred_manifest and jobs:
                pref_path = Path(preferred_manifest).resolve()

                for job in jobs:
                    summary = getattr(job, "stage1_summary", None)
                    manifest_attr = getattr(
                        summary, "manifest_path", None
                    ) if summary is not None else None

                    if manifest_attr is None:
                        continue

                    try:
                        job_path = Path(str(manifest_attr)).resolve()
                    except Exception:
                        continue

                    if job_path == pref_path:
                        quick_job = job
                        self._append_status(
                            "[QuickTrain] Using preferred Stage-1 run "
                            "from Stage-1 manager:\n"
                            f"  city   = {getattr(summary, 'city', '?')}\n"
                            f"  manifest = {pref_path}"
                        )
                        break

            # 1.b If no preferred job matched, fall back to the dialog
            if quick_job is None:
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

        # --------------------------------------------------------------
        # 2. No existing jobs: use TrainController + Stage1Service
        # --------------------------------------------------------------
        csv_path = getattr(self, "csv_path", None)
        if csv_path is None:
            QMessageBox.warning(
                self,
                "No training dataset",
                "Please choose a training data file first.",
            )
            return
    
        city_text = self.city_edit.text().strip()
    
        exp_name = None
        if hasattr(self, "exp_name_edit"):
            exp_name = self.exp_name_edit.text().strip() or None
    
        gui_state = TrainGuiState(
            city_text=city_text,
            csv_path=csv_path,
            experiment_name=exp_name,
        )
    
        # UI lock + run timer for training flow
        self.train_btn.setEnabled(False)
        if hasattr(self, "btn_train_options"):
            self.btn_train_options.setEnabled(False)
        self._update_progress(0.0)
        self._active_job_kind = "train"
        self._update_global_running_state()
    
        # Callbacks back into the GUI so the controller can start threads
        def _start_stage1_cb(city: str, cfg_overrides: Dict[str, Any]) -> None:
            # Save overrides for Stage-1 + Training
            self._cfg_overrides = dict(cfg_overrides)
            self.status_updated.emit("Stage-1: preparing sequences…")
            self._update_progress(0.0)
            self._start_stage1(city=city)
    
        def _start_training_cb(
            manifest_path: str,
            cfg_overrides: Dict[str, Any],
        ) -> None:
            self._cfg_overrides = dict(cfg_overrides)
            self.status_updated.emit(
                "Stage-2: training GeoPrior model (reusing Stage-1)…"
            )
            self._update_progress(0.0)
            self._start_training(manifest_path)
    
        # Delegate the planning / Stage-1 handshake to the controller
        
        self._start_run_timer()
        self.train_controller.start_real_run(
            gui_state=gui_state,
            start_stage1_cb=_start_stage1_cb,
            start_training_cb=_start_training_cb,
        )
        
        # If planning failed or user cancelled at the Stage-1 handshake,
        # no worker thread will have been started. In that case we must
        # stop the run timer and unlock the UI again.
        if not self._any_job_running():
            self._stop_run_timer()
            self._active_job_kind = None
            self._update_global_running_state()
            self.train_btn.setEnabled(True)
            if hasattr(self, "btn_train_options"):
                self.btn_train_options.setEnabled(True)

    # --------------------------------------------------------------
    # Tuning 
    # --------------------------------------------------------------

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
    
        # Keep legacy attribute + workflow env in sync
        self._device_cfg_overrides = dev_overrides or {}
        if hasattr(self, "_wf_env"):
            self._wf_env.device_overrides = self._device_cfg_overrides
    
        if not ok:
            return
    
        # Update GUI-level runs root used by all GUI runs
        self.gui_runs_root = new_root
        self.results_root = new_root  # keep in sync
        if hasattr(self, "_wf_env"):
            self._wf_env.gui_runs_root = Path(self.gui_runs_root)
    
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
    
        if hasattr(self, "log_mgr") and self.log_mgr:
            self.log_mgr.clear()
    
        # Keep RunEnv.dry_mode in sync with the GUI
        if hasattr(self, "_wf_env"):
            self._wf_env.dry_mode = bool(self._is_dry_mode())
    
        job: TuneJobSpec | None = None
        manifest_path: str | None = None
        stage1_root: str | None = None
    
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
        #    derive city, manifest and stage1_root from it.
        if job is not None:
            city = job.stage1.city
            manifest_path = str(job.stage1.manifest_path)
            stage1_root = getattr(job.stage1, "run_dir", manifest_path)
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
            self._stop_run_timer()
            return
    
        # 3) Sync base training config (PDE weights, LR, etc.) from Train tab
        #    This ensures env.geo_cfg is up-to-date before the controller
        #    calls geo_cfg.to_cfg_overrides().
        self._sync_config_from_ui()
    
        # 4) Build search space + max_trials from Tune tab widgets
        tuner_space = self._build_tuner_space_from_ui()
        max_trials: int | None = None
        if hasattr(self, "spin_max_trials"):
            try:
                max_trials = int(self.spin_max_trials.value())
            except Exception:
                max_trials = None
    
        eval_tuned = self.chk_eval_tuned.isChecked()
    
        # 5) Assemble TuneGuiState for the workflow controller
        state = TuneGuiState(
            city_text=city,
            manifest_path=manifest_path,
            stage1_root=stage1_root,
            tuner_search_space=tuner_space,
            max_trials=max_trials,
            eval_tuned=eval_tuned,
        )
    
        # --- dry-run mode --------------------------------------------------
        if self._is_dry_mode():
            # Controller will log + update main progress bar via GUIHooks
            self.tune_controller.dry_preview(state)
            return
    
        # --- real tuning run -----------------------------------------------
        def _start_tuning_from_plan(plan, eval_tuned_flag: bool) -> None:
            """
            Callback passed to TuneController.start_real_run.
    
            It receives a validated TunePlan and is responsible for
            creating and starting the TuningThread, wiring signals,
            and managing GUI buttons / timers.
            """
            # Start / restart timer for tuning flow
            self._start_run_timer()
    
            # Cache overrides for potential reuse / inspection
            self._cfg_overrides = plan.cfg_overrides
    
            # Create and start TuningThread
            th = TuningThread(
                manifest_path=plan.manifest_path,
                cfg_overrides=self._cfg_overrides,
                evaluate_tuned=eval_tuned_flag,
                base_cfg = self.geo_cfg._base_cfg, 
                results_root = self.results_root, 
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
    
        # Delegate planning + logging to TuneController
        self.tune_controller.start_real_run(
            state,
            start_tuning_cb=_start_tuning_from_plan,
        )

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
        self._stop_run_timer()
        
    # --------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------
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
        
        if hasattr(self.geo_cfg, "results_root"):
            self.geo_cfg.results_root = new_root
        
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
            self._active_job_kind = None
            self._update_global_running_state()
            self._stop_run_timer()
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
        self._stop_run_timer()

    @pyqtSlot()
    def _on_infer_clicked(self) -> None:
        """
        Run Stage-3 inference using InferenceThread (non-blocking).
    
        All planning and validation is delegated to InferenceController.
        """
        if self.inference_thread is not None and self.inference_thread.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "Inference is already running.",
            )
            return
    
        # Clear log manager if present (same behaviour as before)
        if hasattr(self, "log_mgr") and self.log_mgr:
            self.log_mgr.clear()
    
        # ------------------------------------------------------------------
        # 1) Build InferenceGuiState from the current widgets
        # ------------------------------------------------------------------
        model_path = self.inf_model_edit.text().strip()
        dataset_key = self.cmb_inf_dataset.currentData() or "test"
        use_future = self.chk_inf_use_future.isChecked()
        manifest_path = self.inf_manifest_edit.text().strip() or None
    
        inputs_npz: str | None = None
        targets_npz: str | None = None
        if dataset_key == "custom" and not use_future:
            inputs_npz = self.inf_inputs_edit.text().strip() or None
            targets_npz = self.inf_targets_edit.text().strip() or None
    
        state = InferenceGuiState(
            model_path=model_path,
            dataset_key=dataset_key,
            use_future_npz=use_future,
            manifest_path=manifest_path,
            inputs_npz=inputs_npz,
            targets_npz=targets_npz,
            cov_target=float(self.sp_inf_cov.value()),
            include_gwl=self.chk_inf_include_gwl.isChecked(),
            batch_size=int(self.sp_inf_batch.value()),
            make_plots=self.chk_inf_plots.isChecked(),
            use_source_calibrator=self.chk_inf_use_source_calib.isChecked(),
            fit_calibrator=self.chk_inf_fit_calib.isChecked(),
            calibrator_path=self.inf_calib_edit.text().strip() or None,
        )
    
        # ------------------------------------------------------------------
        # 2) Dry-run short-circuit: delegate to controller, no threads
        # ------------------------------------------------------------------
        if self._is_dry_mode():
            self.infer_controller.dry_preview(state)
            return
    
        # ------------------------------------------------------------------
        # 3) Real run – delegate planning to controller, we only define
        #    how to actually start InferenceThread.
        # ------------------------------------------------------------------
        
        def _start_infer(plan: InferencePlan) -> None:
            """
            Callback used by InferenceController.start_real_run.
    
            Creates and wires InferenceThread from a validated plan.
            """
            # start / restart timer for inference flow
            self._start_run_timer()
    
            th = InferenceThread(
                model_path=plan.model_path,
                dataset=plan.dataset_key,
                use_stage1_future_npz=plan.use_stage1_future_npz,
                manifest_path=plan.manifest_path,
                stage1_dir=plan.stage1_dir,
                inputs_npz=plan.inputs_npz,
                targets_npz=plan.targets_npz,
                use_source_calibrator=plan.use_source_calibrator,
                calibrator_path=plan.calibrator_path,
                fit_calibrator=plan.fit_calibrator,
                cov_target=plan.cov_target,
                include_gwl=plan.include_gwl,
                batch_size=plan.batch_size,
                make_plots=plan.make_plots,
                cfg_overrides=None,
                parent=self,
            )
            self.inference_thread = th
    
            # Wire signals
            th.log_updated.connect(self.log_updated.emit)
            th.status_updated.connect(self.status_updated.emit)
            th.progress_changed.connect(self._on_thread_progress)
            th.error_occurred.connect(self._on_worker_error)
            th.inference_finished.connect(self._on_inference_finished)
    
            # Disable button while job is active
            self.btn_run_infer.setEnabled(False)
    
            self._active_job_kind = "infer"
            self._update_global_running_state()
    
            th.start()
            # After thread is running, ensure Stop button / labels updated
            self._update_global_running_state()
    
        # Let the controller validate + log + call _start_infer
        self.infer_controller.start_real_run(
            state=state,
            start_infer_cb=_start_infer,
        )

    # --------------------------------------------------------------
    # Transferability
    # --------------------------------------------------------------
    
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
    
        The actual planning / validation is delegated to
        `TransferController`. This slot only:
    
        - protects against concurrent runs;
        - builds `TransferGuiState` from widgets;
        - chooses dry-run vs real-run;
        - provides a callback that knows how to start XferMatrixThread.
        """
        # Prevent concurrent transfer runs
        if self.xfer_thread is not None:
            QMessageBox.information(
                self,
                "Busy",
                "Transferability is already running.",
            )
            return
    
        if hasattr(self, "log_mgr"):
            self.log_mgr.clear()
    
        # ------------------------------------------------------------------
        # 1) Build GUI state snapshot for the controller
        # ------------------------------------------------------------------
        city_a = self.xfer_city_a.text().strip()
        city_b = self.xfer_city_b.text().strip()
    
        # For results_root, we keep previous behaviour:
        #   - prefer text from xfer_results_root if present;
        #   - otherwise fall back to gui_runs_root inside the controller.
        results_root_txt = self.xfer_results_root.text().strip() or None
    
        splits: list[str] = []
        if self.chk_xfer_split_train.isChecked():
            splits.append("train")
        if self.chk_xfer_split_val.isChecked():
            splits.append("val")
        if self.chk_xfer_split_test.isChecked():
            splits.append("test")
    
        calib_modes: list[str] = []
        if self.chk_xfer_cal_none.isChecked():
            calib_modes.append("none")
        if self.chk_xfer_cal_source.isChecked():
            calib_modes.append("source")
        if self.chk_xfer_cal_target.isChecked():
            calib_modes.append("target")
    
        state = TransferGuiState(
            city_a=city_a,
            city_b=city_b,
            results_root=results_root_txt,
            splits=splits,
            calib_modes=calib_modes,
            rescale_to_source=self.chk_xfer_rescale.isChecked(),
            batch_size=int(self.sp_xfer_batch.value()),
            quantiles_override=self._xfer_quantiles_override,
            write_json=self._xfer_write_json,
            write_csv=self._xfer_write_csv,
        )
    
        # ------------------------------------------------------------------
        # 2) Dry-run short-circuit
        # ------------------------------------------------------------------
        if self._is_dry_mode():
            self.transfer_controller.dry_preview(state)
            return
    
        # ------------------------------------------------------------------
        # 3) Real run: let the controller validate and then start the thread
        # ------------------------------------------------------------------
        def _start_xfer_from_plan(plan: TransferPlan) -> None:
            """
            Callback invoked by TransferController once validation succeeds.
    
            Responsible for:
            - starting the run timer;
            - creating XferMatrixThread;
            - wiring Qt signals / buttons;
            - starting the thread and updating global state.
            """
            self._start_run_timer()
    
            th = XferMatrixThread(
                city_a=plan.city_a,
                city_b=plan.city_b,
                results_dir=str(plan.results_root),
                splits=plan.splits,
                calib_modes=plan.calib_modes,
                rescale_to_source=plan.rescale_to_source,
                batch_size=plan.batch_size,
                quantiles_override=plan.quantiles_override,
                out_dir=None,
                write_json=plan.write_json,
                write_csv=plan.write_csv,
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
    
        # Delegate real planning to the controller
        self.transfer_controller.start_real_run(state, _start_xfer_from_plan)

        
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
            self._stop_run_timer()
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
        self._stop_run_timer()


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

    # ------------------------------------------------------------------
    # Thread orchestration
    # ------------------------------------------------------------------
    def _start_stage1(self, city: str) -> None:
        results_root = getattr(self, "results_root", self.gui_runs_root)

        # Make sure we have cfg_overrides dict to work with
        overrides = getattr(self, "_cfg_overrides", {}) or {}
        overrides = dict(overrides)  # shallow copy

        edited_df = self._edited_df

        # If no in-memory dataset is provided, resolve it from _datasets/
        if edited_df is None:
            csv_path_str = choose_dataset_for_city(
                parent=self,
                city=city,
                results_root=Path(results_root),
            )
            if not csv_path_str:
                # User cancelled or no dataset found; dialog already
                # informed them, so just abort Stage-1 quietly.
                
                return

            csv_path = Path(csv_path_str)
            overrides["DATA_DIR"] = str(csv_path.parent)
            overrides["BIG_FN"] = csv_path.name

            # Keep a stable copy for Stage1Thread
            self._cfg_overrides = overrides

            # Optional: small log line in GUI
            self.log_updated.emit(
                f"[Stage-1] Using dataset: {csv_path.name} "
                f"({csv_path.parent})"
            )

        th = Stage1Thread(
            city=city,
            cfg_overrides=self._cfg_overrides,
            clean_run_dir=self.geo_cfg.clean_stage1_dir,
            base_cfg=self.geo_cfg._base_cfg,
            edited_df=edited_df,        # may still be None
            results_root=str(results_root),
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
        #  sync GUI → config for this job as well
        self._sync_config_from_ui()

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
        self.train_btn.setEnabled(False)
        self.btn_train_options.setEnabled(False)
    
        self._append_status(
            "[Job] Starting training "
            f"(city={job.stage1_summary.city}, mode={job.mode}, "
            f"root={job.results_root})"
        )

        # --- Case 1: reuse existing Stage-1, training only -------------------
        if job.mode == "reuse":
            self.status_updated.emit(
                "Stage-2: training GeoPrior model (reusing Stage-1)…"
            )

            if job.stage1_summary is None:
                # Very defensive; normally this should never happen
                self._append_status(
                    "[Job] Cannot reuse Stage-1 – no summary attached "
                    "to TrainJobSpec."
                )
                self.train_btn.setEnabled(True)
                self.btn_train_options.setEnabled(True)
                return

            # Directly reuse the Stage-1 summary; handler will pick
            # up manifest_path and start TrainingThread.
            self._on_stage1_finished(job.stage1_summary)
            return

        # --- Case 2: rebuild/scratch: run Stage-1, then training -------------
        stage1_cfg = cfg.to_stage1_config()
        stage1_cfg["BASE_OUTPUT_DIR"] = str(self.gui_runs_root)

        # --- Case 2: rebuild/scratch: run Stage-1, then training -------------

        # Build cfg_overrides just like in the normal Train path
        self._cfg_overrides = cfg.to_cfg_overrides()
        if getattr(self, "_device_cfg_overrides", None):
            self._cfg_overrides.update(self._device_cfg_overrides)

        # Ensure TRAIN_CSV_PATH is set (we already validated above)
        self._cfg_overrides["TRAIN_CSV_PATH"] = csv_path_str

        # Force rebuild flag into overrides if your backend uses it
        self._cfg_overrides["FORCE_REBUILD"] = True

        # Lock the Stage-1 city to that of the job
        city = job.stage1_summary.city

        self.status_updated.emit("Stage-1: preparing sequences…")
        self._start_stage1(city=city)
  

    @pyqtSlot(object)
    def _on_stage1_finished(self, result: Any) -> None:
        """
        Handle completion of Stage-1 (either reused or freshly built).

        Parameters
        ----------
        result :
            Either

            * a Stage1Summary instance (preferred new path), or
            * a dict with a "manifest_path" entry (legacy path).
        """
        # Stage-1 thread (if any) is no longer running
        self.stage1_thread = None

        if result is None:
            self.log_updated.emit(
                "Stage-1 finished with an empty result object."
            )
            self.status_updated.emit("Stage-1 failed. See log.")
            self.train_btn.setEnabled(True)
            # Optional but nice: re-enable options too
            if hasattr(self, "btn_train_options"):
                self.btn_train_options.setEnabled(True)
            self._stop_run_timer()
            
            return

        manifest_path: str | None = None

        # --- New-style: Stage1Summary-like object -------------------------
        # Any object with a .manifest_path attribute (Stage1Summary)
        attr = getattr(result, "manifest_path", None)
        if attr is not None:
            manifest_path = str(attr)

        # --- Legacy: dict coming from old run_stage1 helper --------------
        elif isinstance(result, dict):
            manifest_path = result.get("manifest_path")

        if not manifest_path:
            self.log_updated.emit(
                "Stage-1 result did not provide a manifest_path."
            )
            self.status_updated.emit(
                "Cannot start training – no Stage-1 manifest."
            )
            self.train_btn.setEnabled(True)
            if hasattr(self, "btn_train_options"):
                self.btn_train_options.setEnabled(True)
            
            self._stop_run_timer()
            self._active_job_kind = None
            self._update_global_running_state()
            return

        self.log_updated.emit(
            "Stage-1 done. Manifest:\n"
            f"  {manifest_path}"
        )
        self.status_updated.emit("Stage-2: training GeoPrior model.")
        self._start_training(manifest_path)


    def _start_training(self, manifest_path: str) -> None:
        """
        Launch the TrainingThread for the given Stage-1 manifest.
    
        Uses the last cfg_overrides prepared by TrainController or
        _run_job (TrainOptions / QuickTrain). Falls back to building
        overrides from the current config if none were set.
        """
        cfg_overrides = getattr(self, "_cfg_overrides", None)
    
        if not isinstance(cfg_overrides, dict):
            # Backward-compatible fallback: rebuild from GeoPriorConfig
            # including any device overrides.
            self._sync_config_from_ui()
            cfg_overrides = self.geo_cfg.to_cfg_overrides()
            dev_overrides = getattr(self, "_device_cfg_overrides", {}) or {}
            cfg_overrides.update(dev_overrides)
            self._cfg_overrides = cfg_overrides
    
        th = TrainingThread(
            manifest_path=manifest_path,
            cfg_overrides=cfg_overrides,
            evaluate_training=self.geo_cfg.evaluate_training,
            results_root=self.geo_cfg.results_root,
            base_cfg = self.geo_cfg._base_cfg, 
            parent=self,
        )
        self.train_thread = th
    
        th.log_updated.connect(self.log_updated.emit)
        th.status_updated.connect(self.status_updated.emit)
        th.progress_changed.connect(self._on_thread_progress)
        th.training_finished.connect(self._on_training_finished)
        th.error_occurred.connect(self._on_worker_error)
    
        self._active_job_kind = "train"
        self._update_global_running_state()
    
        th.start()
        self._update_global_running_state()


    @pyqtSlot(dict)
    def _on_training_finished(self, result: Dict[str, Any]) -> None:
        self.train_thread = None
        cfg_overrides = self.geo_cfg.to_cfg_overrides()
        
        CITY_NAME= cfg_overrides.get(
            'city', 
            cfg_overrides.get(
                "CITY_NAME", cfg_overrides.get(
                    "TRAIN_CSV_PATH", 'Geoprior_city')
                )
        )
        
        
        if not result:
            self.log_updated.emit(
                "Training finished with an empty result dict."
            )
            self.status_updated.emit("Training failed. See log.")
            self.train_btn.setEnabled(True)
            self._active_job_kind = None
            self._update_global_running_state()
            self._stop_run_timer()
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
                    city = CITY_NAME
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
        self.btn_train_options.setEnabled(True)
        self._active_job_kind = None
        self._update_global_running_state()
        self._stop_run_timer()

        
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
        self._stop_run_timer()

    # ------------------------------------------------------------------
    # Window close handling
    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Intercept window close (File → Exit, Ctrl+Q, window X).

        If long-running workflows are active, ask for confirmation
        before closing. Otherwise just accept.
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
                return  # keep the window open

        # XXX TODO: could do any final cleanup, save settings, etc.

        # Hand back to the base class for normal closing
        super().closeEvent(event)


# ----------------------------------------------------------------------
# Entry point helper
# ----------------------------------------------------------------------

def launch_geoprior_gui(theme: str = "fusionlab") -> None:
    
    app = QApplication(sys.argv)
    cfg = GeoPriorConfig.from_defaults () 
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
