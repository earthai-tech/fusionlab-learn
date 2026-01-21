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
- A QTabWidget with 9 tabs:
  [Data, Experimental setup, Preprocess, Train, Tune, Inference,
   Transferability, Results, Tools].
"""

from __future__ import annotations

import sys
import time
import re
import shutil
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from PyQt5.QtCore import ( 
    Qt, 
    pyqtSignal, 
    pyqtSlot, 
    QSize, 
    QPoint, 
    QTimer,
    QSettings, 
    QSignalBlocker, 
    QUrl
)
from PyQt5.QtGui import (
    QCloseEvent,
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QIcon, 
    QDesktopServices
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QTabWidget,
    QCheckBox,
    QFileDialog,
    QDialog,
    QToolButton,
    QStyle, 
)

from .config.store import GeoConfigStore, FieldKey
from .workflows.base import RunEnv, GUIHooks
from .workflows.train import TrainController, TrainGuiState

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
from .services.results_service import ResultsService
from .services.city_manager import CityManager
from .services.stage1_service import Stage1Service

from .ui.data_tab import DataTab
from .ui.mode_manager import ModeManager
from .ui.file_browse import FileBrowseHelper
from .ui.menu_manager import MenuManager
from .ui.splash import LoadingSplash
from .ui.tools_tab import ToolsTab
from .ui.train_tab import TrainTab
from .ui.setup_tab import SetupTab
from .ui.preprocess_tab import PreprocessTab
from .ui.tune_tab import TuneTab
from .ui.inference_tab import InferenceTab
from .ui.xfer_tab import XferTab
from .ui.map_tab import MapTab
from .ui.results_tab import ResultsDownloadTab
from .ui.city_field import CityField
from .ui.console import ConsoleDock, ConsoleVisibilityPolicy

from ..ux_utils import (
    set_app_metadata,
    auto_set_ui_fonts,
    auto_resize_window,
    enable_qt_crash_handler,
    save_window_geometry
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

from .config import GeoPriorConfig, find_stage1_for_city
from .dialogs import ( 
    FeatureConfigDialog,
    ArchitectureConfigDialog , 
    ProbConfigDialog ,
    XferResultsDialog, 
    InferenceOptionsDialog, 
    GeoPriorResultsDialog,
    PhysicsConfigDialog,
    QuickTrainDialog, 
    QuickTuneDialog,
    Stage1ChoiceDialog, 
    TrainOptionsDialog, 
    TuneJobSpec,
    open_dataset_with_editor,
    choose_dataset_for_city, 
    CsvEditDialog, 
    save_dataframe_to_csv, 
    _load_dataset_with_progress
)

from .about import show_about_dialog, DOCS_URL
from .utils.view_utils import _notify_gui_xfer_view
from .utils.clock_timer import RunClockTimer
from .utils.components import RangeListEditor
from .styles import (
    TAB_STYLES,
    LOG_STYLES,
    FLAB_STYLE_SHEET,
    PRIMARY,
    RUN_BUTTON_IDLE,
    DARK_THEME_STYLESHEET,
    MAIN_TAB_STYLES_LIGHT,
    MAIN_TAB_STYLES_DARK 
)
from .jobs import TrainJobSpec, latest_jobs_for_root

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
        self._restore_theme_setting()
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
        
        self.config_store = GeoConfigStore(self.geo_cfg)
        self.data_tab = None
        self.setup_tab = None

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
        
        self._main_splitter = None
        self._console_panel = None
        self._console_last_h = 220
                
        self._prep_refresh_key: tuple[str, str, str] | None = None
        self._prep_cache_best = None
        self._prep_cache_manifest = None
        self._prep_cache_audit = None
        
        # XXX TODO: ENABLE once dark is stable. 
        self.enable_dark_mode = False


    def _init_help_texts(self) -> None:
        """Initialise per-tab help texts and one-shot flags."""
        self._data_help_text = ModeManager.DEFAULT_HELP_TEXTS["data"]
        self._data_tip_shown = False
        
        self._setup_help_text = ModeManager.DEFAULT_HELP_TEXTS["setup"]
        self._setup_tip_shown = False
        
        self._preprocess_help_text = ModeManager.DEFAULT_HELP_TEXTS["preprocess"]
        self._preprocess_tip_shown = False
        
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
        
        self._map_help_text = ModeManager.DEFAULT_HELP_TEXTS["map"]
        self._map_tip_shown = False
        
        self._tools_help_text = ModeManager.DEFAULT_HELP_TEXTS["tools"]
        self._tools_tip_shown = False

    def _init_dialogs_and_paths(self) -> None:
        """Initialise dialogs and main results root paths."""
        
        # Dedicated root so GUI runs don't mix with CLI results
        self.gui_runs_root = Path.home() / ".fusionlab_runs"
        self.gui_runs_root.mkdir(parents=True, exist_ok=True)

        # User-overrideable base results root (defaults to gui_runs_root)
        self.results_root = self.gui_runs_root

    def _init_log_manager(self) -> None:
        """Create the central LogManager once the log widget exists."""

        # ------------------------------
        # Dockable Console (single one)
        # ------------------------------
        self.console = ConsoleDock(
            self,
            mirror_to_all=True,
        )
        self.addDockWidget(
            Qt.BottomDockWidgetArea,
            self.console,
        )

        # main_sess = self.console.session(
        #     self.console._main
        # )
        self.console.visibilityChanged.connect(
            self._on_console_visibility_changed
        )
        self.console.topLevelChanged.connect(
            self._on_console_top_level_changed
        )
        self.console.show()

        if not isinstance(
            getattr(self, "_console_vis_by_tab", None),
            dict,
        ):
            self._console_vis_by_tab = {}

        self.console_policy = ConsoleVisibilityPolicy(
            self.console,
            set_visible=self.set_console_visible,
            hidden_tabs=set(),
            vis_by_tab=self._console_vis_by_tab,
            parent=self,
        )

        self._console_vis_by_tab = (
            self.console_policy.get_vis_map()
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
            store = self.config_store, 
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
            preprocess_btn=self.preprocess_tab.btn_run_stage1,
            train_btn=self.train_btn,
            tune_btn=self.btn_run_tune,
            infer_btn=self.btn_run_infer,
            xfer_btn=self.xfer_tab.btn_run_xfer,
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

    def _restore_theme_setting(self) -> None:
        s = QSettings()
        saved = s.value(
            "ui/theme",
            "",
            type=str,
        )
        saved = (saved or "").strip().lower()
        
        if saved and not getattr(self, "enable_dark_mode", False):
            if saved in {"dark", "fusionlab-dark", "night"}:
                saved = "fusionlab"

        if saved:
            self.theme = saved
    
    
    def set_dark_mode(self, enabled: bool) -> None:
        
        if enabled and not getattr(self, "enable_dark_mode", False):
            self.statusBar().showMessage(
                "Dark mode is disabled for now.",
                5000,
            )
            return

        self.theme = "dark" if enabled else "fusionlab"
        
    
        s = QSettings()
        s.setValue("ui/theme", self.theme)
    
        self._apply_theme()


    def _is_dark(self) -> bool:
        t = (self.theme or "fusionlab").strip().lower()
        return t in {"dark", "fusionlab-dark", "night"}
    
    def _apply_theme(self) -> None:
        dark = self._is_dark()
    
        base = (
            DARK_THEME_STYLESHEET
            if dark
            else FLAB_STYLE_SHEET
        )
    
        main_tabs = (
            MAIN_TAB_STYLES_DARK
            if dark
            else MAIN_TAB_STYLES_LIGHT
        )
    
        # If you want dialogs to match too,
        # apply on the QApplication instance.
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(
                base
                + main_tabs
                + TAB_STYLES
                + LOG_STYLES
            )
        else:
            self.setStyleSheet(
                base
                + main_tabs
                + TAB_STYLES
                + LOG_STYLES
            )

    def _set_window_props(self) -> None:
        self.setWindowTitle("GeoPrior-3.0 Forecaster")
    
        cfg = self.geo_cfg
    
        # Bigger defaults (modern feel)
        base_w = int(getattr(cfg, "ui_base_width", 1180))
        base_h = int(getattr(cfg, "ui_base_height", 820))
        min_w = int(getattr(cfg, "ui_min_width", 1060))
        min_h = int(getattr(cfg, "ui_min_height", 740))
        max_ratio = float(getattr(cfg, "ui_max_ratio", 0.92))
    
        auto_resize_window(
            self,
            settings_key="main_window",
            base_size=(base_w, base_h),
            min_size=(min_w, min_h),
            max_ratio=max_ratio,
        )
    
        ico_dir = Path(__file__).parent
        ico = ico_dir / "geoprior_logo.ico"
        if ico.exists():
            self.setWindowIcon(QIcon(str(ico)))
    
        self._apply_theme()


    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
    
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
    
        self.select_csv_btn = QPushButton("Open dataset…")
        top.addWidget(self.select_csv_btn)

        self.city_mgr = CityManager(self.config_store)
        self.city_field = CityField(
            self.config_store,
            manager=self.city_mgr,
            enable_completer=True,
        )
        top.addWidget(self.city_field, 1)
        # Backward-compatible alias for old code
        self.city_edit = self.city_field.line_edit()

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
        
        # ------------------------------
        # Workspace (tabs + status row)
        # ------------------------------
        workspace = QWidget()
        w_layout = QVBoxLayout(workspace)
        w_layout.setContentsMargins(0, 0, 0, 0)
        w_layout.setSpacing(6)

        self._update_splash(45, "Building tabs…")
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setIconSize(QSize(14, 14))

        self._init_tabs()

        w_layout.addWidget(self.tabs, 1)

        status_row = QHBoxLayout()

        self.status_label = QLabel("? Idle")
        self.status_label.setStyleSheet(
            f"color:{PRIMARY};"
        )
        status_row.addWidget(self.status_label, 1)

        status_row.addStretch(1)

        self.run_timer = RunClockTimer(self)
        self.run_timer.reset()
        self.run_timer.stop()
        self.run_timer.setVisible(False)
        status_row.addWidget(self.run_timer, 0)

        w_layout.addLayout(status_row)

        layout.addWidget(workspace, 1)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    def _init_tabs(self) -> None:
        """Create and register all workflow tabs (v3.2 modular UI)."""
    
        # ==============================================================
        # Core workflow tabs (module-based)
        # ==============================================================
    
        # ------------------------------
        # Train tab (new module)
        # ------------------------------
        self.train_tab = TrainTab(
            store=self.config_store,
            make_card=self._make_card,
            make_run_button=self._make_run_button,
            parent=self,
        )
    
        # Backward-compatible attribute aliases (keep old app.py code working)
        tt = self.train_tab
        self.sp_train_end = tt.sp_train_end
        self.sp_forecast_start = tt.sp_forecast_start
        self.sp_forecast_horizon = tt.sp_forecast_horizon
        self.sp_time_steps = tt.sp_time_steps
    
        self.sp_epochs = tt.sp_epochs
        self.sp_batch_size = tt.sp_batch_size
        self.sp_lr = tt.sp_lr
    
        self.cmb_pde_mode = tt.cmb_pde_mode
        self.sb_lcons = tt.sb_lcons
        self.sb_lgw = tt.sb_lgw
        self.sb_lprior = tt.sb_lprior
        self.sb_lsmooth = tt.sb_lsmooth
        self.sb_lmv = tt.sb_lmv
    
        self.sp_phys_warmup = tt.sp_phys_warmup
        self.sp_phys_ramp = tt.sp_phys_ramp
        self.chk_scale_pde = tt.chk_scale_pde
    
        self.btn_features = tt.btn_features
        self.btn_arch = tt.btn_arch
        self.btn_prob = tt.btn_prob
        self.physics_btn = tt.physics_btn
    
        self.btn_train_options = tt.btn_train_options
        self.chk_eval_training = tt.chk_eval_training
        self.chk_build_future = tt.chk_build_future
        self.chk_clean_stage1 = tt.chk_clean_stage1
        self.train_btn = tt.train_btn
    
        # ------------------------------
        # Tune tab (new module)
        # ------------------------------
        self.tune_tab = TuneTab(
            store=self.config_store,
            make_card=self._make_card,
            make_run_button=self._make_run_button,
            range_editor_cls=RangeListEditor,
            parent=self,
        )
    
        # Backward-compatible attribute aliases
        tun = self.tune_tab
        self.hp_embed_dim = tun.hp_embed_dim
        self.hp_hidden_units = tun.hp_hidden_units
        self.hp_lstm_units = tun.hp_lstm_units
        self.hp_attention_units = tun.hp_attention_units
        self.hp_num_heads = tun.hp_num_heads
        self.hp_vsn_units = tun.hp_vsn_units
        self.hp_dropout = tun.hp_dropout
    
        self.hp_pde_mode = tun.hp_pde_mode
        self.hp_kappa_mode = tun.hp_kappa_mode
        self.hp_scale_pde_bool = tun.hp_scale_pde_bool
        self.hp_hd = tun.hp_hd
    
        self.chk_eval_tuned = tun.chk_eval_tuned
        self.spin_max_trials = tun.spin_max_trials
        self.btn_model_params = tun.btn_model_params
        self.btn_scalars = tun.btn_scalars
    
        self.btn_tune_options = tun.btn_tune_options
        self.btn_run_tune = tun.btn_run_tune
    
        # ------------------------------
        # Inference tab (new module)
        # ------------------------------
        self.inference_tab = InferenceTab(
            store=self.config_store,
            make_card=self._make_card,
            make_run_button=self._make_run_button,
            parent=self,
        )
    
        # Backward-compatible attribute aliases
        it = self.inference_tab
        self.inf_model_edit = it.inf_model_edit
        self.inf_model_btn = it.inf_model_btn
        self.inf_manifest_edit = it.inf_manifest_edit
        self.inf_manifest_btn = it.inf_manifest_btn
        self.cmb_inf_dataset = it.cmb_inf_dataset
        self.chk_inf_use_future = it.chk_inf_use_future
        self.inf_inputs_edit = it.inf_inputs_edit
        self.inf_inputs_btn = it.inf_inputs_btn
        self.inf_targets_edit = it.inf_targets_edit
        self.inf_targets_btn = it.inf_targets_btn
    
        self.chk_inf_use_source_calib = it.chk_inf_use_source_calib
        self.chk_inf_fit_calib = it.chk_inf_fit_calib
        self.inf_calib_edit = it.inf_calib_edit
        self.inf_calib_btn = it.inf_calib_btn
        self.sp_inf_cov = it.sp_inf_cov
        self.chk_inf_include_gwl = it.chk_inf_include_gwl
        self.chk_inf_plots = it.chk_inf_plots
        self.sp_inf_batch = it.sp_inf_batch
    
        self.btn_inf_options = it.btn_inf_options
        self.btn_run_infer = it.btn_run_infer
    
        # ------------------------------
        # Transfer tab (cross-city transferability)
        # ------------------------------
        self.xfer_tab = XferTab(
            store=self.config_store,
            make_card=self._make_card,
            make_run_button=self._make_run_button,
            parent=self,
        )

        # in init tabs (similar to DataTab/SetupTab)
        self.map_tab = MapTab(
            store=self.config_store,
            parent=self,
        )

        # ==============================================================
        # Data / Setup / Preprocess (new central UI modules)
        # ==============================================================
    
        # Data tab: dataset library + editor
        data_tab = DataTab(parent=self)
    
        # Experiment Setup tab: store-backed configuration cards
        setup_tab = SetupTab(
            store=self.config_store,
            parent=self,
        )
    
        # Preprocess tab: stage1 preparation / readiness
        preprocess_tab = PreprocessTab(
            make_card=self._make_card,
            make_run_button=self._make_run_button,
            store=self.config_store,
            parent=self,
        )
    
        # Keep references on the main window
        self.data_tab = data_tab
        self.setup_tab = setup_tab
        self.preprocess_tab = preprocess_tab
    
        # Old attribute name used elsewhere (Inference)
        self.infer_tab = it
    
        # DataTab: propagate column overrides into store (feature_overrides dict)
        self.data_tab.column_overrides_changed.connect(
            lambda p: self.config_store.merge_dict_field(
                "feature_overrides",
                p,
                replace=False,
            )
        )
    
        # DataTab: show default results root from store (fallback to gui_runs_root)
        rr = self.config_store.get_value(
            FieldKey("results_root"),
            default=self.gui_runs_root,
        )
        self.data_tab.set_results_root(rr)
    
        # ==============================================================
        # Register tabs in the main QTabWidget (ordering matters)
        # ==============================================================
    
        # Data
        self._data_tab_index = self.tabs.addTab(
            self.data_tab,
            self._workflow_icon("data.svg", QStyle.SP_DirOpenIcon),
            "Data",
        )
        # Default datasets library location under results root
        self.data_tab.set_datasets_root(Path(self.gui_runs_root) / "_datasets")
    
        # Experiment Setup
        self._setup_tab_index = self.tabs.addTab(
            self.setup_tab,
            self._workflow_icon("setup.svg", QStyle.SP_FileDialogContentsView),
            "Experiment Setup",
        )
    
        # Preprocess
        self._preprocess_tab_index = self.tabs.addTab(
            self.preprocess_tab,
            self._workflow_icon("stage1.svg", QStyle.SP_DriveHDIcon),
            "Preprocess",
        )
    
        # Results tab – browse & download artifacts/runs
        results_tab = ResultsDownloadTab(
            results_root=self.gui_runs_root,
            get_results_root=lambda: self.config_store.get_value(
                FieldKey("results_root"),
                default=self.gui_runs_root,
            ),
            parent=self,
        )
        self.results_tab = results_tab
    
        # Train
        self._train_tab_index = self.tabs.addTab(
            self.train_tab,
            self._workflow_icon("train.svg", QStyle.SP_ComputerIcon),
            "Train",
        )
    
        # Tune
        self._tune_tab_index = self.tabs.addTab(
            self.tune_tab,
            self._workflow_icon("tune.svg", QStyle.SP_FileDialogDetailedView),
            "Tune",
        )
    
        # Inference
        self._infer_tab_index = self.tabs.addTab(
            self.infer_tab,
            self._workflow_icon("inference.svg", QStyle.SP_FileDialogListView),
            "Inference",
        )
    
        # Transfer
        self._xfer_tab_index = self.tabs.addTab(
            self.xfer_tab,
            self._workflow_icon("transfer.svg", QStyle.SP_ArrowRight),
            "Transfer",
        )
    
        # Results
        self._results_tab_index = self.tabs.addTab(
            self.results_tab,
            self._workflow_icon("results.svg", QStyle.SP_DirHomeIcon),
            "Results",
        )
        
        self._map_tab_index = self.tabs.addTab(
            self.map_tab,
            self._workflow_icon(
                "map.svg",
                QStyle.SP_DirIcon,
            ),
            "Map",
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
    
        # ==============================================================
        # Post-build UI sync / auto-discovery
        # ==============================================================
    
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
    
        # Default to Data tab (best first-time UX)
        try:
            self.tabs.setCurrentIndex(self._data_tab_index)
        except Exception:
            pass

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
    ) -> tuple[QWidget, QVBoxLayout]:
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
        Best-effort discovery of the latest transfer run
        under the current results_root.
    
        UI is owned by XferTab in v3.2, so we update the
        tab via its public helpers.
        """
        root_text = str(
            self.config_store.get("results_root", "")
        ).strip()
    
        if not root_text:
            self._xfer_last_result = {}
            self.xfer_tab.set_last_output(None)
            self.xfer_tab.set_view_enabled(False)
            return
    
        result = self.results_svc.discover_last_xfer(
            Path(root_text),
        )
        self._xfer_last_result = result or {}
    
        out_dir = None
        if result:
            out_dir = result.get("out_dir")
    
        self.xfer_tab.set_last_output(out_dir)
        self._update_xfer_view_state()

    # ------------------------------------------------------------------
    # Logging / progress helpers
    # ------------------------------------------------------------------
    def _on_console_visibility_changed(self, vis: bool) -> None:
        self._sync_console_menu()
    
        dock = getattr(self, "console", None)
        if dock is None or not hasattr(self, "tabs"):
            return
    
        # Only remember when NOT floating
        try:
            if dock.isFloating():
                return
        except Exception:
            pass
    
        self._remember_console_for_tab(
            tab_index=self.tabs.currentIndex(),
            visible=bool(vis),
        )
    
    def _on_console_top_level_changed(self, floating: bool) -> None:
        self._sync_console_float_menu()
    
        # If we just docked back, apply policy
        if not floating and hasattr(self, "console_policy"):
            if hasattr(self, "tabs"):
                try:
                    self.console_policy.apply(
                        self.tabs.currentIndex()
                    )
                except Exception:
                    pass

    # -----------------------------------
    # Visibility API (dock-based) 
    # -----------------------------------
    def is_console_visible(self) -> bool:
        dock = getattr(self, "console", None)
        if dock is None:
            return True
        try:
            return bool(dock.isVisible())
        except Exception:
            return True
    
    
    def _console_menu_label(self, visible: bool) -> str:
        return "Hide log panel" if visible else "Show log panel"
    
    
    def _sync_console_menu(self) -> None:
        act = getattr(self, "act_show_log", None)
        if act is None:
            return
    
        vis = bool(self.is_console_visible())
    
        try:
            with QSignalBlocker(act):
                act.setChecked(vis)
        except Exception:
            pass
    
        try:
            act.setText(self._console_menu_label(vis))
            act.setToolTip(
                "Hide the log output panel."
                if vis
                else "Show the log output panel."
            )
        except Exception:
            return
    
    
    def _remember_console_for_tab(
        self,
        *,
        tab_index: int,
        visible: bool,
    ) -> None:
        if tab_index is None:
            return
        try:
            idx = int(tab_index)
        except Exception:
            return
        if idx < 0:
            return
    
        m = getattr(self, "_console_vis_by_tab", None)
        if not isinstance(m, dict):
            m = {}
            self._console_vis_by_tab = m
    
        m[idx] = bool(visible)
    
    
    def set_console_visible(
        self,
        visible: bool,
        *,
        remember: bool = True,
    ) -> None:
        dock = getattr(self, "console", None)
        if dock is None:
            return
    
        v = bool(visible)
        try:
            dock.setVisible(v)
        except Exception:
            pass
    
        self._sync_console_menu()
    
        if not remember or not hasattr(self, "tabs"):
            return
    
        # Only remember when NOT floating
        try:
            if dock.isFloating():
                return
        except Exception:
            pass
    
        self._remember_console_for_tab(
            tab_index=self.tabs.currentIndex(),
            visible=v,
        )
    # -----------------------------------
    # Float menu sync helpers
    # -----------------------------------
    def _console_float_label(self, floating: bool) -> str:
        return "Dock log panel" if floating else "Undock log panel"
    
    
    def _sync_console_float_menu(self) -> None:
        act = getattr(self, "act_float_log", None)
        if act is None:
            return
    
        floating = bool(self.is_console_floating())
    
        try:
            with QSignalBlocker(act):
                act.setChecked(floating)
        except Exception:
            pass
    
        try:
            act.setText(self._console_float_label(floating))
            act.setToolTip(
                "Dock the log panel back into the window."
                if floating
                else "Undock the log panel into a floating window."
            )
        except Exception:
            return        
    
    # -----------------------------------
    # Floating API (dock/undock)
    # -----------------------------------
    def is_console_floating(self) -> bool:
        dock = getattr(self, "console", None)
        if dock is None:
            return False
        try:
            return bool(dock.isFloating())
        except Exception:
            return False
    

    
    def set_console_floating(self, floating: bool) -> None:
        dock = getattr(self, "console", None)
        if dock is None:
            return
    
        f = bool(floating)
    
        try:
            dock.setFloating(f)
            dock.show()
        except Exception:
            pass
    
        self._sync_console_float_menu()
    
        # When docking back, re-apply per-tab policy
        if not f and hasattr(self, "console_policy"):
            if hasattr(self, "tabs"):
                try:
                    self.console_policy.apply(
                        self.tabs.currentIndex()
                    )
                except Exception:
                    pass


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
            "data": (
                getattr(self, "_data_tab_index", -1),
                getattr(
                    self,
                    "_data_help_text",
                    "Load / inspect / edit datasets.",
                ),
            ),
            "setup": (
                getattr(self, "_setup_tab_index", -1),
                getattr(
                    self,
                    "_setup_help_text",
                    ModeManager.DEFAULT_HELP_TEXTS["setup"],
                ),
            ),
            "preprocess": (
                getattr(self, "_preprocess_tab_index", -1),
                getattr(
                    self,
                    "_preprocess_help_text",
                    ModeManager.DEFAULT_HELP_TEXTS["preprocess"],
                ),
            ),
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
            "map": (
                getattr(self, "_map_tab_index", -1),
                getattr(
                    self,
                    "_map_help_text",
                    ModeManager.DEFAULT_HELP_TEXTS["map"],
                ),
            ),
            "tools": (
                getattr(self, "_tools_tab_index", -1),
                getattr(
                    self,
                    "_tools_help_text",
                    ModeManager.DEFAULT_HELP_TEXTS["tools"],
                ),
            ),
        }

        self.mode_mgr.update_for_tab(index, self.tabs, help_texts)
        
    def _update_xfer_view_state(self) -> None:
        out_dir = None
        if self._xfer_last_result:
            out_dir = self._xfer_last_result.get("out_dir")
    
        has_result = bool(out_dir)
    
        # Visibility is handled by set_last_output().
        self.xfer_tab.set_last_output(out_dir)
    
        # Enabled/disabled state:
        self.xfer_tab.set_view_enabled(has_result)


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
        self.city_field.toast.connect(self._on_city_toast) 
        # Connect Stop button to the manager, and manager → "real" stop logic
        self.btn_stop.clicked.connect(self.mode_mgr.on_stop_clicked)
        self.mode_mgr.stop_requested.connect(self._on_stop_requested)

        self.chk_dry_run.toggled.connect(self._on_dry_run_toggled)
  
        self.log_updated.connect(
            lambda m: self.console.log_to(None, m)
        )
        self.status_updated.connect(
            lambda m: self.console.status_to(None, m)
        )
        self.progress_updated.connect(
            lambda v: self.console.progress_to(None, v, "")
        )

        # Train tab (module)
        self.train_tab.run_clicked.connect(self._on_train_clicked)
        self.train_tab.advanced_clicked.connect(
            self._on_train_options_clicked
        )
        self.train_tab.features_clicked.connect(self._on_feature_config)
        self.train_tab.arch_clicked.connect(self._on_arch_config)
        self.train_tab.prob_clicked.connect(self._on_prob_config)
        self.train_tab.physics_clicked.connect(
            self._on_physics_config_clicked
        )

        self.btn_run_tune.clicked.connect(
            self._on_tune_clicked,
        )

        # --- Inference tab ---
        it = self.inference_tab
        
        it.advanced_clicked.connect(self._on_infer_options_clicked)
        it.run_clicked.connect(self._on_infer_clicked)
        
        it.browse_model_clicked.connect(self._on_browse_model)
        it.browse_manifest_clicked.connect(self._on_browse_manifest)
        it.browse_inputs_clicked.connect(self._on_browse_inputs_npz)
        it.browse_targets_clicked.connect(self._on_browse_targets_npz)
        it.browse_calib_clicked.connect(self._on_browse_calibrator)

 
        # --- Transferability tab ---
        xt = self.xfer_tab

        xt.run_clicked.connect(self._on_xfer_clicked)
        xt.view_clicked.connect(self._on_xfer_view_clicked)


        self.data_tab.request_open.connect(
            self._on_open_dataset_data_tab
        )
        self.data_tab.request_open_new.connect(
            self._on_open_dataset_data_tab
        )
        self.data_tab.request_edit.connect(self._on_data_edit_clicked)
        self.data_tab.request_save.connect(self._on_data_save_clicked)
        self.data_tab.request_save_as.connect(self._on_data_save_as_clicked)
        self.data_tab.request_reload.connect(self._on_data_reload_clicked)
        self.data_tab.request_load_saved.connect(
            self._on_data_library_load
        )
        self.data_tab.request_duplicate_saved.connect(
            self._on_data_library_duplicate
        )
        self.data_tab.dataset_changed.connect(
            self.setup_tab.set_dataset_columns,
        )
        # DataTab: results-root controls
        self.data_tab.request_browse_results_root.connect(
            self._on_browse_results_root
        )
        self.data_tab.request_open_results_root.connect(
            self._on_open_data_results_root
        )

        pt = self.preprocess_tab
        
        pt.request_open_dataset.connect(self._on_open_dataset)
        pt.request_refresh.connect(
            lambda: pt.refresh_status(force=True)
        )
        pt.request_run_stage1.connect(self._on_preprocess_run_stage1)
        pt.request_feature_cfg.connect(self._on_open_feature_cfg)
        
        pt.request_open_manifest.connect(self._on_open_prep_manifest)
        pt.request_open_stage1_dir.connect(self._on_open_prep_stage1_dir)
        pt.request_use_for_city.connect(self._on_use_stage1_for_city)
        
        pt.request_browse_results_root.connect(
            self._on_browse_results_root
        )
        pt.request_open_city_root.connect(self._on_open_prep_city_root)

        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        def _apply_initial_tab_state() -> None:
            self._on_tab_changed(self.tabs.currentIndex())
        
        QTimer.singleShot(0, _apply_initial_tab_state)
        
    def _on_city_toast(self, status: str, msg: str) -> None:
        if msg:
            self.status_updated.emit(msg)
            
    def _get_city(self) -> str:
        if hasattr(self, "city_mgr"):
            c = self.city_mgr.get_city()
            if c:
                return c

        if hasattr(self, "city_field"):
            if self.city_field.commit():
                return self.city_mgr.get_city()

        if hasattr(self, "city_edit"):
            return self.city_edit.text().strip()

        return ""

        
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

 
    @pyqtSlot()
    def _on_open_data_results_root(self) -> None:
        rr = self.config_store.get_value(
            FieldKey("results_root"), default=None)
        target = Path(
            str(rr)).expanduser() if rr else Path(
                self.gui_runs_root)
        self._open_path(str(target))
   

    def _on_tab_changed(self, index: int) -> None:
        self._update_mode_button(index)
    
        data_idx = getattr(self, "_data_tab_index", -1)
        res_idx = getattr(self, "_results_tab_index", -1)
        tools_idx = getattr(self, "_tools_tab_index", -1)
        prep_idx = getattr(self, "_preprocess_tab_index", -1)
        setup_idx = getattr(self, "_setup_tab_index", -1)
        map_idx = getattr(self, "_map_tab_index", -1)
    
        hidden = {
            data_idx,
            setup_idx,
            map_idx,
            res_idx,
            tools_idx,
        }
    
        # Update policy with defaults and apply.
        if hasattr(self, "console_policy"):
            self.console_policy.set_hidden_tabs(hidden)
            self.console_policy.apply(index)
        else:
            hide = index in hidden
            self.set_console_visible(not hide, remember=False)
    
        if index == prep_idx:
            self.preprocess_tab.refresh_status(force=False)

        
    def _on_open_prep_city_root(self) -> None:
        pt = self.preprocess_tab
    
        city_root = pt.city_root_path()
        rr_raw = self.config_store.get_value(
            FieldKey("results_root"), default=None)
        rr = Path(str(rr_raw)).expanduser(
            ) if rr_raw else Path.home()
    
        target = Path(city_root) if city_root else rr
        if not target.exists():
            target = rr
    
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


    def _on_preprocess_run_stage1(self) -> None:
        """
        Run Stage-1 from the Preprocess tab.
        
        Store is the single source of truth:
        - city/dataset_path/results_root are patched here
        - Stage-1 options are already written by PreprocessTab checkboxes
        """
        if self.stage1_thread and self.stage1_thread.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "Stage-1 is already running.",
            )
            return
    
        city = self._get_city()
        if not city:
            QMessageBox.warning(
                self,
                "Missing city",
                "Please enter a city name first.",
            )
            return
    
        csv_path = getattr(self, "csv_path", None)
        rr = getattr(self, "gui_runs_root", None)
    
        patch: Dict[str, Any] = {"city": city}
        if csv_path is not None:
            patch["dataset_path"] = csv_path
        if rr is not None:
            patch["results_root"] = str(rr)
    
        self.config_store.patch(patch)
    
        pt = self.preprocess_tab
        pt.btn_run_stage1.setEnabled(False)
    
        # Keep legacy mirror checkboxes (if any) in sync from store
        clean = bool(
            self.config_store.get_value(
                FieldKey("clean_stage1_dir"),
                default=False,
            )
        )
        build_future = bool(
            self.config_store.get_value(
                FieldKey("build_future_npz"),
                default=False,
            )
        )
    
        if hasattr(self, "chk_clean_stage1"):
            self.chk_clean_stage1.setChecked(clean)
        if hasattr(self, "chk_build_future"):
            self.chk_build_future.setChecked(build_future)
    
        self.status_updated.emit("Stage-1: preprocessing…")
        self._update_progress(0.0)
    
        try:
            self._start_stage1(
                city=city,
                results_cb=self._on_preprocess_stage1_done,
                job_kind="preprocess",
            )
        except Exception as exc:
            pt.btn_run_stage1.setEnabled(True)
            self.status_updated.emit("Stage-1: failed to start.")
            self.log_updated.emit(f"[Stage-1] Start failed: {exc}")
            raise

    def _on_preprocess_stage1_done(self, result: Any) -> None:
        """
        Stage-1 completion handler (Preprocess tab).
        """
        self.stage1_thread = None
        self._stop_run_timer()
        self._active_job_kind = None
        self._update_global_running_state()
    
        def _get_manifest_path(obj: Any) -> str | None:
            attr = getattr(obj, "manifest_path", None)
            if attr:
                return str(attr)
            if isinstance(obj, dict):
                p = obj.get("manifest_path")
                return str(p) if p else None
            return None
    
        manifest_path = _get_manifest_path(result)
    
        btn = getattr(self.preprocess_tab, "btn_run_stage1", None)
        if btn is not None:
            btn.setEnabled(True)
    
        if not manifest_path:
            msg = "Stage-1 finished without manifest. See log."
            self.status_updated.emit(msg)
            self.log_updated.emit(f"[Stage-1] {msg}")
            self.preprocess_tab.refresh_status(force=True)
            return
    
        self._prep_last_manifest = manifest_path
    
        self.log_updated.emit(
            "[Stage-1] Finished.\n"
            f"Manifest: {manifest_path}"
        )
        self.status_updated.emit("Idle - Stage-1 ready.")
    
        self.preprocess_tab.refresh_status(force=True)



    @pyqtSlot()
    def _on_open_feature_cfg(self) -> None:
        """
        Backward-compatible alias for PreprocessTab.
    
        PreprocessTab emits `request_feature_cfg`, but the GUI
        implementation uses `_on_feature_config`.
        """
        self._on_feature_config()
    
    
    @pyqtSlot()
    def _on_browse_results_root(self) -> None:
        """
        Browse and set the GUI results root used by all tabs/workflows.
        """

        start_dir = str(getattr(self, "gui_runs_root", Path.home()))
        root = QFileDialog.getExistingDirectory(
            self,
            "Select GUI results root",
            start_dir,
        )
        if not root:
            return
    
        new_root = Path(root)
        self.config_store.patch({"results_root": new_root})

    
        # Core roots
        self.gui_runs_root = new_root
        self.results_root = new_root
    
        # Keep config in sync if field exists
        if hasattr(self.geo_cfg, "results_root"):
            self.geo_cfg.results_root = new_root
    
        # Keep RunEnv in sync (controllers share this env)
        if hasattr(self, "_run_env") and self._run_env is not None:
            self._run_env.gui_runs_root = Path(new_root)
    
        # Transfer tab line edit + refresh last xfer discovery
        if hasattr(self, "xfer_results_root"):
            self.xfer_results_root.setText(str(new_root))
            try:
                self._discover_last_xfer_for_root()
            except Exception as exc:
                self.log_updated.emit(
                    f"[WARN] Could not scan xfer runs: {exc}"
                )
    
        # DataTab datasets root
        if getattr(self, "data_tab", None) is not None:
            try:
                self.data_tab.set_datasets_root(new_root / "_datasets")
                self.data_tab.refresh_library()
                self.data_tab.set_results_root(new_root) 
            except Exception as exc:
                self.log_updated.emit(
                    f"[WARN] DataTab refresh failed: {exc}"
                )
    
        # Results tab (best-effort)
        rt = getattr(self, "results_tab", None)
        if rt is not None:
            setter = getattr(rt, "set_results_root", None)
            if callable(setter):
                try:
                    setter(new_root)
                except Exception:
                    pass
    
        self._append_status(f"[GUI] Results root set to: {new_root}")

    
    def _on_open_prep_manifest(self) -> None:
        s = getattr(self, "_prep_best_stage1", None)
        if s is None:
            return
        self._open_path(str(s.manifest_path))
    
    
    def _on_open_prep_stage1_dir(self) -> None:
        s = getattr(self, "_prep_best_stage1", None)
        if s is None:
            return
        self._open_path(str(s.run_dir))
    
    
    def _on_use_stage1_for_city(self) -> None:
        s = getattr(self, "_prep_best_stage1", None)
        if s is None:
            return
    
        city = self._get_city()
        self.set_preferred_stage1_manifest(
            city=city,
            manifest_path=str(s.manifest_path),
        )
    
    
    def _open_path(self, path_str: str) -> None:
        p = Path(path_str).expanduser()
        if not p.exists():
            QMessageBox.warning(
                self,
                "Not found",
                f"Path does not exist:\n{p}",
            )
            return
    
        try:
            if sys.platform.startswith("win"):
                import os
    
                os.startfile(str(p))
            elif sys.platform == "darwin":
                import subprocess
    
                subprocess.Popen(["open", str(p)])
            else:
                import subprocess
    
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Open failed",
                f"Could not open:\n{p}\n\n{exc}",
            )

    def _sync_data_tab(self) -> None:
        if not hasattr(self, "data_tab"):
            return
    
        self.data_tab.set_dataset(
            self.csv_path,
            self._edited_df,
            city=self._get_city(),
            dirty=False,
        )
    
    
    def _on_data_edit_clicked(self) -> None:
        if self._edited_df is None:
            QMessageBox.information(
                self,
                "No dataset",
                "Load a dataset first.",
            )
            return
    
        dlg = CsvEditDialog(self._edited_df, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
    
        new_df = dlg.edited_dataframe()
        self._edited_df = new_df
    
        # If we have a canonical path, overwrite it
        if self.csv_path:
            try:
                save_dataframe_to_csv(
                    self,
                    new_df,
                    Path(self.csv_path),
                )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Save failed",
                    str(exc),
                )
                return
    
        self._sync_data_tab()
        self.log_updated.emit("[Data] Dataset edited and saved.")
    

    def _on_data_save_clicked(self) -> None:
        if self._edited_df is None:
            QMessageBox.information(
                self,
                "No dataset",
                "Load a dataset first.",
            )
            return
    
        if not self.csv_path:
            self._on_data_save_as_clicked()
            return
    
        try:
            save_dataframe_to_csv(
                self,
                self._edited_df,
                Path(self.csv_path),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
    
        self._sync_data_tab()
        self.log_updated.emit("[Data] Dataset saved.")
    
    
    def _on_data_save_as_clicked(self) -> None:
        if self._edited_df is None:
            QMessageBox.information(
                self,
                "No dataset",
                "Load a dataset first.",
            )
            return
    
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save dataset as CSV",
            "",
            "CSV files (*.csv);;All files (*.*)",
        )
        if not out_path:
            return
    
        try:
            save_dataframe_to_csv(
                self,
                self._edited_df,
                Path(out_path),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
    
        # Update current canonical path to the new one
        self.csv_path = out_path
    
        self._sync_data_tab()
        self.log_updated.emit(
            f"[Data] Dataset saved as:\n  {out_path}"
        )
    
    
    def _on_data_reload_clicked(self) -> None:
        if not self.csv_path:
            QMessageBox.information(
                self,
                "No dataset",
                "No dataset path is known yet.",
            )
            return
    
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as exc:
            QMessageBox.critical(self, "Reload failed", str(exc))
            return
    
        self._edited_df = df
        self._sync_data_tab()
        self.log_updated.emit("[Data] Dataset reloaded from disk.")


    def _on_data_library_load(self, csv_path: str) -> None:
        p = Path(csv_path)
        if not p.exists():
            QMessageBox.warning(
                self,
                "Missing file",
                f"Dataset not found:\n{p}",
            )
            self.data_tab.refresh_library()
            return
    
        try:
            df = _load_dataset_with_progress(self, p)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load failed",
                str(exc),
            )
            return
    
        if df is None:
            return
    
        self.csv_path = str(p)
        self._edited_df = df
    
        res = self.city_mgr.resolve_city_for_open_dataset(
            typed_city=None,
            dataset_path=p,
        )
        if res.city:
            self.city_mgr.apply_resolved_city(res)
            self.city_field.refresh_all()
            
        self._sync_data_tab()
        self.data_tab.refresh_library(select_path=p)
        self.log_updated.emit(f"[Data] Loaded: {p.name}")
    
    
    def _next_versioned_path(self, src: Path) -> Path:
        """
        Produce: base_v1.csv, base_v2.csv, ...
        If src already ends with _vN, continue from N+1.
        """
        base = src.stem
        m = re.match(r"^(.*)_v(\d+)$", base)
        if m:
            base = m.group(1)
            start = int(m.group(2)) + 1
        else:
            start = 1
    
        parent = src.parent
        for i in range(start, start + 10_000):
            cand = parent / f"{base}_v{i}{src.suffix}"
            if not cand.exists():
                return cand
    
        raise RuntimeError("Could not find free version name.")
    
    
    def _on_data_library_duplicate(self, csv_path: str) -> None:
        src = Path(csv_path)
        if not src.exists():
            QMessageBox.warning(
                self,
                "Missing file",
                f"Dataset not found:\n{src}",
            )
            self.data_tab.refresh_library()
            return
    
        try:
            dst = self._next_versioned_path(src)
            shutil.copy2(src, dst)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Duplicate failed",
                str(exc),
            )
            return
    
        self.data_tab.refresh_library(select_path=dst)
        self.log_updated.emit(
            f"[Data] Duplicated:\n{src.name} -> {dst.name}"
        )

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
        kind = self._active_job_kind
        if kind in ("preprocess", "stage1"):
            self._request_stop_stage1()
            return

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
        patch = {
            "train_end_year": self.sp_train_end.value(),
            "forecast_start_year": (
                self.sp_forecast_start.value()
            ),
            "forecast_horizon_years": (
                self.sp_forecast_horizon.value()
            ),
            "time_steps": self.sp_time_steps.value(),
            "epochs": self.sp_epochs.value(),
            "batch_size": self.sp_batch_size.value(),
            "learning_rate": self.sp_lr.value(),
            "pde_mode": self.cmb_pde_mode.currentText(),
            "lambda_cons": self.sb_lcons.value(),
            "lambda_gw": self.sb_lgw.value(),
            "lambda_prior": self.sb_lprior.value(),
            "lambda_smooth": self.sb_lsmooth.value(),
            "lambda_mv": self.sb_lmv.value(),
            "clean_stage1_dir": (
                self.chk_clean_stage1.isChecked()
            ),
            "build_future_npz": (
                self.chk_build_future.isChecked()
            ),
            "evaluate_training": (
                self.chk_eval_training.isChecked()
            ),
            "tuner_search_space": (
                self._build_tuner_space_from_ui()
            ),
        }

        city = self._get_city()
        if city:
            patch["city"] = city

        if hasattr(self, "sp_phys_warmup"):
            patch["phys_warmup_epochs"] = (
                self.sp_phys_warmup.value()
            )
        if hasattr(self, "sp_phys_ramp"):
            patch["phys_ramp_epochs"] = (
                self.sp_phys_ramp.value()
            )
        if hasattr(self, "chk_scale_pde"):
            patch["scale_pde_residuals"] = (
                self.chk_scale_pde.isChecked()
            )

        self.config_store.patch(patch)

    # ------------------------------------------------------------
    #     configure dialog boxes 
    # ------------------------------------------------------------
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
        overs = dlg.get_overrides() or {}
        self.config_store.patch({"feature_overrides": overs})
 
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
        # Flush Train tab widgets first (safe, consistent)
        self._sync_config_from_ui()
    
        before = ProbConfigDialog.snapshot(self.config_store)
    
        ok = ProbConfigDialog.edit(
            store=self.config_store,
            parent=self,
        )
        if not ok:
            return
    
        after = ProbConfigDialog.snapshot(self.config_store)
    
        keys = ["quantiles", "subs_weights", "gwl_weights"]
        changed = [k for k in keys if before.get(k) != after.get(k)]
    
        msg = "Probabilistic configuration updated"
        if changed:
            msg = f"{msg} (keys: {', '.join(changed)})."
        else:
            msg = f"{msg} (no changes)."
    
        self.log_updated.emit(msg)


    @pyqtSlot()
    def _on_physics_config_clicked(self) -> None:
        """
        Open PhysicsConfigDialog and persist changes into GeoPriorConfig
        via GeoConfigStore (single source of truth).
        """
        # Ensure Train tab widgets are pushed into the config first
        self._sync_config_from_ui()
    
        # NEW API: dialog returns a patch dict using GeoPriorConfig keys
        patch = PhysicsConfigDialog.edit(
            parent=self,
            store=self.config_store,
        )
        if not patch:
            return
    
        try:
            # Optional safety check
            self.geo_cfg.ensure_valid()
    
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Physics configuration",
                f"Could not apply physics config:\n\n{exc}",
            )
            return
    
        keys = ", ".join(sorted(patch.keys()))
        self.log_updated.emit(
            f"[Physics] Updated: {keys}"
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
        if hasattr(self, "city_mgr"):
            c = self.city_mgr.get_city()
            if c:
                city = c

        return self._get_preferred_stage1_manifest_for_city(city)

    # --------------------------------------------------------------
    # Tuner search space <-> UI
    # --------------------------------------------------------------
    def _load_tuner_space_into_ui(self) -> None:
        self.tune_tab.refresh_from_store()
    
    def _build_tuner_space_from_ui(self) -> Dict[str, Any]:
        return self.tune_tab.build_space_from_ui()

    
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
        City-bound dataset open.
    
        - Uses CityManager lock rules
        - Saves canonical CSV under _datasets/<city>.csv
        - Patches store with dataset_path + city
        - Ensures city root exists
        """
        mgr = getattr(self, "city_mgr", None)
        if mgr is None:
            self.log_updated.emit("[WARN] CityManager missing")
            return
    
        hint = None
        if mgr.is_locked():
            cur = mgr.get_city()
            if cur:
                hint = cur
        else:
            if hasattr(self, "city_field"):
                hint = self.city_field.city_text()
    
        def _msg(m: str) -> None:
            if m:
                self.status_updated.emit(m)
    
        csv_path, df, city_key = open_dataset_with_editor(
            self,
            gui_runs_root=Path(self.gui_runs_root),
            initial_dir="",
            city_hint=hint,
            normalize_city=mgr.normalize,
            city_message_hook=_msg,
        )
        if csv_path is None:
            return
    
        self.csv_path = str(csv_path)
        self._edited_df = df
    
        if city_key:
            mgr.set_city(city_key, quiet=True)
            mgr.ensure_city_root(
                results_root=Path(self.gui_runs_root),
            )
            if hasattr(self, "city_field"):
                self.city_field.refresh_all()
    
        try:
            self.config_store.patch(
                {
                    "dataset_path": str(self.csv_path),
                    "city": mgr.get_city(),
                    "results_root": str(self.gui_runs_root),
                }
            )
        except Exception as exc:
            self.log_updated.emit(f"[WARN] config patch: {exc}")
    
        if self.data_tab is not None:
            self.data_tab.set_dataset(
                self.csv_path,
                self._edited_df,
                city=mgr.get_city(),
                dirty=True,
            )
        elif self.setup_tab is not None and df is not None:
            self.setup_tab.set_dataset_columns(
                [str(c) for c in df.columns]
            )
    
        self.log_updated.emit(f"Dataset selected: {self.csv_path}")
    
        if not getattr(self, "_feature_tip_shown", False):
            msg = QMessageBox(self)
            msg.setWindowTitle("Check feature configuration")
            msg.setIcon(QMessageBox.Information)
            msg.setTextFormat(Qt.RichText)
            msg.setText(
                "Dataset selected.<br><br>"
                "Please open the <b>“Feature config…”</b> "
                "box to verify the dynamic features and "
                "H-field column before running the PINN."
            )
            msg.exec_()
            self._feature_tip_shown = True
    
        self._sync_data_tab()
        self.data_tab.refresh_library(select_path=self.csv_path)

    @pyqtSlot()
    def _on_open_dataset_data_tab(self) -> None:
        """
        Data-only open from DataTab.
    
        - Lets dataset name follow file stem
        - Does NOT patch city
        - Does NOT create city root
        """
        mgr = getattr(self, "city_mgr", None)
        if mgr is None:
            return
    
        def _msg(m: str) -> None:
            if m:
                self.status_updated.emit(m)
    
        csv_path, df, _city_key = open_dataset_with_editor(
            self,
            gui_runs_root=Path(self.gui_runs_root),
            initial_dir="",
            city_hint=None,
            normalize_city=mgr.normalize,
            city_message_hook=_msg,
        )
        if csv_path is None:
            return
    
        self.csv_path = str(csv_path)
        self._edited_df = df
    
        try:
            self.config_store.patch(
                {
                    "dataset_path": str(self.csv_path),
                    "results_root": str(self.gui_runs_root),
                }
            )
        except Exception as exc:
            self.log_updated.emit(f"[WARN] config patch: {exc}")
    
        if self.data_tab is not None:
            self.data_tab.set_dataset(
                self.csv_path,
                self._edited_df,
                city=mgr.get_city(),
                dirty=True,
            )
    
        self._sync_data_tab()
        self.data_tab.refresh_library(select_path=self.csv_path)

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
            city_text=self._get_city()
    
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
            city_text=self._get_city()
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
    
        city_text=self._get_city()
    
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
    
        city_text=self._get_city()
    
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
            self.console.bind_thread(
              th,
              kind="tune",
              title="Tune",
              start_msg="Tuning…",
             )
                
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
        
        # Update inference tab "Quick actions" with artifacts
        if hasattr(self, "inference_tab") and self.inference_tab:
            self.inference_tab.set_last_outputs(
                {
                    "run_dir": run_dir or "",
                    "csv_eval_path": csv_eval or "",
                    "csv_future_path": csv_future or "",
                    "inference_summary_json": summary_json or "",
                }
            )


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
            self.console.bind_thread(
                th,
                kind="infer",
                title="Inference",
                start_msg="Inference…",
                meta={"dataset": plan.dataset_key, "out_dir": plan.stage1_dir}
            )
            
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
    def _build_xfer_state(self) -> TransferGuiState:
        raw = self.xfer_tab.get_state()
    
        return TransferGuiState(
            city_a=str(raw.get("city_a", "")).strip(),
            city_b=str(raw.get("city_b", "")).strip(),
            results_root=raw.get("results_root") or None,
            splits=list(raw.get("splits") or []),
            calib_modes=list(raw.get("calib_modes") or []),
            rescale_to_source=bool(
                raw.get("rescale_to_source", False)
            ),
            batch_size=int(raw.get("batch_size", 32)),
            quantiles_override=raw.get(
                "quantiles_override",
                None,
            ),
            write_json=bool(raw.get("write_json", True)),
            write_csv=bool(raw.get("write_csv", True)),
        )
        
    
    @pyqtSlot()
    def _on_xfer_clicked(self) -> None:
        """
        Run cross-city transfer matrix.
    
        UI values are collected from XferTab (store-backed).
        Planning/validation is still delegated to the controller.
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
    
        state = self._build_xfer_state()
    
        if self._is_dry_mode():
            self.transfer_controller.dry_preview(state)
            return
    
        # 3) real run callback
        def _start_xfer_from_plan(
            plan: TransferPlan,
        ) -> None:
            self._start_run_timer()
    
            # v3.2 extras live in the store and are already
            # persisted by XferTab, so the thread can read
            # them if it forwards store into run_xfer_matrix.
            
            th = XferMatrixThread(
                city_a=plan.city_a,
                city_b=plan.city_b,
                store=self.config_store,
                results_root=str(plan.results_root),
                splits=plan.splits,
                calib_modes=plan.calib_modes,
                rescale_to_source=plan.rescale_to_source,
                batch_size=plan.batch_size,
                quantiles_override=plan.quantiles_override,
                write_json=plan.write_json,
                write_csv=plan.write_csv,
                parent=self,
            )

            self.xfer_thread = th
    
            self.console.bind_thread(
                th,
                kind="xfer",
                title="Transfer matrix",
                start_msg="Transferability…",
                meta={"city_a": plan.city_a, "city_b": plan.city_b},
            )
            
            th.error_occurred.connect(self._on_worker_error)
            th.xfer_finished.connect(self._on_xfer_finished)
    
            self.xfer_tab.set_run_enabled(False)
    
            self._active_job_kind = "xfer"
            self._update_global_running_state()
    
            th.start()
            self._update_global_running_state()
    
        self.transfer_controller.start_real_run(
            state,
            _start_xfer_from_plan,
        )
    
    
    @pyqtSlot(dict)
    def _on_xfer_finished(
        self,
        result: Dict[str, Any],
    ) -> None:
        self.xfer_thread = None
        self.xfer_tab.set_run_enabled(True)
    
        if not result:
            self.log_updated.emit(
                "Transfer matrix finished with an empty "
                "result dict."
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
    
        self._xfer_last_result = result
    
        if out_dir:
            self.log_updated.emit(
                "Transferability artifacts in:\n"
                f"  {out_dir}"
            )
            self.xfer_tab.set_last_output(out_dir)
    
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
    
        # Show summary dialog (local import keeps app.py clean)
        view_split = "val"
        try:
            view_split = (
                self.xfer_tab.get_state()
                .get("view_split", "val")
            )
        except Exception:
            view_split = "val"
    
        if XferResultsDialog is not None:
            XferResultsDialog.show_for_xfer_result(
                parent=self,
                result=result,
                split=view_split,
                title="Cross-city transfer summary",
            )
    
        self._save_gui_log_for_result(result)
    
        self.status_updated.emit(
            "Idle – transferability complete."
        )
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
                "No transfer results found. "
                "Run the transfer matrix first.",
            )
            return
    
        raw = self.xfer_tab.get_state()
    
        view_kind = str(
            raw.get("view_kind", "calib_panel")
        )
        view_split = str(
            raw.get("view_split", "val")
        )
    
        results_root = raw.get("results_root")
        if not results_root:
            results_root = str(self.gui_runs_root)
    
        r = self._xfer_last_result or {}
        out_dir = r.get("out_dir")
        csv_path = r.get("csv_path")
        json_path = r.get("json_path")
    
        if not (csv_path or json_path or out_dir):
            QMessageBox.warning(
                self,
                "No artifacts",
                "Could not find xfer_results.* in "
                "the last output folder.",
            )
            return
    
        self.log_updated.emit(
            f"Build transferability view ({view_kind}) "
            f"from {csv_path or json_path or 'latest'}."
        )
        self.status_updated.emit(
            "Rendering transferability view…"
        )
        self._update_progress(0.0)
    
        th = XferViewThread(
            view_kind=view_kind,
            results_root=str(results_root),
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
    
        self.console.bind_thread(
            th,
            kind="xfer_view",
            title="Transfer view(city)",
            start_msg="viewing…",
        )
        
        th.error_occurred.connect(self._on_worker_error)
        th.xfer_view_finished.connect(
            self._on_xfer_view_finished
        )
    
        self.xfer_tab.set_view_enabled(False)
        th.start()
    
    
    @pyqtSlot(dict)
    def _on_xfer_view_finished(
        self,
        result: Dict[str, Any],
    ) -> None:
        self.xfer_view_thread = None
        self.xfer_tab.set_view_enabled(True)
    
        if not result:
            self.log_updated.emit(
                "Transferability view finished with an "
                "empty result dict."
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
    
        for key in (
            "svg_path",
            "pdf_path",
            "table_csv",
            "table_tex",
        ):
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
    
        self.log_updated.emit(
            "View summary: " + ", ".join(bits)
        )
    
        src = result.get("source_city")
        tgt = result.get("target_city")
    
        self.log(
            "[Xfer view] Transferability figure saved:\n"
            f"  kind   : {view_kind}\n"
            f"  source : {src}\n"
            f"  target : {tgt}\n"
            f"  file   : {png}"
        )
    
        self.status_updated.emit(
            "Idle – transferability view ready."
        )
        _notify_gui_xfer_view(result)
        self._update_progress(1.0)


    # ------------------------------------------------------------------
    # Thread orchestration
    # ------------------------------------------------------------------
    def _start_stage1(
        self,
        city: str,
        *,
        results_cb=None,
        job_kind: str = "stage1",
    ) -> None:
        results_root = getattr(self, "results_root", self.gui_runs_root)
    
        # Sync GUI → GeoPriorConfig (v3.2 source of truth)
        try:
            self._sync_config_from_ui()
        except Exception:
            pass
    
        # Build overrides from GeoPriorConfig v3.2
        cfg_overrides = {}
        try:
            cfg_overrides = self.geo_cfg.to_cfg_overrides()
        except Exception:
            cfg_overrides = {}
    
        # Merge any controller-provided overrides (TrainController)
        prev = getattr(self, "_cfg_overrides", {}) or {}
        if isinstance(prev, dict):
            cfg_overrides.update(prev)
    
        edited_df = getattr(self, "_edited_df", None)
    
        # If no in-memory dataset, use selected CSV or resolve by city
        if edited_df is None:
            csv_path = getattr(self, "csv_path", None)
    
            if csv_path is None:
                csv_path_str = choose_dataset_for_city(
                    parent=self,
                    city=city,
                    results_root=Path(results_root),
                )
                if not csv_path_str:
                    return
                csv_path = Path(csv_path_str)
            else:
                csv_path = Path(str(csv_path))
    
            cfg_overrides["DATA_DIR"] = str(csv_path.parent)
            cfg_overrides["BIG_FN"] = csv_path.name
    
            self.log_updated.emit(
                f"[Stage-1] Using dataset: {csv_path.name} "
                f"({csv_path.parent})"
            )
    
        # Keep stable copy for Stage1Thread + training reuse
        self._cfg_overrides = dict(cfg_overrides)
    
        th = Stage1Thread(
            city=city,
            cfg_overrides=self._cfg_overrides,
            clean_run_dir=self.geo_cfg.clean_stage1_dir,
            base_cfg=self.geo_cfg._base_cfg,
            edited_df=edited_df,
            results_root=str(results_root),
            parent=self,
        )
        self.stage1_thread = th
        
        job_kind = job_kind  # "preprocess" or "stage1"
        title = "Preprocess" if job_kind == "preprocess" else "Stage-1"
        
        self.console.bind_thread(
          th,
          kind=job_kind,
          title=f"{title} ({city})",
          start_msg=f"{title}: preprocessing…",
        )
        # th.log_updated.connect(self.log_updated.emit)
        # th.status_updated.connect(self.status_updated.emit)
        # th.progress_changed.connect(self._on_thread_progress)
    
        cb = results_cb or self._on_stage1_finished
        th.results_ready.connect(cb)
    
        th.error_occurred.connect(self._on_worker_error)
    
        self._active_job_kind = job_kind
        self._update_global_running_state()
        self._start_run_timer()
    
        th.start()


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
            
        # --- Training lifecycle overwrite (UI-only keys) ---
        st = getattr(self, "config_store", None)
        life = "new"
        base = ""
    
        if st is not None:
            try:
                life = str(
                    st.get("train.lifecycle", "new")
                ).strip().lower()
                base = str(
                    st.get("train.base_model_path", "")
                ).strip()
            except Exception:
                life = "new"
                base = ""
    
        if life not in ("new", "resume", "finetune"):
            life = "new"
    
        cfg_over = {"TRAIN_LIFECYCLE": life}
    
        if life != "new":
            from pathlib import Path
    
            p = Path(base).expanduser()
            if not base or not p.exists():
                QMessageBox.warning(
                    self,
                    "Missing base model",
                    "Resume/Fine-tune needs a valid base "
                    "model path (.keras / .weights.h5 "
                    "or a run directory).",
                )
                self.train_btn.setEnabled(True)
                if hasattr(self, "btn_train_options"):
                    self.btn_train_options.setEnabled(True)
                self._active_job_kind = None
                self._update_global_running_state()
                self._stop_run_timer()
                return
    
            cfg_over["BASE_MODEL_PATH"] = str(p) 
            
        th = TrainingThread(
            manifest_path=manifest_path,
            cfg_overrides=cfg_overrides,
            evaluate_training=self.geo_cfg.evaluate_training,
            results_root=self.geo_cfg.results_root,
            base_cfg=self.geo_cfg._base_cfg,
            parent=self,
            store=st,
            config_overwrite=cfg_over,
        )
     # TODO : IMPLEMENT THE RUN ID to get it here 
        self.train_thread = th
    
        self.console.bind_thread(
            th,
            kind="train",
            title="Train (city)",
            start_msg="Stage-2: training…",
            meta={"city": self._get_city(), "run_id": run_id}
        )

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

        self.xfer_tab.set_has_result(True)

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
        Intercept window close (File -> Exit, Ctrl+Q, window X).
    
        If long-running workflows are active, ask for confirmation
        before closing. On close, persist window geometry so the next
        launch restores the same size/position.
        """
        if self._any_job_running():
            msg = (
                "One or more workflows are still running "
                "(Stage-1, training, tuning, inference or transfer "
                "matrix).\n\n"
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
    
        # Persist window geometry (best-effort).
        save_window_geometry(
            self,
            settings_key="main_window",
        )
        super().closeEvent(event)
    

# ----------------------------------------------------------------------
# Entry point helper
# ----------------------------------------------------------------------
def launch_geoprior_gui(theme: str = "fusionlab") -> None:
    app = QApplication(sys.argv)

    # QSettings identity (needed for geometry persistence).
    set_app_metadata(
        app,
        org_name="FusionLab",
        app_name="GeoPrior",
        org_domain="EarthAI-tech",
    )

    cfg = GeoPriorConfig.from_defaults()
    theme = getattr(cfg, "ui_theme", theme) or "fusionlab"

    # Apply global UI font (and scale).
    scale = float(getattr(cfg, "ui_font_scale", 1.0))
    auto_set_ui_fonts(app, font_scale=scale)

    enable_qt_crash_handler(
        app,
        keep_gui_alive=False,
        show_dialog=False,
    )

    # --- splash with logo ---
    logo_path = Path(__file__).with_name(
        "geoprior_splash.png",
    )
    splash = LoadingSplash(logo_path)
    splash.show()
    app.processEvents()

    splash.set_progress(10, "Loading configuration...")

    # Create main window (pass splash for build progress).
    gui = GeoPriorForecaster(
        theme=theme,
        splash=splash,
    )

    # Larger default size + restore last geometry if available.
    auto_resize_window(
        gui,
        settings_key="main_window",
    )

    splash.set_progress(100, "Ready")
    splash.finish(gui)

    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # High-DPI attributes should be set before QApplication exists
    QApplication.setAttribute(
        Qt.AA_EnableHighDpiScaling,
        True,
    )
    QApplication.setAttribute(
        Qt.AA_UseHighDpiPixmaps,
        True,
    )

    launch_geoprior_gui()

