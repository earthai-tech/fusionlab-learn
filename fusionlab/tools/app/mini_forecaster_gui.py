"""
Mini Subsidence-Forecasting GUI (academic showcase)
"""

from __future__ import annotations 
import os, sys, time

import warnings
import pandas as pd 
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap,  QFont
from PyQt5.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout,
    QFormLayout, 
    QFrame, 
    QPushButton, 
    QLabel,
    QSpinBox, 
    QDoubleSpinBox,
    QComboBox, 
    QTextEdit, 
    QFileDialog, 
    QProgressBar, 
    QLineEdit, 
    QCheckBox,
    QDialog, 
    QMessageBox,
    QToolTip, 
)

from ...registry import ManifestRegistry, _locate_manifest
from .components import ( 
    ProgressManager, 
    WorkerController, 
    ErrorManager,
    ExitController, 
    ModeSwitch
)

from .config import SubsConfig
from .dialog import CsvEditDialog, TunerDialog, ModelChoiceDialog
from .gui_popups import ImagePreviewDialog 
from .notifications import ToastNotification 
from .styles import ( 
    PRIMARY, SECONDARY, 
    FLAB_STYLE_SHEET,
    INFERENCE_OFF, 
    DARK_THEME_STYLESHEET,
    LOG_STYLES
    )
from .threads import TrainingThread, InferenceThread, TunerThread 
from .view import VIS_SIGNALS
  
class MiniForecaster(QMainWindow):
    """The main application window for the Subsidence PINN Mini GUI.

    This class constructs and manages the entire graphical user
    interface. It is responsible for initializing all UI components
    (buttons, input fields, display panels), connecting user actions
    (signals) to their corresponding handler functions (slots), and
    managing the application's state, including the backend `Worker`
    thread that runs the forecasting process.

    Attributes
    ----------
    log_updated : pyqtSignal(str)
        A signal connected to the `_log` slot for updating the UI log.
    status_updated : pyqtSignal(str)
        A signal connected to the status label for displaying current
        workflow status.
    progress_updated : pyqtSignal(int)
        A signal connected to the progress bar for updates.
    coverage_ready : pyqtSignal(float)
        A signal connected to the coverage label for displaying the
        final coverage score.
    worker : Worker
        An instance of the background thread that executes the main
        processing logic.
    """
    
    log_updated      = pyqtSignal(str)
    status_updated   = pyqtSignal(str)
    coverage_ready = pyqtSignal(float) 
    trial_updated  = pyqtSignal(int, int, str)   


    def __init__(self, theme: str = 'light'):
        super().__init__()
        self._log_cache: list[str] = []
        self.setWindowTitle("Fusionlab-learn – PINN Mini GUI")
        self.setFixedSize(980, 660)
        self.file_path: Path | None = None
        
        # app icon (title-bar & task-bar) 
        icon_path = os.path.join(os.path.dirname(__file__),
                                 "fusionlab_learn_logo.ico")
        app_icon  = QIcon(icon_path)          
        self.setWindowIcon(app_icon)
        
        # --- in-window logo (top of GUI) 
        logo_lbl  = QLabel()
        pix_path  = os.path.join(os.path.dirname(__file__), 
                                 "fusionlab_learn_logo.png")
        pix_logo  = QPixmap(pix_path)
        
        if not pix_logo.isNull():   
            logo_lbl.setPixmap(
                pix_logo.scaled(72, 72, Qt.KeepAspectRatio,
                                Qt.SmoothTransformation)
            )
        else:
            logo_lbl.setText("Fusionlab-learn") # graceful fallback 
        
        logo_lbl.setAlignment(Qt.AlignCenter)
        
        # --- Instantiate the Manifest Registry in session-only mode ---
        self.registry = ManifestRegistry(session_only=True)
        
        self._inference_mode = False     # True → Run button will run inference
        self._manifest_path  = None      # filled automatically once detected
        
        # --- Store the active theme ---
        self.theme = ( 
            "dark" if str(theme).lower() =='dark' else'fusionlab'
        )
        
        self._build_ui()
        self.progress_manager = ProgressManager(
                self.progress_bar, 
                self.progress_label, 
                self.percent_label,        # NEW argument
            )
      
        # already have self._log; add after progress_manager
        self.error_mgr = ErrorManager(
            parent_gui = self, log_fn= self._log)

        self.worker_ctl = WorkerController(
                self.stop_btn,
                parent_gui = self,                 # QMessageBox parent
                log_fn     = self._log,
                status_fn  = self.status_updated.emit,   # one-liner status update
        )
        self.worker_ctl.stopped.connect(self._worker_done)
        
        # MiniForecaster.__init__  (just after self.worker_ctl creation)
        self.exit_ctl = ExitController(
            quit_button  = self.quit_btn,      # may be None if you only want title-bar
            parent_gui   = self,
            worker_ctl   = self.worker_ctl,    # so it knows if something's running
            pre_quit_hook= lambda: self._log("✔ clean-up done"),
            log_fn       = self._log,
        )
        
        
        # optional: reset progress-bar when an error dialog closes
        self.error_mgr.handled.connect(self.progress_manager.reset)
        self.tune_mode = ModeSwitch(
            button          = self.tune_btn,
            tint            = SECONDARY,
            tooltip_running = "Tuning in progress – Inference disabled",
            tooltip_idle    = "Click to configure hyper-parameter tuning",
        )
        
        self.log_updated.connect(self._log)
        self.status_updated.connect(self.file_label.setText)
        # self.progress_updated.connect(self.progress_bar.setValue)
        self.coverage_ready.connect(self._set_coverage)
        # self.progress_updated.connect(self._update_progress)
        self._refresh_manifest_state() 
        
        self._preview_windows: list[QDialog] = []        # ← keep refs
        VIS_SIGNALS.figure_saved.connect(self._show_image_popup)

    def _show_image_popup(self, png_path: str) -> None:
        dlg = ImagePreviewDialog(png_path, parent=self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)             # frees memory on close
        dlg.show()                                        # modeless – *no* exec_()
        self._preview_windows.append(dlg)                 # keep it alive
        
    @pyqtSlot(float)
    def _set_coverage(self, value: float):
        # orange & bold *value* only
        self.coverage_lbl.setText(
            f"cov-result:&nbsp;<span style='color:{SECONDARY}; "
            f"font-weight:bold'>{value:.3f}</span>"
        )

    def _build_ui(self):
        """Constructs and assembles the entire GUI layout.

        This method acts as the main orchestrator for building the user
        interface. It is called once during the `__init__` process to
        create all the necessary widgets and arrange them into a
        cohesive layout.

        It follows a structured approach:
        1.  It calls the various private factory methods (`_create_header`,
            `_model_card`, etc.) to instantiate all the individual UI
            panels.
        2.  It creates the main vertical and horizontal layout managers
            that define the overall structure of the application window.
        3.  It adds each created panel and widget into the appropriate
            layout, organizing the configuration controls on the left and
            the log/output panel on the right.
        4.  It sets up the main application footer containing informational
            links and the results display area.

        This method modifies the `self` instance by attaching all created
        UI components as attributes.

        See Also
        --------
        _create_header : Creates the top banner with the logo and title.
        _model_card : Creates the panel for model architecture settings.
        _training_card : Creates the panel for training run parameters.
        _physics_card : Creates the panel for PINN-specific parameters.
        _feature_card : Creates the panel for feature selection inputs.
        """
        root = QWidget(); self.setCentralWidget(root)
        L = QVBoxLayout(root)

        # 0) title
        title = QLabel("Subsidence PINN")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size:22px; color:{PRIMARY}")
        # L.addWidget(logo)
        L.addWidget(title)

        # 1) CSV selector row 
        csv_row = QHBoxLayout()
        csv_row.setSpacing(8)
        
        # left-hand “Select file…” button
        self.file_btn = QPushButton("Select CSV…")
        self.file_btn.clicked.connect(self._choose_file)
        csv_row.addWidget(self.file_btn)
        
        # centre: file-name label grows/shrinks with window
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("font-style:italic;")
        csv_row.addWidget(self.file_label, 1)          # stretch-factor = 1
        
        # ── Tune button  (disabled until a CSV + manifest-able params exist)
        self.tune_btn = QPushButton("Tune")
        self.tune_btn.setObjectName("tune")
        self.tune_btn.setEnabled(False)
        self.tune_btn.setToolTip("Select a CSV first, then click to open tuner setup")
        self.tune_btn.clicked.connect(self._open_tuner_dialog)
        csv_row.addWidget(self.tune_btn)

        # ── inference toggle  (disabled until a manifest is found)
        self.inf_btn = QPushButton("Inference")
        self.inf_btn.setObjectName("inference")
        self.inf_btn.setEnabled(False)                       # default: grey
        self.inf_btn.setToolTip("Load an existing run_manifest.json first.")
        self.inf_btn.clicked.connect(self._toggle_inference_mode)
        csv_row.addWidget(self.inf_btn)
        
        # ► Make them the same width  
        same_w = self.inf_btn.sizeHint().width()      
        self.tune_btn.setFixedWidth(same_w)

        # build stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop")   
        self.stop_btn.setEnabled(False) 
        self.stop_btn.setToolTip("Abort the running workflow")    
        csv_row.addWidget(self.stop_btn)
        
        # right-hand Reset button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("reset")   
        self.reset_btn.setToolTip("Clear selections & log")
        self.reset_btn.setFixedWidth(70)

        self.reset_btn.clicked.connect(self._on_reset)
        csv_row.addWidget(self.reset_btn)
        
        # >>> NEW Quit button -------------------------------------------------
        self.quit_btn = QPushButton("Quit")
        self.quit_btn.setObjectName("quit")
        self.quit_btn.setToolTip(
            "Exit Fusionlab-learn")
        csv_row.addWidget(self.quit_btn)
  
        L.addLayout(csv_row)
        
        # 1-bis)  City / Dataset name row  ← NEW
        city_row = QHBoxLayout()
        city_label = QLabel("City / Dataset:")
        city_row.addWidget(city_label)
        
        self.city_input = QLineEdit()
        self.city_input.setPlaceholderText("e.g. zhongshan")
        city_row.addWidget(self.city_input, 1)  # stretch to full width
        L.addLayout(city_row)

        # 2) config cards --
        cards = QHBoxLayout();
        L.addLayout(cards, 1)
        
        # Store the returned QFrame widgets as instance attributes ------------
        self.model_card = self._model_card()
        self.training_card = self._training_card()
        self.physics_card = self._physics_card()
        self.feature_card = self._feature_card()
        
        cards.addWidget(self.model_card, 1)
        cards.addWidget(self.training_card, 1)
        cards.addWidget(self.physics_card, 1)
        # ▼ NEW feature card spans full width
        L.addWidget(self.feature_card)

        # 3)  Run row  +  Log pane  +  Progress bar  
        bottom = QVBoxLayout()
        L.addLayout(bottom)
        
        # ── row 1 : Run  +  log --
        row = QHBoxLayout()
        
        self.run_btn = QPushButton("Run")
        self.run_btn.setFixedWidth(80)
        self.run_btn.clicked.connect(self._on_run)
        row.addWidget(self.run_btn)
        
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        row.addWidget(self.log, 1)       # stretch
        bottom.addLayout(row)            # ← add FIRST
        
        progress_layout = QHBoxLayout()
    
        # This label will display ETA, Trial, or Epoch info
        self.progress_label = QLabel("")
        self.progress_label.setFixedWidth(120) # Reserve space for text
        self.progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(18)
        # self.progress_bar.setTextVisible(True)
        self.progress_bar.setTextVisible(False)     # we now have a separate % label
        # progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar, 1) 
        
        # Percentage label (right)
        self.percent_label = QLabel("0 %")
        self.percent_label.setFixedWidth(30)
        self.percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        progress_layout.addWidget(self.percent_label)
        
        bottom.addLayout(progress_layout)
        # ── single-row footer  -----------------------------------------------
        footer = QHBoxLayout()
        
        # Learn more” link
        learn = QLabel(
            'About <a href="https://fusion-lab.readthedocs.io/" '
            'style="color:#2E3191;text-decoration:none;">Fusionlab-learn</a>'
        )
        learn.setOpenExternalLinks(True)
        learn.setStyleSheet("font-size:10px;")
        footer.addWidget(learn)
        
        footer.addStretch(1)                 # push next label to the far right
        
        # ❶ coverage label starts empty – will be filled later
        self.coverage_lbl = QLabel("")
        self.coverage_lbl.setObjectName("covLabel") 
        self.coverage_lbl.setStyleSheet("font-size:10px;")
        footer.addWidget(self.coverage_lbl)
        
        footer.addStretch(1)

        # ② copyright / licence 
        about = QLabel(
            '© 2025 <a href="https://earthai-tech.github.io/" '
            'style="color:#2E3191;text-decoration:none;">earthai-tech</a> – BSD-3 Clause'
        )
        about.setOpenExternalLinks(True)
        about.setStyleSheet("font-size:10px;")
        footer.addWidget(about)
        
        bottom.addLayout(footer)
        

    def _training_card(self) -> QFrame:
        """Creates and returns the 'Training Parameters' UI panel.

        This is a private factory method responsible for constructing
        the `QFrame` that holds all widgets related to defining the
        temporal aspects of the training and forecasting process.

        Returns
        -------
        PyQt5.QtWidgets.QFrame
            A fully populated `QFrame` widget containing the input fields
            for 'Train End Year', 'Forecast Start Year', 'Forecast Horizon',
            'Time Steps', 'Quantiles', and 'Checkpoint format'.
        """
        card = QFrame(); card.setObjectName("card")
        lay  = QVBoxLayout(card)
        lay.addWidget(self._title("Training Parameters"))
        lay.addWidget(hline())
    
        form = QFormLayout()
    
        # Train End Year
        self.train_end_year_spin = QSpinBox()
        self.train_end_year_spin.setRange(1980, 2099)
        self.train_end_year_spin.setValue(2022)
        form.addRow("Train End Year:", self.train_end_year_spin)
    
        # Forecast Start Year
        self.forecast_start_year_spin = QSpinBox()
        self.forecast_start_year_spin.setRange(1980, 2050)
        self.forecast_start_year_spin.setValue(2023)
        form.addRow("Forecast Start Year:", self.forecast_start_year_spin)
    
        # Forecast Horizon (years)
        self.forecast_horizon_spin = QSpinBox()
        self.forecast_horizon_spin.setRange(1, 20)
        self.forecast_horizon_spin.setValue(3)
        form.addRow("Forecast Horizon (years):", self.forecast_horizon_spin)
    
        # Time Steps
        self.time_steps_spin = QSpinBox()
        self.time_steps_spin.setRange(1, 50)
        self.time_steps_spin.setValue(5)
        form.addRow("Time Steps (look-back):", self.time_steps_spin)
    
        # Quantiles
        self.quantiles_input = QLineEdit()
        self.quantiles_input.setText("0.1, 0.5, 0.9")
        form.addRow("Quantiles (comma-separated):", self.quantiles_input)
        
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["weights", "keras", "tf"])
        self.save_format_combo.setCurrentText("weights")
        self.save_format_combo.setToolTip(
            "On-GUI training only supports 'weights' reliably.\n"
            "Select 'keras' or 'tf' if you plan to train outside the GUI.")
        form.addRow("Checkpoint format:", self.save_format_combo)
    
        lay.addLayout(form)
        return card

    def _model_card(self) -> QFrame:
        """Creates and returns the 'Model Configuration' UI panel.

        This private factory method is responsible for constructing the
        QFrame widget that holds all UI controls related to the core
        model architecture and main training loop settings. It gathers
        high-level choices that define the experiment.

        The created panel includes widgets for:
        - Selecting the model architecture (e.g., 'TransFlowSubsNet').
        - Setting the number of training Epochs.
        - Defining the Batch Size.
        - Specifying the Learning Rate.
        - Choosing the internal model mode ('pihal' or 'tft').
        - Configuring attention levels and evaluation metrics.

        Returns
        -------
        PyQt5.QtWidgets.QFrame
            A fully populated QFrame widget containing all the necessary
            UI elements for model configuration.

        See Also
        --------
        SubsConfig : The configuration object that these UI settings populate.
        """
        card = QFrame(); card.setObjectName("card")
        lay  = QVBoxLayout(card)
        lay.addWidget(self._title("Model Configuration"))
        lay.addWidget(hline())

        form = QFormLayout()
        self.model_select = QComboBox()
        self.model_select.addItems(["TransFlowSubsNet", "PIHALNet"])
        form.addRow("Architecture:", self.model_select)

        self.epochs = QSpinBox(); self.epochs.setRange(1, 1000)
        self.epochs.setValue(50)
        form.addRow("Epochs:", self.epochs)

        self.batch = QSpinBox(); self.batch.setRange(8, 1024)
        self.batch.setValue(32)
        form.addRow("Batch size:", self.batch)

        self.lr = QDoubleSpinBox(); self.lr.setDecimals(4)
        self.lr.setRange(0.0, 1.0)    # Sets the valid range from 0.0 to 1.0
        self.lr.setSingleStep(0.0001) # Optional: allows fine adjustments
        self.lr.setValue(0.001)
        form.addRow("Learning rate:", self.lr)
        
        # Model Configuration
        model_row = QHBoxLayout()
        # Model type selection
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["pihal", "tft"])
        model_row.addWidget(QLabel("Model Type:"))
        model_row.addWidget(self.model_type_combo)
        
        # Coverage Evaluation checkbox
        self.coverage_checkbox = QCheckBox("Evaluate Coverage")
        self.coverage_checkbox.setChecked(True)  # Default is checked
        model_row.addWidget(self.coverage_checkbox)
        
        form.addRow(model_row)  # Add to the form layout
        
        # Attention Levels
        self.attention_levels_input = QLineEdit()
        self.attention_levels_input.setText("1, 2, 3")
        form.addRow("Attention Levels (comma-separated):",
                    self.attention_levels_input)

        lay.addLayout(form)
        return card

    def _physics_card(self) -> QFrame:
        """Creates and returns the 'Physical Parameters' UI panel.

        This private factory method constructs the QFrame containing all
        widgets that allow the user to control the physics-informed
        components of the PINN models. It is the primary interface for
        injecting domain knowledge or enabling inverse modeling.

        The created panel includes widgets for:
        - Setting the PDE mode ('both', 'consolidation', etc.).
        - Defining the behavior of physical coefficients (C, K, Ss, Q)
          as either 'learnable' or a fixed value.
        - Adjusting the lambda weights for the consolidation and
          groundwater flow physics loss terms.
        - Setting the relative data loss weights between the subsidence
          and GWL prediction targets.

        Returns
        -------
        PyQt5.QtWidgets.QFrame
            A fully populated QFrame widget containing all the necessary
            UI elements for configuring the model's physics.

        See Also
        --------
        fusionlab.params : The module containing the `Learnable` classes.
        TransFlowSubsNet.compile : Where the lambda weights are used.
        """
        card = QFrame(); card.setObjectName("card")
        lay  = QVBoxLayout(card)
        lay.addWidget(self._title("Physical Parameters"))
        lay.addWidget(hline())

        form = QFormLayout()
        self.pinn_coeff_c_input = QComboBox()
        self.pinn_coeff_c_input.addItems(["learnable", "fixed"])
        form.addRow("Pinn Coeff C:", self.pinn_coeff_c_input)
    
        self.lambda_cons = QDoubleSpinBox()
        self.lambda_cons.setRange(0, 10)
        self.lambda_cons.setValue(1)
        form.addRow("λ Consolidation:", self.lambda_cons)

        self.lambda_gw = QDoubleSpinBox()
        self.lambda_gw.setRange(0, 10)
        self.lambda_gw.setValue(1)
        form.addRow("λ GW Flow:", self.lambda_gw)

        self.pde_mode = QComboBox()
        self.pde_mode.addItems(["both", "consolidation", "gw_flow", "none"])
        form.addRow("PDE mode:", self.pde_mode)
        
        # Physical Parameters
        flow_params_row = QHBoxLayout()
        
        # K (learnable/fixed)
        self.gwflow_k_type_combo = QComboBox()
        self.gwflow_k_type_combo.addItems(["learnable", "fixed"])
        flow_params_row.addWidget(QLabel("K:"))
        flow_params_row.addWidget(self.gwflow_k_type_combo)
        
        # Ss (learnable/fixed)
        self.gwflow_ss_type_combo = QComboBox()
        self.gwflow_ss_type_combo.addItems(["learnable", "fixed"])
        flow_params_row.addWidget(QLabel("Ss:"))
        flow_params_row.addWidget(self.gwflow_ss_type_combo)
        
        # Q (learnable/fixed)
        self.gwflow_q_type_combo = QComboBox()
        self.gwflow_q_type_combo.addItems(["learnable", "fixed"])
        flow_params_row.addWidget(QLabel("Q:"))
        flow_params_row.addWidget(self.gwflow_q_type_combo)
        
        form.addRow(flow_params_row)  # Add to the form layou

        # Weights Row (Subsidence and GWL)
        weights_row = QHBoxLayout()
        # Weight for Subsidence
        self.weight_subs_spin = QDoubleSpinBox()
        self.weight_subs_spin.setRange(0.0, 10.0)
        self.weight_subs_spin.setValue(1.0)
        weights_row.addWidget(self.weight_subs_spin)
    
        # Weight for GWL
        self.weight_gwl_spin = QDoubleSpinBox()
        self.weight_gwl_spin.setRange(0.0, 10.0)
        self.weight_gwl_spin.setValue(0.5)
        weights_row.addWidget(self.weight_gwl_spin)
    
        # Add the combined weights row
        form.addRow("Weights (Subs. / GWL):", weights_row)

        lay.addLayout(form)
        return card

    def _feature_card(self) -> QFrame:
        """Creates and returns the 'Feature Selection' UI panel.

        This private factory method constructs the QFrame that allows the
        user to map columns from their input dataset to the distinct
        feature roles required by the underlying forecasting models.

        The panel provides three QLineEdit widgets for the user to input
        comma-separated lists of column names for:
        1. Dynamic past features.
        2. Static (time-invariant) features.
        3. Known future features.

        Each field can also be set to 'auto' to trigger the automatic
        feature detection logic in the backend `SubsConfig` class.

        Returns
        -------
        PyQt5.QtWidgets.QFrame
            A fully populated QFrame widget for defining feature roles.

        See Also
        --------
        SubsConfig.auto_detect_columns : The backend logic for 'auto' mode.
        """
        card = QFrame(); card.setObjectName("card")
        lay  = QVBoxLayout(card)
        lay.addWidget(self._title("Feature Selection"))
        lay.addWidget(hline())
    
        form = QFormLayout()
        # three inputs side-by-side
        row = QHBoxLayout()
    
        self.dyn_feat  = QLineEdit("auto") 
        row.addWidget(self.dyn_feat, 1)
        
        self.stat_feat = QLineEdit("auto")        
        row.addWidget(self.stat_feat, 1)
        self.fut_feat  = QLineEdit("rainfall_mm")
        row.addWidget(self.fut_feat, 1)
    
        row.setSpacing(4)
        form.addRow("Dyn. / Stat. / Future:", row)
        lay.addLayout(form)
        return card

    def _title(self, txt): 
        l = QLabel(txt); l.setObjectName("cardTitle"); return l
        
    def _log(self, msg): 
        """
        Print one line in the QTextEdit **and** keep a copy in memory
        so we can dump everything to disk when the worker stops.
        """
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_cache.append(line)                         # (2) cache
        self.log.append(line)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum()) 
        
    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV file", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            # The CsvEditDialog constructor now tries to read the file,
            # which might fail.
            dlg = CsvEditDialog(str(path), self)
            if dlg.exec_() == QDialog.Accepted:
                self.edited_df = dlg.edited_dataframe()
                self._log(
                    f"CSV preview accepted – {len(self.edited_df)} rows retained."
                )
                self.file_path = Path(path) # Set file_path only on success
                self.file_label.setStyleSheet(f"color:{SECONDARY};")
                self.file_label.setText(f"Selected: {self.file_path.name}")
                self._log(f"CSV chosen → {self.file_path}")
                self.tune_btn.setEnabled(True)
                self.tune_btn.setStyleSheet(f"background:{PRIMARY};")

            else:
                self.edited_df = None
                self._log("CSV preview canceled – keeping original file.")

        except Exception as e:
            # If CsvEditDialog fails to load the file, catch the error
            # and use our new central dialog to show it.
            self._show_error_dialog(
                "File Load Error",
                f"Could not read or process the selected CSV file.\n\n"
                f"Details: {e}"
            )
            return

        if not self.city_input.text().strip():
            self.city_input.setText(self.file_path.stem)
            
        self.progress_bar.setValue(0)
        self._refresh_manifest_state()
        

    def _refresh_manifest_state(self) -> None:
        """
        Sync GUI buttons with the registry status.
    
        Called
          • at start-up
          • after a training / tuning run finishes
          • after the user picks a CSV file
        """
        # ---------- 1.  regular training manifest  ---------------------------
        manifest, tuner_manifest = _locate_manifest(locate_both=True)            
        if manifest:
            self._manifest_path = str(manifest)
            self.inf_btn.setEnabled(True)
            self.inf_btn.setStyleSheet(f"background:{PRIMARY};")
            self.inf_btn.setToolTip("Click to switch to inference mode")
        else:
            self._manifest_path = None
            self.inf_btn.setEnabled(False)
            self.inf_btn.setStyleSheet(f"background:{INFERENCE_OFF};")
            self.inf_btn.setToolTip(
                "Inference becomes available after you train a model"
            )
        # The Tune button is enabled if the user has a CSV **or** a tuner manifest
        tune_available = (tuner_manifest is not None) or (
            self.file_path is not None)
    
        if tune_available:
            self.tune_btn.setEnabled(True)
            # self.tune_btn.setStyleSheet(f"background:{PRIMARY};")
            if tuner_manifest:
                idle_tt =( 
                    "Tuner results found – "
                    "you can retune or inspect them"
                    )
            else:
                idle_tt = (
                    "CSV loaded – click to configure"
                    " hyper-parameter tuning")
        else:
            idle_tt = "Select a CSV first to enable tuning"
            self.tune_btn.setStyleSheet(f"background:{INFERENCE_OFF};")
            
        # make sure the helper has the correct idle tooltip
        self.tune_mode._tt_idle = idle_tt
        self.tune_btn.setToolTip(idle_tt)

    def _pick_manifest_for_inference(self) -> str | None:
        """
        Called right before launching inference.
        If a sibling *tuner_run_manifest.json* exists next to the regular
        *run_manifest.json*, ask the user which one to use.
    
        Returns absolute path to chosen manifest, or None if user aborted.
        """
        if self._manifest_path is None:
            return None
    
        run_dir = Path(self._manifest_path).parent
        tuner_manifest = run_dir / "tuner_run_manifest.json"
        if not tuner_manifest.exists():
            # nothing special – just use the normal manifest
            return self._manifest_path
    
        dlg = ModelChoiceDialog(theme=self.theme, parent=self)
        choice = dlg.choice()
        if choice is None:          # user cancelled
            return None
        return str(tuner_manifest if choice == "tuned" else self._manifest_path)


    def _toggle_inference_mode(self):
        """Flips the GUI between *training* and *inference* modes.

        This method acts as a toggle switch. It updates the internal
        state, provides visual feedback via a toast notification and
        by changing button/panel styles, and updates tooltips to guide
        the user.
        """
        self._inference_mode = not self._inference_mode

        # 1. Show the toast notification for clear feedback
        message = (
            "Inference Mode Activated"
            if self._inference_mode
            else "Training Mode Activated"
        )
        toast = ToastNotification(message, self, theme=self.theme)
        toast.show_toast()

        # 2. Update button styles and tooltips based on the new mode
        if self._inference_mode:
            # --- Inference Mode is ON ---
            self._set_panels_disabled(True)
            # Style the button to look "active" (e.g., orange)
            self.inf_btn.setStyleSheet(
                f"background-color: {SECONDARY}; color: white;")
            self.inf_btn.setToolTip(
                "Inference mode is active. Click again to switch back to Training"
            )
            self.run_btn.setText("RunI")
            run_tooltip = (
                "Launch the inference pipeline using the detected model "
                "and the selected CSV file"
            )
        else:
            # --- Inference Mode is OFF (back to training) ---
            # Revert the button to its normal "enabled" style
            self._set_panels_disabled(False)
            self.inf_btn.setStyleSheet(
                f"background-color: {PRIMARY}; color: white;")
            self.inf_btn.setToolTip(
                "Click to switch to Inference mode - requires a trained model"
            )
            self.run_btn.setText("Run")
            run_tooltip = "Launch the full training and forecasting pipeline"
        
        self.run_btn.setToolTip(run_tooltip)
    
        # 3. Update card borders using the dynamic property
        cards_to_style = [
            self.model_card, self.training_card,
            self.physics_card, self.feature_card
        ]
        for card in cards_to_style:
            card.setProperty("inferenceMode", self._inference_mode)
            card.style().unpolish(card)
            card.style().polish(card)
            
    def _set_panels_disabled(self, disable: bool):
        """Disables or enables the panels and inputs based on inference mode."""
        panels = [
            self.model_card, self.training_card, 
            self.physics_card, self.feature_card
        ]
        
        for panel in panels:
            panel.setEnabled(not disable)  # Disable the panel's interaction
            
            # Change color to gray to indicate it's disabled
            if disable:
                panel.setStyleSheet("background-color: #cccccc;")  # Light gray background
            else:
                panel.setStyleSheet("")  # Reset to normal style
                
    def _on_reset(self):
        """Resets the user interface to its default state.

        This method is the slot connected to the `Reset` button's
        `clicked` signal. It provides a convenient way for the user
        to clear all current inputs and results and start a new
        configuration from scratch.

        Specifically, it performs the following actions:
        - Clears the main log panel.
        - Resets the progress bar to zero.
        - Clears the file path selection and updates the label.
        - Resets all configuration widgets in the different panels
          (e.g., `Model Configuration`, `Training Parameters`) to
          their initial default values.
        - Clears the coverage score label in the footer.
        """
        # reset feature inputs
        self.dyn_feat.setText("auto")
        self.stat_feat.setText("auto")
        self.fut_feat.setText("rainfall_mm")
    
        # clear log + status + progress
        self.log.clear()
        self.file_label.setText("No file selected" if self.file_path is None
                                else f"Selected: {self.file_path.name}")
        self.progress_bar.setValue(0)
        self.city_input.clear()
        self.coverage_lbl.clear() 
        
        # If the GUI is in inference mode when reset is clicked,
        # toggle it back to training mode to ensure a clean state.
        if self._inference_mode:
            self._toggle_inference_mode()
        
        self.run_btn.setStyleSheet("")
        self.stop_btn.setStyleSheet("") 
        self.tune_btn.setEnabled(False)
        self.tune_btn.setStyleSheet("")

        self._log("ℹ Interface reset.")
        
    def _stop_worker(self):
        """
        Prompts the user for confirmation and then gracefully stops
        the active background worker thread.
        """
        # First, check if there is actually a worker to stop.
        if not (hasattr(self, 'active_worker') and 
                self.active_worker and self.active_worker.isRunning()):
            self._refresh_stop_button() 
            return

        # Create and display a confirmation dialog.
        # The question method is modal and returns the button that was clicked.
        reply = QMessageBox.question(
            self,
            'Confirm Stop',
            "Are you sure you want to stop the current workflow?\n\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No  # Default button is 'No'
        )

        # Only proceed if the user confirms.
        if reply == QMessageBox.Yes:
            self._log("⏹️ Workflow stop requested by user.")
            self.status_updated.emit("⏹️ Stopping workflow…")
            
            # Request the interruption. The thread will stop when it
            # next checks the isInterruptionRequested() flag.
            self.active_worker.requestInterruption()
            
            # Immediately provide visual feedback by resetting the UI state.
            self.stop_btn.setEnabled(False)
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run")
            self.run_btn.setStyleSheet("") # Revert to default style
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Stopped")
        
            self.active_worker.requestInterruption()
            self.progress_bar.setValue(0)
            self._refresh_stop_button()

    def _refresh_stop_button(self):
        busy = getattr(self, "active_worker", None) and \
               self.active_worker.isRunning()
        if busy:
            self.stop_btn.setEnabled(True)
            self.stop_btn.setToolTip("Abort the running workflow")
        else:
            self.stop_btn.setEnabled(False)
            self.stop_btn.setToolTip("Nothing to stop – no workflow running")
        
        self.progress_manager.reset()

    def _on_run(self):
        """
        Acts as the main dispatcher for the 'Run' button.

        Checks if the GUI is in training or inference mode and calls the
        appropriate workflow launcher.
    
        """
    
        # First, ensure a file has been loaded and prepared.
        if self.file_path is None:
            self._log("⚠ Please select a CSV file first.")
            QMessageBox.warning(
                self, "No Data", "Please select a CSV data file"
                " before running a workflow.")
            return
            
        self.coverage_lbl.clear()
        
        # Dispatch to the correct workflow based on the UI mode
        if self._inference_mode:
            chosen_manifest = self._pick_manifest_for_inference()
            if chosen_manifest is None:        # user cancelled
                self._log("ℹ Inference cancelled by user.")
                return
            self._manifest_path = chosen_manifest
            self._run_inference()              # ← unchanged
        else:
            self._run_training() # A new helper for clarity
            
    def _run_training(self):
        """Initiates the end-to-end forecasting workflow.

        This method is the slot connected to the `Run` button's
        `clicked` signal. It serves as the primary action trigger
        for the entire application.

        The method orchestrates the following steps:
        1.  Performs a pre-flight check to ensure a data file has
            been selected by the user.
        2.  Gathers all current settings from the various UI input
            widgets (e.g., spin boxes, combo boxes, text fields).
        3.  Parses these settings and instantiates a `SubsConfig`
            object, creating a complete configuration for the run.
        4.  Disables the 'Run' button and enables the 'Stop' button
            to manage the UI state during processing.
        5.  Instantiates the `Worker` background thread, passing it
            the configuration object and the path to the user's data.
        6.  Connects the worker's signals (e.g., for logging,
            progress) to the appropriate UI update slots.
        7.  Starts the worker thread to begin the processing pipeline
            without freezing the GUI.
        """
        self.progress_bar.setValue(0)
        if self.file_path is None:
            self._log("⚠ No CSV selected.")
            return
     
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running…") # Optional: Update text
        # disabled-looking style.
        self.run_btn.setStyleSheet(f"background-color: {INFERENCE_OFF};")
        
        self._log("▶ launch TRAINING workflow …")
        QApplication.processEvents()
        
        def _parse(txt):
            txt = txt.strip()
            return "auto" if txt.lower() == "auto" else [
                t.strip() for t in txt.split(",") if t.strip()]
        
        dyn_list  = _parse(self.dyn_feat.text())
        stat_list = _parse(self.stat_feat.text())
        fut_list  = _parse(self.fut_feat.text())
        
        save_fmt = self.save_format_combo.currentText().lower()
        if save_fmt != "weights":
            self._log(f"⚠ You selected save_format='{save_fmt}'. "
                      "GUI mode is battle-tested only with 'weights'.")

        cfg = SubsConfig(
            city_name     = self.city_input.text().strip() or "unnamed",
            data_dir      = str(self.file_path.parent),
            data_filename = self.file_path.name,
            model_name    = self.model_select.currentText(),
            epochs        = self.epochs.value(),
            batch_size    = self.batch.value(),
            learning_rate = self.lr.value(),
            lambda_cons   = self.lambda_cons.value(),
            lambda_gw     = self.lambda_gw.value(),
            pde_mode      = self.pde_mode.currentText(),
     
            # New parameters for training and forecasting
            train_end_year         = self.train_end_year_spin.value(),
            forecast_start_year    = self.forecast_start_year_spin.value(),
            forecast_horizon_years = self.forecast_horizon_spin.value(),
            time_steps             = self.time_steps_spin.value(),
            quantiles             = [
                float(q) for q in self.quantiles_input.text().split(',')],
            
            # New physical parameters for Pinn Coeff C and weights
            pinn_coeff_c           = self.pinn_coeff_c_input.currentText(),
            weight_subs_pred       = self.weight_subs_spin.value(),
            weight_gwl_pred        = self.weight_gwl_spin.value(),
            
            # New parameters for flow initialization and coverage evaluation
            gwflow_init_k          = self.gwflow_k_type_combo.currentText(),  
            gwflow_init_ss         = self.gwflow_ss_type_combo.currentText(), 
            gwflow_init_q          = self.gwflow_q_type_combo.currentText(),
            
            # Coverage evaluation and model mode
            evaluate_coverage      = self.coverage_checkbox.isChecked(),
            mode                   = self.model_type_combo.currentText(),  
            
            save_format       = save_fmt,
            verbose       = 1,
    
        )
        # hand the Qt emitters to the config
        cfg.log               = self.log_updated.emit
        cfg.save_format       = "weights"      
        cfg.bypass_loading    = True       
        cfg.dynamic_features = dyn_list
        cfg.static_features  = stat_list
        cfg.future_features  = fut_list
        
        # Register the manifest.
        cfg.to_json()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # Visually activate the stop button
        self.stop_btn.setStyleSheet(
            f"background-color: {PRIMARY}; color: white;")

        # start worker 
        self.active_worker = TrainingThread(
            cfg, 
            progress_manager = self.progress_manager,
            edited_df=getattr(self, "edited_df", None), 
            parent=self, 
        )
        self.active_worker.log_updated.connect(self.log_updated.emit)
        self.active_worker.status_updated.connect(self.status_updated.emit)
        self.active_worker.coverage_ready.connect(self.coverage_ready.emit)

        self.active_worker.finished.connect(self._worker_done)
        
        self.active_worker.error_occurred.connect(self.error_mgr.report)
        self.worker_ctl.bind(self.active_worker)      
        self.active_worker.start()
        self._refresh_stop_button() 
        
    def _run_inference(self):
        """Launches the inference workflow using the pre-loaded data.

        This method assumes that a manifest has been found and that the
        user has already selected and potentially edited a CSV file via
        the `_choose_file` method.
        """

        # Pre-flight check: ensure data is actually loaded and ready.
        # This can be from the original file or the CsvEditDialog.
        inference_data = ( 
            self.edited_df if self.edited_df is not None 
            else pd.read_csv(self.file_path)
        )
        if inference_data is None or inference_data.empty:
            QMessageBox.warning(
                self, "Inference Error",
                "No valid data available to run inference on."
            )
            return

        self._log("▶ Launching INFERENCE workflow…")
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Inferring…") 
        self.run_btn.setStyleSheet(f"background-color: {INFERENCE_OFF};")
        
        self.stop_btn.setEnabled(True)
        self.stop_btn.setStyleSheet(
            f"background-color: {PRIMARY}; color: white;")
        self.progress_bar.setValue(0)
        self.progress_bar.repaint()

        # --- Launch the InferenceThread ---
        # It receives the manifest path found by `_refresh_manifest_state`
        # and the in-memory DataFrame prepared by `_choose_file`.
        self.active_worker = InferenceThread(
            manifest_path=self._manifest_path,
            progress_manager = self.progress_manager,
            edited_df=inference_data,
            parent=self,
        )
        self.active_worker.log_msg.connect(self.log_updated.emit)
        self.active_worker.status_msg.connect(self.status_updated.emit)
        
        self.active_worker.finished.connect(self._worker_done)

        self.active_worker.error_occurred.connect(self.error_mgr.report)
        self.worker_ctl.bind(self.active_worker)     
        self.active_worker.start()
        self._refresh_stop_button() 
    
    def _open_tuner_dialog(self):
        if self.file_path is None:
            QMessageBox.warning(self, "No data",
                                "Please select a CSV file first.")
            return
    
        # infer minimal fixed params …
        fixed_params = {
            "static_input_dim": 0,
            "dynamic_input_dim": 1,
            "future_input_dim": 0,
            "output_subsidence_dim": 1,
            "output_gwl_dim": 1,
            "forecast_horizon": self.forecast_horizon_spin.value(),
        }
    
        dlg = TunerDialog(fixed_params, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
    
        cfg_dict = dlg.chosen_config()
        if cfg_dict is None:
            return  # should not happen
    
        # ---------- 2. build a *fresh* SubsConfig for the tuner ---------

        # MiniForecaster._open_tuner_dialog()
        max_trials = cfg_dict["tuner_settings"]["max_trials"]
        fixed_up = cfg_dict.get("fixed_params", {})
        seq_params = cfg_dict.get ('sequence_params', {})
        
        train_end_year = seq_params.get (
            "train_end_year", self.train_end_year_spin.value()
            )
        
        forecast_start_year = seq_params.get (
            "forecast_start_year",  self.forecast_start_year_spin.value(),
            )
        time_steps = fixed_up.pop(
            "max_window_size", self.time_steps_spin.value()) 
        forecast_horizon = fixed_up.pop(
            "forecast_horizon", self.forecast_horizon_spin.value()
        )
        
        self._tuning_max_trials = max_trials
        # self.trialLb.setText("--/--")   # ← show placeholder from the very start

        tune_cfg = SubsConfig(
            city_name = self.city_input.text() or "unnamed",
            model_name = self.model_select.currentText(),
            data_dir   = str(self.file_path.parent),
            data_filename = self.file_path.name,
            forecast_horizon_years = forecast_horizon,
            time_steps = time_steps,
            
            train_end_year         = train_end_year,
            forecast_start_year    = forecast_start_year,

            save_format = "keras",
            bypass_loading = True,          # no need while tuning
            verbose = 1, # for minimal logging
            **fixed_up
            
        )
        tune_cfg.log  = self.log_updated.emit
        
        # tune_cfg.progress_callback = self.progress_updated.emit
    
        # ---------- 3. spawn the thread --------------------------------
        self._log("▶ launch TUNER workflow …")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tune_btn.setEnabled(False)
    
        self.active_worker = TunerThread(
            cfg           = tune_cfg,
            search_space  = cfg_dict["search_space"],
            tuner_kwargs  = cfg_dict["tuner_settings"],
            progress_manager = self.progress_manager,
            edited_df = getattr(self, "edited_df", None),
            parent = self,
        )
        w = self.active_worker
        w.log_updated.connect(self.log_updated.emit)
        w.status_updated.connect(self.status_updated.emit)
        w.tuning_finished.connect(self._worker_done)
        
        w.error_occurred.connect(self.error_mgr.report)
        self.worker_ctl.bind(self.active_worker)   
        
        # **bind** ModeSwitch so the button stays orange while running
        self.tune_mode.bind(w)
        # Inference must be greyed while tuning runs
        self.inf_btn.setEnabled(False)
    
        w.start()
        self._refresh_stop_button() 
        
    def _worker_done(self):
        """
        Called from `self.worker.finished`.  
        Saves the whole cached log to
        <run-output-path>/gui_log_YYYYMMDD_HHMMSS.txt
        and re-enables the buttons.
        """
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run")  
        self.stop_btn.setEnabled(False)
        
        # Use the generic active_worker attribute
        if ( 
                hasattr(self.active_worker, 'cfg') 
                and self.active_worker is not None
            ):
        
            # decide where to write
            out_dir = getattr(self.active_worker.cfg, "run_output_path", ".")
            ts      = time.strftime("%Y%m%d_%H%M%S")
            fname   = os.path.join(out_dir, f"gui_log_{ts}.txt")
    
            try:                                                 # limit size
                MAX_LINES = 10_000          # ≈ a few MB; change at will
                lines     = self._log_cache[-MAX_LINES:]
                with open(fname, "w", encoding="utf-8") as fp:
                    fp.write("\n".join(lines))
                self._log(f"📝 Log saved to: {fname}")
            except Exception as err:
                self._log(f"⚠ Could not write log file ({err})")
        
                
            
            # Clean up the active worker reference
            self.active_worker = None
      
        if self._inference_mode:
            self._toggle_inference_mode()
        
        self.status_updated.emit("⚪ Idle")
        
        self.run_btn.setStyleSheet("")
        self.stop_btn.setStyleSheet("") 
        self.tune_btn.setEnabled(True) 
        self._tuning_max_trials = None

        self.progress_manager.reset() 
        self.worker_ctl.bind(None)  
        self.tune_mode.bind(None) 
        self._refresh_manifest_state()  
    

def hline() -> QFrame:
    """Creates and returns a styled horizontal separator line.

    This is a simple helper function to generate a QFrame configured
    as a horizontal line, useful for visually separating sections
    in a GUI layout.

    Returns
    -------
    PyQt5.QtWidgets.QFrame
        A QFrame widget styled as a horizontal line.
    """
    ln = QFrame()
    ln.setFrameShape(QFrame.HLine)
    ln.setStyleSheet(f"color:{PRIMARY}")
    return ln          

def launch_cli(theme: str = 'fusionlab') -> None:
    """Initializes and launches the main GUI application.

    This function serves as the primary entry point for the desktop
    tool. It is responsible for setting up the PyQt5 application
    environment, applying the selected visual theme, creating the
    main window instance, and starting the Qt event loop.

    Parameters
    ----------
    theme : {'fusionlab', 'light', 'dark', 'geoscience', 'native'}, default='fusionlab'
        The visual theme to apply to the application.
        - 'fusionlab' or 'light': Applies the default clean, light theme.
        - 'dark': Applies a modern dark theme for reduced eye strain.
        - 'geoscience': Applies a custom theme with an earth-toned
          palette loaded from an external `style.qss` file.
        - 'native': Uses the default, unstyled look of the operating
          system without applying any custom stylesheets.

    Notes
    -----
    The function prioritizes loading a `style.qss` file from the
    current directory if ``theme='geoscience'`` is selected. If this
    file is not found, it issues a warning and falls back to the
    default 'fusionlab' theme.
    """
    app = QApplication(sys.argv)
    QToolTip.setFont(QFont("Helvetica Neue", 9))

    # Normalize theme aliases
    if theme.lower() in ('light', 'fusionlab', 'fusionlab-learn'):
        theme = 'fusionlab'

    # --- Theme and Stylesheet Selection ---
    # Define available built-in themes
    theme_stylesheets = {
        "fusionlab": FLAB_STYLE_SHEET,
        "dark": DARK_THEME_STYLESHEET,
    }

    selected_stylesheet = ""
    # Special handling for the external 'geoscience' theme
    s_qss = os.path.join(os.path.dirname(__file__), "style.qss")
    if theme.lower() in ('geoscience', 'geo'):
        if os.path.exists(s_qss):
            with open(s_qss, "r", encoding="utf-8") as f:
                selected_stylesheet = f.read()
        else:
            # Provide a clear warning and fall back to the default theme
            warnings.warn(
                "Theme 'geoscience' was selected, but 'style.qss' was not "
                "found in the current directory. Falling back to the "
                "default 'fusionlab' theme.",
                UserWarning,
                stacklevel=2
            )
            theme = 'fusionlab' 
    
    # Select from built-in themes if no external file was loaded
    if not selected_stylesheet and theme.lower() != 'native':
        selected_stylesheet = theme_stylesheets.get(theme.lower())

    # Apply the stylesheet only if one was selected
    if selected_stylesheet:
        # Add the dynamic property rule for the inference mode border
        inference_border_style = f"""
            QFrame#card[inferenceMode="true"] {{
                border: 2px solid {PRIMARY};
            }}
        """
        final_stylesheet = ( 
            selected_stylesheet + inference_border_style + LOG_STYLES 
        )
        app.setStyleSheet(final_stylesheet)
    
    # --- Instantiate and Run the Application ---
    gui = MiniForecaster(theme=theme)
    gui.show()
    sys.exit(app.exec_())
    
# if __name__ == "__main__":
#     launch_cli()
# mini_forecaster_gui.py  – very bottom
if __name__ == "__main__":
    import traceback, faulthandler
    faulthandler.enable()                       # catches segfault-like crashes

    def qt_excepthook(exctype, value, tb):
        """Let uncaught exceptions bubble to the console *and* keep Qt alive."""
        traceback.print_exception(exctype, value, tb)
        # comment the next line if you prefer the GUI to stay open
        QApplication.quit()

    sys.excepthook = qt_excepthook              # <<< install *before* exec_()

    app = QApplication(sys.argv)
    # … theme / stylesheet logic …
    gui = MiniForecaster(theme="fusionlab")
    gui.show()

    sys.exit(app.exec_())
