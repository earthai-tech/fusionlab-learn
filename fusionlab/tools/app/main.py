"""
Main entry point for the Subsidence Forecasting GUI application.

This script launches a PyQt5-based desktop application that provides a
user-friendly interface for the complex `fusionlab-learn` forecasting
workflow. It integrates data loading, preprocessing, model training,
and visualization into a single, interactive tool.
"""

# 1. Core System & Library Imports
import sys
import os
import time
# import shutil
# import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QFileDialog,
    QTextEdit, QFrame, QGridLayout, QFormLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox
)
from PyQt5.QtGui import  QIcon, QPixmap#, QFont,
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 2. Application-Specific & FusionLab Imports
# This block handles the crucial import of the application's backend logic.
# It checks for both the `fusionlab` library and the necessary backend
# (TensorFlow) before proceeding.
try:
    # First, import the environment setup utility
    from fusionlab.tools.app._config import setup_environment

    # Run the dependency check immediately. This will raise an ImportError
    # with a helpful message if TensorFlow is not installed, preventing
    # the rest of the application from attempting to load.
    setup_environment()

    # If the setup succeeds, proceed to import the processing classes
    from fusionlab.tools.app._processing import (
        SubsConfig, DataProcessor, SequenceGenerator,
        ModelTrainer, Forecaster, ResultsVisualizer
    )
except (ImportError, ModuleNotFoundError) as e:
    # This catch block handles errors if the script can't find its own
    # modules or if the setup_environment() raises an ImportError.
    print("CRITICAL ERROR: Could not start the application.")
    print("Please ensure 'fusionlab-learn' and its dependencies (like "
          "TensorFlow) are correctly installed.")
    print(f"Details: {e}")
    # A simple GUI-less exit for a critical failure
    sys.exit(1)

# 3. GUI Styling and Worker Thread Definition

# --- Style Sheet for a modern dark theme ---
APP_STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #1e293b; /* slate-800 */
        color: #cbd5e1; /* slate-300 */
        font-family: sans-serif;
    }
    QLabel {
        font-size: 14px;
        color: #cbd5e1; /* slate-300 */
    }
    QLabel#title {
        font-size: 28px;
        font-weight: bold;
        color: #ffffff;
    }
    QLabel#description {
        font-size: 16px;
        color: #94a3b8; /* slate-400 */
    }
    QPushButton {
        background-color: #2563eb; /* blue-600 */
        color: white;
        border: none;
        padding: 10px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #1d4ed8; /* blue-700 */
    }
    QPushButton:disabled {
        background-color: #475569; /* slate-600 */
        color: #94a3b8; /* slate-400 */
    }
    QPushButton#resetButton {
        background-color: transparent;
        border: 1px solid #475569; /* slate-600 */
    }
    QPushButton#resetButton:hover {
        background-color: #334155; /* slate-700 */
    }
    QLineEdit, QSpinBox, QComboBox {
        background-color: #0f172a; /* slate-900 */
        border: 1px solid #475569; /* slate-600 */
        padding: 8px;
        border-radius: 6px;
        font-size: 14px;
        color: #ffffff;
    }
    QTextEdit {
        background-color: #020617; /* black */
        color: #e2e8f0; /* slate-200 */
        border: 1px solid #334155; /* slate-700 */
        border-radius: 6px;
        font-family: "Courier New", monospace;
        font-size: 12px;
    }
    QFrame#card {
        background-color: #334155; /* slate-700 */
        border: 1px solid #475569; /* slate-600 */
        border-radius: 8px;
    }
    QFrame#hLine {
        border: none;
        border-top: 1px solid #475569;
    }
    QLabel#cardTitle {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
    }
    QLabel#cardDescription {
        font-size: 13px;
        color: #94a3b8;
    }
"""

# --- Worker Thread to Run the Backend Processing ---
class ProcessingWorker(QThread):
    """
    Runs the entire forecasting workflow in a separate thread to keep
    the GUI responsive. It orchestrates the calls to the different
    processing classes (DataProcessor, ModelTrainer, etc.).
    """
    log_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    progress_updated  = pyqtSignal(int) 
    processing_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: SubsConfig, file_path: str):
        super().__init__()
        self.config = config
        self.file_path = file_path

    def run(self):
        try:
            # Link the worker's log signal to the config object so that all
            # backend classes can emit logs directly to the GUI.
            self.config.log = self.log_updated.emit
            self.config.progress_callback = self.progress_updated.emit
            
            # --- Execute the entire workflow step-by-step ---
            self.status_updated.emit("Starting Data Processing...")
            data_processor = DataProcessor(
                self.config, self.config.log
                )
            # Override data path with the file selected by the user in the GUI
            processed_df = data_processor.run()

            self.status_updated.emit("Generating Sequences for Training...")
            sequence_gen = SequenceGenerator(
                self.config, self.config.log )
            (
                train_dataset, val_dataset
            ) = sequence_gen.run(
                processed_df, data_processor.static_features_encoded
            )

            self.status_updated.emit("Training Model...")
            # We need the input shapes to build the model for the first time
            sample_inputs, _ = next(iter(train_dataset))
            input_shapes = {
                name: tensor.shape for name, tensor in sample_inputs.items()
            }

            trainer = ModelTrainer(self.config, self.config.log)
            best_model = trainer.run(train_dataset, val_dataset, input_shapes)

            self.status_updated.emit("Generating Forecasts...")
            forecaster = Forecaster(self.config, self.config.log)
            forecast_df = forecaster.run(
                model=best_model,
                test_df=sequence_gen.test_df,
                val_dataset=val_dataset,
                static_features_encoded=data_processor.static_features_encoded,
                coord_scaler=sequence_gen.coord_scaler
            )

            self.status_updated.emit("Visualizing Results...")
            visualizer = ResultsVisualizer(
                self.config, self.config.log )
            visualizer.run(forecast_df)

            self.status_updated.emit("Workflow Finished Successfully!")
            self.processing_finished.emit({
                "dataframe": forecast_df.to_string(
                    ) if forecast_df is not None else "No forecast data.",
                "plots": [
                    os.path.join(
                        self.config.run_output_path,
                        (
                            f"{self.config.city_name}_{self.config.model_name}"
                            "_training_history_plot_.png"
                        )
                    ),
                    # Add paths to other plots as they are saved
                ]
            })

        except Exception as e:
            # If any part of the workflow fails, emit an error signal
            self.error_occurred.emit(
                f"An error occurred during processing: {e}\n\n"
                "Please check the log for more details."
            )

# --- Main Application Window ---
class SubsidenceForecasterApp(QMainWindow):
    """
    The main application window for the Subsidence Forecasting Tool.

    This class sets up the entire graphical user interface, including
    the configuration panels, logging areas, and results display. It
    connects UI elements to the backend processing classes and manages
    the application's state.
    """
    def __init__(self):
        """Initializes the application window and its components."""
        super().__init__()
        self.setWindowTitle("Subsidence Forecasting Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Set the application icon
        script_dir = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(script_dir, "fusionlab_learn_logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Initialize the master configuration object, linking the GUI's
        # log function as the callback for the backend.
        self.config = SubsConfig(log_callback=self.add_log)
        self.file_path = None

        # Build the user interface
        self.initUI()
        # Populate the UI with initial default values from the config
        self.update_ui_from_config()
        # Connect signals from UI widgets to handler methods
        self._connect_signals()
        
    def initUI(self):
        """
        Constructs and assembles all UI components for the main window.
        """
        # --- Create all UI components using helper methods ---
        header_widget = self._create_header()
        self.data_input_card = self._create_data_input_card()
        # self.config_card = self._create_config_card()
        self.workflow_config_card = self._create_workflow_config_card()
        
        self.data_config_card = self._create_data_config_card()
        self.model_config_card = self._create_model_config_card()
        self.results_panel = self._create_results_panel()
        
        # --- Assemble the layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        main_layout.addWidget(header_widget)
        
        content_layout = QHBoxLayout()
        
        # Left Panel (Configuration)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self.data_input_card)
        left_layout.addWidget(self.workflow_config_card)

        left_layout.addWidget(self.data_config_card)
        left_layout.addWidget(self.model_config_card)
        
        self.run_button = QPushButton("Run Training & Forecasting")
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("resetButton")
        
        left_layout.addWidget(self.run_button)
        left_layout.addWidget(self.reset_button)
        left_layout.addStretch() # Pushes content to the top
        
        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(self.results_panel, 2)
        main_layout.addLayout(content_layout)
        


    def _create_header(self):
        """Creates the header section with logo, title, and description."""
        header_widget = QWidget()
        layout = QVBoxLayout(header_widget)
        layout.setAlignment(Qt.AlignCenter)

        logo_label = QLabel()
        script_dir = os.path.dirname(os.path.realpath(__file__))
        logo_path = os.path.join(script_dir, "fusionlab_learn_log.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(
                120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)

        title_label = QLabel("Subsidence Forecasting Tool")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        desc_label = QLabel("A GUI for the `fusionlab-learn` PINN workflow.")
        desc_label.setObjectName("description")
        desc_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        return header_widget
    

    # Data-input card  ------------------------------------------------------
    def _create_data_input_card(self) -> QFrame:
        """Return a stylised card that lets the user pick a CSV file."""
        card   = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
    
        # Header
        title = QLabel("Data Input")
        title.setObjectName("cardTitle")
    
        desc  = QLabel("Upload your dataset in CSV format.")
        desc.setObjectName("cardDescription")
    
        # File picker widgets
        self.file_button = QPushButton("Select File…")
        self.file_button.clicked.connect(self._on_select_file)
    
        self.file_label  = QLabel("No file selected.")
        self.file_label.setStyleSheet("font-style: italic; color: #64748b;")
    
        # Assemble
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(self._create_hline())
        layout.addWidget(self.file_button)
        layout.addWidget(self.file_label)
    
        return card
    
    
    def _on_select_file(self) -> None:
        """Slot: open a file dialog and update the label."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.file_label.setStyleSheet("")     # reset italics / colour
            self.file_label.setText(file_path)
            self.selected_file_path = file_path
        else:
            self.file_label.setStyleSheet("font-style: italic; color: #64748b;")
            self.file_label.setText("No file selected.")
    
    
    # Workflow-config card  -----------------------------------------------
    def _create_workflow_config_card(self) -> QFrame:
        """Return a card with the main model & training parameters."""
        card   = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
    
        # Header
        title = QLabel("Workflow Configuration")
        title.setObjectName("cardTitle")
    
        layout.addWidget(title)
        layout.addWidget(self._create_hline())
    
        # Form body
        form = QFormLayout()
        form.setSpacing(10)
    
        # Model selector
        self.model_combo = QComboBox()
        self.model_combo.addItems(["TransFlowSubsNet", "PIHALNet"])
    
        # Year & horizon controls
        self.train_end_year_spin = QSpinBox()
        self.train_end_year_spin.setRange(1980, 2050)
    
        self.forecast_start_year_spin = QSpinBox()
        self.forecast_start_year_spin.setRange(1980, 2050)
    
        self.forecast_horizon_spin = QSpinBox()
        self.forecast_horizon_spin.setRange(1, 20)
    
        self.time_steps_spin = QSpinBox()
        self.time_steps_spin.setRange(1, 50)
    
        # Populate form
        form.addRow("Model:",                    self.model_combo)
        form.addRow("Train End Year:",           self.train_end_year_spin)
        form.addRow("Forecast Start Year:",      self.forecast_start_year_spin)
        form.addRow("Forecast Horizon (years):", self.forecast_horizon_spin)
        form.addRow("Time Steps (look-back):",   self.time_steps_spin)
    
        layout.addLayout(form)
        return card

    # ------------------------------------------------------------------
    # Data / feature-mapping card
    # ------------------------------------------------------------------
    def _create_data_config_card(self) -> QFrame:
        """Return a card that lets the user map CSV columns to roles."""
        card   = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
    
        title = QLabel("Data & Feature Configuration")
        title.setObjectName("cardTitle")
    
        layout.addWidget(title)
        layout.addWidget(self._create_hline())
    
        form = QFormLayout()
        form.setSpacing(10)
    
        # Column inputs
        self.time_col_input   = QLineEdit()
        self.time_col_input.setPlaceholderText("e.g. year")
        self.lon_col_input    = QLineEdit()
        self.lon_col_input.setPlaceholderText("longitude")
        self.lat_col_input    = QLineEdit()
        self.lat_col_input.setPlaceholderText("latitude")
        self.subs_col_input   = QLineEdit()
        self.subs_col_input.setPlaceholderText("subsidence")
        self.gwl_col_input    = QLineEdit()
        self.gwl_col_input.setPlaceholderText("GWL")
    
        # Feature lists
        self.static_features_input  = QLineEdit()
        self.static_features_input.setPlaceholderText("feat_a, feat_b")
        self.dynamic_features_input = QLineEdit()
        self.dynamic_features_input.setPlaceholderText("dyn_a, dyn_b")
        self.future_features_input  = QLineEdit()
        self.future_features_input.setPlaceholderText("rainfall_mm")
    
        # Assemble form
        form.addRow("Time column:",                    self.time_col_input)
        form.addRow("Longitude column:",               self.lon_col_input)
        form.addRow("Latitude column:",                self.lat_col_input)
        form.addRow("Subsidence column:",              self.subs_col_input)
        form.addRow("GWL column:",                     self.gwl_col_input)
        form.addRow("Static features (comma-sep):",    self.static_features_input)
        form.addRow("Dynamic features (comma-sep):",   self.dynamic_features_input)
        form.addRow("Future features (comma-sep):",    self.future_features_input)
        
        self.save_intermediate_check = QCheckBox("Save intermediate artefacts")
        self.include_gwl_check       = QCheckBox("Include GWL in plots")
    
        form.addRow(self.save_intermediate_check)
        form.addRow(self.include_gwl_check)
    
        layout.addLayout(form)
        return card
    
    # Model / training-hyper-param card
    # 
    def _create_model_config_card(self) -> QFrame:
        """Return a card with advanced model & training settings."""
        card   = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
    
        title = QLabel("Model & Training Configuration")
        title.setObjectName("cardTitle")
    
        layout.addWidget(title)
        layout.addWidget(self._create_hline())
    
        form = QFormLayout()
        form.setSpacing(10)
    
        # PDE / physics settings
        self.pde_mode_combo = QComboBox()
        self.pde_mode_combo.addItems(["both", "consolidation", "gw_flow", "none"])
    
        self.lambda_cons_spin = QDoubleSpinBox()
        self.lambda_cons_spin.setRange(0.0, 10.0)
        self.lambda_cons_spin.setDecimals(2)
        self.lambda_cons_spin.setSingleStep(0.1)
    
        self.lambda_gw_spin = QDoubleSpinBox()
        self.lambda_gw_spin.setRange(0.0, 10.0)
        self.lambda_gw_spin.setDecimals(2)
        self.lambda_gw_spin.setSingleStep(0.1)
    
        # Optimisation hyper-params
        self.lr_input = QLineEdit(); self.lr_input.setPlaceholderText("0.001")
    
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
    
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(8, 2048)
        self.batch_size_spin.setSingleStep(8)
    
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
    
        self.quantiles_input = QLineEdit(); self.quantiles_input.setPlaceholderText("0.1, 0.5, 0.9")
    
        # Assemble form
        form.addRow("PDE mode:",              self.pde_mode_combo)
        form.addRow("λ Consolidation:",       self.lambda_cons_spin)
        form.addRow("λ GW Flow:",             self.lambda_gw_spin)
        form.addRow(self._create_hline())
        form.addRow("Learning rate:",         self.lr_input)
        form.addRow("Epochs:",                self.epochs_spin)
        form.addRow("Batch size:",            self.batch_size_spin)
        form.addRow("Early-stop patience:",   self.patience_spin)
        form.addRow("Quantiles (comma-sep):", self.quantiles_input)
        

        # --- extra widgets referenced by update_ui_from_config -----
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["pihal", "tft"])
    
        self.attention_levels_input = QLineEdit()
        self.validation_size_spin   = QDoubleSpinBox()
        self.validation_size_spin.setRange(0.05, 0.5); self.validation_size_spin.setDecimals(2)
    
        self.weight_subs_spin = QDoubleSpinBox(); self.weight_subs_spin.setRange(0.0, 10.0)
        self.weight_gwl_spin  = QDoubleSpinBox(); self.weight_gwl_spin.setRange(0.0, 10.0)
    
        self.evaluate_coverage_check = QCheckBox("Evaluate coverage metrics")
    
        # add them to the form *before* returning
        form.addRow("Mode:", self.mode_combo)
        form.addRow("Attention levels (comma-sep):", self.attention_levels_input)
        form.addRow("Validation split:", self.validation_size_spin)
        form.addRow("Weight subsidence:", self.weight_subs_spin)
        form.addRow("Weight GWL:", self.weight_gwl_spin)
        form.addRow(self.evaluate_coverage_check)
    
        layout.addLayout(form)
        return card
    
    def _create_config_card(self):
        """Creates the main configuration card with all user-adjustable parameters."""
        card   = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
    
        # -- header -------------------------------------------------------------
        title = QLabel("Workflow Configuration")
        title.setObjectName("cardTitle")
        desc  = QLabel("Adjust model, training, and physics parameters.")
        desc.setObjectName("cardDescription")
    
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(self._create_hline())
    
        # -- form ---------------------------------------------------------------
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
    
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["TransFlowSubsNet", "PIHALNet"])
        form_layout.addRow("Model:", self.model_combo)
        form_layout.addRow(self._create_hline())
    
        # Year / horizon controls ----------------------------------------------
        self.train_end_year_spin = QSpinBox()
        self.train_end_year_spin.setRange(1980, 2050)
    
        self.forecast_start_year_spin = QSpinBox()
        self.forecast_start_year_spin.setRange(1980, 2050)
    
        self.forecast_horizon_spin = QSpinBox()
        self.forecast_horizon_spin.setRange(1, 20)
    
        self.time_steps_spin = QSpinBox()
        self.time_steps_spin.setRange(1, 50)
    
        form_layout.addRow("Train End Year:", self.train_end_year_spin)
        form_layout.addRow("Forecast Start Year:", self.forecast_start_year_spin)
        form_layout.addRow("Forecast Horizon:",   self.forecast_horizon_spin)
        form_layout.addRow("Time Steps (Look-back):", self.time_steps_spin)
        form_layout.addRow(self._create_hline())
    
        # PDE mode & lambdas ----------------------------------------------------
        self.pde_mode_combo = QComboBox()
        self.pde_mode_combo.addItems(['both', 'consolidation', 'gw_flow', 'none'])
    
        self.lambda_cons_spin = QDoubleSpinBox()
        self.lambda_cons_spin.setRange(0.0, 10.0)
        self.lambda_cons_spin.setDecimals(2)
        self.lambda_cons_spin.setSingleStep(0.1)
    
        self.lambda_gw_spin = QDoubleSpinBox()
        self.lambda_gw_spin.setRange(0.0, 10.0)
        self.lambda_gw_spin.setDecimals(2)
        self.lambda_gw_spin.setSingleStep(0.1)
    
        form_layout.addRow("PDE Mode:", self.pde_mode_combo)
        form_layout.addRow("Lambda Consolidation:", self.lambda_cons_spin)
        form_layout.addRow("Lambda GW Flow:",        self.lambda_gw_spin)
        form_layout.addRow(self._create_hline())
    
        # Optimisation hyper-params --------------------------------------------
        self.lr_input = QLineEdit()
    
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
    
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(8, 2048)
        self.batch_size_spin.setSingleStep(8)
    
        form_layout.addRow("Learning Rate:", self.lr_input)
        form_layout.addRow("Epochs:",        self.epochs_spin)
        form_layout.addRow("Batch Size:",    self.batch_size_spin)
    
        # -----------------------------------------------------------------------
        layout.addLayout(form_layout)
        return card


    def _create_results_panel(self):
        """Creates the right-hand panel for logs and results."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)

        title = QLabel("Processing Log & Results")
        title.setObjectName("cardTitle")
        self.status_label = QLabel("Ready. Configure and upload a file to start.")
        self.status_label.setObjectName("cardDescription")
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.results_df_output = QTextEdit()
        self.results_df_output.setReadOnly(True)
        self.results_df_output.setVisible(False)
        self.results_plots_layout = QGridLayout()

        card_layout.addWidget(title)
        card_layout.addWidget(self.status_label)
        card_layout.addWidget(self._create_hline())
        card_layout.addWidget(QLabel("Logs:"))
        card_layout.addWidget(self.log_output, 1) # Stretch factor
        card_layout.addWidget(self.results_df_output, 1)
        card_layout.addLayout(self.results_plots_layout, 2)
        
        layout.addWidget(card)
        return panel

    def _create_hline(self):
        """Helper to create a horizontal separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("hLine")
        return line
    
    def update_ui_from_config(self):
        """Sets all UI widget values based on the current config object."""
        # Main Workflow Config
        self.model_combo.setCurrentText(self.config.model_name)
        self.train_end_year_spin.setValue(self.config.train_end_year)
        self.forecast_start_year_spin.setValue(self.config.forecast_start_year)
        self.forecast_horizon_spin.setValue(self.config.forecast_horizon_years)
        self.time_steps_spin.setValue(self.config.time_steps)
        self.pde_mode_combo.setCurrentText(self.config.pde_mode)
        self.lambda_cons_spin.setValue(self.config.lambda_cons)
        self.lambda_gw_spin.setValue(self.config.lambda_gw)
        self.lr_input.setText(str(self.config.learning_rate))
        self.epochs_spin.setValue(self.config.epochs)
        self.batch_size_spin.setValue(self.config.batch_size)
        
        # Data Config
        self.time_col_input.setText(self.config.time_col)
        self.lon_col_input.setText(self.config.lon_col)
        self.lat_col_input.setText(self.config.lat_col)
        self.subs_col_input.setText(self.config.subsidence_col)
        self.gwl_col_input.setText(self.config.gwl_col)
        self.static_features_input.setText(", ".join(self.config.static_features or []))
        self.dynamic_features_input.setText(", ".join(self.config.dynamic_features or []))
        self.future_features_input.setText(", ".join(self.config.future_features or []))
        self.save_intermediate_check.setChecked(self.config.save_intermediate)
        self.include_gwl_check.setChecked(self.config.include_gwl_in_df)
        
        # Model & Prediction Config
        self.quantiles_input.setText(", ".join(map(str, self.config.quantiles or [])))
        self.mode_combo.setCurrentText(self.config.mode)
        self.attention_levels_input.setText(", ".join(self.config.attention_levels))
        self.validation_size_spin.setValue(self.config.validation_size)
        self.patience_spin.setValue(self.config.patience)
        self.weight_subs_spin.setValue(self.config.weight_subs_pred)
        self.weight_gwl_spin.setValue(self.config.weight_gwl_pred)
        self.evaluate_coverage_check.setChecked(self.config.evaluate_coverage)

    def _connect_signals(self):
        """Connects all widget signals to their handler methods."""
        # --- Main action buttons ---
        self.run_button.clicked.connect(self.handle_run_workflow)
        self.reset_button.clicked.connect(self.handle_reset)
        self.file_button.clicked.connect(self.handle_file_change)

        # --- Workflow Config widgets ---
        self.model_combo.currentTextChanged.connect(self.handle_config_change)
        self.train_end_year_spin.valueChanged.connect(self.handle_config_change)
        self.forecast_start_year_spin.valueChanged.connect(self.handle_config_change)
        self.forecast_horizon_spin.valueChanged.connect(self.handle_config_change)
        self.time_steps_spin.valueChanged.connect(self.handle_config_change)
        self.pde_mode_combo.currentTextChanged.connect(self.handle_config_change)
        self.lambda_cons_spin.valueChanged.connect(self.handle_config_change)
        self.lambda_gw_spin.valueChanged.connect(self.handle_config_change)
        self.epochs_spin.valueChanged.connect(self.handle_config_change)
        self.lr_input.textChanged.connect(self.handle_config_change)
        self.batch_size_spin.valueChanged.connect(self.handle_config_change)

        # --- Data Config widgets ---
        self.time_col_input.textChanged.connect(self.handle_config_change)
        self.lon_col_input.textChanged.connect(self.handle_config_change)
        self.lat_col_input.textChanged.connect(self.handle_config_change)
        self.subs_col_input.textChanged.connect(self.handle_config_change)
        self.gwl_col_input.textChanged.connect(self.handle_config_change)
        self.static_features_input.textChanged.connect(self.handle_config_change)
        self.dynamic_features_input.textChanged.connect(self.handle_config_change)
        self.future_features_input.textChanged.connect(self.handle_config_change)
        self.save_intermediate_check.stateChanged.connect(self.handle_config_change)
        self.include_gwl_check.stateChanged.connect(self.handle_config_change)

        # --- Model Config widgets ---
        self.quantiles_input.textChanged.connect(self.handle_config_change)
        self.mode_combo.currentTextChanged.connect(self.handle_config_change)
        self.attention_levels_input.textChanged.connect(self.handle_config_change)
        self.validation_size_spin.valueChanged.connect(self.handle_config_change)
        self.patience_spin.valueChanged.connect(self.handle_config_change)
        self.weight_subs_spin.valueChanged.connect(self.handle_config_change)
        self.weight_gwl_spin.valueChanged.connect(self.handle_config_change)
        self.evaluate_coverage_check.stateChanged.connect(self.handle_config_change)

    def handle_config_change(self):
        """Updates the master config object whenever any UI widget changes."""
        try:
            # Helper to parse comma-separated strings into lists
            def _parse_list(text):
                return [item.strip() for item in text.split(',') if item.strip()]

            def _parse_float_list(text):
                return [float(item.strip()) for item in text.split(',') if item.strip()]

            gui_config = {
                # --- Workflow Config ---
                'model_name': self.model_combo.currentText(),
                'train_end_year': self.train_end_year_spin.value(),
                'forecast_start_year': self.forecast_start_year_spin.value(),
                'forecast_horizon_years': self.forecast_horizon_spin.value(),
                'time_steps': self.time_steps_spin.value(),
                'pde_mode': self.pde_mode_combo.currentText(),
                'lambda_cons': self.lambda_cons_spin.value(),
                'lambda_gw': self.lambda_gw_spin.value(),
                'learning_rate': float(self.lr_input.text()),
                'epochs': self.epochs_spin.value(),
                'batch_size': self.batch_size_spin.value(),
                
                # --- Data Config ---
                'time_col': self.time_col_input.text(),
                'lon_col': self.lon_col_input.text(),
                'lat_col': self.lat_col_input.text(),
                'subsidence_col': self.subs_col_input.text(),
                'gwl_col': self.gwl_col_input.text(),
                'static_features': _parse_list(self.static_features_input.text()),
                'dynamic_features': _parse_list(self.dynamic_features_input.text()),
                'future_features': _parse_list(self.future_features_input.text()),
                'save_intermediate': self.save_intermediate_check.isChecked(),
                'include_gwl_in_df': self.include_gwl_check.isChecked(),

                # --- Model Config ---
                'quantiles': _parse_float_list(self.quantiles_input.text()),
                'mode': self.mode_combo.currentText(),
                'attention_levels': _parse_list(self.attention_levels_input.text()),
                'validation_size': self.validation_size_spin.value(),
                'patience': self.patience_spin.value(),
                'weight_subs_pred': self.weight_subs_spin.value(),
                'weight_gwl_pred': self.weight_gwl_spin.value(),
                'evaluate_coverage': self.evaluate_coverage_check.isChecked(),
            }
            self.config.update_from_gui(gui_config)
            self.add_log("Configuration updated.")
        except (ValueError, TypeError) as e:
            # Handle cases where text input might be invalid (e.g., for learning rate)
            self.add_log(f"Warning: Invalid config value entered - {e}")

    def handle_run_workflow(self):
        """Handles the main 'Run' button click to start the backend worker."""
        if not self.file_path:
            self.add_log("Error: No data file uploaded. Please select a file.")
            return
            
        # Disable UI elements during processing
        self.run_button.setDisabled(True)
        self.run_button.setText("Processing...")
        
        # Clear previous results from the UI
        self.results_df_output.setVisible(False)
        self.results_df_output.clear()
        for i in reversed(range(self.results_plots_layout.count())): 
            widget_to_remove = self.results_plots_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        # Pass the current state of the config object to the worker
        self.worker = ProcessingWorker(self.config, self.file_path)
        self.worker.log_updated.connect(self.add_log)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.processing_finished.connect(self.on_processing_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def handle_file_change(self):
        """Opens a file dialog to select a CSV file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)"
        )
        if path:
            self.file_path = path
            self.config.data_filename = os.path.basename(path)
            self.config.data_dir = os.path.dirname(path)
            self.file_label.setText(f"Loaded: {os.path.basename(path)}")
            self.file_label.setStyleSheet("color: #4ade80;")
            self.add_log(f"Data file selected: \"{path}\"")

    def add_log(self, message: str):
        """Appends a timestamped message to the log window."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def on_processing_finished(self, results: dict):
        """Updates the UI when the workflow completes successfully."""
        self.run_button.setDisabled(False)
        self.run_button.setText("Run Training & Forecasting")
        
        # Display the results dataframe
        self.results_df_output.setText(results.get("dataframe", "No results dataframe found."))
        self.results_df_output.setVisible(True)

        # Display the plot images
        plot_paths = results.get("plots", [])
        for i, path in enumerate(plot_paths):
            if os.path.exists(path):
                pixmap = QPixmap(path)
                plot_label = QLabel()
                plot_label.setPixmap(pixmap.scaled(
                    400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                plot_label.setAlignment(Qt.AlignCenter)
                self.results_plots_layout.addWidget(plot_label, i // 2, i % 2)
            else:
                self.add_log(f"[Warning] Plot image not found at: {path}")

    def on_error(self, error_message: str):
        """Updates the UI when an error occurs in the worker thread."""
        self.run_button.setDisabled(False)
        self.run_button.setText("Run Training & Forecasting")
        self.status_label.setText("An error occurred during the workflow.")
        self.add_log("--- WORKFLOW FAILED ---")
        self.add_log(error_message)
        
    def handle_reset(self):
        """Resets the UI and configuration to their initial state."""
        self.file_path = None
        self.file_label.setText("No file selected.")
        self.file_label.setStyleSheet("font-style: italic; color: #64748b;")
        
        self.log_output.clear()
        self.results_df_output.clear()
        self.results_df_output.setVisible(False)
        
        for i in reversed(range(self.results_plots_layout.count())): 
            widget = self.results_plots_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                
        # Reset config object and update UI to match defaults
        self.config = SubsConfig(log_callback=self.add_log)
        self.update_ui_from_config()
        
        self.add_log("Interface has been reset to default configuration.")
        self.status_label.setText("Ready. Configure parameters and upload a file.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = SubsidenceForecasterApp()
    window.show()
    sys.exit(app.exec_())
