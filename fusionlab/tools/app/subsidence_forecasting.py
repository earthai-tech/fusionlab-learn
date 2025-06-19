
import os 
import sys
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QFileDialog,
    QTextEdit, QFrame, QGridLayout, QFormLayout, QSpinBox
)
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

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

# --- Worker thread for simulating the processing steps ---
class ForecastWorker(QThread):
    """
    Runs the simulated forecasting workflow in a separate thread to avoid
    freezing the GUI.
    """
    log_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    processing_finished = pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        steps = [
            {"msg": "Step 1: Validating configuration and data...", "delay": 0.5},
            {"msg": "Step 2: Loading and preprocessing data...", "delay": 1.5},
            {"msg": "Step 3: Generating PINN data sequences...", "delay": 2.0},
            {"msg": "Step 4: Creating TensorFlow datasets...", "delay": 1.0},
            {"msg": f"Step 5: Training {self.config['modelName']} model for {self.config['epochs']} epochs...", "delay": 5.0},
            {"msg": "Step 6: Generating predictions on test data...", "delay": 1.5},
            {"msg": "Step 7: Formatting results and generating visualizations...", "delay": 2.0},
        ]
        
        self.status_updated.emit("Starting Forecasting Workflow...")
        self.log_updated.emit("--- Starting Forecasting Workflow ---")
        
        for step in steps:
            self.status_updated.emit(step["msg"])
            self.log_updated.emit(step["msg"])
            time.sleep(step["delay"])

        self.log_updated.emit("--- Workflow Complete ---")
        self.status_updated.emit("Workflow finished successfully.")
        
        # Simulate final results
        dummy_csv_data = "sample_idx,forecast_step,coord_t,coord_x,subsidence_q50,subsidence_actual\n0,1,2023.0,113.5,-10.1,-10.2\n0,2,2024.0,113.5,-10.8,-11.0\n1,1,2023.0,113.6,-11.8,-11.9"
        result_plots = [
            "https://placehold.co/600x400/1e293b/ffffff?text=Training+History+Plot",
            "https://placehold.co/600x400/1e293b/ffffff?text=Forecast+Visualization"
        ]
        
        self.processing_finished.emit({
            "dataframe": dummy_csv_data,
            "plots": result_plots
        })

# --- Main Application Window ---
class SubsidenceForecasterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subsidence Forecasting Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.config = {
            'modelName': 'TransFlowSubsNet',
            'trainEndYear': 2022,
            'forecastStartYear': 2023,
            'forecastHorizonYears': 3,
            'timeSteps': 5,
            'epochs': 50,
        }
        self.file_path = None

        self.initUI()

    def initUI(self):
        # --- Main Widget and Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Header ---
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setAlignment(Qt.AlignCenter)
        title_label = QLabel("Subsidence Forecasting Tool")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        desc_label = QLabel("A GUI for the `fusionlab-learn` PINN workflow.")
        desc_label.setObjectName("description")
        desc_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        header_layout.addWidget(desc_label)
        main_layout.addWidget(header_widget)
        
        # --- Main Content Area (Left/Right Panels) ---
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # --- Left Panel: Configuration ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        
        # Data Input Card
        data_card = QFrame()
        data_card.setObjectName("card")
        data_layout = QVBoxLayout(data_card)
        data_title = QLabel("Data Input")
        data_title.setObjectName("cardTitle")
        data_desc = QLabel("Upload your dataset in CSV format.")
        data_desc.setObjectName("cardDescription")
        self.file_button = QPushButton("Select File...")
        self.file_button.clicked.connect(self.handle_file_change)
        self.file_label = QLabel("No file selected.")
        self.file_label.setStyleSheet("font-style: italic; color: #64748b;")
        data_layout.addWidget(data_title)
        data_layout.addWidget(data_desc)
        data_layout.addWidget(QFrame(frameShape=QFrame.HLine, objectName="hLine"))
        data_layout.addWidget(self.file_button)
        data_layout.addWidget(self.file_label)
        left_layout.addWidget(data_card)

        # Workflow Config Card
        config_card = QFrame()
        config_card.setObjectName("card")
        config_layout = QVBoxLayout(config_card)
        config_title = QLabel("Workflow Configuration")
        config_title.setObjectName("cardTitle")
        config_desc = QLabel("Adjust model and training parameters.")
        config_desc.setObjectName("cardDescription")
        config_layout.addWidget(config_title)
        config_layout.addWidget(config_desc)
        config_layout.addWidget(QFrame(frameShape=QFrame.HLine, objectName="hLine"))
        
        form_layout = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["TransFlowSubsNet", "PIHALNet"])
        self.train_end_year_spin = QSpinBox(minimum=1980, maximum=2050)
        self.time_steps_spin = QSpinBox(minimum=1, maximum=50)
        self.forecast_start_year_spin = QSpinBox(minimum=1980, maximum=2050)
        self.forecast_horizon_spin = QSpinBox(minimum=1, maximum=20)
        self.epochs_spin = QSpinBox(minimum=1, maximum=1000)
        
       # --- Corrected Signal Connection Logic ---
        
        # Set default values and connect signals
        widgets = self.get_config_widgets()
        # Connect QSpinBox widgets using valueChanged for immediate updates
        for w in [self.train_end_year_spin, self.time_steps_spin, 
                  self.forecast_start_year_spin, self.forecast_horizon_spin, 
                  self.epochs_spin]:
            w.setValue(self.config.get(w.objectName(), 0)) # Use objectName if set
            w.valueChanged.connect(self.handle_config_change)

        # Connect QComboBox widget using currentTextChanged
        self.model_combo.setCurrentText(self.config['modelName'])
        self.model_combo.currentTextChanged.connect(self.handle_config_change)
        
        form_layout.addRow("Model:", self.model_combo)
        form_layout.addRow("Train End Year:", self.train_end_year_spin)
        form_layout.addRow("Time Steps (Lookback):", self.time_steps_spin)
        form_layout.addRow("Forecast Start Year:", self.forecast_start_year_spin)
        form_layout.addRow("Forecast Horizon:", self.forecast_horizon_spin)
        form_layout.addRow("Training Epochs:", self.epochs_spin)
        
        config_layout.addLayout(form_layout)
        left_layout.addWidget(config_card)

        self.run_button = QPushButton("Run Forecast")
        self.run_button.clicked.connect(self.handle_run_forecast)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("resetButton")
        self.reset_button.clicked.connect(self.handle_reset)
        
        left_layout.addWidget(self.run_button)
        left_layout.addWidget(self.reset_button)
        left_layout.addStretch()

        content_layout.addWidget(left_panel, 1) # Weight 1

        # --- Right Panel: Logs & Results ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        results_card = QFrame()
        results_card.setObjectName("card")
        results_layout = QVBoxLayout(results_card)
        results_title = QLabel("Processing Log & Results")
        results_title.setObjectName("cardTitle")
        self.status_label = QLabel("Ready. Configure parameters and upload a file.")
        self.status_label.setObjectName("cardDescription")
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        
        self.results_df_output = QTextEdit()
        self.results_df_output.setReadOnly(True)
        self.results_df_output.setVisible(False)
        
        self.results_plots_layout = QGridLayout()

        results_layout.addWidget(results_title)
        results_layout.addWidget(self.status_label)
        results_layout.addWidget(QFrame(frameShape=QFrame.HLine, objectName="hLine"))
        results_layout.addWidget(QLabel("Logs:"))
        results_layout.addWidget(self.log_output, 1)
        results_layout.addWidget(self.results_df_output, 1)
        results_layout.addLayout(self.results_plots_layout, 2)
        
        right_layout.addWidget(results_card)
        content_layout.addWidget(right_panel, 2) # Weight 2

    def get_config_widgets(self):
        # This now just returns the widgets for easy access
        return {
            'modelName': self.model_combo,
            'trainEndYear': self.train_end_year_spin,
            'timeSteps': self.time_steps_spin,
            'forecastStartYear': self.forecast_start_year_spin,
            'forecastHorizonYears': self.forecast_horizon_spin,
            'epochs': self.epochs_spin,
        }

    def handle_config_change(self):
        # This handler now reads all values fresh when any one changes
        self.config['modelName'] = self.model_combo.currentText()
        self.config['trainEndYear'] = self.train_end_year_spin.value()
        self.config['timeSteps'] = self.time_steps_spin.value()
        self.config['forecastStartYear'] = self.forecast_start_year_spin.value()
        self.config['forecastHorizonYears'] = self.forecast_horizon_spin.value()
        self.config['epochs'] = self.epochs_spin.value()
        
        self.add_log(f"Configuration updated: {self.config}")

    def handle_file_change(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        if path:
            self.file_path = path
            self.file_label.setText(f"Loaded: {os.path.basename(path)}")
            self.file_label.setStyleSheet("color: #4ade80;") # green-400
            self.add_log(f"File selected: \"{os.path.basename(path)}\"")

    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")

    def handle_run_forecast(self):
        if not self.file_path:
            self.add_log("Error: No data file uploaded. Please select a file.")
            return
            
        self.run_button.setDisabled(True)
        self.run_button.setText("Processing...")
        self.results_df_output.setVisible(False)
        # Clear previous plots
        for i in reversed(range(self.results_plots_layout.count())): 
            self.results_plots_layout.itemAt(i).widget().setParent(None)

        self.worker = ForecastWorker(self.config)
        self.worker.log_updated.connect(self.add_log)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.processing_finished.connect(self.on_processing_finished)
        self.worker.start()

    def on_processing_finished(self, results):
        self.run_button.setDisabled(False)
        self.run_button.setText("Run Forecast")
        self.results_df_output.setText(results["dataframe"])
        self.results_df_output.setVisible(True)

        # This part would be more complex with real plotting
        # For now, we'll just show placeholders
        for i, url in enumerate(results["plots"]):
            label = QLabel(f"Plot {i+1} (Placeholder)")
            # In a real app, you would download the image and set a pixmap
            label.setPixmap(QPixmap("placeholder.png").scaled(300, 200, Qt.KeepAspectRatio))
            self.results_plots_layout.addWidget(label, i // 2, i % 2)

    def handle_reset(self):
        self.file_path = None
        self.file_label.setText("No file selected.")
        self.file_label.setStyleSheet("font-style: italic; color: #64748b;")
        self.log_output.clear()
        self.results_df_output.clear()
        self.results_df_output.setVisible(False)
        for i in reversed(range(self.results_plots_layout.count())): 
            self.results_plots_layout.itemAt(i).widget().setParent(None)
        self.add_log("Interface has been reset.")
        self.status_label.setText("Ready. Configure parameters and upload a file.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = SubsidenceForecasterApp()
    window.show()
    sys.exit(app.exec_())
