"""
Mini Subsidence-Forecasting GUI (academic showcase)

• PyQt5 only
• Two cards:
    1) Model configuration
    2) Physical parameters
• Live log viewer
• Fusionlab colour scheme
"""

from __future__ import annotations 
import os, sys, time
from PyQt5.QtCore    import Qt,QThread,  pyqtSignal
from PyQt5.QtGui     import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QFrame, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QFileDialog, QProgressBar, QLineEdit, 
    QCheckBox
)

from pathlib import Path
# add to the existing import block
from fusionlab.tools.app.config      import SubsConfig
from fusionlab.tools.app.processing  import DataProcessor, SequenceGenerator
from fusionlab.tools.app.modeling    import ModelTrainer, Forecaster
from fusionlab.tools.app.view        import ResultsVisualizer


# ── Fusionlab palette ────────────────────────────────────────────────
PRIMARY   = "#2E3191"   # deep indigo
SECONDARY = "#F28620"   # orange
BG_LIGHT  = "#fafafa"
FG_DARK   = "#1e1e1e"

STYLE_SHEET = f"""
QMainWindow {{
    background: {BG_LIGHT};
    color: {FG_DARK};
    font-family: 'Helvetica Neue', sans-serif;
}}

QFrame#card {{
    background: white;
    border: 2px solid {PRIMARY};
    border-radius: 12px;
}}

QLabel#cardTitle {{
    font-size: 18px;
    font-weight: 600;
    color: {PRIMARY};
}}

QPushButton {{
    background: {PRIMARY};
    color: white;
    border-radius: 6px;
    padding: 6px 12px;
}}
QPushButton:hover {{
    background: {SECONDARY};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: #f0f3ff;
    border: 1px solid {PRIMARY};
    border-radius: 4px;
    padding: 4px;
}}

QTextEdit {{
    background: #f6f6f6;
    border: 1px solid #cccccc;
}}
"""

# helper widgets
def hline() -> QFrame:
    ln = QFrame()
    ln.setFrameShape(QFrame.HLine)
    ln.setStyleSheet(f"color:{PRIMARY}")
    return ln


class Worker(QThread):
    status_msg   = pyqtSignal(str)
    progress_val = pyqtSignal(int)
    log_msg      = pyqtSignal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        try:
            self.status_msg.emit("📊 Pre-processing…")
            processor = DataProcessor(self.cfg, self.log_msg.emit)
            df_proc   = processor.run()

            self.status_msg.emit("🌀 Generating sequences…")
            seq_gen   = SequenceGenerator(self.cfg, self.log_msg.emit)
            train_ds, val_ds = seq_gen.run(
                df_proc, processor.static_features_encoded
            )

            self.status_msg.emit("🔧 Training…")
            sample_inputs, _ = next(iter(train_ds))
            shapes = {k: v.shape for k, v in sample_inputs.items()}
            model  = ModelTrainer(self.cfg, self.log_msg.emit).run(
                train_ds, val_ds, shapes
            )

            self.status_msg.emit("🔮 Forecasting…")
            forecast_df = Forecaster(self.cfg, self.log_msg.emit).run(
                model=model,
                test_df=seq_gen.test_df,
                val_dataset=val_ds,
                static_features_encoded=processor.static_features_encoded,
                coord_scaler=seq_gen.coord_scaler,
            )

            ResultsVisualizer(self.cfg, self.log_msg.emit).run(forecast_df)
            self.status_msg.emit("✔ Forecast finished.")
            self.progress_val.emit(100)

        except Exception as e:
            self.log_msg.emit(f"❌ {e}")


class MiniForecaster(QMainWindow):
    
    # Qt signals that the backend can emit
    log_updated      = pyqtSignal(str)
    status_updated   = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fusionlab-learn – PINN Mini GUI")
        self.setFixedSize(780, 560)
        self.file_path: Path | None = None

        logo = QIcon(os.path.join(os.path.dirname(__file__),
                                  "fusionlab_learn_logo.png"))
        self.setWindowIcon(logo)
        
        self._build_ui()
        self.log_updated.connect(self._log)
        self.status_updated.connect(self.file_label.setText)
        self.progress_updated.connect(self.progress_bar.setValue)
        
            # 
    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        L = QVBoxLayout(root)

        # 0) logo + title ------------------------------------------
        logo = QLabel()
        logo.setPixmap(QPixmap("fusionlab_logo.png").scaled(
            72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setAlignment(Qt.AlignCenter)
        title = QLabel("Subsidence PINN")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size:22px; color:{PRIMARY}")
        L.addWidget(logo); L.addWidget(title)

        # 1) CSV selector row 
        row = QHBoxLayout()
        row.setSpacing(8)
        
        # left-hand “Select file…” button
        self.file_btn = QPushButton("Select CSV…")
        self.file_btn.clicked.connect(self._choose_file)
        row.addWidget(self.file_btn)
        
        # centre: file-name label grows/shrinks with window
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("font-style:italic;")
        row.addWidget(self.file_label, 1)          # stretch-factor = 1
        
        # right-hand Reset button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip("Clear selections & log")
        self.reset_btn.setFixedWidth(70)
        self.reset_btn.setStyleSheet("background:#dadada; color:#333;")
        self.reset_btn.clicked.connect(self._on_reset)
        row.addWidget(self.reset_btn)
        
        L.addLayout(row)

        # 2) config cards --
        cards = QHBoxLayout(); L.addLayout(cards, 1)

        cards.addWidget(self._model_card(), 1)
        cards.addWidget(self._training_card(), 1)
        cards.addWidget(self._physics_card(), 1)
        # ▼ NEW feature card spans full width
        L.addWidget(self._feature_card())

        # 3)  Run row  +  Log pane  +  Progress bar  
        bottom = QVBoxLayout()
        L.addLayout(bottom)
        
        # ── row 1 : Run  +  log --
        row = QHBoxLayout()
        
        self.run_btn = QPushButton("Run")
        self.run_btn.setFixedWidth(60)
        self.run_btn.clicked.connect(self._on_run)
        row.addWidget(self.run_btn)
        
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        row.addWidget(self.log, 1)          # stretch
        bottom.addLayout(row)               # ← add FIRST
        

        # ── full-width progress bar 
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        bottom.addWidget(self.progress_bar)
    
        # ── single-row footer  
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
        
        # ② copyright / licence at the **right**
        about = QLabel(
            '© 2025 <a href="https://earthai-tech.github.io/" '
            'style="color:#2E3191;text-decoration:none;">earthai-tech</a> – BSD-3 Clause'
        )
        about.setOpenExternalLinks(True)
        about.setStyleSheet("font-size:10px;")
        footer.addWidget(about)
        
        bottom.addLayout(footer)

    def _training_card(self) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        lay  = QVBoxLayout(card)
        lay.addWidget(self._title("Training Parameters"))
        lay.addWidget(hline())
    
        form = QFormLayout()
    
        # Train End Year
        self.train_end_year_spin = QSpinBox()
        self.train_end_year_spin.setRange(1980, 2050)
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
    
        lay.addLayout(form)
        return card

    def _model_card(self) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        lay  = QVBoxLayout(card)
        lay.addWidget(self._title("Model Configuration"))
        lay.addWidget(hline())

        form = QFormLayout()
        self.model_select = QComboBox()
        self.model_select.addItems(["TransFlowSubsNet", "PIHALNet"])
        form.addRow("Architecture:", self.model_select)

        self.epochs = QSpinBox(); self.epochs.setRange(1, 1000)
        self.epochs.setValue(200)
        form.addRow("Epochs:", self.epochs)

        self.batch = QSpinBox(); self.batch.setRange(8, 1024)
        self.batch.setValue(32)
        form.addRow("Batch size:", self.batch)

        self.lr = QDoubleSpinBox(); self.lr.setDecimals(4)
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

    # helper
    def _title(self, txt): 
        l = QLabel(txt); l.setObjectName("cardTitle"); return l
        
    def _log(self, msg):  
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())
        # QApplication.processEvents()  
 
    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV file", "", "CSV Files (*.csv)")
        if path:
            self.file_path = Path(path)
            self.file_label.setStyleSheet(f"color:{SECONDARY};")
            self.file_label.setText(f"Selected: {self.file_path.name}")
            self._log(f"CSV chosen → {self.file_path}")

    def _on_reset(self):
        """Clear fields, log and progress bar (does not delete the CSV path)."""
        # reset feature inputs
        self.dyn_feat.setText("auto")
        self.stat_feat.setText("auto")
        self.fut_feat.setText("rainfall_mm")
    
        # clear log + status + progress
        self.log.clear()
        self.file_label.setText("No file selected" if self.file_path is None
                                else f"Selected: {self.file_path.name}")
        self.progress_bar.setValue(0)
    
        self._log("ℹ Interface reset.")
        
        # self.file_path = None


    def _on_run(self):
        self.progress_bar.setValue(0)
        if self.file_path is None:
            self._log("⚠ No CSV selected.")
            return

        self.run_btn.setEnabled(False)
        self._log("▶ launch workflow …")
        QApplication.processEvents()
        
        def _parse(txt):
            txt = txt.strip()
            return "auto" if txt.lower() == "auto" else [
                t.strip() for t in txt.split(",") if t.strip()]
        
        dyn_list  = _parse(self.dyn_feat.text())
        stat_list = _parse(self.stat_feat.text())
        fut_list  = _parse(self.fut_feat.text())

        cfg = SubsConfig(
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
            quantiles             = [float(q) for q in self.quantiles_input.text().split(',')],
            
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
            
            verbose       = 1,
    
        )
        # hand the Qt emitters to the config
        cfg.log               = self.log_updated.emit
        cfg.progress_callback = self.progress_updated.emit
        cfg.save_format       = "weights"       # if you still want TF SavedModel
        cfg.bypass_loading    = True       # as in your large GUI
        cfg.dynamic_features = dyn_list
        cfg.static_features  = stat_list
        cfg.future_features  = fut_list
        
        # ------- start worker --------------------------------------
        self.worker = Worker(cfg, self)
        self.worker.log_msg.connect(self.log_updated.emit)
        self.worker.status_msg.connect(self.status_updated.emit)
        self.worker.progress_val.connect(self.progress_updated.emit)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()

        # self.run_btn.setEnabled(True)

# ── entry point ─────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(open("style.qss").read() if os.path.exists("style.qss") else "")
    app.setStyleSheet(STYLE_SHEET)  # in case STYLE_SHEET already loaded globally
    gui = MiniForecaster(); gui.show()
    sys.exit(app.exec_())
