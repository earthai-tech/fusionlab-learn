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
from PyQt5.QtCore    import Qt, pyqtSignal
from PyQt5.QtGui     import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QFrame, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QFileDialog, QProgressBar, QLineEdit
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

from PyQt5.QtCore import QThread

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
        self.setWindowTitle("fusionlab-learn – PINN Mini GUI")
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
        # row = QHBoxLayout()
        # self.file_btn   = QPushButton("Select CSV…")
        # self.file_btn.clicked.connect(self._choose_file)
        # self.file_label = QLabel("No file selected")
        # self.file_label.setStyleSheet("font-style:italic;")
        # row.addWidget(self.file_btn)
        # row.addWidget(self.file_label, 1)
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
        self.batch.setValue(256)
        form.addRow("Batch size:", self.batch)

        self.lr = QDoubleSpinBox(); self.lr.setDecimals(4)
        self.lr.setValue(0.001)
        form.addRow("Learning rate:", self.lr)

        lay.addLayout(form)
        return card

    def _physics_card(self) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        lay  = QVBoxLayout(card)
        lay.addWidget(self._title("Physical Parameters"))
        lay.addWidget(hline())

        form = QFormLayout()
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
            verbose       = 1,
            # log_callback  = self._log
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
