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
import json 
import pandas as pd 

from PyQt5.QtCore    import Qt,QThread,  pyqtSignal, QAbstractTableModel, QModelIndex
from PyQt5.QtGui     import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QFrame, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QFileDialog, QProgressBar, QLineEdit, 
    QCheckBox
)
from PyQt5.QtWidgets import (
    QDialog,   QTableView, QMessageBox, QAction, QToolBar,  QInputDialog
    
)

from pathlib import Path
# add to the existing import block
from fusionlab.tools.app.config      import SubsConfig
from fusionlab.tools.app.processing  import DataProcessor, SequenceGenerator
from fusionlab.tools.app.modeling    import ModelTrainer, Forecaster
from fusionlab.tools.app.view        import ResultsVisualizer

from fusionlab.tools.app.view import VIS_SIGNALS
from fusionlab.tools.app.gui_popups  import ImagePreviewDialog   

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



# ---------- tiny editable DataFrame model -------
class _PandasModel(QAbstractTableModel):
    """Qt-model that exposes a *pandas* DataFrame (read / write)."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df                

    # -- basic shape -------
    def rowCount   (self, _=QModelIndex()): return len(self._df)
    def columnCount(self, _=QModelIndex()): return self._df.shape[1]

    # -- data ↔ Qt -
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid(): return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return str(self._df.iat[index.row(), index.column()])
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            self._df.iat[index.row(), index.column()] = value
            self.dataChanged.emit(index, index, [role]); return True
        return False

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    # -- header labels 
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return (self._df.columns[section] if orientation == Qt.Horizontal
                else str(section))

    # helper for column rename
    def rename_column(self, col_ix: int, new_name: str):
        self.beginResetModel()
        self._df.columns = (
            list(self._df.columns[:col_ix]) + [new_name] +
            list(self._df.columns[col_ix + 1 :])
        )
        self.endResetModel()

    # exposed to outside world
    @property
    def dataframe(self) -> pd.DataFrame:           # view slice!
        return self._df


# -main dialog -
class CsvEditDialog(QDialog):
    """
    Lightweight viewer / editor for CSV files.

    Parameters
    ----------
    csv_path      : str
    parent        : QWidget | None
    preview_rows  : int        – maximum rows to *display* (rest stays hidden)
    """

    def __init__(self, csv_path: str,
                 parent=None, *, preview_rows: int = 200):
        super().__init__(parent)
        self.setWindowTitle("CSV preview & editing")
        self.resize(720, 300) # 820, 500)

        # ── 1)  read full data (no row-limit !) --------------------
        try:
            self._df_full = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.critical(self, "CSV error", str(e))
            self._df_full = pd.DataFrame()
            self.reject();  return

        # slice for the *table* only (view)
        self._view_rows = min(preview_rows, len(self._df_full))
        self._df_view   = self._df_full.head(self._view_rows)

        # ── 2)  build UI ------------------------------------------
        vbox = QVBoxLayout(self)

        # toolbar
        tb = QToolBar()
        act_del_row = QAction("Delete row(s)", self)
        act_del_col = QAction("Delete col(s)", self)
        act_rename  = QAction("Rename column", self)
        tb.addAction(act_del_row); tb.addAction(act_del_col); tb.addAction(act_rename)
        vbox.addWidget(tb)

        # table
        self.table = QTableView()
        self.model = _PandasModel(self._df_view)
        self.table.setModel(self.model)
        self.table.setSelectionMode(QTableView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        vbox.addWidget(self.table, 1)

        # buttons
        hbtn = QHBoxLayout()
        hbtn.addStretch(1)
        btn_save   = QPushButton("Save / Apply")
        btn_cancel = QPushButton("Cancel")
        hbtn.addWidget(btn_save); hbtn.addWidget(btn_cancel)
        vbox.addLayout(hbtn)

        # signals
        act_del_row.triggered.connect(self._delete_rows)
        act_del_col.triggered.connect(self._delete_cols)
        act_rename .triggered.connect(self._rename_col)
        btn_save  .clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    # -------------- editing helpers --------------------------------
    def _delete_rows(self):
        rows = {ix.row() for ix in self.table.selectionModel().selectedIndexes()}
        if not rows: return
        # translate view-index → full-df index
        full_idx = self._df_full.index[list(rows)]
        self._df_full.drop(index=full_idx, inplace=True)
        self._refresh_view()

    def _delete_cols(self):
        cols = {ix.column() for ix in self.table.selectionModel().selectedIndexes()}
        if not cols: return
        cols_to_drop = self._df_full.columns[list(cols)]
        self._df_full.drop(columns=cols_to_drop, inplace=True)
        self._refresh_view()

    def _rename_col(self):
        cols = {ix.column() for ix in self.table.selectionModel().selectedIndexes()}
        if len(cols) != 1:
            QMessageBox.information(self, "Rename column",
                                     "Select exactly one column.")
            return
        col_ix = cols.pop()
        old = self._df_full.columns[col_ix]
        new, ok = QInputDialog.getText(self, "Rename column",
                                       f"New name for “{old}”:")
        if ok and new:
            self._df_full.rename(columns={old: new}, inplace=True)
            self._refresh_view()

    def _refresh_view(self):
        """Re-slice top N rows *after* edits and refresh model."""
        self._df_view = self._df_full.head(self._view_rows)
        self.model.beginResetModel()
        self.model._df = self._df_view
        self.model.endResetModel()

    # -public API -------------------------------------
    def edited_dataframe(self) -> pd.DataFrame:
        """
        Return the **full** (possibly edited) DataFrame.
        Caller should copy if it wants to keep a private version.
        """
        return self._df_full.copy()

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
    coverage_val = pyqtSignal(float)

    def __init__(self, cfg, edited_df=None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.edited_df  = edited_df   
        
        self._p = lambda frac, lo, hi: int(lo + (hi - lo) * frac)

    def run(self):
        try:
            self.status_msg.emit("📊 Pre-processing…")
            self.progress_val.emit(self._p(0, 0, 10))       # 0 %
            processor = DataProcessor(
                self.cfg, self.log_msg.emit, 
                raw_df=self.edited_df  
            )
            df_proc   = processor.run()
            self.progress_val.emit(self._p(1, 0, 10))       # 10 %

            self.status_msg.emit("🌀 Generating sequences…")
            self.progress_val.emit(self._p(0, 10, 30))      # 10 %
            seq_gen   = SequenceGenerator(self.cfg, self.log_msg.emit)
            train_ds, val_ds = seq_gen.run(
                df_proc, processor.static_features_encoded
            )
            self.progress_val.emit(self._p(1, 10, 30))      # 30 %
            
            self.status_msg.emit("🔧 Training…")
            train_range = (30, 90)
            self.cfg.progress_callback = lambda p: self.progress_val.emit(
                self._p(p / 100, *train_range)
            )
            sample_inputs, _ = next(iter(train_ds))
            shapes = {k: v.shape for k, v in sample_inputs.items()}
            model  = ModelTrainer(self.cfg, self.log_msg.emit).run(
                train_ds, val_ds, shapes
            )
            self.progress_val.emit(train_range[1])          # 90 %

            self.status_msg.emit("🔮 Forecasting…")
            self.progress_val.emit(self._p(0, 90, 100))
            forecast_df = Forecaster(self.cfg, self.log_msg.emit).run(
                model=model,
                test_df=seq_gen.test_df,
                val_dataset=val_ds,
                static_features_encoded=processor.static_features_encoded,
                coord_scaler=seq_gen.coord_scaler,
            )
            self._write_coverage_result () 
            
            ResultsVisualizer(self.cfg, self.log_msg.emit).run(forecast_df)
            self.status_msg.emit("✔ Forecast finished.")
            self.progress_val.emit(100)

        except Exception as e:
            self.log_msg.emit(f"❌ {e}")
            
    def _write_coverage_result(self) :
        if self.cfg.evaluate_coverage and self.cfg.quantiles:
            # the helper wrote '…/coverage_result.json'
            json_path = os.path.join(self.cfg.run_output_path,
                                     "coverage_result.json")
            try:
                with open(json_path, "r", encoding="utf-8") as fp:
                    cv = json.load(fp)["coverage_result"]
                    self.coverage_val.emit(float(cv))
            except Exception as e:
                self.log_msg.emit(
                    f"[WARN] Could not read coverage file: {e}")

class MiniForecaster(QMainWindow):
    
    # Qt signals that the backend can emit
    log_updated      = pyqtSignal(str)
    status_updated   = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    coverage_ready = pyqtSignal(float) 

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
        self.coverage_ready.connect(self._update_coverage_label)
        
        VIS_SIGNALS.figure_saved.connect(self._show_image_popup)

    def _show_image_popup(self, png_path: str) -> None:
        ImagePreviewDialog(png_path, parent=self).exec_()
        
    def _update_coverage_label(self, cv: float):
        """
        Slot connected to `coverage_ready(float)`.
        Shows “cov-result: 0.803”, where the number is orange (SECONDARY).
        """
        self.coverage_lbl.setText(
                f'cov-result: <span style="color:{SECONDARY};'
                f'font-weight:bold;">{cv:.3f}</span>'
            )

        #self.coverage_lbl.setText(f"cov-result: <b>{cv:.3f}</b>")
    
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
        
        # 1-bis)  City / Dataset name row  ← NEW
        city_row = QHBoxLayout()
        city_label = QLabel("City / Dataset:")
        city_row.addWidget(city_label)
        
        self.city_input = QLineEdit()
        self.city_input.setPlaceholderText("e.g. zhongshan")
        city_row.addWidget(self.city_input, 1)  # stretch to full width
        L.addLayout(city_row)

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
        
        # ❶ coverage label starts empty – will be filled later
        self.coverage_lbl = QLabel("")
        self.coverage_lbl.setStyleSheet("font-size:10px;")
        footer.addWidget(self.coverage_lbl)
        
        footer.addStretch(1)

        # ② copyright / licence at the **right**
        about = QLabel(
            '© 2025 <a href="https://earthai-tech.github.io/" '
            'style="color:#2E3191;text-decoration:none;">earthai-tech</a> – BSD-3 Clause'
        )
        about.setOpenExternalLinks(True)
        about.setStyleSheet("font-size:10px;")
        footer.addWidget(about)
        
        bottom.addLayout(footer)
        self.progress_updated.connect(self.progress_bar.setValue)

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
        self.epochs.setValue(50)
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
        if not path: 
            return 
        
        self.file_path = Path(path)
        self.file_label.setStyleSheet(f"color:{SECONDARY};")
        self.file_label.setText(f"Selected: {self.file_path.name}")
        self._log(f"CSV chosen → {self.file_path}")
        
        if not self.city_input.text().strip():
            self.city_input.setText(self.file_path.stem)
        
        # 2) pop-up preview / editor  (only now!)
        dlg = CsvEditDialog(str(self.file_path), self)
        if dlg.exec_() == QDialog.Accepted:
            self.edited_df = dlg.edited_dataframe()
            self._log(
                f"CSV preview accepted – {len(self.edited_df)} rows retained.")
        else:
            self.edited_df = None        # fall back to on-disk CSV
            self._log("CSV preview canceled – keeping original file.")
        
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
        self.city_input.clear()

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
        cfg.save_format       = "weights"       # 'tf'if you still want TF SavedModel
        cfg.bypass_loading    = True       # No need, only for inference.
        cfg.dynamic_features = dyn_list
        cfg.static_features  = stat_list
        cfg.future_features  = fut_list
        
        # ------- start worker --------------------------------------
        self.worker = Worker(
            cfg, 
            edited_df=getattr(self, "edited_df", None), 
            parent=self 
        )
        self.worker.log_msg.connect(self.log_updated.emit)
        self.worker.status_msg.connect(self.status_updated.emit)
        self.worker.progress_val.connect(self.progress_updated.emit)
        self.worker.coverage_val.connect(self.coverage_ready.emit)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()


# ── entry point ─────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(open("style.qss").read() if os.path.exists("style.qss") else "")
    app.setStyleSheet(STYLE_SHEET)  # in case STYLE_SHEET already loaded globally
    gui = MiniForecaster(); gui.show()
    sys.exit(app.exec_())
