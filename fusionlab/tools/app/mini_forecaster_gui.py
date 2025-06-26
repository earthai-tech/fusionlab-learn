"""
Mini Subsidence-Forecasting GUI (academic showcase)
"""

from __future__ import annotations 
import os, sys, time
import json 
import pandas as pd 
from pathlib import Path

from PyQt5.QtCore    import ( 
    Qt, QThread,  pyqtSignal, 
    QAbstractTableModel, 
    QModelIndex, 
    pyqtSlot
)
from PyQt5.QtGui     import QIcon, QPixmap,  QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QFrame, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QFileDialog, QProgressBar, QLineEdit, 
    QCheckBox
)
from PyQt5.QtWidgets import (
    QDialog, QTableView, QMessageBox, QAction, QToolBar, QInputDialog, 
    QToolTip
)

from fusionlab.tools.app.config      import SubsConfig
from fusionlab.tools.app.processing  import DataProcessor, SequenceGenerator
from fusionlab.tools.app.modeling    import ModelTrainer, Forecaster
from fusionlab.tools.app.view        import ResultsVisualizer
from fusionlab.tools.app.view        import VIS_SIGNALS
from fusionlab.tools.app.gui_popups  import ImagePreviewDialog   
from fusionlab.tools.app.inference import PredictionPipeline
from fusionlab.utils._manifest_registry import ( 
    ManifestRegistry, locate_manifest 
)

# Fusionlab-learn palette 
PRIMARY   = "#2E3191"   
SECONDARY = "#F28620"   
BG_LIGHT  = "#fafafa"
FG_DARK   = "#1e1e1e"
PRIMARY_T75    = "rgba(46,49,145,0.75)"      # 75 % alpha
SECONDARY_T70  = "rgba(242,134,32,0.70)"     # 70 % alpha
# ------------------------------------------------------------------ #
#  Inference-mode toggle
# ------------------------------------------------------------------ #
INFERENCE_ON  = PRIMARY
INFERENCE_OFF = "#dadada"        # greyed-out (disabled)

FLAB_STYLE_SHEET = f"""
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

QPushButton#reset, QPushButton#stop {{
    background: #dadada;
    color: #333;
}}
QPushButton#reset:hover:enabled,
QPushButton#stop:hover:enabled {{
    background: {SECONDARY};
    color: white;
}}

QToolTip {{
    /* translucent orange bubble, white text, subtle outline */
    background: {SECONDARY_T70};
    color: white;
    border: 1px solid {SECONDARY};
    border-radius: 4px;
    padding: 4px 6px;
}}

QToolTip {{
    background: {SECONDARY_T70};   /* translucent SECONDARY */
    color: white;
    border: 1px solid {SECONDARY};
    padding: 4px;
    border-radius: 4px;
}}

QPushButton#inference {{
    background: {PRIMARY};      /* overwritten at runtime */
    color: white;
    border-radius: 6px;
    padding: 6px 14px;   /* a tad wider than Stop / Reset */
}} 

QPushButton#inference:disabled {{
    background: {INFERENCE_OFF};      /* grey when no manifest yet      */
    color: #666;
}}

"""
# Fusionlab-learn palette 
PRIMARY   = "#2E3191"   
SECONDARY = "#F28620"   
BG_LIGHT  = "#fafafa"
FG_DARK   = "#1e1e1e"
PRIMARY_T75    = "rgba(46,49,145,0.75)"      # 75 % alpha
SECONDARY_T70  = "rgba(242,134,32,0.70)"     # 70 % alpha
# ------------------------------------------------------------------ #
#  Inference-mode toggle
# ------------------------------------------------------------------ #
INFERENCE_ON  = PRIMARY
INFERENCE_OFF = "#dadada"        # greyed-out (disabled)

# --- Color Palette Definition ---
# Using a central palette makes themes easier to manage.
# Inspired by common UI color systems.
PALETTE = {
    # Primary Brand Colors
    "primary": "#2E3191",    # Deep Blue
    "secondary": "#F28620",  # Orange
    
    # Dark Theme Colors
    "dark_bg": "#1e293b",       # slate-800
    "dark_card_bg": "#334155",  # slate-700
    "dark_input_bg": "#0f172a",  # slate-900
    "dark_border": "#475569",     # slate-600
    "dark_text": "#cbd5e1",      # slate-300
    "dark_text_title": "#ffffff",
    "dark_text_muted": "#94a3b8",   # slate-400
    "dark_reset_bg": "#475569",  # slate-600

    # Light Theme Colors
    "light_bg": "#f8fafc",      # slate-50
    "light_card_bg": "#ffffff",
    "light_input_bg": "#f1f5f9", # slate-100
    "light_border": "#cbd5e1",    # slate-300
    "light_text": "#0f172a",      # slate-900
    "light_text_title": "#2E3191", # Primary color for titles
    "light_text_muted": "#64748b",  # slate-500
    "light_reset_bg": "#e2e8f0", # slate-200
}

# --- Polished Dark Theme Stylesheet ---
DARK_THEME_STYLESHEET = f"""
    /* Main Window and General Widgets */
    QMainWindow, QWidget {{
        background-color: {PALETTE['dark_bg']};
        color: {PALETTE['dark_text']};
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }}
    /* Main Title Label */
    QLabel#title {{
        font-size: 28px;
        font-weight: bold;
        color: {PALETTE['dark_text_title']};
        padding: 10px;
    }}
    /* General Description Label */
    QLabel#description {{
        font-size: 14px;
        color: {PALETTE['dark_text_muted']};
    }}
    /* Card Frame Styling */
    QFrame#card {{
        background-color: {PALETTE['dark_card_bg']};
        border: 1px solid {PALETTE['dark_border']};
        border-radius: 12px;
    }}
    /* Card Title Label */
    QLabel#cardTitle {{
        font-size: 18px;
        font-weight: 600;
        color: {PALETTE['dark_text_title']};
        padding-bottom: 5px;
    }}
    /* Card Description Label */
    QLabel#cardDescription {{
        font-size: 13px;
        color: {PALETTE['dark_text_muted']};
    }}
    /* Main Action Buttons (Run, Select File) */
    QPushButton {{
        background-color: {PALETTE['primary']};
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 6px;
        font-weight: bold;
        outline: none; /* Remove outline on focus */
    }}
    QPushButton:hover {{
        background-color: #4338ca; /* Lighter shade of primary for hover */
    }}
    QPushButton:disabled {{
        background-color: #334155;
        color: {PALETTE['dark_text_muted']};
    }}
    /* Special Buttons (Reset, Stop) */
    QPushButton#resetButton, QPushButton#stopButton {{
        background-color: {PALETTE['dark_reset_bg']};
        color: {PALETTE['dark_text']};
    }}
    QPushButton#resetButton:hover, QPushButton#stopButton:hover {{
        background-color: {PALETTE['dark_border']};
    }}
    /* Input Fields (QLineEdit, QSpinBox, etc.) */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {PALETTE['dark_input_bg']};
        border: 1px solid {PALETTE['dark_border']};
        padding: 8px;
        border-radius: 6px;
        color: white;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {PALETTE['primary']}; /* Highlight on focus */
    }}
    /* Log Output Area */
    QTextEdit {{
        background-color: #020617; /* Near black for contrast */
        color: #e2e8f0;
        border: 1px solid {PALETTE['dark_border']};
        font-family: "Consolas", "Courier New", monospace;
        font-size: 13px;
    }}
    /* Horizontal Separator Line */
    QFrame#hLine {{
        border: none;
        border-top: 1px solid {PALETTE['dark_border']};
    }}
    /* Tooltip Styling */
    QToolTip {{
        background-color: {PALETTE['secondary']};
        color: white;
        border: none;
        padding: 5px;
        border-radius: 4px;
        font-size: 12px;
    }}
"""
# --- Theme Stylesheet (FusionLab Theme) ---
LIGHT_THEME_STYLESHEET = f"""
    /* --- Main Window and General Widgets --- */
    QMainWindow, QWidget {{
        background-color: {PALETTE['light_bg']};
        color: {PALETTE['light_text']};
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }}

    /* --- Header Elements --- */
    QLabel#title {{
        font-size: 28px;
        font-weight: bold;
        color: {PALETTE['light_text_title']};
        padding: 10px 0;
    }}
    QLabel#description {{
        font-size: 14px;
        color: {PALETTE['light_text_muted']};
    }}

    /* --- Card Styling --- */
    QFrame#card {{
        background-color: {PALETTE['light_card_bg']};
        border: 1px solid {PALETTE['light_border']};
        border-radius: 8px;
    }}
    QLabel#cardTitle {{
        font-size: 18px;
        font-weight: 600;
        color: {PALETTE['light_text_title']};
        padding-bottom: 5px;
    }}
    QLabel#cardDescription {{
        font-size: 13px;
        color: {PALETTE['light_text_muted']};
    }}

    /* --- Button Styling --- */
    QPushButton {{
        background-color: {PALETTE['primary']};
        color: white;
        border: none;
        padding: 9px 15px;
        border-radius: 6px;
        font-weight: bold;
        outline: none;
    }}
    QPushButton:hover {{
        background-color: {PALETTE['secondary']};
    }}
    QPushButton:disabled {{
        background-color: #e2e8f0; /* slate-200 */
        color: #94a3b8; /* slate-400 */
    }}
    QPushButton#resetButton, QPushButton#stopButton {{
        background-color: {PALETTE['light_reset_bg']};
        color: {PALETTE['light_text']};
        font-weight: normal;
    }}
    QPushButton#resetButton:hover, QPushButton#stopButton:hover {{
        background-color: #cbd5e1; /* slate-300 */
    }}

    /* --- Input Field Styling --- */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {PALETTE['light_input_bg']};
        border: 1px solid {PALETTE['light_border']};
        padding: 7px;
        border-radius: 6px;
        color: {PALETTE['light_text']};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 2px solid {PALETTE['primary']}; /* Use a thicker, colored border for focus */
        padding: 6px; /* Adjust padding to keep size consistent */
    }}
    QComboBox::drop-down {{
        border: none;
    }}

    /* --- Log Output Area --- */
    QTextEdit {{
        background-color: {PALETTE['light_card_bg']};
        border: 1px solid {PALETTE['light_border']};
        font-family: "Consolas", "Courier New", monospace;
        color: #334155;
    }}

    /* --- Miscellaneous --- */
    QFrame#hLine {{
        border: none;
        border-top: 1px solid {PALETTE['light_input_bg']};
    }}
    QToolTip {{
        background-color: {PALETTE['dark_bg']};
        color: white;
        border: none;
        padding: 5px;
        border-radius: 4px;
        font-size: 12px;
    }}
    QProgressBar {{
        border: 1px solid {PALETTE['light_border']};
        border-radius: 4px;
        text-align: center;
        background-color: {PALETTE['light_input_bg']};
    }}
    QProgressBar::chunk {{
        background-color: {PALETTE['primary']};
        border-radius: 3px;
    }}
"""

class _PandasModel(QAbstractTableModel):
    """A Qt Table Model for exposing a pandas DataFrame.

    This class acts as a bridge between the data model (a pandas
    DataFrame) and the view component (a QTableView). It provides
    the necessary interface that allows Qt to read, display, and
    modify the data from the DataFrame in a table widget.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be displayed and managed by the model.
    parent : QObject, optional
        The parent Qt object for this model, by default None.

    Attributes
    ----------
    _df : pandas.DataFrame
        The internal reference to the DataFrame being managed.

    See Also
    --------
    PyQt5.QtCore.QAbstractTableModel : The base class for custom table models.
    """

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
    def dataframe(self) -> pd.DataFrame:
        return self._df

class CsvEditDialog(QDialog):
    """A dialog window for previewing and editing a CSV file.

    This class provides a lightweight, non-destructive editing
    environment for a loaded dataset. It displays a preview of the
    data in a table and offers tools to delete rows, delete columns,
    and rename columns.

    Changes are made to an internal copy of the DataFrame and are only
    finalized and returned if the user clicks "Save / Apply".

    Parameters
    ----------
    csv_path : str
        The absolute path to the CSV file to be loaded.
    parent : QWidget, optional
        The parent widget for this dialog, by default None.
    preview_rows : int, default=200
        The maximum number of rows to display in the table view. This
        is a performance optimization to prevent lag with very large
        files, while still allowing edits on the complete, underlying
        DataFrame.

    Methods
    -------
    edited_dataframe()
        Returns the full DataFrame with all user edits applied.

    See Also
    --------
    PyQt5.QtWidgets.QDialog : The base class for dialog windows.
    _PandasModel : The data model used to populate the table view.
    """

    def __init__(self, csv_path: str,
                 parent=None, *, preview_rows: int = 200):
        super().__init__(parent)
        self.setWindowTitle("Preview & Editing Data")
        self.resize(820, 400) # 820, 500)

        try:
            self._df_full = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.critical(self, "CSV error", str(e))
            self._df_full = pd.DataFrame()
            self.reject();  return

        # slice for the *table* only (view)
        self._view_rows = min(preview_rows, len(self._df_full))
        self._df_view   = self._df_full.head(self._view_rows)

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

    # -public API 
    def edited_dataframe(self) -> pd.DataFrame:
        """
        Return the **full** (possibly edited) DataFrame.
        Caller should copy if it wants to keep a private version.
        """
        return self._df_full.copy()



class Worker(QThread):
    """Executes the forecasting workflow in a background thread.

    This QThread subclass is designed to run the entire data
    processing and model training pipeline without freezing the main
    GUI thread. It orchestrates the instantiation and execution of
    the various processing classes (`DataProcessor`, `SequenceGenerator`,
    etc.) and emits signals to update the GUI with progress, logs,
    and final results.

    Parameters
    ----------
    cfg : SubsConfig
        A configuration object containing all parameters required for
        the workflow, gathered from the GUI.
    edited_df : pd.DataFrame, optional
        If the user has edited the data in the preview dialog, this
        DataFrame is passed to bypass the initial file loading step.
        If None, the workflow will load data from the file path
        specified in the `cfg` object.
    parent : QObject, optional
        The parent Qt object, by default None.

    Attributes
    ----------
    status_msg : pyqtSignal(str)
        Emits status updates for the main status label in the GUI.
    progress_val : pyqtSignal(int)
        Emits progress updates (0-100) for the progress bar.
    log_msg : pyqtSignal(str)
        Emits detailed log messages to be displayed in the log panel.
    coverage_val : pyqtSignal(float)
        Emits the final calculated coverage score to be displayed in
        the status bar.

    Methods
    -------
    run()
        The main entry point for the thread. Executes the entire
        data processing and forecasting pipeline sequentially.
    _write_coverage_result()
        A helper method to read the saved coverage score from a JSON
        file and emit it via the `coverage_val` signal.

    See Also
    --------
    DataProcessor : Handles the data loading and preprocessing stage.
    SequenceGenerator : Handles sequence generation and dataset creation.
    ModelTrainer : Handles model definition, compilation, and training.
    Forecaster : Handles prediction on new data.
    ResultsVisualizer : Handles the final visualization of results.
    """
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
            self.cfg.progress_callback = lambda p: self.progress_val.emit(
                self._p(p / 100, 0, 10) 
            )
            processor = DataProcessor(
                self.cfg, self.log_msg.emit, 
                raw_df=self.edited_df  
            )
            df_proc   = processor.run()
            self.progress_val.emit(10)

            if self.isInterruptionRequested():          # ← CHECK #1
                return
            self.status_msg.emit("🌀 Generating sequences…")
            self.cfg.progress_callback = lambda p: self.progress_val.emit(
               self._p(p / 100, 10, 30)           # ← divide by 100!
            )      
            seq_gen   = SequenceGenerator(self.cfg, self.log_msg.emit)
            try: 
                train_ds, val_ds = seq_gen.run(
                    df_proc, processor.static_features_encoded, 
                    stop_check=self.isInterruptionRequested
                )
            except InterruptedError:
                return
              
            self.progress_val.emit(30)
            
            if self.isInterruptionRequested():          # ← CHECK #2
                return
            
            self.status_msg.emit("🔧 Training…")
            train_range = (30, 90)
            self.cfg.progress_callback = lambda p: self.progress_val.emit(
                self._p(p/100, *train_range))    # ← same here
            
            sample_inputs, _ = next(iter(train_ds))
            shapes = {k: v.shape for k, v in sample_inputs.items()}
            model  = ModelTrainer(self.cfg, self.log_msg.emit).run(
                train_ds, val_ds, shapes
            )
            self.progress_val.emit(train_range[1])          # 90 %

            if self.isInterruptionRequested():          # ← CHECK #3
                return
            self.status_msg.emit("🔮 Forecasting…")
            forecast_range = (90, 100)
            self.cfg.progress_callback = lambda p: self.progress_val.emit(
                self._p(p / 100, *forecast_range)          
            )
            forecast_df = Forecaster(self.cfg, self.log_msg.emit).run(
                model=model,
                test_df=seq_gen.test_df,
                val_dataset=val_ds,
                static_features_encoded=processor.static_features_encoded,
                coord_scaler=seq_gen.coord_scaler,
            )
            self._write_coverage_result () 
            self.status_msg.emit("✔ Forecast finished.")
            self.progress_val.emit(100)
            
            ResultsVisualizer(self.cfg, self.log_msg.emit).run(forecast_df)
            
        except Exception as e:
            self.log_msg.emit(f"❌ {e}")
            
    def _write_coverage_result(self) :
        if self.cfg.evaluate_coverage and self.cfg.quantiles:
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

    Methods
    -------
    _build_ui()
        Constructs and assembles all widgets into the main layout.
    _on_run()
        Initiates the forecasting workflow by creating and starting
        the `Worker` thread.
    _on_reset()
        Resets the GUI fields and logs to their default state.
    _choose_file()
        Opens a file dialog and handles the selection of a CSV file.
    _stop_worker()
        Requests a graceful interruption of the running `Worker` thread.
    _worker_done()
        A slot that performs cleanup actions after the worker thread
        has finished.
    _log(msg)
        A slot that appends a timestamped message to the UI log panel.

    See Also
    --------
    Worker : The QThread subclass that performs the backend processing.
    CsvEditDialog : The dialog for previewing and editing the input data.
    """
    
    # Qt signals that the backend can emit
    log_updated      = pyqtSignal(str)
    status_updated   = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    coverage_ready = pyqtSignal(float) 

    def __init__(self):
        super().__init__()
        self._log_cache: list[str] = []
        self.setWindowTitle("Fusionlab-learn – PINN Mini GUI")
        self.setFixedSize(980, 660)
        self.file_path: Path | None = None
        
        # --- app icon (title-bar & task-bar) -----------------------------
        icon_path = os.path.join(os.path.dirname(__file__),
                                 "fusionlab_learn_logo.ico")
        app_icon  = QIcon(icon_path)          # QIcon silently handles “file not found”
        self.setWindowIcon(app_icon)
        
        # --- in-window logo (top of GUI) 
        logo_lbl  = QLabel()
        pix_path  = os.path.join(os.path.dirname(__file__), "fusionlab_learn_logo.png")
        pix_logo  = QPixmap(pix_path)
        
        if not pix_logo.isNull():             # only set the pixmap if the file exists
            logo_lbl.setPixmap(
                pix_logo.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            logo_lbl.setText("Fusionlab-learn") # graceful fallback (optional)
        
        logo_lbl.setAlignment(Qt.AlignCenter)
        
        # --- Instantiate the Manifest Registry in session-only mode ---
        # This is the only change required to enable self-cleaning.
        self.registry = ManifestRegistry(session_only=True)
        
        self._inference_mode = False     # True → Run button will run inference
        self._manifest_path  = None      # filled automatically once detected

        self._build_ui()
        self.log_updated.connect(self._log)
        self.status_updated.connect(self.file_label.setText)
        self.progress_updated.connect(self.progress_bar.setValue)
        self.coverage_ready.connect(self._set_coverage)
        self._refresh_manifest_state() 
        
        self._preview_windows: list[QDialog] = []        # ← keep refs
        VIS_SIGNALS.figure_saved.connect(self._show_image_popup)

    def _show_image_popup(self, png_path: str) -> None:
        dlg = ImagePreviewDialog(png_path, parent=self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)             # frees memory on close
        dlg.show()                                        # modeless – *no* exec_()
        self._preview_windows.append(dlg)                 # keep it alive
        
        #ImagePreviewDialog(png_path, parent=self).exec_()
        
    @pyqtSlot(float)
    def _set_coverage(self, value: float):
        # orange & bold *value* only
        self.coverage_lbl.setText(
            f"cov-result:&nbsp;<span style='color:{SECONDARY}; "
            f"font-weight:bold'>{value:.3f}</span>"
        )

        #self.coverage_lbl.setText(f"cov-result: <b>{cv:.3f}</b>")
    
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

        # 0) logo + title
        # logo = QLabel()
        # logo.setPixmap(QPixmap("fusionlab_learn_logo.ico").scaled(
        #     72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # logo.setAlignment(Qt.AlignCenter)
        title = QLabel("Subsidence PINN")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size:22px; color:{PRIMARY}")
        # L.addWidget(logo)
        L.addWidget(title)

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
        
        # ── inference toggle  (disabled until a manifest is found)
        self.inf_btn = QPushButton("Inference")
        self.inf_btn.setObjectName("inference")
        self.inf_btn.setEnabled(False)                       # default: grey
        self.inf_btn.setToolTip("Load an existing run_manifest.json first.")
        self.inf_btn.clicked.connect(self._toggle_inference_mode)
        row.addWidget(self.inf_btn)
        
        # row with Run / log already exists …
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop")   
        self.stop_btn.setEnabled(False) 
        self.stop_btn.setToolTip("Abort the running workflow")                
        self.stop_btn.clicked.connect(self._stop_worker)
        row.addWidget(self.stop_btn)
        # right-hand Reset button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("reset")   
        self.reset_btn.setToolTip("Clear selections & log")
        self.reset_btn.setFixedWidth(70)

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
        self.coverage_lbl.setObjectName("covLabel") 
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

    def _stop_worker(self):
        """Requests a graceful shutdown of the background worker thread.

        This method is connected to the "Stop" button's `clicked` signal.
        It calls `requestInterruption()` on the running `Worker` instance,
        which sets a flag that the worker can check periodically to exit
        its `run` loop cleanly. It also updates the GUI status to
        reflect the stopping action.
        """
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.status_updated.emit("⏹ Stopping workflow …")
            self.stop_btn.setEnabled(False)      # grey it out
            self.worker.requestInterruption()  
        
        # if hasattr(self, "worker") and self.worker.isRunning():
        #     self._log("⏹ Stopping workflow …")
        #     self.worker.requestInterruption()   # graceful
        #     self.worker.wait(500)               # give it 0.5 s

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
        
        # # look for a manifest in registry  → enables inference
        # manifest = locate_manifest(self._log) 
    
        # if manifest is not None:
        #     self._manifest_path = str(manifest)
        #     self.inf_btn.setEnabled(True)
        #     self.inf_btn.setStyleSheet(f"background:{INFERENCE_ON};")
        #     self.inf_btn.setToolTip(
        #         "Trained model detected– click to switch to inference."
        #     )
        # else:
        #     self._manifest_path = None
        #     self.inf_btn.setEnabled(False)
        #     self.inf_btn.setStyleSheet(f"background:{INFERENCE_OFF};")
        #     self.inf_btn.setToolTip(
        #         "Inference is available once a trained run is found nearby."
        #     )
        self.progress_bar.setValue(0)
        

    def _refresh_manifest_state(self) -> None:
        """
        Check whether a trained run exists *in the registry* and toggle the
        Inference button accordingly.  Call this
          • once at start-up
          • again every time a training run finishes.
        """
        # manifest = self.registry.latest_manifest()
        manifest = locate_manifest() #(self._log) # will tract log
        if manifest:
            self._manifest_path = str(manifest)
            self.inf_btn.setEnabled(True)
            self.inf_btn.setStyleSheet(f"background:{PRIMARY};")
            self.inf_btn.setToolTip("Click to switch to inference mode.")
        else:
            self._manifest_path = None
            self.inf_btn.setEnabled(False)
            self.inf_btn.setStyleSheet(f"background:{INFERENCE_OFF};")
            self.inf_btn.setToolTip(
                "Inference becomes available after you train a model.")

    def _toggle_inference_mode(self):
        """Flip the GUI between *training* and *inference* modes."""
        self._inference_mode = not self._inference_mode
    
        if self._inference_mode:                 # → ON
            self.inf_btn.setStyleSheet(f"background:{SECONDARY}" )
            self.inf_btn.setEnabled(False)       # frozen while active
            self.inf_btn.setToolTip("Inference mode active.")
            self.run_btn.setText("RunI")
            self._log("ℹ GUI switched to *Inference* mode – "
                      "Run will now load the saved model.")
        else:                                    # → OFF  (rare, only after run)
            self.inf_btn.setStyleSheet(f"background:{PRIMARY}")
            self.inf_btn.setEnabled(True)
            self.inf_btn.setToolTip(
                "Click to switch the GUI into inference mode.")
            self.run_btn.setText("Run")

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
        self._log("ℹ Interface reset.")
        
        # self.file_path = None

    def _on_run(self):
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
        # ------------------------------------------------------------------
        if self._inference_mode:
            self._run_inference()
            return                       # *do not* go through the training path
        # ------------------------------------------------------------------
        self.coverage_lbl.clear()   
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
            
            save_format       = save_fmt,
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
        
        # Register the manifest.
        cfg.to_json()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
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
        self.worker.finished.connect(self._worker_done)

        # **Stop** button: one-shot lambda that tells the worker to stop
        self.stop_btn.clicked.connect(
            lambda: (self.worker.requestInterruption(),
                     self.status_updated.emit("⏹ Stopping…"))
        )
        
        self.worker.start()
        
    # --------------------------------------------------------------
    def _run_inference(self):
        """Kick off the PredictionPipeline inside the same progress machinery."""
        if not self._manifest_path or not Path(self._manifest_path).exists():
            QMessageBox.warning(self, "Inference",
                                "run_manifest.json not found – aborting.")
            self._toggle_inference_mode()        # reset button colours
            return
    
        self._log("▶ launching *inference* …")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)          # inference is fast; no stop
        self.progress_bar.setValue(0)
        self.progress_bar.repaint() 
    
        # # wire fake progress bar (PredictionPipeline reports 0-100)
        # cfg = SubsConfig()                       # dummy – overwritten by manifest
        # cfg.progress_callback = self.progress_updated.emit
    
        # pipe = PredictionPipeline(
        #     log_callback = self.log_updated.emit,
        # )
        # pipe.config.progress_callback = self.progress_updated.emit
    
        # try:
        #     pipe.run(validation_data_path=str(self.file_path))
        #     self.progress_bar.setValue(100)
        #     self._log("✔ inference finished.")
        # except Exception as err:
        #     self._log(f"❌ inference error: {err}")
        #     QMessageBox.critical(self, "Inference failed", str(err))
        
        # --- launch thread ---------------------------------------------
        self.infer_thr = InferenceThread(
            manifest_path = self._manifest_path,
            csv_path      = str(self.file_path),
            parent        = self,
        )
        self.infer_thr.log_msg.connect(self.log_updated.emit)
        self.infer_thr.progress_val.connect(self.progress_updated.emit)
        self.infer_thr.finished.connect(self._inference_finished)
    
        # Stop button just interrupts the thread
        self.stop_btn.clicked.connect(self.infer_thr.requestInterruption)
    
        self.infer_thr.start()
    
    def _inference_finished(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self._toggle_inference_mode()          # back to training mode

        # self.run_btn.setEnabled(True)
        # self._toggle_inference_mode()            # back to training mode

    def _worker_done(self):
        """
        Called from `self.worker.finished`.  
        Saves the whole cached log to
        <run-output-path>/gui_log_YYYYMMDD_HHMMSS.txt
        and re-enables the buttons.
        """
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # decide where to write
        out_dir = getattr(self.worker.cfg, "run_output_path", ".")
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
    
            self.status_updated.emit("⚪ Idle")
        
        # reset inference toggle (if it was active)
        if self._inference_mode:
            self._inference_mode = False
            self.inf_btn.setStyleSheet("background:%s;" % PRIMARY)
            self.inf_btn.setEnabled(True)
            self.inf_btn.setToolTip(
                "Click to switch the GUI into inference mode.")
            self.run_btn.setText("Run")
        
        self._refresh_manifest_state()  
        
class InferenceThread(QThread):
    log_msg      = pyqtSignal(str)
    progress_val = pyqtSignal(int)

    def __init__(self, manifest_path: str, csv_path: str, parent=None):
        super().__init__(parent)
        self.manifest_path = manifest_path
        self.csv_path      = csv_path

    def run(self):
        # wire fake progress bar (PredictionPipeline reports 0-100)
        cfg = SubsConfig()                       # dummy – overwritten by manifest
        cfg.progress_callback = self.progress_updated.emit
    
        pipe = PredictionPipeline(
            log_callback = self.log_updated.emit,
        )
        pipe.config.progress_callback = self.progress_updated.emit
    
        try:
            pipe.run(validation_data_path=str(self.file_path))
            self.progress_bar.setValue(100)
            self._log("✔ inference finished.")
        except Exception as err:
            self._log(f"❌ inference error: {err}")
            QMessageBox.critical(self, "Inference failed", str(err))
        
        
        
        # try:
        #     pipe = PredictionPipeline(
        #         log_callback = self.log_msg.emit,
        #     )
        #     pipe.config.progress_callback = self.progress_val.emit
        #     pipe.run(validation_data_path=self.csv_path)
        # except Exception as err:
        #     self.log_msg.emit(f"❌ inference error: {err}")
            

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

    This function is the main entry point for the entire desktop
    tool. It is responsible for setting up the PyQt5 application
    environment, applying the custom visual style, creating the main
    window instance, and starting the Qt event loop, which makes the
    application visible and interactive.

    The sequence of operations is:
    1.  Creates a ``QApplication`` instance, the core of any Qt GUI.
    2.  Sets a custom font for tooltips for better readability.
    3.  Loads the custom CSS from the ``STYLE_SHEET`` constant to give
        the application its modern, branded look and feel.
    4.  Instantiates the main window class, :class:`MiniForecaster`.
    5.  Shows the main window to the user.
    6.  Starts the application's event loop by calling `sys.exit(app.exec_())`,
        which keeps the application running until the user closes it.
    
    Parameters
    ------------
    theme (str, optional): The visual theme to apply. Can be
        'light' or 'dark'. Defaults to 'light'.
        
    """
    app = QApplication(sys.argv)
    
    QToolTip.setFont(QFont("Helvetica Neue", 9))
    
    selected_stylesheet = None
    if os.path.exists("style.qss"):
        with open("style.qss", "r", encoding="utf-8") as f:
            selected_stylesheet = f.read()
    else:
        if theme.lower() == 'dark':
            selected_stylesheet = DARK_THEME_STYLESHEET
        elif theme.lower() =='light':
            selected_stylesheet = LIGHT_THEME_STYLESHEET
        elif theme.lower() =='fusionlab': 
            selected_stylesheet = FLAB_STYLE_SHEET   
       
    if selected_stylesheet:
        app.setStyleSheet(selected_stylesheet)

    gui = MiniForecaster()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_cli()
