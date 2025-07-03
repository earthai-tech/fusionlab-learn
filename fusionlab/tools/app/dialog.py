# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
App dialog
"""

from __future__ import annotations 
import json 
import multiprocessing as mpo
import pandas as pd 

from PyQt5.QtCore import Qt
from PyQt5.QtGui     import QFont
from PyQt5.QtWidgets import (
    QVBoxLayout, 
    QHBoxLayout,
    QFormLayout, 
    QFrame, 
    QPushButton, 
    QLabel,
    QSpinBox,
    QComboBox, 
    QTextEdit, 
    QDialogButtonBox, 
    QPlainTextEdit,
    QDialog, 
    QTableView, 
    QMessageBox, 
    QAction,
    QToolBar,
    QInputDialog, 
    QGridLayout, 
    QWidget, 
    # QSpinBox, 
    QDoubleSpinBox,
    # QComboBox, 
    # QTextEdit, 
    # QFileDialog, 
    # QProgressBar, 
    QLineEdit, 
    QCheckBox,
    # QDialog, 
    # QMessageBox,
    # QToolTip, 
    QTabWidget,
    QStackedWidget, 
    QButtonGroup
)

from .notifications import show_resource_warning
from .styles import PRIMARY, SECONDARY, TAB_STYLES 
from .tables import _PandasModel 
from .utils import parse_search_space  

__all__= ["CsvEditDialog", "TunerDialog", "ModelChoiceDialog"]

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


class TunerDialog(QDialog):
    """
    One-stop Hyper-parameter Tuning dialog

    ┌────────────────────── Tuner  ──────────────────────┐
    │  [ Developer ▸ ]   [ Easy Setup ]                  │  
    ├────────────────────────────────────────────────────┤
    │  page widgets go here …                            │
    └────────────────────────────────────────────────────┘
    """
    def __init__(self, fixed_params: dict, parent=None):
        super().__init__(parent)
        # ⇩ show advisory right away
        show_resource_warning(parent=self)
        
        self.setWindowTitle("Hyper-parameter Tuning")
        self.setMinimumSize(900, 620)
        self.setStyleSheet(parent.styleSheet())

        # will hold the final dict for MiniForecaster
        self._chosen_cfg: dict | None = None
        self.fixed_params = fixed_params

        self._build_ui()           # ← creates both pages
        self._connect_signals()    # ← buttons / toggles

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
    
        # ── ❶  Toggle-bar (Developer  /  Easy Setup) ─────────────────────
        bar          = QHBoxLayout()
        self.dev_btn = QPushButton("Developer")
        self.easy_btn = QPushButton("Easy Setup")
        
        # Tool-tips
        self.dev_btn.setToolTip(
            "Power-user view • Write or paste any"
            " valid Python dictionary for the "
            "search-space and adjust every tuner knob manually."
        )
        self.easy_btn.setToolTip(
            "Wizard-style assistant • Fill in a few"
            " ranges and let the GUI generate "
            "the full search-space and tuner settings for you."
        )
    
        # ‣ colour logic: PRIMARY when *not* selected, SECONDARY when selected
        sheet = f"""
        QPushButton {{
            background      : {PRIMARY};
            color           : white;
            padding         : 6px 18px;     /* top/bottom | left/right */
            border          : none;
            border-radius   : 4px;
        }}
        
        QPushButton:checked {{
            background : {SECONDARY};
        }}
        """
        # fixed width & checkable styling
        for btn in (self.dev_btn, self.easy_btn):
            btn.setMinimumWidth(120)
            btn.setCheckable(True)
            btn.setStyleSheet(sheet)
            bar.addWidget(btn)

        # ‣ mutual exclusivity
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self.dev_btn)
        grp.addButton(self.easy_btn)
    
        # put buttons in the bar
        bar.addWidget(self.dev_btn)
        bar.addWidget(self.easy_btn)
        bar.addStretch(1)                 # push them to the left
        root.addLayout(bar)
    
        # ── ❷  Stacked pages ------------------------------------------------
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)     # stretch-factor = 1
    
        self.dev_page  = _DeveloperPage(self.fixed_params, parent=self)
        self.easy_page = _EasyPage(self.fixed_params,     parent=self)
        self.stack.addWidget(self.dev_page)
        self.stack.addWidget(self.easy_page)
    
        # ── ❸  Dialog buttons (OK / Cancel) --------------------------------
        self.btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.btn_box.button(QDialogButtonBox.Ok).setText("Start Tuning")
        root.addWidget(self.btn_box)
    
        # default page
        self.dev_btn.setChecked(True)
        self.stack.setCurrentWidget(self.dev_page)


    def _connect_signals(self):
        self.dev_btn.toggled.connect(
            lambda state: self._show_page(self.dev_page) if state else None)
        self.easy_btn.toggled.connect(
            lambda state: self._show_page(self.easy_page) if state else None)

        self.btn_box.accepted.connect(self._on_accept)
        self.btn_box.rejected.connect(self.reject)

    def _show_page(self, page: QWidget):
        """Switches the stack and keeps toggle buttons in sync."""
        self.stack.setCurrentWidget(page)
        # ensure exclusive behaviour
        self.dev_btn.setChecked(page is self.dev_page)
        self.easy_btn.setChecked(page is self.easy_page)

    # 
    # OK-button handler
    def _on_accept(self):
        # Ask whichever page is visible to assemble the config
        current = self.stack.currentWidget()
        cfg = current.get_config()        # returns None on validation error

        if cfg is None:
            return                    # keep dialog open

        self._chosen_cfg = cfg
        self.accept()                 # closes exec_()

    def chosen_config(self) -> dict | None:
        """Call **after** exec_()."""
        return self._chosen_cfg


class _DeveloperPage(QWidget):
    """
    Your original two-panel developer view – unchanged except for
    being embedded in a QWidget and exposing `get_config()`.
    """
    def __init__(self, fixed_params: dict, parent=None):
        super().__init__(parent)
        self.fixed_params = fixed_params
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(20)

        # --- Left Panel: Fixed Parameters (Read-Only) ---
        left_card = QFrame()
        left_card.setObjectName("card")
        left_layout = QVBoxLayout(left_card)
        left_title = QLabel("<b>Fixed Parameters</b>")
        left_title.setObjectName("cardTitle")
        left_desc = QLabel("These parameters are inferred from your data and are not tuned.")
        left_desc.setObjectName("cardDescription")
        
        fixed_params_text = QTextEdit()
        fixed_params_text.setReadOnly(True)
        fixed_params_text.setText(json.dumps(self.fixed_params, indent=2))
        
        left_layout.addWidget(left_title)
        left_layout.addWidget(left_desc)
        left_layout.addWidget(QFrame(frameShape=QFrame.HLine, objectName="hLine"))
        left_layout.addWidget(fixed_params_text)

        # --- Right Panel: Tuner Configuration ---
        right_card = QFrame()
        right_card.setObjectName("card")
        right_layout = QVBoxLayout(right_card)
        right_title = QLabel("<b>Tuner Configuration</b>")
        right_title.setObjectName("cardTitle")
        # short description for non-experts
        right_desc = QLabel(
            "Choose the tuning algorithm, number of trials, and the "
            "hyper-parameter search space below."
        )
        right_desc.setObjectName("cardDescription")
        right_desc.setWordWrap(True)  
        
        form_layout = QFormLayout()
        self.tuner_type_combo = QComboBox()
        self.tuner_type_combo.addItems(['randomsearch', 'bayesian', 'hyperband'])
        self.max_trials_spin = QSpinBox(minimum=1, maximum=1000, value=10)
        self.executions_spin = QSpinBox(minimum=1, maximum=10, value=1)
        
        form_layout.addRow("Tuner Algorithm:", self.tuner_type_combo)
        form_layout.addRow("Max Trials:", self.max_trials_spin)
        form_layout.addRow("Executions per Trial:", self.executions_spin)
        
        self.search_space_edit = QPlainTextEdit()
        self.search_space_edit.setPlaceholderText(
            '# Enter search space as a Python dictionary, e.g.:\n'
            '{\n'
            '    "learning_rate": [1e-4, 1e-3],\n'
            '    "num_heads": [2, 4],\n'
            '    "K": ["learnable", 1e-5]\n'
            '}'
        )
        self.search_space_edit.setFont(QFont("Consolas", 10))

        right_layout.addWidget(right_title)
        right_layout.addWidget(right_desc)     # ← THIS was missing
        right_layout.addWidget(QFrame(frameShape=QFrame.HLine, objectName="hLine"))
        right_layout.addLayout(form_layout)
        right_layout.addWidget(QLabel("Search Space:"))
        right_layout.addWidget(self.search_space_edit)

        main_layout.addWidget(left_card, 1)
        main_layout.addWidget(right_card, 2)
   
    def get_config(self) -> dict:
        """Returns the configured tuning parameters if the dialog is accepted."""
        try:
            search_space = parse_search_space(
                self.search_space_edit.toPlainText())
            if not isinstance(search_space, dict):
                raise TypeError("Search space must be a valid dictionary.")
        except Exception as e:
            QMessageBox.critical(self, "Invalid Search Space", 
                f"Could not parse the search space dictionary.\n\nError: {e}")
            return None

        return {
        "search_space": search_space,
        "tuner_settings": {
            "tuner_type":        self.tuner_type_combo.currentText(),
            "max_trials":        self.max_trials_spin.value(),
            "executions_per_trial": self.executions_spin.value(),
        },
    }


class _EasyPage(QWidget):
    """
    Wizard-like easy page (what used to be SimpleTunerDialog) –
    now also just a QWidget, returns same dict.
    """
    def __init__(self, fixed_params: dict, parent=None):
        super().__init__(parent)
        self.fixed_params = fixed_params
        self._build_ui()

    def _build_ui(self):
        
        root = QVBoxLayout(self)

        # ? fixed ----------------------------------------------------
        fixed_frm = QFrame(); fixed_frm.setObjectName("card")
        f_lo = QVBoxLayout(fixed_frm)
        lab = QLabel(
            "<b>Fixed (data-driven) parameters</b><br>"
            "<small>These come from your dataset and will not be tuned.</small>"
        )
        f_lo.addWidget(lab)
        txt = QTextEdit(); txt.setReadOnly(True)
        txt.setText(json.dumps(self.fixed_params, indent=2))
        f_lo.addWidget(txt)
        root.addWidget(fixed_frm)

        # Tabs holder
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(TAB_STYLES)
        root.addWidget(self.tabs, 1)

        # ? Model basics 
        self._build_model_tab()

        # ? Model search-space 
        self._build_search_tab()

        # ? Physics search-space 
        self._build_physics_tab()

        # ? System 
        self._build_system_tab()
        
    def _build_model_tab(self) -> None:
        t_model = QWidget()
        grid = QGridLayout(t_model)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
    
        # ── basic widgets ────────────────────────────────────────────────
        self.time_steps   = QSpinBox(minimum=1,  maximum=20,  value=4)
        self.horizon      = QSpinBox(minimum=1,  maximum=10,  value=3)
        self.batch_size   = QSpinBox(minimum=8,  maximum=1024, value=256)
        self.objective    = QLineEdit("val_loss")
    
        self.mode_combo   = QComboBox(); self.mode_combo.addItems(["pihal", "tft"])
        self.activation   = QComboBox(); self.activation.addItems(
            ["relu", "gelu", "swish", "elu", "tanh", "sigmoid", "linear"]
        )
        self.memory_size  = QSpinBox(minimum=10, maximum=500, value=100)
    
        self.use_residuals = QCheckBox("Residuals");  self.use_residuals.setChecked(True)
        self.use_bn        = QCheckBox("Batch-norm")
    
        self.patience_spin = QSpinBox(minimum=1, maximum=100, value=8)   # ← NEW
    
        self.multi_scale_agg = QComboBox(); self.multi_scale_agg.addItems(
            ["last", "average", "flatten", "auto", "sum"]
        )
        self.final_agg   = QComboBox(); self.final_agg.addItems(["last", "average", "flatten"])
        self.encoder_type= QComboBox(); self.encoder_type.addItems(["hybrid", "transformer"])
        self.decoder_stack = QComboBox(); self.decoder_stack.addItems(
            ["cross", "hierarchical", "memory"]
        )
        self.feature_proc = QComboBox(); self.feature_proc.addItems(["vsn", "norm", "none"])
    
        # helper to add one logical row ------------------------------------------------
        def add_row(r: int, label1: str, w1, label2: str, w2):
            grid.addWidget(QLabel(label1), r, 0, Qt.AlignRight)
            grid.addWidget(w1,            r, 1)
            grid.addWidget(QLabel(label2), r, 2, Qt.AlignRight)
            grid.addWidget(w2,            r, 3)
    
        add_row(0, "Time-steps (look-back):", self.time_steps,
                "Forecast horizon:",          self.horizon)
        add_row(1, "Batch size:",             self.batch_size,
                "Tuner objective:",           self.objective)
        add_row(2, "Mode:",                   self.mode_combo,
                "Activation:",                self.activation)
        add_row(3, "Memory size:",            self.memory_size,
                "Multi-scale agg:",           self.multi_scale_agg)
        add_row(4, "Final agg:",              self.final_agg,
                "Encoder type:",              self.encoder_type)
        add_row(5, "Decoder stack:",          self.decoder_stack,
                "Feature processing:",        self.feature_proc)
    
        # ── combined check-boxes + patience ----------------------------------------
        cb_box = QHBoxLayout()
        cb_box.addWidget(self.use_residuals)
        cb_box.addWidget(self.use_bn)
        cb_widget = QWidget(); cb_widget.setLayout(cb_box)
    
        grid.addWidget(cb_widget,       6, 1)              # same column (col-1)
        grid.addWidget(QLabel("Patience:"), 6, 2, Qt.AlignRight)
        grid.addWidget(self.patience_spin,  6, 3)
    
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
    
        self.tabs.addTab(t_model, "Model")

    def __build_model_tab(self):
        # ?  Model / data-driven parameters  ––– now a 2-column grid
        t_model = QWidget()
        grid     = QGridLayout(t_model)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        
    
        self.time_steps  = QSpinBox(minimum=1, maximum=20, value=4)
        self.horizon     = QSpinBox(minimum=1, maximum=10, value=3)
        self.batch_size  = QSpinBox(minimum=8, maximum=1024, value=256)
        self.objective   = QLineEdit("val_loss")
        
        self.mode_combo  = QComboBox()
        self.mode_combo.addItems(["pihal", "tft"])
        self.activation  = QComboBox()
        self.activation.addItems(
            ["relu", "gelu", "swish", "elu", "tanh", "sigmoid", "linear",])
        self.memory_size = QSpinBox(minimum=10, maximum=500, value=100)
        
        self.use_residuals = QCheckBox()
        self.use_residuals.setChecked(True)
        self.use_bn        = QCheckBox()
        self.use_bn.setChecked(False)
        
        self.multi_scale_agg = QComboBox()
        self.multi_scale_agg.addItems(["last", "average",  "flatten", "auto", "sum", ])
        
        self.final_agg  = QComboBox()
        self.final_agg.addItems(["last", "average",  "flatten"])
        
        self.encoder_type = QComboBox()
        self.encoder_type.addItems(["hybrid", "transformer"])
        
        self.decoder_stack = QComboBox()
        self.decoder_stack.addItems(["cross", "hierarchical", "memory"])
        
        self.feature_proc  = QComboBox()
        self.feature_proc.addItems(["vsn", "norm", "none"])
        
        # helper to cut boiler-plate
        def add_row(row: int, col0_lbl: str, w0, col1_lbl: str, w1):
            grid.addWidget(QLabel(col0_lbl), row, 0, Qt.AlignRight)
            grid.addWidget(w0,                row, 1)
            grid.addWidget(QLabel(col1_lbl),  row, 2, Qt.AlignRight)
            grid.addWidget(w1,                row, 3)
        
        add_row(0, "Time-steps (look-back):",  self.time_steps,
                "Forecast horizon:",           self.horizon)
        add_row(1, "Batch size:",             self.batch_size,
                "Tuner objective:",            self.objective)
        add_row(2, "Mode:",                   self.mode_combo,
                "Activation:",                 self.activation)
        add_row(3, "Memory size:",            self.memory_size,
                "Multi-scale agg:",            self.multi_scale_agg)
        add_row(4, "Final agg:",              self.final_agg,
                "Encoder type:",               self.encoder_type)
        add_row(5, "Decoder stack:",          self.decoder_stack,
                "Feature processing:",         self.feature_proc)
        add_row(6,"Use residuals:",              self.use_residuals, 
                "Use batch-norm:",         self.use_bn ) 
                
        
        # stretch so the grid uses available width nicely
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        
        self.tabs.addTab(t_model, "Model")

    # ---------------- Model search-space tab -------------------------
    def _build_search_tab(self):
        t = QWidget(); l = QFormLayout(t)

        # Helper: (label-friendly text , widget-pair)
        self._rng = lambda v,minv,maxv,step=1: (
            QSpinBox(value=v,minimum=minv,maximum=maxv,singleStep=step),
            QSpinBox(value=v*2,minimum=minv,maximum=maxv,singleStep=step)
        )

        # embed_dim
        self.embed_lo, self.embed_hi = self._rng(32, 8, 512, 8)
        # hidden_units
        self.hidden_lo, self.hidden_hi = self._rng(32, 8, 1024, 8)
        # lstm_units
        self.lstm_lo, self.lstm_hi = self._rng(64, 16, 512, 16)
        # attention_units
        self.att_lo, self.att_hi = self._rng(32, 8, 256, 8)
        # num_heads choices
        self.heads_combo = QComboBox(); self.heads_combo.addItems(["2", "4", "8"])
        # dropout
        self.do_lo = QDoubleSpinBox(decimals=2,minimum=0.0,maximum=0.9,value=0.05, singleStep=0.05)
        self.do_hi = QDoubleSpinBox(decimals=2,minimum=0.0,maximum=0.9,value=0.3,  singleStep=0.05)
        # vsn_units
        self.vsn_lo, self.vsn_hi = self._rng(16, 8, 256, 8)
        # learning-rate
        self.lr_lo = QDoubleSpinBox(decimals=6,value=1e-4,minimum=1e-6,maximum=1e-2, singleStep=1e-5)
        self.lr_hi = QDoubleSpinBox(decimals=6,value=1e-3,minimum=1e-6,maximum=1e-1, singleStep=1e-5)

        def row(lbl, w1, w2=None, suffix=""):
            box = QHBoxLayout(); box.addWidget(w1); 
            if w2: box.addWidget(w2); 
            if suffix: box.addWidget(QLabel(suffix))
            l.addRow(lbl, box)

        row("Embedding Dim",  self.embed_lo,  self.embed_hi)
        row("Hidden Units",   self.hidden_lo, self.hidden_hi)
        row("LSTM Units",     self.lstm_lo,   self.lstm_hi)
        row("Attention Units",self.att_lo,    self.att_hi)
        l.addRow("Num Heads", self.heads_combo)
        row("Drop-out",       self.do_lo,     self.do_hi)
        row("VSN Units",      self.vsn_lo,    self.vsn_hi)
        row("Learning-Rate",  self.lr_lo,     self.lr_hi)

        self.tabs.addTab(t, "Search Space")

    # ---------------- Physics search-space tab ------------------------

    def _build_physics_tab(self) -> None:
        t      = QWidget()
        layout = QFormLayout(t)
    
        # ── λ-PDE range ---------------------------------------------------------
        self.lpd_lo = QDoubleSpinBox(decimals=3, value=0.10,
                                     minimum=0.0, maximum=10.0, singleStep=0.05)
        self.lpd_hi = QDoubleSpinBox(decimals=3, value=0.50,
                                     minimum=0.0, maximum=10.0, singleStep=0.05)
        _hbox = QHBoxLayout(); _hbox.addWidget(self.lpd_lo); _hbox.addWidget(self.lpd_hi)
        layout.addRow("λ-PDE", _hbox)
    
        # helper that builds   [ learnable/fixed ▼ |  loSpin | hiSpin ]
        def _scalar_param_row(label: str, default: float):
            box   = QHBoxLayout()
            combo = QComboBox(); combo.addItems(["learnable", "fixed", "range"])
            lo    = QDoubleSpinBox(decimals=6, value=default,
                                   minimum=1e-6, maximum=1.0, singleStep=1e-5)
            hi    = QDoubleSpinBox(decimals=6, value=default*10,
                                   minimum=1e-6, maximum=1.0, singleStep=1e-5)
    
            # when mode ≠ “range” disable the second spinbox
            def _sync(idx: int):
                is_range = combo.currentText() == "range"
                lo.setEnabled(is_range or combo.currentText() == "fixed")
                hi.setEnabled(is_range)
            combo.currentIndexChanged.connect(_sync)
            _sync(0)                     # initialise
    
            box.addWidget(combo); box.addWidget(lo); box.addWidget(hi)
            layout.addRow(label, box)
            return combo, lo, hi
    
        # ── C coefficient -------------------------------------------------------
        self.c_type = QComboBox(); self.c_type.addItems(["learnable", "fixed"])
        self.c_lo   = QDoubleSpinBox(decimals=5, value=1e-3,
                                     minimum=1e-6, maximum=1.0, singleStep=1e-5)
        self.c_hi   = QDoubleSpinBox(decimals=5, value=1e-1,
                                     minimum=1e-6, maximum=1.0, singleStep=1e-5)
        _cbox = QHBoxLayout()
        _cbox.addWidget(self.c_type); _cbox.addWidget(self.c_lo); _cbox.addWidget(self.c_hi)
        self.c_lo.setEnabled(False); self.c_hi.setEnabled(False)          # (learnable) default
        def _sync_c(idx):                                          # enable both spins only
            is_fixed = self.c_type.currentText() == "fixed"        # when “fixed” selected
            self.c_lo.setEnabled(is_fixed); self.c_hi.setEnabled(False)
        self.c_type.currentIndexChanged.connect(_sync_c)
        layout.addRow("Coefficient C", _cbox)
    
        # ── K , Ss , Q ----------------------------------------------------------
        (self.k_mode,  self.k_lo,  self.k_hi)  = _scalar_param_row("K",  1e-4)
        (self.ss_mode, self.ss_lo, self.ss_hi) = _scalar_param_row("Ss", 1e-5)
        (self.q_mode,  self.q_lo,  self.q_hi)  = _scalar_param_row("Q",  0.0)
    
        # ── PDE-mode & loss weights --------------------------------------------
        self.pde_mode = QComboBox(); self.pde_mode.addItems(
            ["both", "consolidation", "gw_flow", "none"])
    
        layout.addRow("PDE mode", self.pde_mode)
    
        self.lw_subs = QDoubleSpinBox(value=1.0, minimum=0.0, maximum=10.0, singleStep=0.1)
        self.lw_gwl  = QDoubleSpinBox(value=1.0, minimum=0.0, maximum=10.0, singleStep=0.1)
        _w = QHBoxLayout(); _w.addWidget(self.lw_subs); _w.addWidget(self.lw_gwl)
        layout.addRow("Loss weights (Subs / GWL)", _w)
    
        self.tabs.addTab(t, "Physics")

    # ---------------- System tab -------------------------------------------------
    def _build_system_tab(self) -> None:
        import platform, psutil
        try:                            # GPU detection (≈ TensorFlow ≥ 2.1)
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            gpu_info = ", ".join(d.name.split("/")[-1] for d in gpus) or "— none —"
        except Exception:
            gpu_info = "— unavailable —"
    
        t      = QWidget()
        layout = QFormLayout(t)
    
        # user-editable knobs ----------------------------------------------------
        self.tuner_algo = QComboBox(); self.tuner_algo.addItems(
            ["randomsearch", "bayesian", "hyperband"])
        self.max_trials = QSpinBox(value=20, minimum=1, maximum=1000)
        self.execs      = QSpinBox(value=1, minimum=1, maximum=10)
        self.cpu_spin   = QSpinBox(value=mpo.cpu_count(), minimum=1,
                                   maximum=mpo.cpu_count())
        self.seed_spin  = QSpinBox(value=42, minimum=0, maximum=99999)
    
        layout.addRow("Tuner algorithm:",    self.tuner_algo)
        layout.addRow("Max trials:",         self.max_trials)
        layout.addRow("Executions / trial:", self.execs)
        layout.addRow("CPUs to use:",        self.cpu_spin)
        layout.addRow("Random seed:",        self.seed_spin)
    
        # ------------------------------------------------------------------------
        # auto-detected machine summary (read-only)
        # ------------------------------------------------------------------------
        sys_box   = QFrame(); sys_box.setObjectName("card")
        sys_layout= QVBoxLayout(sys_box); sys_layout.setContentsMargins(8, 6, 8, 6)
    
        title = QLabel("<b>Machine summary</b>")
        title.setObjectName("cardTitle")
        
        logical  = psutil.cpu_count(logical=True)   # all hardware threads
        physical = psutil.cpu_count(logical=False)  # real cores (None → fallback)
        
        if physical is None:                        # psutil couldn’t detect
            physical = logical
            
        info = QLabel(
            f"<pre>"
            f"OS      : {platform.system()} {platform.release()}<br>"
            f"CPU     : {platform.processor() or 'unknown'}<br>"
            f"Cores   : {logical} ({physical} physical)<br>"
            f"RAM     : {round(psutil.virtual_memory().total/2**30,1)} GB<br>"
            f"GPU(s)  : {gpu_info}"
            f"</pre>"
        )     
        
        info.setTextInteractionFlags(Qt.TextSelectableByMouse)  # allow copy-&-paste
        sys_layout.addWidget(title)
        sys_layout.addWidget(info)
    
        layout.addRow(sys_box)           # add the summary as the last row
        self.tabs.addTab(t, "System")

    def get_config(self) -> dict | None:

        # model-search space helper 
        def hp_range(lo_w, hi_w, step=None, is_float=False):
            lo, hi = lo_w.value(), hi_w.value()
            if not is_float and step:
                return {"min_value": int(lo), "max_value": int(hi), "step": step}
            if is_float:
                return {"min_value": lo, "max_value": hi}
            return {"min_value": lo, "max_value": hi}
        


        search_model = {
            "embed_dim":       hp_range(self.embed_lo,  self.embed_hi, 8),
            "hidden_units":    hp_range(self.hidden_lo, self.hidden_hi, 32),
            "lstm_units":      hp_range(self.lstm_lo,   self.lstm_hi,  32),
            "attention_units": hp_range(self.att_lo,    self.att_hi,  8),
            "num_heads":       [int(self.heads_combo.currentText())],
            "dropout_rate":    hp_range(self.do_lo, self.do_hi, is_float=True),
            "vsn_units":       hp_range(self.vsn_lo, self.vsn_hi, 16),
            "learning_rate":   {
                "min_value": self.lr_lo.value(),
                "max_value": self.lr_hi.value(),
                "sampling":  "log",
            },
        }

        # physics search-space ----------------------------------------
        def _scalar_choice(mode_cmb, lo_spin, hi_spin):
            mode = mode_cmb.currentText()
            if mode == "learnable":
                return ["learnable"]
            if mode == "fixed":
                return [lo_spin.value()]          # single fixed value
            # mode == "range"
            return {
                "min_value": lo_spin.value(),
                "max_value": hi_spin.value(),
                "sampling":  "log" if lo_spin.value() > 0 else "linear",
            }
        
        search_phys = {
            "lambda_pde":   hp_range(self.lpd_lo, self.lpd_hi, is_float=True),
            "pinn_coefficient_C_type": [self.c_type.currentText()],
            "pinn_coefficient_C": _scalar_choice(self.c_type, self.c_lo, self.c_hi),
            "K":  _scalar_choice(self.k_mode,  self.k_lo,  self.k_hi),
            "Ss": _scalar_choice(self.ss_mode, self.ss_lo, self.ss_hi),
            "Q":  _scalar_choice(self.q_mode,  self.q_lo,  self.q_hi),
            "pde_mode": [self.pde_mode.currentText()],
        }
        
        # search_phys = {
        #     "lambda_pde":     hp_range(self.lpd_lo, self.lpd_hi, is_float=True),
        #     "pinn_coefficient_C_type": [self.c_type.currentText()],
        #     "pinn_coefficient_C_value": {
        #         "min_value": self.c_val_lo.value(),
        #         "max_value": self.c_val_hi.value(),
        #         "sampling":  "log",
        #     },
        #     "K": hp_range(self.k_lo, self.k_hi, is_float=True),
        #     "Ss": hp_range(self.ss_lo, self.ss_hi, is_float=True),
        #     "Q":  hp_range(self.q_lo, self.q_hi, is_float=True),
        #     "pde_mode": [self.pde_mode.currentText()],
        # }

        # fixed params update -----------------------------------------
        fixed_upd = {
            **self.fixed_params,                # keep original
            "max_window_size": self.time_steps.value(),
            "forecast_horizon": self.horizon.value(),
            "memory_size":     self.memory_size.value(),
            "multi_scale_agg": self.multi_scale_agg.currentText(),
            "final_agg":       self.final_agg.currentText(),
            "use_residuals":   self.use_residuals.isChecked(),
            "use_batch_norm":  self.use_bn.isChecked(),
            "activation":      self.activation.currentText(),
            "mode":            self.mode_combo.currentText(),
            "architecture_config": {
                "encoder_type":          self.encoder_type.currentText(),
                "decoder_attention_stack": [
                    s.strip() for s in self.decoder_stack.currentText().split('+')
                ],
                "feature_processing":    self.feature_proc.currentText(),
            },
        }

        return {
            "sequence_params": {
                "time_steps":        self.time_steps.value(),
                "forecast_horizon":  self.horizon.value(),
            },
            "tuner_settings": {
                "tuner_type":        self.tuner_algo.currentText(),
                "max_trials":        self.max_trials.value(),
                "executions_per_trial": self.execs.value(),
                "objective":         self.objective.text().strip(),
                "seed":              self.seed_spin.value(),
                "num_cpus":          self.cpu_spin.value(),
                "patience":     self.patience_spin.value(), 
            },
            "fixed_params":  fixed_upd,
            "search_space":  {**search_model, **search_phys},
        }
 
class ModelChoiceDialog(QMessageBox):
    """
    Pops up when both a training *run_manifest.json* and a
    *tuner_run_manifest.json* live in the same run directory.
    Lets the user decide which model should be used.

    Returns:
        "train"  → original training model
        "tuned"  → best model found by HydroTuner
        None     → user pressed Cancel / closed the box
    """
    def __init__(self, theme: str = "light", parent=None):
        super().__init__(parent)
        self.setIcon(QMessageBox.Question)
        self.setWindowTitle("Which model?")
        self.setText(
            "A tuned model has been detected for this run.<br><br>"
            "<b>What do you want to use for inference?</b>"
        )
        self.setInformativeText(
            "• <b>Tuned model</b> – best hyper-parameters selected by HydroTuner<br>"
            "• <b>Training model</b> – original model saved after training"
        )

        # Buttons – use Role so we can test clickedRole later
        self.addButton("Tuned model",       QMessageBox.AcceptRole)
        self.addButton("Training model",    QMessageBox.NoRole)
        # cancel = self.addButton(QMessageBox.Cancel)
        self.addButton(QMessageBox.Cancel)
        # --- some quick styling so it blends with the app theme ---
        if theme == "dark":
            self.setStyleSheet(
                "QMessageBox {background:#334155;}"
                "QLabel{color:#e2e8f0;}"
                "QPushButton{background:#475569;color:white;padding:6px 18px;"
                "border-radius:4px;}"
                "QPushButton:hover{background:#64748b;}"
            )
        else:  # fusionlab / light
            self.setStyleSheet(
                "QMessageBox {background:#ffffff;}"
                "QLabel{color:#0f172a;}"
                "QPushButton{background:#2E3191;color:white;padding:6px 18px;"
                "border-radius:4px;}"
                "QPushButton:hover{background:#4338ca;}"
            )

    # convenience ------------------------------------------------------
    def choice(self) -> str | None:
        role = self.exec_()
        if role == QMessageBox.AcceptRole:
            return "tuned"
        if role == QMessageBox.NoRole:
            return "train"
        return None
