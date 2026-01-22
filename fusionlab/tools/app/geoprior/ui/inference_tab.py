# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Set, Tuple, Any

from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QSignalBlocker
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QScrollArea,
    QFrame
)


from ..config.store import GeoConfigStore
from ..config.prior_schema import FieldKey, CHOICE_SPECS


MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]
MakeRunBtnFn = Callable[[str], QWidget]


def _exists(p: str) -> bool:
    try:
        return bool(p) and os.path.exists(p)
    except Exception:
        return False


def _abspath(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return p


class InferenceTab(QWidget):
    """
    Store-aware Inference tab.

    - UI is in this module (no app.py layout code)
    - GeoConfigStore is the source of truth for
      config-driven inference knobs (uncertainty, runtime)
    - File paths and dataset selectors remain UI state
      (can be promoted to config later if desired)
    """

    run_clicked = pyqtSignal()
    advanced_clicked = pyqtSignal()

    browse_model_clicked = pyqtSignal()
    browse_manifest_clicked = pyqtSignal()
    browse_inputs_clicked = pyqtSignal()
    browse_targets_clicked = pyqtSignal()
    browse_calib_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: MakeCardFn,
        make_run_button: MakeRunBtnFn,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._make_card = make_card
        self._make_run_button = make_run_button

        self._last_outputs: Dict[str, str] = {}

        self._build_ui()
        self._connect_ui()
        self._bind_store()

        self.refresh_from_store()
        self._update_widgets_state()
        self._update_preview()

    # -----------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------
    def refresh_from_store(self) -> None:
        self._sync_ui_from_store()
        self._update_preview()

    def set_last_outputs(self, outputs: Dict[str, str]) -> None:
        out = dict(outputs or {})
    
        # Backward compatible alias
        if "forecast_csv" in out and "csv_future_path" not in out:
            out["csv_future_path"] = out.get("forecast_csv", "")
    
        self._last_outputs = out
        self._sync_last_outputs_ui()
        self._update_preview()


    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        # root = QVBoxLayout(self)
        # root.setContentsMargins(6, 6, 6, 6)
        # root.setSpacing(8)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        page = QWidget(scroll)
        scroll.setWidget(page)

        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.addStretch(0)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        # ---------------------------------------------------------
        # Card A: Model & dataset (left)
        # ---------------------------------------------------------
        model_card, model_box = self._make_card("Model & dataset")

        self.inf_model_edit = QLineEdit()
        self.inf_model_edit.setPlaceholderText(
            "Select .keras model..."
        )
        self.inf_model_btn = QPushButton("Browse...")

        self.inf_manifest_edit = QLineEdit()
        self.inf_manifest_edit.setPlaceholderText(
            "Stage-1 manifest (auto if empty)"
        )
        self.inf_manifest_btn = QPushButton("Browse...")

        self.cmb_inf_dataset = QComboBox()
        self.cmb_inf_dataset.addItem("Validation (val)", "val")
        self.cmb_inf_dataset.addItem("Test (test)", "test")
        self.cmb_inf_dataset.addItem("Train (train)", "train")
        self.cmb_inf_dataset.addItem("Custom NPZ", "custom")

        self.chk_inf_use_future = QCheckBox(
            "Use Stage-1 future NPZ (forecast mode)"
        )

        self.inf_inputs_edit = QLineEdit()
        self.inf_inputs_edit.setPlaceholderText(
            "Custom inputs .npz"
        )
        self.inf_inputs_btn = QPushButton("Inputs...")

        self.inf_targets_edit = QLineEdit()
        self.inf_targets_edit.setPlaceholderText(
            "Optional targets .npz (for metrics)"
        )
        self.inf_targets_btn = QPushButton("Targets...")

        self.sp_inf_batch = QSpinBox()
        self.sp_inf_batch.setRange(1, 2048)
        self.sp_inf_batch.setValue(32)

        g1 = QGridLayout()
        r = 0
        g1.addWidget(QLabel("Model file:"), r, 0)
        g1.addWidget(self.inf_model_edit, r, 1)
        g1.addWidget(self.inf_model_btn, r, 2)
        r += 1

        g1.addWidget(QLabel("Stage-1 manifest:"), r, 0)
        g1.addWidget(self.inf_manifest_edit, r, 1)
        g1.addWidget(self.inf_manifest_btn, r, 2)
        r += 1

        g1.addWidget(QLabel("Dataset:"), r, 0)
        g1.addWidget(self.cmb_inf_dataset, r, 1, 1, 2)
        r += 1

        npz_row = QHBoxLayout()
        npz_row.addWidget(self.chk_inf_use_future)
        npz_row.addSpacing(12)
        npz_row.addWidget(QLabel("Batch size:"))
        npz_row.addWidget(self.sp_inf_batch)
        npz_row.addStretch(1)
        g1.addLayout(npz_row, r, 0, 1, 3)
        r += 1

        g1.addWidget(QLabel("Custom inputs:"), r, 0)
        g1.addWidget(self.inf_inputs_edit, r, 1)
        g1.addWidget(self.inf_inputs_btn, r, 2)
        r += 1

        g1.addWidget(QLabel("Custom targets:"), r, 0)
        g1.addWidget(self.inf_targets_edit, r, 1)
        g1.addWidget(self.inf_targets_btn, r, 2)
        r += 1

        model_box.addLayout(g1)
        model_box.addStretch(1)
        grid.addWidget(model_card, 0,0,alignment=Qt.AlignTop)

        # ---------------------------------------------------------
        # Card B: Uncertainty & outputs (right)
        # ---------------------------------------------------------
        calib_card, calib_box = self._make_card(
            "Uncertainty & outputs"
        )

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
        self.inf_calib_btn = QPushButton("Browse...")

        self.sp_inf_cov = QDoubleSpinBox()
        self.sp_inf_cov.setDecimals(3)
        self.sp_inf_cov.setRange(0.50, 0.99)
        self.sp_inf_cov.setSingleStep(0.01)
        self.sp_inf_cov.setValue(0.80)

        self.cmb_calib_mode = QComboBox()
        modes = CHOICE_SPECS.get(
            "calibration_mode",
            ("none",),
        )
        for m in modes:
            self.cmb_calib_mode.addItem(m, m)

        self.sp_calib_temp = QDoubleSpinBox()
        self.sp_calib_temp.setDecimals(3)
        self.sp_calib_temp.setRange(0.01, 100.0)
        self.sp_calib_temp.setSingleStep(0.10)
        self.sp_calib_temp.setValue(1.0)

        self.sp_cross_pen = QDoubleSpinBox()
        self.sp_cross_pen.setDecimals(4)
        self.sp_cross_pen.setRange(0.0, 1e6)
        self.sp_cross_pen.setSingleStep(0.1)
        self.sp_cross_pen.setValue(0.0)

        self.sp_cross_margin = QDoubleSpinBox()
        self.sp_cross_margin.setDecimals(4)
        self.sp_cross_margin.setRange(0.0, 1e6)
        self.sp_cross_margin.setSingleStep(0.1)
        self.sp_cross_margin.setValue(0.0)

        self.chk_inf_include_gwl = QCheckBox(
            "Include GWL columns in CSV"
        )
        self.chk_inf_include_gwl.setChecked(False)

        self.chk_inf_plots = QCheckBox("Generate plots")
        self.chk_inf_plots.setChecked(True)

        g2 = QGridLayout()
        c = 0
        row_cal = QHBoxLayout()
        row_cal.addWidget(self.chk_inf_use_source_calib)
        row_cal.addSpacing(12)
        row_cal.addWidget(self.chk_inf_fit_calib)
        row_cal.addStretch(1)
        
        g2.addLayout(row_cal, c, 0, 1, 3)
        c += 1


        g2.addWidget(QLabel("Calibrator file:"), c, 0)
        g2.addWidget(self.inf_calib_edit, c, 1)
        g2.addWidget(self.inf_calib_btn, c, 2)
        c += 1

        g2.addWidget(QLabel("Interval level:"), c, 0)
        g2.addWidget(self.sp_inf_cov, c, 1, 1, 2)
        c += 1

        g2.addWidget(QLabel("Calibration mode:"), c, 0)
        g2.addWidget(self.cmb_calib_mode, c, 1, 1, 2)
        c += 1

        g2.addWidget(QLabel("Temperature:"), c, 0)
        g2.addWidget(self.sp_calib_temp, c, 1, 1, 2)
        c += 1

        g2.addWidget(QLabel("Cross penalty:"), c, 0)
        g2.addWidget(self.sp_cross_pen, c, 1, 1, 2)
        c += 1

        g2.addWidget(QLabel("Cross margin:"), c, 0)
        g2.addWidget(self.sp_cross_margin, c, 1, 1, 2)
        c += 1

        row_out = QHBoxLayout()
        row_out.addWidget(self.chk_inf_include_gwl)
        row_out.addSpacing(12)
        row_out.addWidget(self.chk_inf_plots)
        row_out.addStretch(1)
        
        g2.addLayout(row_out, c, 0, 1, 3)
        c += 1


        calib_box.addLayout(g2)
        calib_box.addStretch(1)
        grid.addWidget(calib_card, 0, 1, alignment=Qt.AlignTop,)

        # ---------------------------------------------------------
        # Card C: Run preview (big, uses the empty space)
        # ---------------------------------------------------------
        prev_card, prev_box = self._make_card("Run preview")
        
        self.tree_preview = QTreeWidget()
        self.tree_preview.setColumnCount(2)
        self.tree_preview.setHeaderLabels(["Item", "Value"])
        self.tree_preview.setAlternatingRowColors(True)
        self.tree_preview.setRootIsDecorated(True)
        self.tree_preview.setIndentation(14)
        self.tree_preview.setMinimumHeight(220)
        
        hdr = self.tree_preview.header()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )
        
        prev_box.addWidget(self.tree_preview, 1)
        grid.addWidget(prev_card, 1, 0)

        # ---------------------------------------------------------
        # Card D: Quick actions / last outputs
        # ---------------------------------------------------------
        act_card, act_box = self._make_card("Artifacts & options")

        self.ed_last_dir = QLineEdit()
        self.ed_last_dir.setReadOnly(True)
        self.btn_open_last_dir = QPushButton("Open folder")
        self.btn_open_last_dir.setEnabled(False)
        
        self.ed_last_eval = QLineEdit()
        self.ed_last_eval.setReadOnly(True)
        self.btn_open_last_eval = QPushButton("Open CSV")
        self.btn_open_last_eval.setEnabled(False)
        
        self.ed_last_future = QLineEdit()
        self.ed_last_future.setReadOnly(True)
        self.btn_open_last_future = QPushButton("Open CSV")
        self.btn_open_last_future.setEnabled(False)
        
        self.ed_last_json = QLineEdit()
        self.ed_last_json.setReadOnly(True)
        self.btn_open_last_json = QPushButton("Open JSON")
        self.btn_open_last_json.setEnabled(False)
        
        g4 = QGridLayout()
        rr = 0
        
        g4.addWidget(QLabel("Last output dir:"), rr, 0)
        g4.addWidget(self.ed_last_dir, rr, 1)
        g4.addWidget(self.btn_open_last_dir, rr, 2)
        rr += 1
        
        g4.addWidget(QLabel("Eval CSV:"), rr, 0)
        g4.addWidget(self.ed_last_eval, rr, 1)
        g4.addWidget(self.btn_open_last_eval, rr, 2)
        rr += 1
        
        g4.addWidget(QLabel("Future CSV:"), rr, 0)
        g4.addWidget(self.ed_last_future, rr, 1)
        g4.addWidget(self.btn_open_last_future, rr, 2)
        rr += 1
        
        g4.addWidget(QLabel("Summary JSON:"), rr, 0)
        g4.addWidget(self.ed_last_json, rr, 1)
        g4.addWidget(self.btn_open_last_json, rr, 2)
        rr += 1
        
        act_box.addLayout(g4)
        act_box.addStretch(1)  
        
        self.btn_inf_options = QPushButton("Advanced options...")
        self.btn_inf_options.setMinimumHeight(34)
        self.btn_inf_options.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        act_box.addWidget(self.btn_inf_options)

        grid.addWidget(act_card, 1, 1, alignment=Qt.AlignTop)

        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # root.addLayout(grid)
        root.addLayout(grid)
        root.addStretch(1)

        outer.addWidget(scroll, 1)
        # ---------------------------------------------------------
        # Bottom row: Run
        # ---------------------------------------------------------
        run_row = QHBoxLayout()
        run_row.addStretch(1)
        
        self.btn_run_infer = self._make_run_button("Run inference")
        run_row.addWidget(self.btn_run_infer)
        
        # root.addLayout(run_row)
        outer.addLayout(run_row)

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _connect_ui(self) -> None:
        self.btn_run_infer.clicked.connect(self.run_clicked.emit)
        self.btn_inf_options.clicked.connect(
            self.advanced_clicked.emit
        )

        self.inf_model_btn.clicked.connect(
            self.browse_model_clicked.emit
        )
        self.inf_manifest_btn.clicked.connect(
            self.browse_manifest_clicked.emit
        )
        self.inf_inputs_btn.clicked.connect(
            self.browse_inputs_clicked.emit
        )
        self.inf_targets_btn.clicked.connect(
            self.browse_targets_clicked.emit
        )
        self.inf_calib_btn.clicked.connect(
            self.browse_calib_clicked.emit
        )

        self.cmb_inf_dataset.currentIndexChanged.connect(
            self._update_widgets_state
        )
        self.chk_inf_use_future.toggled.connect(
            self._update_widgets_state
        )

        for w in (
            self.inf_model_edit,
            self.inf_manifest_edit,
            self.inf_inputs_edit,
            self.inf_targets_edit,
            self.inf_calib_edit,
        ):
            w.textChanged.connect(self._update_preview)

        for w in (
            self.chk_inf_use_source_calib,
            self.chk_inf_fit_calib,
            self.chk_inf_include_gwl,
            self.chk_inf_plots,
        ):
            w.toggled.connect(self._update_preview)

        self.sp_inf_batch.valueChanged.connect(
            lambda _v: self._update_preview()
        )

        self.btn_open_last_dir.clicked.connect(
            self._open_last_dir
        )

        # Store-bound knobs: commit on change
        self.sp_inf_cov.valueChanged.connect(
            self._on_interval_level_changed
        )
        self.cmb_calib_mode.currentIndexChanged.connect(
            self._on_calib_mode_changed
        )
        self.sp_calib_temp.valueChanged.connect(
            self._on_calib_temp_changed
        )
        self.sp_cross_pen.valueChanged.connect(
            self._on_cross_pen_changed
        )
        self.sp_cross_margin.valueChanged.connect(
            self._on_cross_margin_changed
        )
        self.btn_open_last_eval.clicked.connect(
            self._open_last_eval
        )
        self.btn_open_last_future.clicked.connect(
            self._open_last_future
        )
        self.btn_open_last_json.clicked.connect(
            self._open_last_json
        )


    def _bind_store(self) -> None:
        self._store.config_changed.connect(self._on_store_changed)

    def _on_store_changed(self, keys: object) -> None:
        _ = keys
        self._sync_ui_from_store()
        self._update_preview()

    # -----------------------------------------------------------------
    # Store sync (only store-driven fields)
    # -----------------------------------------------------------------
    def _sync_ui_from_store(self) -> None:
        s = self._store

        with QSignalBlocker(self.sp_inf_cov):
            v = s.get_value(FieldKey("interval_level"), default=0.80)
            try:
                self.sp_inf_cov.setValue(float(v))
            except Exception:
                pass

        with QSignalBlocker(self.cmb_calib_mode):
            v = s.get_value(
                FieldKey("calibration_mode"),
                default="none",
            )
            idx = self.cmb_calib_mode.findData(str(v))
            if idx >= 0:
                self.cmb_calib_mode.setCurrentIndex(idx)

        with QSignalBlocker(self.sp_calib_temp):
            v = s.get_value(
                FieldKey("calibration_temperature"),
                default=1.0,
            )
            try:
                self.sp_calib_temp.setValue(float(v))
            except Exception:
                pass

        with QSignalBlocker(self.sp_cross_pen):
            v = s.get_value(
                FieldKey("crossing_penalty"),
                default=0.0,
            )
            try:
                self.sp_cross_pen.setValue(float(v))
            except Exception:
                pass

        with QSignalBlocker(self.sp_cross_margin):
            v = s.get_value(
                FieldKey("crossing_margin"),
                default=0.0,
            )
            try:
                self.sp_cross_margin.setValue(float(v))
            except Exception:
                pass

    def _store_set(self, name: str, value: Any) -> None:
        try:
            self._store.set_value_by_key(FieldKey(name), value)
        except Exception:
            return

    def _on_interval_level_changed(self, v: float) -> None:
        self._store_set("interval_level", float(v))

    def _on_calib_mode_changed(self, _i: int) -> None:
        v = self.cmb_calib_mode.currentData() or "none"
        self._store_set("calibration_mode", str(v))
        self._update_preview()

    def _on_calib_temp_changed(self, v: float) -> None:
        self._store_set("calibration_temperature", float(v))

    def _on_cross_pen_changed(self, v: float) -> None:
        self._store_set("crossing_penalty", float(v))

    def _on_cross_margin_changed(self, v: float) -> None:
        self._store_set("crossing_margin", float(v))

    # -----------------------------------------------------------------
    # UI state rules
    # -----------------------------------------------------------------
    def _update_widgets_state(self) -> None:
        key = self.cmb_inf_dataset.currentData() or "test"
        use_future = self.chk_inf_use_future.isChecked()

        custom = (key == "custom") and (not use_future)

        self.inf_inputs_edit.setEnabled(custom)
        self.inf_inputs_btn.setEnabled(custom)

        self.inf_targets_edit.setEnabled(custom)
        self.inf_targets_btn.setEnabled(custom)

        self._update_preview()

    # -----------------------------------------------------------------
    # Preview + quick actions
    # -----------------------------------------------------------------
    def _sync_last_outputs_ui(self) -> None:
        out_dir = self._last_outputs.get("run_dir", "")
        csv_eval = self._last_outputs.get("csv_eval_path", "")
        csv_future = self._last_outputs.get("csv_future_path", "")
        js = self._last_outputs.get("inference_summary_json", "")
    
        self.ed_last_dir.setText(out_dir)
        self.ed_last_eval.setText(csv_eval)
        self.ed_last_future.setText(csv_future)
        self.ed_last_json.setText(js)
    
        self.btn_open_last_dir.setEnabled(_exists(out_dir))
        self.btn_open_last_eval.setEnabled(_exists(csv_eval))
        self.btn_open_last_future.setEnabled(_exists(csv_future))
        self.btn_open_last_json.setEnabled(_exists(js))
    
    def _open_last_eval(self) -> None:
        p = self.ed_last_eval.text().strip()
        if _exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))
    
    def _open_last_future(self) -> None:
        p = self.ed_last_future.text().strip()
        if _exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))
    
    def _open_last_json(self) -> None:
        p = self.ed_last_json.text().strip()
        if _exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _open_last_dir(self) -> None:
        p = self.ed_last_dir.text().strip()
        if _exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _open_last_csv(self) -> None:
        self._open_last_future()
        
    def _pv_section(self, title: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([title, ""])
        item.setFirstColumnSpanned(True)
    
        f = QFont(item.font(0))
        f.setBold(True)
        item.setFont(0, f)
    
        self.tree_preview.addTopLevelItem(item)
        item.setExpanded(True)
        return item
    
    
    def _pv_kv(
        self,
        parent: QTreeWidgetItem,
        key: str,
        val: str,
    ) -> None:
        QTreeWidgetItem(parent, [key, val])

    def _update_preview(self) -> None:
        model_p = self.inf_model_edit.text().strip()
        mani_p = self.inf_manifest_edit.text().strip()

        dkey = self.cmb_inf_dataset.currentData() or "test"
        use_future = self.chk_inf_use_future.isChecked()

        inputs_p = self.inf_inputs_edit.text().strip()
        targets_p = self.inf_targets_edit.text().strip()

        cov = float(self.sp_inf_cov.value())
        mode = self.cmb_calib_mode.currentData() or "none"
        temp = float(self.sp_calib_temp.value())

        src_cal = self.chk_inf_use_source_calib.isChecked()
        fit_cal = self.chk_inf_fit_calib.isChecked()
        cal_p = self.inf_calib_edit.text().strip()

        inc_gwl = self.chk_inf_include_gwl.isChecked()
        plots = self.chk_inf_plots.isChecked()
        bsz = int(self.sp_inf_batch.value())

        run_dir = ""
        if model_p:
            mp = _abspath(model_p)
            run_dir = mp if os.path.isdir(mp) else os.path.dirname(mp)

        warn: list[str] = []
        if not model_p:
            warn.append("- Missing model path.")
        elif not _exists(model_p):
            warn.append("- Model path does not exist.")

        if (dkey == "custom") and (not use_future):
            if not inputs_p:
                warn.append("- Custom inputs NPZ is required.")
            elif not _exists(inputs_p):
                warn.append("- Inputs NPZ does not exist.")

            if targets_p and (not _exists(targets_p)):
                warn.append("- Targets NPZ does not exist.")

        if mani_p and (not _exists(mani_p)):
            warn.append("- Manifest path does not exist.")

        if cal_p and (not _exists(cal_p)):
            warn.append("- Calibrator file does not exist.")

        lines: list[str] = []
        lines.append("Resolved run plan")
        lines.append("-----------------")
        lines.append(f"Dataset: {dkey}  |  future: {use_future}")
        lines.append(f"Batch size: {bsz}")
        lines.append("")
        lines.append(f"Model: {_abspath(model_p) or '(empty)'}")
        lines.append(
            f"Stage-1 manifest: {_abspath(mani_p) or '(auto)'}"
        )
        lines.append(f"Run dir: {run_dir or '(unknown)'}")
        lines.append("")
        lines.append("Uncertainty & calibration (store)")
        lines.append("-------------------------------")
        lines.append(f"Interval level: {cov:.3f}")
        lines.append(f"Mode: {mode}  |  T: {temp:.3f}")
        lines.append(
            f"Cross penalty: {self.sp_cross_pen.value():.4f}"
        )
        lines.append(
            f"Cross margin: {self.sp_cross_margin.value():.4f}"
        )
        lines.append("")
        lines.append("Calibrator inputs (runtime)")
        lines.append("---------------------------")
        lines.append(f"Use source calibrator: {src_cal}")
        lines.append(f"Fit on val split: {fit_cal}")
        lines.append(f"Explicit file: {_abspath(cal_p) or '(none)'}")
        lines.append("")
        lines.append("Outputs (runtime)")
        lines.append("-----------------")
        lines.append(f"Include GWL: {inc_gwl}")
        lines.append(f"Generate plots: {plots}")

        if warn:
            lines.append("")
            lines.append("Warnings")
            lines.append("--------")
            lines.extend(warn)

        self.tree_preview.clear()
        
        ready = "OK" if not warn else "Needs attention"
        sec = self._pv_section("Readiness")
        self._pv_kv(sec, "Status", ready)
        self._pv_kv(sec, "Warnings", str(len(warn)))
        
        sec = self._pv_section("Inputs")
        self._pv_kv(sec, "Dataset", f"{dkey} (future={use_future})")
        self._pv_kv(sec, "Batch size", str(bsz))
        self._pv_kv(sec, "Model", _abspath(model_p) or "(empty)")
        self._pv_kv(
            sec,
            "Stage-1 manifest",
            _abspath(mani_p) or "(auto)",
        )
        
        sec = self._pv_section("Uncertainty (store)")
        self._pv_kv(sec, "Interval level", f"{cov:.3f}")
        self._pv_kv(sec, "Mode", str(mode))
        self._pv_kv(sec, "Temperature", f"{temp:.3f}")
        self._pv_kv(
            sec,
            "Cross penalty",
            f"{self.sp_cross_pen.value():.4f}",
        )
        self._pv_kv(
            sec,
            "Cross margin",
            f"{self.sp_cross_margin.value():.4f}",
        )
        
        sec = self._pv_section("Calibration (runtime)")
        self._pv_kv(sec, "Use source", str(src_cal))
        self._pv_kv(sec, "Fit on val", str(fit_cal))
        self._pv_kv(
            sec,
            "Explicit file",
            _abspath(cal_p) or "(none)",
        )
        
        sec = self._pv_section("Outputs (runtime)")
        self._pv_kv(sec, "Include GWL", str(inc_gwl))
        self._pv_kv(sec, "Generate plots", str(plots))
        
        if warn:
            sec = self._pv_section("Warnings")
            for w in warn:
                self._pv_kv(sec, "", w)

