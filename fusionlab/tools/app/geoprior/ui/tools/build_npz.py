# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
BuildNPZTool: GUI helper to create NPZ files for GeoPriorSubsNet inference.

The goal of this tool is to let users:

* Reuse the active dataset from the main GeoPrior GUI
  (or load a dataset from <results_root>/_datasets).
* Provide the configuration either from a JSON file
  (Stage-1 manifest or NAT-style config) **or**
  manually via a small dialog with spinboxes/combos.
* Call :func:`run_build_npz` to generate NPZ files under
  ``<BASE_OUTPUT_DIR>/<CITY_NAME>_<MODEL_NAME>_npz``.

This is intentionally lighter than the full Stage-1 workflow: it only
builds NPZ sequences for inference, not full train/val/test splits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QMessageBox,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QLineEdit,
    QFileDialog,
    QFormLayout,
    QSpinBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QApplication,
    QSplitter,           
    QListWidget,         
    QListWidgetItem,    
)

from ...styles import SECONDARY_TBLUE
from ...dialogs import PopProgressDialog, choose_dataset_for_city
from ...runs.build_npz import run_build_npz


# ----------------------------------------------------------------------
# Small helper dialog: manual NPZ configuration
# ----------------------------------------------------------------------
class _FeatureLineEdit(QLineEdit):
    """
    QLineEdit that tells the dialog when it gains focus.

    Used so that double-clicking a column knows which feature box to fill.
    """
    focused = pyqtSignal(object)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.focused.emit(self)

class NpzConfigDialog(QDialog):
    """
    Dialog to collect the most important NPZ configuration parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        Active dataset used to populate column combo boxes.
    base_cfg : dict, optional
        Optional base NAT-style config. Values present here will be used
        as defaults when populating the form (if they exist).
    parent : QWidget, optional
        Standard Qt parent.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_cfg: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("NPZ parameters")
        self._df = df
        self._base_cfg = base_cfg or {}
        self._cfg: Dict[str, Any] = {}
        self._active_feature_edit: Optional[QLineEdit] = None

        self._init_ui()

    # ------------------------------------------------------------------
    def _join_from_base_cfg(self, name: str) -> str:
        """
        Join a list-like value from base_cfg into a comma-separated string.
        """
        vals = self._base_cfg.get(name, []) or []
        if isinstance(vals, (list, tuple)):
            return ", ".join(str(v) for v in vals)
        return str(vals)

    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        cols = list(self._df.columns)

        # ==============================================================
        # 1) Available columns (top)
        # ==============================================================
        grp_cols = QGroupBox("Available columns", self)
        cols_layout = QVBoxLayout(grp_cols)

        self.lst_columns = QListWidget()
        self.lst_columns.addItems(cols)
        self.lst_columns.setSelectionMode(QListWidget.SingleSelection)

        hint = QLabel(
            "Double-click a column name to append it to the currently "
            "focused driver-features box below."
        )
        hint.setStyleSheet("color: #555555; font-size: 9pt;")

        cols_layout.addWidget(self.lst_columns)
        cols_layout.addWidget(hint)

        layout.addWidget(grp_cols)

        # ==============================================================
        # 2) Middle row: core columns (left) + time window & mode (right)
        # ==============================================================
        mid_row = QHBoxLayout()
        mid_row.setSpacing(8)

        # ---------- Core columns group (Group 1) -----------------------
        grp_core = QGroupBox("Core columns", self)
        core_form = QFormLayout(grp_core)
        core_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def _mk_col_combo(default_name: Optional[str]) -> QComboBox:
            combo = QComboBox()
            combo.addItems(cols)
            if default_name and default_name in cols:
                combo.setCurrentText(default_name)
            return combo

        self.cmb_time_col = _mk_col_combo(self._base_cfg.get("TIME_COL"))
        self.cmb_lon_col = _mk_col_combo(self._base_cfg.get("LON_COL"))
        self.cmb_lat_col = _mk_col_combo(self._base_cfg.get("LAT_COL"))
        self.cmb_subs_col = _mk_col_combo(
            self._base_cfg.get("SUBSIDENCE_COL")
        )
        self.cmb_gwl_col = _mk_col_combo(self._base_cfg.get("GWL_COL"))
        self.cmb_h_field_col = _mk_col_combo(
            self._base_cfg.get("H_FIELD_COL_NAME")
        )

        core_form.addRow("Time:", self.cmb_time_col)
        core_form.addRow("Longitude:", self.cmb_lon_col)
        core_form.addRow("Latitude:", self.cmb_lat_col)
        core_form.addRow("Subsidence:", self.cmb_subs_col)
        core_form.addRow("GWL:", self.cmb_gwl_col)
        core_form.addRow("H-field:", self.cmb_h_field_col)

        # ---------- Time-window group (Group 2) ------------------------
        grp_time = QGroupBox("Time window & mode", self)
        time_form = QFormLayout(grp_time)
        time_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Infer year range from TIME_COL or "year"
        time_col_guess = self._base_cfg.get("TIME_COL")
        if not time_col_guess or time_col_guess not in cols:
            if "year" in cols:
                time_col_guess = "year"
            elif "YEAR" in cols:
                time_col_guess = "YEAR"
            else:
                time_col_guess = None

        y_min, y_max = 2000, 2030
        if time_col_guess is not None:
            try:
                series = pd.to_numeric(
                    self._df[time_col_guess], errors="coerce"
                ).dropna()
                if len(series):
                    y_min = int(series.min())
                    y_max = int(series.max())
            except Exception:
                pass

        self.spn_train_end = QSpinBox()
        self.spn_train_end.setRange(y_min, y_max)
        self.spn_train_end.setValue(
            int(self._base_cfg.get("TRAIN_END_YEAR", y_max - 1))
        )

        self.spn_forecast_start = QSpinBox()
        self.spn_forecast_start.setRange(y_min, y_max + 10)
        self.spn_forecast_start.setValue(
            int(self._base_cfg.get("FORECAST_START_YEAR", y_max))
        )

        self.spn_forecast_horizon = QSpinBox()
        self.spn_forecast_horizon.setRange(1, 30)
        self.spn_forecast_horizon.setValue(
            int(self._base_cfg.get("FORECAST_HORIZON_YEARS", 3))
        )

        self.spn_time_steps = QSpinBox()
        self.spn_time_steps.setRange(1, 60)
        self.spn_time_steps.setValue(
            int(self._base_cfg.get("TIME_STEPS", 5))
        )

        # For NPZ tool only tft_like is really operational
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("tft_like")
        self.cmb_mode.setCurrentText("tft_like")
        self.cmb_mode.setEnabled(False)

        time_form.addRow("Train end year:", self.spn_train_end)
        time_form.addRow("Forecast start year:", self.spn_forecast_start)
        time_form.addRow(
            "Forecast horizon (years):", self.spn_forecast_horizon
        )
        time_form.addRow("Time steps:", self.spn_time_steps)
        time_form.addRow("Sequence mode:", self.cmb_mode)

        mid_row.addWidget(grp_core, 1)
        mid_row.addWidget(grp_time, 1)
        layout.addLayout(mid_row)

        # ==============================================================
        # 3) Driver feature sets (Group 3) + presets row
        # ==============================================================
        grp_feat = QGroupBox("Driver feature sets", self)
        feat_layout = QVBoxLayout(grp_feat)
        feat_layout.setContentsMargins(8, 8, 8, 8)
        feat_layout.setSpacing(6)

        feat_form = QFormLayout()
        feat_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.txt_opt_num = _FeatureLineEdit(
            self._join_from_base_cfg("OPTIONAL_NUMERIC_FEATURES")
        )
        self.txt_opt_cat = _FeatureLineEdit(
            self._join_from_base_cfg("OPTIONAL_CATEGORICAL_FEATURES")
        )
        self.txt_norm = _FeatureLineEdit(
            self._join_from_base_cfg("ALREADY_NORMALIZED_FEATURES")
        )
        self.txt_future = _FeatureLineEdit(
            self._join_from_base_cfg("FUTURE_DRIVER_FEATURES")
        )
        self.txt_static = _FeatureLineEdit(
            self._join_from_base_cfg("STATIC_DRIVER_FEATURES")
        )
        self.txt_dynamic = _FeatureLineEdit(
            self._join_from_base_cfg("DYNAMIC_DRIVER_FEATURES")
        )

        for edit in (
            self.txt_opt_num,
            self.txt_opt_cat,
            self.txt_norm,
            self.txt_future,
            self.txt_static,
            self.txt_dynamic,
        ):
            edit.focused.connect(self._on_feature_focused)

        feat_form.addRow("Optional numerics:", self.txt_opt_num)
        feat_form.addRow("Optional categoricals:", self.txt_opt_cat)
        feat_form.addRow("Already-normalised:", self.txt_norm)
        feat_form.addRow("Future drivers:", self.txt_future)
        feat_form.addRow("Static drivers:", self.txt_static)
        feat_form.addRow("Dynamic drivers:", self.txt_dynamic)

        feat_layout.addLayout(feat_form)

        # ---- Presets row ---------------------------------------------
        presets_row = QHBoxLayout()
        presets_row.setSpacing(6)

        self.btn_preset_gui = QPushButton("Use GUI defaults")
        self.btn_preset_clear = QPushButton("Clear all")

        presets_row.addWidget(self.btn_preset_gui)
        presets_row.addWidget(self.btn_preset_clear)
        presets_row.addStretch(1)

        feat_layout.addLayout(presets_row)

        layout.addWidget(grp_feat)

        # ==============================================================
        # 4) Buttons
        # ==============================================================
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Column double-click → append to active feature edit
        self.lst_columns.itemDoubleClicked.connect(
            self._on_col_double_clicked
        )

        # Preset buttons
        self.btn_preset_gui.clicked.connect(self._on_preset_gui_defaults)
        self.btn_preset_clear.clicked.connect(self._on_preset_clear)

    # ------------------------------------------------------------------
    def _on_feature_focused(self, edit: QLineEdit) -> None:
        """Remember which feature box is currently active."""
        self._active_feature_edit = edit

    # ------------------------------------------------------------------
    def _on_col_double_clicked(self, item: QListWidgetItem) -> None:
        """
        Append the chosen column name to the currently focused
        driver-features QLineEdit (comma-separated).
        """
        if self._active_feature_edit is None:
            return

        name = item.text().strip()
        if not name:
            return

        current = self._active_feature_edit.text().strip()
        if not current:
            self._active_feature_edit.setText(name)
            return

        parts = [p.strip() for p in current.split(",") if p.strip()]
        if name in parts:
            return  # avoid duplicates
        parts.append(name)
        self._active_feature_edit.setText(", ".join(parts))

    # ------------------------------------------------------------------
    def _on_preset_gui_defaults(self) -> None:
        """
        Restore driver fields from the original GUI/base config.
        """
        self.txt_opt_num.setText(
            self._join_from_base_cfg("OPTIONAL_NUMERIC_FEATURES")
        )
        self.txt_opt_cat.setText(
            self._join_from_base_cfg("OPTIONAL_CATEGORICAL_FEATURES")
        )
        self.txt_norm.setText(
            self._join_from_base_cfg("ALREADY_NORMALIZED_FEATURES")
        )
        self.txt_future.setText(
            self._join_from_base_cfg("FUTURE_DRIVER_FEATURES")
        )
        self.txt_static.setText(
            self._join_from_base_cfg("STATIC_DRIVER_FEATURES")
        )
        self.txt_dynamic.setText(
            self._join_from_base_cfg("DYNAMIC_DRIVER_FEATURES")
        )

    # ------------------------------------------------------------------
    def _on_preset_clear(self) -> None:
        """Clear all driver feature fields."""
        for edit in (
            self.txt_opt_num,
            self.txt_opt_cat,
            self.txt_norm,
            self.txt_future,
            self.txt_static,
            self.txt_dynamic,
        ):
            edit.clear()

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_list(text: str) -> list[str]:
        return [t.strip() for t in text.split(",") if t.strip()]

    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Return the config fragment reflecting the user's choices.

        Only keys that are managed by this dialog are returned. The
        caller is expected to merge this with a base NAT-style config.
        """
        cfg: Dict[str, Any] = {}

        cfg["TIME_COL"] = self.cmb_time_col.currentText().strip()
        cfg["LON_COL"] = self.cmb_lon_col.currentText().strip()
        cfg["LAT_COL"] = self.cmb_lat_col.currentText().strip()
        cfg["SUBSIDENCE_COL"] = self.cmb_subs_col.currentText().strip()
        cfg["GWL_COL"] = self.cmb_gwl_col.currentText().strip()
        cfg["H_FIELD_COL_NAME"] = self.cmb_h_field_col.currentText().strip()

        cfg["TRAIN_END_YEAR"] = int(self.spn_train_end.value())
        cfg["FORECAST_START_YEAR"] = int(self.spn_forecast_start.value())
        cfg["FORECAST_HORIZON_YEARS"] = int(
            self.spn_forecast_horizon.value()
        )
        cfg["TIME_STEPS"] = int(self.spn_time_steps.value())
        cfg["MODE"] = self.cmb_mode.currentText().strip()

        cfg["OPTIONAL_NUMERIC_FEATURES"] = self._parse_list(
            self.txt_opt_num.text()
        )
        cfg["OPTIONAL_CATEGORICAL_FEATURES"] = self._parse_list(
            self.txt_opt_cat.text()
        )
        cfg["ALREADY_NORMALIZED_FEATURES"] = self._parse_list(
            self.txt_norm.text()
        )
        cfg["FUTURE_DRIVER_FEATURES"] = self._parse_list(
            self.txt_future.text()
        )
        cfg["STATIC_DRIVER_FEATURES"] = self._parse_list(
            self.txt_static.text()
        )
        cfg["DYNAMIC_DRIVER_FEATURES"] = self._parse_list(
            self.txt_dynamic.text()
        )

        return cfg


# ----------------------------------------------------------------------
# Main BuildNPZTool widget
# ----------------------------------------------------------------------
class BuildNPZTool(QWidget):
    """
    NPZ builder tool for the Tools tab.

    Parameters
    ----------
    app_ctx : object, optional
        Reference to the main :class:`GeoPriorForecaster` window.
        Used to read:
        - ``csv_path`` and ``_edited_df`` (active dataset);
        - ``geo_cfg`` (for default NAT-style config);
        - ``gui_runs_root`` / ``results_root`` (for BASE_OUTPUT_DIR);
        - ``city_edit`` for CITY_NAME.
    parent : QWidget, optional
        Standard Qt parent.
    """

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_ctx = app_ctx
        self._df: Optional[pd.DataFrame] = None
        self._csv_path: Optional[Path] = None
        self._cfg: Dict[str, Any] = {}
        self._cfg_source: str = "json"  # or "manual"

        self._init_ui()
        self._refresh_from_app()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # Splitter: top = controls, bottom = log
        splitter = QSplitter(Qt.Vertical, self)

        # --------- TOP PANE: dataset + config + run row -----------------
        top_widget = QWidget(self)
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # ------- Dataset header ----------------------------------------
        header = QHBoxLayout()
        header.setSpacing(8)

        self.lbl_dataset = QLabel("No dataset selected.")
        self.lbl_dataset.setStyleSheet("font-weight: 600;")
        header.addWidget(self.lbl_dataset, 1)

        self.btn_use_active = QPushButton("Use active dataset")
        self.btn_pick_dataset = QPushButton("Load from _datasets…")

        self.btn_use_active.setToolTip(
            "Use the dataset currently selected in the main toolbar "
            "(Open dataset…)."
        )
        self.btn_pick_dataset.setToolTip(
            "Pick a dataset for the current city from the "
            "<results_root>/_datasets folder."
        )

        header.addWidget(self.btn_use_active)
        header.addWidget(self.btn_pick_dataset)
        top_layout.addLayout(header)

        for btn in (self.btn_use_active, self.btn_pick_dataset):
            btn.setStyleSheet(
                f"QPushButton:hover {{ background-color: {SECONDARY_TBLUE}; }}"
            )

        # ------- Config source group -----------------------------------
        cfg_group = QGroupBox("Configuration source")
        cfg_layout = QVBoxLayout(cfg_group)
        
        self.rb_config_json = QRadioButton(
            "Use JSON config / Stage-1 manifest"
        )
        self.rb_manual = QRadioButton("Manual parameters (dialog)")
        self.rb_config_json.setChecked(True)
        
        self.cfg_buttons = QButtonGroup(self)
        self.cfg_buttons.addButton(self.rb_config_json)
        self.cfg_buttons.addButton(self.rb_manual)
        
        # --- NEW: put both options on the same row ---------------------
        rb_row = QHBoxLayout()
        rb_row.setSpacing(8)
        rb_row.addWidget(self.rb_config_json)
        rb_row.addWidget(self.rb_manual)
        rb_row.addStretch(1)
        
        cfg_layout.addLayout(rb_row)
        # ---------------------------------------------------------------
        
        # --- JSON row --------------------------------------------------
        json_row = QHBoxLayout()
        json_row.setSpacing(4)
        self.edit_json_path = QLineEdit()
        self.btn_browse_json = QPushButton("Browse…")
        self.btn_browse_json.setToolTip("Select a JSON config/manifest.")
        
        json_row.addWidget(QLabel("JSON file:"))
        json_row.addWidget(self.edit_json_path, 1)
        json_row.addWidget(self.btn_browse_json)
        cfg_layout.addLayout(json_row)
        
        # --- Manual row ------------------------------------------------
        manu_row = QHBoxLayout()
        manu_row.setSpacing(4)
        self.btn_edit_params = QPushButton("Edit NPZ parameters…")
        self.btn_edit_params.setEnabled(False)
        manu_row.addWidget(self.btn_edit_params)
        manu_row.addStretch(1)
        cfg_layout.addLayout(manu_row)

        # --- Config summary --------------------------------------------
        self.lbl_cfg_summary = QLabel("No configuration loaded yet.")
        self.lbl_cfg_summary.setStyleSheet("color: #444444;")
        self.lbl_cfg_summary.setWordWrap(True)
        cfg_layout.addWidget(self.lbl_cfg_summary)

        top_layout.addWidget(cfg_group)

        # ------- Output / run section ---------------------------------
        run_row = QHBoxLayout()
        run_row.setSpacing(8)

        self.lbl_output_dir = QLabel("NPZ run folder: –")
        self.btn_build = QPushButton("Build NPZ now")
        self.btn_build.setEnabled(False)

        run_row.addWidget(self.lbl_output_dir, 1)
        run_row.addWidget(self.btn_build)
        top_layout.addLayout(run_row)

        self.btn_build.setStyleSheet(
            f"QPushButton:hover {{ background-color: {SECONDARY_TBLUE}; }}"
        )

        splitter.addWidget(top_widget)

        # --------- BOTTOM PANE: log ------------------------------------
        log_group = QGroupBox("Tool log", self)
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(4, 4, 4, 4)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(80)  # was 160; free more space
        log_layout.addWidget(self.log_edit)

        splitter.addWidget(log_group)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 120])

        outer.addWidget(splitter)

        # ------- Connections -------------------------------------------
        self.btn_use_active.clicked.connect(self._refresh_from_app)
        self.btn_pick_dataset.clicked.connect(self._choose_from_datasets)

        self.rb_config_json.toggled.connect(self._on_cfg_mode_changed)
        self.btn_browse_json.clicked.connect(self._on_browse_json)
        self.btn_edit_params.clicked.connect(self._on_edit_params)

        self.btn_build.clicked.connect(self._on_build_clicked)

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def _resolve_from_app(
        self,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
        """
        Try to obtain the active dataset from the main GUI.

        Priority:
        1. ``_edited_df`` (in-memory, after Open dataset…).
        2. ``csv_path`` (on-disk CSV).
        """
        ctx = self._app_ctx
        if ctx is None:
            return None, None

        edited = getattr(ctx, "_edited_df", None)
        csv_path = getattr(ctx, "csv_path", None)

        if isinstance(edited, pd.DataFrame) and not edited.empty:
            return edited.copy(), Path(csv_path) if csv_path else None

        if csv_path is not None:
            csv_path = Path(csv_path)
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    return df, csv_path
                except Exception:
                    return None, csv_path

        return None, None

    # ------------------------------------------------------------------
    def _refresh_from_app(self) -> None:
        df, path = self._resolve_from_app()
        if df is None:
            self._set_dataset(None, None)
            self._append_log(
                "[Info] No active dataset in main GUI. Use Open dataset… "
                "first, or load from _datasets."
            )
            return
        self._set_dataset(df, path)

    # ------------------------------------------------------------------
    def _choose_from_datasets(self) -> None:
        ctx = self._app_ctx
        if ctx is None:
            QMessageBox.information(
                self,
                "No context",
                "This tool needs the main GeoPrior window.",
            )
            return

        city = ""
        if hasattr(ctx, "city_edit"):
            city = ctx.city_edit.text().strip()

        if not city:
            QMessageBox.information(
                self,
                "City required",
                "Please enter a city/dataset name in the main toolbar "
                "before picking from _datasets.",
            )
            return

        results_root = getattr(ctx, "gui_runs_root", None) or getattr(
            ctx, "results_root", None
        )
        if not results_root:
            QMessageBox.warning(
                self,
                "Results root not set",
                "Results root is not configured yet.",
            )
            return

        csv_path_str = choose_dataset_for_city(
            parent=self,
            city=city,
            results_root=Path(results_root),
        )
        if not csv_path_str:
            return

        csv_path = Path(csv_path_str)
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Dataset error",
                f"Could not read dataset:\n{csv_path}\n\n{exc}",
            )
            return

        self._set_dataset(df, csv_path)

    # ------------------------------------------------------------------
    def _set_dataset(
        self,
        df: Optional[pd.DataFrame],
        path: Optional[Path],
    ) -> None:
        self._df = df
        self._csv_path = path

        if df is None or df.empty:
            self.lbl_dataset.setText("No dataset selected.")
        else:
            rows, cols = df.shape
            if path is not None:
                self.lbl_dataset.setText(
                    f"Dataset: {path.name} ({rows:,} × {cols})"
                )
            else:
                self.lbl_dataset.setText(
                    f"In-memory dataset ({rows:,} × {cols})"
                )

        self._update_output_dir_label()
        self._update_build_enabled()

    # ------------------------------------------------------------------
    def _update_output_dir_label(self) -> None:
        ctx = self._app_ctx
        base_output_dir = None
        if ctx is not None:
            base_output_dir = getattr(ctx, "gui_runs_root", None) or getattr(
                ctx, "results_root", None
            )

        if base_output_dir is None:
            base_output_dir = os.path.join(os.getcwd(), "results")

        city = ""
        if ctx is not None and hasattr(ctx, "city_edit"):
            city = ctx.city_edit.text().strip() or "geoprior_city"

        model = "GeoPriorSubsNet"
        if ctx is not None and hasattr(ctx, "geo_cfg"):
            model = getattr(ctx.geo_cfg, "model_name", model) or model

        run_dir = os.path.join(base_output_dir, f"{city}_{model}_npz")
        self.lbl_output_dir.setText(f"NPZ run folder: {run_dir}")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def _on_cfg_mode_changed(self) -> None:
        if self.rb_config_json.isChecked():
            self._cfg_source = "json"
            self.btn_edit_params.setEnabled(False)
        else:
            self._cfg_source = "manual"
            self.btn_edit_params.setEnabled(True)

        self._update_build_enabled()

    # ------------------------------------------------------------------
    def _on_browse_json(self) -> None:
        start_dir = os.getcwd()
        current = self.edit_json_path.text().strip()
        if current:
            p = Path(current)
            if p.parent.exists():
                start_dir = str(p.parent)

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select JSON config or manifest",
            start_dir,
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return

        self.edit_json_path.setText(path)
        self._load_config_from_json(Path(path))

    # ------------------------------------------------------------------
    def _load_config_from_json(self, path: Path) -> None:
        if not path.is_file():
            QMessageBox.warning(
                self,
                "Config error",
                f"File does not exist:\n{path}",
            )
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Config error",
                f"Could not read JSON:\n{path}\n\n{exc}",
            )
            return

        # Accept either:
        # - a naked NAT-style config dict; or
        # - a Stage-1 manifest with a top-level "config" key.
        if isinstance(payload, dict) and "config" in payload:
            cfg = dict(payload["config"])
        elif isinstance(payload, dict):
            cfg = dict(payload)
        else:
            QMessageBox.warning(
                self,
                "Config error",
                "JSON content is not a dict / manifest with 'config'.",
            )
            return

        self._cfg = cfg
        self._update_cfg_summary(
            f"Loaded configuration from JSON: {path.name}"
        )
        self._update_build_enabled()

    # ------------------------------------------------------------------
    def _get_base_cfg_from_app(self) -> Dict[str, Any]:
        ctx = self._app_ctx
        if ctx is None or not hasattr(ctx, "geo_cfg"):
            return {}
        geo_cfg = ctx.geo_cfg

        # Prefer the new helper if available
        if hasattr(geo_cfg, "to_stage1_cfg_pure"):
            try:
                return dict(geo_cfg.to_stage1_cfg_pure())
            except Exception:
                pass

        base = getattr(geo_cfg, "_base_cfg", {}) or {}
        return dict(base)

    # ------------------------------------------------------------------
    def _on_edit_params(self) -> None:
        if self._df is None or self._df.empty:
            QMessageBox.information(
                self,
                "Dataset required",
                "Please select or open a dataset first.",
            )
            return

        base_cfg = self._get_base_cfg_from_app()
        dlg = NpzConfigDialog(self._df, base_cfg=base_cfg, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return

        # Merge manual fragment with base_cfg so we keep extra keys
        cfg_fragment = dlg.get_config()
        cfg = dict(base_cfg)
        cfg.update(cfg_fragment)

        # Ensure file paths are consistent with the dataset we are using
        if self._csv_path is not None:
            cfg["DATA_DIR"] = str(self._csv_path.parent)
            cfg["BIG_FN"] = self._csv_path.name

        # City / model may still come from GeoPriorConfig, but we can
        # enforce CITY_NAME from the main toolbar if present.
        ctx = self._app_ctx
        if ctx is not None and hasattr(ctx, "city_edit"):
            city = ctx.city_edit.text().strip()
            if city:
                cfg["CITY_NAME"] = city

        if ctx is not None and hasattr(ctx, "geo_cfg"):
            model = getattr(ctx.geo_cfg, "model_name", None)
            if model:
                cfg["MODEL_NAME"] = model

        self._cfg = cfg
        self._update_cfg_summary("Manual NPZ parameters captured.")
        self._update_build_enabled()

    # ------------------------------------------------------------------
    def _update_cfg_summary(self, msg: str) -> None:
        # Show a short summary: core time window + mode + main columns
        cfg = self._cfg or {}
        time_col = cfg.get("TIME_COL", "TIME_COL?")
        train_end = cfg.get("TRAIN_END_YEAR", "–")
        f_start = cfg.get("FORECAST_START_YEAR", "–")
        f_horiz = cfg.get("FORECAST_HORIZON_YEARS", "–")
        mode = cfg.get("MODE", "pinn_like")

        cols_summary = ", ".join(
            [
                f"time={time_col}",
                f"lon={cfg.get('LON_COL', 'LON?')}",
                f"lat={cfg.get('LAT_COL', 'LAT?')}",
                f"subs={cfg.get('SUBSIDENCE_COL', 'SUBS?')}",
                f"gwl={cfg.get('GWL_COL', 'GWL?')}",
                f"H={cfg.get('H_FIELD_COL_NAME', 'H?')}",
            ]
        )

        text = (
            f"{msg}\n"
            f"Window: train≤{train_end}, forecast from {f_start}, "
            f"horizon={f_horiz}, mode={mode}\n"
            f"Columns: {cols_summary}"
        )
        self.lbl_cfg_summary.setText(text)

    # ------------------------------------------------------------------
    def _update_build_enabled(self) -> None:
        have_dataset = self._df is not None and not self._df.empty
        have_cfg = bool(self._cfg)
        self.btn_build.setEnabled(have_dataset and have_cfg)

    # ------------------------------------------------------------------
    # Logging / helpers
    # ------------------------------------------------------------------
    def _append_log(self, msg: str) -> None:
        self.log_edit.appendPlainText(str(msg))
        # Ensure UI updates during long runs if we stay on the main thread
        QApplication.processEvents()

    # ------------------------------------------------------------------
    def _make_effective_cfg(self) -> Dict[str, Any]:
        """
        Prepare the final config dict to pass to run_build_npz.

        This ensures DATA_DIR / BIG_FN / CITY_NAME / MODEL_NAME are
        consistent with the dataset and GUI context.
        """
        cfg = dict(self._cfg or {})

        # Dataset-driven paths
        if self._csv_path is not None:
            cfg.setdefault("DATA_DIR", str(self._csv_path.parent))
            cfg.setdefault("BIG_FN", self._csv_path.name)

        # City / model from GUI
        ctx = self._app_ctx
        if ctx is not None and hasattr(ctx, "city_edit"):
            city = ctx.city_edit.text().strip()
            if city:
                cfg.setdefault("CITY_NAME", city)

        if ctx is not None and hasattr(ctx, "geo_cfg"):
            model = getattr(ctx.geo_cfg, "model_name", None)
            if model:
                cfg.setdefault("MODEL_NAME", model)

        return cfg

    # ------------------------------------------------------------------
    def _on_build_clicked(self) -> None:
        if self._df is None or self._df.empty:
            QMessageBox.information(
                self,
                "Dataset required",
                "Please select or open a dataset first.",
            )
            return

        if not self._cfg:
            QMessageBox.information(
                self,
                "Configuration required",
                "Please load a JSON config/manifest or define manual "
                "parameters before building NPZ.",
            )
            return

        ctx = self._app_ctx
        results_root = None
        if ctx is not None:
            results_root = getattr(ctx, "gui_runs_root", None) or getattr(
                ctx, "results_root", None
            )

        cfg = self._make_effective_cfg()

        # Decide whether to pass edited_df (in-memory dataset) or let
        # run_build_npz re-read the CSV from DATA_DIR + BIG_FN.
        edited_df: Optional[pd.DataFrame] = None
        if self._csv_path is None:
            # using an in-memory dataset
            edited_df = self._df

        # Progress dialog
        dlg = PopProgressDialog(
            parent=self,
            title="Building NPZ",
            text="Preparing NPZ sequences…",
        )
        dlg.show()
        cb = dlg.as_fraction_callback()

        def _progress(frac: float, msg: str) -> None:
            cb(frac, msg)
            self._append_log(msg)

        def _logger(msg: str) -> None:
            self._append_log(msg)

        self._append_log(
            "[BuildNPZ] Starting NPZ construction with current dataset "
            "and configuration…"
        )

        QApplication.processEvents()

        try:
            result = run_build_npz(
                cfg=cfg,
                logger=_logger,
                results_root=results_root,
                edited_df=edited_df,
                progress_callback=_progress,
            )
        except Exception as exc:
            dlg.finish()
            QMessageBox.critical(
                self,
                "NPZ build failed",
                f"An error occurred while building NPZ:\n{exc}",
            )
            self._append_log(f"[Error] NPZ build failed: {exc}")
            return

        dlg.finish()
        run_dir = result.get("run_dir") or result.get("artifacts_dir", "n/a")
        manifest_path = result.get("manifest_path", "n/a")

        self._append_log(
            f"[BuildNPZ] NPZ build complete.\n"
            f"  Run directory: {run_dir}\n"
            f"  Manifest: {manifest_path}"
        )
        QMessageBox.information(
            self,
            "NPZ build complete",
            f"NPZ sequences successfully built.\n\n"
            f"Run directory:\n{run_dir}",
        )
