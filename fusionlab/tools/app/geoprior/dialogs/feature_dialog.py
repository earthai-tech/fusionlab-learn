# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations
import re
import difflib
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QMessageBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QGroupBox,
    QHBoxLayout,      
    QPushButton,      
    QTextBrowser,     
)

class ColumnsPlainTextEdit(QPlainTextEdit):
    """
    Read-only view of available columns that emits the column name
    on double-click.
    """
    columnActivated = pyqtSignal(str)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        cursor = self.cursorForPosition(event.pos())
        cursor.select(QTextCursor.WordUnderCursor)
        text = cursor.selectedText().strip().strip(",; ")
        if text:
            self.columnActivated.emit(text)
        super().mouseDoubleClickEvent(event)

class FeatureConfigDialog(QDialog):
    """Dialog to edit features and censoring."""

    def __init__(
        self,
        csv_path: Path,
        base_cfg: Dict[str, Any] | None = None,
        current_overrides: Dict[str, Any] | None = None,
        df: pd.DataFrame | None = None,
        parent: None | "QDialog" = None,
    ) -> None:
        super().__init__(parent)
        self._csv_path = Path(csv_path) if csv_path is not None else None
        self._base_cfg = base_cfg or {}
        self._over_in = current_overrides or {}
        self._df = df

        cfg: Dict[str, Any] = dict(self._base_cfg)
        cfg.update(self._over_in)

        # Load columns and initialise matching index
        self.all_columns = self._load_columns()
        self._build_column_index()

        self._init_state_from_cfg(cfg)
        self._build_ui()
        self._populate_from_state()
        self._update_required_styles()   # initial validation styling

        # If we auto-guessed a non-year time column, inform the user once.
        if getattr(self, "_time_warning_message", ""):
            QMessageBox.information(
                self,
                "Time column check",
                self._time_warning_message,
            )

    def _load_columns(self) -> list[str]:
        # Prefer in-memory DataFrame if provided
        if getattr(self, "_df", None) is not None:
            return [str(c) for c in self._df.columns]

        if self._csv_path is None:
            return []
        try:
            df = pd.read_csv(self._csv_path, nrows=5)
        except Exception:
            return []
        return [str(c) for c in df.columns]

    def _normalise_name(self, name: str) -> str:
        """
        Normalise a feature/column name for fuzzy matching:

        - lowercase,
        - convert underscores / hyphens / punctuation to spaces,
        - collapse multiple spaces.
        """
        s = str(name).strip().lower()
        # Replace non-alphanumeric with spaces
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = " ".join(s.split())
        return s

    def _build_column_index(self) -> None:
        """
        Build a normalised index over available dataset columns for
        case-insensitive + fuzzy matching.
        """
        self._norm_to_originals: Dict[str, List[str]] = {}
        for col in self.all_columns or []:
            norm = self._normalise_name(col)
            self._norm_to_originals.setdefault(norm, []).append(col)
        self._norm_keys: List[str] = list(self._norm_to_originals.keys())

    def _auto_map_single(self, default_name: str) -> str:
        """
        Map a single default feature name to a dataset column name.

        Priority:
        1. Exact normalised match.
        2. Fuzzy match on normalised names (difflib).

        Returns the *dataset* column name if found, otherwise an empty
        string (meaning "no match").
        """
        if not default_name or not self.all_columns:
            return ""

        norm = self._normalise_name(default_name)

        # Exact normalised match
        if norm in self._norm_to_originals:
            return self._norm_to_originals[norm][0]

        # Fuzzy match (handles things like "soil_thicknes" vs
        # "Soil thickness")
        if not self._norm_keys:
            return ""

        best = difflib.get_close_matches(
            norm, self._norm_keys, n=1, cutoff=0.75
        )
        if best:
            return self._norm_to_originals[best[0]][0]

        return ""

    def _auto_map_feature_list(self, defaults: List[str]) -> List[str]:
        """
        Map a list of default feature names to dataset columns.

        Uses `_auto_map_single` per name and returns only the features
        for which a dataset column could be found (no duplicates).
        """
        if not defaults:
            return []

        # If we have no dataset columns, keep the original list so the
        # user at least sees something.
        if not self.all_columns:
            return defaults

        mapped: List[str] = []
        used: set[str] = set()
        for name in defaults:
            candidate = self._auto_map_single(name)
            if candidate and candidate not in used:
                mapped.append(candidate)
                used.add(candidate)
        return mapped

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    def _flatten_feats(self, feats: Any) -> List[str]:
        out: List[str] = []
        if not feats:
            return out
        for item in feats:
            if isinstance(item, (list, tuple)):
                if item:
                    out.append(str(item[0]))
            else:
                out.append(str(item))
        return out

    def _init_state_from_cfg(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        # --- Registries from config (raw) -----------------------------
        # 1) raw registries from config (canonical names / tuples)
        raw_num_reg = (
            cfg.get("OPTIONAL_NUMERIC_FEATURES_REGISTRY")
            or cfg.get("OPTIONAL_NUMERIC_FEATURES")
            or []
        )
        raw_cat_reg = (
            cfg.get("OPTIONAL_CATEGORICAL_FEATURES_REGISTRY")
            or cfg.get("OPTIONAL_CATEGORICAL_FEATURES")
            or []
        )
        raw_norm_reg = cfg.get("ALREADY_NORMALIZED_FEATURES", []) or []

        # 2) dataset-filtered registries (for UI + overrides)
        self.opt_numeric_registry = self._filter_registry_to_dataset(
            raw_num_reg
        )
        self.opt_categ_registry = self._filter_registry_to_dataset(
            raw_cat_reg
        )

        # For normalized features we only want dataset columns; if
        # nothing matches, keep it empty so placeholder text is shown.
        norm_flat = self._flatten_feats(raw_norm_reg)
        self.already_norm_feats = self._auto_map_feature_list(norm_flat)

        # 3) If registries are still empty but we have a full DataFrame,
        #    infer sensible defaults from dtypes / cardinalities.
        if not self.opt_numeric_registry and self._df is not None:
            self.opt_numeric_registry = self._infer_numeric_registry_from_df()

        if not self.opt_categ_registry and self._df is not None:
            self.opt_categ_registry = self._infer_categorical_registry_from_df()

        # --- Main feature selections (for the PINN inputs) ------------
        # Base (canonical) feature names -> resolved dataset columns
        # opt_numeric_feats = self._flatten_feats(
        #     cfg.get("OPTIONAL_NUMERIC_FEATURES", []),
        # )
        # opt_static_feats = self._flatten_feats(
        #     cfg.get("OPTIONAL_CATEGORICAL_FEATURES", []),
        # )
        future_feats = [
            str(x) for x in cfg.get("FUTURE_DRIVER_FEATURES", [])
        ]
        
        # These are the explicit driver lists; they can be empty.
        dyn_drivers = [
            str(x) for x in cfg.get("DYNAMIC_DRIVER_FEATURES", [])
        ]
        stat_drivers = [
            str(x) for x in cfg.get("STATIC_DRIVER_FEATURES", [])
        ]
        # already-normalised registry (can also contain tuples)
        normalized_feats = self._flatten_feats(
            cfg.get("ALREADY_NORMALIZED_FEATURES", []),
        )
        
        # # Smart auto-mapping to *dataset* column names
        # self.opt_numeric_feats = self._auto_map_feature_list(opt_numeric_feats)
        # self.opt_static_feats = self._auto_map_feature_list(opt_static_feats)
        self.future_feats = self._auto_map_feature_list(future_feats)
        # Smart auto-mapping to *dataset* column names
        self.numeric_feats = self._auto_map_feature_list(dyn_drivers)
        self.static_feats = self._auto_map_feature_list(stat_drivers)
        
        self.normalized_feats = self._auto_map_feature_list(
            normalized_feats
        )

        # --- Core spatio-temporal / target columns --------------------
        # Defaults from config or sensible fallbacks
        time_default = cfg.get("TIME_COL", "year")
        lon_default = cfg.get("LON_COL", "longitude")
        lat_default = cfg.get("LAT_COL", "latitude")
        subs_default = cfg.get("SUBSIDENCE_COL", "subsidence")
        gwl_default = cfg.get("GWL_COL", "GWL_depth_bgs_z")

        self.time_col = self._auto_map_single(str(time_default))
        self.lon_col = self._auto_map_single(str(lon_default))
        self.lat_col = self._auto_map_single(str(lat_default))
        self.subs_col = self._auto_map_single(str(subs_default))
        self.gwl_col = self._auto_map_single(str(gwl_default))

        # If no explicit "year" column could be mapped but we detect a
        # time-like column (date/month/week/...), pre-select it and
        # remember a gentle warning to show once.
        self._time_warning_message = ""
        if not self.time_col and self.all_columns:
            for col in self.all_columns:
                norm = self._normalise_name(col)
                if re.search(r"\b(date|month|week|time)\b", norm):
                    self.time_col = col
                    self._time_warning_message = (
                        "No column named 'year' could be mapped from the "
                        "configuration.\n\n"
                        f"Column '{col}' looks time-like and has been "
                        "pre-selected as the time index.\n\n"
                        "GeoPrior v3 currently expects an annual time axis "
                        "(one row per year). If your dataset is daily or "
                        "monthly, please derive a 'year' column or "
                        "aggregate the data before running Stage-1. "
                        "Future versions will support finer time steps."
                    )
                    break

        # --- Censoring / H-field --------------------------------------
        censor_block = cfg.get("censoring", {}) or {}
        specs = cfg.get(
            "CENSORING_SPECS",
            censor_block.get("specs", []),
        )
        first = specs[0] if specs else {}

        self._spec_template = dict(first)

        h_col = cfg.get("H_FIELD_COL_NAME")
        if not h_col:
            h_col = first.get("col", "")

        # Map H-field column via the same smart logic
        mapped_h = self._auto_map_single(str(h_col or ""))
        self.h_col = mapped_h or ""

        self.censor_cap = float(first.get("cap", 30.0))
        self.censor_dir = str(first.get("direction", "right"))
        self.censor_mode = str(first.get("eff_mode", "clip"))
        self.censor_flag_thr = float(
            first.get("flag_threshold", 0.5),
        )

        self.use_effective_h = bool(
            cfg.get(
                "USE_EFFECTIVE_H_FIELD",
                censor_block.get(
                    "use_effective_h_field",
                    True,
                ),
            )
        )
        self.flags_as_dynamic = bool(
            cfg.get(
                "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC",
                censor_block.get(
                    "flags_as_dynamic",
                    True,
                ),
            )
        )


    def _show_help_dialog(self) -> None:
        dlg = FeatureConfigHelpDialog(self)
        dlg.exec_()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("Feature configuration")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        # Small header / hint + help button
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        header = QLabel(
            "Select which columns act as dynamic drivers, "
            "static covariates, future drivers and the H-field "
            "used by the PINN."
        )
        header.setWordWrap(True)
        header_row.addWidget(header, 1)

        btn_help = QPushButton("?")
        btn_help.setToolTip("Explain all feature options…")
        btn_help.setFixedSize(22, 22)
        btn_help.setFlat(True)
        btn_help.setCursor(Qt.PointingHandCursor)
        btn_help.setStyleSheet(
            "QPushButton {"
            "  border-radius: 11px;"
            "  font-weight: 700;"
            "}"
        )
        btn_help.clicked.connect(self._show_help_dialog)
        header_row.addWidget(btn_help, 0, Qt.AlignTop)

        layout.addLayout(header_row)


        # --- Available columns box ------------------------------------
        lbl_cols = QLabel("Available columns in dataset:")
        lbl_cols.setStyleSheet("font-weight: 600;")
        layout.addWidget(lbl_cols)

        # tiny visual cue under available columns
        hint_cols = QLabel(
            "<i>Tip: double-click a column to add it to the field that "
            "currently has focus (drivers, registries or H-field).</i>"
        )
        hint_cols.setWordWrap(True)
        hint_cols.setStyleSheet("color: #6b7280; font-size: 11px;")
        layout.addWidget(hint_cols)

        self.txt_cols = ColumnsPlainTextEdit()
        
        self.txt_cols.setReadOnly(True)
        self.txt_cols.setMaximumHeight(90)
        self.txt_cols.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #faf5ff;"
            "  border-radius: 6px;"
            "}"
        )
        layout.addWidget(self.txt_cols)
        


        # --- Two-column main area -----------------------------------------
        two_col = QHBoxLayout()
        two_col.setSpacing(10)
        layout.addLayout(two_col)
        
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        two_col.addLayout(left_col, 1)
        
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        two_col.addLayout(right_col, 1)
        

        # --- Core spatio-temporal / target columns --------------------
        core_box = QGroupBox("Core spatio-temporal & target columns")
        core_box.setStyleSheet(
            "QGroupBox {"
            "  border: 1px solid #e5e7eb;"
            "  border-radius: 6px;"
            "  margin-top: 6px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 8px;"
            "  padding: 0 2px 0 2px;"
            "}"
        )
        core_layout = QGridLayout(core_box)
        core_layout.setColumnStretch(0, 0)
        core_layout.setColumnStretch(1, 1)
        core_layout.setHorizontalSpacing(8)
        core_layout.setVerticalSpacing(4)

        def _make_core_combo() -> QComboBox:
            combo = QComboBox()
            combo.addItem("— Select column —", "")
            for col in self.all_columns:
                combo.addItem(col, col)
            combo.setEditable(False)
            combo.setMinimumWidth(180)
            return combo

        row = 0

        lbl_time = QLabel("Time column (year-based, required)")
        lbl_time.setStyleSheet("font-weight: 600;")
        core_layout.addWidget(lbl_time, row, 0)
        self.cmb_time = _make_core_combo()
        core_layout.addWidget(self.cmb_time, row, 1)

        row += 1
        lbl_lon = QLabel("Longitude column (required)")
        core_layout.addWidget(lbl_lon, row, 0)
        self.cmb_lon = _make_core_combo()
        core_layout.addWidget(self.cmb_lon, row, 1)

        row += 1
        lbl_lat = QLabel("Latitude column (required)")
        core_layout.addWidget(lbl_lat, row, 0)
        self.cmb_lat = _make_core_combo()
        core_layout.addWidget(self.cmb_lat, row, 1)

        row += 1
        lbl_subs = QLabel("Subsidence column (target, required)")
        core_layout.addWidget(lbl_subs, row, 0)
        self.cmb_subs = _make_core_combo()
        core_layout.addWidget(self.cmb_subs, row, 1)

        row += 1
        lbl_gwl = QLabel("Groundwater level column (required)")
        core_layout.addWidget(lbl_gwl, row, 0)
        self.cmb_gwl = _make_core_combo()
        core_layout.addWidget(self.cmb_gwl, row, 1)

        hint_core = QLabel(
            "<i>These columns define the time axis, spatial coordinates "
            "and targets used by GeoPrior. They are mandatory and saved "
            "per city.</i>"
        )
        hint_core.setWordWrap(True)
        core_layout.addWidget(hint_core, row + 1, 0, 1, 2)

        left_col.addWidget(core_box)

        # --- Feature groups -------------------------------------------

        drivers_box = QGroupBox("Driver features")
        drivers_box.setStyleSheet(
            "QGroupBox {"
            "  border: 1px solid #e5e7eb;"
            "  border-radius: 6px;"
            "  margin-top: 6px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 8px;"
            "  padding: 0 2px 0 2px;"
            "}"
        )
        grid_feat = QGridLayout(drivers_box)
        grid_feat.setColumnStretch(0, 0)
        grid_feat.setColumnStretch(1, 1)
        grid_feat.setHorizontalSpacing(10)
        grid_feat.setVerticalSpacing(6)
        
        # PINN-required: dynamic features
        # Dynamic features (pure drivers)
        lbl_dyn = QLabel("Dynamic features")
        lbl_dyn.setStyleSheet("font-weight: 600;")
        grid_feat.addWidget(lbl_dyn, 0, 0)
        
        self.edit_dynamic = QLineEdit()
        self.edit_dynamic.setPlaceholderText(
            "Comma-separated columns, e.g. rainfall_mm, urban_load_global"
        )
        grid_feat.addWidget(self.edit_dynamic, 0, 1)
        
        # New hint under dynamic features
        hint_dyn = QLabel(
            "<i>Optional: leave empty to let GeoPrior auto-detect "
            "dynamic drivers from the dataset.</i>"
        )
        hint_dyn.setWordWrap(True)
        hint_dyn.setStyleSheet("color: #6b7280; font-size: 11px;")
        grid_feat.addWidget(hint_dyn, 1, 0, 1, 2)

        # Optional static & future drivers
        lbl_static = QLabel("Static features")
        grid_feat.addWidget(lbl_static, 2, 0)
        self.edit_static = QLineEdit()
        self.edit_static.setPlaceholderText(
            "Optional, e.g. lithology, lithology_class"
        )
        grid_feat.addWidget(self.edit_static, 2, 1)

        lbl_future = QLabel("Future drivers")
        grid_feat.addWidget(lbl_future, 3, 0)
        self.edit_future = QLineEdit()
        self.edit_future.setPlaceholderText(
            "Optional driver features with future values"
        )
        grid_feat.addWidget(self.edit_future, 3, 1)

        right_col.addWidget(drivers_box)
        
        # --- Additional feature registry -------------------------
        gb_reg = QGroupBox("Additional feature registry")
        reg_layout = QVBoxLayout(gb_reg)
        reg_layout.setContentsMargins(8, 6, 8, 6)
        reg_layout.setSpacing(4)

        hint_reg = QLabel(
            "<i>Entries can be single names or tuples of candidates, "
            "e.g. (rainfall_mm, rainfall, rain_mm, precip_mm); "
            "GeoPrior uses the first name that exists in your dataset. "
            "If you leave a registry empty, GeoPrior will rely on its "
            "automatic numeric/categorical detection.</i>"
        )
        hint_reg.setWordWrap(True)
        reg_layout.addWidget(hint_reg)

        grid_reg = QGridLayout()
        grid_reg.setColumnStretch(0, 0)
        grid_reg.setColumnStretch(1, 1)
        grid_reg.setHorizontalSpacing(8)
        grid_reg.setVerticalSpacing(4)
        reg_layout.addLayout(grid_reg)

        # Row 0: optional numeric registry
        lbl_opt_num = QLabel("Optional numeric features")
        grid_reg.addWidget(lbl_opt_num, 0, 0)
        self.edit_opt_num = QLineEdit()
        self.edit_opt_num.setPlaceholderText(
            "Groups separated by ';', e.g. "
            "rainfall_mm, rainfall, rain_mm; "
            "urban_load_global, normalized_density"
        )
        grid_reg.addWidget(self.edit_opt_num, 0, 1)

        # Row 1: optional categorical registry
        lbl_opt_cat = QLabel("Optional categorical features")
        grid_reg.addWidget(lbl_opt_cat, 1, 0)
        self.edit_opt_cat = QLineEdit()
        self.edit_opt_cat.setPlaceholderText(
            "Columns to force as categorical / one-hot encoded"
        )
        grid_reg.addWidget(self.edit_opt_cat, 1, 1)

        # Row 2: already-normalized features
        lbl_norm = QLabel("Already-normalized features")
        grid_reg.addWidget(lbl_norm, 2, 0)
        self.edit_norm = QLineEdit()
        self.edit_norm.setPlaceholderText(
            "Columns already scaled (e.g. in [0, 1]) to skip re-scaling"
        )
        grid_reg.addWidget(self.edit_norm, 2, 1)

        right_col.addWidget(gb_reg)

        # --- Censoring / H-field group --------------------------------
        censor_box = QGroupBox("Censoring (H-field)")
        censor_box.setStyleSheet(
            "QGroupBox {"
            "  border: 1px solid #e5e7eb;"
            "  border-radius: 6px;"
            "  margin-top: 6px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 8px;"
            "  padding: 0 2px 0 2px;"
            "}"
        )
        grid_c = QGridLayout(censor_box)
        grid_c.setHorizontalSpacing(8)
        grid_c.setVerticalSpacing(6)
        
        # Row 0: H-field (required) spanning all columns
        lbl_h = QLabel("H-field column (required)")
        lbl_h.setStyleSheet("font-weight: 600;")
        grid_c.addWidget(lbl_h, 0, 0)
        self.edit_h_col = QLineEdit()
        self.edit_h_col.setPlaceholderText("e.g. soil_thickness")
        grid_c.addWidget(self.edit_h_col, 0, 1, 1, 3)
        
        # Row 1: Cap | Direction
        grid_c.addWidget(QLabel("Cap"), 1, 0)
        self.spin_cap = QDoubleSpinBox()
        self.spin_cap.setDecimals(4)
        self.spin_cap.setRange(0.0, 1e6)
        grid_c.addWidget(self.spin_cap, 1, 1)

        grid_c.addWidget(QLabel("Direction"), 1, 2)
        self.cmb_dir = QComboBox()
        self.cmb_dir.addItems(["right", "left"])
        grid_c.addWidget(self.cmb_dir, 1, 3)

        # Row 2: Effective mode | Flag threshold
        grid_c.addWidget(QLabel("Effective mode"), 2, 0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(
            ["clip", "cap_minus_eps", "nan_if_censored"],
        )
        grid_c.addWidget(self.cmb_mode, 2, 1)

        grid_c.addWidget(QLabel("Flag threshold"), 2, 2)
        self.spin_flag_thr = QDoubleSpinBox()
        self.spin_flag_thr.setDecimals(3)
        self.spin_flag_thr.setRange(0.0, 1.0)
        self.spin_flag_thr.setSingleStep(0.05)
        grid_c.addWidget(self.spin_flag_thr, 2, 3)

        # Row 3: checkboxes across full width
        self.chk_use_eff = QCheckBox("Use effective H-field")
        grid_c.addWidget(self.chk_use_eff, 3, 0, 1, 4)

        self.chk_flags_dyn = QCheckBox(
            "Include censor flags as dynamic drivers",
        )
        grid_c.addWidget(self.chk_flags_dyn, 4, 0, 1, 4)
        
        left_col.addWidget(censor_box)
        
        # --- Buttons ---------------------------------------------------
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # ------------------------------------------------------------------
        # Track which field was last focused (for double-click insert)
        # ------------------------------------------------------------------
        self._focus_map = {
            self.edit_opt_num: "registry_num",
            self.edit_opt_cat: "registry_cat",
            self.edit_norm: "csv",
            self.edit_dynamic: "csv",
            self.edit_static: "csv",
            self.edit_future: "csv",
            self.edit_h_col: "single",
        }
        for w in self._focus_map.keys():
            w.installEventFilter(self)

        self._last_target = None   # type: Optional[QLineEdit]
        self._last_mode = None     # type: Optional[str]

        # connect double-click signal
        self.txt_cols.columnActivated.connect(self._on_column_activated)
                
        # Slightly larger dialog footprint
        self.resize(850, 420)

        # Validate/colour required fields on the fly
        # self.edit_dynamic.textChanged.connect(self._update_required_styles)
        self.edit_h_col.textChanged.connect(self._update_required_styles)
        self.cmb_time.currentIndexChanged.connect(self._update_required_styles)
        self.cmb_lon.currentIndexChanged.connect(self._update_required_styles)
        self.cmb_lat.currentIndexChanged.connect(self._update_required_styles)
        self.cmb_subs.currentIndexChanged.connect(self._update_required_styles)
        self.cmb_gwl.currentIndexChanged.connect(self._update_required_styles)

    def eventFilter(self, obj, event):
        """
        Remember which editable field was last focused so that
        double-clicking a column knows where to insert it.
        """
        if event.type() == QEvent.FocusIn:
            mode = getattr(self, "_focus_map", {}).get(obj)
            if mode is not None:
                self._last_target = obj
                self._last_mode = mode
        return super().eventFilter(obj, event)

    def _on_column_activated(self, col: str) -> None:
        """
        Called when the user double-clicks a column name in the
        'Available columns' box.

        Behaviour:
        - Optional numeric registry: add as new group (';')
        - Dynamic / static / future / normalized / optional cat:
          append as comma-separated list
        - H-field: replace
        """
        col = col.strip()
        if not col:
            return

        # Use the last field that had focus before the columns box
        target = getattr(self, "_last_target", None)
        mode = getattr(self, "_last_mode", None)

        if target is None or mode is None:
            # No relevant field has ever been focused → ignore
            return

        current = target.text().strip()

        if mode == "single":
            # H-field: just replace
            new_text = col

        elif mode == "csv":
            # Comma-separated fields, avoid duplicates
            if not current:
                new_text = col
            else:
                parts = [p.strip() for p in current.split(",") if p.strip()]
                if col in parts:
                    return
                new_text = current + ", " + col

        elif mode == "registry_num":
            # Optional numeric registry: groups separated by ';'
            # Each double-click adds a new one-column group.
            if not current:
                new_text = col
            else:
                groups = [g.strip() for g in current.split(";") if g.strip()]
                if col in groups:
                    return
                new_text = current.rstrip() + "; " + col

        elif mode == "registry_cat":
            # Optional categorical: treat like csv
            if not current:
                new_text = col
            else:
                parts = [p.strip() for p in current.split(",") if p.strip()]
                if col in parts:
                    return
                new_text = current + ", " + col
        else:
            return

        target.setText(new_text)
        self._update_required_styles()


    def _populate_from_state(self) -> None:
        if self.all_columns:
            cols_str = ", ".join(self.all_columns)
            self.txt_cols.setPlainText(cols_str)

        # Main driver selections
        self.edit_dynamic.setText(", ".join(self.numeric_feats))
        self.edit_static.setText(", ".join(self.static_feats))
        self.edit_future.setText(", ".join(self.future_feats))

        # Registries
        self.edit_opt_num.setText(
            self._registry_to_text(self.opt_numeric_registry))
        self.edit_opt_cat.setText(
            self._registry_to_text(self.opt_categ_registry))
        self.edit_norm.setText(
            ", ".join(self.already_norm_feats))

        # Core columns
        self._set_core_combo_value(self.cmb_time, self.time_col)
        self._set_core_combo_value(self.cmb_lon, self.lon_col)
        self._set_core_combo_value(self.cmb_lat, self.lat_col)
        self._set_core_combo_value(self.cmb_subs, self.subs_col)
        self._set_core_combo_value(self.cmb_gwl, self.gwl_col)

        # H-field + censoring
        self.edit_h_col.setText(self.h_col)
        self.spin_cap.setValue(self.censor_cap)
        self.spin_flag_thr.setValue(self.censor_flag_thr)

        idx = self.cmb_dir.findText(self.censor_dir)
        if idx < 0:
            idx = 0
        self.cmb_dir.setCurrentIndex(idx)

        idx = self.cmb_mode.findText(self.censor_mode)
        if idx < 0:
            idx = 0
        self.cmb_mode.setCurrentIndex(idx)

        self.chk_use_eff.setChecked(self.use_effective_h)
        self.chk_flags_dyn.setChecked(self.flags_as_dynamic)


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_core_combo_value(self, combo: QComboBox, value: str) -> None:
        """Select `value` in a core-column combo, if present."""
        if combo is None:
            return
        value = (value or "").strip()
        if not value:
            combo.setCurrentIndex(0)
            return
        idx = combo.findData(value)
        if idx < 0:
            idx = combo.findText(value)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _parse_list(self, text: str) -> list[str]:
        parts = [p.strip() for p in text.split(",")]
        return [p for p in parts if p]

    def _update_required_styles(self) -> None:
        """
        Highlight PINN-required fields in red when missing:

        - Dynamic features
        - Core spatio-temporal / target columns
        - H-field column
        """
        # dyn_missing = not self._parse_list(self.edit_dynamic.text())
        h_missing = not self.edit_h_col.text().strip()

        time_missing = not self.cmb_time.currentText().strip()
        lon_missing = not self.cmb_lon.currentText().strip()
        lat_missing = not self.cmb_lat.currentText().strip()
        subs_missing = not self.cmb_subs.currentText().strip()
        gwl_missing = not self.cmb_gwl.currentText().strip()

        err_line_style = (
            "QLineEdit {"
            "  border: 1px solid #e11d48;"
            "  background-color: #fff1f2;"
            "}"
        )
        err_combo_style = (
            "QComboBox {"
            "  border: 1px solid #e11d48;"
            "  background-color: #fff1f2;"
            "}"
        )
        ok_style = ""  # let global stylesheet apply

        # self.edit_dynamic.setStyleSheet(
        #     err_line_style if dyn_missing else ok_style
        # )
        self.edit_h_col.setStyleSheet(
            err_line_style if h_missing else ok_style
        )

        self.cmb_time.setStyleSheet(
            err_combo_style if time_missing else ok_style
        )
        self.cmb_lon.setStyleSheet(
            err_combo_style if lon_missing else ok_style
        )
        self.cmb_lat.setStyleSheet(
            err_combo_style if lat_missing else ok_style
        )
        self.cmb_subs.setStyleSheet(
            err_combo_style if subs_missing else ok_style
        )
        self.cmb_gwl.setStyleSheet(
            err_combo_style if gwl_missing else ok_style
        )

    def _registry_to_text(self, reg: Any) -> str:
        """
        Turn a registry list into a compact text representation.

        Format:
            group1; group2; ...

        where each group is either:
            name
        or:
            name1, name2, name3
        """
        if not reg:
            return ""
        groups: list[str] = []
        for item in reg:
            if isinstance(item, (list, tuple)):
                names = [str(x).strip() for x in item if str(x).strip()]
                if not names:
                    continue
                groups.append(", ".join(names))
            else:
                name = str(item).strip()
                if name:
                    groups.append(name)
        return " ; ".join(groups)

    def _parse_registry(self, text: str) -> list[Any]:
        """
        Parse a registry from text.

        Syntax:
            rainfall_mm, rainfall, rain_mm; urban_load_global, urban_load

        - groups are separated by ';'
        - names inside a group are separated by ','

        A group with one name becomes a plain string, a group with more
        names becomes a list of strings.
        """
        if not text or not text.strip():
            return []
        out: list[Any] = []
        groups = [g.strip() for g in text.split(";")]
        for g in groups:
            if not g:
                continue
            names = [n.strip() for n in g.split(",") if n.strip()]
            if not names:
                continue
            if len(names) == 1:
                out.append(names[0])
            else:
                out.append(names)
        return out
    
    def _filter_registry_to_dataset(self, reg: list[Any]) -> list[Any]:
        """
        Keep only entries whose candidates exist in the dataset.

        Each input item can be:
          - a string   -> single candidate name
          - a list/tuple -> multiple candidate names

        We use `_auto_map_single` to resolve each candidate to a
        *dataset* column; if none of the candidates map, the group is
        dropped. The returned registry only contains dataset-safe names.
        """
        if not reg or not self.all_columns:
            return []

        filtered: list[Any] = []
        used: set[str] = set()

        for item in reg:
            # Multi-candidate group
            if isinstance(item, (list, tuple)):
                mapped_group: list[str] = []
                for cand in item:
                    col = self._auto_map_single(str(cand))
                    if col and col not in used:
                        mapped_group.append(col)
                        used.add(col)
                if mapped_group:
                    # if only one survives, store as plain string
                    if len(mapped_group) == 1:
                        filtered.append(mapped_group[0])
                    else:
                        filtered.append(mapped_group)
            else:
                # Single candidate
                col = self._auto_map_single(str(item))
                if col and col not in used:
                    filtered.append(col)
                    used.add(col)

        return filtered
    
    def _infer_numeric_registry_from_df(self) -> list[Any]:
        """
        Heuristic fallback: infer optional numeric registry from `self._df`.

        Each column becomes a single-name group (string). We:
        - keep only numeric dtypes,
        - drop almost-constant columns,
        - drop very low cardinality numeric (likely flags / small enums),
        - drop obvious time-like columns (year, date, month, etc.).
        """
        df = getattr(self, "_df", None)
        if df is None or df.empty:
            return []

        num_df = df.select_dtypes(include=["number"])
        if num_df.empty:
            return []

        n_rows = len(num_df)
        out: list[str] = []

        for col in num_df.columns:
            s = num_df[col]
            nunique = s.nunique(dropna=True)

            # Skip constants
            if nunique <= 1:
                continue

            # Treat low-cardinality numeric as "more categorical" and
            # leave it to the categorical inference instead.
            low_cat_threshold = min(15, max(int(0.05 * n_rows), 5))
            if nunique <= low_cat_threshold:
                continue

            # Skip obvious time-like columns
            norm_name = self._normalise_name(col)
            if re.search(r"\b(year|date|time|month|day|hour)\b", norm_name):
                continue

            out.append(col)

        return out

    def _infer_categorical_registry_from_df(self) -> list[Any]:
        """
        Heuristic fallback: infer optional categorical registry from `self._df`.

        Each column becomes a single-name group (string). We:
        - prefer object / category / bool dtypes,
        - also allow low-cardinality numeric columns,
        - ignore very high-cardinality columns (likely IDs).
        """
        df = getattr(self, "_df", None)
        if df is None or df.empty:
            return []

        max_card = 32   # upper bound for "nice" categorical features
        min_card = 2    # ignore constants / single-category

        out: list[str] = []

        for col in df.columns:
            s = df[col]
            nunique = s.nunique(dropna=True)
            if nunique < min_card or nunique > max_card:
                continue

            # Text / explicit categorical / bool -> categorical
            if (
                s.dtype == "object"
                or pd.api.types.is_categorical_dtype(s)
                or pd.api.types.is_bool_dtype(s)
            ):
                out.append(col)
                continue

            # Low-cardinality numeric (e.g. encoded classes) – also treat
            # as categorical.
            if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
                out.append(col)

        return out
    
    def get_overrides(self) -> Dict[str, Any]:
        """Return config overrides from the dialog."""
        dyn = self._parse_list(self.edit_dynamic.text())
        stat = self._parse_list(self.edit_static.text())
        fut = self._parse_list(self.edit_future.text())

        time_col = self.cmb_time.currentText().strip()
        lon_col = self.cmb_lon.currentText().strip()
        lat_col = self.cmb_lat.currentText().strip()
        subs_col = self.cmb_subs.currentText().strip()
        gwl_col = self.cmb_gwl.currentText().strip()

        col = self.edit_h_col.text().strip() or self.h_col


        spec = dict(self._spec_template)
        spec["col"] = col
        spec["cap"] = float(self.spin_cap.value())
        spec["direction"] = self.cmb_dir.currentText()
        spec["eff_mode"] = self.cmb_mode.currentText()
        spec["flag_threshold"] = float(self.spin_flag_thr.value())

        use_eff = self.chk_use_eff.isChecked()
        flags_dyn = self.chk_flags_dyn.isChecked()

        # NEW: registries
        opt_num_reg = self._parse_registry(self.edit_opt_num.text())
        opt_cat_reg = self._parse_registry(self.edit_opt_cat.text())
        norm_feats = self._parse_list(self.edit_norm.text())

        overrides: Dict[str, Any] = {}

        # # Existing keys: selections used by the GUI / pipeline today
        
        # Pure driver lists (can be empty; auto-detection handles it)
        overrides["DYNAMIC_DRIVER_FEATURES"] = dyn
        overrides["STATIC_DRIVER_FEATURES"] = stat
        overrides["FUTURE_DRIVER_FEATURES"] = fut
        
        overrides["TIME_COL"] = time_col
        overrides["LON_COL"] = lon_col
        overrides["LAT_COL"] = lat_col
        overrides["SUBSIDENCE_COL"] = subs_col
        overrides["GWL_COL"] = gwl_col

        overrides["H_FIELD_COL_NAME"] = col
        overrides["CENSORING_SPECS"] = [spec]
        overrides["censoring"] = {
            "specs": [spec],
            "use_effective_h_field": use_eff,
            "flags_as_dynamic": flags_dyn,
        }
        overrides["USE_EFFECTIVE_H_FIELD"] = use_eff
        overrides["INCLUDE_CENSOR_FLAGS_AS_DYNAMIC"] = flags_dyn

        # explicit registries; the Stage-1 logic can read these
        # if present to override automatic detection.
        if opt_num_reg:
            overrides["OPTIONAL_NUMERIC_FEATURES_REGISTRY"] = opt_num_reg
        if opt_cat_reg:
            overrides["OPTIONAL_CATEGORICAL_FEATURES_REGISTRY"] = opt_cat_reg
        if norm_feats:
            overrides["ALREADY_NORMALIZED_FEATURES"] = norm_feats

        return overrides

    # ------------------------------------------------------------------
    # Validation on OK
    # ------------------------------------------------------------------

    def accept(self) -> None:  # type: ignore[override]
        """
        Ensure PINN-required fields are filled:

        - Core spatio-temporal / target columns
        - H-field column
        """
        # dyn = self._parse_list(self.edit_dynamic.text())
        hcol = self.edit_h_col.text().strip()

        time_col = self.cmb_time.currentText().strip()
        lon_col = self.cmb_lon.currentText().strip()
        lat_col = self.cmb_lat.currentText().strip()
        subs_col = self.cmb_subs.currentText().strip()
        gwl_col = self.cmb_gwl.currentText().strip()

        missing_core = [
            name
            for name, val in [
                ("Time column", time_col),
                ("Longitude column", lon_col),
                ("Latitude column", lat_col),
                ("Subsidence column", subs_col),
                ("Groundwater level column", gwl_col),
            ]
            if not val
        ]

        if not hcol or missing_core:
            msg = QMessageBox(self)
            msg.setWindowTitle("Required features missing")
            msg.setIcon(QMessageBox.Warning)
            msg.setTextFormat(Qt.RichText)

            core_html = "".join(
                f"<li><b>{name}</b></li>" for name in missing_core
            )

            msg.setText(
                "<p>For GeoPriorSubsNet the following fields are required:</p>"
                "<ul>"
                "<li><b>H-field column</b></li>"
                f"{core_html}"
                "</ul>"
                "<p>Please specify them before closing this dialog.</p>"
            )
            msg.exec_()

            self._update_required_styles()
            return

        super().accept()

class FeatureConfigHelpDialog(QDialog):
    """
    Read-only help dialog explaining feature configuration concepts.

    The content is rich-text (HTML) so we can use headings, italics and
    code-style examples.
    """

    def __init__(self, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Feature configuration – help")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        title = QLabel("How GeoPrior uses your features")
        title.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(title)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        browser.setMinimumSize(520, 420)

        help_html = """
        <h3>1. Overview</h3>
        <p>
          This dialog tells <b>GeoPriorSubsNet</b> how to interpret the
          columns in your dataset. The model distinguishes several types
          of inputs:
        </p>
        <ul>
          <li><b>Core spatio-temporal &amp; target columns</b> –
              time, longitude, latitude, subsidence and groundwater
              level;</li>
          <li><b>Dynamic features</b> – time-varying drivers;</li>
          <li><b>Static features</b> – fixed properties of each pixel
              or site;</li>
          <li><b>Future drivers</b> – drivers known or forecast in the
              future;</li>
          <li><b>H-field</b> – the physical field used by the PINN
              (e.g. soil thickness) with optional censoring.</li>
        </ul>
        <p>
          GeoPrior tries to auto-match common names
          (<code>soil_thickness</code> vs <code>Soil thickness</code>,
          <code>rainfall_mm</code> vs <code>rain_mm</code>, etc.) but
          you can always override them here. An additional
          <b>feature registry</b> lets you fine-tune how numeric /
          categorical features and already-normalized columns are
          interpreted.
        </p>

        <h3>2. Core spatio-temporal &amp; target columns</h3>
        <p>
          The <b>Core spatio-temporal &amp; target columns</b> group
          defines the minimal structure of your dataset:
        </p>
        <ul>
          <li><b>Time column</b> – the temporal index;</li>
          <li><b>Longitude column</b> – x-coordinate;</li>
          <li><b>Latitude column</b> – y-coordinate;</li>
          <li><b>Subsidence column</b> – the main target for the model;</li>
          <li><b>Groundwater level column</b> – the GWL target used
              jointly with subsidence.</li>
        </ul>
        <p>
          In the current GeoPrior v3 workflow, the time axis is assumed
          to be <b>year-based</b> (one row per pixel and year). The
          default configuration uses <code>year</code> as the time
          column, <code>longitude</code>/<code>latitude</code> for
          coordinates, <code>subsidence</code> for the target, and
          <code>GWL_depth_bgs_z</code> for groundwater level. When you
          open the dialog, GeoPrior will:
        </p>
        <ul>
          <li>
            First, try to map these default names directly to your
            dataset columns (case-insensitive, fuzzy matching).
          </li>
          <li>
            If no <code>year</code>-like column is found but a column
            looks time-like (e.g. <code>date</code>, <code>month</code>,
            <code>week</code>), it may be pre-selected and a short
            message will remind you that this version expects annual
            data.
          </li>
          <li>
            If a required core column cannot be matched, the combo box is
            left empty and highlighted in red until you choose a column.
          </li>
        </ul>
        <p>
          At the moment, if your dataset is daily or monthly you should
          either:
        </p>
        <ul>
          <li>
            derive a <code>year</code> column and aggregate to yearly
            values, or
          </li>
          <li>
            create a harmonised annual file (e.g. via pre-processing
            scripts) before running Stage-1.
          </li>
        </ul>
        <p>
          Future versions of GeoPrior will support sub-annual time
          steps. For now, think of the core group as the minimal
          contract between your dataset and the PINN: time (year),
          spatial coordinates and the two main targets.
        </p>

        <h3>3. Dynamic features (required)</h3>
        <p>
          Dynamic features are time-varying drivers such as
          <code>rainfall_mm</code> or <code>urban_load_global</code>.
          They are fed to the temporal encoder of the network and are
          crucial for learning how subsidence evolves in time.
        </p>
        <ul>
          <li>
            Use a comma-separated list, e.g.
            <code>rainfall_mm, urban_load_global</code>.
          </li>
          <li>
            GeoPrior will also include any censoring flags as additional
            dynamic channels when
            <i>"Include censor flags as dynamic drivers"</i> is checked.
          </li>
          <li>
            If this field is empty, the PINN cannot run (the dialog will
            highlight it in red and block <b>OK</b>).
          </li>
        </ul>

        <h3>4. Static features</h3>
        <p>
          Static features describe properties that do not change over
          time, such as:
        </p>
        <ul>
          <li><code>lithology</code>, <code>geology</code></li>
          <li><code>lithology_class</code> (categorical)</li>
        </ul>
        <p>
          They are one-hot encoded once and reused across all time
          steps. Static features help the model separate behaviour of
          different soil units, geomorphology classes, etc.
        </p>

        <h3>5. Future drivers</h3>
        <p>
          Future drivers are optional dynamic features for which
          <i>future</i> values are known or forecast. A common pattern
          is to reuse <code>rainfall_mm</code> here when you have
          rainfall forecasts:
        </p>
        <ul>
          <li>
            At training time, the model sees past and current values.
          </li>
          <li>
            At forecasting time, the model reads the
            <code>Future drivers</code> beyond the train end year to
            produce subsidence scenarios up to the chosen horizon.
          </li>
        </ul>

        <h3>6. H-field and censoring</h3>
        <p>
          The <b>H-field</b> column is the physical field used inside
          the physics-informed loss. In the subsidence application this
          is typically something like <code>soil_thickness</code> or a
          similar proxy.
        </p>
        <ul>
          <li><b>H-field column</b> – required. Example:
              <code>soil_thickness</code>.</li>
          <li><b>Cap</b> – values above this threshold are treated as
              censored measurements (e.g. drilling could not reach the
              true thickness).</li>
          <li><b>Direction</b> – whether large values are censored on
              the <i>right</i> tail or on the <i>left</i> tail.</li>
          <li><b>Effective mode</b> – how censored samples are turned
              into an effective H-field:
              <ul>
                <li><code>clip</code>: clamp at the cap;</li>
                <li><code>cap_minus_eps</code>: slightly below the cap;</li>
                <li><code>nan_if_censored</code>: missing, handled by
                    the physics loss.</li>
              </ul>
          </li>
          <li><b>Flag threshold</b> – probability above which a sample
              is considered "censored" and a binary flag is set.</li>
        </ul>
        <p>
          The checkboxes let you:
        </p>
        <ul>
          <li>
            <b>Use effective H-field</b> – enable the transformed H-field
            in the loss instead of the raw column.
          </li>
          <li>
            <b>Include censor flags as dynamic drivers</b> – append
            Boolean indicators so the network can learn different
            behaviour for censored vs. fully observed regions.
          </li>
        </ul>

        <h3>7. Additional feature registry</h3>
        <p>
          The <b>Additional feature registry</b> gives you more control
          over how GeoPrior treats specific columns. Entries can be
          single names or <i>tuples of candidates</i>, for example:
        </p>
        <pre style="background:#f5f5f5;padding:4px;">
  rainfall_mm, rainfall, rain_mm; urban_load_global, normalized_density
        </pre>
        <p>
          Internally this corresponds to groups like
          <code>(rainfall_mm, rainfall, rain_mm)</code> and
          <code>(urban_load_global, normalized_density)</code>. For each
          group GeoPrior selects the <b>first</b> name that exists in
          your dataset.
        </p>

        <h4>7.1 Optional numeric features</h4>
        <p>
          The <b>Optional numeric features</b> registry is a hint for
          numeric drivers that should be treated as potential dynamic
          inputs, even if their names differ between cities or studies.
          Typical examples:
        </p>
        <ul>
          <li><code>(rainfall_mm, rainfall, rain_mm, precip_mm)</code></li>
          <li><code>(urban_load_global, normalized_density, urban_load)</code></li>
        </ul>
        <p>
          If you leave this field empty, GeoPrior infers candidate
          numeric drivers from the dataset (based on dtypes and
          cardinality) and falls back to that automatic inference.
        </p>

        <h4>7.2 Optional categorical features</h4>
        <p>
          The <b>Optional categorical features</b> registry forces some
          columns to be treated as categorical / one-hot encoded.
          This is useful when:
        </p>
        <ul>
          <li>
            the same concept has different names
            (<code>lithology</code> vs <code>geology</code>), or
          </li>
          <li>
            classes are encoded as small integers, and you want them
            as categories rather than numeric magnitudes.
          </li>
        </ul>
        <p>
          Example groups:
        </p>
        <ul>
          <li><code>(lithology, geology)</code></li>
          <li><code>lithology_class</code></li>
        </ul>
        <p>
          If you leave this registry empty, GeoPrior infers categorical
          candidates from the dataset (object / category / bool dtypes
          and low-cardinality numeric columns).
        </p>

        <h4>7.3 Already-normalized features</h4>
        <p>
          Some numeric features may already be normalized, for example
          an index in the range [0, 1] or a z-score. Typical examples:
        </p>
        <ul>
          <li><code>urban_load_global</code></li>
          <li>
            any pre-scaled density or risk index (e.g.
            <code>normalized_urban_load_proxy</code>).
          </li>
        </ul>
        <p>
          List such columns under <b>Already-normalized features</b> so
          GeoPrior <i>skips</i> them when applying the global Min–Max
          scaler. This avoids double-normalisation and keeps your
          physically meaningful scaling.
        </p>
        <p>
          If nothing in this registry matches the dataset columns, the
          field simply remains empty and the global scaler will treat
          all drivers as unscaled.
        </p>

        <h4>7.4 Leaving registries empty</h4>
        <p>
          All registries are optional. When you leave them empty,
          GeoPrior:
        </p>
        <ul>
          <li>
            infers numeric vs categorical features from the DataFrame
            (dtypes + cardinality), and
          </li>
          <li>
            assumes no feature is already normalized unless specified.
          </li>
        </ul>

        <h3>8. Auto-matching behaviour</h3>
        <p>
          The dialog performs a case-insensitive, fuzzy match between
          the default configuration / registries and your dataset
          columns. For instance:
        </p>
        <ul>
          <li><code>soil_thickness</code> ⇔ <code>Soil thickness</code></li>
          <li><code>rainfall_mm</code> ⇔ <code>rain_mm</code></li>
        </ul>
        <p>
          If a match is found, the dataset name is used directly (you
          do <b>not</b> need to rename your file). If no match is
          found, the field stays empty and is highlighted in red when
          required.
        </p>

        <h3>9. Recommended workflow</h3>
        <ol>
          <li>Open your dataset and check the list of available columns.</li>
          <li>Verify that the <b>Core spatio-temporal &amp; target
              columns</b> are correctly populated (time, lon, lat,
              subsidence, GWL).</li>
          <li>Verify that <b>Dynamic features</b> and the
              <b>H-field column</b> are correctly populated.</li>
          <li>Add any static or future drivers that you want the model
              to use.</li>
          <li>
            Optionally adjust the <b>Additional feature registry</b>:
            clarify numeric vs categorical features and mark any
            already-normalized columns.
          </li>
          <li>
              Click <b>OK</b>. If any required field is missing, the
              dialog will tell you what to fix.
          </li>
        </ol>
        """

        browser.setHtml(help_html)
        layout.addWidget(browser)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        # btn_box.accepted.connect(self.accept)
        # Close on either Close or Esc
        layout.addWidget(btn_box)
