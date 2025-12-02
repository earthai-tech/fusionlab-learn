# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
)

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
        self._csv_path = Path(csv_path)
        self._base_cfg = base_cfg or {}
        self._over_in = current_overrides or {}
        self._df = df  
        
        cfg: Dict[str, Any] = dict(self._base_cfg)
        cfg.update(self._over_in)

        self._init_state_from_cfg(cfg)
        self._build_ui()
        self._populate_from_state()

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


    def _init_state_from_cfg(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        self.all_columns = self._load_columns()

        self.numeric_feats = self._flatten_feats(
            cfg.get("OPTIONAL_NUMERIC_FEATURES", []),
        )
        self.static_feats = self._flatten_feats(
            cfg.get("OPTIONAL_CATEGORICAL_FEATURES", []),
        )
        self.future_feats = [
            str(x) for x in cfg.get("FUTURE_DRIVER_FEATURES", [])
        ]

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
        self.h_col = str(h_col or "")

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

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("Feature configuration")
        self.setModal(True)

        layout = QVBoxLayout(self)

        lbl_cols = QLabel("Available columns in CSV:")
        layout.addWidget(lbl_cols)

        self.txt_cols = QPlainTextEdit()
        self.txt_cols.setReadOnly(True)
        self.txt_cols.setMaximumHeight(70)
        layout.addWidget(self.txt_cols)

        grid_feat = QGridLayout()
        layout.addLayout(grid_feat)

        grid_feat.addWidget(QLabel("Dynamic features"), 0, 0)
        self.edit_dynamic = QLineEdit()
        grid_feat.addWidget(self.edit_dynamic, 0, 1)

        grid_feat.addWidget(QLabel("Static features"), 1, 0)
        self.edit_static = QLineEdit()
        grid_feat.addWidget(self.edit_static, 1, 1)

        grid_feat.addWidget(QLabel("Future drivers"), 2, 0)
        self.edit_future = QLineEdit()
        grid_feat.addWidget(self.edit_future, 2, 1)

        # Censoring block
        lbl_censor = QLabel("Censoring (H-field)")
        lbl_censor.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_censor)

        grid_c = QGridLayout()
        layout.addLayout(grid_c)

        grid_c.addWidget(QLabel("H-field column"), 0, 0)
        self.edit_h_col = QLineEdit()
        grid_c.addWidget(self.edit_h_col, 0, 1)

        grid_c.addWidget(QLabel("Cap"), 1, 0)
        self.spin_cap = QDoubleSpinBox()
        self.spin_cap.setDecimals(4)
        self.spin_cap.setRange(0.0, 1e6)
        grid_c.addWidget(self.spin_cap, 1, 1)

        grid_c.addWidget(QLabel("Direction"), 2, 0)
        self.cmb_dir = QComboBox()
        self.cmb_dir.addItems(["right", "left"])
        grid_c.addWidget(self.cmb_dir, 2, 1)

        grid_c.addWidget(QLabel("Effective mode"), 3, 0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(
            ["clip", "cap_minus_eps", "nan_if_censored"],
        )
        grid_c.addWidget(self.cmb_mode, 3, 1)

        grid_c.addWidget(QLabel("Flag threshold"), 4, 0)
        self.spin_flag_thr = QDoubleSpinBox()
        self.spin_flag_thr.setDecimals(3)
        self.spin_flag_thr.setRange(0.0, 1.0)
        self.spin_flag_thr.setSingleStep(0.05)
        grid_c.addWidget(self.spin_flag_thr, 4, 1)

        self.chk_use_eff = QCheckBox("Use effective H-field")
        grid_c.addWidget(self.chk_use_eff, 5, 0, 1, 2)

        self.chk_flags_dyn = QCheckBox(
            "Include censor flags as dynamic drivers",
        )
        grid_c.addWidget(self.chk_flags_dyn, 6, 0, 1, 2)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _populate_from_state(self) -> None:
        if self.all_columns:
            cols_str = ", ".join(self.all_columns)
            self.txt_cols.setPlainText(cols_str)

        self.edit_dynamic.setText(
            ", ".join(self.numeric_feats),
        )
        self.edit_static.setText(
            ", ".join(self.static_feats),
        )
        self.edit_future.setText(
            ", ".join(self.future_feats),
        )

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
    def _parse_list(self, text: str) -> list[str]:
        parts = [p.strip() for p in text.split(",")]
        return [p for p in parts if p]

    def get_overrides(self) -> Dict[str, Any]:
        """Return config overrides from the dialog."""
        dyn = self._parse_list(self.edit_dynamic.text())
        stat = self._parse_list(self.edit_static.text())
        fut = self._parse_list(self.edit_future.text())

        col = self.edit_h_col.text().strip() or self.h_col

        spec = dict(self._spec_template)
        spec["col"] = col
        spec["cap"] = float(self.spin_cap.value())
        spec["direction"] = self.cmb_dir.currentText()
        spec["eff_mode"] = self.cmb_mode.currentText()
        spec["flag_threshold"] = float(
            self.spin_flag_thr.value(),
        )

        use_eff = self.chk_use_eff.isChecked()
        flags_dyn = self.chk_flags_dyn.isChecked()

        overrides: Dict[str, Any] = {}
        overrides["OPTIONAL_NUMERIC_FEATURES"] = dyn
        overrides["OPTIONAL_CATEGORICAL_FEATURES"] = stat
        overrides["FUTURE_DRIVER_FEATURES"] = fut

        overrides["H_FIELD_COL_NAME"] = col
        overrides["CENSORING_SPECS"] = [spec]
        overrides["censoring"] = {
            "specs": [spec],
            "use_effective_h_field": use_eff,
            "flags_as_dynamic": flags_dyn,
        }
        overrides["USE_EFFECTIVE_H_FIELD"] = use_eff
        overrides["INCLUDE_CENSOR_FLAGS_AS_DYNAMIC"] = flags_dyn

        return overrides
