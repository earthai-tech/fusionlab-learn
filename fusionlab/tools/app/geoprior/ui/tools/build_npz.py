# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Build custom NPZ files for GeoPriorSubsNet Stage-1.

from __future__ import annotations

import os
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QGroupBox,
    QSizePolicy,
)

from fusionlab.utils.generic_utils import ensure_directory_exists
# from .....utils.data_utils import nan_ops
# from ...utils.generic_utils import normalize_time_column, print_config_table
# from ...utils.sequence_utils import build_pinn_data_sequences


class BuildNPZTool(QWidget):
    """
    Build custom NPZ files for Stage-1, allowing the user to pass dataset,
    configure overrides, and save the results in a new directory.
    """

    def __init__(self, app_ctx: Optional[object] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._app_ctx = app_ctx
        self._results_root = self._guess_results_root()
        self._init_ui()

    # ------------------------------------------------------------------
    # Results root inference
    # ------------------------------------------------------------------
    def _guess_results_root(self) -> Path:
        ctx = self._app_ctx

        def _as_path(val: Any) -> Optional[Path]:
            if not val:
                return None
            try:
                return Path(str(val))
            except Exception:
                return None

        if ctx is not None:
            for attr in ("gui_runs_root", "results_root"):
                val = _as_path(getattr(ctx, attr, None))
                if val is not None:
                    return val

            geo_cfg = getattr(ctx, "geo_cfg", None)
            if geo_cfg is not None:
                val = _as_path(getattr(geo_cfg, "results_root", None))
                if val is not None:
                    return val

        home_root = Path.home() / ".fusionlab_runs"
        if (home_root / "results").is_dir():
            return home_root / "results"
        return home_root

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Header -----------------------------------------------------
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(6, 6, 6, 6)
        header_layout.setSpacing(6)

        self._title_lbl = QLabel("<b>Build NPZ for Stage-1</b>", self)
        self._title_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        header_layout.addWidget(self._title_lbl)
        layout.addLayout(header_layout)

        # --- Instructions + Generate Button ---------------------------
        instructions_lbl = QLabel(
            "This tool will help you build custom NPZ files for Stage-1. "
            "You can pass a dataset and specify configuration overrides, "
            "and it will generate the necessary NPZ files in a new folder.",
            self,
        )
        instructions_lbl.setWordWrap(True)
        instructions_lbl.setTextFormat(Qt.RichText)

        btn_generate_npz = QPushButton("Generate NPZ Files", self)
        btn_generate_npz.clicked.connect(self._on_generate_npz)

        layout.addWidget(instructions_lbl)
        layout.addWidget(btn_generate_npz)

        # --- Output Area ------------------------------------------------
        self._output_edit = QPlainTextEdit(self)
        self._output_edit.setReadOnly(True)
        self._output_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self._output_edit, stretch=1)

    # ------------------------------------------------------------------
    # Command generation helpers
    # ------------------------------------------------------------------
    def _generate_npz(self) -> str:
        """
        Generate NPZ files for training and validation data from Stage-1.
        """
        # Retrieve config and dataset path
        cfg = self._app_ctx.geo_cfg if self._app_ctx else None
        if cfg is None:
            return "No configuration found."

        city = getattr(cfg, "CITY_NAME", "unknown")
        model = getattr(cfg, "MODEL_NAME", "GeoPriorSubsNet")
        dataset_path = getattr(cfg, "DATA_DIR", "")
        run_dir = getattr(cfg, "BASE_OUTPUT_DIR", "")

        # Define output path
        npz_output_path = os.path.join(run_dir, f"{city}_stage1_npz")

        # Ensure the output directory exists
        ensure_directory_exists(npz_output_path)

        # Load dataset (simplified for this example)
        csv_path = getattr(cfg, "BIG_FN", "")
        dataset = pd.read_csv(os.path.join(dataset_path, csv_path))

        # Preprocessing (based on run_stage1)
        processed_data = self._preprocess_data(dataset)

        # Create and save NPZ
        npz_path = os.path.join(npz_output_path, "stage1_data.npz")
        np.savez_compressed(npz_path, **processed_data)

        return f"NPZ file saved to {npz_path}"

    def _preprocess_data(self, dataset: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Preprocess data: select features, encode, and scale.
        """
        # Example preprocessing steps (extend this based on your requirements)
        # For simplicity, we are only selecting columns here.
        columns_to_use = ["longitude", "latitude", "subsidence", "GWL_depth_bgs_z"]

        dataset = dataset[columns_to_use]
        dataset.fillna(0, inplace=True)

        # Create dummy arrays for features (this should be more detailed)
        inputs = dataset.values
        targets = dataset["subsidence"].values

        return {
            "inputs": inputs,
            "targets": targets
        }

    # ------------------------------------------------------------------
    # Command button handlers
    # ------------------------------------------------------------------
    def _on_generate_npz(self) -> None:
        """
        Generate NPZ file and display path in the output area.
        """
        result = self._generate_npz()
        self._output_edit.setPlainText(result)
