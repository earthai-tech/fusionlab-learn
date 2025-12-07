# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Reproduce Run Helper tool for GeoPrior GUI.
#
# This tool generates Python and shell commands to reproduce a run, either
# training, tuning, or inference, outside the GUI. It extracts all the
# relevant config and environment variables, generating ready-to-run commands.

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
)

class ReproduceRunHelperTool(QWidget):
    """
    Reproduce Run Helper: Generates CLI commands to reproduce a run.
    
    This tool takes the current configuration (dataset, city, model, etc.)
    and generates corresponding Python or shell commands for reproducing
    the run outside the GUI.
    """

    def __init__(self, app_ctx: Optional[object] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._app_ctx = app_ctx
        self._results_root: Path = self._guess_results_root()
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

        self._title_lbl = QLabel("<b>Reproduce Run Helper</b>", self)
        self._title_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        header_layout.addWidget(self._title_lbl)
        layout.addLayout(header_layout)

        # --- Instructions + Generate Buttons ---------------------------
        instructions_lbl = QLabel(
            "This tool will generate ready-to-use Python or shell commands "
            "to reproduce a given run outside the GUI, including the right "
            "config overrides.",
            self,
        )
        instructions_lbl.setWordWrap(True)
        instructions_lbl.setTextFormat(Qt.RichText)

        btn_generate_python = QPushButton("Generate Python Command", self)
        btn_generate_python.clicked.connect(self._on_generate_python_command)

        btn_generate_shell = QPushButton("Generate Shell Command", self)
        btn_generate_shell.clicked.connect(self._on_generate_shell_command)

        layout.addWidget(instructions_lbl)
        layout.addWidget(btn_generate_python)
        layout.addWidget(btn_generate_shell)

        # --- Output Area ------------------------------------------------
        self._output_edit = QPlainTextEdit(self)
        self._output_edit.setReadOnly(True)
        self._output_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self._output_edit, stretch=1)

    # ------------------------------------------------------------------
    # Command generation helpers
    # ------------------------------------------------------------------
    def _generate_python_command(self) -> str:
        """
        Generate the Python command to reproduce the run, based on current config.
        """
        cfg = self._app_ctx.geo_cfg if self._app_ctx else None
        if cfg is None:
            return "No configuration found."

        city = getattr(cfg, "CITY_NAME", "unknown")
        model = getattr(cfg, "MODEL_NAME", "GeoPriorSubsNet")
        dataset_path = getattr(cfg, "DATA_DIR", "")
        run_dir = getattr(cfg, "BASE_OUTPUT_DIR", "")

        python_command = (
            f"python run_stage1.py "
            f"--city {city} "
            f"--model {model} "
            f"--dataset {dataset_path} "
            f"--run_dir {run_dir} "
            f"--config {run_dir}/config.json "
            f"--train_data {dataset_path}/{city}_train.csv"
        )

        return python_command

    def _generate_shell_command(self) -> str:
        """
        Generate the shell command to reproduce the run, based on current config.
        """
        cfg = self._app_ctx.geo_cfg if self._app_ctx else None
        if cfg is None:
            return "No configuration found."

        city = getattr(cfg, "CITY_NAME", "unknown")
        model = getattr(cfg, "MODEL_NAME", "GeoPriorSubsNet")
        dataset_path = getattr(cfg, "DATA_DIR", "")
        run_dir = getattr(cfg, "BASE_OUTPUT_DIR", "")

        shell_command = (
            f"#!/bin/bash\n"
            f"# Shell command to reproduce the GeoPrior run for city: {city}\n\n"
            f"python run_stage1.py \\\n"
            f"  --city {city} \\\n"
            f"  --model {model} \\\n"
            f"  --dataset {dataset_path} \\\n"
            f"  --run_dir {run_dir} \\\n"
            f"  --config {run_dir}/config.json \\\n"
            f"  --train_data {dataset_path}/{city}_train.csv"
        )

        return shell_command

    # ------------------------------------------------------------------
    # Command button handlers
    # ------------------------------------------------------------------
    def _on_generate_python_command(self) -> None:
        """
        Generate the Python command and display it in the output area.
        """
        command = self._generate_python_command()
        self._output_edit.setPlainText(command)

    def _on_generate_shell_command(self) -> None:
        """
        Generate the shell command and display it in the output area.
        """
        command = self._generate_shell_command()
        self._output_edit.setPlainText(command)
