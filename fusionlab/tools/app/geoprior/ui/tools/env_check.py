# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Environment / device check tool.

Provides a quick overview of:
- Python version and platform;
- TensorFlow version (if installed);
- detected GPUs via tf.config.list_physical_devices("GPU");
- any device overrides from the current RunEnv (if available).
"""

from __future__ import annotations

from typing import  Optional
import sys
import platform
import importlib

# from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
)

from ...styles import SECONDARY_TBLUE


class EnvironmentCheckTool(QWidget):
    """
    Simple read-only environment summary.

    Parameters
    ----------
    app_ctx : object, optional
        Main GeoPrior window. If it exposes a ``_run_env`` with a
        ``device_overrides`` dict, that information is also displayed.
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

        self._init_ui()
        self._run_check()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)

        self.lbl_summary = QLabel("TensorFlow / GPU environment.")
        self.lbl_summary.setStyleSheet("font-weight: 600;")
        header.addWidget(self.lbl_summary, 1)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setToolTip("Re-run environment detection.")
        self.btn_refresh.setStyleSheet(
            f"QPushButton:hover {{ background-color: {SECONDARY_TBLUE}; }}"
        )
        header.addWidget(self.btn_refresh)

        layout.addLayout(header)

        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMinimumHeight(220)
        layout.addWidget(self.txt, 1)

        layout.addStretch(1)

        self.btn_refresh.clicked.connect(self._run_check)

    def _run_check(self) -> None:
        """Run environment detection and update the text box."""
        lines = []

        # --- Python / platform -----------------------------------------
        lines.append(f"Python: {sys.version.split()[0]}")
        lines.append(f"Executable: {sys.executable}")
        lines.append(f"Platform: {platform.platform()}")
        lines.append("")

        # --- TensorFlow + GPUs -----------------------------------------
        try:
            tf_spec = importlib.util.find_spec("tensorflow")
        except Exception:
            tf_spec = None

        if tf_spec is None:
            lines.append("TensorFlow: not installed in this environment.")
        else:
            try:
                import tensorflow as tf  # type: ignore

                lines.append(f"TensorFlow: {tf.__version__}")
                try:
                    gpus = tf.config.list_physical_devices("GPU")
                except Exception as exc:  # pragma: no cover
                    lines.append(f"  GPU query failed: {exc}")
                else:
                    if gpus:
                        lines.append(f"  GPUs detected: {len(gpus)}")
                        for g in gpus:
                            name = getattr(g, "name", repr(g))
                            lines.append(f"    - {name}")
                    else:
                        lines.append("  GPUs detected: none (CPU only).")
            except Exception as exc:
                lines.append(
                    f"TensorFlow: import error ({exc.__class__.__name__}: {exc})"
                )

        # --- RunEnv / device overrides (if available) ------------------
        ctx = self._app_ctx
        run_env = getattr(ctx, "_run_env", None) if ctx is not None else None
        dev_overrides = getattr(run_env, "device_overrides", None)

        if dev_overrides:
            lines.append("")
            lines.append("GeoPrior device overrides:")
            for k, v in dev_overrides.items():
                lines.append(f"  {k}: {v!r}")

        self.txt.setPlainText("\n".join(lines))
        self.lbl_summary.setText("Environment check (last run just now).")
