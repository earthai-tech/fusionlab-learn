# geoprior/ui/inference/details.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os
from typing import Optional

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...device_options import runtime_summary_text


__all__ = ["InferenceDetailsPanel"]


def _exists(p: str) -> bool:
    try:
        return bool(p) and os.path.exists(p)
    except Exception:
        return False


def _open_path(p: str) -> None:
    if not _exists(p):
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(p))


class InferenceDetailsPanel(QWidget):
    """
    Sidebar extras [E] shown under the navigator.

    Styling alignment
    -----------------
    This panel reuses Train styles by matching:
    - QFrame#trainNavCard
    - QLabel#trainNavTitle
    - QLabel#sumLine
    - QToolButton#miniAction (same as Train mini-actions)

    Responsibilities
    ----------------
    - Last outputs shortcuts (run dir, eval csv, future csv, summary json)
    - Runtime snapshot summary (device/backend text)
    - Small tips box (static guidance)
    """

    def __init__(
        self,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._run_dir: str = ""
        self._eval_csv: str = ""
        self._future_csv: str = ""
        self._summary_json: str = ""

        self._build_ui()
        self._wire()
        self.refresh_runtime_summary()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def set_last_outputs(self, run_dir: str) -> None:
        self._run_dir = str(run_dir or "").strip()
        self._sync_buttons()

    def set_last_eval_csv(self, path: str) -> None:
        self._eval_csv = str(path or "").strip()
        self._sync_buttons()

    def set_last_future_csv(self, path: str) -> None:
        self._future_csv = str(path or "").strip()
        self._sync_buttons()

    def set_last_summary_json(self, path: str) -> None:
        self._summary_json = str(path or "").strip()
        self._sync_buttons()

    def refresh_runtime_summary(self) -> None:
        try:
            txt = runtime_summary_text()
        except Exception:
            txt = "Runtime summary unavailable."
        self.txt_runtime.setPlainText(txt)

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setObjectName("inferDetails")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # =========================================================
        # Card 1: Last outputs (Train nav card look)
        # =========================================================
        card1 = QFrame(self)
        card1.setObjectName("trainNavCard")
        card1.setFrameShape(QFrame.NoFrame)
        card1.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )

        c1 = QVBoxLayout(card1)
        c1.setContentsMargins(10, 10, 10, 10)
        c1.setSpacing(8)

        t1 = QLabel("Last outputs", card1)
        t1.setObjectName("trainNavTitle")

        row = QHBoxLayout()
        row.setSpacing(6)

        self.btn_open_run = self._mk_mini("Open run")
        self.btn_open_eval = self._mk_mini("Eval CSV")
        self.btn_open_future = self._mk_mini("Future CSV")
        self.btn_open_json = self._mk_mini("Summary JSON")

        row.addWidget(self.btn_open_run)
        row.addWidget(self.btn_open_eval)
        row.addWidget(self.btn_open_future)
        row.addWidget(self.btn_open_json)
        row.addStretch(1)

        self.lbl_out_hint = QLabel(
            "Shortcuts update after a run.",
            card1,
        )
        self.lbl_out_hint.setObjectName("sumLine")
        self.lbl_out_hint.setWordWrap(True)

        c1.addWidget(t1, 0)
        c1.addLayout(row)
        c1.addWidget(self.lbl_out_hint, 0)

        # =========================================================
        # Card 2: Runtime (use normal card styles)
        # =========================================================
        card2 = self._make_card_box("Runtime")
        c2 = QVBoxLayout(card2)
        c2.setContentsMargins(8, 6, 8, 8)
        c2.setSpacing(6)

        self.txt_runtime = QPlainTextEdit(self)
        self.txt_runtime.setReadOnly(True)
        self.txt_runtime.setObjectName("runtimeText")
        self.txt_runtime.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.MinimumExpanding,
        )
        self.txt_runtime.setMinimumHeight(90)

        hint = QLabel(
            "Auto-detected backend / device snapshot.",
            self,
        )
        hint.setObjectName("sumLine")
        hint.setWordWrap(True)

        c2.addWidget(self.txt_runtime, 1)
        c2.addWidget(hint, 0)

        # =========================================================
        # Card 3: Tips (normal card styles)
        # =========================================================
        card3 = self._make_card_box("Tips")
        c3 = QVBoxLayout(card3)
        c3.setContentsMargins(8, 6, 8, 8)
        c3.setSpacing(6)

        tips = QLabel(
            "• Manifest can be empty if auto-resolve is enabled.\n"
            "• Custom NPZ requires inputs; targets are optional.\n"
            "• Outputs are written under the model run directory.",
            self,
        )
        tips.setWordWrap(True)
        tips.setObjectName("tipsText")

        c3.addWidget(tips, 0)

        root.addWidget(card1, 0)
        root.addWidget(card2, 0)
        root.addWidget(card3, 0)
        root.addStretch(1)

        self._sync_buttons()

    # -----------------------------------------------------------------
    # Card helpers
    # -----------------------------------------------------------------
    def _make_card_box(self, title: str) -> QFrame:
        """
        Return the *body* frame with a cardTitle, like app._make_card().
        """
        box = QFrame(self)
        box.setObjectName("card")
        box.setFrameShape(QFrame.StyledPanel)
        box.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )

        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        lbl = QLabel(title, box)
        lbl.setObjectName("cardTitle")
        lbl.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        lbl.setContentsMargins(8, 6, 8, 6)

        outer.addWidget(lbl, 0)

        body = QFrame(box)
        body.setObjectName("cardBody")
        body.setFrameShape(QFrame.NoFrame)

        outer.addWidget(body, 1)
        return body

    def _mk_mini(self, text: str) -> QToolButton:
        """
        Train-like mini action.

        Requires a global style for QToolButton#miniAction
        (already used in other tabs).
        """
        b = QToolButton(self)
        b.setObjectName("miniAction")
        b.setText(text)
        b.setToolButtonStyle(Qt.ToolButtonTextOnly)
        b.setAutoRaise(True)
        b.setCursor(Qt.PointingHandCursor)  # type: ignore
        return b

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.btn_open_run.clicked.connect(
            lambda: _open_path(self._run_dir)
        )
        self.btn_open_eval.clicked.connect(
            lambda: _open_path(self._eval_csv)
        )
        self.btn_open_future.clicked.connect(
            lambda: _open_path(self._future_csv)
        )
        self.btn_open_json.clicked.connect(
            lambda: _open_path(self._summary_json)
        )

    def _sync_buttons(self) -> None:
        self.btn_open_run.setEnabled(_exists(self._run_dir))
        self.btn_open_eval.setEnabled(_exists(self._eval_csv))
        self.btn_open_future.setEnabled(_exists(self._future_csv))
        self.btn_open_json.setEnabled(_exists(self._summary_json))
