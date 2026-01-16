# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.summary

Modern Summary card.

Shows a compact overview of the current configuration:
- Project (city, model, mode, PDE)
- Data paths (dataset, results root)
- Time window (train end, forecast start, horizon, steps)
- Training (epochs, batch, lr)

The card listens to the store and refreshes itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QStyle,
    QWidget,
    QFrame,
    QMenu,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
)

from ....config.store import GeoConfigStore
from .base import CardBase

def _as_path_str(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, Path):
        return str(v)
    s = str(v).strip()
    return s if s else "-"


def _as_num(v: Any) -> str:
    if v is None:
        return "-"
    try:
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)
    except Exception:
        return str(v)

class ElidedLabel(QLabel):
    """
    QLabel that renders *middle-elided* text (…)
    while keeping the full text for tooltip/copy.

    UX:
    - Tooltip shows full text
    - Double-click copies full text
    - Right-click menu: "Copy full path"
    """

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
        *,
        elide_mode: Qt.TextElideMode = Qt.ElideMiddle,
    ) -> None:
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = elide_mode

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self.setMinimumWidth(60)
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.setText(text)

    def fullText(self) -> str:
        return self._full_text

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._full_text = "" if text is None else str(text)
        self._apply_elide()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._apply_elide()

    def _apply_elide(self) -> None:
        # Use contentsRect so padding/frames are respected.
        w = max(0, self.contentsRect().width())
        fm = self.fontMetrics()
        shown = fm.elidedText(self._full_text, self._elide_mode, w)

        # IMPORTANT: call base QLabel.setText to avoid recursion.
        super().setText(shown)

        if self._full_text and self._full_text != "-":
            self.setToolTip(
                f"{self._full_text}\n\n"
                "Double-click to copy • Right-click for menu"
            )
        else:
            self.setToolTip(self._full_text)

    def mouseDoubleClickEvent(self, e) -> None:
        if self._full_text and self._full_text != "-":
            QApplication.clipboard().setText(self._full_text)
        super().mouseDoubleClickEvent(e)

    def contextMenuEvent(self, e) -> None:
        if not self._full_text or self._full_text == "-":
            return

        menu = QMenu(self)
        act = menu.addAction("Copy full path")
        act.triggered.connect(
            lambda: QApplication.clipboard().setText(self._full_text)
        )
        menu.exec_(e.globalPos())

class SummaryCard(CardBase):
    """Store-driven summary."""

    request_copy_snapshot = pyqtSignal()
    request_show_overrides = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="summary",
            title="Summary",
            subtitle=(
                "Quick preview of your experiment setup."
            ),
            parent=parent,
        )

        self.store = store
        self._vals: Dict[str, QLabel] = {}

        self._build()
        self._wire()
        self.refresh()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build(self) -> None:
        body = self.body_layout()
    
        # ------------------------------------------------------------
        # Local styling hooks (objectNames) + safe defaults
        # (Theme-specific colors are best set in styles.py; see below.)
        # ------------------------------------------------------------
        self.setStyleSheet(
            self.styleSheet()
            + "\n"
            + "\n".join(
                [
                    "QFrame#summaryPanel {",
                    "  border-radius: 10px;",
                    "}",
                    "QLabel#summaryPanelTitle {",
                    "  font-weight: 700;",
                    "  font-size: 12px;",
                    "}",
                    "QLabel#summaryKey {",
                    "  font-size: 11px;",
                    "}",
                    "QLabel#summaryValue {",
                    "  font-size: 11px;",
                    "  font-weight: 600;",
                    "}",
                ]
            )
        )
    
        wrap = QWidget(self)
        wrap.setObjectName("summaryWrap")
        outer = QGridLayout(wrap)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setHorizontalSpacing(10)
        outer.setVerticalSpacing(10)
    
        outer.setColumnStretch(0, 1)
        outer.setColumnStretch(1, 1)
    
        def _panel(
            title: str,
            icon: QStyle.StandardPixmap,
        ) -> QFormLayout:
            frame = QFrame(wrap)
            frame.setObjectName("summaryPanel")
            frame.setAttribute(Qt.WA_StyledBackground, True)
            frame.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Minimum,
            )
    
            v = QVBoxLayout(frame)
            v.setContentsMargins(10, 8, 10, 10)
            v.setSpacing(6)
    
            # Header row (icon + title)
            head = QWidget(frame)
            hl = QHBoxLayout(head)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)
    
            ico = QLabel(head)
            pm = self.style().standardIcon(icon).pixmap(16, 16)
            ico.setPixmap(pm)
    
            lab = QLabel(title, head)
            lab.setObjectName("summaryPanelTitle")
    
            hl.addWidget(ico, 0)
            hl.addWidget(lab, 0)
            hl.addStretch(1)
    
            v.addWidget(head, 0)
    
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setHorizontalSpacing(10)
            form.setVerticalSpacing(6)
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.setFormAlignment(Qt.AlignTop)
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

            v.addLayout(form, 1)
            return frame, form
    
        def _row(
            form: QFormLayout,
            key: str,
            label: str,
            parent: QWidget,
        ) -> None:
            k = QLabel(f"{label}:", parent)
            k.setObjectName("summaryKey")
        
            # Use elided label for long paths
            if key in ("dataset_path", "results_root"):
                v = ElidedLabel("-", parent)
                v.setObjectName("summaryPathValue")
                # Keep it single-line (elide does the job)
                v.setWordWrap(False)
            else:
                v = QLabel("-", parent)
                v.setObjectName("summaryValue")
                v.setTextInteractionFlags(Qt.TextSelectableByMouse)
                v.setWordWrap(True)
        
            form.addRow(k, v)
            self._vals[str(key)] = v

        # ------------------------------------------------------------
        # Panels (2 x 2)
        # ------------------------------------------------------------
        proj_frame, proj = _panel(
            "Project",
            QStyle.SP_FileDialogInfoView,
        )
        _row(proj, "city", "City", proj_frame)
        _row(proj, "model_name", "Model", proj_frame)
        _row(proj, "mode", "Mode", proj_frame)
        _row(proj, "pde_mode", "PDE mode", proj_frame)
    
        data_frame, data = _panel(
            "Data paths",
            QStyle.SP_DirOpenIcon,
        )
        _row(data, "dataset_path", "Dataset", data_frame)
        _row(data, "results_root", "Results root", data_frame)
    
        time_frame, timef = _panel(
            "Time window",
            QStyle.SP_FileDialogDetailedView,
        )
        _row(timef, "train_end_year", "Train end", time_frame)
        _row(timef, "forecast_start_year", "Forecast start", time_frame)
        _row(timef, "forecast_horizon_years", "Horizon", time_frame)
        _row(timef, "time_steps", "Time steps", time_frame)
        _row(timef, "__forecast_end", "Forecast end", time_frame)
        _row(timef, "build_future_npz", "Build future NPZ", time_frame)
    
        train_frame, train = _panel(
            "Training",
            QStyle.SP_MediaPlay,
        )
        _row(train, "epochs", "Epochs", train_frame)
        _row(train, "batch_size", "Batch", train_frame)
        _row(train, "learning_rate", "LR", train_frame)
        _row(train, "evaluate_training", "Eval", train_frame)
    
        outer.addWidget(proj_frame, 0, 0)
        outer.addWidget(data_frame, 0, 1)
        outer.addWidget(time_frame, 1, 0)
        outer.addWidget(train_frame, 1, 1)
    
        body.addWidget(wrap, 0)
    
        # Actions (header right)
        btn = self.add_action(
            text="Copy",
            tip="Copy snapshot JSON to clipboard",
            icon=QStyle.SP_DialogSaveButton,
        )
        btn.clicked.connect(self._copy_snapshot)
    
        btn2 = self.add_action(
            text="Diff",
            tip="Show overrides (diff)",
            icon=QStyle.SP_FileDialogDetailedView,
        )
        btn2.clicked.connect(self.request_show_overrides)


    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.store.config_changed.connect(
            lambda _k: self.refresh()
        )
        self.store.config_replaced.connect(
            lambda _cfg: self.refresh()
        )
        self.store.dirty_changed.connect(
            lambda _n: self._sync_badges()
        )

    # -----------------------------------------------------------------
    # Refresh
    # -----------------------------------------------------------------
    def refresh(self) -> None:
        cfg = self.store.cfg

        end_year = (
            int(cfg.forecast_start_year)
            + int(cfg.forecast_horizon_years)
            - 1
        )

        self._set("city", cfg.city)
        self._set("model_name", cfg.model_name)
        self._set("mode", cfg.mode)
        self._set("pde_mode", cfg.pde_mode)

        self._set("dataset_path", cfg.dataset_path)
        self._set("results_root", cfg.results_root)

        self._set("train_end_year", cfg.train_end_year)
        self._set(
            "forecast_start_year",
            cfg.forecast_start_year,
        )
        self._set(
            "forecast_horizon_years",
            cfg.forecast_horizon_years,
        )
        self._set("time_steps", cfg.time_steps)
        self._set("__forecast_end", end_year)

        self._set("epochs", cfg.epochs)
        self._set("batch_size", cfg.batch_size)
        self._set("learning_rate", cfg.learning_rate)

        self._set(
            "build_future_npz",
            "yes" if cfg.build_future_npz else "no",
        )
        self._set(
            "evaluate_training",
            "yes" if cfg.evaluate_training else "no",
        )

        self._sync_badges()

    def _set(self, key: str, v: Any) -> None:
        lab = self._vals.get(str(key))
        if lab is None:
            return
    
        if key in ("dataset_path", "results_root"):
            txt = _as_path_str(v)
        elif isinstance(v, (int, float)):
            txt = _as_num(v)
        else:
            txt = _as_path_str(v)
    
        lab.setText(txt)
    
        # Keep ElidedLabel tooltip (it includes copy instructions)
        if not isinstance(lab, ElidedLabel):
            lab.setToolTip(txt)


    def _sync_badges(self) -> None:
        n = int(self.store.overrides_count())
        self.badge(
            "dirty",
            text=f"{n} overrides",
            accent="warn" if n else "ok",
            tip="Number of overridden fields",
        )

        locked = bool(
            self.store.get("setup.locked", False)
        )
        self.badge(
            "lock",
            text="Locked" if locked else "Unlocked",
            accent="ok" if locked else "warn",
            tip="Lock for run state",
        )

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------
    def _copy_snapshot(self) -> None:
        try:
            payload = self.store.cfg.as_dict()
            txt = json.dumps(payload, indent=2)
            QApplication.clipboard().setText(txt)
            self.request_copy_snapshot.emit()
        except Exception as exc:
            self.store.error_raised.emit(str(exc))
