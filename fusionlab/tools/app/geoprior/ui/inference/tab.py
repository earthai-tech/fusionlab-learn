# geoprior/ui/inference/tab.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os 
from typing import Callable, Optional, Tuple

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QLabel, 
    QScrollArea, 
    QFrame
)

from ...config.store import GeoConfigStore

from .center_panel import InferenceCenterPanel
from .details import InferenceDetailsPanel
from .head import InferenceHeadBar
from .navigator import InferenceNavigator
from .preview import InferencePreviewPanel
from .runtime_snapshot import InferRuntimeSnapshot


MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]
MakeRunBtnFn = Callable[[str], QToolButton]


__all__ = ["InferenceTab"]


class InferenceTab(QWidget):
    """
    Inference tab (Train-like layout).

    This file is assembly-only:
    - splitters
    - placement of A/E, B, C, D, bottom bar

    Backward compatibility:
    - exposes legacy widget attributes used by app.py
    """

    run_clicked = pyqtSignal()
    advanced_clicked = pyqtSignal()

    browse_model_clicked = pyqtSignal()
    browse_manifest_clicked = pyqtSignal()
    browse_inputs_clicked = pyqtSignal()
    browse_targets_clicked = pyqtSignal()
    browse_calib_clicked = pyqtSignal()

    runtime_snapshot_changed = pyqtSignal(object)

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

        self._build_ui()
        self._wire()
        self._finalize_back_compat()

        self.refresh_from_store()

    # ---------------------------------------------------------
    # Public
    # ---------------------------------------------------------
    def refresh_from_store(self) -> None:
        self.center.refresh_from_store()
        self.preview.refresh()
        self.nav.refresh()

    def set_last_outputs(self, outputs: dict) -> None:
        outs = outputs or {}
        self.preview.set_last_outputs(outs)
    
        # Let preview be the source of truth for paths.
        run_dir = str(self.preview.last_run_dir() or "").strip()
    
        eval_csv = str(outs.get("eval_csv", "") or "").strip()
        future_csv = str(outs.get("future_csv", "") or "").strip()
        summary_json = str(outs.get("summary_json", "") or "").strip()
    
        def _exists(p: str) -> bool:
            try:
                return bool(p) and os.path.exists(p)
            except Exception:
                return False
    
        self.nav.set_shortcuts_enabled(
            run_ok=_exists(run_dir),
            eval_ok=_exists(eval_csv),
            future_ok=_exists(future_csv),
            json_ok=_exists(summary_json),
        )
    def _open_path(self, p: str) -> None:

        p = str(p or "").strip()
        if not p or not os.path.exists(p):
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    # ---------------------------------------------------------
    # UI (assembly-only)
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # Main split: left (A+E) vs right work (B+C+D)
        self.split_main = QSplitter(Qt.Horizontal, self)
        self.split_main.setChildrenCollapsible(False)

        # Left column
        left = QWidget(self)
        left.setSizePolicy(
            QSizePolicy.Minimum, # or QSizePolicy.Preferred,
            QSizePolicy.Expanding,
        )
        # left.setMaximumWidth(320)   # or 300
        # # optional: left.setMinimumWidth(260)

        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(8)

        self.nav = InferenceNavigator(
            store=self._store,
            parent=left,
        )
        self.details = InferenceDetailsPanel(parent=left)
        # keep details constructed if you want, but don't show it
        self.details.hide()
        left_l.addWidget(self.nav, 1)
        # left_l.addWidget(self.details, 0)

        # Right work area
        work = QWidget(self)
        work_l = QVBoxLayout(work)
        work_l.setContentsMargins(0, 0, 0, 0)
        work_l.setSpacing(8)

        self.head = InferenceHeadBar(
            store=self._store,
            parent=work,
        )

        self.split_work = QSplitter(Qt.Horizontal, work)
        self.split_work.setChildrenCollapsible(False)

        self.center = InferenceCenterPanel(
            store=self._store,
            make_card=self._make_card,
            parent=self.split_work,
        )

        prev_card, prev_body = self._make_card("Run preview")
        self.preview = InferencePreviewPanel(
            store=self._store,
            parent=prev_card,
        )
        prev_body.addWidget(self.preview, 1)
        
        # Wrap the preview card in an OUTER scroll area (Tune/Preprocess style)
        self._inf_preview_scroll = QScrollArea(self.split_work)
        self._inf_preview_scroll.setWidgetResizable(True)
        self._inf_preview_scroll.setStyleSheet(
            "QScrollArea{background:transparent;}"
        )
        self._inf_preview_scroll.setFrameShape(QFrame.NoFrame)
        self._inf_preview_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self._inf_preview_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        
        _pv_page = QWidget(self._inf_preview_scroll)
        self._inf_preview_scroll.setWidget(_pv_page)
        
        _pv_lay = QVBoxLayout(_pv_page)
        _pv_lay.setContentsMargins(0, 0, 0, 0)
        _pv_lay.setSpacing(10)
        _pv_lay.addWidget(prev_card, 0)
        _pv_lay.addStretch(1)
        
        # Splitter: ONLY TWO widgets
        self.split_work.addWidget(self.center)
        self.split_work.addWidget(self._inf_preview_scroll)

        
        self.split_work.setStretchFactor(0, 2)
        self.split_work.setStretchFactor(1, 2)

        work_l.addWidget(self.head, 0)
        work_l.addWidget(self.split_work, 1)

        self.split_main.addWidget(left)
        self.split_main.addWidget(work)
        self.split_main.setStretchFactor(0, 0)
        self.split_main.setStretchFactor(1, 1)

        root.addWidget(self.split_main, 1)

        # Bottom bar: status + Run
        bot = QHBoxLayout()
        bot.setContentsMargins(0, 0, 0, 0)
        bot.setSpacing(10)

        self.status = self.preview.status_label()
        self.btn_run_infer = self._make_run_button(
            "Run inference"
        )
        self.lbl_run = QLabel("Run:")
        self.lbl_run.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        
        bot.addWidget(self.status, 1)
        bot.addWidget(self.lbl_run)
        bot.addWidget(self.btn_run_infer, 0)

        root.addLayout(bot, 0)

        # Initial splitter sizes (soft)
        # self.split_main.setSizes([320, 900])
        self.split_main.setSizes([280, 900])  # instead of 320
        self.split_work.setSizes([650, 650])

    # ---------------------------------------------------------
    # Wiring
    # ---------------------------------------------------------
    def _wire(self) -> None:
        self.btn_run_infer.clicked.connect(
            self.run_clicked.emit
        )

        # Navigator -> Center focus
        self.nav.section_selected.connect(
            self.center.focus_section
        )

        # Center browse signals -> tab signals
        self.center.browse_model_clicked.connect(
            self.browse_model_clicked.emit
        )
        self.center.browse_manifest_clicked.connect(
            self.browse_manifest_clicked.emit
        )
        self.center.browse_inputs_clicked.connect(
            self.browse_inputs_clicked.emit
        )
        self.center.browse_targets_clicked.connect(
            self.browse_targets_clicked.emit
        )
        self.center.browse_calib_clicked.connect(
            self.browse_calib_clicked.emit
        )

        # Center runtime changes -> preview refresh
        self.center.runtime_changed.connect(
            self.preview.refresh
        )

        # Preview emits runtime snapshot -> nav + app
        self.preview.runtime_snapshot_changed.connect(
            self._on_snapshot
        )

        # Store changes -> refresh preview + nav
        self._store.config_changed.connect(
            lambda _k: self.refresh_from_store()
        )

        self.nav.open_run_clicked.connect(
            lambda: self._open_path(self.preview.last_run_dir())
        )
        self.nav.open_eval_clicked.connect(
            lambda: self._open_path(
                self.preview.last_outputs().get("eval_csv", ""))
        )
        self.nav.open_future_clicked.connect(
            lambda: self._open_path(
                self.preview.last_outputs().get("future_csv", ""))
        )
        self.nav.open_json_clicked.connect(
            lambda: self._open_path(
                self.preview.last_outputs().get("summary_json", ""))
        )
        
        self.center.cmb_inf_dataset.currentIndexChanged.connect(
            lambda _i: self._sync_head_runtime_mode()
        )
        self.center.chk_inf_use_future.toggled.connect(
            lambda _on: self._sync_head_runtime_mode()
        )
        
        # call once at end of _wire or after _build_ui
        self._sync_head_runtime_mode()

    def _on_snapshot(self, snap: InferRuntimeSnapshot) -> None:
        self.nav.set_runtime_snapshot(snap)
        self.head.refresh_from_store()
        self.runtime_snapshot_changed.emit(snap)
        
    def _sync_head_runtime_mode(self) -> None:
        src = str(self.center.cmb_inf_dataset.currentData() or "val")
        fut = bool(self.center.chk_inf_use_future.isChecked())
        self.head.set_runtime_mode(source=src, future=fut)

    # ---------------------------------------------------------
    # Backward-compatible attribute aliases
    # ---------------------------------------------------------
    def _finalize_back_compat(self) -> None:
        c = self.center

        self.inf_model_edit = c.inf_model_edit
        self.inf_model_btn = c.inf_model_btn
        self.inf_manifest_edit = c.inf_manifest_edit
        self.inf_manifest_btn = c.inf_manifest_btn
        self.cmb_inf_dataset = c.cmb_inf_dataset
        self.chk_inf_use_future = c.chk_inf_use_future
        self.inf_inputs_edit = c.inf_inputs_edit
        self.inf_inputs_btn = c.inf_inputs_btn
        self.inf_targets_edit = c.inf_targets_edit
        self.inf_targets_btn = c.inf_targets_btn

        self.chk_inf_use_source_calib = c.chk_inf_use_source_calib
        self.chk_inf_fit_calib = c.chk_inf_fit_calib
        self.inf_calib_edit = c.inf_calib_edit
        self.inf_calib_btn = c.inf_calib_btn
        self.sp_inf_cov = c.sp_inf_cov
        self.chk_inf_include_gwl = c.chk_inf_include_gwl
        self.chk_inf_plots = c.chk_inf_plots
        self.sp_inf_batch = c.sp_inf_batch

        self.btn_inf_options = c.btn_inf_options
