# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Preprocess tab UI for GeoPrior GUI (Stage-1).

This module only builds widgets and wires local button clicks
to signals. Business logic stays in app.py (controller layer).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    # QTableWidget,
    # QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QLineEdit
)

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]
MakeRunBtnFn = Callable[[str], object]


class PreprocessTab(QWidget):
    """Stage-1 preprocessing UI tab."""

    request_open_dataset = pyqtSignal()
    request_refresh = pyqtSignal()
    request_run_stage1 = pyqtSignal()
    request_feature_cfg = pyqtSignal()
    request_open_manifest = pyqtSignal()
    request_open_stage1_dir = pyqtSignal()
    request_use_for_city = pyqtSignal()
    
    request_browse_results_root = pyqtSignal()

    def __init__(
        self,
        *,
        make_card: MakeCardFn,
        make_run_button: MakeRunBtnFn,
        geo_cfg,
        runs_root,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._make_card = make_card
        self._make_run_button = make_run_button

        self._build_ui()
        self.sync_from_config(geo_cfg)
        self.set_results_root(runs_root)

        self._wire()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_results_root(self, runs_root) -> None:
        txt = "" if runs_root is None else str(runs_root)
        self.ed_prep_root.setText(txt)
        self.ed_prep_root.setToolTip(txt)

    def set_context(
        self,
        *,
        city: str,
        csv_path,
        runs_root,
    ) -> None:
        self.lbl_prep_city.setText(
            f"City: {city or '-'}"
        )
        self.lbl_prep_csv.setText(
            f"Dataset: {csv_path or '-'}"
        )
        self.set_results_root(runs_root)

    def set_stage1_status(
        self,
        *,
        state_text: str,
        manifest_text: str,
    ) -> None:
        self.lbl_prep_stage1_state.setText(state_text)
        self.lbl_prep_stage1_manifest.setText(manifest_text)

    def sync_from_config(self, geo_cfg) -> None:
        self.chk_prep_clean.setChecked(
            bool(getattr(geo_cfg, "clean_stage1_dir", False))
        )
        self.chk_prep_build_future.setChecked(
            bool(getattr(geo_cfg, "build_future_npz", False))
        )
        self.chk_prep_auto_reuse.setChecked(
            bool(
                getattr(
                    geo_cfg,
                    "stage1_auto_reuse_if_match",
                    True,
                )
            )
        )
        self.chk_prep_force_rebuild.setChecked(
            bool(
                getattr(
                    geo_cfg,
                    "stage1_force_rebuild_if_mismatch",
                    True,
                )
            )
        )

    def stage1_options(self) -> dict:
        return {
            "clean_stage1_dir": self.chk_prep_clean.isChecked(),
            "build_future_npz": (
                self.chk_prep_build_future.isChecked()
            ),
            "stage1_auto_reuse_if_match": (
                self.chk_prep_auto_reuse.isChecked()
            ),
            "stage1_force_rebuild_if_mismatch": (
                self.chk_prep_force_rebuild.isChecked()
            ),
        }

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        p_layout = QVBoxLayout(self)
        p_layout.setContentsMargins(6, 6, 6, 6)
        p_layout.setSpacing(8)

        # -------------------------------------------------
        # Row 1: Results root (full width)
        # -------------------------------------------------
        root_row = QHBoxLayout()
        root_row.setSpacing(10)
        
        lbl_root = QLabel("Results root:")
        self.ed_prep_root = QLineEdit()
        self.ed_prep_root.setReadOnly(True)
        
        self.btn_prep_browse_root = QPushButton("Browse…")
        self.btn_prep_refresh = QPushButton("Refresh")
        
        root_row.addWidget(lbl_root)
        root_row.addWidget(self.ed_prep_root, 1)
        root_row.addWidget(self.btn_prep_browse_root)
        root_row.addWidget(self.btn_prep_refresh)
        
        p_layout.addLayout(root_row)


        # -------------------------------------------------
        # Top row: 3 cards
        # -------------------------------------------------
        row = QHBoxLayout()
        row.setSpacing(10)
    
        # ---------------- Card 1: Inputs ----------------
        inp_card, inp_box = self._make_card(
            "Inputs (City + Dataset)"
        )
    
        self.lbl_prep_city = QLabel("City: -")
        self.lbl_prep_csv = QLabel("Dataset: -")
    
        for w in (self.lbl_prep_city, self.lbl_prep_csv):
            w.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            inp_box.addWidget(w)
    
        self.btn_prep_open_dataset = QPushButton(
            "Open dataset…"
        )
        self.btn_prep_feature_cfg = QPushButton(
            "Feature config…"
        )
        
        inp_btns = QHBoxLayout()
        inp_btns.setSpacing(8)
        
        inp_btns.addWidget(self.btn_prep_open_dataset)
        inp_btns.addWidget(self.btn_prep_feature_cfg)
        inp_btns.addStretch(1)
        
        inp_box.addLayout(inp_btns)

        row.addWidget(inp_card, 1)
    
        # ------------- Card 2: Stage-1 options ----------
        opt_card, opt_box = self._make_card("Stage-1 options")
    
        self.chk_prep_clean = QCheckBox(
            "Clean Stage-1 run dir before build"
        )
        self.chk_prep_auto_reuse = QCheckBox(
            "Auto-reuse compatible Stage-1 run"
        )
        self.chk_prep_force_rebuild = QCheckBox(
            "Force rebuild if mismatch"
        )
        self.chk_prep_build_future = QCheckBox(
            "Build future NPZ"
        )
    
        for cb in (
            self.chk_prep_clean,
            self.chk_prep_auto_reuse,
            self.chk_prep_force_rebuild,
            self.chk_prep_build_future,
        ):
            opt_box.addWidget(cb)
    
        opt_box.addStretch(1)
        row.addWidget(opt_card, 1)
    
        # -------------- Card 3: Stage-1 status ----------
        stat_card, stat_box = self._make_card("Stage-1 status")
    
        self.lbl_prep_stage1_state = QLabel("Stage-1: unknown")
        self.lbl_prep_stage1_manifest = QLabel("Manifest: -")
    
        for w in (
            self.lbl_prep_stage1_state,
            self.lbl_prep_stage1_manifest,
        ):
            w.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            stat_box.addWidget(w)
    
        btns = QHBoxLayout()
        self.btn_prep_open_manifest = QPushButton(
            "Open manifest"
        )
        self.btn_prep_open_stage1_dir = QPushButton(
            "Open folder"
        )
        self.btn_prep_use_for_city = QPushButton(
            "Use as default for city"
        )
    
        btns.addWidget(self.btn_prep_open_manifest)
        btns.addWidget(self.btn_prep_open_stage1_dir)
        btns.addWidget(self.btn_prep_use_for_city)
        btns.addStretch(1)
        stat_box.addLayout(btns)
    
        row.addWidget(stat_card, 1)
    
        p_layout.addLayout(row)
    

        # -------------------------------------------------
        # Workspace (expands)
        # -------------------------------------------------
        ws_card, ws_box = self._make_card("Stage-1 workspace")
    
        self.ws_tabs = QTabWidget()
    
        self.ws_quicklook = QTextBrowser()
        self.ws_quicklook.setOpenExternalLinks(True)
        self.ws_tabs.addTab(self.ws_quicklook, "Quicklook")
    
        self.ws_readiness = QTextBrowser()
        self.ws_tabs.addTab(self.ws_readiness, "Readiness")
    
        self.ws_features = QTextBrowser()
        self.ws_tabs.addTab(self.ws_features, "Features & scaling")
    
        self.ws_artifacts = QTreeWidget()
        self.ws_artifacts.setHeaderLabels(
            ["Artifact", "Path"]
        )
        self.ws_tabs.addTab(self.ws_artifacts, "Artifacts")
    
        ws_box.addWidget(self.ws_tabs)
        p_layout.addWidget(ws_card, 1)
        
        # -------------------------------------------------
        # Row 3: Feature config (left) + Run (right)
        # -------------------------------------------------
        run_row = QHBoxLayout()
        run_row.setSpacing(10)
        
        self.btn_run_stage1 = self._make_run_button(
            "Run Stage-1 preprocessing"
        )
        
        run_row.addStretch(1)
        run_row.addWidget(self.btn_run_stage1)
        
        p_layout.addLayout(run_row)


    def _wire(self) -> None:
        self.btn_prep_open_dataset.clicked.connect(
            self.request_open_dataset.emit
        )
        self.btn_prep_refresh.clicked.connect(
            self.request_refresh.emit
        )
        self.btn_prep_browse_root.clicked.connect(
            self.request_browse_results_root.emit
        )
    
        self.btn_run_stage1.clicked.connect(
            self.request_run_stage1.emit
        )
        self.btn_prep_feature_cfg.clicked.connect(
            self.request_feature_cfg.emit
        )
        self.btn_prep_open_manifest.clicked.connect(
            self.request_open_manifest.emit
        )
        self.btn_prep_open_stage1_dir.clicked.connect(
            self.request_open_stage1_dir.emit
        )
        self.btn_prep_use_for_city.clicked.connect(
            self.request_use_for_city.emit
        )

    def set_quicklook_html(self, html: str) -> None:
        self.ws_quicklook.setHtml(html or "")

    def set_readiness_html(self, html: str) -> None:
        self.ws_readiness.setHtml(html or "")

    def set_features_html(self, html: str) -> None:
        self.ws_features.setHtml(html or "")

    def set_artifacts_tree(self, items: list[tuple[str, str]]) -> None:
        """
        items: list of (label, path)
        """
        self.ws_artifacts.clear()
        for label, path in items or []:
            it = QTreeWidgetItem([label, path])
            self.ws_artifacts.addTopLevelItem(it)
        self.ws_artifacts.expandAll()
