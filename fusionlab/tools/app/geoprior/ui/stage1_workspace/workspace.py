# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
stage1_workspace.workspace

Stage-1 workspace container:
- Quicklook
- Readiness
- Feature scaling
- Visual checks
- Run history
- Artifacts

UI-only. The controller (app.py) owns business logic and pushes
payloads (manifest/audit/scan/history).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import os 
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .artifacts import Stage1Artifacts
from .feature_scaling import Stage1FeatureScaling
from .quicklook import Stage1Quicklook
from .readiness import (
    CompatibilityResult,
    Stage1Readiness,
    Stage1Scan,
)
from .run_history import Stage1RunHistory, Stage1RunEntry
from .visual_checks import (
    Stage1VisualChecks,
    Stage1VisualData,
)

Json = Dict[str, Any]
PathLike = Union[str, "os.PathLike[str]"]


class Stage1Workspace(QWidget):
    """
    Composite Stage-1 workspace widget.

    Controller responsibilities:
    - load/parse manifest.json and optional stage1_scaling_audit.json
    - compute readiness/compat
    - decide active Stage-1 run and persist preference
    - open files/folders on request
    """

    request_open_path = pyqtSignal(str)
    request_set_active_stage1 = pyqtSignal(str, str)
    request_refresh_history = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        
        self._context_key: tuple[str, str, str, str, str] = (
            "",
            "",
            "",
            "",
            "",
        )

        self._build_ui()
        self._wire()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._context = None
        self.quicklook.clear()
        self.readiness.clear()
        self.feature_scaling.clear()
        self.visual_checks.clear()
        self.run_history.clear()
        self.artifacts.clear()

    def set_context(
        self,
        *,
        city: str,
        csv_path: Optional[PathLike],
        runs_root: Optional[PathLike],
        stage1_dir: Optional[PathLike] = None,
        model: str = "",
    ) -> None:
        key = (
            str(city or "").strip(),
            str(csv_path or ""),
            str(runs_root or ""),
            str(stage1_dir or ""),
            str(model or ""),
        )
        # Guard: if nothing changed, do nothing
        if key == self._context_key:
            return
        self._context_key = key

        kw = dict(
            city=city,
            csv_path=csv_path,
            runs_root=runs_root,
            stage1_dir=stage1_dir,
        )

        self.quicklook.set_context(**kw)
        self.readiness.set_context(**kw)
        self.feature_scaling.set_context(**kw)
        self.artifacts.set_context(**kw)

        self.visual_checks.set_context(
            city=city,
            stage1_dir=stage1_dir,
        )

        self.run_history.set_context(
            runs_root=str(runs_root or ""),
            city=city,
            model=model,
        )
        
    # def set_context(
    #     self,
    #     *,
    #     city: str,
    #     csv_path: Optional[PathLike],
    #     runs_root: Optional[PathLike],
    #     stage1_dir: Optional[PathLike] = None,
    #     model: str = "",
    # ) -> None:
    #     """
    #     Broadcast shared context to all panels.
    #     """
    #     # Normalize to stable comparable strings
    #     new_context = {
    #         "city": (city or "").strip(),
    #         "csv_path": str(csv_path) if csv_path else "",
    #         "runs_root": str(runs_root) if runs_root else "",
    #         "stage1_dir": str(stage1_dir) if stage1_dir else "",
    #         "model": (model or "").strip(),
    #     }
    
    #     # Guard: if nothing changed, do nothing
    #     if self._context == new_context:
    #         return
    #     self._context = new_context
    
    #     kw = dict(
    #         city=new_context["city"],
    #         csv_path=(csv_path if csv_path else None),
    #         runs_root=(runs_root if runs_root else None),
    #         stage1_dir=(stage1_dir if stage1_dir else None),
    #     )
    
    #     self.quicklook.set_context(**kw)
    #     self.readiness.set_context(**kw)
    #     self.feature_scaling.set_context(**kw)
    #     self.artifacts.set_context(**kw)
    
    #     self.visual_checks.set_context(
    #         city=new_context["city"],
    #         stage1_dir=(stage1_dir if stage1_dir else None),
    #     )
    
    #     self.run_history.set_context(
    #         runs_root=new_context["runs_root"],
    #         city=new_context["city"],
    #         model=new_context["model"],
    #     )

    def set_status_all(self, text: str) -> None:
        """
        Set a simple status line on each sub-panel.
        """
        self.quicklook.set_status(text)
        self.readiness.set_status(text)
        self.feature_scaling.set_status(text)
        self.artifacts.set_status(text)

    def set_manifest(self, manifest: Optional[Json]) -> None:
        """
        Push manifest.json dict to panels that use it.
        """
        self.quicklook.set_manifest(manifest)
        self.feature_scaling.set_manifest(manifest)
        self.artifacts.set_manifest(manifest)
        self.visual_checks.set_manifest(manifest)

    def set_scaling_audit(self, audit: Optional[Json]) -> None:
        """
        Push stage1_scaling_audit.json dict to panels that
        can interpret it.
        """
        self.quicklook.set_scaling_audit(audit)
        self.feature_scaling.set_scaling_audit(audit)
        self.visual_checks.set_scaling_audit(audit)

    def set_visual_data(self, data: Optional[Stage1VisualData]) -> None:
        """
        Optionally push DataFrames to Visual checks
        to avoid UI-thread CSV IO.
        """
        self.visual_checks.set_data(data)

    def set_readiness_payload(
        self,
        *,
        options: Optional[Dict[str, Any]] = None,
        scan: Optional[Stage1Scan] = None,
        compat: Optional[CompatibilityResult] = None,
        status: str = "",
    ) -> None:
        """
        Update readiness panel state.
        """
        self.readiness.set_options(options)
        self.readiness.set_stage1_scan(scan)
        self.readiness.set_compatibility(compat)
        if status:
            self.readiness.set_status(status)

    def set_history_entries(
        self,
        entries: Optional[List[Stage1RunEntry]],
    ) -> None:
        """
        Controller-provided run history entries.
        """
        self.run_history.set_entries(entries)

    def scan_history_local(self) -> None:
        """
        Optional convenience: let the widget scan the FS itself.
        """
        self.run_history.refresh_scan()

    def set_artifacts_extra(
        self,
        items: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """
        Add extra (label, path) items under "Extra".
        """
        self.artifacts.set_extra_items(items)

    def set_active_tab(self, name: str) -> None:
        """
        Select a workspace tab by its visible label.
        """
        name_l = (name or "").strip().lower()
        for i in range(self.tabs.count()):
            if (self.tabs.tabText(i) or "").lower() == name_l:
                self.tabs.setCurrentIndex(i)
                return

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tabs = QTabWidget()

        self.quicklook = Stage1Quicklook()
        self.readiness = Stage1Readiness()
        self.feature_scaling = Stage1FeatureScaling()
        self.visual_checks = Stage1VisualChecks()
        self.run_history = Stage1RunHistory()
        self.artifacts = Stage1Artifacts()

        self.tabs.addTab(self.quicklook, "Quicklook")
        self.tabs.addTab(self.readiness, "Readiness")
        self.tabs.addTab(self.feature_scaling, "Feature scaling")
        self.tabs.addTab(self.visual_checks, "Visual checks")
        self.tabs.addTab(self.run_history, "Run history")
        self.tabs.addTab(self.artifacts, "Artifacts")

        layout.addWidget(self.tabs, 1)

    def _wire(self) -> None:
        # Artifacts -> controller open path
        self.artifacts.request_open_path.connect(
            self.request_open_path.emit
        )

        # Run history -> controller open path / set active
        self.run_history.request_open_path.connect(
            self.request_open_path.emit
        )
        self.run_history.request_set_active.connect(
            self.request_set_active_stage1.emit
        )
        self.run_history.request_refresh.connect(
            self.request_refresh_history.emit
        )
