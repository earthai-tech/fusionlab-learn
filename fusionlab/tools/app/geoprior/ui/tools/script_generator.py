# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.tools.script_generator

Script / batch generator tool (UI-only).

Goals
-----
- Modern UX: presets + parameter cards + live preview.
- Generates reproducible CLI scripts (python/bash/powershell).
- No execution here (backend later).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
    QMessageBox,
    QMenu, 
    QScrollArea,
    QSizePolicy

)

from ..icon_utils import try_icon 


class _Chip(QLabel):
    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self.setObjectName("chip")
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setMinimumHeight(22)
        self.setStyleSheet(
            "padding:2px 10px;"
            "border-radius:11px;"
            "background: palette(midlight);"
            "color: palette(text);"
        )


@dataclass
class _Preset:
    key: str
    title: str
    desc: str


class ScriptGeneratorTool(QWidget):
    """
    Script / batch generator (UI-only).

    Parameters
    ----------
    app_ctx : object, optional
        Main GUI context (used later to prefill roots/city).
    store : object, optional
        Config store (future). Not used yet.
    parent : QWidget, optional
        Qt parent.
    """

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        store: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_ctx = app_ctx
        self._store = store

        self._presets: List[_Preset] = [
            _Preset(
                "stage1",
                "Stage-1 preprocessing",
                "Build Stage-1 artifacts for one or more cities.",
            ),
            _Preset(
                "train",
                "Training",
                "Train GeoPriorSubsNet from a manifest (or latest).",
            ),
            _Preset(
                "tune",
                "Tuning",
                "Hyperparameter search (tuning run_*) for a city.",
            ),
            _Preset(
                "infer",
                "Inference",
                "Run inference from a trained model + dataset split.",
            ),
            _Preset(
                "xfer",
                "Transfer matrix",
                "Generate cross-city transferability runs + panels.",
            ),
        ]

        self._build_ui()
        self._connect_ui()
        self._apply_preset("stage1")

    # ------------------------------------------------------------------
    # ToolPageFrame will call refresh() if present
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self._render_preview()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # --- top summary row (compact)
        top = QHBoxLayout()
        top.setSpacing(8)

        self._chip_mode = _Chip("UI-only", self)
        self._chip_mode.setToolTip(
            "This tool generates scripts only. "
            "Execution will be added later."
        )

        self._chip_target = _Chip("Preset: stage1", self)
        self._chip_target.setToolTip("Current preset.")

        self._btn_copy = QToolButton(self)
        self._btn_copy.setAutoRaise(True)
        self._btn_copy.setObjectName("miniAction")
        self._btn_copy.setToolTip("Copy script to clipboard")
        self._btn_copy.setIcon(
            self.style().standardIcon(QStyle.SP_DialogOpenButton)
        )

        self._btn_save = QToolButton(self)
        self._btn_save.setAutoRaise(True)
        self._btn_save.setObjectName("miniAction")
        self._btn_save.setToolTip("Save script…")
        self._btn_save.setIcon(
            self.style().standardIcon(QStyle.SP_DialogSaveButton)
        )

        # self._btn_export_bundle = QToolButton(self)
        # self._btn_export_bundle.setAutoRaise(True)
        # self._btn_export_bundle.setObjectName("miniAction")
        # self._btn_export_bundle.setToolTip(
        #     "Export script + params JSON bundle…"
        # )
        # self._btn_export_bundle.setIcon(
        #     self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        # )
        
        self._btn_export = QToolButton(self)
        self._btn_export.setAutoRaise(True)
        self._btn_export.setObjectName("miniAction")
        self._btn_export.setText("Export ▾")
        self._btn_export.setPopupMode(
            QToolButton.InstantPopup
        )
        
        self._export_menu = QMenu(self._btn_export)
        self._btn_export.setMenu(self._export_menu)
        
        a_bundle = self._export_menu.addAction(
            "Bundle (script + params.json)…"
        )
        a_make = self._export_menu.addAction(
            "Makefile…"
        )
        a_slurm = self._export_menu.addAction(
            "Slurm sbatch…"
        )
        
        self._act_export_bundle = a_bundle
        self._act_export_make = a_make
        self._act_export_slurm = a_slurm

        self._btn_load_tpl = QToolButton(self)
        self._btn_load_tpl.setAutoRaise(True)
        self._btn_load_tpl.setObjectName("miniAction")
        self._btn_load_tpl.setToolTip(
            "Load template from run manifest…"
        )
        ico = try_icon("manifest.svg")
        if ico is None or ico.isNull():
            ico = self.style().standardIcon(
                QStyle.SP_FileDialogDetailedView
            )
        self._btn_load_tpl.setIcon(ico)
        
 
        top.addWidget(self._chip_mode)
        top.addWidget(self._chip_target)
        top.addStretch(1)
        top.addWidget(self._btn_export)
        top.addWidget(self._btn_load_tpl)
        top.addWidget(self._btn_copy)
        top.addWidget(self._btn_save)

        root.addLayout(top)

        split = QSplitter(Qt.Horizontal, self)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        # ==============================================================
        # Left: builder cards
        # ==============================================================
        left_scroll = QScrollArea(self)
        left_scroll.setObjectName("scriptGenLeftScroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        left_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )

        left = QFrame(self)
        left.setObjectName("scriptGenLeft")
        left.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.MinimumExpanding,
        )

        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(10)

        left_scroll.setWidget(left)

        # --- preset picker
        gb_preset = QGroupBox("Preset", left)
        gb_preset.setObjectName("card")
        gl = QGridLayout(gb_preset)
        gl.setContentsMargins(10, 10, 10, 10)
        gl.setHorizontalSpacing(10)
        gl.setVerticalSpacing(8)

        self.cmb_preset = QComboBox(gb_preset)
        for p in self._presets:
            self.cmb_preset.addItem(p.title, p.key)

        self.lbl_preset_desc = QLabel("", gb_preset)
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setObjectName("muted")

        self.cmb_shell = QComboBox(gb_preset)
        self.cmb_shell.addItem("Bash (.sh)", "bash")
        self.cmb_shell.addItem("PowerShell (.ps1)", "ps1")
        self.cmb_shell.addItem("Python snippet (.py)", "py")

        gl.addWidget(QLabel("What to generate:", gb_preset), 0, 0)
        gl.addWidget(self.cmb_preset, 0, 1)
        gl.addWidget(self.lbl_preset_desc, 1, 0, 1, 2)
        gl.addWidget(QLabel("Script format:", gb_preset), 2, 0)
        gl.addWidget(self.cmb_shell, 2, 1)

        ll.addWidget(gb_preset)

        # --- run scope (cities / runs / repeats)
        gb_scope = QGroupBox("Scope", left)
        gb_scope.setObjectName("card")
        gs = QGridLayout(gb_scope)
        gs.setContentsMargins(10, 10, 10, 10)
        gs.setHorizontalSpacing(10)
        gs.setVerticalSpacing(8)
        gb_grid = QGroupBox("Multi-run grid", left)
        gb_grid.setObjectName("card")
        
        gg = QGridLayout(gb_grid)
        gg.setContentsMargins(10, 10, 10, 10)
        gg.setHorizontalSpacing(10)
        gg.setVerticalSpacing(8)

        self.chk_grid = QCheckBox(
            "Enable grid: cities × seeds × ablations",
            gb_grid,
        )
        self.chk_grid.setChecked(False)

        self.edit_seeds = QLineEdit(gb_grid)
        self.edit_seeds.setPlaceholderText(
            "Seeds (comma-separated), e.g. 0,1,2,3"
        )

        self.edit_ablations = QPlainTextEdit(gb_grid)
        self.edit_ablations.setObjectName("codePreview")
        self.edit_ablations.setPlaceholderText(
            "Ablations (one per line):\n"
            "label | extra args\n"
            "baseline |\n"
            "no_phys | --pde-mode off\n"
        )
        self.edit_ablations.setMaximumHeight(120)

        self._chip_jobs = _Chip("Jobs: 1", gb_grid)

        gg.addWidget(self.chk_grid, 0, 0, 1, 2)
        gg.addWidget(QLabel("Seeds:", gb_grid), 1, 0)
        gg.addWidget(self.edit_seeds, 1, 1)
        gg.addWidget(QLabel("Ablations:", gb_grid), 2, 0)
        gg.addWidget(self.edit_ablations, 2, 1)
        gg.addWidget(self._chip_jobs, 3, 0, 1, 2)

        ll.addWidget(gb_grid)

        self.edit_cities = QLineEdit(gb_scope)
        self.edit_cities.setPlaceholderText(
            "Cities (comma-separated), e.g. nansha, zhongshan"
        )

        self.sp_repeats = QSpinBox(gb_scope)
        self.sp_repeats.setRange(1, 50)
        self.sp_repeats.setValue(1)

        self.chk_timestamped = QCheckBox(
            "Include timestamp folder suffix",
            gb_scope,
        )
        self.chk_timestamped.setChecked(True)

        gs.addWidget(QLabel("Cities:", gb_scope), 0, 0)
        gs.addWidget(self.edit_cities, 0, 1)
        gs.addWidget(QLabel("Repeats:", gb_scope), 1, 0)
        gs.addWidget(self.sp_repeats, 1, 1)
        gs.addWidget(self.chk_timestamped, 2, 0, 1, 2)

        ll.addWidget(gb_scope)

        # --- paths / roots (UI now, store later)
        gb_paths = QGroupBox("Paths", left)
        gb_paths.setObjectName("card")
        gp = QGridLayout(gb_paths)
        gp.setContentsMargins(10, 10, 10, 10)
        gp.setHorizontalSpacing(10)
        gp.setVerticalSpacing(8)

        self.edit_results_root = QLineEdit(gb_paths)
        self.edit_results_root.setPlaceholderText(
            "Results root (optional)"
        )

        self.btn_browse_root = QToolButton(gb_paths)
        self.btn_browse_root.setAutoRaise(True)
        self.btn_browse_root.setObjectName("miniAction")
        self.btn_browse_root.setToolTip("Browse results root…")
        self.btn_browse_root.setIcon(
            self.style().standardIcon(QStyle.SP_DirOpenIcon)
        )

        self.edit_manifest = QLineEdit(gb_paths)
        self.edit_manifest.setPlaceholderText(
            "Manifest path (optional)"
        )

        self.btn_browse_manifest = QToolButton(gb_paths)
        self.btn_browse_manifest.setAutoRaise(True)
        self.btn_browse_manifest.setObjectName("miniAction")
        self.btn_browse_manifest.setToolTip("Browse manifest…")
        self.btn_browse_manifest.setIcon(
            self.style().standardIcon(QStyle.SP_FileIcon)
        )

        gp.addWidget(QLabel("Results root:", gb_paths), 0, 0)
        gp.addWidget(self.edit_results_root, 0, 1)
        gp.addWidget(self.btn_browse_root, 0, 2)
        gp.addWidget(QLabel("Manifest:", gb_paths), 1, 0)
        gp.addWidget(self.edit_manifest, 1, 1)
        gp.addWidget(self.btn_browse_manifest, 1, 2)

        ll.addWidget(gb_paths)

        # --- knobs (preview-only for now)
        gb_knobs = QGroupBox("Key knobs (preview)", left)
        gb_knobs.setObjectName("card")
        gk = QGridLayout(gb_knobs)
        gk.setContentsMargins(10, 10, 10, 10)
        gk.setHorizontalSpacing(10)
        gk.setVerticalSpacing(8)

        self.sp_batch = QSpinBox(gb_knobs)
        self.sp_batch.setRange(1, 4096)
        self.sp_batch.setValue(32)

        self.sp_epochs = QSpinBox(gb_knobs)
        self.sp_epochs.setRange(1, 5000)
        self.sp_epochs.setValue(20)

        self.sp_lr = QDoubleSpinBox(gb_knobs)
        self.sp_lr.setDecimals(6)
        self.sp_lr.setRange(1e-8, 10.0)
        self.sp_lr.setValue(0.001)

        self.chk_dry = QCheckBox("Dry-run flag", gb_knobs)
        self.chk_dry.setChecked(False)

        gk.addWidget(QLabel("Batch size:", gb_knobs), 0, 0)
        gk.addWidget(self.sp_batch, 0, 1)
        gk.addWidget(QLabel("Epochs:", gb_knobs), 1, 0)
        gk.addWidget(self.sp_epochs, 1, 1)
        gk.addWidget(QLabel("LR:", gb_knobs), 2, 0)
        gk.addWidget(self.sp_lr, 2, 1)
        gk.addWidget(self.chk_dry, 3, 0, 1, 2)

        ll.addWidget(gb_knobs)
        ll.addStretch(1)

        split.addWidget(left_scroll)

        # ==============================================================
        # Right: preview + notes
        # ==============================================================
        right = QFrame(self)
        right.setObjectName("scriptGenRight")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        self._lbl_preview = QLabel("Generated script", right)
        self._lbl_preview.setObjectName("sectionTitle")

        self._chip_hint = _Chip("Tip: Press Copy / Save", right)
        self._chip_hint.setToolTip(
            "These scripts are intended for reproducible runs."
        )

        hdr.addWidget(self._lbl_preview)
        hdr.addWidget(self._chip_hint)
        hdr.addStretch(1)

        rl.addLayout(hdr)

        self.preview = QPlainTextEdit(right)
        self.preview.setObjectName("codePreview")
        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(280)
        rl.addWidget(self.preview, 1)

        self.notes = QPlainTextEdit(right)
        self.notes.setObjectName("scriptNotes")
        self.notes.setPlaceholderText(
            "Notes (optional): what this script is for, "
            "paper figure, ablation tag, etc."
        )
        self.notes.setMinimumHeight(110)
        rl.addWidget(self.notes, 0)

        split.addWidget(right)
        left_scroll.setMinimumWidth(420)

        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([520, 820])

    def _connect_ui(self) -> None:
        self.cmb_preset.currentIndexChanged.connect(
            self._on_preset_changed
        )
        self.cmb_shell.currentIndexChanged.connect(
            self._render_preview
        )

        for w in (
            self.edit_cities,
            self.sp_repeats,
            self.chk_timestamped,
            self.edit_results_root,
            self.edit_manifest,
            self.sp_batch,
            self.sp_epochs,
            self.sp_lr,
            self.chk_dry,
            self.notes,
            self.edit_ablations, 
            self.chk_grid, 
            self.edit_seeds
        ):
            if isinstance(w, QLineEdit):
                w.textChanged.connect(self._render_preview)
            elif isinstance(w, QPlainTextEdit):
                w.textChanged.connect(self._render_preview)
            else:
                try:
                    w.valueChanged.connect(self._render_preview)
                except Exception:
                    pass
                try:
                    w.toggled.connect(self._render_preview)
                except Exception:
                    pass

        self.btn_browse_root.clicked.connect(
            self._browse_root
        )
        self.btn_browse_manifest.clicked.connect(
            self._browse_manifest
        )

        self._btn_copy.clicked.connect(self._copy_script)
        self._btn_save.clicked.connect(self._save_script)
        self._act_export_bundle.triggered.connect(
            self._export_bundle
        )
        self._act_export_make.triggered.connect(
            self._export_makefile
        )
        self._act_export_slurm.triggered.connect(
            self._export_slurm
        )

        self._btn_load_tpl.clicked.connect(
            self._load_template_from_manifest
        )

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def _on_preset_changed(self) -> None:
        key = self.cmb_preset.currentData()
        if isinstance(key, str):
            self._apply_preset(key)

    def _apply_preset(self, key: str) -> None:
        p = next((x for x in self._presets if x.key == key), None)
        if p is None:
            return
        self.lbl_preset_desc.setText(p.desc)
        self._chip_target.setText(f"Preset: {key}")
        self._render_preview()
        
    def _export_makefile(self) -> None:
        p = self._collect_params()
        jobs = self._expand_jobs(p)
        cmds = self._build_commands(p["preset"], p, jobs)

        out, _ = QFileDialog.getSaveFileName(
            self,
            "Save Makefile",
            str(Path.home() / "Makefile"),
            "Makefile (Makefile);;All files (*.*)",
        )
        if not out:
            return

        txt = self._make_makefile(cmds)
        try:
            Path(out).write_text(txt, encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Export failed",
                f"{exc}",
            )

    def _make_makefile(
        self,
        cmds: List[Tuple[str, str]],
    ) -> str:
        def _tname(s: str) -> str:
            ok = []
            for ch in s:
                if ch.isalnum() or ch in ("_", "-"):
                    ok.append(ch)
                else:
                    ok.append("_")
            return "".join(ok)

        tnames = [_tname(n) for n, _ in cmds]
        lines: List[str] = []
        lines.append(".PHONY: all " + " ".join(tnames))
        lines.append("")
        lines.append("all: " + " ".join(tnames))
        lines.append("")

        for (name, cmd), t in zip(cmds, tnames):
            lines.append(f"{t}:")
            lines.append(f'\t@echo "== {name} =="')
            lines.append(f"\t@{cmd}")
            lines.append("")

        return "\n".join(lines)
    
    def _export_slurm(self) -> None:
        p = self._collect_params()
        jobs = self._expand_jobs(p)
        cmds = self._build_commands(p["preset"], p, jobs)

        name = "geoprior.sbatch"
        out, _ = QFileDialog.getSaveFileName(
            self,
            "Save Slurm sbatch script",
            str(Path.home() / name),
            "Shell (*.sh *.sbatch);;All files (*.*)",
        )
        if not out:
            return

        txt = self._make_slurm(cmds)
        try:
            Path(out).write_text(txt, encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Export failed",
                f"{exc}",
            )

    def _make_slurm(
        self,
        cmds: List[Tuple[str, str]],
    ) -> str:
        n = max(len(cmds), 1)
        lines: List[str] = []
        lines.append("#!/bin/bash")
        lines.append("#SBATCH --job-name=geoprior")
        lines.append("#SBATCH --time=04:00:00")
        lines.append("#SBATCH --cpus-per-task=4")
        lines.append("#SBATCH --mem=16G")
        lines.append("#SBATCH --gres=gpu:1")
        lines.append(f"#SBATCH --array=0-{n-1}")
        lines.append("")
        lines.append("set -e")
        lines.append("")
        lines.append("CMDS=(")
        for _name, cmd in cmds:
            cmd2 = cmd.replace('"', '\\"')
            lines.append(f'  "{cmd2}"')
        lines.append(")")
        lines.append("")
        lines.append('IDX="${SLURM_ARRAY_TASK_ID:-0}"')
        lines.append('echo "IDX=$IDX"')
        lines.append('eval "${CMDS[$IDX]}"')
        lines.append("")
        return "\n".join(lines)
        
    def _load_template_from_manifest(self) -> None:
        start = self.edit_manifest.text().strip()
        if not start:
            start = str(Path.home())

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load run manifest (JSON)",
            start,
            "JSON (*.json);;All files (*.*)",
        )
        if not path:
            return

        data = self._safe_read_json(Path(path))
        if data is None:
            QMessageBox.warning(
                self,
                "Manifest error",
                "Could not read JSON manifest.",
            )
            return

        self.edit_manifest.setText(path)

        preset = self._infer_preset(Path(path), data)
        if preset:
            self._set_preset_key(preset)

        rr = self._pick_cfg(
            data,
            keys=("results_root", "RESULTS_ROOT"),
        )
        if rr and not self.edit_results_root.text().strip():
            self.edit_results_root.setText(str(rr))

        city = self._pick_cfg(
            data,
            keys=("CITY_NAME", "city", "city_name"),
        )
        if city:
            cur = self.edit_cities.text().strip()
            if not cur:
                self.edit_cities.setText(str(city))

        bs = self._pick_cfg(
            data,
            keys=("BATCH_SIZE", "batch_size"),
        )
        if bs is not None:
            try:
                self.sp_batch.setValue(int(bs))
            except Exception:
                pass

        ep = self._pick_cfg(
            data,
            keys=("EPOCHS", "epochs", "max_epochs"),
        )
        if ep is not None:
            try:
                self.sp_epochs.setValue(int(ep))
            except Exception:
                pass

        lr = self._pick_cfg(
            data,
            keys=("LR", "learning_rate", "lr"),
        )
        if lr is not None:
            try:
                self.sp_lr.setValue(float(lr))
            except Exception:
                pass

        self._chip_hint.setText("Loaded from manifest")
        self._render_preview()

    def _safe_read_json(
        self,
        path: Path,
    ) -> Optional[Dict[str, Any]]:
        try:
            txt = path.read_text(encoding="utf-8")
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    def _pick_cfg(
        self,
        data: Dict[str, Any],
        *,
        keys: Tuple[str, ...],
    ) -> Optional[Any]:
        """
        Try common manifest layouts:
        - top-level
        - data["config"]
        - data["cfg"]
        - data["params"]
        """
        pools: List[Dict[str, Any]] = [data]
        for k in ("config", "cfg", "params"):
            v = data.get(k)
            if isinstance(v, dict):
                pools.append(v)

        for pool in pools:
            for kk in keys:
                if kk in pool:
                    return pool.get(kk)
        return None

    def _infer_preset(
        self,
        path: Path,
        data: Dict[str, Any],
    ) -> str:
        name = path.name.lower()
        if "tuning" in name or "run_" in name:
            return "tune"
        if "inference" in name:
            return "infer"
        if "train" in name:
            return "train"
        if "stage1" in name:
            return "stage1"

        stage = self._pick_cfg(
            data,
            keys=("stage", "STAGE", "kind", "KIND"),
        )
        if isinstance(stage, str):
            s = stage.lower()
            if "tune" in s:
                return "tune"
            if "infer" in s:
                return "infer"
            if "train" in s:
                return "train"
            if "stage1" in s:
                return "stage1"

        return "stage1"

    def _set_preset_key(self, key: str) -> None:
        for i in range(self.cmb_preset.count()):
            if self.cmb_preset.itemData(i) == key:
                self.cmb_preset.setCurrentIndex(i)
                return

    # ------------------------------------------------------------------
    # Preview generation (UI-only)
    # ------------------------------------------------------------------
    def _collect_params(self) -> Dict[str, Any]:
        cities = [
            c.strip()
            for c in (self.edit_cities.text() or "").split(",")
            if c.strip()
        ]
        return {
            "preset": str(self.cmb_preset.currentData() or "stage1"),
            "format": str(self.cmb_shell.currentData() or "bash"),
            "cities": cities,
            "repeats": int(self.sp_repeats.value()),
            "timestamped": bool(self.chk_timestamped.isChecked()),
            "results_root": (self.edit_results_root.text() or "").strip(),
            "manifest": (self.edit_manifest.text() or "").strip(),
            "batch_size": int(self.sp_batch.value()),
            "epochs": int(self.sp_epochs.value()),
            "lr": float(self.sp_lr.value()),
            "dry_run": bool(self.chk_dry.isChecked()),
            "notes": (self.notes.toPlainText() or "").strip(),
            "grid_on": bool(self.chk_grid.isChecked()),
            "seeds": (self.edit_seeds.text() or "").strip(),
            "ablations": (self.edit_ablations.toPlainText()
                          or "").strip(),

        }
    def _parse_list(self, s: str) -> List[str]:
        out: List[str] = []
        for x in (s or "").split(","):
            t = x.strip()
            if t:
                out.append(t)
        return out

    def _parse_ablations(
        self,
        txt: str,
    ) -> List[Tuple[str, str]]:
        """
        Lines:  label | extra args
        If no '|', label is used and args empty.
        """
        out: List[Tuple[str, str]] = []
        for raw in (txt or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                a, b = line.split("|", 1)
                lab = a.strip() or "ablation"
                args = b.strip()
            else:
                lab = line
                args = ""
            out.append((lab, args))
        if not out:
            out.append(("baseline", ""))
        return out

    def _expand_jobs(
        self,
        p: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        cities = p.get("cities") or ["<city>"]
        reps = int(p.get("repeats") or 1)

        grid_on = bool(p.get("grid_on", False))
        seeds = self._parse_list(str(p.get("seeds") or ""))
        if not seeds:
            seeds = ["0"]

        abls = self._parse_ablations(str(p.get("ablations") or ""))

        jobs: List[Dict[str, Any]] = []
        for city in cities:
            for r in range(reps):
                if not grid_on:
                    jobs.append(
                        {
                            "city": city,
                            "seed": "",
                            "abl_lab": "",
                            "abl_args": "",
                            "rep": r,
                        }
                    )
                    continue

                for seed in seeds:
                    for lab, args in abls:
                        jobs.append(
                            {
                                "city": city,
                                "seed": seed,
                                "abl_lab": lab,
                                "abl_args": args,
                                "rep": r,
                            }
                        )

        self._chip_jobs.setText(f"Jobs: {len(jobs)}")
        return jobs

    def _render_preview(self) -> None:
        p = self._collect_params()
        fmt = p["format"]
        preset = p["preset"]

        script = self._make_script(fmt, preset, p)
        self.preview.setPlainText(script)

    def _make_script(
        self,
        fmt: str,
        preset: str,
        p: Dict[str, Any],
    ) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cities = p.get("cities") or ["<city>"]

        hdr = [
            "# Auto-generated by GeoPrior – Script / batch generator",
            f"# preset={preset} format={fmt} generated={stamp}",
        ]
        if p.get("notes"):
            hdr.append("#")
            for line in str(p["notes"]).splitlines():
                hdr.append("# " + line)

        # rr = p.get("results_root") or ""
        # mf = p.get("manifest") or ""
        # dry = bool(p.get("dry_run", False))

        # These are UI-only: later we’ll map to real CLI commands.
        # For now, keep them as clean “templates” for the user.
        jobs = self._expand_jobs(p)
        cmds = self._build_commands(preset, p, jobs)

        if fmt == "ps1":
            return self._as_ps1_list(hdr, cmds)
        if fmt == "py":
            return self._as_py_list(hdr, cmds)
        return self._as_bash_list(hdr, cmds)
    
        # if fmt == "ps1":
        #     return self._as_ps1(
        #         hdr,
        #         preset,
        #         cities,
        #         rr,
        #         mf,
        #         dry,
        #         p,
        #     )
        # if fmt == "py":
        #     return self._as_py(
        #         hdr,
        #         preset,
        #         cities,
        #         rr,
        #         mf,
        #         dry,
        #         p,
        #     )

        # return self._as_bash(
        #     hdr,
        #     preset,
        #     cities,
        #     rr,
        #     mf,
        #     dry,
        #     p,
        # )
        
    def _build_commands(
        self,
        preset: str,
        p: Dict[str, Any],
        jobs: List[Dict[str, Any]],
    ) -> List[Tuple[str, str]]:
        """
        Returns list of (label, cmd).
        """
        rr = (p.get("results_root") or "").strip()
        mf = (p.get("manifest") or "").strip()
        dry = bool(p.get("dry_run", False))

        base = "python -m fusionlab.tools.app.geoprior.cli"

        rr_arg = f' --results-root "{rr}"' if rr else ""
        mf_arg = f' --manifest "{mf}"' if mf else ""
        dr_arg = " --dry-run" if dry else ""

        out: List[Tuple[str, str]] = []

        for j in jobs:
            city = j["city"]
            seed = j.get("seed") or ""
            lab = j.get("abl_lab") or ""
            abl = j.get("abl_args") or ""
            rep = int(j.get("rep") or 0)

            tag = [city]
            if seed:
                tag.append(f"s{seed}")
            if lab:
                tag.append(lab)
            if rep:
                tag.append(f"r{rep}")

            name = "_".join(tag)

            extra = ""
            if seed:
                extra += f" --seed {seed}"
            if abl:
                extra += " " + abl.strip()

            if preset == "stage1":
                cmd = (
                    f"{base} stage1 --city {city}"
                    f"{rr_arg}{dr_arg}"
                )
            elif preset == "train":
                cmd = (
                    f"{base} train{mf_arg}{rr_arg}"
                    f" --epochs {p['epochs']}"
                    f" --batch-size {p['batch_size']}"
                    f" --lr {p['lr']}{extra}{dr_arg}"
                )
            elif preset == "tune":
                cmd = (
                    f"{base} tune --city {city}{rr_arg}"
                    f" --max-epochs {p['epochs']}"
                    f" --batch-size {p['batch_size']}"
                    f"{extra}{dr_arg}"
                )
            elif preset == "infer":
                cmd = (
                    f"{base} infer{rr_arg} --dataset test"
                    f" --batch-size {p['batch_size']}"
                    f"{extra}{dr_arg}"
                )
            else:
                cmd = (
                    f"{base} xfer-matrix{rr_arg}"
                    " --city-a <city_a> --city-b <city_b>"
                    f"{extra}{dr_arg}"
                )

            out.append((name, cmd))

        return out

    def _as_bash_list(
        self,
        hdr: List[str],
        cmds: List[Tuple[str, str]],
    ) -> str:
        lines = list(hdr)
        lines += ["", "set -e", ""]
        for name, cmd in cmds:
            lines.append(f'echo "== {name} =="')
            lines.append(cmd)
            lines.append("")
        return "\n".join(lines)

    def _as_ps1_list(
        self,
        hdr: List[str],
        cmds: List[Tuple[str, str]],
    ) -> str:
        lines = list(hdr)
        lines.append("")
        for name, cmd in cmds:
            lines.append(f'Write-Host "== {name} =="')
            lines.append(cmd)
            lines.append("")
        return "\n".join(lines)

    def _as_py_list(
        self,
        hdr: List[str],
        cmds: List[Tuple[str, str]],
    ) -> str:
        lines = list(hdr)
        lines += [
            "",
            "from __future__ import annotations",
            "",
            "import subprocess",
            "",
            "def run(cmd: str) -> None:",
            "    print(cmd)",
            "    subprocess.check_call(cmd, shell=True)",
            "",
        ]
        for name, cmd in cmds:
            lines.append(f'print("== {name} ==")')
            lines.append(f'run(r"""{cmd}""")')
            lines.append("")
        return "\n".join(lines)

    def _as_bash(
        self,
        hdr: List[str],
        preset: str,
        cities: List[str],
        results_root: str,
        manifest: str,
        dry: bool,
        p: Dict[str, Any],
    ) -> str:
        lines = list(hdr)
        lines.append("")
        lines.append("set -e")
        lines.append("")
        if results_root:
            lines.append(f'RESULTS_ROOT="{results_root}"')
        if manifest:
            lines.append(f'MANIFEST="{manifest}"')
        lines.append("")

        # template commands (replace later with real entrypoints)
        cmd = "python -m fusionlab.tools.app.geoprior.cli"
        if preset == "stage1":
            for c in cities:
                lines.append(
                    "{cmd} stage1 --city {c}"
                    + (' --results-root "$RESULTS_ROOT"' if results_root else "")
                    + (" --dry-run" if dry else "")
                )
        elif preset == "train":
            lines.append(
                f"{cmd} train"
                + (' --manifest "$MANIFEST"' if manifest else "")
                + (' --results-root "$RESULTS_ROOT"' if results_root else "")
                + f" --epochs {p['epochs']} --batch-size {p['batch_size']}"
                + (f" --lr {p['lr']}" if p.get("lr") else "")
                + (" --dry-run" if dry else "")
            )
        elif preset == "tune":
            for c in cities:
                lines.append(
                    f"{cmd} tune --city {c}"
                    + (' --results-root "$RESULTS_ROOT"' if results_root else "")
                    + f" --max-epochs {p['epochs']} --batch-size {p['batch_size']}"
                    + (" --dry-run" if dry else "")
                )
        elif preset == "infer":
            lines.append(
                f"{cmd} infer"
                + (' --results-root "$RESULTS_ROOT"' if results_root else "")
                + " --dataset test"
                + f" --batch-size {p['batch_size']}"
                + (" --dry-run" if dry else "")
            )
        else:  # xfer
            lines.append(
                f"{cmd} xfer-matrix"
                + (' --results-root "$RESULTS_ROOT"' if results_root else "")
                + " --city-a <city_a> --city-b <city_b>"
                + (" --dry-run" if dry else "")
            )

        return "\n".join(lines)

    def _as_ps1(
        self,
        hdr: List[str],
        preset: str,
        cities: List[str],
        results_root: str,
        manifest: str,
        dry: bool,
        p: Dict[str, Any],
    ) -> str:
        lines = [l.replace("#", "#") for l in hdr]
        lines.append("")
        if results_root:
            lines.append(f'$RESULTS_ROOT = "{results_root}"')
        if manifest:
            lines.append(f'$MANIFEST = "{manifest}"')
        lines.append("")

        cmd = "python -m fusionlab.tools.app.geoprior.cli"
        if preset == "stage1":
            for c in cities:
                lines.append(
                    f"{cmd} stage1 --city {c}"
                    + (" --results-root $RESULTS_ROOT" if results_root else "")
                    + (" --dry-run" if dry else "")
                )
        elif preset == "train":
            lines.append(
                f"{cmd} train"
                + (" --manifest $MANIFEST" if manifest else "")
                + (" --results-root $RESULTS_ROOT" if results_root else "")
                + f" --epochs {p['epochs']} --batch-size {p['batch_size']}"
                + (f" --lr {p['lr']}" if p.get("lr") else "")
                + (" --dry-run" if dry else "")
            )
        elif preset == "tune":
            for c in cities:
                lines.append(
                    f"{cmd} tune --city {c}"
                    + (" --results-root $RESULTS_ROOT" if results_root else "")
                    + f" --max-epochs {p['epochs']} --batch-size {p['batch_size']}"
                    + (" --dry-run" if dry else "")
                )
        elif preset == "infer":
            lines.append(
                f"{cmd} infer"
                + (" --results-root $RESULTS_ROOT" if results_root else "")
                + " --dataset test"
                + f" --batch-size {p['batch_size']}"
                + (" --dry-run" if dry else "")
            )
        else:
            lines.append(
                f"{cmd} xfer-matrix"
                + (" --results-root $RESULTS_ROOT" if results_root else "")
                + " --city-a <city_a> --city-b <city_b>"
                + (" --dry-run" if dry else "")
            )

        return "\n".join(lines)

    def _as_py(
        self,
        hdr: List[str],
        preset: str,
        cities: List[str],
        results_root: str,
        manifest: str,
        dry: bool,
        p: Dict[str, Any],
    ) -> str:
        lines = [l.replace("#", "#") for l in hdr]
        lines.append("")
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append("import subprocess")
        lines.append("")
        lines.append("def run(cmd: str) -> None:")
        lines.append("    print(cmd)")
        lines.append("    subprocess.check_call(cmd, shell=True)")
        lines.append("")
        rr = f' --results-root "{results_root}"' if results_root else ""
        mf = f' --manifest "{manifest}"' if manifest else ""
        dr = " --dry-run" if dry else ""

        base = 'python -m fusionlab.tools.app.geoprior.cli'
        if preset == "stage1":
            for c in cities:
                lines.append(
                    f'run("{base} stage1 --city {c}{rr}{dr}")'
                )
        elif preset == "train":
            lines.append(
                f'run("{base} train{mf}{rr} --epochs {p["epochs"]}'
                f' --batch-size {p["batch_size"]} --lr {p["lr"]}{dr}")'
            )
        elif preset == "tune":
            for c in cities:
                lines.append(
                    f'run("{base} tune --city {c}{rr}'
                    f' --max-epochs {p["epochs"]} --batch-size {p["batch_size"]}{dr}")'
                )
        elif preset == "infer":
            lines.append(
                f'run("{base} infer{rr} --dataset test --batch-size {p["batch_size"]}{dr}")'
            )
        else:
            lines.append(
                f'run("{base} xfer-matrix{rr} --city-a <city_a> --city-b <city_b>{dr}")'
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _browse_root(self) -> None:
        start = self.edit_results_root.text().strip() or str(Path.home())
        p = QFileDialog.getExistingDirectory(
            self,
            "Select results root",
            start,
        )
        if not p:
            return
        self.edit_results_root.setText(p)

    def _browse_manifest(self) -> None:
        start = self.edit_manifest.text().strip() or str(Path.home())
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select manifest/config",
            start,
            "JSON (*.json);;All files (*.*)",
        )
        if not p:
            return
        self.edit_manifest.setText(p)

    def _copy_script(self) -> None:
        txt = self.preview.toPlainText()
        if not txt.strip():
            return
        QApplication.clipboard().setText(txt)

    def _save_script(self) -> None:
        txt = self.preview.toPlainText()
        if not txt.strip():
            return

        fmt = str(self.cmb_shell.currentData() or "bash")
        ext = {"bash": "sh", "ps1": "ps1", "py": "py"}.get(fmt, "txt")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        preset = str(self.cmb_preset.currentData() or "stage1")
        name = f"geoprior_{preset}_{stamp}.{ext}"

        out, _ = QFileDialog.getSaveFileName(
            self,
            "Save script",
            str(Path.home() / name),
            "All files (*.*)",
        )
        if not out:
            return

        try:
            Path(out).write_text(txt, encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Save failed",
                f"Failed to save:\n{out}\n\n{exc}",
            )

    def _export_bundle(self) -> None:
        """
        Export a small bundle:
        - script.<ext>
        - params.json (UI params snapshot)
        """
        txt = self.preview.toPlainText()
        if not txt.strip():
            return

        fmt = str(self.cmb_shell.currentData() or "bash")
        ext = {"bash": "sh", "ps1": "ps1", "py": "py"}.get(fmt, "txt")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        preset = str(self.cmb_preset.currentData() or "stage1")
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select export folder",
            str(Path.home()),
        )
        if not folder:
            return

        out_dir = Path(folder) / f"geoprior_bundle_{preset}_{stamp}"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"script.{ext}").write_text(
                txt, encoding="utf-8"
            )
            params = self._collect_params()
            (out_dir / "params.json").write_text(
                json.dumps(params, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Export failed",
                f"Failed to export bundle:\n{out_dir}\n\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Exported",
            f"Saved bundle to:\n{out_dir}",
        )
