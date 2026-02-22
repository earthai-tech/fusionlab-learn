# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Manifest browser & validator for GeoPrior GUI Tools tab.
#
# Lets the user:
#   - browse Stage-1 / train / tune / inference / xfer manifests
#     under the current results root;
#   - inspect their JSON contents;
#   - run lightweight integrity checks on artifact paths and
#     library versions;
#   - see compact metrics + tiny ASCII sparklines per run.

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QBrush
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QPlainTextEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QMessageBox,
)

# Optional imports for version checks (guarded below)
try:  # pragma: no cover - environment dependent
    import tensorflow as _tf  # type: ignore
except Exception:  # pragma: no cover
    _tf = None

try:  # pragma: no cover
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None

try:  # pragma: no cover
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover
    _pd = None

try:  # pragma: no cover
    import sklearn as _sk  # type: ignore
except Exception:  # pragma: no cover
    _sk = None


@dataclass
class ManifestEntry:
    """
    Lightweight descriptor for a manifest / summary JSON on disk.
    """

    city: str
    kind: str                     # "stage1", "train", "tune", "infer", "xfer"
    name: str                     # short run id (folder name / label)
    root_dir: Path
    json_path: Optional[Path]
    stage: Optional[str] = None
    model: Optional[str] = None
    timestamp: Optional[str] = None

    # Filled later from JSON
    problems: List[str] = None  # type: ignore[assignment]
    warnings: List[str] = None  # type: ignore[assignment]
    metric_summary: Optional[str] = None
    sparkline: Optional[str] = None
    loss: Optional[float] = None
    r2: Optional[float] = None

    def __post_init__(self) -> None:
        if self.problems is None:
            self.problems = []
        if self.warnings is None:
            self.warnings = []


class ManifestBrowserTool(QWidget):
    """
    Manifest browser & validator for GeoPrior results.

    Left: hierarchical tree of runs grouped by city and kind, with a
    compact metrics column + tiny ASCII sparklines.

    Right: summary panel, integrity checks list, and raw JSON viewer.
    """

    SPARK_CHARS = "▁▂▃▄▅▆▇█"

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_ctx = app_ctx
        self._results_root: Path = self._guess_results_root()
        self._entries: List[ManifestEntry] = []

        self._init_ui()
        self._reload_from_context()

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

        # --- Top: root + controls -------------------------------------
        top_group = QGroupBox("Manifest browser & validator", self)
        top_layout = QHBoxLayout(top_group)
        top_layout.setContentsMargins(8, 6, 8, 6)
        top_layout.setSpacing(6)

        self._root_lbl = QLabel(self)
        self._root_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._root_lbl.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )

        tagline = QLabel(
            "<i>Browse Stage-1 / train / tune / inference / xfer manifests, "
            "see metrics, and run lightweight integrity checks.</i>",
            self,
        )
        tagline.setTextFormat(Qt.RichText)

        left_top = QVBoxLayout()
        left_top.addWidget(self._root_lbl)
        left_top.addWidget(tagline)

        btn_reload = QPushButton("Reload", self)
        btn_reload.clicked.connect(self._reload_from_context)

        btn_validate_all = QPushButton("Validate all", self)
        btn_validate_all.clicked.connect(self._on_validate_all)

        top_layout.addLayout(left_top, stretch=1)
        top_layout.addWidget(btn_reload)
        top_layout.addWidget(btn_validate_all)

        layout.addWidget(top_group)

        # --- Middle: tree + details -----------------------------------
        mid_layout = QHBoxLayout()
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(8)

        # Left: tree
        left_group = QGroupBox("Available manifests", self)
        left_v = QVBoxLayout(left_group)
        left_v.setContentsMargins(8, 6, 8, 6)
        left_v.setSpacing(4)

        # 3 columns now: Run / city | Kind | Metrics
        self._tree = QTreeWidget(self)
        self._tree.setHeaderLabels(["Run / city", "Kind", "Metrics"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setSelectionMode(QTreeWidget.SingleSelection)
        self._tree.itemSelectionChanged.connect(
            self._on_tree_selection_changed
        )

        left_v.addWidget(self._tree)

        # Right: details
        right_group = QGroupBox("Manifest details", self)
        right_v = QVBoxLayout(right_group)
        right_v.setContentsMargins(8, 6, 8, 6)
        right_v.setSpacing(6)

        # Summary box
        summary_group = QGroupBox("Summary", self)
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setContentsMargins(6, 4, 6, 4)

        self._summary_lbl = QLabel(
            "Select a run on the left to see details.", self
        )
        self._summary_lbl.setWordWrap(True)
        self._summary_lbl.setTextFormat(Qt.RichText)
        summary_layout.addWidget(self._summary_lbl)

        # Integrity box
        integrity_group = QGroupBox("Integrity checks", self)
        integrity_layout = QVBoxLayout(integrity_group)
        integrity_layout.setContentsMargins(6, 4, 6, 4)

        self._issues_list = QListWidget(self)
        self._issues_list.setAlternatingRowColors(True)
        integrity_layout.addWidget(self._issues_list)

        # Raw JSON viewer
        json_group = QGroupBox("Raw JSON (read-only)", self)
        json_layout = QVBoxLayout(json_group)
        json_layout.setContentsMargins(6, 4, 6, 4)

        self._json_edit = QPlainTextEdit(self)
        self._json_edit.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self._json_edit.setFont(mono)
        self._json_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        json_layout.addWidget(self._json_edit)

        # Lower action row (for current manifest)
        action_row = QHBoxLayout()
        action_row.setSpacing(6)

        self._btn_open_dir = QPushButton("Open folder", self)
        self._btn_open_dir.clicked.connect(self._on_open_dir_clicked)

        self._btn_copy_path = QPushButton("Copy JSON path", self)
        self._btn_copy_path.clicked.connect(self._on_copy_path_clicked)

        action_row.addWidget(self._btn_open_dir)
        action_row.addWidget(self._btn_copy_path)
        action_row.addStretch(1)

        right_v.addWidget(summary_group)
        right_v.addWidget(integrity_group, stretch=1)
        right_v.addWidget(json_group, stretch=2)
        right_v.addLayout(action_row)

        mid_layout.addWidget(left_group, stretch=1)
        mid_layout.addWidget(right_group, stretch=2)

        layout.addLayout(mid_layout)

    # ------------------------------------------------------------------
    # Reload / scanning logic
    # ------------------------------------------------------------------
    def _reload_from_context(self) -> None:
        self._results_root = self._guess_results_root()
        self._root_lbl.setText(
            f"<b>Results root:</b> "
            f"<span style='color: palette(mid);'>{self._results_root}</span>"
        )
        self._scan_results_root()
        self._populate_tree()
        self._clear_details()

    def _scan_results_root(self) -> None:
        """
        Discover Stage-1 / train / tune / inference / xfer manifests
        under the current results_root.
        """
        self._entries.clear()

        root = self._results_root
        if not root.is_dir():
            return

        for stage1_dir in root.glob("*_stage1"):
            if not stage1_dir.is_dir():
                continue

            stage1_manifest_path = stage1_dir / "manifest.json"
            city, model, ts = self._peek_stage1_meta(stage1_manifest_path)
            if not city:
                name = stage1_dir.name
                if name.endswith("_stage1"):
                    city = name.rsplit("_stage1", 1)[0]
                else:
                    city = name

            # Stage-1 entry
            self._entries.append(
                ManifestEntry(
                    city=city,
                    kind="stage1",
                    name="Stage-1",
                    root_dir=stage1_dir,
                    json_path=(
                        stage1_manifest_path
                        if stage1_manifest_path.is_file()
                        else None
                    ),
                    stage="stage1",
                    model=model,
                    timestamp=ts,
                )
            )

            # Training runs
            for train_dir in stage1_dir.glob("train_*"):
                if not train_dir.is_dir():
                    continue
                train_manifest = train_dir / "manifest.json"
                t_city, t_model, t_ts = self._peek_generic_meta(
                    train_manifest
                )
                self._entries.append(
                    ManifestEntry(
                        city=t_city or city,
                        kind="train",
                        name=train_dir.name,
                        root_dir=train_dir,
                        json_path=(
                            train_manifest if train_manifest.is_file() else None
                        ),
                        stage="stage2",
                        model=t_model or model,
                        timestamp=t_ts,
                    )
                )

            # Tuning runs
            tuning_root = stage1_dir / "tuning"
            if tuning_root.is_dir():
                for run_dir in tuning_root.glob("run_*"):
                    if not run_dir.is_dir():
                        continue
                    summary = run_dir / "tuning_summary.json"
                    t_city, t_model, t_ts = self._peek_generic_meta(summary)
                    self._entries.append(
                        ManifestEntry(
                            city=t_city or city,
                            kind="tune",
                            name=run_dir.name,
                            root_dir=run_dir,
                            json_path=summary if summary.is_file() else None,
                            stage="tuning",
                            model=t_model or model,
                            timestamp=t_ts,
                        )
                    )

            # Inference runs
            inf_root = stage1_dir / "inference"
            if inf_root.is_dir():
                for run_dir in inf_root.iterdir():
                    if not run_dir.is_dir():
                        continue
                    summary = run_dir / "inference_summary.json"
                    i_city, i_model, i_ts = self._peek_generic_meta(summary)
                    self._entries.append(
                        ManifestEntry(
                            city=i_city or city,
                            kind="infer",
                            name=run_dir.name,
                            root_dir=run_dir,
                            json_path=summary if summary.is_file() else None,
                            stage="inference",
                            model=i_model or model,
                            timestamp=i_ts,
                        )
                    )

            # Xfer runs
            xfer_root = stage1_dir / "xfer"
            if xfer_root.is_dir():
                for pair_dir in xfer_root.iterdir():
                    if not pair_dir.is_dir():
                        continue
                    for run_dir in pair_dir.iterdir():
                        if not run_dir.is_dir():
                            continue
                        summary = run_dir / "xfer_results.json"
                        x_city, x_model, x_ts = self._peek_generic_meta(
                            summary
                        )
                        name = f"{pair_dir.name}/{run_dir.name}"
                        self._entries.append(
                            ManifestEntry(
                                city=x_city or city,
                                kind="xfer",
                                name=name,
                                root_dir=run_dir,
                                json_path=summary if summary.is_file() else None,
                                stage="xfer",
                                model=x_model or model,
                                timestamp=x_ts,
                            )
                        )

        # After discovering entries, pre-compute metric summaries/sparklines
        self._populate_metrics_for_entries()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _peek_stage1_meta(
        self, path: Path
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not path.is_file():
            return None, None, None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None, None, None

        city = data.get("city")
        model = data.get("model")
        ts = data.get("timestamp")
        return city, model, ts

    def _peek_generic_meta(
        self, path: Path
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not path.is_file():
            return None, None, None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None, None, None

        city = data.get("city") or data.get("source_city")
        model = data.get("model") or data.get("model_name")
        ts = data.get("timestamp") or data.get("run_timestamp")
        return city, model, ts

    # ------------------------------------------------------------------
    # Metrics / sparkline helpers
    # ------------------------------------------------------------------
    def _populate_metrics_for_entries(self) -> None:
        """
        Pre-load JSON for each entry (if small enough) and fill
        metric_summary + sparkline fields.
        """
        for e in self._entries:
            if not e.json_path or not e.json_path.is_file():
                continue

            try:
                with e.json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            metrics = data.get("metrics_evaluate", {}) or {}
            if not isinstance(metrics, dict):
                metrics = {}

            r2 = metrics.get("r2") or metrics.get("r2_score")
            loss = metrics.get("loss")
            mae = metrics.get("mae")
            cov80 = metrics.get("subs_pred_coverage80") or metrics.get(
                "coverage80"
            )

            e.loss = float(loss) if isinstance(loss, (int, float)) else None
            e.r2 = float(r2) if isinstance(r2, (int, float)) else None

            parts: List[str] = []
            if e.loss is not None:
                parts.append(f"loss={e.loss:.3f}")
            if e.r2 is not None:
                parts.append(f"R²={e.r2:.3f}")
            if isinstance(mae, (int, float)):
                parts.append(f"MAE={float(mae):.3f}")
            if isinstance(cov80, (int, float)):
                parts.append(f"cov80={float(cov80):.3f}")

            e.metric_summary = ", ".join(parts) if parts else None

            # Try to build a loss sparkline from any history traces
            history = (
                data.get("history")
                or data.get("train_history")
                or data.get("loss_history")
            )
            seq: Optional[List[float]] = None

            if isinstance(history, dict):
                for key in ("loss", "train_loss", "total_loss"):
                    vals = history.get(key)
                    if isinstance(vals, list):
                        seq = [float(v) for v in vals if isinstance(v, (int, float))]
                        if seq:
                            break
            elif isinstance(history, list):
                seq = [float(v) for v in history if isinstance(v, (int, float))]

            if seq:
                e.sparkline = self._make_sparkline(seq)
            else:
                e.sparkline = None

    def _make_sparkline(
        self,
        seq: List[float],
        max_len: int = 10,
    ) -> str:
        """
        Map a numeric sequence to a tiny Unicode sparkline.
        """
        if not seq:
            return ""

        # Downsample if needed
        values = list(seq)
        if len(values) > max_len:
            step = len(values) / float(max_len)
            values = [values[int(i * step)] for i in range(max_len)]

        # Filter NaNs
        values = [
            v for v in values if isinstance(v, (int, float)) and not math.isnan(v)
        ]
        if not values:
            return ""

        lo, hi = min(values), max(values)
        if hi == lo:
            ch = self.SPARK_CHARS[len(self.SPARK_CHARS) // 2]
            return ch * len(values)

        chars = []
        for v in values:
            x = (v - lo) / (hi - lo)
            idx = int(round(x * (len(self.SPARK_CHARS) - 1)))
            idx = max(0, min(idx, len(self.SPARK_CHARS) - 1))
            chars.append(self.SPARK_CHARS[idx])
        return "".join(chars)

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------
    def _populate_tree(self) -> None:
        self._tree.clear()
        if not self._entries:
            return

        by_city: Dict[str, Dict[str, List[ManifestEntry]]] = {}
        for e in self._entries:
            city = e.city or "unknown"
            by_city.setdefault(city, {})
            by_city[city].setdefault(e.kind, []).append(e)

        for city in sorted(by_city.keys()):
            city_item = QTreeWidgetItem([city, "city", ""])
            city_item.setFirstColumnSpanned(True)
            self._tree.addTopLevelItem(city_item)

            kinds_order = ["stage1", "train", "tune", "infer", "xfer"]
            labels = {
                "stage1": "Stage-1",
                "train": "Training runs",
                "tune": "Tuning runs",
                "infer": "Inference runs",
                "xfer": "Transfer runs",
            }

            for kind in kinds_order:
                runs = by_city[city].get(kind)
                if not runs:
                    continue

                group_label = f"{labels[kind]} ({len(runs)})"
                group_item = QTreeWidgetItem([group_label, "", ""])
                group_item.setFirstColumnSpanned(True)
                city_item.addChild(group_item)

                for e in sorted(
                    runs,
                    key=lambda x: x.timestamp or x.name,
                    reverse=True,
                ):
                    kind_label = {
                        "stage1": "stage-1",
                        "train": "train",
                        "tune": "tune",
                        "infer": "infer",
                        "xfer": "xfer",
                    }.get(kind, kind)

                    # Metrics column: summary + sparkline
                    metric_bits: List[str] = []
                    if e.metric_summary:
                        metric_bits.append(e.metric_summary)
                    if e.sparkline:
                        metric_bits.append(e.sparkline)
                    metrics_str = "  |  ".join(metric_bits)

                    item = QTreeWidgetItem([e.name, kind_label, metrics_str])
                    item.setData(0, Qt.UserRole, e)

                    # Light color cue if R² is good / bad
                    if e.r2 is not None:
                        if e.r2 >= 0.9:
                            item.setForeground(2, QBrush(Qt.darkGreen))
                        elif e.r2 < 0.5:
                            item.setForeground(2, QBrush(Qt.darkRed))

                    group_item.addChild(item)

        self._tree.expandAll()

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------
    def _on_tree_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            self._clear_details()
            return

        item = items[0]
        entry = item.data(0, Qt.UserRole)
        if not isinstance(entry, ManifestEntry):
            self._clear_details()
            return

        self._show_entry(entry)

    def _clear_details(self) -> None:
        self._summary_lbl.setText(
            "Select a run on the left to see details."
        )
        self._issues_list.clear()
        self._json_edit.setPlainText("")
        self._btn_open_dir.setEnabled(False)
        self._btn_copy_path.setEnabled(False)

    def _show_entry(self, entry: ManifestEntry) -> None:
        self._issues_list.clear()
        self._json_edit.setPlainText("")
        self._btn_open_dir.setEnabled(True)
        self._btn_copy_path.setEnabled(bool(entry.json_path))

        data: Dict[str, Any] = {}
        if entry.json_path and entry.json_path.is_file():
            try:
                with entry.json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                pretty = json.dumps(data, indent=2, ensure_ascii=False)
                self._json_edit.setPlainText(pretty)
            except Exception as exc:
                self._json_edit.setPlainText(
                    f"Could not read JSON:\n{exc}"
                )
        else:
            self._json_edit.setPlainText(
                "No JSON manifest/summary found for this run."
            )

        self._summary_lbl.setText(self._build_summary_html(entry, data))

        problems, warnings = self._validate_entry(entry, data)
        entry.problems = problems
        entry.warnings = warnings

        if not problems and not warnings:
            ok_item = QListWidgetItem("OK – no issues detected.")
            self._issues_list.addItem(ok_item)
            return

        for msg in problems:
            itm = QListWidgetItem(f"ERROR: {msg}")
            itm.setForeground(QBrush(Qt.red))
            self._issues_list.addItem(itm)

        for msg in warnings:
            itm = QListWidgetItem(f"WARN: {msg}")
            itm.setForeground(QBrush(Qt.darkYellow))
            self._issues_list.addItem(itm)

    # ------------------------------------------------------------------
    # Summary + validation
    # ------------------------------------------------------------------
    def _build_summary_html(
        self, entry: ManifestEntry, data: Dict[str, Any]
    ) -> str:
        city = entry.city or "unknown"
        kind = entry.kind
        stage = entry.stage or data.get("stage") or "?"
        model = entry.model or data.get("model") or "GeoPriorSubsNet"
        ts = entry.timestamp or data.get("timestamp") or "?"

        cfg = data.get("config", {}) or {}
        features = cfg.get("features", {}) or {}
        static_feats = features.get("static") or []
        dynamic_feats = features.get("dynamic") or []
        future_feats = features.get("future") or []

        metrics = data.get("metrics_evaluate", {}) or {}
        r2 = metrics.get("r2") or metrics.get("r2_score")
        mae = metrics.get("mae")
        mse = metrics.get("mse")
        cov80 = metrics.get("subs_pred_coverage80") or metrics.get(
            "coverage80"
        )

        bits = [
            f"<b>City:</b> {city}",
            f"<b>Kind:</b> {kind}",
            f"<b>Stage:</b> {stage}",
            f"<b>Model:</b> {model}",
            f"<b>Timestamp:</b> {ts}",
        ]

        if kind in {"stage1", "train", "tune"} and features:
            bits.append(
                "<b>Features:</b> "
                f"{len(static_feats)} static, "
                f"{len(dynamic_feats)} dynamic, "
                f"{len(future_feats)} future"
            )

        metric_bits: List[str] = []
        if r2 is not None:
            metric_bits.append(f"R²={float(r2):.3f}")
        if mae is not None:
            metric_bits.append(f"MAE={float(mae):.3f}")
        if mse is not None:
            metric_bits.append(f"MSE={float(mse):.3f}")
        if cov80 is not None:
            try:
                metric_bits.append(f"cov80={float(cov80):.3f}")
            except Exception:
                pass

        if metric_bits:
            bits.append("<b>Metrics:</b> " + ", ".join(metric_bits))

        path_str = (
            str(entry.json_path)
            if entry.json_path is not None
            else "(no JSON)"
        )
        bits.append(
            "<b>JSON:</b> "
            f"<span style='color: palette(mid);'>{path_str}</span>"
        )

        if entry.sparkline:
            bits.append(
                "<b>Loss sparkline:</b> "
                f"<code>{entry.sparkline}</code>"
            )

        return "<br>".join(bits)

    def _validate_entry(
        self, entry: ManifestEntry, data: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        problems: List[str] = []
        warnings: List[str] = []

        if not entry.root_dir.is_dir():
            problems.append(f"Run directory does not exist: {entry.root_dir}")
        if not entry.json_path or not entry.json_path.is_file():
            problems.append("Manifest/summary JSON file is missing.")

        artifacts = data.get("artifacts", {}) or {}
        paths_section = data.get("paths", {}) or {}

        candidate_paths: List[str] = []

        def _collect_paths(obj: Any) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    _collect_paths(v)
            elif isinstance(obj, list):
                for v in obj:
                    _collect_paths(v)
            elif isinstance(obj, str):
                if (
                    any(sep in obj for sep in ("/", "\\"))
                    and "." in Path(obj).name
                ):
                    candidate_paths.append(obj)

        _collect_paths(artifacts)
        _collect_paths(paths_section)

        missing_paths: List[str] = []
        for p in candidate_paths:
            try:
                if not os.path.exists(p):
                    missing_paths.append(p)
            except Exception:
                continue

        seen: set[str] = set()
        for p in missing_paths:
            if p in seen:
                continue
            seen.add(p)
            problems.append(f"Artifact path not found: {p}")

        versions = data.get("versions", {}) or {}

        def _check_ver(key: str, runtime: Optional[str]) -> None:
            saved = versions.get(key)
            if not saved or not runtime:
                return
            if str(saved) != str(runtime):
                warnings.append(
                    f"{key} version mismatch: run={saved}, "
                    f"current={runtime}"
                )

        if _tf is not None:  # pragma: no cover
            _check_ver("tensorflow", getattr(_tf, "__version__", None))
        if _np is not None:  # pragma: no cover
            _check_ver("numpy", getattr(_np, "__version__", None))
        if _pd is not None:  # pragma: no cover
            _check_ver("pandas", getattr(_pd, "__version__", None))
        if _sk is not None:  # pragma: no cover
            _check_ver("sklearn", getattr(_sk, "__version__", None))

        if entry.kind == "stage1":
            cfg = data.get("config", {}) or {}
            feats = cfg.get("features") or {}
            if not feats:
                warnings.append(
                    "Stage-1 manifest has no 'features' section."
                )
            cols = cfg.get("cols") or {}
            for key in ("time", "lon", "lat", "subsidence", "gwl", "h_field"):
                if key not in cols:
                    warnings.append(
                        f"Stage-1 'cols' is missing '{key}'."
                    )

        return problems, warnings

    # ------------------------------------------------------------------
    # Validate all button
    # ------------------------------------------------------------------
    def _on_validate_all(self) -> None:
        if not self._entries:
            QMessageBox.information(
                self,
                "No manifests",
                "No manifests were found under the current results root.",
            )
            return

        n_total = len(self._entries)
        n_ok = 0
        n_warn = 0
        n_err = 0

        for e in self._entries:
            data: Dict[str, Any] = {}
            if e.json_path and e.json_path.is_file():
                try:
                    with e.json_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    pass

            problems, warnings = self._validate_entry(e, data)
            e.problems = problems
            e.warnings = warnings

            if problems:
                n_err += 1
            elif warnings:
                n_warn += 1
            else:
                n_ok += 1

        msg = (
            f"Validated {n_total} manifest(s):\n"
            f"  OK     : {n_ok}\n"
            f"  Warnings: {n_warn}\n"
            f"  Errors : {n_err}"
        )
        QMessageBox.information(self, "Validation summary", msg)

    # ------------------------------------------------------------------
    # Per-entry actions (open dir / copy path)
    # ------------------------------------------------------------------
    def _current_entry(self) -> Optional[ManifestEntry]:
        items = self._tree.selectedItems()
        if not items:
            return None
        item = items[0]
        entry = item.data(0, Qt.UserRole)
        if isinstance(entry, ManifestEntry):
            return entry
        return None

    def _on_open_dir_clicked(self) -> None:
        entry = self._current_entry()
        if entry is None:
            return
        run_dir = entry.root_dir
        if not run_dir.is_dir():
            QMessageBox.warning(
                self,
                "Folder not found",
                f"Run directory does not exist:\n{run_dir}",
            )
            return

        try:
            if os.name == "nt":  # Windows
                os.startfile(str(run_dir))  # type: ignore[attr-defined]
            elif os.name == "posix":
                if "darwin" in os.sys.platform:
                    os.system(f'open "{run_dir}"')
                else:
                    os.system(f'xdg-open "{run_dir}"')
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Could not open folder",
                f"Failed to open folder:\n{run_dir}\n\n{exc}",
            )

    def _on_copy_path_clicked(self) -> None:
        entry = self._current_entry()
        if entry is None or not entry.json_path:
            return
        from PyQt5.QtWidgets import QApplication

        cb = QApplication.clipboard()
        cb.setText(str(entry.json_path))
        QMessageBox.information(
            self,
            "Path copied",
            f"JSON path copied to clipboard:\n{entry.json_path}",
        )
