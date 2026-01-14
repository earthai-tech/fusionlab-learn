# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
stage1_workspace.run_history

Stage-1 run history panel.

- Lists Stage-1 runs under results root.
- Lets user select one and mark it as active.
- View-only: emits signals for the controller to act.

Scanning strategy:
- Find directories under runs_root that contain "stage1"
- Prefer those containing a manifest.json at dir root
- Read manifest fields: timestamp, city, model, schema_version
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

Json = Dict[str, Any]


@dataclass(frozen=True)
class Stage1RunEntry:
    run_dir: str
    manifest_path: str
    timestamp: str
    city: str
    model: str
    stage: str
    schema_version: str
    has_audit: bool
    mtime_epoch: float


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _safe_read_json(path: Path) -> Optional[Json]:
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        obj = json.loads(txt)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_timestamp(s: str) -> Optional[dt.datetime]:
    """
    Best-effort parse:
    - "YYYY-MM-DD HH:MM:SS"
    - "YYYYMMDD-HHMMSS"
    """
    s = (s or "").strip()
    if not s:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d-%H%M%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def _scan_stage1_runs(
    *,
    runs_root: str,
    city: str = "",
    model: str = "",
) -> List[Stage1RunEntry]:
    root = Path(runs_root) if runs_root else None
    if root is None or not root.exists() or not root.is_dir():
        return []

    city_l = (city or "").strip().lower()
    model_l = (model or "").strip().lower()

    entries: List[Stage1RunEntry] = []

    for p in root.iterdir():
        if not p.is_dir():
            continue

        name_l = p.name.lower()
        if "stage1" not in name_l:
            continue

        if city_l and not name_l.startswith(city_l + "_"):
            continue

        if model_l and model_l not in name_l:
            continue

        manifest = p / "manifest.json"
        if not manifest.exists():
            continue

        m = _safe_read_json(manifest) or {}
        ts = _as_str(m.get("timestamp", "")).strip()

        cc = _as_str(m.get("city", "")).strip()
        mm = _as_str(m.get("model", "")).strip()
        st = _as_str(m.get("stage", "")).strip()
        sv = _as_str(m.get("schema_version", "")).strip()

        if not cc:
            cc = city or ""
        if not mm:
            mm = model or ""
        if not st:
            st = "stage1"
        if not sv:
            sv = "-"

        audit = p / "stage1_scaling_audit.json"
        has_audit = audit.exists()

        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0

        entries.append(
            Stage1RunEntry(
                run_dir=str(p),
                manifest_path=str(manifest),
                timestamp=ts or "-",
                city=cc or "-",
                model=mm or "-",
                stage=st or "-",
                schema_version=sv,
                has_audit=bool(has_audit),
                mtime_epoch=float(mtime),
            )
        )

    def key(e: Stage1RunEntry) -> Tuple[float, float]:
        t = _parse_timestamp(e.timestamp)
        if t is not None:
            return (t.timestamp(), e.mtime_epoch)
        return (0.0, e.mtime_epoch)

    entries.sort(key=key, reverse=True)
    return entries


class Stage1RunHistory(QWidget):
    """
    Stage-1 run history panel.

    Signals
    -------
    request_refresh():
        Ask controller to refresh the list (optional).
    request_set_active(run_dir, manifest_path):
        Ask controller to activate a selected run.
    request_open_path(path):
        Ask controller to open a file/folder.
    """

    request_refresh = pyqtSignal()
    request_set_active = pyqtSignal(str, str)
    request_open_path = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._runs_root: str = ""
        self._city: str = ""
        self._model: str = ""

        self._entries: List[Stage1RunEntry] = []
        self._build_ui()
        self._wire()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._runs_root = ""
        self._city = ""
        self._model = ""
        self._entries = []
        self._filter.setText("")
        self._tree.clear()
        self._details.setHtml("")

    def set_context(
        self,
        *,
        runs_root: str,
        city: str = "",
        model: str = "",
    ) -> None:
        self._runs_root = _as_str(runs_root).strip()
        self._city = _as_str(city).strip()
        self._model = _as_str(model).strip()
        self._lbl_ctx.setText(self._ctx_text())

    def set_entries(self, entries: Optional[List[Stage1RunEntry]]) -> None:
        self._entries = list(entries or [])
        self._rebuild_tree()

    def refresh_scan(self) -> None:
        """
        Local scan (fast). Controller can also own this.
        """
        self.set_entries(
            _scan_stage1_runs(
                runs_root=self._runs_root,
                city=self._city,
                model=self._model,
            )
        )

    def selected_entry(self) -> Optional[Stage1RunEntry]:
        it = self._tree.currentItem()
        if it is None:
            return None
        data = it.data(0, Qt.UserRole)
        if not isinstance(data, dict):
            return None
        idx = data.get("idx")
        if not isinstance(idx, int):
            return None
        if idx < 0 or idx >= len(self._entries):
            return None
        return self._entries[idx]

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(8)

        self._lbl_title = QLabel("Stage-1 Run history")
        self._lbl_title.setObjectName("stage1RunHistoryTitle")

        self._lbl_ctx = QLabel("")
        self._lbl_ctx.setObjectName("stage1RunHistoryCtx")
        self._lbl_ctx.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter...")
        self._btn_scan = QPushButton("Scan")
        self._btn_refresh = QPushButton("Refresh")

        self._btn_set_active = QPushButton("Set active")
        self._btn_open_manifest = QPushButton("Open manifest")
        self._btn_open_folder = QPushButton("Open folder")

        top.addWidget(self._lbl_title)
        top.addStretch(1)
        top.addWidget(self._filter, 1)
        top.addWidget(self._btn_scan)
        top.addWidget(self._btn_refresh)

        lay.addLayout(top)
        lay.addWidget(self._lbl_ctx)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(
            [
                "Timestamp",
                "City",
                "Model",
                "Schema",
                "Audit",
                "Path",
            ]
        )
        self._tree.setUniformRowHeights(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(False)
        self._tree.setContextMenuPolicy(
            Qt.CustomContextMenu
        )

        btns = QHBoxLayout()
        btns.setSpacing(8)
        btns.addWidget(self._btn_set_active)
        btns.addWidget(self._btn_open_manifest)
        btns.addWidget(self._btn_open_folder)
        btns.addStretch(1)

        self._details = QTextBrowser()
        self._details.setOpenExternalLinks(True)

        lay.addWidget(self._tree, 1)
        lay.addLayout(btns)
        lay.addWidget(self._details, 1)

    def _wire(self) -> None:
        self._btn_scan.clicked.connect(self.refresh_scan)
        self._btn_refresh.clicked.connect(
            self.request_refresh.emit
        )
        self._filter.textChanged.connect(self._apply_filter)

        self._tree.currentItemChanged.connect(
            self._on_selection_changed
        )
        self._tree.itemDoubleClicked.connect(
            self._on_double_click
        )
        self._tree.customContextMenuRequested.connect(
            self._on_context_menu
        )

        self._btn_set_active.clicked.connect(
            self._emit_set_active
        )
        self._btn_open_manifest.clicked.connect(
            self._open_manifest
        )
        self._btn_open_folder.clicked.connect(
            self._open_folder
        )

    # ------------------------------------------------------------------
    # Behavior
    # ------------------------------------------------------------------
    def _ctx_text(self) -> str:
        rr = self._runs_root or "-"
        cc = self._city or "-"
        mm = self._model or "-"
        return f"Root: {rr} | City: {cc} | Model: {mm}"

    def _rebuild_tree(self) -> None:
        self._tree.clear()

        for i, e in enumerate(self._entries):
            audit = "yes" if e.has_audit else "no"
            it = QTreeWidgetItem(
                [
                    e.timestamp,
                    e.city,
                    e.model,
                    e.schema_version,
                    audit,
                    e.run_dir,
                ]
            )
            it.setData(0, Qt.UserRole, {"idx": i})
            self._tree.addTopLevelItem(it)

        for c in range(self._tree.columnCount()):
            self._tree.resizeColumnToContents(c)

        self._apply_filter(self._filter.text())
        self._render_details(self.selected_entry())

    def _apply_filter(self, text: str) -> None:
        pat = (text or "").strip().lower()
        for i in range(self._tree.topLevelItemCount()):
            it = self._tree.topLevelItem(i)
            row = " ".join(it.text(c) for c in range(6))
            hit = (pat in row.lower()) if pat else True
            it.setHidden(not hit)

    def _on_selection_changed(self, *_args) -> None:
        self._render_details(self.selected_entry())

    def _render_details(self, e: Optional[Stage1RunEntry]) -> None:
        if e is None:
            self._details.setHtml(
                "<p>No run selected.</p>"
            )
            return

        def row(k: str, v: str) -> str:
            return (
                "<tr>"
                "<td style='padding:2px 10px 2px 0;"
                "font-weight:600;white-space:nowrap;'>"
                f"{k}</td>"
                f"<td style='padding:2px 0;'>{v}</td>"
                "</tr>"
            )

        html = []
        html.append("<html><body>")
        html.append("<h3>Selected run</h3>")
        html.append("<table>")
        html.append(row("timestamp", e.timestamp))
        html.append(row("city", e.city))
        html.append(row("model", e.model))
        html.append(row("stage", e.stage))
        html.append(row("schema_version", e.schema_version))
        html.append(row("audit", "yes" if e.has_audit else "no"))
        html.append(row("run_dir", e.run_dir))
        html.append(row("manifest", e.manifest_path))
        html.append("</table>")
        html.append("</body></html>")
        self._details.setHtml("".join(html))

    def _on_double_click(
        self,
        item: QTreeWidgetItem,
        _col: int,
    ) -> None:
        data = item.data(0, Qt.UserRole) or {}
        idx = data.get("idx")
        if not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._entries):
            return
        self.request_open_path.emit(
            self._entries[idx].run_dir
        )

    def _emit_set_active(self) -> None:
        e = self.selected_entry()
        if e is None:
            return
        self.request_set_active.emit(
            e.run_dir,
            e.manifest_path,
        )

    def _open_manifest(self) -> None:
        e = self.selected_entry()
        if e is None:
            return
        self.request_open_path.emit(e.manifest_path)

    def _open_folder(self) -> None:
        e = self.selected_entry()
        if e is None:
            return
        self.request_open_path.emit(e.run_dir)

    def _on_context_menu(self, pos) -> None:
        it = self._tree.itemAt(pos)
        if it is None:
            return

        data = it.data(0, Qt.UserRole) or {}
        idx = data.get("idx")
        if not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._entries):
            return

        e = self._entries[idx]

        menu = QMenu(self)
        act_active = menu.addAction("Set active")
        act_open = menu.addAction("Open folder")
        act_manifest = menu.addAction("Open manifest")
        menu.addSeparator()
        act_copy = menu.addAction("Copy path")

        chosen = menu.exec_(QCursor.pos())
        if chosen is None:
            return

        if chosen == act_active:
            self.request_set_active.emit(
                e.run_dir,
                e.manifest_path,
            )
            return

        if chosen == act_open:
            self.request_open_path.emit(e.run_dir)
            return

        if chosen == act_manifest:
            self.request_open_path.emit(e.manifest_path)
            return

        if chosen == act_copy:
            cb = QApplication.clipboard()
            if cb is not None:
                cb.setText(e.run_dir)
            return
