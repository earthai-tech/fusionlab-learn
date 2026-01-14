# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
stage1_workspace.artifacts

Artifacts panel for Stage-1 (preprocess) workspace.

This widget is view-only:
- controller pushes manifest + optional extra paths
- widget renders a grouped tree of artifact paths
- double-click can emit a signal to open the path

Primary source:
- manifest.json (dict), specifically:
  - manifest["paths"]["run_dir"]
  - manifest["paths"]["artifacts_dir"]
  - manifest["artifacts"][...]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

Json = Dict[str, Any]
PathLike = Union[str, Path]


@dataclass
class ArtifactsContext:
    city: str = ""
    csv_path: str = ""
    runs_root: str = ""
    stage1_dir: str = ""


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _norm_path(p: Any) -> str:
    s = _as_str(p).strip()
    return s


def _to_file_url(p: PathLike) -> str:
    s = _as_str(p).strip()
    if not s:
        return ""
    try:
        return QUrl.fromLocalFile(s).toString()
    except Exception:
        return ""


def _get(d: Json, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def _is_pathlike(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, (str, Path)):
        return bool(_as_str(x).strip())
    return False


def _looks_like_path(s: str) -> bool:
    if not s:
        return False
    if ":\\" in s or ":/" in s:
        return True
    if s.startswith("\\\\"):
        return True
    if s.startswith("/"):
        return True
    if "/" in s or "\\" in s:
        return True
    return False


class Stage1Artifacts(QWidget):
    """
    Stage-1 artifacts panel (tree view).

    Public API:
    - clear()
    - set_context(...)
    - set_manifest(...)
    - set_extra_items(...)
    - set_status(...)

    Signals:
    - request_open_path(str)
    """

    request_open_path = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._ctx = ArtifactsContext()
        self._manifest: Optional[Json] = None
        self._extra: List[Tuple[str, str]] = []
        self._status: str = ""

        self._build_ui()
        self._rebuild_tree()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._ctx = ArtifactsContext()
        self._manifest = None
        self._extra = []
        self._status = ""
        self._filter.setText("")
        self._rebuild_tree()

    def set_context(
        self,
        *,
        city: str,
        csv_path: Optional[PathLike],
        runs_root: Optional[PathLike],
        stage1_dir: Optional[PathLike] = None,
    ) -> None:
        self._ctx = ArtifactsContext(
            city=_as_str(city),
            csv_path=_as_str(csv_path),
            runs_root=_as_str(runs_root),
            stage1_dir=_as_str(stage1_dir),
        )
        self._rebuild_tree()

    def set_status(self, text: str) -> None:
        self._status = _as_str(text)
        self._lbl_status.setText(self._status or "")

    def set_manifest(self, manifest: Optional[Json]) -> None:
        self._manifest = manifest if isinstance(manifest, dict) else None
        self._rebuild_tree()

    def set_extra_items(
        self,
        items: Optional[List[Tuple[str, str]]],
    ) -> None:
        self._extra = list(items or [])
        self._rebuild_tree()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(10)

        lbl = QLabel("Stage-1 Artifacts")
        lbl.setObjectName("stage1ArtifactsTitle")

        self._lbl_status = QLabel("")
        self._lbl_status.setObjectName("stage1ArtifactsStatus")

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter artifacts...")
        self._filter.textChanged.connect(self._apply_filter)

        top.addWidget(lbl)
        top.addStretch(1)
        top.addWidget(self._filter, 1)
        top.addWidget(self._lbl_status)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Artifact", "Path"])
        self._tree.setUniformRowHeights(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemDoubleClicked.connect(
            self._on_item_double_clicked
        )
        self._tree.setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self._tree.customContextMenuRequested.connect(
            self._on_context_menu
        )

        layout.addLayout(top)
        layout.addWidget(self._tree, 1)

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------
    def _rebuild_tree(self) -> None:
        self._lbl_status.setText(self._status or "")
        self._tree.clear()

        groups = self._collect_groups()
        for g_name, items in groups:
            g_it = QTreeWidgetItem([g_name, ""])
            g_it.setFirstColumnSpanned(True)
            g_it.setData(0, Qt.UserRole, {"group": True})
            self._tree.addTopLevelItem(g_it)

            for label, path in items:
                it = QTreeWidgetItem([label, path])
                it.setData(
                    0,
                    Qt.UserRole,
                    {"path": path, "label": label},
                )
                g_it.addChild(it)

            g_it.setExpanded(True)

        self._apply_filter(self._filter.text())

    def _collect_groups(self) -> List[Tuple[str, List[Tuple[str, str]]]]:
        m = self._manifest if isinstance(self._manifest, dict) else {}

        run_dir = _norm_path(_get(m, "paths", "run_dir", default=""))
        art_dir = _norm_path(
            _get(m, "paths", "artifacts_dir", default="")
        )

        groups: List[Tuple[str, List[Tuple[str, str]]]] = []

        # Group: Run
        run_items: List[Tuple[str, str]] = []
        if run_dir:
            run_items.append(("run_dir", run_dir))
        if art_dir:
            run_items.append(("artifacts_dir", art_dir))

        # If controller didn't pass manifest_path, best-effort.
        s1_dir = self._ctx.stage1_dir.strip()
        if s1_dir and not run_dir:
            run_items.append(("stage1_dir", s1_dir))

        # Common top-level files by convention (optional).
        if s1_dir:
            man = str(Path(s1_dir) / "manifest.json")
            run_items.append(("manifest.json", man))

            aud = str(Path(s1_dir) / "stage1_scaling_audit.json")
            run_items.append(("stage1_scaling_audit.json", aud))

        groups.append(("Run", run_items))

        # Group: CSV
        csv_items: List[Tuple[str, str]] = []
        for k in ("raw", "clean", "scaled"):
            p = _norm_path(_get(m, "artifacts", "csv", k, default=""))
            if p:
                csv_items.append((k, p))
        if csv_items:
            groups.append(("CSV", csv_items))

        # Group: Encoders / scalers
        enc_items: List[Tuple[str, str]] = []
        enc = _get(m, "artifacts", "encoders", default={}) or {}

        # Known paths
        for k in ("coord_scaler", "main_scaler"):
            p = _norm_path(enc.get(k, ""))
            if p:
                enc_items.append((k, p))

        # OHE dict
        ohe = enc.get("ohe")
        if isinstance(ohe, dict):
            for k, p in sorted(ohe.items(), key=lambda kv: kv[0]):
                pp = _norm_path(p)
                if pp:
                    enc_items.append((f"ohe:{k}", pp))

        if enc_items:
            groups.append(("Encoders", enc_items))

        # Group: Sequences
        seq_items: List[Tuple[str, str]] = []
        seq = _get(m, "artifacts", "sequences", default={}) or {}
        p = _norm_path(seq.get("joblib_train_sequences", ""))
        if p:
            seq_items.append(("train_sequences", p))
        if seq_items:
            groups.append(("Sequences", seq_items))

        # Group: Numpy
        np_items: List[Tuple[str, str]] = []
        npz = _get(m, "artifacts", "numpy", default={}) or {}
        for k, p in sorted(npz.items(), key=lambda kv: kv[0]):
            pp = _norm_path(p)
            if pp:
                np_items.append((k, pp))
        if np_items:
            groups.append(("Numpy", np_items))

        # Group: Extra (controller-driven)
        extra_items: List[Tuple[str, str]] = []
        for label, path in self._extra:
            ll = _as_str(label).strip()
            pp = _norm_path(path)
            if ll and pp:
                extra_items.append((ll, pp))
        if extra_items:
            groups.append(("Extra", extra_items))

        # Ensure we always show something.
        if not any(items for _, items in groups):
            groups = [("Run", [])]

        return groups

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _on_item_double_clicked(
        self,
        item: QTreeWidgetItem,
        col: int,
    ) -> None:
        data = item.data(0, Qt.UserRole) or {}
        path = _as_str(data.get("path", "")).strip()
        if not path:
            return
        if not _looks_like_path(path):
            return
        self.request_open_path.emit(path)

    def _on_context_menu(self, pos) -> None:
        it = self._tree.itemAt(pos)
        if it is None:
            return
        data = it.data(0, Qt.UserRole) or {}
        path = _as_str(data.get("path", "")).strip()
        if not path or not _looks_like_path(path):
            return

        menu = QMenu(self)
        act_open = menu.addAction("Open")
        act_copy = menu.addAction("Copy path")
        act_open_dir = menu.addAction("Open folder")

        chosen = menu.exec_(QCursor.pos())
        if chosen is None:
            return

        if chosen == act_open:
            self.request_open_path.emit(path)
            return

        if chosen == act_copy:
            cb = self.clipboard()
            if cb is not None:
                cb.setText(path)
            return

        if chosen == act_open_dir:
            folder = self._containing_dir(path)
            if folder:
                self.request_open_path.emit(folder)
            return

    def _containing_dir(self, path: str) -> str:
        p = Path(path)
        try:
            if p.is_dir():
                return str(p)
        except Exception:
            return ""
        try:
            return str(p.parent)
        except Exception:
            return ""

    def _apply_filter(self, text: str) -> None:
        pat = (text or "").strip().lower()
        if not pat:
            self._show_all()
            return

        for i in range(self._tree.topLevelItemCount()):
            g = self._tree.topLevelItem(i)
            g_visible = False
            for j in range(g.childCount()):
                ch = g.child(j)
                label = _as_str(ch.text(0)).lower()
                path = _as_str(ch.text(1)).lower()
                hit = pat in label or pat in path
                ch.setHidden(not hit)
                if hit:
                    g_visible = True
            g.setHidden(not g_visible)

    def _show_all(self) -> None:
        for i in range(self._tree.topLevelItemCount()):
            g = self._tree.topLevelItem(i)
            g.setHidden(False)
            for j in range(g.childCount()):
                g.child(j).setHidden(False)
