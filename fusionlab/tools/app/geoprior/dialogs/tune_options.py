# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import copy
import json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
    QToolButton,
    QMenu,
    QAction,
)

from ..config.prior_schema import FieldKey
from ..config.store import GeoConfigStore
from ..device_options import DeviceOptionsWidget
from .json_display import JsonAuditDialog,  JsonManifestDialog

MANIFEST_ROLE = Qt.UserRole
AUDIT_ROLE = Qt.UserRole + 1


@dataclass(frozen=True)
class TuneJobSpec:
    city: str
    stage1_manifest: Path


@dataclass(frozen=True)
class _Stage1Run:
    city: str
    timestamp: str
    time_steps: Optional[int]
    horizon_years: Optional[int]
    manifest: Path


class TuneOptionsDialog(QDialog):
    """
    Store-driven advanced tuning dialog (v3.2).

    Layout:
      [ Results root ]
      [ Device options (2-col inside widget) ]
      [ Stage-1 picker (filter + table + actions) ]
    """

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tune options")

        self._store = store
        self._snap_cfg = _cfg_copy(store.cfg)

        self._queued: Optional[TuneJobSpec] = None

        self._build_ui()
        self._wire_ui()

        self._populate_stage1()
        self._sync_actions()

    # -----------------------------
    # Entry point
    # -----------------------------
    @classmethod
    def edit(
        cls,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> Tuple[bool, Optional[TuneJobSpec]]:
        dlg = cls(store=store, parent=parent)
        ok = dlg.exec_() == QDialog.Accepted
        return ok, dlg._queued

    # -----------------------------
    # Snapshot semantics
    # -----------------------------
    def reject(self) -> None:
        try:
            self._store.replace_config(self._snap_cfg)
        except Exception:
            pass
        super().reject()

    # -----------------------------
    # UI
    # -----------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ---------------------------------
        # Results root (full width)
        # ---------------------------------
        root_box = QGroupBox("Results root")
        root_l = QHBoxLayout(root_box)
        root_l.setContentsMargins(8, 8, 8, 8)
        root_l.setSpacing(8)

        self._le_root = QLineEdit(self._read_root_text())
        self._le_root.setPlaceholderText(
            "Folder containing Stage-1 city runs"
        )

        self._btn_browse = QPushButton("Browse…")

        root_l.addWidget(self._le_root, 1)
        root_l.addWidget(self._btn_browse)

        root.addWidget(root_box)

        # ---------------------------------
        # Device options (store-driven)
        # ---------------------------------
        self.device_widget = DeviceOptionsWidget(
            store=self._store,
            parent=self,
        )
        root.addWidget(self.device_widget)

        # ---------------------------------
        # Stage-1 picker (modern)
        # ---------------------------------
        st1_box = QGroupBox("Stage-1 runs")
        st1_v = QVBoxLayout(st1_box)
        st1_v.setContentsMargins(8, 8, 8, 8)
        st1_v.setSpacing(8)

        bar = QHBoxLayout()
        bar.setSpacing(8)

        self._lbl_count = QLabel("Found: 0")
        self._ed_filter = QLineEdit()
        self._ed_filter.setPlaceholderText(
            "Filter by city or timestamp…"
        )

        self._btn_refresh = QPushButton("Refresh")

        bar.addWidget(self._lbl_count)
        bar.addSpacing(10)
        bar.addWidget(self._ed_filter, 1)
        bar.addWidget(self._btn_refresh)

        st1_v.addLayout(bar)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(5)
        self._tree.setHeaderLabels(["City", "Timestamp", "T", "H", "Audit"])

        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setUniformRowHeights(True)

        hdr = self._tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )
        hdr.setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )
        hdr.setSectionResizeMode(
            2,
            QHeaderView.ResizeToContents,
        )
        hdr.setSectionResizeMode(
            3,
            QHeaderView.ResizeToContents,
        )
        hdr.setSectionResizeMode(
            4,
            QHeaderView.ResizeToContents,
        )

        st1_v.addWidget(self._tree, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)

        self._btn_details = QToolButton()
        self._btn_details.setText("Details")
        self._btn_details.setPopupMode(QToolButton.MenuButtonPopup)
        
        self._details_menu = QMenu(self._btn_details)
        self._act_view_manifest = QAction("View manifest", self)
        self._act_view_audit = QAction("View scaling audit", self)
        self._details_menu.addAction(self._act_view_manifest)
        self._details_menu.addAction(self._act_view_audit)
        
        self._btn_details.setMenu(self._details_menu)
        
        self._btn_run = QPushButton("Run selected")
        
        actions.addStretch(1)
        actions.addWidget(self._btn_details)
        actions.addWidget(self._btn_run)

        st1_v.addLayout(actions)

        root.addWidget(st1_box, 1)

        # ---------------------------------
        # Bottom buttons
        # ---------------------------------
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
        )
        self._btns = btns
        root.addWidget(btns)

    def _wire_ui(self) -> None:
        self._btns.accepted.connect(self.accept)
        self._btns.rejected.connect(self.reject)

        self._le_root.editingFinished.connect(
            self._apply_root_and_refresh
        )
        self._btn_browse.clicked.connect(
            self._on_browse_root
        )

        self._btn_refresh.clicked.connect(
            self._populate_stage1
        )
        self._ed_filter.textChanged.connect(
            self._apply_filter
        )

        self._tree.currentItemChanged.connect(
            lambda *_: self._sync_actions()
        )
        # double-click: open manifest (best default)
        self._tree.itemDoubleClicked.connect(
            lambda *_: self._open_manifest_selected()
        )
        
        # main click on the split button: manifest
        self._btn_details.clicked.connect(self._open_manifest_selected)
        
        # menu actions
        self._act_view_manifest.triggered.connect(self._open_manifest_selected)
        self._act_view_audit.triggered.connect(self._open_audit_selected)

        self._btn_run.clicked.connect(self._on_run_sel)

    # -----------------------------
    # Root handling
    # -----------------------------
    def _read_root_text(self) -> str:
        cfg = self._store.cfg
        root = getattr(cfg, "results_root", None)
        return str(root or "")

    def _on_browse_root(self) -> None:
        cur = self._le_root.text().strip()
        new_dir = QFileDialog.getExistingDirectory(
            self,
            "Select results root",
            cur or str(Path.home()),
        )
        if not new_dir:
            return
        self._le_root.setText(new_dir)
        self._apply_root_and_refresh()

    def _apply_root_and_refresh(self) -> None:
        root = self._le_root.text().strip()
        if not root:
            return

        try:
            self._store.set_value_by_key(
                FieldKey("results_root"),
                root,
            )
        except Exception:
            pass

        self._populate_stage1()

    # -----------------------------
    # Stage-1 list
    # -----------------------------
    def _populate_stage1(self) -> None:
        root = Path(self._le_root.text().strip()).expanduser()
        runs = _discover_stage1(root)

        self._tree.setUpdatesEnabled(False)
        try:
            self._tree.clear()

            for r in runs:
                it = QTreeWidgetItem(
                    [
                        r.city,
                        r.timestamp,
                        str(r.time_steps or ""),
                        str(r.horizon_years or ""),
                    ]
                )
                it.setTextAlignment(4, Qt.AlignCenter)
                audit = _find_stage1_audit_json(r.manifest)

                it.setData(0, MANIFEST_ROLE, str(r.manifest))
                it.setData(0, AUDIT_ROLE, str(audit) if audit else "")
                
                it.setText(4, "✓" if audit else "")
                it.setToolTip(4, str(audit) if audit else "No audit found")
                
                self._tree.addTopLevelItem(it)

            self._lbl_count.setText(
                f"Found: {len(runs)}"
            )

        finally:
            self._tree.setUpdatesEnabled(True)

        self._apply_filter(self._ed_filter.text())
        self._sync_actions()

    def _apply_filter(self, text: str) -> None:
        t = (text or "").strip().lower()

        for i in range(self._tree.topLevelItemCount()):
            it = self._tree.topLevelItem(i)
            hay = (
                (it.text(0) + " " + it.text(1))
                .lower()
            )
            it.setHidden(bool(t) and (t not in hay))

    def _sync_actions(self) -> None:
        ok = self._tree.currentItem() is not None
        audit_ok = self._selected_audit() is not None
        
        self._btn_details.setEnabled(ok)
        self._btn_run.setEnabled(ok)
        
        # audit menu item only enabled if audit exists
        self._act_view_audit.setEnabled(bool(audit_ok))


    def _selected_manifest(self) -> Optional[Path]:
        it = self._tree.currentItem()
        if it is None:
            return None
        raw = it.data(0, MANIFEST_ROLE)
        if not raw:
            return None
        try:
            return Path(str(raw))
        except Exception:
            return None
        
    def _selected_audit(self) -> Optional[Path]:
        it = self._tree.currentItem()
        if it is None:
            return None
        raw = it.data(0, AUDIT_ROLE)
        if not raw:
            return None
        p = Path(str(raw))
        return p if p.exists() else None


    def _open_manifest_selected(self) -> None:
        man = self._selected_manifest()
        if man is None:
            return
        JsonManifestDialog.open_file(
            path=man,
            parent=self,
            title="Stage-1 manifest",
        )
    
    def _open_audit_selected(self) -> None:
        audit = self._selected_audit()
        if audit is None:
            return
        JsonAuditDialog.open_file(
            path=audit,
            parent=self,
            title="Stage-1 scaling audit",
        )

            
    def _on_run_sel(self) -> None:
        it = self._tree.currentItem()
        p = self._selected_manifest()
        if it is None or p is None:
            return

        city = it.text(0).strip() or "Unknown"

        self._queued = TuneJobSpec(
            city=city,
            stage1_manifest=p,
        )

        # best-effort: record city in store
        try:
            self._store.set_value_by_key(
                FieldKey("city"),
                city,
            )
        except Exception:
            pass

        self.accept()


# -----------------------------
# Helpers
# -----------------------------
def _cfg_copy(cfg: Any) -> Any:
    try:
        return copy.deepcopy(cfg)
    except Exception:
        return cfg


def _discover_stage1(root: Path) -> List[_Stage1Run]:
    if not root.exists():
        return []

    cands = _find_json_candidates(root)
    runs: List[_Stage1Run] = []

    for p in cands:
        rec = _try_parse_stage1_manifest(p)
        if rec is not None:
            runs.append(rec)

    by_city: Dict[str, _Stage1Run] = {}
    for r in runs:
        prev = by_city.get(r.city)
        if prev is None or r.timestamp > prev.timestamp:
            by_city[r.city] = r

    out = list(by_city.values())
    out.sort(key=lambda x: (x.city.lower(), x.timestamp))
    return out


def _find_json_candidates(root: Path) -> List[Path]:
    pats = (
        "*stage1*.json",
        "*stage-1*.json",
        "*stage_1*.json",
    )

    out: List[Path] = []
    seen: set[Path] = set()

    try:
        dirs = [d for d in root.iterdir() if d.is_dir()]
    except Exception:
        dirs = []

    targets = dirs if dirs else [root]

    for d in targets:
        for pat in pats:
            for p in d.rglob(pat):
                if not p.is_file():
                    continue
                if p in seen:
                    continue
                seen.add(p)
                out.append(p)

    return out

def _find_stage1_audit_json(
    manifest: Path,
) -> Optional[Path]:
    """
    Try to locate stage1 scaling audit JSON near a manifest.

    Heuristics
    ----------
    - Look in manifest parent dir.
    - Look in artifacts/ under that dir.
    - Prefer files containing 'audit' and 'scaling'.
    """
    base = manifest.parent
    cands: List[Path] = []

    pats = (
        "*stage1*scaling*audit*.json",
        "*stage1*audit*.json",
        "*scaling*audit*.json",
        "*audit*.json",
    )

    for root in (base, base / "artifacts"):
        if not root.exists():
            continue
        for pat in pats:
            try:
                cands.extend(list(root.glob(pat)))
            except Exception:
                continue

    for p in cands:
        name = p.name.lower()
        if "scaling" in name and "audit" in name:
            return p

    return cands[0] if cands else None


def _try_parse_stage1_manifest(p: Path) -> Optional[_Stage1Run]:
    try:
        data = json.loads(p.read_text("utf-8"))
    except Exception:
        return None

    if not _looks_like_stage1(data, p):
        return None

    city = _infer_city(data, p)
    ts = _infer_timestamp(data, p)

    t_steps = _infer_int(data, keys=("time_steps", "T"))
    horizon = _infer_int(
        data,
        keys=("horizon_years", "H"),
    )

    return _Stage1Run(
        city=city,
        timestamp=ts,
        time_steps=t_steps,
        horizon_years=horizon,
        manifest=p,
    )


def _looks_like_stage1(data: Dict[str, Any], p: Path) -> bool:
    if "stage1" in p.name.lower():
        return True

    stage = str(data.get("stage", "")).lower()
    if "stage1" in stage:
        return True

    keys = ("city", "time_steps", "horizon_years")
    hit = sum(1 for k in keys if k in data)
    return hit >= 2


def _infer_city(data: Dict[str, Any], p: Path) -> str:
    for k in ("city", "CITY_NAME", "city_name"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    st1 = data.get("stage1")
    if isinstance(st1, dict):
        v = st1.get("city")
        if isinstance(v, str) and v.strip():
            return v.strip()

    try:
        return p.parent.name
    except Exception:
        return "Unknown"


def _infer_timestamp(data: Dict[str, Any], p: Path) -> str:
    v = data.get("timestamp")
    if isinstance(v, str) and v.strip():
        return v.strip()

    try:
        name = p.parent.name
        if name and any(ch.isdigit() for ch in name):
            return name
    except Exception:
        pass

    return "unknown"


def _infer_int(
    data: Dict[str, Any],
    *,
    keys: Iterable[str],
) -> Optional[int]:
    for k in keys:
        v = data.get(k)
        try:
            if v is None:
                continue
            return int(v)
        except Exception:
            continue
    return None
