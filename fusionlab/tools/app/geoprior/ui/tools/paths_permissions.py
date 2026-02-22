# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.tools.paths_permissions

Paths & permissions tool.

- Detect common roots (results/data/cwd/temp/home/python).
- Check existence, read/write, and disk free space.
- Quick actions: open/reveal/copy/test write/create/zip.
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QStyle,
)


@dataclass
class PathEntry:
    key: str
    label: str
    path: Path
    kind: str = "auto"  # auto|dir|file


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


def _fmt_bytes(n: float) -> str:
    if n <= 0:
        return "—"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    v = float(n)
    while v >= 1024.0 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    if i == 0:
        return f"{int(v)} {units[i]}"
    return f"{v:.2f} {units[i]}"


def _safe_access(p: Path, mode: int) -> bool:
    try:
        return os.access(str(p), mode)
    except Exception:
        return False


class PathsPermissionsTool(QWidget):
    """
    Paths & permissions tool.

    Parameters
    ----------
    app_ctx : object, optional
        Main GUI object. Used to detect roots when present.
    parent : QWidget, optional
        Qt parent.
    """

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_ctx = app_ctx

        self._entries: List[PathEntry] = []
        self._last_checks: Dict[str, Dict[str, Any]] = {}

        self._init_ui()
        self.refresh()

    # ------------------------------------------------------------
    # Public hook (ToolPageFrame calls refresh())
    # ------------------------------------------------------------
    def refresh(self) -> None:
        self._entries = self._gather_entries()
        self._populate_tree()
        self._update_overall_chip()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(8)

        self._chip_overall = _Chip("Checking…", self)
        self._chip_overall.setToolTip(
            "Overall status for detected paths."
        )

        self._btn_refresh = QToolButton(self)
        self._btn_refresh.setAutoRaise(True)
        self._btn_refresh.setObjectName("miniAction")
        self._btn_refresh.setToolTip("Refresh checks")
        self._btn_refresh.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload)
        )
        self._btn_refresh.clicked.connect(self.refresh)

        top.addWidget(self._chip_overall)
        top.addStretch(1)
        top.addWidget(self._btn_refresh)

        root.addLayout(top)

        split = QSplitter(Qt.Horizontal, self)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        # ---- left: table
        left = QFrame(self)
        left.setObjectName("pathsLeft")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        self._tree = QTreeWidget(left)
        self._tree.setObjectName("pathsTree")
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setSelectionMode(
            QTreeWidget.SingleSelection
        )
        self._tree.setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self._tree.customContextMenuRequested.connect(
            self._on_menu
        )
        self._tree.currentItemChanged.connect(
            self._on_selected
        )

        self._tree.setHeaderLabels(
            [
                "Name",
                "Path",
                "Exists",
                "Type",
                "R",
                "W",
                "Free",
            ]
        )

        ll.addWidget(self._tree, 1)
        split.addWidget(left)

        # ---- right: details
        right = QFrame(self)
        right.setObjectName("pathsRight")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        chips = QHBoxLayout()
        chips.setSpacing(6)

        self._chip_name = _Chip("No selection", self)
        self._chip_exists = _Chip("Exists: —", self)
        self._chip_rw = _Chip("R/W: —", self)

        chips.addWidget(self._chip_name)
        chips.addWidget(self._chip_exists)
        chips.addWidget(self._chip_rw)
        chips.addStretch(1)

        rl.addLayout(chips)

        btns = QHBoxLayout()
        btns.setSpacing(8)

        self._btn_open = QPushButton("Open", self)
        self._btn_reveal = QPushButton("Reveal", self)
        self._btn_copy = QPushButton("Copy", self)
        self._btn_test = QPushButton("Test write", self)
        self._btn_create = QPushButton("Create", self)
        self._btn_zip = QPushButton("Zip…", self)

        btns.addWidget(self._btn_open)
        btns.addWidget(self._btn_reveal)
        btns.addWidget(self._btn_copy)
        btns.addWidget(self._btn_test)
        btns.addWidget(self._btn_create)
        btns.addWidget(self._btn_zip)
        btns.addStretch(1)

        rl.addLayout(btns)

        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setObjectName("pathsLog")
        self._log.setMinimumHeight(220)
        rl.addWidget(self._log, 1)

        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([720, 420])

        # connect buttons
        self._btn_open.clicked.connect(self._open_selected)
        self._btn_reveal.clicked.connect(self._reveal_selected)
        self._btn_copy.clicked.connect(self._copy_selected)
        self._btn_test.clicked.connect(self._test_selected)
        self._btn_create.clicked.connect(self._create_selected)
        self._btn_zip.clicked.connect(self._zip_selected)

        self._set_actions_enabled(False)

    def _set_actions_enabled(self, on: bool) -> None:
        self._btn_open.setEnabled(on)
        self._btn_reveal.setEnabled(on)
        self._btn_copy.setEnabled(on)
        self._btn_test.setEnabled(on)
        self._btn_create.setEnabled(on)
        self._btn_zip.setEnabled(on)

    # ------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------
    def _gather_entries(self) -> List[PathEntry]:
        out: List[PathEntry] = []
        ctx = self._app_ctx

        def _add(key: str, label: str, p: Any) -> None:
            if not p:
                return
            try:
                out.append(
                    PathEntry(
                        key=key,
                        label=label,
                        path=Path(str(p)),
                        kind="auto",
                    )
                )
            except Exception:
                return

        # results root
        if ctx is not None:
            _add(
                "results_root",
                "Results root",
                getattr(ctx, "results_root", None)
                or getattr(ctx, "gui_runs_root", None),
            )

        # dataset path (best-effort)
        if ctx is not None:
            _add(
                "dataset_path",
                "Dataset path",
                getattr(ctx, "dataset_path", None)
                or getattr(ctx, "active_dataset", None),
            )

        # cwd / home / temp / python
        _add("cwd", "Current working dir", os.getcwd())
        _add("home", "User home", str(Path.home()))
        _add("temp", "Temp dir", tempfile.gettempdir())
        _add("python", "Python executable", sys.executable)

        # de-dup by resolved string
        seen = set()
        uniq: List[PathEntry] = []
        for e in out:
            k = str(e.path)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(e)

        return uniq

    # ------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------
    def _check(self, e: PathEntry) -> Dict[str, Any]:
        p = e.path
        exists = p.exists()

        typ = "—"
        if exists:
            if p.is_dir():
                typ = "dir"
            elif p.is_file():
                typ = "file"
            else:
                typ = "other"

        r_ok = exists and _safe_access(p, os.R_OK)
        w_ok = False
        if exists:
            if p.is_dir():
                w_ok = _safe_access(p, os.W_OK)
            else:
                w_ok = _safe_access(p, os.W_OK)

        free = "—"
        if exists:
            base = p
            if p.is_file():
                base = p.parent
            try:
                du = shutil.disk_usage(str(base))
                free = _fmt_bytes(float(du.free))
            except Exception:
                free = "—"

        return {
            "exists": exists,
            "type": typ,
            "read": bool(r_ok),
            "write": bool(w_ok),
            "free": free,
        }

    # ------------------------------------------------------------
    # Populate
    # ------------------------------------------------------------
    def _populate_tree(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        self._last_checks.clear()

        for e in self._entries:
            chk = self._check(e)
            self._last_checks[e.key] = chk

            it = QTreeWidgetItem(
                [
                    e.label,
                    str(e.path),
                    "yes" if chk["exists"] else "no",
                    str(chk["type"]),
                    "✓" if chk["read"] else "—",
                    "✓" if chk["write"] else "—",
                    str(chk["free"]),
                ]
            )
            it.setData(0, Qt.UserRole, e.key)

            if not chk["exists"] or not chk["read"]:
                it.setToolTip(
                    0,
                    "Missing or not readable.",
                )
            elif chk["type"] == "dir" and not chk["write"]:
                it.setToolTip(
                    0,
                    "Directory not writable.",
                )

            self._tree.addTopLevelItem(it)

        for c, w in enumerate([180, 520, 60, 60, 30, 30, 110]):
            self._tree.setColumnWidth(c, w)

        self._tree.blockSignals(False)

        if self._tree.topLevelItemCount() > 0:
            self._tree.setCurrentItem(self._tree.topLevelItem(0))
        else:
            self._set_actions_enabled(False)
            self._chip_name.setText("No selection")
            self._chip_exists.setText("Exists: —")
            self._chip_rw.setText("R/W: —")
            self._log.setPlainText(
                "[Info] No paths detected from context."
            )

    def _update_overall_chip(self) -> None:
        bad = 0
        warn = 0

        for e in self._entries:
            chk = self._last_checks.get(e.key) or {}
            if not chk.get("exists", False):
                bad += 1
                continue
            if not chk.get("read", False):
                bad += 1
                continue
            if chk.get("type") == "dir" and not chk.get("write", False):
                warn += 1

        if not self._entries:
            self._chip_overall.setText("No paths")
            return

        if bad > 0:
            self._chip_overall.setText(f"Issues: {bad}")
            return
        if warn > 0:
            self._chip_overall.setText(f"Warnings: {warn}")
            return
        self._chip_overall.setText("All good")

    # ------------------------------------------------------------
    # Selection + details
    # ------------------------------------------------------------
    def _selected_entry(self) -> Optional[PathEntry]:
        it = self._tree.currentItem()
        if it is None:
            return None
        key = it.data(0, Qt.UserRole)
        if not isinstance(key, str):
            return None
        for e in self._entries:
            if e.key == key:
                return e
        return None

    def _on_selected(
        self,
        cur: Optional[QTreeWidgetItem],
        _prev: Optional[QTreeWidgetItem],
    ) -> None:
        _ = cur
        e = self._selected_entry()
        if e is None:
            self._set_actions_enabled(False)
            return

        chk = self._last_checks.get(e.key) or self._check(e)

        self._chip_name.setText(e.label)
        self._chip_exists.setText(
            f"Exists: {'yes' if chk.get('exists') else 'no'}"
        )
        self._chip_rw.setText(
            f"R/W: {('R' if chk.get('read') else '-')}"
            f"{('W' if chk.get('write') else '-')}"
        )

        self._set_actions_enabled(True)

        self._log.setPlainText(
            self._detail_text(e, chk)
        )

    def _detail_text(
        self,
        e: PathEntry,
        chk: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        lines.append(f"Name : {e.label}")
        lines.append(f"Key  : {e.key}")
        lines.append(f"Path : {e.path}")
        lines.append("")
        lines.append(f"Exists : {chk.get('exists')}")
        lines.append(f"Type   : {chk.get('type')}")
        lines.append(f"Read   : {chk.get('read')}")
        lines.append(f"Write  : {chk.get('write')}")
        lines.append(f"Free   : {chk.get('free')}")
        lines.append("")
        lines.append("Actions:")
        lines.append("- Open: open folder or file")
        lines.append("- Reveal: open parent folder")
        lines.append("- Test write: create temp file")
        lines.append("- Create: create directory if missing")
        lines.append("- Zip…: export path to zip")
        return "\n".join(lines)

    # ------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------
    def _open_path(self, p: Path) -> None:
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(p))
        )

    def _open_selected(self) -> None:
        e = self._selected_entry()
        if e is None:
            return
        p = e.path
        if not p.exists():
            QMessageBox.information(
                self,
                "Missing path",
                f"Path does not exist:\n{p}",
            )
            return
        self._open_path(p)

    def _reveal_selected(self) -> None:
        e = self._selected_entry()
        if e is None:
            return
        p = e.path
        if p.is_file():
            p = p.parent
        if not p.exists():
            QMessageBox.information(
                self,
                "Missing path",
                f"Path does not exist:\n{p}",
            )
            return
        self._open_path(p)

    def _copy_selected(self) -> None:
        e = self._selected_entry()
        if e is None:
            return
        cb = QApplication.clipboard()
        cb.setText(str(e.path))

    def _test_selected(self) -> None:
        e = self._selected_entry()
        if e is None:
            return
        p = e.path

        if not p.exists():
            QMessageBox.information(
                self,
                "Missing path",
                f"Path does not exist:\n{p}",
            )
            return

        ok, msg = self._test_write(p)
        title = "Write test OK" if ok else "Write test failed"
        QMessageBox.information(self, title, msg)
        self.refresh()

    def _test_write(self, p: Path) -> Tuple[bool, str]:
        try:
            if p.is_dir():
                stamp = time.strftime("%Y%m%d-%H%M%S")
                tmp = p / f".perm_test_{stamp}.tmp"
                tmp.write_text("ok", encoding="utf-8")
                tmp.unlink(missing_ok=True)
                return True, f"Writable:\n{p}"
            if p.is_file():
                with p.open("a", encoding="utf-8") as f:
                    f.write("")
                return True, f"Writable file:\n{p}"
            return False, f"Unsupported path type:\n{p}"
        except Exception as exc:
            return False, f"Failed:\n{p}\n\n{exc}"

    def _create_selected(self) -> None:
        e = self._selected_entry()
        if e is None:
            return
        p = e.path

        if p.exists():
            QMessageBox.information(
                self,
                "Already exists",
                f"Path already exists:\n{p}",
            )
            return

        ans = QMessageBox.question(
            self,
            "Create folder?",
            f"Create directory:\n{p}\n\nProceed?",
        )
        if ans != QMessageBox.Yes:
            return

        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Create failed",
                f"Failed to create:\n{p}\n\n{exc}",
            )
            return

        self.refresh()

    def _zip_selected(self) -> None:
        e = self._selected_entry()
        if e is None:
            return
        p = e.path
        if not p.exists():
            QMessageBox.information(
                self,
                "Missing path",
                f"Path does not exist:\n{p}",
            )
            return

        start = str(p.parent if p.is_file() else p)
        out, _ = QFileDialog.getSaveFileName(
            self,
            "Export zip",
            start,
            "Zip (*.zip)",
        )
        if not out:
            return

        outp = Path(out)
        if outp.suffix.lower() != ".zip":
            outp = outp.with_suffix(".zip")

        ok, msg = self._make_zip(p, outp)
        title = "Zip created" if ok else "Zip failed"
        QMessageBox.information(self, title, msg)

    def _make_zip(self, src: Path, dst_zip: Path) -> Tuple[bool, str]:
        try:
            base = str(dst_zip.with_suffix(""))
            if src.is_dir():
                shutil.make_archive(
                    base, "zip", root_dir=str(src)
                )
            else:
                # zip single file into its parent
                tmpdir = src.parent
                name = src.name
                zpath = base + ".zip"
                import zipfile

                with zipfile.ZipFile(
                    zpath,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                ) as zf:
                    zf.write(str(src), arcname=name)

            return True, f"Saved:\n{base}.zip"
        except Exception as exc:
            return False, f"Failed:\n{dst_zip}\n\n{exc}"

    # ------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------
    def _on_menu(self, pos) -> None:
        it = self._tree.itemAt(pos)
        if it is None:
            return
        self._tree.setCurrentItem(it)

        m = QMenu(self)

        a_open = m.addAction("Open")
        a_reveal = m.addAction("Reveal")
        a_copy = m.addAction("Copy path")
        m.addSeparator()
        a_test = m.addAction("Test write")
        a_create = m.addAction("Create folder")
        m.addSeparator()
        a_zip = m.addAction("Zip…")

        act = m.exec_(self._tree.mapToGlobal(pos))
        if act == a_open:
            self._open_selected()
        elif act == a_reveal:
            self._reveal_selected()
        elif act == a_copy:
            self._copy_selected()
        elif act == a_test:
            self._test_selected()
        elif act == a_create:
            self._create_selected()
        elif act == a_zip:
            self._zip_selected()
