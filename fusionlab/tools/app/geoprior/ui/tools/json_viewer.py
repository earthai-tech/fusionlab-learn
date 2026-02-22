# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.tools.json_viewer

Robust JSON viewer tool (read-only by default).

Features
--------
- Open JSON from disk (config/manifest/diagnostics).
- Tree + text views with split mode.
- Search in text, jump next/prev.
- Status chip: Valid / Error.
- Recents + pinned files (QSettings).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from PyQt5.QtCore import Qt, QSettings, QUrl
from PyQt5.QtGui import (
    QDesktopServices,
    QFontDatabase,
)
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QStyle,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class _JsonDoc:
    path: Optional[Path]
    obj: Any
    text_pretty: str
    text_min: str


class _Chip(QLabel):
    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self.setObjectName("chip")
        self.setMinimumHeight(22)
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setStyleSheet(
            "padding:2px 10px;"
            "border-radius:11px;"
            "background: palette(midlight);"
            "color: palette(text);"
        )

class JsonViewerTool(QWidget):
    """
    JSON viewer tool.

    Notes
    -----
    This tool is embedded under ToolPageFrame, so it does not
    render any big page title/header. Keep it compact.
    """

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_ctx = app_ctx

        self._settings = QSettings("fusionlab", "geoprior")
        self._pins: List[str] = []
        self._recents: List[str] = []

        self._doc: Optional[_JsonDoc] = None
        self._view_mode: str = "split"  # split|tree|text

        self._load_pins()
        self._load_recents()
        self._build_ui()
        self._connect_ui()
        self._refresh_view()

    # -----------------------------------------------------------------
    # Settings
    # -----------------------------------------------------------------
    def _load_pins(self) -> None:
        v = self._settings.value("json_viewer.pins", [])
        self._pins = [s for s in v if isinstance(s, str)]
        self._pins = [s for s in self._pins if s.strip()]

    def _save_pins(self) -> None:
        self._settings.setValue("json_viewer.pins", self._pins)

    def _load_recents(self) -> None:
        v = self._settings.value("json_viewer.recents", [])
        xs = [s for s in v if isinstance(s, str)]
        xs = [s for s in xs if s.strip()]
        # unique, keep order
        out: List[str] = []
        seen = set()
        for s in xs:
            if s not in seen:
                out.append(s)
                seen.add(s)
        self._recents = out[:12]

    def _save_recents(self) -> None:
        self._settings.setValue(
            "json_viewer.recents",
            self._recents[:12],
        )

    def _push_recent(self, p: Path) -> None:
        s = str(p)
        xs = [x for x in self._recents if x != s]
        xs.insert(0, s)
        self._recents = xs[:12]
        self._save_recents()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # --- top bar
        top = QHBoxLayout()
        top.setSpacing(8)

        self._path = QLineEdit(self)
        self._path.setReadOnly(True)
        self._path.setPlaceholderText("Open a JSON file…")

        self._btn_open = QToolButton(self)
        self._btn_open.setAutoRaise(True)
        self._btn_open.setObjectName("miniAction")
        self._btn_open.setToolTip("Open JSON file")
        self._btn_open.setIcon(
            self.style().standardIcon(QStyle.SP_DirOpenIcon)
        )

        self._btn_more = QToolButton(self)
        self._btn_more.setAutoRaise(True)
        self._btn_more.setObjectName("miniAction")
        self._btn_more.setToolTip("Recent / pinned / actions")
        self._btn_more.setPopupMode(
            QToolButton.InstantPopup
        )
        self._menu = QMenu(self._btn_more)
        self._btn_more.setMenu(self._menu)

        self._btn_pin = QToolButton(self)
        self._btn_pin.setAutoRaise(True)
        self._btn_pin.setObjectName("miniAction")
        self._btn_pin.setToolTip("Pin/unpin current file")
        self._btn_pin.setText("★")

        self._btn_view = QToolButton(self)
        self._btn_view.setAutoRaise(True)
        self._btn_view.setObjectName("miniAction")
        self._btn_view.setToolTip("Switch view mode")
        self._btn_view.setText("Split ▾")
        self._btn_view.setPopupMode(
            QToolButton.InstantPopup
        )
        self._view_menu = QMenu(self._btn_view)
        self._btn_view.setMenu(self._view_menu)

        self._search = QLineEdit(self)
        self._search.setPlaceholderText("Search…")

        self._btn_prev = QToolButton(self)
        self._btn_prev.setAutoRaise(True)
        self._btn_prev.setObjectName("miniAction")
        self._btn_prev.setToolTip("Previous match")
        self._btn_prev.setIcon(
            self.style().standardIcon(
                QStyle.SP_ArrowUp
            )
        )

        self._btn_next = QToolButton(self)
        self._btn_next.setAutoRaise(True)
        self._btn_next.setObjectName("miniAction")
        self._btn_next.setToolTip("Next match")
        self._btn_next.setIcon(
            self.style().standardIcon(
                QStyle.SP_ArrowDown
            )
        )

        self._status = _Chip("Idle", self)

        top.addWidget(self._path, 1)
        top.addWidget(self._btn_open)
        top.addWidget(self._btn_more)
        top.addWidget(self._btn_pin)
        top.addWidget(self._btn_view)
        top.addWidget(self._search, 0)
        top.addWidget(self._btn_prev)
        top.addWidget(self._btn_next)
        top.addWidget(self._status)

        root.addLayout(top)

        # --- body
        self._split = QSplitter(Qt.Horizontal, self)
        self._split.setChildrenCollapsible(False)

        self._tree = QTreeWidget(self)
        self._tree.setHeaderLabels(["Key", "Value"])
        self._tree.setUniformRowHeights(True)
        self._tree.setAlternatingRowColors(True)

        self._edit = QPlainTextEdit(self)
        self._edit.setReadOnly(True)
        self._edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._edit.setPlaceholderText(
            "Open a JSON file to view its contents."
        )

        mono = QFontDatabase.systemFont(
            QFontDatabase.FixedFont
        )
        self._edit.setFont(mono)

        self._split.addWidget(self._tree)
        self._split.addWidget(self._edit)
        self._split.setStretchFactor(0, 1)
        self._split.setStretchFactor(1, 2)

        root.addWidget(self._split, 1)

        # menus
        self._build_menus()

    def _build_menus(self) -> None:
        self._menu.clear()

        a_open_folder = QAction("Open folder", self)
        a_open_folder.triggered.connect(
            self._open_containing_folder
        )
        self._menu.addAction(a_open_folder)

        self._menu.addSeparator()

        self._m_recent = self._menu.addMenu("Recent")
        self._m_pins = self._menu.addMenu("Pinned")

        self._menu.addSeparator()

        a_clear = QAction("Clear recents", self)
        a_clear.triggered.connect(self._clear_recents)
        self._menu.addAction(a_clear)

        # view menu
        self._view_menu.clear()
        for key, label in (
            ("split", "Split"),
            ("tree", "Tree"),
            ("text", "Text"),
        ):
            act = QAction(label, self)
            act.triggered.connect(
                lambda _c=False, k=key: self._set_view(k)
            )
            self._view_menu.addAction(act)

        self._rebuild_recent_pin_menus()

    def _rebuild_recent_pin_menus(self) -> None:
        self._m_recent.clear()
        self._m_pins.clear()

        if not self._recents:
            a = QAction("(empty)", self)
            a.setEnabled(False)
            self._m_recent.addAction(a)
        else:
            for s in self._recents:
                p = Path(s)
                act = QAction(p.name, self)
                act.setToolTip(s)
                act.triggered.connect(
                    lambda _c=False, x=p: self.open_path(x)
                )
                self._m_recent.addAction(act)

        if not self._pins:
            a = QAction("(empty)", self)
            a.setEnabled(False)
            self._m_pins.addAction(a)
        else:
            for s in self._pins:
                p = Path(s)
                act = QAction(p.name, self)
                act.setToolTip(s)
                act.triggered.connect(
                    lambda _c=False, x=p: self.open_path(x)
                )
                self._m_pins.addAction(act)

    def _connect_ui(self) -> None:
        self._btn_open.clicked.connect(self._on_open)
        self._btn_pin.clicked.connect(self._toggle_pin)

        self._tree.itemSelectionChanged.connect(
            self._on_tree_select
        )

        self._search.returnPressed.connect(
            self._find_next
        )
        self._btn_next.clicked.connect(self._find_next)
        self._btn_prev.clicked.connect(self._find_prev)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def open_path(self, path: Path) -> None:
        try:
            doc = self._load_json(path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "JSON error",
                f"Failed to open:\n{path}\n\n{exc}",
            )
            self._doc = None
            self._status.setText("Error")
            self._refresh_view()
            return

        self._doc = doc
        self._path.setText(str(path))
        self._push_recent(path)
        self._rebuild_recent_pin_menus()
        self._status.setText("Valid")
        self._refresh_view()

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------
    def _on_open(self) -> None:
        start = self._guess_start_dir()
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Open JSON",
            start,
            "JSON files (*.json);;All files (*.*)",
        )
        if not p:
            return
        self.open_path(Path(p))

    def _guess_start_dir(self) -> str:
        # prefer current file folder
        if self._doc and self._doc.path:
            return str(self._doc.path.parent)

        # else prefer app_ctx results_root if present
        ctx = self._app_ctx
        root = getattr(ctx, "results_root", None)
        if root:
            return str(root)

        return str(Path.cwd())

    def _open_containing_folder(self) -> None:
        if not self._doc or not self._doc.path:
            return
        p = self._doc.path.parent
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(p))
        )

    def _clear_recents(self) -> None:
        self._recents = []
        self._save_recents()
        self._rebuild_recent_pin_menus()

    def _toggle_pin(self) -> None:
        if not self._doc or not self._doc.path:
            return
        s = str(self._doc.path)
        if s in self._pins:
            self._pins = [x for x in self._pins if x != s]
        else:
            self._pins.insert(0, s)
            self._pins = self._pins[:12]
        self._save_pins()
        self._rebuild_recent_pin_menus()
        self._refresh_pin_btn()

    def _refresh_pin_btn(self) -> None:
        if not self._doc or not self._doc.path:
            self._btn_pin.setText("★")
            return
        s = str(self._doc.path)
        self._btn_pin.setText("★" if s in self._pins else "☆")

    def _set_view(self, mode: str) -> None:
        if mode not in ("split", "tree", "text"):
            return
        self._view_mode = mode
        self._btn_view.setText(mode.capitalize() + " ▾")
        self._refresh_view()

    # -----------------------------------------------------------------
    # Load + render
    # -----------------------------------------------------------------
    def _load_json(self, path: Path) -> _JsonDoc:
        txt = path.read_text(encoding="utf-8")
        obj = json.loads(txt)

        pretty = json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        mini = json.dumps(
            obj,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return _JsonDoc(
            path=path,
            obj=obj,
            text_pretty=pretty,
            text_min=mini,
        )

    def _refresh_view(self) -> None:
        self._refresh_pin_btn()

        if not self._doc:
            self._tree.clear()
            self._edit.setPlainText("")
            self._status.setText("Idle")
            return

        # view mode show/hide
        if self._view_mode == "split":
            self._tree.setVisible(True)
            self._edit.setVisible(True)
        elif self._view_mode == "tree":
            self._tree.setVisible(True)
            self._edit.setVisible(False)
        else:
            self._tree.setVisible(False)
            self._edit.setVisible(True)

        self._edit.setPlainText(self._doc.text_pretty)
        self._build_tree(self._doc.obj)

    def _build_tree(self, obj: Any) -> None:
        self._tree.clear()

        root = QTreeWidgetItem(["(root)", self._summ(obj)])
        root.setData(0, Qt.UserRole, obj)
        self._tree.addTopLevelItem(root)

        self._add_children(root, obj, depth=0)
        self._tree.expandToDepth(1)

    def _add_children(
        self,
        parent: QTreeWidgetItem,
        obj: Any,
        depth: int,
    ) -> None:
        if depth >= 8:
            return

        # limits to avoid huge JSON freezes
        max_items = 200

        if isinstance(obj, dict):
            n = 0
            for k, v in obj.items():
                n += 1
                if n > max_items:
                    parent.addChild(
                        QTreeWidgetItem(
                            ["…", "truncated"]
                        )
                    )
                    break
                it = QTreeWidgetItem([str(k), self._summ(v)])
                it.setData(0, Qt.UserRole, v)
                parent.addChild(it)
                self._add_children(it, v, depth + 1)

        elif isinstance(obj, list):
            for i, v in enumerate(obj[:max_items]):
                it = QTreeWidgetItem([f"[{i}]", self._summ(v)])
                it.setData(0, Qt.UserRole, v)
                parent.addChild(it)
                self._add_children(it, v, depth + 1)

            if len(obj) > max_items:
                parent.addChild(
                    QTreeWidgetItem(
                        ["…", "truncated"]
                    )
                )

    def _summ(self, v: Any) -> str:
        if isinstance(v, dict):
            return f"{{{len(v)} keys}}"
        if isinstance(v, list):
            return f"[{len(v)} items]"
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        s = str(v)
        if len(s) > 80:
            s = s[:77] + "…"
        return s

    # -----------------------------------------------------------------
    # Tree -> text focus
    # -----------------------------------------------------------------
    def _on_tree_select(self) -> None:
        sel = self._tree.selectedItems()
        if not sel:
            return
        v = sel[0].data(0, Qt.UserRole)
        if v is None:
            return

        # show a small hint in status
        self._status.setText("Valid")

    # -----------------------------------------------------------------
    # Search (text)
    # -----------------------------------------------------------------
    def _find_next(self) -> None:
        q = (self._search.text() or "").strip()
        if not q:
            return
        if not self._edit.find(q):
            # wrap
            cur = self._edit.textCursor()
            cur.movePosition(cur.Start)
            self._edit.setTextCursor(cur)
            self._edit.find(q)

    def _find_prev(self) -> None:
        q = (self._search.text() or "").strip()
        if not q:
            return
        if not self._edit.find(q, QPlainTextEdit.FindBackward):
            cur = self._edit.textCursor()
            cur.movePosition(cur.End)
            self._edit.setTextCursor(cur)
            self._edit.find(q, QPlainTextEdit.FindBackward)
