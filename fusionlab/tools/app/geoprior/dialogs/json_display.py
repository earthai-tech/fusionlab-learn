# geoprior/services/json_display.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json

from PyQt5.QtCore import Qt, QUrl, QPoint
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..styles import PRIMARY, PALETTE  # SECONDARY,

JsonLike = Union[Dict[str, Any], list, str, int, float, bool, None]


# ---------------------------------------------------------------------
# Preview dialog (unchanged)
# ---------------------------------------------------------------------
class TextPreviewDialog(QDialog):
    def __init__(
        self,
        *,
        title: str,
        text: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        ed = QPlainTextEdit()
        ed.setReadOnly(True)
        ed.setPlainText(text)
        root.addWidget(ed, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)


# ---------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class JsonDisplayOptions:
    max_inline_chars: int = 180
    max_list_inline: int = 8


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
def _read_json_or_warn(
    path: Union[str, Path],
    parent: Optional[QWidget],
) -> Optional[JsonLike]:
    p = Path(path).expanduser()
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception as exc:
        QMessageBox.warning(
            parent,
            "Invalid JSON",
            f"Could not read:\n{p}\n\n{exc}",
        )
        return None


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _looks_like_path(s: str) -> bool:
    if not s:
        return False
    t = s.strip()
    if len(t) < 3:
        return False
    # Windows drive or UNC or Unix root
    if (len(t) >= 3 and t[1:3] == ":\\") or t.startswith("\\\\") or t.startswith("/"):
        return True
    # Common separators
    if "\\" in t or "/" in t:
        return True
    return False


def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, list):
        return "list"
    return type(v).__name__


def _fmt_num(x: Union[int, float]) -> str:
    if isinstance(x, int):
        return str(x)
    ax = abs(float(x))
    if ax != 0.0 and (ax < 1e-4 or ax >= 1e6):
        return f"{x:.6g}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _safe_one(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _fmt_num(v)
    s = str(v)
    return s.replace("\n", " ").strip()


def _can_inline_list(xs: list, opts: JsonDisplayOptions) -> bool:
    if len(xs) > opts.max_list_inline:
        return False
    for x in xs:
        if isinstance(x, (dict, list)):
            return False
    return True


def _format_value(v: Any, opts: JsonDisplayOptions) -> Tuple[str, Optional[str]]:
    if v is None:
        return "null", None
    if isinstance(v, bool):
        return ("true" if v else "false"), None
    if isinstance(v, (int, float)):
        return _fmt_num(v), None
    if isinstance(v, (dict, list)):
        return "", None

    s = str(v)
    flat = s.replace("\n", " ⏎ ")
    if len(flat) <= opts.max_inline_chars:
        return flat, flat

    shown = flat[: opts.max_inline_chars].rstrip() + "…"
    return shown, flat


# ---------------------------------------------------------------------
# Base: modern JSON tree view
# ---------------------------------------------------------------------
class JsonTreeViewBase(QWidget):
    """
    Reusable JSON tree viewer with:
    - title/path
    - search
    - expand/collapse
    - optional "top panel" (manifest summary)
    - context menu (copy/open path)
    """

    def __init__(
        self,
        *,
        title: str,
        parent: Optional[QWidget] = None,
        options: Optional[JsonDisplayOptions] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._opts = options or JsonDisplayOptions()
        self._raw: Optional[JsonLike] = None
        self._path: Optional[Path] = None

        self._build_ui()
        self._wire_ui()

    # ------------------------- override hooks -------------------------

    def _build_top_panel(self) -> Optional[QWidget]:
        """Manifest overrides this to add summary card."""
        return None

    def _right_actions(self) -> List[QWidget]:
        """Manifest overrides this to add quick action buttons."""
        return []

    def _after_set_json(self, data: JsonLike) -> None:
        """Manifest overrides to fill summary."""
        return

    # --------------------------- public API ---------------------------

    def set_json(self, data: JsonLike, *, root_name: str = "json") -> None:
        self._raw = data
        self._tree.setUpdatesEnabled(False)
        try:
            self._tree.clear()
            root = QTreeWidgetItem([root_name, "", _type_name(data)])
            self._set_bold_item(root, True)
            self._tree.addTopLevelItem(root)
            self._fill_item(root, data)
            root.setExpanded(True)
        finally:
            self._tree.setUpdatesEnabled(True)

        self._after_set_json(data)
        self._apply_filter(self._search.text())

    def load_json_file(self, path: Union[str, Path]) -> bool:
        p = Path(path).expanduser()
        data = _read_json_or_warn(p, self)
        if data is None:
            return False
        self._path = p
        self._path_lab.setText(str(p))
        self.set_json(data, root_name=p.name)
        return True

    # ------------------------------ UI -------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Header
        top = QHBoxLayout()
        top.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(2)

        self._ttl = QLabel(self._title)
        self._ttl.setObjectName("jsonTitle")
        self._set_bold_label(self._ttl, True)

        self._path_lab = QLabel("")
        self._path_lab.setObjectName("jsonPath")
        self._path_lab.setTextInteractionFlags(Qt.TextSelectableByMouse)

        left.addWidget(self._ttl)
        left.addWidget(self._path_lab)
        top.addLayout(left, 1)

        self._search = QLineEdit()
        self._search.setObjectName("jsonSearch")
        self._search.setPlaceholderText("Search keys/values…")
        self._search.setClearButtonEnabled(True)

        self._btn_expand = QPushButton("Expand")
        self._btn_expand.setObjectName("ghost")

        self._btn_collapse = QPushButton("Collapse")
        self._btn_collapse.setObjectName("ghost")

        top.addWidget(self._search, 2)

        # right-side extra actions (manifest)
        for w in self._right_actions():
            top.addWidget(w)

        top.addWidget(self._btn_expand)
        top.addWidget(self._btn_collapse)

        root.addLayout(top)

        # Optional panel (manifest summary)
        panel = self._build_top_panel()
        if panel is not None:
            root.addWidget(panel)

        # Tree "card"
        card = QFrame()
        card.setObjectName("jsonCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)

        self._tree = QTreeWidget()
        self._tree.setObjectName("jsonTree")
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Key", "Value", "Type"])
        self._tree.setRootIsDecorated(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tree.setIndentation(16)
        self._tree.header().setStretchLastSection(False)
        self._tree.header().setSectionResizeMode(0, self._tree.header().Stretch)
        self._tree.header().setSectionResizeMode(1, self._tree.header().Stretch)
        self._tree.header().setSectionResizeMode(2, self._tree.header().ResizeToContents)

        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)

        card_lay.addWidget(self._tree)
        root.addWidget(card, 1)

        self._apply_local_styles()

    def _apply_local_styles(self) -> None:
        # Lightweight “modern table” feel without touching global app styles.
        qss = f"""
        QFrame#jsonCard {{
            background: {PALETTE['light_card_bg']};
            border: 1px solid {PALETTE['light_border']};
            border-radius: 12px;
        }}
        QLabel#jsonTitle {{
            font-size: 16px;
            font-weight: 700;
            color: {PALETTE['light_text_title']};
        }}
        QLabel#jsonPath {{
            color: {PALETTE['light_text_muted']};
        }}
        QLineEdit#jsonSearch {{
            padding: 6px 10px;
            border-radius: 8px;
            background: {PALETTE['light_input_bg']};
            border: 1px solid {PALETTE['light_border']};
        }}
        QPushButton#ghost {{
            background: transparent;
            color: {PRIMARY};
            border: 1px solid {PALETTE['light_border']};
            padding: 6px 10px;
            border-radius: 8px;
            font-weight: 600;
        }}
        QPushButton#ghost:hover:enabled {{
            background: rgba(46,49,145,0.08);
        }}
        QTreeWidget#jsonTree {{
            border: none;
        }}
        QHeaderView::section {{
            background: {PRIMARY};
            color: white;
            padding: 8px 10px;
            border: none;
            font-weight: 700;
        }}
        QTreeWidget::item {{
            padding: 6px 8px;
        }}
        QTreeWidget::item:hover {{
            background: rgba(46,49,145,0.06);
        }}
        QTreeWidget::item:selected {{
            background: rgba(242,134,32,0.22);
        }}
        """
        self.setStyleSheet(qss)

    def _wire_ui(self) -> None:
        self._search.textChanged.connect(self._apply_filter)
        self._btn_expand.clicked.connect(lambda: self._expand_all(True))
        self._btn_collapse.clicked.connect(lambda: self._expand_all(False))
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)

    # ---------------------------- actions ----------------------------

    def _expand_all(self, on: bool) -> None:
        self._tree.setUpdatesEnabled(False)
        try:
            it = self._tree.invisibleRootItem()
            for i in range(it.childCount()):
                self._expand_rec(it.child(i), on)
        finally:
            self._tree.setUpdatesEnabled(True)

    def _expand_rec(self, item: QTreeWidgetItem, on: bool) -> None:
        item.setExpanded(on)
        for i in range(item.childCount()):
            self._expand_rec(item.child(i), on)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _: int) -> None:
        # If it looks like a path, try to open.
        full = item.data(1, Qt.UserRole)
        if isinstance(full, str) and _looks_like_path(full):
            self._open_path(full)
            return

        # Otherwise long text preview.
        if not isinstance(full, str):
            return
        shown = item.text(1)
        if shown == full:
            return
        key = item.text(0) or "Value"
        TextPreviewDialog(title=key, text=full, parent=self).exec_()

    def _on_context_menu(self, pos: QPoint) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return

        key = item.text(0)
        val = item.data(1, Qt.UserRole)
        if not isinstance(val, str):
            val = item.text(1)

        m = QMenu(self)

        act_copy_key = m.addAction("Copy key")
        act_copy_val = m.addAction("Copy value")

        act_open = None
        if isinstance(val, str) and _looks_like_path(val):
            m.addSeparator()
            act_open = m.addAction("Open path")

        chosen = m.exec_(self._tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return

        if chosen == act_copy_key:
            QApplication.clipboard().setText(key)
        elif chosen == act_copy_val:
            QApplication.clipboard().setText(str(val))
        elif act_open is not None and chosen == act_open:
            self._open_path(str(val))

    def _open_path(self, p: str) -> None:
        try:
            pp = Path(p)
            # If it's a file, open its parent folder for safety
            target = pp if pp.exists() and pp.is_dir() else pp.parent
            if target.exists():
                from PyQt5.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
            else:
                # fallback: preview
                TextPreviewDialog(title="Path", text=p, parent=self).exec_()
        except Exception:
            TextPreviewDialog(title="Path", text=p, parent=self).exec_()

    # -------------------------- population ---------------------------

    def _fill_item(self, parent: QTreeWidgetItem, v: Any) -> None:
        if isinstance(v, dict):
            self._fill_dict(parent, v)
            return
        if isinstance(v, list):
            self._fill_list(parent, v)
            return

        txt, full = _format_value(v, self._opts)
        parent.setText(1, txt)
        parent.setText(2, _type_name(v))

        # Store full text (also used for path opening)
        if full is not None:
            parent.setData(1, Qt.UserRole, full)
            parent.setToolTip(1, full)

    def _fill_dict(self, parent: QTreeWidgetItem, d: Dict[str, Any]) -> None:
        parent.setText(2, "dict")
        for k in sorted(d.keys(), key=lambda x: str(x)):
            child = QTreeWidgetItem([str(k), "", ""])
            parent.addChild(child)
            self._fill_item(child, d.get(k))

    def _fill_list(self, parent: QTreeWidgetItem, xs: list) -> None:
        parent.setText(2, "list")
        if _can_inline_list(xs, self._opts):
            txt = ", ".join(_safe_one(x) for x in xs)
            parent.setText(1, f"[{txt}]")
            parent.setData(1, Qt.UserRole, parent.text(1))
            return

        parent.setText(1, f"[{len(xs)} items]")
        parent.setData(1, Qt.UserRole, parent.text(1))
        for i, x in enumerate(xs):
            child = QTreeWidgetItem([f"[{i}]", "", ""])
            parent.addChild(child)
            self._fill_item(child, x)

    # ---------------------------- filter -----------------------------

    def _apply_filter(self, text: str) -> None:
        t = (text or "").strip().lower()
        root = self._tree.invisibleRootItem()
        for i in range(root.childCount()):
            self._filter_rec(root.child(i), t)

    def _filter_rec(self, item: QTreeWidgetItem, t: str) -> bool:
        if not t:
            item.setHidden(False)
            for i in range(item.childCount()):
                self._filter_rec(item.child(i), t)
            return True

        hay = ((item.text(0) + " " + item.text(1)).lower())
        hit = t in hay

        child_hit = False
        for i in range(item.childCount()):
            if self._filter_rec(item.child(i), t):
                child_hit = True

        keep = hit or child_hit
        item.setHidden(not keep)
        if child_hit and keep:
            item.setExpanded(True)
        return keep

    # ---------------------------- styling ----------------------------

    def _set_bold_item(self, it: QTreeWidgetItem, on: bool) -> None:
        f = it.font(0)
        f.setBold(on)
        it.setFont(0, f)

    def _set_bold_label(self, lab: QLabel, on: bool) -> None:
        f = QFont(lab.font())
        f.setBold(on)
        lab.setFont(f)


# ---------------------------------------------------------------------
# Audit view = base view as-is
# ---------------------------------------------------------------------
class JsonAuditView(JsonTreeViewBase):
    pass


# ---------------------------------------------------------------------
# Manifest view = base view + summary + quick actions
# ---------------------------------------------------------------------
class JsonManifestView(JsonTreeViewBase):
    def __init__(
        self,
        *,
        title: str,
        parent: Optional[QWidget] = None,
        options: Optional[JsonDisplayOptions] = None,
    ) -> None:
        self._sum_pairs: List[Tuple[QLabel, QLabel]] = []
        self._btn_open_run = QPushButton("Open run dir")
        self._btn_open_run.setObjectName("ghost")
        self._btn_open_art = QPushButton("Open artifacts")
        self._btn_open_art.setObjectName("ghost")
        self._btn_copy_run = QPushButton("Copy run dir")
        self._btn_copy_run.setObjectName("ghost")

        super().__init__(title=title, parent=parent, options=options)

    def _right_actions(self) -> List[QWidget]:
        return [self._btn_open_run, self._btn_open_art, self._btn_copy_run]

    def _build_top_panel(self) -> Optional[QWidget]:
        box = QFrame()
        box.setObjectName("manifestSummary")
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(18)
        lay.setVerticalSpacing(6)

        def add_pair(r: int, c: int, k: str, v: str) -> None:
            lk = QLabel(k)
            lk.setStyleSheet(f"color: {PALETTE['light_text_muted']}; font-size: 11px;")
            lv = QLabel(v)
            lv.setStyleSheet(f"color: {PALETTE['light_text']}; font-weight: 700;")
            lay.addWidget(lk, r, c * 2)
            lay.addWidget(lv, r, c * 2 + 1)
            self._sum_pairs.append((lk, lv))

        # placeholders (filled after json loaded)
        add_pair(0, 0, "schema:", "")
        add_pair(0, 1, "timestamp:", "")
        add_pair(1, 0, "city:", "")
        add_pair(1, 1, "stage/model:", "")
        add_pair(2, 0, "T / H:", "")
        add_pair(2, 1, "features (S/D/F):", "")

        box.setStyleSheet(
            f"""
            QFrame#manifestSummary {{
                background: {PALETTE['light_input_bg']};
                border: 1px solid {PALETTE['light_border']};
                border-radius: 12px;
            }}
            """
        )
        return box

    def _after_set_json(self, data: JsonLike) -> None:
        self._fill_manifest_summary(data)
        self._wire_manifest_actions(data)

    def _wire_manifest_actions(self, data: JsonLike) -> None:
        # Disconnect old to avoid stacking (cheap safe way)
        try:
            self._btn_open_run.clicked.disconnect()
            self._btn_open_art.clicked.disconnect()
            self._btn_copy_run.clicked.disconnect()
        except Exception:
            pass

        run_dir = ""
        art_dir = ""
        if isinstance(data, dict):
            paths = data.get("paths")
            if isinstance(paths, dict):
                run_dir = _as_str(paths.get("run_dir"))
                art_dir = _as_str(paths.get("artifacts_dir"))

        self._btn_open_run.clicked.connect(lambda: self._open_path(run_dir))
        self._btn_open_art.clicked.connect(lambda: self._open_path(art_dir))
        self._btn_copy_run.clicked.connect(lambda: QApplication.clipboard().setText(run_dir or ""))

        # Disable buttons when missing
        self._btn_open_run.setEnabled(bool(run_dir))
        self._btn_copy_run.setEnabled(bool(run_dir))
        self._btn_open_art.setEnabled(bool(art_dir))

    def _fill_manifest_summary(self, data: Any) -> None:
        if not isinstance(data, dict):
            return

        schema = _as_str(data.get("schema_version"))
        ts = _as_str(data.get("timestamp"))
        city = _as_str(data.get("city"))
        stage = _as_str(data.get("stage"))
        model = _as_str(data.get("model"))

        t_h = ""
        cfg = data.get("config")
        if isinstance(cfg, dict):
            t = _as_str(cfg.get("TIME_STEPS"))
            h = _as_str(cfg.get("FORECAST_HORIZON_YEARS"))
            if t or h:
                t_h = f"{t} / {h}"

        feat_counts = ""
        if isinstance(cfg, dict):
            feats = cfg.get("features")
            if isinstance(feats, dict):
                s = feats.get("static") if isinstance(feats.get("static"), list) else []
                d = feats.get("dynamic") if isinstance(feats.get("dynamic"), list) else []
                f = feats.get("future") if isinstance(feats.get("future"), list) else []
                feat_counts = f"{len(s)} / {len(d)} / {len(f)}"

        rows = [
            ("schema:", schema),
            ("timestamp:", ts),
            ("city:", city),
            ("stage/model:", f"{stage} / {model}".strip(" /")),
            ("T / H:", t_h),
            ("features (S/D/F):", feat_counts),
        ]

        for i, (lk, lv) in enumerate(self._sum_pairs):
            if i < len(rows):
                lk.setText(rows[i][0])
                lv.setText(rows[i][1])
            else:
                lk.setText("")
                lv.setText("")


# ---------------------------------------------------------------------
# Base dialog wrapper
# ---------------------------------------------------------------------
class _BaseJsonDialog(QDialog):
    view_cls = JsonTreeViewBase

    def __init__(
        self,
        *,
        title: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.view = self.view_cls(title=title, parent=self)
        root.addWidget(self.view, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)


# ---------------------------------------------------------------------
# Audit dialog
# ---------------------------------------------------------------------
class JsonAuditDialog(_BaseJsonDialog):
    view_cls = JsonAuditView

    @classmethod
    def open_file(
        cls,
        *,
        path: Union[str, Path],
        parent: Optional[QWidget] = None,
        title: str = "Audit details",
    ) -> None:
        dlg = cls(title=title, parent=parent)
        ok = dlg.view.load_json_file(path)
        if ok:
            dlg.exec_()


# ---------------------------------------------------------------------
# Manifest dialog
# ---------------------------------------------------------------------
class JsonManifestDialog(_BaseJsonDialog):
    view_cls = JsonManifestView

    @classmethod
    def open_file(
        cls,
        *,
        path: Union[str, Path],
        parent: Optional[QWidget] = None,
        title: str = "Stage-1 manifest",
    ) -> bool:
        dlg = cls(title=title, parent=parent)
        ok = dlg.view.load_json_file(path)
        if ok:
            dlg.exec_()
            return True
        return False
