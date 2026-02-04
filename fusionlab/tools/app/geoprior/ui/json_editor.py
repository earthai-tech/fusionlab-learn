# geoprior/ui/json_editor.py
# -*- coding: utf-8 -*-

"""
Modern JSON Viewer & Editor dialog.
Provides Tree view (visual) and Text view (raw).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from PyQt5.QtCore import (
    Qt,
    QRegularExpression,
)
from PyQt5.QtGui import (
    QColor,
    QBrush,
    QFont,
    QFontDatabase,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
    QTabWidget,
    QLineEdit,
    QMessageBox,
    QHeaderView,
    QStyle,
)

from .icon_utils import try_icon

# ------------------------------------------------------------
# Syntax Highlighter (Raw JSON)
# ------------------------------------------------------------


class JsonHighlighter(QSyntaxHighlighter):
    """Simple syntax highlighter for JSON text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules = []

        # Keywords: true, false, null
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("#d35400"))  # Orange
        kw_fmt.setFontWeight(QFont.Bold)
        self._rules.append(
            (
                QRegularExpression("\\b(true|false|null)\\b"),
                kw_fmt,
            )
        )

        # Numbers
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor("#2980b9"))  # Blue
        self._rules.append(
            (
                QRegularExpression(
                    "-?\\b\\d+(\\.\\d+)?([eE][+-]?\\d+)?\\b"
                ),
                num_fmt,
            )
        )

        # Keys (Strings followed by colon)
        key_fmt = QTextCharFormat()
        key_fmt.setForeground(QColor("#8e44ad"))  # Purple
        key_fmt.setFontWeight(QFont.Bold)
        self._rules.append(
            (
                QRegularExpression('".*"(?=:)'),
                key_fmt,
            )
        )

        # String values (Strings NOT followed by colon)
        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor("#27ae60"))  # Green
        self._rules.append(
            (
                QRegularExpression('":\\s*".*"'),
                str_fmt,
            )
        )

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(
                    match.capturedStart(),
                    match.capturedLength(),
                    fmt,
                )


# ------------------------------------------------------------
# Tree Item Logic
# ------------------------------------------------------------


class JsonTreeItem(QTreeWidgetItem):
    """
    Tree item with type-aware coloring.
    """

    def __init__(self, parent, key: str, value: Any):
        super().__init__(parent)
        self._key = key
        self._value = value

        # Set Key
        self.setText(0, str(key))

        # Set Value & Style
        self._render_value()

    def _render_value(self) -> None:
        val = self._value
        txt = ""
        color = None

        if isinstance(val, dict):
            txt = f"{{...}} {len(val)} items"
            color = QColor("#7f8c8d")  # Gray
        elif isinstance(val, list):
            txt = f"[...] {len(val)} items"
            color = QColor("#7f8c8d")
        elif isinstance(val, bool):
            txt = str(val).lower()
            color = QColor("#d35400")  # Orange
        elif val is None:
            txt = "null"
            color = QColor("#c0392b")  # Red
        elif isinstance(val, (int, float)):
            txt = str(val)
            color = QColor("#2980b9")  # Blue
        else:
            # String
            txt = str(val)
            color = QColor("#27ae60")  # Green

        self.setText(1, txt)
        if color:
            self.setForeground(1, QBrush(color))

        # Bold key
        f = self.font(0)
        f.setBold(True)
        self.setFont(0, f)


# ------------------------------------------------------------
# JSON Tree Widget
# ------------------------------------------------------------


class JsonTreeWidget(QWidget):
    """
    Displays JSON in a searchable QTreeWidget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        # Search Bar
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter keys...")
        self._search.textChanged.connect(self._on_search)

        # Icon for search
        ico = try_icon("search.svg")
        if not ico:
            # Fallback
            ico = self.style().standardIcon(
                QStyle.SP_FileDialogContentsView
            )

        act = self._search.addAction(
            ico,
            QLineEdit.LeadingPosition,
        )
        act.triggered.connect(self._search.setFocus)

        self._layout.addWidget(self._search)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Key", "Value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)

        # Column sizing
        h = self.tree.header()
        h.setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )
        h.setSectionResizeMode(1, QHeaderView.Stretch)

        self._layout.addWidget(self.tree)

    def load_data(self, data: Any) -> None:
        """Populate tree from dict/list."""
        self.tree.clear()
        self._populate(self.tree, data)
        self.tree.expandToDepth(1)

    def _populate(
        self,
        parent: Union[QTreeWidget, QTreeWidgetItem],
        data: Any,
    ) -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                item = JsonTreeItem(parent, str(k), v)
                if isinstance(v, (dict, list)):
                    self._populate(item, v)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                # Use index as key
                item = JsonTreeItem(parent, str(i), v)
                if isinstance(v, (dict, list)):
                    self._populate(item, v)

    def _on_search(self, text: str) -> None:
        """Hide non-matching rows (simple logic)."""
        term = text.lower()
        
        # Helper to recurse
        def _filter(item: QTreeWidgetItem) -> bool:
            # Check self
            key_txt = item.text(0).lower()
            val_txt = item.text(1).lower()
            match = (term in key_txt) or (term in val_txt)
            
            # Check children
            child_match = False
            for i in range(item.childCount()):
                if _filter(item.child(i)):
                    child_match = True
            
            visible = match or child_match
            item.setHidden(not visible)
            
            # If search is active and we match, expand
            if term and visible:
                item.setExpanded(True)
            return visible

        # Top level
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            _filter(root.child(i))


# ------------------------------------------------------------
# Main Dialog
# ------------------------------------------------------------


class JsonEditorDialog(QDialog):
    """
    Dialog with Tree/Source tabs for JSON editing.
    """

    def __init__(
        self,
        parent: Optional[QWidget],
        title: str,
        file_path: Optional[str] = None,
        data: Optional[Any] = None,
        read_only: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(850, 600)

        self._path = file_path
        self._data = data
        self._read_only = read_only

        # Load file if needed
        if self._data is None and self._path:
            try:
                raw = Path(self._path).read_text("utf-8")
                self._data = json.loads(raw)
            except Exception as e:
                self._data = {"error": str(e)}

        self._build_ui()
        self._sync_to_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header Info
        if self._path:
            info = QLabel(f"<b>File:</b> {self._path}")
            info.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            layout.addWidget(info)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        # -- Tab 1: Tree View --
        self.tree_widget = JsonTreeWidget(self)
        self.tabs.addTab(self.tree_widget, "Tree View")

        # -- Tab 2: Source View --
        self.source_edit = QPlainTextEdit()
        # Modern monospace font
        font = QFontDatabase.systemFont(
            QFontDatabase.FixedFont
        )
        # Try to make it slightly nicer if possible
        if "Consolas" in QFontDatabase().families():
            font.setFamily("Consolas")
        elif "Monaco" in QFontDatabase().families():
            font.setFamily("Monaco")
        font.setPointSize(10)
        self.source_edit.setFont(font)

        self.highlighter = JsonHighlighter(
            self.source_edit.document()
        )
        
        if self._read_only:
             self.source_edit.setReadOnly(True)

        self.tabs.addTab(self.source_edit, "Source (JSON)")

        # Sync logic
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Footer
        btn_box = QHBoxLayout()
        btn_box.setSpacing(8)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #7f8c8d;")
        btn_box.addWidget(self.lbl_status, 1)

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._on_save)
        
        # Only enable save if we have a file and it's editable
        can_save = (not self._read_only) and bool(self._path)
        self.btn_save.setEnabled(can_save)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)

        btn_box.addWidget(self.btn_save)
        btn_box.addWidget(self.btn_close)

        layout.addLayout(btn_box)

    def _sync_to_ui(self) -> None:
        """Data -> Tree & Text."""
        if self._data is None:
            return

        # 1. Text
        try:
            txt = json.dumps(
                self._data,
                indent=2,
                ensure_ascii=False,
            )
        except Exception:
            txt = str(self._data)
        self.source_edit.setPlainText(txt)

        # 2. Tree
        self.tree_widget.load_data(self._data)

        # Stats
        if isinstance(self._data, dict):
            n = len(self._data)
            self.lbl_status.setText(f"Root object: {n} keys")
        elif isinstance(self._data, list):
            n = len(self._data)
            self.lbl_status.setText(f"Root list: {n} items")

    def _on_tab_changed(self, idx: int) -> None:
            """
            Handles tab switching between Tree (viewer) and Source (editor).
            
            - Index 0 (Tree): Parses Source text -> Updates Data -> Refreshes Tree.
            - Index 1 (Source): Dumps Data -> Updates Source text (Pretty Print).
            """
            # ---------------------------------------------------------
            # Case 1: Switching TO Tree View (Index 0)
            # ---------------------------------------------------------
            if idx == 0:
                # 1. Get raw text from the source editor
                raw = self.source_edit.toPlainText().strip()
                
                if not raw:
                    # Handle empty/blank text case
                    self.tree_widget.load_data(None)
                    self._data = None
                    self.lbl_status.setText("Source is empty.")
                    return
    
                try:
                    # 2. Attempt to parse JSON
                    data = json.loads(raw)
                    
                    # 3. Update internal data reference
                    self._data = data
                    
                    # 4. Refresh the Tree Widget
                    self.tree_widget.load_data(data)
                    
                    # 5. Update Status
                    self.lbl_status.setText("Synced from source.")
                    
                except json.JSONDecodeError as e:
                    # Handle invalid JSON gracefully
                    # We stay on the Tree tab but show the error
                    self.lbl_status.setText(f"JSON Parse Error: line {e.lineno} - {e.msg}")
                    # Optional: You could pop a warning, but status bar is less intrusive
                    # QMessageBox.warning(self, "Invalid JSON", f"Could not parse JSON:\n{e}")
    
                except Exception as e:
                    # Generic fallback
                    self.lbl_status.setText(f"Error: {str(e)}")
    
            # ---------------------------------------------------------
            # Case 2: Switching TO Source View (Index 1)
            # ---------------------------------------------------------
            elif idx == 1:
                # Sync Data -> Text (Enforces formatting)
                # Since the Tree is currently read-only, self._data is our valid state.
                # We re-dump it to text to ensure the user sees valid, pretty-printed JSON.
                
                if self._data is not None:
                    try:
                        # 1. Serialize data to pretty-printed string
                        pretty_text = json.dumps(
                            self._data,
                            indent=2,
                            ensure_ascii=False
                        )
                        
                        # 2. Update the Text Editor
                        self.source_edit.setPlainText(pretty_text)
                        self.lbl_status.setText("Synced to source (formatted).")
                        
                    except TypeError as e:
                        # Handle cases where data isn't serializable (unlikely if loaded from JSON)
                        self.lbl_status.setText(f"Serialization Error: {e}")
                else:
                    # If data is None, just clear the text
                    self.source_edit.setPlainText("")

    def _on_save(self) -> None:
        """Save text content to file."""
        if not self._path:
            return

        # Ensure valid JSON
        raw = self.source_edit.toPlainText()
        try:
            obj = json.loads(raw)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Invalid JSON",
                f"Cannot save invalid JSON:\n{e}",
            )
            return

        try:
            # Re-serialize to be safe/pretty
            pretty = json.dumps(
                obj,
                indent=2,
                ensure_ascii=False,
            )
            Path(self._path).write_text(
                pretty,
                encoding="utf-8",
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Failed",
                str(e),
            )

    def get_data(self) -> Any:
        # Return current state
        # (Make sure to parse text if they edited text)
        raw = self.source_edit.toPlainText()
        try:
            return json.loads(raw)
        except Exception:
            return self._data


# ------------------------------------------------------------
# Public Helper
# ------------------------------------------------------------


def open_json_editor(
    parent: Optional[QWidget],
    *,
    title: str,
    path: Optional[str] = None,
    data: Optional[Any] = None,
    read_only: bool = True,
) -> Tuple[bool, Optional[Any]]:
    """
    Opens the modern JSON viewer/editor.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget.
    title : str
        Dialog title.
    path : str, optional
        File path to load/save.
    data : Any, optional
        Direct data object (if path not provided).
    read_only : bool
        If True, disables editing/saving.

    Returns
    -------
    saved : bool
        True if user clicked Save (and file wrote successfully).
    data : Any
        The final data object (or None).
    """
    dlg = JsonEditorDialog(
        parent,
        title=title,
        file_path=path,
        data=data,
        read_only=read_only,
    )
    code = dlg.exec_()
    
    # If accepted (Save clicked), return True
    # If rejected (Close clicked), return False
    return (code == QDialog.Accepted), dlg.get_data()