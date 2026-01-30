# geoprior/utils/generic_utils.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple


def open_json_editor(
    parent,
    *,
    title: str,
    path: Optional[str] = None,
    data: Optional[Any] = None,
    read_only: bool = True,
) -> Tuple[bool, Optional[Any]]:
    """
    Opens a pretty JSON viewer/editor dialog.

    Returns
    -------
    saved : bool
        True if user saved to disk.
    obj : Any or None
        Parsed JSON (possibly edited). None if cancelled.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFontDatabase
    from PyQt5.QtWidgets import (
        QDialog,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QVBoxLayout,
        QWidget,
    )

    obj = data
    fp = None if path is None else str(path)

    if obj is None and fp:
        try:
            obj = json.loads(Path(fp).read_text("utf-8"))
        except Exception as e:
            QMessageBox.warning(
                parent,
                title,
                f"Failed to read JSON:\n{e}",
            )
            return False, None

    txt = ""
    try:
        txt = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        txt = str(obj)

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setModal(True)

    root = QVBoxLayout(dlg)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(8)

    if fp:
        lbl = QLabel(f"<b>File:</b> {fp}", dlg)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setWordWrap(True)
        root.addWidget(lbl)

    editor = QPlainTextEdit(dlg)
    editor.setPlainText(txt)
    editor.setReadOnly(bool(read_only))
    editor.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    root.addWidget(editor, 1)

    bar = QWidget(dlg)
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)

    btn_toggle = QPushButton("Edit", dlg)
    btn_pretty = QPushButton("Pretty", dlg)
    btn_save = QPushButton("Save", dlg)
    btn_close = QPushButton("Close", dlg)

    btn_save.setEnabled((not read_only) and bool(fp))

    def _toggle():
        ro = editor.isReadOnly()
        editor.setReadOnly(not ro)
        btn_toggle.setText("Read" if ro else "Edit")
        btn_save.setEnabled((not editor.isReadOnly()) and bool(fp))

    def _pretty():
        try:
            o = json.loads(editor.toPlainText())
            editor.setPlainText(
                json.dumps(o, indent=2, ensure_ascii=False)
            )
        except Exception as e:
            QMessageBox.warning(dlg, title, f"Invalid JSON:\n{e}")

    def _save():
        if not fp:
            return
        try:
            o = json.loads(editor.toPlainText())
        except Exception as e:
            QMessageBox.warning(dlg, title, f"Invalid JSON:\n{e}")
            return
        try:
            Path(fp).write_text(
                json.dumps(o, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            QMessageBox.warning(dlg, title, f"Save failed:\n{e}")
            return
        dlg.done(1)

    btn_toggle.clicked.connect(_toggle)
    btn_pretty.clicked.connect(_pretty)
    btn_save.clicked.connect(_save)
    btn_close.clicked.connect(dlg.reject)

    lay.addWidget(btn_toggle)
    lay.addWidget(btn_pretty)
    lay.addStretch(1)
    lay.addWidget(btn_save)
    lay.addWidget(btn_close)
    root.addWidget(bar)

    code = dlg.exec_()
    if code == 1:
        try:
            return True, json.loads(editor.toPlainText())
        except Exception:
            return True, obj

    try:
        return False, json.loads(editor.toPlainText())
    except Exception:
        return False, obj
