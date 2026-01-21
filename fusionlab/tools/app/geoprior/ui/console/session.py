# geoprior/ui/console/session.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Dict

from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QTextCursor, QTextDocument
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QApplication
)

from ..log_manager import LogManager
from ...utils.clock_timer import RunClockTimer


@dataclass(frozen=True)
class SessionKey:
    kind: str
    seq: int



@dataclass
class SessionMeta:
    kind: str
    seq: int
    title: str
    city_a: Optional[str] = None
    city_b: Optional[str] = None
    dataset: Optional[str] = None
    run_id: Optional[str] = None
    out_dir: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)


class ConsoleSessionView(QWidget):
    """
    One tab: status + timer + log + progress.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        head = QHBoxLayout()
        head.setContentsMargins(8, 6, 8, 0)
        head.setSpacing(8)

        self.lbl_status = QLabel("? Idle")
        self.lbl_status.setObjectName("consoleStatus")
        self.lbl_status.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )
        
        head.addWidget(self.lbl_status, 1)

        self.timer = RunClockTimer(self)
        self.timer.reset()
        self.timer.stop()
        self.timer.setVisible(False)
        head.addWidget(self.timer, 0)
        
        root.addLayout(head)
        
        self.lbl_meta = QLabel("")
        self.lbl_meta.setObjectName("consoleMeta")
        self.lbl_meta.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_meta.setVisible(False)
        root.addWidget(self.lbl_meta, 0)

        self.log = QPlainTextEdit(self)
        self.log.setObjectName("logWidget")
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

        foot = QHBoxLayout()
        foot.setContentsMargins(8, 0, 8, 8)
        foot.setSpacing(8)

        self.lbl_prog = QLabel("")
        self.lbl_prog.setObjectName("consoleProgText")
        self.lbl_prog.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )
        foot.addWidget(self.lbl_prog, 0)

        self.bar = QProgressBar(self)
        self.bar.setMinimumHeight(18)
        self.bar.setTextVisible(False)
        foot.addWidget(self.bar, 1)

        self.lbl_pct = QLabel("0 %")
        self.lbl_pct.setObjectName("consolePct")
        self.lbl_pct.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        foot.addWidget(self.lbl_pct, 0)

        root.addLayout(foot)


class ConsoleSession(QObject):
    """
    One run session (tab) + helpers.
    """

    def __init__(
        self,
        key: SessionKey,
        view: ConsoleSessionView,
        *,
        title: str,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.key = key
        self.title = str(title)
        self.view = view

        self.log_mgr = LogManager(
            self.view.log,
            mode="collapse",
            cache_limit=10_000,
            ui_limit=800,
            log_dir_name="_log",
        )
        self.meta = SessionMeta(
            kind=key.kind,
            seq=key.seq,
            title=title,
        )
        self.wrap_lines = False
        self.follow_tail = True
        
        self._out_dir: Optional[str] = None
        self._running: bool = False
        self._ok: Optional[bool] = None
        
    def set_meta(
        self,
        *,
        city_a: Optional[str] = None,
        city_b: Optional[str] = None,
        dataset: Optional[str] = None,
        run_id: Optional[str] = None,
        out_dir: Optional[str] = None,
    ) -> None:
        if city_a is not None:
            self.meta.city_a = city_a
        if city_b is not None:
            self.meta.city_b = city_b
        if dataset is not None:
            self.meta.dataset = dataset
        if run_id is not None:
            self.meta.run_id = run_id
        if out_dir is not None:
            self.meta.out_dir = out_dir
    
        parts = []
        if self.meta.city_a and self.meta.city_b:
            parts.append(f"{self.meta.city_a} → {self.meta.city_b}")
        elif self.meta.city_a:
            parts.append(self.meta.city_a)
        if self.meta.dataset:
            parts.append(self.meta.dataset)
        if self.meta.run_id:
            parts.append(f"id={self.meta.run_id}")
        if self.meta.out_dir:
            parts.append(self.meta.out_dir)
    
        text = "  •  ".join(parts)
        self.view.lbl_meta.setText(text)
        self.view.lbl_meta.setVisible(bool(text))
        
    def set_wrap(self, enabled: bool) -> None:
        self.wrap_lines = bool(enabled)
        mode = QPlainTextEdit.WidgetWidth
        if not self.wrap_lines:
            mode = QPlainTextEdit.NoWrap
        self.view.log.setLineWrapMode(mode)
    
    
    def set_follow_tail(self, enabled: bool) -> None:
        self.follow_tail = bool(enabled)
    
    
    def find_next(self, text: str) -> bool:
        t = str(text).strip()
        if not t:
            return False
    
        ok = self.view.log.find(t)
        if ok:
            return True
    
        cur = self.view.log.textCursor()
        cur.movePosition(QTextCursor.Start)
        self.view.log.setTextCursor(cur)
        return bool(self.view.log.find(t))
    
    
    def find_prev(self, text: str) -> bool:
        t = str(text).strip()
        if not t:
            return False
    
        opt = QTextDocument.FindBackward
        ok = self.view.log.find(t, opt)
        if ok:
            return True
    
        cur = self.view.log.textCursor()
        cur.movePosition(QTextCursor.End)
        self.view.log.setTextCursor(cur)
        return bool(self.view.log.find(t, opt))
    
    
    def copy_to_clipboard(self) -> None:
        cur = self.view.log.textCursor()
        txt = cur.selectedText()
        txt = txt.replace("\u2029", "\n").strip()
    
        if not txt:
            txt = self.view.log.toPlainText()
    
        QApplication.clipboard().setText(txt)
    
    def set_out_dir(self, path: Optional[str]) -> None:
        self._out_dir = str(path) if path else None
        self.set_meta(out_dir=self._out_dir)



    def start(self, msg: str) -> None:
        self._running = True
        self._ok = None
        self.view.lbl_status.setText(msg)
        self.view.timer.setVisible(True)
        self.view.timer.cancel_hibernate()
        self.view.timer.restart()
        self.progress(0.0, "")

    def finish(self, ok: bool, msg: str) -> None:
        self._running = False
        self._ok = bool(ok)
        self.view.lbl_status.setText(msg)
        self.progress(1.0, msg)
        self.view.timer.stop()
        self.view.timer.schedule_hibernate(timeout_ms=60_000)

    def log(self, msg: str) -> None:
        self.log_mgr.append(str(msg))
        if self.follow_tail:
            self.view.log.moveCursor(QTextCursor.End)
            self.view.log.ensureCursorVisible()

    def status(self, msg: str) -> None:
        self.view.lbl_status.setText(str(msg))

    def progress(self, frac: float, msg: str) -> None:
        try:
            f = float(frac)
        except Exception:
            f = 0.0
        f = max(0.0, min(1.0, f))
        pct = int(round(100.0 * f))
        self.view.bar.setValue(pct)
        self.view.lbl_pct.setText(f"{pct:d} %")
        if msg:
            self.view.lbl_prog.setText(str(msg))

    def clear(self) -> None:
        self.log_mgr.clear()
        self.view.lbl_status.setText("? Idle")
        self.view.lbl_prog.setText("")
        self.view.bar.setValue(0)
        self.view.lbl_pct.setText("0 %")
        self.view.timer.stop()
        self.view.timer.reset()

    def save_log(self) -> Optional[str]:
        if not self._out_dir:
            return None
        p = self.log_mgr.save_cache(Path(self._out_dir))
        return str(p)
