# geoprior/ui/console/session.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
import re 
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple 

from PyQt5.QtCore import Qt, QObject, QEvent, pyqtSignal
from PyQt5.QtGui import (
    QTextCursor,
    QTextDocument,
    QTextCharFormat,
    QColor,
)
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QApplication, 
    QToolButton,
    QTextEdit,
    QSizePolicy,
)

from ...utils.clock_timer import RunClockTimer
from ..log_manager import LogManager
from .highlighter import ConsoleHighlighter


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

        # --- Header widget (so we can hide it fully) ---
        self._root = root
        
        self._head = QWidget(self)
        self._head.setObjectName("consoleHead")
        
        head = QHBoxLayout(self._head)
        head.setContentsMargins(8, 6, 8, 0)
        head.setSpacing(8)
        
        self.lbl_status = QLabel("")
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
        
        root.addWidget(self._head, 0)

        self.lbl_meta = QLabel("")
        self.lbl_meta.setObjectName("consoleMeta")
        self.lbl_meta.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )
        self.lbl_meta.setVisible(False)
        root.addWidget(self.lbl_meta, 0)

        # Scope line (Step 4)
        self.lbl_scope = QLabel("")
        self.lbl_scope.setObjectName("consoleScope")
        self.lbl_scope.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )
        self.lbl_scope.setVisible(False)
        root.addWidget(self.lbl_scope, 0)

        self.log = QPlainTextEdit(self)
        self.log.setObjectName("logWidget")
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

        # Jump-to-bottom mini button (Step 4)
        self.btn_jump = QToolButton(self.log.viewport())
        self.btn_jump.setObjectName("consoleJumpBottom")
        self.btn_jump.setText("⇣")
        self.btn_jump.setToolTip("Jump to bottom")
        self.btn_jump.setCursor(Qt.PointingHandCursor)
        self.btn_jump.setVisible(False)
        self.btn_jump.clicked.connect(self._jump_bottom)

        sb = self.log.verticalScrollBar()
        sb.valueChanged.connect(self._update_jump_vis)
        sb.rangeChanged.connect(
            lambda *_: self._update_jump_vis()
        )

        self.log.viewport().installEventFilter(self)
        self._pos_jump()
        
        # --- Status bar (bottom) ---
        foot = QHBoxLayout()
        self._foot = foot
        foot.setContentsMargins(8, 0, 8, 8)
        foot.setSpacing(8)
        
        self.chip = QLabel("Idle", self)
        self.chip.setObjectName("consoleStatusChip")
        self.chip.setAlignment(Qt.AlignCenter)
        self.chip.setMinimumWidth(64)
        foot.addWidget(self.chip, 0)
        
        self.lbl_prog = QLabel("", self)
        self.lbl_prog.setObjectName("consoleProgText")
        self.lbl_prog.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )
        self._prog_full = ""
        foot.addWidget(self.lbl_prog, 1)
        
        self.lbl_pct = QLabel("0 %", self)
        self.lbl_pct.setObjectName("consolePct")
        self.lbl_pct.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        foot.addWidget(self.lbl_pct, 0)
        
        self.bar = QProgressBar(self)
        self.bar.setObjectName("consoleProgress")
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)  # thin bar
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setMaximumWidth(180)
        foot.addWidget(self.bar, 0)
        
        root.addLayout(foot)

        self._ui_compact = False

    def set_status_line(self, text: str) -> None:
        """Set header status line (hidden in compact)."""
        t = str(text or "").strip()
        self.lbl_status.setText(t)
        c = bool(getattr(self, "_ui_compact", False))
        self.lbl_status.setVisible((not c) and bool(t))

    def set_compact_ui(self, compact: bool) -> None:
        c = bool(compact)
        self._ui_compact = c
    
        # Collapse header completely
        self._head.setVisible(not c)
    
        # Meta/scope already widgets → hide removes space
        self.lbl_meta.setVisible((not c) and bool(self.lbl_meta.text()))
        self.lbl_scope.setVisible(False)
    
        # Footer: compact -> thin progress only
        self.chip.setVisible(not c)
        self.lbl_prog.setVisible(not c)
        self.lbl_pct.setVisible(not c)
    
        # spacing/margins
        try:
            self._root.setSpacing(6 if not c else 0)
        except Exception:
            pass
    
        try:
            if c:
                self._foot.setContentsMargins(0, 0, 0, 0)
            else:
                self._foot.setContentsMargins(8, 0, 8, 8)
        except Exception:
            pass
    
        if c:
            self.bar.setFixedHeight(4)
            self.bar.setMaximumWidth(16777215)
            self.bar.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Fixed,
            )
        else:
            self.bar.setFixedHeight(8)
            self.bar.setMaximumWidth(180)
            self.bar.setSizePolicy(
                QSizePolicy.Fixed,
                QSizePolicy.Fixed,
            )
    
        # Header status visibility
        self.lbl_status.setVisible(
            (not c) and bool(self.lbl_status.text().strip())
        )
        self.timer.setVisible((not c) and self.timer.isVisible())

    
    def set_prog_text(self, text: str) -> None:
        self._prog_full = str(text or "")
        fm = self.lbl_prog.fontMetrics()
        w = max(20, self.lbl_prog.width())
        self.lbl_prog.setText(
            fm.elidedText(
                self._prog_full,
                Qt.ElideRight,
                w,
            )
        )
    
    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self.set_prog_text(self._prog_full)

    def eventFilter(self, obj, ev) -> bool:
        if obj is self.log.viewport():
            if ev.type() == QEvent.Resize:
                self._pos_jump()
        return super().eventFilter(obj, ev)

    def _pos_jump(self) -> None:
        if not hasattr(self, "btn_jump"):
            return
        self.btn_jump.adjustSize()
        m = 10
        vp = self.log.viewport()
        x = max(0, vp.width() - self.btn_jump.width() - m)
        y = max(0, vp.height() - self.btn_jump.height() - m)
        self.btn_jump.move(x, y)

    def _update_jump_vis(self) -> None:
        sb = self.log.verticalScrollBar()
        show = sb.value() < (sb.maximum() - 2)
        self.btn_jump.setVisible(bool(show))
        if show:
            self._pos_jump()

    def _jump_bottom(self) -> None:
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.btn_jump.setVisible(False)


class ConsoleSession(QObject):
    """
    One run session (tab) + helpers.
    """

    scope_changed = pyqtSignal(str, str)

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
        self._hl = ConsoleHighlighter(self.view.log.document())
        
        self.wrap_lines = False
        self.follow_tail = True
        
        self._out_dir: Optional[str] = None
        self._running: bool = False
        self._ok: Optional[bool] = None
        self.n_lines = 0
        self.n_warn = 0
        self.n_err = 0
        self.pinned = False

        self._paused = False
        self._pre_pause_text = ""
        self._scope_full = ""
        self._scope_compact = ""
        self.log_mgr.pending_changed.connect(
            self._on_pending_changed
        )
        
        self._find_text = ""
        self._find_case = False
        self._find_regex = False
        self._find_word = False
        self._find_spans: List[Tuple[int, int]] = []
        self._set_chip("Idle", "idle")
        self._set_busy(False)
        self.update_scope()

    def _chip_state(self) -> tuple[str, str]:
        if self._paused:
            return "paused", "Paused"
        if self._running:
            return "running", "Running"
        if self._ok is True:
            return "done", "Done"
        if self._ok is False:
            return "failed", "Failed"
        return "idle", "Idle"
    
    
    def _sync_chip(self) -> None:
        state, text = self._chip_state()
        self.view.chip.setText(text)
        self.view.chip.setProperty("state", state)
        self.view.chip.style().unpolish(self.view.chip)
        self.view.chip.style().polish(self.view.chip)
        self.view.chip.update()


    def _set_chip(self, text: str, state: str) -> None:
        self.view.chip.setText(str(text))
        self.view.chip.setProperty("state", str(state))
        self.view.chip.style().unpolish(self.view.chip)
        self.view.chip.style().polish(self.view.chip)
        self.view.chip.update()
    
    def _set_busy(self, on: bool) -> None:
        if on:
            self.view.bar.setRange(0, 0)  # busy
            self.view.lbl_pct.setVisible(False)
        else:
            self.view.bar.setRange(0, 100)
            self.view.lbl_pct.setVisible(True)


    def clear_find(self) -> None:
        self._find_text = ""
        self._find_spans = []
        self.view.log.setExtraSelections([])

    def set_find_state(
        self,
        text: str,
        *,
        case: bool,
        regex: bool,
        word: bool,
        highlight: bool,
    ) -> None:
        t = str(text).strip()
        self._find_text = t
        self._find_case = bool(case)
        self._find_regex = bool(regex)
        self._find_word = bool(word)

        self._find_spans = self._build_spans(
            t,
            case=self._find_case,
            regex=self._find_regex,
            word=self._find_word,
        )

        if not highlight or not t:
            self.view.log.setExtraSelections([])
            return

        self._apply_highlights(self._find_spans)

    def match_stats(self) -> Tuple[int, int]:
        total = int(len(self._find_spans))
        if total <= 0:
            return 0, 0

        cur = self.view.log.textCursor()
        a = int(cur.selectionStart())
        b = int(cur.selectionEnd())

        idx = 0
        for i, (s, e) in enumerate(self._find_spans, 1):
            if a == s and b == e:
                idx = i
                break
        return int(idx), int(total)

    def _build_spans(
        self,
        term: str,
        *,
        case: bool,
        regex: bool,
        word: bool,
    ) -> List[Tuple[int, int]]:
        t = str(term).strip()
        if not t:
            return []

        txt = self.view.log.toPlainText()

        if regex:
            pat = t
        else:
            pat = re.escape(t)

        if word:
            pat = rf"\b(?:{pat})\b"

        flags = 0 if case else re.IGNORECASE
        try:
            rx = re.compile(pat, flags)
        except re.error:
            return []

        spans: List[Tuple[int, int]] = []
        for m in rx.finditer(txt):
            spans.append((int(m.start()), int(m.end())))
        return spans

    def _apply_highlights(
        self,
        spans: List[Tuple[int, int]],
    ) -> None:
        sel: List[QTextEdit.ExtraSelection] = []
        if not spans:
            self.view.log.setExtraSelections(sel)
            return

        pal = self.view.log.palette()
        col = QColor(pal.highlight().color())
        col.setAlpha(70)

        fmt = QTextCharFormat()
        fmt.setBackground(col)

        doc = self.view.log.document()
        for a, b in spans:
            c = QTextCursor(doc)
            c.setPosition(a)
            c.setPosition(b, QTextCursor.KeepAnchor)
            ex = QTextEdit.ExtraSelection()
            ex.cursor = c
            ex.format = fmt
            sel.append(ex)

        self.view.log.setExtraSelections(sel)

    def set_paused(self, on: bool) -> None:
        on = bool(on)
        if on == self._paused:
            return
    
        self._paused = on
    
        if on:
            self._pre_pause_text = getattr(
                self.view, "_prog_full", ""
            )
            n = self.log_mgr.pending_count()
            self.view.set_prog_text(f"Paused ({n:,})")
            self._set_chip("Paused", "paused")
        else:
            self.view.set_prog_text(self._pre_pause_text)
            self._pre_pause_text = ""
    
            if self._running:
                self._set_chip("Running", "running")
            else:
                if self._ok is True:
                    self._set_chip("Done", "done")
                elif self._ok is False:
                    self._set_chip("Failed", "failed")
                else:
                    self._set_chip("Idle", "idle")
    
        self.log_mgr.set_paused(on)

        
    def bump_stats(self, msg: str) -> None:
        self.n_lines += 1
        t = str(msg).lower()
    
        if "traceback" in t or "[error]" in t or " error" in t:
            self.n_err += 1
            return
        if "[warn" in t or "warning" in t:
            self.n_warn += 1
    
    def tab_text(self) -> str:
        pin = "📌 " if self.pinned else ""
        base = f"{pin}{self.title}"
    
        parts = [f"{self.n_lines:,}"]
        if self.n_warn:
            parts.append(f"⚠{self.n_warn}")
        if self.n_err:
            parts.append(f"⛔{self.n_err}")
    
        return base + " · " + " ".join(parts)
    
    def scope_text(self) -> str:
        return str(self._scope_full or "")

    def scope_text_compact(self) -> str:
        return str(self._scope_compact or "")

    def update_scope(self) -> None:
        ui = self.log_mgr.ui_limit()
        n = self.log_mgr.cache_size()
        lim = self.log_mgr.cache_limit()
        ts = "on" if self.log_mgr.show_timestamps() else "off"

        self._scope_full = (
            f"Showing last {ui} lines "
            f"({n}/{lim} cached) "
            f"· timestamps {ts}"
        )
        self._scope_compact = (
            f"last {ui} "
            f"({n}/{lim}) "
            f"· ts {ts}"
        )

        # Legacy label (hidden by default).
        self.view.lbl_scope.setText(self._scope_full)

        try:
            self.scope_changed.emit(
                self._scope_full,
                self._scope_compact,
            )
        except Exception:
            pass

    def set_show_timestamps(self, on: bool) -> None:
        self.log_mgr.set_show_timestamps(bool(on))
        self.update_scope()
        
        
    def is_paused(self) -> bool:
        return bool(self._paused)

    def pending_count(self) -> int:
        return int(self.log_mgr.pending_count())

    def _on_pending_changed(self, n: int) -> None:
        if not self._paused:
            return
        if n < 20 or (n % 20 == 0):
            self.view.set_prog_text(f"Paused ({n:,})")

    def set_ui_limit(self, n: int) -> None:
        self.log_mgr.set_ui_limit(n)
        self.update_scope()

    def out_dir(self) -> Optional[str]:
        return self._out_dir

        
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
        c = bool(getattr(self.view, "_ui_compact", False))
        self.view.lbl_meta.setVisible(bool(text) and (not c))
        
    def set_wrap(self, enabled: bool) -> None:
        self.wrap_lines = bool(enabled)
        mode = QPlainTextEdit.WidgetWidth
        if not self.wrap_lines:
            mode = QPlainTextEdit.NoWrap
        self.view.log.setLineWrapMode(mode)
    
    
    def set_follow_tail(self, enabled: bool) -> None:
        self.follow_tail = bool(enabled)
    
    def find_next(
        self,
        text: str,
        *,
        case: bool,
        regex: bool,
        word: bool,
    ) -> bool:
        t = str(text).strip()
        if not t:
            return False

        spans = self._build_spans(
            t, case=case, regex=regex, word=word
        )
        if not spans:
            return False

        cur = self.view.log.textCursor()
        pos = int(cur.selectionEnd())

        nxt = None
        for a, b in spans:
            if a >= pos:
                nxt = (a, b)
                break
        if nxt is None:
            nxt = spans[0]

        a, b = nxt
        c = self.view.log.textCursor()
        c.setPosition(a)
        c.setPosition(b, QTextCursor.KeepAnchor)
        self.view.log.setTextCursor(c)
        self.view.log.ensureCursorVisible()
        return True

    def find_prev(
        self,
        text: str,
        *,
        case: bool,
        regex: bool,
        word: bool,
    ) -> bool:
        t = str(text).strip()
        if not t:
            return False

        spans = self._build_spans(
            t, case=case, regex=regex, word=word
        )
        if not spans:
            return False

        cur = self.view.log.textCursor()
        pos = int(cur.selectionStart())

        prv = None
        for a, b in spans:
            if b <= pos:
                prv = (a, b)
        if prv is None:
            prv = spans[-1]

        a, b = prv
        c = self.view.log.textCursor()
        c.setPosition(a)
        c.setPosition(b, QTextCursor.KeepAnchor)
        self.view.log.setTextCursor(c)
        self.view.log.ensureCursorVisible()
        return True
    
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
        self.view.set_status_line(str(msg))
    
        self._set_chip("Running", "running")
    
        c = bool(getattr(self.view, "_ui_compact", False))
        self.view.timer.setVisible(not c)
        self.view.timer.cancel_hibernate()
        self.view.timer.restart()
    
        # unknown progress by default -> busy
        self.progress(-1.0, "")
    
    
    def finish(self, ok: bool, msg: str) -> None:
        self._running = False
        self._ok = bool(ok)
        self.view.set_status_line(str(msg))
    
        if self._ok:
            self._set_chip("Done", "done")
        else:
            self._set_chip("Failed", "failed")
    
        self.progress(1.0, str(msg))
    
        self.view.timer.stop()
        self.view.timer.setVisible(False)
        self.view.timer.schedule_hibernate(
            timeout_ms=60_000
        )
    
    def progress(self, frac: float, msg: str) -> None:
        try:
            f = float(frac)
        except Exception:
            f = -1.0
    
        # busy mode if unknown / negative / NaN
        if f < 0.0 or (f != f):
            self._set_busy(True)
            if msg:
                self.view.set_prog_text(str(msg))
            return
    
        self._set_busy(False)
    
        f = max(0.0, min(1.0, f))
        pct = int(round(100.0 * f))
        self.view.bar.setValue(pct)
        self.view.lbl_pct.setText(f"{pct:d} %")
    
        if msg:
            self.view.set_prog_text(str(msg))
    
    
    def clear(self) -> None:
        self.log_mgr.clear()
    
        self._running = False
        self._ok = None
    
        self.view.set_status_line("")
        self._set_busy(False)
        self._set_chip("Idle", "idle")
    
        self.view.set_prog_text("")
        self.view.bar.setValue(0)
        self.view.lbl_pct.setText("0 %")
    
        self.view.timer.stop()
        self.view.timer.reset()
    
        self.n_lines = 0
        self.n_warn = 0
        self.n_err = 0
        self.pinned = False
    
        self.update_scope()


    def log(self, msg: str) -> None:
        self.bump_stats(msg)
        self.log_mgr.append(str(msg))
        if self._paused:
            return
        if self.follow_tail:
            self.view.log.moveCursor(QTextCursor.End)
            self.view.log.ensureCursorVisible()


    def status(self, msg: str) -> None:
        self.view.set_status_line(str(msg))


    def save_log(self) -> Optional[str]:
        if not self._out_dir:
            return None
        p = self.log_mgr.save_cache(Path(self._out_dir))
        return str(p)
