# geoprior/ui/console/dock.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QDockWidget,
    QVBoxLayout,
    QTabWidget,
    QMessageBox,
    QHBoxLayout,
    QLabel,
    QStyle,
    QToolButton,
    QWidget,
    QTabBar, 
    QMenu,
    QAction
)

from .kinds import KINDS, K_MAIN, K_ALL
from .session import (
    SessionKey,
    ConsoleSessionView,
    ConsoleSession,
)
from .actions import ConsoleActions


def _infer_out_dir(result: Any) -> Optional[str]:
    """
    Best-effort: derive output folder from job result.

    Supports common keys and manifest_path.
    """
    if not isinstance(result, dict):
        return None

    cand = [
        "run_output_path",
        "run_dir",
        "output_dir",
        "out_dir",
        "results_dir",
        "xfer_out_dir",
        "stage1_dir",
    ]
    for k in cand:
        v = result.get(k)
        if v:
            return str(v)

    mp = result.get("manifest_path")
    if mp:
        try:
            return str(Path(str(mp)).parent)
        except Exception:
            return None

    return None


class ConsoleDockTitleBar(QWidget):
    pin_toggled = pyqtSignal(bool)
    float_clicked = pyqtSignal()
    close_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("consoleTitleBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumHeight(32)

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(8)

        # ----------------------------
        # Left: icon + title
        # ----------------------------
        self._icon = QLabel(self)
        self._icon.setObjectName("consoleTitleIcon")
        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        self._icon.setPixmap(icon.pixmap(16, 16))

        self._title = QLabel("Console", self)
        self._title.setObjectName("consoleTitle")

        left = QWidget(self)
        left_l = QHBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(6)
        left_l.addWidget(self._icon, 0)
        left_l.addWidget(self._title, 0)

        # ----------------------------
        # Middle: status chip
        # ----------------------------
        self.chip = QLabel("Idle", self)
        self.chip.setObjectName("consoleChip")
        self.set_status("Idle", "idle")

        # ----------------------------
        # Right: pin / float / close
        # ----------------------------
        self._ic_dock = self.style().standardIcon(
            QStyle.SP_TitleBarNormalButton
        )
        self._ic_undock = self.style().standardIcon(
            QStyle.SP_TitleBarMaxButton
        )
        self._ic_close = self.style().standardIcon(
            QStyle.SP_TitleBarCloseButton
        )

        self.btn_pin = QToolButton(self)
        self.btn_pin.setObjectName("miniAction")
        self.btn_pin.setCheckable(True)
        self.btn_pin.setText("📌")
        self.btn_pin.setToolTip("Pin console")

        self.btn_float = QToolButton(self)
        self.btn_float.setObjectName("miniAction")
        self.btn_float.setIcon(self._ic_undock)
        self.btn_float.setToolTip("Undock")

        self.btn_close = QToolButton(self)
        self.btn_close.setObjectName("miniAction")
        self.btn_close.setIcon(self._ic_close)
        self.btn_close.setToolTip("Hide console")

        right = QWidget(self)
        right_l = QHBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(4)
        right_l.addWidget(self.btn_pin, 0)
        right_l.addWidget(self.btn_float, 0)
        right_l.addWidget(self.btn_close, 0)

        # Layout: keep chip centered
        root.addWidget(left, 0, Qt.AlignVCenter)
        root.addStretch(1)
        root.addWidget(self.chip, 0, Qt.AlignCenter)
        root.addStretch(1)
        root.addWidget(right, 0, Qt.AlignVCenter)

        self.btn_pin.toggled.connect(self.pin_toggled)
        self.btn_float.clicked.connect(self.float_clicked)
        self.btn_close.clicked.connect(self.close_clicked)

    def set_floating(self, floating: bool) -> None:
        if floating:
            self.btn_float.setIcon(self._ic_dock)
            self.btn_float.setToolTip("Dock")
        else:
            self.btn_float.setIcon(self._ic_undock)
            self.btn_float.setToolTip("Undock")

    def set_status(self, text: str, state: str) -> None:
        self.chip.setText(str(text))
        self.chip.setProperty("state", str(state))
        self.chip.style().unpolish(self.chip)
        self.chip.style().polish(self.chip)
        self.chip.update()

class ConsoleDock(QDockWidget):
    """
    Dockable Console with session tabs.

    - One tab per run/session
    - Supports concurrent runs across kinds
    - Optional "All" mirror tab
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        title: str = "Console",
        mirror_to_all: bool = True,
    ) -> None:
        super().__init__(title, parent)

        self.setObjectName("logDock")
        self.setAllowedAreas(
            Qt.BottomDockWidgetArea
            | Qt.LeftDockWidgetArea
            | Qt.RightDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )

        self._pinned = False

        self._last_float_size = None
        self._show_ts = True
        self._find_case = False
        self._find_regex = False
        self._find_word = False
        self._find_hl = True
        self._seq: Dict[str, int] = {}
        self._mirror = bool(mirror_to_all)
        
        self._titlebar = ConsoleDockTitleBar(self)
        self.setTitleBarWidget(self._titlebar)

        self._titlebar.close_clicked.connect(
            lambda: self.setVisible(False)
        )
        self._titlebar.float_clicked.connect(
            self._toggle_floating
        )
        self._titlebar.pin_toggled.connect(self.set_pinned)

        self.topLevelChanged.connect(self._on_top_level_changed)

        self._sessions: Dict[SessionKey, ConsoleSession] = {}

        host = QWidget(self)
        root = QVBoxLayout(host)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Optional mini-toolbar (keep simple)
        self.actions = ConsoleActions(host)
        self.actions.clear_clicked.connect(self.clear_current)
        self.actions.copy_clicked.connect(self.copy_current)
        self.actions.save_clicked.connect(self.save_current)
        
        self.actions.wrap_toggled.connect(self._wrap_current)
        self.actions.follow_toggled.connect(self._follow_current)
        
        self.actions.find_next.connect(self._find_next)
        self.actions.find_prev.connect(self._find_prev)
        self.actions.pause_toggled.connect(self._pause_current)
        self.actions.open_out_clicked.connect(self._open_out_current)
        self.actions.copy_out_clicked.connect(self._copy_out_current)
        self.actions.ui_limit_requested.connect(self._ui_limit_current)
                
        root.addWidget(self.actions, 0)

        self.tabs = QTabWidget(host)
        self.tabs.setObjectName("consoleTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._sync_actions)


        # Inline scope/status on same row as tabs.
        self._scope_inline = QLabel("", self.tabs)
        self._scope_inline.setObjectName("consoleScopeInline")
        self._scope_inline.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        self._scope_inline.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        self.tabs.setCornerWidget(
            self._scope_inline,
            Qt.TopRightCorner,
        )

        
        root.addWidget(self.tabs, 1)
        self.setWidget(host)
        
        bar = self.tabs.tabBar()
        bar.setContextMenuPolicy(Qt.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._on_tab_menu)

        # Create Main + All (not closable)
        self._main = self.new_session(K_MAIN.kind, "Main")
        self._all = self.new_session(K_ALL.kind, "History")
        
        # Backward-compatible aliases used by app.py
        self._main_id = self._main
        self._all_id = self._all
        
        self._protect_core_tabs()
        
        self.actions.timestamps_toggled.connect(self._ts_toggled)
        
        self.actions.find_live.connect(self._on_find_live)
        self.actions.find_cleared.connect(self._on_find_cleared)
        self.actions.find_opts_changed.connect(
            self._on_find_opts
        )
        self.actions.highlight_toggled.connect(
            self._on_find_hl
        )


        # Initial density: docked uses compact UI.
        self._apply_density(self.isFloating())
        self._sync_scope_corner()

    def _on_find_opts(self, case, regex, word) -> None:
        self._find_case = bool(case)
        self._find_regex = bool(regex)
        self._find_word = bool(word)
        self._on_find_live(self.actions.edt_find.text())
    
    def _on_find_hl(self, on: bool) -> None:
        self._find_hl = bool(on)
        self._on_find_live(self.actions.edt_find.text())
    
    def _on_find_cleared(self) -> None:
        s = self._current_session()
        if not s:
            return
        s.clear_find()
        self.actions.set_match(0, 0)
    
    def _on_find_live(self, txt: str) -> None:
        s = self._current_session()
        if not s:
            return
        t = str(txt).strip()
        s.set_find_state(
            t,
            case=self._find_case,
            regex=self._find_regex,
            word=self._find_word,
            highlight=self._find_hl,
        )
        cur, total = s.match_stats()
        self.actions.set_match(cur, total)

    def _ts_toggled(self, on: bool) -> None:
        self._show_ts = bool(on)
        for s in self._sessions.values():
            try:
                s.set_show_timestamps(self._show_ts)
            except Exception:
                pass

        self._sync_scope_corner()

    def _session_from_widget(
        self, w: QWidget
    ) -> Optional[ConsoleSession]:
        for s in self._sessions.values():
            if s.view is w:
                return s
        return None
    
    def _on_tab_menu(self, pos) -> None:
        idx = self.tabs.tabBar().tabAt(pos)
        if idx < 0:
            return
        w = self.tabs.widget(idx)
        if w is None:
            return
    
        sess = self._session_from_widget(w)
        if not sess:
            return
    
        # no pin menu for Main/History
        if sess.key in (self._main, self._all):
            return
    
        m = QMenu(self)
        a = QAction("Pin tab", m)
        a.setCheckable(True)
        a.setChecked(bool(sess.pinned))
        a.toggled.connect(
            lambda on, ss=sess: self._set_pinned_tab(ss, on)
        )
        m.addAction(a)
        m.exec_(self.tabs.tabBar().mapToGlobal(pos))
    
    def _set_pinned_tab(
        self, sess: ConsoleSession, on: bool
    ) -> None:
        sess.pinned = bool(on)
        self._refresh_tab(sess)
        
    def _pause_current(self, on: bool) -> None:
        s = self._current_session()
        if not s:
            return
        s.set_paused(on)
        self._sync_actions()

    def _protect_core_tabs(self) -> None:
        bar = self.tabs.tabBar()
        for key in (self._main, self._all):
            try:
                idx = self.tabs.indexOf(self.session(key).view)
            except Exception:
                continue
            if idx >= 0:
                bar.setTabButton(idx, QTabBar.RightSide, None)
        
    def _open_out_current(self) -> None:
        s = self._current_session()
        if not s:
            return
        out = s.out_dir()
        if not out:
            QMessageBox.information(
                self, "No output dir",
                "No output folder was detected yet.",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(out))
    
    def _copy_out_current(self) -> None:
        s = self._current_session()
        if not s:
            return
        out = s.out_dir() or ""
        if not out:
            QMessageBox.information(
                self, "No output dir",
                "No output folder was detected yet.",
            )
            return
        try:
            from PyQt5.QtWidgets import QApplication
            QApplication.clipboard().setText(out)
        except Exception:
            return
    
    def _ui_limit_current(self, n: int) -> None:
        s = self._current_session()
        if s:
            s.set_ui_limit(n)
        self._sync_scope_corner()


    def _toggle_floating(self) -> None:
        if self.isFloating():
            try:
                self._last_float_size = self.size()
            except Exception:
                pass
        self.setFloating(not self.isFloating())

    def _on_top_level_changed(self, floating: bool) -> None:
        if hasattr(self, "_titlebar"):
            self._titlebar.set_floating(bool(floating))

        self._apply_density(bool(floating))

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if self.isFloating():
            try:
                self._last_float_size = self.size()
            except Exception:
                pass
        self._sync_scope_corner()

    def _apply_density(self, floating: bool) -> None:
        compact = not bool(floating)

        try:
            self.actions.set_compact(compact)
        except Exception:
            pass

        for sess in self._sessions.values():
            try:
                sess.view.set_compact_ui(compact)
            except Exception:
                pass

        if floating:
            try:
                sz = getattr(self, "_last_float_size", None)
                if sz is not None:
                    self.resize(sz)
                else:
                    self.resize(1120, 620)
            except Exception:
                try:
                    self.resize(1120, 620)
                except Exception:
                    pass

        self._sync_scope_corner()

    def _set_scope_label(self, full: str, compact: str) -> None:
        if not hasattr(self, "_scope_inline"):
            return

        use_compact = not bool(self.isFloating())
        txt = str(compact) if use_compact else str(full)
        tip = str(full)

        lab = self._scope_inline
        lab.setToolTip(tip)

        max_w = 340 if use_compact else 520
        try:
            lab.setMaximumWidth(int(max_w))
        except Exception:
            pass

        fm = lab.fontMetrics()
        el = fm.elidedText(
            txt,
            Qt.ElideLeft,
            int(max_w),
        )
        lab.setText(el)

    def _sync_scope_corner(self) -> None:
        s = self._current_session()
        if not s:
            return
        self._set_scope_label(
            s.scope_text(),
            s.scope_text_compact(),
        )

    def _on_scope(
        self,
        sess: ConsoleSession,
        full: str,
        compact: str,
    ) -> None:
        cur = self._current_session()
        if cur is not sess:
            return
        self._set_scope_label(full, compact)

    def set_pinned(self, pinned: bool) -> None:
        self._pinned = bool(pinned)
        if self._pinned and self.isFloating():
            self.setFloating(False)

        feats = (
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        if self._pinned:
            feats = QDockWidget.DockWidgetMovable

        self.setFeatures(feats)
        self.show()

    def _sync_titlebar(self) -> None:
        if not hasattr(self, "_titlebar"):
            return
        s = self._current_session()
        if not s:
            return

        # Keep title chip stable: state-only.
        paused = False
        try:
            paused = bool(s.is_paused())
        except Exception:
            paused = bool(getattr(s, "_paused", False))

        if paused:
            state = "paused"
            txt = "Paused"
        elif bool(getattr(s, "_running", False)):
            state = "running"
            txt = "Running"
        else:
            ok = getattr(s, "_ok", None)
            if ok is True:
                state = "done"
                txt = "Done"
            elif ok is False:
                state = "failed"
                txt = "Failed"
            else:
                state = "idle"
                txt = "Idle"

        self._titlebar.set_status(txt, state)

    def _sync_actions(self, *_: object) -> None:
        s = self._current_session()
        if not s:
            return

        can_save = bool(getattr(s, "_out_dir", None))
        self.actions.set_states(
            wrap=bool(s.wrap_lines),
            follow=bool(s.follow_tail),
            paused=bool(s.is_paused()),
            pending=int(s.pending_count()),
            can_save=bool(can_save),
        )
        self._sync_titlebar()
        self._sync_scope_corner()
        
    def _match_stats(self, s: ConsoleSession, t: str) -> tuple[int, int]:
        term = str(t).strip()
        if not term:
            return 0, 0
        txt = s.view.log.toPlainText()
        total = txt.count(term)
        if total <= 0:
            return 0, 0
        pos = s.view.log.textCursor().selectionStart()
        cur = txt[:pos].count(term) + 1
        cur = min(cur, total)
        return int(cur), int(total)

    def copy_current(self) -> None:
        s = self._current_session()
        if s:
            s.copy_to_clipboard()
    
    
    def _wrap_current(self, enabled: bool) -> None:
        s = self._current_session()
        if s:
            s.set_wrap(enabled)
    
    
    def _follow_current(self, enabled: bool) -> None:
        s = self._current_session()
        if s:
            s.set_follow_tail(enabled)
    
    def _find_next(self, text: str) -> None:
        s = self._current_session()
        if not s:
            return
        t = str(text).strip()
        s.set_find_state(
            t,
            case=self._find_case,
            regex=self._find_regex,
            word=self._find_word,
            highlight=self._find_hl,
        )
        ok = s.find_next(
            t,
            case=self._find_case,
            regex=self._find_regex,
            word=self._find_word,
        )
        if not ok:
            s.view.set_prog_text("Not found")
            s.clear_find()
            self.actions.set_match(0, 0)
            return
        cur, total = s.match_stats()
        self.actions.set_match(cur, total)
    
    def _find_prev(self, text: str) -> None:
        s = self._current_session()
        if not s:
            return
        t = str(text).strip()
        s.set_find_state(
            t,
            case=self._find_case,
            regex=self._find_regex,
            word=self._find_word,
            highlight=self._find_hl,
        )
        ok = s.find_prev(
            t,
            case=self._find_case,
            regex=self._find_regex,
            word=self._find_word,
        )
        if not ok:
            s.view.set_prog_text("Not found")
            s.clear_find()
            self.actions.set_match(0, 0)
            return
        cur, total = s.match_stats()
        self.actions.set_match(cur, total)

    # ---------------------------
    # Session creation/access
    # ---------------------------
    def _next_seq(self, kind: str) -> int:
        n = int(self._seq.get(kind, 0)) + 1
        self._seq[kind] = n
        return n

    def new_session(self, kind: str, title: str) -> SessionKey:
        seq = self._next_seq(kind)
        key = SessionKey(kind=kind, seq=seq)

        view = ConsoleSessionView(self.tabs)

        try:
            view.set_compact_ui(not self.isFloating())
        except Exception:
            pass
        sess = ConsoleSession(
            key,
            view,
            title=title,
            parent=self,
        )


        def _scope_cb(full: str, compact: str) -> None:
            self._on_scope(sess, full, compact)

        sess.scope_changed.connect(_scope_cb)
        sess.log_mgr.pending_changed.connect(
            lambda n, ss=sess: self._on_pending(ss, n)
        )
        sess.set_show_timestamps(self._show_ts)
        
        self._sessions[key] = sess
        self.tabs.addTab(view, title)
        return key
    
    def _on_pending(
        self,
        sess: ConsoleSession,
        n: int,
    ) -> None:
        cur = self._current_session()
        if cur is sess:
            self.actions.set_pending(int(n))

    def session(self, key: SessionKey) -> ConsoleSession:
        return self._sessions[key]

    def _current_session(self) -> Optional[ConsoleSession]:
        w = self.tabs.currentWidget()
        if w is None:
            return None
        for s in self._sessions.values():
            if s.view is w:
                return s
        return None

    # ---------------------------
    # User actions
    # ---------------------------
    def clear_current(self) -> None:
        s = self._current_session()
        if s:
            s.clear()

    def save_current(self) -> None:
        s = self._current_session()
        if not s:
            return
        out = s.save_log()
        if not out:
            QMessageBox.information(
                self,
                "No output dir",
                "No output folder was detected yet.",
            )
            return
        QMessageBox.information(
            self,
            "Saved",
            f"Saved:\n{out}",
        )

    def _close_tab(self, idx: int) -> None:
        w = self.tabs.widget(idx)
        if w is None:
            return

        # prevent closing Main/All
        if w is self.session(self._main).view:
            return
        if w is self.session(self._all).view:
            return

        key = None
        for k, s in self._sessions.items():
            if s.view is w:
                key = k
                break

        self.tabs.removeTab(idx)
        w.deleteLater()
        if key:
            self._sessions.pop(key, None)
            
    def _refresh_tab(self, sess: ConsoleSession) -> None:
        idx = self.tabs.indexOf(sess.view)
        if idx >= 0:
            self.tabs.setTabText(idx, sess.tab_text())

    # ---------------------------
    # Core: bind thread -> session
    # ---------------------------
    def bind_thread(
        self,
        th: Any,
        *,
        kind: str,
        title: Optional[str] = None,
        start_msg: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> SessionKey:
        spec = KINDS.get(kind)
        label = title or (spec.label if spec else kind)

        # If you want multiple runs, we create new session each time
        key = self.new_session(kind, label)
        sess = self.session(key)
        self._refresh_tab(sess)
        
        if meta:
            sess.set_meta(**meta)

        # activate tab
        self.tabs.setCurrentWidget(sess.view)

        # start state
        smsg = start_msg or f"{label}: running…"
        sess.start(smsg)

        # Wire YOUR threads.py signals

        def _on_log(m: str) -> None:
            sess.log(m)
            self._refresh_tab(sess)
        
            if self._mirror and kind != "all":
                s_all = self.session(self._all)
                s_all.log(f"[{kind}] {m}")
                self._refresh_tab(s_all)
                
        th.log_updated.connect(_on_log)

        th.status_updated.connect(sess.status)
        th.progress_changed.connect(sess.progress)

        def _on_err(msg: str) -> None:
            sess.log(f"[ERROR] {msg}")
            sess.finish(False, "Failed.")
            self._sync_titlebar()

        th.error_occurred.connect(_on_err)

        def _on_res(result: dict) -> None:
            out = _infer_out_dir(result)
            sess.set_out_dir(out)
            self._sync_actions()

            # Optional: auto-save on finish if you want
            # sess.save_log()
            
        th.results_ready.connect(_on_res)

        def _on_done() -> None:
            if sess._ok is None:
                sess.finish(True, "Done.")
            self._sync_titlebar()
            
        th.finished.connect(_on_done)

        def _on_status(m: str) -> None:
            sess.status(m)
            if self._current_session() is sess:
                self._sync_titlebar()

        th.status_updated.connect(_on_status)


        return key


    def _current_key(self) -> SessionKey:
        s = self._current_session()
        if s is not None:
            return s.key
        return self._main
    
    
    def _key_from_any(self, key: Any) -> SessionKey:
        # NEW: None -> current tab (fallback main)
        if key is None:
            return self._current_key()
    
        if isinstance(key, SessionKey):
            return key
    
        if isinstance(key, str):
            k = key.strip().lower()
            if k in ("current", "active", "tab"):
                return self._current_key()
            if k in ("main", K_MAIN.kind):
                return self._main
            if k in ("all", K_ALL.kind):
                return self._all
    
        return self._main
    
    def _route(
        self,
        key: Any,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        k = self._key_from_any(key)
        try:
            sess = self.session(k)
            fn = getattr(sess, method, None)
            if callable(fn):
                fn(*args, **kwargs)
        except Exception:
            return

    
    def _mirror_log(self, src: SessionKey, msg: str) -> None:
        if not self._mirror:
            return
        if src == self._all:
            return
        try:
            self.session(self._all).log(
                f"[{src.kind}] {msg}"
            )
        except Exception:
            return

    def log_to(self, key: Any, msg: str) -> None:
        k = self._key_from_any(key)
        self._route(k, "log", str(msg))
        try:
            self._refresh_tab(self.session(k))
        except Exception:
            pass
        self._mirror_log(k, str(msg))

    
    def status_to(self, key: Any, msg: str) -> None:
        self._route(key, "status", str(msg))
    
    
    def progress_to(
        self,
        key: Any,
        frac: float,
        msg: str = "",
    ) -> None:
        self._route(key, "progress", frac, str(msg))

    # ---------------------------
    # Mirror (optional)
    # ---------------------------
    def mirror_line(self, kind: str, msg: str) -> None:
        if not self._mirror:
            return
        s = self.session(self._all)
        s.log(f"[{kind}] {msg}")
