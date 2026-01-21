# geoprior/ui/console/dock.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QMessageBox,
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

        self._mirror = bool(mirror_to_all)
        self._seq: Dict[str, int] = {}
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
        
        root.addWidget(self.actions, 0)

        self.tabs = QTabWidget(host)
        self.tabs.setObjectName("consoleTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._sync_actions)
        root.addWidget(self.tabs, 1)

        self.setWidget(host)

        # Create Main + All (not closable)
        self._main = self.new_session(K_MAIN.kind, "Main")
        self._all = self.new_session(K_ALL.kind, "All")
        
        # Backward-compatible aliases used by app.py
        self._main_id = self._main
        self._all_id = self._all

    def _sync_actions(self, *_: object) -> None:
        s = self._current_session()
        if not s:
            return
    
        can_save = bool(getattr(s, "_out_dir", None))
        self.actions.set_states(
            wrap=bool(s.wrap_lines),
            follow=bool(s.follow_tail),
            can_save=bool(can_save),
        )

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
        ok = s.find_next(text)
        if not ok:
            s.view.lbl_prog.setText("Not found")
    
    
    def _find_prev(self, text: str) -> None:
        s = self._current_session()
        if not s:
            return
        ok = s.find_prev(text)
        if not ok:
            s.view.lbl_prog.setText("Not found")

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
        sess = ConsoleSession(
            key,
            view,
            title=title,
            parent=self,
        )
        self._sessions[key] = sess
        self.tabs.addTab(view, title)
        return key

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
            if self._mirror and kind != "all":
                s_all = self.session(self._all)
                s_all.log(f"[{kind}] {m}")
        
        th.log_updated.connect(_on_log)

        th.status_updated.connect(sess.status)
        th.progress_changed.connect(sess.progress)

        def _on_err(msg: str) -> None:
            sess.log(f"[ERROR] {msg}")
            sess.finish(False, "Failed.")

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
        th.finished.connect(_on_done)

        return key

    # inside ConsoleDock (dock.py)


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
