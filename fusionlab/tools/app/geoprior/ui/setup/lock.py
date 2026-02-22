# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.lock

Setup lock/unlock controller.

Goal
----
When "locked", block any UI interaction that can *mutate*
the configuration (inputs + action buttons), while still
allowing non-mutating actions such as "Copy".

This is UI-only: it does not change store values.

How it works
------------
- Walks the widget tree under a given root container.
- Classifies widgets into: allow / block / skip.
- Stores previous enabled states for blocked widgets.
- Disables blocked widgets when locked.
- Restores exact previous enabled states when unlocked.
- Installs an event filter while locked to:
  - re-disable widgets if something re-enables them
  - auto-lock newly added children

Opt-in properties (recommended)
-------------------------------
You can tag widgets with dynamic properties to override
the default classifier:

- lockExempt = True
    Never touched by the lock controller.

- lockRole = "allow" | "copy" | "block"
    Force allow/block.

This is handy for edge cases (custom dialogs, etc.).
"""

from __future__ import annotations

import weakref
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QToolButton,
    QWidget,
)

_LOCK_EXEMPT_PROP = "lockExempt"
_LOCK_ROLE_PROP = "lockRole"


def set_lock_exempt(w: QWidget, on: bool = True) -> None:
    """Mark a widget as never touched by lock/unlock."""
    if w is None:
        return
    w.setProperty(_LOCK_EXEMPT_PROP, bool(on))


def set_lock_role(w: QWidget, role: str) -> None:
    """Force lock role: 'allow'|'copy'|'block'."""
    if w is None:
        return
    w.setProperty(_LOCK_ROLE_PROP, str(role))


class SetupLockController(QObject):
    """Disable/restore Setup UI editability robustly."""

    locked_changed = pyqtSignal(bool)

    def __init__(
        self,
        root: Optional[QWidget] = None,
        *,
        keep_copy: bool = True,
    ) -> None:
        super().__init__(root)

        self._root: Optional[QWidget] = None
        self._locked = False
        self._keep_copy = bool(keep_copy)

        # Previous enabled state for widgets we change.
        self._prev_enabled: "weakref.WeakKeyDictionary[QWidget, bool]"
        self._prev_enabled = weakref.WeakKeyDictionary()

        # Widgets currently watched via eventFilter.
        self._watched: "weakref.WeakSet[QWidget]"
        self._watched = weakref.WeakSet()

        if root is not None:
            self.set_root(root)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def set_root(self, root: QWidget) -> None:
        """Change the container controlled by the lock."""
        if root is self._root:
            return

        # If we move root while locked, unlock first to restore.
        was_locked = self._locked
        if was_locked:
            self.set_locked(False)

        self._root = root
        self._prev_enabled.clear()
        self._watched.clear()

        if was_locked:
            self.set_locked(True)

    def is_locked(self) -> bool:
        return bool(self._locked)

    def set_locked(self, on: bool) -> None:
        on = bool(on)
        if on == self._locked:
            return

        self._locked = on
        if on:
            self._apply_lock()
        else:
            self._restore_unlock()

        self.locked_changed.emit(bool(self._locked))

    def apply_now(self) -> None:
        """Re-apply current lock state (useful after rebuild)."""
        if self._locked:
            self._apply_lock()
        else:
            self._restore_unlock()

    # -----------------------------------------------------------------
    # Internals: lock / unlock
    # -----------------------------------------------------------------
    def _apply_lock(self) -> None:
        root = self._root
        if root is None:
            return

        # Watch root so we can lock new children.
        self._watch(root)

        # Lock all existing descendants.
        for w in root.findChildren(QWidget):
            self._maybe_lock_widget(w)

    def _restore_unlock(self) -> None:
        # Remove event filters first.
        for w in list(self._watched):
            try:
                w.removeEventFilter(self)
            except Exception:
                pass
        self._watched.clear()

        # Restore exact previous enabled states.
        for w, prev in list(self._prev_enabled.items()):
            try:
                w.setEnabled(bool(prev))
            except Exception:
                pass
        self._prev_enabled.clear()

    # -----------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------
    def _maybe_lock_widget(self, w: QWidget) -> None:
        if w is None:
            return

        role = self._role_for(w)
        if role != "block":
            return

        # Store previous enabled state only once.
        if w not in self._prev_enabled:
            try:
                self._prev_enabled[w] = bool(w.isEnabled())
            except Exception:
                self._prev_enabled[w] = True

        try:
            w.setEnabled(False)
        except Exception:
            pass

        self._watch(w)

    def _role_for(self, w: QWidget) -> str:
        """Return 'allow'|'block'|'skip'."""
        # Opt-out.
        if bool(w.property(_LOCK_EXEMPT_PROP)):
            return "skip"

        # Forced role.
        forced = w.property(_LOCK_ROLE_PROP)
        if forced is not None:
            s = str(forced).strip().lower()
            if s in {"allow", "copy"}:
                return "allow"
            if s in {"block", "edit"}:
                return "block"

        # Display-only widgets: keep readable.
        if isinstance(w, (QTextEdit, QPlainTextEdit)):
            try:
                if w.isReadOnly():
                    return "allow"
            except Exception:
                return "allow"

        # Inputs always blocked.
        if isinstance(
            w,
            (
                QLineEdit,
                QComboBox,
                QAbstractSpinBox,
                QCheckBox,
                QRadioButton,
            ),
        ):
            return "block"

        # Buttons: default to block (they can mutate config),
        # except copy-like actions.
        if isinstance(w, (QPushButton, QToolButton, QAbstractButton)):
            if self._keep_copy and self._is_copy_like(w):
                return "allow"
            return "block"

        return "skip"

    def _is_copy_like(self, b: QAbstractButton) -> bool:
        try:
            if bool(b.property("copyAllowed")):
                return True
        except Exception:
            pass

        name = (b.objectName() or "").strip().lower()
        text = (b.text() or "").strip().lower()
        tip = (b.toolTip() or "").strip().lower()

        if "copy" in name:
            return True
        if text.startswith("copy"):
            return True
        if "copy" in tip:
            return True

        return False

    # -----------------------------------------------------------------
    # Robustness: event filter
    # -----------------------------------------------------------------
    def _watch(self, w: QWidget) -> None:
        if w is None:
            return
        if w in self._watched:
            return
        try:
            w.installEventFilter(self)
            self._watched.add(w)
        except Exception:
            return

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:  # noqa: N802
        if not self._locked:
            return False

        # If something re-enables a blocked widget, undo it.
        if ev.type() == QEvent.EnabledChange and isinstance(obj, QWidget):
            w = obj
            if self._role_for(w) == "block" and w.isEnabled():
                QTimer.singleShot(0, lambda: self._force_disable(w))
            return False

        # If new children appear under the root while locked,
        # lock them immediately.
        if ev.type() == QEvent.ChildAdded and self._root is not None:
            if obj is self._root or self._is_descendant(obj, self._root):
                child = getattr(ev, "child", None)
                if callable(child):
                    c = child()
                    if isinstance(c, QWidget):
                        QTimer.singleShot(
                            0,
                            lambda: self._lock_subtree(c),
                        )
            return False

        return False

    def _force_disable(self, w: QWidget) -> None:
        if not self._locked:
            return
        if w is None:
            return
        if self._role_for(w) != "block":
            return
        try:
            w.setEnabled(False)
        except Exception:
            pass

    def _lock_subtree(self, root: QWidget) -> None:
        if root is None:
            return
        self._maybe_lock_widget(root)
        for w in root.findChildren(QWidget):
            self._maybe_lock_widget(w)

    def _is_descendant(self, obj: QObject, root: QWidget) -> bool:
        try:
            p = obj
            while p is not None:
                if p is root:
                    return True
                p = p.parent()
        except Exception:
            return False
        return False


__all__ = [
    "SetupLockController",
    "set_lock_exempt",
    "set_lock_role",
]
