# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Set

import math
import json

from PyQt5.QtCore import QEvent, QObject, QSignalBlocker
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QWidget,
)

from ..config.store import GeoConfigStore


ValueFn = Callable[[], Any]
SetFn = Callable[[Any], None]


def _has_focus(w: QWidget) -> bool:
    try:
        return bool(w.hasFocus())
    except Exception:
        return False


@contextmanager
def _blocked(w: QWidget):
    blk = QSignalBlocker(w)
    try:
        yield
    finally:
        del blk

class _FocusOutFilter(QObject):
    def __init__(self, fn: Callable[[], None]) -> None:
        super().__init__()
        self._fn = fn

    def eventFilter(self, obj, ev):  # type: ignore[override]
        if ev.type() == QEvent.FocusOut:
            try:
                self._fn()
            except Exception:
                pass
        return False


def _clean_lines(text: str) -> List[str]:
    out: List[str] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        out.append(s)
    return out


def _list_to_text(v: Any) -> str:
    if not v:
        return ""
    if isinstance(v, (list, tuple)):
        return "\n".join(str(x) for x in v if x is not None)
    return str(v)


def _list2_to_text(v: Any) -> str:
    if not v:
        return ""
    if not isinstance(v, (list, tuple)):
        return str(v)
    lines: List[str] = []
    for row in v:
        if isinstance(row, (list, tuple)):
            lines.append(", ".join(str(x) for x in row))
        else:
            lines.append(str(row))
    return "\n".join(lines)


def _text_to_list(text: str) -> List[str]:
    return _clean_lines(text)


def _text_to_list2(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for ln in _clean_lines(text):
        parts = [p.strip() for p in ln.split(",")]
        parts = [p for p in parts if p]
        if parts:
            rows.append(parts)
    return rows

class _Binding:
    def __init__(
        self,
        *,
        key: str,
        widget: QWidget,
        pull: ValueFn,
        push: SetFn,
        respect_focus: bool,
    ) -> None:
        self.key = key
        self.widget = widget
        self._pull = pull
        self._push = push
        self._respect_focus = respect_focus

    def pull(self) -> Any:
        return self._pull()

    def push(self, value: Any) -> None:
        if self._respect_focus and _has_focus(self.widget):
            return
        self._push(value)


class Binder:
    """
    Bind Qt widgets to a GeoConfigStore.

    The binder listens to store changes and refreshes widgets.
    Widgets push edits back into the store.
    """

    def __init__(
        self,
        store: GeoConfigStore,
        *,
        respect_focus: bool = True,
    ) -> None:
        self.store = store
        self.respect_focus = respect_focus
        self._bindings: Dict[str, List[_Binding]] = {}
        self._mute = 0

        self.store.config_changed.connect(
            self._on_config_changed
        )
        self.store.config_replaced.connect(
            self._on_config_replaced
        )

    # -----------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------
    @contextmanager
    def mute(self):
        """
        Temporarily ignore widget->store pushes.

        Use while you are doing bulk UI updates.
        """
        self._mute += 1
        try:
            yield
        finally:
            self._mute = max(0, self._mute - 1)

    def refresh_all(self) -> None:
        keys = set(self._bindings.keys())
        self._refresh(keys)

    def refresh_keys(self, keys: Set[str]) -> None:
        self._refresh(set(keys or []))

    # -----------------------------------------------------------------
    # Low-level registration
    # -----------------------------------------------------------------
    def _register(self, b: _Binding) -> None:
        self._bindings.setdefault(b.key, []).append(b)
        # Initial push (store -> widget)
        try:
            val = getattr(self.store.cfg, b.key)
            b.push(val)
        except Exception:
            return

    # -----------------------------------------------------------------
    # Slots
    # -----------------------------------------------------------------
    def _on_config_replaced(self, _cfg: object) -> None:
        self.refresh_all()

    def _on_config_changed(self, keys_obj: object) -> None:
        try:
            keys = set(keys_obj or [])
        except Exception:
            keys = set()
        self._refresh(keys)

    def _refresh(self, keys: Set[str]) -> None:
        if not keys:
            return

        for key in keys:
            items = self._bindings.get(key) or []
            for b in items:
                try:
                    val = getattr(self.store.cfg, key)
                    b.push(val)
                except Exception:
                    continue

    # -----------------------------------------------------------------
    # Widget binds (common)
    # -----------------------------------------------------------------
    def bind_line_edit(
        self,
        key: str,
        w: QLineEdit,
        *,
        to_store: Optional[Callable[[str], Any]] = None,
        from_store: Optional[Callable[[Any], str]] = None,
        on: str = "editingFinished",
    ) -> None:
        """
        Bind QLineEdit to store.

        on:
            - "editingFinished" (default, low-noise)
            - "textChanged" (chatty)
        """
        if to_store is None:
            to_store = lambda s: s
        if from_store is None:
            from_store = lambda v: "" if v is None else str(v)

        def pull() -> Any:
            return to_store(w.text())

        def push(v: Any) -> None:
            txt = from_store(v)
            with _blocked(w):
                w.setText(txt)

        b = _Binding(
            key=key,
            widget=w,
            pull=pull,
            push=push,
            respect_focus=self.respect_focus,
        )
        self._register(b)

        def _commit(*_a):
            if self._mute:
                return
            self.store.patch({key: b.pull()})

        if on == "textChanged":
            w.textChanged.connect(_commit)
        else:
            w.editingFinished.connect(_commit)

    def bind_checkbox(
        self,
        key: str,
        w: QCheckBox,
    ) -> None:
        def pull() -> Any:
            return bool(w.isChecked())

        def push(v: Any) -> None:
            with _blocked(w):
                w.setChecked(bool(v))

        b = _Binding(
            key=key,
            widget=w,
            pull=pull,
            push=push,
            respect_focus=False,
        )
        self._register(b)

        def _commit(_state: int):
            if self._mute:
                return
            self.store.patch({key: b.pull()})

        w.stateChanged.connect(_commit)

    def bind_spin_box(
        self,
        key: str,
        w: QSpinBox,
        *,
        to_store: Optional[Callable[[int], Any]] = None,
        from_store: Optional[Callable[[Any], int]] = None,
    ) -> None:
        if to_store is None:
            to_store = lambda x: int(x)
        if from_store is None:
            from_store = lambda v: int(v)

        def pull() -> Any:
            return to_store(int(w.value()))

        def push(v: Any) -> None:
            try:
                iv = from_store(v)
            except Exception:
                return
            with _blocked(w):
                w.setValue(int(iv))

        b = _Binding(
            key=key,
            widget=w,
            pull=pull,
            push=push,
            respect_focus=self.respect_focus,
        )
        self._register(b)

        def _commit(_val: int):
            if self._mute:
                return
            self.store.patch({key: b.pull()})

        w.valueChanged.connect(_commit)

    def bind_double_spin_box(
        self,
        key: str,
        w: QDoubleSpinBox,
        *,
        to_store: Optional[Callable[[float], Any]] = None,
        from_store: Optional[Callable[[Any], float]] = None,
    ) -> None:
        if to_store is None:
            to_store = lambda x: float(x)
        if from_store is None:
            from_store = lambda v: float(v)

        def pull() -> Any:
            return to_store(float(w.value()))

        def push(v: Any) -> None:
            try:
                fv = from_store(v)
            except Exception:
                return
            with _blocked(w):
                w.setValue(float(fv))

        b = _Binding(
            key=key,
            widget=w,
            pull=pull,
            push=push,
            respect_focus=self.respect_focus,
        )
        self._register(b)

        def _commit(_val: float):
            if self._mute:
                return
            self.store.patch({key: b.pull()})

        w.valueChanged.connect(_commit)

    
    def bind_combo(
        self,
        key: str,
        w: QComboBox,
        *,
        items=None,
        editable: bool = False,
        none_text: str | None = None,
        use_item_data: bool = True,
        signal: str = "currentIndexChanged",
    ) -> None:
        w.setEditable(bool(editable))
    
        if items is not None:
            with _blocked(w):
                w.clear()
                if none_text is not None:
                    w.addItem(str(none_text), None)
                for label, val in items:
                    w.addItem(str(label), val)
    
        def pull() -> Any:
            txt = w.currentText()
            if none_text is not None:
                if txt == str(none_text):
                    return None
    
            if use_item_data:
                data = w.currentData()
                if data is None and w.isEditable():
                    return txt
                return data
    
            return txt
    
        def push(v: Any) -> None:
            with _blocked(w):
                if v is None and none_text is not None:
                    idx0 = w.findText(str(none_text))
                    if idx0 >= 0:
                        w.setCurrentIndex(idx0)
                        return
                    if w.isEditable():
                        w.setEditText(str(none_text))
                        return
                    if w.count() > 0:
                        w.setCurrentIndex(0)
                    return
    
                if use_item_data and w.count() > 0:
                    idx = _find_index_by_data(w, v)
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                        return
                    if w.isEditable():
                        w.setEditText("" if v is None else str(v))
                    return
    
                txt = "" if v is None else str(v)
                idx = w.findText(txt)
                if idx >= 0:
                    w.setCurrentIndex(idx)
                elif w.isEditable():
                    w.setEditText(txt)
    
        b = _Binding(
            key=key,
            widget=w,
            pull=pull,
            push=push,
            respect_focus=self.respect_focus,
        )
        self._register(b)
    
        def _commit(*_a):
            if self._mute:
                return
            self.store.patch({key: b.pull()})
    
        getattr(w, signal).connect(_commit)
        
        # If editable, commit only when user finishes editing
        if w.isEditable() and w.lineEdit() is not None:
            w.lineEdit().editingFinished.connect(_commit)
            
    def get_value(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self.store.cfg, key)
        except Exception:
            return default

    # -----------------------------------------------------------------
    # Optional number helpers (None support)
    # -----------------------------------------------------------------
    def bind_optional_spin_box(
        self,
        key: str,
        spin: QSpinBox,
        enable: QCheckBox,
    ) -> None:
        """
        Bind Optional[int] as:
            enable checked -> spin value
            enable unchecked -> None
        """

        def pull() -> Any:
            if not enable.isChecked():
                return None
            return int(spin.value())

        def push(v: Any) -> None:
            with _blocked(enable), _blocked(spin):
                if v is None:
                    enable.setChecked(False)
                    spin.setEnabled(False)
                else:
                    enable.setChecked(True)
                    spin.setEnabled(True)
                    spin.setValue(int(v))

        b = _Binding(
            key=key,
            widget=spin,
            pull=pull,
            push=push,
            respect_focus=self.respect_focus,
        )
        self._register(b)

        def _commit(*_a):
            if self._mute:
                return
            self.store.patch({key: b.pull()})

        enable.toggled.connect(_commit)
        spin.valueChanged.connect(_commit)

    def bind_optional_double_spin_box(
        self,
        key: str,
        spin: QDoubleSpinBox,
        enable: QCheckBox,
    ) -> None:
        """
        Bind Optional[float] as:
            enable checked -> spin value
            enable unchecked -> None
        """

        def pull() -> Any:
            if not enable.isChecked():
                return None
            return float(spin.value())

        def push(v: Any) -> None:
            with _blocked(enable), _blocked(spin):
                if v is None:
                    enable.setChecked(False)
                    spin.setEnabled(False)
                else:
                    enable.setChecked(True)
                    spin.setEnabled(True)
                    spin.setValue(float(v))

        b = _Binding(
            key=key,
            widget=spin,
            pull=pull,
            push=push,
            respect_focus=self.respect_focus,
        )
        self._register(b)

        def _commit(*_a):
            if self._mute:
                return
            self.store.patch({key: b.pull()})

        enable.toggled.connect(_commit)
        spin.valueChanged.connect(_commit)

    def bind_plain_text(
        self,
        key: str,
        w: QPlainTextEdit,
        *,
        to_store: Optional[Callable[[str], Any]] = None,
        from_store: Optional[Callable[[Any], str]] = None,
        default_on_empty: Any = None,
    ) -> None:
        if to_store is None:
            to_store = lambda s: s
        if from_store is None:
            from_store = lambda v: "" if v is None else str(v)
    
        def pull() -> Any:
            txt = w.toPlainText()
            if default_on_empty is not None and not txt.strip():
                return default_on_empty
            return to_store(txt)
    
        def push(v: Any) -> None:
            txt = from_store(v)
            with _blocked(w):
                w.setPlainText(txt)
    
        b = _Binding(
            key=key,
            widget=w,
            pull=pull,
            push=push,
            respect_focus=self.respect_focus,
        )
        self._register(b)
    
        def _commit() -> None:
            if self._mute:
                return
            try:
                val = b.pull()
            except Exception as exc:
                try:
                    self.store.error_raised.emit(str(exc))
                except Exception:
                    pass
                return
            self.store.patch({key: val})

        filt = _FocusOutFilter(_commit)
        w.installEventFilter(filt)
        setattr(w, "_geo_focus_filter", filt)


    def bind_list_text(
        self,
        key: str,
        w: QPlainTextEdit,
        *,
        item_type: Any = None,
    ) -> None:
        self.bind_plain_text(
            key,
            w,
            to_store=lambda t: _text_to_typed_list(t, item_type),
            from_store=_list_to_text,
            default_on_empty=[],
        )

    
    def bind_list2_text(
        self,
        key: str,
        w: QPlainTextEdit,
    ) -> None:
        self.bind_plain_text(
            key,
            w,
            to_store=_text_to_list2,
            from_store=_list2_to_text,
            default_on_empty=[],
        )
    
    
    def bind_json_text(
        self,
        key: str,
        w: QPlainTextEdit,
        *,
        default_on_empty: Any = None,
    ) -> None:
        def _to_store(txt: str) -> Any:
            s = (txt or "").strip()
            if default_on_empty is not None and not s:
                return default_on_empty
            if not s:
                return default_on_empty
            try:
                return json.loads(s)
            except Exception:
                # fallback: treat as list-of-lines
                return _text_to_list(txt)
    
        def _from_store(v: Any) -> str:
            if v is None:
                return ""
            try:
                return json.dumps(
                    v,
                    indent=2,
                    ensure_ascii=True,
                )
            except Exception:
                return str(v)
    
        self.bind_plain_text(
            key,
            w,
            to_store=_to_store,
            from_store=_from_store,
            default_on_empty=default_on_empty,
        )
    
    
    def bind_optional_list_text(
        self,
        key: str,
        w: QPlainTextEdit,
        enable: QCheckBox,
    ) -> None:
        def pull() -> Any:
            if not enable.isChecked():
                return None
            return _text_to_list(w.toPlainText())
    
        def push(v: Any) -> None:
            with _blocked(enable), _blocked(w):
                if v is None:
                    enable.setChecked(False)
                    w.setEnabled(False)
                    w.setPlainText("")
                else:
                    enable.setChecked(True)
                    w.setEnabled(True)
                    w.setPlainText(_list_to_text(v))
    
        b = _Binding(
            key=key,
            widget=w,
            pull=pull,
            push=push,
            respect_focus=self.respect_focus,
        )
        self._register(b)
    
        def _commit() -> None:
            if self._mute:
                return
            self.store.patch({key: b.pull()})
    
        enable.toggled.connect(lambda *_: _commit())
    
        filt = _FocusOutFilter(_commit)
        w.installEventFilter(filt)
        setattr(w, "_geo_focus_filter_opt", filt)

def _find_index_by_data(w: QComboBox, v: Any) -> int:
    n = w.count()
    for i in range(n):
        if w.itemData(i) == v:
            return i
    return -1

def _coerce_item(s: str, item_type: Any) -> Any:
    if item_type is None:
        return s

    if item_type is int:
        f = float(s)
        if not math.isfinite(f):
            raise ValueError(f"Non-finite int item: {s!r}")
        if not f.is_integer():
            raise ValueError(f"Non-integer item: {s!r}")
        return int(f)

    if item_type is float:
        f = float(s)
        if not math.isfinite(f):
            raise ValueError(f"Non-finite float item: {s!r}")
        return float(f)

    # Generic callable/type
    return item_type(s)


def _text_to_typed_list(text: str, item_type: Any) -> List[Any]:
    items = _clean_lines(text)
    if item_type is None:
        return items

    out: List[Any] = []
    for s in items:
        out.append(_coerce_item(s, item_type))
    return out
