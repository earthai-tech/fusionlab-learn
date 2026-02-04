# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import copy
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from .geoprior_config import GeoPriorConfig
from .prior_schema import FieldKey

_MISSING = object()
_KEY_PAT = re.compile(
    r"^(?P<base>[A-Za-z_][A-Za-z0-9_\.]*?)"
    r"(?:\[(?P<sub>[^\]]+)\])?$"
)

class GeoConfigStore(QObject):
    """
    Reactive store around a live GeoPriorConfig instance.

    Tabs/dialogs should write via this store so we can:
    - emit change signals
    - compute "dirty"/override count
    - batch updates into one notification
    """

    # keys: set[str]
    config_changed = pyqtSignal(object)

    # overrides_count: int
    dirty_changed = pyqtSignal(int)

    # cfg: GeoPriorConfig
    config_replaced = pyqtSignal(object)

    # message: str
    error_raised = pyqtSignal(str)

    def __init__(
        self,
        cfg: GeoPriorConfig,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._valid_keys = set(cfg.__dataclass_fields__.keys())
        self._batch_depth = 0
        self._pending_keys: Set[str] = set()
        self._dirty_count = self._compute_dirty_count()

        # GUI-only keys (not in GeoPriorConfig yet).
        self._extra: Dict[str, Any] = {
            "xfer.view_mode": "map",
            "xfer.city_a_lat": None,
            "xfer.city_a_lon": None,
            "xfer.city_b_lat": None,
            "xfer.city_b_lon": None,
        }

    # -----------------------------------------------------------------
    # Read access
    # -----------------------------------------------------------------
    @property
    def cfg(self) -> GeoPriorConfig:
        return self._cfg

    def overrides_count(self) -> int:
        return self._dirty_count

    def snapshot_overrides(self) -> Dict[str, Any]:
        return dict(self._cfg.to_cfg_overrides())


    # -------------------------------------------------
    # Convenience (string keys)
    # -------------------------------------------------
    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        fkey, extra_key = self._parse_key_str(key)
        if fkey is not None:
            return self.get_value(
                fkey,
                default=default,
            )

        if extra_key in self._extra:
            return self._extra.get(extra_key, default)

        return default

    # def set(
    #     self,
    #     key: str,
    #     value: Any,
    # ) -> bool:
    #     fkey, extra_key = self._parse_key_str(key)
    #     if fkey is None:
    #         old = self._extra.get(extra_key, None)
    #         if old != value:
    #             self._extra[extra_key] = value
    #             self._mark_changed({extra_key})
    #         return True

    #     return self.set_value_by_key(fkey, value)
    
    def set(
        self,
        key: str,
        value: Any,
    ) -> bool:
        fkey, extra_key = self._parse_key_str(key)
        if fkey is None:
            old = self._extra.get(extra_key, _MISSING)
    
            changed = (
                old is _MISSING
                or not _values_equal(old, value)
            )
            if changed:
                self._extra[extra_key] = value
                self._mark_changed({extra_key})
    
            return True
    
        return self.set_value_by_key(fkey, value)

    def _parse_key_str(
        self,
        key: str,
    ) -> Tuple[Optional[FieldKey], str]:
        raw = (key or "").strip()
        m = _KEY_PAT.match(raw)
        if not m:
            return None, raw

        base = (m.group("base") or "").strip()
        sub = m.group("sub") or None
        attr = base.replace(".", "_")

        if attr in self._valid_keys:
            fkey = FieldKey(name=attr, subkey=sub)
            return fkey, raw

        return None, raw


    # -----------------------------------------------------------------
    # Read helpers (schema-aware)
    # -----------------------------------------------------------------
    def get_value(
        self,
        key: FieldKey,
        *,
        default: Any = None,
    ) -> Any:
        if key.name not in self._valid_keys:
            return default

        cur = getattr(self._cfg, key.name, None)
        if not key.is_dict_item():
            return cur

        if not isinstance(cur, dict):
            return default

        return cur.get(key.subkey, default)

    def set_value_by_key(
        self,
        key: FieldKey,
        value: Any,
        *,
        strict_subkey: bool = True,
    ) -> bool:
        name = key.name
        if name not in self._valid_keys:
            self._emit_error(
                f"Unknown config key: {name!r}"
            )
            return False

        if not key.is_dict_item():
            value = self._coerce_value(name, value)
            # old = getattr(self._cfg, name)
            # if old != value:
            #     setattr(self._cfg, name, value)
            #     self._mark_changed({name})
            old = getattr(self._cfg, name)
            if not _values_equal(old, value):
                setattr(self._cfg, name, value)
                self._mark_changed({name})

            return True

        cur = getattr(self._cfg, name)
        if cur is None:
            base: Dict[str, Any] = {}
        elif isinstance(cur, dict):
            base = dict(cur)
        else:
            self._emit_error(
                f"{name!r} is not a dict field."
            )
            return False

        sub = key.subkey
        if strict_subkey and (sub not in base):
            self._emit_error(
                f"Unknown subkey {sub!r} in {name!r}"
            )
            return False

        base[sub] = value
        setattr(self._cfg, name, base)
        self._mark_changed({name})
        
        return True

    # -----------------------------------------------------------------
    # Batch updates
    # -----------------------------------------------------------------
    @contextmanager
    def batch(self):
        self._batch_depth += 1
        try:
            yield self
        finally:
            self._batch_depth = max(0, self._batch_depth - 1)
            if self._batch_depth == 0:
                self._flush_pending()

    # -----------------------------------------------------------------
    # Mutations (public)
    # -----------------------------------------------------------------
    def set_value(
        self,
        key: FieldKey,
        value: Any,
        *,
        strict_subkey: bool = True,
    ) -> bool:
        return self.set_value_by_key(
            key,
            value,
            strict_subkey=strict_subkey,
        )

    def patch(self, updates: Dict[str, Any]) -> Set[str]:
        """
        Patch dataclass fields by name.

        Returns
        -------
        changed : set[str]
            Field names that changed.
        """
        changed: Set[str] = set()

        for key, raw in (updates or {}).items():
            if key not in self._valid_keys:
                self._emit_error(
                    f"Unknown config key: {key!r}"
                )
                continue

            try:
                value = self._coerce_value(key, raw)
                old = getattr(self._cfg, key)
                if old != value:
                    setattr(self._cfg, key, value)
                    changed.add(key)
            except Exception as exc:
                self._emit_error(
                    f"Failed to set {key!r}: {exc}"
                )

        if changed:
            self._mark_changed(changed)

        return changed

    def merge_dict_field(
        self,
        field: str,
        updates: Dict[str, Any],
        *,
        replace: bool = False,
        emit: bool = True,
    ) -> bool:
        """
        Safely update dict-like override fields.

        Use for: feature_overrides, arch_overrides,
        prob_overrides, tuner_search_space, etc.

        This replaces the dict object so change detection
        is reliable (avoids in-place mutation invisibility).
        """
        if field not in self._valid_keys:
            self._emit_error(
                f"Unknown config field: {field!r}"
            )
            return False

        cur = getattr(self._cfg, field)
        if cur is None or not isinstance(cur, dict):
            base: Dict[str, Any] = {}
        else:
            base = dict(cur)

        if replace:
            new = dict(updates or {})
        else:
            new = dict(base)
            new.update(updates or {})

        if new != cur:
            setattr(self._cfg, field, new)
            if emit:
                self._mark_changed({field})
            else:
                self._dirty_count = self._compute_dirty_count()
            return True

        return False

    def replace_config(
        self,
        cfg: GeoPriorConfig,
        *,
        emit: bool = True,
    ) -> None:
        self._cfg = cfg
        self._valid_keys = set(
            cfg.__dataclass_fields__.keys()
        )
        self._dirty_count = self._compute_dirty_count()

        if emit:
            self.config_replaced.emit(cfg)
            self.dirty_changed.emit(self._dirty_count)
            self.config_changed.emit(set(self._valid_keys))
            
    def snapshot(self) -> GeoPriorConfig:
        """
       Return a deep copy of the current config.

       Dialogs use this for Cancel rollback semantics.
       """
        try:
            return copy.deepcopy(self._cfg)
        except Exception:
            # Fallback: rebuild from exported dict
            return GeoPriorConfig(
                **self._cfg.as_dict(),
            )
        
    def is_overridden(self, key: str) -> bool:
        k = str(key or "").strip()
        if not k:
            return False
    
        ov = self.snapshot_overrides() or {}
    
        # Minimal mapping (extend when needed)
        special = {
            "pde_mode": "PDE_MODE_CONFIG",
        }
        nat = special.get(k, k.upper())
        return nat in ov


    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _coerce_value(self, key: str, value: Any) -> Any:
        # Path-ish fields: accept str and coerce.
        if key in {"dataset_path", "results_root"}:
            if value is None:
                return None
            if isinstance(value, Path):
                return value
            if isinstance(value, str) and value.strip():
                return Path(value).expanduser()
            return value

        return value

    def _compute_dirty_count(self) -> int:
        try:
            return len(self._cfg.to_cfg_overrides())
        except Exception:
            return 0

    def _mark_changed(self, keys: Iterable[str]) -> None:
        keys_set = set(keys or [])
        if not keys_set:
            return

        if self._batch_depth > 0:
            self._pending_keys |= keys_set
            return

        self._emit_changed(keys_set)

    def _flush_pending(self) -> None:
        if not self._pending_keys:
            return
        keys = set(self._pending_keys)
        self._pending_keys.clear()
        self._emit_changed(keys)

    def _emit_changed(self, keys: Set[str]) -> None:
        self.config_changed.emit(keys)

        new_dirty = self._compute_dirty_count()
        if new_dirty != self._dirty_count:
            self._dirty_count = new_dirty
            self.dirty_changed.emit(new_dirty)

    def _emit_error(self, msg: str) -> None:
        self.error_raised.emit(str(msg))

def _values_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False

    eq = getattr(a, "equals", None)
    if callable(eq):
        try:
            return bool(eq(b))
        except Exception:
            return False

    eq = getattr(b, "equals", None)
    if callable(eq):
        try:
            return bool(eq(a))
        except Exception:
            return False

    try:
        return bool(a == b)
    except Exception:
        return False
