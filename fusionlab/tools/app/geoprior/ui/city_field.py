# geoprior/ui/city_field.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.city_field

Reusable city input widget (LineEdit + lock button).

- Store-backed (single source of truth via CityManager)
- Optional completer from discovered cities
- Safe icon loading with Qt fallbacks
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import (
    Qt,
    QSignalBlocker,
    QStringListModel,
    pyqtSignal,
    QSize,
)
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QCompleter,
    QHBoxLayout,
    QLineEdit,
    QStyle,
    QToolButton,
    QWidget,
)

from ..config.store import GeoConfigStore
from ..services.city_manager import CityManager
from .icon_utils import try_icon

__all__ = [
    "CityField",
]


class CityField(QWidget):
    """
    Compact city widget (edit + lock).

    Signals
    -------
    city_committed(city_key)
        Emitted after a valid user commit.
    lock_changed(locked)
        Emitted after lock state changes.
    toast(status, message)
        Optional UX hook for toasts:
        status in {"ok","warn","bad"}.
    """

    city_committed = pyqtSignal(str)
    lock_changed = pyqtSignal(bool)
    toast = pyqtSignal(str, str)

    def __init__(
        self,
        store: GeoConfigStore,
        *,
        parent: Optional[QWidget] = None,
        manager: Optional[CityManager] = None,
        enable_completer: bool = True,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._mgr = manager or CityManager(store)

        self._edit = QLineEdit(self)
        self._lock_btn = QToolButton(self)

        self._model = QStringListModel(self)
        self._completer: Optional[QCompleter] = None

        self._locked_icon = QIcon()
        self._unlocked_icon = QIcon()

        self._build_ui(enable_completer)
        self._connect_signals()

        self.refresh_all()

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(
        self,
        enable_completer: bool,
    ) -> None:
        self._edit.setPlaceholderText("e.g. nansha")
        self._edit.setObjectName("cityDatasetEdit")
        self._edit.setClearButtonEnabled(True)

        self._lock_btn.setAutoRaise(True)
        self._lock_btn.setCheckable(True)
        self._lock_btn.setObjectName("cityLockBtn")
        self._lock_btn.setCursor(Qt.PointingHandCursor)

        self._load_icons()
        self._lock_btn.setIconSize(self._icon_size())

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(self._edit, 1)
        lay.addWidget(self._lock_btn, 0)

        if enable_completer:
            self._setup_completer()

    def _setup_completer(self) -> None:
        c = QCompleter(self._model, self)
        c.setCaseSensitivity(Qt.CaseInsensitive)
        c.setCompletionMode(QCompleter.PopupCompletion)

        self._edit.setCompleter(c)
        self._completer = c

    def _icon_size(self) -> QSize:
        px = self.style().pixelMetric(
            QStyle.PM_SmallIconSize
        )
        return QSize(px, px)


    def _load_icons(self) -> None:
        locked = try_icon("locked.svg")
        if locked is not None:
            self._locked_icon = locked
        else:
            self._locked_icon = self.style().standardIcon(
                QStyle.SP_DialogCloseButton
            )

        unlocked = try_icon("unlocked.svg")
        if unlocked is not None:
            self._unlocked_icon = unlocked
        else:
            self._unlocked_icon = self.style().standardIcon(
                QStyle.SP_DialogOpenButton
            )

    # def _icon_dir(self) -> Path:
    #     # geoprior/ui/city_field.py -> geoprior/icons/
    #     base = Path(__file__).resolve().parents[1]
    #     return base / "icons"

    # -------------------------------------------------
    # Signals / store sync
    # -------------------------------------------------
    def _connect_signals(self) -> None:
        self._edit.editingFinished.connect(self._commit_city)
        self._lock_btn.toggled.connect(self._on_lock_toggled)

        self._store.config_changed.connect(self._on_store)
        self._store.config_replaced.connect(self._on_replaced)

    def _on_store(self, keys: object) -> None:
        changed = set(keys or [])
        if ("city" in changed) or ("city.locked" in changed):
            self.refresh_all()

        if "results_root" in changed:
            self.refresh_completer()

    def _on_replaced(self, cfg: object) -> None:
        self.refresh_all()
        self.refresh_completer()

    # -------------------------------------------------
    # Public refresh
    # -------------------------------------------------
    def refresh_all(self) -> None:
        self._refresh_city()
        self._refresh_lock()

    def refresh_completer(self) -> None:
        if self._completer is None:
            return

        cities = self._mgr.discover_cities()
        self._model.setStringList(list(cities or []))

    # -------------------------------------------------
    # Internals
    # -------------------------------------------------
    def _refresh_city(self) -> None:
        city = self._mgr.get_city()
        with QSignalBlocker(self._edit):
            self._edit.setText(city)

    def _refresh_lock(self) -> None:
        locked = self._mgr.is_locked()

        with QSignalBlocker(self._lock_btn):
            self._lock_btn.setChecked(locked)

        if locked:
            self._lock_btn.setIcon(self._locked_icon)
            self._lock_btn.setToolTip("City locked")
            self._edit.setReadOnly(True)
        else:
            self._lock_btn.setIcon(self._unlocked_icon)
            self._lock_btn.setToolTip("City unlocked")
            self._edit.setReadOnly(False)

    def _commit_city(self) -> None:
        if self._mgr.is_locked():
            self.toast.emit("warn", "City is locked.")
            self._refresh_city()
            return

        raw = self._edit.text()
        ok, key, msg = self._mgr.validate(raw)

        if not ok:
            self._store.error_raised.emit(msg)
            self.toast.emit("bad", msg)
            self._refresh_city()
            return

        changed = (key != self._mgr.get_city())
        applied = self._mgr.set_city(key, quiet=True)

        if not applied:
            self._refresh_city()
            return

        if msg:
            self.toast.emit("warn", msg)

        if changed:
            self.city_committed.emit(key)

    def _on_lock_toggled(self, checked: bool) -> None:
        self._mgr.set_locked(bool(checked))
        self._refresh_lock()

        if checked:
            self.toast.emit("ok", "City locked.")
        else:
            self.toast.emit("ok", "City unlocked.")

        self.lock_changed.emit(bool(checked))

    def line_edit(self) -> QLineEdit:
        return self._edit

    def city_text(self) -> str:
        return str(self._edit.text()).strip()

    def set_city_text(self, text: str) -> None:
        with QSignalBlocker(self._edit):
            self._edit.setText(str(text or ""))

    def commit(self) -> bool:
        """
        Commit current text into store via CityManager.

        Returns True if store ends up with a non-empty city.
        """
        self._commit_city()
        return bool(self._mgr.get_city())
