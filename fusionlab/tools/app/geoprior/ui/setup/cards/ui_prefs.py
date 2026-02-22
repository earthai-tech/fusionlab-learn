# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.ui_prefs

UI preferences (theme, sizing, comfort knobs).

Notes
-----
- Theme is stored in QSettings ("ui/theme") so it persists.
- Extra UI-only toggles are stored in QSettings too.
- Window sizing fields are real GeoPriorConfig fields and
  are store-driven via the Binder.
"""

from __future__ import annotations

from typing import Optional, Set

from PyQt5.QtCore import QSettings, QSignalBlocker
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .base import CardBase
from ..bindings import Binder
from ....config.store import GeoConfigStore


class UiPreferencesCard(CardBase):
    """UI preferences card (modern, lightweight)."""

    _WATCH: Set[str] = {
        "ui_base_width",
        "ui_base_height",
        "ui_min_width",
        "ui_min_height",
        "ui_max_ratio",
        "ui_font_scale",
    }

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        binder: Binder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            section_id="ui",
            title="UI preferences",
            subtitle=(
                "Adjust theme, sizing, and comfort options. "
                "Most settings apply immediately."
            ),
            parent=parent,
        )
        self.store = store
        self.binder = binder
        self._qs = QSettings()

        self._build()
        self._wire()
        self.refresh()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build(self) -> None:
        btn_reset = self.add_action(
            text="Reset",
            tip="Reset UI prefs to defaults",
            icon=QStyle.SP_DialogResetButton,
        )
        btn_reset.clicked.connect(self._reset_ui_defaults)

        body = self.body_layout()

        grid = QWidget(self)
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

        g.addWidget(self._build_appearance(grid), 0, 0)
        g.addWidget(self._build_layout(grid), 0, 1)

        body.addWidget(grid, 0)

        note = QLabel(
            "Tip: theme changes persist via QSettings "
            "and are applied on next app start if "
            "live application is not available.",
            self,
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: rgba(30,30,30,0.70);")
        body.addWidget(note, 0)

    def _build_appearance(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Appearance", parent)
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)
        lay.setColumnStretch(1, 1)

        r = 0

        self.cmb_theme = QComboBox(box)
        self.cmb_theme.addItem("FusionLab (light)", "fusionlab")
        self.cmb_theme.addItem("FusionLab Dark", "dark")
        self.cmb_theme.currentIndexChanged.connect(
            self._commit_theme
        )

        lay.addWidget(QLabel("Theme:", box), r, 0)
        lay.addWidget(self.cmb_theme, r, 1)
        r += 1

        self.sp_font = QDoubleSpinBox(box)
        self.sp_font.setRange(0.80, 1.60)
        self.sp_font.setSingleStep(0.05)
        self.sp_font.setDecimals(2)
        self.sp_font.setToolTip(
            "UI font scale.\n"
            "1.00 = default, 1.10 = +10%."
        )
        self.binder.bind_double_spin_box(
            "ui_font_scale",
            self.sp_font,
        )

        lay.addWidget(QLabel("Font scale:", box), r, 0)
        lay.addWidget(self.sp_font, r, 1)
        r += 1

        self.chk_compact = QCheckBox(
            "Compact spacing (denser UI)",
            box,
        )
        self.chk_compact.toggled.connect(
            lambda on: self._set_qs_bool(
                "ui/compact_mode",
                bool(on),
            )
        )

        self.chk_tips = QCheckBox(
            "Show tips and helper text",
            box,
        )
        self.chk_tips.toggled.connect(
            lambda on: self._set_qs_bool(
                "ui/show_tips",
                bool(on),
            )
        )

        lay.addWidget(self.chk_compact, r, 0, 1, 2)
        r += 1
        lay.addWidget(self.chk_tips, r, 0, 1, 2)
        r += 1

        return box

    def _build_layout(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Window & behavior", parent)
        root = QVBoxLayout(box)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        size = QGroupBox("Sizing defaults", box)
        lay = QGridLayout(size)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(8)

        def _sp(minv: int, maxv: int) -> QSpinBox:
            sp = QSpinBox(size)
            sp.setRange(int(minv), int(maxv))
            return sp

        self.sp_base_w = _sp(640, 4000)
        self.sp_base_h = _sp(480, 3000)
        self.sp_min_w = _sp(640, 4000)
        self.sp_min_h = _sp(480, 3000)

        self.binder.bind_spin_box("ui_base_width", self.sp_base_w)
        self.binder.bind_spin_box(
            "ui_base_height",
            self.sp_base_h,
        )
        self.binder.bind_spin_box("ui_min_width", self.sp_min_w)
        self.binder.bind_spin_box(
            "ui_min_height",
            self.sp_min_h,
        )

        self.sp_ratio = QDoubleSpinBox(size)
        self.sp_ratio.setRange(0.40, 1.00)
        self.sp_ratio.setSingleStep(0.02)
        self.sp_ratio.setDecimals(2)
        self.sp_ratio.setToolTip(
            "Max fraction of screen used by the "
            "initial window size."
        )
        self.binder.bind_double_spin_box(
            "ui_max_ratio",
            self.sp_ratio,
        )

        lay.addWidget(QLabel("Base (W):", size), 0, 0)
        lay.addWidget(self.sp_base_w, 0, 1)
        lay.addWidget(QLabel("Base (H):", size), 0, 2)
        lay.addWidget(self.sp_base_h, 0, 3)

        lay.addWidget(QLabel("Min (W):", size), 1, 0)
        lay.addWidget(self.sp_min_w, 1, 1)
        lay.addWidget(QLabel("Min (H):", size), 1, 2)
        lay.addWidget(self.sp_min_h, 1, 3)

        lay.addWidget(QLabel("Max ratio:", size), 2, 0)
        lay.addWidget(self.sp_ratio, 2, 1)

        root.addWidget(size, 0)

        beh = QGroupBox("Safety & comfort", box)
        b = QVBoxLayout(beh)
        b.setContentsMargins(10, 10, 10, 10)
        b.setSpacing(8)

        self.chk_confirm = QCheckBox(
            "Confirm destructive actions",
            beh,
        )
        self.chk_confirm.toggled.connect(
            lambda on: self._set_qs_bool(
                "ui/confirm_actions",
                bool(on),
            )
        )

        self.chk_auto_adv = QCheckBox(
            "Auto-expand Advanced sections",
            beh,
        )
        self.chk_auto_adv.toggled.connect(
            lambda on: self._set_qs_bool(
                "ui/auto_expand_advanced",
                bool(on),
            )
        )

        b.addWidget(self.chk_confirm, 0)
        b.addWidget(self.chk_auto_adv, 0)

        root.addWidget(beh, 0)
        root.addStretch(1)

        return box

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.store.config_changed.connect(self._on_changed)
        self.store.config_replaced.connect(lambda *_a: self.refresh())

    def _on_changed(self, keys: object) -> None:
        if not isinstance(keys, (set, list, tuple)):
            self.refresh()
            return
        ks = {str(k) for k in keys}
        if not ks or (ks & self._WATCH):
            self.refresh()

    # -----------------------------------------------------------------
    # Theme + QSettings helpers
    # -----------------------------------------------------------------
    def _commit_theme(self, *_a: object) -> None:
        theme = str(self.cmb_theme.currentData() or "")
        theme = (theme or "fusionlab").strip().lower()

        self._qs.setValue("ui/theme", theme)
        self.store.set("ui.theme", theme)

        self._apply_theme_live(theme)
        self._refresh_badges()

    def _apply_theme_live(self, theme: str) -> None:
        w = self.window()
        if w is None:
            return

        # Preferred: main window exposes set_dark_mode(bool)
        try:
            if hasattr(w, "set_dark_mode"):
                fn = getattr(w, "set_dark_mode")
                fn(bool(theme == "dark"))
                return
        except Exception:
            pass

        # Fallback: private apply method if present
        try:
            if hasattr(w, "_apply_theme"):
                fn = getattr(w, "_apply_theme")
                fn()
        except Exception:
            pass

    def _set_qs_bool(self, key: str, val: bool) -> None:
        self._qs.setValue(str(key), bool(val))
        self.store.set(str(key).replace("/", "."), bool(val))
        self._refresh_badges()

    def _get_qs_bool(self, key: str, default: bool) -> bool:
        try:
            v = self._qs.value(str(key), default)
            if isinstance(v, str):
                s = v.strip().lower()
                return s in {"1", "true", "yes", "on"}
            return bool(v)
        except Exception:
            return bool(default)

    # -----------------------------------------------------------------
    # Reset + refresh
    # -----------------------------------------------------------------
    def _reset_ui_defaults(self) -> None:
        # Theme
        self._qs.setValue("ui/theme", "fusionlab")
        self.store.set("ui.theme", "fusionlab")

        # Comfort toggles
        self._qs.setValue("ui/compact_mode", False)
        self._qs.setValue("ui/show_tips", True)
        self._qs.setValue("ui/confirm_actions", True)
        self._qs.setValue("ui/auto_expand_advanced", False)

        self.store.set("ui.compact_mode", False)
        self.store.set("ui.show_tips", True)
        self.store.set("ui.confirm_actions", True)
        self.store.set("ui.auto_expand_advanced", False)

        # Config fields reset via config defaults is handled elsewhere,
        # but we keep this card consistent by re-pulling.
        self.refresh()
        self._apply_theme_live("fusionlab")

    def refresh(self) -> None:
        theme = self._qs.value("ui/theme", "fusionlab")
        theme = (str(theme or "fusionlab")).strip().lower()

        idx = self.cmb_theme.findData(theme)
        if idx < 0:
            idx = self.cmb_theme.findData("fusionlab")
        with QSignalBlocker(self.cmb_theme):
            self.cmb_theme.setCurrentIndex(max(0, idx))

        with QSignalBlocker(self.chk_compact):
            self.chk_compact.setChecked(
                self._get_qs_bool("ui/compact_mode", False)
            )
        with QSignalBlocker(self.chk_tips):
            self.chk_tips.setChecked(
                self._get_qs_bool("ui/show_tips", True)
            )
        with QSignalBlocker(self.chk_confirm):
            self.chk_confirm.setChecked(
                self._get_qs_bool("ui/confirm_actions", True)
            )
        with QSignalBlocker(self.chk_auto_adv):
            self.chk_auto_adv.setChecked(
                self._get_qs_bool(
                    "ui/auto_expand_advanced",
                    False,
                )
            )

        self._refresh_badges()

    def _refresh_badges(self) -> None:
        theme = str(self._qs.value("ui/theme", "fusionlab"))
        theme = (theme or "fusionlab").strip().lower()

        self.badge(
            "theme",
            text=("Theme: Dark" if theme == "dark" else "Theme: Light"),
            accent=("warn" if theme == "dark" else ""),
        )

        compact = self._get_qs_bool("ui/compact_mode", False)
        self.badge(
            "density",
            text=("Compact" if compact else "Comfort"),
        )

        fs = float(getattr(self.store.cfg, "ui_font_scale", 1.0))
        self.badge("font", text=f"Font: {fs:.2f}")

        confirm = self._get_qs_bool("ui/confirm_actions", True)
        self.badge(
            "safety",
            text=("Confirm: ON" if confirm else "Confirm: OFF"),
            accent=("ok" if confirm else "warn"),
        )
