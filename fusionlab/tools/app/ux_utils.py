# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Qt UX helpers for GeoPrior.

Goals
-----
- Consistent fonts across platforms.
- Screen-aware window sizing (bigger default window).
- Optional persistent window geometry (QSettings).
- Friendlier crash handling for Qt apps.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import faulthandler
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QFont, QFontDatabase, QGuiApplication
from PyQt5.QtWidgets import QApplication, QToolTip, QWidget, QMessageBox


# ---------------------------------------------------------------------
# App metadata (QSettings)
# ---------------------------------------------------------------------
def set_app_metadata(
    app: QApplication,
    *,
    org_name: str = "FusionLab",
    app_name: str = "GeoPrior",
    org_domain: str = "fusionlab",
) -> None:
    """Set metadata used by QSettings and some OS shells."""
    app.setOrganizationName(org_name)
    app.setApplicationName(app_name)
    app.setOrganizationDomain(org_domain)


# ---------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------
def _first_available_font(candidates: Tuple[str, ...]) -> str:
    try:
        fams = set(QFontDatabase().families())
    except Exception:
        return ""
    for fam in candidates:
        if fam in fams:
            return fam
    return ""


def auto_set_ui_fonts(
    app: QApplication,
    *,
    font_scale: float = 1.0,
    tooltip: bool = True,
    base_point_size: Optional[float] = None,
    use_fusion_style: bool = True,
) -> None:
    """
    Detect platform and apply a reasonable default UI font.
    """
    plat = sys.platform.lower()

    if use_fusion_style:
        try:
            app.setStyle("Fusion")
        except Exception:
            pass

    if plat.startswith("win"):
        family = _first_available_font(
            ("Segoe UI", "Arial", "Tahoma")
        )
        ui_pt = 9.0
        tt_pt = 9.0
    elif plat == "darwin":
        family = _first_available_font(
            (".SF NS Text", "Helvetica Neue", "Arial")
        )
        ui_pt = 12.0
        tt_pt = 11.0
    else:
        family = _first_available_font(
            ("Ubuntu", "Noto Sans", "DejaVu Sans", "Arial")
        )
        ui_pt = 10.0
        tt_pt = 9.0

    if base_point_size is not None:
        ui_pt = float(base_point_size)

    ui_pt = max(6.0, ui_pt * float(font_scale))
    tt_pt = max(6.0, tt_pt * float(font_scale))

    ui_font = QFont(family, int(ui_pt))
    ui_font.setPointSizeF(ui_pt)
    app.setFont(ui_font)

    if tooltip:
        tt_font = QFont(family, int(tt_pt))
        tt_font.setPointSizeF(tt_pt)
        QToolTip.setFont(tt_font)

    # Cosmetic: remove the "?" context help button on dialogs.
    try:
        QApplication.setAttribute(
            Qt.AA_DisableWindowContextHelpButton,
            True,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------
# Window sizing
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class WindowPolicy:
    # Bigger / more modern default window
    base_size: Tuple[int, int] = (1180, 820)
    min_size: Tuple[int, int] = (1060, 740)

    margin: Tuple[int, int] = (90, 90)
    max_ratio: float = 0.92
    center_on_screen: bool = True
    remember_geometry: bool = True


def _available_geometry(window: QWidget):
    scr = window.screen() or QGuiApplication.primaryScreen()
    if scr is None:
        return None
    return scr.availableGeometry()


def restore_window_geometry(
    window: QWidget,
    *,
    settings_key: str,
) -> bool:
    """Restore window geometry from QSettings."""
    s = QSettings()
    geo = s.value(f"{settings_key}/geometry")
    state = s.value(f"{settings_key}/windowState")
    ok = False

    try:
        if geo is not None:
            ok = bool(window.restoreGeometry(geo))
        if state is not None:
            window.restoreState(state)
    except Exception:
        return False

    return ok


def save_window_geometry(
    window: QWidget,
    *,
    settings_key: str,
) -> None:
    """Save window geometry to QSettings."""
    s = QSettings()
    s.setValue(f"{settings_key}/geometry", window.saveGeometry())
    try:
        s.setValue(
            f"{settings_key}/windowState",
            window.saveState(),
        )
    except Exception:
        pass


def auto_resize_window(
    window: QWidget,
    *,
    settings_key: str = "main_window",
    policy: Optional[WindowPolicy] = None,
    base_size: Optional[Tuple[int, int]] = None,
    min_size: Optional[Tuple[int, int]] = None,
    max_ratio: Optional[float] = None,
    margin: Optional[Tuple[int, int]] = None,
    center_on_screen: Optional[bool] = None,
) -> None:
    """
    Resize window nicely on the current screen.

    Prefers persisted geometry if enabled, otherwise uses policy sizing.

    Notes
    -----
    Any of base_size, min_size, max_ratio, margin, center_on_screen can
    override the provided policy (or the default WindowPolicy).
    """
    pol0 = policy or WindowPolicy()

    pol = WindowPolicy(
        base_size=base_size or pol0.base_size,
        min_size=min_size or pol0.min_size,
        margin=margin or pol0.margin,
        max_ratio=float(max_ratio)
        if max_ratio is not None
        else pol0.max_ratio,
        center_on_screen=bool(center_on_screen)
        if center_on_screen is not None
        else pol0.center_on_screen,
        remember_geometry=pol0.remember_geometry,
    )

    if pol.remember_geometry:
        if restore_window_geometry(window, settings_key=settings_key):
            window.setMinimumSize(*pol.min_size)
            return

    avail = _available_geometry(window)
    if avail is None:
        return

    max_w = int(avail.width() * pol.max_ratio) - pol.margin[0]
    max_h = int(avail.height() * pol.max_ratio) - pol.margin[1]

    target_w = min(pol.base_size[0], max_w)
    target_h = min(pol.base_size[1], max_h)

    target_w = max(target_w, pol.min_size[0])
    target_h = max(target_h, pol.min_size[1])

    window.resize(target_w, target_h)
    window.setMinimumSize(*pol.min_size)

    if pol.center_on_screen:
        x = avail.x() + (avail.width() - target_w) // 2
        y = avail.y() + (avail.height() - target_h) // 2
        window.move(max(avail.x(), x), max(avail.y(), y))

# ---------------------------------------------------------------------
# Crash / exception handling
# ---------------------------------------------------------------------
def enable_qt_crash_handler(
    app: QApplication,
    *,
    keep_gui_alive: bool = False,
    show_dialog: bool = False,
    log_path: Optional[str] = None,
) -> None:
    """
    Install a friendlier exception handler for Qt apps.

    Notes
    -----
    - On Windows, faulthandler may surface COM issues, so it is disabled.
    - Also hooks threading.excepthook when available.
    """
    is_win = sys.platform.lower().startswith("win")

    if not is_win:
        try:
            if log_path:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    faulthandler.enable(file=f)
            else:
                faulthandler.enable()
        except Exception:
            pass

    def _emit(text: str) -> None:
        sys.stderr.write(text)
        sys.stderr.flush()
        if log_path:
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass

    def _handler(exctype, value, tb) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n[{ts}] Unhandled exception\n"
        body = "".join(traceback.format_exception(exctype, value, tb))
        _emit(header + body)

        if show_dialog:
            try:
                QMessageBox.critical(
                    None,
                    "GeoPrior crashed",
                    "An unexpected error occurred.\n\n"
                    "A traceback was printed to stderr."
                    + ("\nA log file was written." if log_path else ""),
                )
            except Exception:
                pass

        if not keep_gui_alive:
            try:
                app.quit()
            except Exception:
                pass

    sys.excepthook = _handler

    if hasattr(threading, "excepthook"):

        def _thread_hook(args):
            _handler(args.exc_type, args.exc_value, args.exc_traceback)

        threading.excepthook = _thread_hook
