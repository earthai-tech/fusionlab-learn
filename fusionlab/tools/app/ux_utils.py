# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Small Qt helpers for consistent fonts / window sizing / crash handling.

Usage
-----
In your main entry point:

    app = QApplication(sys.argv)
    auto_set_ui_fonts(app)            # global fonts + Fusion style
    enable_qt_crash_handler(app)      # nice tracebacks on crash

    gui = GeoPriorForecaster()
    gui.show()
    sys.exit(app.exec_())

Inside the main window:

    class GeoPriorForecaster(QMainWindow):
        ...
        def _set_window_props(self) -> None:
            ...
            auto_resize_window(self, base_size=(820, 520))
"""

from __future__ import annotations

import sys
import traceback
import faulthandler
from typing import Tuple, Optional

from PyQt5.QtGui import QFont, QGuiApplication
from PyQt5.QtWidgets import QApplication, QToolTip, QWidget


# ----------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------
def auto_set_ui_fonts(app: QApplication, tooltip: bool = True) -> None:
    """
    Detect platform and apply a reasonable default UI font.

    This keeps the GUI readable across Windows / macOS / Linux with
    different DPI defaults.
    """
    plat = sys.platform.lower()

    if plat.startswith("win"):
        ui_font = QFont("Segoe UI", 9)
        tt_font = QFont("Segoe UI", 9)
    elif plat == "darwin":
        # macOS system defaults
        ui_font = QFont(".SF NS Text", 12)
        tt_font = QFont(".SF NS Text", 11)
    else:
        # Generic Linux defaults (GTK/Qt desktops)
        ui_font = QFont("Ubuntu", 10)
        tt_font = QFont("Ubuntu Mono", 9)

    app.setFont(ui_font)
    if tooltip:
        QToolTip.setFont(tt_font)


# ----------------------------------------------------------------------
# Window geometry
# ----------------------------------------------------------------------
def auto_resize_window(
    window: QWidget,
    base_size: Optional[Tuple[int, int]] = None,
    margin: Optional[Tuple[int, int]] = None,
    min_size: Optional[Tuple[int, int]] = None,
    max_ratio: float = 0.9,
) -> None:
    """
    Resize ``window`` so it fits nicely on the current screen.

    Parameters
    ----------
    window :
        The top–level widget (usually your main window).
    base_size :
        Your “design” size (width, height). Defaults to (980, 660).
    margin :
        Pixels to subtract from available screen size
        (left/right, top/bottom). Defaults to (100, 100).
    min_size :
        Smallest allowed size. Defaults to (800, 600).
    max_ratio :
        Maximum fraction of the available screen area to occupy.
    """
    base_size = base_size or (980, 660)
    margin = margin or (100, 100)
    min_size = min_size or (800, 600)

    # Prefer the screen the window is on; fall back to primary.
    screen = window.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        return

    avail = screen.availableGeometry()
    max_w = int(avail.width() * max_ratio) - margin[0]
    max_h = int(avail.height() * max_ratio) - margin[1]

    target_w = min(base_size[0], max_w)
    target_h = min(base_size[1], max_h)

    target_w = max(target_w, min_size[0])
    target_h = max(target_h, min_size[1])

    window.resize(target_w, target_h)
    window.setMinimumSize(min_size[0], min_size[1])


# ----------------------------------------------------------------------
# Crash / exception handling
# ----------------------------------------------------------------------
def enable_qt_crash_handler(
    app: QApplication,
    keep_gui_alive: bool = False,
) -> None:
    """
    Install a friendlier exception handler for Qt apps.

    On non-Windows, we also enable :mod:`faulthandler` to get C-level
    tracebacks. On Windows we **skip faulthandler** because it can
    surface spurious COM exceptions like ``0x8001010d`` and even
    crash some GUIs.

    Parameters
    ----------
    app :
        The QApplication instance.
    keep_gui_alive : bool, default False
        If True, leave the Qt event loop running after an unhandled
        Python exception. If False, call ``app.quit()`` after printing
        the traceback.
    """
    # 1) Optional C-level fault handler (disabled on Windows)
    if not sys.platform.lower().startswith("win"):
        try:
            # Log native crashes (segfaults, etc.) to stderr.
            faulthandler.enable()
        except Exception:
            # If anything goes wrong, don't break the GUI start-up.
            pass

    # 2) Python-level unhandled exceptions -> print traceback nicely
    def qt_excepthook(exctype, value, tb):
        traceback.print_exception(exctype, value, tb)
        if not keep_gui_alive:
            app.quit()

    sys.excepthook = qt_excepthook
