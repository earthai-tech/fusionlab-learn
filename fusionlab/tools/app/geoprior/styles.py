# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
GeoPrior GUI styles.

Central colour palette and reusable Qt
stylesheets (light / dark theme, tabs,
log widget, error dialog, tuner dialog).
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
#  Base palette
# ------------------------------------------------------------------ #
PRIMARY = "#2E3191"
SECONDARY = "#F28620"
BG_LIGHT = "#fafafa"
FG_DARK = "#1e1e1e"

PRIMARY_T75 = "rgba(46,49,145,0.75)"
SECONDARY_T70 = "rgba(242,134,32,0.70)"

# Inference-mode toggle
INFERENCE_ON = PRIMARY
INFERENCE_OFF = "#dadada"

# Central palette (shared by light / dark)
PALETTE = {
    # Primary brand colours
    "primary": "#2E3191",
    "primary_hover": "#4338ca",
    "secondary": "#F28620",

    # Dark theme
    "dark_bg": "#1e293b",
    "dark_card_bg": "#334155",
    "dark_input_bg": "#0f172a",
    "dark_border": "#475569",
    "dark_text": "#cbd5e1",
    "dark_text_title": "#ffffff",
    "dark_text_muted": "#94a3b8",
    "dark_reset_bg": "#475569",

    # Light theme
    "light_bg": "#f8fafc",
    "light_card_bg": "#ffffff",
    "light_input_bg": "#f1f5f9",
    "light_border": "#cbd5e1",
    "light_text": "#0f172a",
    "light_text_title": "#2E3191",
    "light_text_muted": "#64748b",
    "light_reset_bg": "#e2e8f0",
}

# ------------------------------------------------------------------ #
#  Light theme – main GeoPrior style
# ------------------------------------------------------------------ #
FLAB_STYLE_SHEET = f"""
QMainWindow {{
    background: {BG_LIGHT};
    color: {FG_DARK};
    font-family: 'Helvetica Neue', sans-serif;
}}

QFrame#card {{
    background: white;
    border: 2px solid {PRIMARY};
    border-radius: 12px;
}}

QLabel#cardTitle {{
    font-size: 18px;
    font-weight: 600;
    color: {PRIMARY};
}}

QPushButton {{
    background: {PRIMARY};
    color: white;
    border-radius: 6px;
    padding: 6px 12px;
}}

QPushButton:hover {{
    background: {SECONDARY};
}}

QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox {{
    background: #f0f3ff;
    border: 1px solid {PRIMARY};
    border-radius: 4px;
    padding: 4px;
}}

QTextEdit {{
    background: #f6f6f6;
    border: 1px solid #cccccc;
}}

QPlainTextEdit {{
    background: #f6f6f6;
    border: 1px solid #cccccc;
}}

QPushButton#reset,
QPushButton#stop {{
    background: #dadada;
    color: #333;
}}

QPushButton#reset:hover:enabled,
QPushButton#stop:hover:enabled {{
    background: {SECONDARY};
    color: white;
}}

QPushButton#reset:disabled,
QPushButton#stop:disabled {{
    background: #dadada;
    color: #333;
}}

QPushButton#stop:enabled {{
    background: {PRIMARY};
    color: white;
}}

QToolTip {{
    background: {SECONDARY_T70};
    color: white;
    border: 1px solid {SECONDARY};
    border-radius: 4px;
    padding: 4px 6px;
}}

QPushButton#inference {{
    background: {PRIMARY};
    color: white;
    border-radius: 6px;
    padding: 6px 14px;
}}

QPushButton#inference:disabled {{
    background: {INFERENCE_OFF};
    color: #666;
}}

QPushButton#tune {{
    background: {PRIMARY};
    color: white;
    border-radius: 6px;
    padding: 6px 14px;
}}

QPushButton#tune:disabled {{
    background: {INFERENCE_OFF};
    color: #666;
}}

QFrame#card[inferenceMode="true"] {{
    border: 2px solid #2E3191;
}}

/* QMessageBox – light */
QMessageBox {{
    background-color: {PALETTE['light_bg']};
}}

QMessageBox QLabel {{
    color: {PALETTE['light_text']};
    font-size: 14px;
}}

QMessageBox QPushButton {{
    background-color: {PALETTE['primary']};
    color: white;
    border-radius: 4px;
    padding: 8px 20px;
    min-width: 80px;
}}

QMessageBox QPushButton:hover {{
    background-color: {PALETTE['primary_hover']};
}}

QMessageBox QPushButton:pressed {{
    background-color: {PALETTE['secondary']};
    color: white;
}}
"""

# ------------------------------------------------------------------ #
#  Dark theme
# ------------------------------------------------------------------ #
DARK_THEME_STYLESHEET = f"""
QMainWindow,
QWidget {{
    background-color: {PALETTE['dark_bg']};
    color: {PALETTE['dark_text']};
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
}}

QLabel#title {{
    font-size: 28px;
    font-weight: bold;
    color: {PALETTE['dark_text_title']};
    padding: 10px;
}}

QLabel#description {{
    font-size: 14px;
    color: {PALETTE['dark_text_muted']};
}}

QFrame#card {{
    background-color: {PALETTE['dark_card_bg']};
    border: 1px solid {PALETTE['dark_border']};
    border-radius: 12px;
}}

QLabel#cardTitle {{
    font-size: 18px;
    font-weight: 600;
    color: {PALETTE['dark_text_title']};
    padding-bottom: 5px;
}}

QLabel#cardDescription {{
    font-size: 13px;
    color: {PALETTE['dark_text_muted']};
}}

QPushButton {{
    background-color: {PALETTE['primary']};
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 6px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: #4338ca;
}}

QPushButton:disabled {{
    background-color: #334155;
    color: {PALETTE['dark_text_muted']};
}}

QPushButton#resetButton,
QPushButton#stopButton {{
    background-color: {PALETTE['dark_reset_bg']};
    color: {PALETTE['dark_text']};
}}

QPushButton#resetButton:hover,
QPushButton#stopButton:hover {{
    background-color: {PALETTE['dark_border']};
}}

QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox {{
    background-color: {PALETTE['dark_input_bg']};
    border: 1px solid {PALETTE['dark_border']};
    padding: 8px;
    border-radius: 6px;
    color: white;
}}

QLineEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus {{
    border: 1px solid {PALETTE['primary']};
}}

QTextEdit,
QPlainTextEdit {{
    background-color: #020617;
    color: #e2e8f0;
    border: 1px solid {PALETTE['dark_border']};
    font-family: "Consolas", "Courier New", monospace;
    font-size: 13px;
}}

QFrame#hLine {{
    border: none;
    border-top: 1px solid {PALETTE['dark_border']};
}}

QFrame#card[inferenceMode="true"] {{
    border: 2px solid #F28620;
}}

/* QMessageBox – dark */
QMessageBox {{
    background-color: {PALETTE['dark_card_bg']};
}}

QMessageBox QLabel {{
    color: {PALETTE['dark_text']};
    font-size: 14px;
}}

QMessageBox QPushButton {{
    background-color: {PALETTE['primary']};
    color: white;
    border-radius: 4px;
    padding: 8px 20px;
    min-width: 80px;
}}

QMessageBox QPushButton:hover {{
    background-color: {PALETTE['primary_hover']};
}}

QMessageBox QPushButton:pressed {{
    background-color: {PALETTE['secondary']};
    color: white;
}}
"""

# ------------------------------------------------------------------ #
#  Tabs – shared for both themes
# ------------------------------------------------------------------ #
TAB_STYLES = f"""
QTabBar::tab {{
    background : #F9F7F5;
    color      : black;
    padding    : 6px 14px;
    border-top-left-radius  : 4px;
    border-top-right-radius : 4px;
}}

QTabBar::tab:selected {{
    background : {SECONDARY};
    color      : white;
}}

QTabBar::tab:hover {{
    background : {SECONDARY};
    color      : white;
}}
"""

# ------------------------------------------------------------------ #
#  Log styles
# ------------------------------------------------------------------ #
LOG_STYLES = """
QTextEdit#logWidget {
    background-color: #1e1e1e;
    color: #e2e8f0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
    border: 1px solid rgba(0,0,0,0.25);
    padding: 6px;
}

QPlainTextEdit#logWidget {
    background-color: #1e1e1e;
    color: #e2e8f0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
    border: 1px solid rgba(0,0,0,0.25);
    padding: 6px;
}
"""

_LOG_DOCK_STYLES = f"""
QDockWidget#logDock {{
    border: 1px solid rgba(0,0,0,0.25);
    border-radius: 4px;
    background: #ffffff;
}}

QDockWidget#logDock::separator {{
    width: 0px;
    height: 0px;
}}

QDockWidget#logDock::title {{
    background: {PRIMARY};
    color: white;
    padding: 4px 10px;
    font-weight: 600;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QDockWidget#logDock::title:hover {{
    background: {SECONDARY};
}}

QDockWidget#logDock QTextEdit {{
    border: none;
    background: #fafafa;
    font-family: Consolas, monospace;
    font-size: 11px;
    padding: 6px;
}}
"""

# ------------------------------------------------------------------ #
#  Tuner + error dialog styles
# ------------------------------------------------------------------ #
TUNER_STYLES = f"""
QPushButton {{
    background: {PRIMARY};
    color: white;
    padding: 6px 18px;
    border: none;
    border-radius: 4px;
}}

QPushButton:checked {{
    background: {SECONDARY};
}}
"""

ERROR_STYLES = f"""
QDialog#errorDialog {{
    background: {BG_LIGHT};
    border: 2px solid {PRIMARY};
    border-radius: 8px;
    padding: 12px;
    min-width: 600px;
    max-width: 800px;
}}

QDialog#errorDialog QLabel {{
    color: {FG_DARK};
    font-size: 14px;
}}

QDialog#errorDialog QTextEdit {{
    background: #f6f6f6;
    border: 1px solid #cccccc;
    font-family: Consolas, monospace;
}}

QDialog#errorDialog QPushButton {{
    background: {PRIMARY};
    color: white;
    border-radius: 4px;
    padding: 6px 16px;
    min-width: 80px;
}}

QDialog#errorDialog QPushButton:hover:enabled {{
    background: {SECONDARY};
}}

QDialog#errorDialog QPushButton:disabled {{
    background: {INFERENCE_OFF};
    color: #666;
}}
"""

TUNER_DIALOG_STYLES = f"""
QMessageBox {{
    background-color: {BG_LIGHT};
    border-radius: 8px;
    padding: 12px;
}}

QMessageBox QLabel {{
    color: {FG_DARK};
    font-size: 13px;
    qproperty-alignment: AlignLeft;
}}

QMessageBox QLabel#qt_msgbox_label {{
    font-weight: 600;
    font-size: 14px;
}}

QMessageBox QPushButton {{
    background-color: {PRIMARY};
    color: white;
    border-radius: 4px;
    padding: 4px 12px;
    min-width: 70px;
}}

QMessageBox QPushButton:hover:enabled {{
    background-color: {SECONDARY};
}}

QMessageBox QPushButton:disabled {{
    background-color: {INFERENCE_OFF};
    color: #888;
}}
"""

__all__ = [
    "PRIMARY",
    "SECONDARY",
    "BG_LIGHT",
    "FG_DARK",
    "INFERENCE_ON",
    "INFERENCE_OFF",
    "FLAB_STYLE_SHEET",
    "DARK_THEME_STYLESHEET",
    "TAB_STYLES",
    "LOG_STYLES",
    "ERROR_STYLES",
    "TUNER_DIALOG_STYLES",
]
