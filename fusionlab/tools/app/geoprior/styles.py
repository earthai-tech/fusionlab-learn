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
PRIMARY ="#2E3191"# "#1C478B"# "#1F7DAD"# 
SECONDARY = "#F28620"# "#E66414" # 
BG_LIGHT = "#fafafa"
FG_DARK = "#1e1e1e"

PRIMARY_T75 = "rgba(46,49,145,0.75)"
SECONDARY_T70 = "rgba(242,134,32,0.70)"
SECONDARY_TBLUE = "#3399ff"

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
#  Mode indicator colours
# ------------------------------------------------------------------ #
MODE_DRY_COLOR    = "teal"       # DRY RUN
MODE_TRAIN_COLOR  = PRIMARY      # Train = brand primary
MODE_TUNE_COLOR   = "#8D4004"    # Tuning = neutral brown
MODE_INFER_COLOR  = "#00aa00"    # Inference = green
MODE_XFER_COLOR   = "#CF3476"    # Transferability = magenta
MODE_RESULTS_COLOR = "#4B5563"   # Results = slate grey
MODE_DATA_COLOR = "#0284C7"   # sky blue for Data tab
MODE_SETUP_COLOR = "#7c3aed"
MODE_PREPROCESS_COLOR = "#0ea5e9"

# ------------------------------------------------------------------ #
#  Run button colours
# ------------------------------------------------------------------ #
# RUN_BUTTON_IDLE     = "#16A34A"  # emerald green – clear "Go" CTA
# RUN_BUTTON_HOVER    = "#22C55E"  # slightly brighter on hover
# RUN_BUTTON_DISABLED = "#D1D5DB"  # light grey while running (disabled)

RUN_BUTTON_IDLE     = "#16A34A"  # emerald green
RUN_BUTTON_HOVER    = "#22C55E"  # brighter on hover
RUN_BUTTON_DISABLED = "#9CA3AF"  # greyed while running
# ------------------------------------------------------------------ #
#  Light theme – main GeoPrior style
# ------------------------------------------------------------------ #

FLAB_STYLE_SHEET = f"""
QMainWindow {{
    background: {BG_LIGHT};
    color: {FG_DARK};
    font-family: 'Helvetica Neue', sans-serif;
}}

QWidget#card {{
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
/* ---------- ComboBox popup contrast (Light) ---------- */
QComboBox {{
    color: {PALETTE['light_text']};
    padding-right: 26px;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid {PALETTE['light_border']};
    background: rgba(46, 49, 145, 0.10);
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}}

QComboBox QAbstractItemView,
QComboBox QListView {{
    background: {PALETTE['light_card_bg']};
    color: {PALETTE['light_text']};
    border: 1px solid {PALETTE['light_border']};
    outline: 0;
    padding: 4px;
}}

QComboBox QAbstractItemView::item,
QComboBox QListView::item {{
    padding: 6px 10px;
}}

QComboBox QAbstractItemView::item:hover,
QComboBox QListView::item:hover {{
    background: rgba(46, 49, 145, 0.12);
}}

QComboBox QAbstractItemView::item:selected,
QComboBox QListView::item:selected {{
    background: {PRIMARY};
    color: white;
}}
/* --- ComboBox arrow visibility (Light) --- */
/* --- Combo arrow (PNG) --- */
QComboBox::drop-down {{
    width: 26px;
    border-left: 1px solid {PALETTE['light_border']};
    background: rgba(46, 49, 145, 0.20);   /* stronger tint */
}}

QComboBox::down-arrow:on {{
    image: url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'>\
<path d='M2 3.5 L5 6.5 L8 3.5' fill='none' stroke='{SECONDARY}' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/>\
</svg>");
}}
     /* --- Force visible combo arrow (Qt5-safe) --- */
QComboBox::down-arrow {{
    image: none;                 /* disable default glyph */
    width: 0px;
    height: 0px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid {PRIMARY};  /* triangle color */
    margin-right: 6px;
}}

QComboBox::down-arrow:on {{
    border-top: 7px solid {SECONDARY};
}}      
QComboBox::drop-down {{
    width: 26px;
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
    background-color: rgba(15, 23, 42, 0.96);  /* dark-slate */
    color: #e5e7eb;
    border: 1px solid {PRIMARY};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
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

QWidget#card[inferenceMode="true"] {{
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

/* --- Dedicated style for RUN buttons (Train/Tune/Infer/Xfer) --- */
QPushButton#runButton {{
    background: {RUN_BUTTON_IDLE};
    color: white;
    border-radius: 8px;
    padding: 6px 16px;
    font-weight: 600;
}}

QPushButton#runButton:hover:enabled {{
    background: {RUN_BUTTON_HOVER};
}}

QPushButton#runButton:disabled {{
    background: {RUN_BUTTON_DISABLED};
    color: #374151;  /* darker grey text while running */
}}

QPushButton#runIconButton  {{
    /* icon-only, round hit area */
    background-color: transparent;
    border: none;
    padding: 2px;
    margin: 0;
    min-width: 30px;
    min-height: 30px;
    max-width: 32px;
    max-height: 32px;
}}

QPushButton#runIconButton:hover:enabled {{
    /* soft green halo on hover */
    background-color: rgba(34, 197, 94, 0.16);  /* based on RUN_BUTTON_HOVER */
    border-radius: 16px;
}}

QPushButton#runIconButton:disabled{{
    background-color: transparent;
    /* Qt will already dim the icon; this just removes hover halo */
}}

QToolButton#miniAction,
QPushButton#miniAction {{
    background: transparent;
    border: 1px solid rgba(46,49,145,0.30);   /* PRIMARY with alpha */
    border-radius: 8px;
    padding: 2px 6px;
}}

QToolButton#miniAction:hover:enabled,
QPushButton#miniAction:hover:enabled {{
    background: rgba(51,153,255,0.16);        /* SECONDARY_TBLUE tint */
    border-color: {SECONDARY_TBLUE};
}}

QToolButton#miniAction:pressed,
QPushButton#miniAction:pressed {{
    background: rgba(242,134,32,0.18);        /* SECONDARY tint */
    border-color: {SECONDARY};
}}
  
QToolButton#miniAction:focus {{
    outline: none;
    border: 1px solid rgba(46,49,145,0.70);
    background: rgba(46,49,145,0.10);
}}

QToolButton#miniAction:disabled {{
    border-color: rgba(100,116,139,0.35);
    background: transparent;
}}

QLineEdit#resultsRootEdit {{
    background: rgba(46, 49, 145, 0.08);
    border: 1px solid rgba(46, 49, 145, 0.55);
    border-radius: 6px;
    padding: 4px 8px;
    font-weight: 600;
}}

QLineEdit#resultsRootEdit:hover {{
    border-color: #3399ff;
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

QWidget#card {{
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

/* ---------- ComboBox popup contrast (Dark) ---------- */
QComboBox {{
    color: {PALETTE['dark_text']};
    padding-right: 26px;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid {PALETTE['dark_border']};
    background: rgba(242, 134, 32, 0.12);
}}

QComboBox QAbstractItemView,
QComboBox QListView {{
    background: {PALETTE['dark_card_bg']};
    color: {PALETTE['dark_text']};
    border: 1px solid {PALETTE['dark_border']};
    outline: 0;
    padding: 4px;
}}

QComboBox QAbstractItemView::item,
QComboBox QListView::item {{
    padding: 6px 10px;
}}

QComboBox QAbstractItemView::item:hover,
QComboBox QListView::item:hover {{
    background: rgba(46, 49, 145, 0.22);
}}

QComboBox QAbstractItemView::item:selected,
QComboBox QListView::item:selected {{
    background: {PALETTE['primary']};
    color: white;
}}
QComboBox::down-arrow {{
    image: none;
    width: 0px;
    height: 0px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid white;
    margin-right: 6px;
}}

QComboBox::down-arrow:on {{
    border-top: 7px solid {SECONDARY};
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
QToolButton#miniAction,
QPushButton#miniAction {{
    background: transparent;
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 8px;
    padding: 2px 6px;
}}

QToolButton#miniAction:hover:enabled,
QPushButton#miniAction:hover:enabled {{
    background: rgba(51,153,255,0.18);        /* SECONDARY_TBLUE tint */
    border-color: {SECONDARY_TBLUE};
}}

QToolButton#miniAction:pressed,
QPushButton#miniAction:pressed {{
    background: rgba(242,134,32,0.22);        /* SECONDARY tint */
    border-color: {SECONDARY};
}}

QToolButton#miniAction:focus {{
    outline: none;
    border: 1px solid rgba(46,49,145,0.70);
    background: rgba(46,49,145,0.10);
}}

QToolButton#miniAction:disabled {{
    border-color: rgba(100,116,139,0.35);
    background: transparent;
}}
QLineEdit#resultsRootEdit {{
    background: rgba(46, 49, 145, 0.22);
    border: 1px solid rgba(203, 213, 225, 0.26);
    border-radius: 6px;
    padding: 8px;
    font-weight: 600;
}}

QLineEdit#resultsRootEdit:hover {{
    border: 1px solid #3399ff;
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

MAIN_TAB_STYLES_LIGHT = f"""
/* ===== Main tabs (Light) : modern IDE underline ===== */
QTabWidget#mainTabs::pane {{
    border: none;
    background: transparent;
}}

QTabWidget#mainTabs QTabBar {{
    qproperty-drawBase: 0;
    background: rgba(46, 49, 145, 0.06);
    border: 1px solid rgba(46, 49, 145, 0.22);
    border-radius: 10px;         /* keep your bar look */
    padding: 2px 4px;
}}

QTabWidget#mainTabs QTabBar::tab {{
    background: transparent;
    color: {PALETTE['light_text']};
    border: none;
    border-radius: 0px;

    /* reduced height */
    padding: 3px 10px;
    margin: 0px 4px;
    min-height: 18px;
}}

QTabWidget#mainTabs QTabBar::tab:hover {{
    background: rgba(51, 153, 255, 0.12);   /* SECONDARY_TBLUE tint */
    border-radius: 6px;
}}

QTabWidget#mainTabs QTabBar::tab:selected {{
    color: {PRIMARY};
    font-weight: 600;
    background: transparent;
    border-bottom: 2px solid {SECONDARY};
}}
               
"""

MAIN_TAB_STYLES_DARK = f"""
/* ===== Main tabs (Dark) : modern IDE underline ===== */
QTabWidget#mainTabs::pane {{
    border: none;
    background: transparent;
}}

QTabWidget#mainTabs QTabBar {{
    qproperty-drawBase: 0;
    /* keep the same "bar" idea, just darker */
    background: rgba(46, 49, 145, 0.14);
    border: 1px solid {PALETTE['dark_border']};
    border-radius: 10px;
    padding: 2px 4px;
}}

QTabWidget#mainTabs QTabBar::tab {{
    background: transparent;
    color: {PALETTE['dark_text']};
    border: none;
    border-radius: 0px;

    /* reduced height */
    padding: 3px 10px;
    margin: 0px 4px;
    min-height: 18px;
}}

QTabWidget#mainTabs QTabBar::tab:hover {{
    background: rgba(51, 153, 255, 0.14);
    border-radius: 6px;
}}

QTabWidget#mainTabs QTabBar::tab:selected {{
    color: {PALETTE['dark_text_title']};
    font-weight: 600;
    background: transparent;
    border-bottom: 2px solid {SECONDARY};
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
    "MODE_DRY_COLOR",
    "MODE_TRAIN_COLOR",
    "MODE_TUNE_COLOR",
    "MODE_INFER_COLOR",
    "MODE_XFER_COLOR",
    "MODE_RESULTS_COLOR",
    "RUN_BUTTON_IDLE",
    "RUN_BUTTON_HOVER",
    "RUN_BUTTON_DISABLED",
    
    "MODE_DATA_COLOR",
    "MODE_SETUP_COLOR",
    "MODE_PREPROCESS_COLOR",
    "MAIN_TAB_STYLES_DARK", 
    "MAIN_TAB_STYLES_LIGHT"
    
]

