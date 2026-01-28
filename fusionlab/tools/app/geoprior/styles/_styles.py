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
MODE_MAP_COLOR = "#0F766E"       # Map = teal (distinct from DRY)
MODE_TOOLS_COLOR = "#D97706"     # Tools = orange (distinct and eye-catching)
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

def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.strip().lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a:.2f})"


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
QWidget#cardHeaderRow,
QWidget#cardBadgesRow,
QWidget#cardActionsRow,
QWidget#cardBodyRoot {{
    background: transparent;
}}

QLabel#cardTitle {{
    font-size: 18px;
    font-weight: 600;
    color: {PRIMARY};
}}

QLabel#setupCardSubtitle {{
    font-size: 11px;
}}
QLabel#setupCardSubtitle {{
    color: rgba(30,30,30,0.72);
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

QToolButton#runButton {{
  background: transparent;
  border: none;
  padding: 2px;
  min-width: 40px;
  min-height: 40px;
  max-width: 40px;
  max-height: 40px;
}}
QToolButton#runButton:hover:enabled {{
  background-color: rgba(34, 197, 94, 0.16);
  border-radius: 20px;
}}
QToolButton#runButton:disabled {{
  background: transparent;
}}
               
QToolButton#miniAction,
QPushButton#miniAction {{
    background: transparent;
    border: 1px solid rgba(46,49,145,0.30);   /* PRIMARY with alpha */
    border-radius: 8px;
    padding: 2px 6px;
    /* add one of these */
    color: rgba(30,30,30,0.88);              /* neutral */
   /* or: color: rgba(46,49,145,0.92); */   /* brand/primary */
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
               
QPushButton#miniAction:disabled,
QToolButton#miniAction:disabled {{
    border-color: rgba(100,116,139,0.35);
    background: transparent;
    color: rgba(100,116,139,0.55);
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

QFrame#summaryPanel {{
    border: 1px solid rgba(46,49,145,0.18);
    background: rgba(46,49,145,0.04);
}}

QLabel#summaryPanelTitle {{
    color: rgba(46,49,145,0.96);
}}

QLabel#summaryKey {{
    color: rgba(30,30,30,0.62);
}}

QLabel#summaryValue {{
    color: rgba(30,30,30,0.90);
}}
QLabel#summaryPathValue {{
    padding: 2px 8px;
    border-radius: 8px;
    background: rgba(46,49,145,0.06);
    border: 1px solid rgba(46,49,145,0.18);
}}
 
QLabel#featStatus {{
    padding: 6px 10px;
    border-radius: 10px;
    background: rgba(46,49,145,0.06);
    border: 1px solid rgba(46,49,145,0.18);
}}

QLabel#featCount {{
    min-width: 22px;
    padding: 1px 7px;
    border-radius: 10px;
    background: rgba(46,49,145,0.10);
    border: 1px solid rgba(46,49,145,0.22);
    font-weight: 700;
}}

QLabel#featHint {{
    color: rgba(100,116,139,0.95);
    font-size: 10.5px;
}}

QListWidget#featList {{
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 10px;
    padding: 6px;
    background: rgba(255,255,255,0.90);
}}

QListWidget#featList::item {{
    padding: 4px 6px;
    border-radius: 8px;
}}
QLabel#featMissingChip {{
    min-width: 22px;
    padding: 1px 7px;
    border-radius: 10px;
    background: rgba(242,134,32,0.12);
    border: 1px solid rgba(242,134,32,0.35);
    font-weight: 800;
}}
               
/* ===== Map tab: Data panel ===== */
QScrollArea#mapDataScroll {{ border: none; background: transparent; }}
QWidget#mapDataHost {{ background: transparent; }}

QFrame#mapPanelCard {{
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 12px;
    background: rgba(255,255,255,0.92);
}}
QFrame#mapPanelCard[role="toolbar"] {{
    background: rgba(46,49,145,0.04);
}}

QLabel#mapSectionTitle {{
    font-weight: 700;
    color: rgba(46,49,145,0.96);
}}

QLabel#mapCountChip {{
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(46,49,145,0.06);
    border: 1px solid rgba(46,49,145,0.18);
    color: rgba(30,30,30,0.85);
    font-weight: 700;
}}

QLineEdit#mapSearch {{
    background: rgba(46,49,145,0.06);
    border: 1px solid rgba(46,49,145,0.22);
    border-radius: 10px;
    padding: 6px 10px;
}}

QLabel#mapStatusChip {{
    padding: 6px 10px;
    border-radius: 10px;
    background: rgba(46,49,145,0.06);
    border: 1px solid rgba(46,49,145,0.18);
    color: rgba(30,30,30,0.85);
}}

QTreeWidget#mapTree {{
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 10px;
    background: rgba(255,255,255,0.98);
    alternate-background-color: rgba(46,49,145,0.03);
    outline: 0;
}}
QTreeWidget#mapTree::item {{
    padding: 5px 6px;
    border-radius: 8px;
}}
QTreeWidget#mapTree::item:hover {{
    background: rgba(51,153,255,0.14);
}}
QTreeWidget#mapTree::item:selected {{
    background: rgba(46,49,145,0.18);
    color: rgba(15,23,42,0.95);
}}

QTreeWidget#mapTree QHeaderView::section {{
    background: rgba(46,49,145,0.06);
    border: none;
    padding: 4px 8px;
    font-weight: 600;
}}

/* ===== Map tab: Head (Light) ===== */
QFrame#mapHeadCard{{
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 12px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(46,49,145,0.06),
        stop:1 rgba(255,255,255,0.94)
    );
}}

QLabel#mapHeadPill{{
    padding: 3px 10px;
    border-radius: 10px;
    font-weight: 700;
    border: 1px solid rgba(46,49,145,0.28);
    background: rgba(46,49,145,0.10);
    color: rgba(15,23,42,0.92);
}}

QLabel#mapHeadDataset{{
    padding-left: 2px;
    font-size: 11px;
    color: rgba(30,30,30,0.68);
}}

QLabel#mapHeadKey{{
    font-weight: 700;
    color: rgba(46,49,145,0.90);
}}

QFrame#mapHeadGroup{{
    border: 1px solid rgba(46,49,145,0.16);
    border-radius: 12px;
    background: rgba(255,255,255,0.70);
}}

QComboBox#mapHeadCombo{{
    min-height: 30px;
    padding-left: 8px;
    border-radius: 10px;
    border: 1px solid rgba(46,49,145,0.20);
    background: rgba(46,49,145,0.05);
}}

QToolButton#mapHeadToggle{{
    border: 1px solid rgba(46,49,145,0.22);
    border-radius: 12px;
    padding: 4px 10px;
    font-weight: 700;
    color: rgba(30,30,30,0.86);
    background: rgba(255,255,255,0.65);
}}

QToolButton#mapHeadToggle:hover:enabled{{
    background: rgba(51,153,255,0.14);
    border-color: #3399ff;
}}

QToolButton#mapHeadToggle:checked{{
    background: rgba(46,49,145,0.18);
    border-color: rgba(46,49,145,0.45);
    color: rgba(15,23,42,0.95);
}}

/* Column picker */
QLabel#mapColLabel{{
    font-weight: 800;
    color: rgba(46,49,145,0.90);
}}

QLineEdit#mapColEdit{{
    border-radius: 10px;
    border: 1px solid rgba(46,49,145,0.20);
    background: rgba(46,49,145,0.05);
    padding: 6px 10px;
}}

/* Slightly larger miniAction inside head */
QToolButton#miniAction[role="mapHead"]{{
    padding: 4px 8px;
}}

/* ===== Xfer tab: Advanced options (Light) ===== */
QFrame#xferAdvSection {{
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 12px;
    background: rgba(255,255,255,0.92);
}}

QToolButton#xferAdvToggle {{
    background: rgba(46,49,145,0.05);
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 12px;
    padding: 6px 10px;
    font-weight: 800;
    color: {PALETTE['light_text']};
}}

QToolButton#xferAdvToggle:hover:enabled {{
    background: rgba(51,153,255,0.14);
    border-color: {SECONDARY_TBLUE};
}}

QToolButton#xferAdvToggle:checked {{
    background: rgba(46,49,145,0.10);
    border-color: rgba(46,49,145,0.30);
}}

QWidget#xferAdvBody {{
    background: transparent;
    border-top: 1px solid rgba(46,49,145,0.12);
}}

QLabel#xferAdvTitle {{
    font-size: 14px;
    font-weight: 800;
    color: {PRIMARY};
}}

QLabel#xferAdvChip {{
    min-width: 48px;
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(46,49,145,0.08);
    border: 1px solid rgba(46,49,145,0.22);
    color: rgba(15,23,42,0.90);
    font-weight: 800;
}}

QLabel#xferAdvChip[ok="false"] {{
    background: rgba(242,134,32,0.14);
    border: 1px solid rgba(242,134,32,0.55);
    color: rgba(15,23,42,0.95);
}}
QWidget#xferAdvContent {{
    background: transparent;
}}

QFrame#xferField {{
    border: 1px solid rgba(46,49,145,0.16);
    border-radius: 12px;
    background: rgba(255,255,255,0.92);
}}

QLabel#xferFieldTitle {{
    font-weight: 800;
    color: rgba(46,49,145,0.92);
}}
QGroupBox#xferWarmBox {{
    border: 1px solid rgba(46,49,145,0.14);
    border-radius: 10px;
    margin-top: 8px;
    padding-top: 6px;
}}
QGroupBox#xferWarmBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: rgba(46,49,145,0.92);
    font-weight: 700;
}}
/* -------- Xfer map toolbar: chips + segmented pills (Light) -------- */
QLabel#xferChipA {{
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    border-radius: 11px;
    background: rgba(46,49,145,0.14);
    border: 1px solid rgba(46,49,145,0.35);
    color: rgba(46,49,145,0.95);
    font-weight: 700;
}}

QLabel#xferChipB {{
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    border-radius: 11px;
    background: rgba(242,134,32,0.16);
    border: 1px solid rgba(242,134,32,0.40);
    color: rgba(172,78,10,0.98);
    font-weight: 700;
}}

QWidget#xferSeg {{
    background: rgba(46,49,145,0.06);
    border: 1px solid rgba(46,49,145,0.22);
    border-radius: 10px;
}}

QToolButton#xferSegBtn {{
    background: transparent;
    border: 0px;
    padding: 4px 10px;
    color: rgba(30,30,30,0.78);
}}

QToolButton#xferSegBtn[pos="left"] {{
    border-top-left-radius: 10px;
    border-bottom-left-radius: 10px;
}}
QToolButton#xferSegBtn[pos="right"] {{
    border-top-right-radius: 10px;
    border-bottom-right-radius: 10px;
}}

QToolButton#xferSegBtn:hover:enabled {{
    background: rgba(51,153,255,0.14);
}}

QToolButton#xferSegBtn:checked {{
    background: rgba(46,49,145,0.92);
    color: white;
}}
/* ===== Results tab (Light) ===== */
QTableWidget#resultsTable {{
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 12px;
    background: rgba(255,255,255,0.92);
    alternate-background-color: rgba(46,49,145,0.03);
    outline: 0;
}}
QTableWidget#resultsTable::item {{
    padding: 6px 8px;
}}
QTableWidget#resultsTable::item:hover {{
    background: rgba(51,153,255,0.14);
}}
QTableWidget#resultsTable::item:selected {{
    background: rgba(46,49,145,0.18);
    color: rgba(15,23,42,0.95);
}}
QTableWidget#resultsTable QHeaderView::section {{
    background: rgba(46,49,145,0.06);
    border: none;
    padding: 6px 10px;
    font-weight: 700;
}}
QLineEdit#resultsFilter {{
    border-radius: 10px;
}}

QFrame#resultsSummary {{
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 12px;
    background: rgba(255,255,255,0.92);
}}

QLabel#resultsChip {{
    padding: 3px 10px;
    border-radius: 10px;
    background: rgba(46,49,145,0.06);
    border: 1px solid rgba(46,49,145,0.18);
    color: rgba(30,30,30,0.88);
    font-weight: 700;
}}

QLabel#resultsRootChip,
QLabel#resultsScanChip {{
    padding: 3px 10px;
    border-radius: 10px;
    border: 1px solid rgba(46,49,145,0.22);
    background: rgba(46,49,145,0.06);
    font-weight: 800;
}}

QLabel#resultsRootChip[mode="custom"] {{
    border-color: rgba(242,134,32,0.55);
    background: rgba(242,134,32,0.14);
}}

QLabel#resultsHint {{
    color: rgba(100,116,139,0.95);
    font-size: 11px;
}}
#toolsNavTitle {{ font-weight: 700; font-size: 12pt; }}
#toolsNavFooter {{ color: palette(mid); font-size: 9pt; }}

#toolNavTitle {{ font-weight: 650; }}
#toolNavDesc {{ color: palette(mid); font-size: 9pt; }}

#toolPageTitle {{ font-weight: 750; font-size: 12pt; }}
#toolPageGroup {{
  padding: 2px 8px;
  border-radius: 10px;
  background: palette(midlight);
}}
#toolPageDesc {{ color: palette(mid); }}
#toolPageDivider {{ color: palette(midlight); }}

#toolsCmdTitle {{ font-weight: 650; }}
#toolsCmdSearch {{
  padding: 6px 10px;
  border-radius: 8px;
}}
#toolsNavSearch {{
  padding: 6px 10px;
  border-radius: 8px;
}}
QFrame#deviceMonOptions {{
  border: 1px solid palette(midlight);
  border-radius: 10px;
  padding: 6px;
  background: palette(base);
}}
QTreeWidget#deviceReportTree {{
  border: 1px solid palette(midlight);
  border-radius: 8px;
  background: palette(base);
}}

QTreeWidget#deviceReportTree::item {{
  padding: 4px 6px;
}}

QTreeWidget#deviceReportTree::item:selected {{
  background: palette(highlight);
  color: palette(highlighted-text);
}}
QScrollArea#scriptGenLeftScroll {{
  background: transparent;
}}
QScrollArea#scriptGenLeftScroll QWidget#scriptGenLeft {{
  background: transparent;
}}
               
          
/* ===== Tools: Dataset explorer ===== */

QFrame#dsxTop {{
  border: 1px solid palette(midlight);
  border-radius: 12px;
  background: palette(base);
}}

QLabel#dsxStatusChip {{
  padding: 6px 10px;
  border-radius: 10px;
  background: palette(alternate-base);
  border: 1px solid palette(midlight);
  font-weight: 700;
}}

QFrame#dsxSummary {{
  border: 1px solid palette(midlight);
  border-radius: 12px;
  background: palette(alternate-base);
}}

QLabel#dsxChip {{
  padding: 3px 10px;
  border-radius: 10px;
  background: palette(base);
  border: 1px solid palette(midlight);
  font-weight: 700;
}}

QLineEdit#dsxFilter {{
  padding: 6px 10px;
  border-radius: 10px;
}}

QTableWidget#dsxMissingTable {{
  border: 1px solid palette(midlight);
  border-radius: 12px;
  background: palette(base);
  outline: 0;
}}

QTableWidget#dsxMissingTable::item {{
  padding: 6px 8px;
  border-radius: 8px;
}}

QTableWidget#dsxMissingTable::item:hover {{
  background: palette(alternate-base);
}}

QFrame#dsxPreviewCard {{
  border: 1px solid palette(midlight);
  border-radius: 12px;
  background: palette(base);
}}

QLabel#dsxPreviewTitle {{
  font-weight: 800;
}}

QPlainTextEdit#dsxPreview {{
  border: 1px solid palette(midlight);
  border-radius: 10px;
  background: palette(base);
}}

QFrame#fxCard {{
  border: 1px solid palette(midlight);
  border-radius: 12px;
  background: palette(base);
}}

QLabel#fxTitle {{
  font-weight: 800;
}}

QLabel#fxRoles {{
  color: palette(text);
}}

QLabel#fxNote {{
  color: palette(text);
}}

QLabel#fxMiniTitle {{
  font-weight: 800;
}}

QTableWidget#fxMiniTable {{
  border: 1px solid palette(midlight);
  border-radius: 10px;
  background: palette(base);
}}

QHeaderView::section {{
  padding: 4px 6px;
  border: 0px;
}}
QLabel#dsxChip, QLabel#dsxStatusChip {{
  padding: 4px 10px;
  border-radius: 12px;
  border: 1px solid rgba(46,49,145,0.14);
  background: rgba(46,49,145,0.04);
}}
QFrame[flash="true"] {{
    border: 1px solid palette(highlight);
    border-radius: 10px;
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

QLabel#setupCardSubtitle {{
    font-size: 11px;
}}

QLabel#setupCardSubtitle {{
    color: rgba(232,234,237,0.72);
}}

QWidget#card {{
    background-color: {PALETTE['dark_card_bg']};
    border: 1px solid {PALETTE['dark_border']};
    border-radius: 12px;
}}

QWidget#cardHeaderRow,
QWidget#cardBadgesRow,
QWidget#cardActionsRow,
QWidget#cardBodyRoot {{
    background: transparent;
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

QFrame#summaryPanel {{
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.04);
}}

QLabel#summaryPanelTitle {{
    color: #ffffff;
}}

QLabel#summaryKey {{
    color: rgba(148,163,184,0.92);
}}

QLabel#summaryValue {{
    color: rgba(203,213,225,1.0);
}}
QLabel#summaryPathValue {{
    padding: 2px 8px;
    border-radius: 8px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
}}
QLabel#featStatus {{
    padding: 6px 10px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
}}

QLabel#featCount {{
    min-width: 22px;
    padding: 1px 7px;
    border-radius: 10px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.16);
    font-weight: 700;
}}

QLabel#featHint {{
    color: rgba(148,163,184,0.95);
    font-size: 10.5px;
}}

QListWidget#featList {{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    padding: 6px;
    background: rgba(15,23,42,0.35);
}}

QListWidget#featList::item {{
    padding: 4px 6px;
    border-radius: 8px;
}}
QLabel#featMissingChip {{
    min-width: 22px;
    padding: 1px 7px;
    border-radius: 10px;
    background: rgba(242,134,32,0.18);
    border: 1px solid rgba(242,134,32,0.45);
    font-weight: 800;
}}
               
QListWidget#setupNavList {{
    background: rgba(15,23,42,0.25);
    border: 1px solid rgba(255,255,255,0.14);
}}
/* ===== Map tab: Data panel (Dark) ===== */
QScrollArea#mapDataScroll {{ border: none; background: transparent; }}
QWidget#mapDataHost {{ background: transparent; }}

QFrame#mapPanelCard {{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    background: rgba(15,23,42,0.25);
}}
QFrame#mapPanelCard[role="toolbar"] {{
    background: rgba(46,49,145,0.14);
}}

QLabel#mapSectionTitle {{
    font-weight: 700;
    color: rgba(255,255,255,0.95);
}}

QLabel#mapCountChip {{
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(203,213,225,0.95);
    font-weight: 700;
}}

QLineEdit#mapSearch {{
    background: rgba(2,6,23,0.55);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    padding: 6px 10px;
    color: rgba(226,232,240,0.95);
}}

QLabel#mapStatusChip {{
    padding: 6px 10px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(226,232,240,0.95);
}}

QTreeWidget#mapTree {{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    background: rgba(15,23,42,0.35);
    alternate-background-color: rgba(255,255,255,0.03);
    outline: 0;
}}
QTreeWidget#mapTree::item {{ padding: 5px 6px; border-radius: 8px; }}
QTreeWidget#mapTree::item:hover {{ background: rgba(51,153,255,0.18); }}
QTreeWidget#mapTree::item:selected {{ background: rgba(46,49,145,0.26); }}

QTreeWidget#mapTree QHeaderView::section {{
    background: rgba(255,255,255,0.06);
    border: none;
    padding: 4px 8px;
    font-weight: 600;
    color: rgba(226,232,240,0.92);
}}
               
/* ===== Map tab: Head (Dark) ===== */
QFrame#mapHeadCard{{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(46,49,145,0.22),
        stop:1 rgba(15,23,42,0.28)
    );
}}

QLabel#mapHeadPill{{
    padding: 3px 10px;
    border-radius: 10px;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.06);
    color: rgba(226,232,240,0.95);
}}

QLabel#mapHeadDataset{{
    padding-left: 2px;
    font-size: 11px;
    color: rgba(148,163,184,0.92);
}}

QLabel#mapHeadKey{{
    font-weight: 700;
    color: rgba(226,232,240,0.92);
}}

QFrame#mapHeadGroup{{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    background: rgba(15,23,42,0.25);
}}

QComboBox#mapHeadCombo{{
    min-height: 30px;
    padding-left: 8px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(2,6,23,0.55);
}}

QToolButton#mapHeadToggle{{
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 12px;
    padding: 4px 10px;
    font-weight: 700;
    color: rgba(226,232,240,0.92);
    background: rgba(15,23,42,0.25);
}}

QToolButton#mapHeadToggle:hover:enabled{{
    background: rgba(51,153,255,0.18);
    border-color: #3399ff;
}}

QToolButton#mapHeadToggle:checked{{
    background: rgba(46,49,145,0.30);
    border-color: rgba(46,49,145,0.55);
    color: rgba(255,255,255,0.98);
}}

QLabel#mapColLabel{{
    font-weight: 800;
    color: rgba(226,232,240,0.92);
}}

QLineEdit#mapColEdit{{
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(2,6,23,0.55);
    padding: 6px 10px;
    color: rgba(226,232,240,0.95);
}}

QToolButton#miniAction[role="mapHead"]{{
    padding: 4px 8px;
}}
/* ===== Xfer tab: Advanced options (Dark) ===== */
QFrame#xferAdvSection {{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    background: rgba(15,23,42,0.25);
}}

QToolButton#xferAdvToggle {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    padding: 6px 10px;
    font-weight: 800;
    color: rgba(226,232,240,0.95);
}}

QToolButton#xferAdvToggle:hover:enabled {{
    background: rgba(51,153,255,0.18);
    border-color: {SECONDARY_TBLUE};
}}

QToolButton#xferAdvToggle:checked {{
    background: rgba(46,49,145,0.18);
    border-color: rgba(46,49,145,0.38);
}}

QWidget#xferAdvBody {{
    background: transparent;
    border-top: 1px solid rgba(255,255,255,0.10);
}}

QLabel#xferAdvTitle {{
    font-size: 14px;
    font-weight: 800;
    color: rgba(255,255,255,0.96);
}}

QLabel#xferAdvChip {{
    min-width: 48px;
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(203,213,225,0.95);
    font-weight: 800;
}}

QLabel#xferAdvChip[ok="false"] {{
    background: rgba(242,134,32,0.18);
    border: 1px solid rgba(242,134,32,0.45);
    color: rgba(255,255,255,0.95);
}}
QWidget#xferAdvContent {{
    background: transparent;
}}

QFrame#xferField {{
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    background: rgba(15,23,42,0.22);
}}

QLabel#xferFieldTitle {{
    font-weight: 800;
    color: rgba(226,232,240,0.95);
}}

QGroupBox#xferWarmBox {{
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    margin-top: 8px;
    padding-top: 6px;
}}
QGroupBox#xferWarmBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: rgba(226,232,240,0.95);
    font-weight: 700;
}}
/* -------- Xfer map toolbar: chips + segmented pills (Dark) -------- */
QLabel#xferChipA {{
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    border-radius: 11px;
    background: rgba(99,102,241,0.20);
    border: 1px solid rgba(99,102,241,0.35);
    color: rgba(255,255,255,0.92);
    font-weight: 700;
}}

QLabel#xferChipB {{
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    border-radius: 11px;
    background: rgba(242,134,32,0.22);
    border: 1px solid rgba(242,134,32,0.40);
    color: rgba(255,255,255,0.90);
    font-weight: 700;
}}

QWidget#xferSeg {{
    background: rgba(148,163,184,0.10);
    border: 1px solid rgba(148,163,184,0.22);
    border-radius: 10px;
}}

QToolButton#xferSegBtn {{
    background: transparent;
    border: 0px;
    padding: 4px 10px;
    color: rgba(203,213,225,0.86);
}}

QToolButton#xferSegBtn[pos="left"] {{
    border-top-left-radius: 10px;
    border-bottom-left-radius: 10px;
}}
QToolButton#xferSegBtn[pos="right"] {{
    border-top-right-radius: 10px;
    border-bottom-right-radius: 10px;
}}

QToolButton#xferSegBtn:hover:enabled {{
    background: rgba(51,153,255,0.18);
}}

QToolButton#xferSegBtn:checked {{
    background: rgba(99,102,241,0.86);
    color: white;
}}
/* ===== Results tab (Dark) ===== */
QTableWidget#resultsTable {{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    background: rgba(15,23,42,0.25);
    alternate-background-color: rgba(255,255,255,0.03);
    outline: 0;
}}
QTableWidget#resultsTable::item {{
    padding: 6px 8px;
}}
QTableWidget#resultsTable::item:hover {{
    background: rgba(51,153,255,0.18);
}}
QTableWidget#resultsTable::item:selected {{
    background: rgba(46,49,145,0.26);
}}
QTableWidget#resultsTable QHeaderView::section {{
    background: rgba(255,255,255,0.06);
    border: none;
    padding: 6px 10px;
    font-weight: 700;
    color: rgba(226,232,240,0.92);
}}
QLineEdit#resultsFilter {{
    border-radius: 10px;
}}

QFrame#resultsSummary {{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    background: rgba(15,23,42,0.25);
}}

QLabel#resultsChip {{
    padding: 3px 10px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(226,232,240,0.95);
    font-weight: 700;
}}

QLabel#resultsRootChip,
QLabel#resultsScanChip {{
    padding: 3px 10px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.06);
    color: rgba(226,232,240,0.95);
    font-weight: 800;
}}

QLabel#resultsRootChip[mode="custom"] {{
    border-color: rgba(242,134,32,0.45);
    background: rgba(242,134,32,0.18);
}}

QLabel#resultsHint {{
    color: rgba(148,163,184,0.95);
    font-size: 11px;
}}
#toolsNavTitle {{ font-weight: 700; font-size: 12pt; }}
#toolsNavFooter {{ color: palette(mid); font-size: 9pt; }}

#toolNavTitle {{ font-weight: 650; }}
#toolNavDesc {{ color: palette(mid); font-size: 9pt; }}

#toolPageTitle {{ font-weight: 750; font-size: 12pt; }}
#toolPageGroup {{
  padding: 2px 8px;
  border-radius: 10px;
  background: palette(midlight);
}}
#toolPageDesc {{ color: palette(mid); }}
#toolPageDivider {{ color: palette(midlight); }}

#toolsCmdTitle {{ font-weight: 650; }}    
#toolsCmdSearch {{
  padding: 6px 10px;
  border-radius: 8px;
}}
#toolsNavSearch {{
  padding: 6px 10px;
  border-radius: 8px;
}}
QFrame#deviceMonOptions {{
  border: 1px solid palette(midlight);
  border-radius: 10px;
  padding: 6px;
  background: palette(base);
}} 
QTreeWidget#deviceReportTree {{
  border: 1px solid palette(midlight);
  border-radius: 8px;
  background: palette(base);
}}

QTreeWidget#deviceReportTree::item {{
  padding: 4px 6px;
}}

QTreeWidget#deviceReportTree::item:selected {{
  background: palette(highlight);
  color: palette(highlighted-text);
}}
QScrollArea#scriptGenLeftScroll {{
  background: transparent;
}}
QScrollArea#scriptGenLeftScroll QWidget#scriptGenLeft {{
  background: transparent;
}}
QToolButton#runButton {{
  background: transparent;
  border: none;
  padding: 2px;
  min-width: 40px;
  min-height: 40px;
  max-width: 40px;
  max-height: 40px;
}}
QToolButton#runButton:hover:enabled {{
  background-color: rgba(34, 197, 94, 0.16);
  border-radius: 20px;
}}
QToolButton#runButton:disabled {{
  background: transparent;
}}
              
"""
INF_COMP_SCROLL_LIGHT = """
/* Inference: Computer details scroll area (Light) */
QScrollArea#inferCompScroll {
  background: transparent;
  border: none;
}

QScrollArea#inferCompScroll QWidget {
  background: transparent;
}

QScrollArea#inferCompScroll > QWidget > QWidget {
  background: transparent;
}

/* viewport (the actual painted area) */
QScrollArea#inferCompScroll QWidget#qt_scrollarea_viewport {
  background: transparent;
  border: none;
}

/* subtle scrollbar */
QScrollBar:vertical {
  background: transparent;
  width: 10px;
  margin: 0px;
}
QScrollBar::handle:vertical {
  background: rgba(46,49,145,0.18);
  border-radius: 5px;
  min-height: 18px;
}
QScrollBar::handle:vertical:hover {
  background: rgba(46,49,145,0.28);
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
  height: 0px;
}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
  background: transparent;
}
QScrollBar:horizontal {
  background: transparent;
  height: 10px;
  margin: 0px;
}
QScrollBar::handle:horizontal {
  background: rgba(46,49,145,0.18);
  border-radius: 5px;
  min-width: 18px;
}
QScrollBar::handle:horizontal:hover {
  background: rgba(46,49,145,0.28);
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
  width: 0px;
}
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
  background: transparent;
}
"""

INF_COMP_SCROLL_DARK = """
/* Inference: Computer details scroll area (Dark) */
QScrollArea#inferCompScroll {
  background: transparent;
  border: none;
}

QScrollArea#inferCompScroll QWidget {
  background: transparent;
}

QScrollArea#inferCompScroll > QWidget > QWidget {
  background: transparent;
}

QScrollArea#inferCompScroll QWidget#qt_scrollarea_viewport {
  background: transparent;
  border: none;
}

QScrollBar:vertical {
  background: transparent;
  width: 10px;
  margin: 0px;
}
QScrollBar::handle:vertical {
  background: rgba(226,232,240,0.16);
  border-radius: 5px;
  min-height: 18px;
}
QScrollBar::handle:vertical:hover {
  background: rgba(226,232,240,0.26);
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
  height: 0px;
}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
  background: transparent;
}
QScrollBar:horizontal {
  background: transparent;
  height: 10px;
  margin: 0px;
}
QScrollBar::handle:horizontal {
  background: rgba(226,232,240,0.16);
  border-radius: 5px;
  min-width: 18px;
}
QScrollBar::handle:horizontal:hover {
  background: rgba(226,232,240,0.26);
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
  width: 0px;
}
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
  background: transparent;
}


"""
INFER_CHIP ="""
QLabel#inferHeadTitle {
  font-weight: 900;
  color: rgba(46,49,145,0.96);
  padding-left: 6px;
}

QLabel#inferChip {
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 800;
  background: rgba(46,49,145,0.06);
  border: 1px solid rgba(46,49,145,0.18);
}

QLabel#inferChip[kind="ok"] {
  background: rgba(34,197,94,0.18);
  border-color: rgba(34,197,94,0.28);
}

QLabel#inferChip[kind="off"] {
  background: rgba(148,163,184,0.18);
  border-color: rgba(148,163,184,0.28);
}

QLabel#inferChip[kind="info"] {
  background: rgba(51,153,255,0.14);
  border-color: rgba(51,153,255,0.22);
}
QLabel#inferChip[kind="warn"] {
  background: rgba(245,158,11,0.18);
  border-color: rgba(245,158,11,0.28);
}
QLabel#inferChip[kind="err"] {
  background: rgba(239,68,68,0.16);
  border-color: rgba(239,68,68,0.26);
}
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

TRAIN_TAB_PATCH_LIGHT = f"""
/* ===== Train tab: modern toolbar + disclosure summaries (Light) ===== */

QWidget#trainTopBar {{
  background: rgba(46,49,145,0.04);
  border: 1px solid rgba(46,49,145,0.18);
  border-radius: 12px;
  padding: 4px 6px;
}}

QLabel#sumLine {{
  color: {PALETTE['light_text_muted']};
  font-size: 11px;
}}

QToolButton#disclosure {{
  background: rgba(46,49,145,0.06);
  border: 1px solid rgba(46,49,145,0.18);
  border-radius: 10px;
  padding: 3px 10px;
  font-weight: 700;
  color: rgba(30,30,30,0.84);
}}

QToolButton#disclosure:hover:enabled {{
  background: rgba(51,153,255,0.14);
  border-color: {SECONDARY_TBLUE};
}}

QToolButton#disclosure:checked {{
  background: rgba(46,49,145,0.12);
  border-color: rgba(46,49,145,0.30);
}}

QSplitter::handle {{
  background: transparent;
}}

QSplitter::handle:horizontal {{
  width: 8px;
}}

QSplitter::handle:horizontal:hover {{
  background: rgba(46,49,145,0.08);
  border-radius: 4px;
}}
"""
TRAIN_TAB_PATCH_DARK = f"""
/* ===== Train tab: modern toolbar + disclosure summaries (Dark) ===== */

QWidget#trainTopBar {{
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 4px 6px;
}}

QLabel#sumLine {{
  color: {PALETTE['dark_text_muted']};
  font-size: 11px;
}}

QToolButton#disclosure {{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 10px;
  padding: 3px 10px;
  font-weight: 700;
  color: rgba(226,232,240,0.92);
}}

QToolButton#disclosure:hover:enabled {{
  background: rgba(51,153,255,0.18);
  border-color: {SECONDARY_TBLUE};
}}

QToolButton#disclosure:checked {{
  background: rgba(46,49,145,0.22);
  border-color: rgba(46,49,145,0.40);
}}

QSplitter::handle {{
  background: transparent;
}}

QSplitter::handle:horizontal {{
  width: 8px;
}}

QSplitter::handle:horizontal:hover {{
  background: rgba(255,255,255,0.06);
  border-radius: 4px;
}}


"""
TRAIN_COMP_SCROLL_LIGHT= """
/* Training: Computer details scroll area (Light) */
QScrollArea#trainCompScroll {
  background: transparent;
  border: none;
}

QScrollArea#trainCompScroll QWidget {
  background: transparent;
}

QScrollArea#trainCompScroll > QWidget > QWidget {
  background: transparent;
}

/* viewport */
QScrollArea#trainCompScroll QWidget#qt_scrollarea_viewport {
  background: transparent;
  border: none;
}

/* scoped scrollbars (vertical + horizontal) */
QScrollArea#trainCompScroll QScrollBar:vertical {
  background: transparent;
  width: 10px;
  margin: 0px;
}
QScrollArea#trainCompScroll QScrollBar::handle:vertical {
  background: rgba(46,49,145,0.18);
  border-radius: 5px;
  min-height: 18px;
}
QScrollArea#trainCompScroll QScrollBar::handle:vertical:hover {
  background: rgba(46,49,145,0.28);
}
QScrollArea#trainCompScroll QScrollBar::add-line:vertical,
QScrollArea#trainCompScroll QScrollBar::sub-line:vertical {
  height: 0px;
}
QScrollArea#trainCompScroll QScrollBar::add-page:vertical,
QScrollArea#trainCompScroll QScrollBar::sub-page:vertical {
  background: transparent;
}

QScrollArea#trainCompScroll QScrollBar:horizontal {
  background: transparent;
  height: 10px;
  margin: 0px;
}
QScrollArea#trainCompScroll QScrollBar::handle:horizontal {
  background: rgba(46,49,145,0.18);
  border-radius: 5px;
  min-width: 18px;
}
QScrollArea#trainCompScroll QScrollBar::handle:horizontal:hover {
  background: rgba(46,49,145,0.28);
}
QScrollArea#trainCompScroll QScrollBar::add-line:horizontal,
QScrollArea#trainCompScroll QScrollBar::sub-line:horizontal {
  width: 0px;
}
QScrollArea#trainCompScroll QScrollBar::add-page:horizontal,
QScrollArea#trainCompScroll QScrollBar::sub-page:horizontal {
  background: transparent;
}
"""
TRAIN_COMP_SCROLL_DARK="""

/* Training: Computer details scroll area (Dark) */
QScrollArea#trainCompScroll {
  background: transparent;
  border: none;
}

QScrollArea#trainCompScroll QWidget {
  background: transparent;
}

QScrollArea#trainCompScroll > QWidget > QWidget {
  background: transparent;
}

QScrollArea#trainCompScroll QWidget#qt_scrollarea_viewport {
  background: transparent;
  border: none;
}

QScrollArea#trainCompScroll QScrollBar:vertical {
  background: transparent;
  width: 10px;
  margin: 0px;
}
QScrollArea#trainCompScroll QScrollBar::handle:vertical {
  background: rgba(226,232,240,0.16);
  border-radius: 5px;
  min-height: 18px;
}
QScrollArea#trainCompScroll QScrollBar::handle:vertical:hover {
  background: rgba(226,232,240,0.26);
}
QScrollArea#trainCompScroll QScrollBar::add-line:vertical,
QScrollArea#trainCompScroll QScrollBar::sub-line:vertical { height: 0px; }
QScrollArea#trainCompScroll QScrollBar::add-page:vertical,
QScrollArea#trainCompScroll QScrollBar::sub-page:vertical { background: transparent; }

QScrollArea#trainCompScroll QScrollBar:horizontal {
  background: transparent;
  height: 10px;
  margin: 0px;
}
QScrollArea#trainCompScroll QScrollBar::handle:horizontal {
  background: rgba(226,232,240,0.16);
  border-radius: 5px;
  min-width: 18px;
}
QScrollArea#trainCompScroll QScrollBar::handle:horizontal:hover {
  background: rgba(226,232,240,0.26);
}
QScrollArea#trainCompScroll QScrollBar::add-line:horizontal,
QScrollArea#trainCompScroll QScrollBar::sub-line:horizontal { width: 0px; }
QScrollArea#trainCompScroll QScrollBar::add-page:horizontal,
QScrollArea#trainCompScroll QScrollBar::sub-page:horizontal { background: transparent; }

"""
TRAIN_NAV_LIGHT = """
QFrame#trainNavCard {
  border: 1px solid rgba(46,49,145,0.18);
  border-radius: 12px;
  background: rgba(255,255,255,0.92);
}

QLabel#trainNavTitle {
  font-weight: 800;
  color: rgba(46,49,145,0.96);
}

QListWidget#trainNavList {
  border: 1px solid rgba(46,49,145,0.14);
  border-radius: 12px;
  background: rgba(46,49,145,0.03);
  outline: 0;
  padding: 6px;
}

QListWidget#trainNavList::item {
  padding: 7px 10px;
  border-radius: 10px;
  color: rgba(30,30,30,0.86);
}

QListWidget#trainNavList::item:hover {
  background: rgba(51,153,255,0.14);
}

QListWidget#trainNavList::item:selected {
  background: rgba(46,49,145,0.92);
  color: white;
}
"""

TRAIN_NAV_DARK = """
QFrame#trainNavCard {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  background: rgba(15,23,42,0.25);
}

QLabel#trainNavTitle {
  font-weight: 800;
  color: rgba(226,232,240,0.95);
}

QListWidget#trainNavList {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  background: rgba(255,255,255,0.04);
  outline: 0;
  padding: 6px;
}

QListWidget#trainNavList::item {
  padding: 7px 10px;
  border-radius: 10px;
  color: rgba(226,232,240,0.90);
}

QListWidget#trainNavList::item:hover {
  background: rgba(51,153,255,0.18);
}

QListWidget#trainNavList::item:selected {
  background: rgba(99,102,241,0.86);
  color: white;
}
"""

TRAIN_NAV_ROW ="""
QWidget#navRow {
  border-radius: 10px;
  background: transparent;
}

QWidget#navRow[selected="true"] {
  background: rgba(46,49,145,0.14);
}

QLabel#navText {
  font-weight: 700;
  text-decoration: none;
  border: none;
}

QLabel#navChip {
  padding: 2px 8px;
  border-radius: 9px;
  font-weight: 800;
  min-width: 34px;
}

QLabel#navChip[status="ok"] {
  background: rgba(34,197,94,0.18);
  color: rgba(22,101,52,0.95);
}

QLabel#navChip[status="warn"] {
  background: rgba(245,158,11,0.20);
  color: rgba(146,64,14,0.95);
}

QLabel#navChip[status="err"] {
  background: rgba(239,68,68,0.20);
  color: rgba(127,29,29,0.95);
}

QLabel#navChip[status="off"] {
  background: rgba(148,163,184,0.20);
  color: rgba(30,41,59,0.85);
}
QListWidget#trainNavList::item {
  padding: 0px;
  border: none;
  background: transparent;
}

QListWidget#trainNavList::item:selected {
  background: transparent;
}
QLabel#navChip {
  min-height: 18px;
}
"""
_CONSOLE_STYLES_LIGHT = f"""
QDockWidget#logDock {{
  border: 1px solid {PALETTE['light_border']};
  border-radius: 12px;
  background: {PALETTE['light_card_bg']};
}}

QWidget#consoleTitleBar {{
  background: {_rgba(PALETTE['primary'], 0.04)};
  border-bottom: 1px solid {_rgba(PALETTE['primary'], 0.14)};
}}

QLabel#consoleTitle {{
  color: {PALETTE['light_text_title']};
  font-weight: 700;
}}

QLabel#consoleChip {{
  padding: 2px 10px;
  border-radius: 10px;
  font-weight: 700;
  color: {PALETTE['light_text']};
  background: {_rgba(PALETTE['primary'], 0.06)};
  border: 1px solid {_rgba(PALETTE['primary'], 0.14)};
}}

QLabel#consoleChip[state="running"] {{
  background: {_rgba(SECONDARY_TBLUE, 0.16)};
  border-color: {_rgba(SECONDARY_TBLUE, 0.35)};
}}

QLabel#consoleChip[state="done"] {{
  background: {_rgba(RUN_BUTTON_IDLE, 0.16)};
  border-color: {_rgba(RUN_BUTTON_IDLE, 0.35)};
}}

QLabel#consoleChip[state="failed"] {{
  background: rgba(220,38,38,0.16);
  border-color: rgba(220,38,38,0.35);
}}

QWidget#consoleActions {{
  background: {_rgba(PALETTE['primary'], 0.04)};
  border-bottom: 1px solid {_rgba(PALETTE['primary'], 0.14)};
}}

QLineEdit#consoleFind {{
  border: 1px solid {_rgba(PALETTE['primary'], 0.22)};
  background: {_rgba(PALETTE['primary'], 0.06)};
  border-radius: 12px;
  padding: 6px 10px;
}}

QLineEdit#consoleFind:focus {{
  border-color: {_rgba(SECONDARY_TBLUE, 0.70)};
  background: {_rgba(SECONDARY_TBLUE, 0.08)};
}}

QTabWidget#consoleTabs::pane {{
  border: none;
  border-top: 1px solid {_rgba(PALETTE['primary'], 0.12)};
  background: transparent;
}}

QTabWidget#consoleTabs QTabBar::tab {{
  padding: 6px 12px;
  margin-right: 6px;
  border-radius: 10px;
  background: {_rgba(PALETTE['primary'], 0.06)};
  border: 1px solid {_rgba(PALETTE['primary'], 0.14)};
  color: {_rgba(PALETTE['light_text'], 0.86)};
  font-weight: 700;
}}

QTabWidget#consoleTabs QTabBar::tab:selected {{
  background: {_rgba(PALETTE['primary'], 0.14)};
  border: 1px solid {_rgba(PALETTE['primary'], 0.28)};
  color: {_rgba(PALETTE['primary'], 0.98)};
}}

QTabWidget#consoleTabs QTabBar::tab:hover {{
  background: {_rgba(SECONDARY_TBLUE, 0.10)};
  border-color: {_rgba(SECONDARY_TBLUE, 0.35)};
}}

QPlainTextEdit#logWidget {{
  border: 1px solid {_rgba(PALETTE['primary'], 0.14)};
  border-radius: 10px;
  background: #0b1220;
  color: rgba(226,232,240,0.96);
}}

QProgressBar#consoleProgress {{
  border: 1px solid {_rgba(PALETTE['primary'], 0.18)};
  border-radius: 7px;
  background: {_rgba(PALETTE['primary'], 0.06)};
  height: 12px;
}}

QProgressBar#consoleProgress::chunk {{
  border-radius: 7px;
  background: {PRIMARY_T75};
}}
QFrame#consoleSep {{
  max-width: 1px;
  min-width: 1px;
  background: rgba(46,49,145,0.18);
  margin: 0 6px;
}}

QLabel#consoleMatch {{
  padding: 2px 8px;
  border-radius: 10px;
  border: 1px solid rgba(46,49,145,0.14);
  background: rgba(46,49,145,0.06);
  color: rgba(15,23,42,0.74);
  font-weight: 700;
}}

QToolButton#consolePause:checked {{
  background: rgba(51,153,255,0.10);
  border: 1px solid rgba(51,153,255,0.35);
}}

QToolButton#consoleMore::menu-indicator {{ image: none; }}
QLabel#consolePending {{
  padding: 2px 7px;
  border-radius: 10px;
  border: 1px solid rgba(46,49,145,0.22);
  background: rgba(46,49,145,0.10);
  color: rgba(46,49,145,0.95);
  font-weight: 800;
  min-width: 18px;
  qproperty-alignment: AlignCenter;
}}
QLabel#consoleScopeInline {{
  padding: 0px 10px;
  color: palette(mid);
  font-size: 11px;
}}

"""
_CONSOLE_STYLES_DARK = f"""
QDockWidget#logDock {{
  border: 1px solid {PALETTE['dark_border']};
  border-radius: 12px;
  background: {PALETTE['dark_card_bg']};
}}

QWidget#consoleTitleBar {{
  background: rgba(255,255,255,0.04);
  border-bottom: 1px solid rgba(255,255,255,0.10);
}}

QLabel#consoleTitle {{
  color: {PALETTE['dark_text_title']};
  font-weight: 700;
}}

QLabel#consoleChip {{
  padding: 2px 10px;
  border-radius: 10px;
  font-weight: 700;
  color: {PALETTE['dark_text']};
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
}}

QLabel#consoleChip[state="running"] {{
  background: {_rgba(SECONDARY_TBLUE, 0.18)};
  border-color: {_rgba(SECONDARY_TBLUE, 0.40)};
}}

QLabel#consoleChip[state="done"] {{
  background: {_rgba(RUN_BUTTON_IDLE, 0.18)};
  border-color: {_rgba(RUN_BUTTON_IDLE, 0.40)};
}}

QLabel#consoleChip[state="failed"] {{
  background: rgba(220,38,38,0.18);
  border-color: rgba(220,38,38,0.40);
}}

QWidget#consoleActions {{
  background: rgba(255,255,255,0.03);
  border-bottom: 1px solid rgba(255,255,255,0.10);
}}

QLineEdit#consoleFind {{
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  border-radius: 12px;
  padding: 6px 10px;
}}

QLineEdit#consoleFind:focus {{
  border-color: {_rgba(SECONDARY_TBLUE, 0.70)};
  background: {_rgba(SECONDARY_TBLUE, 0.10)};
}}

QTabWidget#consoleTabs::pane {{
  border: none;
  border-top: 1px solid rgba(255,255,255,0.10);
  background: transparent;
}}

QTabWidget#consoleTabs QTabBar::tab {{
  padding: 6px 12px;
  margin-right: 6px;
  border-radius: 10px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  color: rgba(226,232,240,0.92);
  font-weight: 700;
}}

QTabWidget#consoleTabs QTabBar::tab:selected {{
  background: {_rgba(PALETTE['primary'], 0.22)};
  border: 1px solid {_rgba(PALETTE['primary'], 0.40)};
  color: rgba(255,255,255,0.98);
}}

QTabWidget#consoleTabs QTabBar::tab:hover {{
  background: {_rgba(SECONDARY_TBLUE, 0.14)};
  border-color: {_rgba(SECONDARY_TBLUE, 0.35)};
}}

QPlainTextEdit#logWidget {{
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 10px;
  background: #0b1220;
  color: rgba(226,232,240,0.96);
}}

QProgressBar#consoleProgress {{
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 7px;
  background: rgba(255,255,255,0.06);
  height: 12px;
}}

QProgressBar#consoleProgress::chunk {{
  border-radius: 7px;
  background: {PRIMARY_T75};
}}
"""

_DOCK_CHROME_LIGHT = f"""
/* ===== Generic GeoPrior docks (Light) ===== */

QDockWidget[gpDock="true"] {{
  border: 1px solid {_rgba(PALETTE['primary'], 0.18)};
  border-radius: 12px;
  background: {PALETTE['light_card_bg']};
}}

/* remove default title padding if you use a custom title bar */
QDockWidget[gpDock="true"]::title {{
  padding: 0px;
}}

/* docking splitter handle */
QDockWidget[gpDock="true"]::separator {{
  background: transparent;
  width: 8px;
  height: 8px;
}}
QDockWidget[gpDock="true"]::separator:hover {{
  background: {_rgba(PALETTE['primary'], 0.08)};
  border-radius: 4px;
}}

/* title bar */
QWidget#dockTitleBar {{
  background: {_rgba(PALETTE['primary'], 0.04)};
  border-bottom: 1px solid {_rgba(PALETTE['primary'], 0.14)};
}}

QLabel#dockTitle {{
  color: {PALETTE['light_text_title']};
  font-weight: 800;
}}

QLabel#dockChip {{
  padding: 2px 10px;
  border-radius: 10px;
  font-weight: 800;
  background: {_rgba(PALETTE['primary'], 0.06)};
  border: 1px solid {_rgba(PALETTE['primary'], 0.14)};
  color: {_rgba(PALETTE['light_text'], 0.90)};
}}

QLabel#dockChip[kind="ok"] {{
  background: {_rgba(RUN_BUTTON_IDLE, 0.16)};
  border-color: {_rgba(RUN_BUTTON_IDLE, 0.35)};
}}

QLabel#dockChip[kind="warn"] {{
  background: rgba(245,158,11,0.16);
  border-color: rgba(245,158,11,0.30);
}}

QLabel#dockChip[kind="err"] {{
  background: rgba(239,68,68,0.16);
  border-color: rgba(239,68,68,0.30);
}}

QToolButton#dockBtn {{
  background: transparent;
  border: 1px solid {_rgba(PALETTE['primary'], 0.18)};
  border-radius: 10px;
  padding: 2px 6px;
}}

QToolButton#dockBtn:hover:enabled {{
  background: {_rgba(SECONDARY_TBLUE, 0.12)};
  border-color: {_rgba(SECONDARY_TBLUE, 0.40)};
}}

QToolButton#dockBtn:pressed {{
  background: {_rgba(PALETTE['secondary'], 0.14)};
  border-color: {_rgba(PALETTE['secondary'], 0.45)};
}}

QLineEdit#dockSearch {{
  border: 1px solid {_rgba(PALETTE['primary'], 0.22)};
  background: {_rgba(PALETTE['primary'], 0.06)};
  border-radius: 12px;
  padding: 6px 10px;
}}

QLineEdit#dockSearch:focus {{
  border-color: {_rgba(SECONDARY_TBLUE, 0.70)};
  background: {_rgba(SECONDARY_TBLUE, 0.08)};
}}

QWidget#dockBody {{
  background: transparent;
}}
"""

_DOCK_CHROME_DARK = f"""
/* ===== Generic GeoPrior docks (Dark) ===== */

QDockWidget[gpDock="true"] {{
  border: 1px solid {PALETTE['dark_border']};
  border-radius: 12px;
  background: {PALETTE['dark_card_bg']};
}}

QDockWidget[gpDock="true"]::title {{
  padding: 0px;
}}

QDockWidget[gpDock="true"]::separator {{
  background: transparent;
  width: 8px;
  height: 8px;
}}
QDockWidget[gpDock="true"]::separator:hover {{
  background: rgba(255,255,255,0.06);
  border-radius: 4px;
}}

QWidget#dockTitleBar {{
  background: rgba(255,255,255,0.04);
  border-bottom: 1px solid rgba(255,255,255,0.10);
}}

QLabel#dockTitle {{
  color: {PALETTE['dark_text_title']};
  font-weight: 800;
}}

QLabel#dockChip {{
  padding: 2px 10px;
  border-radius: 10px;
  font-weight: 800;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  color: rgba(226,232,240,0.92);
}}

QLabel#dockChip[kind="ok"] {{
  background: {_rgba(RUN_BUTTON_IDLE, 0.18)};
  border-color: {_rgba(RUN_BUTTON_IDLE, 0.40)};
}}

QLabel#dockChip[kind="warn"] {{
  background: rgba(245,158,11,0.18);
  border-color: rgba(245,158,11,0.35);
}}

QLabel#dockChip[kind="err"] {{
  background: rgba(239,68,68,0.18);
  border-color: rgba(239,68,68,0.35);
}}

QToolButton#dockBtn {{
  background: transparent;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 10px;
  padding: 2px 6px;
}}

QToolButton#dockBtn:hover:enabled {{
  background: {_rgba(SECONDARY_TBLUE, 0.14)};
  border-color: {_rgba(SECONDARY_TBLUE, 0.35)};
}}

QToolButton#dockBtn:pressed {{
  background: {_rgba(PALETTE['secondary'], 0.20)};
  border-color: {_rgba(PALETTE['secondary'], 0.45)};
}}

QLineEdit#dockSearch {{
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  border-radius: 12px;
  padding: 6px 10px;
  color: rgba(226,232,240,0.92);
}}

QLineEdit#dockSearch:focus {{
  border-color: {_rgba(SECONDARY_TBLUE, 0.70)};
  background: {_rgba(SECONDARY_TBLUE, 0.10)};
}}

QWidget#dockBody {{
  background: transparent;
}}
"""
MAP_TOOL_DOCK_PANEL_LIGHT = """
QDockWidget#mapToolDock QFrame#mapToolDockBasic {
  border: 1px solid rgba(46,49,145,0.14);
  border-radius: 12px;
  background: rgba(46,49,145,0.03);
}

QDockWidget#mapToolDock QFrame#mapToolDockAdv {
  border: 1px solid rgba(46,49,145,0.14);
  border-radius: 12px;
  background: rgba(255,255,255,0.80);
}

QDockWidget#mapToolDock QFrame#mapToolDockAdvBody {
  border-top: 1px solid rgba(46,49,145,0.12);
  background: transparent;
}
"""

MAP_TOOL_DOCK_PANEL_DARK = """
QDockWidget#mapToolDock QFrame#mapToolDockBasic {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  background: rgba(255,255,255,0.04);
}

QDockWidget#mapToolDock QFrame#mapToolDockAdv {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  background: rgba(15,23,42,0.18);
}

QDockWidget#mapToolDock QFrame#mapToolDockAdvBody {
  border-top: 1px solid rgba(255,255,255,0.10);
  background: transparent;
}
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

SEARCH_STYLE ="""
QFrame#searchWrap {
    border: 1px solid rgba(0, 0, 0, 40);
    border-radius: 8px;
    background: white;
}

QToolButton#filterToggle {
    padding: 0px;
}

QToolButton#filterToggle:checked {
    border-radius: 6px;
    background: rgba(0, 120, 215, 30);
}

QLineEdit#searchEdit {
    border: none;
    background: transparent;
    padding: 2px 0px;
}
"""

_DOCK_CHROME_DARK += MAP_TOOL_DOCK_PANEL_LIGHT
_DOCK_CHROME_DARK += MAP_TOOL_DOCK_PANEL_DARK

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
    "MAIN_TAB_STYLES_LIGHT", 
    "MODE_MAP_COLOR",
    "MODE_TOOLS_COLOR",
]

