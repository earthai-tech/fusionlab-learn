# geoprior/ui/styles/_map.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

__all__ = [
    "MAP_DOCK_LIGHT",
    "MAP_DOCK_DARK",
]

# Notes:
# - Uses  existing miniAction styling for buttons
# - Keeps overlay background transparent (map remains dominant)
# - Drawer has soft border like miniAction, but more "card"
# - Title matches brand primary + bold, no heavy bar

MAP_DOCK_LIGHT = """
/* ===== Map: Overlay dock (Light) ===== */

QWidget#gpDockOverlay {
  background: transparent;
}

/* Drawer card */
QFrame#gpDockDrawer {
  margin: 8px;
  border: 1px solid rgba(46,49,145,0.24);     /* PRIMARY alpha */
  border-radius: 14px;
  background: rgba(255,255,255,0.92);
}

QFrame#gpDockDrawer:hover {
  border-color: rgba(51,153,255,0.38);        /* SECONDARY_TBLUE alpha */
}

/* Title label (pairs with miniAction) */
QLabel#gpDockTitle {
  font-weight: 800;
  color: rgba(46,49,145,0.92);                /* PRIMARY */
  padding: 2px 4px;
}

/* Make header row feel "tight" (matches miniAction padding) */
QFrame#gpDockDrawer QToolButton#miniAction,
QDialog#gpDockWindow QToolButton#miniAction {
  padding: 2px 6px;
  border-radius: 8px;
}

/* Inner body wrapper where panels are inserted */
QWidget#gpDockBody {
  background: transparent;
  border-radius: 10px;
}

/* Normalize scroll areas inside dock bodies */
QWidget#gpDockBody QScrollArea {
  border: none;
  background: transparent;
}

QWidget#gpDockBody QScrollArea > QWidget {
  background: transparent;
}

QWidget#gpDockBody QScrollArea > QWidget > QWidget {
  background: transparent;
}

/* Floating window */
QDialog#gpDockWindow {
  background: rgba(250,250,250,0.96);         /* BG_LIGHT alpha */
  border: 1px solid rgba(46,49,145,0.22);
  border-radius: 14px;
}

QDialog#gpDockWindow QWidget {
  background: transparent;
}
/* ===== Analytics (Light) ===== */

QFrame#MapAnalyticsPanel {
  background: rgba(250,250,250,0.96);
  border: 1px solid rgba(46,49,145,0.22);
  border-radius: 12px;
}

/* Scope title look to analytics panel only */
QFrame#MapAnalyticsPanel QLabel#gpDockTitle {
  font-weight: 800;
  color: rgba(30,30,30,0.90);
}

/* Inner padding for the analytics body */
QFrame#MapAnalyticsPanel QWidget#gpDockBody {
  padding: 6px;
}

/* Tabs frame */
QTabWidget#gpAnalyticsTabs::pane {
  border: 1px solid rgba(46,49,145,0.18);
  border-radius: 10px;
  top: -1px;
}

"""

MAP_DOCK_DARK = """
/* ===== Map: Overlay dock (Dark) ===== */

QWidget#gpDockOverlay {
  background: transparent;
}

/* Drawer card */
QFrame#gpDockDrawer {
  margin: 8px;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 14px;
  background: rgba(2,6,23,0.55);              /* deep translucent */
}

QFrame#gpDockDrawer:hover {
  border-color: rgba(51,153,255,0.28);
}

/* Title label */
QLabel#gpDockTitle {
  font-weight: 800;
  color: rgba(226,232,240,0.95);              /* slate-ish */
  padding: 2px 4px;
}

/* keep miniAction consistent */
QFrame#gpDockDrawer QToolButton#miniAction,
QDialog#gpDockWindow QToolButton#miniAction {
  padding: 2px 6px;
  border-radius: 8px;
}

/* Inner body */
QWidget#gpDockBody {
  background: transparent;
  border-radius: 10px;
}

QWidget#gpDockBody QScrollArea {
  border: none;
  background: transparent;
}

QWidget#gpDockBody QScrollArea > QWidget {
  background: transparent;
}

QWidget#gpDockBody QScrollArea > QWidget > QWidget {
  background: transparent;
}

/* Floating window */
QDialog#gpDockWindow {
  background: rgba(2,6,23,0.75);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 14px;
}

QDialog#gpDockWindow QWidget {
  background: transparent;
}
/* ===== Analytics (Dark) ===== */

QFrame#MapAnalyticsPanel {
  background: rgba(2,6,23,0.72);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 12px;
}

QFrame#MapAnalyticsPanel QLabel#gpDockTitle {
  font-weight: 800;
  color: rgba(226,232,240,0.95);
}

QFrame#MapAnalyticsPanel QWidget#gpDockBody {
  padding: 6px;
}

QTabWidget#gpAnalyticsTabs::pane {
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 10px;
  top: -1px;
}
QToolButton#mapHeadToggle[variant="mini"] {
  padding: 4px 10px;
}
"""

MAP_HEAD_LIGHT = """
/* ===== Map head (Light) ===== */
QToolButton#mapHeadToggle[variant="mini"] {
  padding: 4px 10px;
}

/* Keep head inputs visually consistent */
QFrame#mapHeadGroup QComboBox,
QFrame#mapHeadGroup QLineEdit {
  min-height: 28px;
  padding: 2px 8px;
}

/* Disabled state: readable + clearly inactive */
QFrame#mapHeadGroup QWidget:disabled {
  color: rgba(40, 40, 60, 140);
}

QFrame#mapHeadGroup QLineEdit:disabled,
QFrame#mapHeadGroup QComboBox:disabled {
  background: rgba(0, 0, 0, 0.04);
  border: 1px solid rgba(46,49,145,0.18);
}

QFrame#mapHeadGroup QComboBox:disabled::drop-down {
  background: rgba(0, 0, 0, 0.03);
}
"""

MAP_HEAD_DARK = """
/* ===== Map head (Dark) ===== */
QToolButton#mapHeadToggle[variant="mini"] {
  padding: 4px 10px;
}

QFrame#mapHeadGroup QComboBox,
QFrame#mapHeadGroup QLineEdit {
  min-height: 28px;
  padding: 2px 8px;
}

/* Disabled state: keep contrast in dark theme */
QFrame#mapHeadGroup QWidget:disabled {
  color: rgba(226, 232, 240, 140);
}

QFrame#mapHeadGroup QLineEdit:disabled,
QFrame#mapHeadGroup QComboBox:disabled {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

QFrame#mapHeadGroup QComboBox:disabled::drop-down {
  background: rgba(255, 255, 255, 0.05);
}
"""
MAP_SELECTION_LIGHT = """
/* ===== Selection drawer (Light) ===== */

QFrame#gpSelectionPanel {
  margin: 8px;
  border: 1px solid rgba(46,49,145,0.22);
  border-radius: 16px;
  background: rgba(255,255,255,0.92);
}

QFrame#gpSelectionPanel:hover {
  border-color: rgba(51,153,255,0.34);
}

/* Title */
QLabel#gpSelTitle {
  font-weight: 800;
  color: rgba(46,49,145,0.92);
  padding: 2px 4px;
}

/* Hint / summary */
QLabel#gpSelHint {
  color: rgba(30,30,30,0.76);
}

QLabel#gpSelSummary {
  color: rgba(30,30,30,0.90);
  padding: 2px 2px;
}

QLabel#gpSelBusy {
  color: rgba(46,49,145,0.78);
  padding: 4px 2px;
}

/* Plot card (SelectionPlot root) */
QWidget#gpSelPlot {
  border: 1px solid rgba(46,49,145,0.16);
  border-radius: 12px;
  background: rgba(255,255,255,0.70);
  padding: 6px;
}
    
QFrame#gpSelectionPanel {
  background: transparent;
}

QFrame#gpSelCard {
  background: rgba(255, 255, 255, 235);
  border-radius: 14px;
  border: 1px solid rgba(0, 0, 0, 28);
}

QScrollArea#gpSelDetails,
QWidget#gpSelDetailsVp {
  background: transparent;
}

QLabel#gpSelTitle {
  font-weight: 600;
}
/* Drag header (popover handle) */
QWidget#gpSelDragBar {
  border-top-left-radius: 14px;
  border-top-right-radius: 14px;
  background: transparent;
}

QWidget#gpSelDragBar:hover {
  background: rgba(0,0,0,0.03);   /* light hover */
}

/* Optional: make title + buttons align nicely */
QWidget#gpSelDragBar QLabel#gpSelTitle {
  padding: 2px 4px;
}

"""

MAP_SELECTION_DARK = """
/* ===== Selection drawer (Dark) ===== */

QFrame#gpSelectionPanel {
  margin: 8px;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 16px;
  background: rgba(2,6,23,0.55);
}

QFrame#gpSelectionPanel:hover {
  border-color: rgba(51,153,255,0.26);
}

QLabel#gpSelTitle {
  font-weight: 800;
  color: rgba(226,232,240,0.95);
  padding: 2px 4px;
}

QLabel#gpSelHint {
  color: rgba(226,232,240,0.72);
}

QLabel#gpSelSummary {
  color: rgba(226,232,240,0.88);
  padding: 2px 2px;
}

QLabel#gpSelBusy {
  color: rgba(148,163,184,0.92);
  padding: 4px 2px;
}

QWidget#gpSelPlot {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  background: rgba(2,6,23,0.34);
  padding: 6px;
}
QFrame#gpSelCard[theme="dark"] {
  background: rgba(26, 26, 28, 235);
  border: 1px solid rgba(255, 255, 255, 22);
}
/* Drag header (popover handle) */
QWidget#gpSelDragBar {
  border-top-left-radius: 14px;
  border-top-right-radius: 14px;
  background: transparent;
}

QWidget#gpSelDragBar:hover {
  background: rgba(255,255,255,0.06);  /* dark hover */
}
"""

MAP_DOCK_LIGHT += MAP_SELECTION_LIGHT
MAP_DOCK_DARK  += MAP_SELECTION_DARK
MAP_DOCK_LIGHT += MAP_HEAD_LIGHT
MAP_DOCK_DARK  += MAP_HEAD_DARK
