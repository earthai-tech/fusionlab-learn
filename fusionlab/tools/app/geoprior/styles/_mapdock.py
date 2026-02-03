# geoprior/ui/styles/_mapdock.py
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

"""

