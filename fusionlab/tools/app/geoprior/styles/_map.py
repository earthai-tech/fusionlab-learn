# geoprior/ui/styles/_map.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

__all__ = [
    "MAP_DOCK_LIGHT",
    "MAP_DOCK_DARK",
]
# SECONDARY
# Notes:
# - Uses  existing miniAction styling for buttons
# - Keeps overlay background transparent (map remains dominant)
# - Drawer has soft border like miniAction, but more "card"
# - Title matches brand primary + bold, no heavy bar

MAP_TAB_LIGHT= """
/* ===== Map tab: Data panel ===== */
QScrollArea#mapDataScroll { border: none; background: transparent; }
QWidget#mapDataHost { background: transparent; }

QFrame#mapPanelCard {
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 12px;
    background: rgba(255,255,255,0.92);
}
QFrame#mapPanelCard[role="toolbar"] {
    background: rgba(46,49,145,0.04);
}

QLabel#mapSectionTitle {
    font-weight: 700;
    color: rgba(46,49,145,0.96);
}

QLabel#mapCountChip {
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(46,49,145,0.06);
    border: 1px solid rgba(46,49,145,0.18);
    color: rgba(30,30,30,0.85);
    font-weight: 700;
}

QLineEdit#mapSearch {
    background: rgba(46,49,145,0.06);
    border: 1px solid rgba(46,49,145,0.22);
    border-radius: 10px;
    padding: 6px 10px;
}

QLabel#mapStatusChip {
    padding: 6px 10px;
    border-radius: 10px;
    background: rgba(46,49,145,0.06);
    border: 1px solid rgba(46,49,145,0.18);
    color: rgba(30,30,30,0.85);
}

QTreeWidget#mapTree {
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 10px;
    background: rgba(255,255,255,0.98);
    alternate-background-color: rgba(46,49,145,0.03);
    outline: 0;
}
QTreeWidget#mapTree::item {
    padding: 5px 6px;
    border-radius: 8px;
}
QTreeWidget#mapTree::item:hover {
    background: rgba(51,153,255,0.14);
}
QTreeWidget#mapTree::item:selected {
    background: rgba(46,49,145,0.18);
    color: rgba(15,23,42,0.95);
}

QTreeWidget#mapTree QHeaderView::section {
    background: rgba(46,49,145,0.06);
    border: none;
    padding: 4px 8px;
    font-weight: 600;
}

/* ===== Map tab: Head (Light) ===== */
QFrame#mapHeadCard{
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 12px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(46,49,145,0.06),
        stop:1 rgba(255,255,255,0.94)
    );
}

QLabel#mapHeadPill{
    padding: 3px 10px;
    border-radius: 10px;
    font-weight: 700;
    border: 1px solid rgba(46,49,145,0.28);
    background: rgba(46,49,145,0.10);
    color: rgba(15,23,42,0.92);
}

QLabel#mapHeadDataset{
    padding-left: 2px;
    font-size: 11px;
    color: rgba(30,30,30,0.68);
}

QLabel#mapHeadKey{
    font-weight: 700;
    color: rgba(46,49,145,0.90);
}

QFrame#mapHeadGroup{
    border: 1px solid rgba(46,49,145,0.16);
    border-radius: 12px;
    background: rgba(255,255,255,0.70);
}

QComboBox#mapHeadCombo{
    min-height: 30px;
    padding-left: 8px;
    border-radius: 10px;
    border: 1px solid rgba(46,49,145,0.20);
    background: rgba(46,49,145,0.05);
}

QToolButton#mapHeadToggle{
    border: 1px solid rgba(46,49,145,0.22);
    border-radius: 12px;
    padding: 4px 10px;
    font-weight: 700;
    color: rgba(30,30,30,0.86);
    background: rgba(255,255,255,0.65);
}

QToolButton#mapHeadToggle:hover:enabled{
    background: rgba(51,153,255,0.14);
    border-color: #3399ff;
}

QToolButton#mapHeadToggle:checked{
    background: rgba(46,49,145,0.18);
    border-color: rgba(46,49,145,0.45);
    color: rgba(15,23,42,0.95);
}

/* Mapping summary pill (Light) */
QToolButton#mapHeadMapping{
    border: 1px solid rgba(46,49,145,0.20);
    border-radius: 12px;
    padding: 4px 12px;
    font-weight: 800;
    color: rgba(15,23,42,0.92);
    background: rgba(255,255,255,0.72);
}

QToolButton#mapHeadMapping:hover:enabled{
    background: rgba(51,153,255,0.12);
    border-color: #3399ff;
}

QToolButton#mapHeadMapping[state="ok"]{
    background: rgba(16,163,74,0.10);
    border-color: rgba(16,163,74,0.42);
}

QToolButton#mapHeadMapping[state="warn"]{
    background: rgba(242,134,32,0.12);
    border-color: rgba(242,134,32,0.55);
}

/* Mapping popover (Light) */
/* ---QWidgetAction menus still paint item/hover behind the widget --- */
QMenu#mapMapMenu {
    background: transparent;
    border: none;
    padding: 0px;
    margin: 0px;
}

QMenu#mapMapMenu::item {
    background: transparent;
    padding: 0px;
    margin: 0px;
}

QMenu#mapMapMenu::item:selected {
    background: transparent;
}

QMenu#mapMapMenu::item:disabled {
    background: transparent;
}

QFrame#mapMapPopover{
    border: 1px solid rgba(46,49,145,0.18);
    border-radius: 12px;

    /* Glass popover: subtle tint top, clean white bottom */
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(46,49,145,0.08),
        stop:0.35 rgba(255,255,255,0.92),
        stop:1 rgba(255,255,255,0.98)
    );
}
QLabel#mapMapTitle{
    font-weight: 900;
    color: rgba(46,49,145,0.92);
}
QLabel#mapMapKey{
    font-weight: 900;
    color: rgba(15,23,42,0.90);
}
    
QLabel#mapMapHint{
    font-size: 11px;
    color: rgba(30,30,30,0.68);
}

QComboBox#mapMapCombo{
    min-height: 30px;
    padding-left: 8px;
    border-radius: 10px;
    border: 1px solid rgba(46,49,145,0.20);
    background: rgba(46,49,145,0.05);
}
/* Column picker */
QLabel#mapColLabel{
    font-weight: 800;
    color: rgba(46,49,145,0.90);
}

QLineEdit#mapColEdit{
    border-radius: 10px;
    border: 1px solid rgba(46,49,145,0.20);
    background: rgba(46,49,145,0.05);
    padding: 6px 10px;
}

/* Slightly larger miniAction inside head */
/* miniAction in map head/popover: compact "chip" look */
QToolButton#miniAction[role="mapHead"]{
    padding: 0px;                               /* icon-only */
    border-radius: 10px;
    border: 1px solid rgba(46,49,145,0.18);
    background: rgba(255,255,255,0.40);
}

QToolButton#miniAction[role="mapHead"]:hover:enabled{
    background: rgba(51,153,255,0.12);
    border-color: #3399ff;
}

QToolButton#miniAction[role="mapHead"]:pressed{
    background: rgba(242,134,32,0.14);
    border-color: rgba(242,134,32,0.55);
}

/* Accent action (OK) */
QToolButton#miniAction[role="mapHead"][accent="true"]{
    background: rgba(46,49,145,0.92);
    border-color: rgba(46,49,145,0.92);
    color: white;
}

QToolButton#miniAction[role="mapHead"][accent="true"]:hover:enabled{
    background: rgba(46,49,145,0.98);
    border-color: rgba(46,49,145,1.0);
}
/* Mapping chip container (Light) */
QFrame#mapHeadMapChip{
    border: 1px solid rgba(46,49,145,0.20);
    border-radius: 12px;
    background: rgba(255,255,255,0.72);
}

QFrame#mapHeadMapChip[state="ok"]{
    background: rgba(16,163,74,0.10);
    border-color: rgba(16,163,74,0.42);
}

QFrame#mapHeadMapChip[state="warn"]{
    background: rgba(242,134,32,0.12);
    border-color: rgba(242,134,32,0.55);
}

QFrame#mapHeadMapChip:hover{
    background: rgba(51,153,255,0.12);
    border-color: #3399ff;
}

/* Dot (Light) */
QLabel#mapHeadMapDot{
    border-radius: 4px;
    background: rgba(46,49,145,0.55);
}

QLabel#mapHeadMapDot[state="ok"]{
    background: rgba(16,163,74,0.95);
}

QLabel#mapHeadMapDot[state="warn"]{
    background: rgba(242,134,32,0.95);
}

QLabel#mapHeadMapDot[state="info"]{
    background: rgba(46,49,145,0.55);
}

/* Button inside chip (Light) */
QToolButton#mapHeadMapBtn{
    border: none;
    background: transparent;
    padding: 4px 0px;
    font-weight: 800;
    color: rgba(15,23,42,0.92);
}
QLabel#mapMapKey{
    font-weight: 800;
}

QComboBox#mapMapCombo{
    min-height: 28px;
    padding-left: 8px;
    border-radius: 10px;
}

QComboBox#mapMapCombo QAbstractItemView{
    border-radius: 10px;
    padding: 4px;
    min-width: 220px;
    max-height: 240px;
}
"""

MAP_TAB_DARK="""
/* ===== Map tab: Data panel (Dark) ===== */
QScrollArea#mapDataScroll { border: none; background: transparent; }
QWidget#mapDataHost { background: transparent; }

QFrame#mapPanelCard {
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    background: rgba(15,23,42,0.25);
}
QFrame#mapPanelCard[role="toolbar"] {
    background: rgba(46,49,145,0.14);
}

QLabel#mapSectionTitle {
    font-weight: 700;
    color: rgba(255,255,255,0.95);
}

QLabel#mapCountChip {
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(203,213,225,0.95);
    font-weight: 700;
}

QLineEdit#mapSearch {
    background: rgba(2,6,23,0.55);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    padding: 6px 10px;
    color: rgba(226,232,240,0.95);
}

QLabel#mapStatusChip {
    padding: 6px 10px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(226,232,240,0.95);
}

QTreeWidget#mapTree {
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    background: rgba(15,23,42,0.35);
    alternate-background-color: rgba(255,255,255,0.03);
    outline: 0;
}
QTreeWidget#mapTree::item { padding: 5px 6px; border-radius: 8px; }
QTreeWidget#mapTree::item:hover { background: rgba(51,153,255,0.18); }
QTreeWidget#mapTree::item:selected { background: rgba(46,49,145,0.26); }

QTreeWidget#mapTree QHeaderView::section {
    background: rgba(255,255,255,0.06);
    border: none;
    padding: 4px 8px;
    font-weight: 600;
    color: rgba(226,232,240,0.92);
}
               
/* ===== Map tab: Head (Dark) ===== */
QFrame#mapHeadCard{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(46,49,145,0.22),
        stop:1 rgba(15,23,42,0.28)
    );
}

QLabel#mapHeadPill{
    padding: 3px 10px;
    border-radius: 10px;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.06);
    color: rgba(226,232,240,0.95);
}

QLabel#mapHeadDataset{
    padding-left: 2px;
    font-size: 11px;
    color: rgba(148,163,184,0.92);
}

QLabel#mapHeadKey{
    font-weight: 700;
    color: rgba(226,232,240,0.92);
}

QFrame#mapHeadGroup{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    background: rgba(15,23,42,0.25);
}

QComboBox#mapHeadCombo{
    min-height: 30px;
    padding-left: 8px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(2,6,23,0.55);
}

QToolButton#mapHeadToggle{
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 12px;
    padding: 4px 10px;
    font-weight: 700;
    color: rgba(226,232,240,0.92);
    background: rgba(15,23,42,0.25);
}

QToolButton#mapHeadToggle:hover:enabled{
    background: rgba(51,153,255,0.18);
    border-color: #3399ff;
}

QToolButton#mapHeadToggle:checked{
    background: rgba(46,49,145,0.30);
    border-color: rgba(46,49,145,0.55);
    color: rgba(255,255,255,0.98);
}
    
/* Mapping summary pill (Dark) */
QToolButton#mapHeadMapping{
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 12px;
    padding: 4px 12px;
    font-weight: 800;
    color: rgba(226,232,240,0.95);
    background: rgba(15,23,42,0.25);
}

QToolButton#mapHeadMapping:hover:enabled{
    background: rgba(51,153,255,0.18);
    border-color: #3399ff;
}

QToolButton#mapHeadMapping[state="ok"]{
    background: rgba(16,163,74,0.14);
    border-color: rgba(16,163,74,0.45);
}

QToolButton#mapHeadMapping[state="warn"]{
    background: rgba(242,134,32,0.16);
    border-color: rgba(242,134,32,0.55);
}

/* Mapping popover (Dark) */
QMenu#mapMapMenu {
    background: transparent;
    border: none;
    padding: 0px;
    margin: 0px;
}

QMenu#mapMapMenu::item {
    background: transparent;
    padding: 0px;
    margin: 0px;
}

QMenu#mapMapMenu::item:selected {
    background: transparent;
}

QMenu#mapMapMenu::item:disabled {
    background: transparent;
}
QFrame#mapMapPopover{
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;

    /* Glass popover: gentle brand lift at top, deep slate bottom */
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(46,49,145,0.18),
        stop:0.35 rgba(15,23,42,0.72),
        stop:1 rgba(15,23,42,0.82)
    );
}
QLabel#mapMapTitle{
    font-weight: 900;
    color: rgba(226,232,240,0.95);
}
QLabel#mapMapKey{
    font-weight: 900;
    color: rgba(226,232,240,0.92);
}
QLabel#mapMapHint{
    font-size: 11px;
    color: rgba(148,163,184,0.92);
}

QComboBox#mapMapCombo{
    min-height: 30px;
    padding-left: 8px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(2,6,23,0.55);
    color: rgba(226,232,240,0.95);
}
QLabel#mapColLabel{
    font-weight: 800;
    color: rgba(226,232,240,0.92);
}

QLineEdit#mapColEdit{
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(2,6,23,0.55);
    padding: 6px 10px;
    color: rgba(226,232,240,0.95);
}

/* miniAction in map head/popover: compact "chip" look (dark) */
QToolButton#miniAction[role="mapHead"]{
    padding: 0px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.06);
    color: rgba(226,232,240,0.92);
}

QToolButton#miniAction[role="mapHead"]:hover:enabled{
    background: rgba(51,153,255,0.16);
    border-color: rgba(51,153,255,0.40);
}

QToolButton#miniAction[role="mapHead"]:pressed{
    background: rgba(242,134,32,0.20);
    border-color: rgba(242,134,32,0.55);
}

/* Accent action (OK) */
QToolButton#miniAction[role="mapHead"][accent="true"]{
    background: rgba(46,49,145,0.70);
    border-color: rgba(46,49,145,0.80);
    color: white;
}

QToolButton#miniAction[role="mapHead"][accent="true"]:hover:enabled{
    background: rgba(46,49,145,0.82);
    border-color: rgba(46,49,145,0.95);
}
/* Mapping chip container (Dark) */
QFrame#mapHeadMapChip{
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 12px;
    background: rgba(15,23,42,0.25);
}

QFrame#mapHeadMapChip[state="ok"]{
    background: rgba(16,163,74,0.14);
    border-color: rgba(16,163,74,0.45);
}

QFrame#mapHeadMapChip[state="warn"]{
    background: rgba(242,134,32,0.16);
    border-color: rgba(242,134,32,0.55);
}

QFrame#mapHeadMapChip:hover{
    background: rgba(51,153,255,0.18);
    border-color: #3399ff;
}

/* Dot (Dark) */
QLabel#mapHeadMapDot{
    border-radius: 4px;
    background: rgba(148,163,184,0.70);
}

QLabel#mapHeadMapDot[state="ok"]{
    background: rgba(16,163,74,0.95);
}

QLabel#mapHeadMapDot[state="warn"]{
    background: rgba(242,134,32,0.95);
}

QLabel#mapHeadMapDot[state="info"]{
    background: rgba(148,163,184,0.70);
}

/* Button inside chip (Dark) */
QToolButton#mapHeadMapBtn{
    border: none;
    background: transparent;
    padding: 4px 0px;
    font-weight: 800;
    color: rgba(226,232,240,0.95);
}
QLabel#mapMapKey{
    font-weight: 900;
    color: rgba(226,232,240,0.92);
}
QComboBox#mapMapCombo{
    min-height: 28px;
    padding-left: 8px;
    border-radius: 10px;
}

QComboBox#mapMapCombo QAbstractItemView{
    border-radius: 10px;
    padding: 4px;
    min-width: 220px;
    max-height: 240px;
}    
"""

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

/* ===== Analytics (Light) ===== */

QFrame#MapAnalyticsPanel {
  background: transparent;
  border: none;
}

QFrame#MapAnalyticsPanel QFrame#gpAnaCard {
  background: rgba(250,250,250,0.96);
  border: 1px solid rgba(46,49,145,0.22);
  border-radius: 12px;
}

QFrame#MapAnalyticsPanel QWidget#gpAnaDragBar {
  background: transparent;
}

QFrame#MapAnalyticsPanel QLabel#gpDockTitle {
  font-weight: 800;
  color: rgba(30,30,30,0.90);
}

QFrame#MapAnalyticsPanel QWidget#gpDockBody {
  padding: 6px;
  background: transparent;
}

QFrame#MapAnalyticsPanel QFrame#gpPlotCard {
  background: rgba(255,255,255,0.75);
  border: 1px solid rgba(46,49,145,0.18);
  border-radius: 10px;
}

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

/* ===== Analytics (Dark) ===== */

QFrame#MapAnalyticsPanel {
  background: transparent;
  border: none;
}

QFrame#MapAnalyticsPanel QFrame#gpAnaCard {
  background: rgba(2,6,23,0.72);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 12px;
}

QFrame#MapAnalyticsPanel QWidget#gpAnaDragBar {
  background: transparent;
}

QFrame#MapAnalyticsPanel QLabel#gpDockTitle {
  font-weight: 800;
  color: rgba(226,232,240,0.95);
}

QFrame#MapAnalyticsPanel QWidget#gpDockBody {
  padding: 6px;
  background: transparent;
}

QFrame#MapAnalyticsPanel QFrame#gpPlotCard {
  background: rgba(2,6,23,0.35);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 10px;
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
QFrame#mapHeadGroup[variant="plain"]{
    border: none;
    background: transparent;
}
QFrame#mapHeadMapChip{
    border: 1px solid rgba(46,49,145,0.20);
    border-radius: 12px;
    background: rgba(255,255,255,0.72);
    min-height: 32px;
}

QFrame#mapHeadMapChip[state="ok"]{
    background: rgba(16,163,74,0.10);
    border-color: rgba(16,163,74,0.42);
}

QFrame#mapHeadMapChip[state="warn"]{
    background: rgba(242,134,32,0.10);
    border-color: rgba(242,134,32,0.50);
}

QFrame#mapHeadMapChip:hover{
    background: rgba(51,153,255,0.12);
    border-color: #3399ff;
}

QLabel#mapHeadMapDot{
    border-radius: 4px;
    background: rgba(46,49,145,0.55);
}

QLabel#mapHeadMapDot[state="ok"]{
    background: rgba(16,163,74,0.95);
}

QLabel#mapHeadMapDot[state="warn"][pulse="0"]{
    background: rgba(242,134,32,0.55);
}

QLabel#mapHeadMapDot[state="warn"][pulse="1"]{
    background: rgba(242,134,32,0.95);
}

QLabel#mapHeadMapKey{
    font-weight: 800;
    color: rgba(15,23,42,0.90);
}

QLabel#mapHeadMapSep{
    color: rgba(15,23,42,0.45);
}

QLabel#mapHeadMapVal{
    font-weight: 600;
    color: rgba(15,23,42,0.78);
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
QFrame#mapHeadGroup[variant="plain"]{
    border: none;
    background: transparent;
}
QFrame#mapHeadMapChip{
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 12px;
    background: rgba(15,23,42,0.25);
    min-height: 32px;
}

QFrame#mapHeadMapChip[state="ok"]{
    background: rgba(16,163,74,0.14);
    border-color: rgba(16,163,74,0.45);
}

QFrame#mapHeadMapChip[state="warn"]{
    background: rgba(242,134,32,0.14);
    border-color: rgba(242,134,32,0.55);
}

QFrame#mapHeadMapChip:hover{
    background: rgba(51,153,255,0.18);
    border-color: #3399ff;
}

QLabel#mapHeadMapDot{
    border-radius: 4px;
    background: rgba(148,163,184,0.70);
}

QLabel#mapHeadMapDot[state="ok"]{
    background: rgba(16,163,74,0.95);
}

QLabel#mapHeadMapDot[state="warn"][pulse="0"]{
    background: rgba(242,134,32,0.55);
}

QLabel#mapHeadMapDot[state="warn"][pulse="1"]{
    background: rgba(242,134,32,0.95);
}

QLabel#mapHeadMapKey{
    font-weight: 800;
    color: rgba(226,232,240,0.92);
}

QLabel#mapHeadMapSep{
    color: rgba(226,232,240,0.42);
}

QLabel#mapHeadMapVal{
    font-weight: 600;
    color: rgba(226,232,240,0.78);
}
"""

MAP_TOOLTAB_LIGHT = """
/* ===== Map tooltab (Light) ===== */

QWidget#mapToolTab {
  background: rgba(255,255,255,0.88);
  border: 1px solid rgba(46,49,145,0.22);
  border-radius: 16px;
}

QWidget#mapToolTab:hover {
  background: rgba(255,255,255,0.94);
  border-color: rgba(51,153,255,0.38);
}

QWidget#mapToolTabHot {
  background: transparent;
}

/* Make each icon sit on a subtle chip */
QWidget#mapToolTab QToolButton#miniAction {
  background: rgba(255,255,255,0.22);
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 10px;
}

QWidget#mapToolTab QToolButton#miniAction:hover {
  background: rgba(0,0,0,0.06);
}

QWidget#mapToolTab QToolButton#miniAction:checked {
  background: rgba(51,153,255,0.18);
  border-color: rgba(51,153,255,0.34);
}

/* Vertical separators */
QWidget#mapToolTab QFrame {
  border: none;
  background: rgba(0,0,0,0.18);
  min-width: 1px;
  max-width: 1px;
  margin: 4px 6px;
}
"""
MAP_TOOLTAB_DARK = """
/* ===== Map tooltab (Dark) ===== */

QWidget#mapToolTab {
  background: rgba(2,6,23,0.70);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 16px;
}

QWidget#mapToolTab:hover {
  background: rgba(2,6,23,0.78);
  border-color: rgba(51,153,255,0.28);
}

QWidget#mapToolTabHot {
  background: transparent;
}

QWidget#mapToolTab QToolButton#miniAction {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 10px;
}

QWidget#mapToolTab QToolButton#miniAction:hover {
  background: rgba(255,255,255,0.10);
}

QWidget#mapToolTab QToolButton#miniAction:checked {
  background: rgba(51,153,255,0.18);
  border-color: rgba(51,153,255,0.26);
}

QWidget#mapToolTab QFrame {
  border: none;
  background: rgba(255,255,255,0.16);
  min-width: 1px;
  max-width: 1px;
  margin: 4px 6px;
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

MAP_DOCK_LIGHT += MAP_TOOLTAB_LIGHT
MAP_DOCK_DARK  += MAP_TOOLTAB_DARK

MAP_DOCK_LIGHT += MAP_SELECTION_LIGHT
MAP_DOCK_DARK  += MAP_SELECTION_DARK
MAP_DOCK_LIGHT += MAP_HEAD_LIGHT
MAP_DOCK_DARK  += MAP_HEAD_DARK

MAP_TAB_LIGHT += MAP_DOCK_LIGHT
MAP_TAB_DARK  +=MAP_DOCK_DARK
