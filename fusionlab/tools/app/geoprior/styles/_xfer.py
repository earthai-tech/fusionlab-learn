from __future__ import annotations

XFER_ADVSEC_LIGHT = """
/* ===== Xfer Map: Advanced panel (Light) ===== */

QWidget#xferMapAdvancedPanel {
  background: transparent;
}

/* Each fold section becomes a soft card */
QFrame#xferMapAdvSection {
  padding: 8px;
  border: 1px solid rgba(46,49,145,0.16);
  border-radius: 14px;
  background: rgba(255,255,255,0.92);
}

QFrame#xferMapAdvSection:hover {
  border-color: rgba(51,153,255,0.38);
}

/* Header button = modern pill */
QToolButton#xferMapAdvHeader {
  background: rgba(46,49,145,0.06);
  border: 1px solid rgba(46,49,145,0.14);
  border-radius: 12px;
  padding: 7px 10px;
  font-weight: 800;
  color: rgba(15,23,42,0.92);
}

QToolButton#xferMapAdvHeader:hover:enabled {
  background: rgba(51,153,255,0.12);
  border-color: rgba(51,153,255,0.45);
}

QToolButton#xferMapAdvHeader:checked {
  background: rgba(46,49,145,0.12);
  border-color: rgba(46,49,145,0.30);
}

/* Hint below header */
QLabel#xferMapAdvHint {
  padding: 0px 10px 2px 10px;
  color: rgba(100,116,139,0.95);
  font-size: 10.5px;
  font-weight: 650;
}

/* Body area (when expanded) */
QFrame#xferMapAdvBody {
  background: transparent;
  border-top: 1px solid rgba(46,49,145,0.10);
  margin-top: 6px;
}

/* Default body labels */
QFrame#xferMapAdvBody QLabel {
  color: rgba(30,30,30,0.88);
}

/* Grid labels in the form-like rows */
QLabel#xferMapAdvLabel {
  color: rgba(46,49,145,0.92);
  font-weight: 800;
  /* If you need alignment, set it in code:
     lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
  */
}

/* Inputs inside the advanced panel */
QWidget#xferMapAdvancedPanel QSpinBox,
QWidget#xferMapAdvancedPanel QDoubleSpinBox,
QWidget#xferMapAdvancedPanel QComboBox {
  min-height: 30px;
  padding: 4px 10px;
  border-radius: 10px;
  background: rgba(46,49,145,0.05);
  border: 1px solid rgba(46,49,145,0.18);
}

QWidget#xferMapAdvancedPanel QSpinBox:hover,
QWidget#xferMapAdvancedPanel QDoubleSpinBox:hover,
QWidget#xferMapAdvancedPanel QComboBox:hover {
  border-color: rgba(51,153,255,0.55);
}

QWidget#xferMapAdvancedPanel QSpinBox:focus,
QWidget#xferMapAdvancedPanel QDoubleSpinBox:focus,
QWidget#xferMapAdvancedPanel QComboBox:focus {
  border-color: rgba(51,153,255,0.80);
  background: rgba(51,153,255,0.08);
}

QWidget#xferMapAdvancedPanel QSpinBox:disabled,
QWidget#xferMapAdvancedPanel QDoubleSpinBox:disabled,
QWidget#xferMapAdvancedPanel QComboBox:disabled {
  background: rgba(148,163,184,0.14);
  border-color: rgba(148,163,184,0.30);
  color: rgba(100,116,139,0.70);
}

/* Interactions block */
QWidget#xferMapInteractionsBlock {
  background: transparent;
}

QFrame#xferMapAdvDivider {
  border: none;
  border-top: 1px solid rgba(46,49,145,0.14);
  margin: 2px 0px;
}

QLabel#xferMapAdvSectionTitle {
  padding-top: 6px;
  font-weight: 800;
  color: rgba(46,49,145,0.96);
}

/* Scoped checkbox polish */
QWidget#xferMapInteractionsBlock QCheckBox {
  font-weight: 700;
  color: rgba(15,23,42,0.88);
}

QWidget#xferMapInteractionsBlock QCheckBox::indicator {
  width: 16px;
  height: 16px;
  margin-right: 8px;
  border-radius: 4px;
  border: 1px solid rgba(46,49,145,0.24);
  background: rgba(46,49,145,0.06);
}

QWidget#xferMapInteractionsBlock QCheckBox::indicator:checked {
  background: rgba(46,49,145,0.90);
  border-color: rgba(46,49,145,0.90);
}
"""

XFER_ADVSEC_DARK = """
/* ===== Xfer Map: Advanced panel (Dark) ===== */

QWidget#xferMapAdvancedPanel {
  background: transparent;
}

QFrame#xferMapAdvSection {
  padding: 8px;
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  background: rgba(15,23,42,0.22);
}

QFrame#xferMapAdvSection:hover {
  border-color: rgba(51,153,255,0.28);
}

QToolButton#xferMapAdvHeader {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 7px 10px;
  font-weight: 800;
  color: rgba(226,232,240,0.95);
}

QToolButton#xferMapAdvHeader:hover:enabled {
  background: rgba(51,153,255,0.14);
  border-color: rgba(51,153,255,0.32);
}

QToolButton#xferMapAdvHeader:checked {
  background: rgba(46,49,145,0.22);
  border-color: rgba(46,49,145,0.40);
}

QLabel#xferMapAdvHint {
  padding: 0px 10px 2px 10px;
  color: rgba(148,163,184,0.95);
  font-size: 10.5px;
  font-weight: 650;
}

QFrame#xferMapAdvBody {
  background: transparent;
  border-top: 1px solid rgba(255,255,255,0.10);
  margin-top: 6px;
}

QFrame#xferMapAdvBody QLabel {
  color: rgba(226,232,240,0.92);
}

QLabel#xferMapAdvLabel {
  color: rgba(226,232,240,0.92);
  font-weight: 800;
}

QWidget#xferMapAdvancedPanel QSpinBox,
QWidget#xferMapAdvancedPanel QDoubleSpinBox,
QWidget#xferMapAdvancedPanel QComboBox {
  min-height: 30px;
  padding: 4px 10px;
  border-radius: 10px;
  background: rgba(2,6,23,0.55);
  border: 1px solid rgba(255,255,255,0.14);
  color: rgba(226,232,240,0.95);
}

QWidget#xferMapAdvancedPanel QSpinBox:hover,
QWidget#xferMapAdvancedPanel QDoubleSpinBox:hover,
QWidget#xferMapAdvancedPanel QComboBox:hover {
  border-color: rgba(51,153,255,0.45);
}

QWidget#xferMapAdvancedPanel QSpinBox:focus,
QWidget#xferMapAdvancedPanel QDoubleSpinBox:focus,
QWidget#xferMapAdvancedPanel QComboBox:focus {
  border-color: rgba(51,153,255,0.70);
  background: rgba(51,153,255,0.10);
}

QWidget#xferMapAdvancedPanel QSpinBox:disabled,
QWidget#xferMapAdvancedPanel QDoubleSpinBox:disabled,
QWidget#xferMapAdvancedPanel QComboBox:disabled {
  background: rgba(148,163,184,0.10);
  border-color: rgba(148,163,184,0.20);
  color: rgba(148,163,184,0.65);
}

QFrame#xferMapAdvDivider {
  border: none;
  border-top: 1px solid rgba(255,255,255,0.12);
  margin: 2px 0px;
}

QLabel#xferMapAdvSectionTitle {
  padding-top: 6px;
  font-weight: 800;
  color: rgba(226,232,240,0.95);
}

QWidget#xferMapInteractionsBlock QCheckBox {
  font-weight: 700;
  color: rgba(226,232,240,0.92);
}

QWidget#xferMapInteractionsBlock QCheckBox::indicator {
  width: 16px;
  height: 16px;
  margin-right: 8px;
  border-radius: 4px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
}

QWidget#xferMapInteractionsBlock QCheckBox::indicator:checked {
  background: rgba(99,102,241,0.86);
  border-color: rgba(99,102,241,0.86);
}
"""

XFER_INTERP_LIGHT = """
/* ===== Xfer Map: Interpretation (Light) ===== */

/* QTextBrowser used as a "doc card" */
QTextBrowser#xferMapInterpDoc {
  background: rgba(46,49,145,0.035);
  border: 1px solid rgba(46,49,145,0.12);
  border-radius: 12px;
  padding: 10px 12px;
  color: rgba(15,23,42,0.92);
  font-size: 11px;
}

/* Make links look like app links */
QTextBrowser#xferMapInterpDoc a {
  color: rgba(51,153,255,0.95);
  text-decoration: none;
  font-weight: 750;
}
QTextBrowser#xferMapInterpDoc a:hover {
  text-decoration: underline;
}

/* Bullets spacing */
QTextBrowser#xferMapInterpDoc ul {
  margin-left: 14px;
  padding-left: 6px;
  margin-top: 6px;
  margin-bottom: 8px;
}
QTextBrowser#xferMapInterpDoc li {
  margin: 2px 0px;
}

/* Tip label below the doc */
QLabel#xferMapInterpTip {
  margin-top: 6px;
  padding: 8px 10px;
  border-radius: 12px;
  background: rgba(51,153,255,0.08);
  border: 1px solid rgba(51,153,255,0.18);
  color: rgba(15,23,42,0.88);
  font-size: 10.6px;
  font-weight: 700;
}

/* Optional: small emoji/icon feel without extra widgets */
QLabel#xferMapInterpTip::before {
  content: "💡  ";
}
"""


XFER_INTERP_DARK = """
/* ===== Xfer Map: Interpretation (Dark) ===== */

QTextBrowser#xferMapInterpDoc {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 10px 12px;
  color: rgba(226,232,240,0.92);
  font-size: 11px;
}

QTextBrowser#xferMapInterpDoc a {
  color: rgba(96,165,250,0.95);
  text-decoration: none;
  font-weight: 750;
}
QTextBrowser#xferMapInterpDoc a:hover {
  text-decoration: underline;
}

QTextBrowser#xferMapInterpDoc ul {
  margin-left: 14px;
  padding-left: 6px;
  margin-top: 6px;
  margin-bottom: 8px;
}
QTextBrowser#xferMapInterpDoc li {
  margin: 2px 0px;
}

QLabel#xferMapInterpTip {
  margin-top: 6px;
  padding: 8px 10px;
  border-radius: 12px;
  background: rgba(51,153,255,0.12);
  border: 1px solid rgba(51,153,255,0.22);
  color: rgba(226,232,240,0.92);
  font-size: 10.6px;
  font-weight: 700;
}

QLabel#xferMapInterpTip::before {
  content: "💡  ";
}
"""
# geoprior/ui/xfer/_xfer_advsec.py (badge styles)
# Works for QTextBrowser#xferMapInterpDoc HTML:
# <span class='gpBadge gpOk'>OK</span>
# <span class='gpBadge gpWarn'>WARN</span> ...

XFER_INTERP_BADGES_LIGHT = r"""
/* ===== Xfer Map: Interpretation badges (Light) ===== */

QTextBrowser#xferMapInterpDoc .gpBadge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 10px;
  letter-spacing: 0.2px;
  margin-right: 6px;
  border: 1px solid transparent;
}

/* status */
QTextBrowser#xferMapInterpDoc .gpOk {
  background: rgba(16,185,129,0.14);
  border-color: rgba(16,185,129,0.30);
  color: rgba(4,120,87,0.95);
}

QTextBrowser#xferMapInterpDoc .gpInfo {
  background: rgba(59,130,246,0.14);
  border-color: rgba(59,130,246,0.30);
  color: rgba(30,64,175,0.95);
}

QTextBrowser#xferMapInterpDoc .gpWarn {
  background: rgba(245,158,11,0.16);
  border-color: rgba(245,158,11,0.34);
  color: rgba(146,64,14,0.95);
}

QTextBrowser#xferMapInterpDoc .gpFail {
  background: rgba(239,68,68,0.14);
  border-color: rgba(239,68,68,0.34);
  color: rgba(153,27,27,0.95);
}

/* optional neutral "tag" badges (schema/badges list) */
QTextBrowser#xferMapInterpDoc .gpTag {
  background: rgba(148,163,184,0.18);
  border-color: rgba(148,163,184,0.32);
  color: rgba(15,23,42,0.85);
}
"""

XFER_INTERP_BADGES_DARK = r"""
/* ===== Xfer Map: Interpretation badges (Dark) ===== */

QTextBrowser#xferMapInterpDoc .gpBadge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 10px;
  letter-spacing: 0.2px;
  margin-right: 6px;
  border: 1px solid transparent;
}

/* status */
QTextBrowser#xferMapInterpDoc .gpOk {
  background: rgba(16,185,129,0.20);
  border-color: rgba(16,185,129,0.32);
  color: rgba(167,243,208,0.95);
}

QTextBrowser#xferMapInterpDoc .gpInfo {
  background: rgba(59,130,246,0.20);
  border-color: rgba(59,130,246,0.32);
  color: rgba(191,219,254,0.95);
}

QTextBrowser#xferMapInterpDoc .gpWarn {
  background: rgba(245,158,11,0.22);
  border-color: rgba(245,158,11,0.34);
  color: rgba(253,230,138,0.95);
}

QTextBrowser#xferMapInterpDoc .gpFail {
  background: rgba(239,68,68,0.20);
  border-color: rgba(239,68,68,0.34);
  color: rgba(254,202,202,0.95);
}

QTextBrowser#xferMapInterpDoc .gpTag {
  background: rgba(148,163,184,0.18);
  border-color: rgba(148,163,184,0.26);
  color: rgba(226,232,240,0.92);
}
"""

XFER_BM_QUICK_LIGHT = """
/* ===== Xfer Map: Basemap quick overlay (Light) ===== */

QFrame#xferBasemapDock {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(46,49,145,0.16);
  border-radius: 12px;
}

QFrame#xferBasemapDock QToolButton {
  border: 1px solid transparent;
  border-radius: 10px;
  padding: 6px 10px;
  font-weight: 800;
  color: rgba(15,23,42,0.92);
}

QFrame#xferBasemapDock QToolButton:hover:enabled {
  background: rgba(51,153,255,0.12);
  border-color: rgba(51,153,255,0.45);
}

/* Pin button state */
QToolButton#xferBasemapBtnPin:checked {
  background: rgba(46,49,145,0.12);
  border-color: rgba(46,49,145,0.30);
}

/* List container */
QWidget#xferBasemapList {
  background: transparent;
}

/* Basemap items */
QToolButton[bmItem="true"] {
  background: transparent;
  padding: 6px 10px;
  text-align: left;
  border-radius: 10px;
}

QToolButton[bmItem="true"]:checked {
  background: rgba(46,49,145,0.10);
  border-color: rgba(46,49,145,0.28);
}

"""

XFER_BM_QUICK_DARK = """
/* ===== Xfer Map: Basemap quick overlay (Dark) ===== */

QFrame#xferBasemapDock {
  background: rgba(15,23,42,0.26);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
}

QFrame#xferBasemapDock QToolButton {
  border: 1px solid transparent;
  border-radius: 10px;
  padding: 6px 10px;
  font-weight: 800;
  color: rgba(226,232,240,0.95);
}

QFrame#xferBasemapDock QToolButton:hover:enabled {
  background: rgba(51,153,255,0.14);
  border-color: rgba(51,153,255,0.32);
}

QToolButton#xferBasemapBtnPin:checked {
  background: rgba(46,49,145,0.22);
  border-color: rgba(46,49,145,0.40);
}

QWidget#xferBasemapList {
  background: transparent;
}

QToolButton[bmItem="true"] {
  background: transparent;
  padding: 6px 10px;
  text-align: left;
  border-radius: 10px;
}

QToolButton[bmItem="true"]:checked {
  background: rgba(99,102,241,0.18);
  border-color: rgba(99,102,241,0.32);
}

"""

XFER_CONTROLS_LIGHT = r"""
/* ===== Xfer Map: QuickBar + Controls (Light) ===== */
QDialog#xferControlsWindow,
QDialog#xferAdvWindow {
  background: transparent;
}

/* Window title (FusionLab PRIMARY) */
QLabel#xferWinTitle {
  color: rgba(46,49,145,0.96);
  font-weight: 950;
  font-size: 12.5px;
  padding: 2px 6px;
}
/* Quick bar container */
QFrame#xferQuickBarDock {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(46,49,145,0.16);
  border-radius: 14px;
}

/* Controls drawer container */
QFrame#xferControlsDock {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(46,49,145,0.16);
  border-radius: 16px;
}

/* Header title */
QLabel#xferControlsTitle {
  color: rgba(15,23,42,0.92);
  font-weight: 900;
  font-size: 12px;
  padding: 2px 6px;
}

/* Scroll area should look like a glass sheet */
QScrollArea#xferControlsScroll {
  background: transparent;
  border: none;
}
QScrollArea#xferControlsScroll QWidget#qt_scrollarea_viewport {
  background: transparent;
}

/* Base toolbutton look inside quick bar + controls */
QFrame#xferQuickBarDock QToolButton,
QFrame#xferControlsDock QToolButton {
  border: 1px solid transparent;
  border-radius: 10px;
  padding: 6px 10px;
  font-weight: 800;
  color: rgba(15,23,42,0.92);
}

/* Hover/pressed feel */
QFrame#xferQuickBarDock QToolButton:hover:enabled,
QFrame#xferControlsDock QToolButton:hover:enabled {
  background: rgba(51,153,255,0.12);
  border-color: rgba(51,153,255,0.45);
}

QFrame#xferQuickBarDock QToolButton:pressed,
QFrame#xferControlsDock QToolButton:pressed {
  background: rgba(46,49,145,0.14);
  border-color: rgba(46,49,145,0.30);
}

/* Pin button (checked) */
QToolButton#xferQuickPin:checked,
QFrame#xferControlsDock QToolButton#miniAction:checked {
  background: rgba(46,49,145,0.12);
  border-color: rgba(46,49,145,0.30);
}

/* Quick cities pill (make it look like a chip) */
QToolButton#xferQuickCities {
  background: rgba(46,49,145,0.06);
  border: 1px solid rgba(46,49,145,0.14);
  border-radius: 12px;
  padding: 6px 12px;
  font-weight: 900;
}
QToolButton#xferQuickCities:hover:enabled {
  background: rgba(51,153,255,0.12);
  border-color: rgba(51,153,255,0.45);
}

/* Segmented buttons in quick bar */
QToolButton[seg="true"] {
  background: rgba(46,49,145,0.05);
  border: 1px solid rgba(46,49,145,0.14);
  border-radius: 10px;
  padding: 6px 10px;
  font-weight: 900;
}
QToolButton[seg="true"]:hover:enabled {
  background: rgba(51,153,255,0.12);
  border-color: rgba(51,153,255,0.45);
}
QToolButton[seg="true"]:checked {
  background: rgba(46,49,145,0.12);
  border-color: rgba(46,49,145,0.30);
}

/* Segmented pill corners + remove double borders */
QToolButton[seg="true"][segPos="l"] {
  border-top-right-radius: 6px;
  border-bottom-right-radius: 6px;
}

QToolButton[seg="true"][segPos="m"] {
  border-radius: 6px;
  border-left-width: 0px;
}

QToolButton[seg="true"][segPos="r"] {
  border-top-left-radius: 6px;
  border-bottom-left-radius: 6px;
  border-left-width: 0px;
}
QToolButton[seg="true"] {
  padding: 6px 9px;
}

/* Step label badge */
QLabel#xferQuickStep {
  background: rgba(148,163,184,0.22);
  border: 1px solid rgba(148,163,184,0.32);
  border-radius: 999px;
  padding: 3px 10px;
  font-weight: 900;
  color: rgba(15,23,42,0.86);
}
/* Play/step pill (icon | text) */
QToolButton#xferQuickPlay {
  background: rgba(148,163,184,0.22);
  border: 1px solid rgba(148,163,184,0.32);
  border-radius: 999px;
  padding: 2px 10px;
  font-weight: 900;
  color: rgba(15,23,42,0.86);
}
QToolButton#xferQuickPlay:hover:enabled {
  background: rgba(51,153,255,0.10);
  border-color: rgba(51,153,255,0.40);
}
QToolButton#xferQuickPlay:checked {
  background: rgba(46,49,145,0.12);
  border-color: rgba(46,49,145,0.30);
}

/* Compact "miniAction" buttons in controls header */
QFrame#xferControlsDock QToolButton#miniAction {
  padding: 5px 8px;
  border-radius: 10px;
}

/* Inputs inside Controls drawer (toolbar widget content) */
QFrame#xferControlsDock QComboBox,
QFrame#xferControlsDock QSpinBox,
QFrame#xferControlsDock QDoubleSpinBox {
  min-height: 30px;
  padding: 4px 10px;
  border-radius: 10px;
  background: rgba(46,49,145,0.05);
  border: 1px solid rgba(46,49,145,0.18);
  color: rgba(15,23,42,0.92);
}
QFrame#xferControlsDock QComboBox:hover,
QFrame#xferControlsDock QSpinBox:hover,
QFrame#xferControlsDock QDoubleSpinBox:hover {
  border-color: rgba(51,153,255,0.55);
}
QFrame#xferControlsDock QComboBox:focus,
QFrame#xferControlsDock QSpinBox:focus,
QFrame#xferControlsDock QDoubleSpinBox:focus {
  border-color: rgba(51,153,255,0.80);
  background: rgba(51,153,255,0.08);
}

/* Sliders inside Controls drawer */
QFrame#xferControlsDock QSlider::groove:horizontal {
  height: 6px;
  border-radius: 999px;
  background: rgba(148,163,184,0.26);
}
QFrame#xferControlsDock QSlider::handle:horizontal {
  width: 14px;
  margin: -5px 0px;
  border-radius: 7px;
  background: rgba(46,49,145,0.70);
  border: 1px solid rgba(46,49,145,0.18);
}
QFrame#xferControlsHdrBar {
  background: transparent;
}
QFrame#xferControlsHdrBar:hover {
  background: rgba(51,153,255,0.06);
  border-radius: 12px;
}
/* ===== QuickBar compact (override) ===== */

QFrame#xferQuickBarDock QToolButton {
  padding: 4px 8px;
}

QToolButton#xferQuickCities {
  padding: 4px 10px;
}

QToolButton[seg="true"] {
  padding: 4px 8px;
}

QLabel#xferQuickStep {
  padding: 2px 8px;
}
QToolButton#xferQuickPlay {
  padding: 2px 8px;
}
"""

XFER_CONTROLS_DARK = r"""
/* ===== Xfer Map: QuickBar + Controls (Dark) ===== */
QDialog#xferControlsWindow,
QDialog#xferAdvWindow {
  background: transparent;
}

/* Window title (FusionLab PRIMARY, brighter for dark) */
QLabel#xferWinTitle {
  color: rgba(129,140,248,0.95);
  font-weight: 950;
  font-size: 12.5px;
  padding: 2px 6px;
}
/* Quick bar container */
QFrame#xferQuickBarDock {
  background: rgba(15,23,42,0.28);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
}

/* Controls drawer container */
QFrame#xferControlsDock {
  background: rgba(15,23,42,0.28);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
}

/* Header title */
QLabel#xferControlsTitle {
  color: rgba(226,232,240,0.95);
  font-weight: 900;
  font-size: 12px;
  padding: 2px 6px;
}

/* Scroll area */
QScrollArea#xferControlsScroll {
  background: transparent;
  border: none;
}
QScrollArea#xferControlsScroll QWidget#qt_scrollarea_viewport {
  background: transparent;
}

/* Base toolbutton look inside quick bar + controls */
QFrame#xferQuickBarDock QToolButton,
QFrame#xferControlsDock QToolButton {
  border: 1px solid transparent;
  border-radius: 10px;
  padding: 6px 10px;
  font-weight: 800;
  color: rgba(226,232,240,0.95);
}

/* Hover/pressed feel */
QFrame#xferQuickBarDock QToolButton:hover:enabled,
QFrame#xferControlsDock QToolButton:hover:enabled {
  background: rgba(51,153,255,0.14);
  border-color: rgba(51,153,255,0.32);
}

QFrame#xferQuickBarDock QToolButton:pressed,
QFrame#xferControlsDock QToolButton:pressed {
  background: rgba(99,102,241,0.18);
  border-color: rgba(99,102,241,0.28);
}

/* Pin button (checked) */
QToolButton#xferQuickPin:checked,
QFrame#xferControlsDock QToolButton#miniAction:checked {
  background: rgba(46,49,145,0.24);
  border-color: rgba(46,49,145,0.40);
}

/* Quick cities chip */
QToolButton#xferQuickCities {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 6px 12px;
  font-weight: 900;
}
QToolButton#xferQuickCities:hover:enabled {
  background: rgba(51,153,255,0.14);
  border-color: rgba(51,153,255,0.32);
}

/* Segmented buttons */
QToolButton[seg="true"] {
  background: rgba(2,6,23,0.55);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 10px;
  padding: 6px 10px;
  font-weight: 900;
  color: rgba(226,232,240,0.95);
}
QToolButton[seg="true"]:hover:enabled {
  border-color: rgba(51,153,255,0.45);
  background: rgba(51,153,255,0.10);
}
QToolButton[seg="true"]:checked {
  background: rgba(99,102,241,0.20);
  border-color: rgba(99,102,241,0.32);
}

/* Segmented pill corners + remove double borders */
QToolButton[seg="true"][segPos="l"] {
  border-top-right-radius: 6px;
  border-bottom-right-radius: 6px;
}

QToolButton[seg="true"][segPos="m"] {
  border-radius: 6px;
  border-left-width: 0px;
}

QToolButton[seg="true"][segPos="r"] {
  border-top-left-radius: 6px;
  border-bottom-left-radius: 6px;
  border-left-width: 0px;
}
QToolButton[seg="true"] {
  padding: 6px 9px;
}

/* Step badge */
QLabel#xferQuickStep {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 999px;
  padding: 3px 10px;
  font-weight: 900;
  color: rgba(226,232,240,0.92);
}
QToolButton#xferQuickPlay {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 999px;
  padding: 2px 10px;
  font-weight: 900;
  color: rgba(226,232,240,0.92);
}
QToolButton#xferQuickPlay:hover:enabled {
  background: rgba(51,153,255,0.12);
  border-color: rgba(51,153,255,0.28);
}
QToolButton#xferQuickPlay:checked {
  background: rgba(99,102,241,0.20);
  border-color: rgba(99,102,241,0.32);
}
    
/* Compact actions in controls header */
QFrame#xferControlsDock QToolButton#miniAction {
  padding: 5px 8px;
  border-radius: 10px;
}

/* Inputs inside Controls drawer */
QFrame#xferControlsDock QComboBox,
QFrame#xferControlsDock QSpinBox,
QFrame#xferControlsDock QDoubleSpinBox {
  min-height: 30px;
  padding: 4px 10px;
  border-radius: 10px;
  background: rgba(2,6,23,0.55);
  border: 1px solid rgba(255,255,255,0.14);
  color: rgba(226,232,240,0.95);
}
QFrame#xferControlsDock QComboBox:hover,
QFrame#xferControlsDock QSpinBox:hover,
QFrame#xferControlsDock QDoubleSpinBox:hover {
  border-color: rgba(51,153,255,0.45);
}
QFrame#xferControlsDock QComboBox:focus,
QFrame#xferControlsDock QSpinBox:focus,
QFrame#xferControlsDock QDoubleSpinBox:focus {
  border-color: rgba(51,153,255,0.70);
  background: rgba(51,153,255,0.10);
}

/* Sliders inside Controls drawer */
QFrame#xferControlsDock QSlider::groove:horizontal {
  height: 6px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
}
QFrame#xferControlsDock QSlider::handle:horizontal {
  width: 14px;
  margin: -5px 0px;
  border-radius: 7px;
  background: rgba(99,102,241,0.78);
  border: 1px solid rgba(255,255,255,0.14);
}

QFrame#xferControlsHdrBar {
  background: transparent;
}
QFrame#xferControlsHdrBar:hover {
  background: rgba(51,153,255,0.06);
  border-radius: 12px;
}
/* ===== QuickBar compact (override) ===== */

QFrame#xferQuickBarDock QToolButton {
  padding: 4px 8px;
}

QToolButton#xferQuickCities {
  padding: 4px 10px;
}

QToolButton[seg="true"] {
  padding: 4px 8px;
}

QLabel#xferQuickStep {
  padding: 2px 8px;
}
QToolButton#xferQuickPlay {
  padding: 2px 8px;
}
"""

XFER_ADVSEC_LIGHT = (
    XFER_ADVSEC_LIGHT
    + XFER_CONTROLS_LIGHT
    + XFER_INTERP_LIGHT
    + XFER_INTERP_BADGES_LIGHT
    + XFER_BM_QUICK_LIGHT
)

XFER_ADVSEC_DARK = (
    XFER_ADVSEC_DARK
    + XFER_CONTROLS_DARK
    + XFER_INTERP_DARK
    + XFER_INTERP_BADGES_DARK
    + XFER_BM_QUICK_DARK
)

