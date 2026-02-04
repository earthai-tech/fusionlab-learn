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

XFER_ADVSEC_LIGHT = (
    XFER_ADVSEC_LIGHT
    + XFER_INTERP_LIGHT
    + XFER_INTERP_BADGES_LIGHT
)
# dark:
XFER_ADVSEC_DARK = (
    XFER_ADVSEC_DARK
    + XFER_INTERP_DARK
    + XFER_INTERP_BADGES_DARK
)

