# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Physics dialog styles (Light / Dark).

This module provides *scoped* QSS patches for the
PhysicsConfigDialog, identified by objectName:
    QDialog#physicsDialog

So it won't leak styles to other dialogs.
"""

from __future__ import annotations

from ._styles import (
    PALETTE,
    PRIMARY,
    SECONDARY,
    # SECONDARY_TBLUE,
)


PHYSICS_DLG_LIGHT = f"""
/* =========================================================
   Physics dialog (Light) - scoped to #physicsDialog
   ========================================================= */

QDialog#physicsDialog {{
  background: {PALETTE["light_bg"]};
  color: {PALETTE["light_text"]};
  font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}}

/* Titles inside this dialog only */
QDialog#physicsDialog QLabel#title {{
  font-weight: 900;
  font-size: 18px;
  color: {PALETTE["light_text_title"]};
}}

QDialog#physicsDialog QLabel#muted {{
  color: {PALETTE["light_text_muted"]};
}}

QDialog#physicsDialog QLabel[role="hint"] {{
  color: {PALETTE["light_text_muted"]};
  font-style: italic;
  font-size: 11px;
}}

/* Cards */
QDialog#physicsDialog QFrame#card {{
  background: {PALETTE["light_card_bg"]};
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 12px;
}}

QDialog#physicsDialog QFrame#settingRow {{
  background: {PALETTE["light_card_bg"]};
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 12px;
}}

QDialog#physicsDialog QFrame#settingRow[sel="true"] {{
  border: 1px solid rgba(0,0,0,0.22);
  background: rgba(46,49,145,0.04);
}}

/* Inputs */
QDialog#physicsDialog QLineEdit,
QDialog#physicsDialog QSpinBox,
QDialog#physicsDialog QDoubleSpinBox,
QDialog#physicsDialog QComboBox {{
  background: {PALETTE["light_input_bg"]};
  border: 1px solid rgba(0,0,0,0.16);
  border-radius: 10px;
  padding: 6px 10px;
}}

QDialog#physicsDialog QLineEdit:focus,
QDialog#physicsDialog QSpinBox:focus,
QDialog#physicsDialog QDoubleSpinBox:focus,
QDialog#physicsDialog QComboBox:focus {{
  border: 1px solid {PRIMARY};
}}

QDialog#physicsDialog QPlainTextEdit,
QDialog#physicsDialog QTextBrowser {{
  background: {PALETTE["light_input_bg"]};
  border: 1px solid rgba(0,0,0,0.14);
  border-radius: 12px;
  padding: 8px;
}}

/* Buttons */
QDialog#physicsDialog QPushButton {{
  background: {PRIMARY};
  color: white;
  border: 0px;
  border-radius: 10px;
  padding: 7px 14px;
  font-weight: 700;
}}

QDialog#physicsDialog QPushButton:hover:enabled {{
  background: {SECONDARY};
}}

QDialog#physicsDialog QPushButton#ghost {{
  background: transparent;
  color: {PALETTE["light_text"]};
  border: 1px solid rgba(0,0,0,0.18);
  border-radius: 10px;
  padding: 6px 12px;
  font-weight: 700;
}}

QDialog#physicsDialog QPushButton#ghost:hover:enabled {{
  background: rgba(0,0,0,0.04);
}}

QDialog#physicsDialog QPushButton#reset {{
  background: {PALETTE["light_reset_bg"]};
  color: {PALETTE["light_text"]};
  border: 1px solid rgba(0,0,0,0.14);
  border-radius: 10px;
  padding: 7px 14px;
  font-weight: 800;
}}

QDialog#physicsDialog QPushButton#reset:hover:enabled {{
  background: rgba(0,0,0,0.08);
}}

/* OK button (already named runButton in your dialog) */
QDialog#physicsDialog QPushButton#runButton:enabled {{
  background: {PRIMARY};
  color: white;
  border-radius: 10px;
  padding: 8px 16px;
  min-width: 90px;
  font-weight: 900;
}}

QDialog#physicsDialog QPushButton#runButton:hover:enabled {{
  background: {SECONDARY};
}}

/* =========================================================
   Pro left navigation (Light)
   ========================================================= */

QDialog#physicsDialog QFrame#physicsNavWrap {{
  background: qlineargradient(
    x1:0, y1:0, x2:0, y2:1,
    stop:0 rgba(46,49,145,0.05),
    stop:1 rgba(255,255,255,0.92)
  );
  border: 1px solid rgba(46,49,145,0.18);
  border-radius: 14px;
}}

QDialog#physicsDialog QLabel#physicsNavTitle {{
  font-weight: 900;
  color: rgba(46,49,145,0.96);
  padding: 6px 8px;
}}

QDialog#physicsDialog QTreeWidget#physicsNav {{
  background: transparent;
  border: none;
  outline: 0;
}}

QDialog#physicsDialog QTreeWidget#physicsNav::item {{
  padding: 7px 10px;
  margin: 2px 2px;
  border-radius: 10px;
  min-height: 26px;
  color: rgba(15,23,42,0.86);
  font-weight: 650;
}}

QDialog#physicsDialog QTreeWidget#physicsNav::item:hover {{
  background: rgba(51,153,255,0.14);
}}

QDialog#physicsDialog QTreeWidget#physicsNav::item:selected {{
  background: rgba(46,49,145,0.14);
  border: 1px solid rgba(46,49,145,0.28);
  color: rgba(15,23,42,0.95);
  font-weight: 900;
}}

QDialog#physicsDialog QTreeWidget#physicsNav::item:selected:hover {{
  background: rgba(46,49,145,0.18);
}}

QDialog#physicsDialog QTreeWidget#physicsNav::item:has-children {{
  color: rgba(100,116,139,0.95);
  font-weight: 900;
  padding-top: 10px;
  padding-bottom: 6px;
}}

/* Splitter handle */
QDialog#physicsDialog QSplitter::handle {{
  background: rgba(0,0,0,0.06);
}}
QDialog#physicsDialog QSplitter::handle:hover {{
  background: rgba(0,0,0,0.10);
}}

/* Badges (Changed / Override / Advanced) */
QDialog#physicsDialog QLabel#badge {{
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 800;
  border: 1px solid rgba(0,0,0,0.14);
  background: rgba(148,163,184,0.18);
  color: rgba(15,23,42,0.86);
}}

QDialog#physicsDialog QLabel#badge[accent="true"] {{
  background: {SECONDARY};
  border: 0px;
  color: white;
}}
"""


PHYSICS_DLG_DARK = f"""
/* =========================================================
   Physics dialog (Dark) - scoped to #physicsDialog
   ========================================================= */

QDialog#physicsDialog {{
  background: {PALETTE["dark_bg"]};
  color: {PALETTE["dark_text"]};
  font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}}

QDialog#physicsDialog QLabel#title {{
  font-weight: 900;
  font-size: 18px;
  color: {PALETTE["dark_text_title"]};
}}

QDialog#physicsDialog QLabel#muted {{
  color: {PALETTE["dark_text_muted"]};
}}

QDialog#physicsDialog QLabel[role="hint"] {{
  color: {PALETTE["dark_text_muted"]};
  font-style: italic;
  font-size: 11px;
}}

QDialog#physicsDialog QFrame#card {{
  background: rgba(15,23,42,0.25);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
}}

QDialog#physicsDialog QFrame#settingRow {{
  background: rgba(15,23,42,0.22);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
}}

QDialog#physicsDialog QFrame#settingRow[sel="true"] {{
  border: 1px solid rgba(46,49,145,0.55);
  background: rgba(46,49,145,0.16);
}}

QDialog#physicsDialog QLineEdit,
QDialog#physicsDialog QSpinBox,
QDialog#physicsDialog QDoubleSpinBox,
QDialog#physicsDialog QComboBox {{
  background: rgba(2,6,23,0.55);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 10px;
  padding: 6px 10px;
  color: rgba(226,232,240,0.95);
}}

QDialog#physicsDialog QLineEdit:focus,
QDialog#physicsDialog QSpinBox:focus,
QDialog#physicsDialog QDoubleSpinBox:focus,
QDialog#physicsDialog QComboBox:focus {{
  border: 1px solid {PRIMARY};
}}

QDialog#physicsDialog QPlainTextEdit,
QDialog#physicsDialog QTextBrowser {{
  background: rgba(2,6,23,0.55);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 12px;
  padding: 8px;
  color: rgba(226,232,240,0.95);
}}

QDialog#physicsDialog QPushButton {{
  background: {PRIMARY};
  color: white;
  border: 0px;
  border-radius: 10px;
  padding: 7px 14px;
  font-weight: 800;
}}

QDialog#physicsDialog QPushButton:hover:enabled {{
  background: {SECONDARY};
}}

QDialog#physicsDialog QPushButton#ghost {{
  background: transparent;
  color: rgba(226,232,240,0.95);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 10px;
  padding: 6px 12px;
  font-weight: 800;
}}

QDialog#physicsDialog QPushButton#ghost:hover:enabled {{
  background: rgba(255,255,255,0.06);
}}

QDialog#physicsDialog QPushButton#reset {{
  background: rgba(148,163,184,0.18);
  color: rgba(226,232,240,0.95);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 10px;
  padding: 7px 14px;
  font-weight: 900;
}}

QDialog#physicsDialog QPushButton#reset:hover:enabled {{
  background: rgba(148,163,184,0.26);
}}

QDialog#physicsDialog QPushButton#runButton:enabled {{
  background: {PRIMARY};
  color: white;
  border-radius: 10px;
  padding: 8px 16px;
  min-width: 90px;
  font-weight: 900;
}}

QDialog#physicsDialog QPushButton#runButton:hover:enabled {{
  background: {SECONDARY};
}}

/* =========================================================
   Pro left navigation (Dark)
   ========================================================= */

QDialog#physicsDialog QFrame#physicsNavWrap {{
  background: qlineargradient(
    x1:0, y1:0, x2:0, y2:1,
    stop:0 rgba(46,49,145,0.22),
    stop:1 rgba(15,23,42,0.28)
  );
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
}}

QDialog#physicsDialog QLabel#physicsNavTitle {{
  font-weight: 900;
  color: rgba(226,232,240,0.96);
  padding: 6px 8px;
}}

QDialog#physicsDialog QTreeWidget#physicsNav {{
  background: transparent;
  border: none;
  outline: 0;
}}

QDialog#physicsDialog QTreeWidget#physicsNav::item {{
  padding: 7px 10px;
  margin: 2px 2px;
  border-radius: 10px;
  min-height: 26px;
  color: rgba(226,232,240,0.90);
  font-weight: 650;
}}

QDialog#physicsDialog QTreeWidget#physicsNav::item:hover {{
  background: rgba(51,153,255,0.18);
}}

QDialog#physicsDialog QTreeWidget#physicsNav::item:selected {{
  background: rgba(46,49,145,0.30);
  border: 1px solid rgba(46,49,145,0.55);
  color: rgba(255,255,255,0.98);
  font-weight: 900;
}}

QDialog#physicsDialog QTreeWidget#physicsNav::item:has-children {{
  color: rgba(100,116,139,0.95);
  font-weight: 900;
  padding-top: 10px;
  padding-bottom: 6px;
}}
QDialog#physicsDialog QSplitter::handle {{
  background: rgba(255,255,255,0.06);
}}
QDialog#physicsDialog QSplitter::handle:hover {{
  background: rgba(255,255,255,0.10);
}}
QDialog#physicsDialog QLabel#badge {{
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 800;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(148,163,184,0.18);
  color: rgba(226,232,240,0.92);
}}

QDialog#physicsDialog QLabel#badge[accent="true"] {{
  background: {SECONDARY};
  border: 0px;
  color: white;
}}
"""