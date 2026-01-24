# -*- coding: utf-8 -*-
from __future__ import annotations

PREP_PATCH_LIGHT = """
/* ===== Preprocess: Build/Inspect segmented mode ===== */
QWidget#prepModeSeg {
  background: rgba(46,49,145,0.06);
  border: 1px solid rgba(46,49,145,0.22);
  border-radius: 10px;
}

QToolButton#prepModeBtn {
  background: transparent;
  border: 0px;
  padding: 4px 12px;
  font-weight: 800;
  color: rgba(30,30,30,0.78);
}

QToolButton#prepModeBtn[pos="left"]  { border-top-left-radius: 10px; border-bottom-left-radius: 10px; }
QToolButton#prepModeBtn[pos="right"] { border-top-right-radius: 10px; border-bottom-right-radius: 10px; }

QToolButton#prepModeBtn:hover:enabled {
  background: rgba(51,153,255,0.14);
}

QToolButton#prepModeBtn:checked {
  background: rgba(46,49,145,0.92);
  color: white;
}

/* ===== Key/value panels (Computer details + Recap) ===== */
QFrame#kvPanel {
  border: 1px solid rgba(46,49,145,0.18);
  border-radius: 12px;
  background: rgba(46,49,145,0.04);
}

QLabel#kvKey {
  color: rgba(100,116,139,0.95);
  font-size: 10.5px;
  font-weight: 700;
}

QLabel#kvVal {
  color: rgba(30,30,30,0.90);
  font-weight: 650;
}

QLabel#kvVal[path="true"] {
  padding: 2px 8px;
  border-radius: 8px;
  background: rgba(46,49,145,0.06);
  border: 1px solid rgba(46,49,145,0.14);
}
"""

PREP_PATCH_DARK = """
QWidget#prepModeSeg {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 10px;
}

QToolButton#prepModeBtn {
  background: transparent;
  border: 0px;
  padding: 4px 12px;
  font-weight: 800;
  color: rgba(226,232,240,0.90);
}

QToolButton#prepModeBtn[pos="left"]  { border-top-left-radius: 10px; border-bottom-left-radius: 10px; }
QToolButton#prepModeBtn[pos="right"] { border-top-right-radius: 10px; border-bottom-right-radius: 10px; }

QToolButton#prepModeBtn:hover:enabled {
  background: rgba(51,153,255,0.18);
}

QToolButton#prepModeBtn:checked {
  background: rgba(99,102,241,0.86);
  color: white;
}

QFrame#kvPanel {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  background: rgba(255,255,255,0.04);
}

QLabel#kvKey {
  color: rgba(148,163,184,0.95);
  font-size: 10.5px;
  font-weight: 700;
}

QLabel#kvVal {
  color: rgba(226,232,240,0.95);
  font-weight: 650;
}

QLabel#kvVal[path="true"] {
  padding: 2px 8px;
  border-radius: 8px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
}
"""
