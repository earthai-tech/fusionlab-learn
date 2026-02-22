# geoprior/about/qss.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from ..styles import PALETTE, PRIMARY


def about_qss() -> str:
    p = PALETTE
    return f"""
QDialog#aboutDialog {{
  background: {p["light_bg"]};
  color: {p["light_text"]};
}}

QFrame#aboutHeader {{
  background: {p["light_card_bg"]};
  border-radius: 12px;
  border: 1px solid {p["light_border"]};
}}

QLabel#aboutTitle {{
  font-size: 20px;
  font-weight: 700;
  color: {p["light_text_title"]};
}}

QLabel#aboutTagline {{
  font-size: 12px;
  color: {p["light_text_muted"]};
}}

QLabel#aboutLinks {{
  color: {PRIMARY};
}}

QLabel#aboutChip {{
  padding: 3px 8px;
  border-radius: 10px;
  background: {p["light_input_bg"]};
  border: 1px solid {p["light_border"]};
  color: {p["light_text_muted"]};
  font-size: 11px;
}}

QListWidget#aboutNav {{
  background: {p["light_card_bg"]};
  border: 1px solid {p["light_border"]};
  border-radius: 12px;
  padding: 6px;
}}

QListWidget#aboutNav::item {{
  padding: 8px 10px;
  border-radius: 10px;
}}

QListWidget#aboutNav::item:selected {{
  background: {PRIMARY};
  color: white;
}}

QFrame#aboutCard {{
  background: {p["light_card_bg"]};
  border-radius: 12px;
  border: 1px solid {p["light_border"]};
}}

QLabel#aboutCardTitle {{
  font-size: 12px;
  font-weight: 700;
  color: {p["light_text_title"]};
}}

QLabel#aboutBody {{
  font-size: 12px;
}}

QLabel#aboutFootnote {{
  font-size: 11px;
  color: {p["light_text_muted"]};
}}

"""
