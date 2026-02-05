# geoprior/utils/generic_utils.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict,Any, Optional, Tuple


_QSUF_RE = re.compile(r"^(?P<base>.*)_(?P<tag>[qp])(?P<n>\d+)$")
_SEP_RE = re.compile(r"[._\s]+")


# ---------------------------------------------------------------------
# Default "smart" scientific label rules (view-only).
# Keys are normalized (lowercase, separators collapsed to "_").
# ---------------------------------------------------------------------
DEFAULT_LABEL_MAP: Dict[str, str] = {
    # Coordinates
    "lon": "Longitude",
    "longitude": "Longitude",
    "lat": "Latitude",
    "latitude": "Latitude",
    "coord_x": "X",
    "coord_y": "Y",
    "x": "X",
    "y": "Y",
    # Time
    "t": "Time",
    "time": "Time",
    "year": "Year",
    # Hydro / climate
    "subsidence": "Subsidence",
    "uplift": "Uplift",
    "gwl": "Groundwater level",
    "groundwater_level": "Groundwater level",
    "groundwater": "Groundwater",
    "head": "Hydraulic head",
    "hydraulic_head": "Hydraulic head",
    "drawdown": "Drawdown",
    "rain": "Rainfall",
    "rainfall": "Rainfall",
    "precip": "Precipitation",
    "precipitation": "Precipitation",
    "temp": "Temperature",
    "temperature": "Temperature",
    "et": "Evapotranspiration",
    "evapotranspiration": "Evapotranspiration",
    "evap": "Evaporation",
    # Physics / parameters
    "k": "Hydraulic conductivity K",
    "hydraulic_conductivity": "Hydraulic conductivity K",
    "logk": "log10(K)",
    "log10k": "log10(K)",
    "ss": "Specific storage Sₛ",
    "s_s": "Specific storage Sₛ",
    "specific_storage": "Specific storage Sₛ",
    "tau": "Time constant τ",
    "heff": "Effective thickness H_eff",
    "h_eff": "Effective thickness H_eff",
    "h": "Head h",
    "u_star": "U*",
    "ustar": "U*",
    # Generic stats
    "mean": "Mean",
    "median": "Median",
    "std": "Std. dev.",
    "stdev": "Std. dev.",
    "var": "Variance",
    "min": "Minimum",
    "max": "Maximum",
}


def normalize_key(name: str) -> str:
    """
    Normalize a name into a stable lookup key.

    Parameters
    ----------
    name : str
        Any identifier, e.g. "subsidence_q50",
        "map.subsidence_q50", "GWL", "coord_x".

    Returns
    -------
    key : str
        Lowercase key with separators collapsed to "_".

    Notes
    -----
    This is intended for *view-only* labelling logic.
    """
    s = str(name or "").strip().lower()
    s = s.replace("-", "_")
    s = _SEP_RE.sub("_", s).strip("_")
    return s


def split_quantile_suffix(name: str) -> Tuple[str, str]:
    """
    Split a trailing quantile/prob suffix from a column name.

    Parameters
    ----------
    name : str
        e.g. "subsidence_q50", "pga_p90".

    Returns
    -------
    base : str
        Name without the suffix.
    tag : str
        "" if no suffix, else "q50" or "p90".

    Examples
    --------
    >>> split_quantile_suffix("subsidence_q50")
    ('subsidence', 'q50')
    >>> split_quantile_suffix("head")
    ('head', '')
    """
    s = str(name or "").strip()
    m = _QSUF_RE.match(s)
    if not m:
        return s, ""
    base = m.group("base")
    tag = f"{m.group('tag')}{m.group('n')}"
    return base, tag


def quantile_tag(name: str, *, zfill: int = 2) -> str:
    """
    Return a compact quantile tag for display (e.g. q05, q50).

    Parameters
    ----------
    name : str
        Source name, possibly including "_q.." or "_p..".
    zfill : int, default=2
        Zero-pad digits (q5 -> q05) when possible.

    Returns
    -------
    tag : str
        "" if none, otherwise "qNN" or "pNN".
    """
    _base, tag = split_quantile_suffix(name)
    if not tag:
        return ""
    t = tag[:1]
    n = tag[1:]
    if n.isdigit():
        n = n.zfill(int(zfill))
    return f"{t}{n}"


def humanize_name(
    name: str,
    *,
    take_last: bool = True,
) -> str:
    """
    Convert a technical key into a readable title.

    Parameters
    ----------
    name : str
        Identifier like "subsidence_q50" or
        "map.obs_col" or "coord_x".
    take_last : bool, default=True
        If True, uses only the last dotted segment.
        Example: "map.subsidence_q50" -> "subsidence_q50".

    Returns
    -------
    text : str
        Title-cased text with underscores removed.

    Notes
    -----
    This does not append units and does not remove
    quantile tags; use `pretty_label()` for that.
    """
    s = str(name or "").strip()
    if take_last and "." in s:
        s = s.split(".")[-1].strip()

    s = s.replace(".", " ").replace("_", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")

    if not s:
        return ""

    parts = [p for p in s.split(" ") if p]
    return " ".join(p[:1].upper() + p[1:] for p in parts)


def pretty_label(
    name: str,
    *,
    unit: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None,
    drop_quantile: bool = True,
    include_quantile: bool = False,
    take_last: bool = True,
) -> str:
    """
    Make a smart, nice label for UI (view-only).

    Parameters
    ----------
    name : str
        Column/key name (e.g. "subsidence_q50").
    unit : str or None, default=None
        Appended as "(unit)" when provided.
    label_map : dict or None, default=None
        Override mapping; merged onto DEFAULT_LABEL_MAP
        (your values win).
    drop_quantile : bool, default=True
        Remove trailing "_qNN" / "_pNN" from the base.
    include_quantile : bool, default=False
        If True, append the quantile tag as "(qNN)" or
        "(pNN)". Useful for legend entries in fan plots.
    take_last : bool, default=True
        If True, use only last dotted segment.

    Returns
    -------
    label : str
        Example: "Subsidence (mm)" or
        "Hydraulic head (m)" or
        "Subsidence (mm) (q50)" if include_quantile.

    Examples
    --------
    >>> pretty_label("subsidence_q50", unit="mm")
    'Subsidence (mm)'
    >>> pretty_label("head_q10", unit="m", include_quantile=True)
    'Hydraulic head (m) (q10)'
    """
    raw = str(name or "").strip()
    if take_last and "." in raw:
        raw = raw.split(".")[-1].strip()

    base, qtag = split_quantile_suffix(raw)
    if not drop_quantile:
        base = raw

    lm: Dict[str, str] = dict(DEFAULT_LABEL_MAP)
    if label_map:
        lm.update(label_map)

    key = normalize_key(base)
    text = lm.get(key)
    if not text:
        text = humanize_name(base, take_last=False)

    u = str(unit or "").strip()
    if u:
        text = f"{text} ({u})"

    if include_quantile and qtag:
        text = f"{text} ({qtag})"

    return text


def pretty_band_label(
    lo: str,
    hi: str,
    *,
    prefix: str = "",
) -> str:
    """
    Build a compact band label like "q10..q90".

    Parameters
    ----------
    lo, hi : str
        Column names, possibly containing q/p tags.
    prefix : str, default=""
        Optional prefix like "PI " -> "PI q10..q90".

    Returns
    -------
    label : str
        A compact label for legends.
    """
    a = quantile_tag(lo) or str(lo or "")
    b = quantile_tag(hi) or str(hi or "")
    p = str(prefix or "").strip()
    if p:
        return f"{p} {a}..{b}"
    return f"{a}..{b}"

# from geoprior.utils import pretty_label

# label = pretty_label(z_col, unit="mm")
# canvas.set_layer(..., label=label)

# Analytics axes:
# ax.set_ylabel(pretty_label(z_col, unit=unit))
# ax.set_title(f"Colored by {pretty_label(z_col, unit=unit)}")

# Fan legend:
# from geoprior.utils import pretty_band_label

# lab = pretty_band_label(lo_q, hi_q, prefix="")


def open_json_editor(
    parent,
    *,
    title: str,
    path: Optional[str] = None,
    data: Optional[Any] = None,
    read_only: bool = True,
) -> Tuple[bool, Optional[Any]]:
    """
    Opens a pretty JSON viewer/editor dialog.

    Returns
    -------
    saved : bool
        True if user saved to disk.
    obj : Any or None
        Parsed JSON (possibly edited). None if cancelled.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFontDatabase
    from PyQt5.QtWidgets import (
        QDialog,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QVBoxLayout,
        QWidget,
    )

    obj = data
    fp = None if path is None else str(path)

    if obj is None and fp:
        try:
            obj = json.loads(Path(fp).read_text("utf-8"))
        except Exception as e:
            QMessageBox.warning(
                parent,
                title,
                f"Failed to read JSON:\n{e}",
            )
            return False, None

    txt = ""
    try:
        txt = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        txt = str(obj)

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setModal(True)

    root = QVBoxLayout(dlg)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(8)

    if fp:
        lbl = QLabel(f"<b>File:</b> {fp}", dlg)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setWordWrap(True)
        root.addWidget(lbl)

    editor = QPlainTextEdit(dlg)
    editor.setPlainText(txt)
    editor.setReadOnly(bool(read_only))
    editor.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    root.addWidget(editor, 1)

    bar = QWidget(dlg)
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)

    btn_toggle = QPushButton("Edit", dlg)
    btn_pretty = QPushButton("Pretty", dlg)
    btn_save = QPushButton("Save", dlg)
    btn_close = QPushButton("Close", dlg)

    btn_save.setEnabled((not read_only) and bool(fp))

    def _toggle():
        ro = editor.isReadOnly()
        editor.setReadOnly(not ro)
        btn_toggle.setText("Read" if ro else "Edit")
        btn_save.setEnabled((not editor.isReadOnly()) and bool(fp))

    def _pretty():
        try:
            o = json.loads(editor.toPlainText())
            editor.setPlainText(
                json.dumps(o, indent=2, ensure_ascii=False)
            )
        except Exception as e:
            QMessageBox.warning(dlg, title, f"Invalid JSON:\n{e}")

    def _save():
        if not fp:
            return
        try:
            o = json.loads(editor.toPlainText())
        except Exception as e:
            QMessageBox.warning(dlg, title, f"Invalid JSON:\n{e}")
            return
        try:
            Path(fp).write_text(
                json.dumps(o, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            QMessageBox.warning(dlg, title, f"Save failed:\n{e}")
            return
        dlg.done(1)

    btn_toggle.clicked.connect(_toggle)
    btn_pretty.clicked.connect(_pretty)
    btn_save.clicked.connect(_save)
    btn_close.clicked.connect(dlg.reject)

    lay.addWidget(btn_toggle)
    lay.addWidget(btn_pretty)
    lay.addStretch(1)
    lay.addWidget(btn_save)
    lay.addWidget(btn_close)
    root.addWidget(bar)

    code = dlg.exec_()
    if code == 1:
        try:
            return True, json.loads(editor.toPlainText())
        except Exception:
            return True, obj

    try:
        return False, json.loads(editor.toPlainText())
    except Exception:
        return False, obj
