# geoprior/ui/utils/path_display.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Optional


ELL = "…"


def _pure_path(s: str):
    if "\\" in s or (len(s) > 1 and s[1] == ":"):
        return PureWindowsPath(s)
    return PurePosixPath(s)


def compact_path(
    path: Optional[str],
    *,
    max_len: int = 62,
    keep_last: int = 2,
) -> str:
    if not path:
        return "-"
    s = str(path)
    if len(s) <= max_len:
        return s

    p = _pure_path(s)
    parts = list(p.parts)
    if not parts:
        return s[: max_len - 1] + ELL

    head = parts[0]
    tail = parts[-(keep_last + 1) :]

    cls = type(p)
    out = str(cls(head, ELL, *tail))
    if len(out) <= max_len:
        return out

    out = str(cls(ELL, *tail[-2:]))
    if len(out) <= max_len:
        return out

    return f"{ELL}{p.anchor}{p.name}" if p.name else out[:max_len]


def set_path_label(
    lbl,
    *,
    prefix: str,
    full_path: Optional[str],
    max_len: int = 62,
) -> None:
    full = "" if full_path is None else str(full_path)
    disp = compact_path(full, max_len=max_len)
    lbl.setText(f"{prefix}{disp}")
    if full:
        lbl.setToolTip(full)
    else:
        lbl.setToolTip("")
