# geoprior/ui/icon_utils.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.icon_utils

Small helpers for resolving GeoPrior SVG icons in a robust way.

Design goals
------------
- No hard-coded absolute paths.
- Works even if modules are moved within geoprior/ui/.
- Fast (small cache).
- Safe fallback (returns None if not found).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from PyQt5.QtGui import QIcon


def _find_project_root(
    start: Path,
    *,
    markers: Sequence[str] = ("geoprior",),
    max_up: int = 12,
) -> Optional[Path]:
    """
    Walk up the filesystem until a folder containing `markers` exists.

    Parameters
    ----------
    start
        Starting file path (typically __file__).
    markers
        Folder names expected to exist at the project root.
    max_up
        Maximum upward steps.

    Returns
    -------
    Path or None
        Root path if detected, else None.
    """
    p = start.resolve()
    for _ in range(max_up):
        if all((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent
    return None


@lru_cache(maxsize=256)
def try_icon_path(name: str) -> Optional[Path]:
    """Resolve an icon file path from the GeoPrior icons folder."""
    nm = (name or "").strip()
    if not nm:
        return None

    here = Path(__file__).resolve()
    root = _find_project_root(
        here,
        markers=("geoprior",),
        max_up=14,
    )
    if root is None:
        return None

    p = root / "geoprior" / "icons" / nm
    if p.exists():
        return p
    return None


def try_icon(
    name: str,
    *,
    fallback: Optional[QIcon] = None,
    size: Optional[int] = None,
) -> QIcon:
    """Resolve an icon with a safe fallback.

    Parameters
    ----------
    name
        File name like "analytics.svg".
    fallback
        Icon returned when the file does not exist.
    size
        Optional render hint (kept for compatibility).
        QIcon handles SVG scaling, so this is not
        required. It is accepted to keep a stable API.

    Returns
    -------
    QIcon
        A valid icon object (may be null).
    """
    _ = size  # kept for call-site compatibility

    p = try_icon_path(name)
    if p is not None:
        ico = QIcon(str(p))
        if not ico.isNull():
            return ico

    if fallback is None:
        return QIcon()
    return fallback


