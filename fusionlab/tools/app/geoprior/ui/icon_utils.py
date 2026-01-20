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
def try_icon(name: str) -> Optional[QIcon]:
    """
    Resolve an icon from the GeoPrior icons folder.

    This searches for: <project_root>/geoprior/icons/<name>

    Parameters
    ----------
    name
        File name like "metric_dashboard.svg".

    Returns
    -------
    QIcon or None
        QIcon if found, else None.
    """
    nm = (name or "").strip()
    if not nm:
        return None

    here = Path(__file__).resolve()
    root = _find_project_root(here, markers=("geoprior",), max_up=14)
    if root is None:
        return None

    p = root / "geoprior" / "icons" / nm
    if not p.exists():
        return None

    return QIcon(str(p))


@lru_cache(maxsize=256)
def try_icon_path(name: str) -> Optional[Path]:
    """
    Like try_icon(), but returns the resolved Path instead of QIcon.
    Useful if you need to pass the path to other APIs.
    """
    nm = (name or "").strip()
    if not nm:
        return None

    here = Path(__file__).resolve()
    root = _find_project_root(here, markers=("geoprior",), max_up=14)
    if root is None:
        return None

    p = root / "geoprior" / "icons" / nm
    if p.exists():
        return p
    return None
