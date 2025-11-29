# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Internal utilities for optional third-party dependencies.

This module centralizes feature flags such as ``HAS_TQDM`` so that
other modules can check availability of extra packages without
repeating try/except import blocks.
"""

from __future__ import annotations

from typing import Any, Optional, Iterable, TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------
# tqdm (progress bar)
# ---------------------------------------------------------------------
HAS_TQDM: bool
TQDM: Optional[Any]

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    HAS_TQDM = False
    _tqdm = None
else:
    HAS_TQDM = True

TQDM = _tqdm

def with_progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    ascii: bool = True,
    leave: bool = False,
    disable: Optional[bool] = None,
    **tqdm_kwargs: Any,
) -> Iterable[T]:
    """
    Wrap an iterable with tqdm if available, else return it unchanged.

    Parameters
    ----------
    iterable :
        Any iterable (e.g. Dataset, list, generator).
    total : int or None, optional
        Expected length for the progress bar.  If None, tries
        ``len(iterable)``; if that fails, tqdm will show an unknown
        total.
    desc : str or None, optional
        Progress bar description (left side label).
    ascii : bool, default=True
        Whether to force an ASCII progress bar (safer on some
        terminals).
    leave : bool, default=False
        Whether to leave the progress bar after completion.
    disable : bool or None, optional
        If True, always disable tqdm (return raw iterable).
        If None, uses tqdm if installed; otherwise falls back.
    **tqdm_kwargs :
        Any additional keyword arguments passed directly to tqdm.

    Returns
    -------
    iterable
        If tqdm is installed and not disabled, a tqdm-wrapped
        iterable; otherwise the original iterable.
    """
    # Explicitly disabled or tqdm not installed → raw iterable
    if disable is True or not HAS_TQDM or TQDM is None:
        return iterable

    try:
        # Try to infer total if not provided
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except:
                total = None

        return TQDM(
            iterable,
            total=total,
            desc=desc,
            ascii=ascii,
            leave=leave,
            **tqdm_kwargs,
        )
    except:
        # Any failure → graceful fallback to plain iterable
        return iterable


__all__ = [
    "HAS_TQDM",
    "TQDM",
    "with_progress",
]
