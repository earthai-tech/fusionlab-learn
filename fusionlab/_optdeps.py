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

from typing import Callable
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


class _TqdmLogStream:
    """
    Minimal file-like stream that forwards tqdm's text output
    to a logger function instead of the terminal.

    It buffers partial lines until a newline or carriage return
    to avoid spamming the logger for every tiny write.
    """

    def __init__(self, log_fn: Callable[[str], None]) -> None:
        self._log_fn = log_fn
        self._buf = ""

    def write(self, s: str) -> int:
        # tqdm sometimes writes bytes or weird objects; be defensive
        s = str(s)
        if not s:
            return 0

        self._buf += s

        # tqdm uses '\r' a lot; treat both '\n' and '\r' as flush points
        for sep in ("\n", "\r"):
            if sep in self._buf:
                parts = self._buf.split(sep)
                # all but last are complete "lines"
                for line in parts[:-1]:
                    line = line.strip()
                    if line:
                        self._log_fn(line)
                # keep the trailing incomplete fragment (if any)
                self._buf = parts[-1]

        return len(s)

    def flush(self) -> None:
        # Flush whatever is left in the buffer as a final line
        if self._buf.strip():
            self._log_fn(self._buf.strip())
        self._buf = ""


def with_progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    ascii: bool = True,
    leave: bool = False,
    disable: Optional[bool] = None,
    log_fn: Optional[Callable[[str], None]] = None,
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
    log_fn : callable or None, optional
        If provided, tqdm's progress bar output is redirected into
        this logger instead of the terminal. The logger must accept
        a single string argument.
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
            except Exception:
                total = None

        tqdm_args = {
            "total": total,
            "desc": desc,
            "ascii": ascii,
            "leave": leave,
        }
        tqdm_args.update(tqdm_kwargs)

        # If a logger is provided, redirect tqdm output to it
        if log_fn is not None:
            tqdm_args["file"] = _TqdmLogStream(log_fn)

        return TQDM(
            iterable,
            **tqdm_args,
        )
    except Exception:
        # Any failure → graceful fallback to plain iterable
        return iterable



__all__ = [
    "HAS_TQDM",
    "TQDM",
    "with_progress",
]
