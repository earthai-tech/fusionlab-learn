# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
fusionlab.utils.panel_cache

Small helpers to build *stable* group keys (pixels) and
filter "feasible" groups for sequence windowing.

Why this exists
---------------
When a pixel has multiple sliding windows (e.g., Zhongshan
2 windows/pixel), splitting *after* windowing leaks pixel
information across splits. The fix is:

  split by pixel key first, then build windows inside split.

This module helps you:
- create a stable pixel key from lon/lat (or any columns)
- compute which pixel keys have enough time points
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


__all__ = [
    "min_required_len",
    "make_group_keys",
    "feasible_group_keys",
    "PanelKeyCache",
]


def min_required_len(time_steps: int, horizon: int) -> int:
    """
    Minimum number of distinct time points needed for a
    single (T, H) sample.

    Parameters
    ----------
    time_steps : int
        Input lookback length T.
    horizon : int
        Forecast horizon H.

    Returns
    -------
    int
        T + H.
    """
    return int(time_steps) + int(horizon)


def _col_to_key_str(
    s: pd.Series,
    *,
    decimals: int,
) -> pd.Series:
    """
    Convert a column to a stable string representation.

    Numeric columns are rounded to `decimals` and formatted
    with fixed decimals to avoid float drift.
    """
    if pd.api.types.is_numeric_dtype(s):
        a = pd.to_numeric(s, errors="coerce").astype(float)
        a = np.round(a.to_numpy(), decimals=int(decimals))
        fmt = "%." + str(int(decimals)) + "f"
        out = pd.Series(
            np.char.mod(fmt, a),
            index=s.index,
            name=s.name,
        )
        return out

    return s.astype(str).fillna("")


def make_group_keys(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    decimals: int = 8,
    key_name: str = "__gid__",
) -> pd.Series:
    """
    Build stable string keys for each row's group id.

    Parameters
    ----------
    df : pandas.DataFrame
        Source table.
    group_cols : sequence of str
        Columns defining the group (e.g., lon/lat).
    decimals : int, default=8
        Rounding for numeric columns.
    key_name : str, default="__gid__"
        Name of the returned Series.

    Returns
    -------
    pandas.Series
        Key per row, like "113.12345600|22.98765400".
    """
    miss = [c for c in group_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing group_cols: {miss}")

    parts = []
    for c in group_cols:
        parts.append(
            _col_to_key_str(df[c], decimals=decimals)
        )

    g = pd.concat(parts, axis=1)
    k = g.astype(str).agg("|".join, axis=1)
    k.name = key_name
    return k


def feasible_group_keys(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    time_col: str,
    min_len: int,
    decimals: int = 8,
) -> np.ndarray:
    """
    Return group keys with >= `min_len` distinct times.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data (typically your training subset).
    group_cols : sequence of str
        Group id columns (pixel id).
    time_col : str
        Time column.
    min_len : int
        Minimum distinct time points required.
    decimals : int, default=8
        Rounding for numeric group cols.

    Returns
    -------
    numpy.ndarray
        Array of feasible keys (dtype object).
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing time_col: {time_col}")

    keys = make_group_keys(
        df,
        group_cols=group_cols,
        decimals=decimals,
    )
    tmp = pd.DataFrame(
        {"__k__": keys, "__t__": df[time_col]}
    )
    tmp = tmp.drop_duplicates(["__k__", "__t__"])
    cnt = tmp.groupby("__k__")["__t__"].nunique()
    ok = cnt[cnt >= int(min_len)].index.to_numpy()
    return ok


@dataclass
class PanelKeyCache:
    """
    Cache feasibility counts per pixel key.

    Use this if you call feasible filtering repeatedly.

    Notes
    -----
    The cache stores counts of distinct times per key.
    You can query feasible keys for any `min_len`.
    """
    group_cols: Sequence[str]
    time_col: str
    decimals: int = 8

    _counts: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "PanelKeyCache":
        """
        Compute distinct-time counts per group key.
        """
        keys = make_group_keys(
            df,
            group_cols=self.group_cols,
            decimals=self.decimals,
        )
        tmp = pd.DataFrame(
            {"__k__": keys, "__t__": df[self.time_col]}
        )
        tmp = tmp.drop_duplicates(["__k__", "__t__"])
        self._counts = tmp.groupby("__k__")["__t__"].nunique()
        return self

    def feasible(self, min_len: int) -> np.ndarray:
        """
        Return feasible keys from cached counts.
        """
        if self._counts is None:
            raise RuntimeError("Call fit(df) first.")
        ok = self._counts[self._counts >= int(min_len)]
        return ok.index.to_numpy()
