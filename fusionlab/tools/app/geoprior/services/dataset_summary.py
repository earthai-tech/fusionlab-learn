# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""Small dataset summary helpers for the GeoPrior GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .column_mapping import ColumnRoleMapper


def build_dataset_summary_text(
    df: pd.DataFrame,
    *,
    mapper: ColumnRoleMapper,
    city: str = "",
    csv_path: Optional[Path] = None,
    include_location: bool = False,
    include_path: bool = False,
    include_basic: bool = True,
) -> str:
    """Build a short, UI-friendly dataset summary."""
    lines: list[str] = []

    if include_location and city:
        lines.append(f"City: {city}")
    if include_path and csv_path is not None:
        lines.append(f"Path: {csv_path}")

    if include_basic:
        lines.append(f"Rows/Cols: {len(df)} / {df.shape[1]}")
        try:
            n_num = df.select_dtypes(include=[np.number]).shape[1]
        except Exception:
            n_num = 0
        n_obj = int(df.shape[1]) - int(n_num)
        lines.append(f"Columns: {n_num} numeric, {n_obj} other")

    time_col = mapper.column_for("time")
    if time_col and time_col in df.columns:
        y0 = _safe_min(df[time_col])
        y1 = _safe_max(df[time_col])
        if y0 is not None and y1 is not None:
            lines.append(f"Time range: {y0} .. {y1}")

    lon_col = mapper.column_for("lon")
    lat_col = mapper.column_for("lat")
    if (
        lon_col
        and lat_col
        and lon_col in df.columns
        and lat_col in df.columns
    ):
        lon0 = _safe_min(df[lon_col])
        lon1 = _safe_max(df[lon_col])
        lat0 = _safe_min(df[lat_col])
        lat1 = _safe_max(df[lat_col])
        if None not in (lon0, lon1, lat0, lat1):
            lines.append(
                "BBox: "
                f"lon[{lon0}, {lon1}], "
                f"lat[{lat0}, {lat1}]"
            )

    subs_col = mapper.column_for("subs")
    if subs_col and subs_col in df.columns:
        s0 = _safe_min(df[subs_col])
        s1 = _safe_max(df[subs_col])
        if s0 is not None and s1 is not None:
            lines.append(f"Subsidence range: {s0} .. {s1}")

    return "\n".join(lines)


def _safe_min(s: pd.Series):
    try:
        v = s.min(skipna=True)
    except Exception:
        return None
    return None if pd.isna(v) else v


def _safe_max(s: pd.Series):
    try:
        v = s.max(skipna=True)
    except Exception:
        return None
    return None if pd.isna(v) else v
