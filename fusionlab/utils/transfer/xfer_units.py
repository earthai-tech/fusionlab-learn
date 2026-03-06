
from __future__ import annotations

from typing import Sequence
import pandas as pd


__all__= [ 
    
   "unit_factor",
    "infer_unit",
    "scale_cols", 
    ]

_FACT = {
    ("m", "m"): 1.0,
    ("mm", "mm"): 1.0,
    ("m", "mm"): 1000.0,
    ("mm", "m"): 1.0 / 1000.0,
}

def unit_factor(from_u: str, to_u: str) -> float:
    fu = str(from_u).strip().lower()
    tu = str(to_u).strip().lower()
    if (fu, tu) in _FACT:
        return float(_FACT[(fu, tu)])
    if fu == tu:
        return 1.0
    raise ValueError(f"Unsupported unit: {fu} -> {tu}")

def infer_unit(
    df: pd.DataFrame,
    *,
    col: str = "subsidence_unit",
    default: str = "mm",
) -> str:
    if df is None or col not in df.columns:
        return default
    s = df[col].dropna()
    if s.empty:
        return default
    u0 = str(s.astype(str).iloc[0]).strip().lower()
    if u0.startswith("mm"):
        return "mm"
    if u0.startswith("m"):
        return "m"
    return default

def scale_cols(
    df: pd.DataFrame,
    cols: Sequence[str],
    factor: float,
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (
                pd.to_numeric(out[c], errors="coerce") * factor
            )
    return out