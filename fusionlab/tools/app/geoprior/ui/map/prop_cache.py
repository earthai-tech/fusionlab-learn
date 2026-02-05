# -*- coding: utf-8 -*-
# geoprior/ui/map/prop_cache.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .prop_utils import (
    extrapolate_scenarios,
    compute_propagation_vectors,
)

def _to_int_year(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.round().astype("Int64")

@dataclass
class PropScenarioInfo:
    timeline: List[int]
    vmin: Optional[float]
    vmax: Optional[float]

class PropagationScenarioCache:
    """
    Immutable propagation scenario cache.

    Schema:
      lon, lat, v, t, sample_idx, _is_simulated
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._df = pd.DataFrame()
        self._frames: Dict[int, pd.DataFrame] = {}
        self._vectors: Dict[int, List[dict]] = {}
        self._timeline: List[int] = []
        self._vmin = None
        self._vmax = None

    @property
    def ready(self) -> bool:
        return (self._df is not None) and (not self._df.empty)

    @property
    def timeline(self) -> List[int]:
        return list(self._timeline)

    def build(
        self,
        base_df: pd.DataFrame,
        *,
        years_to_add: int,
        time_col: str = "t",
        value_col: str = "v",
        mode: str = "absolute",
        legend_policy: str = "global",
    ) -> PropScenarioInfo:

        self.clear()
        if base_df is None or base_df.empty:
            return PropScenarioInfo([], None, None)

        if "_is_simulated" in base_df.columns:
            base_df = base_df[base_df["_is_simulated"] != True].copy()

        df0 = base_df.copy()
        df0["t"] = _to_int_year(df0[time_col])
        df0["v"] = pd.to_numeric(df0[value_col], errors="coerce")
        df0 = df0.dropna(subset=["lon", "lat", "t"])

        if df0.empty:
            return PropScenarioInfo([], None, None)

        id_cols = ["sample_idx"] if "sample_idx" in df0.columns else ["lon", "lat"]

        df1 = extrapolate_scenarios(
            df0,
            years_to_add=int(years_to_add),
            time_col="t",
            id_cols=id_cols,
            value_col="v",
        )

        df1["t"] = _to_int_year(df1["t"])
        df1["v"] = pd.to_numeric(df1["v"], errors="coerce")
        df1 = df1.dropna(subset=["lon", "lat", "t"])

        self._df = df1

        yrs = pd.Index(df1["t"].dropna().astype(int).unique()).sort_values()
        self._timeline = [int(x) for x in yrs]

        vv = df1["v"].dropna()
        if not vv.empty:
            self._vmin = float(vv.min())
            self._vmax = float(vv.max())

        self._frames = {
            int(y): g.copy()
            for y, g in df1.groupby("t", sort=False)
            if pd.notna(y)
        }

        return PropScenarioInfo(self.timeline, self._vmin, self._vmax)

    def frame(self, year: int) -> pd.DataFrame:
        if not self.ready:
            return pd.DataFrame()
        return self._frames.get(int(year), pd.DataFrame())

    def legend_range(
        self,
        *,
        year: Optional[int] = None,
        policy: str = "global",
    ) -> Tuple[Optional[float], Optional[float]]:

        p = str(policy or "global").lower()
        if p == "global":
            return self._vmin, self._vmax

        if p == "frame" and year is not None:
            fr = self.frame(int(year))
            if fr.empty:
                return None, None
            vv = fr["v"].dropna()
            if vv.empty:
                return None, None
            return float(vv.min()), float(vv.max())

        return None, None

    def vectors(self, year: int) -> List[dict]:
        y = int(year)
        if y in self._vectors:
            return list(self._vectors[y])

        fr = self.frame(y)
        if fr.empty:
            self._vectors[y] = []
            return []

        vec = compute_propagation_vectors(
            fr,
            time_col="t",
            value_col="v",
        )
        self._vectors[y] = list(vec or [])
        return list(self._vectors[y])
