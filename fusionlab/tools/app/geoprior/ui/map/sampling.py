# geoprior/ui/map/sampling.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.sampling

Deterministic spatial sampling for visualization.

Goal
----
Use one config (store-backed) so Map points, analytics,
and hotspots all render the same sampled subset.

Config keys
-----------
map.sampling.mode            : "auto"|"off"|"always"
map.sampling.method          : "grid"|"random"
map.sampling.max_points      : int
map.sampling.seed            : int
map.sampling.cell_km         : float
map.sampling.max_per_cell    : int
map.sampling.apply_hotspots  : bool
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import pandas as pd


@dataclass
class SamplingCfg:
    mode: str = "auto"
    method: str = "grid"
    max_points: int = 80000
    seed: int = 0
    cell_km: float = 1.0
    max_per_cell: int = 50
    apply_hotspots: bool = True


def cfg_from_get(get: Callable) -> SamplingCfg:
    mode = str(get("map.sampling.mode", "auto") or "auto")
    method = str(get("map.sampling.method", "grid") or "grid")

    try:
        max_points = int(get("map.sampling.max_points", 80000))
    except Exception:
        max_points = 80000

    try:
        seed = int(get("map.sampling.seed", 0))
    except Exception:
        seed = 0

    try:
        cell_km = float(get("map.sampling.cell_km", 1.0))
    except Exception:
        cell_km = 1.0

    try:
        max_per = int(get("map.sampling.max_per_cell", 50))
    except Exception:
        max_per = 50

    hot = bool(get("map.sampling.apply_hotspots", True))

    mode = mode.strip().lower()
    method = method.strip().lower()

    if mode not in ("auto", "off", "always"):
        mode = "auto"
    if method not in ("grid", "random"):
        method = "grid"

    max_points = max(1, int(max_points))
    max_per = max(1, int(max_per))
    cell_km = max(0.001, float(cell_km))

    return SamplingCfg(
        mode=mode,
        method=method,
        max_points=max_points,
        seed=seed,
        cell_km=cell_km,
        max_per_cell=max_per,
        apply_hotspots=hot,
    )


def _needs_sampling(n: int, cfg: SamplingCfg) -> bool:
    if cfg.mode == "off":
        return False
    if cfg.mode == "always":
        return True
    return int(n) > int(cfg.max_points)


def sample_points(
    df: Optional[pd.DataFrame],
    cfg: SamplingCfg,
    *,
    lon: str = "lon",
    lat: str = "lat",
) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    if df.empty:
        return df

    n = int(len(df))
    if not _needs_sampling(n, cfg):
        return df

    m = min(int(cfg.max_points), n)

    if cfg.method == "random":
        return df.sample(n=m, random_state=int(cfg.seed))

    return _grid_sample(
        df,
        cfg=cfg,
        lon=lon,
        lat=lat,
        max_n=m,
    )


def _grid_sample(
    df: pd.DataFrame,
    *,
    cfg: SamplingCfg,
    lon: str,
    lat: str,
    max_n: int,
) -> pd.DataFrame:
    if (lon not in df.columns) or (lat not in df.columns):
        return df.sample(n=max_n, random_state=int(cfg.seed))

    d = df[[lon, lat]].copy()
    d = d.dropna()
    if d.empty:
        return df.sample(n=max_n, random_state=int(cfg.seed))

    lat0 = float(d[lat].median())
    cos0 = math.cos(math.radians(lat0))
    cos0 = max(1e-6, abs(cos0))

    dlat = float(cfg.cell_km) / 110.574
    dlon = float(cfg.cell_km) / (111.320 * cos0)

    lat_min = float(d[lat].min())
    lon_min = float(d[lon].min())

    gi = ((d[lat] - lat_min) / dlat).astype("int64")
    gj = ((d[lon] - lon_min) / dlon).astype("int64")

    g = d.groupby([gi, gj], sort=False)
    n_cells = int(g.ngroups) if g.ngroups else 1

    per = int(math.ceil(float(max_n) / float(n_cells)))
    per = min(int(cfg.max_per_cell), max(1, per))

    parts = []
    for idx, (_k, gdf) in enumerate(g):
        if len(gdf) <= per:
            parts.append(df.loc[gdf.index])
            continue

        rs = int(cfg.seed) + (idx * 9973)
        parts.append(df.loc[gdf.index].sample(
            n=per,
            random_state=rs,
        ))

    out = pd.concat(parts, axis=0)
    if len(out) > max_n:
        out = out.sample(n=max_n, random_state=int(cfg.seed))
    return out
