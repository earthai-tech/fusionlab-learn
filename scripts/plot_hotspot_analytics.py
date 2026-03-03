# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# scripts/plot_hotspot_analytics.py
#
# Hotspot analytics for decision makers (Nature-friendly figure).
#
# Produces a 2x4 (cities x panels) figure:
#   (1) Hotspot anomaly map |Δs| vs base-year
#   (2) Risk probability map P(|s| >= threshold)
#   (3) Hotspot timeline (n, T0.9, mean/max |Δs|)
#   (4) Priority list (top clusters)
#
# Also exports:
#   - scripts/out/hotspot_points.csv
#   - scripts/out/hotspot_years.csv
#   - scripts/out/hotspot_clusters.csv

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as cfg
from . import utils


_CITY_A = cfg.CITY_CANON.get("ns", "Nansha")
_CITY_B = cfg.CITY_CANON.get("zh", "Zhongshan")


GeoDF = Any

def _slug_city(city: str) -> str:
    return str(city).strip().lower().replace(" ", "_")


# ---------------------------------------------------------------------
# Exceedance probability from quantiles (q10/q50/q90)
# (same logic as fusionlab.utils.transfer.xfer_risk)
# ---------------------------------------------------------------------
def exceed_prob_from_quantiles(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    qs = np.stack([q10, q50, q90], axis=1).astype(float)
    ps0 = np.array([0.1, 0.5, 0.9], dtype=float)

    ok = np.isfinite(qs).all(axis=1)
    out = np.full(qs.shape[0], np.nan, dtype=float)
    if not np.any(ok):
        return out

    qs_ok = qs[ok]
    order = np.argsort(qs_ok, axis=1)
    qs_s = np.take_along_axis(qs_ok, order, axis=1)
    ps_s = np.take(ps0, order)

    T = float(threshold)
    F = np.empty(qs_s.shape[0], dtype=float)

    lo = T <= qs_s[:, 0]
    hi = T >= qs_s[:, 2]
    mid = ~(lo | hi)

    F[lo] = ps_s[lo, 0]
    F[hi] = ps_s[hi, 2]

    if np.any(mid):
        q1 = qs_s[mid, 1]
        seg = (T > q1).astype(int)
        ii = np.arange(seg.size)

        x0 = qs_s[mid][ii, seg]
        x1 = qs_s[mid][ii, seg + 1]
        p0 = ps_s[mid][ii, seg]
        p1 = ps_s[mid][ii, seg + 1]

        den = x1 - x0
        same = den == 0.0

        Fm = np.empty(seg.size, dtype=float)
        Fm[same] = p1[same]
        Fm[~same] = p0[~same] + (
            (T - x0[~same]) / den[~same]
        ) * (p1[~same] - p0[~same])

        F[mid] = Fm

    out[ok] = 1.0 - F
    return out


# ---------------------------------------------------------------------
# IO + canonization
# ---------------------------------------------------------------------
def _require_cols(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    where: str,
) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{where}: missing {miss}")


_ALIASES = dict(cfg._BASE_ALIASES)
_ALIASES.update(
    {
        "subsidence_q10": ("subsidence_q10", "q10", "p10"),
        "subsidence_q90": ("subsidence_q90", "q90", "p90"),
    }
)


def _canonize(
    df: pd.DataFrame,
    *,
    where: str,
    required: Sequence[str],
) -> pd.DataFrame:
    utils.ensure_columns(df, aliases=_ALIASES)
    _require_cols(df, list(required), where=where)
    return df

# def _load_exposure(
#     path: str,
#     *,
#     col: str,
# ) -> pd.DataFrame:
#     p = utils.as_path(path)
#     df = pd.read_csv(p)

#     utils.ensure_columns(
#         df,
#         aliases={
#             "sample_idx": ("sample_idx", "sample_id"),
#             col: (col,),
#         },
#     )

#     if "sample_idx" not in df.columns:
#         raise KeyError("exposure: missing sample_idx")

#     if col not in df.columns:
#         raise KeyError(f"exposure: missing {col}")

#     df["sample_idx"] = pd.to_numeric(
#         df["sample_idx"],
#         errors="coerce",
#     )
#     df[col] = pd.to_numeric(
#         df[col],
#         errors="coerce",
#     )

#     df = df.dropna(subset=["sample_idx", col]).copy()
#     df["sample_idx"] = df["sample_idx"].astype(int)
#     return df[["sample_idx", col]].copy()

def _load_exposure(path: str, *, col: str) -> pd.DataFrame:
    p = utils.as_path(path)
    df = pd.read_csv(p)

    utils.ensure_columns(
        df,
        aliases={"sample_idx": ("sample_idx", "sample_id"), col: (col,)},
    )

    if "sample_idx" not in df.columns:
        raise KeyError("exposure: missing sample_idx")
    if col not in df.columns:
        raise KeyError(f"exposure: missing {col}")

    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce")
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["sample_idx", col]).copy()
    df["sample_idx"] = df["sample_idx"].astype(int)

    # CRITICAL: avoid duplicates that explode rows after merge
    df = (
        df.groupby("sample_idx", as_index=False)[col]
        .mean()
        .copy()
    )
    return df[["sample_idx", col]]

def _attach_exposure(
    pts: pd.DataFrame,
    expo: Optional[pd.DataFrame],
    *,
    col: str,
    default: float,
) -> pd.DataFrame:
    out = pts.copy()
    if expo is None or expo.empty:
        out["exposure"] = float(default)
        return out

    e = expo.rename(columns={col: "exposure"}).copy()
    out = out.merge(e, on="sample_idx", how="left")
    out["exposure"] = out["exposure"].fillna(float(default))
    return out


def _load_eval_df(path: str) -> pd.DataFrame:
    p = utils.as_path(path)
    df = pd.read_csv(p)

    _canonize(
        df,
        where="eval_calibrated",
        required=(
            "sample_idx",
            "coord_t",
            "coord_x",
            "coord_y",
            "subsidence_actual",
            "subsidence_q50",
        ),
    )

    for c in (
        "sample_idx",
        "coord_t",
        "coord_x",
        "coord_y",
        "subsidence_actual",
        "subsidence_q50",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("subsidence_q10", "subsidence_q90"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(
        subset=(
            "sample_idx",
            "coord_t",
            "coord_x",
            "coord_y",
            "subsidence_actual",
            "subsidence_q50",
        )
    ).copy()


def _load_future_df(path: str) -> pd.DataFrame:
    p = utils.as_path(path)
    df = pd.read_csv(p)

    _canonize(
        df,
        where="future_forecast",
        required=(
            "sample_idx",
            "coord_t",
            "coord_x",
            "coord_y",
            "subsidence_q10",
            "subsidence_q50",
            "subsidence_q90",
        ),
    )

    for c in (
        "sample_idx",
        "coord_t",
        "coord_x",
        "coord_y",
        "subsidence_q10",
        "subsidence_q50",
        "subsidence_q90",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(
        subset=(
            "sample_idx",
            "coord_t",
            "coord_x",
            "coord_y",
            "subsidence_q10",
            "subsidence_q50",
            "subsidence_q90",
        )
    ).copy()


def _pick_eval_path(
    art: utils.Artifacts,
    split: str,
) -> Tuple[Optional[Path], str]:
    if split == "val":
        return (art.forecast_val_csv, "val")
    if split == "test":
        return (art.forecast_test_csv, "test")
    if art.forecast_test_csv is not None:
        return (art.forecast_test_csv, "test")
    return (art.forecast_val_csv, "val")


def _pick_future_path(
    art: utils.Artifacts,
    split: str,
) -> Tuple[Optional[Path], str]:
    if split == "val":
        return (art.forecast_future_csv, "val")
    if split == "test":
        return (art.forecast_test_future_csv, "test")
    if art.forecast_test_future_csv is not None:
        return (art.forecast_test_future_csv, "test")
    return (art.forecast_future_csv, "val")


def _resolve_city(
    *,
    city: str,
    src: Optional[str],
    eval_csv: Optional[str],
    future_csv: Optional[str],
    split: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": city}
    out["color"] = cfg.CITY_COLORS.get(city, "#333333")

    if eval_csv and future_csv:
        out["eval_df"] = _load_eval_df(eval_csv)
        out["future_df"] = _load_future_df(future_csv)
        out["split"] = split
        out["src_note"] = "manual"
        return out

    if not src:
        k = city[:2].lower()
        raise ValueError(
            f"{city}: provide --{k}-src or both "
            f"--{k}-eval and --{k}-future"
        )

    art = utils.detect_artifacts(src)
    e_p, e_lab = _pick_eval_path(art, split)
    f_p, f_lab = _pick_future_path(art, split)

    if e_p is None:
        raise FileNotFoundError(
            f"{city}: no eval calibrated CSV under {src}"
        )
    if f_p is None:
        raise FileNotFoundError(
            f"{city}: no future CSV under {src}"
        )

    out["eval_df"] = _load_eval_df(str(e_p))
    out["future_df"] = _load_future_df(str(f_p))
    out["split"] = e_lab
    out["src_note"] = f"{e_p.name} + {f_p.name} ({e_lab}/{f_lab})"
    return out

def _try_load_boundaries(path: Optional[str]):
    if not path:
        return None
    try:
        import geopandas as gpd
    except Exception:
        print("[WARN] geopandas missing; skip boundary.")
        return None

    try:
        gdf = gpd.read_file(utils.as_path(path))
        return gdf
    except Exception as e:
        print(f"[WARN] boundary load failed: {e}")
        return None

def _filter_boundary_for_city(
    gdf: GeoDF,
    *,
    city: str,
    name_field: Optional[str] = None,
) -> GeoDF:
    """
    Try to filter a boundary GeoDataFrame to the requested city.

    Rules:
    - If name_field is provided and exists, use it.
    - Else if column 'city' exists, use it.
    - Else return gdf unchanged.
    """
    if gdf is None:
        return None

    try:
        cols = set(getattr(gdf, "columns", []))
    except:
        return gdf

    key = None
    if name_field and name_field in cols:
        key = str(name_field)
    elif "city" in cols:
        key = "city"

    if not key:
        return gdf

    try:
        s = gdf[key].astype(str).str.strip().str.lower()
        want0 = str(city).strip().lower()
        want1 = _slug_city(city)
        m = (s == want0) | (s.str.replace(" ", "_") == want1)
        sub = gdf.loc[m].copy()
        return sub if not sub.empty else gdf
    except:
        return gdf

def _resolve_boundary_map(
    *,
    boundary: Optional[str],
    ns_boundary: Optional[str],
    zh_boundary: Optional[str],
    boundary_name_field: Optional[str],
    cities: Sequence[str],
) -> Optional[Dict[str, GeoDF]]:
    """
    Return a per-city boundary mapping {city: GeoDataFrame}.
    """
    out: Dict[str, GeoDF] = {}

    g_ns = _try_load_boundaries(ns_boundary)
    if g_ns is not None:
        out[_CITY_A] = g_ns

    g_zh = _try_load_boundaries(zh_boundary)
    if g_zh is not None:
        out[_CITY_B] = g_zh

    g0 = _try_load_boundaries(boundary)
    if g0 is not None:
        for c in cities:
            if c in out:
                continue
            out[c] = _filter_boundary_for_city(
                g0,
                city=str(c),
                name_field=boundary_name_field,
            )

    # drop Nones
    out = {k: v for k, v in out.items() if v is not None}
    return out if out else None
# ---------------------------------------------------------------------
# Annual conversion (subsidence-kind aware)
# ---------------------------------------------------------------------
def _annual_from_cumulative(
    df: pd.DataFrame,
    *,
    value_col: str,
    year_col: str = "coord_t",
    id_col: str = "sample_idx",
) -> pd.Series:
    d = df[[id_col, year_col, value_col]].copy()
    orig_idx = d.index

    d[year_col] = pd.to_numeric(d[year_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[id_col, year_col, value_col])
    d[year_col] = d[year_col].astype(int)

    d = d.sort_values([id_col, year_col])
    prev = d.groupby(id_col)[value_col].shift(1)
    out = d[value_col] - prev

    # If first year has no previous, treat previous as 0 (rebased).
    out = out.fillna(d[value_col])
    out.index = d.index

    return out.reindex(orig_idx)


def _annual_series(
    df: pd.DataFrame,
    *,
    value_col: str,
    kind: str,
) -> pd.Series:
    k = str(kind).strip().lower()
    if k in ("rate", "increment"):
        return pd.to_numeric(
            df[value_col], errors="coerce"
        ).astype(float)
    return _annual_from_cumulative(df, value_col=value_col)


# ---------------------------------------------------------------------
# Raster + clustering
# ---------------------------------------------------------------------
def _extent(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    x = pd.to_numeric(df["coord_x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["coord_y"], errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return (0.0, 1.0, 0.0, 1.0)
    return (
        float(np.nanmin(x)),
        float(np.nanmax(x)),
        float(np.nanmin(y)),
        float(np.nanmax(y)),
    )


def _grid_mean(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    *,
    res: int,
    extent: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    xmin, xmax, ymin, ymax = extent
    xb = np.linspace(xmin, xmax, int(res) + 1)
    yb = np.linspace(ymin, ymax, int(res) + 1)

    sumv, _, _ = np.histogram2d(x, y, bins=[xb, yb], weights=v)
    cnts, _, _ = np.histogram2d(x, y, bins=[xb, yb])

    with np.errstate(invalid="ignore", divide="ignore"):
        z = sumv / cnts
    z[cnts == 0] = np.nan
    return (z.T, extent)

def _grid_hotspot_mask(
    z_d: np.ndarray,
    z_p: np.ndarray,
    *,
    thr: float,
    hotspot_rule: str,
    hotspot_abs: Optional[float],
    risk_min: Optional[float],
) -> np.ndarray:
    r = str(hotspot_rule).strip().lower()

    if r == "abs" and hotspot_abs is not None:
        return np.isfinite(z_d) & (z_d >= float(hotspot_abs))

    if r == "risk" and risk_min is not None:
        return np.isfinite(z_p) & (z_p >= float(risk_min))

    if r == "combo":
        m1 = (
            np.isfinite(z_d) & (z_d >= float(hotspot_abs))
        ) if hotspot_abs is not None else np.zeros_like(z_d, bool)
        m2 = (
            np.isfinite(z_p) & (z_p >= float(risk_min))
        ) if risk_min is not None else np.zeros_like(z_p, bool)
        m = m1 | m2
        if np.any(m):
            return m
        # fallback if both thresholds missing
    # percentile fallback
    return np.isfinite(z_d) & (z_d >= float(thr))

def _compute_persistence_grid(
    *,
    dfp: pd.DataFrame,
    dfy: pd.DataFrame,
    years: Sequence[int],
    extent: Tuple[float, float, float, float],
    grid_res: int,
    hotspot_rule: str,
    hotspot_abs: Optional[float],
    risk_min: Optional[float],
    mode: str,
) -> Tuple[np.ndarray, Tuple[float, float, float, float], int]:
    """
    Compute per-cell hotspot persistence across years.

    mode:
      - fraction: count / valid_years per cell (0..1)
      - count:    number of years a cell is hotspot (0..N)
    """
    yrs = sorted({int(y) for y in years})
    if not yrs:
        z = np.full((int(grid_res), int(grid_res)), np.nan, float)
        return z, extent, 0

    # shape will be inferred from first grid we compute
    pers_cnt = None
    valid_cnt = None
    ext2 = extent

    for yy in yrs:
        sub = dfp.loc[dfp["coord_t"].astype(int).eq(int(yy))]
        if sub.empty:
            continue

        x = pd.to_numeric(sub["coord_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["coord_y"], errors="coerce").to_numpy(float)
        d = pd.to_numeric(sub["delta_abs"], errors="coerce").to_numpy(float)
        p = pd.to_numeric(sub["p_exceed"], errors="coerce").to_numpy(float)

        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        d = d[ok]
        p = p[ok]

        z_d, ext2 = _grid_mean(x, y, d, res=grid_res, extent=extent)
        z_p, _ = _grid_mean(x, y, p, res=grid_res, extent=extent)

        thr = float("nan")
        if "T_q" in dfy.columns:
            tsub = dfy.loc[dfy["year"].astype(int).eq(int(yy))]
            if not tsub.empty:
                thr = float(tsub["T_q"].iloc[0])
        if not np.isfinite(thr):
            d2 = d[np.isfinite(d)]
            thr = float(np.nanquantile(d2, 0.90)) if d2.size else float("nan")
        mask = _grid_hotspot_mask(
            z_d,
            z_p,
            thr=thr,
            hotspot_rule=hotspot_rule,
            hotspot_abs=hotspot_abs,
            risk_min=risk_min,
        )

        valid = np.isfinite(z_d) | np.isfinite(z_p)

        if pers_cnt is None:
            pers_cnt = np.zeros_like(z_d, dtype=float)
            valid_cnt = np.zeros_like(z_d, dtype=float)

        pers_cnt += mask.astype(float)
        valid_cnt += valid.astype(float)

    if pers_cnt is None or valid_cnt is None:
        z = np.full((int(grid_res), int(grid_res)), np.nan, float)
        return z, extent, len(yrs)

    z = pers_cnt.copy()
    z[valid_cnt == 0] = np.nan
    
    mm = str(mode).strip().lower()
    if mm == "fraction":
        with np.errstate(invalid="ignore", divide="ignore"):
            z = z / valid_cnt
        z[valid_cnt == 0] = np.nan

    return z, ext2, len(yrs)

def _label_components(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    lab = np.zeros(m.shape, dtype=int)
    h, w = m.shape
    cur = 0

    for iy in range(h):
        for ix in range(w):
            if not m[iy, ix] or lab[iy, ix] != 0:
                continue
            cur += 1
            stack = [(iy, ix)]
            lab[iy, ix] = cur
            while stack:
                y0, x0 = stack.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    y1 = y0 + dy
                    x1 = x0 + dx
                    if y1 < 0 or y1 >= h or x1 < 0 or x1 >= w:
                        continue
                    if not m[y1, x1] or lab[y1, x1] != 0:
                        continue
                    lab[y1, x1] = cur
                    stack.append((y1, x1))

    return lab


def _cell_centers(
    extent: Tuple[float, float, float, float],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = extent
    ny, nx = shape
    dx = (xmax - xmin) / float(nx)
    dy = (ymax - ymin) / float(ny)
    xs = xmin + (np.arange(nx) + 0.5) * dx
    ys = ymin + (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy


def _cluster_summary(
    *,
    lab: np.ndarray,
    metric: np.ndarray,
    risk: np.ndarray,
    extent: Tuple[float, float, float, float],
    city: str,
    year: int,
    rank_by: str,
) -> pd.DataFrame:
    labs = np.unique(lab)
    labs = labs[labs > 0]
    if labs.size == 0:
        cols = [
            "city",
            "year",
            "cluster_id",
            "n_cells",
            "metric_mean",
            "metric_max",
            "risk_mean",
            "risk_max",
            "centroid_x",
            "centroid_y",
        ]
        return pd.DataFrame(columns=cols)

    xx, yy = _cell_centers(extent, lab.shape)
    rows: List[Dict[str, Any]] = []
    for k in labs:
        m = lab == int(k)
        mv = metric[m]
        rv = risk[m]
        mv = mv[np.isfinite(mv)]
        rv = rv[np.isfinite(rv)]
        if mv.size == 0:
            continue
        cx = float(np.nanmean(xx[m]))
        cy = float(np.nanmean(yy[m]))
        rows.append(
            {
                "city": city,
                "year": int(year),
                "cluster_id": int(k),
                "n_cells": int(np.sum(m)),
                "metric_mean": float(np.nanmean(mv)),
                "metric_max": float(np.nanmax(mv)),
                "risk_mean": float(np.nanmean(rv)) if rv.size else float("nan"),
                "risk_max": float(np.nanmax(rv)) if rv.size else float("nan"),
                "centroid_x": cx,
                "centroid_y": cy,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # out = out.sort_values(
    #     ["metric_mean", "n_cells"],
    #     ascending=[False, False],
    # )
    key = "metric_mean"
    if str(rank_by).strip().lower() == "risk":
        key = "risk_mean"
    
    out = out.sort_values(
        [key, "n_cells"],
        ascending=[False, False],
    )

    out["priority_rank"] = np.arange(1, len(out) + 1)
    return out


# ---------------------------------------------------------------------
# Core computation per city
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CityHotspotData:
    city: str
    color: str
    years: List[int]
    base_year: int
    df_points: pd.DataFrame
    df_years: pd.DataFrame
    extent: Tuple[float, float, float, float]
    
def _choose_hotspot_mask(
    pts: pd.DataFrame,
    *,
    rule: str,
) -> np.ndarray:
    r = str(rule).strip().lower()

    m_pct = pts["is_hotspot_pct"].to_numpy(bool)

    if "is_hotspot_abs" in pts.columns:
        m_abs = pts["is_hotspot_abs"].to_numpy(bool)
    else:
        m_abs = m_pct

    if "is_hotspot_risk" in pts.columns:
        m_rsk = pts["is_hotspot_risk"].to_numpy(bool)
    else:
        m_rsk = m_pct

    if r == "abs":
        return m_abs
    if r == "risk":
        return m_rsk
    if r == "combo":
        return m_abs | m_rsk
    return m_pct

def build_hotspot_tables(
    *,
    city: str,
    color: str,
    eval_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    years: List[int],
    base_year: int,
    subsidence_kind: str,
    hotspot_q: float,
    risk_threshold: float,
    use_abs_risk: bool,
    hotspot_rule: str,
    hotspot_abs: Optional[float],
    risk_min: Optional[float],
    exposure_df: Optional[pd.DataFrame],
    exposure_col: str,
    exposure_default: float,
) -> CityHotspotData:
    """
    Build point/year hotspot tables for one city.

    Robust points:
    - If kind="cumulative", annual rates are derived via diffs.
      For the first requested year, we automatically include the
      previous year if available (future preferred, else eval).
    - Hotspot rule can be percentile/abs/risk/combo (optional).
    - Risk can be exposure-weighted: P * |Δs| * exposure.

    Returns CityHotspotData with:
      df_points: per-point metrics for requested years
      df_years: per-year summaries for requested years
    """
    # ---- normalize years
    years_req = sorted({int(y) for y in years})
    if not years_req:
        raise ValueError("years must be non-empty")

    knd = str(subsidence_kind).strip().lower()
    is_cum = knd == "cumulative"

    y_min = int(min(years_req))
    y_prev = int(y_min - 1)

    # ---- eval: clean year + ids
    ev = eval_df.copy()
    ev["coord_t"] = pd.to_numeric(ev["coord_t"], errors="coerce")
    ev = ev.dropna(subset=["coord_t"]).copy()
    ev["coord_t"] = ev["coord_t"].astype(int)

    # ---- baseline annual (actual) at base_year
    if is_cum:
        a0 = _annual_from_cumulative(
            ev,
            value_col="subsidence_actual",
            year_col="coord_t",
            id_col="sample_idx",
        )
        m_b = ev["coord_t"].eq(int(base_year))
        base_ids = ev.loc[m_b, "sample_idx"].to_numpy(int)
        base_vals = a0.loc[m_b].to_numpy(float)
        base_annual = pd.Series(base_vals, index=base_ids)
    else:
        base_annual = (
            ev.loc[
                ev["coord_t"].eq(int(base_year)),
                ["sample_idx", "subsidence_actual"],
            ]
            .dropna()
            .assign(
                sample_idx=lambda d: pd.to_numeric(
                    d["sample_idx"], errors="coerce"
                )
            )
            .dropna(subset=["sample_idx"])
            .assign(sample_idx=lambda d: d["sample_idx"].astype(int))
            .set_index("sample_idx")["subsidence_actual"]
            .astype(float)
        )

    # ---- future: coerce year + keep needed years
    fu = fut_df.copy()
    fu["coord_t"] = pd.to_numeric(fu["coord_t"], errors="coerce")
    fu = fu.dropna(subset=["coord_t"]).copy()
    fu["coord_t"] = fu["coord_t"].astype(int)

    years_use = set(years_req)
    if is_cum:
        years_use.add(y_prev)

    fu = fu.loc[fu["coord_t"].isin(years_use)].copy()

    # ---- for cumulative: ensure we have y_prev rows per sample
    # Prefer future rows; fill missing from eval quantiles at y_prev.
    if is_cum:
        have_prev = fu.loc[fu["coord_t"].eq(y_prev), "sample_idx"]
        have_prev = pd.to_numeric(have_prev, errors="coerce")
        have_prev = have_prev.dropna().astype(int)
        have_prev_ids = set(have_prev.to_list())

        need_prev = fu.loc[fu["coord_t"].eq(y_min), "sample_idx"]
        need_prev = pd.to_numeric(need_prev, errors="coerce")
        need_prev = need_prev.dropna().astype(int)
        need_prev_ids = set(need_prev.to_list())

        miss_ids = sorted(list(need_prev_ids - have_prev_ids))
        if miss_ids:
            # eval may not have q10/q90; fallback to q50
            c10 = (
                "subsidence_q10"
                if "subsidence_q10" in ev.columns
                else "subsidence_q50"
            )
            c90 = (
                "subsidence_q90"
                if "subsidence_q90" in ev.columns
                else "subsidence_q50"
            )

            anchor = ev.loc[
                ev["coord_t"].eq(y_prev),
                ["sample_idx", "coord_t", "coord_x", "coord_y", c10,
                 "subsidence_q50", c90],
            ].copy()

            anchor = anchor.rename(
                columns={
                    c10: "subsidence_q10",
                    c90: "subsidence_q90",
                }
            )

            anchor["sample_idx"] = pd.to_numeric(
                anchor["sample_idx"], errors="coerce"
            )
            anchor = anchor.dropna(subset=["sample_idx"]).copy()
            anchor["sample_idx"] = anchor["sample_idx"].astype(int)

            anchor = anchor.loc[anchor["sample_idx"].isin(miss_ids)]
            if not anchor.empty:
                fu = pd.concat([fu, anchor], ignore_index=True)

    # ---- aggregate duplicates safely, then annualize
    fu = fu.copy()
    fu["sample_idx"] = pd.to_numeric(fu["sample_idx"], errors="coerce")
    fu = fu.dropna(subset=["sample_idx"]).copy()
    fu["sample_idx"] = fu["sample_idx"].astype(int)

    fu = (
        fu.groupby(["sample_idx", "coord_t"], as_index=False)
        .agg(
            coord_x=("coord_x", "mean"),
            coord_y=("coord_y", "mean"),
            subsidence_q10=("subsidence_q10", "mean"),
            subsidence_q50=("subsidence_q50", "mean"),
            subsidence_q90=("subsidence_q90", "mean"),
        )
        .sort_values(["sample_idx", "coord_t"])
        .reset_index(drop=True)
    )

    q10_a = _annual_series(fu, value_col="subsidence_q10", kind=knd)
    q50_a = _annual_series(fu, value_col="subsidence_q50", kind=knd)
    q90_a = _annual_series(fu, value_col="subsidence_q90", kind=knd)

    fu = fu.copy()
    fu["s_q10"] = q10_a.to_numpy(float)
    fu["s_q50"] = q50_a.to_numpy(float)
    fu["s_q90"] = q90_a.to_numpy(float)

    # Keep only requested years for outputs
    pts = fu.loc[
        fu["coord_t"].isin(set(years_req)),
        ["sample_idx", "coord_t", "coord_x", "coord_y",
         "s_q10", "s_q50", "s_q90"],
    ].copy()

    # ---- anomaly vs baseline annual actual
    pts["s_base"] = pts["sample_idx"].map(base_annual)
    pts["delta_abs"] = np.abs(
        pts["s_q50"].to_numpy(float)
        - pts["s_base"].to_numpy(float)
    )

    # ---- exceedance probability from quantiles
    s10 = pts["s_q10"].to_numpy(float)
    s50 = pts["s_q50"].to_numpy(float)
    s90 = pts["s_q90"].to_numpy(float)
    if bool(use_abs_risk):
        s10 = np.abs(s10)
        s50 = np.abs(s50)
        s90 = np.abs(s90)

    pts["p_exceed"] = exceed_prob_from_quantiles(
        s10,
        s50,
        s90,
        threshold=float(risk_threshold),
    )

    # ---- exposure-weighted risk score
    pts = _attach_exposure(
        pts,
        exposure_df,
        col=str(exposure_col),
        default=float(exposure_default),
    )
    pts["risk_score"] = (
        pts["delta_abs"].to_numpy(float)
        * pts["p_exceed"].to_numpy(float)
        * pts["exposure"].to_numpy(float)
    )

    # ---- percentile thresholds per year (T_q)
    thr_map: Dict[int, float] = {}
    for yy, g in pts.groupby("coord_t", dropna=False):
        a = pd.to_numeric(g["delta_abs"], errors="coerce").to_numpy(float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            thr_map[int(yy)] = float("nan")
        else:
            thr_map[int(yy)] = float(
                np.nanquantile(a, float(hotspot_q))
            )

    pts["T_q"] = pts["coord_t"].map(thr_map)

    # ---- hotspot flags (optional)
    pts["is_hotspot_pct"] = (
        np.isfinite(pts["delta_abs"].to_numpy(float))
        & np.isfinite(pts["T_q"].to_numpy(float))
        & (
            pts["delta_abs"].to_numpy(float)
            >= pts["T_q"].to_numpy(float)
        )
    )

    if hotspot_abs is not None:
        pts["is_hotspot_abs"] = (
            pts["delta_abs"].to_numpy(float) >= float(hotspot_abs)
        )

    if risk_min is not None:
        pts["is_hotspot_risk"] = (
            pts["p_exceed"].to_numpy(float) >= float(risk_min)
        )

    pts["is_hotspot"] = _choose_hotspot_mask(
        pts,
        rule=str(hotspot_rule),
    )

    pts["city"] = str(city)

    # ---- per-year summary (requested years only)
    # rows_y: List[Dict[str, Any]] = []
    # for yy in years_req:
    #     g = pts.loc[pts["coord_t"].eq(int(yy))].copy()
    #     if g.empty:
    #         continue

    #     a = pd.to_numeric(g["delta_abs"], errors="coerce").to_numpy(float)
    #     p = pd.to_numeric(g["p_exceed"], errors="coerce").to_numpy(float)

    #     a2 = a[np.isfinite(a)]
    #     p2 = p[np.isfinite(p)]

    #     h_sel = g["is_hotspot"].to_numpy(bool)
    #     h_pct = g["is_hotspot_pct"].to_numpy(bool)

    #     n_abs = -1
    #     if "is_hotspot_abs" in g.columns:
    #         n_abs = int(np.sum(g["is_hotspot_abs"].to_numpy(bool)))

    #     n_rsk = -1
    #     if "is_hotspot_risk" in g.columns:
    #         n_rsk = int(np.sum(g["is_hotspot_risk"].to_numpy(bool)))

    #     rows_y.append(
    #         {
    #             "city": str(city),
    #             "year": int(yy),
    #             "T_q": float(thr_map.get(int(yy), np.nan)),
    #             "n_hotspots": int(np.sum(h_sel)),
    #             "n_hotspots_pct": int(np.sum(h_pct)),
    #             "n_hotspots_abs": n_abs,
    #             "n_hotspots_risk": n_rsk,
    #             "delta_mean": float(np.nanmean(a2)) if a2.size else float("nan"),
    #             "delta_max": float(np.nanmax(a2)) if a2.size else float("nan"),
    #             "p_mean": float(np.nanmean(p2)) if p2.size else float("nan"),
    #             "risk_sum": float(
    #                 np.nansum(
    #                     pd.to_numeric(
    #                         g["risk_score"],
    #                         errors="coerce",
    #                     ).to_numpy(float)
    #                 )
    #             ),
    #         }
    #     )
    # ---- per-year summary (requested years only)
    rows_y: List[Dict[str, Any]] = []
    ever_set: set[int] = set()
    
    for yy in years_req:
        g = pts.loc[pts["coord_t"].eq(int(yy))].copy()
        if g.empty:
            continue
    
        # robust unique sets (never sum booleans)
        sel_ids = set(
            pd.to_numeric(
                g.loc[g["is_hotspot"], "sample_idx"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    
        pct_ids = set(
            pd.to_numeric(
                g.loc[g["is_hotspot_pct"], "sample_idx"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    
        abs_ids = None
        if "is_hotspot_abs" in g.columns:
            abs_ids = set(
                pd.to_numeric(
                    g.loc[g["is_hotspot_abs"], "sample_idx"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
    
        rsk_ids = None
        if "is_hotspot_risk" in g.columns:
            rsk_ids = set(
                pd.to_numeric(
                    g.loc[g["is_hotspot_risk"], "sample_idx"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
    
        # ever/new based on SELECTED rule
        new_ids = sel_ids - ever_set
        ever_set |= sel_ids
    
        a = pd.to_numeric(g["delta_abs"], errors="coerce").to_numpy(float)
        p = pd.to_numeric(g["p_exceed"], errors="coerce").to_numpy(float)
        a2 = a[np.isfinite(a)]
        p2 = p[np.isfinite(p)]
    
        rows_y.append(
            {
                "city": str(city),
                "year": int(yy),
                "T_q": float(thr_map.get(int(yy), np.nan)),
    
                # timeline variants
                "n_hotspots": int(len(sel_ids)),           # current
                "n_hotspots_ever": int(len(ever_set)),     # monotone
                "n_hotspots_new": int(len(new_ids)),       # additions
    
                # keep rule-specific counts for transparency
                "n_hotspots_pct": int(len(pct_ids)),
                "n_hotspots_abs": int(len(abs_ids)) if abs_ids is not None else -1,
                "n_hotspots_risk": int(len(rsk_ids)) if rsk_ids is not None else -1,
    
                "delta_mean": float(np.nanmean(a2)) if a2.size else float("nan"),
                "delta_max": float(np.nanmax(a2)) if a2.size else float("nan"),
                "p_mean": float(np.nanmean(p2)) if p2.size else float("nan"),
    
                "risk_sum": float(
                    np.nansum(
                        pd.to_numeric(
                            g["risk_score"],
                            errors="coerce",
                        ).to_numpy(float)
                    )
                ),
            }
        )
        
    df_years = pd.DataFrame(rows_y)
    ext = _extent(pts)

    return CityHotspotData(
        city=str(city),
        color=str(color),
        years=years_req,
        base_year=int(base_year),
        df_points=pts,
        df_years=df_years,
        extent=ext,
    )

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
def _axes_cleanup(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _clip_bounds(vals: List[np.ndarray], clip: float) -> Tuple[float, float]:
    good: List[np.ndarray] = []
    for v in vals:
        a = np.asarray(v)
        a = a[np.isfinite(a)]
        if a.size:
            good.append(a)
    if not good:
        return (0.0, 1.0)
    g = np.concatenate(good)
    hi = float(np.nanpercentile(g, clip))
    lo = float(np.nanpercentile(g, 100.0 - clip))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.nanmin(g))
        hi = float(np.nanmax(g))
    return lo, hi


def plot_hotspot_analytics(
    *,
    cities: List[CityHotspotData],
    focus_year: int,
    grid_res: int,
    clip: float,
    cmap_metric: str,
    cmap_prob: str,
    persistence_mode: str,
    cluster_top: int,
    risk_threshold: float,
    show_legend: bool,
    show_title: bool,
    show_panel_titles: bool,
    title: Optional[str],
    out: str,
    out_points: Optional[str],
    out_years: Optional[str],
    out_clusters: Optional[str],
    timeline_mode: str,
    timeline_overlay_current: bool,
    dpi: int,
    font: int,
    cluster_rank: str, 
    add_persistence: bool,
    add_compare: bool, 
    compare_metric: str,
    hotspot_rule:str, 
    hotspot_abs:Optional[float],
    risk_min: Optional[float], 
    ns_future_b: Optional[str] = None, 
    zh_future_b: Optional[str] = None, 
    gdf: Optional[Union[GeoDF, Dict[str, GeoDF]]] = None,

) -> None:
    utils.ensure_script_dirs()
    utils.set_paper_style(fontsize=int(font), dpi=int(dpi))

    cmap1 = plt.get_cmap(str(cmap_metric))
    cmap2 = plt.get_cmap(str(cmap_prob))

    dvals: List[np.ndarray] = []
    for c in cities:
        dvals.append(
            pd.to_numeric(
                c.df_points["delta_abs"], errors="coerce"
            ).to_numpy(float)
        )
    dmin, dmax = _clip_bounds(dvals, float(clip))

    n_rows = len(cities)
    n_cols = 4
    if add_persistence:
        n_cols += 1
    
    have_b = ( 
        ns_future_b is not None 
        or zh_future_b is not None  # or since we can compare for one city only 
        # and if we want to compare both cities obligatory 
    )
    if add_compare and have_b:
        n_cols += 1
        
    fig_w = 2.4 * n_cols + 1.8
    fig_h = 2.5 * n_rows + 1.2

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
    )
    ax_arr = np.array(axes, ndmin=2)

    # plt.subplots_adjust(
    #     left=0.06,
    #     right=0.92 if show_legend else 0.98,
    #     top=0.92,
    #     bottom=0.10,
    #     wspace=0.18,
    #     hspace=0.25,
    # )
    plot_right = 0.88 if show_legend else 0.98
    
    plt.subplots_adjust(
        left=0.06,
        right=plot_right,
        top=0.92,
        bottom=0.10,
        wspace=0.26,
        hspace=0.25,
    )

    all_points: List[pd.DataFrame] = []
    all_years: List[pd.DataFrame] = []
    all_clusters: List[pd.DataFrame] = []

    for r, c in enumerate(cities):
        dfp = c.df_points.copy()
        dfy = c.df_years.copy()
        all_points.append(dfp)
        all_years.append(dfy)

        city = c.city
        col = c.color
        ext = c.extent

        # sub = dfp.loc[dfp["coord_t"].astype(int).eq(int(focus_year))]
        sub = dfp.loc[dfp["coord_t"].astype(int).eq(int(focus_year))]
        focus_y = int(focus_year)
        if sub.empty:
            focus_y = int(np.nanmax(dfp["coord_t"].to_numpy(int)))
            sub = dfp.loc[dfp["coord_t"].astype(int).eq(int(focus_y))]

        # x = pd.to_numeric(sub["coord_x"], errors="coerce").to_numpy(float)
        # y = pd.to_numeric(sub["coord_y"], errors="coerce").to_numpy(float)
        # d = pd.to_numeric(sub["delta_abs"], errors="coerce").to_numpy(float)
        # p = pd.to_numeric(sub["p_exceed"], errors="coerce").to_numpy(float)

        # ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(d)
        # x = x[ok]
        # y = y[ok]
        # d = d[ok]
        # p = p[ok]
        # p = np.where(np.isfinite(p), p, 0.0)

        # # z_d, ext2 = _grid_mean(x, y, d, res=grid_res, extent=ext)
        # z_d, ext2 = _grid_mean(x, y, d, res=grid_res, extent=ext)
        
        # z_p, _ = _grid_mean(x, y, p, res=grid_res, extent=ext)
        # # z_r, _ = _grid_mean(x, y, d * p, res=grid_res, extent=ext)
        # rs = pd.to_numeric(
        # sub["risk_score"], errors="coerce"
        #     ).to_numpy(float)
        
        # ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(d)
        # # also require rs for risk grid
        # rs = rs[ok]
        
        # z_r, _ = _grid_mean(x, y, rs,
        #     res=grid_res, extent=ext)
        
        x = pd.to_numeric(
            sub["coord_x"], errors="coerce"
        ).to_numpy(float)
        y = pd.to_numeric(
            sub["coord_y"], errors="coerce"
        ).to_numpy(float)
        d = pd.to_numeric(
            sub["delta_abs"], errors="coerce"
        ).to_numpy(float)
        p = pd.to_numeric(
            sub["p_exceed"], errors="coerce"
        ).to_numpy(float)
        rs = pd.to_numeric(
            sub["risk_score"], errors="coerce"
        ).to_numpy(float)
        
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(d)
        x = x[ok]
        y = y[ok]
        d = d[ok]
        p = p[ok]
        rs = rs[ok]
        
        p = np.where(np.isfinite(p), p, 0.0)
        rs = np.where(np.isfinite(rs), rs, 0.0)
        
        z_d, ext2 = _grid_mean(
            x, y, d, res=grid_res, extent=ext
        )
        z_p, _ = _grid_mean(
            x, y, p, res=grid_res, extent=ext
        )
        z_r, _ = _grid_mean(
            x, y, rs, res=grid_res, extent=ext
        )

        thr = float("nan")
        if "T_q" in dfy.columns:
            tsub = dfy.loc[dfy["year"].eq(int(focus_y))]
            if not tsub.empty:
                thr = float(tsub["T_q"].iloc[0])
        if not np.isfinite(thr):
            thr = float(np.nanquantile(d, 0.90))

        mask = _grid_hotspot_mask(
            z_d,
            z_p,
            thr=thr,
            hotspot_rule=hotspot_rule,
            hotspot_abs=hotspot_abs,
            risk_min=risk_min,
        )
    
        lab = _label_components(mask)

        cl = _cluster_summary(
            lab=lab,
            metric=z_d,
            risk=z_r,
            extent=ext,
            city=city,
            year=int(focus_y),
            rank_by=cluster_rank,
        )
        all_clusters.append(cl)
        
        # boundaries per city (optional)
        gdf_here = None
        if isinstance(gdf, dict):
            gdf_here = gdf.get(city)
            if gdf_here is None or getattr(gdf_here, "empty", False):
                g_alt = gdf.get(_slug_city(city))
                if g_alt is not None and not getattr(g_alt, "empty", False):
                    gdf_here = g_alt
        else:
            gdf_here = gdf

        # persistence grid (optional)
        z_pers = None
        pers_ext = ext2
        n_pers_years = 0
        if add_persistence:
            z_pers, pers_ext, n_pers_years = _compute_persistence_grid(
                dfp=dfp,
                dfy=dfy,
                years=c.years,
                extent=ext,
                grid_res=int(grid_res),
                hotspot_rule=str(hotspot_rule),
                hotspot_abs=hotspot_abs,
                risk_min=risk_min,
                mode=str(persistence_mode),
            )

        # (1) anomaly map
        ax0 = ax_arr[r, 0]
        ax0.imshow(
            z_d,
            extent=ext2,
            origin="lower",
            cmap=cmap1,
            vmin=dmin,
            vmax=dmax,
            interpolation="nearest",
            aspect="equal",
        )
        if np.any(lab > 0):
            ax0.contour(
                (lab > 0).astype(float),
                levels=[0.5],
                colors="k",
                linewidths=0.7,
                origin="lower",
                extent=ext2,
            )
        if not cl.empty and int(cluster_top) > 0:
            topc = cl.head(int(cluster_top))
            for _, row in topc.iterrows():
                ax0.text(
                    float(row["centroid_x"]),
                    float(row["centroid_y"]),
                    str(int(row["priority_rank"])),
                    color=col,
                    fontsize=max(7, int(font)),
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "circle,pad=0.15",
                        "fc": "white",
                        "ec": col,
                        "lw": 0.8,
                        "alpha": 0.9,
                    },
                )
        try:
            gdf_here.boundary.plot(
                ax=ax0,
                color="k",
                linewidth=0.6,
                alpha=0.7,
            )
        except Exception:
            pass

        _axes_cleanup(ax0)

        # (2) probability map
        ax1 = ax_arr[r, 1]
        ax1.imshow(
            z_p,
            extent=ext2,
            origin="lower",
            cmap=cmap2,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        if np.any(lab > 0):
            ax1.contour(
                (lab > 0).astype(float),
                levels=[0.5],
                colors="k",
                linewidths=0.7,
                origin="lower",
                extent=ext2,
            )
        try:
            gdf_here.boundary.plot(
                ax=ax1,
                color="k",
                linewidth=0.6,
                alpha=0.7,
            )
        except Exception:
            pass
        
        _axes_cleanup(ax1)

        # (3) timeline
        ax2 = ax_arr[r, 2]
        t = dfy.sort_values("year")
        yrs = t["year"].to_numpy(int)

        mode = str(timeline_mode).strip().lower()
        
        col_map = {
            "current": "n_hotspots",
            "ever": "n_hotspots_ever",
            "new": "n_hotspots_new",
        }
        k = col_map.get(mode, "n_hotspots")
        if k not in t.columns:
            k = "n_hotspots"
        
        nh = pd.to_numeric(t[k], errors="coerce").to_numpy(float)
        nh_cur = pd.to_numeric(
            t.get("n_hotspots", np.nan),
            errors="coerce",
        ).to_numpy(float)
        
        rgba_bar = mpl.colors.to_rgba(col, 0.25)
        bar_lab = mode if mode in ("ever", "new") else "current"
        
        ax2.bar(
            yrs,
            nh,
            width=0.72,
            color=rgba_bar,
            edgecolor=col,
            linewidth=1.0,
            zorder=2,
            label=bar_lab,
        )
        
        ax2.set_axisbelow(True)
        ax2.grid(True, axis="y", alpha=0.20, zorder=0)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)
        
        if mode == "ever":
            yl = "# hotspots (ever)"
        elif mode == "new":
            yl = "# hotspots (new)"
        else:
            yl = "# hotspots (current)"
        ax2.set_ylabel(yl)
        
        # ---- optional overlay: dashed current line
        do_overlay = (
            bool(timeline_overlay_current)
            and mode in ("ever", "new")
            and np.any(np.isfinite(nh_cur))
        )
        
        if do_overlay:
            rgba_line = mpl.colors.to_rgba(col, 0.90)
            ax2.plot(
                yrs,
                nh_cur,
                linestyle="--",
                marker="o",
                linewidth=1.2,
                markersize=4.0,
                color=rgba_line,
                zorder=3,
                label="current",
            )
        
            # light, local legend (only for this axis)
            ax2.legend(
                loc="upper right",
                frameon=False,
                fontsize=max(7, int(font) - 1),
                handlelength=2.0,
            )


        ax2b = ax2.twinx()
        tq = pd.to_numeric(t["T_q"], errors="coerce").to_numpy(float)
        dx = pd.to_numeric(t["delta_max"], errors="coerce").to_numpy(float)
        # ax2b.plot(yrs, tq, marker="o", linestyle="--")
        # ax2b.plot(yrs, dx, marker="s", linestyle="-")
        ax2b.plot(
            yrs, tq,
            marker="o",
            linestyle="--",
            linewidth=1.2,
        )
        ax2b.plot(
            yrs, dx,
            marker="s",
            linestyle="-",
            linewidth=1.2,
        )

        ax2.set_xlabel("Year")
        # ax2.set_ylabel("# hotspots")
        # ax2b.set_ylabel("T0.9 / max|Δs|")
        ax2b.set_ylabel(
            r"T$_{0.9}$ / max|Δs|",
            labelpad=1,
        )
        # pull the label *inside* so it doesn't hit ax3 labels
        ax2b.yaxis.set_label_coords(0.98, 0.5)
        ax2b.tick_params(pad=1)

        ax2.grid(True, axis="y", alpha=0.25)

        # (4) priority clusters (Nature-style)
        ax3 = ax_arr[r, 3]
        for s in ("top", "right"):
            ax3.spines[s].set_visible(False)
        ax3.set_axisbelow(True)
        ax3.grid(True, axis="x", alpha=0.18)
        
        if cl.empty:
            ax3.text(
                0.5,
                0.5,
                "No hotspots",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_xticks([])
            ax3.set_yticks([])
        else:
            topc = cl.head(int(cluster_top)).copy()
        
            key = "metric_mean"
            xlab = r"mean |Δs| (mm yr$^{-1}$)"
            if str(cluster_rank).strip().lower() == "risk":
                key = "risk_mean"
                xlab = "mean risk score"
        
            vals = pd.to_numeric(
                topc[key], errors="coerce"
            ).to_numpy(float)
        
            ylab = []
            for _, rr in topc.iterrows():
                rk = int(rr["priority_rank"])
                cid = int(rr["cluster_id"])
                ylab.append(f"{rk}. C{cid}")
        
            yk = np.arange(len(ylab))
            rgba = mpl.colors.to_rgba(col, 0.75)
        
            bars = ax3.barh(
                yk,
                vals,
                color=rgba,
                edgecolor=col,
                linewidth=1.0,
            )
        
            ax3.set_yticks(yk)
            ax3.set_yticklabels(ylab)
            ax3.tick_params(axis="y", length=0)
            ax3.invert_yaxis()
        
            ax3.set_xlabel(xlab)
        
            vmax = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else 1.0
            pad = 0.03 * vmax
            ax3.set_xlim(0.0, vmax + 6.0 * pad)
        
            for b, v in zip(bars, vals):
                if not np.isfinite(v):
                    continue
                ax3.text(
                    float(v) + pad,
                    b.get_y() + 0.5 * b.get_height(),
                    f"{v:.1f}",
                    va="center",
                    ha="left",
                    fontsize=max(7, int(font) - 1),
                )
        
        # (5) persistence panel (optional)
        if add_persistence:
            ax4 = ax_arr[r, 4]
            mm = str(persistence_mode).strip().lower()
            if mm == "count":
                vmin, vmax = 0.0, float(max(1, n_pers_years))
            else:
                vmin, vmax = 0.0, 1.0

            ax4.imshow(
                z_pers,
                extent=pers_ext,
                origin="lower",
                cmap=cmap2,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
            try:
                gdf_here.boundary.plot(
                    ax=ax4,
                    color="k",
                    linewidth=0.6,
                    alpha=0.7,
                )
            except Exception:
                pass
            _axes_cleanup(ax4)

        if show_panel_titles:
            ax0.set_title(
                f"{city} • {focus_y} |Δs| vs {c.base_year}",
                loc="left",
                fontweight="bold",
                pad=4,
            )
            ax1.set_title(
                f"{city} • P(|s|≥{risk_threshold:g})",
                loc="left",
                fontweight="bold",
                pad=4,
            )
            tag = mode if mode in ("current", "ever", "new") else "current"
            ax2.set_title(
                f"{city} • hotspot evolution ({tag})",
                loc="left",
                fontweight="bold",
                pad=4,
            )
            ax3.set_title(
                f"{city} • priority clusters",
                loc="left",
                fontweight="bold",
                pad=4,
            )
            if add_persistence:
                mm = str(persistence_mode).strip().lower()
                tag = "fraction" if mm != "count" else "count"
                ax4.set_title(
                    f"{city} • persistence ({tag})",
                    loc="left",
                    fontweight="bold",
                    pad=4,
                )


    if show_legend:
        # Colorbar strip starts just to the right of the subplot grid
        cbar_x = plot_right + 0.02
        cbar_w = 0.012
    
        if add_persistence:
            # 3 stacked bars with comfortable gaps
            cax_d = fig.add_axes([cbar_x, 0.68, cbar_w, 0.22])
            cax_s = fig.add_axes([cbar_x, 0.40, cbar_w, 0.20])
            cax_p = fig.add_axes([cbar_x, 0.12, cbar_w, 0.22])
        else:
            # 2 stacked bars
            cax_d = fig.add_axes([cbar_x, 0.56, cbar_w, 0.30])
            cax_p = fig.add_axes([cbar_x, 0.16, cbar_w, 0.30])
            cax_s = None
    
        cb1 = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=dmin, vmax=dmax),
                cmap=cmap1,
            ),
            cax=cax_d,
        )
        cb1.set_label(r"|Δs| (mm yr$^{-1}$)")
    
        if add_persistence and cax_s is not None:
            mm = str(persistence_mode).strip().lower()
            vmaxp = float(max(1, len(cities[0].years))) if mm == "count" else 1.0
    
            cbp = fig.colorbar(
                mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(vmin=0.0, vmax=vmaxp),
                    cmap=cmap2,
                ),
                cax=cax_s,
            )
            cbp.set_label(
                "Hotspot persistence (years)"
                if mm == "count"
                else "Hotspot persistence (fraction)"
            )
    
        cb2 = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
                cmap=cmap2,
            ),
            cax=cax_p,
        )
        cb2.set_label(r"P(|s| ≥ T)")
    
    if show_title:
        ttl = utils.resolve_title(
            default="Hotspot analytics — where to act first",
            title=title,
        )
        fig.suptitle(ttl, x=0.02, ha="left")

    utils.save_figure(fig, out, dpi=int(dpi))
 
    p_pts = utils.as_path(out_points) if out_points else \
        utils.resolve_out_out("hotspot_points.csv")
    pd.concat(all_points, ignore_index=True).to_csv(p_pts, index=False)
    print(f"[OK] wrote {p_pts}")

    p_yrs = utils.as_path(out_years) if out_years else \
        utils.resolve_out_out("hotspot_years.csv")
    pd.concat(all_years, ignore_index=True).to_csv(p_yrs, index=False)
    print(f"[OK] wrote {p_yrs}")

    p_cls = utils.as_path(out_clusters) if out_clusters else \
        utils.resolve_out_out("hotspot_clusters.csv")
    pd.concat(all_clusters, ignore_index=True).to_csv(p_cls, index=False)
    print(f"[OK] wrote {p_cls}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _add_args(ap: argparse.ArgumentParser) -> None:
    utils.add_city_flags(ap, default_both=True)

    ap.add_argument("--ns-src", type=str, default=None)
    ap.add_argument("--zh-src", type=str, default=None)

    ap.add_argument("--ns-eval", type=str, default=None)
    ap.add_argument("--zh-eval", type=str, default=None)
    ap.add_argument("--ns-future", type=str, default=None)
    ap.add_argument("--zh-future", type=str, default=None)

    ap.add_argument(
        "--split",
        type=str,
        choices=["auto", "val", "test"],
        default="auto",
    )

    ap.add_argument("--base-year", type=int, default=2022)
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2025, 2026],
    )
    ap.add_argument("--focus-year", type=int, default=None)

    ap.add_argument(
        "--subsidence-kind",
        type=str,
        default="cumulative",
        choices=["cumulative", "rate", "increment"],
    )
    ap.add_argument("--hotspot-quantile", type=float, default=0.90)

    ap.add_argument("--risk-threshold", type=float, default=50.0)
    ap.add_argument("--risk-abs", type=str, default="true")

    ap.add_argument("--grid-res", type=int, default=300)
    ap.add_argument("--clip", type=float, default=98.0)

    ap.add_argument("--cmap-metric", type=str, default="magma")
    ap.add_argument("--cmap-prob", type=str, default="viridis")
    ap.add_argument("--cluster-top", type=int, default=6)

    ap.add_argument("--out-points", type=str, default=None)
    ap.add_argument("--out-years", type=str, default=None)
    ap.add_argument("--out-clusters", type=str, default=None)

    ap.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)
    ap.add_argument("--font", type=int, default=cfg.PAPER_FONT)

    ap.add_argument(
        "--hotspot-rule",
        type=str,
        default="percentile",
        choices=[
            "percentile",
            "abs",
            "risk",
            "combo",
        ],
        help=(
            "Which hotspot definition is used "
            "for counts, clusters, persistence."
        ),
    )
    
    ap.add_argument(
        "--hotspot-abs",
        type=float,
        default=None,
        help="Absolute hotspot: |Δs|>=X (mm/yr).",
    )
    
    ap.add_argument(
        "--risk-min",
        type=float,
        default=None,
        help="Risk hotspot: P(|s|>=T)>=p.",
    )
    
    ap.add_argument(
        "--exposure",
        type=str,
        default=None,
        help=(
            "Optional exposure CSV "
            "(sample_idx,exposure)."
        ),
    )
    
    ap.add_argument(
        "--exposure-col",
        type=str,
        default="exposure",
    )
    
    ap.add_argument(
        "--exposure-default",
        type=float,
        default=1.0,
    )
    ap.add_argument(
        "--timeline-mode",
        type=str,
        default="current",
        choices=["current", "ever", "new"],
        help=(
            "Timeline bars show: "
            "current hotspots (per-year), "
            "ever hotspots (union up to year), "
            "or new hotspots (yearly additions)."
        ),
    )
    ap.add_argument(
        "--timeline-overlay-current",
        type=str,
        default="true",
        help=(
            "Overlay dashed line for current hotspots "
            "on top of ever/new bars."
        ),
    )
    ap.add_argument(
        "--cluster-rank",
        type=str,
        default="delta",
        choices=["delta", "risk"],
        help="Rank clusters by |Δs| or risk score.",
    )
    
    ap.add_argument(
        "--add-persistence",
        action="store_true",
        help="Add a persistence map panel.",
    )

    ap.add_argument(
        "--persistence-mode",
        type=str,
        default="fraction",
        choices=["fraction", "count"],
        help=(
            "Persistence definition: "
            "fraction (0..1) or count (0..N years)."
        ),
    )
     
    ap.add_argument(
        "--boundary",
        type=str,
        default=None,
        help="Optional boundary GeoJSON/SHP.",
    )

    ap.add_argument(
        "--ns-boundary",
        type=str,
        default=None,
        help="Optional Nansha boundary GeoJSON/SHP.",
    )

    ap.add_argument(
        "--zh-boundary",
        type=str,
        default=None,
        help="Optional Zhongshan boundary GeoJSON/SHP.",
    )

    ap.add_argument(
        "--boundary-name-field",
        type=str,
        default=None,
    )
    
    ap.add_argument(
        "--ns-future-b",
        type=str,
        default=None,
    )
    
    ap.add_argument(
        "--zh-future-b",
        type=str,
        default=None,
    )
    
    ap.add_argument(
        "--compare-metric",
        type=str,
        default="risk",
        choices=["risk", "delta", "prob"],
    )
    
    ap.add_argument(
        "--add-compare",
        action="store_true",
        help="Add scenario A vs B panel if B exists.",
    )
    utils.add_plot_text_args(
        ap,
        default_out="hotspot_analytics",
    )


def plot_hotspot_analytics_main(
    argv: Optional[List[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        prog="plot-hotspot-analytics",
        description="Decision-maker hotspot analytics.",
    )
    _add_args(ap)
    args = ap.parse_args(argv)

    show_legend = utils.str_to_bool(args.show_legend, default=True)
    show_title = utils.str_to_bool(args.show_title, default=True)
    show_pt = utils.str_to_bool(args.show_panel_titles, default=True)
    
    overlay_cur = utils.str_to_bool(
        args.timeline_overlay_current,
        default=True,
    )

    years = [int(y) for y in (args.years or [])]
    if not years:
        raise SystemExit("--years must be non-empty")
    focus_year = int(args.focus_year) if args.focus_year is not None \
        else int(max(years))

    cities0 = utils.resolve_cities(args)
    if not cities0:
        cities0 = [_CITY_A, _CITY_B]

    want_a = _CITY_A in cities0
    want_b = _CITY_B in cities0
    if not want_a and not want_b:
        want_a = True
        want_b = True
        
    expo = None
    if args.exposure:
        expo = _load_exposure(
            args.exposure,
            col=str(args.exposure_col),
        )
    
    resolved: List[Dict[str, Any]] = []
    if want_a:
        resolved.append(
            _resolve_city(
                city=_CITY_A,
                src=args.ns_src,
                eval_csv=args.ns_eval,
                future_csv=args.ns_future,
                split=str(args.split),
            )
        )
    if want_b:
        resolved.append(
            _resolve_city(
                city=_CITY_B,
                src=args.zh_src,
                eval_csv=args.zh_eval,
                future_csv=args.zh_future,
                split=str(args.split),
            )
        )

    cities: List[CityHotspotData] = []
    for rc in resolved:
        cities.append(
            build_hotspot_tables(
                city=str(rc["name"]),
                color=str(rc["color"]),
                eval_df=rc["eval_df"],
                fut_df=rc["future_df"],
                years=years,
                base_year=int(args.base_year),
                subsidence_kind=str(args.subsidence_kind),
                hotspot_q=float(args.hotspot_quantile),
                risk_threshold=float(args.risk_threshold),
                use_abs_risk=utils.str_to_bool(args.risk_abs, default=True),
                hotspot_rule=str(args.hotspot_rule),
                hotspot_abs=args.hotspot_abs,
                risk_min=args.risk_min,
                exposure_df=expo,
                exposure_col=str(args.exposure_col),
                exposure_default=float(args.exposure_default),

            )
        )
    # gdf = _try_load_boundaries(args.boundary)
    gdf_map = _resolve_boundary_map(
        boundary=args.boundary,
        ns_boundary=args.ns_boundary,
        zh_boundary=args.zh_boundary,
        boundary_name_field=args.boundary_name_field,
        cities=[c.city for c in cities],
    )
    
    plot_hotspot_analytics(
        cities=cities,
        focus_year=int(focus_year),
        grid_res=int(args.grid_res),
        clip=float(args.clip),
        cmap_metric=str(args.cmap_metric),
        cmap_prob=str(args.cmap_prob),
        persistence_mode=str(args.persistence_mode),
        cluster_top=int(args.cluster_top),
        risk_threshold=float(args.risk_threshold),
        show_legend=bool(show_legend),
        show_title=bool(show_title),
        show_panel_titles=bool(show_pt),
        title=args.title,
        out=str(args.out),
        out_points=args.out_points,
        out_years=args.out_years,
        out_clusters=args.out_clusters,
        dpi=int(args.dpi),
        font=int(args.font),
        cluster_rank=str(args.cluster_rank),
        add_persistence=bool(args.add_persistence),
        hotspot_rule=str(args.hotspot_rule),
        hotspot_abs=args.hotspot_abs,
        risk_min=args.risk_min,
        # boundary=args.boundary,
        timeline_mode=str(args.timeline_mode),
        timeline_overlay_current=bool(overlay_cur),
        add_compare=bool(args.add_compare),
        ns_future_b=args.ns_future_b,
        zh_future_b=args.zh_future_b,
        compare_metric=str(args.compare_metric),
        gdf=gdf_map,

    )


def main(argv: Optional[List[str]] = None) -> None:
    plot_hotspot_analytics_main(argv)


if __name__ == "__main__":
    main()