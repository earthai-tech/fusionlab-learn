
from __future__ import annotations
from typing import ( 
    Dict, Iterable, Tuple, 
    Any, Optional, Literal, 
    Mapping
)
import os 
import json 
import math 
import numpy as np
import pandas as pd

from dataclasses import dataclass
import warnings


ShiftMode = Literal["none", "min", "mean", "value"]
Mode = Literal["add", "overwrite"]


DEFAULT_SUBS_UNIT_TO_SI = 1e-3


def _norm_unit(unit: str | None) -> str:
    u = (unit or "").strip().lower()
    if u in ("m", "meter", "meters"):
        return "m"
    if u in ("mm", "millimeter", "millimeters"):
        return "mm"
    return u


def _unit_factor(
    from_unit: str | None,
    to_unit: str | None,
) -> float:
    fu = _norm_unit(from_unit)
    tu = _norm_unit(to_unit)

    if fu == tu:
        return 1.0

    if fu == "m" and tu == "mm":
        return 1000.0

    if fu == "mm" and tu == "m":
        return 1e-3

    raise ValueError(
        "Unsupported unit conversion: "
        f"{from_unit!r} -> {to_unit!r}."
    )


def _as_float(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(
            f"Cannot convert to float: {x!r}."
        ) from e

    if not math.isfinite(v):
        raise ValueError(
            f"Non-finite value: {v!r}."
        )
    return float(v)


def _as_pos_float(
    x: object,
    *,
    default: float,
) -> float:
    v = _as_float(x)
    if v <= 0.0:
        return float(default)
    return float(v)


def _select_target_cols(
    df: pd.DataFrame,
    *,
    base: str,
    columns: Iterable[str] | None = None,
) -> list[str]:
    if columns is not None:
        return [c for c in columns if c in df.columns]

    b = (base or "").strip()
    if not b:
        return []

    pref = b + "_"
    cols: list[str] = []
    for c in df.columns:
        cs = str(c)
        if cs == b or cs.startswith(pref):
            cols.append(cs)
    return cols


def convert_target_units_df(
    df: pd.DataFrame | None,
    *,
    base: str,
    from_unit: str = "m",
    to_unit: str = "mm",
    mode: Mode = "overwrite",
    suffix: str = "_mm",
    columns: Iterable[str] | None = None,
    unit_col: str | None = None,
    copy_df: bool = True,
    overwrite_cols: bool = False,
    strict: bool = False,
) -> pd.DataFrame | None:
    """
    Convert target-like columns between "m" and "mm".

    Selected columns:
    - `base`
    - `base + "_*"` (quantiles, intervals, ...)

    If mode="add", new columns use `suffix`.
    """
    if df is None or getattr(df, "empty", True):
        return df

    cols = _select_target_cols(
        df,
        base=base,
        columns=columns,
    )
    if not cols:
        return df

    m = (mode or "overwrite").strip().lower()
    if m not in ("add", "overwrite"):
        raise ValueError(
            "mode must be 'add' or 'overwrite'. "
            f"Got {mode!r}."
        )

    # If user asked to "add" but suffix is empty,
    # this is effectively overwrite.
    if m == "add" and (suffix == ""):
        m = "overwrite"

    factor = _unit_factor(from_unit, to_unit)
    out = df.copy() if copy_df else df

    for c in cols:
        dst = c if m == "overwrite" else (c + suffix)
        if (dst in out.columns) and (m == "add"):
            if not overwrite_cols:
                continue

        ser = out[c]
        vals = pd.to_numeric(ser, errors="coerce")
        if strict and (vals.notna().sum() == 0):
            raise ValueError(
                f"Cannot convert column to numeric: {c!r}."
            )
        out[dst] = vals.astype(float) * float(factor)

    ucol = unit_col or (str(base).strip() + "_unit")
    out[ucol] = _norm_unit(to_unit)

    return out


def subs_unit_to_si(
    cfg: Mapping[str, object] | None = None,
    *,
    default: float = DEFAULT_SUBS_UNIT_TO_SI,
    units_prov_key: str = "units_provenance",
    stage1_key: str = "subs_unit_to_si_applied_stage1",
    cfg_key: str = "SUBS_UNIT_TO_SI",
) -> float:
    """
    Subsidence unit->SI factor (to meters).

    Priority:
    1) cfg[units_prov_key][stage1_key]
    2) cfg[cfg_key]
    3) default
    """
    if cfg is None:
        return float(default)

    prov = cfg.get(units_prov_key)
    if isinstance(prov, Mapping):
        v = prov.get(stage1_key)
        if v is not None:
            return _as_pos_float(v, default=default)

    v2 = cfg.get(cfg_key)
    if v2 is not None:
        return _as_pos_float(v2, default=default)

    return float(default)


def subs_native_unit(
    cfg: Mapping[str, object] | None = None,
    *,
    default: str = "mm",
) -> str:
    """
    Infer the "native" subsidence unit from cfg.

    - unit_to_si ~= 1e-3 -> "mm"
    - unit_to_si ~= 1.0  -> "m"
    """
    if cfg is None:
        return _norm_unit(default) or "mm"

    u2si = subs_unit_to_si(cfg)
    if abs(u2si - 1e-3) <= 1e-10:
        return "mm"
    if abs(u2si - 1.0) <= 1e-10:
        return "m"

    return _norm_unit(default) or "mm"


def add_subsidence_mm_columns(
    df: pd.DataFrame | None,
    cfg: Mapping[str, object] | None = None,
    *,
    base: str = "subsidence",
    columns: Iterable[str] | None = None,
    suffix: str = "_mm",
    unit_col: str | None = None,
    copy_df: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame | None:
    """
    Add (or overwrite) subsidence columns in millimeters.

    Always assumes the current values are in meters.
    """
    return convert_target_units_df(
        df,
        base=base,
        from_unit="m",
        to_unit="mm",
        mode="add",
        suffix=suffix,
        columns=columns,
        unit_col=unit_col,
        copy_df=copy_df,
        overwrite_cols=overwrite,
        strict=False,
    )


def add_subsidence_native_unit_columns(
    df: pd.DataFrame | None,
    cfg: Mapping[str, object] | None = None,
    *,
    base: str = "subsidence",
    columns: Iterable[str] | None = None,
    suffix: str = "_native",
    unit_col: str | None = None,
    copy_df: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame | None:
    """
    Add columns in the cfg-inferred native unit.

    If cfg says the native unit was meters, this becomes
    a no-op aside from optional unit_col.
    """
    to_unit = subs_native_unit(cfg)
    return convert_target_units_df(
        df,
        base=base,
        from_unit="m",
        to_unit=to_unit,
        mode="add",
        suffix=suffix,
        columns=columns,
        unit_col=unit_col,
        copy_df=copy_df,
        overwrite_cols=overwrite,
        strict=False,
    )


def finalize_si_scaling_kwargs(
    scaling_kwargs: dict[str, Any],
    *,
    subs_in_si: bool,
    head_in_si: bool,
    thickness_in_si: bool,
    force_identity_affine_if_si: bool = True,
    warn: bool = True,
) -> dict[str, Any]:
    """
    Prevent double SI conversion in GeoPrior scaling_kwargs.
    """
    kw = dict(scaling_kwargs)

    def _f(name: str, default: float) -> float:
        v = kw.get(name, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    kw["subs_unit_to_si"] = _f("subs_unit_to_si", 1.0)
    kw["head_unit_to_si"] = _f("head_unit_to_si", 1.0)
    kw["thickness_unit_to_si"] = _f("thickness_unit_to_si", 1.0)

    kw.setdefault("subs_scale_si", None)
    kw.setdefault("subs_bias_si", None)
    kw.setdefault("head_scale_si", None)
    kw.setdefault("head_bias_si", None)
    kw.setdefault("H_scale_si", None)
    kw.setdefault("H_bias_si", None)

    def _fix(
        *,
        already_si: bool,
        unit_key: str,
        scale_key: str,
        bias_key: str,
        name: str,
    ) -> None:
        if not already_si:
            if kw.get(scale_key) is None:
                kw[scale_key] = 1.0
            if kw.get(bias_key) is None:
                kw[bias_key] = 0.0
            return

        cur = _f(unit_key, 1.0)
        if warn and (cur != 1.0):
            warnings.warn(
                "[GeoPrior SI] "
                f"{name} already SI but {unit_key}={cur}. "
                f"Setting {unit_key}=1.0.",
                RuntimeWarning,
            )
        kw[unit_key] = 1.0

        if force_identity_affine_if_si:
            kw[scale_key] = 1.0
            kw[bias_key] = 0.0
        else:
            if kw.get(scale_key) is None:
                kw[scale_key] = 1.0
            if kw.get(bias_key) is None:
                kw[bias_key] = 0.0

    _fix(
        already_si=subs_in_si,
        unit_key="subs_unit_to_si",
        scale_key="subs_scale_si",
        bias_key="subs_bias_si",
        name="subsidence",
    )
    _fix(
        already_si=head_in_si,
        unit_key="head_unit_to_si",
        scale_key="head_scale_si",
        bias_key="head_bias_si",
        name="head/GWL",
    )
    _fix(
        already_si=thickness_in_si,
        unit_key="thickness_unit_to_si",
        scale_key="H_scale_si",
        bias_key="H_bias_si",
        name="thickness/H_field",
    )

    return kw

# Backward-compat alias (no duplicate body)
finalize_si_affines_and_units = finalize_si_scaling_kwargs


def infer_utm_epsg_from_lonlat(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
) -> int:
    """
    Infer a UTM EPSG code from lon/lat (EPSG:4326).

    Uses standard UTM zoning:
      zone = floor((lon + 180)/6) + 1
      EPSG = 32600 + zone (north), 32700 + zone (south)
    """
    lon = float(np.nanmean(np.asarray(lon_deg, dtype=float)))
    lat = float(np.nanmean(np.asarray(lat_deg, dtype=float)))

    zone = int(np.floor((lon + 180.0) / 6.0) + 1.0)
    zone = max(1, min(zone, 60))
    return (32600 + zone) if (lat >= 0.0) else (32700 + zone)


def lonlat_to_utm_m(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    *,
    src_epsg: int = 4326,
    target_epsg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Convert lon/lat degrees to UTM meters using pyproj.

    Returns
    -------
    x_m, y_m, target_epsg
    """
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)

    if target_epsg is None:
        target_epsg = infer_utm_epsg_from_lonlat(lon, lat)

    try:
        from pyproj import Transformer
    except Exception as e:
        raise ImportError(
            "pyproj is required for lon/lat -> UTM conversion. "
            "Install with: pip install pyproj"
        ) from e

    tr = Transformer.from_crs(
        f"EPSG:{src_epsg}",
        f"EPSG:{target_epsg}",
        always_xy=True,
    )
    x_m, y_m = tr.transform(lon, lat)
    return np.asarray(x_m, float), np.asarray(y_m, float), int(target_epsg)


def _shift_1d(
    a: np.ndarray,
    mode: ShiftMode,
    value: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Shift array by min/mean/custom value (translation only, not scaling).
    """
    a = np.asarray(a, dtype=float)
    if mode == "none":
        return a, 0.0
    if mode == "min":
        s = float(np.nanmin(a))
        return a - s, s
    if mode == "mean":
        s = float(np.nanmean(a))
        return a - s, s
    if mode == "value":
        if value is None:
            raise ValueError("shift mode 'value' requires shift_value.")
        s = float(value)
        return a - s, s
    raise ValueError(f"Unknown shift mode: {mode!r}")


@dataclass
class CoordsPack:
    coords: np.ndarray                  # (B, H, 3) float32
    coord_mins: Dict[str, float]        # {"t":..., "x":..., "y":...} (original mins/means used)
    coord_ranges: Dict[str, float]      # {"t":..., "x":..., "y":...} (after shifting; ranges)
    meta: Dict[str, Any]                # epsg, shift modes, etc.


def make_txy_coords(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    time_shift: ShiftMode = "min",
    xy_shift: ShiftMode = "min",
    time_shift_value: Optional[float] = None,
    x_shift_value: Optional[float] = None,
    y_shift_value: Optional[float] = None,
    dtype: str = "float32",
) -> CoordsPack:
    """
    Build coords tensor (t, x, y) with OPTIONAL shifting (translation only).

    This is designed for your "not normalized" workflow:
      - You keep SI units (years and meters),
      - but you avoid feeding huge UTM magnitudes (e.g. 3e5, 2.5e6)
        into coord MLPs by shifting x,y (and optionally t).

    Notes
    -----
    - This does NOT min-max scale to [0,1]. It only translates.
    - Returning coord_mins/coord_ranges is still useful for logging/debug.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.shape != x.shape or t.shape != y.shape:
        raise ValueError(f"t, x, y must have same shape. Got {t.shape}, {x.shape}, {y.shape}")

    t2, t0 = _shift_1d(t, time_shift, time_shift_value)
    x2, x0 = _shift_1d(x, xy_shift, x_shift_value)
    y2, y0 = _shift_1d(y, xy_shift, y_shift_value)

    # ranges after shifting (safe even if you don't "use" them in model)
    t_rng = float(np.nanmax(t2) - np.nanmin(t2))
    x_rng = float(np.nanmax(x2) - np.nanmin(x2))
    y_rng = float(np.nanmax(y2) - np.nanmin(y2))

    coords = np.concatenate([t2[..., None], x2[..., None], y2[..., None]], axis=-1).astype(dtype)

    return CoordsPack(
        coords=coords,
        coord_mins={"t": t0, "x": x0, "y": y0},
        coord_ranges={"t": t_rng, "x": x_rng, "y": y_rng},
        meta={
            "time_shift": time_shift,
            "xy_shift": xy_shift,
        },
    )




def detect_subsidence_mode(
    df: pd.DataFrame,
    *,
    rate_col: str = "subsidence",
    cum_col: str = "subsidence_cum",
    time_col: str = "year",
    group_cols: Iterable[str] = ("longitude", "latitude", "city"),
    tol_rel: float = 0.05,
    min_points: int = 3,
    max_groups: int = 200,
    random_state: int = 42,
) -> Dict:
    """
    Infer whether subsidence columns represent 'rate', 'cumulative',
    or an inconsistent pair.

    Strategy
    --------
    If both columns exist, check whether:
        Δcum(t_i) ≈ rate(t_i) * Δt_i
    per group (lon/lat[/city]) over i>=1. This is baseline-invariant.

    Returns
    -------
    dict with keys:
        mode: 'cumulative', 'rate', 'pair-consistent', 'unknown'
        details: diagnostics (errors, counts, etc.)
    """
    cols = set(df.columns)
    has_rate = rate_col in cols
    has_cum = cum_col in cols

    if has_cum and not has_rate:
        return {"mode": "cumulative", "details": {"reason": "cum_col present only"}}
    if has_rate and not has_cum:
        return {"mode": "rate", "details": {"reason": "rate_col present only"}}
    if not (has_rate and has_cum):
        return {"mode": "unknown", "details": {"reason": "neither column present"}}

    gcols = [c for c in group_cols if c in cols]
    if not gcols:
        return {
            "mode": "unknown",
            "details": {"reason": "no valid group_cols found in df"},
        }

    # sample groups for speed
    groups = df.groupby(gcols, sort=False)
    keys = list(groups.groups.keys())
    if not keys:
        return {"mode": "unknown", "details": {"reason": "no groups"}}

    rng = np.random.default_rng(random_state)
    if len(keys) > max_groups:
        keys = [keys[i] for i in rng.choice(len(keys), size=max_groups, replace=False)]

    rel_errs = []
    used_groups = 0
    used_points = 0

    for k in keys:
        g = groups.get_group(k).sort_values(time_col)
        if len(g) < min_points:
            continue

        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(float)
        r = pd.to_numeric(g[rate_col], errors="coerce").to_numpy(float)
        c = pd.to_numeric(g[cum_col], errors="coerce").to_numpy(float)

        m = np.isfinite(t) & np.isfinite(r) & np.isfinite(c)
        t, r, c = t[m], r[m], c[m]
        if t.size < min_points:
            continue

        dt = np.diff(t)
        dc = np.diff(c)
        r_next = r[1:]

        m2 = np.isfinite(dt) & np.isfinite(dc) & np.isfinite(r_next) & (dt != 0)
        if m2.sum() < (min_points - 1):
            continue

        dt, dc, r_next = dt[m2], dc[m2], r_next[m2]

        pred_dc = r_next * dt
        denom = np.maximum(np.mean(np.abs(pred_dc)), 1e-12)
        rel = np.mean(np.abs(dc - pred_dc)) / denom

        rel_errs.append(rel)
        used_groups += 1
        used_points += int(m2.sum())

    if used_groups == 0:
        return {"mode": "unknown", "details": {"reason": "insufficient data per group"}}

    med = float(np.median(rel_errs))
    out = {
        "median_rel_error": med,
        "mean_rel_error": float(np.mean(rel_errs)),
        "n_groups": used_groups,
        "n_points": used_points,
        "tol_rel": tol_rel,
    }

    if med <= tol_rel:
        return {"mode": "pair-consistent", "details": out}
    return {"mode": "unknown", "details": out}

def rate_to_cumulative(
    df: pd.DataFrame,
    *,
    rate_col: str = "subsidence",
    cum_col: str = "subsidence_cum",
    time_col: str = "year",
    group_cols: Iterable[str] = ("longitude", "latitude", "city"),
    initial: str = "first_equals_rate_dt",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Build cumulative displacement from a rate series.

    Assumption
    ----------
    rate(t_i) represents the rate over the interval (t_{i-1}, t_i].
    Then:
        cum(t_i) = cum(t_{i-1}) + rate(t_i) * dt_i

    Parameters
    ----------
    initial:
        - 'zero': cum at first time is 0
        - 'first_equals_rate_dt': cum(t0) = rate(t0) * dt_ref
          where dt_ref is median dt in that group (fallback 1).

    Returns
    -------
    DataFrame with cum_col added/overwritten.
    """
    out = df if inplace else df.copy()
    cols = set(out.columns)
    gcols = [c for c in group_cols if c in cols]
    if not gcols:
        raise ValueError("No valid group_cols found in df.")

    out[cum_col] = np.nan

    for _, gidx in out.groupby(gcols, sort=False).groups.items():
        g = out.loc[gidx].sort_values(time_col)
        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(float)
        r = pd.to_numeric(g[rate_col], errors="coerce").to_numpy(float)

        m = np.isfinite(t) & np.isfinite(r)
        t, r = t[m], r[m]
        if t.size == 0:
            continue

        dt = np.diff(t)
        dt_ref = float(np.median(dt[np.isfinite(dt) & (dt != 0)])) if t.size > 1 else 1.0
        if not np.isfinite(dt_ref) or dt_ref == 0:
            dt_ref = 1.0

        c = np.zeros_like(r, dtype=float)
        if initial == "zero":
            c[0] = 0.0
        elif initial == "first_equals_rate_dt":
            c[0] = r[0] * dt_ref
        else:
            raise ValueError("initial must be 'zero' or 'first_equals_rate_dt'.")

        if r.size > 1:
            dt_i = np.diff(t)
            dt_i = np.where((~np.isfinite(dt_i)) | (dt_i == 0), dt_ref, dt_i)
            c[1:] = c[0] + np.cumsum(r[1:] * dt_i)

        # write back aligned to original rows in this group
        out.loc[g.index[m], cum_col] = c

    return out

def cumulative_to_rate(
    df: pd.DataFrame,
    *,
    cum_col: str = "subsidence_cum",
    rate_col: str = "subsidence",
    time_col: str = "year",
    group_cols: Iterable[str] = ("longitude", "latitude", "city"),
    first: str = "cum_over_dtref",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Recover a rate series from cumulative displacement.

    rate(t_i) = (cum(t_i) - cum(t_{i-1})) / dt_i  for i>=1

    first:
        - 'nan': first rate is NaN
        - 'cum_over_dtref': rate(t0) = cum(t0)/dt_ref (dt_ref median dt)

    Returns
    -------
    DataFrame with rate_col added/overwritten.
    """
    out = df if inplace else df.copy()
    cols = set(out.columns)
    gcols = [c for c in group_cols if c in cols]
    if not gcols:
        raise ValueError("No valid group_cols found in df.")

    out[rate_col] = np.nan

    for _, gidx in out.groupby(gcols, sort=False).groups.items():
        g = out.loc[gidx].sort_values(time_col)
        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(float)
        c = pd.to_numeric(g[cum_col], errors="coerce").to_numpy(float)

        m = np.isfinite(t) & np.isfinite(c)
        t, c = t[m], c[m]
        if t.size == 0:
            continue

        dt = np.diff(t)
        dt_ref = float(np.median(dt[np.isfinite(dt) & (dt != 0)])) if t.size > 1 else 1.0
        if not np.isfinite(dt_ref) or dt_ref == 0:
            dt_ref = 1.0

        r = np.zeros_like(c, dtype=float)

        if first == "nan":
            r[0] = np.nan
        elif first == "cum_over_dtref":
            r[0] = c[0] / dt_ref
        else:
            raise ValueError("first must be 'nan' or 'cum_over_dtref'.")

        if c.size > 1:
            dt_i = np.diff(t)
            dt_i = np.where((~np.isfinite(dt_i)) | (dt_i == 0), dt_ref, dt_i)
            r[1:] = np.diff(c) / dt_i

        out.loc[g.index[m], rate_col] = r

    return out



# ---------------------------------------------------------------------
# Evaluation payload unit conversion
# ---------------------------------------------------------------------

_TIME_UNIT_TO_SECONDS: dict[str, float] = {
    "unitless": 1.0,
    "step": 1.0,
    "index": 1.0,
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "day": 86400.0,
    "days": 86400.0,
    "week": 7.0 * 86400.0,
    "weeks": 7.0 * 86400.0,
    # Mean tropical year (365.2425 d)
    "year": 31556952.0,
    "years": 31556952.0,
    "yr": 31556952.0,
    "month": 31556952.0 / 12.0,
    "months": 31556952.0 / 12.0,
}


def _cfg_to_mapping(cfg: object | None) -> Mapping[str, object]:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return cfg
    try:
        return dict(vars(cfg))
    except Exception:
        return {}


def _norm_time_units(time_units: str | None) -> str:
    s = (time_units or "").strip().lower()
    if s in ("", "none", "null"):
        return "unitless"
    if s in ("t", "time", "times"):
        return "unitless"
    if s in ("yrs",):
        return "yr"
    return s


def _seconds_per_time_unit(time_units: str | None) -> float:
    key = _norm_time_units(time_units)
    if key not in _TIME_UNIT_TO_SECONDS:
        keys = sorted(_TIME_UNIT_TO_SECONDS.keys())
        raise ValueError(
            f"Unsupported time_units={time_units!r}. "
            f"Supported: {keys}"
        )
    return float(_TIME_UNIT_TO_SECONDS[key])


def _is_number(x: object) -> bool:
    if isinstance(x, (bool,)):
        return False
    return isinstance(x, (int, float, np.number))


def _scale_obj(obj: object, factor: float) -> object:
    """
    Multiply numbers inside common JSON-like containers.

    Supports scalar numbers, dicts of numbers, and lists/tuples of
    numbers. Non-numeric entries are left unchanged.
    """
    if _is_number(obj):
        return float(obj) * float(factor)

    if isinstance(obj, dict):
        return {k: _scale_obj(v, factor) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [ _scale_obj(v, factor) for v in obj ]

    return obj


def _set_scaled(d: dict[str, Any], key: str, factor: float) -> None:
    if key not in d:
        return
    d[key] = _scale_obj(d[key], factor)


def convert_eval_payload_units(
    payload: Mapping[str, Any],
    cfg: Mapping[str, object] | object | None = None,
    *,
    mode: Literal["si", "interpretable"] = "si",
    scope: Literal["all", "subsidence", "physics"] = "all",
    savefile: str | None = None,
    fmt: Literal["json"] = "json",
    indent: int = 2,
    copy_payload: bool = True,
) -> dict[str, Any]:
    """
    Convert GeoPriorSubsNet evaluation-payload units for reporting.

    This is a **post-processing** helper meant for stage-2 evaluation
    JSON payloads (e.g. ``geoprior_eval_phys_<timestamp>.json``).

    Parameters
    ----------
    payload : Mapping[str, Any]
        The evaluation payload dict. It is expected to contain sections
        like ``metrics_evaluate``, ``point_metrics``, ``per_horizon``,
        ``interval_calibration`` and ``censor_stratified``.
    cfg : mapping or module, optional
        The experiment config (e.g. ``config`` module or ``globals()``).
        We use:
        - ``SUBS_UNIT_TO_SI`` (or stage-1 provenance)
        - ``TIME_UNITS`` (e.g. "year")
    mode : {"si", "interpretable"}, default="si"
        - "si": leave values untouched (current behaviour).
        - "interpretable": convert selected subsidence/physics metrics
          from SI into native units implied by ``SUBS_UNIT_TO_SI``.
    scope : {"all", "subsidence", "physics"}, default="all"
        Which parts to convert when ``mode="interpretable"``.
        - "subsidence": convert only subsidence metrics (MAE/MSE/
          sharpness) to the native unit (e.g. mm).
        - "physics": convert only physics residual rates where units are
          unambiguous (currently ``epsilon_cons_raw`` and
          ``epsilon_gw_raw``).
        - "all": do both.
    savefile : str, optional
        If provided, write the converted payload to this path.
    fmt : {"json"}, default="json"
        Output format when ``savefile`` is provided.
    indent : int, default=2
        JSON indentation.
    copy_payload : bool, default=True
        If True, operate on a deep copy of ``payload``. If False,
        convert in-place (dangerous).

    Returns
    -------
    dict
        Converted payload as a plain ``dict``.

    Notes
    -----
    Subsidence conversions:
    - linear metrics (MAE, sharpness) scale by ``1/SUBS_UNIT_TO_SI``.
    - squared metrics (MSE) scale by ``(1/SUBS_UNIT_TO_SI)^2``.

    Physics conversions (when enabled):
    - ``epsilon_cons_raw`` is treated as a **rate** in [m/s] and is
      converted to ``<subs_native_unit>/<TIME_UNITS>`` (e.g. mm/yr).
    - ``epsilon_gw_raw`` is treated as a **rate** in [1/s] and is
      converted to ``1/<TIME_UNITS>`` (e.g. 1/yr).

    The helper records unit provenance under ``payload["units"]``.
    """
    if payload is None:
        raise ValueError("payload must be a mapping, got None.")

    out: dict[str, Any]
    if copy_payload:
        import copy as _copy
        out = _copy.deepcopy(dict(payload))
    else:
        out = dict(payload)  # shallow

    cfg_m = _cfg_to_mapping(cfg)

    # -----------------------------------------------------------------
    # Resolve unit factors
    # -----------------------------------------------------------------
    u2si = subs_unit_to_si(cfg_m)
    native_u = subs_native_unit(cfg_m)
    subs_from_si = 1.0 / float(u2si)

    time_units = (
        cfg_m.get("TIME_UNITS")
        or out.get("time_units")
        or out.get("TIME_UNITS")
        or "year"
    )
    sec_per = _seconds_per_time_unit(str(time_units))

    # -----------------------------------------------------------------
    # Always attach provenance (even if mode="si")
    # -----------------------------------------------------------------
    out.setdefault("units", {})
    if not isinstance(out["units"], dict):
        out["units"] = {}

    out["units"].update({
        # Core provenance the user asked for
        "subs_unit_to_si": float(u2si),
        "subs_factor_si_to_real": float(subs_from_si),
        "subs_metrics_unit": (
            native_u if mode == "interpretable" else "m"
        ),
        # Extra helpful metadata
        "time_units": str(time_units),
        "seconds_per_time_unit": float(sec_per),
    })

    if mode != "interpretable":
        if savefile:
            _write_payload(out, savefile, fmt=fmt, indent=indent)
        return out

    # -----------------------------------------------------------------
    # 1) Subsidence metrics (SI m -> native unit, e.g. mm)
    # -----------------------------------------------------------------
    do_subs = scope in ("all", "subsidence")
    if do_subs:
        # metrics_evaluate: subs_pred_* metrics
        me = out.get("metrics_evaluate")
        if isinstance(me, dict):
            for k, v in list(me.items()):
                if not _is_number(v):
                    continue
                if not k.startswith("subs_pred_"):
                    continue

                lk = k.lower()
                if "coverage" in lk or "r2" in lk:
                    continue

                if "mse" in lk:
                    me[k] = float(v) * (subs_from_si ** 2)
                elif ("mae" in lk) or ("rmse" in lk) or ("sharpness" in lk):
                    me[k] = float(v) * subs_from_si

        # point_metrics: (mae, mse, r2)
        pm = out.get("point_metrics")
        if isinstance(pm, dict):
            if "mae" in pm:
                _set_scaled(pm, "mae", subs_from_si)
            if "mse" in pm:
                _set_scaled(pm, "mse", subs_from_si ** 2)
            if "rmse" in pm:
                _set_scaled(pm, "rmse", subs_from_si)

        # per_horizon: mae/mse/rmse dicts with keys H1/H2/...
        ph = out.get("per_horizon")
        if isinstance(ph, dict):
            for metric_name, series in ph.items():
                if not isinstance(series, dict):
                    continue
                lk = str(metric_name).lower()
                if lk in ("mae", "rmse", "sharpness"):
                    ph[metric_name] = _scale_obj(series, subs_from_si)
                elif lk == "mse":
                    ph[metric_name] = _scale_obj(series, subs_from_si ** 2)

        # interval_calibration: only *_phys sharpness is physical
        ic = out.get("interval_calibration")
        if isinstance(ic, dict):
            for k, v in list(ic.items()):
                if not _is_number(v):
                    continue
                lk = k.lower()
                if ("sharpness" in lk) and lk.endswith("_phys"):
                    ic[k] = float(v) * subs_from_si

        # censor_stratified: mae_* values
        cs = out.get("censor_stratified")
        if isinstance(cs, dict):
            for k, v in list(cs.items()):
                if not _is_number(v):
                    continue
                lk = k.lower()
                if lk.startswith("mae_") or ("mae" == lk):
                    cs[k] = float(v) * subs_from_si
                elif lk.startswith("mse_") or ("mse" == lk):
                    cs[k] = float(v) * (subs_from_si ** 2)

    # -----------------------------------------------------------------
    # 2) Physics residual rates (SI -> interpretable)
    # -----------------------------------------------------------------
    do_phys = scope in ("all", "physics")
    if do_phys:
        unit_rate = f"{native_u}/{_norm_time_units(str(time_units))}"
        # Prefer converting *_raw variants (scaled eps are often dimensionless)
        me = out.get("metrics_evaluate")
        if isinstance(me, dict):
            if "epsilon_cons_raw" in me and _is_number(me["epsilon_cons_raw"]):
                me["epsilon_cons_raw"] = (
                    float(me["epsilon_cons_raw"])
                    * subs_from_si
                    * sec_per
                )
                out["units"]["epsilon_cons_raw_unit"] = unit_rate

            if "epsilon_gw_raw" in me and _is_number(me["epsilon_gw_raw"]):
                me["epsilon_gw_raw"] = float(me["epsilon_gw_raw"]) * sec_per
                out["units"]["epsilon_gw_raw_unit"] = (
                    f"1/{_norm_time_units(str(time_units))}"
                )

        pdg = out.get("physics_diagnostics")
        if isinstance(pdg, dict):
            if "epsilon_cons_raw" in pdg and _is_number(pdg["epsilon_cons_raw"]):
                pdg["epsilon_cons_raw"] = (
                    float(pdg["epsilon_cons_raw"])
                    * subs_from_si
                    * sec_per
                )
                out["units"]["epsilon_cons_raw_unit"] = unit_rate

            if "epsilon_gw_raw" in pdg and _is_number(pdg["epsilon_gw_raw"]):
                pdg["epsilon_gw_raw"] = float(pdg["epsilon_gw_raw"]) * sec_per
                out["units"]["epsilon_gw_raw_unit"] = (
                    f"1/{_norm_time_units(str(time_units))}"
                )

    if savefile:
        _write_payload(out, savefile, fmt=fmt, indent=indent)

    return out


def _json_default(x: object) -> object:
    # numpy scalars -> python scalars
    if isinstance(x, np.generic):
        return x.item()
    # numpy arrays -> lists
    if isinstance(x, np.ndarray):
        return x.tolist()
    # pandas scalars
    try:
        import pandas as _pd  # local
        if isinstance(x, (_pd.Timestamp,)):
            return x.isoformat()
    except Exception:
        pass
    # fallback: string
    return str(x)


def _write_payload(
    payload: Mapping[str, Any],
    savefile: str,
    *,
    fmt: str = "json",
    indent: int = 2,
) -> None:
    """
    Write a payload to disk.

    Parameters
    ----------
    payload : mapping
        JSON-serializable mapping.
    savefile : str
        Output file path.
    fmt : {"json"}, default="json"
        Output format.
    indent : int, default=2
        JSON indentation.
    """

    fmt0 = (fmt or "").strip().lower()
    if not fmt0:
        ext = os.path.splitext(savefile)[1].lstrip(".").lower()
        fmt0 = ext or "json"

    if fmt0 != "json":
        raise ValueError(
            f"Unsupported fmt={fmt!r}. Only 'json' is supported."
        )

    with open(savefile, "w", encoding="utf-8") as f:
        json.dump(
            dict(payload),
            f,
            indent=int(indent),
            ensure_ascii=False,
            default=_json_default,
        )
