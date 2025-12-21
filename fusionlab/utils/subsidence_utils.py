
from __future__ import annotations
from typing import ( 
    Dict, Iterable, Tuple, 
    Any, Optional, Literal
)
import numpy as np
import pandas as pd

from dataclasses import dataclass
import warnings


ShiftMode = Literal["none", "min", "mean", "value"]


def finalize_si_scaling_kwargs(
    scaling_kwargs: Dict[str, Any],
    *,
    subs_in_si: bool,
    head_in_si: bool,
    thickness_in_si: bool,
    force_identity_affine_if_si: bool = True,
    warn: bool = True,
) -> Dict[str, Any]:
    """
    Make GeoPrior scaling_kwargs consistent and safe against double scaling.

    GeoPriorSubsNet uses:
      - subs: subs_scale_si, subs_bias_si, subs_unit_to_si
      - head: head_scale_si, head_bias_si, head_unit_to_si
      - H/thickness: H_scale_si, H_bias_si, thickness_unit_to_si

    If Stage-1 already converted to SI meters, set unit_to_si=1.0 and
    affine=(1,0) to prevent silent double conversion.
    """
    kw = dict(scaling_kwargs)

    def _as_float(x: Any, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return default

    # Ensure unit keys exist (even if Stage-1 forgot)
    kw["subs_unit_to_si"] = _as_float(kw.get("subs_unit_to_si", 1.0), 1.0)
    kw["head_unit_to_si"] = _as_float(kw.get("head_unit_to_si", 1.0), 1.0)
    kw["thickness_unit_to_si"] = _as_float(kw.get("thickness_unit_to_si", 1.0), 1.0)

    # Ensure affine keys exist (GeoPrior's names)
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
            # Not already SI: keep unit conversion active.
            # If affine missing, default identity.
            if kw.get(scale_key) is None:
                kw[scale_key] = 1.0
            if kw.get(bias_key) is None:
                kw[bias_key] = 0.0
            return

        # already SI: force unit_to_si=1 and (optionally) identity affine
        if warn and _as_float(kw.get(unit_key, 1.0), 1.0) != 1.0:
            warnings.warn(
                f"[GeoPrior SI] {name} is already SI, but {unit_key}={kw[unit_key]}. "
                f"Setting {unit_key}=1.0 to prevent double scaling.",
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

    _fix(already_si=subs_in_si, unit_key="subs_unit_to_si",
         scale_key="subs_scale_si", bias_key="subs_bias_si", name="subsidence")
    _fix(already_si=head_in_si, unit_key="head_unit_to_si",
         scale_key="head_scale_si", bias_key="head_bias_si", name="head/GWL")
    _fix(already_si=thickness_in_si, unit_key="thickness_unit_to_si",
         scale_key="H_scale_si", bias_key="H_bias_si", name="thickness/H_field")

    return kw

def finalize_si_affines_and_units(
    scaling_kwargs: Dict[str, Any],
    *,
    subs_in_si: bool,
    head_in_si: bool,
    thickness_in_si: bool,
    force_identity_affine_if_si: bool = True,
    warn: bool = True,
) -> Dict[str, Any]:
    """
    Make scaling_kwargs consistent and safe against "double conversion".

    If a quantity is already SI (meters), we strongly recommend:
      unit_to_si = 1.0 and explicit affine (scale=1, bias=0).

    This keeps behavior stable even if model internals change, and prevents
    silent shrinking if someone forgets to update unit factors.
    """
    kw = dict(scaling_kwargs)

    def _ensure(name: str, default: float) -> float:
        v = kw.get(name, default)
        try:
            return float(v)
        except Exception:
            return default

    # Ensure keys exist
    kw["subs_unit_to_si"] = _ensure("subs_unit_to_si", 1.0)
    kw["head_unit_to_si"] = _ensure("head_unit_to_si", 1.0)
    kw["thickness_unit_to_si"] = _ensure("thickness_unit_to_si", 1.0)

    # Affine keys are used by your model if present (and then unit_to_si is ignored)
    kw.setdefault("subs_scale_si", None)
    kw.setdefault("subs_bias_si", None)
    kw.setdefault("head_scale_si", None)
    kw.setdefault("head_bias_si", None)

    def _fix_one(prefix: str, already_si: bool):
        unit_key = f"{prefix}_unit_to_si"
        scale_key = f"{prefix}_scale_si"
        bias_key = f"{prefix}_bias_si"

        if not already_si:
            # If user didn't provide explicit affine, keep unit_to_si as the
            # conversion mechanism. Do NOT override explicit affines.
            if kw.get(scale_key) is None:
                kw[scale_key] = 1.0
            if kw.get(bias_key) is None:
                kw[bias_key] = 0.0
            return

        # already SI:
        if warn and kw.get(unit_key, 1.0) != 1.0:
            warnings.warn(
                f"[SI inputs] {unit_key} was {kw[unit_key]} but data is already SI. "
                f"Setting {unit_key}=1.0 to prevent double scaling.",
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

    _fix_one("subs", subs_in_si)
    _fix_one("head", head_in_si)
    _fix_one("thickness", thickness_in_si)

    return kw

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
