
from __future__ import annotations
from typing import ( 
    Dict, Iterable, Tuple, 
    Any, Optional, Literal, 
    Mapping, Union, Sequence 
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


def _is_nonempty_str(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _looks_like_zscore_name(col: str) -> bool:
    if not isinstance(col, str):
        return False
    c = col.strip().lower()
    return c.endswith("_z") or c.endswith("_zscore")


def _first_present(
    cols: set,
    candidates: Sequence[str],
) -> Optional[str]:
    for c in candidates:
        if c and c in cols:
            return c
    return None


def normalize_gwl_alias(
    df: pd.DataFrame,
    gwl_col_user: Optional[str],
    *,
    prefer_depth_bgs: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Normalize common GWL naming aliases.

    Naming-only: unit conversion happens later.
    """
    if not _is_nonempty_str(gwl_col_user):
        return df, gwl_col_user

    cols = set(df.columns)
    user = gwl_col_user.strip()

    depth_m = ["GWL_depth_bgs_m"]
    depth = ["GWL_depth_bgs"]
    generic = ["GWL"]

    aliases_depth_m = {"gwl_depth_bgs_m", "gwl_depth_m"}
    aliases_depth = {"gwl_depth_bgs", "gwl_depth"}
    aliases_generic = {"gwl"}

    u = user.lower()

    if u in aliases_generic:
        order = depth_m + depth + generic
        if not prefer_depth_bgs:
            order = generic + depth + depth_m
        pick = _first_present(cols, order)
        if pick is None:
            return df, user
        if verbose and pick != user:
            print(
                "  [GWL] Config uses "
                f"{user!r}; using {pick!r}."
            )
        return df, pick

    if u in aliases_depth_m and "GWL_depth_bgs_m" in cols:
        if verbose and user != "GWL_depth_bgs_m":
            print(
                "  [GWL] Alias "
                f"{user!r} -> 'GWL_depth_bgs_m'."
            )
        return df, "GWL_depth_bgs_m"

    if u in aliases_depth and "GWL_depth_bgs" in cols:
        if verbose and user != "GWL_depth_bgs":
            print(
                "  [GWL] Alias "
                f"{user!r} -> 'GWL_depth_bgs'."
            )
        return df, "GWL_depth_bgs"

    if user == "GWL_depth_bgs_m" and user not in cols:
        alt = _first_present(cols, depth)
        if alt is not None:
            if verbose:
                print(
                    "  [GWL] 'GWL_depth_bgs_m' "
                    "missing; using 'GWL_depth_bgs'."
                )
            return df, alt

    if user == "GWL_depth_bgs" and user not in cols:
        alt = _first_present(cols, depth_m)
        if alt is not None:
            if verbose:
                print(
                    "  [GWL] 'GWL_depth_bgs' "
                    "missing; using 'GWL_depth_bgs_m'."
                )
            return df, alt

    if user == "GWL" and ("GWL" in cols) and _first_present(
        cols, depth_m + depth
    ) is None:
        if "GWL_depth_bgs" not in cols:
            if verbose:
                print(
                    "  [GWL] Renaming "
                    "'GWL' -> 'GWL_depth_bgs'."
                )
            df.rename(
                columns={"GWL": "GWL_depth_bgs"},
                inplace=True,
            )
            return df, "GWL_depth_bgs"

    return df, user


def resolve_gwl_for_physics(
    df: pd.DataFrame,
    gwl_col_user: Optional[str],
    *,
    prefer_depth_bgs: bool = True,
    allow_keep_zscore_as_ml: bool = True,
    verbose: bool = True,
) -> Tuple[str, Optional[str]]:
    """Pick meters GWL for physics; keep z-score ML-only."""
    df, gwl_col_user = normalize_gwl_alias(
        df,
        gwl_col_user,
        prefer_depth_bgs=prefer_depth_bgs,
        verbose=verbose,
    )

    cols = set(df.columns)

    meters_pref = [
        "GWL_depth_bgs_m",
        "GWL_depth_bgs",
        "GWL",
    ]
    if not prefer_depth_bgs:
        meters_pref = [
            "GWL",
            "GWL_depth_bgs",
            "GWL_depth_bgs_m",
        ]

    zscore_cands = [
        "GWL_depth_bgs_m_z",
        "GWL_depth_bgs_z",
        "GWL_z",
    ]

    meters_avail = _first_present(cols, meters_pref)
    zscore_avail = _first_present(cols, zscore_cands)

    def _is_z(col: str) -> bool:
        if _looks_like_zscore_name(col):
            return True
        return col in zscore_cands

    if _is_nonempty_str(gwl_col_user):
        if gwl_col_user not in cols:
            raise ValueError(
                f"GWL_COL={gwl_col_user!r} not in dataset."
            )

        if _is_z(gwl_col_user):
            if meters_avail is None:
                raise ValueError(
                    "GWL_COL is z-scored but no meters "
                    "GWL exists. Physics needs meters."
                )
            gwl_m = meters_avail
            if allow_keep_zscore_as_ml:
                gwl_z = gwl_col_user
            else:
                gwl_z = None
            if verbose:
                print(
                    "  [GWL] Using "
                    f"{gwl_m!r} for physics; "
                    f"keeping {gwl_z!r} ML-only."
                )
            return gwl_m, gwl_z

        if ("GWL_depth_bgs_m" in cols) and (
            gwl_col_user != "GWL_depth_bgs_m"
        ):
            if verbose:
                print(
                    "  [GWL] Using "
                    "'GWL_depth_bgs_m' for physics "
                    f"(over {gwl_col_user!r})."
                )
            gwl_m = "GWL_depth_bgs_m"
        else:
            gwl_m = gwl_col_user

        if allow_keep_zscore_as_ml and zscore_avail:
            gwl_z = zscore_avail
        else:
            gwl_z = None
        return gwl_m, gwl_z

    if meters_avail is None:
        if zscore_avail is not None:
            raise ValueError(
                "No meters GWL found, but a z-score "
                "column exists. Physics needs meters."
            )
        raise ValueError(
            "No groundwater column found. Expected "
            "'GWL_depth_bgs_m', 'GWL_depth_bgs', or 'GWL'."
        )

    if allow_keep_zscore_as_ml and zscore_avail:
        gwl_z = zscore_avail
    else:
        gwl_z = None
    if verbose:
        if gwl_z is not None:
            print(
                "  [GWL] Auto-selected "
                f"{meters_avail!r} for physics; "
                "z-score kept ML-only."
            )
        else:
            print(
                "  [GWL] Auto-selected "
                f"{meters_avail!r} for physics."
            )
    return meters_avail, gwl_z


def resolve_head_column(
    df: pd.DataFrame,
    *,
    depth_col: str,
    head_col: str = "head_m",
    z_surf_col: Optional[str] = None,
    use_head_proxy: bool = True,
) -> Tuple[str, Optional[str]]:
    """Resolve a head column, creating one if needed."""
    cols = set(df.columns)

    if head_col and head_col in cols:
        z_used = None
        if z_surf_col and z_surf_col in cols:
            z_used = z_surf_col
        elif "z_surf_m" in cols:
            z_used = "z_surf_m"
        elif "z_surf" in cols:
            z_used = "z_surf"
        return head_col, z_used

    z_cands = []
    if z_surf_col:
        z_cands.append(z_surf_col)
    z_cands += ["z_surf_m", "z_surf_corr", "z_surf"]

    z_used = _first_present(cols, z_cands)

    if z_used is not None:
        new_head = "head_m"
        df[new_head] = (
            pd.to_numeric(df[z_used], errors="coerce")
            - pd.to_numeric(df[depth_col], errors="coerce")
        )
        return new_head, z_used

    if use_head_proxy:
        new_head = "head_m"
        df[new_head] = -pd.to_numeric(
            df[depth_col],
            errors="coerce",
        )
        return new_head, None

    raise ValueError(
        "Cannot resolve head. Provide 'head_m' "
        "or z_surf, or enable USE_HEAD_PROXY."
    )

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


def _lonlat_to_local_xy_m(
    lon: Union[np.ndarray, pd.Series],
    lat: Union[np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Local XY meters from lon/lat (fast, city-scale)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    lat0 = np.nanmedian(lat)
    lon0 = np.nanmedian(lon)
    
    # meters per degree approx
    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))

    x = (lon - lon0) * m_per_deg_lon
    y = (lat - lat0) * m_per_deg_lat
    return x, y


def _knn_impute(
    values: Union[np.ndarray, pd.Series],
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    good_mask: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Median kNN impute for values where good_mask is False."""
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    bad_mask = ~good_mask
    if int(bad_mask.sum()) == 0:
        return values.copy()

    good_idx = np.where(good_mask)[0]
    bad_idx = np.where(bad_mask)[0]

    if good_idx.size == 0:
        filled = values.copy()
        filled[bad_idx] = np.nanmedian(values)
        return filled
    
    # If too few good points, fallback
    need = max(5, min(int(k), 20))
    if good_idx.size < need:
        filled = values.copy()
        filled[bad_idx] = np.nanmedian(values[good_idx])
        return filled

    p_good = np.c_[x[good_idx], y[good_idx]]
    p_bad = np.c_[x[bad_idx], y[bad_idx]]

    filled = values.copy()
    n_nei = int(min(int(k), good_idx.size))

    try: # sklearn
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(
            n_neighbors=n_nei,
            algorithm="kd_tree",
        )
        nn.fit(p_good)
        neigh = nn.kneighbors(p_bad, return_distance=False)
        
        for j, nbrs in enumerate(neigh):
            src = good_idx[nbrs]
            filled[bad_idx[j]] = np.nanmedian(values[src])
        return filled
    except: 
        pass 
    

    try: # scipy
        from scipy.spatial import cKDTree
        tree = cKDTree(p_good)
        _, ii = tree.query(p_bad, k=n_nei)

        if ii.ndim == 1:
            ii = ii[:, None]

        for j, nbrs in enumerate(ii):
            src = good_idx[nbrs]
            filled[bad_idx[j]] = np.nanmedian(values[src])
        return filled
    
    except:
        pass
    
    # last resort
    filled[bad_idx] = np.nanmedian(values[good_idx])
    return filled

def _infer_gwl_scale_auto(
    g: np.ndarray,
) -> float:
    """Infer GWL scale from distribution only."""
    g = np.asarray(g, dtype=float)
    g = g[np.isfinite(g)]

    if g.size == 0:
        return 1.0

    gmax = float(np.max(g))
    p95 = float(np.percentile(g, 95))
    p99 = float(np.percentile(g, 99))

    # If it reaches ~100, it is almost surely cm.
    if (gmax >= 60.0) or (p95 >= 50.0) or (p99 >= 60.0):
        return 0.01

    # If the whole city is <= ~35, it is
    # more consistent with meters (e.g. 0-28 m).
    if gmax <= 35.0:
        return 1.0

    # Ambiguous middle zone: default to meters.
    return 1.0

def _resolve_save_path(savefile: Optional[str]) -> Optional[str]:
    """Return a CSV path or None."""
    if not savefile:
        return None

    path = os.fspath(savefile)

    if path.endswith(os.sep):
        return os.path.join(path, "cleaned_gwl_zsurf.csv")

    if os.path.isdir(path):
        return os.path.join(path, "cleaned_gwl_zsurf.csv")

    if not path.lower().endswith(".csv"):
        path = f"{path}.csv"

    return path

def clean_gwl_zsurf(
    df: pd.DataFrame,
    *,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    z_col: str = "z_surf",
    gwl_col: str = "GWL_depth_bgs",
    gwl_scale: Union[str, float] = "auto",
    max_depth_m: float = 50.0,
    z_outlier_iqr_mult: float = 3.0,
    knn_k: int = 25,
    return_report: bool = True,
    savefile: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Optional[str]]:
    """Fix GWL units + robust z_surf, then recompute head_m."""
    df = df.copy()
    
    # --- 0) Basic checks
    required = (lon_col, lat_col, z_col, gwl_col)
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c!r}")

    gwl_raw = f"{gwl_col}_raw"
    z_raw = f"{z_col}_raw"
    

    # Keep raw
    if gwl_raw not in df.columns:
        df[gwl_raw] = df[gwl_col]
    if z_raw not in df.columns:
        df[z_raw] = df[z_col]
        
    # --- 1) Infer GWL scale
    if gwl_scale == "auto":
        g = df[gwl_col].to_numpy(dtype=float)
        gwl_scale_val = _infer_gwl_scale_auto(g)
        # gmax = float(np.nanmax(g))
        # gmed = float(np.nanmedian(g))

        # # Your case: always <=100 and city is low-lying -> cm is the safest default
        # # If discover the collector used decimeters, set gwl_scale=0.1 manually.
        
        # if (gmax <= 120.0) and (gmed >= 5.0):
        #     gwl_scale_val = 0.01  # cm -> m
        # else:
        #     gwl_scale_val = 1.0  # already meters (fallback)
    else:
        gwl_scale_val = float(gwl_scale)

    # --- 2) Convert depth-bgs to meters + clip
    df["GWL_depth_bgs_m"] = np.clip(
        df[gwl_col].astype(float) * gwl_scale_val,
        0.0,
        float(max_depth_m),
    )

    # --- 3) Make z_surf static per (lon,lat) using median (robust to noise)
    z_loc = (
        df.groupby([lon_col, lat_col], sort=False)[z_col]
        .median()
        .rename("z_surf_loc_med")
        .reset_index()
    )

    # --- 4) Robust outlier detection on location-level z_surf
    zvals = z_loc["z_surf_loc_med"].to_numpy(dtype=float)
    q1, q3 = np.nanpercentile(zvals, [25, 75])
    iqr = float(q3 - q1)

    lo = float(q1 - z_outlier_iqr_mult * iqr)
    hi = float(q3 + z_outlier_iqr_mult * iqr)

    good = np.isfinite(zvals) & (zvals >= lo) & (zvals <= hi)
    
    # --- 5) Spatial kNN impute for outliers
    x, y = _lonlat_to_local_xy_m(
        z_loc[lon_col].to_numpy(),
        z_loc[lat_col].to_numpy(),
    )

    z_filled = _knn_impute(
        zvals,
        x,
        y,
        good_mask=good,
        k=int(knn_k),
    )
    
    # Extra safety: clip to robust bounds after imputation
    z_filled = np.clip(z_filled, lo, hi)

    z_loc["z_surf_corr"] = z_filled
    z_loc["z_surf_flag_outlier"] = ~good
    
    # --- 6) Merge corrected z_surf back
    df = df.merge(
        z_loc[[lon_col, lat_col, "z_surf_corr",
               "z_surf_flag_outlier"]],
        on=[lon_col, lat_col],
        how="left",
    )
    
    # Prefer corrected z_surf everywhere
    df["z_surf_m"] = df["z_surf_corr"].astype(float)
    df["head_m"] = df["z_surf_m"] - df["GWL_depth_bgs_m"]

    saved_path = _resolve_save_path(savefile)
    if saved_path is not None:
        df.to_csv(saved_path, index=False)

    report: Optional[Dict[str, Any]] = None
    if return_report:
        pct = [0.01, 0.05, 0.5, 0.95, 0.99]

        report = {
            "gwl_scale_used": float(gwl_scale_val),
            "z_loc_bounds_iqr": {
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lo": float(lo),
                "hi": float(hi),
            },
            "z_loc_outliers": int((~good).sum()),
            "z_loc_total": int(len(z_loc)),
            "depth_m_stats": df["GWL_depth_bgs_m"]
            .describe(percentiles=pct)
            .to_dict(),
            "head_stats": df["head_m"]
            .describe(percentiles=pct)
            .to_dict(),
            "saved_file": saved_path,
        }

    return df, report


def postprocess_eval_json(
    eval_json: Union[str, Mapping[str, Any]],
    *,
    cfg: Optional[Mapping[str, object]] = None,
    scope: Literal["all", "subsidence", "physics"] = "all",
    out_path: Optional[str] = None,
    overwrite: bool = False,
    add_rmse: bool = True,
    force: bool = False,
    indent: int = 2,
) -> Dict[str, Any]:
    """
    Post-hoc convert a Stage-2 evaluation JSON from SI to interpretable units.

    This is a safe wrapper around `convert_eval_payload_units(...)` that:
    - loads a JSON from disk (or accepts a payload dict),
    - infers unit factors from `payload["units"]` when cfg is missing,
    - avoids double conversion unless `force=True`,
    - optionally adds RMSE fields (sqrt(MSE)),
    - writes a converted JSON file if `out_path` is provided.

    Parameters
    ----------
    eval_json :
        Either a file path to the saved JSON, or an in-memory mapping.
    cfg :
        Optional config mapping/module. If missing (or incomplete), this helper
        will synthesize the minimal keys needed for conversion:
        - "SUBS_UNIT_TO_SI"
        - "TIME_UNITS"
    scope :
        Forwarded to `convert_eval_payload_units(...)`.
    out_path :
        If given, write the converted payload there. If a directory is given,
        a filename is generated next to the input name (or "geoprior_eval...").
    overwrite :
        If False and out_path exists, raise.
    add_rmse :
        If True, add RMSE fields wherever MSE is present (metrics_evaluate,
        point_metrics, per_horizon).
    force :
        If False, skip conversion when payload already declares an interpretable
        subsidence unit (e.g., "mm"). If True, always convert.
    indent :
        JSON indent used when writing.

    Returns
    -------
    dict
        The converted payload (always returned as a dict).
    """
    # ----------------------------
    # 1) Load payload if needed
    # ----------------------------
    src_path = None
    if isinstance(eval_json, (str, os.PathLike)):
        src_path = os.fspath(eval_json)
        with open(src_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = dict(eval_json)

    # ----------------------------
    # 2) Build minimal cfg if needed
    # ----------------------------
    cfg_eff: Dict[str, object] = dict(cfg or {})
    units = payload.get("units", {})
    if isinstance(units, dict):
        # Help `subs_unit_to_si(cfg)` by providing SUBS_UNIT_TO_SI when missing.
        if "SUBS_UNIT_TO_SI" not in cfg_eff and "subs_unit_to_si" in units:
            try:
                cfg_eff["SUBS_UNIT_TO_SI"] = float(units["subs_unit_to_si"])
            except Exception:
                pass

        # Help time-rate conversions (epsilon_*_raw) if present.
        if "TIME_UNITS" not in cfg_eff and "time_units" in units:
            cfg_eff["TIME_UNITS"] = str(units["time_units"])

    # ----------------------------
    # 3) Decide whether to convert
    # ----------------------------
    already_unit = ""
    if isinstance(units, dict):
        already_unit = str(units.get("subs_metrics_unit", "")).strip().lower()

    # If already interpretable (e.g. "mm"), do nothing unless forced.
    # Note: in your payloads, SI mode sets subs_metrics_unit="m".
    do_convert = force or (already_unit not in ("mm", "millimeter", "millimeters"))

    out = payload
    if do_convert:
        # Use your existing conversion logic (DRY).
        out = convert_eval_payload_units(  # type: ignore[name-defined]
            payload,
            cfg_eff,
            mode="interpretable",
            scope=scope,
            savefile=None,   # we handle writing ourselves below
            fmt="json",
            indent=indent,
            copy_payload=True,
        )

    # ----------------------------
    # 4) Add RMSE fields (optional)
    # ----------------------------
    if add_rmse:
        def _safe_sqrt(x: Any) -> Optional[float]:
            try:
                v = float(x)
            except Exception:
                return None
            if not math.isfinite(v) or v < 0.0:
                return None
            return float(math.sqrt(v))

        me = out.get("metrics_evaluate")
        if isinstance(me, dict):
            # q50 convenience
            if "subs_pred_mse_q50" in me and "subs_pred_rmse_q50" not in me:
                r = _safe_sqrt(me.get("subs_pred_mse_q50"))
                if r is not None:
                    me["subs_pred_rmse_q50"] = r
            if "gwl_pred_mse_q50" in me and "gwl_pred_rmse_q50" not in me:
                r = _safe_sqrt(me.get("gwl_pred_mse_q50"))
                if r is not None:
                    me["gwl_pred_rmse_q50"] = r

        pm = out.get("point_metrics")
        if isinstance(pm, dict):
            if "mse" in pm and "rmse" not in pm:
                r = _safe_sqrt(pm.get("mse"))
                if r is not None:
                    pm["rmse"] = r

        ph = out.get("per_horizon")
        if isinstance(ph, dict) and "mse" in ph and "rmse" not in ph:
            mse_map = ph.get("mse")
            if isinstance(mse_map, dict):
                rmse_map: Dict[str, float] = {}
                for k, v in mse_map.items():
                    r = _safe_sqrt(v)
                    if r is not None:
                        rmse_map[str(k)] = r
                if rmse_map:
                    ph["rmse"] = rmse_map

    # ----------------------------
    # 5) Write to disk if requested
    # ----------------------------
    if out_path:
        dst = os.fspath(out_path)
        if os.path.isdir(dst):
            # If caller gave a directory, build a filename.
            base = "geoprior_eval_interpretable.json"
            if src_path:
                root = os.path.splitext(os.path.basename(src_path))[0]
                base = f"{root}_interpretable.json"
            dst = os.path.join(dst, base)

        if (not overwrite) and os.path.exists(dst):
            raise FileExistsError(f"File exists: {dst}")

        # Keep UTF-8 friendly (matches your writer style elsewhere).
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=int(indent), ensure_ascii=False)

    return dict(out)

# # -----------------------
# # Recommended usage
# # -----------------------
# from fusionlab.utils.subsidence_utils import clean_gwl_zsurf

# # Nansha (your diagnostics show cm -> m):
# nansha_df2, nansha_rep, _ = clean_gwl_zsurf(
#     nansha_data,
#     gwl_col="GWL_depth_bgs",
#     gwl_scale="auto",   # should pick 0.01
# )

# # Zhongshan (range 0-28 is consistent with meters):
# zhong_df2, zhong_rep, _ = clean_gwl_zsurf(
#     zhongshan_data,
#     gwl_col="GWL_depth_bgs",
#     gwl_scale="auto",   # should pick 1.0
# )

