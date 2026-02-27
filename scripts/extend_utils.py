# extend_utils.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import utils


SubsKind = Literal["cumulative", "rate", "increment"]
OutKind = Literal["same", "cumulative", "rate"]
Method = Literal["linear_last", "linear_fit"]
UncGrowth = Literal["hold", "sqrt", "linear"]


_ALIASES: Dict[str, Tuple[str, ...]] = {
    "sample_idx": ("sample_idx", "sample_id", "sample"),
    "forecast_step": ("forecast_step", "forecast_s", "h"),
    "coord_t": ("coord_t", "year", "t"),
    "coord_x": ("coord_x", "lon", "longitude", "x"),
    "coord_y": ("coord_y", "lat", "latitude", "y"),
    "subsidence_q10": ("subsidence_q10", "q10", "p10"),
    "subsidence_q50": ("subsidence_q50", "q50", "p50"),
    "subsidence_q90": ("subsidence_q90", "q90", "p90"),
    "subsidence_actual": ("subsidence_actual", "actual"),
    "subsidence_unit": ("subsidence_unit", "unit", "units"),
    "calibration_factor": (
        "calibration_factor",
        "cal_factor",
    ),
    "is_calibrated": ("is_calibrated", "calibrated"),
}


@dataclass(frozen=True)
class ExtendCfg:
    kind: SubsKind = "cumulative"
    out_kind: OutKind = "same"
    method: Method = "linear_fit"
    window: int = 3
    unc_growth: UncGrowth = "sqrt"
    unc_scale: float = 1.0
    group_col: str = "sample_idx"
    time_col: str = "coord_t"


def _as_path(x: Any) -> Path:
    return utils.as_path(x)


def _canonize(
    df: pd.DataFrame,
    *,
    required: Sequence[str],
    where: str,
) -> pd.DataFrame:
    utils.ensure_columns(df, aliases=_ALIASES)

    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"{where}: missing {miss}")
    return df


def _coerce(
    df: pd.DataFrame,
    *,
    need_actual: bool,
    cc: ExtendCfg,
) -> pd.DataFrame:
    d = df.copy()

    base = [
        cc.group_col,
        "forecast_step",
        cc.time_col,
        "coord_x",
        "coord_y",
        "subsidence_q10",
        "subsidence_q50",
        "subsidence_q90",
    ]
    if need_actual:
        base.append("subsidence_actual")

    for c in base:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=base).copy()
    d[cc.group_col] = d[cc.group_col].astype(int)
    d["forecast_step"] = d["forecast_step"].astype(int)
    d[cc.time_col] = d[cc.time_col].astype(int)
    return d


def load_eval_csv(
    path: Any,
    *,
    cc: Optional[ExtendCfg] = None,
) -> pd.DataFrame:
    cc2 = cc or ExtendCfg()
    df = pd.read_csv(_as_path(path))

    _canonize(
        df,
        required=[
            cc2.group_col,
            "forecast_step",
            cc2.time_col,
            "coord_x",
            "coord_y",
            "subsidence_q10",
            "subsidence_q50",
            "subsidence_q90",
            "subsidence_actual",
        ],
        where="eval-forecast",
    )
    return _coerce(df, need_actual=True, cc=cc2)


def load_future_csv(
    path: Any,
    *,
    cc: Optional[ExtendCfg] = None,
) -> pd.DataFrame:
    cc2 = cc or ExtendCfg()
    df = pd.read_csv(_as_path(path))

    _canonize(
        df,
        required=[
            cc2.group_col,
            "forecast_step",
            cc2.time_col,
            "coord_x",
            "coord_y",
            "subsidence_q10",
            "subsidence_q50",
            "subsidence_q90",
        ],
        where="future-forecast",
    )
    return _coerce(df, need_actual=False, cc=cc2)


def _infer_unit(df: pd.DataFrame) -> str:
    if "subsidence_unit" in df.columns:
        s = df["subsidence_unit"].dropna()
        if not s.empty:
            return str(s.iloc[0]).strip().lower()
    return "mm"


def _unit_factor(u_from: str, u_to: str) -> float:
    a = (u_from or "").strip().lower()
    b = (u_to or "").strip().lower()

    if a == b:
        return 1.0
    if a == "m" and b == "mm":
        return 1000.0
    if a == "mm" and b == "m":
        return 1e-3

    raise ValueError(f"Unsupported: {u_from}->{u_to}")


def to_mm(df: pd.DataFrame) -> pd.DataFrame:
    u0 = _infer_unit(df)
    if u0 == "mm":
        return df

    fac = _unit_factor(u0, "mm")
    d = df.copy()

    for c in [
        "subsidence_q10",
        "subsidence_q50",
        "subsidence_q90",
        "subsidence_actual",
    ]:
        if c in d.columns:
            d[c] = (
                pd.to_numeric(d[c], errors="coerce")
                * fac
            )

    d["subsidence_unit"] = "mm"
    return d


def _targets(
    y_max: int,
    *,
    add_years: int,
    years: Optional[List[int]],
) -> List[int]:
    if years:
        ys = sorted(set(int(y) for y in years))
        return [y for y in ys if y > int(y_max)]
    return [
        int(y_max) + i
        for i in range(1, add_years + 1)
    ]


def _growth(mode: UncGrowth, h: int) -> float:
    if mode == "hold":
        return 1.0
    if mode == "linear":
        return float(1 + h)
    return float(np.sqrt(1.0 + float(h)))


def _nanline_fit(
    t: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    w = np.isfinite(y).astype(float)
    ws = np.sum(w, axis=1)
    ws = np.where(ws <= 0, 1.0, ws)

    t0 = np.sum(w * t[None, :], axis=1) / ws
    y0 = np.sum(w * y, axis=1) / ws

    dt = t[None, :] - t0[:, None]
    dy = y - y0[:, None]

    num = np.sum(w * dt * dy, axis=1)
    den = np.sum(w * dt * dt, axis=1)
    den = np.where(den <= 0, 1.0, den)

    a = num / den
    b = y0 - a * t0
    return a, b


def _predict(
    t_hist: np.ndarray,
    y_hist: np.ndarray,
    *,
    t_new: int,
    method: Method,
) -> np.ndarray:
    if method == "linear_last":
        y1 = y_hist[:, -1]
        y0 = y_hist[:, -2]
        t1 = float(t_hist[-1])
        t0 = float(t_hist[-2])
        dt = t1 - t0 if t1 != t0 else 1.0
        a = (y1 - y0) / dt
        b = y1 - a * t1
    else:
        a, b = _nanline_fit(t_hist, y_hist)

    return a * float(t_new) + b


def _wide(
    df: pd.DataFrame,
    *,
    col: str,
    cc: ExtendCfg,
) -> pd.DataFrame:
    return df.pivot_table(
        index=cc.group_col,
        columns=cc.time_col,
        values=col,
        aggfunc="mean",
    )


def _base_from_eval(
    ev: pd.DataFrame,
    *,
    year: int,
    col: str,
    ids: pd.Index,
    cc: ExtendCfg,
) -> Optional[np.ndarray]:
    d = ev.loc[
        ev[cc.time_col].astype(int).eq(int(year))
    ]
    if d.empty:
        return None

    w = _wide(d, col=col, cc=cc)
    if int(year) not in w.columns:
        return None

    return w[int(year)].reindex(ids).to_numpy(float)


def _cum_to_rate_all(
    wc: pd.DataFrame,
    *,
    years: np.ndarray,
    base: Optional[np.ndarray],
) -> np.ndarray:
    w = wc.reindex(columns=years).to_numpy(float)
    r = np.full_like(w, np.nan, dtype=float)

    for i in range(int(years.size)):
        if i == 0:
            if base is None:
                continue
            r[:, i] = (w[:, i] - base) / 1.0
            continue

        dt = float(years[i] - years[i - 1])
        if dt == 0:
            dt = 1.0
        r[:, i] = (w[:, i] - w[:, i - 1]) / dt

    return r


def _meta_last(
    fut: pd.DataFrame,
    *,
    ids: pd.Index,
    cc: ExtendCfg,
) -> pd.DataFrame:
    cols = [
        "coord_x",
        "coord_y",
        "subsidence_unit",
        "calibration_factor",
        "is_calibrated",
    ]
    keep = [c for c in cols if c in fut.columns]
    if not keep:
        return pd.DataFrame(index=ids)

    g = fut.sort_values([cc.group_col, cc.time_col])
    last = g.groupby(cc.group_col, sort=False).tail(1)
    m = last.set_index(cc.group_col)[keep]
    return m.reindex(ids)


def extend_future_df(
    future_df: pd.DataFrame,
    *,
    eval_df: Optional[pd.DataFrame] = None,
    add_years: int = 1,
    years: Optional[List[int]] = None,
    cc: Optional[ExtendCfg] = None,
) -> pd.DataFrame:
    """
    Extend a future forecast by 1-2+ years.

    This is extrapolation: it assumes a smooth
    (linear) trend in the last window of years.
    """
    cc2 = cc or ExtendCfg()

    fut = to_mm(future_df)
    ev = to_mm(eval_df) if eval_df is not None else None

    fut = fut.sort_values(
        [cc2.group_col, cc2.time_col]
    ).copy()

    y_max = int(
        np.nanmax(fut[cc2.time_col].to_numpy(int))
    )
    tgt = _targets(
        y_max,
        add_years=add_years,
        years=years,
    )
    if not tgt:
        return fut

    w10 = _wide(fut, col="subsidence_q10", cc=cc2)
    w50 = _wide(fut, col="subsidence_q50", cc=cc2)
    w90 = _wide(fut, col="subsidence_q90", cc=cc2)

    ids = w50.index.astype(int)
    years_all = np.array(sorted(w50.columns.astype(int)))

    k = int(max(2, cc2.window))
    if years_all.size < k:
        k = int(years_all.size)

    t_fit = years_all[-k:]

    if cc2.kind == "cumulative":
        b_year = int(years_all[0] - 1)

        b10 = None
        b50 = None
        b90 = None

        if ev is not None:
            b10 = _base_from_eval(
                ev,
                year=b_year,
                col="subsidence_q10",
                ids=ids,
                cc=cc2,
            )
            b50 = _base_from_eval(
                ev,
                year=b_year,
                col="subsidence_q50",
                ids=ids,
                cc=cc2,
            )
            b90 = _base_from_eval(
                ev,
                year=b_year,
                col="subsidence_q90",
                ids=ids,
                cc=cc2,
            )

        r10_all = _cum_to_rate_all(
            w10,
            years=years_all,
            base=b10,
        )
        r50_all = _cum_to_rate_all(
            w50,
            years=years_all,
            base=b50,
        )
        r90_all = _cum_to_rate_all(
            w90,
            years=years_all,
            base=b90,
        )
    else:
        r10_all = w10.reindex(
            columns=years_all,
        ).to_numpy(float)
        r50_all = w50.reindex(
            columns=years_all,
        ).to_numpy(float)
        r90_all = w90.reindex(
            columns=years_all,
        ).to_numpy(float)

    r10 = r10_all[:, -k:]
    r50 = r50_all[:, -k:]
    r90 = r90_all[:, -k:]

    lo_gap = r50[:, -1] - r10[:, -1]
    hi_gap = r90[:, -1] - r50[:, -1]
    lo_gap = np.where(np.isfinite(lo_gap), lo_gap, 0.0)
    hi_gap = np.where(np.isfinite(hi_gap), hi_gap, 0.0)

    out_kind = cc2.out_kind
    if out_kind == "same":
        if cc2.kind == "cumulative":
            out_kind = "cumulative"
        else:
            out_kind = "rate"

    if out_kind == "cumulative":
        if cc2.kind == "cumulative":
            c10 = w10.reindex(
                columns=years_all,
            ).to_numpy(float)
            c50 = w50.reindex(
                columns=years_all,
            ).to_numpy(float)
            c90 = w90.reindex(
                columns=years_all,
            ).to_numpy(float)

            c10 = c10[:, -1]
            c50 = c50[:, -1]
            c90 = c90[:, -1]
        else:
            # Build a relative cumulative anchor from rates.
            dtv = np.diff(
                years_all,
                prepend=int(years_all[0]) - 1,
            ).astype(float)
            dtv = np.where(dtv <= 0.0, 1.0, dtv)

            rr10 = np.nan_to_num(r10_all, nan=0.0)
            rr50 = np.nan_to_num(r50_all, nan=0.0)
            rr90 = np.nan_to_num(r90_all, nan=0.0)

            c10 = np.cumsum(rr10 * dtv[None, :], axis=1)
            c50 = np.cumsum(rr50 * dtv[None, :], axis=1)
            c90 = np.cumsum(rr90 * dtv[None, :], axis=1)

            c10 = c10[:, -1]
            c50 = c50[:, -1]
            c90 = c90[:, -1]

    last_step = fut.groupby(cc2.group_col)[
        "forecast_step"
    ].max()
    last_step = last_step.reindex(ids).fillna(0).astype(int)

    meta = _meta_last(fut, ids=ids, cc=cc2)

    y_prev = int(years_all[-1])
    new_parts: List[pd.DataFrame] = []

    for j, y_new in enumerate(tgt, start=1):
        r50_new = _predict(
            t_fit,
            r50,
            t_new=int(y_new),
            method=cc2.method,
        )

        fac = _growth(cc2.unc_growth, j)
        fac = fac * float(cc2.unc_scale)

        r10_new = r50_new - lo_gap * fac
        r90_new = r50_new + hi_gap * fac

        if out_kind == "rate":
            q10 = r10_new
            q50 = r50_new
            q90 = r90_new
        else:
            dt = float(int(y_new) - int(y_prev))
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0

            c10 = c10 + r10_new * dt
            c50 = c50 + r50_new * dt
            c90 = c90 + r90_new * dt

            q10 = c10
            q50 = c50
            q90 = c90

        df_new = pd.DataFrame(
            {
                cc2.group_col: ids.to_numpy(int),
                "forecast_step": last_step.to_numpy(int) + j,
                cc2.time_col: int(y_new),
                "subsidence_q10": q10.astype(float),
                "subsidence_q50": q50.astype(float),
                "subsidence_q90": q90.astype(float),
                "extended": True,
                "extend_kind": cc2.kind,
                "extend_method": cc2.method,
                "unc_growth": cc2.unc_growth,
            }
        )

        if not meta.empty:
            df_new = df_new.join(meta.reset_index(drop=True))

        new_parts.append(df_new)

        t_fit = np.append(t_fit[1:], int(y_new))
        r50 = np.column_stack([r50[:, 1:], r50_new])
        y_prev = int(y_new)

    out = pd.concat([fut] + new_parts, ignore_index=True)
    out = out.sort_values(
        [cc2.group_col, cc2.time_col]
    ).copy()

    out[cc2.group_col] = out[cc2.group_col].astype(int)
    out[cc2.time_col] = out[cc2.time_col].astype(int)
    out["forecast_step"] = out["forecast_step"].astype(int)

    if "subsidence_unit" in out.columns:
        out["subsidence_unit"] = "mm"

    return out


def extend_future_csv_file(
    future_csv: Any,
    *,
    eval_csv: Optional[Any] = None,
    out_csv: Optional[Any] = None,
    add_years: int = 1,
    years: Optional[List[int]] = None,
    cc: Optional[ExtendCfg] = None,
) -> Path:
    """
    Load -> extend -> write an updated future CSV.

    If out_csv is relative, it is placed in
    scripts/out/.
    """
    cc2 = cc or ExtendCfg()

    fut = load_future_csv(future_csv, cc=cc2)

    ev = None
    if eval_csv is not None:
        ev = load_eval_csv(eval_csv, cc=cc2)

    out = extend_future_df(
        fut,
        eval_df=ev,
        add_years=add_years,
        years=years,
        cc=cc2,
    )

    name = out_csv or "future_extended.csv"
    p = utils.resolve_out_out(str(name))
    p.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(p, index=False)
    return p

#How to use it (quick examples)
# from scripts import forecast_extend_utils as fx

# cc = fx.ExtendCfg(
#     kind="cumulative",
#     out_kind="same",
#     method="linear_fit",
#     window=3,
#     unc_growth="sqrt",
#     unc_scale=1.15,
# )

# out_path = fx.extend_future_csv_file(
#     future_csv="..._future_calibrated.csv",
#     eval_csv="..._eval_calibrated.csv",
#     out_csv="future_plus_2026.csv",
#     years=[2026],
#     cc=cc,
# )

# print(out_path)
# Produce annual rate instead of cumulative

# cc = fx.ExtendCfg(
#     kind="cumulative",
#     out_kind="rate",
#     method="linear_last",
#     unc_growth="sqrt",
# )

# df_ext = fx.extend_future_df(
#     future_df=future_df,
#     eval_df=eval_df,
#     years=[2026, 2027],
#     cc=cc,
# )