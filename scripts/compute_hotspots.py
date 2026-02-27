# scripts/summarize_hotspots.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import utils as u


@dataclass(frozen=True)
class CityInputs:
    city: str
    eval_csv: Path
    future_csv: Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="summarize-hotspots",
        description=(
            "Compute hotspot characteristics table "
            "(city × year) and export CSV/LaTeX."
        ),
    )

    # City selection (optional)
    u.add_city_flags(p, default_both=True)

    # Preferred "src" mode (auto-discovery)
    p.add_argument("--ns-src", type=str, default=None)
    p.add_argument("--zh-src", type=str, default=None)

    # Direct file mode (overrides src discovery)
    p.add_argument("--ns-eval", "--ns-val", type=str, default=None)
    p.add_argument("--zh-eval", "--zh-val", type=str, default=None)
    p.add_argument("--ns-future", type=str, default=None)
    p.add_argument("--zh-future", type=str, default=None)

    p.add_argument(
        "--baseline-year",
        type=int,
        default=2022,
        help="Baseline year for anomaly (default: 2022).",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Hotspot threshold percentile (default: 90).",
    )

    p.add_argument(
        "--subsidence-kind",
        type=str,
        default="cumulative",
        choices=("cumulative", "rate", "increment"),
        help=(
            "Meaning of subsidence values in CSVs. "
            "cumulative=yearly cumulative values; "
            "rate/increment=annual values directly."
        ),
    )
    p.add_argument(
        "--baseline-source",
        type=str,
        default="actual",
        choices=("actual", "q50"),
        help=(
            "Baseline 2022 source in eval CSV: "
            "actual (default) or q50."
        ),
    )
    p.add_argument(
        "--quantile",
        type=str,
        default="q50",
        choices=("q10", "q50", "q90"),
        help="Forecast quantile used for hotspots (default q50).",
    )

    p.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Explicit years to summarize (e.g., 2025 2026). "
            "If omitted, uses last --n-years years in future CSV."
        ),
    )
    p.add_argument(
        "--n-years",
        type=int,
        default=2,
        help="If --years omitted, use last N future years.",
    )

    p.add_argument(
        "--format",
        type=str,
        default="both",
        choices=("csv", "tex", "both"),
        help="Export format (csv/tex/both).",
    )
    p.add_argument(
        "--out",
        "-o",
        type=str,
        default="tab_hotspots",
        help="Output stem/path (scripts/out if relative).",
    )

    p.add_argument(
        "--caption",
        type=str,
        default=(
            "Characteristics of forecast hotspot clusters. "
            "Hotspots are defined as locations where the absolute "
            "change in annual subsidence relative to 2022 exceeds "
            "the 90th percentile threshold $T_{0.9}$ within each "
            "city and year."
        ),
    )
    p.add_argument(
        "--label",
        type=str,
        default="tab:hotspots",
        help="LaTeX label (default: tab:hotspots).",
    )

    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _as_path(x: Optional[str]) -> Optional[Path]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    return Path(s).expanduser()


def _pick_city_inputs(
    *,
    city: str,
    src: Optional[Path],
    eval_csv: Optional[Path],
    future_csv: Optional[Path],
) -> CityInputs:
    cc = u.canonical_city(city)

    if eval_csv is not None and future_csv is not None:
        return CityInputs(cc, eval_csv, future_csv)

    if src is None:
        raise SystemExit(
            f"[{cc}] Need either --{cc[:2].lower()}-src "
            "or both --*-eval and --*-future."
        )

    art = u.detect_artifacts(src)

    ev = eval_csv
    fu = future_csv

    # eval: prefer TestSet eval, fallback to Validation
    if ev is None:
        ev = art.forecast_test_csv or art.forecast_val_csv
    # future: prefer TestSet future, fallback to generic future
    if fu is None:
        fu = art.forecast_test_future_csv or art.forecast_future_csv

    if ev is None:
        raise SystemExit(f"[{cc}] Could not locate eval CSV.")
    if fu is None:
        raise SystemExit(f"[{cc}] Could not locate future CSV.")

    return CityInputs(cc, Path(ev), Path(fu))


def _infer_unit(df: pd.DataFrame) -> str:
    for c in ("subsidence_unit", "unit", "units"):
        if c in df.columns:
            s = df[c].dropna()
            if not s.empty:
                u0 = str(s.iloc[0]).strip().lower()
                if u0.startswith("mm"):
                    return "mm"
                if u0 == "m" or u0.startswith("meter"):
                    return "m"
    return "mm"


def _to_mm(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    u0 = _infer_unit(df)
    if u0 == "mm":
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * 1000.0
    if "subsidence_unit" in out.columns:
        out["subsidence_unit"] = "mm"
    return out


def _load_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = [
        "sample_idx",
        "coord_t",
        "subsidence_q50",
    ]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing {c!r} in {path}")
    if "subsidence_actual" not in df.columns:
        # allow, but baseline_source=actual will fallback to q50
        df["subsidence_actual"] = np.nan

    for c in ("sample_idx", "coord_t"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("subsidence_actual", "subsidence_q10",
              "subsidence_q50", "subsidence_q90"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["sample_idx", "coord_t"]).copy()
    return df


def _load_future(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = [
        "sample_idx",
        "coord_t",
        "subsidence_q50",
    ]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing {c!r} in {path}")

    for c in ("sample_idx", "coord_t"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("subsidence_q10", "subsidence_q50", "subsidence_q90"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["sample_idx", "coord_t"]).copy()
    return df


def _wide(
    df: pd.DataFrame,
    *,
    value_col: str,
) -> pd.DataFrame:
    d0 = df[["sample_idx", "coord_t", value_col]].copy()
    d0["year"] = pd.to_numeric(d0["coord_t"], errors="coerce")
    d0 = d0.dropna(subset=["sample_idx", "year"]).copy()
    d0["year"] = d0["year"].astype(int)

    g = d0.groupby(["sample_idx", "year"], dropna=False)[value_col]
    s = g.mean()
    w = s.unstack("year")
    w.index = w.index.astype(int)
    return w


def _annual_series(
    *,
    kind: str,
    eval_w: pd.DataFrame,
    fut_w: pd.DataFrame,
    base_year: int,
    years: List[int],
) -> Tuple[pd.Series, Dict[int, pd.Series]]:
    """
    Returns:
      base_2022 (annual), and dict year->annual series.
    """
    k = str(kind).strip().lower()

    # common sample set
    idx = eval_w.index.intersection(fut_w.index)
    ev = eval_w.loc[idx]
    fu = fut_w.loc[idx]

    out: Dict[int, pd.Series] = {}

    if k in ("rate", "increment"):
        base = ev.get(base_year)
        if base is None:
            raise SystemExit("Baseline year missing in eval.")
        base = base.astype(float)

        for y in years:
            if y == base_year:
                out[y] = base
            else:
                if y not in fu.columns:
                    out[y] = pd.Series(np.nan, index=idx)
                else:
                    out[y] = fu[y].astype(float)

        return base, out

    # cumulative -> annual = diff
    if (base_year not in ev.columns) or ((base_year - 1) not in ev.columns):
        raise SystemExit(
            "Need baseline_year and baseline_year-1 in eval "
            "for cumulative->annual conversion."
        )

    base_cum = ev[base_year].astype(float)
    prev_cum = ev[base_year - 1].astype(float)
    base = base_cum - prev_cum

    for y in years:
        if y == base_year:
            out[y] = base
            continue

        if y not in fu.columns:
            out[y] = pd.Series(np.nan, index=idx)
            continue

        if y - 1 in fu.columns:
            out[y] = fu[y].astype(float) - fu[y - 1].astype(float)
            continue

        # first forecast year: use base cumulative
        out[y] = fu[y].astype(float) - base_cum

    return base, out


def _summarize_city(
    *,
    city: str,
    eval_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    years: List[int],
    kind: str,
    base_year: int,
    pct: float,
    baseline_source: str,
    qcol: str,
) -> pd.DataFrame:
    # unify units to mm for all numeric cols used
    eval_df = _to_mm(
        eval_df,
        ["subsidence_actual", "subsidence_q10",
         "subsidence_q50", "subsidence_q90"],
    )
    fut_df = _to_mm(
        fut_df,
        ["subsidence_q10", "subsidence_q50", "subsidence_q90"],
    )

    # pick baseline column in eval
    bs = str(baseline_source).strip().lower()
    if bs == "actual" and eval_df["subsidence_actual"].notna().any():
        base_col = "subsidence_actual"
    else:
        base_col = "subsidence_q50"

    ev_w = _wide(eval_df, value_col=base_col)
    fu_w = _wide(fut_df, value_col=qcol)

    base, annual = _annual_series(
        kind=kind,
        eval_w=ev_w,
        fut_w=fu_w,
        base_year=base_year,
        years=years,
    )

    mean_2022 = float(np.nanmean(base.to_numpy(float)))

    rows: List[Dict[str, object]] = []

    for y in years:
        s_y = annual.get(y)
        if s_y is None:
            continue

        b = base.reindex(s_y.index).astype(float)
        s = s_y.astype(float)

        m = np.isfinite(b.to_numpy()) & np.isfinite(s.to_numpy())
        if not bool(np.any(m)):
            continue

        b2 = b.to_numpy()[m]
        s2 = s.to_numpy()[m]
        ds = np.abs(s2 - b2)

        if not np.isfinite(ds).any():
            continue

        thr = float(np.nanpercentile(ds, float(pct)))
        hot = ds > thr

        n_hot = int(np.sum(hot))

        if n_hot > 0:
            s_hot = s2[hot]
            ds_hot = ds[hot]
            s_min = float(np.nanmin(s_hot))
            s_mean = float(np.nanmean(s_hot))
            s_max = float(np.nanmax(s_hot))
            d_mean = float(np.nanmean(ds_hot))
            d_max = float(np.nanmax(ds_hot))
        else:
            s_min = s_mean = s_max = np.nan
            d_mean = d_max = np.nan

        rows.append(
            {
                "City": u.canonical_city(city),
                "Year": int(y),
                "Hotspots_n": n_hot,
                "s_hot_min": s_min,
                "s_hot_mean": s_mean,
                "s_hot_max": s_max,
                "d_hot_mean": d_mean,
                "d_hot_max": d_max,
                "mean_2022": mean_2022,
                "T_0p9": thr,
            }
        )

    return pd.DataFrame(rows)


def _pick_years_from_future(
    fut_df: pd.DataFrame,
    *,
    base_year: int,
    years: Optional[List[int]],
    n_years: int,
) -> List[int]:
    if years:
        return [int(y) for y in years]

    ys = pd.to_numeric(fut_df["coord_t"], errors="coerce")
    ys = ys.dropna().astype(int)
    uniq = sorted(set(int(y) for y in ys if int(y) > base_year))
    if not uniq:
        return []
    n = max(1, int(n_years))
    return uniq[-n:]


def _fmt_1(x: float) -> str:
    if x is None or not np.isfinite(float(x)):
        return ""
    return f"{float(x):.1f}"


def _tex_sidewaystable(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
) -> str:
    lines: List[str] = []
    lines.append(r"\begin{sidewaystable}[p]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{llrccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& & & \multicolumn{3}{c}{Hotspot subsidence $s_{\mathrm{hot}}$}"
        r" & \multicolumn{2}{c}{Hotspot anomaly $|\Delta s_{\mathrm{hot}}|$ vs.\ 2022}"
        r" & Mean 2022 subsidence & $T_{0.9}$ \\"
    )
    lines.append(
        r"City & Year & Hotspots & \multicolumn{3}{c}{(mm yr$^{-1}$)}"
        r" & \multicolumn{2}{c}{(mm yr$^{-1}$)}"
        r" & (mm yr$^{-1}$) & (mm yr$^{-1}$) \\"
    )
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-8}")
    lines.append(r"& & $n$ & min & mean & max & mean & max & & \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        city = str(r["City"])
        year = int(r["Year"])
        n = int(r["Hotspots_n"])

        smin = _fmt_1(r["s_hot_min"])
        smea = _fmt_1(r["s_hot_mean"])
        smax = _fmt_1(r["s_hot_max"])
        dmea = _fmt_1(r["d_hot_mean"])
        dmax = _fmt_1(r["d_hot_max"])
        m22 = _fmt_1(r["mean_2022"])
        t09 = _fmt_1(r["T_0p9"])

        lines.append(
            f"{city} & {year} & {n} & "
            f"{smin} & {smea} & {smax} & "
            f"{dmea} & {dmax} & {m22} & {t09} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{sidewaystable}")
    lines.append("")
    lines.append(r"\clearpage")
    lines.append("")
    return "\n".join(lines)


def compute_hotspots_main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # ensure scripts/out exists
    u.ensure_script_dirs()  # scripts/out + scripts/figs :contentReference[oaicite:2]{index=2}

    cities = u.resolve_cities(args)

    ns_src = _as_path(args.ns_src)
    zh_src = _as_path(args.zh_src)

    ns_eval = _as_path(args.ns_eval)
    zh_eval = _as_path(args.zh_eval)
    ns_fut = _as_path(args.ns_future)
    zh_fut = _as_path(args.zh_future)

    inputs: List[CityInputs] = []

    if "Nansha" in cities:
        inputs.append(
            _pick_city_inputs(
                city="Nansha",
                src=ns_src,
                eval_csv=ns_eval,
                future_csv=ns_fut,
            )
        )
    if "Zhongshan" in cities:
        inputs.append(
            _pick_city_inputs(
                city="Zhongshan",
                src=zh_src,
                eval_csv=zh_eval,
                future_csv=zh_fut,
            )
        )

    all_rows: List[pd.DataFrame] = []

    for ci in inputs:
        ev = _load_eval(ci.eval_csv)
        fu = _load_future(ci.future_csv)

        years = _pick_years_from_future(
            fu,
            base_year=int(args.baseline_year),
            years=args.years,
            n_years=int(args.n_years),
        )

        if not years:
            raise SystemExit(f"[{ci.city}] No future years found.")

        qcol = "subsidence_" + str(args.quantile).strip().lower()

        d = _summarize_city(
            city=ci.city,
            eval_df=ev,
            fut_df=fu,
            years=years,
            kind=str(args.subsidence_kind),
            base_year=int(args.baseline_year),
            pct=float(args.percentile),
            baseline_source=str(args.baseline_source),
            qcol=qcol,
        )
        all_rows.append(d)

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["City", "Year"]).reset_index(drop=True)

    out_path = u.resolve_out_out(str(args.out))
    stem = out_path
    if stem.suffix:
        stem = stem.with_suffix("")

    if args.format in ("csv", "both"):
        csv_p = stem.with_suffix(".csv")
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(csv_p, index=False)

    if args.format in ("tex", "both"):
        tex_p = stem.with_suffix(".tex")
        tex_p.parent.mkdir(parents=True, exist_ok=True)
        tex = _tex_sidewaystable(
            out,
            caption=str(args.caption),
            label=str(args.label),
        )
        tex_p.write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    compute_hotspots_main()