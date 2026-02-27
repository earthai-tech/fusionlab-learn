# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Cumulative subsidence maps on satellite basemap + optional hotspots.

Layout (2 rows × N columns):

  [2022 observed (cum)] [2022 predicted (cum)]
  [2024 forecast (cum)] [2025 forecast (cum)] ...

Rows: Nansha (top), Zhongshan (bottom).

Background: Esri.WorldImagery satellite tiles (contextily).
Foreground: PS points coloured by cumulative subsidence (mm)
            since a baseline year (default 2020).
Optional: hotspot overlay from Fig.6 hotspot points CSV.

Cumulative definition
---------------------
This script always plots *cumulative since start_year*, but the
input CSV can be:

1) --subsidence-kind=cumulative (default)
   Values are already cumulative. We rebase at the first year
   >= start_year:

     cum(t) = value(t) - value(first)

2) --subsidence-kind=increment (or rate)
   Values are yearly increments. We accumulate:

     cum(t) = sum_{y} increment(y)

CRS handling (lat/lon vs UTM)
-----------------------------
coord_x/coord_y may be:
- lon/lat (degrees)  -> treated as EPSG:4326
- UTM (meters)       -> treated as EPSG:<utm_epsg> (default 32649)

Auto-detection:
- if |x|<=180 and |y|<=90 => lon/lat
- else => UTM

You can override using --coords-mode and --utm-epsg.

Outputs
-------
Saves both PNG and PDF using --out (resolved under scripts/figs
if relative).

Notes
-----
- Code style targets black+ruff with line length 62.
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import contextily as cx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from . import utils


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plot-geo-cumulative",
        description=(
            "Cumulative subsidence maps on a satellite "
            "basemap (GeoPrior)."
        ),
    )

    p.add_argument(
        "--ns-val",
        required=True,
        help="Nansha validation calibrated CSV.",
    )
    p.add_argument(
        "--zh-val",
        required=True,
        help="Zhongshan validation calibrated CSV.",
    )
    p.add_argument(
        "--ns-future",
        required=True,
        help="Nansha future forecast CSV (H*_future).",
    )
    p.add_argument(
        "--zh-future",
        required=True,
        help="Zhongshan future forecast CSV (H*_future).",
    )

    p.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Baseline year for cumulative sum.",
    )
    p.add_argument(
        "--year-val",
        type=int,
        default=2022,
        help="Observed vs predicted panel year.",
    )
    p.add_argument(
        "--years-forecast",
        nargs="*",
        type=int,
        default=[2024, 2025],
        help="Forecast years (cumulative q50).",
    )
    p.add_argument(
        "--subsidence-kind",
        default="cumulative",
        choices=("cumulative", "increment", "rate"),
        help=(
            "Meaning of subsidence values in CSVs. "
            "cumulative=already cumulative (rebase at start_year); "
            "increment/rate=per-year increments (cumsum)."
        ),
    )
    p.add_argument(
        "--clip",
        type=float,
        default=99.0,
        help="Percentile clip for color scale.",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Colormap for cumulative subsidence.",
    )

    p.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="Marker size for PS points.",
    )
    p.add_argument(
        "--point-alpha",
        type=float,
        default=0.90,
        help="Alpha for PS points.",
    )

    p.add_argument(
        "--hotspot-csv",
        default=None,
        help="Optional hotspot points CSV (Fig.6).",
    )
    p.add_argument(
        "--hotspot-field",
        default=None,
        help="Optional hotspot color-by column.",
    )
    p.add_argument(
        "--hotspot-color",
        default="black",
        help="Hotspot color if no hotspot-field.",
    )
    p.add_argument(
        "--hotspot-size",
        type=float,
        default=8.0,
        help="Hotspot marker size.",
    )
    p.add_argument(
        "--hotspot-alpha",
        type=float,
        default=0.95,
        help="Hotspot alpha.",
    )

    p.add_argument(
        "--coords-mode",
        choices=("auto", "lonlat", "utm"),
        default="auto",
        help="Coordinate mode for coord_x/coord_y.",
    )
    p.add_argument(
        "--utm-epsg",
        type=int,
        default=32649,
        help="EPSG for UTM coords (default 32649).",
    )

    p.add_argument(
        "--hide-attribution",
        action="store_true",
        help="Hide contextily attribution.",
    )

    p.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)
    p.add_argument("--font", type=int, default=9)

    utils.add_plot_text_args(
        p,
        default_out="spatial_satellite_cumulative",
    )

    return p

def parse_args(
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:
    p = build_parser()
    return p.parse_args(argv)

# ---------------------------------------------------------------------
# Loading / schema
# ---------------------------------------------------------------------
def _load_csv_base(
    path: str,
    *,
    needed: Iterable[str],
) -> pd.DataFrame:
    p = utils.as_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_csv(p)

    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns in {p}: {miss}")

    for c in ["coord_x", "coord_y", "coord_t"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "forecast_step" in df.columns:
        df["forecast_step"] = pd.to_numeric(
            df["forecast_step"],
            errors="coerce",
        )

    df = df.dropna(subset=["coord_x", "coord_y", "coord_t"])
    return df.copy()


def load_val_csv(path: str) -> pd.DataFrame:
    df = _load_csv_base(
        path,
        needed=(
            "sample_idx",
            "coord_t",
            "coord_x",
            "coord_y",
            "subsidence_actual",
            "subsidence_q50",
        ),
    )
    for c in ["subsidence_actual", "subsidence_q50"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["subsidence_actual", "subsidence_q50"])


def load_future_csv(path: str) -> pd.DataFrame:
    df = _load_csv_base(
        path,
        needed=(
            "sample_idx",
            "coord_t",
            "coord_x",
            "coord_y",
            "subsidence_q50",
        ),
    )
    df["subsidence_q50"] = pd.to_numeric(
        df["subsidence_q50"],
        errors="coerce",
    )
    return df.dropna(subset=["subsidence_q50"])


# ---------------------------------------------------------------------
# CRS inference
# ---------------------------------------------------------------------
def _looks_like_lonlat(x: float, y: float) -> bool:
    return (abs(float(x)) <= 180.0) and (abs(float(y)) <= 90.0)


def infer_xy_crs(
    df: pd.DataFrame,
    *,
    mode: str,
    utm_epsg: int,
) -> str:
    if mode == "lonlat":
        return "EPSG:4326"
    if mode == "utm":
        return f"EPSG:{int(utm_epsg)}"

    xs = pd.to_numeric(df["coord_x"], errors="coerce")
    ys = pd.to_numeric(df["coord_y"], errors="coerce")

    if xs.notna().any() and ys.notna().any():
        x0 = float(xs.dropna().iloc[0])
        y0 = float(ys.dropna().iloc[0])
        if _looks_like_lonlat(x0, y0):
            return "EPSG:4326"

    return f"EPSG:{int(utm_epsg)}"


def to_web_mercator(
    df: pd.DataFrame,
    *,
    src_crs: str,
) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["coord_x"],
            df["coord_y"],
        ),
        crs=src_crs,
    )
    return gdf.to_crs(epsg=3857)


# ---------------------------------------------------------------------
# Cumulative panels
# ---------------------------------------------------------------------
def _panel_df(
    df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    out = df[["coord_x", "coord_y", value_col]].copy()
    return out.rename(columns={value_col: "value"})

def _cum_from_input(
    df: pd.DataFrame,
    *,
    value_col: str,
    kind: str,
) -> pd.Series:
    kind = str(kind).strip().lower()
    g = df.groupby("sample_idx")[value_col]

    if kind in ("increment", "rate"):
        return g.cumsum()

    # kind == "cumulative" (default): rebase at first available year
    return df[value_col] - g.transform("first")

def _cum_title_tag(kind: str) -> str:
    k = str(kind).strip().lower()
    if k in ("increment", "rate"):
        return "cum (from increments)"
    return "cum (rebased)"


def _cum_note(kind: str) -> str:
    k = str(kind).strip().lower()
    if k in ("increment", "rate"):
        return "from increments"
    return "rebased"

def build_city_panels(
    val_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    *,
    start_year: int,
    year_val: int,
    years_forecast: List[int],
    subs_kind: str = "cumulative"
) -> Dict[str, pd.DataFrame]:
    
    val = val_df.copy()
    fut = fut_df.copy()

    val["coord_t"] = val["coord_t"].astype(int)
    fut["coord_t"] = fut["coord_t"].astype(int)

    val = val[val["coord_t"] >= int(start_year)]
    fut = fut[fut["coord_t"] >= int(start_year)]

    # ---- observed cumulative (validation only)
    v0 = val.sort_values(["sample_idx", "coord_t"])
    v0 = v0.drop_duplicates(
        subset=["sample_idx", "coord_t"],
        keep="last",
    )
    # currently our data collected are cumlative so no need 
    # since the prediction should be made on cumulative 
    v0["cum_actual"] = _cum_from_input(
        v0,
        value_col="subsidence_actual",
        kind=subs_kind,
    )
    
    obs = v0[v0["coord_t"] == int(year_val)]
    obs_panel = _panel_df(obs, "cum_actual") if not obs.empty else (
        pd.DataFrame()
    )

    # ---- predicted cumulative (val + future)
    cols = [
        "sample_idx",
        "coord_t",
        "coord_x",
        "coord_y",
        "subsidence_q50",
    ]
    comb = pd.concat(
        [val[cols].copy(), fut[cols].copy()],
        ignore_index=True,
    )

    comb = comb.sort_values(["sample_idx", "coord_t"])
    comb = comb.drop_duplicates(
        subset=["sample_idx", "coord_t"],
        keep="last",
    )
    comb["cum_pred"] = _cum_from_input(
        comb,
        value_col="subsidence_q50",
        kind=subs_kind,
    )

    pred = comb[comb["coord_t"] == int(year_val)]
    pred_panel = _panel_df(pred, "cum_pred") if not pred.empty else (
        pd.DataFrame()
    )

    out: Dict[str, pd.DataFrame] = {}
    if not obs_panel.empty:
        out["obs"] = obs_panel
    if not pred_panel.empty:
        out["pred"] = pred_panel

    for yy in years_forecast:
        sub = comb[comb["coord_t"] == int(yy)]
        if sub.empty:
            continue
        out[f"Y{int(yy)}"] = _panel_df(sub, "cum_pred")

    return out


def clip_bounds(
    arrays: List[np.ndarray],
    *,
    clip: float,
) -> Tuple[float, float]:
    if not arrays:
        return (0.0, 1.0)

    good: List[np.ndarray] = []
    for a in arrays:
        if a.size == 0:
            continue
        aa = a[np.isfinite(a)]
        if aa.size:
            good.append(aa)

    if not good:
        return (0.0, 1.0)

    x = np.concatenate(good)
    p = float(clip)
    p = max(50.0, min(100.0, p))

    lo = float(np.percentile(x, 100.0 - p))
    hi = float(np.percentile(x, p))

    if lo >= hi:
        lo = float(np.min(x))
        hi = float(np.max(x))

    return (lo, hi)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def draw_city_row(
    *,
    axes: np.ndarray,
    row_idx: int,
    city: str,
    panels_3857: Dict[str, gpd.GeoDataFrame],
    col_keys: List[str],
    cmap: mpl.colors.Colormap,
    cum_tag: str,
    vmin: float,
    vmax: float,
    args: argparse.Namespace,
    show_panel_titles: bool,
    hotspots: Optional[pd.DataFrame],
) -> None:
    for j, key in enumerate(col_keys):
        ax = axes[row_idx, j]

        if key not in panels_3857:
            ax.set_axis_off()
            continue

        gdf = panels_3857[key]

        gdf.plot(
            ax=ax,
            column="value",
            cmap=cmap,
            markersize=args.point_size,
            alpha=args.point_alpha,
            vmin=vmin,
            vmax=vmax,
            linewidth=0.0,
        )

        attrib = None if not args.hide_attribution else False
        cx.add_basemap(
            ax,
            source=cx.providers.Esri.WorldImagery,
            alpha=0.95,
            attribution=attrib,
        )

        if hotspots is not None and key.startswith("Y"):
            yy = int(key[1:])
            sub = hotspots[
                (hotspots["city"] == city)
                & (hotspots["year"].astype(int) == yy)
                & (hotspots["kind"] == "forecast")
            ]
            if not sub.empty:
                src = infer_xy_crs(
                    sub,
                    mode=args.coords_mode,
                    utm_epsg=args.utm_epsg,
                )
                hot = to_web_mercator(sub, src_crs=src)

                if args.hotspot_field and (
                    args.hotspot_field in hot.columns
                ):
                    hot.plot(
                        ax=ax,
                        column=args.hotspot_field,
                        cmap="hot_r",
                        markersize=args.hotspot_size,
                        alpha=args.hotspot_alpha,
                        linewidth=0.0,
                        legend=False,
                    )
                else:
                    hot.plot(
                        ax=ax,
                        color=args.hotspot_color,
                        markersize=args.hotspot_size,
                        alpha=args.hotspot_alpha,
                        linewidth=0.0,
                    )

        ax.set_axis_off()

        if not show_panel_titles:
            continue

        if key == "obs":
            ttl = f"{city} • {args.year_val} obs {cum_tag}"
        elif key == "pred":
            ttl = f"{city} • {args.year_val} pred {cum_tag}"
        else:
            ttl = f"{city} • {key[1:]} fcst {cum_tag}"

        ax.set_title(
            ttl,
            loc="left",
            fontweight="bold",
            pad=4,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def plot_geo_cumulative_main(
    argv: Optional[List[str]] = None,
) -> None:
    args = parse_args(argv)
    utils.set_paper_style(fontsize=args.font, dpi=args.dpi)

    cum_tag = _cum_title_tag(args.subsidence_kind)
    note = _cum_note(args.subsidence_kind)

    show_legend = utils.str_to_bool(args.show_legend, default=True)
    show_title = utils.str_to_bool(args.show_title, default=True)
    show_pan_t = utils.str_to_bool(
        args.show_panel_titles,
        default=True,
    )
    show_labels = utils.str_to_bool(args.show_labels, default=True)

    ns_val = load_val_csv(args.ns_val)
    zh_val = load_val_csv(args.zh_val)
    ns_fut = load_future_csv(args.ns_future)
    zh_fut = load_future_csv(args.zh_future)

    ns_pan = build_city_panels(
        ns_val,
        ns_fut,
        start_year=args.start_year,
        year_val=args.year_val,
        years_forecast=list(args.years_forecast),
        subs_kind=args.subsidence_kind,
    )
    zh_pan = build_city_panels(
        zh_val,
        zh_fut,
        start_year=args.start_year,
        year_val=args.year_val,
        years_forecast=list(args.years_forecast),
        subs_kind=args.subsidence_kind,
    )

    col_keys: List[str] = []
    if ("obs" in ns_pan) or ("obs" in zh_pan):
        col_keys.append("obs")
    if ("pred" in ns_pan) or ("pred" in zh_pan):
        col_keys.append("pred")

    for yy in args.years_forecast:
        k = f"Y{int(yy)}"
        if (k in ns_pan) or (k in zh_pan):
            col_keys.append(k)

    if not col_keys:
        raise RuntimeError("No panels found. Check years/CSVs.")

    # CRS conversion per city (supports lon/lat or UTM)
    ns_crs = infer_xy_crs(
        ns_val,
        mode=args.coords_mode,
        utm_epsg=args.utm_epsg,
    )
    zh_crs = infer_xy_crs(
        zh_val,
        mode=args.coords_mode,
        utm_epsg=args.utm_epsg,
    )

    ns_3857: Dict[str, gpd.GeoDataFrame] = {}
    zh_3857: Dict[str, gpd.GeoDataFrame] = {}

    for k in col_keys:
        if k in ns_pan:
            ns_3857[k] = to_web_mercator(ns_pan[k], src_crs=ns_crs)
        if k in zh_pan:
            zh_3857[k] = to_web_mercator(zh_pan[k], src_crs=zh_crs)

    # Global color scale
    vals: List[np.ndarray] = []
    for dct in (ns_pan, zh_pan):
        for k in col_keys:
            if k in dct:
                vals.append(dct[k]["value"].to_numpy())

    vmin, vmax = clip_bounds(vals, clip=args.clip)
    cmap = plt.get_cmap(args.cmap)

    # Optional hotspots
    hot_df: Optional[pd.DataFrame] = None
    if args.hotspot_csv:
        hp = utils.as_path(args.hotspot_csv)
        if hp.exists():
            hot_df = pd.read_csv(hp)
            utils.ensure_columns(
                hot_df,
                aliases={
                    "coord_x": ("coord_x", "x", "lon", "lng"),
                    "coord_y": ("coord_y", "y", "lat"),
                },
            )

    ncols = len(col_keys)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(3.0 * ncols, 6.0),
        constrained_layout=False,
    )
    axes = np.asarray(axes).reshape(2, ncols)

    plt.subplots_adjust(
        left=0.04,
        right=0.86 if show_legend else 0.98,
        top=0.93,
        bottom=0.06,
        wspace=0.05,
        hspace=0.12,
    )

    draw_city_row(
        axes=axes,
        row_idx=0,
        city="Nansha",
        panels_3857=ns_3857,
        col_keys=col_keys,
        cmap=cmap,
        cum_tag=cum_tag,
        vmin=vmin,
        vmax=vmax,
        args=args,
        show_panel_titles=show_pan_t,
        hotspots=hot_df,
    )
    draw_city_row(
        axes=axes,
        row_idx=1,
        city="Zhongshan",
        panels_3857=zh_3857,
        col_keys=col_keys,
        cum_tag=cum_tag,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        args=args,
        show_panel_titles=show_pan_t,
        hotspots=hot_df,
    )

    # Shared colorbar
    if show_legend:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cax = fig.add_axes([0.88, 0.20, 0.02, 0.60])
        cb = fig.colorbar(sm, cax=cax)

        if show_labels:
            lab = utils.label("subsidence_cum")
            cb.set_label(f"{lab} since {int(args.start_year)}")
            
    if show_legend and show_labels:
        lab = utils.label("subsidence_cum")
        cb.set_label(
            f"{lab} since {int(args.start_year)} "
            f"({note})"
        )

    if show_title:
        default = (
            f"Cumulative subsidence since "
            f"{int(args.start_year)} "
            f"({note}) on satellite basemap"
        )
        
    # Suptitle
    if show_title:
        default = (
            f"Cumulative subsidence since {int(args.start_year)} "
            "on satellite basemap"
        )
        ttl = utils.resolve_title(default=default, title=args.title)
        fig.suptitle(
            ttl,
            y=0.98,
            fontsize=11,
            fontweight="bold",
        )

    out = utils.resolve_fig_out(args.out)
    utils.ensure_dir(out.parent)

    png = out.with_suffix(".png")
    pdf = out.with_suffix(".pdf")

    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print(f"[OK] Saved: {png}")
    print(f"[OK] Saved: {pdf}")

def main() -> None:
    plot_geo_cumulative_main()


if __name__ == "__main__":
    main()
    

