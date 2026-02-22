# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# scripts/plot_spatial_forecasts.py
#
# Figure 6 — Spatial validation + constrained forecasts.
#
# Outputs:
#   - scripts/figs/<out>.png + <out>.svg
#   - scripts/out/<hotspot-csv>  (optional)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as cfg
from . import utils



# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def _load_calib_df(path: str) -> pd.DataFrame:
    """
    Load calibrated validation/test CSV.

    We prefer utils.load_forecast_csv() because it already enforces
    the core schema (q10/q50/q90/actual) used elsewhere.
    """
    df = utils.load_forecast_csv(path)
    df = utils.ensure_columns(
        df,
        cfg._CALIB_REQUIRED,
        aliases=cfg._BASE_ALIASES,
        where="calibrated-forecast",
    )
    return df


def _load_future_df(path: str) -> pd.DataFrame:
    """
    Load future forecast CSV (no actual required).

    Must contain coord_x/coord_y/coord_t and subsidence_q50.
    """
    p = utils.as_path(path)
    df = pd.read_csv(p)

    df = utils.ensure_columns(
        df,
        cfg._FUT_REQUIRED,
        aliases=cfg._BASE_ALIASES,
        where="future-forecast",
    )

    for c in [
        "coord_x",
        "coord_y",
        "coord_t",
        "subsidence_q50",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=["coord_x", "coord_y", "coord_t", "subsidence_q50"]
    ).copy()

    return df


def _pick_calib_path(
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
    calib: Optional[str],
    future: Optional[str],
    split: str,
) -> Dict[str, Any]:
    """
    Resolve (calib_df, future_df) for one city.

    Priority:
      1) manual overrides: --*-calib and --*-future
      2) auto-detect from --*-src with --split selection
    """
    out: Dict[str, Any] = {"name": city}
    out["color"] = cfg.CITY_COLORS.get(city, "#333333")

    if calib and future:
        out["calib_df"] = _load_calib_df(calib)
        out["future_df"] = _load_future_df(future)
        out["split"] = split
        out["src_note"] = "manual"
        return out

    if not src:
        raise ValueError(
            f"{city}: provide --{city[:2].lower()}-src "
            f"or both --{city[:2].lower()}-calib and "
            f"--{city[:2].lower()}-future"
        )

    art = utils.detect_artifacts(src)

    c_p, c_lab = _pick_calib_path(art, split)
    f_p, f_lab = _pick_future_path(art, split)

    if c_p is None:
        raise FileNotFoundError(
            f"{city}: no calibrated CSV found under {src}"
        )
    if f_p is None:
        raise FileNotFoundError(
            f"{city}: no future CSV found under {src}"
        )

    out["calib_df"] = _load_calib_df(str(c_p))
    out["future_df"] = _load_future_df(str(f_p))
    out["split"] = c_lab
    out["src_note"] = f"{c_p.name} + {f_p.name} ({c_lab}/{f_lab})"

    return out


# ---------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------
def _coord_kind(df: pd.DataFrame) -> str:
    """
    Heuristic:
      - lon/lat if x in [-180,180] and y in [-90,90]
      - otherwise treat as projected (UTM, etc.)
    """
    x = df["coord_x"].to_numpy()
    y = df["coord_y"].to_numpy()

    if x.size == 0 or y.size == 0:
        return "unknown"

    xm = float(np.nanmax(np.abs(x)))
    ym = float(np.nanmax(np.abs(y)))

    if xm <= 180.0 and ym <= 90.0:
        return "lonlat"
    return "projected"


def _subset_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df.loc[df["coord_t"].astype(int).eq(int(year))].copy()


def _compute_cum_q50(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.sort_values(["sample_idx", "coord_t"]).copy()
    # No need to use cumsum() since the when subsidence_cum, it is 
    # predicted in cumulative already
    df2["subs_cum_q50"] = (
        # df2.groupby("sample_idx")["subsidence_q50"].cumsum()
        df2.groupby("sample_idx")["subsidence_q50"] # .cumsum()
    )
    return df2


def _extent_from_panels(
    panels: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Optional[Tuple[float, float, float, float]]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for _k, (x, y, _v) in panels.items():
        if x.size and y.size:
            xs.append(x)
            ys.append(y)

    if not xs:
        return None

    x0 = float(np.nanmin(np.concatenate(xs)))
    x1 = float(np.nanmax(np.concatenate(xs)))
    y0 = float(np.nanmin(np.concatenate(ys)))
    y1 = float(np.nanmax(np.concatenate(ys)))

    return (x0, x1, y0, y1)


def _grid_mean(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    *,
    res: int,
    extent: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Rasterize (mean in bin) to res x res grid.

    Returns:
      Z (shape [res,res], origin lower)
      extent (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = extent

    xb = np.linspace(xmin, xmax, res + 1)
    yb = np.linspace(ymin, ymax, res + 1)

    sumv, _, _ = np.histogram2d(x, y, bins=[xb, yb], weights=v)
    cnts, _, _ = np.histogram2d(x, y, bins=[xb, yb])

    with np.errstate(invalid="ignore", divide="ignore"):
        z = sumv / cnts
    z[cnts == 0] = np.nan

    return (z.T, extent)


def _clip_bounds(
    vals: List[np.ndarray],
    clip: float,
) -> Tuple[float, float]:
    good = []
    for v in vals:
        if v is None:
            continue
        vv = v[np.isfinite(v)]
        if vv.size:
            good.append(vv)

    if not good:
        return (0.0, 1.0)

    g = np.concatenate(good)

    hi = float(np.nanpercentile(g, clip))
    lo = float(np.nanpercentile(g, 100.0 - clip))

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.nanmin(g))
        hi = float(np.nanmax(g))

    return (lo, hi)


def _axes_cleanup(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _infer_unit(cities: List[Dict[str, Any]]) -> str:
    for c in cities:
        df = c.get("calib_df")
        if isinstance(df, pd.DataFrame):
            if "subsidence_unit" in df.columns:
                u = str(df["subsidence_unit"].iloc[0])
                if u and u.lower() != "nan":
                    return u
    return "mm"


# ---------------------------------------------------------------------
# Plot (Figure 6)
# ---------------------------------------------------------------------
def plot_fig6_spatial_forecasts(
    *,
    cities: List[Dict[str, Any]],
    year_val: int,
    years_fore: List[int],
    cumulative: bool,
    grid_res: int,
    clip: float,
    cmap_name: str,
    hotspot_mode: str,
    hotspot_q: float,
    out: str,
    out_hotspots: Optional[str],
    dpi: int,
    font: int,
    show_legend: bool,
    show_title: bool,
    show_panel_titles: bool,
    title: Optional[str],
) -> None:
    """
    Two columns fixed:
      - observed year_val (actual)
      - predicted year_val (q50)

    Remaining columns:
      - years_fore (q50 or cum q50)

    Rows:
      - one per city (1-city and 2-city layouts supported)
    """
    if not cities:
        raise ValueError("No city data provided.")

    utils.ensure_script_dirs()
    utils.set_paper_style(fontsize=int(font), dpi=int(dpi))

    unit = _infer_unit(cities)
    cmap = plt.get_cmap(cmap_name)

    # Build per-city panel dictionaries and extents
    for c in cities:
        calib_df = c["calib_df"]
        fut_df = c["future_df"]

        obs = _subset_year(calib_df, year_val)
        pred = obs.copy()

        panels: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        x = obs["coord_x"].to_numpy()
        y = obs["coord_y"].to_numpy()
        v0 = obs["subsidence_actual"].to_numpy()
        panels["obs"] = (x, y, v0)

        v1 = pred["subsidence_q50"].to_numpy()
        panels["pred"] = (x, y, v1)

        fut_use = fut_df
        if cumulative:
            fut_use = _compute_cum_q50(fut_df)

        for yy in years_fore:
            dyy = _subset_year(fut_use, int(yy))
            if dyy.empty:
                continue

            xf = dyy["coord_x"].to_numpy()
            yf = dyy["coord_y"].to_numpy()

            if cumulative:
                vf = dyy["subs_cum_q50"].to_numpy()
            else:
                vf = dyy["subsidence_q50"].to_numpy()

            panels[f"Y{int(yy)}"] = (xf, yf, vf)

        c["panels"] = panels

        ext = _extent_from_panels(panels)
        if ext is None:
            raise ValueError(f"{c['name']}: no points to plot.")
        c["extent"] = ext

        c["coord_kind"] = _coord_kind(calib_df)

    # Global vmin/vmax across all cities/panels
    all_vals: List[np.ndarray] = []
    for c in cities:
        for _k, (_x, _y, v) in c["panels"].items():
            all_vals.append(v)

    vmin, vmax = _clip_bounds(all_vals, float(clip))

    # Layout sizes
    n_rows = len(cities)
    n_cols = 2 + len(years_fore)

    fig_w = 2.2 * n_cols + 1.6
    fig_h = 2.6 * n_rows + 1.2

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
    )

    ax_arr = np.array(axes, ndmin=2)

    plt.subplots_adjust(
        left=0.06,
        right=0.90 if show_legend else 0.98,
        top=0.92,
        bottom=0.08,
        wspace=0.06,
        hspace=0.18,
    )

    # Hotspot export accumulator
    hs_rows: List[Dict[str, Any]] = []

    def _add_hotspots(
        *,
        city: str,
        panel: str,
        kind: str,
        year: int,
        extent: Tuple[float, float, float, float],
        z: np.ndarray,
        metric: np.ndarray,
        thr: float,
        base: Optional[np.ndarray],
    ) -> None:
        xmin, xmax, ymin, ymax = extent
        ny, nx = z.shape

        dx = (xmax - xmin) / float(nx)
        dy = (ymax - ymin) / float(ny)

        m = metric >= thr
        idxs = np.argwhere(m)

        for iy, ix in idxs:
            vv = float(z[iy, ix])
            if not np.isfinite(vv):
                continue

            rec: Dict[str, Any] = {
                "city": city,
                "panel": panel,
                "kind": kind,
                "year": int(year),
                "coord_x": float(xmin + (ix + 0.5) * dx),
                "coord_y": float(ymin + (iy + 0.5) * dy),
                "value": vv,
                "metric_value": float(metric[iy, ix]),
                "threshold": float(thr),
                "hotspot_mode": str(hotspot_mode),
                "hotspot_quantile": float(hotspot_q),
            }

            if base is not None:
                rec["baseline_value"] = float(base[iy, ix])

            hs_rows.append(rec)

    def _draw_panel(
        ax: plt.Axes,
        *,
        city: str,
        panel_key: str,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray,
        extent: Tuple[float, float, float, float],
        coord_kind: str,
        delta_base: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray]
        ],
        title_txt: str,
    ) -> Optional[mpl.image.AxesImage]:
        if x.size == 0:
            ax.set_axis_off()
            return None

        z, ext = _grid_mean(x, y, v, res=grid_res, extent=extent)

        aspect = "equal"
        if coord_kind == "lonlat":
            aspect = "auto"

        im = ax.imshow(
            z,
            extent=ext,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect=aspect,
        )

        # Hotspots: contours + optional CSV point export
        if hotspot_mode != "none":
            metric = None
            base_z = None

            if hotspot_mode == "absolute":
                metric = np.abs(z)
            else:
                if delta_base is not None:
                    bx, by, bv = delta_base
                    base_z, _ = _grid_mean(
                        bx,
                        by,
                        bv,
                        res=grid_res,
                        extent=extent,
                    )
                    metric = np.abs(z - base_z)

            if metric is not None and np.isfinite(metric).any():
                thr = float(np.nanquantile(metric, hotspot_q))

                try:
                    ax.contour(
                        (metric >= thr).astype(float),
                        levels=[0.5],
                        colors="k",
                        linewidths=0.7,
                        origin="lower",
                        extent=ext,
                    )
                except Exception:
                    pass

                # Export points
                if hs_rows is not None:
                    if panel_key.startswith("Y"):
                        yy = int(panel_key[1:])
                        kind = "forecast"
                    elif panel_key == "obs":
                        yy = int(year_val)
                        kind = "observed"
                    else:
                        yy = int(year_val)
                        kind = "predicted"

                    _add_hotspots(
                        city=city,
                        panel=panel_key,
                        kind=kind,
                        year=yy,
                        extent=ext,
                        z=z,
                        metric=metric,
                        thr=thr,
                        base=base_z,
                    )

        if show_panel_titles:
            ax.set_title(
                title_txt,
                pad=4,
                fontweight="bold",
            )

        _axes_cleanup(ax)
        return im

    # Delta baselines (obs year) per city
    baselines: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for c in cities:
        x0, y0, v0 = c["panels"]["obs"]
        baselines[c["name"]] = (x0, y0, v0)

    # Draw all panels
    for r, c in enumerate(cities):
        name = str(c["name"])
        panels = c["panels"]
        extent = c["extent"]
        ck = str(c["coord_kind"])

        # fixed first two columns
        keys = ["obs", "pred"]
        for yy in years_fore:
            keys.append(f"Y{int(yy)}")

        for j, k in enumerate(keys):
            ax = ax_arr[r, j]
            x, y, v = panels.get(k, (np.array([]),) * 3)

            if k == "obs":
                t = f"{name} • {year_val} observed"
            elif k == "pred":
                t = f"{name} • {year_val} predicted"
            else:
                yy = int(k[1:])
                tag = "cumulative" if cumulative else "q50"
                t = f"{name} • {yy} {tag}"

            db = baselines.get(name)
            if hotspot_mode == "delta" and k == "obs":
                db = None

            _draw_panel(
                ax,
                city=name,
                panel_key=k,
                x=x,
                y=y,
                v=v,
                extent=extent,
                coord_kind=ck,
                delta_base=db,
                title_txt=t,
            )

    # Shared colorbar
    if show_legend:
        sm = plt.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap,
        )
        cax = fig.add_axes([0.92, 0.20, 0.02, 0.60])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(f"Subsidence ({unit})")

    # Suptitle
    if show_title:
        ttl = utils.resolve_title(
            default="Fig. 6 — Spatial validation + forecasts",
            title=title,
        )
        fig.suptitle(ttl, x=0.02, ha="left")

    # Save figure
    fig_p = utils.resolve_fig_out(out)
    if fig_p.suffix:
        fig_p = fig_p.with_suffix("")

    fig.savefig(
        str(fig_p) + ".png",
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(str(fig_p) + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote {fig_p}.png/.svg")

    # Save hotspot points (to scripts/out/)
    if hotspot_mode != "none":
        if not hs_rows:
            print("[WARN] hotspot enabled but none found.")
            return

        if out_hotspots:
            out_hs = utils.as_path(out_hotspots)
        else:
            out_hs = utils.resolve_out_out("fig6-hotspot-points.csv")

        pd.DataFrame(hs_rows).to_csv(out_hs, index=False)
        print(f"[OK] wrote {out_hs}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _add_fig6_args(ap: argparse.ArgumentParser) -> None:
    utils.add_city_flags(ap, default_both=True)

    ap.add_argument(
        "--ns-src",
        type=str,
        default=None,
        help="Nansha results dir (auto-detect).",
    )
    ap.add_argument(
        "--zh-src",
        type=str,
        default=None,
        help="Zhongshan results dir (auto-detect).",
    )

    ap.add_argument(
        "--ns-calib",
        type=str,
        default=None,
        help="Override Nansha calibrated CSV.",
    )
    ap.add_argument(
        "--zh-calib",
        type=str,
        default=None,
        help="Override Zhongshan calibrated CSV.",
    )

    ap.add_argument(
        "--ns-future",
        type=str,
        default=None,
        help="Override Nansha future CSV.",
    )
    ap.add_argument(
        "--zh-future",
        type=str,
        default=None,
        help="Override Zhongshan future CSV.",
    )

    ap.add_argument(
        "--split",
        type=str,
        choices=["auto", "val", "test"],
        default="auto",
        help="Pick val/test artifacts when using --*-src.",
    )

    ap.add_argument(
        "--year-val",
        type=int,
        default=2022,
        help="Validation year for observed/predicted maps.",
    )
    ap.add_argument(
        "--years-forecast",
        nargs="+",
        type=int,
        default=[2025, 2026],
        help="Forecast years to plot as additional columns.",
    )
    ap.add_argument(
        "--cumulative",
        action="store_true",
        help="Use cumulative q50 for forecast years.",
    )

    ap.add_argument(
        "--grid-res",
        type=int,
        default=300,
        help="Raster resolution (NxN bins).",
    )
    ap.add_argument(
        "--clip",
        type=float,
        default=98.0,
        help="Percentile clip for shared color scale.",
    )
    ap.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name.",
    )

    ap.add_argument(
        "--hotspot",
        type=str,
        choices=["none", "absolute", "delta"],
        default="delta",
        help="Hotspot contour mode.",
    )
    ap.add_argument(
        "--hotspot-quantile",
        type=float,
        default=0.90,
        help="Quantile threshold for hotspots.",
    )
    ap.add_argument(
        "--hotspot-out",
        type=str,
        default=None,
        help="CSV output for hotspot points.",
    )

    ap.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)
    ap.add_argument(
        "--font",
        type=int,
        default=cfg.PAPER_FONT,
    )

    utils.add_plot_text_args(
        ap,
        default_out="fig6-spatial-forecasts",
    )


def plot_fig6_spatial_forecasts_main(
    argv: Optional[List[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        prog="plot-spatial-forecasts",
        description="Fig. 6 spatial maps + forecasts.",
    )
    _add_fig6_args(ap)
    args = ap.parse_args(argv)

    show_legend = utils.str_to_bool(args.show_legend, True)
    show_title = utils.str_to_bool(args.show_title, True)
    show_pt = utils.str_to_bool(args.show_panel_titles, True)

    cities0 = utils.resolve_cities(args)
    if not cities0:
        cities0 = ["Nansha", "Zhongshan"]

    want_ns = "Nansha" in cities0
    want_zh = "Zhongshan" in cities0

    if not want_ns and not want_zh:
        want_ns = True
        want_zh = True

    cities: List[Dict[str, Any]] = []
    if want_ns:
        cities.append(
            _resolve_city(
                city="Nansha",
                src=args.ns_src,
                calib=args.ns_calib,
                future=args.ns_future,
                split=args.split,
            )
        )
    if want_zh:
        cities.append(
            _resolve_city(
                city="Zhongshan",
                src=args.zh_src,
                calib=args.zh_calib,
                future=args.zh_future,
                split=args.split,
            )
        )

    plot_fig6_spatial_forecasts(
        cities=cities,
        year_val=int(args.year_val),
        years_fore=list(args.years_forecast),
        cumulative=bool(args.cumulative),
        grid_res=int(args.grid_res),
        clip=float(args.clip),
        cmap_name=str(args.cmap),
        hotspot_mode=str(args.hotspot),
        hotspot_q=float(args.hotspot_quantile),
        out=str(args.out),
        out_hotspots=args.hotspot_out,
        dpi=int(args.dpi),
        font=int(args.font),
        show_legend=bool(show_legend),
        show_title=bool(show_title),
        show_panel_titles=bool(show_pt),
        title=args.title,
    )


def main(argv: Optional[List[str]] = None) -> None:
    plot_fig6_spatial_forecasts_main(argv)


if __name__ == "__main__":
    main()
