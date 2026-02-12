# scripts/plot_driver_response.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from . import config as cfg
from . import utils


def robust_trend(
    x: pd.Series,
    y: pd.Series,
    *,
    nbins: int = 30,
    min_n: int = 20,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    xv = pd.to_numeric(x, errors="coerce").to_numpy()
    yv = pd.to_numeric(y, errors="coerce").to_numpy()

    m = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[m]
    yv = yv[m]

    if xv.size < max(nbins * 2, 50):
        return None, None

    q = np.linspace(0.0, 1.0, nbins + 1)
    bx = np.quantile(xv, q)

    xi: List[float] = []
    yi: List[float] = []

    for k in range(nbins):
        lo = bx[k]
        hi = bx[k + 1]
        sel = (xv >= lo) & (xv <= hi)
        if int(sel.sum()) >= int(min_n):
            xi.append(float(np.nanmedian(xv[sel])))
            yi.append(float(np.nanmedian(yv[sel])))

    if len(xi) < 3:
        return None, None

    return np.asarray(xi), np.asarray(yi)


def _panel(
    ax,
    df: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    show_ylabel: bool,
    gridsize: int,
    vmin: int,
    trend: bool,
    trend_bins: int,
    trend_min_n: int,
    show_labels: bool,
    show_ticklabels: bool,
) -> object:
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)

    hb = ax.hexbin(
        x[m],
        y[m],
        gridsize=gridsize,
        mincnt=vmin,
        cmap="viridis",
        linewidths=0.0,
        bins="log",
    )

    if trend:
        xi, yi = robust_trend(
            x,
            y,
            nbins=trend_bins,
            min_n=trend_min_n,
        )
        if xi is not None:
            ax.plot(
                xi,
                yi,
                color="k",
                linewidth=1.0,
                alpha=0.9,
            )

    if show_labels:
        ax.set_xlabel(utils.label(xcol))
    else:
        ax.set_xlabel("")

    if show_ylabel and show_labels:
        ax.set_ylabel(utils.label(ycol))
    else:
        ax.set_ylabel("")

    if not show_ticklabels:
        ax.tick_params(
            labelbottom=False,
            labelleft=False,
        )
    else:
        if not show_ylabel:
            ax.tick_params(labelleft=False)

    return hb


def plot_driver_response(
    df: pd.DataFrame,
    *,
    cities: List[str],
    drivers: List[str],
    ycol: str,
    sharey: str,
    out: str,
    gridsize: int,
    vmin: int,
    trend: bool,
    trend_bins: int,
    trend_min_n: int,
    show_legend: bool,
    show_labels: bool,
    show_ticklabels: bool,
    show_title: bool,
    show_panel_titles: bool,
    title: Optional[str],
) -> None:
    utils.ensure_script_dirs()
    utils.set_paper_style()

    nrows = len(cities)
    ncols = len(drivers)

    fig_h = 2.6 * float(nrows)
    fig = plt.figure(figsize=(7.0, fig_h))

    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        wspace=0.30,
        hspace=0.35,
    )

    axes: List[List[object]] = [
        [None for _ in range(ncols)]
        for _ in range(nrows)
    ]

    axes[0][0] = fig.add_subplot(gs[0, 0])

    for j in range(1, ncols):
        ref = axes[0][0] if sharey in ("row", "all") else None
        axes[0][j] = fig.add_subplot(gs[0, j], sharey=ref)

    for i in range(1, nrows):
        ref0 = axes[0][0] if sharey == "all" else None
        axes[i][0] = fig.add_subplot(gs[i, 0], sharey=ref0)

        for j in range(1, ncols):
            if sharey == "all":
                ref = axes[0][0]
            elif sharey == "row":
                ref = axes[i][0]
            else:
                ref = None
            axes[i][j] = fig.add_subplot(gs[i, j], sharey=ref)

    last_hb = None

    for i, city in enumerate(cities):
        dfi = df.loc[df["city"] == city].copy()

        for j, xcol in enumerate(drivers):
            show_ylabel = (j == 0) and (sharey != "all" or i == 0)

            hb = _panel(
                axes[i][j],
                dfi,
                xcol=xcol,
                ycol=ycol,
                show_ylabel=show_ylabel,
                gridsize=gridsize,
                vmin=vmin,
                trend=trend,
                trend_bins=trend_bins,
                trend_min_n=trend_min_n,
                show_labels=show_labels,
                show_ticklabels=show_ticklabels,
            )
            last_hb = hb

            if j == 0 and show_panel_titles:
                axes[i][j].set_title(
                    city,
                    loc="left",
                    fontweight="bold",
                )
            elif j == 0 and not show_panel_titles:
                axes[i][j].set_title("")

    if show_legend and last_hb is not None:
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cb = plt.colorbar(last_hb, cax=cax)
        if show_labels:
            cb.set_label("log$_{10}$(count)")
        else:
            cb.set_label("")
            cb.ax.set_ylabel("")

        if not show_ticklabels:
            cb.ax.tick_params(labelleft=False)

    default_title = (
        "Driver–response sanity: response vs drivers"
    )
    st = utils.resolve_title(
        default=default_title,
        title=title,
    )

    if show_title:
        fig.suptitle(
            st,
            x=0.02,
            y=0.99,
            ha="left",
        )

    base = utils.resolve_fig_out(out)
    if base.suffix:
        base = base.with_suffix("")

    fig.savefig(str(base) + ".png", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote {base}.png/.svg")


def _parse_list(s: str) -> List[str]:
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p]


def _add_plot_driver_response_args(ap) -> None:
    ap.add_argument(
        "--src",
        type=str,
        required=True,
        help="CSV file or directory.",
    )
    ap.add_argument(
        "--file",
        type=str,
        default=None,
        help="Combined CSV name (when --src is dir).",
    )
    ap.add_argument(
        "--ns-file",
        type=str,
        default="nansha_final_main_std.harmonized.csv",
    )
    ap.add_argument(
        "--zh-file",
        type=str,
        default="zhongshan_final_main_std.harmonized.csv",
    )

    utils.add_city_flags(ap, default_both=True)

    ap.add_argument(
        "--year",
        "-y",
        type=str,
        default="all",
    )
    ap.add_argument(
        "--sample-frac",
        type=float,
        default=None,
    )
    ap.add_argument(
        "--sample-n",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--drivers",
        type=str,
        default=",".join(cfg.DRIVER_RESPONSE_DEFAULT_DRIVERS),
        help="Comma list of driver columns.",
    )

    ap.add_argument(
        "--response",
        "--col",
        dest="response",
        type=str,
        default=None,
        help="Response column (e.g. head_m).",
    )
    ap.add_argument(
        "--subs-kind",
        type=str,
        choices=["cum", "rate"],
        default="cum",
        help="If --response is not set, choose "
        "subsidence_cum or subsidence.",
    )

    ap.add_argument(
        "--sharey",
        type=str,
        choices=["none", "row", "all"],
        default="row",
    )

    ap.add_argument(
        "--gridsize",
        type=int,
        default=60,
    )
    ap.add_argument(
        "--vmin",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--trend",
        type=str,
        default="true",
    )
    ap.add_argument(
        "--trend-bins",
        type=int,
        default=30,
    )
    ap.add_argument(
        "--trend-min-n",
        type=int,
        default=20,
    )

    utils.add_plot_text_args(
        ap,
        default_out="figS2_driver_response",
    )


def plot_driver_response_main(
    argv: Optional[List[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        prog="plot-driver-response",
        description="Driver–response sanity plot.",
    )
    _add_plot_driver_response_args(ap)
    args = ap.parse_args(argv)

    cities = utils.resolve_cities(args)
    if not cities:
        cities = ["Nansha", "Zhongshan"]

    src = Path(args.src).expanduser()

    df = utils.load_dataset_any(
        src,
        file=args.file,
        ns_file=args.ns_file,
        zh_file=args.zh_file,
    )

    if "city" not in df.columns:
        raise KeyError("Dataset must contain 'city' column.")

    df = utils.filter_year(df, args.year)
    df = utils.sample_df(
        df,
        sample_frac=args.sample_frac,
        sample_n=args.sample_n,
    )

    drivers = _parse_list(args.drivers)

    utils.ensure_columns(
        df,
        aliases=cfg.COLUMN_ALIASES,
    )

    ycol = args.response
    if not ycol:
        if args.subs_kind == "cum":
            ycol = "subsidence_cum"
        else:
            ycol = "subsidence"

    if ycol not in df.columns:
        alts = cfg.COLUMN_ALIASES.get(ycol, ())
        raise KeyError(
            f"Missing response '{ycol}'. "
            f"Aliases tried: {list(alts)}"
        )

    for d in drivers:
        if d not in df.columns:
            alts = cfg.COLUMN_ALIASES.get(d, ())
            raise KeyError(
                f"Missing driver '{d}'. "
                f"Aliases tried: {list(alts)}"
            )

    trend = utils.str_to_bool(args.trend)

    show_legend = utils.str_to_bool(args.show_legend)
    show_labels = utils.str_to_bool(args.show_labels)
    show_ticks = utils.str_to_bool(args.show_ticklabels)
    show_title = utils.str_to_bool(args.show_title)
    show_panels = utils.str_to_bool(args.show_panel_titles)

    plot_driver_response(
        df,
        cities=cities,
        drivers=drivers,
        ycol=ycol,
        sharey=args.sharey,
        out=args.out,
        gridsize=int(args.gridsize),
        vmin=int(args.vmin),
        trend=trend,
        trend_bins=int(args.trend_bins),
        trend_min_n=int(args.trend_min_n),
        show_legend=show_legend,
        show_labels=show_labels,
        show_ticklabels=show_ticks,
        show_title=show_title,
        show_panel_titles=show_panels,
        title=args.title,
    )
    
def main(argv: Optional[List[str]] = None) -> None:
    plot_driver_response_main(argv)


if __name__ == "__main__":
    main()
