from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from . import config as cfg
from . import utils


def _pick_rmse(
    phys: Dict[str, Any],
    diag: Dict[str, Any],
    mse: float,
) -> float:
    rmse = float("nan")

    pm = (phys.get("point_metrics") or {})
    if "rmse" in pm:
        rmse = utils.to_float(pm.get("rmse"))

    me = (phys.get("metrics_evaluate") or {})
    if np.isnan(rmse):
        rmse = utils.to_float(me.get("subs_pred_rmse"))

    flat = utils.flatten_eval_diag(diag) if diag else {}
    if np.isnan(rmse):
        rmse = utils.to_float(flat.get("rmse"))

    if np.isnan(rmse) and np.isfinite(mse):
        rmse = float(np.sqrt(max(0.0, float(mse))))

    return float(rmse)


def _collect_one(
    *,
    city: str,
    variant: str,
    src: Path,
) -> Dict[str, Any]:
    phys_p = utils.find_phys_json(src)
    diag_p = utils.find_latest(
        src,
        cfg.PATTERNS["eval_diag_json"],
    )

    phys = utils.safe_load_json(phys_p)
    diag = utils.safe_load_json(diag_p)

    phys = utils.phys_json_to_mm(phys)

    r2, mae, mse = utils.pick_point_metrics(phys, diag)
    rmse = _pick_rmse(phys, diag, mse)

    cov, shp = utils.pick_interval_metrics(phys, diag)

    return {
        "city": city,
        "variant": variant,
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "coverage80": cov,
        "sharpness80": shp,
        "phys_json": str(phys_p) if phys_p else "",
        "diag_json": str(diag_p) if diag_p else "",
    }


def collect_fig3_metrics(
    *,
    cities: List[str],
    ns_with: Optional[str],
    ns_no: Optional[str],
    zh_with: Optional[str],
    zh_no: Optional[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for c in cities:
        if c == "Nansha":
            if ns_with:
                rows.append(
                    _collect_one(
                        city=c,
                        variant="with-phys",
                        src=Path(ns_with).expanduser(),
                    )
                )
            if ns_no:
                rows.append(
                    _collect_one(
                        city=c,
                        variant="no-phys",
                        src=Path(ns_no).expanduser(),
                    )
                )
        if c == "Zhongshan":
            if zh_with:
                rows.append(
                    _collect_one(
                        city=c,
                        variant="with-phys",
                        src=Path(zh_with).expanduser(),
                    )
                )
            if zh_no:
                rows.append(
                    _collect_one(
                        city=c,
                        variant="no-phys",
                        src=Path(zh_no).expanduser(),
                    )
                )

    return pd.DataFrame(rows)


def _add_panel_label(
    ax: plt.Axes,
    letter: str,
    enabled: bool,
) -> None:
    if not enabled:
        return

    ax.text(
        -0.14,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _bar_values(
    ax: plt.Axes,
    bars: Any,
    fmt: str,
    enabled: bool,
) -> None:
    if not enabled:
        return

    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
            )


def plot_fig3_core_ablation(
    df: pd.DataFrame,
    *,
    cities: List[str],
    err_metric: str,
    out: str,
    out_csv: str,
    out_tex: Optional[str],
    out_xlsx: Optional[str],
    dpi: int,
    show_legend: bool,
    show_labels: bool,
    show_ticklabels: bool,
    show_title: bool,
    show_panel_titles: bool,
    show_values: bool,
    show_panel_labels: bool,
    title: Optional[str],
) -> None:
    utils.ensure_script_dirs()
    utils.set_paper_style()

    df = df.copy()
    df["city"] = df["city"].map(utils.canonical_city)

    if err_metric not in {"rmse", "mse"}:
        raise ValueError("err_metric must be rmse or mse")

    err_key = err_metric
    err_lab = "RMSE" if err_metric == "rmse" else "MSE"

    unit = "mm"
    err_unit = "mm²" if err_metric == "mse" else "mm"

    def _vals(metric: str, variant: str) -> np.ndarray:
        outv: List[float] = []
        for c in cities:
            sub = df[(df["city"] == c) & (df["variant"] == variant)]
            if sub.empty:
                outv.append(float("nan"))
            else:
                outv.append(float(sub.iloc[0][metric]))
        return np.asarray(outv, dtype=float)

    x = np.arange(len(cities))
    colors = [cfg.CITY_COLORS.get(c, "#777777") for c in cities]

    fig = plt.figure(figsize=(10.0, 4.5))
    gs = GridSpec(
        2,
        4,
        figure=fig,
        wspace=0.9,
        hspace=0.6,
    )

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    ax_d = fig.add_subplot(gs[0, 2])
    ax_e = fig.add_subplot(gs[1, 2])
    ax_f = fig.add_subplot(gs[0, 3])
    ax_g = fig.add_subplot(gs[1, 3])

    _add_panel_label(ax_a, "a", show_panel_labels)
    _add_panel_label(ax_b, "b", show_panel_labels)
    _add_panel_label(ax_c, "c", show_panel_labels)
    _add_panel_label(ax_d, "d", show_panel_labels)
    _add_panel_label(ax_e, "e", show_panel_labels)
    _add_panel_label(ax_f, "f", show_panel_labels)
    _add_panel_label(ax_g, "g", show_panel_labels)

    r2 = _vals("r2", "with-phys")
    mae = _vals("mae", "with-phys")
    err = _vals(err_key, "with-phys")

    bars = ax_a.bar(x, r2, color=colors, edgecolor="none")
    _bar_values(ax_a, bars, "{:.2f}", show_values)
    if show_panel_titles:
        ax_a.set_title("Core (with physics)")
    if show_labels:
        ax_a.set_ylabel("R²")
    if show_ticklabels:
        ax_a.set_xticks(x, cities)
    else:
        ax_a.set_xticks([])
        ax_a.tick_params(labelleft=False)

    bars = ax_b.bar(x, mae, color=colors, edgecolor="none")
    _bar_values(ax_b, bars, "{:.2f}", show_values)
    if show_panel_titles:
        ax_b.set_title(f"MAE (↓, {unit})")
    if show_labels:
        ax_b.set_ylabel(f"MAE ({unit})")
    if show_ticklabels:
        ax_b.set_xticks(x, cities)
    else:
        ax_b.set_xticks([])
        ax_b.tick_params(labelleft=False)

    bars = ax_c.bar(x, err, color=colors, edgecolor="none")
    _bar_values(ax_c, bars, "{:.2f}", show_values)
    if show_panel_titles:
        ax_c.set_title(f"{err_lab} (↓, {err_unit})")
    if show_labels:
        ax_c.set_ylabel(f"{err_lab} ({err_unit})")
    if show_ticklabels:
        ax_c.set_xticks(x, cities)
    else:
        ax_c.set_xticks([])
        ax_c.tick_params(labelleft=False)

    width = 0.36
    with_r2 = _vals("r2", "with-phys")
    no_r2 = _vals("r2", "no-phys")

    def _grouped(
        ax: plt.Axes,
        metric: str,
        fmt: str,
        title_txt: str,
    ) -> Tuple[Any, Any]:
        with_v = _vals(metric, "with-phys")
        no_v = _vals(metric, "no-phys")

        b1 = ax.bar(
            x - width / 2.0,
            with_v,
            width=width,
            color=colors,
            edgecolor="none",
            label="with physics",
        )
        b2 = ax.bar(
            x + width / 2.0,
            no_v,
            width=width,
            color=colors,
            edgecolor="black",
            fill=False,
            hatch="///",
            label="no physics",
        )

        if show_values:
            _bar_values(ax, b1, fmt, True)
            _bar_values(ax, b2, fmt, True)

        if show_ticklabels:
            ax.set_xticks(x, cities)
        else:
            ax.set_xticks([])
            ax.tick_params(labelleft=False)

        if show_panel_titles:
            ax.set_title(title_txt)

        ax.grid(axis="y", alpha=0.2)
        return b1, b2

    _grouped(ax_d, "r2", "{:.2f}", "R² (with vs no)")
    _grouped(ax_e, err_key, "{:.2f}", f"{err_lab} (with vs no)")
    _grouped(ax_f, "coverage80", "{:.3f}", "Coverage@80% (↑)")
    _grouped(ax_g, "sharpness80", "{:.3f}", "Sharpness@80% (↓)")

    if show_labels:
        ax_d.set_ylabel("R²")
        ax_e.set_ylabel(f"{err_lab} ({err_unit})")
        ax_f.set_ylabel("Coverage")
        ax_g.set_ylabel(f"Sharpness ({unit})")

    if show_legend:
        h, lab = ax_d.get_legend_handles_labels()
        fig.legend(
            h,
            lab,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=False,
        )

    if show_title:
        ttl = utils.resolve_title(
            default="Fig. 3 — Core & ablation",
            title=title,
        )
        fig.suptitle(ttl, x=0.02, ha="left")

    fig_p = utils.resolve_fig_out(out)
    if fig_p.suffix:
        fig_p = fig_p.with_suffix("")

    fig.savefig(str(fig_p) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(fig_p) + ".svg", bbox_inches="tight")
    plt.close(fig)

    out_csv_p = utils.resolve_out_out(out_csv)
    df.to_csv(out_csv_p, index=False)

    if out_xlsx:
        out_xlsx_p = utils.resolve_out_out(out_xlsx)
        df.to_excel(out_xlsx_p, index=False)

    if out_tex:
        out_tex_p = utils.resolve_out_out(out_tex)
        tex = df.to_latex(index=False)
        out_tex_p.write_text(tex, encoding="utf-8")

    print(f"[OK] wrote {fig_p}.png/.svg")
    print(f"[OK] wrote {out_csv_p}")


def _add_plot_fig3_args(ap: argparse.ArgumentParser) -> None:
    utils.add_city_flags(ap, default_both=True)

    ap.add_argument("--ns-with", type=str, default=None)
    ap.add_argument("--ns-no", type=str, default=None)
    ap.add_argument("--zh-with", type=str, default=None)
    ap.add_argument("--zh-no", type=str, default=None)

    ap.add_argument(
        "--err-metric",
        type=str,
        choices=["rmse", "mse"],
        default="rmse",
    )

    ap.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)

    ap.add_argument(
        "--out-csv",
        type=str,
        default="ext-table-fig3-metrics.csv",
    )
    ap.add_argument(
        "--out-tex",
        type=str,
        default="ext-table-fig3-metrics.tex",
    )
    ap.add_argument(
        "--out-xlsx",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--write-tex",
        type=str,
        default="true",
    )

    ap.add_argument(
        "--show-values",
        type=str,
        default="true",
    )
    ap.add_argument(
        "--show-panel-labels",
        type=str,
        default="true",
    )

    utils.add_plot_text_args(
        ap,
        default_out="fig3-core-ablation",
    )


def plot_fig3_core_ablation_main(
    argv: Optional[List[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        prog="plot-core-ablation",
        description="Fig. 3 core + ablation.",
    )
    _add_plot_fig3_args(ap)
    args = ap.parse_args(argv)

    cities = utils.resolve_cities(args)
    if not cities:
        cities = ["Nansha", "Zhongshan"]

    df = collect_fig3_metrics(
        cities=cities,
        ns_with=args.ns_with,
        ns_no=args.ns_no,
        zh_with=args.zh_with,
        zh_no=args.zh_no,
    )

    show_legend = utils.str_to_bool(args.show_legend, default=True)
    show_labels = utils.str_to_bool(args.show_labels, default=True)
    show_ticks = utils.str_to_bool(
        args.show_ticklabels,
        default=True,
    )
    show_title = utils.str_to_bool(args.show_title, default=True)
    show_pt = utils.str_to_bool(
        args.show_panel_titles,
        default=True,
    )
    show_vals = utils.str_to_bool(args.show_values, default=True)
    show_pl = utils.str_to_bool(
        args.show_panel_labels,
        default=True,
    )

    out_tex = args.out_tex
    if not utils.str_to_bool(args.write_tex, default=True):
        out_tex = None

    plot_fig3_core_ablation(
        df,
        cities=cities,
        err_metric=args.err_metric,
        out=args.out,
        out_csv=args.out_csv,
        out_tex=out_tex,
        out_xlsx=args.out_xlsx,
        dpi=int(args.dpi),
        show_legend=show_legend,
        show_labels=show_labels,
        show_ticklabels=show_ticks,
        show_title=show_title,
        show_panel_titles=show_pt,
        show_values=show_vals,
        show_panel_labels=show_pl,
        title=args.title,
    )
    
def main(argv: Optional[List[str]] = None) -> None:
    plot_fig3_core_ablation_main(argv)


if __name__ == "__main__":
    main()
