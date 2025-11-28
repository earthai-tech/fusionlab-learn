# fusionlab/tools/app/xfer_view.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Helpers to build cross-city transferability figures for the GUI.

Two main view modes:

1. 'calib_panel'  (Figure Sx style)
   - From xfer_results.csv
   - Shows overall MAE/R² vs calibration + coverage–sharpness panels.

2. 'summary_panel' (Supp Fig S3 style)
   - From xfer_results.json
   - Shows per-horizon MAE + coverage + sharpness (and optional
     overall MSE/R²) and exports a "Table 3" CSV/TeX.

These functions are GUI-friendly wrappers around the original
nat.com scripts.
"""

from __future__ import annotations

import glob
import json
import os
# from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------
# Shared style / helpers
# ---------------------------------------------------------------------

# Generic palette: color for "City A" (source) and "City B" (target)
CITY_A_COLOR = "#1F78B4"   # was Nansha blue
CITY_B_COLOR = "#E31A1C"   # was Zhongshan red

# Keep for backward compatibility if anything else imports these
CITY_COLORS_LOWER = {
    "nansha": CITY_A_COLOR,
    "zhongshan": CITY_B_COLOR,
}
CITY_COLORS_CANON = {
    "Nansha": CITY_A_COLOR,
    "Zhongshan": CITY_B_COLOR,
}


CAL_ORDER = ["none", "source", "target"]
CAL_LABELS = {
    "none": "No calibration",
    "source": "Source-based",
    "target": "Target-based",
}
CAL_MARKERS = {
    "none": "o",
    "source": "^",
    "target": "s",
}

BAR_ALPHA = 0.95
BAR_EDGE_DEFAULT = "white"


def _set_style(fontsize: int = 8, dpi: int = 150) -> None:
    """Compact, Nature-friendly Matplotlib defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------

def latest_xfer_csv(results_root: str) -> Optional[str]:
    """
    Find the newest xfer_results.csv under ``results_root``.

    Looks for: results_root/xfer/*_to_*/*/xfer_results.csv
    """
    pat = os.path.join(results_root, "xfer", "*_to_*", "*", "xfer_results.csv")
    paths = glob.glob(pat)
    if not paths:
        return None
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths[0]


def latest_xfer_json(results_root: str) -> Optional[str]:
    """
    Find the newest xfer_results.json under ``results_root``.
    """
    pat = os.path.join(results_root, "xfer", "*_to_*", "*", "xfer_results.json")
    paths = glob.glob(pat)
    if not paths:
        return None
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths[0]


# ---------------------------------------------------------------------
# VIEW 1  (Figure Sx style) – from CSV
# ---------------------------------------------------------------------

def _load_xfer_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"xfer CSV not found: {path}")
    df = pd.read_csv(path)
    for col in ("source_city", "target_city"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
    return df


def _extract_metric_arrays(
    df: pd.DataFrame,
    metric_col: str,
    split: str = "val",
    cal_order: List[str] = CAL_ORDER,
) -> Dict[str, List[float]]:
    """Return dict[direction] -> list of metric values in CAL_ORDER."""
    out: Dict[str, List[float]] = {}
    for direction in ("A_to_B", "B_to_A"):
        sub = df[(df["direction"] == direction) & (df["split"] == split)]
        vals: List[float] = []
        for cal in cal_order:
            row = sub[sub["calibration"] == cal]
            vals.append(
                float(row.iloc[0][metric_col]) if not row.empty else np.nan
            )
        out[direction] = vals
    return out

def _infer_dir_labels(df: pd.DataFrame, split: str) -> Dict[str, str]:
    """
    Build human-readable labels like "CityA → CityB" for A_to_B/B_to_A
    using source_city/target_city columns.
    """
    labels: Dict[str, str] = {}
    for direction in ("A_to_B", "B_to_A"):
        sub = df[(df["direction"] == direction) & (df["split"] == split)]
        if not sub.empty:
            row = sub.iloc[0]
            src = _canonical_city(row.get("source_city"))
            tgt = _canonical_city(row.get("target_city"))
            labels[direction] = f"{src} → {tgt}"
        else:
            labels[direction] = "A→B" if direction == "A_to_B" else "B→A"
    return labels

def _render_calib_panel(
    df: pd.DataFrame,
    *,
    split: str = "val",
    out_base: str,
    fontsize: int = 8,
    dpi: int = 150,
    add_legend: bool = True,
    add_suptitle: bool = False,
) -> Dict[str, Any]:
    """
    Render the "Figure Sx" style panel and save PNG + SVG.

    Returns a dict with paths + metadata.
    """
    _set_style(fontsize=fontsize, dpi=dpi)

    fig = plt.figure(figsize=(7.0, 4.0))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.2, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.4,
        hspace=0.5,
    )

    ax_mae = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[1, 0])
    ax_cov_ab = fig.add_subplot(gs[0, 1])  # A->B
    ax_cov_ba = fig.add_subplot(gs[1, 1])  # B->A

    mae_dict = _extract_metric_arrays(df, "overall_mae", split=split)
    r2_dict = _extract_metric_arrays(df, "overall_r2", split=split)
    
    x = np.arange(len(CAL_ORDER), dtype=float)
    width = 0.35
    
    # Get labels from the CSV (works for any city pair)
    labels_by_dir = _infer_dir_labels(df, split=split)
    
    # Palette: bars are coloured by target city (A→B uses City B colour, etc.)
    dir_info = {
        "A_to_B": {
            "label": labels_by_dir["A_to_B"],
            "color": CITY_B_COLOR,
        },
        "B_to_A": {
            "label": labels_by_dir["B_to_A"],
            "color": CITY_A_COLOR,
        },
    }

    # --- MAE vs calibration ---
    for i, direction in enumerate(("A_to_B", "B_to_A")):
        vals = mae_dict[direction]
        offset = (-0.5 + i) * width
        ax_mae.bar(
            x + offset,
            vals,
            width=width,
            label=dir_info[direction]["label"],
            color=dir_info[direction]["color"],
            alpha=0.85,
        )

    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels([CAL_LABELS[c] for c in CAL_ORDER], rotation=15)
    ax_mae.set_ylabel("Overall MAE (mm)")
    ax_mae.set_title("(a) Cross-city error vs calibration")
    if add_legend:
        ax_mae.legend(frameon=False, loc="upper right")

    # --- R² vs calibration ---
    for i, direction in enumerate(("A_to_B", "B_to_A")):
        vals = r2_dict[direction]
        offset = (-0.5 + i) * width
        ax_r2.bar(
            x + offset,
            vals,
            width=width,
            color=dir_info[direction]["color"],
            alpha=0.85,
        )

    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels([CAL_LABELS[c] for c in CAL_ORDER], rotation=15)
    ax_r2.set_ylabel(r"Overall $R^2$")
    ax_r2.axhline(0.0, color="#444444", linestyle="--", linewidth=0.8)

    all_r2_vals = np.array(
        r2_dict["A_to_B"] + r2_dict["B_to_A"],
        dtype=float,
    )
    finite_r2 = all_r2_vals[np.isfinite(all_r2_vals)]
    if finite_r2.size > 0:
        ymin = float(finite_r2.min())
        ymax = float(finite_r2.max())
        ax_r2.set_ylim(min(ymin * 1.1, -0.5), max(ymax, 0.1))

    # --- coverage vs sharpness ---
    cov_dict = _extract_metric_arrays(df, "coverage80", split=split)
    shp_dict = _extract_metric_arrays(df, "sharpness80", split=split)
    all_cov: List[float] = []
    all_shp: List[float] = []

    def _plot_cov_sharp(ax, direction: str) -> None:
        covs = np.asarray(cov_dict[direction], dtype=float)
        shps = np.asarray(shp_dict[direction], dtype=float)
        mask = np.isfinite(covs) & np.isfinite(shps)

        for j, cal in enumerate(CAL_ORDER):
            if not np.isfinite(covs[j]) or not np.isfinite(shps[j]):
                continue
            marker = CAL_MARKERS.get(cal, "o")
            # Target colour: B for A→B, A for B→A
            color = CITY_B_COLOR if direction == "A_to_B" else CITY_A_COLOR
            ax.scatter(
                shps[j],
                covs[j],
                marker=marker,
                s=42,
                facecolors=color,
                edgecolors="k",
                linewidths=0.5,
                alpha=0.9,
            )

        if mask.sum() >= 2:
            tgt_city = "zhongshan" if direction == "A_to_B" else "nansha"
            ax.plot(
                shps[mask],
                covs[mask],
                color=CITY_COLORS_LOWER[tgt_city],
                linewidth=0.7,
                alpha=0.8,
            )

        ax.axhline(0.80, color="#444444", linestyle="--", linewidth=0.8)
        all_cov.extend(list(covs[mask]))
        all_shp.extend(list(shps[mask]))

    _plot_cov_sharp(ax_cov_ab, "A_to_B")
    _plot_cov_sharp(ax_cov_ba, "B_to_A")

    for ax in (ax_cov_ab, ax_cov_ba):
        ax.set_xlabel("Sharpness (80% interval, mm)")
        ax.set_ylabel("Coverage (80% interval)")

    if all_cov and all_shp:
        cov_arr = np.asarray(all_cov, float)
        shp_arr = np.asarray(all_shp, float)
        ymin = max(0.0, float(np.min(cov_arr) - 0.05))
        ymax = min(1.0, float(np.max(cov_arr) + 0.05))
        xmin = float(np.min(shp_arr))
        xmax = float(np.max(shp_arr))
        pad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
        xmin -= pad
        xmax += pad
        for ax in (ax_cov_ab, ax_cov_ba):
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(xmin, xmax)

    ax_cov_ab.set_title(
        f"(b) Coverage–sharpness: {labels_by_dir['A_to_B']}"
    )
    ax_cov_ba.set_title(
        f"(c) Coverage–sharpness: {labels_by_dir['B_to_A']}"
    )


    if add_legend:
        cal_handles = [
            Line2D(
                [],
                [],
                marker=CAL_MARKERS[c],
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="k",
                markersize=5,
                label=CAL_LABELS[c],
            )
            for c in CAL_ORDER
        ]
        ax_cov_ba.legend(
            handles=cal_handles,
            frameon=False,
            loc="lower right",
            title="Calibration",
        )

    if add_suptitle:
        fig.suptitle(
            "Cross-city transferability of GeoPriorSubsNet",
            x=0.02,
            y=0.99,
            ha="left",
        )

    png = f"{out_base}.png"
    svg = f"{out_base}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    return {
        "view_kind": "calib_panel",
        "split": split,
        "png_path": png,
        "svg_path": svg,
    }


def make_transferability_panel_from_csv(
    xfer_csv: str,
    *,
    split: str = "val",
    out_base: Optional[str] = None,
    fontsize: int = 8,
    dpi: int = 150,
    add_legend: bool = True,
    add_suptitle: bool = False,
) -> Dict[str, Any]:
    """
    Public wrapper: build Figure-Sx-style panel from a CSV.

    Parameters
    ----------
    xfer_csv : str
        Path to xfer_results.csv produced by xfer_matrix.
    split : {"val","test"}
        Which split to visualise.
    out_base : str, optional
        Base path for figure without extension.  If None, uses
        ``<dir(xfer_csv)>/xfer_transferability``.
    """
    df = _load_xfer_csv(xfer_csv)
    if out_base is None:
        out_base = os.path.join(
            os.path.dirname(xfer_csv),
            "xfer_transferability",
        )
    return _render_calib_panel(
        df,
        split=split,
        out_base=out_base,
        fontsize=fontsize,
        dpi=dpi,
        add_legend=add_legend,
        add_suptitle=add_suptitle,
    )


# ---------------------------------------------------------------------
# VIEW 2  (Supp Fig S3 style) – from JSON
# ---------------------------------------------------------------------


def _canonical_city(name: Optional[str]) -> str:
    """
    Normalise a city label read from results.

    - Handles None/empty safely
    - Title-cases the string
    """
    if not name:
        return ""
    n = str(name).strip()
    # simple normalisation; you can extend later if needed
    return n.replace("_", " ").title()

def get_direction_palette(
    city_a: Optional[str] = None,
    city_b: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Return a consistent colour / label mapping for A→B and B→A.

    Parameters
    ----------
    city_a : str, optional
        Name of City A (source city in the GUI).  Can be any string;
        will be normalised with :func:`_canonical_city`.
    city_b : str, optional
        Name of City B (target city in the GUI).

    Returns
    -------
    palette : dict
        Mapping with two entries, ``"A_to_B"`` and ``"B_to_A"``.
        Each entry is a dict with:

        - ``direction``: "A_to_B" or "B_to_A"
        - ``label``: "CityA → CityB" / "CityB → CityA"
        - ``source_city`` / ``target_city``: canonical names
        - ``facecolor``: fill colour for bars/markers
        - ``edgecolor``: edge colour for bars/markers
        - ``hatch``: hatch pattern for bar plots
        
    Examples 
    ---------
    >>> from fusionlab.tools.app.xfer_view import get_direction_palette

    >>> palette = get_direction_palette(city_a, city_b)
    >>> style_AtoB = palette["A_to_B"]   # label, facecolor, edgecolor, hatch...
    >>> style_BtoA = palette["B_to_A"]

    """

    a = _canonical_city(city_a)
    b = _canonical_city(city_b)

    if not a:
        a = "City A"
    if not b:
        b = "City B"

    return {
        "A_to_B": {
            "direction": "A_to_B",
            "label": f"{a} \u2192 {b}",
            "source_city": a,
            "target_city": b,
            # by convention: coloured by *target* city
            "facecolor": CITY_B_COLOR,
            "edgecolor": CITY_A_COLOR,
            "hatch": "///",
        },
        "B_to_A": {
            "direction": "B_to_A",
            "label": f"{b} \u2192 {a}",
            "source_city": b,
            "target_city": a,
            "facecolor": CITY_A_COLOR,
            "edgecolor": CITY_B_COLOR,
            "hatch": r"\\",
        },
    }

def _flatten_per_horizon(
    col: str,
    d: Dict[str, Any],
    H_guess: int = 6,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    dph = d.get(col) or {}
    if isinstance(dph, dict) and dph:
        keys = sorted(
            dph.keys(),
            key=lambda s: int(str(s).strip("H")),
        )
        for k in keys:
            out[f"{col}.{k}"] = dph.get(k)
    else:
        for i in range(1, H_guess + 1):
            out[f"{col}.H{i}"] = np.nan
    return out


def _collect_rows(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        base = {
            "direction": r.get("direction"),
            "source_city": r.get("source_city"),
            "target_city": r.get("target_city"),
            "split": r.get("split"),
            "calibration": r.get("calibration"),
            "coverage80": r.get("coverage80"),
            "sharpness80": r.get("sharpness80"),
            "overall_mae": r.get("overall_mae"),
            "overall_mse": r.get("overall_mse"),
            "overall_r2": r.get("overall_r2"),
        }
        phys = r.get("physics") or {}
        base["epsilon_prior"] = phys.get("epsilon_prior")
        base["epsilon_cons"] = phys.get("epsilon_cons")

        ev = r.get("keras_eval_scaled") or {}
        base["loss"] = ev.get("loss")
        base["subs_pred_mae"] = ev.get("subs_pred_mae")
        base["gwl_pred_mae"] = ev.get("gwl_pred_mae")

        base.update(_flatten_per_horizon("per_horizon_mae", r))
        base.update(_flatten_per_horizon("per_horizon_r2", r))
        rows.append(base)
    return pd.DataFrame(rows)


def _choose_split(df: pd.DataFrame, prefer: Optional[str]) -> str:
    splits = set(df["split"].unique())
    if prefer and prefer in splits:
        return prefer
    return "test" if "test" in splits else "val"


def _choose_calibration(df: pd.DataFrame, prefer: Optional[str]) -> str:
    if prefer:
        return prefer
    for c in ("target", "source", "none"):
        if c in set(df["calibration"]):
            return c
    return str(df["calibration"].iloc[0])


def _pick_dir_rows(
    df: pd.DataFrame,
    split: str,
    calib: str,
    tag: str,
) -> pd.DataFrame:
    return df[
        (df["split"] == split)
        & (df["calibration"] == calib)
        & (df["direction"] == tag)
    ]

def _direction_style(block: pd.DataFrame) -> Dict[str, Any]:
    """
    Choose bar style purely from transfer direction.

    - A→B : face = City B colour, edge = City A colour, hatch=///
    - B→A : face = City A colour, edge = City B colour, hatch=\\
    """
    
    if block.empty:
        return {
            "facecolor": "0.7",
            "edgecolor": BAR_EDGE_DEFAULT,
            "hatch": None,
        }

    row = block.iloc[0]
    direction = str(row.get("direction") or "").strip()
    palette = get_direction_palette(
        row.get("source_city"),
        row.get("target_city"),
    )
    info = palette.get(direction, {})

    return {
        "facecolor": info.get("facecolor", "0.7"),
        "edgecolor": info.get("edgecolor", BAR_EDGE_DEFAULT),
        "hatch": info.get("hatch", None),
    }

# def _direction_style(block: pd.DataFrame) -> Dict[str, Any]:
#     """
#     Choose bar style purely from transfer direction.

#     - A→B : face = City B colour, edge = City A colour, hatch=///
#     - B→A : face = City A colour, edge = City B colour, hatch=\\
#     """
#     if block.empty:
#         return {
#             "facecolor": "0.7",
#             "edgecolor": BAR_EDGE_DEFAULT,
#             "hatch": None,
#         }

#     row = block.iloc[0]
#     direction = str(row.get("direction") or "").strip()

#     if direction == "A_to_B":
#         face = CITY_B_COLOR
#         edge = CITY_A_COLOR
#         hatch = "///"
#     elif direction == "B_to_A":
#         face = CITY_A_COLOR
#         edge = CITY_B_COLOR
#         hatch = "\\\\"
#     else:
#         face = "0.7"
#         edge = BAR_EDGE_DEFAULT
#         hatch = None

#     return {"facecolor": face, "edgecolor": edge, "hatch": hatch}


def _bar_ax(
    ax: plt.Axes,
    labels: List[str],
    values: List[float],
    style: Dict[str, Any],
    title: str,
    ylabel: Optional[str] = None,
) -> None:
    x = np.arange(len(labels))
    face = style.get("facecolor", "0.7")
    edge = style.get("edgecolor", BAR_EDGE_DEFAULT)
    hatch = style.get("hatch", None)

    ax.bar(
        x,
        values,
        color=face,
        alpha=BAR_ALPHA,
        edgecolor=edge,
        hatch=hatch,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(
        title,
        pad=8,
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    for xi, v in zip(x, values):
        if np.isfinite(v):
            ax.text(
                xi,
                v,
                f"{v:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def make_cross_transfer_from_json(
    xfer_json: str,
    *,
    out_dir: Optional[str] = None,
    prefer_split: Optional[str] = None,
    prefer_calibration: Optional[str] = None,
    show_overall: bool = True,
    fontsize: int = 8,
    dpi: int = 150,
    add_legend: bool = True,
) -> Dict[str, Any]:
    """
    Build a Supp-Fig-S3-style summary figure + Table 3 from a JSON file.

    Parameters
    ----------
    xfer_json : str
        Path to xfer_results.json produced by xfer_matrix.
    out_dir : str, optional
        Directory for outputs.  If None, uses dirname(xfer_json).
    prefer_split : {"val","test",None}
        Preferred split; default is "test if present, else val".
    prefer_calibration : {"target","source","none",None}
        Preferred calibration; default is target>source>none.
    show_overall : bool
        If True and fields present, add row of overall MSE/R² panels.
    """
    if out_dir is None:
        out_dir = os.path.dirname(xfer_json)
    os.makedirs(out_dir, exist_ok=True)

    _set_style(fontsize=fontsize, dpi=dpi)

    with open(xfer_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    df = _collect_rows(results)
    split = _choose_split(df, prefer_split)
    calib = _choose_calibration(df, prefer_calibration)

    AtoB = _pick_dir_rows(df, split, calib, "A_to_B")
    BtoA = _pick_dir_rows(df, split, calib, "B_to_A")

    style_A = _direction_style(AtoB)
    style_B = _direction_style(BtoA)

    # ---- Table 3 CSV + TeX ----------------------------------------
    keep_cols = [
        "direction",
        "source_city",
        "target_city",
        "split",
        "calibration",
        "subs_pred_mae",
        "overall_mae",
        "overall_mse",
        "overall_r2",
        "coverage80",
        "sharpness80",
        "epsilon_prior",
        "epsilon_cons",
    ]
    ph_mae = sorted(
        [c for c in df.columns if c.startswith("per_horizon_mae.")]
    )
    ph_r2 = sorted(
        [c for c in df.columns if c.startswith("per_horizon_r2.")]
    )
    keep_cols.extend(ph_mae + ph_r2)

    table_df = df[
        df["split"].eq(split) & df["calibration"].eq(calib)
    ][keep_cols].copy()

    csv_path = os.path.join(out_dir, "table3_cross_transfer.csv")
    tex_path = os.path.join(out_dir, "table3_cross_transfer.tex")
    table_df.to_csv(csv_path, index=False)

    with open(tex_path, "w", encoding="utf-8") as g:
        g.write("\\begin{table}[t]\n\\centering\n\\small\n")
        g.write("\\begin{tabular}{l l l l l r r r r r r r")
        g.write("".join([" r" for _ in (ph_mae + ph_r2)]))
        g.write("}\n\\toprule\n")

        g.write(
            "Direction & Source & Target & Split & Calib & "
            "MAE (scaled) & MAE & MSE & R$^2$ & Cov$_{80}$ & Sharp$_{80}$ & "
            "$\\varepsilon_{prior}$ & $\\varepsilon_{cons}$"
        )
        for c in ph_mae:
            g.write(
                f" & {c.replace('per_horizon_mae.', 'MAE ')}"
            )
        for c in ph_r2:
            g.write(
                f" & {c.replace('per_horizon_r2.', 'R$^2$ ')}"
            )
        g.write(" \\\\\n\\midrule\n")

        for _, r in table_df.iterrows():
            vals = [r.get(c) for c in keep_cols]
            row = []
            for v in vals:
                if isinstance(v, (float, int)) and pd.notna(v):
                    row.append(f"{v:.3g}")
                else:
                    row.append(str(v))
            g.write(" & ".join(row) + " \\\\\n")

        g.write("\\bottomrule\n\\end{tabular}\n")
        g.write(
            "\\caption{Cross-city transfer (preferred split/calibration). "
            "Scaled MAE are from Keras evaluate; overall metrics are "
            "in physical units.}\n"
        )
        g.write("\\label{tab:cross-transfer}\n\\end{table}\n")

    # ---- Figure ----------------------------------------------------
    Hlabels = [c.replace("per_horizon_mae.", "") for c in ph_mae]
    add_row = (
        show_overall
        and ("overall_mse" in df.columns)
        and ("overall_r2" in df.columns)
    )
    nrows = 3 if add_row else 2

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(8.2, 5.6 if not add_row else 7.6),
    )
    fig.subplots_adjust(hspace=0.38, wspace=0.26)

    axes = np.asarray(axes)

    # Row 1 – per-horizon MAE
    if Hlabels and not AtoB.empty:
        vals = [
            float(AtoB.iloc[0].get(f"per_horizon_mae.{h}", np.nan))
            for h in Hlabels
        ]
        axes[0, 0].set_ylim(
            0, np.nanmax(vals) * 1.15 if np.isfinite(np.nanmax(vals)) else None
        )
        _bar_ax(
            axes[0, 0],
            Hlabels,
            vals,
            style_A,
            title=(
                f"A→B MAE by horizon "
                f"({_canonical_city(AtoB.iloc[0]['source_city'])}→"
                f"{_canonical_city(AtoB.iloc[0]['target_city'])})"
            ),
            ylabel="MAE",
        )
    else:
        axes[0, 0].set_visible(False)

    if Hlabels and not BtoA.empty:
        vals = [
            float(BtoA.iloc[0].get(f"per_horizon_mae.{h}", np.nan))
            for h in Hlabels
        ]
        axes[0, 1].set_ylim(
            0, np.nanmax(vals) * 1.15 if np.isfinite(np.nanmax(vals)) else None
        )
        _bar_ax(
            axes[0, 1],
            Hlabels,
            vals,
            style_B,
            title=(
                f"B→A MAE by horizon "
                f"({_canonical_city(BtoA.iloc[0]['source_city'])}→"
                f"{_canonical_city(BtoA.iloc[0]['target_city'])})"
            ),
            ylabel=None,
        )
    else:
        axes[0, 1].set_visible(False)

    # Row 2 – coverage and sharpness
    cov_vals, cov_lbls, cov_styles = [], [], []
    if not AtoB.empty and pd.notna(AtoB.iloc[0]["coverage80"]):
        cov_vals.append(float(AtoB.iloc[0]["coverage80"]))
        cov_lbls.append("A→B")
        cov_styles.append(style_A)
    if not BtoA.empty and pd.notna(BtoA.iloc[0]["coverage80"]):
        cov_vals.append(float(BtoA.iloc[0]["coverage80"]))
        cov_lbls.append("B→A")
        cov_styles.append(style_B)

    if cov_lbls:
        for i, (lab, val, st) in enumerate(
            zip(cov_lbls, cov_vals, cov_styles)
        ):
            axes[1, 0].bar(
                i,
                val,
                color=st["facecolor"],
                alpha=BAR_ALPHA,
                edgecolor=st["edgecolor"],
                hatch=st["hatch"],
            )
            axes[1, 0].text(
                i,
                val,
                f"{val:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axes[1, 0].axhline(
            0.8,
            ls="--",
            lw=0.8,
            color="#666666",
            alpha=0.8,
        )
        axes[1, 0].set_xticks(range(len(cov_lbls)))
        axes[1, 0].set_xticklabels(cov_lbls)
        axes[1, 0].set_ylim(0.0, 1.0)
        axes[1, 0].set_ylabel("Coverage (80%)")
        axes[1, 0].set_title(
            "Interval coverage",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
    else:
        axes[1, 0].set_visible(False)

    shp_vals, shp_lbls, shp_styles = [], [], []
    if not AtoB.empty and pd.notna(AtoB.iloc[0]["sharpness80"]):
        shp_vals.append(float(AtoB.iloc[0]["sharpness80"]))
        shp_lbls.append("A→B")
        shp_styles.append(style_A)
    if not BtoA.empty and pd.notna(BtoA.iloc[0]["sharpness80"]):
        shp_vals.append(float(BtoA.iloc[0]["sharpness80"]))
        shp_lbls.append("B→A")
        shp_styles.append(style_B)

    if shp_lbls:
        ymax = max(shp_vals) * 1.15 if shp_vals else None
        for i, (lab, val, st) in enumerate(
            zip(shp_lbls, shp_vals, shp_styles)
        ):
            axes[1, 1].bar(
                i,
                val,
                color=st["facecolor"],
                alpha=BAR_ALPHA,
                edgecolor=st["edgecolor"],
                hatch=st["hatch"],
            )
            axes[1, 1].text(
                i,
                val,
                f"{val:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axes[1, 1].set_xticks(range(len(shp_lbls)))
        axes[1, 1].set_xticklabels(shp_lbls)
        axes[1, 1].set_ylabel("Sharpness (80%)")
        if ymax is not None:
            axes[1, 1].set_ylim(0.0, ymax)
        axes[1, 1].set_title(
            "Interval sharpness",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
    else:
        axes[1, 1].set_visible(False)

    # Optional row 3 – overall MSE / R²
    if add_row:

        def _bar_pair(ax, a_val, b_val, ylabel, title):
            vals, styles, lbls = [], [], []
            if pd.notna(a_val):
                vals.append(float(a_val))
                styles.append(style_A)
                lbls.append("A→B")
            if pd.notna(b_val):
                vals.append(float(b_val))
                styles.append(style_B)
                lbls.append("B→A")
            ymax_local = max(vals) * 1.15 if vals else None
            for i, (val, st) in enumerate(zip(vals, styles)):
                ax.bar(
                    i,
                    val,
                    color=st["facecolor"],
                    alpha=BAR_ALPHA,
                    edgecolor=st["edgecolor"],
                    hatch=st["hatch"],
                )
                ax.text(
                    i,
                    val,
                    f"{val:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.set_xticks(range(len(lbls)))
            ax.set_xticklabels(lbls)
            ax.set_ylabel(ylabel)
            if ymax_local is not None:
                ax.set_ylim(0.0, ymax_local)
            ax.set_title(
                title,
                loc="left",
                fontsize=11,
                fontweight="bold",
            )

        a_mse = AtoB.iloc[0]["overall_mse"] if not AtoB.empty else np.nan
        b_mse = BtoA.iloc[0]["overall_mse"] if not BtoA.empty else np.nan
        a_r2 = AtoB.iloc[0]["overall_r2"] if not AtoB.empty else np.nan
        b_r2 = BtoA.iloc[0]["overall_r2"] if not BtoA.empty else np.nan

        _bar_pair(
            axes[2, 0],
            a_mse,
            b_mse,
            ylabel="MSE",
            title="Overall MSE (physical units)",
        )
        _bar_pair(
            axes[2, 1],
            a_r2,
            b_r2,
            ylabel="R$^2$",
            title="Overall R$^2$ (physical units)",
        )

    # Cosmetics
    for ax in np.ravel(axes):
        if isinstance(ax, plt.Axes):
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Cross-city transfer performance",
        y=0.99,
        fontsize=12,
        fontweight="bold",
    )

    if add_legend and (not AtoB.empty or not BtoA.empty):
        handles: List[Patch] = []
        labels: List[str] = []
        if not AtoB.empty:
            hA = Patch(
                facecolor=style_A["facecolor"],
                edgecolor=style_A["edgecolor"],
                hatch=style_A["hatch"],
                label=(
                    f"{_canonical_city(AtoB.iloc[0]['source_city'])}"
                    f"→{_canonical_city(AtoB.iloc[0]['target_city'])}"
                ),
            )
            handles.append(hA)
            labels.append(hA.get_label())
        if not BtoA.empty:
            hB = Patch(
                facecolor=style_B["facecolor"],
                edgecolor=style_B["edgecolor"],
                hatch=style_B["hatch"],
                label=(
                    f"{_canonical_city(BtoA.iloc[0]['source_city'])}"
                    f"→{_canonical_city(BtoA.iloc[0]['target_city'])}"
                ),
            )
            handles.append(hB)
            labels.append(hB.get_label())

        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.98, 0.995),
                frameon=False,
            )

    png = os.path.join(out_dir, "supp_fig_s3_cross_transfer.png")
    pdf = os.path.join(out_dir, "supp_fig_s3_cross_transfer.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "view_kind": "summary_panel",
        "split": split,
        "calibration": calib,
        "png_path": png,
        "pdf_path": pdf,
        "table_csv": csv_path,
        "table_tex": tex_path,
    }

