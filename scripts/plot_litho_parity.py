# scripts/scripts.py
# -*- coding: utf-8 -*-
"""
Fig. S1 — Lithology parity across cities.

Left: normalized composition bars (Nansha vs Zhongshan).
Right: difference bars (Zhongshan - Nansha).

Args:
  --src   dataset directory
  --col   column name (default lithology_class)
  --year  all or a year integer
  --out   output stem/path (saved into scripts/figs/)
  -ns/-zh city codes (default uses both)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from . import config as cfg
from . import utils


CITY_COLORS = cfg.CITY_COLORS


def _erf(x: np.ndarray) -> np.ndarray:
    s = np.sign(x)
    x = np.abs(x)
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5 = -1.453152027, 1.061405429
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1)
        * t
        * np.exp(-x * x)
    )
    return s * y


def _phi(z: float) -> float:
    zz = np.array([z], dtype=float)
    return float(0.5 * (1.0 + _erf(zz / np.sqrt(2.0)))[0])


def chisq_cramers_v(
    counts_2xk: np.ndarray,
) -> Tuple[float, float, int, float]:
    rs = counts_2xk.sum(axis=1, keepdims=True)
    cs = counts_2xk.sum(axis=0, keepdims=True)
    tot = float(counts_2xk.sum())
    if tot <= 0.0:
        return np.nan, np.nan, 0, np.nan

    exp = (rs @ cs) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(((counts_2xk - exp) ** 2) / exp)

    dof = int((counts_2xk.shape[0] - 1) * (counts_2xk.shape[1] - 1))
    pval = np.nan
    if dof > 0 and np.isfinite(chi2):
        k = float(dof)
        z = (
            ((chi2 / k) ** (1.0 / 3.0))
            - (1.0 - 2.0 / (9.0 * k))
        ) / np.sqrt(2.0 / (9.0 * k))
        pval = 1.0 - _phi(float(z))
        pval = float(np.clip(pval, 0.0, 1.0))

    r, c = counts_2xk.shape
    denom = tot * max(1, min(r - 1, c - 1))
    cv = float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan
    return float(chi2), float(pval), dof, cv


def load_city_df(
    src: Path,
    filename: str,
    *,
    year: str = "all",
    sample_frac: float | None = None,
    sample_n: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    fp = (src / filename).expanduser()
    if not fp.exists():
        raise FileNotFoundError(str(fp))

    df = pd.read_csv(fp)

    if year != "all":
        df = df.loc[df["year"] == int(year)].copy()

    if sample_n is not None:
        n = min(int(sample_n), len(df))
        df = df.sample(n=n, random_state=seed)
    elif sample_frac is not None:
        f = float(sample_frac)
        f = max(0.0, min(1.0, f))
        if f < 1.0:
            df = df.sample(frac=f, random_state=seed)

    return df


def compute_proportions(
    ns: pd.DataFrame,
    zh: pd.DataFrame,
    *,
    col: str,
    top_n: int,
    group_others: bool,
) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    c_ns = ns[col].value_counts(dropna=False)
    c_zh = zh[col].value_counts(dropna=False)

    classes = list(set(c_ns.index.tolist()) | set(c_zh.index.tolist()))
    tot = {k: int(c_ns.get(k, 0)) + int(c_zh.get(k, 0)) for k in classes}
    classes = sorted(classes, key=lambda k: tot[k], reverse=True)

    core = classes[: int(top_n)]
    rest = classes[int(top_n) :]

    def _props(vc: pd.Series) -> pd.Series:
        s = float(vc.sum())
        if s <= 0.0:
            s = 1.0
        return vc / s

    p_ns = _props(c_ns)
    p_zh = _props(c_zh)

    data: List[tuple] = []
    for cls in core:
        data.append((cls, "Nansha", float(p_ns.get(cls, 0.0))))
        data.append((cls, "Zhongshan", float(p_zh.get(cls, 0.0))))

    if group_others and len(rest) > 0:
        ns_o = float(p_ns.loc[rest].sum())
        zh_o = float(p_zh.loc[rest].sum())
        data.append(("Others", "Nansha", ns_o))
        data.append(("Others", "Zhongshan", zh_o))
        core = core + ["Others"]

    dfp = pd.DataFrame(
        data,
        columns=["class", "city", "proportion"],
    )

    order = (
        dfp.groupby("class")["proportion"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )
    dfp["class"] = pd.Categorical(
        dfp["class"],
        categories=order,
        ordered=True,
    )
    dfp = dfp.sort_values(["class", "city"]).reset_index(drop=True)

    k = len(core)
    mat = np.zeros((2, k), dtype=float)

    for j, cls in enumerate(core):
        if cls == "Others":
            ns_c = float(c_ns.loc[rest].sum()) if len(rest) else 0.0
            zh_c = float(c_zh.loc[rest].sum()) if len(rest) else 0.0
        else:
            ns_c = float(c_ns.get(cls, 0.0))
            zh_c = float(c_zh.get(cls, 0.0))
        mat[0, j] = ns_c
        mat[1, j] = zh_c

    return dfp, core, mat


def _extract_props(
    dfp: pd.DataFrame,
    classes_order: List[str],
) -> Tuple[List[float], List[float]]:
    ns_props: List[float] = []
    zh_props: List[float] = []

    for cls in classes_order:
        m_ns = (dfp["class"] == cls) & (dfp["city"] == "Nansha")
        m_zh = (dfp["class"] == cls) & (dfp["city"] == "Zhongshan")
        ns_props.append(float(dfp.loc[m_ns, "proportion"].iloc[0]) if m_ns.any() else 0.0)
        zh_props.append(float(dfp.loc[m_zh, "proportion"].iloc[0]) if m_zh.any() else 0.0)

    return ns_props, zh_props


def draw_lithology_parity(
    dfp: pd.DataFrame,
    counts_mat: np.ndarray,
    *,
    col: str,
    outpath: Path,
    sharey: bool,
) -> None:
    utils.ensure_script_dirs()
    utils.set_paper_style()

    chi2, pval, dof, cv = chisq_cramers_v(counts_mat)

    fig = plt.figure(figsize=(7.0, 4.0))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.2, 0.8],
        wspace=0.35,
    )

    axA = fig.add_subplot(gs[0, 0])
    classes = list(dfp["class"].cat.categories)
    y = np.arange(len(classes))

    ns_props, zh_props = _extract_props(dfp, classes)

    h = 0.35
    axA.barh(
        y - h / 2.0,
        ns_props,
        height=h,
        color=CITY_COLORS["Nansha"],
        label="Nansha",
    )
    axA.barh(
        y + h / 2.0,
        zh_props,
        height=h,
        color=CITY_COLORS["Zhongshan"],
        label="Zhongshan",
    )
    axA.set_yticks(y)
    axA.set_yticklabels(classes)
    axA.set_xlabel("Proportion (normalized)")
    axA.set_xlim(0.0, 1.0)
    axA.legend(frameon=False, loc="lower right")

    col_lbl = utils.label(col, with_unit=False)
    axA.set_title(f"{col_lbl} — composition")

    axB = fig.add_subplot(gs[0, 1], sharey=axA if sharey else None)

    diff = np.asarray(zh_props) - np.asarray(ns_props)
    axB.axvline(0.0, color="#444444", lw=0.8)
    axB.barh(y, diff, height=0.6, color="#888888")
    axB.set_yticks(y)

    if sharey:
        axB.tick_params(axis="y", left=False, labelleft=False)
    else:
        axB.set_yticklabels(classes)

    axB.set_xlabel("Δ proportion (Zhongshan − Nansha)")
    axB.set_title("Parity difference")

    stat = (
        f"χ²={chi2:.2f}, dof={dof}, "
        f"p≈{pval:.3f}, V={cv:.3f}"
    )
    fig.suptitle(
        f"Lithology parity across cities • {stat}",
        x=0.02,
        y=0.99,
        ha="left",
    )

    base = utils.resolve_fig_out(str(outpath))
    if base.suffix:
        base = base.with_suffix("")

    fig.savefig(str(base) + ".png", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {base}.png/.svg")


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="plot-litho-parity",
        description="Supplementary Fig. S1.",
    )
    ap.add_argument(
        "--src",
        type=str,
        default=str(
            Path(
                r"F:\repositories\fusionlab-learn"
                r"\data"
            )
        ),
        help="Final dataset directory.",
    )
    ap.add_argument(
        "--ns-file",
        type=str,
        default="nansha_dataset.final.ready.csv",
    )
    ap.add_argument(
        "--zh-file",
        type=str,
        default="zhongshan_dataset.final.ready.csv",
    )

    utils.add_city_flags(ap, default_both=True)

    ap.add_argument(
        "--col",
        "-c",
        type=str,
        default="lithology_class",
    )
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
        "--top-n",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--group-others",
        type=str,
        default="true",
    )
    ap.add_argument(
        "--sharey",
        type=str,
        choices=["true", "false"],
        default="true",
    )
    ap.add_argument(
        "--out",
        "-o",
        type=str,
        default="figS1_lithology_parity",
    )
    return ap


def figS1_lithology_parity_main(argv: List[str] | None = None) -> None:

    ap = _build_argparser()
    args = ap.parse_args(argv)

    cities = utils.resolve_cities(args)
    if set(cities) != {"Nansha", "Zhongshan"}:
        raise ValueError(
            "Fig S1 needs both cities. Use --cities ns,zh "
            "or pass -ns -zh."
        )

    src = Path(args.src).expanduser()
    group_others = utils.str_to_bool(args.group_others)
    sharey = utils.str_to_bool(args.sharey)

    ns = load_city_df(
        src,
        args.ns_file,
        year=args.year,
        sample_frac=None if args.sample_n else args.sample_frac,
        sample_n=args.sample_n,
    )
    zh = load_city_df(
        src,
        args.zh_file,
        year=args.year,
        sample_frac=None if args.sample_n else args.sample_frac,
        sample_n=args.sample_n,
    )

    if args.col not in ns.columns or args.col not in zh.columns:
        raise KeyError(
            f"Column '{args.col}' must exist in both CSVs."
        )

    dfp, _, mat = compute_proportions(
        ns,
        zh,
        col=args.col,
        top_n=args.top_n,
        group_others=group_others,
    )

    draw_lithology_parity(
        dfp,
        mat,
        col=args.col,
        outpath=Path(args.out),
        sharey=sharey,
    )


def main(argv: List[str] | None = None) -> None:
    figS1_lithology_parity_main(argv)


if __name__ == "__main__":
    main()

    
# python -m scripts.scripts \
#   --src "F:\...\final_dataset" \
#   -ns -zh \
#   --col lithology_class \
#   --year all \
#   --out figS1_lithology_parity

# 4) How you run things (from project root)
# With dispatcher
# python -m scripts plot-driver-response --src data
# python -m scripts plot-core-ablation --ns-with ...
# python -m scripts plot-litho-parity --src data -ns -zh

# Or run a single module directly
# python -m scripts.plot_driver_response --src data

# 5) Optional: real shell commands (recommended)

# If you want true commands like plot-driver-response without
# python -m, add to your pyproject.toml:

# [project.scripts]
# plot-driver-response = "scripts.plot_driver_response:main"
# plot-core-ablation = "scripts.plot_core_ablation:main"
# plot-litho-parity = "scripts.plot_litho_parity:main"
# paper-scripts = "scripts.__main__:main"


# Then:

# pip install -e .
# plot-driver-response --src data
# paper-scripts plot-core-ablation --ns-with ...

# Recommendation for planning

# Keep one file = one figure/script (plot_*.py).

# Put all shared CLI flags and style into utils.py
# (add_plot_text_args, add_city_flags, output resolvers).

# Keep config.py as the single truth for units, column aliases,
# and PATTERNS.

# Keep __main__.py as a simple dispatcher only.

# If you want, paste your current scripts/utils.py +
# scripts/config.py (the updated ones) and I’ll make sure the three
# scripts import/use the same helpers in exactly the same way (and keep
# every line ≤62 chars).