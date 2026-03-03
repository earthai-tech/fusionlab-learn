# scripts/plot_physics_sanity.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec

from . import config as cfg
from . import utils


# ================================================================
# 1) Payload loading (v3.2)
#    - Prefer fusionlab loader if available
#    - Fallback to raw NPZ + optional sidecar meta JSON
# ================================================================

def _load_payload(path: str) -> Tuple[dict, dict]:
    try:
        from fusionlab.nn.pinn.io import (  # type: ignore
            load_physics_payload as _lp,
        )

        payload, meta = _lp(path)
        return payload, (meta or {})
    except Exception:
        pass

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Missing payload: {p}")

    if p.suffix.lower() != ".npz":
        raise ValueError(
            "Fallback loader expects .npz. "
            f"Got: {p.name!r}"
        )

    with np.load(str(p)) as z:
        payload = {k: z[k] for k in z.files}

    meta: dict = {}
    mp = str(p) + ".meta.json"
    if Path(mp).exists():
        try:
            meta = json.loads(Path(mp).read_text()) or {}
        except Exception:
            meta = {}

    # v3.2 aliasing: tau_prior <-> tau_closure
    if "tau_prior" not in payload and "tau_closure" in payload:
        payload["tau_prior"] = payload["tau_closure"]
    if "tau_closure" not in payload and "tau_prior" in payload:
        payload["tau_closure"] = payload["tau_prior"]

    if (
        "log10_tau_prior" not in payload
        and "log10_tau_closure" in payload
    ):
        payload["log10_tau_prior"] = payload["log10_tau_closure"]
    if (
        "log10_tau_closure" not in payload
        and "log10_tau_prior" in payload
    ):
        payload["log10_tau_closure"] = payload["log10_tau_prior"]

    return payload, meta


def _pick(payload: dict, *keys: str):
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return None


def _to_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        if a.shape != arrs[0].shape:
            raise ValueError("Arrays must share same shape.")
        m &= np.isfinite(a)
    return m

def _apply_paper_axis_format(
    ax,
    *,
    axis: str = "both",
    scilimits: Tuple[int, int] = (-2, 2),
    use_offset: bool = True,
) -> None:
    """
    Force scientific notation with mathtext (×10^k) on axes.

    - This replaces '1e-5' with '×10^{-5}'.
    - With scilimits=(-2,2), values like 2e-4 show as scientific.
    """
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits(tuple(int(x) for x in scilimits))
    fmt.set_scientific(True)
    fmt.set_useOffset(bool(use_offset))

    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)

    # Make the offset text (×10^k) look consistent and not huge
    try:
        ax.xaxis.get_offset_text().set_size(9)
        ax.yaxis.get_offset_text().set_size(9)
    except Exception:
        pass
    
def _subsample_idx(
    n: int,
    frac: Optional[float],
    max_n: Optional[int],
    seed: int,
) -> np.ndarray:
    idx = np.arange(n)

    if frac is not None:
        f = float(frac)
        if not (0.0 < f <= 1.0):
            raise ValueError("--subsample-frac in (0,1].")
        k = int(np.ceil(f * n))
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(idx, size=k, replace=False)
        idx.sort()

    if max_n is not None and idx.size > int(max_n):
        idx = idx[: int(max_n)]

    return idx


# ================================================================
# 2) Labels / units (use config + meta; avoid hardcoding)
# ================================================================
def _meta_unit(meta: dict, key: str) -> Optional[str]:
    units = (meta or {}).get("units", {}) or {}
    u = units.get(key)
    if u:
        return str(u)
    return None


def _tau_prior_symbol(meta: dict) -> str:
    name = str((meta or {}).get("tau_prior_human_name", ""))
    if "closure" in name.lower():
        return r"\tau_{\mathrm{closure}}"
    return r"\tau_{\mathrm{prior}}"


# def _tau_prior_formula(meta: dict) -> str:
#     f = (meta or {}).get("tau_closure_formula", None)
#     if f:
#         if "kappa_bar" in str(f):
#             return r"\kappa\,H^2S_s/(\pi^2K)"
        
#         return r"H_d^2S_s/(\pi^2\kappa K)"

#     # Single source (config)
#     return cfg.CLOSURES.get("tau_prior", r"\tau_{\mathrm{prior}}")

def _strip_math(s: str) -> str:
    # Remove outer $...$ if present (config strings often include $)
    s = str(s).strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s[1:-1]
    return s

def _sci_tex(x: float, sig: int = 3) -> str:
    """
    Return mantissa×10^{exp} as a LaTeX snippet (no surrounding $).
    """
    if x is None:
        return r"\mathrm{NA}"
    x = float(x)
    if not np.isfinite(x):
        return r"\mathrm{nan}"
    if x == 0.0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10.0**exp)
    mant_s = f"{mant:.{sig}g}"
    return rf"{mant_s}\times 10^{{{exp}}}"


def _strip_dollars(s: str) -> str:
    s = str(s).strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s[1:-1]
    return s


def _unit_to_tex(unit: Optional[str]) -> str:
    if not unit:
        return r"\mathrm{m\,s^{-1}}"
    u = str(unit).strip().lower().replace(" ", "")
    if u in ("m/s", "ms-1", "m*s^-1", "m*s-1", "m·s-1"):
        return r"\mathrm{m\,s^{-1}}"
    if u in ("mm",):
        return r"\mathrm{mm}"
    if u in ("m",):
        return r"\mathrm{m}"
    # fallback: keep as plain text in roman
    return rf"\mathrm{{{unit}}}"


def _axis_offset_inner(ax, axis: str) -> str:
    """
    Return offset text without $...$, e.g. '\\times 10^{-11}'.
    Empty string if none.
    """
    if axis == "x":
        t = ax.xaxis.get_offset_text().get_text()
    else:
        t = ax.yaxis.get_offset_text().get_text()
    t = str(t).strip()
    if not t:
        return ""
    return _strip_dollars(t)


def _embed_offsets_into_labels(
    ax,
    *,
    x_base: str,
    x_unit_tex: Optional[str],
    y_base: Optional[str] = None,
) -> None:
    """
    Hide axis offset texts and embed them into labels.
    """
    ox = _axis_offset_inner(ax, "x")
    oy = _axis_offset_inner(ax, "y")

    # X label
    if ox:
        ut = _unit_to_tex(x_unit_tex)
        ax.set_xlabel(rf"${x_base}\,({ox}\ {ut})$")
        ax.xaxis.get_offset_text().set_visible(False)

    # Y label
    if y_base and oy:
        ax.set_ylabel(rf"${y_base}\,({oy})$")
        ax.yaxis.get_offset_text().set_visible(False)
        
def _tau_prior_formula(meta: dict) -> str:
    """
    Return a LaTeX math expression (NO surrounding $) describing
    the tau closure used in titles/captions.
    """
    # 1) Meta override (if present)
    f = (meta or {}).get("tau_closure_formula", None)
    if f:
        sf = str(f).strip()
        lo = sf.lower()

        # Special variant signaled in meta
        if "kappa_bar" in lo or r"\bar{\kappa}" in sf:
            return (
                r"\frac{H_d^2\,S_s}"
                r"{\pi^2\,\bar{\kappa}\,K}"
            )

        # If meta already contains LaTeX, pass-through (strip $ if any)
        if "\\" in sf or r"\frac" in sf:
            return _strip_math(sf)

        # If meta provides a plain-text-like closure, map to a nice fraction
        # (best-effort for your known patterns)
        if "hd" in lo and "ss" in lo and "kappa" in lo and "pi" in lo:
            # decide whether it's κ_b or κ
            ksym = r"\kappa_b" if ("kappa_b" in lo or "κ_b" in sf) else r"\kappa"
            return (
                r"\frac{H_d^2\,S_s}"
                rf"{{\pi^2\,{ksym}\,K}}"
            )

        # last resort: show it as-is (but make it math-safe)
        return _strip_math(sf)

    # 2) Single source from config
    # cfg.CLOSURES["tau_prior"] includes full "$...$" with "tau_prior ≈ .../..."
    # We extract the RHS and format it as a fraction.
    cfg_expr = _strip_math(cfg.CLOSURES.get("tau_prior", r"\tau_{\mathrm{prior}}"))

    # If config already contains a fraction, just return it
    if r"\frac" in cfg_expr:
        return cfg_expr

    # Your canonical config RHS: H_d^2 S_s / (pi^2 kappa_b K)
    # Return a clean fraction RHS only.
    return (
        r"\frac{H_d^2\,S_s}"
        r"{\pi^2\,\kappa_b\,K}"
    )

    # return r"\frac{\kappa\,H^2\,S_s}{\pi^2\,K}"
# ================================================================
# 3) Stats
# ================================================================
def _r2_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")

    x = x[m]
    y = y[m]
    xm = x - x.mean()
    ym = y - y.mean()

    denom = (xm**2).sum() * (ym**2).sum()
    if denom <= 0:
        return float("nan")

    return float(((xm * ym).sum() ** 2) / denom)


def _eps_prior_ln(tau: np.ndarray, tp: np.ndarray) -> float:
    tau = np.asarray(tau, float).reshape(-1)
    tp = np.asarray(tp, float).reshape(-1)
    m = np.isfinite(tau) & np.isfinite(tp) & (tau > 0) & (tp > 0)
    if m.sum() == 0:
        return float("nan")
    dif = np.log(tau[m]) - np.log(tp[m])
    return float(np.sqrt(np.mean(dif**2)))


def _eps_rms(x: np.ndarray) -> float:
    v = np.asarray(x, float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(v * v)))


# ================================================================
# 4) Panels (no city logic here; purely plotting)
# ================================================================
def _panel_label(ax, letter: str, enabled: bool) -> None:
    if not enabled:
        return
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _hexbin_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    gridsize: int,
    clip_q: Tuple[float, float],
    xlabel: str,
    ylabel: str,
    draw_diag: bool,
) -> object:
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        raise ValueError("No finite points for hexbin.")

    qlo, qhi = float(clip_q[0]), float(clip_q[1])
    xlo, xhi = np.nanpercentile(x, qlo), np.nanpercentile(x, qhi)
    ylo, yhi = np.nanpercentile(y, qlo), np.nanpercentile(y, qhi)

    keep = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    if keep.sum() > 0:
        x = x[keep]
        y = y[keep]

    def _pad(lo, hi, ref):
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = np.nanmin(ref), np.nanmax(ref)
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        p = max(0.05 * span, 1e-12)
        return lo - p, hi + p

    xlo, xhi = _pad(xlo, xhi, x)
    ylo, yhi = _pad(ylo, yhi, y)

    hb = ax.hexbin(
        x,
        y,
        gridsize=int(gridsize),
        mincnt=1,
        bins="log",
        cmap="viridis",
        extent=(xlo, xhi, ylo, yhi),
        linewidths=0.0,
    )

    if draw_diag:
        d0 = min(xlo, ylo)
        d1 = max(xhi, yhi)
        ax.plot(
            [d0, d1],
            [d0, d1],
            linestyle="--",
            linewidth=0.8,
            color="#444444",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    return hb


def _hist_panel(
    ax,
    v: np.ndarray,
    *,
    bins: int,
    color: str,
    xlabel: str,
    show_ylabel: bool,
) -> None:
    v = np.asarray(v, float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite residuals",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel(xlabel)
        if show_ylabel:
            ax.set_ylabel("Density")
        return

    ax.hist(
        v,
        bins=int(bins),
        density=True,
        histtype="step",
        color=color,
    )
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("Density")
    else:
        ax.tick_params(labelleft=False)


# ================================================================
# 5) Case preparation (generic: no Nansha/Zhongshan hardcoding)
#    - src can be a file (NPZ) or a directory (auto-discovery)
#    - city can be explicitly given or inferred from path name
# ================================================================
def _infer_city_from_path(p: Path) -> str:
    s = str(p).lower()
    for k, v in cfg.CITY_CANON.items():
        if k in s:
            return v
    return p.stem.replace("_", " ").title()


def _resolve_payload_path(src: Any) -> Path:
    p = utils.as_path(src)

    if p.is_file():
        return p

    arts = utils.detect_artifacts(p)
    if arts.physics_payload is None:
        raise FileNotFoundError(
            "No physics payload found under: "
            f"{p}\nPatterns: {cfg.PATTERNS['physics_payload']}"
        )
    return arts.physics_payload


def _prepare_case(
    src: Any,
    *,
    city: Optional[str],
    cons_kind: str,
    subsample_frac: Optional[float],
    max_points: Optional[int],
    seed: int,
) -> dict:
    payload_path = _resolve_payload_path(src)
    payload, meta = _load_payload(str(payload_path))

    city_name = utils.canonical_city(
        city or _infer_city_from_path(payload_path)
    )

    # Required fields (v3.2)
    tau = _pick(payload, "tau")
    tp = _pick(payload, "tau_prior", "tau_closure")
    K = _pick(payload, "K", "K_field")
    Ss = _pick(payload, "Ss", "Ss_field")
    Hd = _pick(payload, "Hd", "H")

    if cons_kind == "scaled":
        cons = _pick(payload, "cons_res_scaled", "cons_res_vals")
        cons_key = (
            "cons_res_scaled"
            if "cons_res_scaled" in payload
            else "cons_res_vals"
        )
    else:
        cons = _pick(payload, "cons_res_vals")
        cons_key = "cons_res_vals"

    miss = []
    for name, val in [
        ("tau", tau),
        ("tau_prior/tau_closure", tp),
        ("K", K),
        ("Ss", Ss),
        ("Hd/H", Hd),
        (cons_key, cons),
    ]:
        if val is None:
            miss.append(name)
    if miss:
        raise KeyError(
            f"Missing keys in payload {payload_path.name}: {miss}"
        )

    tau = _to_1d(tau)
    tp = _to_1d(tp)
    K = _to_1d(K)
    Ss = _to_1d(Ss)
    Hd = _to_1d(Hd)
    cons = _to_1d(cons)

    m = _finite_mask(tau, tp, K, Ss, Hd, cons)
    tau, tp, K, Ss, Hd, cons = (
        tau[m],
        tp[m],
        K[m],
        Ss[m],
        Hd[m],
        cons[m],
    )

    idx = _subsample_idx(
        tau.size,
        subsample_frac,
        max_points,
        int(seed),
    )
    tau, tp, K, Ss, Hd, cons = (
        tau[idx],
        tp[idx],
        K[idx],
        Ss[idx],
        Hd[idx],
        cons[idx],
    )

    color = cfg.CITY_COLORS.get(city_name, "#444444")

    return {
        "city": city_name,
        "color": color,
        "path": payload_path,
        "meta": meta,
        "payload": payload,
        "tau": tau,
        "tau_prior": tp,
        "K": K,
        "Ss": Ss,
        "Hd": Hd,
        "cons": cons,
    }


# ================================================================
# 6) Figure assembly (generic loop over cases)
# ================================================================
def render_physics_sanity(
    cases: List[dict],
    *,
    outbase: Path,
    dpi: int,
    fontsize: int,
    gridsize: int,
    hist_bins: int,
    clip_q: Tuple[float, float],
    plot_mode: str,
    tau_scale: str,
    show_labels: bool,
    show_title: bool,
    show_panel_titles: bool,
    show_panel_labels: bool,
    show_ticklabels: bool,
    show_legend: bool,
    title: Optional[str],
    paper_format: bool, 
    paper_no_offset: bool
) -> Dict[str, dict]:
    utils.set_paper_style(dpi=dpi, fontsize=fontsize)

    plot_mode = str(plot_mode).lower()
    if plot_mode not in ("joint", "residual"):
        raise ValueError("--plot-mode joint|residual")

    tau_scale = str(tau_scale).lower()
    if tau_scale not in ("log10", "linear"):
        raise ValueError("--tau-scale log10|linear")

    if plot_mode == "residual":
        tau_scale = "log10"

    if len(cases) != 2:
        raise ValueError(
            "This Nature Figure layout expects exactly 2 cases."
        )

    meta0 = cases[0].get("meta", {}) or {}
    tau_sym = _tau_prior_symbol(meta0)
    tau_form = _tau_prior_formula(meta0)

    # x/y labels (common across cases)
    if plot_mode == "joint":
        if tau_scale == "log10":
            xlab = rf"$\log_{{10}}({tau_sym})$"
            ylab = r"$\log_{10}(\tau)$"
        else:
            xlab = rf"${tau_sym}$"
            ylab = r"$\tau$"
    else:
        xlab = rf"$\log_{{10}}({tau_sym})$"
        ylab = rf"$\log_{{10}}(\tau/{tau_sym})$"

    # Residual axis label: use meta unit if available
    cons_u = _meta_unit(meta0, "cons_res_vals")
    rlab = (
        rf"$R_{{\mathrm{{cons}}}}$ [{cons_u}]"
        if cons_u
        else r"$R_{\mathrm{cons}}$"
    )

    fig = plt.figure(figsize=(7.0, 4.4))
    gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.55)

    def _prep_xy(d: dict):
        tau = d["tau"]
        tp = d["tau_prior"]

        mp = np.isfinite(tau) & np.isfinite(tp) & (tau > 0) & (tp > 0)
        tau = tau[mp]
        tp = tp[mp]

        lt = np.log10(tau)
        lp = np.log10(tp)

        r2 = _r2_corr(lp, lt)
        eps_pr = _eps_prior_ln(tau, tp)

        delta = float(np.median(lt - lp))
        c = float(10.0**delta)

        if plot_mode == "joint":
            if tau_scale == "log10":
                return lp, lt, r2, eps_pr, delta, c
            return tp, tau, r2, eps_pr, delta, c

        return lp, (lt - lp), r2, eps_pr, delta, c

    letters = ["a", "b", "c", "d"]
    stats: Dict[str, dict] = {}
    hist_axes = []
    
    for i, case in enumerate(cases):
        x, y, r2, ep, dlt, cst = _prep_xy(case)

        axL = fig.add_subplot(gs[i, 0])
        axR = fig.add_subplot(gs[i, 1])
        hist_axes.append(axR)

        _panel_label(axL, letters[2 * i], show_panel_labels)
        _panel_label(axR, letters[2 * i + 1], show_panel_labels)

        hb = _hexbin_panel(
            axL,
            x,
            y,
            gridsize=gridsize,
            clip_q=clip_q,
            xlabel=xlab if show_labels else "",
            ylabel=ylab if show_labels else "",
            draw_diag=(plot_mode == "joint"),
        )

        if plot_mode == "joint":
            if tau_scale == "log10":
                axL.plot(
                    [axL.get_xlim()[0], axL.get_xlim()[1]],
                    [
                        axL.get_xlim()[0] + dlt,
                        axL.get_xlim()[1] + dlt,
                    ],
                    color="k",
                    linewidth=0.8,
                )
            else:
                xmin, xmax = axL.get_xlim()
                axL.plot(
                    [xmin, xmax],
                    [cst * xmin, cst * xmax],
                    color="k",
                    linewidth=0.8,
                )
        else:
            axL.axhline(
                0.0,
                linestyle="--",
                color="#444444",
                linewidth=0.8,
            )
            axL.axhline(dlt, color="k", linewidth=0.8)

        if not show_ticklabels:
            axL.tick_params(labelbottom=False, labelleft=False)

        if show_panel_titles:
            axL.set_title(case["city"])

        axL.text(
            0.98,
            0.03,
            fr"$R^2={r2:.3f}$"
            "\n"
            fr"$\varepsilon_{{prior}}={ep:.3f}$"
            "\n"
            fr"$\tau\approx {cst:.1e}\,{tau_sym}$",
            transform=axL.transAxes,
            va="bottom",
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.85,
                linewidth=0.0,
            ),
        )

        if show_legend:
            cb = fig.colorbar(hb, ax=axL, fraction=0.046, pad=0.04)
            cb.set_label(r"$\log_{10}(\mathrm{count})$")

        eps_cons = _eps_rms(case["cons"])
        _hist_panel(
            axR,
            case["cons"],
            bins=hist_bins,
            color=case["color"],
            xlabel=rlab if show_labels else "",
            show_ylabel=show_labels,
        )

        if show_panel_titles:
            if paper_format:
                eps_tex = _sci_tex(eps_cons, sig=3)
                axR.set_title(
                    fr"{case['city']}  "
                    fr"$\varepsilon_{{cons}}={eps_tex}$"
                )
            else:
                axR.set_title(
                    fr"{case['city']}  "
                    fr"$\varepsilon_{{cons}}={eps_cons:.3g}$"
                )

        if not show_ticklabels:
            axR.tick_params(labelbottom=False, labelleft=False)

        stats[case["city"]] = {
            "r2_logtau": r2,
            "eps_prior_rms": ep,
            "eps_cons_rms": eps_cons,
            "tau_ratio_median": float(10.0**dlt),
            "payload": str(case["path"]),
        }

        # --- Nature axis formatting (optional) ---
        if paper_format:
            # Left panels: residual axis can have tiny numbers (e.g., 2e-4)
            # Hist panels: x can be 1e-5 scale; y can be 1e5..1e6 scale
            _apply_paper_axis_format(axL, axis="y", scilimits=(-2, 2))
            _apply_paper_axis_format(axR, axis="both", scilimits=(-2, 2))
            
    if show_title:
        if title:
            st = str(title)
        else:
            if plot_mode == "joint":
                if tau_scale == "log10":
                    st = (
                        r"Physics sanity: "
                        rf"$\log_{{10}}(\tau)$ vs "
                        rf"$\log_{{10}}({tau_sym})$; "
                        r"$R_{\mathrm{cons}}$ distribution"
                    )
                else:
                    st = (
                        r"Physics sanity: "
                        rf"$\tau$ vs ${tau_sym}$; "
                        r"$R_{\mathrm{cons}}$ distribution"
                    )
            else:
                # residual mode
                st = (
                    r"Physics sanity: "
                    rf"$\log_{{10}}\!\left(\tau/{tau_sym}\right)$ "
                    r"vs "
                    rf"${tau_sym}={tau_form}$; "
                    r"$R_{\mathrm{cons}}$ distribution"
                )
    
        fig.suptitle(st, x=0.02, y=0.99, ha="left")

    if paper_format and paper_no_offset:
        # Need a draw so ScalarFormatter computes offset text
        fig.canvas.draw()
    
        unit_tex = cons_u or "m/s"   # what you already inferred
        for axR in hist_axes:
            # x: R_cons with units; y: Density
            _embed_offsets_into_labels(
                axR,
                x_base=r"R_{\mathrm{cons}}",
                x_unit_tex=unit_tex,
                y_base=r"\mathrm{Density}",
            )
        
    outbase.parent.mkdir(parents=True, exist_ok=True)
    utils.save_figure(fig, outbase, dpi=int(dpi))

    return stats


# ================================================================
# 7) Optional extra figures (generic loop over cases)
# ================================================================
def _extra_k_from_tau(
    cases: List[dict],
    *,
    outbase: Path,
    dpi: int,
    fontsize: int,
    show_labels: bool,
    show_title: bool,
) -> None:
    utils.set_paper_style(dpi=dpi, fontsize=fontsize)

    pairs: List[Tuple[np.ndarray, np.ndarray, str, str]] = []
    for c in cases:
        p = c["payload"]
        K = _pick(p, "K")
        Kt = _pick(p, "K_from_tau")
        if K is None or Kt is None:
            continue

        K = _to_1d(K)
        Kt = _to_1d(Kt)
        m = np.isfinite(K) & np.isfinite(Kt) & (K > 0) & (Kt > 0)
        pairs.append(
            (
                np.log10(K[m]),
                np.log10(Kt[m]),
                c["city"],
                c["color"],
            )
        )

    if len(pairs) != len(cases):
        print("[WARN] Missing K_from_tau; skipping extra.")
        return

    fig = plt.figure(figsize=(6.6, 3.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    for i, (x, y, name, col) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(x, y, s=2, alpha=0.35, color=col)

        d0 = min(ax.get_xlim()[0], ax.get_ylim()[0])
        d1 = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([d0, d1], [d0, d1], linestyle="--", color="#444444")

        if show_labels:
            ax.set_xlabel(r"$\log_{10}(K)$")
            ax.set_ylabel(r"$\log_{10}(K_{\mathrm{from}\ \tau})$")

        if show_title:
            ax.set_title(name)

    fig.savefig(str(outbase) + "-k-from-tau.png", bbox_inches="tight")
    fig.savefig(str(outbase) + "-k-from-tau.svg", bbox_inches="tight")
    plt.close(fig)


def _extra_closure_check(
    cases: List[dict],
    *,
    outbase: Path,
    dpi: int,
    fontsize: int,
    show_labels: bool,
    show_title: bool,
) -> None:
    utils.set_paper_style(dpi=dpi, fontsize=fontsize)

    pairs: List[Tuple[np.ndarray, np.ndarray, str, str]] = []
    for c in cases:
        p = c["payload"]
        tp = _pick(p, "tau_prior", "tau_closure")
        tc = _pick(p, "tau_closure_calc")
        if tp is None or tc is None:
            continue

        tp = _to_1d(tp)
        tc = _to_1d(tc)
        m = np.isfinite(tp) & np.isfinite(tc) & (tp > 0) & (tc > 0)
        pairs.append(
            (
                np.log10(tp[m]),
                np.log10(tc[m]),
                c["city"],
                c["color"],
            )
        )

    if len(pairs) != len(cases):
        print("[WARN] Missing tau_closure_calc; skipping extra.")
        return

    fig = plt.figure(figsize=(6.6, 3.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    for i, (x, y, name, col) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(x, y, s=2, alpha=0.35, color=col)

        d0 = min(ax.get_xlim()[0], ax.get_ylim()[0])
        d1 = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([d0, d1], [d0, d1], linestyle="--", color="#444444")

        if show_labels:
            ax.set_xlabel(r"$\log_{10}(\tau_{\mathrm{prior}})$")
            ax.set_ylabel(
                r"$\log_{10}(\tau_{\mathrm{closure\ calc}})$"
            )

        if show_title:
            ax.set_title(name)

    fig.savefig(str(outbase) + "-closure-check.png", bbox_inches="tight")
    fig.savefig(str(outbase) + "-closure-check.svg", bbox_inches="tight")
    plt.close(fig)


# ================================================================
# 8) CLI
# ================================================================
def _parse_csv_list(s: str) -> List[str]:
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p]


def _add_args(ap) -> None:
    ap.add_argument(
        "--src",
        action="append",
        required=True,
        help=(
            "Case source (file .npz or directory). "
            "Provide exactly two: --src A --src B"
        ),
    )
    ap.add_argument(
        "--city",
        action="append",
        default=[],
        help=(
            "Optional city name per --src. "
            "Repeat to match --src order."
        ),
    )

    ap.add_argument(
        "--plot-mode",
        type=str,
        default="joint",
        choices=["joint", "residual"],
    )
    ap.add_argument(
        "--tau-scale",
        type=str,
        default="log10",
        choices=["log10", "linear"],
        help="Only used for --plot-mode joint.",
    )
    ap.add_argument(
        "--cons-kind",
        type=str,
        default="raw",
        choices=["raw", "scaled"],
        help="raw=cons_res_vals; scaled=cons_res_scaled",
    )

    ap.add_argument("--gridsize", type=int, default=120)
    ap.add_argument("--hist-bins", type=int, default=40)

    ap.add_argument(
        "--clip-q",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("QLOW", "QHIGH"),
    )

    ap.add_argument("--subsample-frac", type=float, default=None)
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)
    ap.add_argument("--fontsize", type=int, default=cfg.PAPER_FONT)

    utils.add_plot_text_args(ap, default_out="fig4_physics_sanity")
    ap.add_argument(
        "--show-panel-labels",
        type=str,
        default="true",
        help="Show a/b/c/d panel labels (true/false).",
    )

    ap.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Write summary metrics JSON to scripts/out/.",
    )
    ap.add_argument(
        "--extra",
        type=str,
        default="none",
        help="Comma list: none,k-from-tau,closure",
    )
    
    ap.add_argument(
        "--paper-format",
        action="store_true",
        help=(
            "Paper-style axis formatting: use mathtext "
            "scientific notation (×10^k) instead of 1e±k, "
            "and force sci formatting on small/large axes."
        ),
    )

    ap.add_argument(
        "--paper-no-offset",
        action="store_true",
        help=(
            "If set, hide ×10^k offset text and embed the multiplier "
            "into axis labels (Nature-style). Requires --paper-format."
        ),
    )

def plot_physics_sanity_main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="plot-physics-sanity",
        description=(
            "Figure 4 — physics sanity (v3.2). "
            "No city hardcoding: loop over 2 cases."
        ),
    )
    _add_args(ap)
    args = ap.parse_args(argv)

    utils.ensure_script_dirs()

    show_labels = utils.str_to_bool(args.show_labels)
    show_title = utils.str_to_bool(args.show_title)
    show_panel_titles = utils.str_to_bool(args.show_panel_titles)
    show_panel_labels = utils.str_to_bool(args.show_panel_labels)
    show_ticklabels = utils.str_to_bool(args.show_ticklabels)
    show_legend = utils.str_to_bool(args.show_legend)
    
    paper_format=bool(args.paper_format),
    paper_no_offset=bool(args.paper_no_offset),

    srcs = list(args.src or [])
    if len(srcs) != 2:
        raise ValueError("Provide exactly two --src values.")

    cities = list(args.city or [])
    while len(cities) < len(srcs):
        cities.append(None)

    cases: List[dict] = []
    for s, c in zip(srcs, cities):
        cases.append(
            _prepare_case(
                s,
                city=c,
                cons_kind=args.cons_kind,
                subsample_frac=args.subsample_frac,
                max_points=args.max_points,
                seed=args.seed,
            )
        )

    base = utils.resolve_fig_out(args.out)
    if base.suffix:
        base = base.with_suffix("")

    stats = render_physics_sanity(
        cases,
        outbase=base,
        dpi=int(args.dpi),
        fontsize=int(args.fontsize),
        gridsize=int(args.gridsize),
        hist_bins=int(args.hist_bins),
        clip_q=tuple(args.clip_q),
        plot_mode=args.plot_mode,
        tau_scale=args.tau_scale,
        show_labels=show_labels,
        show_title=show_title,
        show_panel_titles=show_panel_titles,
        show_panel_labels=show_panel_labels,
        show_ticklabels=show_ticklabels,
        show_legend=show_legend,
        title=args.title,
        paper_format=paper_format, 
        paper_no_offset= paper_no_offset
    )

    extras = _parse_csv_list(args.extra)
    if "k-from-tau" in extras:
        _extra_k_from_tau(
            cases,
            outbase=base,
            dpi=int(args.dpi),
            fontsize=int(args.fontsize),
            show_labels=show_labels,
            show_title=show_panel_titles,
        )
    if "closure" in extras:
        _extra_closure_check(
            cases,
            outbase=base,
            dpi=int(args.dpi),
            fontsize=int(args.fontsize),
            show_labels=show_labels,
            show_title=show_panel_titles,
        )

    if args.out_json:
        jout = utils.resolve_out_out(args.out_json)
        if jout.suffix.lower() != ".json":
            jout = jout.with_suffix(".json")
        jout.parent.mkdir(parents=True, exist_ok=True)
        jout.write_text(json.dumps(stats, indent=2))
        print(f"[OK] wrote {jout}")

    print(f"[OK] wrote {base}.png/.svg")


def main(argv: Optional[List[str]] = None) -> None:
    plot_physics_sanity_main(argv)


if __name__ == "__main__":
    main()

# How you run it now (no hard-coded city args)

# If you already have the payload files:

# python -m scripts.plot_physics_sanity \
#   --src results/ns/physics_payload.npz \
#   --src results/zh/physics_payload.npz \
#   --out fig4_physics_sanity \
#   --out-json fig4_physics_sanity.json


# If you pass directories (auto-detects latest payload in each):

# python -m scripts.plot_physics_sanity \
#   --src results/ns_run_dir \
#   --src results/zh_run_dir \
#   --out fig4_physics_sanity \
#   --extra k-from-tau,closure


# If the city name isn’t inferable from path, override it:

# python -m scripts.plot_physics_sanity \
#   --src some/runA \
#   --src some/runB \
#   --city Nansha \
#   --city Zhongshan


