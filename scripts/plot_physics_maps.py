# scripts/plot_physics_maps.py
# -*- coding: utf-8 -*-

"""plot_physics_maps

Figure — Spatial physics fields and "physics tension".

This script renders a 2×3 panel for a single city:

1) log10(K)      — hydraulic conductivity
2) log10(Ss)     — specific storage
3) Hd            — effective drainage thickness / head depth
4) log10(tau)    — learned relaxation time scale
5) log10(tau_p)  — prior / closure time scale
6) Δlog10(tau)   — log10(tau) − log10(tau_p)

Inputs
------
Either:

* --payload PATH         (explicit payload file)
* --src DIR_OR_FILE      (auto-detect physics payload
                          under src)

Optional:

* --coords-npz PATH      (if payload lacks x/y or lon/lat)

The payload is expected to come from
``GeoPriorSubsNet.export_physics_payload``.

Outputs
-------
Two files are written:

* <out>.png
* <out>.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from . import config as cfg
from . import utils


# ------------------------------------------------------------
# Payload loading
# ------------------------------------------------------------
def _load_payload(path: str) -> Tuple[dict, dict]:
    """Load (payload, meta) using v3.2 conventions."""
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

    ext = p.suffix.lower()
    if ext == ".npz":
        with np.load(str(p)) as z:
            payload = {k: z[k] for k in z.files}
    elif ext in {".csv", ".parquet"}:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(
                "CSV/Parquet payloads need pandas/pyarrow."
            ) from exc
        df = (
            pd.read_csv(p)
            if ext == ".csv"
            else pd.read_parquet(p)
        )
        payload = {c: df[c].to_numpy() for c in df.columns}
    else:
        raise ValueError(
            "Unsupported payload extension. "
            "Use .npz/.csv/.parquet."
        )

    meta: dict = {}
    mp = str(p) + ".meta.json"
    if Path(mp).exists():
        try:
            meta = json.loads(Path(mp).read_text()) or {}
        except Exception:
            meta = {}

    # v3.2 aliasing: tau_prior <-> tau_closure
    if (
        "tau_prior" not in payload
        and "tau_closure" in payload
    ):
        payload["tau_prior"] = payload["tau_closure"]
    if (
        "tau_closure" not in payload
        and "tau_prior" in payload
    ):
        payload["tau_closure"] = payload["tau_prior"]

    if (
        "log10_tau_prior" not in payload
        and "log10_tau_closure" in payload
    ):
        payload["log10_tau_prior"] = payload[
            "log10_tau_closure"
        ]
    if (
        "log10_tau_closure" not in payload
        and "log10_tau_prior" in payload
    ):
        payload["log10_tau_closure"] = payload[
            "log10_tau_prior"
        ]

    return payload, meta


def _pick(payload: dict, *keys: str):
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return None


def _to_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


# ------------------------------------------------------------
# Coordinates
# ------------------------------------------------------------
def _infer_city_from_path(p: Path) -> str:
    s = str(p).lower()
    for k, v in cfg.CITY_CANON.items():
        if k in s:
            return v
    return p.stem.replace("_", " ").title()


def _resolve_payload_path(
    *,
    src: Optional[str],
    payload: Optional[str],
) -> Path:
    if payload:
        return Path(payload).expanduser()

    if not src:
        raise ValueError("Provide --payload or --src")

    p = utils.as_path(src)
    if p.is_file():
        return p

    arts = utils.detect_artifacts(p)
    if arts.physics_payload is None:
        pats = cfg.PATTERNS["physics_payload"]
        raise FileNotFoundError(
            "No physics payload found under: "
            f"{p}\nPatterns: {pats}"
        )
    return arts.physics_payload


def _resolve_coords_npz(
    *,
    src: Optional[str],
    coords_npz: Optional[str],
) -> Optional[Path]:
    if coords_npz:
        return Path(coords_npz).expanduser()
    if not src:
        return None
    arts = utils.detect_artifacts(src)
    return arts.coords_npz


def _extract_xy(
    payload: dict,
    *,
    coords: Optional[dict],
    x_key: Optional[str],
    y_key: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, str]:
    candidates = [
        ("x", "y", "xy"),
        ("lon", "lat", "lonlat"),
        ("longitude", "latitude", "lonlat"),
    ]

    for kx, ky, kind in candidates:
        if kx in payload and ky in payload:
            x = _to_1d(payload[kx])
            y = _to_1d(payload[ky])
            return x, y, kind

    if coords is None:
        raise KeyError(
            "No coords in payload or coords-npz."
        )

    if "coords" in coords:
        arr = np.asarray(coords["coords"])
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                "coords['coords'] must be (N,H,3)."
            )
        x = arr[..., 1].reshape(-1)
        y = arr[..., 2].reshape(-1)
        return _to_1d(x), _to_1d(y), "xy"

    if (
        x_key
        and y_key
        and x_key in coords
        and y_key in coords
    ):
        return (
            _to_1d(coords[x_key]),
            _to_1d(coords[y_key]),
            "xy",
        )

    for kx, ky, kind in candidates:
        if kx in coords and ky in coords:
            return (
                _to_1d(coords[kx]),
                _to_1d(coords[ky]),
                kind,
            )

    raise KeyError(
        "Could not find coordinate arrays in coords-npz."
    )


def _maybe_trim(
    a: np.ndarray,
    b: np.ndarray,
    *rest: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, ...]]:
    n = min([a.size, b.size] + [r.size for r in rest])
    if a.size != n:
        a = a[:n]
    if b.size != n:
        b = b[:n]
    out = []
    for r in rest:
        out.append(r[:n] if r.size != n else r)
    return a, b, tuple(out)


# ------------------------------------------------------------
# Gridding
# ------------------------------------------------------------
def _bin_grid(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    *,
    nx: int,
    ny: int,
    agg: str,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    x = _to_1d(x)
    y = _to_1d(y)
    v = _to_1d(v)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if int(m.sum()) == 0:
        raise ValueError("No finite points to grid.")

    x = x[m]
    y = y[m]
    v = v[m]

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    pad_x = 1e-6 * max(abs(xmin), abs(xmax), 1.0)
    pad_y = 1e-6 * max(abs(ymin), abs(ymax), 1.0)

    x_edges = np.linspace(xmin - pad_x, xmax + pad_x, nx + 1)
    y_edges = np.linspace(ymin - pad_y, ymax + pad_y, ny + 1)

    if agg == "median":
        try:
            from scipy.stats import binned_statistic_2d

            stat = binned_statistic_2d(
                y,
                x,
                v,
                statistic="median",
                bins=[y_edges, x_edges],
            ).statistic
            grid = np.asarray(stat, float)
        except Exception:
            agg = "mean"

    if agg == "mean":
        count, _, _ = np.histogram2d(
            y,
            x,
            bins=[y_edges, x_edges],
        )
        s, _, _ = np.histogram2d(
            y,
            x,
            bins=[y_edges, x_edges],
            weights=v,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            grid = s / count
        grid[count == 0] = np.nan

    extent = (
        float(x_edges[0]),
        float(x_edges[-1]),
        float(y_edges[0]),
        float(y_edges[-1]),
    )
    return grid, extent


def _range_q(
    z: np.ndarray,
    qlo: float,
    qhi: float,
) -> Tuple[float, float]:
    v = np.asarray(z, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(v, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        m = float(np.nanmedian(v))
        return m - 1.0, m + 1.0
    return float(lo), float(hi)


def _sym_range_q(
    z: np.ndarray,
    q: float,
) -> Tuple[float, float]:
    v = np.asarray(z, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    a = float(np.nanpercentile(np.abs(v), q))
    a = max(a, 1e-12)
    return -a, a


# ------------------------------------------------------------
# Labels / units
# ------------------------------------------------------------
def _meta_unit(meta: dict, key: str) -> Optional[str]:
    u = (meta or {}).get("units", {}) or {}
    out = u.get(key)
    return str(out) if out else None


def _tau_prior_symbol(meta: dict) -> str:
    name = str((meta or {}).get("tau_prior_human_name", ""))
    if "closure" in name.lower():
        return r"\tau_{\mathrm{closure}}"
    return r"\tau_{\mathrm{prior}}"


def _cbar_label(
    name: str,
    unit: Optional[str],
    *,
    show: bool,
) -> str:
    if not show:
        return ""
    if unit:
        return f"{name} ({unit})"
    return name


_HUMAN = {
    "log10_K": "Hydraulic conductivity",
    "log10_Ss": "Specific storage",
    "Hd": "Effective head depth",
    "log10_tau": "Relaxation time scale",
    "log10_tau_p": "Closure time scale",
    "delta_log10_tau": "Physics tension",
}


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def _axes_cleanup(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.02,
        0.98,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def _overlay_censored(
    ax: plt.Axes,
    mask: np.ndarray,
    *,
    extent: Tuple[float, float, float, float],
) -> None:
    m = np.asarray(mask) > 0.5
    if not np.any(m):
        return

    xmin, xmax, ymin, ymax = extent

    ny, nx = m.shape
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    ax.contourf(
        X,
        Y,
        m.astype(int),
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
        linewidths=0.0,
    )


def plot_physics_maps(
    payload: dict,
    meta: dict,
    *,
    x: np.ndarray,
    y: np.ndarray,
    city: str,
    out: str,
    agg: str,
    grid: int,
    clip_q: Tuple[float, float],
    delta_q: float,
    cmap: str,
    cmap_div: str,
    coord_kind: str,
    dpi: int,
    font: int,
    show_legend: bool,
    show_labels: bool,
    show_ticklabels: bool,
    show_title: bool,
    show_panel_titles: bool,
    show_panel_labels: bool,
    hatch_censored: bool,
    title: Optional[str],
    out_json: Optional[str],
) -> List[str]:
    """Render and save the physics maps panel."""
    utils.ensure_script_dirs()
    utils.set_paper_style(dpi=dpi, fontsize=font)

    tau = _pick(payload, "tau")
    tp = _pick(payload, "tau_prior", "tau_closure")
    K = _pick(payload, "K", "K_field")
    Ss = _pick(payload, "Ss", "Ss_field")
    Hd = _pick(payload, "Hd", "H")

    miss = []
    for name, val in [
        ("tau", tau),
        ("tau_prior", tp),
        ("K", K),
        ("Ss", Ss),
        ("Hd", Hd),
    ]:
        if val is None:
            miss.append(name)
    if miss:
        raise KeyError(f"Missing in payload: {miss}")

    x, y, rest = _maybe_trim(
        _to_1d(x),
        _to_1d(y),
        _to_1d(K),
        _to_1d(Ss),
        _to_1d(Hd),
        _to_1d(tau),
        _to_1d(tp),
    )
    K, Ss, Hd, tau, tp = rest

    eps = 1e-12
    logK = np.log10(np.clip(K, eps, None))
    logSs = np.log10(np.clip(Ss, eps, None))

    logtau = _pick(payload, "log10_tau")
    if logtau is None:
        logtau = np.log10(np.clip(tau, eps, None))
    else:
        logtau = _to_1d(logtau)

    logtp = _pick(
        payload,
        "log10_tau_prior",
        "log10_tau_closure",
    )
    if logtp is None:
        logtp = np.log10(np.clip(tp, eps, None))
    else:
        logtp = _to_1d(logtp)

    delta = logtau - logtp

    cens = _pick(
        payload,
        "censored",
        "soil_thickness_censored",
    )
    if cens is not None:
        cens = _to_1d(cens)

    if coord_kind not in {"auto", "xy", "lonlat"}:
        raise ValueError("coord_kind must be auto/xy/lonlat")
    if coord_kind == "auto":
        x_ok = np.nanmax(np.abs(x)) <= 180.0
        y_ok = np.nanmax(np.abs(y)) <= 90.0
        coord_kind = "lonlat" if (x_ok and y_ok) else "xy"

    aspect = "equal" if coord_kind == "xy" else "auto"

    nx = int(grid)
    ny = int(grid)

    zK, ext = _bin_grid(x, y, logK, nx=nx, ny=ny, agg=agg)
    zSs, _ = _bin_grid(x, y, logSs, nx=nx, ny=ny, agg=agg)
    zHd, _ = _bin_grid(x, y, Hd, nx=nx, ny=ny, agg=agg)
    zt, _ = _bin_grid(x, y, logtau, nx=nx, ny=ny, agg=agg)
    ztp, _ = _bin_grid(x, y, logtp, nx=nx, ny=ny, agg=agg)
    zD, _ = _bin_grid(x, y, delta, nx=nx, ny=ny, agg=agg)

    zC = None
    if cens is not None:
        zC, _ = _bin_grid(
            x,
            y,
            cens,
            nx=nx,
            ny=ny,
            agg="mean",
        )

    qlo, qhi = clip_q
    k_lo, k_hi = _range_q(zK, qlo, qhi)
    ss_lo, ss_hi = _range_q(zSs, qlo, qhi)
    hd_lo, hd_hi = _range_q(zHd, qlo, qhi)
    t_lo, t_hi = _range_q(zt, qlo, qhi)
    tp_lo, tp_hi = _range_q(ztp, qlo, qhi)
    d_lo, d_hi = _sym_range_q(zD, delta_q)

    tau_sym = _tau_prior_symbol(meta)
    top = [
        ("log10_K", r"$\\log_{10} K$", zK, cmap, k_lo, k_hi),
        (
            "log10_Ss",
            r"$\\log_{10} S_s$",
            zSs,
            cmap,
            ss_lo,
            ss_hi,
        ),
        ("Hd", r"$H_d$", zHd, cmap, hd_lo, hd_hi),
    ]
    bot = [
        (
            "log10_tau",
            r"$\\log_{10} \\tau$",
            zt,
            cmap,
            t_lo,
            t_hi,
        ),
        (
            "log10_tau_p",
            rf"$\\log_{{10}} {tau_sym}$",
            ztp,
            cmap,
            tp_lo,
            tp_hi,
        ),
        (
            "delta_log10_tau",
            r"$\\Delta \\log_{10} \\tau$",
            zD,
            cmap_div,
            d_lo,
            d_hi,
        ),
    ]

    fig = plt.figure(figsize=(7.0, 4.6))
    gs = GridSpec(2, 3, figure=fig, wspace=0.34, hspace=0.28)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    letters = ["a", "b", "c", "d", "e", "f"]
    out_paths: List[str] = []

    items = top + bot
    ims = []
    for i, (key, ttl, z, cm, lo, hi) in enumerate(items):
        ax = axes[i]
        _axes_cleanup(ax)

        im = ax.imshow(
            z,
            origin="lower",
            extent=ext,
            cmap=cm,
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
            aspect=aspect,
        )
        ims.append(im)

        if hatch_censored and zC is not None:
            _overlay_censored(ax, zC, extent=ext)

        if show_panel_titles:
            ax.set_title(ttl)
        else:
            ax.set_title("")

        if show_panel_labels:
            _panel_label(ax, letters[i])

        if show_ticklabels:
            ax.set_xticks(np.linspace(ext[0], ext[1], 3))
            ax.set_yticks(np.linspace(ext[2], ext[3], 3))
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if show_labels:
            xlab = "X"
            ylab = "Y"
            if coord_kind == "lonlat":
                xlab = "Longitude"
                ylab = "Latitude"

            if i in {3, 4, 5}:
                ax.set_xlabel(xlab)
            if i in {0, 3}:
                ax.set_ylabel(ylab)

    if show_legend:
        for i, (key, _ttl, _z, _cm, _lo, _hi) in enumerate(
            items
        ):
            ax = axes[i]
            unit = _meta_unit(meta, key)
            if unit is None:
                if key in {"log10_tau_p", "delta_log10_tau"}:
                    unit = cfg.PHYS_UNITS.get("log10_tau")
                else:
                    unit = cfg.PHYS_UNITS.get(key)

            name = _HUMAN.get(key, key)
            cb = fig.colorbar(
                ims[i],
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )
            cb.set_label(
                _cbar_label(name, unit, show=show_labels)
            )

    base = utils.resolve_fig_out(out)
    if base.suffix:
        base = base.with_suffix("")

    city_name = utils.canonical_city(city)
    col = cfg.CITY_COLORS.get(city_name, "#111111")

    if show_title:
        ttl = utils.resolve_title(
            default=(
                f"{city_name} — physics fields and tension"
            ),
            title=title,
        )
        fig.suptitle(ttl, x=0.02, ha="left", color=col)

    fig.savefig(str(base) + ".png", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    plt.close(fig)

    out_paths += [str(base) + ".png", str(base) + ".svg"]

    if out_json:
        jout = utils.resolve_out_out(out_json)
        rec: Dict[str, object] = {
            "city": city_name,
            "payload_keys": sorted(list(payload.keys())),
            "units": (meta or {}).get("units", {}),
            "ranges": {
                "log10_K": [k_lo, k_hi],
                "log10_Ss": [ss_lo, ss_hi],
                "Hd": [hd_lo, hd_hi],
                "log10_tau": [t_lo, t_hi],
                "log10_tau_p": [tp_lo, tp_hi],
                "delta_log10_tau": [d_lo, d_hi],
            },
            "figures": out_paths,
        }
        jout.write_text(
            json.dumps(rec, indent=2),
            encoding="utf-8",
        )

    return out_paths


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def plot_physics_maps_main(
    argv: Optional[List[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        prog="plot-physics-maps",
        description=(
            "Spatial maps of GeoPrior physics fields and "
            "Δlog10(τ) tension."
        ),
    )

    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--payload", type=str, default=None)
    ap.add_argument("--city", type=str, default=None)

    ap.add_argument("--coords-npz", type=str, default=None)
    ap.add_argument("--x-key", type=str, default=None)
    ap.add_argument("--y-key", type=str, default=None)

    ap.add_argument(
        "--coord-kind",
        type=str,
        default="auto",
        choices=["auto", "xy", "lonlat"],
    )

    ap.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
    )
    ap.add_argument("--grid", type=int, default=200)

    ap.add_argument(
        "--clip-q",
        type=float,
        nargs=2,
        default=(2.0, 98.0),
        metavar=("QLOW", "QHIGH"),
    )
    ap.add_argument(
        "--delta-q",
        type=float,
        default=98.0,
        help="Symmetric abs percentile for Δlog10(τ).",
    )
    ap.add_argument("--cmap", type=str, default="viridis")
    ap.add_argument("--cmap-div", type=str, default="RdBu_r")

    ap.add_argument(
        "--hatch-censored",
        type=str,
        default="true",
    )
    ap.add_argument(
        "--show-panel-labels",
        type=str,
        default="true",
        help="Show a–f letters (true/false).",
    )

    ap.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)
    ap.add_argument(
        "--font",
        type=int,
        default=cfg.PAPER_FONT,
    )

    utils.add_plot_text_args(
        ap,
        default_out="fig_physics_maps",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Write ranges JSON to scripts/out/.",
    )

    args = ap.parse_args(argv)

    payload_p = _resolve_payload_path(
        src=args.src,
        payload=args.payload,
    )
    payload, meta = _load_payload(str(payload_p))

    coords_p = _resolve_coords_npz(
        src=args.src,
        coords_npz=args.coords_npz,
    )
    coords = None
    if coords_p is not None and coords_p.exists():
        with np.load(str(coords_p)) as z:
            coords = {k: z[k] for k in z.files}

    x, y, kind = _extract_xy(
        payload,
        coords=coords,
        x_key=args.x_key,
        y_key=args.y_key,
    )

    city = utils.canonical_city(
        args.city or _infer_city_from_path(payload_p)
    )

    show_legend = utils.str_to_bool(
        args.show_legend,
        default=True,
    )
    show_labels = utils.str_to_bool(
        args.show_labels,
        default=True,
    )
    show_ticks = utils.str_to_bool(
        args.show_ticklabels,
        default=True,
    )
    show_title = utils.str_to_bool(
        args.show_title,
        default=True,
    )
    show_pt = utils.str_to_bool(
        args.show_panel_titles,
        default=True,
    )
    show_pl = utils.str_to_bool(
        args.show_panel_labels,
        default=True,
    )
    hatch = utils.str_to_bool(
        args.hatch_censored,
        default=True,
    )

    out_paths = plot_physics_maps(
        payload,
        meta,
        x=x,
        y=y,
        city=city,
        out=str(args.out),
        agg=str(args.agg),
        grid=int(args.grid),
        clip_q=(float(args.clip_q[0]), float(args.clip_q[1])),
        delta_q=float(args.delta_q),
        cmap=str(args.cmap),
        cmap_div=str(args.cmap_div),
        coord_kind=str(args.coord_kind or kind),
        dpi=int(args.dpi),
        font=int(args.font),
        show_legend=show_legend,
        show_labels=show_labels,
        show_ticklabels=show_ticks,
        show_title=show_title,
        show_panel_titles=show_pt,
        show_panel_labels=show_pl,
        hatch_censored=hatch,
        title=args.title,
        out_json=args.out_json,
    )

    print(f"[OK] payload: {payload_p}")
    if coords_p is not None:
        print(f"[OK] coords: {coords_p}")
    for p in out_paths:
        print(f"[OK] wrote {p}")


def main(argv: Optional[List[str]] = None) -> None:
    plot_physics_maps_main(argv)


if __name__ == "__main__":
    main()
