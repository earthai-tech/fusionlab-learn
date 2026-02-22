# scripts/plot_sm3_log_offsets.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config as cfg
from . import utils


_EPS = 1e-12


def _load_meta(path: Path) -> Dict[str, Any]:
    meta_p = Path(str(path) + ".meta.json")
    if not meta_p.exists():
        return {}
    try:
        return json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_payload(path: Path) -> Dict[str, np.ndarray]:
    suf = path.suffix.lower()

    if suf == ".npz":
        with np.load(str(path), allow_pickle=True) as z:
            return {k: np.asarray(z[k]) for k in z.files}

    if suf == ".csv":
        df = pd.read_csv(str(path))
        return {c: df[c].to_numpy() for c in df.columns}

    if suf == ".parquet":
        df = pd.read_parquet(str(path))
        return {c: df[c].to_numpy() for c in df.columns}

    raise ValueError(f"Unsupported payload: {path}")


def _pick(
    payload: Dict[str, np.ndarray],
    *keys: str,
) -> Optional[np.ndarray]:
    for k in keys:
        if k in payload and payload[k] is not None:
            return np.asarray(payload[k])
    return None


def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _safe_log10(x: np.ndarray, eps: float) -> np.ndarray:
    xx = np.clip(np.asarray(x, float), eps, None)
    return np.log10(xx)


def _ensure_same_n(
    cols: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    n0: Optional[int] = None
    out: Dict[str, np.ndarray] = {}

    for k, v in cols.items():
        vv = _as_1d(np.asarray(v))
        if n0 is None:
            n0 = int(vv.size)
        elif int(vv.size) != int(n0):
            raise ValueError(
                f"Inconsistent size for {k}: "
                f"{vv.size} vs {n0}"
            )
        out[k] = vv

    return out


def _prior_series(
    *,
    n: int,
    payload: Dict[str, np.ndarray],
    payload_keys: Tuple[str, ...],
    scalar: Optional[float],
) -> Optional[np.ndarray]:
    arr = _pick(payload, *payload_keys)
    if arr is not None:
        a1 = _as_1d(arr)
        if a1.size != n:
            raise ValueError(
                "Prior array size mismatch: "
                f"{a1.size} vs {n}"
            )
        return a1

    if scalar is None:
        return None

    return np.full((n,), float(scalar), dtype=float)


def build_offsets_table(
    payload: Dict[str, np.ndarray],
    *,
    K_prior: Optional[float],
    Ss_prior: Optional[float],
    Hd_prior: Optional[float],
    eps: float = _EPS,
) -> pd.DataFrame:
    tau = _pick(payload, "tau", "tau_eff")
    tp = _pick(payload, "tau_prior", "tau_closure", "tau_cl")
    K = _pick(payload, "K", "K_eff", "K_field")
    Ss = _pick(payload, "Ss", "Ss_eff", "Ss_field")
    Hd = _pick(payload, "Hd", "H_d", "H", "H_field")

    if tau is None or tp is None:
        raise KeyError(
            "payload must contain tau and tau_prior/tau_closure"
        )
    if K is None or Ss is None or Hd is None:
        raise KeyError("payload must contain K, Ss, Hd (or H)")

    tau = _as_1d(tau)
    tp = _as_1d(tp)
    K = _as_1d(K)
    Ss = _as_1d(Ss)
    Hd = _as_1d(Hd)

    n = int(tau.size)
    if tp.size != n or K.size != n or Ss.size != n or Hd.size != n:
        raise ValueError("payload arrays must have same length")

    log10_tau = _safe_log10(tau, eps)
    log10_tp = _safe_log10(tp, eps)

    cols: Dict[str, np.ndarray] = {
        "log10_tau": log10_tau,
        "log10_tau_prior": log10_tp,
        "delta_log_tau": log10_tau - log10_tp,
        "log10_K": _safe_log10(K, eps),
        "log10_Ss": _safe_log10(Ss, eps),
        "log10_Hd": _safe_log10(Hd, eps),
    }

    Kp = _prior_series(
        n=n,
        payload=payload,
        payload_keys=("K_prior", "K_lith_prior", "k_prior"),
        scalar=K_prior,
    )
    Ssp = _prior_series(
        n=n,
        payload=payload,
        payload_keys=("Ss_prior", "Ss_lith_prior", "Ss0"),
        scalar=Ss_prior,
    )
    Hdp = _prior_series(
        n=n,
        payload=payload,
        payload_keys=("Hd_prior", "H_d_prior", "Hd_lith_prior"),
        scalar=Hd_prior,
    )

    if Kp is not None:
        cols["delta_logK"] = _safe_log10(K, eps) - _safe_log10(Kp, eps)
    if Ssp is not None:
        cols["delta_logSs"] = _safe_log10(Ss, eps) - _safe_log10(Ssp, eps)
    if Hdp is not None:
        cols["delta_logHd"] = _safe_log10(Hd, eps) - _safe_log10(Hdp, eps)

    cols = _ensure_same_n(cols)
    cols["index"] = np.arange(n, dtype=int)

    return pd.DataFrame(cols)


def summarise_offsets(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("delta_")]
    if not cols:
        raise RuntimeError("No delta_* columns to summarise.")

    desc = df[cols].describe(percentiles=[0.05, 0.5, 0.95]).T
    desc = desc.rename(
        columns={
            "mean": "mean",
            "std": "std",
            "5%": "p05",
            "50%": "p50",
            "95%": "p95",
        }
    )
    keep = ["mean", "std", "p05", "p50", "p95"]
    desc = desc[keep]
    desc.index.name = "metric"
    return desc


def _beautify(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.6)


def plot_offsets(
    df: pd.DataFrame,
    *,
    out_base: str,
    dpi: int,
    show_title: bool,
    title: Optional[str],
    bins: int,
) -> List[str]:
    utils.ensure_script_dirs()
    utils.set_paper_style()

    out_paths: List[str] = []

    base = utils.resolve_fig_out(out_base)
    if base.suffix:
        base = base.with_suffix("")

    delta_cols = [
        "delta_logK",
        "delta_logSs",
        "delta_logHd",
        "delta_log_tau",
    ]

    # --------------------------
    # Figure 1: 2x2 hist grid
    # --------------------------
    fig = plt.figure(figsize=(7.2, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    for i, col in enumerate(delta_cols):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        _beautify(ax)

        if col in df.columns:
            x = df[col].to_numpy(float)
            x = x[np.isfinite(x)]
            ax.hist(x, bins=int(bins))
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        else:
            ax.text(
                0.5,
                0.5,
                f"{col} not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    if show_title:
        ttl = utils.resolve_title(
            default="SM3 — log-offset diagnostics",
            title=title,
        )
        fig.suptitle(ttl, x=0.02, ha="left")

    p1 = str(base) + "-hists"
    fig.savefig(p1 + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(p1 + ".svg", bbox_inches="tight")
    plt.close(fig)
    out_paths += [p1 + ".png", p1 + ".svg"]

    # --------------------------
    # Figure 2: tau scatter
    # --------------------------
    if {"log10_tau_prior", "delta_log_tau"}.issubset(df.columns):
        fig2 = plt.figure(figsize=(3.6, 3.0), constrained_layout=True)
        ax2 = fig2.add_subplot(1, 1, 1)
        _beautify(ax2)

        ax2.scatter(
            df["log10_tau_prior"],
            df["delta_log_tau"],
            s=6,
            alpha=0.35,
            rasterized=True,
        )
        ax2.axhline(0.0, linestyle="--", linewidth=0.9)
        ax2.set_xlabel(r"$\log_{10}\tau_{\mathrm{prior}}$")
        ax2.set_ylabel(r"$\delta_\tau$")

        p2 = str(base) + "-tau-scatter"
        fig2.savefig(p2 + ".png", dpi=dpi, bbox_inches="tight")
        fig2.savefig(p2 + ".svg", bbox_inches="tight")
        plt.close(fig2)
        out_paths += [p2 + ".png", p2 + ".svg"]

    return out_paths


def _resolve_payload_path(
    *,
    src: Optional[str],
    payload: Optional[str],
) -> Path:
    if payload:
        return Path(payload).expanduser()

    if not src:
        raise ValueError("Provide --payload or --src")

    p = utils.find_latest(src, cfg.PATTERNS["physics_payload"])
    if p is None:
        raise FileNotFoundError(
            f"No physics payload under: {src}"
        )
    return p


def plot_sm3_log_offsets_main(
    argv: Optional[List[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        prog="plot-sm3-log-offsets",
        description="SM3 log-offset diagnostics (v3.2 payload).",
    )

    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--payload", type=str, default=None)

    ap.add_argument("--K-prior", type=float, default=None)
    ap.add_argument("--Ss-prior", type=float, default=None)
    ap.add_argument("--Hd-prior", type=float, default=None)

    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--dpi", type=int, default=cfg.PAPER_DPI)

    ap.add_argument(
        "--out-raw-csv",
        type=str,
        default="sm3-offsets-raw.csv",
    )
    ap.add_argument(
        "--out-summary-csv",
        type=str,
        default="sm3-offsets-summary.csv",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default="sm3-offsets.json",
    )

    utils.add_plot_text_args(
        ap,
        default_out="sm3-log-offsets",
    )

    args = ap.parse_args(argv)

    payload_p = _resolve_payload_path(
        src=args.src,
        payload=args.payload,
    )

    payload = _load_payload(payload_p)
    meta = _load_meta(payload_p)

    df = build_offsets_table(
        payload,
        K_prior=args.K_prior,
        Ss_prior=args.Ss_prior,
        Hd_prior=args.Hd_prior,
    )
    summ = summarise_offsets(df)

    out_raw = utils.resolve_out_out(args.out_raw_csv)
    out_sum = utils.resolve_out_out(args.out_summary_csv)
    out_js = utils.resolve_out_out(args.out_json)

    df.to_csv(out_raw, index=False)
    summ.to_csv(out_sum)

    show_title = utils.str_to_bool(args.show_title, default=True)

    fig_paths = plot_offsets(
        df,
        out_base=args.out,
        dpi=int(args.dpi),
        show_title=show_title,
        title=args.title,
        bins=int(args.bins),
    )

    payload_js = {
        "payload": str(payload_p.resolve()),
        "meta": meta,
        "n": int(len(df)),
        "columns": list(df.columns),
        "summary": json.loads(
            summ.reset_index().to_json(orient="records")
        ),
        "figures": fig_paths,
        "tables": {
            "raw_csv": str(out_raw),
            "summary_csv": str(out_sum),
        },
    }
    out_js.write_text(
        json.dumps(payload_js, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] payload: {payload_p}")
    print(f"[OK] wrote {out_raw}")
    print(f"[OK] wrote {out_sum}")
    print(f"[OK] wrote {out_js}")
    for p in fig_paths:
        print(f"[OK] wrote {p}")


def main(argv: Optional[List[str]] = None) -> None:
    plot_sm3_log_offsets_main(argv)


if __name__ == "__main__":
    main()

# Auto-discover payload under a run folder
# python -m scripts plot-sm3-log-offsets \
#   --src results/sm3_synth_1d \
#   --out sm3-log-offsets

# Explicit payload file
# python -m scripts plot-sm3-log-offsets \
#   --payload results/.../physics_payload_run_val.npz \
#   --out sm3-log-offsets

# If your payload does not embed priors, pass scalar priors
# python -m scripts plot-sm3-log-offsets \
#   --payload results/.../physics_payload_run_val.npz \
#   --K-prior 1e-7 --Ss-prior 1e-5 --Hd-prior 40 \
#   --out sm3-log-offsets