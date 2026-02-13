# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# scripts/utils.py
#
# Utilities to:
# - enforce paper style (matplotlib rcParams)
# - auto-detect artifacts from a "src" (file or folder)
# - load JSON/CSV robustly (schema checks)
# - harmonize GeoPrior JSON units (raw vs interpretable variants)

from __future__ import annotations

import json
# import os
from dataclasses import dataclass
from pathlib import Path
from typing import ( 
    List, Any, Dict, 
    Iterable, Optional,
    Sequence, 
    Tuple
)

import numpy as np
import pandas as pd

from . import config as cfg


_TRUE = {"1", "true", "yes", "y", "t", "on"}
_FALSE = {"0", "false", "no", "n", "f", "off"}


def str_to_bool(x: object, *, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return default


def add_city_flags(ap, *, default_both: bool = True) -> None:
    ap.add_argument(
        "-ns",
        "--ns",
        "--nansha",
        dest="use_ns",
        action="store_true",
        help="Select Nansha.",
    )
    ap.add_argument(
        "-zh",
        "--zh",
        "--zhongshan",
        dest="use_zh",
        action="store_true",
        help="Select Zhongshan.",
    )
    ap.add_argument(
        "--cities",
        type=str,
        default="ns,zh" if default_both else "",
        help="Comma list: ns,zh (default ns,zh).",
    )


def add_plot_text_args(
    ap,
    *,
    default_out: str,
) -> None:
    """
    Common plot args for Nature workflows.

    Use cases:
    - Hide text for Illustrator editing.
    - Keep titles/labels in SVG as editable text.

    Conventions:
    - "legend" includes colorbar (if present).
    - "labels" means axis labels and cbar label.
    - ticklabels are controlled separately.
    """
    ap.add_argument(
        "--out",
        "-o",
        type=str,
        default=default_out,
        help="Output stem/path (scripts/figs/ if rel).",
    )
    ap.add_argument(
        "--show-legend",
        type=str,
        default="true",
        help="Show legend/colorbar (true/false).",
    )
    ap.add_argument(
        "--show-labels",
        type=str,
        default="true",
        help="Show axis labels (true/false).",
    )
    ap.add_argument(
        "--show-ticklabels",
        type=str,
        default="true",
        help="Show tick labels (true/false).",
    )
    ap.add_argument(
        "--show-title",
        type=str,
        default="true",
        help="Show suptitle (true/false).",
    )
    ap.add_argument(
        "--show-panel-titles",
        type=str,
        default="true",
        help="Show per-row panel titles (true/false).",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override suptitle text.",
    )
    
def find_all(
    src: Any,
    patterns: Sequence[str],
    *,
    must_exist: bool = False,
) -> List[Path]:
    """
    Find all files matching any pattern under src.

    - If src is a file:
        return [src] if it matches, else [].
    - If src is a directory:
        search recursively (rglob) for all patterns.

    Returned list is sorted by mtime (newest first),
    with duplicates removed.
    """
    p = as_path(src)

    if p.is_file():
        name = p.name
        ok = any(_glob_match(name, pat) for pat in patterns)
        if ok:
            return [p]
        if must_exist:
            raise FileNotFoundError(str(p))
        return []

    if not p.exists():
        if must_exist:
            raise FileNotFoundError(str(p))
        return []

    out: Dict[str, Path] = {}
    for pat in patterns:
        for fp in p.rglob(pat):
            if fp.exists():
                out[str(fp.resolve())] = fp

    files = list(out.values())
    files.sort(
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return files


def resolve_title(
    *,
    default: str,
    title: Optional[str],
) -> str:
    if title is None:
        return default
    t = str(title).strip()
    return default if not t else t

def resolve_cities(args) -> List[str]:
    picked: List[str] = []
    if getattr(args, "use_ns", False):
        picked.append("Nansha")
    if getattr(args, "use_zh", False):
        picked.append("Zhongshan")

    if picked:
        return picked

    raw = str(getattr(args, "cities", "") or "")
    parts = [p.strip().lower() for p in raw.split(",")]
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        if p in {"ns", "nansha"}:
            out.append("Nansha")
        elif p in {"zh", "zhongshan"}:
            out.append("Zhongshan")
        else:
            out.append(p.title())
    return out

def resolve_fig_out(out: str) -> Path:
    """
    If out is relative, write under scripts/figs/.
    If out has no suffix, we treat it as a stem.
    """
    p = Path(out).expanduser()
    if not p.is_absolute():
        p = cfg.FIG_DIR / p
    return p

def ensure_columns(
    df: pd.DataFrame,
    *,
    aliases: Dict[str, Tuple[str, ...]],
) -> Dict[str, str]:
    """
    Ensure canonical columns exist by copying from
    the first available alias.

    Returns:
      mapping canonical -> source column used
    """
    used: Dict[str, str] = {}

    for canon, alts in (aliases or {}).items():
        if canon in df.columns:
            used[canon] = canon
            continue

        found = None
        for a in alts:
            if a in df.columns:
                found = a
                break

        if found is not None:
            df[canon] = df[found]
            used[canon] = found

    return used


def load_dataset_any(
    src: Path,
    *,
    file: Optional[str] = None,
    ns_file: str = "nansha_dataset.final.ready.csv",
    zh_file: str = "zhongshan_dataset.final.ready.csv",
) -> pd.DataFrame:
    """
    Load a combined dataset if:
      - src is a CSV file, or
      - src is a dir and --file is provided.

    Otherwise load ns_file + zh_file from src dir and
    concatenate.
    """
    src = Path(src).expanduser()

    if src.is_file():
        return pd.read_csv(src)

    if file:
        fp = (src / file).expanduser()
        if not fp.exists():
            raise FileNotFoundError(str(fp))
        return pd.read_csv(fp)

    ns_fp = (src / ns_file).expanduser()
    zh_fp = (src / zh_file).expanduser()

    if not ns_fp.exists():
        raise FileNotFoundError(str(ns_fp))
    if not zh_fp.exists():
        raise FileNotFoundError(str(zh_fp))

    ns = pd.read_csv(ns_fp)
    zh = pd.read_csv(zh_fp)

    if "city" not in ns.columns:
        ns["city"] = "Nansha"
    if "city" not in zh.columns:
        zh["city"] = "Zhongshan"

    return pd.concat([ns, zh], ignore_index=True)


def filter_year(
    df: pd.DataFrame,
    year: str,
) -> pd.DataFrame:
    if year == "all":
        return df
    y = int(year)
    if "year" not in df.columns:
        return df
    return df.loc[df["year"] == y].copy()


def sample_df(
    df: pd.DataFrame,
    *,
    sample_frac: Optional[float],
    sample_n: Optional[int],
    seed: int = 42,
) -> pd.DataFrame:
    if sample_n is not None:
        n = min(int(sample_n), len(df))
        return df.sample(n=n, random_state=seed)

    if sample_frac is not None:
        f = float(sample_frac)
        f = max(0.0, min(1.0, f))
        if f < 1.0:
            return df.sample(frac=f, random_state=seed)

    return df

# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_script_dirs() -> None:
    ensure_dir(cfg.OUT_DIR)
    ensure_dir(cfg.FIG_DIR)


def as_path(src: Any) -> Path:
    if isinstance(src, Path):
        return src
    return Path(str(src)).expanduser()


def canonical_city(name: str) -> str:
    if not name:
        return name
    k = str(name).strip().lower()
    return cfg.CITY_CANON.get(k, name)


def label(key: str, *, with_unit: bool = True) -> str:
    base = cfg.LABELS.get(key, key.replace("_", " ").title())
    if not with_unit:
        return base
    u = cfg.UNITS.get(key)
    return f"{base} ({u})" if u else base


def phys_label(key: str, *, with_unit: bool = True) -> str:
    base = cfg.PHYS_LABELS.get(key, key)
    if not with_unit:
        return base
    u = cfg.PHYS_UNITS.get(key)
    return f"{base} ({u})" if u else base


# -------------------------------------------------------------------
# Matplotlib style (centralized)
# Replaces repeated set_style() across figure scripts.
# -------------------------------------------------------------------
def set_paper_style(
    *,
    fontsize: int = cfg.PAPER_FONT,
    dpi: int = cfg.PAPER_DPI,
) -> None:
    import matplotlib as mpl

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


# -------------------------------------------------------------------
# Robust file discovery from a "src"
# -------------------------------------------------------------------
def _iter_candidates(
    root: Path,
    patterns: Sequence[str],
) -> Iterable[Path]:
    for pat in patterns:
        yield from root.rglob(pat)

def find_preferred(
    src: Any,
    patterns: Sequence[str],
    *,
    must_exist: bool = False,
) -> Optional[Path]:
    """
    Try patterns in order, returning the first match.

    Unlike find_latest() with multiple patterns,
    this respects priority order.
    """
    root = as_path(src)
    if root.is_file():
        root = root.parent

    for pat in patterns:
        p = find_latest(root, [pat], must_exist=False)
        if p is not None:
            return p

    if must_exist:
        raise FileNotFoundError(
            f"No match under {root} for {patterns}"
        )

    return None


def find_eval_diag_json(src: Any) -> Optional[Path]:
    pats = cfg.PATTERNS.get("eval_diag_json", ())
    if not pats:
        return None
    return find_preferred(src, pats)

def find_latest(
    src: Any,
    patterns: Sequence[str],
    *,
    must_exist: bool = False,
) -> Optional[Path]:
    """
    Find newest file matching any of patterns under src.

    - If src is a file: returns it if it matches any pattern.
    - If src is a directory: searches recursively and returns
      the most recently modified candidate.
    """
    p = as_path(src)

    if p.is_file():
        name = p.name
        ok = any(_glob_match(name, pat) for pat in patterns)
        if ok:
            return p
        if must_exist:
            raise FileNotFoundError(str(p))
        return None

    if not p.exists():
        if must_exist:
            raise FileNotFoundError(str(p))
        return None

    cands = list(_iter_candidates(p, patterns))
    if not cands:
        if must_exist:
            raise FileNotFoundError(
                f"No match under {p} for {patterns}"
            )
        return None

    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def _glob_match(name: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(name, pattern)


@dataclass
class Artifacts:
    src: Path
    phys_json: Optional[Path] = None
    eval_diag_json: Optional[Path] = None
    forecast_test_csv: Optional[Path] = None
    forecast_test_future_csv: Optional[Path] = None
    forecast_val_csv: Optional[Path] = None
    forecast_future_csv: Optional[Path] = None
    physics_payload: Optional[Path] = None
    coords_npz: Optional[Path] = None

def detect_artifacts(src: Any) -> Artifacts:
    """
    Detect common v3.2 artifacts starting from a src path.

    Example:
    - src = ".../results/nansha_..._stage1/train_YYYYmmdd-HHMMSS"
    - We auto-locate:
      geoprior_eval_phys_*.json, eval_diagnostics.json,
      *_calibrated.csv, *_future.csv, physics payload, etc.
    """
    root = as_path(src)
    if root.is_file():
        root = root.parent

    out = Artifacts(src=root)

    out.phys_json = find_latest(root, cfg.PATTERNS["phys_json"])
    out.eval_diag_json = find_eval_diag_json(root)

    out.forecast_val_csv = find_latest(
        root,
        cfg.PATTERNS["forecast_val_csv"],
    )
    out.forecast_future_csv = find_latest(
        root,
        cfg.PATTERNS["forecast_future_csv"],
    )
    out.physics_payload = find_latest(
        root,
        cfg.PATTERNS["physics_payload"],
    )
    out.forecast_test_csv = find_latest(
        root,
        cfg.PATTERNS["forecast_test_csv"],
    )
    out.forecast_test_future_csv = find_latest(
        root,
        cfg.PATTERNS["forecast_test_future_csv"],
    )
    out.coords_npz = find_latest(root, cfg.PATTERNS["coords_npz"])
    return out


# -------------------------------------------------------------------
# Loading helpers (JSON / CSV)
# -------------------------------------------------------------------
def safe_load_json(path: Optional[Any]) -> Dict[str, Any]:
    if not path:
        return {}
    p = as_path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def load_forecast_csv(path: Any) -> pd.DataFrame:
    """
    Load calibrated forecast CSV and enforce basic schema.

    Expected columns are consistent with the uncertainty scripts:
      sample_idx, forecast_step,
      subsidence_q10, subsidence_q50, subsidence_q90,
      subsidence_actual
    """
    p = as_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_csv(p)

    needed = {
        "sample_idx",
        "forecast_step",
        "subsidence_q10",
        "subsidence_q50",
        "subsidence_q90",
        "subsidence_actual",
    }
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {p}: {missing}")

    for c in [
        "subsidence_q10",
        "subsidence_q50",
        "subsidence_q90",
        "subsidence_actual",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["forecast_step"] = pd.to_numeric(
        df["forecast_step"],
        errors="coerce",
    )
    df = df.dropna(subset=["forecast_step"]).copy()
    return df


# -------------------------------------------------------------------
# GeoPrior JSON harmonization
# - v3.2 may produce both:
#   * raw JSON (subs_metrics_unit="m")
#   * interpretable JSON (subs_metrics_unit="mm")
# We standardize to "mm" for plotting/tables.
# -------------------------------------------------------------------
def phys_json_to_mm(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not meta:
        return {}

    out = dict(meta)
    units = dict((meta.get("units") or {}))

    u = str(units.get("subs_metrics_unit", "")).lower()
    if u == "mm":
        return out

    # Raw JSON commonly stores subs metrics in meters.
    # Convert a known set of keys to mm where applicable.
    scale = 1000.0

    def _scale_key(d: Dict[str, Any], key: str) -> None:
        v = d.get(key, None)
        if isinstance(v, (int, float)):
            d[key] = float(v) * scale

    me = dict((out.get("metrics_evaluate") or {}))
    for k in list(me.keys()):
        if k.startswith("subs_pred_"):
            # mae/mse/rmse/sharpness live here.
            # mse should scale by 1e6, but we only scale
            # *distance-like* keys safely:
            if any(x in k for x in ["mae", "rmse", "sharp"]):
                _scale_key(me, k)
            if "mse" in k:
                v = me.get(k, None)
                if isinstance(v, (int, float)):
                    me[k] = float(v) * (scale**2)

    out["metrics_evaluate"] = me

    # Per-horizon width metrics
    ph = dict((out.get("per_horizon") or {}))
    shp = ph.get("sharpness80") or ph.get("sharpness")
    if isinstance(shp, dict):
        ph2 = dict(ph)
        shp2 = dict(shp)
        for hk, hv in shp2.items():
            if isinstance(hv, (int, float)):
                shp2[hk] = float(hv) * scale
        if "sharpness80" in ph2:
            ph2["sharpness80"] = shp2
        else:
            ph2["sharpness"] = shp2
        out["per_horizon"] = ph2

    units["subs_metrics_unit"] = "mm"
    out["units"] = units
    return out


# -------------------------------------------------------------------
# Metric picking (GeoPrior JSON = primary; eval_diag = fallback)
# Mirrors your Figure-2 collection logic.
# -------------------------------------------------------------------
def pick_point_metrics(
    phys_json: Dict[str, Any],
    fallback: Dict[str, Any],
) -> Tuple[float, float, float]:
    r2 = mae = mse = np.nan

    if phys_json:
        pm = phys_json.get("point_metrics", {}) or {}
        if pm:
            r2 = pm.get("r2", r2)
            mae = pm.get("mae", mae)
            mse = pm.get("mse", mse)

        me = phys_json.get("metrics_evaluate", {}) or {}
        if np.isnan(mae):
            mae = me.get("subs_pred_mae", mae)
        if np.isnan(mse):
            mse = me.get("subs_pred_mse", mse)

    flat = flatten_eval_diag(fallback) if fallback else {}
    if np.isnan(r2):
        r2 = flat.get("r2", r2)
    if np.isnan(mae):
        mae = flat.get("mae", mae)
    if np.isnan(mse):
        mse = flat.get("mse", mse)

    return (to_float(r2), to_float(mae), to_float(mse))


def pick_interval_metrics(
    phys_json: Dict[str, Any],
    fallback: Dict[str, Any],
) -> Tuple[float, float]:
    cov = shp = np.nan

    if phys_json:
        im = phys_json.get("interval_metrics", {}) or {}
        if im:
            cov = im.get("coverage80", cov)
            shp = im.get("sharpness80", shp)

        me = phys_json.get("metrics_evaluate", {}) or {}
        if np.isnan(cov):
            cov = me.get("subs_pred_coverage80", cov)
        if np.isnan(shp):
            shp = me.get("subs_pred_sharpness80", shp)

    flat = flatten_eval_diag(fallback) if fallback else {}
    if np.isnan(cov):
        cov = flat.get("coverage80", cov)
    if np.isnan(shp):
        shp = flat.get("sharpness80", shp)

    return (to_float(cov), to_float(shp))


# def flatten_eval_diag(diag: Dict[str, Any]) -> Dict[str, float]:
#     """
#     eval_diagnostics.json is not always a flat schema.
#     We flatten only what we need for paper figures/tables.
#     """
#     if not diag:
#         return {}

#     out: Dict[str, float] = {}

#     for k in ["r2", "mae", "mse", "rmse"]:
#         if k in diag:
#             out[k] = to_float(diag.get(k))

#     # Some versions store uncertainty metrics nested
#     for k in ["coverage80", "sharpness80"]:
#         if k in diag:
#             out[k] = to_float(diag.get(k))

#     return out

def flatten_eval_diag(diag: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten eval diagnostics into keys used by plots.

    Supports:
    - legacy flat schema: {"r2":..., "mae":...}
    - calibration schema:
        {"eval_after": {"coverage":..., "sharpness":...}}
    """
    if not diag:
        return {}

    out: Dict[str, float] = {}

    for k in ["r2", "mae", "mse", "rmse"]:
        if k in diag:
            out[k] = to_float(diag.get(k))

    for k in ["coverage80", "sharpness80"]:
        if k in diag:
            out[k] = to_float(diag.get(k))

    overall_key = str(diag.get("overall_key") or "")
    eval_after = diag.get("eval_after") or {}
    if isinstance(eval_after, dict):
        if overall_key and overall_key in eval_after:
            blk = eval_after.get(overall_key) or {}
            if isinstance(blk, dict):
                eval_after = blk

        cov = eval_after.get("coverage", None)
        shp = eval_after.get("sharpness", None)

        if "coverage80" not in out:
            if cov is not None:
                out["coverage80"] = to_float(cov)

        if "sharpness80" not in out:
            if shp is not None:
                out["sharpness80"] = to_float(shp)

    return out


def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")
    
def resolve_out_out(out: str) -> Path:
    """
    If out is relative, write under scripts/out/.
    """
    p = Path(out).expanduser()
    if not p.is_absolute():
        p = cfg.OUT_DIR / p
    return p


def find_phys_json(src: Any) -> Optional[Path]:
    """
    Prefer interpretable GeoPrior JSON when available.
    """
    pats = cfg.PATTERNS.get("phys_json", ())
    if not pats:
        return None

    root = as_path(src)
    if root.is_file():
        root = root.parent

    p0 = find_latest(root, [pats[0]])
    if p0 is not None:
        return p0

    if len(pats) > 1:
        return find_latest(root, [pats[1]])

    return None
