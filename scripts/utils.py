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
import matplotlib.pyplot as plt
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
    
def add_render_args(
    ap,
    *,
    default: str = "heatmap",
) -> None:
    """
    Common render args for 2D sensitivity plots.

    render:
      - heatmap: pivot->imshow (discrete grid)
      - tricontour: smooth contourf on scattered points
      - pcolormesh: grid-aware shading using real coords
    """
    ap.add_argument(
        "--render",
        type=str,
        default=default,
        choices=[
            "heatmap",
            "tricontour",
            "pcolormesh",
        ],
        help="Render style (heatmap/tricontour/pcolormesh).",
    )
    ap.add_argument(
        "--levels",
        type=int,
        default=14,
        help="Levels for tricontour render.",
    )
    ap.add_argument(
        "--clip",
        type=str,
        default="2,98",
        help="Color scale clip percentiles (lo,hi).",
    )
    ap.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation for duplicate (λc, λp).",
    )
    ap.add_argument(
        "--show-points",
        type=str,
        default="true",
        help="Overlay sampled points (true/false).",
    )

    ap.add_argument(
        "--trend-arrow",
        type=str,
        default="false",
        help="Overlay a trend arrow (true/false).",
    )
    ap.add_argument(
        "--trend-arrow-len",
        type=float,
        default=0.22,
        help="Arrow length in axes fraction.",
    )
    ap.add_argument(
        "--trend-arrow-pos",
        type=str,
        default="0.78,0.14",
        help="Arrow anchor in axes frac: x,y",
    )
    ap.add_argument(
        "--trend-arrow-min-n",
        type=int,
        default=4,
        help="Min points needed to fit trend.",
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


def _norm_fig_formats(
    formats: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    if not formats:
        return ("png", "svg", "pdf", "eps")

    out: list[str] = []
    for f in formats:
        s = str(f).strip().lower().lstrip(".")
        if not s:
            continue
        if s not in out:
            out.append(s)
    return tuple(out)


def resolve_fig_stem(out: Any) -> Path:
    """
    Resolve an output stem under scripts/figs/ if relative.

    If user passes a filename with suffix (e.g. foo.png),
    we treat it as a stem and strip the suffix.
    """
    p = resolve_fig_out(str(out))
    if p.suffix:
        p = p.with_suffix("")
    return p


def save_figure(
    fig: Any,
    out: Any,
    *,
    formats: Optional[Sequence[str]] = None,
    dpi: Optional[int] = None,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    transparent: bool = False,
    close: bool = True,
    verbose: bool = True,
    strict: bool = False,
) -> Dict[str, Path]:
    """
    Save a Matplotlib figure to multiple formats.

    Parameters
    ----------
    fig:
        Matplotlib Figure.
    out:
        Output stem or path. If relative, saved under scripts/figs/.
        If has suffix, suffix is stripped and treated as stem.
    formats:
        Iterable of extensions (png/svg/pdf/eps). Default exports all 4.
    dpi:
        DPI for raster formats (png). If None, uses current rcParams.
    bbox_inches, pad_inches, transparent:
        Passed through to fig.savefig.
    close:
        Close the figure after saving.
    verbose:
        Print a compact "[OK]" line with what was written.
    strict:
        If True, any save failure raises immediately.
        If False, failures are warned and other formats continue.

    Returns
    -------
    written:
        Mapping ext -> output Path for successfully written files.
    """
    fmts = _norm_fig_formats(formats)
    stem = resolve_fig_stem(out)
    ensure_dir(stem.parent)

    raster = {"png", "jpg", "jpeg", "tif", "tiff"}

    written: Dict[str, Path] = {}
    failed: Dict[str, str] = {}

    for ext in fmts:
        p = stem.with_suffix("." + ext)

        kw: Dict[str, Any] = dict(
            bbox_inches=bbox_inches,
            pad_inches=float(pad_inches),
            transparent=bool(transparent),
        )
        if ext in raster and dpi is not None:
            kw["dpi"] = int(dpi)

        try:
            fig.savefig(str(p), **kw)
            written[ext] = p
        except Exception as e:
            if strict:
                raise
            failed[ext] = repr(e)

    if close:
        plt.close(fig)

    if verbose:
        ok_exts = ",".join(sorted(written.keys()))
        msg = f"[OK] wrote {stem} ({ok_exts})"
        print(msg)
        if failed:
            bad = ", ".join(sorted(failed.keys()))
            print(f"[WARN] failed formats: {bad}")

    return written

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
    # out.forecast_val_csv = find_preferred(
    #     root,
    #     cfg.PATTERNS["forecast_val_csv"],
    # )
    
    # out.forecast_test_csv = find_preferred(
    #     root,
    #     cfg.PATTERNS["forecast_test_csv"],
    # )

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
    # if missing:
    #     raise KeyError(f"Missing columns in {p}: {missing}")
    if missing:
        if (
            "subsidence_actual" in missing
            and "future" in p.name.lower()
        ):
            raise KeyError(
                f"{p} looks like a future "
                "forecast (no actual). "
                "Use *_eval*_calibrated.csv."
            )
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
    """
    Normalize GeoPrior phys JSON to "mm" subs metrics.

    v3.2 may produce:
    - interpretable JSON: already in mm (no-op)
    - raw JSON: often distance-like metrics in meters

    We convert:
    - distance-like: MAE / RMSE / sharpness  -> × scale
    - squared:       MSE                     -> × scale²

    This conversion touches (if present):
    - metrics_evaluate: subs_pred_* keys
    - point_metrics
    - per_horizon: mae/mse/rmse/sharpness
    - interval_metrics: sharpness*
    - interval_calibration: sharpness* keys
    """
    if not meta:
        return {}

    out = dict(meta)
    units = dict((out.get("units") or {}))

    u = str(units.get("subs_metrics_unit", "")).lower()
    if u == "mm":
        return out

    # Prefer an explicit factor if provided.
    # Common: 1000 for m -> mm.
    scale = units.get("subs_factor_si_to_real", 1000.0)
    try:
        scale = float(scale)
    except Exception:
        scale = 1000.0

    if not np.isfinite(scale) or scale <= 0:
        scale = 1000.0

    scale2 = float(scale) ** 2

    def _scale_dist(d: Dict[str, Any], key: str) -> None:
        v = d.get(key, None)
        if isinstance(v, (int, float)) and np.isfinite(v):
            d[key] = float(v) * float(scale)

    def _scale_mse(d: Dict[str, Any], key: str) -> None:
        v = d.get(key, None)
        if isinstance(v, (int, float)) and np.isfinite(v):
            d[key] = float(v) * float(scale2)

    def _scale_map(
        d: Dict[str, Any],
        *,
        dist_keys: Tuple[str, ...],
        mse_keys: Tuple[str, ...],
    ) -> Dict[str, Any]:
        if not isinstance(d, dict):
            return {}
        dd = dict(d)
        for k in dist_keys:
            if k in dd:
                _scale_dist(dd, k)
        for k in mse_keys:
            if k in dd:
                _scale_mse(dd, k)
        return dd

    # ---------------------------------------------------------
    # metrics_evaluate (subs_pred_*)
    # ---------------------------------------------------------
    me = dict((out.get("metrics_evaluate") or {}))
    if isinstance(me, dict) and me:
        for k in list(me.keys()):
            if not str(k).startswith("subs_pred_"):
                continue

            kk = str(k).lower()

            # mse family: mse, mse_q50, etc.
            if "mse" in kk:
                _scale_mse(me, k)
                continue

            # distance-like: mae/rmse/sharpness, incl *_q50.
            if (
                "mae" in kk
                or "rmse" in kk
                or "sharp" in kk
            ):
                _scale_dist(me, k)

        out["metrics_evaluate"] = me

    # ---------------------------------------------------------
    # point_metrics
    # ---------------------------------------------------------
    pm = dict((out.get("point_metrics") or {}))
    if isinstance(pm, dict) and pm:
        pm = _scale_map(
            pm,
            dist_keys=("mae", "rmse"),
            mse_keys=("mse",),
        )
        out["point_metrics"] = pm

    # ---------------------------------------------------------
    # per_horizon metrics
    # (only scale numeric dict-of-dict blocks)
    # ---------------------------------------------------------
    ph = dict((out.get("per_horizon") or {}))
    if isinstance(ph, dict) and ph:
        ph2 = dict(ph)

        def _scale_hdict(hd: Dict[str, Any], mul: float) -> Dict[str, Any]:
            outd = dict(hd)
            for hk, hv in outd.items():
                if isinstance(hv, (int, float)) and np.isfinite(hv):
                    outd[hk] = float(hv) * float(mul)
            return outd

        for key, mul in [
            ("mae", scale),
            ("rmse", scale),
            ("sharpness80", scale),
            ("sharpness", scale),
            ("mse", scale2),
        ]:
            blk = ph2.get(key, None)
            if isinstance(blk, dict):
                ph2[key] = _scale_hdict(blk, float(mul))

        out["per_horizon"] = ph2

    # ---------------------------------------------------------
    # interval_metrics (if present)
    # ---------------------------------------------------------
    im = dict((out.get("interval_metrics") or {}))
    if isinstance(im, dict) and im:
        for k in list(im.keys()):
            if "sharp" in str(k).lower():
                _scale_dist(im, k)
        out["interval_metrics"] = im

    # ---------------------------------------------------------
    # interval_calibration
    # NOTE: these often contain both SI and phys fields.
    # Raw JSON can store these in meters; scale all sharpness*
    # keys to mm to make plotting consistent.
    # ---------------------------------------------------------
    ic = dict((out.get("interval_calibration") or {}))
    if isinstance(ic, dict) and ic:
        for k in list(ic.keys()):
            if "sharp" in str(k).lower():
                _scale_dist(ic, k)
        out["interval_calibration"] = ic

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
    """
    Pick interval metrics with calibrated preference.

    Preference order:
      1) phys_json.interval_calibration:
         - calibrated_phys (paper / physical)
         - calibrated      (often SI)
      2) phys_json.interval_metrics
      3) phys_json.metrics_evaluate (may be uncalibrated)
      4) eval diagnostics JSON (flattened)

    Returns
    -------
    coverage80 : float
        Unitless.
    sharpness80 : float
        Returned in mm when we can infer conversion.
    """
    cov = float("nan")
    shp = float("nan")
    shp_src = ""

    def _take(cur: float, v: Any) -> float:
        if np.isfinite(cur):
            return cur
        fv = to_float(v)
        return fv if np.isfinite(fv) else cur

    def _take_tag(cur: float, v: Any, tag: str) -> Tuple[float, str]:
        if np.isfinite(cur):
            return cur, tag
        fv = to_float(v)
        if np.isfinite(fv):
            return fv, tag
        return cur, ""

    # Conversion factor (SI -> "real") when available.
    # Common: 1000 for m -> mm.
    def _si_to_real_factor() -> float:
        if not phys_json:
            return 1000.0

        units = phys_json.get("units") or {}
        fac = units.get("subs_factor_si_to_real", None)
        try:
            fac = float(fac)
        except Exception:
            fac = 1000.0

        if not np.isfinite(fac) or fac <= 0:
            fac = 1000.0
        return float(fac)

    fac = _si_to_real_factor()

    if phys_json:
        ic = phys_json.get("interval_calibration") or {}
        if isinstance(ic, dict):
            cov = _take(
                cov,
                ic.get("coverage80_calibrated_phys"),
            )
            cov = _take(
                cov,
                ic.get("coverage80_calibrated"),
            )

            shp, tag = _take_tag(
                shp,
                ic.get("sharpness80_calibrated_phys"),
                "ic_phys",
            )
            if not tag:
                shp, tag = _take_tag(
                    shp,
                    ic.get("sharpness80_calibrated"),
                    "ic_si",
                )
            if tag:
                shp_src = tag

        im = phys_json.get("interval_metrics") or {}
        if isinstance(im, dict):
            cov = _take(cov, im.get("coverage80"))
            shp, tag = _take_tag(
                shp,
                im.get("sharpness80"),
                "im",
            )
            if tag:
                shp_src = tag

        me = phys_json.get("metrics_evaluate") or {}
        if isinstance(me, dict):
            cov = _take(
                cov,
                me.get("subs_pred_coverage80"),
            )
            shp, tag = _take_tag(
                shp,
                me.get("subs_pred_sharpness80"),
                "me",
            )
            if tag:
                shp_src = tag

    flat = flatten_eval_diag(fallback) if fallback else {}
    cov = _take(cov, flat.get("coverage80"))

    if not np.isfinite(shp):
        shp, tag = _take_tag(
            shp,
            flat.get("sharpness80"),
            "diag",
        )
        if tag:
            shp_src = tag

    # ---------------------------------------------------------
    # Unit harmonization for sharpness
    # - If we used SI-calibrated value from interval_calibration,
    #   convert to mm using fac.
    # - If we fell back to diagnostics and it looks SI (< 1),
    #   convert as well.
    # ---------------------------------------------------------
    if np.isfinite(shp):
        if shp_src == "ic_si":
            shp = float(shp) * float(fac)

        elif shp_src == "diag" and float(shp) < 1.0:
            shp = float(shp) * float(fac)

    return (float(cov), float(shp))

def flatten_eval_diag(
    diag: Dict[str, Any],
) -> Dict[str, float]:
    """
    Flatten eval diagnostics into keys used by plots.

    Supports:
      1) Flat: {"mae":..., "coverage80":...}
      2) "__overall__" block:
         {"2020.0": {...}, "__overall__": {...}}
      3) eval_after:
         {"overall_key": "__overall__",
          "eval_after": {"__overall__": {...}}}
    """
    if not diag:
        return {}

    out: Dict[str, float] = {}

    def _set_if(dst: str, v: Any) -> None:
        if dst in out:
            return
        fv = to_float(v)
        if np.isfinite(fv):
            out[dst] = fv

    blk = diag.get("__overall__")
    if isinstance(blk, dict):
        _set_if("mae", blk.get("overall_mae"))
        _set_if("mse", blk.get("overall_mse"))
        _set_if("rmse", blk.get("overall_rmse"))
        _set_if("r2", blk.get("overall_r2"))
        _set_if("coverage80", blk.get("coverage80"))
        _set_if("sharpness80", blk.get("sharpness80"))

    for k in ["r2", "mae", "mse", "rmse"]:
        if k in diag:
            _set_if(k, diag.get(k))

    for k in ["coverage80", "sharpness80"]:
        if k in diag:
            _set_if(k, diag.get(k))

    overall_key = str(diag.get("overall_key") or "")
    eval_after = diag.get("eval_after") or {}
    if isinstance(eval_after, dict):
        if overall_key and overall_key in eval_after:
            maybe = eval_after.get(overall_key) or {}
            if isinstance(maybe, dict):
                eval_after = maybe

        cov = eval_after.get("coverage", None)
        shp = eval_after.get("sharpness", None)

        if "coverage80" not in out and cov is not None:
            _set_if("coverage80", cov)

        if "sharpness80" not in out and shp is not None:
            _set_if("sharpness80", shp)

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
