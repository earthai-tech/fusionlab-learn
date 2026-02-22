# geoprior/runs/xfer/xfer_utils.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.runs.xfer.xfer_utils

v3.2 utilities extracted from the old stage5 script,
but written for GUI/backend use.

Design goals
------------
- Pure helpers (no Job / no GUI widgets).
- Robust to small manifest/config variations.
- Keep I/O small + predictable for xfer_core.

Notes
-----
- Static features: aligned by name (pad zeros for
  missing, ignore extras).
- Dynamic/Future features: strict by default
  (optional reorder-by-name in soft mode).
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np

from fusionlab.registry.utils import _find_stage1_manifest
from fusionlab.utils.generic_utils import ensure_directory_exists
from fusionlab.nn.calibration import IntervalCalibrator


LogFn = Callable[[str], Any]


# ---------------------------------------------------------------------
# small misc helpers
# ---------------------------------------------------------------------
def _log(log_fn: Optional[LogFn]) -> LogFn:
    return log_fn if callable(log_fn) else print


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> str:
    ensure_directory_exists(path)
    return path


def safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f))
    except Exception:
        return {}


def np_load_dict(path: str) -> Dict[str, Any]:
    return dict(np.load(path))


# ---------------------------------------------------------------------
# manifest + model artifact utilities
# ---------------------------------------------------------------------
def find_stage1_manifest(
    *,
    results_dir: str,
    city: str,
    model_name: Optional[str] = None,
    prefer: str = "timestamp",
) -> str:
    """
    Locate the stage-1 manifest for `city`.

    Parameters
    ----------
    results_dir:
        Base results folder that contains runs.
    city:
        City hint (e.g. "nansha", "zhongshan").
    model_name:
        Optional model name hint.
    prefer:
        Match policy used by _find_stage1_manifest.

    Returns
    -------
    path:
        Manifest path.
    """
    hint = model_name or os.getenv(
        "MODEL_NAME_OVERRIDE",
        "GeoPriorSubsNet",
    )
    return _find_stage1_manifest(
        manual=None,
        base_dir=results_dir,
        city_hint=city,
        model_hint=hint,
        prefer=prefer,
        required_keys=("model", "stage"),
        verbose=0,
    )


def load_stage1_manifest(
    *,
    results_dir: str,
    city: str,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    mpath = find_stage1_manifest(
        results_dir=results_dir,
        city=city,
        model_name=model_name,
    )
    M = safe_load_json(mpath)
    M["manifest_path"] = mpath
    return M


def manifest_run_dir(M: Dict[str, Any]) -> str:
    paths = M.get("paths") or {}
    if isinstance(paths, dict):
        for k in ("run_dir", "stage1_dir", "root_dir"):
            v = paths.get(k)
            if isinstance(v, str) and v:
                return v
    mpath = M.get("manifest_path")
    if isinstance(mpath, str) and os.path.isfile(mpath):
        return os.path.dirname(mpath)
    return ""


def best_model_artifact(
    run_dir: str,
    *,
    prefer_tuned: bool = True,
) -> Optional[str]:
    """
    Return the newest model artifact under run_dir.

    We support:
    - tuned .keras under tuning/
    - train_*/.keras
    - *_best_savedmodel dirs
    """
    if not run_dir or not os.path.isdir(run_dir):
        return None

    pats: List[str] = []
    if prefer_tuned:
        pats.append(
            os.path.join(run_dir, "tuning", "**", "*.keras")
        )
    pats.extend(
        [
            os.path.join(run_dir, "train_*", "*.keras"),
            os.path.join(run_dir, "**", "*_best.keras"),
            os.path.join(run_dir, "**", "*.keras"),
            os.path.join(run_dir, "**", "*_best_savedmodel"),
        ]
    )

    cands: List[Tuple[float, str]] = []
    for pat in pats:
        for p in glob.glob(pat, recursive=True):
            try:
                cands.append((os.path.getmtime(p), p))
            except Exception:
                pass

    if not cands:
        return None

    cands.sort(reverse=True)
    return cands[0][1]


def resolve_bundle_paths(model_path: str) -> Dict[str, Any]:
    """
    Normalize Keras/SavedModel bundle paths.

    Returns keys:
    - run_dir
    - keras_path
    - weights_path (optional)
    - tf_dir (optional)
    - init_manifest_path
    """
    mp = os.path.abspath(model_path)
    run_dir = mp if os.path.isdir(mp) else os.path.dirname(mp)

    tf_dir: Optional[str] = None
    keras_path: Optional[str] = None
    prefix: Optional[str] = None

    if os.path.isdir(mp) and mp.endswith("_best_savedmodel"):
        tf_dir = mp
        prefix = mp[: -len("_best_savedmodel")]
        keras_path = prefix + "_best.keras"
    else:
        keras_path = mp
        if keras_path.endswith("_best.keras"):
            prefix = keras_path[: -len("_best.keras")]
        elif keras_path.endswith(".keras"):
            prefix = keras_path[: -len(".keras")]
        else:
            prefix = os.path.join(run_dir, "model")

        cand = prefix + "_best_savedmodel"
        if os.path.isdir(cand):
            tf_dir = cand

    weights_path = prefix + "_best.weights.h5"
    if not os.path.isfile(weights_path):
        weights_path = None

    init_path = os.path.join(
        run_dir,
        "model_init_manifest.json",
    )

    return {
        "run_dir": run_dir,
        "keras_path": keras_path,
        "weights_path": weights_path,
        "tf_dir": tf_dir,
        "init_manifest_path": init_path,
    }


# ---------------------------------------------------------------------
# config compat (v3.0 NAT vs v3.2 GeoPriorConfig style)
# ---------------------------------------------------------------------
def cfg_get(
    cfg: Dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for k in keys:
        if not k:
            continue
        if k in cfg:
            return cfg[k]
    return default


def cfg_features(cfg: Dict[str, Any], kind: str) -> List[str]:
    """
    kind in {"static","dynamic","future_known","future"}.
    """
    feats = cfg.get("features")
    if isinstance(feats, dict) and kind in feats:
        return [str(x) for x in (feats.get(kind) or [])]

    if kind == "static":
        v = cfg_get(cfg, "static_features", default=[])
        return [str(x) for x in (v or [])]

    if kind == "dynamic":
        v = cfg_get(cfg, "dynamic_features", default=[])
        return [str(x) for x in (v or [])]

    if kind in {"future_known", "future"}:
        v = cfg_get(cfg, "future_known_features", default=[])
        return [str(x) for x in (v or [])]

    return []


def cfg_mode(cfg: Dict[str, Any]) -> str:
    return str(cfg_get(cfg, "mode", "MODE", default="tft_like"))


def cfg_horizon(cfg: Dict[str, Any], fallback: int) -> int:
    v = cfg_get(
        cfg,
        "forecast_horizon_years",
        "FORECAST_HORIZON_YEARS",
        default=fallback,
    )
    try:
        return int(v)
    except Exception:
        return int(fallback)


def cfg_quantiles(
    cfg: Dict[str, Any],
    override: Optional[List[float]] = None,
) -> List[float]:
    if override is not None:
        return [float(q) for q in override]
    q = cfg_get(cfg, "quantiles", "QUANTILES", default=None)
    if q is None:
        return [0.1, 0.5, 0.9]
    try:
        return [float(x) for x in list(q)]
    except Exception:
        return [0.1, 0.5, 0.9]


def feature_list(M: Dict[str, Any], kind: str) -> List[str]:
    cfg = M.get("config") or {}
    if not isinstance(cfg, dict):
        return []
    return cfg_features(cfg, kind)


# ---------------------------------------------------------------------
# npz helpers
# ---------------------------------------------------------------------
def pick_npz(
    M: Dict[str, Any],
    split: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    npzs = (M.get("artifacts") or {}).get("numpy") or {}
    if split == "train":
        xi = npzs["train_inputs_npz"]
        yt = npzs["train_targets_npz"]
        return np_load_dict(xi), np_load_dict(yt)

    if split == "val":
        xi = npzs["val_inputs_npz"]
        yt = npzs["val_targets_npz"]
        return np_load_dict(xi), np_load_dict(yt)

    if split == "test":
        xi = npzs.get("test_inputs_npz")
        yt = npzs.get("test_targets_npz")
        if not xi:
            return None, None
        x = np_load_dict(xi)
        y = np_load_dict(yt) if yt else None
        return x, y

    raise ValueError(split)


def map_targets(y: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not y:
        return {}
    if "subs_pred" in y and "gwl_pred" in y:
        return dict(y)
    if "subsidence" in y and "gwl" in y:
        return {"subs_pred": y["subsidence"], "gwl_pred": y["gwl"]}
    return {}


def infer_input_dims(M: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Infer (static, dynamic, future) dims from manifest.
    """
    seq = (M.get("artifacts") or {}).get("sequences") or {}
    dims = seq.get("dims") or {}

    s_dim = dims.get("static_input_dim")
    d_dim = dims.get("dynamic_input_dim")
    f_dim = dims.get("future_input_dim")

    shapes = (M.get("artifacts") or {}).get("shapes") or {}
    tr_in = shapes.get("train_inputs") or {}

    if s_dim is None:
        sf = tr_in.get("static_features")
        if isinstance(sf, (list, tuple)) and len(sf) >= 2:
            s_dim = sf[-1]

    if d_dim is None:
        df = tr_in.get("dynamic_features")
        if isinstance(df, (list, tuple)) and len(df) >= 3:
            d_dim = df[-1]

    if f_dim is None:
        ff = tr_in.get("future_features")
        if isinstance(ff, (list, tuple)) and len(ff) >= 3:
            f_dim = ff[-1]

    if s_dim is None:
        s_dim = len(feature_list(M, "static"))
    if d_dim is None:
        d_dim = len(feature_list(M, "dynamic"))
    if f_dim is None:
        f_dim = len(feature_list(M, "future"))

    return int(s_dim or 0), int(d_dim or 0), int(f_dim or 0)


# ---------------------------------------------------------------------
# schema alignment (static name-align, dynamic/future strict)
# ---------------------------------------------------------------------
def _schema_diff(
    src: List[str],
    tgt: List[str],
) -> Tuple[List[str], List[str], bool]:
    src = [str(x) for x in (src or [])]
    tgt = [str(x) for x in (tgt or [])]

    missing = [x for x in src if x not in tgt]
    extra = [x for x in tgt if x not in src]

    reorder = False
    if not missing and not extra:
        reorder = src != tgt

    return missing, extra, reorder


def _short_list(xs: List[str], n: int = 8) -> str:
    xs = [str(x) for x in (xs or [])]
    if len(xs) <= n:
        return ", ".join(xs)
    head = ", ".join(xs[:n])
    return f"{head}, ...(+{len(xs) - n})"


def _reorder_last_dim(
    arr: Any,
    src_feats: List[str],
    tgt_feats: List[str],
) -> np.ndarray:
    a = np.asarray(arr)
    name2idx = {n: i for i, n in enumerate(tgt_feats)}
    idx = [int(name2idx[n]) for n in src_feats]
    return a[..., idx].astype(np.float32)


def align_static_to_source(
    X_tgt: Dict[str, Any],
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
    *,
    log_fn: Optional[LogFn] = None,
) -> Dict[str, Any]:
    """
    Align target static features to source order by name.
    Missing -> 0, extras ignored.
    """
    log = _log(log_fn)
    static_src = feature_list(M_src, "static")
    static_tgt = feature_list(M_tgt, "static")

    if not static_src:
        N = int(np.asarray(X_tgt["dynamic_features"]).shape[0])
        out = dict(X_tgt)
        out["static_features"] = np.zeros((N, 0), np.float32)
        return out

    miss, extra, reorder = _schema_diff(static_src, static_tgt)
    if miss or extra or reorder:
        overlap = len([x for x in static_src if x in static_tgt])
        log(
            "[xfer] Static schema differs; aligning by name.\n"
            f"[xfer] overlap={overlap} "
            f"missing={len(miss)} extra={len(extra)}"
        )

    old = X_tgt.get("static_features")
    N = int(np.asarray(X_tgt["dynamic_features"]).shape[0])

    if old is None or int(np.asarray(old).shape[-1]) == 0:
        out = dict(X_tgt)
        out["static_features"] = np.zeros(
            (N, len(static_src)),
            np.float32,
        )
        return out

    old = np.asarray(old).astype(np.float32)
    name2idx = {n: i for i, n in enumerate(static_tgt)}

    new = np.zeros((N, len(static_src)), np.float32)
    for j, name in enumerate(static_src):
        idx = name2idx.get(name)
        if idx is None:
            continue
        if idx < int(old.shape[1]):
            new[:, j] = old[:, idx]

    out = dict(X_tgt)
    out["static_features"] = new
    return out


def _raise_schema_error(
    *,
    kind: str,
    src_city: str,
    tgt_city: str,
    expected_dim: int,
    got_dim: int,
    src_feats: List[str],
    tgt_feats: List[str],
) -> None:
    miss, extra, reorder = _schema_diff(src_feats, tgt_feats)

    lines: List[str] = []
    lines.append(
        f"{kind} schema mismatch ({src_city}->{tgt_city})."
    )
    lines.append(
        f"expected_dim={expected_dim} got_dim={got_dim}"
    )

    if miss:
        lines.append(f"missing_in_target: {_short_list(miss)}")
    if extra:
        lines.append(f"extra_in_target: {_short_list(extra)}")
    if reorder and not (miss or extra):
        lines.append("same names but different ORDER.")

    lines.append("Fix: harmonize Stage-1 feature lists.")
    raise SystemExit("\n".join(lines))


def check_transfer_schema(
    *,
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
    X_tgt: Dict[str, Any],
    allow_reorder_dynamic: bool = False,
    allow_reorder_future: bool = False,
    log_fn: Optional[LogFn] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate transfer schemas and optionally reorder by name.

    Returns
    -------
    X_new:
        Possibly reordered dict (dynamic/future).
    audit:
        Lightweight audit dict for logging/JSON.
    """
    log = _log(log_fn)

    src_city = str(M_src.get("city", "src"))
    tgt_city = str(M_tgt.get("city", "tgt"))

    s_src, d_src, f_src = infer_input_dims(M_src)

    audit: Dict[str, Any] = {
        "src_city": src_city,
        "tgt_city": tgt_city,
        "dynamic_reordered": False,
        "future_reordered": False,
        "dynamic_order_mismatch": False,
        "future_order_mismatch": False,
    }

    s_tgt = int(np.asarray(X_tgt["static_features"]).shape[-1])
    d_tgt = int(np.asarray(X_tgt["dynamic_features"]).shape[-1])
    f_tgt = int(np.asarray(X_tgt["future_features"]).shape[-1])

    if s_src != s_tgt:
        raise SystemExit(
            "Static dim mismatch after alignment.\n"
            f"expected={s_src} got={s_tgt}"
        )

    dyn_src = feature_list(M_src, "dynamic")
    dyn_tgt = feature_list(M_tgt, "dynamic")
    miss, extra, reorder = _schema_diff(dyn_src, dyn_tgt)

    if d_src != d_tgt or miss or extra:
        _raise_schema_error(
            kind="dynamic",
            src_city=src_city,
            tgt_city=tgt_city,
            expected_dim=d_src,
            got_dim=d_tgt,
            src_feats=dyn_src,
            tgt_feats=dyn_tgt,
        )

    X_new = dict(X_tgt)

    if reorder:
        audit["dynamic_order_mismatch"] = True
        if not allow_reorder_dynamic:
            _raise_schema_error(
                kind="dynamic",
                src_city=src_city,
                tgt_city=tgt_city,
                expected_dim=d_src,
                got_dim=d_tgt,
                src_feats=dyn_src,
                tgt_feats=dyn_tgt,
            )
        audit["dynamic_reordered"] = True
        log(
            "[xfer] WARNING: dynamic order differs; "
            "reordering target by name."
        )
        X_new["dynamic_features"] = _reorder_last_dim(
            X_new["dynamic_features"],
            src_feats=dyn_src,
            tgt_feats=dyn_tgt,
        )

    fut_src = feature_list(M_src, "future")
    fut_tgt = feature_list(M_tgt, "future")
    miss, extra, reorder = _schema_diff(fut_src, fut_tgt)

    if f_src != f_tgt or miss or extra:
        _raise_schema_error(
            kind="future",
            src_city=src_city,
            tgt_city=tgt_city,
            expected_dim=f_src,
            got_dim=f_tgt,
            src_feats=fut_src,
            tgt_feats=fut_tgt,
        )

    if reorder:
        audit["future_order_mismatch"] = True
        if not allow_reorder_future:
            _raise_schema_error(
                kind="future",
                src_city=src_city,
                tgt_city=tgt_city,
                expected_dim=f_src,
                got_dim=f_tgt,
                src_feats=fut_src,
                tgt_feats=fut_tgt,
            )
        audit["future_reordered"] = True
        log(
            "[xfer] WARNING: future order differs; "
            "reordering target by name."
        )
        X_new["future_features"] = _reorder_last_dim(
            X_new["future_features"],
            src_feats=fut_src,
            tgt_feats=fut_tgt,
        )

    return X_new, audit


# ---------------------------------------------------------------------
# scaler + rescale helpers
# ---------------------------------------------------------------------
def _load_scalers(enc: Dict[str, Any]) -> Tuple[Any, Any]:
    coord_scaler = None
    cs_path = enc.get("coord_scaler")
    if cs_path and os.path.exists(cs_path):
        try:
            coord_scaler = joblib.load(cs_path)
        except Exception:
            coord_scaler = None

    scaler_info = enc.get("scaler_info")
    if isinstance(scaler_info, str) and os.path.exists(scaler_info):
        try:
            scaler_info = joblib.load(scaler_info)
        except Exception:
            scaler_info = None

    if isinstance(scaler_info, dict):
        for _, v in scaler_info.items():
            if not isinstance(v, dict):
                continue
            if "scaler" in v:
                continue
            p = v.get("scaler_path")
            if p and os.path.exists(p):
                try:
                    v["scaler"] = joblib.load(p)
                except Exception:
                    pass

    return coord_scaler, scaler_info


def _get_scaler(scaler_info: Any, name: str) -> Any:
    if not isinstance(scaler_info, dict):
        return None
    blk = scaler_info.get(name)
    if not isinstance(blk, dict):
        return None
    return blk.get("scaler")


def _transform_with_scaler(scaler: Any, arr: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(arr).astype(np.float32)
    a = np.asarray(arr)
    shp = a.shape
    last = int(shp[-1]) if shp else 1
    flat = a.reshape(-1, last)
    out = scaler.transform(flat)
    return out.reshape(shp).astype(np.float32)


def reproject_dynamic_to_source(
    X_tgt: Dict[str, Any],
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert target dynamic_features to source scaling,
    feature-by-feature (requires both scaler infos).

    This implements the stage5 "strict" rescale mode.
    """
    enc_s = (M_src.get("artifacts") or {}).get("encoders") or {}
    enc_t = (M_tgt.get("artifacts") or {}).get("encoders") or {}

    _, sc_s = _load_scalers(enc_s)
    _, sc_t = _load_scalers(enc_t)

    if not isinstance(sc_s, dict) or not isinstance(sc_t, dict):
        return X_tgt

    dyn_names = feature_list(M_tgt, "dynamic")
    if not dyn_names:
        return X_tgt

    X = dict(X_tgt)
    dyn = np.asarray(X["dynamic_features"]).astype(np.float32)
    dyn2 = np.array(dyn, copy=True)

    for j, name in enumerate(dyn_names):
        s_t = _get_scaler(sc_t, name)
        s_s = _get_scaler(sc_s, name)
        if s_t is None or s_s is None:
            continue

        col = dyn[:, :, j : j + 1]
        phys = s_t.inverse_transform(col.reshape(-1, 1))
        phys = phys.reshape(col.shape).astype(np.float32)
        dyn2[:, :, j : j + 1] = _transform_with_scaler(
            s_s, phys
        )

    X["dynamic_features"] = dyn2
    return X


# ---------------------------------------------------------------------
# calibration utilities
# ---------------------------------------------------------------------
def load_calibrator_near(
    run_dir: str,
    *,
    target: float = 0.80,
) -> Optional[IntervalCalibrator]:
    cands: List[Tuple[float, str]] = []
    pats = [
        os.path.join(run_dir, "interval_factors_80.npy"),
        os.path.join(run_dir, "**", "interval_factors_80.npy"),
    ]
    for pat in pats:
        for p in glob.glob(pat, recursive=True):
            try:
                cands.append((os.path.getmtime(p), p))
            except Exception:
                pass

    if not cands:
        return None

    cands.sort(reverse=True)
    path = cands[0][1]

    try:
        cal = IntervalCalibrator(target=float(target))
        cal.factors_ = np.load(path).astype(np.float32)
        return cal
    except Exception:
        return None


# ---------------------------------------------------------------------
# warm-start sampling utilities
# ---------------------------------------------------------------------
def choose_warm_idx(
    *,
    n_total: int,
    n_samples: int,
    frac: Optional[float],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))

    if frac is not None:
        n = int(max(1, round(n_total * float(frac))))
    else:
        n = int(max(1, min(n_total, int(n_samples))))

    if n >= n_total:
        return np.arange(n_total, dtype=np.int64)

    return rng.choice(
        n_total,
        size=n,
        replace=False,
    ).astype(np.int64)


def slice_npz_dict(
    d: Dict[str, Any],
    idx: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        a = np.asarray(v)
        if a.ndim >= 1 and int(a.shape[0]) >= len(idx):
            out[k] = a[idx]
        else:
            out[k] = a
    return out


__all__ = [
    "LogFn",
    "now_tag",
    "ensure_dir",
    "safe_load_json",
    "np_load_dict",
    "find_stage1_manifest",
    "load_stage1_manifest",
    "manifest_run_dir",
    "best_model_artifact",
    "resolve_bundle_paths",
    "cfg_get",
    "cfg_features",
    "cfg_mode",
    "cfg_horizon",
    "cfg_quantiles",
    "feature_list",
    "pick_npz",
    "map_targets",
    "infer_input_dims",
    "align_static_to_source",
    "check_transfer_schema",
    "reproject_dynamic_to_source",
    "load_calibrator_near",
    "choose_warm_idx",
    "slice_npz_dict",
]
