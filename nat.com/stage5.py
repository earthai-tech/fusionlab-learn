# nat/com/xfer_matrix(v3.2).py
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

# 
# python stage5.py \
#   --city-a nansha \
#   --city-b zhongshan \
#   --strategies baseline xfer warm \
#   --rescale-modes as_is strict \
#   --splits val test \
#   --calib-modes none source target \
#   --warm-split train \
#   --warm-samples 20000 \
#   --warm-epochs 3 \
#   --warm-lr 1e-4
# e.g. python stage5.py ... --warm-split val --warm-samples 5000


from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Callable

import joblib
import numpy as np
import tensorflow as tf

from fusionlab.registry.utils import (
    _find_stage1_manifest,
)
from fusionlab.utils.generic_utils import (
    ensure_directory_exists,
)
from fusionlab.utils.scale_metrics import (
    inverse_scale_target,
    per_horizon_metrics,
    point_metrics,
)
from fusionlab.utils.forecast_utils import (
    format_and_forecast,
)
from fusionlab.utils.nat_utils import (
    ensure_input_shapes,
    extract_preds,
    load_best_hps_near_model,
    map_targets_for_training,
    sanitize_inputs_np,
)
from fusionlab.nn.keras_metrics import (
    coverage80_fn,
    sharpness80_fn,
)
from fusionlab.nn.calibration import (
    IntervalCalibrator,
    apply_calibrator_to_subs,
    fit_interval_calibrator_on_val,
)
from fusionlab.compat.keras import (
    load_inference_model,
    load_model_from_tfv2,
)
from fusionlab.nn.pinn.geoprior.models import (
    GeoPriorSubsNet,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cross-city transfer evaluation "
            "(baseline + transfer + warm-start)."
        )
    )
    p.add_argument("--city-a", default="nansha")
    p.add_argument("--city-b", default="zhongshan")
    p.add_argument(
        "--results-dir",
        default=os.getenv("RESULTS_DIR", "results"),
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
        help="Which eval splits to run.",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["baseline", "xfer"],
        choices=["baseline", "xfer", "warm"],
        help=(
            "baseline: A->A,B->B | "
            "xfer: A->B,B->A | "
            "warm: warm-start A->B,B->A"
        ),
    )
    p.add_argument(
        "--calib-modes",
        "--b-modes",
        dest="calib_modes",
        nargs="+",
        default=["none", "source", "target"],
        choices=["none", "source", "target"],
        help="Calibration modes to evaluate.",
    )
    p.add_argument(
        "--rescale-modes",
        nargs="+",
        default=["as_is"],
        choices=["as_is", "strict"],
        help=(
            "as_is: keep target scaling | "
            "strict: reproject to source scaling"
        ),
    )
    p.add_argument(
        "--rescale-to-source",
        action="store_true",
        help=(
            "Deprecated alias for --rescale-modes strict."
        ),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--quantiles",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Override quantiles (else read manifest)."
        ),
    )

    # Warm-start settings (only when strategy=warm)
    p.add_argument(
        "--warm-split",
        default="train",
        choices=["train", "val"],
        help="Target split used for warm-start.",
    )
    p.add_argument(
        "--warm-samples",
        type=int,
        default=20000,
        help="Max samples used for warm-start.",
    )
    p.add_argument(
        "--warm-frac",
        type=float,
        default=None,
        help="Fraction used (overrides warm-samples).",
    )
    p.add_argument(
        "--warm-epochs",
        type=int,
        default=3,
        help="Warm-start epochs.",
    )
    p.add_argument(
        "--warm-lr",
        type=float,
        default=1e-4,
        help="Warm-start learning rate.",
    )
    p.add_argument(
        "--warm-seed",
        type=int,
        default=123,
        help="Warm-start sampling seed.",
    )
    
    p.add_argument(
        "--allow-reorder-dynamic",
        action="store_true",
        help=(
            "Soft mode: allow reordering target "
            "dynamic features to source order "
            "(only if same names)."
        ),
    )
    p.add_argument(
        "--allow-reorder-future",
        action="store_true",
        help=(
            "Soft mode: allow reordering target "
            "future features to source order "
            "(only if same names)."
        ),
    )
    p.add_argument(
        "--log",
        default="print",
        choices=["print", "none"],
        help="Logging: print or none.",
    )

    args = p.parse_args()

    if args.rescale_to_source:
        args.rescale_modes = ["strict"]

    return args


def _load_manifest_for_city(
    city: str,
    results_dir: str,
) -> Dict[str, Any]:
    mpath = _find_stage1_manifest(
        manual=None,
        base_dir=results_dir,
        city_hint=city,
        model_hint=os.getenv(
            "MODEL_NAME_OVERRIDE",
            "GeoPriorSubsNet",
        ),
        prefer="timestamp",
        required_keys=("model", "stage"),
        verbose=0,
    )
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)


def _best_model_artifact(run_dir: str) -> str | None:
    pats = [
        os.path.join(run_dir, "**", "*_best.keras"),
        os.path.join(run_dir, "**", "*.keras"),
        os.path.join(run_dir, "**", "*_best_savedmodel"),
    ]
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


def _resolve_bundle_paths(model_path: str) -> Dict[str, Any]:
    mp = os.path.abspath(model_path)
    run_dir = mp if os.path.isdir(mp) else os.path.dirname(mp)

    tf_dir = None
    keras_path = None
    prefix = None

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


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_scalers(
    enc: Dict[str, Any],
) -> Tuple[Any, Any]:
    coord_scaler = None
    cs_path = enc.get("coord_scaler")
    if cs_path and os.path.exists(cs_path):
        try:
            coord_scaler = joblib.load(cs_path)
        except Exception:
            coord_scaler = None

    scaler_info = enc.get("scaler_info")
    if (
        isinstance(scaler_info, str)
        and os.path.exists(scaler_info)
    ):
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


def _get_scaler(
    scaler_info: Any,
    name: str,
) -> Any:
    if not isinstance(scaler_info, dict):
        return None
    blk = scaler_info.get(name)
    if not isinstance(blk, dict):
        return None
    return blk.get("scaler")


def _transform_with_scaler(
    scaler: Any,
    arr: np.ndarray,
) -> np.ndarray:
    if scaler is None:
        return arr
    a = np.asarray(arr)
    shp = a.shape
    last = int(shp[-1]) if shp else 1
    flat = a.reshape(-1, last)
    out = scaler.transform(flat)
    return out.reshape(shp).astype(np.float32)


def _ensure_np_inputs(
    x: Dict[str, Any],
    mode: str,
    horizon: int,
) -> Dict[str, Any]:
    x = sanitize_inputs_np(x)
    x = ensure_input_shapes(x, mode, horizon)
    return x


def _pick_npz(M: Dict[str, Any], split: str):
    npzs = M["artifacts"]["numpy"]
    if split == "train":
        xi = npzs["train_inputs_npz"]
        yt = npzs["train_targets_npz"]
        return dict(np.load(xi)), dict(np.load(yt))
    if split == "val":
        xi = npzs["val_inputs_npz"]
        yt = npzs["val_targets_npz"]
        return dict(np.load(xi)), dict(np.load(yt))
    if split == "test":
        xi = npzs.get("test_inputs_npz")
        yt = npzs.get("test_targets_npz")
        if not xi:
            return None, None
        x = dict(np.load(xi))
        y = dict(np.load(yt)) if yt else None
        return x, y
    raise ValueError(split)


def _feature_list(
    M: Dict[str, Any],
    kind: str,
) -> List[str]:
    cfg = M.get("config") or {}
    feats = cfg.get("features") or {}
    out = feats.get(kind) or []
    if isinstance(out, list):
        return [str(x) for x in out]
    return []

def _short_list(xs: List[str], n: int = 8) -> str:
    xs = [str(x) for x in (xs or [])]
    if len(xs) <= n:
        return ", ".join(xs)
    head = ", ".join(xs[:n])
    return f"{head}, ...(+{len(xs) - n})"


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

def _reorder_last_dim(
    arr: Any,
    src_feats: List[str],
    tgt_feats: List[str],
) -> np.ndarray:
    a = np.asarray(arr)
    name2idx = {n: i for i, n in enumerate(tgt_feats)}
    idx = [int(name2idx[n]) for n in src_feats]
    return a[..., idx].astype(np.float32)

def _print_static_alignment_note(
    *,
    src_city: str,
    tgt_city: str,
    static_src: List[str],
    static_tgt: List[str],
    log_fn: Optional[Callable[[str], Any]] = None,
) -> None:
    log = log_fn if callable(log_fn) else print
    
    missing, extra, reorder = _schema_diff(
        static_src,
        static_tgt,
    )
    if not (missing or extra or reorder):
        return

    overlap_n = len(
        [x for x in static_src if x in static_tgt]
    )

    msg = (
        "[stage5] Static schema differs "
        f"({src_city} -> {tgt_city}).\n"
        "[stage5] Target static is aligned to "
        "source by name.\n"
        f"[stage5] overlap={overlap_n} "
        f"missing_in_target={len(missing)} "
        f"extra_in_target={len(extra)}\n"
        "[stage5] Missing source features => 0. "
        "Extra target => ignored.\n"
        "[stage5] Interpretation: transfer uses "
        "shared static info.\n"
    )
    log(msg)

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
    missing, extra, reorder = _schema_diff(
        src_feats,
        tgt_feats,
    )

    lines: List[str] = []
    lines.append(
        f"{kind} feature schema mismatch "
        f"({src_city} -> {tgt_city})."
    )
    lines.append(
        f"expected_dim={expected_dim} got_dim={got_dim}"
    )

    if missing:
        lines.append(
            "missing_in_target: "
            f"{_short_list(missing)}"
        )
    if extra:
        lines.append(
            "extra_in_target: "
            f"{_short_list(extra)}"
        )
    if reorder and not (missing or extra):
        lines.append(
            "same names but different ORDER."
        )

    lines.append(
        "Fix: harmonize Stage-1 feature lists "
        "across cities."
    )
    lines.append(
        "Use same columns and same order for "
        f"{kind} features."
    )

    raise SystemExit("\n".join(lines))

def _check_transfer_schema(
    *,
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
    X_tgt: Dict[str, Any],
    s_src: int,
    d_src: int,
    f_src: int,
    allow_reorder_dynamic: bool = False,
    allow_reorder_future: bool = False,
    log_fn: Optional[Callable[[str], Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    log = log_fn if callable(log_fn) else print

    src_city = str(M_src.get("city", "src"))
    tgt_city = str(M_tgt.get("city", "tgt"))

    schema_audit: Dict[str, Any] = {
        "src_city": src_city,
        "tgt_city": tgt_city,
        "static_aligned": False,
        "dynamic_reordered": False,
        "future_reordered": False,
        "dynamic_order_mismatch": False,
        "future_order_mismatch": False,
        "static_missing_n": 0,
        "static_extra_n": 0,
    }

    static_src = _feature_list(M_src, "static")
    static_tgt = _feature_list(M_tgt, "static")

    miss_s, extra_s, reorder_s = _schema_diff(
        static_src,
        static_tgt,
    )
    schema_audit["static_missing_n"] = len(miss_s)
    schema_audit["static_extra_n"] = len(extra_s)
    schema_audit["static_aligned"] = bool(
        miss_s or extra_s or reorder_s
    )

    _print_static_alignment_note(
        src_city=src_city,
        tgt_city=tgt_city,
        static_src=static_src,
        static_tgt=static_tgt,
        log_fn=log_fn,
    )

    s_tgt = int(X_tgt["static_features"].shape[-1])
    d_tgt = int(X_tgt["dynamic_features"].shape[-1])
    f_tgt = int(X_tgt["future_features"].shape[-1])

    if s_src != s_tgt:
        raise SystemExit(
            "Static dim mismatch after alignment:\n"
            f"expected={s_src} got={s_tgt}\n"
            "Static alignment failed. Check "
            "Stage-1 manifests and arrays."
        )

    dyn_src = _feature_list(M_src, "dynamic")
    dyn_tgt = _feature_list(M_tgt, "dynamic")
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

    if reorder:
        schema_audit["dynamic_order_mismatch"] = True
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

        schema_audit["dynamic_reordered"] = True

        log(
            "[stage5] WARNING: dynamic order differs "
            f"({src_city} -> {tgt_city})."
        )
        log(
            "[stage5] Soft mode: reordering "
            "target dynamic_features to match "
            "source order (by name)."
        )
        log(
            "[stage5] Caution: transfer semantics "
            "depend on correct feature mapping. "
            "Prefer harmonized Stage-1 schemas."
        )

        X_tgt = dict(X_tgt)
        X_tgt["dynamic_features"] = _reorder_last_dim(
            X_tgt["dynamic_features"],
            src_feats=dyn_src,
            tgt_feats=dyn_tgt,
        )

    fut_src = _feature_list(M_src, "future")
    fut_tgt = _feature_list(M_tgt, "future")
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
        schema_audit["future_order_mismatch"] = True
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

        schema_audit["future_reordered"] = True

        log(
            "[stage5] WARNING: future order differs "
            f"({src_city} -> {tgt_city})."
        )
        log(
            "[stage5] Soft mode: reordering "
            "target future_features to match "
            "source order (by name)."
        )

        X_tgt = dict(X_tgt)
        X_tgt["future_features"] = _reorder_last_dim(
            X_tgt["future_features"],
            src_feats=fut_src,
            tgt_feats=fut_tgt,
        )

    return X_tgt, schema_audit


def _align_static_to_source(
    X_tgt: Dict[str, Any],
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
) -> Dict[str, Any]:
    static_src = _feature_list(M_src, "static")
    static_tgt = _feature_list(M_tgt, "static")

    N = int(X_tgt["dynamic_features"].shape[0])

    if not static_src:
        X_tgt["static_features"] = np.zeros(
            (N, 0),
            dtype=np.float32,
        )
        return X_tgt

    old = X_tgt.get("static_features")
    if old is None or int(old.shape[-1]) == 0:
        X_tgt["static_features"] = np.zeros(
            (N, len(static_src)),
            dtype=np.float32,
        )
        return X_tgt

    name2idx = {n: i for i, n in enumerate(static_tgt)}

    new = np.zeros(
        (N, len(static_src)),
        dtype=np.float32,
    )
    for j, name in enumerate(static_src):
        idx = name2idx.get(name)
        if idx is None:
            continue
        if idx < int(old.shape[1]):
            new[:, j] = old[:, idx]

    X_tgt["static_features"] = new
    return X_tgt


def _infer_input_dims(
    M: Dict[str, Any],
) -> Tuple[int, int, int]:
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
        s_dim = len(_feature_list(M, "static"))
    if d_dim is None:
        d_dim = len(_feature_list(M, "dynamic"))
    if f_dim is None:
        f_dim = len(_feature_list(M, "future"))

    return int(s_dim or 0), int(d_dim or 0), int(f_dim or 0)


def _reproject_dynamic_to_source(
    X_tgt: Dict[str, Any],
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
) -> Dict[str, Any]:
    enc_s = M_src["artifacts"]["encoders"]
    enc_t = M_tgt["artifacts"]["encoders"]

    _, sc_s = _load_scalers(enc_s)
    _, sc_t = _load_scalers(enc_t)

    if (
        not isinstance(sc_s, dict)
        or not isinstance(sc_t, dict)
    ):
        return X_tgt

    dyn_names = _feature_list(M_tgt, "dynamic")
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
        col = dyn[:, :, j:j + 1]
        phys = s_t.inverse_transform(col.reshape(-1, 1))
        phys = phys.reshape(col.shape).astype(np.float32)
        dyn2[:, :, j:j + 1] = _transform_with_scaler(
            s_s,
            phys,
        )

    X["dynamic_features"] = dyn2
    return X


def _load_calibrator_near(
    run_dir: str,
    target: float = 0.80,
) -> IntervalCalibrator | None:
    cands = []
    pats = [
        os.path.join(run_dir, "interval_factors_80.npy"),
        os.path.join(
            run_dir,
            "**",
            "interval_factors_80.npy",
        ),
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
        cal = IntervalCalibrator(target=target)
        cal.factors_ = np.load(path).astype(np.float32)
        return cal
    except Exception:
        return None


def _build_geoprior_builder(
    M_src: Dict[str, Any],
    X_sample: Dict[str, Any],
    out_s_dim: int,
    out_g_dim: int,
    horizon: int,
    quantiles: List[float] | None,
    best_hps: Dict[str, Any],
) -> Any:
    cfg = dict(M_src.get("config") or {})

    s_dim = int(X_sample.get("static_features").shape[-1])
    d_dim = int(X_sample.get("dynamic_features").shape[-1])
    f_dim = int(X_sample.get("future_features").shape[-1])

    fixed = dict(
        static_input_dim=s_dim,
        dynamic_input_dim=d_dim,
        future_input_dim=f_dim,
        output_subsidence_dim=int(out_s_dim),
        output_gwl_dim=int(out_g_dim),
        forecast_horizon=int(horizon),
        quantiles=quantiles,
        mode=cfg.get("MODE", "tft_like"),
        pde_mode=cfg.get(
            "PDE_MODE_CONFIG",
            cfg.get("PDE_MODE", "basic"),
        ),
        bounds_mode=cfg.get("BOUNDS_MODE", "soft"),
        residual_method=cfg.get(
            "RESIDUAL_METHOD",
            "autodiff",
        ),
        time_units=cfg.get("TIME_UNITS", "years"),
        scale_pde_residuals=cfg.get(
            "SCALE_PDE_RESIDUALS",
            True,
        ),
        use_effective_h=cfg.get("USE_EFFECTIVE_H", False),
        offset_mode=cfg.get("OFFSET_MODE", "mul"),
        scaling_kwargs=cfg.get("SCALING_KWARGS", None),
    )

    allowed = {
        "embed_dim",
        "hidden_units",
        "lstm_units",
        "num_heads",
        "dropout_rate",
        "memory_size",
        "scales",
        "attention_levels",
        "use_batch_norm",
        "use_residuals",
        "use_vsn",
        "vsn_units",
        "max_window_size",
    }
    hps = {k: v for k, v in best_hps.items() if k in allowed}

    def _builder() -> GeoPriorSubsNet:
        params = dict(fixed)
        params.update(hps)
        return GeoPriorSubsNet(**params)

    return _builder


def _load_source_model(
    M_src: Dict[str, Any],
    X_sample: Dict[str, Any],
    quantiles: List[float] | None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    run_dir = M_src["paths"]["run_dir"]
    best = _best_model_artifact(run_dir)
    if not best:
        raise SystemExit(
            f"No model artifact found under: {run_dir}"
        )

    bundle = _resolve_bundle_paths(best)
    init_p = bundle["init_manifest_path"]
    if os.path.isfile(init_p):
        init_m = _load_json(init_p)
    else:
        init_m = {}

    best_hps = load_best_hps_near_model(
        bundle["keras_path"] or bundle["run_dir"],
    ) or {}

    dims = M_src["artifacts"]["sequences"]["dims"]
    out_s = int(dims["output_subsidence_dim"])
    out_g = int(dims["output_gwl_dim"])
    H = int((M_src.get("config") or {}).get(
        "FORECAST_HORIZON_YEARS",
        1,
    ))

    builder = _build_geoprior_builder(
        M_src=M_src,
        X_sample=X_sample,
        out_s_dim=out_s,
        out_g_dim=out_g,
        horizon=H,
        quantiles=quantiles,
        best_hps=best_hps,
    )

    model = load_inference_model(
        keras_path=bundle["keras_path"],
        weights_path=bundle["weights_path"],
        manifest_path=(
            init_p if os.path.isfile(init_p) else None
        ),
        manifest=init_m if init_m else None,
        builder=builder,
        build_inputs=X_sample,
        out_s_dim=out_s,
        out_g_dim=out_g,
        mode=(
            (M_src.get("config") or {}).get(
                "MODE",
                "tft_like",
            )
        ),
        horizon=H,
        prefer_full_model=False,
    )

    model_pred = model
    if bundle["tf_dir"] is not None:
        try:
            model_pred = load_model_from_tfv2(
                bundle["tf_dir"],
                endpoint="serve",
            )
        except Exception:
            model_pred = model

    return model, model_pred, bundle


def _choose_warm_idx(
    n_total: int,
    n_samples: int,
    frac: float | None,
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


def _slice_npz_dict(
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


def _pinball_loss(
    quantiles: List[float] | None,
) -> Any:
    if not quantiles or len(quantiles) <= 1:
        return tf.keras.losses.MeanSquaredError()

    q = tf.constant(quantiles, dtype=tf.float32)
    q = tf.reshape(q, (1, 1, -1, 1))

    def _loss(y_true, y_pred):
        yt = tf.expand_dims(y_true, axis=2)
        e = yt - y_pred
        return tf.reduce_mean(
            tf.maximum(q * e, (q - 1.0) * e)
        )

    return _loss


def _compile_warm_model(
    model: tf.keras.Model,
    quantiles: List[float] | None,
    lr: float,
) -> List[str]:
    opt = tf.keras.optimizers.Adam(
        learning_rate=float(lr),
    )
    subs_loss = _pinball_loss(quantiles)

    losses = {"subs_pred": subs_loss}
    try:
        model.compile(
            optimizer=opt,
            loss=losses,
        )
        return ["subs_pred"]
    except Exception:
        pass

    losses = {
        "subs_pred": subs_loss,
        "gwl_pred": tf.keras.losses.MSE,
    }
    model.compile(
        optimizer=opt,
        loss=losses,
    )
    return ["subs_pred", "gwl_pred"]


def _make_ds(
    x: Dict[str, Any],
    y: Dict[str, Any],
    batch_size: int,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    n = int(next(iter(x.values())).shape[0])
    ds = ds.shuffle(
        buffer_size=min(n, 50000),
        seed=int(seed),
        reshuffle_each_iteration=True,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def run_one_direction(
    *,
    strategy: str = "xfer",
    rescale_mode: str = "as_is",
    model_pack: Tuple[Any, Any, Dict[str, Any]] | None = None,
    warm_meta: Dict[str, Any] | None = None,
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
    split: str,
    calib_mode: str,
    rescale_to_source: bool,
    batch_size: int,
    quantiles_override: List[float] | None,
    save_dir: str,
    allow_reorder_dynamic: bool = False,
    allow_reorder_future: bool = False,
    log_fn: Optional[Callable[[str], Any]] = None,
) -> Optional[Dict[str, Any]]:
    X_tgt, y_tgt = _pick_npz(M_tgt, split)
    if X_tgt is None:
        return None

    cfg_t = dict(M_tgt.get("config") or {})
    cfg_s = dict(M_src.get("config") or {})

    mode = str(cfg_t.get("MODE", "tft_like"))
    H = int(cfg_t.get("FORECAST_HORIZON_YEARS", 1))
    Q = quantiles_override or cfg_t.get(
        "QUANTILES",
        [0.1, 0.5, 0.9],
    )

    log = log_fn if callable(log_fn) else print

    X_tgt = _ensure_np_inputs(X_tgt, mode, H)
    y_map = map_targets_for_training(y_tgt or {})

    X_tgt = _align_static_to_source(X_tgt, M_src, M_tgt)

    if rescale_to_source:
        X_tgt = _reproject_dynamic_to_source(
            X_tgt,
            M_src,
            M_tgt,
        )

    s_src, d_src, f_src = _infer_input_dims(M_src)

    X_tgt, schema_audit = _check_transfer_schema(
        M_src=M_src,
        M_tgt=M_tgt,
        X_tgt=X_tgt,
        s_src=s_src,
        d_src=d_src,
        f_src=f_src,
        allow_reorder_dynamic=allow_reorder_dynamic,
        allow_reorder_future=allow_reorder_future,
        log_fn=log,
    )


    if rescale_to_source:
        X_tgt = _reproject_dynamic_to_source(
            X_tgt,
            M_src,
            M_tgt,
        )

    if model_pack is None:
        model, model_pred, bundle = _load_source_model(
            M_src=M_src,
            X_sample=X_tgt,
            quantiles=Q,
        )
    else:
        model, model_pred, bundle = model_pack

    model_dir = os.path.dirname(bundle["keras_path"])

    cal = None
    if calib_mode == "source":
        cal = _load_calibrator_near(bundle["run_dir"], 0.80)
    elif calib_mode == "target":
        try:
            vx, vy = _pick_npz(M_tgt, "val")
            if vx is not None:
                vx = _ensure_np_inputs(vx, mode, H)
                vx = _align_static_to_source(vx, M_src, M_tgt)
                vy_m = map_targets_for_training(vy or {})
                ds_v = tf.data.Dataset.from_tensor_slices(
                    (vx, vy_m)
                ).batch(batch_size)
                cal = fit_interval_calibrator_on_val(
                    model,
                    ds_v,
                    target=0.80,
                )
        except Exception:
            cal = None

    pred_dict = model_pred.predict(X_tgt, verbose=0)
    subs_pred, gwl_pred = extract_preds(model, pred_dict)

    if cal is not None and subs_pred.ndim == 4:
        subs_pred = apply_calibrator_to_subs(cal, subs_pred)

    preds = {
        "subs_pred": subs_pred,
        "gwl_pred": gwl_pred,
    }

    enc_s = M_src["artifacts"]["encoders"]
    enc_t = M_tgt["artifacts"]["encoders"]

    coord_scaler, sc_t = _load_scalers(enc_t)
    _, sc_s = _load_scalers(enc_s)

    cols_s = cfg_s.get("cols") or {}
    cols_t = cfg_t.get("cols") or {}

    subs_col_s = cols_s.get("subsidence", "subsidence")
    subs_col_t = cols_t.get("subsidence", "subsidence")

    y_true_phys = None
    if "subs_pred" in y_map and sc_t is not None:
        y_true_phys = inverse_scale_target(
            y_map["subs_pred"][..., :1],
            scaler_info=sc_t,
            target_name=subs_col_t,
        )

    y_pred_point_s = None
    y_pred_phys = None
    if sc_s is not None:
        if subs_pred.ndim == 4:
            q_arr = np.asarray(Q, dtype=np.float32)
            mid = int(np.argmin(np.abs(q_arr - 0.5)))
            y_pred_point_s = subs_pred[:, :, mid, :1]
        else:
            y_pred_point_s = subs_pred[:, :, :1]

        y_pred_phys = inverse_scale_target(
            y_pred_point_s,
            scaler_info=sc_s,
            target_name=subs_col_s,
        )

    metrics_overall = {}
    metrics_h = {}
    if y_true_phys is not None and y_pred_phys is not None:
        metrics_overall = point_metrics(
            y_true_phys,
            y_pred_phys,
            include_r2=True,
        )
        ph = per_horizon_metrics(
            y_true_phys,
            y_pred_phys,
            include_r2=True,
        )
        metrics_h = {
            "mae": ph.get("mae"),
            "r2": ph.get("r2"),
        }

    coverage80 = None
    sharpness80 = None
    if (
        y_true_phys is not None
        and sc_s is not None
        and subs_pred.ndim == 4
    ):
        s_q_phys = inverse_scale_target(
            subs_pred[..., :1],
            scaler_info=sc_s,
            target_name=subs_col_s,
        )
        yt = tf.convert_to_tensor(y_true_phys, tf.float32)
        sq = tf.convert_to_tensor(s_q_phys, tf.float32)
        try:
            coverage80 = float(coverage80_fn(yt, sq).numpy())
            sharpness80 = float(
                sharpness80_fn(yt, sq).numpy()
            )
        except Exception:
            coverage80 = None
            sharpness80 = None

    subs_t_scaler = _get_scaler(sc_t, subs_col_t)
    subs_s_scaler = _get_scaler(sc_s, subs_col_s)

    preds_fmt = dict(preds)
    if (
        subs_s_scaler is not None
        and subs_t_scaler is not None
    ):
        s_phys = inverse_scale_target(
            subs_pred,
            scaler_info=sc_s,
            target_name=subs_col_s,
        )
        preds_fmt["subs_pred"] = _transform_with_scaler(
            subs_t_scaler,
            s_phys,
        )

    y_true_for_format = None
    if y_map:
        y_true_for_format = {}
        if "subs_pred" in y_map:
            y_true_for_format["subsidence"] = (
                y_map["subs_pred"]
            )
        if "gwl_pred" in y_map:
            y_true_for_format["gwl"] = y_map["gwl_pred"]

    train_end = cfg_t.get("TRAIN_END_YEAR")
    f_start = cfg_t.get("FORECAST_START_YEAR")

    grid = None
    if f_start is not None:
        grid = np.arange(
            float(f_start),
            float(f_start) + float(H),
            dtype=float,
        )

    base = (
        f"{M_src.get('city')}_to_{M_tgt.get('city')}_"
        f"{strategy}_{split}_{calib_mode}_{rescale_mode}"
    )
    csv_eval = os.path.join(save_dir, base + "_eval.csv")
    csv_fut = os.path.join(save_dir, base + "_future.csv")

    subs_kind = str(cfg_t.get("SUBSIDENCE_KIND", "rate"))

    df_eval, df_fut = format_and_forecast(
        y_pred=preds_fmt,
        y_true=y_true_for_format,
        coords=X_tgt.get("coords", None),
        quantiles=Q if subs_pred.ndim == 4 else None,
        target_name=subs_col_t,
        scaler_target_name=subs_col_t,
        output_target_name="subsidence",
        target_key_pred="subs_pred",
        component_index=0,
        scaler_info=sc_t,
        coord_scaler=coord_scaler,
        coord_columns=("coord_t", "coord_x", "coord_y"),
        train_end_time=train_end,
        forecast_start_time=f_start,
        forecast_horizon=H,
        future_time_grid=grid,
        dataset_name_for_forecast=(
            f"{strategy}_{split}_{calib_mode}"
        ),
        csv_eval_path=csv_eval,
        csv_future_path=csv_fut,
        eval_metrics=False,
        value_mode=subs_kind,
        input_value_mode=subs_kind,
        output_unit="mm",
        output_unit_from="m",
        output_unit_mode="overwrite",
        output_unit_col="subsidence_unit",
    )

    try:
        model.export_physics_payload(
            X_tgt,
            max_batches=None,
            save_path=os.path.join(
                save_dir,
                base + "_physics_payload.npz",
            ),
            format="npz",
            overwrite=True,
            metadata={
                "city": M_tgt.get("city"),
                "split": split,
            },
        )
    except Exception:
        pass

    return {
        "strategy": strategy,
        "rescale_mode": rescale_mode,
        "warm": warm_meta or {},
        "model_path": bundle["keras_path"],
        "split": split,
        "calibration": calib_mode,
        "quantiles": Q if subs_pred.ndim == 4 else None,
        "coverage80": coverage80,
        "sharpness80": sharpness80,
        "overall_mae": metrics_overall.get("mae"),
        "overall_mse": metrics_overall.get("mse"),
        "overall_r2": metrics_overall.get("r2"),
        "per_horizon_mae": metrics_h.get("mae") or {},
        "per_horizon_r2": metrics_h.get("r2") or {},
        "csv_eval": csv_eval,
        "csv_future": csv_fut,
        "model_dir": model_dir,
        "schema": schema_audit,
    }


def run_warm_start_direction(
    *,
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
    split: str,
    calib_mode: str,
    rescale_to_source: bool,
    batch_size: int,
    quantiles_override: List[float] | None,
    save_dir: str,
    warm_split: str,
    warm_samples: int,
    warm_frac: float | None,
    warm_epochs: int,
    warm_lr: float,
    warm_seed: int,
    rescale_mode: str,
    allow_reorder_dynamic: bool = False,
    allow_reorder_future: bool = False,
    log_fn: Optional[Callable[[str], Any]] = None,
    
) -> Optional[Dict[str, Any]]:
    X_w, y_w = _pick_npz(M_tgt, warm_split)
    if X_w is None or y_w is None:
        return None

    cfg_t = dict(M_tgt.get("config") or {})
    mode = str(cfg_t.get("MODE", "tft_like"))
    H = int(cfg_t.get("FORECAST_HORIZON_YEARS", 1))
    Q = quantiles_override or cfg_t.get(
        "QUANTILES",
        [0.1, 0.5, 0.9],
    )

    X_w = _ensure_np_inputs(X_w, mode, H)
    y_wm = map_targets_for_training(y_w)

    X_w = _align_static_to_source(X_w, M_src, M_tgt)

    s_src, d_src, f_src = _infer_input_dims(M_src)

    X_w,  schema_audit  = _check_transfer_schema(
        M_src=M_src,
        M_tgt=M_tgt,
        X_tgt=X_w,
        s_src=s_src,
        d_src=d_src,
        f_src=f_src,
        allow_reorder_dynamic=allow_reorder_dynamic,
        allow_reorder_future=allow_reorder_future,
        log_fn=log_fn,
    )
    if rescale_to_source:
        X_w = _reproject_dynamic_to_source(
            X_w,
            M_src,
            M_tgt,
        )

    model, model_pred, bundle = _load_source_model(
        M_src=M_src,
        X_sample=X_w,
        quantiles=Q,
    )

    n_total = int(X_w["dynamic_features"].shape[0])
    idx = _choose_warm_idx(
        n_total=n_total,
        n_samples=warm_samples,
        frac=warm_frac,
        seed=warm_seed,
    )

    X_ws = _slice_npz_dict(X_w, idx)
    y_ws = _slice_npz_dict(y_wm, idx)

    warm_keys = _compile_warm_model(
        model,
        quantiles=Q,
        lr=warm_lr,
    )
    y_ws = {k: y_ws[k] for k in warm_keys if k in y_ws}

    ds = _make_ds(
        X_ws,
        y_ws,
        batch_size=batch_size,
        seed=warm_seed,
    )

    model.fit(
        ds,
        epochs=int(warm_epochs),
        verbose=0,
    )

    warm_meta = {
        "warm_split": warm_split,
        "warm_samples": int(len(idx)),
        "warm_frac": warm_frac,
        "warm_epochs": int(warm_epochs),
        "warm_lr": float(warm_lr),
        "warm_seed": int(warm_seed),
        "schema": schema_audit,
    }

    return run_one_direction(
        strategy="warm",
        rescale_mode=rescale_mode,
        model_pack=(model, model, bundle),
        warm_meta=warm_meta,
        M_src=M_src,
        M_tgt=M_tgt,
        split=split,
        calib_mode=calib_mode,
        rescale_to_source=rescale_to_source,
        batch_size=batch_size,
        quantiles_override=quantiles_override,
        save_dir=save_dir,
        log_fn =log_fn 
    )

def main() -> None:
    args = parse_args()
    
    if args.log == "none":
        args.log_fn = lambda *_: None
    else:
        args.log_fn = print

    M_A = _load_manifest_for_city(
        args.city_a,
        args.results_dir,
    )
    M_B = _load_manifest_for_city(
        args.city_b,
        args.results_dir,
    )

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(
        args.results_dir,
        "xfer",
        f"{args.city_a}__{args.city_b}",
        stamp,
    )
    ensure_directory_exists(outdir)

    directions = []
    
    # (tag, src_manifest, tgt_manifest)

    if "baseline" in args.strategies:
        directions.extend(
            [
                ("A_to_A", M_A, M_A),
                ("B_to_B", M_B, M_B),
            ]
        )

    if "xfer" in args.strategies or "warm" in args.strategies:
        directions.extend(
            [
                ("A_to_B", M_A, M_B),
                ("B_to_A", M_B, M_A),
            ]
        )

    results: List[Dict[str, Any]] = []

    for tag, M_src, M_tgt in directions:
        is_baseline = tag in ("A_to_A", "B_to_B")

        for rm in args.rescale_modes:
            strict = rm == "strict"

            for split in args.splits:
                for cm in args.calib_modes:
                    if (
                        is_baseline
                        and "baseline" in args.strategies
                    ):
                        r = run_one_direction(
                            strategy="baseline",
                            rescale_mode=rm,
                            M_src=M_src,
                            M_tgt=M_tgt,
                            split=split,
                            calib_mode=cm,
                            rescale_to_source=strict,
                            batch_size=args.batch_size,
                            quantiles_override=args.quantiles,
                            save_dir=outdir,
                            allow_reorder_dynamic=args.allow_reorder_dynamic,
                            allow_reorder_future=args.allow_reorder_future,
                            log_fn =args.log_fn, 
                        )
                        if r is not None:
                            r["direction"] = tag
                            r["source_city"] = (
                                M_src.get("city")
                            )
                            r["target_city"] = (
                                M_tgt.get("city")
                            )
                            results.append(r)

                    if (
                        (not is_baseline)
                        and "xfer" in args.strategies
                    ):
                        r = run_one_direction(
                            strategy="xfer",
                            rescale_mode=rm,
                            M_src=M_src,
                            M_tgt=M_tgt,
                            split=split,
                            calib_mode=cm,
                            rescale_to_source=strict,
                            batch_size=args.batch_size,
                            quantiles_override=args.quantiles,
                            save_dir=outdir,
                            allow_reorder_dynamic=args.allow_reorder_dynamic,
                            allow_reorder_future=args.allow_reorder_future,
                            log_fn =args.log_fn, 
                        )
                        if r is not None:
                            r["direction"] = tag
                            r["source_city"] = (
                                M_src.get("city")
                            )
                            r["target_city"] = (
                                M_tgt.get("city")
                            )
                            results.append(r)

                    if (
                        (not is_baseline)
                        and "warm" in args.strategies
                    ):
                        r = run_warm_start_direction(
                            M_src=M_src,
                            M_tgt=M_tgt,
                            split=split,
                            calib_mode=cm,
                            rescale_to_source=strict,
                            batch_size=args.batch_size,
                            quantiles_override=args.quantiles,
                            save_dir=outdir,
                            warm_split=args.warm_split,
                            warm_samples=args.warm_samples,
                            warm_frac=args.warm_frac,
                            warm_epochs=args.warm_epochs,
                            warm_lr=args.warm_lr,
                            warm_seed=args.warm_seed,
                            rescale_mode=rm,
                            allow_reorder_dynamic=args.allow_reorder_dynamic,
                            allow_reorder_future=args.allow_reorder_future,
                            log_fn =args.log_fn, 
                        )
                        if r is not None:
                            r["direction"] = tag
                            r["source_city"] = (
                                M_src.get("city")
                            )
                            r["target_city"] = (
                                M_tgt.get("city")
                            )
                            results.append(r)

    js = os.path.join(outdir, "xfer_results.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    import csv

    csv_path = os.path.join(outdir, "xfer_results.csv")

    base_cols = [
        "strategy",
        "rescale_mode",
        "direction",
        "source_city",
        "target_city",
        "split",
        "calibration",
        "overall_mae",
        "overall_mse",
        "overall_r2",
        "coverage80",
        "sharpness80",
        "warm.warm_split",
        "warm.warm_samples",
        "warm.warm_frac",
        "warm.warm_epochs",
        "warm.warm_lr",
        "schema.static_aligned",
        "schema.dynamic_order_mismatch",
        "schema.dynamic_reordered",
        "schema.future_order_mismatch",
        "schema.future_reordered",
        "schema.static_missing_n",
        "schema.static_extra_n",
    ]

    def _sorted_hkeys(keys):
        def _k(k):
            try:
                return int(str(k).strip().split("H")[-1])
            except Exception:
                return 9999

        return sorted(keys, key=_k)

    h_mae_keys = set()
    h_r2_keys = set()
    for r in results:
        h_mae = r.get("per_horizon_mae") or {}
        h_mae_keys |= set(h_mae.keys())
        h_r2 = r.get("per_horizon_r2") or {}
        h_r2_keys |= set(h_r2.keys())

    h_mae_keys = _sorted_hkeys(h_mae_keys)
    h_r2_keys = _sorted_hkeys(h_r2_keys)

    cols = (
        base_cols
        + [f"per_horizon_mae.{k}" for k in h_mae_keys]
        + [f"per_horizon_r2.{k}" for k in h_r2_keys]
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(cols)

        for r in results:
            warm = r.get("warm") or {}
            schema = r.get("schema") or {}
            
            row = [
                r.get("strategy"),
                r.get("rescale_mode"),
                r.get("direction"),
                r.get("source_city"),
                r.get("target_city"),
                r.get("split"),
                r.get("calibration"),
                r.get("overall_mae"),
                r.get("overall_mse"),
                r.get("overall_r2"),
                r.get("coverage80"),
                r.get("sharpness80"),
                warm.get("warm_split"),
                warm.get("warm_samples"),
                warm.get("warm_frac"),
                warm.get("warm_epochs"),
                warm.get("warm_lr"),
                schema.get("static_aligned"),
                schema.get("dynamic_order_mismatch"),
                schema.get("dynamic_reordered"),
                schema.get("future_order_mismatch"),
                schema.get("future_reordered"),
                schema.get("static_missing_n"),
                schema.get("static_extra_n"),
            ]

            ph_mae = r.get("per_horizon_mae") or {}
            ph_r2 = r.get("per_horizon_r2") or {}

            row.extend(
                [ph_mae.get(k, "NA") for k in h_mae_keys]
            )
            row.extend(
                [ph_r2.get(k, "NA") for k in h_r2_keys]
            )

            w.writerow(row)


if __name__ == "__main__":
    main()
