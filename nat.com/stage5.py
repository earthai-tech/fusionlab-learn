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
from typing import Any, Dict, List, Optional, Tuple, Callable, Mapping 
from pathlib import Path
import joblib

import numpy as np
import pandas as pd 
import tensorflow as tf

from fusionlab.utils.generic_utils import (
    ensure_directory_exists,
)
from fusionlab.utils.scale_metrics import (
    inverse_scale_target,
    per_horizon_metrics,
    point_metrics,
    scale_target, 
)
from fusionlab.utils.forecast_utils import (
    format_and_forecast,
)
from fusionlab.utils.nat_utils import (
    ensure_input_shapes,
    # extract_preds,
    map_targets_for_training,
    sanitize_inputs_np,
    load_tuned_hps_near_model, 
    load_trained_hps_near_model, 
    load_hps_auto_near_model, 
)
# from fusionlab.utils.shapes import canonicalize_BHQO
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
from fusionlab.compat.keras_fit import normalize_predict_output

from fusionlab.nn.pinn.geoprior.models import (
    GeoPriorSubsNet,
)
from fusionlab.compat.keras_fit import (
    suppress_compiled_metrics_warning,
)

from fusionlab.deps import with_progress
from fusionlab.utils.generic_utils import vlog
from fusionlab.utils.transfer import xfer_utils as xu 
from fusionlab.utils.transfer import xfer_metrics as xm  
from fusionlab.utils.transfer import xfer_units as xun

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
    p.add_argument(
        "--model-name",
        default="GeoPriorSubsNet",
        help=(
            "Model name token used in filenames, e.g. "
            "'GeoPriorSubsNet' in "
            "<city>_GeoPriorSubsNet_H3_best.keras."
        ),
    )
    p.add_argument(
        "--source-model",
        default="auto",
        choices=["auto", "tuned", "trained"],
        help=(
            "Which Stage-1 source model to use. "
            "auto: prefer tuned if available, else trained."
        ),
    )
    p.add_argument(
        "--source-load",
        default="auto",
        choices=["auto", "full", "weights"],
        help=(
            "How to load the source model. "
            "full: load full .keras, "
            "weights: rebuild + load weights, "
            "auto: try full then fallback to weights."
        ),
    )
    p.add_argument(
        "--hps-mode",
        default="auto",
        choices=["auto", "tuned", "trained"],
        help=(
            "Where to load hyperparameters from when "
            "weights fallback is needed."
        ),
    )
    p.add_argument(
        "--prefer-artifact",
        default="keras",
        choices=["keras", "weights"],
        help=(
            "Which artifact to prefer when searching "
            "for best model under run dirs."
        ),
    )
    p.add_argument(
        "--verbose",
        type=int,
        default=2,
        help=(
            "Verbosity: 0=silent, "
            "1..3=info, 4..5=debug, "
            "6..7=trace."
        ),
    )
    p.add_argument(
        "--progress",
        default="auto",
        choices=["auto", "off", "log"],
        help=(
            "Progress mode: "
            "auto=bar in terminal, "
            "log=redirect tqdm to log_fn, "
            "off=disable."
        ),
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue other jobs even "
            "if one job fails."
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
    p.add_argument(
        "--recompute-missing",
        action="store_true",
        help=(
            "Force recompute metrics from the exported "
            "*_eval.csv via xfer_metrics (coverage/"
            "sharpness + overall/per-horizon point "
            "metrics), overriding existing values. "
            "Useful when scaling/unit/calibration "
            "conventions changed."
        ),
    )
    p.add_argument(
        "--subsidence-unit",
        default="mm",
        choices=["mm", "m"],
        help="Unit for exported CSVs (default: mm).",
    )
    p.add_argument(
        "--subsidence-unit-from",
        default="m",
        choices=["mm", "m"],
        help="Unit of model physical outputs (default: m).",
    )
    p.add_argument(
        "--metrics-unit",
        default=None,
        choices=["mm", "m"],
        help=(
            "Unit for metrics in xfer_results "
            "(default: subsidence-unit)."
        ),
    )
    args = p.parse_args()

    if args.rescale_to_source:
        args.rescale_modes = ["strict"]

    return args

def _make_sink(args):
    if args.log == "none" or int(args.verbose) < 1:
        return (lambda *_a, **_k: None)
    return print


def _mklog(args, sink):
    def _log(msg, *, level=3, depth="auto"):
        vlog(
            str(msg),
            verbose=int(args.verbose),
            level=int(level),
            depth=depth,
            mode="naive",
            vp=False,
            logger=sink,
        )
    return _log


def _keras_verbose(v: int) -> int:
    v = int(v)
    if v >= 6:
        return 2
    if v >= 4:
        return 1
    return 0

def _build_jobs(
    args: argparse.Namespace,
    directions: List[Tuple[str, Any, Any]],
) -> List[Dict[str, Any]]:
    """
    Build Stage-5 jobs.

    Contract per job:
    - tag, src_b, tgt_b
    - split, rm, cm, strict
    - kind: one|warm
    - strategy: baseline|xfer|warm
    """
    jobs: List[Dict[str, Any]] = []

    want_baseline = "baseline" in (args.strategies or [])
    want_xfer = "xfer" in (args.strategies or [])
    want_warm = "warm" in (args.strategies or [])

    for tag, src_b, tgt_b in directions:
        is_base_dir = tag in ("A_to_A", "B_to_B")

        for split in (args.splits or []):
            for cm in (args.calib_modes or []):
                for rm in (args.rescale_modes or []):
                    strict = bool(rm == "strict")

                    if is_base_dir and want_baseline:
                        jobs.append(
                            dict(
                                kind="one",
                                strategy="baseline",
                                tag=tag,
                                src_b=src_b,
                                tgt_b=tgt_b,
                                split=split,
                                cm=cm,
                                rm=rm,
                                strict=strict,
                            )
                        )

                    if (not is_base_dir) and want_xfer:
                        jobs.append(
                            dict(
                                kind="one",
                                strategy="xfer",
                                tag=tag,
                                src_b=src_b,
                                tgt_b=tgt_b,
                                split=split,
                                cm=cm,
                                rm=rm,
                                strict=strict,
                            )
                        )

                    if (not is_base_dir) and want_warm:
                        jobs.append(
                            dict(
                                kind="warm",
                                strategy="warm",
                                tag=tag,
                                src_b=src_b,
                                tgt_b=tgt_b,
                                split=split,
                                cm=cm,
                                rm=rm,
                                strict=strict,
                            )
                        )

    return jobs



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
    weights_path = None
    prefix = None

    if os.path.isdir(mp) and mp.endswith("_best_savedmodel"):
        tf_dir = mp
        prefix = mp[: -len("_best_savedmodel")]
        keras_path = prefix + "_best.keras"
        cand_w = prefix + "_best.weights.h5"
        weights_path = cand_w if os.path.isfile(cand_w) else None

    else:
        base = os.path.basename(mp)

        # --- case A: weights file passed in ---
        if base.endswith("_best.weights.h5"):
            weights_path = mp
            prefix = mp[: -len("_best.weights.h5")]
            cand_k = prefix + "_best.keras"
            keras_path = cand_k if os.path.isfile(cand_k) else None

        elif base.endswith(".weights.h5"):
            weights_path = mp
            prefix = mp[: -len(".weights.h5")]
            cand_k1 = prefix + "_best.keras"
            cand_k2 = prefix + ".keras"
            if os.path.isfile(cand_k1):
                keras_path = cand_k1
            elif os.path.isfile(cand_k2):
                keras_path = cand_k2
            else:
                keras_path = None

        # --- case B: keras file passed in ---
        else:
            keras_path = mp
            if keras_path.endswith("_best.keras"):
                prefix = keras_path[: -len("_best.keras")]
            elif keras_path.endswith(".keras"):
                prefix = keras_path[: -len(".keras")]
            else:
                prefix = os.path.join(run_dir, "model")

            cand_w1 = prefix + "_best.weights.h5"
            cand_w2 = prefix + ".weights.h5"
            if os.path.isfile(cand_w1):
                weights_path = cand_w1
            elif os.path.isfile(cand_w2):
                weights_path = cand_w2
            else:
                weights_path = None

        # savedmodel near prefix
        if prefix:
            cand_tf = prefix + "_best_savedmodel"
            if os.path.isdir(cand_tf):
                tf_dir = cand_tf

    init_path = os.path.join(run_dir, "model_init_manifest.json")

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


def _safe_load(path: str) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        return None


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

    # s_tgt = int(X_tgt["static_features"].shape[-1])
    # d_tgt = int(X_tgt["dynamic_features"].shape[-1])
    # f_tgt = int(X_tgt["future_features"].shape[-1])
    
    s_tgt = int(X_tgt["static_features"].shape[-1])
    d_tgt = int(X_tgt["dynamic_features"].shape[-1])
    
    fk = _future_key(X_tgt)
    if fk is None:
        f_tgt = 0
    else:
        f_tgt = int(np.asarray(X_tgt[fk]).shape[-1])

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

        fk = _future_key(X_tgt)
        if fk is None:
            _raise_schema_error(
                kind="future",
                src_city=src_city,
                tgt_city=tgt_city,
                expected_dim=f_src,
                got_dim=0,
                src_feats=fut_src,
                tgt_feats=[],
            )
        
        X_tgt = dict(X_tgt)
        X_tgt[fk] = _reorder_last_dim(
            X_tgt[fk],
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


def _scaled_ml_numeric_cols(
    M: Mapping[str, Any],
) -> list[str]:
    enc = (M.get("artifacts") or {}).get("encoders") or {}
    cols = enc.get("scaled_ml_numeric_cols") or []
    out: list[str] = []
    for c in cols:
        if isinstance(c, str) and c.strip():
            out.append(c)
    return out


def _index_map(names: list[str]) -> dict[str, int]:
    return {str(n): int(i) for i, n in enumerate(names)}


def _future_key(X: Mapping[str, Any]) -> str | None:
    if "future_known_features" in X:
        return "future_known_features"
    if "future_features" in X:
        return "future_features"
    return None

def _safe_inverse_y(arr, ysi, key, log=None):
    if not isinstance(ysi, Mapping) or not ysi:
        if log:
            log(f"[stage5] NOTE: no y_scaler_info; "
                f"assuming {key} already physical.")
        return np.asarray(arr, np.float32)

    ent = ysi.get(key)
    if not isinstance(ent, Mapping) or not ent:
        if log:
            log(f"[stage5] NOTE: missing y scaler entry "
                f"for {key!r}; identity.")
        return np.asarray(arr, np.float32)

    params = ent.get("params") if isinstance(ent, Mapping) else None
    return inverse_scale_target(
        np.asarray(arr, np.float32),
        scaler_entry=ent,
        target_name=key,
        params=params,
    )


def _can_inverse(ysi: Any, key: str) -> bool:
    if not isinstance(ysi, Mapping) or not ysi:
        return False
    ent = ysi.get(key)
    if not isinstance(ent, Mapping) or not ent:
        return False

    # 1) Params-based scaling is valid
    p = ent.get("params")
    if isinstance(p, Mapping):
        keys = set(p.keys())
        if {"min", "max"} <= keys:
            return True
        if {"mean", "std"} <= keys:
            return True
        if {"scale", "shift"} <= keys:
            return True

    # 2) Scaler-based scaling is valid
    sc = ent.get("scaler", None)
    if sc is not None and hasattr(sc, "inverse_transform"):
        return True

    # 3) scaler_path is also valid (if your hydration loads it later)
    sp = ent.get("scaler_path", None)
    if isinstance(sp, str) and sp.strip():
        return True

    return False



def _safe_scale_y(arr_phys, ysi, key, log=None):
    if not isinstance(ysi, Mapping) or not ysi:
        return np.asarray(arr_phys, np.float32)

    ent = ysi.get(key)
    if not isinstance(ent, Mapping) or not ent:
        return np.asarray(arr_phys, np.float32)

    params = ent.get("params") if isinstance(ent, Mapping) else None
    return scale_target(
        np.asarray(arr_phys, np.float32),
        scaler_entry=ent,
        target_name=key,
        params=params,
    )



def _reproject_dynamic_to_source(
    X_tgt: dict,
    src_b: Any,
    tgt_b: Any,
    *,
    log_fn: Optional[Callable[[str], Any]] = None,
) -> dict:
    """
    Strict transfer:
    target scaled -> physical -> source scaled,
    but ONLY for scaled_ml_numeric_cols.

    Avoids multi-feature scaler shape errors by using
    scaler_entry idx/all_features via inverse_scale_target.
    """
    log = log_fn if callable(log_fn) else (lambda *_: None)

    # M_s = src_b.manifest
    M_t = tgt_b.manifest

    si_s = _bundle_scaler_info(src_b)
    si_t = _bundle_scaler_info(tgt_b)
    
    if not si_s or not si_t:
        return X_tgt

    scaled_cols = _scaled_ml_numeric_cols(M_t)
    if not scaled_cols:
        return X_tgt

    dyn_names = _feature_list(M_t, "dynamic") or []
    fut_names = _feature_list(M_t, "future") or []

    dyn_map = _index_map(dyn_names)
    fut_map = _index_map(fut_names)

    X = dict(X_tgt)

    def _apply_one(
        arr_key: str,
        j: int,
        name: str,
    ) -> None:
        arr = np.asarray(X[arr_key], np.float32)
        arr2 = np.array(arr, copy=True)

        e_t = si_t.get(name)
        e_s = si_s.get(name)

        if not isinstance(e_t, Mapping):
            return
        if not isinstance(e_s, Mapping):
            return

        col = arr[:, :, j : j + 1]
        

        phys = inverse_scale_target(
            col,
            scaler_entry=e_t,
            target_name=name,
        )

        col_s = scale_target(
            phys,
            scaler_entry=e_s,
            target_name=name,
        )

        arr2[:, :, j : j + 1] = col_s.astype(np.float32)
        X[arr_key] = arr2

    touched: list[str] = []

    for name in scaled_cols:
        if name in dyn_map and "dynamic_features" in X:
            _apply_one(
                "dynamic_features",
                dyn_map[name],
                name,
            )
            touched.append(f"dyn:{name}")

        fk = _future_key(X)
        if fk and name in fut_map:
            _apply_one(
                fk,
                fut_map[name],
                name,
            )
            touched.append(f"fut:{name}")

    if touched:
        log("[stage5] strict reproj: " + ", ".join(touched))

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
    manifest: Dict[str, Any] | None = None
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
            cfg.get("PDE_MODE", "both"),
        ),
        bounds_mode=cfg.get("BOUNDS_MODE", "soft"),
        residual_method=cfg.get(
            "RESIDUAL_METHOD",
            "exact",
        ),
        time_units=cfg.get("TIME_UNITS", "year"),
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

    def _builder(_manifest: Dict[str, Any] | None = None) -> GeoPriorSubsNet:
        params = dict(fixed)
        params.update(hps)
        return GeoPriorSubsNet(**params)

    return _builder


def _load_source_model(
    src_b: Any,
    X_sample: Dict[str, Any],
    quantiles: List[float] | None,
    *,
    model_pick: str = "auto",
    load_mode: str = "auto",
    hps_pick: str = "auto",
    model_name: str = "GeoPriorSubsNet",
    prefer_artifact: str = "keras",
    log_fn: Optional[Callable[[str], Any]] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load the "source" model from the source Stage-1 bundle.

    Parameters
    ----------
    src_b:
        Stage-1 bundle for the source city. Must expose:
        - src_b.manifest (dict)
        - src_b.run_dir (str): training run dir
    X_sample:
        One target-like batch used for building inputs if weights
        fallback is required.
    quantiles:
        Quantiles for quantile heads. If None, builder uses cfg.
    model_pick:
        auto|tuned|trained. Where to search first.
    load_mode:
        auto|full|weights. How to load.
    hps_pick:
        auto|tuned|trained. Where to pull hyperparams if needed.
    model_name:
        Token used in filename patterns.
    prefer_artifact:
        keras|weights. Which artifact type to prefer.
    """
    log = log_fn if callable(log_fn) else print

    if model_pick not in ("auto", "tuned", "trained"):
        raise ValueError("model_pick is invalid.")
    if load_mode not in ("auto", "full", "weights"):
        raise ValueError("load_mode is invalid.")
    if hps_pick not in ("auto", "tuned", "trained"):
        raise ValueError("hps_pick is invalid.")
    if prefer_artifact not in ("keras", "weights"):
        raise ValueError("prefer_artifact is invalid.")

    M_src = src_b.manifest

    run_dir = str(getattr(src_b, "run_dir", ""))
    if not run_dir:
        raise ValueError("src_b.run_dir is missing.")

    # stage1_root: parent dir containing training/tuning
    stage1_root = os.path.dirname(run_dir)

    def _best_in_dir(root: str) -> str | None:
        pats: List[str] = []

        if prefer_artifact == "keras":
            pats.append(
                os.path.join(
                    root,
                    "**",
                    f"*_{model_name}_H*_best.keras",
                )
            )
            pats.append(
                os.path.join(root, "**", "*_H*_best.keras")
            )
            pats.append(
                os.path.join(root, "**", "*_best.keras")
            )
            pats.append(
                os.path.join(root, "**", "*.keras")
            )
        else:
            pats.append(
                os.path.join(
                    root,
                    "**",
                    f"*_{model_name}_H*_best.weights.h5",
                )
            )
            pats.append(
                os.path.join(
                    root,
                    "**",
                    "*_H*_best.weights.h5",
                )
            )
            pats.append(
                os.path.join(
                    root,
                    "**",
                    "*_best.weights.h5",
                )
            )

        cands: List[Tuple[float, str]] = []
        for pat in pats:
            for p in glob.glob(pat, recursive=True):
                try:
                    cands.append((os.path.getmtime(p), p))
                except Exception:
                    continue

        if not cands:
            return None

        cands.sort(reverse=True)
        return cands[0][1]

    def _pick_model_path() -> str | None:
        tuned_root = os.path.join(stage1_root, "tuning")

        if model_pick == "trained":
            return _best_in_dir(run_dir)

        if model_pick == "tuned":
            if os.path.isdir(tuned_root):
                return _best_in_dir(tuned_root)
            return None

        # auto: prefer tuned if present, else trained
        if os.path.isdir(tuned_root):
            p = _best_in_dir(tuned_root)
            if p:
                return p

        return _best_in_dir(run_dir)

    best = _pick_model_path()
    if not best:
        raise SystemExit(
            "No model artifact found under: "
            f"{stage1_root}"
        )

    bundle = _resolve_bundle_paths(best)
    init_p = bundle["init_manifest_path"]
    man_path = init_p if os.path.isfile(init_p) else None

    # =================================================
    # Phase A: try full .keras first (no HPS required)
    # =================================================
    model = None
    full_err: Exception | None = None

    if load_mode in ("auto", "full"):
        if bundle.get("keras_path"):
            try:
                model = load_inference_model(
                    keras_path=bundle["keras_path"],
                    weights_path=None,
                    manifest_path=man_path,
                    builder=None,
                    build_inputs=None,
                    prefer_full_model=True,
                    log_fn=log,
                )
            except Exception as e:
                full_err = e
                model = None

    if model is not None and load_mode != "weights":
        model_pred = model

        if bundle.get("tf_dir") is not None:
            try:
                model_pred = load_model_from_tfv2(
                    bundle["tf_dir"],
                    endpoint="serve",
                )
            except Exception:
                model_pred = model

        return model, model_pred, bundle

    # =================================================
    # Phase B: builder + weights fallback (needs HPS)
    # =================================================
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

    if hps_pick == "tuned":
        best_hps = load_tuned_hps_near_model(
            bundle["keras_path"] or bundle["run_dir"],
            prefer="keras",
            required=False,
            log_fn=log,
        )
    elif hps_pick == "trained":
        best_hps = load_trained_hps_near_model(
            bundle["keras_path"] or bundle["run_dir"],
            allowed=allowed,
            required=False,
            log_fn=log,
        )
    else:
        best_hps = load_hps_auto_near_model(
            bundle["keras_path"] or bundle["run_dir"],
            allowed=allowed,
            prefer="keras",
            required=False,
            log_fn=log,
        )

    dims = (
        (M_src.get("artifacts") or {})
        .get("sequences", {})
        .get("dims", {})
        or {}
    )
    out_s = int(dims.get("output_subsidence_dim", 1))
    out_g = int(dims.get("output_gwl_dim", 1))

    cfg_s = dict(M_src.get("config") or {})
    horizon = int(cfg_s.get("FORECAST_HORIZON_YEARS", 1))

    builder = _build_geoprior_builder(
        M_src=M_src,
        X_sample=X_sample,
        out_s_dim=out_s,
        out_g_dim=out_g,
        horizon=horizon,
        quantiles=quantiles,
        best_hps=best_hps or {},
    )

    if not bundle.get("weights_path"):
        raise SystemExit(
            "Weights fallback requested but no "
            "weights file was found."
        )

    try:
        model = load_inference_model(
            keras_path=None,
            weights_path=bundle["weights_path"],
            manifest_path=man_path,
            builder=builder,
            build_inputs=X_sample,
            prefer_full_model=False,
            log_fn=log,
        )
    except Exception as e:
        msg: List[str] = ["Weights fallback failed."]
        if full_err is not None:
            msg.append(f"Full load error: {full_err!r}")
        msg.append(f"Weights error: {e!r}")
        raise SystemExit("\n".join(msg))

    model_pred = model
    if bundle.get("tf_dir") is not None:
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



def _bundle_scaler_info(b) -> dict:
    """
    Return hydrated scaler_info dict for bundle b.

    Supports:
    - dict directly in manifest/config
    - joblib path in manifest/artifacts/encoders
    - scaler_path entries inside dict
    """
    M = b.manifest
    cfg = M.get("config", {}) or {}
    art = M.get("artifacts", {}) or {}
    enc = art.get("encoders", {}) or {}

    si = cfg.get("scaler_info", None)
    if isinstance(si, dict) and si:
        return _hydrate_si_dict(b, si)

    # common: enc["scaler_info"] is a joblib path
    si = enc.get("scaler_info", None)
    if isinstance(si, (str, Path)):
        p = xu.resolve_artifact_path(b.run_dir, si, strict=False)
        if p and p.exists():
            try:
                si = joblib.load(str(p))
            except Exception:
                si = {}
        else:
            si = {}

    if isinstance(si, dict) and si:
        return _hydrate_si_dict(b, si)

    # fallback: scaling_audit may embed it
    aud = b.scaling_audit or {}
    sc = aud.get("scalers", {}) or {}
    si = sc.get("scaler_info", None)
    if isinstance(si, dict) and si:
        return _hydrate_si_dict(b, si)

    return {}

def _bundle_y_scaler_info(b) -> dict:
    M = b.manifest
    cfg = M.get("config", {}) or {}
    art = M.get("artifacts", {}) or {}
    enc = art.get("encoders", {}) or {}

    ysi = (
        enc.get("target_scaler_info")
        or enc.get("y_scaler_info")
        or (art.get("targets", {}) or {}).get("scaler_info")
        or cfg.get("target_scaler_info")
        or cfg.get("y_scaler_info")
    )

    if isinstance(ysi, dict) and ysi:
        return _hydrate_si_dict(b, ysi)

    if isinstance(ysi, (str, Path)):
        p = xu.resolve_artifact_path(b.run_dir, ysi, strict=False)
        if p and p.exists():
            try:
                ysi = joblib.load(str(p))
            except Exception:
                ysi = {}
        else:
            ysi = {}

    if isinstance(ysi, dict) and ysi:
        return _hydrate_si_dict(b, ysi)

    return {}

def _hydrate_si_dict(b, si: dict) -> dict:
    """
    Ensure each entry has v["scaler"] loaded from v["scaler_path"].
    Uses resolve_artifact_path for portability.
    """
    if not isinstance(si, dict):
        return {}

    out = dict(si)
    for _k, v in out.items():
        if not isinstance(v, dict):
            continue
        if v.get("scaler") is not None:
            continue
        sp = v.get("scaler_path", None)
        if not sp:
            continue
        p = xu.resolve_artifact_path(b.run_dir, sp, strict=False)
        if p and p.exists():
            try:
                v["scaler"] = joblib.load(str(p))
            except Exception:
                pass
    return out


def _pick_si_key(si: dict, preferred: str) -> str:
    """
    Pick a key inside scaler_info (best-effort).
    """
    if preferred in si:
        return preferred

    low = {str(k).lower(): str(k) for k in si.keys()}
    pref_l = str(preferred).lower()

    if pref_l in low:
        return low[pref_l]

    for token in ("subs", "subsidence", "gwl", "head"):
        for lk, orig in low.items():
            if token in lk:
                return orig

    return preferred

# def _apply_sklearn_1d(arr, scaler, inverse=False):
#     a = np.asarray(arr)
#     shp = a.shape
#     x = a.reshape(-1, 1)

#     if inverse:
#         y = scaler.inverse_transform(x)
#     else:
#         y = scaler.transform(x)

#     return y.reshape(shp)


# def _reproject(arr, from_info, from_key, to_info, to_key):
#     """
#     Convert arr from "from_key" scaling space
#     into "to_key" scaling space using existing
#     Stage-1 scalers (no fitting).

#     Works for (N,H,1) and (N,H,Q,1) too.
#     """
#     raw = inverse_scale_target(
#         arr,
#         scaler_info=from_info,
#         target_name=from_key,
#     )

#     to_scaler = (to_info.get(to_key) or {}).get("scaler")
#     if to_scaler is None:
#         raise KeyError(f"Missing scaler for {to_key!r}.")

#     return _apply_sklearn_1d(raw, to_scaler, inverse=False)


def _pinball_loss(quantiles):
    if not quantiles or len(quantiles) <= 1:
        return tf.keras.losses.MeanSquaredError()

    q = tf.reshape(tf.constant(quantiles, tf.float32), (1, 1, -1))  # (1,1,Q)

    def _loss(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)

        # Normalize y_pred to (B,H,Q)
        if y_pred.shape.rank == 4 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # (B,H,Q)

        # Normalize y_true to (B,H)
        if y_true.shape.rank == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)  # (B,H,1)
        if y_true.shape.rank == 3 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)  # (B,H)

        # Broadcast y_true across quantiles -> (B,H,1)
        yt = y_true[..., None]
        e = yt - y_pred                          # (B,H,Q)

        return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

    return _loss

def _compile_warm_model(
    model: tf.keras.Model,
    quantiles: List[float] | None,
    lr: float,
) -> List[str]:
    
    from fusionlab.compat.keras import (
        ensure_loss_dict,
        zero_loss,
    )

    opt = tf.keras.optimizers.Adam(
        learning_rate=float(lr),
    )
    subs_loss = _pinball_loss(quantiles)

    # Warm-start commonly provides only subs targets.
    # Make the model tolerate missing head targets.
    sk = getattr(model, "scaling_kwargs", None) or {}
    if not bool(sk.get("allow_missing_targets", False)):
        sk2 = dict(sk)
        sk2["allow_missing_targets"] = True
        setattr(model, "scaling_kwargs", sk2)

    losses = {"subs_pred": subs_loss}

    # Fill any missing output heads with a zero loss.
    out_names = getattr(model, "output_names", None) or []
    if out_names:
        losses = ensure_loss_dict(
            losses,
            output_names=list(out_names),
            fill=zero_loss,
        )
    else:
        # Fallback for unusual builds
        losses["gwl_pred"] = zero_loss

    model.compile(
        optimizer=opt,
        loss=losses,
    )

    # Only subs is truly supervised in warm-start.
    return ["subs_pred"]


def _has_q(
    qs: List[float],
    q: float,
    tol: float = 1e-6,
) -> bool:
    if not qs:
        return False
    a = np.asarray(qs, dtype=float)
    return float(np.min(np.abs(a - float(q)))) <= float(tol)

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

def _fix_horizon_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict) or not d:
        return d
    ks = []
    for k in d.keys():
        s = str(k)
        if s.startswith("H") and s[1:].isdigit():
            ks.append(int(s[1:]))
    if not ks:
        return d

    ks = sorted(set(ks))
    # detect the common off-by-one pattern: starts at 2 and is consecutive
    if ks[0] == 2 and ks == list(range(2, 2 + len(ks))):
        out: Dict[str, Any] = {}
        for k, v in d.items():
            s = str(k)
            if s.startswith("H") and s[1:].isdigit():
                i = int(s[1:]) - 1
                out[f"H{i}"] = v
            else:
                out[s] = v
        return out

    return d

def run_one_direction(
    *,
    strategy: str = "xfer",
    rescale_mode: str = "as_is",
    model_pack: Tuple[Any, Any, Dict[str, Any]] | None = None,
    warm_meta: Dict[str, Any] | None = None,
    src_b: Any,
    tgt_b: Any,
    split: str,
    calib_mode: str,
    rescale_to_source: bool,
    batch_size: int,
    quantiles_override: List[float] | None,
    save_dir: str,
    model_pick: str = "auto",
    load_mode: str = "auto",
    hps_pick: str = "auto",
    model_name: str = "GeoPriorSubsNet",
    prefer_artifact: str = "keras",
    allow_reorder_dynamic: bool = False,
    allow_reorder_future: bool = False,
    recompute_missing: bool = False,
    subs_unit: str = "mm",
    subs_unit_from: str = "m",
    metrics_unit: str | None = None,
    log_fn: Optional[Callable[[str], Any]] = None,
    verbose: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Execute one evaluation job for a single direction.

    Pipeline:
    1) Load target split arrays (Stage-1 NPZ).
    2) Normalize inputs for model inference.
    3) Align target static to source schema.
    4) Optionally reproject target dynamics to source scaling.
    5) Validate schema compatibility (with optional reorders).
    6) Load source model (or reuse provided model_pack).
    7) Optional interval calibration for quantile outputs.
    8) Predict, compute metrics in physical units.
    9) Write formatted eval + forecast CSVs.
    """
    log = log_fn if callable(log_fn) else print

    # ------------------------------
    # Stage-1 manifests + split data
    # ------------------------------
    M_src = src_b.manifest
    M_tgt = tgt_b.manifest

    X_tgt, y_tgt = _pick_npz(M_tgt, split)
    sf = X_tgt.get("static_features", None)
    if sf is None:
        log("[stage5][debug] static_features: MISSING")
    else:
        nz = int(np.count_nonzero(sf))
        log(
            f"[stage5][debug] static_features "
            f"shape={sf.shape} nonzero={nz}"
        )

    if X_tgt is None:
        return None

    cfg_t = dict(M_tgt.get("config") or {})
    mode = str(cfg_t.get("MODE", "tft_like"))
    H = int(cfg_t.get("FORECAST_HORIZON_YEARS", 1))

    Q = (
        quantiles_override
        or cfg_t.get("QUANTILES", [0.1, 0.5, 0.9])
    )

    # ------------------------------
    # Prepare inputs + targets
    # ------------------------------
    X_tgt = _ensure_np_inputs(X_tgt, mode, H)
    y_map = map_targets_for_training(y_tgt or {})

    # Align static: target -> source schema by name.
    X_tgt = _align_static_to_source(X_tgt, M_src, M_tgt)

    # Strict: move target dynamic scaling into source space.
    # baseline never needs strict reprojection
    if str(strategy) == "baseline":
        rescale_to_source = False
        rescale_mode = "as_is"

    if rescale_to_source:
        X_tgt = _reproject_dynamic_to_source(
            X_tgt,
            src_b,
            tgt_b,
        )

    # Validate transfer schema (dynamic/future must match).
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

    # ------------------------------
    # Load or reuse the source model
    # ------------------------------
    if model_pack is None:
        model, model_pred, bundle = _load_source_model(
            # M_src=M_src,
            src_b=src_b, 
            X_sample=X_tgt,
            quantiles=Q,
            model_pick=model_pick,
            load_mode=load_mode,
            hps_pick=hps_pick,
            model_name=model_name,
            prefer_artifact=prefer_artifact,
            log_fn=log,
        )
    else:
        model, model_pred, bundle = model_pack

    model_path = bundle.get("keras_path") or ""
    model_dir = (
        os.path.dirname(model_path)
        if model_path
        else str(bundle.get("run_dir") or "")
    )

    # ------------------------------
    # Optional calibration (quantile)
    # ------------------------------
    cal = None

    if calib_mode == "source":
        cal = _load_calibrator_near(
            str(bundle.get("run_dir") or ""),
            target=0.80,
        )

    elif calib_mode == "target":
        try:
            vx, vy = _pick_npz(M_tgt, "val")
            if vx is not None and vy is not None:
                vx = _ensure_np_inputs(vx, mode, H)
                vx = _align_static_to_source(
                    vx,
                    M_src,
                    M_tgt,
                )

                if rescale_to_source:
                    vx = _reproject_dynamic_to_source(
                        vx,
                        src_b,
                        tgt_b,
                    )

                vx, _ = _check_transfer_schema(
                    M_src=M_src,
                    M_tgt=M_tgt,
                    X_tgt=vx,
                    s_src=s_src,
                    d_src=d_src,
                    f_src=f_src,
                    allow_reorder_dynamic=allow_reorder_dynamic,
                    allow_reorder_future=allow_reorder_future,
                    log_fn=log,
                )

                vy_map = map_targets_for_training(vy)
                ds_v = tf.data.Dataset.from_tensor_slices(
                    (vx, vy_map)
                ).batch(int(batch_size))

                cal = fit_interval_calibrator_on_val(
                    model,
                    ds_v,
                    target=0.80,
                )
        except Exception:
            cal = None

    # ------------------------------
    # Predict
    # ------------------------------

    pred_out = model_pred.predict(
        X_tgt,
        verbose=int(verbose),
    )
    
    pred_dict = normalize_predict_output(
        model_pred,
        x=X_tgt,
        pred_out=pred_out,
        required=("subs_pred", "gwl_pred"),
        batch_n=32,
        log_fn=log,
    )

    if "subs_pred" not in pred_dict or "gwl_pred" not in pred_dict:
        raise KeyError(
            "predict() must return 'subs_pred' and "
            "'gwl_pred'. Got keys="
            f"{list(pred_dict.keys())}"
        )
        
    subs_pred = pred_dict["subs_pred"]
    gwl_pred = pred_dict["gwl_pred"]
    
    # subs_pred, gwl_pred = extract_preds(
    #     model,
    #     pred_dict,
    # )
    
    # =================================================
    # Canonicalize quantile layout to BHQO (robust).
    #
    # IMPORTANT (transfer case):
    #   model outputs are in SOURCE scaling space;
    #   y_true from target bundle is in TARGET scaling.
    #   For MAE tie-break (H==Q), compare against
    #   y_true projected into SOURCE scaling.
    # =================================================
    
    # (1) prepare scalers + keys (must exist BEFORE canonicalize)
    y_si_s = _bundle_y_scaler_info(src_b)
    y_si_t = _bundle_y_scaler_info(tgt_b)
    
    subs_pref_s = str(src_b.target_cols.get("subs", "subs"))
    subs_pref_t = str(tgt_b.target_cols.get("subs", "subs"))
    
    subs_key_s = _pick_si_key(y_si_s, subs_pref_s) if y_si_s else subs_pref_s
    subs_key_t = _pick_si_key(y_si_t, subs_pref_t) if y_si_t else subs_pref_t
    
    # (2) run canonicalize
    subs_pred_raw = subs_pred
    
    y_true_src = None
    if y_map and ("subs_pred" in y_map):
        y_true_tgt = np.asarray(
            y_map["subs_pred"][..., :1],
            np.float32,
        )
    
        if y_si_s and y_si_t:
            y_phys = _safe_inverse_y(
                y_true_tgt,
                y_si_t,
                subs_key_t,
                log=log,
            )
            y_true_src = _safe_scale_y(
                y_phys,
                y_si_s,
                subs_key_s,
                log=log,
            )
        else:
            y_true_src = y_true_tgt
    
    if (y_true_src is not None) and (subs_pred_raw is not None):
        sp = tf.convert_to_tensor(subs_pred_raw)
        yt = tf.convert_to_tensor(y_true_src)
    
        subs_pred_raw = sp.numpy()
    
    elif (
        subs_pred_raw is not None
        and int(np.asarray(subs_pred_raw).ndim) == 4
    ):
        # No y_true available -> cannot disambiguate.
        # Still enforce monotone quantiles.
        subs_pred_raw = np.sort(
            np.asarray(subs_pred_raw),
            axis=2,
        )
    
    subs_pred = subs_pred_raw  # <-- IMPORTANT: feed canonicalized back
    
    # (3) now calibration is safe (expects Q axis)
    if cal is not None and int(np.asarray(subs_pred).ndim) == 4:
        subs_pred = apply_calibrator_to_subs(
            cal,
            subs_pred,
            q_values=Q,
        )
    
    preds = {
        "subs_pred": subs_pred,
        "gwl_pred": gwl_pred,
    }

    # ------------------------------
    # Metrics in physical units
    # ------------------------------
    
    y_si_s = _bundle_y_scaler_info(src_b)
    y_si_t = _bundle_y_scaler_info(tgt_b)

    subs_pref_s = str(src_b.target_cols.get("subs", "subs"))
    subs_pref_t = str(tgt_b.target_cols.get("subs", "subs"))
    
    subs_key_s = _pick_si_key(y_si_s, subs_pref_s) if y_si_s else subs_pref_s
    subs_key_t = _pick_si_key(y_si_t, subs_pref_t) if y_si_t else subs_pref_t
    
    ent_s = (y_si_s or {}).get(subs_key_s) or {}
    ent_t = (y_si_t or {}).get(subs_key_t) or {}
    log(f"[stage5][debug] subs_key_s={subs_key_s!r} "
        f"idx={ent_s.get('idx')} has_scaler={ent_s.get('scaler') is not None}")
    log(f"[stage5][debug] subs_key_t={subs_key_t!r} "
        f"idx={ent_t.get('idx')} has_scaler={ent_t.get('scaler') is not None}")
    
    log(f"[stage5][debug] ent_s keys={list(ent_s.keys())[:10]}")
    log(f"[stage5][debug] ent_t keys={list(ent_t.keys())[:10]}")
    log(f"[stage5][debug] ent_s params={ent_s.get('params')}")
    log(f"[stage5][debug] ent_t params={ent_t.get('params')}")

    # Build comparable arrays (scaled space first)
    # --------
    y_true_scaled = None
    
    if "subs_pred" in y_map:
        y_true_scaled = np.asarray(y_map["subs_pred"][..., :1], np.float32)
    
    if int(subs_pred.ndim) == 4:
        q_arr = np.asarray(Q, dtype=np.float32)
        mid = int(np.argmin(np.abs(q_arr - 0.5)))
        y_pred_scaled = np.asarray(subs_pred[:, :, mid, :1], np.float32)
    else:
        y_pred_scaled = np.asarray(subs_pred[:, :, :1], np.float32)
    
    metrics_overall: Dict[str, Any] = {}
    metrics_h: Dict[str, Any] = {}
    
    # --------
    # Decide space: physical only if BOTH sides can be inverse-scaled
    # --------
    y_true_phys = None
    y_pred_phys = None
    metrics_space = "scaled"

    can_true = _can_inverse(y_si_t, subs_key_t)
    can_pred = _can_inverse(y_si_s, subs_key_s)
    
    if y_true_scaled is not None:
        if can_true and can_pred:
            y_true_phys = _safe_inverse_y(
                y_true_scaled,
                y_si_t,
                subs_key_t,
                log=log,
            )
            y_pred_phys = _safe_inverse_y(
                y_pred_scaled,
                y_si_s,
                subs_key_s,
                log=log,
            )
            log("[stage5] metrics_space=physical")
            yA, yB = y_true_phys, y_pred_phys
        else:
            log(
                "[stage5] metrics_space=scaled "
                "(missing/invalid y scaler for true or pred)"
            )
            yA, yB = y_true_scaled, y_pred_scaled
    
        metrics_overall = point_metrics(yA, yB, use_physical=False)
        
        # Backward compatibility: older point_metrics may omit rmse.
        try:
            if (
                isinstance(metrics_overall, dict)
                and ("rmse" not in metrics_overall)
                and (metrics_overall.get("mse") is not None)
            ):
                metrics_overall["rmse"] = float(
                    np.sqrt(float(metrics_overall["mse"]))
                )
        except Exception:
            pass

        mae_h, r2_h = per_horizon_metrics(yA, yB, use_physical=False)
        mae_h = _fix_horizon_keys(mae_h)
        r2_h = _fix_horizon_keys(r2_h)

        metrics_h = {"mae": mae_h, "r2": r2_h}
        
        metrics_space = "physical"
        log("[stage5] metrics_space=physical")

    
        v = float(np.var(yA))
        log(f"[stage5][debug] var(y_eval)={v:.6g}")
    

    # Coverage/sharpness for quantile outputs (physical).
    coverage80 = None
    sharpness80 = None


    if (
        metrics_space == "physical"
        and y_true_phys is not None
        and y_pred_phys is not None
        and int(subs_pred.ndim) == 4
    ):
        try:
            s_q_phys = _safe_inverse_y(
                subs_pred[..., :1],
                y_si_s,
                subs_key_s,
                log=log,
            )
    
            yt = tf.convert_to_tensor(y_true_phys, tf.float32)
            sq = tf.convert_to_tensor(s_q_phys, tf.float32)
    
            coverage80 = float(coverage80_fn(yt, sq).numpy())
            sharpness80 = float(sharpness80_fn(yt, sq).numpy())
        except Exception:
            coverage80 = None
            sharpness80 = None


    # ------------------------------
    # Formatting + CSV outputs
    # ------------------------------
    # Reproject subs predictions into *target* scaling so
    # format_and_forecast can inverse-scale with target scalers.
    preds_fmt = dict(preds)
    
    if (
        y_si_s
        and y_si_t
        and (subs_key_s in y_si_s)
        and (subs_key_t in y_si_t)
    ):
        s_phys = _safe_inverse_y(
            subs_pred,
            y_si_s,
            subs_key_s,
            log=log,
        )
        preds_fmt["subs_pred"] = _safe_scale_y(
            s_phys,
            y_si_t,
            subs_key_t,
            log=log,
        )

    y_true_for_fmt = None
    if y_map:
        y_true_for_fmt = {}
        if "subs_pred" in y_map:
            y_true_for_fmt["subsidence"] = y_map["subs_pred"]
        if "gwl_pred" in y_map:
            y_true_for_fmt["gwl"] = y_map["gwl_pred"]

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
        f"{M_src.get('city')}_to_"
        f"{M_tgt.get('city')}_"
        f"{strategy}_{split}_"
        f"{calib_mode}_{rescale_mode}"
    )
    csv_eval = os.path.join(save_dir, base + "_eval.csv")
    csv_fut = os.path.join(save_dir, base + "_future.csv")

    subs_kind = str(cfg_t.get("SUBSIDENCE_KIND", "rate"))

    coord_scaler = getattr(tgt_b, "coord_scaler", None)
    if coord_scaler is None:
        scs = getattr(tgt_b, "scalers", None) or {}
        coord_scaler = scs.get("coord_scaler", None)

    _q = Q if int(subs_pred.ndim) == 4 else None

    ff_scaler_info = (
        y_si_t
        if (isinstance(y_si_t, Mapping) and y_si_t and subs_key_t in y_si_t)
        else None
    )
    ff_scaler_name = subs_key_t if ff_scaler_info else None

    df_eval, df_fut = format_and_forecast(
        y_pred=preds_fmt,
        y_true=y_true_for_fmt,
        coords=X_tgt.get("coords", None),
        quantiles=_q,
        target_name=subs_pref_t,
        scaler_target_name= ff_scaler_name, # subs_key_t,
        output_target_name="subsidence",
        target_key_pred="subs_pred",
        component_index=0,
        scaler_info=ff_scaler_info, #si_t,
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
        output_unit=subs_unit,
        output_unit_from=subs_unit_from,
        output_unit_mode="overwrite",
        output_unit_col="subsidence_unit",
    )
    
  
    def _unit_scale_from_eval(df: pd.DataFrame) -> float:
        """
        Return factor to convert eval CSV numbers into meters.
    
        Your eval CSV is usually exported as mm.
        We keep xfer_results metrics in meters (as your current
        overall_mae values show ~0.01–0.03).
        """
        if df is None or "subsidence_unit" not in df.columns:
            return 1.0
        s = df["subsidence_unit"].dropna()
        if s.empty:
            return 1.0
        u0 = str(s.astype(str).iloc[0]).strip().lower()
        if u0.startswith("mm"):
            return 1.0 / 1000.0
        return 1.0
    

    def _eval_metrics_from_df(
        df_eval: pd.DataFrame,
        ) -> tuple[
        xm.EvalSummary | None,
        dict,
        dict,
        dict,
        dict,
        str,
    ]:
        if df_eval is None:
            return None, {}, {}, {}, {}, "mm"
    
        need = ["subsidence_actual", "subsidence_q50"]
        if any(c not in df_eval.columns for c in need):
            return None, {}, {}, {}, {}, "mm"
    
        eval_u = xun.infer_unit(df_eval, default=subs_unit)
        mu = str(metrics_unit or subs_unit).lower()
        fac = xun.unit_factor(eval_u, mu)
    
        cols = [
            "subsidence_actual",
            "subsidence_q10",
            "subsidence_q50",
            "subsidence_q90",
        ]
        dfm = xun.scale_cols(df_eval, cols, fac)
    
        summary = xm.summarize_eval_df(dfm)
    
        # ph_mae: dict = {}
        # ph_r2: dict = {}
        ph_mae: dict = {}
        ph_mse: dict = {}
        ph_rmse: dict = {}
        ph_r2: dict = {}

        if "forecast_step" in dfm.columns:
            steps = pd.to_numeric(dfm["forecast_step"], errors="coerce")
            steps = steps[np.isfinite(steps)].to_numpy(float)
        
            uniq = np.unique(steps)
            uniq.sort()
        
            step_to_h = {
                float(s): f"H{i + 1}"
                for i, s in enumerate(uniq)
            }
        
            for step, g in dfm.groupby("forecast_step", dropna=False):
                try:
                    s = float(step)
                except Exception:
                    continue
        
                h = step_to_h.get(s)
                if h is None:
                    continue
        
                yy = pd.to_numeric(
                    g["subsidence_actual"], errors="coerce"
                ).to_numpy(float)
                pp = pd.to_numeric(
                    g["subsidence_q50"], errors="coerce"
                ).to_numpy(float)
        
                ph_mae[h] = float(xm.mae(yy, pp))
                ph_mse[h] = float(xm.mse(yy, pp))
                ph_rmse[h] = float(xm.rmse(yy, pp))
                ph_r2[h] = float(xm.r2_score(yy, pp))
                
        return summary, ph_mae, ph_mse, ph_rmse, ph_r2, mu
    
    force = bool(recompute_missing)
    
    need_any = (
        force
        or (coverage80 is None)
        or (sharpness80 is None)
        or (metrics_overall.get("mae") is None)
        or (metrics_overall.get("mse") is None)
        or (metrics_overall.get("rmse") is None)
        or (metrics_overall.get("r2") is None)
        or not bool(metrics_h.get("mae"))
        or not bool(metrics_h.get("mse"))
        or not bool(metrics_h.get("rmse"))
        or not bool(metrics_h.get("r2"))
    )
    mu = "" 
    if need_any:
        try:
            summ, ph_mae2, ph_mse2, ph_rmse2, ph_r2_2, mu = (
                _eval_metrics_from_df(df_eval)
            )
            if summ is not None:
                coverage80 = float(summ.coverage80)
                sharpness80 = float(summ.sharpness80)
                metrics_overall["mae"] = float(summ.mae)
                metrics_overall["mse"] = float(summ.mse)
                metrics_overall["rmse"] = float(summ.rmse)
                metrics_overall["r2"] = float(summ.r2)
                if ph_mae2:
                    metrics_h["mae"] = ph_mae2
                if ph_mse2:
                    metrics_h["mse"] = ph_mse2
                if ph_rmse2:
                    metrics_h["rmse"] = ph_rmse2
                if ph_r2_2:
                    metrics_h["r2"] = ph_r2_2
            else:
                mu = str(metrics_unit or subs_unit).lower()
    
                log(
                    "[stage5] metrics recomputed from eval CSV "
                    f"(force={force})"
                )
        except Exception as e:
            log(
                "[stage5] WARN: recompute from eval CSV "
                f"failed: {e!r}"
            )
    # ------------------------------
    # Optional physics payload export
    # ------------------------------
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

    def _stat(a, name):
        a = np.asarray(a)
        log(f"[stage5][debug] {name}: "
            f"min={a.min():.6g} max={a.max():.6g} "
            f"mean={a.mean():.6g} std={a.std():.6g}")
        

    if y_true_scaled is not None:
        _stat(y_true_scaled, "y_true_scaled")
        _stat(y_pred_scaled, "y_pred_scaled")

    ph_mae = _fix_horizon_keys(metrics_h.get("mae") or {})
    ph_mse = _fix_horizon_keys(metrics_h.get("mse") or {})
    ph_rmse = _fix_horizon_keys(metrics_h.get("rmse") or {})
    ph_r2 = _fix_horizon_keys(metrics_h.get("r2") or {})

    return {
        "strategy": strategy,
        "rescale_mode": rescale_mode,
        "warm": warm_meta or {},
        "model_path": model_path,
        "split": split,
        "calibration": calib_mode,
        "quantiles": _q,
        "coverage80": coverage80,
        "sharpness80": sharpness80,
        "overall_mae": metrics_overall.get("mae"),
        "overall_mse": metrics_overall.get("mse"),
        "overall_rmse": metrics_overall.get("rmse"),
        "overall_r2": metrics_overall.get("r2"),
        "per_horizon_mae": ph_mae,
        "per_horizon_mse": ph_mse,
        "per_horizon_rmse": ph_rmse,
        "per_horizon_r2": ph_r2,
        "csv_eval": csv_eval,
        "csv_future": csv_fut,
        "model_dir": model_dir,
        "schema": schema_audit,
        "source_model": model_pick,
        "source_load": load_mode,
        "hps_mode": hps_pick,
        "model_name": model_name,
        "prefer_artifact": prefer_artifact,
        "metrics_source": "eval_csv" if bool(recompute_missing) else "mixed",
        "subsidence_unit": mu,
        "metrics_unit": (metrics_unit or mu),
    }

def run_warm_start_direction(
    *,
    src_b: Any,
    tgt_b: Any,
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
    model_pick: str = "auto",
    load_mode: str = "auto",
    hps_pick: str = "auto",
    model_name: str = "GeoPriorSubsNet",
    prefer_artifact: str = "keras",
    allow_reorder_dynamic: bool = False,
    allow_reorder_future: bool = False,
    recompute_missing: bool = False,
    log_fn: Optional[Callable[[str], Any]] = None,
    verbose: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Warm-start the source model on a subset of target data,
    then evaluate on the requested split via run_one_direction.

    Steps:
    1) Load warm_split arrays from target bundle.
    2) Prepare inputs, align schemas, optional strict rescale.
    3) Load the source model (or weights fallback).
    4) Sample a subset (warm_samples / warm_frac).
    5) Compile a tolerant warm-start loss config.
    6) Fit a few epochs on target subset.
    7) Call run_one_direction with model_pack reuse.
    """
    log = log_fn if callable(log_fn) else print

    M_src = src_b.manifest
    M_tgt = tgt_b.manifest

    # ------------------------------
    # Load warm-start split from tgt
    # ------------------------------
    X_w, y_w = _pick_npz(M_tgt, warm_split)
    if X_w is None or y_w is None:
        return None

    cfg_t = dict(M_tgt.get("config") or {})
    mode = str(cfg_t.get("MODE", "tft_like"))
    H = int(cfg_t.get("FORECAST_HORIZON_YEARS", 1))

    Q = (
        quantiles_override
        or cfg_t.get("QUANTILES", [0.1, 0.5, 0.9])
    )

    # ------------------------------
    # Prepare inputs and targets
    # ------------------------------
    X_w = _ensure_np_inputs(X_w, mode, H)
    y_wm = map_targets_for_training(y_w)

    X_w = _align_static_to_source(X_w, M_src, M_tgt)

    s_src, d_src, f_src = _infer_input_dims(M_src)

    X_w, schema_audit = _check_transfer_schema(
        M_src=M_src,
        M_tgt=M_tgt,
        X_tgt=X_w,
        s_src=s_src,
        d_src=d_src,
        f_src=f_src,
        allow_reorder_dynamic=allow_reorder_dynamic,
        allow_reorder_future=allow_reorder_future,
        log_fn=log,
    )

    if rescale_to_source:
        X_w = _reproject_dynamic_to_source(
            X_w,
            src_b,
            tgt_b,
        )

    # ------------------------------
    # Load the source model
    # ------------------------------
    model, _pred, bundle = _load_source_model(
        src_b=src_b, 
        X_sample=X_w,
        quantiles=Q,
        model_pick=model_pick,
        load_mode=load_mode,
        hps_pick=hps_pick,
        model_name=model_name,
        prefer_artifact=prefer_artifact,
        log_fn=log,
    )

    # ------------------------------
    # Subsample warm-start examples
    # ------------------------------
    n_total = int(X_w["dynamic_features"].shape[0])
    idx = _choose_warm_idx(
        n_total=n_total,
        n_samples=warm_samples,
        frac=warm_frac,
        seed=warm_seed,
    )

    X_ws = _slice_npz_dict(X_w, idx)
    y_ws = _slice_npz_dict(y_wm, idx)

    # ------------------------------
    # Compile + fit warm-start
    # ------------------------------
    warm_keys = _compile_warm_model(
        model,
        quantiles=Q,
        lr=warm_lr,
    )
    y_ws = {
        k: y_ws[k]
        for k in warm_keys
        if k in y_ws
    }

    ds = _make_ds(
        X_ws,
        y_ws,
        batch_size=int(batch_size),
        seed=int(warm_seed),
    )

    with suppress_compiled_metrics_warning():
        model.fit(
            ds,
            epochs=int(warm_epochs),
            verbose=int(verbose),
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

    # ------------------------------
    # Evaluate using warmed weights
    # ------------------------------
    return run_one_direction(
        strategy="warm",
        rescale_mode=rescale_mode,
        model_pack=(model, model, bundle),
        warm_meta=warm_meta,
        src_b=src_b,
        tgt_b=tgt_b,
        split=split,
        calib_mode=calib_mode,
        rescale_to_source=rescale_to_source,
        batch_size=batch_size,
        quantiles_override=quantiles_override,
        save_dir=save_dir,
        model_pick=model_pick,
        load_mode=load_mode,
        hps_pick=hps_pick,
        model_name=model_name,
        prefer_artifact=prefer_artifact,
        allow_reorder_dynamic=allow_reorder_dynamic,
        allow_reorder_future=allow_reorder_future,
        recompute_missing=bool(recompute_missing),
        log_fn=log_fn,
        verbose=verbose,
    )

def main() -> None:
    args = parse_args()

    sink = _make_sink(args)
    log = _mklog(args, sink)

    # keep existing contract for called functions
    args.log_fn = sink

    load_kwargs = dict(
        model_pick=args.source_model,
        load_mode=args.source_load,
        hps_pick=args.hps_mode,
        model_name=args.model_name,
        prefer_artifact=args.prefer_artifact,
    )

    root = Path(args.results_dir)
    
    A = xu.load_stage1_bundle(
        results_root=root,
        city=args.city_a,
        model=args.model_name,
        stage="stage1",
        load_scalers_flag=True,
        strict=False,
    )
    
    B = xu.load_stage1_bundle(
        results_root=root,
        city=args.city_b,
        model=args.model_name,
        stage="stage1",
        load_scalers_flag=True,
        strict=False,
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
    
    if "baseline" in args.strategies:
        directions += [
            ("A_to_A", A, A),
            ("B_to_B", B, B),
        ]
    
    if "xfer" in args.strategies or "warm" in args.strategies:
        directions += [
            ("A_to_B", A, B),
            ("B_to_A", B, A),
        ]


    jobs = _build_jobs(args, directions)
    n_jobs = int(len(jobs))

    kv = _keras_verbose(args.verbose)

    log(
        f"[stage5] planned jobs={n_jobs} "
        f"dirs={len(directions)} "
        f"kv={kv}",
        level=3,
    )

    # Decide whether tqdm output should be redirected
    disable_pb = (
        args.progress == "off"
        or (sink is not print)
        and args.progress == "off"
    )

    pb_log_fn = None
    if args.progress == "log":
        pb_log_fn = sink
    elif args.progress == "auto":
        # terminal: let tqdm draw normally
        # GUI sink: redirect text
        pb_log_fn = None if sink is print else sink

    it = with_progress(
        jobs,
        total=n_jobs,
        desc="stage5 xfer",
        leave=True,
        disable=disable_pb,
        log_fn=pb_log_fn,
        mininterval=1.0,
    )

    results: List[Dict[str, Any]] = []

    for i, job in enumerate(it, start=1):
        tag = job["tag"]
        rm = job["rm"]
        cm = job["cm"]
        split = job["split"]
        strict = bool(job["strict"])
        strat = job["strategy"]

        src_b = job["src_b"]
        tgt_b = job["tgt_b"]

        M_src = src_b.manifest
        M_tgt = tgt_b.manifest

        src_city = M_src.get("city")
        tgt_city = M_tgt.get("city")

        log(
            f"[stage5] ({i}/{n_jobs}) "
            f"{tag} {strat} "
            f"rm={rm} split={split} "
            f"calib={cm} strict={strict} "
            f"{src_city}->{tgt_city}",
            level=3,
        )

        try:
            if job["kind"] == "one":
                r = run_one_direction(
                    strategy=strat,
                    rescale_mode=rm,
                    src_b=src_b,
                    tgt_b=tgt_b,
                    split=split,
                    calib_mode=cm,
                    rescale_to_source=strict,
                    batch_size=args.batch_size,
                    quantiles_override=args.quantiles,
                    save_dir=outdir,
                    allow_reorder_dynamic=(
                        args.allow_reorder_dynamic
                    ),
                    allow_reorder_future=(
                        args.allow_reorder_future
                    ),
                    recompute_missing=bool(args.recompute_missing),
                    log_fn=args.log_fn,
                    verbose=kv,
                    **load_kwargs,
                )
            else:
                r = run_warm_start_direction(
                    src_b=src_b,
                    tgt_b=tgt_b,
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
                    allow_reorder_dynamic=(
                        args.allow_reorder_dynamic
                    ),
                    allow_reorder_future=(
                        args.allow_reorder_future
                    ),
                    recompute_missing=bool(args.recompute_missing),
                    log_fn=args.log_fn,
                    verbose=kv,
                    **load_kwargs,
                )


        except SystemExit as e:
            log(
                f"[stage5] FAIL ({i}/{n_jobs}) "
                f"{tag} {strat}: {e}",
                level=1,
            )
            if not args.continue_on_error:
                raise
            continue

        except Exception as e:
            log(
                f"[stage5] ERROR ({i}/{n_jobs}) "
                f"{tag} {strat}: {e!r}",
                level=1,
            )
            if not args.continue_on_error:
                raise
            continue

        if r is None:
            log(
                f"[stage5] SKIP ({i}/{n_jobs}) "
                f"{tag} {strat}",
                level=2,
            )
            continue

        r["direction"] = tag
        r["source_city"] = src_city
        r["target_city"] = tgt_city
        r["job_index"] = i
        r["job_total"] = n_jobs

        results.append(r)

        log(
            f"[stage5] DONE ({i}/{n_jobs}) "
            f"mae={r.get('overall_mae')} "
            f"r2={r.get('overall_r2')}",
            level=3,
        )

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
        "overall_rmse",
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
    h_mse_keys = set()
    h_rmse_keys = set()
    h_r2_keys = set()
    for r in results:
        h_mae = _fix_horizon_keys(r.get("per_horizon_mae") or {})
        h_mae_keys |= set(h_mae.keys())
        h_mse = _fix_horizon_keys(r.get("per_horizon_mse") or {})
        h_mse_keys |= set(h_mse.keys())
        h_rmse = _fix_horizon_keys(r.get("per_horizon_rmse") or {})
        h_rmse_keys |= set(h_rmse.keys())
        h_r2 = _fix_horizon_keys(r.get("per_horizon_r2") or {})
        h_r2_keys |= set(h_r2.keys())

    h_mae_keys = _sorted_hkeys(h_mae_keys)
    h_mse_keys = _sorted_hkeys(h_mse_keys)
    h_rmse_keys = _sorted_hkeys(h_rmse_keys)
    h_r2_keys = _sorted_hkeys(h_r2_keys)

    cols = (
        base_cols
        + [f"per_horizon_mae.{k}" for k in h_mae_keys]
        + [f"per_horizon_mse.{k}" for k in h_mse_keys]
        + [f"per_horizon_rmse.{k}" for k in h_rmse_keys]
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
                r.get("overall_rmse"),
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

            ph_mae = _fix_horizon_keys(r.get("per_horizon_mae") or {})
            ph_mse = _fix_horizon_keys(r.get("per_horizon_mse") or {})
            ph_rmse = _fix_horizon_keys(r.get("per_horizon_rmse") or {})
            ph_r2 = _fix_horizon_keys(r.get("per_horizon_r2") or {})

            row.extend(
                [ph_mae.get(k, "NA") for k in h_mae_keys]
            )
            row.extend(
                [ph_mse.get(k, "NA") for k in h_mse_keys]
            )
            row.extend(
                [ph_rmse.get(k, "NA") for k in h_rmse_keys]
            )
            row.extend(
                [ph_r2.get(k, "NA") for k in h_r2_keys]
            )

            w.writerow(row)


if __name__ == "__main__":
    main()
