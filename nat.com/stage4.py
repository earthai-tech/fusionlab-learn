# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
r"""
stage4.py
Stage-4: Deterministic/Probabilistic inference for GeoPriorSubsNet (v3.2+).

This script is the inference counterpart of the Stage-2/Stage-3 workflows.
It is written to behave like Stage-2 (robust I/O, scaling-aware, Keras2/3
compatible load paths) while keeping inference clean and reproducible.

Key outputs (when available):
- calibrated forecast CSVs (eval + future)
- inference_summary.json (coverage/sharpness + point metrics in physical units)
- transfer_eval.json (optional scaled-space losses + physics epsilons)

Examples
--------
Usage (tuned model):
  python stage4.py ^
    --stage1-dir results/nansha_GeoPriorSubsNet_stage1 ^
    --model-path results/nansha_GeoPriorSubsNet_stage1/tuning/run_YYYYMMDD-HHMMSS/nansha_GeoPrior_best.keras ^
    --dataset test ^
    --calibrator results/nansha_GeoPriorSubsNet_stage1/train_YYYYMMDD-HHMMSS/interval_factors_80.npy ^
    --eval-losses --eval-physics --no-figs

Usage (trained model, re-fit calibrator on val):
  python stage4.py ^
    --stage1-dir results/nansha_GeoPriorSubsNet_stage1 ^
    --model-path results/nansha_GeoPriorSubsNet_stage1/train_YYYYMMDD-HHMMSS/nansha_GeoPriorSubsNet_H3.keras ^
    --dataset test --fit-calibrator

Cross-validation (A -> B, zero-shot):
  python stage4.py ^
    --stage1-dir results/zhongshan_GeoPriorSubsNet_stage1 ^
    --model-path results/nansha_GeoPriorSubsNet_stage1/train_*/nansha_GeoPriorSubsNet_H3.keras ^
    --dataset test --eval-losses --eval-physics --no-figs

A -> B with source calibrator:
  python stage4.py ^
    --stage1-dir results/zhongshan_GeoPriorSubsNet_stage1 ^
    --model-path results/nansha_GeoPriorSubsNet_stage1/train_*/nansha_GeoPriorSubsNet_H3.keras ^
    --dataset test --use-source-calibrator --eval-losses --eval-physics --no-figs

A -> B with target-val calibration (reported separately):
  python stage4.py ^
    --stage1-dir results/zhongshan_GeoPriorSubsNet_stage1 ^
    --model-path results/nansha_GeoPriorSubsNet_stage1/train_*/nansha_GeoPriorSubsNet_H3.keras ^
    --dataset test --fit-calibrator --eval-losses --eval-physics --no-figs
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# Quiet logs (same spirit as Stage-2/Stage-3)
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------
# fusionlab imports
# ---------------------------------------------------------------------
from fusionlab.registry.utils import _find_stage1_manifest
from fusionlab.utils.generic_utils import (
    default_results_dir,
    ensure_directory_exists,
    getenv_stripped,
)
from fusionlab.utils.forecast_utils import format_and_forecast
from fusionlab.utils.scale_metrics import (
    inverse_scale_target,
    per_horizon_metrics,
    point_metrics,
)

from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn
from fusionlab.nn.calibration import (
    IntervalCalibrator,
    apply_calibrator_to_subs,
    fit_interval_calibrator_on_val,
)


from fusionlab.params import LearnableKappa, LearnableMV, FixedGammaW, FixedHRef
from fusionlab.nn.pinn.geoprior.models import GeoPriorSubsNet

from fusionlab.compat.keras import load_inference_model, load_model_from_tfv2

from fusionlab.utils.nat_utils import (
    compile_geoprior_for_eval,
    ensure_input_shapes,
    extract_preds,
    load_best_hps_near_model,
    map_targets_for_training,
    pick_npz_for_dataset,
    resolve_si_affine,
)
from fusionlab.utils.shapes import canonicalize_BHQO

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GeoPriorSubsNet Stage-4 inference (v3.2+).")

    p.add_argument(
        "--stage1-dir",
        default=None,
        help="Folder containing Stage-1 manifest.json (optional).",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="Direct path to Stage-1 manifest.json (optional).",
    )

    p.add_argument(
        "--model-path",
        required=True,
        help=(
            "Path to a model artifact. Supported:\n"
            "- .keras full model\n"
            "- *_best_savedmodel/ TF export folder\n"
            "- directory containing the above + best weights\n"
        ),
    )

    p.add_argument(
        "--dataset",
        default="test",
        choices=["test", "val", "train", "custom"],
        help="Which split to run inference on.",
    )
    p.add_argument(
        "--inputs-npz",
        default=None,
        help="Custom inputs NPZ (required when --dataset custom).",
    )
    p.add_argument(
        "--targets-npz",
        default=None,
        help="Optional custom targets NPZ (enables metrics when --dataset custom).",
    )

    p.add_argument(
        "--eval-losses",
        action="store_true",
        help="Compute scaled-space MAE/MSE losses via Keras evaluate().",
    )
    p.add_argument(
        "--eval-physics",
        action="store_true",
        help="Compute epsilon_prior/epsilon_cons via model.evaluate_physics().",
    )

    p.add_argument(
        "--calibrator",
        default=None,
        help="Path to interval_factors_80.npy (optional).",
    )
    p.add_argument(
        "--use-source-calibrator",
        action="store_true",
        help="Load calibrator .npy from the SOURCE model directory (if present).",
    )
    p.add_argument(
        "--fit-calibrator",
        action="store_true",
        help="Fit an IntervalCalibrator on validation set if available.",
    )
    p.add_argument(
        "--cov-target",
        type=float,
        default=0.80,
        help="Target coverage for interval calibrator.",
    )

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-figs", action="store_true", help="Skip plotting (kept for CLI parity).")
    p.add_argument(
        "--include-gwl",
        action="store_true",
        help="Include GWL outputs in formatted CSVs when available.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_stage1_manifest(args: argparse.Namespace) -> dict:
    # 1) explicit --manifest
    if args.manifest and os.path.exists(args.manifest):
        return _read_json(args.manifest)

    # 2) explicit --stage1-dir
    if args.stage1_dir:
        mpath = os.path.join(args.stage1_dir, "manifest.json")
        if not os.path.exists(mpath):
            raise SystemExit(f"manifest.json not found in: {args.stage1_dir}")
        return _read_json(mpath)

    # 3) env / auto-discovery (same policy as Stage-2/Stage-3)
    base_dir = default_results_dir()
    city_hint = getenv_stripped("CITY")
    model_hint = getenv_stripped("MODEL_NAME_OVERRIDE", default="GeoPriorSubsNet")
    manual = getenv_stripped("STAGE1_MANIFEST")

    mpath = _find_stage1_manifest(
        manual=manual,
        base_dir=base_dir,
        city_hint=city_hint,
        model_hint=model_hint,
        prefer="timestamp",
        required_keys=("model", "stage"),
        verbose=1,
    )
    return _read_json(mpath)


def _sanitize_inputs_np(X: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in dict(X).items():
        if v is None:
            continue
        arr = np.asarray(v)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim > 0 and np.isfinite(arr).any():
            p99 = float(np.percentile(arr, 99))
            if p99 > 0:
                arr = np.clip(arr, -10.0 * p99, 10.0 * p99)
        out[k] = arr.astype(np.float32, copy=False)

    # keep H_field physically sane if present
    if "H_field" in out:
        out["H_field"] = np.maximum(out["H_field"], 1e-3).astype(np.float32)

    return out


def _load_encoders(M: dict) -> Tuple[Optional[Any], Optional[dict]]:
    enc = (M.get("artifacts", {}) or {}).get("encoders", {}) or {}

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

    # Attach scaler objects if only paths were stored (Stage-1 convention)
    if isinstance(scaler_info, dict):
        for _, v in scaler_info.items():
            if isinstance(v, dict) and ("scaler_path" in v) and ("scaler" not in v):
                p = v.get("scaler_path")
                if p and os.path.exists(p):
                    try:
                        v["scaler"] = joblib.load(p)
                    except Exception:
                        pass

    return coord_scaler, scaler_info if isinstance(scaler_info, dict) else None


def _resolve_bundle_paths(model_path: str) -> Dict[str, Optional[str]]:
    """
    Resolve best bundle paths following Stage-2 naming conventions.

    Returns keys:
    - run_dir
    - keras_path
    - weights_path
    - tf_dir
    - init_manifest_path
    """
    mp = os.path.abspath(model_path)

    # If user points at exported TF folder
    if os.path.isdir(mp) and mp.endswith("_best_savedmodel"):
        bundle_prefix = mp[: -len("_best_savedmodel")]
        run_dir = os.path.dirname(bundle_prefix)
        keras_path = f"{bundle_prefix}_best.keras"
        weights_path = f"{bundle_prefix}_best.weights.h5"
        tf_dir = mp
        init_manifest_path = os.path.join(run_dir, "model_init_manifest.json")
        return dict(
            run_dir=run_dir,
            keras_path=keras_path if os.path.isfile(keras_path) else None,
            weights_path=weights_path if os.path.isfile(weights_path) else None,
            tf_dir=tf_dir,
            init_manifest_path=init_manifest_path if os.path.isfile(init_manifest_path) else None,
        )

    # If user points at a .keras file
    if os.path.isfile(mp) and mp.endswith(".keras"):
        run_dir = os.path.dirname(mp)
        if mp.endswith("_best.keras"):
            bundle_prefix = mp[: -len("_best.keras")]
        else:
            bundle_prefix = mp[: -len(".keras")]
        weights_path = f"{bundle_prefix}_best.weights.h5"
        tf_dir = f"{bundle_prefix}_best_savedmodel"
        init_manifest_path = os.path.join(run_dir, "model_init_manifest.json")
        return dict(
            run_dir=run_dir,
            keras_path=mp,
            weights_path=weights_path if os.path.isfile(weights_path) else None,
            tf_dir=tf_dir if os.path.isdir(tf_dir) else None,
            init_manifest_path=init_manifest_path if os.path.isfile(init_manifest_path) else None,
        )

    # If user points at a directory (run folder)
    if os.path.isdir(mp):
        run_dir = mp
        init_manifest_path = os.path.join(run_dir, "model_init_manifest.json")
        return dict(
            run_dir=run_dir,
            keras_path=None,
            weights_path=None,
            tf_dir=None,
            init_manifest_path=init_manifest_path if os.path.isfile(init_manifest_path) else None,
        )

    raise SystemExit(f"Invalid --model-path: {model_path}")


def _pick_scaler_key(scaler_info: Optional[dict], primary: str, fallbacks: Tuple[str, ...]) -> str:
    if not isinstance(scaler_info, dict):
        return primary
    if primary in scaler_info:
        return primary
    for fb in fallbacks:
        if fb in scaler_info:
            return fb
    return primary


def _build_geoprior_builder(
    *,
    cfg_stage1: dict,
    X_sample: dict,
    out_s_dim: int,
    out_g_dim: int,
    horizon: int,
    mode: str,
    model_init_manifest: Optional[dict],
    best_hps: Optional[dict],
) -> Tuple[Any, dict]:
    """
    Return (builder_fn, resolved_model_kwargs_debug).

    We prefer model_init_manifest.json (Stage-2 artifact) because it is the
    most faithful JSON-safe snapshot of model init parameters.
    """
    init_cfg = (model_init_manifest or {}).get("config", {}) or {}
    init_dims = (model_init_manifest or {}).get("dims", {}) or {}

    # Ensure input shapes (Stage-2 contract)
    X_norm = ensure_input_shapes(X_sample, mode=mode, horizon=horizon)
    s_dim = int(X_norm["static_features"].shape[-1])
    d_dim = int(X_norm["dynamic_features"].shape[-1])
    f_dim = int(X_norm["future_features"].shape[-1])

    # Use init-manifest dims when present (prefer reproducibility)
    s_dim = int(init_dims.get("static_input_dim", s_dim))
    d_dim = int(init_dims.get("dynamic_input_dim", d_dim))
    f_dim = int(init_dims.get("future_input_dim", f_dim))

    quantiles = init_cfg.get("quantiles", cfg_stage1.get("QUANTILES", None))
    if quantiles in ([], (), ""):
        quantiles = None

    time_units = init_cfg.get("time_units", cfg_stage1.get("TIME_UNITS", "years"))

    pde_mode = init_cfg.get("pde_mode", cfg_stage1.get("PDE_MODE_CONFIG", cfg_stage1.get("PDE_MODE", "basic")))
    bounds_mode = init_cfg.get("bounds_mode", cfg_stage1.get("BOUNDS_MODE", "soft"))
    residual_method = init_cfg.get("residual_method", cfg_stage1.get("RESIDUAL_METHOD", "autodiff"))
    scale_pde_residuals = bool(init_cfg.get("scale_pde_residuals", cfg_stage1.get("SCALE_PDE_RESIDUALS", True)))

    # Scaling kwargs: prefer init-manifest, then Stage-1 config
    sk = init_cfg.get("scaling_kwargs", cfg_stage1.get("scaling_kwargs", {})) or {}
    sk = dict(sk)
    sk.setdefault("time_units", time_units)

    # Semantics for GWL variable (Stage-2 contract)
    gwl_kind = str(sk.get("gwl_kind", cfg_stage1.get("GWL_KIND", "depth_bgs"))).lower()
    gwl_sign = str(sk.get("gwl_sign", cfg_stage1.get("GWL_SIGN", "down_positive"))).lower()
    use_head_proxy = bool(sk.get("use_head_proxy", cfg_stage1.get("USE_HEAD_PROXY", True)))
    z_surf_col = sk.get("z_surf_col", cfg_stage1.get("Z_SURF_COL", None))
    gwl_z_meta = sk.get("gwl_z_meta", None)

    subsidence_kind = str(sk.get("subsidence_kind", cfg_stage1.get("SUBSIDENCE_KIND", "cumulative"))).lower()

    # SI affine (used by physics; keep consistent with Stage-2)
    subs_scale_si = sk.get("subs_scale_si")
    subs_bias_si = sk.get("subs_bias_si")
    head_scale_si = sk.get("head_scale_si")
    head_bias_si = sk.get("head_bias_si")

    if subs_scale_si is None or subs_bias_si is None:
        subs_scale_si, subs_bias_si = resolve_si_affine(cfg_stage1, sk, kind="subsidence")
    if head_scale_si is None or head_bias_si is None:
        head_scale_si, head_bias_si = resolve_si_affine(cfg_stage1, sk, kind="head")

    sk["subs_scale_si"] = float(subs_scale_si)
    sk["subs_bias_si"] = float(subs_bias_si)
    sk["head_scale_si"] = float(head_scale_si)
    sk["head_bias_si"] = float(head_bias_si)

    # GeoPrior scalar params
    geo = init_cfg.get("geoprior", {}) or {}
    init_mv = float(geo.get("init_mv", cfg_stage1.get("GEOPRIOR_INIT_MV", 1e-6)))
    init_kappa = float(geo.get("init_kappa", cfg_stage1.get("GEOPRIOR_INIT_KAPPA", 1e-5)))
    gamma_w = float(geo.get("gamma_w", cfg_stage1.get("GEOPRIOR_GAMMA_W", 9810.0)))

    # h_ref can be numeric or a mode string ("auto", etc)
    h_ref_raw = geo.get("h_ref", cfg_stage1.get("GEOPRIOR_H_REF", 0.0))
    if isinstance(h_ref_raw, (int, float, np.number)):
        h_ref = FixedHRef(value=float(h_ref_raw), mode="fixed")
    else:
        # mode string
        h_ref = FixedHRef(value=0.0, mode=str(h_ref_raw))

    kappa_mode = str(geo.get("kappa_mode", cfg_stage1.get("GEOPRIOR_KAPPA_MODE", "auto"))).lower()
    use_effective_h = bool(geo.get("use_effective_h", cfg_stage1.get("USE_EFFECTIVE_H", False)))
    hd_factor = float(geo.get("hd_factor", cfg_stage1.get("GEOPRIOR_HD_FACTOR", 1.0)))
    offset_mode = str(geo.get("offset_mode", cfg_stage1.get("OFFSET_MODE", "mul"))).lower()

    # Base kwargs from init-manifest (best reproducibility)
    base_kwargs: Dict[str, Any] = dict(
        static_input_dim=s_dim,
        dynamic_input_dim=d_dim,
        future_input_dim=f_dim,
        output_subsidence_dim=int(out_s_dim),
        output_gwl_dim=int(out_g_dim),
        forecast_horizon=int(horizon),
        quantiles=list(quantiles) if isinstance(quantiles, (list, tuple)) else None,
        pde_mode=pde_mode,
        mode=mode,
        time_units=time_units,
        bounds_mode=bounds_mode,
        residual_method=residual_method,
        scale_pde_residuals=scale_pde_residuals,
        scaling_kwargs=sk,
        # semantics / SI affine
        subs_scale_si=float(subs_scale_si),
        subs_bias_si=float(subs_bias_si),
        head_scale_si=float(head_scale_si),
        head_bias_si=float(head_bias_si),
        gwl_kind=gwl_kind,
        gwl_sign=gwl_sign,
        use_head_proxy=use_head_proxy,
        z_surf_col=z_surf_col,
        gwl_z_meta=gwl_z_meta,
        subsidence_kind=subsidence_kind,
        # geoprior
        mv=LearnableMV(initial_value=init_mv),
        kappa=LearnableKappa(initial_value=init_kappa),
        gamma_w=FixedGammaW(value=gamma_w),
        h_ref=h_ref,
        kappa_mode=kappa_mode,
        use_effective_h=use_effective_h,
        hd_factor=hd_factor,
        offset_mode=offset_mode,
    )

    # Architecture defaults from init-manifest, then Stage-1 cfg
    # (these affect weight shapes; keep consistent with Stage-2)
    arch_keys = [
        "embed_dim",
        "hidden_units",
        "lstm_units",
        "attention_units",
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
        "multi_scale_agg",
        "final_agg",
        "activation",
    ]
    for k in arch_keys:
        if k in init_cfg:
            base_kwargs[k] = init_cfg[k]
        elif k in cfg_stage1:
            base_kwargs[k] = cfg_stage1[k]

    # Best HP override (only safe keys; mirrors Stage-2 allowlist)
    allowed_hps = {
        "embed_dim",
        "hidden_units",
        "lstm_units",
        "attention_units",
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
        "multi_scale_agg",
        "final_agg",
        "activation",
    }
    best_hps = best_hps or {}
    hp_overrides = {k: v for k, v in best_hps.items() if k in allowed_hps}

    def _builder() -> GeoPriorSubsNet:
        params = dict(base_kwargs)
        params.update(hp_overrides)
        return GeoPriorSubsNet(**params)

    debug_kwargs = dict(base_kwargs)
    debug_kwargs.update(hp_overrides)
    debug_kwargs["quantiles"] = base_kwargs.get("quantiles", None)

    return _builder, debug_kwargs


def _load_interval_calibrator(
    *,
    args: argparse.Namespace,
    model_dir_for_source: str,
    model: GeoPriorSubsNet,
    ds_val: Optional[tf.data.Dataset],
) -> Optional[IntervalCalibrator]:
    # 0) load from source model directory (cross-domain convenience)
    if args.use_source_calibrator and not args.calibrator:
        cand = os.path.join(model_dir_for_source, "interval_factors_80.npy")
        if os.path.exists(cand):
            cal = IntervalCalibrator(target=float(args.cov_target))
            cal.factors_ = np.load(cand).astype(np.float32)
            print(f"[Calibrator] Loaded from source model dir: {cand}")
            return cal

    # 1) explicit path
    if args.calibrator and os.path.exists(args.calibrator):
        cal = IntervalCalibrator(target=float(args.cov_target))
        cal.factors_ = np.load(args.calibrator).astype(np.float32)
        print(f"[Calibrator] Loaded from explicit path: {args.calibrator}")
        return cal

    # 2) fit on validation
    if args.fit_calibrator and ds_val is not None:
        print("[Calibrator] Fitting on validation set...")
        return fit_interval_calibrator_on_val(model, ds_val, target=float(args.cov_target))

    return None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------
    # Load Stage-1 manifest (target-domain contract)
    # -----------------------------------------------------------------
    M = _resolve_stage1_manifest(args)
    cfg_stage1 = M.get("config", {}) or {}

    CITY = M.get("city", "city")
    MODEL_NAME = M.get("model", "GeoPriorSubsNet")

    MODE = str(cfg_stage1.get("MODE", "tft_like"))
    H = int(cfg_stage1.get("FORECAST_HORIZON_YEARS", cfg_stage1.get("FORECAST_HORIZON", 1)))
    FSY = cfg_stage1.get("FORECAST_START_YEAR", None)
    TRAIN_END_YEAR = cfg_stage1.get("TRAIN_END_YEAR", None)

    QUANTILES = cfg_stage1.get("QUANTILES", [0.1, 0.5, 0.9])
    if QUANTILES in ([], (), "", None):
        QUANTILES = None

    # output dims from Stage-1 manifest (as Stage-2 does)
    seq_dims = ((M.get("artifacts", {}) or {}).get("sequences", {}) or {}).get("dims", {}) or {}
    OUT_S_DIM = int(seq_dims.get("output_subsidence_dim", 1))
    OUT_G_DIM = int(seq_dims.get("output_gwl_dim", 1))

    coord_scaler, scaler_info = _load_encoders(M)

    # -----------------------------------------------------------------
    # Output dir (keep under target Stage-1 run_dir to avoid confusion)
    # -----------------------------------------------------------------
    run_dir_target = ((M.get("paths", {}) or {}).get("run_dir", None)) or (args.stage1_dir or os.getcwd())
    inf_dir = os.path.join(run_dir_target, "inference", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ensure_directory_exists(inf_dir)

    # -----------------------------------------------------------------
    # Load dataset inputs/targets
    # -----------------------------------------------------------------
    if args.dataset == "custom":
        if not args.inputs_npz:
            raise SystemExit("--inputs-npz is required when --dataset custom.")
        X = dict(np.load(args.inputs_npz))
        y = dict(np.load(args.targets_npz)) if args.targets_npz else None
    else:
        X, y = pick_npz_for_dataset(M, args.dataset)
        if X is None:
            raise SystemExit(
                f"No NPZs found for dataset='{args.dataset}'. "
                "Re-run Stage-1 with that split or use --dataset custom."
            )

    X = _sanitize_inputs_np(X)
    X = ensure_input_shapes(X, mode=MODE, horizon=H)

    y_map: Dict[str, np.ndarray] = {}
    if y is not None:
        y_map = map_targets_for_training(dict(y)) or {}

    # Optional validation dataset for calibrator fitting
    ds_val = None
    if args.fit_calibrator:
        npz = (M.get("artifacts", {}) or {}).get("numpy", {}) or {}
        v_inp = npz.get("val_inputs_npz")
        v_tgt = npz.get("val_targets_npz")
        if v_inp and v_tgt and os.path.exists(v_inp) and os.path.exists(v_tgt):
            vx = _sanitize_inputs_np(dict(np.load(v_inp)))
            vx = ensure_input_shapes(vx, mode=MODE, horizon=H)
            vy = map_targets_for_training(dict(np.load(v_tgt))) or {}
            if vy:
                ds_val = tf.data.Dataset.from_tensor_slices((vx, vy)).batch(int(args.batch_size))

    # -----------------------------------------------------------------
    # Load/rebuild model (Stage-2 compatible)
    # -----------------------------------------------------------------
    bundle = _resolve_bundle_paths(args.model_path)
    run_dir_source = bundle["run_dir"] or os.path.dirname(os.path.abspath(args.model_path))
    init_manifest_path = bundle["init_manifest_path"]

    model_init_manifest = _read_json(init_manifest_path) if init_manifest_path else None

    # best_hps near model (used only as override when init-manifest is missing)
    best_hps = load_best_hps_near_model(bundle.get("keras_path") or run_dir_source) or {}

    builder, builder_debug = _build_geoprior_builder(
        cfg_stage1=cfg_stage1,
        X_sample=X,
        out_s_dim=OUT_S_DIM,
        out_g_dim=OUT_G_DIM,
        horizon=H,
        mode=MODE,
        model_init_manifest=model_init_manifest,
        best_hps=best_hps,
    )

    print(f"[Manifest] city={CITY} model={MODEL_NAME} mode={MODE} H={H}")
    print(f"[Model] source_run_dir={run_dir_source}")
    if model_init_manifest is not None:
        print(f"[Model] init-manifest={init_manifest_path}")
    if bundle.get("keras_path"):
        print(f"[Model] keras_path={bundle.get('keras_path')}")
    if bundle.get("weights_path"):
        print(f"[Model] weights_path={bundle.get('weights_path')}")
    if bundle.get("tf_dir"):
        print(f"[Model] tf_dir={bundle.get('tf_dir')}")

    # Eval-capable model (GeoPriorSubsNet methods preserved)
    model: GeoPriorSubsNet = load_inference_model(
        keras_path=bundle.get("keras_path"),
        weights_path=bundle.get("weights_path"),
        manifest_path=init_manifest_path,
        manifest=model_init_manifest,
        builder=builder,
        build_inputs=X,
        out_s_dim=OUT_S_DIM,
        out_g_dim=OUT_G_DIM,
        mode=MODE,
        horizon=H,
        prefer_full_model=False,
    )

    # Optional TF SavedModel prediction endpoint (fast), but not eval-capable
    model_pred = model
    if bundle.get("tf_dir"):
        try:
            model_pred = load_model_from_tfv2(bundle["tf_dir"], endpoint="serve")
            print(f"[OK] Loaded TF SavedModel endpoint: {bundle['tf_dir']}")
        except Exception as e:
            print(f"[WARN] Failed to load TF SavedModel endpoint: {e}")
            model_pred = model

    # -----------------------------------------------------------------
    # Calibrator
    # -----------------------------------------------------------------
    cal = _load_interval_calibrator(
        args=args,
        model_dir_for_source=run_dir_source,
        model=model,
        ds_val=ds_val,
    )

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------
    pred_out = model_pred.predict(X, verbose=0) if hasattr(model_pred, "predict") else model_pred(X)
    if isinstance(pred_out, dict):
        pred_dict = pred_out
    else:
        # treat as legacy "data_final"
        pred_dict = {"data_final": pred_out}

    # Always extract via the eval-capable GeoPrior model (split support)
    # subs_pred, gwl_pred = extract_preds(model, pred_dict)

    # # Canonicalize quantile layouts (Stage-2 contract)
    # if subs_pred is not None and getattr(subs_pred, "ndim", 0) == 4:
    #     subs_pred = canonicalize_BHQO_quantiles_np(subs_pred, q_axis="auto")
    # if gwl_pred is not None and getattr(gwl_pred, "ndim", 0) == 4:
    #     gwl_pred = canonicalize_BHQO_quantiles_np(gwl_pred, q_axis="auto")

    # # Apply interval calibrator (subsidence quantiles only)
    # if cal is not None and subs_pred is not None and getattr(subs_pred, "ndim", 0) == 4:
    #     subs_pred = apply_calibrator_to_subs(cal, subs_pred)


    subs_pred, gwl_pred = extract_preds(model, pred_dict)
    
    q_values = list(QUANTILES) if QUANTILES else None
    n_q = len(q_values) if q_values else None
    
    def _canon(xq, y_true=None):
        if xq is None or getattr(xq, "ndim", 0) != 4:
            return xq
        out = canonicalize_BHQO(
            xq,
            y_true=y_true,
            q_values=q_values,
            n_q=n_q,
            layout="BHQO",
            enforce_monotone=False,
            verbose=0,
            log_fn=print,
        )
        # be robust if it returns (arr, layout_used)
        if isinstance(out, tuple):
            return out[0]
        return out
    
    subs_pred = _canon(subs_pred, y_true=y_map.get("subs_pred") if y_map else None)
    gwl_pred  = _canon(gwl_pred,  y_true=y_map.get("gwl_pred")  if y_map else None)
    
    # now calibration is safe (subs_pred is BHQO)
    if cal is not None and getattr(subs_pred, "ndim", 0) == 4:
        subs_pred = apply_calibrator_to_subs(cal, subs_pred)

    # Prepare outputs for formatter
    y_pred_fmt: Dict[str, np.ndarray] = {"subs_pred": subs_pred}
    y_true_fmt: Optional[Dict[str, np.ndarray]] = None

    if args.include_gwl and (gwl_pred is not None):
        y_pred_fmt["gwl_pred"] = gwl_pred

    if y_map:
        y_true_fmt = {"subsidence": y_map.get("subs_pred"), "gwl": y_map.get("gwl_pred")}
        if not args.include_gwl:
            y_true_fmt = {"subsidence": y_map.get("subs_pred")}

    # -----------------------------------------------------------------
    # Format + forecast CSVs (same API as Stage-2)
    # -----------------------------------------------------------------
    cols_cfg = cfg_stage1.get("cols", {}) or {}
    SUBS_COL = cols_cfg.get("subsidence", "subsidence")

    subs_kind = str(cfg_stage1.get("SUBSIDENCE_KIND", "cumulative")).lower()
    if subs_kind not in ("cumulative", "rate"):
        subs_kind = "cumulative"

    # pick scaler target key robustly (Stage-2 logic)
    SUBS_SCALER_KEY = _pick_scaler_key(
        scaler_info=scaler_info,
        primary=SUBS_COL,
        fallbacks=("subsidence", "subs_pred", "subs", "s"),
    )

    base_name = f"{CITY}_{MODEL_NAME}_inference_{args.dataset}_H{H}"
    if cal is not None:
        base_name += "_calibrated"

    csv_eval = os.path.join(inf_dir, base_name + "_eval.csv")
    csv_future = os.path.join(inf_dir, base_name + "_future.csv")

    future_grid = None
    if FSY is not None:
        future_grid = np.arange(float(FSY), float(FSY) + float(H), dtype=float)

    df_eval, df_future = format_and_forecast(
        y_pred=y_pred_fmt,
        y_true=y_true_fmt,
        coords=X.get("coords", None),
        quantiles=QUANTILES if QUANTILES else None,
        target_name=SUBS_COL,
        target_key_pred="subs_pred",
        component_index=0,
        scaler_info=scaler_info,
        scaler_target_name=SUBS_SCALER_KEY,
        output_target_name="subsidence",
        coord_scaler=coord_scaler,
        coord_columns=("coord_t", "coord_x", "coord_y"),
        train_end_time=TRAIN_END_YEAR,
        forecast_start_time=FSY,
        forecast_horizon=H,
        future_time_grid=future_grid,
        eval_forecast_step=None,
        sample_index_offset=0,
        city_name=CITY,
        model_name=MODEL_NAME,
        dataset_name=str(args.dataset),
        dataset_name_for_forecast=f"{args.dataset} forecast",
        csv_eval_path=csv_eval,
        csv_future_path=csv_future,
        time_as_datetime=False,
        time_format=None,
        verbose=1,
        # inference mode: we keep extra metrics in our JSON below
        eval_metrics=False,
        metrics_savefile=None,
        value_mode=subs_kind,
        input_value_mode=subs_kind,
    )

    if df_eval is not None and not df_eval.empty:
        print(f"[OK] Saved EVAL CSV -> {csv_eval}")
    else:
        print("[Info] df_eval is empty (no eval targets or no overlap).")

    if df_future is not None and not df_future.empty:
        print(f"[OK] Saved FUTURE CSV -> {csv_future}")
    else:
        print("[Info] df_future is empty (no future grid requested or empty output).")

    # -----------------------------------------------------------------
    # inference_summary.json (physical units when possible)
    # -----------------------------------------------------------------
    summary: Dict[str, Any] = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "city": CITY,
        "model": MODEL_NAME,
        "horizon": int(H),
        "mode": MODE,
        "calibrated": bool(cal is not None),
        "quantiles": list(QUANTILES) if (subs_pred is not None and getattr(subs_pred, "ndim", 0) == 4) else None,
        "csv_eval": csv_eval if (df_eval is not None and not df_eval.empty) else None,
        "csv_future": csv_future if (df_future is not None and not df_future.empty) else None,
    }

    # Physical-unit metrics (only if we have targets + scaler_info)
    if y_map and (subs_pred is not None) and isinstance(scaler_info, dict):
        y_true_scaled = y_map.get("subs_pred")
        if y_true_scaled is not None:
            # quantile case
            if getattr(subs_pred, "ndim", 0) == 4 and QUANTILES:
                y_true_tf = tf.convert_to_tensor(y_true_scaled, tf.float32)
                s_q_tf = tf.convert_to_tensor(subs_pred, tf.float32)
                summary["coverage80_scaled"] = float(coverage80_fn(y_true_tf, s_q_tf).numpy())
                summary["sharpness80_scaled"] = float(sharpness80_fn(s_q_tf).numpy())

                y_true_phys = inverse_scale_target(
                    y_true_scaled,
                    scaler_info=scaler_info,
                    target_name=SUBS_SCALER_KEY,
                    is_quantile=False,
                )
                s_q_phys = inverse_scale_target(
                    subs_pred,
                    scaler_info=scaler_info,
                    target_name=SUBS_SCALER_KEY,
                    is_quantile=True,
                )

                y_true_phys_tf = tf.convert_to_tensor(y_true_phys, tf.float32)
                s_q_phys_tf = tf.convert_to_tensor(s_q_phys, tf.float32)
                summary["coverage80_phys"] = float(coverage80_fn(y_true_phys_tf, s_q_phys_tf).numpy())
                summary["sharpness80_phys"] = float(sharpness80_fn(s_q_phys_tf).numpy())

                q_list = list(QUANTILES)
                med_idx = q_list.index(0.5) if 0.5 in q_list else 0
                s_med_phys = s_q_phys[..., med_idx, :]

                summary["point_metrics_phys"] = point_metrics(y_true_phys, s_med_phys)
                mae_h, r2_h = per_horizon_metrics(y_true_phys, s_med_phys)
                summary["per_horizon_mae_phys"] = mae_h
                summary["per_horizon_r2_phys"] = r2_h

            # point-only case
            elif getattr(subs_pred, "ndim", 0) == 3:
                y_true_phys = inverse_scale_target(
                    y_true_scaled,
                    scaler_info=scaler_info,
                    target_name=SUBS_SCALER_KEY,
                    is_quantile=False,
                )
                s_pred_phys = inverse_scale_target(
                    subs_pred,
                    scaler_info=scaler_info,
                    target_name=SUBS_SCALER_KEY,
                    is_quantile=False,
                )
                summary["point_metrics_phys"] = point_metrics(y_true_phys, s_pred_phys)
                mae_h, r2_h = per_horizon_metrics(y_true_phys, s_pred_phys)
                summary["per_horizon_mae_phys"] = mae_h
                summary["per_horizon_r2_phys"] = r2_h

    summary_path = os.path.join(inf_dir, "inference_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved inference summary -> {summary_path}")

    # -----------------------------------------------------------------
    # Optional scaled-space eval + physics diagnostics (debug/transfer)
    # -----------------------------------------------------------------
    ds_eval = None
    if y_map:
        ds_eval = tf.data.Dataset.from_tensor_slices((X, y_map)).batch(int(args.batch_size))

    need_eval = (args.eval_losses and ds_eval is not None) or (args.eval_physics and ds_eval is not None)

    # Compile only if needed and not compiled
    if need_eval and getattr(model, "compiled_loss", None) is None:
        try:
            compile_geoprior_for_eval(
                model,
                M=M,
                best_hps=best_hps or None,
                quantiles=list(QUANTILES) if isinstance(QUANTILES, (list, tuple)) else None,
            )
            print("[OK] Recompiled model for evaluation/physics.")
        except Exception as e:
            print(f"[WARN] Could not compile model for eval/physics: {e}")
            args.eval_losses = False
            args.eval_physics = False

    keras_eval = None
    if args.eval_losses and ds_eval is not None:
        try:
            keras_eval = model.evaluate(ds_eval, return_dict=True, verbose=0)
        except Exception as e:
            print(f"[WARN] Keras evaluate failed: {e}")
            keras_eval = None

    physics_diag = None
    if args.eval_physics and ds_eval is not None:
        try:
            phys_raw = model.evaluate_physics(
                ds_eval,
                max_batches=10,
                return_maps=False,
            )
            physics_diag = {
                k: float(v.numpy())
                for k, v in phys_raw.items()
                if getattr(v, "shape", ()) == ()
            }
            print("[OK] Physics diagnostics (approx, first batches):", physics_diag)
        except Exception as e:
            print(f"[WARN] Physics eval failed: {e}")
            physics_diag = None

    transfer_path = os.path.join(inf_dir, "transfer_eval.json")
    with open(transfer_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "csv_eval": summary.get("csv_eval"),
                "csv_future": summary.get("csv_future"),
                "keras_eval_scaled": keras_eval,
                "physics": physics_diag,
                "summary": summary,
                # small debug: builder params that were used when rebuild was needed
                "rebuild_debug": {
                    "used_init_manifest": bool(model_init_manifest is not None),
                    "builder_keys": sorted(list(builder_debug.keys())),
                },
            },
            f,
            indent=2,
        )
    print(f"[OK] Saved transfer eval -> {transfer_path}")


if __name__ == "__main__":
    main()
