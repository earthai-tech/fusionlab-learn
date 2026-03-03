# nat.com/sensitivity_lib.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause

"""Drop-in `run_one()` for nat.com/sensitivity_lib.py.

This implementation is adapted from nat.com/sensitivity.py,
with the heavy steps moved to `build_context()`:
- Stage-1 manifest lookup
- NPZ loading
- tf.data pipeline building

`run_one()` assumes `ctx` contains:
- ctx.manifest, ctx.manifest_path
- ctx.cfg_base
- ctx.base_output_dir
- ctx.scaler_info
- ctx.X_train / ctx.y_train / ctx.X_val / ctx.y_val
- optional ctx.X_test / ctx.y_test
- ctx.ds_train / ctx.ds_val / optional ctx.ds_test
- ctx.dyn_names / ctx.fut_names / ctx.sta_names

It writes all artifacts under `run_dir`.
"""


from __future__ import annotations

import copy
import datetime as dt
import sys
import gc
import json
import os
import re
import joblib 
import platform 

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from fusionlab.registry import _find_stage1_manifest
from fusionlab.utils import (
    default_results_dir,
    ensure_directory_exists,
    getenv_stripped,
    load_nat_config,
    load_nat_config_payload,
    load_scaler_info,
    make_tf_dataset,
    resolve_hybrid_config,
)

from fusionlab.backends.devices import configure_tf_from_cfg


def _sanitize_tag(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s.strip())
    s = s.strip("._-")
    s = s[:120]
    return s or None


@dataclass
class SensitivityContext:
    manifest_path: str
    manifest: Dict[str, Any]
    cfg_base: Dict[str, Any]

    # stage-1 paths / outputs
    base_output_dir: str

    # scalers / encoders
    scaler_info: Dict[str, Any]

    # npz arrays
    X_train: Dict[str, np.ndarray]
    y_train: Dict[str, np.ndarray]
    X_val: Dict[str, np.ndarray]
    y_val: Dict[str, np.ndarray]
    X_test: Optional[Dict[str, np.ndarray]]
    y_test: Optional[Dict[str, np.ndarray]]

    # datasets (cached)
    ds_train: tf.data.Dataset
    ds_val: tf.data.Dataset
    ds_test: Optional[tf.data.Dataset]

    # reusable “static” knobs
    dyn_names: Tuple[str, ...]
    fut_names: Tuple[str, ...]
    sta_names: Tuple[str, ...]
    mode: str
    horizon: int
    batch_size: int


def build_context(
    *,
    city: Optional[str] = None,
    stage1_manifest: Optional[str] = None,
    verbose: int = 1,
) -> SensitivityContext:
    """
    One-time heavy setup:
    - locate Stage-1 manifest
    - resolve cfg (hybrid)
    - load NPZ arrays
    - build tf.data datasets (train/val/test)
    """

    results_dir = default_results_dir()

    payload = load_nat_config_payload()
    cfg_city = (payload.get("city") or "").strip().lower() or None

    city_env = getenv_stripped("CITY")
    city_hint = (city_env or city or cfg_city)

    manual = stage1_manifest or getenv_stripped("STAGE1_MANIFEST")

    manifest_path = _find_stage1_manifest(
        manual=manual,
        base_dir=results_dir,
        city_hint=city_hint,
        model_hint=None,
        prefer="timestamp",
        required_keys=("model", "stage", "artifacts", "config", "paths"),
        filter_fn=None,
        verbose=1 if verbose else 0,
    )

    with open(manifest_path, "r", encoding="utf-8") as f:
        M = json.load(f)

    cfg_global = load_nat_config()
    cfg_manifest = M.get("config", {}) or {}

    cfg = resolve_hybrid_config(
        manifest_cfg=cfg_manifest,
        live_cfg=cfg_global,
        verbose=bool(verbose),
    )

    _ = configure_tf_from_cfg(cfg)

    features = cfg.get("features", {}) or {}
    dyn = tuple(features.get("dynamic", []) or [])
    fut = tuple(features.get("future", []) or [])
    sta = tuple(features.get("static", []) or [])

    mode = str(cfg.get("MODE", cfg.get("mode", "sequence")))
    horizon = int(cfg.get("FORECAST_HORIZON_YEARS", 3))
    batch_size = int(cfg.get("BATCH_SIZE", 32))

    enc = M["artifacts"]["encoders"]
    scaler_info = load_scaler_info(enc)

    npz = M["artifacts"]["numpy"]
    X_train = dict(np.load(npz["train_inputs_npz"]))
    y_train = dict(np.load(npz["train_targets_npz"]))
    X_val = dict(np.load(npz["val_inputs_npz"]))
    y_val = dict(np.load(npz["val_targets_npz"]))

    X_test = None
    y_test = None
    if npz.get("test_inputs_npz") and npz.get("test_targets_npz"):
        X_test = dict(np.load(npz["test_inputs_npz"]))
        y_test = dict(np.load(npz["test_targets_npz"]))

    ds_train = make_tf_dataset(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        mode=mode,
        forecast_horizon=horizon,
        check_npz_finite=True,
        check_finite=True,
        scan_finite_batches=None,
        dynamic_feature_names=list(dyn),
        future_feature_names=list(fut),
    )
    ds_val = make_tf_dataset(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False,
        mode=mode,
        forecast_horizon=horizon,
        check_npz_finite=True,
        check_finite=True,
        scan_finite_batches=None,
        dynamic_feature_names=list(dyn),
        future_feature_names=list(fut),
    )

    ds_test = None
    if X_test is not None and y_test is not None:
        ds_test = make_tf_dataset(
            X_test,
            y_test,
            batch_size=batch_size,
            shuffle=False,
            mode=mode,
            forecast_horizon=horizon,
            check_npz_finite=True,
            check_finite=True,
            scan_finite_batches=None,
            dynamic_feature_names=list(dyn),
            future_feature_names=list(fut),
        )

    base_out = M["paths"]["run_dir"]

    return SensitivityContext(
        manifest_path=str(manifest_path),
        manifest=M,
        cfg_base=cfg,
        base_output_dir=str(base_out),
        scaler_info=scaler_info,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        dyn_names=dyn,
        fut_names=fut,
        sta_names=sta,
        mode=mode,
        horizon=horizon,
        batch_size=batch_size,
    )


def _make_run_dir(
    base_output_dir: str,
    run_tag: Optional[str],
    stable: bool,
) -> str:
    tag = _sanitize_tag(run_tag)
    if stable and tag:
        name = f"sens__{tag}"
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"train_{stamp}"
        if tag:
            name = f"{name}__{tag}"
    out = str(Path(base_output_dir) / name)
    ensure_directory_exists(out)
    return out

def cleanup_between_runs() -> None:
    """Clear TF/Keras graphs between runs."""

    tf.keras.backend.clear_session()
    gc.collect()


def run_one(
    ctx,
    *,
    overrides: Dict[str, Any],
    run_tag: Optional[str],
    stable_run_dir: bool = True,
    eval_max_batches: Optional[int] = None,
    cal_max_batches: Optional[int] = None,
) -> str:
    """Run one physics sensitivity trial.

    Notes
    -----
    - Reuses ctx.ds_train/ctx.ds_val/ctx.ds_test.
    - All artifacts are written to the returned run_dir.
    """

    # --- imports (local to keep module light) -----------------
    from tensorflow.keras.callbacks import (
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
        TerminateOnNaN,
    )

    from fusionlab.api.util import get_table_size
    from fusionlab.backends.devices import configure_tf_from_cfg
    from fusionlab.compat import (
        load_inference_model,
        load_model_from_tfv2,
        normalize_predict_output,
        save_manifest,
        save_model,
    )
    from fusionlab.deps import with_progress
    from fusionlab.plot import plot_eval_future
    from fusionlab.nn import (
        Coverage80,
        LambdaOffsetScheduler,
        MAEQ50,
        MSEQ50,
        Sharpness80,
        _logs_to_py,
        _to_py,
        apply_calibrator_to_subs,
        coverage80_fn,
        # debug_tensor_interval,
        debug_val_interval,
        fit_interval_calibrator_on_val,
        make_weighted_pinball,
        plot_history_in,
        sharpness80_fn,
    )
    from fusionlab.nn.pinn import (
        GeoPriorSubsNet,
        PoroElasticSubsNet,
        autoplot_geoprior_history,
        extract_physical_parameters,
        finalize_scaling_kwargs,
        debug_model_reload,
        load_physics_payload,
        override_scaling_kwargs,
        plot_physics_values_in,
    )
    from fusionlab.params import (
        FixedGammaW,
        FixedHRef,
        LearnableKappa,
        LearnableMV,
    )
    from fusionlab.utils import (
        audit_stage2_handshake,
        best_epoch_and_metrics,
        build_censor_mask,
        calibrate_quantile_forecasts,
        convert_eval_payload_units,
        # default_results_dir,
        deg_to_m_from_lat,
        ensure_directory_exists,
        ensure_input_shapes,
        evaluate_forecast,
        evaluate_point_forecast,
        format_and_forecast,
        inverse_scale_target,
        load_scaler_info,
        map_targets_for_training,
        name_of,
        postprocess_eval_json,
        print_config_table,
        resolve_si_affine,
        save_ablation_record,
        save_all_figures,
        serialize_subs_params,
        should_audit,
    )

    # ----------------------------------------------------------

    cfg = copy.deepcopy(ctx.cfg_base)
    cfg.update(overrides or {})

    # Use caller-provided tag unless cfg has RUN_TAG.
    run_tag2 = _sanitize_tag(run_tag or cfg.get("RUN_TAG"))

    run_dir = _make_run_dir(
        ctx.base_output_dir,
        run_tag=run_tag2,
        stable=stable_run_dir,
    )
    ensure_directory_exists(run_dir)

    # ---------------------------
    # Basic runtime knobs
    # ---------------------------
    FAST_SENS = bool(cfg.get("FAST_SENSITIVITY", False))
    DEBUG = bool(cfg.get("DEBUG", False))

    # Unit post-processing for eval JSON.
    units_mode = str(cfg.get("EVAL_JSON_UNITS_MODE", "si") or "si")
    units_mode = units_mode.strip().lower()
    units_scope = str(cfg.get("EVAL_JSON_UNITS_SCOPE", "all") or "all")
    units_scope = units_scope.strip().lower()

    # ---------------------------
    # Manifest + names
    # ---------------------------
    M = ctx.manifest
    CITY_NAME = M.get("city", cfg.get("CITY_NAME", "nansha"))
    MODEL_NAME = cfg.get("MODEL_NAME", cfg.get("model", None))
    if not MODEL_NAME:
        MODEL_NAME = "GeoPriorSubsNet"

    # ---------------------------
    # Cached tensors/datasets
    # ---------------------------
    X_train = ctx.X_train
    y_train = ctx.y_train
    X_val = ctx.X_val
    y_val = ctx.y_val
    X_test = ctx.X_test
    y_test = ctx.y_test

    train_dataset = ctx.ds_train
    val_dataset = ctx.ds_val
    test_dataset = ctx.ds_test

    DYN_NAMES = list(ctx.dyn_names)
    FUT_NAMES = list(ctx.fut_names)
    STA_NAMES = list(ctx.sta_names)

    # ---------------------------
    # Config essentials
    # ---------------------------
    TIME_STEPS = int(cfg["TIME_STEPS"])
    FORECAST_H = int(cfg["FORECAST_HORIZON_YEARS"])
    MODE = str(cfg["MODE"])
    FORECAST_START_YEAR = int(cfg.get("FORECAST_START_YEAR"))

    # Architecture
    ATTENTION_LEVELS = cfg.get(
        "ATTENTION_LEVELS",
        ["cross", "hierarchical", "memory"],
    )
    EMBED_DIM = int(cfg.get("EMBED_DIM", 32))
    HIDDEN_UNITS = int(cfg.get("HIDDEN_UNITS", 64))
    LSTM_UNITS = int(cfg.get("LSTM_UNITS", 64))
    ATTENTION_UNITS = int(cfg.get("ATTENTION_UNITS", 64))
    NUMBER_HEADS = int(cfg.get("NUMBER_HEADS", 2))
    DROPOUT_RATE = float(cfg.get("DROPOUT_RATE", 0.10))
    MEMORY_SIZE = int(cfg.get("MEMORY_SIZE", 50))
    SCALES = cfg.get("SCALES", [1, 2])
    USE_RESIDUALS = bool(cfg.get("USE_RESIDUALS", True))
    USE_BATCH_NORM = bool(cfg.get("USE_BATCH_NORM", False))
    USE_VSN = bool(cfg.get("USE_VSN", True))
    VSN_UNITS = int(cfg.get("VSN_UNITS", 32))

    # Training
    EPOCHS = int(cfg.get("EPOCHS", 50))

    BATCH_SIZE_CFG = int(cfg.get("BATCH_SIZE", 32))
    if hasattr(ctx, "batch_size"):
        if int(ctx.batch_size) != int(BATCH_SIZE_CFG):
            print(
                "[Warn] Gold mode uses cached tf.data datasets "
                f"built with BATCH_SIZE={int(ctx.batch_size)}; "
                f"ignoring override BATCH_SIZE={int(BATCH_SIZE_CFG)}. "
                "Rebuild context to change batch size."
            )
        BATCH_SIZE = int(ctx.batch_size)
        cfg["BATCH_SIZE"] = BATCH_SIZE
    else:
        BATCH_SIZE = int(BATCH_SIZE_CFG)
    
    LEARNING_RATE = float(cfg.get("LEARNING_RATE", 1e-4))
    VERBOSE = int(cfg.get("VERBOSE", 1))

    LEARNING_RATE = float(cfg.get("LEARNING_RATE", 1e-4))
    VERBOSE = int(cfg.get("VERBOSE", 1))

    # Quantiles
    QUANTILES = cfg.get("QUANTILES", [0.1, 0.5, 0.9])

    def _coerce_qw(d: dict, default: dict) -> dict:
        if not d:
            return default
        out = {}
        for k, v in d.items():
            try:
                q = float(k)
            except Exception:
                q = k
            out[q] = float(v)
        return out

    SUBS_WEIGHTS = _coerce_qw(
        cfg.get("SUBS_WEIGHTS", None)
        or {0.1: 3.0, 0.5: 1.0, 0.9: 3.0},
        {0.1: 3.0, 0.5: 1.0, 0.9: 3.0},
    )
    GWL_WEIGHTS = _coerce_qw(
        cfg.get("GWL_WEIGHTS", None)
        or {0.1: 1.5, 0.5: 1.0, 0.9: 1.5},
        {0.1: 1.5, 0.5: 1.0, 0.9: 1.5},
    )

    # Physics
    TIME_UNITS = str(cfg.get("TIME_UNITS", "year") or "year").lower()
    SCALE_PDE_RESIDUALS = bool(cfg.get("SCALE_PDE_RESIDUALS", True))
    CONS_RESID_METHOD = str(
        cfg.get(
            "CONSOLIDATION_STEP_RESIDUAL_METHOD",
            "exact",
        )
    )

    PDE_MODE = str(cfg.get("PDE_MODE_CONFIG", "off") or "off")
    PDE_MODE = PDE_MODE.strip().lower()
    if PDE_MODE in ("off", "none"):
        PDE_MODE = "none"

    LAMBDA_CONS = float(cfg.get("LAMBDA_CONS", 0.10))
    LAMBDA_GW = float(cfg.get("LAMBDA_GW", 0.01))
    LAMBDA_PRIOR = float(cfg.get("LAMBDA_PRIOR", 0.10))
    LAMBDA_SMOOTH = float(cfg.get("LAMBDA_SMOOTH", 0.01))
    LAMBDA_MV = float(cfg.get("LAMBDA_MV", 0.01))
    LAMBDA_BOUNDS = float(cfg.get("LAMBDA_BOUNDS", 0.0))

    LOSS_WEIGHT_GWL = float(cfg.get("LOSS_WEIGHT_GWL", 0.5))
    LAMBDA_Q = float(cfg.get("LAMBDA_Q", 0.0))
    LOG_Q_DIAGNOSTICS = bool(cfg.get("LOG_Q_DIAGNOSTICS", False))

    MV_LR_MULT = float(cfg.get("MV_LR_MULT", 1.0))
    KAPPA_LR_MULT = float(cfg.get("KAPPA_LR_MULT", 5.0))
    OFFSET_MODE = str(cfg.get("OFFSET_MODE", "mul"))
    LAMBDA_OFFSET = float(cfg.get("LAMBDA_OFFSET", 1.0))

    USE_LAMBDA_OFFSET_SCHED = bool(
        cfg.get("USE_LAMBDA_OFFSET_SCHEDULER", False)
    )
    LAMBDA_OFFSET_UNIT = cfg.get("LAMBDA_OFFSET_UNIT", "epoch")
    LAMBDA_OFFSET_WHEN = cfg.get("LAMBDA_OFFSET_WHEN", "begin")
    LAMBDA_OFFSET_WARMUP = int(cfg.get("LAMBDA_OFFSET_WARMUP", 10))
    LAMBDA_OFFSET_START = cfg.get("LAMBDA_OFFSET_START", None)
    LAMBDA_OFFSET_END = cfg.get("LAMBDA_OFFSET_END", None)
    LAMBDA_OFFSET_SCHEDULE = cfg.get("LAMBDA_OFFSET_SCHEDULE", None)

    # Identifiability
    ident = cfg.get("IDENTIFIABILITY_REGIME", None)
    if isinstance(ident, str):
        s = ident.strip()
        if not s or s.lower() in ("none", "off", "false", "0"):
            ident = None
        else:
            ident = s

    # Model flavour tweaks
    if MODEL_NAME == "HybridAttn-NoPhysics":
        PDE_MODE = "none"
        LAMBDA_CONS = 0.0
        LAMBDA_GW = 0.0
        LAMBDA_PRIOR = 0.0
        LAMBDA_SMOOTH = 0.0
        LAMBDA_BOUNDS = 0.0
        LAMBDA_MV = 0.0
        MV_LR_MULT = 0.0
        KAPPA_LR_MULT = 0.0

    if MODEL_NAME == "PoroElasticSubsNet":
        PDE_MODE = "consolidation"
        LAMBDA_GW = 0.0

    # Censoring
    FEATURES = cfg.get("features", {}) or {}
    CENSOR = cfg.get("censoring", {}) or cfg.get("censor", {}) or {}
    CENSOR_SPECS = CENSOR.get("specs", []) or []
    CENSOR_THRESH = float(CENSOR.get("flag_threshold", 0.5))

    # Determine censor flag index.
    CENSOR_FLAG_IDX_DYN = None
    CENSOR_FLAG_IDX_FUT = None
    CENSOR_FLAG_NAME = None

    for sp in CENSOR_SPECS:
        cand = sp.get("flag_col")
        if not cand:
            base = sp.get("col")
            if base:
                cand = base + sp.get("flag_suffix", "_censored")
        if not cand:
            continue
        if cand in FUT_NAMES and CENSOR_FLAG_IDX_FUT is None:
            CENSOR_FLAG_IDX_FUT = FUT_NAMES.index(cand)
            CENSOR_FLAG_NAME = cand
        if cand in DYN_NAMES and CENSOR_FLAG_IDX_DYN is None:
            CENSOR_FLAG_IDX_DYN = DYN_NAMES.index(cand)
            CENSOR_FLAG_NAME = cand

    if CENSOR_FLAG_IDX_FUT is not None:
        CENSOR_MASK_SOURCE = "future"
        CENSOR_FLAG_IDX = CENSOR_FLAG_IDX_FUT
    elif CENSOR_FLAG_IDX_DYN is not None:
        CENSOR_MASK_SOURCE = "dynamic"
        CENSOR_FLAG_IDX = CENSOR_FLAG_IDX_DYN
    else:
        CENSOR_MASK_SOURCE = None
        CENSOR_FLAG_IDX = None

    # Column naming
    cols_cfg = cfg.get("cols", {}) or {}
    SUBSIDENCE_COL = cols_cfg.get("subsidence", "subsidence")
    GWL_COL = cols_cfg.get("gwl", "GWL")

    # Stage-1 dims
    dims = (
        M.get("artifacts", {})
        .get("sequences", {})
        .get("dims", {})
        or {}
    )
    OUT_S_DIM = int(dims.get("output_subsidence_dim", 1))
    OUT_G_DIM = int(dims.get("output_gwl_dim", 1))

    # Scalers
    encoders = M["artifacts"]["encoders"]
    scaler_info_dict = ctx.scaler_info
    if not scaler_info_dict:
        scaler_info_dict = load_scaler_info(encoders)

    # Attach loaded scaler objects if missing.
    if isinstance(scaler_info_dict, dict):
        for _, v in scaler_info_dict.items():
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

    def _pick_scaler_key(info, preferred, fallbacks=()):
        if not info:
            return preferred
        if preferred in info:
            return preferred
        for k in fallbacks:
            if k in info:
                return k
        low = {k.lower(): k for k in info.keys()}
        for token in ("subs", "subsidence"):
            for lk, orig in low.items():
                if token in lk:
                    return orig
        return preferred

    SUBS_SCALER_KEY = _pick_scaler_key(
        scaler_info_dict,
        preferred=SUBSIDENCE_COL,
        fallbacks=("subsidence", "subs_pred"),
    )
    GWL_SCALER_KEY = _pick_scaler_key(
        scaler_info_dict,
        preferred=GWL_COL,
        fallbacks=("gwl", "gwl_pred"),
    )

    # Coord scaler (optional)
    coord_scaler = None
    cs_path = encoders.get("coord_scaler")
    if cs_path and os.path.exists(cs_path):
        try:
            coord_scaler = joblib.load(cs_path)
        except Exception:
            coord_scaler = None

    # scaling kwargs (Stage-1 source of truth)
    sk_stage1 = (cfg.get("scaling_kwargs") or {}).copy()

    # Resolve driver channel indices.
    if "gwl_dyn_index" in sk_stage1 and sk_stage1["gwl_dyn_index"] is not None:
        GWL_DYN_INDEX = int(sk_stage1["gwl_dyn_index"])
        gwl_dyn_name = DYN_NAMES[GWL_DYN_INDEX]
    else:
        gwl_dyn_name = sk_stage1.get("gwl_dyn_name") or GWL_COL
        if gwl_dyn_name not in DYN_NAMES:
            for cand in (GWL_COL, "z_GWL", "gwl", "GWL"):
                if cand in DYN_NAMES:
                    gwl_dyn_name = cand
                    break
        GWL_DYN_INDEX = int(DYN_NAMES.index(gwl_dyn_name))

    Z_SURF_STATIC_INDEX = sk_stage1.get("z_surf_static_index")
    SUBS_DYN_INDEX = sk_stage1.get("subs_dyn_index")
    sub_dyn_name = sk_stage1.get("subs_dyn_name")

    # Coords normalization and degrees
    coords_normalized = bool(
        sk_stage1.get(
            "coords_normalized",
            sk_stage1.get("normalize_coords", False),
        )
    )
    coord_ranges = sk_stage1.get("coord_ranges") or None

    coords_in_degrees = bool(sk_stage1.get("coords_in_degrees", False))
    deg_to_m_lon = sk_stage1.get("deg_to_m_lon", None)
    deg_to_m_lat = sk_stage1.get("deg_to_m_lat", None)
    coord_order = sk_stage1.get("coord_order", ["t", "x", "y"])

    if coords_in_degrees and (
        deg_to_m_lon is None or deg_to_m_lat is None
    ):
        lat_ref_deg = sk_stage1.get("lat_ref_deg", None)
        if lat_ref_deg is not None and np.isfinite(float(lat_ref_deg)):
            deg_to_m_lon, deg_to_m_lat = deg_to_m_from_lat(
                float(lat_ref_deg)
            )
            sk_stage1["deg_to_m_lon"] = float(deg_to_m_lon)
            sk_stage1["deg_to_m_lat"] = float(deg_to_m_lat)

    # Thickness SI affine
    H_scale_si = sk_stage1.get("H_scale_si", None)
    H_bias_si = sk_stage1.get("H_bias_si", None)

    # Resolve SI affine maps for subs/head
    subs_scale_si = sk_stage1.get("subs_scale_si")
    subs_bias_si = sk_stage1.get("subs_bias_si")
    head_scale_si = sk_stage1.get("head_scale_si")
    head_bias_si = sk_stage1.get("head_bias_si")

    if subs_scale_si is None or subs_bias_si is None:
        subs_scale_si, subs_bias_si = resolve_si_affine(
            cfg,
            scaler_info_dict,
            target_name=SUBSIDENCE_COL,
            prefix="SUBS",
            unit_factor_key="SUBS_UNIT_TO_SI",
            scale_key="SUBS_SCALE_SI",
            bias_key="SUBS_BIAS_SI",
        )
    if head_scale_si is None or head_bias_si is None:
        head_scale_si, head_bias_si = resolve_si_affine(
            cfg,
            scaler_info_dict,
            target_name=GWL_COL,
            prefix="HEAD",
            unit_factor_key="HEAD_UNIT_TO_SI",
            scale_key="HEAD_SCALE_SI",
            bias_key="HEAD_BIAS_SI",
        )

    # Physics bounds
    PHYS_BOUNDS_CFG = cfg.get("PHYSICS_BOUNDS", {}) or {}
    BOUNDS_MODE = str(cfg.get("PHYSICS_BOUNDS_MODE", "soft") or "soft")
    BOUNDS_MODE = BOUNDS_MODE.strip().lower()
    if BOUNDS_MODE in ("off", "none"):
        BOUNDS_MODE = "off"

    default_phys_bounds = {
        "H_min": 5.0,
        "H_max": 80.0,
        "K_min": 1e-8,
        "K_max": 1e-3,
        "Ss_min": 1e-7,
        "Ss_max": 1e-3,
        "tau_min": 7.0 * 86400.0,
        "tau_max": 300.0 * 31556952.0,
    }
    phys_bounds = dict(default_phys_bounds)
    phys_bounds.update(PHYS_BOUNDS_CFG)

    bounds_for_scaling = {
        "H_min": float(phys_bounds["H_min"]),
        "H_max": float(phys_bounds["H_max"]),
        "K_min": float(phys_bounds["K_min"]),
        "K_max": float(phys_bounds["K_max"]),
        "Ss_min": float(phys_bounds["Ss_min"]),
        "Ss_max": float(phys_bounds["Ss_max"]),
        "tau_min": float(phys_bounds["tau_min"]),
        "tau_max": float(phys_bounds["tau_max"]),
        "logK_min": float(np.log(phys_bounds["K_min"])),
        "logK_max": float(np.log(phys_bounds["K_max"])),
        "logSs_min": float(np.log(phys_bounds["Ss_min"])),
        "logSs_max": float(np.log(phys_bounds["Ss_max"])),
        "logTau_min": float(np.log(phys_bounds["tau_min"])),
        "logTau_max": float(np.log(phys_bounds["tau_max"])),
    }

    # GeoPrior parameters
    GEOPRIOR_INIT_MV = float(cfg.get("GEOPRIOR_INIT_MV", 1e-7))
    GEOPRIOR_INIT_KAPPA = float(cfg.get("GEOPRIOR_INIT_KAPPA", 1.0))
    GEOPRIOR_GAMMA_W = float(cfg.get("GEOPRIOR_GAMMA_W", 9810.0))
    GEOPRIOR_H_REF = cfg.get("GEOPRIOR_H_REF", 0.0)
    GEOPRIOR_KAPPA_MODE = cfg.get("GEOPRIOR_KAPPA_MODE", "bar")
    GEOPRIOR_USE_EFF_H = bool(
        cfg.get("GEOPRIOR_USE_EFFECTIVE_H", True)
    )
    GEOPRIOR_HD_FACTOR = float(cfg.get("GEOPRIOR_HD_FACTOR", 0.6))

    GEOPRIOR_H_REF_VALUE = 0.0
    GEOPRIOR_H_REF_MODE = None
    if isinstance(GEOPRIOR_H_REF, (int, float)):
        GEOPRIOR_H_REF_VALUE = float(GEOPRIOR_H_REF)
    else:
        GEOPRIOR_H_REF_MODE = GEOPRIOR_H_REF

    # Training strategy gates
    TRAINING_STRATEGY = str(
        cfg.get("TRAINING_STRATEGY", "data_first")
    ).strip().lower()

    # Compute steps_per_epoch
    X_train_norm = ensure_input_shapes(
        X_train,
        mode=MODE,
        forecast_horizon=FORECAST_H,
    )
    n_train = int(X_train_norm["static_features"].shape[0])
    steps_per_epoch = int(np.ceil(n_train / float(BATCH_SIZE)))

    # Gate policies (reusing sensitivity.py logic)
    q_policy = "always_on"
    q_warmup_epochs = 0
    q_ramp_epochs = 0

    subs_resid_policy = "always_on"
    subs_resid_warmup_epochs = 0
    subs_resid_ramp_epochs = 0

    if TRAINING_STRATEGY == "physics_first":
        q_policy = str(
            cfg.get("Q_POLICY_PHYSICS_FIRST", "warmup_off")
        ).strip().lower()
        q_warmup_epochs = int(
            cfg.get("Q_WARMUP_EPOCHS_PHYSICS_FIRST", 5)
        )
        q_ramp_epochs = int(
            cfg.get("Q_RAMP_EPOCHS_PHYSICS_FIRST", 0)
        )

        subs_resid_policy = str(
            cfg.get(
                "SUBS_RESID_POLICY_PHYSICS_FIRST",
                "warmup_off",
            )
        ).strip().lower()
        subs_resid_warmup_epochs = int(
            cfg.get(
                "SUBS_RESID_WARMUP_EPOCHS_PHYSICS_FIRST",
                5,
            )
        )
        subs_resid_ramp_epochs = int(
            cfg.get(
                "SUBS_RESID_RAMP_EPOCHS_PHYSICS_FIRST",
                0,
            )
        )
        LAMBDA_Q = float(cfg.get("LAMBDA_Q_PHYSICS_FIRST", LAMBDA_Q))
        LOSS_WEIGHT_GWL = float(
            cfg.get("LOSS_WEIGHT_GWL_PHYSICS_FIRST", LOSS_WEIGHT_GWL)
        )

    else:
        LOSS_WEIGHT_GWL = float(
            cfg.get("LOSS_WEIGHT_GWL_DATA_FIRST", LOSS_WEIGHT_GWL)
        )
        LAMBDA_Q = float(cfg.get("LAMBDA_Q_DATA_FIRST", LAMBDA_Q))

        q_policy = str(
            cfg.get("Q_POLICY_DATA_FIRST", "always_on")
        ).strip().lower()
        q_warmup_epochs = int(
            cfg.get("Q_WARMUP_EPOCHS_DATA_FIRST", 0)
        )
        q_ramp_epochs = int(cfg.get("Q_RAMP_EPOCHS_DATA_FIRST", 0))

        subs_resid_policy = str(
            cfg.get("SUBS_RESID_POLICY_DATA_FIRST", "always_on")
        ).strip().lower()
        subs_resid_warmup_epochs = int(
            cfg.get("SUBS_RESID_WARMUP_EPOCHS_DATA_FIRST", 0)
        )
        subs_resid_ramp_epochs = int(
            cfg.get("SUBS_RESID_RAMP_EPOCHS_DATA_FIRST", 0)
        )

    if q_policy == "always_off":
        LAMBDA_Q = 0.0

    q_warmup_steps = max(0, q_warmup_epochs) * steps_per_epoch
    q_ramp_steps = max(0, q_ramp_epochs) * steps_per_epoch

    subs_resid_warmup_steps = (
        max(0, subs_resid_warmup_epochs) * steps_per_epoch
    )
    subs_resid_ramp_steps = (
        max(0, subs_resid_ramp_epochs) * steps_per_epoch
    )

    # MV prior schedule
    MV_PRIOR_MODE = str(
        sk_stage1.get(
            "mv_prior_mode",
            cfg.get("MV_PRIOR_MODE", "calibrate"),
        )
    )
    MV_WEIGHT = float(
        sk_stage1.get("mv_weight", cfg.get("MV_WEIGHT", 1e-3))
    )
    MV_SCHEDULE_UNIT = str(
        sk_stage1.get(
            "mv_schedule_unit",
            cfg.get("MV_SCHEDULE_UNIT", "epoch"),
        )
    ).strip().lower()

    MV_DELAY_EPOCHS = int(
        sk_stage1.get("mv_delay_epochs", cfg.get("MV_DELAY_EPOCHS", 1))
    )
    MV_WARMUP_EPOCHS = int(
        sk_stage1.get(
            "mv_warmup_epochs",
            cfg.get("MV_WARMUP_EPOCHS", 2),
        )
    )

    mv_delay_steps = sk_stage1.get("mv_delay_steps", None)
    mv_warmup_steps = sk_stage1.get("mv_warmup_steps", None)

    if mv_delay_steps is None:
        mv_delay_steps = max(0, MV_DELAY_EPOCHS) * steps_per_epoch
    if mv_warmup_steps is None:
        mv_warmup_steps = max(0, MV_WARMUP_EPOCHS) * steps_per_epoch

    # ---------------------------
    # Logging header
    # ---------------------------
    device_info = configure_tf_from_cfg(cfg)

    config_sections = [
        (
            "Run",
            {
                "CITY_NAME": CITY_NAME,
                "MODEL_NAME": MODEL_NAME,
                "MANIFEST_PATH": ctx.manifest_path,
                "RUN_OUTPUT_PATH": run_dir,
            },
        ),
        (
            "Architecture",
            {
                "TIME_STEPS": TIME_STEPS,
                "FORECAST_HORIZON_YEARS": FORECAST_H,
                "MODE": MODE,
                "ATTENTION_LEVELS": ATTENTION_LEVELS,
                "EMBED_DIM": EMBED_DIM,
                "HIDDEN_UNITS": HIDDEN_UNITS,
                "LSTM_UNITS": LSTM_UNITS,
                "ATTENTION_UNITS": ATTENTION_UNITS,
                "NUMBER_HEADS": NUMBER_HEADS,
                "DROPOUT_RATE": DROPOUT_RATE,
                "MEMORY_SIZE": MEMORY_SIZE,
                "SCALES": SCALES,
            },
        ),
        (
            "Physics",
            {
                "PDE_MODE_CONFIG": PDE_MODE,
                "TIME_UNITS": TIME_UNITS,
                "LAMBDA_CONS": LAMBDA_CONS,
                "LAMBDA_GW": LAMBDA_GW,
                "LAMBDA_PRIOR": LAMBDA_PRIOR,
                "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
                "LAMBDA_BOUNDS": LAMBDA_BOUNDS,
                "LAMBDA_MV": LAMBDA_MV,
                "LAMBDA_Q": LAMBDA_Q,
                "LOSS_WEIGHT_GWL": LOSS_WEIGHT_GWL,
                "MV_LR_MULT": MV_LR_MULT,
                "KAPPA_LR_MULT": KAPPA_LR_MULT,
                "PHYSICS_BOUNDS": phys_bounds,
            },
        ),
        (
            "Training",
            {
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "LEARNING_RATE": LEARNING_RATE,
                "QUANTILES": QUANTILES,
            },
        ),
    ]

    print_config_table(
        config_sections,
        table_width=get_table_size(),
        title=(
            f"{CITY_NAME.upper()} "
            f"{MODEL_NAME} "
            "SENS RUN"
        ),
    )

    print("\nTraining outputs ->", run_dir)

    # ---------------------------
    # Build & compile model
    # ---------------------------
    s_dim_model = int(X_train_norm["static_features"].shape[-1])
    d_dim_model = int(X_train_norm["dynamic_features"].shape[-1])
    f_dim_model = int(X_train_norm["future_features"].shape[-1])

    MODEL_CLASS_REGISTRY = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "PoroElasticSubsNet": PoroElasticSubsNet,
        "HybridAttn-NoPhysics": GeoPriorSubsNet,
    }
    model_cls = MODEL_CLASS_REGISTRY.get(MODEL_NAME, GeoPriorSubsNet)

    sk_model = dict(sk_stage1)
    sk_model.update(
        {
            "bounds": bounds_for_scaling,
            "time_units": TIME_UNITS,
            "coords_normalized": coords_normalized,
            "coord_ranges": coord_ranges or {},
            "coord_order": coord_order,
            "coords_in_degrees": coords_in_degrees,
            "deg_to_m_lon": (
                float(deg_to_m_lon)
                if deg_to_m_lon is not None
                else None
            ),
            "deg_to_m_lat": (
                float(deg_to_m_lat)
                if deg_to_m_lat is not None
                else None
            ),
            "H_scale_si": (
                float(H_scale_si)
                if H_scale_si is not None
                else 1.0
            ),
            "H_bias_si": (
                float(H_bias_si)
                if H_bias_si is not None
                else 0.0
            ),
            "dynamic_feature_names": list(DYN_NAMES),
            "future_feature_names": list(FUT_NAMES),
            "static_feature_names": list(STA_NAMES),
            "gwl_dyn_name": gwl_dyn_name,
            "gwl_dyn_index": int(GWL_DYN_INDEX),
            "z_surf_static_index": (
                int(Z_SURF_STATIC_INDEX)
                if Z_SURF_STATIC_INDEX is not None
                else None
            ),
            "subs_dyn_index": (
                int(SUBS_DYN_INDEX)
                if SUBS_DYN_INDEX is not None
                else None
            ),
            "subs_dyn_name": (
                sub_dyn_name
                if sub_dyn_name is not None
                else SUBSIDENCE_COL
            ),
            "subs_scale_si": subs_scale_si,
            "subs_bias_si": subs_bias_si,
            "head_scale_si": head_scale_si,
            "head_bias_si": head_bias_si,
            "training_strategy": TRAINING_STRATEGY,
            "q_policy": q_policy,
            "q_warmup_epochs": int(q_warmup_epochs),
            "q_ramp_epochs": int(q_ramp_epochs),
            "q_warmup_steps": int(q_warmup_steps),
            "q_ramp_steps": int(q_ramp_steps),
            "log_q_diagnostics": bool(LOG_Q_DIAGNOSTICS),
            "subs_resid_policy": subs_resid_policy,
            "subs_resid_warmup_epochs": int(subs_resid_warmup_epochs),
            "subs_resid_ramp_epochs": int(subs_resid_ramp_epochs),
            "subs_resid_warmup_steps": int(subs_resid_warmup_steps),
            "subs_resid_ramp_steps": int(subs_resid_ramp_steps),
            "loss_weight_gwl": float(LOSS_WEIGHT_GWL),
            "lambda_q": float(LAMBDA_Q),
            "mv_prior_mode": MV_PRIOR_MODE,
            "mv_weight": float(MV_WEIGHT),
            "mv_schedule_unit": MV_SCHEDULE_UNIT,
            "mv_delay_epochs": int(MV_DELAY_EPOCHS),
            "mv_warmup_epochs": int(MV_WARMUP_EPOCHS),
            "mv_delay_steps": int(mv_delay_steps),
            "mv_warmup_steps": int(mv_warmup_steps),
            "mv_steps_per_epoch": int(steps_per_epoch),
            "identifiability_regime": ident,
        }
    )

    sk_model = finalize_scaling_kwargs(sk_model)

    sk_model = override_scaling_kwargs(
        sk_model,
        cfg,
        finalize= finalize_scaling_kwargs,
        dyn_names=DYN_NAMES,
        gwl_dyn_index=GWL_DYN_INDEX,
        base_dir=os.path.dirname(__file__),
        strict=True,
        log_fn=print,
    )

    # Keep it on disk for audit.
    with open(
        os.path.join(run_dir, "scaling_kwargs.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(sk_model, f, indent=2)

    subsmodel_params = {
        "embed_dim": EMBED_DIM,
        "hidden_units": HIDDEN_UNITS,
        "lstm_units": LSTM_UNITS,
        "attention_units": ATTENTION_UNITS,
        "num_heads": NUMBER_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "max_window_size": TIME_STEPS,
        "memory_size": MEMORY_SIZE,
        "scales": SCALES,
        "multi_scale_agg": "last",
        "final_agg": "last",
        "use_residuals": USE_RESIDUALS,
        "use_batch_norm": USE_BATCH_NORM,
        "use_vsn": USE_VSN,
        "vsn_units": VSN_UNITS,
        "mode": MODE,
        "attention_levels": ATTENTION_LEVELS,
        "scale_pde_residuals": SCALE_PDE_RESIDUALS,
        "scaling_kwargs": sk_model,
        "bounds_mode": BOUNDS_MODE,
        "mv": LearnableMV(initial_value=GEOPRIOR_INIT_MV),
        "kappa": LearnableKappa(
            initial_value=GEOPRIOR_INIT_KAPPA
        ),
        "gamma_w": FixedGammaW(value=GEOPRIOR_GAMMA_W),
        "h_ref": FixedHRef(
            value=GEOPRIOR_H_REF_VALUE,
            mode=GEOPRIOR_H_REF_MODE,
        ),
        "kappa_mode": GEOPRIOR_KAPPA_MODE,
        "use_effective_h": GEOPRIOR_USE_EFF_H,
        "hd_factor": GEOPRIOR_HD_FACTOR,
        "offset_mode": OFFSET_MODE,
        "residual_method": CONS_RESID_METHOD,
        "time_units": TIME_UNITS,
    }

    if should_audit(cfg.get("AUDIT_STAGES"), stage="stage2"):
        _ = audit_stage2_handshake(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            time_steps=TIME_STEPS,
            forecast_horizon=FORECAST_H,
            mode=MODE,
            dyn_names=list(DYN_NAMES),
            fut_names=list(FUT_NAMES),
            sta_names=list(STA_NAMES),
            coord_scaler=coord_scaler,
            sk_final=sk_model,
            save_dir=run_dir,
            table_width=get_table_size(),
            title_prefix="STAGE-2 HANDSHAKE AUDIT",
            city=CITY_NAME,
            model_name=MODEL_NAME,
        )

    subs_model_inst = model_cls(
        static_input_dim=s_dim_model,
        dynamic_input_dim=d_dim_model,
        future_input_dim=f_dim_model,
        output_subsidence_dim=OUT_S_DIM,
        output_gwl_dim=OUT_G_DIM,
        forecast_horizon=FORECAST_H,
        quantiles=QUANTILES,
        pde_mode=PDE_MODE,
        identifiability_regime=ident,
        verbose=0,
        **subsmodel_params,
    )

    # Build outputs once.
    for xb, _ in train_dataset.take(1):
        subs_model_inst(xb)
        break

    # Losses
    loss_dict = {
        "subs_pred": (
            make_weighted_pinball(QUANTILES, SUBS_WEIGHTS)
            if QUANTILES
            else tf.keras.losses.MSE
        ),
        "gwl_pred": (
            make_weighted_pinball(QUANTILES, GWL_WEIGHTS)
            if QUANTILES
            else tf.keras.losses.MSE
        ),
    }

    # Compile-time metrics (only if not tracked internally)
    TRACK_AUX_METRICS = bool(
        cfg.get(
            "TRACK_AUX_METRICS",
            cfg.get("TRACK_ADD_ON_METRICS", True),
        )
    )

    if TRACK_AUX_METRICS:
        metrics_arg = None
    else:
        if QUANTILES:
            metrics_arg = {
                "subs_pred": [
                    MAEQ50(name="mae_q50"),
                    MSEQ50(name="mse_q50"),
                    Coverage80(name="coverage80"),
                    Sharpness80(name="sharpness80"),
                ],
                "gwl_pred": [
                    MAEQ50(name="mae_q50"),
                    MSEQ50(name="mse_q50"),
                ],
            }
        else:
            metrics_arg = {
                "subs_pred": ["mae", "mse"],
                "gwl_pred": ["mae", "mse"],
            }

    physics_loss_weights = {
        "lambda_cons": LAMBDA_CONS,
        "lambda_gw": LAMBDA_GW,
        "lambda_prior": LAMBDA_PRIOR,
        "lambda_smooth": LAMBDA_SMOOTH,
        "lambda_bounds": LAMBDA_BOUNDS,
        "lambda_mv": LAMBDA_MV,
        "mv_lr_mult": MV_LR_MULT,
        "lambda_offset": LAMBDA_OFFSET,
        "kappa_lr_mult": KAPPA_LR_MULT,
        "lambda_q": float(LAMBDA_Q),
    }

    loss_weights_dict = {
        "subs_pred": 1.0,
        "gwl_pred": float(LOSS_WEIGHT_GWL),
    }

    out_names = (
        list(getattr(subs_model_inst, "output_names", []))
        or ["subs_pred", "gwl_pred"]
    )

    import keras

    is_keras2 = keras.__version__.startswith("2.")
    if is_keras2:
        loss_arg = [loss_dict[k] for k in out_names]
        lossw_arg = [
            loss_weights_dict.get(k, 1.0) for k in out_names
        ]
        metrics_compile = (
            None
            if metrics_arg is None
            else [metrics_arg.get(k, []) for k in out_names]
        )
    else:
        loss_arg = loss_dict
        lossw_arg = loss_weights_dict
        metrics_compile = metrics_arg

    subs_model_inst.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        loss=loss_arg,
        loss_weights=lossw_arg,
        metrics=metrics_compile,
        **physics_loss_weights,
    )

    # ---------------------------
    # Training
    # ---------------------------
    bundle_prefix = f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_H}"

    best_keras_path = os.path.join(
        run_dir,
        f"{bundle_prefix}_best.keras",
    )
    best_weights_path = os.path.join(
        run_dir,
        f"{bundle_prefix}_best.weights.h5",
    )
    best_tf_dir = os.path.join(
        run_dir,
        f"{bundle_prefix}_best_savedmodel",
    )

    model_init_manifest_path = os.path.join(
        run_dir,
        "model_init_manifest.json",
    )

    model_init_manifest = {
        "model_class": model_cls.__name__,
        "dims": {
            "static_input_dim": int(s_dim_model),
            "dynamic_input_dim": int(d_dim_model),
            "future_input_dim": int(f_dim_model),
            "output_subsidence_dim": int(OUT_S_DIM),
            "output_gwl_dim": int(OUT_G_DIM),
            "forecast_horizon": int(FORECAST_H),
        },
        "config": {
            "quantiles": list(QUANTILES) if QUANTILES else None,
            "pde_mode": PDE_MODE,
            "mode": MODE,
            "time_units": TIME_UNITS,
            "embed_dim": EMBED_DIM,
            "hidden_units": HIDDEN_UNITS,
            "lstm_units": LSTM_UNITS,
            "attention_units": ATTENTION_UNITS,
            "num_heads": NUMBER_HEADS,
            "dropout_rate": DROPOUT_RATE,
            "memory_size": MEMORY_SIZE,
            "scales": list(SCALES) if SCALES else None,
            "use_residuals": bool(USE_RESIDUALS),
            "use_batch_norm": bool(USE_BATCH_NORM),
            "use_vsn": bool(USE_VSN),
            "vsn_units": int(VSN_UNITS),
            "geoprior": {
                "init_mv": float(GEOPRIOR_INIT_MV),
                "init_kappa": float(GEOPRIOR_INIT_KAPPA),
                "gamma_w": float(GEOPRIOR_GAMMA_W),
                "h_ref_value": float(GEOPRIOR_H_REF_VALUE),
                "h_ref_mode": GEOPRIOR_H_REF_MODE,
                "kappa_mode": GEOPRIOR_KAPPA_MODE,
                "use_effective_h": bool(GEOPRIOR_USE_EFF_H),
                "hd_factor": float(GEOPRIOR_HD_FACTOR),
                "offset_mode": OFFSET_MODE,
            },
            "scaling_kwargs": sk_model,
            "identifiability_regime": ident,
        },
    }

    save_manifest(model_init_manifest_path, model_init_manifest)

    callbacks = [
        ModelCheckpoint(
            filepath=best_keras_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    disable_es = bool(cfg.get("DISABLE_EARLY_STOPPING", False))
    if not disable_es:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            )
        )

    csvlog_path = os.path.join(
        run_dir,
        f"{CITY_NAME}_{MODEL_NAME}_train_log.csv",
    )
    callbacks.append(CSVLogger(csvlog_path, append=False))
    callbacks.append(TerminateOnNaN())

    if USE_LAMBDA_OFFSET_SCHED and (not subs_model_inst._physics_off()):
        callbacks.append(
            LambdaOffsetScheduler(
                schedule=LAMBDA_OFFSET_SCHEDULE,
                unit=LAMBDA_OFFSET_UNIT,
                when=LAMBDA_OFFSET_WHEN,
                warmup=LAMBDA_OFFSET_WARMUP,
                start=LAMBDA_OFFSET_START,
                end=LAMBDA_OFFSET_END,
                clamp_positive=True,
                verbose=1,
            )
        )

    print("\nTraining...")
    history = subs_model_inst.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=VERBOSE,
    )

    # Save training summary
    best_epoch, metrics_at_best = best_epoch_and_metrics(history.history)

    training_summary = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "city": CITY_NAME,
        "model": MODEL_NAME,
        "horizon": int(FORECAST_H),
        "best_epoch": (
            int(best_epoch) if best_epoch is not None else None
        ),
        "metrics_at_best": metrics_at_best,
        "final_epoch_metrics": {
            k: float(v[-1]) for k, v in history.history.items()
            if len(v)
        },
        "env": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": device_info,
        },
        "compile": {
            "optimizer": "Adam",
            "learning_rate": float(LEARNING_RATE),
            "loss_weights": loss_weights_dict,
            "metrics": (
                {
                    k: [name_of(m) for m in v]
                    for k, v in metrics_arg.items()
                }
                if metrics_arg
                else {}
            ),
            "physics_loss_weights": physics_loss_weights,
            "lambda_offset": LAMBDA_OFFSET,
        },
        "hp_init": {
            "quantiles": QUANTILES,
            "subs_weights": SUBS_WEIGHTS,
            "gwl_weights": GWL_WEIGHTS,
            "attention_levels": ATTENTION_LEVELS,
            "pde_mode": PDE_MODE,
            "time_steps": int(TIME_STEPS),
            "mode": MODE,
            "model_init_params": serialize_subs_params(
                subsmodel_params,
                cfg,
            ),
            "offset_mode": OFFSET_MODE,
            "scaling_kwargs": {
                "bounds": bounds_for_scaling,
                "time_units": TIME_UNITS,
                "coords_normalized": coords_normalized,
                "coord_ranges": coord_ranges or {},
            },
            "identifiability_regime": ident,
        },
        "paths": {
            "run_dir": run_dir,
            "csv_log": csvlog_path,
            "best_keras": best_keras_path,
            "best_weights": best_weights_path,
            "model_init_manifest": model_init_manifest_path,
        },
    }

    summary_json_path = os.path.join(
        run_dir,
        f"{CITY_NAME}_{MODEL_NAME}_training_summary.json",
    )
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    # Save final model
    final_model_path = os.path.join(
        run_dir,
        f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_H}_final.keras",
    )
    try:
        subs_model_inst.save(final_model_path)
        training_summary["paths"]["final_keras"] = final_model_path
    except Exception as e:
        print("[Warn] Final save failed:", e)

    # Plots (optional)
    if not FAST_SENS:
        history_groups = {
            "Total Loss": ["total_loss"],
            "Data vs Physics": [
                "data_loss",
                "physics_loss_scaled",
                "physics_loss",
            ],
            "Offset Controls": [
                "lambda_offset",
                "physics_mult",
            ],
            "Physics Components": [
                "consolidation_loss",
                "gw_flow_loss",
                "prior_loss",
                "smooth_loss",
                "mv_prior_loss",
                "bounds_loss",
            ],
            "Subsidence MAE": ["subs_pred_mae"],
            "GWL MAE": ["gwl_pred_mae"],
        }
        yscales = {
            "Total Loss": "log",
            "Data vs Physics": "log",
            "Physics Components": "log",
            "Offset Controls": "linear",
            "Subsidence MAE": "linear",
            "GWL MAE": "linear",
        }
        plot_history_in(
            history.history,
            metrics=history_groups,
            title=f"{MODEL_NAME} Training History",
            yscale_settings=yscales,
            layout="subplots",
            savefig=os.path.join(
                run_dir,
                f"{CITY_NAME}_{MODEL_NAME.lower()}_history.png",
            ),
        )
        autoplot_geoprior_history(
            history,
            outdir=run_dir,
            prefix=bundle_prefix,
            style="default",
            log_fn=print,
        )

    # Physical parameters CSV
    phys_model_tag = "geoprior"
    if MODEL_NAME == "PoroElasticSubsNet":
        phys_model_tag = "poroelastic"
    elif MODEL_NAME.startswith("HybridAttn"):
        phys_model_tag = "hybridattn"

    extract_physical_parameters(
        subs_model_inst,
        to_csv=True,
        filename=(
            f"{CITY_NAME}_{MODEL_NAME.lower()}_"
            "physical_parameters.csv"
        ),
        save_dir=run_dir,
        model_name=phys_model_tag,
    )

    # ---------------------------
    # Inference model
    # ---------------------------
    USE_IN_MEMORY_MODEL = bool(
        cfg.get("USE_IN_MEMORY_MODEL", True)
    )
    USE_TF_SAVEDMODEL = bool(cfg.get("USE_TF_SAVEDMODEL", False))

    build_inputs = None
    for xb, _ in val_dataset.take(1):
        build_inputs = xb
        break

    def builder(manifest: dict):
        dims2 = (manifest or {}).get("dims", {}) or {}
        cfgm = (manifest or {}).get("config", {}) or {}
        gp = (cfgm.get("geoprior", {}) or {})

        _subsparams = dict(subsmodel_params)
        _subsparams.update(
            {
                "mv": LearnableMV(
                    initial_value=float(
                        gp.get("init_mv", GEOPRIOR_INIT_MV)
                    )
                ),
                "kappa": LearnableKappa(
                    initial_value=float(
                        gp.get(
                            "init_kappa",
                            GEOPRIOR_INIT_KAPPA,
                        )
                    )
                ),
                "gamma_w": FixedGammaW(
                    value=float(
                        gp.get("gamma_w", GEOPRIOR_GAMMA_W)
                    )
                ),
                "h_ref": FixedHRef(
                    value=float(
                        gp.get(
                            "h_ref_value",
                            GEOPRIOR_H_REF_VALUE,
                        )
                    ),
                    mode=gp.get(
                        "h_ref_mode",
                        GEOPRIOR_H_REF_MODE,
                    ),
                ),
            }
        )

        return model_cls(
            static_input_dim=int(
                dims2.get("static_input_dim", s_dim_model)
            ),
            dynamic_input_dim=int(
                dims2.get("dynamic_input_dim", d_dim_model)
            ),
            future_input_dim=int(
                dims2.get("future_input_dim", f_dim_model)
            ),
            output_subsidence_dim=int(
                dims2.get("output_subsidence_dim", OUT_S_DIM)
            ),
            output_gwl_dim=int(
                dims2.get("output_gwl_dim", OUT_G_DIM)
            ),
            forecast_horizon=int(
                dims2.get("forecast_horizon", FORECAST_H)
            ),
            quantiles=cfgm.get("quantiles", QUANTILES),
            pde_mode=cfgm.get("pde_mode", PDE_MODE),
            identifiability_regime=ident,
            verbose=0,
            **_subsparams,
        )

    save_model(
        model=subs_model_inst,
        keras_path=(best_tf_dir if USE_TF_SAVEDMODEL else best_keras_path),
        weights_path=best_weights_path,
        manifest_path=model_init_manifest_path,
        manifest=model_init_manifest,
        overwrite=True,
        use_tf_format=USE_TF_SAVEDMODEL,
    )

    if USE_IN_MEMORY_MODEL:
        model_inf = subs_model_inst
        print("[Info] Using in-memory model.")
    elif USE_TF_SAVEDMODEL:
        model_inf = load_model_from_tfv2(
            best_tf_dir,
            endpoint="serve",
            custom_objects={
                "GeoPriorSubsNet": GeoPriorSubsNet,
                "PoroElasticSubsNet": PoroElasticSubsNet,
                "LearnableMV": LearnableMV,
                "LearnableKappa": LearnableKappa,
                "FixedGammaW": FixedGammaW,
                "FixedHRef": FixedHRef,
                "make_weighted_pinball": make_weighted_pinball,
            },
        )
    else:
        model_inf = load_inference_model(
            keras_path=best_keras_path,
            weights_path=best_weights_path,
            manifest_path=model_init_manifest_path,
            custom_objects={
                "GeoPriorSubsNet": GeoPriorSubsNet,
                "PoroElasticSubsNet": PoroElasticSubsNet,
                "LearnableMV": LearnableMV,
                "LearnableKappa": LearnableKappa,
                "FixedGammaW": FixedGammaW,
                "FixedHRef": FixedHRef,
                "make_weighted_pinball": make_weighted_pinball,
            },
            compile=False,
            builder=builder,
            build_inputs=build_inputs,
            prefer_full_model=True,
            log_fn=print,
            use_in_memory_model=False,
        )

    if DEBUG and (model_inf is not subs_model_inst):
        _ = debug_model_reload(
            subs_model_inst,
            model_inf,
            val_dataset,
            pred_key="subs_pred",
            also_check=["subs_pred", "gwl_pred"],
            top_weights=30,
            log_fn=print,
        )

    # ---------------------------
    # Interval calibrator
    # ---------------------------
    cal80 = None
    val_for_cal = val_dataset
    if cal_max_batches is not None:
        val_for_cal = val_dataset.take(int(cal_max_batches))

    if QUANTILES:
        print("\nFit interval calibrator (80%)...")
        cal80 = fit_interval_calibrator_on_val(
            model_inf,
            val_for_cal,
            target=0.80,
            q_values=QUANTILES,
        )
        np.save(
            os.path.join(run_dir, "interval_factors_80.npy"),
            cal80.factors_,
        )

    # ---------------------------
    # Forecasting split
    # ---------------------------
    dataset_name_for_forecast = "ValidationSet_Fallback"
    if X_test is not None and y_test is not None:
        X_fore, y_fore = X_test, y_test
        dataset_name_for_forecast = "TestSet"
    else:
        X_fore, y_fore = X_val, y_val

    X_fore = ensure_input_shapes(
        X_fore,
        mode=MODE,
        forecast_horizon=FORECAST_H,
    )
    # y_fore_fmt = map_targets_for_training(y_fore)
    
    def _slice_any(x: Any, n: int) -> Any:
        """Slice nested inputs/targets along the sample axis."""
        if x is None:
            return None
        if isinstance(x, dict):
            return {k: _slice_any(v, n) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(_slice_any(v, n) for v in x)
        try:
            return x[:n]
        except Exception:
            return x
    
    # Keep forecasting tensors and ds_eval consistent when we
    # limit evaluation/export to a subset of batches.
    if eval_max_batches is not None:
        n_take = int(eval_max_batches) * int(BATCH_SIZE)
        X_fore = _slice_any(X_fore, n_take)
        y_fore = _slice_any(y_fore, n_take)
        dataset_name_for_forecast += f"_Take{int(eval_max_batches)}B"
    
    y_fore_fmt = map_targets_for_training(y_fore)

    print(f"\nPredicting on {dataset_name_for_forecast}...")
    pred_out = model_inf.predict(X_fore, verbose=0)
    pred_dict = normalize_predict_output(
        model_inf,
        x=X_fore,
        pred_out=pred_out,
        required=("subs_pred", "gwl_pred"),
        batch_n=32,
        log_fn=print if DEBUG else None,
    )

    s_pred = pred_dict["subs_pred"]
    h_pred = pred_dict["gwl_pred"]

    if QUANTILES:
        if cal80 is not None:
            s_pred_cal = apply_calibrator_to_subs(
                cal80,
                s_pred,
                q_values=QUANTILES,
            )
        else:
            s_pred_cal = s_pred
        predictions_for_formatter = {
            "subs_pred": s_pred_cal,
            "gwl_pred": h_pred,
        }
    else:
        predictions_for_formatter = {
            "subs_pred": s_pred,
            "gwl_pred": h_pred,
        }

    # Point metrics
    ev_point = evaluate_point_forecast(
        model_inf,
        predictions_for_formatter,
        y_true_subs=y_fore_fmt["subs_pred"],
        y_true_gwl=y_fore_fmt.get("gwl_pred"),
        n_q=(len(QUANTILES) if QUANTILES else 3),
        quantiles=(QUANTILES if QUANTILES else None),
        use_physical=True,
        scaler_info=scaler_info_dict,
        subs_target_name=SUBS_SCALER_KEY,
        gwl_target_name=GWL_SCALER_KEY,
    )

    metrics_point = ev_point.get("subs_metrics", {})
    per_h_mae_dict = ev_point.get("subs_mae_h", {})
    per_h_r2_dict = ev_point.get("subs_r2_h", {})

    # CSV outputs
    csv_eval = os.path.join(
        run_dir,
        (
            f"{CITY_NAME}_{MODEL_NAME}_forecast_"
            f"{dataset_name_for_forecast}_H"
            f"{FORECAST_H}_eval.csv"
        ),
    )
    csv_future = os.path.join(
        run_dir,
        (
            f"{CITY_NAME}_{MODEL_NAME}_forecast_"
            f"{dataset_name_for_forecast}_H"
            f"{FORECAST_H}_future.csv"
        ),
    )
    csv_eval_cal = os.path.join(
        run_dir,
        (
            f"{CITY_NAME}_{MODEL_NAME}_forecast_"
            f"{dataset_name_for_forecast}_H"
            f"{FORECAST_H}_eval_calibrated.csv"
        ),
    )
    csv_future_cal = os.path.join(
        run_dir,
        (
            f"{CITY_NAME}_{MODEL_NAME}_forecast_"
            f"{dataset_name_for_forecast}_H"
            f"{FORECAST_H}_future_calibrated.csv"
        ),
    )
    cal_stats_path = os.path.join(
        run_dir,
        (
            f"{CITY_NAME}_{MODEL_NAME}_calibration_stats_"
            f"{dataset_name_for_forecast}_H"
            f"{FORECAST_H}.json"
        ),
    )

    metrics_json = os.path.join(
        run_dir,
        (
            f"{CITY_NAME}_{MODEL_NAME}_eval_diagnostics_"
            f"{dataset_name_for_forecast}_H{FORECAST_H}.json"
        ),
    )

    future_grid = np.arange(
        FORECAST_START_YEAR,
        FORECAST_START_YEAR + FORECAST_H,
        dtype=float,
    )

    y_true_for_format = {
        "subsidence": y_fore_fmt["subs_pred"],
        "gwl": y_fore_fmt.get("gwl_pred"),
    }

    df_eval, df_future = format_and_forecast(
        y_pred=predictions_for_formatter,
        y_true=y_true_for_format,
        coords=X_fore.get("coords", None),
        quantiles=QUANTILES if QUANTILES else None,
        target_name=SUBSIDENCE_COL,
        scaler_target_name=SUBS_SCALER_KEY,
        output_target_name="subsidence",
        target_key_pred="subs_pred",
        component_index=0,
        scaler_info=scaler_info_dict,
        coord_scaler=coord_scaler,
        coord_columns=("coord_t", "coord_x", "coord_y"),
        train_end_time=cfg.get("TRAIN_END_YEAR"),
        forecast_start_time=FORECAST_START_YEAR,
        forecast_horizon=FORECAST_H,
        future_time_grid=future_grid,
        eval_forecast_step=None,
        sample_index_offset=0,
        city_name=CITY_NAME,
        model_name=MODEL_NAME,
        dataset_name=dataset_name_for_forecast,
        csv_eval_path=csv_eval,
        csv_future_path=csv_future,
        time_as_datetime=False,
        time_format=None,
        verbose=1,
        eval_metrics=True,
        metrics_quantile_interval=(0.1, 0.9),
        metrics_per_horizon=True,
        metrics_extra=["pss"],
        metrics_savefile=metrics_json,
        metrics_save_format=".json",
        metrics_time_as_str=True,
        value_mode="cumulative",
        input_value_mode="cumulative",
        output_unit="mm",
        output_unit_from="m",
        output_unit_mode="overwrite",
        output_unit_col="subsidence_unit",
    )

    cal_stats = {
        "skipped": True,
        "reason": "fast_sensitivity",
    }
    if not FAST_SENS:
        df_eval_cal2, df_future_cal2, cal_stats = (
            calibrate_quantile_forecasts(
                df_eval=df_eval,
                df_future=df_future,
                target_name="subsidence",
                interval=(0.1, 0.9),
                target_coverage=0.8,
                use="auto",
                tol=0.02,
                f_max=5.0,
                enforce_monotonic="cummax",
                save_eval=csv_eval_cal,
                save_future=csv_future_cal,
                save_stats=cal_stats_path,
                verbose=2,
            )
        )
        if df_eval_cal2 is not None:
            df_eval = df_eval_cal2
        if df_future_cal2 is not None:
            df_future = df_future_cal2

    # Optional diagnostics JSON from DataFrame
    diag_suffix = "calibrated" if not FAST_SENS else "uncalibrated"
    if df_eval is not None and not df_eval.empty:
        diag_path = os.path.join(
            run_dir,
            (
                f"{CITY_NAME}_{MODEL_NAME}_eval_diagnostics_"
                f"{dataset_name_for_forecast}_H{FORECAST_H}_"
                f"{diag_suffix}.json"
            ),
        )
        _ = evaluate_forecast(
            df_eval,
            target_name="subsidence",
            quantile_interval=(0.1, 0.9),
            per_horizon=True,
            extra_metrics=["pss"],
            savefile=diag_path,
            verbose=1,
        )

    # ---------------------------
    # Evaluate() + physics payload
    # ---------------------------
    # ds_eval_full = test_dataset or val_dataset
    # ds_eval = ds_eval_full
    # if eval_max_batches is not None:
    #     ds_eval = ds_eval_full.take(int(eval_max_batches))

    ds_eval = make_tf_dataset(
        X_fore,
        y_fore,
        batch_size=BATCH_SIZE,
        shuffle=False,
        mode=MODE,
        forecast_horizon=FORECAST_H,
        check_npz_finite=True,
        check_finite=True,
        scan_finite_batches=None,
        dynamic_feature_names=DYN_NAMES,
        future_feature_names=FUT_NAMES,
    )

    if DEBUG and (not FAST_SENS):
        _ = debug_val_interval(
            model_inf,
            ds_eval,
            n_q=len(QUANTILES),
            max_batches=2,
            verbose=1,
        )

    if not USE_IN_MEMORY_MODEL:
        model_inf.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.0),
            loss=loss_arg,
            loss_weights=lossw_arg,
            metrics=metrics_compile,
            **physics_loss_weights,
        )

    eval_results = {}
    phys = {}

    try:
        eval_raw = model_inf.evaluate(
            ds_eval,
            return_dict=True,
            verbose=1,
        )
        eval_results = _logs_to_py(eval_raw)
        for k in ("epsilon_prior", "epsilon_cons", "epsilon_gw"):
            if k in eval_results:
                phys[k] = float(_to_py(eval_raw[k]))
    except Exception as e:
        print("[Warn] evaluate() failed:", e)

    phys_npz_path = os.path.join(
        run_dir,
        f"{CITY_NAME}_phys_payload_run_val.npz",
    )
    try:
        _ = model_inf.export_physics_payload(
            ds_eval,
            # max_batches=eval_max_batches,
            save_path=phys_npz_path,
            format="npz",
            overwrite=True,
            metadata={
                "city": CITY_NAME,
                "split": dataset_name_for_forecast,
                "time_units": TIME_UNITS,
            },
        )
    except Exception as e:
        print("[Warn] export_physics_payload failed:", e)

    # ---------------------------
    # Interval metrics (scaled + phys)
    # ---------------------------
    cov80_uncal = None
    cov80_cal = None
    sharp80_uncal = None
    sharp80_cal = None

    cov80_uncal_phys = None
    cov80_cal_phys = None
    sharp80_uncal_phys = None
    sharp80_cal_phys = None

    censor_metrics = None

    y_true = tf.convert_to_tensor(
        y_fore_fmt["subs_pred"],
        dtype=tf.float32,
    )

    if QUANTILES:
        s_q = tf.convert_to_tensor(s_pred, dtype=tf.float32)
        s_q_cal = tf.convert_to_tensor(
            predictions_for_formatter["subs_pred"],
            dtype=tf.float32,
        )

        cov80_uncal = float(
            coverage80_fn(y_true, s_q).numpy()
        )
        sharp80_uncal = float(
            sharpness80_fn(y_true, s_q).numpy()
        )
        cov80_cal = float(
            coverage80_fn(y_true, s_q_cal).numpy()
        )
        sharp80_cal = float(
            sharpness80_fn(y_true, s_q_cal).numpy()
        )

        # Physical space
        y_true_np = y_true.numpy()
        s_q_np = s_q.numpy()
        s_q_cal_np = s_q_cal.numpy()

        y_true_phys_np = inverse_scale_target(
            y_true_np,
            scaler_info=scaler_info_dict,
            target_name=SUBS_SCALER_KEY,
        )
        s_q_phys_np = inverse_scale_target(
            s_q_np,
            scaler_info=scaler_info_dict,
            target_name=SUBS_SCALER_KEY,
        )
        s_q_cal_phys_np = inverse_scale_target(
            s_q_cal_np,
            scaler_info=scaler_info_dict,
            target_name=SUBS_SCALER_KEY,
        )

        y_true_phys_tf = tf.convert_to_tensor(
            y_true_phys_np,
            dtype=tf.float32,
        )
        s_q_phys_tf = tf.convert_to_tensor(
            s_q_phys_np,
            dtype=tf.float32,
        )
        s_q_cal_phys_tf = tf.convert_to_tensor(
            s_q_cal_phys_np,
            dtype=tf.float32,
        )

        cov80_uncal_phys = float(
            coverage80_fn(y_true_phys_tf, s_q_phys_tf).numpy()
        )
        sharp80_uncal_phys = float(
            sharpness80_fn(y_true_phys_tf, s_q_phys_tf).numpy()
        )
        cov80_cal_phys = float(
            coverage80_fn(
                y_true_phys_tf,
                s_q_cal_phys_tf,
            ).numpy()
        )
        sharp80_cal_phys = float(
            sharpness80_fn(
                y_true_phys_tf,
                s_q_cal_phys_tf,
            ).numpy()
        )

    # Censor mask (no model calls)
    mask = None
    if CENSOR_FLAG_IDX is not None:
        mask_list = []
        for xb, yb in with_progress(
            ds_eval,
            desc="Censor mask (no preds)",
        ):
            H = tf.shape(yb["subs_pred"])[1]
            mask_b = build_censor_mask(
                xb,
                H,
                CENSOR_FLAG_IDX,
                CENSOR_THRESH,
                source=(CENSOR_MASK_SOURCE or "dynamic"),
                reduce_time="any",
                align="broadcast",
            )
            mask_list.append(mask_b)
        mask = tf.concat(mask_list, axis=0) if mask_list else None
        
    if (mask is not None) and QUANTILES:
        med_idx = int(
            np.argmin(
                np.abs(np.asarray(QUANTILES, float) - 0.5)
            )
        )
    
        # Median prediction in model (scaled) space
        s_med = tf.convert_to_tensor(
            predictions_for_formatter["subs_pred"],
            dtype=tf.float32,
        )[..., med_idx, :]
    
        # --- IMPORTANT: align to ds_eval subset length -----------
        n_mask = tf.shape(mask)[0]
        y_true_sub = y_true[:n_mask]
        s_med_sub = s_med[:n_mask]
        mask_sub = mask[:n_mask]  # (N,H,1) bool
    
        # Inverse-scale only the subset (faster + consistent)
        y_true_phys_np = inverse_scale_target(
            y_true_sub.numpy(),
            scaler_info=scaler_info_dict,
            target_name=SUBS_SCALER_KEY,
        )
        s_med_phys_np = inverse_scale_target(
            s_med_sub.numpy(),
            scaler_info=scaler_info_dict,
            target_name=SUBS_SCALER_KEY,
        )
    
        y_true_phys = tf.convert_to_tensor(
            y_true_phys_np,
            dtype=tf.float32,
        )
        s_med_phys = tf.convert_to_tensor(
            s_med_phys_np,
            dtype=tf.float32,
        )
    
        mask_f = tf.cast(mask_sub, tf.float32)
        num_cens = tf.reduce_sum(mask_f) + 1e-8
        num_unc = tf.reduce_sum(1.0 - mask_f) + 1e-8
    
        abs_err = tf.abs(y_true_phys - s_med_phys)
        mae_cens = tf.reduce_sum(abs_err * mask_f) / num_cens
        mae_unc = tf.reduce_sum(abs_err * (1.0 - mask_f)) / num_unc
    
        censor_metrics = {
            "flag_name": CENSOR_FLAG_NAME,
            "threshold": float(CENSOR_THRESH),
            "mae_censored": float(mae_cens.numpy()),
            "mae_uncensored": float(mae_unc.numpy()),
            "subset_batches": int(eval_max_batches)
            if eval_max_batches is not None
            else None,
        }
    # if (mask is not None) and QUANTILES:
    #     med_idx = int(
    #         np.argmin(
    #             np.abs(np.asarray(QUANTILES, float) - 0.5)
    #         )
    #     )
    #     s_med = tf.convert_to_tensor(
    #         predictions_for_formatter["subs_pred"],
    #         dtype=tf.float32,
    #     )[..., med_idx, :]

    #     y_true_phys_np = inverse_scale_target(
    #         y_true.numpy(),
    #         scaler_info=scaler_info_dict,
    #         target_name=SUBS_SCALER_KEY,
    #     )
    #     s_med_phys_np = inverse_scale_target(
    #         s_med.numpy(),
    #         scaler_info=scaler_info_dict,
    #         target_name=SUBS_SCALER_KEY,
    #     )

    #     y_true_phys = tf.convert_to_tensor(
    #         y_true_phys_np,
    #         dtype=tf.float32,
    #     )
    #     s_med_phys = tf.convert_to_tensor(
    #         s_med_phys_np,
    #         dtype=tf.float32,
    #     )

    #     mask_f = tf.cast(mask, tf.float32)
    #     num_cens = tf.reduce_sum(mask_f) + 1e-8
    #     num_unc = tf.reduce_sum(1.0 - mask_f) + 1e-8

    #     abs_err = tf.abs(y_true_phys - s_med_phys)
    #     mae_cens = tf.reduce_sum(abs_err * mask_f) / num_cens
        mae_unc = tf.reduce_sum(abs_err * (1.0 - mask_f)) / num_unc

        censor_metrics = {
            "flag_name": CENSOR_FLAG_NAME,
            "threshold": float(CENSOR_THRESH),
            "mae_censored": float(mae_cens.numpy()),
            "mae_uncensored": float(mae_unc.numpy()),
        }

    # ---------------------------
    # Build + save eval payload
    # ---------------------------
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    payload = {
        "timestamp": stamp,
        "tf_version": tf.__version__,
        "numpy_version": np.__version__,
        "quantiles": QUANTILES,
        "horizon": int(FORECAST_H),
        "batch_size": int(BATCH_SIZE),
        "metrics_evaluate": {
            k: _to_py(v) for k, v in (eval_results or {}).items()
        },
        "physics_diagnostics": phys,
    }

    if QUANTILES:
        payload["interval_calibration"] = {
            "target": 0.80,
            "factors_per_horizon": (
                getattr(cal80, "factors_", None).tolist()
                if (cal80 is not None and hasattr(cal80, "factors_"))
                else None
            ),
            "factors_per_horizon_from_cal_stats": cal_stats,
            "coverage80_uncalibrated": cov80_uncal,
            "coverage80_calibrated": cov80_cal,
            "sharpness80_uncalibrated": sharp80_uncal,
            "sharpness80_calibrated": sharp80_cal,
            "coverage80_uncalibrated_phys": cov80_uncal_phys,
            "coverage80_calibrated_phys": cov80_cal_phys,
            "sharpness80_uncalibrated_phys": sharp80_uncal_phys,
            "sharpness80_calibrated_phys": sharp80_cal_phys,
        }

    if censor_metrics is not None:
        payload["censor_stratified"] = censor_metrics

    if metrics_point:
        payload["point_metrics"] = {
            "mae": metrics_point.get("mae"),
            "mse": metrics_point.get("mse"),
            "r2": metrics_point.get("r2"),
        }

    if per_h_mae_dict:
        payload.setdefault("per_horizon", {})
        payload["per_horizon"]["mae"] = per_h_mae_dict

    if per_h_r2_dict:
        payload.setdefault("per_horizon", {})
        payload["per_horizon"]["r2"] = per_h_r2_dict

    try:
        payload = convert_eval_payload_units(
            payload,
            cfg,
            mode=units_mode,
            scope=units_scope,
        )
    except Exception as e:
        print("[Warn] unit conversion skipped:", e)

    json_out = os.path.join(
        run_dir,
        f"geoprior_eval_phys_{stamp}.json",
    )
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    json_out_interp = os.path.join(
        run_dir,
        f"geoprior_eval_phys_{stamp}_interpretable.json",
    )
    try:
        _ = postprocess_eval_json(
            json_out,
            scope="all",
            out_path=json_out_interp,
            overwrite=True,
            add_rmse=True,
        )
    except Exception:
        pass

    # ---------------------------
    # Ablation record
    # ---------------------------
    ABLCFG = {
        "PDE_MODE_CONFIG": PDE_MODE,
        "GEOPRIOR_USE_EFFECTIVE_H": GEOPRIOR_USE_EFF_H,
        "GEOPRIOR_KAPPA_MODE": GEOPRIOR_KAPPA_MODE,
        "GEOPRIOR_HD_FACTOR": GEOPRIOR_HD_FACTOR,
        "LAMBDA_CONS": LAMBDA_CONS,
        "LAMBDA_GW": LAMBDA_GW,
        "LAMBDA_PRIOR": LAMBDA_PRIOR,
        "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
        "LAMBDA_BOUNDS": LAMBDA_BOUNDS,
        "LAMBDA_MV": LAMBDA_MV,
        "LAMBDA_Q": LAMBDA_Q,
    }

    ival = payload.get("interval_calibration", {}) or {}
    cal_stats2 = ival.get("factors_per_horizon_from_cal_stats", {}) or {}
    ev_after = cal_stats2.get("eval_after", {}) or {}

    def _pick_first(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    abl_cov = _pick_first(
        ev_after.get("coverage"),
        ival.get("coverage80_calibrated_phys"),
        ival.get("coverage80_calibrated"),
        ival.get("coverage80_uncalibrated_phys"),
        ival.get("coverage80_uncalibrated"),
    )
    abl_sharp = _pick_first(
        ev_after.get("sharpness"),
        ival.get("sharpness80_calibrated_phys"),
        ival.get("sharpness80_calibrated"),
        ival.get("sharpness80_uncalibrated_phys"),
        ival.get("sharpness80_uncalibrated"),
    )

    eval_mae = (payload.get("point_metrics", {}) or {}).get("mae")
    eval_mse = (payload.get("point_metrics", {}) or {}).get("mse")
    eval_r2 = (payload.get("point_metrics", {}) or {}).get("r2")

    eval_rmse = None
    if eval_mse is not None:
        try:
            eval_rmse = float(np.sqrt(float(eval_mse)))
        except Exception:
            eval_rmse = None

    save_ablation_record(
        outdir=run_dir,
        city=CITY_NAME,
        model_name=MODEL_NAME,
        cfg=ABLCFG,
        eval_dict={
            "r2": (float(eval_r2) if eval_r2 is not None else None),
            "mse": (float(eval_mse) if eval_mse is not None else None),
            "mae": (float(eval_mae) if eval_mae is not None else None),
            "rmse": (float(eval_rmse) if eval_rmse is not None else None),
            "coverage80": (float(abl_cov) if abl_cov is not None else None),
            "sharpness80": (float(abl_sharp) if abl_sharp is not None else None),
            "units": payload.get("units", {}) if isinstance(payload, dict) else {},
        },
        phys_diag=(phys or {}),
        per_h_mae=per_h_mae_dict,
        per_h_r2=per_h_r2_dict,
    )

    # DONE marker
    def _write_done_marker(
        run_path: Path,
        *,
        city: str,
        pde_mode: str,
        lambda_cons: float,
        lambda_prior: float,
        tag: Optional[str] = None,
    ) -> None:
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "DONE").write_text("", encoding="utf-8")
        d = {
            "city": city,
            "pde_mode": pde_mode,
            "lambda_cons": float(lambda_cons),
            "lambda_prior": float(lambda_prior),
            "run_tag": tag,
        }
        tmp = run_path / "DONE.json.tmp"
        outp = run_path / "DONE.json"
        tmp.write_text(json.dumps(d), encoding="utf-8")
        tmp.replace(outp)

    _write_done_marker(
        Path(run_dir),
        city=CITY_NAME,
        pde_mode=PDE_MODE,
        lambda_cons=LAMBDA_CONS,
        lambda_prior=LAMBDA_PRIOR,
        tag=run_tag2,
    )

    # Physics plots (optional)
    if not FAST_SENS:
        try:
            phys_payload, _ = load_physics_payload(phys_npz_path)
            plot_physics_values_in(
                phys_payload,
                dataset=ds_eval,
                keys=[
                    "cons_res_vals",
                    "log10_tau",
                    "log10_tau_prior",
                    "K",
                    "Ss",
                    "Hd",
                ],
                mode="map",
                transform=None,
                savefig=os.path.join(run_dir, "phys_maps.png"),
            )
        except Exception:
            pass

        try:
            plot_eval_future(
                df_eval=df_eval,
                df_future=df_future,
                target_name=SUBSIDENCE_COL,
                quantiles=QUANTILES,
                spatial_cols=("coord_x", "coord_y"),
                time_col="coord_t",
                eval_years=[FORECAST_START_YEAR - 1],
                future_years=future_grid,
                eval_view_quantiles=[0.5],
                future_view_quantiles=QUANTILES,
                spatial_mode="hexbin",
                hexbin_gridsize=40,
                savefig_prefix=os.path.join(
                    run_dir,
                    f"{CITY_NAME}_subsidence_view",
                ),
                save_fmts=[".png", ".pdf"],
                show=False,
                verbose=1,
            )
        except Exception:
            pass

        try:
            save_all_figures(
                output_dir=run_dir,
                prefix=f"{CITY_NAME}_{MODEL_NAME}_plot_",
                fmts=[".png", ".pdf"],
            )
        except Exception:
            pass

    print(
        f"\n---- {CITY_NAME.upper()} {MODEL_NAME} RUN DONE ----\n"
        f"Artifacts -> {run_dir}\n"
    )

    return run_dir
