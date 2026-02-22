# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.runs.xfer.xfer_core

v3.2 core engine for transfer evaluation.

This is the GUI/backend engine extracted from the old
stage5 script. No argparse; no CLI assumptions.

Public API
----------
- build_plan(...)
- iter_cases(plan)
- run_case(case, plan, caches, ...)
- run_plan(plan, ...)

The GUI calls run_plan via run_xfer_matrix (Job).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional
from typing import Tuple

import numpy as np
import tensorflow as tf

from fusionlab.compat.keras import (
    load_inference_model,
    load_model_from_tfv2,
)
from fusionlab.nn.calibration import (
    apply_calibrator_to_subs,
    fit_interval_calibrator_on_val,
)
from fusionlab.nn.keras_metrics import (
    coverage80_fn,
    sharpness80_fn,
)
from fusionlab.nn.pinn.geoprior.models import GeoPriorSubsNet
from fusionlab.utils.forecast_utils import format_and_forecast
from fusionlab.utils.scale_metrics import (
    inverse_scale_target,
    per_horizon_metrics,
    point_metrics,
)
from fusionlab.utils.nat_utils import (
    ensure_input_shapes,
    extract_preds,
    load_best_hps_near_model,
    map_targets_for_training,
    sanitize_inputs_np,
)

from .xfer_utils import (
    LogFn,
    align_static_to_source,
    cfg_horizon,
    cfg_mode,
    cfg_quantiles,
    ensure_dir,
    find_stage1_manifest,
    load_calibrator_near,
    load_stage1_manifest,
    now_tag,
    pick_npz,
    reproject_dynamic_to_source,
    resolve_bundle_paths,
    check_transfer_schema
)
from .xfer_utils import _load_scalers as _ls
from .xfer_utils import _get_scaler as _gs
from .xfer_utils import _transform_with_scaler as _ts
from .xfer_utils import best_model_artifact
    
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


# ---------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------
@dataclass
class WarmStartConfig:
    split: str = "train"
    samples: int = 20000
    frac: Optional[float] = None
    epochs: int = 3
    lr: float = 1e-4
    seed: int = 123


@dataclass
class XferPlan:
    city_a: str
    city_b: str
    results_dir: str = "results"

    splits: Tuple[str, ...] = ("val", "test")
    strategies: Tuple[str, ...] = ("baseline", "xfer")
    calib_modes: Tuple[str, ...] = ("none", "source", "target")
    rescale_modes: Tuple[str, ...] = ("as_is",)

    batch_size: int = 32
    quantiles: Optional[List[float]] = None

    allow_reorder_dynamic: bool = False
    allow_reorder_future: bool = False

    prefer_tuned: bool = True
    out_dir: Optional[str] = None

    warm: WarmStartConfig = field(
        default_factory=WarmStartConfig
    )


@dataclass
class XferCase:
    direction: str
    src_city: str
    tgt_city: str

    strategy: str
    split: str
    calib_mode: str
    rescale_mode: str


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------
def _log(log_fn: Optional[LogFn]) -> LogFn:
    return log_fn if callable(log_fn) else print


def _strict_from_mode(rescale_mode: str) -> bool:
    return str(rescale_mode).lower().strip() == "strict"


def _ensure_np_inputs(
    x: Dict[str, Any],
    mode: str,
    horizon: int,
) -> Dict[str, Any]:
    x = sanitize_inputs_np(x)
    x = ensure_input_shapes(x, mode, horizon)
    return x


# ---------------------------------------------------------------------
# model builder + loader
# ---------------------------------------------------------------------
def _build_geoprior_builder(
    *,
    M_src: Dict[str, Any],
    X_sample: Dict[str, Any],
    out_s_dim: int,
    out_g_dim: int,
    horizon: int,
    quantiles: Optional[List[float]],
    best_hps: Dict[str, Any],
) -> Callable[[], GeoPriorSubsNet]:
    cfg = dict(M_src.get("config") or {})

    s_dim = int(X_sample["static_features"].shape[-1])
    d_dim = int(X_sample["dynamic_features"].shape[-1])
    f_dim = int(X_sample["future_features"].shape[-1])

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
        use_effective_h=cfg.get(
            "USE_EFFECTIVE_H",
            False,
        ),
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
    *,
    M_src: Dict[str, Any],
    X_sample: Dict[str, Any],
    quantiles: Optional[List[float]],
    prefer_tuned: bool,
) -> Tuple[Any, Any, Dict[str, Any]]:
    run_dir = str((M_src.get("paths") or {}).get("run_dir") or "")
    if not run_dir:
        raise SystemExit("Missing M_src.paths.run_dir")

    

    best = best_model_artifact(run_dir, prefer_tuned=prefer_tuned)
    if not best:
        raise SystemExit(f"No model under: {run_dir}")

    bundle = resolve_bundle_paths(best)
    init_p = bundle["init_manifest_path"]

    init_m: Dict[str, Any] = {}
    if isinstance(init_p, str) and os.path.isfile(init_p):
        with open(init_p, "r", encoding="utf-8") as f:
            init_m = json.load(f)

    best_hps = load_best_hps_near_model(
        bundle["keras_path"] or bundle["run_dir"],
    ) or {}

    dims = (M_src.get("artifacts") or {}).get("sequences") or {}
    dims = dims.get("dims") or {}
    out_s = int(dims.get("output_subsidence_dim", 1))
    out_g = int(dims.get("output_gwl_dim", 1))

    horizon = cfg_horizon(dict(M_src.get("config") or {}), 1)

    builder = _build_geoprior_builder(
        M_src=M_src,
        X_sample=X_sample,
        out_s_dim=out_s,
        out_g_dim=out_g,
        horizon=horizon,
        quantiles=quantiles,
        best_hps=best_hps,
    )

    model = load_inference_model(
        keras_path=bundle["keras_path"],
        weights_path=bundle["weights_path"],
        manifest_path=(init_p if os.path.isfile(init_p) else None),
        manifest=init_m if init_m else None,
        builder=builder,
        build_inputs=X_sample,
        out_s_dim=out_s,
        out_g_dim=out_g,
        mode=cfg_mode(dict(M_src.get("config") or {})),
        horizon=horizon,
        prefer_full_model=True,
    )

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


# ---------------------------------------------------------------------
# warm-start helpers
# ---------------------------------------------------------------------
def _pinball_loss(
    quantiles: Optional[List[float]],
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
    quantiles: Optional[List[float]],
    lr: float,
) -> List[str]:
    opt = tf.keras.optimizers.Adam(
        learning_rate=float(lr),
    )
    subs_loss = _pinball_loss(quantiles)

    losses = {"subs_pred": subs_loss}
    try:
        model.compile(optimizer=opt, loss=losses)
        return ["subs_pred"]
    except Exception:
        pass

    losses = {
        "subs_pred": subs_loss,
        "gwl_pred": tf.keras.losses.MSE,
    }
    model.compile(optimizer=opt, loss=losses)
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


# ---------------------------------------------------------------------
# plan / cases
# ---------------------------------------------------------------------
def build_plan(
    *,
    city_a: str,
    city_b: str,
    results_dir: str,
    splits: Optional[Iterable[str]] = None,
    strategies: Optional[Iterable[str]] = None,
    calib_modes: Optional[Iterable[str]] = None,
    rescale_modes: Optional[Iterable[str]] = None,
    batch_size: int = 32,
    quantiles: Optional[List[float]] = None,
    allow_reorder_dynamic: bool = False,
    allow_reorder_future: bool = False,
    prefer_tuned: bool = True,
    out_dir: Optional[str] = None,
    warm: Optional[WarmStartConfig] = None,
) -> XferPlan:
    p = XferPlan(
        city_a=str(city_a),
        city_b=str(city_b),
        results_dir=str(results_dir),
        batch_size=int(batch_size),
        quantiles=quantiles,
        allow_reorder_dynamic=bool(allow_reorder_dynamic),
        allow_reorder_future=bool(allow_reorder_future),
        prefer_tuned=bool(prefer_tuned),
        out_dir=out_dir,
        warm=warm or WarmStartConfig(),
    )
    if splits is not None:
        p.splits = tuple(splits)
    if strategies is not None:
        p.strategies = tuple(strategies)
    if calib_modes is not None:
        p.calib_modes = tuple(calib_modes)
    if rescale_modes is not None:
        p.rescale_modes = tuple(rescale_modes)
    return p


def iter_cases(plan: XferPlan) -> List[XferCase]:
    cases: List[XferCase] = []

    want_base = "baseline" in plan.strategies
    want_xfer = ("xfer" in plan.strategies) or (
        "warm" in plan.strategies
    )

    dirs: List[Tuple[str, str, str]] = []
    if want_base:
        dirs += [
            ("A_to_A", plan.city_a, plan.city_a),
            ("B_to_B", plan.city_b, plan.city_b),
        ]
    if want_xfer:
        dirs += [
            ("A_to_B", plan.city_a, plan.city_b),
            ("B_to_A", plan.city_b, plan.city_a),
        ]

    for dtag, src, tgt in dirs:
        is_base = dtag in ("A_to_A", "B_to_B")
        for rm in plan.rescale_modes:
            for sp in plan.splits:
                for cm in plan.calib_modes:
                    if is_base and want_base:
                        cases.append(
                            XferCase(
                                direction=dtag,
                                src_city=src,
                                tgt_city=tgt,
                                strategy="baseline",
                                split=sp,
                                calib_mode=cm,
                                rescale_mode=rm,
                            )
                        )
                    if (not is_base) and ("xfer" in plan.strategies):
                        cases.append(
                            XferCase(
                                direction=dtag,
                                src_city=src,
                                tgt_city=tgt,
                                strategy="xfer",
                                split=sp,
                                calib_mode=cm,
                                rescale_mode=rm,
                            )
                        )
                    if (not is_base) and ("warm" in plan.strategies):
                        cases.append(
                            XferCase(
                                direction=dtag,
                                src_city=src,
                                tgt_city=tgt,
                                strategy="warm",
                                split=sp,
                                calib_mode=cm,
                                rescale_mode=rm,
                            )
                        )

    return cases


# ---------------------------------------------------------------------
# main execution
# ---------------------------------------------------------------------
def run_case(
    *,
    case: XferCase,
    plan: XferPlan,
    caches: Dict[str, Any],
    save_dir: str,
    log_fn: Optional[LogFn] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute a single case. Returns a result dict or None.
    """
    # ---- load manifests (cache) ----
    M_src = caches.setdefault("M", {}).get(case.src_city)
    if M_src is None:
        M_src = load_stage1_manifest(
            results_dir=plan.results_dir,
            city=case.src_city,
        )
        M_src["city"] = case.src_city
        M_src["manifest_path"] = find_stage1_manifest(
            results_dir=plan.results_dir,
            city=case.src_city,
        )
        caches.setdefault("M", {})[case.src_city] = M_src

    M_tgt = caches.setdefault("M", {}).get(case.tgt_city)
    if M_tgt is None:
        M_tgt = load_stage1_manifest(
            results_dir=plan.results_dir,
            city=case.tgt_city,
        )
        M_tgt["city"] = case.tgt_city
        M_tgt["manifest_path"] = find_stage1_manifest(
            results_dir=plan.results_dir,
            city=case.tgt_city,
        )
        caches.setdefault("M", {})[case.tgt_city] = M_tgt

    # ---- load data ----
    X_tgt, y_tgt = pick_npz(M_tgt, case.split)
    if X_tgt is None:
        return None

    cfg_t = dict(M_tgt.get("config") or {})
    mode = cfg_mode(cfg_t)
    H = cfg_horizon(cfg_t, 1)
    Q = cfg_quantiles(cfg_t, plan.quantiles)

    X_tgt = _ensure_np_inputs(X_tgt, mode, H)
    y_map = map_targets_for_training(y_tgt or {})

    # ---- align + schema ----
    X_tgt = align_static_to_source(
        X_tgt,
        M_src,
        M_tgt,
        log_fn=log_fn,
    )

    strict = _strict_from_mode(case.rescale_mode)
    if strict:
        X_tgt = reproject_dynamic_to_source(
            X_tgt,
            M_src,
            M_tgt,
        )

    X_tgt, schema_audit = check_transfer_schema(
        M_src=M_src,
        M_tgt=M_tgt,
        X_tgt=X_tgt,
        allow_reorder_dynamic=plan.allow_reorder_dynamic,
        allow_reorder_future=plan.allow_reorder_future,
        log_fn=log_fn,
    )

    # ---- load / reuse source model ----
    key = (case.src_city, tuple(Q), plan.prefer_tuned)
    mp = caches.setdefault("models", {}).get(key)
    if mp is None:
        model, model_pred, bundle = _load_source_model(
            M_src=M_src,
            X_sample=X_tgt,
            quantiles=Q,
            prefer_tuned=plan.prefer_tuned,
        )
        mp = (model, model_pred, bundle)
        caches.setdefault("models", {})[key] = mp
    model, model_pred, bundle = mp

    warm_meta: Dict[str, Any] = {}
    if case.strategy == "warm":
        # warm-start uses target warm split
        X_w, y_w = pick_npz(M_tgt, plan.warm.split)
        if X_w is None or y_w is None:
            return None

        X_w = _ensure_np_inputs(X_w, mode, H)
        y_wm = map_targets_for_training(y_w)

        X_w = align_static_to_source(
            X_w,
            M_src,
            M_tgt,
            log_fn=log_fn,
        )
        X_w, _ = check_transfer_schema(
            M_src=M_src,
            M_tgt=M_tgt,
            X_tgt=X_w,
            allow_reorder_dynamic=plan.allow_reorder_dynamic,
            allow_reorder_future=plan.allow_reorder_future,
            log_fn=log_fn,
        )
        if strict:
            X_w = reproject_dynamic_to_source(
                X_w,
                M_src,
                M_tgt,
            )

        n_total = int(X_w["dynamic_features"].shape[0])
        from .xfer_utils import choose_warm_idx, slice_npz_dict

        idx = choose_warm_idx(
            n_total=n_total,
            n_samples=plan.warm.samples,
            frac=plan.warm.frac,
            seed=plan.warm.seed,
        )
        X_ws = slice_npz_dict(X_w, idx)
        y_ws = slice_npz_dict(y_wm, idx)

        warm_keys = _compile_warm_model(
            model,
            quantiles=Q,
            lr=plan.warm.lr,
        )
        y_ws = {k: y_ws[k] for k in warm_keys if k in y_ws}

        ds = _make_ds(
            X_ws,
            y_ws,
            batch_size=plan.batch_size,
            seed=plan.warm.seed,
        )
        model.fit(
            ds,
            epochs=int(plan.warm.epochs),
            verbose=0,
        )

        model_pred = model
        warm_meta = {
            "warm_split": plan.warm.split,
            "warm_samples": int(len(idx)),
            "warm_frac": plan.warm.frac,
            "warm_epochs": int(plan.warm.epochs),
            "warm_lr": float(plan.warm.lr),
            "warm_seed": int(plan.warm.seed),
        }

    # ---- calibration ----
    cal = None
    if case.calib_mode == "source":
        cal = load_calibrator_near(
            str(bundle.get("run_dir") or ""),
            target=0.80,
        )
    elif case.calib_mode == "target":
        try:
            vx, vy = pick_npz(M_tgt, "val")
            if vx is not None:
                vx = _ensure_np_inputs(vx, mode, H)
                vx = align_static_to_source(
                    vx,
                    M_src,
                    M_tgt,
                    log_fn=log_fn,
                )
                if strict:
                    vx = reproject_dynamic_to_source(
                        vx,
                        M_src,
                        M_tgt,
                    )
                vy_m = map_targets_for_training(vy or {})
                ds_v = tf.data.Dataset.from_tensor_slices(
                    (vx, vy_m)
                ).batch(plan.batch_size)
                cal = fit_interval_calibrator_on_val(
                    model,
                    ds_v,
                    target=0.80,
                )
        except Exception:
            cal = None

    # ---- predict ----
    pred_dict = model_pred.predict(X_tgt, verbose=0)
    subs_pred, gwl_pred = extract_preds(model, pred_dict)

    if cal is not None and subs_pred.ndim == 4:
        subs_pred = apply_calibrator_to_subs(
            cal,
            subs_pred,
        )

    preds = {"subs_pred": subs_pred, "gwl_pred": gwl_pred}

    # ---- metrics + exports ----
    enc_t = (M_tgt.get("artifacts") or {}).get("encoders") or {}
    enc_s = (M_src.get("artifacts") or {}).get("encoders") or {}



    coord_scaler, sc_t = _ls(enc_t)
    _, sc_s = _ls(enc_s)

    cols_s = dict((M_src.get("config") or {}).get("cols") or {})
    cols_t = dict((M_tgt.get("config") or {}).get("cols") or {})

    subs_col_s = cols_s.get("subsidence", "subsidence")
    subs_col_t = cols_t.get("subsidence", "subsidence")

    y_true_phys = None
    if "subs_pred" in y_map and sc_t is not None:
        y_true_phys = inverse_scale_target(
            y_map["subs_pred"][..., :1],
            scaler_info=sc_t,
            target_name=subs_col_t,
        )

    y_pred_phys = None
    if sc_s is not None:
        if subs_pred.ndim == 4:
            q_arr = np.asarray(Q, dtype=np.float32)
            mid = int(np.argmin(np.abs(q_arr - 0.5)))
            y_point = subs_pred[:, :, mid, :1]
        else:
            y_point = subs_pred[:, :, :1]
        y_pred_phys = inverse_scale_target(
            y_point,
            scaler_info=sc_s,
            target_name=subs_col_s,
        )

    metrics_overall: Dict[str, Any] = {}
    metrics_h: Dict[str, Any] = {}
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
            "mae": ph.get("mae") or {},
            "r2": ph.get("r2") or {},
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

    subs_t_scaler = _gs(sc_t, subs_col_t)
    subs_s_scaler = _gs(sc_s, subs_col_s)

    preds_fmt = dict(preds)
    if subs_s_scaler is not None and subs_t_scaler is not None:
        s_phys = inverse_scale_target(
            subs_pred,
            scaler_info=sc_s,
            target_name=subs_col_s,
        )
        preds_fmt["subs_pred"] = _ts(subs_t_scaler, s_phys)

    y_true_for_format = None
    if y_map:
        y_true_for_format = {}
        if "subs_pred" in y_map:
            y_true_for_format["subsidence"] = y_map["subs_pred"]
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
        f"{case.src_city}_to_{case.tgt_city}_"
        f"{case.strategy}_{case.split}_"
        f"{case.calib_mode}_{case.rescale_mode}"
    )
    csv_eval = os.path.join(save_dir, base + "_eval.csv")
    csv_fut = os.path.join(save_dir, base + "_future.csv")

    subs_kind = str(cfg_t.get("SUBSIDENCE_KIND", "rate"))

    format_and_forecast(
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
            f"{case.strategy}_{case.split}_{case.calib_mode}"
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

    return {
        "strategy": case.strategy,
        "rescale_mode": case.rescale_mode,
        "direction": case.direction,
        "source_city": case.src_city,
        "target_city": case.tgt_city,
        "split": case.split,
        "calibration": case.calib_mode,
        "quantiles": Q if subs_pred.ndim == 4 else None,
        "coverage80": coverage80,
        "sharpness80": sharpness80,
        "overall_mae": metrics_overall.get("mae"),
        "overall_mse": metrics_overall.get("mse"),
        "overall_r2": metrics_overall.get("r2"),
        "per_horizon_mae": metrics_h.get("mae") or {},
        "per_horizon_r2": metrics_h.get("r2") or {},
        "warm": warm_meta,
        "csv_eval": csv_eval,
        "csv_future": csv_fut,
        "schema": schema_audit,
        "model_path": bundle.get("keras_path"),
        "model_dir": os.path.dirname(
            str(bundle.get("keras_path") or "")
        ),
    }


def run_plan(
    *,
    plan: XferPlan,
    log_fn: Optional[LogFn] = None,
    progress_fn: Optional[Callable[[int, int], Any]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Run all cases and write JSON+CSV summary.

    Returns
    -------
    dict:
      out_dir, json_path, csv_path, results
    """
    log = _log(log_fn)

    stamp = now_tag()
    out_dir = plan.out_dir
    if not out_dir:
        out_dir = os.path.join(
            plan.results_dir,
            "xfer",
            f"{plan.city_a}__{plan.city_b}",
            stamp,
        )
    ensure_dir(out_dir)

    cases = iter_cases(plan)
    caches: Dict[str, Any] = {}

    results: List[Dict[str, Any]] = []
    n = len(cases)

    for i, c in enumerate(cases, start=1):
        if callable(stop_check) and stop_check():
            break
        if callable(progress_fn):
            progress_fn(i, n)

        r = run_case(
            case=c,
            plan=plan,
            caches=caches,
            save_dir=out_dir,
            log_fn=log_fn,
        )
        if r is not None:
            results.append(r)

    js_path = os.path.join(out_dir, "xfer_results.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(out_dir, "xfer_results.csv")
    cols = [
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
        "schema.dynamic_order_mismatch",
        "schema.dynamic_reordered",
        "schema.future_order_mismatch",
        "schema.future_reordered",
    ]

    def _get(d: Dict[str, Any], k: str) -> Any:
        cur: Any = d
        for p in k.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in results:
            row = {k: _get(d, k) for k in cols}
            w.writerow(row)

    log(f"[xfer] wrote: {js_path}")
    log(f"[xfer] wrote: {csv_path}")

    return {
        "out_dir": out_dir,
        "json_path": js_path,
        "csv_path": csv_path,
        "results": results,
    }


__all__ = [
    "WarmStartConfig",
    "XferCase",
    "XferPlan",
    "build_plan",
    "iter_cases",
    "run_case",
    "run_plan",
]
