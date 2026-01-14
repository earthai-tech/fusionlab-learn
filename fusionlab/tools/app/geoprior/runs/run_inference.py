# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import datetime as dt
import json
import os
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    TYPE_CHECKING,
)

import joblib
import numpy as np
import tensorflow as tf

from .....compat.keras import (
    load_inference_model,
    load_model_from_tfv2,
)
from .....nn.calibration import (
    IntervalCalibrator,
    apply_calibrator_to_subs,
    fit_interval_calibrator_on_val,
)
from .....nn.keras_metrics import (
    coverage80_fn,
    sharpness80_fn,
)
from .....nn.pinn.geoprior.models import GeoPriorSubsNet
from .....plot.forecast import plot_eval_future
from .....utils.forecast_utils import format_and_forecast
from .....utils.generic_utils import ensure_directory_exists
from .....utils.nat_utils import (
    ensure_input_shapes,
    extract_preds,
    load_best_hps_near_model,
    map_targets_for_training,
    pick_npz_for_dataset,
    sanitize_inputs_np,
)
from .....utils.scale_metrics import (
    inverse_scale_target,
    per_horizon_metrics,
    point_metrics,
)

from ..utils.view_utils import _notify_gui_forecast_views

if TYPE_CHECKING:
    from ..config.store import GeoConfigStore

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


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


def _store_cfg_overrides(
    store: Optional["GeoConfigStore"],
) -> Dict[str, Any]:
    if store is None:
        return {}

    # Prefer a dedicated "overrides" snapshot API if present.
    for name in (
        "snapshot_overrides",
        "to_overrides",
        "overrides_snapshot",
        "get_overrides",
    ):
        fn = getattr(store, name, None)
        if callable(fn):
            try:
                out = fn()
                return dict(out) if isinstance(out, dict) else {}
            except Exception:
                return {}

    # Fallback: store.snapshot() or store.to_dict()
    for name in ("snapshot", "to_dict", "as_dict", "dump"):
        fn = getattr(store, name, None)
        if callable(fn):
            try:
                out = fn()
            except Exception:
                continue

            if isinstance(out, dict):
                # Some stores wrap as {"config": {...}}
                cfg = out.get("config", out)
                return dict(cfg) if isinstance(cfg, dict) else {}

    # Fallback: store.config (GeoPriorConfig) -> dict
    cfg_obj = getattr(store, "config", None)
    if cfg_obj is not None:
        for name in ("to_dict", "as_dict", "dict"):
            fn = getattr(cfg_obj, name, None)
            if callable(fn):
                try:
                    out = fn()
                    return dict(out) if isinstance(out, dict) else {}
                except Exception:
                    pass
        try:
            return dict(cfg_obj)
        except Exception:
            pass

    return {}


def _merge_cfg(
    base: Dict[str, Any],
    *,
    store: Optional["GeoConfigStore"],
    cfg_overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = dict(base)

    live = _store_cfg_overrides(store)
    if live:
        cfg.update(live)

    if cfg_overrides:
        cfg.update(cfg_overrides)

    return cfg


def _merge_forecasts(df_a, df_b):
    if df_a is None or df_a.empty:
        return df_b
    if df_b is None or df_b.empty:
        return df_a

    join_cols = []
    for c in (
        "coord_t",
        "coord_x",
        "coord_y",
        "sample_index",
        "sample_id",
    ):
        if c in df_a.columns and c in df_b.columns:
            join_cols.append(c)

    if join_cols:
        b_cols = [
            c
            for c in df_b.columns
            if c not in join_cols and c not in df_a.columns
        ]
        if not b_cols:
            return df_a
        return df_a.merge(
            df_b[join_cols + b_cols],
            on=join_cols,
        )

    b_cols = [
        c for c in df_b.columns
        if c not in df_a.columns
    ]
    if not b_cols:
        return df_a
    return df_a.join(df_b[b_cols])


def run_inference(
    model_path: str,
    dataset: str = "test",
    use_stage1_future_npz: bool = False,
    *,
    manifest_path: Optional[str] = None,
    stage1_dir: Optional[str] = None,
    inputs_npz: Optional[str] = None,
    targets_npz: Optional[str] = None,
    use_source_calibrator: bool = False,
    calibrator_path: Optional[str] = None,
    fit_calibrator: bool = False,
    cov_target: float = 0.80,
    include_gwl: bool = False,
    batch_size: int = 32,
    make_plots: bool = True,
    store: Optional["GeoConfigStore"] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[
        Callable[[float, str], None]
    ] = None,
    **kws,
) -> Dict[str, Any]:
    def log(msg: str) -> None:
        (logger or print)(msg)

    def should_stop() -> bool:
        return bool(stop_check and stop_check())

    def progress(value: float, message: str) -> None:
        if progress_callback is None:
            return
        try:
            v = float(value)
        except Exception:
            v = 0.0
        v = max(0.0, min(1.0, v))
        try:
            progress_callback(v, message)
        except Exception:
            pass

    # 0) manifest
    if manifest_path is not None:
        mf = os.path.abspath(manifest_path)
    elif stage1_dir is not None:
        mf = os.path.join(stage1_dir, "manifest.json")
        if not os.path.exists(mf):
            raise FileNotFoundError(
                "manifest.json not found in "
                f"stage1_dir={stage1_dir!r}"
            )
        mf = os.path.abspath(mf)
    else:
        raise ValueError(
            "run_inference requires 'manifest_path' "
            "or 'stage1_dir' in the GeoPrior GUI."
        )

    with open(mf, "r", encoding="utf-8") as f:
        M = json.load(f)

    cfg_base = dict(M.get("config", {}))
    cfg = _merge_cfg(
        cfg_base,
        store=store,
        cfg_overrides=cfg_overrides,
    )

    city = M.get("city", cfg.get("CITY_NAME", "unknown"))
    model_name = M.get("model", "GeoPriorSubsNet")

    mode = cfg["MODE"]
    H = int(cfg["FORECAST_HORIZON_YEARS"])
    fsy = cfg.get("FORECAST_START_YEAR")
    tey = cfg.get("TRAIN_END_YEAR")
    qs = cfg.get("QUANTILES", [0.1, 0.5, 0.9])

    dims = M["artifacts"]["sequences"]["dims"]
    out_s = int(dims["output_subsidence_dim"])
    out_g = int(dims["output_gwl_dim"])

    log(
        f"[Manifest] city={city} model={model_name} "
        f"(mode={mode}, H={H}, FSY={fsy}, Q={qs})"
    )
    progress(0.05, "Inference: manifest loaded")

    if should_stop():
        log("[Inference] stop_check=True; abort.")
        return {
            "run_dir": None,
            "manifest_path": mf,
            "city": city,
            "model": model_name,
            "dataset": dataset,
        }

    # 1) scalers
    enc = M["artifacts"]["encoders"]

    coord_scaler = None
    cs_path = enc.get("coord_scaler")
    if cs_path and os.path.exists(cs_path):
        try:
            coord_scaler = joblib.load(cs_path)
        except Exception:
            pass

    scaler_info = enc.get("scaler_info")
    if isinstance(scaler_info, str):
        if os.path.exists(scaler_info):
            try:
                scaler_info = joblib.load(scaler_info)
            except Exception:
                pass

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

    progress(0.10, "Inference: scalers loaded")

    # 2) output dir
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    inf_dir = os.path.join(
        M["paths"]["run_dir"],
        "inference",
        f"run_{stamp}",
    )
    ensure_directory_exists(inf_dir)
    log(f"[Inference] Outputs -> {inf_dir}")
    progress(0.12, "Inference: output directory ready")

    if should_stop():
        log("[Inference] stop_check=True; abort.")
        return {
            "run_dir": inf_dir,
            "manifest_path": mf,
            "city": city,
            "model": model_name,
            "dataset": dataset,
        }

    # 3) dataset
    npz = M["artifacts"]["numpy"]

    if use_stage1_future_npz:
        x_p = npz.get("future_inputs_npz")
        y_p = npz.get("future_targets_npz")
        if not x_p or not os.path.exists(x_p):
            raise FileNotFoundError(
                "Stage-1 future NPZ missing. "
                "Re-run Stage-1 with BUILD_FUTURE_NPZ=True."
            )
        X = dict(np.load(x_p))
        if y_p and os.path.exists(y_p):
            y = dict(np.load(y_p))
        else:
            y = None
        dataset = "future"

    elif dataset == "custom":
        if not inputs_npz:
            raise ValueError(
                "inputs_npz is required for dataset='custom'."
            )
        X = dict(np.load(inputs_npz))
        y = None
        if targets_npz:
            y = dict(np.load(targets_npz))

    else:
        X, y = pick_npz_for_dataset(M, dataset)
        if X is None:
            raise RuntimeError(
                f"No NPZs for dataset={dataset!r}."
            )

    X = sanitize_inputs_np(X)
    X = ensure_input_shapes(X, mode, H)
    y_map = map_targets_for_training(y or {})

    progress(0.20, f"Inference: dataset {dataset!r} ready")

    # 4) val dataset for calibrator
    ds_val = None
    if fit_calibrator:
        vx_p = npz.get("val_inputs_npz")
        vy_p = npz.get("val_targets_npz")

        have_val = (
            vx_p
            and vy_p
            and os.path.exists(vx_p)
            and os.path.exists(vy_p)
        )

        if have_val:
            vx = dict(np.load(vx_p))
            vy = dict(np.load(vy_p))
            vx = sanitize_inputs_np(vx)
            vx = ensure_input_shapes(vx, mode, H)
            vy = map_targets_for_training(vy)
            ds_val = tf.data.Dataset.from_tensor_slices(
                (vx, vy)
            ).batch(batch_size)
        else:
            log("[Calibrator] VAL NPZs missing; skip fit.")

    progress(0.30, "Inference: val dataset ready")

    if should_stop():
        log("[Inference] stop_check=True; abort.")
        return {
            "run_dir": inf_dir,
            "manifest_path": mf,
            "city": city,
            "model": model_name,
            "dataset": dataset,
        }

    # 5) model load/rebuild
    bundle = _resolve_bundle_paths(model_path)

    init_m = {}
    init_p = bundle["init_manifest_path"]
    if os.path.isfile(init_p):
        with open(init_p, "r", encoding="utf-8") as f:
            init_m = json.load(f)

    init_cfg = init_m.get("config", {})
    if not isinstance(init_cfg, dict):
        init_cfg = {}
    geo_cfg = init_cfg.get("geoprior", {})
    if not isinstance(geo_cfg, dict):
        geo_cfg = {}

    q_model = init_cfg.get("quantiles", qs)
    h_model = int(init_cfg.get("forecast_horizon", H))
    mode_model = init_cfg.get("mode", mode)

    pde_def = cfg.get("PDE_MODE_CONFIG")
    if pde_def is None:
        pde_def = cfg.get("PDE_MODE", "basic")

    best_hps = load_best_hps_near_model(
        bundle["keras_path"] or bundle["run_dir"]
    ) or {}

    s_dim = int(
        X.get("static_features", np.zeros((1, 0))).shape[-1]
    )
    d_dim = int(X["dynamic_features"].shape[-1])

    f_feat = X.get(
        "future_features",
        np.zeros((1, H, 0)),
    )
    f_dim = int(f_feat.shape[-1])

    fixed = dict(
        static_input_dim=s_dim,
        dynamic_input_dim=d_dim,
        future_input_dim=f_dim,
        output_subsidence_dim=out_s,
        output_gwl_dim=out_g,
        forecast_horizon=h_model,
        quantiles=q_model,
        mode=mode_model,
        pde_mode=init_cfg.get(
            "pde_mode",
            pde_def,
        ),
        bounds_mode=init_cfg.get(
            "bounds_mode",
            cfg.get("BOUNDS_MODE", "soft"),
        ),
        residual_method=init_cfg.get(
            "residual_method",
            cfg.get("RESIDUAL_METHOD", "autodiff"),
        ),
        time_units=init_cfg.get(
            "time_units",
            cfg.get("TIME_UNITS", "years"),
        ),
        scale_pde_residuals=bool(
            init_cfg.get(
                "scale_pde_residuals",
                cfg.get("SCALE_PDE_RESIDUALS", True),
            )
        ),
        use_effective_h=bool(
            init_cfg.get(
                "use_effective_h",
                cfg.get("USE_EFFECTIVE_H", False),
            )
        ),
        hd_factor=float(init_cfg.get("hd_factor", 1.0)),
        offset_mode=init_cfg.get(
            "offset_mode",
            cfg.get("OFFSET_MODE", "mul"),
        ),
        scaling_kwargs=init_cfg.get("scaling_kwargs", None),
    )
    fixed.update(geo_cfg)

    allowed = {
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
    }
    hps = {
        k: v for k, v in best_hps.items()
        if k in allowed
    }

    def builder() -> GeoPriorSubsNet:
        params = dict(fixed)
        params.update(hps)
        return GeoPriorSubsNet(**params)

    mp_init = None
    if os.path.isfile(init_p):
        mp_init = init_p

    log(f"[Model] Loading: {bundle['keras_path']}")
    model = load_inference_model(
        keras_path=bundle["keras_path"],
        weights_path=bundle["weights_path"],
        manifest_path=mp_init,
        manifest=init_m if init_m else None,
        builder=builder,
        build_inputs=X,
        out_s_dim=out_s,
        out_g_dim=out_g,
        mode=mode_model,
        horizon=h_model,
        prefer_full_model=False,
    )
    model_pred = model

    tf_dir = bundle["tf_dir"]
    if tf_dir is not None:
        try:
            model_pred = load_model_from_tfv2(
                tf_dir,
                endpoint="serve",
            )
            log(f"[OK] TF endpoint: {tf_dir}")
        except Exception as e:
            log(f"[Warn] TF endpoint failed: {e}")
            model_pred = model

    progress(0.45, "Inference: model ready")

    if should_stop():
        log("[Inference] stop_check=True; abort.")
        return {
            "run_dir": inf_dir,
            "manifest_path": mf,
            "city": city,
            "model": model_name,
            "dataset": dataset,
        }

    # 6) calibrator
    cal = None

    if use_source_calibrator and not calibrator_path:
        cand = os.path.join(
            os.path.dirname(os.path.abspath(model_path)),
            "interval_factors_80.npy",
        )
        if os.path.exists(cand):
            cal = IntervalCalibrator(target=cov_target)
            cal.factors_ = np.load(cand).astype(np.float32)
            log(f"[Calibrator] Loaded: {cand}")

    if cal is None:
        if (
            calibrator_path
            and os.path.exists(calibrator_path)
        ):
            cal = IntervalCalibrator(target=cov_target)
            fac = np.load(calibrator_path)
            cal.factors_ = fac.astype(np.float32)
            log(f"[Calibrator] Loaded: {calibrator_path}")

    if cal is None and fit_calibrator and ds_val is not None:
        log("[Calibrator] Fitting on validation set...")
        cal = fit_interval_calibrator_on_val(
            model,
            ds_val,
            target=cov_target,
        )

    progress(0.55, "Inference: calibrator ready")

    if should_stop():
        log("[Inference] stop_check=True; abort.")
        return {
            "run_dir": inf_dir,
            "manifest_path": mf,
            "city": city,
            "model": model_name,
            "dataset": dataset,
        }

    # 7) predict
    log("[Inference] Running predict(...)")
    progress(0.60, "Inference: predicting")

    pred_dict = model_pred.predict(X, verbose=0)
    if not isinstance(pred_dict, dict):
        raise TypeError("predict() must return a dict.")

    subs_pred, gwl_pred = extract_preds(model, pred_dict)

    is_q = getattr(subs_pred, "ndim", 0) == 4
    if cal is not None and is_q:
        subs_pred = apply_calibrator_to_subs(cal, subs_pred)

    y_pred = {"subs_pred": subs_pred, "gwl_pred": gwl_pred}
    progress(0.70, "Inference: predictions ready")

    # 8) CSV forecast
    subs_kind = cfg.get("SUBSIDENCE_KIND", "cumulative")
    subs_kind = str(subs_kind).lower()
    if subs_kind not in ("cumulative", "rate"):
        subs_kind = "cumulative"

    cols = cfg.get("cols", cfg_base.get("cols", {})) or {}
    subs_col = cols.get("subsidence", "subsidence")
    gwl_col = cols.get("gwl", "GWL")

    base = f"{city}_{model_name}_inf_{dataset}_H{H}"
    if cal is not None:
        base += "_cal"

    csv_eval = os.path.join(inf_dir, base + "_eval.csv")
    csv_fut = os.path.join(inf_dir, base + "_future.csv")

    fut_grid = None
    if fsy is not None and H is not None:
        fut_grid = np.arange(fsy, fsy + H, dtype=float)

    y_true_fmt: Dict[str, Any] = {}
    if y_map:
        y_true_fmt = {
            "subsidence": y_map.get("subs_pred"),
            "gwl": y_map.get("gwl_pred"),
        }

    df_eval, df_fut = format_and_forecast(
        y_pred=y_pred,
        y_true=y_true_fmt or None,
        coords=X.get("coords", None),
        quantiles=qs if qs else None,
        target_name=subs_col,
        scaler_target_name=subs_col,
        output_target_name="subsidence",
        target_key_pred="subs_pred",
        component_index=0,
        scaler_info=scaler_info,
        coord_scaler=coord_scaler,
        coord_columns=("coord_t", "coord_x", "coord_y"),
        train_end_time=tey,
        forecast_start_time=fsy,
        forecast_horizon=H,
        future_time_grid=fut_grid,
        dataset_name_for_forecast=dataset,
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

    if include_gwl:
        df_ev_g, df_fu_g = format_and_forecast(
            y_pred=y_pred,
            y_true=y_true_fmt or None,
            coords=X.get("coords", None),
            quantiles=qs if qs else None,
            target_name=gwl_col,
            scaler_target_name=gwl_col,
            output_target_name="gwl",
            target_key_pred="gwl_pred",
            component_index=0,
            scaler_info=scaler_info,
            coord_scaler=coord_scaler,
            coord_columns=("coord_t", "coord_x", "coord_y"),
            train_end_time=tey,
            forecast_start_time=fsy,
            forecast_horizon=H,
            future_time_grid=fut_grid,
            dataset_name_for_forecast=dataset,
            csv_eval_path=None,
            csv_future_path=None,
            eval_metrics=False,
            value_mode="cumulative",
            input_value_mode="cumulative",
            output_unit="m",
            output_unit_from="m",
            output_unit_mode="overwrite",
            output_unit_col="gwl_unit",
        )

        df_eval = _merge_forecasts(df_eval, df_ev_g)
        df_fut = _merge_forecasts(df_fut, df_fu_g)

        if df_eval is not None and not df_eval.empty:
            df_eval.to_csv(csv_eval, index=False)
        if df_fut is not None and not df_fut.empty:
            df_fut.to_csv(csv_fut, index=False)

    if df_fut is not None and not df_fut.empty:
        main_csv = csv_fut
    else:
        main_csv = csv_eval

    progress(0.75, "Inference: CSVs written")

    # 9) diagnostics
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eval_json: Dict[str, Any] = {
        "timestamp": ts,
        "dataset": dataset,
        "quantiles": qs if is_q else None,
        "coverage80": None,
        "sharpness80": None,
        "coverage80_phys": None,
        "sharpness80_phys": None,
        "point_metrics_phys": None,
        "per_horizon_phys": None,
    }

    point_phys = None
    per_h_mae = None
    per_h_r2 = None

    if y_map and is_q:
        y_t = tf.convert_to_tensor(
            y_map.get("subs_pred"),
            dtype=tf.float32,
        )
        s_q = tf.convert_to_tensor(
            subs_pred,
            dtype=tf.float32,
        )

        eval_json["coverage80"] = float(
            coverage80_fn(y_t, s_q).numpy()
        )
        eval_json["sharpness80"] = float(
            sharpness80_fn(y_t, s_q).numpy()
        )

        y_phys = inverse_scale_target(
            y_t,
            scaler_info=scaler_info,
            target_name=subs_col,
        )
        s_phys = inverse_scale_target(
            s_q,
            scaler_info=scaler_info,
            target_name=subs_col,
        )
        y_phys_t = tf.convert_to_tensor(
            y_phys,
            dtype=tf.float32,
        )
        s_phys_t = tf.convert_to_tensor(
            s_phys,
            dtype=tf.float32,
        )

        eval_json["coverage80_phys"] = float(
            coverage80_fn(y_phys_t, s_phys_t).numpy()
        )
        eval_json["sharpness80_phys"] = float(
            sharpness80_fn(y_phys_t, s_phys_t).numpy()
        )

        q_arr = np.asarray(qs, dtype=float)
        med = int(np.argmin(np.abs(q_arr - 0.5)))
        s_med = s_phys[..., med, :]

        point_phys = point_metrics(y_phys, s_med)
        ph = per_horizon_metrics(y_phys, s_med)
        per_h_mae = ph.get("mae")
        per_h_r2 = ph.get("r2")

        eval_json["point_metrics_phys"] = point_phys
        eval_json["per_horizon_phys"] = {
            "mae": per_h_mae,
            "r2": per_h_r2,
        }

    progress(0.90, "Inference: diagnostics done")

    summ_p = os.path.join(inf_dir, "inference_summary.json")
    with open(summ_p, "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2)
    log(f"[Inference] Summary JSON -> {summ_p}")
    progress(0.95, "Inference: summary JSON written")

    # 10) plots
    has_df = (
        (df_eval is not None and not df_eval.empty)
        or (df_fut is not None and not df_fut.empty)
    )

    if make_plots and has_df and not should_stop():
        log("[Inference] Plotting forecast views...")
        try:
            eval_years = [tey] if tey is not None else None
            fut_years = fut_grid
            is_cum = subs_kind == "cumulative"

            q_plot = None
            if isinstance(qs, list):
                q_plot = qs

            plot_eval_future(
                df_eval=df_eval,
                df_future=df_fut,
                target_name=subs_col,
                quantiles=q_plot,
                spatial_cols=("coord_x", "coord_y"),
                time_col="coord_t",
                eval_years=eval_years,
                future_years=fut_years,
                eval_view_quantiles=[0.5],
                future_view_quantiles=qs,
                spatial_mode="hexbin",
                hexbin_gridsize=40,
                savefig_prefix=os.path.join(
                    inf_dir,
                    f"{city}_subsidence_view",
                ),
                save_fmts=[".png", ".pdf"],
                show=False,
                verbose=1,
                cumulative=is_cum,
                _logger=log,
            )
            _notify_gui_forecast_views(inf_dir, city)
        except Exception as e:
            log(f"[Warn] plot_eval_future failed: {e}")

    progress(1.0, "Inference: complete")

    return {
        "run_dir": inf_dir,
        "manifest_path": mf,
        "city": city,
        "model": model_name,
        "dataset": dataset,
        "model_path": model_path,
        "csv_path": main_csv,
        "csv_eval_path": csv_eval,
        "csv_future_path": csv_fut,
        "inference_summary_json": summ_p,
        "coverage80": eval_json.get("coverage80"),
        "sharpness80": eval_json.get("sharpness80"),
        "coverage80_phys": eval_json.get("coverage80_phys"),
        "sharpness80_phys": eval_json.get("sharpness80_phys"),
        "point_metrics_phys": point_phys,
        "per_horizon_phys": {
            "mae": per_h_mae,
            "r2": per_h_r2,
        },
    }
