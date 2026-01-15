# nat/com/xfer_matrix.py
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf

from fusionlab._optdeps import with_progress
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
            "Cross-city transfer matrix (A->B and B->A)."
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
        help="Which splits to run.",
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
        "--rescale-to-source",
        action="store_true",
        help=(
            "Reproject target dynamic features to the "
            "source scaling (strict domain test)."
        ),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--quantiles",
        nargs="*",
        type=float,
        default=None,
        help="Override quantiles (else read manifest).",
    )
    return p.parse_args()


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


def run_one_direction(
    *,
    M_src: Dict[str, Any],
    M_tgt: Dict[str, Any],
    split: str,
    calib_mode: str,
    rescale_to_source: bool,
    batch_size: int,
    quantiles_override: List[float] | None,
    save_dir: str,
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

    X_tgt = _ensure_np_inputs(X_tgt, mode, H)
    y_map = map_targets_for_training(y_tgt or {})

    X_tgt = _align_static_to_source(X_tgt, M_src, M_tgt)

    s_src, d_src, f_src = _infer_input_dims(M_src)

    s_tgt = int(X_tgt["static_features"].shape[-1])
    d_tgt = int(X_tgt["dynamic_features"].shape[-1])
    f_tgt = int(X_tgt["future_features"].shape[-1])

    if s_src != s_tgt:
        raise SystemExit(
            "Static dim mismatch in transfer:\n"
            f"  source expects {s_src}\n"
            f"  target has      {s_tgt}\n"
            "Pad/align static_features to source schema."
        )
    if d_src != d_tgt:
        raise SystemExit(
            "Dynamic dim mismatch in transfer:\n"
            f"  source expects {d_src}\n"
            f"  target has      {d_tgt}\n"
            "Harmonize dynamic feature schema."
        )
    if f_src != f_tgt:
        raise SystemExit(
            "Future dim mismatch in transfer:\n"
            f"  source expects {f_src}\n"
            f"  target has      {f_tgt}\n"
            "Harmonize future feature schema."
        )

    if rescale_to_source:
        X_tgt = _reproject_dynamic_to_source(
            X_tgt,
            M_src,
            M_tgt,
        )

    model, model_pred, bundle = _load_source_model(
        M_src=M_src,
        X_sample=X_tgt,
        quantiles=Q,
    )

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
        f"xfer_{split}_{calib_mode}"
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
            f"xfer_{split}_{calib_mode}"
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
    }


def main() -> None:
    args = parse_args()

    M_A = _load_manifest_for_city(
        args.city_a,
        args.results_dir,
    )
    M_B = _load_manifest_for_city(
        args.city_b,
        args.results_dir,
    )

    outdir = os.path.join(
        args.results_dir,
        "xfer",
        f"{args.city_a}_to_{args.city_b}",
        dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    ensure_directory_exists(outdir)

    results: List[Dict[str, Any]] = []

    directions = [
        ("A_to_B", M_A, M_B),
        ("B_to_A", M_B, M_A),
    ]

    for tag, M_src, M_tgt in directions:
        for split in args.splits:
            for cm in args.calib_modes:
                r = run_one_direction(
                    M_src=M_src,
                    M_tgt=M_tgt,
                    split=split,
                    calib_mode=cm,
                    rescale_to_source=args.rescale_to_source,
                    batch_size=args.batch_size,
                    quantiles_override=args.quantiles,
                    save_dir=outdir,
                )
                if r is None:
                    continue
                r["direction"] = tag
                r["source_city"] = M_src.get("city")
                r["target_city"] = M_tgt.get("city")
                results.append(r)

    js = os.path.join(outdir, "xfer_results.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    import csv

    csv_path = os.path.join(outdir, "xfer_results.csv")

    base_cols = [
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
            row = [
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
