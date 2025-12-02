# nat.com/xfer_matrix.py
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations
import os, json, glob, argparse, datetime as dt
import numpy as np
import joblib
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, r2_score

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

from fusionlab._optdeps import with_progress 
from fusionlab.registry.utils import _find_stage1_manifest, reproject_dynamic_scale
from fusionlab.utils.generic_utils import ensure_directory_exists
from fusionlab.utils.scale_metrics import inverse_scale_target
from fusionlab.utils.forecast_utils import format_and_forecast

from fusionlab.nn.pinn.op import extract_physical_parameters
from fusionlab.nn.pinn.models import GeoPriorSubsNet
from fusionlab.params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
from fusionlab.nn.losses import make_weighted_pinball
from fusionlab.nn.keras_metrics import _to_py, coverage80_fn, sharpness80_fn
from fusionlab.nn.calibration import (
    IntervalCalibrator, fit_interval_calibrator_on_val,
    apply_calibrator_to_subs,
)
# from fusionlab.nn.pinn.utils import format_pinn_predictions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-city transfer evaluation matrix (A->B and B->A)."
    )
    p.add_argument("--city-a", default="nansha")
    p.add_argument("--city-b", default="zhongshan")
    p.add_argument("--results-dir", default=os.getenv("RESULTS_DIR", "results"))
    p.add_argument(
        "--splits", nargs="+", default=["val", "test"],
        choices=["val", "test"], help="Which splits to run."
    )
    p.add_argument(
        "--calib-modes", nargs="+",
        default=["none", "source", "target"],
        choices=["none", "source", "target"],
        help="Calibration modes to evaluate."
    )
    p.add_argument(
        "--rescale-to-source", action="store_true",
        help="Reproject target dynamic features to source scaler (strict domain test)."
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--quantiles", nargs="*", type=float, default=None,
                   help="Override quantiles (else read from manifest).")
    return p.parse_args()

# ------------- helpers -------------
def _load_manifest_for_city(city: str, results_dir: str) -> dict:
    mpath = _find_stage1_manifest(
        manual=None,
        base_dir=results_dir,
        city_hint=city,
        model_hint=os.getenv("MODEL_NAME_OVERRIDE", "GeoPriorSubsNet"),
        prefer="timestamp",
        required_keys=("model", "stage"),
        verbose=0,
    )
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)

def _latest_model_under(run_dir: str) -> str | None:
    # Prefer tuned models, else trained
    patt_tuned = os.path.join(run_dir, "tuning", "**", "*.keras")
    patt_trn   = os.path.join(run_dir, "train_*", "*.keras")
    cands = []
    for pat in (patt_tuned, patt_trn):
        for p in glob.glob(pat, recursive=True):
            try:
                cands.append((os.path.getmtime(p), p))
            except Exception:
                pass
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]

def _pick_npz(M: dict, which: str):
    npzs = M["artifacts"]["numpy"]
    if which == "val":
        return dict(np.load(npzs["val_inputs_npz"])), dict(np.load(npzs["val_targets_npz"]))
    if which == "test":
        ti, tt = npzs.get("test_inputs_npz"), npzs.get("test_targets_npz")
        if not ti:
            return None, None
        return dict(np.load(ti)), (dict(np.load(tt)) if tt else None)
    raise ValueError(which)

def _ensure_shapes(x: dict, mode: str, horizon: int) -> dict:
    out = dict(x)
    N = out["dynamic_features"].shape[0]
    if out.get("static_features") is None:
        out["static_features"] = np.zeros((N, 0), np.float32)
    if out.get("future_features") is None:
        t_future = out["dynamic_features"].shape[1] if mode == "tft_like" else horizon
        out["future_features"] = np.zeros((N, t_future, 0), np.float32)
    return out

def _map_targets(y: dict | None) -> dict:
    if not y:
        return {}
    if "subsidence" in y and "gwl" in y:
        return {"subs_pred": y["subsidence"], "gwl_pred": y["gwl"]}
    if "subs_pred" in y and "gwl_pred" in y:
        return y
    return {}

def _load_source_calibrator(source_run_dir: str) -> IntervalCalibrator | None:
    pats = glob.glob(os.path.join(source_run_dir, "train_*", "interval_factors_80.npy"))
    if not pats:
        return None
    p = sorted(pats, key=os.path.getmtime, reverse=True)[0]
    cal = IntervalCalibrator(target=0.80)
    cal.factors_ = np.load(p).astype(np.float32)
    return cal

def _infer_source_input_dims(M_src: dict) -> tuple[int, int]:
    """
    Infer (static_input_dim, dynamic_input_dim) for the source model
    from the Stage-1 manifest.

    Preference order:
    1. artifacts["sequences"]["dims"]["static_input_dim"/"dynamic_input_dim"]
       if present.
    2. artifacts["shapes"]["train_inputs"]["static_features"/"dynamic_features"]
       last axis.
    3. len(config["features"]["static"/"dynamic"]) as a fallback.

    Returns
    -------
    static_dim : int
    dynamic_dim : int
    """
    # 1) Try dims block first (newer manifests might define these)
    seq = (M_src.get("artifacts") or {}).get("sequences") or {}
    dims = seq.get("dims") or {}

    s_src = dims.get("static_input_dim")
    d_src = dims.get("dynamic_input_dim")

    # 2) Fallback to shapes.train_inputs
    shapes = (M_src.get("artifacts") or {}).get("shapes") or {}
    tr_in = shapes.get("train_inputs") or {}

    if s_src is None:
        sf_shape = tr_in.get("static_features")
        if isinstance(sf_shape, (list, tuple)) and len(sf_shape) >= 2:
            s_src = sf_shape[-1]

    if d_src is None:
        df_shape = tr_in.get("dynamic_features")
        if isinstance(df_shape, (list, tuple)) and len(df_shape) >= 3:
            d_src = df_shape[-1]

    # 3) Final fallback: length of feature lists from config
    feats = (M_src.get("config") or {}).get("features") or {}
    if s_src is None:
        s_src = len(feats.get("static") or [])
    if d_src is None:
        d_src = len(feats.get("dynamic") or [])

    # Make sure we always return ints
    return int(s_src or 0), int(d_src or 0)

def _align_static_to_source(
    X_tgt: dict,
    M_src: dict,
    M_tgt: dict,
) -> dict:
    """
    Reproject target static_features into the source city's static basis.

    Source model was built with static_input_dim = len(M_src['config']['features']['static']).
    Target NPZ has static_features columns corresponding to
    M_tgt['config']['features']['static'].

    We build a new static_features array of shape
    (N, len(static_src)) in the *source* order, copying any
    overlapping columns by name and filling missing ones with 0.
    """
    static_src = M_src["config"]["features"]["static"] or []
    static_tgt = M_tgt["config"]["features"]["static"] or []

    # If model has no static features, nothing to do.
    if not static_src:
        # also normalise to shape (N,0) in case target had some
        N = X_tgt["dynamic_features"].shape[0]
        X_tgt["static_features"] = np.zeros((N, 0), dtype=np.float32)
        return X_tgt

    N = X_tgt["dynamic_features"].shape[0]
    old_static = X_tgt.get("static_features")

    # If target has no static at all, just create all-zeros in source basis.
    if old_static is None or old_static.shape[-1] == 0:
        X_tgt["static_features"] = np.zeros(
            (N, len(static_src)), dtype=np.float32
        )
        return X_tgt

    # Map target names -> indices
    name2idx = {name: i for i, name in enumerate(static_tgt)}

    new_static = np.zeros((N, len(static_src)), dtype=np.float32)
    for j, name in enumerate(static_src):
        idx = name2idx.get(name, None)
        if idx is not None and idx < old_static.shape[1]:
            new_static[:, j] = old_static[:, idx]

    X_tgt["static_features"] = new_static
    return X_tgt

# ------------- core evaluation -------------
def run_one_direction(
    M_src: dict,
    M_tgt: dict,
    split: str,
    calib_mode: str,
    rescale_to_source: bool,
    batch_size: int,
    quantiles_override: list[float] | None,
):
    # Load NPZs
    X_tgt, y_tgt = _pick_npz(M_tgt, split)
    if X_tgt is None:
        return None  # no split

    cfg_t = M_tgt["config"]
    MODE  = cfg_t["MODE"]
    # T     = cfg_t["TIME_STEPS"] # Got if from _ensure_shapes dynamics, :NOqa
    H     = cfg_t["FORECAST_HORIZON_YEARS"]
    Q     = quantiles_override or cfg_t.get("QUANTILES", [0.1, 0.5, 0.9])

    # Build input shapes
    X_tgt = _ensure_shapes(X_tgt, MODE, H)
    y_map = _map_targets(y_tgt)
    
    # Align static features to the *source* feature space
    X_tgt = _align_static_to_source(X_tgt, M_src, M_tgt)
    
    # Infer source model input dims from manifest
    s_src, d_src = _infer_source_input_dims(M_src)
    
    print(
        f"[XFER] Source model expects static={s_src} "
        f"dynamic={d_src}"
    )
    
    s_tgt = int(X_tgt["static_features"].shape[-1])
    
    if s_src != s_tgt:
        raise SystemExit(
            "Static feature dimension mismatch in transfer:\n"
            f"  Source model expects static_features dim = {s_src}\n"
            f"  Target NPZ has static_features dim      = {s_tgt}\n"
            "Cross-city transfer requires Stage-1 to export identical "
            "static feature schemas (same feature list and order) for "
            "both cities. Please harmonize Stage-1 config or pad/align "
            "static_features before calling xfer_matrix.py."
        )
        
    d_tgt = int(X_tgt["dynamic_features"].shape[-1])
    if d_src != d_tgt:
        raise SystemExit(
            "Dynamic feature dimension mismatch in transfer:\n"
            f"  Source model expects dynamic_features dim = {d_src}\n"
            f"  Target NPZ has dynamic_features dim      = {d_tgt}\n"
            "Please harmonize the dynamic feature schema across cities."
        )


    # Optional strict domain test: reproject dynamic features to *source* scaling
    if rescale_to_source:
        # (1) target scaler_info can be dict or path in the target manifest
        enc_t = M_tgt["artifacts"]["encoders"]
        scaler_info = enc_t.get("scaler_info")
        if isinstance(scaler_info, str) and os.path.exists(scaler_info):
            scaler_info = joblib.load(scaler_info)
        # (2) source scaler path
        enc_s = M_src["artifacts"]["encoders"]
        src_scaler_path = enc_s.get("main_scaler")
        if not src_scaler_path:
            raise SystemExit("Source 'main_scaler' path missing in manifest.")
        # (3) dynamic feature names (order) from Stage-1 config
        dyn_names = M_tgt["config"]["features"]["dynamic"]
        X_tgt = reproject_dynamic_scale(
            X_np=X_tgt,
            target_scaler_info=scaler_info,
            source_scaler_path=src_scaler_path,
            dynamic_feature_order=dyn_names,
        )

    # Load latest model under *source* run dir
    model_path = _latest_model_under(M_src["paths"]["run_dir"])
    model_dir = os.path.dirname(model_path)

    if not model_path:
        raise SystemExit(f"No .keras found under {M_src['paths']['run_dir']}")
    custom_objects = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        # custom loss factory / class
        "make_weighted_pinball": make_weighted_pinball,
        # custom metrics used in compile
        "coverage80_fn": coverage80_fn,
        "sharpness80_fn": sharpness80_fn,
    }
    
    with custom_object_scope(custom_objects):
        model = load_model(model_path, compile=True)


    # Prepare optional calibrator
    cal = None
    if calib_mode == "source":
        cal = _load_source_calibrator(M_src["paths"]["run_dir"])
    elif calib_mode == "target":
        # fit on target VAL if exist; else fall back to split==val
        try:
            vx, vy = _pick_npz(M_tgt, "val")
            if vx is not None:
                vx = _ensure_shapes(vx, MODE, H)
                vy = _map_targets(vy)
                ds_val = tf.data.Dataset.from_tensor_slices((vx, vy)).batch(batch_size)
                cal = fit_interval_calibrator_on_val(model, ds_val, target=0.80)
        except Exception:
            cal = None
    
    # extract physic parameters 
    try: 
        extract_physical_parameters(
            model, to_csv=True,
            filename=f"{M_tgt.get('city')}_xfer_physical_parameters.csv",
            save_dir=model_dir,
            model_name="geoprior",
        )
    except: 
        pass 
    
    # Predict
    pred = model.predict(X_tgt, verbose=0)
    data_final = pred["data_final"]

    OUT_S_DIM = M_tgt["artifacts"]["sequences"]["dims"]["output_subsidence_dim"]
    # OUT_G_DIM = M_tgt["artifacts"]["sequences"]["dims"]["output_gwl_dim"]

    if data_final.ndim == 4:
        s_q = data_final[..., :OUT_S_DIM]
        g_q = data_final[..., OUT_S_DIM:]
        if cal is not None:
            s_q = apply_calibrator_to_subs(cal, s_q)
        predictions = {"subs_pred": s_q, "gwl_pred": g_q}
    else:
        s_p = data_final[..., :OUT_S_DIM]
        g_p = data_final[..., OUT_S_DIM:]
        predictions = {"subs_pred": s_p, "gwl_pred": g_p}
        
    # --- Per-horizon + overall metrics (subsidence) ---

    # Inverse-scaling + metrics in physical units
    enc = M_tgt["artifacts"]["encoders"]
    coord_scaler = None
    if enc.get("coord_scaler"):
        try:
            coord_scaler = joblib.load(enc["coord_scaler"])
        except Exception:
            pass
    scaler_info = enc.get("scaler_info")
    if isinstance(scaler_info, str) and os.path.exists(scaler_info):
        scaler_info = joblib.load(scaler_info)
    # Attach actual scaler objects, like in training_NATCOM_GEOPRIOR.py
    if isinstance(scaler_info, dict):
        for k, v in scaler_info.items():
            if isinstance(v, dict) and "scaler_path" in v and "scaler" not in v:
                p = v["scaler_path"]
                if p and os.path.exists(p):
                    try:
                        v["scaler"] = joblib.load(p)
                    except Exception:
                        # Keep going even if one scaler fails to load
                        pass
        
    cols_cfg = M_tgt["config"]["cols"]
    SUBS_COL = cols_cfg.get("subsidence", "subsidence")
    # GWL_COL  = cols_cfg.get("gwl", "GWL")
    # target_mapping = {"subs_pred": SUBS_COL, "gwl_pred": GWL_COL}
    # output_dims    = {"subs_pred": OUT_S_DIM, "gwl_pred": OUT_G_DIM}
    
    # --- Per-horizon + overall metrics (subsidence, physical units) ---
    per_horizon_mae = {}
    per_horizon_r2  = {}
    overall_mae = overall_mse = overall_r2 = None

    if y_map and ("subs_pred" in y_map) and scaler_info is not None:
        # True scaled shape: (B, H, OUT_S_DIM)
        y_true_subs_scaled = y_map["subs_pred"]

        # Use point predictions. If probabilistic, take the median quantile.
        if data_final.ndim == 4:
            # predictions["subs_pred"] is (B, H, Q, OUT_S_DIM)
            q_arr = np.asarray(Q, dtype=np.float32)
            med_idx = int(np.argmin(np.abs(q_arr - 0.5)))
            y_pred_subs_scaled = predictions["subs_pred"][:, :, med_idx, :]
        else:
            # (B, H, OUT_S_DIM)
            y_pred_subs_scaled = predictions["subs_pred"]

        # For safety, assume first subsidence channel is the one we care about
        y_true_subs_scaled = y_true_subs_scaled[..., :1]   # (B,H,1)
        y_pred_subs_scaled = y_pred_subs_scaled[..., :1]   # (B,H,1)

        # ---- inverse scale to physical units (mm / mm·yr) ----
        y_true_subs_phys = inverse_scale_target(
            y_true_subs_scaled,
            scaler_info=scaler_info,
            target_name=SUBS_COL,
        )
        y_pred_subs_phys = inverse_scale_target(
            y_pred_subs_scaled,
            scaler_info=scaler_info,
            target_name=SUBS_COL,
        )

        # Compute per-horizon metrics in physical units
        H_eff = y_true_subs_phys.shape[1]
        for h in range(H_eff):
            yt = y_true_subs_phys[:, h, :].reshape(-1)
            yp = y_pred_subs_phys[:, h, :].reshape(-1)
            per_horizon_mae[f"H{h+1}"] = float(mean_absolute_error(yt, yp))
            per_horizon_r2[f"H{h+1}"]  = float(r2_score(yt, yp))

        # Overall metrics across all horizons (flatten) in physical units
        yt_all = y_true_subs_phys.reshape(-1)
        yp_all = y_pred_subs_phys.reshape(-1)
        overall_mae = float(mean_absolute_error(yt_all, yp_all))
        overall_mse = float(np.mean((yt_all - yp_all) ** 2))
        overall_r2  = float(r2_score(yt_all, yp_all))

    # Optional scaled-space evaluation & physics
    eval_scaled = None
    physics_diag = None
    if y_map:
        ds = tf.data.Dataset.from_tensor_slices((X_tgt, y_map)).batch(batch_size)
        try:
            eval_scaled = model.evaluate(ds, return_dict=True, verbose=0)
        except Exception:
            eval_scaled = None
        
    if eval_scaled is not None: 
        print(f"Evaluated {M_tgt.get('city')}:", eval_scaled)

        # Physics diagnostics are already aggregated in eval_results
        phys_keys = ("epsilon_prior", "epsilon_cons")
        try: 
            physics_diag = {
                k: float(_to_py(eval_scaled[k]))
                for k in phys_keys
                if k in eval_scaled
            }
            if physics_diag:
                print("Physics diagnostics (from"
                      f" {M_tgt.get('city')} evaluation):", 
                      physics_diag
                )
        except: 
            physics_diag = None
    
    try:
        model.export_physics_payload(
            X_tgt,
            max_batches=None,
            save_path=os.path.join(
                model_dir, f"{M_tgt.get('city')}_xfer_physics_payload.npz"
                ),
            format="npz",
            overwrite=True,
            metadata={"city": M_tgt.get("city"), "split": "val"},
        )
    except: 
        pass 
    
    # We won't save large CSVs here; just compute coverage/sharpness if possible

    coverage80 = sharpness80 = None
    if y_map and data_final.ndim == 4 and scaler_info is not None:
        ds_calc = tf.data.Dataset.from_tensor_slices((X_tgt, y_map)).batch(batch_size)
        y_true_list, s_q_list = [], []
        for xb, yb in with_progress(
                ds_calc, desc =f"Diagnose {M_tgt.get('city')} xfer-metrics"):
            out = model(xb, training=False)
            s_q_b, _ = model.split_data_predictions(out["data_final"])
            y_true_list.append(yb["subs_pred"])   # (B,H,1)
            s_q_list.append(s_q_b)                # (B,H,Q,1)

        y_true_scaled = tf.concat(y_true_list, axis=0)   # (N,H,1)
        s_q_scaled    = tf.concat(s_q_list,    axis=0)   # (N,H,Q,1)

        # ---- inverse scale to physical units ----
        y_true_phys_np = inverse_scale_target(
            y_true_scaled,
            scaler_info=scaler_info,
            target_name=SUBS_COL,
        )
        s_q_phys_np = inverse_scale_target(
            s_q_scaled,
            scaler_info=scaler_info,
            target_name=SUBS_COL,
        )

        y_true_phys_tf = tf.convert_to_tensor(y_true_phys_np, dtype=tf.float32)
        s_q_phys_tf    = tf.convert_to_tensor(s_q_phys_np,    dtype=tf.float32)

        coverage80  = float(coverage80_fn(y_true_phys_tf, s_q_phys_tf).numpy())
        sharpness80 = float(sharpness80_fn(y_true_phys_tf, s_q_phys_tf).numpy())
        
    # ------------------------------------------------------------------
    # Format calibrated predictions with the robust helper
    #   - df_eval   : evaluation horizon on the target split (if y_true)
    #   - df_future : future horizon [FORECAST_START_YEAR, ..., +H-1]
    # ------------------------------------------------------------------
    # Build canonical y_true mapping (like training_NATCOM_GEOPRIOR.py)
    y_true_for_format = None
    if y_map:
        y_true_for_format = {}
        if "subs_pred" in y_map:
            # use the physical subsidence target name from config
            y_true_for_format["subsidence"] = y_map["subs_pred"]
        if "gwl_pred" in y_map:
            y_true_for_format["gwl"] = y_map["gwl_pred"]

    cfg_t = M_tgt["config"]
    train_end_year      = cfg_t.get("TRAIN_END_YEAR")
    forecast_start_year = cfg_t.get("FORECAST_START_YEAR")

    # Optional explicit future grid; equivalent to letting the helper build it
    future_grid = None
    if forecast_start_year is not None:
        future_grid = np.arange(
            float(forecast_start_year),
            float(forecast_start_year) + float(H),
            dtype=float,
        )

    # Per-direction CSV names (depend on source, target, split, calib-mode)
    base_name = (
        f"{M_src.get('city')}_to_{M_tgt.get('city')}_xfer_"
        f"{split}_{calib_mode}"
    )
    xfer_eval_csv = os.path.join(model_dir, base_name + "_eval.csv")
    xfer_future_csv = os.path.join(model_dir, base_name + "_future.csv")

    df_eval, df_future = format_and_forecast(
        # ---- core data ----
        y_pred=predictions,               # dict with "subs_pred", "gwl_pred"
        y_true=y_true_for_format,         # dict or None
        coords=X_tgt.get("coords", None), # (B,H,3) or None
        quantiles=Q if data_final.ndim == 4 else None,
        target_name=SUBS_COL,             # e.g. "subsidence"
        target_key_pred="subs_pred",      # key inside predictions
        component_index=0,                # first subsidence component

        # ---- scaling / coords ----
        scaler_info=scaler_info,          # Stage-1 scaler_info dict
        coord_scaler=coord_scaler,        # coord MinMax scaler
        coord_columns=("coord_t", "coord_x", "coord_y"),

        # ---- time semantics ----
        train_end_time=train_end_year,         # e.g. 2021 or 2022
        forecast_start_time=forecast_start_year,
        forecast_horizon=H,                    # FORECAST_HORIZON_YEARS
        future_time_grid=future_grid,          # or None to auto-build
        eval_forecast_step=None,               # all horizons collapsed
        eval_export="all",                     # default behaviour

        # ---- bookkeeping / CSV paths ----
        sample_index_offset=0,
        city_name=M_tgt.get("city"),
        model_name=M_src.get("model") or "GeoPriorSubsNet",
        dataset_name=f"xfer_{split}_{calib_mode}",
        csv_eval_path=xfer_eval_csv,
        csv_future_path=xfer_future_csv,

        # ---- formatting options ----
        time_as_datetime=False,
        time_format=None,
        verbose=1,

        # ---- we already compute metrics manually above ----
        eval_metrics=False,
        metrics_column_map=None,
        metrics_quantile_interval=None,
        metrics_per_horizon=False,
        metrics_extra=None,
        metrics_extra_kwargs=None,
        metrics_savefile=None,
        metrics_save_format=".json",
        metrics_time_as_str=True,

        # subsidence is stored/learned as a rate (mm/yr)
        value_mode="rate",
    )

    if df_eval is not None and not df_eval.empty:
        print(f"[xfer] Saved EVAL forecast CSV -> {xfer_eval_csv}")
    else:
        print("[xfer] No eval DF (no y_true or empty).")

    if df_future is not None and not df_future.empty:
        print(f"[xfer] Saved FUTURE forecast CSV -> {xfer_future_csv}")
    else:
        print("[xfer] No future DF (check forecast_start_year / horizon).")

    return {
        "model_path": model_path,
        "split": split,
        "calibration": calib_mode,
        "quantiles": Q if data_final.ndim == 4 else None,
        "keras_eval_scaled": eval_scaled,
        "physics": physics_diag,
        "coverage80": coverage80,
        "sharpness80": sharpness80,
        "overall_mae": overall_mae,
        "overall_mse": overall_mse,
        "overall_r2":  overall_r2,
        "per_horizon_mae": per_horizon_mae,
        "per_horizon_r2":  per_horizon_r2,
        "csv_eval": xfer_eval_csv,
        "csv_future": xfer_future_csv,
    }

# ---------------- main ----------------
def main():
    args = parse_args()
    M_A = _load_manifest_for_city(args.city_a, args.results_dir)
    M_B = _load_manifest_for_city(args.city_b, args.results_dir)

    outdir = os.path.join(args.results_dir, "xfer",
                          f"{args.city_a}_to_{args.city_b}",
                          dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ensure_directory_exists(outdir)

    results = []
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
                )
                if r is not None:
                    r["direction"] = tag
                    r["source_city"] = M_src.get("city")
                    r["target_city"] = M_tgt.get("city")
                    results.append(r)

    # Save JSON
    js = os.path.join(outdir, "xfer_results.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved transfer results JSON -> {js}")

    # Save light CSV
    import csv

    csv_path = os.path.join(outdir, "xfer_results.csv")
    
    base_cols = [
        "direction", "source_city", "target_city", "split", "calibration",
        "overall_mae", "overall_mse", "overall_r2",
        "coverage80", "sharpness80",
        "physics.epsilon_prior", "physics.epsilon_cons",
        "keras_eval_scaled.loss", "keras_eval_scaled.subs_pred_mae",
        "keras_eval_scaled.gwl_pred_mae",
    ]
    
    # Discover horizon keys present in results
    def _sorted_hkeys(keys):
        def _k(k):
            # supports 'H1','H2',... robustly
            try:
                return int(str(k).strip().split("H")[-1])
            except Exception:
                return 9999
        return sorted(keys, key=_k)
    
    h_mae_keys = set()
    h_r2_keys  = set()
    for r in results:
        h_mae_keys |= set((r.get("per_horizon_mae") or {}).keys())
        h_r2_keys  |= set((r.get("per_horizon_r2")  or {}).keys())
    
    h_mae_keys = _sorted_hkeys(h_mae_keys)
    h_r2_keys  = _sorted_hkeys(h_r2_keys)
    
    cols = base_cols + [f"per_horizon_mae.{k}" for k in h_mae_keys] \
                     + [f"per_horizon_r2.{k}"  for k in h_r2_keys]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            phys = r.get("physics") or {}
            ev   = r.get("keras_eval_scaled") or {}
    
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
                phys.get("epsilon_prior"),
                phys.get("epsilon_cons"),
                ev.get("loss"),
                ev.get("subs_pred_mae"),
                ev.get("gwl_pred_mae"),
            ]
    
            # per-horizon values in stable column order (fill with 'NA' if absent)
            ph_mae = r.get("per_horizon_mae") or {}
            ph_r2  = r.get("per_horizon_r2")  or {}
    
            row.extend([ph_mae.get(k, "NA") for k in h_mae_keys])
            row.extend([ph_r2.get(k,  "NA") for k in h_r2_keys])
            w.writerow(row)
    
    print(f"Saved transfer CSV -> {csv_path}")


if __name__ == "__main__":
    main()
