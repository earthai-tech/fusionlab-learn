from __future__ import annotations


import os 
import json 
import warnings 
import datetime as dt 
import joblib 
import numpy as np 

from typing import Optional, Dict, Any, Callable
import tensorflow as tf

from ....._optdeps import with_progress 
from .....utils.generic_utils import ensure_directory_exists
from .....utils.scale_metrics import (
    inverse_scale_target,
    point_metrics,
    per_horizon_metrics,
)
from .....utils.forecast_utils import format_and_forecast

from .....nn.keras_metrics import ( 
    coverage80_fn,
    sharpness80_fn 
)
from .....nn.calibration import (
    IntervalCalibrator,
    fit_interval_calibrator_on_val,
    apply_calibrator_to_subs,
)

from .....plot.forecast import plot_eval_future
from .....utils.nat_utils import (
    load_geoprior_for_inference, 
    pick_npz_for_dataset, 
    sanitize_inputs_np, 
    map_targets_for_training, 
    ensure_input_shapes, 
  )

from ..utils.view_utils import _notify_gui_forecast_views 

# --- quiet logs ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

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
    cfg_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,  
    **kws
) -> Dict[str, Any]:
    """
    Run Stage-3 inference for GeoPriorSubsNet in a GUI-friendly way.

    Parameters
    ----------
    model_path :
        Path to the trained/tuned `.keras` model.
    dataset :
        One of {"train", "val", "test", "custom"}.
    use_stage1_future_npz :
        If True, ignore ``dataset`` / ``inputs_npz`` and instead
        load the future NPZs recorded in the Stage-1 manifest
        (generated when ``BUILD_FUTURE_NPZ=True`` in Stage-1).
        If those artifacts are missing, a clear error is logged
        and a ``FileNotFoundError`` is raised.
    manifest_path :
        Optional explicit path to Stage-1 `manifest.json`.
    stage1_dir :
        Optional Stage-1 directory (must contain `manifest.json`).
    inputs_npz, targets_npz :
        Used only when `dataset="custom"`.
    use_source_calibrator :
        If True and no `calibrator_path` is given, try to load
        `interval_factors_80.npy` from the model directory.
    calibrator_path :
        Explicit `.npy` file with calibrator factors.
    fit_calibrator :
        If True, fit calibrator on val split (if available).
    cov_target :
        Target coverage level for the interval calibrator.
    include_gwl :
        Whether to include GWL columns in the formatted CSV.
    batch_size :
        Batch size for Dataset / inference loops.
    make_plots :
        If True, generate spatial & temporal plots.
    cfg_overrides :
        Optional shallow overrides for `M["config"]`.
    logger :
        Optional logging callable; defaults to `print`.
    stop_check :
        Optional callable returning True when caller wants to abort
        (e.g., GUI "Cancel" button).
    progress_callback : callable or None, default=None
        Optional callable ``progress_callback(value, message)`` used
        by the GUI. ``value`` is a float in [0, 1] encoding the global
        inference progress; ``message`` is a short status string.
    Returns
    -------
    out : dict
        Dictionary with paths to artifacts and main metrics.
    """
    # ----------------------------------------------------------
    # Small helpers
    # ----------------------------------------------------------
    def log(msg: str) -> None:
        (logger or print)(msg)

    def should_stop() -> bool:
        return bool(stop_check and stop_check())
    
    def _progress(value: float, message: str) -> None:
        """Safe wrapper around the GUI progress callback."""
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
            # Never crash inference because the GUI callback misbehaved.
            pass

    # ----------------------------------------------------------
    # 0) Resolve Stage-1 manifest
    # ----------------------------------------------------------
    if manifest_path is not None:
        manifest_file = os.path.abspath(manifest_path)
    elif stage1_dir is not None:
        manifest_file = os.path.join(stage1_dir, "manifest.json")
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(
                f"manifest.json not found in stage1_dir={stage1_dir!r}"
            )
        manifest_file = os.path.abspath(manifest_file)
    else:
        # In the GUI pipeline we always know which Stage-1 run
        # we are continuing from, so environment-based discovery
        # is no longer supported here.
        raise ValueError(
            "run_inference requires either 'manifest_path' or "
            "'stage1_dir' when used from the GeoPrior GUI."
        )

    with open(manifest_file, "r", encoding="utf-8") as f:
        M = json.load(f)

    cfg = dict(M["config"])
    if cfg_overrides:
        cfg.update(cfg_overrides)

    CITY = M.get("city", cfg.get("CITY_NAME", "unknown_city"))
    MODEL_NAME = M.get("model", "GeoPriorSubsNet")

    MODE = cfg["MODE"]
    H = cfg["FORECAST_HORIZON_YEARS"]
    FSY = cfg.get("FORECAST_START_YEAR")
    TRAIN_END_YEAR = cfg.get("TRAIN_END_YEAR")
    QUANTILES = cfg.get("QUANTILES", [0.1, 0.5, 0.9])

    OUT_S_DIM = M["artifacts"]["sequences"]["dims"]["output_subsidence_dim"]
    OUT_G_DIM = M["artifacts"]["sequences"]["dims"]["output_gwl_dim"]

    log(
        f"[Manifest] Loaded city={CITY} model={MODEL_NAME} "
        f"(MODE={MODE}, H={H}, FSY={FSY}, QUANTILES={QUANTILES})"
    )
    _progress(0.05, "Inference: manifest loaded")

    if should_stop():
        log("[Inference] stop_check=True after manifest load; aborting.")
        return {
            "run_dir": None,
            "manifest_path": manifest_file,
            "city": CITY,
            "model": MODEL_NAME,
            "dataset": dataset,
        }
    # ----------------------------------------------------------
    # 1) Load encoders / scalers
    # ----------------------------------------------------------
    enc = M["artifacts"]["encoders"]

    coord_scaler = None
    cs_path = enc.get("coord_scaler")
    if cs_path and os.path.exists(cs_path):
        try:
            coord_scaler = joblib.load(cs_path)
        except Exception:
            pass

    scaler_info = enc.get("scaler_info")
    if isinstance(scaler_info, str) and os.path.exists(scaler_info):
        try:
            scaler_info = joblib.load(scaler_info)
        except Exception:
            pass

    if isinstance(scaler_info, dict):
        for k, v in scaler_info.items():
            if isinstance(v, dict) and "scaler_path" in v and "scaler" not in v:
                p = v["scaler_path"]
                if p and os.path.exists(p):
                    try:
                        v["scaler"] = joblib.load(p)
                    except Exception:
                        pass
                    
    _progress(0.10, "Inference: encoders/scalers loaded")

    # ----------------------------------------------------------
    # 2) Output directory
    # ----------------------------------------------------------
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    inf_dir = os.path.join(M["paths"]["run_dir"], "inference", f"run_{stamp}")
    ensure_directory_exists(inf_dir)
    log(f"[Inference] Outputs will be written to: {inf_dir}")

    _progress(0.12, "Inference: output directory prepared")
    
    if should_stop():
        log("[Inference] stop_check=True before data loading; aborting.")
        return {
            "run_dir": inf_dir,
            "manifest_path": manifest_file,
            "city": CITY,
            "model": MODEL_NAME,
            "dataset": dataset,
        }

    # -------------------------------------------------------------
    # 3) Choose dataset (train/val/test/custom or Stage-1 future)
    # -------------------------------------------------------------
    npz_dict = M["artifacts"]["numpy"]

    if use_stage1_future_npz:
        # Try to locate future_* NPZs written by Stage-1
        fut_inputs = npz_dict.get("future_inputs_npz")
        fut_targets = npz_dict.get("future_targets_npz")

        if not fut_inputs or not os.path.exists(fut_inputs):
            log(
                "[Inference] use_stage1_future_npz=True but no "
                "'future_inputs_npz' entry found in manifest or "
                "the file is missing.\n"
                "           Re-run Stage-1 with BUILD_FUTURE_NPZ=True "
                "to generate future NPZs."
            )
            raise FileNotFoundError(
                "Stage-1 future_* NPZs are not available. "
                "Run Stage-1 with BUILD_FUTURE_NPZ=True first."
            )

        log(f"[Inference] Using Stage-1 future NPZs: {fut_inputs}")
        X = dict(np.load(fut_inputs))
        y = (
            dict(np.load(fut_targets))
            if fut_targets and os.path.exists(fut_targets)
            else None
        )
        # For naming / logging downstream
        dataset = "future"

    elif dataset == "custom":
        if not inputs_npz:
            raise ValueError(
                "--inputs-npz (inputs_npz) required for dataset='custom'"
            )
        X = dict(np.load(inputs_npz))
        y = dict(np.load(targets_npz)) if targets_npz else None
    else:
        X, y = pick_npz_for_dataset(M, dataset)
        if X is None:
            raise RuntimeError(
                f"No NPZs found for dataset={dataset!r}. "
                "Re-run Stage-1 with this split or use dataset='custom'."
            )

    X = sanitize_inputs_np(X)
    X = ensure_input_shapes(X, MODE, H)
    y_map = map_targets_for_training(y or {})
    
    _progress(0.20, f"Inference: dataset {dataset!r} loaded & shaped")
    # ----------------------------------------------------------
    # 4) Optional validation Dataset for calibrator
    # ----------------------------------------------------------
    ds_val = None
    if fit_calibrator:
        if "val_inputs_npz" in npz_dict and "val_targets_npz" in npz_dict:
            vx = dict(np.load(npz_dict["val_inputs_npz"]))
            vy = dict(np.load(npz_dict["val_targets_npz"]))
            vx = sanitize_inputs_np(vx)
            vx = ensure_input_shapes(vx, MODE, H)
            vy = map_targets_for_training(vy)
            ds_val = tf.data.Dataset.from_tensor_slices((vx, vy)).batch(
                batch_size
            )
        else:
            log("[Calibrator] No VAL NPZs found; cannot fit calibrator.")
    
    _progress(0.30, "Inference: validation dataset ready for calibrator")
    
    if should_stop():
        log("[Inference] stop_check=True before model load; aborting.")
        return {
            "run_dir": inf_dir,
            "manifest_path": manifest_file,
            "city": CITY,
            "model": MODEL_NAME,
            "dataset": dataset,
        }

    # ----------------------------------------------------------
    # 5) Load or rebuild model (and compile safely)
    # ----------------------------------------------------------
    log(f"[Model] Loading/rebuilding model from: {model_path}")

    model, _info = load_geoprior_for_inference(
        model_path=model_path,
        manifest=M,
        X_sample=X,
        out_s_dim=OUT_S_DIM,
        out_g_dim=OUT_G_DIM,
        mode=MODE,
        horizon=H,
        quantiles=QUANTILES if isinstance(QUANTILES, list) else None,
        city_name=CITY,
        include_metrics=True,
        verbose=1,
    )

    _progress(0.45, "Inference: model loaded and compiled")
     
    if should_stop():
        log("[Inference] stop_check=True after model load; aborting.")
        return {
            "run_dir": inf_dir,
            "manifest_path": manifest_file,
            "city": CITY,
            "model": MODEL_NAME,
            "dataset": dataset,
        }

    # ----------------------------------------------------------
    # 6) Optional: load / fit interval calibrator
    # ----------------------------------------------------------
    cal = None

    # 0) source-model directory calibrator if requested
    if use_source_calibrator and not calibrator_path:
        cand = os.path.join(
            os.path.dirname(os.path.abspath(model_path)),
            "interval_factors_80.npy",
        )
        if os.path.exists(cand):
            cal = IntervalCalibrator(target=cov_target)
            cal.factors_ = np.load(cand).astype(np.float32)
            log(f"[Calibrator] Loaded from source model dir: {cand}")

    # 1) explicit calibrator path
    if cal is None and calibrator_path and os.path.exists(calibrator_path):
        cal = IntervalCalibrator(target=cov_target)
        cal.factors_ = np.load(calibrator_path).astype(np.float32)
        log(f"[Calibrator] Loaded from explicit path: {calibrator_path}")

    # 2) fit on val if requested
    if cal is None and fit_calibrator and ds_val is not None:
        log("[Calibrator] Fitting on validation set...")
        cal = fit_interval_calibrator_on_val(
            model, ds_val, target=cov_target
        )
    _progress(0.55, "Inference: interval calibrator ready")
    
    if should_stop():
        log("[Inference] stop_check=True before prediction; aborting.")
        return {
            "run_dir": inf_dir,
            "manifest_path": manifest_file,
            "city": CITY,
            "model": MODEL_NAME,
            "dataset": dataset,
        }

    # ----------------------------------------------------------
    # 7) Predict
    # ----------------------------------------------------------
    _progress(0.60, "Inference: running model.predict(...)")

    log("[Inference] Running model.predict(...)")
    pred = model.predict(X, verbose=0)
    data_final = pred["data_final"]
    
    _progress(0.70, "Inference: raw predictions computed")
    
    # Split heads, handle quantiles vs point
    if data_final.ndim == 4:  # (B,H,Q,O_total)
        s_q = data_final[..., :OUT_S_DIM]
        g_q = data_final[..., OUT_S_DIM:]
        if cal is not None:
            s_q = apply_calibrator_to_subs(cal, s_q)
        predictions_for_formatter = {"subs_pred": s_q, "gwl_pred": g_q}
    elif data_final.ndim == 3:  # (B,H,O_total)
        s_p = data_final[..., :OUT_S_DIM]
        g_p = data_final[..., OUT_S_DIM:]
        predictions_for_formatter = {"subs_pred": s_p, "gwl_pred": g_p}
    else:
        raise RuntimeError(f"Unexpected data_final rank: {data_final.ndim}")

    # Prepare y_true for formatter (optional)
    y_true_for_format: Dict[str, Any] = {}
    if y_map:
        y_true_for_format = {
            "subsidence": y_map["subs_pred"],
            "gwl": y_map["gwl_pred"],
        }

    # Dataset for manual diagnostics (scaled space)
    ds_eval = None
    if y_map:
        ds_eval = tf.data.Dataset.from_tensor_slices((X, y_map)).batch(
            batch_size
        )

    # ----------------------------------------------------------
    # 8) Use format_and_forecast to write EVAL/FUTURE CSVs
    # ----------------------------------------------------------
    cols_cfg = cfg.get("cols", {})
    SUBS_COL = cols_cfg.get("subsidence", "subsidence")
    # GWL_COL = cols_cfg.get("gwl", "GWL")

    target_name = SUBS_COL
    target_key_pred = "subs_pred"

    base_name = f"{CITY}_{MODEL_NAME}_inference_{dataset}_H{H}"
    if cal is not None:
        base_name += "_calibrated"

    csv_eval = os.path.join(inf_dir, base_name + "_eval.csv")
    csv_future = os.path.join(inf_dir, base_name + "_future.csv")

    # Future grid: same logic as tuning (FSY .. FSY+H)
    future_grid = None
    if FSY is not None and H is not None:
        future_grid = np.arange(FSY, FSY + H, dtype=float)

    df_eval, df_future = format_and_forecast(
        y_pred=predictions_for_formatter,
        y_true=y_true_for_format or None,
        coords=X.get("coords", None),
        quantiles=QUANTILES if QUANTILES else None,
        target_name=target_name,
        target_key_pred=target_key_pred,
        component_index=0,
        scaler_info=scaler_info,
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
        dataset_name=dataset,
        csv_eval_path=csv_eval,
        csv_future_path=csv_future,
        time_as_datetime=False,
        time_format=None,
        verbose=1,
        # In inference mode we don't compute extra eval metrics here
        eval_metrics=False,
        value_mode="rate",
        logger=log,
    )

    if df_eval is not None and not df_eval.empty:
        log(f"[Inference] Saved calibrated EVAL forecast CSV -> {csv_eval}")
    else:
        log("[Inference] Eval forecast DF is empty (df_eval).")

    if df_future is not None and not df_future.empty:
        log(f"[Inference] Saved calibrated FUTURE forecast CSV -> {csv_future}")
    else:
        log("[Inference] Future forecast DF is empty (df_future).")

    # Choose a "main" CSV path for convenience (prefer FUTURE if available)
    if df_future is not None and not df_future.empty:
        main_csv = csv_future
    else:
        main_csv = csv_eval
    
    _progress(0.75, "Inference: forecast CSVs written")
    # ----------------------------------------------------------
    # 9) Manual diagnostics (coverage/sharpness + physical metrics)
    #     (No Keras model.evaluate(), no physics evaluate)
    # ----------------------------------------------------------
    eval_json: Dict[str, Any] = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "quantiles": QUANTILES if data_final.ndim == 4 else None,
        "coverage80": None,
        "sharpness80": None,
    }

    point_phys = None
    per_h_mae_phys = None
    per_h_r2_phys = None

    if y_map and data_final.ndim == 4 and ds_eval is not None:
        y_true_list, s_q_list = [], []
        
        # Optional: estimate number of batches for GUI progress
        num_batches = None
        try:
            num_batches = int(
                tf.data.experimental.cardinality(ds_eval).numpy()
            )
            if num_batches <= 0:
                num_batches = None
        except Exception:
            num_batches = None
            
        for i, (xb, yb) in enumerate(
            with_progress(ds_eval, desc="Inference interval diagnostics"),
            start=1,
        ):
            if should_stop():
                log(
                    "[Inference] stop_check=True inside interval "
                    "diagnostics loop."
                )
                break
            
            # Inner progress slice: 0.80 → 0.90
            if num_batches is not None:
                inner = max(0.0, min(1.0, i / float(num_batches)))
                frac = 0.80 + 0.10 * inner
                _progress(frac, f"Inference: diagnostics {i}/{num_batches}")
    
            out_b = model(xb, training=False)
            s_q_b, _ = model.split_data_predictions(out_b["data_final"])
            # Use same calibration as for predictions CSV
            if cal is not None:
                s_q_b = apply_calibrator_to_subs(cal, s_q_b)
            y_true_list.append(yb["subs_pred"])   # (B,H,1)
            s_q_list.append(s_q_b)                # (B,H,Q,1)

        if y_true_list and s_q_list:
            y_true = tf.concat(y_true_list, axis=0)    # (N,H,1)
            s_q_all = tf.concat(s_q_list, axis=0)      # (N,H,Q,1)

            # --- scaled coverage & sharpness ---
            eval_json["coverage80"] = float(
                coverage80_fn(y_true, s_q_all).numpy()
            )
            eval_json["sharpness80"] = float(
                sharpness80_fn(y_true, s_q_all).numpy()
            )

            # --- physical coverage & sharpness ---
            y_true_phys_np = inverse_scale_target(
                y_true,
                scaler_info=scaler_info,
                target_name=SUBS_COL,
            )
            s_q_phys_np = inverse_scale_target(
                s_q_all,
                scaler_info=scaler_info,
                target_name=SUBS_COL,
            )
            y_true_phys_tf = tf.convert_to_tensor(
                y_true_phys_np, dtype=tf.float32
            )
            s_q_phys_tf = tf.convert_to_tensor(
                s_q_phys_np, dtype=tf.float32
            )

            eval_json["coverage80_phys"] = float(
                coverage80_fn(y_true_phys_tf, s_q_phys_tf).numpy()
            )
            eval_json["sharpness80_phys"] = float(
                sharpness80_fn(y_true_phys_tf, s_q_phys_tf).numpy()
            )

            # --- physical point metrics (MAE/MSE/R²) ---
            quantiles_arr = np.asarray(QUANTILES, dtype=float)
            med_idx = int(np.argmin(np.abs(quantiles_arr - 0.5)))
            s_med_phys_np = s_q_phys_np[..., med_idx, :]  # (N,H,1)

            point_phys = point_metrics(
                y_true_phys_np,
                s_med_phys_np,
            )
            per_h_mae_phys, per_h_r2_phys = per_horizon_metrics(
                y_true_phys_np,
                s_med_phys_np,
            )

            eval_json["point_metrics_phys"] = {
                "mae": point_phys.get("mae"),
                "mse": point_phys.get("mse"),
                "r2":  point_phys.get("r2"),
            }
            eval_json["per_horizon_phys"] = {
                "mae": per_h_mae_phys,
                "r2":  per_h_r2_phys,
            }
        
        _progress(0.90, "Inference: diagnostics complete")

    summary_path = os.path.join(inf_dir, "inference_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2)
    log(f"[Inference] Saved inference summary JSON -> {summary_path}")

    _progress(0.95, "Inference: summary JSON written")
    
    # ----------------------------------------------------------
    # 10) Plots (optional, no evaluate/physics)
    # ----------------------------------------------------------
    has_any_df = (
        (df_eval is not None and not df_eval.empty)
        or (df_future is not None and not df_future.empty)
    )

    if make_plots and has_any_df and not should_stop():
        log("\n[Inference] Plotting forecast views...")
        try:
            # For eval: last year of training (e.g. 2022)
            eval_years = [TRAIN_END_YEAR] if TRAIN_END_YEAR is not None else None
            # For future: use the same grid passed to format_and_forecast
            future_years = future_grid

            plot_eval_future(
                df_eval=df_eval,       # can be empty; function should handle it
                df_future=df_future,
                target_name=SUBS_COL,
                quantiles=QUANTILES if isinstance(QUANTILES, list) else None,
                spatial_cols=("coord_x", "coord_y"),
                time_col="coord_t",
                eval_years=eval_years,
                future_years=future_years,
                eval_view_quantiles=[0.5],     # compare [actual] vs [q50]
                future_view_quantiles=QUANTILES,
                spatial_mode="hexbin",
                hexbin_gridsize=40,
                savefig_prefix=os.path.join(
                    inf_dir,
                    f"{CITY}_subsidence_view",
                ),
                save_fmts=[".png", ".pdf"],
                show=False,
                verbose=1,
                cumulative=True,
                _logger=log,
            )
            log(f"[Inference] Saved forecast figures in: {inf_dir}")
            
            # notify the GUI that the PNGs were created
            _notify_gui_forecast_views(inf_dir, CITY)
            
        except Exception as e:
            log(f"[Warn] plot_eval_future failed: {e}")
    
        _progress(1.0, "Inference: complete (plots saved)")
    
    if not make_plots or not has_any_df or should_stop():
        _progress(1.0, "Inference: complete")

    # ----------------------------------------------------------
    # 11) Final return payload
    # ----------------------------------------------------------
    return {
        "run_dir": inf_dir,
        "manifest_path": manifest_file,
        "city": CITY,
        "model": MODEL_NAME,
        "dataset": dataset,
        "model_path": model_path,
        "csv_path": main_csv,
        "csv_eval_path": csv_eval,
        "csv_future_path": csv_future,
        "inference_summary_json": summary_path,
        "coverage80": eval_json.get("coverage80"),
        "sharpness80": eval_json.get("sharpness80"),
        "coverage80_phys": eval_json.get("coverage80_phys"),
        "sharpness80_phys": eval_json.get("sharpness80_phys"),
        "point_metrics_phys": point_phys,
        "per_horizon_phys": {
            "mae": per_h_mae_phys,
            "r2": per_h_r2_phys,
        },
    }
