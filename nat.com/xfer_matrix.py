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

# ---- fusionlab imports ----
from fusionlab.registry.utils import _find_stage1_manifest, reproject_dynamic_scale
from fusionlab.utils.generic_utils import ensure_directory_exists

from fusionlab.nn.pinn.op import extract_physical_parameters
from fusionlab.nn.pinn.models import GeoPriorSubsNet
from fusionlab.params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
from fusionlab.nn.losses import make_weighted_pinball
from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn
from fusionlab.nn.calibration import (
    IntervalCalibrator, fit_interval_calibrator_on_val,
    apply_calibrator_to_subs,
)

from fusionlab.nn.pinn.utils import format_pinn_predictions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-city transfer evaluation matrix (A→B and B→A)."
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
    T     = cfg_t["TIME_STEPS"]
    H     = cfg_t["FORECAST_HORIZON_YEARS"]
    Q     = quantiles_override or cfg_t.get("QUANTILES", [0.1, 0.5, 0.9])

    # Build input shapes
    X_tgt = _ensure_shapes(X_tgt, MODE, H)
    y_map = _map_targets(y_tgt)

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
    if not model_path:
        raise SystemExit(f"No .keras found under {M_src['paths']['run_dir']}")
    custom_objects = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
    }
    with custom_object_scope(custom_objects):
        model = load_model(model_path, compile=False)

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
            save_dir=model_path, 
            model_name="geoprior",
        )
    except: 
        pass 
    
    # Predict
    pred = model.predict(X_tgt, verbose=0)
    data_final = pred["data_final"]

    OUT_S_DIM = M_tgt["artifacts"]["sequences"]["dims"]["output_subsidence_dim"]
    OUT_G_DIM = M_tgt["artifacts"]["sequences"]["dims"]["output_gwl_dim"]

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
        
    # XXX TODO: RECHECK --- New section: Per-horizon MAE and R² ---
    per_horizon_mae = {}
    per_horizon_r2 = {}
    # Loop over each forecast horizon (each time step) and calculate MAE and R²
    
    for h in range(H):
        y_true_horizon = y_map["subs_pred"][:, h, :]  # True subsidence for this horizon
        y_pred_horizon = predictions["subs_pred"][:, h, :]  # Predicted subsidence for this horizon

        # Compute MAE and R² for subsidence
        mae = mean_absolute_error(y_true_horizon, y_pred_horizon)
        r2 = r2_score(y_true_horizon, y_pred_horizon)
 
        per_horizon_mae[f"H{h+1}"] = mae
        per_horizon_r2[f"H{h+1}"] = r2  
    # -    

    # Optional scaled-space evaluation & physics
    eval_scaled = None
    physics_diag = None
    if y_map:
        ds = tf.data.Dataset.from_tensor_slices((X_tgt, y_map)).batch(batch_size)
        try:
            eval_scaled = model.evaluate(ds, return_dict=True, verbose=0)
        except Exception:
            eval_scaled = None
    try:
        p = model.evaluate_physics(X_tgt)
        physics_diag = {k: float(v.numpy()) for k, v in p.items()}
    except Exception:
        physics_diag = None
    
    try:
        model.export_physics_payload(
            X_tgt,
            max_batches=None,
            save_path=os.path.join(
                model_path, f"{M_tgt.get('city')}_xfer_physics_payload.npz"
                ),
            format="npz",
            overwrite=True,
            metadata={"city": M_tgt.get("city"), "split": "val"},
    )
    except: 
        pass 
    
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

    cols_cfg = M_tgt["config"]["cols"]
    SUBS_COL = cols_cfg.get("subsidence", "subsidence")
    GWL_COL  = cols_cfg.get("gwl", "GWL")
    target_mapping = {"subs_pred": SUBS_COL, "gwl_pred": GWL_COL}
    output_dims    = {"subs_pred": OUT_S_DIM, "gwl_pred": OUT_G_DIM}

    # We won't save large CSVs here; just compute coverage/sharpness if possible
    coverage80 = sharpness80 = None
    if y_map and data_final.ndim == 4:
        ds_calc = tf.data.Dataset.from_tensor_slices((X_tgt, y_map)).batch(batch_size)
        y_true_list, s_q_list = [], []
        for xb, yb in ds_calc:
            out = model(xb, training=False)
            s_q_b, _ = model.split_data_predictions(out["data_final"])
            y_true_list.append(yb["subs_pred"])
            s_q_list.append(s_q_b)
        y_true = tf.concat(y_true_list, axis=0)
        s_q     = tf.concat(s_q_list,    axis=0)
        coverage80 = float(coverage80_fn(y_true, s_q).numpy())
        sharpness80 = float(sharpness80_fn(y_true, s_q).numpy())
        
    # Format predictions
    xfer_df_path =os.path.join(
        model_path, f"{M_tgt.get('city')}_xfer_predictions_df.npz"
        )
    forecast_df = format_pinn_predictions(
        predictions=predictions,
        y_true_dict=y_map,
        target_mapping=target_mapping,
        scaler_info=scaler_info,
        quantiles=Q,
        forecast_horizon=H,
        output_dims=output_dims,
        include_coords=True,
        include_gwl=False,  # change to True if you want to include GWL columns
        model_inputs=X_tgt,
        evaluate_coverage=True if Q else False,
        savefile=xfer_df_path,
        coord_scaler=coord_scaler,
        verbose=1,
    )
    if forecast_df is not None and not forecast_df.empty:
        print(f"Saved calibrated forecast CSV -> {xfer_df_path}")
    else:
        print("[Warn] Empty forecast DF.")


    return {
        "model_path": model_path,
        "split": split,
        "calibration": calib_mode,
        "quantiles": Q if data_final.ndim == 4 else None,
        "keras_eval_scaled": eval_scaled,
        "physics": physics_diag,
        "coverage80": coverage80,
        "sharpness80": sharpness80,
         # XXX TODO: RECHECK
        # Add more fields as needed (e.g., per-horizon MAE, r^2 from formatted DF)
        "per_horizon_mae": per_horizon_mae,  # New metrics
        "per_horizon_r2": per_horizon_r2,  # New metrics
       
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
    cols = [
        "direction","source_city","target_city","split","calibration",
        "coverage80","sharpness80",
        "physics.epsilon_prior","physics.epsilon_cons",
        "keras_eval_scaled.loss","keras_eval_scaled.subs_pred_mae",
        "keras_eval_scaled.gwl_pred_mae",
    ]
    # XXX TO RECHECK 
    cols.extend(
        [f"per_horizon_mae.H{i+1}" for i in range(args.forecast_horizon)])  # Add per-horizon MAE columns
    cols.extend(
        [f"per_horizon_r2.H{i+1}" for i in range(args.forecast_horizon)])  # Add per-horizon R² columns
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            phys = r.get("physics") or {}
            ev   = r.get("keras_eval_scaled") or {}
            row =[
                r.get("direction"),
                r.get("source_city"),
                r.get("target_city"),
                r.get("split"),
                r.get("calibration"),
                r.get("coverage80"),
                r.get("sharpness80"),
                phys.get("epsilon_prior"),
                phys.get("epsilon_cons"),
                ev.get("loss"),
                ev.get("subs_pred_mae"),
                ev.get("gwl_pred_mae"),
            ]
        # XXX TODO: RECHECK  Add per-horizon MAE and R² 
        for i in range(args.forecast_horizon):
            row.append(r.get("per_horizon_mae", {}).get(f"H{i+1}", "N/A"))
            row.append(r.get("per_horizon_r2", {}).get(f"H{i+1}", "N/A"))
        
        w.writerow(row)
        
    print(f"Saved transfer CSV -> {csv_path}")

if __name__ == "__main__":
    main()
