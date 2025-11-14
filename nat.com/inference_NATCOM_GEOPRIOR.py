# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
inference_NATCOM_GEOPRIOR.py
Stage-3: Deterministic/Probabilistic inference for GeoPriorSubsNet.

Usage (tuned model):
  python nat.com/inference_NATCOM_GEOPRIOR.py ^
    --stage1-dir results/nansha_GeoPriorSubsNet_stage1 ^
    --model-path results/nansha_GeoPriorSubsNet_stage1/tuning/run_YYYYMMDD-HHMMSS/nansha_GeoPrior_best.keras ^
    --dataset test ^
    --calibrator results/nansha_GeoPriorSubsNet_stage1/train_YYYYMMDD-HHMMSS/interval_factors_80.npy

Usage (trained model, re-fit calibrator on val):
  python nat.com/inference_NATCOM_GEOPRIOR.py ^
    --stage1-dir results/nansha_GeoPriorSubsNet_stage1 ^
    --model-path results/nansha_GeoPriorSubsNet_stage1/train_YYYYMMDD-HHMMSS/nansha_GeoPriorSubsNet_H3.keras ^
    --dataset test --fit-calibrator

You can also do --dataset custom --inputs-npz <path> [--targets-npz <path>]

cross -validation 
-------------------
1) A --> B ( Nansha-> Zhongshan ), zero-shot 

python nat.com/inference_NATCOM_GEOPRIOR.py \
  --stage1-dir results/zhongshan_GeoPriorSubsNet_stage1 \
  --model-path results/nansha_GeoPriorSubsNet_stage1/train_*/nansha_GeoPriorSubsNet_H3.keras \
  --dataset test \
  --eval-losses --eval-physics --no-figs

2) A--> B with source calibrator

python nat.com/inference_NATCOM_GEOPRIOR.py \
  --stage1-dir results/zhongshan_GeoPriorSubsNet_stage1 \
  --model-path results/nansha_GeoPriorSubsNet_stage1/train_*/nansha_GeoPriorSubsNet_H3.keras \
  --dataset test \
  --use-source-calibrator \
  --eval-losses --eval-physics --no-figs

3) A-->B with target-val calibration (report separately)

python nat.com/inference_NATCOM_GEOPRIOR.py \
  --stage1-dir results/zhongshan_GeoPriorSubsNet_stage1 \
  --model-path results/nansha_GeoPriorSubsNet_stage1/train_*/nansha_GeoPriorSubsNet_H3.keras \
  --dataset test \
  --fit-calibrator \
  --eval-losses --eval-physics --no-figs


Repeat B--> A by swapping stage1-dir and model-path.

All runs produce:

* formatted prediction CSV (inverse-scaled to mm),
* inference_summary.json (coverage/sharpness),
* transfer_eval.json (losses + physics).

"""

from __future__ import annotations
import os, json, argparse, datetime as dt, joblib, numpy as np, warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# --- quiet logs ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# --- fusionlab imports ---
from fusionlab.utils.generic_utils import ensure_directory_exists
from fusionlab.utils.generic_utils import default_results_dir, getenv_stripped
from fusionlab.registry.utils import _find_stage1_manifest  

from fusionlab.nn.pinn.models import GeoPriorSubsNet
from fusionlab.params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
from fusionlab.nn.losses import make_weighted_pinball
from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn #, _to_py
from fusionlab.nn.calibration import (
    IntervalCalibrator,
    fit_interval_calibrator_on_val,
    apply_calibrator_to_subs,
)
from fusionlab.nn.pinn.utils import format_pinn_predictions
from fusionlab.plot.forecast import plot_forecasts, forecast_view

# ------------------ CLI ------------------
def parse_args():
    p = argparse.ArgumentParser(description="GeoPriorSubsNet inference")
    p.add_argument("--stage1-dir", default=None,
                   help="Folder containing Stage-1 manifest.json (optional)")
    p.add_argument("--manifest", default=None,
                   help="Direct path to Stage-1 manifest.json (optional)")
    p.add_argument("--model-path", required=True,
                   help="Path to .keras model (trained or tuned)")
    p.add_argument("--dataset", default="test",
                   choices=["test", "val", "train", "custom"],
                   help="Which split to run inference on")
    p.add_argument("--eval-losses", action="store_true",
               help="Compute MAE/MSE losses on chosen split.")
    p.add_argument("--eval-physics", action="store_true",
                   help="Compute epsilon_prior/epsilon_cons on chosen split.")
    p.add_argument("--use-source-calibrator", action="store_true",
                   help="Load calibrator .npy from the model directory if present.")

    p.add_argument("--inputs-npz", default=None,
                   help="Custom inputs npz (for --dataset custom)")
    p.add_argument("--targets-npz", default=None,
                   help="Optional custom targets npz (enables metrics)")
    p.add_argument("--calibrator", default=None,
                   help="Path to interval_factors_80.npy (optional)")
    p.add_argument("--fit-calibrator", action="store_true",
                   help="Fit an IntervalCalibrator on val if available")
    p.add_argument("--no-figs", action="store_true", help="Skip plotting")
    p.add_argument("--include-gwl", action="store_true",
                   help="Include GWL columns in formatted CSV")
    p.add_argument("--cov-target", type=float, default=0.80,
                   help="Target coverage for interval calibrator")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()



# ------------------ helpers ------------------
def _resolve_manifest(args) -> dict:
    # 1) explicit --manifest
    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2) explicit --stage1-dir
    if args.stage1_dir:
        mpath = os.path.join(args.stage1_dir, "manifest.json")
        if not os.path.exists(mpath):
            raise SystemExit(f"manifest.json not found in {args.stage1_dir}")
        with open(mpath, "r", encoding="utf-8") as f:
            return json.load(f)

    # 3) env / auto-discovery (same policy as Stage-2/Stage-3)

    RESULTS_DIR = default_results_dir()  # auto-resolve
    CITY_HINT   = getenv_stripped("CITY")  # -> None if unset/empty
    MODEL_HINT  = getenv_stripped("MODEL_NAME_OVERRIDE", default="GeoPriorSubsNet")
    MANUAL      = getenv_stripped("STAGE1_MANIFEST")  # exact path if provided
    
    MANIFEST_PATH = _find_stage1_manifest(
        manual=MANUAL,
        base_dir=RESULTS_DIR,
        city_hint=CITY_HINT,
        model_hint=MODEL_HINT,
        prefer="timestamp",
        required_keys=("model", "stage"),
        verbose=1,
    )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
def _sanitize_inputs_np(X: dict) -> dict:
    X = dict(X)
    for k, v in X.items():
        if v is None:
            continue
        v = np.asarray(v)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if v.ndim > 0 and np.isfinite(v).any():
            p99 = np.percentile(v, 99)
            if p99 > 0:
                v = np.clip(v, -10*p99, 10*p99)
        X[k] = v
    if "H_field" in X:
        X["H_field"] = np.maximum(X["H_field"], 1e-3).astype(np.float32)
    return X

def _ensure_input_shapes(x: dict, mode: str, horizon: int) -> dict:
    out = dict(x)
    N  = out["dynamic_features"].shape[0]
    T  = out["dynamic_features"].shape[1]
    if out.get("static_features") is None:
        out["static_features"] = np.zeros((N, 0), dtype=np.float32)
    if out.get("future_features") is None:
        t_future = T if mode == "tft_like" else horizon
        out["future_features"] = np.zeros((N, t_future, 0), dtype=np.float32)
    return out


# def _ensure_input_shapes(x: dict, mode: str, horizon: int) -> dict:
#     """Add zero-width placeholders if static/future missing."""
#     out = dict(x)
#     N = out["dynamic_features"].shape[0]
#     if out.get("static_features") is None:
#         out["static_features"] = np.zeros((N, 0), dtype=np.float32)
#     if out.get("future_features") is None:
#         t_future = out["dynamic_features"].shape[1] if mode == "tft_like" else horizon
#         out["future_features"] = np.zeros((N, t_future, 0), dtype=np.float32)
#     return out

def _map_targets(y_dict: dict) -> dict:
    # Accept either ('subsidence','gwl') or ('subs_pred','gwl_pred')
    if "subsidence" in y_dict and "gwl" in y_dict:
        return {"subs_pred": y_dict["subsidence"], "gwl_pred": y_dict["gwl"]}
    if "subs_pred" in y_dict and "gwl_pred" in y_dict:
        return y_dict
    # Allow missing targets for pure inference
    return {}

 
def _pick_npz_for_dataset(M: dict, name: str) -> tuple[dict | None, dict | None]:
    npzs = M["artifacts"]["numpy"]
    if name == "train":
        x = dict(np.load(npzs["train_inputs_npz"]))
        y = dict(np.load(npzs["train_targets_npz"]))
        return x, y
    if name == "val":
        x = dict(np.load(npzs["val_inputs_npz"]))
        y = dict(np.load(npzs["val_targets_npz"]))
        return x, y
    if name == "test":
        tin = npzs.get("test_inputs_npz")
        tt  = npzs.get("test_targets_npz")
        if not tin:
            return None, None
        x = dict(np.load(tin))
        y = dict(np.load(tt)) if tt else None
        return x, y
    raise ValueError("Use 'custom' for custom NPZs")

def _load_calibrator(args, model, ds_val, target_cov):
    # 1) explicit .npy
    if args.calibrator and os.path.exists(args.calibrator):
        cal = IntervalCalibrator(target=target_cov)
        cal.factors_ = np.load(args.calibrator).astype(np.float32)
        return cal
    # 2) fit on val set if requested and available
    if args.fit_calibrator and ds_val is not None:
        return fit_interval_calibrator_on_val(model, ds_val, target=target_cov)
    return None

# ------------------ main ------------------
def main():
    args = parse_args()
    M = _resolve_manifest(args)
    print(f"[Manifest] Loaded city={M.get('city')} model={M.get('model')}")


    cfg = M["config"]
    CITY = M.get("city", "nansha")
    MODEL_NAME = M.get("model", "GeoPriorSubsNet")

    MODE = cfg["MODE"]
    T = cfg["TIME_STEPS"]
    H = cfg["FORECAST_HORIZON_YEARS"]
    QUANTILES = cfg.get("QUANTILES", [0.1, 0.5, 0.9])

    OUT_S_DIM = M["artifacts"]["sequences"]["dims"]["output_subsidence_dim"]
    OUT_G_DIM = M["artifacts"]["sequences"]["dims"]["output_gwl_dim"]

    # Encoders (optional but recommended for formatting back to physical units)
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
    # Optionally attach scaler objects for inverse-transform if only indices were stored
    if isinstance(scaler_info, dict):
        for k, v in scaler_info.items():
            if isinstance(v, dict) and "scaler_path" in v and "scaler" not in v:
                p = v["scaler_path"]
                if p and os.path.exists(p):
                    try:
                        v["scaler"] = joblib.load(p)
                    except Exception:
                        pass


    # Output dir
    inf_dir = os.path.join(M["paths"]["run_dir"], "inference", dt.datetime.now(
        ).strftime("%Y%m%d-%H%M%S"))
    ensure_directory_exists(inf_dir)

    # Choose dataset
    if args.dataset == "custom":
        if not args.inputs_npz:
            raise SystemExit("--inputs-npz required for --dataset custom")
        X = dict(np.load(args.inputs_npz))
        y = dict(np.load(args.targets_npz)) if args.targets_npz else None
    else:
        X, y = _pick_npz_for_dataset(M, args.dataset)
        if X is None:
            raise SystemExit(
                f"No NPZs found for dataset='{args.dataset}'."
                " Re-run Stage-1 with a test split or use --dataset custom.")

    X = _sanitize_inputs_np(X)
    X = _ensure_input_shapes(X, MODE, H)
    
    if y:  # y can be None (pure inference)
        # no need to sanitize y (targets) typically, but it's fine if you want
        pass

    y_map = _map_targets(y or {})

    # Datasets for optional calibration/metrics
    batch_size = args.batch_size
    ds_val = None
    npz = M["artifacts"]["numpy"]
    if args.fit_calibrator and "val_inputs_npz" in npz and "val_targets_npz" in npz:
        vx = dict(np.load(npz["val_inputs_npz"]))
        vy = dict(np.load(npz["val_targets_npz"]))
        vx = _sanitize_inputs_np(vx)
        vx = _ensure_input_shapes(vx, MODE, H)
        vy = _map_targets(vy)
        ds_val = tf.data.Dataset.from_tensor_slices((vx, vy)).batch(batch_size)


    # Load model (compile not required for inference)
    custom_objects = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
    }
    with custom_object_scope(custom_objects):
        model = load_model(args.model_path, compile=False)

    # Optional: fit/load calibrator
    cal = _load_calibrator(args, model, ds_val, target_cov=args.cov_target)

    # Predict
    pred = model.predict(X, verbose=0)  # {'data_final': ...}
    data_final = pred["data_final"]

    # Split heads, handle quantiles vs point
    if data_final.ndim == 4:  # (B,H,Q,O_total)
        s_q = data_final[..., :OUT_S_DIM]  # (B,H,Q,Os)
        g_q = data_final[..., OUT_S_DIM:]  # (B,H,Q,Og)
        if cal is not None:
            s_q = apply_calibrator_to_subs(cal, s_q)
        predictions_for_formatter = {"subs_pred": s_q, "gwl_pred": g_q}
    elif data_final.ndim == 3:  # (B,H,O_total) point
        s_p = data_final[..., :OUT_S_DIM]
        g_p = data_final[..., OUT_S_DIM:]
        predictions_for_formatter = {"subs_pred": s_p, "gwl_pred": g_p}
    else:
        raise RuntimeError(f"Unexpected data_final rank: {data_final.ndim}")

    # Prepare y_true for formatter (optional)
    y_true_for_format = {}
    if y_map:
        y_true_for_format = {"subsidence": y_map["subs_pred"], "gwl": y_map["gwl_pred"]}

    # Map column names
    cols_cfg = cfg.get("cols", {})
    SUBS_COL = cols_cfg.get("subsidence", "subsidence")
    GWL_COL  = cols_cfg.get("gwl", "GWL")
    target_mapping = {"subs_pred": SUBS_COL, "gwl_pred": GWL_COL}
    output_dims = {"subs_pred": OUT_S_DIM, "gwl_pred": OUT_G_DIM}

    # Save CSV
    csv_name = (f"{CITY}_{MODEL_NAME}_inference_{args.dataset}_H{H}"
                f"{'_cal' if cal is not None else ''}.csv")
    csv_path = os.path.join(inf_dir, csv_name)

    df = format_pinn_predictions(
        predictions=predictions_for_formatter,
        y_true_dict=y_true_for_format or None,
        target_mapping=target_mapping,
        scaler_info=scaler_info,
        quantiles=QUANTILES if data_final.ndim == 4 else None,
        forecast_horizon=H,
        output_dims=output_dims,
        include_coords=True,
        include_gwl=args.include_gwl,
        model_inputs=X,
        evaluate_coverage=bool(y_map) and data_final.ndim == 4,
        savefile=csv_path,
        coord_scaler=coord_scaler,
        verbose=1,
    )
    print(f"Saved inference CSV -> {csv_path}")

    # Optional quick diagnostics
    eval_json = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "quantiles": QUANTILES if data_final.ndim == 4 else None,
        "coverage80": None,
        "sharpness80": None,
    }
    if y_map and data_final.ndim == 4:
        # Compute interval metrics on-the-fly
        # Build a small dataset for metrics
        ds = tf.data.Dataset.from_tensor_slices((X, y_map)).batch(batch_size)
        y_true_list, s_q_list = [], []
        for xb, yb in ds:
            out = model(xb, training=False)
            s_q_b, _ = model.split_data_predictions(out["data_final"])
            y_true_list.append(yb["subs_pred"])
            s_q_list.append(s_q_b)
        y_true = tf.concat(y_true_list, axis=0)
        s_q    = tf.concat(s_q_list,    axis=0)
        eval_json["coverage80"]  = float(coverage80_fn(y_true, s_q).numpy())
        eval_json["sharpness80"] = float(sharpness80_fn(y_true, s_q).numpy())

    with open(os.path.join(inf_dir, "inference_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2)
        
    print(f"Saved inference summary JSON -> {os.path.join(inf_dir, 'inference_summary.json')}")
    
    # Optional: metrics in scaled space (Keras) + physics diagnostics
    if args.eval_losses and y_map:
        ds_eval = tf.data.Dataset.from_tensor_slices(
            (X, y_map)).batch(args.batch_size)
        eval_results = model.evaluate(
            ds_eval, return_dict=True, verbose=0)
    else:
        eval_results = None
    
    physics_diag = None
    if args.eval_physics:
        try:
            phys_raw = model.evaluate_physics(
                ds_eval,
                max_batches=10,       # e.g. first 10 batches only
                return_maps=False,    # we only need scalar epsilons here
            )
            # Keep only scalar entries when converting to Python floats
            physics_diag = {
                k: float(v.numpy())
                for k, v in phys_raw.items()
                if getattr(v, "shape", ()) == ()    # shape () -> scalar
            }
            print("Physics diagnostics (approx, first 10 batches):", physics_diag)
            
            # physics_diag = {k: float(
            #     v.numpy()) for k, v in model.evaluate_physics(X,   ).items()}
        except Exception as e :
            print(f"[Warn] Physics eval failed: {e}")
            pass
    
    # Persist a single JSON with everything (also keep your existing summary)
    with open(os.path.join(inf_dir, "transfer_eval.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "coverage80": eval_json.get("coverage80"),
            "sharpness80": eval_json.get("sharpness80"),
            "keras_eval": eval_results,         # scaled metrics
            "physics": physics_diag,            # epsilon_prior/epsilon_cons
            "csv_path": csv_path,
        }, f, indent=2)
    

    # Plots
    if (not args.no_figs) and (df is not None) and (len(df) > 0):
        try:
            horizon_steps = [1, H] if H > 1 else [1]
            plot_forecasts(
                forecast_df=df,
                target_name=SUBS_COL,
                quantiles=QUANTILES if data_final.ndim == 4 else None,
                output_dim=OUT_S_DIM,
                kind="spatial",
                horizon_steps=horizon_steps,
                spatial_cols=("coord_x", "coord_y"),
                sample_ids="first_n",
                num_samples=min(3, batch_size),
                max_cols=2,
                figsize=(7, 5.5),
                cbar="uniform",
                verbose=1,
            )
            forecast_view(
                df,
                spatial_cols=("coord_x", "coord_y"),
                time_col="coord_t",
                value_prefixes=[SUBS_COL],
                verbose=1,
                view_quantiles=[0.5] if data_final.ndim == 4 else None,
                savefig=os.path.join(inf_dir, f"{CITY}_forecast_comparison_plot_"),
                save_fmts=[".png", ".pdf"],
            )
            print(f"Saved forecast figures in: {inf_dir}")
        except Exception as e:
            print(f"[Warn] plotting failed: {e}")
    
    


if __name__ == "__main__":
    main()

