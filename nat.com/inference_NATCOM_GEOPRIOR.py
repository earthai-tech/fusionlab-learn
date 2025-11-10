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
    p.add_argument("--stage1-dir", required=True,
                   help="Folder containing Stage-1 manifest.json")
    p.add_argument("--model-path", required=True,
                   help="Path to .keras model (trained or tuned)")
    p.add_argument("--dataset", default="test",
                   choices=["test", "val", "train", "custom"],
                   help="Which split to run inference on")
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
def _load_manifest(stage1_dir: str) -> dict:
    mpath = os.path.join(stage1_dir, "manifest.json")
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_input_shapes(x: dict, mode: str, horizon: int) -> dict:
    """Add zero-width placeholders if static/future missing."""
    out = dict(x)
    N = out["dynamic_features"].shape[0]
    if out.get("static_features") is None:
        out["static_features"] = np.zeros((N, 0), dtype=np.float32)
    if out.get("future_features") is None:
        t_future = out["dynamic_features"].shape[1] if mode == "tft_like" else horizon
        out["future_features"] = np.zeros((N, t_future, 0), dtype=np.float32)
    return out

def _map_targets(y_dict: dict) -> dict:
    # Accept either ('subsidence','gwl') or ('subs_pred','gwl_pred')
    if "subsidence" in y_dict and "gwl" in y_dict:
        return {"subs_pred": y_dict["subsidence"], "gwl_pred": y_dict["gwl"]}
    if "subs_pred" in y_dict and "gwl_pred" in y_dict:
        return y_dict
    # Allow missing targets for pure inference
    return {}

def _pick_npz_for_dataset(
        M: dict, name: str) -> tuple[dict | None, dict | None]:
    npzs = M["artifacts"]["numpy"]
    if name == "train":
        return dict(np.load(npzs["train_inputs_npz"])),
    dict(np.load(npzs["train_targets_npz"]))
    if name == "val":
        return dict(np.load(npzs["val_inputs_npz"])),
    dict(np.load(npzs["val_targets_npz"]))
    if name == "test":
        tin, tt = npzs.get("test_inputs_npz"), 
        npzs.get("test_targets_npz")
        if tin is None:
            return None, None
        x = dict(np.load(tin))
        y = dict(np.load(tt)) if tt else None
        return x, y
    raise ValueError("Use 'custom' codepath for custom NPZs")

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
    M = _load_manifest(args.stage1_dir)

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

    X = _ensure_input_shapes(X, MODE, H)
    y_map = _map_targets(y or {})

    # Datasets for optional calibration/metrics
    batch_size = args.batch_size
    ds_val = None
    if args.fit_calibrator and "val_inputs_npz" in M["artifacts"]["numpy"]:
        vx = dict(np.load(M["artifacts"]["numpy"]["val_inputs_npz"]))
        vy = dict(np.load(M["artifacts"]["numpy"]["val_targets_npz"]))
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

