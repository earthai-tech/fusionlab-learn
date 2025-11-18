# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Forecast future years with a trained/tuned GeoPriorSubsNet.

Outputs
-------
- <CITY>_GeoPriorSubsNet_forecast_Future_H{H}_calibrated.csv
- Optional per-year slices: forecast_year_<YYYY>.csv
- Optional cumulative summary: forecast_cumulative.csv

Notes
-----
- Requires Stage-1 manifest + NPZs and a saved model (.keras).
- With FORECAST_HORIZON_YEARS = H, forecasts cover
  [FORECAST_START_YEAR, ..., FORECAST_START_YEAR + H - 1].

python nat.com/forecast/make_future_forecasts.py \
  --results-dir results \
  --city nansha \
  --prefer tuned \
  --seed-split val \
  --apply-calibration auto \
  --export-years 2022 2025 2026 \
  --export-cumulative

"""

from __future__ import annotations
import os, json, glob, argparse
from typing import Tuple
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# ---- fusionlab imports ----
from fusionlab.registry.utils import _find_stage1_manifest
from fusionlab.utils.generic_utils import ensure_directory_exists
from fusionlab.nn.pinn.utils import format_pinn_predictions
from fusionlab.nn.calibration import (
    IntervalCalibrator, fit_interval_calibrator_on_val, apply_calibrator_to_subs
)
from fusionlab.params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
from fusionlab.nn.losses import make_weighted_pinball
from fusionlab.nn.pinn.models import GeoPriorSubsNet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Make future forecasts from Stage-1/2 artifacts."
    )
    p.add_argument("--results-dir", default="results",
                   help="Root folder that contains Stage-1 runs.")
    p.add_argument("--city", default=None,
                   help="City name hint (e.g., 'nansha', 'zhongshan').")
    p.add_argument("--manifest-path", default=None,
                   help="Explicit manifest path.")
    p.add_argument("--prefer", default="tuned",
                   choices=["tuned", "trained"],
                   help="Model preference (fallback handled automatically).")
    p.add_argument("--seed-split", default="val",
                   choices=["val", "train"],
                   help="Which NPZ to use as inference seeds.")
    p.add_argument("--apply-calibration", default="auto",
                   choices=["auto", "none", "fit_val", "source"],
                   help=("auto: load saved factors if present; "
                         "fit_val: re-fit on val targets; "
                         "source: load interval_factors_80.npy; "
                         "none: raw model quantiles."))
    p.add_argument("--out-dir", default=None,
                   help="Where to save outputs (defaults to Stage-2 run dir).")
    p.add_argument("--export-years", nargs="*",
                   type=int, default=[],
                   help="Optional calendar years to export as separate CSVs.")
    p.add_argument("--export-cumulative", action="store_true",
                   help="Also export cumulative q50 per sample across horizons.")
    return p.parse_args()


# ---------------- helpers ----------------
def _load_manifest(results_dir: str, city_hint: str | None, manifest_path = None ) -> dict:
    mpath = _find_stage1_manifest(
        manual=manifest_path,
        base_dir=results_dir,
        city_hint=city_hint,
        model_hint=os.getenv("MODEL_NAME_OVERRIDE", "GeoPriorSubsNet"),
        prefer="timestamp",
        required_keys=("model", "stage"),
        verbose=1,
    )
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest(path_glob: str) -> str | None:
    pats = glob.glob(path_glob, recursive=True)
    if not pats:
        return None
    pats.sort(key=os.path.getmtime, reverse=True)
    return pats[0]


def _pick_model(run_dir: str, prefer: str = "tuned") -> str:
    """
    Prefer the tuned .keras; else fall back to latest trained .keras.
    """
    # tuned
    tuned = _latest(os.path.join(run_dir, "tuning", "**", "*.keras"))
    trained = _latest(os.path.join(run_dir, "train_*", "*.keras"))
    if prefer == "tuned" and tuned:
        return tuned
    return tuned or trained or ""


def _pick_npz(M: dict, split: str) -> Tuple[dict, dict | None]:
    npzs = M["artifacts"]["numpy"]
    if split == "val":
        return dict(np.load(npzs["val_inputs_npz"])), dict(np.load(npzs["val_targets_npz"]))
    if split == "train":
        return dict(np.load(npzs["train_inputs_npz"])), dict(np.load(npzs["train_targets_npz"]))
    raise ValueError(split)


def _ensure_shapes(X: dict, mode: str, horizon: int) -> dict:
    out = dict(X)
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


def _load_saved_calibrator(run_dir: str) -> IntervalCalibrator | None:
    p = _latest(os.path.join(run_dir, "train_*", "interval_factors_80.npy"))
    if not p:
        return None
    cal = IntervalCalibrator(target=0.80)
    cal.factors_ = np.load(p).astype(np.float32)
    return cal


# ---------------- main ----------------
def main():
    args = parse_args()
    M = _load_manifest(args.results_dir, args.city, args.manifest_path)
    cfg = M["config"]

    CITY = M.get("city")
    MODE = cfg["MODE"]
    H    = int(cfg["FORECAST_HORIZON_YEARS"])
    Q    = cfg.get("QUANTILES", [0.1, 0.5, 0.9])

    # Load seeds
    X_np, y_np = _pick_npz(M, args.seed_split)
    X_np = _ensure_shapes(X_np, MODE, H)
    y_map = _map_targets(y_np)

    # Encoders/scalers
    enc = M["artifacts"]["encoders"]
    coord_scaler = None
    if enc.get("coord_scaler"):
        try:
            coord_scaler = joblib.load(enc["coord_scaler"])
        except Exception:
            pass

    scaler_info = enc.get("scaler_info")
    if isinstance(scaler_info, str) and os.path.exists(scaler_info):
        scaler_info = joblib.load(scaler_info)

    # Model
    model_path = _pick_model(M["paths"]["run_dir"], prefer=args.prefer)
    if not model_path:
        raise SystemExit("No saved model found under Stage-2. "
                         "Run training/tuning first.")
    custom = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
    }
    with custom_object_scope(custom):
        model = load_model(model_path, compile=False)

    # Calibration
    calibrator = None
    if args.apply_calibration == "auto":
        calibrator = _load_saved_calibrator(M["paths"]["run_dir"])
    elif args.apply_calibration == "source":
        calibrator = _load_saved_calibrator(M["paths"]["run_dir"])
    elif args.apply_calibration == "fit_val":
        if y_map:
            ds_val = tf.data.Dataset.from_tensor_slices((X_np, y_map)).batch(32)
            calibrator = fit_interval_calibrator_on_val(model, ds_val, target=0.80)
    # else "none" -> keep None

    # Predict
    out = model.predict(X_np, verbose=0)
    data_final = out["data_final"]
    OUT_S_DIM = M["artifacts"]["sequences"]["dims"]["output_subsidence_dim"]
    OUT_G_DIM = M["artifacts"]["sequences"]["dims"]["output_gwl_dim"]

    if data_final.ndim != 4:
        raise SystemExit("This script expects probabilistic outputs "
                         "(q10/50/90). Re-run with quantiles enabled.")

    s_q = data_final[..., :OUT_S_DIM]
    g_q = data_final[..., OUT_S_DIM:]
    if calibrator is not None:
        s_q = apply_calibrator_to_subs(calibrator, s_q)

    predictions = {"subs_pred": s_q, "gwl_pred": g_q}
    target_mapping = {
        "subs_pred": cfg["cols"].get("subsidence", "subsidence"),
        "gwl_pred":  cfg["cols"].get("gwl", "GWL"),
    }
    output_dims = {"subs_pred": OUT_S_DIM, "gwl_pred": OUT_G_DIM}

    # Format nice forecast DF (adds coord_t as calendar year)
    out_dir = args.out_dir or os.path.join(M["paths"]["run_dir"], "inference")
    ensure_directory_exists(out_dir)
    save_npz_path = os.path.join(out_dir, f"{CITY}_forecast_df.npz")

    forecast_df = format_pinn_predictions(
        predictions=predictions,
        y_true_dict=y_map,
        target_mapping=target_mapping,
        scaler_info=scaler_info,
        quantiles=Q,
        forecast_horizon=H,
        output_dims=output_dims,
        include_coords=True,
        include_gwl=False,
        model_inputs=X_np,
        evaluate_coverage=False,
        savefile=save_npz_path,
        coord_scaler=coord_scaler,
        verbose=0,
    )

    # Save full table
    full_csv = os.path.join(
        out_dir,
        f"{CITY}_GeoPriorSubsNet_forecast_Future_H{H}_calibrated.csv",
    )
    forecast_df.to_csv(full_csv, index=False)
    print(f"[OK] Saved full forecast CSV -> {full_csv}")

    # Optional: per-year exports
    if args.export_years:
        yrs = set(int(y) for y in args.export_years)
        avail = set(map(int, forecast_df["coord_t"].unique()))
        missing = sorted(list(yrs - avail))
        if missing:
            print(
                f"[Warn] requested years beyond horizon or absent: {missing}. "
                f"Available years: {sorted(list(avail))}"
            )
        for y in sorted(list(yrs & avail)):
            ypath = os.path.join(out_dir, f"forecast_year_{y}.csv")
            forecast_df.loc[forecast_df["coord_t"].eq(y)].to_csv(ypath, index=False)
            print(f"  - saved {ypath}")

    # Optional: cumulative q50 per sample_idx across horizons
    if args.export_cumulative:
        cum = (forecast_df
               .sort_values(["sample_idx", "forecast_step"])
               .assign(cum_q50=lambda d: d.groupby("sample_idx")
                       ["subsidence_q50"].cumsum()))
        cum_csv = os.path.join(out_dir, "forecast_cumulative.csv")
        cum.to_csv(cum_csv, index=False)
        print(f"  - saved cumulative summary -> {cum_csv}")


if __name__ == "__main__":
    main()
