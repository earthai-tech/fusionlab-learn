# -*- coding: utf-8 -*-
# Stage-3: Build true-future sequences (if needed) + run final forecast

import os
import json
import argparse
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

from fusionlab.params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
from fusionlab.registry.utils import _find_stage1_manifest

from fusionlab.utils.forecast_utils import format_and_forecast
from fusionlab.utils.generic_utils import ensure_directory_exists
from fusionlab.utils.generic_utils import default_results_dir, getenv_stripped
from fusionlab.utils.sequence_utils import build_future_sequences_npz

from fusionlab.nn.pinn._geoprior_subnet import GeoPriorSubsNet
from fusionlab.nn.losses import make_weighted_pinball
from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn 

def _load_manifest(path: str | None, verbose: int = 1) -> tuple[dict, str]:
    """
    Helper to load the Stage-1 manifest, either from an explicit path
    or by auto-discovering it in the results directory.
    """
    if path:
        manifest_path = path
        if verbose:
            print(f"[Stage-3] Using manifest passed via CLI: {manifest_path}")
    else:
        # Auto-discover from the standard results layout
        RESULTS_DIR = default_results_dir()
        CITY_HINT = getenv_stripped("CITY")
        MODEL_HINT = getenv_stripped("MODEL_NAME_OVERRIDE", default="GeoPriorSubsNet")
        MANUAL = getenv_stripped("STAGE1_MANIFEST")

        manifest_path = _find_stage1_manifest(
            manual=MANUAL,
            base_dir=RESULTS_DIR,
            city_hint=CITY_HINT,      # e.g. "zhongshan"; None → no filter
            model_hint=MODEL_HINT,
            prefer="timestamp",       # or "mtime"
            required_keys=("model", "stage"),
            verbose=verbose,
        )
        if verbose:
            print(f"[Stage-3] Auto-discovered manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return manifest, manifest_path


def main(args):
    # -----------------------------
    # 1. Load manifest + config
    # -----------------------------
    manifest, manifest_path = _load_manifest(args.manifest, verbose=1)

    cfg = manifest["config"]
    city = manifest["city"]
    model_name = manifest["model"]

    print(f"[Manifest] Loaded city={city} model={model_name}")

    train_end_year = cfg["TRAIN_END_YEAR"]
    forecast_start_year = cfg["FORECAST_START_YEAR"]
    forecast_horizon_years = cfg["FORECAST_HORIZON_YEARS"]

    # TIME_STEPS can appear as either UPPER or lower case
    time_steps = cfg.get("TIME_STEPS", cfg.get("time_steps"))

    time_col = cfg["cols"]["time"]
    time_col_num = cfg["cols"]["time_numeric"]
    lon_col = cfg["cols"]["lon"]
    lat_col = cfg["cols"]["lat"]
    subs_col = cfg["cols"]["subsidence"]
    gwl_col = cfg["cols"]["gwl"]
    h_field_col = cfg["cols"]["h_field"]

    static_features = cfg["features"]["static"]
    dynamic_features = cfg["features"]["dynamic"]
    future_features = cfg["features"]["future"]
    group_id_cols = cfg["features"]["group_id_cols"]

    artifacts = manifest["artifacts"]
    numpy_artifacts = artifacts.get("numpy", {})
    encoders = artifacts["encoders"]

    main_scaler_info = encoders["scaler_info"]  # dict per-target

    coord_scaler_path = encoders.get("coord_scaler")
    coord_scaler = joblib.load(coord_scaler_path) if coord_scaler_path else None

    # Prefer an explicit artifacts_dir from manifest if present
    paths = manifest.get("paths", {})
    artifacts_dir = paths.get(
        "artifacts_dir",
        os.path.dirname(artifacts["csv"]["scaled"]),
    )

    ensure_directory_exists(args.output_dir)

    # -----------------------------
    # 2. Ensure future_* NPZ exist
    # -----------------------------
    future_inputs_npz = numpy_artifacts.get("future_inputs_npz")

    if future_inputs_npz is None or not os.path.exists(future_inputs_npz):
        # We need to build them from an (optionally augmented) scaled CSV
        scaled_csv = artifacts["csv"]["scaled"]
        df_scaled = pd.read_csv(scaled_csv)

        # If you need to append scenario drivers for, e.g., 2024–2025,
        # do it here BEFORE calling build_future_sequences_npz, making
        # sure they are already scaled using the same scaler as Stage-1.
        #
        # Example:
        # df_future_scenario = pd.read_csv("scaled_scenario_2024_2025.csv")
        # df_scaled = pd.concat([df_scaled, df_future_scenario],
        #                       ignore_index=True)

        future_paths = build_future_sequences_npz(
            df_scaled=df_scaled,
            time_col=time_col,
            time_col_num=time_col_num,
            lon_col=lon_col,
            lat_col=lat_col,
            subs_col=subs_col,
            gwl_col=gwl_col,
            h_field_col=h_field_col,
            static_features=static_features,
            dynamic_features=dynamic_features,
            future_features=future_features,
            group_id_cols=group_id_cols,
            train_end_year=train_end_year,
            forecast_start_year=forecast_start_year,
            forecast_horizon_years=forecast_horizon_years,
            time_steps=time_steps,
            mode=cfg["MODE"],
            model_name=model_name,
            artifacts_dir=artifacts_dir,
            prefix="future",
            verbose=2,
        )
        future_inputs_npz = future_paths["future_inputs_npz"]

        # Optionally persist the new NPZ path(s) back into the manifest
        numpy_artifacts["future_inputs_npz"] = future_inputs_npz
        if "future_targets_npz" in future_paths:
            numpy_artifacts["future_targets_npz"] = future_paths["future_targets_npz"]
        artifacts["numpy"] = numpy_artifacts
        manifest["artifacts"] = artifacts
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            print("[Stage-3] Updated manifest with future_* NPZ paths.")

    # Load NPZ of inputs for the true future
    fut = np.load(future_inputs_npz)

    coords = fut["coords"].astype(np.float32)  # typically (B, H, 3)
    B, H, D = coords.shape
    if coord_scaler is not None:
        # Flatten → scale → reshape, same convention as Stage-1
        coords_flat = coords.reshape(-1, D)
        coords_scaled_flat = coord_scaler.transform(coords_flat)
        coords_scaled = coords_scaled_flat.reshape(B, H, D).astype(np.float32)
    else:
        coords_scaled = coords

    x_future = {
        "coords": coords_scaled,
        "dynamic_features": fut["dynamic_features"],
        "static_features": fut["static_features"],
        "future_features": fut["future_features"],
    }

    # H_field is optional in the NPZ (for some experiments it may be absent)
    if "H_field" in fut.files:
        x_future["H_field"] = fut["H_field"]
    else:
        x_future["H_field"] = None

    # -----------------------------
    # 3. Load trained model
    # -----------------------------
    
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

    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(args.model_path, compile=False)
        print(f"[Model] Loaded from {args.model_path}")
        
    # Dimensions of the two heads (same as in training script)
    s_dim = getattr(model, "output_subsidence_dim", None)
    g_dim = getattr(model, "output_gwl_dim", None)
    if s_dim is None or g_dim is None:
        # Fallback to manifest info if attributes are missing
        seq_dims = artifacts["sequences"]["dims"]
        s_dim = seq_dims["output_subsidence_dim"]
        g_dim = seq_dims["output_gwl_dim"]

    # -----------------------------
    # 4. Run predictions
    # -----------------------------
    y_pred_raw = model.predict(x_future, batch_size=args.batch_size, verbose=1)

    # GeoPriorSubsNet.predict usually returns {'data_final': ...}
    if isinstance(y_pred_raw, dict):
        if "data_final" in y_pred_raw:
            data_final = y_pred_raw["data_final"]
        else:
            raise KeyError(
                f"Expected key 'data_final' in prediction dict, "
                f"got keys={list(y_pred_raw.keys())}"
            )
    else:
        # Some variants may return the tensor directly
        data_final = y_pred_raw

    # data_final shape is either:
    #   (B, H, Q, O_total) for probabilistic mode
    #   (B, H, O_total)    for point forecasts
    if data_final.ndim == 4:
        # (B, H, Q, O_total)
        subs_pred = data_final[..., :s_dim]         # (B, H, Q, Os)
        # gwl_pred  = data_final[..., s_dim:s_dim+g_dim]   # if needed later
    elif data_final.ndim == 3:
        # (B, H, O_total)
        subs_pred = data_final[..., :s_dim]         # (B, H, Os)
        # gwl_pred  = data_final[..., s_dim:s_dim+g_dim]
    else:
        raise RuntimeError(
            f"Unexpected data_final rank {data_final.ndim}; "
            "expected 3 or 4."
        )

    # coords_scaled is already (B, H, 3) for the forecast horizon
    coords_horizon = coords_scaled

    # -----------------------------
    # 5. Use format_and_forecast
    # -----------------------------
    H = forecast_horizon_years
    
    # Attach actual scaler objects
    if isinstance(main_scaler_info, dict):
        for k, v in main_scaler_info.items():
            if isinstance(v, dict) and "scaler_path" in v and "scaler" not in v:
                p = v["scaler_path"]
                if p and os.path.exists(p):
                    try:
                        v["scaler"] = joblib.load(p)
                    except Exception:
                        # Keep going even if one scaler fails to load
                        pass

    df_eval, df_future = format_and_forecast(
        y_pred={"subs_pred": subs_pred},
        y_true=None,  # real future → no actuals
        coords=coords_horizon,
        quantiles=args.quantiles,  # e.g. [0.1, 0.5, 0.9]
        target_name="subsidence",
        target_key_pred="subs_pred",
        component_index=0,
        scaler_info=main_scaler_info,
        coord_scaler=coord_scaler,
        coord_columns=("coord_t", "coord_x", "coord_y"),
        train_end_time=train_end_year,
        forecast_start_time=forecast_start_year,
        forecast_horizon=forecast_horizon_years,
        csv_eval_path=None,
        csv_future_path=os.path.join(
            args.output_dir,
            f"{city}_GeoPriorSubsNet_forecast_Future_H{H}_"
            f"{forecast_start_year}_{forecast_start_year + H - 1}.csv",
        ),
        # *No* metrics for real future
        eval_metrics=False,
        verbose=2,
        future_mode="abs_cum", # absolute cummulative 
    )

    print("[Stage-3] Future forecast complete.")
    print(f"  Future CSV: {df_future.shape} rows")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Path to Stage-1 manifest.json. If omitted, the script will "
            "attempt to auto-discover it using the standard results layout."
        ),
    )
    ap.add_argument(
        "--model-path",
        required=True,
        help="Path to trained Keras model (from Stage-2)",
    )
    ap.add_argument(
        "--output-dir",
        default="./results_future",
        help="Where to write future forecast CSV",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 0.9],
        help="Quantiles corresponding to subsidence_qXX outputs.",
    )
    parsed_args = ap.parse_args()
    main(parsed_args)
