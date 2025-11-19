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
    --eval-losses --eval-physics --no-figs 

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
from tensorflow.keras.optimizers import Adam

# --- quiet logs ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# --- fusionlab imports ---
from fusionlab._optdeps import with_progress 
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

def _infer_input_dims_from_X(X: dict) -> tuple[int, int, int]:
    """
    Infer (static_input_dim, dynamic_input_dim, future_input_dim)
    from the Stage-1 NPZ inputs dictionary.

    Parameters
    ----------
    X : dict
        Dictionary with keys:
        - 'dynamic_features' (required)
        - 'static_features' (optional, may be None)
        - 'future_features' (optional, may be None)

    Returns
    -------
    static_dim : int
        Last-dimension size of static_features (0 if missing / None).
    dynamic_dim : int
        Last-dimension size of dynamic_features.
    future_dim : int
        Last-dimension size of future_features (0 if missing / None).
    """
    dyn = np.asarray(X["dynamic_features"])
    dynamic_dim = int(dyn.shape[-1])

    static = X.get("static_features")
    static_dim = int(static.shape[-1]) if static is not None else 0

    fut = X.get("future_features")
    future_dim = int(fut.shape[-1]) if fut is not None else 0

    return static_dim, dynamic_dim, future_dim


def _load_best_hps_near_model(model_path: str) -> dict:
    """
    Locate and load tuned hyperparameters stored next to a tuned model.

    Search order inside `run_YYYYMMDD-HHMMSS` directory:
    1. <city>_GeoPrior_best_hps.json (city inferred from model filename)
    2. tuning_summary.json['best_hps']

    Parameters
    ----------
    model_path : str
        Path passed via --model-path, typically something like:
        .../tuning/run_YYYYMMDD-HHMMSS/nansha_GeoPrior_best.keras

    Returns
    -------
    best_hps : dict
        Dictionary of best hyperparameters.

    Raises
    ------
    FileNotFoundError
        If no hyperparameter JSON can be found.
    ValueError
        If a candidate JSON exists but does not contain hyperparameters.
    """
    run_dir = os.path.dirname(os.path.abspath(model_path))
    base = os.path.basename(model_path)

    # Try to infer the city prefix from '<city>_GeoPrior_best.keras'
    city_name = None
    marker = "_GeoPrior_best"
    if marker in base:
        city_name = base.split(marker)[0]

    # 1) Try explicit '<city>_GeoPrior_best_hps.json'
    if city_name is not None:
        hps_path = os.path.join(run_dir, f"{city_name}_GeoPrior_best_hps.json")
        if os.path.exists(hps_path):
            with open(hps_path, "r", encoding="utf-8") as f:
                best_hps = json.load(f)
            if isinstance(best_hps, dict) and best_hps:
                print(f"[Fallback] Loaded best_hps from: {hps_path}")
                return best_hps

    # 2) Try tuning_summary.json["best_hps"]
    summary_path = os.path.join(run_dir, "tuning_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        best_hps = summary.get("best_hps", {})
        if isinstance(best_hps, dict) and best_hps:
            print(f"[Fallback] Loaded best_hps from: {summary_path}")
            return best_hps

    raise FileNotFoundError(
        f"Could not find best hyperparameters near model_path={model_path}.\n"
        f"Looked for '<city>_GeoPrior_best_hps.json' and 'tuning_summary.json'."
    )


def _build_geoprior_from_hps(
    M: dict,
    X: dict,
    best_hps: dict,
    out_s_dim: int,
    out_g_dim: int,
    mode: str,
    horizon: int,
    quantiles: list[float] | None,
) -> GeoPriorSubsNet:
    """
    Reconstruct a GeoPriorSubsNet instance using Stage-1 metadata and
    the tuned hyperparameters.

    This mirrors the configuration used during Stage-2 tuning,
    but is intentionally conservative: only the parameters that are
    clearly defined in Stage-1 manifest + best_hps are used.

    Parameters
    ----------
    M : dict
        Stage-1 manifest dictionary.
    X : dict
        Inputs NPZ dictionary (already sanitized and shape-ensured).
    best_hps : dict
        Hyperparameters loaded from *_GeoPrior_best_hps.json or
        tuning_summary.json["best_hps"].
    out_s_dim : int
        Output dimension for subsidence head.
    out_g_dim : int
        Output dimension for GWL head.
    mode : str
        Sequence mode ('tft_like' or 'pihal_like'), from Stage-1 config.
    horizon : int
        Forecast horizon (number of time steps).
    quantiles : list[float] or None
        Quantiles used for probabilistic outputs.

    Returns
    -------
    model : GeoPriorSubsNet
        A freshly instantiated (and compiled) GeoPriorSubsNet.
    """
    # # Helper: JSON has string keys for quantile weight dicts; coerce to float.
    def _coerce_quantile_weights(d: dict, default: dict) -> dict:
        if not d:
            return default
        out = {}
        for k, v in d.items():
            try:
                q = float(k)
            except (TypeError, ValueError):
                q = k
            out[q] = float(v)
        return out
    
    cfg = M["config"]

    # Infer input dims directly from Stage-1 NPZ arrays
    static_dim, dynamic_dim, future_dim = _infer_input_dims_from_X(X)

    # Attention stack: fall back to a sensible default if not present
    attention_levels = cfg.get(
        "ATTENTION_LEVELS",
        ["cross", "hierarchical", "memory"],
    )

    # Whether we used effective H during Stage-2
    censor_cfg = cfg.get("censoring", {}) or {}
    use_effective_h = censor_cfg.get("use_effective_h_field", True)

    # Feature-processing mode
    use_vsn = bool(best_hps.get("use_vsn", True))
    feature_processing = "vsn" if use_vsn else "dense"

    architecture_config = {
        "encoder_type": "hybrid",
        "decoder_attention_stack": attention_levels,
        "feature_processing": feature_processing,
    }

    loss_weights = {"subs_pred": 1.0, "gwl_pred": 0.5}


    SUBS_WEIGHTS_RAW = cfg.get(
        "SUBS_WEIGHTS",
        {0.1: 3.0, 0.5: 1.0, 0.9: 3.0},
    )
    GWL_WEIGHTS_RAW = cfg.get(
        "GWL_WEIGHTS",
        {0.1: 1.5, 0.5: 1.0, 0.9: 1.5},
    )

    SUBS_WEIGHTS = _coerce_quantile_weights(SUBS_WEIGHTS_RAW, {0.1: 3.0, 0.5: 1.0, 0.9: 3.0})
    GWL_WEIGHTS  = _coerce_quantile_weights(GWL_WEIGHTS_RAW,  {0.1: 1.5, 0.5: 1.0, 0.9: 1.5})
    
    loss_dict = {
        "subs_pred": make_weighted_pinball(
            quantiles, SUBS_WEIGHTS) if quantiles else tf.keras.losses.MSE,
        "gwl_pred":  make_weighted_pinball(
            quantiles, GWL_WEIGHTS)  if quantiles else tf.keras.losses.MSE,
    }
    # Instantiate the model core
    model = GeoPriorSubsNet(
        static_input_dim=static_dim,
        dynamic_input_dim=dynamic_dim,
        future_input_dim=future_dim,
        output_subsidence_dim=out_s_dim,
        output_gwl_dim=out_g_dim,
        forecast_horizon=horizon,
        mode=mode,
        attention_levels=attention_levels,
        quantiles=quantiles,
        # physics switches from best_hps
        pde_mode=best_hps.get("pde_mode", "both"),
        scale_pde_residuals=bool(best_hps.get("scale_pde_residuals", True)),
        kappa_mode=best_hps.get("kappa_mode", "bar"),
        use_effective_h=use_effective_h,
        # architecture hyperparameters
        embed_dim=int(best_hps.get("embed_dim", 32)),
        hidden_units=int(best_hps.get("hidden_units", 96)),
        lstm_units=int(best_hps.get("lstm_units", 96)),
        attention_units=int(best_hps.get("attention_units", 32)),
        num_heads=int(best_hps.get("num_heads", 4)),
        dropout_rate=float(best_hps.get("dropout_rate", 0.1)),
        use_vsn=use_vsn,
        vsn_units=int(best_hps.get("vsn_units", 32)),
        use_batch_norm=bool(best_hps.get("use_batch_norm", True)),
        # geomechanical priors
        mv=float(best_hps.get("mv", 5e-7)),
        kappa=float(best_hps.get("kappa", 1.0)),
        architecture_config = architecture_config
    )

    # Compile for evaluation (metrics optional, but loss is required)
    lr = float(best_hps.get("learning_rate", 5e-5))
    optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        loss_weights=loss_weights,
        # physics loss weights + LR multipliers
        lambda_gw=float(best_hps.get("lambda_gw", 1.0)),
        lambda_cons=float(best_hps.get("lambda_cons", 1.0)),
        lambda_prior=float(best_hps.get("lambda_prior", 1.0)),
        lambda_smooth=float(best_hps.get("lambda_smooth", 1.0)),
        lambda_mv=float(best_hps.get("lambda_mv", 0.0)),
        mv_lr_mult=float(best_hps.get("mv_lr_mult", 1.0)),
        kappa_lr_mult=float(best_hps.get("kappa_lr_mult", 1.0)),
    )

    print(
        "[Fallback] Reconstructed GeoPriorSubsNet from best_hps with "
        f"static_dim={static_dim}, dynamic_dim={dynamic_dim}, "
        f"future_dim={future_dim}, horizon={horizon}, mode={mode}"
    )
    return model

def _infer_best_weights_path(model_path: str) -> str | None:
    """
    Infer the best weights checkpoint path for a tuned GeoPrior model.

    Strategy
    --------
    1) Look for `tuning_summary.json` in the same folder as `model_path`
       and use the stored "best_weights_path" if it exists.
    2) Fallback: replace the `.keras` suffix by `.weights.h5`, assuming
       the naming convention <CITY>_GeoPrior_best.keras ->
       <CITY>_GeoPrior_best.weights.h5.

    Returns
    -------
    str or None
        Absolute path to the weights file if it exists, otherwise None.
    """
    run_dir = os.path.dirname(model_path)

    # 1) Preferred: tuning_summary.json
    summary_path = os.path.join(run_dir, "tuning_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            w = summary.get("best_weights_path")
            if w and os.path.exists(w):
                return w
        except Exception as e:
            print(f"[Warn] Could not read tuning_summary.json for weights: {e}")

    # 2) Name-based guess from the .keras path
    root, ext = os.path.splitext(model_path)
    guess = root + ".weights.h5"
    if os.path.exists(guess):
        return guess

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
    # T = cfg["TIME_STEPS"]
    H = cfg["FORECAST_HORIZON_YEARS"]
    FSY = cfg.get('FORECAST_START_YEAR') 
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

    # ------------------ Load or Rebuild Model ------------------
    custom_objects = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
        # if not already @register_keras_serializable, also add:
        "coverage80_fn": coverage80_fn,
        "sharpness80_fn": sharpness80_fn,
    }
    ckpt_name = "{city}_GeoPrior_best{kind}"
    #ckpt_path = os.path.join(args.model_path, ckpt_name)
    with custom_object_scope(custom_objects):
        try:
            # Preferred path: load the saved Keras model directly
            
            model = load_model(
                os.path.join(args.model_path, ckpt_name.format( city=CITY, kind=".keras")), 
                compile=True
                )
            print(f"[Model] Loaded from {args.model_path}")
        except Exception as e:
            # Robust fallback for tuned runs where load_model fails
            print(
                f"[Warn] load_model('{args.model_path}') failed: {e}\n"
                "[Warn] Attempting to rebuild GeoPriorSubsNet from "
                "best hyperparameters JSON next to the tuned model."
            )
            try:
                best_hps = _load_best_hps_near_model(args.model_path)
            except Exception as e_hps:
                raise SystemExit(
                    "Failed to load tuned hyperparameters for fallback "
                    f"model reconstruction: {e_hps}"
                ) from e

            # Rebuild architecture + compile
            model = _build_geoprior_from_hps(
                M=M,
                X=X,
                best_hps=best_hps,
                out_s_dim=OUT_S_DIM,
                out_g_dim=OUT_G_DIM,
                mode=MODE,
                horizon=H,
                quantiles=QUANTILES if isinstance(QUANTILES, list) else None,
            )
            # Try to load weights into the newly-built model from a
            # dedicated checkpoint saved by the tuner.
            weights_path = _infer_best_weights_path(args.model_path)
            if weights_path is not None:
                try:
                    model.load_weights(
                        os.path.join (weights_path, ckpt_name.format(city=CITY, kind=".weights.h5")),
                        )
                        
                    print(
                        "[Fallback] Loaded weights into rebuilt "
                        f"GeoPriorSubsNet from: {weights_path}"
                    )
                except Exception as e_w:
                    print(
                        "[Warn] Could not load weights from checkpoint:\n"
                        f"       {weights_path}\n"
                        f"       Error: {e_w}\n"
                        "       The rebuilt model is using freshly-"
                        "initialized weights. Predictions will NOT match "
                        "the tuned model."
                    )
            else:
                print(
                    "[Warn] No weights checkpoint found near tuned model.\n"
                    "       Using rebuilt model with freshly-initialized "
                    "weights. Predictions will NOT match the tuned model."
                )

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
        forecast_start_year = FSY, 
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
        for xb, yb in with_progress (ds, desc="Inference Interval diagnostic"):
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
                savefig = os.path.join(inf_dir, f"{CITY}_inference_plot"),
                show =False, 
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
                show =False 
            )
            print(f"Saved forecast figures in: {inf_dir}")
        except Exception as e:
            print(f"[Warn] plotting failed: {e}")
    
    


if __name__ == "__main__":
    main()

