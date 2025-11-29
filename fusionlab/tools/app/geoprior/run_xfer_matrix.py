# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations
import os
import json
import glob
import datetime as dt
import numpy as np
import joblib
import csv
from typing import Sequence, Optional, Dict, Any, Callable

from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

from ...._optdeps import with_progress 
from ....registry.utils import _find_stage1_manifest, reproject_dynamic_scale
from ....utils.generic_utils import ensure_directory_exists
from ....utils.scale_metrics import inverse_scale_target
from ....utils.forecast_utils import format_and_forecast

from ....nn.pinn.op import extract_physical_parameters
from ....nn.pinn.models import GeoPriorSubsNet
from ....params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
from ....nn.losses import make_weighted_pinball
from ....nn.keras_metrics import _to_py, coverage80_fn, sharpness80_fn
from ....nn.calibration import (
    IntervalCalibrator, fit_interval_calibrator_on_val,
    apply_calibrator_to_subs,
)

# ------------- helpers -------------
def _load_manifest_for_city(city: str, results_dir: str, manual=None) -> dict:
    mpath = _find_stage1_manifest(
        manual=manual,
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

def run_xfer_matrix(
    city_a: str,
    city_b: str,
    *,
    results_dir: str = "results",
    splits: Sequence[str] = ("val", "test"),
    calib_modes: Sequence[str] = ("none", "source", "target"),
    rescale_to_source: bool = False,
    batch_size: int = 32,
    quantiles_override: Optional[Sequence[float]] = None,
    out_dir: Optional[str] = None,
    write_json: bool = True,
    write_csv: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None, 
    **kws
) -> Dict[str, Any]:
    """
    Cross-city transferability matrix (A->B and B->A) in a GUI-friendly way.

    This wraps `run_one_direction` for all combinations of:
        - directions: A->B, B->A
        - splits:    e.g. ["val", "test"]
        - calib_modes: ["none", "source", "target"]

    It also optionally writes:
        - xfer_results.json
        - xfer_results.csv

    Parameters
    ----------
    city_a, city_b :
        Names of the two cities (must match Stage-1 manifests).
    results_dir :
        Base results directory where Stage-1 run dirs live.
    splits :
        Iterable of splits to evaluate, e.g. ("val", "test").
    calib_modes :
        Calibration modes to run, e.g. ("none", "source", "target").
    rescale_to_source :
        If True, reproject target dynamic features to the
        source city's MinMax scaling (strict domain test).
    batch_size :
        Batch size for all evaluation loops.
    quantiles_override :
        Optional override of quantiles (else use manifest config).
    out_dir :
        Optional explicit output directory. If None, a new folder
        like `results/xfer/nansha_to_zhongshan/YYYYMMDD-HHMMSS`
        will be created.
    write_json, write_csv :
        If True, write xfer_results.json / xfer_results.csv.
    logger :
        Optional logging function (e.g. GUI text area).
        Defaults to `print`.
    stop_check :
        Optional callable returning True to abort early
        (e.g. GUI "Cancel" button).

    Returns
    -------
    out : dict
        {
          "out_dir": ...,
          "results": [...],
          "json_path": str | None,
          "csv_path": str | None,
        }
    """
    # ----------------- small helpers -----------------
    def log(msg: str) -> None:
        (logger or print)(msg)

    def should_stop() -> bool:
        return bool(stop_check and stop_check())
    
    def _progress(value: float, message: str) -> None:
        """Safe wrapper around GUI progress callback."""
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
            # Never crash xfer because the GUI callback misbehaved.
            pass
    # ----------------- load manifests -----------------
    M_A = _load_manifest_for_city(city_a, results_dir)
    M_B = _load_manifest_for_city(city_b, results_dir)

    if out_dir is None:
        out_dir = os.path.join(
            results_dir,
            "xfer",
            f"{city_a}_to_{city_b}",
            dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    ensure_directory_exists(out_dir)
    log(f"[XFER] Output directory: {out_dir}")
    
    _progress(0.05, "XFER: manifests loaded and output dir ready")
    # ----------------- run all directions/modes -----------------
    results: list[Dict[str, Any]] = []
    directions = [
        ("A_to_B", M_A, M_B),
        ("B_to_A", M_B, M_A),
    ]
    
    total_jobs = len(directions) * len(splits) * len(calib_modes)
    done_jobs = 0
    base = 0.10
    span = 0.80  # we’ll keep [0.10, 0.90] for the combos

    if total_jobs > 0:
        _progress(base, "XFER: starting transfer combinations")
        
    stopped = False
    for tag, M_src, M_tgt in directions:
        if stopped:
            break
        for split in splits:
            if stopped:
                break
            for cm in calib_modes:
                if should_stop():
                    log(
                        "[XFER] stop_check=True; aborting remaining "
                        "directions/splits/modes."
                    )
                    stopped = True
                    # mark as cancelled at current progress
                    frac_cancel = base + span * (
                        done_jobs / max(1, total_jobs)
                    )
                    _progress(frac_cancel, "XFER: cancelled by user")
                    break

                done_jobs += 1
                frac = base + span * (
                    (done_jobs - 1) / max(1, total_jobs)
                )
                msg = (
                    f"XFER: {tag}, split={split}, calib={cm} "
                    f"({done_jobs}/{total_jobs})"
                )
                _progress(frac, msg)

                log(
                    f"[XFER] direction={tag}, split={split}, "
                    f"calib_mode={cm} ..."
                )
                r = run_one_direction(
                    M_src=M_src,
                    M_tgt=M_tgt,
                    split=split,
                    calib_mode=cm,
                    rescale_to_source=rescale_to_source,
                    batch_size=batch_size,
                    quantiles_override=(
                        list(quantiles_override)
                        if quantiles_override is not None
                        else None
                    ),
                    logger=logger,          # keep as-is / optional
                    stop_check=stop_check,  # still stops inner loop
                )
                if r is not None:
                    r["direction"] = tag
                    r["source_city"] = M_src.get("city")
                    r["target_city"] = M_tgt.get("city")
                    results.append(r)

    if not stopped and total_jobs > 0:
        # Move to the upper bound of the combo slice before writing files
        _progress(base + span, "XFER: aggregating and writing results")

    # ----------------- write JSON (optional) -----------------
    json_path = None
    if write_json:
        json_path = os.path.join(out_dir, "xfer_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log(f"[XFER] Saved transfer results JSON -> {json_path}")
        _progress(0.93, "XFER: xfer_results.json written")

    # ----------------- write CSV (optional) -----------------
    csv_path = None
    if write_csv:
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
            "physics.epsilon_prior",
            "physics.epsilon_cons",
            "keras_eval_scaled.loss",
            "keras_eval_scaled.subs_pred_mae",
            "keras_eval_scaled.gwl_pred_mae",
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
            h_mae_keys |= set((r.get("per_horizon_mae") or {}).keys())
            h_r2_keys |= set((r.get("per_horizon_r2") or {}).keys())

        h_mae_keys = _sorted_hkeys(h_mae_keys)
        h_r2_keys = _sorted_hkeys(h_r2_keys)

        cols = (
            base_cols
            + [f"per_horizon_mae.{k}" for k in h_mae_keys]
            + [f"per_horizon_r2.{k}" for k in h_r2_keys]
        )

        csv_path = os.path.join(out_dir, "xfer_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in results:
                phys = r.get("physics") or {}
                ev = r.get("keras_eval_scaled") or {}

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

                ph_mae = r.get("per_horizon_mae") or {}
                ph_r2 = r.get("per_horizon_r2") or {}
                row.extend([ph_mae.get(k, "NA") for k in h_mae_keys])
                row.extend([ph_r2.get(k, "NA") for k in h_r2_keys])
                w.writerow(row)

        log(f"[XFER] Saved transfer CSV -> {csv_path}")
        _progress(0.97, "XFER: xfer_results.csv written")

    # Final completion tick
    _progress(1.0, "XFER: transfer matrix complete")
    
    return {
        "out_dir": out_dir,
        "results": results,
        "json_path": json_path,
        "csv_path": csv_path,
    }

def run_one_direction(
    M_src: dict,
    M_tgt: dict,
    split: str,
    calib_mode: str,
    rescale_to_source: bool,
    batch_size: int,
    quantiles_override: Optional[Sequence[float]] = None,
    logger: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run cross-city transfer in a single direction (source -> target).

    Parameters
    ----------
    M_src, M_tgt :
        Stage-1 manifests for source and target cities (as loaded JSON).
    split :
        Split on the target city to evaluate: "val" or "test".
    calib_mode :
        One of {"none", "source", "target"}:
        - "none"   : no interval calibration
        - "source" : use calibrator from source city training run
        - "target" : fit calibrator on target VAL split
    rescale_to_source :
        If True, reproject target dynamic features into the source
        scaler space (strict domain test).
    batch_size :
        Batch size for tf.data loops and evaluate().
    quantiles_override :
        Optional override of quantiles; if None, read from target config.
    logger :
        Optional logging callable; defaults to `print`.
    stop_check :
        Optional callable with no arguments returning True when GUI
        requests early abort.

    Returns
    -------
    result : dict or None
        Metrics + CSV paths for this direction/split/calib combination,
        or None if the split is not available.
    """

    def log(msg: str) -> None:
        (logger or print)(msg)

    def should_stop() -> bool:
        return bool(stop_check and stop_check())

    # -------- 1) Load NPZs for target city --------
    X_tgt, y_tgt = _pick_npz(M_tgt, split)
    if X_tgt is None:
        log(f"[XFER] No NPZs for target split={split!r}; skipping.")
        return None  # no split

    cfg_t = M_tgt["config"]
    MODE = cfg_t["MODE"]
    H = cfg_t["FORECAST_HORIZON_YEARS"]
    Q = list(quantiles_override) if quantiles_override is not None else \
        cfg_t.get("QUANTILES", [0.1, 0.5, 0.9])

    # Build input shapes
    X_tgt = _ensure_shapes(X_tgt, MODE, H)
    y_map = _map_targets(y_tgt)

    # Align static features to the *source* feature space
    X_tgt = _align_static_to_source(X_tgt, M_src, M_tgt)

    dims_src = M_src["artifacts"]["sequences"]["dims"]
    log(
        f"[XFER] Source model expects static={dims_src.get('static_input_dim')} "
        f"dynamic={dims_src.get('dynamic_input_dim')}"
    )
    log(
        f"[XFER] Target NPZ has   static="
        f"{X_tgt.get('static_features', np.empty((0, 0))).shape[-1]} "
        f"dynamic={X_tgt['dynamic_features'].shape[-1]}"
    )

    s_src = int(dims_src.get("static_input_dim", 0))
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

    if should_stop():
        log("[XFER] stop_check=True before scaling / model load; aborting direction.")
        return None

    # -------- 2) Optional strict domain test: reproject dynamic scaling --------
    if rescale_to_source:
        enc_t = M_tgt["artifacts"]["encoders"]
        scaler_info = enc_t.get("scaler_info")
        if isinstance(scaler_info, str) and os.path.exists(scaler_info):
            scaler_info = joblib.load(scaler_info)

        enc_s = M_src["artifacts"]["encoders"]
        src_scaler_path = enc_s.get("main_scaler")
        if not src_scaler_path:
            raise SystemExit("Source 'main_scaler' path missing in manifest.")

        dyn_names = M_tgt["config"]["features"]["dynamic"]
        X_tgt = reproject_dynamic_scale(
            X_np=X_tgt,
            target_scaler_info=scaler_info,
            source_scaler_path=src_scaler_path,
            dynamic_feature_order=dyn_names,
        )

    if should_stop():
        log("[XFER] stop_check=True before loading model; aborting direction.")
        return None

    # -------- 3) Load latest model under *source* run dir --------
    model_path = _latest_model_under(M_src["paths"]["run_dir"])
    if not model_path:
        raise SystemExit(f"No .keras found under {M_src['paths']['run_dir']}")

    model_dir = os.path.dirname(model_path)
    log(f"[XFER] Using source model: {model_path}")

    custom_objects = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
        "coverage80_fn": coverage80_fn,
        "sharpness80_fn": sharpness80_fn,
    }

    with custom_object_scope(custom_objects):
        model = load_model(model_path, compile=True)

    if should_stop():
        log("[XFER] stop_check=True after model load; aborting direction.")
        return None

    # -------- 4) Prepare optional calibrator --------
    cal: Optional[IntervalCalibrator] = None
    if calib_mode == "source":
        cal = _load_source_calibrator(M_src["paths"]["run_dir"])
        if cal is not None:
            log("[XFER] Using source-city interval calibrator.")
    elif calib_mode == "target":
        # fit on target VAL if it exists
        try:
            vx, vy = _pick_npz(M_tgt, "val")
            if vx is not None:
                vx = _ensure_shapes(vx, MODE, H)
                vy = _map_targets(vy)
                ds_val = tf.data.Dataset.from_tensor_slices((vx, vy)).batch(
                    batch_size
                )
                log("[XFER] Fitting target-city interval calibrator on VAL.")
                cal = fit_interval_calibrator_on_val(
                    model, ds_val, target=0.80
                )
        except Exception as e:
            log(f"[XFER] Target calibrator fit failed: {e}")
            cal = None

    # -------- 5) Export physical parameters (best-effort) --------
    try:
        extract_physical_parameters(
            model,
            to_csv=True,
            filename=f"{M_tgt.get('city')}_xfer_physical_parameters.csv",
            save_dir=model_dir,
            model_name="geoprior",
        )
        log("[XFER] Exported physical parameters CSV.")
    except Exception as e:
        log(f"[XFER] Physical parameter export failed: {e}")

    if should_stop():
        log("[XFER] stop_check=True before prediction; aborting direction.")
        return None

    # -------- 6) Predict on target split --------
    log("[XFER] Running model.predict on target NPZ...")
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

    # -------- 7) Encoders / scalers (target city) --------
    enc = M_tgt["artifacts"]["encoders"]
    coord_scaler = None
    if enc.get("coord_scaler"):
        try:
            coord_scaler = joblib.load(enc["coord_scaler"])
        except Exception as e:
            log(f"[XFER] coord_scaler load failed: {e}")

    scaler_info = enc.get("scaler_info")
    if isinstance(scaler_info, str) and os.path.exists(scaler_info):
        try:
            scaler_info = joblib.load(scaler_info)
        except Exception as e:
            log(f"[XFER] scaler_info load failed: {e}")

    if isinstance(scaler_info, dict):
        for k, v in scaler_info.items():
            if isinstance(v, dict) and "scaler_path" in v and "scaler" not in v:
                p = v["scaler_path"]
                if p and os.path.exists(p):
                    try:
                        v["scaler"] = joblib.load(p)
                    except Exception:
                        # non-fatal
                        pass

    cols_cfg = M_tgt["config"]["cols"]
    SUBS_COL = cols_cfg.get("subsidence", "subsidence")

    # -------- 8) Per-horizon + overall metrics (physical units) --------
    per_horizon_mae: Dict[str, float] = {}
    per_horizon_r2: Dict[str, float] = {}
    overall_mae = overall_mse = overall_r2 = None

    if y_map and ("subs_pred" in y_map) and scaler_info is not None:
        y_true_subs_scaled = y_map["subs_pred"]

        if data_final.ndim == 4:
            q_arr = np.asarray(Q, dtype=np.float32)
            med_idx = int(np.argmin(np.abs(q_arr - 0.5)))
            y_pred_subs_scaled = predictions["subs_pred"][:, :, med_idx, :]
        else:
            y_pred_subs_scaled = predictions["subs_pred"]

        y_true_subs_scaled = y_true_subs_scaled[..., :1]
        y_pred_subs_scaled = y_pred_subs_scaled[..., :1]

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

        H_eff = y_true_subs_phys.shape[1]
        for h in range(H_eff):
            yt = y_true_subs_phys[:, h, :].reshape(-1)
            yp = y_pred_subs_phys[:, h, :].reshape(-1)
            per_horizon_mae[f"H{h+1}"] = float(mean_absolute_error(yt, yp))
            per_horizon_r2[f"H{h+1}"] = float(r2_score(yt, yp))

        yt_all = y_true_subs_phys.reshape(-1)
        yp_all = y_pred_subs_phys.reshape(-1)
        overall_mae = float(mean_absolute_error(yt_all, yp_all))
        overall_mse = float(np.mean((yt_all - yp_all) ** 2))
        overall_r2 = float(r2_score(yt_all, yp_all))

    # -------- 9) Optional scaled-space evaluation & physics --------
    eval_scaled = None
    physics_diag = None
    if y_map and not should_stop():
        ds = tf.data.Dataset.from_tensor_slices((X_tgt, y_map)).batch(
            batch_size
        )
        try:
            eval_scaled = model.evaluate(ds, return_dict=True, verbose=0)
            log(f"[XFER] Scaled evaluation on {M_tgt.get('city')}: {eval_scaled}")
        except Exception as e:
            log(f"[XFER] model.evaluate failed: {e}")
            eval_scaled = None

    if eval_scaled is not None:
        phys_keys = ("epsilon_prior", "epsilon_cons")
        try:
            physics_diag = {
                k: float(_to_py(eval_scaled[k]))
                for k in phys_keys
                if k in eval_scaled
            }
            if physics_diag:
                log(
                    "[XFER] Physics diagnostics from scaled evaluation: "
                    f"{physics_diag}"
                )
        except Exception as e:
            log(f"[XFER] Physics diagnostic extraction failed: {e}")
            physics_diag = None

    # Export physics payload (best-effort)
    try:
        model.export_physics_payload(
            X_tgt,
            max_batches=None,
            save_path=os.path.join(
                model_dir, f"{M_tgt.get('city')}_xfer_physics_payload.npz"
            ),
            format="npz",
            overwrite=True,
            metadata={"city": M_tgt.get("city"), "split": split},
        )
        log("[XFER] Physics payload saved.")
    except Exception as e:
        log(f"[XFER] Physics payload export failed: {e}")

    # -------- 10) Interval metrics: coverage / sharpness in physical space --------
    coverage80 = sharpness80 = None
    if (
        y_map
        and data_final.ndim == 4
        and scaler_info is not None
        and not should_stop()
    ):
        ds_calc = tf.data.Dataset.from_tensor_slices((X_tgt, y_map)).batch(
            batch_size
        )
        y_true_list, s_q_list = [], []
        for xb, yb in with_progress(
            ds_calc, desc=f"Diagnose {M_tgt.get('city')} xfer-metrics"
        ):
            out = model(xb, training=False)
            s_q_b, _ = model.split_data_predictions(out["data_final"])
            y_true_list.append(yb["subs_pred"])
            s_q_list.append(s_q_b)

        if y_true_list and s_q_list:
            y_true_scaled = tf.concat(y_true_list, axis=0)
            s_q_scaled = tf.concat(s_q_list, axis=0)

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

            y_true_phys_tf = tf.convert_to_tensor(
                y_true_phys_np, dtype=tf.float32
            )
            s_q_phys_tf = tf.convert_to_tensor(
                s_q_phys_np, dtype=tf.float32
            )

            coverage80 = float(
                coverage80_fn(y_true_phys_tf, s_q_phys_tf).numpy()
            )
            sharpness80 = float(
                sharpness80_fn(y_true_phys_tf, s_q_phys_tf).numpy()
            )
            log(
                f"[XFER] coverage80={coverage80:.3f}, "
                f"sharpness80={sharpness80:.3f}"
            )

    # -------- 11) Format calibrated predictions (eval + future) --------
    y_true_for_format = None
    if y_map:
        y_true_for_format = {}
        if "subs_pred" in y_map:
            y_true_for_format["subsidence"] = y_map["subs_pred"]
        if "gwl_pred" in y_map:
            y_true_for_format["gwl"] = y_map["gwl_pred"]

    train_end_year = cfg_t.get("TRAIN_END_YEAR")
    forecast_start_year = cfg_t.get("FORECAST_START_YEAR")

    future_grid = None
    if forecast_start_year is not None:
        future_grid = np.arange(
            float(forecast_start_year),
            float(forecast_start_year) + float(H),
            dtype=float,
        )

    base_name = (
        f"{M_src.get('city')}_to_{M_tgt.get('city')}_xfer_"
        f"{split}_{calib_mode}"
    )
    xfer_eval_csv = os.path.join(model_dir, base_name + "_eval.csv")
    xfer_future_csv = os.path.join(model_dir, base_name + "_future.csv")

    df_eval, df_future = format_and_forecast(
        y_pred=predictions,
        y_true=y_true_for_format,
        coords=X_tgt.get("coords", None),
        quantiles=Q if data_final.ndim == 4 else None,
        target_name=SUBS_COL,
        target_key_pred="subs_pred",
        component_index=0,
        scaler_info=scaler_info,
        coord_scaler=coord_scaler,
        coord_columns=("coord_t", "coord_x", "coord_y"),
        train_end_time=train_end_year,
        forecast_start_time=forecast_start_year,
        forecast_horizon=H,
        future_time_grid=future_grid,
        eval_forecast_step=None,
        eval_export="all",
        sample_index_offset=0,
        city_name=M_tgt.get("city"),
        model_name=M_src.get("model") or "GeoPriorSubsNet",
        dataset_name=f"xfer_{split}_{calib_mode}",
        csv_eval_path=xfer_eval_csv,
        csv_future_path=xfer_future_csv,
        time_as_datetime=False,
        time_format=None,
        verbose=1,
        eval_metrics=False,
        metrics_column_map=None,
        metrics_quantile_interval=None,
        metrics_per_horizon=False,
        metrics_extra=None,
        metrics_extra_kwargs=None,
        metrics_savefile=None,
        metrics_save_format=".json",
        metrics_time_as_str=True,
        value_mode="rate",
        logger=log,
    )

    if df_eval is not None and not df_eval.empty:
        log(f"[XFER] Saved EVAL forecast CSV -> {xfer_eval_csv}")
    else:
        log("[XFER] No eval DF (no y_true or empty).")

    if df_future is not None and not df_future.empty:
        log(f"[XFER] Saved FUTURE forecast CSV -> {xfer_future_csv}")
    else:
        log("[XFER] No future DF (check forecast_start_year / horizon).")

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
        "overall_r2": overall_r2,
        "per_horizon_mae": per_horizon_mae,
        "per_horizon_r2": per_horizon_r2,
        "csv_eval": xfer_eval_csv,
        "csv_future": xfer_future_csv,
    }

