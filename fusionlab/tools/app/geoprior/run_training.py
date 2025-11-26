# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
run_training
===========

Stage-2 helper: wrap training_NATCOM_GEOPRIOR into a callable function
usable from CLI and Qt GUI.

The function:

- Finds the correct Stage-1 manifest (unless an explicit path is given).
- Merges global NATCOM config with Stage-1 config.
- Builds GeoPriorSubsNet, trains it, saves weights/models/logs.
- Calibrates intervals, runs evaluation + diagnostics, saves CSV/JSON.
- Returns a small dict with key artifact paths.

Extra:
- `logger` lets you send text to the GUI console instead of print.
- `stop_check` is consulted during training via a custom callback and at
  a few safe checkpoints. When it returns True, training is stopped
  cleanly.
"""

from __future__ import annotations
from typing import Optional, Callable, Dict, Any

import os
import sys
import json
import joblib
import numpy as np
import datetime as dt
import warnings
import platform

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
    Callback,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.data import experimental as tfdata_experimental
# --- silence TF chatter at import time ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")
if hasattr(tf, "autograph") and hasattr(tf.autograph, "set_verbosity"):
    tf.autograph.set_verbosity(0)

from fusionlab._optdeps import with_progress
from fusionlab.api.util import get_table_size
from fusionlab.utils.generic_utils import (
    ensure_directory_exists,
    save_all_figures,
    default_results_dir,
    getenv_stripped,
    print_config_table,
)
from fusionlab.registry.utils import _find_stage1_manifest
from fusionlab.utils.nat_utils import (
    load_nat_config,
    load_nat_config_payload,
    ensure_input_shapes,
    map_targets_for_training,
    make_tf_dataset,
    load_scaler_info,
    save_ablation_record,
    best_epoch_and_metrics,
    build_censor_mask_from_dynamic,
    name_of,
    serialize_subs_params,
)
from fusionlab.utils.forecast_utils import format_and_forecast
from fusionlab.utils.scale_metrics import (
    inverse_scale_target,
    point_metrics,
    per_horizon_metrics,
)
from fusionlab.nn.pinn.models import GeoPriorSubsNet
from fusionlab.params import LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
from fusionlab.nn.losses import make_weighted_pinball
from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn, _to_py
from fusionlab.nn.calibration import (
    fit_interval_calibrator_on_val,
    apply_calibrator_to_subs,
)
from fusionlab.nn.utils import plot_history_in
from fusionlab.nn.pinn.op import extract_physical_parameters
from fusionlab.plot.forecast import plot_eval_future

from fusionlab.tools.app.geoprior.view_utils import ( 
    _notify_gui_forecast_views 
)
from fusionlab.tools.app.callbacks import GuiProgress #  , StopCheckCallback

GUI_CONFIG_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
class StopTrainingOnSignal(Callback):
    """Keras callback that uses a `stop_check` callable to interrupt fit."""

    def __init__(
        self,
        stop_check: Callable[[], bool],
        logger: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.stop_check = stop_check
        self._log = logger or (lambda msg: print(msg, flush=True))

    def on_batch_end(self, batch, logs=None):
        if self.stop_check and self.stop_check():
            self._log("[Stop] stop_check() returned True – stopping training.")
            self.model.stop_training = True


def _coerce_quantile_weights(d: dict | None, default: dict) -> dict:
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


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def run_training(
    manifest_path: Optional[str] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,  
    **kws
) -> Dict[str, Any]:
    """
    Run Stage-2 training & evaluation for GeoPriorSubsNet.

    Parameters
    ----------
    manifest_path : str, optional
        Explicit path to a Stage-1 ``manifest.json``. If ``None``, the
        function will search under ``default_results_dir()`` using the
        same logic as the original script via :func:`_find_stage1_manifest`.
    cfg_overrides : dict, optional
        Extra keys to override in the merged NATCOM config. This is
        applied *after* reading both global config and Stage-1 config.
    logger : callable, optional
        Function ``f(msg: str) -> None`` used for logging instead of
        plain :func:`print`. In the Qt GUI, pass something like
        ``logger=self.progress.log`` or ``self.append_log``.
    stop_check : callable, optional
        Function with signature ``stop_check() -> bool``. If provided,
        a Keras callback will call it at the end of each batch and
        interrupt training when it returns ``True``. A few post-training
        checkpoints also consult it and bail out early.

    Returns
    -------
    dict
        Dictionary with key artifact paths and main metrics, e.g.::

            {
                "run_dir": "...",
                "train_summary_json": "...",
                "metrics_json": "...",
                "eval_csv": "...",
                "future_csv": "...",
            }
    """
    def _progress_from_pct(pct: int) -> None:
        if progress_callback is None:
            return
        frac = max(0.0, min(1.0, pct / 100.0))
        progress_callback(frac, f"Training: {pct:3d}%")

    log = logger or (lambda msg: print(msg, flush=True))

    # ================================================================
    # 1. Locate Stage-1 manifest
    # ================================================================
    RESULTS_DIR = default_results_dir()

    cfg_payload = load_nat_config_payload(root=GUI_CONFIG_DIR)
    CFG_CITY = (cfg_payload.get("city") or "").strip().lower() or None
    CFG_MODEL = cfg_payload.get("model") or "GeoPriorSubsNet"

    CITY_ENV = getenv_stripped("CITY")
    MODEL_ENV = getenv_stripped("MODEL_NAME_OVERRIDE")

    CITY_HINT = CITY_ENV or CFG_CITY
    MODEL_HINT = MODEL_ENV or CFG_MODEL
    MANUAL = getenv_stripped("STAGE1_MANIFEST")

    if manifest_path is None:
        manifest_path = _find_stage1_manifest(
            manual=MANUAL,
            base_dir=RESULTS_DIR,
            city_hint=CITY_HINT,
            model_hint=MODEL_HINT,
            prefer="timestamp",
            required_keys=("model", "stage"),
            verbose=1,
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        M = json.load(f)

    manifest_city = (M.get("city") or "").strip().lower()
    log(f"[Manifest] Loaded city={manifest_city} model={M.get('model')}")

    if CFG_CITY and manifest_city and manifest_city != CFG_CITY:
        raise RuntimeError(
            "[NATCOM] Stage-1 manifest city "
            f"{manifest_city!r} does not match config CITY_NAME {CFG_CITY!r}. "
            "Run Stage-1 for this city first, or set CITY/STAGE1_MANIFEST "
            "to explicitly override."
        )

    # ================================================================
    # 2. Merge global config with Stage-1 config
    # ================================================================
    cfg_global = load_nat_config(root=GUI_CONFIG_DIR)
    cfg_manifest = M.get("config", {}) or {}

    cfg = dict(cfg_global)
    cfg.update(cfg_manifest)
    if cfg_overrides:
        cfg.update(cfg_overrides)

    CITY_NAME = M.get("city", cfg.get("CITY_NAME", "nansha"))
    MODEL_NAME = M.get("model", cfg.get("MODEL_NAME", "GeoPriorSubsNet"))

    FEATURES = cfg.get("features", {}) or {}
    DYN_NAMES = FEATURES.get("dynamic", []) or []

    CENSOR = cfg.get("censoring", {}) or cfg.get("censor", {}) or {}
    CENSOR_SPECS = CENSOR.get("specs", []) or []
    CENSOR_THRESH = float(CENSOR.get("flag_threshold", 0.5))

    # Resolve dynamic censor flag index
    CENSOR_FLAG_IDX = None
    CENSOR_FLAG_NAME = None
    for sp in CENSOR_SPECS:
        cand = sp.get("flag_col")
        if not cand:
            base = sp.get("col")
            if base:
                cand = base + sp.get("flag_suffix", "_censored")
        if cand and cand in DYN_NAMES:
            CENSOR_FLAG_NAME = cand
            CENSOR_FLAG_IDX = DYN_NAMES.index(cand)
            log(f"[Info] Censor flags present in dynamic features: {cand}")
            break

    TIME_STEPS = cfg["TIME_STEPS"]
    FORECAST_HORIZON_YEARS = cfg["FORECAST_HORIZON_YEARS"]
    FORECAST_START_YEAR = cfg["FORECAST_START_YEAR"]
    MODE = cfg["MODE"]

    ATTENTION_LEVELS = cfg.get(
        "ATTENTION_LEVELS", ["cross", "hierarchical", "memory"]
    )
    SCALE_PDE_RESIDUALS = cfg.get("SCALE_PDE_RESIDUALS", True)

    EMBED_DIM = cfg.get("EMBED_DIM", 32)
    HIDDEN_UNITS = cfg.get("HIDDEN_UNITS", 64)
    LSTM_UNITS = cfg.get("LSTM_UNITS", 64)
    ATTENTION_UNITS = cfg.get("ATTENTION_UNITS", 64)
    NUMBER_HEADS = cfg.get("NUMBER_HEADS", 2)
    DROPOUT_RATE = cfg.get("DROPOUT_RATE", 0.10)

    MEMORY_SIZE = cfg.get("MEMORY_SIZE", 50)
    SCALES = cfg.get("SCALES", [1, 2])
    USE_RESIDUALS = cfg.get("USE_RESIDUALS", True)
    USE_BATCH_NORM = cfg.get("USE_BATCH_NORM", False)
    USE_VSN = cfg.get("USE_VSN", True)
    VSN_UNITS = cfg.get("VSN_UNITS", 32)

    QUANTILES = cfg.get("QUANTILES", [0.1, 0.5, 0.9])
    SUBS_WEIGHTS = _coerce_quantile_weights(
        cfg.get("SUBS_WEIGHTS"), {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
    )
    GWL_WEIGHTS = _coerce_quantile_weights(
        cfg.get("GWL_WEIGHTS"), {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}
    )

    PDE_MODE_CONFIG = cfg.get("PDE_MODE_CONFIG", "off")
    LAMBDA_CONS = cfg.get("LAMBDA_CONS", 0.10)
    LAMBDA_GW = cfg.get("LAMBDA_GW", 0.01)
    LAMBDA_PRIOR = cfg.get("LAMBDA_PRIOR", 0.10)
    LAMBDA_SMOOTH = cfg.get("LAMBDA_SMOOTH", 0.01)
    LAMBDA_MV = cfg.get("LAMBDA_MV", 0.01)
    MV_LR_MULT = cfg.get("MV_LR_MULT", 1.0)
    KAPPA_LR_MULT = cfg.get("KAPPA_LR_MULT", 5.0)

    GEOPRIOR_INIT_MV = cfg.get("GEOPRIOR_INIT_MV", 1e-7)
    GEOPRIOR_INIT_KAPPA = cfg.get("GEOPRIOR_INIT_KAPPA", 1.0)
    GEOPRIOR_GAMMA_W = cfg.get("GEOPRIOR_GAMMA_W", 9810.0)
    GEOPRIOR_H_REF = cfg.get("GEOPRIOR_H_REF", 0.0)
    GEOPRIOR_KAPPA_MODE = cfg.get("GEOPRIOR_KAPPA_MODE", "bar")
    GEOPRIOR_USE_EFFECTIVE_H = cfg.get(
        "GEOPRIOR_USE_EFFECTIVE_H",
        CENSOR.get("use_effective_h_field", True),
    )
    GEOPRIOR_HD_FACTOR = cfg.get("GEOPRIOR_HD_FACTOR", 0.6)

    cols_cfg = cfg.get("cols", {})
    SUBSIDENCE_COL = cols_cfg.get("subsidence", "subsidence")
    GWL_COL = cols_cfg.get("gwl", "GWL")

    EPOCHS = cfg.get("EPOCHS", 50)
    BATCH_SIZE = cfg.get("BATCH_SIZE", 32)
    LEARNING_RATE = cfg.get("LEARNING_RATE", 1e-4)

    BASE_OUTPUT_DIR = M["paths"]["run_dir"]
    STAMP = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, f"train_{STAMP}")
    ensure_directory_exists(RUN_OUTPUT_PATH)

    config_sections = [
        ("Run", {
            "CITY_NAME": CITY_NAME,
            "MODEL_NAME": MODEL_NAME,
            "RESULTS_DIR": RESULTS_DIR,
            "MANIFEST_PATH": manifest_path,
            "RUN_OUTPUT_PATH": RUN_OUTPUT_PATH,
        }),
        ("Architecture", {
            "TIME_STEPS": TIME_STEPS,
            "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
            "MODE": MODE,
            "ATTENTION_LEVELS": ATTENTION_LEVELS,
            "EMBED_DIM": EMBED_DIM,
            "HIDDEN_UNITS": HIDDEN_UNITS,
            "LSTM_UNITS": LSTM_UNITS,
            "ATTENTION_UNITS": ATTENTION_UNITS,
            "NUMBER_HEADS": NUMBER_HEADS,
            "DROPOUT_RATE": DROPOUT_RATE,
            "MEMORY_SIZE": MEMORY_SIZE,
            "SCALES": SCALES,
            "USE_RESIDUALS": USE_RESIDUALS,
            "USE_BATCH_NORM": USE_BATCH_NORM,
            "USE_VSN": USE_VSN,
            "VSN_UNITS": VSN_UNITS,
        }),
        ("Physics", {
            "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
            "SCALE_PDE_RESIDUALS": SCALE_PDE_RESIDUALS,
            "LAMBDA_CONS": LAMBDA_CONS,
            "LAMBDA_GW": LAMBDA_GW,
            "LAMBDA_PRIOR": LAMBDA_PRIOR,
            "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
            "LAMBDA_MV": LAMBDA_MV,
            "MV_LR_MULT": MV_LR_MULT,
            "KAPPA_LR_MULT": KAPPA_LR_MULT,
            "GEOPRIOR_INIT_MV": GEOPRIOR_INIT_MV,
            "GEOPRIOR_INIT_KAPPA": GEOPRIOR_INIT_KAPPA,
            "GEOPRIOR_GAMMA_W": GEOPRIOR_GAMMA_W,
            "GEOPRIOR_H_REF": GEOPRIOR_H_REF,
            "GEOPRIOR_KAPPA_MODE": GEOPRIOR_KAPPA_MODE,
            "GEOPRIOR_USE_EFFECTIVE_H": GEOPRIOR_USE_EFFECTIVE_H,
            "GEOPRIOR_HD_FACTOR": GEOPRIOR_HD_FACTOR,
        }),
        ("Training", {
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "QUANTILES": QUANTILES,
            "SUBS_WEIGHTS": SUBS_WEIGHTS,
            "GWL_WEIGHTS": GWL_WEIGHTS,
        }),
    ]

    print_config_table(
        config_sections,
        table_width=get_table_size(),
        title=f"{CITY_NAME.upper()} {MODEL_NAME} TRAINING CONFIG",
        print_fn=log,
    )
    log(f"\nTraining outputs -> {RUN_OUTPUT_PATH}")

    if stop_check and stop_check():
        log("[Stop] Cancelled before dataset loading.")
        return {"run_dir": RUN_OUTPUT_PATH, "cancelled": True}

    if progress_callback is not None:
        progress_callback(0.05, "Training: loading encoders and NPZs...")
        
    # ================================================================
    # 3. Encoders / NPZ loading
    # ================================================================
    encoders = M["artifacts"]["encoders"]

    ohe_block = encoders.get("ohe")
    if isinstance(ohe_block, dict):
        log(f"[Info] {len(ohe_block)} OHE encoders recorded in manifest.")
    elif isinstance(ohe_block, str):
        log("[Info] Single OHE encoder path recorded in manifest.")
    else:
        log("[Info] No OHE encoders recorded (or not needed).")

    main_scaler = None
    ms_path = encoders.get("main_scaler")
    if ms_path and os.path.exists(ms_path):
        try:
            main_scaler = joblib.load(ms_path)
        except Exception as e:
            log(f"[Warn] Could not load main_scaler at {ms_path}: {e}")
    else:
        log("[Warn] main_scaler path missing in manifest or file not found; "
            "continuing without it.")

    coord_scaler = None
    cs_path = encoders.get("coord_scaler")
    if cs_path and os.path.exists(cs_path):
        try:
            coord_scaler = joblib.load(cs_path)
        except Exception as e:
            log(f"[Warn] Could not load coord_scaler at {cs_path}: {e}")

    scaler_info_dict = load_scaler_info(encoders)
    if isinstance(scaler_info_dict, dict):
        for k, v in scaler_info_dict.items():
            if isinstance(v, dict) and "scaler_path" in v and "scaler" not in v:
                p = v["scaler_path"]
                if p and os.path.exists(p):
                    try:
                        v["scaler"] = joblib.load(p)
                    except Exception:
                        pass

    feat_reg = cfg.get("feature_registry", {})
    if feat_reg:
        log("\n[Info] Stage-1 feature registry summary:")
        for k in (
            "resolved_optional_numeric",
            "resolved_optional_categorical",
            "already_normalized",
            "future_drivers_declared",
        ):
            if k in feat_reg:
                log(f"  - {k}: {feat_reg[k]}")

    train_inputs_npz = M["artifacts"]["numpy"]["train_inputs_npz"]
    train_targets_npz = M["artifacts"]["numpy"]["train_targets_npz"]
    val_inputs_npz = M["artifacts"]["numpy"]["val_inputs_npz"]
    val_targets_npz = M["artifacts"]["numpy"]["val_targets_npz"]
    test_inputs_npz = M["artifacts"]["numpy"].get("test_inputs_npz")
    test_targets_npz = M["artifacts"]["numpy"].get("test_targets_npz")

    X_train = dict(np.load(train_inputs_npz))
    y_train = dict(np.load(train_targets_npz))
    X_val = dict(np.load(val_inputs_npz))
    y_val = dict(np.load(val_targets_npz))
    X_test = dict(np.load(test_inputs_npz)) if test_inputs_npz else None
    y_test = dict(np.load(test_targets_npz)) if test_targets_npz else None

    OUT_S_DIM = M["artifacts"]["sequences"]["dims"]["output_subsidence_dim"]
    OUT_G_DIM = M["artifacts"]["sequences"]["dims"]["output_gwl_dim"]

    if progress_callback is not None:
        progress_callback(0.15, "Training: building tf.data datasets...")
        
    # ================================================================
    # 4. Build datasets
    # ================================================================
    train_dataset = make_tf_dataset(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        mode=MODE,
        forecast_horizon=FORECAST_HORIZON_YEARS,
    )
    val_dataset = make_tf_dataset(
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        mode=MODE,
        forecast_horizon=FORECAST_HORIZON_YEARS,
    )

    log("\nDataset sample shapes:")
    for xb, yb in train_dataset.take(1):
        for k, v in xb.items():
            log(f"  X[{k:>16}] -> {tuple(v.shape)}")
        for k, v in yb.items():
            log(f"  y[{k:>16}] -> {tuple(v.shape)}")

    if stop_check and stop_check():
        log("[Stop] Cancelled before model build.")
        return {"run_dir": RUN_OUTPUT_PATH, "cancelled": True}

    # ================================================================
    # 5. Build & compile model
    # ================================================================
    X_train_norm = ensure_input_shapes(
        X_train,
        mode=MODE,
        forecast_horizon=FORECAST_HORIZON_YEARS,
    )
    s_dim_model = X_train_norm["static_features"].shape[-1]
    d_dim_model = X_train_norm["dynamic_features"].shape[-1]
    f_dim_model = X_train_norm["future_features"].shape[-1]

    subsmodel_params = {
        "embed_dim": EMBED_DIM,
        "hidden_units": HIDDEN_UNITS,
        "lstm_units": LSTM_UNITS,
        "attention_units": ATTENTION_UNITS,
        "num_heads": NUMBER_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "max_window_size": TIME_STEPS,
        "memory_size": MEMORY_SIZE,
        "scales": SCALES,
        "multi_scale_agg": "last",
        "final_agg": "last",
        "use_residuals": USE_RESIDUALS,
        "use_batch_norm": USE_BATCH_NORM,
        "use_vsn": USE_VSN,
        "vsn_units": VSN_UNITS,
        "mode": MODE,
        "attention_levels": ATTENTION_LEVELS,
        "scale_pde_residuals": SCALE_PDE_RESIDUALS,
        "mv": LearnableMV(initial_value=GEOPRIOR_INIT_MV),
        "kappa": LearnableKappa(initial_value=GEOPRIOR_INIT_KAPPA),
        "gamma_w": FixedGammaW(value=GEOPRIOR_GAMMA_W),
        "h_ref": FixedHRef(value=GEOPRIOR_H_REF),
        "kappa_mode": GEOPRIOR_KAPPA_MODE,
        "use_effective_h": GEOPRIOR_USE_EFFECTIVE_H,
        "hd_factor": GEOPRIOR_HD_FACTOR,
    }

    subs_model_inst = GeoPriorSubsNet(
        static_input_dim=s_dim_model,
        dynamic_input_dim=d_dim_model,
        future_input_dim=f_dim_model,
        output_subsidence_dim=OUT_S_DIM,
        output_gwl_dim=OUT_G_DIM,
        forecast_horizon=FORECAST_HORIZON_YEARS,
        quantiles=QUANTILES,
        pde_mode=PDE_MODE_CONFIG,
        **subsmodel_params,
    )

    for xb, _ in train_dataset.take(1):
        subs_model_inst(xb)
        break
    subs_model_inst.summary(line_length=110, expand_nested=True)

    loss_dict = {
        "subs_pred": (
            make_weighted_pinball(QUANTILES, SUBS_WEIGHTS)
            if QUANTILES else tf.keras.losses.MSE
        ),
        "gwl_pred": (
            make_weighted_pinball(QUANTILES, GWL_WEIGHTS)
            if QUANTILES else tf.keras.losses.MSE
        ),
    }
    metrics_dict = {
        "subs_pred": ["mae", "mse"]
        + ([coverage80_fn, sharpness80_fn] if QUANTILES else []),
        "gwl_pred": ["mae", "mse"],
    }
    loss_weights_dict = {"subs_pred": 1.0, "gwl_pred": 0.5}
    physics_loss_weights = {
        "lambda_cons": LAMBDA_CONS,
        "lambda_gw": LAMBDA_GW,
        "lambda_prior": LAMBDA_PRIOR,
        "lambda_smooth": LAMBDA_SMOOTH,
        "lambda_mv": LAMBDA_MV,
        "mv_lr_mult": MV_LR_MULT,
        "kappa_lr_mult": KAPPA_LR_MULT,
    }

    subs_model_inst.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, clipnorm=1.0
        ),
        loss=loss_dict,
        metrics=metrics_dict,
        loss_weights=loss_weights_dict,
        **physics_loss_weights,
    )
    log(f"{MODEL_NAME} compiled.")

    if stop_check and stop_check():
        log("[Stop] Cancelled before training.")
        return {"run_dir": RUN_OUTPUT_PATH, "cancelled": True}

    # ================================================================
    # 6. Train
    # ================================================================
    ckpt_name = f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_HORIZON_YEARS}.keras"
    ckpt_path = os.path.join(RUN_OUTPUT_PATH, ckpt_name)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]
    csvlog_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_{MODEL_NAME}_train_log.csv"
    )
    callbacks.append(CSVLogger(csvlog_path, append=False))

    # Figure out batches per epoch once train_dataset is built
    batches_per_epoch = None
    try:
        # Either tf.data.experimental.cardinality or fusionlab._optdeps.with_progress
        batches_per_epoch = int(
            tfdata_experimental.cardinality(train_dataset).numpy()
        )
    except Exception:
        # Fallback; if unknown, you could skip GUI progress or set a dummy value
        batches_per_epoch = 1

    if stop_check is not None:
        callbacks.append(StopTrainingOnSignal(stop_check, logger=log))
    # --- attach GUI progress if a callback was provided ---
    if progress_callback is not None and batches_per_epoch > 0:
        gui_cb = GuiProgress(
            total_epochs=EPOCHS,
            batches_per_epoch=batches_per_epoch,
            update_fn=_progress_from_pct,
            epoch_level=False,  # per-batch → smoother bar
        )
        callbacks.append(gui_cb)
        fit_verbose = 0
    else:
        fit_verbose = cfg.get("FIT_VERBOSE", 1)
        
    log("\nTraining...")
    history = subs_model_inst.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=fit_verbose,
    )
    log(
        f"Best val_loss: "
        f"{min(history.history.get('val_loss', [np.inf])):.4f}"
    )
    if progress_callback is not None:
        progress_callback(0.25, "Training: building GeoPriorSubsNet model...")
    # =================================================================
    # 7. Persist weights / architecture / training summary / run manifest
    # =================================================================
    weights_path = os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_HORIZON_YEARS}.weights.h5",
    )
    arch_json_path = os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_{MODEL_NAME}_architecture.json",
    )
    summary_json_path = os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_{MODEL_NAME}_training_summary.json",
    )
    run_manifest_path = os.path.join(
        RUN_OUTPUT_PATH, f"{CITY_NAME}_{MODEL_NAME}_run_manifest.json"
    )

    try:
        subs_model_inst.save_weights(weights_path)
        log(f"[OK] Saved HDF5 weights -> {weights_path}")
    except Exception as e:
        log(f"[Warn] save_weights('{weights_path}') failed: {e}")

    try:
        with open(arch_json_path, "w", encoding="utf-8") as f:
            f.write(subs_model_inst.to_json())
    except Exception as e:
        log(f"[Warn] to_json failed: {e}")

    best_epoch, metrics_at_best = best_epoch_and_metrics(history.history)

    training_summary = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "city": CITY_NAME,
        "model": MODEL_NAME,
        "horizon": int(FORECAST_HORIZON_YEARS),
        "best_epoch": (
            int(best_epoch) if best_epoch is not None else None
        ),
        "metrics_at_best": metrics_at_best,
        "final_epoch_metrics": {
            k: float(v[-1]) for k, v in history.history.items() if len(v)
        },
        "env": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "compile": {
            "optimizer": "Adam",
            "learning_rate": float(LEARNING_RATE),
            "loss_weights": loss_weights_dict,
            "metrics": {
                k: [name_of(m) for m in v]
                for k, v in metrics_dict.items()
            },
            "physics_loss_weights": physics_loss_weights,
        },
        "hp_init": {
            "quantiles": QUANTILES,
            "subs_weights": SUBS_WEIGHTS,
            "gwl_weights": GWL_WEIGHTS,
            "attention_levels": ATTENTION_LEVELS,
            "pde_mode": PDE_MODE_CONFIG,
            "time_steps": int(TIME_STEPS),
            "use_batch_norm": bool(USE_BATCH_NORM),
            "use_vsn": bool(USE_VSN),
            "vsn_units": int(VSN_UNITS)
            if VSN_UNITS is not None else None,
            "mode": MODE,
            "model_init_params": serialize_subs_params(
                subsmodel_params, cfg
            ),
        },
        "paths": {
            "run_dir": RUN_OUTPUT_PATH,
            "checkpoint_keras": ckpt_path,
            "weights_h5": weights_path,
            "arch_json": arch_json_path,
            "csv_log": csvlog_path,
        },
    }

    final_model_path = os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_HORIZON_YEARS}_final.keras",
    )
    try:
        subs_model_inst.save(final_model_path)
        training_summary["paths"]["final_keras"] = final_model_path
        log(f"[OK] Saved Final keras model -> {final_model_path}")
    except Exception as e:
        log(
            f"[Warn] Saved Final keras model ('{final_model_path}') failed: {e}"
        )

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    run_manifest = {
        "stage": "stage-2-train",
        "city": CITY_NAME,
        "model": MODEL_NAME,
        "config": {
            "TIME_STEPS": TIME_STEPS,
            "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
            "MODE": MODE,
            "ATTENTION_LEVELS": ATTENTION_LEVELS,
            "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
            "QUANTILES": QUANTILES,
        },
        "paths": training_summary["paths"],
        "artifacts": {
            "training_summary_json": summary_json_path,
            "train_log_csv": csvlog_path,
        },
    }
    with open(run_manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    log(
        "[OK] Persisted weights, architecture JSON, CSV log, "
        "training summary, and run manifest."
    )

    if stop_check and stop_check():
        log("[Stop] Cancelled after training; skipping evaluation.")
        return {
            "run_dir": RUN_OUTPUT_PATH,
            "train_summary_json": summary_json_path,
            "run_manifest": run_manifest_path,
            "cancelled": True,
        }

    # ================================================================
    # 8. History plots + physics parameters
    # ================================================================
    history_groups = {
        "Total Loss": ["loss", "val_loss"],
        "Physics Loss": ["physics_loss", "val_physics_loss"],
        "Data Loss": ["data_loss", "val_data_loss"],
        "Component Losses": [
            "consolidation_loss",
            "val_consolidation_loss",
            "gw_flow_loss",
            "val_gw_flow_loss",
            "prior_loss",
            "val_prior_loss",
            "smooth_loss",
            "val_smooth_loss",
        ],
        "Subsidence MAE": ["subs_pred_mae", "val_subs_pred_mae"],
        "GWL MAE": ["gwl_pred_mae", "val_gwl_pred_mae"],
    }
    yscales = {
        "Total Loss": "log",
        "Physics Loss": "log",
        "Data Loss": "log",
        "Component Losses": "log",
        "Subsidence MAE": "linear",
        "GWL MAE": "linear",
    }
    try:
        plot_history_in(
            history.history,
            metrics=history_groups,
            title=f"{MODEL_NAME} Training History",
            yscale_settings=yscales,
            layout="subplots",
            savefig=os.path.join(
                RUN_OUTPUT_PATH,
                f"{CITY_NAME}_{MODEL_NAME.lower()}_training_history_plot",
            ),
        )
    except Exception as e:
        log(f"[Warn] plot_history_in failed: {e}")

    try:
        extract_physical_parameters(
            subs_model_inst,
            to_csv=True,
            filename=f"{CITY_NAME}_{MODEL_NAME.lower()}_physical_parameters.csv",
            save_dir=RUN_OUTPUT_PATH,
            model_name="geoprior",
        )
    except Exception as e:
        log(f"[Warn] extract_physical_parameters failed: {e}")

    # ================================================================
    # 9. Reload best checkpoint for inference
    # ================================================================
    custom_objects_load = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
    }
    try:
        with custom_object_scope(custom_objects_load):
            subs_model_loaded = load_model(ckpt_path, compile=False)
        log("Loaded best model for inference (compile=False).")
    except Exception as e:
        log(
            f"[Warn] Could not load best checkpoint ({e}); "
            "using in-memory model."
        )
        subs_model_loaded = subs_model_inst

    if stop_check and stop_check():
        log("[Stop] Cancelled before calibration.")
        return {
            "run_dir": RUN_OUTPUT_PATH,
            "train_summary_json": summary_json_path,
            "run_manifest": run_manifest_path,
            "cancelled": True,
        }

    # ================================================================
    # 10. Calibrate on validation set
    # ================================================================
    log("\nFitting interval calibrator (target 80%) on validation set...")
    cal80 = fit_interval_calibrator_on_val(
        subs_model_inst, val_dataset, target=0.80
    )
    np.save(
        os.path.join(RUN_OUTPUT_PATH, "interval_factors_80.npy"),
        cal80.factors_,
    )
    log("Calibrator saved.")

    # ================================================================
    # 11. Forecasting (test if available, else val)
    # ================================================================
    dataset_name_for_forecast = "ValidationSet_Fallback"
    X_fore = None
    y_fore = None
    if X_test is not None and y_test is not None:
        X_fore = X_test
        y_fore = y_test
        dataset_name_for_forecast = "TestSet"
    else:
        X_fore = X_val
        y_fore = y_val

    X_fore = ensure_input_shapes(
        X_fore, mode=MODE, forecast_horizon=FORECAST_HORIZON_YEARS
    )
    y_fore_fmt = map_targets_for_training(y_fore)

    log(f"\nPredicting on {dataset_name_for_forecast}...")
    pred_dict = subs_model_loaded.predict(X_fore, verbose=0)
    data_final = pred_dict["data_final"]
    s_dim = subs_model_loaded.output_subsidence_dim

    if QUANTILES:
        s_pred_q_raw = data_final[..., :s_dim]
        h_pred_q_raw = data_final[..., s_dim:]
        s_pred_q_cal = apply_calibrator_to_subs(cal80, s_pred_q_raw)
        predictions_for_formatter = {
            "subs_pred": s_pred_q_cal,
            "gwl_pred": h_pred_q_raw,
        }
    else:
        s_pred_raw = data_final[..., :s_dim]
        h_pred_raw = data_final[..., s_dim:]
        predictions_for_formatter = {
            "subs_pred": s_pred_raw,
            "gwl_pred": h_pred_raw,
        }

    y_true_for_format = {
        "subsidence": y_fore_fmt["subs_pred"],
        "gwl": y_fore_fmt["gwl_pred"],
    }

    csv_eval = os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_{MODEL_NAME}_forecast_"
        f"{dataset_name_for_forecast}_H{FORECAST_HORIZON_YEARS}_calibrated.csv",
    )
    csv_future = os.path.join(
        RUN_OUTPUT_PATH,
        f"{CITY_NAME}_{MODEL_NAME}_forecast_"
        f"{dataset_name_for_forecast}_H{FORECAST_HORIZON_YEARS}_future.csv",
    )

    future_grid = np.arange(
        FORECAST_START_YEAR,
        FORECAST_START_YEAR + FORECAST_HORIZON_YEARS,
        dtype=float,
    )

    df_eval, df_future = format_and_forecast(
        y_pred=predictions_for_formatter,
        y_true=y_true_for_format,
        coords=X_fore.get("coords", None),
        quantiles=QUANTILES if QUANTILES else None,
        target_name=SUBSIDENCE_COL,
        target_key_pred="subs_pred",
        component_index=0,
        scaler_info=scaler_info_dict,
        coord_scaler=coord_scaler,
        coord_columns=("coord_t", "coord_x", "coord_y"),
        train_end_time=cfg.get("TRAIN_END_YEAR"),
        forecast_start_time=FORECAST_START_YEAR,
        forecast_horizon=FORECAST_HORIZON_YEARS,
        future_time_grid=future_grid,
        eval_forecast_step=None,
        sample_index_offset=0,
        city_name=CITY_NAME,
        model_name=MODEL_NAME,
        dataset_name=dataset_name_for_forecast,
        csv_eval_path=csv_eval,
        csv_future_path=csv_future,
        time_as_datetime=False,
        time_format=None,
        verbose=7,
        eval_metrics=True,
        metrics_column_map=None,
        metrics_quantile_interval=(0.1, 0.9),
        metrics_per_horizon=True,
        metrics_extra=["pss"],
        metrics_extra_kwargs=None,
        metrics_savefile=os.path.join(
            RUN_OUTPUT_PATH, "eval_diagnostics.json"
        ),
        metrics_save_format=".json",
        metrics_time_as_str=True,
        value_mode="rate",
        logger = log
    )

    if df_eval is not None and not df_eval.empty:
        log(f"Saved calibrated EVAL forecast CSV -> {csv_eval}")
    else:
        log("[Warn] Empty eval forecast DF.")

    if df_future is not None and not df_future.empty:
        log(f"Saved calibrated FUTURE forecast CSV -> {csv_future}")
    else:
        log("[Warn] Empty future forecast DF.")

    # =============================================================================
    # Evaluate metrics & physics on the forecasting split (+ optional censoring)
    # =============================================================================
    metrics_json_out = None

    if stop_check and stop_check():
        log("[Stop] Cancelled after forecasting; skipping metrics & plots.")
    else:
        eval_results = {}
        phys = {}

        ds_eval = tf.data.Dataset.from_tensor_slices(
            (X_fore, y_fore_fmt)
        ).batch(BATCH_SIZE)

        # --- 2.1 Standard Keras evaluate() + physics metrics ---
        try:
            eval_results = subs_model_inst.evaluate(
                ds_eval, return_dict=True, verbose=1
            )
            log(f"Evaluation: {eval_results}")

            # Physics diagnostics are already aggregated in eval_results
            phys_keys = ("epsilon_prior", "epsilon_cons")
            phys = {
                k: float(_to_py(eval_results[k]))
                for k in phys_keys
                if k in eval_results
            }
            if phys:
                log(f"Physics diagnostics (from evaluate): {phys}")

        except Exception as e:
            log(f"[Warn] Evaluation failed (metrics + physics): {e}")
            eval_results, phys = {}, {}

        #  2.2. Save physic payload (VAL split)
        try:
            subs_model_loaded.export_physics_payload(
                val_dataset,
                max_batches=None,
                save_path=os.path.join(
                    RUN_OUTPUT_PATH, f"{CITY_NAME}_phys_payload_run_val.npz"
                ),
                format="npz",
                overwrite=True,
                metadata={"city": CITY_NAME, "split": "val"},
            )
        except Exception as e:
            log(f"[Warn] export_physics_payload failed: {e}")

        # --- 2.3 Interval diagnostics + optional censor-stratified MAE ---
        cov80_uncal = cov80_cal = None
        sharp80_uncal = sharp80_cal = None
        censor_metrics = None   # will become a dict if we have a flag

        y_true_list, s_q_list, mask_list = [], [], []

        for xb, yb in with_progress(
            ds_eval, desc="Interval-Censoring Diagnostics"
        ):
            out = subs_model_inst(xb, training=False)
            data_final_b = out["data_final"]

            y_true_b = yb["subs_pred"]           # (B, H, 1)
            y_true_list.append(y_true_b)

            if QUANTILES:
                s_q_b, _ = subs_model_inst.split_data_predictions(
                    data_final_b
                )  # (B, H, Q, 1)
                s_q_list.append(s_q_b)

            if CENSOR_FLAG_IDX is not None:
                H = tf.shape(y_true_b)[1]
                mask_b = build_censor_mask_from_dynamic(  # (B, H, 1)
                    xb, H, CENSOR_FLAG_IDX, CENSOR_THRESH
                )
                mask_list.append(mask_b)

        # Stack what we collected
        y_true = tf.concat(y_true_list, axis=0) if y_true_list else None  # (N,H,1)
        s_q = tf.concat(s_q_list, axis=0) if s_q_list else None           # (N,H,Q,1)
        mask = tf.concat(mask_list, axis=0) if mask_list else None        # (N,H,1)

        # --- 2.3.a Interval coverage/sharpness (scaled + physical) ---------------
        cov80_uncal_phys = cov80_cal_phys = None
        sharp80_uncal_phys = sharp80_cal_phys = None
        s_q_cal = None

        if QUANTILES and (y_true is not None) and (s_q is not None):
            # ---------- SCALED metrics ----------
            cov80_uncal = float(coverage80_fn(y_true, s_q).numpy())
            sharp80_uncal = float(sharpness80_fn(y_true, s_q).numpy())

            # Calibrated (apply same calibrator to the whole tensor)
            s_q_cal = apply_calibrator_to_subs(cal80, s_q)  # (N,H,Q,1)
            cov80_cal = float(coverage80_fn(y_true, s_q_cal).numpy())
            sharp80_cal = float(sharpness80_fn(y_true, s_q_cal).numpy())

            # ---------- PHYSICAL metrics ----------
            # inverse-transform y_true and quantiles to physical units
            y_true_phys_np = inverse_scale_target(
                y_true,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )
            s_q_phys_np = inverse_scale_target(
                s_q,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )

            y_true_phys_tf = tf.convert_to_tensor(
                y_true_phys_np, dtype=tf.float32
            )
            s_q_phys_tf = tf.convert_to_tensor(
                s_q_phys_np, dtype=tf.float32
            )

            cov80_uncal_phys = float(
                coverage80_fn(y_true_phys_tf, s_q_phys_tf).numpy()
            )
            sharp80_uncal_phys = float(
                sharpness80_fn(y_true_phys_tf, s_q_phys_tf).numpy()
            )

            if s_q_cal is not None:
                s_q_cal_phys_np = inverse_scale_target(
                    s_q_cal,
                    scaler_info=scaler_info_dict,
                    target_name=SUBSIDENCE_COL,
                )
                s_q_cal_phys_tf = tf.convert_to_tensor(
                    s_q_cal_phys_np, dtype=tf.float32
                )

                cov80_cal_phys = float(
                    coverage80_fn(
                        y_true_phys_tf, s_q_cal_phys_tf
                    ).numpy()
                )
                sharp80_cal_phys = float(
                    sharpness80_fn(
                        y_true_phys_tf, s_q_cal_phys_tf
                    ).numpy()
                )

        # --- 2.3.b Optional censor-stratified MAE (physical units) ---
        s_med = None  # ensure defined
        if (y_true is not None) and (mask is not None):
            if QUANTILES and (s_q is not None):
                med_idx = int(
                    np.argmin(np.abs(np.asarray(QUANTILES) - 0.5))
                )
                s_med = s_q[..., med_idx, :]  # (N, H, 1) scaled
            else:
                # point-forecast: take subsidence head from this pass
                s_pred_list = []
                for xb2, _ in ds_eval:
                    out2 = subs_model_inst(xb2, training=False)
                    s_pred_list.append(out2["data_final"][..., :1])  # (B,H,1)
                s_med = tf.concat(s_pred_list, axis=0)              # scaled

            # Convert both y_true and s_med to physical units
            y_true_phys_np = inverse_scale_target(
                y_true,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )
            s_med_phys_np = inverse_scale_target(
                s_med,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )

            y_true_phys = tf.convert_to_tensor(
                y_true_phys_np, dtype=tf.float32
            )
            s_med_phys = tf.convert_to_tensor(
                s_med_phys_np, dtype=tf.float32
            )

            mask_f = tf.cast(mask, tf.float32)  # (N, H, 1)
            num_cens = tf.reduce_sum(mask_f) + 1e-8
            num_unc = tf.reduce_sum(1.0 - mask_f) + 1e-8

            abs_err = tf.abs(y_true_phys - s_med_phys)
            mae_cens = tf.reduce_sum(abs_err * mask_f) / num_cens
            mae_unc = tf.reduce_sum(abs_err * (1.0 - mask_f)) / num_unc

            censor_metrics = {
                "flag_name": CENSOR_FLAG_NAME,
                "threshold": float(CENSOR_THRESH),
                "mae_censored": float(mae_cens.numpy()),
                "mae_uncensored": float(mae_unc.numpy()),
            }

            log(
                "[CENSOR] MAE censored="
                f"{censor_metrics['mae_censored']:.4f} | "
                f"uncensored={censor_metrics['mae_uncensored']:.4f}"
            )

        # --- 2.4 Point metrics (MAE/MSE/R²), overall + per-horizon ---
        metrics_point = {}
        per_h_mae_dict, per_h_r2_dict = None, None

        if y_true is not None:
            if QUANTILES and (s_q is not None):
                # median index
                med_idx = int(
                    np.argmin(np.abs(np.asarray(QUANTILES) - 0.5))
                )
                s_med_uncal = s_q[..., med_idx, :]  # (N,H,1)

                # Prefer calibrated median if available
                if s_q_cal is not None:
                    s_med_cal = s_q_cal[..., med_idx, :]  # (N,H,1)
                    metrics_point = point_metrics(
                        y_true,
                        s_med_cal,
                        use_physical=True,
                        scaler_info=scaler_info_dict,
                        target_name=SUBSIDENCE_COL,
                    )
                    (
                        per_h_mae_dict,
                        per_h_r2_dict,
                    ) = per_horizon_metrics(
                        y_true,
                        s_med_cal,
                        use_physical=True,
                        scaler_info=scaler_info_dict,
                        target_name=SUBSIDENCE_COL,
                    )
                else:
                    metrics_point = point_metrics(
                        y_true,
                        s_med_uncal,
                        use_physical=True,
                        scaler_info=scaler_info_dict,
                        target_name=SUBSIDENCE_COL,
                    )
                    (
                        per_h_mae_dict,
                        per_h_r2_dict,
                    ) = per_horizon_metrics(
                        y_true,
                        s_med_uncal,
                        use_physical=True,
                        scaler_info=scaler_info_dict,
                        target_name=SUBSIDENCE_COL,
                    )

            else:
                # point-forecast branch
                if s_med is None:
                    s_pred_list = []
                    for xb2, _ in with_progress(
                        ds_eval, desc="Point-forecast Diagnostics"
                    ):
                        out2 = subs_model_inst(xb2, training=False)
                        s_pred_list.append(out2["data_final"][..., :1])
                    s_med = tf.concat(s_pred_list, axis=0)

                metrics_point = point_metrics(
                    y_true,
                    s_med,
                    use_physical=True,
                    scaler_info=scaler_info_dict,
                    target_name=SUBSIDENCE_COL,
                )
                (
                    per_h_mae_dict,
                    per_h_r2_dict,
                ) = per_horizon_metrics(
                    y_true,
                    s_med,
                    use_physical=True,
                    scaler_info=scaler_info_dict,
                    target_name=SUBSIDENCE_COL,
                )

        # Normalize coverage/sharpness choices for ablation record
        coverage80_for_abl = (
            cov80_cal_phys
            if ("cov80_cal_phys" in locals() and cov80_cal_phys is not None)
            else cov80_cal
            if cov80_cal is not None
            else cov80_uncal_phys
            if cov80_uncal_phys is not None
            else cov80_uncal
        )

        sharpness80_for_abl = (
            sharp80_cal_phys
            if ("sharp80_cal_phys" in locals() and sharp80_cal_phys is not None)
            else sharp80_uncal_phys
            if sharp80_uncal_phys is not None
            else sharp80_cal
            if sharp80_cal is not None
            else sharp80_uncal
        )

        # Save summary JSON
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        payload = {
            "timestamp": stamp,
            "tf_version": tf.__version__,
            "numpy_version": np.__version__,
            "quantiles": QUANTILES,
            "horizon": FORECAST_HORIZON_YEARS,
            "batch_size": BATCH_SIZE,
            "metrics_evaluate": {
                k: _to_py(v) for k, v in (eval_results or {}).items()
            },
            "physics_diagnostics": phys,
        }

        if QUANTILES:
            payload["interval_calibration"] = {
                "target": 0.80,
                "factors_per_horizon": getattr(cal80, "factors_", None).tolist()
                if hasattr(cal80, "factors_")
                else None,
                # scaled-space metrics
                "coverage80_uncalibrated": cov80_uncal,
                "coverage80_calibrated": cov80_cal,
                "sharpness80_uncalibrated": sharp80_uncal,
                "sharpness80_calibrated": sharp80_cal,
                # physical-space metrics
                "coverage80_uncalibrated_phys": cov80_uncal_phys,
                "coverage80_calibrated_phys": cov80_cal_phys,
                "sharpness80_uncalibrated_phys": sharp80_uncal_phys,
                "sharpness80_calibrated_phys": sharp80_cal_phys,
            }

        if censor_metrics is not None:
            payload["censor_stratified"] = censor_metrics

        # Attach point metrics & per-horizon into payload
        if metrics_point:
            payload["point_metrics"] = {
                "mae": metrics_point.get("mae"),
                "mse": metrics_point.get("mse"),
                "r2": metrics_point.get("r2"),
            }
        if per_h_mae_dict:
            payload.setdefault("per_horizon", {})
            payload["per_horizon"]["mae"] = per_h_mae_dict
        if per_h_r2_dict:
            payload.setdefault("per_horizon", {})
            payload["per_horizon"]["r2"] = per_h_r2_dict

        json_out = os.path.join(
            RUN_OUTPUT_PATH, f"geoprior_eval_phys_{stamp}.json"
        )
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log(f"Saved metrics + physics JSON -> {json_out}")
        metrics_json_out = json_out

        # ---- Ablation record ---------------------------------------------------
        ABLCFG = {
            "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
            "GEOPRIOR_USE_EFFECTIVE_H": GEOPRIOR_USE_EFFECTIVE_H,
            "GEOPRIOR_KAPPA_MODE": GEOPRIOR_KAPPA_MODE,
            "GEOPRIOR_HD_FACTOR": GEOPRIOR_HD_FACTOR,
            "LAMBDA_CONS": LAMBDA_CONS,
            "LAMBDA_GW": LAMBDA_GW,
            "LAMBDA_PRIOR": LAMBDA_PRIOR,
            "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
            "LAMBDA_MV": LAMBDA_MV,
        }

        # Prefer MAE/MSE directly from evaluate() if they exist; else use computed.
        eval_mae = None
        eval_mse = None
        if isinstance(eval_results, dict):
            eval_mae = eval_results.get("subs_pred_mae")
            eval_mse = eval_results.get("subs_pred_mse")

        save_ablation_record(
            outdir=RUN_OUTPUT_PATH,
            city=CITY_NAME,
            model_name=MODEL_NAME,
            cfg=ABLCFG,
            eval_dict={
                "r2": (metrics_point or {}).get("r2"),
                "mse": float(eval_mse)
                if eval_mse is not None
                else (metrics_point or {}).get("mse"),
                "mae": float(eval_mae)
                if eval_mae is not None
                else (metrics_point or {}).get("mae"),
                "coverage80": coverage80_for_abl,
                "sharpness80": sharpness80_for_abl,
            },
            phys_diag=(phys or {}),
            per_h_mae=per_h_mae_dict,
            per_h_r2=per_h_r2_dict,
        )
        log("Ablation record saved.")

        # =============================================================================
        # Visualization (optional)
        # =============================================================================
        log("\nPlotting forecast views...")
        try:
            plot_eval_future(
                df_eval=df_eval,
                df_future=df_future,
                target_name=SUBSIDENCE_COL,
                quantiles=QUANTILES,
                spatial_cols=("coord_x", "coord_y"),
                time_col="coord_t",
                # Eval: show last eval year (e.g. 2022)
                eval_years=[FORECAST_START_YEAR - 1],
                # Future: use the same grid you passed to format_and_forecast
                future_years=future_grid,
                # For eval: compare [actual] vs [q50] only
                eval_view_quantiles=[0.5],
                # For future: show full [q10, q50, q90]
                future_view_quantiles=QUANTILES,
                spatial_mode="hexbin",      # hotspot view
                hexbin_gridsize=40,
                savefig_prefix=os.path.join(
                    RUN_OUTPUT_PATH,
                    f"{CITY_NAME}_subsidence_view",
                ),
                save_fmts=[".png", ".pdf"],
                show=False,
                verbose=1,
                cummulative=True,
                _logger=log
            )
            # notify the GUI that the PNGs were created
            _notify_gui_forecast_views(RUN_OUTPUT_PATH, CITY_NAME)
            
        except Exception as e:
            log(f"[Warn] plot_eval_future failed: {e}")

        try:
            save_all_figures(
                output_dir=RUN_OUTPUT_PATH,
                prefix=f"{CITY_NAME}_{MODEL_NAME}_plot_",
                fmts=[".png", ".pdf"],
            )
            log(f"Saved all open Matplotlib figures in: {RUN_OUTPUT_PATH}")
        except Exception as e:
            log(f"[Warn] save_all_figures failed: {e}")

    # -------------------------------------------------------------------------
    # Final log + return
    # -------------------------------------------------------------------------
    log(
        f"\n---- {CITY_NAME.upper()} {MODEL_NAME} TRAINING COMPLETE ----\n"
        f"Artifacts -> {RUN_OUTPUT_PATH}\n"
    )
    # Keras fit drives 0.25 → ~0.90 via GuiProgress
    # After evaluation / calibration:
    if progress_callback is not None:
        progress_callback(1.0, "Training: complete (eval & diagnostics done).")
        
    return {
        "run_dir": RUN_OUTPUT_PATH,
        "train_summary_json": summary_json_path,
        "run_manifest": run_manifest_path,
        "eval_csv": csv_eval,
        "future_csv": csv_future,
        "metrics_json": metrics_json_out,
    }
