# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
run_training

Stage-2 helper for GeoPrior GUI.

Wraps the Stage-2 training script logic into a callable function:
- load Stage-1 manifest
- resolve hybrid config (live config + manifest snapshot)
- build/train model
- save bundles + init manifest
- load inference model
- calibrate, forecast, evaluate, export physics payload
- return artifact paths for the GUI
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import datetime as dt
import gc
import json
import os
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint,
    TerminateOnNaN,
)

from ....._optdeps import with_progress
from .....api.util import get_table_size
from .....backends.devices import configure_tf_from_cfg
from .....compat.keras import (
    load_inference_model,
    load_model_from_tfv2,
    save_manifest,
    save_model,
)
from .....nn._shapes import (
    _logs_to_py,
    canonicalize_BHQO_quantiles_np,
    debug_quantile_crossing_np,
    debug_tensor_interval,
    debug_val_interval,
)
from .....nn.calibration import (
    apply_calibrator_to_subs,
    fit_interval_calibrator_on_val,
)
from .....nn.callbacks import LambdaOffsetScheduler
from .....nn.keras_metrics import (
    Coverage80,
    MAEQ50,
    MSEQ50,
    Sharpness80,
    _to_py,
    coverage80_fn,
    sharpness80_fn,
)
from .....nn.losses import make_weighted_pinball
from .....nn.pinn.geoprior.debugs import debug_model_reload
from .....nn.pinn.geoprior.models import GeoPriorSubsNet, PoroElasticSubsNet
from .....nn.pinn.geoprior.payloads import load_physics_payload
from .....nn.pinn.geoprior.plot import autoplot_geoprior_history
from .....nn.pinn.geoprior.plot import plot_physics_values_in
from .....nn.pinn.geoprior.scaling import override_scaling_kwargs
from .....nn.pinn.geoprior.utils import finalize_scaling_kwargs
from .....nn.pinn.op import extract_physical_parameters
from .....nn.utils import plot_history_in
from .....params import FixedGammaW, FixedHRef, LearnableKappa, LearnableMV
from .....plot.forecast import plot_eval_future
from .....utils.audit_utils import audit_stage2_handshake, should_audit
from .....utils.forecast_utils import format_and_forecast
from .....utils.generic_utils import (
    ensure_directory_exists,
    print_config_table,
    save_all_figures,
)
from .....utils.nat_utils import (
    best_epoch_and_metrics,
    build_censor_mask,
    ensure_input_shapes,
    extract_preds,
    load_nat_config,
    load_scaler_info,
    make_tf_dataset,
    map_targets_for_training,
    name_of,
    resolve_hybrid_config,
    resolve_si_affine,
    save_ablation_record,
    serialize_subs_params,
    subs_point_from_out,
)
from .....utils.scale_metrics import (
    inverse_scale_target,
    per_horizon_metrics,
    point_metrics,
)
from .....utils.spatial_utils import deg_to_m_from_lat
from .....utils.subsidence_utils import convert_eval_payload_units

from ..utils.view_utils import _notify_gui_forecast_views
from ..callbacks import ( 
    GuiProgress, 
    StopTrainingOnSignal,
    GuiEarlyStopping,
    GuiEpochLogger, 
    GuiMetricLogger
)

# Silence TF chatter
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")
if hasattr(tf, "autograph") and hasattr(tf.autograph, "set_verbosity"):
    tf.autograph.set_verbosity(0)


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


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _pick_scaler_key(
    scaler_info: dict,
    preferred: str,
    fallbacks: tuple[str, ...] = (),
) -> str:
    if not scaler_info:
        return preferred
    if preferred in scaler_info:
        return preferred
    for k in fallbacks:
        if k in scaler_info:
            return k
    low = {k.lower(): k for k in scaler_info.keys()}
    for token in ("subs", "subsidence"):
        for lk, orig in low.items():
            if token in lk:
                return orig
    return preferred


class _StopCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        stop_check: Optional[Callable[[], bool]],
        log_fn: Callable[[str], None],
    ) -> None:
        super().__init__()
        self._stop_check = stop_check
        self._log = log_fn

    def on_train_batch_end(self, batch, logs=None):
        if self._stop_check and self._stop_check():
            self._log("[GUI] Stop requested. Stopping training...")
            self.model.stop_training = True


class _ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        steps_per_epoch: int,
        epochs: int,
        on_pct: Callable[[int], None],
    ) -> None:
        super().__init__()
        self._spe = max(1, int(steps_per_epoch))
        self._epochs = max(1, int(epochs))
        self._on_pct = on_pct
        self._seen = 0
        self._total = self._spe * self._epochs

    def on_train_batch_end(self, batch, logs=None):
        self._seen += 1
        pct = int(round(100.0 * self._seen / float(self._total)))
        pct = max(0, min(100, pct))
        self._on_pct(pct)


def run_training(
    manifest_path: Optional[str] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    base_cfg: Optional[Dict[str, Any]] = None,
    results_root: Optional[os.PathLike | str] = None,
    evaluate_training: bool = True,
    **kws,
) -> Dict[str, Any]:
    """
    Run Stage-2 training & evaluation for GeoPriorSubsNet (GUI).
    """
    # # Fractions of the *global* pipeline reserved for training
    TRAINING_FRACTION_START = 0.25
    TRAINING_FRACTION_END = 0.80


    # # ------------------------------------------------------------------------
    # # Wall-clock anchor for ETA in the training phase
    # train_start_t = None

    # def _format_eta(seconds: float) -> str:
    #     """Return an ETA string in HH:MM:SS or MM:SS, similar to ProgressManager."""
    #     if seconds is None or seconds < 0 or seconds == float("inf"):
    #         return "--:--"
    #     m, s = divmod(int(seconds), 60)
    #     h, m = divmod(m, 60)
    #     return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    train_start_t: float | None = None


    def _format_eta(seconds: float | None) -> str:
        if seconds is None:
            return "--:--"
        if seconds < 0 or seconds == float("inf"):
            return "--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"


    def _progress_from_pct(pct: int) -> None:
        """
        Bridge Keras/GuiProgress → GUI progress bar, with epoch + ETA info.

        Parameters
        ----------
        pct : int
            Training completion percentage [0, 100] reported by GuiProgress.
        """
        nonlocal train_start_t

        if progress_callback is None:
            return

        # --- clamp pct and map to local training fraction [0, 1] -------
        try:
            p = float(pct)
        except Exception:
            p = 0.0
        p = max(0.0, min(100.0, p))
        frac_local = p / 100.0  # 0..1 *within training*

        # --- timing for ETA is based on *local* training progress -------
        now = time.time()
        if train_start_t is None and frac_local > 0.0:
            train_start_t = now

        if train_start_t is not None and frac_local > 1e-6:
            elapsed = max(0.0, now - train_start_t)
            remaining = elapsed * (1.0 - frac_local) / frac_local
            eta_str = _format_eta(remaining)
        else:
            eta_str = "--:--"

        # --- best-effort epoch index from local fraction ----------------
        epoch_part = ""
        try:
            total_epochs = int(EPOCHS)
        except Exception:
            total_epochs = 0

        if total_epochs > 0 and frac_local > 0.0:
            ep = int(round(frac_local * total_epochs))
            ep = max(1, min(total_epochs, ep))
            epoch_part = f"Epoch {ep}/{total_epochs} – "

        # --- embed local training progress into global run fraction -----
        global_frac = (
            TRAINING_FRACTION_START
            + (TRAINING_FRACTION_END - TRAINING_FRACTION_START) * frac_local
        )

        msg = f"Training – {epoch_part}ETA: {eta_str}"

        try:
            progress_callback(global_frac, msg)
        except Exception:
            pass

    # -----------------------------------------------------------

    log = logger or (lambda msg: print(msg, flush=True))

    if manifest_path is None:
        raise ValueError(
            "run_training requires 'manifest_path' in the GUI."
        )

    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        M = json.load(f)

    if progress_callback:
        progress_callback(0.02, "Stage2: loaded Stage-1 manifest.")

    if results_root is not None:
        results_dir = os.fspath(results_root)
    else:
        run_dir = (M.get("paths") or {}).get("run_dir")
        if run_dir:
            results_dir = os.path.abspath(os.path.dirname(run_dir))
        else:
            results_dir = os.path.abspath(
                os.path.dirname(os.path.dirname(manifest_path))
            )

    # Load live config
    if base_cfg is not None:
        cfg_global = dict(base_cfg)
    else:
        gui_dir = os.path.dirname(__file__)
        config_root = os.path.join(
            os.path.dirname(gui_dir),
            "config",
        )
        cfg_global = load_nat_config(root=config_root)

    cfg_manifest = M.get("config", {}) or {}

    log("\n[Config] Resolving hybrid config (v3.2)...")
    cfg = resolve_hybrid_config(
        manifest_cfg=cfg_manifest,
        live_cfg=cfg_global,
        verbose=False,
    )

    if cfg_overrides:
        cfg = _deep_update(cfg, dict(cfg_overrides))

    device_info = configure_tf_from_cfg(cfg, logger=log)

    CITY_NAME = M.get("city", cfg.get("CITY_NAME", "nansha"))
    MODEL_ENV = (cfg_overrides or {}).get("MODEL_NAME_OVERRIDE")
    MODEL_NAME = MODEL_ENV or cfg.get("MODEL_NAME", "GeoPriorSubsNet")

    DEBUG = bool(cfg.get("DEBUG", False))
    AUDIT_STAGES = cfg.get("AUDIT_STAGES")

    USE_IN_MEMORY_MODEL = bool(cfg.get("USE_IN_MEMORY_MODEL", False))
    USE_TF_SAVEDMODEL = bool(cfg.get("USE_TF_SAVEDMODEL", False))

    FEATURES = cfg.get("features", {}) or {}
    DYN_NAMES = FEATURES.get("dynamic", []) or []
    FUT_NAMES = FEATURES.get("future", []) or []
    STA_NAMES = FEATURES.get("static", []) or []

    # ---- censor flags: FUT preferred, else DYN
    CENSOR = cfg.get("censoring", {}) or cfg.get("censor", {}) or {}
    CENSOR_SPECS = CENSOR.get("specs", []) or []
    CENSOR_THRESH = float(CENSOR.get("flag_threshold", 0.5))

    CENSOR_FLAG_IDX_DYN = None
    CENSOR_FLAG_IDX_FUT = None
    CENSOR_FLAG_NAME = None

    for sp in CENSOR_SPECS:
        cand = sp.get("flag_col")
        if not cand:
            base = sp.get("col")
            if base:
                cand = base + sp.get("flag_suffix", "_censored")

        if cand:
            if cand in FUT_NAMES and CENSOR_FLAG_IDX_FUT is None:
                CENSOR_FLAG_IDX_FUT = FUT_NAMES.index(cand)
                CENSOR_FLAG_NAME = cand
            if cand in DYN_NAMES and CENSOR_FLAG_IDX_DYN is None:
                CENSOR_FLAG_IDX_DYN = DYN_NAMES.index(cand)
                CENSOR_FLAG_NAME = cand

    if CENSOR_FLAG_IDX_FUT is not None:
        CENSOR_MASK_SOURCE = "future"
        CENSOR_FLAG_IDX = CENSOR_FLAG_IDX_FUT
    elif CENSOR_FLAG_IDX_DYN is not None:
        CENSOR_MASK_SOURCE = "dynamic"
        CENSOR_FLAG_IDX = CENSOR_FLAG_IDX_DYN
    else:
        CENSOR_MASK_SOURCE = None
        CENSOR_FLAG_IDX = None

    log(
        f"[Info] Censor mask source={CENSOR_MASK_SOURCE} "
        f"flag={CENSOR_FLAG_NAME} idx={CENSOR_FLAG_IDX}"
    )

    TIME_STEPS = int(cfg["TIME_STEPS"])
    FORECAST_HORIZON_YEARS = int(cfg["FORECAST_HORIZON_YEARS"])
    FORECAST_START_YEAR = int(cfg["FORECAST_START_YEAR"])
    MODE = str(cfg["MODE"])

    # Prob outputs
    QUANTILES = cfg.get("QUANTILES", [0.1, 0.5, 0.9])
    SUBS_WEIGHTS = _coerce_quantile_weights(
        cfg.get("SUBS_WEIGHTS"),
        {0.1: 3.0, 0.5: 1.0, 0.9: 3.0},
    )
    GWL_WEIGHTS = _coerce_quantile_weights(
        cfg.get("GWL_WEIGHTS"),
        {0.1: 1.5, 0.5: 1.0, 0.9: 1.5},
    )

    # Training knobs
    EPOCHS = int(cfg.get("EPOCHS", 50))
    BATCH_SIZE = int(cfg.get("BATCH_SIZE", 32))
    LEARNING_RATE = float(cfg.get("LEARNING_RATE", 1e-4))

    TRACK_AUX_METRICS = bool(
        cfg.get("TRACK_AUX_METRICS", cfg.get("TRACK_ADD_ON_METRICS", True))
    )

    # Physics knobs
    PDE_MODE_CONFIG = str(cfg.get("PDE_MODE_CONFIG", "off") or "off")
    if PDE_MODE_CONFIG in ("off", "none"):
        PDE_MODE_CONFIG = "none"

    SCALE_PDE_RESIDUALS = bool(cfg.get("SCALE_PDE_RESIDUALS", True))
    TIME_UNITS = str(cfg.get("TIME_UNITS", "year") or "year").strip().lower()

    LAMBDA_CONS = float(cfg.get("LAMBDA_CONS", 0.10))
    LAMBDA_GW = float(cfg.get("LAMBDA_GW", 0.01))
    LAMBDA_PRIOR = float(cfg.get("LAMBDA_PRIOR", 0.10))
    LAMBDA_SMOOTH = float(cfg.get("LAMBDA_SMOOTH", 0.01))
    LAMBDA_BOUNDS = float(cfg.get("LAMBDA_BOUNDS", 0.0))
    PHYSICS_BOUNDS_CFG = cfg.get("PHYSICS_BOUNDS", {}) or {}

    PHYSICS_BOUNDS_MODE = str(
        cfg.get("PHYSICS_BOUNDS_MODE", "soft") or "soft"
    ).strip().lower()
    
    _default_phys_bounds = {
        "H_min": 5.0,
        "H_max": 80.0,
        "K_min": 1e-8,
        "K_max": 1e-3,
        "Ss_min": 1e-7,
        "Ss_max": 1e-3,
        "tau_min": 7.0 * 86400.0,
        "tau_max": 300.0 * 31556952.0,
        "tau_min_units": 0.05,
        "tau_max_units": 300.0,
    }
    
    phys_bounds = dict(_default_phys_bounds)
    phys_bounds.update(PHYSICS_BOUNDS_CFG)
    
    bounds_for_scaling = {
        "H_min": float(phys_bounds["H_min"]),
        "H_max": float(phys_bounds["H_max"]),
        "K_min": float(phys_bounds["K_min"]),
        "K_max": float(phys_bounds["K_max"]),
        "Ss_min": float(phys_bounds["Ss_min"]),
        "Ss_max": float(phys_bounds["Ss_max"]),
        "tau_min": float(phys_bounds["tau_min"]),
        "tau_max": float(phys_bounds["tau_max"]),
        "logK_min": float(np.log(phys_bounds["K_min"])),
        "logK_max": float(np.log(phys_bounds["K_max"])),
        "logSs_min": float(np.log(phys_bounds["Ss_min"])),
        "logSs_max": float(np.log(phys_bounds["Ss_max"])),
        "logTau_min": float(np.log(phys_bounds["tau_min"])),
        "logTau_max": float(np.log(phys_bounds["tau_max"])),
    }

    LAMBDA_MV = float(cfg.get("LAMBDA_MV", 0.01))
    MV_LR_MULT = float(cfg.get("MV_LR_MULT", 1.0))
    KAPPA_LR_MULT = float(cfg.get("KAPPA_LR_MULT", 5.0))

    OFFSET_MODE = str(cfg.get("OFFSET_MODE", "mul") or "mul")
    LAMBDA_OFFSET = float(cfg.get("LAMBDA_OFFSET", 1.0))

    LOSS_WEIGHT_GWL = float(cfg.get("LOSS_WEIGHT_GWL", 0.5))
    LAMBDA_Q = float(cfg.get("LAMBDA_Q", 0.0))
    LOG_Q_DIAGNOSTICS = bool(cfg.get("LOG_Q_DIAGNOSTICS", False))

    # Lambda offset scheduler (optional)
    USE_LAMBDA_OFFSET_SCHEDULER = bool(
        cfg.get("USE_LAMBDA_OFFSET_SCHEDULER", False)
    )
    LAMBDA_OFFSET_UNIT = cfg.get("LAMBDA_OFFSET_UNIT", "epoch")
    LAMBDA_OFFSET_WHEN = cfg.get("LAMBDA_OFFSET_WHEN", "begin")
    LAMBDA_OFFSET_WARMUP = int(cfg.get("LAMBDA_OFFSET_WARMUP", 10))
    LAMBDA_OFFSET_START = cfg.get("LAMBDA_OFFSET_START", None)
    LAMBDA_OFFSET_END = cfg.get("LAMBDA_OFFSET_END", None)
    LAMBDA_OFFSET_SCHEDULE = cfg.get("LAMBDA_OFFSET_SCHEDULE", None)

    # GeoPrior scalars
    GEOPRIOR_INIT_MV = float(cfg.get("GEOPRIOR_INIT_MV", 1e-7))
    GEOPRIOR_INIT_KAPPA = float(cfg.get("GEOPRIOR_INIT_KAPPA", 1.0))
    GEOPRIOR_GAMMA_W = float(cfg.get("GEOPRIOR_GAMMA_W", 9810.0))
    GEOPRIOR_H_REF = cfg.get("GEOPRIOR_H_REF", 0.0)
    GEOPRIOR_KAPPA_MODE = str(cfg.get("GEOPRIOR_KAPPA_MODE", "bar"))
    GEOPRIOR_USE_EFFECTIVE_H = bool(
        cfg.get(
            "GEOPRIOR_USE_EFFECTIVE_H",
            CENSOR.get("use_effective_h_field", True),
        )
    )
    GEOPRIOR_HD_FACTOR = float(cfg.get("GEOPRIOR_HD_FACTOR", 0.6))

    GEOPRIOR_H_REF_VALUE = 0.0
    GEOPRIOR_H_REF_MODE = None
    if isinstance(GEOPRIOR_H_REF, (int, float)):
        GEOPRIOR_H_REF_VALUE = float(GEOPRIOR_H_REF)
    else:
        GEOPRIOR_H_REF_MODE = GEOPRIOR_H_REF

    # Columns
    cols_cfg = cfg.get("cols", {}) or {}
    SUBS_MODEL_COL = cols_cfg.get(
        "subs_model",
        cols_cfg.get("subsidence", "subsidence"),
    )
    GWL_MODEL_COL = cols_cfg.get(
        "gwl_model",
        cols_cfg.get("gwl", "GWL"),
    )
    SUBSIDENCE_COL = cols_cfg.get("subsidence", "subsidence")
    GWL_COL = cols_cfg.get("gwl", "GWL")

    # Output directory
    base_output_dir = (M.get("paths") or {}).get("run_dir")
    if not base_output_dir:
        base_output_dir = os.path.dirname(manifest_path)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_output_dir, f"train_{stamp}")
    ensure_directory_exists(run_dir)

    # ---------- encoders/scalers ----------
    encoders = (M.get("artifacts") or {}).get("encoders", {}) or {}
    
    # main_scaler = None
    # ms_path = encoders.get("main_scaler")
    # if ms_path and os.path.exists(ms_path):
    #     try:
    #         main_scaler = joblib.load(ms_path)
    #     except Exception as e:
    #         log(f"[Warn] main_scaler load failed: {e}")
    
    coord_scaler = None
    cs_path = encoders.get("coord_scaler")
    if cs_path and os.path.exists(cs_path):
        try:
            coord_scaler = joblib.load(cs_path)
        except Exception as e:
            log(f"[Warn] coord_scaler load failed: {e}")
    
    scaler_info_dict = load_scaler_info(encoders)
    # Attach scaler objects if only paths are present
    if isinstance(scaler_info_dict, dict):
        for _, v in scaler_info_dict.items():
            if isinstance(v, dict) and "scaler_path" in v and "scaler" not in v:
                p = v.get("scaler_path")
                if p and os.path.exists(p):
                    try:
                        v["scaler"] = joblib.load(p)
                    except Exception:
                        pass
    
    SUBS_SCALER_KEY = _pick_scaler_key(
        scaler_info_dict,
        preferred=SUBSIDENCE_COL,
        fallbacks=("subsidence", "subs_pred"),
    )
    
    GWL_SCALER_KEY = _pick_scaler_key(
        scaler_info_dict,
        preferred=GWL_COL,
        fallbacks=("gwl", "gwl_pred"),
    )

    log(f"[DEBUG] Using SUBS_SCALER_KEY={SUBS_SCALER_KEY!r}"
        f" GWL_SCALER_KEY = {GWL_SCALER_KEY} (SUBSIDENCE_COL={SUBSIDENCE_COL!r})")
    
    main_scaler = None
    ms_path = encoders.get("main_scaler")
    if ms_path and os.path.exists(ms_path):
        try:
            main_scaler = joblib.load(ms_path) # XXX Noqa
        except Exception as e:
            log(f"[Warn] Could not load main_scaler: {e}")
    else:
        log("[Warn] main_scaler path missing in manifest or file"
              " not found; continuing without it.")
        
    coord_scaler = None
    cs_path = encoders.get("coord_scaler")
    if cs_path and os.path.exists(cs_path):
        try:
            coord_scaler = joblib.load(cs_path)
        except Exception as e:
            log(f"[Warn] Could not load coord_scaler: {e}")

    # ---------- Load NPZ ----------
    npz = (M.get("artifacts") or {}).get("numpy", {}) or {}
    X_train = dict(np.load(npz["train_inputs_npz"]))
    y_train = dict(np.load(npz["train_targets_npz"]))
    X_val = dict(np.load(npz["val_inputs_npz"]))
    y_val = dict(np.load(npz["val_targets_npz"]))

    X_test = None
    y_test = None
    if npz.get("test_inputs_npz") and npz.get("test_targets_npz"):
        X_test = dict(np.load(npz["test_inputs_npz"]))
        y_test = dict(np.load(npz["test_targets_npz"]))

    # Output dims from Stage-1
    dims = (M.get("artifacts") or {}).get("sequences", {}) or {}
    OUT_S_DIM = int(dims.get("dims", {}).get("output_subsidence_dim", 1))
    OUT_G_DIM = int(dims.get("dims", {}).get("output_gwl_dim", 1))

    config_sections = [
        ("Run", {
            "CITY_NAME": CITY_NAME,
            "MODEL_NAME": MODEL_NAME,
            "RESULTS_DIR": results_dir,
            "MANIFEST_PATH": manifest_path,
            "RUN_OUTPUT_PATH": run_dir,
        }),
        ("Architecture", {
            "TIME_STEPS": TIME_STEPS,
            "FORECAST_HORIZON_YEARS": FORECAST_HORIZON_YEARS,
            "MODE": MODE,
            "ATTENTION_LEVELS": cfg.get(
                "ATTENTION_LEVELS", ["cross", "hierarchical", "memory"]
                ) ,
            "EMBED_DIM": cfg.get("EMBED_DIM", 32),
            "HIDDEN_UNITS": cfg.get("HIDDEN_UNITS", 64),
            "LSTM_UNITS": cfg.get("LSTM_UNITS", 64),
            "ATTENTION_UNITS": cfg.get("ATTENTION_UNITS", 64),
            "NUMBER_HEADS": cfg.get("NUMBER_HEADS", 2),
            "DROPOUT_RATE": cfg.get("DROPOUT_RATE", 0.10),
            "MEMORY_SIZE": cfg.get("MEMORY_SIZE", 50),
            "SCALES": cfg.get("SCALES", [1, 2]),
            "USE_RESIDUALS": cfg.get("USE_RESIDUALS", True),
            "USE_BATCH_NORM": cfg.get("USE_BATCH_NORM", False),
            "USE_VSN":  cfg.get("USE_VSN", True),
            "VSN_UNITS": cfg.get("VSN_UNITS", 32),
        }),
        ("Physics", {
            "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
            "SCALE_PDE_RESIDUALS": SCALE_PDE_RESIDUALS,
            "TIME_UNITS": TIME_UNITS,
            "LAMBDA_CONS": LAMBDA_CONS,
            "LAMBDA_GW": LAMBDA_GW,
            "LAMBDA_PRIOR": LAMBDA_PRIOR,
            "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
            "LAMBDA_BOUNDS": LAMBDA_BOUNDS, 
            "LAMBDA_MV": LAMBDA_MV,
            "LAMBDA_Q": LAMBDA_Q, 
            "LOSS_WEIGHT_GWL": LOSS_WEIGHT_GWL, 
            "LOG_Q_DIAGNOSTICS": LOG_Q_DIAGNOSTICS, 
            "MV_LR_MULT": MV_LR_MULT,
            "KAPPA_LR_MULT": KAPPA_LR_MULT,
            "GEOPRIOR_INIT_MV": GEOPRIOR_INIT_MV,
            "GEOPRIOR_INIT_KAPPA": GEOPRIOR_INIT_KAPPA,
            "GEOPRIOR_GAMMA_W": GEOPRIOR_GAMMA_W,
            "GEOPRIOR_H_REF": GEOPRIOR_H_REF,
            "GEOPRIOR_KAPPA_MODE": GEOPRIOR_KAPPA_MODE,
            "GEOPRIOR_USE_EFFECTIVE_H": GEOPRIOR_USE_EFFECTIVE_H,
            "GEOPRIOR_HD_FACTOR": GEOPRIOR_HD_FACTOR,
            "PHYSICS_BOUNDS": phys_bounds,
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
        config_sections, table_width =get_table_size(), 
        title=f"{CITY_NAME.upper()} {MODEL_NAME} TRAINING CONFIG",
        log_fn = log
    )
    
    log(f"\nTraining outputs -> {run_dir}")
    # ---------- datasets ----------
    train_ds = make_tf_dataset(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        mode=MODE,
        forecast_horizon=FORECAST_HORIZON_YEARS,
        check_npz_finite=True,
        check_finite=True,
        scan_finite_batches=None,
        dynamic_feature_names=list(DYN_NAMES),
        future_feature_names=list(FUT_NAMES),
    )
    val_ds = make_tf_dataset(
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        mode=MODE,
        forecast_horizon=FORECAST_HORIZON_YEARS,
        check_npz_finite=True,
        check_finite=True,
        scan_finite_batches=None,
        dynamic_feature_names=list(DYN_NAMES),
        future_feature_names=list(FUT_NAMES),
    )

    # ---------- model dims ----------
    X_train_norm = ensure_input_shapes(
        X_train,
        mode=MODE,
        forecast_horizon=FORECAST_HORIZON_YEARS,
    )
    s_dim = int(X_train_norm["static_features"].shape[-1])
    d_dim = int(X_train_norm["dynamic_features"].shape[-1])
    f_dim = int(X_train_norm["future_features"].shape[-1])

    # ---------- scaling kwargs (v3.2 pipeline) ----------
    sk_stage1 = cfg.get("scaling_kwargs", {}) or {}
    sk = dict(sk_stage1)

    conv = cfg.get("conventions", {}) or {}
    
    for _k in (
        "gwl_kind",
        "gwl_sign",
        "gwl_driver_kind",
        "gwl_driver_sign",
        "gwl_target_kind",
        "gwl_target_sign",
        "use_head_proxy",
        "time_units",
    ):
        if _k not in sk and _k in conv:
            sk[_k] = conv[_k]
    
    GWL_KIND = str(
        sk.get("gwl_kind", cfg.get("GWL_KIND", "depth_bgs"))
    ).lower()
    
    GWL_SIGN = str(
        sk.get("gwl_sign", cfg.get("GWL_SIGN", "down_positive"))
    ).lower()
    
    USE_HEAD_PROXY = bool(
        sk.get("use_head_proxy", cfg.get("USE_HEAD_PROXY", True))
    )
    
    Z_SURF_COL = sk.get("z_surf_col", cfg.get("Z_SURF_COL", None))
    
    cols_spec = (cfg.get("cols", {}) or {})
    z_surf_any = (
        Z_SURF_COL
        or cols_spec.get("z_surf_static")
        or cols_spec.get("z_surf_raw")
    )
    
    head_from_depth_rule = None
    if z_surf_any:
        head_from_depth_rule = "z_surf - depth"
    elif USE_HEAD_PROXY:
        head_from_depth_rule = "-depth (proxy)"
    
    gwl_z_meta = {
        "raw_kind": str(conv.get("gwl_kind", GWL_KIND)).lower(),
        "raw_sign": str(conv.get("gwl_sign", GWL_SIGN)).lower(),
        "driver_kind": str(
            conv.get("gwl_driver_kind", "depth")
        ).lower(),
        "driver_sign": str(
            conv.get("gwl_driver_sign", "down_positive")
        ).lower(),
        "target_kind": str(
            conv.get("gwl_target_kind", "head")
        ).lower(),
        "target_sign": str(
            conv.get("gwl_target_sign", "up_positive")
        ).lower(),
        "use_head_proxy": bool(USE_HEAD_PROXY),
        "z_surf_col": z_surf_any,
        "head_from_depth_rule": head_from_depth_rule,
        "cols": {
            "depth_raw": cols_spec.get("depth_raw"),
            "head_raw": cols_spec.get("head_raw"),
            "z_surf_raw": cols_spec.get("z_surf_raw"),
            "depth_model": cols_spec.get("depth_model"),
            "head_model": cols_spec.get("head_model"),
            "z_surf_static": cols_spec.get("z_surf_static"),
            "subs_model": SUBS_MODEL_COL,
        },
    }
    
    sk["gwl_z_meta"] = gwl_z_meta

    # coords normalization/ranges
    
    coords_normalized = bool(
        sk.get("coords_normalized", sk.get("normalize_coords", False))
    )
    
    coord_ranges = sk.get("coord_ranges") or None
    
    if coords_normalized and (not coord_ranges) and (
        coord_scaler is not None
    ):
        if hasattr(coord_scaler, "data_min_") and hasattr(
            coord_scaler, "data_max_"
        ):
            span = coord_scaler.data_max_ - coord_scaler.data_min_
            coord_ranges = {
                "t": float(span[0]),
                "x": float(span[1]),
                "y": float(span[2]),
            }
        elif hasattr(coord_scaler, "scale_"):
            sc = coord_scaler.scale_
            coord_ranges = {
                "t": float(sc[0]),
                "x": float(sc[1]),
                "y": float(sc[2]),
            }
    
    if coords_normalized and not coord_ranges:
        raise RuntimeError(
            "coords_normalized=True but coord_ranges missing."
        )
    
    coords_in_degrees = bool(sk.get("coords_in_degrees", False))
    deg_to_m_lon = sk.get("deg_to_m_lon", None)
    deg_to_m_lat = sk.get("deg_to_m_lat", None)
    coord_order = sk.get("coord_order", ["t", "x", "y"])
    
    if coords_in_degrees and (
        deg_to_m_lon is None or deg_to_m_lat is None
    ):
        lat_ref_deg = sk.get("lat_ref_deg", None)
    
        if lat_ref_deg is None or (
            isinstance(lat_ref_deg, str)
            and lat_ref_deg.strip().lower() == "auto"
        ):
            scaled_csv = (
                (M.get("artifacts") or {})
                .get("csv", {})
                .get("scaled", None)
            )
            lat_col = (cfg.get("cols", {}) or {}).get("lat", None)
            if scaled_csv and lat_col:
                try:
                    _lat = pd.read_csv(
                        scaled_csv,
                        usecols=[lat_col],
                    )[lat_col].to_numpy(dtype=float)
                    lat_ref_deg = float(np.nanmean(_lat))
                except Exception:
                    lat_ref_deg = None
    
        if lat_ref_deg is None or not np.isfinite(float(lat_ref_deg)):
            raise RuntimeError(
                "coords_in_degrees=True but meters/deg missing."
            )
    
        lat_ref_deg = float(lat_ref_deg)
        deg_to_m_lon, deg_to_m_lat = deg_to_m_from_lat(lat_ref_deg)
    
        sk.update(
            {
                "lat_ref_deg": float(lat_ref_deg),
                "deg_to_m_lon": float(deg_to_m_lon),
                "deg_to_m_lat": float(deg_to_m_lat),
            }
        )
    
    H_scale_si = sk.get("H_scale_si", None)
    H_bias_si = sk.get("H_bias_si", None)
    
    if "gwl_dyn_index" in sk_stage1 and (
        sk_stage1["gwl_dyn_index"] is not None
    ):
        GWL_DYN_INDEX = int(sk_stage1["gwl_dyn_index"])
        if not (0 <= GWL_DYN_INDEX < len(DYN_NAMES)):
            raise RuntimeError(
                "Stage-1 gwl_dyn_index out of bounds."
            )
        gwl_dyn_name = DYN_NAMES[GWL_DYN_INDEX]
    else:
        gwl_dyn_name = (
            sk_stage1.get("gwl_dyn_name")
            or sk_stage1.get("gwl_col")
            or GWL_COL
        )
        if gwl_dyn_name not in DYN_NAMES:
            for cand in (
                GWL_COL,
                "z_GWL",
                "Z_GWL",
                "gwl",
                "GWL",
                "depth_to_water",
            ):
                if cand in DYN_NAMES:
                    gwl_dyn_name = cand
                    break
        if gwl_dyn_name not in DYN_NAMES:
            raise RuntimeError(
                "Cannot find GWL driver in dynamic features."
            )
        GWL_DYN_INDEX = int(DYN_NAMES.index(gwl_dyn_name))
    
    Z_SURF_STATIC_INDEX = sk_stage1.get("z_surf_static_index")
    
    SUBS_DYN_INDEX = sk_stage1.get("subs_dyn_index")
    sub_dyn_name = sk_stage1.get("subs_dyn_name")
    
    if SUBS_DYN_INDEX is None and sub_dyn_name:
        if sub_dyn_name in DYN_NAMES:
            SUBS_DYN_INDEX = int(DYN_NAMES.index(sub_dyn_name))

    # SI affine resolve
    subs_scale_si = sk.get("subs_scale_si")
    subs_bias_si = sk.get("subs_bias_si")
    head_scale_si = sk.get("head_scale_si")
    head_bias_si = sk.get("head_bias_si")

    if subs_scale_si is None or subs_bias_si is None:
        subs_scale_si, subs_bias_si = resolve_si_affine(
            cfg,
            scaler_info_dict,
            target_name=SUBSIDENCE_COL,
            prefix="SUBS",
            unit_factor_key="SUBS_UNIT_TO_SI",
            scale_key="SUBS_SCALE_SI",
            bias_key="SUBS_BIAS_SI",
        )

    if head_scale_si is None or head_bias_si is None:
        head_scale_si, head_bias_si = resolve_si_affine(
            cfg,
            scaler_info_dict,
            target_name=GWL_COL,
            prefix="HEAD",
            unit_factor_key="HEAD_UNIT_TO_SI",
            scale_key="HEAD_SCALE_SI",
            bias_key="HEAD_BIAS_SI",
        )

    sk.update(
        {
            "subs_scale_si": float(subs_scale_si),
            "subs_bias_si": float(subs_bias_si),
            "head_scale_si": float(head_scale_si),
            "head_bias_si": float(head_bias_si),
            "coords_normalized": bool(coords_normalized),
            "coord_ranges": coord_ranges or {},
            "coord_order": coord_order,
            "coords_in_degrees": bool(coords_in_degrees),
            "deg_to_m_lon": deg_to_m_lon,
            "deg_to_m_lat": deg_to_m_lat,
            "time_units": TIME_UNITS,
            "dynamic_feature_names": list(DYN_NAMES),
            "future_feature_names": list(FUT_NAMES),
            "static_feature_names": list(STA_NAMES),
            "loss_weight_gwl": float(LOSS_WEIGHT_GWL),
            "lambda_q": float(LAMBDA_Q),
            "log_q_diagnostics": bool(LOG_Q_DIAGNOSTICS),
        }
    )

    # gates/training strategy (same as stage2.py)
    TRAINING_STRATEGY = str(
        cfg.get("TRAINING_STRATEGY", "data_first")
    ).strip().lower()
    if TRAINING_STRATEGY not in ("physics_first", "data_first"):
        raise ValueError(
            "TRAINING_STRATEGY must be 'physics_first' or 'data_first'."
        )

    n_train = int(X_train_norm["static_features"].shape[0])
    steps_per_epoch = int(np.ceil(n_train / float(BATCH_SIZE)))

    MV_PRIOR_MODE = str(
        sk.get("mv_prior_mode", cfg.get("MV_PRIOR_MODE", "calibrate"))
    )
    
    MV_WEIGHT = float(
        sk.get("mv_weight", cfg.get("MV_WEIGHT", 1e-3))
    )
    
    MV_SCHEDULE_UNIT = str(
        sk.get(
            "mv_schedule_unit",
            cfg.get("MV_SCHEDULE_UNIT", "epoch"),
        )
    ).strip().lower()
    
    MV_DELAY_EPOCHS = int(
        sk.get("mv_delay_epochs", cfg.get("MV_DELAY_EPOCHS", 1))
    )
    MV_WARMUP_EPOCHS = int(
        sk.get("mv_warmup_epochs", cfg.get("MV_WARMUP_EPOCHS", 2))
    )
    
    MV_DELAY_STEPS = sk.get("mv_delay_steps", cfg.get("MV_DELAY_STEPS"))
    MV_WARMUP_STEPS = sk.get(
        "mv_warmup_steps",
        cfg.get("MV_WARMUP_STEPS"),
    )
    
    def _int_or_none(v):
        return None if v is None else int(v)
    
    mv_delay_steps = _int_or_none(MV_DELAY_STEPS)
    mv_warmup_steps = _int_or_none(MV_WARMUP_STEPS)
    
    if mv_delay_steps is None:
        mv_delay_steps = max(0, MV_DELAY_EPOCHS) * steps_per_epoch
    if mv_warmup_steps is None:
        mv_warmup_steps = max(0, MV_WARMUP_EPOCHS) * steps_per_epoch
    
    sk.update(
        {
            "mv_prior_mode": MV_PRIOR_MODE,
            "mv_weight": MV_WEIGHT,
            "mv_schedule_unit": MV_SCHEDULE_UNIT,
            "mv_delay_epochs": int(MV_DELAY_EPOCHS),
            "mv_warmup_epochs": int(MV_WARMUP_EPOCHS),
            "mv_delay_steps": int(mv_delay_steps),
            "mv_warmup_steps": int(mv_warmup_steps),
            "mv_steps_per_epoch": int(steps_per_epoch),
        }
    )

    q_policy = "always_on"
    q_warmup_epochs = 0
    q_ramp_epochs = 0

    subs_resid_policy = "always_on"
    subs_resid_warmup_epochs = 0
    subs_resid_ramp_epochs = 0

    if TRAINING_STRATEGY == "physics_first":
        q_policy = str(
            cfg.get("Q_POLICY_PHYSICS_FIRST", "warmup_off")
        ).strip().lower()
        q_warmup_epochs = int(
            cfg.get("Q_WARMUP_EPOCHS_PHYSICS_FIRST", 5)
        )
        q_ramp_epochs = int(
            cfg.get("Q_RAMP_EPOCHS_PHYSICS_FIRST", 0)
        )

        subs_resid_policy = str(
            cfg.get("SUBS_RESID_POLICY_PHYSICS_FIRST", "warmup_off")
        ).strip().lower()
        subs_resid_warmup_epochs = int(
            cfg.get("SUBS_RESID_WARMUP_EPOCHS_PHYSICS_FIRST", 5)
        )
        subs_resid_ramp_epochs = int(
            cfg.get("SUBS_RESID_RAMP_EPOCHS_PHYSICS_FIRST", 0)
        )

        LAMBDA_Q = float(cfg.get("LAMBDA_Q_PHYSICS_FIRST", LAMBDA_Q))
        LOSS_WEIGHT_GWL = float(
            cfg.get("LOSS_WEIGHT_GWL_PHYSICS_FIRST", LOSS_WEIGHT_GWL)
        )
    else:
        LOSS_WEIGHT_GWL = float(
            cfg.get("LOSS_WEIGHT_GWL_DATA_FIRST", LOSS_WEIGHT_GWL)
        )
        LAMBDA_Q = float(cfg.get("LAMBDA_Q_DATA_FIRST", LAMBDA_Q))

        q_policy = str(
            cfg.get("Q_POLICY_DATA_FIRST", "always_on")
        ).strip().lower()
        q_warmup_epochs = int(cfg.get("Q_WARMUP_EPOCHS_DATA_FIRST", 0))
        q_ramp_epochs = int(cfg.get("Q_RAMP_EPOCHS_DATA_FIRST", 0))

        subs_resid_policy = str(
            cfg.get("SUBS_RESID_POLICY_DATA_FIRST", "always_on")
        ).strip().lower()
        subs_resid_warmup_epochs = int(
            cfg.get("SUBS_RESID_WARMUP_EPOCHS_DATA_FIRST", 0)
        )
        subs_resid_ramp_epochs = int(
            cfg.get("SUBS_RESID_RAMP_EPOCHS_DATA_FIRST", 0)
        )

    if q_policy == "always_off":
        LAMBDA_Q = 0.0

    q_warmup_steps = max(0, q_warmup_epochs) * steps_per_epoch
    q_ramp_steps = max(0, q_ramp_epochs) * steps_per_epoch

    subs_resid_warmup_steps = max(0, subs_resid_warmup_epochs) * steps_per_epoch
    subs_resid_ramp_steps = max(0, subs_resid_ramp_epochs) * steps_per_epoch 
    
    sk.update(
        {
            "training_strategy": TRAINING_STRATEGY,
            "q_policy": q_policy,
            "q_warmup_epochs": int(q_warmup_epochs),
            "q_ramp_epochs": int(q_ramp_epochs),
            "q_warmup_steps": int(q_warmup_steps),
            "q_ramp_steps": int(q_ramp_steps),
            "log_q_diagnostics": bool(LOG_Q_DIAGNOSTICS),
            "subs_resid_policy": subs_resid_policy,
            "subs_resid_warmup_epochs": int(subs_resid_warmup_epochs),
            "subs_resid_ramp_epochs": int(subs_resid_ramp_epochs),
            "subs_resid_warmup_steps": int(subs_resid_warmup_steps),
            "subs_resid_ramp_steps": int(subs_resid_ramp_steps),
            "loss_weight_gwl": float(LOSS_WEIGHT_GWL),
            "lambda_q": float(LAMBDA_Q),
            "physics_warmup_steps": int(cfg.get("PHYSICS_WARMUP_STEPS", 500)),
            "physics_ramp_steps": int(cfg.get("PHYSICS_RAMP_STEPS", 500)),
        }
    )

    sk.update(
        {
            "bounds": bounds_for_scaling,
            "time_units": TIME_UNITS,
            "coords_normalized": coords_normalized,
            "coord_ranges": coord_ranges or {},
            "coord_order": coord_order,
            "coords_in_degrees": coords_in_degrees,
            "deg_to_m_lon": (
                float(deg_to_m_lon)
                if deg_to_m_lon is not None
                else None
            ),
            "deg_to_m_lat": (
                float(deg_to_m_lat)
                if deg_to_m_lat is not None
                else None
            ),
            "H_scale_si": (
                float(H_scale_si)
                if H_scale_si is not None
                else 1.0
            ),
            "H_bias_si": (
                float(H_bias_si)
                if H_bias_si is not None
                else 0.0
            ),
            "dynamic_feature_names": list(DYN_NAMES),
            "future_feature_names": list(FUT_NAMES),
            "static_feature_names": list(STA_NAMES),
            "gwl_dyn_name": gwl_dyn_name,
            "gwl_dyn_index": int(GWL_DYN_INDEX),
            "z_surf_static_index": (
                int(Z_SURF_STATIC_INDEX)
                if Z_SURF_STATIC_INDEX is not None
                else None
            ),
            "subs_dyn_index": (
                int(SUBS_DYN_INDEX)
                if SUBS_DYN_INDEX is not None
                else None
            ),
            "subs_dyn_name": (
                sub_dyn_name
                if sub_dyn_name is not None
                else SUBS_MODEL_COL
            ),
            "subs_scale_si": float(subs_scale_si),
            "subs_bias_si": float(subs_bias_si),
            "head_scale_si": float(head_scale_si),
            "head_bias_si": float(head_bias_si),
            "gwl_kind": GWL_KIND,
            "gwl_sign": GWL_SIGN,
            "use_head_proxy": bool(USE_HEAD_PROXY),
            "z_surf_col": Z_SURF_COL,
            "gwl_z_meta": sk.get("gwl_z_meta", None),
            "subsidence_kind": sk.get(
                "subsidence_kind",
                cfg.get("SUBSIDENCE_KIND", "cumulative"),
            ),
            "cons_scale_floor": cfg.get("CONS_SCALE_FLOOR", 1e-7),
            "gw_scale_floor": cfg.get("GW_SCALE_FLOOR", 1e-7),
            "gw_residual_units": cfg.get(
                "GW_RESIDUAL_UNITS",
                "time_unit",
            ),
            "cons_residual_units": cfg.get(
                "CONSOLIDATION_RESIDUAL_UNITS",
                "second",
            ),
            "clip_global_norm": cfg.get("CLIP_GLOBAL_NORM", 5.0),
            "debug_physics_grads": cfg.get("DEBUG_PHYSICS_GRADS", False),
            "scaling_error_policy": cfg.get(
                "SCALING_ERROR_POLICY",
                "warn",
            ),
            "track_aux_metrics": bool(TRACK_AUX_METRICS),
        }
    )
    
    sk = finalize_scaling_kwargs(sk)
    
    sk = override_scaling_kwargs(
        sk,
        cfg,
        finalize=finalize_scaling_kwargs,
        dyn_names=DYN_NAMES,
        gwl_dyn_index=GWL_DYN_INDEX,
        base_dir=os.path.dirname(__file__),
        strict=True,
        log_fn=log,
    )
    
    sk = {k: v for k, v in sk.items() if v is not None}


    with open(
        os.path.join(run_dir, "scaling_kwargs.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(sk, f, indent=2)

    # ---------- build model ----------
    MODEL_CLASS_REGISTRY = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "PoroElasticSubsNet": PoroElasticSubsNet,
        "HybridAttn-NoPhysics": GeoPriorSubsNet,
    }
    model_cls = MODEL_CLASS_REGISTRY.get(MODEL_NAME, GeoPriorSubsNet)

    subsmodel_params = {
        "embed_dim": int(cfg.get("EMBED_DIM", 32)),
        "hidden_units": int(cfg.get("HIDDEN_UNITS", 64)),
        "lstm_units": int(cfg.get("LSTM_UNITS", 64)),
        "attention_units": int(cfg.get("ATTENTION_UNITS", 64)),
        "num_heads": int(cfg.get("NUMBER_HEADS", 2)),
        "dropout_rate": float(cfg.get("DROPOUT_RATE", 0.10)),
        "max_window_size": TIME_STEPS,
        "memory_size": int(cfg.get("MEMORY_SIZE", 50)),
        "scales": cfg.get("SCALES", [1, 2]),
        "multi_scale_agg": "last",
        "final_agg": "last",
        "use_residuals": bool(cfg.get("USE_RESIDUALS", True)),
        "use_batch_norm": bool(cfg.get("USE_BATCH_NORM", False)),
        "use_vsn": bool(cfg.get("USE_VSN", True)),
        "vsn_units": int(cfg.get("VSN_UNITS", 32)),
        "mode": MODE,
        "attention_levels": cfg.get(
            "ATTENTION_LEVELS",
            ["cross", "hierarchical", "memory"],
        ),
        "scale_pde_residuals": SCALE_PDE_RESIDUALS,
        "scaling_kwargs": sk,
        "bounds_mode": str(PHYSICS_BOUNDS_MODE or "soft"),
        "mv": LearnableMV(initial_value=GEOPRIOR_INIT_MV),
        "kappa": LearnableKappa(initial_value=GEOPRIOR_INIT_KAPPA),
        "gamma_w": FixedGammaW(value=GEOPRIOR_GAMMA_W),
        "h_ref": FixedHRef(
            value=GEOPRIOR_H_REF_VALUE,
            mode=GEOPRIOR_H_REF_MODE,
        ),
        "kappa_mode": GEOPRIOR_KAPPA_MODE,
        "use_effective_h": GEOPRIOR_USE_EFFECTIVE_H,
        "hd_factor": GEOPRIOR_HD_FACTOR,
        "offset_mode": OFFSET_MODE,
        "time_units": TIME_UNITS,
    }

    if should_audit(AUDIT_STAGES, stage="stage2"):
        _ = audit_stage2_handshake(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            time_steps=TIME_STEPS,
            forecast_horizon=FORECAST_HORIZON_YEARS,
            mode=MODE,
            dyn_names=list(DYN_NAMES),
            fut_names=list(FUT_NAMES),
            sta_names=list(STA_NAMES),
            coord_scaler=coord_scaler,
            sk_final=sk,
            save_dir=run_dir,
            table_width=get_table_size(),
            title_prefix="STAGE-2 HANDSHAKE AUDIT",
            city=CITY_NAME,
            model_name=MODEL_NAME,
        )

    model = model_cls(
        static_input_dim=s_dim,
        dynamic_input_dim=d_dim,
        future_input_dim=f_dim,
        output_subsidence_dim=OUT_S_DIM,
        output_gwl_dim=OUT_G_DIM,
        forecast_horizon=FORECAST_HORIZON_YEARS,
        quantiles=QUANTILES,
        pde_mode=PDE_MODE_CONFIG,
        verbose=0,
        **subsmodel_params,
    )

    # Build once
    for xb, _ in train_ds.take(1):
        model(xb)
        break

    # Losses
    loss_dict = {
        "subs_pred": (
            make_weighted_pinball(QUANTILES, SUBS_WEIGHTS)
            if QUANTILES
            else tf.keras.losses.MSE
        ),
        "gwl_pred": (
            make_weighted_pinball(QUANTILES, GWL_WEIGHTS)
            if QUANTILES
            else tf.keras.losses.MSE
        ),
    }
    loss_weights_dict = {
        "subs_pred": 1.0,
        "gwl_pred": float(LOSS_WEIGHT_GWL),
    }

    # Metrics
    if TRACK_AUX_METRICS:
        metrics_arg = None
    else:
        if QUANTILES:
            metrics_arg = {
                "subs_pred": [
                    MAEQ50(name="mae_q50"),
                    MSEQ50(name="mse_q50"),
                    Coverage80(name="coverage80"),
                    Sharpness80(name="sharpness80"),
                ],
                "gwl_pred": [
                    MAEQ50(name="mae_q50"),
                    MSEQ50(name="mse_q50"),
                ],
            }
        else:
            metrics_arg = {
                "subs_pred": ["mae", "mse"],
                "gwl_pred": ["mae", "mse"],
            }

    physics_loss_weights = {
        "lambda_cons": float(LAMBDA_CONS),
        "lambda_gw": float(LAMBDA_GW),
        "lambda_prior": float(LAMBDA_PRIOR),
        "lambda_smooth": float(LAMBDA_SMOOTH),
        "lambda_bounds": float(LAMBDA_BOUNDS),
        "lambda_mv": float(LAMBDA_MV),
        "mv_lr_mult": float(MV_LR_MULT),
        "lambda_offset": float(LAMBDA_OFFSET),
        "kappa_lr_mult": float(KAPPA_LR_MULT),
        "lambda_q": float(LAMBDA_Q),
    }

    # Keras2 vs Keras3 compile safety
    try:
        import keras  # type: ignore

        is_keras2 = str(keras.__version__).startswith("2.")
    except Exception:
        is_keras2 = False

    out_names = list(getattr(model, "output_names", [])) or [
        "subs_pred",
        "gwl_pred",
    ]
    if is_keras2:
        loss_arg = [loss_dict[k] for k in out_names]
        lossw_arg = [loss_weights_dict.get(k, 1.0) for k in out_names]
        if metrics_arg is None:
            metrics_compile = None
        else:
            metrics_compile = [metrics_arg.get(k, []) for k in out_names]
    else:
        loss_arg = loss_dict
        lossw_arg = loss_weights_dict
        metrics_compile = metrics_arg

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        loss=loss_arg,
        loss_weights=lossw_arg,
        metrics=metrics_compile,
        **physics_loss_weights,
    )

    # ---------- save init manifest (JSON safe) ----------
    bundle_prefix = f"{CITY_NAME}_{MODEL_NAME}_H{FORECAST_HORIZON_YEARS}"
    best_keras_path = os.path.join(run_dir, f"{bundle_prefix}_best.keras")
    best_weights_path = os.path.join(
        run_dir,
        f"{bundle_prefix}_best.weights.h5",
    )
    best_tf_dir = os.path.join(run_dir, f"{bundle_prefix}_best_savedmodel")
    init_manifest_path = os.path.join(run_dir, "model_init_manifest.json")

    init_manifest = {
        "model_class": model_cls.__name__,
        "dims": {
            "static_input_dim": int(s_dim),
            "dynamic_input_dim": int(d_dim),
            "future_input_dim": int(f_dim),
            "output_subsidence_dim": int(OUT_S_DIM),
            "output_gwl_dim": int(OUT_G_DIM),
            "forecast_horizon": int(FORECAST_HORIZON_YEARS),
        },
        "config": {
            "quantiles": list(QUANTILES) if QUANTILES else None,
            "pde_mode": PDE_MODE_CONFIG,
            "mode": MODE,
            "time_units": TIME_UNITS,
            "geoprior": {
                "init_mv": float(GEOPRIOR_INIT_MV),
                "init_kappa": float(GEOPRIOR_INIT_KAPPA),
                "gamma_w": float(GEOPRIOR_GAMMA_W),
                "h_ref_value": float(GEOPRIOR_H_REF_VALUE),
                "h_ref_mode": GEOPRIOR_H_REF_MODE,
                "kappa_mode": GEOPRIOR_KAPPA_MODE,
                "use_effective_h": bool(GEOPRIOR_USE_EFFECTIVE_H),
                "hd_factor": float(GEOPRIOR_HD_FACTOR),
                "offset_mode": OFFSET_MODE,
            },
            "scaling_kwargs": sk,
        },
    }
    save_manifest(init_manifest_path, init_manifest)

    # ---------- callbacks ----------
    # def _format_eta(seconds: float) -> str:
    #     if seconds is None or seconds < 0 or seconds == float("inf"):
    #         return "--:--"
    #     m, s = divmod(int(seconds), 60)
    #     h, m = divmod(m, 60)
    #     return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    # train_start_t = None

    # def _on_pct(pct: int) -> None:
    #     nonlocal train_start_t
    #     if not progress_callback:
    #         return

    #     now = time.time()
    #     frac = max(0.0, min(1.0, float(pct) / 100.0))
    #     if train_start_t is None and frac > 0.0:
    #         train_start_t = now

    #     if train_start_t is not None and frac > 1e-6:
    #         elapsed = max(0.0, now - train_start_t)
    #         rem = elapsed * (1.0 - frac) / frac
    #         eta = _format_eta(rem)
    #     else:
    #         eta = "--:--"

    #     msg = f"Training - {pct}% (ETA {eta})"
    #     try:
    #         progress_callback(0.25 + 0.55 * frac, msg)
    #     except Exception:
    #         pass


    history_groups = {
        "Total Loss": ["total_loss"],
    
        "Data vs Physics": ["data_loss", "physics_loss_scaled", "physics_loss"],
    
        "Offset Controls": ["lambda_offset", "physics_mult"],
    
        "Physics Components": [
            "consolidation_loss",
            "gw_flow_loss",
            "prior_loss",
            "smooth_loss",
            "mv_prior_loss",
            "bounds_loss",
        ],
    
        "Subsidence MAE": ["subs_pred_mae"],
        "GWL MAE": ["gwl_pred_mae"],
    
        # Keep the group if you want it, but only train key:
        "Physics Loss (Scaled)": ["physics_loss_scaled"],
    }
        
    csvlog_path = os.path.join(run_dir, f"{CITY_NAME}_{MODEL_NAME}_train_log.csv")
        
    cbs = [
        
        GuiEarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=0,
            log_fn=log,
        ),
        
        ModelCheckpoint(
            filepath=best_keras_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        CSVLogger(
            csvlog_path,
            append=False,
        ),
        # EarlyStopping(
        #     monitor="val_loss",
        #     patience=15,
        #     restore_best_weights=True,
        #     verbose=1,
        # ),
        TerminateOnNaN(),
        # _StopCallback(stop_check, log),
        # _ProgressCallback(steps_per_epoch, EPOCHS, _on_pct),
    ]

    # Figure out batches per epoch once train_dataset is built
    # try:
    #     batches_per_epoch = int(
    #         tfdata_experimental.cardinality(train_ds).numpy()
    #     )
    # except Exception:
    #     batches_per_epoch = 1
        
    try:
        card = int(tf.data.experimental.cardinality(train_ds).numpy())
    except Exception:
        card = -1

    batches_per_epoch = card if card > 0 else steps_per_epoch
    
    # Optional: only add epoch logger if we have a logger
    if logger is not None:
        cbs.append(GuiEpochLogger(log_fn=log))
        cbs.append(
            GuiMetricLogger(
                metric_groups=history_groups,
                log_fn=log,
                total_epochs=EPOCHS,
                precision=6,
            )
        )

    if stop_check is not None:
        cbs.append(StopTrainingOnSignal(stop_check, logger=log))
        
    if progress_callback is not None and batches_per_epoch > 0:
        gui_cb = GuiProgress(
            total_epochs=EPOCHS,
            batches_per_epoch=batches_per_epoch,
            update_fn=_progress_from_pct,
            epoch_level=False,
        )
        cbs.append(gui_cb)
        fit_verbose = 0
    else:
        fit_verbose = cfg.get("FIT_VERBOSE", 1)
        
    if USE_LAMBDA_OFFSET_SCHEDULER and (not model._physics_off()):
        cbs.append(
            LambdaOffsetScheduler(
                schedule=LAMBDA_OFFSET_SCHEDULE,
                unit=LAMBDA_OFFSET_UNIT,
                when=LAMBDA_OFFSET_WHEN,
                warmup=LAMBDA_OFFSET_WARMUP,
                start=LAMBDA_OFFSET_START,
                end=LAMBDA_OFFSET_END,
                clamp_positive=True,
                verbose=1,
            )
        )

    # ---------- train ----------
    log(f"\n[Stage2] Training outputs -> {run_dir}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=fit_verbose,
    )

    best_epoch, metrics_at_best = best_epoch_and_metrics(history.history)

    summary_json = os.path.join(
        run_dir,
        f"{CITY_NAME}_{MODEL_NAME}_training_summary.json",
    )
    training_summary = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "city": CITY_NAME,
        "model": MODEL_NAME,
        "horizon": int(FORECAST_HORIZON_YEARS),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "metrics_at_best": metrics_at_best,
        "final_epoch_metrics": {
            k: float(v[-1]) for k, v in history.history.items() if len(v)
        },
        "env": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": device_info,
        },
        "compile": {
            "optimizer": "Adam",
            "learning_rate": float(LEARNING_RATE),
            "loss_weights": loss_weights_dict,
            "metrics": (
                {k: [name_of(m) for m in v] for k, v in metrics_arg.items()}
                if metrics_arg else {}
            ),
            "physics_loss_weights": physics_loss_weights,
            "lambda_offset": float(LAMBDA_OFFSET),
        },
        "hp_init": {
            "quantiles": QUANTILES,
            "subs_weights": SUBS_WEIGHTS,
            "gwl_weights": GWL_WEIGHTS,
            "pde_mode": PDE_MODE_CONFIG,
            "time_steps": int(TIME_STEPS),
            "mode": MODE,
            "model_init_params": serialize_subs_params(subsmodel_params, cfg),
        },
        "paths": {
            "run_dir": run_dir,
            "best_keras": best_keras_path,
            "best_weights": best_weights_path,
            "best_tf_dir": best_tf_dir,
            "model_init_manifest": init_manifest_path,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    if stop_check and stop_check():
        log("[GUI] Stop requested after training. Returning partial artifacts.")
        return {
            "status": "stopped",
            "run_dir": run_dir,
            "training_summary_json": summary_json,
            "best_keras": best_keras_path,
            "best_weights": best_weights_path,
            "model_init_manifest": init_manifest_path,
        }

    yscales = {
        "Total Loss": "log",
        "Data vs Physics": "log",
        "Physics Components": "log",
        "Physics Loss (Scaled)": "log",
        "Offset Controls": "linear",
        "Subsidence MAE": "linear",
        "GWL MAE": "linear",
    }
    
    plot_history_in(
        history.history,
        metrics=history_groups,
        title=f"{MODEL_NAME} Training History",
        yscale_settings=yscales,
        layout="subplots",
        savefig=os.path.join(
            run_dir,
            f"{CITY_NAME}_{MODEL_NAME.lower()}_training_history_plot.png",  # add extension
        ),
    )


    # Save model bundle (TF or keras)
    save_model(
        model=model,
        keras_path=(best_tf_dir if USE_TF_SAVEDMODEL else best_keras_path),
        weights_path=best_weights_path,
        manifest_path=init_manifest_path,
        manifest=init_manifest,
        overwrite=True,
        use_tf_format=USE_TF_SAVEDMODEL,
    )

    # Optional plots/phys params
    try:
        autoplot_geoprior_history(
            history,
            outdir=run_dir,
            prefix=bundle_prefix,
            style="default",
            log_fn=log,
        )
    except Exception:
        pass

    try:
        extract_physical_parameters(
            model,
            to_csv=True,
            filename=f"{CITY_NAME}_{MODEL_NAME.lower()}_physical_parameters.csv",
            save_dir=run_dir,
            model_name="geoprior",
        )
    except Exception:
        pass

    if not evaluate_training:
        return {
            "status": "ok",
            "run_dir": run_dir,
            "training_summary_json": summary_json,
            "best_keras": best_keras_path,
            "best_weights": best_weights_path,
            "best_tf_dir": best_tf_dir,
            "model_init_manifest": init_manifest_path,
        }

    # ---------- inference load ----------
    custom_objects_load = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "PoroElasticSubsNet": PoroElasticSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
    }

    build_inputs = None
    for xb, _ in val_ds.take(1):
        build_inputs = xb
        break

    def builder(mani: dict):
        dims = (mani or {}).get("dims", {}) or {}
        cfgm = (mani or {}).get("config", {}) or {}
        gp = (cfgm.get("geoprior", {}) or {})

        _params = dict(subsmodel_params)
        _params.update(
            {
                "mv": LearnableMV(
                    initial_value=float(gp.get("init_mv", GEOPRIOR_INIT_MV))
                ),
                "kappa": LearnableKappa(
                    initial_value=float(
                        gp.get("init_kappa", GEOPRIOR_INIT_KAPPA)
                    )
                ),
                "gamma_w": FixedGammaW(
                    value=float(gp.get("gamma_w", GEOPRIOR_GAMMA_W))
                ),
                "h_ref": FixedHRef(
                    value=float(gp.get("h_ref_value", GEOPRIOR_H_REF_VALUE)),
                    mode=gp.get("h_ref_mode", GEOPRIOR_H_REF_MODE),
                ),
            }
        )

        return model_cls(
            static_input_dim=int(dims.get("static_input_dim", s_dim)),
            dynamic_input_dim=int(dims.get("dynamic_input_dim", d_dim)),
            future_input_dim=int(dims.get("future_input_dim", f_dim)),
            output_subsidence_dim=int(dims.get("output_subsidence_dim", OUT_S_DIM)),
            output_gwl_dim=int(dims.get("output_gwl_dim", OUT_G_DIM)),
            forecast_horizon=int(dims.get("forecast_horizon", FORECAST_HORIZON_YEARS)),
            quantiles=cfgm.get("quantiles", QUANTILES),
            pde_mode=cfgm.get("pde_mode", PDE_MODE_CONFIG),
            verbose=0,
            **_params,
        )

    if USE_IN_MEMORY_MODEL:
        model_inf = model
        log("[Info] Using in-memory model for inference.")
    elif USE_TF_SAVEDMODEL:
        model_inf = load_model_from_tfv2(
            best_tf_dir,
            endpoint="serve",
            custom_objects=custom_objects_load,
        )
        log(f"[OK] Loaded TF SavedModel inference: {best_tf_dir}")
    else:
        model_inf = load_inference_model(
            keras_path=best_keras_path,
            weights_path=best_weights_path,
            manifest_path=init_manifest_path,
            custom_objects=custom_objects_load,
            compile=False,
            builder=builder,
            build_inputs=build_inputs,
            prefer_full_model=False,
            log_fn=log,
            use_in_memory_model=False,
        )
        log("[OK] Loaded inference model from bundle.")

    if DEBUG and (model_inf is not model):
        try:
            _ = debug_model_reload(
                model,
                model_inf,
                val_ds,
                pred_key="subs_pred",
                also_check=["subs_pred", "gwl_pred"],
                top_weights=30,
                log_fn=log,
            )
        except Exception:
            pass

    # ---------- calibration ----------
    log("[Stage2] Fitting interval calibrator (80%) on val...")
    cal80 = fit_interval_calibrator_on_val(
        model_inf,
        val_ds,
        target=0.80,
    )
    cal_path = os.path.join(run_dir, "interval_factors_80.npy")
    np.save(cal_path, cal80.factors_)

    # ---------- forecast split ----------
    dataset_name = "ValidationSet_Fallback"
    X_fore = X_val
    y_fore = y_val
    if X_test is not None and y_test is not None:
        X_fore = X_test
        y_fore = y_test
        dataset_name = "TestSet"

    X_fore = ensure_input_shapes(
        X_fore,
        mode=MODE,
        forecast_horizon=FORECAST_HORIZON_YEARS,
    )
    y_fore_fmt = map_targets_for_training(y_fore)

    pred_out = model_inf.predict(X_fore, verbose=0)

    if isinstance(pred_out, dict):
        pred_dict = pred_out
    elif isinstance(pred_out, (list, tuple)):
        names = list(getattr(model_inf, "output_names", []) or [])
        if names:
            pred_dict = {
                names[i]: pred_out[i]
                for i in range(min(len(names), len(pred_out)))
            }
        else:
            pred_dict = {"subs_pred": pred_out[0]}
            if len(pred_out) > 1:
                pred_dict["gwl_pred"] = pred_out[1]
    else:
        raise TypeError(f"Unexpected predict type: {type(pred_out)}")

    s_pred = pred_dict.get("subs_pred")
    h_pred = pred_dict.get("gwl_pred")
    if s_pred is None or h_pred is None:
        raise KeyError(
            f"predict() must return subs_pred/gwl_pred. "
            f"Got keys={list(pred_dict.keys())}"
        )

    if QUANTILES:
        s_pred = canonicalize_BHQO_quantiles_np(
            s_pred,
            n_q=len(QUANTILES),
            verbose=1 if DEBUG else 0,
        )
        h_pred = canonicalize_BHQO_quantiles_np(
            h_pred,
            n_q=len(QUANTILES),
            verbose=1 if DEBUG else 0,
        )
        s_pred = apply_calibrator_to_subs(cal80, s_pred)

    if DEBUG and QUANTILES:
        debug_quantile_crossing_np(
            s_pred,
            n_q=len(QUANTILES),
            name="subs_pred",
            verbose=1,
        )

    y_true_for_format = {
        "subsidence": y_fore_fmt["subs_pred"],
        "gwl": y_fore_fmt["gwl_pred"],
    }

    csv_eval = os.path.join(
        run_dir,
        f"{CITY_NAME}_{MODEL_NAME}_forecast_"
        f"{dataset_name}_H{FORECAST_HORIZON_YEARS}_calibrated.csv",
    )
    csv_future = os.path.join(
        run_dir,
        f"{CITY_NAME}_{MODEL_NAME}_forecast_"
        f"{dataset_name}_H{FORECAST_HORIZON_YEARS}_future.csv",
    )

    future_grid = np.arange(
        float(FORECAST_START_YEAR),
        float(FORECAST_START_YEAR + FORECAST_HORIZON_YEARS),
        dtype=float,
    )

    df_eval, df_future = format_and_forecast(
        y_pred={"subs_pred": s_pred, "gwl_pred": h_pred},
        y_true=y_true_for_format,
        coords=X_fore.get("coords", None),
        quantiles=QUANTILES if QUANTILES else None,
        target_name=SUBSIDENCE_COL,
        scaler_target_name=SUBS_SCALER_KEY,
        output_target_name="subsidence",
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
        dataset_name=dataset_name,
        csv_eval_path=csv_eval,
        csv_future_path=csv_future,
        time_as_datetime=False,
        time_format=None,
        verbose=1,
        eval_metrics=True,
        metrics_quantile_interval=(0.1, 0.9),
        metrics_per_horizon=True,
        metrics_extra=["pss"],
        metrics_savefile=os.path.join(run_dir, "eval_diagnostics.json"),
        metrics_save_format=".json",
        metrics_time_as_str=True,
        value_mode="cumulative",
        input_value_mode="cumulative",
        output_unit="mm",
        output_unit_from="m",
        output_unit_mode="overwrite",
        output_unit_col="subsidence_unit",
    )

    try:
        _notify_gui_forecast_views(df_eval, df_future)
    except Exception:
        pass

    # ---------- evaluation + physics ----------
    ds_eval = make_tf_dataset(
        X_fore,
        y_fore,
        batch_size=BATCH_SIZE,
        shuffle=False,
        mode=MODE,
        forecast_horizon=FORECAST_HORIZON_YEARS,
    )

    if DEBUG:
        _ = debug_val_interval(
            model_inf,
            ds_eval,
            n_q=len(QUANTILES),
            max_batches=2,
            verbose=1,
        )

    # If loaded with compile=False, compile dummy for evaluate()
    if not USE_IN_MEMORY_MODEL:
        model_inf.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.0),
            loss=loss_arg,
            loss_weights=lossw_arg,
            metrics=metrics_compile,
            **physics_loss_weights,
        )

    eval_results = {}
    phys = {}
    try:
        eval_raw = model_inf.evaluate(ds_eval, return_dict=True, verbose=1)
        eval_results = _logs_to_py(eval_raw)
        phys_keys = ("epsilon_prior", "epsilon_cons", "epsilon_gw")
        phys = {
            k: float(_to_py(eval_raw[k]))
            for k in phys_keys
            if k in eval_results
        }
    except Exception as e:
        log(f"[Warn] evaluate() failed: {e}")

    phys_npz = os.path.join(run_dir, f"{CITY_NAME}_phys_payload_run_val.npz")
    try:
        _ = model_inf.export_physics_payload(
            ds_eval,
            max_batches=None,
            save_path=phys_npz,
            format="npz",
            overwrite=True,
            metadata={
                "city": CITY_NAME,
                "split": dataset_name,
                "time_units": TIME_UNITS,
            },
        )
    except Exception as e:
        log(f"[Warn] export_physics_payload failed: {e}")

    # Interval diagnostics (optional censoring)
    cov80_uncal = cov80_cal = None
    sharp80_uncal = sharp80_cal = None
    censor_metrics = None

    y_true_list = []
    s_q_list = []
    mask_list = []

    for xb, yb in with_progress(ds_eval, desc="Interval diagnostics"):
        out = model_inf(xb, training=False)
        s_pred_b, _ = extract_preds(model_inf, out)
        y_true_b = yb["subs_pred"]
        y_true_list.append(y_true_b)

        if QUANTILES:
            s_q_list.append(s_pred_b)

        if CENSOR_FLAG_IDX is not None:
            H = tf.shape(y_true_b)[1]
            mask_b = build_censor_mask(
                xb,
                H,
                CENSOR_FLAG_IDX,
                CENSOR_THRESH,
                source=CENSOR_MASK_SOURCE or "dynamic",
                reduce_time="any",
                align="broadcast",
            )
            mask_list.append(mask_b)

    y_true_t = tf.concat(y_true_list, axis=0) if y_true_list else None
    s_q_t = tf.concat(s_q_list, axis=0) if s_q_list else None
    mask_t = tf.concat(mask_list, axis=0) if mask_list else None

    s_q_cal = None
    if QUANTILES and y_true_t is not None and s_q_t is not None:
        cov80_uncal = float(coverage80_fn(y_true_t, s_q_t).numpy())
        sharp80_uncal = float(sharpness80_fn(y_true_t, s_q_t).numpy())
        s_q_cal = apply_calibrator_to_subs(cal80, s_q_t)
        cov80_cal = float(coverage80_fn(y_true_t, s_q_cal).numpy())
        sharp80_cal = float(sharpness80_fn(y_true_t, s_q_cal).numpy())

    if DEBUG and QUANTILES and y_true_t is not None and s_q_t is not None:
        _ = debug_tensor_interval(
            y_true_t,
            s_q_t,
            n_q=len(QUANTILES),
            name="RAW",
            verbose=1,
        )

    # Point metrics (physical units)
    metrics_point = {}
    per_h_mae = None
    per_h_r2 = None

    if y_true_t is not None:
        med_idx = None
        if QUANTILES:
            med_idx = int(
                np.argmin(np.abs(np.asarray(QUANTILES, dtype=float) - 0.5))
            )
            s_med = s_q_cal[..., med_idx, :] if s_q_cal is not None else s_q_t[..., med_idx, :]
        else:
            s_pred_list = []
            for xb, _ in ds_eval:
                out = model_inf(xb, training=False)
                s2 = subs_point_from_out(model_inf, out, QUANTILES, med_idx)
                s_pred_list.append(s2)
            s_med = tf.concat(s_pred_list, axis=0)

        metrics_point = point_metrics(
            y_true_t,
            s_med,
            use_physical=True,
            scaler_info=scaler_info_dict,
            target_name=SUBSIDENCE_COL,
        )
        per_h_mae, per_h_r2 = per_horizon_metrics(
            y_true_t,
            s_med,
            use_physical=True,
            scaler_info=scaler_info_dict,
            target_name=SUBSIDENCE_COL,
        )

        if mask_t is not None:
            y_true_phys_np = inverse_scale_target(
                y_true_t,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )
            s_med_phys_np = inverse_scale_target(
                s_med,
                scaler_info=scaler_info_dict,
                target_name=SUBSIDENCE_COL,
            )
            y_true_phys = tf.convert_to_tensor(y_true_phys_np, dtype=tf.float32)
            s_med_phys = tf.convert_to_tensor(s_med_phys_np, dtype=tf.float32)

            mask_f = tf.cast(mask_t, tf.float32)
            num_c = tf.reduce_sum(mask_f) + 1e-8
            num_u = tf.reduce_sum(1.0 - mask_f) + 1e-8

            abs_err = tf.abs(y_true_phys - s_med_phys)
            mae_c = tf.reduce_sum(abs_err * mask_f) / num_c
            mae_u = tf.reduce_sum(abs_err * (1.0 - mask_f)) / num_u

            censor_metrics = {
                "flag_name": CENSOR_FLAG_NAME,
                "threshold": float(CENSOR_THRESH),
                "mae_censored": float(mae_c.numpy()),
                "mae_uncensored": float(mae_u.numpy()),
            }

    payload = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "tf_version": tf.__version__,
        "numpy_version": np.__version__,
        "quantiles": QUANTILES,
        "horizon": int(FORECAST_HORIZON_YEARS),
        "batch_size": int(BATCH_SIZE),
        "metrics_evaluate": {
            k: _to_py(v) for k, v in (eval_results or {}).items()
        },
        "physics_diagnostics": phys,
        "point_metrics": metrics_point,
        "per_horizon": {
            "mae": per_h_mae,
            "r2": per_h_r2,
        },
        "interval_calibration": {
            "target": 0.80,
            "factors_per_horizon": (
                getattr(cal80, "factors_", None).tolist()
                if hasattr(cal80, "factors_") else None
            ),
            "coverage80_uncalibrated": cov80_uncal,
            "coverage80_calibrated": cov80_cal,
            "sharpness80_uncalibrated": sharp80_uncal,
            "sharpness80_calibrated": sharp80_cal,
        },
    }
    if censor_metrics is not None:
        payload["censor_stratified"] = censor_metrics

    units_mode = str(cfg.get("EVAL_JSON_UNITS_MODE", "si") or "si").strip().lower()
    units_scope = str(cfg.get("EVAL_JSON_UNITS_SCOPE", "all") or "all").strip().lower()
    try:
        payload = convert_eval_payload_units(
            payload,
            cfg,
            mode=units_mode,
            scope=units_scope,
        )
    except Exception as e:
        log(f"[Warn] unit conversion skipped: {e}")
    
    metrics_json = os.path.join(
        run_dir,
        f"geoprior_eval_phys_{payload['timestamp']}.json",
    )
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Ablation record
    ABLCFG = {
        "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
        "GEOPRIOR_USE_EFFECTIVE_H": GEOPRIOR_USE_EFFECTIVE_H,
        "GEOPRIOR_KAPPA_MODE": GEOPRIOR_KAPPA_MODE,
        "GEOPRIOR_HD_FACTOR": GEOPRIOR_HD_FACTOR,
        "LAMBDA_CONS": LAMBDA_CONS,
        "LAMBDA_GW": LAMBDA_GW,
        "LAMBDA_PRIOR": LAMBDA_PRIOR,
        "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
        "LAMBDA_BOUNDS": LAMBDA_BOUNDS,
        "LAMBDA_MV": LAMBDA_MV,
        "LAMBDA_Q": LAMBDA_Q,
    }
    save_ablation_record(
        outdir=run_dir,
        city=CITY_NAME,
        model_name=MODEL_NAME,
        cfg=ABLCFG,
        eval_dict={
            "r2": (metrics_point or {}).get("r2"),
            "mse": (metrics_point or {}).get("mse"),
            "mae": (metrics_point or {}).get("mae"),
            "coverage80": cov80_cal or cov80_uncal,
            "sharpness80": sharp80_cal or sharp80_uncal,
        },
        phys_diag=(phys or {}),
        per_h_mae=per_h_mae,
        per_h_r2=per_h_r2,
    )

    try:
        # 1) Spatial maps (needs coords from dataset)
        phys_payload, _ = load_physics_payload(phys_npz)
        plot_physics_values_in(
            phys_payload,
            dataset=ds_eval,
            keys=[
                "cons_res_vals",
                "log10_tau",
                "log10_tau_prior",
                "K",
                "Ss",
                "Hd",
            ],
            mode="map",
            transform=None,
            savefig=os.path.join(run_dir, "phys_maps.png"),
        )
        
        # 2) Residual distribution (no coords needed)
        plot_physics_values_in(
            phys_payload,
            keys=["cons_res_vals"],
            mode="hist",
            transform="signed_log10",
            savefig=os.path.join(run_dir, "cons_res_hist.png"),
        )
    except: 
        log("Failed to plot physic values in...")
    
    # Optional plots
    try:
        plot_eval_future(
            df_eval=df_eval,
            df_future=df_future,
            target_name=SUBSIDENCE_COL,
            quantiles=QUANTILES,
            spatial_cols=("coord_x", "coord_y"),
            time_col="coord_t",
            eval_years=[FORECAST_START_YEAR - 1],
            future_years=future_grid,
            eval_view_quantiles=[0.5],
            future_view_quantiles=QUANTILES,
            spatial_mode="hexbin",
            hexbin_gridsize=40,
            savefig_prefix=os.path.join(run_dir, f"{CITY_NAME}_subsidence_view"),
            save_fmts=[".png", ".pdf"],
            show=False,
            verbose=1,
        )
    except Exception as e:
        log(f"[Warn] plot_eval_future failed: {e}")

    try:
        save_all_figures(
            output_dir=run_dir,
            prefix=f"{CITY_NAME}_{MODEL_NAME}_plot_",
            fmts=[".png", ".pdf"],
        )
    except Exception:
        pass

    tf.keras.backend.clear_session()
    gc.collect()
    
 
    return {
        "status": "ok",
        "run_dir": run_dir,
        "training_summary_json": summary_json,
        "train_log_csv": csvlog_path,
        "metrics_json": metrics_json,
        "eval_csv": csv_eval,
        "future_csv": csv_future,
        "phys_payload_npz": phys_npz,
        "interval_factors_npy": cal_path,
        "best_keras": best_keras_path,
        "best_weights": best_weights_path,
        "best_tf_dir": best_tf_dir,
        "model_init_manifest": init_manifest_path,
        "run_manifest_json": manifest_path,
        "forecast_eval_csv": csv_eval,
        "forecast_future_csv": csv_future,
    }
