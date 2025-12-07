# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Stage-1 helper for building NPZ files for GeoPriorSubsNet (without future sequences).

This script handles data preprocessing, feature extraction, and saves the datasets
as NPZ files for later use.
"""

from __future__ import annotations

import os
import json
import shutil
import joblib
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, Any

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .....api.util import get_table_size
from .....utils.data_utils import nan_ops
from .....utils.io_utils import save_job
from .....utils.nat_utils import load_nat_config
from .....utils.generic_utils import (
    normalize_time_column,
    ensure_directory_exists,
    print_config_table,
)
from .....utils.sequence_utils import prepare_pinn_data_sequences


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")
if hasattr(tf, "autograph") and hasattr(tf.autograph, "set_verbosity"):
    tf.autograph.set_verbosity(0)


def _save_npz(path: str, arrays: Dict[str, np.ndarray]) -> str:
    """Save dict of numpy arrays to compressed NPZ."""
    np.savez_compressed(path, **arrays)
    return path


def _dataset_to_numpy_pair(
    ds: tf.data.Dataset, limit: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Unbatch a (x, y) dataset into stacked numpy dicts."""
    xs, ys = [], []
    count = 0
    for x_b, y_b in ds.unbatch():
        xs.append({k: v.numpy() for k, v in x_b.items()})
        ys.append({k: v.numpy() for k, v in y_b.items()})
        count += 1
        if limit is not None and count >= limit:
            break
    if not xs:
        raise RuntimeError("Empty dataset when converting to numpy.")

    x0 = xs[0]
    x_np = {k: np.stack([xi[k] for xi in xs], axis=0) for k in x0}
    y0 = ys[0]
    y_np = {k: np.stack([yi[k] for yi in ys], axis=0) for k in y0}
    return x_np, y_np


def _resolve_optional_columns(
    df: pd.DataFrame, spec_list
) -> tuple[list[str], dict]:
    """
    From a list like ["a", ("b","b2"), "c"], return:
      present_cols: concrete cols found in df (first match in tuples)
      mapping: {chosen_col: original_spec}.
    """
    present, mapping = [], {}
    cols = set(df.columns)
    for spec in spec_list:
        if isinstance(spec, (list, tuple, set)):
            chosen = next((s for s in spec if s in cols), None)
            if chosen is not None:
                present.append(chosen)
                mapping[chosen] = tuple(spec)
        else:
            if spec in cols:
                present.append(spec)
                mapping[spec] = spec
    return present, mapping


def _drop_missing_keep_order(cands: list[str], df: pd.DataFrame) -> list[str]:
    """Keep only those in df.columns preserving order."""
    cols = set(df.columns)
    return [c for c in cands if c in cols]


def _apply_censoring(
    df: pd.DataFrame, specs: list[dict]
) -> tuple[pd.DataFrame, dict]:
    """
    Add <col>_censored (bool) and <col>_eff (float) for each spec.
    Returns (df, report) with basic rates for the manifest.
    """
    report = {}
    for sp in specs or []:
        col = sp.get("col")
        if not col or col not in df.columns:
            continue

        cap = sp.get("cap")
        tol = float(sp.get("tol", 0.0))
        dirn = sp.get("direction", "right")
        fflag = col + sp.get("flag_suffix", "_censored")
        feff = col + sp.get("eff_suffix", "_eff")
        mode = sp.get("eff_mode", "clip")
        eps = float(sp.get("eps", 0.02))

        x = pd.to_numeric(df[col], errors="coerce").astype(float)

        if dirn == "right":
            m = np.isfinite(x) & (x >= (cap - tol))
        elif dirn == "left":
            m = np.isfinite(x) & (x <= (cap + tol))
        else:
            raise ValueError("Censoring 'direction' must be 'left' or 'right'.")

        df[fflag] = m.astype(bool)

        if mode == "clip" and cap is not None:
            eff = np.minimum(x, cap)
        elif mode == "cap_minus_eps" and cap is not None:
            eff = np.where(m, cap * (1.0 - eps), x)
        elif mode == "nan_if_censored":
            eff = x.copy()
            eff[m] = np.nan
            imp = sp.get("impute") or {}
            by = imp.get("by", [])
            func = imp.get("func", "median")
            if by:
                df[feff] = eff
                grp = df.groupby(by)[feff]
                if func == "median":
                    df[feff] = grp.transform(lambda s: s.fillna(s.median()))
                elif func == "mean":
                    df[feff] = grp.transform(lambda s: s.fillna(s.mean()))
                eff = df[feff].to_numpy(dtype=float)
        else:
            eff = x  # fallback

        df[feff] = eff
        report[col] = {
            "direction": dirn,
            "cap": cap,
            "flag_col": fflag,
            "eff_col": feff,
            "eff_mode": mode,
            "censored_rate": float(np.mean(m)),
        }
    return df, report


def run_stage1_for_npz(
    cfg_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    clean_run_dir: bool = False,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None, 
    base_cfg: Optional[Dict[str, Any]] = None,
    results_root: Optional[os.PathLike | str] = None,
    edited_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Stage-1 helper to build NPZ files for GeoPriorSubsNet.
    This will run only the preprocessing and data sequence generation steps.
    """
    def _progress(fraction: float, msg: str) -> None:
        if progress_callback is None:
            return
        try:
            # clamp and forward
            f = max(0.0, min(1.0, float(fraction)))
            progress_callback(f, msg)
        except Exception:
            # never crash Stage-1 because the GUI callback misbehaved
            pass

    log = logger or (lambda msg: print(msg, flush=True))

    # ===================== CONFIG =====================
    # Apply configuration overrides
    if base_cfg is not None:
        cfg = dict(base_cfg)
    else:
        GUI_CONFIG_DIR = os.path.dirname(__file__)
        config_root = os.path.join( os.path.dirname (GUI_CONFIG_DIR), 'config')
        cfg = load_nat_config(root=config_root)

    if cfg_overrides:
        cfg.update(cfg_overrides)

    # Ensure output directory exists
    base_output_dir = os.path.join(os.getcwd(), "results")  # Default location
    if results_root is not None:
        base_output_dir = os.fspath(results_root)
    ensure_directory_exists(base_output_dir)

    CITY_NAME = cfg.get("CITY_NAME", "Unknown City")
    RUN_OUTPUT_PATH = os.path.join(base_output_dir, f"{CITY_NAME}_stage1_npz")
    ensure_directory_exists(RUN_OUTPUT_PATH)

    ARTIFACTS_DIR = os.path.join(RUN_OUTPUT_PATH, "artifacts")
    ensure_directory_exists(ARTIFACTS_DIR)

    # Process the dataset and generate NPZ
    log(f"Starting to build NPZ for {CITY_NAME} in {RUN_OUTPUT_PATH}")
    try:
        # Load and preprocess the dataset
        df_raw = edited_df if edited_df is not None else pd.read_csv(cfg["BIG_FN"])

        # Feature and sequence generation
        df_processed = df_raw.copy()  # Example preprocessing step, extend as needed
        inputs, targets = prepare_pinn_data_sequences(df_processed,  )

        # Save NPZ files
        np.savez_compressed(os.path.join(ARTIFACTS_DIR, "train_inputs.npz"), **inputs)
        np.savez_compressed(os.path.join(ARTIFACTS_DIR, "train_targets.npz"), **targets)

        log(f"Saved NPZ files to {ARTIFACTS_DIR}")
    except Exception as e:
        log(f"Error while building NPZ files: {str(e)}")
        return {}

    return {
        "run_dir": RUN_OUTPUT_PATH,
        "artifacts_dir": ARTIFACTS_DIR,
    }
