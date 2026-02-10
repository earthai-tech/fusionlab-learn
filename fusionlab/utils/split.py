# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
fusionlab.utils.split

Group-holdout split for sequence data.

Exports:
  train_windows_T{T}_H{H}.npz
  val_windows_T{T}_H{H}.npz
  test_windows_T{T}_H{H}.npz
  future_inputs_T{T}_H{H}.npz
  splits_groups.json

Leakage fix (Zhongshan 2 windows/pixel):
  split by group_id first, then window inside split.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .panel_cache import (
    make_group_keys,
    min_required_len,
)
from .sequence_utils import (
    build_future_sequences_npz,
)

__all__ = [
    "SplitCfg",
    "split_group_keys",
    "subset_by_keys",
    "write_splits_json",
    "pack_xy_npz",
    "build_group_holdout_npzs",
    "build_future_inputs_npz",
]


@dataclass(frozen=True)
class SplitCfg:
    seed: int = 42
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    decimals: int = 8


def _check_ratios(r: Tuple[float, float, float]) -> None:
    if len(r) != 3:
        raise ValueError("ratios must be (train,val,test).")
    s = float(r[0]) + float(r[1]) + float(r[2])
    if not np.isclose(s, 1.0, atol=1e-8):
        raise ValueError("ratios must sum to 1.0.")


def split_group_keys(
    keys: np.ndarray,
    *,
    cfg: SplitCfg = SplitCfg(),
) -> Dict[str, np.ndarray]:
    _check_ratios(cfg.ratios)

    keys = np.asarray(keys, dtype=object)
    if keys.size < 3:
        raise ValueError("Need >= 3 groups for splits.")

    rng = np.random.default_rng(int(cfg.seed))
    keys = keys[rng.permutation(keys.size)]

    n = int(keys.size)
    n_tr = int(np.floor(cfg.ratios[0] * n))
    n_va = int(np.floor(cfg.ratios[1] * n))
    n_tr = max(1, min(n_tr, n - 2))
    n_va = max(1, min(n_va, n - n_tr - 1))

    tr = keys[:n_tr]
    va = keys[n_tr : n_tr + n_va]
    te = keys[n_tr + n_va :]

    return {"train": tr, "val": va, "test": te}


def subset_by_keys(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    keys: np.ndarray,
    decimals: int = 8,
) -> pd.DataFrame:
    if keys is None or len(keys) == 0:
        return df.iloc[0:0].copy()

    k = make_group_keys(
        df,
        group_cols=group_cols,
        decimals=decimals,
    )
    keep = k.isin(set(map(str, keys)))
    return df.loc[keep].copy()


def write_splits_json(
    path: str,
    *,
    group_cols: Sequence[str],
    time_steps: int,
    horizon: int,
    train_end: Optional[float],
    cfg: SplitCfg,
    splits: Dict[str, np.ndarray],
) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload: Dict[str, Any] = {
        "group_cols": list(group_cols),
        "decimals": int(cfg.decimals),
        "seed": int(cfg.seed),
        "ratios": list(map(float, cfg.ratios)),
        "time_steps": int(time_steps),
        "horizon": int(horizon),
        "min_len": int(min_required_len(time_steps, horizon)),
        "train_end": train_end,
        "counts": {k: int(len(v)) for k, v in splits.items()},
        "splits": {k: list(map(str, v)) for k, v in splits.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def _pick_target_key(
    d: Dict[str, Any],
    *names: str,
) -> str:
    for n in names:
        if n in d:
            return n
    raise KeyError(f"Missing target key in {list(d)}")


def pack_xy_npz(
    x: Dict[str, Any],
    y: Optional[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {
        "coords": np.asarray(x["coords"], np.float32),
        "dynamic_features": np.asarray(
            x["dynamic_features"],
            np.float32,
        ),
        "static_features": np.asarray(
            x.get("static_features", np.zeros((0, 0))),
            np.float32,
        ),
        "future_features": np.asarray(
            x.get("future_features", np.zeros((0, 0, 0))),
            np.float32,
        ),
        "H_field": np.asarray(x["H_field"], np.float32),
    }

    if y is None:
        return out

    s_key = _pick_target_key(y, "subs_pred", "subsidence")
    g_key = _pick_target_key(y, "gwl_pred", "gwl")

    out["subs_pred"] = np.asarray(y[s_key], np.float32)
    out["gwl_pred"] = np.asarray(y[g_key], np.float32)
    return out


def _save_npz(path: str, arrays: Dict[str, np.ndarray]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def build_group_holdout_npzs(
    *,
    df_train: pd.DataFrame,
    artifacts_dir: str,
    group_cols: Sequence[str],
    time_col_used: str,
    x_col_used: str,
    y_col_used: str,
    subs_col: str,
    gwl_target_col: str,
    gwl_dyn_col: str,
    h_field_col: str,
    static_cols: Sequence[str],
    dynamic_cols: Sequence[str],
    future_cols: Sequence[str],
    time_steps: int,
    horizon: int,
    mode: str,
    model_name: str,
    train_end: Optional[float],
    keys_ok: np.ndarray,
    cfg: SplitCfg = SplitCfg(),
    normalize_coords: bool = True,
) -> Dict[str, Any]:
    """
    Build train/val/test windows using group holdout.

    Returns dict containing paths and coord_scaler.
    """
    from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences

    splits = split_group_keys(keys_ok, cfg=cfg)

    spath = os.path.join(artifacts_dir, "splits_groups.json")
    write_splits_json(
        spath,
        group_cols=group_cols,
        time_steps=time_steps,
        horizon=horizon,
        train_end=train_end,
        cfg=cfg,
        splits=splits,
    )

    coord_scaler = None
    out: Dict[str, Any] = {"splits_groups_json": spath}

    def _one(which: str) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
        dfi = subset_by_keys(
            df_train,
            group_cols=group_cols,
            keys=splits[which],
            decimals=cfg.decimals,
        )
        if dfi.empty:
            raise RuntimeError(f"{which} split is empty.")

        fit_cs = bool(normalize_coords) and (which == "train")
        ins, tar, cs = prepare_pinn_data_sequences(
            df=dfi,
            time_col=time_col_used,
            lon_col=x_col_used,
            lat_col=y_col_used,
            subsidence_col=subs_col,
            gwl_col=gwl_target_col,
            gwl_dyn_col=gwl_dyn_col,
            h_field_col=h_field_col,
            dynamic_cols=list(dynamic_cols),
            static_cols=list(static_cols),
            future_cols=list(future_cols),
            group_id_cols=list(group_cols),
            time_steps=int(time_steps),
            forecast_horizon=int(horizon),
            output_subsidence_dim=1,
            output_gwl_dim=1,
            normalize_coords=bool(normalize_coords),
            coord_scaler=coord_scaler,
            fit_coord_scaler=bool(fit_cs),
            return_coord_scaler=True,
            mode=str(mode),
            model=str(model_name),
            verbose=1,
        )
        return ins, tar, cs

    for which in ("train", "val", "test"):
        ins, tar, cs = _one(which)
        if which == "train":
            coord_scaler = cs

        arrays = pack_xy_npz(ins, tar)
        fn = f"{which}_windows_T{time_steps}_H{horizon}.npz"
        p = os.path.join(artifacts_dir, fn)
        out[f"{which}_windows_npz"] = _save_npz(p, arrays)

    out["coord_scaler"] = coord_scaler
    return out


def build_future_inputs_npz(
    *,
    df_scaled: pd.DataFrame,
    artifacts_dir: str,
    time_col: str,
    time_col_num: Optional[str],
    lon_col: str,
    lat_col: str,
    subs_col: str,
    gwl_col: str,
    h_field_col: str,
    static_features: Sequence[str],
    dynamic_features: Sequence[str],
    future_features: Sequence[str],
    group_cols: Sequence[str],
    train_end_time: Any,
    forecast_start_time: Any,
    horizon: int,
    time_steps: int,
    mode: str,
    model_name: str,
    normalize_coords: bool,
    coord_scaler: Any = None,
) -> str:


    prefix = f"future_T{time_steps}_H{horizon}"
    res = build_future_sequences_npz(
        df_scaled=df_scaled,
        time_col=time_col,
        time_col_num=time_col_num,
        lon_col=lon_col,
        lat_col=lat_col,
        subs_col=subs_col,
        gwl_col=gwl_col,
        h_field_col=h_field_col,
        static_features=list(static_features),
        dynamic_features=list(dynamic_features),
        future_features=list(future_features),
        group_id_cols=list(group_cols),
        train_end_time=train_end_time,
        forecast_start_time=forecast_start_time,
        forecast_horizon=int(horizon),
        time_steps=int(time_steps),
        mode=str(mode),
        model_name=str(model_name),
        artifacts_dir=artifacts_dir,
        prefix=prefix,
        normalize_coords=bool(normalize_coords),
        coord_scaler=coord_scaler,
        verbose=1,
    )

    src = res[f"{prefix}_inputs_npz"]
    dst = os.path.join(
        artifacts_dir,
        f"future_inputs_T{time_steps}_H{horizon}.npz",
    )
    shutil.move(src, dst)
    return dst

# from fusionlab.utils.panel_cache import (
#     ensure_feasible_keys_cache,
# )

# df_min = pd.read_csv(
#     used_path,
#     usecols=[LON_COL, LAT_COL, TIME_COL],
# )

# paths = ensure_feasible_keys_cache(
#     df_min,
#     out_dir=ARTIFACTS_DIR,
#     city=CITY_NAME,
#     group_cols=[LON_COL, LAT_COL],
#     time_col=TIME_COL,
#     train_end=TRAIN_END_YEAR,
#     time_steps=TIME_STEPS,
#     horizon=FORECAST_HORIZON_YEARS,
# )

# keys_ok = np.load(paths.keys_npy, allow_pickle=True)
# from fusionlab.utils.split import subset_by_keys

# df_clean = subset_by_keys(
#     df_clean,
#     group_cols=[LON_COL, LAT_COL],
#     keys=keys_ok,
#     decimals=8,
# )
# from fusionlab.utils.split import (
#     SplitCfg,
#     build_group_holdout_npzs,
#     build_future_inputs_npz,
# )

# cfg_split = SplitCfg(seed=42, ratios=(0.7, 0.15, 0.15))

# res = build_group_holdout_npzs(
#     df_train=df_train,
#     artifacts_dir=ARTIFACTS_DIR,
#     group_cols=[LON_COL, LAT_COL],
#     time_col_used=TIME_COL_USED,
#     x_col_used=X_COL_USED,
#     y_col_used=Y_COL_USED,
#     subs_col=SUBS_MODEL_COL,
#     gwl_target_col=GWL_TARGET_COL,
#     gwl_dyn_col=GWL_DYN_COL,
#     h_field_col=H_FIELD_COL,
#     static_cols=static_features,
#     dynamic_cols=dynamic_features,
#     future_cols=future_features,
#     time_steps=TIME_STEPS,
#     horizon=FORECAST_HORIZON_YEARS,
#     mode=MODE,
#     model_name=MODEL_NAME,
#     train_end=TRAIN_END_YEAR,
#     keys_ok=keys_ok,
#     cfg=cfg_split,
#     normalize_coords=normalize_coords,
# )

# future_npz = build_future_inputs_npz(
#     df_scaled=df_scaled,
#     artifacts_dir=ARTIFACTS_DIR,
#     time_col=TIME_COL,
#     time_col_num=TIME_COL_NUM,
#     lon_col=COORD_X_COL,
#     lat_col=COORD_Y_COL,
#     subs_col=SUBS_MODEL_COL,
#     gwl_col=GWL_TARGET_COL,
#     h_field_col=H_FIELD_COL,
#     static_features=static_features,
#     dynamic_features=dynamic_features,
#     future_features=future_features,
#     group_cols=[LON_COL, LAT_COL],
#     train_end_time=TRAIN_END_YEAR,
#     forecast_start_time=FORECAST_START_YEAR,
#     horizon=FORECAST_HORIZON_YEARS,
#     time_steps=TIME_STEPS,
#     mode=MODE,
#     model_name=MODEL_NAME,
#     normalize_coords=normalize_coords,
#     coord_scaler=res["coord_scaler"],
# )
