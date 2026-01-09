# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Utilities for computing error metrics in *physical* units given
Stage-1 scaling metadata.

Main entry points
-----------------
- inverse_scale_target(...)
- point_metrics(...)
- per_horizon_metrics(...)

These are designed to work with the NATCOM pipeline where Stage-1
stores a ``scaler_info`` dict in the manifest:

    scaler_info[target_name] = {
        "scaler_path": ".../main_scaler.joblib",
        "all_features": [...],
        "idx": <index in scaler>,
        "scaler": <fitted MinMaxScaler>  # optional, attached at Stage-2
    }

But they are also flexible enough to handle a bare scaler object,
a scaler path, or manual min/max / mean/std parameters.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import joblib


ArrayLike = Any


def _to_np(x: ArrayLike) -> np.ndarray:
    """Convert Tensor / list-like to a NumPy array."""
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _resolve_stage1_entry(
    scaler_info: Mapping[str, Any] | None,
    target_name: str | None,
    scaler_entry: Mapping[str, Any] | None,
) -> Tuple[Optional[Any], Optional[int], Optional[int]]:
    """
    Resolve (scaler, feature_index, n_features) from Stage-1 metadata.

    Parameters
    ----------
    scaler_info : mapping or None
        Either the full Stage-1 scaler_info dict mapping
        ``feature_name -> entry`` or a single entry.
    target_name : str, optional
        Name of the target in ``scaler_info``.
    scaler_entry : mapping, optional
        Direct entry for the target.

    Returns
    -------
    scaler : object or None
    idx : int or None
        Index of the target in the scaler's feature space.
    n_features : int or None
        Number of features expected by the scaler.
    """
    entry: Mapping[str, Any] | None = None

    # 0) If caller provided a direct entry, trust it.
    if scaler_entry is not None:
        entry = scaler_entry

    # 1) If scaler_info itself looks like a single entry, use it.
    elif isinstance(scaler_info, Mapping) and (
        "idx" in scaler_info or "scaler_path" in scaler_info or "scaler" in scaler_info
    ):
        entry = scaler_info  # already an entry


    # 2) Otherwise, scaler_info is expected to be name -> entry mapping.
    elif isinstance(scaler_info, Mapping) and isinstance(target_name, str):
        # direct hit
        if target_name in scaler_info:
            entry = scaler_info[target_name]
        else:
            # case-insensitive + aliases
            keymap = {k.lower(): k for k in scaler_info.keys() if isinstance(k, str)}
            lname = target_name.lower()

            aliases: list[str] = []
            if lname.endswith("_pred"):
                aliases.append(lname[:-5])

            if "subs" in lname:
                aliases += [
                    "subsidence", "subs", "subs_pred",
                    "subsidence_cum", "subsidence_cum__si"
                ]
            if "gwl" in lname or "head" in lname:
                aliases += [
                    "gwl", "gwl_pred", 
                    "gwl_depth_bgs", "gwl_depth_bgs__si", 
                    "head_m", "head_m__si",
                    ]
                
            # de-dup preserve order
            seen = set()
            aliases = [a for a in aliases if not (a in seen or seen.add(a))]
            
            # try aliases (case-insensitive)
            for a in aliases:
                real = keymap.get(a.lower())
                if real is not None:
                    entry = scaler_info[real]
                    break

    if entry is None:
        return None, None, None

    # 3) Load scaler if needed
    scaler = entry.get("scaler")
    if scaler is None and isinstance(entry.get("scaler_path"), str):
        try:
            scaler = joblib.load(entry["scaler_path"])
        except Exception:
            scaler = None

    idx = entry.get("idx")
    n_features = None

    all_feats = entry.get("all_features")
    if isinstance(all_feats, (list, tuple)):
        n_features = len(all_feats)

    if hasattr(scaler, "n_features_in_"):
        try:
            n_features = int(getattr(scaler, "n_features_in_"))
        except Exception:
            pass

    if n_features is None and idx is not None:
        n_features = int(idx) + 1

    return scaler, idx, n_features

def inverse_scale_target(
    y_scaled: ArrayLike,
    *,
    scaler_info: Mapping[str, Any] | None = None,
    target_name: str | None = None,
    scaler_entry: Mapping[str, Any] | None = None,
    scaler: Any | None = None,
    feature_index: int | None = None,
    n_features: int | None = None,
    params: Mapping[str, float] | None = None,
) -> np.ndarray:
    r"""
    Inverse-transform a scaled target array back to physical units.

    Supports three patterns:

    1. Stage-1 ``scaler_info`` dict (preferred in NATCOM):
       pass ``scaler_info=scaler_info_dict, target_name="subsidence"``.
    2. A bare scaler instance or a path to a joblib dump via ``scaler=...``.
       If multi-feature, also pass ``feature_index`` and optionally
       ``n_features``.
    3. Manual scaling parameters via ``params`` such as
       {\"min\", \"max\"}, {\"mean\", \"std\"}, or {\"scale\", \"shift\"}.

    Parameters
    ----------
    y_scaled : array-like
        Scaled values, e.g. of shape (N, H, 1) or (N,).
    scaler_info : mapping, optional
    target_name : str, optional
    scaler_entry : mapping, optional
    scaler : object or str, optional
    feature_index : int, optional
    n_features : int, optional
    params : mapping, optional
        Manual parameters:
        - {\"min\", \"max\"}  → MinMax scaling
        - {\"mean\", \"std\"} → standardization
        - {\"scale\", \"shift\"} → x = scale * x_scaled + shift.

    Returns
    -------
    np.ndarray
        Array with same shape as ``y_scaled`` but in physical units when
        scaling information is available. If nothing usable is found,
        returns the input as a NumPy array.
    """
    y = _to_np(y_scaled)
    original_shape = y.shape
    y_flat = y.reshape(-1, 1)

    # ---- 1) Manual params ------------------------------------------------
    if params:
        keys = set(params.keys())
        if {"min", "max"} <= keys:
            _min = float(params["min"])
            _max = float(params["max"])
            span = (_max - _min) if (_max - _min) != 0.0 else 1.0
            y_phys = y_flat * span + _min
            return y_phys.reshape(original_shape)

        if {"mean", "std"} <= keys:
            mean = float(params["mean"])
            std = float(params["std"])
            if std == 0.0:
                std = 1.0
            y_phys = y_flat * std + mean
            return y_phys.reshape(original_shape)

        if {"scale", "shift"} <= keys:
            scale = float(params.get("scale", 1.0))
            shift = float(params.get("shift", 0.0))
            y_phys = y_flat * scale + shift
            return y_phys.reshape(original_shape)

        # Unknown combination → fall through
        return y

    # ---- 2) Stage-1 metadata ---------------------------------------------
    stg_scaler, stg_idx, stg_nfeat = _resolve_stage1_entry(
        scaler_info=scaler_info,
        target_name=target_name,
        scaler_entry=scaler_entry,
    )

    if stg_scaler is not None:
        scaler = stg_scaler
        feature_index = stg_idx if feature_index is None else feature_index
        n_features = stg_nfeat if stg_nfeat is not None else n_features

    # ---- 3) Bare scaler or path ------------------------------------------
    if isinstance(scaler, str):
        try:
            scaler = joblib.load(scaler)
        except Exception:
            scaler = None

    if scaler is not None and hasattr(scaler, "inverse_transform"):
        if n_features is None and hasattr(scaler, "n_features_in_"):
            try:
                n_features = int(getattr(scaler, "n_features_in_"))
            except Exception:
                n_features = None

        if n_features is None:
            n_features = 1
        if feature_index is None:
            feature_index = 0

        X = np.zeros((y_flat.shape[0], int(n_features)), dtype=float)
        fi = int(feature_index)
        X[:, fi : fi + 1] = y_flat

        try:
            X_inv = scaler.inverse_transform(X)
            y_phys = X_inv[:, fi : fi + 1]
            return y_phys.reshape(original_shape)
        except Exception:
            return y

    # ---- 4) No usable scaling info --------------------------------------
    return y


def point_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    use_physical: bool = False,
    scaler_info: Mapping[str, Any] | None = None,
    target_name: str | None = None,
    scaler_entry: Mapping[str, Any] | None = None,
    scaler: Any | None = None,
    feature_index: int | None = None,
    n_features: int | None = None,
    params: Mapping[str, float] | None = None,
) -> dict:
    """
    Compute MAE/MSE/R², optionally in physical units.

    If ``use_physical=True``, both y_true and y_pred are inverse-
    transformed via inverse_scale_target before computing metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
        Arbitrary shapes (e.g. (N, H, 1)); flattened internally.
    use_physical : bool, default=False
    scaler_info, target_name, scaler_entry, scaler, feature_index,
    n_features, params :
        Passed through to inverse_scale_target.

    Returns
    -------
    dict
        {\"mae\": ..., \"mse\": ..., \"r2\": ...}
    """
    yt = _to_np(y_true)
    yp = _to_np(y_pred)

    if use_physical:
        yt = inverse_scale_target(
            yt,
            scaler_info=scaler_info,
            target_name=target_name,
            scaler_entry=scaler_entry,
            scaler=scaler,
            feature_index=feature_index,
            n_features=n_features,
            params=params,
        )
        yp = inverse_scale_target(
            yp,
            scaler_info=scaler_info,
            target_name=target_name,
            scaler_entry=scaler_entry,
            scaler=scaler,
            feature_index=feature_index,
            n_features=n_features,
            params=params,
        )

    yt_f = yt.reshape(-1)
    yp_f = yp.reshape(-1)
    diff = yt_f - yp_f

    mae = float(np.nanmean(np.abs(diff)))
    mse = float(np.nanmean(diff ** 2))
    var = float(np.nanvar(yt_f))
    if var > 0.0 and np.isfinite(var):
        r2 = float(1.0 - mse / (var + 1e-12))
    else:
        r2 = float("nan")

    return {"mae": mae, "mse": mse, "r2": r2}


def per_horizon_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    use_physical: bool = False,
    scaler_info: Mapping[str, Any] | None = None,
    target_name: str | None = None,
    scaler_entry: Mapping[str, Any] | None = None,
    scaler: Any | None = None,
    feature_index: int | None = None,
    n_features: int | None = None,
    params: Mapping[str, float] | None = None,
) -> tuple[dict, dict]:
    """
    Per-horizon MAE/R².

    Parameters
    ----------
    y_true, y_pred : array-like
        Shape (N, H, 1) or (N, H).
    use_physical : bool, default=False
        If True, inverse-transform to physical units before computing.

    Returns
    -------
    mae_dict : dict
        {'H1': mae_h1, ...}
    r2_dict : dict
        {'H1': r2_h1, ...}
    """
    Y = _to_np(y_true)
    P = _to_np(y_pred)

    if use_physical:
        Y = inverse_scale_target(
            Y,
            scaler_info=scaler_info,
            target_name=target_name,
            scaler_entry=scaler_entry,
            scaler=scaler,
            feature_index=feature_index,
            n_features=n_features,
            params=params,
        )
        P = inverse_scale_target(
            P,
            scaler_info=scaler_info,
            target_name=target_name,
            scaler_entry=scaler_entry,
            scaler=scaler,
            feature_index=feature_index,
            n_features=n_features,
            params=params,
        )

    Y = np.asarray(Y).squeeze(-1)  # (N, H)
    P = np.asarray(P).squeeze(-1)

    H = Y.shape[1]
    mae_dict: dict[str, float | None] = {}
    r2_dict: dict[str, float | None] = {}

    for h in range(H):
        yy = Y[:, h]
        pp = P[:, h]
        mask = np.isfinite(yy) & np.isfinite(pp)
        if mask.sum() == 0:
            mae_dict[f"H{h+1}"] = None
            r2_dict[f"H{h+1}"] = None
            continue

        d = yy[mask] - pp[mask]
        mae = float(np.mean(np.abs(d)))
        mse = float(np.mean(d ** 2))
        var = float(np.var(yy[mask]))
        r2 = float(1.0 - mse / (var + 1e-12)) if var > 0 else float("nan")

        mae_dict[f"H{h+1}"] = mae
        r2_dict[f"H{h+1}"] = r2

    return mae_dict, r2_dict
