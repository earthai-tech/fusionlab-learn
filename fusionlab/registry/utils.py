# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com> 

from __future__ import annotations

import os
import json
import glob
import datetime as dt
import warnings
from typing import Dict, List
import numpy as np
import joblib
from typing import Callable, Optional, Sequence, Tuple, Union

__all__ = ["find_manifest", "_find_stage1_manifest", "reproject_dynamic_scale"]


def reproject_dynamic_scale(
    X_np: Dict[str, np.ndarray],
    target_scaler_info: Dict,      # from *target* manifest["artifacts"]["encoders"]["scaler_info"]
    source_scaler_path: str,       # from *source* manifest["artifacts"]["encoders"]["main_scaler"]
    dynamic_feature_order: List[str],
) -> Dict[str, np.ndarray]:
    """
    Re-normalize the target city's dynamic features to the source city's
    scaling, without touching raw CSVs.

    Steps
    -----
    1) Inverse-transform each dynamic channel from target scale back to its
       physical units (using target scaler `min_`/`scale_`).
    2) Transform those physical values with the *source* MinMax scaler
       (apply source `min_`/`scale_`).

    This is useful for strict cross-domain testing where inputs to the model
    should be on the *source* normalization even when evaluating on the
    *target* city.

    Parameters
    ----------
    X_np : dict
        Dict of Stage-1 arrays (NPZ-like) containing the key
        ``"dynamic_features"`` with shape ``(N, T, D)``.
    target_scaler_info : dict
        Scaler metadata from the *target* manifest. It must contain:
        - a nested dict that has either:
            {"scaler": MinMaxScaler, "all_features": [...]}  or
            {"scaler_path": "...joblib", "all_features": [...]}
          Many pipelines store this under keys like 'subsidence' or 'GWL'.
          This function auto-detects the first suitable block.
    source_scaler_path : str
        Path to the *source* city's MinMaxScaler joblib file
        (manifest["artifacts"]["encoders"]["main_scaler"]).
    dynamic_feature_order : list of str
        Column names in the exact order of the dynamic feature channels
        (i.e., the D last-dimension of ``X_np["dynamic_features"]``).

    Returns
    -------
    X_out : dict
        Shallow copy of `X_np` with ``"dynamic_features"`` replaced by the
        reprojected array in the *source* city's scaling.

    Notes
    -----
    * Assumes MinMaxScaler(feature_range=(0, 1)).
    * We assume the *source* scaler was trained on (approximately) the same
      feature set as the *target* scaler. If orders differ, we **map** by
      name using the `all_features` list from the *target* info. If the
      source scaler used a different feature set/order, you should keep a
      sidecar file recording the feature list for the source scaler and
      load it similarly (or pass a harmonized list here).
    * Channels not found in the target/source feature list are **passed
      through unchanged** with a warning.

    Examples
    --------
    >>> X2 = reproject_dynamic_scale(
    ...     X_np=X,
    ...     target_scaler_info=manifest_B["artifacts"]["encoders"]["scaler_info"],
    ...     source_scaler_path=manifest_A["artifacts"]["encoders"]["main_scaler"],
    ...     dynamic_feature_order=["GWL", "rainfall_mm", "normalized_urban_load_proxy"]
    ... )
    """
    if "dynamic_features" not in X_np:
        raise KeyError("X_np must contain 'dynamic_features' (N, T, D).")
    dyn = X_np["dynamic_features"]
    if dyn is None or dyn.size == 0:
        return dict(X_np)  # nothing to do

    if dyn.ndim != 3:
        raise ValueError("X_np['dynamic_features'] must be 3D: (N, T, D).")
    N, T, D = dyn.shape
    if len(dynamic_feature_order) != D:
        raise ValueError(
            "Length of dynamic_feature_order must match channels D "
            f"(got {len(dynamic_feature_order)} vs D={D})."
        )

    # -------- extract target scaler + feature list --------
    def _extract_block(si: Dict):
        # Accept a top-level block, or the first nested block with keys present
        if si and isinstance(si, dict) and ("all_features" in si) and (
            "scaler" in si or "scaler_path" in si
        ):
            return si
        if isinstance(si, dict):
            for v in si.values():
                if isinstance(v, dict) and ("all_features" in v) and (
                    "scaler" in v or "scaler_path" in v
                ):
                    return v
        return None

    blk_t = _extract_block(target_scaler_info)
    if blk_t is None:
        raise ValueError(
            "target_scaler_info does not contain a usable block with "
            "('all_features' and 'scaler' or 'scaler_path')."
        )
    t_all = blk_t.get("all_features", None)
    if not t_all or not isinstance(t_all, (list, tuple)):
        raise ValueError("target_scaler_info must provide 'all_features' list.")

    t_scaler = blk_t.get("scaler", None)
    if t_scaler is None:
        t_path = blk_t.get("scaler_path", None)
        if not t_path:
            raise ValueError("No 'scaler' or 'scaler_path' in target_scaler_info.")
        t_scaler = joblib.load(t_path)

    # -------- load source scaler --------
    s_scaler = joblib.load(source_scaler_path)

    # Safety checks: scalers must expose min_/scale_
    for name, sc in (("target", t_scaler), ("source", s_scaler)):
        for attr in ("min_", "scale_"):
            if not hasattr(sc, attr):
                raise AttributeError(f"{name} scaler missing attribute '{attr}'.")

    # Build per-channel transform params by name
    # We map names -> index in target all_features; assume source uses the same
    # feature ordering. If not, users should keep a source 'all_features' list
    # and adapt below accordingly.
    name_to_idx_t = {n: i for i, n in enumerate(t_all)}

    # Prepare output array (copy)
    dyn_out = dyn.astype(np.float32, copy=True)

    # Vectorized per-channel transform
    for ch, fname in enumerate(dynamic_feature_order):
        idx_t = name_to_idx_t.get(fname, None)
        if idx_t is None:
            warnings.warn(
                f"[reproject_dynamic_scale] Feature '{fname}' not found in "
                "target 'all_features'. Channel left unchanged."
            )
            continue

        # target inverse: x = (x_t - min_t) / scale_t
        min_t = float(t_scaler.min_[idx_t])
        sc_t  = float(t_scaler.scale_[idx_t])
        if sc_t == 0.0:
            warnings.warn(
                f"[reproject_dynamic_scale] target scale_==0 for '{fname}'. "
                "Channel left unchanged."
            )
            continue

        # source forward: x_s = x * scale_s + min_s
        # We assume source feature order ~ target order.
        idx_s = idx_t
        min_s = float(s_scaler.min_[idx_s])
        sc_s  = float(s_scaler.scale_[idx_s])
        if sc_s == 0.0:
            warnings.warn(
                f"[reproject_dynamic_scale] source scale_==0 for '{fname}'. "
                "Channel left unchanged."
            )
            continue

        # Do both steps on the (N, T) slice for this channel
        x_tilde = dyn_out[:, :, ch]
        x_phys  = (x_tilde - min_t) / sc_t
        x_src   = x_phys * sc_s + min_s
        dyn_out[:, :, ch] = x_src

    X_out = dict(X_np)
    X_out["dynamic_features"] = dyn_out
    return X_out

def _parse_manifest_ts(value: str | None) -> Optional[float]:
    """Parse a manifest timestamp to POSIX seconds if possible."""
    if not value:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return dt.datetime.strptime(value, fmt).timestamp()
        except Exception:
            continue
    return None


def find_manifest(
    manual: Optional[str] = None,
    base_dir: str = "results",
    city_hint: Optional[str] = None,
    model_hint: Optional[str] = None,
    stage_hint: Optional[str] = None,
    manifest_filename: Union[str, Sequence[str]] = ("manifest.json",),
    search_globs: Optional[Sequence[str]] = None,
    recursive: bool = True,
    return_all: bool = False,
    prefer: str = "mtime",
    filter_fn: Optional[Callable[[dict, str], bool]] = None,
    required_keys: Sequence[str] = (),
    verbose: int = 1,
) -> Union[str, list[str]]:
    """
    Locate a pipeline manifest JSON on disk with flexible discovery
    rules.

    The function follows a priority order:
      1) If ``manual`` path is given and exists, return it.
      2) If ``city_hint``/``model_hint``/``stage_hint`` are given,
         try deterministic folder patterns under ``base_dir``.
      3) Otherwise, glob-search candidate manifest files under
         ``base_dir`` and select the newest according to ``prefer``.

    Parameters
    ----------
    manual : str or None, optional
        Explicit manifest path. If provided and exists, it is
        returned immediately. If provided but missing, a
        ``FileNotFoundError`` is raised.

    base_dir : str, default="results"
        Root directory where experiment folders live.

    city_hint : str or None, optional
        City or dataset label to filter manifests, e.g. "nansha"
        or "zhongshan". Case-insensitive. If None, no city
        filtering is applied.

    model_hint : str or None, optional
        Model name to filter manifests (e.g., "GeoPriorSubsNet").
        If None, no model filtering is applied.

    stage_hint : str or None, optional
        Pipeline stage to filter, e.g. "stage1", "stage2".
        If None, no stage filtering is applied.

    manifest_filename : str or sequence of str, optional
        One or more filenames to consider as manifests. Defaults
        to ("manifest.json",). You may pass wildcards like
        "*.manifest.json" if needed.

    search_globs : sequence of str or None, optional
        Custom glob patterns relative to ``base_dir``. If None,
        sensible defaults are used:
        - "**/manifest.json"  (recursive)
        - "*/*/manifest.json" (shallow)
        - "*_stage*/manifest.json" (stage folders)

    recursive : bool, default=True
        If True and no custom ``search_globs`` are given, the
        recursive pattern "**/<filename>" is included.

    return_all : bool, default=False
        If True, return a list of all matching candidates ordered
        newest-first. If False, return a single best path.

    prefer : {"mtime", "timestamp"}, default="mtime"
        How to rank candidates when auto-selecting:
        - "mtime": use file modification time.
        - "timestamp": use the manifest "timestamp" field if
          present and parseable, else fall back to mtime.

    filter_fn : callable or None, optional
        A predicate ``fn(manifest_dict, path) -> bool``. If given,
        a candidate is kept only if the predicate returns True.

    required_keys : sequence of str, optional
        If provided, candidates must contain all these manifest
        keys or be discarded.

    verbose : int, default=1
        Verbosity level. 0 = silent, 1 = info.

    Returns
    -------
    str or list of str
        The selected manifest path (or a list of candidates if
        ``return_all`` is True).

    Raises
    ------
    FileNotFoundError
        If no manifest can be found after applying all rules.

    Notes
    -----
    * This function does **not** read environment variables by
      itself, keeping it framework-agnostic. Your application can
      pass env-driven hints when calling it.
    * Use ``required_keys`` to ensure the manifest conforms to
      your expected schema (e.g., ("model","stage","city")).
    * Use ``filter_fn`` for advanced filtering, such as checking
      nested fields or custom tags.

    Examples
    --------
    Basic usage with manual path
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> find_manifest(manual="results/run_x/manifest.json")

    Guess by city/model/stage
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> find_manifest(base_dir="results", city_hint="nansha",
    ...               model_hint="GeoPriorSubsNet", stage_hint="stage1")

    Newest valid manifest under results/
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> find_manifest(base_dir="results", required_keys=("model",))

    Return all candidates (newest first)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> find_manifest(base_dir="results", return_all=True)

    Prefer manifest's internal timestamp if available
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> find_manifest(base_dir="results", prefer="timestamp")
    """
    # 1) manual path
    if manual:
        p = os.path.expanduser(os.path.expandvars(manual.strip()))
        if os.path.exists(p):
            if verbose:
                print(f"[Manifest] Using explicit path: {p}")
            return [p] if return_all else p
        raise FileNotFoundError(f"Manifest provided but not found: {p}")

    # 2) deterministic guesses by hints
    #    Try common folder shapes like:
    #    results/<city>_<model>_<stage>/manifest.json
    guesses: list[str] = []
    names = []
    if city_hint:
        names.append(city_hint)
    if model_hint:
        names.append(model_hint)
    if stage_hint:
        names.append(stage_hint)
    if names:
        folder = "_".join(names)
        for fn in (manifest_filename if isinstance(manifest_filename, (list, tuple, set))
                   else (manifest_filename,)):
            guess = os.path.join(base_dir, folder, fn)
            guesses.append(guess)

    for g in guesses:
        g_exp = os.path.expanduser(os.path.expandvars(g))
        if os.path.exists(g_exp):
            if verbose:
                print(f"[Manifest] Using guessed path: {g_exp}")
            return [g_exp] if return_all else g_exp

    # 3) glob search
    filenames = (manifest_filename if isinstance(manifest_filename, (list, tuple, set))
                 else (manifest_filename,))
    if search_globs is None:
        # sensible defaults
        search_globs = []
        if recursive:
            for fn in filenames:
                search_globs.append(os.path.join("**", fn))
        for fn in filenames:
            search_globs.extend([
                os.path.join("*", "*", fn),
                os.path.join("*_stage*", fn),
            ])

    candidates: list[Tuple[float, str]] = []

    for pat in search_globs:
        abs_pat = os.path.join(base_dir, pat)
        for path in glob.glob(abs_pat, recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    m = json.load(f)
            except Exception:
                continue  # unreadable or not JSON

            def _norm(x):
                return str(x).strip().lower() if x is not None else None
            
            m_city  = _norm(m.get("city"))
            m_model = str(m.get("model")).strip() if m.get(
                "model") is not None else None
            m_stage = _norm(m.get("stage"))
            
            if model_hint and m_model and m_model != model_hint:
                continue
            if city_hint and m_city and m_city != _norm(city_hint):
                continue
            if stage_hint and m_stage and m_stage != _norm(stage_hint):
                continue

            # Required keys check
            if required_keys and not all(k in m for k in required_keys):
                continue

            # Custom predicate
            if filter_fn and not filter_fn(m, path):
                continue

            # Scoring for sort
            score: float
            if prefer == "timestamp":
                ts = _parse_manifest_ts(m.get("timestamp"))
                score = ts if ts is not None else os.path.getmtime(path)
            else:
                score = os.path.getmtime(path)

            candidates.append((float(score), path))

    if not candidates:
        raise FileNotFoundError(
            "No manifest found. Consider passing 'manual', or hints "
            "(city_hint/model_hint/stage_hint), or relax filters."
        )

    candidates.sort(reverse=True, key=lambda x: x[0])
    ordered = [p for _, p in candidates]

    if verbose:
        top = ordered[0]
        print(f"[Manifest] Auto-selected newest: {top}")
        if return_all and len(ordered) > 1:
            print(f"[Manifest] {len(ordered)} candidates found.")

    return ordered if return_all else ordered[0]


def _find_stage1_manifest(
    manual: Optional[str] = None,
    base_dir: str = "results",
    city_hint: Optional[str] = None,
    model_hint: Optional[str] = "GeoPriorSubsNet",
    **kwargs,
) -> str:
    """
    Back-compat wrapper that finds a *Stage-1* manifest.

    Parameters
    ----------
    manual : str or None, optional
        Explicit manifest path. See :func:`find_manifest`.

    base_dir : str, default="results"
        Root directory where experiment folders live.

    city_hint : str or None, optional
        City or dataset label to filter manifests.

    model_hint : str or None, optional
        Model name to filter manifests.

    **kwargs
        Forwarded to :func:`find_manifest` (e.g., ``prefer``,
        ``required_keys``, ``verbose``, etc.).

    Returns
    -------
    str
        Selected manifest path.

    Notes
    -----
    Equivalent to:
    ``find_manifest(..., stage_hint="stage1")``.
    """
    return find_manifest(
        manual=manual,
        base_dir=base_dir,
        city_hint=city_hint,
        model_hint=model_hint,
        stage_hint="stage1",
        **kwargs,
    )
