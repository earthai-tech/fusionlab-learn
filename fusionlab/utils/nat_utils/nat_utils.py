# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author : LKouadio <etanoyau@gmail.com>
#
# Utilities for the NATCOM subsidence experiments.
#
# This module is responsible for:
#   - reading `nat.com/config.py`,
#   - keeping `nat.com/config.json` in sync with it,
#   - exposing a flat configuration dictionary to Stage-1,
#     training, and tuning scripts.
#
# Usage from any script (Stage-1, Stage-2) ::
#
#     from fusionlab.utils.nat_utils import load_nat_config
#     cfg = load_nat_config()
#     CITY_NAME = cfg["CITY_NAME"]
#
# On first call, `config.py` is imported, converted to a dict,
# and written to `nat.com/config.json`.  On subsequent calls,
# if `config.py` has not changed, the JSON file is reused.


from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import joblib
from pathlib import Path
import datetime as dt
from typing import ( 
    Any, 
    Dict, 
    Tuple, 
    Optional, 
    Mapping, 
    Union, 
    TYPE_CHECKING
)

import numpy as np


# --- Optional TensorFlow import for GeoPrior helpers -----------------------
try:  # pragma: no cover - defensive import
    import tensorflow as tf # noqa
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except Exception:  # pragma: no cover
    TF_AVAILABLE = False
    tf = None  # type: ignore[assignment]

    class _AdamStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "TensorFlow is required for NATCOM GeoPrior helpers "
                "(e.g. compile_geoprior_for_eval). Please install "
                "`tensorflow>=2.12`."
            )

    Adam = _AdamStub  # type: ignore[assignment]

# ---------------------------------------------------------------------
# Optional TensorFlow typing support
# ---------------------------------------------------------------------
# We avoid importing TensorFlow at runtime from this module to keep it
# lightweight (useful for tooling / docs environments). For type checkers
# and IDEs, we expose a tf name under TYPE_CHECKING.
#
# Use string annotations like "tf.data.Dataset" and "tf.Tensor" so that
# runtime does not need TensorFlow to be installed.

if TYPE_CHECKING:  # pragma: no cover
    import tensorflow as tf  # noqa: F401

# Shared error message used by helpers that need TensorFlow.
TF_IMPORT_ERROR_MSG = (
    "fusionlab.utils.nat_utils: TensorFlow is required for this helper "
    "but could not be imported. Install `tensorflow` to use functions "
    "that construct or consume `tf.data.Dataset` objects."
)


# -------------------------------------------------------------------
# Internal path helpers
# -------------------------------------------------------------------
def _project_root() -> str:
    """
    Return the root directory of the `fusionlab-learn` repository.

    This is computed relative to this file:

        fusionlab-learn/
            fusionlab/
                utils/
                    nat_utils/nat_utils.py   
            nat.com/
                config.py
    """
    here = os.path.abspath(__file__)
    utils_dir = os.path.dirname (os.path.dirname(here))
    fusionlab_dir = os.path.dirname(utils_dir)
    root = os.path.dirname(fusionlab_dir)
    return root


def get_natcom_dir(root ="nat.com") -> str:
    """
    Directory containing NATCOM scripts and configuration,
    typically `<repo_root>/nat.com`.
    """
    return os.path.join(_project_root(), root)


def get_config_paths(root="nat.com") -> Tuple[str, str]:
    """
    Return `(config_py_path, config_json_path)` for NATCOM.
    """
    nat_dir = get_natcom_dir(root = root)
    config_py = os.path.join(nat_dir, "config.py")
    config_json = os.path.join(nat_dir, "config.json")
    return config_py, config_json

def get_default_runs_root(
    root: str = "nat.com",
    runs_dir_name: str = ".fusionlab_runs",
) -> str:
    """
    Return the base directory for GUI run artifacts.

    The default is ``<project_root>/.fusionlab_runs`` where
    ``<project_root>`` is the same root inferred by
    :func:`_project_root`.

    This is *only* a convenience helper; CLI scripts keep
    using their own defaults (usually ``<cwd>/results``).
    The GUI overrides ``BASE_OUTPUT_DIR`` with this path so
    GUI runs do not mix with CLI results.
    """
    proj_root = os.path.dirname(get_natcom_dir(root=root))
    runs_root = os.path.join(proj_root, runs_dir_name)
    os.makedirs(runs_root, exist_ok=True)
    return runs_root

# -------------------------------------------------------------------
# Low-level helpers
# -------------------------------------------------------------------
def _hash_file(path: str) -> str:
    """
    Compute a SHA-256 hash of the file at `path`.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _import_config_module(config_py: str):
    """
    Import `config.py` by absolute path, without assuming it is
    on `sys.path`.
    """
    if not os.path.exists(config_py):
        raise FileNotFoundError(
            f"NATCOM config.py not found at: {config_py}"
        )

    spec = importlib.util.spec_from_file_location("nat_config", config_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {config_py!r}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _is_basic_jsonable(value: Any) -> bool:
    """
    Return True if the value is a simple JSON-serialisable type.
    """
    return isinstance(value, (int, float, str, bool, list, dict))


def _extract_config_dict(module) -> Dict[str, Any]:
    """
    Extract a flat configuration dictionary from the `config`
    module by selecting suitable global variables.

    - Keys starting with '_' are ignored.
    - Functions, classes and modules are ignored.
    - Only basic JSON-like values are kept.

    Environment variables (CITY, MODEL_NAME_OVERRIDE,
    JUPYTER_PROJECT_ROOT) can override some keys.
    """
    cfg: Dict[str, Any] = {}

    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if callable(value):
            continue
        if isinstance(value, type):
            continue
        if _is_basic_jsonable(value):
            cfg[name] = value

    # Build a compact "censoring" block for Stage-2 scripts if
    # it is not already present.
    if "CENSORING_SPECS" in cfg and "censoring" not in cfg:
        censor_block = {
            "specs": cfg["CENSORING_SPECS"],
            "use_effective_h_field": cfg.get(
                "USE_EFFECTIVE_H_FIELD", True
            ),
            "include_flags_as_dynamic": cfg.get(
                "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC", True
            ),
        }
        cfg["censoring"] = censor_block

    # Optional environment overrides (advanced use).
    city_env = os.getenv("CITY", "").strip()
    if city_env:
        cfg["CITY_NAME"] = city_env.lower()

    model_env = os.getenv("MODEL_NAME_OVERRIDE", "").strip()
    if model_env:
        cfg["MODEL_NAME"] = model_env

    root_env = os.getenv("JUPYTER_PROJECT_ROOT", "").strip()
    if root_env:
        cfg["DATA_DIR"] = root_env

    return cfg


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def _refresh_city_files(cfg: Dict[str, Any]) -> None:
    city = str(cfg.get("CITY_NAME", "")).strip().lower()
    var = str(cfg.get("DATASET_VARIANT", "")).strip()

    btmp = cfg.get("BIG_FN_TEMPLATE", None)
    stmp = cfg.get("SMALL_FN_TEMPLATE", None)

    if city and var and isinstance(btmp, str):
        cfg["BIG_FN"] = btmp.format(city=city, variant=var)

    if city and var and isinstance(stmp, str):
        cfg["SMALL_FN"] = stmp.format(city=city, variant=var)


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    changed = False

    city_env = os.getenv("CITY", "").strip()
    if city_env:
        cfg["CITY_NAME"] = city_env.lower()
        changed = True

    model_env = os.getenv("MODEL_NAME_OVERRIDE", "").strip()
    if model_env:
        cfg["MODEL_NAME"] = model_env
        changed = True

    root_env = os.getenv("JUPYTER_PROJECT_ROOT", "").strip()
    if root_env:
        cfg["DATA_DIR"] = root_env
        changed = True

    if changed:
        _refresh_city_files(cfg)

    return cfg


def ensure_config_json(root: str = "nat.com") -> Tuple[Dict[str, Any], str]:
    """
    Ensure that `nat.com/config.json` exists and is consistent
    with `nat.com/config.py`.

    Returns
    -------
    config : dict
        The configuration dictionary (`payload["config"]`).
    json_path : str
        Absolute path to `config.json`.

    Behaviour
    ---------
    - If `config.json` does not exist, it is created from
      `config.py`.
    - If it exists but the SHA-256 hash of `config.py` has
      changed, it is regenerated.
    - Otherwise the existing JSON file is reused.
    """
    
    config_py, config_json = get_config_paths(root=root)
    py_hash = _hash_file(config_py)

    payload: Optional[Dict[str, Any]] = None
    if os.path.exists(config_json):
        try:
            with open(config_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = None

    meta = payload.get("__meta__", {}) if isinstance(payload, dict) else {}
    if (
        isinstance(payload, dict)
        and meta.get("config_py_hash") == py_hash
        and "config" in payload
    ):
        cfg = dict(payload["config"])
        cfg = _apply_env_overrides(cfg)
        return cfg, config_json

    module = _import_config_module(config_py)
    cfg = _extract_config_dict(module)
    cfg = _apply_env_overrides(cfg)

    payload = {
        "city": cfg.get("CITY_NAME"),
        "model": cfg.get("MODEL_NAME"),
        "config": cfg,
        "__meta__": {"config_py_hash": py_hash},
    }

    os.makedirs(os.path.dirname(config_json), exist_ok=True)
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return cfg, config_json



def load_nat_config(root="nat.com") -> Dict[str, Any]:
    """
    High-level helper used by NATCOM scripts.

    Example
    -------
    >>> from fusionlab.utils.nat_utils import load_nat_config
    >>> cfg = load_nat_config()
    >>> CITY_NAME = cfg["CITY_NAME"]
    >>> TIME_STEPS = cfg["TIME_STEPS"]
    """
    cfg, _ = ensure_config_json(root=root)
    return cfg


def load_nat_config_payload(root="nat.com") -> Dict[str, Any]:
    """
    Return the full `config.json` payload, including `city`,
    `model` and `__meta__` fields.

    This is convenient when you also want to see which hash or
    city/model are currently active.
    """
    config_py, config_json = get_config_paths(root=root)
    if not os.path.exists(config_json):
        ensure_config_json(root=root)
    with open(config_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_float1(x):
    if x is None:
        return None
    arr = np.asarray(x).reshape(-1)
    return float(arr[0])

def affine_from_scaler(scaler, idx: int = 0):
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        data_min = np.asarray(scaler.data_min_).reshape(-1)
        data_max = np.asarray(scaler.data_max_).reshape(-1)
        scale = float((data_max - data_min)[idx])
        bias  = float(data_min[idx])
        return scale, bias

    if hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
        sc = np.asarray(scaler.scale_).reshape(-1)
        mu = np.asarray(scaler.mean_).reshape(-1)
        return float(sc[idx]), float(mu[idx])

    if hasattr(scaler, "scale_") and hasattr(scaler, "center_"):
        sc = np.asarray(scaler.scale_).reshape(-1)
        ce = np.asarray(scaler.center_).reshape(-1)
        return float(sc[idx]), float(ce[idx])

    raise TypeError(
        f"Unsupported scaler type for affine inference: {type(scaler)}")


def resolve_si_affine(
    cfg: dict,
    scaler_info: dict,
    *,
    target_name: str,
    prefix: str,              # "SUBS" or "HEAD"
    unit_factor_key: str,     # "SUBS_UNIT_TO_SI" or "HEAD_UNIT_TO_SI"
    scale_key: str,           # "SUBS_SCALE_SI" / "HEAD_SCALE_SI"
    bias_key: str,            # "SUBS_BIAS_SI"  / "HEAD_BIAS_SI"
):
    # 1) explicit overrides win
    scale = cfg.get(scale_key, None)
    bias  = cfg.get(bias_key, None)

    unit_factor = float(cfg.get(unit_factor_key, 1.0))
    auto = bool(cfg.get("AUTO_SI_AFFINE_FROM_STAGE1", True))

    if (scale is None or bias is None) and auto:
        info = scaler_info.get(target_name) or {}
        idx  = int(info.get("idx", 0))
        scaler = info.get("scaler")
        if scaler is None and "scaler_path" in info:
            # load happens elsewhere in your code; keep it simple here
            scaler = info.get("scaler")
        if scaler is None:
            raise RuntimeError(
                f"[{prefix}] Cannot infer SI affine: scaler for target "
                f"{target_name!r} not found in scaler_info."
            )
        s, b = affine_from_scaler(scaler, idx=idx)
        if scale is None:
            scale = s
        if bias is None:
            bias = b

    # 2) apply unit conversion into the affine
    #    y_SI = (y_scaled*scale + bias) * unit_factor
    scale_si = float(scale) * unit_factor
    bias_si  = float(bias)  * unit_factor
    return scale_si, bias_si

# -------------------------------------------------------------------------
# NATCOM training helpers
# -------------------------------------------------------------------------

def map_targets_for_training(
    y_dict: dict,
    subs_key: str = "subsidence",
    gwl_key: str = "gwl",
    subs_pred_key: str = "subs_pred",
    gwl_pred_key: str = "gwl_pred",
) -> dict:
    """
    Standardise target dictionaries to the Keras compile keys.

    This helper enforces a small convention used throughout the
    NATCOM training scripts:

    - Upstream sequence builders typically export raw targets with
      keys ``subsidence`` and ``gwl``.
    - The GeoPrior model is compiled with targets named
      ``subs_pred`` and ``gwl_pred``.

    This function accepts either style and always returns a dict
    keyed by ``subs_pred`` and ``gwl_pred`` for use in Keras.

    Parameters
    ----------
    y_dict : dict
        Dictionary produced by the Stage-1 sequence exporter or by
        a previous training script. Must contain either
        (``subsidence``, ``gwl``) or (``subs_pred``, ``gwl_pred``).
    subs_key : str, default="subsidence"
        Name of the raw subsidence key in ``y_dict``.
    gwl_key : str, default="gwl"
        Name of the raw groundwater-level key in ``y_dict``.
    subs_pred_key : str, default="subs_pred"
        Standardised key for the subsidence prediction target.
    gwl_pred_key : str, default="gwl_pred"
        Standardised key for the GWL prediction target.

    Returns
    -------
    dict
        New dictionary with keys ``subs_pred`` and ``gwl_pred``.

    Raises
    ------
    KeyError
        If the dictionary does not contain either of the expected
        key pairs.
    """
    # Case 1: raw keys from Stage-1 exporter.
    if subs_key in y_dict and gwl_key in y_dict:
        return {
            subs_pred_key: y_dict[subs_key],
            gwl_pred_key:  y_dict[gwl_key],
        }

    # Case 2: already in compiled form.
    if subs_pred_key in y_dict and gwl_pred_key in y_dict:
        return y_dict

    # Anything else is considered an error – we fail loudly so the
    # user can fix the pipeline rather than train on the wrong data.
    raise KeyError(
        f"Targets must contain ({subs_key!r},{gwl_key!r}) or "
        f"({subs_pred_key!r},{gwl_pred_key!r})."
    )


def ensure_input_shapes(
    x: dict,
    mode: str,
    forecast_horizon: int,
) -> dict:
    """
    Ensure presence of zero-width static/future placeholders.

    Stage-1 exporters sometimes omit ``static_features`` or
    ``future_features`` when there are no static/future variables
    for a particular experiment. Keras, however, expects these
    inputs to exist so that the input signature remains stable.

    This helper:

    - Copies the input dict to avoid in-place modification.
    - Ensures ``static_features`` is an array of shape ``(N, 0)``
      if missing.
    - Ensures ``future_features`` is an array of shape
      ``(N, T_future, 0)`` if missing, where:

        * ``T_future = dynamic_features.shape[1]`` when
          ``mode == "tft_like"`` (past+future style).
        * Otherwise, ``T_future = forecast_horizon``.

    Parameters
    ----------
    x : dict
        Dictionary containing at least ``dynamic_features`` with
        shape ``(N, T_dyn, D_dyn)``.
    mode : str
        Model mode. When ``"tft_like"`` the future sequence length
        is inferred from the dynamic sequence.
    forecast_horizon : int
        Forecast horizon in time steps/years for non-TFT modes.

    Returns
    -------
    dict
        Shallow copy of ``x`` with guaranteed
        ``static_features`` and ``future_features`` entries.
    """
    out = dict(x)
    N = out["dynamic_features"].shape[0]

    # Static features: if missing, create a (N, 0) placeholder so
    # the model signature always includes a static input.
    if out.get("static_features") is None:
        out["static_features"] = np.zeros((N, 0), dtype=np.float32)

    # Future features: similar logic – guarantee an array with
    # zero feature width, but correct time length.
    if out.get("future_features") is None:
        if mode == "tft_like":
            t_future = out["dynamic_features"].shape[1]
        else:
            t_future = int(forecast_horizon)
        out["future_features"] = np.zeros((N, t_future, 0), dtype=np.float32)

    return out


def _np_nonfinite_report(
    arr: np.ndarray,
    max_items: int = 5,
) -> Optional[dict[str, Any]]:
    """Return a small report if arr has NaN/Inf."""
    bad = ~np.isfinite(arr)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return None

    idx = np.argwhere(bad)
    head = idx[:max_items]

    samples: list[tuple[tuple[int, ...], Any]] = []
    for ii in head:
        t = tuple(ii.tolist())
        samples.append((t, arr[t]))

    finite = np.isfinite(arr)
    if np.any(finite):
        min_f = float(np.nanmin(arr[finite]))
        max_f = float(np.nanmax(arr[finite]))
    else:
        min_f = None
        max_f = None

    return {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "n_nonfinite": n_bad,
        "first_bad": samples,
        "min_finite": min_f,
        "max_finite": max_f,
    }


def check_npz_dict_finite(
    d: dict,
    name: str,
    feature_names_last_dim: Optional[list[str]] = None,
    max_bad_channels: int = 30,
    max_print: int = 20,
) -> None:
    """
    Validate that all numeric arrays in a dict
    contain only finite values.

    If `feature_names_last_dim` is provided and
    matches v.shape[-1], a per-channel report
    is added (helpful for dyn/fut features).
    """
    problems: list[tuple[str, dict[str, Any]]] = []

    for k, v in d.items():
        if not isinstance(v, np.ndarray):
            continue
        if not np.issubdtype(v.dtype, np.number):
            continue

        rep = _np_nonfinite_report(v)
        if rep is None:
            continue

        if (
            feature_names_last_dim
            and v.ndim >= 2
            and v.shape[-1] == len(feature_names_last_dim)
        ):
            bad = ~np.isfinite(v)
            per_ch = bad.reshape(-1, v.shape[-1]).sum(axis=0)
            bad_idx = np.where(per_ch > 0)[0].tolist()

            rep["bad_channels"] = [
                {
                    "index": int(i),
                    "name": feature_names_last_dim[i],
                    "n_nonfinite": int(per_ch[i]),
                }
                for i in bad_idx[:max_bad_channels]
            ]

        problems.append((str(k), rep))

    if not problems:
        print(f"[OK] {name}: all numeric arrays finite.")
        return

    print(f"\n[NaN/Inf] Non-finite values in {name}:")
    for k, rep in problems[:max_print]:
        sh = rep.get("shape")
        # dt = rep.get("dtype")
        nb = rep.get("n_nonfinite")
        print(f"  - key={k!r} shape={sh} dtype={dt}")
        print(f"    nonfinite={nb}")

        if "bad_channels" in rep:
            print("    bad channels:")
            for ch in rep["bad_channels"]:
                i = ch["index"]
                nm = ch["name"]
                nn = ch["n_nonfinite"]
                print(f"      * {i:>3} {nm:<32} n={nn}")

        fb = rep.get("first_bad", [])
        print(f"    first bad: {fb[:5]}")

    raise RuntimeError(
        f"Stopping: {name} contains NaN/Inf. "
        "Fix Stage-1 export or cleaning."
    )


def scan_tf_dataset_finite(
    ds: Any,
    name: str,
    max_batches: int = 200,
) -> None:
    """
    Eager scan of first N batches.

    Useful to fail *before* model.fit().
    """
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(TF_IMPORT_ERROR_MSG) from exc

    for b, (xb, yb) in enumerate(ds):
        if max_batches is not None and b >= max_batches:
            break

        for k, v in xb.items():
            if v.dtype.is_floating or v.dtype.is_complex:
                tf.debugging.assert_all_finite(
                    v,
                    f"{name}: X[{k}] NaN/Inf "
                    f"at batch {b}",
                )

        for k, v in yb.items():
            if v.dtype.is_floating or v.dtype.is_complex:
                tf.debugging.assert_all_finite(
                    v,
                    f"{name}: y[{k}] NaN/Inf "
                    f"at batch {b}",
                )

    print(f"[OK] {name}: first {max_batches} batches ok.")

def make_tf_dataset(
    X_np: dict,
    y_np: dict,
    batch_size: int,
    shuffle: bool,
    mode: str,
    forecast_horizon: int,
    *,
    seed: int = 42,
    drop_remainder: bool = False,
    reshuffle_each_iter: bool = True,
    prefetch: bool = True,
    check_npz_finite: bool = False,
    check_finite: bool = False,
    scan_finite_batches: int = 0,
    dynamic_feature_names: Optional[list[str]] = None,
    future_feature_names: Optional[list[str]] = None,
) -> "Any":
    """
    Build a `tf.data.Dataset` using NATCOM
    conventions.

    Steps:
    1) ensure_input_shapes(...) for X.
    2) map_targets_for_training(...) for y.
    3) tf.data pipeline (shuffle/batch/prefetch).
    4) optional finite checks (NPZ + tf batches).

    Parameters
    ----------
    X_np : dict
        Input dictionary, typically obtained from ``np.load`` on
        the Stage-1 ``*_inputs_npz`` file.
    y_np : dict
        Target dictionary, typically obtained from ``np.load`` on
        the Stage-1 ``*_targets_npz`` file.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        If ``True``, shuffle the dataset using a fixed seed for
        reproducibility.
    mode : str
        Model mode passed to :func:`ensure_input_shapes`.
    forecast_horizon : int
        Forecast horizon passed to :func:`ensure_input_shapes`.
        
    check_npz_finite : bool
        If True, checks Xin/Yin numpy arrays
        for NaN/Inf before building ds.
    check_finite : bool
        If True, inserts `assert_all_finite`
        checks inside the tf.data pipeline.
    scan_finite_batches : int
        If >0, eagerly scans first N batches
        right away (fails early).
    dynamic_feature_names, future_feature_names
        If provided, used to report bad
        channels for feature tensors.

    Returns
    -------
    tf.data.Dataset
        Dataset of (X, y) pairs.

    Notes
    -----
    TensorFlow is imported lazily inside the function so that
    this module remains importable in environments where TF is
    not installed (for example, for tooling or static analysis).
        
        
    """
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(TF_IMPORT_ERROR_MSG) from exc

    # Normalize inputs/targets to canonical keys.
    Xin = ensure_input_shapes(
        X_np,
        mode=mode,
        forecast_horizon=forecast_horizon,
    )
    Yin = map_targets_for_training(y_np)

    # Optional: stop early if NPZ content is bad.
    if check_npz_finite:
        check_npz_dict_finite(Xin, "X_np (Xin)")
        check_npz_dict_finite(Yin, "y_np (Yin)")

        if (
            dynamic_feature_names
            and "dynamic_features" in Xin
        ):
            check_npz_dict_finite(
                {"dynamic_features": Xin["dynamic_features"]},
                "Xin.dynamic_features",
                feature_names_last_dim=dynamic_feature_names,
            )

        if (
            future_feature_names
            and "future_features" in Xin
        ):
            check_npz_dict_finite(
                {"future_features": Xin["future_features"]},
                "Xin.future_features",
                feature_names_last_dim=future_feature_names,
            )

    # Build dataset.
    ds = tf.data.Dataset.from_tensor_slices((Xin, Yin))

    # Shuffle with a stable seed.
    if shuffle:
        # Prefer dynamic_features for size.
        if "dynamic_features" in Xin:
            n = int(Xin["dynamic_features"].shape[0])
        else:
            # Fallback: first array in Xin.
            first = next(iter(Xin.values()))
            n = int(first.shape[0])

        ds = ds.shuffle(
            buffer_size=max(1, n),
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iter,
        )

    ds = ds.batch(
        batch_size,
        drop_remainder=drop_remainder,
    )

    # Optional: add assert_all_finite into pipeline.
    if check_finite:

        def _assert_batch(xb, yb):
            # Assert only float/complex tensors.
            for k, v in xb.items():
                if v.dtype.is_floating or v.dtype.is_complex:
                    tf.debugging.assert_all_finite(
                        v,
                        f"X[{k}] has NaN/Inf",
                    )
            for k, v in yb.items():
                if v.dtype.is_floating or v.dtype.is_complex:
                    tf.debugging.assert_all_finite(
                        v,
                        f"y[{k}] has NaN/Inf",
                    )
            return xb, yb

        ds = ds.map(
            _assert_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    # Optional: force an eager scan now.
    # This is what stops *before* model.fit().
    if scan_finite_batches and scan_finite_batches > 0:
        scan_tf_dataset_finite(
            ds,
            name="make_tf_dataset",
            max_batches=int(scan_finite_batches),
        )

    return ds

def load_scaler_info(encoders_block: dict) -> dict | None:
    """
    Load the ``scaler_info`` mapping from an encoders block.

    Stage-1 exporters typically store a compact description of the
    scalers used to normalise the data. In many cases this takes
    the form:

    .. code-block:: python

        encoders = {
            "main_scaler": "/path/to/minmax.joblib",
            "coord_scaler": "/path/to/coords.joblib",
            "scaler_info": "/path/to/scaler_info.joblib",
            ...
        }

    where ``scaler_info`` is either a path to a joblib file or an
    already-loaded dictionary.

    This helper returns a dictionary regardless of how it was
    stored, making downstream formatting/evaluation code simpler.

    Parameters
    ----------
    encoders_block : dict
        The ``encoders`` part of the Stage-1 manifest
        (``M["artifacts"]["encoders"]``).

    Returns
    -------
    dict or None
        The loaded ``scaler_info`` dictionary, or ``None`` if not
        present / not loadable.
    """
    si = encoders_block.get("scaler_info")
    if isinstance(si, str) and os.path.exists(si):
        try:
            return joblib.load(si)
        except Exception:
            # If loading fails we fall back to the raw string; the
            # caller can decide how to proceed.
            pass
    return si


def build_censor_mask(
    xb: dict,
    H,
    idx: int | None,
    thresh: float = 0.5,
    *,
    source: str = "dynamic",          # {"dynamic", "future"}
    reduce_time: str = "any",         # {"any", "last", "all"}
    align: str = "broadcast",         # {"broadcast", "crop", "pad_false", "pad_edge", "error"}
) -> "tf.Tensor":
    """
    Build a censor mask aligned to the forecast horizon: (B, H, 1).

    Parameters
    ----------
    source:
        - "dynamic": read flag from xb["dynamic_features"][:, :, idx]
          (history window, length TIME_STEPS).
        - "future":  read flag from xb["future_features"][:, :, idx]
          (forecast window, should have length H).
    reduce_time:
        When source="dynamic" and the flag is effectively static, collapse
        the history axis to one label per sample:
          - "any":  censored if any history step is flagged (robust default)
          - "last": use last history step
          - "all":  censored only if all history steps are flagged
    align:
        How to align to horizon H if time length != H (mainly for safety):
          - "broadcast": repeat a single-step label to all H steps (recommended)
          - "crop":      take last H steps (only works if T > H)
          - "pad_false": pad missing steps with False (if T < H)
          - "pad_edge":  pad missing steps by repeating last (if T < H)
          - "error":     raise if mismatch
    """
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(TF_IMPORT_ERROR_MSG) from exc

    # Resolve batch size
    if "coords" in xb:
        B = tf.shape(xb["coords"])[0]
    else:
        B = tf.shape(xb.get("dynamic_features", xb.get("future_features")))[0]

    H = tf.cast(H, tf.int32)

    if idx is None:
        return tf.zeros((B, H, 1), dtype=tf.bool)

    key = "dynamic_features" if source == "dynamic" else "future_features"
    feat = xb.get(key, None)
    if feat is None:
        return tf.zeros((B, H, 1), dtype=tf.bool)

    nfeat = tf.shape(feat)[-1]

    def _all_false():
        return tf.zeros((B, H, 1), dtype=tf.bool)

    def _align_time(m):  # m: (B, T, 1)
        T = tf.shape(m)[1]

        def _broadcast_from_one(step):
            return tf.tile(step, [1, H, 1])

        if align == "error":
            tf.debugging.assert_equal(T, H, message=f"{key} length != H")
            return m

        if align == "crop":
            # if T < H, this cannot increase length -> would still mismatch
            return tf.cond(T >= H, lambda: m[:, -H:, :], lambda: m)

        if align == "pad_false":
            # pad at front so last steps line up
            pad = tf.maximum(H - T, 0)
            m2 = tf.pad(m, paddings=[[0, 0], [pad, 0], [0, 0]], constant_values=False)
            return m2[:, -H:, :]

        if align == "pad_edge":
            pad = tf.maximum(H - T, 0)
            last = m[:, -1:, :]
            m2 = tf.concat([tf.tile(last, [1, pad, 1]), m], axis=1)
            return m2[:, -H:, :]

        # default: "broadcast"
        # If already H, keep it; else broadcast a single-step summary.
        return tf.cond(
            tf.equal(T, H),
            lambda: m,
            lambda: _broadcast_from_one(m[:, -1:, :]),
        )

    def _build():
        m = (feat[..., idx:idx + 1] > thresh)  # (B, T, 1)

        # If source is dynamic, we usually want a sample-level censor label.
        if source == "dynamic":
            if reduce_time == "any":
                one = tf.reduce_any(m, axis=1, keepdims=True)   # (B,1,1)
                return tf.tile(one, [1, H, 1])                  # (B,H,1)
            if reduce_time == "all":
                one = tf.reduce_all(m, axis=1, keepdims=True)
                return tf.tile(one, [1, H, 1])
            if reduce_time == "last":
                one = m[:, -1:, :]
                return tf.tile(one, [1, H, 1])

        # Otherwise, align time dimension to horizon (future usually already matches)
        return _align_time(m)

    return tf.cond(tf.less(idx, nfeat), _build, _all_false)

def build_censor_mask_from_dynamic(
    xb: dict,
    H: int,
    dyn_idx: int | None,
    thresh: float = 0.5,
) -> "tf.Tensor":
    """
    Build a boolean censoring mask from the dynamic features.

    This is used to stratify metrics by censored/uncensored cells
    based on a flag stored in ``dynamic_features[..., dyn_idx]``.

    The function:

    - Looks up ``dynamic_features`` from the input batch.
    - Applies a threshold on the selected feature column to build
      a mask of shape ``(B, T_dyn, 1)``.
    - If the dynamic time length differs from ``H``, it takes the
      last ``H`` steps (consistent with the forecasting horizon).
    - If no dynamic features or index are available, returns an
      all-False mask of shape ``(B, H, 1)``.
      
    If the censor flag only exists on the history window (T_dyn=TIME_STEPS),
    but the evaluation is done on the forecast horizon (H=FORECAST_HORIZON),
    we must broadcast/pad because slicing cannot increase length.
    
    Parameters
    ----------
    xb : dict
        Batch input dictionary from a ``tf.data.Dataset`` with
        at least ``"dynamic_features"`` and ``"coords"``.
    H : int
        Horizon length for the evaluation (number of time steps).
    dyn_idx : int or None
        Index of the censor flag within ``dynamic_features``.
        If ``None``, returns an all-False mask.
    thresh : float, default=0.5
        Threshold above which a value is considered "censored".

    Returns
    -------
    tf.Tensor
        Boolean mask of shape ``(B, H, 1)`` where True indicates
        censored samples.
    """

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(TF_IMPORT_ERROR_MSG) from exc

    # Resolve B
    if "coords" in xb:
        B = tf.shape(xb["coords"])[0]
    else:
        dyn0 = xb.get("dynamic_features", None)
        B = tf.shape(dyn0)[0] if dyn0 is not None else 0

    H = tf.cast(H, tf.int32)

    # No flag → no censoring
    dyn = xb.get("dynamic_features", None)
    if dyn is None or dyn_idx is None:
        return tf.zeros((B, H, 1), dtype=tf.bool)

    # Defensive: dyn_idx range (works even if dyn.shape[-1] is None)
    nfeat = tf.shape(dyn)[-1]
    def _all_false():
        return tf.zeros((B, H, 1), dtype=tf.bool)

    def _build():
        # (B, T_dyn, 1)
        m_dyn = (dyn[..., dyn_idx:dyn_idx + 1] > thresh)
        T_dyn = tf.shape(m_dyn)[1]

        # Case 1: exact match
        def _same():
            return m_dyn

        # Case 2: history longer than horizon → take last H
        def _crop():
            return m_dyn[:, -H:, :]

        # Case 3: history shorter than horizon → broadcast last observed flag
        def _broadcast_last():
            last = m_dyn[:, -1:, :]              # (B,1,1)
            return tf.tile(last, [1, H, 1])      # (B,H,1)

        return tf.case(
            [
                (tf.equal(T_dyn, H), _same),
                (tf.greater(T_dyn, H), _crop),
                (tf.less(T_dyn, H), _broadcast_last),
            ],
            default=_broadcast_last,
            exclusive=True,
        )

    return tf.cond(tf.less(dyn_idx, nfeat), _build, _all_false)

# -------------------------------------------------------------------------
# Public helpers for Stage-1/Stage-2 NPZ handling and tuned model recovery
# -------------------------------------------------------------------------

def pick_npz_for_dataset(
    manifest: dict,
    split: str,
) -> tuple[dict | None, dict | None]:
    """
    Load (inputs, targets) NPZ arrays for a given dataset split.

    This is a public, reusable version of the internal helper that
    was previously named ``_pick_npz_for_dataset``.

    Parameters
    ----------
    manifest : dict
        Stage-1 manifest dictionary with the structure::

            manifest["artifacts"]["numpy"] = {
                "train_inputs_npz": ...,
                "train_targets_npz": ...,
                "val_inputs_npz": ...,
                "val_targets_npz": ...,
                "test_inputs_npz": ... (optional),
                "test_targets_npz": ... (optional),
            }

    split : {"train", "val", "test"}
        Which dataset to load.

    Returns
    -------
    X : dict or None
        Dictionary of input arrays for the requested split, or ``None``
        if the split is unavailable (e.g. test NPZ missing).

    y : dict or None
        Dictionary of target arrays for the requested split, or ``None``
        if targets are unavailable.

    Raises
    ------
    KeyError
        If the manifest does not contain the expected NPZ entries.
    ValueError
        If ``split`` is not one of ``{"train", "val", "test"}``.
    """
    npzs = manifest.get("artifacts", {}).get("numpy", None)
    if npzs is None:
        raise KeyError(
            "Manifest is missing 'artifacts[\"numpy\"]' section with NPZ paths."
        )

    if split == "train":
        x = dict(np.load(npzs["train_inputs_npz"]))
        y = dict(np.load(npzs["train_targets_npz"]))
        return x, y

    if split == "val":
        x = dict(np.load(npzs["val_inputs_npz"]))
        y = dict(np.load(npzs["val_targets_npz"]))
        return x, y

    if split == "test":
        tin = npzs.get("test_inputs_npz")
        tt = npzs.get("test_targets_npz")
        if not tin:
            # No test split available for this run
            return None, None
        x = dict(np.load(tin))
        y = dict(np.load(tt)) if tt else None
        return x, y

    raise ValueError("split must be one of {'train', 'val', 'test'}.")

def infer_input_dims_from_X(X: dict) -> tuple[int, int, int]:
    """
    Infer (static_input_dim, dynamic_input_dim, future_input_dim) from NPZ inputs.

    This is a public, defensive version of the former
    ``_infer_input_dims_from_X`` helper.

    Parameters
    ----------
    X : dict
        Dictionary with keys:

        - ``'dynamic_features'`` (required, shape (N, T, D_dyn))
        - ``'static_features'`` (optional, shape (N, D_static) or None)
        - ``'future_features'`` (optional, shape (N, T_future, D_future) or None)

    Returns
    -------
    static_dim : int
        Last-dimension size of ``static_features`` (0 if missing or None).

    dynamic_dim : int
        Last-dimension size of ``dynamic_features``. Raises if missing.

    future_dim : int
        Last-dimension size of ``future_features`` (0 if missing or None).

    Raises
    ------
    KeyError
        If ``'dynamic_features'`` is missing in ``X``.
    """
    if "dynamic_features" not in X:
        raise KeyError(
            "X must contain key 'dynamic_features' with shape (N, T, D_dyn)."
        )

    dyn = np.asarray(X["dynamic_features"])
    dynamic_dim = int(dyn.shape[-1])

    static = X.get("static_features", None)
    static_dim = int(np.asarray(static).shape[-1]) if static is not None else 0

    fut = X.get("future_features", None)
    future_dim = int(np.asarray(fut).shape[-1]) if fut is not None else 0

    return static_dim, dynamic_dim, future_dim


def sanitize_inputs_np(X: dict) -> dict:
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



def _npz_to_dict(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _strip_prefix(name: str, prefixes: Tuple[str, ...]) -> str:
    low = name.lower()
    for p in prefixes:
        if low.startswith(p):
            return name[len(p):]
    return name


def _split_bundle_npz(
    data: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    x: Dict[str, np.ndarray] = {}
    y: Dict[str, np.ndarray] = {}

    for k, v in data.items():
        lk = k.lower()

        if lk in ("subs_pred", "gwl_pred"):
            y[k] = v
            continue

        if lk.startswith(("y_", "y.", "target")):
            y[k] = v
            continue

        if lk.startswith(("x_", "x.", "input")):
            x[k] = v
            continue

        x[k] = v

    x2: Dict[str, np.ndarray] = {}
    y2: Dict[str, np.ndarray] = {}

    for k, v in x.items():
        nk = _strip_prefix(
            k,
            ("x_", "x.", "input_", "inputs_"),
        )
        x2[nk] = v

    for k, v in y.items():
        nk = _strip_prefix(
            k,
            ("y_", "y.", "target_", "targets_"),
        )
        y2[nk] = v

    return x2, y2


def _infer_targets_path(inputs_path: Path) -> Path:
    p = Path(inputs_path)
    name = p.name

    repls = (
        ("_inputs", "_targets"),
        ("inputs", "targets"),
        ("_input", "_target"),
        ("input", "target"),
    )

    for a, b in repls:
        if a in name:
            cand = p.with_name(name.replace(a, b))
            if cand.exists():
                return cand

    raise FileNotFoundError(
        "Could not infer targets NPZ from inputs NPZ:\n"
        f"  inputs: {str(p)}\n"
        "Pass a mapping {'inputs':..., 'targets':...} "
        "to load_windows_npz()."
    )


def load_windows_npz(
    path: Union[str, Path, Mapping[str, str]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load Stage-1 windows as (x, y).

    Supported:
    - Bundle NPZ (contains inputs+targets in one file).
    - Mapping {'inputs': <npz>, 'targets': <npz>}.
    - Inputs NPZ only (targets inferred by filename).
    - Directory containing inputs/targets NPZ.

    Returns
    -------
    x : dict[str, np.ndarray]
        Inputs (e.g., static_features, dynamic_features, etc.)
    y : dict[str, np.ndarray]
        Targets (e.g., subs_pred, gwl_pred)
    """
    if isinstance(path, Mapping):
        ip = Path(path["inputs"])
        tp = Path(path["targets"])
        return _npz_to_dict(ip), _npz_to_dict(tp)

    p = Path(path)

    if p.is_dir():
        in_cands = (
            "inputs.npz",
            "train_inputs.npz",
            "X_inputs.npz",
        )
        tg_cands = (
            "targets.npz",
            "train_targets.npz",
            "y_targets.npz",
        )

        ip = None
        tp = None

        for n in in_cands:
            cand = p / n
            if cand.exists():
                ip = cand
                break

        for n in tg_cands:
            cand = p / n
            if cand.exists():
                tp = cand
                break

        if ip is None or tp is None:
            raise FileNotFoundError(
                "Directory does not contain recognizable "
                "inputs/targets NPZ files:\n"
                f"  dir: {str(p)}"
            )

        return _npz_to_dict(ip), _npz_to_dict(tp)

    if not p.exists():
        raise FileNotFoundError(f"Missing NPZ: {str(p)}")

    d = _npz_to_dict(p)
    x, y = _split_bundle_npz(d)

    if y:
        return x, y

    tp = _infer_targets_path(p)
    return x, _npz_to_dict(tp)

def resolve_hybrid_config(
    manifest_cfg: dict,
    live_cfg: dict,
    verbose: bool = True
) -> dict:
    """
    Merge Manifest config (Data Authority) with Live config (Physics Authority).

    Strategy
    --------
    1. Base: Start with Manifest config. This guarantees that data shapes,
       time steps, features, and normalization match the artifacts on disk.
    2. Override: Apply specific keys from Live config (config.py) that control
       architecture, physics equations, loss weights, and training dynamics.

    This allows you to tune the model and physics without re-running Stage 1.

    Parameters
    ----------
    manifest_cfg : dict
        Configuration dictionary loaded from `manifest.json`.
    live_cfg : dict
        Configuration dictionary loaded from the current `config.py`.

    Returns
    -------
    dict
        Merged configuration.
    """
    # 1. Start with Manifest (Data Wins)
    merged = manifest_cfg.copy()

    # 2. Define "Safe" keys that Stage 2 is allowed to override.
    #    (Everything that does NOT affect input data shapes or target columns)
    OVERRIDABLE_KEYS = {
        # track City Name , so we can switch to change city as well 
        "CITY_NAME", "CITY", "MODEL_NAME", 
        
        "USE_IN_MEMORY_MODEL", "DEBUG", "USE_TF_SAVEDMODEL", 
        
        "TRACK_AUX_METRICS", 
        # --- 1. Architecture (Safe to tune if model is rebuilt) ---
        "EMBED_DIM", "HIDDEN_UNITS", "LSTM_UNITS", "ATTENTION_UNITS",
        "NUMBER_HEADS", "DROPOUT_RATE", "MEMORY_SIZE", "SCALES",
        "USE_RESIDUALS", "USE_BATCH_NORM", "USE_VSN", "VSN_UNITS",
        "ATTENTION_LEVELS",

        # --- 2. Physics Toggles & Math ---
        "PDE_MODE_CONFIG", "SCALE_PDE_RESIDUALS",
        "CONSOLIDATION_STEP_RESIDUAL_METHOD",
        "ALLOW_SUBS_RESIDUAL", "OFFSET_MODE",
        "PHYSICS_BOUNDS_MODE", "TIME_UNITS",
        
        # --- 3. Physics Parameters & Initialization ---
        "GEOPRIOR_INIT_MV", "GEOPRIOR_INIT_KAPPA", 
        "GEOPRIOR_GAMMA_W", "GEOPRIOR_H_REF",
        "GEOPRIOR_KAPPA_MODE", "GEOPRIOR_USE_EFFECTIVE_H",
        "GEOPRIOR_HD_FACTOR", "PHYSICS_BOUNDS",

        # --- 4. Loss Weights (Lambdas) ---
        "LAMBDA_CONS", "LAMBDA_GW", "LAMBDA_PRIOR", 
        "LAMBDA_SMOOTH", "LAMBDA_BOUNDS", "LAMBDA_MV", 
        "LAMBDA_Q", "LAMBDA_OFFSET", 
        "LOSS_WEIGHT_GWL",
        "MV_LR_MULT", "KAPPA_LR_MULT",
        "SUBS_WEIGHTS", "GWL_WEIGHTS",  # Safe: weights don't change shape

        # --- 5. Scaling, Stability & Units ---
        "CONS_SCALE_FLOOR", "GW_SCALE_FLOOR",
        "GW_RESIDUAL_UNITS", "CONSOLIDATION_RESIDUAL_UNITS",
        "DT_MIN_UNITS", "Q_WRT_NORMALIZED_TIME",
        "Q_IN_SI", "Q_IN_PER_SECOND", "Q_KIND", "Q_LENGTH_IN_SI",
        "DRAINAGE_MODE", "CLIP_GLOBAL_NORM",
        "DEBUG_PHYSICS_GRADS", "SCALING_ERROR_POLICY",

        # --- 6. Consolidation Drawdown Gates ---
        "CONS_DRAWDOWN_MODE", "CONS_DRAWDOWN_RULE", 
        "CONS_STOP_GRAD_REF", "CONS_DRAWDOWN_ZERO_AT_ORIGIN", 
        "CONS_DRAWDOWN_CLIP_MAX", "CONS_RELU_BETA",

        # --- 7. MV Prior Strategy ---
        "MV_PRIOR_UNITS", "MV_ALPHA_DISP", "MV_HUBER_DELTA",
        "MV_PRIOR_MODE", "MV_WEIGHT", "MV_SCHEDULE_UNIT",
        "MV_DELAY_EPOCHS", "MV_WARMUP_EPOCHS", 
        "MV_DELAY_STEPS", "MV_WARMUP_STEPS",

        # --- 8. Training Strategy & Gates (Physics-First vs Data-First) ---
        "TRAINING_STRATEGY",
        # Physics-First specific overrides
        "Q_POLICY_PHYSICS_FIRST", 
        "Q_WARMUP_EPOCHS_PHYSICS_FIRST", "Q_RAMP_EPOCHS_PHYSICS_FIRST",
        "SUBS_RESID_POLICY_PHYSICS_FIRST", 
        "SUBS_RESID_WARMUP_EPOCHS_PHYSICS_FIRST", "SUBS_RESID_RAMP_EPOCHS_PHYSICS_FIRST",
        "LAMBDA_Q_PHYSICS_FIRST", "LOSS_WEIGHT_GWL_PHYSICS_FIRST",
        # Data-First specific overrides
        "LOSS_WEIGHT_GWL_DATA_FIRST", "LAMBDA_Q_DATA_FIRST",
        "Q_POLICY_DATA_FIRST", 
        "Q_WARMUP_EPOCHS_DATA_FIRST", "Q_RAMP_EPOCHS_DATA_FIRST",
        "SUBS_RESID_POLICY_DATA_FIRST", 
        "SUBS_RESID_WARMUP_EPOCHS_DATA_FIRST", "SUBS_RESID_RAMP_EPOCHS_DATA_FIRST",

        # --- 9. Lambda Offset Scheduler ---
        "USE_LAMBDA_OFFSET_SCHEDULER", "LAMBDA_OFFSET_UNIT",
        "LAMBDA_OFFSET_WHEN", "LAMBDA_OFFSET_WARMUP",
        "LAMBDA_OFFSET_START", "LAMBDA_OFFSET_END", 
        "LAMBDA_OFFSET_SCHEDULE",

        # --- 10. Training Loop & Logging ---
        "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "PATIENCE",
        "LOG_Q_DIAGNOSTICS", "AUDIT_STAGES",
        "EVAL_JSON_UNITS_MODE", "EVAL_JSON_UNITS_SCOPE", 
        
        "VERBOSE"
    }

    updates = []
    for key in OVERRIDABLE_KEYS:
        # If the key exists in your live config (config.py), it overrides manifest
        if key in live_cfg:
            current_val = merged.get(key)
            new_val = live_cfg[key]
            
            # Update only if different (or if manifest didn't have it)
            if new_val != current_val:
                merged[key] = new_val
                updates.append(key)

    if verbose and updates:
        print(f"[Config] Applied {len(updates)} physics/training overrides from config.py:")
        # Print a few examples
        sample = updates[:4]
        print(f"         {', '.join(sample)} ...")
    
    return merged

# -------------------------------------------------------------------------
# Backward-compatible aliases for old private helper names
# -------------------------------------------------------------------------
_pick_npz_for_dataset = pick_npz_for_dataset
_infer_input_dims_from_X = infer_input_dims_from_X