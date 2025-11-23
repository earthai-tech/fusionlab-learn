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
import datetime as dt
from typing import Any, Dict, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

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
                    nat_utils.py   
            nat.com/
                config.py
    """
    here = os.path.abspath(__file__)
    utils_dir = os.path.dirname(here)
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
def ensure_config_json(root="nat.com") -> Tuple[Dict[str, Any], str]:
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

    payload: Dict[str, Any] | None = None
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
        # JSON is in sync with config.py; reuse it.
        return payload["config"], config_json

    # (Re)build configuration from config.py
    module = _import_config_module(config_py)
    config_dict = _extract_config_dict(module)

    payload = {
        "city": config_dict.get("CITY_NAME"),
        "model": config_dict.get("MODEL_NAME"),
        "config": config_dict,
        "__meta__": {
            "config_py_hash": py_hash,
        },
    }

    os.makedirs(os.path.dirname(config_json), exist_ok=True)
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return config_dict, config_json


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


def make_tf_dataset(
    X_np: dict,
    y_np: dict,
    batch_size: int,
    shuffle: bool,
    mode: str,
    forecast_horizon: int,
) -> "tf.data.Dataset":
    """
    Build a ``tf.data.Dataset`` using NATCOM conventions.

    This is the canonical way to turn the Stage-1 NPZ exports into
    a dataset suitable for Keras training/evaluation.

    It performs three steps:

    1. Normalise the input dictionary so all three inputs
       (``static_features``, ``dynamic_features``,
       ``future_features``) are present via
       :func:`ensure_input_shapes`.
    2. Map the target dictionary to the canonical keys
       (``subs_pred``, ``gwl_pred``) via
       :func:`map_targets_for_training`.
    3. Construct a batched, prefetched ``tf.data.Dataset``.

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

    Returns
    -------
    tf.data.Dataset
        A batched, optionally shuffled dataset of (X, y) pairs.

    Notes
    -----
    TensorFlow is imported lazily inside the function so that
    this module remains importable in environments where TF is
    not installed (for example, for tooling or static analysis).
    """
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        # You can catch this in your training script if you want.
        raise ImportError(TF_IMPORT_ERROR_MSG) from exc

    Xin = ensure_input_shapes(X_np, mode, forecast_horizon)
    Yin = map_targets_for_training(y_np)

    ds = tf.data.Dataset.from_tensor_slices((Xin, Yin))
    if shuffle:
        # Use dataset size as buffer and a fixed seed for stable
        # experiments / unit tests.
        ds = ds.shuffle(
            buffer_size=Xin["dynamic_features"].shape[0],
            seed=42,
        )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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


def save_ablation_record(
    outdir: str,
    city: str,
    model_name: str,
    cfg: dict,
    eval_dict: dict | None,
    phys_diag: dict | None = None,
    per_h_mae: dict | None = None,
    per_h_r2: dict | None = None,
) -> None:
    """
    Append a single ablation record to ``ablation_record.jsonl``.

    Each training run (e.g., different physics toggles or weights)
    writes one JSON line containing:

    - Basic run identifiers (city, model, timestamp).
    - Physics configuration (``PDE_MODE_CONFIG``, lambda weights,
      effective head flags, etc.).
    - Key performance metrics (R², MSE, MAE, coverage, sharpness).
    - Optional physics diagnostics (``epsilon_prior``,
      ``epsilon_cons``).
    - Optional per-horizon MAE/R² for more detailed analysis.

    Parameters
    ----------
    outdir : str
        Base output directory for the current run. The ablation
        file is created under ``outdir / "ablation_records"``.
    city : str
        City name (e.g., ``"nansha"`` or ``"zhongshan"``).
    model_name : str
        Model identifier (e.g., ``"GeoPriorSubsNet"``).
    cfg : dict
        Lightweight configuration dictionary containing at least
        the physics-related keys used below.
    eval_dict : dict or None
        Dictionary of evaluation metrics (R², MSE, MAE,
        coverage80, sharpness80). If ``None``, metrics fields
        default to ``None``.
    phys_diag : dict or None, optional
        Physics diagnostics (e.g., from ``evaluate()``) with keys
        such as ``"epsilon_prior"`` and ``"epsilon_cons"``.
    per_h_mae : dict or None, optional
        Per-horizon MAE values (e.g., keyed by year/step).
    per_h_r2 : dict or None, optional
        Per-horizon R² values.

    Notes
    -----
    The output file is a JSON-Lines file, so it can be loaded
    with :func:`load_ablation_jsonl`.
    """
    eval_dict = eval_dict or {}

    rec = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "city": city,
        "model": model_name,
        # Physics toggles / weights
        "pde_mode": cfg.get("PDE_MODE_CONFIG"),
        "use_effective_h": bool(cfg.get("GEOPRIOR_USE_EFFECTIVE_H", True)),
        "kappa_mode": cfg.get("GEOPRIOR_KAPPA_MODE", "bar"),
        "hd_factor": cfg.get("GEOPRIOR_HD_FACTOR", 0.6),
        "lambda_cons": cfg.get("LAMBDA_CONS"),
        "lambda_gw": cfg.get("LAMBDA_GW"),
        "lambda_prior": cfg.get("LAMBDA_PRIOR"),
        "lambda_smooth": cfg.get("LAMBDA_SMOOTH"),
        "lambda_mv": cfg.get("LAMBDA_MV"),
        # Key metrics
        "r2": eval_dict.get("r2"),
        "mse": eval_dict.get("mse"),
        "mae": eval_dict.get("mae"),
        "coverage80": eval_dict.get("coverage80"),
        "sharpness80": eval_dict.get("sharpness80"),
    }

    if phys_diag:
        rec["epsilon_prior"] = phys_diag.get("epsilon_prior")
        rec["epsilon_cons"] = phys_diag.get("epsilon_cons")

    if per_h_mae is not None:
        rec["per_horizon_mae"] = per_h_mae
    if per_h_r2 is not None:
        rec["per_horizon_r2"] = per_h_r2

    abl_dir = os.path.join(outdir, "ablation_records")
    os.makedirs(abl_dir, exist_ok=True)

    jpath = os.path.join(abl_dir, "ablation_record.jsonl")
    with open(jpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"[Ablation] appended -> {jpath}")


def load_ablation_jsonl(path: str) -> pd.DataFrame:
    """
    Load an ablation JSON-Lines file into a :class:`pandas.DataFrame`.

    This is the companion to :func:`save_ablation_record`. Each
    line is parsed as JSON and turned into one row.

    Parameters
    ----------
    path : str
        Path to ``ablation_record.jsonl``.

    Returns
    -------
    pandas.DataFrame
        DataFrame where each row corresponds to one ablation
        record.

    Examples
    --------
    >>> df_abl = load_ablation_jsonl(
    ...     "ablation_records/ablation_record.jsonl"
    ... )
    >>> df_abl.head()
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def best_epoch_and_metrics(
    history: dict,
    monitor: str = "val_loss",
) -> tuple[int | None, dict]:
    """
    Return the best epoch and metrics at that epoch.

    Given a ``History.history`` dictionary produced by
    ``model.fit(...)``, this helper identifies the index of the
    minimum value for the monitored quantity (by default
    ``"val_loss"``) and returns:

    - The epoch index (0-based).
    - A dictionary mapping each metric name to its value at that
      epoch.

    Parameters
    ----------
    history : dict
        The ``history.history`` attribute from Keras training.
    monitor : str, default="val_loss"
        Name of the metric to minimise.

    Returns
    -------
    best_epoch : int or None
        Index of the best epoch, or ``None`` if ``monitor`` is
        not present.
    metrics_at_best : dict
        Mapping from metric name to its value at the best epoch.
        Empty if ``monitor`` is not present.
    """
    if not history or monitor not in history:
        return None, {}

    # nanargmin makes sure NaNs are ignored when searching for the
    # best epoch.
    be = int(np.nanargmin(history[monitor]))
    metrics_at_best = {
        k: float(v[be])
        for k, v in history.items()
        if len(v) > be
    }
    return be, metrics_at_best


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

    dyn = xb.get("dynamic_features", None)
    if dyn is not None:
        # Defensive programming: ensure the requested index is in
        # range before indexing.
        if dyn.shape[-1] and dyn_idx is not None and dyn_idx < dyn.shape[-1]:
            m_dyn = dyn[..., dyn_idx:dyn_idx + 1] > thresh  # (B, T_dyn, 1)
            T_dyn = tf.shape(m_dyn)[1]
            # If time dimension does not match the requested H,
            # keep only the last H steps (matching the forecast
            # window).
            return tf.cond(
                tf.not_equal(T_dyn, H),
                lambda: m_dyn[:, -H:, :],
                lambda: m_dyn,
            )

    # Fallback: no flag available → no censoring anywhere.
    B = tf.shape(xb["coords"])[0]
    return tf.zeros((B, H, 1), dtype=tf.bool)


def name_of(obj: object) -> str:
    """
    Return a human-readable name for an object.

    This utility is handy when serialising compile configurations
    (e.g., turning metric callables into simple strings for JSON
    logs).

    Parameters
    ----------
    obj : object
        Any Python object (function, class instance, etc.).

    Returns
    -------
    str
        ``obj.__name__`` if present, otherwise the class name, and
        finally ``str(obj)`` as a last resort.
    """
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "__class__"):
        return obj.__class__.__name__
    return str(obj)


def serialize_subs_params(
    params: dict,
    cfg: dict | None = None,
) -> dict:
    """
    Make GeoPrior subnet parameters JSON-friendly.

    The training scripts typically pass a dictionary of model
    construction arguments, e.g. ``subsmodel_params``, which
    contains objects such as ``LearnableMV`` or ``FixedGammaW``
    that are not directly JSON-serialisable.

    This helper replaces those objects by small dictionaries
    describing their type and scalar value, optionally using
    values from the NATCOM config dictionary.

    Parameters
    ----------
    params : dict
        Dictionary of model init parameters (e.g.
        ``subsmodel_params`` in ``training_NATCOM_GEOPRIOR.py``).
    cfg : dict, optional
        NATCOM config dictionary. If provided, scalar values are
        taken from:

        - ``GEOPRIOR_INIT_MV``
        - ``GEOPRIOR_INIT_KAPPA``
        - ``GEOPRIOR_GAMMA_W``
        - ``GEOPRIOR_H_REF``

        and used as the authoritative numbers.

    Returns
    -------
    dict
        Copy of ``params`` where scalar GeoPrior parameters are
        replaced by JSON-friendly dictionaries.

    Notes
    -----
    This function does **not** import any of the GeoPrior classes.
    It only introspects attributes like ``initial_value`` or
    ``value`` when the corresponding config entry is missing.
    """
    out = dict(params)
    cfg = cfg or {}

    # Helper to extract a scalar from either the config or the
    # original object (Learnable*/Fixed*).
    def _extract_scalar(obj, cfg_key: str) -> float | None:
        if cfg_key in cfg and cfg[cfg_key] is not None:
            try:
                return float(cfg[cfg_key])
            except Exception:
                pass
        # Fallback: try to read a typical attribute name.
        for attr in ("initial_value", "value"):
            if hasattr(obj, attr):
                try:
                    return float(getattr(obj, attr))
                except Exception:
                    continue
        return None

    if "mv" in out:
        mv_val = _extract_scalar(out["mv"], "GEOPRIOR_INIT_MV")
        out["mv"] = {
            "type": "LearnableMV",
            "initial_value": mv_val,
        }

    if "kappa" in out:
        kap_val = _extract_scalar(out["kappa"], "GEOPRIOR_INIT_KAPPA")
        out["kappa"] = {
            "type": "LearnableKappa",
            "initial_value": kap_val,
        }

    if "gamma_w" in out:
        gw_val = _extract_scalar(out["gamma_w"], "GEOPRIOR_GAMMA_W")
        out["gamma_w"] = {
            "type": "FixedGammaW",
            "value": gw_val,
        }

    if "h_ref" in out:
        href_val = _extract_scalar(out["h_ref"], "GEOPRIOR_H_REF")
        out["h_ref"] = {
            "type": "FixedHRef",
            "value": href_val,
        }

    return out

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


def load_best_hps_near_model(model_path: str) -> dict:
    """
    Locate and load tuned hyperparameters stored next to a tuned model.

    Search order inside the ``run_YYYYMMDD-HHMMSS`` directory containing
    ``model_path``:

    1. ``<city>_GeoPrior_best_hps.json``
       (where ``<city>`` is inferred from the model filename
       ``<city>_GeoPrior_best.keras``).
    2. ``tuning_summary.json['best_hps']``

    Parameters
    ----------
    model_path : str
        Path to a tuned model, e.g.::

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
                print(f"[HP] Loaded best_hps from: {hps_path}")
                return best_hps
            raise ValueError(
                f"File {hps_path!r} exists but does not contain a non-empty dict."
            )

    # 2) Try tuning_summary.json["best_hps"]
    summary_path = os.path.join(run_dir, "tuning_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        best_hps = summary.get("best_hps", {})
        if isinstance(best_hps, dict) and best_hps:
            print(f"[HP] Loaded best_hps from: {summary_path}")
            return best_hps
        raise ValueError(
            f"File {summary_path!r} exists but 'best_hps' is missing or empty."
        )

    raise FileNotFoundError(
        f"Could not find best hyperparameters near model_path={model_path!r}.\n"
        "Looked for '<city>_GeoPrior_best_hps.json' and 'tuning_summary.json'."
    )


def coerce_quantile_weights(
    d: dict | None,
    default: dict,
) -> dict:
    """
    Normalize a quantile-weight mapping to have float keys and float values.

    This helper is useful when reading JSON configs where the quantile
    keys are stored as strings (e.g. ``{'0.1': 3.0, '0.5': 1.0}``).

    Parameters
    ----------
    d : dict or None
        Original dictionary mapping quantile-like keys (str or float) to
        numeric weights. If ``None`` or empty, ``default`` is returned.

    default : dict
        Fallback dictionary to use when ``d`` is ``None`` or empty.

    Returns
    -------
    out : dict
        Dictionary with the same keys and values, but with:

        - keys coerced to float when possible (otherwise left as-is),
        - values coerced to ``float``.
    """
    if not d:
        return default

    out: dict[Any, float] = {}
    for k, v in d.items():
        try:
            q = float(k)
        except (TypeError, ValueError):
            # Non-numeric key (rare): keep as-is
            q = k
        out[q] = float(v)
    return out

def compile_for_eval(
    model: Any,
    manifest: dict,
    best_hps: dict | None,
    quantiles: list[float] | None,
    *,
    include_metrics: bool = True,
) -> Any:
    """
    Recompile a GeoPriorSubsNet instance for evaluation / diagnostics.

    This is intended for:
    - tuned models loaded from a `.keras` archive, or
    - models rebuilt from best_hps.

    It does NOT change the architecture or weights, only the compile
    configuration (optimizer, losses, and physics loss weights).

    Parameters
    ----------
    model : GeoPriorSubsNet
        Loaded or freshly-built GeoPriorSubsNet instance.
    manifest : dict
        Stage-1 manifest; training config is taken from
        ``manifest['config']``.
    best_hps : dict or None
        Dictionary of tuned hyperparameters. If empty/None, reasonable
        defaults are inferred from the manifest.
    quantiles : list of float or None
        Quantiles used for probabilistic subsidence/GWL outputs.
    include_metrics : bool, default=True
        If True, attach MAE/MSE + coverage/sharpness metrics to match
        the training script; if False, only losses are configured.

    Returns
    -------
    model :
        The same model instance, compiled in-place.
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to compile GeoPriorSubsNet. "
            "Install `tensorflow>=2.12` to use "
            "`compile_geoprior_for_eval`."
        )

    # Local imports so nat_utils.py itself stays lightweight
    from fusionlab.nn.losses import make_weighted_pinball
    if include_metrics:
        from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn

    cfg = manifest.get("config", {}) or {}
    best_hps = best_hps or {}

    # ---- 1. Data loss weights / quantile weights -------------------------
    subs_raw = cfg.get(
        "SUBS_WEIGHTS",
        {0.1: 3.0, 0.5: 1.0, 0.9: 3.0},
    )
    gwl_raw = cfg.get(
        "GWL_WEIGHTS",
        {0.1: 1.5, 0.5: 1.0, 0.9: 1.5},
    )

    subs_w = _coerce_quantile_weights(
        subs_raw, {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
    )
    gwl_w = _coerce_quantile_weights(
        gwl_raw, {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}
    )

    if quantiles:
        loss_dict = {
            "subs_pred": make_weighted_pinball(quantiles, subs_w),
            "gwl_pred": make_weighted_pinball(quantiles, gwl_w),
        }
    else:
        mse = tf.keras.losses.MeanSquaredError()
        loss_dict = {"subs_pred": mse, "gwl_pred": mse}

    loss_weights = {"subs_pred": 1.0, "gwl_pred": 0.5}

    # ---- 2. Physics weights: prefer best_hps, fall back to config --------
    def _hp_or_cfg(hp_key: str, cfg_key: str, default: float) -> float:
        if hp_key in best_hps and best_hps[hp_key] is not None:
            return float(best_hps[hp_key])
        if cfg_key in cfg and cfg[cfg_key] is not None:
            return float(cfg[cfg_key])
        return float(default)

    lr = _hp_or_cfg("learning_rate", "LEARNING_RATE", 1e-4)

    physics_kwargs = {
        "lambda_gw": _hp_or_cfg("lambda_gw", "LAMBDA_GW", 1.0),
        "lambda_cons": _hp_or_cfg("lambda_cons", "LAMBDA_CONS", 1.0),
        "lambda_prior": _hp_or_cfg("lambda_prior", "LAMBDA_PRIOR", 0.1),
        "lambda_smooth": _hp_or_cfg("lambda_smooth", "LAMBDA_SMOOTH", 0.01),
        "lambda_mv": _hp_or_cfg("lambda_mv", "LAMBDA_MV", 0.0),
        "mv_lr_mult": _hp_or_cfg("mv_lr_mult", "MV_LR_MULT", 1.0),
        "kappa_lr_mult": _hp_or_cfg(
            "kappa_lr_mult", "KAPPA_LR_MULT", 1.0
        ),
    }

    compile_kwargs: dict[str, Any] = {
        "optimizer": Adam(learning_rate=lr),
        "loss": loss_dict,
        "loss_weights": loss_weights,
        **physics_kwargs,
    }

    if include_metrics:
        metrics_dict = {
            "subs_pred": ["mae", "mse"]
            + ([coverage80_fn, sharpness80_fn] if quantiles else []),
            "gwl_pred": ["mae", "mse"],
        }
        compile_kwargs["metrics"] = metrics_dict

    model.compile(**compile_kwargs)
    return model

def compile_geoprior_for_eval(
    model: Any,  # type: ignore[override]
    manifest: dict,
    best_hps: dict,
    quantiles: list[float] | None,
) -> Any:
    """
    (Re)compile a GeoPriorSubsNet-like model for evaluation.

    This helper uses the Stage-1 manifest and tuned hyperparameters to
    configure:

    - the pinball losses for subsidence and GWL outputs,
    - loss weights for the two heads,
    - physics loss weights (lambda_*),
    - learning rate and LR multipliers.

    TensorFlow and fusionlab are imported lazily inside this function so
    that ``nat_utils`` can be imported even in non-TF environments.

    Parameters
    ----------
    model : GeoPriorSubsNet-like
        An instance of the GeoPriorSubsNet model (or any model exposing
        the same compile signature).

    manifest : dict
        Stage-1 manifest dictionary. The ``config`` entry is used to
        retrieve default loss weights and physics settings.

    best_hps : dict
        Hyperparameters loaded from the tuning run
        (e.g. via :func:`load_best_hps_near_model`).

    quantiles : list of float or None
        Quantile levels used for probabilistic outputs. If ``None``,
        mean-squared error is used instead of pinball loss.

    Returns
    -------
    model
        The same model instance, compiled in-place.

    Raises
    ------
    ImportError
        If TensorFlow or fusionlab's ``make_weighted_pinball`` cannot be
        imported.
    """
    cfg = manifest.get("config", {}) or {}

    # Lazy imports so nat_utils.py is importable without TensorFlow
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.optimizers import Adam  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "compile_geoprior_for_eval requires TensorFlow. "
            "Please install 'tensorflow>=2.12' to use this helper."
        ) from e

    try:
        from fusionlab.nn.losses import make_weighted_pinball  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "compile_geoprior_for_eval requires "
            "'fusionlab.nn.losses.make_weighted_pinball'. "
            "Ensure fusionlab is installed and importable."
        ) from e

    # Base loss weights between subsidence and GWL heads
    loss_weights = {"subs_pred": 1.0, "gwl_pred": 0.5}

    # Quantile-specific weights from config (with robust defaults)
    subs_raw = cfg.get("SUBS_WEIGHTS", {0.1: 3.0, 0.5: 1.0, 0.9: 3.0})
    gwl_raw = cfg.get("GWL_WEIGHTS", {0.1: 1.5, 0.5: 1.0, 0.9: 1.5})

    subs_weights = coerce_quantile_weights(
        subs_raw, {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
    )
    gwl_weights = coerce_quantile_weights(
        gwl_raw, {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}
    )

    if quantiles:
        loss_subs = make_weighted_pinball(quantiles, subs_weights)
        loss_gwl = make_weighted_pinball(quantiles, gwl_weights)
    else:
        loss_subs = tf.keras.losses.MSE
        loss_gwl = tf.keras.losses.MSE

    loss_dict = {"subs_pred": loss_subs, "gwl_pred": loss_gwl}

    # Learning rate: tuned value or config fallback
    lr_default = cfg.get("LEARNING_RATE", 5e-5)
    lr = float(best_hps.get("learning_rate", lr_default))
    optimizer = Adam(learning_rate=lr)

    # Physics loss weights and LR multipliers
    lambda_gw = float(best_hps.get("lambda_gw", cfg.get("LAMBDA_GW", 1.0)))
    lambda_cons = float(best_hps.get("lambda_cons", cfg.get("LAMBDA_CONS", 1.0)))
    lambda_prior = float(best_hps.get("lambda_prior", cfg.get("LAMBDA_PRIOR", 1.0)))
    lambda_smooth = float(
        best_hps.get("lambda_smooth", cfg.get("LAMBDA_SMOOTH", 1.0))
    )
    lambda_mv = float(best_hps.get("lambda_mv", cfg.get("LAMBDA_MV", 0.0)))
    mv_lr_mult = float(best_hps.get("mv_lr_mult", cfg.get("MV_LR_MULT", 1.0)))
    kappa_lr_mult = float(
        best_hps.get("kappa_lr_mult", cfg.get("KAPPA_LR_MULT", 1.0))
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        loss_weights=loss_weights,
        # physics loss weights + LR multipliers
        lambda_gw=lambda_gw,
        lambda_cons=lambda_cons,
        lambda_prior=lambda_prior,
        lambda_smooth=lambda_smooth,
        lambda_mv=lambda_mv,
        mv_lr_mult=mv_lr_mult,
        kappa_lr_mult=kappa_lr_mult,
    )
    return model


def build_geoprior_from_hps(
    manifest: dict,
    X_sample: dict,
    best_hps: dict,
    out_s_dim: int,
    out_g_dim: int,
    mode: str,
    horizon: int,
    quantiles: list[float] | None,
) -> Any:
    """
    Reconstruct a GeoPriorSubsNet from Stage-1 metadata + tuned HPs.

    This function is primarily intended as a **robust fallback** when
    ``tf.keras.models.load_model`` cannot deserialize a tuned model.
    It reconstructs the network geometry and physics settings from:

    - Stage-1 ``manifest['config']`` (for fixed architecture / physics),
    - tuned hyperparameters (for variable architecture / physics),
    - the Stage-1 input NPZ (for input dimensions).

    Parameters
    ----------
    manifest : dict
        Stage-1 manifest dictionary.

    X_sample : dict
        Inputs NPZ dictionary (already sanitized and passed through
        :func:`ensure_input_shapes`). Only shapes are used.

    best_hps : dict
        Hyperparameters loaded via :func:`load_best_hps_near_model`.

    out_s_dim : int
        Output dimension for the subsidence head.

    out_g_dim : int
        Output dimension for the GWL head.

    mode : str
        Sequence mode, e.g. ``"tft_like"`` or ``"pihal_like"``.

    horizon : int
        Forecast horizon (number of time steps).

    quantiles : list of float or None
        Quantile levels for probabilistic outputs.

    Returns
    -------
    model : GeoPriorSubsNet
        A freshly instantiated and compiled GeoPriorSubsNet instance.

    Raises
    ------
    ImportError
        If GeoPriorSubsNet cannot be imported from fusionlab.
    """
    try:
        from fusionlab.nn.pinn.models import GeoPriorSubsNet  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "build_geoprior_from_hps requires "
            "'fusionlab.nn.pinn.models.GeoPriorSubsNet'. "
            "Ensure fusionlab is installed and importable."
        ) from e

    cfg = manifest.get("config", {}) or {}

    # Infer input dims directly from NPZ
    static_dim, dynamic_dim, future_dim = infer_input_dims_from_X(X_sample)

    # Attention stack: fall back to a sensible default if not present
    attention_levels = cfg.get(
        "ATTENTION_LEVELS",
        ["cross", "hierarchical", "memory"],
    )

    # Whether we used effective H during Stage-2
    censor_cfg = cfg.get("censoring", {}) or {}
    use_effective_h = censor_cfg.get("use_effective_h_field", True)

    # Feature-processing mode controlled by tuned HPs
    use_vsn = bool(best_hps.get("use_vsn", True))
    feature_processing = "vsn" if use_vsn else "dense"

    architecture_config = {
        "encoder_type": "hybrid",
        "decoder_attention_stack": attention_levels,
        "feature_processing": feature_processing,
    }

    # Instantiate the model core with tuned settings
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
        # geomechanical priors (floats interpreted internally by the model)
        mv=float(best_hps.get("mv", 5e-7)),
        kappa=float(best_hps.get("kappa", 1.0)),
        architecture_config=architecture_config,
    )

    # Compile using the shared helper (losses + physics weights)
    compile_geoprior_for_eval(
        model=model,
        manifest=manifest,
        best_hps=best_hps,
        quantiles=quantiles,
    )

    print(
        "[Fallback] Reconstructed GeoPriorSubsNet from best_hps with "
        f"static_dim={static_dim}, dynamic_dim={dynamic_dim}, "
        f"future_dim={future_dim}, horizon={horizon}, mode={mode}"
    )
    return model


def infer_best_weights_path(model_path: str) -> str | None:
    """
    Infer the best-weights checkpoint path for a tuned GeoPrior model.

    Strategy
    --------
    1. Look for ``tuning_summary.json`` in the same folder as
       ``model_path`` and return the stored ``\"best_weights_path\"``
       if it exists and the file is present on disk.
    2. Fallback: replace the ``.keras`` suffix of ``model_path`` by
       ``.weights.h5``, assuming the convention::

           <CITY>_GeoPrior_best.keras
           -> <CITY>_GeoPrior_best.weights.h5

    Parameters
    ----------
    model_path : str
        Path to the tuned model archive (usually ``.keras``).

    Returns
    -------
    weights_path : str or None
        Absolute path to the weights file if found, otherwise ``None``.
    """
    run_dir = os.path.dirname(os.path.abspath(model_path))

    # 1) Preferred: tuning_summary.json
    summary_path = os.path.join(run_dir, "tuning_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            w = summary.get("best_weights_path")
            if w and os.path.exists(w):
                return w
        except Exception as e:  # pragma: no cover - defensive
            print(f"[Warn] Could not read tuning_summary.json for weights: {e}")

    # 2) Name-based guess from the .keras path
    root, ext = os.path.splitext(model_path)
    guess = root + ".weights.h5"
    if os.path.exists(guess):
        return guess

    return None

def load_or_rebuild_geoprior_model(
    model_path: str,
    manifest: dict,
    X_sample: dict,
    out_s_dim: int,
    out_g_dim: int,
    mode: str,
    horizon: int,
    quantiles: list[float] | None,
    city_name: str | None = None,
    compile_on_load: bool = True,
    verbose: int = 1,
):
    """
    Load a tuned GeoPriorSubsNet from disk, with robust rebuild fallback.

    This helper centralizes the logic:

    1. Try to load the model from ``model_path`` via
       :func:`tf.keras.models.load_model`, with all required custom
       objects registered (GeoPriorSubsNet, LearnableMV, etc.).

    2. If loading fails (e.g. due to environment or serialization
       changes), attempt a robust fallback:
       - Load the tuned hyperparameters via
         :func:`load_best_hps_near_model`.
       - Rebuild a compatible GeoPriorSubsNet instance using
         :func:`build_geoprior_from_hps`, based on Stage-1
         ``manifest['config']`` and an input sample ``X_sample``.
       - Find the best weights checkpoint via
         :func:`infer_best_weights_path` and load them into the
         rebuilt model, if available.

    Parameters
    ----------
    model_path : str
        Path to the tuned model archive (usually ``.keras``) produced
        by the tuner, e.g.::

            .../tuning/run_YYYYMMDD-HHMMSS/nansha_GeoPrior_best.keras

    manifest : dict
        Stage-1 manifest dictionary; its ``"config"`` entry is used to
        reconstruct the compile/physics configuration when rebuilding.

    X_sample : dict
        One NPZ inputs dictionary (e.g. validation or train NPZ) that
        has already been sanitized and passed through
        :func:`ensure_input_shapes`. Only its shapes are used to infer
        input dimensions.

    out_s_dim : int
        Output dimension for the subsidence head
        (usually from ``M['artifacts']['sequences']['dims']``).

    out_g_dim : int
        Output dimension for the GWL head.

    mode : str
        Sequence mode, e.g. ``"tft_like"`` or ``"pihal_like"``.

    horizon : int
        Forecast horizon (number of time steps).

    quantiles : list of float or None
        Quantile levels used for probabilistic outputs. If ``None``,
        the model is treated as a point-forecast model.

    city_name : str or None, optional
        City name for log messages. If ``None``, a neutral label is
        used in logs.

    compile_on_load : bool, default=True
        Whether to pass ``compile=True`` to :func:`load_model`. If
        ``False``, the model is loaded uncompiled, and only the
        rebuilt branch is compiled via
        :func:`build_geoprior_from_hps`.

    verbose : int, default=1
        Verbosity level for log messages (0 = silent, 1 = info).

    Returns
    -------
    model :
        A GeoPriorSubsNet instance (or compatible model) ready for
        evaluation/prediction.

    best_hps : dict or None
        Dictionary of tuned hyperparameters if they were loaded during
        the fallback path; otherwise ``None``.

    Raises
    ------
    ImportError
        If TensorFlow or required fusionlab components cannot be
        imported.

    RuntimeError
        If both direct loading and fallback reconstruction fail.
    """
    label_city = city_name or "GeoPrior"

    # --- Lazy imports so nat_utils can be imported without TF/fusionlab ---
    try:
        import tensorflow as tf  # type: ignore # noqa
        from tensorflow.keras.models import load_model  # type: ignore
        from tensorflow.keras.utils import custom_object_scope  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "load_or_rebuild_geoprior_model requires TensorFlow. "
            "Please install 'tensorflow>=2.12' to use this helper."
        ) from e

    try:
        from fusionlab.nn.pinn.models import GeoPriorSubsNet  # type: ignore
        from fusionlab.params import (  # type: ignore
            LearnableMV,
            LearnableKappa,
            FixedGammaW,
            FixedHRef,
        )
        from fusionlab.nn.losses import make_weighted_pinball  # type: ignore
        from fusionlab.nn.keras_metrics import (  # type: ignore
            coverage80_fn,
            sharpness80_fn,
        )
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "load_or_rebuild_geoprior_model requires fusionlab components "
            "(GeoPriorSubsNet, LearnableMV, etc.). Ensure fusionlab is "
            "installed and importable."
        ) from e

    # ------------------- 1) Try direct load_model -------------------------
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

    best_hps: dict | None = None

    with custom_object_scope(custom_objects):
        if verbose:
            print(f"[Model] Attempting to load tuned model from: {model_path}")

        try:
            model = load_model(model_path, compile=compile_on_load)
            if verbose:
                print(f"[Model] Successfully loaded tuned model for {label_city} "
                      f"from: {model_path}")
            return model, best_hps
        except Exception as e_load:
            if verbose:
                print(
                    f"[Warn] load_model('{model_path}') failed: {e_load}\n"
                    "[Warn] Attempting robust fallback: rebuild GeoPriorSubsNet "
                    "from tuned hyperparameters."
                )

    # ------------------- 2) Fallback: rebuild + load weights --------------
    # 2.1 Hyperparameters near the tuned model
    try:
        best_hps = load_best_hps_near_model(model_path)
    except Exception as e_hps:
        raise RuntimeError(
            "Failed to load tuned hyperparameters for fallback model "
            f"reconstruction near model_path={model_path!r}: {e_hps}"
        ) from e_hps

    # 2.2 Rebuild architecture + compile using Stage-1 manifest + best_hps
    try:
        model = build_geoprior_from_hps(
            manifest=manifest,
            X_sample=X_sample,
            best_hps=best_hps,
            out_s_dim=out_s_dim,
            out_g_dim=out_g_dim,
            mode=mode,
            horizon=horizon,
            quantiles=quantiles,
        )
    except Exception as e_build:
        raise RuntimeError(
            "Failed to reconstruct GeoPriorSubsNet from best_hps. "
            f"Error: {e_build}"
        ) from e_build

    # 2.3 Load weights into the rebuilt model, if a checkpoint is found
    weights_path = infer_best_weights_path(model_path)
    if weights_path is not None:
        try:
            model.load_weights(weights_path)
            if verbose:
                print(
                    "[Fallback] Loaded weights into rebuilt GeoPriorSubsNet "
                    f"from: {weights_path}"
                )
        except Exception as e_w:
            # We still return the rebuilt model, but warn that it is not
            # weight-identical to the tuned run.
            if verbose:
                print(
                    "[Warn] Could not load weights from checkpoint:\n"
                    f"       {weights_path}\n"
                    f"       Error: {e_w}\n"
                    "       The rebuilt model is using freshly-initialized "
                    "weights. Predictions will NOT match the tuned model."
                )
    else:
        if verbose:
            print(
                "[Warn] No weights checkpoint found near tuned model.\n"
                "       Using rebuilt model with freshly-initialized "
                "weights. Predictions will NOT match the tuned model."
            )

    return model, best_hps

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

# def map_targets(y_dict: dict) -> dict:
#     # Accept either ('subsidence','gwl') or ('subs_pred','gwl_pred')
#     if "subsidence" in y_dict and "gwl" in y_dict:
#         return {"subs_pred": y_dict["subsidence"], "gwl_pred": y_dict["gwl"]}
#     if "subs_pred" in y_dict and "gwl_pred" in y_dict:
#         return y_dict
#     # Allow missing targets for pure inference
#     return {}

# -------------------------------------------------------------------------
# Backward-compatible aliases for old private helper names
# -------------------------------------------------------------------------
safe_compile = compile_for_eval 
_pick_npz_for_dataset = pick_npz_for_dataset
_infer_input_dims_from_X = infer_input_dims_from_X
_load_best_hps_near_model = load_best_hps_near_model
_coerce_quantile_weights = coerce_quantile_weights
_compile_geoprior_for_eval = compile_geoprior_for_eval
_build_geoprior_from_hps = build_geoprior_from_hps
_infer_best_weights_path = infer_best_weights_path

