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
from typing import Any, Dict, Tuple


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


def get_natcom_dir() -> str:
    """
    Directory containing NATCOM scripts and configuration,
    typically `<repo_root>/nat.com`.
    """
    return os.path.join(_project_root(), "nat.com")


def get_config_paths() -> Tuple[str, str]:
    """
    Return `(config_py_path, config_json_path)` for NATCOM.
    """
    nat_dir = get_natcom_dir()
    config_py = os.path.join(nat_dir, "config.py")
    config_json = os.path.join(nat_dir, "config.json")
    return config_py, config_json


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
def ensure_config_json() -> Tuple[Dict[str, Any], str]:
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
    config_py, config_json = get_config_paths()
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


def load_nat_config() -> Dict[str, Any]:
    """
    High-level helper used by NATCOM scripts.

    Example
    -------
    >>> from fusionlab.utils.nat_utils import load_nat_config
    >>> cfg = load_nat_config()
    >>> CITY_NAME = cfg["CITY_NAME"]
    >>> TIME_STEPS = cfg["TIME_STEPS"]
    """
    cfg, _ = ensure_config_json()
    return cfg


def load_nat_config_payload() -> Dict[str, Any]:
    """
    Return the full `config.json` payload, including `city`,
    `model` and `__meta__` fields.

    This is convenient when you also want to see which hash or
    city/model are currently active.
    """
    config_py, config_json = get_config_paths()
    if not os.path.exists(config_json):
        ensure_config_json()
    with open(config_json, "r", encoding="utf-8") as f:
        return json.load(f)


