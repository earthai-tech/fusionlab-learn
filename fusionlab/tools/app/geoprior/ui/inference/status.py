# geoprior/ui/inference/status.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Any, Dict

from ...config.prior_schema import FieldKey
from ...config.store import GeoConfigStore


__all__ = ["compute_infer_nav"]


def _ok(txt: str = "OK") -> Dict[str, str]:
    return {"status": "ok", "text": txt}


def _warn(txt: str = "Fix") -> Dict[str, str]:
    return {"status": "warn", "text": txt}


def _err(txt: str = "Err") -> Dict[str, str]:
    return {"status": "err", "text": txt}


def _get_fk(
    store: GeoConfigStore,
    key: str,
    default: Any,
) -> Any:
    try:
        return store.get_value(FieldKey(key), default=default)
    except Exception:
        return default


def _get_ui(
    store: GeoConfigStore,
    key: str,
    default: Any,
) -> Any:
    try:
        return store.get(key, default)
    except Exception:
        return default


def compute_infer_nav(
    store: GeoConfigStore,
) -> Dict[str, Dict[str, str]]:
    """
    Compute navigator chip statuses for Inference.

    Keys
    ----
    - artifacts
    - uncert
    - calib
    - outputs
    - adv

    Notes
    -----
    This is store-driven only. Runtime paths are validated
    in the preview panel, not here.
    """
    out: Dict[str, Dict[str, str]] = {}

    # Artifacts are runtime (model/manifest/dataset/custom npz),
    # so we cannot validate in store. Keep "Set" to prompt user.
    out["artifacts"] = _warn("Set")

    cov = float(_get_fk(store, "interval_level", 0.80))
    if cov < 0.50 or cov > 0.99:
        out["uncert"] = _warn("Fix")
    else:
        out["uncert"] = _ok("OK")

    mode = str(
        _get_fk(store, "calibration_mode", "none") or "none"
    ).strip()

    temp = float(
        _get_fk(store, "calibration_temperature", 1.0)
    )
    if not mode:
        out["calib"] = _warn("Set")
    elif temp <= 0.0:
        out["calib"] = _warn("T?")
    else:
        out["calib"] = _ok("OK")

    pen = float(_get_fk(store, "crossing_penalty", 0.0))
    mar = float(_get_fk(store, "crossing_margin", 0.0))
    if pen < 0.0 or mar < 0.0:
        out["calib"] = _warn("Fix")

    # Outputs are currently runtime checkboxes on the UI.
    # If you later store them, swap _get_ui -> _get_fk.
    inc_gwl = bool(_get_ui(store, "infer.include_gwl", False))
    plots = bool(_get_ui(store, "infer.plots", True))

    if (not isinstance(inc_gwl, bool)) or (
        not isinstance(plots, bool)
    ):
        out["outputs"] = _warn("Fix")
    else:
        out["outputs"] = _ok("OK")

    out["adv"] = _ok("OK")
    return out
