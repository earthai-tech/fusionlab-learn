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
    - uncertainty
    - calibration
    - outputs
    - advanced

    Notes
    -----
    Store-driven only. Runtime paths are validated
    in the preview panel, not here.
    """
    out: Dict[str, Dict[str, str]] = {}

    # Artifacts are runtime (model/manifest/dataset/custom npz).
    out["artifacts"] = _warn("Set")

    # ------------------ Uncertainty (store) -------------------
    cov = float(_get_fk(store, "interval_level", 0.80))
    if cov < 0.50 or cov > 0.99:
        out["uncertainty"] = _warn("Fix")
    else:
        out["uncertainty"] = _ok("OK")

    # ------------------ Calibration (store) -------------------
    mode = str(
        _get_fk(store, "calibration_mode", "none") or "none"
    ).strip()

    temp = float(
        _get_fk(store, "calibration_temperature", 1.0)
    )

    pen = float(_get_fk(store, "crossing_penalty", 0.0))
    mar = float(_get_fk(store, "crossing_margin", 0.0))

    cal = _ok("OK")
    if not mode:
        cal = _warn("Set")
    elif temp <= 0.0:
        cal = _warn("T?")
    if pen < 0.0 or mar < 0.0:
        cal = _warn("Fix")

    out["calibration"] = cal

    # ------------------ Outputs (UI/runtime today) ------------
    inc_gwl = bool(_get_ui(store, "infer.include_gwl", False))
    plots = bool(_get_ui(store, "infer.plots", True))

    if (not isinstance(inc_gwl, bool)) or (
        not isinstance(plots, bool)
    ):
        out["outputs"] = _warn("Fix")
    else:
        out["outputs"] = _ok("OK")

    # ------------------ Advanced ------------------------------
    out["advanced"] = _ok("OK")
    return out
