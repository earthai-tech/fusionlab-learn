# geoprior/ui/inference/plan.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Any, List

from ...config.prior_schema import FieldKey
from ...config.store import GeoConfigStore


__all__ = ["build_plan_text"]


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


def build_plan_text(store: GeoConfigStore) -> str:
    """
    Build a compact plan summary for the Inference head bar.

    Notes
    -----
    This intentionally summarizes store-backed inference knobs.
    Runtime paths (model/npz) are validated and displayed
    in the preview panel.
    """
    cov = float(_get_fk(store, "interval_level", 0.80))
    mode = str(
        _get_fk(store, "calibration_mode", "none") or "none"
    ).strip()

    temp = float(
        _get_fk(store, "calibration_temperature", 1.0)
    )
    pen = float(_get_fk(store, "crossing_penalty", 0.0))
    mar = float(_get_fk(store, "crossing_margin", 0.0))

    inc_gwl = bool(_get_ui(store, "infer.include_gwl", False))
    plots = bool(_get_ui(store, "infer.plots", True))

    parts: List[str] = []
    parts.append(f"cov={cov:.2f}")
    parts.append(f"mode={mode}")
    parts.append(f"T={temp:.2f}")

    if pen > 0.0 or mar > 0.0:
        parts.append(f"cross={pen:.2g}/{mar:.2g}")

    if inc_gwl:
        parts.append("gwl=1")
    else:
        parts.append("gwl=0")

    if plots:
        parts.append("plots=1")
    else:
        parts.append("plots=0")

    return "Inference: " + ", ".join(parts)
