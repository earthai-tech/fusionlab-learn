# geoprior/ui/tune/plan.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Tuple

from ...config.prior_schema import FieldKey
from ...config.store import GeoConfigStore


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


def space_stats(space: Dict[str, Any]) -> Tuple[int, int]:
    """
    Return (active_dims, total_keys) for a tuner_search_space dict.
    """
    if not isinstance(space, dict):
        return (0, 0)

    total = len(space)
    active = 0

    for _, v in space.items():
        if v is None:
            continue
        if isinstance(v, list):
            if len(v) > 0:
                active += 1
            continue
        if isinstance(v, dict):
            if len(v) > 0:
                active += 1
            continue

        active += 1

    return (active, total)


def build_plan_text(store: GeoConfigStore) -> str:
    space = _get_fk(store, "tuner_search_space", {})
    max_trials = int(_get_fk(store, "tuner_max_trials", 20))

    obj = str(_get_ui(store, "tune.objective", "") or "").strip()
    algo = str(_get_ui(store, "tune.algorithm", "") or "").strip()

    dev = str(_get_fk(store, "tf_device_mode", "auto") or "auto")
    dev = dev.strip().lower()

    ep = int(_get_fk(store, "epochs", 0))
    bs = int(_get_fk(store, "batch_size", 0))
    lr = float(_get_fk(store, "learning_rate", 0.0))

    a, t = space_stats(space)

    lines = []
    lines.append("Tune plan")
    lines.append(f"• Trials: max={max_trials}")
    if obj or algo:
        parts = []
        if obj:
            parts.append(f"objective={obj}")
        if algo:
            parts.append(f"algo={algo}")
        lines.append("• Search: " + ", ".join(parts))

    if ep or bs or lr:
        lines.append(
            f"• Trial: epochs={ep}, batch={bs}, lr={lr:g}"
        )

    lines.append(f"• Space: {a}/{t} dims active")
    lines.append(f"• Device: {dev}")

    return "\n".join(lines)
