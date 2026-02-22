# geoprior/ui/tune/status.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

from ...config.store import GeoConfigStore

from .plan import _get_fk, _get_ui, space_stats


def _ok(txt: str = "OK") -> Dict[str, str]:
    return {"status": "ok", "text": txt}


def _warn(txt: str = "Fix") -> Dict[str, str]:
    return {"status": "warn", "text": txt}


def _err(txt: str = "Err") -> Dict[str, str]:
    return {"status": "err", "text": txt}


def compute_tune_nav(store: GeoConfigStore) -> Dict[str, Dict[str, str]]:
    """
    Keys:
      - space
      - algo
      - trial
      - compute
      - adv
    """
    out: Dict[str, Dict[str, str]] = {}

    space = _get_fk(store, "tuner_search_space", {})
    a, t = space_stats(space)
    if a <= 0:
        out["space"] = _warn("Add")
    else:
        out["space"] = _ok("OK")

    obj = str(_get_fk(store, "tuner_objective", "") or "").strip()
    if not obj:
        obj = str(_get_ui(store, "tune.objective", "") or "").strip()

    algo = str(_get_ui(store, "tune.algorithm", "") or "").strip()
    if not algo:
        algo = str(_get_fk(store, "tuner_algo", "") or "").strip()
        
    if (not obj) and (not algo):
        out["algo"] = _warn("Set")
    else:
        out["algo"] = _ok("OK")

    ep = int(_get_fk(store, "epochs", 0))
    lr = float(_get_fk(store, "learning_rate", 0.0))
    if ep <= 0 or lr <= 0.0:
        out["trial"] = _warn("Fix")
    else:
        out["trial"] = _ok("OK")

    dev = str(_get_fk(store, "tf_device_mode", "auto") or "auto")
    dev = dev.strip().lower()
    if dev not in {"auto", "cpu", "gpu"}:
        out["compute"] = _warn("Dev?")
    else:
        out["compute"] = _ok("OK")

    out["adv"] = _ok("OK")
    return out
