# geoprior/ui/preprocess/status.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Preprocess navigator chip status computation.

Pure helpers (store in -> dict out).
UI (tab.py) applies chips via PreprocessNavigator.set_chip(...).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...config.store import GeoConfigStore
from ...config.prior_schema import FieldKey


__all__ = ["compute_preprocess_nav"]


def _ok(txt: str = "OK") -> Dict[str, str]:
    return {"status": "ok", "text": txt}


def _warn(txt: str = "Fix") -> Dict[str, str]:
    return {"status": "warn", "text": txt}


def _off(txt: str = "—") -> Dict[str, str]:
    return {"status": "off", "text": txt}


def _get_fk(store: GeoConfigStore, key: str, default: Any) -> Any:
    return store.get_value(FieldKey(key), default=default)


def compute_preprocess_nav(
    store: GeoConfigStore,
    *,
    preview: Optional[Dict[str, Any]] = None,
    decision: str = "",
) -> Dict[str, Dict[str, str]]:
    """
    Keys:
      - inputs
      - policy
      - status
      - plan
    """
    out: Dict[str, Dict[str, str]] = {}

    city = str(_get_fk(store, "city", "") or "").strip()
    rr_raw = _get_fk(store, "results_root", None)
    rr = "" if rr_raw is None else str(rr_raw).strip()
    ds = _get_fk(store, "dataset_path", None)
    ds_ok = bool(ds)

    inputs_ok = bool(city and rr and ds_ok)
    out["inputs"] = _ok("OK") if inputs_ok else _warn("Add")

    auto_reuse = bool(
        _get_fk(store, "stage1_auto_reuse_if_match", True)
    )
    force_rb = bool(
        _get_fk(store, "stage1_force_rebuild_if_mismatch", True)
    )

    if auto_reuse and force_rb:
        out["policy"] = _ok("Auto")
    elif auto_reuse and (not force_rb):
        out["policy"] = _warn("Auto")
    else:
        out["policy"] = _warn("Man")

    dec = str(decision or "").strip()
    if not dec:
        b = preview or {}
        dec = str(b.get("decision") or "").strip()

    if not inputs_ok:
        out["status"] = _off("—")
        out["plan"] = _off("—")
        return out

    if not dec or dec.upper().startswith("WAIT"):
        out["status"] = _warn("Wait")
        out["plan"] = _warn("Wait")
        return out

    if dec == "REUSE":
        out["status"] = _ok("OK")
        out["plan"] = _ok("OK")
        return out

    if dec in ("BUILD", "REBUILD"):
        out["status"] = _warn("Run")
        out["plan"] = _warn("Run")
        return out

    if dec.startswith("FOUND"):
        out["status"] = _warn("Use")
        out["plan"] = _warn("Use")
        return out

    out["status"] = _warn("Fix")
    out["plan"] = _warn("Fix")
    return out
