
from typing import Optional, Dict, Any 
 
from .store import GeoConfigStore

def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _store_cfg_overrides(
    store: Optional["GeoConfigStore"],
) -> Dict[str, Any]:
    if store is None:
        return {}
    try:
        snap = store.snapshot_overrides()
    except Exception:
        return {}
    return dict(snap or {})

def _merge_overrides(
    *parts: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in parts:
        if p:
            out.update(p)
    return out