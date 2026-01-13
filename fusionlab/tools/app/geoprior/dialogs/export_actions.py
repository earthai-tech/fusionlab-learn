# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
export_actions.py

Actual export action for the Tune tab.

Builds an export payload from the GeoConfigStore and writes
it to JSON or YAML using UI preferences saved in cfg._meta.

Kinds:
- search_space
- config_snapshot
- config_overrides
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget

from ..config.store import GeoConfigStore
from ..config.geoprior_config import default_tuner_search_space


_ALLOWED_KIND = (
    "search_space",
    "config_snapshot",
    "config_overrides",
)
_ALLOWED_FMT = ("json", "yaml")


def export_with_saved_prefs(
    store: GeoConfigStore,
    *,
    parent: Optional[QWidget] = None,
) -> bool:
    """
    Export according to cfg._meta preferences.

    Returns True on success.
    """
    try:
        meta = getattr(store.cfg, "_meta", None)
    except Exception:
        meta = None

    if not isinstance(meta, dict):
        meta = {}

    kind = str(meta.get("tuner_export_kind", "search_space") or "")
    fmt = str(meta.get("tuner_export_format", "json") or "json")
    path = str(meta.get("tuner_export_path", "") or "")

    pretty = bool(meta.get("tuner_export_pretty", True))
    inc_def = bool(meta.get("tuner_export_include_defaults", False))
    inc_ov = bool(meta.get("tuner_export_include_overrides", True))

    if kind not in _ALLOWED_KIND:
        kind = "search_space"
    if fmt not in _ALLOWED_FMT:
        fmt = "json"

    out_path = _resolve_export_path(
        path,
        fmt=fmt,
        kind=kind,
        parent=parent,
    )
    if not out_path:
        return False

    payload = build_export_payload(
        store,
        kind=kind,
        include_defaults=inc_def,
        include_overrides=inc_ov,
    )

    try:
        _write_payload(out_path, payload, fmt=fmt, pretty=pretty)
    except Exception as exc:
        QMessageBox.critical(
            parent,
            "Export failed",
            str(exc) or "Failed to export.",
        )
        return False

    # Persist the selected path back to meta.
    with store.batch():
        store.merge_dict_field(
            "_meta",
            {"tuner_export_path": str(out_path)},
            replace=False,
        )

    QMessageBox.information(
        parent,
        "Export complete",
        f"Wrote: {out_path}",
    )
    return True


def build_export_payload(
    store: GeoConfigStore,
    *,
    kind: str,
    include_defaults: bool,
    include_overrides: bool,
) -> Dict[str, Any]:
    """
    Build export payload object.
    """
    cfg = store.cfg

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    header: Dict[str, Any] = {
        "__kind__": str(kind),
        "__created__": stamp,
        "__city__": str(getattr(cfg, "city", "") or ""),
        "__dataset_path__": _as_str(getattr(cfg, "dataset_path", None)),
        "__results_root__": _as_str(getattr(cfg, "results_root", None)),
    }

    if kind == "search_space":
        cur = dict(getattr(cfg, "tuner_search_space", {}) or {})
        out: Dict[str, Any] = {"search_space": _jsonable(cur)}

        if include_defaults:
            dflt = default_tuner_search_space(
                offset_mode=str(getattr(cfg, "offset_mode", "mul") or "mul"),
            )
            out["defaults"] = _jsonable(dflt)

        if include_overrides:
            dflt2 = default_tuner_search_space(
                offset_mode=str(getattr(cfg, "offset_mode", "mul") or "mul"),
            )
            out["overrides"] = _jsonable(
                _diff_dict(cur, dflt2),
            )

        header.update(out)
        return header

    if kind == "config_overrides":
        overrides = store.snapshot_overrides()
        header["overrides"] = _jsonable(overrides)

        if include_defaults:
            base = getattr(cfg, "_base_cfg", {}) or {}
            header["base_config"] = _jsonable(base)

        return header

    # kind == "config_snapshot"
    base_cfg = getattr(cfg, "_base_cfg", {}) or {}
    overrides2 = store.snapshot_overrides()

    resolved = dict(base_cfg)
    resolved.update(overrides2)

    header["resolved_config"] = _jsonable(resolved)

    if include_defaults:
        header["base_config"] = _jsonable(base_cfg)

    if include_overrides:
        header["overrides"] = _jsonable(overrides2)

    return header


# ---------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------
def _write_payload(
    path: Path,
    payload: Dict[str, Any],
    *,
    fmt: str,
    pretty: bool,
) -> None:
    path = Path(path)

    os.makedirs(str(path.parent), exist_ok=True)

    if fmt == "json":
        indent = 2 if pretty else None
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                indent=indent,
                ensure_ascii=True,
                sort_keys=False,
            )
        return

    # YAML
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "YAML export requires 'pyyaml'. "
            "Install it or switch to JSON."
        ) from exc

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            payload,
            f,
            sort_keys=False,
            allow_unicode=False,
        )


# ---------------------------------------------------------------------
# Path chooser
# ---------------------------------------------------------------------
def _resolve_export_path(
    raw: str,
    *,
    fmt: str,
    kind: str,
    parent: Optional[QWidget],
) -> Optional[Path]:
    fmt = str(fmt).strip().lower()
    ext = "json" if fmt == "json" else "yaml"

    p = (raw or "").strip()
    if not p:
        sugg = f"{kind}.{ext}"
        flt = f"{ext.upper()} Files (*.{ext});;All Files (*)"
        path, _ = QFileDialog.getSaveFileName(
            parent,
            "Export to file",
            sugg,
            flt,
        )
        if not path:
            return None
        p = path

    out = Path(p).expanduser()

    if fmt == "json" and out.suffix.lower() != ".json":
        out = out.with_suffix(".json")

    if fmt == "yaml":
        if out.suffix.lower() not in (".yaml", ".yml"):
            out = out.with_suffix(".yaml")

    return out


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, Path):
        return str(x)
    return str(x)


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _diff_dict(cur: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (cur or {}).items():
        if k not in base:
            out[k] = v
            continue
        if base.get(k) != v:
            out[k] = v
    return out
