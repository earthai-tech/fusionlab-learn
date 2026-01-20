# geoprior/services/city_manager.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.services.city_manager

Store-backed city service (single source of truth).

This service centralizes:
- normalization + validation of city keys
- lock/unlock behavior (UI-only via store._extra)
- dataset-open resolution rules
- city-root folder creation under results_root
- city discovery from results_root

Notes
-----
- Canonical city is stored in GeoPriorConfig.city.
- Lock state is stored in store._extra["city.locked"].
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .results_layout import (
    build_city_root_name,
    iter_city_roots,
)

from ..config.store import GeoConfigStore


__all__ = [
    "CityResolveResult",
    "CityManager",
]


_LOCK_KEY = "city.locked"

# We disallow "_" because city root folders use:
#   <city>_<model>_stage1
# Keeping city free of "_" avoids ambiguity.
_INVALID_SEP_RE = re.compile(r"[_\s/\\]+")

_ALLOWED_RE = re.compile(r"[^a-z0-9\-]+")

_DASH_RE = re.compile(r"-{2,}")


@dataclass(frozen=True, slots=True)
class CityResolveResult:
    """
    Result of resolving a city key for dataset-open.

    status:
        "ok" | "warn" | "bad"
    source:
        "locked" | "typed" | "dataset" | "fallback"
    """

    city: str
    status: str
    message: str
    source: str


class CityManager:
    """
    Store-backed city manager.

    Parameters
    ----------
    store:
        GeoConfigStore instance (single source of truth).
    """

    def __init__(
        self,
        store: GeoConfigStore,
    ) -> None:
        self._store = store

    # -------------------------------------------------
    # Normalization + validation
    # -------------------------------------------------
    def normalize(
        self,
        raw: str,
    ) -> str:
        """
        Normalize raw city input into a safe city key.

        Rules
        -----
        - lowercase
        - replace whitespace/underscores/slashes with "-"
        - keep only [a-z0-9-]
        - collapse repeated "-"
        - strip leading/trailing "-"
        """
        s = (raw or "").strip().lower()
        if not s:
            return ""

        s = _INVALID_SEP_RE.sub("-", s)
        s = _ALLOWED_RE.sub("", s)
        s = _DASH_RE.sub("-", s)
        s = s.strip("-")

        return s

    def validate(
        self,
        raw: str,
    ) -> tuple[bool, str, str]:
        """
        Validate raw city input.

        Returns
        -------
        ok, city_key, message
        """
        raw_s = (raw or "").strip()
        key = self.normalize(raw_s)

        if not key:
            msg = (
                "City is empty. Please rename your dataset "
                "or type a city name."
            )
            return False, "", msg

        if key != raw_s.lower():
            msg = f"City normalized to '{key}'."
            return True, key, msg

        return True, key, ""

    # -------------------------------------------------
    # Store accessors
    # -------------------------------------------------
    def get_city(self) -> str:
        return str(self._store.get("city", "") or "")

    def set_city(
        self,
        city: str,
        *,
        quiet: bool = False,
    ) -> bool:
        ok, key, msg = self.validate(city)
        if not ok:
            if not quiet:
                self._store.error_raised.emit(msg)
            return False

        self._store.set("city", key)
        return True

    # -------------------------------------------------
    # Lock state (UI-only)
    # -------------------------------------------------
    def is_locked(self) -> bool:
        return bool(self._store.get(_LOCK_KEY, False))

    def set_locked(
        self,
        locked: bool,
    ) -> None:
        self._store.set(_LOCK_KEY, bool(locked))

    def toggle_locked(self) -> bool:
        now = not self.is_locked()
        self.set_locked(now)
        return now

    # -------------------------------------------------
    # Dataset-open resolution
    # -------------------------------------------------
    def resolve_city_for_open_dataset(
        self,
        typed_city: Optional[str],
        dataset_path: Optional[Path],
    ) -> CityResolveResult:
        """
        Resolve the city to use when opening a dataset.

        Rules
        -----
        - if locked: keep current city (if non-empty)
        - else if typed city provided: use it
        - else: use dataset stem
        - always normalize + validate
        """
        cur = self.get_city()
        locked = self.is_locked()

        if locked and cur:
            return CityResolveResult(
                city=cur,
                status="ok",
                message="City locked.",
                source="locked",
            )

        raw = (typed_city or "").strip()
        if raw:
            ok, key, msg = self.validate(raw)
            if not ok:
                return CityResolveResult(
                    city="",
                    status="bad",
                    message=msg,
                    source="typed",
                )
            return CityResolveResult(
                city=key,
                status="warn" if msg else "ok",
                message=msg,
                source="typed",
            )

        stem = self._dataset_stem(dataset_path)
        if stem:
            ok, key, msg = self.validate(stem)
            if not ok:
                return CityResolveResult(
                    city="",
                    status="bad",
                    message=msg,
                    source="dataset",
                )
            base = (
                msg
                or "City inferred from dataset name."
            )
            return CityResolveResult(
                city=key,
                status="warn" if msg else "ok",
                message=base,
                source="dataset",
            )

        return CityResolveResult(
            city="",
            status="bad",
            message=(
                "No dataset selected and no city provided."
            ),
            source="fallback",
        )

    def _dataset_stem(
        self,
        dataset_path: Optional[Path],
    ) -> str:
        if dataset_path is None:
            return ""
        try:
            return dataset_path.stem
        except Exception:
            return ""

    # -------------------------------------------------
    # Filesystem helpers
    # -------------------------------------------------
    def ensure_city_root(
        self,
        *,
        results_root: Optional[Path] = None,
        model: Optional[str] = None,
        stage: int = 1,
    ) -> Optional[Path]:
        """
        Ensure the city root folder exists and return it.

        If city is empty, returns None.
        """
        city = self.get_city()
        if not city:
            return None

        cfg = self._store.cfg

        root = results_root or getattr(cfg, "results_root", None)
        if root is None:
            return None

        model_name = model or getattr(cfg, "model_name", None)
        if not model_name:
            model_name = "GeoPriorSubsNet"

        name = build_city_root_name(
            city=city,
            model=model_name,
            stage=stage,
        )
        path = Path(root) / name
        path.mkdir(parents=True, exist_ok=True)

        return path

    def discover_cities(
        self,
        results_root: Optional[Path] = None,
    ) -> list[str]:
        """
        Discover city keys from existing results_root.
        """
        cfg = self._store.cfg
        root = results_root or getattr(cfg, "results_root", None)
        if root is None:
            return []

        cities: set[str] = set()
        for spec in iter_city_roots(Path(root)):
            if spec.city:
                cities.add(spec.city)

        return sorted(cities)

    # -------------------------------------------------
    # Convenience: apply resolution into store
    # -------------------------------------------------
    def apply_resolved_city(
        self,
        res: CityResolveResult,
        *,
        lock_if_ok: bool = False,
    ) -> bool:
        """
        Apply a resolved city to the store.

        Returns
        -------
        bool
            True if applied.
        """
        if not res.city:
            return False

        self._store.set("city", res.city)
        if lock_if_ok and (res.status != "bad"):
            self.set_locked(True)

        return True
