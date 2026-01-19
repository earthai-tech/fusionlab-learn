# geoprior/ui/xfer/index.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.index

Reusable discovery layer for Xfer UI.

Wraps:
- ui/map/utils.py : scan_results_root()

Provides:
- caching
- small selectors
- a single place for "guess best file" heuristics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ..map.utils import (
    MapCity,
    MapFile,
    MapJob,
    scan_results_root,
)


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


@dataclass(frozen=True)
class XferIndex:
    root: Path
    cities_t: Tuple[MapCity, ...]

    # -------------------------
    # Basic selectors
    # -------------------------
    def cities(self) -> List[str]:
        return [c.city for c in self.cities_t]

    def city_obj(self, city: str) -> Optional[MapCity]:
        key = _norm(city)
        for c in self.cities_t:
            if _norm(c.city) == key:
                return c
        return None

    def jobs(self, city: str) -> List[MapJob]:
        c = self.city_obj(city)
        if not c:
            return []
        return list(c.jobs)

    def job(
        self,
        city: str,
        job_kind: str,
        job_id: str,
    ) -> Optional[MapJob]:
        c = self.city_obj(city)
        if not c:
            return None

        k = _norm(job_kind)
        j = str(job_id or "").strip()
        for jb in c.jobs:
            if _norm(jb.kind) == k and jb.job_id == j:
                return jb
        return None

    def files(
        self,
        city: str,
        job_kind: str,
        job_id: str,
        *,
        split: Optional[str] = None,
    ) -> List[MapFile]:
        jb = self.job(city, job_kind, job_id)
        if not jb:
            return []
        if not split:
            return list(jb.files)

        sp = _norm(split)
        return [f for f in jb.files if _norm(f.kind) == sp]

    # -------------------------
    # Heuristic selector
    # -------------------------
    def guess_xfer_file(
        self,
        city_a: str,
        city_b: str,
        *,
        split: str = "val",
        prefer_tuned: bool = True,
    ) -> Optional[Path]:
        """
        Return a best-guess forecast CSV path for:
        (city_a model) producing forecasts for (city_b).

        Heuristic (minimal for now):
        - prefer tuned jobs if requested
        - prefer newest job_id
        - prefer file names containing:
          "{city_b}_" and "xfer" and "forecast"
        - fallback to any file of given split
        """
        c = self.city_obj(city_a)
        if not c:
            return None

        want_peer = _norm(city_b)
        sp = _norm(split)

        kinds = ["tuning", "train", "inference"]
        if not prefer_tuned:
            kinds = ["train", "tuning", "inference"]

        jobs = list(c.jobs)
        # jobs were scanned sorted; pick newest first
        jobs.sort(key=lambda j: j.job_id, reverse=True)

        for kind in kinds:
            for jb in jobs:
                if _norm(jb.kind) != _norm(kind):
                    continue

                cand = [
                    f
                    for f in jb.files
                    if _norm(f.kind) == sp
                ]
                if not cand:
                    continue

                best = self._pick_peer_file(
                    cand,
                    peer=want_peer,
                )
                if best:
                    return best.path

        return None

    def _pick_peer_file(
        self,
        files: Sequence[MapFile],
        *,
        peer: str,
    ) -> Optional[MapFile]:
        """
        Prefer: '{peer}_...xfer...forecast...' in name.
        """
        best = None
        best_score = -1

        for f in files:
            name = _norm(f.path.name)

            score = 0
            if peer and peer in name:
                score += 5
            if "xfer" in name:
                score += 3
            if "forecast" in name:
                score += 2
            if "calibrated" in name:
                score += 1

            if score > best_score:
                best_score = score
                best = f

        return best


class XferIndexCache:
    """
    Cache wrapper to avoid rescanning on every UI change.
    """

    def __init__(self) -> None:
        self._root: Optional[Path] = None
        self._idx: Optional[XferIndex] = None

    def scan(self, results_root: Path) -> XferIndex:
        root = Path(results_root).expanduser()

        if self._idx is not None and self._root == root:
            return self._idx

        cities = tuple(scan_results_root(root))
        idx = XferIndex(root=root, cities_t=cities)
        self._root = root
        self._idx = idx
        return idx

    def clear(self) -> None:
        self._root = None
        self._idx = None
