# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Helpers for discovering GUI-managed datasets for a given city.
#
# These functions are UI-agnostic and only deal with the filesystem.
# The GeoPrior GUI can build a nice dialog on top of them.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DatasetCandidate:
    """
    Small record describing a dataset CSV.

    Parameters
    ----------
    city : str
        Normalised city/dataset slug (spaces replaced by '_').
    path : Path
        Absolute path to the CSV file.
    root_kind : str
        Either "results_root" (current GUI root) or "default_root"
        (fallback under ``~/.fusionlab_runs``).
    """

    city: str
    path: Path
    root_kind: str  # "results_root" or "default_root"

    @property
    def mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0

    def pretty_root(self) -> str:
        if self.root_kind == "results_root":
            return "Current results root"
        if self.root_kind == "default_root":
            return "Default runs (~/.fusionlab_runs)"
        return self.root_kind


def _city_slug(city: str) -> str:
    slug = (city or "").strip()
    if not slug:
        slug = "geoprior_city"
    return slug.replace(" ", "_")


def collect_datasets_for_city(
    root: Path,
    city: str,
    *,
    root_kind: str,
) -> List[DatasetCandidate]:
    """
    Return all CSV datasets for a given city under ``root/_datasets``.

    Matching pattern
    ----------------
    - ``{city}.csv``
    - ``{city}_<int>.csv``

    Parameters
    ----------
    root : Path
        Results root (GUI runs root or fallback root).
    city : str
        City name from config / GUI.
    root_kind : str
        Label describing the root ("results_root" / "default_root").

    Returns
    -------
    list of DatasetCandidate
        Sorted by modification time (newest first).
    """
    root = Path(root)
    ds_dir = root / "_datasets"
    if not ds_dir.is_dir():
        return []

    slug = _city_slug(city)
    candidates: List[DatasetCandidate] = []

    for p in ds_dir.glob("*.csv"):
        stem = p.stem
        if stem == slug or stem.startswith(f"{slug}_"):
            candidates.append(
                DatasetCandidate(
                    city=slug,
                    path=p.resolve(),
                    root_kind=root_kind,
                )
            )

    # Newest first
    candidates.sort(key=lambda c: c.mtime, reverse=True)
    return candidates

