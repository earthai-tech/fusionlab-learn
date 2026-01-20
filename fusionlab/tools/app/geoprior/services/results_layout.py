# geoprior/services/results_layout.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.services.results_layout

Pure helpers for building and parsing the results-root
layout.

Layout rules (current)
---------------------
1) City root directory:
   <city>_<model>_stage<k>
   Example:
   nansha_GeoPriorSubsNet_stage1

2) Runs inside a city root:
   - train_<run_id>/
     Example:
     train_20260117-091736/

   - tuning/run_<run_id>/
   - inference/run_<run_id>/
     Example:
     tuning/run_20251119-192752/
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence


__all__ = [
    "CityRootSpec",
    "RunSpec",
    "build_city_root_name",
    "parse_city_root_name",
    "iter_city_roots",
    "iter_runs",
    "list_city_roots",
    "list_runs",
    "select_best_run",
]
_CITY_ROOT_RE = re.compile(
    r"^(?P<city>[^_]+)_(?P<model>.+)"
    r"_stage(?P<stage>\d+)$"
)

_TRAIN_DIR_RE = re.compile(
    r"^train_(?P<run_id>.+)$"
)

_RUN_DIR_RE = re.compile(
    r"^run_(?P<run_id>.+)$"
)

_FLOW_KINDS: Sequence[str] = (
    "tuning",
    "inference",
)


# --- add near other regex constants ---
_RUN_TS_RE = re.compile(
    r"^(?P<ymd>\d{8})-(?P<hms>\d{6})$"
)

# --- add at the bottom (after list_runs) ---
def select_best_run(
    runs: Sequence[RunSpec],
    *,
    prefer: Sequence[str] = (
        "inference",
        "train",
        "tuning",
    ),
) -> Optional[RunSpec]:
    """
    Select the most relevant run.

    Strategy
    --------
    - Prefer run kinds by order in `prefer`.
    - Within a kind, pick the latest run_id if it looks
      like "YYYYMMDD-HHMMSS".
    - Fallback: use directory mtime, then run_id.

    Returns
    -------
    RunSpec | None
    """
    if not runs:
        return None

    for kind in prefer:
        cand = [r for r in runs if r.kind == kind]
        if cand:
            return max(cand, key=_run_sort_key)

    return max(list(runs), key=_run_sort_key)


def _run_sort_key(run: RunSpec) -> tuple[int, float, str]:
    ts = _run_id_ts_score(run.run_id)
    mt = _path_mtime(run.path)
    return ts, mt, run.run_id


def _run_id_ts_score(run_id: str) -> int:
    m = _RUN_TS_RE.match(run_id or "")
    if m is None:
        return 0
    ymd = m.group("ymd")
    hms = m.group("hms")
    try:
        return int(f"{ymd}{hms}")
    except ValueError:
        return 0


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@dataclass(frozen=True, slots=True)
class CityRootSpec:
    """Parsed city root folder description."""

    city: str
    model: str
    stage: int
    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Parsed run folder description."""

    kind: str
    run_id: str
    path: Path


def build_city_root_name(
    *,
    city: str,
    model: str,
    stage: int = 1,
) -> str:
    """
    Build a city root directory name.

    Parameters
    ----------
    city:
        City key (already normalized if needed).
    model:
        Model name (may include underscores).
    stage:
        Stage integer (default: 1).

    Returns
    -------
    str
        City root folder name:
        "<city>_<model>_stage<stage>".
    """
    return f"{city}_{model}_stage{stage}"


def parse_city_root_name(
    name: str,
) -> Optional[tuple[str, str, int]]:
    """
    Parse a city root directory name.

    Parameters
    ----------
    name:
        Folder name like "<city>_<model>_stage<k>".

    Returns
    -------
    (city, model, stage) or None
    """
    m = _CITY_ROOT_RE.match(name)
    if m is None:
        return None

    city = m.group("city")
    model = m.group("model")
    stage_s = m.group("stage")

    try:
        stage = int(stage_s)
    except ValueError:
        return None

    return city, model, stage


def iter_city_roots(
    results_root: Path,
) -> Iterator[CityRootSpec]:
    """
    Yield city root folders found under results_root.

    Notes
    -----
    - Ignores non-directories.
    - Ignores folders starting with "_" or ".".
    """
    if not results_root.exists():
        return
    if not results_root.is_dir():
        return

    for p in results_root.iterdir():
        if not p.is_dir():
            continue

        name = p.name
        if name.startswith("_") or name.startswith("."):
            continue

        parsed = parse_city_root_name(name)
        if parsed is None:
            continue

        city, model, stage = parsed
        yield CityRootSpec(
            city=city,
            model=model,
            stage=stage,
            name=name,
            path=p,
        )


def iter_runs(
    city_root: Path,
) -> Iterator[RunSpec]:
    """
    Yield runs within a city root folder.

    Recognized patterns
    -------------------
    - train_<run_id>/
    - tuning/run_<run_id>/
    - inference/run_<run_id>/
    """
    if not city_root.exists():
        return
    if not city_root.is_dir():
        return

    for p in city_root.iterdir():
        if not p.is_dir():
            continue

        name = p.name

        m_train = _TRAIN_DIR_RE.match(name)
        if m_train is not None:
            yield RunSpec(
                kind="train",
                run_id=m_train.group("run_id"),
                path=p,
            )
            continue

        if name in _FLOW_KINDS:
            yield from _iter_flow_runs(
                kind=name,
                flow_dir=p,
            )


def _iter_flow_runs(
    *,
    kind: str,
    flow_dir: Path,
) -> Iterator[RunSpec]:
    for p in flow_dir.iterdir():
        if not p.is_dir():
            continue

        m = _RUN_DIR_RE.match(p.name)
        if m is None:
            continue

        yield RunSpec(
            kind=kind,
            run_id=m.group("run_id"),
            path=p,
        )


def list_city_roots(
    results_root: Path,
) -> list[CityRootSpec]:
    """List city roots (materialized)."""
    return list(iter_city_roots(results_root))


def list_runs(
    city_root: Path,
) -> list[RunSpec]:
    """List runs (materialized)."""
    return list(iter_runs(city_root))
