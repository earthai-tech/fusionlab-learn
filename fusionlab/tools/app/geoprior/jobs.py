# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Small helpers for managing GeoPrior training jobs.

This module is intentionally GUI-agnostic: it exposes simple
dataclasses and functions that summarise which Stage-1 runs
exist under a given results root, and converts them into
"job specs" that the GUI can execute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from .config import ( 
    GeoConfigStore, 
    Stage1Summary, 
    discover_stage1_runs
)

JobMode = Literal["reuse", "rebuild"]

@dataclass
class TuneRunInfo:
    """
    Lightweight summary of the *latest* tuning run for a city.

    Attributes
    ----------
    city : str
        City / dataset label.
    run_dir : Path
        Directory of the tuning run, e.g.
        ``.../nansha_GeoPriorSubsNet_stage1/tuning/run_20251121-141933``.
    summary_path : Optional[Path]
        Path to ``tuning_summary.json`` if present.
    summary : Optional[Dict[str, Any]]
        Parsed JSON content of ``tuning_summary.json`` (may be ``None``).
    """

    city: str
    run_dir: Path
    summary_path: Optional[Path]
    summary: Optional[Dict[str, Any]]

def _resolve_current_cfg(
    *,
    current_cfg: Optional[Dict[str, Any]],
    store: Optional["GeoConfigStore"],
    config_overwrite: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    base: Dict[str, Any] = dict(current_cfg or {})

    if not base and store is not None:
        try:
            base = dict(store.snapshot_overrides() or {})
        except Exception:
            base = {}

    if config_overwrite:
        base.update(config_overwrite)

    return base or None

def _latest_tune_run_for_stage1(
        stage1_dir: Path, city: str
    ) -> Optional[TuneRunInfo]:
    """
    Inspect a Stage-1 directory and, if a ``tuning/`` subfolder exists,
    return a summary of the latest tuning run (by folder name).

    Parameters
    ----------
    stage1_dir : Path
        Path to ``<city>_GeoPriorSubsNet_stage1``.
    city : str
        City label.

    Returns
    -------
    TuneRunInfo or None
        Info about the latest tuning run, or ``None`` if no runs found.
    """
    tuning_root = stage1_dir / "tuning"
    if not tuning_root.is_dir():
        return None

    run_dirs = sorted(
        [
            p for p in tuning_root.iterdir()
            if p.is_dir() and p.name.startswith("run_")
        ]
    )
    if not run_dirs:
        return None

    run_dir = run_dirs[-1]  # last = latest by timestamp naming
    summary_path = run_dir / "tuning_summary.json"
    summary: Optional[Dict[str, Any]] = None
    if summary_path.is_file():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = None

    return TuneRunInfo(
        city=city,
        run_dir=run_dir,
        summary_path=summary_path if summary_path.exists() else None,
        summary=summary,
    )

def discover_tune_jobs_for_root(
    results_root: Path,
    *,
    store: Optional["GeoConfigStore"] = None,
    current_cfg: Optional[Dict[str, Any]] = None,
    config_overwrite: Optional[Dict[str, Any]] = None,
) -> Dict[str, TuneRunInfo]:
    """
    Return a mapping of city -> latest tuning run info (if any) for the
    given results root.

    This *does not* filter by whether tuning runs exist: if there is a
    Stage-1 directory but no ``tuning/`` folder yet, the city will still
    appear in the Training dialog via ``latest_jobs_for_root``; here we
    only provide extra info for the Tune dialogs.

    Parameters
    ----------
    results_root : Path
        Root directory for GeoPrior runs (e.g. ``.fusionlab_runs`` or
        ``F:/tests``).

    Returns
    -------
    dict
        Mapping ``city -> TuneRunInfo`` for cities with at least one
        tuning run.
    """
    results_root = Path(results_root)
    tune_infos: Dict[str, TuneRunInfo] = {}

    for job in latest_jobs_for_root(
        results_root,
        current_cfg=current_cfg,
        store=store,
        config_overwrite=config_overwrite,
    ):
        # Skip cities that don't actually have a Stage-1 summary
        if job.stage1_summary is None:
            continue

        stage1_dir = job.stage1_summary.run_dir
        city = job.city

        tune_info = _latest_tune_run_for_stage1(stage1_dir, city)
        if tune_info is not None:
            tune_infos[city] = tune_info

    return tune_infos

@dataclass
class TrainJobSpec:
    """
    Lightweight description of a training job.

    Parameters
    ----------
    city :
        Name of the city / dataset.
    results_root :
        Base directory containing all GeoPrior runs for this
        project (e.g. ``~/.fusionlab_runs`` or a user-chosen
        folder).
    stage1_summary :
        Summary of the Stage-1 run that will be used for this
        job. ``None`` indicates that Stage-1 still needs to be
        run.
    mode :
        Either ``"reuse"`` (use existing Stage-1 artifacts) or
        ``"rebuild"`` (rerun Stage-1 from scratch before
        training).
    """

    city: str
    results_root: Path
    stage1_summary: Optional[Stage1Summary]
    mode: JobMode = "reuse"

    @property
    def has_stage1(self) -> bool:
        """Return True if this job already has a Stage-1 summary."""
        return self.stage1_summary is not None

    @property
    def manifest_path(self) -> Optional[Path]:
        """
        Convenience accessor for the underlying Stage-1 manifest
        path.

        Returns
        -------
        pathlib.Path or None
        """
        if self.stage1_summary is None:
            return None
        return self.stage1_summary.manifest_path

    @property
    def timestamp(self) -> Optional[str]:
        """Timestamp of the associated Stage-1 run, if any."""
        if self.stage1_summary is None:
            return None
        return self.stage1_summary.timestamp

def group_stage1_by_city(
    results_root: Path,
    current_cfg: Optional[Dict[str, Any]] = None,
    *,
    store: Optional["GeoConfigStore"] = None,
    config_overwrite: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Stage1Summary]]:
    """
    Group all discovered Stage-1 runs by city.

    Parameters
    ----------
    results_root :
        Root directory to search for ``manifest.json`` files.
    current_cfg :
        Optional minimal Stage-1 config snapshot. If given, it will
        be passed through to :func:`discover_stage1_runs` so that
        ``Stage1Summary.config_match`` and ``config_diffs`` are
        computed relative to the current GUI settings.

    Returns
    -------
    dict
        Mapping ``city -> [Stage1Summary, ...]``. Lists are ordered
        in the same way as the underlying
        :func:`discover_stage1_runs` result (currently sorted by
        timestamp).
    """
    results_root = Path(results_root)
    cur = _resolve_current_cfg(
        current_cfg=current_cfg,
        store=store,
        config_overwrite=config_overwrite,
    )
    summaries = discover_stage1_runs(
        results_root,
        current_cfg=cur,
    )

    by_city: Dict[str, List[Stage1Summary]] = {}
    for s in summaries:
        by_city.setdefault(s.city, []).append(s)
    return by_city


def latest_jobs_for_root(
    results_root: Path,
    current_cfg: Optional[Dict[str, Any]] = None,
    default_mode: JobMode = "reuse",
    *,
    store: Optional["GeoConfigStore"] = None,
    config_overwrite: Optional[Dict[str, Any]] = None,
) -> List[TrainJobSpec]:
    """
    Build one :class:`TrainJobSpec` per city under ``results_root``.

    For each city we keep only the *latest* Stage-1 run according
    to the timestamp embedded in its manifest.

    Parameters
    ----------
    results_root :
        Root directory to search for Stage-1 manifests.
    current_cfg :
        Optional minimal Stage-1 config snapshot (see
        :func:`group_stage1_by_city`).
    default_mode :
        Initial value for the ``mode`` attribute of each job. The
        GUI can later change this to ``"rebuild"`` if the user
        explicitly requests a fresh Stage-1.

    Returns
    -------
    list of TrainJobSpec
        Jobs are sorted alphabetically by city name.
    """
    results_root = Path(results_root)
    by_city = group_stage1_by_city(
        results_root,
        current_cfg=current_cfg,
        store=store,
        config_overwrite=config_overwrite,
    )

    jobs: List[TrainJobSpec] = []
    for city, runs in by_city.items():
        if not runs:
            continue
        # discover_stage1_runs already returns summaries sorted by
        # timestamp ascending, so the last one is the latest.
        latest = runs[-1]
        jobs.append(
            TrainJobSpec(
                city=city,
                results_root=results_root,
                stage1_summary=latest,
                mode=default_mode,
            )
        )

    # Stable order: alphabetical by city (case-insensitive).
    jobs.sort(key=lambda j: j.city.lower())
    return jobs
