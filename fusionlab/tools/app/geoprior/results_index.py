# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Index GeoPrior results (artifacts, train, tuning, inference, xfer).

This module is GUI-agnostic. It exposes small dataclasses and a single
entry point :func:`discover_results_for_root` that summarises all
available jobs under a given ``results_root``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import json

from .config import Stage1Summary
from .jobs import latest_jobs_for_root


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
@dataclass
class TrainRunSummary:
    city: str
    stamp: str
    run_dir: Path
    summary_json: Optional[Path] = None


@dataclass
class TuneRunSummary:
    city: str
    stamp: str
    run_dir: Path
    tuning_summary: Optional[Path] = None
    best_model: Optional[Path] = None
    best_hps: Optional[Path] = None


@dataclass
class InferenceRunSummary:
    city: str
    stamp: str
    run_dir: Path
    dataset: Optional[str] = None
    summary_json: Optional[Path] = None


@dataclass
class CityResults:
    city: str
    stage1_dir: Path
    artifacts_dir: Optional[Path] = None
    train_runs: List[TrainRunSummary] = field(default_factory=list)
    tune_runs: List[TuneRunSummary] = field(default_factory=list)
    inference_runs: List[InferenceRunSummary] = field(default_factory=list)


@dataclass
class XferRunSummary:
    city_a: str
    city_b: str
    stamp: str
    run_dir: Path
    json_path: Optional[Path] = None
    csv_path: Optional[Path] = None


@dataclass
class ResultsIndex:
    results_root: Path
    cities: Dict[str, CityResults] = field(default_factory=dict)
    xfer_runs: List[XferRunSummary] = field(default_factory=list)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_read_json(path: Path) -> dict:
    if not path or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _discover_train_runs(stage1_dir: Path, city: str) -> List[TrainRunSummary]:
    runs: List[TrainRunSummary] = []
    for d in sorted(stage1_dir.glob("train_*")):
        if not d.is_dir():
            continue
        stamp = d.name.replace("train_", "", 1)
        summary_json = None
        # Any *_training_summary.json in that folder
        for cand in d.glob("*_training_summary.json"):
            summary_json = cand
            break
        runs.append(
            TrainRunSummary(
                city=city,
                stamp=stamp,
                run_dir=d,
                summary_json=summary_json,
            )
        )
    return runs


def _discover_tune_runs(stage1_dir: Path, city: str) -> List[TuneRunSummary]:
    base = stage1_dir / "tuning"
    runs: List[TuneRunSummary] = []
    if not base.is_dir():
        return runs

    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if not d.name.startswith("run_"):
            continue
        stamp = d.name.replace("run_", "", 1)
        tuning_summary = d / "tuning_summary.json"

        best_model = None
        best_hps = None
        # Look for best model and best HPS JSON in that run folder
        for cand in d.glob(f"{city}_GeoPrior_best.keras"):
            best_model = cand
            break
        for cand in d.glob(f"{city}_GeoPrior_best_hps.json"):
            best_hps = cand
            break

        runs.append(
            TuneRunSummary(
                city=city,
                stamp=stamp,
                run_dir=d,
                tuning_summary=tuning_summary if tuning_summary.is_file() else None,
                best_model=best_model,
                best_hps=best_hps,
            )
        )
    return runs

def _discover_inference_runs(stage1_dir: Path, city: str) -> List[InferenceRunSummary]:
    """Discover inference runs for a given Stage-1 directory.

    Supports both the old ``run_<stamp>`` convention and the newer
    ``<stamp>``-only convention used for inference job folders. Any
    directory under ``stage1_dir / "inference"`` is treated as a run
    directory; the ``stamp`` is normalised by stripping a leading
    ``"run_"`` prefix if present.
    """
    base = stage1_dir / "inference"
    runs: List[InferenceRunSummary] = []
    if not base.is_dir():
        return runs

    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        name = d.name
        if name.startswith("run_"):
            stamp = name.replace("run_", "", 1)
        else:
            stamp = name

        summary_json = d / "inference_summary.json"
        dataset: Optional[str] = None
        if summary_json.is_file():
            payload = _safe_read_json(summary_json)
            dataset = payload.get("dataset")

        runs.append(
            InferenceRunSummary(
                city=city,
                stamp=stamp,
                run_dir=d,
                dataset=dataset,
                summary_json=summary_json if summary_json.is_file() else None,
            )
        )
    return runs


def _discover_city_results(
    stage1_summary: Stage1Summary,
) -> CityResults:
    city = stage1_summary.city
    stage1_dir = Path(stage1_summary.run_dir)
    artifacts_dir = stage1_dir / "artifacts"
    if not artifacts_dir.is_dir():
        artifacts_dir = None

    return CityResults(
        city=city,
        stage1_dir=stage1_dir,
        artifacts_dir=artifacts_dir,
        train_runs=_discover_train_runs(stage1_dir, city),
        tune_runs=_discover_tune_runs(stage1_dir, city),
        inference_runs=_discover_inference_runs(stage1_dir, city),
    )


def _discover_xfer_runs(results_root: Path) -> List[XferRunSummary]:
    xfer_root = results_root / "xfer"
    out: List[XferRunSummary] = []
    if not xfer_root.is_dir():
        return out

    for pair_dir in sorted(p for p in xfer_root.iterdir() if p.is_dir()):
        name = pair_dir.name
        if "_to_" not in name:
            continue
        city_a, city_b = name.split("_to_", 1)
        for run_dir in sorted(p for p in pair_dir.iterdir() if p.is_dir()):
            stamp = run_dir.name
            json_path = run_dir / "xfer_results.json"
            csv_path = run_dir / "xfer_results.csv"
            out.append(
                XferRunSummary(
                    city_a=city_a,
                    city_b=city_b,
                    stamp=stamp,
                    run_dir=run_dir,
                    json_path=json_path if json_path.is_file() else None,
                    csv_path=csv_path if csv_path.is_file() else None,
                )
            )
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def discover_results_for_root(results_root: Path | str) -> ResultsIndex:
    """
    Discover all result artefacts under ``results_root``.

    Uses :func:`latest_jobs_for_root` so each city is anchored on the
    latest Stage-1 summary (one ``stage1_dir`` per city).
    """
    root = Path(results_root).expanduser().resolve()
    index = ResultsIndex(results_root=root)

    for job in latest_jobs_for_root(root):
        if job.stage1_summary is None:
            continue
        city = job.city
        if city in index.cities:
            # Already discovered (should not usually happen, but keep
            # behaviour stable).
            continue
        index.cities[city] = _discover_city_results(job.stage1_summary)

    index.xfer_runs = _discover_xfer_runs(root)
    return index



