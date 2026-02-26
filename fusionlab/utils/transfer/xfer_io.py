# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""fusionlab.utils.transfer.xfer_io

I/O helpers for the GeoPrior transferability matrix.

Folder layout supported
-----------------------
Given a pair folder like::

  results/xfer/nansha__zhongshan/
    20260127-085337/
      xfer_results.csv
      xfer_results.json
      nansha_to_zhongshan_*_eval.csv
      nansha_to_zhongshan_*_future.csv

This module resolves:
- a "src" that is already a run dir
- a "src" that is a pair dir (pick newest timestamp)
- explicit csv/json paths

All functions accept str/Path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


_TS_RE = re.compile(r"^\d{8}-\d{6}$")


@dataclass(frozen=True)
class XferResolved:
    """Resolved paths for a transfer run."""

    run_dir: Path
    csv_path: Path
    json_path: Optional[Path]


def as_path(x: Any) -> Path:
    """Coerce an input into a ``Path``."""
    if isinstance(x, Path):
        return x
    return Path(str(x))


def is_timestamp_dir(p: Path) -> bool:
    """Return True if ``p.name`` matches YYYYMMDD-HHMMSS."""
    try:
        return bool(_TS_RE.match(p.name)) and p.is_dir()
    except OSError:
        return False


def list_timestamp_dirs(pair_dir: Path) -> List[Path]:
    """List timestamp-named run dirs under a pair dir."""
    pair_dir = as_path(pair_dir)
    if not pair_dir.exists():
        return []
    out: List[Path] = []
    for c in pair_dir.iterdir():
        if is_timestamp_dir(c):
            out.append(c)
    out.sort(key=lambda p: p.name, reverse=True)
    return out


def resolve_run_dir(src: Any) -> Path:
    """Resolve ``src`` into a concrete run directory.

    Rules
    -----
    - If ``src`` is a directory containing xfer_results.csv,
      treat it as a run dir.
    - Else if ``src`` is a pair directory containing multiple
      timestamp dirs, pick the newest.

    Raises
    ------
    FileNotFoundError
        If no run dir can be resolved.
    """
    p = as_path(src)

    if p.is_file():
        p = p.parent

    if p.is_dir() and (p / "xfer_results.csv").exists():
        return p

    # Pair directory case.
    runs = list_timestamp_dirs(p)
    if runs:
        run = runs[0]
        if (run / "xfer_results.csv").exists():
            return run

    raise FileNotFoundError(
        f"Could not resolve run dir from: {src!r}"
    )


def resolve_xfer_paths(src: Any) -> XferResolved:
    """Resolve paths to xfer_results.{csv,json}.

    Parameters
    ----------
    src:
        Run dir, pair dir, or a direct path.

    Returns
    -------
    XferResolved
        Resolved run_dir and artifact paths.
    """
    p = as_path(src)

    if p.is_file() and p.name.endswith(".csv"):
        run_dir = p.parent
        csv_path = p
        json_path = run_dir / "xfer_results.json"
        if not json_path.exists():
            json_path = None
        return XferResolved(
            run_dir=run_dir,
            csv_path=csv_path,
            json_path=json_path,
        )

    if p.is_file() and p.name.endswith(".json"):
        run_dir = p.parent
        csv_path = run_dir / "xfer_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(str(csv_path))
        return XferResolved(
            run_dir=run_dir,
            csv_path=csv_path,
            json_path=p,
        )

    run_dir = resolve_run_dir(p)
    csv_path = run_dir / "xfer_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
    json_path = run_dir / "xfer_results.json"
    if not json_path.exists():
        json_path = None
    return XferResolved(
        run_dir=run_dir,
        csv_path=csv_path,
        json_path=json_path,
    )


def load_xfer_results_csv(src: Any) -> pd.DataFrame:
    """Load xfer_results.csv from a run or pair directory."""
    res = resolve_xfer_paths(src)
    df = pd.read_csv(res.csv_path)
    df["_run_dir"] = str(res.run_dir)
    df["_csv_path"] = str(res.csv_path)
    return df


def load_xfer_results_json(src: Any) -> List[Dict[str, Any]]:
    """Load xfer_results.json as a list of dicts.

    Notes
    -----
    This JSON is your "single record" per job.
    It's often useful for reading per-job file paths.
    """
    res = resolve_xfer_paths(src)
    if res.json_path is None:
        raise FileNotFoundError(
            f"No xfer_results.json under {res.run_dir}"
        )
    with res.json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise TypeError("Expected a JSON list.")
    return obj  # type: ignore[return-value]


# ----------------------------------------------------------
# Per-job CSV loaders
# ----------------------------------------------------------

_REQUIRED_EVAL = (
    "sample_idx",
    "forecast_step",
    "coord_t",
    "subsidence_actual",
    "subsidence_q10",
    "subsidence_q50",
    "subsidence_q90",
    "coord_x",
    "coord_y",
)

_REQUIRED_FUT = (
    "sample_idx",
    "forecast_step",
    "coord_t",
    "subsidence_q10",
    "subsidence_q50",
    "subsidence_q90",
    "coord_x",
    "coord_y",
)


def _ensure_cols(
    df: pd.DataFrame,
    req: Tuple[str, ...],
) -> None:
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns: {miss}")


def load_eval_csv(path: Any) -> pd.DataFrame:
    """Load an ``*_eval.csv`` (with actuals)."""
    p = as_path(path)
    df = pd.read_csv(p)
    _ensure_cols(df, _REQUIRED_EVAL)
    df["_csv_path"] = str(p)
    return df


def load_future_csv(path: Any) -> pd.DataFrame:
    """Load a ``*_future.csv`` (no actuals)."""
    p = as_path(path)
    df = pd.read_csv(p)
    _ensure_cols(df, _REQUIRED_FUT)
    df["_csv_path"] = str(p)
    return df


def iter_job_csv_paths(
    xfer_json: List[Dict[str, Any]],
    *,
    kind: str = "eval",
) -> Iterable[Path]:
    """Yield eval/future csv paths from xfer_results.json."""
    if kind not in {"eval", "future"}:
        raise ValueError("kind must be 'eval' or 'future'.")

    key = "csv_eval" if kind == "eval" else "csv_future"

    for rec in xfer_json:
        if key not in rec:
            continue
        raw = rec.get(key)
        if not raw:
            continue
        yield as_path(raw)
