# geoprior/ui/map/utils.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.utils

Scanning helpers for MapTab auto data discovery.

We detect: (v3.0)
<results_root>/
  <city>_<model>_stage1/
    train_<jobid>/
      *_forecast_*_calibrated.csv
      *_forecast_*_future.csv
      geoprior_eval*_interpretable.json",
      geoprior_eval*.json",
    tuning/
      run_<jobid>/
        *.csv
        *.json
    inference/
      run_<jobid>/
        *.csv
We never rename files on disk.
We only generate shorter display labels for the UI.
"""

from __future__ import annotations

import re
import json
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_CITY_PAT = re.compile(
    r"^(?P<city>.+?)_(?P<model>.+?)_(?P<stage>stage\d+)$"
)

_TRAIN_PAT = re.compile(r"^train_(?P<job>.+)$")
_RUN_PAT = re.compile(r"^(?:run|train)_(?P<job>.+)$")


@dataclass(frozen=True)
class MapFile:
    path: Path
    display: str
    kind: str  # "val" | "future" | "other"


@dataclass(frozen=True)
class MapJob:
    kind: str  # "train" | "tuning" | "inference" | ...
    job_id: str
    root: Path
    files: Tuple[MapFile, ...]


@dataclass(frozen=True)
class MapCity:
    city: str
    model: str
    stage: str
    root: Path
    jobs: Tuple[MapJob, ...]


def parse_city_dir(name: str) -> Optional[Tuple[str, str, str]]:
    s = str(name or "").strip()
    m = _CITY_PAT.match(s)
    if not m:
        return None

    return (
        (m.group("city") or "").strip(),
        (m.group("model") or "").strip(),
        (m.group("stage") or "").strip(),
    )


def scan_results_root(results_root: Path) -> List[MapCity]:
    root = Path(results_root).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    out: List[MapCity] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue

        parsed = parse_city_dir(p.name)
        if not parsed:
            continue

        city, model, stage = parsed
        jobs = tuple(scan_city_jobs(p))
        if not jobs:
            continue

        out.append(
            MapCity(
                city=city,
                model=model,
                stage=stage,
                root=p,
                jobs=jobs,
            )
        )
    return out


def scan_city_jobs(city_root: Path) -> List[MapJob]:
    out: List[MapJob] = []

    for p in sorted(city_root.iterdir()):
        if not p.is_dir():
            continue

        m = _TRAIN_PAT.match(p.name)
        if m:
            job_id = (m.group("job") or "").strip()
            files = tuple(scan_job_files(p))
            if files:
                out.append(
                    MapJob(
                        kind="train",
                        job_id=job_id,
                        root=p,
                        files=files,
                    )
                )
            continue

        if p.name in {"tuning", "inference"}:
            out.extend(scan_job_bucket(p, kind=p.name))

    return out


def scan_job_bucket(bucket: Path, *, kind: str) -> List[MapJob]:
    out: List[MapJob] = []

    for p in sorted(bucket.iterdir()):
        if not p.is_dir():
            continue

        m = _RUN_PAT.match(p.name)
        if not m:
            continue

        job_id = (m.group("job") or "").strip()
        files = tuple(scan_job_files(p))
        if not files:
            continue

        out.append(
            MapJob(
                kind=str(kind),
                job_id=job_id,
                root=p,
                files=files,
            )
        )

    return out


def scan_job_files(job_root: Path) -> List[MapFile]:
    """
    Collect forecast CSVs under a job folder.

    We keep it simple:
    - any *.csv
    - prefer those containing "forecast"
    """
    # Also catch nested outputs (e.g. exports/ subfolders).
    csvs = sorted(job_root.rglob("*.csv"))
    if not csvs:
        return []

    forecast = [p for p in csvs if "forecast" in p.name.lower()]
    src = forecast or csvs

    groups: Dict[str, List[Path]] = {
        "val": [],
        "future": [],
        "other": [],
    }

    for p in src:
        k = classify_file(p.name)
        groups[k].append(p)

    out: List[MapFile] = []
    for k in ("val", "future", "other"):
        ps = groups.get(k, [])
        for i, p in enumerate(ps, start=1):
            disp = short_file_label(p.name, kind=k, idx=i)
            out.append(
                MapFile(
                    path=p,
                    display=disp,
                    kind=k,
                )
            )

    return out


def classify_file(filename: str) -> str:
    s = str(filename or "").lower()
    # if "calibrated" in s:
    #     return "val"
    # if "validationset" in s:
    #     return "val"
    if "future" in s:
        return "future"
    # Eval CSVs (Validation/Test) include actuals.
    if "eval" in s:
        return "val"
    if "validationset" in s:
        return "val"
    if "testset" in s:
        return "val"
    # Calibrated eval files are still "val".
    if "calibrated" in s:
        return "val"

    return "other"


def short_file_label(
    filename: str,
    *,
    kind: str,
    idx: int,
) -> str:
    tag = extract_tag(filename)

    low = str(filename or "").lower()
    prefix = ""
    if "testset" in low:
        prefix = "Test"
    elif "validationset" in low:
        prefix = "Val"

    if kind == "val":
        # base = "Val"
        base = prefix or "Val"
        
    elif kind == "future":
        # base = "Future"
        base = f"{prefix} Future" if prefix else "Future"
    else:
        # base = "File"
        base = prefix or "File"

    if idx > 1:
        base = f"{base}{idx}"

    if tag:
        return f"{base} · {tag}"

    return base


def extract_tag(filename: str) -> str:
    """
    Try to extract a short tag from the filename.

    Example: "..._Fallback_H3_future.csv" -> "H3"
    """
    s = str(filename or "")

    if "_Fallback_" in s:
        tail = s.split("_Fallback_", 1)[1]
        tok = tail.split("_", 1)[0].strip()
        if tok:
            return tok
    # Generic: "..._H3_..." -> "H3"
    m = re.search(r"_(H\d+)(?:_|$)", s, flags=re.IGNORECASE)
    if m:
        return str(m.group(1)).upper()

    return ""


def unique_str(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values or []:
        s = str(v or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

@dataclass(frozen=True)
class ForecastMeta:
    cols: Tuple[str, ...]

    n_rows: int
    n_points: int

    time_col: Optional[str]
    step_col: Optional[str]
    sample_col: Optional[str]
    unit_col: Optional[str]

    x_col: Optional[str]
    y_col: Optional[str]

    year_min: Optional[int]
    year_max: Optional[int]
    step_min: Optional[int]
    step_max: Optional[int]

    time_values: Tuple[int, ...]
    value_items: Tuple[Tuple[str, str], ...]

    quantile_label: str
    unit: Optional[str]

_X_CAND = [
    "coord_x",
    "x",
    "easting",
    "lon",
    "longitude",
]

_Y_CAND = [
    "coord_y",
    "y",
    "northing",
    "lat",
    "latitude",
]

_T_CAND = [
    "coord_t",
    "t",
    "time",
    "year",
    "date",
]

_STEP_CAND = [
    "forecast_step",
    "step",
    "horizon",
]

_UNIT_CAND = [
    "subsidence_unit",
    "unit",
    "units",
]

_Q_PAT = re.compile(
    r"^(?P<base>.+?)_q(?P<q>\d{1,3})$",
    re.IGNORECASE,
)


def load_forecast_meta(
    path: Path,
    *,
    colmap: Optional[Dict[str, str]] = None,
) -> ForecastMeta:
    

    p = Path(path)

    head = pd.read_csv(p, nrows=0)
    cols = [str(c) for c in head.columns]
    cols_t = tuple(cols)

    cm = dict(colmap or {})

    time_col = cm.get("time") or _pick(cols, _T_CAND)
    step_col = cm.get("step") or _pick(cols, _STEP_CAND)
    samp_col = cm.get("sample") or _pick(cols, ["sample_idx"])
    unit_col = cm.get("unit") or _pick(cols, _UNIT_CAND)

        
    x_col = cm.get("x") or _pick(cols, _X_CAND)
    y_col = cm.get("y") or _pick(cols, _Y_CAND)

    actual = cm.get("actual") or _pick(cols, ["subsidence_actual"])
    pred = cm.get("pred") or _pick(cols, ["subsidence_pred"])

    qgroups = find_quantile_groups(cols)

    use = [c for c in [time_col, step_col, samp_col] if c]
    use += [c for c in [unit_col, x_col, y_col] if c]
    use += [c for c in [actual, pred] if c]
    use += [c for c in _flatten_qgroups(qgroups) if c]

    df = pd.read_csv(
        p,
        usecols=list(dict.fromkeys(use)),
    )

    n_rows = int(len(df))
    n_points = _count_points(df, samp_col)

    year_min, year_max, years = _range_vals(df, time_col)
    step_min, step_max, _ = _range_vals(df, step_col)

    unit = None
    if unit_col and unit_col in df.columns:
        s = df[unit_col].dropna()
        unit = str(s.iloc[0]) if len(s) else None
        
    if not unit:
        unit = _infer_unit_from_eval_json(p) 
        
    items, qlab = build_value_items(
        cols=cols,
        actual=actual,
        pred=pred,
        qgroups=qgroups,
    )

    return ForecastMeta(
        cols=cols_t,
        n_rows=n_rows,
        n_points=n_points,
        time_col=time_col,
        step_col=step_col,
        sample_col=samp_col,
        unit_col=unit_col,
        x_col=x_col,
        y_col=y_col,
        year_min=year_min,
        year_max=year_max,
        step_min=step_min,
        step_max=step_max,
        time_values=years,
        value_items=items,
        quantile_label=qlab,
        unit=unit,
    )

def find_quantile_groups(
    cols: List[str],
) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}

    for c in cols:
        m = _Q_PAT.match(c)
        if not m:
            continue

        base = (m.group("base") or "").strip()
        q = int(m.group("q"))
        tag = f"q{q:02d}"

        out.setdefault(base, {})
        out[base][tag] = c

    return out


def _flatten_qgroups(
    qgroups: Dict[str, Dict[str, str]],
) -> List[str]:
    out: List[str] = []
    for base in sorted(qgroups.keys()):
        g = qgroups[base]
        for k in sorted(g.keys()):
            out.append(g[k])
    return out


def build_value_items(
    *,
    cols: List[str],
    actual: Optional[str],
    pred: Optional[str],
    qgroups: Dict[str, Dict[str, str]],
) -> Tuple[Tuple[Tuple[str, str], ...], str]:
    items: List[Tuple[str, str]] = []

    if actual:
        items.append(("Actual", actual))
    if pred:
        items.append(("Pred", pred))

    bases = list(qgroups.keys())

    # Prefer "subsidence" if present.
    bases_sorted = sorted(
        bases,
        key=lambda b: (b.lower() != "subsidence", b),
    )

    for base in bases_sorted:
        g = qgroups[base]
        for qtag in sorted(g.keys()):
            label = f"{base} · {qtag.upper()}"
            items.append((label, g[qtag]))

    qlab = ", ".join(sorted(bases_sorted))
    qlab = qlab or "-"

    return tuple(items), qlab

def _pick(cols: List[str], cand: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for k in cand:
        if k.lower() in low:
            return low[k.lower()]
    return None


def _find_quantiles(
    cols: List[str],
    *,
    base: str,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in cols:
        m = _Q_PAT.match(c)
        if not m:
            continue
        b = (m.group("base") or "").lower()
        if b != base.lower():
            continue
        q = int(m.group("q"))
        out[f"q{q:02d}"] = c
    return out


def _count_points(df, samp_col: Optional[str]) -> int:
    if samp_col and samp_col in df.columns:
        return int(df[samp_col].nunique())
    return int(len(df))


def _range_vals(df, col: Optional[str]):
    if not col or col not in df.columns:
        return None, None, tuple()
    s = df[col].dropna()
    if not len(s):
        return None, None, tuple()
    vals = sorted({int(v) for v in s.tolist()})
    return vals[0], vals[-1], tuple(vals)


def _value_items(
    actual: Optional[str],
    pred: Optional[str],
    qcols: Dict[str, str],
) -> Tuple[Tuple[str, str], ...]:
    out: List[Tuple[str, str]] = []

    if actual:
        out.append(("Actual", actual))
    if pred:
        out.append(("Pred", pred))

    for k in sorted(qcols.keys()):
        out.append((k.upper(), qcols[k]))

    return tuple(out)


def _infer_unit_from_eval_json(p: Path) -> Optional[str]:
    

    pats = (
        "*geoprior_eval*_interpretable.json",
        "*geoprior_eval*.json",
    )

    for d in (p.parent, p.parent.parent):
        if d is None or not d.exists():
            continue

        for pat in pats:
            for jf in d.glob(pat):
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        obj = json.load(f)

                    units = obj.get("units") or {}
                    u = units.get("subs_metrics_unit") or ""
                    u = str(u).strip()
                    if u:
                        return u
                except Exception:
                    continue

    return None
