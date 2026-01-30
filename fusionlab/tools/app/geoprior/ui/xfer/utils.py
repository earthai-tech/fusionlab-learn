# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.utils

Xfer results discovery helpers for the Xfer Map mode.

We support two layouts:

(v3.2)
<results_root>/
  xfer/
    <citya>__<cityb>/
      <jobid>/
        <src>_to_<tgt>_<strategy>_<split>_<calib>_
        <rescale>_eval.csv
        <src>_to_<tgt>_<strategy>_<split>_<calib>_
        <rescale>_future.csv
        xfer_results.csv
        xfer_results.json

(v3.0 fallback)
Delegates to geoprior.ui.map.utils.scan_results_root.

We never rename files on disk. We only build short UI labels.

Line length target: <= 62 chars (black config).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..map.utils import MapCity, MapFile, MapJob
from ..map.utils import scan_results_root as scan_v30_root


__all__ = [
    "XferPair",
    "XferCsvMeta",
    "encode_job_id",
    "decode_job_id",
    "is_v32_root",
    "parse_pair_dir",
    "parse_xfer_csv_name",
    "scan_xfer_results_root",
    "scan_xfer_v32",
    "select_xfer_csv",
]


_XFER_DIR = "xfer"

_PAIR_PAT = re.compile(
    r"^(?P<a>.+?)__(?P<b>.+?)$"
)

# <src>_to_<tgt>_<strategy>_<split>_<calib>_<rescale>_
# (eval|future).csv
_XFER_CSV_PAT = re.compile(
    r"^(?P<src>.+?)_to_(?P<tgt>.+?)_"
    r"(?P<strategy>[^_]+)_"
    r"(?P<split>[^_]+)_"
    r"(?P<calib>[^_]+)_"
    r"(?P<rescale>.+?)_"
    r"(?P<kind>eval|future)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class XferPair:
    """
    Pair directory name split into (a,b).
    """
    a: str
    b: str
    name: str
    path: Path


@dataclass(frozen=True)
class XferCsvMeta:
    """
    Parsed info from a v3.2 xfer CSV filename.
    """
    src_city: str
    tgt_city: str
    strategy: str
    split: str
    calibration: str
    rescale_mode: str
    kind: str  # "eval" | "future"


def encode_job_id(pair: str, job_id: str) -> str:
    """
    Encode a v3.2 job id with its pair to make it unique.

    Stored in K_MAP_*_JOB_ID keys, later decoded.
    """
    p = str(pair or "").strip()
    j = str(job_id or "").strip()
    if not p:
        return j
    if not j:
        return p
    return f"{p}/{j}"


def decode_job_id(job_id: str) -> Tuple[str, str]:
    """
    Reverse of encode_job_id().

    Returns (pair, job).
    """
    s = str(job_id or "").strip()
    if "/" not in s:
        return "", s
    a, b = s.split("/", 1)
    return a.strip(), b.strip()


def parse_pair_dir(
    name: str,
    *,
    base: Optional[Path] = None,
) -> Optional[XferPair]:
    s = str(name or "").strip()
    m = _PAIR_PAT.match(s)
    if not m:
        return None

    a = (m.group("a") or "").strip()
    b = (m.group("b") or "").strip()
    if not a or not b:
        return None

    p = Path(base or ".") / s
    return XferPair(a=a, b=b, name=s, path=p)


def parse_xfer_csv_name(
    filename: str,
) -> Optional[XferCsvMeta]:
    s = str(filename or "").strip()
    m = _XFER_CSV_PAT.match(s)
    if not m:
        return None

    src = (m.group("src") or "").strip()
    tgt = (m.group("tgt") or "").strip()
    strat = (m.group("strategy") or "").strip().lower()
    split = (m.group("split") or "").strip().lower()
    calib = (m.group("calib") or "").strip().lower()
    rsc = (m.group("rescale") or "").strip().lower()
    kind = (m.group("kind") or "").strip().lower()

    if not src or not tgt or kind not in ("eval", "future"):
        return None

    return XferCsvMeta(
        src_city=src,
        tgt_city=tgt,
        strategy=strat or "xfer",
        split=split or "val",
        calibration=calib or "source",
        rescale_mode=rsc or "strict",
        kind=kind,
    )


def is_v32_root(results_root: Path) -> bool:
    """
    True if results_root looks like v3.2 xfer layout.
    """
    root = Path(results_root).expanduser()
    xroot = root / _XFER_DIR
    if not xroot.exists() or not xroot.is_dir():
        return False

    for p in xroot.iterdir():
        if not p.is_dir():
            continue
        if parse_pair_dir(p.name) is None:
            continue
        for j in p.iterdir():
            if not j.is_dir():
                continue
            # at least one directional csv
            if any(_XFER_CSV_PAT.match(f.name or "")
                   for f in j.glob("*.csv")):
                return True

    return False


def scan_xfer_results_root(
    results_root: Path,
) -> List[MapCity]:
    """
    Scan results_root for xfer map candidates.

    - Try v3.2 xfer layout first.
    - If nothing found, fallback to v3.0 scanner.

    Returns MapCity list compatible with the current
    XferMapController expectations.
    """
    root = Path(results_root).expanduser()

    v32 = scan_xfer_v32(root)
    if v32:
        return v32

    return scan_v30_root(root)


def scan_xfer_v32(results_root: Path) -> List[MapCity]:
    """
    Build a "city-centric" index from v3.2 xfer folders.

    Each MapCity represents a target city.
    Each MapJob represents a (pair/jobid) folder.
    Each MapFile is a directional eval/future csv
    whose *target city* matches the MapCity.
    """
    root = Path(results_root).expanduser()
    xroot = root / _XFER_DIR
    if not xroot.exists() or not xroot.is_dir():
        return []
    # city -> enc_job -> (job_dir, files)
    city_jobs: Dict[str, Dict[str, Tuple[
        Path,
        List[MapFile],
    ]]]
    city_jobs = {}

    for pair_dir in sorted(xroot.iterdir()):
        if not pair_dir.is_dir():
            continue
        pair = parse_pair_dir(pair_dir.name, base=xroot)
        if pair is None:
            continue

        for job_dir in sorted(
            pair_dir.iterdir(),
            reverse=True,
        ):
            if not job_dir.is_dir():
                continue

            enc = encode_job_id(pair.name, job_dir.name)

            items = _scan_job_files(job_dir)
            if not items:
                continue

            for meta, mf in items:
                tgt = str(meta.tgt_city)
                city_jobs.setdefault(tgt, {})
                if enc not in city_jobs[tgt]:
                    city_jobs[tgt][enc] = (job_dir, [])
                city_jobs[tgt][enc][1].append(mf)

    out: List[MapCity] = []
    for city in sorted(city_jobs.keys()):
        jobs = _build_city_jobs(city_jobs[city])
        if not jobs:
            continue
        out.append(
            MapCity(
                city=city,
                model="xfer",
                stage="xfer",
                root=xroot,
                jobs=tuple(jobs),
            )
        )

    return out


def _build_city_jobs(
    enc_map: Dict[str, Tuple[Path, List[MapFile]]],
) -> List[MapJob]:
    out: List[MapJob] = []

    for enc in sorted(enc_map.keys(), reverse=True):
        job_dir, files = enc_map[enc]
        if not files:
            continue

        # Stable ordering: val -> future
        files = sorted(files, key=lambda f: f.kind)

        out.append(
            MapJob(
                kind="xfer",
                job_id=str(enc),
                root=Path(job_dir),
                files=tuple(files),
            )
        )

    return out


def _scan_job_files(
    job_dir: Path,
) -> List[Tuple[XferCsvMeta, MapFile]]:
    """
    Return (meta, MapFile) for directional eval/future csvs.

    We intentionally ignore xfer_results.csv/json as they
    are not forecast maps.
    """
    out: List[Tuple[XferCsvMeta, MapFile]] = []

    csvs = sorted(job_dir.glob("*.csv"))
    if not csvs:
        return out

    # group for duplicate labels
    seen: Dict[str, int] = {}

    for p in csvs:
        if p.name.lower() == "xfer_results.csv":
            continue

        meta = parse_xfer_csv_name(p.name)
        if meta is None:
            continue

        kind = "val" if meta.kind == "eval" else "future"
        key = _label_key(meta, kind)
        seen[key] = int(seen.get(key, 0)) + 1
        idx = int(seen[key])

        disp = _short_xfer_label(
            meta,
            kind=kind,
            idx=idx,
        )
        out.append(
            (
                meta,
                MapFile(
                    path=Path(p),
                    display=disp,
                    kind=kind,
                ),
            )
        )

    return out


def _label_key(meta: XferCsvMeta, kind: str) -> str:
    return (
        f"{kind}|{meta.strategy}|{meta.split}|"
        f"{meta.calibration}|{meta.rescale_mode}|"
        f"{meta.src_city}|{meta.tgt_city}"
    )


def _short_xfer_label(
    meta: XferCsvMeta,
    *,
    kind: str,
    idx: int,
) -> str:
    """
    Short UI label for a directional xfer CSV.
    spec = (
        f"{meta.split}/{meta.calibration}/"
        f"{meta.rescale_mode}"
    )
    Example:
    Val · xfer · val/source/strict · A→B
    """
    base = "Val" if kind == "val" else "Future"
    if int(idx) > 1:
        base = f"{base}{int(idx)}"

    spec = (
        f"{meta.split}/{meta.calibration}/"
        f"{meta.rescale_mode}"
    )
    dire = f"{meta.src_city}→{meta.tgt_city}"

    tag = str(meta.strategy or "xfer").lower()
    return f"{base} · {tag} · {spec} · {dire}"


def select_xfer_csv(
    results_root: Path,
    *,
    city_a: str,
    city_b: str,
    target_city: str,
    kind: str = "eval",
    strategy: Optional[str] = None,
    split: Optional[str] = None,
    calibration: Optional[str] = None,
    rescale_mode: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Optional[Path]:
    """
    Pick a v3.2 xfer CSV by metadata (best-effort).

    Notes
    -----
    - target_city decides the direction:
        target=A => src=B
        target=B => src=A
    - If job_id is None, we pick the newest folder.
    - Filters are gradually relaxed if no hit.

    Returns
    -------
    Path or None
    """
    root = Path(results_root).expanduser()
    xroot = root / _XFER_DIR
    if not xroot.exists() or not xroot.is_dir():
        return None

    a = str(city_a or "").strip()
    b = str(city_b or "").strip()
    t = str(target_city or "").strip()
    if not a or not b or not t:
        return None

    if t == a:
        src, tgt = b, a
    elif t == b:
        src, tgt = a, b
    else:
        # explicit target not part of pair
        src, tgt = a, t

    # Locate the pair dir (either order).
    pair = _pick_pair_dir(xroot, a, b)
    if pair is None:
        return None

    # Optional: narrow to a specific job folder.
    jobs = _iter_job_dirs(pair, job_id=job_id)
    if not jobs:
        return None

    # Progressive relaxation.
    wants: List[Dict[str, Optional[str]]] = [
        dict(
            strategy=strategy,
            split=split,
            calibration=calibration,
            rescale_mode=rescale_mode,
        ),
        dict(
            strategy=strategy,
            split=split,
            calibration=calibration,
            rescale_mode=None,
        ),
        dict(
            strategy=strategy,
            split=split,
            calibration=None,
            rescale_mode=None,
        ),
        dict(
            strategy=strategy,
            split=None,
            calibration=None,
            rescale_mode=None,
        ),
        dict(
            strategy=None,
            split=None,
            calibration=None,
            rescale_mode=None,
        ),
    ]

    k = str(kind or "eval").strip().lower()
    if k not in ("eval", "future"):
        k = "eval"

    for filt in wants:
        hit = _find_csv_in_jobs(
            jobs,
            src=src,
            tgt=tgt,
            kind=k,
            strategy=filt.get("strategy"),
            split=filt.get("split"),
            calibration=filt.get("calibration"),
            rescale_mode=filt.get("rescale_mode"),
        )
        if hit is not None:
            return hit

    return None


def _pick_pair_dir(
    xroot: Path,
    a: str,
    b: str,
) -> Optional[Path]:
    cand = [f"{a}__{b}", f"{b}__{a}"]
    for nm in cand:
        p = xroot / nm
        if p.exists() and p.is_dir():
            return p
    return None


def _iter_job_dirs(
    pair_dir: Path,
    *,
    job_id: Optional[str],
) -> List[Path]:
    if job_id:
        _, jid = decode_job_id(str(job_id))
        p = pair_dir / str(jid)
        if p.exists() and p.is_dir():
            return [p]
        return []

    # newest first (lexicographic works for timestamps)
    return [
        p
        for p in sorted(
            pair_dir.iterdir(),
            reverse=True,
        )
        if p.is_dir()
    ]



def _find_csv_in_jobs(
    jobs: Iterable[Path],
    *,
    src: str,
    tgt: str,
    kind: str,
    strategy: Optional[str],
    split: Optional[str],
    calibration: Optional[str],
    rescale_mode: Optional[str],
) -> Optional[Path]:
    src = str(src).strip().lower()
    tgt = str(tgt).strip().lower()

    st = _norm(strategy)
    sp = _norm(split)
    ca = _norm(calibration)
    rs = _norm(rescale_mode)

    for jd in jobs:
        for p in sorted(jd.glob("*.csv")):
            meta = parse_xfer_csv_name(p.name)
            if meta is None:
                continue

            if meta.kind != kind:
                continue

            if meta.src_city.strip().lower() != src:
                continue
            if meta.tgt_city.strip().lower() != tgt:
                continue

            if st and meta.strategy != st:
                continue
            if sp and meta.split != sp:
                continue
            if ca and meta.calibration != ca:
                continue
            if rs and meta.rescale_mode != rs:
                continue

            return Path(p)

    return None


def _norm(v: Optional[str]) -> str:
    return str(v or "").strip().lower()
