# geoprior/ui/xfer/insights.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import MapPoint


@dataclass(frozen=True)
class Badge:
    name: str
    color: str
    shape: str
    pulse: bool
    tip: str
    mae: float


def _safe_f(x: Any, d: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(d)


def _latest_xfer_dir(
    root: Path,
    a: str,
    b: str,
) -> Optional[Path]:
    base = root / "xfer" / f"{a}_to_{b}"
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.name, reverse=True)
    return dirs[0]


def _load_records(p: Path) -> List[Dict[str, Any]]:
    try:
        txt = p.read_text(encoding="utf-8")
        obj = json.loads(txt)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass
    return []


def _rank_score(r: Dict[str, Any]) -> float:
    mae = _safe_f(r.get("overall_mae", 1e9), 1e9)
    cov = _safe_f(r.get("coverage80", 0.0), 0.0)
    shp = _safe_f(r.get("sharpness80", 0.0), 0.0)
    cov_pen = abs(cov - 0.80)
    return mae + (60.0 * cov_pen) + (0.05 * shp)

def _pick_best(
    recs: List[Dict[str, Any]],
    *,
    split: str,
    direction: str,
) -> Optional[Dict[str, Any]]:
    dd = [
        r for r in recs
        if str(r.get("split", "")) == str(split)
        and str(r.get("direction", "")) == str(direction)
    ]
    if not dd:
        return None
    return min(dd, key=_rank_score)


def _badge_from(
    r: Dict[str, Any],
    *,
    direction: str,
) -> Badge:
    mae = _safe_f(r.get("overall_mae", 1e9), 1e9)
    cov = _safe_f(r.get("coverage80", 0.0), 0.0)
    shp = _safe_f(r.get("sharpness80", 0.0), 0.0)
    r2 = _safe_f(r.get("overall_r2", 0.0), 0.0)
    cal = str(r.get("calibration", "none") or "none")

    # Simple quality heuristic (stable + readable)
    if mae <= 15.0 and cov >= 0.60:
        grade = "Good"
        color = "#2E7D32"
        shape = "diamond"
        pulse = False
    elif mae <= 30.0 and cov >= 0.45:
        grade = "Caution"
        color = "#F9A825"
        shape = "triangle"
        pulse = True
    else:
        grade = "Poor"
        color = "#C62828"
        shape = "square"
        pulse = True

    nm = "A→B" if direction == "A_to_B" else "B→A"

    tip = (
        f"{nm}  {grade}\n"
        f"calib={cal}\n"
        f"cov80={cov:.3f}  sharp80={shp:.2f}\n"
        f"MAE={mae:.2f}  R2={r2:.3f}"
    )

    return Badge(
        name=f"{nm}  {grade}",
        color=color,
        shape=shape,
        pulse=pulse,
        tip=tip,
        mae=mae,
    )


def _mid_with_offset(
    a_lat: float,
    a_lon: float,
    b_lat: float,
    b_lon: float,
    side: float,
) -> Tuple[float, float]:
    mx = (a_lon + b_lon) * 0.5
    my = (a_lat + b_lat) * 0.5

    dx = b_lon - a_lon
    dy = b_lat - a_lat
    n = math.sqrt((dx * dx) + (dy * dy))
    if n <= 1e-12:
        return my, mx

    # perpendicular unit
    ox = -dy / n
    oy = dx / n

    mag = 0.06 * max(abs(dx), abs(dy), 0.2)
    return (my + (oy * mag * side), mx + (ox * mag * side))


def build_xfer_badges(
    *,
    results_root: Path,
    city_a: str,
    city_b: str,
    split: str,
    a_lat: float,
    a_lon: float,
    b_lat: float,
    b_lon: float,
    want_ab: bool,
    want_ba: bool,
) -> List[Tuple[str, str, List[MapPoint], Dict[str, Any]]]:
    out: List[
        Tuple[str, str, List[MapPoint], Dict[str, Any]]
    ] = []

    d = _latest_xfer_dir(results_root, city_a, city_b)
    if d is None:
        return out

    jf = d / "xfer_results.json"
    recs = _load_records(jf)
    if not recs:
        return out

    if want_ab:
        r = _pick_best(recs, split=split, direction="A_to_B")
        if r is not None:
            b = _badge_from(r, direction="A_to_B")
            lat, lon = _mid_with_offset(
                a_lat, a_lon, b_lat, b_lon, +1.0
            )
            pt = MapPoint(lat=lat, lon=lon, v=b.mae, tip=b.tip)
            out.append(
                (
                    "XFER_AB",
                    b.name,
                    [pt],
                    {
                        "stroke": b.color,
                        "fillMode": "fixed",
                        "fillColor": b.color,
                        "shape": b.shape,
                        "radius": 10,
                        "opacity": 0.95,
                        "enableTooltip": True,
                        "pulse": b.pulse,
                    },
                )
            )

    if want_ba:
        r = _pick_best(recs, split=split, direction="B_to_A")
        if r is not None:
            b = _badge_from(r, direction="B_to_A")
            lat, lon = _mid_with_offset(
                a_lat, a_lon, b_lat, b_lon, -1.0
            )
            pt = MapPoint(lat=lat, lon=lon, v=b.mae, tip=b.tip)
            out.append(
                (
                    "XFER_BA",
                    b.name,
                    [pt],
                    {
                        "stroke": b.color,
                        "fillMode": "fixed",
                        "fillColor": b.color,
                        "shape": b.shape,
                        "radius": 10,
                        "opacity": 0.95,
                        "enableTooltip": True,
                        "pulse": b.pulse,
                    },
                )
            )

    return out
